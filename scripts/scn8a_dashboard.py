"""
SCN8A-DEE (EIEE13) Dashboard
==============================
41-patient cohort · SCN8A (12q13.13) · Nav1.6 Voltage-Gated Sodium Channel α Subunit
SCN8A encephalopathy: pathogenic de novo variant in SCN8A (sodium voltage-gated channel
alpha subunit 8, 12q13.13) causing early-onset developmental and epileptic encephalopathy
(DEE). SCN8A (Nav1.6) is the predominant sodium channel at nodes of Ranvier and the axon
initial segment (AIS) of mature excitatory and inhibitory neurons — critical for repetitive
high-frequency firing; unlike SCN2A (Nav1.2, neonatal) Nav1.6 peaks postnatally and
remains the dominant channel in adult cortex.

PATHOMECHANISM: ~95% of SCN8A-DEE variants cause gain-of-function (GOF) — enhanced
persistent Na+ current (INaP), impaired fast inactivation, left-shifted activation, or
window current expansion — driving neuronal hyperexcitability, status epilepticus, and
catastrophic early-onset DEE. The GOF mechanism makes Na-channel blockers the PREFERRED
treatment class (contrast with SCN2A LOF where they are contraindicated).

GOF (Gain-of-Function) → EIEE13 / SCN8A-DEE: persistent Na+ channel opening, impaired
inactivation, enhanced window current → high-frequency burst discharges → neonatal or
early-infantile seizures, movement disorder, global cognitive impairment.
TREAT WITH Na-channel blockers: PHT IV bridge → CBZ/OXC oral maintenance.

KEY DISTINCTION FROM SCN2A: SCN8A is almost exclusively GOF; the LOF phenotype
(~5% of cases) causes a milder self-limited epilepsy or intellectual disability
WITHOUT DEE. Therefore, the GOF/LOF treatment pivot present in SCN2A is not the
primary clinical dilemma in SCN8A — instead the challenge is drug resistance (>90%
DRE) and the co-occurring movement disorder (dystonia, hyperkinetic movements).

MOVEMENT DISORDER: 30–50% of SCN8A-DEE patients develop a prominent movement disorder
(dystonia, choreiform, hyperkinetic) that is Nav1.6-mediated and may paradoxically
IMPROVE with CBZ/OXC. This is a unique clinical signature distinguishing SCN8A-DEE
from other sodium-channel DEEs.

QUINIDINE: Quinidine (a Na-channel blocker with additional Kv11.1 activity) has been
reported to reduce seizures in SCN8A GOF via selective blockade of the window current
at therapeutic plasma levels (2–5 mg/L). Evidence is anecdotal/single-arm Phase 2;
QTc monitoring is mandatory.

AED NOTE: PHT (IV) first-line acute/NICU → CBZ or OXC oral. LTG is ABSOLUTELY
CONTRAINDICATED in SCN8A GOF (Na-channel activator — induces status epilepticus).
VGB has limited efficacy. HLA-B*1502 (Asian ancestry) CPIC Level A mandatory before
CBZ/OXC. Na/SIADH monitoring with OXC (>CBZ risk).
DISEASE-MODIFYING: Antisense oligonucleotide (ASO) and AAV-mediated gene therapy
programmes are in preclinical / early Phase 1 development.
"""

import random
from datetime import datetime

SEED = 9177  # dashboard 177
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "De novo SCN8A GOF missense (severe DEE cluster)",
        "n": 24, "pct": 58,
        "category": "De-novo-SCN8A-GOF-severe-DEE",
        "mechanism": (
            "Most prevalent SCN8A-DEE variant class (~58%): de novo missense variants causing "
            "pronounced gain-of-function of Nav1.6 — enhanced persistent sodium current (INaP), "
            "impaired fast inactivation, or left-shifted voltage activation. Nav1.6 is the "
            "dominant sodium channel at the axon initial segment (AIS) and nodes of Ranvier of "
            "mature cortical, subcortical, and cerebellar neurons. GOF leads to pathological "
            "high-frequency bursting, cortical hyperexcitability, and catastrophic early-onset "
            "DEE (onset day 1 to 18 months). Associated with severe intellectual disability, "
            "drug-resistant epilepsy (>90%), and prominent movement disorder (30–50%). "
            "Common variant hotspots: p.Arg1872Gln, p.Asn1768Asp (most severe), p.Val1763Leu."
        ),
        "eeg_signature": (
            "Burst-suppression (neonatal, synchronous — distinct from asynchronous STXBP1 BS), "
            "transitioning to multifocal independent spike-wave complexes; interictal "
            "high-amplitude polyspike-wave; ictal pattern: low-voltage fast activity (LVFA) "
            "or recruiting rhythmic fast discharge at seizure onset — identical GOF ictal "
            "signature seen in SCN2A-GOF and KCNQ2-GOF."
        ),
        "mri": "Often normal early; progressive volume loss / cortical atrophy in severe cases.",
        "clinical_note": (
            "GOF severity correlates with variant position in Nav1.6 DI-IV transmembrane "
            "segments and DIII-DIV linker (inactivation gate). p.Asn1768Asp is the 'ultra-severe' "
            "variant — refractory neonatal SE, survival to childhood rare without aggressive "
            "management. Na-channel blockers (CBZ/OXC/PHT) are FIRST-LINE — early initiation "
            "within 24–72h of seizure onset is associated with improved seizure control."
        ),
    },
    {
        "etiology": "De novo SCN8A GOF missense (moderate / self-limited-overlap)",
        "n": 7, "pct": 17,
        "category": "De-novo-SCN8A-GOF-moderate",
        "mechanism": (
            "Moderate GOF variants (~17%): partial gain-of-function with less pronounced INaP "
            "enhancement. Seizure onset 1–18 months; better initial seizure control with "
            "CBZ/OXC; subset achieves seizure freedom. Movement disorder less prominent. "
            "Cognitive outcomes more variable (moderate–severe ID rather than profound). "
            "Overlapping phenotype with BFNIS-like (benign neonatal-infantile) at one end."
        ),
        "eeg_signature": "Focal or multifocal spike-wave; LVFA ictal onset; less severe interictal burden.",
        "mri": "Usually normal; occasional T2 signal changes in basal ganglia (movement disorder cases).",
        "clinical_note": (
            "Respond better to CBZ/OXC than severe GOF cases; seizure freedom achievable in "
            "~20–30%. Movement disorder (if present) often improves with Na-channel blocker "
            "titration — a therapeutic bonus unique to SCN8A compared with other DEEs."
        ),
    },
    {
        "etiology": "De novo SCN8A truncating / LoF (ultra-rare, distinct mild phenotype)",
        "n": 4, "pct": 10,
        "category": "De-novo-SCN8A-LOF",
        "mechanism": (
            "Rare LOF variants (~10% of SCN8A pathogenic de novo variants): haploinsufficiency of "
            "Nav1.6 causes a milder phenotype — intellectual disability (ID) +/- epilepsy, "
            "autism spectrum features, but NOT catastrophic neonatal DEE. The absence of "
            "persistent Na+ channel opening means this class does NOT cause the same degree of "
            "neuronal hyperexcitability as GOF. Critically, Na-channel blockers are NOT the "
            "first-line and may not benefit LOF — however, unlike SCN2A LOF, SCN8A LOF does "
            "not show the same acute catastrophic worsening with Na-channel blockers. Treat "
            "seizures (if present) empirically with broad-spectrum AEDs (LEV, VPA)."
        ),
        "eeg_signature": "Non-specific focal or generalised spike-wave; no burst-suppression.",
        "mri": "Usually normal.",
        "clinical_note": (
            "These patients should NOT be lumped with GOF-DEE for treatment purposes. Functional "
            "class determination (patch-clamp, computational modelling, ClinGen evidence) is "
            "mandatory before initiating Na-channel blocker therapy."
        ),
    },
    {
        "etiology": "De novo SCN8A splice site / structural variant",
        "n": 3, "pct": 7,
        "category": "De-novo-SCN8A-splice-structural",
        "mechanism": (
            "Splice site variants and small structural (copy number) variants (~7%): may cause "
            "GOF (exon skipping removing regulatory exons) or LOF (haploinsufficiency). Functional "
            "consequence must be determined by RNA studies or ClinGen Expert Panel review before "
            "treatment. Phenotype intermediate between severe GOF-DEE and mild LOF-ID."
        ),
        "eeg_signature": "Variable — depends on functional class.",
        "mri": "Variable.",
        "clinical_note": (
            "Splice variants require functional RNA analysis; structural variants (deletions/duplications) "
            "need ACMG-AMP variant classification. Pending functional class, initiate broad-spectrum AED "
            "(LEV) and await expert panel interpretation."
        ),
    },
    {
        "etiology": "Clinical SCN8A-negative (phenocopy / unexplained DEE)",
        "n": 3, "pct": 8,
        "category": "Clinical-SCN8A-negative",
        "mechanism": (
            "Patients with clinical features highly consistent with SCN8A-DEE (early-onset focal "
            "DEE + movement disorder + response to Na-channel blockers) but no pathogenic SCN8A "
            "variant identified on standard sequencing. May represent: deep intronic variants not "
            "captured by exome, somatic mosaicism below detection threshold, digenic epilepsy, "
            "or phenotypic overlap with other Nav-channel DEEs (SCN2A, SCN3A, SCN9A). "
            "Recommend: RNA studies, low-level mosaic testing, Nav-channel gene panel re-analysis."
        ),
        "eeg_signature": "SCN8A-like: LVFA ictal onset, multifocal spike-wave.",
        "mri": "Usually normal.",
        "clinical_note": (
            "Empiric Na-channel blocker trial may be warranted given clinical phenotype; "
            "re-analyse with trio exome + RNA studies."
        ),
    },
]

# ── Seizure Types (4 primary, N=41 cohort) ──────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal tonic / clonic (neonatal-infantile onset)",
        "prevalence_pct": 92,
        "onset_age": "Day 1 – 18 months (median 4 months)",
        "eeg_correlate": (
            "Focal ictal onset: low-voltage fast activity (LVFA) or recruiting rhythmic discharge "
            "(alpha/beta frequency) over frontal or temporal leads; rapid secondary generalisation "
            "in severe GOF. Interictal: multifocal independent spike-wave complexes."
        ),
        "clinical_tip": (
            "The combination of focal tonic seizures + early onset + LVFA ictal EEG pattern + "
            "movement disorder is the clinical 'fingerprint' of SCN8A-DEE. Confirm GOF status → "
            "initiate CBZ or OXC within 24–72h. If IV access needed: PHT IV bridge."
        ),
    },
    {
        "type": "Focal-to-bilateral tonic-clonic (FBTCS)",
        "prevalence_pct": 76,
        "onset_age": "Infancy to childhood",
        "eeg_correlate": (
            "Focal onset LVFA → rapid bilateral spread → generalised tonic-clonic discharge. "
            "Post-ictal suppression. High SUDEP risk with nocturnal FBTCS."
        ),
        "clinical_tip": (
            "Nocturnal FBTCS are the primary SUDEP risk factor in SCN8A-DEE. Ensure nocturnal "
            "supervision, consider bedside O2 saturation monitoring. Maximise CBZ/OXC dose "
            "to TDM target before declaring drug resistance."
        ),
    },
    {
        "type": "Absence-like (brief atypical; non-convulsive SE)",
        "prevalence_pct": 38,
        "onset_age": "Childhood (2–8 years)",
        "eeg_correlate": (
            "Brief generalised spike-wave at 2–3.5 Hz (slower than typical absence 3 Hz); "
            "often with irregular morphology; EEG background abnormal. Non-convulsive status "
            "epilepticus (NCSE) may present as prolonged unresponsive episodes."
        ),
        "clinical_tip": (
            "Do NOT use ethosuximide (ESM) for SCN8A absence-like — ESM blocks T-type Ca2+ but "
            "has no effect on Nav1.6. CBZ/OXC remain first-line. Consider CLB adjunct if "
            "absence-like seizures persist on Na-channel blocker."
        ),
    },
    {
        "type": "Epileptic spasms / IS (infantile spasms)",
        "prevalence_pct": 24,
        "onset_age": "3–18 months (overlap with West syndrome phenotype)",
        "eeg_correlate": (
            "Modified hypsarrhythmia or high-amplitude chaotic polyspike-wave; individual spasms "
            "correlate with EMG burst + EEG attenuation (electrodecrement). May coexist with "
            "multifocal spike-wave."
        ),
        "clinical_tip": (
            "SCN8A spasms are distinct from STXBP1/CDKL5 spasms — they do NOT respond well to "
            "ACTH or VGB. Prioritise Na-channel blocker optimisation (CBZ/OXC). If IS persists "
            "after 2 AED failures, consider ACTH trial but anticipate partial response only. "
            "KD (ketogenic diet) may suppress spasms in drug-resistant cases."
        ),
    },
]

# ── Triggers (8 primary) ──────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / hyperthermia", "rate_pct": 85,
     "note": "Most common SCN8A trigger. Nav1.6 GOF channels open more readily at elevated temperature (Q10 effect). Aggressive fever management (paracetamol + ibuprofen + tepid sponging). Rescue AED (diazepam rectal / MDZ buccal) at fever onset per family seizure action plan."},
    {"trigger": "Intercurrent illness (non-febrile)", "rate_pct": 73,
     "note": "Metabolic stress, dehydration, electrolyte shifts lower seizure threshold. Ensure AED continuity during illness. IV access plan for hospital admissions."},
    {"trigger": "Missed / delayed AED dose", "rate_pct": 68,
     "note": "CBZ/OXC half-life is 12–24h; missed dose causes trough-level drop → breakthrough seizure. Extended-release formulations (CBZ-XR, OXC-XR) reduce trough fluctuations. Pill organiser + alarm reminders mandatory."},
    {"trigger": "Sleep deprivation", "rate_pct": 56,
     "note": "Reduced sleep lowers cortical inhibition. Sleep hygiene protocol essential. Melatonin 0.5–3 mg can improve sleep architecture in SCN8A (no AED interaction)."},
    {"trigger": "Emotional / physical stress", "rate_pct": 43,
     "note": "HPA axis activation → neuronal excitability. Caregiver counselling on stress identification. Benzodiazepine rescue plan for prolonged stress-triggered clusters."},
    {"trigger": "Abrupt AED withdrawal / rapid taper", "rate_pct": 38,
     "note": "Rebound hyperexcitability if CBZ/OXC tapered too quickly. Minimum taper rate: 10% dose reduction per 2 weeks. NEVER abrupt discontinuation. Escalating-dose rescue protocol for planned taper."},
    {"trigger": "Na-channel blocker cessation (rebound SE risk)", "rate_pct": 29,
     "note": "Abrupt CBZ/OXC withdrawal in GOF SCN8A can precipitate refractory SE / SUDEP. Hospital admission recommended for any planned CBZ/OXC discontinuation. IV PHT ready."},
    {"trigger": "Puberty / hormonal shifts", "rate_pct": 18,
     "note": "Oestrogen lowers, progesterone raises seizure threshold. Adolescent girls may need AED dose review at puberty onset. Catamenial exacerbation documented in ~15% of female SCN8A-DEE."},
]

# ── Treatments (8 lines) ──────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Phenytoin IV (PHT / Fosphenytoin)",
        "evidence": "Level B",
        "indication": "GOF SCN8A-DEE — acute seizure control / NICU bridge",
        "dose": (
            "Loading dose: 15–20 mg/kg PE IV at ≤1 mg PE/kg/min (max 50 mg/min adult). "
            "Maintenance: 4–8 mg/kg/day PE IV ÷ 2–3 doses. "
            "Transition to oral CBZ/OXC as soon as tolerated (typically 48–72h)."
        ),
        "moa": (
            "Na-channel blocker — binds fast-inactivated state of Nav1.6, stabilises "
            "inactivated conformation, reduces persistent Na+ current (INaP). "
            "IV administration provides rapid therapeutic levels (therapeutic range 10–20 µg/mL)."
        ),
        "efficacy": "Rapid seizure cessation in GOF neonatal/infantile SE; bridge to oral maintenance.",
        "safety": (
            "Cardiac monitoring mandatory (QTc prolongation, AV block at rapid infusion rates). "
            "Purple glove syndrome (extravasation) — use central IV if possible. "
            "Enzyme inducer: ↓ levels of other AEDs, warfarin, oral contraceptives."
        ),
        "monitoring": "PHT free-level target 1–2 µg/mL (or total 10–20 µg/mL); ECG; liver function.",
        "contraindications": "Hypersensitivity; 2nd/3rd degree AV block; bradycardia.",
    },
    {
        "drug": "Carbamazepine (CBZ)",
        "evidence": "Level B",
        "indication": "GOF SCN8A-DEE — oral first-line maintenance",
        "dose": (
            "Start: 5 mg/kg/day PO ÷ 2 doses; titrate over 2–4 weeks to 10–25 mg/kg/day "
            "(neonates/infants) or 10–30 mg/kg/day (children/adults). "
            "Extended-release (CBZ-XR): preferred for twice-daily dosing. "
            "TDM target: 4–12 µg/mL (free level 1–3 µg/mL)."
        ),
        "moa": (
            "Na-channel blocker — preferential binding to fast-inactivated Nav1.6, "
            "reduces window current and persistent INaP; frequency-dependent block "
            "reduces repetitive high-frequency firing. "
            "Also active at Nav1.2 (SCN2A-GOF cross-effect)."
        ),
        "efficacy": "Seizure freedom in ~15–25% SCN8A-GOF; ≥50% reduction in ~45% at optimal TDM levels.",
        "safety": (
            "SIADH (hyponatraemia) — Na+ monitoring q4wk first 3 months. "
            "HLA-B*1502 (Asian ancestry): SJS/TEN risk — CPIC Level A genotyping MANDATORY before initiation. "
            "Hepatic enzyme induction: reduces levels of OXC-MHD, LEV, VPA, hormonal contraceptives. "
            "Teratogenic: CPIC / FDA category D in pregnancy (neural tube defects, cardiac defects). "
            "Aplastic anaemia / agranulocytosis (rare): CBC at baseline + 6 weeks."
        ),
        "monitoring": "CBZ TDM 4–12 µg/mL; Na+ q4wk; LFT q3M; CBC baseline+6wk; HLA-B*1502 before start.",
        "contraindications": "HLA-B*1502 positive (Asian ancestry); absence seizures as primary epilepsy; bone marrow suppression.",
    },
    {
        "drug": "Oxcarbazepine (OXC)",
        "evidence": "Level B",
        "indication": "GOF SCN8A-DEE — oral first-line (infant-preferred over CBZ)",
        "dose": (
            "Start: 5–10 mg/kg/day PO ÷ 2 doses; titrate over 2–4 weeks to 20–40 mg/kg/day "
            "(infants/children). OXC suspension 60 mg/mL available for infants. "
            "Active metabolite MHD target: 12–24 µmol/L."
        ),
        "moa": (
            "Prodrug → active metabolite monohydroxyderivative (MHD / licarbazepine). "
            "MHD blocks fast-inactivated Na-channels with higher selectivity than CBZ; "
            "less enzyme induction than CBZ (CYP3A4 inhibition minor); fewer drug interactions. "
            "Movement disorder improvement documented with OXC in SCN8A."
        ),
        "efficacy": "Comparable to CBZ for seizure reduction; better tolerated in infants. Movement disorder improves in ~35%.",
        "safety": (
            "SIADH risk HIGHER than CBZ (hyponatraemia in ~20–30% infants). "
            "Na+ monitoring q2wk first month, then q4wk. "
            "HLA-B*1502: cross-reactivity risk with CBZ — CPIC Level A applies. "
            "Less enzyme induction than CBZ (preferred when polypharmacy or hormonal contraception). "
            "Teratogenic: Category D."
        ),
        "monitoring": "MHD level 12–24 µmol/L; Na+ q2wk×1M then q4wk; HLA-B*1502; LFT q6M.",
        "contraindications": "HLA-B*1502 positive; severe SIADH history.",
    },
    {
        "drug": "Phenobarbital (PB)",
        "evidence": "Level C",
        "indication": "Neonatal/infantile adjunct — acute seizure control + GABA augmentation",
        "dose": (
            "Loading dose: 15–20 mg/kg IV slowly. "
            "Maintenance neonates: 3–5 mg/kg/day; infants/children: 3–6 mg/kg/day. "
            "TDM target: 15–40 µg/mL (typically 20–35 µg/mL for SCN8A)."
        ),
        "moa": (
            "GABA-A receptor positive allosteric modulator — prolongs Cl⁻ channel open time, "
            "increases inhibitory tone. Secondary Na-channel blockade at high doses. "
            "Broad-spectrum antiseizure but sedating; often used as acute bridge before "
            "Na-channel blocker optimisation."
        ),
        "efficacy": "Moderate efficacy as adjunct; neonatal SE control; rarely monotherapy long-term.",
        "safety": "Sedation; cognitive dulling (avoid long-term); respiratory depression at high doses. Enzyme inducer.",
        "monitoring": "PB TDM 15–40 µg/mL; sedation assessment; respiratory rate neonates.",
        "contraindications": "Severe respiratory depression; porphyria; prior PB hypersensitivity.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level C",
        "indication": "Adjunct for residual seizures; LOF SCN8A broad-spectrum option",
        "dose": (
            "20–40 mg/kg/day IV/PO ÷ 2 doses (infants); up to 60 mg/kg/day (children/adults). "
            "IV solution available for hospital use. No titration required (can start at target)."
        ),
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) binding — reduces vesicular neurotransmitter "
            "release; modulates GABA-A receptor. No Na-channel blockade — safe as adjunct in GOF "
            "where additional Na-channel blockers may not add benefit."
        ),
        "efficacy": "Modest adjunct benefit; useful bridge during CBZ/OXC titration. LOF: first-line broad-spectrum option.",
        "safety": "Behavioural side-effects (irritability, aggression) in ~20–30%. Renal dose adjustment. Generally well-tolerated.",
        "monitoring": "Behavioural assessment CBCL at 4 and 12 weeks; renal function.",
        "contraindications": "Known LEV hypersensitivity; significant psychiatric history (relative).",
    },
    {
        "drug": "Ketogenic Diet 4:1 KD",
        "evidence": "Level B",
        "indication": "SCN8A-DEE drug-resistant epilepsy (DRE) after ≥2 AED failures",
        "dose": (
            "Classic KD 4:1 ratio (fat:carbohydrate+protein) introduced over 3–5 days in hospital. "
            "BHB (beta-hydroxybutyrate) target: 2–4 mmol/L (serum). "
            "KetoCal formula for infants unable to take solid food."
        ),
        "moa": (
            "Ketone bodies (BHB, acetoacetate) reduce neuronal excitability via multiple mechanisms: "
            "opening KATP channels, enhancing mitochondrial GABA synthesis, reducing glutamate "
            "release, mild Nav1.6 inhibition. Mechanistically complementary to Na-channel blockers "
            "in SCN8A-GOF."
        ),
        "efficacy": "~45% achieve ≥50% seizure reduction in SCN8A-DEE; ~10–15% seizure freedom.",
        "safety": (
            "Hospitalisation required for initiation. Dyslipidaemia, growth restriction, renal stones, "
            "carnitine deficiency, micronutrient deficiency (selenium, zinc, vitamins). "
            "Continue CBZ/OXC alongside KD. "
            "Contraindicated: POLG mutation (mandatory exclusion before VPA; not a direct KD CI), "
            "fatty acid oxidation defects, pyruvate carboxylase deficiency."
        ),
        "monitoring": "BHB 2–4 mmol/L; fasting lipids q3M; renal USS q6M; micronutrients q6M; growth percentiles.",
        "contraindications": "Fatty acid oxidation defects; pyruvate carboxylase deficiency; porphyria.",
    },
    {
        "drug": "Quinidine (investigational — SCN8A GOF-specific)",
        "evidence": "Phase 2 / anecdotal",
        "indication": "Refractory GOF SCN8A-DEE — investigational adjunct",
        "dose": (
            "Adult/adolescent: 200–400 mg PO every 6–8h (target plasma level 2–5 mg/L). "
            "Paediatric dosing extrapolated (5–10 mg/kg/day ÷ 3 doses); no established paediatric PK. "
            "Use only under specialist supervision with cardiac monitoring."
        ),
        "moa": (
            "Class Ia antiarrhythmic — Na-channel blocker (state-dependent; also Kv11.1/hERG blocker). "
            "Proposed mechanism in SCN8A: preferential block of Nav1.6 window current at therapeutic "
            "plasma concentrations; may have selectivity advantage over CBZ at Nav1.6 persistent "
            "current vs sodium-channel fast block. Case series (Johannesen 2019) reported "
            "marked seizure reduction in 4/8 refractory SCN8A-GOF patients."
        ),
        "efficacy": "Anecdotal; case series only. Potential ≥50% reduction in highly selected refractory cases.",
        "safety": (
            "QTc prolongation (mandatory ECG monitoring at baseline and each dose change). "
            "CONTRAINDICATED if QTc >500ms. Thrombocytopenia. GI side-effects. "
            "Multiple drug interactions (CYP2D6 inhibitor). Not licensed for epilepsy — off-label only."
        ),
        "monitoring": "QTc ECG baseline + q4wk; quinidine plasma level 2–5 mg/L; CBC q4wk.",
        "contraindications": "QTc >500ms; myasthenia gravis; concurrent QT-prolonging drugs; CYP2D6 poor metaboliser without dose adjustment.",
    },
    {
        "drug": "Lamotrigine (LTG) — ABSOLUTE CONTRAINDICATION in SCN8A GOF",
        "evidence": "CONTRAINDICATED",
        "indication": "DO NOT USE in SCN8A GOF",
        "dose": "N/A — contraindicated.",
        "moa": (
            "LTG has a paradoxical activating effect on Nav1.6 GOF variants: "
            "it binds the inactivated state BUT its slower kinetics and voltage-dependence "
            "result in INCREASED window current in certain GOF channel conformations. "
            "Multiple case reports of acute status epilepticus precipitated by LTG in SCN8A-DEE. "
            "Same mechanism as SCN2A-GOF LTG worsening."
        ),
        "efficacy": "HARMFUL — worsens seizures acutely in SCN8A GOF.",
        "safety": (
            "ABSOLUTELY CONTRAINDICATED: LTG has precipitated refractory focal status epilepticus "
            "and acute seizure exacerbation in SCN8A-GOF patients. This is a class effect shared "
            "with SCN2A-GOF, SCN1A-GOF (rare), and other Nav GOF DEEs. "
            "If LTG is accidentally started and acute worsening occurs: discontinue immediately, "
            "load PHT IV, admit to PICU."
        ),
        "monitoring": "N/A.",
        "contraindications": "ALL SCN8A GOF variants — ABSOLUTE CONTRAINDICATION. Urgent discontinuation if accidentally prescribed.",
    },
]

# ── AED Monitoring Panel (8 items) ───────────────────────────────────────────
AED_MONITORING = [
    {"item": "CBZ TDM (total)", "target": "4–12 µg/mL", "frequency": "4–6 weeks post-titration, then q3M", "rationale": "Free level preferred if hypoalbuminaemia or polypharmacy; induction may lower CBZ and raise other AED levels."},
    {"item": "OXC-MHD level", "target": "12–24 µmol/L", "frequency": "4–6 weeks post-titration, then q3M", "rationale": "MHD (active metabolite) is the pharmacodynamic marker; OXC levels not clinically useful."},
    {"item": "Na⁺ (SIADH monitoring)", "target": "135–145 mmol/L", "frequency": "q2wk×1M (OXC), q4wk×3M then q3M (CBZ)", "rationale": "OXC causes SIADH (hyponatraemia) in 20–30% of infants/children. CBZ risk lower but present. Hold if Na <130 mmol/L; fluid restrict + dose reduce."},
    {"item": "HLA-B*1502 genotype", "target": "Negative (before CBZ/OXC start)", "frequency": "Once at baseline (Asian ancestry mandatory; offer all)", "rationale": "CPIC Level A: positive HLA-B*1502 → 10× SJS/TEN risk with CBZ/OXC. If positive: switch to PHT (also carries risk; prefer LEV/VPA/LCS). Non-Asian: HLA-A*3101 risk."},
    {"item": "LFT (liver function)", "target": "Within 2× ULN", "frequency": "Baseline, 6 weeks, then q6M (CBZ enzyme induction)", "rationale": "CBZ enzyme induction elevates GGT (not hepatotoxic per se) but rare idiosyncratic hepatitis. Hold if >3× ULN ALT; stop if >8× ULN."},
    {"item": "QTc ECG (quinidine)", "target": "QTc <500ms", "frequency": "Baseline + each dose change + q4wk (if quinidine used)", "rationale": "Quinidine prolongs QTc dose-dependently. Stop if QTc >500ms. Do not co-prescribe with other QT-prolonging drugs (amiodarone, azithromycin, chlorpromazine)."},
    {"item": "Neurodevelopment (Bayley-III)", "target": "Cognitive, motor, language subscales", "frequency": "q6M age 0–3Y, then q12M", "rationale": "SCN8A-DEE causes progressive cognitive impairment. Serial Bayley-III tracks trajectories; informs school placement, NDIS/disability support planning."},
    {"item": "EEG (awake + sleep 2h)", "target": "Spike-index reduction; background improvement", "frequency": "q6M or after major AED change", "rationale": "Interictal EEG spike-index correlates with seizure burden. Background deterioration (increasing diffuse slowing) may precede cognitive decline — clinical red flag for treatment escalation."},
]

# ── Lifecycle Windows (6) ─────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal NICU",
        "age_range": "Day 1 – 28d",
        "focus": "Acute seizure control; GOF functional class determination; IV AED bridge",
        "key_action": (
            "IV PHT/Fosphenytoin loading + PB adjunct. "
            "Urgent SCN8A molecular confirmation (singleton or trio exome). "
            "HLA-B*1502 genotyping. "
            "Functional class → GOF confirmed → start oral CBZ/OXC at NICU discharge."
        ),
    },
    {
        "window": "Early Infantile",
        "age_range": "1 – 6 months",
        "focus": "Oral Na-channel blocker titration; movement disorder recognition",
        "key_action": (
            "CBZ or OXC titration to TDM target. "
            "Watch for SIADH (Na+ monitoring). "
            "Developmental assessment (Bayley-III baseline). "
            "Identify movement disorder (dystonia/hyperkinetic) — may improve with CBZ/OXC. "
            "Seizure action plan to family; rescue AED supply."
        ),
    },
    {
        "window": "Late Infantile",
        "age_range": "6 – 18 months",
        "focus": "Drug resistance declared; add-on therapy; KD initiation",
        "key_action": (
            "If 2 AEDs failed → DRE confirmed → KD referral. "
            "Spasms (IS) if present: optimise Na-channel blocker (not ACTH first-line). "
            "Genetic counselling for recurrence risk. "
            "Early intervention (physio, OT, SLP, visual) — window of neuroplasticity."
        ),
    },
    {
        "window": "Early Childhood",
        "age_range": "18 months – 5 years",
        "focus": "Seizure burden; cognitive trajectory; education planning",
        "key_action": (
            "Annual EEG + MRI if cognitive regression. "
            "School entry planning (special education, 1:1 aide). "
            "Caregiver burnout screen. "
            "Review KD tolerance; consider VNS referral (3 AED failures + 2Y DRE). "
            "SUDEP risk counselling (nocturnal supervision, bedside monitoring)."
        ),
    },
    {
        "window": "School Age",
        "age_range": "5 – 12 years",
        "focus": "DRE management; movement disorder; SUDEP prevention",
        "key_action": (
            "VNS if ≥3 AED failures. "
            "Movement disorder review — botulinum toxin, baclofen, neuromotor physiotherapy. "
            "Presurgical evaluation if clear focal onset zone (MEG + SEEG + PET). "
            "SUDEP: nocturnal supervision, movement alarm mattress, night camera. "
            "Annual DEXA if KD (bone density). "
            "Adolescence planning: puberty, contraception (CBZ/OXC interactions), driving ban."
        ),
    },
    {
        "window": "Adolescence / Adult",
        "age_range": "12 years +",
        "focus": "Lifelong AED; SUDEP; hormonal interactions; independence / carer needs",
        "key_action": (
            "Transition to adult neurology. "
            "Contraception: CBZ/OXC reduce hormonal OCP efficacy — IUD or progestogen depot preferred. "
            "Pregnancy: teratogenic risk CBZ/OXC (Category D); valproate EXCLUDED (POLG). "
            "Driving: ALL SCN8A-DEE seizure-active patients — driving excluded. "
            "SUDEP counselling: long-term nocturnal supervision or wearable seizure alert. "
            "Social care / NDIS transition planning for dependent adults."
        ),
    },
]

# ── Concepts (14) ─────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "SCN8A / Nav1.6", "definition": "Sodium voltage-gated channel alpha subunit 8 (gene, 12q13.13) encoding Nav1.6 — the predominant sodium channel at axon initial segments (AIS) and nodes of Ranvier of mature excitatory and inhibitory neurons in brain and spinal cord. Expressed postnatally, peaking by 2–3 years — explains why SCN8A-DEE typically presents after day 1 (contrast SCN2A Nav1.2 which peaks neonatally). Critical for repetitive high-frequency firing; GOF in Nav1.6 causes pathological burst discharges and DEE."},
    {"term": "GOF-DEE (Gain-of-Function DEE)", "definition": "Enhanced Nav1.6 channel function due to pathogenic missense variant: increased persistent Na+ current (INaP), impaired fast inactivation, hyperpolarising shift in activation voltage (channel opens more easily), or expanded window current (overlap of activation and inactivation curves). Net effect: neurons fire more readily and cannot stop — seizures, encephalopathy, movement disorder. Opposite of LOF; distinct treatment strategy: Na-channel blockers are FIRST-LINE."},
    {"term": "EIEE13 (MIM#614558)", "definition": "Early Infantile Epileptic Encephalopathy 13 — OMIM designation for SCN8A-DEE. Characterised by: (1) early-onset focal seizures (day 1 to 18 months), (2) drug-resistant epilepsy, (3) global developmental delay/intellectual disability, (4) movement disorder (30–50%). The 'EIEE' series (EIEE1–100+) designates genetically defined developmental and epileptic encephalopathies."},
    {"term": "Persistent Na Current (INaP)", "definition": "A small but sustained Na+ current that persists even after the normal fast-inactivation transient Na+ current decays. In normal neurons, INaP is <1% of peak Na+ current. In GOF SCN8A variants, INaP is enhanced 2–10-fold — sufficient to keep neurons in a sustained depolarised state, enabling high-frequency burst discharges without a refractory period. Na-channel blockers (CBZ, OXC, PHT) preferentially suppress INaP at therapeutic concentrations."},
    {"term": "Movement Disorder (SCN8A-specific)", "definition": "A prominent extrapyramidal or hyperkinetic movement disorder — dystonia, choreiform movements, or hyperkinetic syndrome — occurring in 30–50% of SCN8A-DEE patients. Mechanistically Nav1.6-mediated (basal ganglia and cerebellar circuit dysfunction). Clinically important because: (1) it helps distinguish SCN8A-DEE from other Nav-channel DEEs, and (2) it PARADOXICALLY IMPROVES with CBZ/OXC optimisation — providing an additional treatment target beyond seizures."},
    {"term": "Drug-Resistant Epilepsy (DRE)", "definition": "Failure of adequate trials of 2 tolerated and appropriately chosen AEDs (ILAE 2010 definition). In SCN8A-DEE, DRE affects >90% of patients despite Na-channel blocker use — the defining challenge of the condition. DRE milestone triggers escalation: add-on therapy (CLB, LEV), ketogenic diet, VNS, presurgical evaluation, investigational (quinidine, ASO therapy)."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "SUDEP risk in SCN8A-DEE is among the highest of all genetic epilepsies — estimated 5–10 per 1,000 patient-years. Mechanism: nocturnal FBTCS → post-ictal generalised EEG suppression (PGES) → central apnoea → cardiorespiratory failure. Risk mitigation: nocturnal supervision, movement/apnoea alarm mattress, seizure alert device (Empatica E4), optimise nocturnal seizure control with CBZ/OXC. SUDEP counselling mandatory from diagnosis."},
    {"term": "Quinidine (SCN8A investigational)", "definition": "Class Ia antiarrhythmic with dual Na-channel (including Nav1.6) and Kv11.1 (hERG) blocking activity. Proposed to selectively block SCN8A GOF window current at therapeutic plasma concentrations (2–5 mg/L). Evidence limited to small case series (Johannesen 2019: 4/8 responders). QTc monitoring mandatory; not licensed for epilepsy; use under compassionate/investigational access only. ASO therapy in preclinical development may supersede quinidine."},
    {"term": "HLA-B*1502 SJS/TEN", "definition": "HLA-B*1502 allele (Southeast/East Asian ancestry) confers 10× increased risk of Stevens–Johnson syndrome / toxic epidermal necrolysis with CBZ/OXC. CPIC Level A evidence mandates genotyping before initiation in all patients with Asian ancestry (Han Chinese, Thai, Malaysian, Vietnamese, Philippine, South Asian). If HLA-B*1502 positive: avoid CBZ/OXC; use PHT (HLA-B*1502 also increases PHT risk in some populations — prefer LEV). Non-Asian ancestry: HLA-A*3101 (European) increases CBZ hypersensitivity risk (milder reactions)."},
    {"term": "SIADH (CBZ/OXC)", "definition": "Syndrome of Inappropriate ADH Secretion — antidiuretic hormone release stimulated by CBZ/OXC causing free water retention and dilutional hyponatraemia. OXC risk higher (20–30% in infants) than CBZ (<10%). Signs: Na+ <135 → reduced alertness, vomiting; Na+ <125 → seizures, coma. Management: fluid restriction (500–750 mL/m²/day), dose reduction; if Na+ <125 mmol/L: hold drug, IV 3% saline (carefully). Prevention: monitor Na+ q2wk first month."},
    {"term": "LTG-ABSOLUTE-CI-SCN8A", "definition": "Lamotrigine (LTG) is ABSOLUTELY CONTRAINDICATED in SCN8A GOF. LTG binds fast-inactivated Na-channels but in GOF variants its altered kinetics result in net INCREASED persistent Na+ current or window current — paradoxically worsening seizures. Multiple case reports document acute focal status epilepticus precipitated by LTG in SCN8A-DEE. If LTG is accidentally started and seizure exacerbation occurs: immediate discontinuation, IV PHT loading, PICU admission. The SCN8A Alliance maintains a mandatory LTG contraindication advisory."},
    {"term": "Electroclinical Dissociation", "definition": "Phenomenon where EEG demonstrates ongoing ictal activity despite apparent clinical quiescence (absent visible clinical seizure signs). Common in SCN8A-DEE after AED loading — particularly after IV benzodiazepines or PB which may suppress clinical signs while ictal discharges persist. Important implication: continued EEG monitoring mandatory after clinical seizure cessation in NICU/PICU; do NOT discontinue AED support based on clinical observation alone."},
    {"term": "SCN8A Alliance", "definition": "Patient advocacy and research foundation for SCN8A families (scn8a.net). Maintains: (1) international SCN8A patient registry (>800 patients), (2) LTG contraindication advisory, (3) connexion to ASO/gene therapy trials, (4) emergency hospital protocol card for families. Clinicians should recommend registration at diagnosis. Alliance collaborates with Meisler Lab (University of Michigan) — primary SCN8A research group."},
    {"term": "CPIC / Nav1.6 Pharmacogenomics Hub", "definition": "Clinical Pharmacogenomics Implementation Consortium (CPIC) guidelines govern HLA-B*1502 / CBZ-OXC (Level A) and HLA-A*3101 / CBZ (Level A for European). PharmGKB and CPIC guidelines are integrated into prescribing decision support in electronic health records. For SCN8A-DEE: HLA-B*1502 genotyping is the single most important pre-treatment pharmacogenomic test — failure to test before CBZ/OXC initiation in at-risk ancestry is a patient safety incident."},
]

# ── Absolute Contraindications (4) ────────────────────────────────────────────
ABSOLUTE_CONTRAINDICATIONS = [
    {
        "drug": "Lamotrigine (LTG)",
        "scope": "ALL SCN8A GOF variants — ABSOLUTE CONTRAINDICATION",
        "mechanism": "Na-channel activating effect at GOF variant Nav1.6 → acute seizure exacerbation / status epilepticus",
        "action": "NEVER initiate. If accidentally prescribed and worsening occurs: immediate discontinuation, IV PHT loading, PICU admission.",
        "evidence": "Multiple case reports; SCN8A Alliance mandatory advisory.",
    },
    {
        "drug": "CBZ / OXC in HLA-B*1502 positive patients",
        "scope": "Asian ancestry — ABSOLUTE CONTRAINDICATION unless HLA-B*1502 negative",
        "mechanism": "HLA-B*1502 → 10× SJS/TEN risk with CBZ/OXC (CPIC Level A); life-threatening cutaneous adverse reaction",
        "action": "Genotype BEFORE prescribing. If positive: use PHT (with HLA-B caution), LEV, or CLB; avoid CBZ/OXC.",
        "evidence": "CPIC HLA-B*1502 CBZ guideline 2023 Level A.",
    },
    {
        "drug": "VPA in POLG-positive patients",
        "scope": "Polymerase-gamma (POLG) mutation — ABSOLUTE CONTRAINDICATION",
        "mechanism": "VPA inhibits hepatic mitochondrial beta-oxidation; in POLG mutation → Alpers-Huttenlocher syndrome → fatal hepatic failure",
        "action": "POLG testing mandatory before initiating VPA in any DEE. If POLG positive: VPA excluded for life.",
        "evidence": "ACMG/CPIC; package insert Black Box Warning.",
    },
    {
        "drug": "Hospital NPO without IV AED continuity",
        "scope": "ALL SCN8A-DEE patients requiring nil-per-os (fasting) for procedures",
        "mechanism": "Oral AED held → CBZ/OXC trough → breakthrough seizure / status epilepticus / SUDEP",
        "action": "MANDATORY: IV PHT/fosphenytoin or IV LEV bridge during NPO periods. Hospital admission protocol card to be provided to family for all elective procedures.",
        "evidence": "ILAE/ILAE-DEE consensus; SCN8A Alliance hospital protocol advisory.",
    },
]

# ── Monitoring Thresholds (10) ────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Seizure onset <6 months", "action": "Initiate urgent molecular diagnosis (trio exome); SCN8A GOF confirmation before treatment pivot."},
    {"threshold": "CBZ TDM 4–12 µg/mL", "action": "Titrate to upper target range (10–12 µg/mL) before declaring CBZ failure in GOF; check for enzyme induction (auto-induction)."},
    {"threshold": "OXC-MHD 12–24 µmol/L", "action": "MHD target zone; if seizure breakthrough at 24 µmol/L → switch to CBZ (higher ceiling) or add adjunct."},
    {"threshold": "Na+ <130 mmol/L (SIADH)", "action": "Hold CBZ/OXC; fluid restrict; if Na+ <125 mmol/L: IV 3% NaCl cautiously (max 1–2 mEq/L/h correction rate)."},
    {"threshold": "HLA-B*1502 positive", "action": "Do NOT start CBZ/OXC; select alternative Na-channel blocker (PHT IV short-term) or broad-spectrum AED."},
    {"threshold": "QTc >500ms (quinidine)", "action": "Immediately stop quinidine; cardiology review; do not rechallenge."},
    {"threshold": "2 AED failures → KD", "action": "Refer to ketogenic diet programme; initiate KD within 4 weeks of 2nd AED failure."},
    {"threshold": "3 AED failures → VNS", "action": "Refer to epilepsy surgery / VNS programme; VNS implantation if surgery not candidate."},
    {"threshold": "Seizure-free 2Y → CBZ taper", "action": "Minimum 2 years seizure-free on CBZ/OXC before gradual taper (10%/2wk). Hospital access plan during taper."},
    {"threshold": "LFT >3× ULN → hold CBZ; >8× ULN → stop", "action": "Liver function monitoring: hold CBZ if transaminases >3× ULN; permanent discontinuation if >8× ULN or clinical hepatitis signs."},
]

# ── Standards (8) ─────────────────────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE 2022", "title": "International League Against Epilepsy Classification of DEE", "relevance": "Defines SCN8A-DEE within the Developmental and Epileptic Encephalopathy classification framework; mandates functional class (GOF/LOF) for treatment decisions."},
    {"standard": "NICE NG217 2022", "title": "NICE Epilepsies in Children, Young People and Adults", "relevance": "UK guideline for genetic epilepsy management; sodium-channel blocker recommendation for SCN8A-GOF; CBZ monitoring protocol."},
    {"standard": "CPIC HLA-B CBZ 2023", "title": "Clinical Pharmacogenomics Implementation Consortium HLA-B Genotype and Carbamazepine Dosing Guideline", "relevance": "Level A recommendation: HLA-B*1502 genotyping mandatory before CBZ/OXC initiation in patients of Asian ancestry; drives prescribing safety."},
    {"standard": "ACNS EEG 2021", "title": "American Clinical Neurophysiology Society Consensus Statement on EEG in Critically Ill Neonates and Children", "relevance": "Standards for neonatal EEG monitoring in SCN8A-DEE; electroclinical dissociation detection; CEEG duration recommendations."},
    {"standard": "ACMG-AMP 2015", "title": "ACMG/AMP Variant Classification Standards", "relevance": "Pathogenicity classification for SCN8A variants (Pathogenic/Likely Pathogenic/VUS); functional evidence integration for GOF/LOF determination."},
    {"standard": "EAN Neonatal SE 2019", "title": "European Academy of Neurology Neonatal Status Epilepticus Guideline", "relevance": "IV AED protocol for neonatal SE: PHT/fosphenytoin loading, PB adjunct; monitoring for electroclinical dissociation."},
    {"standard": "ILAE Dietary Therapies 2018", "title": "ILAE Consensus Report on Dietary Therapies for Epilepsy", "relevance": "KD 4:1 ratio protocol for refractory DEE including SCN8A; initiation, monitoring, and discontinuation criteria."},
    {"standard": "Meisler 2019 NEJM", "title": "Meisler et al. 2019 New England Journal of Medicine — Sodium Channelopathies in Neurodevelopmental Disorders", "relevance": "Primary reference establishing SCN8A clinical spectrum, GOF pathomechanism, and treatment framework; foundational paper for SCN8A-DEE management."},
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "Meisler-2019-NEJM", "title": "Meisler et al. (2019). Sodium Channelopathies in Neurodevelopmental Disorders. New England Journal of Medicine.", "relevance": "Definitive clinical spectrum and pathomechanism review for SCN8A-DEE; GOF/LOF framework; treatment algorithm."},
    {"ref": "Larsen-2015-Neurology", "title": "Larsen et al. (2015). The phenotypic spectrum of SCN8A encephalopathy. Neurology.", "relevance": "SCN8A-DEE clinical cohort; GOF treatment response to CBZ/PHT; movement disorder characterisation."},
    {"ref": "Blanchard-2017-Epilepsia", "title": "Blanchard et al. (2017). De novo gain-of-function and loss-of-function mutations of SCN8A in patients with intellectual disabilities and epilepsy. Epilepsia.", "relevance": "GOF vs LOF genotype-phenotype correlations; functional patch-clamp validation; treatment implications."},
    {"ref": "Johannesen-2019-EBM", "title": "Johannesen et al. (2019). Genotype-phenotype correlations in SCN8A-related disorders reveal prognostic and therapeutic implications. Brain.", "relevance": "Quinidine treatment response in refractory SCN8A-GOF; 4/8 responders; QTc monitoring protocol; movement disorder improvement with Na-channel blockers."},
    {"ref": "Gardella-2016-Epilepsia", "title": "Gardella et al. (2016). Activating variants of SCN8A in patients with generalized epilepsy. Epilepsia.", "relevance": "EEG ictal signatures (LVFA, recruiting discharge) in SCN8A-GOF; interictal multifocal spike-wave; electroclinical dissociation documentation."},
    {"ref": "Estacion-2014-JNeurosci", "title": "Estacion et al. (2014). A novel de novo mutation of SCN8A (Nav1.6) with enhanced channel activation in a child with epileptic encephalopathy. Neuroscience.", "relevance": "Patch-clamp demonstration of enhanced window current and persistent INaP in SCN8A-GOF; mechanistic basis for Na-channel blocker efficacy."},
]

# ── Patient generator ──────────────────────────────────────────────────────────
_CATEGORIES = [
    ("De-novo-SCN8A-GOF-severe-DEE", "GOF-severe"),
    ("De-novo-SCN8A-GOF-severe-DEE", "GOF-severe"),
    ("De-novo-SCN8A-GOF-severe-DEE", "GOF-severe"),
    ("De-novo-SCN8A-GOF-moderate", "GOF-moderate"),
    ("De-novo-SCN8A-LOF", "LOF"),
    ("De-novo-SCN8A-splice-structural", "Indeterminate"),
    ("Clinical-SCN8A-negative", "Phenocopy"),
]
_WEIGHTS = [24, 24, 10, 7, 4, 3, 3]  # proportional to etiology N

_DISEASE_PHASES = [
    "Neonatal-NICU", "Early-infantile", "Late-infantile",
    "Early-childhood", "School-age", "Adolescence-adult",
]
_TREATMENTS_SHORT = ["CBZ", "OXC", "PB+OXC", "CBZ+LEV", "OXC+CLB", "CBZ+KD", "OXC+KD", "LEV+KD"]
_SEIZURE_CONTROL = ["seizure-free", ">90%reduction", "50-90%reduction", "partial", "drug-resistant"]
_CONTROL_WEIGHTS = [8, 10, 12, 25, 45]  # >90% DRE realistic


def _wt_choice(options, weights):
    total = sum(weights)
    r = random.random() * total
    cumul = 0
    for opt, w in zip(options, weights):
        cumul += w
        if r <= cumul:
            return opt
    return options[-1]


def _generate_patients():
    pts = []
    cat_list = []
    for cat, count in zip(
        [e["category"] for e in ETIOLOGY_CATALOG],
        [e["n"] for e in ETIOLOGY_CATALOG],
    ):
        cat_list.extend([cat] * count)
    random.shuffle(cat_list)

    for i, cat in enumerate(cat_list[:41]):
        fclass = "GOF-severe" if "GOF-severe" in cat else (
            "GOF-moderate" if "GOF-moderate" in cat else (
            "LOF" if "LOF" in cat else "Indeterminate"))
        onset_days = random.randint(0, 360) if "severe" in cat else random.randint(60, 540)
        age_months = random.randint(6, 180)
        cbz_level = round(random.uniform(3.5, 12.5), 1) if fclass.startswith("GOF") else None
        mhd_level = round(random.uniform(10.0, 26.0), 1) if fclass.startswith("GOF") and random.random() > 0.5 else None
        na = round(random.uniform(128.0, 145.0), 1)
        hla_tested = random.random() > 0.25
        hla_pos = random.random() < 0.08 if hla_tested else None
        kd_on = random.random() < 0.30
        bhb = round(random.uniform(1.5, 4.2), 1) if kd_on else None
        polg_tested = random.random() > 0.20
        movement_disorder = random.random() < 0.38 if fclass.startswith("GOF") else False
        ctrl = _wt_choice(_SEIZURE_CONTROL, _CONTROL_WEIGHTS)
        vns = random.random() < 0.18
        phase = random.choice(_DISEASE_PHASES)
        pts.append({
            "id": f"SCN8A-{i+1:03d}",
            "age_months": age_months,
            "sex": random.choice(["M", "F"]),
            "onset_age_days": onset_days,
            "functional_class": fclass,
            "category": cat,
            "disease_phase": phase,
            "current_treatment": random.choice(_TREATMENTS_SHORT),
            "seizure_control": ctrl,
            "kd_on": kd_on,
            "bhb_mmoll": bhb,
            "cbz_level_ugml": cbz_level,
            "mhd_umoll": mhd_level,
            "na_mmoll": na,
            "hla_b1502_tested": hla_tested,
            "hla_b1502_positive": hla_pos,
            "polg_tested": polg_tested,
            "movement_disorder": movement_disorder,
            "vns_implanted": vns,
            "eeg_lvfa_pattern": fclass.startswith("GOF") and random.random() > 0.30,
        })
    return pts


def get_overview():
    pts = _generate_patients()
    gof_pts = [p for p in pts if p["functional_class"].startswith("GOF")]
    lof_pts = [p for p in pts if p["functional_class"] == "LOF"]
    gof_free = sum(1 for p in gof_pts if p["seizure_control"] == "seizure-free")
    dre = sum(1 for p in pts if p["seizure_control"] == "drug-resistant")
    movement = sum(1 for p in pts if p["movement_disorder"])
    kd_on = sum(1 for p in pts if p["kd_on"])
    polg = sum(1 for p in pts if p["polg_tested"])
    hla = sum(1 for p in pts if p["hla_b1502_tested"])
    return {
        "syndrome": "SCN8A Encephalopathy (SCN8A-DEE / EIEE13)",
        "gene": "SCN8A (12q13.13)",
        "protein": "Nav1.6 Voltage-Gated Sodium Channel α Subunit",
        "inheritance": "De novo (>97%); autosomal dominant",
        "omim": "EIEE13 — MIM#614558",
        "n_patients": 41,
        "eeg_hallmark": "LVFA (low-voltage fast activity) ictal onset; multifocal spike-wave; synchronous neonatal burst-suppression (GOF-severe)",
        "key_biomarker": "SCN8A variant functional class (GOF vs LOF) — mandatory before AED selection",
        "key_aha": (
            "SCN8A-DEE is ALMOST EXCLUSIVELY GOF — CBZ/OXC first-line. "
            "NEVER use lamotrigine (LTG) in SCN8A GOF — causes acute SE. "
            "Movement disorder (dystonia/hyperkinetic) in 30–50% — paradoxically improves with CBZ/OXC. "
            "SUDEP risk very high — nocturnal supervision + seizure alert MANDATORY."
        ),
        "kpis": {
            "gof_pct": round(len(gof_pts) / 41 * 100),
            "lof_pct": round(len(lof_pts) / 41 * 100),
            "gof_seizure_free_pct": round(gof_free / max(len(gof_pts), 1) * 100),
            "dre_pct": round(dre / 41 * 100),
            "movement_disorder_pct": round(movement / 41 * 100),
            "kd_on_pct": round(kd_on / 41 * 100),
            "polg_tested_pct": round(polg / 41 * 100),
            "hla_tested_pct": round(hla / 41 * 100),
        },
        "etiologies": [
            {"etiology": e["etiology"], "n": e["n"], "pct": e["pct"]}
            for e in ETIOLOGY_CATALOG
        ],
        "seizure_type_prevalence": {s["type"]: s["prevalence_pct"] for s in SEIZURE_TYPES},
        "trigger_seizure_rates": {t["trigger"]: t["rate_pct"] for t in TRIGGERS},
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "clinical_alerts": [
            "🚨 LTG (Lamotrigine) ABSOLUTELY CONTRAINDICATED in SCN8A GOF — causes acute status epilepticus",
            "🚨 HLA-B*1502 genotyping MANDATORY before CBZ/OXC initiation (Asian ancestry — CPIC Level A)",
            "⚡ First-line: PHT IV bridge (NICU) → oral CBZ or OXC (GOF confirmed)",
            "⚡ Movement disorder (dystonia/hyperkinetic) in 30–50% — may improve with CBZ/OXC optimisation",
            "🧪 SIADH: Na+ monitoring q2wk×1M with OXC (higher risk than CBZ); hold if Na+ <130 mmol/L",
            "🚨 SUDEP risk very high — nocturnal supervision, seizure alert device, bedside O2 SAT monitoring MANDATORY",
            "⚡ POLG exclusion mandatory before VPA initiation (fatal hepatic failure in POLG+VPA)",
            "🧪 Quinidine: investigational GOF-specific agent — QTc ECG monitoring mandatory; stop if QTc >500ms",
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_breakdown():
    pts = _generate_patients()
    return {
        "patients": pts,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": AED_MONITORING,
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "standards": STANDARDS,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "absolute_contraindications": ABSOLUTE_CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "standards": STANDARDS,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
