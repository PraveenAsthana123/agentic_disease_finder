"""
KCNT1 Encephalopathy — EIMFS Dashboard
=========================================
41-patient cohort · KCNT1 (9q34.3) · KNa1.1 / Slack / Slo2.2 Sodium-Activated K+ Channel
KCNT1 encephalopathy: pathogenic gain-of-function (GOF) variants in KCNT1 (potassium sodium-
activated channel subfamily T member 1, 9q34.3) causing the most severe neonatal–infantile
DEE phenotype: EIMFS (Epilepsy of Infancy with Migrating Focal Seizures; OMIM 614959) and,
at the milder end, autosomal-dominant NFLE (Nocturnal Frontal Lobe Epilepsy, OMIM 615005).

CHANNEL BIOLOGY: KCNT1 encodes KNa1.1 (also called Slack / Slo2.2), a high-conductance
sodium-activated potassium channel expressed in cortical and thalamic interneurons, layer-5
pyramidal cells, and GABAergic circuits. Physiologically, KNa1.1 opens after high-frequency
neuronal firing, allowing K+ efflux and generating a slow after-hyperpolarisation (sAHP) that
dampens repetitive burst activity (a "brake" on excitation). GOF variants cause constitutively
enhanced channel activity → excessive K+ efflux → paradoxical hyperexcitability, likely via
disproportionate interneuron deactivation, disrupting cortical inhibitory circuits.

EIMFS HALLMARK: Migrating, multifocal, polymorphous focal seizures with shifting hemispheric
onset — a unique electrographic signature (migrating EEG focal discharges that "hop" between
hemispheres and lobes within and between seizures). Onset 1–4 months of age. Seizures are
nearly continuous (100+ per day at peak), resistant to all conventional AEDs. Prognosis:
profound intellectual disability, minimal motor and language development, SUDEP risk HIGH.

QUINIDINE CONTROVERSY: Quinidine (a voltage-gated K-channel blocker with secondary Na-channel
activity) was initially proposed to reduce KCNT1 GOF activity. Early case reports (Milligan
2014, Bearden 2014) showed striking benefit. However, the first RCT — KCNT1 in EIMFS (Numis
2020, Epilepsia) — showed NO significant reduction in seizure frequency vs placebo. Quinidine
is therefore NOT recommended for EIMFS based on current evidence. Some clinicians continue use
for NFLE-KCNT1 (milder phenotype, longer treatment window, positive case series).

MOST EFFECTIVE NON-PHARMACOLOGICAL: Ketogenic Diet (KD 4:1 ratio) — multiple case series
report 50–80% seizure reduction in EIMFS, often the single most effective intervention.
Phenobarbital is first-line acute; clobazam and levetiracetam adjunct; VPA requires POLG
exclusion (mandatory); vigabatrin (SHARE REMS, USA) for IS component.

AED NOTES: No true disease-specific Na-channel blocker therapy (contrast SCN8A-GOF).
Broad-spectrum approach: PB acute → CLB focal adjunct → KD 4:1 → VNS (DRE ≥3 AEDs).
Avoid: vigabatrin without SHARE REMS; VPA without POLG exclusion; quinidine for EIMFS.
DISEASE-MODIFYING: ASO antisense oligonucleotide programmes (preclinical) and AAV9-based
gene silencing (early feasibility) are in development for KCNT1 GOF.
"""

import random
from datetime import datetime

SEED = 9178  # dashboard 178
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "De novo KCNT1 GOF missense (EIMFS, severe)",
        "n": 20, "pct": 49,
        "category": "De-novo-KCNT1-GOF-EIMFS-severe",
        "mechanism": (
            "Most prevalent KCNT1-DEE variant class (~49%): de novo missense variants in KCNT1 "
            "causing pronounced gain-of-function of KNa1.1 — constitutive channel over-opening, "
            "enhanced Na+-dependent K+ efflux, and paradoxical cortical hyperexcitability via "
            "interneuron network disruption. Common variant hotspots include p.Arg474His (most "
            "frequent EIMFS variant), p.Arg474Cys, p.Gly288Ser, and p.Lys629Glu, all clustering "
            "in the cytoplasmic RCK (Regulator of K+ Conductance) domain that is critical for "
            "Na+-dependent gating. GOF leads to nearly continuous multifocal migrating seizures "
            "in infancy, profound developmental regression, and drug-resistant epilepsy (>90%)."
        ),
        "eeg_signature": (
            "EIMFS hallmark: migrating focal ictal discharges that shift between hemispheres and "
            "lobes within a single seizure (inter-ictal → ictal transition in consecutive foci). "
            "High-amplitude rhythmic theta/delta discharge with rapid spread; background suppression "
            "between clusters; absence of burst-suppression distinguishes from KCNQ2/SCN8A-DEE."
        ),
        "mri": "Usually normal in early infancy; progressive diffuse volume loss/atrophy in severe cases.",
        "clinical_note": (
            "EIMFS is clinically diagnosed by the migrating seizure pattern on EEG + onset <6 months. "
            "Molecular confirmation (trio exome or KCNT1 gene panel) mandatory to distinguish from "
            "SCN1A-EIMFS, SLC25A22, CACNA1A and other EIMFS causes (15+ genes known). "
            "GOF functional class: patch-clamp or computational modelling if functional status uncertain. "
            "Ketogenic diet (4:1) should be initiated early — highest evidence for seizure reduction."
        ),
    },
    {
        "etiology": "De novo KCNT1 GOF missense (moderate / NFLE-overlap)",
        "n": 9, "pct": 22,
        "category": "De-novo-KCNT1-GOF-moderate",
        "mechanism": (
            "Moderate GOF variants (~22%): de novo KCNT1 missense variants with partial gain-of-function "
            "causing a less catastrophic phenotype — focal seizures (often nocturnal, frontal-onset), "
            "developmental disability (mild–moderate ID rather than profound), and some capacity for "
            "speech/language acquisition. Overlapping phenotype between EIMFS and NFLE. Response to "
            "AEDs (CLB, PB, LEV) better than severe GOF. KD still recommended as adjunct. "
            "NFLE-like: seizures during NREM sleep, hypermotor semiology, frontal EEG focus."
        ),
        "eeg_signature": "Frontal-dominant focal discharge; NREM-sleep activation; less migratory pattern than severe GOF.",
        "mri": "Usually normal.",
        "clinical_note": (
            "Better prognosis than severe EIMFS: some patients achieve schooling, aided communication. "
            "CLB + KD combination often achieves ≥50% seizure reduction. Quinidine anecdotally positive "
            "in NFLE-KCNT1 (distinct from EIMFS where RCT showed no benefit) — QTc monitoring mandatory "
            "before trialling. Genetic counselling essential: GOF variants may be dominantly inherited."
        ),
    },
    {
        "etiology": "Familial KCNT1 GOF (NFLE-AD, autosomal dominant)",
        "n": 6, "pct": 15,
        "category": "Familial-KCNT1-GOF-NFLE-AD",
        "mechanism": (
            "Autosomal-dominant familial KCNT1 GOF (~15%): the same GOF mechanism transmitted in "
            "families — typically the NFLE phenotype (Nocturnal Frontal Lobe Epilepsy). Onset "
            "childhood to adulthood; seizures are nocturnal, hypermotor, frontal-onset; inter-ictal "
            "EEG often normal. Variable expressivity within families (some members severely affected, "
            "others mildly). Not all NFLE-KCNT1 is severe: familial cases tend to have milder "
            "developmental impact than de novo EIMFS. KCNT1 is the single most common identified "
            "monogenic cause of autosomal-dominant NFLE (~7–13% of familial NFLE)."
        ),
        "eeg_signature": "Frontal focal discharges during NREM stage 2; ictal pattern: rhythmic theta, semi-periodic frontal.",
        "mri": "Normal.",
        "clinical_note": (
            "Genetic counselling critical: 50% transmission risk to offspring. CBZ/OXC often partially "
            "effective for nocturnal seizures. Quinidine trials in NFLE-KCNT1 reported positive in "
            "small series; requires QTc baseline + repeat monitoring. Quality of life often manageable; "
            "driving safety and nocturnal supervision key counselling points."
        ),
    },
    {
        "etiology": "De novo KCNT1 splice site / structural variant",
        "n": 3, "pct": 8,
        "category": "De-novo-KCNT1-splice-structural",
        "mechanism": (
            "Splice site and copy number variants (~8%): may produce partial GOF (exon-skipping causing "
            "truncation of regulatory RCK2 domain with paradoxical GOF) or LOF. Functional consequence "
            "must be determined by RNA studies, patch-clamp analysis, or ClinGen Expert Panel review. "
            "Phenotype intermediate — between severe EIMFS and NFLE — depending on functional impact."
        ),
        "eeg_signature": "Variable — depends on functional class; migratory if GOF, focal-frontal if NFLE-like.",
        "mri": "Variable.",
        "clinical_note": (
            "Whole-exome + RNA panels recommended; standard exome may miss deep-intronic splice variants. "
            "Pending functional class, broad-spectrum AED (LEV, CLB) + early KD consultation warranted."
        ),
    },
    {
        "etiology": "Clinical KCNT1-negative (EIMFS phenocopy)",
        "n": 3, "pct": 6,
        "category": "Clinical-KCNT1-negative-EIMFS-phenocopy",
        "mechanism": (
            "Patients with migrating focal seizure pattern (EIMFS EEG signature + onset <6 months) "
            "without pathogenic KCNT1 variant on standard sequencing (~6%). Other EIMFS genes: "
            "SCN1A, SLC25A22 (mitochondrial glutamate carrier), CACNA1A, PLCB1, ATP1A3, TBC1D24, "
            "SLC35A2, FOXG1, and others. Somatic mosaicism (KCNT1) below detection threshold also "
            "possible. Comprehensive EIMFS gene panel or trio exome + mtDNA analysis recommended."
        ),
        "eeg_signature": "EIMFS migratory pattern — indistinguishable from KCNT1-positive on routine EEG.",
        "mri": "Usually normal; basal ganglia signal changes in SLC25A22 variants.",
        "clinical_note": (
            "Phenotypic overlap with other EIMFS genes alters management: SLC25A22 requires "
            "mitochondrial workup; CACNA1A may respond differently. Empiric KD is reasonable pending "
            "genetic diagnosis; avoid VPA until POLG excluded."
        ),
    },
]

# ── Seizure Types (4 primary, N=41 cohort) ──────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Migrating focal seizures (EIMFS signature)",
        "prevalence_pct": 90,
        "onset_age": "1–4 months (median 10 weeks)",
        "eeg_correlate": (
            "Defining EIMFS EEG pattern: sequential focal ictal discharges beginning in one hemisphere, "
            "then shifting to the contralateral hemisphere or a different lobe within the same seizure "
            "or successive seizures — the 'migrating' pattern. Ictal discharge: rhythmic alpha–theta "
            "(6–8 Hz) evolving to higher amplitude, followed by post-ictal voltage attenuation. "
            "Multiple simultaneous bilateral independent focal discharges ('multifocal storm') at peak."
        ),
        "clinical_tip": (
            "The migrating EEG pattern is the diagnostic fingerprint of EIMFS. Request 24-hour VEEG "
            "at first evaluation — routine EEG may miss the migratory phenomenon during inter-ictal "
            "periods. Confirm KCNT1 with trio exome. Initiate KD consultation simultaneously with "
            "PB loading — do not wait for genetic result if EIMFS EEG pattern is present."
        ),
    },
    {
        "type": "Focal tonic / clonic (non-migratory)",
        "prevalence_pct": 78,
        "onset_age": "Birth to 6 months",
        "eeg_correlate": (
            "Focal low-voltage fast activity or rhythmic discharge at frontal/temporal onset; "
            "may not migrate in less severe variants; interictal multifocal independent spike-wave."
        ),
        "clinical_tip": (
            "Differentiate from KCNQ2-DEE (neonatal, more stereotyped, no migration) and SCN8A-DEE "
            "(LVFA, movement disorder). KCNT1 focal seizures respond partially to PB and CLB; "
            "Na-channel blockers are NOT specifically indicated (no GOF Nav channel in KCNT1)."
        ),
    },
    {
        "type": "Infantile spasms (West syndrome overlap)",
        "prevalence_pct": 45,
        "onset_age": "3–9 months",
        "eeg_correlate": (
            "Modified hypsarrhythmia or multifocal high-amplitude spike-wave; spasms may co-exist "
            "with ongoing migrating focal seizures. Hypsarrhythmia pattern less classic than idiopathic "
            "IS — more multifocal, asymmetric. ACTH may reduce spasm cluster frequency."
        ),
        "clinical_tip": (
            "IS in the context of ongoing EIMFS: ACTH 4 IU/kg/day for IS component alongside "
            "KD for focal seizures. VGB (SHARE REMS) is second-line for IS; note theoretical concern "
            "of VGB worsening focal seizures in some EIMFS — monitor closely."
        ),
    },
    {
        "type": "Focal-to-bilateral tonic-clonic (FBTCS)",
        "prevalence_pct": 55,
        "onset_age": "Infancy to early childhood",
        "eeg_correlate": (
            "Focal onset with rapid bilateral spread; post-ictal suppression. High SUDEP risk "
            "with nocturnal FBTCS — requires nocturnal monitoring and SUDEP counselling."
        ),
        "clinical_tip": (
            "Nocturnal FBTCS: bedside pulse oximetry, consider CPAP review for co-morbid OSA. "
            "Maximise CLB and KD before declaring DRE. VNS after 3 AED failures provides "
            "modest benefit in EIMFS (20–30% responder rate, rarely seizure-free)."
        ),
    },
]

# ── Triggers (8) ──────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / hyperthermia", "rate_pct": 88,
     "note": "Fever-provoked seizure clusters (Dravet-like thermoregulatory sensitivity) — most potent KCNT1 trigger. Preventive: antipyretics at 38°C threshold; cool environment; written fever management plan."},
    {"trigger": "Intercurrent illness / infection", "rate_pct": 76,
     "note": "Systemic illness amplifies seizure burden even without fever. Admission threshold low; IV AED continuation mandatory during NPO."},
    {"trigger": "Missed / delayed AED dose", "rate_pct": 65,
     "note": "PB and CLB must not be missed — rebound seizure clusters. Caregiver pill-alarm + backup IM midazolam (buccal) prescription essential."},
    {"trigger": "Sleep deprivation", "rate_pct": 54,
     "note": "Disrupted sleep architecture worsens nocturnal seizure burden, especially in NFLE-KCNT1. Structured sleep hygiene mandatory; avoid co-sleeping (SUDEP risk)."},
    {"trigger": "Stress / overstimulation", "rate_pct": 42,
     "note": "Sensory and emotional stress precipitates clusters. Caregiver education on sensory regulation; low-stimulation hospital environment."},
    {"trigger": "AED taper / withdrawal", "rate_pct": 38,
     "note": "AED withdrawal without specialist supervision risks prolonged seizure clusters and status epilepticus. Written status action plan mandatory."},
    {"trigger": "Puberty (NFLE-onset)",  "rate_pct": 28,
     "note": "Familial NFLE-KCNT1: hormonal changes at puberty may unmask or worsen nocturnal seizures. Increased EEG monitoring at puberty onset; AED review."},
    {"trigger": "Feeding / vagal stimulation (neonatal)", "rate_pct": 22,
     "note": "Neonatal KCNT1 patients occasionally exhibit feeding-triggered seizures (vagal reflex). NGT feeding in quiet setting; slow feeds; prone positioning avoided."},
]

# ── Treatments (8) ────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Phenobarbital (PB)",
        "evidence": "Level-B — acute neonatal/infantile first-line",
        "indication": "First-line AED for acute neonatal and infantile EIMFS seizure control; bridge to KD.",
        "dose": "Loading: 20 mg/kg IV; maintenance 3–5 mg/kg/day PO/IV in 1–2 divided doses. Target TDM 20–40 µg/mL.",
        "moa": (
            "GABAa receptor positive allosteric modulator — prolongs Cl– channel open time, "
            "hyperpolarises neuronal membrane. GABA-A modulation partially compensates for "
            "KCNT1-driven interneuron dysfunction."
        ),
        "efficacy": "Reduces seizure frequency 30–60% in EIMFS; rarely seizure-free. Most effective acute agent.",
        "safety": "Sedation, paradoxical agitation, respiratory depression (IV load), cognitive dulling on long-term use.",
        "monitoring": "TDM q4–6 weeks; liver function panel q6 months; respiratory monitoring during IV loading.",
        "contraindications": "Porphyria (absolute); severe hepatic impairment (relative).",
    },
    {
        "drug": "Clobazam (CLB)",
        "evidence": "Level-B — focal seizure adjunct",
        "indication": "Adjunct for focal and migrating focal seizures; combination with PB often first-line doublet.",
        "dose": "0.05 mg/kg/day increasing to 0.3–1.0 mg/kg/day in 2 divided doses (max 40 mg/day). Titrate over 4 weeks.",
        "moa": (
            "Benzodiazepine — GABA-A positive modulator (1,5-benzodiazepine selective for α2/α3 subunits). "
            "More tolerable sedation profile than clonazepam. Tolerance develops over weeks–months in some."
        ),
        "efficacy": "50% seizure reduction in ~40% of EIMFS patients; benefit often partial and time-limited (tolerance).",
        "safety": "Sedation, hypersalivation, ataxia; tolerance/rebound seizure at abrupt withdrawal.",
        "monitoring": "Assess sedation level monthly; active metabolite N-desmethylclobazam TDM if poor response or toxicity.",
        "contraindications": "Severe respiratory insufficiency; avoid abrupt discontinuation.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level-C — broad-spectrum adjunct",
        "indication": "Adjunct in EIMFS and NFLE-KCNT1; added to PB or CLB.",
        "dose": "20 mg/kg/day IV/PO → up to 60 mg/kg/day in 2–3 divided doses.",
        "moa": "SV2A synaptic vesicle glycoprotein modulator — reduces synaptic vesicle fusion/neurotransmitter release.",
        "efficacy": "20–30% seizure reduction as adjunct; rarely transformative in EIMFS.",
        "safety": "Irritability, aggression (5–10%); rare hepatic effect; renal dose adjustment.",
        "monitoring": "Behaviour assessment at each clinic visit; CMP q6 months.",
        "contraindications": "None specific to KCNT1; dose reduce in renal impairment.",
    },
    {
        "drug": "Ketogenic Diet (KD 4:1 ratio)",
        "evidence": "Level-B — most effective non-pharmacological therapy for EIMFS",
        "indication": "Highly recommended for all EIMFS/KCNT1-DEE patients; initiate after 1st or 2nd AED failure.",
        "dose": (
            "4:1 fat:carbohydrate+protein ratio; initiation with 24–48h gradual fasting protocol "
            "or graded induction (avoid hypoglycaemia). Maintain BHB 2–4 mmol/L. Supplemented with "
            "selenium, zinc, carnitine, vitamins B/D/calcium. Dietitian supervision mandatory."
        ),
        "moa": (
            "Ketone bodies (BHB, AcAc) as alternative neuronal fuel → altered neuronal metabolism → "
            "reduced glutamate excitotoxicity; KATP channel opening; reduced ROS; adenosine-mediated "
            "inhibition. Mechanism in KCNT1 likely metabolic stabilisation of interneuron circuits."
        ),
        "efficacy": (
            "Multiple case series: 50–80% seizure reduction in EIMFS on KD (Rizzo 2016, Caraballo 2015). "
            "Higher responder rates than any single AED in EIMFS. ~15% seizure freedom on KD."
        ),
        "safety": "Hyperlipidaemia, nephrolithiasis (renal US q6M), growth restriction (monitor height/weight), constipation, acidosis.",
        "monitoring": "BHB 2–4 mmol/L; lipid panel q6M; renal US q6M; micronutrient levels (Se, Zn, carnitine) q6M; growth chart.",
        "contraindications": "POLG mutations (mitochondrial), pyruvate carboxylase deficiency, fatty acid oxidation disorders (FAO screen before initiation).",
    },
    {
        "drug": "ACTH (Adrenocorticotropic hormone)",
        "evidence": "Level-B — infantile spasms component (UKISS protocol)",
        "indication": "West syndrome (IS) component in KCNT1-DEE; adjunct to KD and focal AEDs.",
        "dose": "4 IU/kg/day IM x 2 weeks (UKISS protocol); taper over 2 weeks. Prednisolone alternative: 10 mg TDS x 2 weeks.",
        "moa": "Adrenocortical stimulation → cortisol elevation → reduced CRH (pro-epileptic neuropeptide in IS); direct anti-epileptic ACTH receptor in CNS.",
        "efficacy": "EEG response (hypsarrhythmia resolution) at day 14 in ~40% of IS-KCNT1 component; clinical spasm cessation variable.",
        "safety": "Hypertension, hyperglycaemia, Cushingoid features, infection risk, irritability, growth suppression.",
        "monitoring": "BP daily during course; glucose monitoring; infection surveillance; day-14 EEG (UKISS criterion).",
        "contraindications": "Active infection; live vaccines during and 3 months after ACTH course.",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "evidence": "Level-C — IS component (SHARE REMS required, USA)",
        "indication": "Second-line for IS component in KCNT1-DEE; NOT first-line for EIMFS focal seizures.",
        "dose": "50–150 mg/kg/day in 2 divided doses. FDA SHARE REMS enrolment required in USA.",
        "moa": "Irreversible GABA-T inhibitor → elevated synaptic GABA; theoretical concern of excess GABA reducing KNa1.1 compensatory action.",
        "efficacy": "IS suppression in ~40% (UKISS); efficacy for EIMFS focal seizures uncertain — monitor for worsening.",
        "safety": "Permanent visual field defect (VFD) — Goldman perimetry q3M mandatory; MRI T2 signal changes in infants; weight gain.",
        "monitoring": "Goldman perimetry q3M (SHARE REMS); OCT q6M; MRI if behaviour change; ophthalmology enrolment at initiation.",
        "contraindications": "Without SHARE REMS enrolment (USA); pre-existing visual field defect (relative).",
    },
    {
        "drug": "Quinidine",
        "evidence": "NOT RECOMMENDED for EIMFS — negative RCT (Numis 2020); anecdotal use in NFLE-KCNT1 only",
        "indication": (
            "EIMFS: NOT indicated — RCT (Numis 2020, Epilepsia) showed NO significant seizure reduction vs placebo. "
            "NFLE-KCNT1: anecdotal positive reports in small series; if trialled, requires cardiology clearance + QTc monitoring."
        ),
        "dose": "If trialled in NFLE-KCNT1: 3–6 mg/kg/day in 3 divided doses PO; target plasma 2–5 mg/L. NOT for EIMFS.",
        "moa": "Kv11.1 (hERG) blocker; secondary Na-channel blockade; reduces KCNT1 GOF channel open probability at high concentrations.",
        "efficacy": "EIMFS RCT: null result. NFLE case series: 50% seizure reduction in ~3/5 patients. Evidence remains anecdotal for NFLE.",
        "safety": "QTc prolongation → torsades de pointes; GI upset; haematological (thrombocytopenia); hypotension.",
        "monitoring": "Baseline QTc ECG; QTc every 2 weeks for 3 months then monthly; withhold if QTc >500 ms.",
        "contraindications": "EIMFS (negative RCT — do not use); QTc >450 ms baseline; concurrent QT-prolonging drugs; G6PD deficiency.",
    },
    {
        "drug": "Valproate (VPA)",
        "evidence": "Level-C — broad-spectrum adjunct; POLG exclusion MANDATORY",
        "indication": "Broad-spectrum adjunct for EIMFS if multiple AED failures and KD contraindicated; POLG exclusion mandatory.",
        "dose": "20–40 mg/kg/day in 2–3 divided doses; target TDM 50–100 µg/mL. IV valproate for acute seizure management.",
        "moa": "Na-channel blockade; GABA transaminase inhibition; HDAC inhibition; T-type calcium channel blockade.",
        "efficacy": "Modest adjunct effect in EIMFS; rarely transformative.",
        "safety": "Fatal hepatic failure in POLG mutations (contraindicated); teratogenic (REMS); hyperammonaemia; weight gain; hair loss.",
        "monitoring": "POLG gene panel MANDATORY before initiation; TDM q4–8 weeks; LFT + ammonia q3M; carnitine levels.",
        "contraindications": "POLG mutations (absolute — fatal mitochondrial hepatopathy); mitochondrial disease (relative); childbearing (teratogenicity REMS).",
    },
]

# ── AED Monitoring Panel (8) ──────────────────────────────────────────────────
AED_MONITORING = [
    {"item": "PB TDM", "target": "20–40 µg/mL", "frequency": "q4–6 weeks", "rationale": "Narrow therapeutic window; toxicity (sedation, ataxia) at >40; sub-therapeutic below 15."},
    {"item": "KD BHB (ketone bodies)", "target": "2–4 mmol/L", "frequency": "Daily (home) / q2 weeks (clinic)", "rationale": "Ketonemia target for seizure reduction; BHB < 2 = insufficient ketosis; > 5 = risk acidosis."},
    {"item": "KD micronutrients (Se, Zn, carnitine)", "target": "Normal range", "frequency": "q6 months", "rationale": "KD deplets selenium (cardiomyopathy risk), zinc, L-carnitine; supplement and monitor."},
    {"item": "VGB Goldman perimetry (SHARE REMS)", "target": "No VFD progression", "frequency": "q3 months", "rationale": "Vigabatrin causes irreversible bilateral concentric VFD — SHARE REMS mandatory requirement."},
    {"item": "ACTH BP / glucose", "target": "BP <95th pct; BG <10 mmol/L", "frequency": "Daily during ACTH course", "rationale": "ACTH causes mineralocorticoid-mediated hypertension and glucocorticoid-mediated hyperglycaemia."},
    {"item": "EEG / VEEG (migration tracking)", "target": "Seizure reduction; migration pattern", "frequency": "q6 months (2h minimum)", "rationale": "Assess migration pattern, background maturation, response to KD — key outcome metric in EIMFS."},
    {"item": "Bayley-III developmental", "target": "Domain scores; trajectory", "frequency": "q6 months", "rationale": "Profound developmental delay universal; track trajectory for AED/KD treatment decisions and intervention planning."},
    {"item": "Quinidine QTc ECG (if trialled in NFLE)", "target": "QTc < 500 ms", "frequency": "q2 weeks × 3M then monthly", "rationale": "Quinidine prolongs QTc → torsades risk; withhold if QTc > 500 ms. NOT for EIMFS (negative RCT)."},
]

# ── Absolute Contraindications (4) ───────────────────────────────────────────
ABSOLUTE_CI = [
    {
        "drug": "Quinidine — NOT RECOMMENDED in EIMFS (negative RCT 2020)",
        "scope": "EIMFS (all KCNT1-DEE infants)",
        "mechanism": (
            "Quinidine inhibits KCNT1 GOF channel activity in vitro and in Xenopus oocyte models. "
            "However, the first RCT in EIMFS (Numis et al. 2020, Epilepsia) demonstrated NO statistically "
            "significant reduction in seizure frequency vs placebo. CNS penetration may be insufficient "
            "at safe plasma levels; pharmacokinetics in infants unpredictable."
        ),
        "action": "Do NOT prescribe quinidine for KCNT1-EIMFS. Anecdotal use in NFLE-KCNT1 only (with cardiology + QTc monitoring).",
        "evidence": "Numis et al. 2020 Epilepsia — phase 2 RCT (n=9 EIMFS) — primary endpoint missed (p=0.38).",
    },
    {
        "drug": "Valproate (VPA) — POLG exclusion MANDATORY",
        "scope": "All patients until POLG panel result confirmed negative",
        "mechanism": (
            "VPA causes fatal mitochondrial hepatopathy in patients with POLG (polymerase gamma) mutations "
            "— the enzyme essential for mitochondrial DNA replication. KCNT1 variants may co-occur with "
            "other genetic variants including POLG. VPA inhibits mitochondrial beta-oxidation → hepatocellular "
            "necrosis + lactic acidosis in POLG-deficient patients → liver failure, Alpers syndrome, death."
        ),
        "action": "Order POLG panel before initiating VPA in ANY infant with DEE. If POLG result pending, do not start VPA.",
        "evidence": "EAN Consensus 2019; NICE NG217; ACMG POLG variant pathogenicity guidelines.",
    },
    {
        "drug": "Vigabatrin (VGB) without SHARE REMS enrolment (USA)",
        "scope": "USA prescribers only; standard enrolment requirement for all VGB use",
        "mechanism": (
            "VGB causes irreversible bilateral concentric visual field defects (VFD) in 20–40% of children "
            "and 30–50% of adults. The FDA SHARE (Support, Help and Resources for Epilepsy) REMS program "
            "mandates mandatory enrolment of prescriber, pharmacy, and patient before dispensing."
        ),
        "action": "Enrol patient in SHARE REMS before prescribing VGB; baseline Goldman perimetry + OCT; repeat q3M.",
        "evidence": "FDA SHARE REMS (2009, updated 2020); Pellock 2012 systematic review.",
    },
    {
        "drug": "Hospital NPO without IV AED continuation",
        "scope": "All KCNT1-DEE patients (EIMFS or NFLE) during any hospital procedure/surgery",
        "mechanism": (
            "Nil per os (NPO) interrupts oral PB, CLB, and KD — all of which are continuous brain "
            "protection mechanisms in EIMFS. Seizure breakthrough during anaesthesia or post-operative "
            "NPO periods can cause status epilepticus. IV PB or IV LEV must be prescribed as KD-equivalent "
            "bridge; KD lipid emulsion IV if available in perioperative setting."
        ),
        "action": "Pre-operative planning: IV PB bridge + IV LEV; alert anaesthesiology to KCNT1 diagnosis; KD team consultation.",
        "evidence": "ILAE Dietary Therapies 2018; EAN Neonatal SE 2019.",
    },
]

# ── Lifecycle Windows (6) ────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal NICU (0–28d)",
        "age_range": "0 – 28 days",
        "focus": "Acute seizure control, molecular diagnosis initiation, POLG exclusion",
        "key_action": "PB loading IV; POLG panel STAT; trio exome sent; ACNS EEG within 24h of NICU admission.",
    },
    {
        "window": "Early infantile — EIMFS peak (1–6M)",
        "age_range": "1 – 6 months",
        "focus": "Migration pattern confirmation, KD initiation, IS component management",
        "key_action": "24h VEEG for migration confirmation; KD 4:1 initiation with dietitian; ACTH if IS present.",
    },
    {
        "window": "Late infantile (6–18M)",
        "age_range": "6 – 18 months",
        "focus": "KD optimisation, developmental assessment, VNS evaluation if DRE ≥3 AEDs",
        "key_action": "Bayley-III q6M; VNS referral if ≥3 AED failures + KD; Goldman perimetry if VGB used.",
    },
    {
        "window": "Early childhood — DRE (18M–5Y)",
        "age_range": "18 months – 5 years",
        "focus": "DRE management, KD continuation, school/therapy planning, SUDEP counselling",
        "key_action": "Annual VEEG + neurodevelopmental profile; SUDEP counselling; bedside SpO2 monitoring plan.",
    },
    {
        "window": "School age (5–12Y)",
        "age_range": "5 – 12 years",
        "focus": "Seizure control optimisation, cognitive/communication, caregiver support",
        "key_action": "Educational placement (SEN); AAC device assessment; annual VEEG; medication adherence review.",
    },
    {
        "window": "Adolescence / Adult — NFLE onset (12Y+)",
        "age_range": "12 years and above",
        "focus": "NFLE-KCNT1 new-onset in familial cases; transition planning; driving safety; genetics",
        "key_action": "NFLE: nocturnal VEEG; CBZ/OXC trial; quinidine with cardiology (QTc) if NFLE. EIMFS: transition planning.",
    },
]

# ── Key Concepts (14) ──────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "KCNT1 / KNa1.1 / Slack / Slo2.2",
     "definition": "Gene (9q34.3) encoding the sodium-activated potassium channel KNa1.1 (also known as Slack or Slo2.2). Expressed in cortical interneurons and layer-5 pyramidal neurons; generates a slow after-hyperpolarisation that dampens burst firing. GOF mutations disrupt this 'brake' mechanism."},
    {"term": "EIMFS (Epilepsy of Infancy with Migrating Focal Seizures)",
     "definition": "OMIM 614959. Severe DEE characterised by the EEG hallmark of migrating focal ictal discharges that shift between hemispheres and lobes. Onset <6 months; profound DD; drug-resistant. KCNT1 GOF accounts for ~20% of EIMFS cases; 15+ other genes causative."},
    {"term": "GOF Variant — KCNT1",
     "definition": "Gain-of-function variants enhance KNa1.1 constitutive channel activity, increasing K+ efflux. Paradoxically pro-epileptic: likely via preferential deactivation of fast-firing GABAergic interneurons, disrupting cortical inhibitory circuits. Most common hotspot: p.Arg474His (RCK domain)."},
    {"term": "Migrating Focal Seizures EEG Signature",
     "definition": "The pathognomonic EEG pattern of EIMFS: sequential focal ictal discharges beginning in one hemisphere then shifting to the contralateral hemisphere or a different lobe within or between seizures. Requires prolonged EEG recording (24h VEEG) to capture the migration phenomenon."},
    {"term": "Quinidine — Negative RCT (EIMFS 2020)",
     "definition": "Quinidine blocks KCNT1 GOF channels in vitro but the phase 2 RCT (Numis et al. 2020, Epilepsia; n=9 EIMFS) showed NO significant seizure reduction vs placebo (p=0.38). NOT recommended for EIMFS. Anecdotal use in NFLE-KCNT1 continues but without RCT support."},
    {"term": "KD 4:1 — Most Effective in EIMFS",
     "definition": "Ketogenic diet (4:1 fat:CHO+protein) has the highest evidence for seizure reduction in EIMFS — 50–80% reduction in multiple case series (Rizzo 2016, Caraballo 2015). Should be initiated after 1st or 2nd AED failure, not deferred to last resort."},
    {"term": "POLG Exclusion before VPA",
     "definition": "POLG (polymerase gamma) mutations cause fatal hepatic failure with VPA. Any infant with DEE must have POLG panel completed before initiating VPA. KCNT1 variants may co-occur with other pathogenic variants including POLG in consanguineous families."},
    {"term": "VGB SHARE REMS (FDA)",
     "definition": "FDA SHARE (Support, Help and Resources for Epilepsy) REMS mandates enrolment of prescriber, pharmacy, and patient before dispensing vigabatrin (VGB). Mandatory Goldman perimetry q3M + OCT q6M. Failure to enrol = dispensing violation."},
    {"term": "SUDEP — KCNT1",
     "definition": "Sudden Unexpected Death in Epilepsy. High risk in EIMFS (severe DRE, nocturnal FBTCS, prone positioning). SUDEP counselling mandatory. Nocturnal SpO2 monitoring; supervised sleep environments; prone positioning avoided. SUDEP rate in EIMFS estimated 3–5× higher than general DEE population."},
    {"term": "West Syndrome (IS) — KCNT1 Overlap",
     "definition": "IS component occurs in ~45% of KCNT1-DEE. Hypsarrhythmia (modified/multifocal in KCNT1) + flexion spasms + developmental regression. ACTH first-line (UKISS); VGB second-line (SHARE REMS). IS on top of ongoing migrating focal seizures requires parallel treatment strategies."},
    {"term": "NFLE-KCNT1 (Autosomal-Dominant Nocturnal Frontal Lobe Epilepsy)",
     "definition": "OMIM 615005. Milder KCNT1 GOF phenotype: nocturnal hypermotor seizures with frontal-onset; dominant inheritance; onset childhood–adulthood. KCNT1 accounts for ~7–13% of familial NFLE. Variable expressivity. CBZ/OXC partially effective; quinidine anecdotally positive in small series."},
    {"term": "Interneuron Loss — GOF Paradox",
     "definition": "KNa1.1 GOF paradoxically causes hyperexcitability despite being a potassium channel (expected to be hyperpolarising). Proposed mechanism: fast-firing GABAergic interneurons depend on KNa1.1 for rapid recovery between bursts; constitutive GOF deactivates interneurons preferentially → disinhibition → cortical hyperexcitability."},
    {"term": "KCNT1 Alliance",
     "definition": "Patient advocacy organisation for KCNT1-related epilepsies. Resources for families: genetic counselling, clinical trial registry, research liaison. Website: kcnt1epilepsy.com. International network of KCNT1 researchers and clinicians."},
    {"term": "ASO / Gene Therapy — KCNT1 (Preclinical)",
     "definition": "Antisense oligonucleotide (ASO) programmes targeting KCNT1 GOF allele-specific silencing are in preclinical development (mouse EIMFS models). AAV9-based gene replacement/silencing feasibility studies under evaluation. No Phase 1 human trial open as of 2026."},
]

# ── Evidence Standards (8) ──────────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE-2022", "title": "ILAE Classification and Definition of Epilepsy Syndromes (2022)", "relevance": "EIMFS and NFLE-KCNT1 syndrome classification; DEE diagnostic framework."},
    {"standard": "NICE-NG217", "title": "Epilepsies: diagnosis and management (NG217, 2022)", "relevance": "AED selection, KD recommendation, SUDEP counselling, genetic testing pathway."},
    {"standard": "FDA-SHARE-REMS-VGB", "title": "FDA SHARE REMS — Vigabatrin (2009, updated 2020)", "relevance": "Mandatory enrolment for VGB; perimetry frequency; pharmacy dispensing requirements."},
    {"standard": "ILAE-Dietary-Therapies-2018", "title": "ILAE Dietary Therapies Consensus (2018)", "relevance": "KD 4:1 initiation protocol, monitoring parameters, contraindications, perioperative bridge."},
    {"standard": "ACMG-AMP-2015", "title": "ACMG/AMP Standards for Variant Interpretation (2015)", "relevance": "KCNT1 GOF variant classification; ClinGen KCNT1 Expert Panel evidence framework."},
    {"standard": "ACNS-EEG-2021", "title": "ACNS Standardised Critical Care EEG Terminology (2021)", "relevance": "EIMFS migrating focal seizure EEG classification; neonatal EEG terminology."},
    {"standard": "EAN-NeonatalSE-2019", "title": "EAN Neonatal Seizure / Status Epilepticus Consensus (2019)", "relevance": "PB loading protocol; IV AED continuation during NPO; perioperative management."},
    {"standard": "Numis-2020-RCT-Quinidine", "title": "Numis et al. 2020 Epilepsia — Quinidine RCT in EIMFS (Phase 2)", "relevance": "Primary evidence basis for NOT using quinidine in EIMFS; null result; QTc safety data."},
]

# ── Thresholds (10) ────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Onset < 6M → KCNT1/EIMFS molecular diagnosis",
     "action": "Request trio exome + KCNT1 panel STAT if migrating focal EEG pattern in infants <6 months."},
    {"threshold": "PB TDM 20–40 µg/mL",
     "action": "Below 15: increase dose. Above 45: reduce dose. Check at steady state (5 half-lives)."},
    {"threshold": "KD BHB 2–4 mmol/L",
     "action": "Below 2: insufficient ketosis — review diet composition, carb sources. Above 5: assess for acidosis."},
    {"threshold": "VGB Goldman VFD q3M (SHARE REMS)",
     "action": "Any new VFD on perimetry: reduce VGB dose; if progressive, discontinue. OCT confirmation q6M."},
    {"threshold": "ACTH day-14 EEG criterion",
     "action": "IS: check EEG at day 14 — if hypsarrhythmia persists, consider ACTH non-responder; switch strategy."},
    {"threshold": "2 AED failures → KD initiation",
     "action": "Do not defer KD to last resort. After 2 AEDs without ≥50% reduction: initiate KD 4:1 with dietitian."},
    {"threshold": "3 AED failures → VNS evaluation",
     "action": "After 3 AED failures + KD: refer for VNS implantation (20–30% responder rate in EIMFS)."},
    {"threshold": "POLG panel before VPA",
     "action": "POLG result must be negative before initiating VPA. Pending result = do not start VPA."},
    {"threshold": "Seizure-free 2Y → consider AED taper",
     "action": "Only for NFLE-KCNT1 (milder). EIMFS: do not taper — recurrence near-universal."},
    {"threshold": "QTc > 500 ms — stop quinidine",
     "action": "If quinidine trialled in NFLE-KCNT1: withhold and cardiology review if QTc exceeds 500 ms."},
]

# ── References (6) ─────────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "Barcia 2012 NatGenet",
     "title": "De novo gain-of-function KCNT1 channel mutations cause malignant migrating partial seizures of infancy",
     "relevance": "Discovery paper establishing KCNT1 GOF as the major cause of EIMFS; p.Arg474His first described."},
    {"ref": "Milligan 2014 AnnNeurol",
     "title": "KCNT1 gain of function in two epilepsy phenotypes is reversed by quinidine",
     "relevance": "First case report of quinidine benefit in KCNT1 NFLE; generated the quinidine hypothesis."},
    {"ref": "Numis 2020 Epilepsia",
     "title": "Efficacy and tolerability of quinidine in KCNT1-associated epilepsy: a phase 2 RCT",
     "relevance": "RCT (n=9 EIMFS): quinidine showed NO significant seizure reduction vs placebo — defines current standard of NOT using quinidine in EIMFS."},
    {"ref": "McTague 2013 NatMed",
     "title": "Migrating partial seizures of infancy: expansion of the electroclinical, radiological and pathological disease spectrum",
     "relevance": "Natural history of EIMFS; EEG migration criteria; multi-gene causation landscape."},
    {"ref": "Rizzo 2016 EurJPaediatrNeurol",
     "title": "Efficacy of ketogenic diet in children with KCNT1-related epilepsy",
     "relevance": "Largest case series reporting KD efficacy in KCNT1-EIMFS: 50–80% seizure reduction — basis for Level-B KD recommendation."},
    {"ref": "Lim 2016 EurJHumGenet",
     "title": "KCNT1 mutations in patients with autosomal dominant nocturnal frontal lobe epilepsy",
     "relevance": "Established KCNT1 as cause of NFLE-AD; genotype-phenotype spectrum from EIMFS to NFLE."},
]

# ── Patient Generator ──────────────────────────────────────────────────────────
_CATEGORY_POOL = [
    ("De-novo-KCNT1-GOF-EIMFS-severe", "GOF-severe", 20),
    ("De-novo-KCNT1-GOF-moderate", "GOF-moderate", 9),
    ("Familial-KCNT1-GOF-NFLE-AD", "GOF-familial", 6),
    ("De-novo-KCNT1-splice-structural", "GOF-splice", 3),
    ("Clinical-KCNT1-negative-EIMFS-phenocopy", "KCNT1-negative", 3),
]

_TREATMENTS_POOL = [
    "PB+CLB", "PB+CLB+KD", "PB+LEV", "CLB+KD", "PB+CLB+LEV+KD",
    "KD+CLB", "PB+VGB+ACTH", "CLB+VGB", "PB+CLB+VPA+KD",
]

_PHASES = ["Neonatal-NICU", "Early-infantile", "Late-infantile", "Early-childhood", "School-age", "Adolescent"]
_CONTROL = ["drug-resistant", "drug-resistant", "drug-resistant", "partial-control", "partial-control", "seizure-free"]
_SEXES = ["M", "F"]


def _gen_patients():
    random.seed(SEED)
    patients = []
    pid = 1
    for cat, fclass, n in _CATEGORY_POOL:
        for _ in range(n):
            onset_days = random.randint(2, 90) if "EIMFS-severe" in cat else (
                random.randint(30, 180) if "moderate" in cat else random.randint(365, 1460)
            )
            age_months = random.randint(6, 72) if "EIMFS" in cat or "moderate" in cat else random.randint(36, 240)
            sex = random.choice(_SEXES)
            phase = random.choice(_PHASES[:4]) if "EIMFS" in cat else random.choice(_PHASES)
            control = random.choice(_CONTROL)
            tx = random.choice(_TREATMENTS_POOL)
            pb_level = round(random.uniform(18, 42), 1) if "PB" in tx else None
            na = round(random.uniform(128, 142), 1)
            migration = "Yes" if "EIMFS" in cat else random.choice(["Yes", "No"])
            kd_active = "KD" in tx
            vns = random.random() < 0.25 and "EIMFS" in cat
            patients.append({
                "id": f"KN-{pid:03d}",
                "age_months": age_months,
                "sex": sex,
                "onset_age_days": onset_days,
                "category": cat,
                "functional_class": fclass,
                "disease_phase": phase,
                "current_treatment": tx,
                "seizure_control": control,
                "pb_level_ugml": pb_level,
                "na_mmoll": na,
                "migration_pattern": migration,
                "kd_active": kd_active,
                "vns_implanted": vns,
            })
            pid += 1
    random.shuffle(patients)
    return patients


# ── Public API Functions ──────────────────────────────────────────────────────
def get_overview():
    pts = _gen_patients()
    gof_severe = sum(1 for p in pts if p["functional_class"] == "GOF-severe")
    gof_mod = sum(1 for p in pts if p["functional_class"] == "GOF-moderate")
    gof_fam = sum(1 for p in pts if p["functional_class"] == "GOF-familial")
    dre = sum(1 for p in pts if p["seizure_control"] == "drug-resistant")
    sz_free = sum(1 for p in pts if p["seizure_control"] == "seizure-free")
    kd_on = sum(1 for p in pts if p["kd_active"])
    vns = sum(1 for p in pts if p["vns_implanted"])
    migration = sum(1 for p in pts if p["migration_pattern"] == "Yes")

    return {
        "syndrome": "KCNT1 Encephalopathy (EIMFS / NFLE-KCNT1-DEE)",
        "gene": "KCNT1",
        "chromosome": "9q34.3",
        "protein": "KNa1.1 / Slack / Slo2.2 (Sodium-Activated Potassium Channel)",
        "inheritance": "De novo (EIMFS) / Autosomal Dominant (NFLE-KCNT1)",
        "omim_eimfs": "614959",
        "omim_nfle": "615005",
        "eeg_hallmark": "Migrating multifocal focal ictal discharges shifting between hemispheres (EIMFS pathognomonic)",
        "key_biomarker": "Migrating EEG pattern on 24h VEEG + KCNT1 GOF on trio exome",
        "n_patients": 41,
        "kpis": {
            "gof_severe_pct": round(gof_severe / 41 * 100),
            "gof_moderate_pct": round(gof_mod / 41 * 100),
            "gof_familial_pct": round(gof_fam / 41 * 100),
            "dre_pct": round(dre / 41 * 100),
            "seizure_free_pct": round(sz_free / 41 * 100),
            "kd_on_pct": round(kd_on / 41 * 100),
            "vns_implanted_pct": round(vns / 41 * 100),
            "migration_positive_pct": round(migration / 41 * 100),
        },
        "clinical_alerts": [
            "🚨 QUINIDINE NOT RECOMMENDED IN EIMFS — phase 2 RCT (Numis 2020) showed NO significant seizure reduction vs placebo. Do not prescribe quinidine for KCNT1-EIMFS.",
            "⚠️ POLG EXCLUSION MANDATORY before VPA — fatal mitochondrial hepatopathy in POLG carriers. Order POLG panel before any VPA initiation.",
            "⚠️ VGB REQUIRES FDA SHARE REMS enrolment (USA) — Goldman perimetry q3M mandatory; irreversible VFD risk.",
            "⚡ KD 4:1 FIRST-LINE at 2nd AED failure — do NOT defer to last resort; 50–80% seizure reduction in EIMFS case series.",
            "🚨 NPO/SURGERY: IV PB bridge + IV LEV mandatory; KD lipid emulsion if available. Alert anaesthesiology of KCNT1 diagnosis.",
            "⚡ EEG MIGRATION PATTERN: requires 24h VEEG — routine 20-min EEG may miss migratory phenomenon. Essential for diagnosis.",
        ],
        "etiologies": [{"etiology": e["etiology"], "pct": e["pct"], "n": e["n"]} for e in ETIOLOGY_CATALOG],
        "seizure_type_prevalence": {s["type"]: s["prevalence_pct"] for s in SEIZURE_TYPES},
        "trigger_seizure_rates": {t["trigger"]: t["rate_pct"] for t in TRIGGERS},
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "key_aha": (
            "KCNT1-EIMFS: the migrating focal EEG pattern is the clinical fingerprint — request 24h VEEG. "
            "Ketogenic diet 4:1 has the highest seizure reduction evidence in EIMFS (50–80% in case series). "
            "Do NOT give quinidine (negative RCT). Exclude POLG before VPA. Initiate KD at 2nd AED failure."
        ),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_breakdown():
    pts = _gen_patients()
    return {
        "patients": pts,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": AED_MONITORING,
        "n_patients": 41,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_definitions():
    return {
        "absolute_contraindications": ABSOLUTE_CI,
        "thresholds": THRESHOLDS,
        "concepts": CONCEPTS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
