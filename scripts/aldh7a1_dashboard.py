"""
ALDH7A1 Pyridoxine-Dependent Epilepsy (PDE-ALDH7A1 / Antiquitin Deficiency)
=============================================================================
41-patient cohort · ALDH7A1 (5q23.2) · Antiquitin / Alpha-aminoadipic semialdehyde dehydrogenase
PDE-ALDH7A1: the prototype treatable metabolic epilepsy — autosomal recessive LOF variants in
ALDH7A1 block lysine catabolism, causing P6C accumulation that chemically inactivates PLP (the
active form of pyridoxine / Vitamin B6), leading to PLP-dependent enzyme failure, GABA/glutamate
imbalance, and neonatal/infantile seizures that RESPOND DRAMATICALLY to IV pyridoxine within 1 hour.

ALDH7A1 BIOLOGY:
ALDH7A1 (Aldehyde Dehydrogenase 7 Family Member A1 / Antiquitin, 5q23.2) encodes the enzyme
alpha-aminoadipic semialdehyde (AASA) dehydrogenase, which catalyses the conversion of
alpha-aminoadipic semialdehyde (AASA) to alpha-aminoadipic acid in the lysine catabolism
saccharopine pathway. In ALDH7A1 deficiency:
  AASA accumulates in urine, plasma, CSF (diagnostic biomarker).
  Its cyclic form Δ1-piperideine-6-carboxylate (P6C) also accumulates.
  P6C condenses irreversibly with pyridoxal 5'-phosphate (PLP, active B6) via Knoevenagel
  condensation → depletes functional PLP → failure of >50 PLP-dependent enzymes including
  glutamic acid decarboxylase (GAD, GABA synthesis) and AADC (dopamine/serotonin synthesis) →
  GABA-deficient, seizure-prone brain + neurotransmitter deficiency.

PRECISION MECHANISM — WHY PYRIDOXINE WORKS:
  Supplemental high-dose pyridoxine overrides the P6C-mediated PLP inactivation by providing
  excess substrate that sustains PLP-dependent enzyme activity despite ongoing P6C condensation.
  Additionally, pyridoxine/PLP directly restores GAD activity → restored GABA → seizure control.
  EEG RESPONSE WITHIN 1 HOUR of IV pyridoxine 30 mg/kg = diagnostic AND therapeutic.

TRIPLE THERAPY (van Karnebeek 2012 — most effective regimen):
  ① Pyridoxine 15-30 mg/kg/day (oral maintenance after IV diagnostic trial)
  ② Folinic acid (leucovorin) 3-5 mg/kg/day — addresses secondary CSF folate deficiency
     (P6C-PLP condensation also depletes 5-methyltetrahydrofolate → secondary cerebral folate
     deficiency → neurotransmitter deficiency beyond GABA)
  ③ Lysine-restricted diet ≤60 mg/kg/day total dietary lysine — reduces substrate flux through
     the blocked ALDH7A1 enzyme → less AASA/P6C accumulation → less PLP inactivation →
     improved developmental outcomes even when seizures already controlled by pyridoxine alone.
  + L-arginine 300-400 mg/kg/day: competes with lysine for LAT1 BBB transporter → reduces CNS
    lysine delivery → less AASA production in the brain.

INHERITANCE: Autosomal recessive — biallelic LOF variants required. Heterozygous carriers
(parents) are phenotypically normal. Each pregnancy: 25% affected, 50% carrier, 25% unaffected.
Estimated prevalence 1:64,000 (Netherlands registry) to 1:400,000 globally — likely underestimated
due to late/missed diagnosis (many neonates treated empirically with pyridoxine but never tested).

KEY SAFETY PEARLS:
• EVERY unexplained neonatal seizure → pyridoxine trial 30 mg/kg IV MANDATORY (NICE NG217, EAN 2019)
  — administer under cardiac/EEG monitoring (rare apnoea risk at IV doses)
• Pyridoxine high-dose sensory neuropathy: adults >500 mg/day chronically — neurophysiology screen
• AASA urine is the gold-standard diagnostic — perform BEFORE commencing empiric pyridoxine where
  possible, as pyridoxine treatment normalises AASA within days (Mills 2006)
• P6C-PLP condensation IRREVERSIBLE: once PLP inactivated, only exogenous pyridoxine rescues it;
  stop pyridoxine → seizure recurrence within hours-days (Pearl 2022)
• Triple therapy superior to pyridoxine alone for developmental outcomes (van Karnebeek 2012):
  IQ +15-20 points in lysine-restricted vs pyridoxine-only cohorts
• Phenocopy: PNPO (pyridox(am)ine 5'-phosphate oxidase) deficiency mimics PDE but requires
  PLP (not pyridoxine) — always trial pyridoxine THEN PLP if pyridoxine fails
"""

import random
from datetime import datetime

SEED = 9186  # dashboard 186
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ───────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": (
            "ALDH7A1 biallelic LOF — classic neonatal PDE "
            "(homozygous or compound heterozygous null alleles)"
        ),
        "n": 18, "pct": 44,
        "category": "ALDH7A1-biallelic-LOF-classic-neonatal",
        "functional_class": "LOF-classic-neonatal",
        "mechanism": (
            "Most prevalent class (~44%): biallelic loss-of-function ALDH7A1 variants (homozygous "
            "truncating frameshift/nonsense, or compound heterozygous null alleles). Complete or "
            "near-complete abolition of antiquitin AASA-dehydrogenase activity → maximal AASA/P6C "
            "accumulation → maximal PLP inactivation → profound GABA/neurotransmitter deficiency "
            "from birth. Classic neonatal onset: seizures within hours-to-days of life, "
            "often burst-suppression on EEG. AASA urine markedly elevated (>10 µmol/mmol "
            "creatinine). Pyridoxine response: dramatic — 80% seizure cessation within 1 hour "
            "of IV B6, with EEG normalisation. Requires lifelong pyridoxine; stopping → acute "
            "seizure recurrence within 24 hours. ACMG pathogenicity: PVS1 + PS2 + PM2 → "
            "Pathogenic. Most benefit from triple therapy (pyridoxine + folinic acid + lysine "
            "restriction) for developmental outcomes."
        ),
        "eeg_signature": (
            "Classic neonatal burst-suppression (BS): high-voltage bursts of mixed theta/delta "
            "activity with superimposed sharp waves, alternating with periods of electrocerebral "
            "suppression. BS pattern RESOLVES within 1-4 hours of IV pyridoxine administration — "
            "this time-locked EEG response is pathognomonic for PDE. Inter-ictal: multifocal "
            "sharp waves, generalised spike-wave after treatment commencement, which gradually "
            "normalise with adequate pyridoxine maintenance dosing. Ictal: bilateral clonic or "
            "tonic-clonic morphology on EEG correlating with clinical neonatal seizures. "
            "Background: generalised voltage suppression between bursts in untreated state; "
            "normalises to age-appropriate pattern within weeks of triple therapy."
        ),
        "clinical_note": (
            "Diagnostic algorithm: (1) blood/urine for AASA before pyridoxine if possible — "
            "AASA normalises within 24-48 hours of treatment. (2) IV pyridoxine 30 mg/kg over "
            "30 min under EEG + cardiac monitoring — document EEG response. (3) If responder: "
            "maintenance 15-30 mg/kg/day oral + folinic acid + lysine restriction. (4) "
            "Molecular confirmation: ALDH7A1 gene sequencing + CNV analysis. (5) Parental "
            "carrier testing + reproductive counselling (25% recurrence per pregnancy)."
        ),
    },
    {
        "etiology": (
            "ALDH7A1 biallelic missense — partial enzyme activity, late-onset or atypical PDE"
        ),
        "n": 12, "pct": 29,
        "category": "ALDH7A1-biallelic-missense-partial-activity",
        "functional_class": "LOF-partial-atypical",
        "mechanism": (
            "~29%: biallelic missense ALDH7A1 variants retaining partial (5-30%) antiquitin "
            "enzymatic activity. Residual activity reduces P6C accumulation rate — seizure onset "
            "later (infantile 3-18 months, or even to 3 years). EEG may not show classic "
            "burst-suppression. AASA urine still elevated (4-8 µmol/mmol creatinine) but less "
            "dramatically than null allele class. Pyridoxine response: partial to complete — "
            "75% respond, but seizure cessation may take days rather than hours. Diagnosis often "
            "delayed because atypical presentation doesn't trigger immediate PDE workup. "
            "Common pathogenic missense: E399Q (mild), A149E (moderate), T301N (moderate-severe). "
            "Residual enzyme activity quantified by Escherichia coli expression system or patient "
            "fibroblast assay — useful to guide precision dosing."
        ),
        "eeg_signature": (
            "Infantile onset: hypsarrhythmia variant (modified hypsarrhythmia) in ~30% — especially "
            "when onset overlaps West syndrome age window (3-12 months). Partial responders to "
            "pyridoxine: residual focal spikes or centrotemporal epileptiform activity persisting "
            "after pyridoxine initiation, requiring addition of folinic acid/lysine restriction "
            "for full EEG normalisation. Late-onset cases (>12 months): focal temporal epileptiform "
            "discharges indistinguishable from focal epilepsy — PDE diagnosis requires high "
            "clinical suspicion and AASA urine testing."
        ),
        "clinical_note": (
            "Clinical trap: atypical PDE (late-onset, partial response) mimics common focal "
            "or generalised epilepsy syndromes. All infants <3 years with unexplained epilepsy "
            "should have AASA urine screening. Pyridoxine trial (oral 30 mg/kg/day for 4-6 weeks "
            "if IV not feasible) can be diagnostic. Response: reduction in seizure frequency "
            "≥50% within 4 weeks of adequate B6 dosing = presumptive PDE pending AASA confirmation."
        ),
    },
    {
        "etiology": "ALDH7A1 biallelic splice-site variants",
        "n": 5, "pct": 12,
        "category": "ALDH7A1-biallelic-splice-site",
        "functional_class": "LOF-splice",
        "mechanism": (
            "~12%: biallelic splice-site variants (canonical ±1/±2 or deep intronic cryptic "
            "splice alterations). Phenotype depends on exon skipping impact: variants causing "
            "in-frame exon skipping → partial LOF (milder, late-onset) vs. frameshift/NMD → "
            "complete LOF (classic neonatal). RNA analysis from fibroblasts essential to "
            "characterise splice impact. NGS-based copy number analysis simultaneously. AASA "
            "elevated in all confirmed cases. Pyridoxine response correlated with residual "
            "enzyme activity from splice outcome."
        ),
        "eeg_signature": (
            "Variable — mirrors the functional class of the splice outcome. In-frame skipping: "
            "often atypical infantile EEG pattern. Frameshift/NMD outcome: classic neonatal "
            "burst-suppression. Splice variants require mRNA expression studies (RT-PCR from "
            "fibroblasts or blood RNA) to assign pathogenicity class."
        ),
        "clinical_note": (
            "Variant curation challenge: intronic variants remote from canonical splice sites "
            "require SpliceAI score >0.5 or MinSplice/CADD-Splice modelling for pathogenicity "
            "support. Both parents as obligate heterozygous carriers confirms biallelic status "
            "in index case. RNA studies (ACMG BS3 functional evidence) elevate VUS to LP/P."
        ),
    },
    {
        "etiology": (
            "ALDH7A1 biallelic CNV — deletion/duplication involving one or both alleles"
        ),
        "n": 3, "pct": 7,
        "category": "ALDH7A1-biallelic-CNV",
        "functional_class": "LOF-CNV",
        "mechanism": (
            "~7%: chromosomal copy number variants (CNVs) involving ALDH7A1 at 5q23.2. "
            "Typically single-exon to whole-gene deletions causing haploinsufficiency — when "
            "combined with a second-allele sequence variant (compound het with CNV), results "
            "in biallelic LOF. Rarely: homozygous deletion. Array CGH or SNP array required "
            "to detect CNVs missed by standard NGS. Phenotype: severe neonatal PDE (classic "
            "burst-suppression). CNV-PDE patients less likely to have detectable AASA "
            "in early post-treatment samples — gene-based testing primary diagnostic tool."
        ),
        "eeg_signature": (
            "Classic burst-suppression neonatal pattern in CNV-LOF cases. Full EEG normalisation "
            "with pyridoxine, consistent with complete enzyme deficiency. Some patients retain "
            "persistent theta slowing for weeks after seizure cessation — reflects ongoing "
            "neurodevelopmental impact of early PLP deficiency."
        ),
        "clinical_note": (
            "When ALDH7A1 sequence analysis negative but clinical PDE picture compelling: "
            "perform ALDH7A1 copy number analysis (MLPA or chromosomal microarray). "
            "Additionally, consider mosaicism testing (>20% VAF threshold for pathogenicity)."
        ),
    },
    {
        "etiology": (
            "Clinical PDE — ALDH7A1 negative phenocopy "
            "(PNPO / PROSC / PLPBP / other PLP-metabolism defects)"
        ),
        "n": 3, "pct": 7,
        "category": "clinical-PDE-ALDH7A1-negative-phenocopy",
        "functional_class": "PDE-phenocopy",
        "mechanism": (
            "~7%: pyridoxine-responsive neonatal seizures in patients where ALDH7A1 sequencing "
            "and copy number are negative. Differential: (1) PNPO deficiency — "
            "pyridox(am)ine 5'-phosphate oxidase, the enzyme that converts dietary pyridoxine/PNP "
            "to active PLP; PNPO-DEE responds BETTER to PLP than pyridoxine. (2) PROSC/PLPBP "
            "deficiency (pyridoxal 5'-phosphate homeostasis protein). (3) Hyperprolinaemia type II. "
            "(4) Antiquitin activity variant not detectable by current methods. Key diagnostic: "
            "if pyridoxine response partial → trial PLP (pyridoxal 5-phosphate) 60 mg/kg/day → "
            "full response = PNPO. CSF neurotransmitters (PLP, 5-MTHF, neurotransmitter metabolites) "
            "differentiate PDE subtypes."
        ),
        "eeg_signature": (
            "PNPO phenocopy: often more severe EEG suppression between bursts than classic PDE-ALDH7A1; "
            "may not respond to pyridoxine but shows EEG response to PLP within 2-4 hours. "
            "PROSC phenocopy: similar to ALDH7A1-PDE on EEG. Differentiation requires "
            "biochemical profiling (CSF PLP, plasma PLP/PMP/PA) and molecular testing."
        ),
        "clinical_note": (
            "Algorithm for ALDH7A1-negative PDE phenocopy: (1) continue triple therapy empirically "
            "— pyridoxine + folinic acid. (2) Add PLP trial: 60 mg/kg/day for 4 weeks — "
            "superior response to PLP confirms PNPO. (3) WES/WGS panel including PNPO, PROSC, "
            "PLPBP, ALDH4A1, PHGDH genes. (4) CSF neurotransmitter profile (5-MTHF, HVA, HIAA, "
            "PLP) — abnormal in PNPO, PROSC. (5) Enrol in international PDE consortium registry."
        ),
    },
]

# ── Seizure Types (4) ─────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Neonatal multifocal clonic seizures (EIMFS-like / classic PDE)",
        "prevalence_pct": 90,
        "onset_age": "Hours to 7 days of life (prenatal intrauterine seizures reported in 30%)",
        "eeg_correlate": (
            "Burst-suppression (BS): high-voltage delta-theta bursts 3-5 µV/cm alternating with "
            "electrocerebral quiescence (0.5-1 µV/cm); burst duration 2-20 sec, suppression "
            "period 5-60 sec. Ictal: bilateral clonic/tonic activity superimposed on burst "
            "onset. BS RESOLVES within 1-4 hours of IV pyridoxine 30 mg/kg — this time-locked "
            "EEG normalisation is the strongest clinical diagnostic confirmation of PDE-ALDH7A1."
        ),
        "clinical_tip": (
            "Any neonate with burst-suppression EEG unresponsive to phenobarbitone — administer "
            "IV pyridoxine 30 mg/kg (over 30 min) under continuous EEG + cardiac monitoring. "
            "Document EEG response in real-time. Stop seizures and burst-suppression within "
            "1 hour = presumptive PDE. Collect urine/blood AASA BEFORE pyridoxine if clinically "
            "feasible (AASA normalises within 24-48 h of treatment)."
        ),
    },
    {
        "type": "Infantile spasms / West syndrome component",
        "prevalence_pct": 32,
        "onset_age": "3-12 months (particularly in partial-activity biallelic missense variants)",
        "eeg_correlate": (
            "Modified hypsarrhythmia: high-amplitude chaotic interictal pattern with multifocal "
            "spikes and background disorganisation, but less chaotic than idiopathic West syndrome. "
            "Modified pattern reflects ongoing PLP deficiency from under-treated ALDH7A1 defect. "
            "After pyridoxine: hypsarrhythmia resolves in 50-70% without ACTH; remainder require "
            "ACTH + pyridoxine combination (ACTH 20-30 IU/day + pyridoxine 30 mg/kg/day)."
        ),
        "clinical_tip": (
            "When West syndrome does not respond to ACTH/VGB within 4 weeks: screen AASA urine "
            "and pipecolic acid. Late-presentation PDE with West syndrome component may be "
            "partial-activity missense variant — missed neonatal diagnosis. Pyridoxine adds "
            "to ACTH efficacy in PDE-West overlap cases."
        ),
    },
    {
        "type": "Generalised tonic-clonic seizures (breakthrough / subtherapeutic B6)",
        "prevalence_pct": 44,
        "onset_age": "Any age — breakthrough on inadequate pyridoxine dosing or during illness",
        "eeg_correlate": (
            "3-5 Hz generalised spike-wave (GSW) and polyspike-wave during GTCS. Background "
            "generalised slowing proportional to duration of breakthrough seizure event. "
            "Interictally: persistent 2-3 Hz delta slowing in patients with subtherapeutic "
            "pyridoxine — indicates ongoing PLP deficiency. Breakthrough seizures are the "
            "most common cause of ER presentations in established PDE patients — invariably "
            "from missed B6 dose, dose not adjusted for weight gain, or febrile illness."
        ),
        "clinical_tip": (
            "PDE patient presenting with breakthrough seizure: (1) check last B6 dose and "
            "current weight — dose often subtherapeutic as child grows. (2) Administer rescue "
            "dose pyridoxine IV 30 mg/kg or PO 50 mg/kg stat. (3) Check plasma PLP level: "
            "target >50 nmol/L (ideally 80-150 nmol/L). (4) Ensure lysine-restricted diet "
            "adherent — febrile illness increases lysine catabolism → more AASA/P6C → "
            "more PLP inactivation → more seizure risk."
        ),
    },
    {
        "type": "Focal seizures (late-onset atypical PDE — often misdiagnosed)",
        "prevalence_pct": 20,
        "onset_age": "3 months to 3 years (atypical/late-onset partial-activity variants)",
        "eeg_correlate": (
            "Focal epileptiform discharges: temporal or frontotemporal spike-wave activity "
            "indistinguishable from structural focal epilepsy on EEG alone. No burst-suppression. "
            "Background: mild to moderate generalised slowing reflecting subtle PLP deficiency "
            "affecting neuronal metabolism globally despite focal EEG semiology. MRI: may show "
            "periventricular T2 signal changes (gliosis from early PLP-deficient state), "
            "rarely normal. Key: AASA urine screening in any infant with unexplained focal "
            "epilepsy — even if EEG does not suggest metabolic aetiology."
        ),
        "clinical_tip": (
            "Late-onset PDE (>3 months) is frequently misdiagnosed as focal epilepsy of unknown "
            "cause. Clinical clues: (1) epilepsy refractory to conventional AEDs but "
            "AASA not yet tested; (2) intellectual disability disproportionate to seizure "
            "burden; (3) positive family history (AR — sibling with unexplained ID or "
            "epilepsy). AASA urine is cheap, reliable, and diagnostic — test ALL unexplained "
            "epilepsies in children <3 years."
        ),
    },
]

# ── Seizure Triggers (8) ──────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Subtherapeutic pyridoxine dose (weight not updated)",
        "rate_pct": 92,
        "note": (
            "Most common cause of breakthrough seizures. Pyridoxine dosed in mg/kg — as child "
            "grows, dose must be recalculated at every clinic visit. Target: 15-30 mg/kg/day "
            "oral maintenance. PLP plasma level target 80-150 nmol/L. Under-dosing "
            "= ongoing P6C-PLP inactivation → breakthrough within hours-days."
        ),
    },
    {
        "trigger": "Missed pyridoxine dose / poor adherence",
        "rate_pct": 78,
        "note": (
            "Even 1-2 missed doses → breakthrough seizures within 24-48 hours in complete LOF "
            "patients (half-life of exogenous PLP in CNS is short). Adherence must be monitored "
            "at every visit. Electronic adherence tracking (smart pill dispensers) recommended "
            "for adolescents/adults with compliance history. Plasma PLP is adherence proxy."
        ),
    },
    {
        "trigger": "Febrile illness / infection",
        "rate_pct": 72,
        "note": (
            "Fever dramatically increases protein catabolism → increased lysine flux → more "
            "AASA/P6C production → more PLP inactivation → acute seizure risk. Sick-day "
            "protocol: increase pyridoxine to 30-40 mg/kg/day during illness. Parents must "
            "have emergency pyridoxine prescription at home. ER doctors: administer IV "
            "pyridoxine in febrile PDE patient with seizure."
        ),
    },
    {
        "trigger": "High dietary lysine intake",
        "rate_pct": 55,
        "note": (
            "Excess dietary lysine → increased ALDH7A1 substrate flux → more AASA/P6C production "
            "even at steady state. Target: total dietary lysine ≤60 mg/kg/day on lysine-restricted "
            "diet. High-lysine foods: meat, fish, legumes, dairy — all must be carefully portioned "
            "on specialised metabolic diet. Metabolic dietitian input mandatory for PDE-triple-therapy."
        ),
    },
    {
        "trigger": "Rapid growth phase (infancy/puberty)",
        "rate_pct": 42,
        "note": (
            "Accelerated growth → increased anabolic protein turnover → increased endogenous "
            "lysine catabolism → more AASA/P6C despite dietary lysine restriction. Pyridoxine "
            "dose must be proactively increased during growth spurts (infancy 0-12 months, "
            "puberty 10-14 years). Monthly weight checks in infancy + dose adjustment essential."
        ),
    },
    {
        "trigger": "Dietary non-adherence (lysine restriction abandoned)",
        "rate_pct": 35,
        "note": (
            "In adolescent and adult patients, lysine restriction compliance typically declines. "
            "Subtherapeutic diet → worsening neuropsychological functioning even without "
            "clinical seizures (subclinical PLP deficiency). Psychoeducation, dietitian "
            "monitoring, and patient-reported dietary logs are key for long-term adherence."
        ),
    },
    {
        "trigger": "Folate depletion",
        "rate_pct": 22,
        "note": (
            "Secondary cerebral folate deficiency is an under-recognised co-morbidity of PDE. "
            "P6C-PLP condensation also depletes 5-methyltetrahydrofolate (5-MTHF) via folate "
            "cycle impairment → impaired one-carbon metabolism → secondary neurotransmitter "
            "deficit. Monitor serum folate + CSF 5-MTHF annually. Folinic acid 3-5 mg/kg/day "
            "is the standard adjunct; folic acid is NOT equivalent — folinic acid bypasses "
            "the dihydrofolate reductase step."
        ),
    },
    {
        "trigger": "AED polypharmacy — enzyme inducers reducing B6 efficacy",
        "rate_pct": 18,
        "note": (
            "CYP enzyme-inducing AEDs (phenobarbitone, phenytoin, carbamazepine, oxcarbazepine) "
            "accelerate pyridoxine metabolism and increase renal excretion → lower steady-state "
            "PLP. Avoid enzyme inducers in PDE if possible. If unavoidable: increase pyridoxine "
            "dose by 20-30% and monitor plasma PLP monthly. Phenobarbitone is commonly used "
            "in neonates before PDE diagnosis — inform all ER teams of interaction."
        ),
    },
]

# ── Treatments (8) ────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Pyridoxine (Vitamin B6) — IV diagnostic + oral maintenance",
        "evidence": "Level A — First-Line, MANDATORY trial in unexplained neonatal seizures",
        "indication": (
            "DIAGNOSTIC TRIAL: IV pyridoxine 30 mg/kg over 30 min in any unexplained neonatal "
            "seizure — EEG response within 1 hour = presumptive PDE. MAINTENANCE: 15-30 mg/kg/day "
            "oral once confirmed. Lifelong — stopping → seizure recurrence within 24-48 hours."
        ),
        "dose": (
            "IV trial: 30 mg/kg (max 500 mg) over 30 min — under EEG + cardiac monitoring. "
            "Oral maintenance: 15-30 mg/kg/day in 2-3 divided doses. Adult: 200-300 mg/day. "
            "Sick-day dose: increase to 30-40 mg/kg/day during febrile illness."
        ),
        "moa": (
            "Provides excess PLP substrate that overcomes P6C-mediated PLP inactivation → "
            "restores PLP-dependent enzyme activity (GAD, AADC, kynureninase) → normalises "
            "GABA synthesis → seizure control. Does NOT reduce P6C/AASA accumulation "
            "(substrate still produced) — hence combination with lysine restriction is superior."
        ),
        "efficacy": (
            "IV trial: 80-90% seizure-free within 1 hour in biallelic LOF-classic-neonatal. "
            "Partial-activity variants: 70-75% respond within days. AASA urine normalises "
            "within 24-48 h but rebounds if pyridoxine stopped. Long-term: 65-75% seizure-free "
            "on pyridoxine monotherapy; triple therapy improves developmental outcome significantly."
        ),
        "safety": (
            "IV bolus: risk of apnoea (rare, 1-2%) — have bag-valve mask, epinephrine at bedside. "
            "Oral chronic: generally safe at therapeutic doses ≤200 mg/day in adults. "
            "HIGH DOSES (>200-500 mg/day in adults, >15 mg/kg/day for years): sensory "
            "neuropathy (subacute reversible) — monitor with clinical exam + NCS annually. "
            "Rarely: photosensitivity, dermatitis."
        ),
        "monitoring": (
            "Plasma PLP: target 80-150 nmol/L — q3M initially, then q6M. AASA urine q6M "
            "to verify metabolic control. Weight-based dose recalculation at every visit. "
            "NCS annually in adults for peripheral neuropathy (high-dose monitoring)."
        ),
        "contraindications": (
            "No absolute contraindications. Relative: caution with high IV doses in neonates "
            "with haemodynamic instability (apnoea risk — administer slowly over 30 min). "
            "Peripheral neuropathy monitoring mandatory at doses >200 mg/day adults."
        ),
    },
    {
        "drug": "Folinic acid (Leucovorin) — secondary cerebral folate deficiency adjunct",
        "evidence": "Level B — Adjunct, recommended in all confirmed PDE-ALDH7A1",
        "indication": (
            "Addresses secondary CSF folate depletion caused by P6C-mediated impairment of "
            "folate metabolism. Standard adjunct alongside pyridoxine in confirmed PDE. "
            "Improves developmental and neurotransmitter outcomes beyond seizure control."
        ),
        "dose": (
            "3-5 mg/kg/day oral in 2 divided doses. Folinic acid (not folic acid — folinic "
            "bypasses DHFR step). CSF 5-MTHF target: >40 nmol/L. If CSF 5-MTHF very low "
            "(<10 nmol/L): consider intrathecal folinic acid in specialist centres."
        ),
        "moa": (
            "Folinic acid (5-formyltetrahydrofolate) bypasses the DHFR reduction step — "
            "directly replenishes 5-MTHF pool → restores one-carbon metabolism → normalises "
            "monoamine neurotransmitter synthesis (dopamine, serotonin) and methylation "
            "reactions → improved cognitive and behavioural outcomes."
        ),
        "efficacy": (
            "Seizure-specific efficacy modest (pyridoxine is primary anti-seizure agent). "
            "Developmental benefit: children on pyridoxine + folinic acid have significantly "
            "higher IQ scores and better language outcomes vs pyridoxine alone (van Karnebeek "
            "2012 triple-therapy cohort). 5-MTHF normalisation in CSF within 3-6 months."
        ),
        "safety": (
            "Excellent safety profile. No known dose-dependent toxicity at standard doses. "
            "Very rarely: GI upset. Does NOT mask B12 deficiency (unlike folic acid). "
            "Safe in pregnancy — particularly important in women with PDE planning pregnancy."
        ),
        "monitoring": (
            "CSF 5-MTHF: baseline and 6-12 months after initiation. Serum folate + B12: "
            "annual. Developmental milestones: Bayley-III/Griffiths q12M. Response to folinic "
            "acid adjunct often reflected in improved developmental trajectory rather than "
            "EEG/seizure metrics."
        ),
        "contraindications": "No absolute contraindications at therapeutic doses.",
    },
    {
        "drug": "Lysine-restricted diet — triple therapy component",
        "evidence": "Level B — Component of triple therapy, recommended for developmental outcome",
        "indication": (
            "Reduces upstream substrate availability in ALDH7A1 pathway → less AASA/P6C "
            "production → less PLP inactivation → better metabolic control. Shown to improve "
            "developmental outcomes (IQ +15-20 pts) even when seizures already controlled "
            "by pyridoxine. Initiated as early as possible — ideally within first weeks of life."
        ),
        "dose": (
            "Total dietary lysine: ≤60 mg/kg/day (infants) / ≤50 mg/kg/day (older children) "
            "via amino acid formula (lysine-free) + natural low-lysine protein. Managed by "
            "metabolic dietitian. L-arginine 300-400 mg/kg/day added as BBB competitor to "
            "further reduce CNS lysine delivery. Monitor plasma lysine q3M: target <100 µmol/L."
        ),
        "moa": (
            "Dietary lysine restriction → reduced lysine catabolism through the saccharopine "
            "pathway → reduced AASA/P6C production even with absent ALDH7A1 → less P6C "
            "available to condense with PLP → less PLP inactivation → less P6C-PLP adduct "
            "formation → improved PLP availability for GAD/AADC activity."
        ),
        "efficacy": (
            "Seizure control: modest incremental benefit over pyridoxine alone. Developmental: "
            "van Karnebeek 2012 triple-therapy cohort showed IQ benefit of +15-20 points "
            "vs pyridoxine monotherapy in matched cohorts. Language and adaptive behaviour "
            "scores also improved. Earlier initiation → greater benefit. Biomarker: plasma "
            "lysine <100 µmol/L confirms dietary adherence."
        ),
        "safety": (
            "Under-restriction → insufficient benefit. Over-restriction → essential amino acid "
            "deficiency (lysine is essential amino acid for growth). Must supplement with "
            "complete amino acid formula to ensure adequate essential AA intake except lysine. "
            "Monitor growth: weight/length/head circumference monthly in infancy. "
            "Nutritional deficiency (zinc, selenium, calcium) risk on restricted diets — "
            "regular micronutrient profiling mandatory."
        ),
        "monitoring": (
            "Plasma lysine q3M (target <100 µmol/L). AASA urine q6M. Growth parameters monthly "
            "(infancy), quarterly thereafter. Micronutrient panel (Zn, Se, Ca, Fe) q6M. "
            "Dietitian clinic visit q3M minimum in infancy; q6M when stable."
        ),
        "contraindications": (
            "Not contraindicated but requires specialist metabolic dietitian: risk of "
            "nutritional deficiency without proper amino acid supplementation. Not appropriate "
            "without metabolic team oversight."
        ),
    },
    {
        "drug": "Pyridoxal 5'-phosphate (PLP) — active cofactor form",
        "evidence": "Level B — Second-line if pyridoxine response incomplete; first-line in PNPO",
        "indication": (
            "Active cofactor form of B6 — directly provides the enzyme cofactor without "
            "requiring conversion by PNPO (relevant in PNPO deficiency phenocopy). In "
            "ALDH7A1-PDE: consider if pyridoxine response incomplete (partial-activity "
            "missense variants). May provide superior cofactor repletion in select patients. "
            "PNPO deficiency: PLP first-line (pyridoxine NOT effective — PNPO cannot convert "
            "it to PLP)."
        ),
        "dose": (
            "30-60 mg/kg/day oral in 4-6 divided doses (short half-life). "
            "Neonatal: 30 mg/kg/day initial. Titrate to clinical response and plasma PLP level "
            "(target 80-150 nmol/L). Available as compounded preparation."
        ),
        "moa": (
            "PLP (pyridoxal 5'-phosphate) is the direct enzyme cofactor — no PNPO conversion "
            "required. In ALDH7A1-PDE with PNPO co-expression limitation, PLP supplementation "
            "provides cofactor that bypasses any conversion bottleneck. In PNPO deficiency: "
            "dietary pyridoxine/pyridoxamine cannot be converted to PLP — PLP supplementation "
            "is the only rescue."
        ),
        "efficacy": (
            "In ALDH7A1-PDE partial responders to pyridoxine: switching to PLP improves "
            "seizure control in ~40% of cases (case series level evidence). "
            "In PNPO phenocopy: >90% seizure cessation within 4-6 hours of PLP — "
            "confirms PNPO diagnosis by therapeutic response. Some centres use PLP as "
            "initial therapy to cover both ALDH7A1 and PNPO in neonatal emergency."
        ),
        "safety": (
            "Similar safety profile to pyridoxine. IV preparation not commercially available "
            "(use IV pyridoxine for acute diagnostic trial). Oral PLP: nausea, GI upset "
            "possible at higher doses. Sensory neuropathy risk at very high chronic doses "
            "(same as pyridoxine — monitor with NCS annually)."
        ),
        "monitoring": "Plasma PLP target 80-150 nmol/L. Clinical seizure diary. EEG q6M.",
        "contraindications": "No absolute contraindications.",
    },
    {
        "drug": "L-Arginine — lysine BBB competitor, CNS-targeted adjunct",
        "evidence": "Level C — Adjunct to triple therapy",
        "indication": (
            "Competes with lysine for LAT1 (L-type amino acid transporter 1) at the BBB → "
            "reduces CNS lysine uptake → less AASA/P6C production in the brain → less PLP "
            "inactivation at the site of action. Mercimek-Mahmutoglu 2014: arginine reduces "
            "CSF AASA and improves PLP availability."
        ),
        "dose": (
            "L-arginine: 300-400 mg/kg/day oral in 3 divided doses. Titrate to plasma arginine "
            "target: 100-200 µmol/L. Measure plasma lysine:arginine ratio — target ratio <1.0 "
            "(arginine > lysine) to ensure competitive inhibition at BBB."
        ),
        "moa": (
            "LAT1 (SLC7A5) transports large neutral amino acids including both lysine (basic "
            "AA, lower affinity) and arginine across the BBB. Excess arginine competitively "
            "inhibits lysine transport → reduced CNS lysine → less lysine catabolism via "
            "saccharopine pathway in neurons → less AASA/P6C production in the brain → "
            "less PLP inactivation → better intracellular PLP availability for GAD."
        ),
        "efficacy": (
            "Mercimek-Mahmutoglu 2014 (Ann Neurol): arginine significantly reduced plasma "
            "and CSF AASA in ALDH7A1-PDE patients. Developmental outcomes improved in "
            "triple-therapy + arginine vs triple-therapy alone in single-centre cohort. "
            "Data from <50 patients — Level C evidence. Currently incorporated into most "
            "specialist metabolic epilepsy centre protocols as 4th component of therapy."
        ),
        "safety": (
            "Generally well tolerated at therapeutic doses. Excess arginine → "
            "hyperargininaemia → rare hyperammonaemia at very high doses. Monitor plasma "
            "arginine. GI discomfort (arginine is osmotically active). Avoid in "
            "ornithine transcarbamylase (OTC) deficiency carriers — rare co-occurrence."
        ),
        "monitoring": "Plasma arginine q3M (target 100-200 µmol/L). Ammonia if clinically indicated.",
        "contraindications": "Caution in urea cycle disorder carriers. Avoid in OTC deficiency.",
    },
    {
        "drug": "Phenobarbitone (Phenobarbital) — acute neonatal bridge pending B6 response",
        "evidence": "Level C — Acute neonatal seizure bridge ONLY",
        "indication": (
            "Standard acute neonatal AED BEFORE pyridoxine diagnosis. Used as bridge therapy "
            "in NICU when seizure aetiology unknown. Stop as soon as PDE confirmed and B6 "
            "response achieved. Do NOT continue as chronic maintenance AED in confirmed PDE. "
            "Note: PB is CYP enzyme inducer → increases pyridoxine metabolism → higher "
            "pyridoxine dose requirement if PB continued."
        ),
        "dose": "Loading 20 mg/kg IV; maintenance 3-5 mg/kg/day IV/oral during acute phase only.",
        "moa": "GABAa receptor positive allosteric modulator (GABA-A potentiation) — non-specific.",
        "efficacy": (
            "Partial seizure suppression in PDE pending pyridoxine (GABA pathway incompletely "
            "functional due to PLP deficiency → PB partially effective). Does NOT address "
            "underlying metabolic defect. Seizure breakthrough common despite PB loading "
            "in PDE — clinical red flag triggering pyridoxine trial."
        ),
        "safety": (
            "Standard PB risks: CNS depression, apnoea (neonatal), paradoxical hyperactivity, "
            "enzyme induction accelerating drug metabolism. In PDE: enzyme induction increases "
            "pyridoxine turnover — increase B6 dose by 20-30% if PB continued beyond acute phase."
        ),
        "monitoring": "PB TDM target 20-40 µg/mL. Wean and stop within 3-6 months of confirmed B6 response.",
        "contraindications": "Not for chronic maintenance in confirmed PDE. Avoid in POLG-related epilepsy.",
    },
    {
        "drug": "ACTH / Corticosteroids — West syndrome component adjunct",
        "evidence": "Level C — Adjunct for West syndrome component in PDE",
        "indication": (
            "When PDE presents with or evolves into West syndrome (infantile spasms). "
            "Combination of ACTH + pyridoxine superior to either alone for spasm cessation "
            "in PDE-West overlap. Steroid course: short (4-6 weeks). Pyridoxine must be "
            "maintained throughout and after ACTH course."
        ),
        "dose": (
            "ACTH (repository corticotropin): 20-30 IU/day IM for 4 weeks, taper. "
            "Alternative: prednisolone 4 mg/kg/day for 2 weeks, taper. "
            "Always co-administered with pyridoxine 30 mg/kg/day."
        ),
        "moa": (
            "ACTH → cortisol → downregulates CRH (corticotropin-releasing hormone) → "
            "reduces CRH-mediated hyperexcitability in immature limbic system → spasm "
            "cessation. In PDE: restores ACTH-cortisol axis disrupted by PLP deficiency "
            "(PLP required for POMC/ACTH synthesis) — synergistic with pyridoxine."
        ),
        "efficacy": (
            "Spasm cessation: ~60-70% with ACTH alone; ~80-85% ACTH + pyridoxine "
            "combined in PDE-West cases. EEG: hypsarrhythmia resolution in ~75% at 2 weeks. "
            "Always followed by continued pyridoxine maintenance — ACTH course does not "
            "replace B6 therapy."
        ),
        "safety": "Standard ACTH risks: hypertension, immunosuppression, GI bleeding, growth suppression. Short course only.",
        "monitoring": "BP, glucose, electrolytes, weight weekly during ACTH course. Infection surveillance.",
        "contraindications": (
            "Active infection (relative CI). Cardiac failure. Do NOT use as monotherapy "
            "in PDE-West — always with pyridoxine."
        ),
    },
    {
        "drug": "Levetiracetam (LEV) — breakthrough seizure adjunct (not substitute for B6)",
        "evidence": "Level C — Short-term breakthrough seizure adjunct only",
        "indication": (
            "Adjunct for breakthrough seizures while optimising pyridoxine dose. "
            "NOT a replacement for pyridoxine. Used as bridge or adjunct in partial B6 "
            "responders, or during dose optimisation phase. Some PDE patients remain on "
            "LEV as second-line adjunct long-term — acceptable if stable."
        ),
        "dose": "20-40 mg/kg/day in 2 divided doses. Titrate to response.",
        "moa": "SV2A synaptic vesicle protein modulator — reduces glutamate release, broad mechanism.",
        "efficacy": (
            "Modest adjunct efficacy in PDE — 30-40% ≥50% seizure reduction vs placebo when "
            "added to B6. Most benefit in partial B6 responders with residual seizures from "
            "inadequate metabolic control. Does not address underlying PLP deficiency."
        ),
        "safety": (
            "Excellent safety profile. Main concern: behavioural/mood effects (irritability, "
            "aggression) — relevant in PDE patients already at risk for behavioural issues "
            "from PLP deficiency. No enzyme induction — preferred adjunct over PB/CBZ/OXC."
        ),
        "monitoring": "Seizure diary. Behavioural assessment at each visit. No TDM required routinely.",
        "contraindications": "Hypersensitivity. Not for use as pyridoxine substitute.",
    },
]

# ── Monitoring Items (8) ──────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "AASA urine (alpha-aminoadipic semialdehyde)",
        "target": ">4 µmol/mmol creatinine = PDE diagnostic threshold; <1 = normal",
        "frequency": "At diagnosis; q6M for metabolic control monitoring; collect BEFORE initiating B6",
        "rationale": (
            "ALDH7A1 deficiency → AASA accumulation (direct metabolic product of blocked step). "
            "AASA urine is the gold-standard diagnostic biomarker — highly sensitive and specific "
            "for ALDH7A1-PDE. Normalises within 24-48 hours of pyridoxine — collect BEFORE "
            "treatment whenever possible. On triple therapy, AASA should remain <2 µmol/mmol "
            "(residual from incomplete lysine restriction); rising AASA = diet or dose failure."
        ),
    },
    {
        "item": "Plasma PLP (pyridoxal 5'-phosphate) level",
        "target": "80-150 nmol/L (therapeutic range for PDE); <50 = subtherapeutic",
        "frequency": "q3M initially; q6M when stable",
        "rationale": (
            "PLP plasma level confirms adequate pyridoxine supplementation and guides dose "
            "adjustment. Subtherapeutic PLP → ongoing PLP inactivation by P6C → seizure "
            "risk. Post-IV trial PLP rises within 1-2 hours — can monitor in real-time. "
            "PLP above 300 nmol/L in adults: peripheral neuropathy risk — adjust dose if elevated."
        ),
    },
    {
        "item": "Plasma lysine",
        "target": "<100 µmol/L (dietary adherence on lysine restriction)",
        "frequency": "q3M",
        "rationale": (
            "Plasma lysine reflects dietary lysine intake. Target <100 µmol/L confirms "
            "adherence to lysine-restricted diet. Elevated lysine → more AASA/P6C production "
            "→ more PLP inactivation → suboptimal metabolic control. Lysine-arginine ratio "
            "should also be checked: target arginine:lysine >1.0 to ensure BBB competition "
            "is effective in patients on arginine supplementation."
        ),
    },
    {
        "item": "CSF 5-MTHF (5-methyltetrahydrofolate) + neurotransmitters",
        "target": "CSF 5-MTHF >40 nmol/L; CSF PLP >5 nmol/L; HVA/HIAA in reference range",
        "frequency": "Baseline; q12M or after clinical change",
        "rationale": (
            "P6C-PLP condensation depletes CSF folate (5-MTHF) → secondary cerebral folate "
            "deficiency → impaired monoamine neurotransmitter synthesis (dopamine, serotonin). "
            "CSF 5-MTHF monitoring confirms folinic acid efficacy. Neurotransmitter metabolites "
            "(HVA, HIAA, GABA) reflect global PLP-dependent enzyme function. Low CSF GABA "
            "= ongoing GAD deficiency despite treatment → dose insufficiency. Lumbar puncture "
            "under anaesthesia — minimise invasiveness: combine with other CSF sampling."
        ),
    },
    {
        "item": "Developmental/neuropsychological assessment",
        "target": "Bayley-III or Griffiths: age-appropriate; WISC-V IQ if school age",
        "frequency": "q6M in infancy; q12M thereafter",
        "rationale": (
            "PDE outcome is strongly correlated with metabolic control (particularly triple "
            "therapy initiation). Developmental monitoring detects regression or plateau "
            "before clinical seizure recurrence. Language outcomes (expressive vocabulary, "
            "receptive language age) are sensitive markers of folinic acid + lysine "
            "restriction efficacy. Neuropsychologist assessment q12M. SLP input if language delay."
        ),
    },
    {
        "item": "EEG (routine + sleep EEG)",
        "target": "Age-appropriate EEG background; no epileptiform activity",
        "frequency": "At diagnosis; q6M; after any clinical change or breakthrough seizure",
        "rationale": (
            "EEG monitors metabolic control — persistent interictal abnormalities signal "
            "subtherapeutic treatment. In infancy: check for persisting burst-suppression "
            "features or modified hypsarrhythmia. In childhood: focal or generalised "
            "epileptiform activity = suboptimal PLP availability. Sleep EEG captures "
            "sleep-potentiated discharges. Real-time EEG monitoring during IV B6 trial "
            "documents therapeutic response (1-hour response window)."
        ),
    },
    {
        "item": "Peripheral neuropathy screen (clinical + NCS)",
        "target": "Normal nerve conduction velocities; no sensory symptoms",
        "frequency": "Annual from age 8 years; earlier if symptomatic",
        "rationale": (
            "Sensory peripheral neuropathy is a dose-dependent toxicity of high-dose pyridoxine "
            "(>200-500 mg/day in adults, equivalent weight-based threshold in children). "
            "Mechanism: megadose B6 causes dorsal root ganglion neurotoxicity. Subclinical "
            "neuropathy detectable by NCS before symptoms appear. Annual screening enables "
            "dose adjustment to prevent progression. NCS: sural + median sensory amplitudes "
            "and velocities. Also screen for neuropathy as comorbidity of PLP deficiency "
            "itself (axonal neuropathy from GAD deficiency in peripheral neurons)."
        ),
    },
    {
        "item": "Brain MRI (structural + advanced imaging)",
        "target": "No progressive changes; periventricular signal resolving on treatment",
        "frequency": "At diagnosis; q12-24M; earlier if developmental plateau",
        "rationale": (
            "MRI findings in untreated or late-treated PDE: periventricular T2 hyperintensity "
            "(gliosis from early PLP-deficient state), delayed myelination, corpus callosum "
            "thinning, cerebral atrophy in severe/late cases. With early effective triple "
            "therapy: MRI normalises progressively. Late-treated cases: irreversible gliosis. "
            "MRI at diagnosis establishes baseline. Serial q12-24M detects progressive "
            "change vs expected improvement. DWI useful in acute neonatal phase to assess "
            "injury extent."
        ),
    },
]

# ── Lifecycle Windows (6) ─────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Prenatal / Neonatal-NICU",
        "age_range": "0-28 days",
        "focus": "Emergency recognition, IV pyridoxine trial, AASA collection, triple therapy initiation",
        "key_action": "IV pyridoxine 30 mg/kg STAT under EEG + cardiac monitoring; collect AASA BEFORE B6",
    },
    {
        "window": "Early Infantile",
        "age_range": "1-12 months",
        "focus": "Establish oral maintenance dose; initiate lysine restriction + folinic acid",
        "key_action": "Metabolic dietitian + triple therapy; monthly weight for dose adjustment; Bayley-III q6M",
    },
    {
        "window": "Late Infantile / Toddler",
        "age_range": "1-3 years",
        "focus": "Developmental monitoring; dose titration to weight; detect late-onset missed cases",
        "key_action": "Quarterly plasma lysine + PLP; folinic acid optimisation; SLP if language delay",
    },
    {
        "window": "Preschool / School Age",
        "age_range": "3-12 years",
        "focus": "Education support, adherence, neuropathy surveillance, growth monitoring on diet",
        "key_action": "Annual NCS from age 8; WISC-V IQ q12M; dietitian q6M; school IEP support",
    },
    {
        "window": "Adolescence",
        "age_range": "12-18 years",
        "focus": "Diet adherence challenges, transition to adult care, reproductive counselling",
        "key_action": "Psychoeducation for adherence; adult metabolic team transition plan; contraception/genetics",
    },
    {
        "window": "Adulthood",
        "age_range": "18 years+",
        "focus": "Reproductive planning (AR genetics), long-term B6 neuropathy, lifestyle integration",
        "key_action": "Genetic counselling for family planning; annual NCS; lysine diet adaptation; PDE consortium registry",
    },
]

# ── Clinical Alerts (5) ───────────────────────────────────────────────────────
ALERTS = [
    "🚨 MANDATORY: IV pyridoxine 30 mg/kg trial in ALL unexplained neonatal seizures — administer under EEG + cardiac monitoring (apnoea risk — bag-mask at bedside)",
    "🚨 COLLECT AASA URINE before pyridoxine treatment wherever possible — AASA normalises within 24-48h of B6, making post-treatment diagnosis difficult",
    "⚡ TRIPLE THERAPY SUPERIOR: pyridoxine alone → good seizure control but IQ −15-20 pts vs pyridoxine + folinic acid + lysine restriction; initiate triple therapy at diagnosis",
    "⚠️ B6 DOSE ADJUSTMENT MANDATORY AT EVERY VISIT — dose in mg/kg; as child grows → dose becomes subtherapeutic → breakthrough seizures. Check weight and recalculate.",
    "⚠️ SICK-DAY PROTOCOL: fever/illness increases lysine catabolism → more AASA/P6C → seizure risk — increase B6 to 30-40 mg/kg/day during illness; parents must have home supply",
]

# ── Key Concepts / Definitions (14) ──────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "ALDH7A1 / Antiquitin",
        "definition": (
            "Aldehyde Dehydrogenase 7 Family Member A1 (5q23.2). Encodes antiquitin, "
            "the enzyme alpha-aminoadipic semialdehyde (AASA) dehydrogenase — a key enzyme "
            "in the lysine catabolism saccharopine pathway. LOF → AASA/P6C accumulation → "
            "PLP inactivation → pyridoxine-dependent epilepsy."
        ),
    },
    {
        "term": "Pyridoxine-Dependent Epilepsy (PDE-ALDH7A1)",
        "definition": (
            "Autosomal recessive metabolic epilepsy (OMIM #266100) caused by biallelic "
            "LOF variants in ALDH7A1. Characterised by neonatal/infantile seizures responsive "
            "to pyridoxine (Vitamin B6). Most common treatable genetic neonatal epilepsy. "
            "Part of ILAE 2022 genetic/metabolic epilepsy classification."
        ),
    },
    {
        "term": "AASA (alpha-aminoadipic semialdehyde)",
        "definition": (
            "The direct metabolic substrate of ALDH7A1 enzyme. Accumulates in urine, plasma, "
            "and CSF in ALDH7A1 deficiency. Diagnostic gold standard: urine AASA "
            ">4 µmol/mmol creatinine = PDE. Normalises within 24-48h of pyridoxine — "
            "collect BEFORE treatment. The cyclic form is Δ1-piperideine-6-carboxylate (P6C)."
        ),
    },
    {
        "term": "P6C-PLP Knoevenagel Condensation",
        "definition": (
            "The key molecular mechanism of PDE (Mills 2006): P6C (cyclic form of AASA) "
            "undergoes spontaneous Knoevenagel condensation with pyridoxal 5'-phosphate (PLP) "
            "→ irreversible covalent adduct → functional PLP depletion → failure of >50 "
            "PLP-dependent enzymes (GAD, AADC, kynureninase, cystathionine β-synthase) "
            "→ GABA/neurotransmitter deficiency → seizures."
        ),
    },
    {
        "term": "PLP (Pyridoxal 5'-Phosphate)",
        "definition": (
            "The biologically active form of Vitamin B6 — essential cofactor for >50 enzymes. "
            "Critical: GAD (glutamic acid decarboxylase, GABA synthesis) and AADC "
            "(aromatic L-amino acid decarboxylase, dopamine/serotonin synthesis). "
            "PLP inactivation by P6C → global failure of PLP-dependent metabolism → "
            "GABA deficiency → neonatal seizures. Pyridoxine supplementation restores PLP."
        ),
    },
    {
        "term": "Triple Therapy (PDE-ALDH7A1)",
        "definition": (
            "van Karnebeek 2012 triple-therapy regimen for optimal PDE outcomes: "
            "(1) Pyridoxine 15-30 mg/kg/day + (2) Folinic acid 3-5 mg/kg/day + "
            "(3) Lysine-restricted diet ≤60 mg/kg/day + optional L-arginine. "
            "Achieves IQ advantage of +15-20 points vs pyridoxine alone by addressing "
            "all three metabolic consequences of ALDH7A1 deficiency."
        ),
    },
    {
        "term": "Pipecolic Acid",
        "definition": (
            "Secondary metabolic biomarker of ALDH7A1 deficiency. Pipecolic acid is the "
            "alternative lysine catabolism product that accumulates when the saccharopine "
            "pathway is blocked. Plasma/CSF pipecolic acid elevated in >85% of ALDH7A1-PDE. "
            "Used as secondary diagnostic marker alongside AASA. Also elevated in Zellweger "
            "syndrome (peroxisomal disorders) — distinguish by context."
        ),
    },
    {
        "term": "Folinic Acid (Leucovorin) in PDE",
        "definition": (
            "5-formyltetrahydrofolate — provides active folate bypassing the DHFR reduction "
            "step. Essential in PDE because P6C-PLP condensation depletes CSF 5-MTHF "
            "(secondary cerebral folate deficiency) → impaired one-carbon metabolism → "
            "reduced monoamine neurotransmitter synthesis. Folinic acid ≠ folic acid "
            "(folic acid requires DHFR conversion; folinic acid is directly active)."
        ),
    },
    {
        "term": "Lysine-Restricted Diet (PDE)",
        "definition": (
            "Dietary reduction of lysine to ≤60 mg/kg/day → reduces substrate flux through "
            "blocked ALDH7A1 step → less AASA/P6C production → less PLP inactivation → "
            "improved metabolic control. Superior to pyridoxine alone for developmental "
            "outcomes. Managed by metabolic dietitian using lysine-free amino acid formula "
            "+ controlled natural protein."
        ),
    },
    {
        "term": "PNPO Deficiency (PDE Phenocopy)",
        "definition": (
            "Pyridox(am)ine 5'-phosphate oxidase deficiency — cannot convert dietary "
            "pyridoxine to active PLP. Clinically indistinguishable from ALDH7A1-PDE. "
            "KEY DIFFERENCE: responds to PLP but NOT to pyridoxine. "
            "Differentiation algorithm: if pyridoxine fails → trial PLP 60 mg/kg/day → "
            "response = PNPO. Molecular: PNPO gene testing. CSF PLP very low in PNPO."
        ),
    },
    {
        "term": "Autosomal Recessive ALDH7A1 Inheritance",
        "definition": (
            "Both parents are obligate heterozygous carriers (phenotypically normal). "
            "Each pregnancy: 25% risk of affected child (biallelic LOF). "
            "Carrier frequency: ~1:200 in general population. Consanguinity increases risk. "
            "Prenatal diagnosis: molecular testing of foetal DNA after chorionic villus "
            "sampling (CVS) at 10-12 weeks or amniocentesis at 15-20 weeks."
        ),
    },
    {
        "term": "Pyridoxine Peripheral Neuropathy (Dose-Dependent Toxicity)",
        "definition": (
            "Sensory peripheral neuropathy from chronic high-dose pyridoxine: dorsal root "
            "ganglion (DRG) neurotoxicity. Risk threshold: adults >200-500 mg/day chronically; "
            "children: monitor weight-adjusted doses above 15 mg/kg/day long-term. "
            "Subclinical: detected on NCS before symptoms. Annual NCS from age 8 years. "
            "Management: reduce to minimum effective dose maintaining seizure control."
        ),
    },
    {
        "term": "L-Arginine BBB Competition",
        "definition": (
            "L-arginine competes with lysine for LAT1 (SLC7A5) transporter at the blood-brain "
            "barrier → reduces CNS lysine delivery → less intra-neuronal AASA/P6C production "
            "→ less PLP inactivation at the site of pathology. Mercimek-Mahmutoglu 2014: "
            "reduced plasma and CSF AASA with arginine supplementation. "
            "Dose: 300-400 mg/kg/day. Plasma arginine target: 100-200 µmol/L."
        ),
    },
    {
        "term": "PDE Consortium Registry",
        "definition": (
            "International registry for ALDH7A1-PDE and related B6-responsive epilepsies. "
            "Maintained by the International PDE Consortium (van Karnebeek, Gospe, Mills, "
            "Pearl). Collects genotype-phenotype data, long-term outcomes, treatment "
            "responses. All confirmed PDE patients should be enrolled. Registry data "
            "informs treatment guidelines and enables clinical trial recruitment for "
            "emerging therapies (mRNA/gene therapy preclinical work underway)."
        ),
    },
]

# ── Standards (8) ─────────────────────────────────────────────────────────────
STANDARDS = [
    {
        "standard": "ILAE-2022",
        "title": "International League Against Epilepsy 2022 Epilepsy Classification",
        "relevance": "PDE-ALDH7A1 classified as genetic/metabolic epilepsy — structural/metabolic aetiology category",
    },
    {
        "standard": "NICE-NG217",
        "title": "NICE Guideline NG217: Epilepsies — Diagnosis and Management (UK, 2022)",
        "relevance": "Recommends pyridoxine trial (30 mg/kg IV) in ALL unexplained neonatal seizures; recognises PDE as treatable genetic epilepsy",
    },
    {
        "standard": "Mills-2006-NatMed",
        "title": "Mills et al. 2006 — Nature Medicine — ALDH7A1 gene discovery",
        "relevance": "Landmark paper identifying ALDH7A1 (antiquitin) as the PDE gene and establishing P6C-PLP condensation mechanism",
    },
    {
        "standard": "van-Karnebeek-2012-JIMD",
        "title": "van Karnebeek et al. 2012 — JIMD — Triple therapy protocol",
        "relevance": "Evidence base for triple therapy (pyridoxine + folinic acid + lysine restriction) — IQ benefit +15-20 points vs monotherapy",
    },
    {
        "standard": "ACMG-AMP-2015",
        "title": "ACMG/AMP Standards for Variant Classification (Richards 2015, Genet Med)",
        "relevance": "Framework for ALDH7A1 variant pathogenicity: PVS1 (null alleles) + PS2 (de novo/biallelic AR) + PM2 (absent gnomAD)",
    },
    {
        "standard": "ACNS-EEG-2021",
        "title": "American Clinical Neurophysiology Society Neonatal EEG Guidelines 2021",
        "relevance": "Standards for neonatal EEG monitoring — burst-suppression recognition, ictal morphology, pyridoxine response documentation",
    },
    {
        "standard": "EAN-NeonatalSE-2019",
        "title": "European Academy of Neurology — Neonatal Status Epilepticus 2019",
        "relevance": "Recommends pyridoxine trial in treatment-refractory neonatal seizures — Level A evidence for empiric B6 trial",
    },
    {
        "standard": "Mercimek-Mahmutoglu-2014-AnnNeurol",
        "title": "Mercimek-Mahmutoglu et al. 2014 — Ann Neurology — Arginine adjunct",
        "relevance": "Evidence for L-arginine as 4th therapy component reducing CNS lysine via BBB competition in PDE-ALDH7A1",
    },
]

# ── Thresholds (10) ───────────────────────────────────────────────────────────
THRESHOLDS = [
    {
        "threshold": "Pyridoxine IV 30 mg/kg trial — MANDATORY in unexplained neonatal seizures",
        "action": "Administer IV pyridoxine 30 mg/kg over 30 min with continuous EEG + cardiac monitoring in all unexplained neonatal seizures unresponsive to standard AEDs",
    },
    {
        "threshold": "AASA urine >4 µmol/mmol creatinine — diagnostic PDE threshold",
        "action": "AASA elevated → confirm ALDH7A1 molecular testing; initiate triple therapy; parental carrier testing",
    },
    {
        "threshold": "EEG burst-suppression resolution within 1 hour of IV B6",
        "action": "Time-locked EEG response = pathognomonic for PDE → confirm diagnosis; proceed to triple therapy; genetic confirmation",
    },
    {
        "threshold": "Plasma PLP <50 nmol/L — subtherapeutic",
        "action": "Increase pyridoxine dose by 20-30%; check weight-based calculation; assess adherence; exclude enzyme inducers",
    },
    {
        "threshold": "Plasma lysine >100 µmol/L — diet non-adherent",
        "action": "Reassess lysine-restricted diet adherence with metabolic dietitian; adjust amino acid formula; psychosocial support for older patients",
    },
    {
        "threshold": "Pyridoxine peripheral neuropathy screen (adults >200 mg/day)",
        "action": "Annual NCS for sensory neuropathy from age 8 years; reduce to minimum effective dose if neuropathy detected",
    },
    {
        "threshold": "CSF 5-MTHF <40 nmol/L — secondary cerebral folate deficiency",
        "action": "Increase folinic acid dose; confirm not using folic acid as substitute; reassess metabolic control; neurotransmitter profile",
    },
    {
        "threshold": "Pyridoxine trial NEGATIVE → trial PLP 60 mg/kg/day",
        "action": "If IV pyridoxine 30 mg/kg produces no EEG/clinical response within 1 hour → trial IV/oral PLP to exclude PNPO phenocopy",
    },
    {
        "threshold": "AASA normalised on treatment but clinical relapse",
        "action": "Measure AASA after briefly holding B6 (1-2 days) to unmask elevated AASA confirming ongoing metabolic PDE; check PLP adherence",
    },
    {
        "threshold": "Developmental plateau or regression — any age",
        "action": "Assess for suboptimal metabolic control; repeat AASA, PLP, plasma lysine, CSF 5-MTHF; escalate triple therapy; neuropsych assessment urgently",
    },
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "ref": "Mills et al. 2006 — Nature Medicine",
        "title": "A pyridoxine-dependent seizure syndrome caused by mutations in antiquitin (ALDH7A1)",
        "relevance": "Landmark discovery paper identifying ALDH7A1 as the PDE gene; established P6C-PLP Knoevenagel condensation mechanism; defined AASA as diagnostic biomarker",
    },
    {
        "ref": "van Karnebeek et al. 2012 — Journal of Inherited Metabolic Disease",
        "title": "Lysine restricted diet for pyridoxine-dependent epilepsy: first evidence and future trials",
        "relevance": "Established triple therapy protocol (B6 + folinic acid + lysine restriction); demonstrated IQ benefit +15-20 pts vs B6 monotherapy",
    },
    {
        "ref": "Coughlin et al. 2015 — Journal of Inherited Metabolic Disease",
        "title": "Genotype-phenotype correlations in ALDH7A1 deficiency: a multicenter study",
        "relevance": "Largest genotype-phenotype cohort study; established partial-activity missense variants as late-onset PDE; guided variant-specific counselling",
    },
    {
        "ref": "Mercimek-Mahmutoglu et al. 2014 — Annals of Neurology",
        "title": "Efficacy of arginine supplementation in ALDH7A1 pyridoxine-dependent epilepsy",
        "relevance": "Demonstrated L-arginine reduces plasma/CSF AASA via LAT1 BBB competition; evidence base for arginine as 4th therapy component",
    },
    {
        "ref": "Pearl et al. 2022 — Pediatric Neurology",
        "title": "Pyridoxine-dependent epilepsy: practical guide to diagnosis and management",
        "relevance": "Comprehensive clinical practice review; algorithm for diagnostic workup, triple therapy initiation, monitoring thresholds, and phenocopy differentiation",
    },
    {
        "ref": "Plecko et al. 2007 — Annals of Neurology",
        "title": "Biochemical characterisation and follow-up of ALDH7A1-PDE: pipecolic acid and AASA",
        "relevance": "Established pipecolic acid as secondary biomarker; longitudinal biomarker trajectories on treatment; rationale for AASA over pipecolic acid as primary marker",
    },
]


# ── Patient Cohort (N=41) ─────────────────────────────────────────────────────
def _make_patients():
    random.seed(SEED)
    patients = []
    classes = [
        ("LOF-classic-neonatal",    18, "ALDH7A1-biallelic-LOF-classic-neonatal"),
        ("LOF-partial-atypical",    12, "ALDH7A1-biallelic-missense-partial-activity"),
        ("LOF-splice",               5, "ALDH7A1-biallelic-splice-site"),
        ("LOF-CNV",                  3, "ALDH7A1-biallelic-CNV"),
        ("PDE-phenocopy",            3, "clinical-PDE-ALDH7A1-negative-phenocopy"),
    ]
    treatments_pool = [
        "Pyridoxine+Folinic+LysineRestriction",
        "Pyridoxine+Folinic",
        "Pyridoxine+LysineRestriction+Arginine",
        "Pyridoxine+Folinic+LysineRestriction+Arginine",
        "PLP+Folinic+LysineRestriction",
        "Pyridoxine+LEV",
        "Pyridoxine+Folinic+LEV",
    ]
    phases = [
        "Neonatal-NICU-stabilisation",
        "Infantile-triple-therapy-initiation",
        "Maintenance-diet-established",
        "School-age-monitoring",
        "Adolescent-adherence",
    ]
    controls = ["seizure-free", "partial-control", "drug-resistant"]
    pid = 1
    for fc, n, cat in classes:
        for _ in range(n):
            if fc == "LOF-classic-neonatal":
                onset_days = random.randint(0, 7)
                age_months = random.randint(6, 36)
                ctrl = random.choices(controls, weights=[80, 15, 5])[0]
                pyridoxine_resp = random.choice(["complete", "complete", "complete", "partial"])
                aasa_baseline = round(random.uniform(10, 30), 1)
                aasa_ontrx = round(random.uniform(0.5, 2.5), 1)
                lysine_restricted = True
                folinic = True
            elif fc == "LOF-partial-atypical":
                onset_days = random.randint(30, 540)
                age_months = random.randint(12, 72)
                ctrl = random.choices(controls, weights=[65, 25, 10])[0]
                pyridoxine_resp = random.choice(["partial", "partial", "complete", "none"])
                aasa_baseline = round(random.uniform(4, 12), 1)
                aasa_ontrx = round(random.uniform(1.0, 4.0), 1)
                lysine_restricted = random.random() > 0.3
                folinic = random.random() > 0.2
            elif fc == "LOF-splice":
                onset_days = random.randint(0, 180)
                age_months = random.randint(8, 48)
                ctrl = random.choices(controls, weights=[70, 20, 10])[0]
                pyridoxine_resp = random.choice(["complete", "partial", "partial"])
                aasa_baseline = round(random.uniform(5, 20), 1)
                aasa_ontrx = round(random.uniform(0.8, 3.5), 1)
                lysine_restricted = random.random() > 0.3
                folinic = True
            elif fc == "LOF-CNV":
                onset_days = random.randint(0, 3)
                age_months = random.randint(6, 24)
                ctrl = random.choices(controls, weights=[75, 15, 10])[0]
                pyridoxine_resp = random.choice(["complete", "complete", "partial"])
                aasa_baseline = round(random.uniform(12, 25), 1)
                aasa_ontrx = round(random.uniform(0.6, 2.0), 1)
                lysine_restricted = True
                folinic = True
            else:  # phenocopy
                onset_days = random.randint(0, 30)
                age_months = random.randint(6, 30)
                ctrl = random.choices(controls, weights=[60, 30, 10])[0]
                pyridoxine_resp = random.choice(["partial", "none", "complete"])
                aasa_baseline = round(random.uniform(0.5, 3.5), 1)
                aasa_ontrx = round(random.uniform(0.3, 2.0), 1)
                lysine_restricted = random.random() > 0.5
                folinic = True

            plp_level = round(random.uniform(
                80 if ctrl == "seizure-free" else 30,
                160 if ctrl == "seizure-free" else 80
            ), 0)
            plasma_lysine = round(random.uniform(
                40 if lysine_restricted else 120,
                100 if lysine_restricted else 250
            ), 0)

            patients.append({
                "id": f"PDE-{pid:03d}",
                "age_months": age_months,
                "sex": random.choice(["M", "F"]),
                "onset_age_days": onset_days,
                "functional_class": fc,
                "category": cat,
                "disease_phase": random.choice(phases),
                "current_treatment": random.choice(treatments_pool),
                "seizure_control": ctrl,
                "pyridoxine_response": pyridoxine_resp,
                "plp_level_nmoll": int(plp_level),
                "plasma_lysine_umoll": int(plasma_lysine),
                "aasa_baseline": aasa_baseline,
                "aasa_on_treatment": aasa_ontrx,
                "lysine_restricted": lysine_restricted,
                "folinic_acid": folinic,
                "pip_acid_elevated": fc in ("LOF-classic-neonatal", "LOF-CNV"),
            })
            pid += 1
    return patients


PATIENTS = _make_patients()


# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    """ALDH7A1-PDE (Pyridoxine-Dependent Epilepsy / Antiquitin Deficiency) — overview endpoint."""
    total = sum(e["n"] for e in ETIOLOGY_CATALOG)
    sz_free = sum(1 for p in PATIENTS if p["seizure_control"] == "seizure-free")
    dre = sum(1 for p in PATIENTS if p["seizure_control"] == "drug-resistant")
    lysine_n = sum(1 for p in PATIENTS if p["lysine_restricted"])
    folinic_n = sum(1 for p in PATIENTS if p["folinic_acid"])
    b6_resp = sum(1 for p in PATIENTS if p["pyridoxine_response"] in ("complete", "partial"))
    aasa_elevated = sum(1 for p in PATIENTS if p["aasa_baseline"] >= 4.0)
    return {
        "syndrome": "ALDH7A1 Pyridoxine-Dependent Epilepsy (PDE-ALDH7A1 / Antiquitin Deficiency)",
        "gene": "ALDH7A1 — 5q23.2 — Alpha-aminoadipic semialdehyde dehydrogenase (Antiquitin)",
        "inheritance": "Autosomal recessive (biallelic LOF — 25% recurrence per pregnancy)",
        "eeg_hallmark": "Neonatal burst-suppression → resolves within 1 hour of IV pyridoxine 30 mg/kg (pathognomonic)",
        "key_biomarker": "AASA urine >4 µmol/mmol creatinine (gold-standard diagnostic — collect BEFORE B6)",
        "precision_therapy": "IV pyridoxine 30 mg/kg (diagnostic + therapeutic) → triple therapy: B6 + folinic acid + lysine restriction",
        "n_patients": total,
        "kpis": {
            "b6_responsive_pct": round(b6_resp / total * 100),
            "aasa_positive_pct": round(aasa_elevated / total * 100),
            "dre_pct": round(dre / total * 100),
            "seizure_free_pct": round(sz_free / total * 100),
            "lysine_restricted_pct": round(lysine_n / total * 100),
            "folinic_acid_pct": round(folinic_n / total * 100),
        },
        "etiologies": [
            {"etiology": e["etiology"][:60], "n": e["n"], "pct": e["pct"]}
            for e in ETIOLOGY_CATALOG
        ],
        "seizure_type_prevalence": {s["type"][:50]: s["prevalence_pct"] for s in SEIZURE_TYPES},
        "trigger_seizure_rates": {t["trigger"][:50]: t["rate_pct"] for t in TRIGGERS},
        "lifecycle_windows": LIFECYCLE,
        "clinical_alerts": ALERTS,
        "key_aha": (
            "PDE-ALDH7A1 is the most common treatable genetic neonatal epilepsy — IV pyridoxine "
            "30 mg/kg trial is MANDATORY in unexplained neonatal seizures. Missing the diagnosis "
            "= irreversible neurodevelopmental harm; treating with B6 = seizure-free life."
        ),
        "generated_at": datetime.now().isoformat(),
        "dashboard_id": 186,
    }


def get_breakdown():
    """ALDH7A1-PDE — breakdown endpoint (full clinical detail)."""
    return {
        "patients": PATIENTS,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "alerts": ALERTS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }


def get_definitions():
    """ALDH7A1-PDE — definitions endpoint (14 key concepts + contraindications + thresholds)."""
    return {
        "syndrome": "ALDH7A1 Pyridoxine-Dependent Epilepsy (PDE-ALDH7A1 / Antiquitin Deficiency)",
        "definitions": DEFINITIONS,
        "absolute_contraindications": [
            {
                "drug": "Withhold pyridoxine trial in unexplained neonatal seizures",
                "scope": "ABSOLUTE — all unexplained neonatal seizures unresponsive to PB",
                "mechanism": "Failure to trial → missed PDE diagnosis → ongoing PLP deficiency → progressive neuronal death → irreversible cognitive impairment",
                "action": "IV pyridoxine 30 mg/kg MANDATORY in all unexplained neonatal seizures — under EEG + cardiac monitoring",
                "evidence": "NICE NG217 / EAN Neonatal SE 2019 / ILAE 2022 — Level A recommendation",
            },
            {
                "drug": "AASA post-pyridoxine (diagnostic error)",
                "scope": "CRITICAL — AASA normalises within 24-48h of B6",
                "mechanism": "Pyridoxine treatment → AASA normalises → false-negative AASA → missed PDE diagnosis. Collect urine AASA BEFORE pyridoxine whenever clinically feasible",
                "action": "AASA collection protocol: (1) midstream urine specimen BEFORE IV B6; (2) if already treated: withhold B6 for 24-48h (under monitoring) then recheck AASA",
                "evidence": "Mills 2006 NatMed — AASA normalises within 24-48h of B6 treatment",
            },
            {
                "drug": "Lysine-rich diet without supplementation (diet without metabolic oversight)",
                "scope": "HIGH RISK — increases AASA/P6C substrate load",
                "mechanism": "Excess dietary lysine → increased ALDH7A1 substrate flux → more AASA/P6C → more PLP inactivation → worsening encephalopathy despite B6",
                "action": "Metabolic dietitian mandatory for lysine restriction protocol; amino acid formula for complete nutrition; avoid high-lysine foods (meat excess, legumes, dairy unrestricted)",
                "evidence": "van Karnebeek 2012 JIMD — lysine restriction protocol and monitoring guidelines",
            },
            {
                "drug": "Folic acid as substitute for folinic acid",
                "scope": "HIGH RISK — ineffective for cerebral folate deficiency in PDE",
                "mechanism": "Folic acid requires DHFR conversion to 5-MTHF. P6C-PLP condensation also impairs DHFR pathway → folic acid cannot bypass DHFR bottleneck → CSF 5-MTHF not restored",
                "action": "Use FOLINIC ACID (leucovorin / 5-formyltetrahydrofolate) NOT folic acid. Confirm prescription specifies leucovorin/folinic acid.",
                "evidence": "PDE management guidelines — van Karnebeek 2012; Pearl 2022",
            },
        ],
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }
