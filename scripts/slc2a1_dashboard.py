"""
SLC2A1 Epilepsy — GLUT1 Deficiency Syndrome (Glut1-DS / De Vivo Disease)
=========================================================================
41-patient cohort · SLC2A1 (1p34.2) · Glucose Transporter Type 1 (GLUT1)

SLC2A1 / GLUT1 BIOLOGY:
SLC2A1 (1p34.2) encodes GLUT1 (Glucose Transporter Type 1), the principal facilitative glucose
transporter at the blood-brain barrier (BBB), choroid plexus, and astrocytic end-feet. GLUT1 is a
12-transmembrane-domain protein that mediates sodium-independent, bidirectional, concentration-
dependent glucose transport across the BBB and into brain parenchymal cells.

GLUT1-DS PATHOPHYSIOLOGY:
Heterozygous or biallelic SLC2A1 loss-of-function variants → reduced GLUT1 protein expression/
function → impaired glucose delivery across the BBB → HYPOGLYCORRHACHIA (low CSF glucose) despite
normal blood glucose → cerebral glucose deficiency (brain uses ~20% of body glucose despite 2% mass)
→ neuronal energy failure in glucose-dependent pathways → epilepsy + movement disorder + cognitive.

PRECISION THERAPY MECHANISM — WHY KETOGENIC DIET IS CURATIVE:
When dietary fat is the primary fuel → hepatic β-oxidation → ketogenesis (β-hydroxybutyrate
[β-OHB] + acetoacetate) → ketone bodies cross BBB via MCT1 (monocarboxylate transporter 1,
NOT requiring GLUT1) → enter TCA cycle (β-OHB → acetoacetyl-CoA → acetyl-CoA) → ATP synthesis.
The ketogenic diet BYPASSES the defective GLUT1 transporter entirely — ketones substitute for
glucose as the primary neuronal fuel source.
KD efficacy in Glut1-DS: >90% seizure-free or >50% reduction (Level A evidence). EEG normalises
in 70-85%. Movement disorder improves in 60%. Cognitive trajectory improves vs untreated (but
structural changes from delayed diagnosis may be irreversible). KD should be initiated at diagnosis
without waiting for 2 AED failures (unlike other epilepsies) — it is the PRECISION treatment.

METHYLXANTHINE ABSOLUTE CONTRAINDICATION:
Caffeine, theophylline, aminophylline, and all methylxanthines are competitive GLUT1 inhibitors.
Mechanism: methylxanthines bind the glucose-binding site on GLUT1 → competitive inhibition →
further reduces already-impaired glucose transport → acute hypoglycorrhachia worsening → seizure
exacerbation + movement disorder flare.
Clinical impact: a single espresso (80 mg caffeine) can precipitate acute seizure exacerbation in
Glut1-DS. Cola drinks, energy drinks, chocolate (theobromine — closely related methylxanthine) ALL
implicated. School nurses, GP, A&E must be informed: NO CAFFEINE in any form.

HYPOGLYCORRHACHIA — DIAGNOSTIC HALLMARK:
CSF glucose: <2.2 mmol/L (40 mg/dL) OR CSF:plasma glucose ratio <0.45 (normal ≥0.65)
Important: plasma glucose MUST be measured simultaneously (fasting) for ratio calculation.
False negatives: ketogenic diet normalises CSF glucose → test BEFORE starting KD.
Confirmatory: erythrocyte glucose uptake assay (50% of normal in heterozygous) + SLC2A1 sequencing.

PAROXYSMAL EXERCISE-INDUCED DYSKINESIA (PED) — PATHOGNOMONIC:
PED = involuntary choreiform/dystonic movements triggered by prolonged exercise (5-20 min sustained
walking/running/cycling), relieved by rest within 5-30 min. Mechanism: exercise → increased muscle
glucose consumption → relative glucose deficit → worsened GLUT1-mediated hypoglycorrhachia in
exercise state. Treated effectively by KD (ketones available during exercise). PED in a child with
epilepsy = GLUT1-DS until proven otherwise. EEG during PED: typically normal (not ictal).

INHERITANCE SPECTRUM:
  AD heterozygous de novo (~90%):  most common — single SLC2A1 allele sufficient for classic Glut1-DS
  AD heterozygous familial (~10%): parent-to-child; variable expressivity within families
  AR homozygous/compound-het:      severe phenotype — <5% of cases; onset neonatal/infantile
  Mosaic SLC2A1:                   ~5% somatic mosaic — milder/atypical presentation
  Large deletion 1p34.2:           encompasses SLC2A1 — haploinsufficiency mechanism

MOST COMMON SLC2A1 VARIANTS (ACMG):
  p.Arg153Cys (c.457C>T)   — most common missense; Europe (~8%)
  p.Glu146Lys (c.436G>A)   — transmembrane domain; common US/UK
  Exon 1-10 deletions       — ~12% of cases (MLPA required for detection)
  p.Arg333His (c.998G>A)    — protein folding/trafficking defect
  p.Gly91Asp (c.272G>A)     — severe phenotype; reduced protein stability

KEY SAFETY PEARLS:
• METHYLXANTHINES ABSOLUTE CI — caffeine (coffee/cola/chocolate), theophylline, aminophylline
  ALL inhibit GLUT1 competitively → acute seizure worsening. Document prominently in EMR.
• PHENOBARBITONE CAUTION — reduces GLUT1 mRNA expression ~30% (phenobarbitone-responsive
  element in SLC2A1 promoter) → worsens cerebral glucose delivery. Use only for acute SE bridging.
• VALPROATE: avoid long-term — some evidence VPA competitively reduces GLUT1 activity;
  use LEV/CLB as preferred adjuncts.
• FASTING ABSOLUTE CI — overnight fasts >4h → nadir blood glucose → maximal GLUT1 impairment
  → seizure trigger. Provide late-evening snack (KD-appropriate). Emergency glucose gel available.
• NEVER delay KD for "AED trials" — Glut1-DS is the one epilepsy where KD is first-line therapy,
  not reserved for drug-resistance. Earlier KD initiation → better cognitive outcome.
"""

import random
from datetime import datetime

SEED = 9188  # dashboard 188
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ───────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": (
            "SLC2A1 heterozygous de novo — classic GLUT1-DS "
            "(loss-of-function, haploinsufficiency)"
        ),
        "n": 21, "pct": 51,
        "category": "SLC2A1-het-de-novo-classic",
        "functional_class": "AD-GLUT1-haploinsufficiency-classic",
        "mechanism": (
            "Most prevalent class (~51%): de novo heterozygous SLC2A1 variants (missense, nonsense, "
            "frameshift, splice-site, or exonic deletion) causing GLUT1 haploinsufficiency. Single "
            "functional GLUT1 allele insufficient to meet cerebral glucose demand → hypoglycorrhachia "
            "(CSF glucose <2.2 mmol/L; CSF:plasma ratio <0.45) → epilepsy onset typically 1-4 years. "
            "Classic triad: epilepsy (multiple seizure types, worsened by fasting/exercise/morning), "
            "movement disorder (ataxia, spasticity, paroxysmal exercise-induced dyskinesia [PED]), "
            "intellectual disability (cognitive impairment correlates with duration of undiagnosed "
            "hypoglycorrhachia — earlier KD diagnosis → better cognitive outcome). ACMG: PVS1/PS2 + "
            "PM2 → Pathogenic. GLUT1 expression reduced to ~40-60% of normal in erythrocyte assay."
        ),
        "eeg_signature": (
            "Glut1-DS EEG: (1) 2.5-4 Hz generalised spike-and-wave (GSW) — most common, resembles "
            "childhood absence epilepsy (CAE) but ATYPICAL: worsens with fasting/morning, improves "
            "after meals/KD; (2) multi-focal IEDs (frontal > occipital) — reflecting diffuse energy "
            "failure without fixed structural focus; (3) hypsarrhythmia in severe neonatal-onset cases; "
            "PATHOGNOMONIC: EEG NORMALISATION after eating (improved glucose delivery) or after ketone "
            "loading. EEG during PED: typically NORMAL (movement disorder is metabolic, not ictal). "
            "Background: diffuse theta/delta slowing proportional to degree of cerebral glucose deficit. "
            "Sleep EEG: IED burden decreases in NREM (lower metabolic demand)."
        ),
        "clinical_note": (
            "Diagnostic algorithm: (1) Simultaneous fasting CSF + plasma glucose (>4h fast): "
            "CSF <2.2 mmol/L + ratio <0.45 → diagnostic threshold. (2) SLC2A1 gene sequencing "
            "(NGS panel/WES) + MLPA (exon deletions ~12%). (3) Erythrocyte glucose uptake assay "
            "(50% of normal confirms haploinsufficiency). (4) Lumbar puncture BEFORE KD initiation "
            "— KD normalises CSF glucose → false negative if LP after KD start. Common misdiagnoses: "
            "CAE (absent fasting/exercise pattern), JME, ADHD, tic disorder (for movement symptoms). "
            "Red flags: absence-like seizures + morning-predominance + movement disorder + fasting "
            "trigger = GLUT1-DS until proven otherwise. Start KD at diagnosis — do not wait."
        ),
    },
    {
        "etiology": (
            "SLC2A1 heterozygous familial AD — moderate Glut1-DS "
            "(parent-to-child, variable expressivity)"
        ),
        "n": 9, "pct": 22,
        "category": "SLC2A1-het-familial-AD",
        "functional_class": "AD-GLUT1-familial-variable",
        "mechanism": (
            "Familial heterozygous SLC2A1 — autosomal dominant transmission from an affected parent "
            "(penetrance ~90%, expressivity variable — some carriers have only PED without epilepsy; "
            "others have full triad). Same haploinsufficiency mechanism as de novo but often diagnosed "
            "when a parent's seizure/movement history prompts family cascade testing. Cognitive outcome "
            "sometimes better in familial cases — earlier diagnosis facilitated by family history. "
            "EEG improvement after meals noted by alert families before diagnosis confirmed. "
            "Common misclassification: parent diagnosed as 'JME' or 'epilepsy NOS' for years before "
            "Glut1-DS identified in child → parental re-evaluation + LP indicated."
        ),
        "eeg_signature": (
            "Familial Glut1-DS: same generalised 2.5-4 Hz GSW as de novo; often milder burden "
            "than de novo when parent has partial GLUT1 function (some missense alleles retain "
            "partial activity). Diurnal variation of IED burden (higher morning/fasting, lower "
            "post-prandial) is characteristic and can guide LP timing. Video-EEG during overnight "
            "fast vs 2h post-meal allows before/after comparison — diagnostic in clinic."
        ),
        "clinical_note": (
            "Cascade screening: when Glut1-DS confirmed in a child → test parent(s) with CSF glucose "
            "ratio (LP rarely needed — erythrocyte glucose uptake assay + SLC2A1 sequencing sufficient "
            "if variant known). Parents on long-term AEDs without KD — switch to KD if confirmed Glut1-DS. "
            "Genetic counselling: 50% risk per pregnancy. Prenatal/preimplantation genetic testing (PGT) "
            "available for familial SLC2A1 variants. Same contraindications apply to all carriers: "
            "NO methylxanthines even in 'mildly affected' parent."
        ),
    },
    {
        "etiology": (
            "SLC2A1 biallelic AR — severe neonatal/infantile Glut1-DS "
            "(homozygous or compound heterozygous)"
        ),
        "n": 4, "pct": 10,
        "category": "SLC2A1-biallelic-AR-severe",
        "functional_class": "AR-GLUT1-severe-biallelic",
        "mechanism": (
            "Rare biallelic SLC2A1 variants (homozygous or compound heterozygous) causing near-complete "
            "GLUT1 deficiency — erythrocyte glucose uptake <20% of normal (vs ~50% in heterozygous). "
            "Profound cerebral glucose deficiency from birth → neonatal/early infantile epilepsy + "
            "severe hypotonia + microcephaly (progressive) + profound intellectual disability. "
            "Severity correlates with residual GLUT1 activity: homozygous p.Arg333His → severe; "
            "compound-het missense/missense → intermediate severe. KD response still substantial "
            "but may not fully compensate for profound GLUT1 deficiency. Some require additional "
            "supplementation (triheptanoin). AR inheritance → 25% recurrence risk per pregnancy."
        ),
        "eeg_signature": (
            "Severe neonatal-onset: burst-suppression or hypsarrhythmia (West syndrome EEG pattern) "
            "— high-amplitude chaotic spikes/slow with intersuppression. Infantile spasms in ~40% "
            "of severe biallelic cases. As disease progresses: multifocal IEDs + background slowing "
            "that dramatically improves on KD within 4-8 weeks. CSF glucose may be unmeasurable "
            "(<1.0 mmol/L) in biallelic severe cases. Erythrocyte assay essential for quantification."
        ),
        "clinical_note": (
            "Differentiate from: STXBP1-DEE (no CSF glucose abnormality), CDKL5-DEE (no GLUT1 "
            "deficit), structural brain malformation (no hypoglycorrhachia), metabolic — GLUT1-DS "
            "UNIQUE: low CSF glucose WITHOUT hypoglycaemia. Emergency investigations: fasting CSF + "
            "plasma glucose, blood gas (lactate normal — not mitochondrial), plasma amino acids, "
            "urine organic acids. If CSF glucose unmeasurably low → emergency KD initiation before "
            "genetic confirmation (LP result is sufficient to start). Outcome better with very early "
            "KD (first weeks of life) despite severity."
        ),
    },
    {
        "etiology": (
            "SLC2A1 missense partial LOF — mild Glut1-DS "
            "(absence-predominant, cognitive-sparing phenotype)"
        ),
        "n": 5, "pct": 12,
        "category": "SLC2A1-missense-partial-LOF-mild",
        "functional_class": "AD-GLUT1-partial-LOF-mild",
        "mechanism": (
            "Missense SLC2A1 variants with residual GLUT1 transport activity (20-40% of normal in "
            "heterozygous state → effective transport capacity 60-70% of normal). Phenotype attenuated: "
            "predominantly absence-like seizures (may be sole manifestation), mild PED, normal or "
            "near-normal cognition. CSF:plasma ratio often 0.40-0.49 (borderline — easy to miss on "
            "spot CSF glucose). Erythrocyte assay shows ~65-75% of normal (vs 50% in classic). "
            "Often misdiagnosed as CAE (childhood absence epilepsy) and treated with ETH/VPA — "
            "ETH may have marginal effect; VPA suboptimal. KD response excellent even in mild cases. "
            "Identified increasingly on gene panels for 'treatment-resistant absence epilepsy'."
        ),
        "eeg_signature": (
            "Mild Glut1-DS: classic 3 Hz (sometimes 2.5-3.5 Hz) GSW, clinically and electrographically "
            "resembling CAE, but: (1) briefer absences; (2) morning/fasting predominance; (3) EEG "
            "response to glucose loading (IED burden decreases 30 min post-meal); (4) PED history "
            "in a 'CAE' patient → suspect Glut1-DS. No hypsarrhythmia. Background: often normal. "
            "Video-EEG during fasting vs fed state is most efficient diagnostic maneuver."
        ),
        "clinical_note": (
            "Diagnostic pitfall: CSF:plasma ratio 0.40-0.49 is often misread as 'borderline normal' "
            "and Glut1-DS dismissed. CORRECT threshold: ratio <0.45 = diagnostic when combined with "
            "clinical syndrome. Molecular confirmation (erythrocyte assay + SLC2A1 sequencing) "
            "essential for borderline LP results. In mild cases: MAD (Modified Atkins Diet) may be "
            "sufficient rather than strict classic 4:1 KD — equal efficacy, better adherence in "
            "older children. Do NOT use ETH as primary — it has no benefit in Glut1-DS absence. "
            "VPA competes with GLUT1 → avoid. LEV is safe adjunct if KD incompletely controls seizures."
        ),
    },
    {
        "etiology": (
            "SLC2A1 negative — clinical Glut1-DS phenocopy "
            "(hypoglycorrhachia, genetic-negative, MCT1/alternative transporter)"
        ),
        "n": 2, "pct": 5,
        "category": "SLC2A1-negative-phenocopy",
        "functional_class": "GLUT1-phenocopy-alternative-transporter",
        "mechanism": (
            "Clinical Glut1-DS phenotype (epilepsy + movement disorder + hypoglycorrhachia [CSF:plasma "
            "ratio <0.45]) with negative SLC2A1 sequencing and MLPA. Erythrocyte glucose uptake: "
            "variable (may be normal or mildly reduced — BBB GLUT1 may be preferentially affected). "
            "Candidate genes: MCT1 (monocarboxylate transporter 1, SLC16A1 — co-transports lactate/"
            "pyruvate/ketones across BBB; bi-directional phenotype with ketone transport also impaired), "
            "SLC45A1 (sucrose-proton transporter at choroid plexus). Some may represent deep intronic/"
            "regulatory SLC2A1 variants not detected by standard sequencing panels → WGS indicated. "
            "KD response: variable — MCT1 defects may paradoxically worsen on high-fat diet "
            "(ketone transport also impaired → ketoacidosis risk)."
        ),
        "eeg_signature": (
            "Phenocopy Glut1-DS: identical EEG pattern to classic Glut1-DS (2.5-4 Hz GSW, multi-focal "
            "IEDs, diurnal fasting/fed variation) — hypoglycorrhachia mechanism is the common substrate "
            "regardless of transporter gene involved. If MCT1 defect: EEG response to KD may be absent "
            "or partial (ketones cannot cross BBB efficiently either). Lactate also low in CSF (MCT1 "
            "transports lactate) — CSF lactate <1.0 mmol/L can indicate MCT1 co-deficiency."
        ),
        "clinical_note": (
            "Management approach: trial KD regardless of genetic confirmation if CSF:plasma ratio <0.45. "
            "Monitor response carefully — if seizures worsen on KD or blood ketones high but clinical "
            "unimproved → consider MCT1 defect. For MCT1-suspect: alternative fuel strategy needed. "
            "Whole genome sequencing + CSF metabolomics (amino acids, neurotransmitters, organic acids, "
            "lactate) → complete workup. SLC2A1 MLPA + deep intronic sequencing before declaring negative. "
            "Research referral: Glut1-DS Alliance registry accepts phenocopy cases for WGS research."
        ),
    },
]

# ── Patient Roster (N=41) ─────────────────────────────────────────────────────
def _make_patients():
    patients = []
    pid = 1
    specs = [
        # (cat, func_class, n, age_range_mo, onset_range_y, typical_control)
        ("SLC2A1-het-de-novo-classic",       "AD-GLUT1-haploinsufficiency-classic",      21, (12, 168), (1.0, 5.0),  "KD-controlled"),
        ("SLC2A1-het-familial-AD",           "AD-GLUT1-familial-variable",                9, (18, 240), (1.5, 8.0),  "KD-controlled"),
        ("SLC2A1-biallelic-AR-severe",       "AR-GLUT1-severe-biallelic",                 4, (2, 36),   (0.1, 0.8),  "drug-resistant"),
        ("SLC2A1-missense-partial-LOF-mild", "AD-GLUT1-partial-LOF-mild",                 5, (36, 216), (3.0, 12.0), "KD-controlled"),
        ("SLC2A1-negative-phenocopy",        "GLUT1-phenocopy-alternative-transporter",   2, (12, 96),  (1.0, 7.0),  "partially-controlled"),
    ]
    phases_map = {
        "SLC2A1-het-de-novo-classic":       ["diagnostic-workup","KD-initiation","KD-optimisation","KD-maintenance","school-age-stable"],
        "SLC2A1-het-familial-AD":           ["presymptomatic-cascade","KD-initiation","KD-optimisation","adolescent-MAD","adult-stable"],
        "SLC2A1-biallelic-AR-severe":       ["neonatal-crisis","KD-emergency","KD-stabilisation","progressive-disability"],
        "SLC2A1-missense-partial-LOF-mild": ["misdiagnosed-CAE","Glut1-DS-confirmed","MAD-initiation","MAD-stable"],
        "SLC2A1-negative-phenocopy":        ["hypoglycorrhachia-confirmed","KD-trial","WGS-pending"],
    }
    control_colors = {
        "KD-controlled":        "#198754",
        "drug-resistant":       "#dc3545",
        "partially-controlled": "#fd7e14",
    }
    for cat, func_class, n, age_range, onset_range, control_type in specs:
        phases = phases_map[cat]
        for _ in range(n):
            sex = random.choice(["M", "F"])
            age_months = random.randint(*age_range)
            onset_years = round(random.uniform(*onset_range), 1)
            phase = random.choice(phases)
            # CSF glucose ratio — key biomarker
            if cat == "SLC2A1-biallelic-AR-severe":
                csf_plasma_ratio = round(random.uniform(0.18, 0.34), 2)
                csf_glucose_mmol = round(random.uniform(0.8, 1.6), 1)
            elif cat == "SLC2A1-missense-partial-LOF-mild":
                csf_plasma_ratio = round(random.uniform(0.36, 0.44), 2)
                csf_glucose_mmol = round(random.uniform(1.8, 2.2), 1)
            elif cat == "SLC2A1-negative-phenocopy":
                csf_plasma_ratio = round(random.uniform(0.38, 0.44), 2)
                csf_glucose_mmol = round(random.uniform(1.9, 2.1), 1)
            else:
                csf_plasma_ratio = round(random.uniform(0.25, 0.43), 2)
                csf_glucose_mmol = round(random.uniform(1.3, 2.1), 1)

            # KD status
            on_kd = cat not in ("SLC2A1-biallelic-AR-severe",)
            kd_ratio = random.choice(["3:1", "4:1"]) if on_kd else None
            if cat == "SLC2A1-missense-partial-LOF-mild":
                kd_ratio = random.choice(["MAD", "2:1", "MCT"])
            beta_ohb = round(random.uniform(2.1, 4.2), 1) if on_kd else round(random.uniform(0.1, 0.4), 1)

            # PED present
            ped_present = cat != "SLC2A1-biallelic-AR-severe" and random.random() < 0.62

            # Treatment
            if on_kd:
                txs = [random.choice(["KD", "KD+LEV", "KD+CLB", "MAD", "MAD+LEV"])]
            else:
                txs = [random.choice(["LEV+CLB", "PB+LEV", "LEV+VGB"])]

            # Methylxanthine exposure history
            mxe_exposed = random.random() < 0.42  # many diagnosed after caffeine-triggered exacerbation

            patients.append({
                "id": f"GLUT1-{pid:03d}",
                "age_months": age_months,
                "sex": sex,
                "onset_years": onset_years,
                "category": cat,
                "functional_class": func_class,
                "disease_phase": phase,
                "current_treatment": txs[0],
                "seizure_control": control_type,
                "seizure_control_color": control_colors[control_type],
                "csf_plasma_ratio": csf_plasma_ratio,
                "csf_glucose_mmol": csf_glucose_mmol,
                "ped_present": ped_present,
                "on_kd": on_kd,
                "kd_ratio": kd_ratio,
                "beta_ohb_mmol": beta_ohb,
                "methylxanthine_exposure_hx": mxe_exposed,
                "slc2a1_variant": random.choice([
                    "p.Arg153Cys", "p.Glu146Lys", "p.Gly91Asp", "p.Arg333His",
                    "Exon3-del", "Exon5-7-del", "p.Ser66Phe", "p.Lys256Arg",
                    None,
                ]) if cat != "SLC2A1-negative-phenocopy" else None,
            })
            pid += 1
    return patients

PATIENTS = _make_patients()

# ── Seizure Types (4 core) ─────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Absence-like / Myoclonic-absence Seizures",
        "prevalence_pct": 85,
        "onset_age": "1-5 years (peak 2-4 years); can be first decade",
        "eeg_correlate": (
            "Generalised 2.5-4 Hz spike-and-wave discharge (GSW) — often atypical vs CAE: "
            "slower frequency (2.5-3 Hz), variable amplitude, may have poly-spike complexes "
            "preceding slow wave. PATHOGNOMONIC FINDING: GSW burden increases during fasting "
            "(test: EEG after 4h fast then 30-60 min post-meal/glucose → IED reduction confirms "
            "metabolic mechanism). GSW duration: 3-30 seconds (shorter than CAE). Clinical: "
            "brief staring + eyelid flutter ± mild automatisms; often myoclonic component "
            "(head nod, arm jerk). Hyperventilation may trigger (as in CAE) but fasting trigger "
            "is MORE SPECIFIC for Glut1-DS."
        ),
        "clinical_tip": (
            "KEY DIFFERENTIATOR FROM CAE: (1) Morning/fasting-predominant — 'worst seizures before "
            "breakfast'; (2) Improve after meals; (3) Associated movement disorder (ataxia/PED); "
            "(4) ETH response partial-at-best (not abolished as in CAE); (5) EEG diurnal variation "
            "on prolonged ambulatory. If 'CAE' patient not completely controlled on ETH or has fasting "
            "pattern → IMMEDIATE LP for CSF glucose before any other AED change. KD produces "
            "complete absence control in >80% within 3 months."
        ),
    },
    {
        "type": "Paroxysmal Exercise-Induced Dyskinesia (PED)",
        "prevalence_pct": 62,
        "onset_age": "2-15 years; often precedes or accompanies seizure onset",
        "eeg_correlate": (
            "EEG during PED: typically NORMAL (PED is a metabolic movement disorder, NOT epileptic). "
            "This is crucial — do NOT start AEDs for PED alone (they will not work). Post-exercise "
            "EEG within 30 min of prolonged exercise: may show mild theta slowing (cerebral glucose "
            "depletion) without ictal correlate. If EEG shows ictal activity during movement episodes "
            "→ reconsider diagnosis (could be focal motor seizures, not PED)."
        ),
        "clinical_tip": (
            "PED triggers: prolonged walking/running (5-20 min, exercise-intensity-dependent), "
            "climbing stairs, cycling, swimming. Relief: rest within 5-30 min. KD resolution: "
            "PED resolves in >85% within 3-6 months of KD initiation (ketones available during "
            "exercise without requiring GLUT1). Before KD: give glucose gel pre-exercise. "
            "Emergency: if PED severe → check blood glucose (rule out hypoglycaemia), give "
            "oral glucose/KD snack. PED in a child with ANY seizure type = Glut1-DS until LP rules out."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "prevalence_pct": 40,
        "onset_age": "Often later in disease course (secondary to untreated hypoglycorrhachia)",
        "eeg_correlate": (
            "Generalised paroxysmal fast activity (GPFA) → rhythmic clonic discharge → post-ictal "
            "attenuation. Background: diffuse slowing proportional to glucose deprivation state. "
            "Pre-ictal: often GSW burst preceding GTCS (absence → GTCS evolution, especially "
            "during morning/fasting). Post-ictal suppression prominent. On KD: GTCS frequency "
            "typically reduces >75% within 3 months."
        ),
        "clinical_tip": (
            "GTCS in Glut1-DS: manage acutely with IV glucose (0.5 g/kg) as FIRST MEASURE "
            "if LP not yet done (diagnostic + therapeutic). If on KD: IV glucose disrupts ketosis — "
            "acceptable in emergency but restart KD promptly. Chronic management: KD controls GTCS "
            "in >70%. Adjunct if breakthrough: LEV 20-60 mg/kg/day (safe with KD). Avoid VPA "
            "(GLUT1 competition). SUDEP risk: counsel family (GTCS frequency is key SUDEP driver)."
        ),
    },
    {
        "type": "Atonic Seizures / Drop Attacks",
        "prevalence_pct": 38,
        "onset_age": "2-8 years; more common in biallelic AR severe cases",
        "eeg_correlate": (
            "EEG during atonic drop: brief (<3s) generalised spike-wave or polyspike-wave "
            "followed by sudden atonia and fall. High-amplitude generalised slow-wave complex "
            "immediately preceding fall visible on EEG. Background between drops: diffuse slow. "
            "Atonic seizures in Glut1-DS: typically respond well to KD (within 2-4 months). "
            "Corpus callosotomy may be considered for drug-resistant drop attacks if KD fails — "
            "but KD must be optimised first (3+ months at target ketosis) before surgical referral."
        ),
        "clinical_tip": (
            "Protective helmet MANDATORY for patients with drop attacks — head trauma risk. "
            "KD is the treatment of choice (Level A evidence). Na-channel blockers (CBZ, OXC, PHT, "
            "LCM) NOT recommended — may worsen atonic/myoclonic components. Rufinamide Level-C "
            "evidence for refractory drop attacks as adjunct if KD insufficient. Corpus callosotomy "
            "as palliative for severely refractory drop attacks (after 2+ year optimised KD trial). "
            "CLB adjunct (0.25-0.5 mg/kg/day) can reduce drop attack frequency as KD bridge."
        ),
    },
]

# ── Seizure Triggers (8 core) ─────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fasting / prolonged meal gap (>4 hours)",
        "prevalence_pct": 92,
        "mechanism": (
            "Reduced blood glucose during fast → maximally stresses GLUT1-deficient transport → "
            "acute hypoglycorrhachia worsening. Overnight fast (10-12h) is the highest-risk period. "
            "Morning seizures before breakfast are the hallmark. CSF glucose reaches nadir 4-6h "
            "into fast. KD maintains ketones during fasting → eliminates this trigger."
        ),
        "management": (
            "Late-evening KD snack (11 PM) mandatory. Never skip breakfast. Emergency glucose gel "
            "for seizure clusters (buccal glucose) if not on KD. On KD: ketone supply maintained "
            "during fast — this trigger resolved in >90% of KD-adherent patients. School: late snack "
            "before PE/sport. Hospital fasting protocol: IV dextrose + ketone drink pre-procedure."
        ),
    },
    {
        "trigger": "Morning / dawn (nadir blood glucose after overnight fast)",
        "prevalence_pct": 88,
        "mechanism": (
            "Cortisol surge at dawn (03:00-06:00) + post-overnight-fast glucose nadir → "
            "GLUT1-limited transport at its worst → peak seizure frequency in the hour before "
            "breakfast and during first-morning activities. Parents report 'wakes up having seizures'. "
            "EEG ambulatory: IED burden peaks 06:00-09:00 pre-breakfast, troughs 10:00-12:00."
        ),
        "management": (
            "Early breakfast mandatory (within 30 min of waking). Late-evening carbohydrate/KD snack. "
            "KD completely abolishes morning trigger in >85% (ketones maintained overnight). "
            "Non-KD patients: glucose drink immediately on waking (10 g glucose) as bridge measure. "
            "School: no morning PE before eating. Medical letter for 'must eat before PE' available."
        ),
    },
    {
        "trigger": "Methylxanthines (caffeine, theophylline, theobromine)",
        "prevalence_pct": 62,
        "mechanism": (
            "Competitive GLUT1 inhibition: caffeine/theophylline bind glucose-binding pocket of "
            "GLUT1 transmembrane domain → reversible competitive inhibition → acutely reduces "
            "glucose transport → seizure exacerbation within 30-120 min of ingestion. Sources: "
            "coffee (80-120 mg), cola (40-50 mg), energy drinks (80-250 mg), chocolate (theobromine "
            "5-30 mg/serving), tea (40-60 mg), asthma medications (aminophylline/theophylline IV)."
        ),
        "management": (
            "ABSOLUTE PROHIBITION — all methylxanthine sources: coffee, tea, cola, energy drinks, "
            "dark/milk chocolate, cocoa, guarana, mate. Decaffeinated products: STILL contain "
            "~5-10 mg caffeine — HIGH RISK. Medical ID bracelet: 'No caffeine/theophylline'. "
            "School nurse letter: no chocolate in class. A&E: IV aminophylline/theophylline "
            "CONTRAINDICATED — use alternative bronchodilator (salbutamol). Document allergy in EMR."
        ),
    },
    {
        "trigger": "Prolonged exercise / physical exertion (>10 minutes sustained)",
        "prevalence_pct": 75,
        "mechanism": (
            "Sustained exercise → skeletal muscle glucose consumption increases 10-20× → blood "
            "glucose falls → GLUT1-mediated BBB transport worsens → acute cerebral glucose "
            "deficit → seizure + PED. Exercise glucose demand cannot be met by deficient GLUT1. "
            "On KD: ketones are mobilised during exercise (adipose lipolysis → β-OHB) → ketone "
            "BBB transport via MCT1 (intact) compensates → exercise tolerance restored."
        ),
        "management": (
            "Before exercise (non-KD): 15-20g glucose 15 min pre-exercise. On KD: KD snack + "
            "oral glucose gel available. Short exercise (<10 min) generally tolerated even off KD. "
            "Competitive sport: possible on KD — many Glut1-DS patients engage fully in sport on "
            "well-established KD. Avoid empty-stomach exercise. Glucose monitoring device useful "
            "for exercise planning. PED resolves on KD — no specific exercise restriction on KD."
        ),
    },
    {
        "trigger": "Fever / intercurrent illness",
        "prevalence_pct": 68,
        "mechanism": (
            "Fever → increased metabolic rate → increased cerebral glucose demand → exacerbates "
            "GLUT1-limited supply. Also: fever → reduced appetite/intake → relative fasting → "
            "compounding glucose deficit. Acute illness can disrupt KD (nausea/vomiting → breaks "
            "ketosis → abrupt seizure increase within 24-48h). Sick-day management critical."
        ),
        "management": (
            "Sick-day plan: (1) Maintain KD foods — KD formula/tube feed if unable to eat. "
            "(2) IV ketone drink (KetoCal/Liquigen) if vomiting. (3) Paracetamol for fever — safe. "
            "(4) Avoid ibuprofen on empty stomach on KD. (5) Hospital letter: 'KD patient — do NOT "
            "give glucose-containing IV fluids unless seizure emergency'. (6) Emergency IV glucose "
            "acceptable in seizure emergency but disrupts ketosis — restart KD promptly. "
            "(7) Theophylline for asthma/bronchospasm CONTRAINDICATED — use salbutamol/ipratropium."
        ),
    },
    {
        "trigger": "Missed KD / carbohydrate transgression",
        "prevalence_pct": 78,
        "mechanism": (
            "Break in ketosis (KD miss >24h or significant carbohydrate ingestion) → rapid "
            "fall in blood ketones (β-OHB half-life ~2h) → brain reverts to GLUT1-deficient "
            "glucose transport → seizure breakthrough within 12-48h of KD break. Even small "
            "carbohydrate transgressions (birthday cake, school lunch mistake) can precipitate "
            "cluster seizures. This explains why KD compliance is paramount — near-zero carbs."
        ),
        "management": (
            "Strict KD adherence education for patient, family, school staff, grandparents. "
            "Medical letter for school, parties, restaurants. Emergency plan for accidental "
            "carbohydrate ingestion: immediate KD snack + ketone drink (Ketocal). Monitor blood "
            "ketones (target β-OHB 2-4 mmol/L) — fingerprick ketone meter at home. If ketones "
            "fall <1.5 mmol/L: call dietitian. Seizure cluster after carb transgression: "
            "CLB rescue + re-establish ketosis — do NOT routinely give glucose."
        ),
    },
    {
        "trigger": "Sleep deprivation / irregular sleep",
        "prevalence_pct": 45,
        "mechanism": (
            "Sleep deprivation → reduced slow-wave sleep → increased sympathetic tone → "
            "raised cortisol/adrenaline → glucose counter-regulation reduces relative "
            "CNS glucose availability. Combined with circadian nadir of blood glucose during "
            "sleep → compounding effect. Less specific than fasting/methylxanthine triggers "
            "in Glut1-DS but clinically relevant."
        ),
        "management": (
            "Regular sleep routine. Adequate sleep duration (8-10h for school-age). "
            "Late-evening KD snack prevents overnight glucose nadir. Morning seizures "
            "(sleep deprivation + fasting combined) — worst risk window. EEG often shows "
            "increased IED burden in sleep-deprived ambulatory recordings."
        ),
    },
    {
        "trigger": "High-carbohydrate meal / glucose spike (non-KD patients)",
        "prevalence_pct": 38,
        "mechanism": (
            "Paradoxical post-prandial glucose spike → hyperglycaemia → reactive hypoglycaemia "
            "1-2h later (glucose counter-regulation) → GLUT1-limited nadir → seizure exacerbation. "
            "More relevant in non-KD or MAD patients with some carbohydrate intake. In strict KD: "
            "this trigger eliminated. EEG shows IED increase 90-120 min post high-GI meal "
            "in some Glut1-DS patients."
        ),
        "management": (
            "Low glycaemic index diet even if not on strict KD (brown rice, oats, legumes vs "
            "sugary foods). Avoid sugary drinks (high GI). If on MAD: prefer low-GI carbohydrate "
            "sources within daily carb allowance (20-30g net). Regular small meals (3h frequency) "
            "vs large infrequent meals. Blood glucose monitoring 90 min post-meals in unstable patients."
        ),
    },
]

# ── Treatments (8 AEDs / Interventions) ───────────────────────────────────────
TREATMENTS = [
    {
        "name": "Ketogenic Diet (4:1 or 3:1 classic KD)",
        "evidence": "Level A",
        "line": "FIRST-LINE — Precision Therapy (not drug-resistant reserve)",
        "dose": (
            "Classical KD: 4:1 (fat:protein+carb) or 3:1 ratio. Fat 80-90% kcal, "
            "carbohydrate <5-10g/day net, protein 1-2g/kg/day. Dietitian-calculated per "
            "patient based on age/weight/activity. Target β-OHB 2-4 mmol/L (fingerprick "
            "ketone meter). Initiation: hospital (or day-patient) admission for monitoring. "
            "Wean and adjustments q3M with dietitian + neurologist."
        ),
        "moa": (
            "Provides β-hydroxybutyrate (β-OHB) + acetoacetate as alternative CNS fuel. "
            "Ketones cross BBB via MCT1 (intact in GLUT1-DS) → TCA cycle → ATP production. "
            "BYPASSES defective GLUT1 completely — replaces glucose as primary neuronal fuel."
        ),
        "efficacy": (
            ">90% seizure-free or >50% reduction (Level A, Verrotti 2012 Epilepsia). "
            "Absence seizures: >80% complete control. PED: resolution in >85%. "
            "Movement disorder: significant improvement 60-70%. Cognition: trajectory "
            "improves vs untreated — earlier initiation → better outcome."
        ),
        "safety": (
            "Short-term: nausea/vomiting, constipation, hypoglycaemia (first days). "
            "Long-term: dyslipidaemia (monitor lipids q3M), nephrolithiasis (citrate supplement), "
            "growth impact (dietitian monitoring q3M), selenium/carnitine depletion (supplement), "
            "acidosis (bicarbonate supplement if pH <7.35). Bone density (DXA q2y). "
            "BENEFITS outweigh risks when initiated early — life-changing therapy."
        ),
        "monitoring": "β-OHB q1-4wk · Lipids q3M · Growth z-score q3M · Selenium/carnitine q6M · DXA q2y",
    },
    {
        "name": "Modified Atkins Diet (MAD)",
        "evidence": "Level B",
        "line": "Alternative precision therapy — older children/adults; mild Glut1-DS",
        "dose": (
            "Net carbohydrate allowance: 10-20g/day (child) or 15-30g/day (adult). "
            "Fat ad libitum (typically 60-70% kcal). No calorie or protein restriction. "
            "Less restrictive than classic KD — easier adherence in school-age and older. "
            "Target β-OHB 1.5-3 mmol/L (slightly lower than classic KD)."
        ),
        "moa": (
            "Same mechanism as classic KD: ketogenesis → β-OHB BBB transport via MCT1 → "
            "alternative CNS fuel. Lower ketone levels than 4:1 KD but sufficient for Glut1-DS "
            "(partial GLUT1 function means lower supplementary ketone requirement than severe)."
        ),
        "efficacy": (
            "In Glut1-DS: comparable seizure control to classic KD in mild-moderate phenotype "
            "(Klepper 2008 Epilepsia). Better adherence → real-world effectiveness similar or "
            "superior. PED: resolves in >75%. Absence control: >70%."
        ),
        "safety": (
            "Better lipid profile than 4:1 KD. Fewer GI side effects. Less growth restriction. "
            "Constipation less common. Preferred for adolescents/adults who cannot maintain 4:1. "
            "Monitor: same lipid/growth/selenium/carnitine as classic KD."
        ),
        "monitoring": "β-OHB q2-4wk · Lipids q3M · Growth q3M · Selenium/carnitine q6M",
    },
    {
        "name": "MCT (Medium-Chain Triglyceride) Diet",
        "evidence": "Level B",
        "line": "KD variant — younger children; tube-fed patients",
        "dose": (
            "MCT oil (C8:C10 ratio 60:40) 30-60% kcal from MCT. Lower fat:carb ratio than "
            "classic KD but MCTs produce more ketones per gram than LCT → equivalent ketosis "
            "with less dietary fat → more varied diet possible. Sachets (Liquigen/MCT oil) "
            "added to foods or via nasogastric/PEG tube."
        ),
        "moa": (
            "MCT (C8 caprylic + C10 capric acid) preferentially metabolised in liver → "
            "β-oxidation → ketogenesis without requiring glucose — direct ketone production "
            "independent of GLUT1. Also: C8 directly crosses BBB and is metabolised to "
            "acetyl-CoA in astrocytes → supports neuronal energy independently."
        ),
        "efficacy": "Comparable seizure control to classic KD in GLUT1-DS. Better palatability for infants.",
        "safety": (
            "GI side effects higher (diarrhoea, nausea) — dose-titration essential. "
            "Monitor: same as classic KD. Avoid in patients with intestinal malabsorption. "
            "Preferred for tube-fed patients (Liquigen via NG/PEG)."
        ),
        "monitoring": "β-OHB q2-4wk · GI tolerance · Lipids q3M · Weight q4wk (infants)",
    },
    {
        "name": "Levetiracetam (LEV)",
        "evidence": "Level C",
        "line": "Adjunct — breakthrough seizures on KD / bridge before KD initiation",
        "dose": (
            "20-60 mg/kg/day PO divided BD-TDS. IV loading 60 mg/kg in emergency. "
            "Max: 3000 mg/day (adults). Titrate: 10-20 mg/kg/day increments q2wk. "
            "Levels: not routinely required (efficacy not well correlated with plasma level). "
            "Renal dose adjustment if eGFR <80 ml/min."
        ),
        "moa": (
            "SV2A (synaptic vesicle protein 2A) binding → reduces neurotransmitter release. "
            "DOES NOT affect GLUT1 transport — safe adjunct. Broad-spectrum — covers "
            "absence, myoclonic, GTCS. No known GLUT1 interaction."
        ),
        "efficacy": "Variable in Glut1-DS alone; useful adjunct on KD for breakthrough seizures (~30% additional reduction).",
        "safety": (
            "Common: behavioural effects (irritability, aggression, 'levetiracetam rage') especially "
            "in young children — pyridoxine B6 10-20 mg/day may attenuate. Safe in liver/kidney "
            "(primarily renally cleared). No interaction with GLUT1. No teratogenicity data concern "
            "for males; female adolescents: NICE NG217 pregnancy precautions."
        ),
        "monitoring": "Plasma level if suspected non-compliance · Renal function q6M if renally impaired",
    },
    {
        "name": "Clobazam (CLB)",
        "evidence": "Level C",
        "line": "Adjunct — absence/drop attacks; rescue for cluster seizures",
        "dose": (
            "0.1-0.3 mg/kg/day PO OD-BD. Max 1 mg/kg/day (children), 40 mg/day (adults). "
            "Rescue dose: 0.3-0.5 mg/kg PO/buccal during seizure cluster. "
            "Norclobazam TDM (active metabolite): 50-300 µg/L (routine if on enzyme inducers)."
        ),
        "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine; less sedating than 1,4-BDZ). Broad-spectrum.",
        "efficacy": "Absence and drop attack reduction as adjunct to KD. Cluster rescue effective.",
        "safety": (
            "Sedation (less than clonazepam). Tolerance (rotate off for 4-6 weeks if efficacy wanes). "
            "Withdrawal seizures if abrupt stop — taper over 4-8 weeks. Behavioural: disinhibition "
            "in children. No direct GLUT1 interaction. Safe with KD."
        ),
        "monitoring": "Norclobazam TDM q6M · Sedation assessment · Withdrawal protocol if stopping",
    },
    {
        "name": "Triheptanoin (C7 odd-chain fatty acid)",
        "evidence": "Level C",
        "line": "Investigational — refractory Glut1-DS; severe biallelic cases",
        "dose": (
            "1-2 g/kg/day PO with meals (divided TDS-QDS). Max 35% total kcal. "
            "Mixed with foods/formula. Research doses variable: 1g/kg (pilot studies). "
            "Available as UX007 in clinical trials. Not routinely licensed — specialist centre only."
        ),
        "moa": (
            "C7 (heptanoic acid) → hepatic β-oxidation → produces C5 ketones (β-hydroxypentanoate "
            "+ β-ketopentanoate) as anaplerotic TCA substrates. C5 ketones may cross BBB even in "
            "MCT1-impaired states. Also: direct anaplerosis of TCA cycle intermediates (oxaloacetate, "
            "succinyl-CoA) → bypasses BOTH GLUT1 and standard ketone transport pathways. "
            "Particularly useful if MCT1 phenocopy co-exists."
        ),
        "efficacy": "Pilot data: 50% seizure reduction in KD-refractory Glut1-DS (Mochel 2016, J Inherit Metab Dis). PED improvement noted.",
        "safety": "GI effects (nausea, diarrhoea). Monitor liver enzymes. Specialist centre only. Not for first-line.",
        "monitoring": "LFTs q3M · Plasma C5 ketones (if assay available) · Clinical response diary",
    },
    {
        "name": "Phenobarbitone (PB)",
        "evidence": "Level C",
        "line": "CAUTION — bridging/acute SE only; avoid long-term",
        "dose": (
            "Loading SE: 20 mg/kg IV (max 1000 mg) at 1 mg/kg/min. "
            "Maintenance (short-term only): 3-5 mg/kg/day PO OD. "
            "Target level: 65-170 µmol/L (15-40 µg/mL) — TDM-guided. "
            "Do NOT use as chronic AED in Glut1-DS — GLUT1 downregulation risk."
        ),
        "moa": "GABA-A positive allosteric modulator. Broad-spectrum. BUT: phenobarbitone-responsive element in SLC2A1 promoter → chronic PB reduces GLUT1 mRNA ~30% → worsens cerebral glucose delivery.",
        "efficacy": "Effective for acute SE bridging. WORSENS Glut1-DS if used long-term.",
        "safety": (
            "CAUTION: long-term PB reduces GLUT1 expression → worsens hypoglycorrhachia. "
            "Use only for acute SE (≤48h course) while KD initiated. Cognitive: PB has negative "
            "cognitive effects in children (avoid in developing brain). Sedation. No hepatotoxicity "
            "in Glut1-DS (unlike POLG). Withdrawal: must taper slowly."
        ),
        "monitoring": "PB level q24h in SE · Limit to ≤48h if possible · Plan KD initiation concurrently",
    },
    {
        "name": "Glucose supplementation (acute / emergency)",
        "evidence": "Level C",
        "line": "Acute/emergency only (pre-KD or seizure emergency) — NOT chronic treatment",
        "dose": (
            "Acute seizure cluster / SE: 0.5 g/kg IV glucose (e.g., 2 mL/kg 25% dextrose). "
            "Oral pre-KD bridge: 10-20 g glucose drink q3-4h (prevents fasting trigger). "
            "Buccal glucose gel: 10g (GlucoGel) for cluster seizure rescue pre-KD. "
            "On established KD: glucose is CONTRAINDICATED as a routine measure (breaks ketosis)."
        ),
        "moa": "Directly supplies glucose substrate → increases blood glucose → improves GLUT1-mediated CSF glucose delivery. Temporary — does NOT fix GLUT1 deficiency. KD is curative; glucose is a bridge.",
        "efficacy": "Rapid seizure cessation in acute hypoglycorrhachia-triggered seizures. TEMPORARY — do not rely on as definitive treatment.",
        "safety": (
            "Acute: safe. Chronic: high-carb diet WITHOUT KD is NOT a treatment — brain glucose "
            "delivery remains impaired despite blood glucose rise (GLUT1 deficiency limits "
            "BBB transport regardless of plasma concentration above ~5 mmol/L). KD initiation "
            "is the definitive treatment — glucose is a pre-KD bridge only."
        ),
        "monitoring": "Blood glucose monitoring · Transition to KD at earliest opportunity",
    },
]

# ── Contraindications (4) ──────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "name": "Methylxanthines — ALL forms (caffeine / theophylline / aminophylline / theobromine)",
        "severity": "ABSOLUTE CONTRAINDICATION",
        "color": "#dc3545",
        "mechanism": (
            "Competitive GLUT1 inhibition — methylxanthines bind the glucose-binding domain of "
            "GLUT1 protein → reversible but significant reduction in glucose transport across "
            "BBB → acute hypoglycorrhachia worsening → seizure exacerbation + PED. Sources: "
            "coffee (espresso 80-120 mg caffeine), cola (Pepsi/Coke 40-50 mg), energy drinks "
            "(80-250 mg), dark chocolate/cocoa (theobromine 30-200 mg + caffeine 10-60 mg), "
            "milk chocolate (theobromine 50-100 mg), green/black tea (40-60 mg), guarana, mate. "
            "Medical: aminophylline IV (asthma) + theophylline (COPD) — must use salbutamol/ipratropium. "
            "Even decaffeinated: 5-10 mg caffeine residual — HIGH RISK in Glut1-DS."
        ),
        "clinical_action": (
            "Document as DRUG ALLERGY in all medical records. Medical ID bracelet. "
            "Letters to: school nurse (no chocolate/cocoa products), GP (no theophylline), "
            "A&E (no aminophylline). Emergency protocol: salbutamol (not theophylline) for "
            "bronchospasm. Dietitian education at KD initiation."
        ),
    },
    {
        "name": "Phenobarbitone (long-term chronic use)",
        "severity": "HIGH RISK — avoid chronic",
        "color": "#fd7e14",
        "mechanism": (
            "Phenobarbitone-response element (PBRE) in SLC2A1 gene promoter — chronic PB "
            "activates PBRE → transcriptional repression of SLC2A1 → reduced GLUT1 mRNA "
            "and protein expression (up to ~30% reduction in animal models). In Glut1-DS, "
            "already haploinsufficient GLUT1 → additional PB-induced downregulation → "
            "worsened hypoglycorrhachia → paradoxical seizure worsening. Short-term (≤48h "
            "SE bridging) acceptable — chronic maintenance use contraindicated."
        ),
        "clinical_action": (
            "Restrict PB to acute SE management only (≤48h). Initiate KD concurrently. "
            "If child was on long-term PB before Glut1-DS diagnosis → plan supervised taper "
            "as KD establishes. Never initiate PB as chronic maintenance AED in confirmed "
            "Glut1-DS. Document in notes: 'Glut1-DS — no long-term phenobarbitone.'"
        ),
    },
    {
        "name": "Valproate / sodium valproate (VPA)",
        "severity": "HIGH RISK — avoid; MHRA reproductive risk adolescent females",
        "color": "#fd7e14",
        "mechanism": (
            "VPA competes with glucose for GLUT1-mediated transport (some evidence of competitive "
            "inhibition at therapeutic concentrations). Also: VPA → hyperammonaemia → altered CNS "
            "energy metabolism. Additionally: VPA is reproductive-risk drug (MHRA PREVENT 2024) — "
            "all adolescent females require pregnancy prevention programme. In Glut1-DS specifically: "
            "some families report VPA triggering seizure exacerbation (GLUT1 competition hypothesis). "
            "Multiple safer alternatives available (LEV, CLB, KD adjuncts) → avoid VPA."
        ),
        "clinical_action": (
            "Prefer LEV/CLB as adjuncts. If VPA was started before Glut1-DS diagnosis — "
            "plan supervised switch to LEV. MHRA PREVENT 2024: no VPA for females of "
            "childbearing potential without pregnancy prevention programme and specialist sign-off. "
            "Document in notes: 'Glut1-DS — prefer VPA-free regimen.'"
        ),
    },
    {
        "name": "Fasting / glucose deprivation (>4h fast, including perioperative fasting)",
        "severity": "ABSOLUTE CONTRAINDICATION — mandatory sick-day and surgical protocol",
        "color": "#dc3545",
        "mechanism": (
            "Prolonged fasting → blood glucose falls → GLUT1-limited BBB transport at maximum "
            "deficit → acute hypoglycorrhachia → seizure cluster/SE risk. Perioperative fasting "
            "is highest-risk scenario: prolonged NBM (nil-by-mouth) + anaesthetic agents altering "
            "glucose metabolism → multiple compounding factors → intra/post-operative SE. "
            "Hospital staff must be educated: standard NBM protocols unsafe for Glut1-DS patients."
        ),
        "clinical_action": (
            "Surgical protocol: coordinate with anaesthetics. Minimise NBM time. "
            "IV 10% dextrose (if not on KD) or IV KD formula (if on KD) during NBM period. "
            "Morning surgery preferred (shorter fast). Wake up with oral KD snack/glucose immediately. "
            "Medical letter for all planned procedures. Emergency card: 'Glut1-DS — fasting "
            "precipitates seizures — IV glucose if oral intake impossible.' Sick-day plan with "
            "KD-appropriate fluids for acute illness/vomiting."
        ),
    },
]

# ── Monitoring (8 items) ──────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "CSF:plasma glucose ratio (LP)",
        "frequency": "Once at diagnosis (before KD); repeat if KD changed or phenotype unclear",
        "rationale": (
            "Diagnostic: ratio <0.45 (with plasma >4 mmol/L fasting) confirms GLUT1 deficiency. "
            "CSF glucose <2.2 mmol/L is threshold. LP MUST be done before KD (KD normalises CSF "
            "glucose → false negative if LP after KD). Repeat LP only if: clinical deterioration "
            "on KD, uncertain diagnosis, or weaning KD."
        ),
    },
    {
        "item": "Blood β-hydroxybutyrate (β-OHB) ketone levels",
        "frequency": "Daily (first 4 weeks KD initiation) → weekly → q4wk on stable KD",
        "rationale": (
            "KD efficacy monitoring. Target: β-OHB 2-4 mmol/L (fingerprick ketone meter at home). "
            "<1.5 mmol/L → inadequate ketosis → review diet compliance, ratio, calorie intake. "
            ">5 mmol/L → assess for metabolic acidosis (check blood gas). Morning fasting β-OHB "
            "most representative. Document levels on seizure/KD diary."
        ),
    },
    {
        "item": "Fasting lipid panel (total cholesterol, LDL, HDL, TG)",
        "frequency": "Baseline + q3M (first year KD) → q6M if stable",
        "rationale": (
            "High-fat KD raises total cholesterol (primarily LDL) in ~30% → monitor for "
            "hypercholesterolaemia (rare clinically significant events but document). TG: "
            "usually falls on KD (reduced carbohydrate). If LDL >4.0 mmol/L: dietitian "
            "review (adjust fat composition — more MUFA/PUFA). If LDL >5.0 mmol/L: "
            "consider lower-ratio KD or MAD; rarely: statin (specialist decision)."
        ),
    },
    {
        "item": "Growth monitoring (height, weight, BMI z-scores)",
        "frequency": "q3M (children on KD) — dietitian review at each visit",
        "rationale": (
            "KD calorie restriction risk: growth faltering especially in young children. "
            "KD caloric prescription: adequate for growth (REE + activity factor + growth allowance). "
            "If height z-score falls >1 SD: increase protein/total calories (adjust KD ratio 3:1). "
            "Pubertal delay reported on long-term KD — monitor Tanner staging annually from age 8."
        ),
    },
    {
        "item": "Neuropsychological assessment",
        "frequency": "q12M (or at school transition)",
        "rationale": (
            "Cognitive trajectory: KD improves/stabilises cognition in most Glut1-DS patients. "
            "Domains: language, attention, memory, executive function, academic achievement. "
            "Earlier KD initiation → less cognitive deficit at school entry. WISC/Bayley-III for "
            "age. EEG normalisation correlates with cognitive improvement. IEP (individual "
            "education plan) support for children with intellectual disability."
        ),
    },
    {
        "item": "EEG (±ambulatory or long-term video-EEG)",
        "frequency": "q6M (first 2 years) → q12M if stable; ambulatory for fasting/fed comparison",
        "rationale": (
            "Track IED burden reduction on KD. EEG normalisation is a secondary outcome target. "
            "Ambulatory 48h EEG with diary: captures diurnal variation (fasting vs fed, morning "
            "vs afternoon) — quantitative IED burden. On KD: EEG improvement within 3-6 months "
            "expected. Lack of EEG improvement at 6M → review ketosis level, consider KD ratio "
            "increase or specialist referral."
        ),
    },
    {
        "item": "Selenium, carnitine, vitamin D, bone health (DXA)",
        "frequency": "Selenium/carnitine q6M · Vitamin D q6M · DXA q2y (on KD >2 years)",
        "rationale": (
            "KD depletion risks: selenium (antioxidant, cardiac) → supplement if low; "
            "carnitine (fat metabolism) → supplement if plasma free carnitine <20 µmol/L; "
            "vitamin D (low on KD due to fat restriction of dairy sometimes; supplement 800-1000 IU); "
            "bone density: KD associated with osteopenia (low bone turnover + acidosis) → DXA "
            "at 2y KD; add calcium supplement + vitamin D. Renal: urine calcium/creatinine ratio "
            "q6M (nephrolithiasis risk → potassium citrate supplement if ratio >0.6)."
        ),
    },
    {
        "item": "SLC2A1 erythrocyte glucose uptake assay",
        "frequency": "Once (at diagnosis, if variant uncertain or LP borderline)",
        "rationale": (
            "Functional GLUT1 assay: erythrocytes express GLUT1 as their only glucose transporter → "
            "erythrocyte glucose uptake (3-O-methylglucose) reflects GLUT1 function. Heterozygous: "
            "~50% of normal. Biallelic severe: <20% of normal. Mild partial LOF: 55-70% of normal. "
            "Confirmatory when: LP borderline (ratio 0.40-0.49), VUS on SLC2A1 sequencing, or "
            "family member with uncertain carrier status. Not needed if SLC2A1 variant confirmed "
            "pathogenic + LP clearly diagnostic."
        ),
    },
]

# ── Lifecycle Windows (6) ─────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "phase": "Pre-diagnosis / genetic workup",
        "window": "Birth to seizure onset (variable)",
        "focus": (
            "Presymptomatic period — no clinical manifestation. If familial SLC2A1 known: "
            "proactive LP in infancy (before seizure onset if possible). Otherwise: first "
            "presentation with seizures triggers Glut1-DS workup. Goal: diagnose before "
            "prolonged hypoglycorrhachia causes irreversible cognitive injury."
        ),
    },
    {
        "phase": "Infancy / toddler — KD initiation",
        "window": "0-3 years (typical seizure onset 1-3 years)",
        "focus": (
            "LP at diagnosis (before KD). Emergency KD initiation (inpatient or day-patient). "
            "LP, SLC2A1 sequencing, erythrocyte assay. MCT or 4:1 KD per age/feeding method. "
            "Breastfeeding: continue + add MCT supplement. Formula-fed: KetoCal Infant. "
            "Rapid seizure reduction expected within 4-8 weeks. PED resolves within 3-6 months. "
            "Parental education: methylxanthine prohibition, fasting protocol, ketone monitoring."
        ),
    },
    {
        "phase": "Early childhood — KD optimisation",
        "window": "3-7 years",
        "focus": (
            "KD ratio optimisation (3:1 vs 4:1 — balance efficacy vs growth). "
            "Neuropsych assessment: school readiness. IEP initiation if needed. "
            "EEG: track IED normalisation. Growth monitoring. Pubertal onset "
            "anticipation from age 6 (early puberty possible on KD). "
            "PED: confirm resolution. SUDEP counselling (GTCS frequency)."
        ),
    },
    {
        "phase": "School age — KD maintenance / MAD transition",
        "window": "7-13 years",
        "focus": (
            "Some patients transition from 4:1 KD to MAD at school entry (easier adherence, "
            "school lunch compatibility). School letter: no caffeine/chocolate. PE teacher "
            "education on PED and fasting triggers. Pre-exercise snack protocol. "
            "Annual neuropsych. DXA if >2y on KD. Social challenges of strict diet — "
            "psychologist referral. Pubertal monitoring. Drug review: can LEV/CLB be weaned "
            "if seizure-free >2y on KD?"
        ),
    },
    {
        "phase": "Adolescence — diet liberalisation / transition",
        "window": "13-18 years",
        "focus": (
            "KD adherence challenges (social eating, parties, alcohol — beer high-carb). "
            "MAD preferred for adolescents. Driving assessment (seizure-free criteria varies "
            "by jurisdiction — typically 12 months seizure-free). Transition to adult neurology. "
            "Pregnancy counselling: SLC2A1 AD 50% inheritance risk. VPA PREVENT programme if "
            "on VPA (MHRA 2024). Discussion: can KD ever be weaned? (Evidence: lifetime KD "
            "recommended; many patients relapse on KD weaning — individual decision.)"
        ),
    },
    {
        "phase": "Adulthood — MAD / lifetime management",
        "window": "18+ years",
        "focus": (
            "MAD or 2:1 KD — most adults find 4:1 unsustainable. Some patients: excellent "
            "seizure control on MAD indefinitely. Employment: sedentary jobs favoured (driving "
            "restrictions). Movement disorder: ataxia may persist even on KD (structural damage "
            "from delayed diagnosis). Pregnancy: maintain KD/MAD throughout — KD safe in "
            "pregnancy (fetal uses maternal ketones). Genetic counselling for offspring risk. "
            "Annual neurology review. Methylxanthine prohibition lifelong."
        ),
    },
]

# ── Key Concepts / Definitions (14) ──────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "SLC2A1 / GLUT1",
        "definition": (
            "SLC2A1 (solute carrier family 2 member 1), chromosome 1p34.2. Encodes GLUT1 "
            "(Glucose Transporter Type 1) — 492 amino acids, 12-transmembrane-domain facilitative "
            "hexose transporter. Principal glucose transporter at the blood-brain barrier (BBB) "
            "endothelium and choroid plexus. Also highly expressed in erythrocytes (enables "
            "functional erythrocyte glucose uptake diagnostic assay). GLUT1 transports glucose by "
            "facilitated diffusion (sodium-independent, bidirectional, concentration-driven). Km for "
            "glucose ~1-2 mmol/L — operates near saturation at physiological glucose concentrations "
            "→ haploinsufficiency is immediately rate-limiting for cerebral glucose supply."
        ),
    },
    {
        "term": "GLUT1-DS (De Vivo Disease / GLUT1 Deficiency Syndrome)",
        "definition": (
            "First described by De Vivo et al. 1991 (NEJM) in two children with infantile seizures, "
            "developmental delay, and low CSF glucose without systemic hypoglycaemia. Caused by "
            "SLC2A1 LOF variants. Prevalence: ~1:90,000 live births (De Vivo 2002). Triad: "
            "epilepsy + movement disorder + cognitive impairment. PED pathognomonic. Treatment "
            "with ketogenic diet curative (seizure-free >90%). ILAE classification: 'GLUT1-DS' "
            "as developmental and epileptic encephalopathy (DEE) when onset <5 years."
        ),
    },
    {
        "term": "Hypoglycorrhachia",
        "definition": (
            "Abnormally low CSF glucose concentration. In Glut1-DS: CSF glucose <2.2 mmol/L "
            "(<40 mg/dL) OR CSF:plasma ratio <0.45 (normal ≥0.65). Plasma glucose must be "
            "measured simultaneously after ≥4h fast. Differential: bacterial meningitis (CSF "
            "WBC elevated), tuberculous meningitis, fungal meningitis, subarachnoid haemorrhage "
            "(glucose consumed), hypoglycaemia (plasma glucose also low — excluded by ratio). "
            "GLUT1-DS UNIQUE: hypoglycorrhachia with normal plasma glucose — pathognomonic pattern."
        ),
    },
    {
        "term": "Blood-Brain Barrier GLUT1 (BBB-GLUT1)",
        "definition": (
            "GLUT1 is expressed at the luminal and abluminal membranes of BBB endothelial cells "
            "at approximately 10-100× higher density than most peripheral tissues. BBB-GLUT1 is "
            "responsible for >90% of cerebral glucose uptake (remainder via GLUT3 — neuronal). "
            "GLUT1 haploinsufficiency → BBB transport capacity halved → CNS glucose delivery "
            "insufficient to meet high neuronal metabolic demand (~5.6 mmol/100g/min). "
            "Ketones bypass BBB-GLUT1 via MCT1 — the basis of KD precision therapy."
        ),
    },
    {
        "term": "Paroxysmal Exercise-Induced Dyskinesia (PED)",
        "definition": (
            "Involuntary choreiform or dystonic limb movements triggered by sustained exercise "
            "(5-20 min), relieved by rest within 5-30 min. EEG: normal during episodes (not "
            "epileptic). Pathognomonic for Glut1-DS in context of epilepsy. Mechanism: exercise → "
            "muscle glucose consumption → relative cerebral glucose deficit → movement circuit "
            "dysfunction (basal ganglia — high GLUT1 expression). Also reported in SLC2A1-negative "
            "patients (MCT1 defect, phosphoglycerate kinase deficiency). PED + epilepsy = Glut1-DS "
            "until proven otherwise. KD resolves PED in >85%."
        ),
    },
    {
        "term": "Ketogenic Diet (KD) — Precision Therapy for Glut1-DS",
        "definition": (
            "High-fat, adequate-protein, very-low-carbohydrate diet producing nutritional ketosis "
            "(blood β-OHB 2-4 mmol/L). Classic ratio: 4:1 (fat:protein+carb by weight) provides "
            "80-90% kcal from fat. Ketone bodies (β-OHB + acetoacetate) cross BBB via MCT1 "
            "(intact in Glut1-DS) → TCA cycle → ATP. In Glut1-DS: KD is FIRST-LINE precision "
            "therapy, not last resort. ILAE Dietary Therapies commission (2018): KD Level A "
            "evidence in Glut1-DS. Response expected within 4-8 weeks (seizures) and 3-6 months "
            "(movement disorder, EEG). Duration: lifelong in most patients (KD weaning often "
            "leads to seizure recurrence). Dietitian supervision essential."
        ),
    },
    {
        "term": "β-Hydroxybutyrate (β-OHB)",
        "definition": (
            "Primary ketone body produced during KD (accounts for ~70% of total ketones). "
            "Hepatic β-oxidation of fatty acids → acetyl-CoA → ketogenesis in mitochondria → "
            "β-OHB transported in blood → MCT1-mediated BBB entry → astrocytic/neuronal "
            "β-OHB dehydrogenase → acetoacetate → acetyl-CoA → TCA cycle → ATP. "
            "Measurement: fingerprick ketone meters (Precision Xtra / Freestyle Optium Neo) "
            "give real-time β-OHB. Target in Glut1-DS: 2.0-4.0 mmol/L. <1.5 mmol/L → "
            "inadequate; >5 mmol/L → assess for acidosis. Fasting morning value most representative."
        ),
    },
    {
        "term": "Methylxanthine GLUT1 Inhibition",
        "definition": (
            "Caffeine, theophylline, aminophylline, theobromine, and pentoxifylline are methylated "
            "xanthine derivatives that competitively inhibit GLUT1-mediated glucose transport. "
            "Mechanism: methylxanthines bind the transmembrane glucose-transport cavity of GLUT1 → "
            "competitive inhibition (reversible, concentration-dependent). Ki for caffeine at GLUT1: "
            "~0.8-1.2 mmol/L (achievable with 2-3 cups coffee in GLUT1-heterozygous context where "
            "baseline transport is already compromised). Clinically: any dose of caffeine in Glut1-DS "
            "is potentially hazardous — no safe dose established. ABSOLUTE prohibition in all Glut1-DS."
        ),
    },
    {
        "term": "De Vivo Disease (eponym)",
        "definition": (
            "Eponym for Glut1-DS, honouring Darryl De Vivo (Columbia University) who first "
            "characterised the syndrome in 1991 (NEJM). De Vivo identified: (1) low CSF glucose, "
            "(2) normal plasma glucose, (3) seizures, (4) developmental delay, and (5) therapeutic "
            "response to high-fat diet (predecessor of formalised KD protocol). Still used in "
            "some clinical settings — synonymous with 'GLUT1-DS', 'SLC2A1 deficiency syndrome', "
            "and 'glucose transporter protein syndrome' (GTPS, older terminology)."
        ),
    },
    {
        "term": "Modified Atkins Diet (MAD)",
        "definition": (
            "Liberalised variant of KD: carbohydrate restricted to 10-20g net/day (child) or "
            "20-30g/day (adult), with fat ad libitum and no protein/calorie restriction. Produces "
            "moderate ketosis (β-OHB 1-3 mmol/L). In Glut1-DS: equivalent efficacy to classic KD "
            "for seizure control in mild-moderate phenotype (Klepper 2008). Benefits: easier to "
            "follow in school/social settings, better adherence in adolescents. Recommended as "
            "transition from 4:1 KD in older children or for mild Glut1-DS at outset. "
            "Still prohibits all methylxanthines."
        ),
    },
    {
        "term": "CSF:Plasma Glucose Ratio",
        "definition": (
            "Key diagnostic ratio for hypoglycorrhachia. Normal: ≥0.65 (fasting). "
            "Glut1-DS diagnostic threshold: <0.45 (sensitivity ~97%, specificity ~90% when "
            "combined with clinical syndrome). Important caveats: (1) Plasma glucose MUST be "
            "measured simultaneously (fasting — 4h minimum); (2) LP performed BEFORE KD initiation "
            "— KD normalises ratio → false negative; (3) Values 0.45-0.49 ('borderline'): "
            "perform erythrocyte glucose uptake assay + SLC2A1 sequencing for confirmation; "
            "(4) Inflammatory CSF (meningitis) causes falsely low CSF glucose — check WBC/protein; "
            "(5) Hypoglycaemia (plasma <3 mmol/L): ratio unreliable — wait until plasma normalised."
        ),
    },
    {
        "term": "Erythrocyte Glucose Uptake Assay (Functional GLUT1 Assay)",
        "definition": (
            "Erythrocytes express GLUT1 as their ONLY glucose transporter → provide accessible "
            "cell population to quantify GLUT1 function in vivo. Assay: 3-O-methylglucose (3-OMG) "
            "uptake into washed erythrocytes at 0°C (prevents metabolism) → measured by HPLC or "
            "enzymatic assay → compared to age-matched controls. Result: "
            "Heterozygous Glut1-DS: ~50% of normal uptake. Biallelic severe: <20% of normal. "
            "Mild partial LOF: 55-70%. Particularly useful: VUS on SLC2A1, borderline LP result, "
            "phenocopy/genetic-negative. NOT affected by KD (unlike CSF glucose)."
        ),
    },
    {
        "term": "SLC2A1 Alliance (Patient Registry)",
        "definition": (
            "International patient registry and advocacy organisation for Glut1-DS. Coordinated "
            "by the Glut1 Deficiency Foundation (USA) and Glut1-DS Research Foundation (Europe). "
            "Registry functions: natural history data, genetic variant database, clinical trial "
            "enrolment, dietary resource sharing, family networking. Research database includes "
            ">500 genetically confirmed cases. Available at glut1ds.org. Referral appropriate "
            "for all confirmed Glut1-DS cases — provides peer support and research access. "
            "New therapies under trial: isoform-specific SLC2A1 AAV gene therapy (preclinical)."
        ),
    },
    {
        "term": "Triheptanoin (C7 / UX007)",
        "definition": (
            "Triheptanoin (glycerol triheptanoate) — triglyceride of C7 (heptanoic acid, "
            "odd-chain fatty acid). Anaplerotic substrate: C7 → β-oxidation produces "
            "C5-ketones (β-hydroxypentanoate + β-ketopentanoate) AND propionyl-CoA → "
            "succinyl-CoA (direct TCA anaplerosis). Advantages over C8/C10 (standard MCT): "
            "C5-ketones may cross BBB via pathways independent of GLUT1 AND MCT1 → potentially "
            "useful in MCT1-deficient phenocopy cases. Phase 3 trial (UX007, Ultragenyx) in "
            "Glut1-DS — results mixed; approved for LCHAD/VLCAD in USA but investigational for "
            "Glut1-DS. Not first-line; specialist centre only."
        ),
    },
]

# ── Standards (8) ─────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE-2022 (Classification of Epilepsies — GLUT1-DS as DEE)",
    "NICE-NG217 (Epilepsies: Diagnosis and Management, 2022)",
    "Glut1-DS-International-Consensus-2013 (Leen et al., Brain 2013 — diagnostic + treatment)",
    "ILAE-Dietary-Therapies-Commission-2018 (KD Level A evidence in GLUT1-DS)",
    "ACMG-AMP-2015 (Variant interpretation — SLC2A1)",
    "ACNS-EEG-Guidelines-2021 (EEG reporting standards)",
    "De-Vivo-1991-NEJM (Original description — CSF glucose diagnostic criteria)",
    "Klepper-2004-AnnNeurol (100-patient series — phenotype-genotype spectrum)",
]

# ── Thresholds (10) ───────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "CSF glucose <2.2 mmol/L fasting → diagnostic for Glut1-DS", "value": "<2.2 mmol/L"},
    {"threshold": "CSF:plasma glucose ratio <0.45 → diagnostic threshold (Leen 2013)", "value": "<0.45"},
    {"threshold": "KD initiation: at diagnosis (not after 2-AED failures — Glut1-DS exception)", "value": "Diagnosis"},
    {"threshold": "β-OHB target on KD: 2.0-4.0 mmol/L (fingerprick morning fasting)", "value": "2.0-4.0 mmol/L"},
    {"threshold": "Lipid panel q3M on KD — LDL >5.0 mmol/L: review KD ratio", "value": "LDL >5.0"},
    {"threshold": "Methylxanthine ANY dose → ABSOLUTE CI — no safe dose in Glut1-DS", "value": "Zero tolerance"},
    {"threshold": "Fasting >4h → seizure trigger protocol (glucose gel/late snack)", "value": ">4h fast"},
    {"threshold": "KD trial minimum 3 months at target ketosis before declaring failure", "value": "3 months"},
    {"threshold": "Growth z-score fall >1 SD on KD → dietitian urgent review (increase calories)", "value": ">1 SD drop"},
    {"threshold": "SUDEP risk: GTCS frequency — counsel annually (NICE NG217)", "value": "Annual"},
]

# ── Key References (6) ────────────────────────────────────────────────────────
REFERENCES = [
    "De Vivo DC et al. (1991). NEJM 325:703-709 — First description of GLUT1 deficiency syndrome",
    "Klepper J et al. (2004). Ann Neurol 55:476-488 — 100-patient genotype-phenotype spectrum",
    "Leen WG et al. (2013). Brain 136:3438-3446 — International diagnostic consensus and guidelines",
    "Verrotti A et al. (2012). Epilepsia 53:1503-1509 — KD efficacy meta-analysis in GLUT1-DS",
    "Suls A et al. (2008). Ann Neurol 64:539-550 — SLC2A1 mutations in idiopathic generalized epilepsies",
    "Pong AW et al. (2012). Pediatr Neurol 47:397-403 — Erythrocyte glucose uptake functional assay",
]


# ── API response builders ──────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    n_ped = sum(1 for p in PATIENTS if p["ped_present"])
    n_kd = sum(1 for p in PATIENTS if p["on_kd"])
    n_kd_ctrl = sum(1 for p in PATIENTS if p["seizure_control"] == "KD-controlled")
    n_mxe = sum(1 for p in PATIENTS if p["methylxanthine_exposure_hx"])
    avg_ratio = round(sum(p["csf_plasma_ratio"] for p in PATIENTS) / n, 2)

    return {
        "dashboard": "SLC2A1 / GLUT1-DS — De Vivo Disease",
        "dashboard_id": "slc2a1",
        "dashboard_number": 188,
        "syndrome": "GLUT1 Deficiency Syndrome (Glut1-DS / De Vivo Disease)",
        "gene": "SLC2A1 (1p34.2) — Glucose Transporter Type 1 (GLUT1)",
        "inheritance": "Autosomal dominant (de novo ~90%) · Rare AR biallelic (severe) · AD familial ~10%",
        "eeg_hallmark": (
            "Generalised 2.5-4 Hz spike-and-wave — WORSE WITH FASTING, IMPROVES AFTER MEALS/KD. "
            "EEG normalisation on established KD in 70-85%. EEG during PED: NORMAL (metabolic, not ictal)."
        ),
        "key_biomarker": "CSF:plasma glucose ratio <0.45 (fasting) + CSF glucose <2.2 mmol/L",
        "precision_therapy": "Ketogenic Diet (4:1 or 3:1 KD) / Modified Atkins Diet — FIRST LINE AT DIAGNOSIS",
        "prevalence": "~1 in 90,000 live births (De Vivo 2002)",
        "kpis": {
            "total_patients": n,
            "ped_present": n_ped,
            "ped_pct": round(100 * n_ped / n),
            "on_kd": n_kd,
            "kd_pct": round(100 * n_kd / n),
            "kd_controlled": n_kd_ctrl,
            "kd_controlled_pct": round(100 * n_kd_ctrl / n),
            "methylxanthine_exposure_hx": n_mxe,
            "methylxanthine_pct": round(100 * n_mxe / n),
            "avg_csf_plasma_ratio": avg_ratio,
            "etiology_classes": len(ETIOLOGY_CATALOG),
            "seizure_types": len(SEIZURE_TYPES),
            "treatments": len(TREATMENTS),
        },
        "critical_alerts": [
            {
                "alert": "METHYLXANTHINES ABSOLUTE CI — caffeine/theophylline/theobromine competitively inhibit GLUT1",
                "action": "Document as ALLERGY in EMR. Medical ID bracelet. No coffee/cola/chocolate ever. No aminophylline IV.",
                "severity": "ABSOLUTE CONTRAINDICATION",
                "color": "danger",
            },
            {
                "alert": "LP BEFORE KD — CSF glucose normalises on KD → false negative if LP delayed",
                "action": "LP (fasting 4h minimum) BEFORE first dose of ketogenic diet. Document CSF:plasma ratio.",
                "severity": "MANDATORY",
                "color": "warning",
            },
            {
                "alert": "KD IS FIRST-LINE — do not delay for 2-AED trial (unlike all other epilepsies)",
                "action": "Initiate KD at diagnosis. Glut1-DS is the exception: KD first, AEDs adjunct.",
                "severity": "PRECISION THERAPY",
                "color": "success",
            },
            {
                "alert": "FASTING PROTOCOL MANDATORY — perioperative, illness, school, sport",
                "action": "Sick-day plan, surgical protocol letter, pre-exercise snack. Fasting >4h = seizure risk.",
                "severity": "MANDATORY PROTOCOL",
                "color": "warning",
            },
        ],
        "pathway_summary": (
            "GLUT1-DS MANAGEMENT PATHWAY: (1) Suspect: absence/seizures + morning predominance + "
            "PED + movement disorder → (2) Fasting LP (CSF:plasma glucose ratio) BEFORE KD → "
            "(3) SLC2A1 sequencing + erythrocyte assay → (4) KD initiation at diagnosis "
            "(dietitian referral urgent) → (5) Target β-OHB 2-4 mmol/L → (6) Seizure/PED/EEG "
            "monitoring q3-6M → (7) Neuropsych q12M → (8) Lifelong KD/MAD."
        ),
        "standards": STANDARDS,
        "references": REFERENCES,
        "updated_at": datetime.now().strftime("%Y-%m-%d"),
    }


def get_breakdown():
    return {
        "dashboard_id": "slc2a1",
        "patients": PATIENTS,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "thresholds": THRESHOLDS,
        "summary": {
            "n_patients": len(PATIENTS),
            "n_etiology_classes": len(ETIOLOGY_CATALOG),
            "n_seizure_types": len(SEIZURE_TYPES),
            "n_triggers": len(TRIGGERS),
            "n_treatments": len(TREATMENTS),
            "n_contraindications": len(CONTRAINDICATIONS),
            "n_monitoring_items": len(MONITORING),
            "n_lifecycle_phases": len(LIFECYCLE),
        },
        "updated_at": datetime.now().strftime("%Y-%m-%d"),
    }


def get_definitions():
    return {
        "dashboard_id": "slc2a1",
        "definitions": DEFINITIONS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "contraindications": CONTRAINDICATIONS,
        "n_concepts": len(DEFINITIONS),
        "updated_at": datetime.now().strftime("%Y-%m-%d"),
    }
