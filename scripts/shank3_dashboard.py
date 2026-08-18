"""
SHANK3 Epilepsy — Phelan-McDermid Syndrome / 22q13.33 Deletion
================================================================
40-patient cohort · SHANK3 (22q13.33) · De novo dominant (deletion 75% / LOF 25%)
Postsynaptic Density Scaffold Protein · mGluR5-AMPA-NMDA Scaffolding
PMS (OMIM #606232) · Autism + Intellectual Disability + Regression + Seizures

SHANK3 PROTEIN BIOLOGY:
SHANK3 (SH3 and multiple ankyrin repeat domains 3) is the master organiser of the
postsynaptic density (PSD) at glutamatergic synapses:
  - 1740 aa, 198 kDa — largest SHANK family member (SHANK1/2/3)
  - N-terminal: ankyrin repeat domain (ARD) — scaffold assembly
  - Central: SH3, PDZ, proline-rich — binds Homer, Shank2, GKAP/SAPAP
  - C-terminal: SAM domain — self-multimerisation → PSD nanodomains
  - Directly binds:
      Homer (bridges mGluR5 → SHANK3 → AMPA receptors)
      GKAP/SAPAP (connects SHANK3 → MAGUKs → NMDARs)
      IRSp53 → actin cytoskeleton → dendritic spine morphology
      PSD-95 indirectly via SAPAP → NMDAR anchoring
  - Function: scaffold for mGluR5, NMDAR, AMPAR co-localisation at PSD;
    regulate LTP/LTD balance; dendritic spine density and morphology

DISEASE MECHANISM:
  22q13.3 deletion (75%): terminal deletion from SHANK3 locus to telomere (size 0.3-9 Mb);
    all deletions include SHANK3; larger deletions add ARSA, MAPK8IP2 (IB2), ADSL etc.;
    strictly haploinsufficient (one functional copy insufficient for normal PSD assembly);
    larger deletions → more severe (seizures more likely with >3 Mb; ADSL loss → purine
    synthesis defect → regression; IB2 loss → cerebellar signs)

  SHANK3 point mutation/truncating LOF (25%): de novo frameshift, nonsense, splice site;
    same haploinsufficiency; milder somatic variants reported in ASD without PMS

SHANK3 HAPLOINSUFFICIENCY → EPILEPSY:
  Reduced SHANK3 → ↓mGluR5 density/coupling at PSD → reduced mGluR5-LTD →
    LTP/LTD imbalance (net hyperexcitability of glutamatergic synapses);
  NMDAR under-anchoring → compensatory NMDAR upregulation at extra-synaptic sites
    (NR2B-containing extra-synaptic NMDARs) → excitatory drive increase;
  Dendritic spine loss (↓density, ↓volume) → altered coincidence detection →
    network E/I imbalance;
  Seizure prevalence: ~30-40% of PMS patients (Soorya 2013, Bonaglia 2011);
    febrile (40%) and afebrile (60%); CSWS/ESES <5%; typical onset 2-6 years;
    focal > generalised; drug-responsive in ~60%

REGRESSION IN SHANK3 / PMS:
  50-70% show language/motor regression (SHANK3 Regression Syndrome — SRS);
  Most commonly triggered by fever/illness, anaesthesia, sleep deprivation;
  Loss of 5-20 words, motor coordination, toilet training;
  Often partially reversible with 6-12 months recovery period;
  EEG may show CSWS (continuous spike-wave during slow sleep) during regression;
  IGF-1 trial data: regression may be arrested/reversed (Kolevzon 2014, phase I open-label)

PRECISION THERAPY — IGF-1 (INSULIN-LIKE GROWTH FACTOR 1):
  IGF-1 (mecasermin/Increlex) → IGF-1R → PI3K/Akt/mTOR → PSD-95 + SHANK3 expression
    upregulation via increased protein synthesis (mTOR-S6K1 pathway);
  Restores mGluR5-Homer-SHANK3 complex assembly at PSD;
  Phase I open-label (Kolevzon 2014, 9 patients): improved social responsiveness,
    ABC scores, motor function; some language gains; partial seizure improvement;
  Mechanism: bypasses SHANK3 synthesis defect by increasing residual allele protein output
    via mTOR (IGF-1 → PI3K → mTOR-C1 → 4EBP1 + S6K1 → SHANK3 protein synthesis);
  IGF-1 + D-cycloserine (GlyB agonist): augments NMDAR function at synapses lacking
    SHANK3 scaffold → dual approach to restore E/I balance

CONTRAINDICATIONS IN SHANK3:
  1. Phenytoin/Carbamazepine HIGH RISK: Na-channel blockers can worsen behaviour/regression;
     anecdotal reports of increased regression frequency; avoid as first-line in PMS
  2. Vigabatrin HIGH RISK: GABA-T inhibitor → visual field defects; SHANK3 patients
     cannot reliably report vision changes (ID, absent speech) → perimetry impossible →
     HIGH RISK irreversible visual field loss undetected → avoid unless no alternative
  3. Valproate + IGF-1: VPA inhibits mTOR → blunts IGF-1 therapeutic mechanism;
     use VPA cautiously if IGF-1 trial planned
  4. Abrupt AED withdrawal: regression risk in context of fever (PMS vulnerability);
     any seizure provocation during unmonitored AED changes can trigger prolonged regression
  5. Benzodiazepine chronic use: behavioural disinhibition in ASD/ID; long-term CI;
     rescue use acceptable
  6. General anaesthesia risk: documented post-anaesthetic regression in PMS; pre-anaesthetic
     IGF-1 optimisation and neurology liaison mandatory

GENETICS:
  Gene:        SHANK3 (22q13.33) — also known as ProSAP2, PSAP2
  Protein:     SH3 and multiple ankyrin repeat domains 3 (1740 aa, 198 kDa)
  Inheritance: De novo dominant (deletion 75%; point mutation 25%); <1% familial
  pLI:         0.98 (highly intolerant to LoF)
  Incidence:   ~1:15,000-30,000 (PMS); SHANK3 point mutation in ASD ~1:100 (milder)
  OMIM:        PMS #606232; SHANK3 gene *606230
  First report: Phelan & McDermid 1985 (clinical); SHANK3 mapped 2001 (Wilson 2003)

KEY STANDARDS:
  ILAE 2022 · NICE NG217 · Phelan & McDermid 2001 ·
  Kolevzon et al. 2014 (J Autism Dev Disord — IGF-1 phase I) ·
  Soorya et al. 2013 (Sci Transl Med — PMS natural history) ·
  Leblond et al. 2014 (Am J Hum Genet — SHANK3 cohort) ·
  Bonaglia et al. 2011 (Eur J Hum Genet — 22q13 deletion) ·
  MHRA VPPP 2021 · ACMG-AMP 2015 · NICE NG224 2022 · WHO ICF 2019
"""
import random

# ── Etiology catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "22q13.3 Terminal Deletion — Classic PMS (Large, >3 Mb)",
        "category": "22q13-deletion-large-PMS-classic-38%",
        "pct": 38,
        "n": 15,
        "mechanism": (
            "Large terminal deletion of 22q13.3 spanning SHANK3 plus additional genes "
            "(ARSA, MAPK8IP2/IB2, ADSL, ACTN1); haploinsufficiency of multiple PSD "
            "and synaptic scaffolding genes amplifies phenotype severity. IB2/MAPK8IP2 loss "
            "adds cerebellar ataxia and coordination disorder; ADSL haploinsufficiency → "
            "purine synthesis defect (succinyladenosine accumulates → neurotoxic → regression "
            "episodes more frequent and severe). SHANK3 remains the dominant driver of ASD "
            "and epilepsy; larger deletions (>5 Mb) associated with 55-65% seizure prevalence "
            "vs ~25% for small deletions <1 Mb. Cortical malformations (simplified gyral "
            "pattern, thin corpus callosum) in 30%. EEG: multifocal spikes, CSWS rare (5%)."
        ),
        "seizure_types": ["Focal impaired awareness", "Febrile", "GTCS"],
        "age_onset_range": "2-8 years",
        "drug_response": "Partial (60% ≥50% reduction)",
        "typical_aed": "Valproate ± Levetiracetam",
        "regression_risk": "High (IB2/ADSL loss)",
        "igf1_candidate": True,
    },
    {
        "etiology": "22q13.3 Terminal Deletion — Small (<1 Mb), SHANK3-Only",
        "category": "22q13-deletion-small-SHANK3-only-28%",
        "pct": 28,
        "n": 11,
        "mechanism": (
            "Small terminal deletion (<1 Mb) disrupting only SHANK3 (and sometimes RABL2B "
            "and ACR); represents the minimum critical region for PMS. Seizure prevalence "
            "lower (~25% vs ~45% for large deletions); phenotype dominated by SHANK3 haploinsufficiency "
            "alone: ASD, intellectual disability (moderate-severe), absent/minimal speech, "
            "neonatal hypotonia. Dendritic spine density and AMPAR surface expression reduced "
            "by ~40% in rodent models (Bozdagi 2010). mGluR5-Homer coupling reduced → LTD "
            "impairment → net excitability increase. Seizures: focal onset, age 2-6 years, "
            "triggered by fever. IGF-1 highest response rate in this class (cleanest "
            "genotype-phenotype for SHANK3 precision therapy)."
        ),
        "seizure_types": ["Focal with fever", "Myoclonic (rare)"],
        "age_onset_range": "2-6 years",
        "drug_response": "Good (70% ≥50% reduction)",
        "typical_aed": "Valproate",
        "regression_risk": "Moderate",
        "igf1_candidate": True,
    },
    {
        "etiology": "SHANK3 De Novo Point Mutation — Truncating LOF",
        "category": "SHANK3-point-mutation-truncating-LOF-20%",
        "pct": 20,
        "n": 8,
        "mechanism": (
            "De novo frameshift, nonsense, or canonical splice-site variants causing "
            "haploinsufficiency without chromosomal deletion. No additional gene loss → "
            "phenotype attributable purely to SHANK3. Seizure prevalence ~20%; ASD + "
            "intellectual disability core features. Truncating variants in SAM domain "
            "(C-terminal) → loss of PSD nanodomains (SHANK3 self-multimerisation fails); "
            "PDZ domain truncations → Homer/mGluR5 uncoupling; ANK domain truncations → "
            "severe (NR2B over-activity). EEG: typically normal interictal or non-specific "
            "changes; ictal: focal or generalised depending on variant location. IGF-1 "
            "response: similar to small deletion class. Genetic diagnosis by WES/WGS (22q13 "
            "deletion FISH/CMA false-negative for point mutations)."
        ),
        "seizure_types": ["Focal impaired awareness", "GTCS"],
        "age_onset_range": "3-10 years",
        "drug_response": "Good (75% ≥50% reduction)",
        "typical_aed": "Lamotrigine or Valproate",
        "regression_risk": "Moderate",
        "igf1_candidate": True,
    },
    {
        "etiology": "SHANK3 Mosaicism — Somatic or Germline Low-Level",
        "category": "SHANK3-mosaicism-10%",
        "pct": 10,
        "n": 4,
        "mechanism": (
            "Somatic mosaic SHANK3 deletion or point mutation present in a subset of "
            "cells (typically 20-80% mosaic); milder phenotype commensurate with mosaic "
            "allele frequency. Parents with low-level mosaicism may have minimally affected "
            "phenotype but transmit full de novo variant to offspring. Seizures in ~15% "
            "of mosaic cases; predominantly focal febrile. EEG: often normal. Critical "
            "diagnostic implication: saliva/blood mosaic SHANK3 may underestimate neuronal "
            "mosaic fraction (neuronal-specific mosaicism) → brain MRI molecular imaging "
            "insufficient to exclude. WGS with >100x depth or single-cell sequencing for "
            "accurate mosaic fraction in brain (CSF cfDNA emerging)."
        ),
        "seizure_types": ["Febrile", "Focal (mild)"],
        "age_onset_range": "3-12 years",
        "drug_response": "Excellent (85% seizure-free)",
        "typical_aed": "Levetiracetam (short course)",
        "regression_risk": "Low",
        "igf1_candidate": False,
    },
    {
        "etiology": "Phenocopy — 22q13 Mimic (non-SHANK3 deletion, ASD+epilepsy)",
        "category": "Phenocopy-non-SHANK3-4%",
        "pct": 4,
        "n": 2,
        "mechanism": (
            "Rare (4%) ASD+ID+epilepsy cohort with 22q13 CMA abnormality not disrupting "
            "SHANK3 proper (partial deletions sparing SHANK3, or nearby copy-number variants "
            "affecting MAPK8IP2, SYNGAP-adjacent, or 22q11.2 distal); or PMS-like phenotype "
            "with different genetic cause (e.g. SHANK1, SHANK2, SYNGAP1, ADNP). Important "
            "to distinguish: IGF-1 therapeutic rationale applies specifically to SHANK3 "
            "deficiency; ADNP has its own precision approach (VPA for ACTG1 modulation). "
            "CMA should characterise deletion boundaries; WES/WGS for point mutations; "
            "SHANK3 copy number by MLPA or ddPCR to confirm haploinsufficiency."
        ),
        "seizure_types": ["Variable"],
        "age_onset_range": "Variable",
        "drug_response": "Variable",
        "typical_aed": "Per underlying diagnosis",
        "regression_risk": "Variable",
        "igf1_candidate": False,
    },
]

# ── Seizure catalog ───────────────────────────────────────────────────────────
SEIZURE_CATALOG = [
    {
        "type": "Focal Impaired Awareness Seizure (FIAS)",
        "prevalence_pct": 72,
        "onset": "Frontal or temporal",
        "duration": "30-120 seconds",
        "eeg": "Focal interictal spikes (frontal > temporal); ictal: rhythmic theta discharge",
        "semiology": (
            "Behavioural arrest, staring, automatisms (oro-manual in temporal; "
            "hypermotor in frontal onset). Postictal confusion 5-30 min. "
            "Often clusters in non-REM sleep. Secondary generalisation in 20%. "
            "VIDEO-EEG: hypermotor frontal onset mimics ADNFLE — distinguish by "
            "SHANK3 genetics and diurnal distribution."
        ),
        "clinical_tip": (
            "Temporal vs frontal onset matters for lateralisation and surgery candidacy "
            "in refractory PMS epilepsy; SEEG preferred (scalp EEG low yield in PMS). "
            "Frontal network predominance — consider frontal resection in DRE."
        ),
    },
    {
        "type": "Febrile Seizure / Febrile Seizure Plus (FS+)",
        "prevalence_pct": 65,
        "onset": "Generalised or focal+generalised",
        "duration": "1-5 minutes",
        "eeg": "Post-ictal slowing; interictal usually normal",
        "semiology": (
            "First seizure often febrile (temperature >38°C); GTCS or prolonged focal. "
            "Recurrent febrile seizures in 40% of cases. Fever also triggers "
            "REGRESSION (loss of language/motor) independently — regression and febrile "
            "seizures co-occur → difficult to distinguish fever-triggered regression from "
            "ictal-postictal regression. AED prophylaxis for prolonged or clustered febrile "
            "seizures recommended in PMS (Soorya 2013 consensus)."
        ),
        "clinical_tip": (
            "Every febrile illness in PMS = dual risk: (1) febrile seizure → AED rescue, "
            "(2) regression episode → IGF-1 optimisation + intensive speech/OT. "
            "Hospital admission criteria: fever + seizure + ANY new skill loss."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizure (GTCS)",
        "prevalence_pct": 45,
        "onset": "Diffuse cortical",
        "duration": "1-3 minutes",
        "eeg": "Generalised spike-wave or polyspike; 3-6 Hz",
        "semiology": (
            "Bilateral tonic extension → clonic phase. Postictal confusion/fatigue "
            "prolonged (30-120 min) in PMS due to baseline cognitive impairment. "
            "Nocturnal predominance. GTCS rate higher with large deletions (>3 Mb) "
            "vs small SHANK3-only deletions. Status epilepticus risk: ~10% (fever-triggered)."
        ),
        "clinical_tip": (
            "Valproate first-line for GTCS in PMS (broad-spectrum; Level B). "
            "Phenytoin/CBZ avoid (Na-channel blockers worsen behaviour). "
            "SE protocol: midazolam buccal → lorazepam IV → levetiracetam IV → "
            "valproate IV; avoid phenytoin in PMS SE."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "prevalence_pct": 25,
        "onset": "Bilateral synchronous",
        "duration": "< 1 second",
        "eeg": "Polyspike-wave; 4-6 Hz; enhanced by sleep deprivation",
        "semiology": (
            "Brief bilateral jerks, often on waking. May be subtle (head-drops, "
            "cup-drop). Myoclonus in PMS may be epileptic or non-epileptic "
            "(SHANK3 actin-pathway movement disorder); EEG correlation mandatory. "
            "Myoclonic SE rare but reported in large deletions."
        ),
        "clinical_tip": (
            "Levetiracetam for myoclonus (Level C); caution: behavioural "
            "irritability 15-30% in ASD/ID with LEV — switch to brivaracetam "
            "if irritability emerges. Valproate also effective for myoclonus."
        ),
    },
    {
        "type": "Epileptic Spasms / CSWS (rare)",
        "prevalence_pct": 12,
        "onset": "Symmetric flexion",
        "duration": "1-3 seconds (spasm series 5-30 min)",
        "eeg": "Hypsarrhythmia (spasms) or CSWS during slow sleep (regression phase)",
        "semiology": (
            "Infantile spasms rare (<5% PMS); CSWS (continuous spike-wave during slow "
            "sleep) in regression phase: abrupt loss of speech, motor regression, "
            "cognitive decline; may be mistaken for Landau-Kleffner syndrome. "
            "CSWS in PMS correlates with clinical regression and requires "
            "treatment (corticosteroids ± VPA). Hypsarrhythmia: ACTH protocol "
            "if infantile onset."
        ),
        "clinical_tip": (
            "Overnight EEG mandatory in any PMS patient with language regression — "
            "CSWS diagnosis changes management (steroids) and prognosis. "
            "CSWS treatment: prednisolone 2 mg/kg/day × 4 weeks then taper; "
            "ACTH for spasms (UKISS protocol)."
        ),
    },
]

# ── Trigger catalog ───────────────────────────────────────────────────────────
TRIGGER_CATALOG = [
    {"trigger": "Fever / Febrile Illness", "pct": 88, "notes": "Most common; dual risk: seizure + regression. Temperature ≥38°C = pre-emptive midazolam buccal + hospitalisation protocol for PMS. Antipyretics early (paracetamol + ibuprofen)."},
    {"trigger": "Sleep Deprivation", "pct": 72, "notes": "Missed or disrupted sleep → seizure cluster; also worsens myoclonus. Melatonin 3-6 mg for sleep consolidation (Level C; also seizure-reduction benefit in ASD/ID epilepsy)."},
    {"trigger": "General Anaesthesia / Post-Operative", "pct": 62, "notes": "Post-anaesthetic regression in 50-65% PMS (language + motor loss, 6-12 months recovery). Pre-anaesthetic neurology consultation mandatory. IGF-1 optimisation before elective surgery."},
    {"trigger": "Missed AED dose", "pct": 55, "notes": "Any dose gap in context of baseline low seizure threshold. AED adherence challenges in ID/ASD (sensory aversion to medication, pill refusal). Consider liquid formulations."},
    {"trigger": "Psychological stress / Routine change", "pct": 48, "notes": "ASD vulnerability to environmental change → heightened cortical excitability. Structured daily routine and social stories for medical procedures."},
    {"trigger": "Illness / Infection (non-febrile)", "pct": 42, "notes": "Systemic illness lowers threshold even without fever. Metabolic stress (dehydration, vomiting → valproate toxicity risk)."},
    {"trigger": "Menstrual cycle (catamenial, adolescents)", "pct": 22, "notes": "Adolescent females with PMS and seizures: catamenial pattern in 20-25%. Progesterone supplementation (clobazam perimenstrual) considered Level C."},
    {"trigger": "Photosensitivity (rare)", "pct": 8, "notes": "Photosensitive EEG response in ~8%; generalised polyspike-wave to IPS. Avoid strobe lights, ensure screen flicker rate >60 Hz."},
]

# ── Treatment catalog ─────────────────────────────────────────────────────────
TREATMENT_CATALOG = [
    {
        "drug": "Valproate (VPA)",
        "level": "Level B",
        "role": "First-line broad-spectrum (GTCS + focal + myoclonus)",
        "dose": "20-40 mg/kg/day in 2 doses; target TDM 75-120 µg/mL",
        "moa": "Na+ channel stabilisation + GABA transaminase inhibition + T-type Ca2+ block",
        "efficacy": "65-70% ≥50% reduction; 30% seizure-free",
        "monitoring": "VPA TDM, LFT, FBC, NH3 (hyperammonaemia in ID), weight",
        "shank3_note": (
            "Caution if IGF-1 trial planned: VPA inhibits mTOR → reduces IGF-1 therapeutic "
            "benefit. Preferentially choose LEV or LTG if IGF-1 trial imminent. "
            "Screen POLG1 before VPA in any developmental epilepsy (Alpers-Huttenlocher risk)."
        ),
        "ci_flag": False,
    },
    {
        "drug": "Lamotrigine (LTG)",
        "level": "Level B",
        "role": "First-line focal seizures; adjunct for GTCS; well-tolerated in ASD/ID",
        "dose": "0.3 mg/kg/day start (slow titration); target 100-300 mg/day",
        "moa": "Na+ and Ca2+ channel stabilisation; reduces glutamate release",
        "efficacy": "60-65% ≥50% reduction; good tolerability profile in PMS",
        "monitoring": "LTG levels (AED interactions); DRESS/SJS rash watch (slow titration mandatory)",
        "shank3_note": (
            "Generally well-tolerated in PMS/ASD; slow titration essential (DRESS risk). "
            "Compatible with IGF-1. Preferred over carbamazepine in PMS."
        ),
        "ci_flag": False,
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "role": "Adjunct; useful for myoclonus; IV form for SE",
        "dose": "20-60 mg/kg/day; IV: 60 mg/kg (max 4500 mg) for SE",
        "moa": "SV2A binding → reduced vesicle-mediated neurotransmitter release",
        "efficacy": "55-60% ≥50% reduction; IV LEV useful in SE (avoid phenytoin)",
        "monitoring": "Behavioural irritability screen (15-30% ASD/ID); CBC (rare thrombocytopaenia)",
        "shank3_note": (
            "Behavioural irritability HIGH risk in PMS/ASD (30% vs 3-8% general epilepsy). "
            "Use explicit consent + monthly behavioural checklist (ABC-Community). "
            "Switch to brivaracetam if irritability emerges (lower SV2A affinity → less behaviour). "
            "IV LEV = preferred for SE over phenytoin in PMS."
        ),
        "ci_flag": False,
    },
    {
        "drug": "IGF-1 / Mecasermin (Increlex) — PRECISION",
        "level": "Level C (Phase I open-label, Kolevzon 2014)",
        "role": "Precision Rx: mTOR-SHANK3 pathway restoration; seizure + regression",
        "dose": "0.04-0.12 mg/kg SC twice daily (titrate over 4 weeks); max 0.12 mg/kg BID",
        "moa": "IGF-1R → PI3K → Akt → mTOR-C1 → ↑SHANK3 protein translation + mGluR5 PSD coupling",
        "efficacy": "Phase I n=9: 55% improved social responsiveness; 33% seizure reduction; motor gains",
        "monitoring": "Blood glucose (hypoglycaemia — administer with meal); IGF-1 levels; bone age; growth",
        "shank3_note": (
            "SHANK3-SPECIFIC PRECISION THERAPY: IGF-1 increases residual SHANK3 protein "
            "from functional allele via mTOR-dependent translation. Administer with food "
            "(hypoglycaemia risk). Avoid VPA during trial (mTOR inhibition). "
            "Most effective in SHANK3-only small deletion and point mutation classes. "
            "Phase II trial (NCT02677935) ongoing — consult nearest PMS research centre. "
            "Not yet FDA/EMA approved for epilepsy (compassionate use / trial)."
        ),
        "ci_flag": False,
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "role": "Adjunct for focal; rescue (acute cluster); catamenial add-on",
        "dose": "0.1-0.3 mg/kg/day (max 40 mg/day); rescue: 0.5 mg/kg buccal",
        "moa": "GABAA positive allosteric modulator (benzodiazepine site, α1/α2)",
        "efficacy": "50-55% as adjunct; tolerance develops in 6-12 months",
        "monitoring": "Sedation, drooling (especially in hypotonic PMS); withdrawal seizures on stop",
        "shank3_note": (
            "Long-term CLB limited by tolerance and behavioural disinhibition in ASD. "
            "Reserve for acute cluster rescue (buccal midazolam preferred for home rescue). "
            "Perimenstrual CLB (days -3 to +3) for catamenial pattern."
        ),
        "ci_flag": False,
    },
    {
        "drug": "ACTH / Prednisolone (Corticosteroids)",
        "level": "Level B (CSWS/spasms); Level C (regression)",
        "role": "CSWS in regression phase; infantile spasms (rare); severe regression",
        "dose": "Prednisolone 2 mg/kg/day × 4 weeks then taper; ACTH UKISS protocol for spasms",
        "moa": "Anti-inflammatory; GABA-A receptor modulation; ACTH peptide receptor in cortex",
        "efficacy": "CSWS: 60-70% EEG response + language gains; spasms: 50-60% cessation",
        "monitoring": "BP, glucose, infection, bone density (prolonged use), adrenal suppression",
        "shank3_note": (
            "CSWS + REGRESSION = urgent corticosteroid consideration. "
            "Overnight EEG MANDATORY before starting (diagnose CSWS vs non-epileptic regression). "
            "ACTH preferred for infantile spasms (higher cessation rate). "
            "Language/motor gains with corticosteroids in CSWS/PMS — document baseline Bayley."
        ),
        "ci_flag": False,
    },
    {
        "drug": "Melatonin (sleep co-morbidity)",
        "level": "Level C",
        "role": "Sleep disorder co-treatment; secondary seizure reduction",
        "dose": "3-6 mg PO at bedtime (controlled-release preferred)",
        "moa": "MT1/MT2 receptor → sleep phase consolidation; also anticonvulsant properties",
        "efficacy": "70-75% improved sleep onset/duration in ASD/ID; secondary 20-30% seizure reduction",
        "monitoring": "Morning sleepiness; growth suppression with high-dose long-term (controversial)",
        "shank3_note": (
            "Sleep deprivation is a major trigger in PMS. Melatonin addresses "
            "both primary sleep disorder (ubiquitous in ASD/ID) and secondary seizure risk. "
            "Safe and well-tolerated; start at 3 mg controlled-release."
        ),
        "ci_flag": False,
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level C",
        "role": "Drug-resistant epilepsy; adjunct for myoclonus/GTCS",
        "dose": "3:1 or 4:1 ratio (fat:protein+carb); dietitian-supervised; min 3 months trial",
        "moa": "Ketone bodies → KATP channel opening + GABA enhancement + mTOR inhibition",
        "efficacy": "KD in ASD/ID epilepsy: 45-55% ≥50% reduction; may improve ASD behaviours",
        "monitoring": "BHB levels (target 2-5 mmol/L); lipids; renal stones; growth; selenium; carnitine",
        "shank3_note": (
            "KD-mTOR inhibition effect is complementary to IGF-1-mTOR activation — "
            "these should NOT be combined (opposing mTOR signals). "
            "Choose KD for DRE without IGF-1 trial plan; IGF-1 trial with LTG/VPA for mTOR-enabled patients. "
            "Feeding challenges in PMS (sensory hypersensitivity, g-tube in 20%) — KD feasibility assessment."
        ),
        "ci_flag": False,
    },
]

# ── Contraindication catalog ──────────────────────────────────────────────────
CI_CATALOG = [
    {
        "drug": "Phenytoin (PHT) / Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "risk": "HIGH RISK — worsen behaviour and seizures in PMS/ASD/ID",
        "mechanism": (
            "Na-channel blockers reduce dendritic spine excitability; in context of "
            "SHANK3 haploinsufficiency with already-reduced PSD density, Na-channel "
            "blockade further impairs synaptic transmission → behavioural regression, "
            "irritability, worsening of cognitive status. CBZ/OXC also enzyme-inducing "
            "(CYP3A4) → reduces LTG/CLB levels. PHT in SE: avoided in PMS (use LEV IV)."
        ),
        "action": "Avoid as first-line; use LEV IV for SE; document CI on emergency care plan",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "risk": "HIGH RISK — visual field defect undetectable in non-verbal PMS patients",
        "mechanism": (
            "VGB causes permanent visual field defects (nasal ring scotoma) in 30-40% "
            "with cumulative dose-dependent risk. Perimetry mandatory for monitoring. "
            "PMS patients with absent speech, ID, ASD → UNABLE to report visual symptoms "
            "or cooperate with perimetry → silent irreversible visual loss. "
            "Ophthalmology VEP and ERG: technically feasible but insensitive for early "
            "field loss. VGB absolutely avoidable in PMS given this monitoring impossibility."
        ),
        "action": "AVOID in PMS — visual monitoring impossible; use VPA/LTG/LEV instead",
    },
    {
        "drug": "Valproate during IGF-1 trial",
        "risk": "HIGH RISK — VPA inhibits mTOR → blunts IGF-1 therapeutic mechanism",
        "mechanism": (
            "Valproic acid inhibits histone deacetylase (HDAC) and indirectly inhibits "
            "mTOR complex 1 via TSC1/2 pathway upregulation → reduced protein translation → "
            "blunted SHANK3 upregulation in response to IGF-1. Clinical evidence: VPA + "
            "IGF-1 combination → reduced social responsiveness gain vs IGF-1 alone "
            "(post-hoc analysis, Kolevzon 2014). Switch VPA to LTG or LEV if IGF-1 trial planned."
        ),
        "action": "Transition VPA to LTG or LEV before IGF-1 trial; 4-week washout",
    },
    {
        "drug": "Abrupt AED withdrawal (any AED)",
        "risk": "ABSOLUTE — seizure cluster + regression trigger",
        "mechanism": (
            "Any abrupt AED discontinuation risks breakthrough seizures which in PMS "
            "can trigger prolonged regression (language/motor loss). SHANK3 homeostatic "
            "plasticity is impaired → lower capacity to re-establish equilibrium after "
            "seizure clusters → regression lasting 6-12 months. "
            "AED changes must be gradual: reduce by 10-25%/week; hospital standby for "
            "any change in a patient with prior regression history."
        ),
        "action": "Never abrupt stop; taper 10-25%/week; hospital alert for regression watch",
    },
    {
        "drug": "General anaesthesia (unplanned/unoptimised)",
        "risk": "HIGH RISK — post-anaesthetic regression in 50-65% PMS",
        "mechanism": (
            "Volatile anaesthetics (isoflurane, sevoflurane) → transient NMDA receptor "
            "blockade + GABAA enhancement → altered synaptic plasticity in already "
            "vulnerable SHANK3-haploinsufficient brain → prolonged synaptogenesis disruption. "
            "Post-anaesthetic regression: language loss, motor regression, ASD behaviour "
            "worsening lasting 6-12 months (Holder 2012 case series). Propofol-based TIVA "
            "may have lower regression risk (retrospective data only). "
            "IGF-1 pre-operatively (2 weeks) may mitigate (not yet evidence-based)."
        ),
        "action": "Pre-anaesthetic neurology consult; document regression baseline; TIVA preferred; post-op PICU monitoring",
    },
    {
        "drug": "Benzodiazepines (chronic use)",
        "risk": "HIGH RISK — behavioural disinhibition in ASD/ID; dependency",
        "mechanism": (
            "Chronic benzodiazepine use in ASD/ID → paradoxical behavioural activation "
            "(disinhibition) in 20-30%: increased aggression, self-injurious behaviour, "
            "hyperactivity. Tolerance develops within 3-6 months → loss of AED efficacy. "
            "Reserve for: acute SE rescue (midazolam buccal/IM), acute cluster rescue, "
            "catamenial CLB (intermittent). Never prescribe as daily maintenance in PMS."
        ),
        "action": "Rescue use only; document behavioural monitoring; chronic use requires MDT review",
    },
]

# ── Monitoring catalog ────────────────────────────────────────────────────────
MONITORING_CATALOG = [
    {"item": "CMA (Chromosomal Microarray) — First-Line Genetic", "frequency": "Once at diagnosis", "note": "22q13.3 deletion characterisation: deletion size, boundaries; predicts severity (IB2, ADSL additional genes). Array CGH ≥60K probe density."},
    {"item": "WES/WGS if CMA negative", "frequency": "Once (SHANK3 point mutation)", "note": "SHANK3 point mutations not detected by CMA; WES preferred; Sanger confirmation. Parents also sequenced (de novo confirmation)."},
    {"item": "SHANK3 MLPA / ddPCR (mosaic cases)", "frequency": "Once + if mosaic suspected", "note": "Mosaic fractions <20% may be missed by standard WES; ddPCR detects down to 1% mosaic; blood + buccal swab."},
    {"item": "Overnight Video-EEG (CSWS protocol)", "frequency": "Annual; urgent if regression", "note": "CSWS during slow sleep → regression phase. Mandatory before corticosteroid decision. Also assess: sleep spindles, K-complexes (absent in CSWS)."},
    {"item": "EEG (routine + sleep-deprived)", "frequency": "6-monthly × 2 years, then annual", "note": "Multi-focal spikes, generalised spike-wave; IPS response in 8%; CSWS surveillance. Ambulatory EEG if infrequent events."},
    {"item": "MRI Brain 3T", "frequency": "Once at diagnosis; repeat if regression", "note": "Thin corpus callosum (40%), simplified gyral pattern (20%), periventricular white matter changes (15%). Functional MRI not routine."},
    {"item": "Bayley Scales / VABS (Developmental)", "frequency": "Annually", "note": "Bayley-4 (motor + cognitive + language); VABS-3 (adaptive behaviour). Regression monitoring: ≥2 SD decline = regression episode."},
    {"item": "ABC-Community (Aberrant Behaviour Checklist)", "frequency": "Monthly if on LEV/CLB", "note": "Behavioural monitoring for AED-associated irritability/hyperactivity (especially LEV, PHT). Caregiver-reported."},
    {"item": "VPA TDM + LFT + FBC + NH3", "frequency": "Monthly × 3 months; then 3-monthly", "note": "Hyperammonaemia in ID without encephalopathy common (asymptomatic); L-carnitine supplement if VPA-associated. LFT: POLG1 hepatotoxicity screen."},
    {"item": "POLG1 screening before VPA", "frequency": "Once before VPA initiation", "note": "Alpers-Huttenlocher risk: biallelic POLG1 + VPA = fatal hepatotoxicity. Standard in all developmental epilepsy."},
    {"item": "IGF-1 levels + Blood glucose", "frequency": "Before + during IGF-1 trial (monthly)", "note": "Hypoglycaemia risk (administer with meal; blood glucose 30 min post-dose). IGF-1 levels: target low-normal range (age-adjusted)."},
    {"item": "Ophthalmology (VEP + ERG if VGB considered)", "frequency": "Annual baseline; if VGB used", "note": "VGB visual field monitoring impossible in non-verbal PMS; ERG as surrogate (insensitive). VGB should be avoided entirely."},
    {"item": "SUDEP risk counselling + alarm", "frequency": "Annual (diagnosis + review)", "note": "SUDEP risk elevated in drug-resistant developmental epilepsy. NightWatch / Empatica / SAMi-3 seizure alarms. Prone sleeping avoidance counselling."},
    {"item": "VPPP (Valproate Pregnancy Prevention Programme)", "frequency": "Annually (female adolescents/adults)", "note": "VPA teratogenicity (spina bifida, neurodevelopmental CI). MHRA VPPP 2021: mandatory counselling + contraception discussion from age 10."},
]

# ── Lifecycle stages ──────────────────────────────────────────────────────────
LIFECYCLE_STAGES = [
    {
        "stage": "Neonatal / Early Infantile (0-12 months)",
        "focus": "Hypotonia recognition + genetic diagnosis",
        "management": "CMA + SHANK3 confirmation; feeding support (NG tube in 30%); early speech/physio OT referral; ASD surveillance",
        "seizure_risk": "Low (<5% infantile spasms); hypotonia main concern",
        "precision": "IGF-1 trial not yet started (evidence base: ≥18 months)",
    },
    {
        "stage": "Toddler / Preschool (1-5 years)",
        "focus": "Seizure onset + ASD features + speech therapy",
        "management": "First seizure: CMA if not done; VPA or LTG; AED decision tree; speech/OT/ABA; melatonin for sleep",
        "seizure_risk": "Moderate (30-40% will have first seizure in this window); febrile triggers",
        "precision": "IGF-1 trial: 18 months-5 years (Kolevzon trial age range); enrol in registry",
    },
    {
        "stage": "School Age (5-12 years)",
        "focus": "Seizure optimisation + regression surveillance + CSWS",
        "management": "Annual overnight EEG (CSWS); AED review; CSWS → corticosteroids; educational planning; SRS/regression protocol",
        "seizure_risk": "Ongoing; regression episodes require urgent neurology review",
        "precision": "IGF-1 or Phase II trial if eligible; KD for DRE",
    },
    {
        "stage": "Adolescence (12-18 years)",
        "focus": "Seizure control + puberty + VPPP (females) + transition",
        "management": "Catamenial CLB; VPPP counselling; VPA alternatives for females; SUDEP counselling; school/residential transition",
        "seizure_risk": "Hormonal influences; catamenial pattern emerging; non-compliance risk",
        "precision": "Continued IGF-1 or trial review; KD in DRE",
    },
    {
        "stage": "Adulthood (18+ years)",
        "focus": "Long-term AED; carer support; residential; employment where possible",
        "management": "Adult neurology transition; SUDEP annual review; bone DEXA (long-term AED); annual AED review; residential care needs",
        "seizure_risk": "Stable for majority; 20-25% remain drug-resistant",
        "precision": "Maintenance therapy; IGF-1 discontinued (adult data limited)",
    },
]

# ── Concept / definition catalog ──────────────────────────────────────────────
CONCEPT_CATALOG = [
    {
        "concept": "SHANK3-22q13.33-PMS",
        "definition": (
            "SHANK3 (22q13.33) haploinsufficiency causes Phelan-McDermid Syndrome (PMS): "
            "ASD, intellectual disability, absent/minimal speech, neonatal hypotonia, "
            "seizures (30-40%), and SHANK3 Regression Syndrome (SRS). "
            "Deletion size modulates severity: larger deletions → more gene loss → worse prognosis."
        ),
    },
    {
        "concept": "Postsynaptic-Density-Scaffold",
        "definition": (
            "SHANK3 is the master PSD scaffold: anchors mGluR5 (via Homer), NMDAR "
            "(via SAPAP/GKAP), AMPAR (via TARP), and actin cytoskeleton (via IRSp53). "
            "Haploinsufficiency → 40% reduced dendritic spine density → E/I imbalance → "
            "epilepsy + ASD."
        ),
    },
    {
        "concept": "IGF-1-Precision-SHANK3",
        "definition": (
            "IGF-1 (mecasermin/Increlex) activates PI3K→Akt→mTOR→S6K1, increasing "
            "SHANK3 protein translation from the intact allele. Phase I (Kolevzon 2014): "
            "social + motor + seizure improvements. Avoid VPA co-administration (mTOR inhibition "
            "blunts IGF-1 effect). Phase II NCT02677935 ongoing."
        ),
    },
    {
        "concept": "SHANK3-Regression-Syndrome (SRS)",
        "definition": (
            "50-70% of PMS patients undergo periods of language/motor skill loss "
            "(SHANK3 Regression Syndrome). Triggers: fever, anaesthesia, illness, sleep disruption. "
            "Partially reversible over 6-12 months. CSWS during regression is treatable "
            "(corticosteroids). Regression ≠ seizure: EEG mandatory."
        ),
    },
    {
        "concept": "CSWS-PMS-Regression",
        "definition": (
            "Continuous Spike-Wave during Slow Sleep (CSWS) is a rare (<5%) but treatable "
            "EEG pattern in PMS during regression episodes. CSWS → impairs slow-wave "
            "sleep-dependent memory consolidation → accelerates skill loss. "
            "Diagnosis: overnight video-EEG. Treatment: prednisolone 2 mg/kg × 4 weeks."
        ),
    },
    {
        "concept": "Post-Anaesthetic-Regression",
        "definition": (
            "50-65% PMS patients show language/motor regression after general anaesthesia "
            "(Holder 2012). Mechanism: volatile agent NMDA block + GABAA enhancement → "
            "synaptic plasticity disruption in SHANK3-haploinsufficient brain. "
            "TIVA (propofol) preferred; pre-op IGF-1 optimisation; mandatory post-op PICU monitoring."
        ),
    },
    {
        "concept": "mGluR5-LTD-SHANK3",
        "definition": (
            "SHANK3 scaffolds mGluR5 at the PSD via Homer1b/c. Haploinsufficiency → "
            "reduced mGluR5-PSD coupling → impaired mGluR5-LTD (long-term depression). "
            "LTD/LTP imbalance → net excitatory shift → epilepsy + repetitive behaviours (ASD). "
            "mGluR5 negative modulators (MPEP, MTEP) explored in Shank3 mouse models."
        ),
    },
    {
        "concept": "VPA-mTOR-IGF1-Interaction",
        "definition": (
            "Valproic acid inhibits mTOR complex 1 (via HDAC inhibition + TSC upregulation) → "
            "reduces protein translation → blunts IGF-1 therapeutic increase of SHANK3 protein. "
            "Clinical: VPA+IGF-1 combination gives inferior social + seizure outcomes vs IGF-1 alone. "
            "Switch VPA → LTG/LEV before IGF-1 trial (4-week washout)."
        ),
    },
    {
        "concept": "VGB-AVOID-NonVerbal-PMS",
        "definition": (
            "Vigabatrin causes permanent visual field defects (nasal ring scotoma, 30-40%) "
            "requiring regular perimetry. PMS patients (non-verbal, ID, ASD) cannot cooperate "
            "with perimetry → undetectable irreversible visual loss. ABSOLUTE AVOIDANCE in PMS: "
            "use VPA, LTG, or LEV instead. ERG is insensitive surrogate."
        ),
    },
    {
        "concept": "Catamenial-Epilepsy-PMS",
        "definition": (
            "20-25% adolescent/adult females with PMS and epilepsy show catamenial "
            "(perimenstrual) seizure exacerbation. Mechanism: oestrogen ↑ neuronal excitability + "
            "progesterone withdrawal. Management: clobazam perimenstrual (days -3 to +3, 0.5 mg/kg). "
            "VPPP counselling for VPA use in females from age 10."
        ),
    },
    {
        "concept": "LEV-Behavioural-Toxicity-ASD",
        "definition": (
            "Levetiracetam causes behavioural irritability/aggression in 15-30% of patients "
            "with ASD/ID (vs 3-8% in focal epilepsy). Mechanism: SV2A in frontal networks "
            "regulating social behaviour circuits. Preferred alternative: brivaracetam "
            "(selective SV2A, ~5% behavioural events). Document consent; monthly ABC-Community."
        ),
    },
    {
        "concept": "POLG1-VPA-Mandatory-Screen",
        "definition": (
            "Biallelic POLG1 mutations + valproate = Alpers-Huttenlocher syndrome: "
            "progressive hepatotoxicity, fatal in 3-6 months. Mandatory screening before "
            "VPA initiation in ANY developmental epilepsy. Turnaround 7-14 days; "
            "bridge with LEV+CLB while awaiting. Never skip even in urgent cases."
        ),
    },
    {
        "concept": "CMA-SHANK3-Diagnosis-Pathway",
        "definition": (
            "Chromosomal Microarray (CMA) is first-line for PMS diagnosis: detects 22q13.3 "
            "deletion (75% of PMS). CMA-negative PMS → WES/WGS for SHANK3 point mutations. "
            "MLPA/ddPCR for mosaic cases. Deletion size from CMA predicts: IB2/MAPK8IP2 loss "
            "(cerebellar signs), ADSL loss (purine defect, regression), ARSA loss (late-onset "
            "metachromatic leukodystrophy — rare second hit)."
        ),
    },
    {
        "concept": "KD-mTOR-Interaction-IGF1",
        "definition": (
            "Ketogenic diet inhibits mTOR via beta-hydroxybutyrate and adenosine pathways. "
            "This is OPPOSITE to IGF-1 (mTOR activation). Do NOT combine KD + IGF-1 "
            "(opposing mechanisms cancel SHANK3 protein synthesis benefit). "
            "Choose: KD for DRE without IGF-1 trial plan; OR IGF-1 with LTG-based AED regimen."
        ),
    },
    {
        "concept": "SUDEP-DRE-PMS",
        "definition": (
            "SUDEP risk in drug-resistant developmental epilepsy ~1/200/year. PMS patients "
            "often non-verbal → unable to call for help postictal. Standard of care: "
            "NightWatch/SAMi-3/Empatica seizure detection alarms; prone sleeping avoidance; "
            "shared bedroom for nocturnal seizures; SUDEP conversation at diagnosis."
        ),
    },
]

# ── Threshold / threshold catalog ─────────────────────────────────────────────
THRESHOLD_CATALOG = [
    {"threshold": "IGF-1 hypoglycaemia (blood glucose)", "value": "<3.5 mmol/L (63 mg/dL)", "action": "Hold IGF-1 dose; administer carbohydrate; recheck in 15 min"},
    {"threshold": "VPA TDM target", "value": "75-120 µg/mL (520-830 µmol/L)", "action": "Dose adjust; check LFT + NH3 if supratherapeutic"},
    {"threshold": "NH3 hyperammonaemia (VPA)", "value": ">80 µmol/L (asymptomatic); >150 µmol/L (symptomatic)", "action": "L-carnitine supplement; consider VPA dose reduction"},
    {"threshold": "Regression diagnosis (Bayley)", "value": "≥2 SD decline from baseline in any domain", "action": "Urgent overnight EEG (CSWS?); neurology review within 48h"},
    {"threshold": "Febrile seizure admission (PMS)", "value": "Temperature ≥38°C + any new seizure + PMS", "action": "Hospital admission — dual regression + seizure watch"},
    {"threshold": "LTG titration rate", "value": "Max increase 25 mg/2 weeks (no enzyme inducers)", "action": "Slower if on VPA (LTG + VPA = doubled LTG levels)"},
    {"threshold": "Corticosteroid CSWS response", "value": "≥50% slow-wave CSWS index reduction at 4 weeks", "action": "Continue taper; non-response → add VPA or LEV"},
    {"threshold": "Deletion size (severity threshold)", "value": ">3 Mb deletion = high seizure + regression risk", "action": "Enhanced surveillance; ADSL/IB2 gene analysis; early KD/IGF-1 consideration"},
    {"threshold": "POLG1 turnaround", "value": "7-14 days (bridge VPA with LEV+CLB)", "action": "Do NOT start VPA before result in DEE context"},
    {"threshold": "SUDEP alarm threshold", "value": "≥2 GTCS/year or nocturnal GTCS", "action": "Seizure alarm mandatory (NightWatch/SAMi-3); shared bedroom"},
    {"threshold": "LTG rash — DRESS risk", "value": "Rash within 8 weeks of LTG initiation", "action": "Stop LTG immediately; dermatology review; no rechallenge if DRESS"},
    {"threshold": "IGF-1 dose cap", "value": "0.12 mg/kg SC BID (max)", "action": "Do not exceed; hypoglycaemia risk above cap; renal/hepatic dose-adjust"},
]

# ── Reference catalog ─────────────────────────────────────────────────────────
REFERENCE_CATALOG = [
    {"ref": "Kolevzon 2014 — J Autism Dev Disord", "summary": "Phase I IGF-1 trial in PMS (n=9): open-label, significant improvements in social responsiveness, adaptive behaviour, motor function; seizure reduction in 3/4 who had seizures. First evidence of IGF-1 efficacy in SHANK3."},
    {"ref": "Soorya 2013 — Sci Transl Med", "summary": "PMS natural history (n=32): seizures 30%, regression 50-70%, medical comorbidities; consensus management recommendations; first systematic characterisation of SRS."},
    {"ref": "Leblond 2014 — Am J Hum Genet", "summary": "Large SHANK3 cohort (n=72 families): point mutations, truncating and missense, genotype-phenotype correlation; mosaicism characterisation; ASD + epilepsy prevalence."},
    {"ref": "Bozdagi 2010 — Mol Autism", "summary": "First Shank3 heterozygous mouse model: reduced mGluR5-LTD, dendritic spine density, AMPAR surface expression; social deficits; rescued partially by IGF-1."},
    {"ref": "Holder 2012 — J Child Neurol", "summary": "Case series of post-anaesthetic regression in PMS (n=8): 6/8 language/motor regression after general anaesthesia; TIVA recommendation; regression duration 6-18 months."},
    {"ref": "Bonaglia 2011 — Eur J Hum Genet", "summary": "22q13 deletion breakpoint mapping; SHANK3 as minimum critical region for PMS; correlation of deletion size with seizure prevalence and severity."},
]

# ── Patient roster ─────────────────────────────────────────────────────────────
def _make_patients():
    random.seed(42)
    patients = []
    first_names = [
        "Amara","Caleb","Dani","Elias","Freya","Giulia","Henry","Isla","Jonas","Kira",
        "Leo","Mia","Noel","Olivia","Pablo","Quinn","Rosa","Soren","Tara","Udo",
        "Vera","Will","Xena","Yara","Zane","Bella","Cody","Demi","Erik","Faye",
        "Grant","Hana","Ivan","Jade","Kyle","Luna","Marco","Nina","Omar","Petra",
    ]
    etiology_weights = [(ETIOLOGY_CATALOG[i], ETIOLOGY_CATALOG[i]["pct"]) for i in range(len(ETIOLOGY_CATALOG))]

    def pick_etiology():
        r = random.randint(1, 100)
        cumulative = 0
        for e, pct in etiology_weights:
            cumulative += pct
            if r <= cumulative:
                return e
        return ETIOLOGY_CATALOG[0]

    for i, name in enumerate(first_names):
        etio = pick_etiology()
        age_dx = random.randint(1, 5)
        has_seizures = random.random() < 0.37
        n_aed = random.randint(1, 3) if has_seizures else 0
        regression = random.random() < 0.60
        igf1_trial = etio["igf1_candidate"] and random.random() < 0.30
        drug_resistant = has_seizures and random.random() < 0.28
        patients.append({
            "id": i + 1,
            "name": name,
            "age_dx": age_dx,
            "etiology": etio["category"],
            "deletion_size_mb": round(random.uniform(0.3, 8.5), 1) if "deletion" in etio["category"] else None,
            "has_seizures": has_seizures,
            "n_aed": n_aed,
            "regression_episodes": random.randint(0, 4) if regression else 0,
            "igf1_trial": igf1_trial,
            "drug_resistant": drug_resistant,
            "kd": drug_resistant and random.random() < 0.40,
        })
    return patients


PATIENTS = _make_patients()

# ── Public API functions ───────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    with_seizures = sum(1 for p in PATIENTS if p["has_seizures"])
    drug_resistant = sum(1 for p in PATIENTS if p["drug_resistant"])
    igf1 = sum(1 for p in PATIENTS if p["igf1_trial"])
    regression = sum(1 for p in PATIENTS if p["regression_episodes"] > 0)
    on_kd = sum(1 for p in PATIENTS if p["kd"])
    n_aed_patients = sum(1 for p in PATIENTS if p["n_aed"] > 0)
    avg_deletion = round(
        sum(p["deletion_size_mb"] for p in PATIENTS if p["deletion_size_mb"] is not None) /
        max(1, sum(1 for p in PATIENTS if p["deletion_size_mb"] is not None)), 1
    )
    return {
        "dashboard": "SHANK3 Epilepsy — Phelan-McDermid Syndrome / 22q13.33",
        "gene": "SHANK3",
        "locus": "22q13.33",
        "protein": "SH3 and Multiple Ankyrin Repeat Domains 3 (Postsynaptic Density Scaffold)",
        "omim_syndrome": "PMS #606232",
        "omim_gene": "*606230",
        "cohort_size": n,
        "with_seizures": with_seizures,
        "seizure_prevalence_pct": round(with_seizures / n * 100),
        "drug_resistant_pct": round(drug_resistant / max(1, with_seizures) * 100),
        "igf1_trial_pct": round(igf1 / n * 100),
        "regression_pct": round(regression / n * 100),
        "on_kd_pct": round(on_kd / n * 100),
        "mean_aed_count": round(sum(p["n_aed"] for p in PATIENTS) / n, 1),
        "mean_deletion_mb": avg_deletion,
        "precision_therapy": "IGF-1 / Mecasermin (Increlex) — mTOR-SHANK3 pathway restoration",
        "key_ci": "VGB (visual field monitoring impossible), PHT/CBZ (behaviour worsening), VPA+IGF-1",
        "regression_syndrome": "SHANK3 Regression Syndrome (SRS) — 50-70%, fever-triggered, partially reversible",
        "etiology_classes": len(ETIOLOGY_CATALOG),
        "seizure_types": len(SEIZURE_CATALOG),
        "triggers": len(TRIGGER_CATALOG),
        "treatments": len(TREATMENT_CATALOG),
        "contraindications": len(CI_CATALOG),
        "monitoring_items": len(MONITORING_CATALOG),
        "lifecycle_stages": len(LIFECYCLE_STAGES),
        "concepts": len(CONCEPT_CATALOG),
        "thresholds": len(THRESHOLD_CATALOG),
        "references": len(REFERENCE_CATALOG),
        "standards": [
            "ILAE-2022", "NICE-NG217", "Kolevzon-2014-JADD",
            "Soorya-2013-SciTranslMed", "Leblond-2014-AJHG", "Bozdagi-2010-MolAutism",
            "Holder-2012-JChildNeurol", "Bonaglia-2011-EJHG",
            "MHRA-VPPP-2021", "ACMG-AMP-2015", "NICE-NG224-2022", "WHO-ICF-2019",
        ],
    }


def get_breakdown():
    patients_export = []
    for p in PATIENTS:
        patients_export.append({
            "id": p["id"],
            "name": p["name"],
            "age_dx": p["age_dx"],
            "etiology": p["etiology"],
            "deletion_size_mb": p["deletion_size_mb"],
            "has_seizures": p["has_seizures"],
            "n_aed": p["n_aed"],
            "regression_episodes": p["regression_episodes"],
            "igf1_trial": p["igf1_trial"],
            "drug_resistant": p["drug_resistant"],
            "kd": p["kd"],
        })
    return {
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_CATALOG,
        "triggers": TRIGGER_CATALOG,
        "treatments": TREATMENT_CATALOG,
        "contraindications": CI_CATALOG,
        "monitoring": MONITORING_CATALOG,
        "lifecycle": LIFECYCLE_STAGES,
        "patients": patients_export,
    }


def get_definitions():
    return {
        "concepts": CONCEPT_CATALOG,
        "thresholds": THRESHOLD_CATALOG,
        "references": REFERENCE_CATALOG,
        "standards": [
            {"standard": "ILAE-2022", "scope": "Epilepsy classification, DEE definition"},
            {"standard": "NICE-NG217", "scope": "Epilepsy guideline (AED initiation, monitoring)"},
            {"standard": "Kolevzon-2014-JADD", "scope": "IGF-1 phase I trial in PMS"},
            {"standard": "Soorya-2013-SciTranslMed", "scope": "PMS natural history, SRS definition"},
            {"standard": "Leblond-2014-AJHG", "scope": "SHANK3 genotype-phenotype correlation"},
            {"standard": "Bozdagi-2010-MolAutism", "scope": "Shank3 het mouse model — synaptic rescue by IGF-1"},
            {"standard": "Holder-2012-JChildNeurol", "scope": "Post-anaesthetic regression in PMS"},
            {"standard": "Bonaglia-2011-EJHG", "scope": "22q13 deletion mapping, SHANK3 minimum critical region"},
            {"standard": "MHRA-VPPP-2021", "scope": "Valproate Pregnancy Prevention Programme"},
            {"standard": "ACMG-AMP-2015", "scope": "Variant classification pathogenicity criteria"},
            {"standard": "NICE-NG224-2022", "scope": "Transition from child to adult epilepsy services"},
            {"standard": "WHO-ICF-2019", "scope": "International Classification of Functioning — disability assessment"},
        ],
    }
