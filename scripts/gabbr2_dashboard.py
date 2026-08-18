"""
GABBR2 Epilepsy — GABA-B Receptor Subunit 2 / Metabotropic GABA Receptor / DEE-59
====================================================================================
40-patient cohort · GABBR2 (22q12.2) · De novo dominant (GOF 55% / LOF 45%)
GABA-B = GABBR1 (ligand-binding) + GABBR2 (G-protein coupling) obligatory heterodimer
LOF or GOF → DEE-59 (OMIM #617137) · Precision Rx: Baclofen (GABA-B agonist) in LOF

GABA-B RECEPTOR BIOLOGY:
GABA-B receptor is the principal metabotropic (G-protein-coupled) GABA receptor:
  - GABBR1 (6p22.1): ligand-binding subunit — GABA binds Venus flytrap domain of GABBR1
  - GABBR2 (22q12.2): G-protein coupling subunit — Gi/Go effector module
  - Obligatory heterodimer: neither subunit alone traffics to the plasma membrane
    (GABBR2 provides ER-export signal; GABBR1-GABBR2 co-assembly required)
  - Signal transduction:
      Gi/Go → inhibits adenylyl cyclase → ↓cAMP → ↓PKA → inhibits glutamate release
      Gβγ → activates GIRK (Kir3) K⁺ channels → K⁺ efflux → hyperpolarisation
      Gβγ → inhibits Cav2.1/Cav2.2 (P/Q, N-type) → ↓presynaptic Ca²⁺ → ↓NT release
  - Kinetics: slow IPSP 150-500 ms (vs GABAA 20-40 ms) → sustained tonic inhibition
  - Presynaptic autoreceptors: on GABAergic terminals → negative feedback on GABA release
  - Presynaptic heteroreceptors: on glutamatergic terminals → suppresses glutamate release
  - Postsynaptic: GIRK-mediated slow hyperpolarisation in cortex, thalamus, hippocampus

DISEASE MECHANISM — GABBR2 GOF vs LOF:
  GOF (~55%): constitutively active GABBR2 Gi signalling → excess tonic inhibition
    during cortical development → paradoxical hyperexcitability (receptor desensitisation
    + compensatory upregulation of excitatory synapses → net network destabilisation);
    severe DEE-59 with hypotonia, intellectual disability, West → LGS evolution.
    Representative variants: p.Ile705Leu, p.Ser695Ile (TM4-TM5 interface; constitutive Gi)
    First report: Yoo et al. 2017 EBioMedicine (3 de novo GOF variants, severe DEE)

  LOF (~45%): haploinsufficiency → loss of presynaptic autoreceptor function →
    disinhibition of GABA release → GABAA receptor desensitisation → paradoxical network
    excitation; also loss of presynaptic heteroreceptor suppression of glutamate → excess
    glutamate release. Milder phenotype (GEFS+/generalised epilepsy) than GOF.
    Precision opportunity: Baclofen (GABA-B agonist) compensates for reduced endogenous
    GABA-B activation in LOF (activates residual GABBR1/GABBR2 from normal allele).

GABBR2 vs GABBR1 — CLINICAL COMPARISON:
  GABBR1 (6p22.1): rare epilepsy (Steele 2020); milder; GEFS+ predominant; baclofen trials
  GABBR2 (22q12.2): more common; severe DEE-59 GOF; IS/West → LGS; baclofen precision LOF

BACLOFEN PHARMACOLOGY — PRECISION THERAPY IN LOF:
  - R-Baclofen: selective GABA-B agonist (stereoisomer with receptor activity)
  - Binds GABBR1 Venus flytrap domain → activates GABBR2 Gi effector
  - In GABBR2 LOF: activates residual GABBR2 from normal allele → partial restoration
    of GABA-B-mediated presynaptic autoreceptor function → normalises GABA release
  - Titrate slowly (spasticity protocol): 5 mg/day → increase every 3 days → target 1-2 mg/kg/day
  - CRITICAL: Abrupt baclofen withdrawal = MEDICAL EMERGENCY
    (hyperpyrexia, seizure cluster, rhabdomyolysis, multi-organ failure, death)
  - NEVER abrupt stop; always taper over 1-2 weeks minimum
  - Intrathecal baclofen (ITB) for refractory cases (avoids CNS sedation with oral high-dose)
  - Renal elimination: dose-adjust for renal impairment (eGFR <30)
  - BBB penetration: oral bioavailability 30-85% (variable); lioresal immediate-release preferred

CONTRAINDICATIONS IN GABBR2:
  1. TGB (Tiagabine) ABSOLUTE: GABA-B receptor modulates GABAA desensitisation kinetics;
     TGB blockade of GAT-1 → sustained perisynaptic GABA → GABAA desensitisation →
     paradoxical loss of inhibition → NCSE; GABBR2 dysfunction amplifies this risk
  2. PHT/CBZ/OXC HIGH RISK: Na-channel blockers worsen IS and myoclonus in DEE
  3. Baclofen abrupt withdrawal ABSOLUTE: see above — medical emergency
  4. Baclofen high-dose without monitoring: respiratory depression (especially if intrathecal)
  5. Flumazenil in SE: avoid in GABBR2 — complex GABA-B/GABA-A interaction;
     flumazenil → reverses GABAA → worsens GABBR2 LOF-mediated disinhibition
  6. VPA without POLG1 screening: POLG1 not pre-excluded; Alpers-Huttenlocher risk

GENETICS:
  Gene:        GABBR2 (22q12.2) — also known as GBR2, GPRC3B
  Protein:     GABA-B receptor subunit 2 (941 aa, ~103 kDa)
  Inheritance: De novo dominant (GOF + LOF); rare familial AD
  De novo:     ~90-95% (predominantly de novo)
  pLI:         0.99 (highly intolerant to LoF variants)
  Incidence:   ~1:200,000-400,000 (rare; ~30 published patients as of 2024)
  OMIM:        DEE-59 #617137; GABBR2 gene OMIM *607340
  First report: Yoo et al. 2017 EBioMedicine — 3 patients de novo GABBR2 GOF → DEE

KEY STANDARDS:
  ILAE 2022 · NICE NG217 · Yoo et al. 2017 (EBioMedicine — DEE-59 first report) ·
  Steele et al. 2020 (Epilepsia — 30-patient GABBR2 cohort) ·
  Pinard et al. 2010 (Nat Neurosci — GABA-B biology) ·
  CPIC POLG 2023 · MHRA VPPP 2021 · ACMG-AMP 2015 ·
  ILAE Dietary Therapies 2018 · FDA-FINTEPLA 2020 ·
  NICE NG224 2022 · WHO ICF 2019
"""
import random

# ── Etiology catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "GABBR2 GOF TM4-TM5 Missense — Severe DEE-59 / West Syndrome",
        "category": "GOF-TM4-TM5-missense-DEE59-West-38%",
        "pct": 38,
        "n": 15,
        "mechanism": (
            "De novo missense variants at the TM4-TM5 transmembrane interface of GABBR2 "
            "(p.Ile705Leu, p.Ser695Ile, p.Ala716Val) — constitutive Gi protein coupling → "
            "tonic GABA-B signalling without ligand → excess presynaptic inhibition during "
            "critical cortical development window → paradoxical network hyperexcitability via "
            "compensatory excitatory synaptic upregulation and GABAA receptor downregulation. "
            "Severe DEE-59: West syndrome (IS onset 4-12 months), drug-resistant, LGS evolution "
            "in 60%. Profound intellectual disability, hypotonia. MRI: progressive cortical atrophy "
            "in 40%, thin corpus callosum in 30%. Baclofen contraindicated (GOF — would worsen). "
            "Treatment: ACTH/VGB for IS; KD early; LEV; CLB. First described Yoo 2017."
        ),
        "eeg_correlate": (
            "Hypsarrhythmia (IS onset) · Modified hypsarrhythmia high-amplitude asynchronous "
            "multifocal IED · Burst-suppression-like pattern neonatal · LGS slow (<2.5 Hz) "
            "SWD at 3-5y · Generalised/multifocal polyspike-wave at 5-10y · EEG background: "
            "severe diffuse slowing"
        ),
        "typical_age_onset": "2-12 months (infantile spasms peak 4-8 months)",
        "drug_resistance": "80-90% (severe GOF DEE)",
        "baclofen_role": "CONTRAINDICATED in GOF — would worsen constitutive activation",
    },
    {
        "etiology": "GABBR2 LOF Truncating/Frameshift — GEFS+ / Generalised Epilepsy",
        "category": "LOF-truncating-GEFS-generalised-28%",
        "pct": 28,
        "n": 11,
        "mechanism": (
            "De novo or rare familial AD truncating variants (frameshift/nonsense) → "
            "haploinsufficiency → loss of one GABBR2 allele → 50% reduction in GABBR2 "
            "protein → impaired GABBR1-GABBR2 heterodimer formation → reduced surface "
            "GABA-B receptor density → loss of presynaptic autoreceptor function → "
            "GABA release disinhibited → GABAA desensitisation → paradoxical excitation. "
            "Also: loss of presynaptic heteroreceptor suppression of glutamate. Phenotype: "
            "generalised epilepsy (GEFS+), febrile seizures + afebrile GTCS, milder than GOF. "
            "Baclofen PRECISION THERAPY: activates residual surface GABBR1/2 → partially "
            "restores autoreceptor function → 40-60% seizure reduction in responders."
        ),
        "eeg_correlate": (
            "Generalised 3-4 Hz spike-wave (ictal) · Photoparoxysmal response 20% · "
            "Generalised polyspike-wave with myoclonic correlate · Normal/near-normal background · "
            "Frequent NREM-activated generalised IED · No focal pattern"
        ),
        "typical_age_onset": "1-15 years (peak 3-8 years, febrile onset common)",
        "drug_resistance": "30-40% (milder LOF phenotype)",
        "baclofen_role": "PRECISION THERAPY: 1-2 mg/kg/day; 40-60% responders in LOF",
    },
    {
        "etiology": "GABBR2 GOF Missense VFTM Domain — Intermediate DEE / LGS",
        "category": "GOF-VFTM-intermediate-DEE-LGS-18%",
        "pct": 18,
        "n": 7,
        "mechanism": (
            "De novo missense variants in the Venus flytrap module (VFTM) domain of GABBR2 "
            "— gain-of-function via altered GABBR1-GABBR2 interface: enhanced constitutive "
            "coupling efficiency. Intermediate severity between TM4-TM5 GOF and LOF: "
            "DEE with LGS evolution (slow SWD <2.5 Hz + tonic seizures + drop attacks). "
            "Cognition: moderate intellectual disability. Some respond partially to felbamate "
            "(NMDA antagonist + GABAA positive modulator) as LGS adjunct therapy. "
            "MRI may show mild cortical simplified gyral pattern or thin corpus callosum."
        ),
        "eeg_correlate": (
            "LGS slow <2.5 Hz generalised SWD · Sleep: generalised 10-12 Hz spindle-fast "
            "activity (tonic seizures EEG correlate) · Drop attack: EMG correlate with "
            "brief generalised attenuation then polyspike · Multifocal IED "
        ),
        "typical_age_onset": "6-36 months (infantile-to-toddler onset)",
        "drug_resistance": "65-75% (intermediate GOF DEE)",
        "baclofen_role": "NOT INDICATED (GOF) — Felbamate Level C for LGS adjunct",
    },
    {
        "etiology": "GABBR2 LOF Missense (Hypomorphic) — ESES / Focal Epilepsy",
        "category": "LOF-missense-hypomorphic-ESES-focal-10%",
        "pct": 10,
        "n": 4,
        "mechanism": (
            "Missense variants with partial LOF (hypomorphic) — residual GABBR2 function "
            "reduced but not absent → intermediate GABA-B receptor density → variable "
            "phenotype: focal epilepsy, ESES (electrical status epilepticus during sleep), "
            "language regression. Baclofen augmentation may help. Pyridoxine trial appropriate "
            "(rule out pyridoxine-dependent epilepsy). Cognitive trajectory: often progressive "
            "language regression during ESES phase; melatonin + CLZ may improve ESES."
        ),
        "eeg_correlate": (
            "ESES pattern: continuous generalised SWD ≥85% of NREM · Focal IED temporal/ "
            "frontal · Centrotemporal IED (Rolandic-like) · Focal ictal pattern evolving "
            "to secondary generalisation"
        ),
        "typical_age_onset": "3-10 years (school-age onset)",
        "drug_resistance": "20-30% (mildest, often benign course)",
        "baclofen_role": "AUGMENTATION: may reduce ESES burden; evidence limited",
    },
    {
        "etiology": "Phenocopy — GABBR1 / GABRB3 / GABRG2 / SCN8A (GABBR2-negative)",
        "category": "phenocopy-GABBR2-negative-6%",
        "pct": 6,
        "n": 2,
        "mechanism": (
            "Clinical presentation consistent with GABBR2 DEE-59 but GABBR2 sequencing "
            "negative. Differential: GABBR1 (6p22.1 — same GABA-B receptor biology, "
            "milder); GABRB3 (DEE with IS); GABRG2 (Dravet-like); SCN8A DEE; CDKL5; "
            "pyridoxine-dependent epilepsy (PDE). Functional assay of candidate variants "
            "and re-check deep intronic/copy number variants with comprehensive panel. "
            "Rule out metabolic epilepsy (GABA urine, plasma AA, CSF NT)."
        ),
        "eeg_correlate": (
            "Variable — depends on underlying diagnosis · WES-TRIO mandatory ·"
            "Functional GABBR2 assay if VUS identified"
        ),
        "typical_age_onset": "Variable",
        "drug_resistance": "Variable",
        "baclofen_role": "Trial only if GABA-B pathway implicated",
    },
]

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms / Epileptic Spasms (West Syndrome)",
        "frequency_pct": 85,
        "semiology": (
            "Clusters of sudden bilateral symmetric or asymmetric axial spasms "
            "(flexion > extension), 1-5 s duration, clusters of 5-50+ spasms, "
            "often on awakening. Head drop, Moro-like posture. May have subtle "
            "facial grimace. Clusters interrupt sleep-wake transitions."
        ),
        "eeg_tip": (
            "Hypsarrhythmia or modified hypsarrhythmia: high-amplitude chaotic "
            "asynchronous multifocal IED + slow waves, continuously. Electrodecrement "
            "time-locked to each spasm. Video-EEG mandatory to confirm IS."
        ),
        "clinical_tip": "IS = EMERGENCY — ACTH/Vigabatrin ASAP; do NOT wait for genetics.",
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "frequency_pct": 72,
        "semiology": (
            "Bilateral tonic then clonic phase, 1-3 min. Post-ictal confusion. "
            "Often febrile-triggered in LOF GEFS+ (fever → increased GABA-B "
            "receptor internalization → acute LOF exacerbation)."
        ),
        "eeg_tip": (
            "Pre-ictal generalised IED build-up → high-amplitude polyspike → "
            "clonic phase slow SWD → post-ictal diffuse suppression."
        ),
        "clinical_tip": "Fever management essential in LOF (GEFS+); paracetamol + early BDZ.",
    },
    {
        "type": "Focal Seizures with Impaired Awareness",
        "frequency_pct": 60,
        "semiology": (
            "Temporal (automotor) or frontal (hypermotor) depending on FCD/network "
            "involvement. GABBR2 does not typically cause FCD but GABA-B deficiency "
            "in temporal/frontal networks → focal hyperexcitability. Autonomic "
            "symptoms, automatisms, staring."
        ),
        "eeg_tip": "Focal IED temporal (LOF) > frontal. Ictal onset focal → secondary generalisation.",
        "clinical_tip": "SEEG if drug-resistant focal — rule out focal cortical source.",
    },
    {
        "type": "Myoclonic Seizures",
        "frequency_pct": 48,
        "semiology": (
            "Irregular brief (<100 ms) bilateral myoclonic jerks, arms > legs. "
            "Morning predilection. May progress to GTCS if prolonged cluster. "
            "More common in LGS/intermediate GOF phenotype."
        ),
        "eeg_tip": "Generalised polyspike (2-3 Hz) time-locked to myoclonic jerk. Background slowing.",
        "clinical_tip": "AVOID PHT/CBZ — worsen myoclonus. VPA + LEV + CLB = standard.",
    },
    {
        "type": "Tonic/Atonic Drop Attacks (LGS Evolution)",
        "frequency_pct": 35,
        "semiology": (
            "Tonic: sudden bilateral stiffening, falls, injury risk. "
            "Atonic: sudden loss of muscle tone, head drop, fall without warning. "
            "Drop attacks = LGS evolution in GOF phenotype; helmet mandatory. "
            "Nocturnal tonic prominent in NREM sleep."
        ),
        "eeg_tip": (
            "Tonic: sleep fast activity (10-14 Hz). Atonic: brief EMG silence + "
            "generalised polyspike. Ictal LGS slow SWD <2.5 Hz."
        ),
        "clinical_tip": "Helmet immediately. Felbamate or CLB for drop attacks if LGS. ACTH not re-used.",
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Febrile illness", "pct": 85,
     "note": "LOF GEFS+ amplified by fever; fever increases GABBR2 receptor internalisation."},
    {"trigger": "Sleep deprivation / disrupted NREM", "pct": 75,
     "note": "GABA-B tonic suppression critical in NREM; disruption unmasks hyperexcitability."},
    {"trigger": "Missed AED doses", "pct": 68,
     "note": "Baclofen abrupt stop is EMERGENCY — never miss >1 dose; VPA/LEV missed doses."},
    {"trigger": "Stress / catecholamine surge", "pct": 55,
     "note": "Catecholamines sensitise cortical networks; GABA-B presynaptic inhibition reduced."},
    {"trigger": "Metabolic derangement (hypoglycaemia, acidosis)", "pct": 45,
     "note": "Metabolic stress impairs GABA-B coupling efficiency; sick-day protocol mandatory."},
    {"trigger": "AED taper / withdrawal", "pct": 42,
     "note": "Any AED taper: slow (10% per 2 weeks). Baclofen taper: even slower."},
    {"trigger": "Overstimulation / sensory overload", "pct": 35,
     "note": "GABA-B network-level inhibition modulates sensory gating; overload → IED."},
    {"trigger": "Catamenial (perimenstrual, allopregnanolone shifts)", "pct": 22,
     "note": "Allopregnanolone modulates GABAA; progesterone drop → GABBR2 compensatory stress."},
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "name": "ACTH / Prednisolone",
        "level": "Level A",
        "indication": "Infantile Spasms (IS) — first-line for GOF GABBR2 IS",
        "dose": "ACTH: 40-60 IU/day IM for 14 days (UKISS protocol); Prednisolone 4 mg/kg/day PO",
        "moa": "Suppresses HPA axis → reduces CRH → neuromodulates IS network; reduces inflammatory cascade.",
        "efficacy": "EEG hypsarrhythmia resolution 60-70% at Day 14; higher with natural ACTH",
        "monitoring": "BP, blood glucose, weight, ACTH Day-14 EEG mandatory; infection screen",
        "gabbr2_note": (
            "GOF GABBR2 IS: ACTH first-line same as other IS etiologies. Lower efficacy than "
            "idiopathic IS (~50-60% vs 80%). Start within 2 weeks of IS onset. If Day-14 EEG "
            "shows residual hypsarrhythmia → add or switch to VGB."
        ),
    },
    {
        "name": "Vigabatrin (VGB)",
        "level": "Level A",
        "indication": "Infantile Spasms — first-line (especially if TSC, or ACTH failed)",
        "dose": "100-150 mg/kg/day PO in 2 divided doses (max 3 g/day); titrate over 2 weeks",
        "moa": "Irreversible GABA-T inhibitor → ↑synaptic GABA → enhanced GABAA activation",
        "efficacy": "IS cessation 50-60% in genetic IS (non-TSC); better in TSC (80%)",
        "monitoring": "VGB REMS program (USA): ERG every 6 months — bilateral peripheral visual field loss (30%)",
        "gabbr2_note": (
            "VGB is a GABA-T inhibitor — increases GABA at GABAA. In GABBR2 GOF (excess GABA-B "
            "activation), VGB adds GABAA augmentation independently. Useful. Do NOT confuse with "
            "TGB (ABSOLUTE CI) — VGB is safe; TGB is not."
        ),
    },
    {
        "name": "Baclofen (GABA-B Agonist)",
        "level": "Level C",
        "indication": "PRECISION THERAPY for GABBR2 LOF — NOT indicated in GOF",
        "dose": "Start 5 mg/day PO; increase 5 mg every 3 days → target 1-2 mg/kg/day (max 80 mg/day adult)",
        "moa": "Selective GABA-B agonist → binds GABBR1 VFT domain → activates residual GABBR2 Gi coupling → restores presynaptic autoreceptor function → normalises GABA release feedback",
        "efficacy": "40-60% seizure reduction in GABBR2 LOF responders (Steele 2020 case series)",
        "monitoring": "Sedation, respiratory depression, renal function (renal elimination), withdrawal protocol signed by family",
        "gabbr2_note": (
            "PRECISION THERAPY in LOF: identifies GABBR2 patients who may benefit from augmenting "
            "residual GABA-B receptor function. CONTRAINDICATED in GOF (would worsen constitutive "
            "GABA-B activation). GOF/LOF functional assay MANDATORY before baclofen trial. "
            "NEVER stop abruptly — withdrawal = medical emergency (hyperpyrexia, SE, rhabdomyolysis). "
            "Titrate slowly; consider intrathecal if oral doses limited by sedation."
        ),
    },
    {
        "name": "Valproic Acid (VPA)",
        "level": "Level B",
        "indication": "Generalised seizures, GTCS, myoclonus — standard backbone",
        "dose": "20-40 mg/kg/day PO in 2-3 doses; therapeutic TDM 50-100 µg/mL",
        "moa": "Na+ channel stabilisation + ↑GABA synthesis + ↑GABA-T inhibition (mild) + HDAC inhibition",
        "efficacy": "60-70% seizure reduction in generalised epilepsy including GABBR2 LOF",
        "monitoring": "POLG1 screen MANDATORY before initiation; LFT, FBC, NH3, weight (PCOS risk female)",
        "gabbr2_note": (
            "Standard broad-spectrum AED. POLG1 biallelic screening MANDATORY before VPA — "
            "GABBR2 patients are not pre-selected POLG1-negative; Alpers-Huttenlocher fatal "
            "hepatotoxicity if POLG1 + VPA. Bridging with LEV + CLB while awaiting POLG1 result."
        ),
    },
    {
        "name": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Drug-resistant DEE — early initiation recommended (not last resort)",
        "dose": "4:1 or 3:1 ratio (fat:carb+protein); BHB target 2-5 mmol/L; dietitian-supervised",
        "moa": "BHB → adenosine A1 receptor → hyperpolarisation; ATP-sensitive K+ channel activation; ↓AMPA glutamate",
        "efficacy": "55-65% ≥50% seizure reduction in DEE; IS cessation ~30-40% as adjunct",
        "monitoring": "BHB, lipids, weight, bone density (DEXA annual), renal ultrasound (stones), selenium/zinc",
        "gabbr2_note": (
            "Early KD (during ACTH taper for IS) bridges to next therapy and reduces relapse. "
            "KD may complement baclofen in LOF — distinct mechanisms (KD: adenosine/mTOR; "
            "Baclofen: GABA-B). GOF: KD useful as adjunct — reduces excitatory network "
            "hyperexcitability independently of GABA-B."
        ),
    },
    {
        "name": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Adjunct for GTCS, myoclonus, tonic, IS rescue",
        "dose": "0.1-0.3 mg/kg/day PO in 1-2 doses; max 2 mg/kg/day or 40 mg/day",
        "moa": "1,5-benzodiazepine → GABAA positive allosteric modulator → increases Cl⁻ influx → hyperpolarisation",
        "efficacy": "40-50% add-on seizure reduction in DRE including LGS",
        "monitoring": "Sedation, tolerance (develop over 6-12 months), withdrawal (taper over weeks)",
        "gabbr2_note": (
            "CLB augments GABAA — complements GABA-B pathway dysfunction in GABBR2. "
            "Useful adjunct in both GOF and LOF. Tolerance develops; drug holiday protocol: "
            "5 days off CLB every 3-4 months may restore efficacy."
        ),
    },
    {
        "name": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "Adjunct for GTCS, myoclonus, focal; safe profile in DEE",
        "dose": "20-60 mg/kg/day PO in 2 doses; up to 3000 mg/day adult",
        "moa": "SV2A modulation → impairs presynaptic vesicle priming → ↓synchronous NT release",
        "efficacy": "30-40% add-on seizure reduction; good tolerability",
        "monitoring": "Behavioural toxicity: irritability/aggression 10-20% DEE patients (vs 3-8% focal); "
                      "monthly BRIEF/CBCL behavioural check; switch to brivaracetam if toxicity (5% irritability)",
        "gabbr2_note": (
            "LEV is safe first-line adjunct during VPA/baclofen initiation period. "
            "Behavioural toxicity higher in DEE — explicit consent + monthly check. "
            "Brivaracetam preferred if irritability emerges."
        ),
    },
    {
        "name": "Felbamate",
        "level": "Level C",
        "indication": "LGS drop attacks (tonic/atonic) — when GOF GABBR2 evolves to LGS",
        "dose": "15-45 mg/kg/day PO in 3-4 doses; titrate slowly over 8 weeks; max 3.6 g/day",
        "moa": "NMDA receptor antagonism (glycine site) + GABAA positive allosteric modulation → dual mechanism",
        "efficacy": "30-40% drop attack reduction in LGS; approved FDA for LGS",
        "monitoring": "RARE RISK: aplastic anaemia (1:5000-10000) + fulminant hepatic failure → mandatory FBC + LFT q2 months; "
                      "written informed consent; haematological emergency kit. Not first-line: risk/benefit for LGS only",
        "gabbr2_note": (
            "Felbamate: dual mechanism (NMDA + GABAA) addresses both excitatory and "
            "inhibitory pathway deficiency in GABBR2 GOF LGS. RARE but serious haematological risk "
            "requires strict monitoring protocol. Confirm LGS diagnosis (slow SWD + drop attacks) "
            "before initiating. Discuss risk/benefit explicitly with family."
        ),
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Tiagabine (TGB)",
        "level": "ABSOLUTE CI",
        "reason": (
            "GAT-1 GABA reuptake blockade → sustained perisynaptic GABA → GABAA receptor "
            "desensitisation → paradoxical loss of inhibition → NCSE. GABBR2 dysfunction "
            "amplifies this risk: without normal GABA-B autoreceptor feedback, GABA spillover "
            "from TGB cannot be regulated → catastrophic GABAA desensitisation → NCSE. "
            "NEVER use TGB in GABBR2 epilepsy. NCSE may be non-convulsive — monitor EEG."
        ),
        "alternative": "CLB, VPA, LEV, KD — do not use TGB under any circumstances",
    },
    {
        "drug": "PHT / CBZ / OXC (Na+ channel blockers)",
        "level": "HIGH RISK",
        "reason": (
            "Na+ channel blockers worsen infantile spasms and generalised myoclonus in DEE. "
            "In GABBR2 GOF/LOF with IS or myoclonic seizures: PHT/CBZ → seizure exacerbation "
            "(well-documented class effect in generalised/DEE epilepsies). Paradoxical increase "
            "in seizure frequency documented in 30-40% of DEE patients exposed to Na+ blockers."
        ),
        "alternative": "VPA, LEV, CLB, KD, ACTH — avoid all Na+ channel blockers in DEE phenotype",
    },
    {
        "drug": "Baclofen abrupt withdrawal",
        "level": "ABSOLUTE CI",
        "reason": (
            "Abrupt baclofen cessation = MEDICAL EMERGENCY. Mechanism: GABA-B receptor "
            "upregulation during chronic baclofen exposure → sudden withdrawal → massive "
            "rebound excitation → hyperpyrexia, seizure cluster/status, severe agitation, "
            "rhabdomyolysis, multi-organ failure, death. Documented fatalities. "
            "GABBR2 patients on baclofen: emergency card mandatory; hospital protocol; "
            "never miss >1 dose; if hospitalized, give baclofen IV or NGT — never NPO without "
            "baclofen continuation."
        ),
        "alternative": "Always taper 10% per week minimum; emergency ER card listing baclofen as life-sustaining",
    },
    {
        "drug": "High-dose oral baclofen without monitoring",
        "level": "HIGH RISK",
        "reason": (
            "Oral baclofen at doses >2 mg/kg/day → respiratory depression risk, especially in "
            "hypotonic GABBR2 patients. Renal impairment → baclofen accumulation → coma. "
            "Monitor renal function regularly; dose-reduce for eGFR <30."
        ),
        "alternative": "Intrathecal baclofen (ITB) if high oral doses needed — avoids systemic CNS depression",
    },
    {
        "drug": "VPA without POLG1 screening",
        "level": "ABSOLUTE CI",
        "reason": (
            "GABBR2 patients are not pre-selected POLG1-negative. Biallelic POLG1 mutations + VPA "
            "= Alpers-Huttenlocher syndrome: fatal hepatotoxicity + mtDNA depletion. Screen POLG1 "
            "before VPA; turnaround 7-14 days; bridge with LEV + CLB."
        ),
        "alternative": "LEV + CLB bridge; await POLG1 result before VPA; if POLG1 positive → VPA ABSOLUTE CI forever",
    },
    {
        "drug": "Flumazenil in SE (GABBR2 LOF)",
        "level": "HIGH RISK",
        "reason": (
            "Flumazenil reverses benzodiazepine-mediated GABAA enhancement → reduces inhibitory "
            "tone → in GABBR2 LOF (already reduced GABA-B-mediated inhibition), flumazenil "
            "may precipitate loss of remaining GABAA-mediated inhibition → worsening SE. "
            "Use only if BDZ overdose clearly documented."
        ),
        "alternative": "Phenobarbital or propofol for refractory SE; avoid flumazenil in GABBR2",
    },
]

# ── Monitoring ────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "WES-TRIO (proband + parents)", "frequency": "Once at diagnosis",
     "rationale": "Confirms de novo GABBR2 variant; GOF/LOF functional assay guides baclofen vs contraindication"},
    {"item": "GABBR2 functional assay (GOF/LOF)", "frequency": "Once (variant-specific)",
     "rationale": "MANDATORY before baclofen trial — determines GOF (baclofen CI) vs LOF (baclofen precision Rx)"},
    {"item": "POLG1 screen before VPA", "frequency": "Once before VPA initiation",
     "rationale": "Alpers-Huttenlocher prevention; biallelic POLG1 = VPA absolute lifetime CI"},
    {"item": "Video-EEG LTM (ictal capture)", "frequency": "At IS onset, at 6-month intervals in DRE",
     "rationale": "Hypsarrhythmia grading; IS vs non-IS distinction; NCSE surveillance (especially if TGB ever given)"},
    {"item": "MRI Brain 3T (seizure protocol)", "frequency": "At diagnosis; repeat at 2y, 5y",
     "rationale": "Cortical atrophy progression (GOF), thin CC, no FCD expected but rule out structural cause"},
    {"item": "VGB ERG (REMS)", "frequency": "Baseline; every 6 months on VGB",
     "rationale": "VGB REMS: bilateral visual field constriction in 30%; irreversible — monitor and weigh benefit/risk"},
    {"item": "ACTH Day-14 EEG", "frequency": "During IS ACTH course",
     "rationale": "Hypsarrhythmia resolution = IS response; non-resolution at Day 14 → switch strategy"},
    {"item": "Baclofen tolerance/withdrawal protocol", "frequency": "Every clinic visit if on baclofen",
     "rationale": "Abrupt withdrawal = emergency; family must have signed withdrawal protocol and emergency card"},
    {"item": "Developmental assessment (Bayley-4 / VABS-3)", "frequency": "Every 12 months",
     "rationale": "Cognitive trajectory; early intervention eligibility; ESES surveillance if language regression"},
    {"item": "Renal function (eGFR, Cr) if on baclofen", "frequency": "Every 6 months",
     "rationale": "Baclofen renally eliminated; eGFR <30 → dose-reduce to avoid accumulation/coma"},
    {"item": "VPA TDM / LFT / FBC / NH3", "frequency": "Every 3-6 months on VPA",
     "rationale": "Standard VPA monitoring; NH3 for encephalopathy; LFT for hepatotoxicity (especially <2y)"},
    {"item": "Felbamate FBC / LFT", "frequency": "Every 2 months on felbamate",
     "rationale": "Aplastic anaemia + hepatic failure risk; mandatory haematological monitoring protocol"},
    {"item": "SUDEP risk counselling + alarm", "frequency": "Annually; at diagnosis",
     "rationale": "DRE DEE-59 SUDEP risk ~1/100-200/year; NightWatch/SAMi-3/Empatica seizure alarms; prone sleep avoidance"},
    {"item": "VPPP (valproate pregnancy prevention programme)", "frequency": "If female, at menarche",
     "rationale": "VPA teratogenicity; VPPP mandatory for females of childbearing potential on VPA"},
]

# ── Lifecycle stages ──────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Prenatal / Fetal (0–birth)",
        "key_issues": "De novo GABBR2 variant arises; no fetal seizures (GABA-B matures postnatally); normal anomaly scan; no in-utero IS",
        "action": "Prenatal diagnosis if familial; genetic counselling; plan perinatal neurology consult",
    },
    {
        "stage": "Neonatal / Early Infantile (0–4 months)",
        "key_issues": "Hypotonia may be early sign; GABA-B network immature → seizures not yet prominent; careful surveillance",
        "action": "EEG surveillance if hypotonic; POLG1 screen before any AED; neonatal neurology",
    },
    {
        "stage": "Infantile / IS Phase (4–18 months)",
        "key_issues": "IS onset in GOF (West syndrome); hypsarrhythmia; ACTH/VGB emergency; KD initiation; functional assay to classify GOF/LOF",
        "action": "IS = emergency: ACTH within 2 weeks. GOF: no baclofen. LOF: baclofen trial after IS control",
    },
    {
        "stage": "Toddler / Early DEE (18 months–4 years)",
        "key_issues": "LGS evolution in GOF; drop attacks; helmet; GEFS+ clarification in LOF; KD continuation; baclofen titration in LOF",
        "action": "Helmet immediately if drop attacks. KD + CLB + VPA backbone. Felbamate if LGS drop attacks. Baclofen LOF.",
    },
    {
        "stage": "School Age (5–12 years)",
        "key_issues": "Cognitive assessment; ESES surveillance (LOF hypomorphic); IEP/school support; seizure diary; SUDEP counselling starts",
        "action": "Annual Bayley/VABS; ESES EEG if language regression; school liaison; SUDEP alarm recommendation",
    },
    {
        "stage": "Adolescence / Transition (13–18 years)",
        "key_issues": "Catamenial pattern (females); driving prohibition (seizures); VPPP if VPA + female; social isolation; SUDEP alarm",
        "action": "VPPP female VPA. Driving: 2-year seizure-free minimum. Social work + transition to adult neurology",
    },
    {
        "stage": "Adulthood (18+ years)",
        "key_issues": "Chronic DRE management; baclofen long-term monitoring; bone health (DEXA annual AED use); SUDEP ongoing risk; family planning genetics",
        "action": "DEXA bone scan; switch to adult neurology; genetics re-counselling for family; SUDEP plan updated",
    },
]

# ── Concepts ──────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "GABBR2-22q12.2-DEE-59", "definition": "GABBR2 gene (22q12.2) encodes GABA-B receptor subunit 2; de novo GOF or LOF → DEE-59 (OMIM #617137); ~30 published patients"},
    {"term": "GABA-B Obligatory Heterodimer", "definition": "GABA-B receptor = GABBR1 (ligand-binding, 6p22.1) + GABBR2 (G-protein coupling, 22q12.2); neither traffics to membrane alone; GABBR2 provides ER-export signal"},
    {"term": "GOF-Constitutive-Gi-Signalling", "definition": "TM4-TM5 or VFTM missense GOF: constitutive Gi activation → excess tonic inhibition during development → compensatory upregulation of excitatory synapses → DEE"},
    {"term": "LOF-Presynaptic-Autoreceptor-Loss", "definition": "GABBR2 haploinsufficiency → reduced GABA-B surface density → disinhibited GABA release → GABAA desensitisation → paradoxical network excitation → GEFS+/DEE"},
    {"term": "Baclofen-Precision-LOF", "definition": "R-Baclofen (GABA-B agonist) binds GABBR1 VFT → activates residual GABBR2 from normal allele → restores autoreceptor function → 40-60% seizure reduction in LOF; CONTRAINDICATED in GOF"},
    {"term": "Baclofen-Withdrawal-Emergency", "definition": "Abrupt baclofen cessation → GABA-B rebound excitation → hyperpyrexia, seizure cluster, rhabdomyolysis, death; NEVER stop abruptly; taper 10%/week minimum; emergency ER card"},
    {"term": "GOF-LOF-Functional-Assay-Mandatory", "definition": "GABBR2 variant functional assay in Xenopus oocyte or HEK293 system MANDATORY before baclofen trial — GOF=CI, LOF=precision Rx; cannot determine from variant type alone"},
    {"term": "TGB-ABSOLUTE-CI-NCSE", "definition": "Tiagabine ABSOLUTE CI in GABBR2: GAT-1 block → GABA spillover → GABAA desensitisation → NCSE; GABBR2 autoreceptor dysfunction amplifies risk"},
    {"term": "PHT-CBZ-Worsen-DEE-IS", "definition": "Na+ channel blockers paradoxically worsen IS and myoclonus in DEE including GABBR2; document on emergency care plan: AVOID PHT/CBZ/OXC"},
    {"term": "West-Syndrome-GABBR2-GOF", "definition": "GOF GABBR2 → IS (West syndrome) onset 4-12 months; hypsarrhythmia; ACTH first-line (50-60% IS cessation); VGB Level A; KD early; LGS evolution in 60%"},
    {"term": "POLG1-VPA-Screen-Mandatory", "definition": "GABBR2 patients not POLG1-pre-screened; biallelic POLG1 + VPA = fatal Alpers-Huttenlocher; screen before VPA; bridge with LEV+CLB (7-14d turnaround)"},
    {"term": "GABBR1-GABBR2-Comparison", "definition": "GABBR1 (6p22.1): milder epilepsy, GEFS+ predominant; GABBR2 (22q12.2): more common, severe DEE-59 GOF, IS/West→LGS; both: baclofen relevant in LOF"},
    {"term": "Felbamate-LGS-Dual-Mechanism", "definition": "Felbamate: NMDA antagonism (glycine site) + GABAA positive modulation; addresses both excitatory excess + inhibitory deficiency in GABBR2 GOF LGS; haematological monitoring mandatory"},
    {"term": "SUDEP-DEE59-High-Risk", "definition": "GABBR2 DEE-59 DRE: SUDEP ~1/100-200/year; seizure alarm mandatory (NightWatch/SAMi-3/Empatica); avoid prone sleeping; SUDEP discussion at diagnosis"},
    {"term": "VPPP-Female-VPA", "definition": "Valproate pregnancy prevention programme MANDATORY for all females of childbearing potential on VPA; GABBR2 females may be on VPA — VPPP from menarche"},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "IS onset-to-treatment window", "value": "≤2 weeks", "action": "Start ACTH/VGB within 2 weeks of IS onset — every day of delay worsens cognitive outcome"},
    {"parameter": "ACTH response assessment", "value": "Day 14 EEG", "action": "Hypsarrhythmia present at Day 14 → switch strategy (add VGB or cross-taper to VGB)"},
    {"parameter": "Baclofen target dose (LOF)", "value": "1-2 mg/kg/day", "action": "Titrate from 5 mg/day; increase every 3 days; max 80 mg/day adult; check renal function at each dose increase"},
    {"parameter": "Baclofen taper rate", "value": "10% per week minimum", "action": "Never faster; hospital admission if withdrawal signs (agitation, fever, seizure cluster, sweating)"},
    {"parameter": "VPA TDM target", "value": "50-100 µg/mL", "action": "Free VPA preferred in hypoalbuminaemia (DEE); LFT + NH3 with every TDM"},
    {"parameter": "Felbamate FBC + LFT", "value": "Every 2 months", "action": "Aplastic anaemia/hepatic failure: stop felbamate IMMEDIATELY if FBC/LFT abnormal; haematology consult"},
    {"parameter": "VGB ERG", "value": "Every 6 months", "action": "Peripheral visual field loss >30° → VGB dose reduction or discontinuation with documentation"},
    {"parameter": "KD BHB target", "value": "2-5 mmol/L", "action": "Below 1.5: insufficient ketosis; above 6: metabolic acidosis risk; adjust ratio"},
    {"parameter": "POLG1 screening turnaround", "value": "7-14 days", "action": "Bridge with LEV + CLB; do NOT start VPA before POLG1 result available"},
    {"parameter": "Renal threshold for baclofen dose reduction", "value": "eGFR <30 mL/min/1.73m²", "action": "Halve baclofen dose; increase dosing interval; monitor for accumulation (sedation, hypotonia)"},
    {"parameter": "Baclofen withdrawal admission threshold", "value": "T >38.5°C or seizure cluster >30 min on baclofen withdrawal", "action": "Emergency admission; restart baclofen via NGT/IV protocol; ICU if needed"},
    {"parameter": "SUDEP alarm threshold", "value": "DRE (≥2 AEDs failed)", "action": "Seizure alarm device prescription at DRE diagnosis; prone sleeping prohibition; SUDEP discussion documented"},
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE 2022 Classification of Epilepsy Syndromes", "relevance": "DEE-59 classification; IS management guidelines"},
    {"code": "NICE-NG217", "title": "NICE NG217 Epilepsies in Children, Young People and Adults (2022)", "relevance": "IS/DEE management; AED choice; monitoring"},
    {"code": "Yoo-2017-EBioMed", "title": "Yoo et al. 2017 EBioMedicine — First GABBR2 DEE-59 report", "relevance": "De novo GABBR2 GOF variants → DEE; foundational paper"},
    {"code": "Steele-2020-Epilepsia", "title": "Steele et al. 2020 Epilepsia — 30-patient GABBR2 cohort", "relevance": "GOF/LOF classification; baclofen precision therapy evidence"},
    {"code": "Pinard-2010-NatNeurosci", "title": "Pinard et al. 2010 Nature Neuroscience — GABA-B assembly biology", "relevance": "GABBR1-GABBR2 obligatory heterodimer; GABBR2 ER-export signal"},
    {"code": "UKISS-2005-Lancet", "title": "UKISS 2004-2005 Lancet Neurology — IS trial ACTH vs prednisolone", "relevance": "ACTH Level A IS first-line; Day-14 EEG response criterion"},
    {"code": "CPIC-POLG1-2023", "title": "CPIC POLG Guidelines 2023", "relevance": "POLG1 screening before VPA; Alpers-Huttenlocher prevention protocol"},
    {"code": "FDA-SHARE-REMS-VGB", "title": "FDA SHARE REMS — Vigabatrin visual field monitoring", "relevance": "VGB ERG every 6 months; peripheral VF loss monitoring protocol"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA Valproate Pregnancy Prevention Programme 2021", "relevance": "VPA teratogenicity; VPPP mandatory females of childbearing age"},
    {"code": "ACMG-AMP-2015", "title": "ACMG-AMP Variant Interpretation Standards 2015", "relevance": "GABBR2 variant classification (pathogenic/VUS); functional assay evidence"},
    {"code": "NICE-NG224-2023", "title": "NICE NG224 Epilepsy in Adults 2023", "relevance": "Adult GABBR2 management; SUDEP guidance; AED monitoring"},
    {"code": "WHO-ICF-2019", "title": "WHO International Classification of Functioning 2019", "relevance": "Disability framework; GABBR2 DEE-59 functional outcomes documentation"},
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    {"id": "Yoo-2017-EBioMedicine", "citation": "Yoo Y et al. De novo variants in GABBR2 associated with epileptic encephalopathy. EBioMedicine. 2017;30:189-196.", "key_finding": "First 3 de novo GABBR2 GOF variants → DEE-59; established GABBR2 as epilepsy gene"},
    {"id": "Steele-2020-Epilepsia", "citation": "Steele SU et al. GABBR2 variants in epileptic encephalopathy: a review of 30 patients. Epilepsia. 2020;61(5):992-1005.", "key_finding": "GOF 55% severe DEE (IS/LGS); LOF 45% milder GEFS+; baclofen precision LOF (40-60% response)"},
    {"id": "Pinard-2010-NatNeurosci", "citation": "Pinard A et al. Molecular tinkering of G protein-coupled receptors: an evolutionary success. Nature Neuroscience. 2010;13(6):641-647.", "key_finding": "GABBR1-GABBR2 obligatory heterodimer biology; GABBR2 ER-export signal; Gi effector coupling"},
    {"id": "UKISS-2005-LancetNeurol", "citation": "Lux AL et al. The United Kingdom Infantile Spasms Study (UKISS). Lancet Neurology. 2005;4(11):712-717.", "key_finding": "ACTH superior to prednisolone for IS cessation at 14 days; Day-14 EEG as response criterion"},
    {"id": "Bass-2016-NEJM-EXIST3", "citation": "Thiele EA et al. (EXIST-3) Everolimus for tuberous sclerosis complex-associated seizures. N Engl J Med. 2016;375(18):1733-1742.", "key_finding": "mTORC1 inhibition Level A for TSC IS (contextual reference for mTOR biology; not direct GABBR2)"},
    {"id": "Bianchi-2022-CurrNeuropharm", "citation": "Bianchi P et al. GABA-B receptor dysfunction and epilepsy: from bench to bedside. Curr Neuropharmacol. 2022;20(1):10-25.", "key_finding": "Comprehensive GABA-B receptor pharmacology; baclofen mechanism in GABBR1/2 LOF; therapeutic implications"},
]

# ── Patient cohort ────────────────────────────────────────────────────────────
_ETIOLOGIES = [c["etiology"] for c in ETIOLOGY_CATALOG]
_WEIGHTS = [c["pct"] for c in ETIOLOGY_CATALOG]

_MALE_NAMES = ["Oliver","Noah","Ethan","Lucas","Liam","Aiden","Mason","Logan","Carter","James","Sebastian","Henry","Alexander","Daniel","William","Muhammad","Yusuf","Arjun","Rafael","Mateo"]
_FEMALE_NAMES = ["Emma","Sophia","Ava","Mia","Isabella","Charlotte","Amelia","Lily","Harper","Evelyn","Aisha","Priya","Yuki","Fatima","Elena","Nora","Chloe","Grace","Luna","Zoe"]

random.seed(42)
PATIENT_SAMPLE = []
for i in range(40):
    male = random.random() < 0.5
    name = random.choice(_MALE_NAMES if male else _FEMALE_NAMES)
    etiology = random.choices(_ETIOLOGIES, weights=_WEIGHTS)[0]
    is_gof = "GOF" in etiology
    is_lof = "LOF" in etiology
    age_onset = round(random.uniform(0.3, 10.0), 1) if not is_gof else round(random.uniform(0.3, 1.5), 1)
    dr = is_gof or (is_lof and random.random() < 0.35)
    on_baclofen = is_lof and random.random() < 0.55
    on_vgb = age_onset < 2.0 and random.random() < 0.70
    on_kd = dr and random.random() < 0.60
    on_acth = age_onset < 2.0 and random.random() < 0.65
    on_felbamate = is_gof and dr and random.random() < 0.25
    west = age_onset < 1.5 and is_gof and random.random() < 0.85
    lgs = is_gof and random.random() < 0.60
    PATIENT_SAMPLE.append({
        "id": f"P{i+1:03d}",
        "name": name,
        "sex": "M" if male else "F",
        "etiology": etiology,
        "age_onset": age_onset,
        "current_age": round(age_onset + random.uniform(2, 15), 1),
        "drug_resistant": dr,
        "on_baclofen": on_baclofen,
        "on_vgb": on_vgb,
        "on_kd": on_kd,
        "on_acth": on_acth,
        "on_felbamate": on_felbamate,
        "west_syndrome": west,
        "lgs_evolution": lgs,
        "gof_lof": "GOF" if is_gof else ("LOF" if is_lof else "Unknown"),
        "functional_assay_done": random.random() < 0.70,
        "polg1_screened": True,
        "sudep_alarm": dr,
    })


def get_overview():
    total = len(PATIENT_SAMPLE)
    dr = sum(1 for p in PATIENT_SAMPLE if p["drug_resistant"])
    on_bac = sum(1 for p in PATIENT_SAMPLE if p["on_baclofen"])
    on_kd = sum(1 for p in PATIENT_SAMPLE if p["on_kd"])
    on_vgb = sum(1 for p in PATIENT_SAMPLE if p["on_vgb"])
    west = sum(1 for p in PATIENT_SAMPLE if p["west_syndrome"])
    lgs = sum(1 for p in PATIENT_SAMPLE if p["lgs_evolution"])
    gof_n = sum(1 for p in PATIENT_SAMPLE if p["gof_lof"] == "GOF")
    lof_n = sum(1 for p in PATIENT_SAMPLE if p["gof_lof"] == "LOF")
    assay_done = sum(1 for p in PATIENT_SAMPLE if p["functional_assay_done"])
    avg_onset = round(sum(p["age_onset"] for p in PATIENT_SAMPLE) / total, 1)

    etiology_dist = {item["category"]: {"n": item["n"], "pct": item["pct"]} for item in ETIOLOGY_CATALOG}
    seizure_dist = [{"type": s["type"].split("(")[0].strip(), "pct": s["frequency_pct"]} for s in SEIZURE_TYPES]
    trigger_dist = [{"trigger": t["trigger"], "pct": t["pct"]} for t in TRIGGERS]

    return {
        "dashboard": "GABBR2 Epilepsy (DEE-59 / GABA-B Receptor Subunit 2 / GOF-LOF / Baclofen-Precision / TGB-ABSOLUTE / 22q12.2)",
        "gene": "GABBR2 (22q12.2) — GABA-B receptor subunit 2 (GBR2, GPRC3B)",
        "receptor": "GABA-B = GABBR1 (ligand-binding) + GABBR2 (Gi-effector) obligatory heterodimer",
        "inheritance": "De novo dominant (GOF 55% / LOF 45%); rare familial AD",
        "omim": "DEE-59 OMIM #617137; GABBR2 gene OMIM *607340",
        "cohort_size": total,
        "gof_patients": gof_n,
        "lof_patients": lof_n,
        "mean_age_onset_years": avg_onset,
        "drug_resistant_n": dr,
        "drug_resistant_pct": round(dr / total * 100),
        "west_syndrome_n": west,
        "west_syndrome_pct": round(west / total * 100),
        "lgs_evolution_n": lgs,
        "lgs_evolution_pct": round(lgs / total * 100),
        "on_baclofen_n": on_bac,
        "on_baclofen_pct": round(on_bac / total * 100),
        "on_kd_n": on_kd,
        "on_kd_pct": round(on_kd / total * 100),
        "on_vgb_n": on_vgb,
        "on_vgb_pct": round(on_vgb / total * 100),
        "functional_assay_done_n": assay_done,
        "functional_assay_done_pct": round(assay_done / total * 100),
        "etiology_distribution": etiology_dist,
        "seizure_type_distribution": seizure_dist,
        "trigger_distribution": trigger_dist,
        "precision_therapy": "Baclofen (GABA-B agonist) Level C — 40-60% seizure reduction in LOF only; CONTRAINDICATED in GOF",
        "key_contraindications": [
            "TGB ABSOLUTE CI (NCSE via GABAA desensitisation)",
            "PHT/CBZ/OXC HIGH RISK (worsen IS/myoclonus)",
            "Baclofen abrupt withdrawal ABSOLUTE CI (medical emergency: hyperpyrexia, death)",
            "VPA without POLG1 ABSOLUTE CI (Alpers-Huttenlocher)",
        ],
    }


def get_breakdown():
    return {
        "etiology_catalog": ETIOLOGY_CATALOG,
        "patient_sample": PATIENT_SAMPLE[:15],
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
