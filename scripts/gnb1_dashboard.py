"""
GNB1 Epilepsy — Developmental and Epileptic Encephalopathy / Infantile Spasms / West Syndrome
==============================================================================================
40-patient cohort · GNB1 (1p36.33) · Gβ1 G-protein β1 Subunit · AD de novo · ACTH / Vigabatrin

GNB1 BIOLOGY:
GNB1 (1p36.33) encodes the Gβ1 subunit (G-protein beta-1) — the most widely expressed
member of the β-subunit family of heterotrimeric G proteins (Gβ1–Gβ5).

KEY POINTS:
  1. HETEROTRIMERIC G-PROTEIN CYCLE:
     Inactive: Gα·GDP tightly bound to Gβγ dimer (GNB1 + Gγ) at plasma membrane.
     GPCR activation → receptor conformational change → Gα exchanges GDP for GTP →
     Gα-GTP + free Gβγ dimer dissociate to activate separate effectors.
     GNB1 (Gβ1) forms obligate dimers with Gγ subunits (Gγ2, Gγ7, Gγ12 most common);
     the Gβγ dimer is functionally inseparable and essential for signal transduction.

  2. Gβ1/Gβγ EFFECTOR FUNCTIONS IN NEURONS:
     (a) GIRK channels (Kir3.1/Kir3.2): Gβγ directly gates GIRK → K+ efflux →
         hyperpolarization → inhibition of neuronal firing. Key for synaptic inhibition
         downstream of GABA-B, dopamine D2, adenosine A1, opioid μ receptors.
     (b) Voltage-gated Ca²⁺ channels: Gβγ tonic inhibition of N-type (Cav2.2) and
         P/Q-type (Cav2.1) Ca²⁺ channels → reduces vesicle fusion → reduces
         neurotransmitter release (primarily at excitatory glutamatergic terminals
         with high Cav2.1/Cav2.2 density).
     (c) AC (adenylyl cyclase): Gβγ inhibits AC2/AC5/AC6 isoforms → reduces cAMP →
         reduces PKA activation → altered AMPA receptor phosphorylation/trafficking.
     (d) PLC-β: Gβγ activates PLC-β2/PLC-β3 → IP3 + DAG → ER Ca²⁺ release + PKC.

  3. GNB1 PATHOGENIC VARIANTS — MECHANISM OF DISEASE:
     De novo missense variants in GNB1 alter the β-propeller fold of Gβ1, disrupting:
     • Proper Gβγ dimer assembly or stability (severe hypomorphs → LOF)
     • Gα binding interface → uncoupling from Gα → constitutive Gβγ signaling (GOF)
     • Effector binding surface (blade 1/2/6 of β-propeller) → altered effector selectivity
     Net result: dysregulated Gβγ effector coupling → reduced GIRK activation →
     reduced neuronal hyperpolarization → cortical hyperexcitability → DEE / infantile spasms.
     Reduced Cav inhibition → more glutamate release at excitatory synapses → net
     excitatory/inhibitory imbalance aggravated.

  4. WHY INFANTILE SPASMS / WEST SYNDROME:
     The immature brain (0–12 months) is uniquely vulnerable to Gβγ dysfunction:
     (a) GIRK2 expression peaks in postnatal infancy — loss of Gβγ→GIRK2 signaling is
         maximally disruptive in this window → disinhibition of thalamic relay neurons
         → hypsarrhythmia (chaotic high-voltage slow waves + multifocal spikes).
     (b) Subcortical (brainstem/hypothalamic) GPCR signalling via GNB1 regulates CRH
         (corticotropin-releasing hormone) and ACTH secretion. GNB1 GOF variants may
         drive CRH hypersecretion — explaining why ACTH therapy (suppressing CRH→ACTH
         axis) has a specific rationale in GNB1 encephalopathy.
     (c) Excitatory/inhibitory ratio is physiologically higher in neonates (GABA is
         depolarising early in development via high intracellular Cl⁻, NKCC1 > KCC2);
         any additional excitatory drive tips into spasms.

  5. GENOTYPE–PHENOTYPE (Hemati 2018, Rodan 2021):
     Blade 1 variants (p.I80N, p.I80T, p.G77R): severe DEE — most disruptive to
       Gβ1 fold → early infantile spasms; worst neurodevelopmental prognosis.
     Blade 2 variants (p.R96H, p.K89T): moderate DEE — infantile spasms; variable
       ACTH response; ~50% retain limited speech.
     Blade 6 variants (p.G117R, p.R314C): variable DEE/GGE — occasionally milder;
       some focal seizures; responsive to standard AEDs.
     All variants: de novo (>98%); pLI ~0.97.

  6. POPULATION GENETICS:
     GNB1 locus: 1p36.33 (same chromosome arm as GABRD, RELN).
     pLI ~0.97 (high intolerance). OMIM DEE42 → Actually GNB1 encephalopathy is
     OMIM #616139 (GNB1 Intellectual Disability, Autosomal Dominant = MENTAL
     RETARDATION, AUTOSOMAL DOMINANT 42; MRD42; which includes DEE). Discovery:
     Hemati et al. 2018 Am J Hum Genet 103(2):243-258.
     Prevalence: ~150–200 confirmed cases worldwide (as of 2024); underdiagnosed.

CLINICAL DIAGNOSIS:
  - Infantile spasms onset 3–12 months (flex/ext spasms in series, often on waking)
  - EEG: hypsarrhythmia (modified or classic) → evolves to multifocal spikes, slow background
  - MRI: often normal at onset; may show non-specific atrophy or delayed myelination later
  - Hypotonia, absent or limited developmental milestones
  - Movement abnormalities: tremor, dystonia, movement stereotypies (choreoathetosis in subset)
  - Autism spectrum features in older patients
  - No dysmorphic features (unlike chromosomal syndromes with IS)

INHERITANCE AND GENETICS:
  AUTOSOMAL DOMINANT. De novo: >98%. Familial: <2% (very rare somatic mosaicism in parents).
  Locus: 1p36.33. Gene: GNB1 (10 exons; 340 aa protein; β-propeller WD-repeat structure).
  pLI ~0.97. Key de novo variants: p.I80N · p.I80T · p.G77R · p.R96H · p.K89T ·
  p.G117R · p.R314C · p.A52T · p.P89L.

KEY REFERENCES:
  Hemati et al. 2018 Am J Hum Genet 103(2):243-258 — GNB1 discovery in DEE cohort of 35 patients
  Rodan et al. 2021 J Med Genet 58(7):454-461 — expanded cohort; genotype-phenotype correlations
  Aoi et al. 2019 Mol Genet Genomic Med 7(12):e1014 — Japanese cohort; ACTH response
  Bhatt et al. 2023 Epilepsia — gene-epilepsy reference standard
  Bhattacharya & Bhatt 2020 Front Neurol — ACTH in genetic epilepsies
  ILAE Commission on Classification and Terminology 2022
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GNB1-blade1-severe-DEE",
        "pct": 38,
        "etiology": "GNB1 blade-1 missense — severe DEE / classic West syndrome (blade-1 of β-propeller)",
        "mechanism": (
            "Missense variants in WD-repeat blade 1 (residues 50–90) of the Gβ1 β-propeller fold. "
            "This blade forms key contacts with Gα and with blade 7 (closure blade). "
            "p.I80N and p.I80T disrupt the hydrophobic core → partial protein misfolding → "
            "reduced Gβ1 stability → impaired Gβγ dimer formation → loss of GIRK2 activation "
            "and loss of Ca²⁺ channel inhibition. Results in severe early-onset DEE: "
            "infantile spasms by age 3–6 months; hypsarrhythmia; minimal ACTH response in ~30%."
        ),
        "typical_variants": "p.I80N (most recurrent) · p.I80T · p.G77R · p.A52T",
        "onset_age_months": 4,
        "outcome": "Poor — refractory spasms; severe ID; non-verbal in >80%; KD considered after 2 AED failures",
    },
    {
        "category": "GNB1-blade2-moderate-DEE",
        "pct": 30,
        "etiology": "GNB1 blade-2 missense — moderate DEE / West syndrome with ACTH response",
        "mechanism": (
            "Variants in blade 2 (residues 91–130): p.R96H alters a residue at the Gβγ–effector "
            "binding surface; p.K89T disrupts a charged interface. Less severe structural disruption "
            "than blade-1 variants. Gβ1 protein is partially functional → residual GIRK2 activation. "
            "Infantile spasms onset 4–9 months; modified hypsarrhythmia. ACTH response: ~60%. "
            "Approximately 40% achieve spasm cessation with first-line ACTH/prednisolone."
        ),
        "typical_variants": "p.R96H (second most recurrent) · p.K89T · p.P89L · p.R116C",
        "onset_age_months": 6,
        "outcome": "Moderate — spasms often respond to ACTH; transition to Lennox-Gastaut spectrum in 40%; limited speech in 50%",
    },
    {
        "category": "GNB1-blade6-variable",
        "pct": 17,
        "etiology": "GNB1 blade-6 missense — variable DEE / focal epilepsy / GGE overlap",
        "mechanism": (
            "Blade 6 (residues 300–340) forms the C-terminal closure region of the β-propeller; "
            "p.G117R (unusual in spanning blade junction) and p.R314C affect the outermost blade. "
            "Milder structural disruption → partial Gβγ function retained. Phenotype variable: "
            "some patients have focal seizures without spasms; some have absence-like episodes; "
            "autism spectrum features prominent. May not present as classic West syndrome."
        ),
        "typical_variants": "p.G117R · p.R314C · p.T328I · p.N87S (blade 2/6 junction)",
        "onset_age_months": 8,
        "outcome": "Variable — some partial remission of seizures; focal epilepsy responds to LCM/LEV; cognition: moderate ID",
    },
    {
        "category": "GNB1-truncating-DEE",
        "pct": 10,
        "etiology": "GNB1 truncating / frameshift — haploinsufficiency DEE",
        "mechanism": (
            "Nonsense or frameshift variants → nonsense-mediated mRNA decay → 50% reduction in "
            "Gβ1 protein. Pure haploinsufficiency. Complete Gβγ dimer reduction → uniform reduction "
            "in GIRK2 activation and Ca²⁺ channel inhibition across all GPCR pathways. Phenotype "
            "overlaps with missense: infantile spasms or early-onset focal DEE. Truncating variants "
            "are less common (~10%) since many are lost to NMD and do not reach clinical ascertainment "
            "unless exome sequencing is performed. Responds to ACTH ~ similarly to blade-2 variants."
        ),
        "typical_variants": "p.Q79* · c.507+1G>T (splice) · c.236delA (frameshift) · p.R45*",
        "onset_age_months": 5,
        "outcome": "Moderate-poor — similar to blade-2; spasm control in ~55% with ACTH; moderate ID",
    },
    {
        "category": "GNB1-negative-phenocopy",
        "pct": 5,
        "etiology": "GNB1-negative DEE phenocopy — clinically GNB1-like but no pathogenic variant",
        "mechanism": (
            "West syndrome / infantile spasms with GNB1-like features (de novo, no dysmorphic, "
            "MRI normal) but GNB1 sequencing negative. Consider: CDKL5 (X-linked, especially females), "
            "ARX (males), STXBP1 (most common single-gene DEE), TSC1/TSC2 (subtle cortical tubers), "
            "KCNQ2 (neonatal onset preceding spasms), KAT6A, DYRK1A. Functional G-protein studies "
            "for VUS classification. ~5% in referral cohorts without identified pathogenic variant."
        ),
        "typical_variants": "VUS on GNB1 · or negative panel · consider GNB2/GNG variants (rare)",
        "onset_age_months": 6,
        "outcome": "Variable — depends on underlying aetiology; comprehensive panel recommended",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Epileptic Spasms (Infantile Spasms / West Syndrome)",
        "pct": 95,
        "description": (
            "Epileptic spasms — cardinal seizure type of GNB1 encephalopathy. "
            "Sudden, brief (1–2 sec) symmetric or asymmetric flexion/extension of neck, trunk, "
            "and limbs. Occur in clusters (5–100 spasms per cluster) on waking from sleep or naps. "
            "Associated with hypsarrhythmia on EEG (high-amplitude chaotic multifocal slow waves "
            "and spikes). Onset: 3–12 months (peak 4–8 months). Electrodecremental response on EEG "
            "is the ictal correlate of each spasm."
        ),
        "eeg": "Hypsarrhythmia (classic or modified): continuous high-amplitude (>300 µV) chaotic delta + multifocal spikes. Ictal: electrodecremental (EEG amplitude suddenly flattens during spasm). Post-ictal: recurrence of hypsarrhythmia.",
        "semiology": "Sudden flexion of neck + trunk + arms (Salaam attack); or extension; or mixed. Brief (1–2 sec). Clusters of 5–100 on waking. Cry or discomfort during clusters. No post-ictal confusion (unlike focal seizures).",
        "clinical_tips": (
            "Video-EEG ESSENTIAL for diagnosis — home video misses 40% of spasms. "
            "Start ACTH or prednisolone within days of confirmed diagnosis (delay worsens outcome). "
            "GNB1 encephalopathy: ACTH response rate ~55%; vigabatrin response ~50%. "
            "In TSC context vigabatrin is first-line (UKISS trial) — but GNB1 is NOT TSC, "
            "so ACTH or prednisolone is preferred first-line per UKISS/ICNA guidelines."
        ),
    },
    {
        "type": "Focal Seizures (Temporal / Frontal Onset)",
        "pct": 62,
        "description": (
            "Focal seizures — emerge as infantile spasms evolve or remit. Most common in "
            "blade-6 GNB1 variants. Temporal onset: aura (autonomic), oro-alimentary automatisms, "
            "focal clonic. Frontal onset: tonic posturing, hypermotor, nocturnal. "
            "In post-spasm evolution (Lennox-Gastaut transition), focal seizures become more prominent "
            "and respond to LCM or LTG. EEG: focal spike-wave, may be multifocal."
        ),
        "eeg": "Focal temporal or frontal spikes/sharp waves. Ictal: rhythmic theta build-up. Multifocal background abnormalities common.",
        "semiology": "Temporal: staring, lip-smacking, automatisms, déjà vu in older children. Frontal: tonic/hypermotor, nocturnal, short duration.",
        "clinical_tips": (
            "Focal seizures in GNB1 often emerge AFTER spasm cessation (residual epilepsy). "
            "LCM (lacosamide) effective for focal GNB1 seizures: slow Na+ inactivation MOA, "
            "no GGE aggravation risk (unlike CBZ in GGE). "
            "VPA appropriate if Lennox-Gastaut evolution — POLG1 mandatory before starting."
        ),
    },
    {
        "type": "Generalized Tonic-Clonic Seizures (GTCS)",
        "pct": 45,
        "description": (
            "GTCS — typically emerge in later childhood/adolescence as GNB1 epilepsy evolves "
            "from West syndrome. May indicate Lennox-Gastaut spectrum transition or independent "
            "GGE-like component. Bilateral symmetrical tonic phase → clonic phase. "
            "Risk factor for SUDEP in refractory GNB1 DEE. "
            "Nocturnal GTCS particularly concerning — SUDEP monitoring mandatory."
        ),
        "eeg": "Generalized polyspike-wave. Background: slow and disorganised. Evolves from slow SWD as age increases.",
        "semiology": "Tonic (10–20 sec) → clonic. Tongue bite risk. Incontinence. Post-ictal confusion. Longer than spasms.",
        "clinical_tips": (
            "GTCS in GNB1 signals LGS-transition or evolving epilepsy. "
            "VPA + CLB or VPA + rufinamide are LGS-appropriate regimens. "
            "Rescue midazolam (buccal) 0.3 mg/kg for GTCS >2 min. "
            "KD strongly considered at this stage if 2+ AEDs have failed."
        ),
    },
    {
        "type": "Tonic Seizures (Nocturnal / Sleep-Related)",
        "pct": 38,
        "description": (
            "Tonic seizures — nocturnal sustained muscle contraction, 5–20 seconds. "
            "Classic feature of Lennox-Gastaut spectrum (LGS), which GNB1 encephalopathy "
            "may evolve into by 3–5 years. EEG: fast (>10 Hz) paroxysmal recruiting rhythm. "
            "Risk of injury (fall); padded bed rails required. May be asymmetric."
        ),
        "eeg": "Paroxysmal fast activity (PFA): high-frequency (10–20 Hz) low-amplitude discharge during tonic phase. Diffuse slow SWD on background.",
        "semiology": "Sudden muscle stiffening, arms ± legs, brief cry, autonomic features (flushing, apnoea). No post-ictal confusional state.",
        "clinical_tips": (
            "Tonic seizures + slow SWD + cognitive impairment = LGS criteria met. "
            "Rufinamide is specifically approved for LGS tonic/atonic seizures. "
            "Avoid TGB (NCSE risk in GGE-like/LGS). VPA + rufinamide + KD is standard LGS triad."
        ),
    },
    {
        "type": "Myoclonic Seizures / Myoclonic-Atonic (Doose-like)",
        "pct": 22,
        "description": (
            "Myoclonic seizures — brief bilateral myoclonic jerks (arms > legs). "
            "May be myoclonic-atonic (drop attacks) in some GNB1 patients — sudden "
            "myoclonus followed by loss of tone → fall. EEG: generalised polyspike-wave "
            "2–3 Hz. This pattern suggests overlap with Doose/myoclonic-atonic epilepsy spectrum "
            "in the milder GNB1 variants (blade-6). Worsened by CBZ/OXC."
        ),
        "eeg": "Bilateral polyspike-wave burst time-locked to myoclonic jerk. Generalised 2-3 Hz. May show focal onset in some.",
        "semiology": "Bilateral arm jerk ± head drop. Drop attacks (myoclonic-atonic): sudden fall without warning. Very brief post-ictal state.",
        "clinical_tips": (
            "Myoclonic seizures in GNB1 → AVOID CBZ/OXC (aggravates myoclonus and absence). "
            "VPA is the drug of choice if no POLG1 variants. ESM can be added for pure myoclonic. "
            "KD particularly effective for myoclonic-atonic component (Doose overlap)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Waking from Sleep / Sleep-Wake Transition",
        "pct": 95,
        "notes": (
            "PATHOGNOMONIC trigger for infantile spasms — clusters almost exclusively occur on "
            "waking from sleep (morning, nap transition). NREM-to-REM and NREM-to-wake "
            "transitions promote hypersynchronous discharges in immature thalamo-cortical "
            "circuits already destabilised by GNB1 Gβγ dysfunction. "
            "CLINICAL IMPORTANCE: Spasm clusters during feeding, diaper change 30-60 min "
            "after waking are diagnostic red flags. Video-EEG in morning hours is highest-yield."
        ),
    },
    {
        "trigger": "Fever / Febrile Illness",
        "pct": 75,
        "notes": (
            "Fever (≥38°C) significantly increases spasm frequency and can trigger "
            "febrile-onset status epilepticus (FOSE) in GNB1 DEE. "
            "Mechanism: fever → increased metabolic demand on impaired Gβγ-GIRK circuit → "
            "reduced hyperpolarization capacity → bursting threshold lowered. "
            "MANAGEMENT: Maintain AEDs during illness; acetaminophen early (do NOT wait for "
            "high fever). Emergency letter for A&E with AED protocol."
        ),
    },
    {
        "trigger": "Missed AED Dose / Medication Error",
        "pct": 68,
        "notes": (
            "Missing ACTH, VPA, or LEV doses triggers rebound spasm clusters or focal seizures "
            "within 12-24 hours (ACTH abrupt withdrawal causes adrenal suppression and "
            "rebound cortical hyperexcitability). "
            "ACTH taper is particularly critical — abrupt ACTH discontinuation = medical emergency. "
            "Use ACTH taper protocol: reduce by 10% per week over 6–8 weeks from peak dose. "
            "Families must have written ACTH tapering schedule."
        ),
    },
    {
        "trigger": "Intercurrent Illness (non-febrile) / Metabolic Stress",
        "pct": 58,
        "notes": (
            "GI illness (vomiting, diarrhoea) disrupts oral AED absorption → reduced levels → "
            "seizure breakthrough. Dehydration → electrolyte abnormalities (↑ Na+ with ACTH; "
            "↓ Na+ with OXC/CBZ) → lower seizure threshold. "
            "During any illness: ensure AED delivery (consider IV formulation if vomiting); "
            "check Na+ / glucose with ACTH therapy; IV access early if multiple vomits."
        ),
    },
    {
        "trigger": "Sensory Stimulation (tactile, auditory, light)",
        "pct": 48,
        "notes": (
            "Some GNB1 patients have stimulus-sensitive spasms — tactile (touch, bath), "
            "sudden sound, or visual stimuli can trigger spasm clusters in the newborn/infant period. "
            "Mechanism: sensory stimulation → thalamocortical activation → in hypsarrhythmic "
            "background → triggers synchronised discharge. "
            "MINIMISE stimulation during clusters; dim lights and reduce noise during cluster."
        ),
    },
    {
        "trigger": "Hyperthermia (Hot Bath, Overheating)",
        "pct": 40,
        "notes": (
            "Non-febrile hyperthermia (hot bath, warm car) can trigger spasm clusters — "
            "similar to SCN1A-Dravet thermolability but via Gβγ circuit destabilisation. "
            "Advise: lukewarm baths only; avoid prolonged exposure to heat; car A/C in summer. "
            "Temperature dysregulation is also a concern during ACTH therapy (immunosuppression + "
            "mineralocorticoid effects → fluid retention, blood pressure changes)."
        ),
    },
    {
        "trigger": "ACTH / Corticosteroid Taper (Rebound)",
        "pct": 35,
        "notes": (
            "REBOUND SPASMS after ACTH taper are common in GNB1 (30–40%) — require second "
            "ACTH course or transition to vigabatrin or KD maintenance. "
            "ACTH TAPER RULE: never reduce by >10% per week; if spasms return during taper, "
            "increase back to previous dose and hold for 2 weeks before re-tapering. "
            "Post-ACTH transition: maintain VGB or VPA as bridge. Second-course ACTH allowed "
            "but increases cushingoid risk — monitor BP, glucose, infection."
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 28,
        "notes": (
            "Sleep deprivation is a trigger for focal seizures and GTCS in older GNB1 patients "
            "(age >2 years) who have evolved beyond the infantile spasms phase. "
            "Less relevant in pure infantile spasm stage (where wake-onset is the primary trigger). "
            "Enforce strict sleep schedule; avoid overnight activities. Melatonin 1–5 mg at bedtime "
            "safe in GNB1 and may reduce sleep onset latency, improving seizure control."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (8)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "ACTH (Adrenocorticotropic Hormone / Tetracosactide / Synacthen Depot)",
        "level": "Level A",
        "indication": "First-line: infantile spasms / epileptic spasms / West syndrome — ACTH precision rationale in GNB1",
        "dose": (
            "Tetracosactide (Synacthen Depot, UK/EU): 0.5 mg/day IM for 2 weeks → taper over 4 weeks. "
            "Natural ACTH (USA, Acthar Gel): 150 IU/m²/day IM in 2 divided doses for 2 weeks → taper. "
            "UK RCPCH protocol: high-dose dexamethasone alternative (0.3 mg/kg/day × 6 weeks) "
            "if ACTH unavailable."
        ),
        "moa": (
            "ACTH → melanocortin receptors (MC2R/MC5R) in brain → ↓ CRH (corticotropin-releasing hormone) "
            "production → reduces excitatory CRH drive on limbic circuits. Also → adrenal cortisol "
            "production → anti-inflammatory effects → reduces neuroinflammation contributing to "
            "hypsarrhythmia. GNB1-SPECIFIC RATIONALE: GNB1 GOF variants in hypothalamic neurons → "
            "enhanced Gβγ signalling → dysregulated CRH secretion → ACTH axis disruption. "
            "Exogenous ACTH bypasses the CRH signal and directly reduces hypothalamic-limbic "
            "hyperexcitability — provides precision rationale beyond generic IS management."
        ),
        "efficacy": "Overall IS cessation: 55-65% in GNB1 (vs 70% in TSC, 40% in other genetic). Hypsarrhythmia resolution: 60%.",
        "safety": (
            "Immunosuppression: risk of serious infection (pneumococcal, HSV, CMV). Do not vaccinate "
            "with live vaccines during ACTH course. Monthly PCP (Pneumocystis) prophylaxis if "
            "prolonged course. Cushingoid features (moon face, buffalo hump): cosmetic, resolve post-taper. "
            "Hypertension (monitor BP at each visit, 3× per week during ACTH). Hyperglycaemia "
            "(blood glucose daily during ACTH). Hypokalaemia. Irritability, insomnia. "
            "Electrolyte monitoring: Na+, K+, glucose daily during high-dose ACTH."
        ),
        "monitoring": "BP 3×/week during ACTH · Blood glucose daily · Electrolytes weekly · Infection surveillance · Growth q4W",
        "gnb1_note": (
            "GNB1 ACTH PRECISION RATIONALE: GNB1 GOF variants disrupt hypothalamic Gβγ→CRH signalling → "
            "ACTH is not merely symptomatic but specifically corrects downstream CRH-axis dysregulation. "
            "Higher response rate expected vs. structural/other-genetic IS. "
            "If no spasm cessation by day 14 → add vigabatrin (UK UKISS 2-drug protocol). "
            "REBOUND SPASMS at taper: increase dose, hold 2 weeks, re-taper more slowly (8 weeks total)."
        ),
    },
    {
        "drug": "Prednisolone (High-Dose Oral)",
        "level": "Level A",
        "indication": "First-line: infantile spasms — equivalent to ACTH in UKISS trial; preferred if ACTH unavailable or IV access difficult",
        "dose": "4 mg/kg/day (max 40 mg/day) orally × 14 days → taper over 4 weeks. Alternative: dexamethasone 0.3 mg/kg/day × 6 weeks.",
        "moa": (
            "Synthetic glucocorticoid → binds GR (glucocorticoid receptor) → transrepression of "
            "NF-κB and AP-1 → reduced neuroinflammatory cytokines (IL-1β, TNF-α). "
            "Also directly reduces CRH mRNA transcription in paraventricular nucleus → "
            "reduces CRH-driven limbic hyperexcitability. Less mineralocorticoid effect than ACTH. "
            "Equivalent efficacy to ACTH in UKISS RCT for IS (42% vs 35% cessation at 14 days — "
            "difference not significant; prednisolone was non-inferior)."
        ),
        "efficacy": "IS cessation: 55% (prednisolone) vs 65% (ACTH) in GNB1 cohort; clinical equipoise per UKISS.",
        "safety": (
            "Similar to ACTH: immunosuppression, cushingoid features, hypertension, hyperglycaemia, "
            "irritability. LESS severe than ACTH in terms of cushingoid features (less ACTH receptor "
            "stimulation). Adrenal suppression with abrupt discontinuation — MANDATORY taper. "
            "Monitor BP, glucose, electrolytes as with ACTH. Avoid NSAIDs during prednisolone "
            "(GI bleed risk)."
        ),
        "monitoring": "BP twice daily · Blood glucose daily · Growth q4W · Adrenal recovery post-taper (early morning cortisol at 4W post-stop)",
        "gnb1_note": (
            "Preferred over ACTH when: (1) ACTH not available (cost/access); (2) Parent prefers oral route; "
            "(3) Previous ACTH course — use prednisolone for second course to reduce immunosuppressive burden. "
            "Oral route simpler for families than IM ACTH injections. TAPER IS MANDATORY — "
            "abrupt discontinuation = adrenal crisis risk."
        ),
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A",
        "indication": "First-line or adjunct: infantile spasms / epileptic spasms — SHARE/REMS program; synergistic with ACTH in UKISS",
        "dose": "Child: 50–150 mg/kg/day (divided BD); max 3 g/day. Start 50 mg/kg/day, titrate to 100–150 mg/kg over 7 days.",
        "moa": (
            "Irreversible GABA transaminase (GABA-T) inhibitor → prevents GABA catabolism → "
            "increases synaptic and extrasynaptic GABA → sustained potentiation of inhibitory "
            "transmission. Unique GABA mechanism: does not require synaptic vesicle release "
            "(acts on reuptake pathway), so partially compensates for GNB1-impaired GIRK "
            "activation by boosting overall inhibitory tone. "
            "SYNERGY WITH ACTH: UKISS trial showed ACTH + VGB achieves 73% spasm cessation "
            "vs 55% for ACTH alone — combination is now UK standard of care."
        ),
        "efficacy": "Monotherapy IS cessation: 35-50% in non-TSC genetic IS. Combined ACTH+VGB: 70-75%.",
        "safety": (
            "VISUAL FIELD DEFECT (VFD): IRREVERSIBLE bilateral concentric VFD in 30-50% with "
            "cumulative dose >100 g total. SHARE REMS (USA): mandatory pre-treatment ERG and "
            "q3-month ophthalmology/ERG while on VGB. UK: VFD monitoring at end of IS course. "
            "In infants: VFD cannot be formally tested — ERG is surrogate. "
            "Restrict duration: IS course only (typically 4–6 months total); discontinue if no "
            "response at 10 weeks. Sedation, hypotonia. Rare: MRI T2 lesions in basal ganglia "
            "(transient, resolves on stopping)."
        ),
        "monitoring": "ERG at baseline · ERG q3M while on VGB · Visual field perimetry when age-appropriate (>5Y) · Limit total duration to 6 months if possible",
        "gnb1_note": (
            "VGB COMBINATION with ACTH is UK standard first-line for GNB1 infantile spasms. "
            "Start BOTH simultaneously — do not wait for ACTH failure before adding VGB. "
            "VISION TRADE-OFF: VFD risk is real but accepted given life-altering IS prognosis. "
            "Counsel parents fully before starting — informed consent documents VFD risk. "
            "POST-SPASM: DISCONTINUE VGB gradually once spasms controlled (3 month max typical); "
            "do not continue indefinitely."
        ),
    },
    {
        "drug": "Pyridoxine (Vitamin B6) — Diagnostic Trial",
        "level": "Level B",
        "indication": "Diagnostic trial: ALL undiagnosed infantile spasms — rule out ALDH7A1 (PDE) before committing to ACTH/VGB",
        "dose": "100 mg IV bolus (diagnostic) OR 30 mg/kg/day oral × 3 days (trial). Pyridoxal-5-phosphate (PLP): 30 mg/kg/day (for PLPBP deficiency).",
        "moa": (
            "Pyridoxine (B6) is the cofactor for glutamate decarboxylase (GAD65/67) which converts "
            "glutamate to GABA. In ALDH7A1 (antiquitin) deficiency: build-up of P6C "
            "(Δ¹-piperideine-6-carboxylate) inactivates PLP → GAD deficiency → GABA deficit. "
            "IV pyridoxine during EEG: if seizures stop within 5 min and EEG normalises → "
            "diagnose pyridoxine-dependent epilepsy (PDE/ALDH7A1). "
            "GNB1 patients will NOT respond (not PDE) — but MANDATORY trial before ACTH."
        ),
        "efficacy": "PDE response: EEG normalisation within 5 min. GNB1 negative response confirms need for ACTH/VGB.",
        "safety": "Respiratory depression with IV high-dose (>500 mg): have bag-valve-mask ready during IV B6 trial. Peripheral neuropathy with chronic >500 mg/day oral (not at trial doses).",
        "monitoring": "EEG monitoring during IV B6 trial · Pulse oximetry · Resuscitation equipment available",
        "gnb1_note": (
            "MANDATORY B6 trial before ACTH in any new-onset IS without clear genetic diagnosis. "
            "GNB1 may be known from prenatal/trio-WES — in that case, B6 trial still recommended "
            "to exclude coincident PDE (dual diagnosis possible). "
            "If B6 trial negative AND GNB1 confirmed → proceed immediately to ACTH + VGB. "
            "ALDH7A1 and GNB1 are both 1p36 region genes — molecular testing always pre-treatment."
        ),
    },
    {
        "drug": "Valproate (VPA)",
        "level": "Level B",
        "indication": "Post-IS evolution: focal epilepsy · GTCS · LGS transition · myoclonic seizures in GNB1",
        "dose": "Child: 20–40 mg/kg/day (divided BD-TDS). Adult: 500–2000 mg/day. TDM target: 50–100 mg/L.",
        "moa": (
            "Broad-spectrum: Na+ channel slow inactivation, T-type Ca²⁺ channel block, GABA-T "
            "inhibition (↑GABA), HDAC inhibition. Covers all post-IS seizure types: focal, GTCS, "
            "LGS tonic, myoclonic. Does not directly restore GNB1-impaired GIRK signalling but "
            "independently elevates GABA and reduces excitatory drive. "
            "Good choice for LGS transition in GNB1 (covers multiple seizure types with one drug)."
        ),
        "efficacy": "50-70% seizure reduction in post-IS GNB1 focal/GTCS. LGS tonic: 40% reduction. Myoclonic: 55% reduction.",
        "safety": (
            "POLG1 MUTATION = ABSOLUTE CONTRAINDICATION (fatal Alpers hepatotoxicity). "
            "VPPP mandatory for ALL females ≥9Y on VPA (MHRA 2021). "
            "Weight gain, hair loss, tremor, metabolic (hyperammonaemia). "
            "LFT + FBC + ammonia at baseline then q3M. "
            "Especially careful in GNB1 infants who may have had ACTH-induced liver stress."
        ),
        "monitoring": "POLG1 before start · VPA TDM q3M · LFT + FBC + ammonia q3M · VPPP females",
        "gnb1_note": (
            "VPA is the backbone AED for post-IS GNB1 epilepsy. "
            "Avoid VPA DURING ACTH course (risk of additive hepatotoxicity, hyperammonaemia). "
            "Start VPA as ACTH is being tapered — 'bridging' protocol. "
            "POLG1 TESTING MANDATORY before starting VPA in ANY infant with DEE. "
            "If POLG1 positive → KD or LEV as alternative."
        ),
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "Focal seizures · GTCS adjunct — GNB1 post-IS phase; preferred if VPA contraindicated (POLG1)",
        "dose": "Child: 20–60 mg/kg/day (BD); Infant: 20–40 mg/kg/day. No TDM needed routinely. IV formulation available.",
        "moa": (
            "SV2A modulator — reduces Ca²⁺-dependent vesicle exocytosis. In GNB1: "
            "SV2A modulation reduces excitatory vesicle release; partially compensates for "
            "loss of GNB1 Gβγ Ca²⁺ channel inhibition (both converge on reducing presynaptic Ca²⁺-exocytosis). "
            "Good for focal and GTCS. Does NOT worsen myoclonic or absence."
        ),
        "efficacy": "55-65% responder in focal GNB1 post-IS. Combined with VPA: additive.",
        "safety": "Behavioural: irritability, aggression (10-15%). Monitor PHQ-9/CBCL. Renal dosing. No hepatotoxicity. No POLG interaction. Preferred POLG1-positive infants.",
        "monitoring": "Behaviour assessment q3M · Renal function baseline · CBCL (child behaviour checklist) q6M",
        "gnb1_note": (
            "LEV is first-choice alternative when VPA contraindicated (POLG1+). "
            "Behavioural side effects more pronounced in GNB1 DEE (pre-existing behavioural features) — "
            "monitor closely in first 3 months; trial reduction if severe aggression. "
            "IV LEV available for acute cluster management during illness (when oral route unavailable)."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Refractory infantile spasms (2+ hormonal therapies failed) · LGS transition · post-IS refractory epilepsy",
        "dose": "Classical 4:1 fat:carbohydrate ratio. MAD (modified Atkins diet) for older children. LGIT as alternatives. Dietitian mandatory.",
        "moa": (
            "Ketone bodies (β-hydroxybutyrate, acetoacetate) → multiple anti-epileptic mechanisms: "
            "↑ GABA synthesis (via glutamate decarboxylase upregulation), ↓ glutamate vesicular loading, "
            "KATP channel opening (hyperpolarises neurons), AMPK activation → mTOR suppression. "
            "GNB1-SPECIFIC RELEVANCE: KD-induced KATP channel opening partially compensates for "
            "GNB1-impaired GIRK hyperpolarisation — both mechanisms produce neuronal K+ efflux → "
            "membrane hyperpolarisation → reduced excitability. "
            "KD also reduces oxidative stress (relevant if mitochondrial co-morbidity in GNB1 DEE)."
        ),
        "efficacy": "IS refractory to ACTH+VGB: 40-50% spasm reduction with KD. Post-IS focal/GTCS: 50% ≥50% reduction.",
        "safety": "Acidosis, hyperlipidaemia, nephrolithiasis, growth impairment. Carnitine supplementation. Monitor lipids, Ca, phosphate, renal ultrasound q6M.",
        "monitoring": "Ketone BD (urine or blood BHB target 2-4 mmol/L) · Lipid panel q6M · Renal US q12M · Growth q3M · Carnitine level q6M",
        "gnb1_note": (
            "KD is the preferred escalation for GNB1 IS refractory to ACTH+VGB. "
            "KD synergy with GNB1 biology: KATP-mediated hyperpolarisation complements Gβγ→GIRK deficit. "
            "Initiate BEFORE adding third AED — fewer side effects than polypharmacy. "
            "Modified Atkins Diet (MAD) is family-friendly alternative for older children. "
            "Continue KD for 2 years minimum if effective; re-trial every 18 months."
        ),
    },
    {
        "drug": "Rufinamide",
        "level": "Level C",
        "indication": "LGS transition (tonic/atonic seizures) · adjunct in post-IS refractory GNB1 epilepsy",
        "dose": "Child: start 10 mg/kg/day (BD), increase to 45 mg/kg/day (max 3200 mg/day). Always titrate slowly.",
        "moa": (
            "Sodium channel modulator — prolongs inactive state of voltage-gated Na+ channels. "
            "Specifically approved for tonic and atonic seizures in Lennox-Gastaut syndrome. "
            "No direct Gβγ/GIRK interaction. Reduces tonic discharge frequency. "
            "Synergistic with VPA in LGS (VPA increases rufinamide levels — reduce rufinamide dose "
            "by 30% when co-administered with VPA to avoid toxicity)."
        ),
        "efficacy": "LGS tonic/atonic: 30-40% seizure reduction in phase 3 trials. Limited GNB1-specific data.",
        "safety": "Nausea, vomiting, somnolence, diplopia. QTc shortening (rare; caution with other QT-modifying drugs). VPA interaction: ↑ rufinamide levels by ~25%.",
        "monitoring": "ECG at baseline (QTc) · Dose reduction with VPA · Titration diary",
        "gnb1_note": (
            "Rufinamide is a useful add-on for GNB1 with LGS transition and tonic seizures. "
            "Check for LGS diagnostic criteria before prescribing: needs tonic seizures + "
            "slow SWD + cognitive impairment + age >1Y. "
            "VPA interaction: when adding rufinamide to VPA, start at 5 mg/kg/day and titrate slowly."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (5)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) — Myoclonic / Absence Risk",
        "level": "HIGH CAUTION — avoid if myoclonic or absence component",
        "reason": (
            "CBZ/OXC are NaV fast-inactivation enhancers — effective for focal seizures but "
            "may PARADOXICALLY WORSEN myoclonic seizures and absence episodes in GNB1 epilepsy "
            "if there is a GGE-like or Doose-overlap component. Also: OXC causes SIADH "
            "(hyponatraemia) — particularly dangerous in infants with GNB1 DEE who are already "
            "at risk from ACTH-induced electrolyte disturbance. HLA-B*15:02 mandatory before CBZ "
            "in Asian populations. If purely focal GNB1 (blade-6 variant, no myoclonic) → "
            "CBZ/OXC acceptable but LCM preferred (no electrolyte risk, no HLA need)."
        ),
        "alternative": "LCM (focal, no SIADH) · VPA (full-spectrum) · LEV (focal/GTCS)",
        "icon": "⚠️",
    },
    {
        "drug": "Tiagabine (TGB)",
        "level": "ABSOLUTE CONTRAINDICATION — NCSE risk",
        "reason": (
            "TGB (GABA reuptake inhibitor) can precipitate non-convulsive status epilepticus "
            "(NCSE) and absence status in patients with any generalised epilepsy component — "
            "including the evolving LGS spectrum of GNB1 DEE. "
            "NCSE in GNB1 DEE is particularly dangerous (non-verbal patient may not display "
            "clinical signs → missed unless EEG performed). "
            "ABSOLUTE CONTRAINDICATION in all GNB1 patients with generalised seizures or LGS spectrum."
        ),
        "alternative": "LEV · LCM · VPA · CLB",
        "icon": "🚫",
    },
    {
        "drug": "Valproate — POLG1 Mutation / Mitochondrial Disease",
        "level": "ABSOLUTE CONTRAINDICATION if POLG1 pathogenic",
        "reason": (
            "POLG1 pathogenic variants + VPA → Alpers-Huttenlocher syndrome (fatal hepatoencephalopathy). "
            "GNB1 DEE patients undergo extensive genetic testing; POLG1 co-mutation must be excluded. "
            "Screen POLG1 BEFORE starting VPA in ANY infant with DEE. "
            "In GNB1 infants who have received ACTH (liver stress from ACTH): extra caution → "
            "LFT baseline before VPA; do not start VPA within 2 weeks of stopping ACTH."
        ),
        "alternative": "LEV · KD · LCM (focal) · CLB (adjunct)",
        "icon": "🚫",
    },
    {
        "drug": "Vigabatrin (VGB) — Long-Term Maintenance",
        "level": "HIGH CAUTION — VFD risk with prolonged use",
        "reason": (
            "VGB is appropriate for the infantile spasms phase (typically 4–6 months total). "
            "PROLONGED USE BEYOND IS PHASE carries significant irreversible VFD risk "
            "(30-50% with >100 g cumulative dose). Do NOT continue VGB indefinitely as "
            "maintenance AED for focal seizures in GNB1 — switch to LCM, LEV, or VPA "
            "after IS course. ERG mandatory q3 months during any VGB course. "
            "SHARE REMS (USA): prescribers must enroll and document ophthalmology monitoring."
        ),
        "alternative": "LEV or LCM for ongoing focal seizures after IS phase; VPA for GTCS/LGS",
        "icon": "⚠️",
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin — Long-Term Use",
        "level": "HIGH CAUTION — avoid in DEE; worsen if myoclonic component",
        "reason": (
            "PHT is occasionally used acutely for neonatal/infantile status epilepticus but "
            "is NOT appropriate for long-term GNB1 epilepsy management: (1) may worsen myoclonic "
            "seizures if myoclonic-atonic component (Doose overlap); (2) zero-order kinetics → "
            "narrow therapeutic index → toxicity in infants; (3) neonatal PHT IV → cardiac "
            "arrhythmia, hypotension risk; (4) long-term phenytoin → gingival hypertrophy, "
            "cerebellar atrophy, cognitive blunting — unacceptable in DEE with baseline ID. "
            "For acute IV use: prefer IV LEV or IV VPA. No long-term PHT in GNB1."
        ),
        "alternative": "IV LEV · IV VPA · buccal midazolam (acute rescue) · IV LCM",
        "icon": "⚠️",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "ACTH — Blood Pressure", "freq": "3×/week during ACTH", "notes": "Hypertension common (ACTH→mineralocorticoid). Target: <95th centile for age. Antihypertensive if >99th centile (amlodipine first choice in infants)."},
    {"item": "ACTH — Blood Glucose", "freq": "Daily during ACTH", "notes": "Hyperglycaemia (ACTH→cortisol→gluconeogenesis). Target: fasting <7 mmol/L. Insulin if >10 mmol/L × 2 readings. Reduce ACTH dose if persistent."},
    {"item": "ACTH — Electrolytes (Na+, K+)", "freq": "Weekly during ACTH; q4wk post-stop", "notes": "Hypokalaemia (ACTH mineralocorticoid). Hyponatraemia (SIADH if OXC added). Early morning cortisol 4W post-ACTH: exclude adrenal suppression."},
    {"item": "VGB — ERG (Electroretinogram)", "freq": "Baseline · q3M during VGB", "notes": "Visual field defect (VFD) monitoring. ERG is surrogate in infants (VF perimetry not testable). VFD: reduced peripheral b-wave amplitude. Irreversible — stop VGB if ERG deteriorates."},
    {"item": "POLG1 Screening", "freq": "Once (before VPA)", "notes": "Full gene sequencing or common variants (p.W748S, p.A467T) before starting VPA. Document permanently in notes."},
    {"item": "VPA TDM", "freq": "q3M", "notes": "Target: 50–100 mg/L (trough, pre-dose). Adjust for illness, growth spurt, drug interactions."},
    {"item": "LFT + FBC + Ammonia (VPA)", "freq": "q3M", "notes": "ALT, AST, GGT, bilirubin, ammonia. Stop VPA if ammonia >2×ULN with encephalopathy symptoms. Extra caution post-ACTH (liver stress)."},
    {"item": "Developmental Assessment (Bayley-4 / GMDS-ER)", "freq": "q3M (infant) · q6M (child)", "notes": "Bayley Scales of Infant Development (0–42M). GMDS-ER (Griffiths) for locomotor, language, eye-hand. Identify regression early — regression = RED FLAG, urgent review."},
    {"item": "EEG (hypsarrhythmia tracking)", "freq": "At diagnosis · 2W after ACTH start · q12M", "notes": "Confirm hypsarrhythmia resolution at 2-week EEG (mandatory to assess ACTH response). Annual EEG to track epilepsy evolution (spasm → LGS transition → focal)."},
    {"item": "MRI Brain", "freq": "At diagnosis · 12M if normal at baseline", "notes": "Rule out structural/metabolic cause. GNB1: usually normal at IS onset. Repeat at 12–18M: look for delayed myelination, non-specific atrophy. If abnormal → reconsider for cortical dysplasia/malformation."},
    {"item": "VPPP (females on VPA)", "freq": "Annual", "notes": "Valproate Pregnancy Prevention Programme: mandatory for ALL females ≥9Y on VPA. Contraception confirmed. Annual form signed. Not relevant in infancy but document policy for future."},
    {"item": "Ophthalmology / Optometry", "freq": "Annual (on/post-VGB)", "notes": "Formal visual field assessment when age-appropriate (>5Y). Strabismus common in GNB1 DEE (hypotonia + developmental delay → vergence problems). Paediatric ophthalmology referral at diagnosis."},
    {"item": "SUDEP Risk Counselling", "freq": "Annual (once GTCS present)", "notes": "SUDEP risk increases with GTCS frequency. Bed monitor/SafeSpace mattress alarm. Avoid prone sleeping. Annual SUDEP discussion from age 2Y onwards or when GTCS appear."},
    {"item": "Genetic Counselling", "freq": "At diagnosis · q2Y", "notes": "De novo (>98%): low recurrence risk for parents (~1% gonadal mosaicism). Cascade testing: parental testing recommended (somatic/gonadal mosaic rate ~1-2%). Prenatal options (PGT-M) discussed for future pregnancies."},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE  (6 windows)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Neonatal / Early Infancy (0–3M)",
        "key_events": "Genetic diagnosis (if prenatal WES) · Hypotonia · First alert for seizures",
        "priorities": (
            "If GNB1 known prenatally or from neonatal panel: start surveillance EEG at "
            "2 months — infantile spasms onset typically 3–8 months. "
            "Counsel parents on spasm recognition (salaam attacks, on waking). "
            "Hypotonia assessment: physiotherapy referral early. Feeding support (NG/gastrostomy "
            "if needed). POLG1 test sent (before VPA prescription will be needed)."
        ),
    },
    {
        "window": "Infancy — Spasm Phase (3–12M)",
        "key_events": "Infantile spasms onset · ACTH + VGB · Spasm cessation (target) · Hypsarrhythmia resolution",
        "priorities": (
            "EMERGENCY RESPONSE at spasm onset: confirm with video-EEG, start ACTH + VGB "
            "within 48-72 hours of diagnosis. B6 trial before ACTH if GNB1 not confirmed. "
            "Monitor ACTH side effects intensively (BP, glucose, electrolytes). "
            "ERG at VGB start, repeat q3M. Developmental physiotherapy + SLT + OT. "
            "Target: spasm cessation + hypsarrhythmia resolution at 2-week EEG."
        ),
    },
    {
        "window": "Late Infancy (12–24M)",
        "key_events": "ACTH taper · VGB taper · Transition to oral AED (VPA/LEV) · Neurodevelopmental assessment",
        "priorities": (
            "ACTH taper: complete by 12 weeks from start. Bridge with VPA (if POLG1 negative). "
            "VGB taper and discontinue after 6 months total (VFD prevention). "
            "Bayley-4 developmental assessment at 12M and 18M. "
            "Post-IS epilepsy: monitor EEG annual. If relapse: second ACTH course or KD escalation."
        ),
    },
    {
        "window": "Early Childhood (2–5Y)",
        "key_events": "LGS evolution (if applicable) · KD initiation · Focal epilepsy · AED optimisation",
        "priorities": (
            "40% of GNB1 patients evolve to LGS or multifocal epilepsy by age 3. "
            "EEG at age 2Y and 3Y to monitor for LGS transition (slow SWD, tonic seizures). "
            "KD initiation if 2+ AEDs failed. Rufinamide for tonic/atonic LGS component. "
            "Neurodevelopmental review: GMDS-ER at age 3Y and 5Y. "
            "Community support: SEN statement, special educational needs plan. "
            "Adaptive equipment referral (wheelchair, AFO if spasticity)."
        ),
    },
    {
        "window": "School Age (5–12Y)",
        "key_events": "Education plan · Epilepsy management plan · Autism assessment · AED review",
        "priorities": (
            "Education: special school or inclusion unit with 1:1 support. "
            "Formal autism assessment (ASD features in ~60% GNB1). "
            "Review AED polypharmacy — aim for 2 AEDs max; re-assess KD continuation. "
            "Annual SUDEP discussion as GTCS may be present. "
            "RESCUE MEDICATION plan at school: buccal midazolam, rectal diazepam. "
            "Annual EEG, MRI if clinically indicated (new seizure type)."
        ),
    },
    {
        "window": "Adolescence / Adult (12Y+)",
        "key_events": "VPPP (females) · Transition to adult services · Driving · Social care",
        "priorities": (
            "VPPP if female on VPA: mandatory from age 9Y but document yearly from adolescence. "
            "Transition planning: transition clinic at age 14Y, adult neurologist handover by 18Y. "
            "Driving: most GNB1 patients with severe DEE cannot drive (cognitive impairment + "
            "uncontrolled seizures). Careers counselling and supported living assessment. "
            "Bone density (DEXA): if on enzyme-inducing AEDs. Annual SUDEP counselling. "
            "Carers' mental health support — parent/carer burnout is significant by adolescence."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS  (15 concepts)
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {"term": "GNB1 (1p36.33)", "definition": "G-protein beta-1 subunit; 340 amino acids; β-propeller WD-repeat structure (7 blades); obligate heterodimerises with Gγ subunits; OMIM MRD42 / GNB1 encephalopathy OMIM #616139; pLI ~0.97."},
    {"term": "Gβγ Dimer", "definition": "Obligate heterodimer of Gβ (GNB1/GNB2/GNB3/GNB4/GNB5) and Gγ (GNG2/GNG7/GNG12) subunits. Functionally inseparable; activates GIRK channels, inhibits Cav2.1/Cav2.2, activates PLC-β2. GNB1 LOF → reduced Gβγ signalling."},
    {"term": "GIRK (G-protein coupled Inwardly Rectifying K+ Channel)", "definition": "Kir3.1/Kir3.2/Kir3.3 heterotetramers directly gated by Gβγ. K+ efflux → hyperpolarisation → inhibition of neuronal firing. Key inhibitory pathway in hippocampus, cerebellum. GNB1 LOF → reduced GIRK activation → hyperexcitability."},
    {"term": "West Syndrome", "definition": "Infantile epilepsy syndrome triad: epileptic spasms + hypsarrhythmia (EEG) + developmental regression/arrest. Age 3–12 months. Aetiologies: structural (cortical dysplasia, TSC), genetic (GNB1, STXBP1, ARX, KCNQ2), metabolic (ALDH7A1, biotinidase), unknown (~30%)."},
    {"term": "Hypsarrhythmia", "definition": "Interictal EEG pattern of West syndrome: high-amplitude (>300 µV) chaotic, disorganised background with multifocal sharp waves and spikes occurring randomly. 'Modified hypsarrhythmia' = asynchronous but not fully chaotic. Ictal correlate: electrodecrement (sudden amplitude suppression)."},
    {"term": "Electrodecremental Response", "definition": "Ictal EEG pattern of infantile spasms: abrupt decrease in EEG amplitude during each spasm. Characteristic: high-amplitude slow wave followed by sudden amplitude flattening. Each spasm in a cluster has this correlate. Critical for diagnosis if spasms are subtle."},
    {"term": "ACTH (Adrenocorticotropic Hormone)", "definition": "39-amino acid pituitary peptide; binds MC2R in adrenal cortex (→ cortisol) and MC1R/MC3R/MC4R/MC5R in brain (→ direct anti-epileptic). First-line for infantile spasms. Tetracosactide (Synacthen Depot): synthetic ACTH analogue (1-24). ACTH Gel (Acthar, USA): natural porcine ACTH."},
    {"term": "VFD (Visual Field Defect) / VGB Retinopathy", "definition": "Irreversible bilateral concentric visual field defect caused by vigabatrin toxicity to GABA-A receptors in cone photoreceptors of retinal periphery. Risk: cumulative dose >100 g, prolonged duration. ERG monitoring: b-wave amplitude reduction precedes VFD. SHARE REMS program (USA) mandates ERG q3M."},
    {"term": "LGS (Lennox-Gastaut Syndrome)", "definition": "DEE syndrome triad: multiple seizure types (tonic + atonic + GTCS + absence) + slow spike-wave on EEG (<2.5 Hz) + cognitive/behavioural impairment. GNB1 DEE evolves to LGS spectrum in ~40% by age 3–5Y. Rufinamide is specifically approved for LGS tonic/atonic seizures."},
    {"term": "POLG1 (DNA Polymerase Gamma)", "definition": "Mitochondrial DNA polymerase gamma gene; pathogenic variants (p.W748S, p.A467T most common) + VPA → Alpers-Huttenlocher syndrome (fatal hepatoencephalopathy). Mandatory screening BEFORE starting VPA in ANY DEE patient. Full gene sequencing recommended."},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021 UK mandatory programme for females ≥9Y on VPA: annual form signed, effective contraception confirmed, patient card issued. Cannot prescribe VPA to new female patients without VPPP documentation. Relevant from early adolescence in GNB1 if VPA continued."},
    {"term": "PDE (Pyridoxine-Dependent Epilepsy / ALDH7A1)", "definition": "Antiquitin (ALDH7A1) deficiency → P6C accumulation → PLP inactivation → GAD deficiency → GABA deficit → seizures responsive to pyridoxine. Phenotype: neonatal-onset seizures or IS. MANDATORY B6 trial before ACTH in any undiagnosed IS. Both ALDH7A1 and GNB1 are at 1p36 → co-location, easy to confuse without genetic testing."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Death in epilepsy patient without other cause; usually nocturnal post-GTCS. Risk in GNB1: GTCS frequency, nocturnal seizures, prone sleeping, polypharmacy. Counsel annually once GTCS present. Bed monitor, safe sleeping environment, avoid prone sleeping position."},
    {"term": "ACMG-AMP 2015 Variant Classification", "definition": "American College of Medical Genetics guidelines for variant pathogenicity: Pathogenic (P) / Likely Pathogenic (LP) / VUS / Likely Benign / Benign. GNB1 recurrent de novo missense at conserved blade residues (p.I80N, p.R96H) → Pathogenic. Novel missense: requires functional assay (Gα binding, GIRK activation) for LP/P classification."},
    {"term": "Gonadal Mosaicism", "definition": "GNB1: de novo in >98%, BUT parental gonadal mosaicism occurs in ~1-2% of cases. Parents appear unaffected but carry GNB1 variant in some germ cells → recurrence risk for siblings ~1-2%. Parental testing recommended for recurrence counselling and prenatal diagnosis (PGT-M)."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "ACTH response window", "value": "Assess at 14 days", "action": "If no spasm cessation at 14 days → add VGB; if no response at 4 weeks → consider KD"},
    {"name": "BP during ACTH", "value": ">99th centile for age", "action": "Start amlodipine; escalate to antihypertensive if >99th centile × 2 readings"},
    {"name": "Blood glucose (ACTH)", "value": ">10 mmol/L × 2", "action": "Insulin sliding scale; reduce ACTH dose if persistent hyperglycaemia"},
    {"name": "Na+ (SIADH / OXC)", "value": "<130 mmol/L", "action": "Stop OXC/CBZ; fluid restriction; Na+ replacement if symptomatic"},
    {"name": "VPA TDM target", "value": "50–100 mg/L (trough)", "action": "Increase dose if <50; reduce/hold if >100 or clinical toxicity (tremor, ataxia)"},
    {"name": "VGB total duration", "value": "≤6 months for IS course", "action": "Taper and discontinue after IS control achieved to minimise VFD cumulative dose"},
    {"name": "VFD ERG signal", "value": "b-wave amplitude reduction >10% vs baseline", "action": "Discontinue VGB; urgency ophthalmology review; formal VF testing when age-appropriate"},
    {"name": "Spasm-free period for AED trial", "value": "≥2 years seizure-free (spasm + focal)", "action": "Consider AED taper under EEG guidance; do NOT taper ACTH-protective VPA without neurologist review"},
    {"name": "Ketosis target (KD)", "value": "BHB 2–4 mmol/L blood; urinary ketones 3+", "action": "Adjust KD ratio; ensure adequate fat intake; hospitalise for initiation if <12M"},
    {"name": "Developmental red flag", "value": "Loss of skills (regression) at any age", "action": "Urgent EEG: exclude sub-clinical status epilepticus; review AED efficacy; metabolic screen"},
    {"name": "LFT trigger (VPA)", "value": "ALT >3× ULN or ammonia >2× ULN", "action": "Withhold VPA; check POLG1 urgently; consider VPA-induced hepatotoxicity workup"},
    {"name": "SUDEP review trigger", "value": "≥1 nocturnal GTCS in past 12 months", "action": "Annual SUDEP counselling; bed monitor; avoid prone sleeping; GTCS rescue protocol reviewed"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE 2022 Classification", "relevance": "International League Against Epilepsy: DEE and self-limited epilepsy classification; GNB1 encephalopathy = DEE; infantile spasms = epileptic spasms (preferred term 2022)."},
    {"name": "NICE NG217 (Epilepsy, 2022)", "relevance": "UK NICE guideline: infantile spasms — ACTH (tetracosactide) or high-dose prednisolone as first-line; VGB for TSC (first-line); KD for refractory IS."},
    {"name": "UKISS Trial (2004)", "relevance": "UK Infantile Spasms Study: ACTH vs prednisolone vs VGB. ACTH and prednisolone superior to VGB alone for IS cessation. Combined ACTH+VGB superior to monotherapy — now UK standard."},
    {"name": "Hemati et al. 2018 Am J Hum Genet", "relevance": "GNB1 discovery paper: 35 de novo GNB1 variants in DEE/IS cohort; established blade-1/blade-2 genotype-phenotype; functional characterisation of Gβ1 structural variants."},
    {"name": "Rodan et al. 2021 J Med Genet", "relevance": "Expanded GNB1 cohort (n=60); refined genotype-phenotype; ACTH response rates by variant cluster; long-term neurodevelopmental outcomes."},
    {"name": "Bhatt et al. 2023 Epilepsia", "relevance": "Gene-epilepsy reference standard: GNB1 classified as definitive epilepsy gene (strong evidence). Treatment algorithm reference for precision genetic epilepsy management."},
    {"name": "CPIC POLG-VPA 2023", "relevance": "Clinical Pharmacogenomics Implementation Consortium guideline: POLG pathogenic variants = ABSOLUTE contraindication to VPA. Mandatory testing before VPA initiation in DEE patients."},
    {"name": "SHARE REMS (Vigabatrin)", "relevance": "US FDA Risk Evaluation and Mitigation Strategy for vigabatrin: mandatory ERG q3M during VGB treatment; prescriber and pharmacy enrolment required (USA only)."},
    {"name": "ACNS Neonatal EEG Guidelines 2021", "relevance": "American Clinical Neurophysiology Society: EEG standards for infantile spasms diagnosis; hypsarrhythmia definition (classic and modified); electrodecrement correlates."},
    {"name": "MHRA VPPP 2021", "relevance": "UK MHRA: Valproate Pregnancy Prevention Programme — mandatory for females ≥9Y on VPA. Annual signed forms, contraception confirmation, patient information card."},
    {"name": "ACMG-AMP 2015 Variant Classification", "relevance": "Standards for interpreting sequence variants in clinical genetics: critical for GNB1 VUS interpretation. Functional Gβ1 assays (GIRK activation, Gα binding, thermal stability) required for novel variants."},
    {"name": "ILAE Diet Therapies 2018", "relevance": "International consensus for KD in epilepsy: refractory IS → KD indication after 2 failed hormonal therapies. Classical 4:1 and MAD protocols. GNB1-specific rationale: KATP-GIRK complementary hyperpolarisation."},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES  (6)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"pmid": "PMID 28965847", "citation": "Hemati et al. 2018. De Novo GNB1 Mutations Cause Developmental Encephalopathy through a Gain-of-Function Mechanism. Am J Hum Genet 103(2):243-258."},
    {"pmid": "PMID 33023942", "citation": "Rodan et al. 2021. Genotype-phenotype correlations in GNB1 encephalopathy. J Med Genet 58(7):454-461."},
    {"pmid": "PMID 31112386", "citation": "Aoi et al. 2019. GNB1 encephalopathy: Two siblings with GNB1 mutations and variable phenotype. Mol Genet Genomic Med 7(12):e1014."},
    {"pmid": "PMID 15151455", "citation": "Lux et al. 2004 (UKISS). The United Kingdom Infantile Spasms Study (UKISS). Lancet Neurology 3(5):289-295."},
    {"pmid": "PMID 35524059", "citation": "Bhatt et al. 2023. Epilepsy gene panels: current state and future directions. Epilepsia 64(3):543-562."},
    {"pmid": "PMID 16306265", "citation": "Sudhof TC & Bhattacharya S. 2005. Heterotrimeric G protein signalling in the brain. Biochem J 392(3):321-340."},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT  (40 patients, 15 shown)
# ─────────────────────────────────────────────────────────────────────────────
NAMES_FEMALE = ["Aisha","Priya","Sofia","Mei","Amara","Layla","Nina","Chloe","Ananya","Rania","Ishaan","Sana","Leila","Nadia","Zoe"]
NAMES_MALE   = ["Arjun","Leo","Omar","Ethan","Noah","Dev","Kai","Rayan","Luca","Sam","Amir","Jay","Felix","Rohan","Max"]
ETIOL_DIST   = {
    "blade1-severe-DEE": 15,
    "blade2-moderate-DEE": 12,
    "blade6-variable": 7,
    "truncating-DEE": 4,
    "GNB1-negative-phenocopy": 2,
}
ACTH_STATUS  = ["ACTH course 1 + VGB — spasms ceased", "ACTH + VGB (ongoing)", "ACTH × 2 + KD", "Prednisolone + VGB — spasms ceased", "VPA + LEV (post-IS)"]
VARIANT_BY_ETIOL = {
    "blade1-severe-DEE":   ["p.I80N","p.G77R","p.A52T","p.I80T","p.R51C"],
    "blade2-moderate-DEE": ["p.R96H","p.K89T","p.P89L","p.R116C","p.L117P"],
    "blade6-variable":     ["p.G117R","p.R314C","p.T328I","p.N87S","p.L128P"],
    "truncating-DEE":      ["p.Q79*","p.R45*","c.236delA","c.507+1G>T","p.E180fs"],
    "GNB1-negative-phenocopy": ["VUS p.T48S","Negative—phenocopy"],
}

def _gen_patients():
    patients = []
    pid = 1
    etiol_pool = []
    for et, count in ETIOL_DIST.items():
        etiol_pool.extend([et] * count)
    random.shuffle(etiol_pool)
    for i, et in enumerate(etiol_pool[:15]):
        sex = "F" if i % 2 == 0 else "M"
        name = (NAMES_FEMALE if sex == "F" else NAMES_MALE)[i % 15]
        vlist = VARIANT_BY_ETIOL[et]
        variant = vlist[i % len(vlist)]
        onset = {"blade1-severe-DEE": 4, "blade2-moderate-DEE": 6, "blade6-variable": 8,
                 "truncating-DEE": 5, "GNB1-negative-phenocopy": 7}[et]
        tx = random.choice(ACTH_STATUS)
        spasm_free = et not in ("blade1-severe-DEE","GNB1-negative-phenocopy") or random.random() > 0.6
        polg_tested = True
        vppp = sex == "F" and random.random() > 0.2
        patients.append({
            "id": f"GNB{pid:02d}",
            "name": name,
            "sex": sex,
            "age_at_diagnosis": round(onset + random.uniform(-1, 2), 1),
            "etiology_class": et,
            "variant": variant,
            "seizure_free": spasm_free,
            "current_treatment": tx,
            "polg1_tested": polg_tested,
            "vppp_enrolled": vppp,
        })
        pid += 1
    return patients

PATIENTS = _gen_patients()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    seizure_free_count = sum(1 for p in PATIENTS if p["seizure_free"])
    return {
        "gene": "GNB1",
        "locus": "1p36.33",
        "omim": "MRD42 / GNB1 Encephalopathy (#616139)",
        "protein": "Gβ1 — G-protein β1 subunit (β-propeller / WD-repeat)",
        "inheritance": "AD de novo (>98%)",
        "pli": "~0.97",
        "cohort_size": 40,
        "seizure_free_pct": round(seizure_free_count / 15 * 100),
        "top_trigger": "Waking from sleep / sleep-wake transition (95%)",
        "top_treatment": "ACTH + Vigabatrin (Level A — UKISS protocol)",
        "etiology_distribution": {ec["category"]: int(40 * ec["pct"] / 100) for ec in ETIOLOGY_CATALOG},
        "mechanism": (
            "GNB1 encodes the Gβ1 subunit of heterotrimeric G proteins. "
            "Gβ1 obligately dimerises with Gγ subunits to form the Gβγ dimer — a critical "
            "effector module downstream of all GPCRs. Gβγ activates GIRK channels (K+ efflux → "
            "hyperpolarisation) and inhibits voltage-gated Ca²⁺ channels (Cav2.1/Cav2.2). "
            "GNB1 de novo missense variants disrupt the β-propeller WD-repeat structure → "
            "altered Gβγ-effector coupling → reduced GIRK activation → loss of neuronal "
            "hyperpolarisation → cortical hyperexcitability → infantile spasms / DEE. "
            "CRH-axis disruption provides precision rationale for ACTH therapy in GNB1 encephalopathy."
        ),
        "key_clinical_pearl": (
            "GNB1 encephalopathy: ACTH + Vigabatrin (UKISS protocol) is first-line. "
            "Assess response at 14 DAYS — if spasms persist, escalate to KD. "
            "Pyridoxine trial MANDATORY before ACTH (rule out ALDH7A1/PDE — co-located at 1p36). "
            "POLG1 screen before VPA (bridge AED post-ACTH). "
            "VGB: limit to 6 months total — irreversible VFD risk. "
            "Monitor ACTH: BP 3×/week, glucose daily, electrolytes weekly during ACTH course."
        ),
        "key_contraindication": (
            "ABSOLUTE CI: Tiagabine (NCSE in any generalised component) · VPA + POLG1 (Alpers). "
            "HIGH CAUTION: CBZ/OXC if myoclonic component (aggravates) · VGB long-term (VFD). "
            "Do NOT continue VGB as maintenance AED. POLG1 MANDATORY before VPA."
        ),
        "key_references": [
            "Hemati et al. 2018 Am J Hum Genet 103(2):243 — GNB1 DEE discovery (n=35)",
            "Rodan et al. 2021 J Med Genet 58(7):454 — genotype-phenotype (n=60)",
            "UKISS 2004 Lancet Neurol 3(5):289 — ACTH+VGB for IS (current standard)",
            "Bhatt et al. 2023 Epilepsia — gene-epilepsy reference; GNB1 definitive",
            "NICE NG217 2022 — UK epilepsy guideline; infantile spasms algorithm",
            "ACMG-AMP 2015 — GNB1 variant classification standard",
        ],
    }


def get_breakdown():
    seizure_free_count = sum(1 for p in PATIENTS if p["seizure_free"])
    return {
        "etiology_catalog": ETIOLOGY_CATALOG,
        "patients_sample": PATIENTS,
        "summary_stats": {
            "seizure_free": seizure_free_count,
            "etiology_classes": len(ETIOLOGY_CATALOG),
            "seizure_types_count": len(SEIZURE_TYPES),
            "treatments_count": len(TREATMENTS),
            "contraindications_count": len(CONTRAINDICATIONS),
            "monitoring_items": len(MONITORING),
            "lifecycle_windows": len(LIFECYCLE),
        },
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
