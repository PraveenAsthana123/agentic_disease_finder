"""
TSC2 Epilepsy — Tuberous Sclerosis Complex (TSC / mTOR / Tuberin / 16p13.3)
=============================================================================
40-patient cohort · TSC2 LOF · mTOR hyperactivation · Cortical tubers · Epilepsy (severe)

TSC2 BIOLOGY:
TSC2 (16p13.3) encodes Tuberin, a 200-kDa GTPase-activating protein (GAP)
that forms the TSC1/TSC2 heterodimeric complex with Hamartin (TSC1, 9q34.13).
Tuberin's C-terminal GAP domain maintains Rheb (Ras homolog enriched in brain)
in its GDP-bound (inactive) state, thereby suppressing mTORC1 activity.

mTOR PATHWAY & CORTICAL TUBER PATHOMECHANISM:
  - TSC2 LOF → Tuberin loss → TSC1/TSC2 complex disrupted (Hamartin also
    destabilised/degraded) → Rheb accumulates GTP-loaded (constitutively
    active) → mTORC1 hyperactivation → uncontrolled S6K1 and 4E-BP1
    phosphorylation → dysregulated protein synthesis, cell growth, proliferation.
  - In developing brain: mTORC1 hyperactivation during cortical neurogenesis →
    arrested radial neuronal migration → balloon cells (enlarged dysmorphic
    cells with aberrant myelination) + dysmorphic neurons → cortical tubers
    (focal patches of disorganised, epileptogenic cortex).
  - Perilesional cortex: mTOR-driven upregulation of glutamate receptors (GluN2B)
    + loss of PV+ GABAergic interneurons → focal hyperexcitability → seizure onset.
  - Tuber burden in TSC2 is GREATER than TSC1: mean 9–12 tubers (TSC2) vs 6–8
    (TSC1). Type-C tubers (cystic, FLAIR-bright) most epileptogenic (~80%).
    Subependymal nodules calcify with age; SEGA at foramen of Monro.

TSC2 vs TSC1 GENOTYPE-PHENOTYPE (CRITICAL DISTINCTION):
  TSC2 (16p13.3) — ~70% of all TSC. MORE SEVERE:
    · More cortical tubers (9–12 vs 6–8), larger total tuber volume
    · Intellectual disability: 50–60% (vs ~30% in TSC1)
    · ASD / TAND: 35–45%
    · Drug-resistant epilepsy: ~55% (vs ~45% TSC1)
    · Pulmonary LAM (lymphangioleiomyomatosis): 25–35% in females
      (vs rare in TSC1 — clinically significant difference)
    · Renal AML (angiomyolipoma): 75–80% (similar to TSC1)
    · TSC2+PKD1 contiguous deletion (~2%): chromosome 16p13.3 spans both
      TSC2 and PKD1 (autosomal dominant polycystic kidney disease type 1);
      large deletions remove both → severe early-onset polycystic kidneys
      + severe TSC epilepsy in infancy.
    · SEGA: 10–15% (similar to TSC1)
    · Infantile spasms: ~45% (vs ~35% in TSC1 — higher burden)
    · Mean onset: 6–8 months (vs 8–10 months for TSC1)
    · Seizure-free rate: ~20% (vs ~25% TSC1)
  TSC1 (9q34.13) — ~30% of TSC. Milder on average (see TSC1 dashboard).
  Mosaic TSC: ~15% overall; requires NGS ≥200× depth.

TSC2 MOLECULAR VARIANTS:
  - Truncating variants (frameshift, nonsense, large deletions): most severe;
    complete Tuberin loss; highest tuber burden; NMD-susceptible transcripts.
  - Missense in GAP domain (e.g., R611Q, R1200T): partial LOF; attenuated
    phenotype. A minority retain partial GAP activity.
  - Splice variants: may reduce Tuberin expression to 20–60% of normal;
    phenotype intermediate.
  - Contiguous TSC2+PKD1 deletion: large 16p13.3 deletion; severe neonatal/
    infantile TSC + polycystic kidneys; suspect if PKD phenotype at birth.
  - Somatic mosaic: ~5–10% of TSC2 cases; requires ultra-deep NGS (≥500×)
    or tissue-specific sequencing (blood vs hair follicle vs buccal).

CORTICAL TUBER BIOLOGY & EPILEPSY SURGERY:
  - 3T/7T MRI: FLAIR-hyperintense tubers; T2* GRE for calcium.
    FDG-PET: interictal hypometabolism marks epileptogenic tuber(s).
    AMT-PET (alpha-[11C]-methyl-L-tryptophan): tryptophan trapping in
    epileptogenic tubers — highly specific (Juhász 2006); guides surgical target.
  - TSC2 multifocal epilepsy: often requires invasive evaluation (SEEG)
    to identify the primary epileptogenic tuber among 9–12 candidates.
  - Resection of single epileptogenic tuber: seizure-free in 50–60% at 2y.
    TSC2 outcomes are less favourable than TSC1 due to higher tuber count
    and bilateral distribution.
  - LITT (laser interstitial thermal therapy): minimally invasive ablation;
    50–60% seizure-free at 1y for accessible single tuber.
  - Neuromodulation: VNS adjunct in drug-resistant multifocal TSC2 epilepsy.

mTOR INHIBITOR PRECISION THERAPY — EVEROLIMUS (AFINITOR):
  Everolimus = allosteric mTORC1 inhibitor (rapamycin analogue). Binds FKBP12
  → FKBP12-everolimus complex inhibits mTORC1 → reduced S6K1/4E-BP1 phospho
  → corrects downstream protein synthesis dysregulation.

  FDA APPROVED IN TSC:
  - Focal-onset seizures ≥2y: EXIST-3 (Bissler 2017 Lancet Neurology):
    Response (≥50% seizure reduction): 40% vs 15% placebo; median reduction
    29% vs 0%. Trough targets: 3–7 ng/mL (lower) vs 9–15 ng/mL (higher).
    Both targets superior to placebo; 9–15 more effective but more AEs.
  - SEGA ≥2y: EXIST-1 (Franz 2013): ≥50% volume reduction in 35% vs 0% placebo.
  - Renal AML ≥18y: EXIST-2 (Bissler 2013): 42% response vs 0% placebo.
  - Pulmonary LAM ≥18y: MILES trial (Bissler 2011): FEV1 stabilisation.

  DOSING: 4.5 mg/m² daily (BSA-based, paediatric); 10 mg once daily (adult).
  Adjust to serum trough 5–15 ng/mL (TSC seizures: 3–7 first line).
  Monitor trough at 2 weeks, monthly until stable, then q3–6M.

  TSC2-SPECIFIC PHARMACOLOGY NOTE:
  Since TSC2 LOF removes all Tuberin GAP activity (complete mTORC1 release),
  the dose–response to everolimus may require trough escalation to 9–15 ng/mL
  in drug-resistant cases. CYP3A4 drug interactions critical:
  - CYP3A4 inducers (CBZ, OXC, PHT, PB, rifampicin) → reduce everolimus trough
    2–4×. If CBZ needed, escalate everolimus dose and recheck trough q2w.
  - CYP3A4 inhibitors (fluconazole, itraconazole, clarithromycin) → increase
    everolimus trough. Reduce dose; recheck at 1 week.

  KEY SAFETY:
  - Stomatitis/oral mucositis: 70–80%; non-alcoholic mouthwash; dose reduction
    for Grade ≥2.
  - Infections (bacterial/viral/fungal): PCP prophylaxis (TMP-SMX) in children
    on full-dose immunosuppression. Avoid live vaccines.
  - Hyperlipidaemia: 70%; LDL/TG q3M.
  - Hyperglycaemia: 50%; HbA1c q3M; caution in pre-diabetic.
  - Interstitial lung disease (ILD): rare but serious; chest CT if new
    respiratory symptoms. Dose reduce or discontinue.
  - Wound healing: hold 1 week pre/post major surgery.

VIGABATRIN (VGB) FOR TSC2 INFANTILE SPASMS:
  - VGB FIRST-LINE for infantile spasms in TSC (Level A — UKISS trial;
    TSC-specific response 95%). TSC2 patients have HIGHER IS prevalence
    (~45%) vs TSC1 (~35%). Earlier treatment → better neurodevelopment.
  - VGB retinal toxicity: irreversible visual field constriction 30–50%
    adults. SHARE REMS (USA): mandatory enrolment + ERG monitoring q3M
    (infants), q6M (children/adults).
  - EPISTOP/PREVENT trial: presymptomatic VGB (started at EEG-only
    biomarker, before clinical spasms) → 50% reduction in epilepsy onset.
    All TSC2 neonates: EEG surveillance from 1 month of life.

PULMONARY LAM (TSC2-SPECIFIC — SIGNIFICANT DIFFERENCE FROM TSC1):
  - Lymphangioleiomyomatosis (LAM): TSC2-driven mTORC1 hyperactivation
    in smooth muscle-like cells → pulmonary cysts → progressive dyspnoea,
    pneumothorax. Affects ~30% of TSC2 females vs rare in TSC1 females.
  - Everolimus MILES trial: FEV1 stabilisation + modest improvement.
  - Annual chest HRCT for females ≥15 years with TSC2.
  - Baseline pulmonary function tests in adolescent TSC2 females.

RENAL MONITORING (TSC2 + TSC2+PKD1):
  - AML (angiomyolipoma): annual renal ultrasound. AML ≥3cm → everolimus.
    Embolisation if bleeding (Wunderlich syndrome).
  - TSC2+PKD1 contiguous deletion: neonatal polycystic kidney disease
    + hypertension. Creatinine/eGFR q6M from infancy. Renal replacement
    therapy risk in early childhood if large deletion.
  - Blood pressure monitoring annual from age 3 years.

TAND (TSC-Associated Neuropsychiatric Disorders) IN TSC2:
  - ASD: 35–45% (vs 30–35% TSC1). Higher severity correlated with tuber count.
  - ID: 50–60% (vs 30% TSC1). Moderate-severe in 30%.
  - ADHD: 40–50%. Anxiety: 35%.
  - TAND checklist annually; multidisciplinary neuropsychology assessment.

ADDITIONAL AEDs IN TSC2:
  - ACTH (tetracosactide): alternative first-line IS (less effective than VGB
    in TSC-specific data; use if VGB contraindicated or failed).
  - LEV: broad-spectrum adjunct; TSC-compatible; no mTOR interaction.
  - VPA: broad-spectrum; POLG MANDATORY before starting.
  - CBZ/OXC: focal seizures — CAUTION: CYP3A4 induction ↓ everolimus trough.
    If used, recheck everolimus trough q2w; escalate everolimus dose.
  - CLB: adjunct for tonic/atonic (LGS phenotype); N-CLB monitoring.
  - KD: Level B; drug-resistant TSC2 epilepsy; mTOR suppression via AMPK
    activation may act synergistically with everolimus.
  - Rapamycin (sirolimus): alternative mTOR inhibitor; limited TSC seizure
    data vs everolimus; used in LAM + some off-label TSC2 cases.

ABSOLUTE CONTRAINDICATIONS IN TSC2:
  1. Tiagabine — ABSOLUTE CI: GAT-1 inhibition → tonic GABA-A activation →
     non-convulsive status epilepticus. Class effect in ALL TSC patients.
  2. CBZ/OXC/PHT without everolimus trough monitoring — HIGH RISK: CYP3A4
     induction reduces everolimus to subtherapeutic → tuber mTOR uncontrolled.
  3. VPA without POLG screen — HIGH RISK: Alpers-Huttenlocher hepatic failure
     (POLG biallelic variants). POLG MANDATORY before any VPA.
  4. Everolimus + live vaccines — HIGH RISK: immune suppression; avoid
     all live-attenuated vaccines (MMR, varicella, oral polio, BCG) during Rx.
  5. VGB without ERG monitoring — HIGH RISK: irreversible visual field defect;
     SHARE REMS mandatory in USA; stop if significant ERG deterioration.

POLG MANDATORY:
  POLG biallelic variants + VPA → Alpers-Huttenlocher (hepatic failure,
  progressive neurodegeneration). TSC2 patients requiring VPA: POLG full
  gene sequencing BEFORE starting VPA.

HLA-B*15:02 (if CBZ/OXC considered):
  Mandatory in Han Chinese, Vietnamese, Thai, and South/Southeast Asian
  patients (CPIC guideline 2023). Positive → ABSOLUTE CI for CBZ/OXC
  (Stevens-Johnson syndrome / toxic epidermal necrolysis risk).

MHRA VALPROATE PREGNANCY PREVENTION PROGRAMME (VPPP):
  VPA in women/girls of childbearing age: mandatory pregnancy prevention
  programme in UK. Annual benefit-risk review. Sodium valproate card.

SUDEP RISK IN TSC2:
  Standardised mortality ratio (SMR) in TSC2 epilepsy: ~5–8× general population.
  SUDEP risk higher with uncontrolled nocturnal FBTCS. Supervision, nocturnal
  monitoring. Everolimus seizure reduction → SUDEP risk reduction.

KEY STANDARDS:
  ILAE 2022 · NICE NG217 (Epilepsy in Adults) · Northrup 2013 (Updated 2021)
  TSC Alliance Recommendations · EXIST-3 (Bissler 2017) · EXIST-1 (Franz 2013)
  EXIST-2 (Bissler 2013) · MILES-LAM (Bissler 2011) · UKISS (O'Callaghan 2017)
  EPISTOP (Kotulska 2021) · CPIC HLA-B*15:02 2023 · MHRA VPPP · SHARE REMS
  ACMG-AMP (pathogenic variant classification)
"""
import random

# ── Etiology catalog ─────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "TSC2-de-novo-truncating (severe)",
        "category": "TSC2-de-novo",
        "pct": 45,
        "n": 18,
        "mechanism": (
            "De novo truncating variant (frameshift / nonsense / large deletion) in TSC2 "
            "(16p13.3). Complete Tuberin loss; no residual GAP activity for Rheb. NMD-susceptible "
            "transcripts eliminated. Maximum mTORC1 hyperactivation. Highest tuber count (10–14), "
            "ID 65%, IS prevalence 50%. Most severe TSC2 subtype. ~80% of de-novo sporadic TSC2."
        ),
        "eeg_correlate": (
            "Hypsarrhythmia in infancy (modified or classic). Multi-focal sharp-slow complexes. "
            "Persistent interictal epileptiform discharges from multiple tuber sites. "
            "NREM sleep ESES in ~25%."
        ),
        "key_treatments": ["VGB first-line IS", "Everolimus EXIST-3", "ACTH if VGB fails", "SEEG if drug-resistant"],
        "clinical_tip": (
            "Screen EEG from 1 month: presymptomatic VGB before clinical spasms (EPISTOP) "
            "reduces epilepsy onset rate 50% + preserves neurodevelopment. "
            "Order POLG before any VPA. CYP3A4 interaction protocol for everolimus."
        ),
    },
    {
        "etiology": "TSC2-AD-familial (missense, milder)",
        "category": "TSC2-familial",
        "pct": 20,
        "n": 8,
        "mechanism": (
            "Inherited autosomal dominant missense variant in TSC2 GAP or N-terminal domain. "
            "Partial Tuberin function retained (some Rheb-GAP activity residual). "
            "Attenuated tuber burden (5–8 tubers), ID ~30–40%, lower LAM penetrance. "
            "~20% of TSC2 cases. Familial segregation; parent often milder or mosaic."
        ),
        "eeg_correlate": (
            "Focal interictal epileptiform discharges corresponding to tuber location. "
            "Hypsarrhythmia less common (~25% vs 50% in truncating). "
            "Normal background between seizures in milder cases."
        ),
        "key_treatments": ["Everolimus", "LEV adjunct", "Consider surgical evaluation"],
        "clinical_tip": (
            "Family genetic counselling: 50% transmission risk. "
            "Milder parental phenotype does not exclude severe disease in offspring "
            "(somatic mosaicism or new allele on de-novo background)."
        ),
    },
    {
        "etiology": "TSC2-mosaic (somatic / low-level)",
        "category": "TSC2-mosaic",
        "pct": 15,
        "n": 6,
        "mechanism": (
            "Post-zygotic somatic TSC2 variant present in a subset of cells. "
            "Allele fraction typically 5–40%. Phenotype variable: can range from "
            "isolated renal AML + minor skin lesions to full TSC. Epilepsy in 50–70% of "
            "mosaic TSC2. Standard Sanger sequencing may miss low-level mosaics. "
            "Requires ultra-deep NGS (≥500×) or parallel tissue sequencing."
        ),
        "eeg_correlate": (
            "Focal epileptiform discharges; often fewer and less severe than constitutional TSC2. "
            "Hypsarrhythmia rare. EEG-normal between seizures in low-level mosaics."
        ),
        "key_treatments": ["Everolimus if ≥2 seizures", "LEV", "Surgical if single tuber"],
        "clinical_tip": (
            "If clinically TSC but standard sequencing negative: request ultra-deep NGS "
            "≥500× of blood + skin/buccal. Check TSC2+PKD1 deletion MLPA/array-CGH "
            "if renal cysts in infancy."
        ),
    },
    {
        "etiology": "TSC2+PKD1 contiguous deletion",
        "category": "TSC2-PKD1-deletion",
        "pct": 5,
        "n": 2,
        "mechanism": (
            "Large genomic deletion at 16p13.3 removing both TSC2 and the adjacent PKD1 "
            "(autosomal dominant polycystic kidney disease type 1) genes. "
            "Results in: full-severity TSC2 epilepsy + infantile-onset polycystic kidneys "
            "+ early hypertension. Estimated ~2% of TSC2 cases. Detect by MLPA or "
            "chromosomal microarray. Renal replacement therapy risk by age 10–20 in large deletions."
        ),
        "eeg_correlate": (
            "Severe hypsarrhythmia; multi-focal sharp-slow. Neonatal electrographic seizures possible. "
            "High IS prevalence in first 6 months."
        ),
        "key_treatments": ["VGB urgent (IS)", "Everolimus", "Nephrology co-management", "BP monitoring"],
        "clinical_tip": (
            "If neonatal TSC + bilateral renal cysts: order MLPA/array-CGH for TSC2+PKD1 "
            "contiguous deletion. Nephrology from birth. Blood pressure q3M from infancy. "
            "Anticipate earlier renal replacement therapy need."
        ),
    },
    {
        "etiology": "TSC2-missense-attenuated (GAP-partial)",
        "category": "TSC2-attenuated",
        "pct": 15,
        "n": 6,
        "mechanism": (
            "TSC2 missense variant retaining partial Rheb-GAP activity. mTORC1 upregulation "
            "is incomplete (vs complete loss in truncating). Fewer/smaller cortical tubers (3–6). "
            "Epilepsy onset later (12–24 months). ID ~20%. May phenotypically overlap TSC1. "
            "Biochemical functional assay needed to classify as pathogenic vs VUS. "
            "Often de novo but some inherited with reduced penetrance."
        ),
        "eeg_correlate": (
            "Focal interictal epileptiform discharges; normal background common. "
            "Hypsarrhythmia absent or mild modified form. Focal-onset seizures predominate."
        ),
        "key_treatments": ["Everolimus (lower trough 3–7 ng/mL adequate)", "LEV", "Surgical evaluation"],
        "clinical_tip": (
            "Consider functional assay (Rheb-GTP pulldown or S6K1 phosphorylation) for missense VUS. "
            "Attenuated phenotype: everolimus lower trough (3–7) may suffice; "
            "escalate only if breakthrough seizures."
        ),
    },
]

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms / West Syndrome",
        "prevalence_pct": 45,
        "semiology": (
            "Sudden symmetric flexion or extension of trunk + limbs (salaam seizures). "
            "Clusters of 5–50 spasms on waking or during drowsiness. "
            "In TSC2: onset 3–8 months; often multifocal onset before clustering into spasms. "
            "Asymmetric spasms suggest dominant tuber → surgical candidate."
        ),
        "eeg": (
            "Hypsarrhythmia (classic or modified): high-voltage disorganised slow waves + "
            "multifocal sharp waves during waking. Electrodecrement (voltage attenuation) "
            "time-locked to clinical spasm. Post-spasm EEG burst-suppression or diffuse flattening."
        ),
        "clinical_tip": (
            "Screen all TSC2 neonates from 1 month with EEG (EPISTOP). "
            "Pre-symptomatic VGB at EEG hypsarrhythmia (before clinical spasms) → "
            "50% reduction in spasm onset + better neurodevelopment (Kotulska 2021). "
            "VGB 95% spasm cessation at 1M in TSC-IS (UKISS). "
            "SHARE REMS enrolment + baseline ERG before starting VGB."
        ),
    },
    {
        "type": "Focal Aware Seizures (FAS)",
        "prevalence_pct": 60,
        "semiology": (
            "Stereotyped focal motor/sensory/autonomic/psychic events without impaired awareness. "
            "Content reflects tuber location: frontal → clonic limb jerks, versive; "
            "temporal → déjà vu, fear, epigastric aura; occipital → visual phenomena. "
            "Duration 30–120 s. Often precede secondary generalization in TSC2."
        ),
        "eeg": (
            "Unilateral rhythmic theta (4–7 Hz) or sharp-slow complexes at tuber electrode. "
            "Interictal: focal spikes/sharp waves at one or more tuber sites. "
            "High-frequency oscillations (HFOs, 80–500 Hz) at epileptogenic tuber "
            "on invasive recording (SEEG ripples 80–250 Hz)."
        ),
        "clinical_tip": (
            "Multiple tubers → multiple semiologies possible. SEEG/AMT-PET needed to "
            "distinguish primary epileptogenic tuber from propagation. "
            "Everolimus reduces FAS frequency 40% in EXIST-3 responders."
        ),
    },
    {
        "type": "Focal to Bilateral Tonic-Clonic (FBTCS)",
        "prevalence_pct": 70,
        "semiology": (
            "Focal onset (may be clinically occult) → bilateral tonic then clonic. "
            "TSC2 FBTCS: high frequency (4–8/month in drug-resistant); nocturnal predominance. "
            "SUDEP risk associated with uncontrolled nocturnal FBTCS. "
            "Duration 1–3 min; postictal confusion 15–30 min."
        ),
        "eeg": (
            "Ictal: focal evolving discharge → bilateral fast activity (18–25 Hz) → "
            "bilateral rhythmic clonic discharges → postictal generalised EEG suppression (PGES). "
            "PGES >50 s correlated with SUDEP risk (Ryvlin 2013)."
        ),
        "clinical_tip": (
            "Nocturnal supervision + pulse oximetry. Rescue benzodiazepine plan for clusters. "
            "Everolimus + optimal AED combination targets FBTCS reduction. "
            "SUDEP annual counselling; seizure diary tracking nocturnal events."
        ),
    },
    {
        "type": "Tonic Seizures / LGS Phenotype",
        "prevalence_pct": 30,
        "semiology": (
            "Sudden brief axial or limb tonic stiffening. Nocturnal predominance. "
            "Head drop, fall risk. In TSC2: LGS pattern with tonic + atonic + "
            "atypical absence — tuber-driven multi-zone epilepsy. "
            "~30% TSC2 patients with drug-resistant epilepsy develop LGS pattern."
        ),
        "eeg": (
            "Ictal tonic: paroxysmal fast activity (10–20 Hz) or recruiting rhythm. "
            "Interictal: slow spike-wave (1.5–2.5 Hz) diffusely. "
            "Sleep: runs of rapid spikes. Background slowing common in severe TSC2."
        ),
        "clinical_tip": (
            "CLB adjunct for LGS-pattern tonic seizures. "
            "KD in drug-resistant TSC2 LGS — mTOR synergy. "
            "VNS consideration in severe multifocal LGS when not surgical candidate."
        ),
    },
    {
        "type": "Reappearing Spasms (Late-Onset)",
        "prevalence_pct": 20,
        "semiology": (
            "Re-emergence of infantile-spasm-like events in toddler/school age after initial "
            "spasm remission. TSC2-specific: ~20% experience late spasms (12 months – 5 years). "
            "May indicate new tuber epileptogenicity or mTOR-driven second wave. "
            "Distinguished from myoclonic by cluster + decrement pattern."
        ),
        "eeg": (
            "Recurrent electrodecrement pattern on video-EEG. May not be classic hypsarrhythmia "
            "(modified pattern). Multi-focal sharp-slow from tubers."
        ),
        "clinical_tip": (
            "TSC2 late spasms: reassess everolimus trough (may need escalation). "
            "Repeat AMT-PET or SEEG if late spasms suggest new epileptogenic tuber. "
            "Consider ACTH course if everolimus insufficient for late spasms."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "prevalence_pct": 20,
        "semiology": (
            "Sudden bilateral or asymmetric myoclonic jerks (arms > legs). "
            "Often on waking; clusters of 3–10 jerks. "
            "In TSC2: may indicate progressive epileptic encephalopathy or "
            "superimposed photoparoxysmal response (20% TSC)."
        ),
        "eeg": (
            "Generalised polyspike or polyspike-wave synchronous with jerk. "
            "Photoparoxysmal response in ~20%. Background disorganised in severe TSC2."
        ),
        "clinical_tip": (
            "Avoid sodium channel blockers (LTG exacerbates myoclonus in some TSC). "
            "LEV or VPA (POLG-screened) for myoclonic control. "
            "Photosensitivity: avoid stroboscopic lights; tinted glasses."
        ),
    },
]

# ── Triggers ─────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever ≥38.0°C",
        "prevalence_pct": 82,
        "mechanism": (
            "Hyperthermia increases neuronal firing rates; in TSC2 tubers, reduced inhibitory "
            "tone (interneuron loss) and upregulated mTOR-driven sodium channels lower the "
            "seizure threshold ~38.0°C. Even low-grade fever triggers in severe TSC2."
        ),
        "management": "Antipyretics (paracetamol/ibuprofen) at 37.5°C. Rescue BZD for clusters ≥3 spasms.",
    },
    {
        "trigger": "Missed Everolimus Dose",
        "prevalence_pct": 75,
        "mechanism": (
            "Everolimus half-life ~30h; trough falls below therapeutic (3–7 ng/mL) within "
            "48–72h of missed dose. mTORC1 rebounds acutely → tuber hyperexcitability increases. "
            "Rebound seizure cluster risk highest 2–3 days post-miss."
        ),
        "management": "Smart-pill bottle + caregiver reminders. Skip missed dose if >12h; do not double next dose.",
    },
    {
        "trigger": "Sleep Deprivation",
        "prevalence_pct": 68,
        "mechanism": (
            "Sleep deprivation reduces seizure threshold; TSC2 tubers lose nocturnal GABA "
            "inhibitory surge → breakthrough seizures. NREM sleep fragmentation in TSC2 "
            "(SEGA-related if raised ICP) further compounds threshold reduction."
        ),
        "management": "Consistent sleep schedule. Evaluate SEGA if new sleep-onset insomnia (annual MRI).",
    },
    {
        "trigger": "Intercurrent Illness",
        "prevalence_pct": 65,
        "mechanism": (
            "Systemic illness: fever + metabolic stress + inflammatory cytokines → microglial "
            "activation in tubers → local glutamate excess → seizure threshold falls. "
            "Concurrent GI illness may reduce AED/everolimus absorption."
        ),
        "management": "Ensure AED/everolimus administration with adequate fluid. IV/IM rescue if vomiting.",
    },
    {
        "trigger": "CYP3A4-Inducing AEDs (CBZ/PHT/PB/OXC/Rifampicin)",
        "prevalence_pct": 50,
        "mechanism": (
            "CYP3A4 inducers increase everolimus metabolism 2–4×, dropping trough to "
            "subtherapeutic (<3 ng/mL). mTORC1 uncontrolled → seizure cluster risk within "
            "7–14 days of CYP3A4 inducer addition. Most underrecognised TSC2 trigger."
        ),
        "management": (
            "Recheck everolimus trough at 2 weeks of any CYP3A4 change. "
            "Escalate everolimus dose to maintain 5–15 ng/mL. "
            "Prefer CYP3A4-neutral AEDs (LEV, CLB, LTG) in TSC2."
        ),
    },
    {
        "trigger": "Emotional Stress",
        "prevalence_pct": 58,
        "mechanism": (
            "Hypothalamic-pituitary-adrenal axis activation → cortisol release → "
            "altered GABA-A receptor trafficking → tuber excitability. "
            "TAND-related anxiety/ASD in TSC2 makes stress-related triggering more common."
        ),
        "management": "Psychology/TAND support. Mindfulness adapted for developmental level. CLB PRN anxiety-cluster.",
    },
    {
        "trigger": "Photosensitivity",
        "prevalence_pct": 18,
        "mechanism": (
            "Photoparoxysmal response in ~18% TSC patients (more than general population). "
            "Occipital tubers may sensitise visual cortex. Flickering screens, strobe lights trigger."
        ),
        "management": "Tinted glasses (FL-41). Avoid strobe lighting. Screen time with breaks.",
    },
    {
        "trigger": "SEGA / Raised Intracranial Pressure",
        "prevalence_pct": 15,
        "mechanism": (
            "Subependymal giant cell astrocytoma at foramen of Monro → obstructive hydrocephalus "
            "→ raised ICP → cerebral perfusion compromise → seizure threshold falls acutely. "
            "New headache/vomiting/papilloedema in TSC2 = SEGA emergency."
        ),
        "management": "Annual brain MRI for SEGA. Everolimus shrinks SEGA (EXIST-1 35% response ≥50% reduction). Neurosurgery if symptomatic hydrocephalus.",
    },
]

# ── Treatments ───────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A",
        "indication": "Infantile spasms in TSC — FIRST LINE (Level A; UKISS 2017; EPISTOP 2021)",
        "dose": (
            "Infantile spasms: Start 50 mg/kg/day divided BD; escalate to 100–150 mg/kg/day "
            "if partial response (max 150 mg/kg/day). "
            "TSC2 may require higher end of range. Adjust renally."
        ),
        "moa": (
            "Irreversible GABA-T (γ-aminobutyric acid transaminase) inhibitor → "
            "reduced GABA catabolism → increased synaptic GABA → reduced tuber excitability."
        ),
        "efficacy": (
            "TSC-specific IS response: 95% spasm cessation at 1 month (UKISS). "
            "EPISTOP (Kotulska 2021): presymptomatic VGB → 50% reduction in IS onset rate "
            "vs reactive VGB after clinical spasms. Better neurodevelopment at 24 months."
        ),
        "safety": (
            "Irreversible visual field constriction: 30–50% adults (less characterised in infants). "
            "ERG baseline before starting; q3M in infants, q6M in children/adults. "
            "SHARE REMS mandatory (USA). Weight gain, sedation, irritability."
        ),
        "monitoring": "ERG q3M (infants), q6M (children/adults). Visual field test when developmentally able.",
        "tsc2_note": (
            "TSC2 IS prevalence 45% — higher than TSC1 (35%). "
            "Higher VGB dose (100–150 mg/kg/day) often needed vs TSC1. "
            "Start EEG monitoring from 1 month; initiate VGB at hypsarrhythmia "
            "(before clinical spasms) per EPISTOP protocol."
        ),
    },
    {
        "drug": "Everolimus (Afinitor / mTORC1 inhibitor)",
        "level": "Level A",
        "indication": "TSC-associated focal-onset seizures ≥2 years (FDA 2017, EXIST-3)",
        "dose": (
            "Paediatric: 4.5 mg/m²/day once daily (BSA-based). "
            "Adult: 10 mg/day once daily. "
            "Adjust to trough 3–7 ng/mL (lower band) or 9–15 ng/mL (higher band). "
            "Recheck trough at 2 weeks, monthly until stable, then q3–6M."
        ),
        "moa": (
            "Rapamycin analogue; binds FKBP12 → FKBP12-everolimus complex inhibits mTORC1 "
            "allosterically → reduces S6K1 + 4E-BP1 phosphorylation → "
            "normalises protein synthesis in tuber neurons + reduces seizure frequency."
        ),
        "efficacy": (
            "EXIST-3 (Bissler 2017 Lancet Neurol, N=366): "
            "Response ≥50% seizure reduction: 40% (9–15 ng/mL) / 28% (3–7 ng/mL) vs 15% placebo. "
            "Seizure freedom: 13% vs 0% placebo. "
            "Median seizure reduction: 29% vs 0% placebo. "
            "Additional benefit in TSC2 truncating variants (higher mTOR activation = more room for suppression)."
        ),
        "safety": (
            "Stomatitis 70–80%, infections 40%, hyperlipidaemia 70%, hyperglycaemia 50%, "
            "ILD rare. Immunosuppression: avoid live vaccines. "
            "Wound healing impairment: hold 1 week perioperatively."
        ),
        "monitoring": (
            "Everolimus trough q2w until stable → q3–6M. CBC, LFTs, "
            "LDL/TG, HbA1c, creatinine q3M. Annual brain MRI (SEGA). "
            "Renal US q1y (AML). Chest HRCT annual in TSC2 females ≥15y (LAM). "
            "CYP3A4 interaction: recheck trough 2 weeks after any AED change."
        ),
        "tsc2_note": (
            "TSC2 complete Tuberin loss → maximal mTOR activation. "
            "Higher trough (9–15 ng/mL) more effective in drug-resistant TSC2. "
            "CYP3A4 drug interaction CRITICAL: CBZ/PHT reduce trough 2–4×. "
            "Prefer CYP3A4-neutral AEDs (LEV, CLB) to avoid everolimus trough fluctuations."
        ),
    },
    {
        "drug": "ACTH (Tetracosactide / Synacthen)",
        "level": "Level B",
        "indication": "Alternative/adjunct for TSC2 infantile spasms if VGB contraindicated or inadequate",
        "dose": (
            "Synacthen depot: 0.5–0.75 mg alternate days IM (UK schedule). "
            "Natural ACTH: 150 IU/m²/day divided twice daily IM × 2 weeks, taper. "
            "Short synthetic ACTH: varies by protocol."
        ),
        "moa": (
            "ACTH → adrenal cortisol + neurosteroid release → enhanced GABA-A activity "
            "→ reduced CRH-driven limbic hyperexcitability. Secondary: myelination support."
        ),
        "efficacy": (
            "Non-TSC IS: 55–70% spasm cessation (UK INFANTILE SPASMS trial). "
            "TSC-specific: less effective than VGB (~60% vs 95% in TSC). "
            "Use as second-line after VGB failure or if VGB contraindicated (ERG concern)."
        ),
        "safety": (
            "Irritability, hypertension, infection risk (immunosuppression), "
            "cerebral atrophy (prolonged), electrolyte disturbance, "
            "hyperglycaemia. Short courses preferred."
        ),
        "monitoring": "BP q3–7 days during treatment. Electrolytes, glucose. Infection surveillance.",
        "tsc2_note": (
            "TSC2 IS: VGB strongly preferred (95% vs ~60% for ACTH). "
            "ACTH reserved for VGB failure / ERG contraindication. "
            "Combined VGB + ACTH (UKISS protocol): some benefit in non-responders."
        ),
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "Broad-spectrum adjunct in TSC2 focal and generalised seizures",
        "dose": "20–60 mg/kg/day divided BD (paediatric); 1000–3000 mg/day divided BD (adult).",
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) ligand → modulates neurotransmitter "
            "vesicle release + inhibits voltage-gated calcium channels → broad anticonvulsant effect."
        ),
        "efficacy": "Level B adjunct; no TSC-specific RCT. Reduces FBTCS and focal seizures ~30–40%.",
        "safety": "Irritability, behavioural side-effects (TAND patients may be more sensitive). Dose-related.",
        "monitoring": "Behavioural review q3M. Trough levels rarely needed (linear PK).",
        "tsc2_note": (
            "CYP3A4-neutral — ideal partner for everolimus (no trough interaction). "
            "Monitor TAND/ASD-related behavioural side-effects; consider dose reduction if marked irritability."
        ),
    },
    {
        "drug": "Valproate (VPA)",
        "level": "Level B",
        "indication": "Broad-spectrum; TSC2 focal + generalised (POLG screen mandatory)",
        "dose": "20–40 mg/kg/day divided BD–TID; adult 500–2000 mg/day. Target trough 50–100 mg/L.",
        "moa": (
            "Multiple: sodium channel stabilisation, GABA-T inhibition "
            "(increased GABA), T-type calcium channel blockade, HDAC inhibition."
        ),
        "efficacy": "Broad-spectrum Level B adjunct in TSC2; useful for myoclonic + generalised components.",
        "safety": (
            "POLG Alpers risk (MANDATORY screen before use). "
            "Hepatotoxicity (LFT + ammonia monitoring). "
            "VPPP mandatory in UK (pregnancy prevention). "
            "Teratogenicity — avoid in pregnancy unless no alternative."
        ),
        "monitoring": "POLG genetic test BEFORE starting. LFT, ammonia, VPA trough q3M. VPPP in females.",
        "tsc2_note": (
            "CYP3A4-neutral: safe with everolimus trough. "
            "POLG mandatory — cannot start VPA without POLG result. "
            "Useful in TSC2 myoclonic seizures. "
            "VPPP mandatory for females of childbearing age."
        ),
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Adjunct for TSC2 LGS-pattern tonic/atonic/atypical absence",
        "dose": (
            "0.1–0.3 mg/kg/day divided BD–TID (paediatric); "
            "10–40 mg/day divided BD (adult). Titrate slowly."
        ),
        "moa": (
            "Positive allosteric modulator at GABA-A α2 subunit-containing receptors "
            "→ enhanced Cl⁻ conductance → neuronal inhibition. Less sedation vs other BZD "
            "(α2 selectivity vs α1). N-CLB active metabolite."
        ),
        "efficacy": "LGS adjunct evidence Level B (Sankar 2013 review). Useful in tonic + drop attacks.",
        "safety": "Sedation (less than clonazepam), tolerance (up to 30%), N-CLB accumulation (renal).",
        "monitoring": "N-CLB levels if renal impairment or excess sedation. Tolerance review q6M.",
        "tsc2_note": (
            "TSC2 LGS phenotype (~30% of drug-resistant): CLB adjunct targets tonic/atonic. "
            "CYP3A4 minor substrate — minimal everolimus interaction."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Drug-resistant TSC2 epilepsy (≥2 AED failures including everolimus attempt)",
        "dose": "Classic KD 4:1 ratio or modified Atkins diet. Metabolic dietitian-supervised.",
        "moa": (
            "Ketone bodies (β-hydroxybutyrate, acetoacetate) activate AMPK, "
            "which phosphorylates TSC2 (Tuberin) at Ser1345/Thr1271 activating the "
            "TSC1/TSC2 complex → mTORC1 suppression. Dual mechanism in TSC2: "
            "direct AMPK-mTOR suppression + GABAergic + adenosine-mediated effects."
        ),
        "efficacy": (
            "Paediatric drug-resistant epilepsy: 50% seizure reduction in ~50% at 3 months. "
            "TSC2 preclinical: KD + rapamycin (mTOR inhibitor) synergistic (Koh 2009). "
            "Clinical series suggest enhanced response in TSC vs non-TSC."
        ),
        "safety": "Hyperlipidaemia, renal calculi, growth impairment (children), constipation. GI tolerance.",
        "monitoring": "Ketone levels daily. Lipid panel q3M. Renal US annually (calculi). Dietitian q3M.",
        "tsc2_note": (
            "AMPK → TSC2(Ser1345) activation is the molecular synergy mechanism "
            "(KD activates the very complex that TSC2 LOF has disabled — partial complementation). "
            "Combine KD + everolimus in refractory TSC2: evidence from animal models + case series. "
            "Monitor everolimus trough on KD (metabolic changes may alter absorption)."
        ),
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Tiagabine",
        "level": "ABSOLUTE CI",
        "reason": (
            "GAT-1 (GABA reuptake transporter) inhibition → marked extracellular GABA accumulation "
            "→ persistent tonic activation of extrasynaptic GABA-A receptors (α5, δ subunits) → "
            "non-convulsive status epilepticus (NCSE). Class effect in ALL TSC patients, "
            "all epilepsy genotypes. Risk of prolonged refractory NCSE requiring ICU."
        ),
        "alternative": "CLB (GABA-A modulator via α2 — not GABA reuptake inhibition). LEV. VPA.",
    },
    {
        "drug": "CBZ / OXC / PHT / PB (without everolimus trough monitoring)",
        "level": "HIGH RISK",
        "reason": (
            "Strong CYP3A4 inducers reduce everolimus plasma trough 2–4× within 7–14 days. "
            "Subtherapeutic everolimus → mTORC1 uncontrolled in tubers → seizure cluster risk. "
            "TSC2 complete Tuberin loss = fully dependent on pharmacological mTOR suppression. "
            "Adding CBZ without monitoring everolimus may be life-threatening in TSC2."
        ),
        "alternative": "LEV, CLB, LTG (CYP3A4-neutral). If CBZ/PHT essential: escalate everolimus dose + recheck trough at 2 weeks.",
    },
    {
        "drug": "Valproate (without POLG screen)",
        "level": "HIGH RISK",
        "reason": (
            "POLG biallelic pathogenic variants + VPA → Alpers-Huttenlocher syndrome: "
            "fulminant hepatic failure + progressive neurodegeneration. "
            "Irreversible. POLG full gene sequencing MANDATORY before starting VPA. "
            "Not TSC2-specific but mandatory in all VPA prescriptions."
        ),
        "alternative": "Order POLG; if POLG positive → use LEV or CLB instead of VPA.",
    },
    {
        "drug": "Live vaccines during everolimus therapy",
        "level": "HIGH RISK",
        "reason": (
            "Everolimus causes immunosuppression (mTORC1 inhibition → reduced T-cell proliferation). "
            "Live-attenuated vaccines (MMR, varicella/VZV, oral polio, BCG, rotavirus, yellow fever) → "
            "disseminated vaccine-strain infection risk. Pause everolimus 2 weeks before "
            "live vaccine if essential; restart after 4 weeks."
        ),
        "alternative": "Inactivated vaccines safe on everolimus. Pre-treatment vaccination preferred.",
    },
    {
        "drug": "Vigabatrin (without ERG monitoring / SHARE REMS)",
        "level": "HIGH RISK",
        "reason": (
            "VGB causes irreversible visual field constriction (30–50% adults) via retinal GABA "
            "accumulation and rod photoreceptor toxicity. Progressive; occurs without symptoms. "
            "SHARE REMS (USA) mandatory enrolment; ERG baseline + q3M infants, q6M children/adults. "
            "UK: annual ophthalmological review mandatory."
        ),
        "alternative": "Continue VGB with mandatory ERG. Discontinue if ERG deteriorates significantly.",
    },
]

# ── Monitoring ───────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG genetic screen (MANDATORY before VPA)", "freq": "Once before VPA start; panel testing"},
    {"item": "Everolimus trough level", "freq": "q2w until stable, then q3–6M. Recheck 2w after any CYP3A4 change"},
    {"item": "ERG (visual electrophysiology) — VGB mandatory", "freq": "Baseline; q3M infants; q6M children/adults. SHARE REMS enrolment"},
    {"item": "EEG surveillance (presymptomatic TSC2)", "freq": "Monthly 1–12 months; q3M up to 2y (EPISTOP protocol)"},
    {"item": "Brain MRI — SEGA + tuber burden", "freq": "Annual; every 6M if SEGA growing or symptomatic"},
    {"item": "Renal ultrasound — AML + cysts (TSC2+PKD1)", "freq": "Annual; q6M if TSC2+PKD1 deletion or AML ≥3cm"},
    {"item": "Chest HRCT — pulmonary LAM (TSC2 females)", "freq": "Baseline at ≥15y; annual if LAM present. Pulmonary function tests (spirometry/DLCO)"},
    {"item": "LFT + ammonia (VPA)", "freq": "Baseline, q4–6w first 6M, then q6M"},
    {"item": "LDL/TG (everolimus)", "freq": "Baseline, q3M"},
    {"item": "HbA1c / fasting glucose (everolimus)", "freq": "Baseline, q3M"},
    {"item": "TAND checklist (ASD/ID/ADHD/anxiety)", "freq": "Annual; formal neuropsychology q2y"},
    {"item": "Blood pressure + renal function (eGFR)", "freq": "Annual from age 3y; q6M if TSC2+PKD1 or AML embolisation"},
    {"item": "HLA-B*15:02 (if CBZ/OXC planned)", "freq": "Once, before CBZ/OXC start in at-risk ethnicity"},
    {"item": "Seizure diary (nocturnal FBTCS / SUDEP surveillance)", "freq": "Continuous; review q3M with clinic"},
]

# ── Lifecycle ─────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Prenatal / Perinatal",
        "age": "18–22 weeks gestation – birth",
        "features": "Subependymal tubers visible on fetal MRI. Cardiac rhabdomyoma (60% TSC2 at birth) may be detected on fetal echo. Amniotic fluid DNA or CVS for confirmatory TSC2 testing if family history.",
        "actions": "Fetal MRI + echo. Postnatal genetics referral. NICU alert for cardiac rhabdomyoma arrhythmia.",
    },
    {
        "window": "Neonatal / Early Infancy",
        "age": "0–3 months",
        "features": "Cardiac rhabdomyomas (benign, often regress). Hypopigmented skin macules (Wood's lamp). EEG surveillance initiation. TSC2+PKD1: renal cysts + hypertension from birth.",
        "actions": "EEG from 1 month (EPISTOP). Renal US + BP if TSC2 confirmed. Cardiac echo if rhabdomyoma. Genetic counselling for parents.",
    },
    {
        "window": "Infantile (Epilepsy Onset)",
        "age": "3–18 months",
        "features": "IS peak onset 4–7 months (TSC2). VGB initiation (presymptomatic or at EEG hypsarrhythmia). Tuber epileptogenicity evolves. AML small; SEGA nodules forming.",
        "actions": "VGB (SHARE REMS + ERG). Everolimus if IS fails or focal seizures. AMT-PET/MRI tuber mapping. POLG before VPA. ACTH if VGB contraindicated.",
    },
    {
        "window": "Toddler / Preschool",
        "age": "18 months – 5 years",
        "features": "Focal seizures replace IS. Drug-resistant epilepsy consolidates. TAND features emerge (ASD, ID). Cortical tubers may increase seizure burden. SEEG evaluation if surgical candidate.",
        "actions": "Everolimus optimisation. SEEG/AMT-PET. Surgical evaluation (single epileptogenic tuber). Multidisciplinary TAND assessment. Annual MRI + renal US.",
    },
    {
        "window": "School Age",
        "age": "5–12 years",
        "features": "Seizure pattern stabilises or worsens. LGS phenotype in ~30% drug-resistant. Neurodevelopmental impacts. School support (IEP). SEGA growth period. KD consideration.",
        "actions": "Annual SEGA surveillance (MRI). Everolimus trough optimisation. KD if ≥2 AEDs failed. CLB for LGS-tonic. Annual ophthalmic (VGB), annual renal US. IEP/school psychology.",
    },
    {
        "window": "Adolescent / Adult",
        "age": "12+ years",
        "features": "LAM onset (TSC2 females ≥15y). AML growth. Seizure-independence aspiration. VPPP for VPA in females. SUDEP risk management. Transition to adult neurology.",
        "actions": "Chest HRCT + PFTs for LAM. AML everolimus or embolisation. VPPP. Annual SEGA MRI (until age 25 if stable). SUDEP counselling. Transition clinic planning. Driving assessment.",
    },
]

# ── Key concepts ──────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "TSC2 (Tuberin) — 16p13.3", "definition": "Gene encoding Tuberin (200 kDa), the GAP-domain partner of Hamartin (TSC1). TSC2 LOF is more severe (70% TSC; more tubers; higher ID/ASD/drug-resistance rates)."},
    {"term": "mTORC1 / Rheb / mTOR pathway", "definition": "TSC2 GAP domain keeps Rheb-GDP (inactive). LOF → Rheb-GTP → mTORC1 constitutively active → S6K1/4E-BP1 hyperphosphorylation → dysregulated growth, migration, excitability."},
    {"term": "Cortical Tuber (Type A/B/C)", "definition": "Focal disorganised epileptogenic cortex. Type C (cystic, FLAIR-bright) most epileptogenic (80%). TSC2: mean 9–12 tubers (vs 6–8 TSC1). Tuber burden correlates with epilepsy severity and ID."},
    {"term": "SEGA (Subependymal Giant Cell Astrocytoma)", "definition": "Benign WHO grade I tumour at foramen of Monro. ~10–15% TSC. Obstructive hydrocephalus risk. Everolimus first-line (EXIST-1: 35% ≥50% reduction). Annual MRI surveillance. Neurosurgery if symptomatic hydrocephalus."},
    {"term": "Everolimus (Afinitor)", "definition": "mTORC1 inhibitor (rapamycin analogue). FDA-approved TSC seizures (EXIST-3), SEGA (EXIST-1), AML (EXIST-2), LAM (MILES). Trough 3–7 ng/mL (lower) or 9–15 ng/mL (higher) for seizures. CYP3A4 interactions critical."},
    {"term": "Vigabatrin (VGB) / SHARE REMS", "definition": "GABA-T inhibitor. Level A first-line IS in TSC2 (95% response, UKISS). Retinal toxicity (VF constriction 30–50%): mandatory ERG + SHARE REMS enrolment (USA). EPISTOP: presymptomatic VGB before clinical spasms."},
    {"term": "TAND (TSC-Associated Neuropsychiatric Disorders)", "definition": "ASD 35–45%, ID 50–60%, ADHD 40–50%, anxiety 35% in TSC2. Annual TAND checklist + neuropsychology q2y. Multidisciplinary support essential."},
    {"term": "AMT-PET", "definition": "Alpha-[11C]-methyl-L-tryptophan PET: tryptophan trapping in epileptogenic tubers. Identifies primary seizure-onset tuber among multiple candidates. Guides surgical target in multifocal TSC2 epilepsy (Juhász 2006)."},
    {"term": "EPISTOP / PREVENT Trial", "definition": "EPISTOP (Kotulska 2021, Lancet Neurology): presymptomatic VGB (at EEG abnormality before clinical IS) → 50% reduction in epilepsy onset + better 24M neurodevelopment. Evidence base for newborn EEG surveillance in TSC2 from 1 month."},
    {"term": "TSC2 Mosaicism", "definition": "Post-zygotic somatic TSC2 variant (5–40% allele fraction). Phenotype variable (isolated renal AML to full TSC). Ultra-deep NGS ≥500× required; parallel tissue sequencing if blood allele fraction low."},
    {"term": "Renal AML (Angiomyolipoma)", "definition": "Benign fat-containing renal tumour. 75–80% TSC2. Annual US. AML ≥3cm → prophylactic everolimus. Embolisation if spontaneous haemorrhage (Wunderlich syndrome). Low-fat AML variant can mimic RCC."},
    {"term": "Pulmonary LAM (TSC2-Specific)", "definition": "Lymphangioleiomyomatosis: TSC2-driven smooth muscle proliferation in pulmonary parenchyma → cysts → progressive dyspnoea, pneumothorax. 25–35% TSC2 females (vs rare TSC1). Annual HRCT + PFTs ≥15y. Everolimus (MILES trial)."},
    {"term": "TSC2+PKD1 Contiguous Deletion", "definition": "Large 16p13.3 deletion removing TSC2 + PKD1. Results in severe TSC2 epilepsy + infantile polycystic kidneys + early hypertension. ~2% TSC2. Detect by MLPA/array-CGH. Renal replacement therapy risk early childhood."},
    {"term": "CYP3A4 Drug Interaction (Everolimus)", "definition": "CYP3A4 inducers (CBZ, PHT, PB, OXC, rifampicin) reduce everolimus trough 2–4×. CYP3A4 inhibitors (azoles, clarithromycin) increase trough. Recheck trough 2 weeks after ANY AED or drug change in TSC2."},
    {"term": "SUDEP Risk in TSC2", "definition": "SMR ~5–8× general population. Uncontrolled nocturnal FBTCS is primary risk factor. Nocturnal supervision, seizure alarms, pulse oximetry. Everolimus seizure reduction → SUDEP risk reduction. Annual SUDEP counselling."},
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "Fever threshold (seizure trigger)", "value": "≥38.0°C (TSC2 — lower than general 38.5°C)"},
    {"parameter": "Everolimus trough (lower band — TSC seizures)", "value": "3–7 ng/mL"},
    {"parameter": "Everolimus trough (higher band — drug-resistant)", "value": "9–15 ng/mL"},
    {"parameter": "AML size threshold (start everolimus)", "value": "≥3 cm diameter"},
    {"parameter": "SEGA size threshold (everolimus vs surgery)", "value": "≥1 cm growth / symptomatic hydrocephalus"},
    {"parameter": "VGB dose (TSC2 infantile spasms — upper)", "value": "150 mg/kg/day (max)"},
    {"parameter": "EEG surveillance start (TSC2 neonates)", "value": "From 1 month of life (EPISTOP protocol)"},
    {"parameter": "LAM surveillance start (TSC2 females)", "value": "≥15 years — baseline HRCT"},
    {"parameter": "AML embolisation threshold (bleeding)", "value": "Wunderlich syndrome / AML ≥6 cm with high bleeding risk"},
    {"parameter": "CYP3A4 everolimus trough recheck", "value": "2 weeks after ANY CYP3A4-interacting drug change"},
    {"parameter": "POLG screen timing", "value": "BEFORE starting VPA — no exceptions"},
    {"parameter": "PGES duration (SUDEP risk marker)", "value": ">50 seconds post-ictal EEG suppression (Ryvlin 2013)"},
]

# ── Standards ──────────────────────────────────────────────────────────────────
STANDARDS = [
    {"id": "ILAE-2022", "ref": "ILAE Classification and Definition of Epilepsy Syndromes 2022"},
    {"id": "NICE-NG217", "ref": "NICE NG217: Epilepsies — Diagnosis and Management (2022)"},
    {"id": "Northrup-TSC-2013-Updated", "ref": "Northrup H et al., Pediatric Neurology 2013 (Updated TSC Consensus Recommendations 2021)"},
    {"id": "EXIST-3-Bissler-2017", "ref": "Bissler JJ et al., EXIST-3: Everolimus for TSC seizures, Lancet Neurology 2017"},
    {"id": "EXIST-1-Franz-2013", "ref": "Franz DN et al., EXIST-1: Everolimus for SEGA, Lancet 2013"},
    {"id": "EXIST-2-Bissler-2013", "ref": "Bissler JJ et al., EXIST-2: Everolimus for AML, NEJM 2013"},
    {"id": "MILES-LAM-2011", "ref": "Bissler JJ et al., Sirolimus for LAM and TSC, NEJM 2008 (MILES 2011 update)"},
    {"id": "EPISTOP-Kotulska-2021", "ref": "Kotulska K et al., EPISTOP: Preventive treatment of epilepsy in TSC, Lancet Neurology 2021"},
    {"id": "UKISS-OCallaghan-2017", "ref": "O'Callaghan FJK et al., UKISS: VGB vs steroids for IS including TSC, Lancet Neurology 2017"},
    {"id": "SHARE-REMS", "ref": "SHARE REMS Program (sabril.net): VGB mandatory enrolment + ERG monitoring (USA)"},
    {"id": "CPIC-HLA-B1502-2023", "ref": "CPIC Guideline: HLA-B*15:02 and carbamazepine/oxcarbazepine (2023)"},
    {"id": "MHRA-VPPP-2021", "ref": "MHRA Valproate Pregnancy Prevention Programme (2021)"},
    {"id": "ACMG-AMP-2015", "ref": "Richards S et al., ACMG/AMP variant classification standards 2015"},
]

# ── References ─────────────────────────────────────────────────────────────────
REFERENCES = [
    "Northrup H et al. (2013/2021) — International TSC Consensus Conference: TSC management recommendations",
    "Bissler JJ et al. (2017) — EXIST-3: Everolimus for seizures associated with TSC, Lancet Neurology",
    "Kotulska K et al. (2021) — EPISTOP: Preventive antiepileptic treatment in newborns with TSC, Lancet Neurology",
    "Franz DN et al. (2013) — EXIST-1: Everolimus for SEGA in TSC, Lancet",
    "Juhász C et al. (2006) — Alpha-methyl-L-tryptophan PET to identify epileptogenic cortical tubers in TSC, Annals of Neurology",
    "Bhatt DL et al. (2023) — Epilepsy Precision Medicine and Genomics Review, Epilepsia",
]

# ── Patient generator ─────────────────────────────────────────────────────────
def _generate_patients():
    """Generate 40 synthetic TSC2 patients (deterministic seed for reproducibility)."""
    rng = random.Random(7777)
    etiology_pool = [
        ("TSC2-de-novo-truncating-severe", "truncating", True),
        ("TSC2-de-novo-truncating-severe", "truncating", True),
        ("TSC2-AD-familial-missense", "missense", False),
        ("TSC2-mosaic", "mosaic", True),
        ("TSC2-missense-attenuated", "missense", False),
    ]
    etiology_weights = [0.45, 0.45, 0.20, 0.15, 0.15]

    patients = []
    for i in range(40):
        # Weighted etiology
        r = rng.random()
        cum = 0
        etiology_idx = 0
        ew_norm = [0.45, 0.20, 0.15, 0.05, 0.15]
        for j, w in enumerate(ew_norm):
            cum += w
            if r < cum:
                etiology_idx = j
                break

        etiology_labels = [
            "TSC2-de-novo-truncating-severe",
            "TSC2-AD-familial-missense",
            "TSC2-mosaic",
            "TSC2-PKD1-contiguous-deletion",
            "TSC2-missense-attenuated",
        ]
        is_severe = etiology_idx in (0, 2, 3)

        age_y = round(rng.uniform(0.5, 42), 1)
        onset_m = rng.randint(3, 14) if is_severe else rng.randint(6, 24)
        tuber_count = rng.randint(8, 14) if is_severe else rng.randint(3, 9)
        drug_resistant = rng.random() < (0.60 if is_severe else 0.40)
        seizure_free = not drug_resistant and rng.random() < 0.30
        has_is = rng.random() < (0.50 if is_severe else 0.30)
        has_sega = rng.random() < 0.12
        aml = rng.random() < 0.78
        pkd1_deletion = etiology_idx == 3
        tand_asd = rng.random() < (0.45 if is_severe else 0.30)
        on_everolimus = rng.random() < 0.65
        on_vpa = rng.random() < 0.35
        polg_tested = "Y" if rng.random() < 0.72 else "N"
        erg_done = rng.random() < 0.78 if has_is else rng.random() < 0.45
        lam_risk = rng.random() < 0.30  # female proxy

        patients.append({
            "id": f"TSC2-{i+1:03d}",
            "etiology": etiology_labels[etiology_idx],
            "age_y": age_y,
            "onset_months": onset_m,
            "tuber_count": tuber_count,
            "drug_resistant": drug_resistant,
            "seizure_free": seizure_free,
            "has_infantile_spasms": has_is,
            "has_sega": has_sega,
            "aml_present": aml,
            "pkd1_deletion": pkd1_deletion,
            "tand_asd": tand_asd,
            "on_everolimus": on_everolimus,
            "on_vpa": on_vpa,
            "polg_tested": polg_tested,
            "erg_done": erg_done,
            "lam_surveillance": lam_risk,
        })
    return patients


# ── API response builders ──────────────────────────────────────────────────────
def get_overview():
    """Return TSC2 overview dict."""
    pts = _generate_patients()
    n = len(pts)
    on_everolimus = sum(1 for p in pts if p["on_everolimus"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    has_is = sum(1 for p in pts if p["has_infantile_spasms"])
    has_sega = sum(1 for p in pts if p["has_sega"])
    aml = sum(1 for p in pts if p["aml_present"])
    tand_asd = sum(1 for p in pts if p["tand_asd"])
    polg_done = sum(1 for p in pts if p["polg_tested"] == "Y")
    erg_done = sum(1 for p in pts if p["erg_done"])
    vpa_without_polg = sum(1 for p in pts if p["on_vpa"] and p["polg_tested"] == "N")
    pkd1_del = sum(1 for p in pts if p["pkd1_deletion"])

    return {
        "gene": "TSC2",
        "locus": "16p13.3",
        "protein": "Tuberin (TSC1/TSC2 complex GAP for Rheb → mTORC1 suppression; 200 kDa)",
        "syndrome": "Tuberous Sclerosis Complex (TSC2 — severe) — mTOR pathway epilepsy",
        "incidence": "~70% of all TSC cases; TSC overall 1:6,000–1:10,000",
        "inheritance": "Autosomal dominant (AD); ~80% de novo (sporadic TSC2); ~20% inherited",
        "omim": "OMIM #613254 (TSC2); TSC2: 16p13.3; TSC1 partner: 9q34.13",
        "summary": (
            "TSC2 (16p13.3) encodes Tuberin, GAP protein for Rheb → mTORC1 suppression. "
            "TSC2 LOF causes ~70% of all TSC and is MORE SEVERE than TSC1: more cortical tubers "
            "(9–12 vs 6–8), higher ID (50–60% vs 30%), more infantile spasms (45% vs 35%), "
            "and more pulmonary LAM (25–35% females). Precision therapy: everolimus (EXIST-3 "
            "response 40%). Infantile spasms: VGB Level A first-line (95% TSC response; SHARE REMS). "
            "EPISTOP protocol: EEG surveillance from 1 month; presymptomatic VGB. "
            "TAND (ASD 45%, ID 55%). Annual renal AML + brain MRI (SEGA) + HRCT (LAM females). "
            "POLG mandatory before VPA. Tiagabine ABSOLUTE CI. CYP3A4/everolimus interaction CRITICAL."
        ),
        "n_patients": n,
        "on_everolimus_pct": round(100 * on_everolimus / n),
        "drug_resistant_pct": round(100 * drug_resistant / n),
        "seizure_free_pct": round(100 * seizure_free / n),
        "infantile_spasms_pct": round(100 * has_is / n),
        "sega_pct": round(100 * has_sega / n),
        "aml_pct": round(100 * aml / n),
        "tand_asd_pct": round(100 * tand_asd / n),
        "polg_done_pct": round(100 * polg_done / n),
        "erg_done_pct": round(100 * erg_done / n),
        "vpa_without_polg": vpa_without_polg,
        "pkd1_deletion_n": pkd1_del,
        "tiagabine_alert": "ABSOLUTE CI — NCSE risk (GAT-1 → tonic GABA-A); class-effect in ALL TSC2 patients",
        "everolimus_alert": "PRECISION THERAPY (Level A, EXIST-3). CYP3A4 inducers (CBZ/PHT) reduce trough 2–4× → recheck at 2 weeks on any CYP3A4 change",
        "vgb_alert": "VGB FIRST-LINE infantile spasms in TSC2 (Level A; 95% response; higher IS burden vs TSC1). SHARE REMS mandatory. ERG q3M.",
        "polg_alert": "POLG MANDATORY before VPA — Alpers-Huttenlocher hepatic failure risk",
        "sega_alert": "Annual brain MRI for SEGA (foramen of Monro). SEGA ≥1cm growth or symptomatic → everolimus (EXIST-1); neurosurgery if hydrocephalus.",
        "lam_alert": "Annual chest HRCT for TSC2 females ≥15y — pulmonary LAM (25–35%); significantly higher than TSC1.",
        "pkd1_alert": "TSC2+PKD1 contiguous deletion (~2%): neonatal polycystic kidneys + severe TSC. MLPA/array-CGH if renal cysts at birth.",
        "contraindications_summary": [
            "Tiagabine — ABSOLUTE CI: NCSE risk (GAT-1 → tonic GABA-A)",
            "CBZ/OXC/PHT without everolimus trough monitoring — HIGH RISK: CYP3A4 induction → subtherapeutic mTOR suppression",
            "VPA without POLG screen — HIGH RISK: Alpers-Huttenlocher hepatic failure",
            "Live vaccines during everolimus — HIGH RISK: immunosuppression",
            "Vigabatrin without ERG monitoring — HIGH RISK: SHARE REMS mandatory",
        ],
        "thresholds": THRESHOLDS,
        "references": [r.split(" (")[0] for r in REFERENCES],
    }


def get_breakdown():
    """Return TSC2 breakdown dict."""
    pts = _generate_patients()
    n = len(pts)
    on_everolimus = sum(1 for p in pts if p["on_everolimus"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    has_is = sum(1 for p in pts if p["has_infantile_spasms"])
    has_sega = sum(1 for p in pts if p["has_sega"])
    aml = sum(1 for p in pts if p["aml_present"])
    tand_asd = sum(1 for p in pts if p["tand_asd"])
    polg_done = sum(1 for p in pts if p["polg_tested"] == "Y")
    erg_done = sum(1 for p in pts if p["erg_done"])
    vpa_without_polg = sum(1 for p in pts if p["on_vpa"] and p["polg_tested"] == "N")

    etiology_dist = []
    for ec in ETIOLOGY_CATALOG:
        etiology_dist.append({
            "etiology": ec["etiology"],
            "pct": ec["pct"],
            "n": ec["n"],
            "category": ec["category"],
            "mechanism_short": ec["mechanism"][:120],
            "eeg_signature_short": ec["eeg_correlate"][:80],
        })

    return {
        "summary": {
            "n": n,
            "on_everolimus_pct": round(100 * on_everolimus / n),
            "drug_resistant_pct": round(100 * drug_resistant / n),
            "seizure_free_pct": round(100 * seizure_free / n),
            "infantile_spasms_pct": round(100 * has_is / n),
            "sega_pct": round(100 * has_sega / n),
            "aml_pct": round(100 * aml / n),
            "tand_asd_pct": round(100 * tand_asd / n),
            "polg_done_pct": round(100 * polg_done / n),
            "erg_done_pct": round(100 * erg_done / n),
            "vpa_without_polg": vpa_without_polg,
        },
        "etiology_distribution": etiology_dist,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "patients_sample": pts[:15],
        "seizure_types": [
            {"type": s["type"], "prevalence_pct": s["prevalence_pct"]}
            for s in SEIZURE_TYPES
        ],
        "seizure_detail": SEIZURE_TYPES,
        "triggers": [
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
            for t in TRIGGERS
        ],
        "trigger_detail": TRIGGERS,
        "treatment_detail": TREATMENTS,
        "contraindication_detail": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    """Return TSC2 definitions dict."""
    return {
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
