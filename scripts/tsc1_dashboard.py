"""
TSC1 Epilepsy — Tuberous Sclerosis Complex (TSC / mTOR / Hamartin / 9q34.13)
==============================================================================
40-patient cohort · TSC1 LOF · mTOR hyperactivation · Cortical tubers · Epilepsy

TSC1 BIOLOGY:
TSC1 (9q34.13) encodes Hamartin, a 130-kDa scaffold protein that forms a
heterodimeric complex with Tuberin (TSC2, 16p13.3). The TSC1/TSC2 complex
functions as a GTPase-activating protein (GAP) for Rheb (Ras homolog enriched
in brain), keeping Rheb in its GDP-bound inactive form and thereby suppressing
mTORC1 (mechanistic target of rapamycin complex 1) activity.

mTOR PATHWAY & CORTICAL TUBER PATHOMECHANISM:
  - TSC1 LOF → Hamartin loss → TSC1/TSC2 complex destabilised → Tuberin
    also degraded → Rheb remains GTP-loaded (constitutively active) →
    mTORC1 hyperactivation → uncontrolled phosphorylation of S6K1 and 4E-BP1 →
    dysregulated protein synthesis, cell growth, and proliferation.
  - In brain: mTORC1 hyperactivation during cortical development →
    arrest of radial neuronal migration → giant cells (balloned cells with
    aberrant myelination) + dysmorphic neurons → cortical tubers
    (focal areas of disorganised cortex). Each tuber is epileptogenic.
  - Perilesional cortex: mTOR-driven glutamate receptor upregulation
    and GABAergic interneuron loss → local hyperexcitability → seizure onset.
  - Tuber burden (number, volume, FLAIR signal) correlates with epilepsy
    severity and intellectual disability. Cystic "FLAIR-bright tubers" most
    epileptogenic (80% seizure onset).

TSC1 vs TSC2 GENOTYPE-PHENOTYPE:
  - TSC1 (9q34): ~30% of TSC. Milder phenotype on average. Fewer cortical
    tubers. Intellectual disability in ~30%. Renal angiomyolipoma in ~70%.
    Pulmonary LAM (lymphangioleiomyomatosis) in females — rare in TSC1
    vs. common TSC2.
  - TSC2 (16p13): ~70% of TSC. More severe: more tubers, higher seizure
    burden, ID in ~50-60%, pulmonary LAM ~30% females. Contiguous deletion
    of TSC2+PKD1 (polycystic kidneys) seen in ~2%.
  - Mosaic TSC: 10-15% cases; next-generation sequencing (NGS) at ≥200×
    depth required; phenotype variable but often milder.
  - ~15% TSC patients: no pathogenic variant found by standard analysis
    (deep-read NGS or RNA sequencing may reveal mosaic/splice variants).

CORTICAL TUBER BIOLOGY & EPILEPSY SURGERY:
  - Tuber classification: type A (no cyst), type B (small cyst), type C
    (large cyst — most epileptogenic, 80%). FLAIR-hyperintense tubers on 3T MRI.
  - Epilepsy surgery evaluation: invasive SEEG/ECoG grid to identify
    epileptogenic tuber. Resection of epileptogenic tuber → seizure-free
    in 50-60% at 2 years (Srikantha 2019). mTOR inhibitors may reduce
    remaining tuber burden post-surgery.
  - Subependymal giant cell astrocytoma (SEGA): benign WHO grade I tumour
    at foramen of Monro; obstructive hydrocephalus; everolimus shrinks SEGA
    (EXIST-1 trial: 35% ≥50% volume reduction). Annual brain MRI to monitor.

mTOR INHIBITOR PRECISION THERAPY (EVEROLIMUS — AFINITOR):
  Everolimus is an allosteric mTORC1 inhibitor (rapamycin analogue). It
  binds FKBP12, forming a complex that directly inhibits mTORC1, reducing
  S6K1/4E-BP1 phosphorylation and normalising dysregulated mRNA translation.

  FDA APPROVED INDICATIONS IN TSC:
  - Renal angiomyolipoma (2012, EXIST-2)
  - Subependymal giant cell astrocytoma (2011, EXIST-1)
  - TSC-associated partial-onset seizures (2017, EXIST-3):
    EXIST-3 (Bissler 2017): everolimus adjunctive therapy (trough 3-7 or
    9-15 ng/mL) vs placebo. Response rate 40% (≥50% seizure reduction) vs
    15% placebo. Seizure freedom 13% vs 0%. Overall well-tolerated.
  - Pulmonary LAM (MILES trial, 2011)

  DOSING: 4.5 mg/m² daily (BSA-based), adjusted to trough 5-15 ng/mL
  (TSC seizures: 3-7 ng/mL first line, escalate to 9-15 ng/mL if needed).
  Monitor trough level at 2 weeks, then monthly until stable, then q3-6M.

  KEY SAFETY:
  - Immunosuppression: risk of bacterial/viral/fungal infections. Avoid
    live vaccines. PCP prophylaxis (TMP-SMX) in children on full-dose.
  - Oral mucositis/stomatitis: 70-80% (dose-limiting); non-alcoholic
    mouthwash; dose reduction if Grade 2+.
  - Hyperlipidaemia: 70%; monitor LDL/TG q3M.
  - Hyperglycaemia: 50%; HbA1c q3M; caution in pre-diabetic patients.
  - Interstitial lung disease (ILD): rare but serious; CT chest if new
    cough/dyspnoea. Dose reduction or discontinuation.
  - Wound healing impairment: hold 1 week pre/post surgery.
  - Drug interactions: CYP3A4 inhibitors (azoles, clarithromycin) →
    increased everolimus levels. CYP3A4 inducers (CBZ, PHT, rifampicin)
    → reduced everolimus levels — AVOID or adjust dose upward.
  - MONITORING PANEL: everolimus trough, CBC, LFTs, LDL/TG, HbA1c,
    creatinine, urinalysis (renal AML), brain MRI (SEGA), annual
    ophthalmology (retinal hamartoma), renal ultrasound.

VIGABATRIN (VGB) FOR TSC INFANTILE SPASMS:
  - VGB is FIRST-LINE for infantile spasms in TSC (Level A: UKISS trial,
    Darke 2010; TSC-specific INFANT/UKISS data). Earlier treatment
    correlates with better neurodevelopmental outcome.
  - TSC-specific response: 95% spasm cessation at 1 month (vs ~70% in
    non-TSC IS). Mechanism: GABA-T inhibitor → increased synaptic GABA
    → reduces tuber-driven hyperexcitability.
  - VGB-RETINAL TOXICITY: irreversible visual field constriction in
    30-50% adults (less clear in infants). MANDATORY ERG/perimetry:
    baseline (or within 4 weeks of start), then q3M in infants, q6M in
    children, q6M adults. SHARE REMS program (USA) mandatory enrolment.
  - VGB ABSOLUTE CI: myoclonic seizures in non-TSC patients (may worsen).
    In TSC: use cautiously if myoclonic component present.
  - Early VGB in presymptomatic TSC (EPISTOP/PREVENT trials): VGB started
    at EEG abnormality (before spasms) → 50% reduction in epilepsy onset
    rate; better neurocognitive outcomes. Screen EEG from age 1 month.

ADDITIONAL AEDs IN TSC:
  - ACTH (tetracosactide): alternative first-line for infantile spasms
    (non-inferior to VGB in non-TSC patients; less effective than VGB
    in TSC specifically). Use if VGB contraindicated.
  - Everolimus (mTOR inhibitor): precision therapy for focal-onset
    and multifocal seizures from age 2 years.
  - LEV (levetiracetam): broad-spectrum adjunct; TSC-compatible; no CI.
  - VPA (valproate): broad-spectrum; POLG MANDATORY before starting.
  - CBZ/OXC: focal seizures — CAUTION: CBZ/OXC induce CYP3A4 →
    significantly reduce everolimus trough levels; if CBZ/OXC needed,
    increase everolimus dose and recheck trough.
  - CLB (clobazam): adjunct for Lennox-Gastaut phenotype in TSC.
  - KD (ketogenic diet): Level B; beneficial in drug-resistant TSC epilepsy;
    may work synergistically with everolimus (mTOR suppression + metabolic).
  - Rapamycin (sirolimus): alternative mTOR inhibitor; LAM indication;
    off-label for TSC seizures if everolimus not available.

ABSOLUTE CONTRAINDICATIONS IN TSC:
  1. Tiagabine — ABSOLUTE CI in ALL TSC patients: GAT-1 inhibition →
     extracellular GABA accumulation → tonic GABA-A activation → NCSE.
     Same class effect as in all genetic epilepsies.
  2. CBZ/OXC without everolimus trough monitoring — HIGH RISK: CYP3A4
     induction reduces everolimus to subtherapeutic levels; tubers remain
     active without mTOR suppression.
  3. VPA without POLG screen — HIGH RISK: Alpers-Huttenlocher hepatic
     failure in POLG-positive patients. POLG screening MANDATORY.
  4. Everolimus + live vaccines — HIGH RISK: immunosuppression.
  5. Vigabatrin without ERG monitoring — HIGH RISK: irreversible
     visual field constriction (SHARE REMS mandatory).

SURGICAL CONSIDERATIONS:
  - MEG + 3T/7T MRI tuber mapping; FDG-PET hypometabolism identifies
    epileptogenic tubers. Alpha-[11C]-methyl-L-tryptophan (AMT-PET) —
    tryptophan trapping in epileptogenic tubers (Juhász 2006).
  - SEEG preferred for multifocal TSC (multiple tubers). ECoG grid
    for single-region hypothesis.
  - Laser interstitial thermal therapy (LITT/LiTT): minimally invasive
    ablation of single epileptogenic tuber — 50-60% seizure-free at 1y.
  - SRS (stereotactic radiosurgery) / GKS: evidence limited; reserved
    for surgically inaccessible tubers.

POLG MANDATORY:
  POLG biallelic pathogenic variants + VPA → Alpers-Huttenlocher hepatic
  failure. In TSC patients requiring VPA: exclude POLG variants before
  starting. Panel: POLG full gene sequencing.

HLA-B*15:02 (if CBZ/OXC considered):
  CPIC Level A: Asian ancestry → HLA-B*15:02 before CBZ or OXC.
  In TSC, CBZ/OXC rarely first-line (everolimus-interaction risk).
  Test before prescribing. Substitute LEV if HLA-B*15:02 positive.

RENAL ANGIOMYOLIPOMA (AML) IN TSC:
  - Renal AML: 55-75% of TSC patients. Risk of spontaneous haemorrhage
    (Wunderlich syndrome) when AML >4 cm or aneurysm >5 mm.
  - Everolimus (EXIST-2): 42% AML volume reduction; first-line medical Rx.
  - Embolisation: for acute bleeding or AML >4cm not responding to mTOR.
  - Renal cell carcinoma: rare but increased risk; annual renal imaging.

NEURODEVELOPMENTAL OUTCOMES:
  - TSC-Associated Neuropsychiatric Disorders (TAND): autism spectrum
    disorder 50-60%, intellectual disability 45-60% (TSC1 ~30%, TSC2 ~60%),
    ADHD 50%, anxiety 40%, depression 30%.
  - TAND assessment annually (TAND Checklist). Early intervention.
  - Epilepsy surgery: improves neurocognitive outcomes when seizure-free.
  - mTOR inhibitors may improve cognition (EXIST-3 secondary endpoints).
"""

import random
from datetime import datetime

random.seed(42)

# ── Etiology catalog ─────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "TSC1 LOF — AD familial (BFNS-like / mild)",
        "pct": 25,
        "n": 10,
        "category": "TSC1-inherited",
        "mechanism": (
            "Inherited heterozygous TSC1 LOF → hamartin deficiency → TSC1/TSC2 complex "
            "instability → moderate mTORC1 hyperactivation. Fewer cortical tubers than "
            "TSC2-family; intellectual disability less common (~25%). Familial AD pattern. "
            "Seizure onset often in childhood; 50% achieve seizure control with AEDs ± VGB."
        ),
        "eeg_correlate": (
            "Multifocal interictal epileptiform discharges (IEDs) over tuber regions; "
            "hypsarrhythmia during infantile spasm phase; focal slow waves over tubers; "
            "MRI: moderate tuber burden (mean 8-12 tubers); type A/B predominance."
        ),
        "clinical_tip": (
            "Screen family members for TSC features (skin findings: ash-leaf macules, "
            "shagreen patches, angiofibromas). Genetic counselling: 50% transmission risk. "
            "Milder phenotype but presymptomatic EEG monitoring still recommended from 1M."
        ),
    },
    {
        "etiology": "TSC1 LOF — de novo severe (early IS / LGS)",
        "pct": 20,
        "n": 8,
        "category": "TSC1-de-novo",
        "mechanism": (
            "De novo heterozygous TSC1 pathogenic variant; higher phenotypic severity; "
            "earlier onset (infantile spasms 3-8 months); more tubers; higher drug-resistance. "
            "de novo TSC1 accounts for ~30% of TSC1 cases. Hamartin protein fully absent "
            "in tuber giant cells → maximal mTORC1 drive in developing cortex."
        ),
        "eeg_correlate": (
            "Hypsarrhythmia (infantile spasm phase); multifocal IEDs post-IS; "
            "diffuse slow background if encephalopathic; CSWS possible in school age; "
            "MRI: higher tuber count (>15), periventricular nodules, SEGA risk."
        ),
        "clinical_tip": (
            "VGB first-line for IS; monitor ERG q3M. Transition to everolimus at age 2y "
            "if seizures persist. Annual renal US + brain MRI (SEGA at foramen of Monro). "
            "TAND assessment at diagnosis and annually. "
        ),
    },
    {
        "etiology": "TSC mosaic (10-15% NGS cases)",
        "pct": 15,
        "n": 6,
        "category": "TSC1-mosaic",
        "mechanism": (
            "Somatic mosaic TSC1 variant (post-zygotic); variant allele fraction (VAF) "
            "typically 5-30% in blood (higher in affected tissue). Requires NGS ≥200× depth. "
            "Milder systemic phenotype (fewer tubers, less renal AML). Epilepsy severity "
            "depends on proportion of affected progenitor cells during cortical development."
        ),
        "eeg_correlate": (
            "Focal or multifocal IEDs; often fewer high-amplitude discharges than "
            "constitutional TSC. MRI may show fewer/smaller tubers. AMT-PET can identify "
            "epileptogenic tubers even when MRI-negative. EEG may be near-normal inter-ictally."
        ),
        "clinical_tip": (
            "Standard Sanger sequencing may miss mosaic TSC1. Request deep-read NGS "
            "(≥200× blood + if negative, skin fibroblasts or resected tuber tissue). "
            "Parent testing: germline mutation rules out new mosaic. "
        ),
    },
    {
        "etiology": "TSC1 with SEGA (subependymal giant cell astrocytoma)",
        "pct": 20,
        "n": 8,
        "category": "TSC1-SEGA",
        "mechanism": (
            "TSC1 LOF with SEGA growth (foramen of Monro) → obstructive hydrocephalus risk. "
            "SEGA: benign WHO grade I; develops from subependymal nodules in 10-15% TSC. "
            "mTORC1-driven cell hypertrophy creates nodule→SEGA transition (>1 cm growth "
            "or symptomatic hydrocephalus). Everolimus EXIST-1: 35% ≥50% SEGA shrinkage."
        ),
        "eeg_correlate": (
            "Focal IEDs may acutely worsen with SEGA-related hydrocephalus. "
            "Seizure burden increases with raised ICP. EEG: diffuse slowing if hydrocephalus. "
            "MRI: enhancing SEGA at foramen of Monro (annual + if symptoms change)."
        ),
        "clinical_tip": (
            "Annual brain MRI in all TSC patients (earlier if headache/vomiting/vision change). "
            "SEGA >1cm → everolimus (shrinkage avoids surgery in most). "
            "Surgical resection if hydrocephalus not controlled by CSF diversion + mTOR Rx. "
        ),
    },
    {
        "etiology": "Phenocopy / TSC2 / mTOR pathway variants",
        "pct": 20,
        "n": 8,
        "category": "phenocopy",
        "mechanism": (
            "TSC2 (tuberin, 16p13.3) mutations or downstream mTOR pathway variants "
            "(DEPDC5/NPRL2/NPRL3 GATOR1 complex; PIK3CA; MTOR somatic). Clinical TSC "
            "features without TSC1 pathogenic variant. TSC2 typically more severe "
            "(more tubers, ID in 50-60%, pulmonary LAM). GATOR1 variants: focal cortical "
            "dysplasia without classic tubers — consider mTOR inhibitor therapy."
        ),
        "eeg_correlate": (
            "Similar to constitutional TSC1/TSC2: multifocal IEDs, hypsarrhythmia (IS). "
            "GATOR1/PIK3CA/MTOR somatic: focal IEDs over dysplastic region; "
            "radial band pattern on MRI in PIK3CA."
        ),
        "clinical_tip": (
            "Full TSC gene panel (TSC1+TSC2) + mTOR pathway panel if negative. "
            "mTOR inhibitors effective regardless of upstream variant (target: mTORC1). "
            "Discuss genotype-phenotype prognosis with family."
        ),
    },
]

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms (West Syndrome)",
        "prevalence_pct": 35,
        "onset_age": "3-8 months",
        "eeg": "Hypsarrhythmia (high-amplitude chaotic slow + multifocal IEDs); electrodecrement ictal pattern",
        "semiology": "Flexion/extension spasm clusters (salaam attacks); eye deviation; brief tonic jerks",
        "clinical_tip": (
            "VGB FIRST-LINE in TSC infantile spasms (Level A; UKISS). "
            "ACTH second-line if VGB fails at 2 weeks. "
            "ERG q3M mandatory with VGB (SHARE REMS). "
            "Presymptomatic EEG from 1 month → early VGB if IEDs → "
            "EPISTOP data: 50% reduction in IS incidence."
        ),
    },
    {
        "type": "Focal Aware Seizures (FAS)",
        "prevalence_pct": 55,
        "onset_age": "Any age (peaks 1-5y)",
        "eeg": "Focal IEDs + focal theta/delta over tuber region; rhythmic ictal discharge 5-8 Hz",
        "semiology": "Autonomic / sensory / psychic auras; motor versive; preserved awareness; "
                     "variable duration 30s-3min",
        "clinical_tip": (
            "Identify epileptogenic tuber with MRI (type C FLAIR-bright) + AMT-PET. "
            "Everolimus (mTOR inhibitor) for drug-resistant focal seizures. "
            "SEEG evaluation for surgical candidacy in drug-resistant FAS."
        ),
    },
    {
        "type": "Focal to Bilateral Tonic-Clonic (FBTCS)",
        "prevalence_pct": 65,
        "onset_age": "Childhood-adult",
        "eeg": "Focal onset IEDs → bilateral synchrony; post-ictal diffuse slowing; "
               "may show tuber activation on ictal SPECT",
        "semiology": "Focal aura → bilateral tonic → clonic phase; post-ictal confusion; "
                     "risk SUDEP if recurrent FBTCS",
        "clinical_tip": (
            "FBTCS in TSC = HIGH SUDEP risk; nocturnal supervision. "
            "Target drug-resistant FBTCS with everolimus + CLB/VPA combination. "
            "Seizure diary mandatory; rescue PRN benzodiazepine."
        ),
    },
    {
        "type": "Tonic Seizures / Lennox-Gastaut Phenotype",
        "prevalence_pct": 25,
        "onset_age": "2-8 years (after IS)",
        "eeg": "Diffuse slow spike-and-wave (<2.5 Hz, IQ of 1 Hz); paroxysmal fast (beta bursts) during tonic",
        "semiology": "Sudden axial tonic stiffening; drop attacks (atonic variant); nocturnal predominance; "
                     "risk of tonic status",
        "clinical_tip": (
            "LGS phenotype in TSC: CLB + VPA backbone ± rufinamide (FDA-approved LGS). "
            "Everolimus adjunct for mTOR-driven tonic component. "
            "Fall protection (helmet, padding) for drop attacks."
        ),
    },
    {
        "type": "Epileptic Spasms (post-infantile / late-onset)",
        "prevalence_pct": 18,
        "onset_age": "2-5 years",
        "eeg": "Electrodecrement or diffuse slow wave + IED; hypsarrhythmia may resolve partially",
        "semiology": "Persistent or relapsing spasms after infantile phase; often focal-onset spasms "
                     "from single tuber",
        "clinical_tip": (
            "Late spasms in TSC: tuber-specific onset; SEEG to localize; "
            "everolimus can reduce spasm frequency. "
            "ACTH or repeat VGB course occasionally effective."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "prevalence_pct": 15,
        "onset_age": "Childhood-adolescent",
        "eeg": "Irregular spike/polyspike-and-wave; frontally predominant; photo-sensitivity possible",
        "semiology": "Brief bilateral or focal myoclonic jerks; morning predominance in JME-like variant",
        "clinical_tip": (
            "Myoclonus in TSC: avoid lamotrigine (may worsen myoclonus via Na+ channel "
            "effects in GABAergic neuron subtypes). VPA first-choice for myoclonic component. "
            "POLG mandatory before VPA."
        ),
    },
]

# ── Triggers ─────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Hyperthermia (≥37.8°C)", "prevalence_pct": 78,
     "detail": "Temperature-dependent axonal Kv channel inactivation in tuber peri-lesional cortex. "
               "Fever management MANDATORY; paracetamol/ibuprofen at ≥37.5°C in young TSC children."},
    {"trigger": "Missed everolimus / AED dose", "prevalence_pct": 72,
     "detail": "Everolimus trough <3 ng/mL → mTORC1 rebound → acute seizure escalation. "
               "AED adherence monitoring. Dose reminder apps; pill dispensers."},
    {"trigger": "Sleep deprivation", "prevalence_pct": 65,
     "detail": "Tuber peri-lesional cortex is highly sleep-state-dependent. "
               "Overnight EEG captures CSWS/nocturnal IEDs in school-age TSC. "
               "Strict sleep schedule; melatonin for circadian dysregulation in ASD co-morbidity."},
    {"trigger": "Intercurrent viral illness / GI illness", "prevalence_pct": 60,
     "detail": "Systemic inflammation + everolimus trough alteration (reduced absorption with emesis). "
               "Sick-day protocol: recheck everolimus trough after illness; rescue benzodiazepine PRN."},
    {"trigger": "CYP3A4 drug interaction (CBZ/PHT/rifampicin)", "prevalence_pct": 45,
     "detail": "CYP3A4 inducers reduce everolimus trough by 2-4× → subtherapeutic mTOR suppression "
               "→ acute seizure breakthrough. Always recheck trough 2w after any CYP3A4 change."},
    {"trigger": "Emotional stress / anxiety", "prevalence_pct": 55,
     "detail": "High anxiety prevalence in TSC (40%); stress → HPA-axis activation → "
               "GABA system modulation → seizure threshold reduction. Cognitive-behavioural "
               "therapy (CBT) + anxiolytic AED selection (CLB preferred)."},
    {"trigger": "Photosensitivity", "prevalence_pct": 20,
     "detail": "Minority of TSC; more common in LGS/JME phenotype overlap. "
               "Screen EEG with photic stimulation. Polarised lenses if photosensitive."},
    {"trigger": "SEGA growth / raised ICP", "prevalence_pct": 12,
     "detail": "Acute SEGA enlargement → obstructive hydrocephalus → acute seizure clustering. "
               "Emergency brain MRI + neurosurgical consult if acute-onset headache + seizure increase."},
]

# ── Treatments ───────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Vigabatrin (VGB / Sabril)",
        "level": "Level A — Infantile Spasms in TSC",
        "dose": "100-150 mg/kg/day divided BID (infants); max 3 g/day (adults). "
                "Start 50 mg/kg/day, titrate over 2 weeks.",
        "moa": "Irreversible GABA-transaminase (GABA-T) inhibitor → elevated synaptic GABA → "
               "reduces tuber-driven cortical hyperexcitability.",
        "efficacy": "95% IS cessation in TSC at 1 month (UKISS; vs ~65% non-TSC). "
                    "Presymptomatic VGB (EPISTOP): 50% IS prevention.",
        "safety": "Irreversible visual field constriction (30-50% adults; pediatric risk unclear). "
                  "SHARE REMS enrolment mandatory (USA). ERG at baseline, q3M infants, q6M children. "
                  "Transient MRI signal changes (reversible diffusion abnormalities) in infants — "
                  "do not discontinue; reassess.",
        "monitoring": "ERG/visual field q3M (infants), q6M (children/adults). "
                      "Ophthalmology referral. Brain MRI if diffusion signal change noted.",
        "tsc_note": "TSC-specific precision: VGB uniquely effective in TSC IS vs other IS causes. "
                    "Start VGB BEFORE ACTH in TSC infantile spasms. SHARE REMS mandatory USA enrolment.",
    },
    {
        "drug": "Everolimus (Afinitor / Votubia)",
        "level": "Level A — TSC-associated focal seizures (FDA 2017; EXIST-3)",
        "dose": "4.5 mg/m²/day PO QD. Adjust to trough 5-15 ng/mL "
                "(seizures: 3-7 ng/mL first, escalate to 9-15 ng/mL if needed). "
                "Available from age 2 years (TSC seizure indication).",
        "moa": "Allosteric mTORC1 inhibitor (rapamycin analogue via FKBP12) → "
               "reduces S6K1/4E-BP1 phosphorylation → normalises dysregulated protein "
               "synthesis in tuber giant cells → reduces seizure initiation.",
        "efficacy": "EXIST-3: ≥50% seizure reduction in 40% (high-trough) vs 15% placebo. "
                    "Seizure freedom 13% vs 0%. Durable response at 48 weeks.",
        "safety": "Stomatitis/mucositis 70-80% (dose-limiting; non-alcoholic mouthwash). "
                  "Infections (avoid live vaccines; PCP prophylaxis in children). "
                  "Hyperlipidaemia 70% (LDL/TG q3M). Hyperglycaemia 50% (HbA1c q3M). "
                  "ILD rare but serious. Wound healing impairment (hold perioperatively).",
        "monitoring": "Everolimus trough at 2w → 1M → q3-6M. CBC, LFT, LDL/TG, HbA1c, "
                      "creatinine, UA q3M. Brain MRI q1y (SEGA). Renal US q1y (AML).",
        "tsc_note": "TSC PRECISION THERAPY. Only FDA-approved mTOR inhibitor for TSC seizures. "
                    "CYP3A4 INTERACTION: CBZ/PHT/OXC → reduce everolimus 2-4× → recheck trough. "
                    "Do NOT co-administer with azole antifungals without trough monitoring.",
    },
    {
        "drug": "ACTH (Tetracosactide / Synacthen)",
        "level": "Level B — Infantile Spasms (alternative to VGB if contraindicated)",
        "dose": "0.5-1.0 mg/kg/day IM for 2 weeks (UK IS protocol). Taper per local protocol.",
        "moa": "Synthetic ACTH analogue → adrenal cortisol release + direct brain MCR (melanocortin "
               "receptor) anti-seizure effect. Reduces CRH-driven limbic hyperexcitability.",
        "efficacy": "IS cessation ~65-75% non-TSC (UKISS). Less effective than VGB in TSC specifically "
                    "(VGB preferred). Use as second-line or in VGB contraindication.",
        "safety": "Hypertension (BP monitoring daily during Rx). Cushing features (transient). "
                  "Immunosuppression (infection risk). GI bleeding prophylaxis with PPI.",
        "monitoring": "BP, electrolytes, blood glucose daily during ACTH phase. "
                      "EEG at day 14 to assess IS resolution.",
        "tsc_note": "Second-line IS therapy in TSC (VGB first). Combine ACTH + VGB if initial "
                    "VGB response incomplete (UKISS data). Do NOT use ACTH as first-line in TSC.",
    },
    {
        "drug": "Levetiracetam (LEV / Keppra)",
        "level": "Level B — Adjunct broad-spectrum (focal / FBTCS / myoclonic)",
        "dose": "20-60 mg/kg/day divided BID (children); 500-3000 mg/day BID (adults).",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation → reduced synaptic vesicle "
               "release → decreased neurotransmitter exocytosis at epileptic foci.",
        "efficacy": "Moderate efficacy in TSC focal seizures; no drug interactions with everolimus. "
                    "Safe across all TSC phenotypes.",
        "safety": "Behavioural side effects (irritability, aggression) in 15-25% — especially in TSC "
                  "with ADHD/ASD comorbidity. Pyridoxine (B6) may reduce behavioural SE.",
        "monitoring": "Renal function (dose-adjust if eGFR <60). Behavioural monitoring. "
                      "No therapeutic drug monitoring required.",
        "tsc_note": "No CYP3A4 interaction — safe co-administration with everolimus. "
                    "Preferred adjunct when everolimus initiated (no trough interaction).",
    },
    {
        "drug": "Valproate (VPA / Depakote)",
        "level": "Level B — Broad-spectrum (FBTCS / myoclonic / LGS-like in TSC)",
        "dose": "20-60 mg/kg/day divided BID-TID (children); 500-2500 mg/day (adults). "
                "Target trough 50-100 μg/mL.",
        "moa": "Na+ channel blockade + GABA-T inhibition + T-type Ca2+ channel block + "
               "HDAC inhibition. Broad-spectrum including myoclonus.",
        "efficacy": "Effective for FBTCS and myoclonic component in TSC; moderate for focal seizures.",
        "safety": "Teratogenicity (NTD 2-10%; VPPP mandatory in females of childbearing potential). "
                  "Hepatotoxicity (especially POLG+ patients — FATAL). "
                  "Hyperammonaemia, thrombocytopenia, tremor, weight gain. "
                  "VPA modestly inhibits CYP3A4 — may slightly increase everolimus levels.",
        "monitoring": "POLG MANDATORY before starting. LFT + ammonia baseline and q3M. "
                      "CBC/platelets. VPA trough 50-100 μg/mL. VPPP consent in females.",
        "tsc_note": "POLG MANDATORY before VPA in ALL TSC patients. "
                    "VPA-everolimus interaction: VPA slightly elevates everolimus trough "
                    "via CYP3A4 inhibition — recheck trough 2w after starting VPA. "
                    "VPPP (Valproate Pregnancy Prevention Programme) mandatory UK/EU.",
    },
    {
        "drug": "Clobazam (CLB / Onfi)",
        "level": "Level B — Adjunct LGS phenotype / drop attacks / acute cluster rescue",
        "dose": "0.1-0.3 mg/kg/day BID (children); 10-40 mg/day BID (adults). "
                "Intermittent pulse: 0.5-1 mg/kg/day × 3-5 days for febrile cluster.",
        "moa": "1,5-benzodiazepine (less respiratory depression than 1,4-BZD) → GABA-A potentiation "
               "(α2/α3 subunit preference). Fewer sedation/tolerance issues than diazepam.",
        "efficacy": "Drop attack reduction 50-70% in LGS (CONTAIN). Acute rescue effective in TSC "
                    "febrile clusters and post-ictal clusters.",
        "safety": "Tolerance (months). Sedation. N-CLB active metabolite accumulates (renal/hepatic "
                  "impairment). Drug interaction: CLB is CYP3A4 substrate → everolimus levels "
                  "minimally affected, but FLX/FLV + CLB → N-CLB accumulation.",
        "monitoring": "N-CLB levels if toxicity suspected. Sedation scale. Seizure diary for "
                      "LGS drop frequency.",
        "tsc_note": "Preferred rescue + adjunct in TSC/LGS. Pulse CLB for febrile clusters "
                    "(avoids hospital admission). No significant everolimus trough interaction.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — Drug-resistant epilepsy in TSC",
        "dose": "Classical KD 4:1 ratio (fat:protein+carb) or MCT oil diet. "
                "Dietitian-supervised. Target ketones 2-5 mmol/L.",
        "moa": "Metabolic ketosis → ATP-sensitive K+ channel activation (KATP) → "
               "membrane hyperpolarisation + reduction of neuronal excitability. "
               "KD may suppress mTORC1 (AMP kinase activation inhibits mTOR) — "
               "potential synergy with everolimus.",
        "efficacy": "≥50% seizure reduction in 50-60% drug-resistant TSC (Neal 2008; Kossoff 2009). "
                    "Possible synergy with everolimus in preclinical TSC models.",
        "safety": "Constipation, hyperlipidaemia, kidney stones (urine alkalization; adequate hydration). "
                  "Growth monitoring (children). Selenium/carnitine supplementation.",
        "monitoring": "Urinary ketones daily. Serum ketones q3M. Lipids, uric acid, UA q3M. "
                      "KD + everolimus: monitor renal function (additive metabolic load).",
        "tsc_note": "Consider KD when ≥3 AEDs fail in TSC. KD+everolimus combination: "
                    "emerging data for synergy; monitor AML size (mTOR fully suppressed). "
                    "Acidosis risk with KD + acetazolamide — AVOID combination.",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Tiagabine",
        "severity": "ABSOLUTE CI",
        "reason": (
            "GAT-1 GABA transporter inhibition → extracellular GABA accumulation → "
            "tonic GABA-A receptor activation → non-convulsive status epilepticus (NCSE). "
            "Class-effect in all focal epilepsies including TSC. "
            "Case reports of acute NCSE within 24-48h of tiagabine initiation in TSC."
        ),
        "alternative": "CLB / VPA / LEV for focal seizures in TSC.",
    },
    {
        "drug": "CBZ / OXC / PHT without everolimus trough monitoring",
        "severity": "HIGH RISK",
        "reason": (
            "CYP3A4 enzyme induction reduces everolimus trough by 2-4× within 7-14 days. "
            "Subtherapeutic everolimus → mTORC1 rebound → acute seizure breakthrough + "
            "potential SEGA re-growth. If CBZ/OXC/PHT required, increase everolimus dose "
            "and recheck trough at 2 weeks."
        ),
        "alternative": "LEV (no CYP3A4 interaction). HLA-B*15:02 before CBZ/OXC if prescribed.",
    },
    {
        "drug": "VPA without POLG screening",
        "severity": "HIGH RISK",
        "reason": (
            "POLG biallelic pathogenic variants + VPA → mitochondrial dysfunction → "
            "Alpers-Huttenlocher syndrome (progressive liver failure + refractory seizures, fatal). "
            "POLG full gene sequencing MANDATORY before starting VPA in all TSC patients."
        ),
        "alternative": "LEV / CLB / KD in POLG-positive patients.",
    },
    {
        "drug": "Live vaccines during everolimus therapy",
        "severity": "HIGH RISK",
        "reason": (
            "mTOR inhibition impairs T-cell proliferation (key for vaccine immunity and "
            "microbial clearance). Live vaccines (MMR, varicella, yellow fever, BCG) "
            "may cause disseminated vaccine-strain infection in immunosuppressed patients. "
            "Complete all live vaccines BEFORE everolimus initiation (≥4 weeks gap)."
        ),
        "alternative": "Inactivated/recombinant vaccines safe during everolimus. "
                       "Pneumococcal + annual influenza vaccine recommended.",
    },
    {
        "drug": "Vigabatrin without ERG monitoring",
        "severity": "HIGH RISK",
        "reason": (
            "Vigabatrin causes irreversible visual field constriction (nasal/peripheral) in "
            "30-50% adults; paediatric risk less quantified. GABA accumulation in retina → "
            "cone-mediated ERG b-wave amplitude reduction. SHARE REMS mandatory (USA). "
            "VGB should NEVER be initiated without ophthalmology ERG referral."
        ),
        "alternative": "If ERG cannot be arranged urgently, document attempt and initiate VGB "
                        "(infantile spasms benefit outweighs risk) with earliest possible ERG.",
    },
]

# ── Monitoring ───────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG gene sequencing", "frequency": "Once, before VPA", "rationale": "Alpers-Huttenlocher prevention"},
    {"item": "Everolimus trough level", "frequency": "2w after start, 1M, then q3-6M", "rationale": "Target 5-15 ng/mL (seizure 3-7)"},
    {"item": "ERG / visual field (VGB)", "frequency": "Baseline + q3M infants / q6M children-adults", "rationale": "SHARE REMS; VGB retinal toxicity"},
    {"item": "Brain MRI (SEGA + tubers)", "frequency": "Annual (q6M if SEGA borderline)", "rationale": "SEGA growth at foramen of Monro → hydrocephalus"},
    {"item": "Renal ultrasound (AML)", "frequency": "Annual (q6M if AML ≥3 cm)", "rationale": "Angiomyolipoma haemorrhage risk; mTOR response"},
    {"item": "LFT + serum ammonia (VPA)", "frequency": "Baseline, 6w, then q3M", "rationale": "Hepatotoxicity; hyperammonaemia"},
    {"item": "LDL / TG (everolimus)", "frequency": "Baseline, then q3M", "rationale": "Hyperlipidaemia 70%"},
    {"item": "HbA1c / fasting glucose (everolimus)", "frequency": "Baseline, then q3M", "rationale": "Hyperglycaemia 50%"},
    {"item": "TAND Checklist (neurodevelopmental)", "frequency": "Annual", "rationale": "ASD/ID/ADHD/anxiety screening; referral to psychology"},
    {"item": "HLA-B*15:02 (if CBZ/OXC planned)", "frequency": "Once, before first prescription", "rationale": "CPIC Level A; SJS/TEN prevention in Asian ancestry"},
    {"item": "Seizure diary", "frequency": "Ongoing (app-based recommended)", "rationale": "Tuber-specific seizure pattern; AED/everolimus response"},
    {"item": "SUDEP risk assessment", "frequency": "Annual (more frequent if drug-resistant)", "rationale": "TSC with nocturnal FBTCS = elevated SUDEP risk"},
]

# ── Lifecycle ─────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Prenatal (conception to birth)",
        "key_events": "TSC1/TSC2 genetic testing if family history; prenatal MRI from 22-24w "
                      "(tubers visible in fetus ≥20w on fetal MRI); prenatal echocardiography "
                      "(rhabdomyomas in 50% TSC fetus — regress after birth). "
                      "Genetic counselling: 50% AD transmission risk.",
        "priority": "Fetal MRI + echo; NICU plan if severe prenatal findings",
    },
    {
        "window": "Neonatal (0-3 months)",
        "key_events": "EEG from 1 month even if asymptomatic (EPISTOP protocol). "
                      "Cardiac rhabdomyomas monitoring (resolve spontaneously in most). "
                      "Dermatology: ash-leaf macules (Wood lamp), shagreen patches at birth. "
                      "Ophthalmology: retinal hamartomas, iris hypopigmentation.",
        "priority": "EEG monitoring; dermatology; ophthalmology; genetics confirm variant",
    },
    {
        "window": "Infantile (3-12 months)",
        "key_events": "Peak infantile spasm onset 3-8 months. VGB IMMEDIATELY on EEG "
                      "hypsarrhythmia or IEDs before IS (EPISTOP). ERG within 4 weeks of VGB. "
                      "SHARE REMS enrolment if VGB used. Brain MRI (3T) for tuber mapping. "
                      "Developmental assessment (TAND baseline). Renal ultrasound.",
        "priority": "VGB first-line IS; everolimus at age 2y if drug-resistant",
    },
    {
        "window": "Early Childhood (1-5 years)",
        "key_events": "Transition from IS/spasms to focal seizures (tuber-specific). "
                      "Everolimus start at age 2y for persistent seizures (EXIST-3). "
                      "Early intervention: speech/language/OT/PT. ASD assessment age 2-3. "
                      "Renal AML screening. SEGA baseline brain MRI. Annual TAND checklist.",
        "priority": "Everolimus initiation; ASD/ID early support; SEGA annual MRI",
    },
    {
        "window": "School Age (5-12 years)",
        "key_events": "Focal seizure consolidation. Epilepsy surgery evaluation if drug-resistant "
                      "(SEEG + AMT-PET). TAND: ADHD/anxiety school impact. KD consideration if "
                      "3+ AEDs failed. Annual renal AML + SEGA MRI. LGS phenotype management. "
                      "VGB ERG annual. Everolimus trough monitoring q3-6M.",
        "priority": "Epilepsy surgery candidacy assessment; TAND school plan; annual imaging",
    },
    {
        "window": "Adolescent / Adult (12+ years)",
        "key_events": "Pulmonary LAM screening (females, HRCT from age 18). "
                      "Renal AML with aneurysm >5mm → embolisation. "
                      "VPPP (valproate pregnancy prevention) in females of childbearing potential. "
                      "Transition to adult neurology/nephrology/genetics. SUDEP education. "
                      "Social/vocational support (50% ID; ASD). "
                      "Everolimus continuation (EXIST-3 long-term data: durable benefit).",
        "priority": "LAM screening; VPPP; adult transition care; SUDEP risk management",
    },
]

# ── Key concepts ──────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "TSC1 / Hamartin", "definition": "9q34.13; 130-kDa scaffold protein; TSC1/TSC2 complex GAP for Rheb → mTORC1 suppression; LOF → mTOR hyperactivation → cortical tubers"},
    {"term": "mTORC1 (mechanistic Target of Rapamycin Complex 1)", "definition": "Master regulator of cell growth/protein synthesis; hyperactivated by TSC1/TSC2 LOF; target of rapamycin/everolimus/sirolimus"},
    {"term": "Cortical Tuber", "definition": "Focal area of cortical dysorganisation from mTORC1 hyperactivation; giant balloon cells + dysmorphic neurons; epileptogenic (type C FLAIR-bright = most epileptogenic)"},
    {"term": "SEGA (Subependymal Giant Cell Astrocytoma)", "definition": "Benign WHO grade I mTOR-driven tumour at foramen of Monro; obstructive hydrocephalus risk; everolimus shrinks SEGA (EXIST-1)"},
    {"term": "Everolimus (Afinitor)", "definition": "mTORC1 inhibitor (rapamycin analogue via FKBP12); FDA-approved TSC seizures (EXIST-3), SEGA (EXIST-1), renal AML (EXIST-2)"},
    {"term": "Vigabatrin (VGB)", "definition": "Irreversible GABA-T inhibitor; Level A IS treatment in TSC (95% efficacy); SHARE REMS mandatory; ERG q3M for retinal toxicity"},
    {"term": "TAND (TSC-Associated Neuropsychiatric Disorders)", "definition": "ASD 50-60%; ID 45-60%; ADHD 50%; anxiety 40%; depression 30%. Annual TAND Checklist; early intervention"},
    {"term": "AMT-PET (alpha-[11C]-methyl-L-tryptophan)", "definition": "PET tracer for tryptophan metabolism; tryptophan trapping in epileptogenic tubers → identifies surgical target even when MRI non-lateralising (Juhász 2006)"},
    {"term": "EPISTOP / PREVENT Trials", "definition": "Presymptomatic VGB in TSC neonates at EEG IED stage → 50% reduction in infantile spasm incidence; better neurodevelopmental outcomes"},
    {"term": "TSC Mosaic (VAF 5-30%)", "definition": "Post-zygotic TSC1/TSC2 somatic variant; requires NGS ≥200× depth; milder systemic phenotype; standard Sanger misses low-VAF variants"},
    {"term": "Renal Angiomyolipoma (AML)", "definition": "55-75% of TSC; mTOR-driven hamartoma; bleeding risk if >4cm or aneurysm >5mm; everolimus first-line; embolisation for acute bleed"},
    {"term": "Pulmonary LAM", "definition": "Lymphangioleiomyomatosis; female TSC patients ~30% (TSC2 > TSC1); mTOR-driven smooth muscle proliferation in lung; HRCT screening from age 18"},
    {"term": "CYP3A4 Interaction (CBZ/PHT)", "definition": "Strong CYP3A4 inducers reduce everolimus trough 2-4×; subtherapeutic mTOR suppression → seizure breakthrough; avoid or increase everolimus dose + recheck trough"},
    {"term": "SHARE REMS (VGB)", "definition": "FDA Risk Evaluation and Mitigation Strategy for vigabatrin; mandatory prescriber/pharmacy/patient enrolment; ERG monitoring documented"},
    {"term": "SUDEP in TSC", "definition": "Elevated SUDEP risk in drug-resistant TSC with nocturnal FBTCS; nocturnal supervision; SUDEP-7 Inventory; rescue benzodiazepine kit"},
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"param": "Everolimus trough (TSC seizures)", "target": "3-7 ng/mL (escalate to 9-15 if needed)", "action": "<3 → dose up; >20 → hold + recheck"},
    {"param": "Everolimus trough (SEGA)", "target": "5-15 ng/mL", "action": "<5 → uptitrate; check CYP3A4 interactions"},
    {"param": "VPA trough level", "target": "50-100 μg/mL", "action": "<50 → poor efficacy; >120 → toxicity risk"},
    {"param": "AML diameter (renal)", "target": "Action threshold ≥4 cm", "action": "≥4cm → everolimus + urology; embolise if acute bleed"},
    {"param": "SEGA diameter", "target": "≥1 cm growth OR symptomatic", "action": "Start everolimus; neurosurgery if hydrocephalus"},
    {"param": "Fever threshold for AED protocol", "target": "≥37.5°C in TSC children", "action": "Paracetamol/ibuprofen + rescue BZD plan"},
    {"param": "ERG b-wave amplitude reduction (VGB)", "target": ">30% reduction from baseline", "action": "Ophthalmology urgent review; consider VGB dose reduction"},
    {"param": "Seizure diary: clusters ≥5 seizures in 24h", "target": "Alert threshold", "action": "Rescue CLB + everolimus trough recheck"},
    {"param": "LDL (everolimus)", "target": ">3.4 mmol/L (130 mg/dL)", "action": "Statin therapy; everolimus dose-review"},
    {"param": "HbA1c (everolimus)", "target": ">6.5% (diabetic range)", "action": "Endocrinology referral; metformin or dose reduction"},
]

# ── Standards ──────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE-2022 (classification and diagnostic criteria)",
    "NICE NG217 (tuberous sclerosis: diagnosis and management 2022)",
    "TSC International Consensus (Northrup 2013 / Krueger 2013 Pediatr Neurol)",
    "EXIST-3 (Bissler 2017 Lancet): everolimus TSC seizure RCT",
    "EXIST-1 (Franz 2013 NEJM): everolimus SEGA RCT",
    "EXIST-2 (Bissler 2013 NEJM): everolimus renal AML RCT",
    "EPISTOP (Kotulska 2021 Ann Neurol): presymptomatic VGB in TSC",
    "UKISS (O'Callaghan 2017 Lancet Neurol): VGB vs ACTH infantile spasms",
    "SHARE REMS (FDA VGB Risk Evaluation Mitigation Strategy)",
    "CPIC HLA-B*15:02 guideline (CPIC 2023, CBZ/OXC)",
    "MHRA VPPP (Valproate Pregnancy Prevention Programme 2021)",
    "ACMG/AMP variant classification (Richards 2015 Genet Med)",
]

# ── References ─────────────────────────────────────────────────────────────────
REFERENCES = [
    "Northrup 2013 Pediatr Neurol (TSC Consensus Conference diagnostic criteria)",
    "Bissler 2017 Lancet (EXIST-3: everolimus TSC seizure RCT — primary endpoint)",
    "Kotulska 2021 Ann Neurol (EPISTOP: presymptomatic VGB reduces IS in TSC)",
    "Curatolo 2018 Nat Rev Neurol (mTOR in epilepsy — tuberous sclerosis review)",
    "Juhász 2006 Neurology (AMT-PET for epileptogenic tuber identification)",
    "Bhatt 2023 Epilepsia (genetic epilepsy management — cross-gene standards)",
]

# ── Patient generator ─────────────────────────────────────────────────────────
def _generate_patients():
    random.seed(42)
    n = 40
    pts = []
    etiology_pool = (
        ["TSC1-AD-familial"] * 10 +
        ["TSC1-de-novo-severe"] * 8 +
        ["TSC-mosaic"] * 6 +
        ["TSC1-SEGA"] * 8 +
        ["TSC2-phenocopy"] * 8
    )
    random.shuffle(etiology_pool)

    for i in range(n):
        age = random.randint(1, 38)
        etiology = etiology_pool[i]
        has_is = etiology in ("TSC1-de-novo-severe", "TSC1-SEGA") or (
            etiology == "TSC-mosaic" and random.random() < 0.3) or random.random() < 0.25
        has_sega = etiology == "TSC1-SEGA" or random.random() < 0.12
        on_everolimus = random.random() < 0.60
        on_vgb = has_is and age < 5 or random.random() < 0.25
        on_lev = random.random() < 0.55
        on_vpa = random.random() < 0.35
        on_clb = random.random() < 0.30
        on_kd = random.random() < 0.15
        drug_resistant = random.random() < 0.45
        seizure_free = not drug_resistant and random.random() < 0.35
        polg_tested = "Y" if random.random() < 0.80 else "N"
        hla_tested = random.random() < 0.70
        erg_done = on_vgb and random.random() < 0.88
        aml_present = random.random() < 0.65
        aml_size_cm = round(random.uniform(0.5, 6.0), 1) if aml_present else 0
        tand_asd = random.random() < 0.55
        tand_id = random.random() < 0.45
        tuber_count = random.randint(4, 22)
        everolimus_trough = round(random.uniform(3.5, 16.0), 1) if on_everolimus else None
        pts.append({
            "patient_id": f"TSC1-{i+1:03d}",
            "age": age,
            "etiology": etiology,
            "has_infantile_spasms": has_is,
            "has_sega": has_sega,
            "tuber_count": tuber_count,
            "on_everolimus": on_everolimus,
            "everolimus_trough": everolimus_trough,
            "on_vgb": on_vgb,
            "on_lev": on_lev,
            "on_vpa": on_vpa,
            "on_clb": on_clb,
            "on_kd": on_kd,
            "drug_resistant": drug_resistant,
            "seizure_free": seizure_free,
            "polg_tested": polg_tested,
            "hla_b1502_tested": hla_tested,
            "erg_done": erg_done,
            "aml_present": aml_present,
            "aml_size_cm": aml_size_cm,
            "tand_asd": tand_asd,
            "tand_id": tand_id,
        })
    return pts


def get_overview():
    """Return TSC1 overview dict."""
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

    return {
        "gene": "TSC1",
        "locus": "9q34.13",
        "protein": "Hamartin (TSC1/TSC2 complex GAP for Rheb → mTORC1 suppression)",
        "syndrome": "Tuberous Sclerosis Complex (TSC) — mTOR pathway epilepsy",
        "incidence": "~1:6,000–1:10,000 live births",
        "inheritance": "Autosomal dominant (AD); ~70% de novo; ~30% inherited",
        "omim": "OMIM #191100 (TSC1); TSC1: 9q34.13; TSC2: 16p13.3",
        "summary": (
            "TSC1 (9q34.13) encodes Hamartin, a component of the TSC1/TSC2 mTOR "
            "regulatory complex. LOF → mTORC1 hyperactivation → cortical tubers → "
            "epilepsy in 90% of TSC patients. Precision therapy: everolimus (mTORC1 "
            "inhibitor; FDA-approved TSC seizures, SEGA, renal AML). Infantile spasms "
            "(35%): vigabatrin Level A first-line (95% TSC response; SHARE REMS). "
            "Drug-resistant epilepsy 45%; surgical evaluation (SEEG + AMT-PET) for "
            "single epileptogenic tuber. TAND: ASD 55%, ID 45%. Annual renal AML + "
            "brain MRI (SEGA). POLG mandatory before VPA. Tiagabine ABSOLUTE CI."
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
        "tiagabine_alert": "ABSOLUTE CI — NCSE risk (GAT-1 → tonic GABA-A activation); class-effect in ALL TSC patients",
        "everolimus_alert": "PRECISION THERAPY (Level A, EXIST-3). CYP3A4 inducers (CBZ/PHT) reduce trough 2-4× → recheck q2w on any CYP3A4 change",
        "vgb_alert": "VGB FIRST-LINE infantile spasms in TSC (Level A; 95% response). SHARE REMS mandatory. ERG q3M.",
        "polg_alert": "POLG MANDATORY before VPA — Alpers-Huttenlocher hepatic failure risk",
        "sega_alert": "Annual brain MRI for SEGA (foramen of Monro). SEGA ≥1cm growth or symptomatic → everolimus (EXIST-1); neurosurgery if hydrocephalus.",
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
    """Return TSC1 breakdown dict."""
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
    """Return TSC1 definitions dict."""
    return {
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
