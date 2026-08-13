"""
SCN2A-DEE (EIEE11) Dashboard
==============================
41-patient cohort · SCN2A (2q24.3) · Nav1.2 Voltage-Gated Sodium Channel α2 Subunit
SCN2A encephalopathy: pathogenic de novo variant in SCN2A (sodium voltage-gated channel
alpha subunit 2, 2q24.3) causing early-infantile developmental epileptic encephalopathy
(DEE). SCN2A (Nav1.2) is the principal action-potential-generating channel of excitatory
pyramidal neurons; variant functional class (GOF vs LOF) completely reverses treatment
strategy — the most critical biomarker-to-treatment axis in all of genetic epilepsy.

GOF (Gain-of-Function) → EIEE11: persistent Na+ channel opening → neuronal
hyperexcitability → neonatal seizures. TREAT WITH Na-channel blockers (CBZ, OXC, PHT).

LOF (Loss-of-Function) → West syndrome / ASD-DEE / LGS-like: Nav1.2 haploinsufficiency
→ impaired axonal action potential propagation → paradoxically severe epilepsy.
Na-channel blockers are HARMFUL in LOF — they worsen seizures. AVOID CBZ/OXC/PHT.

KEY EEG (GOF): Burst-suppression neonatal (hemisynchronous, contrast STXBP1 which is
asynchronous) → multifocal focal seizures → LVFA (low-voltage fast activity) at seizure
onset. Seizure freedom achievable in ~40-50% GOF with CBZ/OXC.

KEY EEG (LOF): No burst-suppression; hypsarrhythmia-like at 3-15 months (West); slow
spike-wave in LGS-like evolution; background suppression with focal independent spikes.

AED NOTE: GOF = CBZ/OXC first-line (Level B). LOF = LEV/VPA/KD; Na-channel blockers
contraindicated. Phenotype-guided treatment is MANDATORY — variant functional class
must be determined before choosing AEDs.
DISEASE-MODIFYING: SCN2A ASO (antisense oligonucleotide) therapy in clinical trials;
gene therapy programme in early development. Refer to Family SCN2A Foundation registry.
"""

import random
from datetime import datetime

SEED = 9176  # dashboard 176
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "De novo SCN2A GOF severe (EIEE11 / Ohtahara-like neonatal)",
        "n": 16, "pct": 39,
        "category": "De-novo-SCN2A-GOF-severe-EIEE11",
        "mechanism": (
            "Most common and most severe SCN2A variant class in DEE cohorts (~39%): de novo "
            "missense variants causing prominent gain-of-function (GOF) of Nav1.2 — the "
            "channel remains persistently open, fails to inactivate, or has shifted "
            "activation threshold to more hyperpolarised voltages. "
            "Nav1.2 (SCN2A) is expressed predominantly in the axon initial segment (AIS) "
            "and proximal axon of excitatory pyramidal neurons from mid-gestation; it is the "
            "dominant Nav isoform in neonatal cortical neurons (Nav1.6 replaces it distally "
            "by 2-3 weeks post-birth). GOF → sustained depolarisation → repetitive "
            "high-frequency action potential bursting → network seizures. "
            "Key GOF variants (residues in DII-DIII linker and VSD): A263V, R853Q, L1563V, "
            "R1882Q (most frequent worldwide — ~10% of all SCN2A-DEE). "
            "Functional characterisation: whole-cell patch clamp in Xenopus oocyte or "
            "HEK293T cells shows: (a) persistent Na+ current (INaP) elevated 10-30× "
            "above wild-type; (b) negative shift in V½ activation (-10 to -20 mV); "
            "(c) slowed fast inactivation (tau-fast 2-3× prolonged); (d) incomplete slow "
            "inactivation. Combined: channel cannot be turned off after the action potential. "
            "Neonatal expression window: because Nav1.2 is the dominant neonatal AIS channel, "
            "GOF → seizure onset within HOURS to 3 DAYS of life (cf. KCNQ2: within 72h; "
            "STXBP1: 1-5 days). Earliest-onset genetic epilepsy outside KCNQ2-GOF. "
            "Pharmacogenomic consequence: Na-channel blockers (CBZ, OXC, PHT, LTG) stabilise "
            "the INACTIVATED state of Nav1.2 — restoring the normal gating cycle → TREAT GOF "
            "with Na-channel blockers. CBZ achieves seizure FREEDOM in ~40-50% GOF EIEE11 "
            "patients — a remarkable result for any DEE."
        ),
        "eeg_correlate": (
            "GOF EIEE11 EEG features: "
            "Neonatal (0-72 hours): Burst-suppression pattern — HIGH AMPLITUDE bursts "
            "(200-600 µV, poly-spike or LVFA, 2-8 s) alternating with suppression (< 5 µV). "
            "Critically: SCN2A GOF burst-suppression is HEMISYNCHRONOUS (bilaterally "
            "symmetric, simultaneous) — in contrast to STXBP1 which is ASYNCHRONOUS. "
            "Ictal signature: low-voltage fast activity (LVFA, 20-50 Hz) at seizure onset "
            "→ evolves to slow wave discharge; tonic stiffening correlates with LVFA. "
            "Seizure duration: typically 1-3 minutes per event; high frequency (> 10/day) "
            "in first week of life. "
            "Interictal: multifocal independent sharp waves (frontal, central); suppression "
            "intervals > 10 s between bursts in severe cases. "
            "Response to CBZ: EEG normalisation (suppression of burst-suppression → age- "
            "appropriate background) within 24-72h in responders — a diagnostic/therapeutic "
            "tool. Failure to respond to CBZ within 48h: reconsider LOF variant or non-SCN2A."
        ),
        "mri_finding": (
            "GOF SCN2A EIEE11 MRI: "
            "(1) Normal MRI at birth in ~50-60% of GOF cases — do NOT use normal MRI to "
            "exclude SCN2A-DEE. "
            "(2) Delayed myelination: T2 hyperintensity in posterior white matter (corpus "
            "callosum body, posterior limb of internal capsule) — ~25% at 6-18 months; "
            "less severe than STXBP1. "
            "(3) Thin corpus callosum: ~20% of GOF EIEE11 — body and splenium hypoplasia. "
            "(4) Focal cortical signal change: rare in pure GOF — consider concomitant "
            "structural lesion if present (FCD type 2b mimics). "
            "(5) CBZ-responders (seizure freedom): MRI often NORMAL or near-normal — "
            "myelination tracks to age-appropriate norms on serial scans, correlating "
            "with favourable cognitive outcome (IQ 60-80 range achievable). "
            "MRI spectroscopy: typically normal (helps exclude mitochondrial / metabolic)."
        ),
        "clinical_note": (
            "R1882Q is the most frequent SCN2A-GOF variant worldwide (~10% of SCN2A-DEE); "
            "screen all unsolved neonatal onset DEE with hemisynchronous BS + LVFA. "
            "Start rapid genomic workup (WES trio) within 48h of neonatal seizure; "
            "targeted neonatal DEE panel (KCNQ2/SCN2A/STXBP1/SCN8A/ARX) is faster. "
            "Trial CBZ (or OXC) IMMEDIATELY once GOF functionally confirmed or clinically "
            "suspected — do not wait for full WES results if panel positive for GOF variant. "
            "Critical safety: NEVER start CBZ/OXC empirically before variant functional "
            "class is known — LOF worsening is severe and potentially fatal."
        ),
    },
    {
        "etiology": "De novo SCN2A LOF (West / ASD-DEE / LGS-like)",
        "n": 12, "pct": 29,
        "category": "De-novo-SCN2A-LOF-West-ASD-DEE",
        "mechanism": (
            "Second most common SCN2A class (~29%): de novo loss-of-function variants "
            "(nonsense, frameshift, splice-site, large deletion) causing Nav1.2 "
            "haploinsufficiency. LOF is PARADOXICALLY epileptogenic — a counterintuitive "
            "mechanism explained by Nav1.2's developmental expression switch: "
            "Nav1.2 is expressed in INHIBITORY interneurons > excitatory neurons in "
            "early development. Loss of Nav1.2 preferentially reduces INHIBITORY tone "
            "(GABAergic interneuron action potential propagation impaired) → E/I imbalance "
            "→ seizures, even though less Na+ current seems intuitively anti-epileptic. "
            "Variant types: truncating (NMD) ~15%; splice-site ~8%; missense-LOF ~6%. "
            "LOF phenotype is DISTINCT from GOF: "
            "(a) Later onset: 3-15 months (not neonatal) — after Nav1.2 AIS expression "
            "reduces and Nav1.6 takes over, exposing the interneuron haploinsufficiency. "
            "(b) West syndrome / infantile spasms in ~60% of LOF cases. "
            "(c) ASD + intellectual disability (ID) prominent — Nav1.2 LOF is one of the "
            "most common single-gene causes of ASD (Autism Spectrum Disorder). "
            "(d) LGS-like evolution (slow spike-wave + tonic) in ~30% by school age. "
            "TREATMENT REVERSAL: Na-channel blockers (CBZ/OXC/PHT/LTG) further reduce "
            "Nav1.2 function → worsen interneuron dysfunction → severe seizure exacerbation. "
            "AVOID Na-channel blockers in LOF — clinical emergency if inadvertently given."
        ),
        "eeg_correlate": (
            "LOF SCN2A EEG: "
            "No burst-suppression at birth (LOF onset is postnatal, not neonatal-to-fetal). "
            "West syndrome phase (3-15 months): "
            "Hypsarrhythmia or modified hypsarrhythmia (LOF-hyps tends to be more SYNCHRONOUS "
            "than GOF equivalent; interispasm intervals less suppressed). "
            "Ictal: electrodecrement (amplitude attenuation) coinciding with infantile spasms. "
            "Interictal: multifocal sharp waves predominantly centrotemporal and occipital; "
            "slow-wave background with focal enhancement at spasm-cluster onset. "
            "LGS-like evolution (school age, ~30%): "
            "Generalised slow spike-wave (< 2.5 Hz) + paroxysmal fast (10-15 Hz) in NREM "
            "sleep; polyspike-wave during tonic seizures. "
            "ASD overlap: ESES (electrical status epilepticus in sleep) reported in ~15% "
            "of SCN2A-LOF with ASD — overnight EEG important for sleep EEG abnormalities. "
            "Biomarker: Na-channel blocker trial → EEG worsening (increased discharge "
            "frequency) is a RED FLAG for LOF — supports functional class re-evaluation."
        ),
        "mri_finding": (
            "LOF SCN2A MRI: "
            "(1) Frequently NORMAL at first scan — may remain normal even in severe cases. "
            "(2) Thin corpus callosum in ~25% — body and genu; correlates with ASD severity. "
            "(3) Reduced white matter volume / simplified gyral pattern in severe LOF (~15%). "
            "(4) No burst-suppression-associated MRI changes (since there is no neonatal BS). "
            "(5) Hippocampal asymmetry: reported in ~10% — relationship to ASD unclear. "
            "Structural MRI in LOF: purpose is exclusion of focal lesion (FCD — resectable!) "
            "rather than confirmation of LOF diagnosis. Epilepsy protocol MRI (3T, thin-cut, "
            "SWI, post-contrast) recommended."
        ),
        "clinical_note": (
            "LOF SCN2A: delay of 3-15 months before seizure onset — diagnosis more delayed "
            "than GOF; often first presents as West syndrome. "
            "ASD co-diagnosis in ~70% of LOF — neuropsychology and autism assessment at "
            "diagnosis. AVOID CBZ/OXC/PHT — document variant functional class in medical "
            "record with 'NA-CHANNEL BLOCKERS CONTRAINDICATED: LOF SCN2A' alert. "
            "ACTH + VGB for infantile spasms phase (same as cryptogenic IS); KD if ACTH/VGB "
            "insufficient. ASO gene therapy trials: refer eligible LOF patients to registry."
        ),
    },
    {
        "etiology": "De novo SCN2A GOF missense — moderate (neonatal-infantile)",
        "n": 6, "pct": 15,
        "category": "De-novo-SCN2A-GOF-moderate-neonatal-infantile",
        "mechanism": (
            "Third class (~15%): de novo missense GOF variants with PARTIAL or MODERATE "
            "gain-of-function, producing a less severe phenotype than EIEE11. "
            "Characterised by: (a) smaller persistent Na+ current increase (5-15× WT vs. "
            "10-30× in EIEE11); (b) less severe activation shift (-5 to -10 mV vs. -15 "
            "to -20 mV); (c) faster inactivation recovery than severe GOF. "
            "Clinical phenotype: "
            "(a) Onset at 2 days - 3 months (later than severe GOF, earlier than LOF). "
            "(b) Self-limited neonatal epilepsy (SLNE) spectrum in milder variants — "
            "seizures remit within 6 months; may not require long-term AED. "
            "(c) BFNIS (Benign Familial Neonatal-Infantile Seizures) overlap — familial "
            "GOF at moderate level; autosomal dominant; excellent prognosis. "
            "(d) Developmental outcomes: significantly better than EIEE11 — normal/mild ID "
            "in ~70%; ASD in ~20%. "
            "Treatment: CBZ/OXC responsive (~70% respond); seizure freedom in ~60%; "
            "lower dose needed vs. EIEE11 severe GOF. "
            "Prognosis counselling: moderate GOF is a DISTINCT entity from EIEE11 — "
            "families should not be given severe GOF outcome data for moderate variants."
        ),
        "eeg_correlate": (
            "Moderate GOF SCN2A EEG: "
            "Neonatal: Burst-suppression milder — briefer suppression intervals (5-8 s vs. "
            "> 10 s in severe GOF); higher inter-burst amplitude (~20-50 µV vs. < 5 µV). "
            "Rapid improvement on CBZ: background normalisation within 24-48h in responders. "
            "SLNE spectrum: focal seizures (centrotemporal > temporal; bilateral tonic-clonic "
            "onset) remitting by 6 months — EEG normalises; sleep-stage specific spikes "
            "may persist 12-24 months after clinical remission. "
            "Interictal (BFNIS familial): centrotemporal sharp waves that are age-limited; "
            "fully resolve by 5 years in most; no ongoing treatment needed after remission."
        ),
        "mri_finding": (
            "Moderate GOF SCN2A: "
            "Near-normal MRI in > 80% — contrast to severe EIEE11. "
            "Mild delayed myelination in ~10% — resolves on serial scan by 18-24 months. "
            "Normal MRI strongly correlates with good cognitive outcome in moderate GOF. "
            "MRI in BFNIS familial: normal in all affected family members with mild GOF."
        ),
        "clinical_note": (
            "Moderate GOF: reassurance and careful AED choice (CBZ/OXC). "
            "Counsel families: 'moderate GOF SCN2A' is NOT 'EIEE11-severe' — significantly "
            "better prognosis. Document functional class to prevent CBZ over-dosing. "
            "SLNE variant: early CBZ trial — may achieve seizure freedom and taper by 6-12M. "
            "BFNIS familial: segregate family; siblings at 50% risk — surveillance EEG "
            "in neonatal period; rescue plan for febrile seizure cluster."
        ),
    },
    {
        "etiology": "Familial SCN2A (BFNIS / AD-SLNE-DEE spectrum)",
        "n": 4, "pct": 10,
        "category": "Familial-SCN2A-BFNIS-AD-SLNE-DEE",
        "mechanism": (
            "Fourth class (~10%): familial (autosomal dominant) SCN2A variants segregating in "
            "multiple generations — typically benign to moderate GOF. "
            "Spectrum: (a) Classical BFNIS — benign neonatal-infantile seizures, mild GOF, "
            "remit by 12 months, normal development (Heron 2002, Nat Genet). "
            "(b) Familial SCN2A-DEE — rarer; GOF variants with incomplete penetrance; "
            "variable severity within family (some members have BFNIS, others have persistent "
            "DEE). "
            "Genetic mechanism: BFNIS alleles produce GOF with smaller persistent current "
            "than sporadic EIEE11 — explains the favourable prognosis. However, same allele "
            "can produce DEE in some family members (modifier genes / environmental factors). "
            "De novo vs. familial distinction: ~90% of SCN2A-DEE is de novo; 10% familial. "
            "Recurrence risk counselling: if parent affected (BFNIS), 50% recurrence; "
            "if de novo EIEE11, < 1% recurrence (somatic/germline mosaicism ~2%). "
            "Testing: all first-degree relatives of familial SCN2A should receive testing "
            "and EEG in neonatal period — allows pre-emptive CBZ if seizure onset detected."
        ),
        "eeg_correlate": (
            "Familial BFNIS SCN2A EEG: "
            "Neonatal: no burst-suppression (GOF too mild); focal clinical seizures "
            "(tonic/clonic, lateralised) with normal inter-ictal background. "
            "Ictal: rhythmic alpha-theta discharge from centrotemporal region, evolving "
            "bilaterally; duration 1-3 minutes; clustering in first 6 months. "
            "Interictal: normal background; brief centrotemporal sharp waves during drowsiness; "
            "Normalises by 12-24 months in BFNIS. "
            "Familial DEE variant (moderate GOF with DEE in index case): EEG between "
            "BFNIS-mild and EIEE11-severe; variable within family."
        ),
        "mri_finding": (
            "Familial BFNIS SCN2A: MRI normal in all BFNIS cases — "
            "structural MRI primarily to exclude alternative diagnosis. "
            "Familial DEE variant: mild delayed myelination possible in severely-affected "
            "family members; BFNIS-phenotype family members: normal."
        ),
        "clinical_note": (
            "BFNIS: reassurance — benign prognosis; seizures self-limited to first year. "
            "Carbamazepine or oxcarbazepine: first-line; low dose; plan taper at 12 months. "
            "Genetic counselling: 50% recurrence risk; neonatal EEG surveillance in subsequent "
            "pregnancies; written neonatal seizure rescue plan for delivery team. "
            "Important: do not conflate BFNIS with EIEE11-GOF prognosis — different trajectories."
        ),
    },
    {
        "etiology": "Clinical SCN2A-DEE — SCN2A-negative (expanded testing pending)",
        "n": 3, "pct": 7,
        "category": "Clinical-SCN2A-DEE-SCN2A-negative",
        "mechanism": (
            "Residual category (~7%): patients meeting clinical criteria for SCN2A-DEE "
            "(neonatal hemisynchronous BS + LVFA ictal pattern + CBZ response or failure) "
            "but with non-diagnostic standard SCN2A sequencing. "
            "Differential diagnoses to actively exclude: "
            "(a) SCN2A deep intronic variant — RNA-seq or long-read WGS required; "
            "~5-8% of unsolved SCN2A-DEE may have deep intronic pseudo-exon insertions. "
            "(b) SCN2A mosaic — ultra-deep sequencing (> 500×) of blood and/or fibroblast; "
            "somatic mosaic GOF can produce isolated focal epilepsy without family history. "
            "(c) Other neonatal BS DEEs mimicking SCN2A-GOF: KCNQ2-GOF (hemisynchronous "
            "BS, CBZ not first-line), STXBP1-DEE (asynchronous BS), SCN8A-EIEE13 "
            "(later onset, B2-LVFA ictal signature). "
            "(d) Metabolic: pyridoxine-dependent epilepsy (ALDH7A1) — empiric IV pyridoxine "
            "100 mg trial with EEG mandatory in NICU. "
            "(e) Structural: FCD type 2b (bilateral tonic seizures, LVFA ictal). "
            "Approach: full genomic workup (WES trio → RNA-seq → WGS); parallel metabolic "
            "screen; epilepsy protocol MRI; pyridoxine trial."
        ),
        "eeg_correlate": (
            "SCN2A-negative neonatal DEE with GOF-like EEG: "
            "Hemisynchronous BS + LVFA ictal: prioritise KCNQ2 (different CBZ response, "
            "OXC is first-line for KCNQ2-GOF); SCN8A (later onset, more diffuse evolution). "
            "CBZ response without confirmed GOF variant: consider empiric CBZ/OXC trial "
            "under specialist supervision — if EEG normalises within 48h, supports GOF "
            "mechanism even if variant not yet classified. "
            "Pyridoxine EEG trial (100 mg IV): EEG normalisation within 30 minutes = "
            "pyridoxine-dependent epilepsy (PDX-DE, ALDH7A1). MANDATORY in NICU."
        ),
        "mri_finding": (
            "SCN2A-negative neonatal DEE: "
            "MRI is the pivotal discriminator for structural causes (FCD 2b): "
            "look for subtle cortical thickening + T2 signal + blurring of GM-WM junction. "
            "If MRI normal + BS + CBZ-responsive: support for SCN2A GOF deep intronic "
            "or mosaic variant; proceed to RNA-seq. "
            "Metabolic MRI pattern (bilateral basal ganglia T2, diffuse WM): exclude "
            "mitochondrial, organic aciduria before genetic result."
        ),
        "clinical_note": (
            "SCN2A-negative: mandatory NICU screen — pyridoxine IV trial + CSF glucose "
            "(GLUT1) + plasma amino acids + urine organic acids + biotinidase. "
            "Empiric CBZ/OXC trial (with EEG) if clinical/EEG phenotype strongly suggests "
            "GOF SCN2A — early response (< 48h EEG normalisation) is clinically meaningful "
            "even without confirmed variant. Refer to specialist DEE MDT for WGS + RNA-seq."
        ),
    },
]

# ── Seizure Types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Tonic / Focal Neonatal Seizures (GOF)",
        "prevalence_pct": 88,
        "eeg_correlate": (
            "LVFA (Low-Voltage Fast Activity) ictal onset at 20-50 Hz from frontal or "
            "centrotemporal regions, evolving over 5-20 seconds to slow wave discharge; "
            "hemisynchronous burst-suppression between events. Clinically: tonic "
            "stiffening or focal clonic; autonomic features (tachycardia, apnoea) common "
            "in severe GOF neonates. Electroclinical dissociation (ECD) less common than "
            "in STXBP1-DEE (~30% after PB, vs. > 60% in STXBP1) — clinical manifestations "
            "better correlate with EEG in SCN2A-GOF."
        ),
        "clinical_tip": (
            "GOF: CBZ is FIRST-LINE — 5-10 mg/kg/day in 2 doses; titrate to 15-20 mg/kg/day "
            "if tolerated; TDM target 4-12 µg/mL. Response expected within 24-72h (EEG "
            "normalisation). If CBZ unavailable: OXC 10-20 mg/kg/day; PHT 20 mg/kg IV load. "
            "NEVER give CBZ/OXC without confirmed functional class — fatal worsening in LOF. "
            "PB as acute bridge ONLY (20 mg/kg IV) while awaiting CBZ oral absorption. "
            "cEEG monitoring mandatory — treat to EEG suppression, not just clinical cessation."
        ),
    },
    {
        "type": "Focal-to-Bilateral Tonic-Clonic Seizures (FBTCS)",
        "prevalence_pct": 72,
        "eeg_correlate": (
            "GOF: LVFA focal onset → rapid bilateral evolution; centrotemporal or frontal "
            "origin most common; seizure duration 1-4 minutes. Post-ictal voltage "
            "attenuation (< 20 µV) for 20-60 s followed by burst-suppression resumption "
            "in neonates. "
            "LOF: FBTCS with slower onset; generalised poly-spike (3-5 Hz) preceding "
            "tonic phase; more LGS-like in older LOF patients. "
            "Both types: sensitive to sleep deprivation — spike density increases in "
            "NREM sleep on overnight EEG."
        ),
        "clinical_tip": (
            "GOF: CBZ/OXC maintenance eliminates FBTCS in ~50% responders; "
            "rescue: buccal midazolam 0.3 mg/kg (> 6 months). "
            "LOF: LEV 20-60 mg/kg/day or VPA 20-40 mg/kg/day (POLG excluded); "
            "KD for drug-resistant FBTCS in LOF. "
            "Fever plan: written emergency protocol — FBTCS triggered by fever in ~75%; "
            "antipyretics at 37.5°C; rescue BZD plan for clusters (3+ seizures in 24h)."
        ),
    },
    {
        "type": "Infantile Spasms / West Syndrome (LOF)",
        "prevalence_pct": 42,
        "eeg_correlate": (
            "LOF-predominant seizure type (onset 3-15 months — NEVER neonatal in LOF). "
            "Hypsarrhythmia: high-amplitude, disorganised spike-wave; less asymmetric "
            "than STXBP1-modified hypsarrhythmia; often more synchronous in LOF SCN2A. "
            "Ictal: electrodecrement at spasm onset — amplitude attenuation 50-80%; "
            "spasm cluster: 5-25 spasms on awakening. "
            "EEG WARNING: if infantile spasms are present AND Na-channel blocker prescribed "
            "(unrecognised LOF) → EEG worsening (increased spike frequency, new focal "
            "discharges) within 24-72h = urgent AED withdrawal + replacement with ACTH/VGB."
        ),
        "clinical_tip": (
            "LOF infantile spasms: ACTH (Tetracosactide 0.5 mg IM alternate days × 14 days) "
            "or prednisolone (40 mg/day × 14 days) FIRST-LINE — UKISS protocol. "
            "Add VGB (50-150 mg/kg/day) per UKISS combination arm — SHARE REMS mandatory. "
            "KD: early consideration (before/instead of 2nd-line AED) — KD + ACTH more "
            "effective than AED polypharmacy in LOF West. "
            "CRITICAL: NEVER give CBZ/OXC for LOF infantile spasms — document 'LOF: "
            "Na-channel blockers CONTRAINDICATED' prominently in prescribing system."
        ),
    },
    {
        "type": "Myoclonic / Myoclonic-Atonic Seizures (LOF)",
        "prevalence_pct": 28,
        "eeg_correlate": (
            "LOF-predominant in school-age LGS-like evolution: "
            "Generalised polyspike-wave (3-4 Hz) bilaterally synchronous; brief (0.5-2 s) "
            "correlated with myoclonic jerk; negative myoclonus (atonia interrupting "
            "sustained posture) seen in ~10% — centroparietal spike-wave correlate. "
            "Myoclonic-atonic (MAE/Doose-like) in LOF SCN2A: drop attacks; injury risk; "
            "careful for confusion with tonic seizures (different AED implications). "
            "GOF patients: myoclonus uncommon (< 10%) — if present on CBZ, consider "
            "CBZ dose reduction or switch to OXC (fewer myoclonic side effects). "
            "EEG differentiation: myoclonic = poly-spike-wave; tonic = LVFA/fast activity."
        ),
        "clinical_tip": (
            "LOF myoclonic-atonic: VPA (20-40 mg/kg/day, POLG excluded) + clobazam adjunct; "
            "KD (4:1 classic) highly effective — seizure freedom in ~15-20% of MAE-LOF. "
            "Clonazepam 0.05-0.2 mg/kg/day for myoclonic type. "
            "Avoid Na-channel blockers (LTG can worsen myoclonus in LOF) — "
            "lamotrigine contraindicated in LOF SCN2A myoclonic component. "
            "Protective equipment (helmet) for drop attacks; physiotherapy assessment "
            "for ataxia commonly associated with LOF myoclonic-atonic type."
        ),
    },
]

# ── Seizure Triggers ──────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / febrile illness", "prevalence_pct": 82,
     "note": "Fever reduces CBZ plasma levels (distribution volume increase) AND increases "
             "Na+ channel activation — double-whammy for GOF patients. Aggressive antipyretics "
             "at 37.5°C; rescue midazolam/diazepam plan for febrile clusters; check CBZ "
             "TDM after febrile illness (may need dose increase)."},
    {"trigger": "Missed / late AED dose", "prevalence_pct": 75,
     "note": "CBZ has shorter half-life than VPA/LEV — missed dose → rapid plasma drop; "
             "twice-daily modified-release CBZ preferred over immediate-release to reduce "
             "trough-level troughs. Written dose schedule mandatory; hospital NPO protocol "
             "MUST include IV phenytoin substitution for oral CBZ if NPO > 12h."},
    {"trigger": "Intercurrent illness (GI, viral)", "prevalence_pct": 68,
     "note": "GI illness reduces CBZ absorption; IV PHT 15 mg/kg can substitute in hospital "
             "for oral CBZ if vomiting. LEV IV available for LOF patients. "
             "Acute illness: increase monitoring frequency (CBZ level at 48h if illness)."},
    {"trigger": "Sleep deprivation", "prevalence_pct": 55,
     "note": "SCN2A seizures (both GOF and LOF) worsen with sleep deprivation — spike density "
             "in NREM sleep elevated. Melatonin (0.5-5 mg nocte) for sleep consolidation; "
             "strict sleep schedule; avoid overnight travel / night shifts in adolescents."},
    {"trigger": "Rapid CBZ dose reduction or withdrawal (GOF)", "prevalence_pct": 45,
     "note": "CBZ withdrawal seizure exacerbation in GOF — never abrupt taper. "
             "Minimum 6-week taper per dose reduction step; seizure-free 2+ years before "
             "taper discussion. LOF patients: equivalent caution for VPA/LEV withdrawal."},
    {"trigger": "Hyperthermia (bath, heat)", "prevalence_pct": 38,
     "note": "Same mechanism as fever — temperature-dependent Nav1.2 GOF activation. "
             "Lukewarm baths only (< 37°C); avoid prolonged sun exposure; air conditioning "
             "in summer; written heat emergency plan."},
    {"trigger": "Inadvertent Na-channel blocker (in LOF patients)", "prevalence_pct": 30,
     "note": "LOF SCN2A patients given CBZ/OXC/PHT/LTG → acute seizure exacerbation "
             "within 24-72h. Prescribers must be alerted via allergy/drug alert system: "
             "'SODIUM CHANNEL BLOCKERS CONTRAINDICATED: SCN2A LOF'. Document in patient "
             "summary card; Medic Alert bracelet for LOF patients."},
    {"trigger": "Puberty / hormonal change (catamenial)", "prevalence_pct": 20,
     "note": "Catamenial exacerbation in adolescent females (both GOF and LOF); "
             "progesterone modulates Na+ channel expression. CBZ + clobazam pulsed "
             "(perimenstrual) in GOF catamenial exacerbation."},
]

# ── Treatment Catalog ─────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Carbamazepine (CBZ) — GOF SCN2A FIRST-LINE",
        "evidence": "Level B — Specific GOF SCN2A efficacy; first-line for GOF EIEE11 and moderate GOF",
        "dose": "Start: 5 mg/kg/day in 2 divided doses; titrate to 15-25 mg/kg/day; "
                "MR formulation preferred; max 35 mg/kg/day in neonates (weight-based); "
                "children: 10-35 mg/kg/day; TDM target 4-12 µg/mL",
        "moa": "Voltage-dependent Na+ channel inactivation state stabiliser — binds to the "
               "inactivated (closed, non-conducting) state of Nav1.2, prolonging the refractory "
               "period and preventing repetitive firing. In GOF SCN2A (persistent current / "
               "failed inactivation), CBZ restores the normal gating cycle by 'rescuing' "
               "inactivation. Dose-dependent — higher doses needed for EIEE11-severe GOF "
               "vs. moderate GOF (proportional to persistent current magnitude). "
               "CONTRAINDICATED IN LOF SCN2A — reduces Nav1.2 total function further.",
        "efficacy": (
            "GOF EIEE11: seizure FREEDOM in ~40-50% (remarkable for any DEE); > 50% "
            "seizure reduction in ~70%. Response within 24-72h (EEG normalisation). "
            "Moderate GOF: ~70% achieve seizure freedom. CBZ is the closest to a precision "
            "medicine treatment in all of genetic epilepsy — variant-specific, predictable, "
            "rapid-onset response. Long-term: continued efficacy in most responders; "
            "dose increase needed with growth."
        ),
        "safety": "DERMATOLOGICAL: SJS/TEN risk — HLA-B*1502 mandatory screening in Asian "
                  "patients (CPIC Level A); avoid CBZ if HLA-B*1502 positive. "
                  "Hyponatraemia (SIADH) — sodium monitoring q4 weeks (CBZ > OXC risk). "
                  "Hepatic enzyme induction (CYP3A4/2C9/1A2) — reduces levels of co-meds "
                  "(OCP, warfarin, other AEDs). Dizziness, diplopia, ataxia at high doses. "
                  "Cardiac: AV block at toxic levels — ECG if bradycardia/syncope.",
        "monitoring": "CBZ TDM 4-12 µg/mL (trough, pre-dose); HLA-B*1502 before starting "
                      "(Asian patients — CPIC Level A mandatory); Na+ q4 weeks (SIADH); "
                      "LFT at baseline + 6 months (enzyme induction); ECG if cardiac history.",
    },
    {
        "drug": "Oxcarbazepine (OXC) — GOF SCN2A Alternative / Infant-preferred",
        "evidence": "Level B — GOF SCN2A (preferred over CBZ in infants — fewer drug interactions)",
        "dose": "Start: 10 mg/kg/day in 2 doses; titrate to 20-40 mg/kg/day; "
                "TDM: MHD (monohydroxy-derivative) 12-24 µg/mL; suspension available",
        "moa": "Prodrug (hepatic metabolism to MHD); MHD stabilises Na+ channel inactivated state "
               "— same mechanism as CBZ but slightly different binding kinetics. "
               "HIGHER HYPONATRAEMIA risk than CBZ (SIADH more frequent). "
               "LESS enzyme induction than CBZ — preferred in infants on multiple medications. "
               "CONTRAINDICATED IN LOF SCN2A — same LOF worsening mechanism as CBZ.",
        "efficacy": "Similar to CBZ in GOF SCN2A; some patients who are CBZ-intolerant (diplopia, "
                    "ataxia) tolerate OXC. Seizure freedom in ~40-45% GOF EIEE11; ~65% moderate GOF.",
        "safety": "HYPONATRAEMIA: more frequent than CBZ — Na+ monitoring q4 weeks mandatory; "
                  "risk higher in infants < 6 months; switch to CBZ if Na < 130 mEq/L and "
                  "clinically symptomatic. HLA-B*1502: cross-reactivity with CBZ (CPIC Level A — "
                  "screen before OXC in Asian patients). Less enzyme induction than CBZ.",
        "monitoring": "MHD TDM 12-24 µg/mL; Na+ q4 weeks (mandatory — SIADH risk); "
                      "HLA-B*1502 (CPIC Level A, Asian patients); weight/growth monthly.",
    },
    {
        "drug": "Phenytoin (PHT) — Acute NICU GOF bridge",
        "evidence": "Level B — Acute neonatal seizure control (NICU bridge to oral CBZ); NOT for maintenance",
        "dose": "Loading: 20 mg/kg IV at max 1 mg/kg/min (cardiac monitor); "
                "Maintenance: 5-8 mg/kg/day IV/PO; TDM 10-20 µg/mL",
        "moa": "Na+ channel inactivated-state stabiliser (same class as CBZ/OXC but different "
               "binding pocket). IV formulation provides rapid GOF seizure control in NICU "
               "when oral CBZ not yet absorbed. CONTRAINDICATED IN LOF SCN2A (worsens LOF).",
        "efficacy": "Rapid IV onset (10-20 min) — useful in acute NICU management of severe GOF "
                    "EIEE11. Limited long-term use due to gingival hypertrophy, cerebellar "
                    "atrophy, enzyme induction. Bridge to oral CBZ only.",
        "safety": "Cardiac: AV block / bradyarrhythmia during IV bolus — rate limit < 1 mg/kg/min; "
                  "cardiac monitor mandatory. Tissue necrosis if extravasation (IV only via central "
                  "line or large-bore peripheral). Long-term: cerebellar atrophy, gingival "
                  "hypertrophy, enzyme induction, osteopenia. AVOID FOR MAINTENANCE.",
        "monitoring": "Cardiac monitor during IV loading; TDM 10-20 µg/mL; ECG if bradycardia; "
                      "do NOT use long-term — transition to CBZ within 48-72h.",
    },
    {
        "drug": "Phenobarbital (PB) — Neonatal bridge (non-specific)",
        "evidence": "Level B — First-line neonatal acute seizure control (bridge to CBZ in GOF; "
                    "or bridge to ACTH/VGB in LOF before diagnosis confirmed)",
        "dose": "Loading: 20 mg/kg IV over 20 min; Maintenance: 3-5 mg/kg/day; "
                "TDM 20-40 µg/mL (neonates); 10-40 µg/mL (older)",
        "moa": "GABA-A positive allosteric modulator (barbiturate) — non-specific, not targeted "
               "to Nav1.2. Used as bridge in NICU before GOF vs. LOF variant class confirmed "
               "and before CBZ oral absorption established. Electroclinical dissociation "
               "reported in ~30% after PB in SCN2A-DEE (less than STXBP1 ~60%).",
        "efficacy": "Partial seizure suppression in neonatal SCN2A-GOF (50-60%) — sufficient "
                    "as short-term NICU bridge. NOT sufficient alone for GOF maintenance "
                    "— transition to CBZ/OXC as soon as variant class confirmed.",
        "safety": "Sedation, respiratory depression at high doses; enzyme induction (CYP enzymes); "
                  "reduces CBZ levels if co-administered — monitor CBZ TDM after PB initiation.",
        "monitoring": "TDM 20-40 µg/mL (neonates); respiratory monitor during IV loading; "
                      "wean PB gradually after CBZ established (overlap 2 weeks minimum).",
    },
    {
        "drug": "Levetiracetam (LEV) — LOF adjunct",
        "evidence": "Level C — Adjunct for LOF SCN2A (broad-spectrum; does NOT affect Nav1.2)",
        "dose": "20-60 mg/kg/day PO/IV in 2 divided doses; max 3 g/day",
        "moa": "SV2A ligand — presynaptic vesicle release modulation; no Na+ channel effect "
               "→ SAFE in LOF SCN2A (does not reduce Nav1.2 function). Good adjunct for "
               "focal seizures and myoclonic type in LOF patients.",
        "efficacy": "20-35% responder rate (> 50% reduction) in LOF SCN2A focal seizures; "
                    "limited efficacy for infantile spasms as monotherapy; IV formulation "
                    "useful in acute LOF management when oral route unavailable.",
        "safety": "Behavioural side effects (irritability, agitation) 10-20%; "
                  "hyponatraemia rare; no enzyme induction; renal dose adjustment (eGFR < 30). "
                  "Switch to brivaracetam if LEV behavioural problems intolerable.",
        "monitoring": "TDM 12-46 mg/L; behavioural CBCL assessment at 4 weeks; renal function "
                      "6-monthly. SAFE to use in LOF SCN2A — no worsening of LOF mechanism.",
    },
    {
        "drug": "Valproate (VPA) — LOF broad-spectrum (POLG excluded)",
        "evidence": "Level C — LOF SCN2A broad-spectrum (after POLG exclusion); NOT for GOF",
        "dose": "20-40 mg/kg/day PO in 2-3 divided doses; TDM 50-100 mg/L; "
                "ER formulation for compliance; IV if oral unavailable",
        "moa": "Multiple mechanisms: T-type Ca2+ channel block; GABA transaminase inhibition "
               "(increases GABA); Na+ channel inhibition (MINOR component — lower affinity than "
               "CBZ/OXC; less LOF-worsening risk than CBZ but still use with caution in LOF). "
               "Broad-spectrum coverage: myoclonic, atonic, absence, GTCS. ",
        "efficacy": "LOF SCN2A: 25-35% responder rate for myoclonic-atonic and GTCS; "
                    "less effective than KD for drug-resistant LOF. "
                    "GOF: NOT recommended — insufficient Nav1.2 inactivation vs. CBZ; "
                    "may worsen if LOF component present.",
        "safety": "POLG EXCLUSION MANDATORY before VPA (fatal Alpers hepatotoxicity). "
                  "Teratogenicity (spina bifida, NTD) — REMS programme in some countries; "
                  "avoid in women of childbearing age if alternatives exist. "
                  "Hepatotoxicity (LFT + ammonia monitoring); weight gain; tremor; "
                  "thrombocytopaenia; drug interactions (CYP inhibition — raises CBZ-epoxide).",
        "monitoring": "VPA TDM 50-100 mg/L; LFT + ammonia q6 months; POLG exclusion before start; "
                      "platelet count at baseline + 6 months; weight monthly.",
    },
    {
        "drug": "Ketogenic Diet (KD, 4:1 classic) — LOF DRE",
        "evidence": "Level B — Drug-resistant LOF SCN2A (myoclonic-atonic / West-residual / LGS-like)",
        "dose": "4:1 fat:(protein+carb) ratio; BHB target 2-4 mmol/L; "
                "MCT diet alternative; hospitalised initiation; G-tube if swallowing unsafe",
        "moa": "Ketone bodies (BHB, AcAc) — direct Nav channel inhibition (minor; beneficial in "
               "LOF as reduces residual excitatory transmission); GABA-B upregulation; "
               "purinergic (adenosine A1) anti-seizure effect; reduces glycolytic flux. "
               "KD does NOT selectively modulate Nav1.2 — broad anti-seizure mechanism "
               "without the LOF-worsening seen with Na-channel blockers.",
        "efficacy": "LOF SCN2A DRE: > 50% seizure reduction in ~50-60%; seizure freedom ~15-20%. "
                    "Particularly effective for myoclonic-atonic and infantile spasm residual. "
                    "GOF SCN2A: less evidence; consider if CBZ insufficient (may combine KD + CBZ).",
        "safety": "GI side effects (nausea, constipation); growth deceleration; "
                  "selenium/zinc/carnitine deficiency; renal calculi (citrate supplementation); "
                  "hyperlipidaemia; cardiomyopathy (selenium-deficient). KD + CBZ: no major "
                  "interaction but monitor lipids (CBZ enzyme induction may affect KD fats).",
        "monitoring": "BHB twice weekly (target 2-4 mmol/L); lipids q6 months; micronutrients "
                      "(Se, Zn, carnitine, vitamin D) annually; DEXA annually; growth monthly.",
    },
    {
        "drug": "Antisense Oligonucleotide (ASO) — Investigational LOF/GOF",
        "evidence": "Phase 1/2 — SCN2A-specific ASO gene modulation therapy (research)",
        "dose": "Investigational; intrathecal administration in current trials; "
                "eligibility: confirmed pathogenic SCN2A variant; enrol via Family SCN2A "
                "Foundation / ClinicalTrials.gov (NCT numbers updated annually)",
        "moa": "LOF: ASO upregulates SCN2A expression from wild-type allele (NMD rescue or "
               "enhancer approach) — restores Nav1.2 protein level toward normal. "
               "GOF: ASO reduces expression of GOF allele (allele-selective silencing) "
               "— reduces persistent Na+ current while preserving WT Nav1.2. "
               "Precision medicine: different ASO molecule for LOF vs. GOF — "
               "variant class determination mandatory before trial enrolment.",
        "efficacy": "Phase 1 (safety/PK) ongoing — no clinical efficacy data available 2026; "
                    "pre-clinical (Scn2a mouse): significant seizure reduction and "
                    "behavioural improvement. Phase 2 efficacy expected 2027-2028.",
        "safety": "Phase 1 safety profile under evaluation; intrathecal administration risks "
                  "(CSF pleocytosis, meningitis); trial-specific monitoring protocol.",
        "monitoring": "Clinical trial protocol — refer eligible patients to Family SCN2A "
                      "Foundation registry for trial access and safety monitoring protocol.",
    },
]

# ── Absolute Contraindications ────────────────────────────────────────────────
ABSOLUTE_CONTRAINDICATIONS = [
    {
        "drug": "CBZ / OXC / PHT / LTG in LOF SCN2A patients",
        "severity": "ABSOLUTE CI — may cause acute severe seizure exacerbation (LOF worsening)",
        "reason": (
            "Na-channel blockers (carbamazepine, oxcarbazepine, phenytoin, lamotrigine) "
            "reduce available Nav1.2 current — in LOF haploinsufficiency, this further "
            "impairs interneuron action potential propagation, worsening E/I imbalance and "
            "precipitating severe seizure exacerbation within 24-72h of initiation. "
            "Clinical emergency: if a LOF SCN2A patient inadvertently receives CBZ/OXC "
            "and seizes more — IMMEDIATELY withdraw Na-channel blocker, start diazepam "
            "IV bridge, and initiate appropriate LOF treatment (ACTH/VGB/KD/LEV/VPA). "
            "Prevention: document 'LOF SCN2A: Na-CHANNEL BLOCKERS ABSOLUTELY CI' in "
            "prescribing system allergy field; Medic Alert bracelet for LOF patients; "
            "written card in patient summary for all clinical settings."
        ),
    },
    {
        "drug": "Valproate (VPA) — WITHOUT POLG EXCLUSION",
        "severity": "ABSOLUTE CI (until POLG excluded) — fatal hepatotoxicity risk",
        "reason": (
            "POLG (mitochondrial DNA polymerase gamma) mutations cause Alpers-Huttenlocher "
            "syndrome (AHS), which is FATAL with valproate exposure (acute liver failure). "
            "POLG-positive epilepsy can clinically resemble SCN2A-LOF DEE. "
            "POLG panel (or WES with POLG analysis) is MANDATORY before VPA in any child "
            "with unexplained DEE. Cannot be excluded by clinical features alone. "
            "Once POLG excluded: VPA may be used with caution in LOF SCN2A."
        ),
    },
    {
        "drug": "Empiric Na-channel blocker without variant functional classification",
        "severity": "ABSOLUTE CI — must confirm GOF vs. LOF before prescribing CBZ/OXC/PHT",
        "reason": (
            "The GOF-vs-LOF distinction determines OPPOSITE treatment strategies. "
            "Starting CBZ/OXC empirically before variant functional class is confirmed "
            "risks ACUTE SEIZURE WORSENING in LOF patients (30% of SCN2A-DEE is LOF). "
            "Exception: NICU with strong clinical/EEG evidence of GOF (hemisynchronous BS + "
            "LVFA ictal + onset < 3 days) and functional prediction pending — careful "
            "empiric CBZ trial under continuous EEG monitoring with immediate discontinuation "
            "plan if EEG worsens. THIS IS A SPECIALIST DECISION only."
        ),
    },
    {
        "drug": "Hospital NPO without IV/NG AED substitution",
        "severity": "ABSOLUTE OPERATIONAL CI",
        "reason": (
            "SCN2A-GOF patients on oral CBZ made NPO (surgery, procedures) without IV "
            "phenytoin or fosphenytoin substitution are at high risk of status epilepticus "
            "(CBZ half-life ~12-36h; levels fall rapidly without enteral dosing). "
            "LOF patients on oral LEV/VPA: IV LEV available as direct substitution; "
            "IV VPA available if enteral route lost. "
            "Anaesthesia/surgery teams MUST have a written AED substitution protocol "
            "documented in the surgical consent checklist. Do NOT omit AEDs perioperatively."
        ),
    },
]

# ── AED Monitoring ─────────────────────────────────────────────────────────────
AED_MONITORING = [
    {"item": "CBZ TDM (carbamazepine level)", "target": "4-12 µg/mL (trough, pre-dose)",
     "frequency": "At day 5-7 after initiation; after dose changes; every 6 months stable",
     "rationale": "Narrow therapeutic window; auto-induction at 2-4 weeks reduces own levels — "
                  "recheck TDM 4 weeks after starting CBZ"},
    {"item": "OXC / MHD TDM (oxcarbazepine metabolite)", "target": "MHD 12-24 µg/mL",
     "frequency": "At day 7-14 after initiation; after dose changes; q6 months",
     "rationale": "Active metabolite MHD drives efficacy and SIADH risk — guide dose titration"},
    {"item": "Sodium (Na+) — SIADH monitoring", "target": "Na+ > 135 mEq/L",
     "frequency": "Baseline; then every 4 weeks on CBZ/OXC; after illness",
     "rationale": "CBZ and OXC cause SIADH; Na < 130 symptomatic = dose reduce or switch; "
                  "higher risk in infants and in combination with other SIADH-promoting drugs"},
    {"item": "HLA-B*1502 genotyping (CPIC Level A)", "target": "Negative before CBZ/OXC start",
     "frequency": "Once (genotype does not change) — mandatory in Asian patients before CBZ/OXC",
     "rationale": "HLA-B*1502 (common in Han Chinese, Thai, Vietnamese, Malay) associates with "
                  "CBZ/OXC-induced SJS/TEN; CPIC Guideline Level A: avoid CBZ/OXC if positive"},
    {"item": "Liver function tests (LFT — CBZ induction)", "target": "< 3× ULN",
     "frequency": "Baseline; 6 weeks after starting CBZ; then 6-monthly",
     "rationale": "CBZ is a potent CYP inducer; transaminase rise common in first 6 weeks "
                  "(hepatic adaptation); > 3× ULN with symptoms: investigate + dose review"},
    {"item": "Neurodevelopmental assessment (Bayley-III / WPPSI)", "target": "Track trajectory",
     "frequency": "6-monthly (age 0-3y); annually (3-18y)",
     "rationale": "SCN2A-DEE cognitive outcome highly variable (normal-to-severe ID); "
                  "trajectory informs educational planning and treatment intensity decisions"},
    {"item": "VPA TDM (if VPA used in LOF)", "target": "50-100 mg/L",
     "frequency": "Day 7-14 after initiation; q6 months; after dose change",
     "rationale": "VPA narrow therapeutic window; sub-therapeutic = seizure breakthrough; "
                  "supra-therapeutic = tremor, hepatotoxicity"},
    {"item": "EEG (sleep + wake, 2h minimum)", "target": "Seizure burden / background improvement",
     "frequency": "At 4 weeks after major AED change; 6-monthly; after clinical concerns",
     "rationale": "GOF: EEG normalisation confirms CBZ response; LOF: EEG trajectory informs "
                  "ACTH response (day 14 UKISS criterion for IS)"},
]

# ── Lifecycle Windows ─────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {"window": "Neonatal / NICU (GOF)", "age_range": "0–28 days",
     "focus": "Hemisynchronous BS + LVFA ictal; PB bridge; rapid genomic panel (SCN2A/KCNQ2/STXBP1); "
              "POLG exclusion; HLA-B*1502 testing; start CBZ once GOF confirmed; cEEG monitoring",
     "key_action": "PB bridge → CBZ/OXC once GOF confirmed; cEEG; HLA-B*1502; rapid genomics"},
    {"window": "Early Infantile (LOF)", "age_range": "1–6 months",
     "focus": "Infantile spasms onset (LOF patients — NO BS in neonatal period for LOF); "
              "ACTH + VGB UKISS protocol; EEG day 14 assessment; VGB SHARE REMS; "
              "AVOID Na-channel blockers in LOF; early KD consideration",
     "key_action": "ACTH + VGB (LOF); EEG day 14; SHARE REMS enrolment; CBZ (GOF) stable dose"},
    {"window": "Late Infantile", "age_range": "6–18 months",
     "focus": "GOF: CBZ dose titration with growth; TDM q3 months; speech/motor assessment. "
              "LOF: post-IS multifocal epilepsy; KD initiation; ASD screening (M-CHAT); "
              "OT/PT/SLP referral; VGB VFD monitoring q3 months (SHARE REMS)",
     "key_action": "GOF: CBZ TDM + growth; LOF: ASD screen + KD + OT/PT/SLP"},
    {"window": "Early Childhood", "age_range": "18 months–5 years",
     "focus": "GOF CBZ responders: often stable seizure freedom — reassess taper at 2Y seizure-free. "
              "GOF non-responders / LOF DRE: LEV + KD combination; LGS-like evolution in ~30% LOF; "
              "VNS referral after 3 AED failures; SUDEP counselling; feeding/G-tube assessment",
     "key_action": "GOF: CBZ taper discussion at 2Y seizure-free; LOF: VNS if 3+ AED failures"},
    {"window": "School Age", "age_range": "5–12 years",
     "focus": "GOF: majority seizure-free on CBZ — annual EEG + TDM; normal-borderline cognition "
              "in ~60% GOF responders; special education if ID. LOF: severe GDD in ~70%; "
              "annual EEG + neurodevelopmental assessment; SUDEP risk ongoing; "
              "adolescent transition planning; Medic Alert bracelet documentation (LOF)",
     "key_action": "SUDEP counselling; nocturnal alarm; annual EEG + TDM; school IEP planning"},
    {"window": "Adolescence / Adult", "age_range": "12+ years",
     "focus": "Adult epilepsy transition; reproductive counselling (< 1% recurrence for de novo); "
              "CBZ teratogenicity counselling (GOF females); driving exclusion (LOF DRE); "
              "BFNIS familial: taper consideration after 2-5Y seizure-free; "
              "SCN2A ASO gene therapy trial eligibility; Family SCN2A Foundation registry",
     "key_action": "Adult transition; reproductive counselling; driving assessment; ASO trial enrolment"},
]

# ── Clinical Standards ─────────────────────────────────────────────────────────
STANDARDS = [
    {"std": "ILAE 2022", "title": "Classification of Seizures and Epilepsy Syndromes",
     "note": "SCN2A-DEE: classified as DEE; EIEE11 (MIM #613721) for GOF severe; "
             "LOF: West syndrome etiology, ASD-DEE, LGS-like DEE"},
    {"std": "NICE NG217 2022", "title": "Epilepsies: Diagnosis and Management (UK)",
     "note": "Genetic DEE pathway; ACTH for infantile spasms; SUDEP §1.15; "
             "specialist referral within 4 weeks of IS onset"},
    {"std": "CPIC CBZ/OXC + HLA-B*1502 2023", "title": "Clinical Pharmacogenomics Guideline (Level A)",
     "note": "Level A: avoid CBZ/OXC if HLA-B*1502 positive (Asian patients) — "
             "SJS/TEN risk; mandatory genotyping before first prescription"},
    {"std": "UKISS Trial 2004 (Lux, Lancet)", "title": "ACTH vs VGB for Infantile Spasms (LOF SCN2A)",
     "note": "ACTH + VGB combination superior to VGB alone for hypsarrhythmia resolution; "
             "applicable to LOF SCN2A infantile spasms"},
    {"std": "FDA SHARE REMS (Vigabatrin)", "title": "Sabril REMS Program",
     "note": "Mandatory enrolment before VGB in USA; Goldman VF q3M; OCT q6M — "
             "applies to LOF SCN2A patients on VGB for infantile spasms"},
    {"std": "ACMG-AMP 2015", "title": "Variant Classification Standards",
     "note": "GOF vs LOF functional classification required — not just P/LP; "
             "functional studies (patch clamp / computational) recommended for VUS at critical residues"},
    {"std": "ACNS EEG Guidelines 2021", "title": "Critical Care EEG Terminology and Monitoring",
     "note": "cEEG mandatory in SCN2A NICU (GOF); electroclinical dissociation monitoring; "
             "LVFA ictal pattern classification"},
    {"std": "Wolff 2019 Am J Hum Genet", "title": "SCN2A GOF vs LOF functional classification framework",
     "note": "Landmark 2019 paper establishing GOF (neonatal) vs LOF (infantile-later onset) "
             "clinical framework; basis for current treatment decision algorithm"},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Seizure onset < 3 days of life + hemisynchronous BS",
     "action": "Priority GOF SCN2A / KCNQ2 genomic panel; empiric CBZ only under specialist supervision with cEEG"},
    {"threshold": "CBZ TDM 4-12 µg/mL (trough, GOF)",
     "action": "Target for seizure suppression; > 12 = toxicity (diplopia, ataxia, Na-channel over-block)"},
    {"threshold": "OXC MHD TDM 12-24 µg/mL",
     "action": "Active metabolite guide; < 12 = sub-therapeutic; > 24 = Na+ watch + toxicity risk"},
    {"threshold": "Na+ < 130 mEq/L on CBZ/OXC",
     "action": "Symptomatic SIADH: reduce dose or switch agent; IV saline if severe (Na < 120)"},
    {"threshold": "HLA-B*1502 positive (Asian patients)",
     "action": "ABSOLUTE: do NOT start CBZ or OXC — use alternative Na-channel blocker (PHT/LTG caution) or change class"},
    {"threshold": "Infantile spasms onset 3-15 months in SCN2A patient",
     "action": "LOF phenotype likely — CONFIRM variant functional class before ANY Na-channel blocker; ACTH + VGB first"},
    {"threshold": "EEG worsening within 72h of Na-channel blocker",
     "action": "EMERGENCY: LOF SCN2A — immediately withdraw CBZ/OXC/PHT; diazepam IV bridge; specialist review"},
    {"threshold": "2 AED failures → KD discussion (LOF DRE)",
     "action": "ILAE DRE definition met: formally offer KD trial (Level B for DRE DEE)"},
    {"threshold": "Seizure-free 2+ years on CBZ (GOF)",
     "action": "Discuss CBZ dose reduction / taper (minimum 6-week per step); EEG before and during taper"},
    {"threshold": "POLG exclusion before VPA",
     "action": "MANDATORY — fatal Alpers hepatotoxicity; POLG panel or WES before VPA in any DEE"},
]

# ── Concepts (Definitions) ────────────────────────────────────────────────────
CONCEPTS = [
    {
        "term": "SCN2A (Nav1.2)",
        "definition": (
            "Voltage-gated sodium channel alpha subunit 2 gene (SCN2A) at 2q24.3, encoding "
            "Nav1.2 — one of the nine mammalian voltage-gated sodium channel alpha isoforms. "
            "Nav1.2 is the principal action-potential-generating channel in the axon initial "
            "segment (AIS) of cortical pyramidal neurons during fetal and early neonatal "
            "development, transitioning to a more distributed axonal location as Nav1.6 "
            "replaces it at the distal AIS after the first postnatal week. "
            "Nav1.2 is also critical for action potential propagation in fast-spiking "
            "GABAergic interneurons — the LOF epilepsy paradox explained by this expression."
        ),
    },
    {
        "term": "GOF vs LOF SCN2A — Treatment Pivot",
        "definition": (
            "The most clinically critical biomarker in SCN2A-DEE: variant FUNCTIONAL CLASS "
            "determines OPPOSITE treatment strategies. "
            "GOF (Nav1.2 persistently open) → TREAT with Na-channel blockers (CBZ, OXC). "
            "LOF (Nav1.2 haploinsufficiency) → AVOID Na-channel blockers (worsen seizures); "
            "use ACTH/VGB (for IS) / LEV / VPA / KD. "
            "No other DEE has this degree of treatment-axis polarity — functional class "
            "determination is a MEDICAL EMERGENCY in SCN2A-DEE patients requiring acute "
            "AED selection. Patch-clamp functional assay (research) or clinical algorithm "
            "(GOF = neonatal onset + BS; LOF = infantile onset + no BS) guides acute decision."
        ),
    },
    {
        "term": "Persistent Sodium Current (INaP) — GOF Mechanism",
        "definition": (
            "Persistent Na+ current (INaP): small fraction of Na+ channels that fail to "
            "inactivate after the action potential, maintaining a sustained inward Na+ current "
            "below the firing threshold. In GOF SCN2A variants: INaP is 10-30× elevated "
            "above wild-type → the membrane cannot fully repolarise → sustained depolarisation "
            "→ repetitive high-frequency bursting → seizure. "
            "CBZ/OXC preferentially block INaP (higher affinity for inactivated state) — "
            "the pharmacological basis of the GOF treatment rationale."
        ),
    },
    {
        "term": "EIEE11 (MIM #613721)",
        "definition": (
            "OMIM designation for SCN2A-related Early Infantile Epileptic Encephalopathy, "
            "type 11 (EIEE11). Caused by de novo GOF SCN2A variants; neonatal onset; "
            "ILAE preferred current term: 'SCN2A-DEE'. Includes the full GOF spectrum from "
            "severe EIEE11 (Ohtahara-like, neonatal burst-suppression) to moderate GOF "
            "(SLNE / BFNIS spectrum). LOF SCN2A now considered a distinct entity from "
            "EIEE11 — 'SCN2A-LOF-DEE' or 'SCN2A-ASD-DEE'."
        ),
    },
    {
        "term": "BFNIS (Benign Familial Neonatal-Infantile Seizures)",
        "definition": (
            "Autosomal dominant benign epilepsy of neonatal-infantile period caused by mild "
            "GOF SCN2A variants (Heron 2002, Nat Genet). Onset 2 days - 6 months; self-limited; "
            "seizures remit by 12 months; normal development in most. "
            "BFNIS and EIEE11-GOF are a phenotypic continuum — severity correlates with "
            "degree of persistent Na+ current increase. BFNIS families: 50% recurrence; "
            "neonatal EEG surveillance recommended for subsequent siblings."
        ),
    },
    {
        "term": "LVFA (Low-Voltage Fast Activity) — Ictal Signature",
        "definition": (
            "Low-Voltage Fast Activity: EEG ictal onset pattern characterised by abrupt "
            "attenuation of background amplitude and emergence of high-frequency (20-80 Hz) "
            "low-amplitude activity — the EEG correlate of depolarisation block in seizure "
            "onset zone neurons. LVFA in SCN2A-GOF: onset from AIS of pyramidal neurons "
            "(Nav1.2-rich); bilateral hemisphere involvement produces hemisynchronous pattern. "
            "Clinically: tonic stiffening correlates with LVFA ictal phase. "
            "Discriminates GOF from LOF: GOF typically has LVFA onset; LOF typically has "
            "slower poly-spike-wave or electrodecrement onset."
        ),
    },
    {
        "term": "HLA-B*1502 — SJS/TEN Risk",
        "definition": (
            "Human Leukocyte Antigen B*1502 allele: strongly associated with "
            "carbamazepine and oxcarbazepine-induced Stevens-Johnson Syndrome (SJS) and "
            "toxic epidermal necrolysis (TEN) in Han Chinese, Thai, Vietnamese, and Malay "
            "populations (~8-10% carrier frequency). CPIC Level A Guideline: screen all "
            "Asian patients for HLA-B*1502 before prescribing CBZ/OXC; if positive, "
            "AVOID CBZ/OXC and use alternative. Non-Asian populations: lower SJS/TEN risk "
            "from HLA-B*1502 but not zero — clinical vigilance maintained."
        ),
    },
    {
        "term": "SCN2A ASO Gene Therapy",
        "definition": (
            "Antisense oligonucleotides (ASOs) targeting SCN2A mRNA — precision medicine "
            "approach differentiated by GOF vs. LOF. "
            "LOF ASO: upregulates wild-type allele via splice-modulation or NMD rescue. "
            "GOF ASO: allele-selective silencing of pathogenic allele — reduces persistent "
            "Na+ current while preserving WT Nav1.2. "
            "Administered intrathecally (IT) for CNS penetration. "
            "Phase 1/2 trials in progress as of 2026; animal models (Scn2a mouse) show "
            "significant seizure reduction and behavioural normalisation. "
            "Refer eligible patients (confirmed SCN2A GOF or LOF, ongoing seizures) to "
            "Family SCN2A Foundation patient registry."
        ),
    },
    {
        "term": "SIADH (Syndrome of Inappropriate ADH) — CBZ/OXC",
        "definition": (
            "Syndrome of Inappropriate Antidiuretic Hormone secretion: CBZ and OXC increase "
            "ADH release or potentiate ADH renal action → excess water reabsorption → "
            "dilutional hyponatraemia (Na < 135 mEq/L). "
            "OXC causes SIADH more frequently than CBZ (~25% OXC vs ~10% CBZ). "
            "Clinical features: headache, nausea, cognitive slowing, seizure exacerbation "
            "(hyponatraemia lowers seizure threshold). "
            "Management: Na+ monitoring q4 weeks; fluid restriction if mild SIADH; "
            "dose reduction if Na < 130; switch to PHT if Na < 125 symptomatic."
        ),
    },
    {
        "term": "SUDEP (Sudden Unexpected Death in Epilepsy) — SCN2A",
        "definition": (
            "SUDEP risk in SCN2A-DEE: ELEVATED for drug-resistant LOF and uncontrolled GOF. "
            "GOF responders (CBZ seizure-free): SUDEP risk approaches general population. "
            "LOF DRE: high SUDEP risk — nocturnal monitoring (baby monitor/movement sensor), "
            "supervised sleeping, rescue midazolam buccal, NICE NG217 §1.15 counselling. "
            "Mechanism: post-ictal generalised EEG suppression (PGES) → respiratory and/or "
            "cardiac arrest. SUDEP counselling mandatory at first diagnosis discussion."
        ),
    },
    {
        "term": "Nav1.2 Developmental Expression Switch",
        "definition": (
            "Critical developmental neurobiology: Nav1.2 is the dominant AIS sodium channel "
            "isoform in fetal and neonatal neurons (expressed from mid-gestation); Nav1.6 "
            "(SCN8A) gradually replaces Nav1.2 at the distal AIS from the second postnatal "
            "week onward (the 'Nav1.2→Nav1.6 switch'). "
            "Clinical implication: (1) GOF SCN2A seizures are EARLIEST in life (AIS is Nav1.2-rich); "
            "(2) LOF seizure ONSET is DELAYED (Nav1.2 haploinsufficiency in interneurons manifests "
            "after the early neonatal window when Nav1.6 is not yet compensatory); "
            "(3) explains why GOF typically remits in BFNIS after first months — Nav1.6 takes "
            "over GOF-Nav1.2 function at the AIS."
        ),
    },
    {
        "term": "Electroclinical Dissociation — SCN2A-GOF",
        "definition": (
            "ECD in SCN2A-GOF: clinical seizures cease after phenobarbital loading while EEG "
            "seizures continue — occurs in ~30% of GOF EIEE11 neonates after PB (less common "
            "than in STXBP1-DEE where ECD occurs in > 60%). "
            "Continuous EEG (cEEG) mandatory in NICU to detect ECD. "
            "After CBZ/OXC initiation: true EEG seizure suppression expected in responders "
            "(not just clinical cessation) — confirm EEG normalisation as treatment response."
        ),
    },
    {
        "term": "Family SCN2A Foundation",
        "definition": (
            "Patient advocacy and research foundation for SCN2A-related disorders; "
            "maintains the international SCN2A patient registry; facilitates access to "
            "clinical trials (ASO gene therapy, GOF-specific precision medicine); "
            "provides family support and clinician education resources. "
            "All newly diagnosed SCN2A-DEE patients should be referred to the Foundation "
            "registry — data contributes to natural history cohort and trial eligibility."
        ),
    },
    {
        "term": "CPIC (Clinical Pharmacogenomics Implementation Consortium)",
        "definition": (
            "International body producing evidence-based clinical pharmacogenomics guidelines "
            "for drug-gene interactions. CPIC Level A (highest evidence): actionable genotype "
            "with clear clinical recommendation. "
            "HLA-B*1502 + CBZ/OXC = CPIC Level A: avoid CBZ and OXC in HLA-B*1502-positive "
            "patients; genotype before first prescription. Mandatory for Asian patients "
            "receiving CBZ/OXC for SCN2A-GOF treatment."
        ),
    },
    {
        "term": "Axon Initial Segment (AIS) — Nav1.2 Hub",
        "definition": (
            "The axon initial segment: a specialised ~20-60 µm compartment at the proximal "
            "axon where action potentials (APs) are initiated. Nav1.2 is densely clustered at "
            "the AIS by the AIS scaffold proteins (AnkyrinG, βIV-spectrin). "
            "GOF SCN2A at the AIS: persistent Na+ current at the precise site where APs are "
            "generated → maximal hyperexcitability. CBZ/OXC high efficacy in GOF is partly "
            "explained by the AIS localisation — high drug access to the largest Nav1.2 cluster."
        ),
    },
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "ref": "Wolff 2019 Am J Hum Genet",
        "title": "Genetic and Phenotypic Heterogeneity Suggest Therapeutic Implications in SCN2A-Related Disorders",
        "note": "Landmark paper: GOF (neonatal) vs LOF (infantile/later) framework; "
                "Na-channel blocker efficacy in GOF vs worsening in LOF; 71-patient cohort"
    },
    {
        "ref": "Begemann 2019 Epilepsia",
        "title": "Modification of phenotype in SCN2A-related epilepsy by SCN2A variants and environmental factors",
        "note": "GOF pharmacogenomics; CBZ efficacy correlates with persistent current magnitude; "
                "precision medicine treatment algorithm"
    },
    {
        "ref": "Heron 2002 Nat Genet",
        "title": "Sodium-channel defects in benign familial neonatal-infantile seizures",
        "note": "Landmark: BFNIS linked to SCN2A; mild GOF in familial neonatal seizures; "
                "benign prognosis and autosomal dominant inheritance established"
    },
    {
        "ref": "Sanders 2018 Nat Neurosci",
        "title": "Progress in understanding and treating SCN2A-mediated disorders",
        "note": "Comprehensive SCN2A review: GOF/LOF mechanisms, Nav1.2 developmental switch, "
                "ASD-LOF connection, therapeutic landscape including ASO approaches"
    },
    {
        "ref": "Lux 2004 Lancet (UKISS)",
        "title": "The United Kingdom Infantile Spasms Study comparing vigabatrin with prednisolone or tetracosactide",
        "note": "ACTH + VGB combination for infantile spasms (LOF SCN2A IS phase) — "
                "RCT basis for IS protocol applicable to LOF SCN2A"
    },
    {
        "ref": "Ogiwara 2009 J Neurosci",
        "title": "Nav1.2 haploinsufficiency in excitatory neurons and autism-epilepsy in SCN2A knockout mice",
        "note": "Pre-clinical: Scn2a+/- mouse model; ASD + epilepsy phenotype; "
                "interneuron LOF mechanism; basis for LOF therapeutic strategy"
    },
]

# ── Patient Generator ─────────────────────────────────────────────────────────
def _generate_patients():
    random.seed(SEED)
    categories = [
        ("De-novo-SCN2A-GOF-severe-EIEE11", 16),
        ("De-novo-SCN2A-LOF-West-ASD-DEE", 12),
        ("De-novo-SCN2A-GOF-moderate-neonatal-infantile", 6),
        ("Familial-SCN2A-BFNIS-AD-SLNE-DEE", 4),
        ("Clinical-SCN2A-DEE-SCN2A-negative", 3),
    ]
    sexes = ["M", "F"]
    disease_phases = ["Neonatal-NICU-GOF", "CBZ-stable-GOF", "IS-phase-LOF",
                      "Post-IS-DRE-LOF", "School-age-ASD-LOF", "BFNIS-remission"]
    gof_aeds = ["CBZ", "OXC", "CBZ+LEV", "OXC+CLB", "CBZ+VPA(POLG-excl)", "PHT-bridge→CBZ"]
    lof_aeds = ["LEV+VPA", "ACTH-completed+VGB+LEV", "VPA+CLB", "LEV+KD", "KD+CLB", "LEV+CLB+KD"]
    seizure_controls = ["drug-resistant", "partial-control", "seizure-free"]
    gof_weights = [0.30, 0.30, 0.40]  # GOF better response to CBZ
    lof_weights = [0.60, 0.25, 0.15]  # LOF worse prognosis

    patients = []
    pid = 1
    for cat, n in categories:
        is_gof = "GOF" in cat or "BFNIS" in cat
        for _ in range(n):
            sex = random.choice(sexes)
            if is_gof:
                onset_days = random.randint(0, 5)  # neonatal GOF
                aed = random.choice(gof_aeds)
                control = random.choices(seizure_controls, weights=gof_weights)[0]
                phase = random.choice(["Neonatal-NICU-GOF", "CBZ-stable-GOF", "BFNIS-remission"]
                                       if "BFNIS" in cat else ["Neonatal-NICU-GOF", "CBZ-stable-GOF"])
                cbz_level = round(random.uniform(4.5, 11.0), 1) if "CBZ" in aed else None
                mhd_level = round(random.uniform(12.5, 22.0), 1) if "OXC" in aed else None
                na_level = round(random.uniform(131, 143), 1)
                age_months = random.randint(0, 60)
            else:
                onset_days = random.randint(90, 450)  # LOF: 3-15 months in days
                aed = random.choice(lof_aeds)
                control = random.choices(seizure_controls, weights=lof_weights)[0]
                phase = random.choice(["IS-phase-LOF", "Post-IS-DRE-LOF", "School-age-ASD-LOF"])
                cbz_level = None
                mhd_level = None
                na_level = round(random.uniform(133, 142), 1)
                age_months = random.randint(3, 180)

            kd_on = "Y" if "KD" in aed else "N"
            bhb = round(random.uniform(1.8, 4.2), 1) if kd_on == "Y" else None
            vgb_on = "Y" if "VGB" in aed else "N"
            share_enrolled = "Y" if vgb_on == "Y" else "N"
            polg_tested = random.choice(["Y", "Y", "Y", "N"])
            hla_tested = "Y" if is_gof else "N"
            hla_result = random.choice(["Negative", "Negative", "Negative", "Positive"]) if hla_tested == "Y" else "N/A"
            asd_dx = random.choice(["Y", "N", "N"]) if not is_gof else "N"

            patients.append({
                "id": f"SCN2A{pid:03d}",
                "age_months": age_months,
                "sex": sex,
                "onset_age_days": onset_days,
                "category": cat,
                "functional_class": "GOF" if is_gof else "LOF",
                "disease_phase": phase,
                "current_treatment": aed,
                "seizure_control": control,
                "cbz_level_ugml": cbz_level,
                "mhd_level_ugml": mhd_level,
                "na_level_meql": na_level,
                "kd_on": kd_on,
                "bhb_mmoll": bhb,
                "vgb_on": vgb_on,
                "share_rems_enrolled": share_enrolled,
                "polg_tested": polg_tested,
                "hla_b1502_tested": hla_tested,
                "hla_b1502_result": hla_result,
                "asd_diagnosis": asd_dx,
            })
            pid += 1
    return patients


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    pts = _generate_patients()
    n = len(pts)
    gof_pts = [p for p in pts if p["functional_class"] == "GOF"]
    lof_pts = [p for p in pts if p["functional_class"] == "LOF"]
    seizure_free = sum(1 for p in pts if p["seizure_control"] == "seizure-free")
    drug_resistant = sum(1 for p in pts if p["seizure_control"] == "drug-resistant")
    kd_on = sum(1 for p in pts if p["kd_on"] == "Y")
    vgb_on = sum(1 for p in pts if p["vgb_on"] == "Y")
    polg_tested = sum(1 for p in pts if p["polg_tested"] == "Y")
    hla_tested = sum(1 for p in pts if p["hla_b1502_tested"] == "Y")
    asd_dx = sum(1 for p in pts if p["asd_diagnosis"] == "Y")
    gof_seizure_free = sum(1 for p in gof_pts if p["seizure_control"] == "seizure-free")
    lof_drug_resistant = sum(1 for p in lof_pts if p["seizure_control"] == "drug-resistant")

    return {
        "syndrome": "SCN2A Encephalopathy (SCN2A-DEE / EIEE11)",
        "gene": "SCN2A (2q24.3)",
        "inheritance": "De novo in ~90%; ~10% familial (BFNIS / AD-SLNE spectrum)",
        "n_patients": n,
        "n_gof": len(gof_pts),
        "n_lof": len(lof_pts),
        "key_gene": "SCN2A (2q24.3) — Nav1.2 voltage-gated sodium channel; AIS-dominant neonatal; "
                    "GOF → persistent Na+ current → neonatal seizures; LOF → interneuron haploinsufficiency → IS/ASD",
        "eeg_hallmark": (
            "GOF: Hemisynchronous burst-suppression (neonatal) + LVFA ictal onset → CBZ normalises EEG within 48h. "
            "LOF: No burst-suppression; West/hypsarrhythmia (3-15M) → LGS-like slow spike-wave. "
            "CRITICAL: GOF vs LOF EEG pattern guides OPPOSITE treatment decisions."
        ),
        "key_biomarker": (
            "SCN2A variant functional class (GOF vs LOF) — determines treatment axis. "
            "GOF: CBZ TDM 4-12 µg/mL / OXC MHD 12-24 µg/mL. HLA-B*1502 before CBZ/OXC (Asian). "
            "LOF: POLG exclusion before VPA; VGB SHARE REMS; BHB 2-4 mmol/L (KD)."
        ),
        "key_aha": (
            "SCN2A GOF and LOF require OPPOSITE AED treatment — most critical pharmacogenomic axis in all of genetic epilepsy. "
            "GOF: Na-channel blockers (CBZ/OXC) — seizure freedom ~40-50%. "
            "LOF: Na-channel blockers ABSOLUTELY CONTRAINDICATED — acute worsening in 24-72h. "
            "EIEE11 neonatal: hemisynchronous BS + LVFA ictal (contrast STXBP1: asynchronous BS). "
            "HLA-B*1502 mandatory before CBZ/OXC in Asian patients (SJS/TEN — CPIC Level A). "
            "SIADH: Na+ monitoring q4 weeks on CBZ/OXC (OXC higher risk). "
            "LOF: ASD in ~70% — early autism screening (M-CHAT at 18 months). "
            "ASO gene therapy trials (LOF + GOF): refer to Family SCN2A Foundation registry."
        ),
        "etiologies": [
            {"etiology": "De novo SCN2A GOF severe (EIEE11)", "category": "De-novo-SCN2A-GOF-severe-EIEE11", "pct": 39},
            {"etiology": "De novo SCN2A LOF (West / ASD-DEE)", "category": "De-novo-SCN2A-LOF-West-ASD-DEE", "pct": 29},
            {"etiology": "De novo SCN2A GOF moderate (neonatal-infantile)", "category": "De-novo-SCN2A-GOF-moderate-neonatal-infantile", "pct": 15},
            {"etiology": "Familial SCN2A (BFNIS / AD-SLNE-DEE)", "category": "Familial-SCN2A-BFNIS-AD-SLNE-DEE", "pct": 10},
            {"etiology": "Clinical SCN2A-DEE — SCN2A-negative", "category": "Clinical-SCN2A-DEE-SCN2A-negative", "pct": 7},
        ],
        "seizure_type_prevalence": {
            "Tonic / Focal Neonatal Seizures (GOF)": 88,
            "Focal-to-Bilateral Tonic-Clonic (FBTCS)": 72,
            "Infantile Spasms / West Syndrome (LOF)": 42,
            "Myoclonic / Myoclonic-Atonic (LOF)": 28,
        },
        "trigger_seizure_rates": {
            "Fever / febrile illness": 82,
            "Missed / late AED dose": 75,
            "Intercurrent illness (GI, viral)": 68,
            "Sleep deprivation": 55,
            "Rapid CBZ dose reduction (GOF)": 45,
            "Hyperthermia (bath, heat)": 38,
            "Inadvertent Na-channel blocker (LOF)": 30,
            "Puberty / hormonal change": 20,
        },
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "kpis": {
            "gof_pct": round(len(gof_pts) / n * 100, 1),
            "lof_pct": round(len(lof_pts) / n * 100, 1),
            "seizure_free_pct": round(seizure_free / n * 100, 1),
            "drug_resistant_pct": round(drug_resistant / n * 100, 1),
            "gof_seizure_free_pct": round(gof_seizure_free / len(gof_pts) * 100, 1) if gof_pts else 0,
            "lof_drug_resistant_pct": round(lof_drug_resistant / len(lof_pts) * 100, 1) if lof_pts else 0,
            "kd_on_pct": round(kd_on / n * 100, 1),
            "asd_dx_pct": round(asd_dx / n * 100, 1),
            "polg_tested_pct": round(polg_tested / n * 100, 1),
            "hla_tested_pct": round(hla_tested / n * 100, 1),
        },
        "clinical_alerts": [
            "⚡ GOF vs LOF — OPPOSITE TREATMENTS: GOF = CBZ/OXC first-line; LOF = Na-channel blockers ABSOLUTELY CONTRAINDICATED.",
            "🧬 GOF EIEE11: hemisynchronous burst-suppression + LVFA ictal (contrast STXBP1: asynchronous); onset < 3 days.",
            "💊 CBZ in GOF: seizure freedom ~40-50%; EEG normalisation within 48h confirms GOF response.",
            "🚨 LOF EMERGENCY: CBZ/OXC prescribed → EEG worsens within 72h = IMMEDIATELY withdraw, diazepam bridge, specialist review.",
            "🧪 HLA-B*1502 MANDATORY before CBZ/OXC in Asian patients (CPIC Level A) — SJS/TEN risk.",
            "🔬 Na+ monitoring q4 weeks on CBZ/OXC: OXC > CBZ SIADH risk; Na < 130 = dose reduce or switch.",
            "🧠 LOF: ASD in ~70% — early M-CHAT screening at 18 months; neuropsychology referral at diagnosis.",
            "🛡️ SUDEP counselling — LOF DRE: nocturnal alarm + rescue BZD plan (NICE NG217 §1.15).",
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
