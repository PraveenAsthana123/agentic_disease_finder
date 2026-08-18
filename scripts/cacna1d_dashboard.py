"""
CACNA1D Epilepsy — SANDD Syndrome / DEE + Congenital Deafness / Cav1.3 L-type HVA Ca²⁺ Channel / GOF-LOF-Dual / 3p14.3
========================================================================================================================
40-patient cohort · CACNA1D (3p14.3) · Cav1.3 (α1D) L-type HVA Ca²⁺ Channel · AD GOF de novo + AR LOF
OMIM: SANDD syndrome #614896 (Sinoatrial node dysfunction and deafness) · *114205 CACNA1D gene

KEY CACNA1D BIOLOGY — Cav1.3 LOW-THRESHOLD L-TYPE HVA Ca²⁺ CHANNEL:
CACNA1D (3p14.3) encodes Cav1.3 (α1D), the LOW-THRESHOLD L-type (dihydropyridine-sensitive) Ca²⁺ channel.
Critical distinction from Cav1.2 (CACNA1C):
  · Cav1.3 V1/2 activation ≈ −40 to −55 mV — activates at near-resting potential (UNIQUE among L-types)
  · Cav1.2 V1/2 activation ≈ −10 to −20 mV — true HVA, requires larger depolarization
  · This low threshold enables Cav1.3 to drive PACEMAKER currents (SA node + dorsal root ganglia)
  · Cav1.3 is the PRIMARY L-type Ca²⁺ channel in cochlear inner hair cells (IHC synaptic ribbon)
  · Cav1.3 enriched: SA node · AV node · adrenal zona glomerulosa · dopaminergic SNc/VTA neurons · cochlear IHC · cortex/hippocampus

Cav1 (L-type HVA) subfamily:
  · Cav1.1 (CACNA1S, 1q32.1): skeletal muscle EC-coupling; malignant hyperthermia/hypoKPP2
  · Cav1.2 (CACNA1C, 12p13.33): cardiac+neuronal dominant; Timothy Syndrome/LQTS8/DEE
  · Cav1.3 (CACNA1D, 3p14.3): cochlear IHC + SA node pacemaker + dopaminergic; SANDD/DEE [THIS GENE]
  · Cav1.4 (CACNA1F, Xp11.23): retinal photoreceptors; CSNB2 (congenital stationary night blindness)

KEY CLINICAL NOTES:
  1. GOF MECHANISM — IMPAIRED VOLTAGE-DEPENDENT INACTIVATION + WINDOW CURRENT EXPANSION:
     - CACNA1D GOF (de novo dominant-negative / activating missense) → slowed VDI + CDI
       → prolonged window Ca²⁺ current at −55 to −30 mV (sub-threshold for most neurons)
       → persistent autonomous Ca²⁺ influx in cortical/hippocampal neurons → DEE
       → adrenal zona glomerulosa GOF → autonomous aldosterone production → PRIMARY ALDOSTERONISM
       → excess Cav1.3 current in SA node → tachycardia / sinus dysrhythmia (paradoxical: unlike LOF bradycardia)
     - GOF syndrome: DEE + autism + developmental delay; ALDOSTERONE excess (hypertension) in ~30%
  2. LOF MECHANISM — SANDD SYNDROME (AR biallelic):
     - CACNA1D LOF (biallelic recessive): no functional Cav1.3 → no IHC ribbon synapse Ca²⁺ trigger
       → congenital profound sensorineural deafness (SNHL) — Cav1.3 is IRREPLACEABLE in cochlear IHC
       → SA node: loss of Cav1.3 pacemaker current → sinus bradycardia / sick sinus syndrome (SSS)
       → pacemaker implantation often required in SANDD
       → epilepsy is RARE in SANDD LOF (unlike GOF where epilepsy is dominant feature)
  3. BIOPHYSICAL DISTINCTION Cav1.3 vs Cav1.2 — CRITICAL FOR PHARMACOLOGY:
     - Cav1.3 lower V1/2 (−40–55 mV) enables activation by small dendritic depolarizations
     - DHP SELECTIVITY: isradipine/nitrendipine have ~10× higher affinity for Cav1.3 vs Cav1.2
       at hyperpolarized holding potentials; clinically exploited in Parkinson neuroprotection (STEADY trial)
     - Verapamil blocks Cav1.3 but less efficiently than Cav1.2 (intracellular blocker; less state-dependent)
     - Amlodipine and nimodipine have partial Cav1.3 activity
  4. PRECISION THERAPY — ISRADIPINE (CACNA1D-specific DHP):
     - Isradipine 2.5–10 mg/day (adult oral); 0.05–0.1 mg/kg/day (pediatric — off-label)
     - Mechanism: state-dependent DHP block of Cav1.3 window current at low holding potentials
     - Neuroprotective precedent: STEADY trial (NCT02168842) — isradipine in early Parkinson's
       targeting Cav1.3-driven dopaminergic Ca²⁺ overload; trial negative for PD primary endpoint
       but established Cav1.3 engagement in vivo. Epilepsy GOF: case series evidence only.
     - Side effects: reflex tachycardia, hypotension, peripheral oedema — monitor BP at each visit
     - Aldosterone reduction: in GOF with primary aldosteronism → additional benefit (dual mechanism)
  5. CARDIAC MONITORING — BIDIRECTIONAL PHENOTYPE:
     - LOF/SANDD: sinus bradycardia/SSS → PACEMAKER evaluation mandatory; Holter 48h at diagnosis
     - GOF-DEE: sinus tachycardia / dysrhythmia possible; QTc usually normal (unlike CACNA1C TS)
     - KEY DISTINCTION: CACNA1C TS has LQTS8 (QTc >500 ms, 2:1 AV block → fatal arrhythmia risk)
       but CACNA1D GOF does NOT cause significant QT prolongation — different pharmacological CI
  6. ALDOSTERONE CONNECTION — PRIMARY ALDOSTERONISM IN GOF:
     - ~30% GOF patients have autonomous adrenal aldosterone → hypertension + hypokalemia
     - Aldosterone adenoma-equivalent biochemistry (aldosterone:renin ratio elevated)
     - Isradipine can suppress zona glomerulosa Cav1.3 GOF → reduce aldosterone production
     - Adrenalectomy NOT recommended (no solitary adenoma; bilateral GOF channel activity)
     - MRA (mineralocorticoid receptor antagonists: spironolactone/eplerenone) for aldosterone excess
  7. COCHLEAR IMPLANT FOR SANDD DEAFNESS:
     - SANDD deafness is cochlear (IHC synaptic, not spiral ganglion), not retrocochlear
     - Cochlear implant is EFFECTIVE in SANDD (spiral ganglion neurons intact — afferent pathway preserved)
     - ABR (auditory brainstem response) absent/flat from birth — no distortion product OAE (DPOAEs present initially)
     - Preadaptation for CI by age 12 months optimal for language development
  8. ALDH7A1, POLG1 MANDATORY EXCLUSION before relevant AED choice
"""

import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG — 5-CLASS GOF-LOF SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-DEE-Autism-Primary-Aldosteronism",
        "pct": 35,
        "mechanism": "De novo dominant CACNA1D GOF (A749G/G407R/I750M equivalent classes) → impaired VDI + CDI → persistent Cav1.3 window current at −55 to −30 mV → cortical Ca²⁺ overload + adrenal Cav1.3 autonomous aldosterone",
        "phenotype": "DEE onset 4–18 months + autism/ASD (85%) + intellectual disability (90%) + primary aldosteronism with hypertension (~30%) + mild sinus tachycardia; NO deafness (cochlear IHC intact in GOF)",
        "eeg_pattern": "Hypsarrhythmia (West) or multifocal IEDs; focal cortical onset evolving to GTCS; rarely absence",
        "severity": "Severe DEE; pharmacoresistant in 55%; aldosterone comorbidity increases CV risk",
        "reference": "Ortner 2014 Cell Physiol Biochem 34:1375; Pinggera 2015 Cell 160:1037; Scholl 2013 Nat Genet 45:1050",
    },
    {
        "category": "GOF-DEE-Autism-Normotensive",
        "pct": 28,
        "mechanism": "De novo GOF → neuronal hyperexcitability dominant; adrenal Cav1.3 activity insufficient for aldosterone excess; normotensive",
        "phenotype": "DEE onset 6–24 months + autism/ASD (80%) + ID (88%); blood pressure normal; hearing intact; NO SANDD features",
        "eeg_pattern": "Hypsarrhythmia or focal multifocal IEDs; West syndrome evolution; delta slowing background",
        "severity": "Severe DEE; intellectually profound; respond partially to LEV/VPA/ACTH",
        "reference": "Duflocq 2019 Hum Mol Genet; Scholl 2013 Nat Genet 45:1050 — GOF de novo spectrum",
    },
    {
        "category": "LOF-SANDD-Biallelic",
        "pct": 20,
        "mechanism": "Biallelic LOF CACNA1D (compound heterozygous frameshift/splice/missense) → no functional Cav1.3 → IHC ribbon synapse failure (deafness) + SA node pacemaker loss (bradycardia/SSS)",
        "phenotype": "Profound congenital sensorineural deafness + sinus bradycardia / sick sinus syndrome; epilepsy RARE (5–15%); autism occasional; normal intellect possible in pure SANDD",
        "eeg_pattern": "Usually normal; focal IEDs if epilepsy present; NO hypsarrhythmia in pure SANDD",
        "severity": "Cardiac: high (SSS → syncope/SCA risk without pacemaker); Hearing: profound; Seizures: mild if present",
        "reference": "Baig 2011 Nat Genet 43:776 — SANDD original CACNA1D LOF biallelic family; pacemaker + CI treatment",
    },
    {
        "category": "GOF-Mosaic-Partial-Phenotype",
        "pct": 12,
        "mechanism": "Mosaic CACNA1D GOF (postzygotic de novo) → partial tissue expression → milder DEE + partial aldosterone elevation; normotensive or borderline hypertensive",
        "phenotype": "Mild-moderate DEE + autistic features; focal epilepsy of infancy; aldosterone mildly elevated; sinus rhythm mostly normal",
        "eeg_pattern": "Focal IEDs (temporal/frontal); febrile seizure plus; evolving GGE-like in some; normal background occasionally",
        "severity": "Moderate; better prognosis than germline GOF; some AED responsiveness",
        "reference": "Kakiuchi 2014; Striessnig 2015 Pflugers Arch — mosaic CACNA1D GOF functional consequence modeling",
    },
    {
        "category": "Phenocopy-Panel-Negative",
        "pct": 5,
        "mechanism": "SANDD-like or DEE+autism phenotype; comprehensive CACNA1D panel negative; other gene (KCNT1/KCNQ2/ATP1A3) identified in some",
        "phenotype": "Clinical SANDD-overlap or DEE+autism; multi-gene panel with CACNA1D negative; alternative molecular diagnosis sought",
        "eeg_pattern": "Variable; overlaps with CACNA1D phenotype; re-analysis with WGS recommended",
        "severity": "Moderate-severe; prognosis depends on true molecular diagnosis",
        "reference": "Multi-gene panel (CACNA1D + CACNA1C + KCNT1 + ATP1A3 + SCN1A) recommended in SANDD + DEE overlap",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES — 5 CLASSES
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms (West Syndrome)",
        "frequency_pct": 58,
        "eeg": "Hypsarrhythmia (modified/classic) — background chaotic high-amplitude multifocal spikes + slow waves; interictal IEDs between clusters",
        "semiology": "Flexion or extension spasm clusters (Salaam attacks); 5–50 spasms/cluster; arousal-dependent; crying at onset of cluster; clustering on awakening from NREM",
        "onset_age": "4–12 months median (GOF-DEE); may appear earlier than CACNA1C TS (TS median 3 months)",
        "duration": "Individual spasm 0.5–2 s; cluster 2–20 min",
        "clinical_tip": "ACTH Level A first-line for West syndrome; vigabatrin Level A (avoid in SANDD LOF — VFD monitoring critical); EEG response within 2 weeks → better outcome. Rule out: CACNA1C TS (cardiac QTc), KCNQ2-DEE (neonatal), SCN2A (3–6 months)",
        "lateralization": "bilateral symmetric or mildly asymmetric",
    },
    {
        "type": "Focal Impaired Awareness Seizures",
        "frequency_pct": 65,
        "eeg": "Focal theta/alpha ictal onset (temporal or frontal); recruiting rhythm; post-ictal regional attenuation; IEDs between seizures",
        "semiology": "Staring + unresponsiveness + automatisms (hand fumbling, lip smacking, head turn); duration 30–120 s; post-ictal confusion 1–5 min; often evolve to BTCS in untreated patients",
        "onset_age": "6–36 months in GOF-DEE; later onset possible in mosaic variants",
        "duration": "60–90 s typical",
        "clinical_tip": "Temporal lobe semiology common in GOF — distinguish from mesial temporal sclerosis (no structural lesion in CACNA1D-DEE on MRI; functional Cav1.3 Ca²⁺ overload mechanism). LEV Level B; CBZ/OXC with EEG monitoring (not absolute CI but monitor for worsening)",
        "lateralization": "focal left > right temporal in corpus",
    },
    {
        "type": "GTCS (Bilateral Tonic-Clonic)",
        "frequency_pct": 52,
        "eeg": "Generalized high-frequency ictal discharge (12–25 Hz recruiting); tonic phase → 3–5 Hz clonic phase; post-ictal diffuse suppression",
        "semiology": "Classic tonic → clonic sequence; 1–3 min; post-ictal lethargy/sleep; tongue bite; urinary incontinence in older patients; triggered by fever/sleep deprivation",
        "onset_age": "Evolution from focal seizures; may emerge 12–48 months post-onset",
        "duration": "1–3 min",
        "clinical_tip": "VPA Level B (broad-spectrum; POLG1 MANDATORY before VPA); LEV adjunct; avoid CBZ-monotherapy (focal seizure aggravation possible). POLG1 screening critical — CACNA1D+VPA+POLG1 = Alpers risk same as any epilepsy",
        "lateralization": "bilateral",
    },
    {
        "type": "Focal Tonic Seizures",
        "frequency_pct": 38,
        "eeg": "Focal fast activity (beta/gamma ictal pattern) from frontal; recruiting rapid discharge; electrodecrement in some",
        "semiology": "Sustained limb/head/trunk tonic posturing; often nocturnal; brief (10–30 s); may occur in clusters during NREM sleep",
        "onset_age": "Evolves from infantile spasms in severe GOF-DEE; can occur at any age in established DEE",
        "duration": "10–30 s",
        "clinical_tip": "Nocturnal frontal tonic seizures in CACNA1D-DEE may mimic NFLE (nocturnal frontal lobe epilepsy — SCN8A, CHRNA4). Correct diagnosis critical: isradipine precision therapy applicable only in CACNA1D GOF",
        "lateralization": "frontal dominant; may be contralateral to tonic posture",
    },
    {
        "type": "Febrile Seizures / GEFS+ Overlap",
        "frequency_pct": 30,
        "eeg": "Normal or mild slowing post-ictal; generalized spike-wave if GEFS+ component; no hypsarrhythmia",
        "semiology": "Fever-triggered GTCS; prolonged febrile seizures (>15 min in 25%); febrile status epilepticus; evolves to afebrile epilepsy in 60% within 2 years",
        "onset_age": "6 months – 6 years (febrile phase); afebrile continuation post-6 years",
        "duration": "2–15 min (mean 8 min for febrile seizures)",
        "clinical_tip": "CACNA1D mosaic GOF or mild LOF heterozygous can present as febrile seizure plus (GEFS+-like). Levetiracetam effective for febrile status; avoid routine prophylaxis. KEY: prolonged febrile seizure in a deaf child → screen CACNA1D LOF (SANDD + occasional epilepsy)",
        "lateralization": "bilateral in febrile GTCS; may have focal pre-febrile component",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS — 8 CLASSES
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Intercurrent Illness", "rate_pct": 85, "note": "Strongest trigger in CACNA1D-DEE; fever lowers seizure threshold via temperature-dependent Cav1.3 window current increase; aggressive antipyretics (paracetamol/ibuprofen) recommended at first sign of fever; rectal diazepam rescue plan mandatory"},
    {"trigger": "Sleep Deprivation / Arousal", "rate_pct": 72, "note": "Infantile spasm clusters peak on arousal from NREM; sleep deprivation lowers cortical threshold; consistent sleep schedule; avoid overnight travel / jet lag"},
    {"trigger": "Missed AED / Dose Omission", "rate_pct": 68, "note": "AED non-adherence (common in infants — vomiting/refusal); VPA liquid formulations improve adherence; levetiracetam stable pharmacokinetics; use monitored dispensers"},
    {"trigger": "Stress / Emotional Arousal", "rate_pct": 55, "note": "Limbic Cav1.3 activation by catecholamines (noradrenaline, adrenaline) can sensitize cortical circuits; stress management integral to seizure prevention plan in older children"},
    {"trigger": "Infection (Non-Febrile)", "rate_pct": 48, "note": "Systemic inflammation (cytokine-mediated IL-1β/TNF-α) lowers neuronal threshold via modulation of voltage-gated channels; GI infections particularly destabilizing (AED absorption impaired + fever risk)"},
    {"trigger": "AED Taper / Withdrawal", "rate_pct": 42, "note": "Abrupt AED taper (ACTH, VPA, LEV) → rebound seizure surge; structured taper over ≥3 months; never abrupt cessation"},
    {"trigger": "Photosensitivity (GOF only)", "rate_pct": 25, "note": "Mild photosensitivity in 25% GOF-DEE (vs CACNA1C TS where photo-sensitivity rare); PPR testing at baseline; LED screens with 120 Hz refresh rate; avoid stroboscopic light exposure"},
    {"trigger": "Aldosterone-Related Electrolyte Shift (GOF)", "rate_pct": 20, "note": "Primary aldosteronism (30% GOF) → hypokalemia → hyperexcitability → seizure threshold lowering; electrolyte monitoring quarterly; spironolactone/eplerenone for aldosterone excess improves seizure control by correcting hypokalemia"},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS — 8 CLASSES
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "ACTH / Prednisolone (West Syndrome — infantile spasms)",
        "level": "Level A",
        "moa": "ACTH → MC2R → adrenal steroid production → reduces CRH-mediated hyperexcitability; anti-inflammatory; suppresses ACTH-mediated NMDA sensitization. Prednisolone: direct glucocorticoid effect on GABAergic maturation",
        "dose": "ACTH: 20–40 IU/day IM × 2 weeks (UK protocol); Prednisolone 4 mg/kg/day × 2 weeks (UKISS 2004 equivalent protocol). Taper over 4–6 weeks post-spasm cessation",
        "efficacy": "Spasm cessation 72% at 2 weeks (UKISS); hypsarrhythmia resolution 67%. CACNA1D-DEE: similar response rates to genetic West syndrome cohort",
        "monitoring": "BP (hypertension), glucose (hyperglycaemia), infection (immunosuppression), electrolytes (hypernatraemia), cushingoid features, ophthalmic pressure",
        "cacna1d_note": "First-line for West syndrome component regardless of CACNA1D GOF/LOF; GOF aldosterone elevation may worsen corticosteroid-induced hypertension — BP monitoring intensified in GOF patients during ACTH course",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "moa": "SV2A binding → presynaptic vesicle release modulation; indirect Ca²⁺ current reduction via SV2A-dependent synaptic tuning; POLG-safe (mitochondria-sparing)",
        "dose": "Pediatric: 20–60 mg/kg/day in 2 doses; Adult: 1000–3000 mg/day. IV loading 20–30 mg/kg for status. Renal dose adjustment (eGFR <60)",
        "efficacy": "GOF focal seizures: 55–60% ≥50% reduction; broad-spectrum; useful for GTCS and focal spasms",
        "monitoring": "Behavioural: irritability/aggression (SV2A LEVETIRACETAM behavioural scale q12W); CBC annually; no hepatotoxicity; POLG-safe (unlike VPA)",
        "cacna1d_note": "Preferred first-line broad-spectrum AED in CACNA1D-DEE. POLG1 safe — use LEV when POLG1 not yet tested. Behavioural side effects 25–35% in ASD/ID patients — pyridoxine 50–100 mg/day may reduce irritability",
    },
    {
        "drug": "Valproate (VPA)",
        "level": "Level B",
        "moa": "Na⁺ channel fast inactivation + T-type (CACNA1H/G/I) blockade + GABA-transaminase inhibition + HCN channel modulation; broad-spectrum",
        "dose": "Pediatric: 20–40 mg/kg/day; Adults: 400–2000 mg/day. TDM target 50–100 μg/mL",
        "efficacy": "GOF-GTCS: 60–70%; infantile spasms add-on: 45%; broad-spectrum cover for mixed epilepsy in DEE",
        "monitoring": "POLG1 MANDATORY before VPA (CPIC Level A — Alpers risk in POLG1 pathogenic variants). LFT + ammonia + FBC q3M. VPPP females ≥12y (MHRA 2021). Teratogenicity (VMPC/NTD) — prescribe with FA 5 mg/day",
        "cacna1d_note": "POLG1 testing MANDATORY before VPA in CACNA1D-DEE — same rule as all epilepsy syndromes. If POLG1 positive → VPA ABSOLUTE CI. If POLG1 clear → VPA Level B broad-spectrum option for mixed DEE seizures",
    },
    {
        "drug": "Isradipine (DHP precision L-type Cav1.3 blocker)",
        "level": "Level C (case evidence; CACNA1D GOF precision only)",
        "moa": "Dihydropyridine class calcium channel blocker with preferential Cav1.3 selectivity at physiological holding potentials (≈10× selectivity over Cav1.2 in state-dependent block). Reduces pathological Cav1.3 window current at −55 to −30 mV. Additional benefit: suppresses zona glomerulosa Cav1.3 GOF → reduces autonomous aldosterone production",
        "dose": "Pediatric off-label: 0.05–0.1 mg/kg/day in 2–3 doses (titrate from 0.025 mg/kg/day); Adult: 2.5–10 mg/day in 2 doses. Monitor BP before each dose escalation",
        "efficacy": "GOF-DEE case reports/series: 40–60% seizure reduction in confirmed GOF patients. Aldosterone reduction: 30–50% in hyperaldosterone-GOF cohort. Functional assay (patch clamp) should confirm GOF mechanism before prescribing",
        "monitoring": "BP (hypotension; especially first 2 weeks); heart rate (reflex tachycardia); ankle oedema; electrolytes (aldosterone reduction may improve hypokalemia — monitor K⁺). ECG annually (not LQTS risk as with CACNA1C — but Cav1.3 cardiac channel effects)",
        "cacna1d_note": "CACNA1D GOF PRECISION THERAPY — isradipine is the most rational pharmacological choice for confirmed GOF patients. KEY DISTINCTION vs CACNA1C: verapamil is preferred for CACNA1C/Cav1.2 (Timothy Syndrome); isradipine preferred for CACNA1D/Cav1.3 GOF (lower V1/2 → state-dependent DHP block more effective at Cav1.3). NOT indicated for LOF/SANDD — would worsen cardiac bradycardia",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A (West syndrome first-line — use with VFD monitoring)",
        "moa": "Irreversible GABA-transaminase inhibitor → elevated synaptic GABA → reduced cortical hyperexcitability. Additive/synergistic with ACTH in West syndrome",
        "dose": "Pediatric: 100–150 mg/kg/day (infants); Adult: 1–3 g/day in 2 doses. SHARE REMS (USA): enrollment mandatory due to VFD risk",
        "efficacy": "West syndrome: spasm cessation 54–67% (comparable to ACTH); EEG resolution 50%. Strongest evidence for TSC (not CACNA1D) but applicable to West syndrome any etiology",
        "monitoring": "ERG (electroretinogram) q3M mandatory (SHARE REMS); formal visual field testing q6M from age 3y; MRI for MRI-tractography VFD signature (post-chiasmatic white matter); discontinue if VFD documented (irreversible risk)",
        "cacna1d_note": "SANDD LOF patients: VGB adds VFD risk to deafness — INCREASED CAUTION in SANDD (dual sensory impairment risk); prefer ACTH monotherapy in SANDD + West; if VGB unavoidable, ERG q3M is NON-NEGOTIABLE and inform family of dual sensory risk",
    },
    {
        "drug": "Spironolactone / Eplerenone (MRA for GOF-aldosteronism)",
        "level": "Level B (for primary aldosteronism component in GOF)",
        "moa": "Mineralocorticoid receptor antagonist → blocks aldosterone action at distal tubule → potassium retention + sodium/water excretion → corrects hypokalemia + hypertension caused by autonomous zona glomerulosa Cav1.3 GOF → secondary seizure threshold normalization",
        "dose": "Spironolactone: 25–100 mg/day; Eplerenone (more selective): 25–50 mg twice daily. Titrate to aldosterone:renin ratio normalization and K⁺ normalization",
        "efficacy": "Aldosterone-related seizure aggravation: correction of hypokalemia → seizure threshold improvement in 50–60% GOF patients with documented aldosteronism",
        "monitoring": "K⁺ + creatinine q4–6W until stable; BP fortnightly; spironolactone: gynaecomastia in males (switch to eplerenone); avoid in renal impairment (eGFR <30)",
        "cacna1d_note": "UNIQUE to CACNA1D GOF — no other epilepsy gene has aldosterone-driven seizure complication. Treat aldosteronism as primary disorder (endocrinology co-management); MRA improves both cardiovascular and seizure control in documented GOF+hyperaldosteronism",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B (adjunct for focal/GTCS in DEE)",
        "moa": "GABA-A positive allosteric modulator (benzodiazepine site at α2/α5 subunits); reduces network excitability; long half-life (norclobazam active metabolite t½ 40–50h) → once/twice daily dosing",
        "dose": "Pediatric: 0.1–0.3 mg/kg/day; Adult: 10–30 mg/day in 2 doses. Cluster dosing effective for breakthrough seizures",
        "efficacy": "GOF-DEE add-on: 45–55% ≥50% seizure reduction; useful for daily focal/GTCS clusters",
        "monitoring": "Sedation (especially in ASD/ID patients with polypharmacy); tolerance at 6–12 months; CBC annually; hepatic function",
        "cacna1d_note": "Useful add-on in CACNA1D-DEE polypharmacy (LEV + VPA + CLB); tolerance less pronounced than clonazepam; preferred BZD choice in chronic epilepsy management for GOF-DEE",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B (DRE ≥2 AED failures)",
        "moa": "Beta-hydroxybutyrate (BHB) reduces neuronal Ca²⁺ entry via Cav-channel hyperpolarization shift; reduces glycolytic flux → lower excitatory amino acid production; KATP channel opening → membrane hyperpolarization. May directly modulate Cav1.3 window current threshold",
        "dose": "Classic KD 4:1 ratio or MAD (modified Atkins diet); initiation in metabolic unit; target BHB 2–5 mmol/L; maintained for minimum 3 months before efficacy assessment",
        "efficacy": "DRE infantile/early-childhood DEE: 50–60% ≥50% seizure reduction; ~10% seizure-free. Infantile spasms add-on: 30–40% additional response",
        "monitoring": "Lipids (LDL/TG) q3M; renal function (renal stones 3–5%); growth parameters q3M; micronutrient panel; DEXA bone density annual",
        "cacna1d_note": "HIGH PRIORITY in CACNA1D-DEE + DRE — BHB may provide mechanistic Cav1.3 modulation. Consider at ≥2 AED failures (typically by age 18 months in severe GOF-West). Ensure POLG1 clear before KD (KD generates acetyl-CoA flux that stresses mitochondria)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS — 6 CLASSES
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Tiagabine (TGB) — ABSOLUTE CI",
        "risk": "NCSE (non-convulsive status epilepticus) risk in all focal epilepsy types; potentiates GABAergic shunting → focal status; also prolongs QT interval (use caution in any cardiac comorbidity — including CACNA1D SANDD with cardiac monitoring)",
        "mechanism": "GABA reuptake inhibitor → excess synaptic GABA in focal circuits → aberrant shunting inhibition → NCSE; QT prolongation independent of seizure effect",
        "level": "ABSOLUTE — never use in CACNA1D-DEE or SANDD with epilepsy",
    },
    {
        "drug": "VPA in POLG1 Pathogenic Variants — ABSOLUTE CI",
        "risk": "Alpers-Huttenlocher syndrome risk: VPA inhibits POLG1-driven mitochondrial replication → acute hepatic failure + cortical necrosis → death. POLG1 mutations cause MERRF-like PME, AHS, liver failure.",
        "mechanism": "VPA → direct mitochondrial toxin in POLG1 LOF background; hepatic mitochondrial DNA depletion → acute fulminant hepatic failure",
        "level": "ABSOLUTE — POLG1 testing before VPA is CPIC Level A; if POLG1 positive → switch to LEV/CLB/ZNS",
    },
    {
        "drug": "Isradipine in LOF/SANDD — ABSOLUTE CI",
        "risk": "In SANDD LOF patients: Cav1.3 is already non-functional; isradipine would further suppress residual SA node Cav1.3/Cav1.2 compensatory pacemaker current → worsening of sinus bradycardia/SSS → syncope / complete heart block risk",
        "mechanism": "DHP blockade of SA node L-type channels in already-LOF background → cardioinhibitory → life-threatening bradycardia/AV block",
        "level": "ABSOLUTE in SANDD/LOF — isradipine ONLY for confirmed GOF patients; genotype-guided precision",
    },
    {
        "drug": "Vigabatrin (long-term) — HIGH RISK: Visual Field Defect",
        "risk": "Irreversible bilateral peripheral VFD (visual field defect) in 30–40% with long-term use (>3 years); cumulative dose-dependent. SANDD patients already deaf → dual sensory impairment from VFD would be devastating",
        "mechanism": "Irreversible GABA-transaminase inhibition → GABA accumulation in retinal Müller cells → cone photoreceptor toxicity → peripheral VFD starting nasal then extending",
        "level": "HIGH RISK with long-term use; ABSOLUTE contraindication in SANDD LOF (deaf + blind risk); use only for West syndrome acute course with mandatory ERG monitoring",
    },
    {
        "drug": "Carbamazepine / Oxcarbazepine (CBZ/OXC) — HIGH RISK in GOF focal",
        "risk": "CBZ/OXC enhance Na⁺ channel fast inactivation → can aggravate absence-like/myoclonic components in GOF-DEE; may worsen infantile spasms component; CACNA1D-DEE does NOT have sodium channel pathophysiology as primary driver → Na-blocker may not address root Ca²⁺ channel mechanism. EEG monitoring mandatory",
        "mechanism": "State-dependent Na⁺ channel blockade → disinhibition of surviving GABAergic circuits? Absence-spike aggravation via thalamic modulation possible in DEE background. Not an absolute CI (unlike CACNA1C where cardiac QTc risk adds concern) but monitor closely",
        "level": "HIGH RISK without EEG monitoring; CBZ/OXC not first-line in CACNA1D-DEE; may be useful for focal component in older mosaic-GOF patients after EEG confirms no aggravation",
    },
    {
        "drug": "Class Ia/III Antiarrhythmics (SANDD LOF cardiac management) — AVOID in context",
        "risk": "SANDD patients with SSS may be considered for antiarrhythmic agents — CAUTION: class III (amiodarone/sotalol) and class Ia (quinidine) can worsen SSS; for SANDD bradycardia the ONLY safe and effective treatment is PACEMAKER (class I recommendation) — antiarrhythmics are not indicated and may be harmful",
        "mechanism": "Class III drugs prolong action potential → may worsen AV block or sinus arrest in SSS background; pacemaker is mechanical solution for LOF-bradycardia",
        "level": "AVOID — pacemaker is definitive treatment for SANDD SSS; antiarrhythmics not indicated and may worsen",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING — 14 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG1 testing before VPA (CPIC Level A)", "frequency": "Once (before first VPA dose)", "rationale": "Alpers-Huttenlocher hepatic failure risk; mandatory for all epilepsies before VPA; CACNA1D-DEE patients no exception"},
    {"item": "Aldosterone:Renin ratio (GOF patients)", "frequency": "At diagnosis; q12M if elevated; post-MRA titration", "rationale": "GOF zona glomerulosa Cav1.3 autonomous aldosterone → primary aldosteronism in ~30%; correction improves seizure control + cardiovascular risk"},
    {"item": "Blood pressure monitoring (GOF + MRA/isradipine)", "frequency": "Fortnightly during isradipine titration; monthly steady-state", "rationale": "Isradipine risk of hypotension; aldosterone excess causes hypertension; dual monitoring required in GOF + isradipine"},
    {"item": "Holter 48h + ECG (SANDD LOF + all CACNA1D)", "frequency": "At diagnosis; annually; before AED changes", "rationale": "SANDD: SSS detection → pacemaker indication. GOF: sinus tachycardia/dysrhythmia monitoring. DISTINCT from CACNA1C (where QTc >500 ms is the marker — CACNA1D QTc typically normal)"},
    {"item": "Audiology / ABR (SANDD LOF)", "frequency": "At diagnosis; q6M in infancy; CI assessment by 6–12 months", "rationale": "SANDD profound congenital SNHL; cochlear implant highly effective (spiral ganglion intact); early CI (<12 months) optimizes language outcome"},
    {"item": "Ophthalmology / ERG (vigabatrin recipients)", "frequency": "q3M during VGB treatment (SHARE REMS mandatory)", "rationale": "VGB irreversible VFD; ERG detects pre-symptomatic cone damage; critical in SANDD where dual deafness+blindness would be catastrophic"},
    {"item": "Seizure diary (digital app) + parent-reported trigger log", "frequency": "Ongoing continuous; reviewed q3M", "rationale": "Fever + sleep deprivation are dominant triggers; prospective tracking enables intervention plan tailoring; essential for GOF cluster management"},
    {"item": "Electrolytes: Na⁺ / K⁺ / Cl⁻ (GOF aldosteronism)", "frequency": "q3M in GOF with aldosteronism; q6M other", "rationale": "Primary aldosteronism → hypokalemia → seizure threshold lowering; MRA correction monitored by K⁺ normalization"},
    {"item": "VPA TDM + LFT + ammonia + FBC (if VPA used)", "frequency": "q3M steady-state", "rationale": "Standard VPA monitoring; hepatotoxicity surveillance; hyperammonaemia independent of liver failure (carnitine supplementation if elevated)"},
    {"item": "VPPP counselling (females ≥12y on VPA)", "frequency": "Annual; before any hormonal contraception start", "rationale": "VPA → VMPC + NTD teratogenicity (MHRA 2021 Pregnancy Prevention Programme mandatory); FA 5 mg/day prescribed simultaneously"},
    {"item": "Neurodevelopmental / Autism assessment (GOF)", "frequency": "q6M infancy; q12M early childhood; annual school-age", "rationale": "ASD 80–85% in GOF-DEE; early intervention (ABA, speech, OT) changes developmental trajectory; ADOS-2 from 18 months"},
    {"item": "Renal function + lipid panel (ketogenic diet)", "frequency": "q3M on KD", "rationale": "KD renal stone risk 3–5%; LDL elevation; growth monitoring; DEXA bone density if >2 years on KD"},
    {"item": "SUDEP risk counselling + seizure response plan", "frequency": "Annual review; update at each seizure escalation", "rationale": "SUDEP risk elevated in DRE-DEE; nocturnal GTCS highest risk; seizure detection device (Empatica E4 / Nightwatch) discussed; prone position avoidance"},
    {"item": "Genetic counselling (AD GOF: 50% risk; AR LOF: 25% risk)", "frequency": "At diagnosis; preconceptual; prenatal if requested", "rationale": "GOF AD → 50% transmission risk; some GOF de novo (mosaicism → empirical recurrence <1%). LOF AR (SANDD) → 25% risk per pregnancy; PGD available"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE — 6 STAGES
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Pre-Symptomatic / Neonatal (0–3 months)",
        "focus": "SANDD LOF: deafness identified on newborn ABR screening → cardiology referral (Holter 48h) → pacemaker assessment. GOF: no neonatal deafness; seizure onset anticipated 4–12 months; family education on infant spasm recognition",
        "key_action": "ABR screening, ECG, Holter; genetics for CACNA1D pathogenic variant; aldosterone:renin ratio at 3 months (GOF)",
    },
    {
        "stage": "Infantile (4–18 months) — GOF-DEE West Syndrome Peak",
        "focus": "GOF: West syndrome onset — ACTH + VGB (or ACTH alone if SANDD concern); isradipine initiation after GOF confirmed by functional assay; neurodevelopmental stimulation; cochlear implant (SANDD) by 12 months",
        "key_action": "ACTH Level A for spasms; EEG q4W during active spasms; isradipine in GOF; CI referral in SANDD",
    },
    {
        "stage": "Early Childhood (18 months – 5 years)",
        "focus": "GOF: focal epilepsy evolution; polypharmacy optimization (LEV + VPA ± CLB); ASD early intervention; aldosterone management (spironolactone). SANDD: pacemaker fitting if SSS confirmed; language rehabilitation with CI",
        "key_action": "Polypharmacy rationalization; KD if ≥2 AED failures; VPPP pre-education if VPA; CI mapping (SANDD); speech/OT therapy",
    },
    {
        "stage": "Childhood (5–12 years)",
        "focus": "School inclusion with epilepsy action plan; SUDEP counselling to family; aldosterone BP management; annual epilepsy review; QoL + psychosocial assessment. SANDD: educational support for deaf students; KD safety review",
        "key_action": "Annual neurology + cardiology; Aldosterone q12M; school seizure action plan; ADOS-2 updated",
    },
    {
        "stage": "Adolescence (12–25 years)",
        "focus": "VPPP mandatory for females ≥12y on VPA (MHRA 2021); driving (3y seizure-free minimum); psychosocial: SUDEP anxiety, relationships, alcohol avoidance. Isradipine adolescent BP impact. SANDD: cochlear implant upgrade; hearing aid optimization",
        "key_action": "VPPP + FA 5 mg/day; contraception counselling; driving legislation; psychosocial support; transition to adult neurology",
    },
    {
        "stage": "Adult Reproductive (25+ years)",
        "focus": "Pregnancy planning: VPA avoidance (teratogen); isradipine in pregnancy (relative safety data limited — case-by-case); GOF genetic counselling 50% risk; SANDD AR genetic counselling 25% risk; pacemaker longevity (SANDD)",
        "key_action": "Preconception review; VPA cessation 3 months preconception; folic acid 5 mg; PGD discussion; pacemaker battery check (SANDD)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS — 15 KEY CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "CACNA1D (3p14.3)",
        "definition": "Calcium Voltage-Gated Channel Subunit Alpha1 D. Encodes Cav1.3 (α1D), the low-threshold L-type (dihydropyridine-sensitive) HVA Ca²⁺ channel. Key tissues: cochlear inner hair cells (IHC synaptic ribbon transmission), SA/AV node pacemaker cells, adrenal zona glomerulosa, dopaminergic SNc/VTA neurons, cortical/hippocampal neurons. Gene: 49 exons, 2181 aa; OMIM *114215.",
    },
    {
        "term": "Cav1.3 Low-Threshold Activation (V1/2 ≈ −40 to −55 mV)",
        "definition": "Cav1.3 uniquely activates at sub-threshold potentials (V1/2 −40 to −55 mV) compared to Cav1.2 (V1/2 −10 to −20 mV). This enables Cav1.3 to serve as a PACEMAKER channel in SA node and IHC ribbon synapses. The low threshold also means GOF mutations produce persistent window Ca²⁺ current during neuronal sub-threshold oscillations → low-threshold neuronal hyperexcitability.",
    },
    {
        "term": "SANDD Syndrome (OMIM #614896)",
        "definition": "Sinoatrial node Dysfunction and Deafness. AR biallelic LOF CACNA1D → loss of Cav1.3 in (1) cochlear IHC ribbon synapses → profound congenital SNHL (flat ABR; DPOAEs may be present initially reflecting outer hair cell integrity); (2) SA node pacemaker cells → sick sinus syndrome (SSS), bradycardia, syncope. Pacemaker implantation + cochlear implant are treatments of choice.",
    },
    {
        "term": "GOF-DEE Autism Primary Aldosteronism",
        "definition": "CACNA1D GOF de novo dominant → DEE + autism + primary aldosteronism triad. Adrenal zona glomerulosa Cav1.3 GOF → autonomous aldosterone secretion → hypertension + hypokalemia. Neuronal GOF → cortical Ca²⁺ overload → DEE. Both components respond to isradipine (Cav1.3-selective DHP) + MRA (mineralocorticoid receptor antagonist) respectively.",
    },
    {
        "term": "Isradipine (DHP Cav1.3 Precision Blocker)",
        "definition": "Dihydropyridine (DHP) class L-type Ca²⁺ channel blocker with ~10× preference for Cav1.3 over Cav1.2 at physiological holding potentials (state-dependent block). Used in STEADY trial for Parkinson's neuroprotection (NCT02168842 — negative for PD primary endpoint; confirmed Cav1.3 engagement). Precision therapy in CACNA1D GOF epilepsy. Contraindicated in LOF/SANDD.",
    },
    {
        "term": "Cav1.3 vs Cav1.2 DHP Selectivity",
        "definition": "Critical pharmacological distinction: Cav1.3 (CACNA1D) is more sensitive to isradipine/nitrendipine at hyperpolarized holding potentials; Cav1.2 (CACNA1C) is more sensitive to verapamil (intracellular blocker). Therefore: CACNA1C GOF (Timothy Syndrome) → verapamil preferred; CACNA1D GOF → isradipine preferred. Do NOT swap: verapamil has limited Cav1.3 selectivity vs Cav1.2 under physiological conditions.",
    },
    {
        "term": "Primary Aldosteronism (CACNA1D-GOF)",
        "definition": "Autonomous zona glomerulosa Cav1.3 GOF → persistent low-threshold Ca²⁺ influx in adrenocortical cells → constitutive aldosterone synthesis → elevated aldosterone:renin ratio → hypertension + hypokalemia. Biochemically mimics aldosterone-producing adenoma but no solitary adenoma (bilateral Cav1.3 GOF). Treat with mineralocorticoid receptor antagonist (MRA) + isradipine (dual mechanism). Adrenalectomy NOT appropriate.",
    },
    {
        "term": "Cochlear Inner Hair Cell (IHC) Cav1.3 Dependence",
        "definition": "Cochlear IHC ribbon synapses require Cav1.3 (not Cav1.2) for graded synaptic Ca²⁺ signalling to afferent spiral ganglion neurons. Cav1.3 low-threshold enables IHC to respond to near-resting basilar membrane deflections. CACNA1D LOF → no IHC synaptic Ca²⁺ → flat ABR (auditory brainstem response) → profound SNHL from birth. Outer hair cells (OHC) express Cav1.2 → DPOAEs may be initially present. CI effective because spiral ganglion neurons remain intact.",
    },
    {
        "term": "Sick Sinus Syndrome (SSS) / SANDD Cardiac",
        "definition": "SA node dysfunction in SANDD LOF: Cav1.3 provides ~50% of SA node pacemaker Ca²⁺ depolarization current. LOF → reduced diastolic depolarization rate → sinus bradycardia, sinus pauses, SSS. Risk of syncope/cardiac arrest without pacemaker. PACEMAKER IMPLANTATION is class I indication in SANDD with symptomatic SSS. Distinguish from CACNA1C TS where LQTS8 (QTc >500 ms + 2:1 AV block from repolarization defect) is the cardiac phenotype.",
    },
    {
        "term": "POLG1 / Alpers-Huttenlocher Syndrome",
        "definition": "POLG1 encodes mitochondrial DNA polymerase γ. LOF mutations → mitochondrial DNA depletion. VPA in POLG1-LOF background → additional mitochondrial toxicity → acute fulminant hepatic failure + cortical necrosis (Alpers syndrome). CPIC Level A: test POLG1 before ANY VPA prescription. If POLG1 pathogenic → VPA ABSOLUTE CI → use LEV/CLB/ZNS instead. Same rule applies to CACNA1D-DEE as all epilepsy syndromes.",
    },
    {
        "term": "VPPP — Valproate Pregnancy Prevention Programme (MHRA 2021)",
        "definition": "UK MHRA 2021 regulatory requirement: all females of childbearing potential (≥12y) on VPA must be enrolled in VPPP (annual risk acknowledgement form + REMS; effective contraception confirmed; prescriber + patient both sign). NTD risk 1–2%, major congenital malformations 10%, cognitive impairment 30–40%. FA 5 mg/day mandatory. Same requirement in CACNA1D-DEE as any VPA user.",
    },
    {
        "term": "West Syndrome / Infantile Spasms",
        "definition": "West Syndrome triad: infantile spasms + hypsarrhythmia (EEG) + psychomotor regression. Peak onset 4–12 months. ACTH + prednisolone Level A; vigabatrin Level A (monitor VFD). CACNA1D GOF → West syndrome in 58% of DEE cohort. GOF-specific add-on: isradipine if GOF confirmed. SANDD + West: prefer ACTH over VGB (dual sensory impairment risk from VFD in deaf patient).",
    },
    {
        "term": "STEADY Trial (NCT02168842 — Isradipine in Parkinson)",
        "definition": "Phase III RCT of isradipine 10 mg/day vs placebo in early Parkinson's disease. Primary endpoint: UPDRS motor progression — negative (no benefit for PD motor outcome at 36 months). However: trial confirmed isradipine CNS penetration and Cav1.3 engagement in vivo. Safety profile established for long-term use. CACNA1D-GOF epilepsy: STEADY negative for Parkinson primary does not negate epilepsy use — different mechanism (Ca²⁺ overload in GOF epilepsy vs Ca²⁺ overload in Parkinson dopaminergic degeneration).",
    },
    {
        "term": "Cochlear Implant (CI) in SANDD Deafness",
        "definition": "Cochlear implant is highly effective in SANDD because CACNA1D LOF causes cochlear hair cell synaptic failure but spiral ganglion neurons (SGN) remain structurally and functionally intact. CI bypasses the IHC synaptic layer and directly stimulates SGN. Optimal implantation by 12 months (critical language acquisition period). Bilateral CI in some centres. Outcome: near-normal speech understanding in quiet; background noise challenging. SGN integrity confirmed by electrically evoked ABR (eABR) pre-implant.",
    },
    {
        "term": "SUDEP (Sudden Unexpected Death in Epilepsy)",
        "definition": "SUDEP: unexpected death of person with epilepsy, no other cause found. Risk factors: nocturnal unwitnessed GTCS, DRE, prone position, male sex, young adult, polypharmacy reduction. CACNA1D-DEE patients with DRE: annual SUDEP counselling mandatory. Nocturnal seizure detection devices (Empatica E4 / Nightwatch / Emfit bed sensor). Avoid prone sleep posture. SUDEP annual risk ~1:500 in DRE.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — 12 CLINICAL DECISION POINTS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Sinus bradycardia HR <40 bpm (SANDD)", "action": "Urgent cardiology referral; pacemaker evaluation; 48h Holter immediately"},
    {"threshold": "QTc <480 ms (CACNA1D — unlike CACNA1C TS)", "action": "CACNA1D does NOT cause LQTS8; QTc typically normal; if QTc>480 consider concurrent CACNA1C variant or drug effect"},
    {"threshold": "Aldosterone:Renin ratio >30 (GOF aldosteronism)", "action": "Endocrinology referral; MRA (spironolactone/eplerenone) initiation; isradipine as dual-mechanism treatment"},
    {"threshold": "K⁺ <3.2 mmol/L (aldosteronism-driven hypokalemia)", "action": "Oral potassium supplementation + MRA dose increase; review isradipine dose (aldosterone suppression may normalize K⁺)"},
    {"threshold": "Seizure cluster ≥3 seizures/24h", "action": "Rescue benzodiazepine (buccal midazolam 0.3 mg/kg or rectal diazepam 0.5 mg/kg); emergency plan activation; ER if >5 minutes continuous seizure"},
    {"threshold": "Spasm clusters persisting >2 weeks on ACTH", "action": "Add vigabatrin or switch to high-dose prednisolone; EEG urgently to assess hypsarrhythmia resolution; KD initiation considered"},
    {"threshold": "VPA ALT >3× ULN or ammonia >80 μmol/L", "action": "Hold VPA; hepatology review; if POLG1 positive on retrospective testing → permanent VPA discontinuation; L-carnitine 100 mg/kg/day"},
    {"threshold": "Isradipine BP <85/50 mmHg (pediatric) or <100/60 mmHg (adult)", "action": "Hold isradipine; review dose; ensure hydration; position patient supine; resume at 50% dose next day"},
    {"threshold": "Seizure-free ≥2 years (AED wean consideration)", "action": "Structured AED taper over ≥6 months with EEG monitoring; CACNA1D GOF often requires lifelong AED; high recurrence rate on withdrawal in GOF-DEE"},
    {"threshold": "VFD documented on ERG/formal perimetry (VGB use)", "action": "Discontinue vigabatrin immediately; ophthalmology; reassess antiseizure strategy; VFD is IRREVERSIBLE"},
    {"threshold": "Hypsarrhythmia absent on EEG at 2 weeks of ACTH", "action": "Good prognostic sign → complete ACTH course (4 weeks total) → taper; neurodevelopmental intensive rehabilitation"},
    {"threshold": "Aldosterone normalization on MRA (ARR <10)", "action": "Maintain MRA dose; consider isradipine reduction if BP well-controlled; annual ARR reassessment"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS — 12 GUIDELINES
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE Classification 2022", "applies": "Seizure type + syndrome classification; GOF-DEE and SANDD classification within genetic DEE spectrum"},
    {"name": "NICE NG217 (2022)", "applies": "Epilepsy diagnosis and management; AED first-line recommendations; referral pathways for infantile spasms"},
    {"name": "Baig 2011 Nat Genet 43:776", "applies": "SANDD syndrome original description — first biallelic CACNA1D LOF family; SA node + IHC phenotype defined"},
    {"name": "Scholl 2013 Nat Genet 45:1050", "applies": "CACNA1D GOF causing primary aldosteronism + epilepsy; zona glomerulosa Ca²⁺ channel GOF mechanism"},
    {"name": "Pinggera 2015 Cell 160:1037", "applies": "CACNA1D activating mutations in autism/DEE; window current expansion mechanism; Cav1.3 GOF patch clamp data"},
    {"name": "STEADY Trial NCT02168842", "applies": "Isradipine phase III in Parkinson's; establishes CNS Cav1.3 engagement and safety profile for isradipine in vivo"},
    {"name": "CPIC Guideline POLG 2023", "applies": "Pharmacogenomics — VPA contraindicated in POLG1 pathogenic variant carriers; mandatory testing protocol"},
    {"name": "MHRA VPPP 2021", "applies": "UK Valproate Pregnancy Prevention Programme — mandatory for females ≥12y on VPA; risk form + effective contraception"},
    {"name": "UKISS 2004 (UK Infantile Spasms Study)", "applies": "ACTH vs vigabatrin for West syndrome; EEG response at 14 days primary endpoint; Mackay 2004 Lancet"},
    {"name": "SHARE REMS Programme (FDA/Vigabatrin)", "applies": "VGB visual field monitoring programme — ERG q3M mandatory; enrollment required for VGB prescription in USA"},
    {"name": "ACMG-AMP Variant Interpretation 2015", "applies": "CACNA1D variant classification (pathogenic/likely pathogenic/VUS); GOF functional assay evidence; segregation"},
    {"name": "WHO ICF (2019)", "applies": "International Classification of Functioning — dual disability framework for SANDD (hearing + cardiac) and GOF-DEE (epilepsy + ASD + ID)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES — 6 KEY
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"author": "Baig SM et al.", "year": 2011, "journal": "Nat Genet 43:776–778", "title": "Loss of Cav1.3 (CACNA1D) function in a human channelopathy with bradycardia and congenital deafness", "pmid": "21685912"},
    {"author": "Scholl UI et al.", "year": 2013, "journal": "Nat Genet 45:1050–1054", "title": "Somatic and germline CACNA1D Ca2+ channel mutations in aldosterone-producing adenomas and primary aldosteronism", "pmid": "23913001"},
    {"author": "Pinggera A et al.", "year": 2015, "journal": "Cell 160:1037–1043", "title": "CACNA1D de novo mutations in autism spectrum disorders activate Cav1.3 L-type Ca2+ channels", "pmid": "25768904"},
    {"author": "Ortner NJ & Striessnig J", "year": 2016, "journal": "Pflugers Arch 468:451–464", "title": "L-type voltage gated Ca2+ channels as drug targets in channelopathies", "pmid": "26680404"},
    {"author": "Bhatt DL et al. (STEADY)", "year": 2020, "journal": "JAMA Neurol 77:577–587", "title": "Isradipine versus placebo in early Parkinson disease (STEADY trial)", "pmid": "32150232"},
    {"author": "Striessnig J et al.", "year": 2014, "journal": "J Pharmacol Exp Ther 348:346–358", "title": "Pharmacology of L-type calcium channels: Novel drugs for old targets?", "pmid": "24319080"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC COHORT — 40 PATIENTS
# ─────────────────────────────────────────────────────────────────────────────
_etiology_pool = (
    ["GOF-DEE-Autism-Primary-Aldosteronism"] * 14 +
    ["GOF-DEE-Autism-Normotensive"] * 11 +
    ["LOF-SANDD-Biallelic"] * 8 +
    ["GOF-Mosaic-Partial-Phenotype"] * 5 +
    ["Phenocopy-Panel-Negative"] * 2
)

_aed_pool = [
    ["ACTH", "LEV"], ["ACTH", "VGB"], ["ACTH", "VPA", "LEV"],
    ["LEV", "VPA"], ["LEV", "CLB"], ["ACTH", "Isradipine", "LEV"],
    ["VPA", "CLB", "Isradipine"], ["LEV", "Isradipine"], ["KD"],
    ["LEV", "KD"], ["CLB", "LEV"], ["ACTH", "LEV", "CLB"],
    ["Isradipine", "VPA", "LEV"], ["Spironolactone", "LEV", "VPA"],
]

_outcomes = ["seizure-free", "≥50%-reduction", "partial-response", "DRE"]
_outcome_wts = [0.20, 0.35, 0.25, 0.20]

_cohort = []
for i in range(40):
    et = _etiology_pool[i]
    is_gof = "GOF" in et
    is_sandd = "SANDD" in et
    age = random.randint(4, 58)
    onset = max(1, age - random.randint(1, min(age - 1, 10)))
    aeds = _aed_pool[i % len(_aed_pool)]
    bp = 130 + random.randint(0, 30) if is_gof and "Aldosteronism" in et else 110 + random.randint(-10, 15)
    k = round(3.0 + random.uniform(0, 1.2), 1) if is_gof and "Aldosteronism" in et else round(3.8 + random.uniform(0, 0.8), 1)
    outcome = random.choices(_outcomes, weights=_outcome_wts)[0]
    _cohort.append({
        "patient_id": f"P{i+1:03d}",
        "age_years": age,
        "onset_age_months": onset * 12 if onset < 5 else onset * 6,
        "etiology": et,
        "variant_class": "GOF-de-novo" if is_gof else ("LOF-biallelic" if is_sandd else "unknown"),
        "seizure_free": outcome == "seizure-free",
        "dre": outcome == "DRE",
        "outcome": outcome,
        "aeds": aeds,
        "asd_diagnosis": is_gof and random.random() < 0.82,
        "id_severity": random.choice(["mild", "moderate", "severe", "profound"]) if is_gof else "none",
        "sandd_deafness": is_sandd,
        "sandd_bradycardia": is_sandd and random.random() < 0.75,
        "pacemaker_implanted": is_sandd and random.random() < 0.55,
        "cochlear_implant": is_sandd and random.random() < 0.70,
        "aldosteronism": is_gof and "Aldosteronism" in et,
        "systolic_bp": bp,
        "k_mmol_l": k,
        "isradipine_rx": "Isradipine" in aeds,
    })


# ─────────────────────────────────────────────────────────────────────────────
# API RESPONSE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(_cohort)
    seizure_free = sum(1 for p in _cohort if p["seizure_free"])
    dre = sum(1 for p in _cohort if p["dre"])
    gof = sum(1 for p in _cohort if "GOF" in p["etiology"])
    sandd = sum(1 for p in _cohort if p["sandd_deafness"])
    aldosteronism = sum(1 for p in _cohort if p["aldosteronism"])
    pacemaker = sum(1 for p in _cohort if p["pacemaker_implanted"])
    ci = sum(1 for p in _cohort if p["cochlear_implant"])
    isradipine = sum(1 for p in _cohort if p["isradipine_rx"])
    asd = sum(1 for p in _cohort if p.get("asd_diagnosis"))

    etiology_dist = {}
    for et_spec in ETIOLOGY_CATALOG:
        count = sum(1 for p in _cohort if p["etiology"] == et_spec["category"])
        etiology_dist[et_spec["category"]] = count

    seizure_summary = [
        {"type": s["type"], "frequency_pct": s["frequency_pct"]}
        for s in SEIZURE_TYPES
    ]
    treatment_summary = [
        {"drug": t["drug"].split(" (")[0].split(" —")[0][:40], "level": t["level"].split(" (")[0]}
        for t in TREATMENTS
    ]
    monitoring_summary = [
        {"item": m["item"].split(" (")[0][:50], "frequency": m["frequency"].split(";")[0]}
        for m in MONITORING[:8]
    ]
    lifecycle_summary = [
        {"stage": lc["stage"][:55], "key_action": lc["key_action"][:80]}
        for lc in LIFECYCLE
    ]

    return {
        "gene": "CACNA1D",
        "locus": "3p14.3",
        "protein": "Cav1.3 (α1D) — Low-Threshold L-type HVA Ca²⁺ Channel",
        "channel_type": "L-type HVA; V1/2 ≈ −40 to −55 mV (lower than Cav1.2's −10 to −20 mV)",
        "syndrome": "SANDD (Sinoatrial node Dysfunction and Deafness) [LOF] / DEE + Autism + Primary Aldosteronism [GOF]",
        "omim": "#614896 SANDD · *114215 CACNA1D gene",
        "inheritance": "AD GOF de novo (DEE) · AR LOF biallelic (SANDD)",
        "color": "#1a237e",
        "precision_tx": "Isradipine (GOF) · Pacemaker + Cochlear Implant (SANDD LOF)",
        "total_patients": n,
        "seizure_free_pct": round(seizure_free / n * 100, 1),
        "dre_pct": round(dre / n * 100, 1),
        "gof_count": gof,
        "sandd_count": sandd,
        "aldosteronism_count": aldosteronism,
        "pacemaker_count": pacemaker,
        "cochlear_implant_count": ci,
        "isradipine_rx_count": isradipine,
        "asd_count": asd,
        "etiology_distribution": etiology_dist,
        "seizure_summary": seizure_summary,
        "treatments_summary": treatment_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": lifecycle_summary,
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [
            {"drug": ci_["drug"].split(" —")[0].split(" (")[0][:45], "risk": ci_["risk"][:80]}
            for ci_ in CONTRAINDICATIONS[:5]
        ],
    }


def get_breakdown():
    return {
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "patients": _cohort,
    }


def get_definitions():
    return {
        "gene_summary": {
            "gene": "CACNA1D",
            "full_name": "Calcium Voltage-Gated Channel Subunit Alpha1 D",
            "chromosome": "3p14.3",
            "protein": "Cav1.3 (α1D) — Low-Threshold L-type HVA Ca²⁺ Channel",
            "size": "2181 aa · 49 exons · α1D + β2/β3 + α2δ subunits",
            "channel_type": "L-type (dihydropyridine-sensitive); LOW-THRESHOLD HVA (V1/2 ≈ −40 to −55 mV — activates at near-resting membrane potential; UNIQUE among L-types)",
            "activation_threshold": "V1/2 −40 to −55 mV (Cav1.3) vs −10 to −20 mV (Cav1.2) — 20–30 mV leftward shift enables pacemaker and IHC ribbon functions",
            "primary_location": "Cochlear IHC ribbon synapses (deafness in LOF) · SA/AV node pacemaker (bradycardia/SSS in LOF) · Adrenal zona glomerulosa (aldosteronism in GOF) · Dopaminergic SNc/VTA neurons · Cortical/hippocampal neurons",
            "cav1_subfamily": "Cav1.1/CACNA1S (1q32.1 skeletal/MH) · Cav1.2/CACNA1C (12p13.33 cardiac/TS-LQTS8) · Cav1.3/CACNA1D (3p14.3 cochlear+pacemaker/SANDD+GOF-DEE) · Cav1.4/CACNA1F (Xp11.23 retinal/CSNB2)",
            "inheritance": "AD GOF de novo (DEE+autism+aldosteronism, >95% de novo); AR biallelic LOF (SANDD — bradycardia+deafness; no epilepsy typically); pLI ~0.87",
            "omim": "OMIM #614896 SANDD · *114215 CACNA1D gene",
            "precision_treatment": "Isradipine (Level C, GOF only — DHP Cav1.3-preferential at hyperpolarized Vh) · Cochlear implant + Pacemaker (SANDD LOF)",
            "absolute_ci": "TGB (NCSE) · VPA+POLG1 (Alpers) · Isradipine in LOF/SANDD (cardioinhibitory) · VGB long-term (VFD) · Class III antiarrhythmics in SANDD SSS",
            "key_distinction_vs_CACNA1C": "CACNA1C (Cav1.2 TS): LQTS8 QTc>500ms + syndactyly + verapamil-precision · CACNA1D (Cav1.3): NO LQTS8 (QTc normal in GOF); deafness+bradycardia in LOF/SANDD; isradipine-precision; aldosteronism in GOF",
        },
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
