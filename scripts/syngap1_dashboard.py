"""
SYNGAP1 Encephalopathy (SYNGAPathy / SYNGAP1-DEE / MRD5)
==========================================================
41-patient cohort · SYNGAP1 (6p21.32) · SynGAP1 · Ras/Rap GTPase-activating protein
SYNGAP1-related intellectual disability and DEE (SYNGAPathy): de novo heterozygous
pathogenic variants in SYNGAP1 (6p21.32), encoding the Ras/Rap GTPase-activating
protein SynGAP1, cause one of the most common single-gene intellectual disability +
epilepsy syndromes. SYNGAP1 accounts for ~1-3% of unexplained moderate-severe
intellectual disability and is the second most frequently mutated gene in epileptic
DEE cohorts (after SCN1A/Dravet).

SYNGAP1 BIOLOGY: SynGAP1 is a master regulator of Ras-ERK-MAPK and Rap-p38-MAPK
signalling pathways in dendritic spines. It is highly concentrated at the postsynaptic
density (PSD) of excitatory synapses, where it constitutes ~5% of PSD protein content.
SynGAP1 functions as a brake on Ras activity — upon NMDA receptor activation → Ca²⁺
influx → CaMKII → SynGAP1 phosphorylation → translocation out of PSD → Ras-GTP
accumulation → ERK/MAPK activation → AMPA receptor insertion → LTP. Haploinsufficiency
(one functional copy) removes this brake: basal Ras-ERK over-activation → premature
spine maturation → excessive synaptic strengthening → loss of Hebbian plasticity
specificity → intellectual disability + epilepsy. The excitatory/inhibitory imbalance
arises from cortical network hyperexcitability (excess AMPA receptor surface expression)
and impaired GABAergic interneuron integration.

EPILEPSY PHENOTYPE — SYNGAPathy: Seizures typically emerge at 2–5 years after a period
of developmental delay (DDD since infancy). The epilepsy has a distinctive multi-type
phenotype:
① Myoclonic-atonic seizures (drop attacks) — the most disabling type in SYNGAP1;
  sudden flexion + atonia → child collapses forward; highest injury risk
② Eyelid myoclonia with absences — pathognomonic of SYNGAP1; exquisitely sensitive
  to EYE CLOSURE (EC-sensitivity) and photic stimulation; resembles Jeavons syndrome
③ Atypical absence seizures — prolonged, variable duration, less abrupt than typical absence
④ Generalized tonic-clonic (GTCS) — less frequent but high parental anxiety

The EEG hallmarks of SYNGAP1 are: (a) eye-closure-induced paroxysmal activity (ECIPA),
(b) photoparoxysmal response (PPR) to IPS, (c) generalised spike-wave during HV, and
(d) background slowing proportional to cognitive severity. The combination of eye-closure
sensitivity + myoclonic-atonic + intellectual disability strongly suggests SYNGAP1.

TREATMENT — KEY POINTS:
• VPA (valproate) is the cornerstone — broadest efficacy across all SYNGAP1 seizure
  types (myoclonic-atonic, absence, GTCS). Level B evidence.
• Etosuximide (ETH) is synergistic with VPA for atypical absence and eyelid myoclonia
  — the VPA+ETH combination is the first-line standard for absence-predominant SYNGAP1.
• Clobazam (CLB) as add-on reduces myoclonic-atonic frequency.
• KETOGENIC DIET is the most effective intervention for drug-resistant drop attacks
  in SYNGAP1 — ~50-60% ≥50% seizure reduction in atonic-predominant patients.
• AVOID: Carbamazepine, oxcarbazepine, phenytoin — sodium channel blockers
  exacerbate myoclonic-atonic seizures in generalised epilepsies.
• CAUTION: Lamotrigine — unpredictable in SYNGAP1; some patients worsen myoclonic
  component; monitor closely and withdraw immediately if drop attack frequency increases.
• Fenfluramine is being investigated — SYNGAP1 mouse models show FFA reduces
  hyperexcitability (Ras-ERK pathway; serotonin-mediated K+ channel activation).

INVESTIGATIONAL / PRECISION MEDICINE:
• MEK inhibitors (PD0325901, Selumetinib) — direct Ras-ERK pathway blockade;
  SYNGAP1 mouse rescue studies show cognitive improvement; Phase I/II in planning.
• ASO gene therapy — SynGAP1 protein restoration via mRNA stabilisation approaches.
• SYNGAP1 Research Fund drives clinical trial pipeline — enrolment via SYNGAP1RF.org.

SAFETY PEARLS:
• DROP ATTACK HELMET: mandatory for fall protection in myoclonic-atonic SYNGAP1.
• PHOTO AVOIDANCE: light-modulating glasses, screen filters, avoid flickering lights.
• EYE-CLOSURE TESTING: IPS + EC EEG protocol mandatory at diagnosis.
• STRABISMUS: ~30% of SYNGAP1 patients — annual ophthalmology referral.
• AUTISM SCREENING: ~50% meet ASD criteria — ADOS-2/ADI-R by 36 months.
"""

import random
from datetime import datetime

SEED = 9181  # dashboard 181
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "Pathogenic SYNGAP1 truncating / frameshift — severe DEE (de novo)",
        "n": 16, "pct": 40,
        "category": "De-novo-SYNGAP1-truncating-frameshift-DEE-severe",
        "mechanism": (
            "Most prevalent class (~40%): de novo nonsense, frameshift, or canonical splice-site "
            "variants causing premature stop codon (PTC) and NMD-mediated mRNA degradation → "
            "SynGAP1 haploinsufficiency. Loss of one functional SYNGAP1 allele reduces SynGAP1 "
            "PSD protein content by ~50%, removing the Ras-ERK brake in dendritic spines. "
            "Basal Ras-GTP accumulation → constitutive ERK phosphorylation → premature spine "
            "maturation → excessive AMPA receptor surface trafficking → network hyperexcitability "
            "and loss of Hebbian plasticity specificity. Severe phenotype: profound intellectual "
            "disability (ID, Bayley-III <1st percentile in most), severe speech/language delay, "
            "epilepsy onset 2-4 years, frequent drug-resistant drop attacks. ~60% meet ASD "
            "criteria. Hypotonia and strabismus common (~35%). De novo confirmed by trio "
            "exome/genome sequencing. Truncating variants at all SYNGAP1 exons are pathogenic — "
            "PTC before exon 12 causes complete haploinsufficiency (most severe); PTC near "
            "C-terminus may have partial loss phenotype. SYNGAP1 ClinGen dosage sensitivity: "
            "Haploinsufficiency Score 3 (Sufficient Evidence)."
        ),
        "eeg_signature": (
            "Hallmark SYNGAP1-DEE EEG: generalised spike-wave (2.5-4 Hz) with DRAMATIC "
            "eye-closure sensitivity (EC-paroxysmal activity — ECIPA). Waking: diffuse background "
            "slowing proportional to cognitive severity (theta-dominant background in most); "
            "generalised polyspike-wave during HV. Photic stimulation: photoparoxysmal response "
            "(PPR) type II-IV in ~65% (generalised spike-wave outlasting photic stimulus). "
            "Myoclonic-atonic (drop attack) ictal: generalised polyspike → atonic phase on EMG. "
            "Sleep: sleep-potentiated spike-wave; no CSWS pattern (unlike EAS syndromes)."
        ),
        "mri": (
            "Usually normal or non-specific volume loss proportional to ID severity. No "
            "structural malformations — normal MRI in SYNGAP1 does not exclude diagnosis. "
            "MRI mandatory at diagnosis to exclude structural epilepsy substrate. Rarely: "
            "periventricular white matter signal change (non-specific). Corpus callosum "
            "thinning in ~8% of severe phenotype."
        ),
        "clinical_note": (
            "DROP ATTACK HELMET mandatory in myoclonic-atonic SYNGAP1 — head injury from "
            "forward falls is the leading acute morbidity. Prescribe protective headgear at "
            "first visit. AVOID CBZ/OXC/PHT — worsening of myoclonic-atonic seizures is a "
            "recognised class effect of Na-channel blockers in generalised epilepsies. "
            "VPA + ETH combination is first-line; if drops persist after 2 AEDs → refer for "
            "ketogenic diet evaluation. SYNGAP1 genetic result: trio exome is gold standard; "
            "targeted gene panels acceptable if SYNGAP1 is included with all exons and "
            "intron-exon boundaries covered."
        ),
    },
    {
        "etiology": "Pathogenic SYNGAP1 missense (LOF) — moderate DEE (de novo)",
        "n": 12, "pct": 29,
        "category": "De-novo-SYNGAP1-missense-LOF-moderate",
        "mechanism": (
            "De novo missense variants affecting SynGAP1 functional domains (~29%). Key domains "
            "affected: RasGAP catalytic domain (residues 727-1147) — most pathogenic missense "
            "variants cluster here, reducing GAP catalytic efficiency; PH/C2 domains "
            "(membrane-targeting); coiled-coil domain (oligomerisation); PDZ-binding motif at "
            "C-terminus (interaction with PSD-95 and MUPP1 — scaffold localisation). Missense "
            "LOF variants reduce Ras-GAP activity by 30-80% (variant-dependent functional assay "
            "data) → partial Ras-ERK over-activation → intermediate phenotype. Moderate-severe "
            "intellectual disability; epilepsy typically less drug-resistant than truncating class; "
            "eyelid myoclonia and absence more prominent than drop attacks in some missense "
            "subgroups. ASD features in ~45%. Strabismus in ~25%. Genotype-phenotype correlation: "
            "missense in RasGAP domain = most severe; PDZ domain missense = milder (incomplete "
            "PSD targeting). ClinVar SYNGAP1 missense: interpret with functional GAP assay data "
            "(PS3 criteria) — pathogenic in silico alone insufficient (many missense VUS in SYNGAP1)."
        ),
        "eeg_signature": (
            "Eye-closure sensitivity (ECIPA) and photoparoxysmal response present but often less "
            "pronounced than truncating class. Generalised 3-4 Hz spike-wave during HV. Background "
            "slowing moderate (theta) rather than severe (delta). Myoclonic-atonic less frequent; "
            "eyelid myoclonia + absence predominate in missense subgroup. Sleep: normal sleep "
            "architecture; no CSWS. Seizure frequency variable — some patients have long seizure-"
            "free periods on VPA alone."
        ),
        "mri": "Normal in 88% of missense class. Non-specific white matter signal in 2/12.",
        "clinical_note": (
            "Missense SYNGAP1 VUS interpretation challenge: many missense variants of uncertain "
            "significance (VUS) exist in ClinVar. Request: (1) parental testing to confirm de "
            "novo status (strongest evidence), (2) functional GAP assay data if published for "
            "specific variant, (3) SynGAP1 protein quantification from lymphoblasts (if available). "
            "Phenotypic consistency (ID + EC-sensitivity EEG + absence + SYNGAP1 missense de "
            "novo) supports likely pathogenic reclassification. Refer to SynGAP Research Fund "
            "registry to contribute genotype-phenotype data."
        ),
    },
    {
        "etiology": "Pathogenic SYNGAP1 splice-site variant — DEE (de novo)",
        "n": 6, "pct": 15,
        "category": "De-novo-SYNGAP1-splice-site-DEE",
        "mechanism": (
            "Canonical ±1/2 or deep-intronic SYNGAP1 splice variants (~15%). Aberrant splicing "
            "causes exon skipping (partial LOF), intron retention (LOF or NMD-triggering), or "
            "cryptic exon activation. Phenotype intermediate to severe depending on residual "
            "SynGAP1 functional protein. Canonical splice-site variants (PVS1_Strong by ACMG) "
            "should be classified as pathogenic/likely pathogenic. Deep-intronic variants may "
            "be missed on standard exome sequencing — WGS + RNA sequencing from fibroblasts or "
            "blood required for deep-intronic VUS with strong phenotypic concordance. Alternative "
            "splicing of SYNGAP1 generates multiple isoforms (α1, α2, β, γ) with distinct "
            "C-terminal sequences — isoform-specific effects on PSD targeting and GAP activity "
            "add complexity to genotype-phenotype correlation. α1 isoform (longest, most "
            "abundant in cortex) — CaM-binding at C-terminus; α2 — coiled-coil C-terminal; "
            "β — ACTH-binding domain. Splice variants affecting α1-specific exons (17-20) may "
            "disproportionately impair CaM-regulated GAP activity."
        ),
        "eeg_signature": (
            "Generalised spike-wave with eye-closure sensitivity in 4/6 patients. One patient "
            "with deep-intronic variant had focal onset (temporo-parietal) — unusual for SYNGAP1 "
            "but reported in splice variants affecting specific isoforms. Sleep EEG: generalised "
            "IED activation during drowsiness; no CSWS. HV: robust generalised spike-wave "
            "activation in 5/6 (distinguishes from focal epilepsy misdiagnosis)."
        ),
        "mri": "Normal in 5/6. One patient: periventricular nodular heterotopia (incidental — causal role uncertain).",
        "clinical_note": (
            "If SYNGAP1 splice variant found on standard exome (shallow intronic coverage): "
            "request RNA sequencing from fibroblasts to confirm aberrant splicing (converts "
            "VUS to LP). WGS improves detection of deep-intronic SYNGAP1 variants in negative "
            "exome with strong phenotype. ACMG PVS1 criteria: canonical ±1/2 splice variant "
            "in SYNGAP1 = PVS1_Strong → classify as likely pathogenic even without RNA data "
            "if phenotype concordant and de novo confirmed."
        ),
    },
    {
        "etiology": "SYNGAP1 CNV deletion 6p21 — DEE (de novo)",
        "n": 4, "pct": 10,
        "category": "De-novo-SYNGAP1-CNV-deletion-6p21-DEE",
        "mechanism": (
            "Contiguous gene deletion at 6p21 encompassing SYNGAP1 (~10%). Deletions range "
            "from focal SYNGAP1-only deletions (pure haploinsufficiency, phenotype similar to "
            "point mutation) to large 6p21 deletions encompassing additional genes (DAAM2, "
            "GFAP, VEGFA, C6orf48) — larger deletions have additional syndromic features. "
            "Chromosomal microarray (CMA/SNP array) or WGS detects CNVs missed by standard "
            "exome. In this cohort: 2 patients with isolated SYNGAP1 deletion (40-120 kb, "
            "exons 4-18); 2 patients with larger 6p21 deletion (>500 kb) with additional "
            "cardiac and ophthalmological features. ClinGen SYNGAP1 dosage sensitivity: "
            "Haploinsufficiency Score 3 — deletions are pathogenic regardless of size if "
            "SYNGAP1 coding region is disrupted. Large 6p21 deletions: refer to clinical "
            "genetics for comprehensive multi-organ assessment (echocardiogram, ophthalmology)."
        ),
        "eeg_signature": (
            "Similar to point mutation class: generalised spike-wave, EC-sensitivity, PPR. "
            "Two patients with large 6p21 deletion had additional focal temporal spikes "
            "(possible contribution from co-deleted gene). Background more severely slowed "
            "in large CNV patients (reflecting broader haploinsufficiency)."
        ),
        "mri": "Focal deletion: normal (2/2). Large 6p21 deletion: mild cortical volume loss + thin CC (2/2).",
        "clinical_note": (
            "CNV at 6p21 detected on microarray — confirm SYNGAP1 is within deletion by "
            "breakpoint mapping (WGS or aCGH with targeted probe density). Report co-deleted "
            "genes for clinical genetics assessment. Cardiac screening for large 6p21 CNV "
            "(VEGFA haploinsufficiency may impact vasculogenesis). Ophthalmology referral "
            "mandatory for large 6p21 deletions."
        ),
    },
    {
        "etiology": "Clinical SYNGAP1-negative phenocopy — unexplained DEE with SYNGAP1-like features",
        "n": 3, "pct": 6,
        "category": "Clinical-SYNGAP1-negative-phenocopy",
        "mechanism": (
            "Patients with SYNGAP1-like phenotype (ID + myoclonic-atonic + EC-sensitivity) but "
            "no SYNGAP1 pathogenic variant (~6%). Differential: (1) SYNGAP1 somatic mosaicism "
            "(detectable only by deep sequencing of blood >500× or neuronal tissue); (2) Other "
            "Ras-MAPK pathway genes — BRAF, RAF1, MAP2K1/2, SHOC2 (RASopathies with epilepsy); "
            "(3) Myoclonic-atonic epilepsy without genetic diagnosis — Doose syndrome (30-50% "
            "no gene found); (4) Other DEE genes with myoclonic-atonic: KIAA2022, NEXMIF, "
            "KCNB1, SLC6A1 (myoclonic-atonic epilepsy gene par excellence), CHD2 (photosensitive "
            "myoclonic-atonic); (5) Chromosomal imbalance on microarray not yet detected. "
            "Management: treat seizure phenotype regardless of genetic result — VPA + ETH "
            "first-line for myoclonic-atonic + absence phenotype; KD for drug-resistance."
        ),
        "eeg_signature": "EC-sensitivity and PPR present in 2/3 — EEG indistinguishable from SYNGAP1-positive cases.",
        "mri": "Normal in all 3. Structural MRI does not exclude genetic DEE.",
        "clinical_note": (
            "SYNGAP1-negative phenocopy: order (1) chromosomal microarray (CMA), (2) comprehensive "
            "DEE gene panel (500+ genes including SLC6A1, CHD2, NEXMIF, KIAA2022, KCNB1, BRAF), "
            "(3) metabolic screen (urine amino acids, organic acids, CSF glucose/lactate), "
            "(4) deep sequencing for somatic SYNGAP1 mosaicism if high phenotypic suspicion. "
            "SLC6A1 (myoclonic-atonic epilepsy) is the most important differential — often "
            "has similar myoclonic-atonic + absence phenotype, variable EC-sensitivity."
        ),
    },
]

# ── Seizure Types (4) ──────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Myoclonic-atonic seizures / drop attacks (SYNGAP1 signature)",
        "prevalence_pct": 80,
        "onset_age": "2–5 years (median ~3 years)",
        "eeg_correlate": (
            "HALLMARK SYNGAP1 EEG: generalised polyspike (2-4 Hz) → atonic phase correlating "
            "with drop. Surface EMG: brief myoclonic burst (50-200 ms) followed by generalised "
            "EMG atonia (200-500 ms) → fall. EEG during drop: rapid polyspike → generalised "
            "voltage attenuation. CRITICAL: drops are often VERY BRIEF (child may fall and "
            "immediately stand again — family may not recognise as seizure). Video-EEG with "
            "EMG is mandatory for accurate seizure classification. Myoclonic-atonic EEG is "
            "distinct from pure atonic (no preceding myoclonia) and pure myoclonic (no atonia). "
            "Waking EEG background: theta-dominant slowing proportional to cognitive severity. "
            "Eye-closure induces paroxysmal activity (ECIPA) in ~72% — the EC sensitivity is "
            "PATHOGNOMONIC in combination with myoclonic-atonic + ID."
        ),
        "clinical_tip": (
            "DROP ATTACK HELMET MANDATORY from first clinic visit — head trauma from forward "
            "fall is the primary injury in SYNGAP1 drop attacks. Prescribe immediately. "
            "Video-EEG with EMG essential for diagnosis — parent descriptions of 'stumbling' "
            "or 'clumsiness' often represent unrecognised drop attacks. Teach parents to video "
            "events on smartphone. EEG must include PHOTIC STIMULATION and EYE-CLOSURE testing "
            "(EC in darkness: child closes eyes → look for EC-induced paroxysms). "
            "AVOID CBZ/OXC/PHT — sodium channel blockers classically worsen myoclonic-atonic "
            "seizures in generalised epilepsies."
        ),
    },
    {
        "type": "Eyelid myoclonia with absences (eye-closure sensitivity / Jeavons-like)",
        "prevalence_pct": 70,
        "onset_age": "2–6 years (often co-incident with or following myoclonic-atonic onset)",
        "eeg_correlate": (
            "PATHOGNOMONIC SYNGAP1 SIGNATURE: rapid eyelid myoclonia (flickering) induced by "
            "eye closure (EC) and/or photic stimulation. EEG: high-amplitude generalised "
            "spike-wave (4-6 Hz) triggered within 0.5-3 seconds of eye closure. ECIPA "
            "(Eye-Closure-induced Paroxysmal Activity) — spike-wave burst suppressed by eye "
            "opening. This EC-sensitivity is PRESENT IN ~70% OF SYNGAP1 patients and is far "
            "more prevalent than in other DEE syndromes. Photoparoxysmal response (PPR) Type "
            "II-IV on IPS in ~50-65% — most sensitive at 15-25 Hz. Eyelid myoclonia absences: "
            "brief (3-6 seconds), subtle — child appears to 'flutter' or 'stare with blinking'. "
            "MUST test EC and IPS in all SYNGAP1 patients — failure to test = missed eyelid "
            "myoclonia diagnosis → inadequate treatment."
        ),
        "clinical_tip": (
            "EC-SENSITIVITY TESTING PROTOCOL: room lights on → child asked to close eyes → "
            "observe for eyelid flutter + EEG paroxysm → immediately open eyes → EEG suppression. "
            "Repeat in darkened room (darkness amplifies EC-sensitivity). Test must be done "
            "AT EVERY ROUTINE EEG — EC-sensitivity may vary with AED status. "
            "PHOTIC STIMULATION: IPS from 1-60 Hz — PPR peaks 15-25 Hz in SYNGAP1. "
            "MANAGEMENT: ETH (etosuximide) is the most effective drug for EC-sensitivity + "
            "eyelid myoclonia. VPA+ETH combination superior to VPA alone for this seizure type. "
            "Light-modulating glasses (FL-41 tint) reduce photic trigger. Advise avoidance of: "
            "sunlight flickering through trees, video games, disco lights, TV at close distance."
        ),
    },
    {
        "type": "Atypical absence seizures",
        "prevalence_pct": 65,
        "onset_age": "2–5 years",
        "eeg_correlate": (
            "Generalised 2.5-3.5 Hz spike-wave (slower than typical childhood absence 3 Hz, "
            "more irregular) — consistent with atypical absence. Duration variable: 5-30 "
            "seconds. HV activation prominent. Onset/offset less abrupt than typical absence "
            "(not so sudden — child may have prodromal blink or eye-flutter). SYNGAP1 atypical "
            "absence may be confused with typical CAE — distinguish by: slower ictal SW "
            "frequency, background slowing, concurrent myoclonic-atonic, EC-sensitivity, "
            "intellectual disability (absent in CAE). Video-EEG: reduced responsiveness during "
            "spike-wave burst; may continue walking slowly (unlike CAE — which causes abrupt "
            "behavioural arrest)."
        ),
        "clinical_tip": (
            "SYNGAP1 atypical absence responds to ETH (etosuximide) and VPA — combination "
            "superior to monotherapy for absence-predominant SYNGAP1. Distinguish from "
            "childhood absence epilepsy (CAE): SYNGAP1 has intellectual disability, myoclonic- "
            "atonic seizures, EC-sensitivity — CAE has none of these. ETH dose: 250-500 mg/day "
            "(child), 500-1500 mg/day (adolescent/adult); slow titration to minimise GI side "
            "effects. VPA+ETH combination: most effective for SYNGAP1 absence; monitor for "
            "additive thrombocytopenia."
        ),
    },
    {
        "type": "Generalised tonic-clonic seizures (GTCS)",
        "prevalence_pct": 55,
        "onset_age": "3–6 years (often triggered by fever or missed AED)",
        "eeg_correlate": (
            "Classic generalised GTCS EEG: recruiting rhythm → high-amplitude polyspike-wave → "
            "post-ictal generalised voltage attenuation and slowing. GTCS in SYNGAP1 typically "
            "triggered by fever, missed AED, or sleep deprivation — not spontaneous. Duration "
            "usually 1-3 minutes. Secondary GTCS from focal-onset rare in pure SYNGAP1 — if "
            "focal onset precedes GTCS → investigate for co-existing focal structural epilepsy. "
            "SUDEP risk: elevated in SYNGAP1-DEE — nocturnal GTCS monitoring advisable (bed "
            "alarm, SIDS monitor in younger children). Seizure action plan mandatory: rescue "
            "medication (buccal midazolam) for prolonged GTCS >5 minutes."
        ),
        "clinical_tip": (
            "FEVER PROTOCOL: SYNGAP1 families must have a written fever action plan — "
            "antipyretics (paracetamol/ibuprofen) at first sign of fever; rescue medication "
            "prescribed (buccal midazolam). Fever triggers GTCS in ~62% of SYNGAP1 patients. "
            "SUDEP: counsel families — ensure they understand nocturnal seizure risk; "
            "recommend sleeping position (supine or lateral) and seizure alarms. "
            "VPA is first-line for GTCS in SYNGAP1 — excellent efficacy for generalised seizures. "
            "AVOID: CBZ/OXC/PHT (worsens myoclonic-atonic and does not help GTCS in "
            "generalised epilepsy; risk of aggravation outweighs any potential GTCS benefit)."
        ),
    },
]

# ── Triggers (8) ──────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Eye-closure (ECIPA — Eye-Closure-induced Paroxysmal Activity, pathognomonic)",
        "rate_pct": 72,
        "mechanism": (
            "Physiological eye-closure triggers paroxysmal EEG activity in SYNGAP1 — ECIPA. "
            "Mechanism: alpha rhythm suppression with eye closure (Berger effect) → transient "
            "cortical hyperexcitability window → generalised spike-wave in sensitised SYNGAP1 "
            "network. EC in darkness amplifies response (eliminates visual suppression of "
            "alpha). ECIPA is SPECIFIC to certain generalised epilepsies: Jeavons syndrome, "
            "SYNGAP1-DEE, and rarely Dravet/CHD2. In SYNGAP1, ECIPA prevalence (~70%) is "
            "uniquely high and often the first EEG clue to the diagnosis."
        ),
        "management": (
            "Teach family: avoid sudden eye-closure in bright light (Venetian blind effect). "
            "Photochromic glasses outdoors. FL-41 tint for indoor flickering lights. "
            "ETH (etosuximide) is most effective drug for EC-triggered eyelid myoclonia. "
            "EEG protocol: always test eye-closure in standard EEG recording."
        ),
    },
    {
        "trigger": "Photic stimulation (IPS / photoparoxysmal response, PPR)",
        "rate_pct": 68,
        "mechanism": (
            "Intermittent photic stimulation (IPS) induces PPR in ~50-68% of SYNGAP1 patients — "
            "one of the highest photosensitivity rates in DEE syndromes. SYNGAP1 photosensitivity "
            "peaks at 15-25 Hz (standard IPS frequency band). Visual cortex hyperexcitability "
            "from Ras-ERK over-activation → lower threshold for photic-driven oscillation. "
            "Flickering TV screens (especially pre-digital era), video games (pattern/flash), "
            "sunlight through trees/car windows (8-25 Hz visual flicker) trigger PPR. "
            "Photoparoxysmal response Type III-IV = self-sustained GSW outlasting IPS = "
            "high clinical significance → self-sustaining discharge risk."
        ),
        "management": (
            "Photosensitivity precautions: (1) maintain distance from TV/screen (>2.5m); "
            "(2) use screen filter or blue-light glasses; (3) cover one eye when approaching "
            "flickering light source; (4) FL-41 tint glasses outdoors; (5) avoid video games "
            "with rapid visual patterns; (6) polarised sunglasses in bright sunlight. "
            "VPA reduces photosensitivity in ~60% of SYNGAP1. LEV also anti-photosensitive "
            "effect (adjunct role). ETH primarily effective for EC-seizures, less so for IPS-"
            "triggered generalised seizures."
        ),
    },
    {
        "trigger": "Fever / febrile illness",
        "rate_pct": 62,
        "mechanism": (
            "Fever lowers seizure threshold across all SYNGAP1 seizure types — particularly "
            "GTCS and myoclonic-atonic clusters. Hyperthermia → reduced GABA-A receptor function "
            "→ network hyperexcitability → lowered threshold in already-sensitised Ras-ERK "
            "hyperactive SYNGAP1 cortex. Febrile GTCS in SYNGAP1 may be prolonged (>5 minutes) "
            "→ require rescue medication. SYNGAP1 patients do not have the temperature-sensitive "
            "Na-channel mechanism of Dravet (SCN1A) but fever is still a common trigger via "
            "systemic hyperthermia and sleep disruption during illness."
        ),
        "management": (
            "FEVER ACTION PLAN (mandatory, written): (1) Temperature ≥37.5°C → paracetamol "
            "15 mg/kg immediately (don't wait for higher fever); (2) antipyretic alternation "
            "(paracetamol + ibuprofen 4-hourly); (3) rescue medication: buccal midazolam "
            "0.3-0.5 mg/kg (max 10 mg) if GTCS >5 minutes or cluster >3 drops in 1 hour; "
            "(4) emergency contact numbers at bedside; (5) hospital threshold: prolonged "
            "seizure or GTCS clustering. Educate family and school staff. VPA + paracetamol "
            "interaction (minor — mild VPA level elevation): monitor clinically."
        ),
    },
    {
        "trigger": "Missed / late AED dose",
        "rate_pct": 55,
        "mechanism": (
            "Missed AED dose → rapid decline in serum AED level → lowered seizure threshold → "
            "myoclonic-atonic cluster or GTCS in SYNGAP1. VPA has 8-16h half-life — a single "
            "missed dose causes significant level drop within 12-24h. ETH has 30-60h half-life "
            "— slightly more forgiving. CLB active metabolite norclobazam: 70-100h half-life — "
            "most forgiving. Patient/caregiver reliability with AED adherence critical: SYNGAP1 "
            "children cannot self-manage — complete caregiver dependence on medication delivery. "
            "School AED protocols essential — midday doses must be supervised by trained staff."
        ),
        "management": (
            "ADHERENCE STRATEGIES: (1) pill organiser + phone alarm reminders; (2) school "
            "AED administration plan (EHCP/IEP medication section); (3) MMAS-8 adherence "
            "screening at each clinic visit; (4) 'missed dose' rescue: if VPA dose missed "
            "<4h → take immediately; if >4h → skip and resume next scheduled dose (do NOT "
            "double-dose); (5) home diary: track each dose given; (6) for travelling: "
            "carry AED in hand luggage + written prescription for international travel. "
            "VPA twice-daily extended-release preferred over three-times-daily for adherence."
        ),
    },
    {
        "trigger": "Stress / anxiety / emotional arousal",
        "rate_pct": 48,
        "mechanism": (
            "Emotional stress triggers myoclonic-atonic clusters in ~48% of SYNGAP1 caregivers' "
            "reports. Mechanism: stress → HPA axis activation → cortisol → amygdala-mediated "
            "cortical arousal → reduced GABAergic inhibitory tone → network hyperexcitability. "
            "ASD-related anxiety (50% SYNGAP1 have ASD) compounds the stress-seizure cycle: "
            "sensory overload → anxiety → increased myoclonic-atonic frequency → more anxiety. "
            "School transitions, examinations, new environments, and social demands are common "
            "precipitants in school-age SYNGAP1 children."
        ),
        "management": (
            "ASD/anxiety management integral to epilepsy management in SYNGAP1. Refer to "
            "paediatric psychologist/psychiatrist for ASD-focused anxiety intervention. "
            "School: sensory-friendly classroom, consistent routine, advance warning of "
            "changes. SSRI for anxiety (fluoxetine preferred — minimal epilepsy risk; avoid "
            "bupropion which lowers seizure threshold). Melatonin for sleep-anxiety comorbidity. "
            "Caregiver education: anticipate seizure clusters after stressful events; have "
            "rescue medication accessible."
        ),
    },
    {
        "trigger": "Sleep deprivation / disrupted sleep",
        "rate_pct": 42,
        "mechanism": (
            "Sleep deprivation is a well-recognised generalised epilepsy trigger. In SYNGAP1, "
            "sleep architecture is often abnormal — REM behaviour disorder, night wakings, "
            "and OSA (obstructive sleep apnoea — ~20% SYNGAP1 due to hypotonia) compound the "
            "seizure risk. Sleep deprivation → reduced NREM slow-wave sleep → loss of sleep- "
            "dependent seizure suppression → next-day myoclonic-atonic increase. SYNGAP1 "
            "families commonly experience caregiver sleep deprivation from nocturnal seizures "
            "→ caregiver burnout → adherence difficulties → vicious cycle."
        ),
        "management": (
            "SLEEP HYGIENE: regular bedtime, dark/quiet room, screen-free 1h before bed "
            "(screen light triggers EC-seizures AND disrupts circadian rhythm). Melatonin "
            "1-5 mg at bedtime for sleep-onset difficulties (safe, effective in ASD/epilepsy). "
            "Sleep study (polysomnography) if OSA suspected — hypotonia + SYNGAP1 increases "
            "OSA risk; untreated OSA worsens seizure control and cognitive function. "
            "Caregiver support: overnight respite care, peer caregiver support networks "
            "(SynGAP Research Fund Family Network)."
        ),
    },
    {
        "trigger": "Screen use / video games / visual pattern stimuli",
        "rate_pct": 35,
        "mechanism": (
            "Visual pattern stimuli and flickering from screens trigger photosensitive seizures "
            "in SYNGAP1 — particularly eyelid myoclonia and myoclonic-atonic clusters. Video "
            "game consoles (30-60 fps frame rate), striped patterns, scrolling text, and "
            "rapidly changing scenes are recognised triggers. Social media video (TikTok, "
            "YouTube Shorts with rapid cuts) increasingly reported as trigger. This is a "
            "lifestyle trigger that parents and schools need concrete guidance on."
        ),
        "management": (
            "SCREEN PRECAUTIONS: (1) maintain TV/monitor distance ≥2.5 m; (2) reduce "
            "screen brightness and contrast; (3) use epilepsy-safe display settings (no "
            "rapid flash modes); (4) avoid video games with stroboscopic effects (check "
            "Harding Test compliance — UK games industry standard for photosensitive "
            "epilepsy); (5) blue-light filtering glasses during screen use; (6) limit "
            "continuous screen sessions to <30 min with break; (7) advise school: "
            "computer screens at safe distance, epilepsy-safe software only."
        ),
    },
    {
        "trigger": "Hyperventilation (HV)",
        "rate_pct": 28,
        "mechanism": (
            "Hyperventilation (HV) triggers atypical absence seizures and myoclonic-atonic "
            "clusters in SYNGAP1 via hypocapnia → cerebral vasoconstriction → cortical "
            "excitability increase → generalised spike-wave activation. HV is used as a "
            "diagnostic EEG activation procedure (3 min HV protocol) — SYNGAP1 shows "
            "robust generalised spike-wave activation during HV, diagnostic in absence-"
            "predominant cases. Real-world triggers: vigorous exercise, crying (prolonged), "
            "breath-holding episodes. Less clinically significant than eye-closure or fever."
        ),
        "management": (
            "HV as trigger: counsel families that vigorous play, crying, or exercise may "
            "trigger brief absence clusters — not a reason to restrict activity but awareness "
            "helps school staff recognise seizures. DIAGNOSTIC: HV activation in EEG is "
            "a USEFUL DIAGNOSTIC PROCEDURE in SYNGAP1 — request 3-min HV in all SYNGAP1 "
            "EEGs. HV + EC testing together provide high sensitivity for SYNGAP1 EEG signature."
        ),
    },
]

# ── Treatments (8) ────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA) — Sodium valproate / Valproic acid",
        "level": "Level B",
        "indication": "First-line — broad-spectrum: myoclonic-atonic, atypical absence, eyelid myoclonia, GTCS",
        "dose": (
            "Start: 10-15 mg/kg/day in 2 divided doses; titrate by 5-10 mg/kg/week. "
            "Target: 20-40 mg/kg/day; TDM target 50-100 μg/mL (myoclonic-atonic: aim 70-100). "
            "Chrono/extended-release formulation preferred for adherence (once/twice daily). "
            "Max: 60 mg/kg/day (weight-adjusted); avoid >2500 mg/day in adults."
        ),
        "moa": (
            "Multiple mechanisms: (1) Na-channel inactivation enhancement — reduces high-frequency "
            "repetitive firing; (2) GABA synthesis increase (promotes GAD enzyme activity); "
            "(3) T-type Ca²⁺ channel block (absence mechanism); (4) Ras-ERK modulation (preclinical "
            "data — may partially counteract SYNGAP1 haploinsufficiency pathway). VPA's "
            "broad-spectrum action across all SYNGAP1 seizure types makes it uniquely suited "
            "as the backbone of SYNGAP1 polytherapy."
        ),
        "efficacy": "≥50% reduction in myoclonic-atonic: ~60% of SYNGAP1 patients; seizure freedom: ~15-20%.",
        "safety": (
            "MANDATORY MONITORING: LFT + FBC + ammonia baseline → q3M year 1 → q6M thereafter. "
            "Teratogenicity: ABSOLUTE CI in women of childbearing potential without adequate "
            "contraception — neural tube defects, cognitive/behavioural teratogenicity (NEAD "
            "study data). VPA Prevent Register in UK. Weight gain: counsel/monitor BMI. "
            "Hair loss: biotin supplementation, reassurance. POLG testing mandatory before "
            "VPA if Alpers/POLG phenotype suspected (progressive neurological regression)."
        ),
        "monitoring": "VPA TDM (50-100 μg/mL); LFT + FBC + ammonia q3M year 1; weight/BMI; reproductive counselling in females.",
    },
    {
        "drug": "Etosuximide (ETH / Zarontin)",
        "level": "Level B",
        "indication": "First-line for eyelid myoclonia + atypical absence — used in combination with VPA",
        "dose": (
            "Start: 250 mg/day (single or twice daily); titrate by 250 mg every 1-2 weeks. "
            "Target: 500-1500 mg/day (child/adolescent); max 2000 mg/day. "
            "TDM: 40-100 μg/mL (50-80 therapeutic sweet spot for most patients). "
            "Syrup available (250 mg/5 mL) for young children."
        ),
        "moa": (
            "Primary mechanism: T-type calcium channel (Cav3.1/3.2) blockade in thalamic "
            "relay neurons — disrupts thalamo-cortical spike-wave oscillation generating "
            "absence seizures. Also modulates persistent Na-current (INaP). For eyelid "
            "myoclonia: reduces cortical excitability during alpha-frequency EC rhythm via "
            "T-channel suppression in occipital-thalamic loop. ETH is more selective for "
            "T-channels than VPA → less sedation, less weight gain, complementary mechanism "
            "to VPA → VPA+ETH combination is synergistic for absence and eyelid myoclonia."
        ),
        "efficacy": "VPA+ETH combination vs VPA monotherapy: ~30% greater absence seizure reduction in SYNGAP1 series.",
        "safety": (
            "Generally well-tolerated. GI side effects (nausea, vomiting, anorexia) most common "
            "— reduce with food, slow titration. Behavioural: agitation, mood lability in ~10% "
            "(usually dose-dependent). Rare: blood dyscrasia (aplastic anaemia — very rare, "
            "FBC if concerns). No hepatotoxicity. No teratogenicity data in SYNGAP1 specifically "
            "(use with caution in pregnancy; folic acid supplementation). Drug interaction with "
            "VPA: ETH levels may be reduced by VPA enzyme induction (VPA is enzyme inhibitor — "
            "actually raises ETH levels slightly; monitor TDM)."
        ),
        "monitoring": "ETH TDM (40-100 μg/mL); FBC q6M; behavioural monitoring; weight.",
    },
    {
        "drug": "Clobazam (CLB / Onfi / Frisium)",
        "level": "Level C",
        "indication": "Adjunct for myoclonic-atonic reduction — add-on after VPA (+/- ETH)",
        "dose": (
            "Start: 0.1-0.2 mg/kg/day at bedtime; titrate by 0.1 mg/kg/week. "
            "Target: 0.3-0.8 mg/kg/day in 2 divided doses (concentrate 60-70% at bedtime "
            "for nocturnal-predominant drops). Max: 1 mg/kg/day or 40 mg/day. "
            "Intermittent CLB: 5-10 days around high-risk periods (illness, travel)."
        ),
        "moa": (
            "GABA-A receptor positive allosteric modulator (benzodiazepine site) — preferentially "
            "binds α2 and α3 GABA-A subunits (less sedating than classic 1,4-BDZ which bind α1). "
            "Reduces myoclonic-atonic via enhanced GABAergic inhibition in cortical-thalamic "
            "circuits. Active metabolite: norclobazam (10-fold longer half-life 70-100h) — "
            "steady-state norclobazam provides sustained anti-seizure effect. Tolerance: "
            "benzodiazepine tolerance may develop at 3-12M; useful strategy is drug holiday "
            "or intermittent use."
        ),
        "efficacy": "~40-50% ≥50% reduction in drop attacks as add-on to VPA in SYNGAP1.",
        "safety": (
            "Sedation (dose-dependent — concentrate dosing at bedtime reduces daytime sedation). "
            "CYP2C19 polymorphism affects norclobazam levels: poor metabolisers (PM) have high "
            "norclobazam → increased sedation/toxicity; ultra-rapid metabolisers (UM) have low "
            "norclobazam → poor response. CYP2C19 genotyping clarifies anomalous responses. "
            "Tolerance: rotate with drug holidays (1-2 weeks CLB-free every 4-6 months). "
            "Withdrawal: taper slowly (10% per week) to avoid seizure clusters. Hyperactivity "
            "paradoxical reaction in young children with ASD — reduce dose if hyperactivity appears."
        ),
        "monitoring": "Norclobazam TDM (50-300 ng/mL); CYP2C19 genotype if anomalous response; behavioural monitoring.",
    },
    {
        "drug": "Levetiracetam (LEV / Keppra)",
        "level": "Level C",
        "indication": "Adjunct for GTCS + myoclonic component — second/third-line add-on",
        "dose": (
            "Start: 10-15 mg/kg/day in 2 divided doses; titrate by 10 mg/kg/week. "
            "Target: 30-60 mg/kg/day (child); 1000-3000 mg/day (adult). "
            "IV formulation available for acute seizure clusters."
        ),
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) ligand — modulates presynaptic vesicle "
            "release cycle; also inhibits Ca²⁺ channels and reduces intracellular Ca²⁺ "
            "buffering. Anti-myoclonic and anti-GTCS effects via SV2A at central synapses. "
            "Additionally has anti-photosensitive effect: LEV reduces PPR in IPS — useful "
            "adjunct for SYNGAP1 photosensitivity component. No significant hepatic metabolism "
            "(renal excretion) → minimal drug interactions with VPA/ETH/CLB."
        ),
        "efficacy": "~30-40% ≥50% reduction in GTCS as add-on; moderate effect on myoclonic-atonic in SYNGAP1.",
        "safety": (
            "SYNGAP1 / ASD SPECIFIC CONCERN: LEV-associated behavioural side effects "
            "(irritability, aggression, mood lability) occur in ~20-30% of neurotypical patients "
            "and are significantly MORE COMMON in children with ASD/ID — prevalence up to 40-50% "
            "in SYNGAP1 cohorts (ASD + ID confers higher risk). Monitor behaviour closely after "
            "initiation. If significant behavioural worsening → dose reduction or trial of "
            "brivaracetam (BRV — related SV2A ligand with better tolerability profile). "
            "Vitamin B6 (pyridoxine) 50-100 mg/day may mitigate LEV-associated irritability."
        ),
        "monitoring": "Behavioural monitoring (ABBQ irritability subscale at each visit); renal function annually.",
    },
    {
        "drug": "Ketogenic Diet (KD) — 4:1 or modified Atkins",
        "level": "Level B",
        "indication": "Drug-resistant myoclonic-atonic (drop attacks) — after ≥2 AED failures",
        "dose": (
            "Classic KD 4:1 (fat:protein+carbohydrate ratio) or 3:1; Modified Atkins Diet (MAD) "
            "as less restrictive alternative. Introduce over 1-2 weeks with dietitian supervision. "
            "Urine/blood ketone monitoring: target beta-hydroxybutyrate (BHB) 2-4 mmol/L for "
            "seizure control. Calorically adequate: match expected energy intake for weight. "
            "Supplement: multivitamin, calcium, selenium, zinc, carnitine."
        ),
        "moa": (
            "Multiple anti-seizure mechanisms: (1) metabolic ketosis → ketone bodies (BHB, "
            "acetoacetate) as alternative brain energy substrate → alters neuronal excitability; "
            "(2) KATP channel opening (BHB → mitochondrial KATP) → membrane hyperpolarisation; "
            "(3) reduced glucose flux → less mTOR activation → reduced cortical excitability; "
            "(4) increased GABA synthesis (ketone metabolism through glutamate decarboxylase); "
            "(5) SYNGAP1-specific preclinical data: KD reduces Ras-ERK over-activation in "
            "SYNGAP1 mouse models via metabolic pathway crosstalk — mechanistic basis for "
            "exceptional KD response in SYNGAP1."
        ),
        "efficacy": "~50-60% ≥50% drop attack reduction in SYNGAP1 (higher than average DEE population); 15-20% seizure freedom.",
        "safety": (
            "Growth monitoring essential (KD affects growth in children). Kidney stones: "
            "adequate hydration, urinary citrate levels, potassium citrate supplementation if "
            "low. Hyperlipidaemia: fasting lipid panel q6M; consider lipid-modified KD if "
            "LDL elevated. Cardiomyopathy: rare; ECG + echo at baseline + if symptoms. "
            "GI: constipation (increase fluid/fibre within carb limits); nausea during initiation. "
            "Growth: height/weight centile q3M; adjust KD prescription with dietitian if "
            "growth faltering. Selenium + zinc + carnitine supplementation: mandatory. "
            "Intercurrent illness: ketosis may deepen → monitor closely; IV glucose if unwell."
        ),
        "monitoring": "BHB 2-4 mmol/L; urine ketones daily; lipids q6M; growth q3M; renal USS annually; micronutrients.",
    },
    {
        "drug": "Fenfluramine (FFA / Fintepla) — OFF-LABEL for SYNGAP1",
        "level": "Level C (investigational / off-label)",
        "indication": "Drug-resistant myoclonic-atonic and absence — emerging option after KD failure",
        "dose": (
            "Off-label dosing based on Dravet/LGS trials: 0.1 mg/kg/day → titrate over 6 weeks "
            "to 0.2-0.35 mg/kg/day (max 26 mg/day). Twice-daily administration. "
            "FINTEPLA REMS (FDA) mandatory if used in USA — echocardiogram q6M."
        ),
        "moa": (
            "Multimechanism: (1) serotonin (5-HT) release + reuptake inhibition → 5-HT₂C "
            "receptor activation → positive modulation of voltage-gated K⁺ channels (Kv4.2) "
            "→ membrane hyperpolarisation; (2) sigma-1 receptor agonism → reduced ER stress; "
            "(3) SYNGAP1-SPECIFIC: FFA activates sigma-1 → phosphorylates SynGAP1 at S1512 "
            "(CaMKII site) → enhances residual SynGAP1 GAP activity → partial Ras-ERK "
            "normalisation in SYNGAP1 haploinsufficiency (preclinical, mouse model). This "
            "potential mechanistic synergy with SYNGAP1 haploinsufficiency is the rationale "
            "for FFA in SYNGAPathy — not yet confirmed in human trials."
        ),
        "efficacy": "Preclinical SYNGAP1 model: 40-60% reduction in myoclonic-atonic. Human SYNGAP1 data: case series only (<10 patients); promising.",
        "safety": (
            "CARDIAC MONITORING MANDATORY: echocardiography at baseline and q6M — fenfluramine "
            "historically associated with valvular heart disease (dexfenfluramine 1990s). "
            "At current low doses (≤0.35 mg/kg/day), no valvulopathy detected in Dravet trials, "
            "but monitoring remains mandatory. Anorexia/weight loss: monitor growth carefully "
            "in children (compete with KD for caloric restriction). Fatigue, somnolence: "
            "common initiation side effects. Pulmonary hypertension: measure at baseline; "
            "contraindicated if baseline pulmonary hypertension."
        ),
        "monitoring": "Echo q6M (REMS); weight monthly during titration; BP; anorexia/growth.",
    },
    {
        "drug": "Perampanel (PER / Fycompa)",
        "level": "Level C",
        "indication": "Adjunct for drug-resistant myoclonic-atonic and GTCS — AMPA receptor antagonist",
        "dose": (
            "Start: 2 mg at bedtime; titrate by 2 mg/week. Target: 8-12 mg/day at bedtime "
            "(max 12 mg/day). Once-daily bedtime dosing preferred (long half-life 105h). "
            "Reduce by 50% if given with enzyme inducers (CBZ, PHT — avoid in SYNGAP1 anyway)."
        ),
        "moa": (
            "First selective non-competitive AMPA receptor antagonist — directly reduces "
            "postsynaptic AMPA-mediated excitatory transmission. In SYNGAP1: excess AMPA "
            "receptor surface trafficking (from Ras-ERK over-activation → GluA1 phosphorylation "
            "→ AMPA insertion) is a core pathomechanism. PER directly counters this excess "
            "AMPA activity — a mechanistically rational precision approach for SYNGAP1. "
            "Anti-myoclonic effect: reduces AMPA-dependent myoclonic burst generation. "
            "Anti-GTCS: reduces generalised spread via AMPA block. Anti-photosensitive effect "
            "documented in photosensitive epilepsy trials."
        ),
        "efficacy": "~35-45% ≥50% reduction in myoclonic-atonic as add-on in generalised epilepsy; limited SYNGAP1-specific data.",
        "safety": (
            "SYNGAP1 / ASD SPECIFIC CONCERN: PER causes dizziness, irritability, aggression, "
            "and mood disturbance in up to 20-25% — again higher risk in ASD/ID population. "
            "Start LOW (2 mg) and titrate slowly; monitor ABAS behavioural score. Ataxia "
            "at higher doses: monitor gait in SYNGAP1 (pre-existing hypotonia). "
            "CAUTION: behavioural AE risk similar to LEV in ASD — monitor closely and "
            "withdraw if significant aggression or mood disturbance."
        ),
        "monitoring": "Behavioural monitoring at each visit; gait assessment; weight (anorexia at high doses).",
    },
    {
        "drug": "Lamotrigine (LTG) — CAUTION / CONDITIONAL USE ONLY",
        "level": "Level C (with CAUTION — may exacerbate myoclonic-atonic)",
        "indication": "CONDITIONAL: may be considered ONLY for GTCS-predominant SYNGAP1 with minimal myoclonic-atonic; MONITOR CLOSELY",
        "dose": (
            "Very slow titration MANDATORY: start 0.15 mg/kg/day weeks 1-2; 0.3 mg/kg/day "
            "weeks 3-4; increase by 0.3 mg/kg/every 1-2 weeks. Target: 5-15 mg/kg/day (child). "
            "Halve titration rate with VPA co-administration (VPA doubles LTG levels via UGT "
            "enzyme inhibition). Max: 15 mg/kg/day with VPA; 25 mg/kg/day without VPA."
        ),
        "moa": (
            "Na-channel inactivation enhancement (frequency-dependent) + Ca²⁺ channel block "
            "(P/Q-type). Anti-GTCS and anti-focal seizure efficacy. RISK IN SYNGAP1: LTG "
            "can EXACERBATE myoclonic-atonic seizures in some generalised epilepsies — "
            "mechanism uncertain (possible destabilisation of absence → myoclonic transition). "
            "This risk is variable in SYNGAP1: some patients tolerate well; others show "
            "dramatic worsening of drop attack frequency within 2 weeks of dose increase. "
            "THEREFORE: use ONLY when myoclonic-atonic are well-controlled by other AEDs and "
            "GTCS remains the primary treatment target."
        ),
        "efficacy": "Anti-GTCS: good. Anti-myoclonic-atonic: RISK OF WORSENING — do NOT use as primary drop-attack treatment.",
        "safety": (
            "MYOCLONIC-ATONIC EXACERBATION: if drop attack frequency increases after LTG "
            "initiation or dose increase → STOP LTG IMMEDIATELY and return to pre-LTG AED "
            "regimen. Stevens-Johnson syndrome (SJS) / TEN: risk highest during rapid "
            "titration — SLOW TITRATION IS MANDATORY. VPA+LTG combination: LTG levels "
            "doubled by VPA → risk of SJS increases if LTG titrated rapidly on VPA background. "
            "Rash: any rash after LTG initiation → stop drug, assess for SJS."
        ),
        "monitoring": "Drop attack frequency at each dose change; rash surveillance; LTG TDM (2-20 μg/mL); behavioural monitoring.",
    },
]

# ── Contraindications (4) ─────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) — AVOID",
        "severity": "AVOID — may exacerbate myoclonic-atonic seizures",
        "reason": (
            "Sodium channel blockers (CBZ, OXC, PHT, ESL) are contraindicated in generalised "
            "epilepsies with myoclonic-atonic seizures. In SYNGAP1: Na-channel blockers suppress "
            "inhibitory interneurons disproportionately → disinhibition → paradoxical increase "
            "in myoclonic-atonic seizures. Multiple case reports and DEE cohort data document "
            "dramatic worsening of drop attacks within days-weeks of CBZ/OXC initiation. "
            "Risk: any prescriber considering 'focal epilepsy' in a child with SYNGAP1-like "
            "phenotype may inadvertently prescribe CBZ → catastrophic drop attack exacerbation. "
            "If CBZ/OXC given and drop attacks worsen → STOP IMMEDIATELY."
        ),
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin — AVOID",
        "severity": "AVOID — exacerbates myoclonic-atonic seizures",
        "reason": (
            "Phenytoin (and fosphenytoin) — Na-channel blocker — shares the contraindication "
            "with CBZ in SYNGAP1 myoclonic-atonic epilepsy. Additionally: PHT has unfavourable "
            "pharmacokinetics in children (non-linear kinetics → toxicity risk), "
            "cosmetic side effects (gingival hyperplasia, hirsutism), and potential for "
            "cognitive worsening. PHT is sometimes used in acute seizure management (IV for "
            "status epilepticus) — this is ACCEPTABLE as rescue (IV fosphenytoin single-dose) "
            "if midazolam fails in status, but should NOT be initiated as maintenance AED "
            "in SYNGAP1."
        ),
    },
    {
        "drug": "Lamotrigine (LTG) — CAUTION, monitor for myoclonic exacerbation",
        "severity": "CAUTION — not absolute CI but significant worsening risk",
        "reason": (
            "Lamotrigine may exacerbate myoclonic-atonic seizures in SYNGAP1 — particularly "
            "if used as monotherapy or primary treatment for drop attacks. The mechanism is "
            "uncertain but likely involves partial Na-channel effect on inhibitory interneurons "
            "generating myoclonic-atonic bursts. Clinical instruction: DO NOT use LTG as "
            "primary treatment for myoclonic-atonic SYNGAP1. If used for GTCS with minimal "
            "drops: MONITOR weekly for first 4 weeks; immediately withdraw if drop frequency "
            "increases by >50% from baseline."
        ),
    },
    {
        "drug": "Vigabatrin (VGB) — AVOID in absence-predominant SYNGAP1",
        "severity": "AVOID — worsens generalised absence seizures",
        "reason": (
            "Vigabatrin (VGB) — GABA transaminase inhibitor — is contraindicated in "
            "generalised epilepsies with absence seizures because it paradoxically worsens "
            "spike-wave absence by disrupting GABA-mediated thalamic inhibitory gating. "
            "SYNGAP1 atypical absence is worsened by VGB. Additionally: VGB causes "
            "irreversible visual field constriction (VF defects in ~40% long-term users) — "
            "requires Goldman VF testing q6M (SHARE REMS monitoring in USA). Only acceptable "
            "use in SYNGAP1: if focal epilepsy co-exists with tuberous sclerosis complex "
            "features (rare)."
        ),
    },
]

# ── Monitoring Items (8) ──────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "VPA serum TDM (total valproate)",
        "frequency": "Steady-state (5 days after each dose change); then q3M",
        "rationale": (
            "Target 50-100 μg/mL. For myoclonic-atonic-predominant SYNGAP1: aim upper end "
            "70-100 μg/mL for maximum anti-atonic efficacy. Free VPA level if hypoalbuminaemia "
            "(low albumin displaces VPA from binding — free fraction elevated → toxicity at "
            'lower total levels). VPA + ETH combination: ETH may raise VPA levels slightly '
            "(ETH inhibits VPA metabolism marginally). VPA + CLB: mutual PK interaction — "
            "CLB raises VPA 10-20%; VPA raises norclobazam. Monitor both levels together."
        ),
    },
    {
        "item": "VPA LFTs, FBC, serum ammonia",
        "frequency": "Baseline; q3M year 1; q6M thereafter",
        "rationale": (
            "VPA hepatotoxicity: LFT >3× ULN → withhold VPA, specialist review. Age <2Y + "
            "polytherapy: highest Reye-like hepatotoxicity risk. Thrombocytopenia: FBC q6M — "
            "platelet <50,000 → reduce VPA dose. Hyperammonaemia: measure ammonia if "
            "encephalopathy, behavioural deterioration, or unexplained vomiting on VPA — "
            "ammonia >80 μmol/L with symptoms → VPA dose reduction + L-carnitine. "
            "POLG testing: mandatory before VPA if regression/failure to thrive/unexplained LFT "
            "elevation — POLG mutations cause VPA hepatotoxicity."
        ),
    },
    {
        "item": "EEG with photic stimulation (IPS) and eye-closure (EC) protocol",
        "frequency": "At diagnosis; q6-12M or after medication change",
        "rationale": (
            "EC testing and IPS mandatory in EVERY SYNGAP1 EEG — failure to test misses "
            "eyelid myoclonia and photosensitivity diagnosis. EC protocol: standard EEG + "
            "eye-closure in light → eye-closure in dark. IPS: 1-60 Hz sweep, look for PPR "
            "Type II-IV (Waltz criteria). Quantify ECIPA duration and PPR threshold frequency "
            "as treatment response biomarkers. Sleep EEG: activated EEG during drowsiness "
            "and light sleep shows IED increase — useful for severity tracking. "
            "Telemetry EEG + video-EMG essential for drop attack classification (polyspike "
            "→ atonia EMG confirms myoclonic-atonic vs pure atonic)."
        ),
    },
    {
        "item": "Neuropsychological / developmental assessment",
        "frequency": "At diagnosis; q6-12M",
        "rationale": (
            "SYNGAP1 causes progressive developmental plateau (not regression like epileptic "
            "encephalopathy) — growth of cognitive skills is impaired but not reversed. "
            "Bayley-III (0-42M): tracks cognitive, language, motor development. "
            "WAIS-IV / WISC-V (>6Y): IQ profile (typically moderate-severe ID: FSIQ 35-60). "
            "Vineland-II: adaptive behaviour — key for educational and NDIS/EHCP planning. "
            "ADI-R + ADOS-2: autism assessment at 30-36M. ADOS-2 Module 1/2 annually "
            "thereafter. PHQ-9A / CBCL: anxiety/mood from adolescence. Seizure control "
            "impacts cognitive outcomes — AED sedation also contributes (monitor CBCL "
            "attention scale with each AED change)."
        ),
    },
    {
        "item": "Ophthalmology assessment (strabismus screening)",
        "frequency": "At diagnosis; annually",
        "rationale": (
            "Strabismus occurs in ~25-35% of SYNGAP1 patients — convergent squint most common. "
            "Mechanism: SynGAP1 expression in extraocular motor nuclei (EOM nuclei) + "
            "cerebellar vermis projections → oculomotor coordination deficit. Amblyopia "
            "risk if strabismus untreated → patching, corrective lenses, surgical correction "
            "if severe. Nystagmus: less common (~8%); visual acuity testing annually. "
            "Refractive error (hyperopia, myopia) common in ID — formal ophthalmology "
            "assessment + glasses prescription prevents amblyopia."
        ),
    },
    {
        "item": "Autism / ASD screening (ADOS-2 + ADI-R)",
        "frequency": "At diagnosis (if ≥18M); then annually",
        "rationale": (
            "~50% of SYNGAP1 meet DSM-5 ASD criteria. ASD diagnosis enables access to "
            "speech-language therapy, ABA (Applied Behaviour Analysis), and NDIS/EHCP "
            "support. Untreated ASD worsens behavioural morbidity and stress-triggered "
            "seizure clusters. ADOS-2 Module 1 (pre-verbal) at 18-24M; Module 2 (phrase "
            "speech) from 2-3Y. ADI-R: parent interview for ASD history. Social communication "
            "questionnaire (SCQ) as screening tool between formal assessments. "
            "ASD + ID together predict most functional outcome in SYNGAP1 — ASD management "
            "is as important as epilepsy management."
        ),
    },
    {
        "item": "Sleep assessment + polysomnography (PSG)",
        "frequency": "Baseline assessment; PSG if OSA suspected clinically",
        "rationale": (
            "Sleep disorders in SYNGAP1: ~30% have significant sleep disturbance (SYNGAP1 "
            "Research Fund survey data). OSA risk ~15-20% (hypotonia → upper airway "
            "instability). Restless legs / periodic limb movements. REM behaviour disorder "
            "(rare). Sleep deprivation worsens seizure control in SYNGAP1. OSA: "
            "untreated reduces sleep quality → seizure aggravation + cognitive worsening. "
            "PSG referral if: snoring + sleep apnoea symptoms (witnessed apnoea, gasping, "
            "non-restorative sleep), tonsillar hypertrophy, or unexplained seizure worsening "
            "without AED change. Treatment: adenotonsillectomy (first-line for paediatric "
            "OSA) or CPAP (older adolescents/adults)."
        ),
    },
    {
        "item": "AED adherence (MMAS-8) and seizure diary review",
        "frequency": "q6M or at each clinic visit",
        "rationale": (
            "SYNGAP1 patients are fully dependent on caregiver medication administration — "
            "missed doses are caregiver-error, not patient non-compliance. MMAS-8 (8-item "
            "Morisky Medication Adherence Scale) screens for systematic adherence issues. "
            "Seizure diary: paper or app (e.g., SeizureTracker, Epsy) — track seizure "
            "frequency by type (drops separately from absences from GTCS). Drop attack "
            "frequency is primary outcome measure in clinical trials — baseline diary "
            "essential before AED change. Review diary at each visit: cluster patterns "
            "identify triggers (illness days, school stress, screen use). "
            "Consider continuous seizure monitoring: GTC seizure detector watch "
            "(Embrace2/Empatica, EpiWatch) for nocturnal GTCS monitoring + SUDEP risk."
        ),
    },
]

# ── Lifecycle Windows (6) ────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Infancy — developmental delay detection (0–18 months)",
        "key_events": (
            "Global developmental delay (GDD) apparent from 6-12M: delayed motor milestones "
            "(sitting >9M, walking >18M), absent or limited babbling, hypotonia. "
            "No seizures yet in most. Strabismus may be apparent by 12M. "
            "SYNGAP1 often NOT diagnosed at this stage — GDD evaluated by paediatrician "
            "without genetic testing."
        ),
        "management_focus": (
            "Early intervention services: physiotherapy, occupational therapy, speech-language "
            "therapy from diagnosis. Referral to clinical genetics for GDD workup — "
            "chromosomal microarray + gene panel including SYNGAP1. Ophthalmology referral "
            "for strabismus. Family support and counselling."
        ),
    },
    {
        "window": "Toddler — first seizures (18 months – 3 years)",
        "key_events": (
            "First myoclonic-atonic seizures or atypical absences appear. Eyelid myoclonia "
            "begins — family notices eye-fluttering with eye-closure. GDD continues. "
            "ASD features emerge (reduced joint attention, limited language). "
            "Seizures may be misdiagnosed as 'febrile convulsions' or 'clumsiness'."
        ),
        "management_focus": (
            "Video-EEG with EC and IPS protocol. SYNGAP1 genetic testing if not done. "
            "Initiate VPA +/- ETH for seizure control. Drop attack helmet IMMEDIATELY. "
            "ASD assessment (ADOS-2 Module 1). Seizure action plan and family education. "
            "Register with SYNGAP1 Research Fund (SYNGAP1RF.org) natural history study."
        ),
    },
    {
        "window": "Preschool — seizure peak (3–6 years)",
        "key_events": (
            "Myoclonic-atonic drop attacks at peak frequency — up to 10-50 drops/day "
            "in severe cases. Eyelid myoclonia with absences concurrent. Photosensitivity "
            "emerges. Injury from falls (head/face trauma) is major morbidity. "
            "Language delay severe — many non-verbal or single-word phrase users by age 5. "
            "ASD confirmed in ~50%. School entry typically in specialist ASD/ID settings."
        ),
        "management_focus": (
            "Optimise AED regimen (VPA + ETH; add CLB or LEV if needed). KD evaluation "
            "if ≥2 AED failures with >50% drop attacks remaining. Protective helmet always "
            "worn. Seizure diary to quantify drops (separate count from absences/GTCS). "
            "School placement: SEND/special educational needs environment with 1:1 support. "
            "ABA therapy for ASD. Wheelchair/mobility assessment if falls prevent safe "
            "ambulation."
        ),
    },
    {
        "window": "School age — drug-resistant epilepsy management (6–12 years)",
        "key_events": (
            "Drug-resistant epilepsy (DRE) declared if ≥2 AED failures. KD or VNS/device "
            "therapy considered. Drop attacks may plateau or slowly reduce. ASD management "
            "intensive. Puberty may alter seizure pattern. Adaptive skills develop slowly "
            "with intensive intervention. Sleep disorders peak (OSA, sleep disruption)."
        ),
        "management_focus": (
            "KD evaluation if not yet trialled (highest evidence for drops). VNS/RNS/DBS "
            "referral for DRE not responding to KD. Fenfluramine or perampanel as add-on. "
            "Annual ophthalmology. Sleep study if OSA suspected. EHCP/IEP school plan "
            "review. Transition planning for secondary school. SYNGAP1 clinical trial "
            "enrolment (check clinicaltrials.gov for SYNGAP1 studies). "
            "MMAS-8 adherence + seizure diary review every 6M."
        ),
    },
    {
        "window": "Adolescence (12–18 years)",
        "key_events": (
            "Seizure frequency may improve in some SYNGAP1 patients (25-30%) entering "
            "adolescence — less consistent than Dravet/CSWS. Puberty may transiently worsen "
            "GTCS. ASD social challenges intensify. Reproductive health planning essential "
            "(VPA teratogenicity — switch females to alternative if possible). "
            "Transition to adult services begins at 14-16Y."
        ),
        "management_focus": (
            "REPRODUCTIVE COUNSELLING: VPA in females → switch to alternative if possible "
            "(LEV, CLB, KD) — VPA teratogenicity risk demands early counselling in female "
            "SYNGAP1 adolescents. Menstrual irregularity on VPA: gynaecology referral. "
            "Transition to adult neurology: structured handover plan. Supported living "
            "assessment (intellectual disability social care). SYNGAP1 research enrolment "
            "(Phase I/II MEK inhibitor or ASO trials if available)."
        ),
    },
    {
        "window": "Adulthood (18+ years)",
        "key_events": (
            "Most SYNGAP1 adults require supported living (community residential or family "
            "home with carer). Epilepsy in ~80% continues to adulthood (unlike some childhood "
            "syndromes). Seizures may reduce in severity/frequency but rarely resolve. "
            "ID and ASD are lifelong. Employment: supported employment possible in mild cases; "
            "sheltered/day centre for most."
        ),
        "management_focus": (
            "Adult neurology service with DEE expertise. Continue AED optimisation. "
            "AED simplification: if seizures well-controlled → trial careful reduction of "
            "polypharmacy (one drug at a time, with EEG monitoring). "
            "Physical health screening: metabolic syndrome (VPA weight gain + low activity), "
            "osteoporosis (VPA effect on bone density — DXA scan, vitamin D + calcium "
            "supplementation). Mental health: depression and anxiety screening annually. "
            "SYNGAP1 Research Fund: lifelong patient registry participation supports "
            "natural history data collection for future therapy development."
        ),
    },
]

# ── Concepts (14) ────────────────────────────────────────────────────────────
CONCEPTS = [
    {
        "term": "SYNGAP1-SynGAP1-RasGAP-dendritic-spine",
        "definition": (
            "SYNGAP1 (6p21.32) encodes SynGAP1 — a Ras/Rap GTPase-activating protein "
            "constituting ~5% of postsynaptic density (PSD) protein at excitatory synapses. "
            "SynGAP1 is the molecular brake on Ras-ERK-MAPK signalling in dendritic spines, "
            "regulating AMPA receptor trafficking and synaptic strength."
        ),
    },
    {
        "term": "Ras-ERK-MAPK-pathway-haploinsufficiency",
        "definition": (
            "SYNGAP1 haploinsufficiency removes the Ras-GAP brake → basal Ras-GTP elevation → "
            "constitutive ERK phosphorylation → excess AMPA receptor (GluA1) surface trafficking "
            "→ premature spine maturation → network hyperexcitability + loss of synaptic "
            "plasticity specificity → intellectual disability + epilepsy."
        ),
    },
    {
        "term": "SYNGAP1-DEE-MRD5-ILAE-2022",
        "definition": (
            "SYNGAP1-related DEE (OMIM 612621; MRD5 = Mental Retardation, Autosomal Dominant 5) "
            "is formally recognised by ILAE 2022 as a genetic DEE subtype. Characterised by "
            "de novo heterozygous SYNGAP1 pathogenic variants, moderate-severe ID, and a "
            "distinctive epilepsy phenotype (myoclonic-atonic + eyelid myoclonia + EC-sensitivity)."
        ),
    },
    {
        "term": "Myoclonic-atonic-seizures-drop-attacks",
        "definition": (
            "Myoclonic-atonic seizures: brief (50-200 ms) generalised myoclonic jerk followed "
            "immediately by generalised atonic phase → forward fall (drop attack). EEG: "
            "polyspike → generalised attenuation. Most disabling SYNGAP1 seizure type — "
            "head/face trauma is primary injury. Protective helmet mandatory from first presentation."
        ),
    },
    {
        "term": "Eyelid-myoclonia-eye-closure-sensitivity-ECIPA",
        "definition": (
            "Eyelid myoclonia: rapid eyelid flutter triggered by eye closure (EC) and/or "
            "photic stimulation (IPS). ECIPA (Eye-Closure-induced Paroxysmal Activity): "
            "generalised spike-wave within 0.5-3 seconds of EC on EEG. Present in ~70% "
            "of SYNGAP1 — pathognomonic combination with myoclonic-atonic + ID. "
            "Also seen in Jeavons syndrome and CHD2-DEE."
        ),
    },
    {
        "term": "Photosensitivity-SYNGAP1-PPR",
        "definition": (
            "Photoparoxysmal response (PPR) to intermittent photic stimulation (IPS): present "
            "in ~50-65% of SYNGAP1. Peak sensitivity at 15-25 Hz IPS. PPR Type III-IV "
            "(self-sustained GSW outlasting IPS) indicates high clinical photosensitivity. "
            "Environmental precautions essential: screen distance, FL-41 glasses, avoiding "
            "flickering light sources."
        ),
    },
    {
        "term": "CBZ-OXC-PHT-AVOID-myoclonic-atonic",
        "definition": (
            "Sodium channel blockers (carbamazepine, oxcarbazepine, phenytoin, eslicarbazepine) "
            "are contraindicated in generalised epilepsies with myoclonic-atonic seizures — "
            "they paradoxically exacerbate drop attacks by disproportionately suppressing "
            "inhibitory interneuron firing. SYNGAP1 prescribers must be aware of this "
            "contraindication to avoid catastrophic drop attack worsening."
        ),
    },
    {
        "term": "Ketogenic-Diet-drop-attacks-SYNGAP1",
        "definition": (
            "The Ketogenic Diet (4:1 classical or modified Atkins) achieves ~50-60% ≥50% "
            "reduction in myoclonic-atonic drop attacks in SYNGAP1 — one of the highest "
            "responder rates in DEE syndromes. Preclinical data: KD reduces Ras-ERK "
            "over-activation in SYNGAP1 mouse models. Considered after ≥2 AED failures."
        ),
    },
    {
        "term": "Fenfluramine-SYNGAP1-investigational",
        "definition": (
            "Fenfluramine (FFA) is being investigated in SYNGAP1-DEE based on preclinical data: "
            "FFA activates sigma-1 receptor → phosphorylates SynGAP1 at S1512 → partially "
            "restores GAP activity → reduces Ras-ERK over-activation. Phase II trials in "
            "SYNGAP1 not yet completed; off-label use case series showing promise in drug-"
            "resistant myoclonic-atonic."
        ),
    },
    {
        "term": "MEK-inhibitor-Ras-ERK-pathway-preclinical",
        "definition": (
            "MEK inhibitors (PD0325901, Selumetinib/AZD6244, Binimetinib) directly block "
            "ERK1/2 phosphorylation downstream of constitutive Ras-GTP in SYNGAP1. "
            "Mouse model: MEK inhibition rescues cognitive deficits + reduces seizure-like "
            "activity. Phase I/II trials in SYNGAP1-DEE in planning. Precision medicine "
            "approach targeting the primary pathomechanism."
        ),
    },
    {
        "term": "VPA-ETH-combination-myoclonic-atonic-absence",
        "definition": (
            "VPA + Etosuximide combination is the first-line polytherapy for SYNGAP1 with "
            "both myoclonic-atonic and absence/eyelid myoclonia components. Complementary "
            "mechanisms: VPA (broad-spectrum Na-channel + GABA) + ETH (selective T-type "
            "Ca²⁺ channel) → synergistic seizure control with manageable side-effect profile."
        ),
    },
    {
        "term": "SYNGAP1-Research-Fund-patient-org",
        "definition": (
            "The SynGAP Research Fund (SYNGAP1RF.org) is the international patient advocacy "
            "organisation for SYNGAP1-DEE. Services: natural history registry, clinical trial "
            "pipeline, family support network, clinician directory. Referral at diagnosis "
            "enables registry enrolment and future trial access."
        ),
    },
    {
        "term": "ASD-ID-SYNGAP1-comorbidity",
        "definition": (
            "~50% of SYNGAP1 patients meet DSM-5 ASD criteria; 100% have intellectual "
            "disability (mild to profound; typically moderate-severe). ASD + ID together "
            "are the primary long-term functional determinants in SYNGAP1 — early intensive "
            "ASD intervention (ABA, SLT, OT) is as important as epilepsy management. "
            "ADOS-2/ADI-R at 30-36 months mandatory."
        ),
    },
    {
        "term": "Strabismus-ophthalmology-SYNGAP1",
        "definition": (
            "Strabismus (convergent squint) in ~25-35% of SYNGAP1 — among the highest rates "
            "of any single-gene neurodevelopmental disorder. Annual ophthalmology review "
            "mandatory. SynGAP1 is expressed in extraocular motor nuclei and cerebellar "
            "projections — oculomotor coordination deficit is a direct SYNGAP1 phenotypic "
            "feature, not a secondary effect of ID."
        ),
    },
]

# ── Clinical Standards (8) ────────────────────────────────────────────────────
STANDARDS = [
    {
        "code": "ILAE-2022",
        "title": "ILAE 2022 Epilepsy Syndrome Classification",
        "scope": (
            "SYNGAP1-DEE formally recognised as genetic DEE subtype in ILAE 2022 "
            "classification. Provides diagnostic framework for myoclonic-atonic + "
            "eyelid myoclonia + EC-sensitivity phenotype."
        ),
    },
    {
        "code": "NICE-NG217",
        "title": "NICE Guideline NG217 (Epilepsies 2022)",
        "scope": (
            "UK standard for AED selection in childhood epilepsies. NG217 Section 4 "
            "(generalised epilepsy): VPA + ETH as combination options for myoclonic-atonic; "
            "CBZ/OXC contraindicated in generalised epilepsy with myoclonic features."
        ),
    },
    {
        "code": "ILAE-Dietary-Therapies-2018",
        "title": "ILAE Task Force on Dietary Therapies 2018",
        "scope": (
            "International consensus on KD indication, initiation, monitoring, and "
            "discontinuation in drug-resistant epilepsy. Standard protocol for BHB "
            "monitoring and KD micronutrient supplementation."
        ),
    },
    {
        "code": "ACMG-AMP-2015",
        "title": "ACMG-AMP Variant Interpretation Standards 2015",
        "scope": (
            "SYNGAP1 variant classification — haploinsufficiency gene (PVS1 applicable "
            "to truncating/splice variants). ClinGen SYNGAP1 Expert Panel curations "
            "available on ClinVar. PVS1_Strong for truncating; functional GAP assay "
            "data as PS3 for missense classification."
        ),
    },
    {
        "code": "ACNS-EEG-2021",
        "title": "ACNS Guidelines for EEG",
        "scope": (
            "Standardised EEG protocol including photic stimulation (IPS) and "
            "eye-closure testing methodology. PPR classification (Waltz criteria). "
            "Video-EMG for myoclonic-atonic seizure classification."
        ),
    },
    {
        "code": "Hamdan-2009-NatGenet",
        "title": "Hamdan FF et al. 2009 Nature Genetics — SYNGAP1 discovery",
        "scope": (
            "Landmark discovery: de novo SYNGAP1 heterozygous mutations cause non-syndromic "
            "intellectual disability. First genotype-phenotype characterisation of SYNGAP1-DEE. "
            ">600 citations."
        ),
    },
    {
        "code": "Mignot-2016-Brain",
        "title": "Mignot C et al. 2016 Brain — SYNGAP1 phenotypic spectrum",
        "scope": (
            "Largest SYNGAP1 cohort at time of publication (57 patients). Comprehensive "
            "phenotypic characterisation: myoclonic-atonic, eyelid myoclonia, EC-sensitivity, "
            "ASD prevalence, genotype-phenotype correlations."
        ),
    },
    {
        "code": "Vlaskamp-2019-Neurology",
        "title": "Vlaskamp DRM et al. 2019 Neurology — seizure characterisation",
        "scope": (
            "Definitive seizure characterisation study: myoclonic-atonic 80%, eyelid "
            "myoclonia 70%, atypical absence 65%, GTCS 55%. Video-EEG with EMG data "
            "establishing seizure taxonomy for SYNGAP1-DEE."
        ),
    },
]

# ── Clinical Thresholds (10) ────────────────────────────────────────────────────
THRESHOLDS = [
    {
        "threshold": "DDD onset red flag for SYNGAP1",
        "value": "Global developmental delay apparent 12-24M",
        "action": "Chromosomal microarray + DEE gene panel including SYNGAP1; clinical genetics referral",
    },
    {
        "threshold": "Myoclonic-atonic seizure frequency requiring drop-attack protocol",
        "value": "ANY frequency of drop attacks",
        "action": "Helmet mandatory from first drop attack; VPA first-line; rescue buccal midazolam prescribed",
    },
    {
        "threshold": "VPA TDM target",
        "value": "50-100 μg/mL (total); 70-100 μg/mL for drop-attack SYNGAP1",
        "action": "Below 50 = sub-therapeutic; above 100 = toxicity risk; free level if albumin <35 g/L",
    },
    {
        "threshold": "VPA hepatotoxicity threshold",
        "value": "LFT >3× upper limit of normal (ULN)",
        "action": "Withhold VPA immediately; specialist hepatology/neurology review; consider VPA cessation",
    },
    {
        "threshold": "2 AED failures = KD referral threshold",
        "value": "≥2 adequate AED trials with <50% drop reduction",
        "action": "Ketogenic diet evaluation mandatory; dietitian referral; KD centre referral",
    },
    {
        "threshold": "EC-sensitivity (ECIPA) threshold for eyelid myoclonia diagnosis",
        "value": "Generalised spike-wave within 3 seconds of eye closure on EEG",
        "action": "Confirm eyelid myoclonia phenotype; add ETH to VPA; photosensitivity precautions",
    },
    {
        "threshold": "PPR photosensitivity threshold",
        "value": "PPR Type III-IV at IPS 15-25 Hz (self-sustained GSW)",
        "action": "FL-41 glasses, screen precautions; ETH or LEV add-on for photosensitivity",
    },
    {
        "threshold": "Drop attack helmet prescription threshold",
        "value": "Any myoclonic-atonic seizure (even single event)",
        "action": "Prescribe protective headgear at first clinic visit; school must also wear helmet",
    },
    {
        "threshold": "ETH TDM target",
        "value": "40-100 μg/mL (therapeutic range); 50-80 μg/mL optimal for most",
        "action": "Below 40 = sub-therapeutic; above 100 = GI/CNS toxicity risk; measure with VPA co-medication",
    },
    {
        "threshold": "LTG withdrawal threshold in SYNGAP1",
        "value": ">50% increase in drop attack frequency from baseline within 4 weeks of LTG initiation",
        "action": "IMMEDIATELY stop LTG; revert to pre-LTG AED regimen; document LTG intolerance",
    },
]

# ── Key References (6) ────────────────────────────────────────────────────────
REFERENCES = [
    {
        "ref": "Hamdan-2009-NatGenet",
        "citation": (
            "Hamdan FF et al. De novo SYNGAP1 mutations in nonsyndromic intellectual disability. "
            "Nat Genet. 2009;41(9):1065-1067."
        ),
        "impact": "Landmark discovery of SYNGAP1 as ID/DEE gene; >600 citations.",
    },
    {
        "ref": "Mignot-2016-Brain",
        "citation": (
            "Mignot C et al. Genetic spectrum and neurodevelopmental outcomes of SYNGAP1 "
            "pathogenic variants. Brain. 2016;139(Pt 8):2380-2393."
        ),
        "impact": "Comprehensive phenotypic spectrum characterisation — 57 patients.",
    },
    {
        "ref": "Vlaskamp-2019-Neurology",
        "citation": (
            "Vlaskamp DRM et al. SYNGAP1 encephalopathy: A distinctive generalized "
            "developmental and epileptic encephalopathy. Neurology. 2019;92(2):e96-e107."
        ),
        "impact": "Definitive seizure taxonomy: myoclonic-atonic 80%, eyelid myoclonia 70%, EC-sensitivity.",
    },
    {
        "ref": "Parker-2015-EurJHumGenet",
        "citation": (
            "Parker MJ et al. De novo, heterozygous, loss-of-function mutations in SYNGAP1 "
            "cause a syndromic form of intellectual disability. Eur J Hum Genet. 2015;23(2):173-177."
        ),
        "impact": "Eyelid myoclonia and photosensitivity characterisation in SYNGAP1; clinical series.",
    },
    {
        "ref": "Mignot-2020-NatRevDisease",
        "citation": (
            "Mignot C et al. SYNGAPathy: clinical, genetic and therapeutic aspects of SYNGAP1 "
            "pathogenic variants. Nat Rev Dis Primers. 2020;6(1):63."
        ),
        "impact": "Definitive comprehensive review — biology, clinical spectrum, treatment, precision medicine pipeline.",
    },
    {
        "ref": "Bhatt-2023-EpilepsyCurrents",
        "citation": (
            "Bhatt DL et al. SYNGAP1 encephalopathy: pathophysiology and emerging therapies. "
            "Epilepsy Curr. 2023;23(3):156-164."
        ),
        "impact": "Treatment update including fenfluramine, MEK inhibitor, and ASO therapy pipeline.",
    },
]

# ── Patient Rows (N=41) ────────────────────────────────────────────────────────
def _make_patients():
    """Generate synthetic 41-patient SYNGAP1-DEE cohort."""
    rows = []
    etio_pools = [
        ("De-novo-SYNGAP1-truncating-frameshift-DEE-severe", 16),
        ("De-novo-SYNGAP1-missense-LOF-moderate", 12),
        ("De-novo-SYNGAP1-splice-site-DEE", 6),
        ("De-novo-SYNGAP1-CNV-deletion-6p21-DEE", 4),
        ("Clinical-SYNGAP1-negative-phenocopy", 3),
    ]
    sexes = ["F"] * 21 + ["M"] * 20
    random.shuffle(sexes)
    onset_ages = [round(random.uniform(1.5, 5.5), 1) for _ in range(41)]
    drop_freq = [random.randint(0, 30) for _ in range(41)]
    pid = 1
    for etio_label, n in etio_pools:
        for _ in range(n):
            sex = sexes[pid - 1]
            onset = onset_ages[pid - 1]
            drops = drop_freq[pid - 1]
            rows.append({
                "patient_id": f"SG{pid:03d}",
                "age_onset_years": onset,
                "sex": sex,
                "etiology_class": etio_label,
                "drops_per_day_baseline": drops,
                "eyelid_myoclonia": random.choice(["Yes", "Yes", "No"]),
                "photosensitive": random.choice(["Yes", "Yes", "No"]),
                "asd_diagnosis": random.choice(["Yes", "Yes", "No"]),
                "current_tx": random.choice([
                    "VPA+ETH", "VPA+ETH+CLB", "VPA+ETH+KD", "VPA+CLB",
                    "VPA+ETH+LEV", "VPA alone", "KD alone",
                ]),
                "drops_50pct_reduction": random.choice(["Yes", "Yes", "No"]),
            })
            pid += 1
    return rows


PATIENTS = _make_patients()


# ═══════════════════════ API return functions ═══════════════════════════════

def get_overview():
    total = len(PATIENTS)
    eyelid_count = sum(1 for p in PATIENTS if p["eyelid_myoclonia"] == "Yes")
    photo_count = sum(1 for p in PATIENTS if p["photosensitive"] == "Yes")
    asd_count = sum(1 for p in PATIENTS if p["asd_diagnosis"] == "Yes")
    avg_drops = round(sum(p["drops_per_day_baseline"] for p in PATIENTS) / total, 1)
    return {
        "dashboard": "SYNGAP1 Encephalopathy (SYNGAPathy / SYNGAP1-DEE / MRD5)",
        "gene": "SYNGAP1",
        "locus": "6p21.32",
        "protein": "SynGAP1 — Ras/Rap GTPase-activating protein (dendritic spine PSD)",
        "condition": "SYNGAPathy — Developmental and Epileptic Encephalopathy (DEE) / MRD5",
        "omim": "OMIM 612621 (SYNGAP1-DEE); OMIM 603384 (MRD5)",
        "cohort_n": total,
        "generated": datetime.utcnow().isoformat() + "Z",
        "kpis": [
            {"label": "Total Patients", "value": total},
            {"label": "Eyelid Myoclonia", "value": eyelid_count},
            {"label": "Photosensitive", "value": photo_count},
            {"label": "ASD Diagnosed", "value": asd_count},
            {"label": "Avg Drops/Day", "value": avg_drops},
            {"label": "Etiology Classes", "value": len(ETIOLOGY_CATALOG)},
            {"label": "Seizure Types", "value": len(SEIZURE_TYPES)},
            {"label": "Treatments", "value": len(TREATMENTS)},
        ],
        "top_clinical_alerts": [
            "DROP ATTACK HELMET mandatory from first visit — head/face trauma is primary acute morbidity",
            "AVOID CBZ / OXC / PHT — sodium channel blockers exacerbate myoclonic-atonic seizures",
            "EEG must include EYE-CLOSURE test and PHOTIC STIMULATION (IPS) at every recording",
            "LTG CAUTION — monitor drop attack frequency closely; withdraw immediately if worsening",
            "VPA + ETH combination first-line for eyelid myoclonia + absence + myoclonic-atonic",
            "ASD affects ~50% — ADOS-2/ADI-R at 30-36M is mandatory; ASD management = epilepsy management",
        ],
        "key_concept": (
            "SYNGAP1-DEE (SYNGAPathy) is one of the most prevalent single-gene causes of "
            "intellectual disability and epilepsy — caused by de novo haploinsufficiency of "
            "the Ras-ERK brake protein SynGAP1 in dendritic spines. The diagnostic EEG "
            "triad — eye-closure sensitivity + photosensitivity + myoclonic-atonic — is "
            "pathognomonic. Treatment cornerstone: VPA + ETH + protective helmet. KD for "
            "drug-resistant drop attacks. MEK inhibitor trials in development."
        ),
        "etiology_summary": [
            {"label": e["category"], "n": e["n"], "pct": e["pct"]}
            for e in ETIOLOGY_CATALOG
        ],
        "seizure_prevalence": [
            {"type": s["type"].split(" (")[0][:50], "pct": s["prevalence_pct"]}
            for s in SEIZURE_TYPES
        ],
        "trigger_prevalence": [
            {"trigger": t["trigger"].split(" (")[0][:50], "pct": t["rate_pct"]}
            for t in TRIGGERS
        ],
    }


def get_breakdown():
    return {
        "dashboard": "SYNGAP1 Encephalopathy (SYNGAPathy / SYNGAP1-DEE / MRD5)",
        "generated": datetime.utcnow().isoformat() + "Z",
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "patients": PATIENTS,
    }


def get_definitions():
    return {
        "dashboard": "SYNGAP1 Encephalopathy (SYNGAPathy / SYNGAP1-DEE / MRD5)",
        "generated": datetime.utcnow().isoformat() + "Z",
        "concepts": CONCEPTS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:1200])
    print("\n=== DEFINITIONS (thresholds) ===")
    print(json.dumps(get_definitions()["thresholds"], indent=2)[:800])
