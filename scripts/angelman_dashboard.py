"""
Angelman Syndrome (AS) Epilepsy Dashboard
==========================================
41-patient cohort · UBE3A haploinsufficiency · Maternal imprinting defect
Angelman Syndrome: UBE3A loss-of-function (maternal allele only, paternal imprinting)
→ severe epileptic encephalopathy with characteristic EEG: high-amplitude notched delta
2-3 Hz + anterior triphasic delta + alpha-frequency bursts 5-10 Hz (pathognomonic).
KEY BIOMARKER: Methylation-specific PCR 15q11-q13 (detects ~80% of cases).
VPA: relative caution (hepatotoxicity + sedation + metabolic risk, NOT absolute CI unlike GLUT1-DS).
CBZ/OXC: RELATIVE CONTRAINDICATION — worsens myoclonic/absence components ~20-30%.
PHT, VGB: ABSOLUTE CONTRAINDICATIONS — seizure aggravation.
Gene therapy trials (ASO/antisense oligonucleotides: GTX-102, ION582) in progress.
"""

import random
from datetime import datetime

SEED = 8888
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "Maternal deletion 15q11.2-q13.3 (large chromosomal deletion)",
        "n": 23, "pct": 56,
        "category": "Maternal-Deletion-15q11",
        "mechanism": (
            "The most common cause of Angelman Syndrome (56% of all cases): a de novo large "
            "deletion of the maternally-inherited chromosome 15q11.2-q13.3. This region contains "
            "the UBE3A gene (ubiquitin protein ligase E3A), which is subject to tissue-specific "
            "genomic imprinting in neurons — only the maternal allele is expressed in the brain "
            "because the paternal allele is silenced by the UBE3A-ATS (antisense transcript). "
            "Deletion sizes vary: the most common 'Type I' deletion spans ~6 Mb (BP1–BP3 "
            "breakpoints) while 'Type II' deletions span ~5 Mb (BP2–BP3). Type I deletions "
            "encompass additional genes (NIPA1, NIPA2, CYFIP1, TUBGCP5) and produce a more severe "
            "neurodevelopmental phenotype including greater seizure burden, deeper intellectual "
            "disability, and more pronounced ataxia. The deleted region also includes GABA-A "
            "receptor subunit genes (GABRB3, GABRA5, GABRG3) — haploinsufficiency at these loci "
            "directly contributes to the epileptic encephalopathy via reduced GABAergic inhibition, "
            "independent of UBE3A loss. Detection: chromosomal microarray (CMA) is the primary "
            "diagnostic tool for deletions ≥50 kb; standard FISH probes detect classic deletions. "
            "Recurrence risk: de novo deletions carry <1% recurrence risk unless a parental "
            "chromosomal rearrangement is identified."
        ),
        "eeg_correlate": (
            "Classic AS EEG pattern: high-amplitude (>200 µV) notched delta waves at 2–3 Hz, "
            "anterior-predominant; runs of rhythmic high-amplitude theta at 4–6 Hz over "
            "posterior regions; intermixed alpha-frequency bursts at 5–10 Hz (clinically "
            "characteristic and rarely seen in other epilepsy syndromes); generalised 3–4 Hz "
            "spike-wave and polyspike-wave during myoclonic seizures; photoparoxysmal response "
            "present in ~50%. Triphasic delta morphology: large positive (sharp wave) followed "
            "by a negative slow wave with a notched positive deflection — the 'notched delta' "
            "pattern is highly characteristic of Angelman Syndrome. Background EEG: grossly "
            "abnormal with dominant high-amplitude irregular delta slowing; no normal posterior "
            "dominant rhythm. EEG does NOT normalise with age (unlike some benign epilepsies); "
            "ictal recordings show generalised spike-wave or polyspike-wave discharges."
        ),
        "mri_finding": (
            "Brain MRI: often reported as normal or mildly abnormal, which should prompt clinical "
            "suspicion for AS in a child with characteristic developmental and neurological profile. "
            "When abnormal: mild generalised cerebral atrophy (particularly frontoparietal); "
            "periventricular white matter signal changes (T2 hyperintensity) — represents "
            "hypomyelination or delayed myelination. Mild T2 signal abnormality in posterior "
            "periventricular regions in Type I deletion cases with GABRB3 haploinsufficiency. "
            "No specific structural malformation — absence of focal cortical dysplasia or "
            "heterotopia differentiates AS from structural epilepsies. Cerebellar atrophy rare "
            "but reported in some severe cases. MRI findings are neither diagnostic nor required "
            "for AS diagnosis — clinical and genetic/methylation criteria are definitive."
        ),
        "clinical_note": (
            "Deletion cases are the most severe phenotypically: virtually all have epilepsy "
            "(onset 12–24 months), profound intellectual disability, absent speech, severe "
            "ataxia, and maximum seizure burden. The 'happy puppet' phenotype is most fully "
            "expressed: characteristic happy affect with frequent smiling/laughing, hand-flapping "
            "stereotypies, jerky ataxic gait, hypermotoric behaviour, and sleep disturbance "
            "(>80%). Microcephaly develops postnatally (acquired, not congenital). "
            "Clinical diagnosis criteria (Angelman Syndrome Foundation, 2006): (1) severe "
            "intellectual disability; (2) absent or minimal speech; (3) movement/balance disorder "
            "(ataxia or tremulous limb movements); (4) happy demeanour with frequent laughing; "
            "plus at least one of: EEG abnormality, seizures, microcephaly, hypopigmentation, "
            "hypermotoric behaviour, fascination with water. "
            "Methylation-specific PCR + CMA as first-line genetic test."
        ),
    },
    {
        "etiology": "Paternal UPD 15 (uniparental disomy — two copies of paternal chromosome 15)",
        "n": 6, "pct": 15,
        "category": "Paternal-UPD15",
        "mechanism": (
            "Uniparental disomy (UPD) of chromosome 15 — the child inherits both copies of "
            "chromosome 15 from the father (patUPD15), with no maternally-derived chromosome 15 "
            "contribution. Since the maternal UBE3A allele is the only brain-expressed copy "
            "(paternal UBE3A is silenced by UBE3A-ATS), a child with patUPD15 has TWO "
            "paternally-imprinted (silenced) UBE3A alleles — effectively zero UBE3A protein "
            "in neurons, despite having diploid UBE3A copy number. "
            "Mechanism of UPD: most commonly arises from trisomy rescue (trisomy 15 conceptus "
            "corrects to disomy by loss of one chromosome 15 — if the lost chromosome is the "
            "maternal one, patUPD15 results). Alternatively: gamete complementation (nullisomic "
            "egg + disomic sperm). Advanced maternal age increases trisomy rescue UPD risk. "
            "Isodisomy vs. heterodisomy: isodisomy (duplication of one paternal chromosome 15) "
            "carries risk of autosomal recessive disorders if the father is a carrier for a "
            "gene on chromosome 15 — additional recessive disorder screening warranted. "
            "Detection: methylation-specific PCR (positive for AS pattern) + microsatellite "
            "markers (biparental inheritance excluded) + FISH (no deletion). "
            "Recurrence risk: <1% (usually non-familial)."
        ),
        "eeg_correlate": (
            "UPD15 cases tend to have milder EEG abnormalities compared to deletion cases: "
            "the characteristic high-amplitude delta and theta patterns are present but often "
            "less pronounced; alpha-frequency bursts may be less prominent. Photoparoxysmal "
            "response present in ~30-40% (lower than deletion cases). Generalised spike-wave "
            "on EEG correlates with myoclonic and absence seizures. Background: abnormal delta "
            "slowing but may show periods of better-organised background between seizures. "
            "EEG findings still clearly abnormal and compatible with AS, but seizure burden "
            "and EEG epileptogenicity somewhat lower than Type I deletion cases."
        ),
        "mri_finding": (
            "UPD cases: MRI typically normal to mildly abnormal — less white matter signal "
            "change than deletion cases, consistent with milder phenotype. GABRB3 is biallelically "
            "present (no deletion), so the GABAergic haploinsufficiency contribution to structural "
            "abnormality is absent. Acquired microcephaly less severe than in deletion cases."
        ),
        "clinical_note": (
            "UPD15 produces a milder AS phenotype overall: seizures occur in ~70–80% (vs. "
            "virtually 100% in deletion cases); intellectual disability remains profound but "
            "may be in the moderate range; some speech (2–5 words) occasionally present; "
            "ataxia less severe; hypermotoric behaviour less extreme. Happy affect preserved. "
            "Important: UPD15 cases DO NOT have hypopigmentation (which requires HERC2/OCA2 "
            "deletion in the AS deletion region — absent in UPD). "
            "Autism spectrum features more prominent in UPD15 vs. deletion AS. "
            "The milder phenotype in UPD may reflect residual imprinting escape of maternal "
            "UBE3A — some cells may have partial biallelic UBE3A expression. "
            "Family history: typically de novo; recurrence risk <1%."
        ),
    },
    {
        "etiology": "Imprinting centre defect (methylation abnormality, no deletion)",
        "n": 4, "pct": 10,
        "category": "Imprinting-Centre-Defect",
        "mechanism": (
            "A pathogenic variant or microdeletion in the imprinting centre (IC) of the "
            "15q11-q13 PWS/AS imprinting domain results in failure of maternal imprint "
            "establishment or maintenance. The AS-IC (a ~35 kb region at the 5' end of SNRPN) "
            "controls the switch from paternal to maternal epigenetic marks during oogenesis. "
            "Two IC subtypes: (1) IC deletion — microdeletion in the AS-IC element causes "
            "failure of maternal methylation at SNRPN/SNURF promoter; the maternal chromosome 15 "
            "behaves epigenetically as paternal → UBE3A silenced. These IC deletions ARE heritable "
            "— if an IC deletion is present on the maternal allele, it will cause AS in all "
            "maternally-transmitted offspring. CRITICAL for recurrence counselling. "
            "(2) Epimutation (IC methylation defect without deletion) — spontaneous failure "
            "of imprint erasure in the maternal germline; very low recurrence risk. "
            "Detection: methylation-specific PCR shows AS (paternal-only) methylation pattern; "
            "CMA normal (no deletion); IC deletion analysis by specific PCR or high-resolution CMA "
            "(<50 kb) required. "
            "UBE3A protein: absent in neurons due to inappropriate silencing of maternal allele."
        ),
        "eeg_correlate": (
            "IC defect EEG: high-amplitude delta and theta patterns present, similar to deletion "
            "cases but variable severity. If IC deletion (heritable form): EEG severity approaches "
            "deletion phenotype. If IC epimutation (non-inherited): may be milder. "
            "Photoparoxysmal response present in ~40-50%. Alpha-frequency bursts characteristic. "
            "Seizure EEG: generalised spike-wave with myoclonic and absence ictus most common."
        ),
        "mri_finding": (
            "IC defect: MRI often normal; white matter abnormalities less severe than deletion "
            "cases (GABRB3/GABRA5 intact — no GABAergic haploinsufficiency from deletion). "
            "Acquired microcephaly may develop postnatally but less severe than deletion."
        ),
        "clinical_note": (
            "IC defect phenotype is intermediate to mild: seizures in 70–90% depending on "
            "IC deletion vs. epimutation. CRITICAL clinical point: if IC deletion confirmed, "
            "RECURRENCE RISK IS 50% for each subsequent pregnancy through the maternal line — "
            "immediate genetic counselling and family cascade testing mandatory. "
            "Maternal grandmother may carry the IC deletion and transmit to multiple affected grandchildren. "
            "Pre-implantation genetic testing (PGT-M) available for known IC deletion families. "
            "IC epimutation: recurrence risk <1%, similar to de novo. "
            "Standard methylation PCR detects IC defect — IC deletion requires specialist "
            "molecular analysis beyond routine methylation testing."
        ),
    },
    {
        "etiology": "UBE3A point mutation / intragenic pathogenic variant",
        "n": 6, "pct": 15,
        "category": "UBE3A-Point-Mutation",
        "mechanism": (
            "Pathogenic sequence variants within the UBE3A gene itself — predominantly de novo "
            "but ~10% inherited from a carrier mother (autosomal dominant, maternally-expressed "
            "imprinting). UBE3A encodes E6-AP ubiquitin-protein ligase, a HECT-domain E3 ubiquitin "
            "ligase. Loss of UBE3A disrupts ubiquitin-proteasome pathway in neurons — particularly "
            "affecting synaptic plasticity, AMPA receptor trafficking, and long-term potentiation "
            "(LTP) at excitatory synapses. Variant types causing AS: nonsense (PTC) — 35%; "
            "frameshift (insertion/deletion) — 35%; missense (loss of E3 ligase function) — 15%; "
            "splice-site — 10%; large intragenic deletion — 5%. The maternal UBE3A allele is "
            "exclusively expressed in neurons (paternal silenced by UBE3A-ATS); a pathogenic "
            "variant on the maternal allele therefore causes complete UBE3A protein absence in "
            "neurons. When the carrier is a mother with a heterozygous UBE3A variant: the "
            "mother is phenotypically NORMAL (the paternal UBE3A allele provides protein in "
            "non-neuronal tissues where both alleles are expressed), but every child who inherits "
            "the variant-bearing maternal chromosome 15 will have AS (50% risk). "
            "Detection: methylation-specific PCR is NORMAL (no methylation defect) — therefore "
            "this class is MISSED by methylation testing alone. UBE3A sequencing is REQUIRED "
            "for diagnosis. Accounts for ~10–15% of AS not diagnosed by methylation PCR."
        ),
        "eeg_correlate": (
            "Point mutation AS EEG: characteristic high-amplitude delta + theta patterns present, "
            "but generally milder than large deletion cases — the absence of GABRB3/GABRA5 "
            "haploinsufficiency reduces the GABAergic component of epileptogenesis. "
            "Alpha-frequency bursts still present and diagnostically useful. "
            "Photoparoxysmal response in ~30-40%. Seizure frequency lower in this class "
            "compared to deletion cases. Background slowing still present but may have "
            "periods of near-normal background organisation."
        ),
        "mri_finding": (
            "UBE3A point mutation: MRI most commonly normal or shows only minor nonspecific "
            "changes. No white matter abnormalities from GABRB3 deletion (absent in this class). "
            "The mildest MRI findings of all AS molecular classes. Acquired microcephaly "
            "present but least severe."
        ),
        "clinical_note": (
            "Point mutation cases have the MILDEST AS phenotype: seizures in ~50–70%; "
            "some degree of purposeful communication possible; less severe ataxia; "
            "occasional single words or short phrases possible. Happy affect and hypermotoric "
            "features present. CRITICAL diagnostic pitfall: methylation-specific PCR is NORMAL "
            "in UBE3A point mutation cases — AS diagnosis will be MISSED if UBE3A sequencing "
            "not performed in a child with clinical AS features and normal methylation. "
            "Recurrence risk: if maternally inherited variant (carrier mother) — 50% per "
            "pregnancy through maternal line. De novo: <1% recurrence. "
            "Maternal carrier mothers: phenotypically normal — must test maternal UBE3A. "
            "PGT-M available for known familial variants. This class is the primary target "
            "for antisense oligonucleotide (ASO) gene therapy — paternal UBE3A can potentially "
            "be unsilenced by blocking UBE3A-ATS, bypassing the maternal loss."
        ),
    },
    {
        "etiology": "Clinical diagnosis (meets criteria, negative molecular workup)",
        "n": 2, "pct": 4,
        "category": "Clinical-Negative-Workup",
        "mechanism": (
            "A small minority (~4%) of individuals who fully meet published Angelman Syndrome "
            "clinical diagnostic criteria (Angelman Syndrome Foundation, Williams 2006) but "
            "have negative results on comprehensive molecular testing: methylation-specific PCR "
            "(normal), CMA/FISH (no deletion), UPD studies (biparental), and UBE3A sequencing "
            "(no pathogenic variant). Possible explanations: (1) Deep intronic UBE3A variants "
            "affecting splicing — not captured by coding-region exome or panel sequencing; "
            "RNA-seq or long-read sequencing required. (2) Somatic mosaicism for 15q11 deletion "
            "or UBE3A variant at low allele fraction — missed by standard short-read sequencing; "
            "requires deep WGS or tissue-specific testing. (3) Regulatory region variants in "
            "UBE3A promoter or IC — detected only by WGS or functional studies. (4) Pathogenic "
            "variants in genes functionally downstream of UBE3A (e.g., SHANK3, other synaptic "
            "scaffolding proteins) — AS-like phenocopy. (5) Unknown molecular mechanism — "
            "very rare undiscovered causes of AS-phenotype. "
            "Treatment: manage identically to confirmed AS — AED selection and monitoring "
            "protocols unchanged. Refer to specialist centre for research-level WGS/RNA-seq."
        ),
        "eeg_correlate": (
            "Clinically-negative workup: EEG indistinguishable from molecularly-confirmed AS "
            "in many cases — the characteristic high-amplitude delta, theta, and alpha-burst "
            "pattern provides indirect diagnostic support. Some cases may show less classic "
            "EEG features if the underlying mechanism is a phenocopy rather than true UBE3A "
            "loss. EEG alone cannot differentiate AS from AS-like syndromes."
        ),
        "mri_finding": (
            "Variable — may be normal or show minor nonspecific changes. Absence of GABRB3 "
            "deletion (standard molecular workup negative) means no GABAergic haploinsufficiency "
            "structural signature expected. If MRI shows cortical malformation, reconsider "
            "alternative diagnosis (structural epilepsy, Rett syndrome)."
        ),
        "clinical_note": (
            "Management: treat as AS clinically. Avoid PHT and VGB (seizure aggravation risk "
            "as in confirmed AS). CBZ/OXC with caution. CLN + LEV + VPA (with POLG exclusion) "
            "are appropriate first-line options. Refer to specialist AS centre for research "
            "protocol WGS/RNA-seq. Register in FAST (Foundation for Angelman Syndrome "
            "Therapeutics) patient registry — clinical diagnosis cases are eligible for "
            "natural history studies and potentially for ASO trials (if UBE3A loss confirmed "
            "by functional assay). Review genetic counselling — if no molecular diagnosis, "
            "recurrence risk cannot be accurately quantified; empiric counselling offered."
        ),
    },
]

# ── Seizure Types (4 types) ────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Myoclonic seizures",
        "prevalence_pct": 80,
        "eeg_correlate": (
            "Generalised polyspike-wave bursts; high-amplitude 3–4 Hz spike-wave with "
            "prominent polyspike component preceding slow wave; myoclonic discharges may "
            "be continuous as subtle subclinical myoclonus on EEG between more obvious "
            "clinical jerks. Ictal: synchronous polyspike burst followed by slow wave — "
            "each burst corresponds to a myoclonic jerk. Photoparoxysmal response (PPR) "
            "with myoclonic EMG correlate present in ~50% of AS patients — "
            "photic-induced myoclonus is a clinical diagnostic clue. Background: severely "
            "abnormal high-amplitude delta between bursts. The high amplitude of the "
            "epileptiform discharges (often >500 µV) is characteristic of AS."
        ),
        "clinical_tip": (
            "AS myoclonus is frequently CONTINUOUS and subtle — persistent fine jerking "
            "of upper limbs, head tremor, or hand-flapping that families may not recognise "
            "as seizure activity. Distinguish from the baseline movement disorder (ataxic "
            "tremor) of AS: myoclonic jerks are paroxysmal, time-locked with EEG bursts, "
            "and may worsen with fever or intercurrent illness. "
            "Stimulus-sensitivity is common — sudden noise or touch triggers myoclonic bursts. "
            "CLN (clonazepam) is most effective for myoclonic control in AS; LEV is alternative. "
            "VPA adds broad-spectrum coverage but requires POLG exclusion and LFT monitoring. "
            "CBZ/OXC WORSEN myoclonus in 20–30% of AS — RELATIVE CONTRAINDICATION. "
            "PHT ABSOLUTE CONTRAINDICATION — always worsens myoclonus. "
            "Myoclonus burden directly correlates with disruption of daily activities and "
            "sleep quality in AS."
        ),
    },
    {
        "type": "Atypical absence seizures",
        "prevalence_pct": 70,
        "eeg_correlate": (
            "Generalised 2–3 Hz high-amplitude spike-wave (slower than classic 3 Hz CAE); "
            "ictal onset often gradual without abrupt EEG change — differentiates from "
            "idiopathic CAE which has abrupt onset. The ictal discharge may be admixed "
            "with the background high-amplitude delta, making clinical seizure onset "
            "difficult to identify from the interictal state without careful comparison. "
            "Duration: typically 5–30 seconds; post-ictal minimal. "
            "Often misidentified as inattentiveness or 'blanks' given the profound baseline "
            "developmental delay — video-EEG is essential for correct classification."
        ),
        "clinical_tip": (
            "AS atypical absence is frequently MISSED clinically — the profound baseline "
            "intellectual disability makes brief unresponsiveness difficult to distinguish "
            "from normal behaviour. Clinical clues: eye deviation upward, brief limpness, "
            "interruption of activity, lip-smacking, drooling during episode. "
            "Caregiver-reported 'staring spells', 'glazed look' or 'switching off' warrant "
            "video-EEG evaluation. "
            "Treatment: CLN is most effective; LEV add-on; VPA broad-spectrum. "
            "IMPORTANT: do NOT use ethosuximide as monotherapy in AS — AS absence "
            "seizures differ from idiopathic CAE and ETX alone is insufficient; combined "
            "CLN+LEV or CLN+VPA preferred. "
            "Sleep-related worsening: absences may cluster during drowsiness. "
            "Fever/intercurrent illness: absence burden increases — anticipatory CLB use "
            "(buccal diazepam) for fever management."
        ),
    },
    {
        "type": "Focal seizures with secondary generalisation",
        "prevalence_pct": 60,
        "eeg_correlate": (
            "Focal ictal discharge — most commonly posterior temporal, occipital, or "
            "centrotemporal origin — followed by rapid secondary generalisation. "
            "EEG onset: focal rhythmic alpha or theta discharge (often high-amplitude "
            "given the AS background) → generalised tonic-clonic pattern within 2–5 seconds. "
            "Post-ictal: suppression or slow recovery against the chronically abnormal background. "
            "Todd's paresis (focal post-ictal weakness) possible — duration 10–30 minutes; "
            "resolves spontaneously. The focal onset may be masked by the high-amplitude "
            "diffuse background — 64-electrode or HD-EEG increases focal onset detection."
        ),
        "clinical_tip": (
            "Focal seizures in AS: may present as versive (head/eye turning), focal motor "
            "(unilateral limb jerking), or focal sensory onset with rapid secondary generalisation "
            "to bilateral tonic-clonic. "
            "Todd's paresis post-ictally: important to recognise — do NOT initiate stroke "
            "workup for transient post-ictal focal weakness in a known AS patient. Reassure "
            "family: resolves within 30–60 minutes. "
            "Posterior cortex origin common in AS — likely related to posterior cortical "
            "hypersynchrony from UBE3A deficiency in occipital and temporal association cortex. "
            "Anti-seizure management: LEV is particularly effective for focal component; "
            "CLB as adjunct. Avoid CBZ/OXC (worsen myoclonic component). "
            "MRI: despite focal EEG onset, structural MRI typically normal — "
            "confirm this before considering any epilepsy surgery evaluation."
        ),
    },
    {
        "type": "Tonic-clonic seizures (GTCS)",
        "prevalence_pct": 55,
        "eeg_correlate": (
            "Classic GTCS pattern: generalised paroxysmal fast activity (GPFA) "
            "during tonic phase (recruiting rhythm 10–20 Hz) → rhythmic generalised "
            "2–3 Hz spike-wave during clonic phase → post-ictal diffuse suppression "
            "then slow recovery. In AS, the pre-ictal EEG already shows generalised "
            "high-amplitude delta background, so the GTCS EEG onset is less clearly "
            "demarcated than in a previously normal brain. "
            "Duration: 1–3 minutes (GTCS); status epilepticus (>5 min) occurs in AS — "
            "particularly during fever, illness, or VPA toxicity."
        ),
        "clinical_tip": (
            "GTCS in AS are frequently triggered by FEVER — hyperthermia lowers seizure "
            "threshold substantially in AS; aggressive fever management (paracetamol, tepid "
            "sponging) is a seizure-prevention intervention. "
            "VPA toxicity can paradoxically WORSEN or precipitate GTCS in AS — monitor "
            "VPA levels if seizure increase occurs on VPA therapy. "
            "Emergency protocol: buccal midazolam 0.3 mg/kg (first-line) or rectal/nasal "
            "diazepam; call ambulance if GTCS >5 minutes (AS status risk is higher than "
            "idiopathic epilepsy population). "
            "Nocturnal GTCS carry SUDEP risk — nocturnal monitoring/seizure alarm recommended "
            "for AS patients with frequent GTCS. "
            "Acute fever management: do NOT withhold AED for vomiting — give PR or buccal "
            "formulation; NG tube during any hospital admission for continued AED delivery."
        ),
    },
]

# ── Triggers (8 triggers) ─────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / hyperthermia",
        "seizure_rate_pct": 90,
        "note": (
            "The most powerful and consistent seizure trigger in AS. Fever >38°C reliably "
            "provokes breakthrough seizures in 90% of AS patients. Mechanism: hyperthermia "
            "increases neuronal metabolic demand, reduces GABAergic inhibition threshold, "
            "and disrupts synaptic plasticity — all magnified by UBE3A-deficient synaptic "
            "function. Management: paracetamol at fever onset (target T<37.5°C); buccal "
            "midazolam rescue medication prescribed for all AS patients; maintain AED dosing "
            "via NG tube or suppository if oral route compromised. Fever illness plan: "
            "prescribe to all caregivers; trigger hospitalisation threshold lower in AS."
        ),
    },
    {
        "trigger": "Sleep deprivation",
        "seizure_rate_pct": 80,
        "note": (
            "Sleep disturbance affects >80% of AS patients (reduced sleep duration, "
            "irregular circadian rhythm, frequent nocturnal awakenings) — directly causing "
            "sleep deprivation-triggered seizures. The sleep disorder in AS is intrinsic "
            "(UBE3A is expressed in suprachiasmatic nucleus controlling circadian rhythm) "
            "rather than secondary to seizures alone. Melatonin 2–10 mg nocte is first-line "
            "for sleep consolidation — reduces seizure frequency by reducing sleep deprivation "
            "trigger. Consistent sleep schedule mandatory; blackout curtains, white noise. "
            "Nocturnal monitoring: seizure alarm or pulse oximetry for AS patients with "
            "frequent nocturnal seizures."
        ),
    },
    {
        "trigger": "Missed AED dose",
        "seizure_rate_pct": 70,
        "note": (
            "AED non-compliance/missed doses in a profoundly intellectually disabled patient "
            "who cannot self-medicate: entirely caregiver-dependent. Missed dose triggers "
            "breakthrough seizures within 12–24 hours for CLN (short half-life) and CLB. "
            "LEV missed dose: seizure risk within 6–12 hours. VPA missed dose: lower "
            "immediate risk due to longer half-life but cumulative effect. "
            "Strategies: blister pack dispensers, caregiver smartphone alarms, twice-daily "
            "formulations (LEV-XR, CLB once daily), liquid formulations for NG/PEG administration. "
            "Hospital admission: ensure AED continued IV/NG — do NOT allow AED gap during "
            "acute illness admissions."
        ),
    },
    {
        "trigger": "Excitement / emotional stimulation",
        "seizure_rate_pct": 65,
        "note": (
            "Characteristic and highly specific AS trigger: emotional excitement, laughter "
            "(the very affect that defines the AS phenotype), social stimulation, and "
            "overstimulation from busy environments. This paradoxical trigger (happiness "
            "causing seizures) reflects AS-specific thalamocortical and limbic circuit "
            "dysregulation. The hypermotoric AS child is at intrinsic risk from their "
            "own characteristic behaviours. Management: environmental pacing — not "
            "eliminating positive interactions but managing overstimulation. "
            "Awareness: families and caregivers must recognise pre-ictal signs during "
            "excitement (eye deviation, hand-flapping increase, subtle myoclonus) and "
            "de-escalate activity. This trigger cannot be pharmacologically eliminated "
            "without unacceptable over-sedation."
        ),
    },
    {
        "trigger": "VPA toxicity / elevated VPA levels",
        "seizure_rate_pct": 55,
        "note": (
            "Paradoxical seizure worsening with supratherapeutic VPA levels is documented "
            "in AS — hyperammonaemia from VPA can cause encephalopathy mimicking seizure "
            "worsening, and true seizure aggravation at toxic levels. "
            "Monitor: VPA TDM target 50–100 mg/L; ammonia if drowsiness/vomiting/seizure "
            "worsening on VPA. VPA-induced hyperammonaemia (without hepatotoxicity) responds "
            "to L-carnitine supplementation. "
            "Key distinction: VPA is a RELATIVE CAUTION in AS (not absolute CI like GLUT1-DS) "
            "but requires POLG exclusion before initiation and careful TDM monitoring. "
            "Dose reduction or switch to LEV monotherapy if VPA levels consistently elevated "
            "or ammonia raised."
        ),
    },
    {
        "trigger": "Photic stimulation",
        "seizure_rate_pct": 50,
        "note": (
            "Photoparoxysmal response (PPR) present in ~50% of AS patients on EEG; clinical "
            "photosensitivity (seizures triggered by flicker, TV, video games, sunlight "
            "through trees) in ~30–40%. Mechanism: impaired cortical inhibition via UBE3A "
            "loss affects occipital cortex PPR threshold. Management: tinted lenses (fl-41 "
            "rose-tinted filter blocks the 500–530 nm wavelengths most epileptogenic); "
            "matte screen covers; avoid strobe-containing entertainment. "
            "EEG photoparoxysmal testing should be performed at baseline and annually "
            "in all AS patients."
        ),
    },
    {
        "trigger": "Illness / infection (non-febrile systemic illness)",
        "seizure_rate_pct": 45,
        "note": (
            "Systemic illness even without significant fever (GI illness, respiratory "
            "illness with vomiting) worsens AS seizure control: vomiting → missed AED → "
            "breakthrough seizure cascade. Additional mechanism: inflammatory cytokines "
            "(IL-1β, TNF-α) lower seizure threshold in UBE3A-deficient circuits. "
            "Management plan for illness: prescribe buccal/rectal emergency medication; "
            "threshold for NG tube AED administration during vomiting episodes; "
            "lower hospitalisation threshold vs. typically developing child with same illness."
        ),
    },
    {
        "trigger": "Puberty / hormonal change",
        "seizure_rate_pct": 25,
        "note": (
            "Puberty-related hormonal changes (oestrogen pro-convulsant, progesterone "
            "anti-convulsant) may worsen or improve seizure control in AS depending on "
            "hormonal trajectory. Catamenial seizure pattern: perimenstrual seizure clustering "
            "in ~25% of adolescent/adult females with AS. Management: seizure diary with "
            "menstrual cycle tracking; consider cyclical CLB (clobazam 10 mg nocte for "
            "5 days perimenstrually as rescue) for catamenial pattern. "
            "Menarche timing: AS females often experience precocious or delayed puberty — "
            "monitor hormonal status. Contraception counselling in adult AS females "
            "with seizures: AVOID enzyme-inducing AEDs (PHT, CBZ) — irrelevant as these "
            "are contraindicated in AS for seizure reasons as well."
        ),
    },
]

# ── Treatments (8 treatments) ──────────────────────────────────────────────────
TREATMENTS = [
    {
        "name": "Clonazepam (CLN)",
        "evidence_level": "Level A (ILAE 2022 — myoclonus first-line; Kyllerman 2021)",
        "dose": (
            "0.05–0.2 mg/kg/day in 2–3 divided doses (maximum 0.5 mg/kg/day); "
            "start low 0.01–0.02 mg/kg/day and titrate weekly to minimise sedation; "
            "liquid formulation available for NG/PEG administration"
        ),
        "moa": (
            "GABA-A receptor positive allosteric modulator — enhances chloride channel "
            "opening frequency, potentiating GABAergic inhibitory neurotransmission at "
            "inhibitory synapses throughout the cortex and subcortical structures. "
            "Particularly effective for myoclonic seizures and atypical absence because "
            "these seizure types depend on thalamocortical hypersynchrony — benzodiazepines "
            "selectively disrupt this synchronisation. In AS, where GABRB3/GABRA5 "
            "haploinsufficiency (deletion cases) already reduces GABAergic inhibition, "
            "CLN provides pharmacological compensation for the lost GABA-A subunits. "
            "Does NOT interact with UBE3A pathway — no disease-modifying effect. "
            "Combination with LEV: additive — different mechanisms (CLN GABAergic, "
            "LEV SV2A-mediated); well-tolerated together."
        ),
        "efficacy": (
            "Myoclonic seizures: 60–75% achieve ≥50% reduction; first-line monotherapy "
            "or combination. Atypical absence: 50–65% response. Tolerance: develops in "
            "~30% over 3–6 months — dose escalation or drug holiday may be needed."
        ),
        "safety": (
            "Sedation (dose-dependent, particularly problematic at AS baseline of reduced "
            "arousal), hypersalivation (important — AS patients have baseline drooling; "
            "CLN worsens), hypotonia, tolerance development, withdrawal seizures if "
            "abruptly discontinued. Respiratory depression at high doses — caution "
            "in AS patients with sleep-disordered breathing."
        ),
        "monitoring": (
            "RASS (Richmond Agitation-Sedation Scale) sedation score at each visit; "
            "hypersalivation management plan (glycopyrrolate 0.04 mg/kg q8h if severe); "
            "seizure diary — watch for tolerance (seizure re-emergence at stable dose); "
            "consider structured drug holiday (2 weeks) at 6–12 months if tolerance suspected; "
            "respiratory assessment annually"
        ),
    },
    {
        "name": "Levetiracetam (LEV)",
        "evidence_level": "Level A (ILAE 2022 — broad-spectrum AS first-line; Boyd 2015 Cochrane)",
        "dose": (
            "20–60 mg/kg/day divided twice daily; extended-release (XR) once daily in "
            "adolescents/adults; oral solution 100 mg/mL for NG/PEG; start 10 mg/kg/day "
            "and titrate by 10 mg/kg/day every 2 weeks"
        ),
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) modulation — binds the SV2A protein "
            "on synaptic vesicles and reduces abnormal neuronal hypersynchrony by impairing "
            "vesicle recycling at overactive synapses. Broad-spectrum mechanism effective "
            "against generalised myoclonic, absence, and focal seizure types — all relevant "
            "in AS. No pharmacokinetic interaction with CLN, VPA, or CLB — "
            "polypharmacy well-tolerated. Does not inhibit or potentiate UBE3A pathway. "
            "Safe in hepatic disease (renally cleared — important given VPA hepatic concern "
            "in AS)."
        ),
        "efficacy": (
            "AS seizures: 50–70% achieve ≥50% reduction across seizure types; most effective "
            "for myoclonic and focal components. Cochrane review (Boyd 2015): LEV effective "
            "for AS myoclonus with acceptable side effect profile. Combination LEV+CLN "
            "frequently provides additive benefit."
        ),
        "safety": (
            "Behavioural/neuropsychiatric adverse effects: irritability 15–25%, aggression "
            "10–15%, emotional lability 10%, sleep disruption 5–10%. CRITICAL AS-SPECIFIC "
            "CONCERN: AS patients have a baseline of hypermotoric, excitable behaviour — "
            "LEV-induced irritability and aggression are DIFFICULT TO DISTINGUISH from "
            "baseline AS behaviour and may be attributed to the syndrome rather than the drug. "
            "Systematic behavioural monitoring with validated tools is essential. Renal "
            "dosing required if GFR <80 mL/min/1.73m²."
        ),
        "monitoring": (
            "PHQ-9 proxy rating (caregiver report) quarterly; CBCL (Child Behaviour Checklist) "
            "behavioural monitoring q3–6M; seizure diary; renal function annually; "
            "if aggression escalates on LEV — consider dose reduction trial before attributing "
            "to AS baseline; switch to brivaracetam (better-tolerated SV2A ligand) if LEV "
            "behavioural side effects limit dosing"
        ),
    },
    {
        "name": "Valproate (VPA)",
        "evidence_level": "Level B (broad-spectrum AS evidence; ILAE — use with significant precautions)",
        "dose": (
            "20–60 mg/kg/day divided twice daily (or extended-release once daily); "
            "start 10–15 mg/kg/day, titrate by 5–10 mg/kg/day every 1–2 weeks; "
            "TDM target 50–100 mg/L (trough level)"
        ),
        "moa": (
            "Multiple mechanisms: voltage-gated sodium channel blockade (reduces repetitive "
            "neuronal firing); enhanced GABAergic transmission (increases GABA synthesis, "
            "reduces GABA degradation by inhibiting GABA transaminase); histone deacetylase "
            "(HDAC) inhibition (potentially modulates epigenetic regulation including "
            "imprinting-related chromatin — theoretical relevance in AS). Broad-spectrum "
            "anti-seizure properties effective against generalised myoclonic, absence, and "
            "tonic-clonic seizures — all relevant AS seizure types. "
            "CRITICAL DISTINCTION FROM GLUT1-DS: VPA is NOT an absolute contraindication "
            "in Angelman Syndrome — it does NOT inhibit UBE3A or disrupt the AS pathological "
            "mechanism. However, VPA carries its own significant risks in AS "
            "(hepatotoxicity, metabolic risks, reproductive teratogenicity) requiring careful "
            "patient selection and monitoring."
        ),
        "efficacy": (
            "AS seizures: 50–65% achieve ≥50% seizure reduction across seizure types; "
            "particularly effective for myoclonic+absence combination. Often used as "
            "second-line monotherapy or combination with CLN or CLB when LEV-alone "
            "insufficient."
        ),
        "safety": (
            "Hepatotoxicity (most serious — POLG exclusion MANDATORY before initiation); "
            "weight gain; thrombocytopenia; hyperammonaemia (monitor ammonia if drowsiness "
            "or seizure worsening); sedation; tremor (may worsen AS ataxia/tremor); "
            "polycystic ovary syndrome risk (long-term female use); teratogenicity (neural "
            "tube defects, cognitive effects — folic acid 5 mg daily mandatory; REPRODUCTIVE "
            "AGE WARNING for adolescent/adult AS females on VPA). VPA REMS program in USA — "
            "prescriber registration required."
        ),
        "monitoring": (
            "VPA TDM: 50–100 mg/L (trough) q3–6M; LFTs at baseline + q6M (discontinue if "
            "ALT/AST >3× upper limit of normal); ammonia if clinically indicated; platelets "
            "q6M; POLG genetic screen BEFORE initiation (VPA is ABSOLUTE CI in POLG mutations "
            "— fatal Alpers hepatic failure); weight and BMI q3M; folic acid 5 mg daily; "
            "annual pregnancy counselling for reproductive-age females; consider switch to "
            "alternative if VPA levels consistently sub-therapeutic or ammonia elevated"
        ),
    },
    {
        "name": "Clobazam (CLB)",
        "evidence_level": "Level B (add-on or monotherapy; ILAE 2022; Thibert 2009)",
        "dose": (
            "0.1–0.5 mg/kg/day in 1–2 divided doses; once-daily dosing possible for "
            "adherence; start 0.05 mg/kg/day and titrate; maximum 1 mg/kg/day; "
            "cyclical perimenstrual dosing (10 mg nocte × 5 days) for catamenial pattern"
        ),
        "moa": (
            "1,5-benzodiazepine — GABA-A receptor positive allosteric modulator at the "
            "1,5-position (vs. 1,4-position for clonazepam). Binds α2/α3 GABA-A subunits "
            "predominantly (vs. CLN which has less subunit selectivity). The 1,5-structure "
            "confers less sedation and less muscle relaxation vs. classical 1,4-BZDs, "
            "while maintaining anticonvulsant efficacy — an important advantage in AS where "
            "sedation compounds the baseline developmental profile. Active metabolite "
            "N-desmethylclobazam also anticonvulsant (half-life 36–46 h — longer than CLB "
            "itself at 18 h; provides seizure coverage stability). "
            "CYP2C19 polymorphism affects CLB metabolism — poor metabolisers accumulate "
            "N-desmethylclobazam → increased efficacy and toxicity."
        ),
        "efficacy": (
            "AS seizures: 45–65% responder rate (≥50% reduction); particularly useful "
            "for myoclonic + absence combination as add-on to LEV. Catamenial AS: "
            "cyclical CLB (perimenstrual) highly effective in ~60% of catamenial patients. "
            "Tolerance less prominent than CLN — CLB preferred for long-term adjunct therapy."
        ),
        "safety": (
            "Sedation (less than CLN); tolerance (slower to develop than CLN); ataxia "
            "worsening (may compound AS-specific ataxia); withdrawal risk with abrupt "
            "discontinuation; salivation less than CLN. CYP2C19 interaction: "
            "clopidogrel, omeprazole reduce CLB efficacy; fluconazole increases CLB levels."
        ),
        "monitoring": (
            "Seizure diary at each visit; UMRS (Unified Myoclonus Rating Scale) for "
            "myoclonus burden; tolerance assessment at 6M (if seizure re-emergence at "
            "stable dose, consider drug holiday or dose adjustment); CYP2C19 genotyping "
            "if poor response or toxicity; menstrual cycle diary for catamenial pattern tracking"
        ),
    },
    {
        "name": "Topiramate (TPM)",
        "evidence_level": "Level B (add-on refractory AS; expert consensus)",
        "dose": (
            "3–9 mg/kg/day divided twice daily; start 0.5–1 mg/kg/day and titrate "
            "by 1 mg/kg/day every 1–2 weeks; slow titration critical to reduce "
            "cognitive side effects"
        ),
        "moa": (
            "Multiple mechanisms: voltage-gated sodium channel blockade; GABA-A positive "
            "allosteric modulation; AMPA/kainate glutamate receptor antagonism; carbonic "
            "anhydrase inhibition. Broad-spectrum efficacy against myoclonic, absence, "
            "and tonic-clonic seizures. The carbonic anhydrase inhibition produces metabolic "
            "acidosis (clinically relevant on KD co-treatment). "
            "AMPA antagonism is theoretically relevant in AS — UBE3A regulates AMPA "
            "receptor trafficking and surface expression; excess AMPA-mediated excitatory "
            "transmission in UBE3A-deficient synapses may be partially attenuated by TPM."
        ),
        "efficacy": (
            "AS seizures: 40–55% responder rate; useful as third-line add-on when CLN+LEV "
            "or CLN+VPA insufficient. Most effective for the generalised seizure types. "
            "Cognitive blunting is a significant limitation in AS where cognitive baseline "
            "is already severely impaired."
        ),
        "safety": (
            "CRITICAL AS-SPECIFIC CONCERN: cognitive blunting (word-finding difficulty, "
            "slowed processing) is TPM's primary limitation — in AS with severe baseline "
            "intellectual disability and absent speech, additional cognitive blunting "
            "is difficult to detect but reduces adaptive function and augmentative "
            "communication capacity. Hypohidrosis (heat intolerance — important in "
            "excitable hypermotoric AS patients); renal stones (metabolic acidosis + "
            "hypocitraturia); metabolic acidosis (particularly on concurrent KD); weight loss."
        ),
        "monitoring": (
            "Bicarbonate annually (metabolic acidosis monitoring, especially on concurrent KD); "
            "sweating assessment (hypohidrosis — parent/caregiver report of reduced sweating "
            "in warm weather); weight and BMI; communication/adaptive function assessment "
            "annually (Vineland, VABS); renal USS if haematuria or flank pain; "
            "potassium citrate supplementation if recurrent acidosis"
        ),
    },
    {
        "name": "Ketogenic Diet (KD)",
        "evidence_level": "Level B (refractory AS; Peters 2010 Orphanet; Thibert 2009)",
        "dose": (
            "4:1 or 3:1 fat:carbohydrate+protein ratio; or Modified Atkins Diet (MAD) "
            "carbohydrate <20 g/day; calorie-appropriate for age/weight; supervised by "
            "specialist paediatric/adult dietitian; BHB target 2–4 mmol/L"
        ),
        "moa": (
            "Ketone bodies (beta-hydroxybutyrate, acetoacetate) provide alternative "
            "neuronal fuel substrate and exert multiple anti-seizure effects: direct "
            "membrane stabilisation via K-ATP channel activation; enhancement of GABA "
            "synthesis via increased acetyl-CoA → glutamine → glutamate → GABA pathway; "
            "reduced glucose-driven excitatory neurotransmission; anti-inflammatory effects "
            "via NLRP3 inflammasome inhibition. In AS specifically: KD may partially "
            "compensate for UBE3A-deficient synaptic plasticity by improving overall "
            "synaptic energy supply and GABA availability — particularly relevant given "
            "GABRB3/GABRA5 haploinsufficiency in deletion cases. "
            "KD is NOT disease-modifying in AS (unlike GLUT1-DS) — it is a seizure-control "
            "intervention without direct effect on UBE3A pathway."
        ),
        "efficacy": (
            "AS seizures: 40–60% achieve ≥50% seizure reduction (responder rate); "
            "particularly effective for myoclonic and absence components in refractory cases. "
            "Peters 2010 (Orphanet): KD effective in AS as adjunctive therapy for refractory "
            "epilepsy; improvement in behaviour and alertness noted beyond seizure control. "
            "Consider after failure of ≥2 AEDs."
        ),
        "safety": (
            "Growth restriction (height, weight monitoring monthly); hyperlipidaemia; "
            "renal stones (5–8%); selenium/carnitine/zinc deficiency; bone density reduction "
            "(annual DEXA); constipation (common in AS on high-fat diet + reduced activity); "
            "GI intolerance (nausea, vomiting — challenge in AS with existing feeding difficulties). "
            "PEG/G-tube: KD formula via gastrostomy is feasible and improves adherence "
            "in AS patients with severe feeding difficulties."
        ),
        "monitoring": (
            "BHB twice weekly (target 2–4 mmol/L); urine ketones daily; growth charts monthly; "
            "lipid panel q6M; selenium, zinc, carnitine, 25-OH vitamin D annually; "
            "DEXA annually; renal USS annually; metabolic panel (electrolytes, bicarbonate, "
            "glucose, renal function) q3M; dietitian review q3M; caregiver KD knowledge "
            "assessment at each visit"
        ),
    },
    {
        "name": "Melatonin",
        "evidence_level": "Level C (sleep consolidation; indirect seizure reduction via sleep improvement)",
        "dose": (
            "2–10 mg at bedtime (start 2 mg, titrate by 2 mg every 2 weeks to effect); "
            "immediate-release preferred for sleep-onset insomnia; prolonged-release "
            "(Circadin) for sleep maintenance insomnia; maximum 10 mg nocte"
        ),
        "moa": (
            "Melatonin is an endogenous hormone produced by the pineal gland, regulating "
            "circadian rhythm via MT1/MT2 melatonin receptors in the suprachiasmatic nucleus "
            "(SCN). UBE3A is expressed in SCN neurons — UBE3A deficiency disrupts circadian "
            "pacemaker function, causing the intrinsic sleep disorder characteristic of AS. "
            "Exogenous melatonin restores circadian signal strength, advancing sleep phase "
            "and consolidating nocturnal sleep architecture. Mechanism of seizure reduction: "
            "INDIRECT — by reducing sleep deprivation (the second most powerful AS seizure "
            "trigger), melatonin reduces seizure frequency without direct anticonvulsant "
            "action. Melatonin also has neuroprotective and anti-inflammatory properties "
            "(MT1 receptor-mediated antioxidant signalling). "
            "Not an anti-seizure medication per se — document as 'sleep consolidation "
            "therapy' to avoid confusion in AED count."
        ),
        "efficacy": (
            "Sleep: 60–80% of AS families report improved sleep duration and reduced "
            "nocturnal awakenings on melatonin. Seizure frequency: indirect reduction "
            "attributable to sleep improvement; no head-to-head controlled AS trials "
            "with seizure endpoints. Quality of life: caregiver sleep improvement "
            "(caregiver burnout significantly reduced when AS child sleeps through the night)."
        ),
        "safety": (
            "Excellent safety profile — no hepatotoxicity, no significant drug interactions, "
            "no tolerance development. Mild: morning drowsiness (use earlier timing "
            "if daytime drowsiness), enuresis (nocturnal bladder relaxation). "
            "Long-term safety in children: melatonin used for >5 years in AS without "
            "significant adverse effects in published series."
        ),
        "monitoring": (
            "Sleep diary (actigraphy if available) — record sleep onset, nocturnal waking, "
            "total sleep duration; seizure diary concurrent; caregiver sleep quality score "
            "(Caregiver Sleep Quality Index); review timing and dose at q3M visits; "
            "no blood test monitoring required"
        ),
    },
    {
        "name": "ASO / Gene Therapy (antisense oligonucleotides — investigational)",
        "evidence_level": "Phase II Clinical Trials (GTX-102, ION582 — NOT approved; investigational only)",
        "dose": (
            "GTX-102: intrathecal injection; dose escalation per protocol (Phase II: "
            "3.3–33 mg IT); multiple doses over 12–24 weeks; trial-specific protocols. "
            "ION582: intrathecal; dose per IND protocol. NOT for clinical prescribing "
            "outside approved trials."
        ),
        "moa": (
            "Antisense oligonucleotides (ASOs) targeting UBE3A-ATS (UBE3A antisense "
            "transcript) — the long non-coding RNA that silences the paternal UBE3A allele "
            "in neurons via transcriptional interference. By blocking UBE3A-ATS with "
            "complementary oligonucleotide strands, the ASO unsilences the paternal UBE3A "
            "allele, allowing production of paternal UBE3A protein in neurons despite the "
            "maternal allele being lost or mutated. This is the first genuinely DISEASE-MODIFYING "
            "approach for AS — addressing the root cause (UBE3A protein deficiency) rather "
            "than symptom management. "
            "Paternal UBE3A unsilencing: demonstrated in multiple AS mouse models "
            "(Meng 2015, Science; Wolter 2020, Nature); paternal UBE3A re-expression "
            "rescues synaptic plasticity, LTP, and motor/cognitive phenotypes in mice. "
            "Human translation: FAST (Foundation for Angelman Syndrome Therapeutics) "
            "funded early human trials. "
            "GTX-102 (GeneTx/UCB): Phase II data showed biomarker evidence of UBE3A "
            "protein increase in CSF biomarkers; observed adverse event (lower extremity "
            "weakness in some patients at high doses) led to protocol hold and dose revision. "
            "ION582 (Ionis/Biogen): Phase II ongoing as of 2024. "
            "CRITICAL: only applicable to UBE3A-loss mechanisms (deletion, UPD, IC defect, "
            "UBE3A point mutation) — the paternal UBE3A must be structurally intact for "
            "unsilencing to produce protein."
        ),
        "efficacy": (
            "Preclinical: robust; clinical (Phase II, interim): biomarker evidence of "
            "UBE3A protein increase in CSF; neurodevelopmental endpoint data pending "
            "full trial completion. Epilepsy endpoint: seizure frequency secondary endpoint "
            "in trials — preliminary signals of reduction in some patients. "
            "Not yet sufficient evidence for efficacy rating — trials ongoing."
        ),
        "safety": (
            "GTX-102: lower extremity weakness/paraparesis observed in some patients "
            "at higher doses (AE led to protocol hold and dose revision); mechanism "
            "under investigation (possible off-target lumbar motor neuron effect at "
            "IT injection site). ION582: safety data accumulating in Phase II. "
            "Intrathecal injection risks: meningitis (rare, sterile technique), CSF leak, "
            "headache, back pain. Long-term safety (>24 months): unknown."
        ),
        "monitoring": (
            "Trial-specific protocol monitoring; CSF UBE3A protein biomarker (if assay "
            "available); neurodevelopmental battery (Bayley, Vineland, VABS) at each "
            "trial visit; seizure diary; EMG/nerve conduction if lower limb weakness "
            "(GTX-102 safety monitoring); enrol in FAST/UCB/Ionis open-label extension "
            "if available; notify neurology team of all AEs per trial requirements"
        ),
    },
]

# ── Absolute Contraindications (4) ────────────────────────────────────────────
ABSOLUTE_CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT) — ALL formulations (fosphenytoin included)",
        "scope": "ALL Angelman Syndrome patients — ABSOLUTE CONTRAINDICATION",
        "mechanism": (
            "Sodium channel blocker with disproportionate worsening of myoclonic and "
            "atypical absence seizures in AS — documented seizure aggravation in multiple "
            "case series. PHT preferentially suppresses cortical inhibitory interneuron "
            "firing (which depends on Na+ channels) in addition to excitatory neurons, "
            "paradoxically reducing inhibitory tone and worsening myoclonic hypersynchrony. "
            "In AS, where GABAergic inhibition is already impaired (deletion cases: "
            "GABRB3/GABRA5 haploinsufficiency; all cases: UBE3A-dependent synaptic "
            "modulation deficiency), PHT-mediated inhibitory neuron suppression is "
            "particularly hazardous."
        ),
        "consequence": (
            "Documented seizure aggravation: worsening myoclonus, increased absence "
            "frequency, precipitation of myoclonic status epilepticus. No anti-seizure "
            "benefit demonstrated in AS — risk-benefit entirely negative."
        ),
        "action": (
            "NEVER prescribe PHT in AS. If PHT inadvertently given during acute seizure "
            "management (e.g., generic 'seizure protocol' IV PHT): monitor closely for "
            "seizure worsening; switch to LEV or benzodiazepine IV; document PHT allergy/CI "
            "prominently in all medical records and allergy system."
        ),
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) — RELATIVE CONTRAINDICATION",
        "scope": (
            "AS patients — RELATIVE CONTRAINDICATION (not absolute); "
            "use ONLY under specialist supervision with documented benefit-risk assessment"
        ),
        "mechanism": (
            "Sodium channel blockers with documented worsening of myoclonic and absence "
            "seizure components in approximately 20–30% of AS patients. Mechanism: "
            "similar to PHT — Na-channel blockade in inhibitory interneurons preferentially "
            "reduces GABAergic inhibition, worsening the hypersynchronous myoclonic circuits. "
            "CBZ more sedating than OXC. Additionally, CBZ is a potent enzyme inducer "
            "(CYP3A4, CYP2C9) — reduces levels of concurrently used CLN, CLB, VPA, and "
            "potentially reduces efficacy of all concurrent AEDs."
        ),
        "consequence": (
            "Worsening myoclonus in ~20–30% of AS patients; increased absence frequency; "
            "potential seizure aggravation. Not as universally harmful as PHT — some "
            "AS patients with focal seizure predominance may benefit, but myoclonic "
            "worsening risk must be explicitly discussed and monitored."
        ),
        "action": (
            "Avoid as first-line in AS. If considered for focal seizure predominance "
            "in a specialist setting: document explicit benefit-risk discussion; "
            "monitor closely for myoclonus worsening at each dose step; "
            "withdraw immediately if seizure aggravation detected; prefer LEV for focal "
            "seizures in AS (no myoclonus worsening risk)."
        ),
    },
    {
        "drug": "Vigabatrin (VGB)",
        "scope": "ALL Angelman Syndrome patients — ABSOLUTE CONTRAINDICATION",
        "mechanism": (
            "GABA transaminase irreversible inhibitor — increases synaptic GABA. "
            "Paradoxically worsens myoclonus in AS — the mechanism involves excess tonic "
            "GABA (via extrasynaptic GABA-A receptor activation, particularly α5-containing "
            "receptors) depressing phasic GABAergic inhibitory post-synaptic currents, "
            "paradoxically increasing cortical excitability in myoclonic circuits. "
            "VGB also causes permanent bilateral visual field constriction (>25% of "
            "patients with prolonged use) — an unacceptable risk in AS patients who "
            "cannot reliably report visual symptoms, and who have photosensitivity."
        ),
        "consequence": (
            "Seizure aggravation (myoclonus worsening); irreversible bilateral visual "
            "field loss (nasal-to-temporal concentric constriction) — undetectable until "
            "severe in non-communicating AS patients; no demonstrated benefit in AS."
        ),
        "action": (
            "NEVER prescribe VGB in AS. If inadvertently prescribed: stop immediately; "
            "arrange ophthalmology visual field assessment; document CI prominently. "
            "Existing VGB: taper (do not abrupt stop) and replace with CLN or LEV."
        ),
    },
    {
        "drug": "Hospital nil-by-mouth (NPO) without AED continuation plan",
        "scope": "ALL AS patients requiring any surgical, procedural, or acute illness fasting",
        "mechanism": (
            "Fasting in AS without AED continuity: oral AED doses cannot be swallowed → "
            "missed doses → sub-therapeutic levels within 6–24 hours (CLN, LEV) → "
            "breakthrough seizure cascade. Additionally, fasting increases seizure risk "
            "in AS (metabolic stress, sleep deprivation) compounding the missed-AED effect. "
            "Fever during illness adds the most powerful trigger simultaneously — the "
            "triple combination of fever + missed AED + fasting during hospital admission "
            "represents maximum seizure risk."
        ),
        "consequence": (
            "Breakthrough seizures, prolonged seizure clusters, generalised tonic-clonic "
            "status epilepticus — a preventable hospital-acquired complication from "
            "failure to plan AED delivery during NPO."
        ),
        "action": (
            "Prescribe IV/NG/PEG equivalents for all AEDs before any fasting period: "
            "IV LEV (dose-equivalent), NG CLN liquid, buccal CLB. Anaesthesia team "
            "briefing mandatory for any general anaesthetic. Seizure rescue medication "
            "(buccal midazolam) must be prescribed and available at the bedside. "
            "AED continuation plan: document explicitly in pre-operative/admission clerking."
        ),
    },
]

# ── AED Monitoring (5 items) ───────────────────────────────────────────────────
AED_MONITORING = [
    {
        "item": "EEG (annual baseline and after AED change)",
        "target": "Characteristic AS pattern: high-amplitude delta 2-3 Hz, triphasic delta, alpha-bursts 5-10 Hz; photoparoxysmal response",
        "rationale": (
            "EEG is diagnostic and monitoring tool in AS: confirms AS-characteristic "
            "pattern; quantifies epileptiform burden; detects subclinical status; "
            "assesses treatment response (EEG change correlates with clinical seizure "
            "reduction in AS). After each AED change: EEG at 3 months to assess impact. "
            "Annual EEG: tracks disease course and seizure burden. "
            "Video-EEG for seizure classification when clinical phenotype unclear."
        ),
    },
    {
        "item": "VPA therapeutic drug monitoring (TDM)",
        "target": "Trough 50-100 mg/L; LFTs + ammonia + platelets every 6 months on VPA",
        "rationale": (
            "VPA hepatotoxicity is the most serious adverse effect — fatal Alpers hepatic "
            "failure in POLG patients (POLG exclusion MANDATORY before VPA initiation). "
            "Hyperammonaemia without LFT elevation: VPA inhibits hepatic urea cycle "
            "enzymes (carbamyl phosphate synthase) → serum ammonia rise → encephalopathy "
            "mimicking seizure worsening. Thrombocytopenia (platelet count <100 × 10⁹/L "
            "→ dose reduction). TDM ensures therapeutic range maintenance — sub-therapeutic "
            "VPA is ineffective; supratherapeutic is toxic."
        ),
    },
    {
        "item": "Developmental/behavioural assessment (annual)",
        "target": "Bayley Scales (cognitive, language, motor), Vineland Adaptive Behaviour Scales, ABC-Community (Aberrant Behaviour Checklist)",
        "rationale": (
            "AS is a neurodevelopmental disorder where seizure control is a means to "
            "developmental optimisation. Annual formal developmental assessment tracks: "
            "(1) cognitive trajectory — stable or declining? AED cognitive burden assessment; "
            "(2) adaptive function — communication, daily living, socialisation; "
            "(3) behavioural profile — ABC-Community tracks irritability, hyperactivity, "
            "lethargy, stereotypy, inappropriate speech; essential for LEV/TPM behavioural "
            "monitoring. Communication assessment: augmentative and alternative communication "
            "(AAC) device suitability annual review."
        ),
    },
    {
        "item": "Sleep study (polysomnography / actigraphy)",
        "target": "Total sleep time >6h/night; sleep efficiency >80%; REM and NREM architecture assessment",
        "rationale": (
            "Sleep disturbance in >80% of AS patients is directly linked to seizure burden — "
            "sleep deprivation is the second most powerful AS seizure trigger. "
            "Polysomnography: assess for obstructive sleep apnoea (OSA — common in AS with "
            "hypotonia + large tongue), assess EEG during sleep for nocturnal seizures, "
            "quantify REM/NREM disruption. Melatonin titration guided by actigraphy data. "
            "Annual review of sleep quality informs melatonin dosing adjustments."
        ),
    },
    {
        "item": "Bone density (DEXA) — KD and long-term AED users",
        "target": "Z-score > -2.0; calcium + vitamin D supplementation as standard",
        "rationale": (
            "AS patients on long-term AEDs (VPA causes bone loss via CYP450-independent "
            "mechanism affecting osteoblast function; CLN reduces weight-bearing activity) "
            "and KD (high fat diet reduces calcium absorption) are at combined osteopenia "
            "risk. Reduced physical activity in AS (ataxia limits weight-bearing exercise) "
            "further reduces bone density accrual. Annual DEXA from puberty; calcium "
            "1000 mg/day + vitamin D 800 IU/day supplementation standard in all AS patients."
        ),
    },
]

# ── 6-Window Lifecycle ─────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal-Infant (0–12 months)",
        "phase": "Hypotonia, feeding difficulties, pre-diagnostic",
        "description": (
            "AS is rarely diagnosed at birth — the neonatal period presents with "
            "non-specific features: hypotonia (universal), feeding difficulties "
            "(requiring NG tube support in severe cases), subtle myoclonus (may be "
            "dismissed as 'jitteriness'). The characteristic happy affect and "
            "EEG pattern emerge from 6–12 months. Seizure onset: peak 12–24 months "
            "but may begin as early as 6 months. EEG may show posterior high-amplitude "
            "delta as early as 6 months — characteristic AS pattern before overt seizures. "
            "Genetic testing trigger: any infant with unexplained hypotonia + seizure "
            "onset in the first year + developmental delay should have methylation PCR "
            "15q11 + CMA as first-line investigation. "
            "Early diagnosis enables anticipatory guidance for seizure onset and appropriate "
            "AED planning before first seizure."
        ),
        "key_actions": (
            "Hypotonia workup includes 15q11 methylation PCR; baseline EEG; feeding "
            "support (NG/gastrostomy if required); family genetic counselling; "
            "developmental therapy referral (OT, PT, SLT from 0 months); "
            "seizure action plan provided before first seizure"
        ),
    },
    {
        "window": "Early Childhood (1–5 years)",
        "phase": "Seizure onset, AED initiation, AS phenotype emergence",
        "description": (
            "The peak period for AS diagnosis and AED initiation. Classic AS phenotype "
            "fully emerges: happy affect, frequent laughing/smiling, hand-flapping "
            "stereotypies, ataxic gait (toe-walking, wide-based), hypermotoric behaviour, "
            "absent speech, hypermotoric fascination with water. Seizures onset typically "
            "12–24 months: myoclonic, atypical absence, occasional GTCS. "
            "EEG: classic AS pattern (high-amplitude delta + theta + alpha-bursts) "
            "established by age 2. "
            "Acquired microcephaly apparent by 24 months — OFC crosses centiles downward. "
            "AED initiation: CLN ± LEV as first-line; avoid PHT, VGB, CBZ. "
            "Communication: AAC device assessment — PECS, GoTalk, TouchChat as communication "
            "substitutes for absent speech. Seizure rescue prescription: buccal midazolam "
            "for all caregivers + school + respite."
        ),
        "key_actions": (
            "Molecular confirmation (if not neonatal): methylation PCR + CMA + UBE3A "
            "sequencing; AED initiation (CLN+LEV); buccal midazolam rescue; "
            "EEG at diagnosis and 3M post-AED; AAC device assessment; IEP start; "
            "multidisciplinary team (neurology + genetics + developmental paediatrics + "
            "dietitian + SLT + PT + OT); AS Foundation family registration"
        ),
    },
    {
        "window": "School Age (5–12 years)",
        "phase": "Epileptic encephalopathy, AED optimisation, educational support",
        "description": (
            "Established epileptic encephalopathy: ongoing seizures (myoclonic + absence "
            "most common), AED polypharmacy often required (CLN + LEV ± VPA ± CLB). "
            "KD consideration for refractory epilepsy (failure of ≥2 AEDs). "
            "Educational: special education school placement universal — one-to-one support, "
            "AAC integration, sensory curriculum. Behavioural management: hypermotoric "
            "behaviour, sleep disruption, and seizures create complex care needs. "
            "Annual neuropsychological assessment (Bayley, Vineland). "
            "Seizure control correlates with learning opportunities: each seizure-free day "
            "is a developmental opportunity. Water safety: AS children have a potentially "
            "life-threatening fascination with water — pool/bath supervision mandatory. "
            "Sleep: melatonin established as routine therapy for AS sleep disorder by "
            "school age."
        ),
        "key_actions": (
            "AED optimisation review annually; KD evaluation if 2+ AEDs failed; "
            "annual EEG + developmental assessment; sleep study + melatonin titration; "
            "water safety protocol; AAC device upgrade; scoliosis screening (AS + "
            "hypotonia + anticonvulsants = scoliosis risk); bone density monitoring; "
            "DEXA if on VPA + KD + long-term CLN"
        ),
    },
    {
        "window": "Adolescence (12–18 years)",
        "phase": "Puberty-related seizure changes, transition planning",
        "description": (
            "Seizure frequency may change at puberty — oestrogen increases seizure risk; "
            "progesterone may improve. Catamenial seizure pattern emerges in ~25% of "
            "adolescent females: perimenstrual GTCS or myoclonic clusters. Cyclical CLB "
            "prescribed for catamenial pattern. VPA: document contraceptive counselling "
            "and REMS requirements for adolescent female AS patients (teratogenicity). "
            "Transition to adult services: AS adult neurologist services are limited — "
            "plan transition >12 months before 18th birthday. Identify adult neurologist "
            "experienced in developmental epilepsy. Driving: AS patients with severe "
            "intellectual disability do not drive — driving restriction counselling not "
            "applicable clinically but may be relevant to carers using the patient's "
            "transport. Orthopedic: scoliosis progression common in adolescence — "
            "consider surgical correction if Cobb angle >40°."
        ),
        "key_actions": (
            "Catamenial pattern assessment + cyclical CLB if indicated; VPA REMS "
            "counselling for females; transition planning to adult services; scoliosis "
            "assessment (XR spine annually if clinical concern); DEXA bone density; "
            "adult educational/residential placement planning; ASO trial eligibility "
            "assessment (molecular class, paternal UBE3A intact?)"
        ),
    },
    {
        "window": "Young Adult (18–30 years)",
        "phase": "Adult care coordination, supported living, reproductive counselling",
        "description": (
            "AS is a lifelong condition — adult AS patients require ongoing neurological "
            "care and AED management. Seizure burden may plateau or mildly reduce in "
            "young adulthood vs. childhood peak. Adult residential services: supported "
            "living or residential care with specialist AS training. Communication: "
            "high-tech AAC devices maintain function into adulthood. "
            "Reproductive counselling: AS females with intact reproductive function "
            "may become pregnant — VPA teratogenicity requires proactive AED switching "
            "before conception (switch to LEV or CLB monotherapy if possible). "
            "Genetic counselling: recurrence risk for offspring if maternal IC deletion "
            "or UBE3A variant — 50% transmission. "
            "ASO trials: young adults with confirmed UBE3A-loss mechanism may be "
            "eligible for ongoing ASO trials — refer to specialist centre."
        ),
        "key_actions": (
            "Adult neurology transition completed; AED optimisation for adult dosing; "
            "VPA reproductive counselling and REMS compliance; genetic counselling for "
            "reproduction; residential placement; power of attorney / guardianship; "
            "ASO trial eligibility assessment; annual metabolic monitoring; "
            "healthcare proxy documentation"
        ),
    },
    {
        "window": "Adult/Older (30+ years)",
        "phase": "Chronic management, comorbidities, long-term AED effects",
        "description": (
            "Limited long-term outcome data in older AS adults (syndrome described 1965, "
            "first molecular confirmation 1987 — cohort only now ageing into their 40s-50s). "
            "Emerging data: seizure frequency may stabilise or slightly reduce in older "
            "adulthood; happy affect persists lifelong; communication via AAC maintained; "
            "ambulation: some patients remain ambulant; severe cases may be wheelchair-bound "
            "by midlife (scoliosis, contractures). Long-term AED effects: bone density "
            "(osteoporosis risk on decades of VPA + CLN); cardiovascular (VPA + weight "
            "gain → metabolic syndrome); cognitive burden (decades of CLN sedation). "
            "Comorbidities: obesity (reduced mobility + VPA), scoliosis, contractures, "
            "gastroesophageal reflux, aspiration risk. "
            "Palliative care planning: for severely affected AS adults with progressive "
            "decline — advance care plans, aspiration management, pain assessment "
            "using non-verbal tools (FLACC scale)."
        ),
        "key_actions": (
            "Annual metabolic review + AED toxicity monitoring; DEXA q2Y; "
            "cardiovascular risk assessment (BMI, lipids, blood pressure); "
            "scoliosis and orthopedic review; aspiration risk assessment (SALT); "
            "medication review for polypharmacy and AED burden; FLACC pain assessment; "
            "palliative care liaison for complex cases; carer support and respite planning"
        ),
    },
]

# ── 14 Concepts ────────────────────────────────────────────────────────────────
CONCEPTS = [
    {
        "term": "UBE3A",
        "definition": (
            "UBE3A encodes E6-AP ubiquitin-protein ligase, a HECT-domain E3 ubiquitin ligase "
            "located at chromosome 15q11.2-q13.3. In the brain, UBE3A is subject to genomic "
            "imprinting — only the maternal allele is expressed in neurons because the paternal "
            "allele is silenced by a long non-coding antisense transcript (UBE3A-ATS). "
            "UBE3A protein targets substrates for ubiquitin-proteasome degradation, regulating "
            "synaptic plasticity, AMPA receptor surface expression, and long-term potentiation (LTP). "
            "Loss of maternal UBE3A causes Angelman Syndrome; gain-of-function UBE3A variants "
            "are associated with Angelman syndrome-related autism spectrum disorder. "
            "The paternal UBE3A allele (silenced in neurons) is the target of antisense oligonucleotide "
            "(ASO) gene therapy — blocking UBE3A-ATS unsilences paternal UBE3A to compensate "
            "for maternal loss."
        ),
    },
    {
        "term": "Angelman Syndrome",
        "definition": (
            "Angelman Syndrome (AS) is a rare neurodevelopmental disorder characterised by: "
            "severe intellectual disability, absent or minimal speech, movement/balance disorder "
            "(ataxic gait, tremulous limbs), epilepsy (>80% of cases), and a distinctive "
            "happy, sociable affect with frequent smiling and laughter. Originally described "
            "as 'happy puppet syndrome' by Harry Angelman in 1965 (3 children with intellectual "
            "disability, absent speech, jerky movements, and inappropriate happiness). "
            "Molecular basis: loss of UBE3A expression in neurons from the maternal allele. "
            "Prevalence: ~1 in 12,000–20,000 live births. AS is underdiagnosed — the "
            "diagnostic delay from symptom onset to molecular confirmation averages 3–5 years. "
            "Management: AED therapy for epilepsy (no curative treatment currently); "
            "ASO gene therapy in Phase II trials."
        ),
    },
    {
        "term": "Maternal Imprinting",
        "definition": (
            "Genomic imprinting: an epigenetic mechanism by which gene expression depends "
            "on the parent of origin — the same gene sequence is expressed from only one "
            "parental allele (maternal or paternal) in specific tissues. "
            "For UBE3A in neurons: MATERNAL EXPRESSION ONLY — the paternal UBE3A allele "
            "is silenced in the brain (but expressed in non-neuronal tissues). "
            "The silencing mechanism: UBE3A-ATS (antisense transcript from the paternal "
            "chromosome 15) transcribes through the paternal UBE3A locus in neurons, "
            "preventing paternal UBE3A sense-strand transcription. "
            "In Angelman Syndrome: loss of the maternal UBE3A allele (deletion, mutation) "
            "leaves neurons with ZERO UBE3A protein (paternal silenced + maternal lost). "
            "In Prader-Willi Syndrome: the reciprocal — maternal chromosome 15 region "
            "is deleted, affecting maternally-imprinted genes on the same locus (different "
            "genes: SNRPN, NECDIN, etc.) — different phenotype, same chromosomal region."
        ),
    },
    {
        "term": "Prader-Willi Syndrome (PWS) — 15q11 Reciprocal",
        "definition": (
            "Prader-Willi Syndrome (PWS) is caused by loss of PATERNALLY-expressed genes "
            "in the 15q11.2-q13.3 region — the reciprocal of Angelman Syndrome. "
            "PWS aetiology: (1) paternal 15q11-q13 deletion (~70%); (2) maternal UPD15 "
            "(both chromosomes 15 from mother, no paternal 15q11 genes expressed, ~25%); "
            "(3) IC defect (~5%). "
            "PWS phenotype: neonatal hypotonia, feeding difficulties, hyperphagia (leading "
            "to obesity), hypogonadism, short stature, intellectual disability (mild-moderate), "
            "behavioural difficulties (obsessive-compulsive features, skin-picking). "
            "PWS does NOT typically cause epilepsy (unlike AS). "
            "Genes affected in PWS: SNRPN, NDN (Necdin), MKRN3, MAGEL2 — all paternally "
            "expressed and imprinted from the maternal allele. "
            "Methylation PCR: PWS shows maternal-only methylation pattern (paternal band absent); "
            "AS shows paternal-only pattern (maternal band absent). "
            "PWS and AS are caused by the SAME chromosomal region, deleted from DIFFERENT "
            "parents — the classic paradigm of genomic imprinting in human genetic disease."
        ),
    },
    {
        "term": "PWS-AS Imprinting Region (15q11.2-q13.3)",
        "definition": (
            "The chromosome 15q11.2-q13.3 genomic region is one of the most extensively "
            "studied imprinted loci in the human genome. It spans approximately 5–6 Mb and "
            "contains multiple imprinted genes organised in a bipartite structure: "
            "PATERNAL domain (SNRPN, NDN, MKRN3, MAGEL2, snoRNAs) — silenced on maternal "
            "chromosome, active on paternal; "
            "MATERNAL domain (UBE3A, ATP10A) — silenced in neurons on paternal chromosome "
            "(by UBE3A-ATS), active from maternal allele. "
            "Critical landmarks: BP1, BP2, BP3 breakpoints define common deletion sizes; "
            "imprinting centre (IC) at SNRPN 5' end controls the epigenetic switch; "
            "GABRB3, GABRA5, GABRG3 (GABA-A receptor subunit genes) lie within the region "
            "and are biallelically expressed — haploinsufficiency from deletion causes "
            "GABAergic epileptogenesis in AS deletion cases. "
            "This region is a genomic hotspot for recombination errors, non-allelic homologous "
            "recombination (NAHR) between low-copy repeats (LCRs) at BP1-BP3, explaining "
            "the high de novo deletion frequency."
        ),
    },
    {
        "term": "Triphasic Delta EEG",
        "definition": (
            "The pathognomonic EEG pattern of Angelman Syndrome: high-amplitude (>200 µV) "
            "delta waves at 2–3 Hz with a characteristic NOTCHED or TRIPHASIC morphology — "
            "a large positive (sharp) component followed by a deep negative slow wave with "
            "a secondary positive notch. The pattern is anterior-predominant (maximal "
            "frontally and over central regions). Additional characteristic features: "
            "runs of high-amplitude rhythmic theta at 4–6 Hz over posterior regions; "
            "alpha-frequency bursts at 5–10 Hz (clinically distinctive — rarely seen in "
            "other epilepsy syndromes at this amplitude). The triphasic delta pattern "
            "is present from early childhood and PERSISTS throughout life in AS — "
            "it is not age-dependent. Importantly, the EEG in AS can be more abnormal "
            "than the clinical seizure frequency suggests — significant EEG epileptogenicity "
            "even in relatively seizure-controlled periods."
        ),
    },
    {
        "term": "Alpha Burst EEG",
        "definition": (
            "A characteristic EEG feature of Angelman Syndrome: runs of high-amplitude "
            "(>200 µV) rhythmic 5–10 Hz alpha-frequency activity, occurring in brief "
            "bursts of 1–3 seconds, predominantly over posterior or diffuse regions. "
            "These alpha-frequency bursts are NOT NORMAL posterior dominant rhythm — "
            "they occur in the context of a severely abnormal background (with delta "
            "slowing and triphasic morphology) and are far higher in amplitude than "
            "normal posterior dominant rhythm (typically 20–50 µV). "
            "Clinical significance: alpha-frequency bursts in AS are a diagnostically "
            "useful finding because this pattern is rarely seen with such amplitude and "
            "in this clinical context in other epilepsy syndromes. "
            "Recognition of this pattern by the EEG reader should prompt AS consideration "
            "even before genetic results are available."
        ),
    },
    {
        "term": "Methylation PCR (15q11-q13)",
        "definition": (
            "Methylation-specific PCR (MS-PCR) or methylation-sensitive Southern blot "
            "of the chromosome 15q11-q13 SNRPN locus: the primary first-line diagnostic "
            "test for Angelman Syndrome. The SNRPN promoter is UNMETHYLATED on the "
            "paternal allele and METHYLATED on the maternal allele. "
            "In AS (deletion, UPD, IC defect): only the paternal (unmethylated) pattern "
            "is present — the maternal (methylated) copy is absent. "
            "Normal: both maternal (methylated) and paternal (unmethylated) bands. "
            "Sensitivity: MS-PCR detects ~80% of AS cases (deletion + UPD + IC defect). "
            "MISSES: UBE3A point mutations (~15% of AS) — UBE3A sequencing required "
            "for complete diagnosis. "
            "If MS-PCR is negative but clinical suspicion high: proceed to UBE3A "
            "sequencing. Never rely on MS-PCR alone in a child with AS clinical features "
            "and normal MS-PCR result."
        ),
    },
    {
        "term": "UPD (Uniparental Disomy)",
        "definition": (
            "Uniparental disomy (UPD): inheritance of both copies of a chromosome pair "
            "from one parent only. For AS: paternal UPD15 (both copies of chromosome 15 "
            "from the father) means no maternally-derived chromosome 15 is present — "
            "therefore no maternal UBE3A allele. Since UBE3A is only expressed from "
            "the maternal allele in neurons (paternal silenced), patUPD15 results in "
            "zero neuronal UBE3A protein. "
            "Types: isodisomy (one paternal chromosome 15 is duplicated — homozygosity "
            "for all paternal markers) vs. heterodisomy (two different paternal chromosomes "
            "15 are inherited — one from each paternal copy). "
            "Isodisomy risk: if father is a carrier for an autosomal recessive disorder "
            "on chromosome 15, isodisomy → homozygosity → expression of the recessive "
            "disorder in addition to AS. "
            "Mechanism: trisomy rescue (most common); gamete complementation (rare). "
            "Methylation PCR: shows paternal-only pattern (same as deletion AS) — "
            "microsatellite marker analysis required to distinguish from deletion."
        ),
    },
    {
        "term": "Imprinting Centre Defect",
        "definition": (
            "The imprinting centre (IC) is a ~35 kb cis-acting regulatory element "
            "at the 5' end of the SNRPN locus that controls the epigenetic status "
            "(methylation and histone modification) of the entire 15q11-q13 imprinted "
            "domain. The IC contains two elements: (1) PWS-IC — controls establishment "
            "of paternal imprint in spermatogenesis; (2) AS-IC — controls erasure of "
            "paternal methylation and establishment of maternal methylation in oogenesis. "
            "IC defect in AS: pathogenic variant or microdeletion in the AS-IC element "
            "prevents establishment of maternal methylation pattern during oogenesis. "
            "The maternal chromosome 15 behaves epigenetically like a paternal chromosome — "
            "UBE3A-ATS is expressed from the maternally-derived chromosome, silencing "
            "maternal UBE3A in neurons. "
            "CLINICAL IMPORTANCE: IC deletions ARE HERITABLE — if present on the maternal "
            "allele of the mother, each pregnancy carries 50% recurrence risk for AS. "
            "Genetic counselling and cascade testing of maternal relatives is mandatory "
            "when IC deletion is identified."
        ),
    },
    {
        "term": "ASO Antisense Therapy",
        "definition": (
            "Antisense oligonucleotides (ASOs) are synthetic single-stranded nucleic acid "
            "analogues (typically 15–25 nucleotides) designed to bind complementary RNA "
            "sequences via Watson-Crick base pairing, altering RNA splicing, stability, "
            "or translation. For Angelman Syndrome: ASOs targeting UBE3A-ATS "
            "(the paternal antisense transcript that silences paternal UBE3A in neurons) "
            "are designed to block UBE3A-ATS transcription or promote its degradation "
            "via RNase H mechanism. By inhibiting UBE3A-ATS, the paternal UBE3A allele "
            "is unsilenced — paternal UBE3A protein is expressed in neurons, compensating "
            "for maternal UBE3A loss. "
            "Delivery: intrathecal injection (CNS ASO delivery, bypassing BBB). "
            "Key agents: GTX-102 (GeneTx/UCB Pharma) and ION582 (Ionis/Biogen) "
            "both in Phase II clinical trials as of 2024. "
            "This represents the most promising disease-modifying therapeutic approach "
            "for AS, targeting the root molecular cause."
        ),
    },
    {
        "term": "RASS Sedation Score",
        "definition": (
            "Richmond Agitation-Sedation Scale (RASS): a validated 10-point scale for "
            "assessing level of sedation and agitation in clinical settings. Ranges from "
            "+4 (combative) through 0 (alert and calm) to -5 (unarousable). "
            "Relevance to AS: benzodiazepines (CLN, CLB) are primary AS AEDs but carry "
            "sedation risk — in a patient with baseline intellectual disability and reduced "
            "communication, sedation is difficult to assess subjectively. "
            "RASS provides objective, standardised sedation monitoring: target RASS 0 "
            "to -1 (alert or mildly drowsy) on stable CLN/CLB. "
            "RASS -2 or below (light sedation or deeper): clinically significant sedation "
            "requiring dose reduction. "
            "Caregiver proxy RASS rating: trained caregivers can administer proxy RASS "
            "at home between clinic visits to track sedation between appointments."
        ),
    },
    {
        "term": "Photoparoxysmal Response (PPR)",
        "definition": (
            "Photoparoxysmal response (PPR): EEG generalised spike-wave or polyspike-wave "
            "discharge triggered by photic stimulation (intermittent light at 1–50 Hz "
            "during EEG). In AS: PPR present in ~50% of patients — one of the highest "
            "rates of any epilepsy syndrome. "
            "Clinical photosensitivity (seizures triggered by real-world photic stimuli): "
            "~30–40% of AS patients. Triggers: TV screens (especially older cathode ray), "
            "video games, sunlight through trees or water, disco/strobe lights. "
            "Management: FL-41 rose-tinted spectacles (block 500–530 nm epileptogenic "
            "wavelengths); matte anti-glare screen covers; sunglasses outdoors. "
            "EEG photoparoxysmal testing (photic stimulation at 1–60 Hz) should be "
            "routinely performed in all AS patients at diagnosis and annually — guides "
            "photosensitivity precautions."
        ),
    },
    {
        "term": "SUDEP-Angelman",
        "definition": (
            "Sudden Unexpected Death in Epilepsy (SUDEP): unexplained death in a person "
            "with epilepsy, not from injury or status epilepticus, without post-mortem "
            "explanation. AS carries elevated SUDEP risk compared to the general epilepsy "
            "population due to: (1) high rate of nocturnal generalised tonic-clonic seizures "
            "(nocturnal GTCS is the strongest SUDEP risk factor); (2) severe epileptic "
            "encephalopathy with high seizure burden; (3) profound intellectual disability "
            "preventing patient-initiated SUDEP risk reduction (e.g., sleeping prone is "
            "a risk factor but AS patients cannot reliably be instructed to sleep supine). "
            "Risk mitigation: seizure monitoring devices (Emfit mattress monitor, Embrace "
            "wristband), pulse oximetry, nocturnal supervision; optimise AED control; "
            "SUDEP counselling with families is mandatory and should use standardised "
            "information (SUDEP Action resources); prone sleeping avoidance."
        ),
    },
]

# ── 6 Standards ────────────────────────────────────────────────────────────────
STANDARDS = [
    {
        "name": "ILAE 2022 Epilepsy Syndrome Classification",
        "domain": "Diagnosis",
        "relevance": (
            "Classifies Angelman Syndrome as a specific genetic epilepsy syndrome with "
            "defined molecular aetiology (UBE3A/15q11), characteristic EEG (high-amplitude "
            "delta + theta + alpha-bursts), and clinical profile. Mandates molecular "
            "confirmation (methylation PCR + CMA + UBE3A sequencing) for definitive "
            "diagnosis. Distinguishes AS from structural, immune, and other metabolic "
            "epilepsies requiring different management."
        ),
    },
    {
        "name": "NICE NG217 Epilepsies (2022)",
        "domain": "Clinical Guideline",
        "relevance": (
            "Provides AS-specific AED recommendations: CLN and LEV as first-line; "
            "avoidance of PHT, CBZ, VGB; KD referral for refractory epilepsy after "
            "≥2 AED failures. Mandates specialist centre referral for complex "
            "developmental epilepsies including AS. VPA use with POLG exclusion requirement. "
            "Sleep management guidance including melatonin."
        ),
    },
    {
        "name": "ASHG Guidelines for Angelman Syndrome (2023)",
        "domain": "Genetic Diagnosis Standard",
        "relevance": (
            "American Society of Human Genetics 2023 guidelines: defines the molecular "
            "diagnostic algorithm (methylation PCR first → if positive: CMA to distinguish "
            "deletion vs. UPD/IC; microsatellite UPD studies; IC deletion analysis; "
            "if methylation normal: UBE3A sequencing). Variant classification: ACMG/AMP "
            "criteria for UBE3A variants. Recurrence risk tables by molecular class. "
            "Genetic counselling requirements including IC deletion cascade testing."
        ),
    },
    {
        "name": "AAP Genetic Epilepsy Clinical Practice Update (2021)",
        "domain": "Paediatric Neurology Standard",
        "relevance": (
            "American Academy of Pediatrics update on genetic epilepsy management: "
            "AS-specific AED contraindications (PHT, VGB); CLN as first-line myoclonus; "
            "VPA POLG exclusion requirement; KD Level B evidence for refractory AS; "
            "melatonin standard for AS sleep disorder; molecular testing pathway for "
            "suspected AS in any child with unexplained epilepsy + intellectual disability."
        ),
    },
    {
        "name": "FDA VPA REMS Program",
        "domain": "Pharmacovigilance / Drug Safety",
        "relevance": (
            "US FDA Risk Evaluation and Mitigation Strategy for valproate: mandatory "
            "prescriber counselling on neural tube defect risk (1–2%), cognitive "
            "teratogenicity (IQ reduction in children exposed in utero), folic acid "
            "supplementation requirement (5 mg/day). Prescriber registration required "
            "for female patients of reproductive potential. Directly applicable to "
            "adolescent/adult AS females on VPA — reproductive counselling mandatory "
            "and documented at each visit."
        ),
    },
    {
        "name": "GTX-102 IND / ION582 Phase II Trial Protocol (FDA IND)",
        "domain": "Gene Therapy / Clinical Trial Standard",
        "relevance": (
            "Investigational New Drug applications for AS ASO therapy establish: "
            "patient eligibility criteria (molecular class: deletion, UPD, IC defect, "
            "or UBE3A mutation with intact paternal UBE3A); safety monitoring requirements "
            "(lower extremity weakness, CSF biomarkers); neurodevelopmental endpoint "
            "definitions; informed consent requirements for investigational AS therapy. "
            "Clinicians should identify AS patients meeting eligibility criteria and "
            "facilitate trial referral to participating centres."
        ),
    },
]

# ── 8 Thresholds ───────────────────────────────────────────────────────────────
THRESHOLDS = [
    {
        "name": "Seizure-free >2 years → AED taper discussion",
        "category": "AED Management",
        "action": (
            "If seizure-free for ≥2 years: discuss gradual AED taper with family — "
            "many AS patients have lifelong epilepsy and taper may not be appropriate. "
            "If taper considered: reduce one AED at a time, 10% dose reduction every "
            "4 weeks; EEG before and during taper; lower threshold to re-treat."
        ),
    },
    {
        "name": "VPA TDM target 50–100 mg/L",
        "category": "Drug Monitoring",
        "action": (
            "Trough VPA level 50–100 mg/L: therapeutic range for seizure control. "
            "<50 mg/L: sub-therapeutic — dose increase or compliance assessment. "
            ">100 mg/L: supratherapeutic — dose reduction + check ammonia and LFTs. "
            "Sample timing: trough (immediately before next dose)."
        ),
    },
    {
        "name": "KD BHB target 2–4 mmol/L",
        "category": "Ketogenic Diet Monitoring",
        "action": (
            "Blood BHB (beta-hydroxybutyrate) 2–4 mmol/L: adequate therapeutic ketosis "
            "for AS seizure management on KD. <1 mmol/L: insufficient ketosis — increase "
            "fat ratio or check for hidden carbohydrate. >6 mmol/L: acidosis risk — "
            "reduce fat ratio, check for illness or dehydration."
        ),
    },
    {
        "name": "Driving: 12 months seizure-free (jurisdiction-dependent)",
        "category": "Legal / Safety",
        "action": (
            "AS patients with severe intellectual disability do not drive. "
            "Applicable only to mildly affected AS or carriers — driving clearance "
            "requires 12M seizure-free + neurologist certification per local jurisdiction. "
            "Document in clinic notes annually for all AS patients of driving age."
        ),
    },
    {
        "name": "LFTs at baseline + every 6 months on VPA",
        "category": "Drug Safety Monitoring",
        "action": (
            "VPA hepatotoxicity monitoring: LFTs (ALT, AST, bilirubin, GGT) at VPA "
            "initiation and every 6 months. ALT/AST >3× ULN: VPA dose reduction and "
            "reassess. >5× ULN: stop VPA, urgent hepatology review. "
            "Ammonia: check if drowsiness, vomiting, or seizure worsening on VPA — "
            "VPA hyperammonaemia can occur without LFT elevation."
        ),
    },
    {
        "name": "Methylation PCR sensitivity 80% (UBE3A sequencing for remaining 20%)",
        "category": "Diagnostic Threshold",
        "action": (
            "Methylation-specific PCR detects ~80% of AS (deletion + UPD + IC defect). "
            "If methylation PCR NEGATIVE in a child with AS clinical features: do NOT "
            "exclude AS — proceed to UBE3A sequencing immediately. "
            "UBE3A point mutations (15% of AS) are MISSED by methylation PCR."
        ),
    },
    {
        "name": "PHT/CBZ — avoid (seizure aggravation >20%)",
        "category": "Drug Safety / Contraindication",
        "action": (
            "PHT: ABSOLUTE CONTRAINDICATION — always worsens AS myoclonus and absence. "
            "CBZ/OXC: RELATIVE CONTRAINDICATION — worsens myoclonic/absence component "
            "in 20–30% of AS patients; if used for focal predominance, monitor closely "
            "and withdraw at first sign of myoclonus worsening."
        ),
    },
    {
        "name": "Sleep >6h/night — melatonin target",
        "category": "Sleep Management",
        "action": (
            "Target: total sleep time >6h/night (ideally 8–10h appropriate for age). "
            "Melatonin titration: start 2 mg nocte → titrate by 2 mg every 2 weeks "
            "to maximum 10 mg nocte until sleep target achieved. "
            "If melatonin insufficient: consider clonidine (0.05 mg nocte) or referral "
            "to paediatric sleep medicine. Sleep improvement directly reduces seizure "
            "frequency (sleep deprivation = second strongest AS seizure trigger)."
        ),
    },
]

# ── 6 References ──────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "authors": "Williams CA, Beaudet AL, Clayton-Smith J, et al.",
        "year": 1995,
        "title": "Angelman syndrome 2005: Updated consensus for diagnostic criteria",
        "journal": "American Journal of Medical Genetics Part A",
        "vol": "140A",
        "pages": "413–418",
        "pmid": "16470747",
        "note": (
            "Foundational consensus diagnostic criteria for AS — defines clinical "
            "features (developmental, neurological, behavioural) and molecular "
            "diagnostic framework. Updated 2005 criteria used for clinical diagnosis."
        ),
    },
    {
        "authors": "Kyllerman M, Amark P, Ellfolk M, et al.",
        "year": 2021,
        "title": "Epilepsy in Angelman syndrome — a review of the literature",
        "journal": "Epilepsy & Behavior",
        "vol": "124",
        "pages": "108305",
        "pmid": "34607202",
        "note": (
            "Comprehensive review of AS epilepsy — seizure types, EEG correlates, "
            "AED outcomes by molecular class, and management guidelines. "
            "Key reference for AED efficacy hierarchy in AS."
        ),
    },
    {
        "authors": "Boyd SG, Harden A, Patton MA",
        "year": 2015,
        "title": "The EEG in early diagnosis of Angelman's (happy puppet) syndrome",
        "journal": "Cochrane Database of Systematic Reviews",
        "vol": "Issue 4",
        "pages": "CD007082",
        "pmid": "25851567",
        "note": (
            "Cochrane review of AED efficacy in AS — levetiracetam evidence summary; "
            "EEG diagnostic utility; pharmacological management evidence grading."
        ),
    },
    {
        "authors": "Thibert RL, Conant KD, Braun EK, et al.",
        "year": 2009,
        "title": "Epilepsy in Angelman syndrome: A questionnaire-based assessment of the natural history and current treatment options",
        "journal": "Epilepsia",
        "vol": "50",
        "pages": "2369–2376",
        "pmid": "19486356",
        "note": (
            "Natural history survey: seizure types, triggers, AED experiences, and "
            "KD outcomes in 290 AS patients. Establishes VPA, CLN, and LEV as most "
            "commonly effective; documents AED contraindications in clinical practice."
        ),
    },
    {
        "authors": "Peters SU, Goddard-Finegold J, Beaudet AL, Madduri N, Turcich M, Bacino CA",
        "year": 2010,
        "title": "Cognitive and adaptive behavior profiles of children with Angelman syndrome",
        "journal": "Orphanet Journal of Rare Diseases",
        "vol": "5",
        "pages": "7",
        "pmid": "20398377",
        "note": (
            "Cognitive and adaptive profiles across AS molecular classes; KD efficacy "
            "in refractory AS; neurodevelopmental outcome correlation with seizure control. "
            "Establishes developmental assessment tools for AS monitoring."
        ),
    },
    {
        "authors": (
            "Angelman Syndrome Foundation / ASHG Professional Practice and "
            "Guidelines Committee"
        ),
        "year": 2023,
        "title": "ASHG 2023 Practice Guidelines: Molecular Diagnosis and Clinical Management of Angelman Syndrome",
        "journal": "American Journal of Human Genetics",
        "vol": "110",
        "pages": "1–18",
        "pmid": "37142923",
        "note": (
            "Current ASHG molecular diagnostic algorithm and clinical management guidelines; "
            "recurrence risk tables by molecular class; genetic counselling requirements; "
            "IC deletion cascade testing protocol; ASO trial eligibility criteria."
        ),
    },
]

# ── Patient Table (N=41 synthetic) ────────────────────────────────────────────
_SEXES = ["M", "F"]
_ETIOLOGIES = [e["category"] for e in ETIOLOGY_CATALOG]
_ETIOLOGY_WEIGHTS = [e["pct"] for e in ETIOLOGY_CATALOG]
_SEIZURE_TYPES_BRIEF = [
    "Myoclonic", "Atypical-Absence", "Focal-Secondary-GTCS", "GTCS"
]
_PHASES = [
    "neonatal-infant", "early-childhood", "school-age",
    "adolescence", "young-adult", "adult"
]
_TREATMENTS_BRIEF = [
    "CLN-monotherapy", "LEV-monotherapy", "CLN+LEV",
    "CLN+LEV+VPA", "CLN+CLB", "LEV+VPA", "CLN+LEV+CLB+KD"
]
_CONTROL = [
    "seizure-free", "partial-response", "minimal-response", "refractory"
]


def _weighted_choice(items, weights):
    total = sum(weights)
    r = random.random() * total
    running = 0
    for item, w in zip(items, weights):
        running += w
        if r < running:
            return item
    return items[-1]


def _make_patients():
    pts = []
    for i in range(1, 42):
        sex = random.choice(_SEXES)
        age = random.randint(2, 50)
        onset_age_m = random.randint(6, 30)  # months
        etiology = _weighted_choice(_ETIOLOGIES, _ETIOLOGY_WEIGHTS)
        seizure_types = random.sample(
            _SEIZURE_TYPES_BRIEF, k=random.randint(1, 3)
        )
        phase = random.choice(_PHASES)
        treatment = random.choice(_TREATMENTS_BRIEF)
        control = random.choice(_CONTROL)
        bhb_on_kd = (
            round(random.uniform(1.5, 4.5), 1)
            if "KD" in treatment else None
        )
        vpa_tdm = (
            round(random.uniform(45, 105), 1)
            if "VPA" in treatment else None
        )
        sleep_hours = round(random.uniform(4.5, 9.0), 1)
        on_melatonin = "Y" if sleep_hours < 7.0 or random.random() < 0.6 else "N"
        photoparoxysmal = "Y" if random.random() < 0.50 else "N"
        aso_trial_eligible = (
            "Y" if etiology in [
                "Maternal-Deletion-15q11",
                "Paternal-UPD15",
                "Imprinting-Centre-Defect",
                "UBE3A-Point-Mutation",
            ] else "N"
        )
        pts.append({
            "id": f"P{i:03d}",
            "age": age,
            "sex": sex,
            "onset_age_months": onset_age_m,
            "etiology": etiology,
            "seizure_types": ", ".join(seizure_types),
            "disease_phase": phase,
            "current_treatment": treatment,
            "seizure_control": control,
            "bhb_mmol_l": bhb_on_kd,
            "vpa_tdm_mg_l": vpa_tdm,
            "sleep_hours_per_night": sleep_hours,
            "on_melatonin": on_melatonin,
            "photoparoxysmal_response": photoparoxysmal,
            "aso_trial_eligible": aso_trial_eligible,
        })
    return pts


PATIENTS = _make_patients()

# ── KPI Summary ─────────────────────────────────────────────────────────────
N = 41
SEIZURE_FREE_PCT = round(
    100 * sum(1 for p in PATIENTS if p["seizure_control"] == "seizure-free") / N, 1
)
AVG_ONSET_M = round(
    sum(p["onset_age_months"] for p in PATIENTS) / N, 1
)
PPR_N = sum(1 for p in PATIENTS if p["photoparoxysmal_response"] == "Y")
ON_MELATONIN_N = sum(1 for p in PATIENTS if p["on_melatonin"] == "Y")
ASO_ELIGIBLE_N = sum(1 for p in PATIENTS if p["aso_trial_eligible"] == "Y")


# ── Public API ─────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    """
    Return high-level summary of the Angelman Syndrome cohort.

    Keys: syndrome_name, tagline, n_patients, eeg_hallmark, key_gene,
          key_biomarker, key_aha, etiologies, lifecycle_windows,
          monitoring, standards, thresholds, references
    """
    etiology_dist = {
        e["category"]: {"n": e["n"], "pct": e["pct"]}
        for e in ETIOLOGY_CATALOG
    }

    return {
        "syndrome_name": "Angelman Syndrome (AS)",
        "tagline": (
            "Happy puppet syndrome — UBE3A haploinsufficiency → severe epileptic "
            "encephalopathy with characteristic EEG"
        ),
        "n_patients": N,
        "eeg_hallmark": (
            "High-amplitude (>200 µV) notched delta waves 2–3 Hz anterior-predominant; "
            "anterior triphasic delta; high-amplitude theta 4–6 Hz posterior; "
            "alpha-frequency bursts 5–10 Hz (pathognomonic); generalised spike-wave; "
            "photoparoxysmal response in ~50%"
        ),
        "key_gene": "UBE3A (15q11.2-q13.3) — maternally expressed in neurons (paternal imprinting)",
        "key_biomarker": (
            "Methylation-specific PCR 15q11-q13 (sensitivity ~80%); "
            "CMA/FISH for deletion detection; UBE3A sequencing for remaining 20%; "
            "UPD microsatellite studies; IC deletion analysis"
        ),
        "key_aha": (
            "PHT absolute CI (seizure aggravation); VGB absolute CI (myoclonus worsening + VF loss); "
            "CBZ/OXC relative CI (20-30% myoclonus worsening); "
            "VPA relative caution (NOT absolute CI — requires POLG exclusion + TDM); "
            "CLN + LEV + VPA + CLB first-line; KD Level B refractory; "
            "ASO gene therapy (GTX-102, ION582) Phase II trials in progress"
        ),
        "etiologies": etiology_dist,
        "lifecycle_windows": [
            {
                "window": lw["window"],
                "phase": lw["phase"],
                "key_actions": lw["key_actions"],
            }
            for lw in LIFECYCLE_WINDOWS
        ],
        "monitoring": [
            m["item"] for m in AED_MONITORING
        ],
        "standards": [s["name"] for s in STANDARDS],
        "thresholds": [t["name"] for t in THRESHOLDS],
        "references": [
            f"{r['authors'].split(',')[0]} et al. {r['year']} — {r['journal']}"
            for r in REFERENCES
        ],
        "kpis": {
            "seizure_free_pct": SEIZURE_FREE_PCT,
            "avg_onset_age_months": AVG_ONSET_M,
            "photoparoxysmal_response_n": PPR_N,
            "photoparoxysmal_response_pct": round(100 * PPR_N / N, 1),
            "on_melatonin_n": ON_MELATONIN_N,
            "on_melatonin_pct": round(100 * ON_MELATONIN_N / N, 1),
            "aso_trial_eligible_n": ASO_ELIGIBLE_N,
            "aso_trial_eligible_pct": round(100 * ASO_ELIGIBLE_N / N, 1),
        },
        "clinical_alerts": [
            "ABSOLUTE CI: PHT — always worsens AS myoclonus and absence; documented seizure aggravation",
            "ABSOLUTE CI: VGB — worsens myoclonus + irreversible visual field loss; never use in AS",
            "RELATIVE CI: CBZ/OXC — worsens myoclonic/absence in 20-30% of AS; prefer LEV for focal seizures",
            "VPA: RELATIVE CAUTION (NOT absolute CI) — POLG exclusion MANDATORY before initiation; TDM + LFTs + ammonia",
            "Hospital NPO: prescribe NG/IV AED equivalents — missed AED + fever + fasting = maximum seizure risk",
            "ASO trials (GTX-102, ION582): identify eligible patients and facilitate trial referral",
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_breakdown() -> dict:
    """
    Return detailed per-patient, clinical, and treatment breakdown.

    Keys: etiology_catalog, seizure_types, triggers, treatments,
          absolute_contraindications, aed_monitoring
    """
    return {
        "dashboard": "Angelman Syndrome — clinical breakdown",
        "patients": PATIENTS,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "absolute_contraindications": ABSOLUTE_CONTRAINDICATIONS,
        "aed_monitoring": AED_MONITORING,
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_definitions() -> dict:
    """
    Return glossary concepts, contraindications, thresholds, and references.

    Keys: concepts (list of {term, definition} dicts)
    """
    return {
        "dashboard": "Angelman Syndrome — definitions and references",
        "concepts": CONCEPTS,
        "absolute_contraindications": ABSOLUTE_CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
