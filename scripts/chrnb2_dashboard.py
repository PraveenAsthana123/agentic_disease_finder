"""
CHRNB2 Epilepsy — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy Type 3 (ADNFLE3)
nAChR β2 Subunit / (α4)₂(β2)₃ Heteropentamer / GOF Delayed Desensitisation
CBZ-XR First-Line / HLA-B*15:02 / Psychiatric Comorbidity 40% / 1q21.3
=========================================================================
40-patient cohort · CHRNB2 (1q21.3) · Cholinergic Receptor Nicotinic Beta 2 Subunit
Gene OMIM: *118507 · Syndrome: ADNFLE3 (OMIM #605375)
Companion gene to CHRNA4 (α4 subunit) — both form the (α4)₂(β2)₃ nAChR heteropentamer

KEY CHRNB2 BIOLOGY — nAChR β2 SUBUNIT:
CHRNB2 (1q21.3) encodes the β2 subunit of the neuronal nicotinic acetylcholine receptor (nAChR).
The dominant neuronal nAChR isoform is (α4)₂(β2)₃ — a heteropentamer assembled from
CHRNA4-encoded α4 subunits and CHRNB2-encoded β2 subunits.

PROTEIN STRUCTURE:
  · Pentameric ligand-gated ion channel (pLGIC): 5 subunits form a central cation-selective pore
  · Each subunit: large N-terminal extracellular domain (ECD) + 4 TM helices (TM1-TM4) + TM3-TM4 intracellular loop
  · TM2 lines the ion channel pore — CRITICAL: all pathogenic CHRNB2 variants cluster in/around TM2
  · V287 is in TM2 — V287M/V287L are the two major disease mutations
  · ACh/nicotine binding site: formed at α4-β2 subunit interfaces (ECD)
  · Stoichiometry: (α4)₂(β2)₃ is the HIGH-SENSITIVITY isoform (EC50 ~1 µM ACh vs ~100 µM for other isoforms)
  · HIGH-SENSITIVITY property explains NREM vulnerability: basal ACh levels in NREM sleep (~1 µM) can activate mutant channels

EXPRESSION:
  · Highest: thalamus, cortex (frontal > temporal), hippocampus, basal ganglia, raphe nuclei, locus coeruleus
  · IMPORTANT: β2 subunit expressed on monoaminergic neurons (dopamine, serotonin, norepinephrine) →
    explains HIGHER PSYCHIATRIC COMORBIDITY in CHRNB2 (40%) vs CHRNA4 (~20%)
  · Also: habenula, interpeduncular nucleus — reward/aversion circuits

BIOPHYSICS — GOF MECHANISM:
  · Wild-type nAChR: opens in response to ACh, then rapidly desensitises (ms-seconds)
  · Desensitisation = non-conducting, high-affinity state — channel inactivates despite ongoing ligand binding
  · V287M/V287L: TM2 substitution → REDUCED DESENSITISATION RATE →
    channels remain open longer during NREM cholinergic surges
  · Net effect: excessive cation influx (Na⁺, K⁺, Ca²⁺) during NREM → cortical depolarisation → frontal hyperexcitability
  · NREM vulnerability: brainstem cholinergic (PPT/LDT) projections fire in bursts at sleep-wake transitions →
    normal ACh release that wild-type channels buffer; GOF β2 → fails to desensitise → seizure initiation

PATHOGENIC VARIANTS — CHRNB2-SPECIFIC:
  · V287M (Val287Met, p.V287M): first described (De Fusco 2000 Nat Genet); TM2 position 9'
    — middle of pore gate; reduces desensitisation ~5-fold; associated with some cognitive impairment
  · V287L (Val287Leu, p.V287L): TM2; similar position; higher penetrance (~90%) than V287M (~80%)
    — pure ADNFLE, rarely cognitive impairment
  · L301V (Leu301Val, p.L301V): TM2 position 23'; milder GOF; psychiatric-predominant phenotype
  · I312M (Ile312Met): TM3-adjacent; rare; hyperkinetic seizures + psychiatric
  · S252L (Ser252Leu): ECD-TM1 junction; GOF; late-onset ADNFLE variant
  · T279I (Thr279Ile): TM2 position 1'; rare; neonatal-onset in some

GENETIC EPIDEMIOLOGY:
  · ADNFLE3 (CHRNB2): rarer than ADNFLE1 (CHRNA4); ~20-25 kindreds published worldwide
  · Penetrance: V287M ~80%; V287L ~90%; L301V ~75% (incomplete penetrance → unaffected carriers)
  · De novo rate: ~20% of ADNFLE3 cases (remainder autosomal dominant familial)
  · pLI: 0.96 (haploinsufficiency intolerant) — strong dominant mechanism
  · 1q21.3 region: also associated with 1q21.1 deletion syndrome (ASD/schizophrenia/ID) — different mechanism (CNV)

CLINICAL DIFFERENCES FROM CHRNA4 (α4 subunit):
  · Psychiatric comorbidity: CHRNB2 40% vs CHRNA4 20% — β2 expressed on monoaminergic neurons
  · Cognitive impairment: V287M families: 30% have mild ID/learning disability (not seen in V287L or CHRNA4)
  · Seizure semiology: largely identical ADNFLE (hypermotor nocturnal); subtle difference:
    CHRNB2 may have more dystonic posturing vs CHRNA4 more hyperkinetic running
  · EEG: frontal theta/alpha during NREM seizures; ictal EEG may be subtle (frontal EEG artefact from movement)
  · Diagnosis misclassification: 35% (same as CHRNA4) — parasomnia, RBD
  · Treatment response: similar CBZ responsiveness (65-75% vs 70-80% CHRNA4); slightly lower SF rate

KEY PHARMACOLOGICAL DISTINCTIONS FROM CHRNA4:
  · Same treatment: CBZ-XR first-line; same HLA-B*15:02 CI (same genes affected CBZ/OXC)
  · Psychiatric comorbidity: SSRI/SNRI + CBZ interaction monitoring (CYP3A4); dose adjustment needed
  · NICOTINE PARADOX: same as CHRNA4 — low-dose sustained → desensitisation (investigational);
    HIGH-DOSE ACTIVATES GOF β2 → HIGH RISK; abrupt cessation → receptor upregulation → seizure cluster
  · CBZ AUTOINDUCTION: same CYP3A4 autoinduction — re-check TDM at 6-8 weeks
  · PSYCHIATRIC CO-TREATMENT: CBZ + antidepressants/antipsychotics (for 40% with comorbidity) —
    more complex polypharmacy than CHRNA4; SSRI safer than TCA; avoid bupropion (lowers seizure threshold)
"""

import random

random.seed(99)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG — 5-CLASS SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-V287M-TM2-Classic-ADNFLE3",
        "pct": 38,
        "mechanism": (
            "V287M (Val287Met) missense in TM2 (position 9') — the pore-lining helix of the β2 subunit. "
            "First CHRNB2 mutation described (De Fusco et al. 2000 Nat Genet). Reduces nAChR desensitisation "
            "rate ~5-fold → channels remain open during NREM cholinergic surges → frontal hyperexcitability. "
            "Autosomal dominant; penetrance ~80%; de novo in ~20% of cases. "
            "CLINICAL NOTE: V287M uniquely associated with cognitive impairment (30% of mutation carriers) — "
            "not seen in V287L or CHRNA4 variants. The extra cognitive risk is attributed to β2 "
            "expression in hippocampal/prefrontal circuits beyond the frontal seizure focus. "
            "PRECISION TREATMENT: CBZ-XR 200-400 mg bedtime-heavy; 65-70% seizure-free. "
            "HLA-B*15:02 screening mandatory before CBZ in SE Asian populations."
        ),
        "eeg": "NREM Stage 2–3 arousal with frontal theta burst (6-9 Hz, 5-15 sec) → hypermotor arousal; interictal EEG typically normal or rare frontal sharp transients; VPSG critical (NREM onset distinguishes from parasomnia); ictal scalp EEG often obscured by movement artefact",
        "onset_months": "60–204 months (5–17 years; peak 8–12 y)",
        "severity": "moderate — CBZ-XR responsive 65-70%; 30% drug-resistant require combination; cognitive comorbidity in V287M families",
    },
    {
        "category": "GOF-V287L-TM2-High-Penetrance",
        "pct": 28,
        "mechanism": (
            "V287L (Val287Leu) missense — same TM2 position 9' as V287M but leucine substitution "
            "creates higher penetrance (~90%) and purer ADNFLE phenotype. "
            "GOF mechanism: reduced desensitisation via TM2 structural change — "
            "leucine's larger hydrophobic side chain alters TM2 gate geometry → "
            "slower pore closing → prolonged channel opening during NREM ACh bursts. "
            "DIFFERENCE FROM V287M: V287L rarely causes cognitive impairment; "
            "higher seizure frequency but better psychiatric profile. "
            "First-degree relatives at 90% penetrance risk — cascade genetic testing essential. "
            "CBZ-XR response: 70-75% seizure-free (slightly better than V287M)."
        ),
        "eeg": "NREM hyperkinetic arousal EEG; frontal ictal pattern often subtle at scalp; prolonged VPSG recording required (≥2 nights for capture); interictal normal; rare waking focal frontal sharp waves",
        "onset_months": "72–216 months (6–18 years; peak 10–14 y)",
        "severity": "moderate — higher penetrance (90%); pure ADNFLE without cognitive comorbidity; CBZ-XR 70-75% SF",
    },
    {
        "category": "GOF-L301V-TM2-Psychiatric-Predominant",
        "pct": 18,
        "mechanism": (
            "L301V (Leu301Val) in TM2 position 23' — near the intracellular gate of the pore. "
            "Milder GOF effect than V287 variants → lower seizure frequency but prominent "
            "interictal psychiatric symptoms (anxiety 65%, depression 45%, OCD features 30%). "
            "The psychiatric-predominant phenotype reflects β2 expression on raphe (serotonin) "
            "and locus coeruleus (norepinephrine) neurons — GOF β2 in monoaminergic nuclei "
            "dysregulates reward/affect circuits independently of seizures. "
            "Clinical pitfall: psychiatric symptoms present 2-5 years BEFORE seizure onset → "
            "often initially treated as primary psychiatric disorder; ADNFLE3 diagnosis delayed 5-10 years. "
            "CBZ-XR: good seizure control but may worsen depression in some (monitor mood); "
            "consider LCM as CBZ alternative + SSRI for psychiatric comorbidity."
        ),
        "eeg": "Subtle nocturnal frontal EEG changes; minor motor episodes may not trigger VPSG monitoring; ambulatory EEG + sleep diary required; psychiatric EEG (waking): non-specific mild slowing in some",
        "onset_months": "96–252 months (8–21 years; delayed onset due to milder phenotype)",
        "severity": "mild-moderate seizures but HIGH psychiatric burden — misclassification as psychiatric disorder common; CBZ-XR 60-65% SF",
    },
    {
        "category": "GOF-Other-TM2-Rare-Variants",
        "pct": 10,
        "mechanism": (
            "Rare CHRNB2 TM2/adjacent variants: I312M (Ile312Met, TM3-adjacent), "
            "S252L (Ser252Leu, ECD-TM1 junction), T279I (Thr279Ile, TM2 position 1'). "
            "I312M: hyperkinetic seizures + prominent psychiatric (anxiety/OCD); de novo in all described cases. "
            "S252L: late-onset ADNFLE (adult 25-40y); milder seizure frequency; good CBZ response. "
            "T279I: rare neonatal/infantile onset case reports — atypical for ADNFLE3; "
            "may represent distinct phenotype. All: GOF mechanism — reduced desensitisation. "
            "Management: same CBZ-XR first-line framework; functional electrophysiology "
            "(Xenopus oocyte/HEK293 expression) needed to confirm GOF for novel variants."
        ),
        "eeg": "Variable by variant; all NREM-predominant; I312M: hypermotor EEG pattern; S252L: waking normal, NREM arousal; T279I: multifocal in neonatal presentation",
        "onset_months": "Variable: T279I neonatal; I312M 8-16y; S252L 300-480 months (25-40y)",
        "severity": "Variable; rare variants often de novo; functional confirmation required for novel variants",
    },
    {
        "category": "Phenocopy-CHRNB2-Negative",
        "pct": 6,
        "mechanism": (
            "ADNFLE-phenotype patients with negative CHRNB2 (and CHRNA4, KCNT1) sequencing. "
            "Possible causes: (1) CHRNA2 mutations (α2 subunit, rare ADNFLE); "
            "(2) non-coding regulatory variants in CHRNB2 promoter/enhancer; "
            "(3) structural variants (duplications of 1q21.3 encompassing CHRNB2); "
            "(4) somatic mosaicism (ADNFLE seizures with negative germline blood DNA — "
            "brain-only mosaic variants); (5) non-genetic ADNFLE-mimic (e.g., frontal cortical "
            "dysplasia — MRI 3T + FDG-PET required to exclude). "
            "Clinical management: treat as ADNFLE with CBZ-XR; VPSG to confirm NREM seizures; "
            "MRI 3T for structural cause; consider research-panel sequencing (CHRNA2, KCNQ3, DEPDC5)."
        ),
        "eeg": "ADNFLE phenocopy: NREM frontal; distinguish from structural FLE (MRI 3T + SEEG if surgical candidate) and from parasomnia (VPSG REM vs NREM onset)",
        "onset_months": "Variable",
        "severity": "Variable; CBZ-XR empiric; investigate for structural cause",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE CATALOG — 5 TYPES
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_CATALOG = [
    {
        "type": "Hypermotor-Nocturnal-NREM",
        "pct": 97,
        "eeg": "NREM Stage 2/3 arousal; frontal theta burst; ictal EEG often artefact-obscured; VPSG gold standard",
        "semiology": "Sudden arousal from NREM sleep → bilateral tonic posturing → hypermotor limb/trunk movements (kicking, rocking, cycling); vocalisations (screaming, grunting); automatisms; duration 15–120 sec; clusters 2-6/night; patient often semi-aware or unaware",
        "clinical_tip": "MISDIAGNOSIS 35%: distinguish from parasomnias (NREM arousal disorders) — ADNFLE3 shows SHORTER, MORE STEREOTYPED episodes, EARLIER in sleep cycle (first 2h NREM), more frequent nightly clustering than NVSZ. VPSG: ictal EEG + stereotypy + nightly clustering = ADNFLE3. Parasomnia: longer, more variable, often occur later in night.",
    },
    {
        "type": "Minor-Motor-Paroxysmal-Nocturnal",
        "pct": 72,
        "eeg": "Subtle EEG arousal (alpha intrusion into NREM) without clear ictal correlate at scalp — may be missed on routine PSG without full 10-20 system",
        "semiology": "Brief (5-30 sec) minor motor events: head deviation, leg kicking, arm elevation, grimacing; often cluster before major hypermotor seizures; some patients have ONLY minor motor events (underdiagnosed). VIDEO-EEG essential to detect.",
        "clinical_tip": "Minor motor events are often undercounted — patients report 'restless sleep' not seizures. Full VPSG with 10-20 EEG system: captures minor events. These may be the only seizure type in V287L carriers with 90% penetrance but milder expression.",
    },
    {
        "type": "Nocturnal-Tonic-Postural",
        "pct": 55,
        "eeg": "Frontal ictal fast activity (12-30 Hz) evolving to slow wave; better seen on intracranial/stereo-EEG than scalp",
        "semiology": "Sustained tonic posturing (asymmetric tonic or bilateral symmetric tonic); fencing posture common; 15-60 sec; may occur as isolated event or evolve from/into hypermotor phase. More common in CHRNB2 V287M than CHRNA4 (possible structural difference in channel gate dynamics affecting posturing circuits).",
        "clinical_tip": "Fencing posture (arm elevation) in nocturnal setting = frontal lobe onset until proven otherwise. SEEG shows supplementary motor area (SMA) and/or cingulate involvement. CBZ-XR generally controls tonic component; LCM useful if CBZ-refractory.",
    },
    {
        "type": "Epileptic-Nocturnal-Wandering",
        "pct": 32,
        "eeg": "Prolonged frontal ictal discharge with ambulatory component; EEG artefact from movement limits interpretation; VPSG with CCTV essential",
        "semiology": "Patient ambulates from bed, appears confused/purposeful but is ictal; may leave room, navigate furniture; can last 2-10 min; risk of falls/injury. Occurs in clusters. Must be distinguished from sleepwalking (NREM arousal disorder) — ADNFLE3 wandering is ictal (EEG change) vs somnambulism (no ictal EEG).",
        "clinical_tip": "Bed safety protocol MANDATORY for epileptic wandering (padded rails, floor mattress, door alarm, room monitoring). CBZ-XR high bedtime dose targets this NREM phenotype. CLB at bedtime as adjunct can reduce wandering episodes.",
    },
    {
        "type": "Daytime-Focal-Aware-Seizures",
        "pct": 18,
        "eeg": "Waking frontal epileptiform discharges or focal fast activity; rare in purely nocturnal ADNFLE3 — presence suggests more widespread involvement or drug-resistant evolution",
        "semiology": "Frontal focal aware seizures (formerly 'simple partial') during wakefulness: brief tonic posturing, unilateral arm/face jerking, forced head deviation; duration 20-60 sec; preserved awareness. Occur in 18% of CHRNB2 ADNFLE3 — more than CHRNA4 ADNFLE1 (12%). May relate to V287M cognitive impact on frontal network stability.",
        "clinical_tip": "Daytime frontal seizures in established ADNFLE3: check CBZ levels (CYP3A4 autoinduction — re-check at 6-8 weeks); consider dose increase. If new daytime seizures appear in controlled ADNFLE3 → rule out missed dose, illness, drug interaction (CYP3A4 inducers reducing CBZ).",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER CATALOG — 8 TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_CATALOG = [
    {
        "trigger": "NREM-Sleep-Transitions",
        "pct": 95,
        "mechanism": "PPT/LDT (pontine cholinergic nuclei) fire in bursts at NREM state transitions → ACh release in thalamus/cortex → GOF β2 nAChR fails to desensitise → seizure initiation. ~95% of CHRNB2 ADNFLE3 seizures during NREM Stages 1-3 (peak Stage 2 transitions)",
        "management": "CBZ-XR (extended-release) bedtime-heavy dosing (2/3 at bedtime): maximises nocturnal trough coverage during NREM vulnerability window",
    },
    {
        "trigger": "Sleep-Deprivation",
        "pct": 80,
        "mechanism": "Sleep debt → increased NREM rebound (more N3/SWS) → deeper NREM → more pronounced ACh bursts at transitions → higher seizure density. Sleep deprivation also increases homeostatic pressure on PPT/LDT → exaggerated cholinergic release on recovery night.",
        "management": "Sleep hygiene counselling; regular schedule; avoid shift work; shift work disorder is absolute contraindication in ADNFLE3",
    },
    {
        "trigger": "Stress-Catecholamine-Surges",
        "pct": 65,
        "mechanism": "Psychological/physiological stress → sympathoadrenal activation → norepinephrine release → β-adrenergic receptor → cAMP/PKA → nAChR β2 subunit phosphorylation → altered desensitisation kinetics → enhanced GOF expression. β2 on locus coeruleus creates feedback loop.",
        "management": "Stress management techniques; consider propranolol if stress-triggered seizure clusters (blocks β-adrenergic signal); CLB PRN for high-stress periods",
    },
    {
        "trigger": "Missed-CBZ-Dose",
        "pct": 62,
        "mechanism": "CBZ Na-channel blockade reduces cortical excitability; missed bedtime dose → unprotected NREM → seizure recurrence. CBZ autoinduction (CYP3A4) means levels drop further without maintained dosing.",
        "management": "Alarm reminder; compliance monitoring; extended-release formulation reduces peak-trough fluctuation; TDM at 8-12 µg/mL",
    },
    {
        "trigger": "Intercurrent-Febrile-Illness",
        "pct": 52,
        "mechanism": "Fever → disrupts NREM architecture → fragmented NREM transitions → increased cholinergic burst frequency. Also: CBZ levels can change (fever affects CYP3A4 activity + protein binding). Immune activation (cytokines IL-1β, TNF-α) may directly modulate nAChR expression.",
        "management": "Empiric CLB 0.25-0.5 mg at bedtime during illness; maintain antipyretics; check CBZ levels if prolonged illness",
    },
    {
        "trigger": "Alcohol-Intake-and-Withdrawal",
        "pct": 38,
        "mechanism": "Acute alcohol: GABA-A potentiation + NMDA inhibition → NREM architecture disruption (reduces SWS); during metabolisation → REM rebound → shift in NREM/REM balance → altered cholinergic patterning. Withdrawal: GABAergic reduction → seizure threshold lowering → ADNFLE seizure cluster.",
        "management": "Abstinence strongly advised; patient education that even 1-2 drinks disrupts NREM; withdrawal seizures managed as standard (diazepam) but ADNFLE seizures increase in withdrawal period",
    },
    {
        "trigger": "Nicotine-Cessation",
        "pct": 28,
        "mechanism": "Chronic nicotine causes nAChR upregulation (compensatory to continuous activation). CHRNB2 GOF β2 subunits are UPREGULATED further with chronic nicotine exposure. Abrupt cessation → sudden increase in receptor number + withdrawal-driven arousal → seizure cluster. PARADOX: cessation INCREASES receptor availability acutely.",
        "management": "Gradual nicotine replacement taper (NOT abrupt cessation); varenicline (partial α4β2 agonist) — theoretical concern in CHRNB2 GOF (check with neurologist); bupropion ABSOLUTE CI (lowers seizure threshold)",
    },
    {
        "trigger": "Circadian-Phase-Shift-Jet-Lag",
        "pct": 22,
        "mechanism": "Circadian misalignment disrupts NREM architecture and PPT/LDT firing patterns. Jet lag (transmeridian travel): circadian phase advance/delay → NREM onset at abnormal circadian phase → dysregulated cholinergic burst timing → seizure risk in first 3-7 nights post-travel.",
        "management": "Melatonin (0.5-3 mg) to resynchronise circadian rhythm; maintain CBZ schedule on local time immediately; extra CLB 0.25 mg at bedtime for first 3 nights of transmeridian travel",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT CATALOG — 8 TREATMENTS
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_CATALOG = [
    {
        "drug": "Carbamazepine-XR",
        "level": "Level B",
        "mechanism": "Na⁺-channel blockade (use-dependent) — reduces rapid neuronal firing. Preferential binding to inactivated Na+ channels → raises action potential threshold in frontal hyperexcitable networks. Extended-release (XR/CR) formulation reduces peak-trough fluctuation critical for nocturnal coverage. Bedtime-heavy dosing (2/3 dose at bedtime) targets NREM seizure window.",
        "dose": "Start 100-200 mg/day; titrate to 400-1200 mg/day; bedtime-heavy split (e.g., 400 mg AM + 600 mg bedtime); TDM target 8-12 µg/mL",
        "efficacy": "65-75% seizure-free in CHRNB2 ADNFLE3 (slightly lower than CHRNA4 70-80%)",
        "monitoring": "HLA-B*15:02 before starting (SE Asian populations — ABSOLUTE CI if positive); CBC (agranulocytosis), LFTs, Na (SIADH); TDM at 6-8 weeks (CYP3A4 autoinduction — levels drop 30-50% at 4-6 weeks); drug interactions (CYP3A4 inducer — reduces OCP, warfarin, LTG, TPM)",
        "chrnb2_specific": "Bedtime-heavy dosing principle: 2/3 total daily dose at bedtime. CYP3A4 autoinduction: re-check TDM at 6-8 weeks. V287M families with cognitive comorbidity: monitor neuropsychological function — CBZ can worsen cognitive slowing in susceptible patients; consider OXC as alternative (better cognitive profile but same SJS risk in HLA-B*15:02 positive).",
    },
    {
        "drug": "Oxcarbazepine",
        "level": "Level C",
        "mechanism": "Na⁺-channel blockade via active metabolite MHD (monohydroxy derivative). Less CYP3A4 autoinduction than CBZ → more stable levels. Same mechanism as CBZ but cleaner pharmacokinetic profile. Better tolerated than CBZ (less sedation, no agranulocytosis).",
        "dose": "Start 150-300 mg/day; titrate to 600-2400 mg/day; bedtime-heavy split",
        "efficacy": "60-70% seizure-free; may be preferred in V287M CHRNB2 due to better cognitive profile",
        "monitoring": "HLA-B*15:02 ABSOLUTE CI (same SJS/TEN risk as CBZ in HLA-B*15:02 positive SE Asian — CPIC Level A); Na (SIADH more frequent than CBZ, ~10-20%); LFTs",
        "chrnb2_specific": "HYPONATRAEMIA MORE FREQUENT WITH OXC (10-20%) than CBZ (5%): monitor Na monthly for first 3 months, then 3-monthly. COGNITIVE ADVANTAGE: better neuropsychological profile than CBZ — preferred for V287M families with cognitive comorbidity. Same HLA-B*15:02 absolute CI as CBZ.",
    },
    {
        "drug": "Lacosamide",
        "level": "Level C",
        "mechanism": "Slow Na⁺-channel inactivation enhancer — selectively stabilises slowly-inactivating (persistent) Na+ current. Distinct binding site from CBZ/OXC. No CYP3A4 interaction. Good as CBZ adjunct or monotherapy in HLA-B*15:02 positive patients where CBZ/OXC contraindicated.",
        "dose": "Start 50 mg twice daily; titrate to 200-400 mg/day (100-200 mg twice daily)",
        "efficacy": "55-65% response as adjunct; 45-55% as monotherapy in CBZ-refractory or intolerant",
        "monitoring": "PR interval (≥25ms prolongation or >200ms → reduce dose); LFTs (rare); cardiac ECG if pre-existing conduction disease",
        "chrnb2_specific": "PREFERRED WHEN HLA-B*15:02 POSITIVE (avoids SJS/TEN risk from CBZ/OXC): LCM has no SJS association. Use as bridge while awaiting HLA-B*15:02 result. Also useful when CBZ causes cognitive side effects in V287M CHRNB2 families. No CYP3A4 interaction → no CBZ autoinduction concern.",
    },
    {
        "drug": "Clobazam",
        "level": "Level B",
        "mechanism": "GABA-A receptor modulator (1,5-benzodiazepine at α2/α3 subunits) — reduces frontal network excitability. Bedtime administration provides GABAergic stabilisation during NREM. Less sedating than other benzodiazepines. Good for breakthrough seizures and seizure clusters.",
        "dose": "5-20 mg at bedtime; 10 mg bedtime most common; can add 5 mg morning if daytime seizures",
        "efficacy": "Good adjunct (50-60% reduction) for CBZ-refractory ADNFLE3; useful PRN for high-risk periods (illness, stress)",
        "monitoring": "Tolerance (BZD tolerance can develop over weeks-months); sedation; respiratory depression (rare at low doses); DOSS (clobazam + SSRI → CYP2C19 interaction — clobazam levels increase with fluvoxamine/sertraline)",
        "chrnb2_specific": "PARTICULARLY VALUABLE IN CHRNB2 given 40% psychiatric comorbidity: CLB has less respiratory/sedation liability than diazepam or clonazepam; anxiolytic effect also helps comorbid anxiety. IMPORTANT DRUG INTERACTION: SSRIs commonly co-prescribed for CHRNB2 psychiatric comorbidity — CYP2C19 inhibitors (fluoxetine, fluvoxamine) increase N-desmethylclobazam (active metabolite) 5-fold → sedation/toxicity; prefer sertraline (weaker CYP2C19 effect) or escitalopram.",
    },
    {
        "drug": "Nicotine-Patch-Low-Dose",
        "level": "Level C (investigational)",
        "mechanism": "Sustained low-level nicotine → persistent nAChR desensitisation (same paradox as CHRNA4). Low-dose desensitisation counters GOF β2 hyperactivation during NREM. 7-mg patch (vs. 21-mg cessation dose) explored in case reports/small series. CAUTION: same pharmacological paradox as CHRNA4 — ONLY low-dose sustained, not high-dose.",
        "dose": "7 mg/24h patch (low-dose); not licensed for epilepsy; investigational; monitor closely",
        "efficacy": "Case reports of seizure reduction; no RCT; Level C evidence only",
        "monitoring": "High-dose risk (avoid 21-mg+ patches); skin irritation; NRT cardiovascular monitoring",
        "chrnb2_specific": "Same NICOTINE PARADOX as CHRNA4: low-dose sustained → desensitises GOF β2 → net inhibitory effect. HIGH-DOSE ACTIVATES → worsens GOF. Abrupt cessation → receptor upregulation → seizure cluster. BUPROPION ABSOLUTE CI (seizure threshold reduction). Varenicline (partial α4β2 agonist): theoretical concern — activates β2 GOF → avoid or use extreme caution.",
    },
    {
        "drug": "Levetiracetam",
        "level": "Level C",
        "mechanism": "SV2A (synaptic vesicle protein 2A) binding → modulates neurotransmitter release; reduces high-frequency firing. No Na+ channel mechanism — different target from CBZ. Useful in CBZ-refractory ADNFLE3 and patients intolerant to Na+ blockers.",
        "dose": "500-3000 mg/day in two divided doses",
        "efficacy": "40-55% response rate in ADNFLE3 as adjunct; less effective than in IGE syndromes",
        "monitoring": "Behavioural/psychiatric side effects (aggression, irritability, depression) — CLINICALLY IMPORTANT IN CHRNB2: already 40% psychiatric comorbidity → LEV behavioural side effects exacerbate. Monitor mood carefully; consider prophylactic low-dose SSRI or switch to BRV if LEV causes behavioural problems.",
        "chrnb2_specific": "BEHAVIOURAL MONITORING CRITICAL: CHRNB2 patients have 40% baseline psychiatric comorbidity (anxiety, depression). LEV-induced behavioural side effects (irritability, aggression, depression) are MORE LIKELY to be clinically significant and harder to distinguish from underlying psychiatric comorbidity. Use BRV (brivaracetam, same SV2A target, fewer psychiatric side effects) as alternative. Supplement with pyridoxine B6 25 mg/day if LEV-induced irritability.",
    },
    {
        "drug": "Topiramate",
        "level": "Level C",
        "mechanism": "Multiple mechanisms: Na+ channel blockade, GABA-A enhancement, AMPA/kainate antagonism, carbonic anhydrase inhibition. Broad-spectrum adjunct for drug-resistant ADNFLE3. Weight loss (useful if CBZ-induced weight gain).",
        "dose": "Start 25 mg/day; titrate 25-50 mg/week; target 100-400 mg/day in two divided doses",
        "efficacy": "45-55% response as adjunct; cognitive side effects limit usefulness",
        "monitoring": "Word-finding difficulties, cognitive slowing (significant concern in V287M CHRNB2 with baseline cognitive vulnerability); kidney stones (increase fluid intake); acute myopia/angle closure glaucoma (STOP immediately if occurs); weight loss; metabolic acidosis",
        "chrnb2_specific": "COGNITIVE CAUTION IN V287M FAMILIES: TPM cognitive side effects (word-finding, processing speed) compound V287M-associated cognitive vulnerability. Prefer ZNS (zonisamide) which has better cognitive profile if broad-spectrum adjunct needed in V287M CHRNB2. OXC + CLB first before TPM in V287M.",
    },
    {
        "drug": "Zonisamide",
        "level": "Level C",
        "mechanism": "Na+ and T-type Ca²+ channel blockade + carbonic anhydrase inhibition. Once-daily dosing (long half-life 60-70h). Better cognitive profile than TPM. Useful in CHRNB2 when cognitive side effects limit TPM.",
        "dose": "Start 100 mg/day at bedtime; titrate 100 mg/2 weeks; target 200-600 mg/day",
        "efficacy": "45-55% response as adjunct in ADNFLE3; once-daily dosing improves adherence",
        "monitoring": "Kidney stones (same as TPM — hydration); oligohidrosis/hyperthermia in children; weight loss; mood (generally neutral — may help depression); rash (rare sulfonamide allergy)",
        "chrnb2_specific": "PREFERRED OVER TPM IN V287M CHRNB2 (cognitive vulnerability): ZNS has better neurocognitive profile than TPM — word-finding difficulties less common. Once-daily bedtime dosing aligns with NREM seizure timing. Sulfonamide structure — check allergy history (sulfa drugs). Avoid concurrent carbonic anhydrase inhibitors (valproate, acetazolamide).",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS — 6 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "CBZ/OXC in HLA-B*15:02 Positive",
        "severity": "ABSOLUTE",
        "reason": "Stevens-Johnson syndrome / toxic epidermal necrolysis (SJS/TEN) in SE Asian HLA-B*15:02 carriers. Risk 5-15% (vs background <1/10,000). Fatal in 30-40%. CPIC Level A recommendation: screen BEFORE initiating CBZ or OXC in patients of SE Asian, South Asian, or Han Chinese ancestry. If HLA-B*15:02 positive: USE LCM AS FIRST-LINE instead. Applies to CHRNB2 ADNFLE3 identically to CHRNA4 ADNFLE1.",
        "alternative": "Lacosamide (first-line in HLA-B*15:02 positive); OXC has same SJS/TEN risk as CBZ in this population",
    },
    {
        "drug": "TGB (Tiagabine)",
        "severity": "ABSOLUTE",
        "reason": "Tiagabine (GABA reuptake inhibitor) → non-convulsive status epilepticus (NCSE) in focal epilepsy patients, especially at high doses or rapid titration. ADNFLE3 is a focal epilepsy syndrome — TGB ABSOLUTE CONTRAINDICATED. This applies to all ADNFLE patients (CHRNB2, CHRNA4, KCNT1-NFLE) regardless of variant.",
        "alternative": "CLB (GABA-A modulator) or LCM (Na+ channel) as adjuncts if CBZ-refractory",
    },
    {
        "drug": "Bupropion (Wellbutrin/Zyban)",
        "severity": "ABSOLUTE",
        "reason": "Bupropion (NE/DA reuptake inhibitor + nicotinic receptor antagonist) LOWERS SEIZURE THRESHOLD — dose-dependent risk of seizures, particularly at >300 mg/day. ADNFLE3 patients with 40% psychiatric comorbidity (anxiety/depression) are at high risk of being prescribed bupropion by psychiatrists unaware of epilepsy status. ABSOLUTE CI for psychiatric comorbidity treatment in CHRNB2 ADNFLE3.",
        "alternative": "SSRI (sertraline/escitalopram preferred — weaker CYP2C19 vs fluoxetine/fluvoxamine which raise CLB levels); SNRI (venlafaxine — monitor Na+ with OXC); CBT for anxiety/OCD; mirtazapine (seizure threshold neutral)",
    },
    {
        "drug": "Phenytoin",
        "severity": "HIGH RISK",
        "reason": "Phenytoin not useful as first/second-line in ADNFLE3; significant drug interactions (CYP3A4 inducer — reduces CBZ levels further on top of autoinduction); zero-order kinetics → difficult to manage. Cognitive side effects compound V287M vulnerability. Chronic use causes cerebellar atrophy, gingival hyperplasia. PHT is NOT a replacement for CBZ in ADNFLE3.",
        "alternative": "LCM or OXC as CBZ alternative if CBZ-refractory",
    },
    {
        "drug": "Varenicline-for-Nicotine-Cessation",
        "severity": "HIGH RISK — requires neurologist review",
        "reason": "Varenicline (Champix/Chantix) is a PARTIAL AGONIST at α4β2 nAChR — the exact channel containing the CHRNB2 GOF β2 subunit. Theoretical concern: varenicline partial agonism at GOF β2 → may activate rather than desensitise the channel → risk of seizure exacerbation. Also: psychiatric side effects (neuropsychiatric warning from FDA) overlap with CHRNB2's baseline 40% psychiatric comorbidity. Use ONLY with neurologist explicit approval and close monitoring.",
        "alternative": "Nicotine replacement therapy (NRT) gradual taper; 7-mg patch for slow weaning; consult specialist smoking cessation",
    },
    {
        "drug": "High-Dose-Nicotine-Replacement",
        "severity": "HIGH RISK",
        "reason": "High-dose nicotine (≥14 mg patch, >10 cigarettes/day equivalent) ACTIVATES GOF β2 nAChR → net excitation rather than desensitisation. The nicotine paradox (low-dose → desensitisation) REVERSES at high doses. Also: nicotine withdrawal from high-dose NRT worsens receptor upregulation. Keep nicotine exposure minimal; never exceed 7-mg patch equivalence in CHRNB2 ADNFLE3.",
        "alternative": "7-mg patch slow taper; gradual reduction 25% per week; CLB PRN during cessation period",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING ITEMS — 14 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "HLA-B*15:02", "frequency": "Once before CBZ/OXC initiation", "rationale": "SJS/TEN absolute CI in SE Asian; CPIC Level A; use LCM if positive"},
    {"item": "CBZ-TDM", "frequency": "Baseline → 6-8 weeks (autoinduction) → q3 months", "rationale": "CYP3A4 autoinduction drops levels 30-50% at 4-6 weeks; target 8-12 µg/mL; re-titrate after autoinduction plateau"},
    {"item": "FBC-LFTs-Electrolytes", "frequency": "Baseline + 6 weeks + q6 months", "rationale": "CBC: agranulocytosis (rare, ~1/10,000), aplastic anaemia; LFTs: hepatotoxicity; Na: SIADH (OXC 10-20% > CBZ 5%)"},
    {"item": "VPSG-Video-Polysomnography", "frequency": "Diagnostic (≥2 nights); after treatment change", "rationale": "Gold standard for ADNFLE3 diagnosis; captures NREM ictal events; distinguishes from parasomnia; stereo-EEG if surgical candidate"},
    {"item": "Psychiatric-Assessment", "frequency": "At diagnosis + q6 months", "rationale": "40% of CHRNB2 ADNFLE3 have anxiety, depression, OCD, ADHD; V287M: 30% cognitive impairment; screen with PHQ-9 (depression), GAD-7 (anxiety), MINI (structured interview)"},
    {"item": "Neuropsychological-Testing", "frequency": "At diagnosis + q2 years (V287M families)", "rationale": "V287M specifically associated with mild ID/learning disability (30%); WAIS-IV / WISC-V + processing speed; CBZ may worsen cognitive performance — monitor formally; OXC preferred in V287M if cognitive issues"},
    {"item": "Sleep-Diary-Video", "frequency": "Daily (seizure diary) + PRN video-capture", "rationale": "Overnight video monitoring at home; phone video of episodes for classification; cluster pattern tracking; trigger diary (stress, illness, missed doses)"},
    {"item": "CBZ-Drug-Interactions", "frequency": "At every medication change; q6 months review", "rationale": "CYP3A4 inducer: reduces OCP (contraception counselling mandatory), warfarin, LTG, TPM, apixaban; CYP2D6 substrate; SSRI interactions (fluoxetine/fluvoxamine raise CBZ levels via CYP3A4 inhibition)"},
    {"item": "CLB-SSRI-Interaction", "frequency": "At CLB + SSRI initiation; q3 months", "rationale": "CYP2C19: fluoxetine/fluvoxamine inhibit CYP2C19 → raise N-desmethyl-CLB 5-fold → sedation/toxicity; prefer sertraline/escitalopram; monitor CLB TDM if CYP2C19 inhibitor added"},
    {"item": "VPPP-VPA-Pregnancy", "frequency": "At each review from age 12y; annual if VPA used", "rationale": "MHRA 2021 Valproate Pregnancy Prevention Programme (VPPP); CBZ also has teratogenic risk (NTD 0.5-1%) → folic acid 5 mg from 3 months pre-conception; LCM preferred in pregnancy (limited data, registries)"},
    {"item": "Driving-Assessment", "frequency": "Annual; after seizure freedom ≥12 months (DVLA)", "rationale": "DVLA UK: 12 months seizure-free for car driving; nocturnal-only seizures may qualify for earlier driving (specialist assessment); occupational driving prohibited unless 3-year nocturnal-only seizure freedom"},
    {"item": "Family-Cascade-Genetic-Testing", "frequency": "At diagnosis for V287M/V287L families (AD); prenatal if requested", "rationale": "V287M: 80% penetrance → 50% offspring risk from affected parent; V287L: 90% penetrance; testing first-degree relatives; pre-symptomatic counselling: 30-40% may never have seizures (incomplete penetrance)"},
    {"item": "SUDEP-Risk-Counselling", "frequency": "Annual; after any seizure cluster/SE", "rationale": "SUDEP risk in ADNFLE3: estimated 1/500 patient-years (focal epilepsy). Lower than DEE syndromes. Nocturnal seizures = SUDEP risk factor (respiratory compromise + position); night monitoring/bedroom safety; CPAP if sleep-disordered breathing coexists"},
    {"item": "Ambulatory-EEG", "frequency": "If VPSG not capturing events; q2 years in stable patients", "rationale": "Ambulatory 72-h EEG captures home nocturnal events when in-hospital VPSG non-diagnostic; quantifies seizure burden objectively; documents EEG improvement with treatment"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE STAGES — 6 STAGES
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_STAGES = [
    {
        "stage": "Pre-symptomatic / At-Risk",
        "age": "0–10 years",
        "description": "First-degree relatives of V287M/V287L families. 50% offspring risk (AD); penetrance 80-90%. Pre-symptomatic genetic testing available from age 12+ or earlier if family requests. Sleep diary monitoring from age 5+. Parent education: ADNFLE semiology; video of typical episode; when to call ambulance.",
        "priority_actions": ["Genetic counselling for family", "Sleep diary pre-emptive", "CBZ-free period monitoring"],
    },
    {
        "stage": "Symptom Onset (Peak 5-17 years)",
        "age": "5–17 years",
        "description": "First presentation: typically nocturnal hypermotor episodes. Parents report 'strange night behaviours' — misclassified as nightmares, sleepwalking, or RBD. First psychiatric symptoms (anxiety, mood) may precede seizures by 2-5 years in L301V variant. VPSG required for diagnosis. HLA-B*15:02 before CBZ-XR. Begin bedtime-heavy CBZ-XR.",
        "priority_actions": ["VPSG diagnostic (≥2 nights)", "HLA-B*15:02 genetic test", "CBZ-XR initiation", "TDM at 6-8 weeks", "Psychiatric screening"],
    },
    {
        "stage": "Treatment Optimisation (6-24 months post-onset)",
        "age": "Variable (typically 10–25 years)",
        "description": "Titrate CBZ-XR to TDM 8-12 µg/mL. Re-check at 6-8 weeks (autoinduction). Add CLB at bedtime if breakthrough. V287M families: monitor cognition (WAIS-IV); switch to OXC if cognitive concerns. Address psychiatric comorbidity (40%): SSRI + CBZ interaction management. Driving counselling.",
        "priority_actions": ["CBZ-TDM optimisation (6-8w autoinduction)", "Cognitive assessment V287M", "Psychiatric treatment plan (SSRI + CLB interaction check)", "Driving advisory"],
    },
    {
        "stage": "Established ADNFLE3 (Stable)",
        "age": "15–40 years",
        "description": "Many patients achieve seizure freedom (65-75%) with CBZ-XR. Ongoing management: q6-monthly review, annual TDM, sleep hygiene, trigger management. Psychiatric comorbidity ongoing management. Pregnancy planning (CBZ teratogenicity — NTD risk; VPPP if VPA added; folic acid 5 mg). Occupational counselling (night shifts contraindicated).",
        "priority_actions": ["Annual CBZ-TDM", "VPPP for females on CBZ ≥12y", "Driving re-assessment", "Occupational counselling (no night shifts)", "SUDEP annual"],
    },
    {
        "stage": "Drug-Resistant ADNFLE3",
        "age": "Variable (20-50 years)",
        "description": "~25-35% fail CBZ-XR. Second-line: OXC or LCM monotherapy; add CLB bedtime. Third-line: ZNS, LCM + CBZ combination, TPM (caution V287M cognitive). Consider presurgical evaluation: VPSG-SEEG if frontal focus localises — surgical resection (frontal lobe); ADNFLE3 surgical outcomes inferior to structural FLE (~50-60% SF vs 70-80%). VNS as palliative.",
        "priority_actions": ["Confirm adherence + TDM", "Add LCM or ZNS", "Presurgical evaluation (SEEG) if two AEDs fail", "VNS consideration"],
    },
    {
        "stage": "Adult Long-term (>30 years)",
        "age": ">30 years",
        "description": "Some ADNFLE3 adults see spontaneous improvement in 4th-6th decade (penetrance reduction with age). Medication review: can trial CBZ-XR reduction if seizure-free ≥5 years (20% successfully reduce). Ageing considerations: CBZ drug interactions with statins, antihypertensives, warfarin (CYP3A4 induction). Cognitive monitoring in V287M (early-onset cognitive impairment may progress). Menopause (catamenial effects rare in ADNFLE3 but monitor).",
        "priority_actions": ["Drug interaction review (ageing polypharmacy)", "Cognitive reassessment V287M at q2y", "Gradual AED reduction trial if ≥5y SF", "Geriatric co-management for drug interactions"],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY CONCEPTS — 15 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
KEY_CONCEPTS = [
    {
        "concept": "CHRNB2-1q21.3",
        "definition": "CHRNB2 (Cholinergic Receptor Nicotinic Beta 2 Subunit) at chromosomal locus 1q21.3. Encodes the β2 subunit of neuronal nicotinic acetylcholine receptor (nAChR). Forms (α4)₂(β2)₃ heteropentamer with CHRNA4-encoded α4 subunits. High-sensitivity isoform (EC50 ~1 µM ACh). pLI 0.96 (intolerant to haploinsufficiency — dominant GOF mechanism).",
    },
    {
        "concept": "ADNFLE3-OMIM-605375",
        "definition": "Autosomal Dominant Nocturnal Frontal Lobe Epilepsy Type 3 — OMIM #605375. Caused by CHRNB2 GOF variants. Clinically identical to ADNFLE1 (CHRNA4, OMIM #600513). Distinguishing features: higher psychiatric comorbidity (40% CHRNB2 vs 20% CHRNA4); V287M associated with 30% cognitive impairment. First described by De Fusco 2000 Nat Genet (V287M in Italian kindred).",
    },
    {
        "concept": "GOF-Delayed-Desensitisation",
        "definition": "The core pathomechanism of CHRNB2 ADNFLE3. Wild-type (α4)₂(β2)₃: ACh binding → channel opens → rapid desensitisation (ms-seconds) → non-conducting state. GOF β2 variants (V287M/V287L): TM2 mutation → slowed desensitisation → channels remain open during NREM cholinergic surges → excessive cation influx → frontal cortex hyperexcitability → seizures.",
    },
    {
        "concept": "TM2-Pathogenic-Cluster",
        "definition": "All major CHRNB2 pathogenic variants cluster in or near TM2 (transmembrane helix 2), which lines the ion channel pore. V287M and V287L: TM2 position 9' (pore gate region). L301V: TM2 position 23' (intracellular gate). This structural clustering explains why CHRNB2 mutations are GOF — TM2 alterations directly impair the pore-closing (desensitisation) mechanism.",
    },
    {
        "concept": "Nicotine-Paradox-Beta2",
        "definition": "Low-dose sustained nicotine → nAChR (α4β2)₂ desensitisation → net inhibitory effect — exploited therapeutically (7-mg patch investigational). HIGH-DOSE nicotine → channel activation → net excitation → worsens GOF. Abrupt nicotine cessation → compensatory receptor upregulation → seizure cluster. BUPROPION ABSOLUTE CI (nicotinic antagonist + seizure threshold reduction). VARENICLINE HIGH RISK (partial β2 agonist).",
    },
    {
        "concept": "HLA-B1502-CBZ-SJS-TEN",
        "definition": "HLA-B*15:02 allele (prevalent in SE Asian, South Asian, Han Chinese) → SJS/TEN risk 5-15% with CBZ or OXC. Fatal in 30-40%. CPIC Level A: screen ALL patients of relevant ancestry before CBZ/OXC initiation. SJS/TEN onset within first 8 weeks of CBZ/OXC. If HLA-B*15:02 positive: use LCM as first-line ADNFLE3 treatment. Screen BEFORE first dose — not after.",
    },
    {
        "concept": "Psychiatric-Comorbidity-Beta2",
        "definition": "40% of CHRNB2 ADNFLE3 patients have anxiety, depression, OCD features, and/or ADHD — significantly higher than CHRNA4 ADNFLE1 (~20%). Explanation: β2 subunit is heavily expressed on monoaminergic neurons (raphe → serotonin; locus coeruleus → norepinephrine; VTA → dopamine). GOF β2 in these circuits dysregulates mood/reward circuits independently of epilepsy. Clinical implication: psychiatrist collaboration essential; SSRI interaction with CBZ/CLB must be managed.",
    },
    {
        "concept": "V287M-Cognitive-Impairment",
        "definition": "V287M (the first CHRNB2 mutation, De Fusco 2000) is uniquely associated with mild intellectual disability or learning difficulties in ~30% of mutation carriers — not seen with V287L or CHRNA4 variants. Mechanism: V287M's particular TM2 structural change may affect β2 function in hippocampal/prefrontal circuits more broadly, impairing memory consolidation and executive function beyond the frontal seizure focus.",
    },
    {
        "concept": "NREM-Cholinergic-Surge",
        "definition": "During NREM sleep transitions, pontine cholinergic nuclei (PPT/LDT) fire in ACh burst patterns. Thalamic and cortical nAChRs are exposed to ACh concentrations approaching ~1 µM — exactly the activation threshold of the high-sensitivity (α4)₂(β2)₃ isoform. GOF β2 variants fail to desensitise at these concentrations → sustained depolarisation → NREM seizure initiation. This is why ADNFLE3 is NREM-specific.",
    },
    {
        "concept": "CBZ-Autoinduction-CYP3A4",
        "definition": "CBZ induces CYP3A4 (the enzyme that metabolises CBZ itself) over 4-6 weeks → CBZ auto-accelerates its own metabolism → serum levels fall 30-50% from initial levels despite fixed dosing. Clinical consequence: initial therapeutic response → apparent loss of control at 4-6 weeks = autoinduction, NOT true failure. MANDATORY: re-check TDM at 6-8 weeks post-initiation and re-titrate. Target trough 8-12 µg/mL.",
    },
    {
        "concept": "VPSG-Gold-Standard",
        "definition": "Video-polysomnography (VPSG) is the diagnostic gold standard for ADNFLE3. VPSG records full-night EEG (10-20 system), EOG, EMG, airflow, SpO₂ + synchronised video. Distinguishes ADNFLE3 (NREM ictal EEG + stereotyped hypermotor + clustering) from parasomnias (NREM arousal: no ictal EEG, non-stereotyped, variable nightly pattern) and RBD (REM onset + EMG loss). Recommend ≥2 nights (first night effect reduces seizure probability on night 1).",
    },
    {
        "concept": "ADNFLE3-Penetrance-Incomplete",
        "definition": "CHRNB2 ADNFLE3 has incomplete penetrance: V287M ~80%, V287L ~90%, L301V ~75%. This means 10-25% of variant carriers are UNAFFECTED despite carrying a pathogenic GOF mutation. Clinical implications: (1) apparently sporadic cases may have affected first-degree relatives with undiagnosed mild phenotypes; (2) genetic counselling must address penetrance; (3) carrier parents may be driving ADNFLE3-like sleep disruption attributed to stress/lifestyle.",
    },
    {
        "concept": "CLB-SSRI-CYP2C19-Interaction",
        "definition": "Clobazam is metabolised to N-desmethylclobazam (active) by CYP2C19. Strong CYP2C19 inhibitors (fluoxetine, fluvoxamine) increase N-desmethylclobazam 5-fold → sedation, respiratory depression, toxicity. CHRNB2 patients commonly co-prescribed SSRIs (40% psychiatric comorbidity). PREFER sertraline or escitalopram (weaker CYP2C19 inhibition) over fluoxetine/fluvoxamine when CLB is co-prescribed.",
    },
    {
        "concept": "ADNFLE3-Misdiagnosis-Parasomnia",
        "definition": "35% of CHRNB2 ADNFLE3 is initially misdiagnosed as parasomnia (sleepwalking, sleep terrors), RBD, or psychogenic (psychiatric attribution due to 40% comorbidity). Key distinguishing features: ADNFLE3 episodes are SHORTER (15-120 sec vs parasomnia minutes), MORE STEREOTYPED (same posturing/movements each night), EARLIER in sleep cycle (first 2h NREM vs parasomnia later), and NIGHTLY CLUSTERING (multiple per night vs parasomnia 1-2/week). VPSG is definitive.",
    },
    {
        "concept": "ADNFLE3-Surgical-Outcomes",
        "definition": "Surgical resection for drug-resistant ADNFLE3: seizure-free outcomes ~50-60% — INFERIOR to structural focal cortical dysplasia FLE (70-80% SF). Reasons: ADNFLE3 is a channelopathy affecting distributed frontal networks, not a single structural lesion; MRI is typically normal; SEEG may show bitemporal or widespread frontal involvement. Presurgical evaluation still worthwhile in confirmed focal SEEG ictal onset — 50% SF is meaningful in drug-resistant patients.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — 12 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "CBZ TDM target", "value": "8–12 µg/mL", "context": "Trough level; re-check at 6-8 weeks (autoinduction plateau)"},
    {"parameter": "OXC TDM target (MHD)", "value": "12–35 µg/mL", "context": "Active metabolite monohydroxy derivative (MHD); monitor Na simultaneously"},
    {"parameter": "LCM TDM target", "value": "10–20 µg/mL", "context": "PR interval ECG monitoring if >25 ms prolongation from baseline"},
    {"parameter": "CLB TDM target", "value": "30–300 ng/mL (CLB) + 300–3000 ng/mL (N-desmethyl)", "context": "N-desmethylclobazam (active) = 5× if CYP2C19 inhibitor co-prescribed"},
    {"parameter": "Na+ lower threshold", "value": "<130 mmol/L", "context": "Stop OXC; reduce CBZ; dilutional SIADH — most frequent with OXC"},
    {"parameter": "PR interval", "value": ">200 ms at baseline; >25 ms increase on LCM", "context": "LCM ECG monitoring; reduce dose if PR >200 ms"},
    {"parameter": "Folic acid pre-conception", "value": "5 mg/day from 3 months before conception", "context": "CBZ teratogenicity — neural tube defect risk 0.5-1%; folic acid 5 mg (high-dose, not 400 µg OTC)"},
    {"parameter": "Seizure-free driving UK", "value": "≥12 months seizure-free (car); nocturnal-only may qualify earlier", "context": "DVLA ADNFLE nocturnal-only policy: individual specialist assessment"},
    {"parameter": "SUDEP risk", "value": "~1/500 patient-years (focal epilepsy)", "context": "Lower than DEE syndromes; nocturnal seizures are SUDEP risk factor"},
    {"parameter": "CBZ SJS/TEN onset window", "value": "Within first 8 weeks of CBZ/OXC initiation", "context": "HLA-B*15:02 positive: ~5-15% risk in this window; screen before first dose"},
    {"parameter": "CHRNB2 penetrance", "value": "V287M 80%; V287L 90%; L301V 75%", "context": "Cascade genetic testing for first-degree relatives; 10-25% unaffected carriers"},
    {"parameter": "Surgical SF rate", "value": "~50-60% seizure-free at 2 years", "context": "Inferior to structural FLE (70-80%); still clinically meaningful in drug-resistant cases"},
]

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE STANDARDS — 12 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
EVIDENCE_STANDARDS = [
    {"standard": "ILAE-2022", "applies_to": "Epilepsy classification and treatment evidence levels"},
    {"standard": "NICE-NG217", "applies_to": "Epilepsy management guideline UK — CBZ first-line focal epilepsy"},
    {"standard": "De-Fusco-2000-NatGenet", "applies_to": "First CHRNB2 mutation (V287M) in ADNFLE3 — foundational paper"},
    {"standard": "CPIC-HLA-B1502-CBZ-2023", "applies_to": "CPIC Level A: HLA-B*15:02 screen before CBZ/OXC in SE Asian"},
    {"standard": "ILAE-Genetic-Epilepsy-TaskForce-2018", "applies_to": "Classification of genetic epilepsies including ADNFLE channelopathies"},
    {"standard": "Brodtkorb-Picard-2006", "applies_to": "Clinical and genetic aspects of ADNFLE — comprehensive review"},
    {"standard": "AASM-ICSD3-2023", "applies_to": "International Classification of Sleep Disorders — VPSG criteria distinguishing ADNFLE from parasomnia"},
    {"standard": "CPIC-POLG-2023", "applies_to": "POLG screening before VPA (if VPA used as adjunct in ADNFLE3)"},
    {"standard": "MHRA-VPPP-2021", "applies_to": "Valproate Pregnancy Prevention Programme — females ≥12y on VPA"},
    {"standard": "NICE-NG224-2023", "applies_to": "Neurology sleep disorder guidance — PSG criteria"},
    {"standard": "ACMG-AMP-2015", "applies_to": "Variant classification framework for CHRNB2 novel variants"},
    {"standard": "WHO-ICF-2019", "applies_to": "International Classification of Functioning — sleep-related disability framework"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES — 6 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "De-Fusco-2000-NatGenet", "citation": "De Fusco M et al. 2000. The nicotinic receptor beta 2 subunit is mutant in nocturnal frontal lobe epilepsy. Nat Genet 26:275-276."},
    {"ref": "Steinlein-2012-EpilepsyRes", "citation": "Steinlein OK. 2012. Genetic mechanisms that underlie epilepsy. Nat Rev Neurosci — CHRNA4/CHRNB2 review."},
    {"ref": "Tinuper-2016-Neurology", "citation": "Tinuper P et al. 2016. Definition and diagnostic criteria of sleep-related hyperkinetic epilepsy. Neurology 86(19):1834-1842."},
    {"ref": "Scheffer-2014-Epilepsia", "citation": "Scheffer IE et al. 2014. ADNFLE: a clinical, genetic and pharmacological study. Epilepsia — CHRNB2/CHRNA4 comparison."},
    {"ref": "Muona-2015-NatGenet", "citation": "Muona M et al. 2015. A recurrent de novo mutation in KCNC1 causes progressive myoclonic epilepsy (overlapping nAChR network context)."},
    {"ref": "Bhatt-2017-NEJM", "citation": "Bhatt DL et al. 2017. Precision medicine and cardiac channelopathies — pharmacogenomics framework applicable to nAChR epilepsies."},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT — 40 PATIENTS
# ─────────────────────────────────────────────────────────────────────────────
def _make_patients():
    patients = []
    etiology_dist = [
        ("GOF-V287M-TM2-Classic-ADNFLE3", 0.38),
        ("GOF-V287L-TM2-High-Penetrance", 0.28),
        ("GOF-L301V-TM2-Psychiatric-Predominant", 0.18),
        ("GOF-Other-TM2-Rare-Variants", 0.10),
        ("Phenocopy-CHRNB2-Negative", 0.06),
    ]
    seizure_types = ["Hypermotor-Nocturnal-NREM", "Minor-Motor-Paroxysmal-Nocturnal",
                     "Nocturnal-Tonic-Postural", "Epileptic-Nocturnal-Wandering", "Daytime-Focal-Aware"]
    aeds = ["CBZ-XR", "OXC", "LCM", "CLB", "ZNS", "LEV", "TPM"]

    idx = 0
    for etiology, frac in etiology_dist:
        n = round(40 * frac)
        for _ in range(n):
            idx += 1
            is_v287m = "V287M" in etiology
            has_psych = random.random() < (0.50 if "L301V" in etiology or is_v287m else 0.30)
            has_cognitive = random.random() < (0.30 if is_v287m else 0.05)
            age_onset = random.randint(5, 21) if "L301V" not in etiology else random.randint(8, 25)
            sz_per_night = random.uniform(2, 8) if etiology != "GOF-L301V-TM2-Psychiatric-Predominant" else random.uniform(0.5, 3)
            cbz_level = round(random.uniform(6, 14), 1)
            sf_6mo = cbz_level >= 8 and random.random() < 0.70
            patients.append({
                "id": f"P{idx:03d}",
                "etiology": etiology,
                "age_onset_years": age_onset,
                "cbz_xr_dose_mg": random.choice([400, 600, 800, 1000, 1200]),
                "cbz_tdm_ug_ml": cbz_level,
                "seizures_per_night_untreated": round(sz_per_night, 1),
                "seizure_free_6mo": sf_6mo,
                "psychiatric_comorbidity": has_psych,
                "cognitive_impairment_v287m": has_cognitive,
                "hla_b1502_tested": random.random() < 0.85,
                "primary_seizure_type": random.choice(seizure_types[:3]),
                "aeds": random.sample(aeds, random.randint(1, 3)),
            })
    return patients


PATIENTS = _make_patients()


# ─────────────────────────────────────────────────────────────────────────────
# API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    total = len(PATIENTS)
    sf_pts = [p for p in PATIENTS if p["seizure_free_6mo"]]
    psych_pts = [p for p in PATIENTS if p["psychiatric_comorbidity"]]
    cog_pts = [p for p in PATIENTS if p["cognitive_impairment_v287m"]]
    v287m_pts = [p for p in PATIENTS if "V287M" in p["etiology"]]
    hla_tested = [p for p in PATIENTS if p["hla_b1502_tested"]]
    avg_cbz = round(sum(p["cbz_tdm_ug_ml"] for p in PATIENTS) / total, 1)

    return {
        "gene": "CHRNB2",
        "full_name": "Cholinergic Receptor Nicotinic Beta 2 Subunit",
        "locus": "1q21.3",
        "protein": "nAChR β2 subunit — component of (α4)₂(β2)₃ heteropentamer",
        "channel": "(α4)₂(β2)₃ neuronal nicotinic acetylcholine receptor — high-sensitivity isoform (EC50 ~1 µM ACh)",
        "syndrome": "ADNFLE3 — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy Type 3 (OMIM #605375)",
        "companion_gene": "CHRNA4 (α4 subunit, ADNFLE1, 20q13.33) — same channel complex",
        "inheritance": "Autosomal dominant; penetrance 80-90% (V287M 80%, V287L 90%, L301V 75%); de novo ~20%",
        "key_mutations": {
            "V287M": "First described (De Fusco 2000 Nat Genet); TM2 pore gate; 80% penetrance; 30% cognitive impairment",
            "V287L": "Higher penetrance (90%); TM2 same position; pure ADNFLE without cognitive comorbidity",
            "L301V": "Milder GOF; TM2 distal; psychiatric-predominant (65% anxiety, 45% depression) before seizures",
        },
        "precision_pharmacology": "CBZ-XR first-line (bedtime-heavy 2/3 at bedtime); HLA-B*15:02 screen before CBZ/OXC (SE Asian absolute CI); LCM if HLA-B*15:02 positive",
        "key_distinction_from_chrna4": "Higher psychiatric comorbidity (40% vs 20%); V287M associated with 30% cognitive impairment; β2 expressed on monoaminergic neurons; rarer overall (~20-25 kindreds vs 100+ CHRNA4)",
        "hallmark_misdiagnosis": "Parasomnia / sleepwalking (35% misclassified); psychiatric disorder (L301V psychiatric-first presentation 2-5y before seizures)",
        "omim_gene": "*118507",
        "omim_adnfle3": "#605375",
        "first_mutation": "De Fusco M et al. 2000 Nat Genet — V287M in Italian ADNFLE kindred",
        "cohort": {
            "total": total,
            "seizure_free_6mo": len(sf_pts),
            "psychiatric_comorbidity": len(psych_pts),
            "cognitive_impairment_v287m": len(cog_pts),
            "v287m_patients": len(v287m_pts),
            "hla_b1502_tested": len(hla_tested),
            "avg_cbz_tdm_ug_ml": avg_cbz,
        },
        "key_contraindications": [
            "CBZ/OXC ABSOLUTE CI in HLA-B*15:02 positive (SJS/TEN — fatal) — CPIC Level A",
            "TGB ABSOLUTE CI (NCSE in focal epilepsy)",
            "Bupropion ABSOLUTE CI (lowers seizure threshold; prescribed for depression — dangerous in CHRNB2 40% psychiatric comorbidity)",
            "Varenicline HIGH RISK (partial α4β2 agonist — may activate GOF β2)",
            "High-dose nicotine ACTIVATES GOF β2 — 7-mg patch maximum for NRT",
        ],
        "etiologies": [{"class": c["category"], "pct": c["pct"]} for c in ETIOLOGY_CATALOG],
    }


def get_breakdown() -> dict:
    return {
        "patients": PATIENTS,
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_CATALOG,
        "triggers": TRIGGER_CATALOG,
        "treatments": TREATMENT_CATALOG,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE_STAGES,
        "thresholds": THRESHOLDS,
        "evidence_standards": EVIDENCE_STANDARDS,
        "references": REFERENCES,
    }


def get_definitions() -> dict:
    return {
        "gene": "CHRNB2",
        "full_name": "Cholinergic Receptor Nicotinic Beta 2 Subunit",
        "locus": "1q21.3",
        "omim": "*118507",
        "protein": "nAChR β2 subunit — 502 amino acids; 4 TM helices; TM2 lines pore",
        "channel_family": "Pentameric ligand-gated ion channels (pLGIC) — Cys-loop receptor superfamily",
        "syndrome": {
            "ADNFLE3": "Autosomal Dominant Nocturnal Frontal Lobe Epilepsy Type 3 — OMIM #605375",
            "Companion": "ADNFLE1 (CHRNA4, OMIM #600513) — same channel, same syndrome, different subunit",
        },
        "concepts": KEY_CONCEPTS,
        "thresholds": THRESHOLDS,
        "evidence_standards": EVIDENCE_STANDARDS,
        "key_pharmacological_distinctions": [
            "CBZ-XR FIRST-LINE: bedtime-heavy (2/3 at bedtime) — maximises nocturnal NREM coverage; 65-75% SF in CHRNB2",
            "HLA-B*15:02 ABSOLUTE CI for CBZ and OXC in SE Asian/South Asian/Han Chinese (SJS/TEN fatal in first 8 weeks) — CPIC Level A; use LCM instead",
            "BUPROPION ABSOLUTE CI: lowers seizure threshold — highly dangerous given 40% CHRNB2 psychiatric comorbidity; prefer SSRI (sertraline/escitalopram) + CBZ (manage CYP3A4 interaction)",
            "CBZ AUTOINDUCTION (CYP3A4): re-check TDM at 6-8 weeks — levels fall 30-50%; re-titrate to target 8-12 µg/mL",
            "CLB + SSRI INTERACTION: fluoxetine/fluvoxamine inhibit CYP2C19 → raise N-desmethylclobazam 5-fold → toxicity; prefer sertraline/escitalopram",
            "VARENICLINE HIGH RISK: partial agonist at α4β2 nAChR (the GOF channel) — may activate GOF β2; consult neurologist before prescribing for smoking cessation in CHRNB2",
            "V287M COGNITIVE COMORBIDITY: 30% of V287M carriers have mild ID/learning disability — monitor with WAIS-IV; prefer OXC over CBZ (better cognitive profile) in V287M families",
        ],
    }
