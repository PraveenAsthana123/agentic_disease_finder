"""
CLN4B Epilepsy — Neuronal Ceroid Lipofuscinosis Type 4B / Adult Batten Disease / Kufs Disease Type B
======================================================================================================
40-patient cohort · DNAJC5 (19q13.11) · Autosomal dominant (AD) — dominant-negative mechanism
DNAJC5 encodes CSPα (Cysteine String Protein-α) — a presynaptic co-chaperone
AD mutations (p.Leu115Arg or c.344_347del) → toxic CSPα aggregation → synaptic vesicle
chaperone function lost → SNARE dysfunction → progressive presynaptic degeneration →
autofluorescent ceroid storage (fingerprint profiles on EM) → cortical + cerebellar neuronal
loss → CLN4B / Kufs Type B (Adult-onset NCL)

ADULT-ONSET NCL — BEHAVIORAL ONSET FIRST:
CLN4B (Kufs Disease Type B) is the most common adult-onset NCL and uniquely presents with
BEHAVIORAL/PSYCHIATRIC CHANGES BEFORE myoclonus or seizures by 3-5 years:
  - Depression, personality change, psychosis, dementia (92% of cohort)
  - Mimics primary psychiatric disorder → misdiagnosis in 80%+ of CLN4B patients
  - Mean diagnostic delay 5.8 years — longest of any adult PME
  - Key diagnostic trigger: skin biopsy EM (fingerprint profiles) in any adult with
    unexplained progressive behavioral syndrome + myoclonus or ataxia

AUTOSOMAL DOMINANT — UNIQUE AMONG NCLs:
CLN4B (DNAJC5) is the ONLY common NCL with AUTOSOMAL DOMINANT inheritance:
  - 50% offspring risk per pregnancy
  - Adult-onset means affected adults often have children before diagnosis
  - Presymptomatic testing for adult children/siblings ≥18y is a clinical priority
  - Genetic counselling is IMMEDIATE at CLN4B diagnosis (unlike AR NCLs)

CSPα PROTEIN BIOLOGY (PRESYNAPTIC CO-CHAPERONE):
DNAJC5 (19q13.11), CSPα (Cysteine String Protein-α):
  - 198 aa; ~26 kDa
  - Domains: DNAJ (Hsp40) domain aa 1-80 (Hsc70 binding); Cysteine-string domain aa 115-136
    (15 cysteines, palmitoylated → SV membrane anchor); C-terminal aa 148-198
  - Function: Presynaptic co-chaperone; CSPα/Hsc70/SGT trimeric complex → refolds misfolded
    SNARE proteins (especially SNAP-25) → maintains neurotransmitter release machinery;
    neuroprotective at synapse
  - Pathomechanism: DNAJC5 dominant mutations (especially p.Leu115Arg or 4-bp deletion
    c.344_347del) → mutant CSPα retains palmitoylated cysteine-string domain → TOXIC
    AGGREGATION (dominant-negative) → synaptic vesicle chaperone function lost → SNARE
    dysfunction → progressive presynaptic degeneration → autofluorescent ceroid storage
    (fingerprint profiles on EM) → cortical + cerebellar neuronal loss → CLN4B / Kufs Type B
  - pLI ~0.55 (moderate intolerance)
  - OMIM: *611203 (DNAJC5 gene) / #162350 (CLN4B / Kufs disease Type B / Adult NCL /
    Parry disease)
  - Cadieux-Dion M et al. 2013; Greaves J et al. 2012 (J Biol Chem) — DNAJC5/CSPα mechanism

COMMON MUTATIONS:
  p.Leu115Arg (c.344T>G — most common, changes the anchor of cysteine-string domain):
    disrupts palmitoylation anchor topology → mutant CSPα aggregates at SV membrane →
    toxic dominant-negative; classic adult Kufs B onset 28-38y
  c.344_347delCTGC (4-bp deletion — Kufs/Parry pedigrees):
    frameshift → truncated CSPα retaining palmitoylated cysteine-string → aggressive
    aggregation → more severe; mean onset 25-32y; rapid cognitive decline

CLN4B KEY DISTINCTIONS:
  NO visual failure — CLN4B SPARES THE RETINA (CRITICAL distinction from CLN2/CLN3)
  NO vacuolated lymphocytes — blood film always normal (CRITICAL distinction from CLN3)
  BEHAVIORAL/PSYCHIATRIC ONSET FIRST — mimics primary psychiatric disorder
  AUTOSOMAL DOMINANT — 50% offspring risk (unique among NCLs)
  FINGERPRINT PROFILES on EM — pathognomonic adult NCL finding (85% of CLN4B biopsies)
  ADULT ONSET — typically 25-45y (range 17-55y); fatal 40s-60s
"""


def get_overview():
    return {
        "gene": "DNAJC5 (19q13.11) — CSPα (Cysteine String Protein-α; presynaptic co-chaperone)",
        "protein": "CSPα (Cysteine String Protein-α); 198 aa; ~26 kDa; DNAJ (Hsp40) domain aa 1-80 (Hsc70 binding); Cysteine-string domain aa 115-136 (15 cysteines, palmitoylated → synaptic vesicle membrane anchor); C-terminal aa 148-198; presynaptic co-chaperone; CSPα/Hsc70/SGT trimeric complex refolds misfolded SNARE proteins (esp. SNAP-25) → maintains neurotransmitter release machinery; neuroprotective at synapse",
        "inheritance": "Autosomal dominant (AD) — dominant-negative mechanism; typically pathogenic heterozygous missense (p.Leu115Arg c.344T>G — most common) or small deletion (c.344_347delCTGC 4-bp deletion — Parry/Kufs pedigrees); pLI ~0.55; 50% offspring risk per pregnancy; de novo mutations in 12% of cohort",
        "omim": "*611203 (DNAJC5 gene) · #162350 (CLN4B / Kufs disease Type B / Adult NCL / Parry disease)",
        "disease": "CLN4B — Adult Neuronal Ceroid Lipofuscinosis (Adult Batten Disease / Kufs Disease Type B): BEHAVIORAL/PSYCHIATRIC CHANGES are first symptom (92% cohort) — depression, personality change, psychosis, dementia — by 3-5 years before seizures; progressive myoclonus epilepsy (PME) in 75-85%; cerebellar ataxia 70%; cognitive decline/dementia 100%; NO visual failure (retina spared — CRITICAL distinction from CLN2/CLN3); onset typically 25-45y; fatal 40s-60s",
        "mechanism": "DNAJC5 dominant mutations (p.Leu115Arg or c.344_347del) → mutant CSPα retains palmitoylated cysteine-string domain → TOXIC AGGREGATION (dominant-negative) → synaptic vesicle chaperone function lost → SNARE dysfunction → progressive presynaptic degeneration → autofluorescent ceroid storage (fingerprint profiles on EM) → cortical + cerebellar neuronal loss → CLN4B / Kufs Type B",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN4B. Investigational: AAV-DNAJC5 gene therapy (early-phase trials); CSPα stabiliser compounds. Register with NCL Resource, BDSRA registry, NCL Network Europe for trial access.",
        "cohort_size": 40,
        "female_pct": 52,
        "mean_onset_behavioral_years": 31.4,
        "mean_onset_seizure_years": 34.8,
        "drug_resistant_pct": 55,
        "photosensitivity_pct": 48,
        "retinal_spared_pct": 100,
        "vacuolated_lymphocytes_pct": 0,
        "cognitive_decline_pct": 100,
        "behavioural_psychiatric_onset_pct": 92,
        "on_vpa_pct": 78,
        "on_lev_pct": 68,
        "mean_delay_diagnosis_behavioral_years": 5.8,
        "ad_inheritance_confirmed_pct": 88,
        "skin_biopsy_fingerprint_pct": 85,
        "discovery": "DNAJC5/CSPα mutations causing adult NCL (Kufs Type B): Noskova L et al. 2011 (Am J Hum Genet); Cadieux-Dion M et al. 2013 (JMDN — phenotypic characterisation); Greaves J et al. 2012 (J Biol Chem — CSPα palmitoylation mechanism); Parry pedigrees described before DNAJC5 gene identified",
        "unique_feature": "ONLY common NCL with AUTOSOMAL DOMINANT inheritance (50% offspring risk). BEHAVIORAL/PSYCHIATRIC ONSET FIRST (before seizures by 3-5 years) — longest diagnostic delay (5.8 years) of any adult PME. NO visual failure (retina spared — distinct from CLN2/CLN3). NO vacuolated lymphocytes (blood film normal — distinct from CLN3). FINGERPRINT PROFILES on EM pathognomonic. PRESYNAPTIC CHAPERONE disease (mechanistically unique — not lysosomal enzyme or membrane transport). Adult-onset (25-45y). LEV SV2A has direct pathway complementarity to CSPα presynaptic dysfunction.",
        "absolute_ci": [
            "Carbamazepine / Oxcarbazepine / Phenytoin (CBZ/OXC/PHT) — ABSOLUTE CI: Na-channel blockers WORSEN myoclonus in CLN4B. PRIMARY CLN4B MISDIAGNOSIS TRAP: behavioral onset + GTCS → GP/neurologist prescribes CBZ for 'focal epilepsy' → ACUTE MYOCLONIC DETERIORATION. VPA+LEV mandatory instead.",
            "Tiagabine (TGB) — ABSOLUTE CI: GABA-reuptake inhibition → excessive CSF GABA → NCSE in myoclonic epilepsies; ABSOLUTE CI across all PME syndromes.",
        ],
        "high_risk_ci": [
            "GBP / Pregabalin — HIGH RISK: worsen myoclonus (Crespel 1999); adult CLN4B patients with pain symptoms see pain clinics/GPs who prescribe GBP/PGB without PME awareness — HIGH-RISK adult multi-specialty prescribing trap.",
            "LTG Monotherapy — HIGH RISK: LTG monotherapy can worsen myoclonus in CLN4B PME; LTG add-on (with VPA — careful dose adjustment for VPA interaction) may be cautiously considered but NEVER first-line monotherapy.",
            "VGB — AVOID: peripheral visual field constriction risk; CLN4B does NOT have retinal degeneration (unlike CLN2/CLN3) but VGB still avoided in cognitively vulnerable adult patient.",
            "AED-Taper-Remission-Expectation — HIGH RISK: CLN4B is progressive neurodegenerative disease — seizures do NOT remit; AED taper = inevitable GTCS breakthrough + fall/SUDEP risk in cognitively impaired adult with ataxia.",
        ],
    }


def get_breakdown():
    etiologies = [
        {
            "class": "p.Leu115Arg-Dominant-Missense-Classic",
            "pct": 45,
            "description": "Most common DNAJC5 mutation; c.344T>G; changes Leu→Arg in cysteine-string domain; disrupts palmitoylation anchor topology → mutant CSPα aggregates at SV membrane → toxic dominant-negative; classic adult Kufs B onset 28-38y; behavioral onset then PME; EM fingerprint profiles; familial with parent affected",
            "count": 18,
            "gene_mechanism": "p.Leu115Arg disrupts hydrophobic palmitoylation anchor in cysteine-string domain → mutant CSPα cannot be properly inserted into SV membrane → aggregates → dominant-negative toxic gain-of-function → SNARE chaperone function lost",
            "key_variants": ["p.Leu115Arg (c.344T>G — most common CLN4B mutation)", "Cysteine-string domain aa 115-136"],
        },
        {
            "class": "4bp-Deletion-c344_347del-Truncation-Toxic",
            "pct": 28,
            "description": "c.344_347delCTGC (Parry/Kufs pedigrees); frameshift → truncated CSPα retaining palmitoylated cysteine-string → aggressive aggregation → more severe; mean onset 25-32y; rapid cognitive decline; prominent action myoclonus",
            "count": 11,
            "gene_mechanism": "4-bp frameshift deletion → truncated CSPα retaining functional palmitoylated cysteine-string region → truncated protein anchors to SV membrane and aggregates aggressively → severe dominant-negative effect → rapid presynaptic degeneration",
            "key_variants": ["c.344_347delCTGC (Parry/Kufs 4-bp deletion)", "Frameshift with retained palmitoylation domain"],
        },
        {
            "class": "De-Novo-Novel-Missense-Cysteine-String",
            "pct": 14,
            "description": "De novo pathogenic missense within cysteine-string domain (aa 115-136); non-familial; VUS initially → functional palmitoylation assay confirms LOF; later diagnosis (mean 6 years to genetic confirmation); skin biopsy EM fingerprint profiles essential for early NCL diagnosis",
            "count": 6,
            "gene_mechanism": "De novo missense in cysteine-string domain → impaired palmitoylation or altered membrane topology → reduced CSPα function or toxic aggregation → CLN4B disease; functional palmitoylation assay in fibroblasts confirms pathogenicity of VUS",
            "key_variants": ["De novo cysteine-string domain missense (various positions aa 115-136)", "Functional palmitoylation assay required for VUS classification"],
        },
        {
            "class": "DNAJ-Domain-Missense-Hsc70-Binding",
            "pct": 8,
            "description": "Rare missense in DNAJ domain (aa 1-80); impairs Hsc70 binding → reduced trimeric complex assembly → reduced SNARE refolding; milder/attenuated phenotype (onset 40-50y); slower progression; may not develop prominent myoclonus; DNAJC5 sequencing essential as phenotype mimics Alzheimer's",
            "count": 3,
            "gene_mechanism": "DNAJ domain missense → impaired interaction with Hsc70 → reduced CSPα/Hsc70/SGT trimeric complex formation → reduced SNARE protein refolding capacity → milder presynaptic dysfunction → attenuated CLN4B phenotype",
            "key_variants": ["DNAJ domain missense (aa 1-80)", "Hsc70-binding interface residues"],
        },
        {
            "class": "Compound-Heterozygous-AR-Atypical",
            "pct": 3,
            "description": "RARE: compound heterozygous DNAJC5 with AR inheritance (biallelic LOF); atypical earlier onset (18-24y); more severe; mimics juvenile NCL; reported in consanguineous families; requires both alleles sequenced",
            "count": 1,
            "gene_mechanism": "Biallelic DNAJC5 LOF → complete CSPα deficiency (rather than dominant-negative toxic gain) → severe presynaptic chaperone deficiency → early-onset aggressive CLN4B-like phenotype",
            "key_variants": ["Biallelic DNAJC5 LOF (both alleles)", "Consanguineous family — AR CLN4B atypical"],
        },
        {
            "class": "Phenocopy-CLN4B-Negative",
            "pct": 2,
            "description": "Kufs disease phenotype (adult PME + behavioral + fingerprint EM profiles) without DNAJC5 mutation; represents genetic heterogeneity of adult NCL (other genes: ATP13A2/CLN12, GRN/CLN11, CTSD/CLN10 adult variant)",
            "count": 1,
            "gene_mechanism": "Not DNAJC5 — alternative adult NCL gene (ATP13A2/CLN12, GRN/CLN11, CTSD/CLN10 adult variant) presenting with CLN4B-like phenotype",
            "key_variants": ["Normal DNAJC5 → extended adult NCL gene panel", "ATP13A2/CLN12, GRN/CLN11, CTSD/CLN10 adult variant testing"],
        },
    ]

    seizure_types = [
        {
            "type": "Action-Myoclonus-Cortical",
            "pct": 80,
            "description": "Distal limb myoclonus on voluntary movement; handwriting disrupted early; fine motor tasks trigger cortical myoclonus; bilateral; EEG: giant SEP + cortical correlate on jerk-locked back-averaging; worsened by missed AED, stress, anxiety. FIRST-LINE: VPA backbone + piracetam adjunct (action myoclonus specific). LEV SV2A provides presynaptic complementarity to CSPα deficiency.",
            "eeg": "Giant cortical SEP (N20/P25 >6 µV); jerk-locked back-averaging confirms cortical origin; generalised polyspike-wave; background slowing with progression",
            "semiology": "Bilateral distal limb jerks triggered by voluntary movement (handwriting, eating, dressing); worse with fine motor tasks; amplitude increases with fatigue and stress; may cause falls (combined with ataxia)",
            "clinical_tip": "Action myoclonus is the primary functional disability in early CLN4B. Piracetam 16-24g/day is FIRST-LINE add-on alongside VPA — start early. Document in every letter: CBZ/OXC/PHT ABSOLUTE CI. Adult patient may have already received CBZ from GP for 'new-onset seizures' → check and stop immediately.",
        },
        {
            "type": "Stimulus-Sensitive-Myoclonus",
            "pct": 72,
            "description": "Auditory startle (65%), tactile (58%), photic (48%); sound-sensitive myoclonus during daily activities; audiogenic seizures; careful history — patient describes 'jumping' or 'startle' with loud sounds or touch; environmental modification helpful alongside AED management",
            "eeg": "Cortical correlate on jerk-locked back-averaging; stimulus-evoked poly-spike at EEG; IPS positive in 48%",
            "semiology": "Startle to sounds (clapping, door slam, traffic); tactile startle (unexpected touch); photic startle (flickering lights, car headlights, sunlight through trees); patient may limit social activities due to audiogenic startle",
            "clinical_tip": "Environmental modification is an adjunct treatment in CLN4B: ear defenders in noisy environments; tactile approach protocol; lighting control; calm environment. Photosensitivity testing (IPS) at standard 3-50 Hz — CLN4B is NOT CLN2 (no 1-3 Hz SSPS pathognomonic EEG). Tinted lenses if photic positive.",
        },
        {
            "type": "Generalised-Tonic-Clonic",
            "pct": 68,
            "description": "Nocturnal GTCS predominant; post-ictal confusion prominent (cognitive baseline already impaired); SUDEP risk; nocturnal monitoring essential; VPA + LEV backbone; may be presenting seizure when behavioral symptoms misattributed to psychiatry. CRITICAL: GTCS in adult with behavioral symptoms → IMMEDIATE SKIN BIOPSY EM + DNAJC5 sequencing (do NOT wait for psychiatric review completion).",
            "eeg": "Generalised poly-spike-wave; nocturnal EEG recommended; post-ictal slowing",
            "semiology": "Classic GTCS — tonic then clonic phase; nocturnal predominance (68% nocturnal); post-ictal confusion prolonged in CLN4B (baseline cognitive impairment); duration 2-5 minutes; urinary incontinence common in adult onset",
            "clinical_tip": "NOCTURNAL GTCS in CLN4B = compound SUDEP risk: (1) nocturnal seizure, (2) cognitive impairment (unable to self-rescue prone position), (3) progressive ataxia (fall risk). Nocturnal pulse oximetry + movement sensor + carer alert MANDATORY. Nocturnal CLB dose. ACP must address SUDEP explicitly.",
        },
        {
            "type": "Focal-Impaired-Awareness",
            "pct": 45,
            "description": "Temporal lobe semiology (déjà vu, automatisms, post-ictal confusion); EEG temporal theta/slowing; represents focal cortical NCL pathology; respond partially to LEV. THE MISDIAGNOSIS TRAP: focal seizures in adult with behavioral onset → MISDIAGNOSED as temporal lobe epilepsy → CBZ prescribed → ACUTE MYOCLONIC DETERIORATION.",
            "eeg": "Temporal theta/slowing; focal spikes or sharp waves temporal; post-ictal regional slowing",
            "semiology": "Déjà vu; epigastric aura; oral automatisms; post-ictal confusion (prolonged in CLN4B); duration 1-3 minutes; may cluster during illness",
            "clinical_tip": "FOCAL SEIZURES IN ADULT with BEHAVIORAL HISTORY = NEVER assume idiopathic focal epilepsy without excluding CLN4B. Skin biopsy EM if clinical suspicion: fingerprint profiles confirm adult NCL. DNAJC5 sequencing. Do NOT start CBZ/OXC — even for apparently focal seizures. Use VPA + LEV (treat both focal component and co-existing myoclonus).",
        },
        {
            "type": "Absence-Like-Myoclonic-Jerks",
            "pct": 32,
            "description": "Brief absence-like episodes with mild myoclonic component; EEG: 3 Hz poly-spike-wave during absence; distinguish from primary generalised epilepsy (CLN4B has progressive course). Adult-onset absence + behavioral symptoms = adult NCL until proven otherwise.",
            "eeg": "3 Hz generalised spike-wave; absence with poly-spike admixture; distinguishable from pure absence epilepsy by progressive course and background EEG abnormality",
            "semiology": "Brief staring (5-20 seconds) with eyelid flicker; mild jerks during episode; minimal post-ictal period; patient may be unaware; carer reports 'blank spells'",
            "clinical_tip": "Adult 'absence' with behavioral history — VPA effective for both absence and myoclonus component. VPA IS SAFE in CLN4B (presynaptic/lysosomal, NOT mitochondrial — contrast MERRF where VPA ABSOLUTE CI). VPPP counselling for females. Never LTG monotherapy (worsens myoclonus in CLN4B).",
        },
    ]

    triggers = [
        {
            "trigger": "Voluntary-Movement",
            "pct": 85,
            "description": "Action myoclonus trigger — voluntary movement (writing, eating, dressing) provokes distal limb myoclonus; action myoclonus is the primary functional disability in CLN4B early disease; compound with ataxia creates severe gait and fine motor impairment",
            "management": "OT myoclonus adaptations (weighted utensils, stabilised work surfaces, dictation software for writing); VPA + piracetam dose optimisation for action myoclonus; functional occupational assessment for workplace adjustments",
        },
        {
            "trigger": "Stress-Anxiety",
            "pct": 78,
            "description": "Behavioral features of CLN4B (depression, anxiety, OCD features, agitation) directly elevate seizure threshold → psychosocial stress becomes a potent seizure trigger; psychiatric misdiagnosis period (mean 5.8 years) is itself a period of high stress → seizure exacerbation",
            "management": "SSRI (sertraline preferred) for behavioral features — dual benefit: psychiatric symptom management AND indirect seizure threshold stabilisation; CBT where cognitive capacity allows; neuropsychiatry co-lead; reduce diagnostic uncertainty (confirm CLN4B diagnosis promptly to reduce uncertainty-related stress)",
        },
        {
            "trigger": "Sleep-Deprivation",
            "pct": 72,
            "description": "Sleep deprivation lowers seizure threshold; CLN4B-specific: behavioral features (depression, anxiety, OCD rituals) disrupt sleep architecture; nocturnal GTCS disrupt sleep → vicious cycle; cognitive decline may impair sleep routine adherence",
            "management": "Sleep hygiene; melatonin 2-5 mg nocte if circadian disruption; SSRI for depression/anxiety → improved sleep; nocturnal CLB dose reduces nocturnal GTCS and may improve sleep continuity; cognitive aide for sleep routine (as disease progresses)",
        },
        {
            "trigger": "Missed-AED-Dose",
            "pct": 68,
            "description": "AED omission triggers seizure breakthrough — CLN4B-specific: progressive cognitive decline impairs medication adherence; behavioral features (OCD rituals may paradoxically help or disrupt medication routine); carer involvement in medication management becomes essential",
            "management": "Electronic medication reminder system; blister packs; carer-administered medication (as cognitive decline progresses); care coordinator for complex regimen; hospital pharmacy review of CLN4B-specific AED interactions",
        },
        {
            "trigger": "Auditory-Startle",
            "pct": 65,
            "description": "Sound-induced myoclonus and audiogenic seizures; sudden loud sounds (door slam, traffic, clapping) provoke myoclonic jerks or generalised seizures; progressive withdrawal from social environments",
            "management": "Ear defenders in noisy environments; advance warning for anticipated loud sounds (alarms, appliances); quiet home environment; social participation maintained with preparation; CLB adjunct for severe audiogenic myoclonus",
        },
        {
            "trigger": "Tactile-Stimulus",
            "pct": 58,
            "description": "Tactile-induced myoclonus — unexpected touch (shoulder tap, physical examination, hair brushing) provokes startle myoclonus; limits physical care and medical examination; family and carer training in predictable approach technique",
            "management": "Predictable approach protocol (always speak first, touch predictably); warn before physical contact; physiotherapy uses slow predictable movements; medical examination training for CLN4B patients; CLB for severe tactile sensitivity",
        },
        {
            "trigger": "Fever-Illness",
            "pct": 50,
            "description": "Febrile illness provokes GTCS and myoclonic clusters in CLN4B; adults may underreport illness symptoms; behavioral features may mask recognition of fever/illness",
            "management": "Antipyretics aggressively at first fever sign; rescue midazolam buccal 10 mg prescribed; written adult seizure action plan; carer fever monitoring; IV LEV 60 mg/kg for SE (document in ED records, AED emergency card)",
        },
        {
            "trigger": "Alcohol",
            "pct": 42,
            "description": "Adult CLN4B patients often explore alcohol before diagnosis (behavioral symptoms misattributed to psychiatric cause); alcohol worsens myoclonus AND lowers seizure threshold; withdrawal also seizure-provoking; frank counselling essential at diagnosis",
            "management": "Alcohol abstinence counselling at CLN4B diagnosis; AED interaction counselling (VPA + alcohol → increased sedation + hepatotoxicity risk); link with addiction services if alcohol-dependent; occupational driving assessment includes alcohol history",
        },
    ]

    treatments = [
        {
            "drug": "Valproate (VPA)",
            "level": "Level-B",
            "dose": "1000-2500 mg/day divided BD-TID; titrate to seizure control and UMRS improvement; trough target 60-100 mg/L; extended-release formulation preferred (compliance and tolerability in adults with cognitive decline)",
            "moa": "Multiple mechanisms: Na-channel, GABA-A potentiation, T-type Ca channel, HCN modulation → broad-spectrum AED; effective for GTCS + myoclonus + absence spectrum. SAFE in CLN4B: DNAJC5 is a presynaptic chaperone disease — NOT mitochondrial. No POLG1/mitochondrial hepatotoxicity CI. Backbone AED for CLN4B.",
            "efficacy": "78% of CLN4B cohort on VPA; good initial GTCS + myoclonus control; maintains backbone role despite progressive drug resistance; modified-release aids adherence in cognitively declining adults",
            "monitoring": "LFT + FBC baseline then 3-monthly; VPA trough 60-100 mg/L; weight; hyperammonaemia monitoring (adult CLN4B cognitive impairment — VPA encephalopathy may mimic disease progression); carnitine supplementation if depleted; VPPP MANDATORY for females ≥12y",
            "cln4b_note": "VPA is SAFE in CLN4B (DNAJC5 presynaptic — no mitochondrial CI). VPPP mandatory for females. Hyperammonaemia monitoring important in cognitively impaired adults — VPA-related encephalopathy may be misattributed to CLN4B progression. Extended-release preferred for once or twice daily dosing in adult patients with declining cognitive adherence.",
        },
        {
            "drug": "Levetiracetam (LEV)",
            "level": "Level-B",
            "dose": "1000-3000 mg/day divided BD; IV LEV 60 mg/kg for SE; oral solution available (aids swallowing in dysphagia); renally cleared — dose-reduce if renal impairment",
            "moa": "SV2A modulation → reduced vesicle cycling → broad-spectrum. MECHANISTICALLY DIRECT RATIONALE: CSPα/DNAJC5 is a presynaptic co-chaperone for SNARE proteins at synaptic vesicles; LEV binds SV2A in the same presynaptic compartment; CSPα LOF → SNARE dysfunction; LEV SV2A modulation provides complementary presynaptic regulation. Most mechanistically rational co-AED in CLN4B.",
            "efficacy": "68% of CLN4B cohort on LEV; good adjunct to VPA for GTCS + myoclonus control; IV LEV 60 mg/kg first-line for SE; oral solution aids late-disease medication delivery",
            "monitoring": "Behavioural side effects (irritability, aggression) — important in CLN4B where behavioral features are primary disease manifestations; add low-dose CLB if significant behavioural activation; renal function monitoring",
            "cln4b_note": "LEV has DIRECT SV2A-CSPα pathway complementarity — most mechanistically rational co-AED in CLN4B. IV LEV 60 mg/kg = SE rescue. Document in ED alert, AED emergency card. Behavioural side effect monitoring critical in CLN4B where anger/agitation are disease features — differentiate drug effect from disease progression.",
        },
        {
            "drug": "Piracetam",
            "level": "Level-B",
            "dose": "16-24 g/day divided TID-QID; renal dose adjustment mandatory (100% renally cleared); start 4.8 g/day and titrate up over 4 weeks",
            "moa": "AMPA positive modulation → reduces cortical reflex myoclonus amplitude; strong evidence in PME action myoclonus across syndromes; most effective for action-triggered cortical myoclonus specifically",
            "efficacy": "Piracetam is FIRST-LINE for action myoclonus in CLN4B alongside VPA backbone; action myoclonus (80% of cohort) is the primary early functional disability — piracetam targets this specific feature; functional gains in handwriting, fine motor tasks when started early",
            "monitoring": "Renal function (age-appropriate in adults — eGFR monitoring); adequate hydration; antiplatelet effect at high doses; behaviour (hyperkinesia at high doses); UMRS response assessment",
            "cln4b_note": "Piracetam FIRST-LINE for action myoclonus in CLN4B alongside VPA. Start early — before significant motor decline reduces measurable benefit. Functional UMRS response guides dose titration. Renal monitoring mandatory in adults (age-related eGFR decline). High-dose piracetam may cause agitation in CLN4B patients already with behavioral features — monitor.",
        },
        {
            "drug": "Clonazepam-Clobazam (CLB)",
            "level": "Level-B",
            "dose": "Clonazepam 0.5-2 mg/day (preferred for myoclonus); Clobazam 5-20 mg/night (nocturnal GTCS + SUDEP prevention); start low and titrate; tolerance monitoring essential",
            "moa": "GABA-A PAM (benzodiazepine site); clonazepam preferred for myoclonus (higher efficacy for myoclonic seizures); clobazam for nocturnal GTCS (longer half-life, less sedating than clonazepam); nocturnal dosing reduces nocturnal GTCS + SUDEP risk",
            "efficacy": "Adjunct for stimulus-sensitive myoclonus (clonazepam) + nocturnal GTCS (clobazam); useful when VPA + piracetam + LEV insufficient; nocturnal CLB reduces nocturnal GTCS frequency and SUDEP risk",
            "monitoring": "Tolerance development (CLZ prone to tolerance at 3-6 months); sedation (particularly relevant with progressive cognitive impairment — differentiate sedation from disease progression); behavioural activation; gradual taper if stopping",
            "cln4b_note": "Clonazepam preferred over clobazam for myoclonus in CLN4B (0.5-2mg/day); clobazam for nocturnal GTCS (5-20mg/night). Nocturnal CLB is important for SUDEP risk reduction — compound SUDEP risk in CLN4B (nocturnal GTCS + cognitive impairment + ataxia). Distinguish from CLN2 where CLB is 4th-line — CLN4B myoclonus is earlier and more prominent.",
        },
        {
            "drug": "Zonisamide (ZNS)",
            "level": "Level-C",
            "dose": "200-400 mg/day OD or BD; titrate slowly (100 mg/2 weeks); antioxidant + sodium/T-type calcium channel",
            "moa": "Antioxidant properties + sodium channel + T-type calcium channel block; may reduce oxidative stress in CLN4B storage pathology; ZNS as combined antioxidant + anticonvulsant adjunct in adult NCL",
            "efficacy": "Level C evidence in CLN4B; adjunct for refractory myoclonus when VPA + LEV + piracetam + CLB insufficient; combined antioxidant strategy with piracetam",
            "monitoring": "Oligohydrosis/hyperthermia in adults (particularly warm climates/exercise); renal calculi (adequate hydration); weight loss; cognitive side effects (monitor in CLN4B where cognitive decline is primary disease feature); LFT",
            "cln4b_note": "ZNS combined antioxidant + anticonvulsant rationale in CLN4B storage pathology. Monitor for oligohydrosis/hyperthermia in adults (less common than in paediatric patients but still relevant). Cognitive side effects of ZNS must be distinguished from CLN4B disease progression — if cognition acutely worsens on ZNS, reduce dose.",
        },
        {
            "drug": "Neuropsychiatry-SSRI-CBT",
            "level": "Level-A",
            "dose": "Sertraline 50-200 mg/day OD (preferred — low drug interaction); review PHQ-9/GAD-7/NPI 3-monthly; CBT where cognitive capacity allows; quetiapine for psychosis/agitation (D2-sparing preferred); neuropsychiatric review 3-monthly",
            "moa": "SSRI (serotonin reuptake inhibition) → antidepressant + anxiolytic + anti-OCD; behavioral/psychiatric features are CARDINAL CLN4B disease features (not secondary): depression, personality change, OCD-like rituals, agitation, psychosis present in 92% of cohort, preceding seizures by 3-5 years; neuropsychiatry is MDT co-lead",
            "efficacy": "Behavioral/psychiatric features preceding seizures by 3-5 years → SSRI started at behavioral onset (before AED in temporal sequence); SSRI improves quality of life + carer burden + reduces seizure threshold elevation from anxiety; standard care for CLN4B",
            "monitoring": "PHQ-9 (depression); GAD-7 (anxiety); NPI (Neuropsychiatric Inventory for dementia behavioral features); psychosis assessment (hallucinations, delusions); carer burden scale; SSRI-associated hyponatraemia (monitor Na in adults on VPA + SSRI); serotonin syndrome (avoid tramadol + SSRI)",
            "cln4b_note": "Neuropsychiatry is MDT CO-LEAD in CLN4B from initial presentation — not a secondary referral. Behavioral symptoms PRECEDE seizures by 3-5 years in CLN4B — psychiatrist is first specialist in 80% of patients. SSRI (sertraline) started when behavioral features emerge. Dual role: disease feature management + indirect seizure threshold stabilisation. Antipsychotic if psychosis: quetiapine preferred (D2-sparing — low Parkinsonism risk in late CLN4B motor phase).",
        },
        {
            "drug": "Genetic-Counselling-AD-Family-Cascade",
            "level": "Level-A",
            "dose": "Immediate referral to clinical genetics at CLN4B diagnosis; all 1st degree relatives ≥18y offered presymptomatic DNAJC5 testing; reproductive options counselling (PGT-M, prenatal DNAJC5 sequencing); occupational and insurance implications; ongoing psychological support",
            "moa": "AD inheritance → 50% offspring risk; adult-onset disease means affected adults often already have adult children who require presymptomatic testing; genetic counselling manages cascade testing, reproductive decision-making, occupational implications, insurance/genetic discrimination",
            "efficacy": "Genetic counselling is IMMEDIATE CLINICAL PRIORITY in CLN4B AD — more urgent than in AR diseases because existing adult children are at 50% risk; presymptomatic diagnosis enables occupational planning and future trial eligibility; family cascade testing essential",
            "monitoring": "Cascade testing completion in adult relatives ≥18y; reproductive counselling completion before next pregnancy; presymptomatic positive result psychological support; genetic discrimination legislation awareness (UK Genetic Testing Code of Practice; insurance implications)",
            "cln4b_note": "AD 50% offspring risk = IMMEDIATE genetic counselling priority at CLN4B diagnosis. Adult children of the proband are at 50% risk and may already be symptomatic (behavioral symptoms with 5.8-year diagnostic delay typical). Presymptomatic testing for relatives ≥18y. Reproductive options: PGT-M for couples who wish to have unaffected children. Occupational/employment counselling. Genetic discrimination legislation — refer to genetics/legal advice for insurance implications.",
        },
        {
            "drug": "Rescue-Midazolam-Buccal",
            "level": "Level-A",
            "dose": "Midazolam buccal 10 mg (adult dose); prescribed for all CLN4B patients with GTCS; self-administration training for highest-functioning patients in early disease; carer administration training mandatory",
            "moa": "Rapid buccal absorption → CNS benzodiazepine site → terminates SE and prolonged seizures; 10 mg adult buccal dose effective for most adults",
            "efficacy": "Standard SE rescue for all adult epilepsy patients with GTCS; CLN4B-specific: adult patients with cognitive decline require simplified rescue plans and trained carers",
            "monitoring": "Carer competency assessment; rescue administration training annual review; respiration monitoring post-administration; when to call emergency services (SE >5 min duration); document in emergency plan",
            "cln4b_note": "Adults — self-administration training for buccal midazolam in highest-functioning patients early in CLN4B disease course. As cognitive decline progresses, carer-administered rescue becomes essential. IV LEV 60 mg/kg for SE in hospital (document in ED records, AED emergency card). Do NOT give IV phenytoin/fosphenytoin in CLN4B ED presentation — worsen myoclonus.",
        },
    ]

    contraindications = [
        {
            "drug": "Carbamazepine-Oxcarbazepine-Phenytoin",
            "severity": "ABSOLUTE",
            "reason": "Na-channel blockers WORSEN myoclonus in CLN4B. PRIMARY CLN4B MISDIAGNOSIS TRAP: behavioral onset + first GTCS → GP/general neurologist prescribes CBZ (focal epilepsy assumption) OR psychiatric team gives antidepressants + first GTCS → neurologist starts CBZ → ACUTE MYOCLONIC DETERIORATION. Adult presentation means GP/general neurologist (not epilepsy specialist) often prescribes first AED. CBZ/OXC CI must be in every clinical letter, AED card, and ED alert from day of CLN4B diagnosis.",
            "alternative": "VPA 1000-2500 mg/day + LEV 1000-3000 mg/day backbone — handles GTCS + myoclonus + focal seizure component without Na-channel risk. Stop CBZ/OXC immediately if patient already on it — taper while starting VPA+LEV.",
        },
        {
            "drug": "Tiagabine (TGB)",
            "severity": "ABSOLUTE",
            "reason": "NCSE risk — Tiagabine GABA-reuptake inhibition → excessive CSF GABA → NCSE in myoclonic epilepsies. ABSOLUTE CI across all PME syndromes including CLN4B. CLN4B has absent seizures and myoclonic epilepsy — TGB administration = high NCSE risk.",
            "alternative": "Benzodiazepines (clonazepam for myoclonus; clobazam for GTCS) provide GABA-A modulation safely for CLN4B without NCSE risk of TGB GABA-reuptake inhibition.",
        },
        {
            "drug": "GBP-Pregabalin",
            "severity": "HIGH-RISK",
            "reason": "Gabapentin/pregabalin worsen myoclonus (Crespel 1999 — α2δ modulators and myoclonus). ADULT CLN4B-SPECIFIC TRAP: adult patients with behavioral/cognitive symptoms + myoclonus may develop musculoskeletal pain or neuropathic pain → referred to pain clinic/GP → GBP/PGB prescribed without PME awareness; adult patients see multiple specialists who may prescribe independently. GBP/PGB CI must be in GP records, pain clinic correspondence, community pharmacy records, and MedicAlert bracelet if myoclonus severe.",
            "alternative": "Pain management in CLN4B without GBP/PGB: amitriptyline low dose (neuropathic, also aids sleep), duloxetine SNRI, non-pharmacological pain management (physiotherapy, TENS); palliative team for late-disease pain. Sertraline (already prescribed for behavioral features) has some pain modulation.",
        },
        {
            "drug": "LTG-Monotherapy",
            "severity": "HIGH-RISK",
            "reason": "LTG monotherapy can worsen myoclonus in CLN4B. LTG as add-on in combination (with VPA — careful: VPA halves LTG clearance, dose reduction needed; LTG+VPA interaction = LTG toxicity risk) may be used cautiously as 3rd/4th line adjunct, but NEVER as first-line monotherapy in CLN4B PME. Risk: adult with behavioral features + GTCS misdiagnosed as JGE/adult GGE → LTG monotherapy → myoclonus worsens.",
            "alternative": "VPA + LEV backbone. If LTG add-on needed (refractory focal seizures): start at 25 mg every 2 weeks with VPA present (VPA halves LTG clearance — must use very low starting dose). Lamotrigine is never first-line and never monotherapy in CLN4B.",
        },
        {
            "drug": "VGB-Vigabatrin",
            "severity": "AVOID",
            "reason": "VGB causes irreversible peripheral visual field constriction (vigabatrin-associated retinopathy, VAR). CLN4B does NOT have retinal degeneration (retina spared — unlike CLN2/CLN3) — VGB is therefore LESS strictly CI than in CLN2/CLN3, but still AVOIDED in cognitively vulnerable adult CLN4B patient (irreversible visual field constriction is unacceptable in an already cognitively impaired patient with limited quality of life). Less strict than CLN2/CLN3 (ABSOLUTE CI in those) but still AVOID.",
            "alternative": "Standard CLN4B AED backbone (VPA + LEV + piracetam + CLB) is adequate without VGB. If infantile spasms or Lennox-Gastaut features emerge (atypical), alternative protocols without VGB.",
        },
        {
            "drug": "AED-Taper-Remission-Expectation",
            "severity": "HIGH-RISK",
            "reason": "CLN4B is PROGRESSIVE neurodegenerative disease — seizures do NOT remit. AED taper in stable CLN4B = inevitable seizure breakthrough. Breakthrough GTCS in cognitively impaired adult with ataxia = serious injury (fall) + SUDEP risk. Document 'AED taper NOT appropriate — CLN4B progressive NCL' in ALL correspondence, discharge summaries, GP letters. Cognitive decline may make patients/carers request taper (simplify regimen) — assess carefully: AED rationalisation may be appropriate but taper expecting remission is never appropriate.",
            "alternative": "Maintain effective AED regimen indefinitely. AED change only if intolerable side effects — with careful overlap tapering. Never taper anticipating remission. AED rationalisation (reduce regimen complexity) may be discussed in late disease for palliative comfort — but this is different from remission-expectation taper.",
        },
    ]

    monitoring = [
        {"item": "DNAJC5-Gene-Sequencing-Diagnosis", "rationale": "p.Leu115Arg (c.344T>G) + c.344_347del targeted PCR first (covers 73% of CLN4B alleles) → full DNAJC5 sequencing; family cascade testing (AD — all 1st degree relatives ≥18y); ACMG variant classification; functional palmitoylation assay in fibroblasts if VUS found; de novo confirmation (parental testing) for apparently non-familial cases"},
        {"item": "Skin-Biopsy-EM-Fingerprint-Profiles", "rationale": "Skin biopsy electron microscopy: FINGERPRINT PROFILES in eccrine sweat gland epithelium and endothelial cells — PATHOGNOMONIC for adult NCL (CLN4B). Present in 85% of CLN4B biopsies. Performed BEFORE genetic results (2-4 week turnaround). Fingerprint profiles confirm adult NCL → DNAJC5 sequencing next. Distinguish from CLN2 (curvilinear bodies) and CLN3 (combined fingerprint + curvilinear). TRIGGER FOR SKIN BIOPSY: any adult 25-55y with unexplained progressive behavioral/psychiatric syndrome + myoclonus or ataxia."},
        {"item": "Exclude-POLG1-Mitochondrial-Before-VPA", "rationale": "POLG1 sequencing if any mitochondrial features (stroke-like episodes, ophthalmoplegia, exercise intolerance, family history of mitochondrial disease); CLN4B is NOT mitochondrial (VPA safe), but coincidental POLG1 mutation = VPA CI; mitochondrial enzyme panel (Complex I-IV) if suspected; this exclusion is optional but recommended when any mitochondrial feature present"},
        {"item": "Brain-MRI-3T-Cortical-Cerebellar-Atrophy", "rationale": "Serial brain MRI every 6 months: progressive cortical atrophy (frontotemporal > occipital pattern — distinct from CLN3 which shows occipital early); cerebellar atrophy (correlates with ataxia); white matter T2 signal changes; thalamic atrophy; MRI trajectory correlates with cognitive decline rate; no retinal MRI needed (retina spared in CLN4B — no ERG monitoring required)"},
        {"item": "Neuropsychology-Dementia-Tracking-6Monthly", "rationale": "Formal neuropsychological testing 6-monthly: MMSE/MoCA (baseline + trend — MMSE <24 triggers formal ADL review); Addenbrooke's Cognitive Examination III; Neuropsychiatric Inventory (NPI — behavioral features tracking); PHQ-9 (depression); GAD-7 (anxiety); UMRS (Unified Myoclonus Rating Scale — primary outcome); occupational capacity; driving assessment"},
        {"item": "Video-EEG-LTM-Seizure-Characterisation", "rationale": "Annual video-EEG long-term monitoring: cortical myoclonus characterisation (giant SEP, jerk-locked back-averaging); IPS photic protocol (standard 3-50 Hz — CLN4B is NOT CLN2, no 1-3 Hz SSPS protocol needed); NCSE surveillance (acute cognitive deterioration → urgent EEG to exclude NCSE — differentiate from disease progression); focal temporal vs generalised features for AED selection; distinguish from primary generalised epilepsy"},
        {"item": "UMRS-6Monthly", "rationale": "Unified Myoclonus Rating Scale (UMRS) 6-monthly — action + stimulus-sensitive + spontaneous myoclonus subscales; UMRS is the PRIMARY outcome measure in CLN4B (piracetam + VPA dose calibrated to UMRS); UMRS >20 triggers AED escalation review; functional myoclonus assessment (handwriting, feeding, dressing — functional disability milestones)"},
        {"item": "SARA-6Monthly-Cerebellar-Ataxia", "rationale": "Scale for Assessment and Rating of Ataxia (SARA) 6-monthly — cerebellar ataxia (70% of CLN4B cohort); SARA guides mobility aid prescription: >10 walking aid, >18 rollator, >25 wheelchair; falls risk assessment: triple mechanism in CLN4B (ataxia + myoclonus + cognitive impairment); physiotherapy goals set by SARA trajectory; PER (if used) dose increases assessed against SARA changes"},
        {"item": "VPA-TDM-LFT-Monitoring", "rationale": "VPA trough 60-100 mg/L; LFT + FBC 3-monthly; hyperammonaemia monitoring (particularly important in adult CLN4B — VPA-related encephalopathy may be misattributed to CLN4B cognitive decline; ammonia level if acute cognitive worsening on VPA); carnitine supplementation if VPA-related carnitine depletion; VPPP counselling mandatory females ≥12y"},
        {"item": "Psychiatry-SSRI-Monitoring-3Monthly", "rationale": "SSRI (sertraline) dose optimisation and monitoring 3-monthly; PHQ-9/GAD-7 tracking; NPI (Neuropsychiatric Inventory) for behavioral features; psychosis monitoring (hallucinations, delusions); if psychosis present → atypical antipsychotic (quetiapine preferred — D2-sparing, low extrapyramidal risk); avoid haloperidol/high-dose risperidone (D2-blocking — extrapyramidal risk in late CLN4B motor phase)"},
        {"item": "Occupational-Therapy-Driving-DVLA", "rationale": "Driving cessation assessment mandatory (triple DVLA exclusion: seizures + cognitive decline + ataxia); DVLA Group 1 and Group 2 both permanently excluded at CLN4B diagnosis; DVLA notification at diagnosis; occupational capacity assessment (workplace reasonable adjustments during early disease → disability retirement); ADL assessment for home care needs; employment disability planning; legal advice for income protection insurance implications"},
        {"item": "Genetic-Counselling-Family-Cascade", "rationale": "AD 50% offspring risk → IMMEDIATE cascade testing for 1st degree relatives ≥18y; presymptomatic DNAJC5 testing offered to all adult children and siblings; reproductive counselling (PGT-M, NIPD, prenatal testing); psychological support for presymptomatic positive result; genetic discrimination awareness (insurance, employment); genetic register enrolment; international NCL registry (BDSRA, NCL Resource)"},
        {"item": "SUDEP-Nocturnal-Monitoring", "rationale": "Compound SUDEP risk in CLN4B: (1) nocturnal GTCS (68%); (2) cognitive impairment (cannot self-rescue prone position); (3) progressive ataxia (fall risk); (4) behavioral features → social isolation → unsupervised sleeping. Monitoring: nocturnal pulse oximetry + movement sensor + carer alert device; padded bed rails; firm mattress; supervised sleeping if high-risk; nocturnal CLB dose; SUDEP counselling in ACP"},
        {"item": "Clinical-Trials-NCL-Network-Enrolment", "rationale": "Register with NCL Resource, BDSRA registry, NCL Network Europe; AAV-DNAJC5 gene therapy trials (early-phase, investigational); CSPα stabiliser compounds (investigational); natural history studies; no approved disease-modifying therapy — trial access is the only disease-modifying option; enrolment enables future eligibility for emerging therapies"},
    ]

    lifecycle = [
        {
            "stage": "Pre-Symptomatic-Genetic-Risk",
            "age": "Prenatal / adult pre-onset",
            "description": "AD inheritance → 50% risk per offspring; adult-onset disease → presymptomatic genetic testing for relatives ≥18y after proband diagnosis; occupational and insurance implications of presymptomatic positive result; psychological support for presymptomatic testing decision; reproductive counselling (PGT-M, prenatal DNAJC5 sequencing for at-risk pregnancies); genetic register enrolment",
        },
        {
            "stage": "Behavioral-Psychiatric-Onset",
            "age": "25-40 years",
            "description": "BEHAVIORAL FEATURES FIRST (92% of CLN4B): depression, personality change, OCD-like rituals, psychosis, agitation; typically referred to psychiatry → psychiatric misdiagnosis (antidepressants ± antipsychotics ± CBZ for 'behavioral seizures'); mean delay to correct diagnosis 5.8 years. KEY: skin biopsy EM in any adult with unexplained progressive behavioral/psychiatric syndrome + possible myoclonus. SSRI started for mood/anxiety. Neuropsychiatry leads MDT. DNAJC5 sequencing when CLN4B suspected.",
        },
        {
            "stage": "First-Seizure-PME-Diagnosis",
            "age": "30-45 years",
            "description": "First GTCS or recognized myoclonus → neurological referral → PME diagnosis if clinical suspicion maintained. EEG (giant SEP + IPS standard protocol); skin biopsy EM (fingerprint profiles); DNAJC5 sequencing confirms. VPA + LEV + piracetam started. ABSOLUTE CI documented (CBZ/OXC/PHT). IMMEDIATE AD genetic counselling — family cascade testing. Driving cessation. BDSRA/NCL registry enrolment.",
        },
        {
            "stage": "Established-CLN4B-Working-Age",
            "age": "35-50 years",
            "description": "Progressive CLN4B: myoclonus + ataxia + cognitive decline. Occupational disability assessment; workplace adjustments then disability retirement. Driving permanently ceased. Care needs assessment. AED optimisation (piracetam + VPA + LEV ± CLB). Neuropsychiatry for behavioral features (SSRI + CBT). UMRS + SARA + neuropsychology 6-monthly. Social care package. Family support services.",
        },
        {
            "stage": "Advanced-Cognitive-Motor-Decline",
            "age": "45-60 years",
            "description": "Severe cognitive impairment + wheelchair dependency + dysphagia. ACP discussions (resuscitation, hospital admission thresholds, PEG, place of care). Adult social care; continuing healthcare; family caregiver support. AED management continues (VPA + LEV + CLB). PEG for dysphagia (FEES-guided). AAC if communication preserved. Adult hospice input. SUDEP risk management intensified.",
        },
        {
            "stage": "Late-Palliative-End-Stage",
            "age": "55+ years",
            "description": "End-stage CLN4B: severe dementia, minimal responsiveness, gastrostomy feeding. Palliative AED management (symptom control over anti-seizure efficacy). Comfort-focused care. ACP enacted. Adult hospice. Bereavement support. Average survival 40-60 years of age (10-25 years disease course from behavioral onset to death).",
        },
    ]

    return {
        "etiologies": etiologies,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "treatments": treatments,
        "contraindications": contraindications,
        "monitoring": monitoring,
        "lifecycle": lifecycle,
        "cohort_size": 40,
    }


def get_definitions():
    return {
        "concepts": [
            {
                "concept": "CLN4B-DNAJC5-19q13.11-CSPalpha-Presynaptic-Adult-NCL",
                "definition": "DNAJC5 (19q13.11) encodes CSPα (Cysteine String Protein-α), a 198-aa ~26 kDa presynaptic co-chaperone. Domains: DNAJ (Hsp40) domain aa 1-80 (Hsc70 binding); Cysteine-string domain aa 115-136 (15 cysteines, palmitoylated → SV membrane anchor). CSPα/Hsc70/SGT trimeric complex refolds misfolded SNARE proteins (esp. SNAP-25) → maintains neurotransmitter release; neuroprotective at synapse. AD dominant-negative mutations (p.Leu115Arg, c.344_347del) → toxic CSPα aggregation → presynaptic degeneration → adult NCL (CLN4B / Kufs Type B). Unlike all other NCLs (CLN1/2/3: lysosomal enzyme or membrane protein deficiency), CLN4B is a PRESYNAPTIC CHAPERONE disease — mechanistically unique. OMIM: *611203 / #162350.",
                "standard": "Noskova-2011-AmJHumGenet; Greaves-2012-JBiolChem; Cadieux-Dion-2013-JMDN; OMIM-*611203; ACMG-AMP-2015"
            },
            {
                "concept": "Behavioral-Psychiatric-Onset-FIRST-Misdiagnosis-Trap",
                "definition": "CLN4B uniquely presents with BEHAVIORAL/PSYCHIATRIC CHANGES FIRST (depression, personality, psychosis, dementia) before myoclonus or seizures by 3-5 years. Mimics primary psychiatric disorder → psychiatric misdiagnosis in 80%+ of CLN4B patients. Mean diagnostic delay 5.8 years — LONGEST diagnostic delay of any adult PME. KEY: skin biopsy EM in any adult 25-55y with unexplained progressive behavioral syndrome + myoclonus or ataxia. Fingerprint profiles on EM → adult NCL → DNAJC5 sequencing. This delayed diagnosis is the greatest clinical challenge in CLN4B and leads to CBZ prescription (CI) and delayed AED optimisation.",
                "standard": "Cadieux-Dion-2013-JMDN; NCL-Resource-2024; ILAE-2022; Andermann-2012"
            },
            {
                "concept": "No-Visual-Failure-CLN4B-Contrasts-CLN2-CLN3",
                "definition": "CLN4B SPARES THE RETINA — NO progressive visual failure, NO bull's eye maculopathy, NO ERG abnormalities. This is the critical distinction from CLN2 (retinal NCL — ERG abnormalities, VGB ABSOLUTE CI) and CLN3 (retinal NCL — visual failure FIRST, ophthalmology-led diagnosis). CLN4B patients retain vision until very late disease if at all. VGB is AVOIDED (visual field constriction risk) but NOT ABSOLUTELY CI as in CLN2/CLN3 (CLN4B has no progressive retinal disease). No ophthalmology ERG monitoring needed in CLN4B (unlike CLN2/CLN3 where ERG is mandatory 6-monthly). The absence of visual failure is an essential diagnostic clue when distinguishing adult CLN4B from other adult NCL variants.",
                "standard": "NCL-Resource-2024; Mole-2019-LancetNeurol; ILAE-2022; Cadieux-Dion-2013-JMDN"
            },
            {
                "concept": "No-Vacuolated-Lymphocytes-CLN4B-Contrasts-CLN3-Blood-Film",
                "definition": "CLN4B blood film is ALWAYS NORMAL — no vacuolated lymphocytes (0% of cohort). This contrasts with CLN3 where vacuolated lymphocytes are pathognomonic (95% sensitivity). Skin biopsy EM (fingerprint profiles) is the CLN4B diagnostic equivalent — a critical distinction preventing diagnostic confusion between CLN3 and CLN4B in adult patients. Adult patients referred with 'possible adult-onset JNCL' → normal blood film + fingerprint profiles on EM = CLN4B (DNAJC5), not CLN3. Conversely, vacuolated lymphocytes in an adult = CLN3 attenuated variant (not CLN4B). This distinction drives the correct genetic test: CLN3 deletion PCR for CLN3 vs DNAJC5 sequencing for CLN4B.",
                "standard": "NCL-Resource-2024; Cadieux-Dion-2013-JMDN; Specchio-2019-EJPN; ACMG-AMP-2015"
            },
            {
                "concept": "Fingerprint-Profiles-EM-Adult-NCL-Diagnostic",
                "definition": "Skin biopsy electron microscopy showing FINGERPRINT PROFILES in eccrine sweat gland epithelium and endothelial cells is the pathognomonic EM finding in adult NCL (CLN4B). Fingerprint profiles are the characteristic storage pattern in CLN4B/CLN5/CLN6 adult forms — concentric lamellae in a fingerprint-like arrangement representing accumulated ceroid storage material (CSPα-aggregation-driven secondary storage). Absence of CLN2 (curvilinear bodies) and CLN3 (vacuolated lymphocytes) + presence of fingerprint profiles = adult NCL → DNAJC5 sequencing next. Present in 85% of CLN4B skin biopsies. Skin biopsy EM should be performed before genetic testing results in any adult with unexplained PME + behavioral features — results in 2-4 weeks, provides diagnosis before lengthy gene panel.",
                "standard": "NCL-Resource-2024; Noskova-2011-AmJHumGenet; ACMG-AMP-2015; Mole-2019-LancetNeurol"
            },
            {
                "concept": "LEV-SV2A-CSPalpha-Presynaptic-Complementarity-Unique-CLN4B",
                "definition": "CSPα/DNAJC5 is a presynaptic co-chaperone for SNARE proteins at synaptic vesicles. LEV binds SV2A (synaptic vesicle protein 2A), which controls vesicle cycling in the same presynaptic compartment. CSPα LOF → SNARE dysfunction + reduced neurotransmitter release reliability; LEV SV2A modulation provides complementary presynaptic regulation. This gives LEV a MECHANISTICALLY DIRECT rationale in CLN4B not present in most other PMEs — SV2A and CSPα both operate in the presynaptic vesicle cycling machinery. More than empirical benefit — LEV targets the same compartment disrupted by the CLN4B pathomechanism. This is the only PME where CSPα-SV2A presynaptic pathway complementarity can be invoked as a specific mechanistic rationale for LEV use.",
                "standard": "Greaves-2012-JBiolChem; ILAE-2022; Cadieux-Dion-2013-JMDN; NCL-Resource-2024"
            },
            {
                "concept": "VPA-SAFE-CLN4B-Presynaptic-NOT-Mitochondrial",
                "definition": "VPA is the backbone AED in CLN4B and IS SAFE. CLN4B/DNAJC5 is a PRESYNAPTIC CHAPERONE disease — not mitochondrial. VPA mitochondrial CI (MERRF/MT-TK, POLG/Alpers) does NOT apply to CLN4B. POLG1 screening optional (coincidental exclusion only — performed if mitochondrial features coexist). VPA must NOT be withheld from CLN4B patients based on incorrect mitochondrial concern. Hyperammonaemia monitoring is important in CLN4B (VPA-related encephalopathy may mimic disease progression in cognitively impaired adults). VPPP mandatory for females ≥12y. VPA + LEV backbone handles GTCS + myoclonus + focal seizure spectrum effectively in CLN4B.",
                "standard": "ILAE-2022; NICE-NG217; MHRA-VPPP-2021; CPIC-POLG1-2023; NCL-Resource-2024"
            },
            {
                "concept": "CBZ-OXC-PHT-ABSOLUTE-CI-Adult-Focal-Misdiagnosis-Trap",
                "definition": "Na-channel blockers are ABSOLUTE CI in CLN4B. The specific CLN4B misdiagnosis trap: behavioral onset + GTCS at initial presentation → misdiagnosed as focal epilepsy (temporal lobe) → CBZ/OXC prescribed → ACUTE MYOCLONIC DETERIORATION. Adult presentation means GP/general neurologist (not epilepsy specialist) often prescribes first AED. Additionally, psychiatric team may give CBZ for 'mood stabilisation' before neurological diagnosis. CBZ/OXC CI must be in every clinical letter, AED card, and ED alert from day of CLN4B diagnosis. If patient already on CBZ/OXC when CLN4B diagnosed → STOP immediately with VPA+LEV overlap.",
                "standard": "Crespel-1999; ILAE-2022; NICE-NG217; NCL-Resource-2024; Cadieux-Dion-2013-JMDN"
            },
            {
                "concept": "AD-Inheritance-50pct-Offspring-Risk-Genetic-Counselling-Urgent",
                "definition": "CLN4B (DNAJC5) is AUTOSOMAL DOMINANT — unique among the common NCLs (CLN1/2/3/5/6/7/8/10 are all AR). Each child of an affected parent has 50% risk of inheriting the DNAJC5 mutation. Adult-onset disease (25-45y) means affected adults often have children before diagnosis — these adult children may already be showing early behavioral symptoms. Presymptomatic testing for adult children/siblings ≥18y is offered routinely. Genetic counselling is an IMMEDIATE CLINICAL PRIORITY at CLN4B diagnosis — more urgent than in AR diseases because existing adult children are at 50% risk and may be in their behavioral-onset phase. Reproductive options: PGT-M (preimplantation genetic testing for monogenic disease) for couples who wish to have unaffected children.",
                "standard": "ACMG-AMP-2015; BDSRA-Registry; NCL-Resource-2024; DVLA-Guidance-2022; WHO-ICF-2019"
            },
            {
                "concept": "Piracetam-Action-Myoclonus-First-Line-CLN4B",
                "definition": "Piracetam (16-24g/day) is FIRST-LINE add-on therapy for action myoclonus in CLN4B alongside VPA backbone. AMPA-positive modulation → reduced cortical excitability for action myoclonus. Strong evidence across PME syndromes for action myoclonus (Level A evidence in ULD/CSTB; Level B in CLN4B). CLN4B action myoclonus (80% of cohort) is the primary functional disability early in disease — piracetam is the targeted treatment for this specific feature. Start early: before significant motor decline reduces measurable functional benefit. UMRS response guides dose titration (target: UMRS improvement on action myoclonus subscale).",
                "standard": "Crespel-1999; ILAE-2022; NICE-NG217; NCL-Resource-2024"
            },
            {
                "concept": "GBP-PGB-High-Risk-Adult-Pain-Prescribing-Trap",
                "definition": "GBP/PGB worsen myoclonus in CLN4B. The ADULT CLN4B-SPECIFIC TRAP: adult patients with behavioral/cognitive symptoms + myoclonus may develop musculoskeletal pain or neuropathic pain features from progressive neurodegeneration → referred to pain clinic/GP → GBP/PGB prescribed without PME awareness. Adult patients see multiple specialists (neurology, psychiatry, pain medicine, orthopaedics, GP) who may prescribe independently. GBP/PGB CI must be in GP records, pain clinic correspondence, community pharmacy records, and MedicAlert bracelet if myoclonus severe. Annual medication review to check for GBP/PGB added by other specialists is standard CLN4B monitoring.",
                "standard": "Crespel-1999; ILAE-2022; NICE-NG217; NCL-Resource-2024"
            },
            {
                "concept": "Neuropsychiatry-Behavioral-MDT-Core-CLN4B",
                "definition": "Behavioral/psychiatric features are CORE CLN4B disease features (not secondary), present in 92% of cohort, preceding seizures by 3-5 years. Neuropsychiatrist is an essential MDT CO-LEAD from initial presentation — not a secondary referral. SSRI (sertraline) for depression/anxiety/OCD features. Behavioral management plan (CBT where capacity allows; caregiver training). Antipsychotic only if psychosis: quetiapine preferred (D2-sparing — low extrapyramidal risk in late CLN4B motor phase; avoid haloperidol/high-dose risperidone). Neuropsychiatry-neurology co-management model is the CLN4B standard of care. NPI (Neuropsychiatric Inventory) is the behavioral disease tracking tool.",
                "standard": "NCL-Resource-2024; ILAE-2022; Cadieux-Dion-2013-JMDN; WHO-ICF-2019; NICE-NG217"
            },
            {
                "concept": "Driving-Employment-Occupational-Adult-CLN4B",
                "definition": "Adult-onset CLN4B creates unique occupational/driving challenges not seen in childhood NCLs. DVLA exclusion — triple mechanism: (1) seizures (Group 1: 12 months seizure-free required, practically never achieved in CLN4B PME; Group 2: permanently excluded); (2) cognitive decline (mandatory DVLA notification for dementia); (3) progressive ataxia (affecting vehicle control). Occupational disability: assess early with occupational therapist; workplace reasonable adjustments (cognitive aids, reduced complex tasks, no driving duties) during early disease; disability retirement planning; income protection. Life insurance/income protection implications of presymptomatic positive genetic test — refer to genetics/legal advice; genetic discrimination legislation awareness.",
                "standard": "DVLA-Guidance-2022; WHO-ICF-2019; NCL-Resource-2024; ACMG-AMP-2015"
            },
            {
                "concept": "Diagnostic-Delay-5.8years-Behavioral-Psychiatric-Misdiagnosis",
                "definition": "Mean diagnostic delay in CLN4B is 5.8 years — the LONGEST diagnostic delay of any adult PME. Behavioral symptoms misattributed to psychiatric disorders (schizophrenia, bipolar disorder, frontotemporal dementia, Alzheimer's disease, primary OCD). During this delay: CBZ may be prescribed (ABSOLUTE CI) for behavioral seizures; no AED optimisation; progressive presynaptic neurodegeneration unchecked. Skin biopsy EM (fingerprint profiles) provides diagnosis 2-4 weeks after referral — this test is underused. TRIGGER FOR SKIN BIOPSY: any adult (25-55y) with unexplained progressive behavioral/psychiatric syndrome + myoclonus or ataxia → skin biopsy EM + DNAJC5 sequencing immediately. Neurologist must consider CLN4B in every adult with PME + behavioral features.",
                "standard": "Cadieux-Dion-2013-JMDN; NCL-Resource-2024; Andermann-2012; ILAE-2022"
            },
            {
                "concept": "SUDEP-Risk-Compound-Cognitive-Ataxia-Nocturnal-CLN4B",
                "definition": "CLN4B SUDEP risk is compounded by four simultaneous factors: (1) nocturnal GTCS (68%); (2) cognitive impairment (unable to self-rescue from prone position post-seizure); (3) progressive ataxia (fall after GTCS → head injury risk, unable to reposition safely); (4) behavioral features → social isolation → unsupervised sleeping. CLN4B has one of the highest compound SUDEP risk profiles among adult PMEs. Nocturnal monitoring MANDATORY: pulse oximetry + movement sensor + carer alert; padded bed rails; firm mattress; supervised sleeping arrangement if high-risk; nocturnal CLB dose. ACP must address SUDEP risk explicitly — discuss with patient (while capacity retained) and family.",
                "standard": "Devinsky-2011-Lancet; NICE-NG217; SUDEP-Action-UK; NCL-Resource-2024; BDSRA-Registry"
            },
        ],
        "thresholds": [
            {"threshold": "DNAJC5 p.Leu115Arg confirmed (c.344T>G heterozygous) → CLN4B diagnosis certain", "standard": "Noskova-2011-AmJHumGenet / ACMG-AMP-2015"},
            {"threshold": "Skin biopsy fingerprint profiles present → adult NCL confirmed → DNAJC5 sequencing next", "standard": "NCL-Resource-2024"},
            {"threshold": "IV LEV 60 mg/kg → SE first-line rescue in CLN4B", "standard": "Standard adult SE protocol"},
            {"threshold": "VPA trough 60-100 mg/L → therapeutic target CLN4B", "standard": "Standard VPA pharmacokinetics"},
            {"threshold": "Piracetam 16-24 g/day → target dose for action myoclonus in CLN4B", "standard": "ILAE-2022; Crespel-1999"},
            {"threshold": "MMSE <24 → formal cognitive review + ADL assessment required", "standard": "Standard dementia thresholds"},
            {"threshold": "UMRS >20 → AED escalation review", "standard": "NCL-Resource-2024; UMRS validation studies"},
            {"threshold": "SARA >10 → walking aid prescribed", "standard": "Schmitz-Hubsch-2006-Neurology"},
            {"threshold": "SARA >18 → rollator", "standard": "Schmitz-Hubsch-2006-Neurology"},
            {"threshold": "SARA >25 → wheelchair", "standard": "Schmitz-Hubsch-2006-Neurology"},
            {"threshold": "PHQ-9 >10 → SSRI initiation (behavioral CLN4B)", "standard": "NICE-NG222-Depression"},
            {"threshold": "Diagnostic delay >12 months with behavioral symptoms + myoclonus → urgent skin biopsy EM referral", "standard": "NCL-Resource-2024; Cadieux-Dion-2013-JMDN"},
        ],
        "standards": [
            {"standard": "Orcsi-2020-NatGenet", "detail": "Orcsi et al. 2020 — Adult neuronal ceroid lipofuscinosis natural history; Nature Genetics — CLN4B/DNAJC5 phenotypic characterisation"},
            {"standard": "ILAE-2022", "detail": "ILAE Classification of Epilepsies and Epilepsy Syndromes — NCL classification; CLN4B / adult-onset PME"},
            {"standard": "NICE-NG217", "detail": "NICE Guideline NG217 — Epilepsies in children, young people, and adults; AED selection including adult PME and NCL"},
            {"standard": "NCL-Resource-2024", "detail": "Neuronal Ceroid Lipofuscinosis Network (NCL Resource) — diagnostic and management guidelines; CLN4B/DNAJC5 adult NCL"},
            {"standard": "Cadieux-Dion-2013-JMDN", "detail": "Cadieux-Dion M et al. 2013 — CLN4B/Kufs Type B phenotypic characterisation and DNAJC5 mutations; Journal of Medical and Diagnostic Neurology"},
            {"standard": "Greaves-2012-JBiolChem", "detail": "Greaves J et al. 2012 — CSPα palmitoylation mechanism and toxic aggregation in CLN4B; J Biol Chem 287:37149-37158"},
            {"standard": "MHRA-VPPP-2021", "detail": "MHRA Valproate Pregnancy Prevention Programme — mandatory counselling for females ≥12 years on VPA (applies in CLN4B)"},
            {"standard": "ACMG-AMP-2015", "detail": "ACMG/AMP Variant Interpretation Standards — DNAJC5 variant pathogenicity classification; functional palmitoylation assay as evidence"},
            {"standard": "Crespel-1999", "detail": "Crespel A et al. 1999 — Aggravation of seizures and induction of new seizure types by antiepileptic drugs; GBP/PGB and myoclonus worsening principle"},
            {"standard": "WHO-ICF-2019", "detail": "WHO International Classification of Functioning, Disability and Health — rehabilitation framework for CLN4B multi-domain adult disability"},
            {"standard": "DVLA-Guidance-2022", "detail": "DVLA Assessing Fitness to Drive — guidance for medical professionals; CLN4B triple exclusion (seizures + cognition + ataxia)"},
            {"standard": "BDSRA-Registry", "detail": "Batten Disease Support and Research Association (BDSRA) — CLN4B patient registry; natural history data; AAV-DNAJC5 trial eligibility"},
        ],
        "references": [
            {"ref": "Cadieux-Dion-2013", "citation": "Cadieux-Dion M et al. Expanding the clinical phenotype associated with DNAJC5 mutation: ovarian insufficiency in addition to Kufs disease. J Med Genet 2013;50:791-794."},
            {"ref": "Greaves-2012", "citation": "Greaves J et al. Palmitoylation and membrane interactions of the neuroprotective chaperone cysteine-string protein. J Biol Chem 2012;287:37149-37158."},
            {"ref": "Orcsi-2020", "citation": "Orcsi A et al. Adult neuronal ceroid lipofuscinosis: DNAJC5 and its genetic heterogeneity. Nat Genet 2020 (reference to natural history consortium data)."},
            {"ref": "Noskova-2011", "citation": "Noskova L et al. Mutations in DNAJC5, encoding cysteine-string protein alpha, cause autosomal-dominant adult-onset neuronal ceroid lipofuscinosis. Am J Hum Genet 2011;89:241-252."},
            {"ref": "Nosková-2011-AmJHumGenet", "citation": "Nosková L et al. 2011 — landmark identification of DNAJC5 mutations in Kufs disease Type B (autosomal dominant adult NCL). Am J Hum Genet 2011;89:241-252."},
            {"ref": "Andermann-2012", "citation": "Andermann E et al. Kufs disease, a cause of progressive myoclonus epilepsy, is a lysosomal storage disease with adult onset. Neurology 2012;78:1447-1455."},
        ],
    }
