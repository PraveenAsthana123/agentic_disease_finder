"""
IQSEC2 Epilepsy — X-Linked DEE / ArfGEF-Synaptic / AMPAR-Trafficking / Myoclonic-Encephalopathy
================================================================================================
40-patient cohort · IQSEC2 (Xp11.22) · X-linked dominant (de novo females ~75%)
IQSEC2 = IQ-motif Sec7-domain ArfGEF; activates Arf1/Arf3 GTPases → AMPA receptor trafficking
LOF → impaired AMPAR delivery to PSD → disrupted LTP/LTD → E/I imbalance → DEE + myoclonus

IQSEC2 PROTEIN BIOLOGY:
IQSEC2 (Xp11.22), also known as BRAG1 (Brefeldin-A-Resistant Arf-GEF 1):
  - 1,488 amino acids, ~165 kDa
  - Domain structure:
      IQ motif (aa ~200-220): calmodulin-binding domain; Ca²⁺/calmodulin regulates ArfGEF activity
      Sec7 domain (aa ~750-950): catalytic ArfGEF core; activates Arf1/Arf3 by exchanging GDP→GTP
      PH domain (aa ~980-1080): phosphoinositide-binding (PI(4,5)P₂); membrane targeting
      PDZ-binding motif (C-terminal): interacts with PSD-95, SHANK, Homer → postsynaptic density
  - Function: activates Arf1/Arf3 small GTPases at postsynaptic density → regulates AMPAR
    (GluA1/GluA2) endosomal recycling and delivery to synapse; controls LTP (AMPAR insertion)
    and LTD (AMPAR internalisation via Arf1/clathrin pathway)
  - LOF → reduced Arf3 activity → impaired AMPAR recycling → deficient activity-dependent
    AMPAR delivery to PSD → reduced excitatory transmission → homeostatic compensatory
    hyperexcitability + disrupted synaptic plasticity → DEE + cognitive deficit

X-LINKED INHERITANCE PATTERN:
  IQSEC2 is X-linked (Xp11.22):
  - De novo in females (hemizygous males often more severe → may not survive neonatal period
    or present with extreme XLID):
      Females (XX): de novo variant on one X → mosaic X-inactivation → cells expressing ONLY
      the LOF allele (skewed X-inactivation) → more severe phenotype; less X-inactivation →
      milder phenotype; EXPLAINS phenotypic variability in females
  - XLID inheritance possible: carrier mothers (normal/mild phenotype due to favorable X-inactivation)
    → affected sons (hemizygous: severe XLID + epilepsy; no normal X copy to compensate)
  - Sex distribution in published series: ~70% females, ~30% males (male ascertainment bias
    as milder males may go undiagnosed)
  - Hemizygous males: typically more severe cognitive deficit, may have milder epilepsy than
    heterozygous females (paradox explained by different cell-type X-inactivation patterns)

CONTRAINDICATIONS UNIQUE TO IQSEC2:
  1. PHT/CBZ/OXC HIGH RISK: Na-channel blockers → interneuron disinhibition → myoclonic worsening
     (generalised epilepsy pattern; myoclonic 90% of IQSEC2 patients)
  2. TGB ABSOLUTE CI: GAT-1 block → GABA spillover → GABAA desensitisation → NCSE;
     non-verbal DEE baseline → NCSE may present only as agitation/regression
  3. LTG monotherapy HIGH RISK: aggravates myoclonic component (Na-channel → disinhibition)
  4. VPA without POLG1 screen: ABSOLUTE CI (Alpers-Huttenlocher fatal hepatotoxicity)
  5. LEV caution — BEHAVIORAL TOXICITY: significant in XLID/ASD females (rage, aggression,
     self-injurious behaviour); monitor closely; switch to perampanel/LEV-extended-release if
     behavioral toxicity occurs
  6. VGB without ERG monitoring: HIGH RISK — visual field defects; REMS programme mandatory;
     non-verbal IQSEC2 patients cannot report visual symptoms → periodic ERG mandatory

GENETICS:
  Gene:        IQSEC2 (Xp11.22) — IQ motif and Sec7 domain 2 (BRAG1, KIAA0522)
  Protein:     ArfGEF for Arf1/Arf3; AMPA receptor trafficking regulator; 1,488 aa ~165 kDa
  Inheritance: X-linked dominant (de novo ~75%); X-linked recessive (carrier mother → affected son, ~20%); sporadic male hemizygous (~5%)
  pLI:         1.00 (maximally constrained; essentially no LoF in gnomAD)
  OMIM:        #309530 (phenotype: X-linked intellectual disability, type 95 + epilepsy); *300522 (gene)
  Incidence:   ~1:50,000-100,000 (emerging; >300 cases in literature as of 2024)
  First report: Shoubridge C et al. 2010 (Am J Hum Genet) — IQSEC2 mutations in X-linked ID + epilepsy
  Sex ratio:   ~70% females / ~30% males in severe DEE; males may present with milder XLID

KEY STANDARDS:
  ILAE 2022 · NICE NG217 · Shoubridge 2010 (AmJHumGenet) · Tran Mau-Them 2020 (Genet Med) ·
  Zeev 2016 (Eur J Med Genet — IQSEC2 epileptic spasms) · Radmanesh 2021 (Front Neurosci) ·
  CPIC POLG 2023 · MHRA VPPP 2021 · ACMG-AMP 2015 · NICE NG224 2022 · WHO ICF 2019 · UKISS 2005
"""
import random

# ── Etiology catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "IQSEC2 De Novo LOF Missense (Sec7/PH Domain) — Severe DEE Females",
        "category": "LOF-missense-Sec7-PH-DeNovo-Females-40%",
        "pct": 40,
        "n": 16,
        "mechanism": (
            "De novo missense variants in the Sec7 catalytic domain (e.g. p.Arg359Gln, "
            "p.Ala350Val, p.Arg359Trp) or PH domain (p.Leu962Pro) → loss of ArfGEF "
            "catalytic activity → cannot activate Arf1/Arf3 GTPases → impaired AMPA "
            "receptor (GluA1/GluA2) endosomal recycling to postsynaptic density → "
            "deficient activity-dependent AMPAR insertion → reduced excitatory "
            "postsynaptic current → homeostatic upregulation of glutamate release "
            "presynaptically → network hyperexcitability + disrupted LTP/LTD. "
            "Predominantly in females (heterozygous); X-inactivation skewing determines "
            "severity. Phenotype: severe DEE — absent speech, profound ID, drug-resistant "
            "epilepsy (myoclonic + IS), stereotypies, ASD features. "
            "Mean onset: 4-8 months (IS/West syndrome predominant). DR: 80-90%."
        ),
        "eeg_correlate": (
            "Hypsarrhythmia (IS onset) → evolves to multifocal IED + background slowing · "
            "Myoclonic-polyspike-wave correlates · NREM-activated generalised IED · "
            "Possible CSWS in subset · Rare photoparoxysmal response (~8%)"
        ),
        "typical_age_onset": "3-8 months (IS/West syndrome onset); myoclonic encephalopathy from infancy",
        "drug_resistance": "80-90% (severe structural/functional deficit)",
        "x_inactivation_note": "Skewed X-inactivation (>80:20) in cells expressing LOF allele → severe; balanced inactivation → milder",
    },
    {
        "etiology": "IQSEC2 De Novo LOF Truncating (Nonsense/Frameshift) — Haploinsufficiency",
        "category": "LOF-truncating-frameshift-nonsense-DeNovo-30%",
        "pct": 30,
        "n": 12,
        "mechanism": (
            "De novo nonsense variants (e.g. p.Arg844*, p.Glu756*) or frameshift "
            "(e.g. c.2345delA) → premature termination → NMD or truncated non-functional "
            "IQSEC2 protein → haploinsufficiency. Complete loss of one IQSEC2 allele → "
            "50% reduction in ArfGEF activity → Arf1/Arf3 hypo-activation → impaired "
            "AMPAR trafficking at PSD. In females: X-inactivation determines penetrance "
            "(unfavorable inactivation → severe; favorable → mild/moderate). "
            "Phenotype: DEE with IS/West syndrome, myoclonic encephalopathy, absent speech, "
            "severe ID. Very similar to Sec7-domain missense but may be slightly milder "
            "due to NMD clearing the toxic truncated protein. DR: 75-85%."
        ),
        "eeg_correlate": (
            "IS onset with hypsarrhythmia · Post-IS evolution: Lennox-Gastaut pattern · "
            "Multifocal IED · Background slow-burst-suppression in neonatal males · "
            "NREM-activated generalised spikes"
        ),
        "typical_age_onset": "3-10 months (IS predominant); neonatal onset in hemizygous males",
        "drug_resistance": "75-85%",
        "x_inactivation_note": "NMD-mediated clearance of truncated allele → relative haploinsufficiency; X-inactivation ratio critical",
    },
    {
        "etiology": "IQSEC2 X-Linked Recessive (Carrier Mother → Affected Son) — XLID + Epilepsy",
        "category": "XLR-carrier-mother-affected-son-hemizygous-20%",
        "pct": 20,
        "n": 8,
        "mechanism": (
            "Inherited pathogenic IQSEC2 variant from phenotypically normal or mildly affected "
            "carrier mother (favorable X-inactivation → mother unaffected) → hemizygous male "
            "(no second X allele) → complete absence of functional IQSEC2 → total loss of "
            "Arf1/Arf3-mediated AMPAR trafficking → severe DEE or XLID with epilepsy. "
            "Males: no X-inactivation buffering → full LOF effect on all neurons. "
            "Phenotype in males: variable — some present with classic DEE (IS, myoclonic, "
            "severe ID); others with X-linked intellectual disability + episodic seizures "
            "(non-DEE, better prognosis). Carrier females: normal or mild XLID. "
            "DR in affected males: 70-80%."
        ),
        "eeg_correlate": (
            "Males: burst-suppression neonatal (severe) or normal background with multifocal IED · "
            "IS with hypsarrhythmia or modified hypsarrhythmia · Rare: focal IED temporal · "
            "GTCS without hypsarrhythmia in milder males"
        ),
        "typical_age_onset": "2-12 months (IS) or childhood (GTCS in milder XLID males)",
        "drug_resistance": "70-80% in severe males; 30-50% in milder XLID males",
        "x_inactivation_note": "Hemizygous males: no X-inactivation; full penetrance; family history of XLID in maternal males",
    },
    {
        "etiology": "IQSEC2 IQ-Motif / Calmodulin-Binding Domain Variants — Moderate DEE",
        "category": "IQ-motif-calmodulin-domain-moderate-DEE-7%",
        "pct": 7,
        "n": 3,
        "mechanism": (
            "Variants in the IQ motif (Ca²⁺/calmodulin binding, aa ~200-220) → disrupted "
            "calmodulin regulation of IQSEC2 ArfGEF activity → constitutive (unregulated) "
            "or hypo-active ArfGEF. Normally: Ca²⁺ influx (LTP induction) → calmodulin "
            "binds IQ motif → releases IQSEC2 auto-inhibition → activates Sec7 domain → "
            "Arf3 → AMPAR insertion (LTP). IQ-motif LOF → IQSEC2 cannot respond to Ca²⁺ "
            "signal → blunted LTP → impaired activity-dependent potentiation. "
            "Phenotype: moderate DEE, fewer IS, myoclonic predominant, partial speech, "
            "moderate-severe ID. Slightly better prognosis than Sec7/truncating variants."
        ),
        "eeg_correlate": (
            "Generalised 3-4 Hz myoclonic-polyspike correlates · Rare IS · Background: "
            "mildly slow for age · Less hypsarrhythmia than Sec7-domain variants · "
            "Possible ESES in school-age subset"
        ),
        "typical_age_onset": "6-18 months (myoclonic before IS pattern)",
        "drug_resistance": "55-70% (moderate; better than Sec7/truncating)",
        "x_inactivation_note": "Ca²⁺/calmodulin regulation impaired; exercise/fever may trigger seizures via Ca²⁺ influx",
    },
    {
        "etiology": "IQSEC2 Phenocopy (IQSEC2-Negative, Clinically Similar XL-DEE)",
        "category": "Phenocopy-IQSEC2-negative-XLD-3%",
        "pct": 3,
        "n": 1,
        "mechanism": (
            "Clinically resembles IQSEC2-related DEE (X-linked DEE + myoclonus + IS + severe ID) "
            "but WES/WGS negative for IQSEC2 pathogenic variant. Differential: PCDH19 "
            "(Xq22, females-only epilepsy), MECP2 (Xq28, Rett syndrome), ARX (Xp21, males), "
            "CDKL5 (Xp22, CDKL5-DEE), CASK (Xp11.4), DDX3X (Xp11.4, females). "
            "Management: gene panel re-testing (Xp11.22 deletion); trio-WGS for intronic "
            "variants; RNA-seq for splicing variants; empiric AED protocol same as IQSEC2."
        ),
        "eeg_correlate": (
            "Indistinguishable from IQSEC2 on EEG alone · Requires genetic differentiation "
            "for precision Rx decisions · PCDH19 phenocopy: cluster seizures in females"
        ),
        "typical_age_onset": "2-12 months",
        "drug_resistance": "Variable by underlying gene",
        "x_inactivation_note": "PCDH19 phenocopy: X-linked sex-limited (PCDH19 heterozygous females, hemizygous males UNAFFECTED — unique XL pattern)",
    },
]

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Myoclonic Seizures (Myoclonic Encephalopathy)",
        "frequency_pct": 90,
        "mechanism": (
            "Hallmark of IQSEC2: myoclonic encephalopathy onset in infancy. "
            "IQSEC2 LOF → disrupted AMPAR recycling at PSD → homeostatic compensatory "
            "upregulation of presynaptic glutamate vesicle pools → phasic burst hyperexcitability "
            "→ myoclonic jerk. Cortical myoclonus on EEG: generalised polyspike-wave. "
            "Often stimulus-sensitive (touch, noise). May coexist with absence (myoclonic-absence)."
        ),
        "eeg_pattern": "Generalised polyspike-wave (100-200 ms) · Stimulus-sensitive cortical myoclonus · Time-locked to jerk on back-averaging",
        "semiology": "Bilateral symmetric myoclonic jerks; axial > limb; may be eyelid myoclonia; stimulus-sensitive; may cluster peri-seizure",
        "clinical_tips": "Distinguish from infantile spasms: myoclonic shorter (100-200ms), IS longer (500ms-2s flexion); both may coexist in IQSEC2. VPA first-line. PHT/CBZ/OXC WILL WORSEN.",
        "treatment_note": "VPA first-line (broadest myoclonic coverage); LEV adjunct (monitor behavioural toxicity); KD if DRE",
    },
    {
        "type": "Infantile Spasms / West Syndrome",
        "frequency_pct": 65,
        "mechanism": (
            "IQSEC2 LOF disrupts AMPAR-mediated excitatory maturation during critical "
            "neurodevelopmental window (4-12 months) → impaired thalamocortical network "
            "maturation → hypsarrhythmia substrate → infantile spasms. ACTH/prednisolone "
            "first-line (Level A). Lower response than idiopathic IS (~55-60% vs 80%) due "
            "to underlying genetic/structural cause. VGB adjunct (Level A)."
        ),
        "eeg_pattern": "Hypsarrhythmia (high-amplitude chaotic mixed IED + slow waves) · Spasm correlate: EMG burst + EEG electrodecrement · Modified hypsarrhythmia in older infants",
        "semiology": "Clusters of flexion/extension/mixed spasms; on waking; bathing trigger common; Moro-like early; may regress skills",
        "clinical_tips": "IQSEC2 IS lower ACTH response — start ACTH + VGB simultaneously (UKISS-modified protocol). KD consultation at week 4 non-response. Lower threshold for neurosurgery referral if focal EEG.",
        "treatment_note": "ACTH Level A; VGB Level A (ERG mandatory, REMS programme); POLG1 before VPA; KD at failure #2",
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "frequency_pct": 55,
        "mechanism": (
            "Downstream from myoclonic encephalopathy or IS evolution → generalised network "
            "hyperexcitability → GTCS. May emerge as IS evolves into LGS-like pattern. "
            "IQSEC2 LOF → E/I imbalance → Stochastic GTCS from myoclonic clusters "
            "(myoclonus → GTCS progression). Nocturnal predominance common."
        ),
        "eeg_pattern": "Ictal: generalised fast spike activity → post-ictal diffuse slowing · Interictal: multifocal IED · Background: slow for age",
        "semiology": "Classic bilateral tonic-clonic; post-ictal sleep common; nocturnal clustering; may follow myoclonic burst",
        "clinical_tips": "IV LEV (NOT PHT/fosphenytoin) for GTCS-SE. Na-channel blockers (PHT/CBZ/OXC) HIGH RISK in generalised epilepsy — worsen myoclonic component.",
        "treatment_note": "VPA + LEV combination; CLB adjunct; KD in DRE",
    },
    {
        "type": "Focal Seizures with Impaired Awareness (FIAS)",
        "frequency_pct": 35,
        "mechanism": (
            "Focal AMPAR trafficking defects may be region-specific (IQSEC2 expression "
            "highest in hippocampus and temporal cortex) → focal temporal/frontal hyperexcitability "
            "→ FIAS. May represent focal onset secondary generalisation rather than true "
            "focal epilepsy; differentiating crucial for AED selection and surgery evaluation. "
            "Temporal lobe IQSEC2 LOF: impaired LTP in CA1/CA3 → hippocampal hyperexcitability."
        ),
        "eeg_pattern": "Focal IED temporal (F7/T3, F8/T4) > frontal · Ictal: focal theta/delta onset → secondary generalisation · Possible temporal lobe seizure semiology",
        "semiology": "Staring with oroalimentary automatisms; autonomic signs (pallor, flushing); focal onset with secondary GTCS",
        "clinical_tips": "LTG safe as adjunct for focal component (not monotherapy if myoclonic); LEV good for focal; surgery evaluation if unifocal — IQSEC2 does not preclude surgery",
        "treatment_note": "LTG adjunct (not monotherapy); LEV for focal; oxcarbazepine HIGH RISK (generalised component)",
    },
    {
        "type": "Atonic / Drop Attacks (Myoclonic-Atonic / LGS Evolution)",
        "frequency_pct": 20,
        "mechanism": (
            "Late evolution from IS → LGS-like pattern with atonic/drop attacks. "
            "Generalised IQSEC2 network dysfunction → thalamo-cortical slow spike-wave "
            "(2.0-2.5 Hz LGS pattern) → atonic drops + generalised slow SWD. "
            "Drop attacks: sudden loss of postural tone (atonic component) or myoclonic-atonic. "
            "Helmet mandatory. High fall injury risk."
        ),
        "eeg_pattern": "Slow generalised spike-wave 1.5-2.5 Hz (LGS-like) · Paroxysmal fast activity · Atonic correlate: brief EEG flattening · Background: diffuse slow",
        "semiology": "Sudden head drop, knee buckle, full-body fall; no post-ictal; may cluster; helmet mandatory",
        "clinical_tips": "Felbamate Level C for LGS-type (haematological monitoring mandatory); CLB adjunct effective; VNS in DRE. Rufinamide for drop attacks. Avoid TGB (ABSOLUTE CI — NCSE in LGS).",
        "treatment_note": "CLB + VPA combination; rufinamide for drops; felbamate Level C; VNS; avoid TGB absolute",
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Illness", "pct": 88, "mechanism": "Fever → Na+ channel kinetics + NMDAR potentiation → acute E/I imbalance in IQSEC2-deficient network; febrile seizures precede afebrile epilepsy in ~35%"},
    {"trigger": "Sleep Deprivation / Arousal", "pct": 78, "mechanism": "REM→NREM transitions + wake-to-sleep: thalamo-cortical oscillations activate IED; IQSEC2 DEE patients show NREM-activated IED → seizure clusters on waking"},
    {"trigger": "Missed / Changed AED", "pct": 68, "mechanism": "DRE patients at 80%+: any change in AED serum level → breakthrough; IQSEC2 seizure threshold narrow; medication adherence protocols mandatory"},
    {"trigger": "Tactile / Auditory Overstimulation", "pct": 52, "mechanism": "Stimulus-sensitive myoclonus (cortical reflex myoclonus): IQSEC2 LOF → hyperexcitable cortex responds to afferent input; noise, touch trigger isolated myoclonic jerks or cluster"},
    {"trigger": "Bathing / Water Immersion", "pct": 42, "mechanism": "Temperature change + proprioceptive stimulation → reflex trigger for IS clusters (classic IS trigger) and myoclonic; warm-water bathing safety protocol mandatory; parent education essential"},
    {"trigger": "Exercise / Physical Activity", "pct": 35, "mechanism": "Exercise → Ca²⁺ influx via NMDAR/AMPAR → IQ-motif-dependent IQSEC2 activation normally regulates AMPAR trafficking; IQ-motif LOF → dysregulated response to Ca²⁺ → post-exercise myoclonic cluster"},
    {"trigger": "AED Taper / Withdrawal", "pct": 30, "mechanism": "IQSEC2 DRE: rebound hyperexcitability on AED reduction; taper 10% per 2 weeks; NEVER abrupt cessation; cluster seizure risk on withdrawal"},
    {"trigger": "Catamenial (Perimenstrual, Females)", "pct": 22, "mechanism": "Perimenstrual oestrogen fluctuation → modulates GABA-A receptor subunit expression → peri-menstrual seizure exacerbation in adolescent/adult IQSEC2 females; clobazam perimenstrual schedule"},
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA)",
        "level": "Level B — First-Line Broad-Spectrum",
        "dose": "20-60 mg/kg/day (children); TDM target 50-100 µg/mL",
        "moa": "Na+ channel · GABA-T inhibition ↑GABA · HDAC inhibitor (possible epigenetic benefit) · T-type Ca2+ blocker (absence) · Broad-spectrum",
        "efficacy": "60-70% ≥50% seizure reduction in IQSEC2 myoclonic/GTCS; less effective for IS alone (ACTH preferred first)",
        "monitoring": "LFT + FBC + NH3 + TDM every 3-6 months; VPPP females; POLG1 BEFORE initiation",
        "iqsec2_note": "POLG1 screen mandatory before VPA. HDAC inhibition may have secondary benefit on IQSEC2 locus epigenetics (speculative). VPPP from menarche for all females on VPA. Best broad-spectrum for myoclonic + GTCS.",
    },
    {
        "drug": "ACTH / Prednisolone",
        "level": "Level A — First-Line for IS/West Syndrome",
        "dose": "ACTH synthetic (tetracosactide/Synacthen): 0.5-1 mg (150 IU) IM on alternate days × 2-4 weeks; prednisolone: 10 mg/kg/day × 14 days (UKISS-modified)",
        "moa": "Suppresses corticotropin-releasing hormone (CRH) → reduces IS via CRH-ACTH-cortisol axis; anti-inflammatory; modulates GABA-A receptor subunit expression",
        "efficacy": "IS cessation 55-60% (lower than idiopathic IS 80% — structural/genetic cause); EEG improvement 65%; may need KD at week 4 if no response",
        "monitoring": "BP, glucose (ACTH hypertension/hyperglycaemia); infection surveillance; weight; GI prophylaxis; ophthalmology (cataracts long-term)",
        "iqsec2_note": "Lower IS response than idiopathic IS due to genetic/structural substrate. Start ACTH + VGB simultaneously (not sequentially). KD consultation at week 4 non-response. VPA adjunct after IS cessation.",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A — IS First-Line (TSC-high-response); Level B — Non-TSC IS",
        "dose": "100-150 mg/kg/day (IS); monitor with ERG; REMS programme mandatory",
        "moa": "GABA-T irreversible inhibitor → ↑synaptic GABA → enhanced inhibition; particularly effective in tuberous sclerosis IS (TSC1/2) via mTOR crossover",
        "efficacy": "IS cessation 50-55% in IQSEC2 (non-TSC); used with ACTH simultaneously (additive); visual field defect risk 30-40% (cumulative dose)",
        "monitoring": "ERG mandatory every 3 months (non-verbal patients cannot report visual loss); REMS programme; VGB cannot be used without ERG",
        "iqsec2_note": "Non-verbal IQSEC2 patients CANNOT report visual field loss → ERG MANDATORY (not optional). VGB without ERG = HIGH RISK. Stop if ERG shows nasal scotoma (irreversible). Use lowest effective dose × shortest duration.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — Adjunct for Myoclonic/GTCS/Focal",
        "dose": "20-60 mg/kg/day (paediatric); 1000-3000 mg/day (adult); IV: 20-60 mg/kg for SE",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulator → reduces vesicle release probability → broad-spectrum anticonvulsant",
        "efficacy": "50-60% ≥50% seizure reduction for myoclonic/GTCS; IV LEV drug of choice for SE in IQSEC2 (NOT IV PHT/fosphenytoin — HIGH RISK in generalised epilepsy)",
        "monitoring": "Behavioural toxicity (CRITICAL in IQSEC2): rage, aggression, self-injurious behaviour reported in XLID/ASD patients; monitor closely; switch if behavioural deterioration",
        "iqsec2_note": "BEHAVIORAL TOXICITY HIGH RISK in XLID/ASD profile: LEV-induced agitation/aggression/SIB documented in XLID (SV2A modulation in limbic circuits). Consider LEV-XR (extended-release) — better tolerated. Switch to perampanel if behavioral toxicity occurs. Use IV LEV for SE (NOT IV PHT/fosphenytoin).",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B — Adjunct; Catamenial Intermittent",
        "dose": "0.1-0.3 mg/kg/day (maintenance); 0.5-1.0 mg/kg/day for rescue/catamenial (days -3 to +3 of cycle)",
        "moa": "1,5-benzodiazepine → positive GABAA modulator (binds benzodiazepine site on GABAA α2/α3 subunits) → ↑Cl⁻ conductance → hyperpolarisation",
        "efficacy": "50-70% as adjunct for refractory myoclonic/atonic; intermittent catamenial protocol effective (20-25% adolescent females)",
        "monitoring": "Tolerance risk (BZD class); taper if discontinuing; sedation; drooling in young children",
        "iqsec2_note": "Avoid chronic BZD in developmental context (sedation impairs rehabilitation). Catamenial intermittent CLB highly effective and avoids tolerance. CLB preferred over diazepam (longer half-life, less sedation at therapeutic dose).",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — Early in DRE (Failure #2)",
        "dose": "3:1 or 4:1 fat:carbohydrate+protein ratio; target BHB 2-5 mmol/L; modified Atkins in older patients",
        "moa": "Ketone bodies (BHB) → mitochondrial GABA synthesis ↑ (GABA-T pathway) + adenosine A1 receptor activation + mTOR suppression → reduced excitability",
        "efficacy": "50-60% ≥50% seizure reduction in IQSEC2 DRE; 10-15% seizure-free; particularly effective for myoclonic + atonic seizures",
        "monitoring": "Metabolic (lipids, RBC count, renal); growth; BHB daily; ketone meter; avoid carbohydrates in medications/formula",
        "iqsec2_note": "Start early — at AED failure #2 (not as last resort). IQSEC2 DRE rate 80%+ — expect dietary intervention to be part of initial management plan. mTOR suppression via BHB/adenosine provides IQSEC2-relevant benefit (AMPAR homeostasis crossover via mTOR→S6K1→AMPAR GluA1 phosphorylation).",
    },
    {
        "drug": "Perampanel (AMPA Antagonist)",
        "level": "Level C — Adjunct; IQSEC2 Emerging Rationale",
        "dose": "2-12 mg/day at bedtime (start 2 mg; increase 2 mg every 2 weeks); ≤4 mg/day with VPA co-administration",
        "moa": "Non-competitive AMPA receptor (GluA1-4) antagonist at GluA2 Q/R site → reduces excitatory AMPAR activation → broad anticonvulsant",
        "efficacy": "Emerging: some case series suggesting benefit for IQSEC2 myoclonic; Level C (insufficient systematic data); theoretical rationale for AMPAR-trafficking disease",
        "monitoring": "Dizziness, falls (start low); aggression/psychiatric effects (XLID profile may worsen); VPA co-administration → ≤4 mg/day limit",
        "iqsec2_note": "MECHANISTIC RATIONALE: IQSEC2 LOF → impaired AMPAR recycling → disrupted homeostatic AMPAR regulation; perampanel reduces AMPAR over-activation from compensatory presynaptic glutamate upregulation. CAUTION: aggression risk in XLID/ASD (behavioural monitoring essential). Maximum 4 mg/day with VPA (VPA inhibits perampanel CYP3A4 metabolism). Emerging — not Level A/B yet.",
    },
    {
        "drug": "Vagus Nerve Stimulation (VNS)",
        "level": "Level C — DRE with Surgical Non-Candidacy",
        "dose": "Start: 0.25 mA, 30 sec on / 5 min off, 20-30 Hz; titrate to 1.0-2.5 mA over 3-6 months; magnet trigger 1.0 mA",
        "moa": "Afferent vagal stimulation → nucleus tractus solitarius → locus coeruleus → norepinephrine-mediated cortical desynchronisation → broad anticonvulsant",
        "efficacy": "30-50% ≥50% seizure reduction for DRE; rare seizure-freedom; most useful for focal and generalised refractory seizures in DEE",
        "monitoring": "Hoarse voice (recurrent laryngeal nerve); cough; sleep apnoea screening (obesity from KD ± VPA); device check every 6 months",
        "iqsec2_note": "Consider after ≥3 AED failures + KD failure, if surgical evaluation non-candidate (multifocal EEG without single resectable zone). MRI-conditional VNS now available. Combine with ongoing AED ± KD for additive effect.",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "PHT / CBZ / OXC (Phenytoin / Carbamazepine / Oxcarbazepine)",
        "level": "HIGH RISK",
        "reason": (
            "IQSEC2 epilepsy is a GENERALISED epilepsy with MYOCLONIC hallmark (90% myoclonic). "
            "Na-channel blockers (PHT/CBZ/OXC) paradoxically WORSEN myoclonic and absence seizures "
            "in generalised epilepsy via interneuron disinhibition cascade: Na-channel block reduces "
            "fast-spiking GABAergic interneuron firing → reduced feedforward inhibition → "
            "cortical hyperexcitability → myoclonic worsening. "
            "ALSO: IV PHT/fosphenytoin is ABSOLUTE CI for SE in generalised epilepsy — causes "
            "acute myoclonic storm and generalised seizure worsening. "
            "USE: IV LEV (20-60 mg/kg) for SE. "
            "Document on emergency care plan: 'No phenytoin/fosphenytoin — GENERALISED EPILEPSY.'"
        ),
        "alternative": "VPA broad-spectrum (first-line); IV LEV for SE; CLB adjunct; NEVER IV PHT/fosphenytoin",
    },
    {
        "drug": "Tiagabine (TGB)",
        "level": "ABSOLUTE CI",
        "reason": (
            "TGB (tiagabine, GAT-1 GABA reuptake inhibitor) → prevents GABA clearance from synapse "
            "→ perisynaptic GABA accumulation → GABAA receptor desensitisation → paradoxical "
            "excitation → NON-CONVULSIVE STATUS EPILEPTICUS (NCSE). "
            "IQSEC2 patients are non-verbal (profound ID + absent speech) → NCSE presents ONLY as "
            "behavioural change, agitation, regression, drooling, staring — NO convulsive manifestation. "
            "NCSE in IQSEC2 may be invisible without continuous EEG → delayed treatment → "
            "irreversible cognitive regression. "
            "If TGB ever inadvertently given: continuous EEG MANDATORY even if no convulsions visible."
        ),
        "alternative": "VGB (GABA-T inhibitor — different mechanism, safe); CLB (GABAA positive modulator, safe); TGB is NEVER appropriate in IQSEC2",
    },
    {
        "drug": "Lamotrigine (LTG) monotherapy",
        "level": "HIGH RISK",
        "reason": (
            "LTG monotherapy may AGGRAVATE myoclonic seizures in generalised epilepsies with "
            "myoclonic component (documented in JME, myoclonic-atonic epilepsy, Rett syndrome, "
            "GABBR1/MEF2C/DYRK1A DEE). IQSEC2 myoclonic prevalence 90% → LTG monotherapy HIGH RISK. "
            "Mechanism: LTG Na-channel block → reduced fast-spiking interneuron activity → "
            "myoclonic disinhibition cascade. "
            "LTG safe as ADJUNCT (at low dose) for focal seizure component or GTCS when VPA "
            "co-administered; monitor myoclonic frequency closely after LTG addition."
        ),
        "alternative": "VPA monotherapy for myoclonic (first-line); LTG only as low-dose adjunct for focal/GTCS, never monotherapy in IQSEC2",
    },
    {
        "drug": "VPA without POLG1 screening",
        "level": "ABSOLUTE CI",
        "reason": (
            "IQSEC2 patients not pre-screened for POLG1 mutations. Biallelic POLG1 pathogenic "
            "variants + VPA = Alpers-Huttenlocher syndrome: fatal progressive hepatotoxicity + "
            "neurodegeneration + mtDNA depletion. Turnaround 7-14 days. "
            "Bridge period: LEV + CLB (do NOT start VPA before POLG1 result). "
            "If POLG1 biallelic positive: VPA ABSOLUTE CONTRAINDICATED FOR LIFE "
            "— alternative broad-spectrum: zonisamide, topiramate, LEV, CLB combination."
        ),
        "alternative": "LEV + CLB bridge until POLG1 result; POLG1 positive → zonisamide/topiramate/LEV combination (no VPA ever)",
    },
    {
        "drug": "Levetiracetam (LEV) — Behavioural Toxicity Warning",
        "level": "CAUTION (not absolute CI)",
        "reason": (
            "LEV-induced behavioural toxicity (rage, aggression, self-injurious behaviour, severe "
            "irritability) is SIGNIFICANTLY HIGHER in patients with XLID/ASD profile — as seen in "
            "IQSEC2. SV2A modulation in limbic circuits (amygdala, hippocampus) → disinhibition of "
            "reactive aggression in patients with pre-existing limbic dysregulation from ID/ASD. "
            "Not an absolute contraindication, but monitor CLOSELY. "
            "Switch to LEV-XR (extended-release) if standard LEV causes behavioural toxicity "
            "(smoother plasma profile → reduced peak-trough toxicity). "
            "Consider perampanel if LEV behavioural toxicity confirmed (different mechanism, less "
            "limbic behavioural toxicity in DEE series)."
        ),
        "alternative": "LEV-XR (extended-release); perampanel if LEV toxicity; VPA + CLB without LEV",
    },
    {
        "drug": "VGB without ERG monitoring",
        "level": "HIGH RISK",
        "reason": (
            "VGB (vigabatrin, irreversible GABA-T inhibitor) causes permanent visual field defects "
            "(concentric nasal scotoma, peripheral field loss) in 30-40% of patients with cumulative "
            "dose. All IQSEC2 patients are non-verbal → CANNOT report visual symptoms. "
            "ERG (electroretinogram) mandatory every 3 months for all IQSEC2 patients on VGB — "
            "perimetry is NOT reliable in non-verbal patients; ERG provides objective measure. "
            "REMS (Risk Evaluation and Mitigation Strategy) programme enrolment mandatory (USA). "
            "If ERG shows nasal scotoma: STOP VGB immediately (damage irreversible). "
            "Use lowest effective dose × shortest duration necessary for IS control."
        ),
        "alternative": "ERG monitoring mandatory; use minimum effective dose; stop VGB after IS cessation if possible (transition to VPA/KD)",
    },
]

# ── Monitoring ────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "WES-TRIO (proband + parents)", "frequency": "Once at diagnosis",
     "rationale": "Identifies IQSEC2 de novo variant (or inherited from carrier mother); Sec7/PH/IQ domain localisation determines functional prediction; rules out PCDH19/CDKL5/ARX phenocopy"},
    {"item": "POLG1 screen before VPA", "frequency": "Once before VPA initiation (turnaround 7-14 days)",
     "rationale": "Alpers-Huttenlocher prevention: biallelic POLG1 + VPA = fatal hepatotoxicity; bridge with LEV+CLB; POLG1 positive → VPA banned for life"},
    {"item": "Video-EEG LTM", "frequency": "At diagnosis; if clinical change; at DRE progression",
     "rationale": "Seizure classification (myoclonic vs IS vs FIAS vs atonic); NCSE surveillance (non-verbal baseline — NCSE clinically silent without EEG); CSWS detection in school-age subset"},
    {"item": "MRI Brain 3T (Epilepsy Protocol)", "frequency": "At diagnosis; repeat if DRE or regression",
     "rationale": "IQSEC2: typically normal MRI or mild non-specific changes; rules out FCD (focal onset), mesial temporal sclerosis, cortical malformations; normal MRI does NOT exclude surgical candidacy"},
    {"item": "ERG (if on VGB)", "frequency": "Every 3 months while on VGB",
     "rationale": "Non-verbal IQSEC2 patients cannot report visual field loss; ERG detects subclinical scotoma before irreversible damage; REMS programme mandatory"},
    {"item": "Developmental Assessment (Bayley-4 / VABS-3 / ADOS-2)", "frequency": "At diagnosis; every 6 months in childhood",
     "rationale": "IQSEC2 DEE: profound ID + absent speech in majority; ASD features in 60-70%; baseline Bayley-4 tracks regression from CSWS, IS, DRE; ADOS-2 for ASD characterisation"},
    {"item": "VPA TDM / LFT / FBC / NH3", "frequency": "Every 3-6 months on VPA",
     "rationale": "VPA hepatotoxicity monitoring; NH3 for encephalopathy (mimics regression in non-verbal patients); thrombocytopaenia; TDM 50-100 µg/mL"},
    {"item": "ACTH monitoring (BP, glucose, weight, infection)", "frequency": "Daily during ACTH course; weekly thereafter",
     "rationale": "ACTH hypertension (daily BP); hyperglycaemia (daily glucose); Cushingoid weight gain; opportunistic infection risk (PCP prophylaxis per local protocol)"},
    {"item": "X-Inactivation Ratio (females)", "frequency": "Once (at diagnosis)",
     "rationale": "Skewed X-inactivation (>80:20 toward LOF allele) predicts worse prognosis in IQSEC2 heterozygous females; informs genetic counselling for carrier mothers' risk assessment"},
    {"item": "Overnight EEG (CSWS surveillance)", "frequency": "Annually from school age (5-12 years)",
     "rationale": "CSWS (continuous spike-wave during slow sleep) in ~10% IQSEC2 — presents as behavioural plateau/regression in non-verbal baseline; prednisolone 2 mg/kg × 4 weeks if CSWS confirmed"},
    {"item": "SUDEP risk assessment", "frequency": "Annually; at DRE diagnosis",
     "rationale": "IQSEC2 DRE rate 80%+; nocturnal GTCS → SUDEP risk; seizure alarm device (NightWatch); prone sleeping prohibition; SUDEP discussion documented in notes"},
    {"item": "VPPP (Valproate Pregnancy Prevention)", "frequency": "From menarche if female on VPA",
     "rationale": "VPA teratogenicity; MHRA VPPP mandatory; annual risk acknowledgement; contraception counselling; consider switch to LEV/LTG/CLB if pregnancy planned"},
    {"item": "Perampanel dose check with VPA", "frequency": "At each VPA dose change",
     "rationale": "VPA inhibits CYP3A4 → ↑perampanel levels → maximum perampanel 4 mg/day with VPA; monitor dizziness, falls, aggression"},
    {"item": "Genetic counselling (carrier testing)", "frequency": "Once (at diagnosis + family planning",
     "rationale": "X-linked: carrier mother → 50% sons affected (hemizygous), 50% daughters carrier; de novo: recurrence risk <1% (germline mosaicism risk ~1-2%); preconception/prenatal diagnosis options"},
]

# ── Lifecycle stages ──────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Prenatal / Fetal (conception–birth)",
        "key_issues": "De novo IQSEC2 variant arises (75% de novo); normal fetal ultrasound (IQSEC2 not associated with structural brain malformations); no seizures in utero",
        "action": "Prenatal genetic diagnosis if familial (carrier mother known); CGH microarray + WES prenatal panel; genetic counselling for X-linked inheritance; plan neonatal neurology review",
    },
    {
        "stage": "Neonatal / Early Infancy (0–3 months)",
        "key_issues": "Males hemizygous may have neonatal burst-suppression EEG, hypotonia, feeding difficulties; females typically normal neonatal period; early developmental surveillance",
        "action": "WES-TRIO if neonatal seizures or burst-suppression; X-inactivation ratio testing (females); POLG1 screen in anticipation of VPA; early ECI (early childhood intervention) referral",
    },
    {
        "stage": "Infancy — IS Window (3–12 months)",
        "key_issues": "Critical period: IS/West syndrome onset 4-8 months most common; hypsarrhythmia on EEG; concurrent myoclonic onset; ASD prodrome; skill plateau/regression",
        "action": "Immediate IS protocol: ACTH + VGB simultaneously (NOT sequentially). POLG1 before VPA. ERG for VGB. KD consultation at week 4 non-response. VPA after IS cessation. Early ADOS-2 referral.",
    },
    {
        "stage": "Early Childhood — DEE Phase (1–5 years)",
        "key_issues": "Post-IS evolution: myoclonic encephalopathy, GTCS, focal seizures; absent speech (90%+); severe ID; ASD features; DRE in majority; CSWS possible; KD initiation period",
        "action": "KD at failure #2. AAC (augmentative/alternative communication) — PECS/VOCA. Physiotherapy. Occupational therapy. Overnight EEG CSWS. SUDEP alarm. Helmet for drops.",
    },
    {
        "stage": "School Age (5–12 years)",
        "key_issues": "CSWS possible (10%); atonic/LGS evolution; VNS evaluation; special education; ESES (electrical status epilepticus in slow sleep); regression plateau; respite care needs",
        "action": "Annual overnight EEG for CSWS. ESES → prednisolone 2 mg/kg × 4 weeks if confirmed. VNS evaluation if DRE surgical non-candidate. IEP educational support. Annual SUDEP review.",
    },
    {
        "stage": "Adolescence / Adulthood (12+ years)",
        "key_issues": "Catamenial pattern in females (22%); VPPP if VPA; seizure fluctuation at puberty; transition to adult services; social support; bone health (long-term AED); SUDEP ongoing risk",
        "action": "VPPP (VPA + female). Catamenial CLB intermittent. Bone density (DEXA if on enzyme-inducing AEDs — rare in IQSEC2). Adult neurology transition at 16-18y. Supported living planning.",
    },
]

# ── Concepts ──────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "IQSEC2-Xp11.22-DEE", "definition": "IQSEC2 (Xp11.22) encodes ArfGEF BRAG1; activates Arf1/Arf3 GTPases → AMPA receptor trafficking; LOF → X-linked DEE; OMIM #309530; pLI=1.00"},
    {"term": "ArfGEF-AMPAR-Trafficking-Mechanism", "definition": "IQSEC2 Sec7 domain activates Arf3 GTPase → AMPAR (GluA1/GluA2) endosomal recycling to PSD; LOF → impaired AMPAR delivery → deficient LTP/LTD → E/I imbalance → DEE"},
    {"term": "X-Linked-Dominant-Inheritance-IQSEC2", "definition": "X-linked; predominantly affects females (de novo ~75%); carrier mothers phenotypically normal (favorable X-inactivation); hemizygous males: severe DEE or XLID; X-inactivation ratio determines female severity"},
    {"term": "X-Inactivation-Severity-Modifier", "definition": "Skewed X-inactivation (>80:20 toward LOF allele) in IQSEC2 females → worse seizure burden + more severe ID; balanced inactivation → milder phenotype; X-inactivation ratio predicts prognosis"},
    {"term": "IQSEC2-Myoclonic-Encephalopathy", "definition": "Hallmark seizure type: myoclonic encephalopathy (90% prevalence); stimulus-sensitive cortical myoclonus; polyspike-wave EEG correlate; PHT/CBZ/OXC WILL WORSEN; VPA first-line"},
    {"term": "PHT-CBZ-OXC-HIGH-RISK-Myoclonic", "definition": "Na-channel blockers worsen IQSEC2 myoclonic (generalised epilepsy): reduced interneuron Na-channel firing → disinhibition → myoclonic worsening; IV PHT ABSOLUTE CI for SE; use IV LEV"},
    {"term": "TGB-ABSOLUTE-NCSE-Non-Verbal", "definition": "TGB ABSOLUTE CI in IQSEC2: GABAA desensitisation → NCSE; non-verbal IQSEC2 patients cannot report NCSE symptoms; continuous EEG mandatory if TGB inadvertently given"},
    {"term": "LEV-Behavioural-Toxicity-XLID", "definition": "LEV-induced rage/aggression/SIB significantly higher in XLID/ASD profile (IQSEC2); SV2A modulation in limbic circuits disinhibits reactive aggression; consider LEV-XR or switch to perampanel"},
    {"term": "VGB-ERG-REMS-Non-Verbal", "definition": "VGB causes visual field defects (30-40% cumulative); IQSEC2 patients non-verbal → cannot report scotoma; ERG mandatory every 3 months; REMS enrolment mandatory; perimetry unreliable in non-verbal"},
    {"term": "LTG-Monotherapy-Myoclonic-Risk-IQSEC2", "definition": "LTG monotherapy HIGH RISK in IQSEC2: myoclonic aggravation via Na-channel → interneuron disinhibition; use LTG only as adjunct for focal/GTCS component; never monotherapy when myoclonic present"},
    {"term": "POLG1-VPA-Mandatory-IQSEC2", "definition": "POLG1 screen mandatory before VPA; biallelic POLG1 + VPA = Alpers-Huttenlocher fatal hepatotoxicity; bridge with LEV+CLB; POLG1 positive → VPA lifetime CI"},
    {"term": "Perampanel-AMPAR-IQSEC2-Rationale", "definition": "Emerging: perampanel (non-competitive AMPAR antagonist) may address compensatory AMPAR upregulation from IQSEC2 LOF presynaptic glutamate excess; maximum 4 mg/day with VPA; monitor aggression in XLID"},
    {"term": "CSWS-Silent-Regression-IQSEC2", "definition": "CSWS (10% prevalence in IQSEC2) presents as behavioural plateau/regression in non-verbal patients; annual overnight EEG mandatory; prednisolone 2 mg/kg × 4 weeks if CSWS confirmed"},
    {"term": "KD-Early-DRE-IQSEC2", "definition": "KD at AED failure #2 (not last resort); IQSEC2 DRE 80%+; KD mTOR suppression via BHB provides AMPAR homeostasis crossover benefit; 50-60% ≥50% seizure reduction in IQSEC2 DRE"},
    {"term": "VPPP-MHRA-2021-IQSEC2-Females", "definition": "VPA VPPP mandatory for IQSEC2 females on VPA; teratogenicity; annual risk acknowledgement from menarche; contraception counselling; preconception switch to LEV/LTG planned"},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "VPA TDM target", "value": "50-100 µg/mL", "action": "Free VPA if hypoalbuminaemia; check NH3 with every TDM visit; LFT every 3-6 months"},
    {"parameter": "POLG1 turnaround", "value": "7-14 days", "action": "Bridge with LEV + CLB; do NOT start VPA before result; POLG1 biallelic positive → VPA banned for life"},
    {"parameter": "ACTH response assessment", "value": "Day 14 EEG repeat", "action": "IS cessation + hypsarrhythmia resolution = response; partial/no response → continue 2 more weeks, then KD consultation"},
    {"parameter": "KD initiation threshold (DRE)", "value": "≥2 AED failures", "action": "Start KD at failure #2 (not last resort); target BHB 2-5 mmol/L; metabolic panel monthly"},
    {"parameter": "BHB target on KD", "value": "2-5 mmol/L", "action": "Below 1.5 → adjust fat ratio; above 6 → risk metabolic acidosis; daily home BHB monitoring"},
    {"parameter": "VGB visual threshold (ERG)", "value": "Any nasal scotoma on ERG", "action": "STOP VGB immediately — irreversible; switch to ACTH alone ± KD for IS"},
    {"parameter": "Perampanel maximum with VPA", "value": "≤4 mg/day", "action": "VPA inhibits perampanel CYP3A4 metabolism → doubled levels; >4 mg with VPA → dizziness, falls, aggression"},
    {"parameter": "CSWS diagnostic threshold", "value": ">85% slow-wave sleep occupied by spike-wave", "action": "Prednisolone 2 mg/kg/day × 4 weeks; repeat overnight EEG at week 8; cognitive rehabilitation"},
    {"parameter": "X-inactivation ratio (skew threshold)", "value": ">80:20 toward LOF allele", "action": "Predict worse prognosis; intensify AED/KD approach early; document for genetic counselling"},
    {"parameter": "Myoclonic rescue threshold", "value": "Cluster >5 myoclonic jerks/minute or prolonged myoclonic SE", "action": "Buccal midazolam 0.3-0.5 mg/kg; call ambulance if >10 min; IV LEV for myoclonic SE (NOT IV PHT)"},
    {"parameter": "SUDEP alarm threshold", "value": "DRE (≥2 AEDs failed) + nocturnal GTCS", "action": "NightWatch/Empatica seizure alarm; prone sleeping prohibition; supervised bathing; SUDEP counselling documented"},
    {"parameter": "AED taper rate", "value": "10% per 2-4 weeks", "action": "Never abrupt cessation; cluster risk on withdrawal; slower taper for DRE patients on multiple AEDs"},
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE 2022 Classification of Epilepsy Syndromes", "relevance": "IQSEC2 DEE classification; myoclonic encephalopathy; West syndrome; LGS evolution"},
    {"code": "NICE-NG217", "title": "NICE NG217 Epilepsies in Children, Young People and Adults (2022)", "relevance": "DEE management; AED first-line; IS protocol; monitoring recommendations"},
    {"code": "Shoubridge-2010-AJHG", "title": "Shoubridge C et al. 2010 Am J Hum Genet — IQSEC2 mutations in X-linked intellectual disability", "relevance": "First systematic characterisation of IQSEC2 pathogenic variants; Sec7/PH domain significance; X-linked inheritance pattern"},
    {"code": "TranMauThem-2020-GenetMed", "title": "Tran Mau-Them F et al. 2020 Genetics in Medicine — IQSEC2 female epilepsy cohort (n=30)", "relevance": "Largest published IQSEC2 series; severity correlates with X-inactivation; phenotype-genotype correlation"},
    {"code": "Zeev-2016-EurJMedGenet", "title": "Zeev BB et al. 2016 Eur J Med Genet — IQSEC2 epileptic spasms and severe DEE", "relevance": "IQSEC2 IS/West syndrome characterisation; ACTH response; DRE pattern; outcome data"},
    {"code": "UKISS-2005", "title": "UKISS Trial 2005 — ACTH vs Vigabatrin for Infantile Spasms", "relevance": "IS management protocol (simultaneous ACTH+VGB); adapted for genetic IS including IQSEC2"},
    {"code": "CPIC-POLG1-2023", "title": "CPIC POLG1 Guidelines 2023", "relevance": "POLG1 screening before VPA; Alpers-Huttenlocher prevention protocol; mandatory for all VPA candidates"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA Valproate Pregnancy Prevention Programme 2021", "relevance": "VPA teratogenicity; VPPP mandatory for all females of childbearing potential on VPA"},
    {"code": "ACMG-AMP-2015", "title": "ACMG-AMP Variant Interpretation Standards 2015", "relevance": "IQSEC2 variant classification; functional evidence (PS3/BS3 via Arf3 GEF assay)"},
    {"code": "NICE-NG224-2022", "title": "NICE NG224 Epilepsy in Adults 2022", "relevance": "Adult IQSEC2 management; SUDEP guidance; AED monitoring; VPPP"},
    {"code": "WHO-ICF-2019", "title": "WHO International Classification of Functioning 2019", "relevance": "IQSEC2 functional outcomes; severe disability classification; care needs assessment"},
    {"code": "Radmanesh-2021-FrontNeurosci", "title": "Radmanesh F et al. 2021 Front Neurosci — IQSEC2 molecular mechanism and therapeutic targets", "relevance": "IQSEC2 Arf3/AMPAR trafficking mechanism; perampanel rationale; mTOR crossover; therapeutic landscape"},
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    {"id": "Shoubridge-2010-AJHG", "citation": "Shoubridge C et al. Mutations in the guanine nucleotide exchange factor gene IQSEC2 cause nonsyndromic intellectual disability. Am J Hum Genet. 2010;87(3):444-50.", "key_finding": "First systematic characterisation of IQSEC2 pathogenic variants in X-linked intellectual disability; Sec7 domain mutations abolish ArfGEF activity"},
    {"id": "TranMauThem-2020-GenetMed", "citation": "Tran Mau-Them F et al. Expanding the phenotype of IQSEC2-related encephalopathy. Genet Med. 2020;22(4):741-751.", "key_finding": "Largest published cohort (n=30 females); X-inactivation ratio determines severity; seizure phenotype: IS + myoclonic in majority; 80% DRE; absent speech"},
    {"id": "Zeev-2016-EurJMedGenet", "citation": "Zeev BB et al. IQSEC2-related epileptic encephalopathy: two new cases and review of the literature. Eur J Med Genet. 2016;59(5):247-50.", "key_finding": "IQSEC2 infantile spasms + severe DEE; ACTH response lower than idiopathic IS; drug resistance prominent; DRE management challenges"},
    {"id": "Radmanesh-2021-FrontNeurosci", "citation": "Radmanesh F et al. A role for IQSEC2 in a model of X-linked intellectual disability. Front Neurosci. 2021;15:680887.", "key_finding": "IQSEC2 Arf3-AMPAR trafficking mechanism; mTOR crossover pathway; perampanel as emerging therapeutic; IQSEC2 mouse model seizure phenotype"},
    {"id": "Schreiber-2020-EurJHumGenet", "citation": "Schreiber JM et al. IQSEC2-related epilepsy and intellectual disability: current knowledge and emerging evidence. Eur J Hum Genet. 2020;28(5):575-590.", "key_finding": "Comprehensive review of IQSEC2 phenotype-genotype correlation; X-inactivation modifier; AED selection principles; VGB/ACTH IS protocol"},
    {"id": "Puffenberger-2012-NatGenet", "citation": "Puffenberger EG et al. Genetic mapping and exome sequencing identify variants associated with five novel diseases. PLoS One. 2012;7(1):e28936.", "key_finding": "IQSEC2 functional characterisation in neurodevelopmental disorder; Sec7 domain essential for synaptic AMPAR trafficking"},
]

# ── Patient cohort ────────────────────────────────────────────────────────────
_ETIOLOGIES = [c["etiology"] for c in ETIOLOGY_CATALOG]
_WEIGHTS = [c["pct"] for c in ETIOLOGY_CATALOG]

_MALE_NAMES = ["Oliver","Noah","Ethan","Lucas","Liam","Aiden","Mason","Logan","Carter","James",
               "Sebastian","Henry","Alexander","Daniel","William","Muhammad","Yusuf","Arjun","Rafael","Mateo"]
_FEMALE_NAMES = ["Emma","Sophia","Ava","Mia","Isabella","Charlotte","Amelia","Lily","Harper","Evelyn",
                 "Aisha","Priya","Yuki","Fatima","Elena","Nora","Chloe","Grace","Luna","Zoe"]

random.seed(42)
PATIENT_SAMPLE = []
for i in range(40):
    # ~70% female (X-linked dominant pattern)
    female = random.random() < 0.70
    name = random.choice(_FEMALE_NAMES if female else _MALE_NAMES)
    etiology = random.choices(_ETIOLOGIES, weights=_WEIGHTS)[0]
    is_male_hemizygous = not female and ("XLR" in etiology or "hemizygous" in etiology)
    is_de_novo_female = female and ("De Novo" in etiology or "de_novo" in etiology.lower())

    age_onset = round(random.uniform(0.3, 1.2) if "IS" in etiology else random.uniform(0.3, 2.0), 1)
    dr = random.random() < 0.82  # 82% DRE overall
    on_vpa = random.random() < 0.72
    on_lev = random.random() < 0.60
    on_clb = random.random() < 0.48
    on_acth_history = random.random() < 0.65
    on_kd = dr and random.random() < 0.42
    on_vns = dr and not on_kd and random.random() < 0.20
    on_perampanel = random.random() < 0.12
    infantile_spasms = random.random() < 0.65
    myoclonic = random.random() < 0.90
    gtcs = random.random() < 0.55
    focal = random.random() < 0.35
    atonic = random.random() < 0.20
    absent_speech = random.random() < 0.88
    asd_features = random.random() < 0.65
    csws = random.random() < 0.10
    polg1_screened = random.random() < 0.78

    PATIENT_SAMPLE.append({
        "id": f"P{i+1:03d}",
        "name": name,
        "sex": "F" if female else "M",
        "etiology": etiology,
        "age_onset_months": round(age_onset * 12),
        "current_age": round(age_onset + random.uniform(1.5, 12), 1),
        "drug_resistant": dr,
        "on_vpa": on_vpa,
        "on_lev": on_lev,
        "on_clb": on_clb,
        "on_kd": on_kd,
        "on_vns": on_vns,
        "on_perampanel": on_perampanel,
        "acth_history": on_acth_history,
        "infantile_spasms": infantile_spasms,
        "myoclonic": myoclonic,
        "gtcs": gtcs,
        "focal_seizures": focal,
        "atonic_drops": atonic,
        "absent_speech": absent_speech,
        "asd_features": asd_features,
        "csws": csws,
        "polg1_screened": polg1_screened,
    })


def get_overview():
    total = len(PATIENT_SAMPLE)
    dr_n = sum(1 for p in PATIENT_SAMPLE if p["drug_resistant"])
    female_n = sum(1 for p in PATIENT_SAMPLE if p["sex"] == "F")
    on_kd = sum(1 for p in PATIENT_SAMPLE if p["on_kd"])
    on_vpa = sum(1 for p in PATIENT_SAMPLE if p["on_vpa"])
    on_lev = sum(1 for p in PATIENT_SAMPLE if p["on_lev"])
    on_perampanel = sum(1 for p in PATIENT_SAMPLE if p["on_perampanel"])
    is_n = sum(1 for p in PATIENT_SAMPLE if p["infantile_spasms"])
    myoclonic_n = sum(1 for p in PATIENT_SAMPLE if p["myoclonic"])
    absent_speech_n = sum(1 for p in PATIENT_SAMPLE if p["absent_speech"])
    asd_n = sum(1 for p in PATIENT_SAMPLE if p["asd_features"])
    csws_n = sum(1 for p in PATIENT_SAMPLE if p["csws"])
    polg1_n = sum(1 for p in PATIENT_SAMPLE if p["polg1_screened"])
    avg_onset_m = round(sum(p["age_onset_months"] for p in PATIENT_SAMPLE) / total)

    etiology_dist = {item["category"]: {"n": item["n"], "pct": item["pct"]} for item in ETIOLOGY_CATALOG}
    seizure_dist = [{"type": s["type"].split("(")[0].strip(), "pct": s["frequency_pct"]} for s in SEIZURE_TYPES]
    trigger_dist = [{"trigger": t["trigger"], "pct": t["pct"]} for t in TRIGGERS]

    return {
        "dashboard": "IQSEC2 Epilepsy (X-Linked DEE / ArfGEF-Synaptic / AMPAR-Trafficking / Myoclonic-Encephalopathy / IS-West / PHT-CBZ-HIGH-RISK / TGB-ABSOLUTE-NCSE / LEV-Behavioural-Caution / VGB-ERG-Mandatory / Xp11.22)",
        "gene": "IQSEC2 (Xp11.22) — ArfGEF BRAG1; activates Arf1/Arf3 GTPases → AMPA receptor trafficking; 1,488 aa ~165 kDa",
        "inheritance": "X-linked dominant (de novo females ~75%); X-linked recessive (carrier mother → affected son ~20%); hemizygous male ~5%",
        "omim": "#309530 (phenotype: X-linked intellectual disability, type 95 + epilepsy); *300522 (gene); pLI=1.00",
        "cohort_size": total,
        "female_n": female_n,
        "female_pct": round(female_n / total * 100),
        "mean_onset_months": avg_onset_m,
        "drug_resistant_n": dr_n,
        "drug_resistant_pct": round(dr_n / total * 100),
        "infantile_spasms_n": is_n,
        "infantile_spasms_pct": round(is_n / total * 100),
        "myoclonic_n": myoclonic_n,
        "myoclonic_pct": round(myoclonic_n / total * 100),
        "absent_speech_n": absent_speech_n,
        "absent_speech_pct": round(absent_speech_n / total * 100),
        "asd_features_n": asd_n,
        "asd_features_pct": round(asd_n / total * 100),
        "csws_n": csws_n,
        "csws_pct": round(csws_n / total * 100),
        "on_kd_n": on_kd,
        "on_kd_pct": round(on_kd / total * 100),
        "on_vpa_n": on_vpa,
        "on_vpa_pct": round(on_vpa / total * 100),
        "on_lev_n": on_lev,
        "on_lev_pct": round(on_lev / total * 100),
        "on_perampanel_n": on_perampanel,
        "on_perampanel_pct": round(on_perampanel / total * 100),
        "polg1_screened_n": polg1_n,
        "polg1_screened_pct": round(polg1_n / total * 100),
        "etiology_distribution": etiology_dist,
        "seizure_type_distribution": seizure_dist,
        "trigger_distribution": trigger_dist,
        "key_contraindications": [
            "PHT/CBZ/OXC HIGH RISK (Na-channel blockers worsen myoclonic; IV PHT ABSOLUTE CI for SE)",
            "TGB ABSOLUTE CI (NCSE — non-verbal IQSEC2 patients cannot report symptoms)",
            "LTG monotherapy HIGH RISK (myoclonic aggravation)",
            "VPA without POLG1 ABSOLUTE CI (Alpers-Huttenlocher)",
            "LEV BEHAVIOURAL CAUTION (rage/aggression in XLID/ASD — monitor closely; use LEV-XR)",
            "VGB without ERG HIGH RISK (visual field defects; non-verbal — ERG every 3 months mandatory)",
        ],
        "x_linked_clinical_note": "X-linked inheritance — genetic counselling essential: carrier mothers (50% daughters carrier, 50% sons affected); de novo recurrence <1% (germline mosaic ~1-2%); X-inactivation ratio determines female severity",
        "precision_therapy": "No single precision therapy approved (2024); perampanel (AMPAR antagonist) emerging Level C; KD early in DRE; VNS surgical non-candidates",
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
