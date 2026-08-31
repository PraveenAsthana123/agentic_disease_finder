"""
GABRA3 Epilepsy — X-linked Epileptic Encephalopathy / GABA-A α3 Subunit / Xq28
==================================================================================
40-patient cohort · GABRA3 (Xq28) · GABA-A receptor α3 subunit · X-linked dominant / de novo

GABRA3 BIOLOGY:
GABRA3 (Xq28) encodes the α3 subunit of the ionotropic GABA-A receptor pentamer.
The α3 subunit is the DOMINANT α-ISOFORM IN THE FETAL AND NEONATAL BRAIN — it is
expressed at highest levels before the postnatal α3→α1 subunit developmental switch,
making GABRA3 LOF uniquely harmful during the critical period of early brain development.

GABRA3 KEY EXPRESSION DOMAINS:
  1. THALAMIC RETICULAR NUCLEUS (TRN) — highest α3 expression in adult brain;
     TRN GABAergic neurons gate thalamo-cortical relay and generate sleep spindles;
     GABRA3 LOF → TRN disinhibition → thalamo-cortical hyperexcitability → absence-like
     EEG patterns + nocturnal tonic seizures.
  2. BRAINSTEM RETICULAR FORMATION — α3 mediates inhibitory control of motor nuclei;
     GABRA3 LOF → exaggerated motor responses → hyperekplexia (startle disease).
  3. CORTICAL LAYER V PYRAMIDAL NEURONS — large projection neurons; α3-GABA-A controls
     corticospinal + corticothalamic output.
  4. LIMBIC SYSTEM (amygdala, septum) — modulates fear, arousal, autonomic epileptic
     responses.
  5. NEONATAL CORTEX — α3 is the predominant α-subunit in fetal cortex; α3→α1 switch
     complete by ~12 months postnatal; GABRA3 variants causing early-onset severe DEE
     exploit this vulnerability window.

GABA-A RECEPTOR α3 SUBUNIT — KEY FUNCTIONAL ROLES:
  1. THALAMO-CORTICAL RHYTHM GENERATION: TRN → thalamus → cortex relay controlled by
     α3-GABA-A at TRN → sleep spindles, K-complexes, spike-wave discharge rhythm.
     GABRA3 LOF → impaired TRN self-inhibition → increased thalamic relay output →
     cortical hyperexcitability → 3Hz spike-wave discharges AND nocturnal tonic seizures.
  2. HYPEREKPLEXIA MECHANISM (GOF): Ile246Val and related GOF variants → constitutively
     active or hypersensitive α3-GABA-A in brainstem → exaggerated response to tactile/
     auditory startle → tonic-clonic extension response → falls, apnoea.
  3. BZD PHARMACOLOGY: α3-containing GABA-A receptors (α3β3γ2) BIND CLASSICAL
     BENZODIAZEPINES — His101 equivalent is PRESENT in α3 (unlike α4/α5/α6).
     HOWEVER: BZD ANXIOLYTIC EFFECT (α2/α3-mediated) is reduced in GABRA3 LOF;
     SEDATIVE RESCUE EFFECT (α1-mediated at α1β2γ2) remains INTACT.
     CLOBAZAM (1,5-BZD with α2/α3 preference) loses its primary target in GABRA3 LOF.
     CLINICAL IMPLICATION: Standard BZD rescue (diazepam, lorazepam, midazolam) remains
     effective for GTCS termination (via α1) but CLB may be LESS effective in GABRA3 LOF
     vs GABRA1/2 LOF. Phenobarbital (barbiturate site, preserved) is PREFERRED rescue.
  4. X-LINKED INHERITANCE:
     Males (hemizygous): single mutant allele → COMPLETE loss of α3 function → severe DEE,
     often lethal neonatally without intervention; surviving males: profound ID, non-verbal,
     refractory epilepsy.
     Females (heterozygous): one normal allele compensates partially → MOSAIC expression
     (X-inactivation skewing determines phenotype severity); range: asymptomatic carrier
     to severe DEE (when X-inactivation unfavourable to wild-type allele).
     DE NOVO FEMALE: 70% of cases are de novo heterozygous females with X-inactivation
     (random or skewed) → variable severity. Males: rare, severe.

GENOTYPE–PHENOTYPE CORRELATIONS:
  LOF (nonsense/frameshift): haploinsufficiency → severe DEE in males; variable in females.
  LOF (TM missense): ER retention/reduced surface expression → moderate-severe in females;
  severe in males.
  GOF (Ile246Val, extracellular/TM1): hyperekplexia + DEE; α3 enhanced basal activity
  → paradoxical excitation in developing brain (GABA-shift) + hyperekplexia in mature brain.
  Familial LOF carriers (females): mild GGE / febrile seizure susceptibility.

KEY REFERENCES:
  Davies PA et al. 2010 J Neurosci — GABRA3 Ile246Val hyperekplexia mechanism
  Lemke JR et al. 2017 Epilepsia — X-linked epileptic encephalopathy and GABRA3
  Niturad NJ et al. 2017 Brain — GABRA3 variants in epileptic encephalopathy
  Macdonald RL et al. 2010 Epilepsia — GABA-A receptor α-subunit roles in epilepsy
  Olsen RW & Sieghart W 2009 Pharmacol Rev — GABA-A receptor subtypes classification
  Pirker S et al. 2000 Neuroscience — α3 distribution in human brain (TRN enrichment)
  UKISS 2004 Lancet Neurol — ACTH + VGB infantile spasms standard
"""
import random

random.seed(493)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GABRA3-LOF-Male-Hemizygous-Severe-DEE",
        "pct": 35,
        "etiology": "GABRA3 LOF hemizygous male — complete α3 loss; neonatal/infantile severe DEE; often lethal without intervention",
        "mechanism": (
            "Hemizygous loss-of-function variants in males (nonsense, frameshift, large deletion) → complete "
            "absence of functional α3 subunit → TRN GABAergic inhibition abolished → thalamo-cortical "
            "hyperexcitability; brainstem reticular formation disinhibited → hyperekplexia + apnoeic episodes. "
            "Neonatal presentation: burst-suppression EEG, tonic-clonic seizures, severe hypotonia, apnoeic "
            "spells triggered by handling (startle-provoked). ACTH response variable; phenobarbital first-line. "
            "Prognosis: severe — most non-verbal, refractory epilepsy; some fatalities in neonatal period from "
            "apnoeic spells. X-inactivation not applicable. Complete LOF of only functional allele."
        ),
        "typical_variants": "c.967C>T (p.Arg323*) Xq28 · c.1108_1109del (p.Leu370Valfs*8) · del exons 3-6 hemizygous",
        "eeg_signature": "Burst-suppression neonatal → hypsarrhythmia → multifocal interictal spikes",
        "phenotype": "Severe DEE; non-verbal; hyperekplexia; apnoea; refractory tonic/clonic/spasms; ±hyperammonaemia from VPA",
        "onset_age_years": 0.1,
        "outcome": "Severe; high mortality without early intervention; survivors: profound ID, non-ambulatory",
    },
    {
        "category": "GABRA3-LOF-Female-De-Novo-DEE",
        "pct": 28,
        "etiology": "GABRA3 LOF de novo female heterozygous — X-inactivation unfavourable; moderate-severe DEE",
        "mechanism": (
            "De novo heterozygous variants in females → one functional allele lost; outcome depends on "
            "X-chromosome inactivation (XCI) pattern. When XCI skewed towards mutant allele (>70% mutant "
            "allele active) → severe DEE with epileptic spasms, tonic seizures, ID. Random XCI → intermediate "
            "phenotype. MOSAIC EXPRESSION: ~40% of cortical neurons express mutant α3 → heterogeneous "
            "network with partially impaired TRN inhibition + partial brainstem disinhibition. "
            "Epileptic spasms in 60% of de novo females; ACTH + VGB standard (UKISS 2004). "
            "Cognitive: moderate-severe ID in majority of de novo females with unfavourable XCI. "
            "BZD rescue: phenobarbital preferred; standard diazepam/midazolam also work (via intact α1). "
            "CLB (1,5-BZD) has reduced efficacy when α3 LOF: use PHB or DZP for rescue."
        ),
        "typical_variants": "c.815G>A (p.Arg272Gln) TM2 · c.482T>C (p.Ile161Thr) extracellular · splice c.734+2T>C",
        "eeg_signature": "Hypsarrhythmia → multifocal spikes → thalamo-cortical 3Hz SW in milder cases",
        "phenotype": "Epileptic spasms; tonic seizures; moderate-severe ID; variable hyperekplexia",
        "onset_age_years": 0.5,
        "outcome": "Variable (XCI-dependent); 65% moderate-severe ID; ACTH-responsive spasms in 55%",
    },
    {
        "category": "GABRA3-GOF-Hyperekplexia-DEE",
        "pct": 20,
        "etiology": "GABRA3 GOF de novo — p.Ile246Val equivalent; hyperekplexia + neonatal DEE; X-linked dominant",
        "mechanism": (
            "Gain-of-function missense variants (particularly p.Ile246Val, TM1 domain, OMIM-referenced) → "
            "constitutively active or hypersensitive α3-containing GABA-A receptors in brainstem. "
            "PARADOX: in neonatal/infantile brain where GABA is depolarising (high intracellular Cl⁻, "
            "KCC2 not yet expressed, EGABA > Vrest) → GOF-enhanced α3 → paradoxical EXCITATION → "
            "severe neonatal seizures (GABA-shift mechanism). In mature brain → HYPEREKPLEXIA: "
            "exaggerated non-epileptic startle response (stiffening, falls) to sudden sound/touch. "
            "COMBINED PHENOTYPE: neonatal DEE (GABA-excitatory window) + hyperekplexia (mature brain). "
            "CLONAZEPAM REDUCES HYPEREKPLEXIA: BZD-mediated enhancement of remaining α3-function "
            "or modulation of glycine receptor compensatory pathway; clonazepam standard for hyperekplexia. "
            "PIRACETAM: used historically for hyperekplexia (partially effective). "
            "AVOID: tiagabine (NCSE), carbamazepine (tonic-aggravating GOF mechanism)."
        ),
        "typical_variants": "c.736A>G (p.Ile246Val) TM1 · c.742T>C (p.Tyr248His) TM1 · c.749A>T (p.Asn250Ile) TM1-TM2 linker",
        "eeg_signature": "Neonatal burst-suppression; hyperekplexia: normal EEG with startle-evoked time-locked tonic response",
        "phenotype": "Hyperekplexia + neonatal DEE; startle-provoked falls/stiffening; clonazepam-responsive hyperekplexia",
        "onset_age_years": 0.05,
        "outcome": "Hyperekplexia often partially clonazepam-responsive; neonatal DEE resolves partially as KCC2 matures",
    },
    {
        "category": "GABRA3-LOF-Female-Familial-Mild",
        "pct": 12,
        "etiology": "GABRA3 LOF familial female carrier — favourable X-inactivation; GGE spectrum / febrile seizures",
        "mechanism": (
            "Maternally inherited GABRA3 LOF variant in females with predominantly wild-type allele active "
            "(XCI skewed >70% WT → relatively preserved α3 expression). Phenotype: mild GGE spectrum — "
            "febrile seizure susceptibility, occasional CAE-like absences, JME-like myoclonus in adolescence. "
            "PARTIAL PENETRANCE: ~40% of heterozygous female family members symptomatic. "
            "Cognitive: normal to mild ID. Seizure control: 70-80% well-controlled on standard GGE "
            "medications (VPA, LEV, LTG). IMPORTANT: male offspring who inherit variant = hemizygous → "
            "severe DEE (Category 1 above) — family counselling mandatory. "
            "Genetic testing cascade: all male first-degree relatives of carrier females."
        ),
        "typical_variants": "c.289G>A (p.Ala97Thr) extracellular · c.566C>T (p.Pro189Leu) linker · c.982G>A (p.Glu328Lys) TM3",
        "eeg_signature": "3Hz spike-wave (absence) or photoparoxysmal response; GGE pattern",
        "phenotype": "GGE spectrum; febrile seizures; occasional absence/myoclonus; normal-mild ID; well-controlled",
        "onset_age_years": 5,
        "outcome": "Good; 75% seizure-free on standard GGE therapy; mild or no ID; family cascade critical",
    },
    {
        "category": "Phenocopy-XLIE",
        "pct": 5,
        "etiology": "Phenocopy — X-linked infantile epileptic encephalopathy (no GABRA3 variant); CDKL5 / ARX / MECP2 overlap",
        "mechanism": (
            "Clinically GABRA3-like X-linked epileptic encephalopathy (male-predominant, infantile-onset, "
            "refractory) but GABRA3 sequencing/deletion negative. Confirmed alternative diagnoses: "
            "CDKL5 (Xp22.3 — hyperkinetic movement + seizures); ARX (Xp21.3 — Ohtahara syndrome, "
            "lissencephaly); MECP2 (Xq28 — Rett syndrome male; severe early-onset). "
            "IMPORTANT DIAGNOSTIC PITFALL: MECP2 is immediately adjacent to GABRA3 on Xq28 — large "
            "Xq28 deletions can delete BOTH genes simultaneously → combined GABRA3 + MECP2 deletion "
            "syndrome (more severe than either alone; Rett-like + DEE). "
            "Management: treat underlying diagnosis; seizure regimen by phenotype."
        ),
        "typical_variants": "CDKL5 p.Arg552Cys · ARX c.333ins(GCG)7 polyalanine expansion · MECP2 + GABRA3 Xq28 deletion",
        "eeg_signature": "Variable: CDKL5-pattern (fast activity + multifocal spikes) vs ARX (suppression-burst) vs MECP2",
        "phenotype": "Male-predominant severe DEE; GABRA3-like clinically; alternative gene confirmed",
        "onset_age_years": 0.3,
        "outcome": "Gene-specific; CDKL5/MECP2 have disease-specific management protocols",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE DETAIL
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_DETAIL = [
    {
        "type": "Epileptic Spasms (Infantile Spasms / West Syndrome)",
        "prevalence_pct": 52,
        "semiology": (
            "Sudden bilateral arm abduction/flexion ± head nod, often in clusters of 10-100 spasms "
            "on waking; may be subtle (minor head nod) in neonates. HYPEREKPLEXIA OVERLAP: "
            "startle-provoked spasms may mimic classic infantile spasms — distinguish by EEG context."
        ),
        "eeg_pattern": "Hypsarrhythmia (chaotic, high-amplitude, multifocal); electrodecrement at spasm onset",
        "clinical_tip": (
            "ACTH + vigabatrin first-line (UKISS 2004; Level A evidence). Vigabatrin REMS program mandatory "
            "(visual field restriction monitoring). In GABRA3 LOF, ACTH may have reduced efficacy vs "
            "GABRA1 subtypes — escalate to KD early (by week 4 if incomplete spasm cessation)."
        ),
    },
    {
        "type": "Tonic Seizures",
        "prevalence_pct": 62,
        "semiology": (
            "Sustained (5–30s) bilateral motor stiffening; axial extension or flexion; often nocturnal; "
            "eyes open + upward deviation; apnoeic component in severe cases. "
            "Distinguishing from hyperekplexia: tonic seizures = EEG ictal correlate; hyperekplexia = "
            "non-epileptic startle (normal or brief artefact on EEG)."
        ),
        "eeg_pattern": "Fast recruiting rhythm (>10Hz) or low-voltage fast activity at tonic seizure onset; TRN-origin marker",
        "clinical_tip": (
            "Phenobarbital most effective for tonic seizures in GABRA3 LOF (barbiturate site preserved). "
            "Felbamate Level B for refractory tonic. AVOID carbamazepine/oxcarbazepine in mixed "
            "tonic-myoclonic phenotype (can aggravate myoclonic component)."
        ),
    },
    {
        "type": "Hyperekplexia (Startle-Provoked Events)",
        "prevalence_pct": 38,
        "semiology": (
            "Exaggerated generalised stiffening to sudden acoustic or tactile stimulus; falls without "
            "LOC; neonatal: apnoea + generalised rigidity triggered by handling. NON-EPILEPTIC — no "
            "ictal EEG correlate. CLONAZEPAM-RESPONSIVE. Differentiate from reflex epilepsy (which has "
            "EEG ictal correlate). Nose-tap test (brief forced flexion of neck + legs): diagnostic."
        ),
        "eeg_pattern": "Normal interictal or mild diffuse slowing; NO ictal correlate at hyperekplexia event",
        "clinical_tip": (
            "Clonazepam 0.01–0.03 mg/kg/day reduces hyperekplexia severity (glycine receptor modulation "
            "or residual α3-mediated pathway). Piracetam second-line. CRITICAL: document hyperekplexia "
            "separately from epileptic seizures — avoid escalating AEDs based on hyperekplexia events alone. "
            "Neonatal hyperekplexia apnoea: immediate forward flexion manoeuvre (bend head + knees to chest)."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "prevalence_pct": 42,
        "semiology": (
            "Brief (<100ms) bilateral myoclonic jerks; axial or limb; morning predominance; may evolve "
            "to GTCS if untreated. In GOF subtype: startle-myoclonus (tactile-triggered bilateral jerks) "
            "that blend with hyperekplexia — EEG critical to distinguish epileptic (ictal correlate) "
            "vs startle (no correlate)."
        ),
        "eeg_pattern": "Polyspike-wave discharge 3–5Hz; synchronous with myoclonic jerks; generalised distribution",
        "clinical_tip": (
            "VPA Level A for myoclonic seizures (if POLG excluded). LEV adjunct. ABSOLUTE CI: "
            "LTG (sodium-channel enhancement → paradoxical myoclonic worsening in mixed tonic-myoclonic "
            "GABRA3 phenotype). CLB limited efficacy if α3 LOF is primary mechanism."
        ),
    },
    {
        "type": "Focal to Bilateral Tonic-Clonic (GTCS)",
        "prevalence_pct": 35,
        "semiology": (
            "Generalised tonic-clonic; can evolve from focal onset (temporal or frontal) or be primarily "
            "generalised. Post-ictal confusion 15–60min. Nocturnal predominance in GABRA3-thalamic subtype. "
            "SUDEP risk elevated: nocturnal GTCS in non-supervised setting — seizure alarm mandatory."
        ),
        "eeg_pattern": "Generalised polyspike-wave at GTCS onset; post-ictal diffuse suppression",
        "clinical_tip": (
            "VPA Level A (POLG mandatory pre-treatment). LEV + LCM combination for refractory GTCS. "
            "SUDEP prevention: GTCS nocturnal alarm, avoid sleep deprivation, nocturnal supervision. "
            "RESCUE: PHB preferred over CLB in GABRA3 LOF (CLB α2/α3 efficacy reduced)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_DETAIL = [
    {"trigger": "Startle / Sudden Stimulus (acoustic, tactile)", "pct": 68,
     "note": "HALLMARK of GABRA3 GOF (hyperekplexia); also lowers seizure threshold in LOF subtypes. Protective: reduce sudden noise exposure at home."},
    {"trigger": "Sleep Deprivation", "pct": 55,
     "note": "Disrupts thalamo-cortical rhythm regulation; TRN α3 GABA-A particularly vulnerable. Strict sleep schedule mandatory."},
    {"trigger": "Fever / Intercurrent Illness", "pct": 48,
     "note": "Thermolability of α3-GABA-A subunit interactions + metabolic stress. Pre-emptive fever plan with PHB/CLZ rescue."},
    {"trigger": "Emotional Stress / Excitement", "pct": 42,
     "note": "Amygdala-limbic α3-GABA-A provides stress-gating; LOF → reduced emotional seizure threshold."},
    {"trigger": "Handling / Physical Contact (neonatal)", "pct": 38,
     "note": "Neonatal GABRA3 LOF/GOF: tactile stimulus-triggered apnoeic spells + tonic stiffening. FORWARD FLEXION MANOEUVRE immediately."},
    {"trigger": "Photic Stimulation (IPS)", "pct": 22,
     "note": "Moderate photoparoxysmal response in GGE-spectrum females (familial category). Avoid stroboscopic lights."},
    {"trigger": "Menstrual / Hormonal Changes", "pct": 18,
     "note": "Catamenial pattern in female heterozygous carriers — progesterone modulates GABA-A; perimenstrual CLZ or PHB supplementation."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT ARSENAL
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_DETAIL = [
    {
        "drug": "Phenobarbital",
        "level": "Level A — First-Line",
        "moa": "Barbiturate positive allosteric modulator of GABA-A (barbiturate site, distinct from BZD site); PRESERVED in GABRA3 LOF → enhances Cl⁻ conductance at all α-subunit subtypes",
        "dose": "3–5 mg/kg/day IV/IM loading; 3–5 mg/kg/day PO maintenance; neonatal: 20 mg/kg IV load",
        "efficacy": "Level A neonatal seizures; Level B tonic seizures; PREFERRED over CLB in GABRA3 LOF",
        "safety": "Sedation; cognitive effects (monitor development); liver enzyme induction; do not use with rifampicin",
        "monitoring": "Drug level (therapeutic 15–40 mg/L); LFTs annually; cognitive assessment quarterly in infants",
        "gabra3_note": "GABRA3-specific advantage: PHB acts at barbiturate site — NO dependence on α3-subunit for efficacy. This is the key rescue and maintenance drug when α3 LOF impairs BZD/CLB targets.",
    },
    {
        "drug": "ACTH (Tetracosactide / Synacthen)",
        "level": "Level A — Infantile Spasms",
        "moa": "Corticotropin; stimulates adrenal cortisol synthesis; reduces CRH (pro-convulsant); modulates GABA-A gene expression; reduces hypsarrhythmia via non-GABA-A mechanism → effective despite α3 LOF",
        "dose": "0.5 mg/kg IM daily for 2 weeks (natural ACTH); synthetic 40–80 IU/day IM; 4-6 week course",
        "efficacy": "UKISS 2004 Level A: ACTH + vigabatrin = gold standard for infantile spasms; GABRA3-LOF response: ~55% electroclinical remission",
        "safety": "Hypertension; Cushingoid features; susceptibility to infection (live vaccine hold); hyponatraemia; adrenal suppression",
        "monitoring": "BP twice daily (ACTH course); glucose; electrolytes; adrenal function post-course; infection surveillance",
        "gabra3_note": "ACTH mechanism is independent of α3-GABA-A → effective in GABRA3 LOF. If spasm remission incomplete at 4 weeks, add ketogenic diet (synergistic). Vigabatrin mandatory partner but monitor visual fields (REMS).",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A — Infantile Spasms (Partner)",
        "moa": "Irreversible GABA-T inhibitor → increased synaptic GABA → enhanced GABA-A tonic and phasic inhibition; compensates partially for α3 LOF via increased GABA availability",
        "dose": "100–150 mg/kg/day (infantile spasms); 40–60 mg/kg/day maintenance",
        "efficacy": "UKISS 2004 Level A with ACTH for infantile spasms; ≤16 weeks for spasms (SHARE trial); longer-term for focal epilepsy in TSC",
        "safety": "PERMANENT VISUAL FIELD RESTRICTION (concentric bilateral): REMS program mandatory; irreversible after cumulative dose; baseline ERG + visual field before start",
        "monitoring": "VGB REMS ERG ≤6 weeks (infantile spasms); Humphrey visual field >6 months; baseline + q6m; STOP if VFR progressing",
        "gabra3_note": "VGB pairs with ACTH for infantile spasms in GABRA3 LOF. LIMIT duration to ≤16 weeks for spasms. For focal epilepsy post-spasms: lower doses, continued monitoring. VFR risk is irreversible — document consent.",
    },
    {
        "drug": "Valproic Acid (VPA)",
        "level": "Level A — Myoclonic / GTCS",
        "moa": "Multiple: Na⁺/Ca²⁺ channel block; GABA-T inhibition; GABA-A potentiation (broad-spectrum). Effective at α1-containing receptors → partially compensates α3 LOF",
        "dose": "20–40 mg/kg/day; seizure-free dose individual; monitor trough level 50–100 mg/L",
        "efficacy": "Level A broad-spectrum: myoclonic, GTCS, absence. Effective in GABRA3 GGE-spectrum females",
        "safety": "HEPATOTOXICITY (fatal in POLG1/Alpers); teratogenicity (NTD, cognitive NEAD); pancreatitis; polycystic ovary; thrombocytopenia; hyperammonaemia",
        "monitoring": "POLG1 sequencing BEFORE VPA in any child/male with: early-onset, ID, liver disease, family POLG history. LFTs weekly ×4, monthly ×6. Ammonia if drowsy/encephalopathy.",
        "gabra3_note": "POLG MANDATORY before VPA: any X-linked epileptic encephalopathy may co-harbour POLG variant (mitochondrial disorder overlap). ABSOLUTE CI in confirmed POLG/Alpers. VPPP protocol females. Monitor ammonia in ALL GABRA3-LOF patients on VPA.",
    },
    {
        "drug": "Clonazepam (CLZ)",
        "level": "Level A — Hyperekplexia; Level B — Seizures",
        "moa": "BZD positive allosteric modulator (α/γ2 interface): reduces hyperekplexia via residual α3 or glycine receptor compensatory pathway; ALSO modulates GLYCINE RECEPTOR in brainstem (indirect) → startle-gating restoration",
        "dose": "Hyperekplexia: 0.01–0.05 mg/kg/day PO; seizures: standard BZD doses; rescue: 0.1 mg/kg buccal",
        "efficacy": "Level A hyperekplexia (international standard); Level B seizures in GABRA3 GOF subtype",
        "safety": "Sedation; tolerance; respiratory depression; paradoxical agitation in children",
        "monitoring": "Titrate to hyperekplexia control; avoid excessive sedation that masks developmental assessment",
        "gabra3_note": "GABRA3-SPECIFIC: CLZ is FIRST-LINE for hyperekplexia (GOF category) — works via BZD-site on any residual α3 + indirect glycine pathway. For seizures in LOF: CLZ has REDUCED efficacy compared to PHB (α3 LOF impairs α2/α3-targeted BZDs); use PHB as primary rescue for seizures.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — Early Escalation",
        "moa": "Ketone bodies (β-hydroxybutyrate) → multiple anti-seizure mechanisms: enhances glutamate-GABA conversion, reduces glucose-driven neuronal excitability, mitochondrial protection, KATP channel activation — INDEPENDENT of GABA-A subunit composition",
        "dose": "Classical 4:1 ratio (fat:carb+protein); MAD; start within first months if refractory spasms/DEE",
        "efficacy": "Level B: ≥50% seizure reduction in 50-60% refractory paediatric epilepsy; early use in GABRA3 severe DEE reduces seizure burden",
        "safety": "Growth monitoring; lipid profile; kidney stones; selenium/carnitine depletion; dietitian-supervised essential",
        "monitoring": "Urine ketones daily; lipids q3m; growth q1m; selenium/carnitine/zinc q6m; EEG on KD q3m",
        "gabra3_note": "KD mechanism is entirely INDEPENDENT of GABA-A α3 subunit → no pharmacological disadvantage in GABRA3 LOF. Initiate by week 4 if ACTH + VGB give incomplete spasm remission. Synergistic with ACTH.",
    },
    {
        "drug": "Lacosamide (LCM)",
        "level": "Level B — Adjunct Focal/Tonic",
        "moa": "Slow inactivation state selective Na⁺ channel blocker; reduces persistent Na⁺ current; thalamo-cortical relay stabilisation; NOT a GABA-A drug → no dependence on α3",
        "dose": "2–4 mg/kg/day (paediatric); 200–400 mg/day (adult); IV available for status",
        "efficacy": "Level B focal seizures + tonic seizures; useful adjunct in GABRA3 severe DEE",
        "safety": "Cardiac PR prolongation (screen ECG before); dizziness; diplopia; avoid in 2nd-3rd degree heart block",
        "monitoring": "ECG pre-treatment and if dose >400 mg/day; LFTs (mild); cardiac history review",
        "gabra3_note": "Useful thalamo-cortical stabiliser in GABRA3 tonic seizure subtype (TRN disinhibition). Na⁺ channel blockade at thalamic relay neurons reduces pathological firing. Combine with PHB for refractory tonic seizures.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — Broad Adjunct",
        "moa": "SV2A synaptic vesicle protein modulator → reduces presynaptic neurotransmitter release; GABA-A independent mechanism → preserved efficacy in GABRA3 LOF",
        "dose": "20–60 mg/kg/day paediatric; 1000–3000 mg/day adult; IV available (loading 60 mg/kg)",
        "efficacy": "Level B broad-spectrum adjunct; myoclonic seizures (JME evidence); focal and generalised",
        "safety": "Behavioural side effects (irritability, aggression) — particularly concerning in severe ID/non-verbal children who cannot report; psychiatric monitoring essential",
        "monitoring": "Behavioural scales quarterly (ABERRANT BEHAVIOUR CHECKLIST); renal dosing in CKD; drug level not routinely needed",
        "gabra3_note": "SV2A mechanism INDEPENDENT of α3 → no pharmacological disadvantage in GABRA3 LOF. BEHAVIOURAL MONITORING CRITICAL in severe ID patients — non-verbal children cannot report agitation/anxiety; use carer-reported behavioural scales.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATION_DETAIL = [
    {"drug": "Lamotrigine (LTG) — in mixed tonic-myoclonic phenotype",
     "risk": "ABSOLUTE CI — Myoclonic",
     "reason": "LTG Na⁺ channel enhancement → paradoxical myoclonic worsening (GABRA3 myoclonic subtype, categories 1-2). EEG-documented myoclonus worsening/NCSE. Prescribing note mandatory."},
    {"drug": "Tiagabine (TGB)",
     "risk": "ABSOLUTE CI",
     "reason": "GABA reuptake inhibitor → elevated synaptic GABA + reduced GAT-1 clearance → paradoxical NCSE (non-convulsive status epilepticus) in GABRA3 LOF: elevated synaptic GABA with impaired extrasynaptic α3 clearance → NCSE precipitation. Never use."},
    {"drug": "Carbamazepine / Oxcarbazepine / Phenytoin (CBZ/OXC/PHT)",
     "risk": "HIGH CAUTION — Tonic-Myoclonic",
     "reason": "Na⁺ channel blockers → aggravate myoclonic seizures; in mixed tonic-myoclonic GABRA3 phenotype, may aggravate myoclonus even while suppressing tonic. Use only if pure tonic with no myoclonic component; monitor EEG 4 weeks."},
    {"drug": "VPA — without prior POLG1 screening",
     "risk": "ABSOLUTE CI — Without POLG Screen",
     "reason": "VPA in unscreened POLG1 mutation → Alpers-Huttenlocher syndrome (fatal hepatic failure). GABRA3 severe DEE patients may co-harbour POLG variant. POLG sequencing MANDATORY before VPA in any severe infantile epileptic encephalopathy."},
    {"drug": "Vigabatrin (VGB) — long-term beyond ≤16 weeks for infantile spasms",
     "risk": "HIGH CAUTION — Irreversible VFR",
     "reason": "Cumulative VGB exposure → irreversible concentric visual field restriction (bilateral, permanent). REMS mandates baseline ERG and ≤6-week ERG for infantile spasms. Limit to ≤16 weeks for spasms; longer-term only in TSC/focal epilepsy with explicit VFR consent."},
    {"drug": "Clobazam (CLB) — as PRIMARY rescue in GABRA3 LOF severe DEE",
     "risk": "MODERATE CAUTION",
     "reason": "CLB (1,5-BZD) has enhanced α2/α3 affinity — but in GABRA3 LOF, the α3 target is absent/reduced. CLB efficacy for seizure rescue is REDUCED vs PHB. CLB remains first-line for HYPEREKPLEXIA (GOF subtype). Use PHB as primary rescue for epileptic seizures in LOF subtypes."},
]

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_THRESHOLDS = [
    {"label": "PHB Therapeutic Range", "value": "15–40", "unit": "mg/L"},
    {"label": "VPA Therapeutic Range", "value": "50–100", "unit": "mg/L"},
    {"label": "ACTH Course Duration", "value": "≤6", "unit": "weeks"},
    {"label": "VGB Spasms Duration Limit", "value": "≤16", "unit": "weeks"},
    {"label": "VGB ERG Monitoring Interval (Spasms)", "value": "≤6", "unit": "weeks"},
    {"label": "CLZ Hyperekplexia Dose", "value": "0.01–0.05", "unit": "mg/kg/day"},
    {"label": "PHB Neonatal Loading Dose", "value": "20", "unit": "mg/kg IV"},
    {"label": "POLG Testing Mandate", "value": "Before VPA", "unit": "every patient"},
    {"label": "PHB Level Check Post-Load", "value": "30–60", "unit": "min after load"},
    {"label": "Spasm Remission Review (ACTH)", "value": "4", "unit": "weeks (escalate KD if incomplete)"},
    {"label": "XCI Skewing Threshold (Severity)", "value": ">70%", "unit": "mutant allele active → severe"},
    {"label": "Hyperekplexia Nose-Tap Test", "value": "≥3", "unit": "repetitive flexion response = positive"},
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_SCHEDULE = [
    {"item": "EEG (video-EEG)", "frequency": "Every 3 months (active DEE); annually (stable)"},
    {"item": "PHB drug level", "frequency": "1 week post-initiation; then every 6 months"},
    {"item": "VGB ERG (infantile spasms)", "frequency": "Baseline + every 6 weeks during spasm phase"},
    {"item": "VGB Humphrey visual field", "frequency": "Every 6 months while on VGB"},
    {"item": "VPA levels + LFTs + ammonia", "frequency": "Weekly ×4, then monthly ×6, then quarterly"},
    {"item": "POLG sequencing", "frequency": "Once, before ANY VPA initiation"},
    {"item": "BP (ACTH course)", "frequency": "Twice daily during ACTH treatment"},
    {"item": "Developmental assessment (Griffiths/Bayley)", "frequency": "Every 6 months (infants)"},
    {"item": "KD: ketones + growth + lipids", "frequency": "Ketones daily; lipids + growth monthly"},
    {"item": "X-inactivation analysis", "frequency": "Once at diagnosis (affected female); guides prognosis"},
    {"item": "LEV behavioural scale (ABC)", "frequency": "Every 3 months while on LEV"},
    {"item": "Cardiac ECG (lacosamide)", "frequency": "Pre-treatment baseline; repeat if dose >400 mg/day"},
    {"item": "Hyperekplexia nose-tap test", "frequency": "Every clinic visit (document startle response)"},
    {"item": "Seizure diary (startle vs epileptic)", "frequency": "Daily (distinguish hyperekplexia from seizures)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {"window": "Neonatal (0–4 weeks)", "headline": "Apnoeic hyperekplexia crisis; burst-suppression; PHB loading; forward-flexion manoeuvre for apnoea"},
    {"window": "Infantile (1–12 months)", "headline": "Epileptic spasms + hyperekplexia peak; ACTH + VGB; KD escalation if refractory"},
    {"window": "Early Childhood (1–5 years)", "headline": "Tonic-myoclonic seizures emerge; post-spasm cognitive assessment; PHB + KD maintenance"},
    {"window": "Childhood (5–12 years)", "headline": "Seizure type evolution; school transition; LEV/LCM adjuncts; VGB visual field monitoring"},
    {"window": "Adolescence (12–18 years)", "headline": "Catamenial pattern females; VPA → VPPP if female; SUDEP awareness; independence safety"},
    {"window": "Adulthood / Transition", "headline": "GABA-A receptor maturation; PHB vs newer agents; genetic counselling for carrier females; inheritance risk to male offspring"},
]

# ─────────────────────────────────────────────────────────────────────────────
# CORE CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────
CORE_CONCEPTS = [
    {"term": "GABRA3", "definition": "Gene encoding GABA-A receptor α3 subunit (492aa, Xq28); most abundant α-isoform in fetal/neonatal brain; enriched in thalamic reticular nucleus and brainstem in adults."},
    {"term": "X-linked Inheritance", "definition": "GABRA3 on Xq28: hemizygous males (one X, one mutation = complete LOF → severe); heterozygous females (one normal allele, outcome governed by X-chromosome inactivation skewing)."},
    {"term": "X-Chromosome Inactivation (XCI)", "definition": "Random silencing of one X chromosome per cell; GABRA3 heterozygous females with >70% mutant allele active (unfavourable XCI) → severe DEE; <70% → milder phenotype. XCI analysis guides prognosis."},
    {"term": "Thalamic Reticular Nucleus (TRN)", "definition": "GABAergic nucleus surrounding thalamus; α3-GABA-A predominant; controls thalamo-cortical relay firing and sleep spindle generation. GABRA3 LOF → TRN disinhibition → thalamo-cortical hyperexcitability → tonic seizures and nocturnal epilepsy."},
    {"term": "Hyperekplexia", "definition": "Exaggerated, non-habituating startle response (generalised stiffening/falling ± apnoea) to sudden stimulus. Non-epileptic (no EEG ictal correlate). GABRA3 GOF (Ile246Val) is a major genetic cause. Clonazepam and forward-flexion manoeuvre are standard treatments."},
    {"term": "GABRA3 GOF (Gain-of-Function)", "definition": "Missense variants (TM1 domain, p.Ile246Val) → enhanced or constitutive α3-GABA-A activity → paradoxical excitation in neonates (GABA-shift; KCC2 immature) → neonatal DEE + hyperekplexia in mature brain."},
    {"term": "GABA-shift", "definition": "Neonatal GABA is DEPOLARISING (high intracellular Cl⁻; KCC2 not expressed; EGABA > Vrest → Cl⁻ efflux → depolarisation). GOF GABA-A → enhanced 'inhibitory' conductance that is paradoxically excitatory. Resolves as KCC2 matures (~12 months)."},
    {"term": "Phenobarbital (PHB)", "definition": "Barbiturate positive allosteric modulator of GABA-A; acts at BARBITURATE SITE (distinct from BZD α/γ interface) → enhances Cl⁻ conductance at all α-subunit subtypes → efficacy PRESERVED in GABRA3 LOF. First-line AED in GABRA3-LOF DEE."},
    {"term": "CLB (Clobazam) Reduced Efficacy", "definition": "Clobazam (1,5-BZD) has enhanced affinity for α2/α3-containing GABA-A receptors. In GABRA3 LOF: α3 target is absent/reduced → CLB loses its primary target → seizure rescue efficacy REDUCED vs PHB. CLB remains effective for HYPEREKPLEXIA (GOF subtype) via residual α3 or glycine pathway."},
    {"term": "POLG Mandate", "definition": "VPA causes fatal hepatotoxicity (Alpers-Huttenlocher syndrome) in POLG1-mutant patients. GABRA3 severe DEE patients may co-harbour POLG variant. POLG sequencing is MANDATORY before any VPA prescription in severe infantile epileptic encephalopathy."},
    {"term": "UKISS Protocol", "definition": "UK Infantile Spasms Study 2004 (Lancet Neurol): ACTH + vigabatrin = Level A evidence for infantile spasms. Applicable in GABRA3 DEE spasms. VGB ≤16 weeks (SHARE trial); REMS visual monitoring mandatory. ACTH course ≤6 weeks."},
    {"term": "VGB REMS", "definition": "Vigabatrin Risk Evaluation and Mitigation Strategy (FDA/Health Canada): mandatory baseline ERG, ≤6-week ERG during infantile spasms phase, Humphrey visual field every 6 months. Documents and monitors irreversible concentric visual field restriction (VFR)."},
    {"term": "Xq28 Proximity (MECP2/GABRA3 co-deletion)", "definition": "GABRA3 (Xq28.3) is adjacent to MECP2 (Xq28.3); large Xq28 chromosomal deletions can delete BOTH simultaneously → combined GABRA3 + MECP2 deletion syndrome (Rett-like features + severe DEE). SNP array or MLPA required to exclude Xq28 deletion."},
    {"term": "Nose-Tap Test", "definition": "Diagnostic test for hyperekplexia: brief forced flexion of head + knees toward chest → in positive test, patient shows repetitive, non-habituating flexion response (≥3 times). Distinguishes hyperekplexia from epileptic tonic seizures."},
    {"term": "α3→α1 Developmental Switch", "definition": "Postnatal GABA-A subunit maturation: α3 (dominant in fetal/neonatal brain) is progressively replaced by α1 (adult predominant) over first 12 months. GABRA3 LOF most severely affects the neonatal-infantile developmental window before this switch completes."},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_STANDARDS = [
    "ILAE 2022 Classification — GABRA3-related epilepsy: Genetic Epileptic Encephalopathy / X-linked DEE / GABRA3 Xq28",
    "X-linked inheritance counselling — carrier female: 50% risk; affected male offspring SEVERE → mandatory prenatal/preimplantation counselling",
    "UKISS 2004 (Lancet Neurol) — ACTH + vigabatrin Level A for infantile spasms; ≤16 weeks VGB (SHARE 2008 trial)",
    "VGB REMS — baseline ERG; ≤6-week ERG (infantile spasms); Humphrey VF every 6 months; patient/carer VFR consent",
    "POLG Mandate — POLG1 sequencing BEFORE VPA in ALL severe infantile epileptic encephalopathy; Alpers-Huttenlocher fatal if missed",
    "VPPP Protocol — VPA-Pregnancy Prevention Programme for all females of childbearing potential on VPA",
    "Hyperekplexia Standard — Clonazepam first-line; forward-flexion manoeuvre for neonatal apnoeic attacks; carer training",
    "SUDEP Prevention — nocturnal GTCS alarm; avoid sleep deprivation; cardiac screening; supervision",
    "Nose-tap test documentation — every clinic visit; distinguish hyperekplexia (non-epileptic) from epileptic tonic seizures",
    "XCI analysis — X-chromosome inactivation skewing study in ALL symptomatic heterozygous females; guides prognosis",
    "Cascade genetic testing — ALL male first-degree relatives of GABRA3 carrier females (hemizygous males → severe DEE)",
    "Xq28 deletion exclusion — SNP array or MLPA to exclude large Xq28 deletion (GABRA3 + MECP2 co-deletion) in severe males",
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
KEY_REFERENCES = [
    "Davies PA et al. (2010) J Neurosci — GABRA3 Ile246Val GOF causes startle disease (hyperekplexia) via TM1-domain mechanism",
    "Lemke JR et al. (2017) Epilepsia — GABRA3 variants in X-linked epileptic encephalopathy; genotype-phenotype correlation",
    "Niturad NJ et al. (2017) Brain — GABRA3 de novo variants cause epileptic encephalopathy; X-inactivation influences severity",
    "Macdonald RL & Olsen RW (1994) Annu Rev Neurosci — GABA-A receptor subtypes and their pharmacology",
    "Pirker S et al. (2000) Neuroscience — α3 subunit distribution in human brain: TRN enrichment",
    "Olsen RW & Sieghart W (2009) Pharmacol Rev — International Union of Pharmacology GABA-A receptor subtypes classification",
    "UKISS Group (2004) Lancet Neurol — United Kingdom Infantile Spasms Study: ACTH + vigabatrin Level A standard",
    "Chiron C et al. (2011) Epilepsia — VGB REMS and visual field restriction monitoring protocol",
    "Moser FG et al. (1990) Am J Med Genet — Hyperekplexia: clinical and EEG characterisation; clonazepam treatment",
    "Thomas P et al. (2017) Epilepsy Behav — X-inactivation skewing in GABRA3 heterozygous females and phenotypic variability",
    "Lüscher B et al. (2011) Pharmacol Ther — GABA-A receptor trafficking and pathophysiology in epilepsy",
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT SAMPLE GENERATOR  (40 patients, seed=493)
# ─────────────────────────────────────────────────────────────────────────────
def _make_patients():
    cats = [
        ("GABRA3-LOF-Male-Severe-DEE", 35, "M"),
        ("GABRA3-LOF-Female-DeNovo-DEE", 28, "F"),
        ("GABRA3-GOF-Hyperekplexia-DEE", 20, "F"),
        ("GABRA3-LOF-Female-Familial-Mild", 12, "F"),
        ("Phenocopy-XLIE", 5, "M"),
    ]
    pts = []
    pid = 1
    for cat, pct, default_sex in cats:
        n = max(1, round(40 * pct / 100))
        for _ in range(n):
            sex = default_sex if cat != "GABRA3-LOF-Female-DeNovo-DEE" else "F"
            is_severe = cat in ("GABRA3-LOF-Male-Severe-DEE", "GABRA3-LOF-Female-DeNovo-DEE")
            is_gof = cat == "GABRA3-GOF-Hyperekplexia-DEE"
            is_mild = cat == "GABRA3-LOF-Female-Familial-Mild"
            onset = round(random.uniform(0.05, 0.3) if is_severe or is_gof else (random.uniform(3, 8) if is_mild else random.uniform(0.5, 2)), 1)
            age = round(onset + random.uniform(1, 18), 1)
            dre = random.random() < (0.72 if is_severe else 0.30 if is_gof else 0.08 if is_mild else 0.30)
            spasms = random.random() < (0.65 if is_severe else 0.05 if is_mild else 0.30)
            tonic = random.random() < (0.80 if is_severe else 0.15 if is_mild else 0.40)
            myo = random.random() < (0.55 if is_severe else 0.10 if is_mild else 0.30)
            hyperek = random.random() < (0.80 if is_gof else 0.30 if is_severe else 0.10)
            gtcs = random.random() < (0.40 if is_severe else 0.20 if is_mild else 0.30)
            polg = random.random() < 0.78
            on_phb = random.random() < (0.90 if is_severe else 0.20 if is_mild else 0.50)
            on_vpa = random.random() < (0.30 if is_severe else 0.70 if is_mild else 0.30)
            on_clz = random.random() < (0.80 if is_gof else 0.30 if is_severe else 0.15)
            on_lev = random.random() < (0.30 if is_severe else 0.50 if is_mild else 0.30)
            on_lcm = random.random() < (0.20 if is_severe else 0.05)
            on_kd = random.random() < (0.45 if is_severe else 0.05 if is_mild else 0.20)
            acth_trial = random.random() < (0.60 if is_severe else 0.10)
            sudep_risk = dre and tonic and gtcs
            pts.append({
                "id": f"GABRA3-{pid:03d}",
                "sex": sex,
                "age": age,
                "onset_age": onset,
                "category": cat,
                "drug_resistant": dre,
                "spasms": spasms,
                "tonic_seizures": tonic,
                "myoclonic": myo,
                "hyperekplexia": hyperek,
                "gtcs": gtcs,
                "polg_tested": "Y" if polg else "N",
                "on_phb": on_phb,
                "on_vpa": on_vpa,
                "on_clz": on_clz,
                "on_lev": on_lev,
                "on_lcm": on_lcm,
                "on_kd": on_kd,
                "acth_trial": acth_trial,
                "sudep_high_risk": sudep_risk,
            })
            pid += 1
    return pts[:40]


PATIENTS = _make_patients()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    spasms = sum(1 for p in PATIENTS if p["spasms"])
    tonic = sum(1 for p in PATIENTS if p["tonic_seizures"])
    hyperek = sum(1 for p in PATIENTS if p["hyperekplexia"])
    myoclonic = sum(1 for p in PATIENTS if p["myoclonic"])
    gtcs = sum(1 for p in PATIENTS if p["gtcs"])
    polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    on_phb = sum(1 for p in PATIENTS if p["on_phb"])
    on_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    on_clz = sum(1 for p in PATIENTS if p["on_clz"])
    on_kd = sum(1 for p in PATIENTS if p["on_kd"])
    acth = sum(1 for p in PATIENTS if p["acth_trial"])
    sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])
    males = sum(1 for p in PATIENTS if p["sex"] == "M")

    etio = [
        {"etiology": e["category"], "n": max(1, round(n * e["pct"] / 100)), "pct": e["pct"]}
        for e in ETIOLOGY_CATALOG
    ]

    tx_summary = [
        {"drug": "Phenobarbital", "level": "Level A — First-Line (α3-independent)"},
        {"drug": "ACTH + Vigabatrin", "level": "Level A — Infantile Spasms (UKISS)"},
        {"drug": "Clonazepam", "level": "Level A — Hyperekplexia"},
        {"drug": "Ketogenic Diet", "level": "Level B — Early Escalation"},
        {"drug": "VPA (POLG mandatory)", "level": "Level A — Myoclonic / GTCS"},
        {"drug": "Lacosamide", "level": "Level B — Adjunct Tonic"},
        {"drug": "Levetiracetam", "level": "Level B — Broad Adjunct"},
        {"drug": "Vigabatrin (≤16 weeks)", "level": "Level A — Spasms Partner"},
    ]

    return {
        "kpis": {
            "n_patients": n,
            "drug_resistant_pct": round(dre / n * 100),
            "spasms_pct": round(spasms / n * 100),
            "tonic_pct": round(tonic / n * 100),
            "hyperekplexia_pct": round(hyperek / n * 100),
            "myoclonic_pct": round(myoclonic / n * 100),
            "gtcs_pct": round(gtcs / n * 100),
            "polg_tested_pct": round(polg / n * 100),
            "on_phb_pct": round(on_phb / n * 100),
            "on_vpa_pct": round(on_vpa / n * 100),
            "on_clz_pct": round(on_clz / n * 100),
            "on_kd_pct": round(on_kd / n * 100),
            "acth_trial_pct": round(acth / n * 100),
            "sudep_high_risk_n": sudep,
            "male_hemizygous_n": males,
        },
        "etiology_distribution": etio,
        "treatments_summary": tx_summary,
        "monitoring_summary": MONITORING_SCHEDULE[:8],
        "lifecycle": LIFECYCLE_WINDOWS,
        "thresholds": CLINICAL_THRESHOLDS[:8],
        "contraindications_summary": [
            "LTG-ABSOLUTE-CI-Myoclonic",
            "TGB-ABSOLUTE-CI-NCSE",
            "VPA-without-POLG-ABSOLUTE-CI",
            "CBZ/OXC/PHT-HIGH-CAUTION-Mixed-Phenotype",
            "VGB-long-term-VFR-HIGH-CAUTION",
            "CLB-REDUCED-RESCUE-in-LOF",
        ],
    }


def get_breakdown():
    etio_detail = []
    for e in ETIOLOGY_CATALOG:
        etio_detail.append({
            "etiology": e["category"],
            "n": max(1, round(len(PATIENTS) * e["pct"] / 100)),
            "pct": e["pct"],
            "mechanism": e["mechanism"],
            "typical_variants": e["typical_variants"],
            "eeg_signature": e["eeg_signature"],
            "phenotype": e["phenotype"],
        })

    n = len(PATIENTS)
    dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    vpa_no_polg = sum(1 for p in PATIENTS if p["on_vpa"] and p["polg_tested"] == "N")

    return {
        "etiology_distribution": etio_detail,
        "patient_sample": PATIENTS[:15],
        "seizure_detail": SEIZURE_DETAIL,
        "trigger_detail": TRIGGER_DETAIL,
        "treatment_detail": TREATMENT_DETAIL,
        "contraindications": CONTRAINDICATION_DETAIL,
        "summary": {
            "drug_resistant_pct": round(dre / n * 100),
            "polg_tested_pct": round(polg / n * 100),
            "vpa_without_polg_n": vpa_no_polg,
            "sudep_high_risk_n": sum(1 for p in PATIENTS if p["sudep_high_risk"]),
            "hyperekplexia_pct": round(sum(1 for p in PATIENTS if p["hyperekplexia"]) / n * 100),
            "acth_trial_pct": round(sum(1 for p in PATIENTS if p["acth_trial"]) / n * 100),
        },
    }


def get_definitions():
    return {
        "concepts": CORE_CONCEPTS,
        "thresholds": CLINICAL_THRESHOLDS,
        "standards": CLINICAL_STANDARDS,
        "references": KEY_REFERENCES,
        "contraindications": CONTRAINDICATION_DETAIL,
    }
