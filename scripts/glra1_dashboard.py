"""
GLRA1 Hyperekplexia — Hyperekplexia Type 1 / Glycine Receptor Alpha-1 / 5q33.1
=================================================================================
40-patient cohort · GLRA1 (5q33.1) · Glycine receptor α1 subunit · AD/AR

GLRA1 BIOLOGY:
GLRA1 (5q33.1) encodes the glycine receptor alpha-1 subunit (422 aa), the primary
ligand-binding subunit of the inhibitory glycine receptor (GlyR) pentamer at
brainstem and spinal cord synapses. GLRA1 is the MOST COMMON GENETIC CAUSE OF
HYPEREKPLEXIA — responsible for ~70% of all genetically confirmed hyperekplexia
families worldwide.

GLYCINE RECEPTOR STRUCTURE AND FUNCTION:
  The GlyR is a member of the Cys-loop superfamily (same as GABA-A receptor).
  Adult GlyR is a pentamer of α1 + β subunits (α1₂β₃ stoichiometry).
  Each subunit has:
    - N-terminal extracellular domain (ECD): Cys-loop, agonist (glycine) binding
    - 4 transmembrane helices: TM1-TM4
    - TM2 lines the ion pore (chloride channel)
    - Intracellular loop (TM3-TM4): regulatory, scaffold binding
  KEY FUNCTIONAL SITE: Arg271 (TM2, position 2') — the CRITICAL GATE RESIDUE.
    Arg271 contributes to the selectivity filter and gating mechanism.
    Arg271Gln or Arg271Leu → impaired glycine-gated Cl⁻ conductance → loss of
    inhibitory glycinergic transmission → hyperekplexia.

GLRA1 EXPRESSION AND PHYSIOLOGICAL ROLE:
  1. BRAINSTEM RETICULAR FORMATION — key locus; glycinergic inhibition gates
     acoustic and tactile startle responses via startle circuit: cochlear nucleus →
     MNTB → caudal pontine reticular nucleus (PnC) → spinal cord motor neurons.
     GLRA1 LOF at PnC → loss of habituation → exaggerated generalised startle.
  2. SPINAL CORD (dorsal and ventral horn) — glycinergic interneurons mediate
     Renshaw cell inhibition and reciprocal inhibition; GLRA1 LOF → spinal
     hyperexcitability → hypertonia and exaggerated reflexes.
  3. SUPERIOR COLLICULUS — visual startle pathway; auditory startle; blink reflex.
  4. HIPPOCAMPUS AND CORTEX — minor role in adult brain; developmental role in
     early maturation (before GABAergic dominance).

INHERITANCE PATTERNS:
  Autosomal Dominant (AD): one mutant allele → dominant-negative effect on GlyR
    pentamer assembly or channel gating (most missense, especially Arg271 residue).
    ~60% of hyperekplexia families; highly penetrant.
  Autosomal Recessive (AR): biallelic LOF variants (frameshift/nonsense) → complete
    GLRA1 protein absence → severe neonatal phenotype; ~10% of families.
  De Novo: ~30% of AD cases are de novo missense (not inherited).

GENOTYPE–PHENOTYPE CORRELATIONS:
  Arg271Gln/Arg271Leu (TM2): dominant-negative; severe classic hyperekplexia.
  Other TM2/TM1-TM2 linker missense: moderate; variable expressivity.
  Haploinsufficiency (dominant LOF): milder; incomplete penetrance; GGE-like overlap.
  Biallelic null (recessive): severe neonatal rigid-baby; ± co-occurring seizures.

KEY REFERENCES:
  Shiang R et al. (1993) Cell — first identification of Arg271Gln/Arg271Leu in hyperekplexia
  Ryan SG et al. (1992) Nat Genet — hereditary hyperekplexia linkage to 5q
  Rajendra S et al. (1994) EMBO J — TM2 Arg271 mechanism (glycine-gated Cl⁻)
  Lynch JW (2004) Physiol Rev — glycine receptor ion channel physiology and pharmacology
  Harvey RJ et al. (2008) Neuron — GLRB companion gene in hyperekplexia
  Rees MI et al. (2002) Nat Genet — SLC6A5/GlyT2 mutations in hyperekplexia type 2
  Vigevano F et al. (1989) Neuropediatrics — Vigevano (forward-flexion) manoeuvre
  Thomas RH et al. (2010) Brain — genetic spectrum of hyperekplexia
  Carta E et al. (2012) Hum Mol Genet — genotype-phenotype in GLRA1 hyperekplexia
  ILAE Task Force 2017 — hyperekplexia vs epilepsy differential diagnosis
"""
import random

random.seed(495)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GLRA1-Arg271-Dominant-Classic",
        "pct": 40,
        "etiology": "GLRA1 Arg271Gln or Arg271Leu dominant missense — classic severe hyperekplexia; neonatal rigid-baby",
        "mechanism": (
            "Arg271 (TM2 position 2') is the CRITICAL GATE RESIDUE of the glycine receptor pore. "
            "Arg271Gln (c.812G>A) and Arg271Leu (c.812G>T) are dominant-negative: the mutant α1 subunit "
            "co-assembles with wild-type α1 and β subunits to form pentamers with severely impaired "
            "glycine-gated Cl⁻ conductance. Glycine EC₅₀ shifts 10-fold rightward (Rajendra 1994). "
            "The dominant-negative effect means that even one mutant allele poisons the majority of "
            "GlyR pentamers → brainstem reticular formation (PnC) disinhibited → exaggerated startle "
            "circuit activation on any auditory/tactile stimulus → generalised tonic stiffening. "
            "NEONATAL RIGID-BABY SYNDROME: 100% of Arg271 cases present at birth with generalised "
            "hypertonia (rigid baby), exaggerated Moro reflex, and apnoeic startle episodes. "
            "APNOEA: tactile stimulus (handling) → sustained tonic stiffening → respiratory muscle "
            "rigidity → apnoea → hypoxia → risk of SUDEP-equivalent in neonatal period. "
            "FORWARD-FLEXION MANOEUVRE (Vigevano manoeuvre) TERMINATES APNOEA IMMEDIATELY — standard "
            "of care; all carers must be trained before discharge. "
            "Inheritance: AD, highly penetrant; ~30% de novo in this class."
        ),
        "typical_variants": "c.812G>A (p.Arg271Gln) TM2 · c.812G>T (p.Arg271Leu) TM2 · c.811C>T (p.Arg271Cys) TM2",
        "eeg_signature": "Normal EEG; startle event: brief bilateral EMG burst (non-epileptic); nose-tap test: sustained non-habituating flexion response",
        "phenotype": "Severe neonatal rigid-baby; exaggerated startle; apnoeic episodes; clonazepam-responsive; improves age 2-4 years",
        "onset_age_years": 0.0,
        "outcome": "Good with clonazepam; apnoea risk greatest neonatal-infantile; 70-80% asymptomatic or mild by adolescence",
    },
    {
        "category": "GLRA1-LOF-Recessive-Severe",
        "pct": 15,
        "etiology": "GLRA1 biallelic LOF (recessive) — frameshift/nonsense/splice; severe neonatal hyperekplexia ± seizures",
        "mechanism": (
            "Biallelic GLRA1 loss-of-function variants (homozygous or compound heterozygous frameshift, "
            "nonsense, or essential splice site) → complete absence of GLRA1 protein → no functional α1β "
            "pentamers in brainstem/spinal cord → total loss of inhibitory glycinergic tone. "
            "MOST SEVERE PHENOTYPE: neonatal rigid-baby with continuous hypertonia even at rest (not only "
            "startle-triggered), frequent spontaneous apnoeic episodes, feeding difficulties from pharyngeal "
            "hypertonicity. EEG: often abnormal (non-specific high-amplitude, sometimes burst-suppression "
            "pattern in most severe cases reflecting secondary cortical consequences). "
            "CO-OCCURRING SEIZURES: 25-35% of recessive cases develop epileptic seizures (myoclonic, tonic, "
            "absence) distinct from hyperekplexia events — requires both anti-seizure and anti-hyperekplexia "
            "management. "
            "EEG CRITICAL: document startle events (EMG artefact, non-ictal) separately from genuine ictal "
            "discharges — important for directing correct therapy. "
            "Inheritance: AR; often consanguineous families; MENA and South Asian enrichment."
        ),
        "typical_variants": "c.1090_1091insT (p.Ser364Ilefs*6) · c.877C>T (p.Gln293*) nonsense · c.IVS7+1G>A splice · c.1175del (p.Leu392*)",
        "eeg_signature": "Neonatal: non-specific high-amplitude slow; ± burst-suppression in severe. Startle: non-ictal EMG; if seizures: polyspike-wave",
        "phenotype": "Severe rigid-baby; spontaneous + startle apnoea; pharyngeal tone impaired; ± epileptic seizures (25-35%); consanguinity",
        "onset_age_years": 0.0,
        "outcome": "Variable; seizure component may be refractory; hyperekplexia severity partly improves with clonazepam; developmental delay in 30%",
    },
    {
        "category": "GLRA1-Other-Dominant-Missense",
        "pct": 25,
        "etiology": "GLRA1 non-Arg271 dominant missense — TM1-TM2 linker, TM2-TM3, ECD; moderate hyperekplexia",
        "mechanism": (
            "Missense variants outside Arg271 in TM1 (Ala52, Ile244), TM1-TM2 linker (Gln266), "
            "TM2-TM3 linker (Gly342, Asn365), extracellular domain — dominant-negative or dominant "
            "haploinsufficiency mechanism depending on position and structural effect. "
            "MODIFIER EFFECT: Non-Arg271 missense generally retains partial glycine-gated conductance "
            "(EC₅₀ shift 2-5x vs 10x for Arg271) → MODERATE hyperekplexia: present from infancy, "
            "but less severe neonatal apnoea, less rigid-baby rigidity. "
            "VARIABLE EXPRESSIVITY: Within the same family, clinical severity varies widely; some "
            "carriers may be virtually asymptomatic (mild startle response only). "
            "INCOMPLETE PENETRANCE possible for some ECD variants. "
            "NATURAL HISTORY: Most improve substantially by age 3-5 years as secondary GABAergic "
            "compensation develops. Clonazepam effective but doses may be lower than Arg271 class. "
            "Important: non-Arg271 dominant missense = less likely to cause severe neonatal apnoea "
            "but still requires forward-flexion manoeuvre training and neonatal monitoring."
        ),
        "typical_variants": "c.797T>C (p.Ile266Thr) TM1-TM2 linker · c.257T>C (p.Ile86Thr) ECD · c.1031G>A (p.Arg344Gln) TM2-TM3 · c.865G>A (p.Ala289Thr) TM2",
        "eeg_signature": "Normal EEG; moderate startle events (EMG artefact); reduced habituation rate vs Arg271; nose-tap test positive",
        "phenotype": "Moderate hyperekplexia; mild-moderate neonatal rigidity; rare apnoea; improves early childhood; variable expressivity in family",
        "onset_age_years": 0.0,
        "outcome": "Good; clonazepam effective; most seizure-free by adolescence; normal development in majority",
    },
    {
        "category": "GLRA1-LOF-Dominant-Haploinsufficiency",
        "pct": 10,
        "etiology": "GLRA1 dominant haploinsufficiency (nonsense/frameshift het) — mild/incomplete hyperekplexia; GGE-like overlap",
        "mechanism": (
            "Heterozygous frameshift or nonsense GLRA1 variants → protein truncation; NMD-degraded transcript → "
            "haploinsufficiency (~50% reduction in α1 protein). "
            "Unlike dominant-negative missense (Arg271), haploinsufficiency allows remaining WT α1 to form "
            "some functional α1β pentamers → PARTIAL loss of glycinergic inhibition → MILDER phenotype. "
            "PHENOTYPIC RANGE: Some het carriers have only subclinical startle enhancement (positive "
            "nose-tap test on exam but no spontaneous events); others have clear clinical hyperekplexia. "
            "GGE-LIKE OVERLAP: A minority of GLRA1 het-LOF carriers present with febrile seizures, "
            "absence epilepsy, or JME-like phenotype — mechanistically, glycine receptor haploinsufficiency "
            "can modestly reduce seizure threshold in predisposed individuals. "
            "FAMILY SCREENING: when proband has de novo severe hyperekplexia and parents have only "
            "mild startle → parent may carry het-LOF with subclinical phenotype; risk to siblings 50%. "
            "Clonazepam required only in symptomatic individuals; asymptomatic carriers monitored."
        ),
        "typical_variants": "c.456_457del (p.His153Glnfs*14) het · c.1132C>T (p.Gln378*) het · c.IVS4+2T>C splice het · c.301C>T (p.Arg101*) het",
        "eeg_signature": "Normal or mild diffuse slowing; if seizures: generalised spike-wave; nose-tap: mildly positive or borderline",
        "phenotype": "Mild/subclinical hyperekplexia; variable penetrance; occasional febrile seizures / GGE-like; positive family history",
        "onset_age_years": 2.0,
        "outcome": "Good; mild or no functional impairment; seizures (if present) responsive to standard GGE therapy; clonazepam only if symptomatic",
    },
    {
        "category": "Phenocopy-GlyR-Other",
        "pct": 10,
        "etiology": "Phenocopy — GLRB / SLC6A5 / GPHN / ARHGEF9 hyperekplexia or non-genetic mimics (GLRA1 negative)",
        "mechanism": (
            "Clinically GLRA1-like hyperekplexia (neonatal rigid-baby, exaggerated startle, apnoea) but "
            "GLRA1 sequencing and deletion analysis NEGATIVE. Confirmed alternative diagnoses include: "
            "GLRB (OMIM 138492, 4q32.1): glycine receptor beta subunit; ~5% of hyperekplexia; AR/AD; "
            "  GLRB variants reduce WT β expression → α1homomeric GlyR forms (reduced conductance). "
            "SLC6A5 / GlyT2 (OMIM 604159, 11p15.1): glycine transporter-2; AR; hyperekplexia type 2 (OMIM 614618); "
            "  reduced glycine reuptake → paradoxically depleted presynaptic glycine stores → impaired "
            "  glycinergic inhibition. Most common recessive non-GLRA1 cause. "
            "GPHN / Gephyrin (OMIM 603930, 14q23.3): glycine receptor anchoring protein; LOF → reduced "
            "  GlyR clustering at postsynaptic densities → impaired inhibition; also causes hyperekplexia. "
            "ARHGEF9 / CollybistinX-linked (OMIM 300429, Xq11.1): RhoGEF anchoring GlyR at inhibitory "
            "  synapses; X-linked recessive; hyperekplexia + epilepsy + intellectual disability in males. "
            "NON-GENETIC MIMICS: Neonatal tetanus, stiff-baby syndrome (opsoclonus-myoclonus, PNKP), "
            "  biotinidase deficiency, methylmalonic aciduria — must be excluded before genetic diagnosis."
        ),
        "typical_variants": "GLRB p.Trp170Ser (AR) · SLC6A5 c.1219C>T (AR) · GPHN frameshift · ARHGEF9 hemizygous (X-linked) · Non-genetic excluded",
        "eeg_signature": "Variable: GLRB/SLC6A5 similar to GLRA1; GPHN/ARHGEF9 may have ictal discharges; ARHGEF9 males: epileptiform",
        "phenotype": "GLRA1-like clinically; alternative gene confirmed; GPHN/ARHGEF9 may have epilepsy + ID component",
        "onset_age_years": 0.0,
        "outcome": "Gene-specific; GLRB/SLC6A5 respond to clonazepam; ARHGEF9 males: complex (epilepsy + hyperekplexia + ID)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HYPEREKPLEXIA EVENT TYPES  (distinct from seizure types — these are NOT epileptic)
# ─────────────────────────────────────────────────────────────────────────────
EVENT_DETAIL = [
    {
        "type": "Exaggerated Startle Response (Generalised Tonic Stiffening)",
        "prevalence_pct": 100,
        "semiology": (
            "Triggered by sudden auditory (loud noise, clap), tactile (touching, handling), or "
            "visual (bright flash) stimulus. Response: 1) INITIAL SCARE — rapid eye-blink, arm abduction, "
            "then 2) GENERALISED STIFFENING — sustained bilateral tonic rigidity of limbs and trunk "
            "(unlike epileptic tonic-clonic: NO clonic phase, NO post-ictal state, rapid recovery "
            "within seconds). FALLS WITHOUT LOSS OF CONSCIOUSNESS: ambulatory patients fall stiffly "
            "forward — injury risk. NON-HABITUATING: normal startle habituates after 3-5 repetitions; "
            "hyperekplexia startle does NOT habituate → the non-habituation distinguishes it from "
            "the normal exaggerated startle of infants/anxious individuals."
        ),
        "eeg_pattern": "NORMAL EEG (essential finding) — the generalised stiffening has NO ictal correlate. Only surface EMG artefact seen during event. CRITICAL: normal EEG at event onset rules out epileptic tonic seizure.",
        "clinical_tip": (
            "Nose-tap test (Iles–Vigevano test): tap tip of nose suddenly → POSITIVE = repetitive "
            "non-habituating tonic flexion response (legs flex, arms flex, ≥3 repetitions). "
            "Pathognomonic for hyperekplexia. Perform at every clinic visit to monitor response to "
            "clonazepam (improvement = positive test becomes weaker or habituates). "
            "DOCUMENT: clearly label events as 'non-epileptic hyperekplexia events' in all "
            "correspondence — prevents unnecessary AED escalation."
        ),
    },
    {
        "type": "Neonatal Apnoeic Hyperekplexia (Emergency)",
        "prevalence_pct": 72,
        "semiology": (
            "MOST DANGEROUS MANIFESTATION. Sequence: sudden stimulus (handling, nappy change, "
            "loud noise) → generalised tonic stiffening → respiratory muscles rigid → sustained "
            "APNOEA (5-60 seconds) → cyanosis → hypoxia → risk of hypoxic injury or death. "
            "In Arg271 and recessive classes, this occurs frequently in neonatal period with minimal "
            "stimulation. May appear 'out of the blue' without obvious trigger. "
            "CRITICAL DISTINCTION FROM NEONATAL SEIZURES: apnoea is TONIC (stiff, not jerking), "
            "no ictal EEG, immediately responsive to forward-flexion manoeuvre — seizures are not "
            "terminated by positioning. EEG monitoring during event confirms non-ictal nature."
        ),
        "eeg_pattern": "EEG: NON-ICTAL — only EMG artefact during tonic phase; no epileptiform discharge. EEG during apnoeic episode: essential to confirm non-epileptic aetiology.",
        "clinical_tip": (
            "FORWARD-FLEXION MANOEUVRE (Vigevano manoeuvre): IMMEDIATE FIRST-LINE TREATMENT for "
            "apnoeic hyperekplexia. Technique: flex head toward chest + flex knees toward abdomen "
            "(foetal position) → releases tonic rigidity within seconds → restores breathing. "
            "ALL CARERS (parents, nurses, first responders) MUST LEARN BEFORE DISCHARGE. "
            "Prescribe clonazepam orally (0.01-0.05 mg/kg/day) to reduce frequency. "
            "If apnoea persists despite manoeuvre: bag-mask ventilation; do NOT give phenytoin "
            "or phenobarbital (not effective for hyperekplexia). NICU monitoring mandatory in "
            "severe neonatal cases until clonazepam effective."
        ),
    },
    {
        "type": "Neonatal Generalised Hypertonia (Rigid-Baby Syndrome)",
        "prevalence_pct": 85,
        "semiology": (
            "Continuous generalised hypertonia at rest in neonates with severe GLRA1 mutations "
            "(especially Arg271 and recessive classes). The infant is persistently stiff (rigid-baby), "
            "with difficulty flexing limbs for examination, poor feeding (pharyngeal rigidity), "
            "and reduced spontaneous movement. "
            "DISTINGUISHES FROM HYPOTONIA: opposite of floppy baby — RIGID baby. "
            "RESOLVES over weeks to months on clonazepam and with developmental maturation "
            "(secondary GABAergic compensation). Some degree of hypertonia may persist. "
            "DIFFERENTIAL: neonatal tetanus (Clostridium tetani toxin — check immunisation history), "
            "stiff-baby syndrome, Kernicterus, glycine encephalopathy (hyperGlycinaemia — plasma "
            "glycine elevated), biotinidase deficiency."
        ),
        "eeg_pattern": "Normal EEG at rest; no interictal epileptiform discharges in isolated GLRA1 hypertonia (unlike glycine encephalopathy/NKH which has burst-suppression)",
        "clinical_tip": (
            "Clonazepam IV/NG in NICU reduces hypertonia progressively. Physiotherapy + careful handling "
            "protocols. METABOLIC SCREEN mandatory: plasma amino acids (exclude non-ketotic "
            "hyperglycinaemia/NKH — CSF:plasma glycine ratio >0.08 = NKH, NOT hyperekplexia). "
            "URINE ORGANIC ACIDS: exclude methylmalonic aciduria. BIOTINIDASE ACTIVITY. "
            "These metabolic causes are TREATABLE and must not be missed."
        ),
    },
    {
        "type": "Childhood/Adult Startle Falls (Drop Attacks)",
        "prevalence_pct": 55,
        "semiology": (
            "In older patients with inadequate or no treatment: sudden startle (doorbell, "
            "car horn, unexpected touch) → whole-body stiffening → falls to ground stiffly "
            "(like a falling tree). NO LOC, immediate recovery. HIGH INJURY RISK (head trauma, "
            "fractures). Patients develop PHOBIC AVOIDANCE of public spaces, loud environments. "
            "Hyperekplexia falls are often misclassified as drop attacks from epilepsy or "
            "cataplexy — EEG during an event is diagnostic (no ictal correlate). "
            "On adequate clonazepam: falls dramatically reduced or eliminated. "
            "Helmet and protective equipment for severe cases during titration."
        ),
        "eeg_pattern": "Normal EEG; non-ictal during startle event. Ambulatory EEG + video synchronised crucial to document fall mechanism.",
        "clinical_tip": (
            "Clonazepam optimisation first-line. SAFETY ENVIRONMENT: remove sharp furniture, "
            "padded rugs, avoid standing near stairs/pools unsupervised. "
            "Medical alert bracelet: 'Hyperekplexia — not epilepsy — forward-flexion if rigid'. "
            "Driver safety assessment mandatory in adults (startle while driving is a road risk). "
            "Piracetam second-line if clonazepam inadequate."
        ),
    },
    {
        "type": "Neonatal Apnoea of Epileptic Origin (co-existing, recessive class only)",
        "prevalence_pct": 22,
        "semiology": (
            "IN RECESSIVE CLASS ONLY: 25-35% develop genuine epileptic seizures DISTINCT from "
            "hyperekplexia events. Seizure types: myoclonic jerks (brief, ictal correlate on EEG), "
            "tonic seizures (longer, EEG ictal), absence seizures. These patients require DUAL "
            "THERAPY: clonazepam for hyperekplexia events + appropriate AED for epileptic seizures. "
            "CRITICAL: document every event with video-EEG to classify. Treatment mismatch "
            "(treating hyperekplexia events with AEDs, or epileptic seizures with only CLZ) "
            "leads to under-treatment of one component. "
            "In some recessive biallelic GLRA1 cases: complex-mixed phenotype resembling SCN1A-Dravet."
        ),
        "eeg_pattern": "HYPEREKPLEXIA events: non-ictal. EPILEPTIC seizures: polyspike-wave (myoclonic), low-voltage fast (tonic), generalised spike-wave (absence). Differentiation requires synchronised video-EEG.",
        "clinical_tip": (
            "VIDEO-EEG mandatory for recessive cases — classify EACH event type. "
            "Anti-seizure: VPA (if POLG excluded) or LEV for myoclonic/absence. "
            "Anti-hyperekplexia: clonazepam first-line. "
            "AVOID: LTG (aggravates myoclonic seizures if present). CBZ/OXC (may worsen myoclonic)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_DETAIL = [
    {"trigger": "Sudden Auditory Stimulus (loud noise, clap, doorbell)",
     "pct": 98, "note": "Primary trigger. Acoustic startle circuit: cochlear nucleus → MNTB → PnC → spinal motor neuron. GLRA1 LOF at PnC → non-habituating startle. Reduce sudden noise exposure: soft doorbells, advance warning before touching infant."},
    {"trigger": "Sudden Tactile Stimulus (unexpected touch, handling, nappy change)",
     "pct": 92, "note": "CRITICAL in neonates — routine handling triggers apnoeic episodes. All carers trained in forward-flexion manoeuvre before NICU discharge. Handle gently, with advance verbal warning."},
    {"trigger": "Sleep/Wake Transition (hypnic jerk context)",
     "pct": 68, "note": "Startle events more common at sleep onset or awakening when cortical inhibition is reduced. Monitor for nocturnal hyperekplexia. Clonazepam sedative effect can reduce nocturnal events."},
    {"trigger": "Emotional Arousal (excitement, fright, anxiety)",
     "pct": 55, "note": "Heightened arousal lowers startle threshold. In children: laughter or sudden joy can trigger. Cognitive-behavioural relaxation reduces anticipatory anxiety in older patients."},
    {"trigger": "Visual Flash / Photic Stimulus",
     "pct": 32, "note": "Visual pathway startle (superior colliculus). Avoid stroboscopic environments. Tinted glasses may reduce light-triggered events. Distinguishing from photoparoxysmal response requires EEG."},
    {"trigger": "Fever / Intercurrent Illness",
     "pct": 28, "note": "Thermal sensitivity of GlyR: elevated temperature reduces glycine EC50 further in mutant receptors → worsening hyperekplexia during febrile illness. Pre-emptive clonazepam dose increase plan."},
    {"trigger": "Stress / Sleep Deprivation",
     "pct": 22, "note": "Global excitability increase reduces startle threshold. Sleep hygiene critical. Stress management (especially in adolescents/adults with phobic avoidance behaviours)."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT ARSENAL
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_DETAIL = [
    {
        "drug": "Clonazepam (CLZ)",
        "level": "Level A — First-Line (All Classes)",
        "moa": (
            "Benzodiazepine positive allosteric modulator of GABA-A receptors — increases GABA-A "
            "Cl⁻ conductance → compensates for lost glycinergic inhibition at brainstem PnC and "
            "spinal cord → reduces startle circuit hyperexcitability. Does NOT directly enhance GlyR "
            "(glycine receptor is not a BZD target), but GABAergic compensation is potent enough to "
            "markedly reduce hyperekplexia events in the majority of patients."
        ),
        "dose": "Neonate: 0.01-0.03 mg/kg/day PO/NG; infant: 0.02-0.05 mg/kg/day; child: 0.05-0.1 mg/kg/day; titrate to effect",
        "efficacy": "Level A: international standard first-line for all genetic hyperekplexia; 80-90% significant event reduction",
        "safety": "Sedation (dose-limiting); respiratory depression (neonates — monitor SpO₂); tolerance (long-term); paradoxical agitation in children",
        "monitoring": "SpO₂ monitoring neonatal initiation; sedation scale; respiratory rate; developmental milestones (avoid excessive sedation during critical periods)",
        "glra1_note": "GLRA1-SPECIFIC: Clonazepam is effective via GABAergic compensation (not direct GlyR action). Higher doses may be needed in Arg271 class vs non-Arg271 (more severe dominant-negative → greater compensation required). Dose reduction attempt at age 3-4 years when natural GlyR maturation improves.",
    },
    {
        "drug": "Forward-Flexion Manoeuvre (Vigevano Manoeuvre)",
        "level": "Level A — Acute Apnoea (Emergency First Aid)",
        "moa": (
            "Physical manoeuvre that terminates apnoeic hyperekplexia attacks by flexing the "
            "patient into the foetal position (head toward chest, knees toward abdomen). "
            "Mechanism: flexion activates proprioceptive/cutaneous feedback that inhibits the "
            "tonic rigidity via spinal cord flexion reflexes and brainstem reciprocal inhibition. "
            "The manoeuvre releases the tonic stiffening within seconds → restores spontaneous "
            "respiratory effort. Described by Vigevano et al. (1989) for neonatal hyperekplexia."
        ),
        "dose": "Technique: hold infant with one hand behind head/neck, other behind knees; flex head toward chest while flexing knees toward abdomen simultaneously; hold 3-5 seconds → release",
        "efficacy": "Level A: immediate life-saving; terminates neonatal apnoea in seconds; ALL carers must demonstrate competency before patient discharge",
        "safety": "Safe when performed correctly; gentle pressure only; do NOT force flexion in newborns with myelomeningocele or suspected cervical spine instability",
        "monitoring": "Document carer training in medical record; repeat demonstration at every visit; ensure any new carer (grandparent, nursery) is also trained",
        "glra1_note": "GLRA1 APNOEA STANDARD: This manoeuvre is the ONLY intervention that terminates hyperekplexia apnoea instantly. Bag-mask ventilation is the backup if manoeuvre fails. NEVER administer AEDs (phenytoin, phenobarbital) for apnoeic hyperekplexia events — they are ineffective and may cause respiratory depression.",
    },
    {
        "drug": "Sodium Valproate (VPA)",
        "level": "Level B — Adjunct or Recessive-Seizure Component",
        "moa": "Multiple mechanisms: GAT-1 GABA reuptake inhibition → increased synaptic GABA → GABAergic compensation for glycinergic deficit. Na⁺ channel modulation. GABA-T inhibition. Additionally used for EPILEPTIC SEIZURES in recessive-class GLRA1 patients.",
        "dose": "20-40 mg/kg/day in divided doses; monitor trough level 50-100 mg/L",
        "efficacy": "Level B for hyperekplexia adjunct (limited evidence); Level A for myoclonic/absence seizures in recessive-class co-morbid epilepsy",
        "safety": "POLG SCREEN MANDATORY before use in any infant/child with encephalopathy. Hepatotoxicity (fatal in POLG/Alpers). Teratogenicity. Pancreatitis. Hyperammonaemia.",
        "monitoring": "POLG1 sequencing before initiation. LFTs weekly ×4 then monthly ×6. Ammonia if drowsy. VPA level trough.",
        "glra1_note": "In recessive GLRA1 with co-existing epileptic seizures: VPA Level A for myoclonic component. POLG MANDATORY before VPA regardless of GLRA1 diagnosis. For isolated hyperekplexia: VPA is second-line adjunct to clonazepam with limited evidence; piracetam preferred as adjunct.",
    },
    {
        "drug": "Piracetam",
        "level": "Level C — Historical Second-Line Adjunct",
        "moa": "Unclear; possible modulation of AMPA receptor function; reduces startle-evoked EMG amplitude in some patients; may act on neuroprotective pathways. Used historically before clonazepam became standard.",
        "dose": "Adult/adolescent: 2.4-4.8 g/day in divided doses; paediatric: 40-100 mg/kg/day",
        "efficacy": "Level C (historical case series); modestly reduces event frequency in some patients; less effective than clonazepam; useful as add-on when clonazepam causes unacceptable sedation",
        "safety": "Generally well-tolerated; agitation in some; rare bleeding tendency (platelet aggregation reduction); avoid in renal impairment",
        "monitoring": "Renal function (dose adjust in CKD); platelet count if surgical procedures planned",
        "glra1_note": "Use when clonazepam is insufficient and before resorting to higher CLZ doses (to reduce sedation). Piracetam + low-dose CLZ may achieve better efficacy/tolerability balance than high-dose CLZ alone in older children and adults.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — Adjunct (Seizures in Recessive Class)",
        "moa": "SV2A synaptic vesicle modulator → reduces neurotransmitter release → broad-spectrum anti-seizure; used for epileptic seizures in recessive-class GLRA1 with comorbid epilepsy (NOT for hyperekplexia events per se)",
        "dose": "20-60 mg/kg/day; 1000-3000 mg/day adult; IV available",
        "efficacy": "Level B for focal and generalised seizures; myoclonic component in recessive GLRA1 epilepsy",
        "safety": "Behavioural side effects (irritability, aggression) — particularly problematic in already-stressed hyperekplexia families; monitor behavioural scales",
        "monitoring": "Behavioural scales quarterly; renal dosing in CKD",
        "glra1_note": "Use ONLY for epileptic seizure component (EEG-confirmed ictal events) in recessive GLRA1. LEV is NOT effective for non-epileptic hyperekplexia events. Ensure family understands distinction between epileptic seizures (LEV target) and hyperekplexia events (clonazepam + manoeuvre).",
    },
    {
        "drug": "Clonazepam Dose Weaning (Age 3-5 years)",
        "level": "Standard Practice — Natural History Management",
        "moa": "Natural neurological maturation: (1) postnatal switch from predominantly glycinergic to GABAergic inhibitory dominance; (2) secondary GABAergic upregulation compensates for chronic GlyR deficit; (3) reduced brainstem sensitivity to startle with cortical inhibitory maturation. Result: hyperekplexia events reduce in severity and frequency in late infancy/early childhood.",
        "dose": "Attempt gradual dose reduction starting age 3-4 years; reduce by 10-20% every 4-8 weeks; stop if symptom-free for >12 months",
        "efficacy": "70-80% of patients can reduce or discontinue clonazepam by late childhood/adolescence; residual mild startle may persist without functional impairment",
        "safety": "Withdrawal seizures if weaned too rapidly — taper slowly. Rebound hyperekplexia if weaned during stress/illness — have rescue clonazepam plan.",
        "monitoring": "Careful clinical monitoring during weaning; nose-tap test at each visit; parent/patient-reported event frequency diary",
        "glra1_note": "GLRA1-SPECIFIC: Arg271 class often requires clonazepam into adolescence (dominant-negative mechanism ongoing). Non-Arg271 dominant missense: earlier weaning often possible. Recessive with epilepsy: AED weaning separate from CLZ weaning.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS / HIGH CAUTIONS
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATION_DETAIL = [
    {
        "drug": "Phenytoin (PHT) / Carbamazepine (CBZ) for hyperekplexia",
        "risk": "INEFFECTIVE (not contraindicated per se, but WRONG DRUG)",
        "reason": "Na⁺ channel blockers have NO efficacy for glycine receptor–mediated hyperekplexia. Administration for apnoeic hyperekplexia wastes critical time and may cause respiratory depression. Never use PHT/CBZ as primary management for GLRA1-related apnoea or stiffening. Reserve for confirmed co-existing epileptic seizures only.",
    },
    {
        "drug": "VPA — without prior POLG screening (recessive class ± epilepsy)",
        "risk": "ABSOLUTE CI — Without POLG Screen",
        "reason": "Recessive GLRA1 patients with comorbid epilepsy who need VPA: POLG1 sequencing MANDATORY first. Any severe infantile encephalopathy may co-harbour POLG variant — VPA in POLG = Alpers-Huttenlocher (fatal hepatic failure). Cannot be assumed safe without testing.",
    },
    {
        "drug": "LTG (Lamotrigine) — if myoclonic seizures present (recessive class)",
        "risk": "ABSOLUTE CI — Myoclonic Seizures",
        "reason": "If recessive GLRA1 patient has comorbid myoclonic seizures (EEG-confirmed): LTG aggravates myoclonic component. LTG is safe in pure hyperekplexia (no myoclonic epileptic component) but high-risk if myoclonic co-occurs.",
    },
    {
        "drug": "Rapid Clonazepam withdrawal",
        "risk": "HIGH CAUTION — Rebound Hyperekplexia + Withdrawal Seizures",
        "reason": "Abrupt CLZ discontinuation → BZD withdrawal (autonomic instability, withdrawal seizures) + rebound severe hyperekplexia (often worse than pre-treatment baseline). Taper over weeks-months minimum. Always have rescue CLZ available during weaning.",
    },
    {
        "drug": "Neonatal Discharge without Forward-Flexion Training",
        "risk": "ABSOLUTE SAFETY REQUIREMENT",
        "reason": "Discharging a neonate with GLRA1 hyperekplexia without documented forward-flexion manoeuvre training for ALL primary carers is a patient safety failure. Apnoeic attacks outside hospital without knowledge of the manoeuvre = risk of hypoxic death. Competency-based training + written instruction card mandatory.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_THRESHOLDS = [
    {"label": "CLZ Starting Dose (Neonate)", "value": "0.01–0.03", "unit": "mg/kg/day"},
    {"label": "CLZ Maintenance Dose (Child)", "value": "0.05–0.1", "unit": "mg/kg/day"},
    {"label": "Forward-Flexion Response Time", "value": "<5", "unit": "seconds (apnoea terminates)"},
    {"label": "Nose-Tap Test Threshold (Positive)", "value": "≥3", "unit": "non-habituating repetitions"},
    {"label": "Weaning Attempt Age", "value": "3–5", "unit": "years (gradual 10-20%/4-8 weeks)"},
    {"label": "Plasma Glycine (NKH screen)", "value": ">0.08", "unit": "CSF:plasma ratio = NKH, not hyperekplexia"},
    {"label": "VPA Level (if used)", "value": "50–100", "unit": "mg/L trough"},
    {"label": "Apnoea Monitor Duration (Neonates)", "value": "Until CLZ effective", "unit": "continuous SpO₂ + apnoea alarm"},
    {"label": "CLZ Taper Rate", "value": "10–20%", "unit": "per 4-8 weeks (slow taper)"},
    {"label": "Piracetam Dose (if adjunct)", "value": "40–100", "unit": "mg/kg/day (paediatric)"},
    {"label": "GlyR Maturation Window", "value": "2–5", "unit": "years (spontaneous improvement expected)"},
    {"label": "Nose-Tap Test Frequency", "value": "Every clinic visit", "unit": "documents CLZ response"},
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_SCHEDULE = [
    {"item": "SpO₂ / Apnoea monitor (neonates on CLZ initiation)", "frequency": "Continuous; until stable on CLZ and apnoea-free ≥48h"},
    {"item": "Video-EEG during event", "frequency": "Once (to confirm non-epileptic aetiology); repeat if seizures suspected"},
    {"item": "Nose-tap test (hyperekplexia response gauge)", "frequency": "Every clinic visit (document habituation / event frequency)"},
    {"item": "CLZ dose review", "frequency": "Every 3 months (titrate to effect; monitor sedation)"},
    {"item": "Developmental assessment (Griffiths/Bayley)", "frequency": "Every 6 months (0-3 years); annually (3-6 years)"},
    {"item": "Forward-flexion manoeuvre carer competency", "frequency": "Confirmed before discharge; re-assessed annually; any new carer"},
    {"item": "Metabolic screen (plasma AA, urine OA, biotinidase)", "frequency": "Once at diagnosis (exclude NKH, MMA, biotinidase deficiency)"},
    {"item": "CSF glycine / CSF:plasma glycine ratio", "frequency": "Once if NKH suspected (bursting EEG, elevated plasma glycine)"},
    {"item": "Plasma VPA level (if on VPA)", "frequency": "1 week post-initiation; then every 6 months"},
    {"item": "LFTs + ammonia (if on VPA)", "frequency": "Weekly ×4, monthly ×6, quarterly thereafter"},
    {"item": "POLG1 sequencing (before VPA in any encephalopathy)", "frequency": "Once, before first VPA prescription"},
    {"item": "CLZ weaning trial", "frequency": "Consider at age 3-5 years; gradual taper; monitor event diary"},
    {"item": "Event diary (hyperekplexia events per day/week)", "frequency": "Ongoing; reviewed at each clinic visit"},
    {"item": "Genetic cascade testing (family members at risk)", "frequency": "After proband diagnosis; AD: 50% risk to siblings/offspring"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {"window": "Neonatal (0–4 weeks)", "headline": "Rigid-baby + apnoeic hyperekplexia crisis; NICU monitoring; CLZ initiation; forward-flexion carer training; metabolic screen (exclude NKH)"},
    {"window": "Infantile (1–12 months)", "headline": "CLZ dose titration; apnoea resolution; development monitoring; video-EEG event characterisation; recessive: watch for co-occurring seizures"},
    {"window": "Early Childhood (1–5 years)", "headline": "Natural GlyR maturation reduces severity; CLZ weaning attempt at age 3-4y; developmental follow-up; school safety plan for startle falls"},
    {"window": "Childhood (5–12 years)", "headline": "CLZ dose reduction/cessation in many; safety environment at school; phobic avoidance counselling; swimming safety (supervised only)"},
    {"window": "Adolescence (12–18 years)", "headline": "Peer awareness; driving assessment mandatory; piracetam adjunct if residual; psychosocial support (phobic avoidance); genetic counselling for family planning"},
    {"window": "Adulthood", "headline": "Most asymptomatic or mild; residual startle in minority; AD: 50% risk to offspring (prenatal genetics); occasional CLZ restart in stressful periods"},
]

# ─────────────────────────────────────────────────────────────────────────────
# CORE CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────
CORE_CONCEPTS = [
    {"term": "GLRA1", "definition": "Gene encoding glycine receptor alpha-1 subunit (422 aa, 5q33.1); primary agonist-binding subunit of the adult inhibitory GlyR pentamer (α1₂β₃). Most common genetic cause of hyperekplexia (~70% of families)."},
    {"term": "Hyperekplexia (Startle Disease)", "definition": "Non-epileptic disorder characterised by exaggerated, NON-HABITUATING startle responses to sudden auditory/tactile/visual stimuli → generalised tonic stiffening, falls, neonatal apnoea. Normal EEG at event. Treatment: clonazepam + forward-flexion manoeuvre."},
    {"term": "Glycine Receptor (GlyR) Alpha-1", "definition": "Cys-loop Cl⁻ channel; TM2 helix lines the pore; Arg271 (TM2 position 2') is the critical gate residue. Glycine binding opens Cl⁻ channel → hyperpolarisation → inhibition of brainstem startle circuit neurons and spinal motor neurons."},
    {"term": "Arg271 Dominant-Negative Mechanism", "definition": "Arg271Gln or Arg271Leu: mutant α1 subunit co-assembles with WT α1 in GlyR pentamers; even one mutant subunit poisons glycine-gated conductance (10-fold EC50 shift); dominant effect means 50% allele load impairs >50% of functional receptors."},
    {"term": "Non-Habituating Startle", "definition": "Pathognomonic feature: normal startle habituates after 3-5 repetitions; hyperekplexia startle does NOT habituate on repeated stimulus. Basis of the nose-tap test."},
    {"term": "Forward-Flexion (Vigevano) Manoeuvre", "definition": "LIFE-SAVING emergency treatment for apnoeic hyperekplexia: flex head toward chest + knees toward abdomen simultaneously → terminates tonic rigidity in seconds → restores breathing. ALL carers must learn before discharge. Not effective for epileptic seizures — distinguishing feature."},
    {"term": "Nose-Tap Test (Iles-Vigevano test)", "definition": "Tap tip of nose sharply: positive result = repetitive, non-habituating flexion response (≥3 repetitions). Pathognomonic for hyperekplexia. Used at every clinic visit to gauge severity and treatment response."},
    {"term": "Rigid-Baby Syndrome", "definition": "Neonatal generalised hypertonia (stiffness at rest + at startle) in severe GLRA1 hyperekplexia. Distinguishes from hypotonic (floppy) conditions. Differential: neonatal tetanus, NKH (non-ketotic hyperglycinaemia), biotinidase deficiency."},
    {"term": "Non-Ketotic Hyperglycinaemia (NKH)", "definition": "Critical DIFFERENTIAL DIAGNOSIS for rigid-baby: NKH (glycine cleavage system defect) → elevated plasma + CSF glycine → burst-suppression EEG + severe encephalopathy. CSF:plasma glycine ratio >0.08 = NKH. GLRA1 hyperekplexia: plasma glycine NORMAL."},
    {"term": "Clonazepam (CLZ) Mechanism in Hyperekplexia", "definition": "CLZ is NOT a GlyR drug. It acts on GABA-A receptors → GABAergic compensation for the loss of glycinergic inhibition at brainstem PnC and spinal cord. This indirect compensation reduces startle circuit hyperexcitability effectively."},
    {"term": "Natural History and GlyR Maturation", "definition": "Hyperekplexia severity decreases with age (typically by age 2-5 years) due to: (1) secondary GABAergic upregulation compensates for GlyR deficit; (2) postnatal maturation of corticospinal inhibitory pathways. Most patients can reduce/stop CLZ in childhood."},
    {"term": "Brainstem Startle Circuit", "definition": "Acoustic startle: sound → cochlear nucleus → MNTB (medial nucleus of the trapezoid body) → PnC (caudal pontine reticular nucleus) → spinal cord ventral horn motor neurons. GLRA1 LOF disinhibits PnC → exaggerated, non-habituating startle. CLZ increases GABAergic inhibition at PnC."},
    {"term": "Hyperekplexia vs Epileptic Tonic Seizure", "definition": "KEY DISTINCTION: Hyperekplexia: NO ictal EEG correlate; triggered by startle; terminated by forward-flexion manoeuvre; non-epileptic; clonazepam + manoeuvre (not AEDs). Epileptic tonic seizure: ictal EEG; not reliably trigger-specific; NOT terminated by manoeuvre; treat with AEDs."},
    {"term": "GLRB (Glycine Receptor Beta)", "definition": "Companion gene (4q32.1): β subunit partners with GLRA1 in adult GlyR pentamers. GLRB mutations (AR/AD) cause hyperekplexia type 1B. SLC6A5/GlyT2 (11p15.1) mutations cause hyperekplexia type 2. GPHN (gephyrin) and ARHGEF9 (collybistin) cause hyperekplexia via GlyR anchoring defects."},
    {"term": "Driver Safety Assessment (Adults)", "definition": "Hyperekplexia startle while driving = sudden loss of vehicle control risk. All adult patients must have formal driving assessment by neurologist + relevant licensing authority notification. Patients with frequent uncontrolled events should not drive."},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_STANDARDS = [
    "ILAE 2017 Classification — Hyperekplexia: non-epileptic paroxysmal event; differential diagnosis of epileptic tonic seizures; EEG mandatory to classify events",
    "Forward-Flexion Manoeuvre (Vigevano 1989) — Level A first-line for neonatal apnoeic hyperekplexia; competency-based carer training BEFORE hospital discharge, documented in medical record",
    "Clonazepam — Level A first-line pharmacological treatment; dose: 0.01-0.05 mg/kg/day neonate-infant; titrate to event control with minimal sedation",
    "Nose-Tap Test — performed at every clinic visit; documents severity and treatment response; positive = non-habituating response ≥3 repetitions",
    "Metabolic Screen at Diagnosis — plasma amino acids (exclude NKH), urine organic acids, biotinidase activity; CSF glycine if NKH suspected (EEG burst-suppression + elevated plasma Gly)",
    "Video-EEG — at least one synchronised event recording to confirm non-epileptic aetiology (no ictal correlate); repeat if clinical picture changes",
    "POLG Screening — MANDATORY before VPA in any infant/child with encephalopathy or recessive GLRA1 with comorbid seizures (Alpers-Huttenlocher risk)",
    "Genetic Cascade Testing — AD families: 50% offspring/sibling risk; prenatal/preimplantation genetic testing available; recessive families: carrier testing for parents and siblings",
    "Medical Alert Bracelet — 'Hyperekplexia — not epilepsy — forward-flexion manoeuvre if rigid apnoea'; prevents inappropriate AED administration by emergency services",
    "NICU Monitoring — continuous SpO₂ and apnoea alarm until CLZ effective and apnoea-free ≥48h; delay discharge until forward-flexion competency confirmed in all primary carers",
    "Driver Safety — all adults with hyperekplexia: formal driving assessment mandatory; uncontrolled events = driving prohibition; notify licensing authority per jurisdiction",
    "Natural History Monitoring — attempt CLZ weaning age 3-5 years; 70-80% can reduce or stop; persistence of events into adulthood warrants genetic re-evaluation (consider phenocopy genes)",
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
KEY_REFERENCES = [
    "Shiang R et al. (1993) Cell — Discovery: missense mutations (Arg271Gln/Arg271Leu) in GLRA1 cause hyperekplexia",
    "Ryan SG et al. (1992) Nat Genet — Hereditary hyperekplexia linkage to chromosome 5q (pre-GLRA1 cloning)",
    "Rajendra S et al. (1994) EMBO J — TM2 Arg271 is the critical gate residue; mutation reduces glycine-gated Cl⁻ conductance",
    "Lynch JW (2004) Physiol Rev — Molecular structure and physiology of the glycine receptor chloride channel",
    "Harvey RJ et al. (2008) Neuron — GLRB companion subunit mutations in hyperekplexia",
    "Rees MI et al. (2002) Nat Genet — SLC6A5 (GlyT2) mutations in hyperekplexia type 2 (non-GLRA1)",
    "Vigevano F et al. (1989) Neuropediatrics — Forward-flexion manoeuvre for neonatal hyperekplexia apnoea",
    "Thomas RH & Rees MI (2014) Clin Genet — Hyperekplexia genetic spectrum: GLRA1 accounts for ~70%",
    "Carta E et al. (2012) Hum Mol Genet — Genotype-phenotype correlations in GLRA1 hyperekplexia",
    "ILAE Task Force (2017) Epilepsia — Non-epileptic paroxysmal events: classification and differential diagnosis",
    "Lynch JW & Pierce KD (2006) Biochem Pharmacol — Pharmacology of inhibitory glycine receptors",
    "Davies CH (2010) Br J Pharmacol — Clonazepam mechanism and its role in startle disease management",
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT SAMPLE GENERATOR  (40 patients, seed=495)
# ─────────────────────────────────────────────────────────────────────────────
def _make_patients():
    cats = [
        ("GLRA1-Arg271-Dominant-Classic", 40, None),
        ("GLRA1-LOF-Recessive-Severe", 15, None),
        ("GLRA1-Other-Dominant-Missense", 25, None),
        ("GLRA1-LOF-Dominant-Haploinsufficiency", 10, None),
        ("Phenocopy-GlyR-Other", 10, None),
    ]
    pts = []
    pid = 1
    for cat, pct, _ in cats:
        n = max(1, round(40 * pct / 100))
        for _ in range(n):
            is_arg271 = cat == "GLRA1-Arg271-Dominant-Classic"
            is_recessive = cat == "GLRA1-LOF-Recessive-Severe"
            is_other_dom = cat == "GLRA1-Other-Dominant-Missense"
            is_haplo = cat == "GLRA1-LOF-Dominant-Haploinsufficiency"
            is_phenocopy = cat == "Phenocopy-GlyR-Other"

            sex = random.choice(["M", "F"])
            onset = round(random.uniform(0.0, 0.05) if is_arg271 or is_recessive else
                          (random.uniform(0.0, 0.1) if is_other_dom else
                           (random.uniform(0.5, 5) if is_haplo else random.uniform(0.0, 0.3))), 2)
            age = round(onset + random.uniform(1, 25), 1)

            # Hyperekplexia severity
            apnoea = random.random() < (0.90 if is_arg271 else 0.80 if is_recessive else
                                        0.50 if is_other_dom else 0.15 if is_haplo else 0.60)
            rigid_baby = random.random() < (0.95 if is_arg271 else 0.90 if is_recessive else
                                            0.60 if is_other_dom else 0.10 if is_haplo else 0.50)
            startle_falls = random.random() < (0.70 if is_arg271 else 0.65 if is_recessive else
                                               0.55 if is_other_dom else 0.35 if is_haplo else 0.55)
            # Epileptic seizures
            epileptic_sz = random.random() < (0.05 if is_arg271 else 0.30 if is_recessive else
                                              0.05 if is_other_dom else 0.15 if is_haplo else 0.20)
            # Treatment
            on_clz = random.random() < (0.95 if is_arg271 or is_recessive or is_other_dom else
                                        0.55 if is_haplo else 0.85)
            on_vpa = random.random() < (0.05 if not is_recessive else 0.30)
            on_piracetam = random.random() < (0.20 if is_arg271 or is_recessive else 0.10)
            manoeuvre_trained = random.random() < (0.98 if is_arg271 or is_recessive else
                                                   0.85 if is_other_dom else 0.60 if is_haplo else 0.80)
            nose_tap_positive = random.random() < (0.98 if is_arg271 else 0.90 if is_recessive else
                                                   0.85 if is_other_dom else 0.50 if is_haplo else 0.80)
            metabolic_screened = random.random() < 0.85
            video_eeg_done = random.random() < 0.78
            polg_tested = random.random() < (0.80 if is_recessive and on_vpa else 0.30)

            pts.append({
                "id": f"GLRA1-{pid:03d}",
                "sex": sex,
                "age": age,
                "onset_age": onset,
                "category": cat,
                "apnoeic_events": apnoea,
                "rigid_baby": rigid_baby,
                "startle_falls": startle_falls,
                "epileptic_seizures": epileptic_sz,
                "on_clonazepam": on_clz,
                "on_vpa": on_vpa,
                "on_piracetam": on_piracetam,
                "forward_flexion_trained": manoeuvre_trained,
                "nose_tap_positive": nose_tap_positive,
                "metabolic_screened": metabolic_screened,
                "video_eeg_done": video_eeg_done,
                "polg_tested": polg_tested,
            })
            pid += 1
    return pts[:40]


PATIENTS = _make_patients()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    apnoea = sum(1 for p in PATIENTS if p["apnoeic_events"])
    rigid = sum(1 for p in PATIENTS if p["rigid_baby"])
    falls = sum(1 for p in PATIENTS if p["startle_falls"])
    epileptic = sum(1 for p in PATIENTS if p["epileptic_seizures"])
    on_clz = sum(1 for p in PATIENTS if p["on_clonazepam"])
    on_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    trained = sum(1 for p in PATIENTS if p["forward_flexion_trained"])
    nose_tap = sum(1 for p in PATIENTS if p["nose_tap_positive"])
    metabolic = sum(1 for p in PATIENTS if p["metabolic_screened"])
    video_eeg = sum(1 for p in PATIENTS if p["video_eeg_done"])

    etio = [
        {"etiology": e["category"], "n": max(1, round(n * e["pct"] / 100)), "pct": e["pct"]}
        for e in ETIOLOGY_CATALOG
    ]

    tx_summary = [
        {"drug": "Clonazepam", "level": "Level A — First-Line All Classes"},
        {"drug": "Forward-Flexion Manoeuvre", "level": "Level A — Acute Apnoea (Vigevano 1989)"},
        {"drug": "Sodium Valproate (POLG mandatory)", "level": "Level B — Recessive-Class Seizures"},
        {"drug": "Piracetam", "level": "Level C — Historical Second-Line Adjunct"},
        {"drug": "Levetiracetam", "level": "Level B — Recessive-Class Epileptic Seizures"},
        {"drug": "CLZ Weaning (Age 3-5y)", "level": "Standard Practice — Natural History"},
    ]

    return {
        "kpis": {
            "n_patients": n,
            "apnoeic_events_pct": round(apnoea / n * 100),
            "rigid_baby_pct": round(rigid / n * 100),
            "startle_falls_pct": round(falls / n * 100),
            "epileptic_seizures_pct": round(epileptic / n * 100),
            "on_clonazepam_pct": round(on_clz / n * 100),
            "on_vpa_pct": round(on_vpa / n * 100),
            "forward_flexion_trained_pct": round(trained / n * 100),
            "nose_tap_positive_pct": round(nose_tap / n * 100),
            "metabolic_screened_pct": round(metabolic / n * 100),
            "video_eeg_done_pct": round(video_eeg / n * 100),
        },
        "etiology_distribution": etio,
        "treatments_summary": tx_summary,
        "monitoring_summary": MONITORING_SCHEDULE[:8],
        "lifecycle": LIFECYCLE_WINDOWS,
        "thresholds": CLINICAL_THRESHOLDS[:8],
        "contraindications_summary": [
            "PHT/CBZ-WRONG-DRUG-for-hyperekplexia",
            "VPA-without-POLG-ABSOLUTE-CI",
            "LTG-ABSOLUTE-CI-if-myoclonic-seizures",
            "Rapid-CLZ-withdrawal-HIGH-CAUTION",
            "Discharge-without-forward-flexion-training-ABSOLUTE-SAFETY-FAILURE",
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
    return {
        "etiology_distribution": etio_detail,
        "patient_sample": PATIENTS[:15],
        "event_detail": EVENT_DETAIL,
        "trigger_detail": TRIGGER_DETAIL,
        "treatment_detail": TREATMENT_DETAIL,
        "contraindications": CONTRAINDICATION_DETAIL,
        "summary": {
            "apnoeic_pct": round(sum(1 for p in PATIENTS if p["apnoeic_events"]) / n * 100),
            "rigid_baby_pct": round(sum(1 for p in PATIENTS if p["rigid_baby"]) / n * 100),
            "epileptic_seizures_pct": round(sum(1 for p in PATIENTS if p["epileptic_seizures"]) / n * 100),
            "forward_flexion_trained_pct": round(sum(1 for p in PATIENTS if p["forward_flexion_trained"]) / n * 100),
            "metabolic_screened_pct": round(sum(1 for p in PATIENTS if p["metabolic_screened"]) / n * 100),
            "video_eeg_done_pct": round(sum(1 for p in PATIENTS if p["video_eeg_done"]) / n * 100),
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
