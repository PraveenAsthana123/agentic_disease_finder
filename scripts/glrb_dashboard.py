"""
GLRB Hyperekplexia — Hyperekplexia Type 1B / Glycine Receptor Beta-1 / 4q32.1
===============================================================================
40-patient cohort · GLRB (4q32.1) · Glycine receptor β1 subunit · AR/AD

GLRB BIOLOGY:
GLRB (4q32.1) encodes the glycine receptor beta-1 subunit (474 aa), the obligate
structural partner of GLRA1 (α1) in forming the adult inhibitory glycine receptor (GlyR)
pentamer. The normal adult GlyR stoichiometry is α1₂β₃ — two GLRA1 subunits + three
GLRB subunits assembled around the Cl⁻ ion pore.

GLRB ACCOUNTS FOR ~5% OF GENETIC HYPEREKPLEXIA — much rarer than GLRA1 (~70%) but
clinically important as it requires the same emergency management (forward-flexion
manoeuvre, clonazepam) and correct diagnostic workup.

GLYCINE RECEPTOR BETA SUBUNIT — STRUCTURE AND ROLE:
  GLRB is a member of the Cys-loop superfamily; its architecture mirrors GLRA1:
    - N-terminal extracellular domain (ECD): Cys-loop (aa1-208); does NOT directly bind
      glycine (binding is on α1 subunits), but β contributes to glycine sensitivity by
      modulating pore opening kinetics through α-β subunit interface contacts.
    - 4 transmembrane helices: TM1 (aa209-229), TM2 (aa230-251) — pore-lining,
      TM3 (aa255-275), TM4 (aa280-304)
    - Intracellular loop (TM3-TM4) aa305-455: GEPHYRIN (GPHN) BINDING DOMAIN —
      this is the principal scaffold-anchoring domain; GLRB binds gephyrin to anchor
      the GlyR pentamer at the postsynaptic density of glycinergic synapses.
    - Short C-terminal cytoplasmic tail (aa456-474)

GLRB LOF CONSEQUENCE — HOMOMERIC α1 FORMATION:
  When GLRB protein is absent or non-functional:
    - α1 subunits form homomeric pentamers (α1₃ or α1₅) instead of α1₂β₃
    - Homomeric GlyR has DIFFERENT pharmacology: lower glycine sensitivity (EC50 ~3× higher),
      altered picrotoxin sensitivity, reduced single-channel conductance
    - Net effect: REDUCED INHIBITORY GLYCINERGIC TRANSMISSION → hyperekplexia
    - Additionally: GEPHYRIN CANNOT ANCHOR homomeric α1 pentamers (gephyrin binds GLRB
      intracellular loop, not GLRA1) → reduced synaptic GlyR clustering → further loss
      of inhibitory tone
    - GLRB LOF = DUAL MECHANISM: (1) homomeric α1 with reduced conductance + (2) loss
      of gephyrin-mediated synaptic anchoring → compound postsynaptic glycinergic deficit

INHERITANCE PATTERNS:
  Autosomal Recessive (AR): biallelic GLRB LOF (most common GLRB mutation class)
    ~60% of GLRB hyperekplexia; Trp170Ser (North African/MENA founder); Gly254Asp,
    splice variants; severe-to-moderate phenotype.
  Autosomal Dominant (AD): dominant-negative missense variants (~25%)
    Met177Arg (ECD), Arg316Gly (TM2-TM3 linker), Tyr228Ser (TM1); the mutant β subunit
    co-assembles with WT α1 and WT β → poisons pentamer folding/gating.
  De Novo Dominant: ~15% of AD cases; not inherited; variable expression.

GENOTYPE-PHENOTYPE CORRELATIONS:
  Trp170Ser (AR): severe neonatal; MENA founder; most common recessive allele worldwide.
  Gly254Asp (AR): TM2 pore; severe; altered channel gating in homomeric α1.
  Met177Arg (AD): dominant-negative; moderate; N-African/European.
  Arg316Gly (AD): TM2-TM3; dominant-negative; moderate-severe; European.
  ECD folding variants (AR): intermediate severity; respond better to clonazepam.
  Gephyrin-interface variants: milder (anchoring but not conductance impaired).

COMPARISON TO GLRA1:
  GLRA1 (~70%) > SLC6A5 (~15%) > GLRB (~5%) — in frequency
  GLRB phenotype is generally MILDER than GLRA1 Arg271 class but can overlap with
  severe GLRA1 recessive class; neonatal apnoea and rigid-baby syndrome present but
  at somewhat lower frequency.

KEY REFERENCES:
  Rees MI et al. (1994) Hum Mol Genet — GLRB mutations and hyperekplexia (first evidence)
  Harvey RJ et al. (2008) Neuron — GLRB companion gene; α1₂β₃ stoichiometry;
    GLRB LOF → homomeric α1 formation mechanism
  Schaefer N et al. (2015) Hum Mutat — GLRB mutation spectrum; phenotype correlations
  Bode A & Lynch JW (2013) J Biol Chem — GLRB ECD variants; gating effects
  Dumoulin A et al. (2009) J Neurosci — gephyrin binding to GLRB intracellular loop;
    synaptic anchoring mechanism
  Thomas RH & Rees MI (2014) Clin Genet — hyperekplexia genetic spectrum; GLRB ~5%
  Lynch JW (2004) Physiol Rev — glycine receptor ion channel physiology
  Vigevano F et al. (1989) Neuropediatrics — forward-flexion manoeuvre
"""
import random

random.seed(497)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GLRB-AR-Biallelic-LOF",
        "pct": 45,
        "etiology": "GLRB biallelic LOF (AR) — Trp170Ser or Gly254Asp or splice; severe-to-moderate hyperekplexia",
        "mechanism": (
            "Biallelic GLRB loss-of-function (homozygous or compound heterozygous frameshift, "
            "nonsense, splice, or missense) → complete absence or non-functional GLRB protein. "
            "Without β subunit: α1 subunits form homomeric α1₃ or α1₅ pentamers — reduced glycine "
            "sensitivity (EC50 ~3× higher than α1₂β₃) and reduced single-channel conductance → "
            "insufficient Cl⁻ influx on glycine binding → brainstem PnC and spinal cord disinhibited. "
            "DUAL DEFICIT: (1) homomeric α1 reduced conductance PLUS (2) gephyrin CANNOT anchor "
            "homomeric α1 pentamers at postsynaptic densities (gephyrin binds GLRB intracellular "
            "loop, not GLRA1) → further reduction in functional synaptic GlyR density. "
            "TRPW170SER (Trp170Ser, c.510G>C): most common worldwide AR allele; North African/"
            "MENA founder; Trp170 is in the ECD and critical for β subunit folding/assembly. "
            "GLY254ASP (Gly254Asp, c.761G>A): TM2 pore residue; disrupts channel gating in "
            "homomeric α1 as well; severe. "
            "Consanguineous families strongly enriched. NEONATAL RIGID-BABY common but slightly "
            "less severe than GLRA1 Arg271 class."
        ),
        "typical_variants": "c.510G>C (p.Trp170Ser) ECD · c.761G>A (p.Gly254Asp) TM2 · c.IVS5+1G>A splice · c.1174C>T (p.Arg392*) nonsense",
        "eeg_signature": "Normal EEG (non-epileptic); startle: EMG artefact only; severe: ± non-specific high-amplitude slow in neonatal period; nose-tap test positive",
        "phenotype": "Severe-to-moderate rigid-baby; apnoeic episodes; exaggerated startle; clonazepam-responsive; consanguinity; MENA enrichment",
        "onset_age_years": 0.0,
        "outcome": "Good with clonazepam; dual mechanism (reduced conductance + gephyrin-loss) may need slightly higher CLZ doses; improves by age 3-5 years",
    },
    {
        "category": "GLRB-AD-Dominant-Negative-Missense",
        "pct": 25,
        "etiology": "GLRB dominant-negative missense (AD) — Met177Arg, Arg316Gly; moderate hyperekplexia",
        "mechanism": (
            "Heterozygous GLRB missense variants with dominant-negative mechanism: the mutant β subunit "
            "co-assembles with WT α1 and WT β subunits into pentamers with impaired gating or stability. "
            "MET177ARG (Met177Arg, c.530T>G): ECD residue critical for α1-β subunit interface; "
            "dominant-negative reduces glycine-activated Cl⁻ conductance ~50% in heterozygotes. "
            "ARG316GLY (Arg316Gly, c.946C>G): TM2-TM3 linker; affects channel-gating kinetics; "
            "moderate-severe; de novo in European probands. "
            "TYR228SER (Tyr228Ser, c.683A>C): TM1; impairs TM packing; co-assembles poorly. "
            "PHENOTYPE: similar to GLRA1 non-Arg271 dominant class — neonatal rigid-baby (moderate), "
            "exaggerated startle, partial apnoea (less than GLRA1 Arg271 class). "
            "VARIABLE EXPRESSIVITY within families (same GLRB AD allele → wide clinical spectrum). "
            "TREATMENT: clonazepam first-line; doses typically moderate (less than AR class). "
            "Inheritance: AD; ~20% de novo; penetrance near-complete for Met177Arg."
        ),
        "typical_variants": "c.530T>G (p.Met177Arg) ECD α1-β interface · c.946C>G (p.Arg316Gly) TM2-TM3 · c.683A>C (p.Tyr228Ser) TM1 · c.1024G>A (p.Gly342Ser) TM3",
        "eeg_signature": "Normal EEG; moderate startle (EMG artefact); reduced habituation; nose-tap test positive (moderate response)",
        "phenotype": "Moderate hyperekplexia; neonatal rigid-baby; partial apnoea risk; variable expressivity; AD family history in 80%",
        "onset_age_years": 0.0,
        "outcome": "Good; clonazepam moderate dose; 75-85% significant event reduction; most improve by early childhood; normal development in majority",
    },
    {
        "category": "GLRB-ECD-Folding-Intermediate",
        "pct": 15,
        "etiology": "GLRB ECD folding/stability variants (AR) — partial GLRB expression; intermediate phenotype",
        "mechanism": (
            "Biallelic GLRB missense variants located in the ECD that do not abolish protein "
            "expression but impair β subunit folding, stability, or α1-β interface formation. "
            "Partial GLRB function (10-40% of WT activity) → partial α1₂β₃ pentamer formation "
            "alongside increased homomeric α1 formation → intermediate reduction in glycinergic "
            "inhibition → INTERMEDIATE hyperekplexia phenotype. "
            "KEY VARIANTS: Gln242Arg (c.725A>G), Leu275Arg (c.824T>G) — ECD fold-stability; "
            "retain some β expression but α1-β interface impaired; South Asian enrichment. "
            "GEPHYRIN ANCHORING: partially preserved when some β protein present (gephyrin binds "
            "GLRB intracellular loop aa305-455 directly) → synaptic GlyR loss less severe than "
            "biallelic null. "
            "PHENOTYPE: Neonatal hyperekplexia (moderate), lower apnoea risk, may partially "
            "improve without clonazepam (if residual β function is enough), but still requires "
            "CLZ for most. Mild rigid-baby in neonatal period."
        ),
        "typical_variants": "c.725A>G (p.Gln242Arg) ECD · c.824T>G (p.Leu275Arg) ECD · c.500G>A (p.Cys167Tyr) Cys-loop · c.IVS3+5G>A weak splice",
        "eeg_signature": "Normal EEG; mild-to-moderate startle response; nose-tap test: positive but weakly; habituation partially preserved",
        "phenotype": "Intermediate hyperekplexia; moderate neonatal rigid-baby; low apnoea risk; South Asian enrichment; some cases diagnosed in childhood",
        "onset_age_years": 0.0,
        "outcome": "Good; clonazepam low-to-moderate dose; excellent response; most symptom-free by early childhood; normal development",
    },
    {
        "category": "GLRB-Gephyrin-Interface-Anchor-Defect",
        "pct": 10,
        "etiology": "GLRB intracellular loop / gephyrin-interface variants — GlyR synaptic anchoring defect; mild hyperekplexia ± ID",
        "mechanism": (
            "Biallelic or de-novo GLRB missense variants in the intracellular loop (TM3-TM4, "
            "aa305-455), specifically at the GEPHYRIN (GPHN) BINDING DOMAIN — reduce or abolish "
            "gephyrin-mediated anchoring of GlyR pentamers at the postsynaptic density. "
            "CONSEQUENCE: α1₂β₃ pentamers assemble normally (GLRB protein present, ECD intact) "
            "but FAIL TO CLUSTER at glycinergic synapses → diffuse, non-synaptic GlyR → reduced "
            "postsynaptic glycinergic inhibition despite normal pentamer assembly and conductance. "
            "GEPHYRIN BINDING: key residues at GLRB aa381-420 (GPHN binding motif); variants here "
            "selectively disrupt anchoring without affecting channel gating. "
            "PHENOTYPE: Milder hyperekplexia (startle, mild rigid-baby, rare apnoea); SOME PATIENTS "
            "HAVE INTELLECTUAL DISABILITY / autism-like features (gephyrin anchors BOTH GlyR and "
            "GABA-A receptors at inhibitory synapses — GLRB gephyrin-interface variants may "
            "partially disrupt GABAergic synaptogenesis in addition to glycinergic). "
            "IMPORTANT DDx: GPHN/Gephyrin direct mutations cause more severe combined "
            "GlyR+GABAR anchoring loss → more severe ID; GLRB-gephyrin-interface = partial."
        ),
        "typical_variants": "c.1141A>T (p.Arg381Trp) GPHN-motif · c.1187G>A (p.Gly396Asp) GPHN-motif · c.1253C>A (p.Pro418Gln) GPHN-motif · de-novo het",
        "eeg_signature": "Normal or mild slow background; startle: mild EMG artefact; nose-tap test: mildly positive; if ID: may have non-specific epileptiform features",
        "phenotype": "Mild hyperekplexia; mild rigid-baby; rare apnoea; ± mild ID/autism spectrum; GlyR synaptic clustering defect despite normal conductance",
        "onset_age_years": 0.0,
        "outcome": "Good hyperekplexia prognosis; clonazepam low dose; ID component (if present) persists and requires separate support; normal life expectancy",
    },
    {
        "category": "Phenocopy-GLRB-Negative",
        "pct": 5,
        "etiology": "Phenocopy — GLRB-sequencing negative; alternative glycinergic or startle aetiology",
        "mechanism": (
            "Clinically GLRB-like hyperekplexia (startle, mild rigid-baby, moderate apnoea) but "
            "GLRB sequencing and MLPA NEGATIVE. Confirmed alternative diagnoses include: "
            "SLC6A5 / GlyT2 (OMIM 604159, 11p15.1): glycine transporter-2; AR; the MOST COMMON "
            "  recessive non-GLRA1 cause of hyperekplexia; reduces presynaptic glycine reuptake "
            "  → paradoxically depleted presynaptic stores → impaired vesicular glycine release. "
            "GPHN / Gephyrin (OMIM 603930, 14q23.3): AR/de novo dominant; reduced postsynaptic "
            "  GlyR clustering + GABA-A receptor anchoring → combined inhibitory defect → "
            "  hyperekplexia + epilepsy + ID (more severe than GLRB gephyrin-interface). "
            "ARHGEF9 / Collybistin (OMIM 300429, Xq11.1): X-linked recessive; RhoGEF that "
            "  activates neuroligin-2 complex to cluster gephyrin + GlyR; hemizygous males: "
            "  hyperekplexia + epilepsy + severe ID; female carriers: variable. "
            "GLRA1 negative sequencing result in referral for GLRB → always send FULL panel: "
            "  GLRB + SLC6A5 + GPHN + ARHGEF9 simultaneously."
        ),
        "typical_variants": "SLC6A5 c.1219C>T (p.Gln407*) AR · GPHN c.506delC frameshift · ARHGEF9 hemizygous Arg290His · Non-genetic excluded",
        "eeg_signature": "Variable: SLC6A5 similar to GLRB; GPHN/ARHGEF9 males: epileptiform; EEG essential to classify",
        "phenotype": "GLRB-like clinically; alternative confirmed: SLC6A5/GPHN/ARHGEF9; GPHN/ARHGEF9 add epilepsy + ID",
        "onset_age_years": 0.0,
        "outcome": "Gene-specific; SLC6A5 responds to clonazepam; GPHN/ARHGEF9 complex (epilepsy + ID managed separately)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HYPEREKPLEXIA EVENT TYPES
# ─────────────────────────────────────────────────────────────────────────────
EVENT_DETAIL = [
    {
        "type": "Exaggerated Startle Response (Generalised Tonic Stiffening)",
        "prevalence_pct": 100,
        "semiology": (
            "Triggered by sudden auditory, tactile, or visual stimulus → generalised tonic "
            "stiffening (bilateral, non-clonic, no post-ictal state). "
            "In GLRB hyperekplexia the response is generally LESS SEVERE than GLRA1 Arg271 class "
            "— shorter duration of stiffening, less forceful, more rapid recovery. "
            "NON-HABITUATING startle: pathognomonic; normal startle habituates after 3-5 repetitions, "
            "hyperekplexia startle does not. The nose-tap test exploits this non-habituation. "
            "FALLS WITHOUT LOC: ambulatory patients fall stiffly — injury risk; same management "
            "as GLRA1 (safety environment, clonazepam optimisation, helmet if severe)."
        ),
        "eeg_pattern": "NORMAL EEG — no ictal correlate during generalised stiffening. Only surface EMG artefact seen. ESSENTIAL FINDING: confirms non-epileptic aetiology. Nose-tap test positive (repetitive, non-habituating flexion response ≥3 repetitions).",
        "clinical_tip": (
            "Document all startle events as 'non-epileptic hyperekplexia events' to prevent "
            "inappropriate AED escalation. Nose-tap test at every visit quantifies severity and "
            "monitors clonazepam response. GLRB patients typically have milder nose-tap response "
            "than GLRA1 Arg271 — set expectations appropriately."
        ),
    },
    {
        "type": "Neonatal Apnoeic Hyperekplexia",
        "prevalence_pct": 55,
        "semiology": (
            "Neonatal apnoea triggered by handling or sudden stimuli → tonic stiffening → "
            "respiratory muscle rigidity → apnoea → cyanosis → hypoxia. "
            "In GLRB hyperekplexia: less frequent and less severe than GLRA1 Arg271 class "
            "(AR biallelic GLRB ~55% apnoea risk vs GLRA1 Arg271 ~90%). "
            "DUAL MECHANISM (GLRB-specific): loss of synaptic GlyR clustering (gephyrin) "
            "in addition to reduced conductance → apnoea may be more stimulus-specific and "
            "less continuous than GLRA1 biallelic null class. "
            "MANAGEMENT: identical to GLRA1 — forward-flexion manoeuvre immediately; "
            "clonazepam; NICU monitoring; carer training BEFORE discharge."
        ),
        "eeg_pattern": "EEG: NON-ICTAL during tonic phase — EMG artefact only; no epileptiform discharge. EEG monitoring during apnoeic episode mandatory to confirm non-epileptic aetiology.",
        "clinical_tip": (
            "FORWARD-FLEXION MANOEUVRE (Vigevano manoeuvre): immediate life-saving first-aid. "
            "Technique: flex head toward chest + knees toward abdomen → releases tonic rigidity "
            "within seconds → restores breathing. ALL carers trained before NICU discharge. "
            "Clonazepam: 0.01-0.03 mg/kg/day PO/NG in neonates; SpO₂ monitoring mandatory. "
            "GLRB apnoea is generally self-limiting earlier than GLRA1 Arg271 class."
        ),
    },
    {
        "type": "Neonatal Rigid-Baby Syndrome",
        "prevalence_pct": 68,
        "semiology": (
            "Generalised hypertonia at rest in neonates with GLRB LOF — rigid-baby presentation. "
            "GENERALLY MILDER THAN GLRA1 Arg271 class: the dual-mechanism deficit (reduced "
            "conductance + gephyrin-loss) still impairs glycinergic tone sufficiently for "
            "rigid-baby, but the severity of continuous hypertonia is typically lower. "
            "Spontaneous recovery by age 2-4 months in mild cases on clonazepam. "
            "METABOLIC DIFFERENTIAL: plasma amino acids mandatory (exclude NKH — glycine "
            "cleavage system defect; plasma + CSF glycine NORMAL in GLRB hyperekplexia). "
            "GLRB biallelic null (AR) class: near-identical phenotype to GLRA1 LOF recessive "
            "class; diagnosis requires gene panel (cannot distinguish on clinical grounds alone)."
        ),
        "eeg_pattern": "Normal EEG at rest; no interictal epileptiform discharges in isolated GLRB rigid-baby (unlike NKH which has burst-suppression). Plasma glycine NORMAL (key DDx from NKH).",
        "clinical_tip": (
            "METABOLIC SCREEN mandatory at presentation: plasma amino acids (exclude NKH), "
            "urine organic acids, biotinidase activity. "
            "CSF glycine if plasma glycine elevated or EEG burst-suppression. "
            "GLRB diagnosis requires gene panel: GLRB + GLRA1 + SLC6A5 + GPHN + ARHGEF9. "
            "Gene panel mandatory — GLRB alone cannot be excluded on clinical grounds."
        ),
    },
    {
        "type": "Childhood/Adult Startle Falls (Drop Attacks)",
        "prevalence_pct": 48,
        "semiology": (
            "Older GLRB patients: sudden startle → whole-body stiffening → falls stiffly. "
            "LOWER FREQUENCY THAN GLRA1 in same age groups: homomeric α1 pentamers retain "
            "more glycinergic function than total absence (GLRA1 null); falls less frequent. "
            "PHOBIC AVOIDANCE: patients avoid loud public spaces; significant psychosocial burden. "
            "INJURY RISK: head trauma, fractures — helmet and safety environment for severe cases. "
            "MISDIAGNOSIS: often classified as epileptic drop attacks before EEG confirms "
            "non-ictal nature. Video-EEG during event is diagnostic."
        ),
        "eeg_pattern": "Normal EEG during startle fall; non-ictal; only EMG artefact. Ambulatory video-EEG + accelerometry useful for capturing falls outside clinic.",
        "clinical_tip": (
            "Clonazepam optimisation first-line for falls. Safety environment: remove sharp "
            "furniture, padded floor. Driving assessment mandatory for adults — startle while "
            "driving = road risk. Medical alert bracelet. If CLZ inadequate: piracetam adjunct."
        ),
    },
    {
        "type": "Gephyrin-Anchoring Class — Mild Hyperekplexia ± ID",
        "prevalence_pct": 10,
        "semiology": (
            "Specific to GLRB intracellular-loop / gephyrin-interface variants: mild hyperekplexia "
            "(exaggerated startle, rarely apnoea) PLUS intellectual disability or autism-like "
            "features in some patients. GlyR pentamers form normally (normal conductance) but "
            "fail to cluster at glycinergic synapses → reduced postsynaptic GlyR density → "
            "reduced startle inhibition. "
            "ID component: gephyrin also anchors GABA-A receptors; partial gephyrin-interaction "
            "loss at GLRB-interface may perturb GABAergic synaptogenesis in addition to GlyR "
            "clustering → combined inhibitory synapse defect → cognitive phenotype in some. "
            "Phenotypically overlaps with GPHN/gephyrin direct mutations (GPHN is more severe)."
        ),
        "eeg_pattern": "Normal EEG for hyperekplexia events; if ID component: non-specific slow background; occasional non-epileptic slow waves; no typical epileptiform discharges unless GPHN (more severe).",
        "clinical_tip": (
            "DISTINGUISH from GPHN direct mutations (gephyrin gene, 14q23.3): GPHN = more severe "
            "combined GlyR + GABA-A anchoring loss → severe ID + epilepsy + hyperekplexia. "
            "GLRB gephyrin-interface = milder partial defect → mild hyperekplexia ± mild ID. "
            "Neuropsychological assessment at age 18-24 months; early intervention if ID confirmed."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_DETAIL = [
    {"trigger": "Sudden Auditory Stimulus (loud noise, clap, doorbell)",
     "pct": 96, "note": "Primary trigger; same acoustic startle circuit as GLRA1. Soft doorbells, advance warning before infant handling. GLRB startle is slightly less sustained than GLRA1 Arg271 — shorter tonic phase."},
    {"trigger": "Sudden Tactile Stimulus (unexpected touch, handling, nappy change)",
     "pct": 88, "note": "Critical in neonates with GLRB AR biallelic class. Forward-flexion manoeuvre training mandatory before any neonatal discharge. Gentle advance-warning handling protocol reduces apnoea frequency."},
    {"trigger": "Sleep/Wake Transition",
     "pct": 60, "note": "Hypnic context; cortical inhibition reduced at transitions. Monitor for nocturnal hyperekplexia. Clonazepam sedative effect provides partial night coverage."},
    {"trigger": "Emotional Arousal (excitement, fright, anxiety)",
     "pct": 50, "note": "Heightened arousal lowers startle threshold. Cognitive-behavioural relaxation reduces anticipatory anxiety in older patients with GLRB."},
    {"trigger": "Visual Flash / Photic Stimulus",
     "pct": 28, "note": "Visual pathway startle; avoid stroboscopic environments; distinguish from photoparoxysmal response (requires EEG). GLRB generally lower rate than GLRA1."},
    {"trigger": "Fever / Intercurrent Illness",
     "pct": 25, "note": "Thermal sensitivity of homomeric α1 GlyR (GLRB LOF) may be higher than α1₂β₃ — elevated temperature shifts glycine EC50 further upward. Pre-emptive CLZ increase plan during febrile illness."},
    {"trigger": "Stress / Sleep Deprivation",
     "pct": 18, "note": "Global excitability increase; sleep hygiene important. Particularly affects adolescents with phobic avoidance. Residual events often stress-triggered in adulthood."},
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
            "spinal cord → reduces startle circuit hyperexcitability. Same mechanism as for GLRA1 "
            "hyperekplexia: GABAergic compensation, not direct GlyR action."
        ),
        "dose": "Neonate: 0.01-0.03 mg/kg/day PO/NG; infant: 0.02-0.05 mg/kg/day; child: 0.05-0.1 mg/kg/day; lower doses often sufficient vs GLRA1 Arg271 class",
        "efficacy": "Level A: international standard first-line; 80-90% significant event reduction; GLRB patients often respond at lower doses than GLRA1 Arg271 class (milder phenotype)",
        "safety": "Sedation (dose-limiting); respiratory depression (neonates — SpO₂ monitoring); tolerance; paradoxical agitation in infants",
        "monitoring": "SpO₂ monitoring on CLZ initiation (neonates); sedation scale; respiratory rate; developmental milestones",
        "glrb_note": "GLRB-SPECIFIC: GLRB AR biallelic and AD dominant-negative classes generally require similar doses to GLRA1 non-Arg271 class (not Arg271 class levels). Gephyrin-interface class: often respond to very low doses (anchoring defect without severe conductance loss). Attempt weaning at age 3-4 years — GLRB natural history improvement may be faster than GLRA1.",
    },
    {
        "drug": "Forward-Flexion Manoeuvre (Vigevano Manoeuvre)",
        "level": "Level A — Acute Apnoea (Emergency First Aid)",
        "moa": (
            "Physical manoeuvre that terminates apnoeic hyperekplexia by flexing the patient into "
            "the foetal position (head toward chest, knees toward abdomen). Activates proprioceptive "
            "feedback → inhibits brainstem startle circuit tonic discharge → restores spontaneous "
            "breathing within seconds. Life-saving. Vigevano et al. (1989)."
        ),
        "dose": "Flex head toward chest while flexing knees toward abdomen; hold 3-5 seconds; release; repeat if needed",
        "efficacy": "Level A: immediate life-saving; terminates apnoea in seconds; ALL carers must demonstrate competency before discharge",
        "safety": "Safe when performed correctly; gentle only; avoid force near cervical spine instability",
        "monitoring": "Documented carer training in medical record; re-assess any new carer; repeat demonstration at every visit",
        "glrb_note": "GLRB APNOEA NOTE: GLRB apnoea is present in the AR biallelic class (~55%) and AD class (~30%). Even though less frequent than GLRA1 Arg271 class, the manoeuvre is EQUALLY mandatory — apnoea outside NICU without trained carer is life-threatening regardless of severity. Never discharge GLRB neonates without documented forward-flexion competency.",
    },
    {
        "drug": "Piracetam",
        "level": "Level C — Second-Line Adjunct",
        "moa": "Modulation of AMPA receptor function; reduces startle-evoked EMG amplitude in some patients; used historically as adjunct when clonazepam inadequate or poorly tolerated",
        "dose": "Adult/adolescent: 2.4-4.8 g/day in divided doses; paediatric: 40-100 mg/kg/day",
        "efficacy": "Level C (limited evidence); modestly reduces event frequency; useful as add-on when CLZ causes unacceptable sedation",
        "safety": "Generally well-tolerated; occasional agitation; rare bleeding tendency; avoid in renal impairment",
        "monitoring": "Renal function; platelet count if surgical procedures planned",
        "glrb_note": "GLRB: piracetam adjunct when CLZ sedation is limiting, particularly in GLRB gephyrin-interface class (milder phenotype, CLZ often already at low dose). Evidence base is entirely from case series; no GLRB-specific trial data.",
    },
    {
        "drug": "Glycine Supplementation (Experimental)",
        "level": "Level C — Investigational Only",
        "moa": (
            "GLRB-SPECIFIC CONSIDERATION (not applicable to GLRA1): In GLRB LOF, homomeric α1 "
            "pentamers form with EC50 ~3× higher than α1₂β₃. Supplemental glycine (high-dose oral) "
            "has been explored to overcome the reduced agonist sensitivity of homomeric α1 — by "
            "flooding synaptic cleft with glycine, the lower-affinity receptor can be activated. "
            "THEORETICAL RATIONALE ONLY: no clinical trial data; glycine crosses BBB poorly; "
            "systemic hyperglycinaemia risks NMDA receptor over-activation (excitotoxicity). "
            "NEVER use without neuromuscular disease specialist and formal glycine monitoring. "
            "DO NOT use as alternative to clonazepam — not clinically validated."
        ),
        "dose": "Investigational: 0.3-0.6 g/kg/day PO in divided doses; plasma glycine monitoring mandatory (target: modest elevation; not > 2× ULN)",
        "efficacy": "Level C (theoretical); anecdotal reports only; not validated in clinical series",
        "safety": "Risk: systemic hyperglycinaemia → NMDA co-agonist at cortical receptors → excitotoxicity risk; nausea; NOT STANDARD OF CARE",
        "monitoring": "Plasma glycine levels; weekly initially; neurotoxicity monitoring; immediately discontinue if no benefit within 4-8 weeks",
        "glrb_note": "GLRB-SPECIFIC only — NOT applicable to GLRA1, SLC6A5, GPHN hyperekplexia. Do not use without specialist input. Clonazepam remains the only validated pharmacotherapy.",
    },
    {
        "drug": "Clonazepam Weaning (Age 3-5 years)",
        "level": "Standard Practice — Natural History Management",
        "moa": "Natural neurological maturation: secondary GABAergic upregulation compensates for chronic GlyR deficit; cortical inhibitory maturation reduces brainstem sensitivity. GLRB: homomeric α1 pentamers may partially normalise (developmental subunit switching) as brain matures.",
        "dose": "Attempt gradual dose reduction at age 3-4 years; reduce by 10-20% every 4-8 weeks; stop if symptom-free for >12 months",
        "efficacy": "~75-85% of GLRB patients reduce/discontinue CLZ by late childhood (may be slightly faster than GLRA1 Arg271 class due to milder phenotype)",
        "safety": "Withdrawal seizures if tapered too rapidly; rebound hyperekplexia; slow taper over weeks-months",
        "monitoring": "Nose-tap test at each visit; parent-reported event diary; staged taper with rescue CLZ available",
        "glrb_note": "GLRB natural history: GLRB dominant-negative and ECD-folding classes typically show earlier improvement than AR biallelic null class. Gephyrin-interface class: CLZ often discontinued by age 2-3 years. AR biallelic: similar to GLRA1 non-Arg271 class timeline.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS / HIGH CAUTIONS
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATION_DETAIL = [
    {
        "drug": "Phenytoin (PHT) / Carbamazepine (CBZ) for hyperekplexia",
        "risk": "INEFFECTIVE — WRONG DRUG",
        "reason": "Na⁺ channel blockers have NO efficacy for glycine receptor–mediated hyperekplexia (GLRB, GLRA1, or SLC6A5). Never administer for apnoeic hyperekplexia or startle events. Reserve only for confirmed co-existing epileptic seizures (EEG-confirmed ictal events).",
    },
    {
        "drug": "VPA — without prior POLG screening (if co-existing epilepsy)",
        "risk": "ABSOLUTE CI — Without POLG Screen",
        "reason": "If GLRB hyperekplexia patient develops co-existing epileptic seizures requiring VPA: POLG1 sequencing MANDATORY first. Any severe infantile encephalopathy may harbour POLG mutation — VPA in POLG = Alpers-Huttenlocher fatal hepatic failure.",
    },
    {
        "drug": "High-dose Glycine Supplementation (without monitoring)",
        "risk": "HIGH CAUTION — Experimental Only",
        "reason": "Systemic hyperglycinaemia risks NMDA receptor over-activation (glycine is a co-agonist at NMDAr). Not validated in clinical series. Never use without specialist input and plasma glycine monitoring. Clonazepam is the validated treatment.",
    },
    {
        "drug": "Neonatal Discharge without Forward-Flexion Training",
        "risk": "ABSOLUTE SAFETY REQUIREMENT",
        "reason": "Even though GLRB apnoea risk is lower than GLRA1 Arg271, it is still present in the AR biallelic class (~55%) and AD class (~30%). Discharging without documented forward-flexion manoeuvre competency in all primary carers is a patient safety failure.",
    },
    {
        "drug": "Single-gene sequencing (GLRB only) for panel presentation",
        "risk": "DIAGNOSTIC PITFALL — Panel Required",
        "reason": "Clinical hyperekplexia with GLRB-like presentation: always send the full hyperekplexia gene panel (GLRB + GLRA1 + SLC6A5 + GPHN + ARHGEF9) simultaneously. Phenotypic overlap is high; sequential single-gene testing delays diagnosis by months.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_THRESHOLDS = [
    {"label": "CLZ Starting Dose (Neonate)", "value": "0.01–0.03", "unit": "mg/kg/day"},
    {"label": "CLZ Maintenance Dose (Child)", "value": "0.02–0.08", "unit": "mg/kg/day (lower than GLRA1 Arg271)"},
    {"label": "Forward-Flexion Response Time", "value": "<5", "unit": "seconds (apnoea terminates)"},
    {"label": "Nose-Tap Test Threshold (Positive)", "value": "≥3", "unit": "non-habituating repetitions"},
    {"label": "Weaning Attempt Age", "value": "3–4", "unit": "years (may be earlier than GLRA1)"},
    {"label": "Plasma Glycine (NKH screen)", "value": ">0.08", "unit": "CSF:plasma ratio = NKH (GLRB: NORMAL glycine)"},
    {"label": "GLRB Frequency of Hyperekplexia", "value": "~5", "unit": "% of all genetic hyperekplexia"},
    {"label": "Homomeric α1 EC50 vs α1₂β₃", "value": "~3×", "unit": "higher (reduced glycine sensitivity)"},
    {"label": "Apnoea Risk AR Biallelic", "value": "~55", "unit": "% (vs 90% GLRA1 Arg271)"},
    {"label": "Rigid-Baby Risk AR Biallelic", "value": "~68", "unit": "% (vs 95% GLRA1 Arg271)"},
    {"label": "CLZ Taper Rate", "value": "10–20%", "unit": "per 4-8 weeks (slow taper)"},
    {"label": "GLRB Gene Panel Requirement", "value": "5-gene panel", "unit": "GLRB+GLRA1+SLC6A5+GPHN+ARHGEF9"},
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_SCHEDULE = [
    {"item": "SpO₂ / Apnoea monitor (neonates on CLZ initiation)", "frequency": "Continuous; until stable on CLZ and apnoea-free ≥48h"},
    {"item": "Video-EEG during event", "frequency": "Once (confirm non-epileptic); repeat if seizures suspected or ID co-occurs"},
    {"item": "Nose-tap test (hyperekplexia severity gauge)", "frequency": "Every clinic visit (document non-habituation and event frequency)"},
    {"item": "CLZ dose review", "frequency": "Every 3 months (titrate to effect; monitor sedation)"},
    {"item": "Metabolic screen (plasma AA, urine OA, biotinidase)", "frequency": "Once at diagnosis (exclude NKH, MMA, biotinidase deficiency; GLRB glycine NORMAL)"},
    {"item": "Hyperekplexia gene panel (GLRB+GLRA1+SLC6A5+GPHN+ARHGEF9)", "frequency": "Once at diagnosis; full panel simultaneously"},
    {"item": "Developmental assessment (Griffiths/Bayley)", "frequency": "Every 6 months (0-3 years); annually (3-6 years); especially gephyrin-interface class"},
    {"item": "Forward-flexion manoeuvre carer competency", "frequency": "Confirmed before discharge; re-assessed annually; any new carer"},
    {"item": "POLG1 sequencing (before VPA if co-existing epilepsy)", "frequency": "Once, before first VPA prescription"},
    {"item": "Gephyrin-interface class: neuropsychological assessment", "frequency": "At age 18-24 months; annually (3-6 years); early intervention if ID"},
    {"item": "CLZ weaning trial", "frequency": "Consider at age 3-4 years (GLRB); gradual taper; monitor event diary"},
    {"item": "Event diary (hyperekplexia events per day/week)", "frequency": "Ongoing; reviewed at each clinic visit"},
    {"item": "Genetic cascade testing (family members)", "frequency": "AR: carrier testing for parents and siblings; AD: 50% risk to offspring"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {"window": "Neonatal (0–4 weeks)", "headline": "Rigid-baby (moderate) + apnoeic hyperekplexia (AR biallelic ~55%); NICU monitoring; CLZ initiation; forward-flexion carer training; metabolic screen; full hyperekplexia gene panel"},
    {"window": "Infantile (1–12 months)", "headline": "CLZ dose titration; apnoea resolution; development monitoring; video-EEG event characterisation; gephyrin-interface class: watch for developmental delay"},
    {"window": "Early Childhood (1–5 years)", "headline": "Natural GlyR maturation; CLZ weaning earlier than GLRA1 Arg271 (3-4y); developmental follow-up; school safety plan for startle falls; gephyrin-interface: neuropsychological assessment"},
    {"window": "Childhood (5–12 years)", "headline": "CLZ reduction/cessation in most; safety environment at school; phobic avoidance counselling; GLRB AD class: variable expressivity monitoring in affected family members"},
    {"window": "Adolescence (12–18 years)", "headline": "Peer awareness; driving assessment mandatory (adults); piracetam adjunct if residual; psychosocial support; genetic counselling for family planning (AD: 50% risk to offspring)"},
    {"window": "Adulthood", "headline": "Most asymptomatic or mild residual startle; AD GLRB: 50% risk to offspring; prenatal/preimplantation genetics available; rare CLZ restart in stressful periods"},
]

# ─────────────────────────────────────────────────────────────────────────────
# CORE CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────
CORE_CONCEPTS = [
    {"term": "GLRB", "definition": "Gene encoding the glycine receptor beta-1 subunit (474 aa, 4q32.1); the obligate structural partner of GLRA1 (α1) in forming the adult inhibitory GlyR pentamer (α1₂β₃). GLRB accounts for ~5% of genetic hyperekplexia. OMIM Gene 138492."},
    {"term": "Adult GlyR Stoichiometry (α1₂β₃)", "definition": "The normal adult inhibitory GlyR is a pentamer of 2 GLRA1 (α1) + 3 GLRB (β) subunits. GLRB LOF → α1 forms homomeric pentamers (α1₃/α1₅) with ~3× higher EC50 and reduced Cl⁻ conductance → insufficient glycinergic inhibition."},
    {"term": "Homomeric α1 Formation (GLRB LOF Mechanism)", "definition": "When GLRB is absent: GLRA1 subunits self-assemble as homomeric α1₃ or α1₅ pentamers. These have lower glycine sensitivity (EC50 ~3× higher than α1₂β₃) and reduced single-channel conductance → net reduction in inhibitory Cl⁻ flux → hyperekplexia."},
    {"term": "GLRB Gephyrin Binding Domain", "definition": "GLRB intracellular loop (TM3-TM4, aa305-455) contains the GEPHYRIN (GPHN) binding motif (~aa381-420). Gephyrin anchors GlyR at postsynaptic densities; GLRB gephyrin-interface variants → reduced synaptic GlyR clustering despite normal conductance → hyperekplexia ± ID."},
    {"term": "Dual Mechanism of GLRB LOF", "definition": "GLRB loss causes: (1) homomeric α1 formation with reduced glycine sensitivity AND (2) loss of gephyrin-mediated synaptic anchoring (gephyrin binds GLRB, not GLRA1) → compound postsynaptic glycinergic deficit. Partly explains why GLRB null phenotype can overlap GLRA1 severity."},
    {"term": "Trp170Ser (Most Common GLRB AR Variant)", "definition": "c.510G>C (p.Trp170Ser): MENA/North African founder allele, most common AR GLRB variant worldwide; Trp170 is in the ECD and critical for β subunit folding and α1-β subunit interface assembly."},
    {"term": "Non-Habituating Startle", "definition": "Pathognomonic feature of hyperekplexia (GLRB and GLRA1): normal startle habituates after 3-5 repetitions; hyperekplexia startle does NOT habituate. Basis of the nose-tap test. GLRB patients often show milder non-habituating response than GLRA1 Arg271 class."},
    {"term": "Forward-Flexion (Vigevano) Manoeuvre", "definition": "LIFE-SAVING emergency treatment for apnoeic hyperekplexia: flex head toward chest + knees toward abdomen → terminates tonic rigidity in seconds → restores breathing. MANDATORY competency-based training for all carers before discharge. Applies equally to GLRB and GLRA1 patients."},
    {"term": "Nose-Tap Test (Iles-Vigevano Test)", "definition": "Tap tip of nose sharply: POSITIVE = repetitive non-habituating flexion response ≥3 repetitions. Pathognomonic for hyperekplexia. Monitor at every clinic visit. GLRB patients typically have milder response than GLRA1 Arg271 — calibrate threshold accordingly."},
    {"term": "Hyperekplexia Gene Panel Requirement", "definition": "GLRB hyperekplexia is clinically indistinguishable from GLRA1, SLC6A5 (GlyT2), GPHN, and ARHGEF9 hyperekplexia. A 5-gene panel (GLRB+GLRA1+SLC6A5+GPHN+ARHGEF9) sent simultaneously is the diagnostic standard — sequential single-gene testing wastes months."},
    {"term": "NKH Differential Diagnosis", "definition": "Non-ketotic hyperglycinaemia (glycine cleavage system defect) causes rigid-baby + burst-suppression EEG + elevated CSF:plasma glycine ratio >0.08. GLRB hyperekplexia: plasma glycine NORMAL, EEG normal. Plasma amino acids mandatory at presentation to exclude NKH."},
    {"term": "GLRB vs GLRA1 Severity Comparison", "definition": "GLRA1 is more severe on average (especially Arg271 dominant-negative). GLRB AR biallelic: similar to GLRA1 LOF recessive. GLRB AD dominant-negative: similar to GLRA1 non-Arg271 class. GLRB gephyrin-interface: mildest class. GLRB phenotype overlaps broadly — gene panel distinguishes."},
    {"term": "Gephyrin (GPHN) as Direct Mutation — DDx", "definition": "GPHN/gephyrin direct mutations (14q23.3) cause more severe disease than GLRB gephyrin-interface: GPHN LOF abolishes ALL gephyrin function → combined GlyR + GABA-A anchoring loss → severe hyperekplexia + epilepsy + severe ID. GLRB gephyrin-interface = partial (only GlyR anchoring at GLRB-gephyrin interface impaired)."},
    {"term": "Clonazepam Mechanism in GLRB Hyperekplexia", "definition": "CLZ is NOT a GlyR drug — acts via GABA-A receptor potentiation → GABAergic compensation for reduced glycinergic inhibition at brainstem PnC and spinal cord. Same mechanism as GLRA1 hyperekplexia. GLRB patients often need lower doses than GLRA1 Arg271 due to milder baseline deficit."},
    {"term": "Driver Safety Assessment", "definition": "Hyperekplexia startle while driving = sudden loss of vehicle control risk. All adult GLRB patients must have formal driving assessment by neurologist + notification to relevant licensing authority. Uncontrolled events = driving prohibition."},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_STANDARDS = [
    "ILAE 2017 Classification — Hyperekplexia: non-epileptic paroxysmal event; differential diagnosis of epileptic tonic seizures; video-EEG mandatory to classify each event type",
    "Hyperekplexia Gene Panel — 5-gene panel simultaneously: GLRB + GLRA1 + SLC6A5 + GPHN + ARHGEF9; never sequential single-gene testing; phenotypic overlap is high",
    "Forward-Flexion Manoeuvre — Level A first-line for acute neonatal apnoea; competency-based carer training documented before hospital discharge",
    "Clonazepam — Level A first-line pharmacological treatment; GLRB: dose titrated to effect (typically lower than GLRA1 Arg271 class); SpO₂ monitoring in neonates",
    "Nose-Tap Test — performed at every clinic visit; positive = non-habituating response ≥3 repetitions; documents severity and treatment response",
    "Metabolic Screen at Diagnosis — plasma amino acids (exclude NKH — GLRB glycine is NORMAL), urine organic acids, biotinidase activity",
    "Video-EEG — at least one synchronised event recording to confirm non-epileptic aetiology; repeat if clinical picture changes or ID co-occurs (gephyrin class)",
    "POLG Screening — MANDATORY before VPA in any infant/child with encephalopathy or co-existing epilepsy",
    "Gephyrin-Interface Class: Neuropsychological Assessment — at 18-24 months; early intervention if ID confirmed; distinguish from GPHN direct mutation (more severe)",
    "Medical Alert Bracelet — 'Hyperekplexia — not epilepsy — forward-flexion manoeuvre if rigid apnoea'; prevents inappropriate AED administration by emergency services",
    "NICU Monitoring — continuous SpO₂ and apnoea alarm until CLZ effective and apnoea-free ≥48h; forward-flexion competency confirmed in all primary carers before discharge",
    "Driver Safety — all adults with GLRB hyperekplexia: formal driving assessment mandatory; uncontrolled events = driving prohibition; notify licensing authority",
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
KEY_REFERENCES = [
    "Rees MI et al. (1994) Hum Mol Genet — First evidence of GLRB mutations in hereditary hyperekplexia",
    "Harvey RJ et al. (2008) Neuron — GLRB companion gene; α1₂β₃ stoichiometry; homomeric α1 formation mechanism in GLRB LOF",
    "Schaefer N et al. (2015) Hum Mutat — GLRB mutation spectrum; genotype-phenotype correlations; gephyrin-interface class",
    "Bode A & Lynch JW (2013) J Biol Chem — GLRB ECD variants; channel gating in homomeric α1 vs α1₂β₃",
    "Dumoulin A et al. (2009) J Neurosci — GLRB intracellular loop; gephyrin binding domain; synaptic GlyR anchoring mechanism",
    "Lynch JW (2004) Physiol Rev — Molecular structure and physiology of the glycine receptor chloride channel",
    "Thomas RH & Rees MI (2014) Clin Genet — Hyperekplexia genetic spectrum; GLRB ~5%; panel-based diagnosis",
    "Vigevano F et al. (1989) Neuropediatrics — Forward-flexion manoeuvre for neonatal hyperekplexia apnoea",
    "Carta E et al. (2012) Hum Mol Genet — Genotype-phenotype in hyperekplexia; GLRB vs GLRA1 comparison",
    "ILAE Task Force (2017) Epilepsia — Non-epileptic paroxysmal events: classification and differential diagnosis",
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT SAMPLE GENERATOR  (40 patients, seed=497)
# ─────────────────────────────────────────────────────────────────────────────
def _make_patients():
    cats = [
        ("GLRB-AR-Biallelic-LOF", 45, None),
        ("GLRB-AD-Dominant-Negative-Missense", 25, None),
        ("GLRB-ECD-Folding-Intermediate", 15, None),
        ("GLRB-Gephyrin-Interface-Anchor-Defect", 10, None),
        ("Phenocopy-GLRB-Negative", 5, None),
    ]
    pts = []
    pid = 1
    for cat, pct, _ in cats:
        n = max(1, round(40 * pct / 100))
        for _ in range(n):
            is_ar = cat == "GLRB-AR-Biallelic-LOF"
            is_ad = cat == "GLRB-AD-Dominant-Negative-Missense"
            is_ecd = cat == "GLRB-ECD-Folding-Intermediate"
            is_geph = cat == "GLRB-Gephyrin-Interface-Anchor-Defect"
            is_pheno = cat == "Phenocopy-GLRB-Negative"

            sex = random.choice(["M", "F"])
            onset = round(random.uniform(0.0, 0.05) if is_ar or is_ad else
                          (random.uniform(0.0, 0.1) if is_ecd else
                           (random.uniform(0.0, 0.15) if is_geph else
                            random.uniform(0.0, 0.3))), 2)
            age = round(onset + random.uniform(1, 25), 1)

            # Hyperekplexia severity
            apnoea = random.random() < (0.55 if is_ar else 0.30 if is_ad else
                                        0.15 if is_ecd else 0.05 if is_geph else 0.45)
            rigid_baby = random.random() < (0.68 if is_ar else 0.55 if is_ad else
                                            0.35 if is_ecd else 0.12 if is_geph else 0.50)
            startle_falls = random.random() < (0.55 if is_ar else 0.48 if is_ad else
                                               0.38 if is_ecd else 0.25 if is_geph else 0.48)
            # Co-existing features
            epileptic_sz = random.random() < (0.05 if is_ar else 0.05 if is_ad else
                                              0.03 if is_ecd else 0.08 if is_geph else 0.15)
            intel_disability = random.random() < (0.05 if is_ar else 0.03 if is_ad else
                                                  0.02 if is_ecd else 0.35 if is_geph else 0.10)
            # Treatment
            on_clz = random.random() < (0.95 if is_ar or is_ad else
                                        0.80 if is_ecd else 0.50 if is_geph else 0.85)
            on_piracetam = random.random() < (0.15 if is_ar or is_ad else 0.08)
            manoeuvre_trained = random.random() < (0.97 if is_ar or is_ad else
                                                   0.80 if is_ecd else 0.55 if is_geph else 0.78)
            nose_tap_positive = random.random() < (0.95 if is_ar else 0.88 if is_ad else
                                                   0.75 if is_ecd else 0.52 if is_geph else 0.80)
            metabolic_screened = random.random() < 0.87
            video_eeg_done = random.random() < 0.80
            panel_tested = random.random() < 0.92
            polg_tested = random.random() < (0.70 if epileptic_sz else 0.20)

            pts.append({
                "id": f"GLRB-{pid:03d}",
                "sex": sex,
                "age": age,
                "onset_age": onset,
                "category": cat,
                "apnoeic_events": apnoea,
                "rigid_baby": rigid_baby,
                "startle_falls": startle_falls,
                "epileptic_seizures": epileptic_sz,
                "intellectual_disability": intel_disability,
                "on_clonazepam": on_clz,
                "on_piracetam": on_piracetam,
                "forward_flexion_trained": manoeuvre_trained,
                "nose_tap_positive": nose_tap_positive,
                "metabolic_screened": metabolic_screened,
                "video_eeg_done": video_eeg_done,
                "gene_panel_tested": panel_tested,
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
    id_ = sum(1 for p in PATIENTS if p["intellectual_disability"])
    on_clz = sum(1 for p in PATIENTS if p["on_clonazepam"])
    trained = sum(1 for p in PATIENTS if p["forward_flexion_trained"])
    nose_tap = sum(1 for p in PATIENTS if p["nose_tap_positive"])
    metabolic = sum(1 for p in PATIENTS if p["metabolic_screened"])
    video_eeg = sum(1 for p in PATIENTS if p["video_eeg_done"])
    panel = sum(1 for p in PATIENTS if p["gene_panel_tested"])

    etio = [
        {"etiology": e["category"], "n": max(1, round(n * e["pct"] / 100)), "pct": e["pct"]}
        for e in ETIOLOGY_CATALOG
    ]

    tx_summary = [
        {"drug": "Clonazepam", "level": "Level A — First-Line All Classes"},
        {"drug": "Forward-Flexion Manoeuvre", "level": "Level A — Acute Apnoea (Vigevano 1989)"},
        {"drug": "Piracetam", "level": "Level C — Second-Line Adjunct"},
        {"drug": "Glycine Supplementation", "level": "Level C — Investigational (GLRB-specific only)"},
        {"drug": "CLZ Weaning (Age 3-4y)", "level": "Standard Practice — Natural History"},
    ]

    return {
        "kpis": {
            "n_patients": n,
            "apnoeic_events_pct": round(apnoea / n * 100),
            "rigid_baby_pct": round(rigid / n * 100),
            "startle_falls_pct": round(falls / n * 100),
            "epileptic_seizures_pct": round(epileptic / n * 100),
            "intellectual_disability_pct": round(id_ / n * 100),
            "on_clonazepam_pct": round(on_clz / n * 100),
            "forward_flexion_trained_pct": round(trained / n * 100),
            "nose_tap_positive_pct": round(nose_tap / n * 100),
            "metabolic_screened_pct": round(metabolic / n * 100),
            "video_eeg_done_pct": round(video_eeg / n * 100),
            "gene_panel_tested_pct": round(panel / n * 100),
        },
        "etiology_distribution": etio,
        "treatments_summary": tx_summary,
        "monitoring_summary": MONITORING_SCHEDULE[:8],
        "lifecycle": LIFECYCLE_WINDOWS,
        "thresholds": CLINICAL_THRESHOLDS[:8],
        "contraindications_summary": [
            "PHT/CBZ-WRONG-DRUG-for-hyperekplexia",
            "VPA-without-POLG-ABSOLUTE-CI",
            "High-dose-glycine-without-monitoring-HIGH-CAUTION",
            "Discharge-without-forward-flexion-training-ABSOLUTE-SAFETY-FAILURE",
            "Single-gene-GLRB-only-sequencing-DIAGNOSTIC-PITFALL-use-5-gene-panel",
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
            "intellectual_disability_pct": round(sum(1 for p in PATIENTS if p["intellectual_disability"]) / n * 100),
            "forward_flexion_trained_pct": round(sum(1 for p in PATIENTS if p["forward_flexion_trained"]) / n * 100),
            "metabolic_screened_pct": round(sum(1 for p in PATIENTS if p["metabolic_screened"]) / n * 100),
            "video_eeg_done_pct": round(sum(1 for p in PATIENTS if p["video_eeg_done"]) / n * 100),
            "gene_panel_tested_pct": round(sum(1 for p in PATIENTS if p["gene_panel_tested"]) / n * 100),
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
