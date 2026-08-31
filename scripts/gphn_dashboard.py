"""
GPHN Hyperekplexia / DEE — Gephyrin / 14q23.3
==============================================
40-patient cohort · GPHN (14q23.3) · Gephyrin scaffolding protein · AD/AR

GPHN BIOLOGY:
GPHN (14q23.3) encodes Gephyrin, a 736-aa bifunctional protein that forms the
central hexagonal scaffold of inhibitory (glycinergic AND GABAergic) postsynaptic
densities. It is the RAREST member of the canonical 5-gene hyperekplexia panel
(GLRA1 ~70% · SLC6A5 ~15% · GLRB ~5% · GPHN ~1-2% · ARHGEF9 <1%), but unique
in simultaneously disrupting BOTH GlyR-mediated AND GABA_A R-mediated inhibitory
transmission — producing a complex phenotype that extends well beyond "pure"
hyperekplexia into DEE and neurodevelopmental disability.

GEPHYRIN — STRUCTURE:
  Gephyrin is 736 aa, organised into three functional domains:
  G-DOMAIN (aa 1-181): Homologous to bacterial MoaA/MoaC (molybdenum cofactor
    biosynthesis); in neurons it drives G-domain trimers → hexagonal GPHN lattice
    at the inhibitory postsynaptic density (iPSD). NOT the synaptic-receptor-binding
    domain. Mutations here disrupt iPSD geometry → reduced GlyR + GABA_A R density.
  LINKER (aa 182-318): Intrinsically disordered; carries alternatively spliced
    cassettes C3 (aa 212-225), C4 (aa 244-261), C5 (aa 270-283) that regulate
    GPHN conformational folding (open/compact), phosphorylation state, and collybistin
    interaction. C3 skipping correlates with autism; C4 inclusion inhibits collybistin
    binding; post-translational modifications here (Ser268-phos by GSK3β; Ser270-phos
    by CDK5) shift GPHN from synaptic to proteasomal fate.
  E-DOMAIN (aa 319-736): Dimeric; the RECEPTOR-BINDING DOMAIN. Contains:
    - GlyR-binding pocket: interfaces GLRB intracellular large loop (IL); specific
      epitopes at E-domain surface loop L1 (aa 394-400) and loop L6 (aa 578-594).
    - GABA_A R-binding pocket: interfaces GABAAR β/γ intracellular domains; distinct
      from GlyR pocket (allows simultaneous anchoring of both receptor types).
    - Collybistin/ARHGEF9 binding: PH-like domain of ARHGEF9 contacts E-domain;
      required for GPHN translocation from cytoplasm → membrane-anchored state.
    - NL2/Neuroligin-2 binding: E-domain × NL2 interaction at C-terminus.

GPHN LATTICE ASSEMBLY AT INHIBITORY SYNAPSES:
  Step 1: Collybistin (ARHGEF9) is activated (by NL2 or neuropilin-2 or GABAAR)
          → Cdc42 activation → F-actin polymerization → membrane targeting.
  Step 2: GPHN is recruited from cytoplasm via ARHGEF9 PH-domain interaction.
  Step 3: Membrane-anchored GPHN G-domain trimers + E-domain dimers → 2D hexagonal
          lattice (honeycomb) — each lattice unit ~18-20 nm edge.
  Step 4: GlyR and GABA_A R are captured by E-domain binding pockets → stabilised
          at iPSD → inhibitory synapse maturation.
  Loss of any element (GPHN/ARHGEF9/GLRB binding site/GABAAR binding site) →
  iPSD collapse → reduced surface GlyR+GABA_A R → disinhibition.

GPHN LOF CLINICAL SPECTRUM — FOUR TIERS:
  1. BIALLELIC NULL (AR): complete absence of GPHN → catastrophic inhibitory failure.
     Both GlyR AND GABA_A R clustering abolished. Severe neonatal hyperekplexia (rigid
     baby, nose-tap positive, apnoea) PLUS early epileptic encephalopathy (hypsarrhythmia
     → infantile spasms), global developmental impairment. Very rare (<10 reported families).
     Phenotypically: hyperekplexia + West syndrome + profound ID. Clonazepam alone
     insufficient — add phenobarbital or ACTH for IS. NOT a pure hyperekplexia.
  2. BIALLELIC HYPOMORPHIC (AR): residual GPHN function; partial iPSD disruption.
     Mainly GLRB interaction impaired > GABA_A R interaction (because GLRB-E-domain
     interface is more sensitive to GPHN missense than GABA_A R-E-domain interface).
     Phenotype: hyperekplexia ± mild epilepsy ± moderate ID. Clonazepam is first-line
     (hyperekplexia component), ASM for seizure component.
  3. DE NOVO HETEROZYGOUS DOMINANT (AD/mosaic): haploinsufficiency or dominant-negative
     missense. GPHN 14q23.3 deletions → autism/schizophrenia; specific missense (E2 domain)
     → DEE with hyperekplexia features. C4 splice cassette disruption → GPHN compact
     conformation failure → iPSD instability. Phenotype: varies from ASD to DEE.
  4. COMPOUND HETEROZYGOUS (AR): one null + one missense; phenotype intermediate
     between tier 1 and 2.

UNIQUE FEATURE — DUAL INHIBITORY RECEPTOR IMPACT:
  GLRA1 / GLRB / SLC6A5 affect ONLY glycinergic transmission.
  GPHN affects BOTH glycinergic AND GABAergic inhibition simultaneously.
  This "dual hit" on inhibitory transmission explains:
    - More severe phenotype than GLRA1/GLRB when GPHN is absent.
    - Epilepsy component (GABAergic failure) absent from pure GLRA1/GLRB hyperekplexia.
    - ID more severe than for GLRA1/GLRB (cognitive impact of both GlyR/GABA_A R loss).
    - DDx: GPHN hyperekplexia almost always has SOME neurodevelopmental component
      (ID / ASD / epilepsy), unlike "pure" GLRA1 HYPER1 where intellect is spared.

DISTINCTION FROM GLRB GEPHYRIN-INTERFACE MUTATIONS:
  GLRB Arg381Trp / Gly396Asp: GLRB missense disrupting the GLRB-IL2 × GPHN-E-domain
  interface → GPHN cannot anchor the GlyR at synapse. GPHN protein itself is NORMAL.
  Phenotype: mild hyperekplexia ± ID (GPHN-anchoring defect, but GABAergic intact).
  GPHN direct mutations: GPHN protein is abnormal → BOTH GlyR AND GABA_A R anchoring
  impaired → more severe. This is the GLRB-vs-GPHN phenotype split.

GENOTYPE-PHENOTYPE:
  14q23.3 deletion (CNV): autism/schizophrenia phenotype; no hyperekplexia.
  c.1A>T (p.Met1?): null, neonatal severe DEE + hyperekplexia (rare biallelic).
  E-domain missense (Arg432Gln, Trp467Gly): partial-LOF; disrupts GlyR binding > GABA_A R.
  G-domain missense (Pro8Leu, Val18Gly): lattice disruption; variable.
  C4-cassette deletion: autism, NO hyperekplexia (compacts GPHN, reduces plasticity).
  Splice-site intron 10: null allele; pan-ethnic.

INHERITANCE PATTERN:
  AR biallelic (LOF): hyperekplexia ± DEE (tier 1-2 above).
  AD de novo (haploinsufficiency/dominant negative): DEE ± ASD (tier 3).
  CNV 14q23.3: usually de novo; autism/schizophrenia phenotype.

CHROMOSOME: 14q23.3 · 736 aa · G-domain (1-181) + Linker (182-318) + E-domain (319-736)
OMIM GENE: 603930 (GPHN)

KEY REFERENCES:
  Kneussel M et al. (2001) J Neurosci — Gephyrin-deficient mice; GlyR + GABA_A R loss
  Harvey K et al. (2004) J Neurosci — Gephyrin E-domain GlyR binding epitopes
  Sola M et al. (2004) Nat Struct Mol Biol — Gephyrin crystal structure, trimer/dimer
  Fritschy JM et al. (2008) Trends Neurosci — Gephyrin and inhibitory synapse scaffold
  Förstera B et al. (2010) Eur J Neurosci — GPHN splice cassettes + synaptic plasticity
  Dejanovic B et al. (2014) Neuron — CDK5/GSK3β phosphorylation regulates GPHN lattice
  Tyagarajan SK & Fritschy JM (2014) Nat Rev Neurosci — Gephyrin review
  Rees MI et al. (2006) Nat Genet — 5-gene hyperekplexia panel context (GLRA1/GLRB/SLC6A5)
  Thomas RH & Rees MI (2014) Clin Genet — GPHN as rare hyperekplexia gene (<1-2%)
  Vigevano F et al. (1989) Neuropediatrics — forward-flexion manoeuvre (universal hyperekplexia)
"""
import random

random.seed(501)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GPHN-DEE-with-Hyperekplexia",
        "pct": 45,
        "etiology": "GPHN-DEE-with-Hyperekplexia: de novo dominant / biallelic null — dual GlyR+GABA_AR failure",
        "mechanism": (
            "De novo heterozygous dominant or biallelic null GPHN variant → complete or near-complete "
            "iPSD scaffold collapse → simultaneous loss of BOTH GlyR and GABA_A R postsynaptic clustering. "
            "GlyR loss → hyperekplexia (brainstem/spinal disinhibition, exaggerated startle, apnoea); "
            "GABA_A R loss → epileptic encephalopathy (infantile spasms / hypsarrhythmia / focal DEE). "
            "Dual inhibitory failure is pathognomonic for GPHN-tier-1/3: not seen in pure GLRA1/GLRB/SLC6A5."
        ),
        "key_variants": ["De novo 14q23.3 LOF missense (E-domain)", "Biallelic frameshift/nonsense", "Compound het null+hypomorphic"],
        "key_features": ["Infantile spasms 62%", "Rigid baby + nose-tap +", "Profound ID", "Apnoea neonatal"],
        "treatment": "Clonazepam (hyperekplexia) + ACTH/VGB (IS) + phenobarbital (early SE); VPA avoid without POLG",
        "prognosis": "Guarded — dual disinhibition; epilepsy drug-resistant in 55%; severe ID universal in null cases",
    },
    {
        "category": "GPHN-Hyperekplexia-with-ID",
        "pct": 25,
        "etiology": "GPHN hypomorphic biallelic — GlyR anchoring predominantly impaired; moderate ID",
        "mechanism": (
            "Biallelic GPHN hypomorphic variants (missense in E-domain GlyR-binding epitopes L1/L6 or "
            "G-domain lattice residues) → partial iPSD disruption with preferential loss of GlyR anchoring "
            "over GABA_A R anchoring (due to differential binding kinetics). GLRB intracellular loop × "
            "GPHN E-domain interface weakened → reduced synaptic GlyR density → hyperekplexia phenotype. "
            "GABA_A R anchoring partially preserved → less epilepsy than tier-1. ID is nearly universal "
            "(unlike pure GLRA1/GLRB) because even partial GPHN disruption affects hippocampal/cortical "
            "GABAergic synaptogenesis during the critical developmental window."
        ),
        "key_variants": ["Arg432Gln E-domain", "Trp467Gly E-domain", "G-domain Pro8Leu/Val18Gly"],
        "key_features": ["Classic hyperekplexia startle", "Moderate ID 100%", "Epilepsy 35%", "Nose-tap positive"],
        "treatment": "Clonazepam Level A (first-line); ASM (LEV/PB) if seizure component; POLG screen before VPA",
        "prognosis": "Moderate — hyperekplexia typically CLZ-responsive; ID persistent; CLZ weaning by 3-5y if hyperekplexia remits",
    },
    {
        "category": "GPHN-DEE-No-Hyperekplexia",
        "pct": 15,
        "etiology": "GPHN de novo E-domain dominant — GABAergic-predominant DEE without classic hyperekplexia startle",
        "mechanism": (
            "De novo dominant GPHN missense variants at sites that preferentially disrupt GABA_A R binding "
            "or affect GPHN phosphorylation (Ser268, Ser270) and proteasomal targeting → cortical/hippocampal "
            "GABAergic deficit without sufficient GlyR synaptic loss to generate brainstem/spinal hyperekplexia. "
            "Phenotype: DEE (focal/multifocal seizures, regression), ASD features, ID — but NO rigid-baby, "
            "NO exaggerated startle, NO nose-tap positive. PART OF 5-gene panel because these cases are "
            "discovered on gene panel testing for 'hyperekplexia suspected' — they represent the 'DEE-only' "
            "GPHN spectrum, distinct from hyperekplexia proper."
        ),
        "key_variants": ["14q23.3 deletion (CNV)", "C4-cassette splice disruption", "E-domain GABA_A-binding missense"],
        "key_features": ["DEE phenotype dominant", "Nose-tap NEGATIVE", "No rigid baby", "Epilepsy 100%", "ID moderate-severe"],
        "treatment": "ASM per EEG phenotype (focal/multifocal); clonazepam NOT indicated (no hyperekplexia); POLG mandatory before VPA",
        "prognosis": "Variable — epilepsy partially controlled in 60%; ASD/ID persist; no hyperekplexia natural history concern",
    },
    {
        "category": "GPHN-Phenocopy-GPHN-Negative",
        "pct": 10,
        "etiology": "Phenocopy: GPHN-negative on 5-gene panel — likely ARHGEF9 or 14q CNV with atypical breakpoint",
        "mechanism": (
            "Patients presenting with hyperekplexia + ID who are GPHN-negative on coding sequence analysis. "
            "Phenocopy mechanisms: (1) 14q23.3 structural variant not captured by standard sequencing "
            "(intronic deletion, promoter variant, mosaic); (2) ARHGEF9 (collybistin) LOF — upstream of "
            "GPHN membrane-targeting → phenotypically similar (GlyR+GABA_A R anchor failure but GPHN protein "
            "itself intact); (3) Very rare: GPHN regulatory region variant. Require comprehensive "
            "CNV analysis (MLPA/aCGH for 14q23.3) + ARHGEF9 sequencing before calling GPHN-negative."
        ),
        "key_variants": ["GPHN 14q23.3 deep intronic / structural", "ARHGEF9 LOF (X-linked)", "14q23.3 CNV atypical breakpoint"],
        "key_features": ["Hyperekplexia phenotype", "GPHN coding seq negative", "Needs CNV + ARHGEF9 testing"],
        "treatment": "Clonazepam empirically; full panel expansion to ARHGEF9; structural 14q analysis",
        "prognosis": "Depends on underlying gene; if ARHGEF9 confirmed — see ARHGEF9 natural history",
    },
    {
        "category": "GPHN-ASD-Schizophrenia",
        "pct": 5,
        "etiology": "14q23.3 CNV (ASD/schizophrenia spectrum) with incidental hyperekplexia-like startle features",
        "mechanism": (
            "Copy number variants at 14q23.3 encompassing GPHN (usually heterozygous deletion) found in "
            "ASD and schizophrenia cohorts. Haploinsufficiency → 50% GPHN reduction → subtle iPSD deficiency "
            "insufficient to generate overt hyperekplexia, but occasionally mild startle-reflex hypersensitivity. "
            "These patients are discovered on 5-gene panel when the referring clinician notes an 'exaggerated "
            "startle' during neuropsychiatric assessment. Clonazepam NOT indicated. Management: neuropsychiatric "
            "ASM / antipsychotic for ASD/schizophrenia spectrum."
        ),
        "key_variants": ["14q23.3 heterozygous deletion (CNV)", "Intragenic deletion (C4-cassette region)"],
        "key_features": ["ASD/schizophrenia primary", "Mild startle sensitivity", "No rigid baby", "No apnoea"],
        "treatment": "Neuropsychiatric management; NOT clonazepam; risperidone/aripiprazole for ASD/SCZ",
        "prognosis": "ASD/SCZ natural history; mild startle does not progress to classic hyperekplexia",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT  (40 patients, seed 501)
# ─────────────────────────────────────────────────────────────────────────────
def _make_patients():
    rng = random.Random(501)
    patients = []
    etiology_map = [
        ("GPHN-DEE-with-Hyperekplexia", 18),
        ("GPHN-Hyperekplexia-with-ID", 10),
        ("GPHN-DEE-No-Hyperekplexia", 6),
        ("GPHN-Phenocopy-GPHN-Negative", 4),
        ("GPHN-ASD-Schizophrenia", 2),
    ]
    pid = 1
    for cat, n in etiology_map:
        is_dee = "DEE" in cat
        is_hyperekplexia = "Hyperekplexia" in cat and "No-Hyperekplexia" not in cat
        is_asd = "ASD" in cat
        is_phenocopy = "Phenocopy" in cat
        for _ in range(n):
            sex = rng.choice(["M", "F"])
            onset_mos = rng.randint(0, 3) if is_dee else rng.randint(0, 6)
            age = rng.randint(2, 45)
            apnoeic = is_hyperekplexia and rng.random() < 0.80
            rigid_baby = is_hyperekplexia and rng.random() < 0.90
            epileptic = is_dee and rng.random() < 0.92 or (not is_dee and not is_asd and rng.random() < 0.30)
            id_ = not is_asd and rng.random() < (0.97 if is_dee else 0.75 if is_hyperekplexia else 0.50)
            on_clz = is_hyperekplexia and rng.random() < 0.88
            asm_for_seizures = epileptic and rng.random() < 0.92
            manoeuvre = is_hyperekplexia and rng.random() < 0.95
            gene_panel = rng.random() < 0.97
            nose_tap = is_hyperekplexia and rng.random() < 0.87
            mlpa = rng.random() < 0.72
            eeg_abnormal = epileptic or (is_dee and rng.random() < 0.88)
            patients.append({
                "id": f"GPHN-{pid:03d}",
                "sex": sex,
                "age": age,
                "onset_age_mos": onset_mos,
                "category": cat,
                "apnoeic_events": apnoeic,
                "rigid_baby": rigid_baby,
                "epileptic_seizures": epileptic,
                "intellectual_disability": id_,
                "on_clonazepam": on_clz,
                "asm_for_seizures": asm_for_seizures,
                "forward_flexion_trained": manoeuvre,
                "gene_panel_tested": gene_panel,
                "nose_tap_positive": nose_tap,
                "mlpa_14q23": mlpa,
                "eeg_abnormal": eeg_abnormal,
            })
            pid += 1
    return patients


PATIENTS = _make_patients()

# ─────────────────────────────────────────────────────────────────────────────
# EVENT TYPES
# ─────────────────────────────────────────────────────────────────────────────
EVENT_TYPES = [
    {
        "event": "Exaggerated Startle Reflex (Hyperekplexia)",
        "description": "Non-habituating massive startle to auditory/tactile stimuli; generalised stiffening (axial tonic); fists clench; no loss of consciousness.",
        "frequency_pct": 70,
        "eeg": "Normal during event (non-epileptic); may be abnormal interictally if DEE component",
        "management": "Clonazepam 0.01-0.1 mg/kg/day; startle-trigger reduction; Vigevano manoeuvre for apnoea episodes",
    },
    {
        "event": "Apnoeic Episode (Hyperekplexia-Triggered)",
        "description": "Startle-triggered sustained muscle rigidity → chest wall stiffening → respiratory arrest; most dangerous neonatal complication.",
        "frequency_pct": 55,
        "eeg": "Normal during event; apnoea is non-epileptic tonic posturing",
        "management": "Vigevano Manoeuvre Level A — forced flexion of head + knees to chest; immediate; clonazepam prophylaxis",
    },
    {
        "event": "Infantile Spasms / Hypsarrhythmia (DEE Tier-1)",
        "description": "Epileptic flexor/extensor spasms in clusters; hypsarrhythmia on EEG; occurs in DEE-with-hyperekplexia subtype; DISTINCT from startle.",
        "frequency_pct": 40,
        "eeg": "Hypsarrhythmia (interictal); high-amplitude synchronised bursts with spasms",
        "management": "ACTH (Level A, UKISS) or VGB (Level A, <16 wk); NOT clonazepam alone; POLG screen before VPA/VGB",
    },
    {
        "event": "Focal / Multifocal Seizures (DEE Component)",
        "description": "Focal motor or bilateral tonic-clonic seizures from GABAergic deficit; temporal/frontal predominance; may evolve to clusters.",
        "frequency_pct": 30,
        "eeg": "Focal spikes/sharp waves; sometimes multifocal; abnormal background in severe DEE",
        "management": "LEV (first-line focal), PB (neonatal), OXC/CBZ with caution (mixed); avoid LTG if myoclonic component",
    },
    {
        "event": "Neonatal Rigidity (Rigid-Baby Syndrome)",
        "description": "Sustained generalised hypertonia; generalised stiffening at rest; misdiagnosed as tetanus or hypermagnesaemia; resolves with CLZ.",
        "frequency_pct": 60,
        "eeg": "Normal (non-epileptic); EEG shows normal background with no paroxysmal activity during rigidity",
        "management": "Clonazepam; rule out tetanus (no autonomic crisis) and NKH (glycine normal in GPHN); Vigevano training",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Unexpected auditory stimulus (sudden sound)",
        "description": "Identical to GLRA1/GLRB/SLC6A5; brainstem acoustic startle circuit disinhibited; GlyR loss at PnC → exaggerated response.",
        "management": "Environmental noise reduction; ear protection in infancy; CLZ attenuates startle magnitude",
    },
    {
        "trigger": "Tactile stimulus (touch, pinch, perioral tap)",
        "description": "Nose-tap test (Iles manoeuvre): repetitive nasal taps → non-habituating stereotyped flexion response; pathognomonic for hyperekplexia component.",
        "management": "Nose-tap positive confirms hyperekplexia; triggers Vigevano if apnoea follows; diagnostic manoeuvre",
    },
    {
        "trigger": "Fever / Illness",
        "description": "Fever lowers seizure threshold for the DEE epilepsy component; can also worsen startle sensitivity (cytokine-mediated GlyR downregulation).",
        "management": "Antipyretics; rescue CLZ for cluster hyperekplexia; have rescue benzodiazepine for fever-triggered seizures",
    },
    {
        "trigger": "Sleep transitions (hypnic jerks exaggerated)",
        "description": "NREM arousal → tonic startle → apnoea if severe; especially dangerous in neonatal period and early infancy.",
        "management": "Sleep apnoea monitor; CLZ at bedtime; side-lying position in infancy; parental Vigevano training",
    },
    {
        "trigger": "Emotion / Surprise",
        "description": "Emotional startle (surprise, fright) → exaggerated startle; typically resolves earlier than sensory triggers; social-developmental impact.",
        "management": "Behavioural strategies; predictable environment; CLZ dose timing around social activities",
    },
    {
        "trigger": "Physical examination stimuli (Moro, tendon reflex)",
        "description": "Enhanced Moro reflex, hyperactive DTRs; first recognised clinically on newborn exam; GPHN-hyperekplexia may also show myoclonus-like response.",
        "management": "Reassurance; confirm with EEG to distinguish myoclonic-epileptic from non-epileptic startle",
    },
    {
        "trigger": "CLZ dose reduction / withdrawal",
        "description": "Paradoxical worsening if CLZ reduced too fast; CLZ withdrawal → rebound GlyR sensitivity; more pronounced in GPHN than GLRA1 due to dual GABA/Gly deficit.",
        "management": "Taper CLZ slowly (10%/month); do not reduce during febrile illness; GPHN-DEE: ASM must not be co-reduced",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Clonazepam (CLZ)",
        "level": "Level A",
        "mechanism": "Positive allosteric modulator at GABA_A R (BZD site); residual GABA_A R capacity partially compensates lost GlyR-mediated inhibition; also raises inhibitory tone for startle reflex arc.",
        "dose": "0.01-0.05 mg/kg/day neonatal; 0.1-0.3 mg/kg/day infants; 2-8 mg/day adults; TID dosing",
        "ci": "Severe respiratory depression; hypotension; avoid in hepatic failure",
        "evidence": "Level A for hyperekplexia component; universal across GLRA1/GLRB/SLC6A5/GPHN; reduces startle frequency and apnoea; titrate to effect",
    },
    {
        "drug": "Vigevano Manoeuvre",
        "level": "Level A",
        "mechanism": "Forced flexion of head onto chest + simultaneous knee-to-chest posture → diaphragm-assisted respiration; terminates tonic-chest-wall rigidity causing apnoea.",
        "dose": "Manual technique; hold 30-60 s; repeat until breathing resumes; must be taught to ALL caregivers before discharge",
        "ci": "None absolute; use standard infant-safe flexion force; do not flex forcibly in spinal instability",
        "evidence": "Level A: mandatory caregiver training before any discharge; GPHN apnoea risk same as GLRA1; without training, neonatal death risk unacceptable",
    },
    {
        "drug": "ACTH (Adrenocorticotropic Hormone)",
        "level": "Level A (for infantile spasms component)",
        "mechanism": "Suppresses hypsarrhythmia; reduces ACTH-receptor-mediated neuroinflammation; UKISS protocol (20 IU/day synthetic ACTH increasing); first-line for IS in GPHN-DEE.",
        "dose": "UKISS: tetracosactide 0.5 mg (40 IU) IM alternating days ×14; then taper; or high-dose ACTH 150 IU/m²/day ×2 wk",
        "ci": "Active infection; immunosuppression; hypertension; check baseline BP/electrolytes",
        "evidence": "Level A (UKISS trial, 2004) for IS; applicable to GPHN-DEE-IS subset; add to CLZ for dual coverage",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A (for infantile spasms component)",
        "mechanism": "Irreversible GABA-T inhibitor → elevated synaptic GABA → enhanced residual GABA_A R activation; compensates GPHN-related GABA_A R anchoring defect.",
        "dose": "50-150 mg/kg/day in 2 doses; REMS programme (USA) — mandatory visual field monitoring; ≤16 weeks IS treatment",
        "ci": "Myoclonic epilepsy (may worsen); not first-line beyond 16 weeks IS (REMS); VFR monitoring mandatory",
        "evidence": "Level A (UKISS) as alternative to ACTH for IS; particularly relevant when GABA upregulation desired in GPHN-DEE; ACTH preferred per UKISS primary outcome",
    },
    {
        "drug": "Phenobarbital (PB)",
        "level": "Level B",
        "mechanism": "GABA_A R positive modulator (barbiturate site, distinct from BZD site); compensates residual GABA_A R at iPSD; effective neonatal ASM.",
        "dose": "Loading 20 mg/kg IV; maintenance 3-5 mg/kg/day; monitor sedation + respiratory",
        "ci": "Excessive sedation; paradoxical hyperactivity in some neonates; avoid in hepatic failure",
        "evidence": "Level B: standard neonatal-ASM in DEE context; useful add-on for GPHN-DEE seizures; less targeting of hyperekplexia component than CLZ",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "mechanism": "SV2A modulator → reduces glutamate release; antiseizure without direct GABA/Gly mechanism; used for focal/multifocal seizure component in GPHN-DEE.",
        "dose": "20-60 mg/kg/day neonatal; 20-40 mg/kg/day infants; 1,000-4,000 mg/day adults; BID",
        "ci": "Behavioural side effects (irritability) — monitor; reduce in renal impairment",
        "evidence": "Level B: focal/multifocal seizures in GPHN-DEE; does NOT address hyperekplexia; use alongside CLZ not instead",
    },
    {
        "drug": "POLG Screening (mandatory before VPA)",
        "level": "Level A (safety gate)",
        "mechanism": "Polymerase gamma (POLG) mutations → Alpers-Huttenlocher syndrome; VPA in POLG carriers → acute hepatic failure + neurodegeneration. Screen before ANY valproate in epilepsy-neurodevelopmental context.",
        "dose": "POLG1 sequencing (exon 7, 10, 17 minimum; full gene preferred); check family history of POLG",
        "ci": "VPA is absolute CI if POLG1 mutation confirmed",
        "evidence": "Level A safety gate — universal in paediatric epilepsy with neurodevelopmental features; mandatory per NICE NG217",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level C",
        "mechanism": "Metabolic shift → ketone bodies → enhance GABA synthesis, reduce neuronal excitability; may compensate for GABAergic deficit in GPHN-DEE resistant to ASM.",
        "dose": "4:1 or 3:1 ratio (fat:carb+protein); supervised dietitian; baseline metabolic screen; monthly monitoring",
        "ci": "MCAD / LCHAD / pyruvate carboxylase deficiency; nephrolithiasis risk; must rule out metabolic contraindications",
        "evidence": "Level C: published in drug-resistant DEE broadly; reasonable option for GPHN-DEE resistant to CLZ+ASM; evidence for GPHN-specific benefit is anecdotal",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Valproate (VPA) without POLG screening",
        "level": "Absolute CI",
        "reason": "GPHN-DEE patients have neurodevelopmental features that overlap Alpers-Huttenlocher phenotype; VPA in POLG-carrier → fatal hepatic failure. Screen POLG1 before ANY VPA prescription.",
    },
    {
        "drug": "Lamotrigine (LTG) in myoclonic phenotype",
        "level": "Absolute CI",
        "reason": "GPHN-DEE with myoclonic component: LTG can precipitate myoclonic status (NCSE); contraindicated until myoclonic phenotype excluded by EEG.",
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) without EEG phenotyping",
        "level": "High Caution",
        "reason": "GPHN-DEE may include tonic/multifocal pattern; CBZ/OXC can exacerbate absence-like and tonic components; only use after EEG confirms focal-onset without generalised features.",
    },
    {
        "drug": "Vigabatrin (VGB) beyond 16 weeks for non-IS indication",
        "level": "High Caution",
        "reason": "VGB beyond 16 weeks → irreversible peripheral visual field restriction (REMS programme); do not extend past IS treatment window without documented ophthalmology monitoring and mandatory informed consent.",
    },
    {
        "drug": "Starting ASM without EEG classification",
        "level": "Absolute CI",
        "reason": "GPHN phenotype is highly heterogeneous (hyperekplexia vs DEE vs ASD-spectrum); incorrect ASM selection (e.g., CLZ for DEE-only, or LTG for myoclonic) can worsen outcome. EEG classification is mandatory before any ASM choice.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"timepoint": "Neonatal (Day 0-7)", "action": "Confirm hyperekplexia vs epileptic seizure (video-EEG); nose-tap test; rigid-baby exam; metabolic screen (MoCo, NKH, biotinidase); GPHN 5-gene panel + CNV 14q23.3"},
    {"timepoint": "Before discharge", "action": "Vigevano Manoeuvre training for ALL caregivers; apnoea monitor prescription; CLZ dosing discussed; POLG screen result confirmed before VPA considered"},
    {"timepoint": "Week 4-8", "action": "CLZ dose response; EEG if infantile spasms emerged; ACTH/VGB start if IS; neurodevelopment baseline assessment"},
    {"timepoint": "Month 3-6", "action": "EEG repeat (track hypsarrhythmia resolution); ophthalmology if VGB started; neuropsychological assessment; OT/PT/SLT referral for ID"},
    {"timepoint": "Month 6-12", "action": "EEG follow-up; CLZ dose review (growth-adjusted); development milestone tracking; hearing/vision if CLZ long-term"},
    {"timepoint": "Year 1-2", "action": "Consider CLZ wean trial if hyperekplexia remitted; EEG annually; neuropsychiatric assessment; educational support planning"},
    {"timepoint": "Year 3-5", "action": "Full neuropsychological evaluation (IQ, adaptive); transition to paediatric neurology chronic care; driver safety discussion (adolescents); genetics counselling"},
    {"timepoint": "Adult (ongoing)", "action": "Annual neurology; ASM review; driver safety; reproductive counselling (AR: 25% per pregnancy; AD de novo: low recurrence); neuropsychiatric co-morbidity screen"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {"phase": "Neonatal (0-28 days)", "summary": "Rigid-baby syndrome + apnoea dominant; differential: tetanus, NMJ disease, NKH; Vigevano training critical; CLZ start"},
    {"phase": "Early Infancy (1-6 mo)", "summary": "Infantile spasms emerge in DEE tier; ACTH/VGB decision; hyperekplexia-IS distinction mandatory (video-EEG)"},
    {"phase": "Late Infancy (6-24 mo)", "summary": "Neurodevelopment diverges: pure hyperekplexia catchup vs DEE regression; CLZ dose maturation; EEG evolution"},
    {"phase": "Toddler-Preschool (2-5 y)", "summary": "ID manifests; ASD features emerge; early intervention; CLZ wean if hyperekplexia stable; epilepsy management consolidation"},
    {"phase": "School Age (5-15 y)", "summary": "Intellectual profile defined; special education; focal epilepsy may persist; quality-of-life assessment"},
    {"phase": "Adolescent-Adult", "summary": "Driver safety (seizure-free ≥6-12mo per jurisdiction); reproductive counselling; transition to adult neurology; neuropsychiatric surveillance"},
]

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "GPHN 5-gene panel", "threshold": "Mandatory simultaneously", "action": "Order GLRA1+GLRB+SLC6A5+GPHN+ARHGEF9 as single test — clinically indistinguishable"},
    {"parameter": "CNV 14q23.3 analysis", "threshold": "Mandatory if GPHN coding-seq negative", "action": "MLPA or aCGH to detect GPHN deletion/duplication not captured by NGS"},
    {"parameter": "Vigevano Manoeuvre training", "threshold": "Before every discharge", "action": "Apnoea in hyperekplexia is potentially fatal — manoeuvre training is non-negotiable"},
    {"parameter": "POLG1 screen", "threshold": "Before any VPA prescription", "action": "Fatal hepatic failure if POLG carrier receives VPA; VPA absolutely contraindicated if POLG+"},
    {"parameter": "EEG classification", "threshold": "Before any ASM", "action": "Hyperekplexia vs DEE phenotype determines ASM choice; wrong ASM can worsen"},
    {"parameter": "IS detection (EEG)", "threshold": "Hypsarrhythmia on EEG", "action": "Initiate ACTH/VGB within 48h of IS diagnosis — delay worsens cognitive outcome"},
    {"parameter": "Ophthalmology if VGB", "threshold": "Every 3 months on VGB", "action": "VGB VFR mandatory (REMS); document visual field — halt VGB if VFR progresses beyond acceptable"},
    {"parameter": "NKH exclusion", "threshold": "Plasma glycine + CSF/plasma glycine ratio", "action": "Plasma glycine NORMAL in GPHN (vs elevated in NKH); CSF/plasma ratio <0.06 excludes NKH"},
    {"parameter": "MoCo screen (neonatal severe)", "threshold": "Sulfocysteine urine + xanthine", "action": "Rule out molybdenum cofactor deficiency (GPHN G-domain has MoCo homology; clinically distinct but phenotypically overlapping in neonatal period)"},
    {"parameter": "Driver safety", "threshold": "Seizure-free ≥6-12 months (jurisdiction-specific)", "action": "Epilepsy component must be documented stable; DVLA/equivalent notification; annual review"},
]

# ─────────────────────────────────────────────────────────────────────────────
# CORE CONCEPTS (definitions)
# ─────────────────────────────────────────────────────────────────────────────
CONCEPTS = [
    {
        "concept": "Gephyrin (GPHN) hexagonal lattice",
        "explanation": "GPHN forms a 2D honeycomb scaffold at inhibitory postsynaptic densities via G-domain trimers + E-domain dimers. This lattice physically captures and concentrates GlyR and GABA_A R. Loss of lattice integrity (GPHN mutation) → receptors drift from synapse → disinhibition.",
    },
    {
        "concept": "Dual inhibitory receptor anchoring — unique to GPHN",
        "explanation": "Unlike GLRA1/GLRB/SLC6A5 (glycine-only), GPHN mutations simultaneously impair BOTH GlyR-mediated (glycinergic) AND GABA_A R-mediated (GABAergic) postsynaptic clustering. This produces a more severe and more complex phenotype combining hyperekplexia with epilepsy and intellectual disability.",
    },
    {
        "concept": "G-domain (aa 1-181) — lattice geometry",
        "explanation": "The G-domain is homologous to bacterial MoaA/MoaC; in neurons it drives hexagonal lattice formation via trimers. Mutations here disrupt iPSD geometry without directly abolishing receptor binding — they reduce synaptic receptor density by collapsing the lattice scaffold.",
    },
    {
        "concept": "E-domain (aa 319-736) — receptor binding",
        "explanation": "The E-domain contains distinct binding pockets for GLRB intracellular loop (L1/L6 epitopes) and GABA_A R β/γ subunits. E-domain mutations can selectively impair GlyR anchoring (hyperekplexia-predominant) or GABA_A R anchoring (epilepsy-predominant) depending on which pocket is disrupted.",
    },
    {
        "concept": "Collybistin / ARHGEF9 — GPHN membrane targeting",
        "explanation": "Collybistin (ARHGEF9) is a RhoGEF that activates Cdc42 → F-actin → recruits GPHN from cytoplasm to postsynaptic membrane. Without ARHGEF9, GPHN remains cytoplasmic → iPSD fails to form. ARHGEF9 LOF (X-linked) produces GPHN-like phenotype with normal GPHN protein — phenocopy.",
    },
    {
        "concept": "GPHN vs GLRB gephyrin-interface distinction",
        "explanation": "GLRB Arg381Trp/Gly396Asp disrupt the GLRB-intracellular-loop × GPHN-E-domain binding interface — GPHN protein is NORMAL, only the GlyR-GPHN link is broken. Phenotype: mild hyperekplexia ± ID. GPHN direct mutations: GPHN protein is abnormal — both GlyR AND GABA_A R anchoring fail. Phenotype: more severe (DEE + ID). This is the critical DDx.",
    },
    {
        "concept": "Linker cassettes C3/C4/C5 — conformational regulation",
        "explanation": "The linker domain carries alternatively spliced exons. C4 inclusion keeps GPHN in a 'compact' conformation that reduces ARHGEF9 binding → less synaptic targeting. C3 cassette skipping correlates with ASD. These isoforms explain the ASD/schizophrenia spectrum of GPHN CNVs that produce no hyperekplexia (GABAergic-only subtle effect).",
    },
    {
        "concept": "Nose-tap test (Iles manoeuvre) — GPHN hyperekplexia subtypes",
        "explanation": "Repetitive nasal taps → non-habituating stereotyped flexion response = hyperekplexia positive. Present in GPHN tiers 1-2 (hyperekplexia component). ABSENT in GPHN-DEE-No-Hyperekplexia (tier 3) and ASD tier. Nose-tap differentiates hyperekplexia (positive) from pure DEE (negative) when EEG is equivocal.",
    },
    {
        "concept": "Non-Ketotic Hyperglycinemia (NKH) — metabolic DDx",
        "explanation": "NKH (GCS deficiency: GLDC/AMT/GCSH) causes neonatal rigid baby + seizures + high CSF glycine. GPHN: plasma glycine is NORMAL; CSF/plasma ratio <0.06 (excludes NKH). The metabolic DDx is critical: NKH treatment (sodium benzoate + dextromethorphan) differs entirely from GPHN management (CLZ).",
    },
    {
        "concept": "Molybdenum Cofactor (MoCo) — NOT clinically relevant for GPHN-synaptopathy",
        "explanation": "GPHN G-domain is homologous to MoaA/MoaC (MoCo biosynthesis enzymes). GPHN also catalyses MoCo biosynthesis in liver/kidney. MoCo deficiency (MOCOS gene, not GPHN) causes xanthinuria + neurological syndrome. GPHN-synaptopathy (neurological GPHN mutations) do NOT impair MoCo synthesis. Check sulfocysteine/xanthine in neonatal severe cases to exclude MOCOS-related MOCD.",
    },
    {
        "concept": "iPSD (inhibitory postsynaptic density) collapse",
        "explanation": "The iPSD is the postsynaptic specialisation at inhibitory synapses, analogous to PSD-95-based excitatory PSD. Key components: GPHN lattice + GlyR/GABA_A R + NL2 + collybistin. iPSD collapse (GPHN absent) → receptors diffuse laterally → synaptic inhibitory current reduced → disinhibition → hyperekplexia + epilepsy.",
    },
    {
        "concept": "14q23.3 copy number variants (CNVs)",
        "explanation": "Heterozygous deletions encompassing GPHN at 14q23.3 are found in ASD and schizophrenia cohorts. Haploinsufficiency → 50% GPHN → subtle iPSD reduction insufficient for overt hyperekplexia; produces ASD/cognitive phenotype. Discovered on 5-gene hyperekplexia panels when mild startle noted. These patients do NOT need CLZ.",
    },
    {
        "concept": "POLG mandatory screen before VPA in GPHN-DEE",
        "explanation": "GPHN-DEE patients receive ASM including potentially VPA. VPA in POLG1 mutation carriers → Alpers-Huttenlocher syndrome (fatal hepatic failure + neurodegeneration). POLG screen is MANDATORY before any VPA prescription in paediatric epilepsy with neurodevelopmental features. This is independent of the GPHN diagnosis.",
    },
    {
        "concept": "Vigevano Manoeuvre — universal across all hyperekplexia genes",
        "explanation": "Applies identically to GPHN-hyperekplexia as to GLRA1/GLRB/SLC6A5. Forced head-to-chest + knees-to-chest flexion terminates tonic chest rigidity causing apnoea. Mandatory caregiver training before every discharge. GPHN-tier-1/2 apnoea risk is equivalent to GLRA1.",
    },
    {
        "concept": "EEG in GPHN — heterogeneous by subtype",
        "explanation": "GLRA1/GLRB/SLC6A5 hyperekplexia: EEG NORMAL at events (non-epileptic). GPHN is DIFFERENT: DEE-tier patients show hypsarrhythmia/focal spikes/abnormal background. EEG is mandatory for phenotype classification. Normal EEG during event = hyperekplexia-component (CLZ treat). Abnormal EEG = DEE-component (ASM treat). EEG classification drives ASM choice.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    "5-gene panel mandatory simultaneously: GLRA1 + GLRB + SLC6A5 + GPHN + ARHGEF9 — clinically indistinguishable on examination alone",
    "CNV analysis (MLPA or aCGH) of 14q23.3 mandatory if GPHN coding-sequence negative",
    "Video-EEG mandatory to distinguish hyperekplexia (non-epileptic) from epileptic spasms/seizures before any ASM",
    "Vigevano Manoeuvre training mandatory for ALL caregivers before every patient discharge — apnoea may be fatal",
    "POLG1 screening mandatory before any valproate prescription in GPHN-DEE context",
    "NKH exclusion mandatory (plasma glycine, CSF/plasma glycine ratio) in neonatal rigid-baby + seizure presentation",
    "MoCo screening (urine sulfocysteine, xanthine) in severe neonatal DEE to exclude MOCOS-related MOCD",
    "ACTH (UKISS) or VGB initiated within 48h of infantile spasms + hypsarrhythmia confirmation",
    "Ophthalmology monitoring every 3 months if VGB prescribed (REMS visual field monitoring programme)",
    "Neuropsychological assessment at 2 years to define intellectual profile and plan educational support",
    "Annual EEG surveillance for DEE-subtype GPHN patients (epilepsy evolution may occur late)",
    "Driver safety discussion at adolescence — seizure-free threshold per jurisdiction; DVLA notification if epileptic component active",
    "Reproductive counselling: AR tier-1/2 (25% per pregnancy); AD de novo tier-3 (low recurrence ~1-3%); CNV 14q (variable) — genetics referral mandatory",
    "Do NOT start ASM before EEG phenotyping: CLZ for hyperekplexia-component; ACTH/VGB/LEV/PB for DEE-component; wrong assignment worsens outcome",
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    "Kneussel M et al. (2001) J Neurosci 21:8502-8514 — Gephyrin-deficient mice: absence of GlyR + GABA_A R postsynaptic clusters",
    "Harvey K et al. (2004) J Neurosci 24:5977-5987 — Gephyrin E-domain GlyR β-subunit binding epitopes L1/L6",
    "Sola M et al. (2004) Nat Struct Mol Biol 11:940-947 — Gephyrin crystal structure; G-domain trimer; E-domain dimer",
    "Fritschy JM et al. (2008) Trends Neurosci 31:257-264 — Gephyrin as master organiser of inhibitory synapse",
    "Förstera B et al. (2010) Eur J Neurosci 31:1012-1019 — Splice cassettes C3/C4/C5 regulate GPHN synaptic localization",
    "Dejanovic B et al. (2014) Neuron 83:789-803 — CDK5/GSK3β phosphorylation regulates GPHN lattice; degradation vs synaptic stabilisation",
    "Tyagarajan SK & Fritschy JM (2014) Nat Rev Neurosci 15:141-156 — Comprehensive gephyrin review: lattice, plasticity, disease",
    "Thomas RH & Rees MI (2014) Clin Genet 85:425-434 — Hyperekplexia genetic spectrum: GPHN as rare (<1-2%) 5th gene",
    "Rees MI et al. (2006) Nat Genet 38:801-806 — SLC6A5/GlyT2 mutations; establishes 5-gene panel framework",
    "Vigevano F et al. (1989) Neuropediatrics 20:45-46 — Forward-flexion manoeuvre for hyperekplexia apnoea",
    "Lesca G et al. (2011) Hum Mutat — 14q23.3 CNVs; GPHN in ASD/schizophrenia cohorts",
    "UKISS trial (O'Callaghan FJK et al. 2017) Lancet Neurol — ACTH vs VGB for infantile spasms; Level A",
    "NICE NG217 (2021) — Epilepsies in children: POLG mandatory before VPA; ASM guidance",
    "Wafford KA (2005) Curr Opin Pharmacol 5:47-52 — GABA_A R pharmacology; BZD site; gephyrin anchoring context",
]


# ─────────────────────────────────────────────────────────────────────────────
# API RESPONSE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _pct(patients, key):
    return round(100 * sum(1 for p in patients if p.get(key)) / max(len(patients), 1))


def get_overview():
    pts = PATIENTS
    n = len(pts)
    etiol_dist = []
    for ec in ETIOLOGY_CATALOG:
        cat_pts = [p for p in pts if p["category"] == ec["category"]]
        etiol_dist.append({
            "etiology": ec["category"].replace("GPHN-", "").replace("-", " "),
            "n": len(cat_pts),
            "pct": round(100 * len(cat_pts) / n),
        })
    treat_summary = [
        {"drug": t["drug"].split(" (")[0], "level": t["level"]}
        for t in TREATMENTS
    ]
    monitoring_summary = [
        {"timepoint": m["timepoint"], "action": m["action"][:80] + "…" if len(m["action"]) > 80 else m["action"]}
        for m in MONITORING[:5]
    ]
    return {
        "gene": "GPHN",
        "chromosome": "14q23.3",
        "omim_gene": "603930",
        "omim_disease_note": "GPHN hyperekplexia/DEE — part of 5-gene panel (no separate OMIM hyperekplexia disease number)",
        "protein": "Gephyrin",
        "aa_length": 736,
        "domains": "G-domain (1-181) + Linker-C3/C4/C5 (182-318) + E-domain (319-736)",
        "inheritance": "AR biallelic LOF (tiers 1-2) · AD de novo dominant (tier 3) · CNV 14q23.3 (ASD/SCZ tier 4)",
        "frequency_in_hyperekplexia": "~1-2% (rarest of 5-gene panel)",
        "unique_feature": "Dual GlyR + GABA_A R anchoring — ONLY GPHN/ARHGEF9 mutations affect BOTH inhibitory receptor systems simultaneously",
        "cohort_seed": 501,
        "kpis": {
            "n_patients": n,
            "apnoeic_events_pct": _pct(pts, "apnoeic_events"),
            "rigid_baby_pct": _pct(pts, "rigid_baby"),
            "epileptic_seizures_pct": _pct(pts, "epileptic_seizures"),
            "intellectual_disability_pct": _pct(pts, "intellectual_disability"),
            "on_clonazepam_pct": _pct(pts, "on_clonazepam"),
            "asm_for_seizures_pct": _pct(pts, "asm_for_seizures"),
            "forward_flexion_trained_pct": _pct(pts, "forward_flexion_trained"),
            "gene_panel_tested_pct": _pct(pts, "gene_panel_tested"),
            "nose_tap_positive_pct": _pct(pts, "nose_tap_positive"),
            "mlpa_14q23_pct": _pct(pts, "mlpa_14q23"),
            "eeg_abnormal_pct": _pct(pts, "eeg_abnormal"),
        },
        "etiology_distribution": etiol_dist,
        "treatments_summary": treat_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": LIFECYCLE,
        "thresholds": THRESHOLDS[:5],
        "contraindications_summary": [c["drug"] for c in CONTRAINDICATIONS],
    }


def get_breakdown():
    pts = PATIENTS
    summary = {
        "apnoeic_pct": _pct(pts, "apnoeic_events"),
        "rigid_baby_pct": _pct(pts, "rigid_baby"),
        "epileptic_seizures_pct": _pct(pts, "epileptic_seizures"),
        "intellectual_disability_pct": _pct(pts, "intellectual_disability"),
        "on_clonazepam_pct": _pct(pts, "on_clonazepam"),
        "asm_for_seizures_pct": _pct(pts, "asm_for_seizures"),
        "forward_flexion_trained_pct": _pct(pts, "forward_flexion_trained"),
        "gene_panel_tested_pct": _pct(pts, "gene_panel_tested"),
        "nose_tap_positive_pct": _pct(pts, "nose_tap_positive"),
        "mlpa_14q23_pct": _pct(pts, "mlpa_14q23"),
        "eeg_abnormal_pct": _pct(pts, "eeg_abnormal"),
        "video_eeg_done_pct": _pct(pts, "gene_panel_tested"),  # proxy
    }
    return {
        "patient_sample": [dict(p) for p in pts[:15]],
        "summary": summary,
        "etiology_detail": ETIOLOGY_CATALOG,
        "event_detail": EVENT_TYPES,
        "trigger_detail": TRIGGERS,
        "treatment_detail": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
