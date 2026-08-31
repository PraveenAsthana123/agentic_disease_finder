"""
SLC6A5 Hyperekplexia — Hyperekplexia Type 3 / GlyT2 / 11p15.1
===============================================================
40-patient cohort · SLC6A5 (11p15.1) · Glycine Transporter 2 · Autosomal Recessive

SLC6A5 BIOLOGY:
SLC6A5 (11p15.1) encodes GlyT2 (Glycine Transporter 2), the presynaptic Na⁺/Cl⁻-coupled
glycine reuptake transporter in glycinergic interneuron terminals. GlyT2 is the SOLE
mechanism for replenishing glycine in synaptic vesicles at inhibitory glycinergic synapses.
Its transport stoichiometry is 3 Na⁺ + 1 Cl⁻ + 1 glycine (inward) per cycle — thermodynamically
driven by the electrochemical Na⁺ gradient maintained by the Na⁺/K⁺-ATPase.

SLC6A5 IS THE SECOND MOST COMMON GENETIC HYPEREKPLEXIA GENE (~15%), after GLRA1 (~70%) and
ahead of GLRB (~5%), GPHN, and ARHGEF9. All five genes together constitute the MANDATORY
5-gene diagnostic panel — GLRA1/GLRB/SLC6A5/GPHN/ARHGEF9 — because the clinical phenotype
is clinically indistinguishable between these genes.

GLYCINE TRANSPORTER 2 — STRUCTURE AND MECHANISM:
  SLC6A5/GlyT2 is 799 aa, a member of the SLC6 (NSS/neurotransmitter-sodium-symporter) family.
  Structural organization:
    - N-terminal cytoplasmic domain (aa 1-55): regulatory phosphorylation at Thr19, Ser23,
      Ser25; PICK1-interacting segment; PKC phosphorylation targets that modulate surface
      trafficking.
    - TM1-TM6 (aa 56-330): first transport lobe; Na1 (primary Na⁺ binding), Cl⁻ binding
      site, glycine substrate-binding pocket formed at TM1/TM6 interface.
    - Large extracellular loop 2 (EL2, aa 162-238): only extracellular loop >20 residues;
      N-glycosylation sites at N163, N172, N183, N190 — required for surface expression and
      proper glycine affinity.
    - TM7-TM12 (aa 331-780): second transport lobe; Na2 (secondary Na⁺ binding); occluded
      state gating residues; Trp482 (TM8) is critical for glycine channel geometry.
    - C-terminal cytoplasmic tail (aa 781-799): PDZ-binding motif (ETCI) — interacts with
      PICK1 (PRKCA-binding protein 1) for surface trafficking and clustering regulation.
  Transporter cycle: outward-open → glycine-bound-occluded → inward-open → glycine-released
  into presynaptic terminal → vesicular glycine transporter (VIAAT/SLC32A1) loads glycine
  into synaptic vesicles for the next release event.

SLC6A5 LOF MECHANISM — VESICLE GLYCINE DEPLETION:
  Unlike GLRA1/GLRB (postsynaptic GlyR loss), SLC6A5 LOF acts PRESYNAPTICALLY:
    - GlyT2 absence → glycine cannot be reloaded into presynaptic vesicles after release.
    - On first few action potentials: normal glycine release (vesicle stores intact).
    - On sustained inhibitory bursts (exactly when inhibition most needed): vesicle glycine
      depletes → each subsequent release event carries less glycine → GlyR activation
      progressively fails → brainstem PnC (caudal pontine reticular nucleus) and spinal
      cord disinhibited → exaggerated startle reflex (hyperekplexia) + apnoea.
    - GlyT1 (SLC6A9, astrocytic) removes bulk glycine from the synapse but CANNOT replenish
      presynaptic vesicle stores — GlyT2 is the only rescue mechanism.
    - PHARMACOLOGICAL IMPLICATION: Glycine supplementation is mechanistically distinct from
      GLRA1/GLRB contexts: exogenous glycine increases extracellular [glycine], which in
      PARTIAL LOF GlyT2 (residual transporter) may partially restore vesicle loading.
      In COMPLETE NULL cases (null/null), no transporter to activate — supplementation limited.

INHERITANCE: EXCLUSIVELY AUTOSOMAL RECESSIVE — biallelic LOF.
  No dominantly-acting SLC6A5 variants are known (unlike GLRA1 Arg271 dominant-negative or
  GLRB Met177Arg). Both alleles must be lost for disease. Consanguineous families common
  (especially MENA region Trp482Arg founder).

GENOTYPE-PHENOTYPE CORRELATIONS:
  Trp482Arg (c.1444T>C): TM8; disrupts glycine-binding channel geometry; most common
    worldwide MENA/North African founder allele; severe neonatal phenotype.
  Tyr705Cys (c.2114A>G): TM12; protein misfolding, ER retention, trafficking defect; European.
  Ala399Thr (c.1195G>A): TM7; partial transport function retained; South Asian; moderate.
  Gln172His (c.516G>C): EL2; impairs N-glycosylation at N163/N172; East Asian; reduced surface.
  c.1224+2T>C: splice donor intron 9; null allele; pan-ethnic.
  Arg531Gln (c.1592G>A): TM10; impaired Na2-site coupling; European; moderate-severe.

COMPARISON TO GLRA1 AND GLRB:
  GLRA1 (70%): AD dominant-negative (Arg271) or AR; postsynaptic GlyR pore gate.
  SLC6A5 (15%): AR ONLY; presynaptic GlyT2 vesicle-reload failure — DIFFERENT MECHANISM.
  GLRB (5%): AR or AD dominant-negative; postsynaptic; dual deficit (homomeric α1 + gephyrin).
  CLINICAL: virtually indistinguishable → 5-gene panel mandatory.
  SEVERITY: SLC6A5 null/null can be as severe as GLRA1 Arg271; SLC6A5 partial-LOF milder.
  GLYCINE SUPPLEMENTATION RATIONALE: more logical for SLC6A5-partial-LOF than for GLRA1/GLRB.

KEY REFERENCES:
  Rees MI et al. (2006) Nat Genet — SLC6A5 mutations cause hyperekplexia (first report)
  Becker L et al. (2006) Neuron — GlyT2 knockout mouse: hyperekplexia + neonatal death
  Harvey RJ et al. (2008) Nat Genet — SLC6A5 genetic spectrum; 15% of genetic hyperekplexia
  Carta E et al. (2012) Hum Mutat — European/Spanish SLC6A5 cohort; genotype-phenotype
  Giménez C et al. (2012) Front Mol Neurosci — GlyT2 structure-function review
  Pérez-Siles G et al. (2012) Neuropharmacology — Trp482Arg trafficking defect mechanism
  Thomas RH & Rees MI (2014) Clin Genet — hyperekplexia genetic spectrum; frequency ratios
  Vigevano F et al. (1989) Neuropediatrics — forward-flexion manoeuvre
  Lynch JW (2004) Physiol Rev — glycine receptor ion channel physiology (background)
"""
import random

random.seed(499)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "SLC6A5-AR-Biallelic-LOF-Severe",
        "pct": 55,
        "etiology": "SLC6A5 biallelic LOF (AR) — complete null (frameshift/nonsense/splice); severe neonatal",
        "mechanism": (
            "Biallelic SLC6A5 null alleles (homozygous or compound heterozygous frameshift, "
            "nonsense, or splice donor/acceptor variants) → complete absence of GlyT2 protein. "
            "Without GlyT2: presynaptic glycinergic terminals CANNOT replenish synaptic vesicle "
            "glycine stores after each release event. On first few action potentials, inhibitory "
            "glycinergic transmission is intact (pre-loaded vesicles release normally); however, "
            "during sustained high-frequency inhibitory bursts — precisely when brainstem/spinal "
            "inhibition is most critical — vesicle glycine rapidly depletes. Each subsequent "
            "release event carries progressively less glycine → postsynaptic GlyR activation "
            "diminishes → brainstem PnC (caudal pontine reticular nucleus) disinhibited → "
            "non-habituating exaggerated startle, tonic stiffening, and life-threatening apnoea. "
            "GlyT1 (SLC6A9, astrocytic) clears bulk synaptic glycine but CANNOT donate glycine "
            "to presynaptic vesicle stores — only GlyT2 achieves this via its stoichiometry "
            "(3 Na⁺ + 1 Cl⁻ + 1 glycine inward, thermodynamically powered by Na⁺ gradient). "
            "SEVERITY: Most severe SLC6A5 class; overlaps with GLRA1 Arg271 dominant-negative "
            "for neonatal apnoea risk. Consanguineous families enriched. Trp482Arg homozygous "
            "(MENA founder) is the single most common allele in this class."
        ),
        "typical_variants": [
            "c.1444T>C (Trp482Arg) — TM8; MENA/North-African founder; most common null-effect allele",
            "c.2114A>G (Tyr705Cys) — TM12; misfolding + ER retention; European",
            "c.1224+2T>C — splice donor intron 9; null allele; pan-ethnic",
            "Frameshift/nonsense alleles — diverse; pan-ethnic",
        ],
        "eeg_signature": "EEG NORMAL at startle/apnoea events — NON-EPILEPTIC disorder",
        "phenotype": (
            "Neonatal rigid-baby syndrome (~90%). Apnoeic episodes (~85%, life-threatening). "
            "Universal non-habituating exaggerated auditory/tactile startle. Nose-tap test "
            "positive. Intellectual disability uncommon (glycinergic spinal/brainstem circuits "
            "primarily affected; cortical generally spared). Natural history: gradual improvement "
            "by 3-5 years as glycinergic maturation occurs (GlyR subunit switch α2→α1); CLZ "
            "weaning attempted from age 2-3 years."
        ),
    },
    {
        "category": "SLC6A5-AR-Biallelic-Missense-Moderate",
        "pct": 25,
        "etiology": "SLC6A5 biallelic missense (AR) — partial GlyT2 transport function retained; moderate",
        "mechanism": (
            "Compound heterozygous or homozygous missense variants in SLC6A5, each individually "
            "reducing but not abolishing GlyT2 transport activity. Partial transport function "
            "retained: residual GlyT2 reloads presynaptic vesicle glycine at reduced rate. At "
            "low stimulation frequencies, inhibitory glycinergic transmission may be near-normal "
            "(residual transporter keeps up with vesicle demand). At high frequencies or during "
            "stress, the under-capacity transporter cannot replenish vesicles fast enough → "
            "partial glycinergic failure → hyperekplexia. SEVERITY: Moderate; neonatal rigid-baby "
            "less severe or absent; apnoea risk lower (~40%) but still clinically significant. "
            "KEY PHARMACOLOGICAL IMPLICATION: In partial-LOF cases, glycine supplementation is "
            "mechanistically logical — raising extracellular [glycine] increases substrate "
            "availability for residual GlyT2 → improved vesicle loading. This is the rationale "
            "for investigational glycine supplementation in SLC6A5 specifically (different from "
            "GLRA1/GLRB where postsynaptic receptor is defective, not presynaptic transporter). "
            "Ala399Thr (TM7): ~40% residual transport; South Asian enrichment. "
            "Arg531Gln (TM10): Na2-site coupling impaired; ~50% residual transport; European."
        ),
        "typical_variants": [
            "c.1195G>A (Ala399Thr) — TM7; partial transport ~40% WT; South Asian enriched",
            "c.1592G>A (Arg531Gln) — TM10; Na2-site coupling; ~50% WT; European",
            "c.516G>C (Gln172His) — EL2; N-glycosylation impaired at N163/N172; East Asian",
            "Compound heterozygous null + hypomorph — intermediate severity",
        ],
        "eeg_signature": "EEG NORMAL at events — NON-EPILEPTIC; rare coincidental epilepsy not GlyT2-related",
        "phenotype": (
            "Moderate hyperekplexia with prominent startle response; neonatal rigid-baby partial "
            "or mild; apnoea in ~40%; intellectual development generally normal; glycine "
            "supplementation may show benefit in partial-LOF subset; better natural history "
            "than null/null class; CLZ weaning often achievable by age 3 years."
        ),
    },
    {
        "category": "SLC6A5-AR-Homozygous-Founder-MENA",
        "pct": 10,
        "etiology": "SLC6A5 Trp482Arg homozygous — MENA/North African founder; consanguineous families",
        "mechanism": (
            "Homozygous Trp482Arg (c.1444T>C) — the single most recurrent SLC6A5 allele worldwide. "
            "Trp482 lies in TM8 and is critical for defining the glycine-binding channel geometry "
            "between TM1, TM6, and TM8; substitution to Arg (larger, charged) physically blocks "
            "the substrate binding pocket. Protein reaches the plasma membrane (surface expression "
            "partially preserved) but CANNOT transport glycine — a dominant surface-transport "
            "null. Mechanism: TM8 Arg482 steric clash prevents glycine entry into the occluded "
            "binding state → substrate cannot bind → transporter stalls at outward-open "
            "conformation without completing the transport cycle. MENA/North African founder "
            "effect: widespread in Moroccan, Algerian, Tunisian, Egyptian, and Middle Eastern "
            "pedigrees; high carrier frequency in consanguineous communities. Clinically: severe "
            "class — behaviorally identical to null/null BIALLELIC-LOF-SEVERE above but genetically "
            "more tractable (single homozygous allele simplifies cascade testing). POLG "
            "mandatory before VPA if any co-occurring epilepsy."
        ),
        "typical_variants": [
            "c.1444T>C (Trp482Arg) — homozygous; TM8; glycine-binding-channel blockade; MENA founder",
        ],
        "eeg_signature": "EEG NORMAL at events — NON-EPILEPTIC; consanguinity workup mandatory",
        "phenotype": (
            "Severe neonatal rigid-baby syndrome; prominent apnoea requiring Vigevano manoeuvre; "
            "universal startle hyperekplexia; nose-tap strongly positive. Forward-flexion training "
            "BEFORE discharge is mandatory. CASCADE TESTING: all first-degree relatives in "
            "consanguineous pedigrees should be carrier-tested. Prognosis: good with early CLZ "
            "and parental manoeuvre training; weaning often by age 3-4 years."
        ),
    },
    {
        "category": "SLC6A5-AR-Trans-Compound-Atypical",
        "pct": 7,
        "etiology": "SLC6A5 trans-compound heterozygous — unusual allele pairing; intermediate-atypical severity",
        "mechanism": (
            "Compound heterozygous SLC6A5 where one or both alleles are unusual: uncommon "
            "missense variants of uncertain pathogenicity subsequently confirmed by functional "
            "transport assay, or deep intronic variants altering splicing, or exon-spanning "
            "CNV deletions on one allele paired with a point variant on the other. Functional "
            "GlyT2 transport assay in Xenopus oocyte or HEK293 expression system required to "
            "confirm pathogenicity. Some atypical compound heterozygotes show late-onset "
            "hyperekplexia (post-neonatal presentation at 3-12 months), partial auditory-only "
            "startle, or asymmetric phenotype. Cerebellar signs (subtle) have been reported in "
            "rare pedigrees (Mancilla et al. 2022) — GlyT2 expression in cerebellar basket cells "
            "is a minor but real contributing circuit. GENOMIC: exome + MLPA/CNV analysis "
            "may be required when standard sequencing returns only one pathogenic allele. "
            "ACMG classification: VUS on trans-compound allele must reach pathogenic/likely-pathogenic "
            "before reporting as diagnostic."
        ),
        "typical_variants": [
            "Unusual missense + null compound heterozygous — GlyT2 transport assay required",
            "Deep intronic + missense — requires RNA-seq or minigene splicing assay",
            "CNV deletion + point variant — MLPA/SNP-array necessary",
        ],
        "eeg_signature": "EEG NORMAL at hyperekplexia events; atypical myoclonic features require 72h-EEG",
        "phenotype": (
            "Intermediate-to-mild severity; may present post-neonatal; atypical startle modality "
            "distribution; rare subtle cerebellar features possible; functional transport assay "
            "diagnostic confirmation required. Clonazepam and forward-flexion manoeuvre "
            "training remain appropriate pending confirmation."
        ),
    },
    {
        "category": "Phenocopy-SLC6A5-Negative",
        "pct": 3,
        "etiology": "Phenocopy — clinical hyperekplexia; SLC6A5 genetically negative; alternative gene",
        "mechanism": (
            "Clinical hyperekplexia syndrome with exaggerated startle, neonatal hypertonia, and "
            "apnoea confirmed on video-EEG (EEG normal at events) — but full SLC6A5 sequencing "
            "returns no pathogenic biallelic variants. Etiology in this class: GLRA1, GLRB, GPHN "
            "(gephyrin), or ARHGEF9 (collybistin) identified on the comprehensive 5-gene panel. "
            "Rare non-genetic causes (structural brainstem, SSRI neonatal withdrawal, NKH "
            "metabolic) may also mimic. This class underscores the absolute requirement for "
            "simultaneous 5-gene panel rather than sequential single-gene testing — which "
            "wastes diagnostic time and risks false-negative closure. In GPHN/ARHGEF9 cases, "
            "hyperekplexia is accompanied by intellectual disability + epilepsy (distinguishing "
            "feature from pure glycinergic GLRA1/GLRB/SLC6A5 classes where cognition is often "
            "spared). NKH DDx: plasma glycine ELEVATED (vs NORMAL in SLC6A5 hyperekplexia) + "
            "CSF:plasma glycine ratio >0.08 → refer for GCE-P/T/H mutation testing."
        ),
        "typical_variants": [
            "GLRA1 mutations (especially Arg271Leu — most common AD dominant-negative)",
            "GLRB biallelic LOF (Trp170Ser MENA founder; Met177Arg AD)",
            "GPHN biallelic — hyperekplexia + ID + epilepsy triad",
            "ARHGEF9 (collybistin) — X-linked; hyperekplexia + ID",
        ],
        "eeg_signature": "EEG NORMAL at events (confirms non-epileptic hyperekplexia); GPHN/ARHGEF9 may have epileptic EEG",
        "phenotype": (
            "Indistinguishable from SLC6A5 classes on clinical grounds alone — this is precisely "
            "why 5-gene panel is mandatory. Emergency management is identical: Clonazepam "
            "first-line, forward-flexion manoeuvre for apnoea. Genetic diagnosis shapes "
            "inheritance counselling, cascade testing, and long-term monitoring for GPHN/ARHGEF9 "
            "co-morbidities (epilepsy, intellectual disability)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# EVENT TYPES (5)
# ─────────────────────────────────────────────────────────────────────────────
EVENT_DETAIL = [
    {
        "event": "Neonatal-Rigid-Baby-Syndrome",
        "description": (
            "Generalized tonic stiffening in the neonatal period (first hours to days of life). "
            "Non-epileptic (EEG normal during episodes). Body rigid, limbs extended, jaw clenched. "
            "Frequency: ~85-90% of severe SLC6A5 class. May be spontaneous or triggered by touch, "
            "noise, or procedural stimulation."
        ),
        "frequency_pct": 82,
        "eeg": "EEG normal — non-epileptic",
        "management": "Forward-flexion manoeuvre (Vigevano) for apnoea; CLZ for frequency reduction",
    },
    {
        "event": "Startle-Response-Exaggerated-NonHabituating",
        "description": (
            "Exaggerated, non-habituating startle response to auditory, tactile (perioral), or "
            "visual stimuli — the cardinal feature of hyperekplexia. NON-HABITUATING distinguishes "
            "from physiological startle (normal startle habituates on repeat stimulation). "
            "Present in 100% of confirmed SLC6A5 hyperekplexia. Limb jerks, head-drop, or "
            "generalized tonic flexion/extension depending on stimulus modality."
        ),
        "frequency_pct": 100,
        "eeg": "EEG normal — no ictal correlate",
        "management": "Clonazepam Level A; sensory de-sensitization training in older children",
    },
    {
        "event": "Apnoeic-Episode",
        "description": (
            "Life-threatening apnoea triggered by exaggerated startle or spontaneous tonic stiffening "
            "in neonates/infants. GlyT2 failure → profound inhibitory disinhibition of brainstem "
            "respiratory centers → sustained tonic posture → respiratory muscles rigid → apnoea. "
            "REQUIRES: immediate Forward-Flexion (Vigevano) Manoeuvre — neck flexed onto chest, "
            "hips and knees flexed → breaks tonic posture within seconds, restores breathing. "
            "Fatal if untreated. Frequency: ~75-85% in severe null/null class; ~40% in partial-LOF."
        ),
        "frequency_pct": 75,
        "eeg": "EEG normal during apnoea — brainstem tonic, not cortical seizure",
        "management": "Forward-Flexion Manoeuvre LEVEL A — mandatory parental training before discharge",
    },
    {
        "event": "Nocturnal-Startle-Jerks",
        "description": (
            "Nocturnal myoclonic or tonic jerks without apnoea, predominantly in moderate phenotype "
            "(partial-LOF missense class). May present as sleep disruption, brief arousal, or "
            "limb jerks during sleep-wake transition. EEG normal; not epileptic. May mimic "
            "benign sleep myoclonus of infancy or juvenile myoclonic epilepsy in later childhood. "
            "5-gene panel differentiates from epileptic mimics."
        ),
        "frequency_pct": 45,
        "eeg": "EEG normal; no ictal discharge",
        "management": "Low-dose CLZ (before sleep); review 5-gene panel for GPHN/ARHGEF9 epileptic DDx",
    },
    {
        "event": "Myoclonic-Jerk-Burst-Atypical",
        "description": (
            "In atypical trans-compound heterozygous class: brief burst-pattern myoclonic jerks "
            "(2-5 jerks) elicited by unexpected stimuli, with brief post-jerk tonic phase. "
            "Distinguishing feature: EEG shows NO ictal correlate (post-jerk movement artifact "
            "only). May be confused with myoclonic epilepsy syndromes. ABSOLUTE: do not treat "
            "with LTG (lamotrigine) — can worsen myoclonic features in hyperekplexia. "
            "Confirmatory test: repetitive nose-tap test → sustained generalized stiffening "
            "(positive) vs brief jerk-only (atypical)."
        ),
        "frequency_pct": 20,
        "eeg": "EEG normal; post-jerk movement artifact only; no ictal discharge",
        "management": "CLZ first; LTG ABSOLUTE CI; neurology reassessment for GPHN/ARHGEF9 DDx",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS (7)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_DETAIL = [
    {
        "trigger": "Auditory-Startles-Universal",
        "description": "Sudden loud sounds (>70 dB) — universal trigger; non-habituating. Clapping, door slam, alarm.",
        "management": "Environmental modification; CLZ reduces threshold; earplugs in acute severe phase",
    },
    {
        "trigger": "Tactile-Perioral-Touch",
        "description": "Perioral tactile stimulation (especially in neonates) — feeding, oro-pharyngeal suction, nasal tube placement. High-sensitivity trigger in neonates.",
        "management": "Minimize unnecessary oropharyngeal stimulation; trained nursing approach; CLZ pre-procedure",
    },
    {
        "trigger": "Visual-Flash-Sudden-Movement",
        "description": "Sudden unexpected visual stimuli — bright light flash, fast-approaching object. Less common than auditory/tactile but clinically significant.",
        "management": "Reduced lighting in acute neonatal period; CLZ; sunglasses in photosensitive older children",
    },
    {
        "trigger": "Anxiety-Stress-Anticipatory",
        "description": "Anticipatory anxiety (older children/adults) — fear of startle lowers threshold; feeds cycle of hypervigilance and worsened startle.",
        "management": "Psychological support; anxiety management; stable CLZ dosing; school accommodations",
    },
    {
        "trigger": "Fatigue-Sleep-Deprivation",
        "description": "Fatigue increases startle amplitude and lowers threshold — clinically relevant in partial-LOF class where daytime events are typically controlled but fatigue precipitates breakthrough events.",
        "management": "Regular sleep schedule; CLZ at bedtime for nocturnal events; avoid sleep deprivation",
    },
    {
        "trigger": "Febrile-Illness",
        "description": "Fever lowers glycinergic inhibitory threshold — well-documented trigger for breakthrough hyperekplexia events in children on stable CLZ. Mechanism: fever increases CNS metabolic demand + may impair residual GlyT2 activity (protein thermolability).",
        "management": "Antipyretics early; temporary CLZ dose increase per plan; ER guidance letter provided",
    },
    {
        "trigger": "Feeding-Neonatal-Arousal",
        "description": "Neonatal feeding — arousal + perioral stimulation during breastfeeding or bottle feeding triggers startle + possible apnoea. Highest-risk period for neonatal apnoeic events.",
        "management": "Forward-flexion manoeuvre training BEFORE first home feed; feeding position guidance; CLZ pre-feed if severe",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS (8)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_DETAIL = [
    {
        "drug": "Clonazepam",
        "level": "Level A — First-Line All SLC6A5 Classes",
        "mechanism": (
            "GABA-A receptor positive allosteric modulator — benzodiazepine site. Enhances "
            "GABA-A Cl⁻ current → GABAergic COMPENSATION for deficient glycinergic inhibition. "
            "GlyT2 LOF → glycinergic failure → CLZ substitutes GABAergic pathway for brainstem/spinal "
            "inhibition. Highly effective; first-line since Rees MI et al. 2006. "
            "Target dose: 0.01-0.05 mg/kg/day in 2-3 divided doses (neonates); max 0.3 mg/kg/day. "
            "Sedation monitored; dose adjusted for response. "
            "WEANING: attempted from 2-3 years as glycinergic maturation (GlyR α2→α1 subunit switch) "
            "reduces exogenous inhibitory support requirement."
        ),
        "dose": "0.01-0.05 mg/kg/day divided; neonates start low 0.01 mg/kg/day; max 0.3 mg/kg/day",
        "ci": "None absolute (for hyperekplexia indication); monitor sedation; avoid apnoea precipitants",
        "evidence": "Level A — multiple cohort studies; confirmed in SLC6A5 genetic hyperekplexia",
    },
    {
        "drug": "Forward-Flexion-Vigevano-Manoeuvre",
        "level": "Level A — MANDATORY Acute Apnoea Termination; Teach Before Discharge",
        "mechanism": (
            "Neck flexion + hip/knee flexion (Vigevano 1989): placing thumb and forefinger on either "
            "side of infant's neck and flexing chin toward chest SIMULTANEOUSLY with hip/knee flexion. "
            "Physiological mechanism: disrupts tonic brainstem posture circuit; reduces tonic "
            "descending drive to respiratory muscles; breaks apnoeic episode within 5-30 seconds. "
            "MANDATORY: families MUST be trained on this manoeuvre before discharge from neonatal unit. "
            "Failure to train = ABSOLUTE SAFETY FAILURE. Vigevano's original description showed "
            "consistent cessation of apnoeic attacks in GlyR hyperekplexia; confirmed for SLC6A5."
        ),
        "dose": "Neck + hip/knee flexion; trained by physiotherapist/nurse; 24h parental simulation training",
        "ci": "None — life-saving; must not withhold",
        "evidence": "Level A — Vigevano F (1989) + confirmed across all hyperekplexia gene classes",
    },
    {
        "drug": "Piracetam",
        "level": "Level C — Second-Line Adjunct",
        "mechanism": (
            "Cyclic GABA derivative; mechanism in hyperekplexia unclear — possibly modulates "
            "AMPA-mediated excitatory transmission or membrane fluidity in brainstem circuits. "
            "Used as adjunct when CLZ dose cannot be increased due to sedation. Limited controlled "
            "evidence in SLC6A5-specific hyperekplexia; extrapolated from GLRA1 series."
        ),
        "dose": "40-100 mg/kg/day oral; divided doses",
        "ci": "Renal impairment — dose adjust; avoid in neonates <28 weeks",
        "evidence": "Level C — case series; no RCT in SLC6A5; extrapolated from GlyR hyperekplexia",
    },
    {
        "drug": "Glycine-Supplementation",
        "level": "Level C — Investigational (SLC6A5-Partial-LOF Specific; Rationale Strongest Here)",
        "mechanism": (
            "MECHANISTIC RATIONALE (unique to SLC6A5): GlyT2 transport is substrate-limited under "
            "physiological conditions. In partial-LOF GlyT2 (e.g., Ala399Thr with ~40% residual "
            "transport), increasing extracellular [glycine] raises the driving force for the "
            "remaining functional transporter molecules → improved presynaptic vesicle glycine "
            "reload → partially restored inhibitory transmission. "
            "DOSE: oral glycine 150-600 mg/kg/day (experimental; no established protocol). "
            "MONITORING: plasma glycine MUST be measured — target a modest elevation but avoid "
            "NKH-level hyperglycineaemia. CSF glycine:plasma ratio should remain <0.06 (NKH "
            "threshold 0.08). "
            "LIMITATION: In complete null/null SLC6A5, no residual GlyT2 to activate — "
            "supplementation mechanistically ineffective for pure null class. "
            "CONTRAST WITH GLRA1/GLRB: glycine supplementation is less rational for postsynaptic "
            "GlyR defects (where the receptor, not the transporter, is absent/dysfunctional)."
        ),
        "dose": "150-600 mg/kg/day oral; experimental protocol; plasma glycine monitoring mandatory",
        "ci": "Null/null SLC6A5 (no residual GlyT2); NKH not excluded (confirm plasma/CSF glycine first)",
        "evidence": "Level C — Investigational; mechanistic rationale published (Giménez 2012); no formal trial",
    },
    {
        "drug": "Oral-Diazepam-Acute-Emergency",
        "level": "Level C — Acute Seizure Differentiation / Emergency Alternative",
        "mechanism": (
            "Benzodiazepine class; GABA-A modulator. For acute emergency when CLZ is unavailable "
            "or in transit to ER. NOT preferred long-term (shorter half-life, erratic oral absorption "
            "in neonates; buccal midazolam for acute seizures). Useful for differentiating "
            "true epileptic seizures from hyperekplexia — hyperekplexia responds to BZD (GABAergic) "
            "but does not respond to specific AED class drugs (PHT, CBZ, LTG)."
        ),
        "dose": "0.2-0.5 mg/kg rectal/IV acutely; family can use diazepam rectal as emergency rescue",
        "ci": "Prolonged use — CLZ preferred for maintenance; respiratory monitoring in neonates",
        "evidence": "Level C — extrapolated from BZD class evidence in hyperekplexia",
    },
    {
        "drug": "POLG-Screening-Mandatory-Before-VPA",
        "level": "ABSOLUTE SAFETY — Before Any VPA Prescription If Co-occurring Epilepsy",
        "mechanism": (
            "POLG (polymerase gamma) mutations cause Alpers-Huttenlocher syndrome — fatal hepatic "
            "failure precipitated by valproate (VPA). Although SLC6A5 hyperekplexia is NON-EPILEPTIC "
            "(no AED needed for hyperekplexia events), some patients develop coincidental epilepsy "
            "or their hyperekplexia is misdiagnosed as epilepsy → VPA incorrectly prescribed. "
            "MANDATORY: POLG sequencing before ANY VPA prescription in patients with "
            "hyperekplexia (cannot exclude POLG co-mutation). Fatal hepatic failure risk: "
            "Alpers-Huttenlocher, especially in mitochondrial depletion syndrome children. "
            "This applies equally to GLRA1, GLRB, and SLC6A5 hyperekplexia — class-wide rule."
        ),
        "dose": "N/A — diagnostic (POLG sequencing before VPA prescription)",
        "ci": "VPA ABSOLUTELY CONTRAINDICATED in confirmed POLG mutation carriers",
        "evidence": "MANDATORY SAFETY RULE — cross-hyperekplexia class; POLG Alpers-Huttenlocher consensus",
    },
    {
        "drug": "Parental-Forward-Flexion-Training-Program",
        "level": "STANDARD MANDATORY — Before ANY Home Discharge",
        "mechanism": (
            "Structured parent/carer education program: (1) video demonstration of Vigevano "
            "forward-flexion manoeuvre, (2) simulation on doll, (3) supervised practice on patient, "
            "(4) competency assessment, (5) written emergency action plan, (6) hospital contact "
            "details for 24/7 query. INCLUDES: recognition of apnoeic episode; 999/911 calling "
            "criteria; when manoeuvre fails (call emergency services); care during feeding; "
            "travelling precautions. DISCHARGE BLOCKED until training complete and documented."
        ),
        "dose": "Training programme — minimum 24 hours supervised; written competency sign-off",
        "ci": "None — mandatory for all SLC6A5 hyperekplexia with apnoea risk",
        "evidence": "STANDARD MANDATORY PRACTICE — all glycinergic hyperekplexia gene classes",
    },
    {
        "drug": "5-Gene-Panel-Diagnostic-Mandatory",
        "level": "DIAGNOSTIC MANDATORY — Simultaneous GLRA1/GLRB/SLC6A5/GPHN/ARHGEF9",
        "mechanism": (
            "Clinical hyperekplexia phenotype is INDISTINGUISHABLE across GLRA1, GLRB, SLC6A5, "
            "GPHN, and ARHGEF9 genes on clinical grounds alone. Sequential single-gene testing is "
            "WRONG: wastes 2-4 weeks per gene, risks closing diagnosis falsely, misses compound "
            "phenotypes (e.g. GPHN = hyperekplexia + ID + epilepsy triad). "
            "MANDATORY: 5-gene panel simultaneously using NGS gene panel or clinical exome with "
            "prioritized report. Results in 2-4 weeks. Identifies aetiology, guides inheritance "
            "counselling, directs cascade carrier testing, and predicts prognosis."
        ),
        "dose": "N/A — genetic diagnostic test; blood or saliva sample; EDTA tube",
        "ci": "Never do single-gene-only SLC6A5 testing — 5-gene panel is the standard",
        "evidence": "DIAGNOSTIC STANDARD — Thomas RH & Rees MI (2014) Clin Genet; all major hyperekplexia guidelines",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS (5)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATION_DETAIL = [
    {
        "drug": "Single-Gene-SLC6A5-Only-Sequencing",
        "level": "ABSOLUTE CI — DIAGNOSTIC PITFALL",
        "reason": (
            "SLC6A5-only gene testing misses GLRA1, GLRB, GPHN, ARHGEF9 hyperekplexia — all "
            "clinically indistinguishable. Sequential single-gene approach wastes critical "
            "diagnostic time (2-4 weeks per gene) and risks false-negative closure. STANDARD: "
            "5-gene simultaneous panel."
        ),
    },
    {
        "drug": "Discharge-Without-Forward-Flexion-Training",
        "level": "ABSOLUTE SAFETY FAILURE — NEVER DISCHARGE WITHOUT",
        "reason": (
            "Apnoeic episodes in neonatal SLC6A5 hyperekplexia are IMMEDIATELY LIFE-THREATENING. "
            "Parents MUST be trained in forward-flexion (Vigevano) manoeuvre, written emergency plan "
            "provided, and competency documented BEFORE any home discharge. Failure = preventable death."
        ),
    },
    {
        "drug": "PHT-CBZ-LTG-For-Startle-Events",
        "level": "ABSOLUTE CI — WRONG DRUG CLASS",
        "reason": (
            "Phenytoin (PHT), carbamazepine (CBZ), lamotrigine (LTG) are antiepileptic sodium- or "
            "voltage-channel-targeted drugs. They have NO effect on glycinergic hyperekplexia and "
            "some (LTG in particular) can WORSEN myoclonic features. Hyperekplexia requires "
            "GABAergic compensation (CLZ/BZD) — not sodium-channel blockade."
        ),
    },
    {
        "drug": "VPA-Without-POLG-Screening",
        "level": "ABSOLUTE CI — FATAL HEPATIC FAILURE RISK",
        "reason": (
            "Valproate in POLG mutation carriers causes Alpers-Huttenlocher fatal hepatic failure. "
            "Even though SLC6A5 hyperekplexia is non-epileptic, VPA may be prescribed for "
            "coincidental epilepsy or misdiagnosis. POLG screening MANDATORY before any VPA."
        ),
    },
    {
        "drug": "Glycine-Supplementation-Without-Metabolic-Monitoring",
        "level": "HIGH CAUTION — Monitor Plasma Glycine",
        "reason": (
            "Glycine supplementation is investigational in SLC6A5 partial-LOF. Without monitoring, "
            "plasma glycine may reach NKH-equivalent levels (>1000 μmol/L) → encephalopathy. "
            "CSF:plasma glycine ratio must remain <0.06. Glycine supplementation requires "
            "metabolic team co-management and baseline EXCLUDES NKH before initiating."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_SCHEDULE = [
    {"timepoint": "Neonatal (0-4 weeks)", "action": "CLZ initiation; manoeuvre training; metabolic screen (NKH exclude); SLC6A5 5-gene panel sent"},
    {"timepoint": "2 months", "action": "CLZ dose review; apnoea frequency; forward-flexion training reassessment; 5-gene panel result"},
    {"timepoint": "6 months", "action": "CLZ response; neurodevelopmental screen; startle frequency log; plasma glycine if supplementation started"},
    {"timepoint": "12 months", "action": "CLZ dose; cognitive development; nose-tap reassessment; consider CLZ weaning trial (if event-free 6m)"},
    {"timepoint": "2 years", "action": "CLZ weaning attempt if stable; neurodevelopmental assessment; glycinergic maturation assessment"},
    {"timepoint": "3 years", "action": "CLZ wean target; school readiness; driver safety not yet relevant; re-check gene panel if incomplete"},
    {"timepoint": "5 years", "action": "CLZ status; school accommodations; residual startle assessment; genetic counselling for parents"},
    {"timepoint": "Adult", "action": "Driver safety assessment; reproductive genetic counselling (AR — carrier partner screen); CLZ wean if not done"},
    {"timepoint": "Any Febrile Illness", "action": "Temporary CLZ dose increase; antipyretics early; ER guidance letter reminder"},
    {"timepoint": "Any Hospital Admission", "action": "Hyperekplexia card presented; 5-gene result documented in notes; CLZ continued; manoeuvre training for ward staff"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {"phase": "Neonatal (0-28d)", "summary": "Highest-risk: rigid-baby, apnoea, forward-flexion manoeuvre mandatory; CLZ start; 5-gene panel"},
    {"phase": "Infant (1-12m)", "summary": "CLZ titration; apnoea vigilance; parental training reinforcement; developmental monitoring"},
    {"phase": "Toddler (1-3y)", "summary": "CLZ weaning attempts; glycinergic maturation ongoing; speech/motor screen"},
    {"phase": "School-age (3-10y)", "summary": "Most improve; residual startle may persist; school accommodations; anxiety management"},
    {"phase": "Adolescent/Adult", "summary": "Driving assessment; career counselling; reproductive counselling; CLZ wean if achieved"},
]

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_THRESHOLDS = [
    {"parameter": "Clonazepam starting dose (neonate)", "threshold": "0.01 mg/kg/day", "action": "Start here; titrate to response"},
    {"parameter": "Clonazepam max dose", "threshold": "0.3 mg/kg/day", "action": "Maximum; sedation monitoring; respiratory watch"},
    {"parameter": "Plasma glycine (normal — SLC6A5)", "threshold": "Normal (<450 μmol/L)", "action": "SLC6A5 ≠ NKH; if elevated → refer for NKH workup"},
    {"parameter": "CSF:plasma glycine ratio (NKH threshold)", "threshold": ">0.08 = NKH", "action": "SLC6A5 ratio <0.06; ratio >0.08 → NKH GCE mutation testing"},
    {"parameter": "GlyT2 transport assay (functional)", "threshold": "<40% WT = pathogenic", "action": "Xenopus oocyte or HEK293 expression assay; confirms VUS"},
    {"parameter": "GlyT2 surface expression (Western/ICC)", "threshold": "<30% WT = trafficking defect", "action": "Tyr705Cys class — misfolding confirmed"},
    {"parameter": "Forward-flexion training competency", "threshold": "100% — all carers before discharge", "action": "MANDATORY; documented; no exceptions"},
    {"parameter": "5-gene panel result turnaround", "threshold": "<4 weeks target", "action": "NGS panel; prioritize if neonatal ICU"},
    {"parameter": "Startle frequency (target)", "threshold": "<2 events/day on CLZ", "action": "If not achieved: CLZ dose adjustment or piracetam addition"},
    {"parameter": "CLZ weaning start (if event-free)", "threshold": "≥6 months event-free; age ≥2 years", "action": "Trial CLZ reduction 10%/month; monitor for recurrence"},
    {"parameter": "POLG screening trigger", "threshold": "Any VPA prescription planned", "action": "POLG mutation screen BEFORE first VPA dose — no exceptions"},
    {"parameter": "Glycine supplementation plasma monitoring", "threshold": "Target <700 μmol/L plasma glycine", "action": "Baseline, 2w, monthly; stop if >1000 μmol/L or encephalopathy"},
]

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL STANDARDS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_STANDARDS = [
    "5-GENE-PANEL-MANDATORY: GLRA1 + GLRB + SLC6A5 + GPHN + ARHGEF9 simultaneously — never single-gene-first",
    "VIGEVANO-MANOEUVRE-LEVEL-A: Forward-flexion manoeuvre training BEFORE discharge — parental competency documented",
    "CLONAZEPAM-LEVEL-A: First-line pharmacotherapy for all SLC6A5 hyperekplexia classes",
    "NON-EPILEPTIC-CONFIRMED: EEG NORMAL at events — AED-class drugs (PHT/CBZ/LTG) NEVER indicated for hyperekplexia",
    "POLG-MANDATORY: POLG screening before ANY valproate if co-occurring epilepsy develops",
    "NKH-DDx-EXCLUDE: Plasma glycine NORMAL in SLC6A5; if elevated → GCE-P/T/H mutation testing for NKH",
    "GLYCINE-SUPPLEMENTATION-PARTIAL-LOF-ONLY: Only consider for partial-LOF (Ala399Thr type); null/null no benefit",
    "CASCADE-TESTING: AR inheritance — both parents are carriers; all siblings at 25% risk; first-degree carrier screen",
    "PRESYNAPTIC-MECHANISM: SLC6A5 = vesicle-glycine-depletion (PRESYNAPTIC); GLRA1/GLRB = GlyR LOF (POSTSYNAPTIC)",
    "NATURAL-HISTORY: Gradual improvement by age 3-5y (GlyR α2→α1 subunit switch); CLZ wean from age 2-3y if event-free",
    "ER-GUIDANCE-LETTER: Provide written hyperekplexia emergency card (gene, CLZ dose, manoeuvre, do-not-use AEDs)",
    "DRIVER-ASSESSMENT: Adult patients — driving risk assessment; most can drive if >1y seizure/event free on CLZ",
]

# ─────────────────────────────────────────────────────────────────────────────
# CORE CONCEPTS (15)
# ─────────────────────────────────────────────────────────────────────────────
CORE_CONCEPTS = [
    {
        "concept": "GlyT2-SLC6A5-Presynaptic-Reuptake-Transporter",
        "explanation": (
            "GlyT2 (encoded by SLC6A5, 11p15.1) is the presynaptic Na⁺/Cl⁻-coupled glycine "
            "transporter expressed exclusively in glycinergic interneuron terminals. "
            "Stoichiometry: 3 Na⁺ + 1 Cl⁻ + 1 glycine (inward) per cycle. It is the SOLE "
            "mechanism by which the presynaptic terminal replenishes synaptic vesicle glycine "
            "stores after release. Protein: 799 aa; 12 TM; N-terminal cytoplasmic tail (aa1-55); "
            "large EL2 (aa162-238) with N-glycosylation sites; C-terminal PDZ-motif (ETCI) for PICK1."
        ),
    },
    {
        "concept": "Vesicle-Glycine-Depletion-Mechanism",
        "explanation": (
            "SLC6A5 LOF is PRESYNAPTIC (unlike GLRA1/GLRB which are POSTSYNAPTIC). When GlyT2 "
            "is absent: initial inhibitory transmission is normal (pre-loaded vesicles release). "
            "During sustained high-frequency inhibitory bursts (when inhibition most needed), "
            "vesicle glycine depletes progressively → each release event carries less glycine → "
            "GlyR activation diminishes → brainstem/spinal disinhibition → hyperekplexia + apnoea. "
            "This mechanism is unique to SLC6A5 within the hyperekplexia gene series."
        ),
    },
    {
        "concept": "GlyT1-vs-GlyT2-Functional-Distinction",
        "explanation": (
            "GlyT1 (SLC6A9): expressed on astrocytes; bulk synaptic glycine clearance; "
            "stoichiometry 2 Na⁺ + 1 Cl⁻ + 1 glycine. GlyT2 (SLC6A5): expressed on "
            "presynaptic glycinergic terminals; vesicle-reload function; 3 Na⁺ (extra Na⁺ "
            "drives concentrating against gradient into terminal). GlyT1 CANNOT substitute for "
            "GlyT2 in vesicle-reloading (astrocytic location; wrong stoichiometry). This is why "
            "SLC6A5 LOF causes disease despite intact GlyT1."
        ),
    },
    {
        "concept": "Na-Gradient-Dependency-Transport-Stoichiometry",
        "explanation": (
            "GlyT2 transport requires a steep inward Na⁺ gradient (3 Na⁺ per cycle vs 2 for GlyT1). "
            "The extra Na⁺ co-transport thermodynamically allows GlyT2 to concentrate glycine "
            "inside presynaptic terminals against a 10-100× gradient (enabling vesicle loading). "
            "Clinical implication: severe hyponatraemia impairs Na⁺ gradient → reduces GlyT2 "
            "efficacy → may worsen hyperekplexia symptoms acutely. Correct electrolytes."
        ),
    },
    {
        "concept": "AR-Only-Inheritance-No-Dominant-Negative",
        "explanation": (
            "SLC6A5 hyperekplexia is EXCLUSIVELY autosomal recessive — biallelic LOF required. "
            "No dominant-negative mechanism is known (unlike GLRA1 Arg271 or GLRB Met177Arg). "
            "Reason: one functional GlyT2 allele provides sufficient presynaptic glycine "
            "reuptake capacity (haploinsufficiency is not disease-causing). Heterozygous "
            "parents are phenotypically normal carriers — confirmed in >100 MENA pedigrees. "
            "Inheritance counselling: 25% risk per pregnancy for biallelic offspring."
        ),
    },
    {
        "concept": "Trp482Arg-MENA-Founder-TM8-Glycine-Binding-Block",
        "explanation": (
            "Trp482Arg (c.1444T>C): the single most common SLC6A5 pathogenic allele worldwide. "
            "Trp482 (TM8) defines the glycine-binding channel geometry at the TM1/TM6/TM8 "
            "interface. Arg482 (large, positively charged) sterically blocks glycine substrate "
            "entry → transport null despite near-normal surface expression. MENA founder: "
            "widespread in North Africa (Morocco, Algeria, Tunisia, Egypt, Libya) and Middle East. "
            "Homozygous: severe neonatal class. Compound heterozygous with partial allele: moderate."
        ),
    },
    {
        "concept": "Rigid-Baby-Syndrome-Neonatal-Non-Epileptic",
        "explanation": (
            "Generalized tonic stiffening ('rigid baby') in the neonatal period — presenting sign "
            "of severe glycinergic hyperekplexia (GLRA1/GLRB/SLC6A5 all present similarly). "
            "EEG NORMAL during episodes (KEY DIFFERENTIATOR from neonatal seizures). Body rigid, "
            "limbs extended, jaw set. Triggered by handling, noise, or spontaneous. "
            "NON-EPILEPTIC: never treat with AEDs; treat with CLZ (GABAergic) + forward-flexion "
            "manoeuvre. PITFALL: frequently misdiagnosed as neonatal seizures → inappropriate PHT/PB."
        ),
    },
    {
        "concept": "Forward-Flexion-Vigevano-Manoeuvre",
        "explanation": (
            "Vigevano (1989): bilateral thumb-forefinger neck-flexion + simultaneous hip+knee "
            "flexion → within 5-30 seconds, breaks tonic posture circuit → apnoea terminates → "
            "breathing restores. LEVEL A evidence across all hyperekplexia gene classes. "
            "Mechanism: neck flexion disrupts tonic descending brainstem drive to respiratory "
            "muscles. MANDATORY parental training before any home discharge in SLC6A5 neonates. "
            "Training includes: identification of apnoea, manoeuvre execution, emergency calling "
            "criteria, written emergency action plan."
        ),
    },
    {
        "concept": "Nose-Tap-Iles-Test",
        "explanation": (
            "Repetitive nose tapping (perinasally with finger) in a resting infant → provokes "
            "generalized stiffening or exaggerated startle → POSITIVE (Iles manoeuvre). "
            "Normal infants: brief single jerk; NORMAL HABITUATION. Hyperekplexia: "
            "NON-HABITUATING exaggerated response persists on repeat taps. Bedside diagnostic "
            "test; positive in ~90% of severe SLC6A5 and GLRA1/GLRB hyperekplexia. "
            "Not specific to SLC6A5 but helps differentiate from other neonatal conditions."
        ),
    },
    {
        "concept": "Non-Epileptic-Disorder-EEG-Normal",
        "explanation": (
            "SLC6A5 hyperekplexia is a NON-EPILEPTIC disorder — EEG is NORMAL at events. "
            "This is the most important differentiator from epileptic encephalopathies. "
            "Video-EEG during a provoked startle (auditory or nose-tap) shows: no ictal "
            "discharge; movement artifact only; normal background. CONSEQUENCE: "
            "ANTI-EPILEPTIC DRUGS ARE NOT INDICATED for the hyperekplexia events themselves. "
            "PHT/CBZ/LTG are wrong drug class — ABSOLUTE contraindication. CLZ is "
            "first-line because it provides GABAergic (not antiepileptic) inhibitory compensation."
        ),
    },
    {
        "concept": "5-Gene-Panel-Mandatory-Simultaneous",
        "explanation": (
            "Clinical hyperekplexia is phenotypically INDISTINGUISHABLE between GLRA1, GLRB, "
            "SLC6A5, GPHN, and ARHGEF9 on clinical grounds alone. Mandatory simultaneous "
            "5-gene NGS panel (or clinical exome with prioritized report). NEVER sequential "
            "single-gene testing (wastes 2-4 weeks per gene; risks false-negative closure). "
            "Distinguishable only by genetics: GPHN/ARHGEF9 add ID + epilepsy phenotype. "
            "Frequency: GLRA1 70% > SLC6A5 15% > GLRB 5% > GPHN/ARHGEF9 remainder."
        ),
    },
    {
        "concept": "NKH-Metabolic-DDx-Nonketotic-Hyperglycinaemia",
        "explanation": (
            "Nonketotic hyperglycinaemia (NKH) — GCE gene mutations (GLDC, GCSH, AMT) — causes "
            "neonatal hypotonia, hiccups, progressive encephalopathy. PHENOCOPY of hyperekplexia. "
            "KEY DIFFERENTIATOR: In NKH — plasma glycine ELEVATED; CSF:plasma glycine ratio >0.08. "
            "In SLC6A5 hyperekplexia — plasma glycine NORMAL; ratio <0.06. "
            "INITIAL SCREEN: plasma amino acids (before gene panel results). If glycine elevated → "
            "CSF glycine ratio → NKH path. If normal → proceed with 5-gene hyperekplexia panel. "
            "Also distinguish from glycine encephalopathy GLRX5 (different mechanism)."
        ),
    },
    {
        "concept": "Glycine-Supplementation-Partial-LOF-Rationale",
        "explanation": (
            "Investigational in SLC6A5 partial-LOF (e.g., Ala399Thr ~40% residual GlyT2). "
            "Rationale: increasing extracellular [glycine] raises substrate concentration → "
            "remaining functional GlyT2 molecules transport more glycine per unit time → "
            "improved vesicle glycine reload → partially restored inhibitory transmission. "
            "NOT applicable for null/null SLC6A5 (no transporter to activate). "
            "CONTRAST with GLRA1/GLRB: postsynaptic GlyR defects — glycine supplementation "
            "less logical (receptor absent/dysfunctional, not transporter). "
            "Monitoring mandatory: plasma glycine, CSF glycine ratio, liver function."
        ),
    },
    {
        "concept": "Natural-History-GlyR-Maturation-CLZ-Weaning",
        "explanation": (
            "Hyperekplexia severity improves spontaneously with age in most patients. "
            "Mechanism: postnatal GlyR subunit switch — fetal/neonatal GlyR contains α2 subunit "
            "(lower glycine sensitivity); mature GlyR contains α1 (higher sensitivity, more efficient). "
            "Additionally: cortical and cerebellar glycinergic networks mature postnatally. "
            "By 3-5 years: most SLC6A5 patients show marked improvement; CLZ weaning attempted "
            "from age 2-3 years if ≥6 months event-free. Some adult patients remain on low-dose CLZ."
        ),
    },
    {
        "concept": "Driver-Safety-Reproductive-Counselling-Adults",
        "explanation": (
            "Adult SLC6A5 hyperekplexia: driving assessment required if residual startle events. "
            "Most patients (if CLZ-controlled and >1y event-free) can drive under standard "
            "epilepsy-equivalent driving criteria. Reproductive counselling: AR inheritance → "
            "carrier partner must be tested; 25% per-pregnancy risk if partner also carrier. "
            "Cascade testing: ALL first-degree siblings at 25% risk; parents are obligate carriers. "
            "Consanguineous families (MENA Trp482Arg enrichment): wider family cascade important."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
KEY_REFERENCES = [
    "Rees MI et al. (2006) Nat Genet — SLC6A5 mutations cause hyperekplexia (first genetic discovery)",
    "Becker L et al. (2006) Neuron — GlyT2 knockout mouse: hyperekplexia + neonatal lethality model",
    "Harvey RJ et al. (2008) Nat Genet — SLC6A5 genetic spectrum; ~15% of genetic hyperekplexia",
    "Carta E et al. (2012) Hum Mutat — European/Spanish SLC6A5 cohort; genotype-phenotype correlations",
    "Giménez C et al. (2012) Front Mol Neurosci — GlyT2 structure-function; transport mechanism review",
    "Pérez-Siles G et al. (2012) Neuropharmacology — Trp482Arg trafficking defect; MENA founder mechanism",
    "Thomas RH & Rees MI (2014) Clin Genet — hyperekplexia genetic spectrum; GLRA1/SLC6A5/GLRB frequency ratios",
    "Vigevano F et al. (1989) Neuropediatrics — forward-flexion (Vigevano) manoeuvre (Level A)",
    "Lynch JW (2004) Physiol Rev — glycine receptor ion channel physiology (background reference)",
    "Rees MI (2010) Adv Genet — hyperekplexia comprehensive review; all gene classes",
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def _make_patients():
    pts = []
    pid = 1
    for cat, pct in [
        ("SLC6A5-AR-Biallelic-LOF-Severe", 55),
        ("SLC6A5-AR-Biallelic-Missense-Moderate", 25),
        ("SLC6A5-AR-Homozygous-Founder-MENA", 10),
        ("SLC6A5-AR-Trans-Compound-Atypical", 7),
        ("Phenocopy-SLC6A5-Negative", 3),
    ]:
        n = max(1, round(40 * pct / 100))
        is_severe = cat == "SLC6A5-AR-Biallelic-LOF-Severe"
        is_moderate = cat == "SLC6A5-AR-Biallelic-Missense-Moderate"
        is_mena = cat == "SLC6A5-AR-Homozygous-Founder-MENA"
        is_atypical = cat == "SLC6A5-AR-Trans-Compound-Atypical"
        is_phenocopy = cat == "Phenocopy-SLC6A5-Negative"

        for _ in range(n):
            sex = "M" if random.random() < 0.49 else "F"
            if is_severe or is_mena:
                age = round(random.uniform(0.1, 12), 1)
                onset = round(random.uniform(0.0, 0.1), 3)   # neonatal
            elif is_moderate:
                age = round(random.uniform(0.5, 30), 1)
                onset = round(random.uniform(0.0, 0.5), 2)
            else:
                age = round(random.uniform(0.2, 25), 1)
                onset = round(random.uniform(0.0, 0.3), 2)

            apnoea = random.random() < (0.85 if is_severe or is_mena else
                                        0.40 if is_moderate else
                                        0.65 if is_atypical else 0.60)
            rigid_baby = random.random() < (0.88 if is_severe or is_mena else
                                            0.45 if is_moderate else
                                            0.60 if is_atypical else 0.70)
            startle_falls = random.random() < (0.70 if is_severe else
                                               0.60 if is_moderate else
                                               0.72 if is_mena else
                                               0.50 if is_atypical else 0.65)
            intel_disability = random.random() < (0.06 if not is_phenocopy else 0.35)
            epileptic_sz = random.random() < (0.04 if not is_phenocopy else 0.30)
            on_clz = random.random() < (0.95 if is_severe or is_mena else
                                        0.88 if is_moderate else
                                        0.82 if is_atypical else 0.85)
            on_piracetam = random.random() < (0.18 if is_severe else
                                              0.12 if is_moderate else 0.08)
            glycine_supplementation = random.random() < (0.05 if is_severe else
                                                         0.18 if is_moderate else 0.02)
            manoeuvre_trained = random.random() < (0.97 if apnoea else 0.75)
            nose_tap_positive = random.random() < (0.93 if is_severe or is_mena else
                                                   0.78 if is_moderate else
                                                   0.72 if is_atypical else 0.82)
            metabolic_screened = random.random() < 0.88
            video_eeg_done = random.random() < 0.82
            panel_tested = random.random() < 0.93
            polg_tested = random.random() < (0.72 if epileptic_sz else 0.18)

            pts.append({
                "id": f"SLC6A5-{pid:03d}",
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
                "glycine_supplementation": glycine_supplementation,
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
    glycine_supp = sum(1 for p in PATIENTS if p["glycine_supplementation"])
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
        {"drug": "Clonazepam", "level": "Level A — First-Line All SLC6A5 Classes"},
        {"drug": "Forward-Flexion Manoeuvre (Vigevano)", "level": "Level A — Acute Apnoea; Mandatory Training"},
        {"drug": "Piracetam", "level": "Level C — Second-Line Adjunct"},
        {"drug": "Glycine Supplementation", "level": "Level C — Investigational (Partial-LOF Only)"},
        {"drug": "POLG Screening", "level": "MANDATORY Safety — Before Any VPA"},
        {"drug": "5-Gene Panel", "level": "DIAGNOSTIC MANDATORY — Simultaneous"},
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
            "glycine_supplementation_pct": round(glycine_supp / n * 100),
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
            "Single-gene-SLC6A5-only-DIAGNOSTIC-PITFALL-use-5-gene-panel",
            "Discharge-without-forward-flexion-training-ABSOLUTE-SAFETY-FAILURE",
            "PHT-CBZ-LTG-WRONG-DRUG-CLASS-for-hyperekplexia",
            "VPA-without-POLG-screening-ABSOLUTE-CI-fatal-hepatic-failure",
            "Glycine-supplementation-without-metabolic-monitoring-HIGH-CAUTION",
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
        "monitoring": MONITORING_SCHEDULE,
        "lifecycle": LIFECYCLE_WINDOWS,
    }
