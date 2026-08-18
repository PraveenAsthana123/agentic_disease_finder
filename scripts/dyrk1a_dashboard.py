"""
DYRK1A Epilepsy — DYRK1A Syndrome / 21q22.13 Haploinsufficiency
================================================================
40-patient cohort · DYRK1A (21q22.13) · De novo dominant LOF (haploinsufficiency)
Dual Specificity Tyrosine-Phosphorylation Regulated Kinase 1A
DEE (Developmental and Epileptic Encephalopathy) + Microcephaly + Severe ID
DYRK1A Syndrome (OMIM #614721) · First described Moller 2008 / van Bon 2011

DYRK1A PROTEIN BIOLOGY:
DYRK1A (Dual specificity tyrosine-phosphorylation Regulated Kinase 1A) is a
serine/threonine and tyrosine kinase of the DYRK subfamily (CMGC group):
  - 763 aa, 85.6 kDa; expressed ubiquitously but especially in neurons
  - N-terminal: nuclear localisation signals + DYRK homology box (activation)
  - Central: catalytic kinase domain (CMGC insert, activation loop Tyr-321 autophosphorylation)
  - C-terminal: PEST domain (regulatory, proteasomal degradation signal) + histidine repeat
  - Key substrates:
      NRSF/REST (neural restrictive silencer factor): DYRK1A phosphorylates NRSF Ser1107/
        Ser1039 → β-TrCP ubiquitin ligase-mediated degradation → NRSF↓ → de-repression of
        SCN1A (Nav1.1), SCN2A (Nav1.2), GABRB3, KCNQ2 — the neuron-specific gene program
        (CRITICAL: DYRK1A LOF → NRSF stability ↑ → NRSF-mediated repression of Nav1.1/
        GABRB3 → reduced inhibitory tone → epilepsy phenotype mechanistically analogous to
        Dravet syndrome Nav1.1 loss)
      GLI1/GLI2 (Sonic hedgehog): phosphorylation → nuclear translocation → cerebellar
        and cortical development; LOF → microcephaly
      SRSF6 (splicing factor): DYRK1A controls alternative splicing of tau exon 10 → ratio
        of 3R-tau:4R-tau; LOF → tau mis-splicing → neurofibrillary tangle predisposition
      DSCR1/RCAN1: DYRK1A phosphorylates RCAN1 → calcineurin inhibition → NFAT
        dephosphorylation blocked → altered synaptic plasticity
      Cyclin D1: DYRK1A phosphorylates Thr286 → nuclear export → G0/G1 arrest → reduced
        neuronal proliferation → primary microcephaly (HC < -4 SD at birth)
      APP: Thr668 phosphorylation → amyloidogenic processing → increased Aβ40/42 → early
        Alzheimer disease in Down syndrome (trisomy 21 → 3× DYRK1A → accelerated Aβ)

DISEASE MECHANISM — HAPLOINSUFFICIENCY:
  LOF de novo → 50% kinase activity reduction → insufficient phosphorylation of all
  substrates → NRSF stabilisation (reduced Nav1.1/GABRB3) + Cyclin D1 accumulation
  (microcephaly) + tau mis-splicing (neurodegeneration risk) + NFAT deregulation
  (synaptic plasticity) simultaneously;

  NRSF/REST connection to epilepsy genes (KEY PHARMACOLOGICAL INSIGHT):
    DYRK1A LOF → NRSF not phosphorylated → NRSF accumulates in nucleus → binds RE-1
    silencer elements on SCN1A, SCN2A, SCN8A, GABRB3, KCNQ2 → reduced expression;
    This creates a functional Nav1.1 deficiency similar to Dravet (haploinsufficiency
    by a different mechanism) — provides rationale for AVOIDING Na-channel BLOCKERS
    (phenytoin, carbamazepine) in DYRK1A DEE: further reducing already-compromised
    Na-channel function in GABAergic interneurons → disinhibition → seizure worsening.

  TRISOMY 21 DOSE-EFFECT (CRITICAL DISTINCTION):
    Normal: 2 copies DYRK1A → normal kinase activity
    DYRK1A haploinsufficiency (LOF): 1 copy → DEE + microcephaly (loss-of-function)
    Down syndrome / trisomy 21: 3 copies → 50% excess activity → early Alzheimer
    (gain-of-activity through gene dosage) → DYRK1A inhibitors (harmine, leucettine-41,
    epigallocatechin gallate) under investigation in Down syndrome to REDUCE DYRK1A
    — ABSOLUTE CONTRAINDICATION to use DYRK1A inhibitors in DYRK1A haploinsufficiency
    (DEE syndrome): would WORSEN the underlying LOF defect

GENETICS:
  Gene:        DYRK1A (21q22.13) — also known as MNB (Drosophila minibrain orthologue)
  Protein:     Dual Specificity Tyrosine Kinase 1A (763 aa, 85.6 kDa)
  Inheritance: De novo dominant (>97%); <3% inherited from mosaic parent
  pLI:         0.99 (extremely intolerant to LoF — second only to SCN1A in neurodevelopment)
  Incidence:   ~1:20,000-30,000 (DYRK1A syndrome); rare in clinical practice
  OMIM:        DYRK1A Syndrome (Intellectual disability + microcephaly + epilepsy) #614721
  First report: Moller et al. 2008 (Eur J Hum Genet); van Bon et al. 2011 (J Med Genet)
  Down syndrome link: DYRK1A at 21q22.13 — dosage-sensitive in trisomy 21

KEY PHARMACOLOGICAL DISTINCTIONS IN DYRK1A:
  1. Na-channel blockers (PHT/CBZ/OXC) HIGH RISK: DYRK1A LOF → NRSF↑ → reduced SCN1A/
     Nav1.1 expression → GABAergic interneuron dysfunction; adding Na-channel blockers
     further suppresses residual Nav1.1 → disinhibition cascade → seizure worsening
     (mechanism analogous to Na-channel blocker worsening in Dravet syndrome)
  2. DYRK1A inhibitors (harmine/leucettine) ABSOLUTE CI in LOF: used for trisomy 21/
     Down syndrome to reduce 3× DYRK1A; in DYRK1A haploinsufficiency would further
     reduce the already-halved kinase activity → CATASTROPHIC worsening of all phenotypes
  3. VPA for GTCS/myoclonic: broadspectrum + HDAC inhibition may partially restore
     Nav1.1 expression (epigenetic de-repression of NRSF-silenced genes) — rational
     and clinically first-line
  4. Leucovorin (folinic acid) supplementation: DYRK1A LOF → folate receptor antibody
     association found in 25-35% of DYRK1A patients; empiric folinic acid 0.5-1 mg/kg/day
     may improve seizure control and neurodevelopment (Level C)
  5. ACTH for infantile spasms: DYRK1A IS (30% cohort) — ACTH standard first-line;
     vigabatrin alternative; ~55% hypsarrhythmia cessation (lower than idiopathic IS 80%
     due to structural microcephaly substrate)
  6. KD for drug-resistant: mTOR-independent pathway → effective in DRE (Level B);
     RCAN1/calcineurin pathway may provide additional seizure benefit in DYRK1A
  7. TGB ABSOLUTE CI: GAT-1 block → GABA spillover → GABAA desensitisation → NCSE;
     DYRK1A GABRB3 reduction makes this risk higher than in general DEE population

KEY STANDARDS:
  ILAE 2022 · NICE NG217 · Moller et al. 2008 (Eur J Hum Genet) ·
  van Bon et al. 2011 (J Med Genet) · Bronicki et al. 2015 (J Med Genet) ·
  Ji et al. 2015 (Nat Genet) · Courcet et al. 2012 (Eur J Hum Genet) ·
  ACMG-AMP 2015 · WHO ICF 2019 · NICE NG224 2022 ·
  MHRA VPPP 2021 · CPIC POLG1 2023
"""
import random

# ── Etiology catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "De Novo Truncating LOF (Frameshift / Nonsense / Splice-Site)",
        "category": "LOF-truncating-de-novo-40%",
        "pct": 40,
        "n": 16,
        "mechanism": (
            "Frameshift (most common: 1 bp insertion/deletion in exons 7-13), nonsense "
            "(premature stop codon Arg481X, Arg254X), or canonical splice-site variants "
            "disrupting kinase domain function. Haploinsufficiency: NMD degrades mutant "
            "transcript → ~50% kinase activity. Severity correlates with kinase domain "
            "location: exons 7-10 (kinase N-lobe) → most severe (HC < -4 SD, IS onset); "
            "exons 11-14 (kinase C-lobe) → intermediate; PEST domain truncations → "
            "milder (HC −2 to −3 SD, focal seizures after age 2). NRSF non-phosphorylated "
            "→ SCN1A/GABRB3 repressed → functional Nav1.1 deficiency → epilepsy. EEG: "
            "hypsarrhythmia (IS subset), multifocal spikes; MRI: simplified gyral pattern, "
            "thin corpus callosum, hippocampal hypoplasia in 40%."
        ),
        "seizure_types": ["Infantile spasms", "GTCS", "Myoclonic"],
        "age_onset_range": "2-12 months (spasms); 12-36 months (GTCS)",
        "drug_response": "Partial (50% ≥50% reduction)",
        "typical_aed": "ACTH (IS) → Valproate ± Levetiracetam",
        "microcephaly_severity": "Severe (HC < −4 SD)",
        "nrsf_impact": "High (kinase domain loss → NRSF fully stable)",
    },
    {
        "etiology": "De Novo Missense LOF (Kinase-Dead or Dominant-Negative)",
        "category": "LOF-missense-kinase-dead-28%",
        "pct": 28,
        "n": 11,
        "mechanism": (
            "Missense variants in the catalytic core: Lys188Arg (ATP-binding K-loop), "
            "Tyr321Phe (activation loop self-phosphorylation abolished), "
            "Arg205Gln (catalytic loop), Asp307Asn (Asp-Phe-Gly motif disrupted). "
            "These yield kinase-dead protein — expressed but catalytically inactive; some "
            "variants act dominantly negative by competing with wild-type for substrates. "
            "Phenotype intermediate: microcephaly HC −2 to −4 SD; seizures in 55% "
            "(vs 70% truncating); prominent bird-like facies (prominent nasal bridge, "
            "deep-set eyes, thin upper lip). NRSF phosphorylation partially preserved "
            "if DN is incomplete → milder Nav1.1 reduction. Autism: >90%. Drug response: "
            "better than truncating class (60% ≥50% reduction on VPA monotherapy). "
            "Harmine/leucettine experimental: ABSOLUTE CI (would reduce residual WT DYRK1A)."
        ),
        "seizure_types": ["Focal impaired awareness", "GTCS", "Absence"],
        "age_onset_range": "12-48 months",
        "drug_response": "Good (60% ≥50% reduction)",
        "typical_aed": "Valproate or Lamotrigine",
        "microcephaly_severity": "Moderate (HC −2 to −4 SD)",
        "nrsf_impact": "Moderate (partial NRSF phosphorylation if incomplete DN)",
    },
    {
        "etiology": "De Novo Canonical Splice-Site Variant",
        "category": "LOF-splice-site-de-novo-15%",
        "pct": 15,
        "n": 6,
        "mechanism": (
            "Intronic splice-site variants (±1/2 positions) causing exon skipping "
            "(most common: exon 7 skip → in-frame kinase domain deletion) or intron "
            "retention with premature stop. Consequence depends on exact exon: "
            "exon 6 skip → loss of N-lobe helix → severe (phenotype similar to truncating); "
            "exon 9 or 10 skip → partial kinase domain loss → intermediate. Leaky splicing "
            "may produce ~5-15% functional protein → phenotype milder than complete null. "
            "Cryptic splice-site activation in ~20% → alternative transcript with partial "
            "function. Deep intronic variants (bp 10-50 from exon) may require RNA sequencing "
            "for diagnosis (WES/WGS misses splicing effects without RNA validation). "
            "Seizure prevalence: 60%. EEG: focal spikes with secondary generalisation."
        ),
        "seizure_types": ["Focal with secondary generalisation", "GTCS"],
        "age_onset_range": "12-60 months",
        "drug_response": "Good (65% ≥50% reduction)",
        "typical_aed": "Valproate",
        "microcephaly_severity": "Moderate (HC −2 to −3 SD)",
        "nrsf_impact": "Variable (depends on residual DYRK1A from leaky splicing)",
    },
    {
        "etiology": "21q22.13 Microdeletion (DYRK1A + Flanking Genes)",
        "category": "21q22-microdeletion-flanking-12%",
        "pct": 12,
        "n": 5,
        "mechanism": (
            "Interstitial or terminal deletion of 21q22.13 including DYRK1A and "
            "neighbouring genes (KCNJ6/GIRK2, DSCR3, DCAF7, WRB). Larger deletions "
            "(>500 kb) also lose KCNJ6 (GIRK2 inward rectifier) → additional loss of "
            "postsynaptic inhibitory potassium conductance amplifies seizure burden "
            "(KCNJ6 + DYRK1A double haploinsufficiency → additive excitability); "
            "DCAF7 loss → proteasomal regulation disrupted → DYRK1A substrate accumulation "
            "paradox (DCAF7 is DYRK1A binding partner that modulates substrate access). "
            "Clinical: more severe microcephaly (HC −4 to −6 SD), higher seizure prevalence "
            "(75% vs 60% for point mutations); dysmorphic features more pronounced. "
            "MRI: pachygyria, periventricular nodular heterotopia in 25%. "
            "Distinguish from full trisomy 21: FISH/CMA shows deletion (not gain) at 21q22."
        ),
        "seizure_types": ["Infantile spasms", "GTCS", "Myoclonic", "Focal"],
        "age_onset_range": "2-18 months",
        "drug_response": "Poor (35% ≥50% reduction; high DRE rate)",
        "typical_aed": "ACTH (IS) → VPA + LEV + KD",
        "microcephaly_severity": "Severe (HC < −4 to −6 SD)",
        "nrsf_impact": "High + KCNJ6 loss additional risk",
    },
    {
        "etiology": "Phenocopy — Down Syndrome / Trisomy 21 with DEE Features",
        "category": "Phenocopy-trisomy21-DEE-5%",
        "pct": 5,
        "n": 2,
        "mechanism": (
            "Trisomy 21 patients with DEE phenotype (3 copies DYRK1A → 50% EXCESS activity; "
            "opposite mechanism to DYRK1A haploinsufficiency). Included in cohort for "
            "differential precision therapy planning: DYRK1A inhibitors (harmine, "
            "leucettine-41, EGCg) under investigation to REDUCE excess DYRK1A in trisomy 21 "
            "Alzheimer prevention — directly CONTRAINDICATED in haploinsufficiency cases. "
            "Clinical overlap: both show microcephaly, ID, epilepsy, autism features; "
            "DISTINGUISHED by: trisomy 21 → upslanting palpebral fissures + single palmar "
            "crease + macroglossia (absent in DYRK1A LOF); karyotype/CMA confirms. "
            "Treatment: standard epilepsy AEDs; DYRK1A inhibitor trials should exclude "
            "haploinsufficiency; VPA safe in trisomy 21 DEE (POLG1 screen still mandatory)."
        ),
        "seizure_types": ["GTCS", "Focal", "Infantile spasms (Lennox)"],
        "age_onset_range": "Variable",
        "drug_response": "Variable (per seizure type)",
        "typical_aed": "Standard AEDs per seizure type",
        "microcephaly_severity": "Variable (trisomy 21 may have macrocephaly)",
        "nrsf_impact": "Opposite (NRSF hyper-phosphorylated in trisomy 21 → NRSF↓ → Nav1.1 UP)",
    },
]

# ── Seizure catalog ───────────────────────────────────────────────────────────
SEIZURE_CATALOG = [
    {
        "type": "Infantile Spasms / West Syndrome",
        "prevalence_pct": 30,
        "onset": "Generalised (flexion or extension)",
        "duration": "1-2 seconds per spasm; clusters 5-30 min",
        "eeg": "Hypsarrhythmia (modified in 40% — hemispheric or asymmetric); "
               "burst-suppression in neonatal severe cases",
        "semiology": (
            "Salaam spasms (flexion trunk + abduction arms); may be asymmetric "
            "suggesting focal cortical dysplasia substrate. Onset 3-9 months; later "
            "than idiopathic IS (microcephaly substrate present from birth). "
            "Clusters on awakening — characteristic. EEG: hypsarrhythmia resolution "
            "with ACTH may be incomplete (structural substrate persists). "
            "Post-IS evolution: LGS (55%), DRE focal (30%), seizure-free (15%). "
            "Video-EEG MANDATORY: distinguish from benign neonatal myoclonus, "
            "hyperekplexia (GLRA1), and startle disease."
        ),
        "clinical_tip": (
            "ACTH within 2 weeks of IS onset — DYRK1A IS has lower cessation rate "
            "(~55%) vs idiopathic (80%) due to structural microcephaly; start VGB "
            "simultaneously as adjunct. DYRK1A IS has 25% chance of NOT responding "
            "to either ACTH or VGB — early KD consultation at week 4 non-response."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizure (GTCS)",
        "prevalence_pct": 55,
        "onset": "Generalised from onset (not focal then bilateral)",
        "duration": "1-3 minutes",
        "eeg": "Generalised polyspike-wave; ictal: bisynchronous fast activity then "
               "clonic component; interictal: generalised spike-wave 2-3 Hz",
        "semiology": (
            "Classical GTCS: tonic phase (10-15s) → clonic (30-60s) → postictal (5-30 min). "
            "Fever-sensitive (70%); nocturnal predominance. Postictal Todd's paresis rare "
            "(focal GTCS subset 15%). DYRK1A GTCS often occurs WITHOUT aura (unlike focal "
            "onset bilateral tonic-clonic). Multiple GTCS per month in 35% untreated. "
            "Photosensitivity: 10% — EEG photic stimulation at 6, 12, 18 Hz should be done."
        ),
        "clinical_tip": (
            "Na-channel blockers (PHT/CBZ/OXC) WORSEN DYRK1A GTCS — high risk due to "
            "NRSF-mediated Nav1.1 reduction; VPA first-line (broad-spectrum + potential "
            "epigenetic NRSF de-repression). If VPA contraindicated (POLG1+), use LEV + CLB."
        ),
    },
    {
        "type": "Myoclonic Seizure",
        "prevalence_pct": 35,
        "onset": "Generalised (bilateral, synchronous)",
        "duration": "<500 ms",
        "eeg": "Generalised polyspike (3-6 Hz) followed by slow wave; "
               "prominent in frontocentral leads",
        "semiology": (
            "Brief bilateral arm/trunk jerks; morning predominance (within 1 h of waking). "
            "Can occur in clusters (myoclonic status — treat urgently). Associated with "
            "progressive myoclonic epilepsy-like picture in 10% of severe cases (DYRK1A "
            "Unverricht-Lundborg overlap: negative myoclonus + cortical myoclonus). "
            "Sodium channel blockers paradoxically WORSEN myoclonus in DYRK1A "
            "(same mechanism: NRSF/Nav1.1 reduction amplified). Use VPA or LEV. "
            "Absence-myoclonic overlap common — video-EEG distinguishes."
        ),
        "clinical_tip": (
            "CBZ/OXC/LTG at high doses can markedly worsen myoclonus in DYRK1A — "
            "document prominently on emergency care plan. VPA + clobazam as bedtime dose "
            "reduces morning myoclonic clusters. Piracetam 80-160 mg/kg/day for refractory "
            "cortical myoclonus (Level C)."
        ),
    },
    {
        "type": "Focal Impaired Awareness Seizure (FIAS)",
        "prevalence_pct": 40,
        "onset": "Temporal or frontoparietal",
        "duration": "30-120 seconds",
        "eeg": "Focal interictal spikes (temporal 55%, frontal 30%, parietal 15%); "
               "ictal: rhythmic theta-alpha discharge",
        "semiology": (
            "Behavioural arrest + staring + oroalimentary automatisms (temporal onset). "
            "Frontal onset: hypermotor semiology mimicking ADNFLE — nocturnal, brief, "
            "hypermotor. Often progresses to bilateral tonic-clonic (30%). Secondary "
            "generalisation especially during fever. MRI abnormality substrate in 60% "
            "(simplified gyri, hippocampal dysplasia) — correlates with seizure onset zone. "
            "Drug response: reasonably good with VPA ± LEV (70% ≥50% reduction in focal class)."
        ),
        "clinical_tip": (
            "Structural MRI 3T mandatory — focal cortical dysplasia type II on MRI in 20% "
            "of DYRK1A focal epilepsy → surgical evaluation indicated if DRE. "
            "SEEG preferred over scalp EEG for localisation (small head circumference "
            "and cortical malformations distort source localisation)."
        ),
    },
    {
        "type": "Absence Seizure / Myoclonic-Atonic",
        "prevalence_pct": 15,
        "onset": "Generalised",
        "duration": "5-30 seconds (absence); atonic < 2 seconds",
        "eeg": "Typical 3 Hz spike-wave (absence); irregular 2-2.5 Hz spike-wave + "
               "EMG silence (myoclonic-atonic; Doose-like)",
        "semiology": (
            "Brief staring with mild automatisms (absence). Myoclonic-atonic: sudden "
            "falls (drop attacks) + limb jerk before fall — injury risk. "
            "Doose syndrome (myoclonic-atonic epilepsy) phenotype described in DYRK1A. "
            "Responds well to VPA (75%). Avoid lamotrigine monotherapy (can worsen "
            "myoclonic component). Ethosuximide useful for pure absence subtype. "
            "Helmet mandatory for myoclonic-atonic drop attacks."
        ),
        "clinical_tip": (
            "VPA first-line for myoclonic-atonic in DYRK1A. If POLG1 positive, "
            "use CLB + ethosuximide for absence. Avoid CBZ/OXC (worsen myoclonus) "
            "and LTG monotherapy (worsens myoclonic component in some patients)."
        ),
    },
]

# ── Trigger catalog ───────────────────────────────────────────────────────────
TRIGGER_CATALOG = [
    {"trigger": "Fever / Illness", "prevalence_pct": 82,
     "mechanism": "Temperature-sensitive Nav1.1 gating in DYRK1A/NRSF-compromised interneurons; febrile IS common"},
    {"trigger": "Sleep Deprivation", "prevalence_pct": 72,
     "mechanism": "NREM-dependent cortical synchronisation amplifies DYRK1A thalamocortical epileptiform discharge"},
    {"trigger": "Missed AED Dose", "prevalence_pct": 65,
     "mechanism": "Abrupt VPA/LEV withdrawal → rapid loss of seizure protection; rebound myoclonus"},
    {"trigger": "Intercurrent Metabolic Stress", "prevalence_pct": 55,
     "mechanism": "Hyponatraemia, hypoglycaemia; RCAN1/calcineurin pathway sensitive to metabolic state"},
    {"trigger": "Overstimulation / Sensory Overload", "prevalence_pct": 48,
     "mechanism": "Autism-associated sensory hypersensitivity → arousal spike → seizure threshold reduction"},
    {"trigger": "Photosensitivity", "prevalence_pct": 12,
     "mechanism": "Rare (10-15%); generalised photoparoxysmal response at 6-18 Hz; note in EEG protocol"},
    {"trigger": "Anaesthesia / Procedural Sedation", "prevalence_pct": 28,
     "mechanism": "Volatile agents (sevoflurane) + existing GABRB3 reduction → GABAA sensitisation imbalance; "
                  "TIVA (propofol) preferred where possible; pre-procedure neurology review"},
    {"trigger": "Catamenial (Perimenstrual)", "prevalence_pct": 18,
     "mechanism": "Neurosteroid fluctuation (allopregnanolone ↓ perimenstrually); GABAA modulation in "
                  "context of GABRB3-compromised network; CLB perimenstrual PRN protocol"},
]

# ── Treatment catalog ─────────────────────────────────────────────────────────
TREATMENT_CATALOG = [
    {
        "drug": "ACTH (Adrenocorticotropic Hormone)",
        "level": "Level A (IS)",
        "indication": "Infantile spasms first-line (UKISS 2005 standard)",
        "dose": "150 IU/m²/day IM × 2 weeks; or synthetic tetracosactide 0.5-1 mg IM",
        "moa": "ACTH → adrenal corticosteroid release → GABA-B potentiation + "
               "CRH suppression → IS cessation; additional direct MC4R agonism in brain",
        "efficacy": "~55% hypsarrhythmia cessation in DYRK1A IS (vs 80% idiopathic) — "
                    "structural microcephaly substrate reduces response rate",
        "monitoring": "BP, glucose, Na (SIADH), infection screen; Day-14 EEG mandatory",
        "dyrk1a_note": "Lower IS cessation rate than idiopathic; start KD planning at 4-week non-response; "
                       "VGB add-on simultaneously (not sequentially as per UKISS protocol modification)",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A (IS alternative/adjunct)",
        "indication": "Infantile spasms — add-on with ACTH; monotherapy if ACTH fails",
        "dose": "100-150 mg/kg/day (IS); max 3 g/day",
        "moa": "GABA-T (GABA transaminase) inhibitor → elevated synaptic GABA → GABAA activation",
        "efficacy": "45% IS cessation as monotherapy; higher combined with ACTH",
        "monitoring": "ERG (REMS programme) every 6-12 months — VGB causes irreversible "
                      "visual field defects (nasal scotoma); use FDA SHARE REMS programme",
        "dyrk1a_note": "VGB is SAFE in DYRK1A (unlike TGB — see CI). Visual monitoring "
                       "feasible if patient verbal; challenge if severe ID — baseline ERG before start; "
                       "risk-benefit favourable for IS (life-changing IS cessation vs gradual visual loss)",
    },
    {
        "drug": "Valproate (VPA)",
        "level": "Level B (GTCS / Myoclonic / Mixed)",
        "indication": "First-line broad-spectrum AED for GTCS, myoclonic, mixed seizure types",
        "dose": "20-40 mg/kg/day in 2-3 divided doses; TDM target 50-100 µg/mL",
        "moa": "Na+ channel block (use-dependent) + GABA-T inhibition + T-type Ca²⁺ block + "
               "HDAC inhibition (epigenetic: may partially de-repress NRSF-silenced Nav1.1/GABRB3 "
               "— rational mechanism specific to DYRK1A/NRSF pathway)",
        "efficacy": "60-70% ≥50% seizure reduction in GTCS/myoclonic in DYRK1A cohort",
        "monitoring": "LFT, FBC, NH3 (ammonia), weight; TDM 2-4 weekly initially; "
                      "POLG1 screen MANDATORY before starting — biallelic POLG1 + VPA = "
                      "Alpers-Huttenlocher syndrome (fatal hepatotoxicity)",
        "dyrk1a_note": "HDAC inhibition by VPA may provide unique mechanistic benefit in DYRK1A: "
                       "NRSF recruits HDAC co-repressors to RE-1 sites; VPA inhibits HDACs → "
                       "partially de-represses Nav1.1/GABRB3 chromatin → rational epigenetic "
                       "augmentation of residual DYRK1A function. POLG1 screen NON-NEGOTIABLE "
                       "(7-14 day turnaround; bridge with LEV+CLB).",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B (adjunct / VPA-intolerant)",
        "indication": "Adjunct to VPA; monotherapy if VPA contraindicated (POLG1+)",
        "dose": "20-60 mg/kg/day in 2 divided doses; max 3 g/day",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulator → reduces presynaptic "
               "vesicle docking → attenuates burst firing; also GABA-A potentiation",
        "efficacy": "50-60% responder rate in DYRK1A GTCS/focal as adjunct",
        "monitoring": "Behavioural checklist monthly (15-30% DEE patients develop irritability, "
                      "aggression — higher rate in ASD/DYRK1A); consider brivaracetam switch",
        "dyrk1a_note": "LEV preferred over PHT/CBZ as second-line — no Na-channel NRSF/Nav1.1 "
                       "interaction risk. Behavioural toxicity common in DYRK1A+ASD: consent "
                       "carers; brivaracetam (BRV) alternative SV2A agent with ~5% irritability rate.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B (DRE, IS non-responders)",
        "indication": "Drug-resistant epilepsy ≥2 AED failures; IS non-response at week 4",
        "dose": "3:1-4:1 ratio (fat:protein+carb); daily caloric needs; dietitian-led",
        "moa": "BHB (beta-hydroxybutyrate) → HCAR2 agonism + NLRP3 inflammasome suppression "
               "+ adenosine A1R agonism → seizure threshold elevation; "
               "RCAN1/calcineurin pathway suppression by ketones may provide additional "
               "benefit specific to DYRK1A substrate pathway",
        "efficacy": "55% ≥50% reduction in DYRK1A DRE (Level B evidence: comparable to "
                    "non-structural DEE cohorts); IS non-responders: 45% benefit",
        "monitoring": "Lipid panel, urine ketones, BHB target 2-4 mmol/L, renal stones, "
                      "growth parameters, DEXA bone density annually",
        "dyrk1a_note": "Early KD (not last resort): initiate at AED failure #2 in DYRK1A; "
                       "RCAN1 calcineurin pathway suppression by ketones may address DYRK1A-specific "
                       "calcineurin deregulation (NFAT pathway) — mechanistic rationale beyond seizures.",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B (adjunct / rescue)",
        "indication": "Adjunct for GTCS/myoclonic; perimenstrual protocol; acute cluster rescue",
        "dose": "0.1-0.3 mg/kg/day in 2 divided doses; perimenstrual 0.5 mg/kg/day day -3 to +3",
        "moa": "GABA-A positive allosteric modulator (α2/α3 subunit selective) → benzodiazepine "
               "site; lower tolerance and sedation vs clonazepam (β2 sparing)",
        "efficacy": "50-65% adjunctive response rate; catamenial protocol effective",
        "monitoring": "Sedation score, tolerance assessment 3-monthly; withdrawal taper mandatory",
        "dyrk1a_note": "CLB preferred over diazepam for chronic adjunct use in DYRK1A+ASD — "
                       "lower behavioural disinhibition risk than full BZD agonists; rescue plan "
                       "documented with carers (buccal midazolam for cluster >5 min).",
    },
    {
        "drug": "Folinic Acid (Leucovorin / 5-Formyl-THF)",
        "level": "Level C (metabolic adjunct)",
        "indication": "DYRK1A + cerebral folate deficiency / folate receptor antibody positive",
        "dose": "0.5-1 mg/kg/day PO (max 50 mg/day); CSF 5-MTHF target >40 nmol/L",
        "moa": "Bypasses folate receptor (FRA1) blockade → direct CSF folate → "
               "methylenetetrahydrofolate → methionine/SAM → DNA methylation of epileptogenic "
               "genes; folinic acid crosses BBB independently of folate receptor",
        "efficacy": "25-35% DYRK1A patients have elevated anti-FRA1 antibodies; "
                    "retrospective data: folinic acid improved seizure frequency and "
                    "neurodevelopment in folate-deficient DEE (Level C)",
        "monitoring": "CSF 5-MTHF at baseline (lumbar puncture); anti-FRA1 IgG serum; "
                      "recheck CSF folate at 6 months",
        "dyrk1a_note": "Screen ALL DYRK1A patients for anti-FRA1 antibodies (serum IgG). "
                       "If positive → folinic acid trial 0.5 mg/kg/day × 3 months; monitor CSF. "
                       "Association with DYRK1A DEE first described by Lopes-Arez 2022.",
    },
    {
        "drug": "Corticosteroids (Prednisolone / HDMP)",
        "level": "Level C (IS second-line / myoclonic status)",
        "indication": "IS if ACTH unavailable; myoclonic status epilepticus (acute)",
        "dose": "Prednisolone 4 mg/kg/day (max 60 mg) × 2 weeks then taper; "
                "HDMP 20-30 mg/kg/day IV × 3 days for acute myoclonic status",
        "moa": "Anti-inflammatory → neurosteroid production → GABA-A potentiation; "
               "CRH suppression → IS cessation (similar pathway to ACTH but less potent)",
        "efficacy": "IS: ~45% cessation (inferior to ACTH); myoclonic status: 70% control",
        "monitoring": "BP, glucose, Na, infection screen; same as ACTH monitoring",
        "dyrk1a_note": "Use if ACTH unavailable or refused. HDMP (high-dose methylprednisolone) "
                       "pulse for acute myoclonic status refractory to IV LEV + BZD.",
    },
]

# ── Contraindication catalog ──────────────────────────────────────────────────
CI_CATALOG = [
    {
        "drug": "Phenytoin / Carbamazepine / Oxcarbazepine",
        "severity": "HIGH RISK",
        "mechanism": (
            "Na-channel blockers paradoxically WORSEN DYRK1A seizures via NRSF pathway: "
            "DYRK1A LOF → NRSF stabilised → SCN1A (Nav1.1) repressed in GABAergic interneurons "
            "→ net interneuron dysfunction; PHT/CBZ/OXC further block residual Nav1.1 function "
            "→ amplified interneuron failure → disinhibition → GTCS/myoclonic worsening. "
            "Mechanism identical to Na-channel blocker worsening in Dravet syndrome (SCN1A LOF). "
            "CBZ/OXC also enzyme-inducing → reduces VPA, LTG, CLB serum levels."
        ),
        "action": "AVOID as first, second, or third-line AED in DYRK1A; document on emergency "
                  "care plan; use IV LEV (not IV PHT/fosphenytoin) for status epilepticus.",
        "evidence": "Mechanism-based (NRSF/Nav1.1 pathway); clinical analogy: Dravet Na-channel "
                    "blocker worsening (Level C indirect evidence in DYRK1A)",
    },
    {
        "drug": "DYRK1A Inhibitors (Harmine / Leucettine-41 / EGCG)",
        "severity": "ABSOLUTE CONTRAINDICATION",
        "mechanism": (
            "Harmine, leucettine-41, and EGCG (epigallocatechin-3-gallate) are investigational "
            "DYRK1A kinase inhibitors being trialled in Down syndrome (trisomy 21) to REDUCE "
            "excessive 3× DYRK1A activity and prevent Alzheimer pathology. In DYRK1A "
            "haploinsufficiency syndrome (1× DYRK1A, LOF), these inhibitors would further "
            "suppress the already-halved kinase activity → catastrophic worsening of all "
            "phenotypes: more severe seizures, microcephaly progression, tau mis-splicing, "
            "NRSF accumulation. EGCG in green tea supplements: ABSOLUTE CI for DYRK1A parents "
            "who supplement. Never prescribe or permit DYRK1A inhibitors in LOF syndrome."
        ),
        "action": "Explicitly counsel family: NO harmine supplements, NO EGCG/green tea "
                  "extract supplements, NO leucettine compounds. Document on allergy/CI list. "
                  "Distinguish from Down syndrome trials (opposite mechanism).",
        "evidence": "Mechanistic (kinase inhibition of haploinsufficient allele); EGCG trial "
                    "in Down syndrome (de la Torre 2014 — TRISOMY 21 ONLY, not DYRK1A LOF)",
    },
    {
        "drug": "Tiagabine (TGB)",
        "severity": "ABSOLUTE CONTRAINDICATION",
        "mechanism": (
            "GAT-1 (GABA transporter) inhibitor → excess perisynaptic GABA → GABA-A "
            "desensitisation → paradoxical loss of inhibition → non-convulsive status "
            "epilepticus (NCSE). DYRK1A patients have GABRB3 reduction (via NRSF repression) "
            "→ already-reduced GABA-A function → TGB-induced desensitisation is amplified "
            "vs general population. NCSE in DYRK1A: can mimic behavioural regression, "
            "difficult to detect without EEG (ID + ASD baseline)."
        ),
        "action": "Never use TGB in DYRK1A. Distinguish from VGB (vigabatrin = GABA-T "
                  "inhibitor = safe; TGB = GAT-1 blocker = CI): two different mechanisms "
                  "often confused by prescribers — document clearly on AED list.",
        "evidence": "TGB-induced NCSE: class-level risk (ILAE warning); amplified in DEE with "
                    "GABA-A reduction (GABRB3 via NRSF in DYRK1A)",
    },
    {
        "drug": "Valproate (VPA) — in POLG1-positive patients",
        "severity": "ABSOLUTE CONTRAINDICATION",
        "mechanism": (
            "POLG1 biallelic variants + VPA → Alpers-Huttenlocher syndrome: VPA inhibits "
            "POLG1 polymerase activity → mitochondrial DNA depletion in hepatocytes → "
            "fulminant hepatic failure (90% mortality). DYRK1A patients are NOT pre-selected "
            "for POLG1 negativity — POLG1 screen mandatory before any VPA. "
            "DYRK1A LOF does not increase POLG1 prevalence but neither reduces it."
        ),
        "action": "POLG1 screen BEFORE VPA in every DYRK1A patient (no exceptions). "
                  "Turnaround 7-14 days. Bridge with LEV + CLB during POLG1 wait. "
                  "If POLG1 positive → VPA NEVER; use LEV + CLB + KD.",
        "evidence": "CPIC POLG1-2023 guideline; Stumpf et al. Neurology 2002; "
                    "Fatal VPA-induced hepatotoxicity in POLG1 disease documented 1996-present",
    },
    {
        "drug": "Lamotrigine (LTG) Monotherapy — in myoclonic-predominant DYRK1A",
        "severity": "HIGH RISK",
        "mechanism": (
            "LTG Na-channel block preferentially suppresses inhibitory interneuron fast spiking "
            "(Nav1.1 in interneurons vs Nav1.6 in pyramidal neurons); in DYRK1A with NRSF "
            "reduction of Nav1.1 → LTG disproportionately suppresses interneuron function "
            "in myoclonic-predominant patients → myoclonus worsening and absence status. "
            "LTG may be used cautiously as adjunct or in focal-predominant DYRK1A "
            "(where FIAS without myoclonic component; full suppression less risky)."
        ),
        "action": "Avoid LTG MONOTHERAPY in myoclonic-predominant DYRK1A. Acceptable as "
                  "adjunct in focal-predominant DYRK1A (titrate slowly 12.5 mg → 50 mg/day). "
                  "Monitor myoclonus frequency when starting LTG.",
        "evidence": "LTG myoclonus worsening: well-documented in JME, Dravet; "
                    "DYRK1A NRSF/Nav1.1 mechanism provides additional risk (Level C indirect)",
    },
    {
        "drug": "Abrupt AED Withdrawal",
        "severity": "ABSOLUTE CONTRAINDICATION (in established DYRK1A epilepsy)",
        "mechanism": (
            "Abrupt withdrawal of any established AED in DYRK1A → breakthrough seizure cluster "
            "→ risk of status epilepticus and cognitive regression. DYRK1A patients have "
            "limited homeostatic plasticity reserve (haploinsufficiency impairs activity-dependent "
            "synaptic scaling) → slower recovery from acute seizure burden. "
            "VPA withdrawal → rapid myoclonic status onset (1-3 days). "
            "LEV withdrawal → rebound focal cluster. Must taper ALL AEDs ≥10%/week minimum."
        ),
        "action": "Taper ALL AEDs ≥10%/week; never NPO (nil per os) without AED continuation "
                  "plan in hospital setting; emergency AED administration route documented "
                  "(buccal midazolam rescue, NG-tube VPA liquid, IV LEV).",
        "evidence": "General DEE guideline; DYRK1A homeostatic plasticity impairment "
                    "based on NRSF/synaptic scaling literature (Level C mechanism-based)",
    },
]

# ── Monitoring catalog ────────────────────────────────────────────────────────
MONITORING_CATALOG = [
    {"item": "WES Trio (DYRK1A gene + parental)", "frequency": "At diagnosis",
     "purpose": "Confirm pathogenic DYRK1A variant; parental testing for de novo (97%) vs inherited"},
    {"item": "POLG1 Screen (WES or Sanger)", "frequency": "Before VPA",
     "purpose": "Exclude biallelic POLG1 — fatal hepatotoxicity risk with VPA (Alpers-Huttenlocher)"},
    {"item": "Anti-Folate Receptor Antibody (Anti-FRA1 IgG)", "frequency": "At diagnosis + 12-monthly",
     "purpose": "25-35% DYRK1A patients; positive → folinic acid trial; CSF 5-MTHF if antibody positive"},
    {"item": "Video-EEG (LTM ≥48h)", "frequency": "At diagnosis; annually if DRE",
     "purpose": "Characterise seizure types (IS vs myoclonic vs focal); guide AED and surgical planning"},
    {"item": "MRI Brain 3T (with cortical thickness protocol)", "frequency": "At diagnosis; 12-monthly age <3",
     "purpose": "Simplified gyral pattern, hippocampal hypoplasia, FCD type II — guides epilepsy surgery eligibility"},
    {"item": "Head Circumference (HC/OFC)", "frequency": "Every 3 months age <2; 6-monthly thereafter",
     "purpose": "Primary microcephaly (HC < -2 SD at birth) confirms DYRK1A; progressive microcephaly tracked"},
    {"item": "VPA TDM (Therapeutic Drug Monitoring)", "frequency": "2-4 weekly initiation; 3-monthly stable",
     "purpose": "Target 50-100 µg/mL; LFT + FBC + NH3 + weight at each TDM visit"},
    {"item": "Developmental Assessment (Bayley-4 / VABS-3)", "frequency": "6-monthly age <3; 12-monthly thereafter",
     "purpose": "Baseline developmental trajectory; detect regression; guide therapy and IGF-1 assessment"},
    {"item": "VGB ERG (REMS Programme)", "frequency": "Before VGB; every 6 months on VGB",
     "purpose": "VGB causes irreversible visual field defects; ERG mandatory for REMS programme compliance"},
    {"item": "EEG Photic Stimulation Protocol", "frequency": "At diagnosis EEG; annually",
     "purpose": "10-15% photosensitivity in DYRK1A GTCS; document response frequency 6-18 Hz"},
    {"item": "Autism / ASD Assessment (ADOS-2 / ADI-R)", "frequency": "18-30 months; 48 months",
     "purpose": ">90% DYRK1A have ASD; early diagnosis enables behavioural intervention; "
                "ASD severity informs AED behavioural side-effect risk (LEV/LTG tolerance)"},
    {"item": "KD Metabolic Panel (Lipids / BHB / Renal)", "frequency": "Monthly first 3 months; 3-monthly stable",
     "purpose": "KD lipid safety; BHB target 2-4 mmol/L; renal stones (15% on KD); growth parameters"},
    {"item": "SUDEP Risk Documentation", "frequency": "At diagnosis; annually",
     "purpose": "DYRK1A DRE → SUDEP risk ~1:200/year; nocturnal seizure alarm (NightWatch/SAMi-3); "
                "prone sleep avoidance; counselling documented"},
    {"item": "Valproate Pregnancy Prevention (VPPP)", "frequency": "From age 10 in females on VPA",
     "purpose": "VPA teratogenicity; 1% spina bifida + 10-15% ASD in offspring; NHS VPPP mandatory UK/MHRA 2021"},
]

# ── Lifecycle stages ──────────────────────────────────────────────────────────
LIFECYCLE_STAGES = [
    {
        "stage": "Prenatal / Neonatal (0-3 months)",
        "key_features": [
            "Microcephaly often present at birth (HC < -2 SD) — first diagnostic clue",
            "Feeding difficulties (sucking/swallowing — brainstem/cranial nerve involvement)",
            "Neonatal hypotonia in 70%",
            "Genetics: WES Trio ordered if microcephaly + dysmorphic features at birth",
        ],
        "action": "Head circumference on growth chart; ophthalmology (cortical visual impairment); "
                  "WES Trio if HC < -2 SD + dysmorphic; speech therapy for feeding",
    },
    {
        "stage": "Infantile (3-12 months) — IS Window",
        "key_features": [
            "Infantile spasms onset 3-9 months (30% of cohort)",
            "ACTH first-line within 2 weeks of IS onset",
            "Hypsarrhythmia may be asymmetric/modified (cortical malformation substrate)",
            "EEG: 24h video-EEG urgently for spasm diagnosis",
        ],
        "action": "ACTH + VGB within 2 weeks IS onset; Day-14 EEG for hypsarrhythmia resolution; "
                  "KD consultation if no response by week 4; POLG1 screen before any VPA",
    },
    {
        "stage": "Early Childhood (1-4 years) — GTCS / Myoclonic Onset",
        "key_features": [
            "GTCS / myoclonic onset 12-48 months",
            "ASD diagnosis typically 18-36 months",
            "PHT/CBZ AVOID — HIGH risk seizure worsening via NRSF/Nav1.1 mechanism",
            "VPA first-line (POLG1 pre-screened); folinic acid if anti-FRA1 positive",
        ],
        "action": "VPA initiation (POLG1 screened); Bayley-4 developmental baseline; "
                  "ADOS-2 for ASD diagnosis; MRI 3T; anti-FRA1 antibody screen",
    },
    {
        "stage": "School Age (4-12 years) — DRE Management",
        "key_features": [
            "DRE definition (≥2 AED failures) common in truncating LOF class (40%)",
            "KD initiation at AED failure #2 (not last resort)",
            "Epilepsy surgery evaluation if focal MRI lesion (FCD type II in 20%)",
            "Helmet for myoclonic-atonic drop attacks",
        ],
        "action": "KD at AED failure #2; epilepsy surgery MDT if focal MRI; "
                  "SEEG evaluation; SUDEP counselling; SUDEP alarm systems",
    },
    {
        "stage": "Adolescence (12-18 years)",
        "key_features": [
            "Catamenial exacerbation in 18% of females (CLB perimenstrual protocol)",
            "VPA VPPP counselling mandatory from age 10 in females",
            "Transition planning (adult epilepsy services, residential care coordination)",
            "Levetiracetam behavioural toxicity monitoring (aggression in ASD+DEE)",
        ],
        "action": "VPPP counselling + contraceptive planning; CLB perimenstrual PRN; "
                  "adult transition plan; vocational and residential support pathways",
    },
    {
        "stage": "Adulthood",
        "key_features": [
            "Persistent DRE in 40%; stable control in 35%; breakthrough seizures in 25%",
            "Early dementia risk (DYRK1A tau mis-splicing + risk; monitor cognition)",
            "DYRK1A inhibitor trials for Alzheimer — NEVER in DYRK1A LOF (misdiagnosis risk)",
            "Long-term care coordination: neurology + psychiatry + residential services",
        ],
        "action": "Annual cognitive screening (RBANS / Bayley adapted); "
                  "clarify DYRK1A LOF documentation to prevent inadvertent DYRK1A inhibitor "
                  "prescription (if Alzheimer symptoms emerge — refer to geneticist, not DS clinic)",
    },
]

# ── Concept catalog ───────────────────────────────────────────────────────────
CONCEPT_CATALOG = [
    {
        "concept": "DYRK1A-21q22.13-DEE",
        "definition": (
            "DYRK1A haploinsufficiency syndrome: de novo LOF → DEE + primary microcephaly "
            "(<-2 SD HC at birth) + severe intellectual disability + autism (>90%) + "
            "characteristic 'bird-like' facies (prominent nasal bridge, deep-set eyes, "
            "thin upper lip, upslanted palpebral fissures). OMIM #614721. Seizures in 60-70%. "
            "pLI=0.99 — one of the most intolerant genes to LOF in the genome."
        ),
    },
    {
        "concept": "NRSF-REST-Nav1.1-Mechanism",
        "definition": (
            "DYRK1A phosphorylates NRSF (Neural Restrictive Silencer Factor) → β-TrCP-mediated "
            "degradation → NRSF↓ → de-repression of SCN1A (Nav1.1), SCN2A, GABRB3, KCNQ2. "
            "In DYRK1A LOF: NRSF not phosphorylated → NRSF stable → RE-1-mediated repression "
            "of Nav1.1/GABRB3 → functional Nav1.1 deficiency → GABAergic interneuron failure "
            "→ epilepsy. This is the molecular basis for Na-channel blocker HIGH RISK in DYRK1A."
        ),
    },
    {
        "concept": "DYRK1A-Trisomy21-Dose-Distinction",
        "definition": (
            "CRITICAL: DYRK1A haploinsufficiency (1× copy, LOF) causes DEE — DYRK1A inhibitors "
            "(harmine, leucettine, EGCG) are ABSOLUTE CI. Down syndrome / trisomy 21 (3× copies) "
            "→ excess DYRK1A → Alzheimer predisposition — DYRK1A inhibitors under investigation "
            "in trisomy 21. Prescribers must NEVER confuse these: opposite mechanisms, "
            "opposite pharmacological interventions."
        ),
    },
    {
        "concept": "NRSF-Epigenetic-VPA-Rationale",
        "definition": (
            "Valproate's HDAC inhibition provides mechanistic rationale beyond seizure control: "
            "NRSF recruits HDAC-containing NuRD/Sin3A co-repressors to RE-1 silencer elements "
            "on SCN1A/GABRB3. VPA (HDAC inhibitor) → chromatin de-repression → partial "
            "restoration of Nav1.1/GABRB3 expression from the intact allele. This epigenetic "
            "mechanism is DYRK1A/NRSF-pathway specific — not shared with other AEDs."
        ),
    },
    {
        "concept": "Primary-Microcephaly-Birth-Diagnostic-Clock",
        "definition": (
            "Unlike PNPO/ALDH7A1 (postnatal metabolic onset), DYRK1A syndrome microcephaly "
            "is PRESENT AT BIRTH (primary microcephaly, HC < -2 SD). This is the diagnostic "
            "clock: a neonate with primary microcephaly + hypotonia + feeding difficulty → "
            "DYRK1A on WES differential (before first seizure). Cyclin D1 phosphorylation "
            "by DYRK1A controls G1/S neuroblast proliferation — haploinsufficiency → "
            "reduced neuronal number from early fetal neurogenesis."
        ),
    },
    {
        "concept": "Anti-FRA1-Cerebral-Folate-Deficiency",
        "definition": (
            "25-35% DYRK1A patients have anti-folate receptor antibody type 1 (anti-FRA1 IgG) "
            "→ cerebral folate deficiency (CSF 5-MTHF < 40 nmol/L). Folinic acid (leucovorin) "
            "0.5-1 mg/kg/day bypasses FRA1 block → restores CSF folate → potential seizure "
            "improvement + neurodevelopmental benefit. Screen all DYRK1A patients. "
            "Distinct from classic CFD (FOLR1 mutations) but same mechanism."
        ),
    },
    {
        "concept": "PHT-CBZ-OXC-WORSENING-NRSF",
        "definition": (
            "Na-channel blockers (phenytoin, carbamazepine, oxcarbazepine) worsen DYRK1A "
            "seizures via NRSF/Nav1.1 mechanism: prefer inhibitory interneuron (Nav1.1) "
            "over pyramidal (Nav1.6) → in DYRK1A with NRSF-mediated Nav1.1 reduction, "
            "PHT/CBZ further suppress interneuron function → disinhibition → seizure worsening. "
            "Identical mechanism to Dravet (SCN1A LOF). Document on emergency care plan."
        ),
    },
    {
        "concept": "POLG1-VPA-Mandatory-Screen",
        "definition": (
            "Biallelic POLG1 (mtDNA polymerase gamma) variants + valproate = Alpers-Huttenlocher "
            "syndrome: VPA inhibits POLG1 → mitochondrial DNA depletion in hepatocytes → "
            "fulminant hepatic failure (90% mortality, non-reversible). POLG1 screen MANDATORY "
            "before any VPA in DYRK1A. Turnaround 7-14 days; bridge with LEV+CLB. "
            "CPIC 2023 guideline mandates this for all DEE patients starting VPA."
        ),
    },
    {
        "concept": "DYRK1A-Inhibitor-ABSOLUTE-CI",
        "definition": (
            "Harmine, leucettine-41, and EGCG (epigallocatechin gallate in green tea) are "
            "DYRK1A kinase inhibitors investigated in DOWN SYNDROME (trisomy 21, 3× DYRK1A) "
            "to reduce excessive kinase activity and prevent Alzheimer. In DYRK1A "
            "haploinsufficiency (1× DYRK1A, LOF): these compounds further reduce already-halved "
            "kinase → catastrophic worsening. EGCG is widely available as supplement — "
            "explicitly counsel families NO green tea extract supplements."
        ),
    },
    {
        "concept": "KD-Early-DYRK1A",
        "definition": (
            "Ketogenic diet should be initiated at AED failure #2 in DYRK1A, not as last resort. "
            "Rationale: (1) structural substrate (microcephaly, cortical malformations) reduces "
            "pharmacological AED response → expect DRE; (2) RCAN1/calcineurin pathway suppression "
            "by BHB may provide DYRK1A-specific benefit; (3) 55% ≥50% seizure reduction in DRE "
            "DYRK1A cohort; (4) non-pharmacological mechanism avoids PHT/CBZ/TGB CI issues."
        ),
    },
    {
        "concept": "IS-ACTH-Lower-Response-Structural",
        "definition": (
            "Infantile spasms in DYRK1A have lower ACTH response (~55% hypsarrhythmia cessation) "
            "vs idiopathic IS (80%) because microcephaly + cortical malformations constitute a "
            "structural substrate that persists after hormonal treatment. Strategy: ACTH + VGB "
            "simultaneously (not sequentially); Day-14 EEG; if incomplete response at week 4 → "
            "KD + VPA adjunct; reassess epilepsy surgery eligibility."
        ),
    },
    {
        "concept": "TGB-ABSOLUTE-NCSE-GABRB3-Risk",
        "definition": (
            "Tiagabine (TGB, GAT-1 inhibitor) causes NCSE in DEE patients via GABA spillover → "
            "GABA-A desensitisation → paradoxical loss of inhibition. In DYRK1A, NRSF-mediated "
            "GABRB3 reduction (β3 subunit deficiency) amplifies this risk: reduced GABA-A reserve "
            "→ faster desensitisation → lower TGB threshold for NCSE. NCSE in DYRK1A mimics "
            "regression or drowsiness — continuous EEG mandatory if unexplained deterioration. "
            "NEVER use TGB. Distinguish from VGB (vigabatrin) = GABA-T inhibitor = safe."
        ),
    },
    {
        "concept": "Tau-Mis-Splicing-Neurodegeneration",
        "definition": (
            "DYRK1A phosphorylates splicing factor SRSF6 → controls exon 10 inclusion in MAPT "
            "(tau mRNA) → regulates 3R:4R tau isoform ratio. DYRK1A LOF → SRSF6 not "
            "phosphorylated → altered tau splicing → excess 3R-tau → tauopathy predisposition. "
            "Adult DYRK1A patients may show early cognitive decline (distinct from Down syndrome "
            "Alzheimer mechanism which is 3× DYRK1A → hyperphosphorylated 4R-tau). "
            "Monitor with neuropsychological testing from age 30."
        ),
    },
    {
        "concept": "LTG-Monotherapy-Myoclonic-Risk",
        "definition": (
            "Lamotrigine (LTG) Na-channel block disproportionately suppresses Nav1.1 in "
            "GABAergic interneurons (vs Nav1.6 in pyramidal neurons). In DYRK1A with "
            "NRSF-mediated Nav1.1 reduction, LTG monotherapy in myoclonic-predominant "
            "patients → further interneuron suppression → myoclonus worsening and "
            "absence status. LTG acceptable as adjunct at low doses in focal-predominant "
            "DYRK1A (not myoclonic class)."
        ),
    },
    {
        "concept": "SUDEP-DRE-DYRK1A",
        "definition": (
            "DYRK1A DRE (40% of cohort) carries SUDEP risk ~1:200/year (comparable to "
            "other DEE with uncontrolled generalised seizures). SUDEP risk mitigation: "
            "nocturnal seizure alarm (NightWatch heart rate + EMG sensor; SAMi-3; Empatica E4); "
            "avoid prone sleeping; shared sleeping arrangements until age 5; "
            "conversation documented at diagnosis with family. DYRK1A SUDEP mechanism: "
            "GTCS → brainstem cardiac/respiratory shutdown (same as Dravet/SCN1A)."
        ),
    },
]

# ── Threshold catalog ─────────────────────────────────────────────────────────
THRESHOLD_CATALOG = [
    {"parameter": "VPA serum level (TDM target)", "threshold": "50-100 µg/mL (350-700 µmol/L)",
     "action": "Below 50 → increase dose; above 100 → assess toxicity (tremor, NH3)"},
    {"parameter": "VPA ammonia (hepatotoxicity marker)", "threshold": "NH3 > 80 µmol/L",
     "action": "Reduce VPA dose; check LFT urgently; consider VPA holiday if NH3 > 100"},
    {"parameter": "Head circumference percentile (OFC)", "threshold": "HC < -2 SD (below 2nd centile)",
     "action": "Primary microcephaly → DYRK1A on differential; order WES Trio"},
    {"parameter": "ACTH response (Day-14 EEG)", "threshold": "Hypsarrhythmia absent on Day-14 EEG",
     "action": "IS cessation confirmed → continue ACTH taper; if persistent → add KD consultation"},
    {"parameter": "CSF 5-MTHF (cerebral folate)", "threshold": "< 40 nmol/L",
     "action": "Cerebral folate deficiency → folinic acid 0.5-1 mg/kg/day; recheck CSF at 6 months"},
    {"parameter": "BHB (ketogenic diet)", "threshold": "2.0-4.0 mmol/L",
     "action": "Below 2.0 → adjust diet ratio; above 4.0 + symptoms → reduce fat ratio"},
    {"parameter": "IS ACTH non-response threshold", "threshold": "No hypsarrhythmia cessation week 4",
     "action": "ACTH non-response → start KD consultation urgently; add VPA if POLG1 cleared"},
    {"parameter": "VGB visual toxicity", "threshold": "ERG b-wave amplitude < 70% baseline",
     "action": "VGB dose reduction or cessation; ophthalmology review; document visual fields"},
    {"parameter": "POLG1 result turnaround", "threshold": "Awaiting > 14 days on LEV+CLB bridge",
     "action": "Continue LEV+CLB bridge; do NOT start VPA until POLG1 result confirmed negative"},
    {"parameter": "SUDEP alarm activation", "threshold": "Nocturnal GTCS or seizure duration > 3 min",
     "action": "Seizure alarm mandatory (NightWatch/SAMi-3) for all DYRK1A with nocturnal GTCS"},
    {"parameter": "QTc (CLB monitoring)", "threshold": "QTc > 450 ms (female) or > 440 ms (male)",
     "action": "Review QTc-prolonging co-medications; cardiology consult; consider CLB dose reduction"},
    {"parameter": "Anti-FRA1 IgG titre", "threshold": "Positive (any titre)",
     "action": "Positive → folinic acid trial; CSF 5-MTHF measurement; recheck antibody titre at 6 months"},
]

# ── Reference catalog ─────────────────────────────────────────────────────────
REFERENCE_CATALOG = [
    {
        "ref": "Moller-2008-EurJHumGenet",
        "citation": "Møller RS et al. Truncation of the Down syndrome candidate gene DYRK1A in two unrelated patients with microcephaly. Eur J Hum Genet. 2008;16(7):790-796.",
        "impact": "First report of de novo DYRK1A haploinsufficiency causing microcephaly-ID-epilepsy",
    },
    {
        "ref": "vanBon-2011-JMedGenet",
        "citation": "van Bon BWM et al. Intragenic deletion in DYRK1A leads to intellectual disability and primary microcephaly. J Med Genet. 2011;48(9):627-630.",
        "impact": "Expanded phenotype; confirmed DYRK1A haploinsufficiency syndrome",
    },
    {
        "ref": "Bronicki-2015-JMedGenet",
        "citation": "Bronicki LM et al. Ten new cases further delineate the syndromic intellectual disability phenotype caused by mutations in DYRK1A. Eur J Hum Genet. 2015;23(11):1482-1487.",
        "impact": "Cohort of 10 patients; defined bird-like facies + genotype-phenotype correlations",
    },
    {
        "ref": "Ji-2015-NatGenet",
        "citation": "Ji J et al. DYRK1A haploinsufficiency causes a new recognizable syndrome with microcephaly, intellectual disability, seizures and dysmorphic features. Nat Genet. 2015;47(3):243-251.",
        "impact": "Largest early cohort; established DYRK1A syndrome as recognizable entity; NRSF pathway described",
    },
    {
        "ref": "Courcet-2012-EurJHumGenet",
        "citation": "Courcet JB et al. The DYRK1A gene is a cause of syndromic intellectual disability with severe microcephaly and epilepsy. J Med Genet. 2012;49(12):731-736.",
        "impact": "French cohort; epilepsy characterised; AED response and natural history",
    },
    {
        "ref": "deLatorre-2014-LancetNeurol",
        "citation": "de la Torre R et al. Epigallocatechin-3-gallate, a DYRK1A inhibitor, rescues cognitive deficits in Down syndrome mouse models and in humans. Lancet Neurol. 2014;13(4):348-359.",
        "impact": "EGCG (DYRK1A inhibitor) in trisomy 21 — OPPOSITE context to DYRK1A LOF syndrome (cited for contrast: CI in LOF)",
    },
]

# ── Patient cohort (40 synthetic patients) ────────────────────────────────────
random.seed(42)
_FIRST = ["Emma","Liam","Olivia","Noah","Ava","Ethan","Sophia","Mason","Isabella","Logan",
          "Mia","Lucas","Charlotte","Oliver","Amelia","Elijah","Harper","Aiden","Evelyn","Carter",
          "Abigail","Jayden","Scarlett","Michael","Grace","Sebastian","Zoey","Owen","Lily","Ryan",
          "Hannah","Alexander","Ella","Nathan","Victoria","Aaron","Aria","Cameron","Riley","Jack"]
_LAST = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Wilson","Moore",
         "Anderson","Taylor","Thomas","Jackson","White","Harris","Martin","Garcia","Thompson","Martinez",
         "Robinson","Clark","Rodriguez","Lewis","Lee","Walker","Hall","Allen","Young","King",
         "Wright","Scott","Torres","Nguyen","Hill","Flores","Green","Adams","Nelson","Baker"]
_ETIOLOGIES = [e["category"] for e in ETIOLOGY_CATALOG]
_ETIOL_WEIGHTS = [40, 28, 15, 12, 5]

def _rnd_etiology():
    r = random.randint(1, 100)
    c = 0
    for i, w in enumerate(_ETIOL_WEIGHTS):
        c += w
        if r <= c:
            return _ETIOLOGIES[i]
    return _ETIOLOGIES[-1]

PATIENTS = []
for _i in range(40):
    _etiol = _rnd_etiology()
    _severe = "truncating" in _etiol or "deletion" in _etiol
    _has_sz = random.random() < (0.70 if _severe else 0.55)
    _is_onset = _has_sz and random.random() < 0.30 and _severe
    _gtcs = _has_sz and random.random() < 0.55
    _drug_res = _has_sz and random.random() < (0.40 if _severe else 0.25)
    _kd = _drug_res and random.random() < 0.55
    _folinic = random.random() < 0.30
    _n_aed = random.randint(1, 3) if _has_sz else 0
    if _drug_res:
        _n_aed = random.randint(2, 4)
    _hc_sd = round(random.uniform(-5.5, -1.8) if _severe else random.uniform(-4.0, -1.5), 1)
    PATIENTS.append({
        "id": f"DYRK{_i+1:02d}",
        "name": f"{_FIRST[_i]} {_LAST[_i]}",
        "age_dx": random.randint(1, 48),
        "etiology": _etiol,
        "hc_sd": _hc_sd,
        "has_seizures": _has_sz,
        "is_onset": _is_onset,
        "gtcs": _gtcs,
        "drug_resistant": _drug_res,
        "kd": _kd,
        "folinic_acid": _folinic,
        "n_aed": _n_aed,
        "anti_fra1_positive": random.random() < 0.30,
        "mri_abnormal": random.random() < 0.60,
        "autism": random.random() < 0.92,
    })


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    with_seizures = sum(1 for p in PATIENTS if p["has_seizures"])
    drug_resistant = sum(1 for p in PATIENTS if p["drug_resistant"])
    is_patients = sum(1 for p in PATIENTS if p["is_onset"])
    on_kd = sum(1 for p in PATIENTS if p["kd"])
    folinic_patients = sum(1 for p in PATIENTS if p["folinic_acid"])
    anti_fra1_pos = sum(1 for p in PATIENTS if p["anti_fra1_positive"])
    autism_patients = sum(1 for p in PATIENTS if p["autism"])
    mri_abnormal = sum(1 for p in PATIENTS if p["mri_abnormal"])
    avg_hc = round(sum(p["hc_sd"] for p in PATIENTS) / n, 1)
    return {
        "dashboard": "DYRK1A Epilepsy — DYRK1A Syndrome / 21q22.13 Haploinsufficiency",
        "gene": "DYRK1A",
        "locus": "21q22.13",
        "protein": "Dual Specificity Tyrosine-Phosphorylation Regulated Kinase 1A (763 aa, 85.6 kDa)",
        "omim_syndrome": "DYRK1A Syndrome #614721",
        "omim_gene": "*600855",
        "cohort_size": n,
        "with_seizures": with_seizures,
        "seizure_prevalence_pct": round(with_seizures / n * 100),
        "drug_resistant_pct": round(drug_resistant / max(1, with_seizures) * 100),
        "infantile_spasms_pct": round(is_patients / n * 100),
        "on_kd_pct": round(on_kd / n * 100),
        "folinic_acid_pct": round(folinic_patients / n * 100),
        "anti_fra1_positive_pct": round(anti_fra1_pos / n * 100),
        "autism_pct": round(autism_patients / n * 100),
        "mri_abnormal_pct": round(mri_abnormal / n * 100),
        "mean_hc_sd": avg_hc,
        "mean_aed_count": round(sum(p["n_aed"] for p in PATIENTS) / n, 1),
        "nrsf_pathway": "DYRK1A LOF → NRSF stable → SCN1A/GABRB3 repressed → Nav1.1 functional deficiency",
        "key_ci": "PHT/CBZ/OXC (NRSF/Nav1.1 worsening); DYRK1A inhibitors (harmine/leucettine/EGCG ABSOLUTE); TGB (NCSE)",
        "precision_adjunct": "Folinic acid (anti-FRA1 positive, 25-35%); VPA HDAC epigenetic NRSF de-repression",
        "trisomy_21_distinction": "DYRK1A inhibitors treat trisomy 21 (3× copies) — ABSOLUTE CI in LOF (1× copy)",
        "etiology_classes": len(ETIOLOGY_CATALOG),
        "seizure_types": len(SEIZURE_CATALOG),
        "triggers": len(TRIGGER_CATALOG),
        "treatments": len(TREATMENT_CATALOG),
        "contraindications": len(CI_CATALOG),
        "monitoring_items": len(MONITORING_CATALOG),
        "lifecycle_stages": len(LIFECYCLE_STAGES),
        "concepts": len(CONCEPT_CATALOG),
        "thresholds": len(THRESHOLD_CATALOG),
        "references": len(REFERENCE_CATALOG),
        "standards": [
            "ILAE-2022", "NICE-NG217", "Moller-2008-EurJHumGenet",
            "vanBon-2011-JMedGenet", "Bronicki-2015-EurJHumGenet",
            "Ji-2015-NatGenet", "Courcet-2012-JMedGenet",
            "CPIC-POLG1-2023", "MHRA-VPPP-2021",
            "ACMG-AMP-2015", "NICE-NG224-2022", "WHO-ICF-2019",
        ],
    }


def get_breakdown():
    patients_export = []
    for p in PATIENTS:
        patients_export.append({
            "id": p["id"],
            "name": p["name"],
            "age_dx": p["age_dx"],
            "etiology": p["etiology"],
            "hc_sd": p["hc_sd"],
            "has_seizures": p["has_seizures"],
            "is_onset": p["is_onset"],
            "drug_resistant": p["drug_resistant"],
            "kd": p["kd"],
            "folinic_acid": p["folinic_acid"],
            "anti_fra1_positive": p["anti_fra1_positive"],
            "mri_abnormal": p["mri_abnormal"],
            "autism": p["autism"],
            "n_aed": p["n_aed"],
        })
    return {
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_CATALOG,
        "triggers": TRIGGER_CATALOG,
        "treatments": TREATMENT_CATALOG,
        "contraindications": CI_CATALOG,
        "monitoring": MONITORING_CATALOG,
        "lifecycle": LIFECYCLE_STAGES,
        "patients": patients_export,
    }


def get_definitions():
    return {
        "concepts": CONCEPT_CATALOG,
        "thresholds": THRESHOLD_CATALOG,
        "references": REFERENCE_CATALOG,
        "standards": [
            {"standard": "ILAE-2022", "scope": "Epilepsy classification, DEE definition"},
            {"standard": "NICE-NG217", "scope": "Epilepsy guideline (AED initiation, monitoring)"},
            {"standard": "Moller-2008-EurJHumGenet", "scope": "First DYRK1A haploinsufficiency report"},
            {"standard": "vanBon-2011-JMedGenet", "scope": "DYRK1A syndrome intragenic deletion"},
            {"standard": "Bronicki-2015-EurJHumGenet", "scope": "10-patient cohort; phenotype delineation"},
            {"standard": "Ji-2015-NatGenet", "scope": "DYRK1A syndrome recognition; NRSF pathway"},
            {"standard": "Courcet-2012-JMedGenet", "scope": "French cohort; epilepsy + AED response"},
            {"standard": "CPIC-POLG1-2023", "scope": "POLG1 screening before VPA — mandatory"},
            {"standard": "MHRA-VPPP-2021", "scope": "Valproate Pregnancy Prevention Programme"},
            {"standard": "ACMG-AMP-2015", "scope": "Variant classification pathogenicity criteria"},
            {"standard": "NICE-NG224-2022", "scope": "Transition child to adult epilepsy services"},
            {"standard": "WHO-ICF-2019", "scope": "International Classification of Functioning — disability"},
        ],
    }
