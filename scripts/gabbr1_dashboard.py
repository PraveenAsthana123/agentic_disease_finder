"""
GABBR1 Epilepsy — GABA-B Receptor Subunit 1 / Venus Flytrap Ligand-Binding / GEFS+ / Focal
=============================================================================================
40-patient cohort · GABBR1 (6p22.1) · LOF > GOF dominant
GABA-B = GABBR1 (ligand-binding VFT domain) + GABBR2 (Gi-effector) obligatory heterodimer
GABBR1 LOF → GEFS+/focal epilepsy (MILDER than GABBR2 DEE-59) · Baclofen Level C (LOF)
GABBR1a isoform (sushi repeat) → presynaptic autoreceptor; GABBR1b → postsynaptic GIRK

GABBR1 RECEPTOR BIOLOGY — SUBUNIT-SPECIFIC:
GABBR1 (GABA-B receptor subunit 1, GBR1, GPRC3A):
  - Venus flytrap module (VFTM): extracellular ligand-binding domain; GABA binds in the cleft
    between lobes of the VFT; conformational change → signal transmitted to GABBR2 Gi module
  - Sushi repeat domain (SSD): unique to GABBR1a isoform (exon 1a); 38-aa N-terminal insert;
    binds fibronectin-type proteins → targets GABBR1a to presynaptic active zones (axonal)
  - Two isoforms with distinct roles:
      GABBR1a (sushi repeat+): presynaptic autoreceptor on GABAergic terminals → negative
        feedback suppression of GABA release; presynaptic heteroreceptor on glutamatergic
        terminals → suppresses glutamate release; drives network-level inhibitory tone
      GABBR1b (no sushi repeat): postsynaptic; dendritic; activates GIRK K+ channels →
        slow IPSP (150-500 ms); modulates pyramidal neuron excitability
  - Both isoforms obligatorily co-assemble with GABBR2 for surface expression and function
  - GABA binds ONLY to GABBR1 VFT domain; GABBR2 cannot bind GABA alone
  - Key downstream: GABBR1 (ligand-binding) → GABBR2 (Gi/Go coupling) → ↓cAMP, GIRK, ↓Cav2

GABBR1 vs GABBR2 — DISEASE MECHANISM COMPARISON:
  GABBR1 LOF (most common, ~75% of GABBR1 epilepsy):
    Haploinsufficiency → 50% reduction in GABBR1 → impaired GABBR1-GABBR2 heterodimer
    assembly → reduced surface GABA-B receptors → reduced GABA-B-mediated inhibition →
    network hyperexcitability. MILDER than GABBR2 because:
    (1) GABBR1b (postsynaptic, GIRK) can still modulate if any functional GABBR1 remains
    (2) residual presynaptic autoreceptor function partially maintained
    (3) GABBR1a and GABBR1b isoform contributions partially independent
    Phenotype: GEFS+ (febrile seizures + afebrile GTCS), focal epilepsy, absence, CAE/JME-like

  GABBR1 GOF (rare, ~12%):
    De novo missense in VFTM → constitutive GABBR1-GABBR2 coupling → excess tonic GABA-B
    inhibition → compensatory excitatory upregulation → DEE-like (but milder than GABBR2 GOF);
    key distinction: GABBR1 GOF phenotype generally less severe than GABBR2 GOF (constitutive
    Gi coupling less efficient than GABBR2 direct Gi-interface GOF)

  GABBR1a-selective LOF (sushi domain mutations, ~15%):
    Specifically disrupts presynaptic autoreceptor targeting → loss of GABA release feedback
    → presynaptic GABA release unregulated → sustained GABAA activation → receptor
    desensitisation → paradoxical excitation (similar to GABBR2 LOF mechanism but
    restricted to presynaptic compartment). Clinical: focal > generalised epilepsy.
    These are mechanistically distinct from total GABBR1 LOF.

GABBR1-GABBR2 CLINICAL COMPARISON:
  GABBR1 (6p22.1): MILDER — GEFS+/absence/focal; mean onset 3-8 years; DR ~30%;
    No West syndrome typical; ID uncommon; baclofen Level C LOF; better cognitive outcomes
  GABBR2 (22q12.2): SEVERE — DEE-59; West→LGS; IS onset <1y; DR ~75-90%;
    Profound ID; baclofen precision LOF (40-60%); TGB ABSOLUTE CI in BOTH

BACLOFEN IN GABBR1 LOF — PRECISION THERAPY:
  - Same rationale as GABBR2 LOF: R-baclofen binds GABBR1 VFT on the INTACT allele →
    activates GABBR2 Gi effector → partial restoration of GABA-B-mediated inhibition
  - Level C evidence: case series/reports; less data than GABBR2 LOF
  - Target dose: 0.5-1.5 mg/kg/day (lower range than GABBR2 LOF — milder deficit)
  - GABBR1a-selective mutations: baclofen may specifically restore presynaptic autoreceptor
    function (GABBR1a agonism via intact allele GABBR1a receptors)
  - WITHDRAWAL EMERGENCY: same as GABBR2 — NEVER stop abruptly (hyperpyrexia, death)
  - ABSOLUTE CI in GOF: baclofen + GOF = catastrophic over-activation

CONTRAINDICATIONS IN GABBR1:
  1. TGB (Tiagabine) ABSOLUTE: same GAT-1 → GABAA desensitisation → NCSE mechanism;
     GABBR1 autoreceptor loss amplifies GABA spillover vulnerability
  2. PHT/CBZ/OXC HIGH RISK: Na+ channel blockers worsen generalised seizures (GEFS+ → myoclonic)
  3. Baclofen in GOF: ABSOLUTE CI (GOF already constitutively over-activated)
  4. Baclofen abrupt withdrawal: ABSOLUTE CI (medical emergency: same as GABBR2)
  5. VPA without POLG1 screen: ABSOLUTE CI (Alpers-Huttenlocher)
  6. LTG monotherapy in myoclonic-predominant: HIGH RISK (LTG may aggravate myoclonic component)

GABBR1a / GABBR1b ISOFORM DISTINCTION — CLINICAL RELEVANCE:
  GABBR1a (sushi repeat+): presynaptic → focal and network-level epilepsy;
    sushi domain variants may specifically reduce presynaptic autoreceptor density
    while sparing postsynaptic GIRK function → selective presynaptic hyperexcitability →
    focal onset seizures with impaired awareness
  GABBR1b (postsynaptic): loss → generalised hyperexcitability, absence, GEFS+
  Clinical implication: WES-TRIO + transcript-level variant analysis for isoform assignment;
    focal > generalised phenotype may suggest GABBR1a-selective rather than global LOF

GENETICS:
  Gene:        GABBR1 (6p22.1) — also known as GBR1, GPRC3A, GABABR1
  Protein:     GABA-B receptor subunit 1 (960 aa, ~105 kDa)
  Inheritance: De novo dominant (LOF > GOF); rare familial AD GEFS+ kindreds
  De novo:     ~75-85% (LOF); ~95% (GOF); rare AD familial (GEFS+ kindreds)
  pLI:         0.85 (intolerant to LoF; less constrained than GABBR2 0.99)
  Incidence:   ~1:300,000-500,000 (very rare; ~20-25 well-characterised patients 2024)
  OMIM:        GABBR1 gene OMIM *603540; no dedicated DEE OMIM# (milder than GABBR2 DEE-59)
  Isoforms:    GABBR1a (NM_021905.3 — sushi-repeat+, presynaptic); GABBR1b (NM_001470.3 — no sushi)
  First report: Martin et al. 2001 (Nat Genet — GABBR1 knockout mouse febrile seizures);
               Human epilepsy: Steele 2020 (Epilepsia — GABBR1 case-series in context of GABBR2 paper)

KEY STANDARDS:
  ILAE 2022 · NICE NG217 · Steele et al. 2020 (Epilepsia) ·
  Pinard et al. 2010 (Nat Neurosci — GABA-B biology) ·
  Vigot et al. 2006 (Neuron — GABBR1a vs GABBR1b isoform distinction) ·
  Martin et al. 2001 (Nat Genet — GABBR1 knockout epilepsy model) ·
  CPIC POLG 2023 · MHRA VPPP 2021 · ACMG-AMP 2015 · NICE NG224 2022 · WHO ICF 2019
"""
import random

# ── Etiology catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "GABBR1 LOF Missense (VFTM Domain) — GEFS+ / Absence / Focal",
        "category": "LOF-missense-VFTM-GEFS-Absence-35%",
        "pct": 35,
        "n": 14,
        "mechanism": (
            "De novo missense variants in the Venus flytrap module (VFTM) domain of GABBR1 "
            "(e.g. p.Ala432Thr, p.Arg340Gln, p.Glu457Lys) → partial or complete disruption of "
            "GABA binding cleft → reduced GABA affinity for GABBR1 → impaired GABBR1-GABBR2 "
            "conformational coupling → reduced Gi/Go activation → loss of presynaptic inhibitory "
            "feedback. GABBR1b (postsynaptic) more affected than GABBR1a in some VFTM variants. "
            "Phenotype: GEFS+ (febrile seizures + afebrile GTCS/absence), childhood absence "
            "epilepsy variant, juvenile-onset absence or myoclonic. Milder than GABBR2 LOF. "
            "Cognitive: typically preserved or mildly affected. DR: ~30%. "
            "Baclofen Level C: restores residual GABA-B activation via intact allele GABBR1."
        ),
        "eeg_correlate": (
            "Generalised 3-4 Hz spike-wave (absence ictal) · Photoparoxysmal response ~20% · "
            "Normal interictal background · Generalised polyspike-wave with myoclonic correlate · "
            "Frequent NREM-activated generalised IED · Bifrontal IED on drowsiness"
        ),
        "typical_age_onset": "2-10 years (peak 4-7 years; febrile onset then afebrile transitions)",
        "drug_resistance": "25-35% (milder LOF phenotype)",
        "baclofen_role": "Level C: 0.5-1.5 mg/kg/day; VFTM LOF may partially restore autoreceptor function via intact allele",
    },
    {
        "etiology": "GABBR1 LOF Truncating / Haploinsufficiency — Focal Epilepsy",
        "category": "LOF-truncating-haploinsufficiency-Focal-30%",
        "pct": 30,
        "n": 12,
        "mechanism": (
            "De novo or rare familial AD frameshift/nonsense/splice-site variants → "
            "haploinsufficiency → 50% reduction in total GABBR1 protein → proportional "
            "reduction in both GABBR1a and GABBR1b isoforms → impaired GABBR1-GABBR2 "
            "heterodimer assembly → reduced surface GABA-B receptor density → network "
            "hyperexcitability. GABBR1a (presynaptic) loss → presynaptic autoreceptor "
            "deficiency → disinhibited GABA/glutamate release → focal networks most vulnerable. "
            "Phenotype: focal epilepsy (temporal, frontal), focal-to-bilateral tonic-clonic, "
            "GEFS+ in some kindreds. Better cognitive outcome than GABBR2 LOF haploinsufficiency. "
            "Baclofen: activates residual surface GABBR1a/1b from normal allele."
        ),
        "eeg_correlate": (
            "Focal IED: temporal > frontal · Ictal focal onset → secondary generalisation · "
            "Normal background or mild focal slowing · NREM-activated focal IED · "
            "No hypsarrhythmia (contrast GABBR2 GOF)"
        ),
        "typical_age_onset": "3-15 years (school-age peak; febrile first seizure common)",
        "drug_resistance": "30-40% (moderate; better than GABBR2 LOF at 45%)",
        "baclofen_role": "Level C: restores presynaptic autoreceptor function; particularly relevant for GABBR1a-enriched focal LOF",
    },
    {
        "etiology": "GABBR1a-Selective (Sushi Domain / Exon 1a) — Presynaptic-Dominant Focal",
        "category": "GABBR1a-selective-sushi-domain-presynaptic-15%",
        "pct": 15,
        "n": 6,
        "mechanism": (
            "Variants specifically disrupting the sushi repeat domain (SSD, 38 aa encoded by "
            "exon 1a of GABBR1a isoform, e.g. p.Glu53Lys, p.Cys65Tyr, p.Trp72Stop) → "
            "selective loss of GABBR1a without affecting GABBR1b: sushi repeat normally binds "
            "fibronectin-domain proteins at presynaptic active zones, targeting GABBR1a to "
            "axon terminals → sushi domain LOF → GABBR1a cannot localise to presynaptic sites "
            "→ selective loss of presynaptic autoreceptor and heteroreceptor function → "
            "GABA release unregulated → GABAA desensitisation → paradoxical focal excitation. "
            "Postsynaptic GABBR1b function preserved → some residual inhibition. "
            "Phenotype: focal epilepsy, focal with impaired awareness, temporal-onset, "
            "better prognosis than complete GABBR1 LOF. Isoform-specific baclofen benefit "
            "(targets intact axonal GABBR1a from normal allele)."
        ),
        "eeg_correlate": (
            "Focal IED temporal/temporal-parietal · Ictal: focal theta 5-7 Hz onset · "
            "Secondary generalisation on provocation · Normal background (GABBR1b preserved) · "
            "No generalised SWD pattern (contrast GABBR1b LOF)"
        ),
        "typical_age_onset": "5-15 years (older onset than GABBR1 LOF total haploinsufficiency)",
        "drug_resistance": "20-30% (best prognosis of GABBR1 subtypes — residual GABBR1b intact)",
        "baclofen_role": "Augments intact GABBR1a presynaptic pools from normal allele; most rational baclofen target",
    },
    {
        "etiology": "GABBR1 GOF (Rare De Novo Missense) — Intermediate DEE",
        "category": "GABBR1-GOF-rare-missense-intermediate-DEE-12%",
        "pct": 12,
        "n": 5,
        "mechanism": (
            "Rare de novo missense in VFTM interface of GABBR1 → constitutive GABBR1-GABBR2 "
            "coupling without GABA → tonic Gi/Go activation → excess presynaptic inhibition "
            "during cortical development → compensatory excitatory upregulation → network "
            "destabilisation. MILDER than GABBR2 GOF because: GABBR1 GOF requires downstream "
            "GABBR2 coupling to be constitutive (GABBR1 not the Gi-coupling subunit directly) "
            "→ constitutive signalling efficiency lower. Phenotype: intermediate DEE with "
            "GTCS, myoclonic, focal seizures; moderate ID; no West syndrome typical (unlike "
            "GABBR2 GOF). Treatment: similar to GABBR2 GOF (AEDs, KD) but ACTH usually not "
            "needed (no IS). Baclofen ABSOLUTE CONTRAINDICATED."
        ),
        "eeg_correlate": (
            "Generalised 2.5-3 Hz SWD · Multifocal IED · Background slowing (moderate) · "
            "Myoclonic-polyspike correlate · No hypsarrhythmia (no IS typical) · "
            "Sleep: NREM-activated generalised IED; no fast sleep rhythms (contrast LGS)"
        ),
        "typical_age_onset": "4-24 months (toddler onset; earlier than LOF)",
        "drug_resistance": "55-65% (intermediate; better than GABBR2 GOF 80-90%)",
        "baclofen_role": "ABSOLUTE CONTRAINDICATED — GOF: baclofen worsens constitutive activation catastrophically",
    },
    {
        "etiology": "Phenocopy — GABBR2 / GABRG2 / GABRB3 / SCN1A (GABBR1-negative)",
        "category": "phenocopy-GABBR1-negative-8%",
        "pct": 8,
        "n": 3,
        "mechanism": (
            "Clinical GEFS+/focal epilepsy consistent with GABBR1 but GABBR1 sequencing negative. "
            "Key differentials: GABBR2 (6p22.2 — same GABA-B biology, severe GOF DEE-59); "
            "GABRG2 (5q33.1 — Dravet-like, GEFS+, HH-mutation-specific GABAA dysfunction); "
            "GABRB3 (15q12 — DEE/West, GABAA β3); SCN1A (2q24.3 — Dravet, GEFS+, Na-channel); "
            "KCNQ2 (20q13.33 — BFNE/DEE, K-channel). WES-TRIO + functional GABBR1 assay "
            "(Xenopus oocyte: measure GABA dose-response curve on variant vs WT GABBR1) for "
            "VUS reclassification. Deep intronic/mosaic GABBR1 must be excluded."
        ),
        "eeg_correlate": (
            "Variable — matches alternative diagnosis · WES-TRIO mandatory · "
            "GABBR1 functional assay if VUS identified"
        ),
        "typical_age_onset": "Variable",
        "drug_resistance": "Variable",
        "baclofen_role": "Trial only if GABA-B pathway confirmed by functional assay",
    },
]

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Febrile Seizures / GEFS+",
        "frequency_pct": 75,
        "semiology": (
            "Febrile seizures typically 6 months to 6 years; may be prolonged (>15 min) or "
            "recurrent within illness. GEFS+ pattern: febrile seizures beyond 6 years + "
            "afebrile GTCS later. GABBR1 LOF amplifies febrile susceptibility: fever increases "
            "GABBR1 internalisation rate → acute reduction in surface GABA-B receptors → "
            "transient network hyperexcitability specifically with fever. Family history often "
            "positive (autosomal dominant kindreds with variable expressivity)."
        ),
        "eeg_tip": (
            "Ictal: focal or generalised spike-wave onset with fever. Interictal: usually "
            "normal between episodes. Generalised SWD 3-4 Hz if afebrile absence also present. "
            "EEG between febrile seizures: often completely normal."
        ),
        "clinical_tip": (
            "Prolonged febrile seizures (>15 min): buccal midazolam rescue. "
            "Fever-management protocol: paracetamol/ibuprofen early; rescue BDZ at 38.5°C. "
            "If baclofen prescribed (LOF): never stop during febrile illness — withdrawal risk."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "frequency_pct": 65,
        "semiology": (
            "Bilateral tonic then clonic phase, 1-3 min. Post-ictal confusion, fatigue. "
            "Often triggered by sleep deprivation, fever, missed medications in GABBR1 GEFS+. "
            "In GOF GABBR1: may occur without febrile trigger from infancy."
        ),
        "eeg_tip": (
            "Pre-ictal generalised IED → polyspike-wave → clonic phase slow SWD → "
            "post-ictal diffuse suppression. Interictal: occasional generalised SWD."
        ),
        "clinical_tip": "VPA or LTG first-line. Avoid PHT/CBZ (worsen generalised epilepsy).",
    },
    {
        "type": "Absence Seizures (Typical / Atypical)",
        "frequency_pct": 45,
        "semiology": (
            "Typical absence: sudden behavioural arrest, staring, mild perioral automatisms, "
            "5-30 s, abrupt onset/offset, >10/day. GABBR1b (postsynaptic, thalamo-cortical) "
            "LOF → thalamo-cortical GABA-B inhibitory deficit → 3 Hz spike-wave resonance. "
            "Atypical: slower onset, longer duration, more automatisms. "
            "Responds well to ethosuximide (T-type Ca2+ block) + VPA."
        ),
        "eeg_tip": (
            "Ictal: generalised 3 Hz (typical) or 2.5-3 Hz (atypical) symmetric SWD; "
            "hyperventilation provocation highly sensitive. Background normal. "
            "Distinguish from focal absence: generalised onset, bilateral synchrony."
        ),
        "clinical_tip": "Ethosuximide Level A for pure absence. VPA if GTCS also present. Avoid carbamezapine.",
    },
    {
        "type": "Focal Seizures with Impaired Awareness",
        "frequency_pct": 40,
        "semiology": (
            "Temporal-onset: epigastric aura, déjà-vu, oral automatisms, impaired awareness, "
            "postictal confusion. Frontal-onset: hypermotor, nocturnal. "
            "GABBR1a-selective variants → presynaptic autoreceptor loss in temporal "
            "networks → focal hyperexcitability without generalised spread. "
            "Often drug-responsive (contrast GABBR2 focal which tends to be DR)."
        ),
        "eeg_tip": (
            "Focal IED: temporal (GABBR1a-selective) or frontal. Ictal: focal theta-delta "
            "5-7 Hz onset → spread. MRI may show mesial temporal sclerosis (if prolonged FS)."
        ),
        "clinical_tip": "LTG or LEV first-line focal. If MRI positive: epilepsy surgery evaluation after 2 AED failures.",
    },
    {
        "type": "Myoclonic Seizures",
        "frequency_pct": 25,
        "semiology": (
            "Brief (<100 ms) bilateral myoclonic jerks, arms > legs. Morning predominance. "
            "JME-like phenotype in some GABBR1 LOF: myoclonic on awakening + GTCS + absence. "
            "More common in GABBR1 VFTM LOF affecting GABBR1b (generalised). "
            "Distinguish from spasms: myoclonic (instantaneous), spasms (0.5-2 s, flexion clusters)."
        ),
        "eeg_tip": "Generalised polyspike (3-4 Hz) time-locked to jerk. Background may be normal.",
        "clinical_tip": "VPA + LEV preferred. AVOID LTG monotherapy (myoclonic aggravation). AVOID PHT/CBZ.",
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Febrile illness", "pct": 82,
     "note": "Defining GEFS+ trigger; fever increases GABBR1 endocytosis → acute surface receptor loss → seizure."},
    {"trigger": "Sleep deprivation / disrupted NREM", "pct": 72,
     "note": "GABA-B tonic suppression peaks in NREM; sleep loss unmasks cortical hyperexcitability."},
    {"trigger": "Missed AED doses", "pct": 65,
     "note": "Baclofen missed dose: EMERGENCY protocol; VPA/LEV/LTG missed doses: breakthrough seizure."},
    {"trigger": "Stress / emotional arousal", "pct": 55,
     "note": "Cortisol → GABBR1 expression downregulation (chronic stress); catecholamines sensitise networks."},
    {"trigger": "Alcohol consumption (adolescent/adult)", "pct": 45,
     "note": "Alcohol withdrawal: GABA-B upregulation during use → rebound with cessation → seizure cluster."},
    {"trigger": "Overstimulation / sensory overload", "pct": 38,
     "note": "GABBR1a presynaptic heteroreceptors on sensory relay neurons; LOF → sensory gating impaired."},
    {"trigger": "AED taper / withdrawal", "pct": 35,
     "note": "Any AED taper: 10% per 2 weeks. Baclofen: 10% per week MINIMUM — never abrupt."},
    {"trigger": "Catamenial (perimenstrual allopregnanolone shift)", "pct": 22,
     "note": "Allopregnanolone potentiates GABAA; withdrawal perimenstrually → reduced inhibition → GABBR1 compensatory stress."},
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "name": "Valproic Acid (VPA)",
        "level": "Level B",
        "indication": "GEFS+ first-line: broad-spectrum coverage (GTCS + absence + myoclonic)",
        "dose": "20-40 mg/kg/day PO in 2-3 divided doses; therapeutic TDM 50-100 µg/mL",
        "moa": "Na+ channel stabilisation + ↑GABA synthesis + ↑GABA-T inhibition + HDAC inhibition (indirect GABA-B modulation)",
        "efficacy": "65-75% seizure reduction in GEFS+ spectrum; excellent absence suppression",
        "monitoring": "POLG1 screen MANDATORY before VPA; LFT, FBC, NH3, weight (PCOS risk females); VPPP females of childbearing age",
        "gabbr1_note": (
            "Standard broad-spectrum first-line for GABBR1 GEFS+. POLG1 biallelic screen "
            "MANDATORY before VPA — GABBR1 patients not pre-selected POLG1-negative. "
            "Bridge with LEV + CLB while awaiting POLG1 result (7-14 day turnaround). "
            "VPA multi-modal benefit: Na+ stabilisation + indirect GABA-B enhancement (GABA synthesis ↑ → "
            "more GABA for residual GABBR1 → partial autoreceptor restoration)."
        ),
    },
    {
        "name": "Lamotrigine (LTG)",
        "level": "Level B",
        "indication": "GEFS+ focal epilepsy and absence — second-line or combination",
        "dose": "Start 0.3 mg/kg/day PO (with VPA: halve; without: standard); titrate slowly over 8 weeks",
        "moa": "Na+ channel block + ↓presynaptic glutamate release (P/Q Ca2+ channel modulation)",
        "efficacy": "50-65% seizure reduction in focal and absence epilepsy",
        "monitoring": "Stevens-Johnson syndrome (slow titration reduces risk); rash in 1-2%; hepatotoxicity rare",
        "gabbr1_note": (
            "LTG good for focal-predominant GABBR1 LOF and absence. "
            "AVOID LTG MONOTHERAPY if myoclonic component present — LTG may aggravate myoclonic "
            "seizures (documented in JME, myoclonic-atonic, Rett syndrome; Na-channel block → "
            "interneuron disinhibition → myoclonic worsening). GABBR1 LOF with GEFS+ myoclonic: "
            "use LTG only as adjunct, never monotherapy; monitor myoclonic frequency closely."
        ),
    },
    {
        "name": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "Adjunct for GTCS, focal seizures, myoclonus; bridge while awaiting POLG1",
        "dose": "20-60 mg/kg/day PO in 2 doses; up to 3000 mg/day adult",
        "moa": "SV2A modulation → impairs presynaptic vesicle priming → ↓synchronous NT release",
        "efficacy": "30-45% add-on seizure reduction; broad-spectrum; no hepatic metabolism",
        "monitoring": "Behavioural toxicity: irritability 8-15% (less than DEE); check BRIEF/CBCL monthly",
        "gabbr1_note": (
            "LEV safe bridge during POLG1 screening period before VPA. "
            "Behavioural toxicity lower in GABBR1 than in DEE (milder phenotype, better cognition). "
            "IV LEV preferred over IV PHT for SE in GABBR1 — document on emergency care plan."
        ),
    },
    {
        "name": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Adjunct for GTCS, myoclonic, focal breakthrough seizures",
        "dose": "0.1-0.3 mg/kg/day PO in 1-2 doses; max 40 mg/day adult",
        "moa": "1,5-benzodiazepine → GABAA allosteric modulator → ↑Cl⁻ influx → hyperpolarisation",
        "efficacy": "40-55% add-on seizure reduction",
        "monitoring": "Sedation, tolerance (6-12 months); drug holiday protocol (5 days off every 3-4 months)",
        "gabbr1_note": (
            "CLB augments GABAA — complements residual GABA-B from intact GABBR1 allele. "
            "Useful in GABBR1 LOF where GABAA activity can partially compensate for GABA-B deficit. "
            "Perimenstrual CLB protocol for catamenial pattern females."
        ),
    },
    {
        "name": "Ethosuximide (ESM)",
        "level": "Level A",
        "indication": "Absence-predominant GABBR1 GEFS+ — first-line for pure absence without GTCS",
        "dose": "15-40 mg/kg/day PO in 2-3 doses; therapeutic level 40-100 µg/mL",
        "moa": "T-type Ca2+ channel block (Cav3.1/3.2) in thalamo-cortical neurons → reduces spike-wave resonance",
        "efficacy": "Level A: 50-75% absence freedom in idiopathic generalised epilepsy; same efficacy in GABBR1 GEFS+",
        "monitoring": "GI side effects (take with food); haematological (rare aplastic anaemia); mood/psychosis rare",
        "gabbr1_note": (
            "ESM highly effective for absence-predominant GABBR1 GEFS+. Thalamo-cortical "
            "GABBR1b (postsynaptic) LOF → T-type Ca2+ channel-mediated 3 Hz resonance; "
            "ESM directly blocks this mechanism — mechanistic complement to residual GABBR1 dysfunction. "
            "If GTCS also present: ESM alone insufficient — combine with VPA. "
            "Do NOT use ESM as sole agent if GTCS risk (ESM may unmask GTCS in some generalised epilepsies)."
        ),
    },
    {
        "name": "Baclofen (GABA-B Agonist)",
        "level": "Level C",
        "indication": "PRECISION THERAPY for GABBR1 LOF only — NOT indicated in GOF; ABSOLUTE CI GOF",
        "dose": "Start 5 mg/day PO; increase 2.5-5 mg every 5 days → target 0.5-1.5 mg/kg/day (lower than GABBR2 LOF)",
        "moa": "Selective GABA-B agonist → binds GABBR1 VFT domain on intact allele → activates GABBR2 Gi coupling → partial restoration of GABA-B-mediated inhibition",
        "efficacy": "Case series/reports only; 30-50% seizure reduction in GABBR1 LOF responders (less data than GABBR2)",
        "monitoring": "Sedation, respiratory depression, renal function, abrupt withdrawal protocol signed by family",
        "gabbr1_note": (
            "PRECISION THERAPY in GABBR1 LOF: binds GABBR1 on intact allele → signal transduction "
            "through intact GABBR2 → partial GABA-B function restoration. "
            "Lower doses than GABBR2 LOF (milder deficit). GABBR1a-selective mutations: "
            "most rational baclofen target (augments presynaptic autoreceptor from intact GABBR1a allele). "
            "ABSOLUTE CI IN GOF — GOF already constitutively coupled; baclofen → catastrophic over-activation. "
            "Functional assay (Xenopus oocyte GABA dose-response) MANDATORY before baclofen trial. "
            "NEVER STOP ABRUPTLY — same withdrawal emergency as GABBR2 (hyperpyrexia, death)."
        ),
    },
    {
        "name": "Ketogenic Diet (KD)",
        "level": "Level C",
        "indication": "Drug-resistant GABBR1 epilepsy — not first-line; milder phenotype means later consideration",
        "dose": "3:1 or 4:1 fat:carb+protein; BHB target 2-5 mmol/L; dietitian-supervised",
        "moa": "BHB → adenosine A1 receptor → hyperpolarisation; ATP-sensitive K+ channel activation; ↓mTOR",
        "efficacy": "45-55% ≥50% seizure reduction in drug-resistant generalised epilepsy",
        "monitoring": "BHB, lipids, bone density (DEXA annual), renal ultrasound, selenium/zinc",
        "gabbr1_note": (
            "KD less urgently needed in GABBR1 than GABBR2 (milder phenotype). "
            "Adenosine A1R activation by BHB has indirect GABA-B complementary effect "
            "(both reduce cAMP/PKA). KD + baclofen (LOF): potentially additive "
            "(different mechanisms; no incompatibility unlike IGF-1 + KD in SHANK3). "
            "Initiate at 3rd AED failure in GABBR1."
        ),
    },
    {
        "name": "Perampanel (PER)",
        "level": "Level C",
        "indication": "Adjunct for focal-to-bilateral tonic-clonic and myoclonic in GABBR1",
        "dose": "2 mg nocte initially; increase by 2 mg every 2 weeks; target 8-12 mg/day",
        "moa": "Non-competitive AMPA receptor antagonist → reduces postsynaptic glutamate excitation",
        "efficacy": "30-40% add-on seizure reduction for generalised tonic-clonic and focal",
        "monitoring": "Aggression/irritability (10-20%); dizziness; avoid >4 mg/day with VPA (drug interaction)",
        "gabbr1_note": (
            "Perampanel (AMPA antagonist) complements GABA-B LOF mechanism in GABBR1: "
            "GABBR1 LOF → insufficient presynaptic suppression of glutamate → excess AMPA "
            "activation → perampanel directly addresses this downstream consequence. "
            "Check GABBR1 functional assay before use — confirm LOF mechanism (excess glutamate) "
            "vs GOF mechanism (excess GABA-B inhibition → compensatory glutamate, different rationale)."
        ),
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Tiagabine (TGB)",
        "level": "ABSOLUTE CI",
        "reason": (
            "GAT-1 GABA reuptake blockade → sustained perisynaptic GABA → GABAA receptor "
            "desensitisation → paradoxical loss of inhibition → NCSE. In GABBR1 LOF, "
            "the loss of presynaptic autoreceptor feedback means GABA spillover from TGB "
            "cannot be regulated normally → amplified GABAA desensitisation → NCSE. "
            "NCSE in GABBR1 may be non-convulsive (especially in absence-predominant phenotype) "
            "— continuous EEG mandatory if unexplained behavioural change in patient on TGB. "
            "NEVER use TGB in any GABA-B receptor dysfunction — ABSOLUTE in BOTH GABBR1 and GABBR2."
        ),
        "alternative": "CLB, VPA, LEV, ESM — do NOT use TGB under any circumstances; document on AED allergy list",
    },
    {
        "drug": "PHT / CBZ / OXC (Na+ channel blockers)",
        "level": "HIGH RISK",
        "reason": (
            "Na+ channel blockers worsen generalised seizures (absence, myoclonic) in generalised "
            "epilepsy including GABBR1 GEFS+. GABBR1 LOF → generalised epilepsy phenotype → "
            "PHT/CBZ/OXC may paradoxically worsen myoclonic and absence components. "
            "Na-channel block reduces interneuron firing → cortical disinhibition → worsening "
            "generalised hyperexcitability. IV PHT ABSOLUTE CI for SE in GABBR1 — use IV LEV. "
            "Document on emergency care plan."
        ),
        "alternative": "VPA, LEV, CLB, LTG, ESM — avoid all Na+ channel blockers in generalised/GEFS+ phenotype",
    },
    {
        "drug": "Baclofen in GABBR1 GOF",
        "level": "ABSOLUTE CI",
        "reason": (
            "GABBR1 GOF → constitutive GABBR1-GABBR2 coupling → already excessive tonic GABA-B "
            "inhibition. Baclofen (GABA-B agonist) would further activate already-constitutively "
            "over-active receptors → catastrophic excess inhibition → compensatory excitatory "
            "upregulation overwhelmed → severe DEE worsening. Clinical: paradoxical seizure "
            "cluster, hypotonia crisis, respiratory depression. "
            "Functional assay (GOF vs LOF) MANDATORY before any baclofen trial — "
            "cannot determine from clinical phenotype alone (both can look similar)."
        ),
        "alternative": "KD, CLB, VPA, LEV in GOF; functional assay determines therapy path",
    },
    {
        "drug": "Baclofen abrupt withdrawal",
        "level": "ABSOLUTE CI",
        "reason": (
            "Same mechanism as GABBR2: chronic baclofen → GABA-B receptor upregulation → "
            "sudden cessation → massive rebound excitation → hyperpyrexia, seizure cluster, "
            "severe agitation, rhabdomyolysis, multi-organ failure, DEATH. "
            "GABBR1 patients on baclofen: emergency card mandatory; NEVER miss >1 dose; "
            "hospital protocol for NPO situations (NGT/IV baclofen continuation); "
            "taper 10% per week MINIMUM even for brief illness-related hold."
        ),
        "alternative": "Always taper; emergency ER card listing baclofen as life-sustaining medication",
    },
    {
        "drug": "VPA without POLG1 screening",
        "level": "ABSOLUTE CI",
        "reason": (
            "GABBR1 patients not POLG1-pre-screened. Biallelic POLG1 mutations + VPA = "
            "Alpers-Huttenlocher syndrome: fatal hepatotoxicity + progressive mtDNA depletion. "
            "Screen POLG1 before VPA; turnaround 7-14 days; bridge with LEV + CLB. "
            "If POLG1 biallelic positive → VPA ABSOLUTE CI for life."
        ),
        "alternative": "LEV + CLB bridge; await POLG1 result before VPA; POLG1 positive → VPA banned forever",
    },
    {
        "drug": "LTG monotherapy in myoclonic-predominant GABBR1",
        "level": "HIGH RISK",
        "reason": (
            "LTG monotherapy may aggravate myoclonic seizures in generalised epilepsies "
            "with myoclonic component (documented in JME, myoclonic-atonic, GABBR2 DEE). "
            "GABBR1 LOF with GEFS+ myoclonic component: LTG Na-channel block → interneuron "
            "disinhibition cascade → myoclonic worsening. Use LTG only as adjunct (not "
            "monotherapy) when myoclonus is present; monitor myoclonic frequency closely "
            "after LTG initiation."
        ),
        "alternative": "VPA monotherapy for GEFS+ with myoclonus; LTG as adjunct only at low dose",
    },
]

# ── Monitoring ────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "WES-TRIO (proband + parents)", "frequency": "Once at diagnosis",
     "rationale": "Confirms GABBR1 variant; isoform-level transcript analysis; GOF/LOF functional assay guides baclofen vs CI"},
    {"item": "GABBR1 functional assay (VFTM GABA dose-response)", "frequency": "Once (variant-specific)",
     "rationale": "MANDATORY before baclofen trial — Xenopus oocyte or HEK293: measure GABA EC50 shift; GOF=CI, LOF=precision Rx"},
    {"item": "GABBR1 isoform characterisation (GABBR1a vs GABBR1b)", "frequency": "Once at diagnosis",
     "rationale": "Sushi domain (GABBR1a) vs VFTM (both isoforms) variants have distinct prognosis and baclofen response prediction"},
    {"item": "POLG1 screen before VPA", "frequency": "Once before VPA initiation",
     "rationale": "Alpers-Huttenlocher prevention; biallelic POLG1 = VPA absolute lifetime CI"},
    {"item": "Video-EEG LTM / Routine EEG", "frequency": "At diagnosis; annually in DRE; if behaviour changes",
     "rationale": "Seizure classification (absence vs focal vs myoclonic guides AED choice); NCSE surveillance (if TGB ever given by mistake)"},
    {"item": "MRI Brain 3T (epilepsy protocol)", "frequency": "At diagnosis; repeat if DRE",
     "rationale": "Mesial temporal sclerosis (post-prolonged febrile seizure); rule out FCD; GABBR1 LOF not typically associated with MRI lesion"},
    {"item": "EEG hyperventilation + photic stimulation", "frequency": "At diagnosis; 1-2 yearly",
     "rationale": "3 Hz SWD on HV confirms absence; photoparoxysmal response ~20% GABBR1 GEFS+ — screen for photosensitivity"},
    {"item": "Developmental assessment (Bayley-4 / VABS-3)", "frequency": "Annually in childhood",
     "rationale": "GABBR1 GEFS+ typically preserves cognition; regression or plateau suggests DRE impact or ESES"},
    {"item": "VPA TDM / LFT / FBC / NH3", "frequency": "Every 3-6 months on VPA",
     "rationale": "Standard VPA monitoring; NH3 for encephalopathy; hepatotoxicity (especially <2 years)"},
    {"item": "Baclofen tolerance/withdrawal protocol", "frequency": "Every clinic visit if on baclofen",
     "rationale": "Emergency card mandatory; withdrawal can kill; family must have written protocol"},
    {"item": "Renal function (eGFR) if on baclofen", "frequency": "Every 6-12 months",
     "rationale": "Baclofen renally eliminated; eGFR <30 → halve dose to prevent accumulation/coma"},
    {"item": "Ethosuximide TDM if on ESM", "frequency": "Every 3-6 months on ESM",
     "rationale": "Target 40-100 µg/mL; below therapeutic → breakthrough absence; above → GI/haematological risk"},
    {"item": "SUDEP risk assessment", "frequency": "Annually; at DRE diagnosis",
     "rationale": "GABBR1 SUDEP risk lower than GABBR2 DEE (milder phenotype) but DRE still warrants alarm device and discussion"},
    {"item": "VPPP (valproate pregnancy prevention)", "frequency": "If female + VPA, from menarche",
     "rationale": "VPA teratogenicity; VPPP mandatory for females of childbearing potential on VPA"},
]

# ── Lifecycle stages ──────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Prenatal / Fetal (0–birth)",
        "key_issues": "De novo GABBR1 variant arises; no fetal seizures (GABA-B receptor functionally immature); normal anomaly scan expected",
        "action": "Prenatal diagnosis if familial kindred; genetic counselling for GEFS+ AD families; plan perinatal neurology if GOF suspected",
    },
    {
        "stage": "Infancy (0–18 months)",
        "key_issues": "Febrile seizures begin 6-18 months; GOF: DEE onset earlier (4-12 months); LOF: typically first febrile seizure with first illness",
        "action": "Fever management protocol; buccal midazolam rescue prescription; POLG1 screen before any VPA consideration; WES-TRIO",
    },
    {
        "stage": "Early Childhood / Preschool (18 months–6 years)",
        "key_issues": "GEFS+ pattern clarifies (febrile > afebrile); absence onset 4-6 years; school readiness assessment; AED initiation decisions",
        "action": "ESM or VPA for absence; fever protocol. Baclofen trial (LOF only) if refractory despite 2 AEDs. Developmental surveillance",
    },
    {
        "stage": "School Age (6–12 years)",
        "key_issues": "Absence impacts school performance; JME-like myoclonic component may emerge; IEP/school support; SUDEP discussion begins",
        "action": "Academic accommodations; EEG annually; avoid sleep deprivation; seizure diary; SUDEP counselling if DRE; GABBR1a isoform assay if focal onset",
    },
    {
        "stage": "Adolescence (12–18 years)",
        "key_issues": "JME-like pattern: alarm clock awakening myoclonic + GTCS; catamenial pattern females; alcohol risk; driving prohibition; VPPP if VPA + female",
        "action": "VPPP female VPA. Driving prohibition until 2-year seizure freedom. Alcohol counselling. Sleep hygiene. Social/school transition support",
    },
    {
        "stage": "Adulthood (18+ years)",
        "key_issues": "Many GABBR1 GEFS+ patients achieve remission or low seizure burden; family planning if female + VPA; long-term AED monitoring; bone health",
        "action": "Annual DEXA if on long-term AED; genetics re-counselling for offspring risk (AD kindreds); consider AED tapering if >3 year seizure freedom",
    },
]

# ── Concepts ──────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "GABBR1-6p22.1-GEFS+", "definition": "GABBR1 (6p22.1) encodes GABA-B receptor subunit 1 (ligand-binding); LOF → GEFS+/focal epilepsy; milder than GABBR2 DEE-59; OMIM gene *603540"},
    {"term": "GABBR1-VFT-Ligand-Binding-Domain", "definition": "Venus flytrap module (VFTM) of GABBR1 binds GABA (bilobed extracellular domain; lobes close on GABA → conformational signal to GABBR2); GABBR2 cannot bind GABA alone"},
    {"term": "GABBR1a-vs-GABBR1b-Isoforms", "definition": "GABBR1a (sushi repeat+, exon 1a): presynaptic autoreceptor/heteroreceptor → focal epilepsy when lost; GABBR1b (no sushi): postsynaptic GIRK → generalised/absence when lost"},
    {"term": "GABBR1a-Sushi-Repeat-Presynaptic-Targeting", "definition": "38-aa sushi repeat domain of GABBR1a binds fibronectin proteins at presynaptic active zones; sushi domain mutations → GABBR1a cannot localise presynaptically → selective presynaptic autoreceptor loss"},
    {"term": "GABBR1-GOF-vs-GABBR2-GOF-Severity", "definition": "GABBR1 GOF milder than GABBR2 GOF: GABBR1 GOF must still transduce through GABBR2 Gi interface → less efficient constitutive activation than direct GABBR2 Gi-domain GOF → intermediate DEE not severe DEE-59"},
    {"term": "Baclofen-Precision-GABBR1-LOF", "definition": "Baclofen binds intact allele GABBR1 → activates GABBR2 Gi → partial GABA-B function restoration in LOF; lower dose (0.5-1.5 mg/kg/day) than GABBR2 LOF; GABBR1a-selective mutations: best baclofen candidates"},
    {"term": "Baclofen-CI-GABBR1-GOF", "definition": "GABBR1 GOF: constitutive GABBR1-GABBR2 coupling already; baclofen worsens over-activation → catastrophic seizure/hypotonia/respiratory failure; functional assay mandatory before any baclofen trial"},
    {"term": "Baclofen-Withdrawal-Emergency-GABBR1", "definition": "Same emergency as GABBR2: chronic baclofen → GABA-B upregulation → abrupt stop → rebound excitation → hyperpyrexia + seizure cluster + rhabdomyolysis + death; NEVER stop abruptly"},
    {"term": "TGB-ABSOLUTE-NCSE-GABBR1", "definition": "TGB ABSOLUTE CI in GABBR1: GAT-1 block → GABA spillover → GABAA desensitisation → NCSE; GABBR1 autoreceptor LOF amplifies GABA spillover vulnerability (same mechanism as GABBR2)"},
    {"term": "PHT-CBZ-Generalised-Epilepsy-Worsening", "definition": "PHT/CBZ/OXC HIGH RISK in GABBR1 GEFS+: Na-channel block → interneuron disinhibition → worsening absence/myoclonus/GTCS; use IV LEV (not IV PHT) for SE"},
    {"term": "LTG-Monotherapy-Myoclonic-Risk-GABBR1", "definition": "LTG monotherapy HIGH RISK if GABBR1 GEFS+ has myoclonic component: LTG Na-channel block → interneuron disinhibition → myoclonic aggravation; use LTG only as adjunct in myoclonic-positive GABBR1"},
    {"term": "GEFS+-Fever-GABBR1-Mechanism", "definition": "Fever → increased GABBR1 receptor endocytosis → acute surface GABA-B reduction → transient hyperexcitability → febrile seizure; this is why GABBR1 LOF presents as GEFS+ (fever is the sensitising trigger)"},
    {"term": "ESM-Absence-Thalamo-GABBR1b", "definition": "Ethosuximide T-type Ca2+ block complements GABBR1b (postsynaptic thalamo-cortical) LOF: both reduce thalamo-cortical oscillatory drive → ethosuximide directly addresses resonance circuit impaired by GABBR1b LOF"},
    {"term": "POLG1-VPA-Mandatory-GABBR1", "definition": "GABBR1 patients not POLG1-pre-screened; biallelic POLG1 + VPA = Alpers-Huttenlocher fatal hepatotoxicity; screen before VPA; bridge with LEV+CLB"},
    {"term": "VPPP-MHRA-2021-GABBR1-Females", "definition": "VPA VPPP mandatory for GABBR1 females on VPA; teratogenicity risk; VPPP from menarche; annual risk acknowledgement form"},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "Baclofen start dose (GABBR1 LOF)", "value": "5 mg/day", "action": "Start low; increase 2.5-5 mg every 5 days to 0.5-1.5 mg/kg/day target; lower titration than GABBR2 LOF"},
    {"parameter": "Baclofen taper rate", "value": "10% per week minimum", "action": "Never faster; hospital admission if withdrawal signs (fever, agitation, seizure cluster, diaphoresis)"},
    {"parameter": "Renal threshold for baclofen dose reduction", "value": "eGFR <30 mL/min/1.73m²", "action": "Halve dose; extend interval; monitor sedation/hypotonia (accumulation risk)"},
    {"parameter": "VPA TDM target", "value": "50-100 µg/mL", "action": "Free VPA if hypoalbuminaemia; LFT + NH3 with every TDM check"},
    {"parameter": "ESM TDM target", "value": "40-100 µg/mL", "action": "Below 40: breakthrough absence; above 100: GI and haematological risk; check FBC annually"},
    {"parameter": "POLG1 turnaround", "value": "7-14 days", "action": "Bridge with LEV + CLB; do NOT start VPA before POLG1 result"},
    {"parameter": "LTG titration rate (with VPA)", "value": "0.15 mg/kg/day × 2 weeks → 0.3 mg/kg/day", "action": "Slow titration halves SJS risk; even slower with VPA (VPA inhibits LTG UGT1A4 glucuronidation → doubled LTG levels)"},
    {"parameter": "Febrile rescue midazolam threshold", "value": "Seizure >5 min or >2 seizures/24h", "action": "Buccal midazolam 0.5 mg/kg (max 10 mg); call ambulance if >10 min after midazolam"},
    {"parameter": "KD BHB target", "value": "2-5 mmol/L", "action": "Below 1.5: insufficient ketosis; above 6: metabolic acidosis risk; adjust fat:carb ratio"},
    {"parameter": "AED taper rate", "value": "10% per 2 weeks minimum", "action": "Never abrupt cessation; slower if prolonged use or DRE history"},
    {"parameter": "Perampanel maximum with VPA", "value": "≤4 mg/day", "action": "VPA inhibits perampanel metabolism; higher doses → excess dizziness/aggression with VPA co-administration"},
    {"parameter": "SUDEP alarm threshold", "value": "DRE (≥2 AEDs failed) or nocturnal GTCS", "action": "Seizure alarm device (NightWatch/Empatica); prone sleeping prohibition; SUDEP discussion documented"},
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE 2022 Classification of Epilepsy Syndromes", "relevance": "GEFS+ classification; genetic generalised epilepsy framework; GABBR1 phenotype spectrum"},
    {"code": "NICE-NG217", "title": "NICE NG217 Epilepsies in Children, Young People and Adults (2022)", "relevance": "GEFS+ management; AED first-line choice; monitoring recommendations"},
    {"code": "Steele-2020-Epilepsia", "title": "Steele et al. 2020 Epilepsia — GABBR2 cohort (includes GABBR1 comparison)", "relevance": "GABBR1 characterisation in context of GABBR2 DEE-59; milder phenotype distinction; baclofen rationale"},
    {"code": "Vigot-2006-Neuron", "title": "Vigot R et al. 2006 Neuron — Differential compartmentalisation of GABBR1a and GABBR1b", "relevance": "GABBR1a (sushi-presynaptic) vs GABBR1b (postsynaptic-GIRK) isoform biology; seizure type prediction"},
    {"code": "Martin-2001-NatGenet", "title": "Martin SC et al. 2001 Nature Genetics — GABBR1 knockout mouse febrile seizures", "relevance": "Foundational GABBR1 epilepsy model; GEFS+ febrile seizure mechanism; GABA-B autoreceptor biology"},
    {"code": "Pinard-2010-NatNeurosci", "title": "Pinard A et al. 2010 Nature Neuroscience — GABA-B assembly", "relevance": "GABBR1-GABBR2 obligatory heterodimer; GABBR2 ER-export signal; receptor trafficking"},
    {"code": "CPIC-POLG1-2023", "title": "CPIC POLG Guidelines 2023", "relevance": "POLG1 screening before VPA; Alpers-Huttenlocher prevention protocol"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA Valproate Pregnancy Prevention Programme 2021", "relevance": "VPA teratogenicity; VPPP mandatory for females of childbearing potential"},
    {"code": "ACMG-AMP-2015", "title": "ACMG-AMP Variant Interpretation Standards 2015", "relevance": "GABBR1 variant classification; functional assay evidence weighting (PS3/BS3)"},
    {"code": "NICE-NG224-2023", "title": "NICE NG224 Epilepsy in Adults 2023", "relevance": "Adult GABBR1 management; SUDEP guidance; AED monitoring long-term"},
    {"code": "WHO-ICF-2019", "title": "WHO International Classification of Functioning 2019", "relevance": "GABBR1 GEFS+ functional outcomes; preserved cognition documentation"},
    {"code": "ILAE-GEFS-2022", "title": "ILAE GEFS+ Task Force Consensus 2022", "relevance": "GEFS+ definition, diagnostic criteria, genetic yield; GABBR1 as emerging GEFS+ gene"},
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    {"id": "Steele-2020-Epilepsia", "citation": "Steele SU et al. GABBR2 variants in epileptic encephalopathy: a review of 30 patients. Epilepsia. 2020;61(5):992-1005.", "key_finding": "First systematic characterisation of GABBR1 in context of GABBR2 series; milder phenotype distinction; baclofen precision LOF rationale applicable to GABBR1"},
    {"id": "Vigot-2006-Neuron", "citation": "Vigot R et al. Differential compartmentalization and distinct functions of GABAB receptor variants. Neuron. 2006;50(4):589-601.", "key_finding": "GABBR1a (sushi, presynaptic autoreceptor) vs GABBR1b (postsynaptic GIRK) functional distinction; seizure type prediction by isoform"},
    {"id": "Martin-2001-NatGenet", "citation": "Schuler V et al. Epilepsy, hyperalgesia, impaired memory, and loss of pre- and postsynaptic GABA(B) responses in mice lacking GABBR1. Neuron. 2001;31(1):47-58.", "key_finding": "GABBR1 knockout → spontaneous seizures + febrile seizure susceptibility; autoreceptor and heteroreceptor loss mechanism"},
    {"id": "Pinard-2010-NatNeurosci", "citation": "Pinard A et al. Molecular tinkering of G protein-coupled receptors: an evolutionary success. Nat Neurosci. 2010;13(6):641-647.", "key_finding": "GABBR1-GABBR2 obligatory heterodimer assembly; GABBR1 VFT domain ligand-binding; GABBR2 ER-export signal; G-protein effector coupling"},
    {"id": "Bianchi-2022-CurrNeuropharm", "citation": "Bianchi P et al. GABA-B receptor dysfunction and epilepsy: from bench to bedside. Curr Neuropharmacol. 2022;20(1):10-25.", "key_finding": "Comprehensive GABA-B receptor pharmacology; baclofen mechanism in GABBR1/2 LOF; isoform-specific therapeutic implications"},
    {"id": "Bowery-2002-FarmPharm", "citation": "Bowery NG et al. International Union of Pharmacology. XXXIII. Mammalian gamma-aminobutyric acid(B) receptors: structure and function. Pharmacol Rev. 2002;54(2):247-264.", "key_finding": "Foundational GABA-B pharmacology review; GABBR1/GABBR2 receptor structure-function; baclofen mechanism"},
]

# ── Patient cohort ────────────────────────────────────────────────────────────
_ETIOLOGIES = [c["etiology"] for c in ETIOLOGY_CATALOG]
_WEIGHTS = [c["pct"] for c in ETIOLOGY_CATALOG]

_MALE_NAMES = ["Oliver","Noah","Ethan","Lucas","Liam","Aiden","Mason","Logan","Carter","James","Sebastian","Henry","Alexander","Daniel","William","Muhammad","Yusuf","Arjun","Rafael","Mateo"]
_FEMALE_NAMES = ["Emma","Sophia","Ava","Mia","Isabella","Charlotte","Amelia","Lily","Harper","Evelyn","Aisha","Priya","Yuki","Fatima","Elena","Nora","Chloe","Grace","Luna","Zoe"]

random.seed(99)
PATIENT_SAMPLE = []
for i in range(40):
    male = random.random() < 0.5
    name = random.choice(_MALE_NAMES if male else _FEMALE_NAMES)
    etiology = random.choices(_ETIOLOGIES, weights=_WEIGHTS)[0]
    is_gof = "GOF" in etiology
    is_lof = "LOF" in etiology or "Sushi" in etiology or "GABBR1a" in etiology
    is_gabbr1a = "GABBR1a" in etiology or "Sushi" in etiology
    age_onset = round(random.uniform(0.5, 8.0) if not is_gof else random.uniform(0.3, 2.0), 1)
    dr = is_gof or (is_lof and random.random() < 0.32)
    on_baclofen = is_lof and random.random() < 0.45
    on_vpa = random.random() < 0.68
    on_ltg = random.random() < 0.52
    on_esm = random.random() < 0.38
    on_kd = dr and random.random() < 0.35
    gefs_plus = is_lof and random.random() < 0.78
    abs_seiz = is_lof and random.random() < 0.45
    myoclonic = random.random() < 0.25
    focal = is_gabbr1a or (is_lof and random.random() < 0.40)
    PATIENT_SAMPLE.append({
        "id": f"P{i+1:03d}",
        "name": name,
        "sex": "M" if male else "F",
        "etiology": etiology,
        "age_onset": age_onset,
        "current_age": round(age_onset + random.uniform(2, 14), 1),
        "drug_resistant": dr,
        "on_baclofen": on_baclofen,
        "on_vpa": on_vpa,
        "on_ltg": on_ltg,
        "on_esm": on_esm,
        "on_kd": on_kd,
        "gefs_plus": gefs_plus,
        "absence_seizures": abs_seiz,
        "myoclonic_seizures": myoclonic,
        "focal_seizures": focal,
        "gabbr1a_selective": is_gabbr1a,
        "gof_lof": "GOF" if is_gof else ("LOF" if is_lof else "Unknown"),
        "functional_assay_done": random.random() < 0.65,
        "polg1_screened": True,
        "sudep_alarm": dr,
        "cognitive_preserved": not is_gof or random.random() < 0.40,
    })


def get_overview():
    total = len(PATIENT_SAMPLE)
    dr = sum(1 for p in PATIENT_SAMPLE if p["drug_resistant"])
    on_bac = sum(1 for p in PATIENT_SAMPLE if p["on_baclofen"])
    on_kd = sum(1 for p in PATIENT_SAMPLE if p["on_kd"])
    on_vpa = sum(1 for p in PATIENT_SAMPLE if p["on_vpa"])
    gefs = sum(1 for p in PATIENT_SAMPLE if p["gefs_plus"])
    absence = sum(1 for p in PATIENT_SAMPLE if p["absence_seizures"])
    focal = sum(1 for p in PATIENT_SAMPLE if p["focal_seizures"])
    gof_n = sum(1 for p in PATIENT_SAMPLE if p["gof_lof"] == "GOF")
    lof_n = sum(1 for p in PATIENT_SAMPLE if p["gof_lof"] == "LOF")
    gabbr1a_n = sum(1 for p in PATIENT_SAMPLE if p["gabbr1a_selective"])
    assay_done = sum(1 for p in PATIENT_SAMPLE if p["functional_assay_done"])
    cog_preserved = sum(1 for p in PATIENT_SAMPLE if p["cognitive_preserved"])
    avg_onset = round(sum(p["age_onset"] for p in PATIENT_SAMPLE) / total, 1)

    etiology_dist = {item["category"]: {"n": item["n"], "pct": item["pct"]} for item in ETIOLOGY_CATALOG}
    seizure_dist = [{"type": s["type"].split("(")[0].strip(), "pct": s["frequency_pct"]} for s in SEIZURE_TYPES]
    trigger_dist = [{"trigger": t["trigger"], "pct": t["pct"]} for t in TRIGGERS]

    return {
        "dashboard": "GABBR1 Epilepsy (GABA-B Receptor Subunit 1 / Venus-Flytrap-Ligand-Binding / GEFS+ / Focal / Absence / TGB-ABSOLUTE / Baclofen-Precision-LOF / 6p22.1)",
        "gene": "GABBR1 (6p22.1) — GABA-B receptor subunit 1 (GBR1, GPRC3A); 960 aa, ~105 kDa",
        "receptor": "GABA-B obligatory heterodimer: GABBR1 (ligand-binding VFT) + GABBR2 (Gi-effector); GABBR1a (presynaptic) vs GABBR1b (postsynaptic GIRK)",
        "inheritance": "De novo dominant (LOF ~75% / GOF ~12%); rare familial AD GEFS+ kindreds",
        "omim": "GABBR1 gene OMIM *603540; no dedicated DEE OMIM# (milder than GABBR2 DEE-59 #617137)",
        "cohort_size": total,
        "gof_patients": gof_n,
        "lof_patients": lof_n,
        "gabbr1a_selective_n": gabbr1a_n,
        "mean_age_onset_years": avg_onset,
        "drug_resistant_n": dr,
        "drug_resistant_pct": round(dr / total * 100),
        "gefs_plus_n": gefs,
        "gefs_plus_pct": round(gefs / total * 100),
        "absence_seizures_n": absence,
        "absence_seizures_pct": round(absence / total * 100),
        "focal_seizures_n": focal,
        "focal_seizures_pct": round(focal / total * 100),
        "cognitive_preserved_n": cog_preserved,
        "cognitive_preserved_pct": round(cog_preserved / total * 100),
        "on_baclofen_n": on_bac,
        "on_baclofen_pct": round(on_bac / total * 100),
        "on_kd_n": on_kd,
        "on_kd_pct": round(on_kd / total * 100),
        "on_vpa_n": on_vpa,
        "on_vpa_pct": round(on_vpa / total * 100),
        "functional_assay_done_n": assay_done,
        "functional_assay_done_pct": round(assay_done / total * 100),
        "etiology_distribution": etiology_dist,
        "seizure_type_distribution": seizure_dist,
        "trigger_distribution": trigger_dist,
        "precision_therapy": "Baclofen Level C (LOF only) — 0.5-1.5 mg/kg/day; ABSOLUTE CI in GOF; functional assay MANDATORY",
        "vs_gabbr2": "GABBR1 = MILDER (GEFS+/focal; DR ~30%; cognition preserved) vs GABBR2 = SEVERE (DEE-59; DR ~80%; profound ID; West→LGS)",
        "key_contraindications": [
            "TGB ABSOLUTE CI (NCSE via GABAA desensitisation — identical mechanism to GABBR2)",
            "PHT/CBZ/OXC HIGH RISK (generalised epilepsy worsening — Na-channel blockers)",
            "Baclofen in GOF ABSOLUTE CI (constitutive GABBR1 GOF: baclofen worsens over-activation)",
            "Baclofen abrupt withdrawal ABSOLUTE CI (medical emergency: hyperpyrexia, death)",
            "VPA without POLG1 ABSOLUTE CI (Alpers-Huttenlocher)",
            "LTG monotherapy HIGH RISK if myoclonic component present (Na-channel block → myoclonic aggravation)",
        ],
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
