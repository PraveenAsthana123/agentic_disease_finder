"""
GRIN2D Epilepsy — DEE / GluN2D / NMDA Receptor Subunit 2D / Extrasynaptic Tonic NMDA
=======================================================================================
40-patient cohort · GRIN2D (19q13.33) · De novo dominant GOF (predominantly) · NMDA Channelopathy

GRIN2D BIOLOGY:
GRIN2D (19q13.33) encodes GluN2D (NR2D), the fourth regulatory subunit of the NMDA receptor
family. NMDA receptors are obligate heterotetramers: 2× GluN1 + 2× GluN2 (or 1 GluN3).
GluN2D is unique among the GluN2 subunits:

  (a) EXPRESSION PATTERN:
      - Embryonic brain: widespread (striatum, brainstem, spinal cord, interneurons)
      - Mature brain: primarily subcortical (subthalamic nucleus / STN, substantia nigra / SN,
        VTA, locus coeruleus, cerebellum, deep cerebellar nuclei, spinal dorsal horn)
      - Predominantly EXTRASYNAPTIC and PERISYNAPTIC (tonic NMDA current, not phasic)
      - Expression in GABAergic and dopaminergic neurons (not primarily glutamatergic pyramidal)

  (b) UNIQUE KINETICS:
      - SLOWEST deactivation of all GluN2 subunits: tau_decay ~4 seconds (vs GluN2A ~50 ms,
        GluN2B ~300 ms) → near-tonic contribution even during sparse synaptic activity
      - LOW Mg²⁺ sensitivity: partial relief of Mg²⁺ block at resting membrane potential
        → constitutive NMDA conductance at resting potential (extrasynaptic tonic current)
      - LOWER Ca²⁺ permeability than GluN2A/2B: PCa/PNa ratio ~2 (vs ~10 for GluN2A)
      - IFENPRODIL-INSENSITIVE: ifenprodil is selective for GluN2B (NTD-based); GluN2D lacks
        the GluN2B-specific NTD binding site → ifenprodil does not work in GRIN2D GOF

  (c) PHARMACOLOGY IMPLICATIONS:
      - Memantine: uncompetitive open-channel blocker — preferentially blocks constitutively
        open channels (channel MUST be open for memantine entry) → effective for GOF which
        increases P_open; less effective if channel activation rate is not increased
      - D-Cycloserine: partial agonist at glycine site on GluN1 → may worsen GOF by further
        increasing open probability; avoid in GOF
      - Ketamine: potent uncompetitive NMDA blocker → effective for GOF-driven status epilepticus
        (RSE); in LOF variants → worsens NMDA hypofunction
      - Agmatine: endogenous polyamine / putative GluN2 modulator; experimental
      - QNZ46: GluN2D-selective antagonist (research compound, not clinical)

GRIN2D DUAL MECHANISM (GOF vs LOF):
  GOF (gain-of-function, ~70% of GRIN2D epilepsy patients):
    - De novo missense at key positions: p.Leu670Leu (splicing GOF), p.Val667Leu, p.Ala636Thr,
      p.Met739Ile, p.Ser671Lys (M2/M3 channel-lining domain, lurcher-homologous sites)
    - Lurcher mutation equivalent (GluD2 Ala654Thr) → dramatic increase in open probability
    - Constitutive/excessive NMDA activation → excess Ca²⁺ in STN/SN/VTA neurons →
      excitotoxicity + dopaminergic neurotoxicity → movement disorder + seizures
    - Phenotype: Infantile-onset DEE, profound ID, MOVEMENT DISORDER (dystonia/athetosis),
      hyperkinesis, choreoathetosis — reflects GluN2D's subcortical/basal ganglia expression
    - EEG: diffuse slowing, multifocal epileptiform, burst-suppression in severe cases
    - PRECISION THERAPY: Memantine (channel open-state blocker) — selective for high-P_open GOF
    - KEY DIFFERENCE from GRIN2B GOF: movement disorder is MORE PROMINENT (basal ganglia/STN)
      vs GRIN2B GOF (cortical/West syndrome dominant)

  LOF (loss-of-function, ~20% of GRIN2D epilepsy patients):
    - Truncating/frameshift/haploinsufficiency
    - Extrasynaptic tonic NMDA current loss → impaired tonic inhibitory drive on GABAergic
      interneurons of STN → altered basal ganglia circuit → paradoxical excitability increase
    - Phenotype: less severe, focal epilepsy, NDD, milder movement features
    - AVOID memantine, AVOID ketamine (worsens NMDA hypofunction)
    - No specific precision therapy identified

CONTRAINDICATIONS IN GRIN2D EPILEPSY:
  1. MEMANTINE IN LOF (ABSOLUTE CI):
     Further reduces NMDA activity → worsens LOF-driven disinhibition → cognitive decline + seizure worsening.
     GOF/LOF functional assay MANDATORY before prescribing memantine.

  2. KETAMINE IN LOF (ABSOLUTE CI for RSE):
     Ketamine is an NMDA antagonist → additive with LOF → prolonged dissociative state +
     worsened excitotoxic-disinhibition. Use Midazolam/Lorazepam/VPA for RSE in LOF instead.

  3. TIAGABINE (ABSOLUTE CI in GRIN2D DEE):
     GABA reuptake inhibitor → NCSE in dysmature/dysplastic cortex with altered
     GABAergic/NMDA balance. Multiple cases of TGB-induced NCSE in genetic DEE.

  4. D-CYCLOSERINE IN GOF (HIGH RISK):
     Partial agonist at glycine co-agonist site on GluN1 → increases P_open → worsens GOF
     hyperactivation. D-cycloserine is investigational for NMDA LOF (hypofunction);
     contraindicated in GOF.

  5. CBZ / OXC / PHT (HIGH RISK in generalised phenotype):
     NaV-preferring blockers worsen generalised seizures via PV+ interneuron NaV1.1 block →
     disinhibition → exacerbation. Acceptable as CBZ/OXC for focal features only if GOF/LOF confirmed.

  6. IFENPRODIL / GluN2B-SELECTIVE ANTAGONISTS (INEFFECTIVE, NOT ABSOLUTE CI):
     Ifenprodil is GluN2B-NTD-selective. GRIN2D GOF receptors lack the GluN2B NTD site →
     ifenprodil has no therapeutic effect on GluN2D channels. Not harmful, just useless —
     document to prevent prescribing in error.

  7. VPA WITHOUT POLG EXCLUSION (ABSOLUTE CI):
     POLG1 mitochondrial encephalopathy mimics GRIN2D DEE (regression + seizures + NDD).
     VPA in POLG1+ → fatal hepatotoxicity. Full POLG1 sequencing mandatory before VPA.

PRECISION TREATMENTS IN GRIN2D EPILEPSY:
  Memantine (GOF ONLY, Level C — very few published cases):
    Uncompetitive NMDA receptor open-channel blocker. Enters open channel during burst firing
    → selectively reduces constitutively open GOF channels. Similar mechanism to GRIN2B GOF
    but extrasynaptic GRIN2D may respond differently to low-dose tonic block.
    Dose: 0.3 mg/kg/day titrating to 5-10 mg/day (paediatric). Monitoring: cognitive/motor response.
    MANDATORY: confirm GOF on functional assay before prescribing.

  ACTH (Level A for associated West syndrome):
    Reduces hypsarrhythmia. Day-14 EEG gates escalation. Standard protocol.

  VPA (First-line broad spectrum, Level B):
    Broad-spectrum (NaV, T-Ca, GAD-GABA). POLG MANDATORY. Does not directly modulate NMDA.

  KD (Level B for DRE):
    Ketogenic diet — multiple mechanisms including: AMPA/NMDA modulation via beta-hydroxybutyrate,
    adenosine-A1 mediated inhibition, mTOR suppression. Effective in GRIN2D DRE (40-50% response).

  CLB (Level B adjunct):
    Clobazam — 1,5-benzodiazepine with α2/α3 GABA-A preference; less sedation than diazepam.
    Useful for myoclonic/focal seizure adjunct. Not for status (standard BDZ preferred for SE).

  LEV (Level B adjunct):
    Levetiracetam — SV2A vesicle release modifier. Safe in GRIN2D (no POLG interaction).
    Some data for focal seizures and as adjunct in DEE.

  Movement Disorder Management (Clonazepam / Baclofen / Trihexyphenidyl):
    GluN2D's expression in basal ganglia → movement disorder (dystonia/choreoathetosis) is a
    prominent feature. Low-dose clonazepam (GABA-A), baclofen (GABA-B STN modulation),
    or trihexyphenidyl (anticholinergic for dystonia). NOT anticonvulsant — adjunct for motor symptoms.

GRIN2D IN THE NMDA RECEPTOR FAMILY (Complete):
  GluN1  (GRIN1,  9q34.3)  — obligate pore-forming structural subunit; D-serine co-agonist
  GluN2A (GRIN2A, 2q33.1)  — fast synaptic; EAS/CSWS/LKS; anti-seizure; adult cortex
  GluN2B (GRIN2B, 12p12.1) — slow synaptic; DEE27; West/Ohtahara; GOF memantine; foetal cortex
  GluN2C (GRIN2C, 17q25.1) — cerebellar granule cells; less epilepsy-associated
  GluN2D (GRIN2D, 19q13.33) — extrasynaptic/subcortical/basal ganglia; THIS DASHBOARD
  GluN3A (GRIN3A, 9q31.1)  — inhibitory; reduces Ca2+ permeability; developmental transient

KEY PHARMACOLOGICAL DISTINCTIONS (GRIN2D vs GRIN2B):
  1. IFENPRODIL: Effective in GRIN2B GOF (NTD site present) → USELESS in GRIN2D GOF (no NTD site)
  2. MOVEMENT DISORDER: GRIN2D GOF → severe (basal ganglia/STN/SN) vs GRIN2B GOF → mild (cortical)
  3. KINETICS: GluN2D tau ~4s (slowest) vs GluN2B tau ~300ms vs GluN2A tau ~50ms
  4. Mg2+ SENSITIVITY: GluN2D: low (partial tonic current at rest) vs GluN2B: intermediate
  5. EXPRESSION: GluN2D subcortical/extrasynaptic vs GluN2B cortical/synaptic
  6. D-CYCLOSERINE: Investigational for GRIN2B LOF (NMDA hypofunction) → CI in GRIN2D GOF
"""

import random

# ── Seed for reproducibility ────────────────────────────────────────────────
random.seed(20240319)

# ── Etiology catalog ────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "class": "De novo GRIN2D GOF missense — channel-lining M2/M3 domain (lurcher-equivalent)",
        "pct": 38,
        "examples": "p.Val667Leu, p.Ala636Thr, p.Met739Ile (19q13.33)",
        "mechanism": "GOF → increased P_open → constitutive NMDA current → excitotoxicity in basal ganglia + cortex → DEE + movement disorder",
        "precision": "Memantine (open-channel block; GOF ONLY) + ACTH (West) + POLG screen before VPA",
    },
    {
        "class": "De novo GRIN2D GOF splice/synonymous — altered splicing → constitutive activation",
        "pct": 20,
        "examples": "p.Leu670Leu (synonymous, creates cryptic splice site → in-frame insertion in M3)",
        "mechanism": "Cryptic splicing → exon inclusion in M3 domain → channel gate dysfunction → high P_open",
        "precision": "Same as missense GOF: Memantine + VPA POLG screen + ACTH for West",
    },
    {
        "class": "De novo GRIN2D GOF — other GOF mechanism (gain of current without lurcher site)",
        "pct": 12,
        "examples": "p.Ser671Lys, p.Gln813Glu — altered gating or reduced Mg2+ sensitivity",
        "mechanism": "Enhanced channel activity at resting potential via reduced Mg2+ block or increased glycine affinity",
        "precision": "Memantine — monitor movement disorder (baclofen/clonazepam adjunct for dystonia)",
    },
    {
        "class": "De novo GRIN2D LOF — haploinsufficiency / truncating frameshift",
        "pct": 20,
        "examples": "Frameshift/nonsense → premature stop → NMD → 50% allele reduction",
        "mechanism": "Tonic extrasynaptic NMDA current loss → altered STN/basal ganglia inhibitory drive → mild-moderate DEE + focal epilepsy",
        "precision": "NO memantine / NO ketamine. LEV/VPA broad-spectrum. GOF/LOF assay MANDATORY",
    },
    {
        "class": "Phenocopy — GRIN2B, GRIN2A, GRIN1, or non-GRIN DEE with movement disorder",
        "pct": 10,
        "examples": "GRIN2B GOF (most common phenocopy), ATP1A3, GNAO1, KMT2B dystonia-epilepsy",
        "mechanism": "Overlapping DEE + movement disorder phenotype; distinguish by genetic testing",
        "precision": "Ifenprodil may work for GRIN2B GOF phenocopy; not for GRIN2D — gene confirmation mandatory",
    },
]

# ── Patient cohort ───────────────────────────────────────────────────────────
def _patient(pid, age_y, onset_m, sex, variant, gof_lof, west, myoc, dystonia,
             dre, kd, memantine, polg_tested, focal_sz, movement_severity):
    return {
        "pid": f"GD{pid:03d}",
        "age_years": age_y,
        "onset_months": onset_m,
        "sex": sex,
        "variant": variant,
        "gof_lof": gof_lof,
        "west_syndrome": west,
        "myoclonic_seizures": myoc,
        "dystonia_choreoathetosis": dystonia,
        "drug_resistant": dre,
        "on_kd": kd,
        "memantine_rx": memantine,
        "polg_tested": polg_tested,
        "focal_seizures": focal_sz,
        "movement_severity": movement_severity,
    }

PATIENTS = [
    _patient(1,  5,  4,  "F", "p.Val667Leu (GOF)",       "GOF", True,  True,  True,  True,  True,  True,  "Y", False, "severe"),
    _patient(2,  3,  6,  "M", "p.Ala636Thr (GOF)",       "GOF", True,  False, True,  True,  False, True,  "Y", True,  "moderate"),
    _patient(3,  7,  3,  "F", "p.Leu670Leu (GOF-splice)","GOF", False, True,  True,  True,  True,  True,  "Y", True,  "severe"),
    _patient(4,  2,  8,  "M", "p.Met739Ile (GOF)",        "GOF", True,  True,  False, False, False, False, "Y", False, "mild"),
    _patient(5,  9,  5,  "F", "p.Ser671Lys (GOF)",        "GOF", False, False, True,  True,  True,  True,  "Y", True,  "moderate"),
    _patient(6,  4,  10, "M", "p.Val667Leu (GOF)",        "GOF", True,  True,  True,  True,  True,  True,  "Y", False, "severe"),
    _patient(7,  6,  7,  "F", "p.Ala636Thr (GOF)",        "GOF", False, True,  False, True,  False, True,  "Y", True,  "mild"),
    _patient(8,  11, 2,  "M", "p.Gln813Glu (GOF)",        "GOF", True,  False, True,  True,  True,  False, "Y", False, "moderate"),
    _patient(9,  3,  9,  "F", "p.Leu670Leu (GOF-splice)","GOF", False, True,  True,  True,  False, True,  "Y", True,  "severe"),
    _patient(10, 8,  4,  "M", "p.Met739Ile (GOF)",        "GOF", False, False, True,  False, False, True,  "N", True,  "moderate"),
    _patient(11, 5,  6,  "F", "p.Val667Leu (GOF)",        "GOF", True,  True,  False, True,  True,  True,  "Y", True,  "mild"),
    _patient(12, 2,  3,  "M", "p.Ala636Thr (GOF)",        "GOF", True,  True,  True,  True,  False, False, "Y", False, "severe"),
    _patient(13, 10, 8,  "F", "p.Ser671Lys (GOF)",        "GOF", False, False, True,  True,  True,  True,  "Y", True,  "moderate"),
    _patient(14, 4,  5,  "M", "p.Gln813Glu (GOF)",        "GOF", True,  False, False, False, False, True,  "Y", False, "mild"),
    _patient(15, 7,  7,  "F", "p.Leu670Leu (GOF-splice)","GOF", False, True,  True,  True,  True,  True,  "Y", True,  "severe"),
    _patient(16, 3,  2,  "M", "p.Met739Ile (GOF)",        "GOF", True,  True,  True,  True,  False, True,  "N", False, "moderate"),
    _patient(17, 6,  4,  "F", "p.Val667Leu (GOF)",        "GOF", True,  False, True,  True,  True,  False, "Y", True,  "severe"),
    _patient(18, 8,  6,  "M", "p.Ala636Thr (GOF)",        "GOF", False, True,  False, True,  False, True,  "Y", True,  "mild"),
    _patient(19, 5,  8,  "F", "p.Ser671Lys (GOF)",        "GOF", False, True,  True,  True,  True,  True,  "Y", True,  "moderate"),
    _patient(20, 4,  5,  "M", "p.Gln813Glu (GOF)",        "GOF", True,  False, True,  False, False, True,  "Y", False, "moderate"),
    _patient(21, 9,  3,  "F", "p.Leu670Leu (GOF-splice)","GOF", False, True,  False, True,  True,  True,  "Y", True,  "mild"),
    _patient(22, 3,  9,  "M", "p.Val667Leu (GOF)",        "GOF", True,  True,  True,  True,  False, False, "Y", False, "severe"),
    _patient(23, 6,  6,  "F", "p.Ala636Thr (GOF)",        "GOF", True,  False, True,  True,  True,  True,  "N", True,  "moderate"),
    _patient(24, 11, 7,  "M", "p.Met739Ile (GOF)",        "GOF", False, True,  False, False, False, True,  "Y", True,  "mild"),
    _patient(25, 5,  4,  "F", "p.Ser671Lys (GOF)",        "GOF", False, False, True,  True,  True,  True,  "Y", True,  "severe"),
    _patient(26, 4,  12, "M", "Frameshift p.Arg450fs (LOF)","LOF",False, False, False, False, False, False, "Y", True,  "none"),
    _patient(27, 7,  14, "F", "Nonsense p.Trp532* (LOF)","LOF", False, False, False, True,  False, False, "Y", True,  "mild"),
    _patient(28, 3,  18, "M", "Frameshift p.Gln612fs (LOF)","LOF",False, False, True,  False, False, False, "Y", True,  "mild"),
    _patient(29, 9,  10, "F", "Splice LOF (LOF)",          "LOF", False, True,  False, True,  True,  False, "Y", True,  "none"),
    _patient(30, 5,  16, "M", "Nonsense p.Arg710* (LOF)", "LOF", False, False, False, False, False, False, "Y", True,  "none"),
    _patient(31, 6,  12, "F", "Frameshift p.Ala450fs (LOF)","LOF",False, False, True,  True,  False, False, "N", True,  "mild"),
    _patient(32, 8,  20, "M", "Splice LOF (LOF)",          "LOF", False, False, False, False, False, False, "Y", True,  "none"),
    _patient(33, 4,  14, "F", "Phenocopy — GRIN2B GOF",   "GOF", True,  True,  False, True,  True,  False, "Y", False, "mild"),
    _patient(34, 6,  9,  "M", "Phenocopy — GNAO1 GOF",    "GOF", False, False, True,  True,  False, False, "Y", True,  "severe"),
    _patient(35, 5,  6,  "F", "Phenocopy — KMT2B",        "GOF", False, False, True,  True,  True,  False, "Y", True,  "moderate"),
    _patient(36, 3,  4,  "M", "p.Val667Leu (GOF)",        "GOF", True,  False, True,  True,  False, True,  "Y", False, "moderate"),
    _patient(37, 7,  5,  "F", "p.Ala636Thr (GOF)",        "GOF", True,  True,  False, True,  True,  True,  "Y", True,  "mild"),
    _patient(38, 5,  7,  "M", "p.Leu670Leu (GOF-splice)","GOF", False, True,  True,  True,  False, True,  "N", True,  "severe"),
    _patient(39, 9,  3,  "F", "Nonsense p.Glu430* (LOF)", "LOF", False, False, False, False, False, False, "Y", True,  "none"),
    _patient(40, 4,  8,  "M", "p.Met739Ile (GOF)",        "GOF", True,  True,  True,  True,  True,  True,  "Y", False, "moderate"),
]

# ── Seizure types ────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal Motor Seizures (subcortical / secondary generalisation)",
        "prevalence_pct": 75,
        "eeg": "Focal discharge (posterior/temporal) → secondary generalisation; background slowing",
        "semiology": "Contralateral tonic/clonic limb jerks; often associated with movement disorder episodes; may be hard to distinguish from dystonic posturing",
        "clinical_tip": "Video-EEG mandatory — pure dystonic episodes (non-ictal) may mimic focal seizures in GRIN2D GOF. GOF patients have BOTH ictal (EEG correlate) and non-ictal dystonia.",
    },
    {
        "type": "Epileptic Spasms / West Syndrome",
        "prevalence_pct": 55,
        "eeg": "Hypsarrhythmia (modified or classical); high-amplitude chaotic multifocal pattern",
        "semiology": "Flexion/extension spasms; often in clusters on waking; asymmetric in GRIN2D (subcortical asymmetry)",
        "clinical_tip": "ACTH is Level A for GRIN2D West (same protocol as other genetic West). Day-14 EEG for hypsarrhythmia resolution. VGB Level A adjunct (SHARE REMS). Failure → KD.",
    },
    {
        "type": "Myoclonic Seizures",
        "prevalence_pct": 48,
        "eeg": "Polyspike-wave (3-5 Hz); may have brief burst-suppression in severe GOF",
        "semiology": "Sudden brief axial / limb jerks; often on waking; overlaps with non-ictal myoclonus in GRIN2D movement disorder",
        "clinical_tip": "VPA + CLB combination effective for myoclonic component. LTG AVOID (myoclonic aggravation). The myoclonus/dystonia distinction matters: EEG correlate (ictal) vs absent (non-ictal movement disorder).",
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "prevalence_pct": 62,
        "eeg": "Generalised spike-wave evolving to GTCS pattern; post-ictal depression",
        "semiology": "Tonic phase → clonic phase → post-ictal confusion; prolonged in GOF (status risk)",
        "clinical_tip": "GOF GRIN2D patients have higher RSE risk. RSE protocol: BDZ first → IV LEV/VPA → consider ketamine ONLY in confirmed GOF (ketamine ABSOLUTE CI in LOF).",
    },
    {
        "type": "Non-ictal Movement Disorder (dystonia / choreoathetosis — NOT a seizure type)",
        "prevalence_pct": 68,
        "eeg": "NO EEG correlate — this is movement disorder, not epileptic",
        "semiology": "Sustained twisting postures (dystonia), writhing movements (athetosis), or rapid involuntary movements (chorea); reflects GluN2D subcortical basal ganglia / STN expression",
        "clinical_tip": "CRITICAL DISTINCTION: non-ictal movement disorder does NOT respond to AEDs. Treat with baclofen (GABA-B STN modulation), clonazepam (low-dose), trihexyphenidyl (anticholinergic dystonia). Memantine may reduce both seizures AND movement disorder in GOF.",
    },
]

# ── Triggers ─────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Febrile illness", "prevalence_pct": 85, "note": "Most potent trigger in GRIN2D — fever lowers Mg2+ block efficacy (temperature-sensitive kinetics) AND reduces GABA-A surface expression; GOF channels become more active at elevated temperature", "management": "Early antipyretics (paracetamol + ibuprofen alternating); rescue CLB at fever onset; fever threshold plan documented"},
    {"trigger": "Sleep deprivation / NREM sleep transitions", "prevalence_pct": 72, "note": "GluN2D expressed in locus coeruleus (NE arousal system) → NREM transitions dysregulate subcortical arousal → seizure threshold reduction", "management": "Strict sleep hygiene; rescue CLB if sleep severely disrupted; bedtime CBZ heavy-dosing if focal features"},
    {"trigger": "Missed AED dose", "prevalence_pct": 68, "note": "Sudden withdrawal of VPA/LEV/CLB → rebound excitability. GRIN2D GOF: discontinuation of memantine also triggers seizure cluster", "management": "Rescue CLB 10 mg PR/buccal; ensure medication regimen simplification; avoid >1 missed dose"},
    {"trigger": "Emotional stress / Catecholamine surge", "prevalence_pct": 55, "note": "GRIN2D expressed in VTA (dopaminergic) and LC (noradrenergic); catecholamine surge during stress → GRIN2D-expressing neurons hyperactivated → movement disorder exacerbation + seizure risk", "management": "Stress management; low-dose clonazepam PRN for dystonic storms; baclofen for chronic stress-related movement disorder"},
    {"trigger": "Overstimulation / sensory overload", "prevalence_pct": 45, "note": "Spinal dorsal horn GluN2D expression (pain/sensory) → sensory overload may trigger cortical spread. More prominent in infantile/toddler age", "management": "Sensory regulation (OT assessment); calm environment; rescue therapy available"},
    {"trigger": "AED taper / rapid reduction", "prevalence_pct": 42, "note": "Any rapid AED taper risks seizure cluster in DRE. Memantine taper in GOF patients is especially risky (tonic NMDA suppression withdrawal → burst)", "management": "Taper ≤10% per 2 weeks; never abrupt; GRIN2D GOF: memantine taper >6 weeks if reducing"},
    {"trigger": "Intercurrent illness / metabolic stress", "prevalence_pct": 38, "note": "Dehydration, electrolyte disturbances, metabolic acidosis (especially on KD) → membrane potential shifts → increased GOF NMDA activity", "management": "Sick-day plan: maintain AEDs, extra hydration; KD patients: monitor BHB + electrolytes during illness"},
    {"trigger": "Catamenial (in post-pubertal females)", "prevalence_pct": 28, "note": "Progesterone withdrawal → allopregnanolone drop → reduced tonic GABA-A inhibition → NMDA-GOF unopposed → perimenstrual cluster", "management": "CLB 10 mg/day days 23-28; ganaxolone (neurosteroid analogue) investigational; hormonal contraceptives for cycle regulation"},
]

# ── Treatments ───────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Memantine",
        "evidence": "Level C (case series; GOF ONLY)",
        "dose": "Start 0.3 mg/kg/day → titrate over 4-8 weeks to 5-10 mg/day (paediatric); adult 10-20 mg/day (split doses)",
        "moa": "Uncompetitive NMDA receptor open-channel blocker; preferentially blocks constitutively open GOF channels; moderate-affinity (Kd ~1 µM); rapid-offset kinetics (does not completely suppress physiological NMDA)",
        "efficacy": "~50% seizure reduction in published GRIN2D GOF cases; movement disorder improvement in some (reflects GluN2D subcortical expression); LOF: worsens — ABSOLUTE CI",
        "monitoring": "Cognitive/motor serial assessment (BAYLEY III in infants, VABS in older children); QTc at baseline and 4 weeks (marginal QTc effect); renal function annually (memantine renal excretion)",
        "grin2d_note": "GOF/LOF functional assay MANDATORY before prescribing. Monitor movement disorder — memantine may reduce dystonia AND seizures in GOF. Low-dose tonic block preferred (5 mg/day) over high doses to preserve physiological extrasynaptic NMDA tonic current.",
    },
    {
        "drug": "VPA (Valproate)",
        "evidence": "Level B (broad-spectrum DEE standard of care)",
        "dose": "10-60 mg/kg/day in 2-3 divided doses; TDM target 50-100 µg/mL",
        "moa": "NaV block, T-type Ca2+ block, GAD enhancement (↑GABA synthesis), weak GABA-T inhibition, histone deacetylase inhibition",
        "efficacy": "50-70% seizure reduction for myoclonic + GTCS components in GRIN2D. Does not directly modulate NMDA receptor.",
        "monitoring": "POLG1 full gene sequencing MANDATORY before VPA. LFT + FBC + NH3 at baseline, 3M, 6M, then annually. VPPP (valproate-related polycystic ovaries) in post-pubertal females.",
        "grin2d_note": "POLG exclusion is not optional — GRIN2D DEE phenotype mimics POLG Alpers (regression + seizures). VPA in undiagnosed POLG1 → fatal hepatic failure within weeks.",
    },
    {
        "drug": "ACTH (Corticotrophin)",
        "evidence": "Level A for associated West syndrome / epileptic spasms",
        "dose": "Standard West syndrome protocol: ACTH 40-80 IU/day IM for 2 weeks → taper. UK UKISS: high-dose ACTH vs vigabatrin (Lux 2004). NZ/AUS: tetracosactide (synthetic).",
        "moa": "Suppresses CRH (epileptogenic in immature brain); reduces neuroinflammation; promotes GABA-A surface expression; melanocortin 2R signalling in adrenal + brain",
        "efficacy": "60-70% hypsarrhythmia resolution at 2 weeks (Day-14 EEG gate). Spasm-free 50-60% at 6M. Less durable than TSC response; GRIN2D genetic etiology limits long-term outcome.",
        "monitoring": "BP + glucose daily (side effects: hypertension, Cushingoid, infection risk); Day-14 EEG for response gate. Electrolytes weekly. No live vaccines during ACTH.",
        "grin2d_note": "ACTH first-line for GRIN2D West (UKISS protocol). VGB adjunct (SHARE REMS visual field ERG q3M). Failure at day 14 → escalate to KD or memantine (if GOF confirmed).",
    },
    {
        "drug": "VGB (Vigabatrin)",
        "evidence": "Level A for associated West syndrome (UKISS/CHILD trial)",
        "dose": "100-150 mg/kg/day in 2 divided doses (infantile spasms); FDA-approved for IS",
        "moa": "Irreversible GABA-T inhibitor → ↑GABA in synaptic cleft → enhanced inhibitory tone",
        "efficacy": "50-60% spasm freedom in West syndrome. Inferior to ACTH alone but equivalent in UKISS when combined with ACTH. Not for GRIN2D focal or myoclonic seizures.",
        "monitoring": "ERG (electroretinogram) q6 months — irreversible concentric visual field loss in 30-50% long-term users (Goldman perimetry from age 5+). SHARE/REMS program mandatory in USA.",
        "grin2d_note": "Use ONLY for West syndrome / epileptic spasms in GRIN2D. NOT for focal or myoclonic seizures. Limit duration to 6 months in infants (visual field monitoring from age 5+). SHARE REMS programme mandatory.",
    },
    {
        "drug": "KD (Ketogenic Diet)",
        "evidence": "Level B for drug-resistant GRIN2D epilepsy",
        "dose": "4:1 ratio (fat:carb+protein) or MCT oil protocol; 1500-2000 kcal/day adjusted for age/weight",
        "moa": "β-hydroxybutyrate (BHB): adenosine A1 receptor agonist (inhibitory), NMDA receptor modulation (reduces Na+ current through NMDA), mTOR pathway suppression, mitochondrial biogenesis",
        "efficacy": "40-50% ≥50% seizure reduction in GRIN2D DRE at 3 months. Movement disorder may partially improve (BHB modulation of basal ganglia metabolism). Best in GOF with residual focal seizures post-memantine.",
        "monitoring": "BHB 2-4 mmol/L (urine/blood ketosis monitoring); lipid panel at 3M/6M; weight/growth; selenium/zinc/carnitine supplementation; renal ultrasound annually (nephrolithiasis 5-8%).",
        "grin2d_note": "Introduce after 2 AED failures (ILAE drug resistance criteria). GRIN2D GOF: KD + memantine combination may be synergistic (NMDA channel block + metabolic NMDA modulation). Avoid during KD illness days — risk of metabolic acidosis + seizure cluster.",
    },
    {
        "drug": "CLB (Clobazam)",
        "evidence": "Level B adjunct (focal + myoclonic seizures in DEE)",
        "dose": "0.2-0.3 mg/kg/day (children); adult 10-40 mg/day split; start low, titrate over 2-4 weeks",
        "moa": "1,5-benzodiazepine; preferential affinity for GABA-A receptors containing α2/α3 subunits (less sedation vs 1,4-BDZ like diazepam which is α1-preferring). GABA-A positive allosteric modulator.",
        "efficacy": "40-60% responder rate for myoclonic and focal seizures as adjunct. PRN use for movement disorder dystonic storms (GRIN2D-specific benefit — baclofen + CLB combination for dystonia).",
        "monitoring": "Sedation (dose-titrate); CYP2C19 interaction with SSRIs (fluoxetine/fluvoxamine → N-desmethyl-CLB 5×↑); tolerance with long-term use; withdrawal plan if tapering.",
        "grin2d_note": "CLB is preferred adjunct in GRIN2D movement disorder patients (1,5-BDZ → less motor sedation than 1,4-BDZ at equivalent anticonvulsant dose). Also useful PRN for catamenial exacerbation (days 23-28).",
    },
    {
        "drug": "LEV (Levetiracetam)",
        "evidence": "Level B adjunct (safe in GRIN2D; no POLG/NMDA interaction)",
        "dose": "10-60 mg/kg/day in 2 divided doses; TDM target 12-46 µg/mL",
        "moa": "SV2A vesicle release modifier → reduces presynaptic neurotransmitter release (glutamate + GABA). Does not directly modulate NMDA receptors.",
        "efficacy": "30-50% seizure reduction as adjunct for focal and GTCS in GRIN2D. Safe in LOF and GOF — no GOF/LOF-specific contraindication. Monitor for behavioural side effects (irritability, aggression — esp. in ASD comorbidity).",
        "monitoring": "Renal dosing (LEV 60-70% renal excretion); TDM if renal impairment; CGAS/VABS for behaviour; no LFT required.",
        "grin2d_note": "Safe choice regardless of GOF/LOF status. First-line adjunct while awaiting functional assay result. Behavioural side effects (irritability, agitation) are more common in GRIN2D patients with ASD comorbidity — switch to BRV (brivaracetam, same SV2A target, fewer behavioural effects) if needed.",
    },
    {
        "drug": "Baclofen (movement disorder adjunct — NOT anticonvulsant)",
        "evidence": "Level C (GRIN2D movement disorder; off-label)",
        "dose": "2.5-5 mg TID initially; children 0.3-0.75 mg/kg/day; adult 40-80 mg/day; ITB (intrathecal baclofen) for severe refractory dystonia",
        "moa": "GABA-B agonist → presynaptic inhibition of glutamate release in subthalamic nucleus (STN) → reduced excitatory drive to globus pallidus/output nuclei → reduces dystonic posturing. NOT an anticonvulsant.",
        "efficacy": "40-60% improvement in dystonia severity (BFMDRS) in GRIN2D GOF. Combination with trihexyphenidyl (anticholinergic) for mixed dystonia-choreoathetosis. ITB for severe cases unresponsive to oral.",
        "monitoring": "Sedation, muscle weakness, respiratory depression (high dose); abrupt withdrawal → life-threatening baclofen withdrawal syndrome (fever, seizures, rhabdomyolysis); renal dosing; ITB: pump site infection, catheter dislodgement",
        "grin2d_note": "Unique to GRIN2D movement disorder — reflects GluN2D STN/SN/VTA expression. Baclofen modulates the basal ganglia circuit that becomes dysregulated in GRIN2D GOF. NOT for seizure control. Taper slowly — withdrawal is dangerous.",
    },
]

# ── Contraindications ────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Memantine in LOF GRIN2D",
        "severity": "ABSOLUTE CI",
        "mechanism": "Further reduces NMDA activity in already-hypoactive LOF channels → worsens NMDA hypofunction-driven disinhibition → seizure worsening + cognitive decline",
        "clinical_consequence": "Cognitive regression, increased seizure frequency, worsening movement features",
        "alternative": "LEV + VPA (POLG screen mandatory) + CLB adjunct; baclofen for movement disorder",
        "grin2d_note": "GOF/LOF functional assay MANDATORY before prescribing memantine. Phenotypic distinction (West + severe movement disorder = more likely GOF; focal + milder = LOF) is insufficient — variants at same position can be GOF or LOF depending on effect on open probability.",
    },
    {
        "drug": "Ketamine (IV RSE) in LOF GRIN2D",
        "severity": "ABSOLUTE CI for RSE in LOF",
        "mechanism": "Ketamine is potent NMDA antagonist — additive with LOF-driven NMDA hypofunction → prolonged dissociative encephalopathy + worsened disinhibitory hyperexcitability after ketamine offset",
        "clinical_consequence": "Prolonged LOF-disinhibition encephalopathy; cognitive harm; potentially rebound seizure cluster on ketamine offset",
        "alternative": "RSE protocol in LOF: BDZ (Stage 1) → IV LEV 60 mg/kg (Stage 2) → IV VPA 40 mg/kg (Stage 3) → barbiturate coma (Stage 4). No ketamine.",
        "grin2d_note": "GOF RSE: ketamine is appropriate and potentially beneficial (NMDA block). LOF RSE: use barbiturate coma pathway instead. Document GOF/LOF status on the emergency plan.",
    },
    {
        "drug": "Tiagabine",
        "severity": "ABSOLUTE CI in GRIN2D DEE",
        "mechanism": "GAT-1 GABA reuptake inhibitor → excess tonic GABA → NCSE in dysmature/dysplastic cortex with altered GABAergic/NMDA balance. GRIN2D alters the excitatory/inhibitory ratio — TGB exacerbates this instability",
        "clinical_consequence": "Non-convulsive status epilepticus (NCSE); may be subclinical; fatal outcome reported in DEE patients",
        "alternative": "CLB (1,5-BDZ) is safe GABA-A enhancer; no GABA reuptake inhibitor in GRIN2D",
        "grin2d_note": "Absolute rule: no tiagabine in any GRIN2D patient regardless of seizure type. NCSE risk is independent of GOF/LOF status — GRIN2D alters excitatory/inhibitory balance in ways that make the cortex susceptible to TGB-induced NCSE.",
    },
    {
        "drug": "D-Cycloserine in GOF GRIN2D",
        "severity": "HIGH RISK in GOF",
        "mechanism": "D-cycloserine is a partial agonist at the glycine co-agonist site on GluN1 → increases P_open of NMDA receptors → worsens GOF hyperactivation in GRIN2D",
        "clinical_consequence": "Seizure worsening, increased excitotoxicity risk in GOF variants",
        "alternative": "Memantine (channel blocker, opposite mechanism) for GOF. D-cycloserine is investigational for LOF/hypofunction states.",
        "grin2d_note": "D-cycloserine has been explored in GRIN2B LOF / schizophrenia (NMDA hypofunction models). GRIN2D GOF is the opposite — avoid d-cycloserine. Ifenprodil is different: useless in GRIN2D (no NTD site), but not harmful.",
    },
    {
        "drug": "CBZ / OXC / PHT in generalised phenotype",
        "severity": "HIGH RISK",
        "mechanism": "Na-channel blockers (NaV-preferring) worsen generalised seizures via PV+ interneuron NaV1.1 block → GABAergic disinhibition → seizure aggravation (same mechanism as in Dravet/JME)",
        "clinical_consequence": "Paradoxical increase in seizure frequency; myoclonic worsening; spasm exacerbation",
        "alternative": "VPA (broad-spectrum); LEV; CLB. CBZ/OXC only if focal seizures are dominant AND GOF status excludes myoclonic/spasm component.",
        "grin2d_note": "GRIN2D patients with West syndrome + generalised phenotype: strict avoidance of NaV blockers. If focal seizures are predominant in LOF variant (no myoclonic component), CBZ/OXC may be used cautiously — but HLA-B*15:02 MANDATORY before CBZ/OXC in SE Asian ancestry.",
    },
    {
        "drug": "VPA without POLG1 exclusion",
        "severity": "ABSOLUTE CI",
        "mechanism": "POLG1 mitochondrial encephalopathy (Alpers syndrome) mimics GRIN2D DEE — same presenting features (infantile regression + seizures + NDD). VPA in POLG1 compound heterozygote/homozygote → mitochondrial respiratory chain dysfunction → fatal hepatic failure",
        "clinical_consequence": "Fatal acute liver failure; median 6 weeks from VPA initiation",
        "alternative": "Full POLG1 gene sequencing BEFORE VPA. LEV safe while awaiting result. If POLG1+: LEV + CLB + KD (bypass VPA entirely)",
        "grin2d_note": "POLG exclusion is mandatory for ALL patients with infantile/childhood-onset DEE before VPA — not just GRIN2D. The GRIN2D phenotype overlaps with POLG-Alpers sufficiently that this rule cannot be bypassed.",
    },
]

# ── Monitoring ───────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "GOF/LOF functional assay (NMDA receptor expression + electrophysiology)", "timing": "Before memantine/ketamine prescription; ideally at diagnosis", "rationale": "Determines precision therapy eligibility. Xenopus oocyte or HEK293 expression system + two-electrode voltage clamp. Variant functional data in ClinVar/GRIN-registry (UNC GRIN disorders)."},
    {"item": "POLG1 full gene sequencing", "timing": "Before VPA initiation (mandatory)", "rationale": "Alpers syndrome overlap; fatal VPA hepatotoxicity in POLG1+."},
    {"item": "Movement disorder severity (BFMDRS / Barry-Albright Dystonia Scale)", "timing": "Baseline + 3-monthly", "rationale": "GluN2D subcortical expression → movement disorder is major comorbidity; track separately from seizures; guides baclofen/trihexyphenidyl dosing."},
    {"item": "VPA TDM (target 50-100 µg/mL)", "timing": "2 weeks post-start, then 3-monthly", "rationale": "Narrow therapeutic window; multiple drug interactions."},
    {"item": "LFT + FBC + NH3 on VPA", "timing": "Baseline, 3M, 6M, annually", "rationale": "Hepatotoxicity monitoring; ammonia elevated in 30% on VPA (valproate-related hyperammonaemia — POLG1 must already be excluded)."},
    {"item": "VGB ERG / visual field (SHARE REMS / Goldman)", "timing": "Baseline + q6M on VGB; Goldman perimetry from age 5+", "rationale": "Irreversible concentric visual field loss; SHARE registration mandatory in USA for VGB."},
    {"item": "EEG Video-LTM (long-term monitoring)", "timing": "At baseline; after each treatment change; PRN for movement disorder vs seizure distinction", "rationale": "Critical for distinguishing non-ictal dystonia (no EEG correlate) from focal motor seizures (EEG correlate). GRIN2D-specific: both can occur together."},
    {"item": "Developmental/cognitive serial assessment (BAYLEY III, VABS, WPPSI, WISC)", "timing": "6-monthly (infants); annually (children)", "rationale": "Track developmental trajectory; measure memantine cognitive benefit in GOF; LEV behavioural side effects."},
    {"item": "MRI Brain (3T; basal ganglia protocol)", "timing": "At diagnosis; repeat at 12M if abnormal", "rationale": "GRIN2D GOF may show signal change in basal ganglia/subthalamic nucleus (excitotoxic injury). Rule out structural FCD (surgery candidate if FCD present)."},
    {"item": "ACTH monitoring (BP + glucose daily during ACTH course)", "timing": "Daily during 2-week ACTH course; Day-14 EEG", "rationale": "ACTH side effects: hypertension, Cushingoid, hyperglycaemia, infection risk. Day-14 EEG is GO/NO-GO gate for ACTH continuation."},
    {"item": "VPPP (Valproate-associated foetal abnormalities / MHRA 2021)", "timing": "Annual counselling in post-pubertal females on VPA", "rationale": "VPA teratogenicity (neural tube defects 10× ↑); pregnancy prevention programme mandatory in UK (MHRA 2021) + EU."},
    {"item": "SUDEP annual risk assessment", "timing": "Annually from diagnosis", "rationale": "GRIN2D GOF DRE patients with GTCS have elevated SUDEP risk. Document nocturnal seizure monitoring plan, safe sleeping position (supine), caregiver education."},
    {"item": "Baclofen withdrawal monitoring (if on baclofen)", "timing": "When reducing/stopping baclofen; inpatient if high-dose", "rationale": "Abrupt baclofen withdrawal → life-threatening withdrawal syndrome: fever, seizures, rhabdomyolysis, autonomic instability. Taper ≥10% per 1-2 weeks."},
    {"item": "KD metabolic panel (BHB, lipids, electrolytes, urine calcium:creatinine)", "timing": "Monthly ×3, then 3-monthly on KD", "rationale": "Ketosis monitoring (BHB 2-4 mmol/L); nephrolithiasis screening (urine Ca:Cr > 0.6 = risk); hyperlipidaemia monitoring."},
]

# ── Lifecycle ────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {"stage": "Neonatal / NICU (0-28 days)", "features": "Occasional neonatal-onset GOF with burst-suppression; seizures often subclinical; EEG monitoring MANDATORY if suspected. Movement abnormalities (limb jerking ± dystonia) may precede first clearly ictal event."},
    {"stage": "Infantile (1-12 months)", "features": "West syndrome onset (peak 4-6 months); hypsarrhythmia; epileptic spasms; ACTH initiation. Profound hypotonia in LOF. Movement disorder (dystonia) becomes apparent — separate from spasms. POLG + GOF/LOF assay urgently."},
    {"stage": "Toddler / Early Childhood (1-5 years)", "features": "Post-West evolution: focal motor seizures + GTCS; memantine initiation (if GOF confirmed); KD if DRE after 2 AEDs. Movement disorder trajectory — baclofen/trihexyphenidyl added for dystonia. Developmental delay profile emerges."},
    {"stage": "School Age (5-12 years)", "features": "DRE plateau or gradual improvement; KD long-term outcomes; memantine continues in GOF responders; cognitive assessment (WISC/WPPSI); special education plan; seizure freedom assessment (surgery if FCD identified on MRI)."},
    {"stage": "Adolescence (12-18 years)", "features": "VPPP mandatory in VPA females; catamenial exacerbation addressed (CLB days 23-28 / ganaxolone investigational); movement disorder may improve or plateau; puberty-related hormonal effects on GRIN2D expression; transition planning."},
    {"stage": "Adulthood (18+ years)", "features": "Continuation of established AEDs; SUDEP risk monitoring; vocational + social support (profound ID in most GRIN2D GOF); memantine cognitive benefit maintained; movement disorder management via neurologist + physiotherapy."},
]

# ── Key concepts ─────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "GRIN2D (19q13.33)", "definition": "Gene encoding GluN2D (NR2D), the slowest-deactivating NMDA receptor subunit. Predominantly subcortical/extrasynaptic expression (STN, SN, VTA, cerebellum, LC). De novo mutations cause infantile-onset DEE with movement disorder."},
    {"term": "GluN2D / NR2D subunit", "definition": "Fourth NMDA receptor regulatory subunit. Tau_decay ~4 seconds (slowest GluN2). Low Mg2+ sensitivity → partial tonic current at resting potential. Ifenprodil-insensitive (no NTD site). Ca2+ permeability lower than GluN2A/2B."},
    {"term": "Extrasynaptic Tonic NMDA Current", "definition": "GluN2D primarily mediates tonic (sustained, low-level) NMDA current at extrasynaptic sites, contrasting with phasic (burst) synaptic NMDA mediated by GluN2A/2B. GOF → excessive tonic NMDA activation in subcortical neurons."},
    {"term": "GOF Lurcher-Equivalent Mutations", "definition": "Mutations in GluN2D at positions homologous to the lurcher mutation (GluD2 Ala654Thr in cerebellar ataxia mice) — within M2/M3 transmembrane channel-lining domain. Dramatically increase open probability → constitutive NMDA current."},
    {"term": "Subcortical Movement Disorder (Dystonia/Choreoathetosis)", "definition": "Hallmark of GRIN2D GOF — basal ganglia (STN/SN) GluN2D expression makes these nuclei hyperactivated in GOF → dystonia, choreoathetosis, hyperkinesia. NON-ICTAL — no EEG correlate. Must be distinguished from focal motor seizures by Video-EEG."},
    {"term": "Memantine Precision Therapy", "definition": "Uncompetitive NMDA open-channel blocker. Selectively reduces GOF NMDA over-activity by blocking constitutively open channels. GOF ONLY — ABSOLUTE CI in LOF. GOF/LOF functional assay mandatory before prescribing."},
    {"term": "Ifenprodil Insensitivity (GRIN2D vs GRIN2B)", "definition": "Ifenprodil selectively binds GluN2B N-terminal domain (NTD) → GluN2B-specific block. GluN2D lacks this NTD binding site → ifenprodil has NO effect on GluN2D receptors. Do not confuse GRIN2D with GRIN2B when selecting NMDA antagonists."},
    {"term": "D-Cycloserine", "definition": "Partial agonist at GluN1 glycine co-agonist site → increases NMDA P_open. Investigational for GRIN2B/GRIN2A LOF (NMDA hypofunction). CONTRAINDICATED in GRIN2D GOF (increases already-excessive NMDA activation)."},
    {"term": "POLG1 Alpers Overlap", "definition": "POLG1 Alpers syndrome (mitochondrial encephalopathy) mimics GRIN2D DEE. VPA in POLG1+ → fatal hepatotoxicity. POLG1 full gene sequencing mandatory before VPA in all GRIN2D patients."},
    {"term": "Baclofen / STN modulation", "definition": "GABA-B agonist → reduces excitatory glutamate drive to STN output nuclei → reduces dystonic posturing. NOT an anticonvulsant. Treats the non-ictal movement disorder component of GRIN2D GOF. Abrupt withdrawal: life-threatening syndrome."},
    {"term": "GRIN2D vs GRIN2B comparison", "definition": "Both DEE genes. GRIN2B: synaptic, cortical, GluN2B (12p12.1), tau 300ms, ifenprodil-sensitive, West/Ohtahara dominant. GRIN2D: extrasynaptic, subcortical, GluN2D (19q13.33), tau 4s, ifenprodil-insensitive, movement disorder dominant."},
    {"term": "West Syndrome in GRIN2D", "definition": "Epileptic spasms + hypsarrhythmia in GRIN2D (55%). ACTH Level A. VGB adjunct (SHARE REMS). GRIN2D West may have asymmetric spasms (reflecting subcortical asymmetry). Memantine (if GOF) after West phase as adjunct."},
    {"term": "KD + Memantine Combination", "definition": "Proposed synergistic combination in GRIN2D GOF: memantine provides direct NMDA channel block; BHB (KD) modulates NMDA via adenosine A1 and metabolic pathways. Small case series suggest benefit; no RCT."},
    {"term": "SUDEP in GRIN2D DRE", "definition": "Sudden Unexpected Death in Epilepsy — elevated risk in GRIN2D GOF with DRE + GTCS. Subcortical GluN2D expression in autonomic nuclei (LC, VTA) may contribute to ictal autonomic dysregulation. Nocturnal monitoring + supine position essential."},
    {"term": "GRIN Disorders Network", "definition": "International registry at UNC Chapel Hill (Bhatt/Bhambhani) for GRIN1/GRIN2A/GRIN2B/GRIN2D epilepsy. Variant functional data, precision therapy matching, and trial access. Submit new GRIN2D cases for phenotype-genotype correlation."},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    "GOF/LOF assay: MANDATORY before memantine or ketamine prescription",
    "POLG1 full gene panel: MANDATORY before VPA (any GRIN2D patient)",
    "Memantine dose: start 0.3 mg/kg/day; target 5-10 mg/day paediatric (GRIN2D GOF only)",
    "VPA TDM target: 50-100 µg/mL (POLG excluded first)",
    "ACTH Day-14 EEG: GO/NO-GO gate for West syndrome — hypsarrhythmia resolution required",
    "VGB ERG: q6 months mandatory on vigabatrin (SHARE REMS program)",
    "KD ketosis: BHB 2-4 mmol/L; urine Ca:Cr > 0.6 → nephrolithiasis risk",
    "Baclofen taper: ≥10% reduction per 1-2 weeks (never abrupt — withdrawal syndrome)",
    "Drug resistance: 2 appropriate AED failures → DRE criteria (ILAE 2010) → KD + GOF/LOF-directed therapy",
    "SUDEP: annual counselling; nocturnal seizure monitoring; document safe-sleeping plan",
    "MRI: 3T basal ganglia protocol at diagnosis; repeat 12M if signal abnormality (excitotoxic change)",
    "Movement disorder: BFMDRS at baseline + 3-monthly; baclofen + trihexyphenidyl if BFMDRS > 20",
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE 2022 Classification of Seizure Types and Epilepsy Syndromes",
    "NICE NG217 Epilepsies in Children, Young People and Adults (2022)",
    "Li et al. 2019 Ann Neurol — GRIN2D GOF: first report of de novo GRIN2D mutations in DEE",
    "Bhatt et al. 2017 AJHG / Bhambhani 2022 Epilepsia — GRIN disorders network / registry",
    "CPIC POLG Guideline 2023 — VPA in POLG-associated disease (fatal hepatotoxicity)",
    "MHRA VPPP 2021 — Valproate pregnancy prevention programme (post-pubertal females)",
    "UKISS Trial (Lux 2004 Lancet) — ACTH vs VGB for West syndrome (Level A)",
    "FDA SHARE/REMS Vigabatrin — visual field monitoring q6 months mandatory",
    "ACMG-AMP 2015 — Variant pathogenicity classification (evidence-based; GOF/LOF evidence tier)",
    "WHO ICF-2019 — Disability framework for GRIN2D rehabilitation and outcome measurement",
    "ILAE Genetic Epilepsy Task Force 2022 — GRIN family classification",
    "EAN Neonatal Seizures Guideline 2019 — acute management in neonatal-onset GRIN2D",
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Li D et al. (2019) GRIN2D recurrent de novo dominant mutation causes a severe epileptic encephalopathy treatable with NMDA receptor channel blockers. Ann Neurol 86(5):686-698.",
    "Bhatt DL et al. (2022) GRIN disorders: genotype-phenotype correlation and precision therapies. Epilepsia 63(5):1073-1085.",
    "Lemke JR et al. (2016) Delineating the GRIN1 phenotypic spectrum: a distinct genetic epilepsy with intellectual disability. Nat Med 22(5):547-551.",
    "Hansen KB et al. (2018) Structure, function, and pharmacology of glutamate receptor ion channels. Pharmacol Rev 70(3):453-486.",
    "Ogden KK, Bhatt DL et al. (2014) Molecular pharmaology of the NMDA receptor. Adv Pharmacol 65:169-224.",
    "Lujan R & Bhatt DL (2023) GRIN2D-related epilepsy: subcortical expression, movement disorder, and memantine. Epilepsia Open 8(2):288-302.",
]


# ── API response functions ────────────────────────────────────────────────────
def get_overview():
    gof_n = sum(1 for p in PATIENTS if p["gof_lof"] == "GOF")
    lof_n = sum(1 for p in PATIENTS if p["gof_lof"] == "LOF")
    west_n = sum(1 for p in PATIENTS if p.get("west_syndrome"))
    dystonia_n = sum(1 for p in PATIENTS if p.get("dystonia_choreoathetosis"))
    dre_n = sum(1 for p in PATIENTS if p.get("drug_resistant"))
    kd_n = sum(1 for p in PATIENTS if p.get("on_kd"))
    mem_n = sum(1 for p in PATIENTS if p.get("memantine_rx"))
    polg_n = sum(1 for p in PATIENTS if p.get("polg_tested") == "Y")
    focal_n = sum(1 for p in PATIENTS if p.get("focal_seizures"))
    avg_onset = round(sum(p["onset_months"] for p in PATIENTS) / len(PATIENTS), 1)

    return {
        "gene": "GRIN2D",
        "locus": "19q13.33",
        "inheritance": "De novo dominant (predominantly GOF); rare LOF haploinsufficiency",
        "protein": "GluN2D (NR2D) — NMDA receptor subunit 2D; slowest-deactivating (tau ~4s); extrasynaptic/subcortical/basal ganglia expression; ifenprodil-insensitive",
        "mechanism": (
            "GOF (70%): constitutive NMDA activation in subcortical neurons (STN/SN/VTA/cerebellum) → "
            "excitotoxicity + basal ganglia dysregulation → DEE + movement disorder (dystonia/choreoathetosis). "
            "LOF (20%): tonic extrasynaptic NMDA loss → STN disinhibition → paradoxical hyperexcitability + focal DEE."
        ),
        "key_aha": (
            "GRIN2D GOF = DEE + MOVEMENT DISORDER (not just epilepsy). "
            "GOF/LOF assay MANDATORY — memantine TREATS GOF but WORSENS LOF. "
            "Ifenprodil is USELESS (no GluN2D NTD site — unlike GRIN2B). "
            "Non-ictal dystonia ≠ focal seizure: Video-EEG mandatory. "
            "Baclofen (STN) for movement disorder; memantine for seizures (GOF only). "
            "POLG exclusion before VPA."
        ),
        "n_patients": 40,
        "gof_pct": round(gof_n / 40 * 100),
        "lof_pct": round(lof_n / 40 * 100),
        "west_syndrome_pct": round(west_n / 40 * 100),
        "dystonia_choreoathetosis_pct": round(dystonia_n / 40 * 100),
        "drug_resistant_pct": round(dre_n / 40 * 100),
        "on_kd_pct": round(kd_n / 40 * 100),
        "memantine_rx_pct": round(mem_n / 40 * 100),
        "polg_done_pct": round(polg_n / 40 * 100),
        "focal_seizure_pct": round(focal_n / 40 * 100),
        "avg_onset_months": avg_onset,
        "contraindications_summary": [
            "Memantine in LOF GRIN2D — ABSOLUTE CI (worsens NMDA hypofunction → seizure + cognitive worsening)",
            "Ketamine IV in LOF GRIN2D RSE — ABSOLUTE CI (additive LOF worsening)",
            "Tiagabine — ABSOLUTE CI in GRIN2D DEE (NCSE in all GRIN2D variants)",
            "D-Cycloserine in GOF — HIGH RISK (increases NMDA P_open → worsens GOF)",
            "CBZ / OXC / PHT in generalised phenotype — HIGH RISK (NaV1.1 interneuron block → aggravation)",
            "VPA without POLG1 exclusion — ABSOLUTE CI (fatal hepatotoxicity in POLG+)",
        ],
        "thresholds": THRESHOLDS,
        "references": [r.split(".")[0] + " et al." for r in REFERENCES],
    }


def get_breakdown():
    gof_n = sum(1 for p in PATIENTS if p["gof_lof"] == "GOF")
    lof_n = sum(1 for p in PATIENTS if p["gof_lof"] == "LOF")
    west_n = sum(1 for p in PATIENTS if p.get("west_syndrome"))
    dystonia_n = sum(1 for p in PATIENTS if p.get("dystonia_choreoathetosis"))
    dre_n = sum(1 for p in PATIENTS if p.get("drug_resistant"))
    kd_n = sum(1 for p in PATIENTS if p.get("on_kd"))
    mem_n = sum(1 for p in PATIENTS if p.get("memantine_rx"))
    polg_not_tested = sum(1 for p in PATIENTS if p.get("polg_tested") == "N")

    return {
        "summary": {
            "n_patients": 40,
            "gof_pct": round(gof_n / 40 * 100),
            "lof_pct": round(lof_n / 40 * 100),
            "west_pct": round(west_n / 40 * 100),
            "dystonia_pct": round(dystonia_n / 40 * 100),
            "drug_resistant_pct": round(dre_n / 40 * 100),
            "kd_pct": round(kd_n / 40 * 100),
            "memantine_rx_pct": round(mem_n / 40 * 100),
            "polg_not_tested_count": polg_not_tested,
        },
        "etiology_distribution": ETIOLOGY_CATALOG,
        "patients_sample": PATIENTS,
        "seizure_types": [
            {"type": s["type"], "prevalence_pct": s["prevalence_pct"]}
            for s in SEIZURE_TYPES
        ],
        "seizure_detail": SEIZURE_TYPES,
        "triggers": [
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
            for t in TRIGGERS
        ],
        "trigger_detail": TRIGGERS,
        "treatment_detail": TREATMENTS,
        "contraindication_detail": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
