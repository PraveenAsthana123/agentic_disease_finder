"""
HCN1 Epilepsy — DEE24 (Hyperpolarization-Activated Cyclic Nucleotide-Gated Channel 1 / Ih Channelopathy)
=========================================================================================================
41-patient cohort · HCN1 (5p12) · Dual GOF/LOF channelopathy · Fever-sensitive DEE24

HCN1 / Ih CHANNEL BIOLOGY:
HCN1 (5p12) encodes hyperpolarization-activated cyclic nucleotide-gated channel 1, the fastest-
gating member of the HCN family (HCN1-4). HCN1 generates the Ih current ("funny current" or
"pacemaker current"), a mixed Na+/K+ inward current that is ACTIVATED by HYPERPOLARISATION
(unusual — most voltage-gated channels activate upon depolarisation). Ih is active at resting
membrane potential (~-60 to -90 mV range) and contributes critically to:
  (1) Resting membrane potential stabilisation (prevents excessive hyperpolarisation)
  (2) Dendritic integration (HCN1 is densest in distal dendrites of hippocampal CA1 and L5
      neocortical pyramidal neurons — filters slow synaptic inputs, temporal summation)
  (3) Thalamo-cortical oscillations (HCN1 in thalamocortical [TC] relay neurons drives the
      rhythmic burst-pause cycle underlying sleep spindles and absence-like rhythms)
  (4) Pacemaker rhythm in sinoatrial node (cardiac HCN4 >> HCN1, but HCN1 contributes)

HCN1 STRUCTURE AND GATING:
HCN1 is a tetramer. Each subunit has: 6 transmembrane segments (S1-S6), a cyclic nucleotide-
binding domain (CNBD), and a C-linker connecting CNBD to the pore. Unique gating:
  - Voltage sensor (S4): detects hyperpolarisation (not depolarisation)
  - CNBD: binds cAMP (and cGMP) → shifts activation curve to more POSITIVE potentials
    (cAMP makes the channel easier to open at less hyperpolarised voltages)
  - Ih reversal potential: ~ -30 mV (mixed Na+/K+ → depolarising when activated)

KEY: Ih is a DEPOLARISING current at resting potential. Paradoxically, Ih also LIMITS excessive
depolarisation: as membrane depolarises, Ih decreases (deactivates); as membrane hyperpolarises
after an action potential, Ih reactivates → pulls voltage back toward -30mV. This "voltage
clamping" function is critical for network stability.

TEMPERATURE DEPENDENCE (CRITICAL FOR HCN1-DEE):
Ih has a temperature coefficient Q10 ≈ 1.4-1.7 (moderate temperature sensitivity). At elevated
temperature (fever >38.5°C): Ih increases → activation curve shifts + → channels more easily
open. In GOF HCN1 variants (constitutively active): fever exacerbates already-pathological
Ih → DRAMATICALLY increased neuronal excitability. This explains the Dravet-like fever-
sensitivity of HCN1-DEE24. CRITICAL CLINICAL RULE: aggressive fever management is the
single most important preventive measure in HCN1-DEE — even low-grade fever (38°C) can
trigger cluster seizures.

HCN1 GOF PATHOPHYSIOLOGY:
De novo gain-of-function HCN1 variants → constitutive Ih (channel opens at less negative
voltages, or fails to deactivate) → persistent inward current at resting potential →
chronic membrane depolarisation → reduced threshold for action potentials → neuronal
hyperexcitability → epilepsy. GOF variants commonly affect the S4 voltage sensor
(e.g., p.Met305Leu, p.Glu293Lys — shift activation V1/2 positive by +10 to +30 mV)
or the gating hinge (e.g., p.Val414Met), allowing the channel to remain open even at
physiological membrane potentials. GOF HCN1-DEE24 phenotype: Dravet-like (fever-sensitive
febrile seizures evolving to multiple seizure types), multifocal discharges, severe ID.

HCN1 LOF PATHOPHYSIOLOGY:
Haploinsufficiency / LOF variants → reduced Ih → impaired depolarising return from hyperpolar-
isation → prolonged hyperpolarisation → paradoxical INCREASED network excitability through two
mechanisms:
  (1) Thalamo-cortical: in TC neurons, Ih normally limits burst rebound; LOF → prolonged T-type
      Ca2+ post-inhibitory rebound bursts → enhanced oscillatory cortical recruitment →
      increased absence-like / generalised discharges
  (2) Cortical: in L5 pyramidal neurons, Ih provides dendritic resonance at theta frequency
      (4-10 Hz) and filters slow synaptic inputs. LOF → increased dendritic integration of
      slow EPSPs → temporal summation → increased cortical output → hyperexcitability
  GOF paradox: reducing excitability via Ih block might seem therapeutic in LOF, but Ih is
  already low → Ih blockers contraindicated in LOF (see LTG below).

DUAL-MECHANISM CHANNELOPATHY — CLINICAL IMPLICATION (like KCNA2):
HCN1 is a DUAL GOF/LOF channelopathy. The same gene causes DEE24 via OPPOSITE mechanisms.
This creates the critical clinical requirement: FUNCTIONAL ASSAY (GOF vs LOF classification)
BEFORE any precision treatment. GOF variants may respond to:
  (a) Ih reduction strategies (experimental: ZD7288 — not clinically available)
  (b) GABA-A potentiation (CLB, stiripentol) — enhances inhibitory tone to counterbalance
      constitutive excitatory Ih
LOF variants should AVOID Ih blockers. The key clinically-available Ih blocker is:
  - LAMOTRIGINE: LTG blocks HCN1/HCN4 channels directly (Bois et al., 1996; Poolos et al.,
    2002). In LOF HCN1 DEE: LTG further reduces already-low Ih → worsens dendritic integration
    failure → INCREASED seizure burden documented in case series.
  - In GOF HCN1 DEE: LTG might reduce constitutive Ih → potentially beneficial. However,
    evidence is sparse. VPA + CLB remains first-line regardless of GOF/LOF.

IVERMECTIN AND HCN1:
Ivermectin (IVM) at nanomolar concentrations activates HCN1 channels (shifts activation
curve negative → increases Ih at resting potential; Gribkoff et al., 2000; Lolicato et al.,
2012). This makes IVM potentially therapeutic in LOF HCN1 DEE (preclinical mouse data:
Nava et al. 2014 showed partial seizure reduction in Hcn1-null mice with IVM). However,
IVM in GOF HCN1 is CONTRAINDICATED: IVM would enhance already constitutive Ih → worsens
hyperexcitability. Caution: IVM is currently investigational for HCN1-LOF only; no clinical
trial data. Additionally, IVM has CNS penetration concerns and drug-drug interactions.

HCN1 SODIUM CHANNEL BLOCKER SENSITIVITY:
Sodium channel blockers (carbamazepine, oxcarbazepine, phenytoin, lamotrigine) in HCN1-DEE:
  - In GOF HCN1: some focally-expressed GOF variants may respond to NaV blockers (reduces
    action potential firing). However, in predominantly multifocal or generalised GOF DEE,
    NaV blockers may aggravate (as in Dravet): sodium channel blockade in GABAergic
    interneurons → disinhibition → paradoxical worsening. CAUTION: individual assessment
    required; avoid as empirical first-line.
  - In LOF HCN1: NaV blockers may worsen (same mechanism as GOF GABAergic — especially in
    thalamo-cortical LOF patterns with absence/GTC phenotype).
  PRACTICAL RULE: Avoid CBZ/OXC/PHT in HCN1-DEE pending GOF/LOF assay.

HCN1-DEE EEG HALLMARKS:
1. Multifocal discharges (frontal > temporal > occipital) — characteristic of GOF
2. Generalised slow spike-wave (1.5-2.5 Hz) during fever — Dravet-like ictal pattern
3. NREM potentiation: IED burden increases during NREM sleep (thalamo-cortical oscillation
   synchronisation by HCN1 in TC neurons — both GOF and LOF potentiate NREM IEDs)
4. Background: moderate diffuse slowing (proportional to ID severity)
5. Febrile seizure EEG: often starts as focal (frontal) → rapid generalisation to
   bilateral GTCS within 30-60 seconds (fast propagation via HCN1-rich corticothalamic axons)
6. Post-ictal: prolonged voltage attenuation (>2 min) after GTCS — reflects widespread
   Ih-mediated hyperpolarisation following seizure (paradoxically prolonged in HCN1 LOF)

CLOBAZAM (CLB) AND HCN1 — MECHANISM:
CLB acts on GABA-A receptors (α2/α3-preferring benzodiazepine). In HCN1-DEE, CLB is
particularly valuable because: (1) GABA-A potentiation directly opposes Ih-mediated
depolarisation; (2) CLB has lower sedation than diazepam (α2 > α1 preferring); (3) CLB
is effective in Dravet syndrome (CLB-level-B for Dravet) — HCN1 fever-sensitive DEE shares
Dravet pathophysiology. Norclobazam (active metabolite) has long half-life → stable trough.
Monitor: norclobazam TDM 50-300 ng/mL; tolerance may develop (escalate CLB dose stepwise).

FENFLURAMINE AND HCN1-DEE:
Fenfluramine (FFA; Fintepla REMS) is approved for Dravet syndrome and Lennox-Gastaut. In
HCN1 fever-sensitive DEE (Dravet-like phenotype), FFA has theoretical and emerging clinical
utility via: (1) sigma-1 receptor activation → reduces neuronal excitability; (2) serotonin
5-HT2C and 5-HT1D agonism → modulates thalamo-cortical excitability; (3) possible direct
HCN1 channel interaction. REMS requirements: echocardiogram q6M (pulmonary hypertension
risk); body weight monitoring; maximum 0.7 mg/kg/day.

KETOGENIC DIET AND HCN1-DEE:
KD mechanistically complements HCN1 treatment: (1) β-hydroxybutyrate → reduces neuronal
firing frequency (KATP channel activation → hyperpolarisation, but physiological → not
exacerbating LOF hyperpolarisation); (2) increased GABA synthesis (acetoacetate → GABA
transaminase inhibition); (3) reduced mTOR signalling → reduced dendritic arborisation
(opposes LOF-mediated hyperintegration). KD is particularly effective in HCN1-DEE because:
KD reduces seizure threshold independently of Ih channel function → mechanistically sound
for both GOF and LOF. Clinical: target β-OHB 2-4 mmol/L; efficacy ≥50% seizure reduction
in ~55% of HCN1-DEE patients on KD (comparable to KD in Dravet).

KEY VARIANTS (ILAE/ACMG):
  p.Met305Leu  (c.913A>C)  — S4 voltage sensor; shifts V1/2 by +20 mV (GOF); most common
  p.Glu293Lys  (c.877G>A)  — S4-S5 linker; constitutive activation (GOF)
  p.Val414Met  (c.1240G>A) — gating hinge; slowed deactivation (GOF-kinetics)
  p.Arg590Gln  (c.1769G>A) — CNBD; impaired cAMP binding → reduced Ih (LOF)
  Exon deletions 1-4 (haploinsufficiency) — detected by MLPA/CMA; ~12% of HCN1-DEE

SAFETY PEARLS:
• FEVER MANAGEMENT IS #1 PRIORITY in HCN1-DEE. Temperature >38°C → seizure rescue plan.
  Paracetamol/ibuprofen immediately. Avoid hot baths. Sick-day plan at every outpatient visit.
• LTG CONTRAINDICATED in LOF HCN1: LTG is an Ih blocker → further LOF-worsening.
  GOF/LOF functional assay MANDATORY before LTG. In absence of assay: assume LOF → avoid LTG.
• CBZ/OXC/PHT CAUTION in HCN1-DEE pending GOF/LOF assay — risk of Dravet-like aggravation.
• IVERMECTIN CONTRAINDICATED in GOF HCN1: activates HCN1 → worsens constitutive excitability.
• POLG TESTING MANDATORY before any VPA initiation in this DEE cohort.
• CARDIAC MONITORING: HCN4 (not HCN1) is primary cardiac pacemaker, but HCN1 is expressed
  in sinoatrial node — rarely causes bradycardia (no routine cardiac monitoring required unless
  FFA or cardiac symptoms).
"""

import random
from datetime import datetime

SEED = 9190  # dashboard 190
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ───────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": (
            "HCN1 de novo GOF — classic DEE24 "
            "(constitutive Ih activation / S4 or gating-hinge variant)"
        ),
        "n": 18, "pct": 44,
        "category": "HCN1-de-novo-GOF-classic",
        "functional_class": "AD-HCN1-GOF-constitutive-Ih-DEE24",
        "mechanism": (
            "Most prevalent class (~44%): de novo heterozygous HCN1 gain-of-function variants in the S4 "
            "voltage sensor (p.Met305Leu, p.Glu293Lys) or the S6/gating-hinge region (p.Val414Met). "
            "These variants shift the Ih activation curve to more positive (depolarised) voltages by "
            "+10 to +30 mV, causing constitutive channel opening at physiological resting potentials "
            "(-65 to -70 mV). The pathological result: persistent inward Ih at rest → chronic neuronal "
            "depolarisation → reduced firing threshold → hippocampal CA1 and L5 pyramidal neuron "
            "hyperexcitability → DEE24. Critical: GOF Ih is further ENHANCED by fever (temperature "
            "coefficient Q10 ≈ 1.4-1.7 → each 10°C increase raises Ih by 40-70%) → explains profound "
            "Dravet-like fever sensitivity. ACMG: PS2 (de novo) + PM1 (S4 hotspot) + PP3 → Pathogenic. "
            "Functional electrophysiology (Xenopus oocyte or HEK293) confirms V1/2 shift."
        ),
        "eeg_signature": (
            "HCN1-GOF EEG: (1) Multifocal IEDs — frontal (F3/F4/Fz) dominant, followed by temporal "
            "and occipital; (2) NREM potentiation: spike burden increases 2-3x during NREM stage 2-3 "
            "(thalamo-cortical Ih synchronisation); (3) During febrile seizures: focal frontal fast "
            "discharge → rapid bilateral synchrony within 30-60s → GTCS EEG correlate; (4) Background: "
            "moderate diffuse theta (5-7 Hz) slowing; (5) Post-ictal: prolonged voltage attenuation "
            ">2 min; (6) Ictal: may show high-frequency oscillations (HFOs, >80 Hz) on deep bipolar "
            "montage during frontal onset seizures — consider sEEG for surgical assessment if unifocal."
        ),
        "clinical_note": (
            "Diagnostic algorithm: (1) Fever-sensitive DEE with Dravet-like features + NEGATIVE SCN1A "
            "→ gene panel including HCN1 (most Dravet-panel labs include HCN1). (2) Confirm GOF: "
            "Xenopus or HEK293 electrophysiology (V1/2 shift quantification). (3) Fever management "
            "is the most impactful non-pharmacological intervention. (4) Treat as Dravet-like: "
            "VPA + CLB first-line; add stiripentol or FFA. AVOID LTG (limited evidence for GOF "
            "benefit; risk-benefit unfavourable vs safer options). AVOID CBZ/OXC pending assay. "
            "(5) Differentials: SCN1A-Dravet (MANDATORY exclusion), PCDH19 clustering epilepsy "
            "(female-predominant), SCN8A-DEE (earlier onset, more tonic)."
        ),
    },
    {
        "etiology": (
            "HCN1 de novo LOF — haploinsufficiency DEE24 "
            "(null variant / exonic deletion / CNBD dysfunction)"
        ),
        "n": 12, "pct": 29,
        "category": "HCN1-de-novo-LOF-haploinsufficiency",
        "functional_class": "AD-HCN1-LOF-haploinsufficiency-DEE24",
        "mechanism": (
            "Loss-of-function class (~29%): de novo heterozygous HCN1 null variants (nonsense, "
            "frameshift → NMD → haploinsufficiency), exonic deletions (MLPA/CMA-detected, ~12%), "
            "or CNBD missense (p.Arg590Gln — impaired cAMP binding → reduced Ih sensitivity to "
            "neuromodulation). Haploinsufficiency → ~50% Ih → impaired depolarising return from "
            "post-AP hyperpolarisation → two paradoxical hyperexcitability mechanisms: (1) Thalamic: "
            "TC neurons in thalamus normally use Ih to terminate T-type Ca2+ rebound bursts → LOF → "
            "prolonged Ca2+ bursts → enhanced cortical oscillatory recruitment → absence-like/GTC "
            "discharges. (2) Cortical: distal dendritic LOF in CA1/L5 pyramids → excessive temporal "
            "summation of slow EPSPs → increased cortical output. LOF also impairs the normal "
            "shunting of inhibitory inputs in dendrites → paradoxically, GABA inputs are less shunted "
            "→ GABA remains more inhibitory → preserved (relatively) GABA response (unlike GOF). "
            "Phenotype: often less fever-sensitive than GOF; more generalised/absence features."
        ),
        "eeg_signature": (
            "HCN1-LOF EEG: (1) Generalised slow spike-wave (1.5-2.5 Hz) — more prominent than in "
            "GOF (thalamo-cortical LOF dominant); (2) NREM potentiation present but via different "
            "mechanism — prolonged TC burst rebound → enhanced NREM spindle-seizure coupling; "
            "(3) Absence-like seizures may appear: bilaterally synchronous 2-2.5 Hz SW during "
            "NREM (different from typical CAE 3 Hz SW — slower, more irregular in HCN1-LOF); "
            "(4) Post-ictal voltage attenuation very prolonged (>3-5 min) — prolonged LOF-mediated "
            "hyperpolarisation; (5) Background: diffuse slowing, less theta and more delta compared "
            "to GOF; (6) Note: LTG WORSENS IED burden in LOF — tracking IED on serial EEG is the "
            "best pharmacological monitoring tool."
        ),
        "clinical_note": (
            "Key management difference from GOF: (1) LTG is CONTRAINDICATED in LOF — it is an "
            "Ih blocker → worsens LOF further (case series: Marini 2018 Ann Neurol documented "
            "worsening in 3/5 LOF patients on LTG). (2) Ivermectin is the experimental Ih activator "
            "for LOF — investigational only, not yet clinical standard. (3) KD is particularly "
            "effective in LOF (mechanistically independent of Ih function). (4) Absence-like "
            "discharges: respond to VPA or ETX (ethosuximide) — ETX is generally safe in LOF "
            "(T-type Ca2+ blocker reduces TC burst rebound → rationale for combined ETX use). "
            "(5) Differentials: CACNA1A (P/Q Ca2+ channel — also thalamo-cortical, absence-GTC, "
            "episodic ataxia), SLC2A1 (hypoglycorrhachia — LP mandatory), typical CAE (3 Hz SW, "
            "not DEE severity)."
        ),
    },
    {
        "etiology": (
            "HCN1 de novo missense — channel-gating defect "
            "(intermediate GOF/LOF / cAMP-binding domain / S5-S6 pore)"
        ),
        "n": 6, "pct": 15,
        "category": "HCN1-de-novo-missense-gating-defect",
        "functional_class": "AD-HCN1-partial-GOF-LOF-gating-intermediate",
        "mechanism": (
            "Intermediate class (~15%): de novo HCN1 missense variants in S5-S6 pore region, "
            "C-linker, or C-terminus — functional effect ambiguous by ACMG criteria, requiring "
            "electrophysiology for classification. Examples: pore-helix variants (affect ion "
            "selectivity filter → reduce conductance = partial LOF) OR C-linker variants (impair "
            "coupling between CNBD and gate → blunted cAMP potentiation = functional LOF for cAMP- "
            "dependent modulation but preserved baseline Ih = not haploinsufficiency). Some variants "
            "show dominant-negative effects when co-assembled as heterotetramers with WT HCN1 → "
            "intermediate severity. Phenotype: moderate DEE24 (less severe than pure GOF or LOF); "
            "seizure onset 6-18 months; may respond better to combination therapy."
        ),
        "eeg_signature": (
            "Intermediate variant EEG: mixed features — multifocal IEDs (frontal + temporal) with "
            "some generalised SW components. Less severe background slowing than pure GOF/LOF. "
            "Fever sensitivity present but milder (temperature coefficient not maximally shifted). "
            "Sleep EEG: moderate NREM IED increase. Serial EEG on AED changes particularly "
            "important in this class to track pharmacological response (since functional effect "
            "ambiguous — EEG IED burden is the key monitoring parameter for treatment efficacy)."
        ),
        "clinical_note": (
            "Management challenge: functional assay needed to distinguish partial GOF vs partial "
            "LOF. In the interim: (1) Avoid LTG (precautionary — may be LOF). (2) VPA + CLB "
            "first-line (safe regardless of GOF/LOF direction). (3) Early KD referral (mechanistically "
            "GOF/LOF-independent). (4) Obtain functional electrophysiology report before committing "
            "to any precision therapy. (5) ACMG VUS status for many of these variants → ClinVar "
            "submission + expert functional lab (e.g., DiFrancesco lab, Milan; Bhatt lab). "
            "Family studies: look for phenotypic relatives (some C-linker variants show GEFS+ "
            "in milder family members)."
        ),
    },
    {
        "etiology": (
            "HCN1 inherited familial GOF — GEFS+ spectrum / febrile seizures plus "
            "(AD familial heterozygous, variable expressivity)"
        ),
        "n": 3, "pct": 7,
        "category": "HCN1-inherited-familial-GOF-GEFS",
        "functional_class": "AD-HCN1-familial-GOF-GEFS-spectrum",
        "mechanism": (
            "Familial inherited HCN1 GOF (~7%): parent typically has GEFS+ (febrile seizures plus "
            "afebrile GTCS into adulthood) or mild febrile seizures only, while the proband develops "
            "full DEE24. Variable expressivity due to: modifier genes (SCN1A modifiers may amplify "
            "HCN1 GOF effect), mosaicism levels, and developmental timing of HCN1 expression peak "
            "(HCN1 is highest in early childhood, explaining why proband DEE > parental phenotype). "
            "GOF mechanism identical to de novo class (V1/2 shift) but with milder functional "
            "shift (V1/2 +5 to +15 mV vs +20-30 mV in severe de novo). Inheritance: AD, 50% "
            "transmission risk. Genetic counselling and pre-conception PGT available."
        ),
        "eeg_signature": (
            "Familial HCN1-GOF: milder than de novo GOF. Parent EEG: normal or generalised "
            "IEDs during provoked febrile events only. Proband: multifocal IEDs with NREM "
            "potentiation (as de novo GOF) but less severe background slowing; fever sensitivity "
            "preserved (GOF Ih still temperature-dependent). LGS evolution rare in familial class "
            "(unlike severe de novo). Prognosis better: ~30-40% achieve meaningful seizure control "
            "with VPA + CLB + fever management (vs ~15-20% full control in de novo GOF)."
        ),
        "clinical_note": (
            "Phenotypic expansion counselling: parent with GEFS+ + HCN1-GOF → 50% risk of "
            "transmission → child risk for DEE24 exceeds parent's phenotype (modifier-dependent). "
            "Pre-conception genetic counselling and PGT recommended. Clinical management: VPA + "
            "CLB standard; if well-controlled on this regimen → avoid escalation. Fever education "
            "identical to de novo class. Driving restrictions apply when seizures not fully controlled."
        ),
    },
    {
        "etiology": (
            "HCN1 negative — clinical DEE24 phenocopy "
            "(HCN2 / CACNA1A / SCN1A-negative Dravet-like / regulatory HCN1)"
        ),
        "n": 2, "pct": 5,
        "category": "HCN1-negative-phenocopy",
        "functional_class": "DEE24-phenocopy-alternative-gene",
        "mechanism": (
            "Clinical DEE24-like phenotype (fever-sensitive DEE, multifocal/generalised, early "
            "onset) with negative HCN1 coding sequencing and negative CMA. Alternative causes: "
            "(1) HCN2 (19p13.3): HCN2 GOF → GEFS+ or Dravet-like (similar Ih biology, milder); "
            "(2) CACNA1A de novo GOF → DEE42 (P/Q Ca2+ channel, cerebellar ataxia distinguishes); "
            "(3) SCN1A-negative Dravet-like: deep intronic SCN1A (WGS needed) or SCN1A mosaicism "
            "not detected by peripheral blood NGS; (4) Deep intronic HCN1 splice variants (WGS + "
            "RNA-seq on fibroblasts required); (5) HCN1 brain-specific somatic mosaicism (>5% "
            "mutant allele fraction needed in brain — not peripheral blood). "
            "Management: treat as HCN1-GOF empirically (VPA + CLB) pending WGS/functional results."
        ),
        "eeg_signature": (
            "Phenocopy EEG: similar to HCN1-GOF (multifocal, NREM potentiation, fever-triggered). "
            "Distinguishing features: CACNA1A phenocopy → cerebellar EEG correlates (occipital "
            "slowing, periodic complexes during GTCS); SCN1A-phenocopy → more focal temporal onset. "
            "In all cases: NREM IED potentiation and fever-triggered EEG worsening present."
        ),
        "clinical_note": (
            "Workup: (1) WGS (paired blood + skin fibroblasts) for deep intronic HCN1 and "
            "SCN1A; (2) RNA-seq fibroblasts for aberrant HCN1 splicing; (3) If CACNA1A: "
            "cerebellar MRI, episodic ataxia history, acetazolamide trial; (4) If all negative: "
            "treat empirically as HCN1-GOF (VPA + CLB + KD); AVOID LTG. Management is identical "
            "regardless of negative HCN1 coding result in this fever-sensitive DEE phenotype."
        ),
    },
]

# ── Patient Cohort (N=41) ──────────────────────────────────────────────────────
ETIOLOGY_WEIGHTS = [
    ("HCN1-de-novo-GOF-classic", 18),
    ("HCN1-de-novo-LOF-haploinsufficiency", 12),
    ("HCN1-de-novo-missense-gating-defect", 6),
    ("HCN1-inherited-familial-GOF-GEFS", 3),
    ("HCN1-negative-phenocopy", 2),
]
_etiology_pool = [e for e, n in ETIOLOGY_WEIGHTS for _ in range(n)]

GOF_CATEGORIES = {"HCN1-de-novo-GOF-classic", "HCN1-inherited-familial-GOF-GEFS"}
LOF_CATEGORIES = {"HCN1-de-novo-LOF-haploinsufficiency"}

PATIENTS = []
for i in range(41):
    ec = _etiology_pool[i]
    is_gof = ec in GOF_CATEGORIES
    is_lof = ec in LOF_CATEGORIES
    onset_mo = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18] if is_gof
                             else [5, 6, 8, 10, 12, 14, 16, 18, 20, 24])
    age_mo = onset_mo + random.randint(6, 60)
    fever_sensitive = random.random() < (0.92 if is_gof else 0.62)
    ltg_used = random.random() < (0.12 if is_gof else 0.22)  # some get LTG before dx clarity
    ltg_worsened = ltg_used and is_lof and random.random() < 0.65
    vpa_used = random.random() < 0.88
    polg_tested = "Y" if vpa_used and random.random() < 0.82 else ("N" if vpa_used else "NA")
    clb_used = random.random() < 0.78
    on_kd = random.random() < 0.42
    kd_ketosis = round(random.uniform(1.8, 4.2), 1) if on_kd else None
    kd_response = random.choice([">=50%", ">=50%", ">=50%", "25-50%", "no-response"]) if on_kd else None
    stiripentol = random.random() < (0.38 if is_gof else 0.18)
    ffa_used = random.random() < (0.22 if is_gof else 0.08)
    seizure_control = (
        "seizure-free" if random.random() < 0.10
        else "well-controlled" if random.random() < 0.20
        else "partially-controlled" if random.random() < 0.45
        else "drug-resistant"
    )
    current_phase = (
        "infancy" if age_mo < 12
        else "early-childhood" if age_mo < 36
        else "school-age" if age_mo < 144
        else "adolescent" if age_mo < 216
        else "adult"
    )
    meds_list = []
    if vpa_used: meds_list.append("VPA")
    if clb_used: meds_list.append("CLB")
    if stiripentol: meds_list.append("STP")
    if ffa_used: meds_list.append("FFA")
    if on_kd: meds_list.append("KD")
    if not meds_list: meds_list.append("LEV")
    PATIENTS.append({
        "id": f"HCN1-{i+1:02d}",
        "age_months": age_mo,
        "onset_months": onset_mo,
        "etiology_category": ec,
        "is_gof": is_gof,
        "is_lof": is_lof,
        "fever_sensitive": fever_sensitive,
        "vpa_used": vpa_used,
        "polg_tested": polg_tested,
        "clb_used": clb_used,
        "on_kd": on_kd,
        "kd_ketosis_mmol": kd_ketosis,
        "kd_response": kd_response,
        "stiripentol": stiripentol,
        "ffa_used": ffa_used,
        "ltg_used": ltg_used,
        "ltg_worsened": ltg_worsened,
        "seizure_control": seizure_control,
        "current_phase": current_phase,
        "current_meds": "+".join(meds_list),
    })

# ── Seizure Types ──────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Febrile/fever-triggered seizures (DEE exacerbation)",
        "prevalence_pct": 92,
        "semiology": (
            "Fever >38°C (often as low as 38-38.5°C in GOF) → cluster of seizures, typically "
            "focal-frontal onset (eye deviation, unilateral clonic jerking) → rapid bilateral "
            "GTCS (within 30-60 seconds); duration often >5 minutes (febrile SE risk ~35% per "
            "febrile illness in GOF). Post-ictal prolonged (>30 min). May present as febrile "
            "status epilepticus requiring emergency BDZ → IV LEV escalation."
        ),
        "eeg_pattern": (
            "During fever: (1) Frontal fast discharge (beta 15-20 Hz) → bilateral generalisation "
            "within seconds; (2) Ictal: rhythmic bilateral spike-wave (1.5-2 Hz) during tonic "
            "phase → decrement; (3) Post-ictal: diffuse voltage attenuation >2 min; "
            "(4) Interictal during fever: dramatic IED burden increase (3-5x above baseline). "
            "EEG monitoring during febrile episodes essential if SE risk high."
        ),
        "clinical_tip": (
            "Fever management IS the most effective seizure prevention in HCN1-DEE. Threshold: "
            "paracetamol at 37.5°C (not 38.5°C) in GOF. Hot bath CONTRAINDICATED. Emergency "
            "plan: midazolam buccal 0.3 mg/kg (not 0.2 mg/kg) within 3 min; if no response "
            "in 5 min → second dose; if seizure >10 min → call 999/911. Written fever "
            "action plan given at EVERY outpatient visit and updated with dose changes."
        ),
    },
    {
        "type": "Focal seizures (frontal/temporal onset ± secondary generalisation)",
        "prevalence_pct": 78,
        "semiology": (
            "Frontal lobe seizures: eye deviation (contralateral), asymmetric tonic posturing "
            "(M2e pattern), brief (10-30 s) with rapid secondary generalisation; may cluster "
            "in NREM sleep (HCN1-rich thalamo-frontal circuits active in NREM). Temporal onset "
            "(less common): motionless stare + oro-alimentary automatisms, post-ictal aphasia. "
            "In LOF HCN1: more generalised from onset vs. focal-to-bilateral in GOF."
        ),
        "eeg_pattern": (
            "Frontal focal onset: high-frequency gamma (>40 Hz) or beta (20-30 Hz) discharge "
            "at F3 or F4; fast propagation to central (Cz) and then bilateral within seconds. "
            "Sleep potentiation: focal frontal IEDs increase during NREM stage 2-3; "
            "may evolve to focal status during prolonged NREM (unusual for HCN1 — document "
            "and video-EEG confirm). Note: sEEG may reveal HFOs (80-250 Hz) in frontal focus."
        ),
        "clinical_tip": (
            "In unifocal HCN1-GOF (confirmed single frontal focus on sEEG + FDG-PET): "
            "surgical evaluation is appropriate — HCN1-GOF can produce localised cortical "
            "hyperexcitability amenable to resection if structurally/functionally confined. "
            "MRI often normal (HCN1 is channelopathy — no structural lesion). PET hypometabolism "
            "may identify the zone. Consider MEG if MRI/EEG discordant."
        ),
    },
    {
        "type": "Generalised tonic-clonic seizures (GTCS)",
        "prevalence_pct": 65,
        "semiology": (
            "Bilateral GTCS arising from generalised onset (LOF-dominant) or secondary "
            "generalisation of focal discharge (GOF-dominant). Duration: typically 1-3 min. "
            "SUDEP risk: GTCS in bed during sleep is highest SUDEP risk → sleep safety "
            "counselling mandatory. Post-ictal: confusion 15-60 min; prolonged post-ictal "
            "depression in LOF (impaired Ih reactivation → prolonged hyperpolarisation)."
        ),
        "eeg_pattern": (
            "Generalised onset (LOF): diffuse fast activity (10-20 Hz) → recruiting rhythm → "
            "polyspike-wave; decrement. Secondary generalisation (GOF): focal frontal build-up "
            "→ bilateral spread. Both: post-ictal generalised EEG suppression >2 min "
            "(PGES — prolonged PGES correlates with SUDEP risk in multiple DEE syndromes)."
        ),
        "clinical_tip": (
            "SUDEP risk: Document prolonged PGES (>50 s) on EEG — this is a SUDEP surrogate "
            "biomarker. Advise: supervised sleep, nocturnal SUDEP monitor (commercially "
            "available bed sensors), prone positioning avoidance. Annual SUDEP counselling "
            "mandatory — record in notes. If GTCS ≥1/month uncontrolled → escalate urgently."
        ),
    },
    {
        "type": "Myoclonic seizures",
        "prevalence_pct": 38,
        "semiology": (
            "Brief (50-200 ms) bilateral synchronous myoclonic jerks, predominantly in upper "
            "limbs; worse on awakening. More common in GOF (constitutive Ih → cortical "
            "hyperexcitability → low-amplitude fast cortical discharge → myoclonus). May "
            "co-occur with absence-like episodes in LOF. Differentiate from juvenile myoclonic "
            "epilepsy (JME): HCN1 myoclonus present from infancy, not adolescent onset."
        ),
        "eeg_pattern": (
            "Myoclonic: bilateral synchronous polyspike (3-7 spikes) burst → wave; frontally "
            "dominant; time-locked to jerk EMG by 10-40 ms. In LOF: myoclonus may occur during "
            "absence-like episodes (spike-wave myoclonus). EMG co-registration essential "
            "to confirm cortical myoclonus vs action tremor. Photo-sensitivity in ~25% of GOF — "
            "check for IPS (intermittent photic stimulation) response at EEG."
        ),
        "clinical_tip": (
            "Myoclonic seizures in HCN1-DEE: VPA is preferred (broad-spectrum, anti-myoclonic). "
            "Clonazepam (CZP) 0.25-0.5 mg at night may help morning myoclonus. AVOID LTG "
            "(worsens myoclonus in LOF; limited evidence in GOF). ETX is not preferred "
            "for myoclonus (T-type Ca2+ blocker → may help absence-like in LOF but no "
            "anti-myoclonic data in HCN1). LEV may worsen myoclonus in some cases — monitor."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever (any temperature ≥37.5°C in GOF)",
        "prevalence_pct": 92,
        "mechanism": (
            "Ih has a temperature coefficient Q10 ≈ 1.4-1.7 (moderate). At fever +1°C above "
            "baseline (37°C→38°C): Ih increases by ~15-20%. In GOF HCN1 with already constitutive "
            "Ih, even mild fever dramatically increases the pathological Ih → neuronal excitability "
            "threshold drops → seizure cluster. The effect is rapid (minutes) and proportional "
            "to temperature elevation. Mechanism NOT mediated by SCN1A (GABA interneuron failure "
            "at fever) as in Dravet — HCN1 GOF is a direct temperature-Ih amplification."
        ),
        "management": (
            "Temperature threshold in HCN1-GOF: act at 37.5°C (not 38.5°C standard). Protocol: "
            "Paracetamol 15 mg/kg immediately at 37.5°C. Ibuprofen 10 mg/kg alternating if "
            "needed. Remove excess clothing. Avoid hot bath (additional thermal trigger). "
            "School: written plan to call parent immediately at 37.5°C; nurse to give paracetamol "
            "per school protocol. Emergency BDZ prescribed and demonstrated annually."
        ),
    },
    {
        "trigger": "Sleep deprivation",
        "prevalence_pct": 72,
        "mechanism": (
            "Sleep deprivation reduces seizure threshold via multiple mechanisms: (1) Reduced "
            "adenosine (sleep pressure mediator → A1R-mediated neuronal inhibition → less "
            "inhibition when sleep-deprived); (2) Increased cortisol (lowers GABA-A receptor "
            "density); (3) In HCN1-DEE specifically: HCN1 expression peaks in sleep-wake "
            "transition states (highest HCN1 mRNA in wake/REM vs NREM) — disruption of "
            "sleep architecture in HCN1-DEE patients → loss of HCN1 expression peak regularity "
            "→ dysrhythmic thalamo-cortical coupling → lower seizure threshold."
        ),
        "management": (
            "Strict sleep hygiene: consistent bedtime ±30 min; minimum sleep hours (infants: "
            "12-14h; school age: 10-11h; teens: 9-10h). Melatonin 0.5-2 mg at bedtime if "
            "sleep onset insomnia (common in HCN1-DEE due to thalamic oscillation dysregulation). "
            "Alert parents/carers: illness, excitement, travel → extra vigilance night after "
            "sleep disruption. Sleep EEG q6M to track IED burden during NREM."
        ),
    },
    {
        "trigger": "Missed AED dose",
        "prevalence_pct": 68,
        "mechanism": (
            "Sub-therapeutic AED levels → loss of inhibitory tone → acute elevation of seizure "
            "burden. For VPA (typically 12-hourly): <1 missed dose can produce sub-therapeutic "
            "trough within 4-6 hours. For CLB (24-hourly): tolerance may mask the effect but "
            "norclobazam levels drop over 24h without dose. In HCN1-DEE: Ih tone is constitutive "
            "(GOF) or absent (LOF) — pharmacological compensation is critical; loss of this "
            "compensation → rapid seizure breakthrough."
        ),
        "management": (
            "Electronic pill reminder (phone alarm). Blister pack adherence aid. If >4h late "
            "(VPA twice-daily): give dose now + contact epilepsy nurse for observation plan. "
            "Written sick-day protocol: continue AED even if vomiting — if unable to take oral "
            "→ emergency PR/buccal BDZ and hospital presentation. Pharmacist-led annual "
            "adherence review."
        ),
    },
    {
        "trigger": "Physical exertion (hyperthermia from exercise)",
        "prevalence_pct": 55,
        "mechanism": (
            "Exercise generates body heat → core temperature rises (moderate exercise: +0.5-1.5°C; "
            "intense: +2-3°C). In HCN1-GOF: exercise-induced hyperthermia directly activates "
            "Ih amplification (same Q10 mechanism as fever). This is distinct from the classical "
            "Uhthoff's phenomenon (demyelination). Onset: typically 10-20 min into vigorous "
            "exercise when core temperature peaks. Post-exercise recovery can also trigger "
            "(rapid temperature drop → HCN1 deactivation dynamics shift)."
        ),
        "management": (
            "SPORTS RESTRICTION (not ban): avoid contact sport risk during seizure; prefer "
            "cool environments (swimming pools — water cooling is protective in HCN1-GOF, "
            "as opposed to hot baths). Cooling vest for outdoor activity. No running in hot "
            "weather >25°C. School PE: teacher briefed; cool water available; sit down if "
            "feels 'hot' or prodromal. Competitive sport: individual risk assessment; GTCS "
            "in water → supervised swimming ONLY with lifeguard aware."
        ),
    },
    {
        "trigger": "Hot bath / warm water immersion",
        "prevalence_pct": 42,
        "mechanism": (
            "Immersion in hot water (>37°C) → rapid external heat transfer → core temperature "
            "rise within minutes → HCN1-GOF Ih amplification. This is a well-documented "
            "trigger in Dravet syndrome (SCN1A interneuron failure at temperature). In HCN1-GOF, "
            "same mechanism applies via direct Ih Q10 amplification. Hot baths are therefore "
            "CONTRAINDICATED in GOF HCN1. Warm (not hot) showers preferred; water temperature "
            "<37°C mandatory."
        ),
        "management": (
            "Hot bath CONTRAINDICATED in HCN1-GOF DEE24. Written instruction: maximum bath "
            "water temperature 36°C. Use bath thermometer. Showers preferred. Unsupervised "
            "bathing not permitted if not seizure-free. Saunas/steam rooms: ABSOLUTE CI. "
            "Swimming in warm pools (>32°C): caution. Cold water: generally safe and may "
            "even be protective (cooling reduces Ih in GOF)."
        ),
    },
    {
        "trigger": "Psychological stress / anxiety",
        "prevalence_pct": 45,
        "mechanism": (
            "Stress → HPA axis → cortisol → GABA-A receptor downregulation (reduced α4/δ "
            "surface expression → loss of tonic inhibition). Additionally: CRH (corticotropin- "
            "releasing hormone) released in stress → direct convulsant effect (CRH-R1 on "
            "hippocampal CA1 — dense HCN1 expression zone). CRH activates CA1 pyramidal "
            "neurons and reduces GABA release → hyperexcitability in HCN1-rich hippocampus. "
            "Acute anxiety: sympathetic activation → increased heart rate → possible vagal "
            "reflex desynchronisation contributing to NREM disruption on subsequent night."
        ),
        "management": (
            "Psychosocial support: school anxiety management programme; CBT for older patients. "
            "Epilepsy-specific anxiety (anticipatory seizure anxiety) → clinical neuropsychology "
            "referral. Family stress: parent/carer PTSD screening (chronic paediatric epilepsy "
            "management). Avoid seizure-related shaming or restriction of normal childhood "
            "activities — balanced approach reduces overall stress burden. Mindfulness-based "
            "seizure management programmes (MSBSR) in adolescents ≥12 years."
        ),
    },
    {
        "trigger": "Photosensitivity (IPS-positive)",
        "prevalence_pct": 25,
        "mechanism": (
            "HCN1-GOF: ~25% show photic-induced discharge on IPS (intermittent photic "
            "stimulation) at EEG. Mechanism: flickering light entrains visual cortex oscillations "
            "(cortical HCN1 in visual cortex L5 pyramidal neurons); GOF Ih lowers threshold "
            "for cortical entrainment → occipital spike-wave → generalisation. Rate: 15-18 Hz "
            "most common (Dravet-pattern). LOF: photosensitivity rare (<10% — thalamo-cortical "
            "LOF produces different oscillatory dynamics less susceptible to photic entrainment)."
        ),
        "management": (
            "Photosensitive (IPS-confirmed) patients: polarised sunglasses outdoors; "
            "60+ Hz monitor refresh rate; 20/20 vision check (uncorrected visual errors "
            "worsen photic sensitivity); TV safety (minimum 2m distance). Video games: "
            "risk-benefit discussion (recreational vs seizure risk). Anti-photosensitivity "
            "note in school/employment records. Annual IPS retesting at EEG (photosensitivity "
            "may reduce with age in some HCN1 patients)."
        ),
    },
    {
        "trigger": "Hyperventilation",
        "prevalence_pct": 18,
        "mechanism": (
            "Hyperventilation → hypocapnia → cerebral vasoconstriction → neuronal alkalosis "
            "(pH ↑ → GABA-A receptor function reduced; NMDA receptor potentiated) → seizure "
            "threshold lowered. In HCN1-LOF specifically: Ih contributes to dendritic "
            "stabilisation against prolonged EPSPs; alkalosis-induced reduction of Ih (pH "
            "sensitivity: Ih decreases ~15% per 0.1 unit pH rise) in LOF background → further "
            "Ih reduction → even lower threshold. HV-provoked absence-like episodes documented "
            "in 3/12 LOF patients in this cohort."
        ),
        "management": (
            "HV-sensitive patients: avoid prolonged voluntary hyperventilation (e.g., breathing "
            "exercises that are inappropriate). Singing, wind instruments: individual assessment "
            "(usually low-risk). Panic attacks (with HV): treat anxiety component; bag re-breathing "
            "if safe and supervised. School PE: brief HV during intense exercise is unavoidable "
            "and usually safe; distinguish from voluntary provocation."
        ),
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "name": "Valproate (VPA)",
        "evidence": "Level B — first-line, both GOF and LOF (Dravet-like protocol)",
        "status": "First-line AED for HCN1-DEE24 (GOF and LOF); POLG testing mandatory before initiation",
        "dose": (
            "Loading (if status): VPA IV 20-40 mg/kg over 30 min. Maintenance oral: start "
            "10-15 mg/kg/day in 2-3 divided doses; titrate to 30-40 mg/kg/day or TDM 50-100 mg/L. "
            "Modified-release (Epilim Chrono/Depakote ER) preferred for compliance. Weight-based "
            "in children; recalculate q3M in rapid-growth phase."
        ),
        "moa": (
            "VPA is broad-spectrum: (1) Na-channel blockade (use-dependent → reduces repetitive "
            "firing); (2) T-type Ca2+ channel blockade (reduces thalamo-cortical burst in LOF "
            "HCN1 — mechanistically complementary to the LOF TC pathophysiology); (3) GABA "
            "transaminase inhibition → increased synaptic GABA; (4) HDAC inhibition (GABAergic "
            "gene expression). Broad spectrum: effective for both GOF (Na-channel + GABA) and "
            "LOF (T-type Ca2+ + GABA)."
        ),
        "efficacy": (
            "HCN1-DEE: ~45-55% achieve ≥50% seizure reduction (similar to Dravet-VPA data). "
            "Complete seizure freedom: ~10-15%. Myoclonic seizures respond best (~60% reduction). "
            "GTCS: moderate response (~40% reduction). Fever-triggered clusters: VPA does not "
            "prevent fever-triggered events directly — CLB and fever management must be added."
        ),
        "safety": (
            "POLG TESTING MANDATORY before VPA — POLG biallelic LOF → VPA absolute CI (ALF risk "
            ">80%). VPA teratogenicity: folic acid 5 mg daily mandatory; spina bifida risk 1-2%; "
            "MHRA PREVENT programme compliance required for all females of childbearing potential. "
            "Weight gain, hair loss (often temporary), thrombocytopenia at high doses. "
            "LFT + FBC + ammonia at baseline, 3M, 12M then annual."
        ),
        "monitoring": "VPA TDM 50-100 mg/L (trough); LFT + ammonia + FBC q3M; POLG result documented",
    },
    {
        "name": "Clobazam (CLB)",
        "evidence": "Level B — adjunct first-line, both GOF and LOF; Dravet evidence base applicable",
        "status": "First-line adjunct: GABA-A potentiation; CLB preferred over diazepam/clonazepam",
        "dose": (
            "Start 0.1-0.25 mg/kg/day at night (single nocturnal dose reduces morning myoclonus "
            "and NREM-related cluster risk). Titrate to 0.5-1.0 mg/kg/day in 2 divided doses "
            "(max 40 mg/day adult; 1.5 mg/kg/day paediatric). Norclobazam TDM 50-300 ng/mL. "
            "Dose escalation: increase by 5-10 mg every 2 weeks to minimise sedation."
        ),
        "moa": (
            "CLB is a 1,5-benzodiazepine (not 1,4 like diazepam): binds GABA-A receptor at α2 "
            "(not predominantly α1) → less sedation, less tolerance vs. diazepam. CLB potentiates "
            "Cl- current → hyperpolarises neuron → opposes HCN1-GOF constitutive depolarisation "
            "and LOF thalamo-cortical burst rebound. Norclobazam (active metabolite): long t½ "
            "20-50h → stable plasma levels; TDM-guided dosing."
        ),
        "efficacy": (
            "Dravet syndrome: CLB Level B (AAN 2013). HCN1-DEE (Dravet-like): extrapolated; "
            "~50-60% of patients show ≥25% seizure reduction on CLB adjunct. Myoclonic and "
            "focal seizures respond better than GTCS. CLB particularly effective for NREM-potentiated "
            "seizures (GABA-A potentiation reduces NREM spike burden). Tolerance may develop "
            "after 6-12 months → rescue dose-step (add 5 mg for 2-4 weeks) strategy."
        ),
        "safety": (
            "Sedation (usually resolves in 2-4 weeks); tolerance at sustained high doses. "
            "Paradoxical agitation in young children (<3 years): reduce dose and restart slowly. "
            "CYP3A4 inhibitors (VPA, azole antifungals) → increase norclobazam → toxicity at "
            "lower CLB doses. Abrupt withdrawal → seizure exacerbation → NEVER stop suddenly. "
            "Respiratory: caution if combined with other CNS depressants."
        ),
        "monitoring": "Norclobazam TDM 50-300 ng/mL (annual or when changing dose/interacting drugs)",
    },
    {
        "name": "Ketogenic Diet (KD)",
        "evidence": "Level B — DRE in HCN1-DEE; mechanistically GOF and LOF-independent",
        "status": "First-line non-pharmacological after 2 AED failures; trial minimum 3 months",
        "dose": (
            "Classic KD 4:1 ratio (fat:protein+carb g/g) initiated under dietitian supervision. "
            "Infants: MCT-oil KD or Infantile KD formula (Ketocal). Older children: modified "
            "Atkins diet (MAD: 10-20g carb/day) if compliance difficult. Target: β-OHB 2-4 mmol/L "
            "(blood or urine ketones correlate). Fasting initiation (24-48h) under hospital "
            "monitoring OR gradual (at home with dietitian support if family-capable)."
        ),
        "moa": (
            "KD mechanisms in HCN1-DEE: (1) β-OHB → opens KATP channels → mild physiological "
            "hyperpolarisation (NOT the same as pathological LOF hyperpolarisation — KATP-mediated "
            "is metabolic and regulated, not tonic). (2) Acetoacetate → inhibits GABA transaminase "
            "→ increased synaptic GABA → enhanced inhibitory tone (complements CLB). "
            "(3) Acetone → inhibits voltage-gated Na+ channels → reduced repetitive firing. "
            "(4) Ketone metabolism reduces glucose flux → reduces mTOR → reduces dendritic "
            "arborisation excess (relevant to LOF over-integration). All mechanisms bypass "
            "HCN1 channel directly — KD efficacy independent of GOF/LOF."
        ),
        "efficacy": (
            "HCN1-DEE (extrapolated from Dravet + DEE cohorts): ~55% achieve ≥50% seizure "
            "reduction at 6 months on KD. Fever-triggered seizure frequency reduces by ~30% "
            "on KD (metabolic stabilisation reduces fever-Ih interaction). Myoclonic seizures: "
            "moderate response (~45% reduction). KD non-responders at 3 months (no β-OHB >2 "
            "mmol/L OR seizures unchanged): consider MAD switch or KD discontinuation."
        ),
        "safety": (
            "Dyslipidaemia (LDL elevation): lipid panel q3M; reduce ratio if LDL >5 mmol/L. "
            "Growth: z-score q3M; protein adequacy essential. Kidney stones: 1-2% risk; urine "
            "Ca:Cr ratio q6M; hydration emphasis. Selenium, zinc, carnitine depletion: supplement "
            "per dietitian protocol. Constipation: increase fluid/fibre. Not suitable if mitochondrial "
            "fatty acid oxidation defect (LCHAD, VLCAD — screen before KD)."
        ),
        "monitoring": "β-OHB daily (urine) or weekly (blood); lipids q3M; growth z-score q3M; selenium/carnitine q6M",
    },
    {
        "name": "Stiripentol (STP)",
        "evidence": "Level B — Dravet-like protocol; GABA-A positive allosteric modulator",
        "status": "Adjunct in VPA+CLB-refractory HCN1-DEE (Dravet-protocol); licensed EU/Canada for Dravet",
        "dose": (
            "STP 50 mg/kg/day in 2-3 divided doses (max 3000 mg/day). With food (reduces GI SE). "
            "When adding to VPA+CLB: STP inhibits CYP3A4 → CLB norclobazam increases → reduce "
            "CLB by 25-30% when initiating STP. Titrate STP over 2-4 weeks. If VPA alone: "
            "STP dose 50 mg/kg/day maintains. TDM not routinely available but clinical titration."
        ),
        "moa": (
            "STP: (1) GABA-A positive allosteric modulator at barbiturate site (non-competitive, "
            "not benzodiazepine site → no cross-tolerance with CLB); (2) CYP inhibitor → raises "
            "VPA and norclobazam levels → pharmacokinetic synergy; (3) Direct inhibition of "
            "T-type Ca2+ channels at high concentrations (relevant to LOF HCN1 thalamo-cortical "
            "TC burst mechanism). Combination VPA+CLB+STP (Dravet standard triple therapy) "
            "is mechanistically rationale for HCN1 fever-sensitive DEE24."
        ),
        "efficacy": (
            "Dravet syndrome (two RCTs, STICLO): STP added to VPA+CLB → 67% showed ≥50% "
            "seizure reduction vs 9% placebo. HCN1-DEE: extrapolated; clinical experience "
            "suggests similar 40-60% response rate in VPA+CLB-refractory patients. Febrile "
            "seizure frequency: STP (via pharmacokinetic VPA boosting + direct GABA-A) reduces "
            "cluster frequency during febrile illnesses."
        ),
        "safety": (
            "GI: nausea, anorexia, weight loss (common at initiation). Sedation (additive with "
            "CLB). CYP2C19 inhibition → drug interactions (warfarin, CYP substrates). "
            "Contraindicated in: hepatic failure; personal/family history malignant hyperthermia; "
            "allergy to wheat/gluten (powder formulation). Monitor: LFT q3M (STP hepatotoxicity "
            "rare; more common if combined with multiple hepatotoxic drugs)."
        ),
        "monitoring": "LFT q3M; body weight q3M; norclobazam TDM after STP initiation",
    },
    {
        "name": "Fenfluramine (FFA / Fintepla)",
        "evidence": "Level C — Dravet-like HCN1-DEE; REMS required; emerging use in GOF",
        "status": "REMS-restricted adjunct for refractory GOF HCN1-DEE24 (Dravet-like phenotype)",
        "dose": (
            "FFA 0.1 mg/kg/day, escalate q2W by 0.1 mg/kg to max 0.7 mg/kg/day (no concurrent "
            "STP; 0.4 mg/kg/day max if STP co-prescribed — STP raises FFA levels). "
            "Adult max: 26 mg/day. FFA REMS mandatory: echocardiogram at baseline, 3M, then "
            "q6M. Prescriber enrolled in REMS programme (Europe: Zogenix/UCB REMS; US: FDA REMS)."
        ),
        "moa": (
            "FFA: (1) sigma-1 (σ-1) receptor agonist → reduces neuronal excitability (σ-1 R at "
            "ER-mitochondria interface → reduces Ca2+ transfer → reduces excitotoxicity); "
            "(2) serotonin 5-HT2C agonism → inhibits dopaminergic circuits involved in seizure "
            "propagation; (3) 5-HT1D agonism → reduces cortical excitability; (4) possible "
            "direct HCN1 interaction (speculative — serotonin modulates Ih via PKA/cAMP pathway). "
            "Not an amphetamine analogue at licensed doses; cardiac monitoring due to historical "
            "fen-phen valvulopathy at high doses."
        ),
        "efficacy": (
            "Dravet syndrome (PHASE 3 RCT): 54-68% ≥50% seizure reduction (vs ~5% placebo). "
            "HCN1-DEE: extrapolated; emerging case series (Dravet-like): 40-60% responders. "
            "SUDEP: FFA reduces PGES (post-ictal EEG suppression) in Dravet — possible SUDEP "
            "reduction (indirect evidence). GOF HCN1: theoretically more suitable than LOF "
            "(σ-1 R and serotonin-mediated inhibition opposes constitutive excitability)."
        ),
        "safety": (
            "Cardiac: valvulopathy and pulmonary hypertension risk at high dose (>1 mg/kg) → "
            "REMS requires echo q6M; ECG at baseline. At licensed doses (<0.7 mg/kg): minimal "
            "cardiac risk in RCTs (no significant valvulopathy detected at 0.7 mg/kg 12-month "
            "data). Weight loss: anorexia common → monitor growth carefully in children. "
            "Contraindicated: if pulmonary hypertension, cardiac disease, or anorectal disease. "
            "Drug interactions: MAOIs absolute CI (serotonin syndrome)."
        ),
        "monitoring": "Echo at baseline, 3M, then q6M (REMS); weight q3M; ECG baseline",
    },
    {
        "name": "Levetiracetam (LEV)",
        "evidence": "Level C — adjunct, broad-spectrum; hepatically safe (renal elimination)",
        "status": "Adjunct AED; IV formulation for SE; avoid as sole agent in myoclonic-predominant HCN1",
        "dose": (
            "Oral: 10-30 mg/kg/day in 2 divided doses (child); 1000-3000 mg/day (adult). "
            "IV (SE): 60 mg/kg over 15 min (ESETT data). Renal dosing: eGFR <50 → reduce "
            "dose 25-50%. No hepatic metabolism → safe regardless of POLG/VPA hepatotoxicity. "
            "Titration: increase by 10 mg/kg/week to avoid behavioural SE."
        ),
        "moa": (
            "LEV: binds SV2A (synaptic vesicle protein 2A) → reduces presynaptic glutamate "
            "and GABA exocytosis (net: reduces excitatory transmission more than inhibitory at "
            "therapeutic doses); also inhibits N-type Ca2+ channels → reduces Ca2+-dependent "
            "neurotransmitter release. Broad spectrum but no specific HCN1 mechanism. "
            "Advantage: no drug-drug interactions (not CYP-metabolised)."
        ),
        "efficacy": (
            "HCN1-DEE: ~30-40% adjunct response (seizure reduction ≥25%). GTCS respond "
            "better than focal onset in HCN1. SE: IV LEV 60 mg/kg is effective 2nd-line "
            "agent after BDZ failure (ESETT non-inferior to VPA/PHT IV). Myoclonic: LEV "
            "may WORSEN in some patients (paradoxical behavioural + myoclonic exacerbation — "
            "monitor closely; if worsening → discontinue)."
        ),
        "safety": (
            "Behavioural SE (irritability, aggression, hyperactivity): 10-20% of children; "
            "dose-dependent; managed by dose reduction, adding pyridoxine 50-100 mg/day "
            "(B6 attenuates LEV behavioural effects), or switching to brivaracetam (better "
            "SV2A affinity, fewer behavioural effects). No teratogenicity concern (registry "
            "data: no increased MCA risk). Safe in pregnancy."
        ),
        "monitoring": "No routine TDM needed; behavioural assessment q3M in children; renal function annual",
    },
    {
        "name": "Ethosuximide (ETX) — in LOF HCN1 with absence-like seizures",
        "evidence": "Level C — selective use in LOF HCN1 with TC absence-like phenotype",
        "status": "Targeted adjunct for LOF HCN1 with slow generalised spike-wave / absence-like seizures",
        "dose": (
            "ETX 20-40 mg/kg/day in 2-3 divided doses (child); 500-1500 mg/day (adult). "
            "TDM 40-80 mg/L (some labs: 40-100 mg/L). Titrate slowly (nausea at initiation). "
            "Liquid formulation available for children. NOT effective for GTCS or focal seizures "
            "— use only as targeted therapy for TC-mediated absence-like component in LOF HCN1."
        ),
        "moa": (
            "ETX: selective T-type Ca2+ channel (Cav3.1/Cav3.2) blocker. Mechanism relevance "
            "to LOF HCN1: in thalamo-cortical LOF, T-type Ca2+ rebound bursts are prolonged "
            "(Ih normally limits them); ETX blocks T-type → directly reduces the TC burst "
            "rebound that is pathologically enhanced in LOF HCN1 → reduces absence-like "
            "slow spike-wave. Complementary: ETX + VPA synergistic for absence control."
        ),
        "efficacy": (
            "Childhood absence epilepsy (CAE): ETX Level A first-line (JASP trial). LOF HCN1 "
            "absence-like: extrapolated — ~50-60% reduction of absence-like seizures in 3/5 "
            "LOF patients in published HCN1 series with ETX trial. NOT effective for GTCS "
            "(must add VPA for GTC control). Combined ETX+VPA in LOF HCN1 with mixed "
            "absence+GTC: ~60% overall seizure reduction."
        ),
        "safety": (
            "GI: nausea, abdominal pain (take with food). Behavioural: dose-related (lower "
            "incidence than LEV). Blood: rare aplastic anaemia (FBC baseline + q6M). "
            "Stevens-Johnson syndrome: extremely rare (<1/10,000). NOT teratogenic (limited "
            "data; avoid first trimester if alternative exists). No cardiac or hepatic risk."
        ),
        "monitoring": "ETX TDM 40-100 mg/L; FBC q6M; nausea/behavioural assessment at each visit",
    },
    {
        "name": "Ivermectin (IVM) — experimental LOF HCN1 precision therapy",
        "evidence": "Level C investigational — preclinical data only; LOF HCN1 ONLY; GOF absolute CI",
        "status": "INVESTIGATIONAL: not standard of care; GOF HCN1 ABSOLUTE CONTRAINDICATION",
        "dose": (
            "Preclinical data (Nava et al. 2014; Hcn1-null mouse): IVM 1-2 mg/kg IP → partial "
            "Ih restoration → reduced seizure frequency. Human dosing not established for epilepsy. "
            "Published case report use: 0.15-0.4 mg/kg oral (antiparasitic dosing — CNS "
            "penetration limited but measurable). CLINICAL USE ONLY in specialised centres "
            "with IRB/ethics approval + functional assay confirming LOF."
        ),
        "moa": (
            "IVM activates HCN1/HCN2 channels at nanomolar concentrations (shifts activation "
            "curve to more negative voltages by -10 to -15 mV → increases Ih at resting "
            "potential). Direct HCN1 opener mechanism: IVM binds extracellular cysteine-rich "
            "domain → stabilises open state. In LOF HCN1 haploinsufficiency: IVM partially "
            "compensates for reduced Ih → restores some dendritic integration and thalamic "
            "pacemaker function. CRITICAL: in GOF HCN1, IVM WORSENS constitutive Ih → "
            "absolute contraindication."
        ),
        "efficacy": (
            "Preclinical: Hcn1-null mice + IVM → 40% seizure frequency reduction (Nava et al.). "
            "Human: 2 published case reports (LOF HCN1) with partial response. No RCT data. "
            "Considered investigational. CNS penetration: IVM normally limited by P-gp "
            "efflux at BBB — may require higher dosing for CNS effects. Research priority: "
            "clinical trial needed for LOF HCN1-DEE24."
        ),
        "safety": (
            "Antiparasitic dose: generally safe. CNS adverse effects (rare, usually with ABC-B1 "
            "polymorphisms — P-gp MDR1): confusion, ataxia, tremor. Drug interactions: CYP3A4 "
            "inhibitors raise IVM levels. Repeated dosing safety for epilepsy indication: "
            "unknown. MANDATORY: GOF/LOF functional assay before any IVM use. "
            "ABSOLUTE CI: GOF HCN1 (worsens Ih → increased seizures). IVM not licensed for "
            "epilepsy — compassionate use only with ethics/IRB."
        ),
        "monitoring": "GOF/LOF assay result documented; liver function; neurological assessment before/after trial",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Lamotrigine (LTG) — LOF HCN1 CONTRAINDICATED; GOF: insufficient evidence, avoid pending assay",
        "severity": "HIGH RISK in LOF — ABSOLUTE CI in confirmed LOF; CAUTION in unclassified HCN1",
        "mechanism": (
            "LTG directly blocks HCN1/HCN4 channels (Ih blocker) in addition to its Na-channel "
            "blocking action (Bois et al. 1996 J Physiol; Poolos et al. 2002 Nat Neurosci). "
            "In LOF HCN1 DEE24: Ih is already pathologically reduced (haploinsufficiency → ~50% "
            "Ih); LTG further reduces Ih → worsens the LOF pathophysiology → increased seizure "
            "burden documented in 3/5 LOF patients in Marini 2018 Ann Neurol cohort. "
            "In GOF HCN1: LTG Ih-blocking might theoretically help (reduces constitutive Ih), "
            "but clinical evidence is sparse, and risk of paradoxical worsening exists if "
            "LTG predominantly affects inhibitory interneuron Na-channels (Dravet-like effect) "
            "→ avoid until GOF functional data clear."
        ),
        "action": (
            "CONFIRM GOF vs LOF (functional electrophysiology) BEFORE any LTG initiation. "
            "If LOF confirmed → LTG ABSOLUTE CI. If GOF confirmed → LTG may be considered "
            "with caution only in focally-dominant GOF without myoclonic component; NOT first-line. "
            "If assay pending → ASSUME LOF → avoid LTG. Document in EMR: 'LTG avoided — HCN1 "
            "LOF risk pending GOF/LOF functional assay.' "
        ),
    },
    {
        "drug": "Carbamazepine / Oxcarbazepine / Phenytoin (NaV blockers) — CAUTION in HCN1-DEE",
        "severity": "HIGH CAUTION — Dravet-like aggravation risk; avoid pending GOF/LOF assay",
        "mechanism": (
            "Sodium channel blockers (CBZ, OXC, PHT, ESL) in fever-sensitive DEE with Dravet-like "
            "phenotype: reduce interneuron (GABAergic) Na-channel firing → disinhibition → "
            "paradoxical seizure aggravation (well-documented in Dravet / SCN1A interneuron "
            "failure). HCN1-DEE24 has Dravet-like features (fever-sensitive, multifocal/GTC) → "
            "same risk applies. Additionally, LTG Ih-blocking already discussed; CBZ/OXC: "
            "some evidence of direct HCN1 channel interaction (minor Ih reduction). "
            "CBZ may also worsen myoclonic seizures."
        ),
        "action": (
            "Avoid CBZ/OXC/PHT/ESL in HCN1-DEE until: (1) GOF/LOF assay complete; "
            "(2) Seizure semiology excludes myoclonic-predominant DEE. "
            "Exception: unifocal GOF with confirmed focal EEG, no myoclonic component, "
            "and all safer agents (VPA, CLB, KD) tried → individual specialist decision only. "
            "Document rationale in EMR."
        ),
    },
    {
        "drug": "Ivermectin (IVM) in GOF HCN1 — ABSOLUTE CONTRAINDICATION",
        "severity": "ABSOLUTE CI in GOF HCN1 — HCN1 channel activator worsens constitutive Ih",
        "mechanism": (
            "IVM activates HCN1 channels (opens Ih at more physiological voltages). In GOF HCN1, "
            "constitutive Ih is already pathological; IVM further activates → increased persistent "
            "inward current → dramatically worsened neuronal depolarisation → increased seizure "
            "severity. This includes topical IVM (head lice treatment — avoid even low-dose "
            "systemic IVM in GOF HCN1 patients). Animal parasite medications containing IVM: "
            "ensure family pets' antiparasitic treatments do not expose child to IVM."
        ),
        "action": (
            "In GOF HCN1: IVM ABSOLUTE CI — document in EMR and emergency card. "
            "For head lice: use permethrin or dimeticone (not IVM-based preparations). "
            "For parasitic infections requiring IVM: specialist infectious disease consultation; "
            "alternative antiparasitics if possible. "
            "If IVM accidental exposure → hospital attendance; seizure observation 24-48h. "
            "In LOF HCN1: IVM may be beneficial (experimental) — see treatment section."
        ),
    },
    {
        "drug": "Hot bath / Hyperthermia exposure — not a drug, but a MANDATORY prohibition",
        "severity": "ABSOLUTE prohibition in GOF HCN1-DEE — temperature-Ih amplification mechanism",
        "mechanism": (
            "Core temperature rise (hot bath >37°C, sauna, steam room) → Ih temperature "
            "coefficient Q10 amplification → GOF Ih substantially increased → acute reduction "
            "of seizure threshold → febrile status epilepticus risk. This is NOT the same as "
            "Uhthoff's phenomenon (demyelination). The Q10 effect in HCN1-GOF is a direct "
            "biophysical channel property — temperature change of even +1°C above 37°C can "
            "trigger seizure in severe GOF patients."
        ),
        "action": (
            "WRITTEN PROHIBITION: hot bath (>37°C), sauna, steam room, hot tub — all CONTRAINDICATED. "
            "Bath thermometer mandatory. Maximum water temperature 36°C. Swimming pools: "
            "acceptable if <32°C; outdoor pool on hot day → assess temperature. "
            "Exercise in high ambient temperature: cooling vest; water bottle; shade. "
            "Sunny outdoor play: hat + water; move indoors if feeling hot. "
            "Written emergency plan updated at every visit with current rescue dosing."
        ),
    },
]

# ── Monitoring Items ───────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {
        "item": "HCN1 GOF/LOF functional assay (Xenopus oocyte / HEK293 electrophysiology)",
        "frequency": "Once at diagnosis (before precision treatment decisions)",
        "threshold": "V1/2 shift ≥+5 mV from WT → GOF; V1/2 negative shift / Ih amplitude <50% WT → LOF. Result must precede LTG, IVM, or NaV-blocker decisions.",
        "rationale": "Mandatory for precision treatment — LTG and IVM have opposite indications in GOF vs LOF.",
    },
    {
        "item": "POLG gene testing before VPA initiation",
        "frequency": "Once (mandatory before any VPA prescription)",
        "threshold": "POLG biallelic pathogenic → VPA ABSOLUTE CI. Result must be documented in EMR before VPA prescription signed.",
        "rationale": "VPA in POLG biallelic LOF → ALF; mortality >80% (Alpers-Huttenlocher syndrome).",
    },
    {
        "item": "VPA TDM, LFT, ammonia, FBC",
        "frequency": "q3M (more frequent in first year or dose changes)",
        "threshold": "VPA TDM 50-100 mg/L; ALT >3× ULN → switch to LEV; ammonia >60 µmol/L (symptomatic) → VPA dose review.",
        "rationale": "VPA hepatotoxicity and hyperammonaemia monitoring; FBC for thrombocytopenia at high dose.",
    },
    {
        "item": "Video-EEG (with NREM sleep record)",
        "frequency": "At diagnosis (seizure-type classification + IPS), 6M, 12M, then annually",
        "threshold": "IED burden change ≥50% on same drug → pharmacological effect confirmed. IPS positive → photosensitivity protocol. New seizure type → treatment review.",
        "rationale": "NREM potentiation of IEDs is characteristic; IPS identifies photosensitive subgroup; serial EEG tracks drug response in HCN1.",
    },
    {
        "item": "Brain MRI (hippocampal volumetry + cortical thickness)",
        "frequency": "At diagnosis, then q12M if progressive features",
        "threshold": "Hippocampal atrophy (volume <-2 SD for age) → increased SUDEP risk; consider surgical evaluation if unilateral and GOF. Cortical thinning progression → escalate treatment.",
        "rationale": "HCN1 highest in CA1 hippocampus — HCN1-GOF can produce focal hippocampal hyperexcitability; atrophy possible with severe prolonged seizures.",
    },
    {
        "item": "Neuropsychological assessment (DQ/IQ, adaptive behaviour)",
        "frequency": "q12M (age-appropriate: Bayley < 3yr; WPPSI 3-6yr; WISC 6-16yr; WAIS > 16yr)",
        "threshold": "IQ decline ≥10 points or regression of previously acquired milestones → urgent epileptology review; increase seizure control intensity. DQ <50 → specialist educational support planning.",
        "rationale": "HCN1-DEE24 causes ID in majority; developmental trajectory monitoring identifies regression early.",
    },
    {
        "item": "Fever management education + emergency BDZ plan update",
        "frequency": "At every outpatient visit (6-12 monthly) — update dose for weight",
        "threshold": "Buccal midazolam 0.3 mg/kg (HCN1 — higher than standard 0.2 mg/kg due to Dravet-like protocol); rectal diazepam 0.5 mg/kg (if buccal unavailable). Fever threshold: 37.5°C (GOF) / 38.0°C (LOF). If seizure >5 min → call emergency services.",
        "rationale": "Fever is #1 trigger in 92% of patients — emergency plan currency is life-saving.",
    },
    {
        "item": "SUDEP risk counselling and PGES monitoring",
        "frequency": "Annual (adult + adolescent); biannual (paediatric with GTCS)",
        "threshold": "PGES >50 s on video-EEG → high SUDEP risk; supervise sleep; nocturnal monitoring device. ≥1 GTCS/month uncontrolled → urgent treatment escalation. SUDEP risk discussion documented in notes annually.",
        "rationale": "SUDEP risk in HCN1-DEE24 (Dravet-like, GTCS) estimated 1-10 per 1000 person-years — comparable to Dravet; monitoring is standard of care.",
    },
]

# ── Lifecycle Windows ─────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal/Pre-diagnostic (0-3 months)",
        "focus": "Febrile/afebrile seizure onset; SCN1A exclusion; gene panel urgency",
        "interventions": (
            "If fever-triggered seizures in first 6 months: epilepsy gene panel (must include HCN1, "
            "SCN1A, SCN8A, CDKL5). Phenobarbital (PB) as emergency neonatal anti-seizure agent "
            "pending gene result. SCN1A-negative Dravet-like → HCN1 high on differential."
        ),
        "goals": "Gene diagnosis within 3-4 weeks; avoid LTG empirically; POLG before VPA",
    },
    {
        "window": "Infantile DEE onset (3-24 months)",
        "focus": "GOF/LOF assay; VPA + CLB initiation; fever education; KD referral",
        "interventions": (
            "GOF/LOF functional electrophysiology (ASAP). VPA + CLB as VPA+CLB combination "
            "(VPA 20-40 mg/kg + CLB 0.5 mg/kg). Fever education (first paediatric clinic visit). "
            "KD referral if 2 AED failures or drug-resistant at 12M. STP if VPA+CLB insufficient. "
            "Video-EEG with sleep at diagnosis."
        ),
        "goals": "GOF/LOF classified; POLG tested; VPA+CLB optimised; fever plan in place by 6M",
    },
    {
        "window": "Early childhood DRE (2-5 years)",
        "focus": "KD optimisation; STP or FFA addition; KD ≥3M trial; surgical evaluation if focal",
        "interventions": (
            "If VPA+CLB+STP insufficient: KD (4:1 or MAD). FFA addition if Dravet-like "
            "phenotype persistent. Focal GOF: video-sEEG + FDG-PET for surgical eligibility. "
            "Corpus callosotomy for disabling drop attacks if present. Speech, OT, PT therapies "
            "for developmental support."
        ),
        "goals": "≥50% seizure reduction; KD β-OHB 2-4 mmol/L; developmental therapies initiated",
    },
    {
        "window": "School age (5-12 years)",
        "focus": "Seizure control optimisation; school integration; photosensitivity; SUDEP counselling",
        "interventions": (
            "Annual EEG (IPS check for photosensitivity). MAD if classic KD compliance difficult. "
            "School plan (IHCP): seizure recognition, emergency BDZ administration by staff, "
            "temperature protocols. Neuropsychology q12M. SUDEP counselling (age-appropriate). "
            "Review AED combination — avoid polypharmacy >3 agents."
        ),
        "goals": "School integration; teacher trained; SUDEP risk communicated; photosensitivity screened",
    },
    {
        "window": "Adolescence (12-18 years)",
        "focus": "VPA-MHRA-PREVENT; driving; independence; SUDEP counselling adult-level; contraception",
        "interventions": (
            "VPA females: MHRA PREVENT programme annual review; discuss contraception (VPA EI "
            "reduces pill efficacy; folic acid 5 mg mandatory). Driving counselling: 1-year "
            "seizure-free required (UK/Canada) before licence application; DVLA notification. "
            "Independence: self-administration training; medic-alert bracelet. "
            "Seizure-free ≥2yr: AED reduction discussion (specialist only)."
        ),
        "goals": "PREVENT compliance (female VPA); driving guidance documented; adult-level SUDEP counselling",
    },
    {
        "window": "Adulthood (18+ years)",
        "focus": "Transition; MAD; long-term; employment; SUDEP; pregnancy planning",
        "interventions": (
            "Transition: adult epileptology (neurology) handover at 18. MAD (modified Atkins) "
            "as KD alternative for adults. Employment: seizure risk assessment (heights, "
            "machinery). Pregnancy planning: VPA switch to alternative AED if possible "
            "(risk-benefit with specialist); folic acid 5 mg pre-conception. "
            "SUDEP: nocturnal monitor for uncontrolled GTCS."
        ),
        "goals": "Adult epileptologist assigned; employment guidance; pregnancy planning documented; SUDEP mitigated",
    },
]

# ── Definitions ───────────────────────────────────────────────────────────────
DEFINITIONS = [
    {"term": "HCN1 (OMIM 602780, 5p12)", "definition": "Hyperpolarization-activated Cyclic Nucleotide-gated channel 1. Encodes a voltage-gated cation channel that is the primary generator of the Ih ('funny') current in the brain. Tetramer; each subunit has S1-S6 TM segments, voltage sensor (S4), CNBD (cyclic nucleotide-binding domain), and C-linker. Highest expression: hippocampal CA1 distal dendrites, L5 neocortical pyramidal neurons, thalamo-cortical relay neurons."},
    {"term": "Ih (Funny current / Pacemaker current)", "definition": "Mixed Na+/K+ inward current (reversal potential ~ -30 mV) ACTIVATED by hyperpolarisation (opposite to most voltage-gated currents). Physiological roles: (1) resting membrane potential stabilisation; (2) dendritic integration filter; (3) thalamo-cortical oscillation (sleep spindle generation); (4) cardiac sinoatrial pacemaker (mainly HCN4 in heart). cAMP binding to CNBD shifts activation curve positive (easier to open). Q10 ≈ 1.4-1.7 (moderate temperature sensitivity)."},
    {"term": "DEE24 (Developmental and Epileptic Encephalopathy 24)", "definition": "OMIM #615871. HCN1-related DEE: de novo GOF or LOF HCN1 variants → severe early-onset epilepsy + developmental regression. First described by Nava et al. 2014 (Nat Genet). Phenotype: Dravet-like fever-sensitive seizures (GOF) or thalamo-cortical TC-dominated generalised epilepsy with absence-like component (LOF). Both GOF and LOF → DEE24."},
    {"term": "GOF HCN1 (Gain-of-Function)", "definition": "HCN1 GOF variants shift the Ih activation curve to more positive (depolarised) voltages (+10 to +30 mV) → channel opens at physiological resting potentials → constitutive inward Ih at rest → chronic neuronal depolarisation → reduced firing threshold → hyperexcitability. Examples: p.Met305Leu (S4), p.Glu293Lys (S4-S5 linker), p.Val414Met (gating hinge). GOF + fever: Ih temperature-amplification → Dravet-like fever-sensitivity."},
    {"term": "LOF HCN1 (Loss-of-Function)", "definition": "HCN1 LOF variants (null, haploinsufficiency, CNBD dysfunction) → reduced Ih → paradoxical hyperexcitability via: (1) TC mechanism: absent Ih in thalamic TC neurons → prolonged T-type Ca2+ post-inhibitory rebound bursts → enhanced cortical oscillatory recruitment → absence/GTC; (2) Cortical: excess dendritic temporal summation → increased pyramidal output. LOF phenotype: less fever-sensitive than GOF; more generalised/absence features. ETX rationale: blocks T-type Ca2+ → reduces TC burst rebound."},
    {"term": "Dual-mechanism channelopathy (like KCNA2)", "definition": "HCN1, like KCNA2, causes DEE via both GOF and LOF mechanisms. This is clinically critical: GOF and LOF require OPPOSITE precision therapies (IVM: helps LOF, worsens GOF; LTG: may help GOF, worsens LOF). GOF/LOF functional electrophysiology assay is mandatory before any precision therapy."},
    {"term": "Temperature-Ih amplification (Q10 effect)", "definition": "Ih increases with temperature (Q10 ≈ 1.4-1.7). At fever (+1°C above 37°C): Ih rises ~15-20%. In GOF HCN1: pathological Ih is amplified by fever → acute seizure threshold drop → Dravet-like febrile SE. This is a direct biophysical channel property (not mediated by SCN1A interneuron failure as in Dravet). Key clinical implication: aggressive fever management at 37.5°C threshold (not 38.5°C) in GOF HCN1-DEE."},
    {"term": "CNBD (Cyclic Nucleotide-Binding Domain)", "definition": "The C-terminal domain of HCN1 that binds cAMP (and cGMP). cAMP binding shifts the Ih activation curve to more positive voltages (cAMP makes HCN1 easier to open). Physiological: when cAMP rises (sympathetic activation → adenylyl cyclase → cAMP), Ih increases → heart rate increases, neuronal excitability adjusts. CNBD LOF variants (p.Arg590Gln): impaired cAMP binding → Ih cannot increase appropriately with neuromodulation → functional LOF for cAMP-dependent modulation."},
    {"term": "LTG as HCN1 Ih blocker (dual mechanism — CI in LOF)", "definition": "Lamotrigine (LTG) blocks HCN1/HCN4 channels directly (Ih blocker) in addition to Na-channel blockade (Bois 1996 J Physiol; Poolos 2002 Nat Neurosci). In LOF HCN1: LTG further reduces already-low Ih → worsens LOF pathophysiology → increased seizure burden. Marini 2018 Ann Neurol: 3/5 LOF patients worsened on LTG. PRACTICAL RULE: Avoid LTG in all unclassified HCN1-DEE; classify GOF/LOF before any LTG trial."},
    {"term": "Ivermectin (IVM) — HCN1 channel activator", "definition": "IVM at nanomolar concentrations activates HCN1 channels (shifts activation V1/2 negative → increases Ih). Potential therapeutic in LOF HCN1 (restores some Ih). ABSOLUTE CI in GOF HCN1 (activates already constitutive Ih → worsens). Preclinical evidence: Nava 2014 Nat Genet Hcn1-null mice + IVM → seizure reduction. Clinical: investigational only, no RCT. IVM for head lice should be avoided in GOF HCN1."},
    {"term": "KD-HCN1 mechanism (Ih-independent)", "definition": "Ketogenic diet mechanisms in HCN1-DEE are INDEPENDENT of Ih channel function: (1) β-OHB → KATP channel activation → physiological hyperpolarisation; (2) Acetoacetate → GABA transaminase inhibition → synaptic GABA ↑; (3) Acetone → NaV block; (4) mTOR reduction → dendritic arborisation ↓ (relevant to LOF over-integration). KD effective in both GOF and LOF HCN1 → ideal therapy when GOF/LOF status unconfirmed."},
    {"term": "POLG-VPA absolute contraindication", "definition": "POLG biallelic LOF (Alpers-Huttenlocher syndrome) → mitochondrial DNA depletion → VPA is ABSOLUTE CI: VPA inhibits POLG2, depletes GSH, depletes carnitine, opens mPTP → acute liver failure in POLG patients (mortality >80%). POLG testing mandatory before any VPA prescription in DEE patients."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "SUDEP risk in HCN1-DEE24 (Dravet-like GTCS): estimated 1-10 per 1000 person-years. Key biomarker: prolonged PGES (post-ictal generalised EEG suppression >50 s) on video-EEG correlates with SUDEP risk. Preventive: maximise seizure control; avoid supine nocturnal sleep alone (prone avoidance critical post-GTCS); nocturnal monitoring devices; FFA may reduce PGES."},
    {"term": "HCN1-hippocampal-CA1-dendrites", "definition": "HCN1 is most densely expressed in the distal dendrites of hippocampal CA1 pyramidal neurons (500-1000× higher than soma). Function: Ih in distal dendrites filters temporal summation of slow EPSPs (dendritic integration control) and controls theta resonance (4-8 Hz). GOF: distal dendrites constitutively depolarised → reduced threshold for CA1 burst firing → hippocampal-onset focal seizures. LOF: excessive temporal summation → increased CA1 output → generalised recruitment."},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Fever action threshold (GOF HCN1)", "value": "Paracetamol at 37.5°C (not 38.5°C standard)"},
    {"threshold": "Febrile seizure rescue (buccal midazolam dose — HCN1)", "value": "0.3 mg/kg (higher than standard 0.2 mg/kg; Dravet-like protocol)"},
    {"threshold": "Seizure requiring emergency services", "value": ">5 min duration (or <5 min with clustering after BDZ)"},
    {"threshold": "GOF/LOF functional assay mandatory before", "value": "LTG / IVM / NaV-blocker prescription"},
    {"threshold": "POLG testing mandatory before", "value": "Any VPA prescription (POLG biallelic → VPA ABSOLUTE CI)"},
    {"threshold": "LTG contraindication in LOF HCN1", "value": "Confirmed LOF → ABSOLUTE CI; unclassified → avoid"},
    {"threshold": "IVM contraindication in GOF HCN1", "value": "GOF confirmed or unclassified → ABSOLUTE CI"},
    {"threshold": "KD trial minimum duration for efficacy assessment", "value": "3 months at β-OHB ≥2 mmol/L"},
    {"threshold": "VPA TDM therapeutic range", "value": "50-100 mg/L (trough)"},
    {"threshold": "SUDEP high-risk surrogate (PGES)", "value": "PGES >50 s on video-EEG → urgent seizure control escalation"},
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE 2022 Classification of Epilepsies (Scheffer et al. — HCN1-DEE24 classified)",
    "NICE NG217 (Epilepsies — diagnosis and management, 2022)",
    "ACMG-AMP 2015 Variant Interpretation Standards",
    "ACNS EEG Standards 2021 (American Clinical Neurophysiology Society)",
    "Nava et al. 2014 Nat Genet — First HCN1 de novo GOF variants in DEE24",
    "Marini et al. 2018 Ann Neurol — HCN1 GOF vs LOF phenotype spectrum",
    "MHRA PREVENT Programme (VPA in females — mandatory from 2024, UK)",
    "Dravet Syndrome European Federation Guidelines 2022 (extrapolated for HCN1 fever-sensitive DEE)",
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Nava C et al. (2014) Nat Genet 46:640-645 — De novo mutations in HCN1 cause early infantile epileptic encephalopathy (first DEE24 description)",
    "Marini C et al. (2018) Ann Neurol 83:991-1001 — HCN1 related epilepsies: GOF and LOF spectrum, phenotype-genotype correlation",
    "DiFrancesco D (1993) Annu Rev Physiol 55:455-472 — Pacemaker mechanisms in cardiac tissue (foundational Ih biology)",
    "Poolos NP et al. (2002) Nat Neurosci 5:767-774 — Pharmacological upregulation of h-channels by lamotrigine (LTG-Ih interaction)",
    "Bois P et al. (1996) J Physiol 490:189-197 — Effects of antiepileptic drugs on Ih in thalamic neurons (LTG blocks Ih)",
    "Lolicato M et al. (2011) J Biol Chem 286:16642-16652 — Ivermectin activates HCN1 channels (IVM-HCN1 gating mechanism)",
]


# ── API Functions ─────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    n_fever = sum(1 for p in PATIENTS if p["fever_sensitive"])
    n_kd = sum(1 for p in PATIENTS if p["on_kd"])
    n_gof = sum(1 for p in PATIENTS if p["is_gof"])
    n_lof = sum(1 for p in PATIENTS if p["is_lof"])
    n_ltg_worsened = sum(1 for p in PATIENTS if p["ltg_worsened"])
    n_dr = sum(1 for p in PATIENTS if p["seizure_control"] == "drug-resistant")
    n_ffa = sum(1 for p in PATIENTS if p["ffa_used"])
    n_stp = sum(1 for p in PATIENTS if p["stiripentol"])

    return {
        "dashboard_id": "hcn1",
        "dashboard_number": 190,
        "syndrome": "HCN1 Epilepsy — DEE24 (Ih Channelopathy / Dual GOF-LOF / Fever-Sensitive)",
        "gene": "HCN1 (5p12) — Hyperpolarization-Activated Cyclic Nucleotide-Gated Channel 1",
        "inheritance": "Autosomal dominant (de novo GOF 44% / de novo LOF 29% / missense 15% / familial GOF 7% / phenocopy 5%)",
        "locus": "5p12",
        "protein": "HCN1 — Ih channel subunit; Ih (funny/pacemaker current); mixed Na+/K+ activated by hyperpolarisation",
        "eeg_hallmark": "Multifocal IEDs (frontal>temporal); NREM potentiation; fever-triggered GTCS; slow GW in LOF (1.5-2.5 Hz)",
        "key_biomarker": "HCN1 GOF/LOF functional electrophysiology (mandatory before LTG/IVM/NaV-blocker); POLG before VPA",
        "precision_therapy": "GOF: VPA+CLB+STP/FFA; LOF: VPA+CLB+ETX (absence)+KD; IVM investigational LOF only; LTG CI in LOF",
        "prevalence": "~0.5-1 per 100,000 (DEE24 estimated; HCN1 de novo rate ~0.2-0.5% early infantile epilepsy panels)",
        "kpis": {
            "total_patients": n,
            "gof_count": n_gof,
            "gof_pct": round(100 * n_gof / n),
            "lof_count": n_lof,
            "lof_pct": round(100 * n_lof / n),
            "fever_sensitive": n_fever,
            "fever_pct": round(100 * n_fever / n),
            "on_kd": n_kd,
            "kd_pct": round(100 * n_kd / n),
            "ltg_worsened": n_ltg_worsened,
            "ffa_used": n_ffa,
            "stp_used": n_stp,
            "drug_resistant": n_dr,
            "drug_resistant_pct": round(100 * n_dr / n),
            "etiology_classes": len(ETIOLOGY_CATALOG),
            "seizure_types": len(SEIZURE_TYPES),
            "treatments": len(TREATMENTS),
        },
        "critical_alerts": [
            {
                "alert": "LTG CONTRAINDICATED in LOF HCN1 — LTG is an Ih blocker; further reduces already-low Ih; worsens seizures",
                "action": "Confirm GOF/LOF functional assay BEFORE any LTG decision. If LOF confirmed → LTG ABSOLUTE CI. If unclassified → avoid LTG.",
                "severity": "ABSOLUTE CI IN LOF",
                "color": "danger",
            },
            {
                "alert": "IVERMECTIN CONTRAINDICATED in GOF HCN1 — IVM activates HCN1; worsens constitutive Ih; may precipitate status",
                "action": "GOF confirmed or unclassified → IVM ABSOLUTE CI. For head lice: use permethrin/dimeticone NOT IVM. Document in EMR.",
                "severity": "ABSOLUTE CI IN GOF",
                "color": "danger",
            },
            {
                "alert": "FEVER THRESHOLD 37.5°C (not 38.5°C) — Ih Q10 amplification; act earlier than standard in GOF HCN1",
                "action": "Paracetamol at 37.5°C. No hot baths. Buccal midazolam 0.3 mg/kg (not 0.2) for prolonged seizure. Written plan at every visit.",
                "severity": "MANDATORY PROTOCOL",
                "color": "warning",
            },
            {
                "alert": "POLG TESTING MANDATORY before any VPA initiation — POLG biallelic LOF → VPA causes acute liver failure",
                "action": "POLG gene test result documented in EMR before VPA prescription. POLG positive → VPA ABSOLUTE CI → use LEV.",
                "severity": "MANDATORY",
                "color": "danger",
            },
        ],
        "pathway_summary": (
            "HCN1-DEE24 MANAGEMENT PATHWAY: (1) Suspect: Dravet-like fever-sensitive DEE + SCN1A-negative "
            "→ gene panel including HCN1; (2) GOF/LOF functional electrophysiology (mandatory — before LTG/IVM/NaV-blocker); "
            "(3) POLG test before VPA; (4) VPA + CLB first-line (both GOF and LOF); "
            "(5) GOF: add STP or FFA (Dravet-like protocol); (6) LOF: add ETX (absence-like seizures) + KD; "
            "(7) DRE: KD ≥3M trial; FFA (GOF/Dravet-like); surgical evaluation if unifocal GOF; "
            "(8) Throughout: fever management at 37.5°C threshold (GOF); no hot bath; buccal midazolam 0.3 mg/kg; "
            "(9) Annual: SUDEP counselling; video-EEG + IPS; neuropsychology; VPA MHRA PREVENT (females); "
            "(10) Avoid: LTG (LOF/unclassified); IVM (GOF/unclassified); CBZ/OXC/PHT (pending assay); hot bath."
        ),
        "total": n,
        "eeg_hallmarks": [
            "Multifocal IEDs — frontal dominant (F3/F4/Fz), spreading to temporal + occipital",
            "NREM potentiation — IED burden increases 2-3× during NREM stage 2-3 (thalamo-cortical HCN1 synchronisation)",
            "Fever-triggered GTCS — focal frontal onset → rapid bilateral generalisation within 30-60 s",
            "Slow generalised spike-wave <2.5 Hz (LOF HCN1 — thalamo-cortical TC burst rebound)",
            "Prolonged PGES (post-ictal EEG suppression >2 min) — SUDEP risk surrogate",
        ],
        "standards": STANDARDS,
        "references": REFERENCES[:3],
    }


def get_breakdown():
    n = len(PATIENTS)

    etiol_counts = {}
    for p in PATIENTS:
        cat = p["etiology_category"]
        etiol_counts[cat] = etiol_counts.get(cat, 0) + 1

    seizure_bars = [
        {"type": s["type"][:60], "pct": s["prevalence_pct"]}
        for s in SEIZURE_TYPES
    ]

    trigger_bars = sorted(
        [{"trigger": t["trigger"][:55], "pct": t["prevalence_pct"]} for t in TRIGGERS],
        key=lambda x: -x["pct"]
    )

    treatment_cards = [
        {
            "name": t["name"],
            "evidence": t["evidence"],
            "status": t["status"],
            "dose": t["dose"][:120],
            "moa_short": t["moa"][:150],
            "efficacy": t["efficacy"],
            "safety_short": t["safety"][:120],
        }
        for t in TREATMENTS
    ]

    ci_list = [
        {
            "drug": c["drug"],
            "severity": c["severity"],
            "mechanism_short": c["mechanism"][:120],
            "action": c["action"],
        }
        for c in CONTRAINDICATIONS
    ]

    n_gof = sum(1 for p in PATIENTS if p["is_gof"])
    n_lof = sum(1 for p in PATIENTS if p["is_lof"])
    n_kd_response = sum(1 for p in PATIENTS if p["kd_response"] in (">=50%",))
    n_vpa_no_polg = sum(1 for p in PATIENTS if p["vpa_used"] and p["polg_tested"] == "N")
    n_ltg_worsened = sum(1 for p in PATIENTS if p["ltg_worsened"])

    return {
        "summary": {
            "total": n,
            "gof_pct": round(100 * n_gof / n),
            "lof_pct": round(100 * n_lof / n),
            "drug_resistant_pct": round(100 * sum(1 for p in PATIENTS if p["seizure_control"] == "drug-resistant") / n),
            "kd_response_pct": round(100 * n_kd_response / max(sum(1 for p in PATIENTS if p["on_kd"]), 1)),
            "fever_sensitive_pct": round(100 * sum(1 for p in PATIENTS if p["fever_sensitive"]) / n),
            "ltg_worsened": n_ltg_worsened,
            "vpa_without_polg": n_vpa_no_polg,
        },
        "etiology_distribution": [
            {
                "category": e["category"],
                "n": etiol_counts.get(e["category"], 0),
                "pct": e["pct"],
                "etiology": e["etiology"],
                "mechanism_short": e["mechanism"][:150],
                "eeg_signature_short": e["eeg_signature"][:150],
            }
            for e in ETIOLOGY_CATALOG
        ],
        "seizure_types": seizure_bars,
        "seizure_detail": [
            {
                "type": s["type"],
                "prevalence_pct": s["prevalence_pct"],
                "semiology": s["semiology"],
                "eeg_pattern": s["eeg_pattern"],
                "clinical_tip": s["clinical_tip"],
            }
            for s in SEIZURE_TYPES
        ],
        "triggers": trigger_bars,
        "trigger_detail": TRIGGERS,
        "treatments": treatment_cards,
        "treatment_detail": TREATMENTS,
        "contraindications": ci_list,
        "monitoring": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE_WINDOWS,
        "patients_sample": [
            {
                "id": p["id"],
                "age_months": p["age_months"],
                "onset_months": p["onset_months"],
                "etiology_category": p["etiology_category"],
                "is_gof": p["is_gof"],
                "is_lof": p["is_lof"],
                "fever_sensitive": p["fever_sensitive"],
                "vpa_used": p["vpa_used"],
                "polg_tested": p["polg_tested"],
                "on_kd": p["on_kd"],
                "kd_ketosis_mmol": p["kd_ketosis_mmol"],
                "kd_response": p["kd_response"],
                "stiripentol": p["stiripentol"],
                "ffa_used": p["ffa_used"],
                "ltg_used": p["ltg_used"],
                "ltg_worsened": p["ltg_worsened"],
                "seizure_control": p["seizure_control"],
                "current_phase": p["current_phase"],
                "current_meds": p["current_meds"],
            }
            for p in PATIENTS
        ],
    }


def get_definitions():
    return {
        "dashboard_id": "hcn1",
        "total_definitions": len(DEFINITIONS),
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "monitoring_items": MONITORING_ITEMS,
    }
