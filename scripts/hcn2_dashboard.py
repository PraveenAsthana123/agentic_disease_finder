"""
HCN2 Epilepsy — Febrile Seizures / GEFS+ / Childhood Absence Epilepsy / GGE Spectrum
======================================================================================
40-patient cohort · HCN2 (19p13.3) · Ih Pacemaker Channel / TC-Dominant / LOF-Primary

HCN2 CHANNEL BIOLOGY — THALAMIC RELAY (TC) DOMINANT Ih PACEMAKER:
HCN2 (19p13.3) encodes the HCN2 subunit (863 aa) of the hyperpolarization-activated cyclic
nucleotide-gated (HCN) channel family. HCN channels generate the Ih current ("funny current"
or "pacemaker current") — a mixed Na+/K+ inward current ACTIVATED by HYPERPOLARISATION
(V1/2 activation ≈ −85 to −95 mV at rest; cAMP shifts to −70 to −80 mV).

HCN SUBFAMILY COMPARISON (4 members):
  · HCN1 (5p12):   FASTEST gating; V1/2 ≈ −70 mV; minimal cAMP shift (+3 mV); hippocampus/
                    neocortex L5 dendrites DOMINANT; Dravet-like DEE24 (GOF+LOF).
  · HCN2 (19p13.3): MEDIUM gating; V1/2 ≈ −90 mV; LARGE cAMP shift (+15 to +25 mV);
                    TC thalamic relay neurons DOMINANT; febrile seizures / GEFS+ / CAE (LOF).
  · HCN3 (1q22):   SLOW gating; V1/2 ≈ −77 mV; minimal cAMP sensitivity; olfactory bulb /
                    hypothalamus; rare epilepsy association.
  · HCN4 (15q24):  SLOWEST gating; V1/2 ≈ −100 mV; LARGEST cAMP shift (+25 mV); cardiac SA
                    node DOMINANT; sick sinus syndrome (not epilepsy).

HCN2-SPECIFIC STRUCTURAL FEATURES:
  1. Six transmembrane segments (S1-S6) per subunit; tetrameric channel (4 subunits).
  2. S4 voltage sensor INVERTED relative to K+/Na+ channels: detects HYPERPOLARISATION
     not depolarisation — unique molecular mechanism.
  3. CNBD (Cyclic Nucleotide-Binding Domain): HCN2 CNBD has highest cAMP affinity of the
     HCN subfamily (Kd ~0.1 μM). cAMP binding → C-linker conformational change → channel
     opens at LESS negative potentials → Ih increases.
  4. Ih reversal potential: ~−30 mV (mixed Na+/K+). Activated at rest (−65 to −95 mV) →
     produces DEPOLARISING inward current → partial membrane voltage "clamping" preventing
     excessive hyperpolarisation.
  5. HCN2 gating is temperature-sensitive (Q10 ≈ 1.5-2.0): fever → increased Ih activation
     rate (faster kinetics) → alters TC relay neuron timing → febrile seizure threshold.

HCN2 EXPRESSION PATTERN (why HCN2-LOF causes TC oscillation):
  · TC (thalamic relay) neurons: HCN2 is the DOMINANT HCN subunit (> HCN1 in TC); Ih
    generated primarily by HCN2 homotetramers or HCN1/HCN2 heterotetramers in TC neurons.
  · TRN (thalamic reticular nucleus): HCN4 > HCN2 > HCN1; TRN Ih modulates burst duration.
  · Brainstem: high HCN2 expression (locus coeruleus, dorsal raphe) → monoaminergic
    modulation of thalamo-cortical tone via cAMP pathway.
  · Hippocampus: moderate HCN2 (HCN1 predominant in CA1 dendrites).
  · Neocortex: moderate; HCN1 > HCN2 in layer V pyramidal neurons.

HCN2 LOF MECHANISM — THE KEY EPILEPSY STORY:
NORMAL FUNCTION in TC neurons:
  · After TRN GABA-B IPSP: TC membrane → −90 to −100 mV (deep hyperpolarisation).
  · Ih (HCN2-mediated) activates at this potential → slowly depolarises TC back toward
    −70 mV → limits the depth and duration of TC hyperpolarisation.
  · This Ih-mediated depolarising "sag" CONTROLS the timing and amplitude of the subsequent
    T-type Ca²⁺ (Cav3.1/Cav3.2) rebound LTCS (low-threshold Ca²⁺ spike).
  · Ih sag is the BRAKE on TC oscillatory amplitude.

HCN2 LOF PATHOPHYSIOLOGY:
  · Reduced Ih → SLOWER, WEAKER depolarising sag after TRN GABA-B IPSP.
  · TC membrane stays hyperpolarised LONGER → GREATER de-inactivation of Cav3.1/Cav3.2
    T-type Ca²⁺ channels (V1/2 inactivation ≈ −80 mV; longer at −100 mV = more Cav3 opens).
  · LARGER Cav3.1/Cav3.2 LTCS rebound → STRONGER TC burst → MORE powerful 3-Hz SWD.
  · Net effect: HCN2 LOF REMOVES the Ih BRAKE → runaway thalamo-cortical 3-Hz oscillation.
  · KEY experimental evidence: Ludwig et al. 2003 (Nat Neurosci) — HCN2 knockout mice:
    (1) SPONTANEOUS ABSENCE SEIZURES (3-4 Hz SWD, EEG-confirmed, behavioural arrest);
    (2) SINUS DYSRHYTHMIA (SA node HCN2 loss → heart rate irregularity — NOT bradycardia).
    This is the foundational genetic proof that HCN2 controls TC oscillatory amplitude.

FEVER + HCN2 (GEFS+ MECHANISM):
  · HCN2 has HIGH temperature sensitivity (Q10 2.0 for activation kinetics).
  · In LOF HCN2: fever → remaining HCN2 Ih kinetics ACCELERATED but current amplitude REDUCED.
  · Net effect: faster gating but less total Ih → further disruption of TC timing at fever
    temperatures → lower seizure threshold → febrile seizures.
  · GEFS+ pedigrees: heterozygous HCN2 LOF → reduced Ih ~50% → FS threshold ↓ → FS persist
    beyond 6Y (FS+) → some family members develop full CAE or GTCS-alone.

CAMP MODULATION — CLINICAL IMPLICATION:
  · Elevated cAMP (β-adrenergic stimulation, stress, exercise) → shifts HCN2 activation V1/2
    to less negative potentials → INCREASES Ih → paradoxically PROTECTS against oscillation.
  · In LOF HCN2: cAMP modulation partially RESCUES reduced Ih (smaller rescue than WT
    because LOF variants often affect CNBD directly or reduce surface expression).
  · Clinical relevance: emotional stress, exercise, caffeine → complex HCN2 phenotype
    modulation via cAMP pathway (can trigger OR suppress seizures depending on context).

PRECISION CONSIDERATIONS FOR HCN2 (different from HCN1-DEE24):
  1. NO DEDICATED PRECISION THERAPY for HCN2 GGE: no approved HCN opener (IVM not
     indicated in febrile/GGE phenotype).
  2. ETX (ethosuximide): INDIRECT mechanism — ETX blocks Cav3.1/Cav3.2 in TC neurons →
     reduces the LTCS rebound that is AMPLIFIED by HCN2 LOF. ETX Level A for CAE phenotype
     (addresses the downstream T-type Ca²⁺ consequence of HCN2 LOF).
  3. LAMOTRIGINE CONTRAINDICATED in LOF: LTG directly blocks HCN1/HCN2 channels
     (Poolos et al. 2002 Nat Neurosci) → further reduces already-low Ih → WORSENS TC
     oscillation → published case series: LTG worsened absence in LOF HCN2 → seizure
     INCREASE documented. This is the critical clinical safety rule for HCN2-LOF GGE.
  4. QUINIDINE: blocks HCN channels — also CONTRAINDICATED in LOF HCN2.
  5. FEVER MANAGEMENT: less aggressive than HCN1-DEE24 (GGE phenotype milder), but
     fever counselling mandatory for GEFS+ families.

HCN2 vs HCN1 — CLINICAL DIFFERENTIATION TABLE:
  Feature              | HCN1 (DEE24)                 | HCN2 (GEFS+/GGE)
  Severity             | SEVERE DEE (cognitive impact) | MILD-MODERATE GGE
  Mechanism            | GOF + LOF (dual)              | LOF predominant
  Main phenotype       | Dravet-like fever-sensitive DEE| Febrile seizures / GEFS+ / CAE
  OMIM                 | DEE24 (#615871)               | GEFS+6 associated (609200)
  LTG                  | CI in LOF; avoid pending assay| CI in LOF (consistent)
  IVM                  | CI in GOF; investigational LOF| Not relevant (GGE phenotype)
  Functional assay     | MANDATORY (GOF/LOF)           | Helpful; LOF usually primary
  TC vs hippocampal    | Hippocampus/cortex dominant   | TC neurons dominant
  KO mouse             | Seizures + dendrite defects   | Absence + sinus dysrhythmia

KEY REFERENCES:
  Ludwig A et al. 2003 Nat Neurosci 6(1):75-80 — HCN2 KO mice: absence epilepsy + SA rhythm
  Tang B et al. 2008 Nat Genet 40(9):1122-1127 — HCN2 mutations in febrile seizures
  DiFrancesco D. 2010 Physiol Rev 90(3):899-969 — HCN/If channel comprehensive biology
  Poolos NP et al. 2002 Nat Neurosci 5(8):767-774 — LTG blocks Ih in hippocampal neurons
  Bender RA & Baram TZ. 2008 Prog Neurobiol 84(2):128-149 — HCN channels in febrile seizures
  ILAE 2022 Operational classification of seizure types and epilepsy syndromes. Epilepsia 63(6)
"""

import random

random.seed(230)  # dashboard 230

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes, N=40)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "LOF-Febrile-Seizures-GEFS",
        "pct": 40,
        "etiology": "HCN2 LOF missense / haploinsufficiency — Febrile Seizures and GEFS+ spectrum",
        "mechanism": (
            "Heterozygous HCN2 loss-of-function variants (missense or haploinsufficiency via "
            "frameshift/nonsense) reduce surface-expressed HCN2 protein or impair channel gating. "
            "Approximately 50% reduction in TC neuron Ih → slower, weaker depolarising sag after "
            "TRN GABA-B IPSP → prolonged TC hyperpolarisation → greater Cav3.1/Cav3.2 de-inactivation "
            "window → lower febrile seizure threshold (fever raises HCN2 gating kinetics but cannot "
            "compensate for reduced channel number). LOF HCN2 pedigrees: simple FS in mildly affected "
            "members → FS+ (continuing beyond 6Y) in index cases. Tang et al. 2008 identified HCN2 "
            "missense variants (p.R591Q, p.V246L) in GEFS+ families with reduced Ih in heterologous "
            "expression. Onset: 6–36 months (first febrile seizure); often familial with variable "
            "expressivity (some relatives only simple FS, others full GEFS+ or CAE)."
        ),
        "typical_variants": "Missense (CNBD, S4-S5 linker, C-linker) · Nonsense/frameshift haploinsufficiency",
        "onset_age_years": 1,
        "outcome": "55–65% remit FS by age 6; 30% evolve GEFS+ (FS beyond 6Y); 15% develop CAE or GTCS-alone; overall prognosis GOOD vs HCN1-DEE24",
    },
    {
        "category": "LOF-CAE-Absence",
        "pct": 25,
        "etiology": "HCN2 LOF — Childhood Absence Epilepsy (CAE) via enhanced TC oscillation",
        "mechanism": (
            "HCN2 LOF in TC relay neurons removes the Ih BRAKE on thalamo-cortical 3-Hz oscillation. "
            "Normal Ih (HCN2-mediated) provides depolarising sag that LIMITS the depth and duration of "
            "TC hyperpolarisation after TRN GABA-B IPSPs → constrains Cav3.1/Cav3.2 de-inactivation "
            "→ moderates LTCS amplitude → limits 3-Hz SWD power. With HCN2 LOF: TC stays hyperpolarised "
            "LONGER → LARGER Cav3.1/Cav3.2 de-inactivation → BIGGER LTCS → STRONGER 3-Hz SWD → clinical "
            "absence seizures. This is exactly what Ludwig 2003 demonstrated in HCN2 KO mice: spontaneous "
            "3-4 Hz SWD absence seizures on EEG with behavioural arrest. CAE onset typically 5–10Y; "
            "3-Hz SWD activated by HV (>80% yield in HCN2-CAE). GGE phenotype: milder than CACNA1G/H "
            "GOF because mechanism is INDIRECT removal of brake, not direct T-type GOF."
        ),
        "typical_variants": "LOF missense in CNBD (reduces cAMP sensitivity → impaired Ih enhancement) · S4 voltage sensor",
        "onset_age_years": 7,
        "outcome": "60–70% CAE remit adolescence with ETX±VPA; 25% evolve JAE/JME requiring long-term AED; 10% drug-resistant",
    },
    {
        "category": "LOF-GEFS-Plus-Spectrum",
        "pct": 20,
        "etiology": "HCN2 LOF — Full GEFS+ spectrum (multi-member families, variable phenotype)",
        "mechanism": (
            "GEFS+ (Genetic Epilepsy with Febrile Seizures Plus): HCN2 LOF variants found in multi-"
            "generation pedigrees with multiple affected members showing DIFFERENT phenotypic severity "
            "from the same variant. Mechanism: reduced HCN2 Ih → lower febrile seizure threshold + "
            "altered TC oscillatory control. Variable expressivity within families: (a) Simple FS only "
            "(mildest — haploinsufficiency partially compensated); (b) FS+ (FS continuing beyond 6Y); "
            "(c) FS+ with absence (TC component); (d) FS+ with GTCS; (e) CAE or GGE without FS in some "
            "members (TC oscillation dominant). Penetrance: ~60-70% (some carriers unaffected). "
            "Modifier genes (other HCN subunits, cAMP pathway) likely determine phenotype severity."
        ),
        "typical_variants": "Low-penetrance LOF missense in transmembrane domains or C-linker (familial GEFS+ pedigrees)",
        "onset_age_years": 2,
        "outcome": "Familial prognosis: 50% remit; 30% GEFS+ long-term; 15% GGE long-term; 5% DRE; overall FAVOURABLE",
    },
    {
        "category": "LOF-GGE-GTCS-Alone",
        "pct": 10,
        "etiology": "HCN2 LOF — Genetic Generalised Epilepsy with GTCS-alone",
        "mechanism": (
            "HCN2 LOF → enhanced TC 3-Hz oscillation → cortical spread beyond absence-sustaining "
            "threshold → generalised tonic-clonic seizures (GTCS) without typical absence. These "
            "patients have TC oscillatory pathology but cortical inhibitory reserve prevents typical "
            "3-Hz SWD-locked absences; instead, TC burst propagates to GTCS directly. EEG: "
            "generalised irregular polyspike-wave (≥3 Hz) without classic 3-Hz absence; normal "
            "interictal unless sleep EEG captures polyspike-wave runs. Onset: adolescence–young adult. "
            "Phenotype: GTCS-alone on waking; no or rare absences; family history GEFS+. Treatment: "
            "VPA/LEV preferred (no ETX first-line — absence component absent)."
        ),
        "typical_variants": "LOF missense in CNBD or C-linker; low penetrance; often de novo in GTCS-alone",
        "onset_age_years": 15,
        "outcome": "VPA/LEV: 70–80% GTCS-free; occasional drug-resistance; lifelong AED needed in most",
    },
    {
        "category": "Phenocopy-GGE-No-HCN2",
        "pct": 5,
        "etiology": "GGE phenocopy — absence/GEFS+ without pathogenic HCN2 variant",
        "mechanism": (
            "GGE or GEFS+ phenotype clinically indistinguishable from HCN2-associated epilepsy but "
            "with no pathogenic HCN2 variant on comprehensive panel sequencing (WES or targeted). "
            "Differential aetiology: CACNA1G/H/I GOF (T-type Ca²⁺ enhancement — same TC oscillation "
            "phenotype), SCN1A GEFS+ (most common GEFS+ gene), GABRG2 LOF, other GGE gene. "
            "Clinical teaching: HCN2 accounts for a SMALL proportion of GEFS+ (<5%); SCN1A remains "
            "most common. Treat as standard GGE empirically (ETX/VPA per SANAD II); genetic counselling "
            "for familial GEFS+ to guide cascade testing."
        ),
        "typical_variants": "No HCN2 variant identified; treat as idiopathic GGE",
        "onset_age_years": 5,
        "outcome": "Standard GGE prognosis; empirical AED; ongoing genetic re-evaluation as panels expand",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5 types)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Febrile Seizures (FS/FS+)",
        "pct": 92,
        "eeg_pattern": (
            "Febrile seizure EEG (obtained post-ictally): focal or generalised slowing; no ictal "
            "EEG usually captured (brief). Inter-ictal: normal background; no SWD between febrile "
            "events. EEG during fever: generalised delta slowing (fever-induced); occasional focal "
            "theta-delta in frontotemporal region. KEY SIGN: EEG normalises within 24h post-fever — "
            "persistent SWD after fever resolves → suggests CAE/GGE component, not pure FS."
        ),
        "semiology": (
            "Febrile seizures: generalised tonic-clonic seizures occurring with fever (≥38°C). "
            "Duration: typically <5 min (simple FS); >15 min or focal = complex FS. HCN2-FS: "
            "clinically indistinguishable from non-genetic FS. FS+ feature: FS continuing beyond "
            "age 6 years (key clinical trigger for GEFS+ genetic workup). DISTINGUISH: FS+ is a "
            "GEFS+ descriptor (seizures with fever persisting past usual remission age); it does NOT "
            "mean a prolonged seizure."
        ),
        "clinical_tip": (
            "GEFS+ WORKUP TRIGGER: First FS: observe, paracetamol. FS BEYOND AGE 6 YEARS: genetic "
            "panel (SCN1A, SCN1B, SCN2A, HCN2, GABRG2 — SCN1A most common). Family history of FS "
            "in ≥2 members: GEFS+ pedigree → gene panel for proband + targeted testing for relatives. "
            "Rescue medication: buccal/nasal midazolam for FS ≥5 min."
        ),
    },
    {
        "type": "Typical Absence Seizures",
        "pct": 55,
        "eeg_pattern": (
            "Classic 3-Hz (2.5–4 Hz) bilateral, synchronous, symmetric spike-wave discharge (SWD). "
            "Abrupt onset and termination; ictal EEG: high-amplitude SWD with frontocentral maximum; "
            "background: normal. HCN2-CAE SWD: SIMILAR to idiopathic CAE — clinically indistinguishable "
            "by EEG alone. HV (hyperventilation) 3 minutes: induces absence in >80% of untreated cases "
            "— critical adequacy test for ETX (should NOT induce if ETX therapeutic). Photic driving "
            "superimposed in 20% (PPR). IPS (intermittent photic stimulation): low-yield in HCN2-CAE "
            "(unlike CACNA1H higher PPR rate)."
        ),
        "semiology": (
            "Typical absence: sudden behavioural arrest (5–30s); eye flutter (upward gaze/3-Hz blink); "
            "stare; may have oral automatisms; NO falling; NO post-ictal state; immediate full recovery. "
            "DISTINGUISHING from focal seizures: abrupt onset/offset; no post-ictal; bilateral EEG; "
            "paediatric onset (5–10Y). DISTINGUISHING from daydreaming: absence cannot be interrupted "
            "by touch/noise during episode; daydreaming can."
        ),
        "clinical_tip": (
            "ETX TRIAL THRESHOLD: ≥1 typical absence on EEG (even brief) → start ETX (SANAD II: ETX "
            "superior to VPA and LEV for absence outcome). ETX ADEQUACY TEST: 3-min HV during clinic "
            "EEG → should suppress SWD at therapeutic dose (40–100 mg/L). If HV still induces absence "
            "at 60 mg/L → increase toward 100 mg/L before declaring ETX failure."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "pct": 45,
        "eeg_pattern": (
            "Pre-ictal: generalised polyspike-wave or spike-wave burst (may be brief). Tonic phase EEG: "
            "fast (≥16 Hz) generalised recruiting rhythm. Clonic phase: polyspike-wave synchronised with "
            "clonic jerks (3–5 Hz decelerating). Post-ictal: generalised suppression then diffuse delta "
            "slowing (2–3 Hz). In HCN2-GGE: GTCS typically occurs on WAKING from sleep (morning GTCS "
            "most common pattern in JAE/JME evolution); sleep EEG: IED burden peaks in NREM stage 2-3 "
            "(TC oscillatory enhancement by HCN2 LOF potentiated by thalamo-cortical NREM rhythms)."
        ),
        "semiology": (
            "GTCS sequence: loss of consciousness → tonic stiffening (10–20s; ictal cry from glottis) → "
            "clonic jerks (30–120s; progressively coarser then decelerating) → flaccidity → post-ictal "
            "confusion (minutes–hours). In GEFS+: GTCS triggered by fever (febrile GTCS) or sleep "
            "deprivation/alcohol. In JAE/JME evolution: morning GTCS on waking; may have preceding "
            "absence cluster (EEG: polyspike-wave bursts pre-GTCS)."
        ),
        "clinical_tip": (
            "MORNING GTCS + HCN2: HIGH-YIELD INVESTIGATION — 24h sleep EEG or sleep-deprived EEG to "
            "capture polyspike-wave. FIRST GTCS MANAGEMENT: VPA or LEV (ETX not effective for GTCS). "
            "AVOID CBZ/OXC/PHT (GGE-aggravating: may WORSEN absence or myoclonic components). "
            "DRIVING: national guidelines (typically ≥12 months seizure-free required)."
        ),
    },
    {
        "type": "Febrile Seizures Plus (FS+)",
        "pct": 35,
        "eeg_pattern": (
            "FS+ interictal EEG: normal between febrile events in childhood; may develop GGE changes "
            "(3-Hz SWD, polyspike-wave) in adolescence if phenotype evolves. Key EEG finding: FS+ "
            "without GGE evolution → normal EEG (cannot confirm FS+ by EEG alone — diagnosis is "
            "CLINICAL). FS+ WITH GGE evolution: interictal SWD (1–4 Hz) on routine or sleep EEG "
            "preceding the first afebrile seizure — this is the EEG warning of GEFS+ phenotype shift."
        ),
        "semiology": (
            "FS+ (Febrile Seizures Plus): GEFS+ defining phenotype — typical febrile seizures that "
            "CONTINUE BEYOND AGE 6 YEARS (when FS would normally have remitted). May also have "
            "AFEBRILE seizures (absence, GTCS) — these define the GEFS+ SPECTRUM. FS+ alone: "
            "febrile GTCS only; no afebrile seizures. GEFS+ with FS+ plus absence: most common "
            "HCN2 pedigree phenotype. Critical clinical point: FS+ IS BENIGN — reassure families. "
            "Mean FS+ remission: 14Y. Reassess if GTCS emerges."
        ),
        "clinical_tip": (
            "FS+ MANAGEMENT: no daily AED required for pure FS+ (rescue midazolam for prolonged FS). "
            "INDICATION for daily AED in FS+: ≥3 FS in 6 months; complex FS; status epilepticus; "
            "or PARENTAL PREFERENCE with counselling. Genetic testing MANDATORY for FS+ families "
            "(SCN1A first — GEFS+ most common; then HCN2, GABRG2, SCN1B panel)."
        ),
    },
    {
        "type": "Myoclonic Jerks",
        "pct": 15,
        "eeg_pattern": (
            "EEG correlate of myoclonic jerks: brief (<100 ms) generalised polyspike-wave burst; "
            "bifrontal maximum; time-locked to EMG jerk on polygraphy. In HCN2-GGE with myoclonic "
            "phenotype: polyspike-wave (1–3 spikes preceding wave); background otherwise normal. "
            "Photoparoxysmal response (PPR) occasionally present (~18%). Morning bias: polyspike-wave "
            "clusters on waking EEG (similar to JME phenotype). ALERT: if myoclonic EEG shows atypical "
            "features (slow polyspike-wave <2.5 Hz, multifocal, asymmetric) → rethink HCN2 → consider "
            "LGS or other DEE if also developmental regression."
        ),
        "semiology": (
            "Myoclonic jerks: brief (<2s) involuntary muscle contractions; bilateral; predominantly "
            "upper limbs / shoulders; morning predominance (30–60 min after waking); associated with "
            "brief consciousness impairment (<0.5s) or consciousness preserved. DISTINGUISH from "
            "physiological hypnic jerks (normal, on sleep onset only). In HCN2 GGE: myoclonic jerks "
            "seen in JME-like phenotype (GEFS+ family member with JME features) — RARE but documented."
        ),
        "clinical_tip": (
            "MYOCLONIC + ABSENCE (JME-OVERLAP): if HCN2-GGE patient develops myoclonic jerks PLUS "
            "absence → full JME assessment. VPA/LEV preferred for JME-overlap (ETX may worsen "
            "myoclonic component without VPA cover). CRITICAL SAFETY: AVOID LTG if myoclonic "
            "component present (LTG can worsen myoclonic in JME regardless of HCN2 status)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever (≥38°C)",
        "pct": 95,
        "mechanism": (
            "Primary trigger in GEFS+ phenotype. Fever alters HCN2 Ih kinetics (Q10 2.0): higher "
            "temperature → faster HCN2 gating rate but REDUCED total Ih amplitude in LOF → TC "
            "timing disrupted → lower FS threshold. Counsel: paracetamol (15 mg/kg) at first sign "
            "of fever ≥38°C; buccal midazolam ready for seizure >5 min. Sick-day plan at every visit."
        ),
    },
    {
        "trigger": "Febrile Illness",
        "pct": 78,
        "mechanism": (
            "Concurrent illness (viral URTIs, otitis, gastroenteritis) compounds fever-related Ih "
            "disruption with systemic inflammatory cytokines (IL-6, TNF-α) → additional neuronal "
            "excitability via: (1) PGE2 modulation of GABA-A receptors; (2) direct cytokine effects "
            "on HCN channels (IL-1β reduces Ih in hippocampal neurons). Counsel: hydration; antipyretic "
            "adherence; avoid known sick-day AED interactions."
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 72,
        "mechanism": (
            "Sleep deprivation → increased NREM thalamo-cortical synchrony → greater TC oscillatory "
            "burden potentiated by HCN2 LOF (Ih brake already reduced, NREM further amplifies). "
            "Sleep EEG shows IED clustering in NREM stage 2-3. Counsel: minimum 8h/night; regular "
            "sleep schedule (particularly relevant for adolescents with JME-like phenotype evolution)."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 65,
        "mechanism": (
            "AED discontinuation → loss of pharmacological suppression of TC oscillation → rebound "
            "seizure. ETX withdrawal: Ih-independent (ETX works on T-type Ca²⁺, not HCN2) → absence "
            "rebound within 24-48h of missed doses. Counsel: consistent dosing; pill organiser; "
            "travel supplies. ETX twice-daily scheduling improves adherence vs three-times-daily."
        ),
    },
    {
        "trigger": "Hyperventilation (HV)",
        "pct": 60,
        "mechanism": (
            "HV → respiratory alkalosis → decreased ionised Ca²⁺ → increased neuronal excitability → "
            "absence seizure precipitation. TC relay neuron Cav3.1/Cav3.2 threshold lowered by "
            "alkalosis (shifts T-type voltage-dependence) — amplified in HCN2 LOF where TC LTCS "
            "threshold already lowered. HV is the standard ABSENCE PROVOCATION test: 3 min HV during "
            "clinic EEG. Clinical teaching: EVERY untreated HCN2-CAE patient should show HV-induced "
            "absence; failure to do so → question diagnosis or ETX adequacy if pre-treated."
        ),
    },
    {
        "trigger": "Stress / Emotional Arousal",
        "pct": 48,
        "mechanism": (
            "Psychological stress → sympathetic activation → elevated catecholamines → cAMP surge in "
            "TC neurons. cAMP shifts HCN2 activation V1/2 more positive (less negative) → INCREASES "
            "residual Ih partially. PARADOX: stress-elevated cAMP may PARTIALLY COMPENSATE for HCN2 "
            "LOF → reduced seizure rate during acute stress in some patients. However, post-stress "
            "exhaustion → cAMP normalisation → re-exposure of full LOF effect → delayed seizure risk. "
            "Counsel: stress management; recognise the post-stress vulnerability window."
        ),
    },
    {
        "trigger": "Catamenial (Menstrual Cycle)",
        "pct": 22,
        "mechanism": (
            "Pre-menstrual oestrogen peak → increased neuronal excitability (E2 downregulates GABA-A "
            "α4-containing receptors in thalamus; enhances NMDA). Progesterone withdrawal (days 22-28) "
            "→ GABA-A neurosteroid withdrawal → reduced inhibitory tone. In HCN2-GGE females: "
            "catamenial pattern in ~22% — absence/GTCS clustering perimenstrually. Management: "
            "cycle-adjusted CLB (5-10 mg days 20-28) proven effective in catamenial epilepsy."
        ),
    },
    {
        "trigger": "Alcohol Consumption",
        "pct": 20,
        "mechanism": (
            "Alcohol (ethanol) POTENTIATES GABA-A → initial seizure suppression; ALCOHOL WITHDRAWAL "
            "→ GABA-A downregulation → acute excitability surge. In GGE patients: alcohol even in "
            "moderate amounts disrupts sleep architecture → indirect trigger via sleep deprivation. "
            "Counsel: abstinence preferred for GGE; if drinks, maximum 1-2 units; NEVER rapid "
            "cessation after heavy drinking (withdrawal seizure risk)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (8)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Ethosuximide (ETX) — Level A (CAE/Absence phenotype)",
        "level": "Level A — SANAD II 2021 (superior for absence vs VPA and LEV); SANAD 2007",
        "dose": "Adult: 500–1500 mg/day in 2 divided doses; Child: 20–40 mg/kg/day; TDM target: 40–100 mg/L",
        "moa": (
            "ETX blocks T-type (Cav3.1/Cav3.2) Ca²⁺ channels in TC relay neurons (Coulter 1989 "
            "Ann Neurol) → reduces LTCS rebound amplitude → suppresses TC burst → ATTENUATES the "
            "3-Hz SWD that is amplified by HCN2 LOF. INDIRECT mechanism for HCN2-CAE: ETX addresses "
            "the DOWNSTREAM T-type Ca²⁺ consequence of HCN2 LOF (does not directly restore Ih). "
            "ETX also blocks Na+ channels (minor) and GABA-A currents (minor). "
            "PRIMARY FIRST-LINE for typical absence / CAE phenotype in HCN2 GGE."
        ),
        "efficacy": "SANAD II: ETX 45% best seizure outcome vs VPA 44% vs LEV 35% for absence; SANAD 2007: ETX non-inferior VPA, superior tolerability",
        "safety": "GI (nausea, vomiting, anorexia) most common (take with food); headache; rare: lupus-like, blood dyscrasias, Stevens-Johnson (monitor); NO hepatotoxicity (POLG-independent). SAFE in pregnancy relative risk (LOW teratogenicity vs VPA).",
        "monitoring": "TDM q6M (40–100 mg/L; target 80–100 mg/L if partial response); HV-SWD abolition as bedside adequacy test; FBC at 3M (rare aplastic anaemia); LFT not required",
        "hcn2_note": "HCN2-SPECIFIC: ETX Level A for CAE/absence component. For pure GTCS-alone HCN2-GGE (no absence): ETX NOT INDICATED — use VPA/LEV instead. ETX + VPA combination if breakthrough GTCS on ETX monotherapy.",
    },
    {
        "drug": "Valproate (VPA) — Level B (broad-spectrum; POLG1-mandatory; VPPP females)",
        "level": "Level B — SANAD II 2021; NICE NG217 (second-line for absence; first-line for GTCS/JME-overlap)",
        "dose": "Adult: 500–2500 mg/day; Child: 20–60 mg/kg/day; TDM: 50–100 mg/L; slow titration",
        "moa": (
            "Broad-spectrum mechanism: (1) GABA-T inhibition → increased synaptic GABA; "
            "(2) T-type Ca²⁺ channel block (Cav3.1/Cav3.2 at therapeutic concentrations → "
            "addresses same downstream target as ETX); (3) NaV channel block (persistent Na+ "
            "current reduction); (4) HCN1/HCN2 Ih modulation (minor — VPA reported to modulate "
            "Ih in some models, potentially partially compensating HCN2 LOF). Preferred for "
            "patients with GTCS + absence components."
        ),
        "efficacy": "SANAD II: VPA comparable to ETX for absence; superior for GTCS; SANAD 2007: VPA vs ETX vs LTG — VPA best for GTCS+absence combined; JME: VPA best long-term seizure control",
        "safety": "MAJOR RISKS: hepatotoxicity (POLG1 — ABSOLUTE CI); teratogenicity (NTD 1-2%, VPA syndrome — VPPP mandatory females 9-55Y); weight gain; tremor; hair loss; metabolic: hyperammonaemia. Pancreatitis: rare but serious.",
        "monitoring": "POLG1 testing MANDATORY before initiation; VPPP annual form (females); TDM q3M; LFT + FBC + ammonia q3M; weight monthly; folic acid 5 mg/day; pregnancy test before each prescription",
        "hcn2_note": "HCN2-GGE: VPA preferred over LEV for patients with absence + GTCS combination (SANAD II — VPA better for combined GGE). FEMALES: prefer ETX (CAE-only) or LEV (GTCS/JME) to AVOID VPA VPPP burden if seizure type permits.",
    },
    {
        "drug": "Levetiracetam (LEV) — Level B (POLG-safe; females; GTCS/JME phenotype)",
        "level": "Level B — SANAD II 2021; ILAE 2022 (POLG-safe first-line for females and POLG-positive)",
        "dose": "Adult: 1000–3000 mg/day in 2 divided doses; Child: 20–60 mg/kg/day; IV load available",
        "moa": (
            "SV2A (synaptic vesicle protein 2A) binding → modulates vesicle trafficking and "
            "neurotransmitter release; GABA-A potentiation (indirect); inhibits N-type Ca²⁺ channels; "
            "reduces high-voltage-activated Ca²⁺ current. POLG-SAFE (no mitochondrial toxicity). "
            "For HCN2-GEFS+ females: LEV + ETX combination avoids VPA VPPP burden."
        ),
        "efficacy": "SANAD II: LEV WORST for absence (35% best outcome — ETX preferred for pure CAE); LEV effective for GTCS-alone and JME-overlap (65–75% GTCS-free); well-tolerated except behavioural",
        "safety": "Behavioural: irritability, aggression, mood lability (up to 15%); pyridoxine B6 50-100 mg/day may mitigate; no hepatotoxicity; no teratogenicity (SAFE in pregnancy); NO weight gain",
        "monitoring": "TDM q6M (20–40 mg/L); mood assessment; B6 supplementation; renal function (LEV renally cleared — reduce dose if eGFR <50)",
        "hcn2_note": "HCN2-CAE (pure absence): ETX PREFERRED over LEV (SANAD II). HCN2-GTCS-alone or GEFS+ with GTCS: LEV preferred in FEMALES (avoids VPPP). HCN2 + POLG1 positive: LEV mandatory (VPA absolutely contraindicated).",
    },
    {
        "drug": "Clobazam (CLB) — Level B (catamenial; nocturnal; GEFS+ adjunct)",
        "level": "Level B — ILAE 2022 catamenial epilepsy; Cochrane adjunct trials; NICE NG217",
        "dose": "Adult adjunct: 10–30 mg/day (nocturnal or cycle-adjusted); Child: 0.25–1 mg/kg/day in 1-2 doses",
        "moa": (
            "Benzodiazepine preferring α2/α3 GABA-A subunits (lower sedation than diazepam which "
            "also hits α1). CLB → GABA-A chloride influx → neuronal hyperpolarisation → reduces "
            "TC relay neuron excitability. Norclobazam (active metabolite): t1/2 36-46h → sustained "
            "trough levels. In catamenial epilepsy: CLB days 20-28 exploits the highest catamenial "
            "seizure risk window with short-course therapy → less tolerance risk."
        ),
        "efficacy": "Catamenial: cycle-adjusted CLB 40-65% reduction in perimenstrual seizure density (Feely 1982; Reddy 2004); nocturnal GEFS+ adjunct: 50-60% reduction in nocturnal GTCS; tolerance may develop after 3-6 months continuous use",
        "safety": "Sedation; cognitive dulling with continuous use; tolerance/dependence (avoid >6-8 weeks continuous without break); withdrawal seizures if abrupt stop; norclobazam drug interactions (CYP2C19); teratogenicity LOW (benzodiazepine class — cleft palate risk minimal at low dose)",
        "monitoring": "Norclobazam TDM 50-300 ng/mL; mood/cognition assessment; tolerance monitoring (breakthrough seizures); schedule regular drug holidays if continuous use",
        "hcn2_note": "HCN2-GGE catamenial pattern: FIRST-LINE CLB cycle-adjusted (days 20-28, 5-10 mg/day). Nocturnal GEFS+: low-dose nocturnal CLB (5-10 mg QHS). Not monotherapy for primary absence.",
    },
    {
        "drug": "Lamotrigine (LTG) — CAUTION: HIGH RISK in LOF HCN2 (Ih blocker)",
        "level": "CAUTION — Level C for GTCS-alone only (NEVER for pure absence or LOF HCN2 without caution)",
        "dose": "Slow titration mandatory: 25 mg/2 weeks → 25 mg/week; target 100-300 mg/day (monotherapy); HALF rate with VPA co-administration",
        "moa": (
            "LTG: (1) NaV block (principal mechanism — reduces persistent Na+ current); (2) DIRECT Ih "
            "BLOCK (Bois et al. 1996 J Physiol; Poolos et al. 2002 Nat Neurosci): LTG reduces Ih "
            "in hippocampal neurons by ~30-50% at therapeutic concentrations (10-15 μM plasma level). "
            "This Ih-blocking property is CLINICALLY IMPORTANT: in LOF HCN2 where Ih is ALREADY "
            "REDUCED, LTG further suppresses residual Ih → WORSENS TC oscillation → ABSENCE "
            "AGGRAVATION. Published case series (Marini 2010 Epilepsia) document LTG-induced "
            "absence worsening in GGE patients — mechanism likely Ih block exacerbating LOF."
        ),
        "efficacy": "SANAD 2007: LTG inferior to VPA for GTCS-alone in GGE; NOT effective for typical absence (may worsen); SANAD II: LTG worse than ETX for absence; LTG reasonable for GTCS-alone ONLY",
        "safety": "SERIOUS: SJS/TEN (0.1% — slow titration mandatory); rash (10%); diplopia; dizziness; blurred vision; PARADOXICAL WORSENING of absence/myoclonic in GGE — THIS IS THE KEY SAFETY RISK FOR HCN2-LOF",
        "monitoring": "EEG BEFORE and AFTER LTG initiation in any GGE/HCN2 patient (to detect absence worsening); TDM 3-15 mg/L; slow titration; rash surveillance — stop immediately if any rash",
        "hcn2_note": "HCN2-LOF: LTG HIGH RISK — LTG is an Ih blocker (Poolos 2002); worsens LOF HCN2. RULE: in any patient with confirmed HCN2 LOF + absence phenotype → LTG CONTRAINDICATED (same logic as HCN1-LOF). For HCN2-GTCS-alone without absence: LTG may be used with EEG monitoring — BUT prefer VPA/LEV. ALWAYS obtain EEG 4-8 weeks after LTG initiation in HCN2-GGE to detect silent absence worsening.",
    },
    {
        "drug": "Zonisamide (ZNS) — Level C (dual T-type + NaV; DRE adjunct)",
        "level": "Level C — NICE NG217 adjunct; Cochrane DRE; limited GGE-specific trial data",
        "dose": "Adult: 100–500 mg/day (once daily or BD); Child: 4–12 mg/kg/day; slow titration 25-50 mg/2-4 weeks",
        "moa": (
            "Dual mechanism: (1) T-type Ca²⁺ block (Cav3.1/Cav3.2 — same target as ETX, addressing "
            "downstream consequence of HCN2 LOF); (2) Persistent NaV block; (3) Carbonic anhydrase "
            "inhibition (→ acidosis → slight anticonvulsant effect). In HCN2-GGE: ZNS addresses "
            "both the T-type Ca²⁺ (absence mechanism) and NaV (GTCS mechanism) → potentially useful "
            "in combined phenotype. CARBONIC ANHYDRASE inhibition → METABOLIC ACIDOSIS → bicarbonate "
            "monitoring mandatory (ZNS + KD: compound acidosis risk — see monitoring)."
        ),
        "efficacy": "Open-label GGE adjunct: 40-55% responder rate for absence + GTCS combination; ZNS vs ETX: no head-to-head; ZNS second-line after ETX/VPA failure",
        "safety": "Renal stones (3-4% — hydration 2L/day); weight loss (anorexia); cognitive dulling; metabolic acidosis; oligohydrosis/hyperthermia in children; CAUTION with carbonic anhydrase inhibitors (risk compounded)",
        "monitoring": "Serum HCO3 q6M (threshold: <18 mmol/L → dose reduce or switch); renal function; weight; urinalysis for stones; temperature monitoring in children (oligohydrosis risk)",
        "hcn2_note": "HCN2-GGE: ZNS useful adjunct for DRE with mixed absence + GTCS phenotype (dual T-type + NaV action). NOT first-line. Avoid ZNS + KD combination without very careful HCO3 monitoring (dual acidosis risk).",
    },
    {
        "drug": "Ketogenic Diet (KD) — Level B (DRE; HCN2-independent mechanism)",
        "level": "Level B — ILAE Dietary Therapies 2018; Cochrane DRE; NICE NG217 adjunct",
        "dose": "Classic KD 4:1 (fat:carb+protein); MAD (modified Atkins); LGIT; BKPD; target BHB 2-4 mmol/L",
        "moa": (
            "Multiple anticonvulsant mechanisms — importantly INDEPENDENT of HCN2 Ih: "
            "(1) β-hydroxybutyrate (BHB) → KATP channel activation → neuronal membrane "
            "hyperpolarisation (DOES NOT worsen HCN2 LOF — different K+ channel, physiological); "
            "(2) Mitochondrial complex I support → reduced reactive oxygen species; "
            "(3) Increased GABA synthesis (acetoacetate inhibits GABA transaminase); "
            "(4) BDNF upregulation → enhanced inhibitory synaptogenesis; "
            "(5) HCN expression: KD may modestly UPREGULATE HCN2 expression in TC neurons "
            "(preclinical data) → potentially partially compensates HCN2 LOF. "
            "MECHANISTICALLY APPROPRIATE for HCN2-GGE (HCN2-independent anticonvulsant action + "
            "possible partial Ih restoration)."
        ),
        "efficacy": "DRE GGE: 40-55% ≥50% seizure reduction (similar to CACNA1G/H/I T-type GGE); MAD better tolerability; sustained 3-year efficacy in 30%",
        "safety": "Dyslipidaemia (LDL↑ — monitor); kidney stones (acidosis + dehydration); GI intolerance; growth retardation in children if prolonged; selenium/carnitine deficiency; QT prolongation (monitor ECG); NOT for POLG1 (ketosis amplifies mitochondrial stress)",
        "monitoring": "BHB daily (target 2-4 mmol/L); lipid panel q3M; weight/height monthly; renal function q6M; carnitine level q6M; urine calcium:creatinine ratio; ECG baseline",
        "hcn2_note": "HCN2-GGE DRE: KD HIGH-PRIORITY 2nd-line after ≥2 AED failures. KD mechanism INDEPENDENT of HCN2 Ih → safe in LOF context. KD + ZNS: monitor HCO3 closely (dual carbonic anhydrase/acidosis risk). Possible HCN2 expression upregulation benefit (preclinical).",
    },
    {
        "drug": "Phenobarbital (PB) — Level C (acute febrile status; bridge therapy)",
        "level": "Level C — NICE NG217 acute status; WHO Essential Medicine; not first-line chronic GGE",
        "dose": "Febrile SE load: 20 mg/kg IV (max 1 g) at ≤60 mg/min; chronic: 1-5 mg/kg/day; TDM: 10-40 mg/L",
        "moa": (
            "PB: GABA-A positive allosteric modulator (increases Cl- channel open time) + NaV block. "
            "Effective for febrile status epilepticus (acute IV load) when benzodiazepines fail. "
            "In GEFS+/HCN2 febrile SE: PB IV is second-line rescue (after IV lorazepam/diazepam). "
            "CHRONIC PB: not recommended for GGE long-term (cognitive side effects; tolerance; "
            "WHO Essential Medicine status → appropriate low-resource acute use)."
        ),
        "efficacy": "Febrile SE: PB 60-70% termination rate (after failed BZD); chronic GGE: inferior to VPA/ETX for absence (no absence efficacy); some GTCS reduction",
        "safety": "Sedation; cognitive dulling (learning/memory); hyperactivity in children; tolerance/dependence; enzyme induction (CYP3A4/2C — drug interactions); osteomalacia; teratogenicity class D",
        "monitoring": "TDM q6M (10-40 mg/L); cognitive assessment; bone density (long-term); withdrawal plan (taper ≥4 weeks)",
        "hcn2_note": "HCN2-GEFS+ febrile SE: PB IV is the standard second-line acute agent after benzodiazepine failure. NOT for chronic GEFS+/CAE management. Consider EARLIER KD initiation rather than chronic PB in young children with DRE.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (5)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "CBZ / OXC / PHT (Carbamazepine / Oxcarbazepine / Phenytoin) — ABSOLUTE CI",
        "risk": "ABSOLUTE CONTRAINDICATION in GGE/absence phenotype",
        "mechanism": (
            "NaV blockers selectively suppress fast-spiking inhibitory interneurons (PV+ cells) "
            "more than pyramidal neurons at therapeutic concentrations → disinhibition of thalamic "
            "relay system → PARADOXICAL worsening of absence seizures and 3-Hz SWD. Well-documented "
            "class effect: CBZ/OXC/PHT → absence status epilepticus, absence worsening in GGE. "
            "HCN2-GGE: same absolute rule applies. Even for GTCS-alone subtype with no current absence "
            "— CBZ/OXC/PHT can UNMASK absence tendency or trigger absence status. NEVER prescribe as "
            "empirical first-line in any GGE patient."
        ),
        "alternative": "ETX (absence); VPA/LEV (GTCS); CLB (nocturnal/catamenial)",
    },
    {
        "drug": "Lamotrigine (LTG) — HIGH RISK in LOF HCN2 (Ih blocker → absence worsening)",
        "risk": "HIGH RISK: Ih block worsens LOF HCN2 → absence aggravation; contraindicated in confirmed LOF + absence phenotype",
        "mechanism": (
            "LTG is a clinically significant Ih (HCN1/HCN2) blocker at therapeutic plasma "
            "concentrations (Poolos 2002 Nat Neurosci; Bois 1996 J Physiol). In HCN2-LOF GGE: "
            "Ih is ALREADY reduced 50% → LTG further reduces residual Ih → TC relay neurons stay "
            "hyperpolarised longer → greater T-type Ca²⁺ de-inactivation → STRONGER 3-Hz SWD → "
            "INCREASED absence burden. Published evidence: LTG-induced absence aggravation in GGE "
            "(Marini 2010 Epilepsia 51:1542). For LOF HCN2 + absence phenotype: LTG CONTRAINDICATED. "
            "For GTCS-alone with NO absence + confirmed HCN2 carrier (not functionally confirmed LOF): "
            "LTG may be cautiously considered WITH EEG monitoring — prefer VPA/LEV."
        ),
        "alternative": "ETX (absence); VPA/LEV (GTCS); ZNS (DRE combined phenotype)",
    },
    {
        "drug": "Tiagabine (TGB) — ABSOLUTE CI (NCSE in GGE)",
        "risk": "ABSOLUTE CONTRAINDICATION: TGB → non-convulsive status epilepticus (NCSE) in GGE patients",
        "mechanism": (
            "TGB blocks GABA reuptake transporters (GAT-1) → INCREASES tonic extracellular GABA → "
            "paradoxically activates GABA-B receptors on TC relay neurons → ENHANCED TC rebound → "
            "PROMOTES thalamo-cortical 3-Hz oscillation → ABSENCE STATUS / NCSE in GGE. "
            "Class effect documented across GGE types. In HCN2-GGE where TC oscillatory gain is "
            "already elevated due to LOF Ih → TGB is catastrophic → ABSOLUTE CI. No exceptions."
        ),
        "alternative": "CLB/VPA/ETX/LEV for breakthrough seizures",
    },
    {
        "drug": "Valproate (VPA) + POLG1 mutation — ABSOLUTE CI (Alpers-Huttenlocher)",
        "risk": "ABSOLUTE CONTRAINDICATION: VPA in POLG1 pathogenic variant carriers → fulminant hepatic failure",
        "mechanism": (
            "POLG1 encodes mitochondrial DNA polymerase γ. POLG1 pathogenic variants (p.A467T, "
            "p.W748S most common) → mitochondrial DNA depletion → impaired mtDNA replication. "
            "VPA inhibits mitochondrial β-oxidation → in POLG1 carriers → catastrophic energy "
            "failure in hepatocytes → Alpers-Huttenlocher syndrome (fulminant hepatic failure, "
            "neurological regression, death). POLG1 TESTING MANDATORY before VPA initiation in "
            "any epileptic encephalopathy or refractory epilepsy. For HCN2-GGE with mild phenotype: "
            "POLG1 testing recommended before VPA (especially if early onset, regression, or "
            "family history of mitochondrial disease)."
        ),
        "alternative": "LEV (POLG-safe, first-line for POLG1 carriers); ETX (absence); CLB adjunct",
    },
    {
        "drug": "Vigabatrin (VGB) — HIGH RISK in GGE (NCSE + Visual Field Defects)",
        "risk": "HIGH RISK: VGB promotes absence/NCSE in GGE; cumulative irreversible visual field defects (VFD)",
        "mechanism": (
            "VGB irreversibly inhibits GABA-T → excess GABA → same GABA-B TC activation mechanism "
            "as TGB → promotes TC oscillation → absence/NCSE risk in GGE (less severe than TGB "
            "but still HIGH RISK). Additionally: VGB causes cumulative irreversible visual field "
            "constriction (VFD) — up to 30-40% with prolonged use (Aldington 2000). SHARE REMS "
            "(VGB): mandatory baseline + q3M ERG/visual field assessment. For HCN2-GGE: VGB has "
            "no GGE indication — avoid."
        ),
        "alternative": "ACTH/prednisolone (if infantile spasms co-exist); ETX/VPA/LEV (GGE management)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG1 testing before VPA", "frequency": "Mandatory (once, before first VPA prescription)"},
    {"item": "ETX TDM (therapeutic drug monitoring)", "frequency": "At 4 weeks initiation, then q6M (target 40–100 mg/L; upper range 80–100 mg/L before failure declaration)"},
    {"item": "HV-SWD abolition test (ETX adequacy)", "frequency": "At 6-8 weeks post-ETX initiation: 3-min HV during clinic EEG — SWD should be suppressed at therapeutic dose"},
    {"item": "EEG monitoring (post-LTG initiation)", "frequency": "If LTG initiated in HCN2-GGE: baseline EEG then repeat 4-8 weeks after (detect silent absence worsening)"},
    {"item": "VPA TDM", "frequency": "q3M (target 50–100 mg/L); check at 2 weeks if dose changed"},
    {"item": "LFT + FBC + serum ammonia (VPA)", "frequency": "Baseline, 4 weeks, then q3M on VPA"},
    {"item": "EEG baseline + annual", "frequency": "Baseline routine EEG (ideally awake+drowsy); annual sleep EEG or if seizure worsening"},
    {"item": "Cognitive / developmental assessment", "frequency": "q6M (Bayley-4 if <3Y; WPPSI/WISC if school-age; WAIS adults); particularly if CAE with school impact"},
    {"item": "MRI brain at diagnosis", "frequency": "Once at diagnosis (normal expected in HCN2-GGE; structural abnormality → rethink diagnosis or DRE workup)"},
    {"item": "VPPP documentation (females, VPA)", "frequency": "Annual mandatory VPPP form (females 9-55Y on VPA): specialist + GP co-signature; 2 contraceptive methods; pregnancy test"},
    {"item": "SUDEP risk assessment", "frequency": "Annual: uncontrolled GTCS ≥3/year + nocturnal + DRE = HIGH RISK; rescue plan; wearable seizure alert; nocturnal supervision counselling"},
    {"item": "HV-SWD response (diagnostic)", "frequency": "At diagnosis: 3-min HV during EEG to confirm absence phenotype and HV sensitivity; repeat if clinical doubt"},
    {"item": "Catamenial diary (females)", "frequency": "Monthly cycle diary if catamenial pattern suspected; compare seizure dates to cycle phases; guide CLB timing"},
    {"item": "Genetic counselling", "frequency": "Once at diagnosis; repeat at family planning; cascade testing for first-degree relatives in GEFS+ families (SCN1A/HCN2 panel)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS  (6)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Infancy — Febrile Seizures (6M–3Y)",
        "key_issues": (
            "First presentation: febrile seizure. GEFS+ genetic workup trigger: FS+ (>6Y), ≥3 FS, "
            "family history. Rescue: buccal/nasal midazolam. Fever counselling. No daily AED for "
            "simple FS. Paediatric neurology referral for complex FS or FS+. HCN2 gene panel at first "
            "GEFS+ suspicion."
        ),
    },
    {
        "window": "Early Childhood — CAE Onset (4–10Y)",
        "key_issues": (
            "CAE onset: typical absences; school impact (absences during lessons); HV induction in "
            "clinic; ETX initiation (Level A). School seizure action plan. AVOID CBZ/OXC/PHT. "
            "Cognitive monitoring (school performance). ETX adequacy HV test at 6-8 weeks. "
            "Absence diary for parents."
        ),
    },
    {
        "window": "Late Childhood / Adolescence (10–18Y)",
        "key_issues": (
            "CAE remission assessment (60-70% remit by 13-14Y); if absence persists → JAE/JME "
            "phenotype reclassification. GTCS emergence (morning GTCS): add VPA or LEV. Adolescent "
            "triggers: alcohol counselling; sleep hygiene; driving restrictions (seizure-free period). "
            "VPA in females: VPPP discussion by age 12Y."
        ),
    },
    {
        "window": "Female Reproductive Years",
        "key_issues": (
            "VPPP MANDATORY if VPA. Catamenial pattern identification (CLB cycle-adjusted). Pregnancy "
            "planning: switch from VPA → ETX (CAE) or LEV (GTCS) ≥3 months before conception. Folic "
            "acid 5 mg/day. Contraception: enzyme-inducers reduce pill efficacy (not relevant for "
            "ETX/LEV/CLB but relevant if ZNS). Obstetric neurologist referral for pregnancy planning."
        ),
    },
    {
        "window": "Seizure-Free Monitoring (12M goal)",
        "key_issues": (
            "Target ≥12 months seizure-free before driving clearance. AED tapering discussion: CAE "
            "remission ≥3 seizure-free years → EEG-guided taper. JME-overlap: lifelong therapy "
            "usually required. Annual SUDEP counselling for any uncontrolled GTCS. Annual EEG reassess. "
            "GEFS+ FS-only: consider AED-free if FS remitted ≥2Y."
        ),
    },
    {
        "window": "Adult — Chronic GGE (>25Y)",
        "key_issues": (
            "Lifelong therapy for JME-overlap or refractory GGE. Annual review: seizure diary, AED "
            "tolerability, BMI (VPA weight), bone health (PB/PHT long-term — not primary HCN2 AEDs). "
            "SUDEP risk reassessment. Genetic counselling for offspring planning. Psychosocial support: "
            "employment accommodation, driving status, quality-of-life assessment."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "ETX therapeutic range (HCN2-CAE)", "value": "40–100 mg/L; target 80–100 mg/L before declaring ETX failure in HCN2-GGE (full range needed)"},
    {"name": "VPA therapeutic range", "value": "50–100 mg/L (TDM q3M on VPA)"},
    {"name": "HV-SWD abolition (ETX adequacy test)", "value": "3-min clinic HV should NOT induce absence at therapeutic ETX; SWD on HV → sub-therapeutic (check TDM)"},
    {"name": "LFT/FBC VPA toxicity threshold", "value": "ALT/AST >3× ULN → reduce VPA; >10× ULN → STOP immediately; ammonia >80 μmol/L → VPA toxicity workup"},
    {"name": "Seizure-free period for driving", "value": "≥12 months seizure-free (jurisdiction-dependent); document annually"},
    {"name": "SUDEP high-risk threshold", "value": "Uncontrolled GTCS ≥3/year + nocturnal + DRE + non-adherence = HIGH SUDEP RISK"},
    {"name": "Catamenial CLB threshold", "value": "≥2× seizure rate days 20–3 of cycle → CLB 5-10 mg/day days 20-28"},
    {"name": "VPPP VPA documentation", "value": "Annual VPPP form: specialist + GP; 2 contraceptive methods confirmed; females 9-55Y"},
    {"name": "POLG1 pre-VPA screen", "value": "Mandatory: refractory early epilepsy, cognitive regression, liver disease, family mitochondrial history"},
    {"name": "KD ketosis target", "value": "Serum BHB 2–4 mmol/L; urine ketones 4+; BHB daily monitoring first month"},
    {"name": "ZNS HCO3 monitoring", "value": "Serum bicarbonate q6M on ZNS; threshold: <18 mmol/L → dose reduce or switch"},
    {"name": "LTG plasma level (monitoring if used cautiously)", "value": "3-15 mg/L; EEG repeat 4-8 weeks post-LTG start in HCN2-GGE (absence check)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE 2022", "description": "Operational classification of seizure types and epilepsy syndromes (Epilepsia 63[6])"},
    {"name": "NICE NG217 (2022)", "description": "Epilepsies in children, young people, and adults — AED selection GGE, GEFS+"},
    {"name": "SANAD 2007 (Lancet)", "description": "ETX vs VPA vs LTG in absence — ETX and VPA superior to LTG; SANAD II: ETX superior for absence"},
    {"name": "SANAD II 2021 (NEJM)", "description": "ETX vs VPA vs LEV in GGE — ETX superior for absence; LEV worst for absence; VPA best for GTCS"},
    {"name": "Ludwig 2003 (Nat Neurosci)", "description": "HCN2 KO mice: spontaneous 3-4 Hz SWD absence seizures + sinus dysrhythmia — foundational HCN2 epilepsy genetics"},
    {"name": "Tang 2008 (Nat Genet)", "description": "HCN2 mutations (p.R591Q, p.V246L) in human febrile seizure GEFS+ families — clinical validation"},
    {"name": "Poolos 2002 (Nat Neurosci)", "description": "LTG blocks Ih in hippocampal neurons — basis for LTG Ih-block contraindication in HCN2-LOF"},
    {"name": "DiFrancesco 2010 (Physiol Rev)", "description": "HCN/If channel biology: comprehensive biophysics review; HCN2 cAMP sensitivity, gating, temperature"},
    {"name": "CPIC POLG Guidelines 2023", "description": "Clinical Pharmacogenomics Implementation Consortium: POLG mutation → VPA absolute CI"},
    {"name": "MHRA VPPP 2021", "description": "UK MHRA valproate pregnancy prevention programme (VPPP) — mandatory annual documentation"},
    {"name": "ACMG-AMP 2015", "description": "Variant classification framework (Pathogenic/LP/VUS/LB/Benign) — HCN2 LOF variant assessment"},
    {"name": "ILAE Dietary Therapies 2018", "description": "Expert consensus: KD, MAD, LGIT for drug-resistant epilepsy (Epilepsia 59[8]:1646-1659)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES  (6)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    "Ludwig A et al. Nat Neurosci 2003;6(1):75-80 — HCN2 KO mice: spontaneous absence epilepsy + sinus dysrhythmia",
    "Tang B et al. Nat Genet 2008;40(9):1122-1127 — HCN2 mutations in human febrile seizures and GEFS+",
    "Poolos NP et al. Nat Neurosci 2002;5(8):767-774 — LTG blocks Ih in hippocampal neurons (basis for HCN2-LOF LTG CI)",
    "DiFrancesco D. Physiol Rev 2010;90(3):899-969 — HCN/If channel comprehensive biology; cAMP modulation",
    "ILAE 2022 Operational classification of seizure types and epilepsy syndromes. Epilepsia 63(6)",
    "NICE NG217 (2022) — Epilepsies: diagnosis and management guidelines (AED selection GGE/GEFS+)",
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS  (15 key concepts)
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "HCN2 (19p13.3)",
        "definition": "Hyperpolarization-Activated Cyclic Nucleotide-Gated Channel 2; encodes HCN2 channel subunit (863 aa); 19p13.3; generates Ih 'funny' pacemaker current; TC (thalamic relay) neuron dominant; LOF → reduced Ih → enhanced thalamo-cortical oscillation → GEFS+/CAE/GGE.",
    },
    {
        "term": "Ih (Funny / Pacemaker) Current",
        "definition": "Non-selective cation current (Na+/K+) activated by HYPERPOLARIZATION (V1/2 ≈ −90 mV HCN2); reversal potential ≈ −30 mV → depolarising inward current at rest. Ih provides depolarising 'sag' after IPSP → LIMITS depth of TC hyperpolarisation → BRAKES TC oscillatory amplitude. HCN2 LOF removes this brake → runaway 3-Hz SWD.",
    },
    {
        "term": "HCN Subfamily (HCN1-4)",
        "definition": "4 HCN subunit genes: HCN1 (5p12, fastest, hippocampus/L5 cortex, DEE24 GOF+LOF) · HCN2 (19p13.3, medium, TC-dominant, GEFS+/CAE LOF) · HCN3 (1q22, slow, olfactory, rare epilepsy) · HCN4 (15q24, slowest, SA node dominant, sick sinus). HCN2 is the TC-dominant subunit; its LOF causes the thalamo-cortical oscillatory phenotype.",
    },
    {
        "term": "TC-Dominant Expression (HCN2)",
        "definition": "HCN2 is the PRIMARY Ih-generating subunit in thalamic relay (TC) neurons. HCN2 homotetramers and HCN1/HCN2 heterotetramers provide most of the Ih in TC cells. HCN2 LOF → TC Ih reduced 50% → slower sag → prolonged TC hyperpolarisation → greater Cav3.1/Cav3.2 de-inactivation → larger LTCS → stronger 3-Hz SWD. This TC dominance distinguishes HCN2 from HCN1 (hippocampus/cortex dominant).",
    },
    {
        "term": "cAMP Modulation of HCN2",
        "definition": "HCN2 CNBD has highest cAMP affinity in HCN family (Kd ~0.1 μM). cAMP binding shifts HCN2 V1/2 activation +15 to +25 mV → Ih activates at less negative potentials → increased Ih. Physiological: sympathetic stimulation → cAMP surge → increased HCN2 Ih → faster TC recovery. In LOF HCN2: cAMP modulation provides PARTIAL rescue (fewer channel molecules even if gating improved). Clinical: emotional stress may paradoxically modulate seizure risk via cAMP pathway.",
    },
    {
        "term": "Febrile Seizures (FS)",
        "definition": "Seizures occurring with fever (≥38°C); peak incidence 18M–3Y; 3-5% of children. SIMPLE FS: generalised, <15 min, once per fever. COMPLEX FS: focal, >15 min, or multiple in 24h. HCN2 LOF → lower FS threshold (fever alters HCN2 gating kinetics + remaining Ih reduced). FS per se do not require daily AED; complex FS or FS+ → GEFS+ workup.",
    },
    {
        "term": "GEFS+ (Genetic Epilepsy with Febrile Seizures Plus)",
        "definition": "Familial epilepsy syndrome: FS + FS+ (FS beyond age 6) ± afebrile seizures (absence, GTCS, myoclonic) in multiple family members. Variable expressivity within pedigrees. Most common gene: SCN1A (Dravet in severe end). HCN2: accounts for ~3-5% of GEFS+ families. HCN2 GEFS+ phenotype: MILDER than SCN1A GEFS+ (no DEE, no Dravet).",
    },
    {
        "term": "FS+ (Febrile Seizures Plus)",
        "definition": "FS continuing beyond age 6 years (when FS would typically remit). Defining phenotype of GEFS+ spectrum. FS+ patients may also have afebrile seizures (absence, GTCS) — but FS+ designation requires FS persisting >6Y. PROGNOSIS: GOOD — majority remit by 14Y. Daily AED generally NOT required for pure FS+; rescue medication (midazolam) is standard.",
    },
    {
        "term": "LTCS (Low-Threshold Ca²⁺ Spike)",
        "definition": "Rebound depolarisation in TC neurons: de-inactivated Cav3.1/Cav3.2 T-type channels generate Ca²⁺ spike → burst of Na⁺ APs → TC burst. HCN2 LOF: TC stays hyperpolarised longer after TRN GABA-B IPSP → deeper Cav3.1/Cav3.2 de-inactivation → LARGER LTCS → stronger 3-Hz oscillatory drive. ETX blocks the LTCS by reducing Cav3.1/Cav3.2 current — addresses the DOWNSTREAM consequence of HCN2 LOF.",
    },
    {
        "term": "Thalamo-Cortical 3-Hz SWD",
        "definition": "3-Hz spike-wave discharge: hallmark of generalised absence epilepsy; generated by TC burst-pause oscillation coupled to cortex via layer VI projections. HCN2 LOF amplifies the TC burst amplitude → stronger SWD. Clinically: typical absence seizures on EEG. Abolished by ETX (T-type Ca²⁺ block) and VPA (broad-spectrum). WORSENED by CBZ/OXC/PHT (NaV block → TC disinhibition) and LTG (Ih block → further LOF).",
    },
    {
        "term": "LTG Ih Block (HCN2-LOF Safety Rule)",
        "definition": "Lamotrigine (LTG) directly blocks HCN1/HCN2 Ih at therapeutic concentrations (Poolos 2002; Bois 1996). In HCN2-LOF GGE: Ih already reduced 50% → LTG further suppresses residual Ih → TC stays hyperpolarised longer → greater T-type Ca²⁺ de-inactivation → WORSENED 3-Hz SWD → INCREASED absence burden. Clinical rule: LTG CONTRAINDICATED in confirmed LOF HCN2 + absence phenotype. If LTG unavoidable (GTCS-only, no absence) → EEG monitoring mandatory after initiation.",
    },
    {
        "term": "ETX — Indirect Mechanism in HCN2-GGE",
        "definition": "ETX Level A for CAE: ETX blocks T-type Cav3.1/Cav3.2 in TC neurons (Coulter 1989) → reduces LTCS amplitude → suppresses 3-Hz SWD. In HCN2-LOF GGE: ETX addresses the DOWNSTREAM CONSEQUENCE (amplified LTCS) without directly restoring Ih. ETX does NOT fix HCN2 LOF. But by reducing T-type Ca²⁺ rebound, ETX compensates for the missing Ih brake → effective suppression of absence. COMPLEMENTARY mechanism to HCN2 LOF physiology.",
    },
    {
        "term": "VPPP (MHRA Valproate Pregnancy Prevention Programme)",
        "definition": "Mandatory annual documentation for all females 9-55Y on valproate (UK MHRA 2021). Requirements: specialist + GP co-signature; 2 contraceptive methods confirmed; pregnancy test; patient information card. VPA teratogenicity: NTD 1-2%; fetal valproate syndrome; neurodevelopmental risk. In HCN2-GGE females: prefer ETX (CAE-only, low teratogenicity) or LEV (GTCS/JME, safe in pregnancy) to MINIMISE VPA VPPP burden.",
    },
    {
        "term": "POLG1-Alpers Syndrome",
        "definition": "POLG1 (mitochondrial DNA polymerase γ) pathogenic variant + VPA → fulminant hepatic failure (Alpers-Huttenlocher syndrome). POLG1 variants (p.A467T, p.W748S most common) → mtDNA depletion. VPA inhibits mitochondrial β-oxidation → catastrophic energy failure in POLG1 → irreversible hepatic failure. Screening mandatory before VPA in any epilepsy patient.",
    },
    {
        "term": "ACMG-AMP 2015 Variant Classification",
        "definition": "5-tier classification: Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign. HCN2 LOF assessment: PS2 (de novo) + PVS1 (frameshift/nonsense LOF) + PM1 (CNBD hotspot) + PP3 (in silico) + functional assay (reduced Ih in heterologous expression, Tang 2008). GOF (rare): elevated Ih at less negative potentials → confirmed by patch-clamp Xenopus or HEK293.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT  (40 patients)
# ─────────────────────────────────────────────────────────────────────────────
_SYNDROMES = [
    "Febrile seizures plus (GEFS+)",
    "Childhood absence epilepsy (CAE)",
    "Juvenile absence epilepsy (JAE)",
    "Genetic epilepsy with febrile seizures plus spectrum (GEFS+ spectrum)",
    "Generalised epilepsy with GTCS alone",
]
_ETIOLOGY_TYPES = [e["category"] for e in ETIOLOGY_CATALOG]
_ETIOLOGY_WEIGHTS = [e["pct"] for e in ETIOLOGY_CATALOG]
_SEIZURE_TYPES_LIST = [s["type"] for s in SEIZURE_TYPES]
_TREATMENT_DRUGS = [t["drug"].split(" (")[0] for t in TREATMENTS]
_GENDERS = ["Male", "Female", "Non-binary"]
_ONSET_AGES = list(range(1, 18))

random.seed(230)
_cohort = []
for i in range(40):
    etiology = random.choices(_ETIOLOGY_TYPES, weights=_ETIOLOGY_WEIGHTS, k=1)[0]
    gender = random.choices(_GENDERS, weights=[40, 55, 5], k=1)[0]
    onset_age = random.choice(_ONSET_AGES)
    seizure_free = random.random() < 0.60
    has_absence = random.random() < 0.55
    has_gtcs = random.random() < 0.45
    has_fs = random.random() < 0.92
    _cohort.append({
        "patient_id": f"EPAT{i+1:03d}",
        "etiology": etiology,
        "onset_age": onset_age,
        "current_age": onset_age + random.randint(2, 30),
        "gender": gender,
        "syndrome": random.choice(_SYNDROMES),
        "seizure_free": seizure_free,
        "drug_resistant": not seizure_free and random.random() < 0.18,
        "primary_treatment": random.choices(_TREATMENT_DRUGS, k=1)[0],
        "etx_on": has_absence and random.random() < 0.75,
        "vpa_on": random.random() < 0.35,
        "lev_on": random.random() < 0.30,
        "clb_on": random.random() < 0.25,
        "ltg_used": random.random() < 0.12,
        "ltg_worsened": random.random() < 0.40,
        "catamenial": gender == "Female" and random.random() < 0.22,
        "hv_swd_positive": has_absence and random.random() < 0.82,
        "febrile_seizures": has_fs,
        "gtcs_present": has_gtcs,
        "absence_present": has_absence,
        "polg_tested": random.random() < 0.78,
        "on_kd": random.random() < 0.12,
        "vpa_vppp": gender == "Female" and random.random() < 0.60,
    })


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(_cohort)
    seizure_free_n = sum(1 for p in _cohort if p["seizure_free"])
    drug_resistant_n = sum(1 for p in _cohort if p["drug_resistant"])
    etx_n = sum(1 for p in _cohort if p["etx_on"])
    fs_n = sum(1 for p in _cohort if p["febrile_seizures"])
    absence_n = sum(1 for p in _cohort if p["absence_present"])
    gtcs_n = sum(1 for p in _cohort if p["gtcs_present"])
    catamenial_n = sum(1 for p in _cohort if p["catamenial"])
    ltg_worsened_n = sum(1 for p in _cohort if p["ltg_used"] and p["ltg_worsened"])
    hv_swd_n = sum(1 for p in _cohort if p["hv_swd_positive"])

    etiology_dist = []
    for e in ETIOLOGY_CATALOG:
        count = sum(1 for p in _cohort if p["etiology"] == e["category"])
        etiology_dist.append({
            "category": e["category"],
            "count": count,
            "pct": round(count / n * 100, 1),
        })

    treatment_summary = [
        {"drug": t["drug"].split(" (")[0], "level": t["level"].split(" —")[0]}
        for t in TREATMENTS
    ]
    monitoring_summary = [
        {"item": m["item"], "frequency": m["frequency"]}
        for m in MONITORING[:6]
    ]
    lifecycle_summary = [
        {"window": lc["window"], "key": lc["key_issues"][:90] + "…"}
        for lc in LIFECYCLE
    ]
    seizure_summary = [{"type": s["type"], "pct": s["pct"]} for s in SEIZURE_TYPES]

    return {
        "kpis": {
            "n_patients": n,
            "seizure_free_pct": round(seizure_free_n / n * 100, 1),
            "drug_resistant_pct": round(drug_resistant_n / n * 100, 1),
            "on_etx_n": etx_n,
            "febrile_seizures_n": fs_n,
            "absence_n": absence_n,
            "gtcs_n": gtcs_n,
            "catamenial_n": catamenial_n,
            "ltg_worsened_n": ltg_worsened_n,
            "hv_swd_n": hv_swd_n,
            "avg_age_years": round(sum(p["current_age"] for p in _cohort) / n, 1),
        },
        "etiology_distribution": etiology_dist,
        "seizure_summary": seizure_summary,
        "treatments_summary": treatment_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": lifecycle_summary,
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [
            {
                "drug": ci["drug"].split(" (")[0].split(" /")[0].split(" —")[0],
                "risk": ci["risk"],
            }
            for ci in CONTRAINDICATIONS[:5]
        ],
    }


def get_breakdown():
    return {
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "patients": _cohort,
    }


def get_definitions():
    return {
        "gene_summary": {
            "gene": "HCN2",
            "full_name": "Hyperpolarization-Activated Cyclic Nucleotide-Gated Channel 2",
            "chromosome": "19p13.3",
            "protein": "HCN2 channel subunit — 863 aa; mixed Na+/K+ non-selective cation channel",
            "channel_type": "HCN (funny/pacemaker) current — Ih; activated by HYPERPOLARIZATION",
            "activation_threshold": "V1/2 ≈ −90 mV at rest; cAMP shifts +15 to +25 mV (to −65 to −75 mV)",
            "inactivation_kinetics": "No classical inactivation (HCN channels do not inactivate); medium gating speed (HCN1 fastest, HCN4 slowest)",
            "primary_location": "TC (thalamic relay) neurons DOMINANT; also brainstem, hippocampus (moderate), cortex",
            "tc_vs_hippocampus": "TC-DOMINANT (HCN2) vs HCN1 hippocampus/L5-cortex dominant — KEY distinction from HCN1",
            "inheritance": "AD LOF (primary); reduced penetrance (~65-70%); familial GEFS+ pedigrees",
            "omim": "602781 (gene); GEFS+6 (609200); no dedicated DEE OMIM number — GGE spectrum, NOT DEE",
            "severity": "MILDER than HCN1-DEE24 — GGE phenotype; no obligate cognitive impairment",
            "lof_mechanism": "LOF → reduced TC Ih → prolonged TC hyperpolarisation → larger Cav3.1/Cav3.2 LTCS → 3-Hz SWD → absence/GEFS+",
            "ltg_contraindication": "LTG BLOCKS Ih (Poolos 2002) → WORSENS LOF HCN2 — HIGH RISK in confirmed LOF + absence",
            "absolute_ci": "CBZ / OXC / PHT (GGE aggravation); TGB (NCSE); VPA+POLG1 (Alpers); LTG HIGH RISK in LOF",
        },
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
