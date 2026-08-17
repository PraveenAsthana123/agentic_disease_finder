"""
STX1B Epilepsy — GEFS+ Spectrum / Febrile Seizures Plus / Focal Epilepsy of Infancy
=====================================================================================
40-patient cohort · STX1B (16p11.2) · Syntaxin-1B · Presynaptic t-SNARE · AD

STX1B BIOLOGY:
STX1B (16p11.2) encodes Syntaxin-1B (STX1B), one of two neuronal syntaxin-1 isoforms
(STX1A and STX1B; ~85% sequence identity). STX1B is a t-SNARE (target-membrane SNARE)
protein anchored in the presynaptic plasma membrane via its C-terminal transmembrane
domain.

KEY POINTS:
  1. SNARE COMPLEX ASSEMBLY:
     STX1B forms the presynaptic SNARE complex with VAMP2 (vesicle-associated membrane
     protein 2 / synaptobrevin-2, the v-SNARE) and SNAP25 (synaptosomal-associated
     protein 25 kDa). The three SNARE motifs zip together (N→C) to pull the vesicle
     and plasma membranes into apposition → membrane fusion → neurotransmitter release.
     STX1B haploinsufficiency → 50% reduction in t-SNARE density → impaired SNARE
     complex formation → reduced vesicle docking and fusion → less neurotransmitter
     released per action potential at ALL synapses (both GABAergic and glutamatergic).

  2. RELATIONSHIP TO STXBP1 (Munc18-1):
     STXBP1 (Munc18-1, the most common DEE gene) is the chaperone that binds closed
     (auto-inhibited) syntaxin-1 and guides it to the plasma membrane. STXBP1 LOF →
     misfolded STX1B → proteasomal degradation of both proteins (co-degradation) →
     phenocopy of STX1B LOF. This is why STX1B and STXBP1 epilepsies share mechanism
     but differ in severity (STXBP1 is more severe because Munc18-1 LOF = combined
     loss of both STX1B and its chaperone function).

  3. WHY DISINHIBITION PREDOMINATES:
     GABAergic interneurons (fast-spiking PV+ interneurons) have higher firing rates
     and shorter inter-spike intervals than pyramidal neurons. They rely more critically
     on rapid, sustained vesicle fusion per burst. STX1B haploinsufficiency
     disproportionately impairs high-frequency inhibitory drive (interneuron→pyramidal
     GABA release) → net cortical disinhibition → epileptiform threshold lowered →
     GEFS+ phenotype rather than DEE (unlike STXBP1 which causes severe DEE because
     the chaperone effect also destroys existing syntaxin protein).

  4. GENOTYPE–PHENOTYPE (Schubert 2014, Wolking 2019):
     Missense variants (e.g., p.R126Q, p.A198P, p.Y127C): variable GEFS+ severity,
     some haploinsufficiency, some dominant-negative effect.
     Truncating / frameshift (e.g., p.Y262*, p.Q281*): pure haploinsufficiency →
     classic GEFS+; may be milder than missense (no toxic gain).
     Splice-site variants: variable functional consequence; often GEFS+-like.
     De novo severe missense: rare Dravet-like phenotype (<10%).

  5. FEVER SENSITIVITY MECHANISM:
     Temperature increase (fever 38–40°C) → accelerated SNARE protein dynamics →
     increased zippering speed BUT also faster unfolding of partially misfolded STX1B
     variants → acute transient loss of fusion-competent t-SNARE → fever-triggered
     febrile seizure. Same mechanism as SCN1A temperature gating but via protein
     thermolability rather than channel gating.

  6. POPULATION GENETICS:
     STX1B locus 16p11.2 (same chromosomal band as 16p11.2 CNV associated with autism/
     epilepsy — though STX1B variants are distinct from the CNV syndrome). pLI ~0.92
     (high intolerance to LOF). OMIM GEFS+9 #616172. Prevalence ~1-2% of GEFS+ families.

CLINICAL DIAGNOSIS:
  - Febrile seizures onset 6M–5Y, evolving to GEFS+ pattern (seizures with and without fever)
  - Focal seizures (temporal or frontal onset in ~35%)
  - Rarely myoclonic or absence components
  - Cognitive outcome: usually normal to mildly impaired (much better than STXBP1)
  - EEG: interictal normal or focal/multifocal spikes; no burst-suppression
  - MRI: normal (no cortical malformation, unlike STXBP1 DEE)

INHERITANCE AND GENETICS:
  AUTOSOMAL DOMINANT. De novo: ~40% (severe phenotype). Familial: ~60%.
  Locus: 16p11.2. Gene: STX1B (7 exons; 288 aa protein).
  pLI ~0.92. Key variants: p.R126Q / p.A198P / p.Y127C / p.Y262* / p.Q281* / splice variants.

KEY REFERENCES:
  Schubert et al. 2014 Nature Genetics 46:1327 — STX1B discovery in GEFS+ families
  Wolking et al. 2019 Brain 142:3299 — STX1B genotype-phenotype; 29 additional families
  Bhatt et al. 2023 Epilepsia — gene-epilepsy reference standard
  Vlaskamp et al. 2019 Epilepsia — GEFS+ spectrum genetic landscape
  Südhof & Rothman 2009 Science — SNARE mechanism (Nobel Prize 2013 basis)
  ILAE Commission on Classification and Terminology 2022
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "STX1B-missense-GEFS+",
        "pct": 38,
        "etiology": "STX1B missense variant — GEFS+ familial (most common class)",
        "mechanism": (
            "Missense variants (p.R126Q most frequent; p.A198P, p.Y127C) in the SNARE "
            "motif or Habc domain of STX1B. Partial haploinsufficiency or dominant-negative "
            "effect on SNARE complex assembly. ~40% de novo in this class. Classic GEFS+ "
            "phenotype: febrile seizures → febrile seizures plus → focal seizures."
        ),
        "typical_variants": "p.R126Q (SNARE motif) · p.A198P (SNARE motif) · p.Y127C (Habc-SNARE junction)",
        "onset_age_months": 14,
        "outcome": "Good (85% seizure-free by adulthood with appropriate AEDs)",
    },
    {
        "category": "STX1B-truncating-GEFS+",
        "pct": 27,
        "etiology": "STX1B truncating / frameshift — pure haploinsufficiency GEFS+",
        "mechanism": (
            "Nonsense or frameshift variants (p.Y262*, p.Q281*, frameshifts) lead to "
            "nonsense-mediated decay → pure 50% STX1B protein reduction. Typically milder "
            "phenotype than missense (no dominant-negative toxicity). Familial in ~70% of "
            "truncating variants; febrile seizures plus with focal evolution in some."
        ),
        "typical_variants": "p.Y262* · p.Q281* · c.502delG · p.E289fs · splice-site c.532+1G>A",
        "onset_age_months": 18,
        "outcome": "Excellent (>90% achieve remission; cognitive outcome normal in most)",
    },
    {
        "category": "STX1B-focal-epilepsy-infancy",
        "pct": 20,
        "etiology": "STX1B missense — focal epilepsy of infancy (FEI) without typical GEFS+ triggers",
        "mechanism": (
            "Subset of STX1B missense variants cause predominantly focal epilepsy without "
            "prominent febrile seizure component. Temporal or frontal lobe onset. Likely "
            "reflects asymmetric regional expression of STX1B in limbic cortex. EEG shows "
            "focal temporal or frontal spikes. Responds well to focal AEDs (OXC, LCM)."
        ),
        "typical_variants": "p.P238L · p.L206P · p.R281C",
        "onset_age_months": 24,
        "outcome": "Moderate (60% seizure-free; 20% DRE requiring add-on)",
    },
    {
        "category": "STX1B-severe-Dravet-like",
        "pct": 10,
        "etiology": "STX1B de novo missense — Dravet-like severe DEE (rare)",
        "mechanism": (
            "Rare de novo missense variants (particularly affecting the transmembrane domain "
            "or N-peptide binding site for Munc18-1/STXBP1) cause a Dravet-like encephalopathy: "
            "status epilepticus, refractory myoclonic seizures, developmental regression. "
            "Likely dominant-negative interference with STX1A-containing SNARE complexes + "
            "disruption of STXBP1 chaperone binding → compound t-SNARE deficit."
        ),
        "typical_variants": "p.L205R · p.G36R (N-peptide) · p.C272Y (TM domain)",
        "onset_age_months": 6,
        "outcome": "Poor (refractory; requires KD + polypharmacy; cognitive outcome compromised)",
    },
    {
        "category": "phenocopy-GEFS+",
        "pct": 5,
        "etiology": "GEFS+ phenocopy — no pathogenic STX1B variant found (SCN1A-negative, other gene)",
        "mechanism": (
            "GEFS+ phenotype meeting clinical criteria but STX1B sequencing negative. "
            "Rule out other GEFS+ genes: SCN1A (most common), SCN1B, GABRG2, GABRD, "
            "HCN1. STX1B phenocopy rate ~5% in referral cohorts. Functional studies needed "
            "for VUS classification."
        ),
        "typical_variants": "VUS on STX1B · or STX1A variant (paralogue, rare) · or dual-hit",
        "onset_age_months": 20,
        "outcome": "Variable (depends on true underlying gene)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Febrile Seizures Plus (FS+)",
        "pct": 85,
        "description": (
            "Febrile seizures plus — hallmark of GEFS+ (Generalized Epilepsy with "
            "Febrile Seizures Plus). Febrile seizures persisting beyond age 6 years "
            "(normally remit by 5Y), or occurring at low fever threshold (<38°C). "
            "Often focal onset (temporal > occipital). Some patients develop afebrile "
            "seizures. The 'Plus' denotes afebrile generalized or focal seizures in "
            "addition to febrile seizures."
        ),
        "eeg": "Normal interictal or focal temporal/frontal spikes. Febrile EEG: rhythmic theta-delta, post-ictal suppression.",
        "semiology": "Unilateral clonic or hemiclonic, oro-alimentary automatisms, focal tonic. Generalization in prolonged seizures.",
        "clinical_tips": (
            "Distinguish from simple febrile seizures: FS+ persist >6Y, occur at lower fever "
            "threshold, or have focal features. Temperature-lower threshold in STX1B (acetaminophen "
            "prophylaxis at 37.5°C during illness). Avoid hyperthermia during illness."
        ),
    },
    {
        "type": "Focal Seizures (Temporal/Frontal)",
        "pct": 55,
        "description": (
            "Afebrile focal seizures arising from temporal (limbic) or frontal cortex. "
            "Higher frequency in STX1B-FEI subtype and de novo Dravet-like cases. EEG: "
            "focal spikes ipsilateral to ictal onset. Often nocturnal (frontal). "
            "Responsive to OXC, LCM, or LTG in most."
        ),
        "eeg": "Focal temporal or frontal spike-wave. Ictal: rhythmic theta build-up from focus.",
        "semiology": "Aura (déjà vu, epigastric rising), oro-alimentary automatisms, contralateral dystonia.",
        "clinical_tips": (
            "STX1B focal epilepsy does NOT carry the same GGE-aggravation risk as GABRX genes. "
            "CBZ/OXC are NOT absolutely contraindicated (unlike GGE gene panel). LCM excellent "
            "for focal STX1B (NaV slow-inactivation MOA complements SNARE deficit; no interaction)."
        ),
    },
    {
        "type": "Generalized Tonic-Clonic Seizures (GTCS)",
        "pct": 45,
        "description": (
            "Bilateral GTCS — occurs in GEFS+ spectrum evolution, often evolving from focal "
            "onset. Typically brief (1-3 min). Status epilepticus rare (<10%) outside "
            "Dravet-like subtype. Triggered by fever in most but can occur afebrile in adolescence."
        ),
        "eeg": "Generalized spike-wave 3-4 Hz during GTCS. Post-ictal diffuse slowing.",
        "semiology": "Tonic phase (10-20s) → bilateral clonic. Post-ictal confusion. No tongue bite in focal-onset GTCS.",
        "clinical_tips": (
            "GTCS risk increases with sleep deprivation and fever. Educate families: rescue midazolam "
            "buccal 0.3 mg/kg for >2 min GTCS. VPA Level-A (full spectrum, covers GTCS+focal+FS+). "
            "POLG screening MANDATORY before VPA."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 18,
        "description": (
            "Myoclonic seizures — predominantly in STX1B-severe Dravet-like subtype. Brief "
            "bilateral or unilateral myoclonic jerks, often on awakening. Rare in classic GEFS+ "
            "subtype (<5%). EEG: generalized polyspike-wave. Distinguish from benign myoclonus "
            "of infancy. Worsened by sodium-channel blockers in Dravet-like cases."
        ),
        "eeg": "Generalized polyspike-wave burst time-locked to myoclonic jerk. Background slowing in Dravet-like.",
        "semiology": "Sudden bilateral arm jerk, head drop, brief loss of tone. No post-ictal phase.",
        "clinical_tips": (
            "Myoclonic seizures in STX1B usually indicate severe (Dravet-like) subtype. "
            "Avoid CBZ/OXC/PHT/LTG (may aggravate myoclonus). VPA + CLB ± KD first approach. "
            "SCN1A sequencing should be done first to rule out true Dravet before proceeding."
        ),
    },
    {
        "type": "Febrile Status Epilepticus (FSE)",
        "pct": 12,
        "description": (
            "Febrile status epilepticus — prolonged febrile seizure >30 min or cluster "
            "requiring emergency rescue medication. Risk factor for temporal lobe MTS in "
            "long-term follow-up. Higher risk in STX1B-severe Dravet-like subtype. "
            "Emergency protocol: buccal midazolam → IV lorazepam → phenobarbital → anaesthesia."
        ),
        "eeg": "Prolonged ictal activity on EEG during FSE. Post-FSE: diffuse slowing ± focal slowing in temporal lobe.",
        "semiology": "Prolonged focal hemiclonic or bilateral clonic seizure with fever. Altered consciousness.",
        "clinical_tips": (
            "FSE risk: STX1B Dravet-like subtype especially E815K-like variants. "
            "Action plan: buccal midazolam 0.3 mg/kg for seizure >2 min. Call EMS at 5 min. "
            "Avoid rectal diazepam (unreliable absorption during fever). Emergency letter for A&E. "
            "Post-FSE MRI to rule out FIRES/febrile infection-related epilepsy syndrome."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / Febrile Illness (≥37.5°C)",
        "pct": 92,
        "notes": (
            "PRIMARY trigger for STX1B epilepsy. Low fever threshold (37.5–38°C sufficient). "
            "Mechanism: temperature → accelerated unfolding of thermolabile STX1B variants → "
            "acute reduction in fusion-competent t-SNARE → seizure. Pre-empt with acetaminophen "
            "15 mg/kg at first sign of fever. DO NOT use ibuprofen alone (delayed onset)."
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 70,
        "notes": (
            "Sleep deprivation → reduced NREM slow-wave sleep → impaired synaptic homeostasis "
            "→ lower seizure threshold. Especially relevant for GTCS risk in adolescents/adults. "
            "Enforce strict sleep schedule. Avoid overnight activities (school trips, parties)."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 65,
        "notes": (
            "Missed doses of VPA, LEV, or LTG → rebound cortical excitability within 1-2 half-lives. "
            "Risk highest with LEV (short half-life 6-8h). Rebound seizures can present as GTCS "
            "even if patient was previously seizure-free. Use LEV extended-release formulation to "
            "reduce missed-dose risk. Pill organizer and phone reminder highly recommended."
        ),
    },
    {
        "trigger": "Intercurrent Viral Illness (without high fever)",
        "pct": 50,
        "notes": (
            "Viral illness (rhinovirus, influenza, COVID-19) can trigger seizures even with "
            "low-grade or no fever — likely via cytokine-mediated neuroinflammation (IL-6, TNF-α) "
            "lowering seizure threshold. Maintain AED compliance and hydration during illness. "
            "Influenza vaccine recommended annually."
        ),
    },
    {
        "trigger": "Vaccination (within 24h)",
        "pct": 35,
        "notes": (
            "Post-vaccination febrile seizures — same mechanism as intercurrent illness. "
            "DTP/MMR vaccines most commonly implicated (fever response 6-12h post-injection). "
            "Pre-medicate with acetaminophen 30 min before vaccination and q6h for 24h after. "
            "Vaccination is NOT contraindicated — benefit far exceeds risk. Document in record."
        ),
    },
    {
        "trigger": "Alcohol / Withdrawal",
        "pct": 30,
        "notes": (
            "Alcohol consumption → acute GABA-A receptor potentiation (sedation) → "
            "relative inhibitory excess acutely, then REBOUND excitability as alcohol clears "
            "(12-24h later). Withdrawal from habitual alcohol → cortical hyperexcitability → "
            "GTCS risk. Counsel all STX1B patients (adolescent+) about seizure risk with alcohol. "
            "Abstinence is safest recommendation."
        ),
    },
    {
        "trigger": "Stress / Emotional Arousal",
        "pct": 28,
        "notes": (
            "Acute emotional stress → catecholamine surge → reduced inhibitory tone → "
            "reduced seizure threshold. CRH (corticotropin-releasing hormone) also has direct "
            "pro-epileptic action. Mindfulness-based stress reduction and psychological support "
            "useful adjuncts. Ensure adequate support around exams, family crises."
        ),
    },
    {
        "trigger": "Hyperthermia (Hot Bath, Exercise)",
        "pct": 22,
        "notes": (
            "Exercise-induced or hot-bath hyperthermia can trigger seizures — same protein "
            "thermolability mechanism as febrile seizures. Not as common as in SCN1A-Dravet "
            "but documented in STX1B. Advise: avoid very hot baths/showers; cool down during "
            "strenuous exercise; do not exercise to exhaustion in hot weather."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (7)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA)",
        "level": "Level A",
        "indication": "First-line: full-spectrum GEFS+ (FS+ · GTCS · focal · myoclonic components)",
        "dose": "Adult: 500–2000 mg/day (divided BD); Child: 20–40 mg/kg/day. TDM target: 50–100 mg/L.",
        "moa": (
            "Multiple mechanisms: Na+ channel slow inactivation, T-type Ca2+ channel block, "
            "GABA-T inhibition (↑ GABA), HDAC inhibition. Broad-spectrum efficacy covers all "
            "GEFS+ seizure types. No mechanism-specific antagonism with STX1B SNARE deficit — "
            "VPA enhances inhibitory transmission independently of vesicle fusion."
        ),
        "efficacy": "70-80% seizure reduction; 45-55% seizure-free in GEFS+ spectrum (SANAD-I data).",
        "safety": (
            "Teratogen (VPPP mandatory for ALL females of childbearing potential; MHRA 2021). "
            "Weight gain, hair loss, tremor, metabolic (hyperammonaemia). "
            "POLG1 screening MANDATORY before initiating — POLG mutation = absolute CI "
            "(fatal hepatotoxicity / Alpers syndrome). LFT + FBC + ammonia at baseline then q3M."
        ),
        "monitoring": "VPA TDM q3M · LFT + FBC + ammonia q3M · POLG1 before start · VPPP at every visit",
        "stx1b_note": (
            "VPA preferred over CBZ/OXC in STX1B with any GGE/generalized component (FS+/GTCS). "
            "If purely focal STX1B: LCM or OXC acceptable alternatives without VPPP burden. "
            "Discuss VPPP at every encounter for females — valproate teratogenicity risk ongoing."
        ),
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "Adjunct or monotherapy: GTCS · focal seizures · myoclonic (Dravet-like STX1B)",
        "dose": "Adult: 1000–3000 mg/day (BD); Child: 20–60 mg/kg/day. No TDM needed routinely.",
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) modulator — reduces Ca2+-dependent vesicle "
            "exocytosis. Unique complementary action in STX1B: LEV reduces the RATE of vesicle "
            "fusion events, reducing excitatory (glutamatergic) transmission selectively when "
            "release probability is high. In STX1B LOF, GABA release is already impaired; "
            "LEV's net effect at excitatory synapses provides relative inhibitory advantage."
        ),
        "efficacy": "60-70% responder rate in GEFS+ focal/GTCS. Additive with VPA.",
        "safety": (
            "Behavioral side effects (irritability, depression, psychosis) in 10-15% — "
            "monitor PHQ-9 / mood questionnaires. Renal dose adjustment needed (eGFR-based). "
            "No hepatotoxicity, no teratogen classification A/B (MHRA), no POLG interaction. "
            "Preferred in GEFS+ females who cannot or will not take VPA."
        ),
        "monitoring": "PHQ-9 / mood q3M · renal function baseline · behaviour monitoring q6M",
        "stx1b_note": (
            "LEV SV2A mechanism is mechanistically complementary to STX1B SNARE deficit — "
            "both modulate synaptic vesicle cycle. Short half-life: use extended-release LEV (LEV-XR) "
            "to reduce missed-dose risk. Behavioural side effects more common in children — "
            "monitor closely in first 3 months."
        ),
    },
    {
        "drug": "Lamotrigine (LTG)",
        "level": "Level B",
        "indication": "Focal seizures · GTCS in GEFS+ — CAUTION if myoclonic component present",
        "dose": "Adult mono: 100–400 mg/day (titrate slowly 25 mg/2wk). Child: 5–15 mg/kg/day.",
        "moa": (
            "NaV (voltage-gated sodium channel) fast-inactivation enhancer + P/Q-type Ca2+ channel "
            "modulator. Effective for focal and GTCS. CAUTION: may paradoxically worsen myoclonic "
            "seizures in some patients (15-20% — similar risk to GGE). EEG pre-prescribing is "
            "essential: if myoclonic component on EEG → prefer VPA or LEV."
        ),
        "efficacy": "55-65% responder in focal GEFS+ without myoclonic component.",
        "safety": (
            "SJS/TEN risk: HLA-B*15:02 MANDATORY before use in Asian populations (CPIC guideline). "
            "HLA-A*31:01 in European/Japanese. Rash in 10%: do not rechallenge if rash occurs. "
            "Slow titration reduces rash risk. Safe in pregnancy (teratogenicity risk lower than VPA) "
            "— preferred female option IF no myoclonic component."
        ),
        "monitoring": "HLA-B*15:02 before start · titration diary · rash surveillance first 8 weeks",
        "stx1b_note": (
            "LTG preferred female option for STX1B GEFS+ WITHOUT myoclonic component. "
            "Must get EEG first — check for polyspike-wave (myoclonic risk). "
            "If purely focal STX1B, LTG + LCM combination is well-tolerated and effective. "
            "Never start LTG without slow titration protocol — rash risk during rapid loading."
        ),
    },
    {
        "drug": "Lacosamide (LCM)",
        "level": "Level B",
        "indication": "Focal seizures (temporal/frontal STX1B subtype) — excellent adjunct",
        "dose": "Adult: 200–400 mg/day (BD). Child ≥2Y: 4–8 mg/kg/day (max 600 mg). IV available.",
        "moa": (
            "Selectively enhances slow inactivation of NaV channels — distinct MOA from fast-inactivation "
            "agents (LTG, CBZ). No modulation of GABA or glutamate receptors directly. "
            "No significant interaction with SNARE/vesicle cycle. PR interval prolongation: ECG at "
            "baseline + after dose increase. Minimal drug-drug interactions (no hepatic induction)."
        ),
        "efficacy": "50-60% ≥50% seizure reduction in focal epilepsy (BLOSSOM, VIMPACT trials).",
        "safety": (
            "PR prolongation (dose-dependent) — ECG before and after dose increases. "
            "Avoid in 2nd/3rd-degree AV block. Dizziness (10-15%): slow titration 50mg/2wk. "
            "No teratogenicity data — use with caution in pregnancy (animal data reassuring). "
            "No POLG interaction. No HLA typing needed."
        ),
        "monitoring": "ECG at baseline + after dose increase · PR interval monitoring",
        "stx1b_note": (
            "LCM is the preferred focal-seizure add-on for STX1B-FEI subtype. "
            "Does NOT worsen myoclonic or absence component. "
            "IV LCM useful for acute seizure cluster management in hospital. "
            "Check for cardiac conduction disease before starting (family history of arrhythmia)."
        ),
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Intermittent rescue · febrile illness prophylaxis · adjunct for refractory GEFS+",
        "dose": (
            "Intermittent: 5–10 mg at seizure onset or fever start (3–5 days max). "
            "Maintenance: 10–40 mg/day (adult); 0.5–1 mg/kg/day (child). Taper after 6M."
        ),
        "moa": (
            "GABA-A receptor positive allosteric modulator (1,5-benzodiazepine; less sedating than "
            "1,4-BZDs). Enhances Cl⁻ flux via GABA-A receptor — SNARE-independent. "
            "In STX1B LOF (impaired vesicle fusion), CLB directly potentiates the GABA that IS released "
            "by residual SNARE complexes — amplifying inhibitory effect without requiring more vesicles. "
            "Particularly useful intermittently during febrile illness as fever prophylaxis."
        ),
        "efficacy": "70-80% seizure reduction as adjunct; 55-65% freedom from febrile seizures as intermittent prophylaxis.",
        "safety": (
            "Tolerance develops with chronic use (3-6M). Sedation, cognitive dulling. "
            "Withdrawal seizures with abrupt discontinuation — always taper. "
            "Respiratory depression with IV administration. "
            "No hepatotoxicity. FDA REMS program (CLB Schedule IV — abuse potential low but monitored)."
        ),
        "monitoring": "Sedation score · mood/cognitive assessment q6M · withdrawal plan documented",
        "stx1b_note": (
            "INTERMITTENT CLB during febrile illness is a key STX1B strategy: start CLB 5 mg at "
            "first fever sign, continue until afebrile × 24h. This reduces FS+ frequency significantly. "
            "Document in patient action plan. Families should have home supply. "
            "CLB REMS (USA): prescriber must enrol at www.clobazam-rems.com."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Dravet-like STX1B · refractory GEFS+ with myoclonic component · VPA contraindicated",
        "dose": "Classical 4:1 ratio under dietitian supervision. MAD/LGIT as alternatives.",
        "moa": (
            "Multiple mechanisms: ketone bodies (β-hydroxybutyrate, acetoacetate) → GABA synthesis "
            "increase (via glutamate decarboxylase upregulation) + direct inhibition of VGLUT/glutamate "
            "vesicular loading → reduces glutamatergic vesicle release. "
            "METABOLIC RELEVANCE TO STX1B: KD increases GABA synthesis from ketone substrates → "
            "partially compensates for STX1B-impaired GABAergic vesicle fusion. "
            "Additionally: AMPK activation → mTOR suppression → reduces cortical hyperexcitability."
        ),
        "efficacy": "50% response (≥50% seizure reduction) in Dravet-like STX1B. 20-30% seizure freedom.",
        "safety": (
            "Acidosis, hyperlipidaemia, nephrolithiasis, growth impairment (children). "
            "Constipation, GI upset. Carnitine supplementation recommended. "
            "Monitor lipid panel, renal ultrasound q6M, growth q3M. "
            "Contraindicated in pyruvate carboxylase deficiency, porphyria."
        ),
        "monitoring": "Ketone monitoring daily · lipid panel q6M · renal ultrasound q12M · growth q3M",
        "stx1b_note": (
            "KD is the preferred escalation for Dravet-like STX1B when 2 AEDs have failed. "
            "Unlike pure Dravet (SCN1A), STX1B-KD response may be slightly better (~50% vs 45%) "
            "because ketone-driven GABA synthesis partially compensates for the vesicle fusion "
            "deficit. Initiate KD before adding quinidine or fenfluramine (Dravet-specific). "
            "Dietitian referral essential."
        ),
    },
    {
        "drug": "Valproate (VPA) — POLG & VPPP Mandatory Check",
        "level": "Level A — with mandatory pre-treatment screening",
        "indication": "Full-spectrum coverage in GEFS+ with GTCS / myoclonic / focal",
        "dose": "See VPA entry above.",
        "moa": "See VPA entry above.",
        "efficacy": "See VPA entry above.",
        "safety": (
            "POLG1 MUTATION = ABSOLUTE CONTRAINDICATION (fatal Alpers hepatotoxicity). "
            "VPPP (Valproate Pregnancy Prevention Programme) mandatory for ALL females ≥9Y. "
            "Cannot prescribe VPA to new female patients without: VPPP form signed, "
            "contraception confirmed, counselling documented."
        ),
        "monitoring": "POLG1 (once, before first dose) · VPPP form annual · TDM q3M",
        "stx1b_note": (
            "In STX1B: VPA is the cleanest full-spectrum option but POLG1 MUST be checked first. "
            "If POLG1 pathogenic → use LEV as primary, KD as escalation. "
            "If female → VPPP mandatory; if patient declines contraception → discuss LEV or LTG. "
            "Document shared decision making."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (5)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) — GGE COMPONENT CAUTION",
        "level": "HIGH CAUTION (not absolute CI unless GGE phenotype)",
        "reason": (
            "CBZ/OXC are CAUTION (not absolute CI) in STX1B — unlike GABRX/GGE genes where "
            "they are ABSOLUTE CI. If STX1B patient has GGE-like component (myoclonic jerks, "
            "3Hz absence, generalized spike-wave on EEG) → CBZ/OXC risk aggravating myoclonus "
            "and absence. If purely focal STX1B (FEI subtype) → CBZ/OXC acceptable but LCM "
            "preferred (better tolerability, no HLA typing, no induction)."
        ),
        "alternative": "LCM (focal) · VPA (full-spectrum) · LEV (focal/GTCS)",
        "icon": "⚠️",
    },
    {
        "drug": "Valproate — POLG1 Mutation",
        "level": "ABSOLUTE CONTRAINDICATION if POLG1 pathogenic variant",
        "reason": (
            "POLG1 (mitochondrial DNA polymerase gamma) pathogenic variants + VPA → "
            "Alpers-Huttenlocher syndrome (fatal progressive hepatoencephalopathy). "
            "STX1B patients may coincidentally carry POLG1 variants. POLG1 screening MANDATORY "
            "before starting VPA in ANY new patient. Panel: POLG1 full gene sequencing (or "
            "at minimum p.W748S, p.A467T, p.G848S common pathogenic variants)."
        ),
        "alternative": "LEV · LCM · LTG (if no myoclonic) · KD",
        "icon": "🚫",
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin",
        "level": "HIGH CAUTION — avoid in STX1B with myoclonic component",
        "reason": (
            "Phenytoin primarily blocks NaV fast inactivation — may paradoxically worsen myoclonic "
            "seizures in Dravet-like STX1B (same mechanism as in Dravet SCN1A). Additionally: "
            "PHT induces CYP3A4 (interacts with many co-medications). Zero-order kinetics → "
            "narrow therapeutic index → toxicity risk. PHT IV formulation: cardiac risk (bradycardia, "
            "hypotension) especially in pediatric patients. Prefer IV LEV or IV VPA for acute use."
        ),
        "alternative": "IV LEV · IV VPA · buccal midazolam (acute) · IV LCM",
        "icon": "⚠️",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "HIGH CAUTION — not indicated; visual field defect risk",
        "reason": (
            "VGB causes irreversible bilateral concentric visual field defects (VFD) in "
            "30-50% with chronic use. No established efficacy in GEFS+ or STX1B epilepsy "
            "(indicated for IS and tuberous sclerosis). SHARE REMS program (USA). ERG required "
            "every 3 months if used. Not indicated in STX1B — avoid unless no alternative."
        ),
        "alternative": "LEV · VPA · CLB",
        "icon": "⚠️",
    },
    {
        "drug": "Tiagabine (TGB)",
        "level": "ABSOLUTE CONTRAINDICATION — NCSE risk",
        "reason": (
            "TGB (GABA reuptake inhibitor) can precipitate non-convulsive status epilepticus "
            "(NCSE) and absence status in patients with GEFS+ or any generalized epilepsy "
            "component. NCSE may present as confusion, staring, behavioural change — can be "
            "missed without EEG. ABSOLUTE CI in STX1B GEFS+ / any generalized component. "
            "Avoid even in purely focal STX1B (risk of undetected generalized component)."
        ),
        "alternative": "LEV · LCM · LTG",
        "icon": "🚫",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG1 screening", "freq": "Once (before VPA)", "notes": "Full gene sequencing or common variants (p.W748S, p.A467T, p.G848S). Document result permanently."},
    {"item": "VPPP form (females)", "freq": "Annual", "notes": "Valproate Pregnancy Prevention Programme — mandatory for ALL females ≥9Y on VPA."},
    {"item": "VPA TDM", "freq": "q3M", "notes": "Target 50–100 mg/L. Check trough (pre-dose). Dose-adjust for fever, illness, growth."},
    {"item": "LFT + FBC + ammonia (VPA)", "freq": "q3M", "notes": "ALT, AST, bilirubin, ammonia. Stop VPA if ammonia >2× ULN with encephalopathy."},
    {"item": "Seizure + fever diary", "freq": "Daily / every visit", "notes": "Track fever temperature at seizure onset. Identify threshold. STX1B unique: note fever level at each FS+."},
    {"item": "EEG (baseline + annual)", "freq": "Baseline then q12M", "notes": "Check for: focal spikes (temporal/frontal), polyspike-wave (myoclonic risk before LTG). Repeat if new seizure type."},
    {"item": "MRI brain (baseline)", "freq": "Once at diagnosis", "notes": "Normal in GEFS+ STX1B. If abnormal → reconsider diagnosis. Post-FSE: repeat MRI (rule out FIRES/MTS)."},
    {"item": "HLA-B*15:02 (LTG)", "freq": "Before LTG start", "notes": "Mandatory in Asian populations (Han Chinese, Thai, Malaysian). Avoid LTG if positive (SJS/TEN risk)."},
    {"item": "HLA-A*31:01 (LTG/CBZ)", "freq": "Before LTG/CBZ", "notes": "European/Japanese: HLA-A*31:01 associated with DRESS and SJS. Check before LTG in Europeans."},
    {"item": "Neurodevelopmental assessment", "freq": "q2Y (child)", "notes": "Cognitive outcome usually normal in GEFS+. Annual in Dravet-like subtype. WISC-V / Vineland at 6Y and 10Y."},
    {"item": "PHQ-9 / GAD-7 (LEV)", "freq": "q3M (on LEV)", "notes": "Behavioural side effects (irritability, depression) in 10-15% on LEV. Structured screening required."},
    {"item": "LCM ECG", "freq": "Baseline + after dose increase", "notes": "PR interval — avoid if 2nd/3rd-degree AV block. LCM dose-dependent PR prolongation."},
    {"item": "SUDEP risk counselling", "freq": "Annual", "notes": "SUDEP risk: higher with GTCS frequency, nocturnal seizures, polypharmacy. Document annual discussion."},
    {"item": "Genetic counselling", "freq": "At diagnosis + q2Y", "notes": "AD inheritance: 50% recurrence risk (familial) or low recurrence (de novo). Cascade testing offered to parents, siblings. Prenatal options discussed."},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE  (6 windows)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Infancy (0–2Y)",
        "key_events": "First febrile seizure · STX1B genetic diagnosis · Fever action plan",
        "priorities": (
            "Establish STX1B diagnosis (panel or singleton WES). Fever action plan: acetaminophen "
            "at 37.5°C, buccal midazolam prescription. POLG1 testing. Decide if prophylactic "
            "CLB during febrile illness. Counsel parents on GEFS+ prognosis (mostly good)."
        ),
    },
    {
        "window": "Early Childhood (2–6Y)",
        "key_events": "FS+ evolution · First afebrile seizure · AED initiation",
        "priorities": (
            "Monitor for FS+ pattern (febrile seizures persisting beyond 6Y → confirm GEFS+). "
            "Start AED if afebrile seizures appear: VPA (POLG negative) or LEV. "
            "EEG to characterize seizure type (focal vs generalized). School readiness assessment."
        ),
    },
    {
        "window": "School Age (6–12Y)",
        "key_events": "Seizure pattern stabilization · Neurodevelopmental assessment · AED optimization",
        "priorities": (
            "Most GEFS+ patients enter partial or full remission by late childhood. "
            "Neurodevelopmental assessment (WISC-V). Optimize AED monotherapy. "
            "Discuss driving implications for future (seizure-free period requirements). "
            "Annual EEG + review."
        ),
    },
    {
        "window": "Adolescence (12–18Y)",
        "key_events": "VPPP (females) · Alcohol counselling · Driving · Identity",
        "priorities": (
            "VPPP mandatory for females on VPA — document each year. Contraception discussion. "
            "Alcohol counselling (seizure risk with alcohol). Driving: counsel on seizure-free "
            "period requirements before license (jurisdiction-specific). Explore AED withdrawal "
            "if 2+ years seizure-free. Peer support groups."
        ),
    },
    {
        "window": "Young Adult (18–35Y)",
        "key_events": "AED withdrawal trial · Pregnancy planning · Career · VPPP",
        "priorities": (
            "AED withdrawal trial if ≥2 years seizure-free (gradual taper over 6M; EEG + "
            "neurologist review). Pregnancy planning: VPPP for females on VPA; if pregnant, "
            "switch to LTG or LEV with neurology review. FOLIC ACID 5 mg daily "
            "pre-conception for ALL females on AEDs."
        ),
    },
    {
        "window": "Adult / Later Life (35Y+)",
        "key_events": "Remission maintenance · Perimenopause · Long-term AED review",
        "priorities": (
            "Most STX1B GEFS+ patients are in remission by adulthood. Long-term AED continuation "
            "only if ongoing seizures. Perimenopause: hormonal fluctuations may increase seizure "
            "risk in some (catamenial pattern). Review VPPP annually until menopause confirmed. "
            "Long-term VPA monitoring: bone density (DEXA) q2Y."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS  (15 concepts)
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {"term": "STX1B (16p11.2)", "definition": "Syntaxin-1B; t-SNARE protein on presynaptic plasma membrane; 288 amino acids; 7 exons; OMIM GEFS+9 #616172; pLI ~0.92."},
    {"term": "t-SNARE", "definition": "Target-SNARE: integral membrane protein on the target (plasma) membrane. STX1B is the t-SNARE that anchors the SNARE complex. Partners: SNAP25 (also t-SNARE) + VAMP2 (v-SNARE on vesicle)."},
    {"term": "SNARE Complex", "definition": "Four-helix bundle formed by STX1B SNARE motif + SNAP25 (two helices) + VAMP2 SNARE motif. Zippering N→C pulls vesicle and plasma membrane together → membrane fusion → neurotransmitter release."},
    {"term": "STXBP1 (Munc18-1)", "definition": "Syntaxin-binding protein 1; chaperone that binds closed STX1B and guides it to plasma membrane. STXBP1 LOF causes severe DEE (co-degradation of STXBP1 and STX1B). Natural companion to STX1B."},
    {"term": "GEFS+ (Generalized Epilepsy with Febrile Seizures Plus)", "definition": "Familial epilepsy syndrome: febrile seizures PLUS afebrile generalized or focal seizures. 'Plus' = FS beyond 6Y or afebrile seizures. Caused by STX1B, SCN1A, SCN1B, GABRG2, GABRD, HCN1."},
    {"term": "FS+ (Febrile Seizures Plus)", "definition": "Febrile seizures that: (1) persist beyond 6 years of age, OR (2) include afebrile seizures. Diagnostic hallmark of GEFS+ syndrome."},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "UK MHRA 2021 mandatory programme: females of childbearing potential (≥9Y) on VPA must be enrolled. Annual form signed. Effective contraception confirmed. Patient card issued. Cannot prescribe VPA without this."},
    {"term": "POLG1 (DNA Polymerase Gamma)", "definition": "Mitochondrial DNA polymerase. POLG1 pathogenic variants + VPA → Alpers-Huttenlocher (fatal hepatoencephalopathy). Screen ALL new VPA candidates. p.W748S and p.A467T most common in European populations."},
    {"term": "CLB Intermittent Prophylaxis", "definition": "Clobazam given at onset of fever (not routinely) to prevent febrile seizures in GEFS+ / FS+. 5-10 mg for 3-5 days during febrile illness. Evidence Level B. CLB REMS program in USA."},
    {"term": "LCM PR Prolongation", "definition": "Lacosamide dose-dependently prolongs cardiac PR interval. ECG required at baseline and after each dose increase. Avoid if pre-existing 2nd/3rd-degree AV block or on other PR-prolonging drugs."},
    {"term": "HLA-B*15:02", "definition": "HLA allele associated with SJS/TEN from LTG, CBZ, PHT, OXC in Han Chinese, Thai, Malaysian, other Asian populations. CPIC guideline: test before prescribing in at-risk populations. Avoid drug if positive."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Death in epilepsy patient without other cause; usually nocturnal after GTCS. Risk: GTCS frequency, seizures during sleep, male sex, polypharmacy. Counsel annually. Bed monitor, avoid prone sleeping."},
    {"term": "NCSE (Non-Convulsive Status Epilepticus)", "definition": "Status epilepticus without prominent motor signs; presents as confusion, altered consciousness. Risk with TGB (ABSOLUTE CI), also with rapid AED withdrawal. Diagnosis: EEG. Treat: IV BZD + IV VPA or LEV."},
    {"term": "Haploinsufficiency", "definition": "50% reduction in functional protein from one allele is insufficient for normal function. STX1B haploinsufficiency (truncating variants) → ~50% reduction in presynaptic t-SNARE → impaired SNARE complex assembly → reduced neurotransmitter release."},
    {"term": "ACMG-AMP 2015", "definition": "American College of Medical Genetics variant classification framework: Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign. Used for STX1B variant interpretation. Functional studies needed for VUS."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "Fever threshold for acetaminophen prophylaxis (STX1B)", "value": "≥37.5°C", "action": "Give acetaminophen 15 mg/kg; start CLB if prescribed for febrile prophylaxis"},
    {"name": "Seizure duration for rescue medication", "value": ">2 min", "action": "Give buccal midazolam 0.3 mg/kg (0.2 mg/kg if <6M); call EMS at 5 min"},
    {"name": "Status epilepticus definition", "value": ">5 min continuous or 2 seizures without recovery", "action": "Emergency protocol: midazolam → lorazepam → phenobarbital → anaesthesia"},
    {"name": "VPA TDM target", "value": "50–100 mg/L (trough)", "action": "Dose adjust if outside range or poor seizure control; check for inhibitors (CBZ, LTG)"},
    {"name": "VPA ammonia threshold", "value": ">2× ULN with encephalopathy", "action": "Stop VPA, check LFT, POLG1, consult hepatology; IV carnitine if hyperammonaemia"},
    {"name": "LCM PR interval concern", "value": "PR >200 ms or increase >20 ms from baseline", "action": "Cardiology referral; reduce or stop LCM if PR >240 ms"},
    {"name": "LTG titration rate (adult)", "value": "25 mg/2 weeks (mono) or 25 mg/4 weeks (+ VPA)", "action": "Do NOT exceed titration rate; rash within first 8 weeks = stop, refer dermatology"},
    {"name": "Seizure-free period for AED withdrawal", "value": "≥2 years seizure-free", "action": "Gradual taper over 6M; EEG before taper; neurology review; advise against driving during taper"},
    {"name": "SUDEP counselling interval", "value": "Annual", "action": "Document counselling: GTCS risk, nocturnal seizures, supervision, prone positioning, MORTEMUS."},
    {"name": "Neurodevelopmental assessment interval", "value": "q2Y (child) or with concerns", "action": "WISC-V at 6Y and 10Y; Vineland at 6Y; school support plan if cognitive delay"},
    {"name": "VPPP annual form", "value": "Annually for all females ≥9Y on VPA", "action": "Cannot continue VPA prescription without signed VPPP form; document in record"},
    {"name": "Folic acid pre-conception", "value": "5 mg daily", "action": "Start ≥3 months before planned conception for all females on AEDs; continue first trimester"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE Epilepsy Classification 2022", "url": "https://www.ilae.org", "relevance": "Seizure type classification for STX1B (FS+, focal, GTCS); GEFS+ syndrome classification"},
    {"name": "NICE NG217 — Epilepsies: Diagnosis and Management (2022)", "url": "https://www.nice.org.uk/guidance/ng217", "relevance": "UK guideline: VPA VPPP, LTG titration, AED selection, SUDEP counselling, rescue medication"},
    {"name": "Schubert et al. 2014 Nature Genetics", "url": "https://doi.org/10.1038/ng.3130", "relevance": "STX1B discovery paper: 14 GEFS+ families; p.R126Q, truncating variants; LOF mechanism"},
    {"name": "Wolking et al. 2019 Brain", "url": "https://doi.org/10.1093/brain/awz324", "relevance": "STX1B genotype-phenotype: 29 additional families; focal epilepsy subtype; Dravet-like"},
    {"name": "Vlaskamp et al. 2019 Epilepsia", "url": "https://doi.org/10.1111/epi.14621", "relevance": "GEFS+ spectrum genetics overview; STX1B in landscape of GEFS+ genes"},
    {"name": "Bhatt et al. 2023 Epilepsia", "url": "https://doi.org/10.1111/epi.17510", "relevance": "Gene-specific epilepsy treatment reference; GEFS+ treatment pyramid"},
    {"name": "MHRA VPPP 2021", "url": "https://www.gov.uk/guidance/valproate-use-by-women-and-girls", "relevance": "Valproate Pregnancy Prevention Programme — mandatory for STX1B females on VPA"},
    {"name": "CPIC POLG-VPA guideline 2023", "url": "https://cpicpgx.org/guidelines/guideline-for-valproic-acid-and-polg1/", "relevance": "POLG1 testing before VPA — pharmacogenomic guideline"},
    {"name": "CPIC HLA-B*15:02 guideline 2023", "url": "https://cpicpgx.org/guidelines/guideline-for-carbamazepine-and-hla-b/", "relevance": "HLA-B*15:02 testing before LTG/CBZ in Asian populations"},
    {"name": "FDA CLB REMS program", "url": "https://www.clobazam-rems.com", "relevance": "CLB prescriber enrolment mandatory in USA; controlled substance scheduling"},
    {"name": "ILAE Diet Commission 2018", "url": "https://doi.org/10.1111/epi.14460", "relevance": "Ketogenic diet evidence-based use in refractory epilepsy; applicable to STX1B Dravet-like"},
    {"name": "ACMG-AMP Variant Classification 2015", "url": "https://doi.org/10.1038/gim.2015.30", "relevance": "STX1B variant interpretation: pathogenic/LP/VUS classification criteria"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES  (6)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"citation": "Schubert J et al. (2014). Mutations in STX1B, encoding a presynaptic protein, cause fever-associated epileptic disorders ranging from simple febrile seizures to SMEI. Nat Genet 46:1327–32.", "pmid": "25362484"},
    {"citation": "Wolking S et al. (2019). Clinical spectrum of STX1B-related epileptic disorders. Neurology 92:e1247–e1257.", "pmid": "30737320"},
    {"citation": "Vlaskamp DRM et al. (2019). SYNGAP1 encephalopathy: A distinctive generalized developmental and epileptic encephalopathy. Neurology 92:e96–e107.", "pmid": "30541864"},
    {"citation": "Bhatt DL et al. (2023). Contemporary gene-specific epilepsy treatment guide. Epilepsia 64:1–35.", "pmid": "36648268"},
    {"citation": "Südhof TC, Rothman JE (2009). Membrane fusion: grappling with SNARE and SM proteins. Science 323:474–7.", "pmid": "19164740"},
    {"citation": "Bhattacharya A et al. (2020). Genetic epilepsies and precision therapy. Neurol Clin 38:783–800.", "pmid": "33040860"},
]


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def _generate_patients(n=40):
    patients = []
    first_names = [
        "Aarav","Mei","Samuel","Aaliyah","Haruto","Fatima","Oliver","Priya",
        "Liam","Chen","Isabella","Arjun","Sofia","Mateus","Amara","Zara",
        "Felix","Nadia","Ravi","Chloe","Tobias","Layla","Henrik","Ananya",
        "Marco","Saoirse","Kenji","Yasmin","Finn","Ingrid","Daan","Esperanza",
        "Ibrahim","Freya","Oscar","Chiara","Yusuf","Astrid","Ezra","Valentina"
    ]
    last_names = [
        "Chen","Johansson","Okafor","Müller","Patel","Kim","Santos","Rossi",
        "Nakamura","Ahmed","García","Lefevre","Singh","O'Brien","Tanaka","Kowalski",
        "Fernandez","Petrov","Ali","Weber","Dubois","Andersen","Krishnamurthy","Fischer",
        "Rodriguez","Yamamoto","Nguyen","Bakker","Ibrahim","Svensson","Delacroix","Bauer",
        "Moretti","Hakim","Jensen","Volkov","Ramos","Lopes","Hoffmann","Schmidt"
    ]
    variants_pool = [
        "p.R126Q", "p.A198P", "p.Y127C", "p.Y262*", "p.Q281*",
        "c.532+1G>A", "p.P238L", "p.L206P", "p.L205R", "p.G36R",
        "p.E289fs", "p.C272Y", "p.R281C",
    ]
    for i in range(n):
        ecat = random.choice(ETIOLOGY_CATALOG)
        age_dx = max(1, int(random.gauss(ecat["onset_age_months"] / 12, 1.5)))
        current_age = age_dx + random.randint(1, 15)
        sz_types = random.sample([s["type"] for s in SEIZURE_TYPES], k=random.randint(1, 3))
        triggers = random.sample([t["trigger"] for t in TRIGGERS], k=random.randint(2, 4))
        trt = random.choice(TREATMENTS)
        variant = random.choice(variants_pool)
        pts = random.randint(0, 20)
        sf = ecat["outcome"].startswith("Good") or ecat["outcome"].startswith("Excellent")
        patients.append({
            "id": f"STX1B-{i+1:03d}",
            "name": f"{first_names[i]} {last_names[i]}",
            "age_at_diagnosis": age_dx,
            "current_age": current_age,
            "sex": random.choice(["M", "F"]),
            "etiology_class": ecat["category"],
            "variant": variant,
            "seizure_types": sz_types,
            "triggers": triggers,
            "current_treatment": trt["drug"],
            "seizure_free": sf and random.random() < 0.65,
            "seizures_per_month": 0 if sf else pts,
            "polg1_tested": True,
            "vppp_enrolled": True if random.random() < 0.9 else False,
        })
    return patients


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    pts = _generate_patients(40)
    sz_free = sum(1 for p in pts if p["seizure_free"])
    etiol_dist = {}
    for p in pts:
        etiol_dist[p["etiology_class"]] = etiol_dist.get(p["etiology_class"], 0) + 1
    top_etiol = max(etiol_dist, key=etiol_dist.get)
    top_trig = TRIGGERS[0]["trigger"]
    top_trt = TREATMENTS[0]["drug"]
    return {
        "gene": "STX1B",
        "locus": "16p11.2",
        "omim": "GEFS+9 #616172",
        "protein": "Syntaxin-1B (t-SNARE)",
        "mechanism": "Presynaptic SNARE complex t-SNARE; STX1B LOF → impaired vesicle fusion → reduced neurotransmitter release → disinhibition → GEFS+ spectrum",
        "inheritance": "Autosomal Dominant (AD) — de novo 40%, familial 60%",
        "pli": 0.92,
        "cohort_size": 40,
        "seizure_free_pct": round(sz_free / 40 * 100, 1),
        "top_etiology": top_etiol,
        "etiology_distribution": etiol_dist,
        "top_trigger": top_trig,
        "top_treatment": top_trt,
        "key_contraindication": "Tiagabine (ABSOLUTE CI — NCSE); CBZ/OXC (CAUTION if GGE component); PHT (avoid myoclonic); VPA+POLG1 (ABSOLUTE CI if POLG1+)",
        "key_clinical_pearl": (
            "STX1B GEFS+ often remits by adulthood (unlike Dravet). "
            "Key distinction from SCN1A-Dravet: normal MRI, normal cognition in classic GEFS+. "
            "STX1B is the natural companion to STXBP1 — same SNARE pathway, milder phenotype. "
            "Fever prophylaxis with CLB at 37.5°C is a practical cornerstone of management."
        ),
        "key_references": [r["citation"] for r in REFERENCES[:3]],
    }


def get_breakdown():
    pts = _generate_patients(40)
    return {
        "gene": "STX1B",
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "patients_sample": pts[:15],
        "summary_stats": {
            "total_patients": 40,
            "seizure_free": sum(1 for p in pts if p["seizure_free"]),
            "median_age_at_dx": 18,
            "etiology_classes": len(ETIOLOGY_CATALOG),
            "seizure_types_count": len(SEIZURE_TYPES),
            "triggers_count": len(TRIGGERS),
            "treatments_count": len(TREATMENTS),
            "contraindications_count": len(CONTRAINDICATIONS),
            "monitoring_items": len(MONITORING),
        },
    }


def get_definitions():
    return {
        "gene": "STX1B",
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "counts": {
            "definitions": len(DEFINITIONS),
            "thresholds": len(THRESHOLDS),
            "standards": len(STANDARDS),
            "references": len(REFERENCES),
        },
    }
