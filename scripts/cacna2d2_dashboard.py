"""
CACNA2D2 Epilepsy — Epileptic Encephalopathy with Cerebellar Atrophy & Hypotonia / α2-δ-2 Auxiliary Subunit / Gabapentinoid-Binding Protein / AR-LOF / 3p21.3
===============================================================================================================================================================
40-patient cohort · CACNA2D2 (3p21.3) · Voltage-Gated Ca²⁺ Channel Auxiliary Subunit α2-δ-2 · AR biallelic LOF
Gene OMIM: *602843 · Syndrome: Epileptic Encephalopathy with Cerebellar Atrophy, Hypotonia, and Impaired Intellectual Development (EECAT)

KEY CACNA2D2 BIOLOGY — α2-δ-2 AUXILIARY SUBUNIT OF VOLTAGE-GATED Ca²⁺ CHANNELS:
CACNA2D2 (3p21.3) encodes the α2-δ-2 (alpha2-delta-2) auxiliary subunit of voltage-gated calcium channels.
The α2-δ subunit family has 4 members: α2-δ-1 (CACNA2D1), α2-δ-2 (CACNA2D2), α2-δ-3 (CACNA2D3), α2-δ-4 (CACNA2D4).

PROTEIN STRUCTURE:
  · α2 and δ peptides are encoded by a single gene and post-translationally cleaved; reassociated via disulfide bond.
  · α2 faces extracellularly; δ is GPI-anchored (or transmembrane in some isoforms) in the outer leaflet.
  · The α2-δ-2 subunit co-assembles with Cav1 and Cav2 α1 subunits to form functional channel complexes.
  · MOLECULAR WEIGHT: α2 ~130 kDa; δ ~28 kDa; combined ~150 kDa.

α2-δ-2 FUNCTIONS:
  1. CHANNEL TRAFFICKING: α2-δ promotes forward trafficking of α1 pore subunits from ER→Golgi→plasma membrane.
     α2-δ-2 LOF → reduced surface expression of Ca²⁺ channels → reduced total calcium current.
  2. CHANNEL KINETICS: modulates voltage-dependence and kinetics of inactivation; stabilises channel complexes.
  3. SYNAPTOGENESIS: α2-δ subunits (especially α2-δ-2) are involved in synapse formation — extracellular matrix
     interactions with thrombospondins (TSP1/TSP2) regulate excitatory synapse development.
  4. PRESYNAPTIC Ca²⁺ ENTRY: α2-δ-2 supports Cav2.1 (P/Q-type) and Cav2.2 (N-type) at active zones →
     neurotransmitter release modulation.

EXPRESSION PATTERN:
  · Highly expressed in cerebellum (Purkinje cells, granule cells — highest density), cortex, hippocampus.
  · Particularly enriched in presynaptic terminals of glutamatergic and GABAergic neurons.
  · PURKINJE CELLS: α2-δ-2 is DOMINANT α2-δ subunit → LOF causes cerebellar dysfunction preferentially.
  · ANIMAL MODEL: CACNA2D2 null mice (Barclay 2001 Nat Neurosci) develop:
      - Spike-wave discharges (SWD) on EEG
      - Cerebellar ataxia with ducky gait (tail suspension: hindlimb clasping)
      - Purkinje cell loss over time
      - "Ducky" mutant mice — spontaneous LOF mutation studied since 1990s.

GABAPENTINOID BINDING — KEY CLINICAL RELEVANCE:
  · Gabapentin (Neurontin) and Pregabalin (Lyrica) were originally discovered to bind α2-δ subunits.
  · PRIMARY GABAPENTINOID TARGET: α2-δ-1 (CACNA2D1) → analgesic + anticonvulsant effects in pain syndromes.
  · α2-δ-2 (CACNA2D2) is ALSO a gabapentinoid binding site — lower affinity than α2-δ-1.
  · CLINICAL IMPLICATION: In CACNA2D2 LOF patients, the α2-δ-2 subunit is already dysfunctional.
    Gabapentinoids targeting this subunit have reduced/unpredictable pharmacodynamics in CACNA2D2 patients.
    Additionally, gabapentinoids exacerbate cerebellar symptoms (dizziness, ataxia, vertigo) which are
    already severe in CACNA2D2 patients — HIGH RISK of worsening cerebellar ataxia.
  · GABAPENTIN/PREGABALIN in CACNA2D2: HIGH RISK — preferentially AVOID (worsen cerebellar ataxia;
    reduced efficacy on α2-δ-2; α2-δ-1 may remain a partial target but clinical benefit unproven).

PHENYTOIN / CARBAMAZEPINE RISK:
  · PHT causes dose-dependent cerebellar toxicity (cerebellar atrophy with chronic use) — particularly
    dangerous in CACNA2D2 patients who already have cerebellar atrophy/hypoplasia.
  · CBZ chronic use also associated with cerebellar toxicity (less prominent than PHT).
  · PHT: HIGH RISK — cerebellar atrophy + Purkinje cell loss compound pre-existing atrophy.
  · CBZ/OXC: HIGH RISK in myoclonic phenotype; moderate risk cerebellar toxicity long-term.

PRECISION TREATMENT NOTE:
  · No approved precision therapy for CACNA2D2 DEE as of 2026.
  · Investigational approaches: α2-δ-2-deficient neurons may respond to NMDA receptor modulators,
    K-ATP channel openers (diazoxide — reduces neuronal excitability), mTOR pathway normalisation.
  · Ketogenic diet: most effective DRE option; reduces Cav2 channel activity via ketone body effects.
  · Riboflavin (B2): reported benefit in one CACNA2D2 case (Shahar 2016) — mechanism unclear; trial warranted.

CEREBELLAR-EPILEPSY RELATIONSHIP:
  · Cerebellar circuits can modulate seizure propagation; Purkinje cell-dentate nucleus pathway
    influences thalamo-cortical excitability. Cerebellar stimulation (DBS/TMS) may suppress cortical SWD.
  · Cerebellar atrophy progression: monitor serial MRI every 12–24 months.
  · Cerebellar hypoplasia vs atrophy: some patients have congenital hypoplasia; others develop progressive atrophy.
"""

import random

random.seed(99)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG — 5-CLASS LOF SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "Biallelic-LOF-Truncating-Severe-EECAT",
        "pct": 42,
        "mechanism": "Biallelic truncating variants (frameshift/nonsense/splice-site) → complete loss of α2-δ-2 protein → maximal reduction of Cav1/Cav2 trafficking → absent synaptic α2-δ-2 → severe cerebellar Purkinje cell Ca²⁺ deficiency → EECAT + profound cerebellar atrophy",
        "eeg": "Hypsarrhythmia in infancy → later multifocal spike-wave + generalised SWD; cerebellar atrophy on MRI from early infancy",
        "onset_months": "3–8 months",
        "severity": "severe",
    },
    {
        "category": "Biallelic-LOF-Missense-Moderate-EECAT",
        "pct": 28,
        "mechanism": "Biallelic missense variants → partial α2-δ-2 dysfunction; reduced Cav2.1/Cav2.2 trafficking; residual channel expression (5–30% of wildtype) → moderate cerebellar atrophy + epilepsy; some cognitive sparing relative to truncating",
        "eeg": "Generalised SWD 2.5–3.5 Hz; focal spike-wave in some; improved EEG over years in subset",
        "onset_months": "6–18 months",
        "severity": "moderate–severe",
    },
    {
        "category": "Biallelic-LOF-CNV-Deletion-3p21",
        "pct": 15,
        "mechanism": "3p21.3 homozygous or hemizygous deletion encompassing CACNA2D2; co-deleted genes may include CNTN4 or other 3p21 loci → contiguous gene syndrome features possible (cardiac, craniofacial) in addition to EECAT core phenotype",
        "eeg": "Multifocal epileptiform discharges; seizures often multiple types simultaneously",
        "onset_months": "2–6 months",
        "severity": "severe",
    },
    {
        "category": "Compound-Heterozygous-LOF-Intermediate",
        "pct": 10,
        "mechanism": "Trans compound heterozygous (one truncating + one missense LOF); intermediate α2-δ-2 residual function → moderate EECAT; cerebellar features often present but milder cerebellar atrophy progression",
        "eeg": "Generalised + focal spike-wave; some tonic predominance; occasional absence-like episodes",
        "onset_months": "6–24 months",
        "severity": "moderate",
    },
    {
        "category": "Phenocopy-Panel-Negative",
        "pct": 5,
        "mechanism": "Epileptic encephalopathy with cerebellar atrophy phenocopy; CACNA2D2 panel negative; alternative genetic or metabolic cause (ITPR1/RELN/SLC1A3/congenital disorders of glycosylation); CACNA2D2 phenotype suspected clinically but unconfirmed molecularly",
        "eeg": "Variable; usually severe DEE pattern",
        "onset_months": "3–18 months",
        "severity": "moderate–severe",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES — 5 TYPES
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms (West Syndrome onset)",
        "frequency_pct": 65,
        "eeg_correlate": "Hypsarrhythmia (chaotic high-voltage); electrodecrement at spasm; inter-spasm multifocal spikes; often evolves to Lennox-Gastaut pattern",
        "semiology": "Flexor/extensor/mixed spasms; clusters of 10–30 on waking; subtle spasms possible in neonates (eye deviation, facial flush)",
        "clinical_tip": "ACTH is Level A for West syndrome; vigabatrin Level A (UKISS 2004). In CACNA2D2 patients, vigabatrin VFD monitoring (ERG q3M SHARE REMS) is non-negotiable — baseline VF before VGB start.",
    },
    {
        "type": "Generalised Tonic-Clonic (GTCS)",
        "frequency_pct": 80,
        "eeg_correlate": "Generalised polyspike/polyspike-wave burst → EEG suppression post-ictally; often evolves from infantile spasms pattern; paroxysmal fast activity in tonic phase",
        "semiology": "Tonic stiffening (bilateral) → clonic jerks; often nocturnal in CACNA2D2; prolonged post-ictal ataxia (worsens existing baseline ataxia by 30–120 min post-GTCS)",
        "clinical_tip": "Post-GTCS ataxia is prolonged in CACNA2D2 (cerebellar baseline dysfunction magnifies Todd's paresis equivalent). Educate caregivers: ataxia after GTCS is NOT a new neurological event.",
    },
    {
        "type": "Myoclonic Seizures",
        "frequency_pct": 72,
        "eeg_correlate": "Polyspike-wave or generalised SWD 2.5–4 Hz; enhanced by photic stimulation (30–35%) and drowsiness; JERK-LOCKED AVERAGE: cortical myoclonus pattern with back-averaging",
        "semiology": "Brief bilateral jerks; upper extremities most prominent; often provoked by touch, startle, drowsiness; exacerbated by fatigue (relevant to ataxic gait — myoclonus + ataxia = high fall risk)",
        "clinical_tip": "ASSESS MYOCLONUS TYPE: if cortical myoclonus → SEP giant potentials (not always present in CACNA2D2 unlike EPM). Distinguish from spasticity tremor. VPA + CLB combination effective for myoclonus in CACNA2D2.",
    },
    {
        "type": "Tonic Seizures",
        "frequency_pct": 55,
        "eeg_correlate": "Paroxysmal fast activity (10–20 Hz) over frontal leads; EEG flattening during tonic phase; may occur in clusters during sleep (NREM stage N2 activation — distinguishing from Lennox-Gastaut requires clinical correlation)",
        "semiology": "Axial tonic → axial+limb; eye deviation; brief (5–20 s); nocturnal clustering. In CACNA2D2, tonic seizures have more prominent cerebellar posturing component (decerebrate posturing in severe cases).",
        "clinical_tip": "Tonic seizures during sleep are poorly responsive to all standard AEDs in CACNA2D2 DEE. KD is most effective. CLB adjunct reduces frequency. AVOID PHT for tonic (cerebellar toxicity worsens ataxia long-term).",
    },
    {
        "type": "Focal Seizures with Impaired Awareness",
        "frequency_pct": 38,
        "eeg_correlate": "Focal rhythmic theta/alpha activity; temporal or frontoparietal; secondary generalisation common; interictal regional spike-wave; EEG may show predominant laterality (right > left in some reports)",
        "semiology": "Behavioural arrest + automatisms (oral/manual); head deviation; rare in early infancy; more common as children age (shift from generalised to focal pattern by 3–5 years in subset)",
        "clinical_tip": "Focal seizures in CACNA2D2 suggest secondary focal epileptogenesis (cortical dysmaturation rather than true FCD). MRI often shows cerebellar atrophy + cortical/subcortical signal changes rather than focal lesion.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS — 8 KEY TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Hyperthermia", "pct": 88, "note": "Temperature ≥37.5°C → markedly lowers seizure threshold in CACNA2D2; proactive paracetamol (15 mg/kg q4h) + tepid sponging for any febrile illness; worsens cerebellar ataxia concomitantly"},
    {"trigger": "Sleep Deprivation / Disruption", "pct": 78, "note": "REM/NREM sleep modulates thalamo-cortical synchrony; CACNA2D2 patients have high nocturnal GTCS rate; strict sleep schedule; monitor for OSA (hypotonia → pharyngeal collapse)"},
    {"trigger": "Missed AED Doses", "pct": 72, "note": "High seizure burden returns rapidly within 24h of missed dose; AED non-adherence in hypotonic patients (swallowing difficulty) → liquid formulations preferred; NG/PEG tube if oral intake unreliable"},
    {"trigger": "Intercurrent Illness (non-febrile)", "pct": 62, "note": "Metabolic stress (dehydration, vomiting, electrolyte disturbance) worsens channelopathy → increase monitoring during any illness; IV valproate if cannot take orally"},
    {"trigger": "Fatigue / Physical Overexertion", "pct": 52, "note": "CACNA2D2 patients fatigue rapidly (cerebellar + hypotonia); post-exertional myoclonus clusters; scheduled rest periods; adapted PE programme"},
    {"trigger": "Photosensitivity", "pct": 35, "note": "PPR type III–IV on IPS (intermittent photic stimulation); avoid stroboscopic environments; 10–14 Hz most provocative; polarised glasses for photosensitive CACNA2D2 patients"},
    {"trigger": "Startle / Tactile Stimulation", "pct": 30, "note": "Cortical myoclonus may be startle-provoked; loud noises, unexpected touch → myoclonic burst; reduce sudden startles in care environment; warn caregivers about handling technique"},
    {"trigger": "AED Withdrawal / Rapid Taper", "pct": 65, "note": "CACNA2D2 patients have high relapse rate on AED withdrawal; do NOT taper unless seizure-free ≥3 years; taper ≥12 months minimum; EEG monitoring during taper (q3M)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS — 8 KEY AEDs/INTERVENTIONS
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproic Acid (VPA) — Depakene/Depakote (Level B, First-line broad-spectrum)",
        "level": "Level B (ILAE; observational series; no CACNA2D2-specific RCT)",
        "dose": "20–40 mg/kg/day in 2–3 divided doses; target TDM 50–100 mg/L (trough); neonates/infants: start 10 mg/kg/day, titrate weekly",
        "moa": "Multi-mechanism: Na+ channel inactivation, GABA-T inhibition (↑ synaptic GABA), T-type Ca²⁺ channel blockade (Cav3.1/3.2), HDAC inhibition (histone acetylation → gene regulatory). Broad-spectrum: GTCS + myoclonus + tonic + infantile spasms.",
        "efficacy": "35–50% ≥50% seizure reduction in CACNA2D2 DEE (observational); best for myoclonic component; partial response to tonic/GTCS",
        "monitoring": "VPA TDM (50–100 mg/L trough) q3M; LFT + FBC + ammonia q3M; POLG1 exclusion BEFORE initiation (VPA-induced hepatotoxicity in POLG1 pathogenic variant carriers — FATAL); VPPP for females ≥12y (MHRA 2021); carnitine if deficiency suspected",
        "cacna2d2_note": "First-line choice for broad-spectrum coverage of GTCS + myoclonus. POLG1 must be excluded before use. If POLG1 positive or cannot be excluded urgently → choose LEV instead.",
    },
    {
        "drug": "Levetiracetam (LEV) — Keppra (Level B, First-line or VPA alternative)",
        "level": "Level B (ILAE; broad observational use in DEE; no CACNA2D2 RCT)",
        "dose": "20–60 mg/kg/day in 2 divided doses; IV equivalent dose 1:1 if NPO; titrate over 4 weeks",
        "moa": "SV2A (synaptic vesicle protein 2A) binding → modulates vesicle priming and neurotransmitter release; reduces Ca²⁺-dependent exocytosis; mild Cav2 modulation (secondary). No hepatotoxicity risk — preferred over VPA when POLG1 unknown/positive.",
        "efficacy": "30–45% ≥50% seizure reduction; good tolerability; behavioural side effects (irritability, aggression) reported in 20–30% — important in non-verbal CACNA2D2 patients (express as self-injurious behaviour)",
        "monitoring": "Renal function q6M (LEV is renally cleared; reduce dose if eGFR <50); behavioural monitoring (CARS-2 + carer report q12W); serum LEV not routinely needed (clinical titration)",
        "cacna2d2_note": "Preferred when POLG1 cannot be excluded or when VPA hepatotoxicity risk is high. Reduces GTCS and tonic seizures. Behavioural SE monitoring critical in non-verbal patients.",
    },
    {
        "drug": "ACTH (Adrenocorticotropic Hormone) — Level A for Infantile Spasms (West Syndrome)",
        "level": "Level A (UKISS 2004 Lancet; ILAE consensus for West syndrome/infantile spasms)",
        "dose": "High-dose synthetic ACTH: 150 IU/m²/day IM × 14 days → taper; or natural ACTH (Acthar Gel) 150 units/kg/day × 2 weeks; 2-week EEG assessment mandatory",
        "moa": "CRH receptor agonist → adrenal steroid release → suppresses CRH-driven hippocampal/amygdala hyperexcitability; direct CNS effect via MC receptor on ACTH fragments; reduces hypsarrhythmia rapidly (2–4 days)",
        "efficacy": "55–70% EEG response (cessation of hypsarrhythmia) at 14 days in CACNA2D2 infantile spasms; lower sustained remission than idiopathic West (genetic DEE has higher relapse after ACTH).",
        "monitoring": "BP (daily × 14 days), glucose (daily), weight, electrolytes, infection signs; AVOID live vaccines during ACTH; fundoscopy 4 weeks post-ACTH (glaucoma risk); EEG at Day 14 (primary endpoint)",
        "cacna2d2_note": "First-line for infantile spasms onset in CACNA2D2. If EEG response at Day 14 is partial, extend ACTH × 2 additional weeks before switching. Relapse common in CACNA2D2 (unlike tuberous sclerosis).",
    },
    {
        "drug": "Clobazam (CLB) — Onfi / Frisium (Level B, Adjunct for myoclonus + tonic)",
        "level": "Level B (observational; ILAE 2022 adjunct recommendation for DEE)",
        "dose": "0.1–0.3 mg/kg/day (paediatric); max 10 mg/day in children <2y; 10–30 mg/day adults; in 1–2 divided doses (once daily if tolerated — long T½ ~36h active metabolite N-desmethylclobazam)",
        "moa": "1,5-benzodiazepine (more selective GABA-A agonist than 1,4-BZDs like diazepam); α1/α5 GABA-A subunit; anticonvulsant + mild muscle relaxant; N-desmethylclobazam (active metabolite) T½ ~72h",
        "efficacy": "40–55% ≥50% seizure reduction (myoclonic + tonic in CACNA2D2); tolerance develops in 30–50% over 6–12 months (GABA-A downregulation); drug holiday protocol helps reset efficacy",
        "monitoring": "Respiratory function (hypotonia → apnoea risk with CLB — review at each visit); sedation scale (UMSS); salivary secretion (aspiration risk in hypotonic CACNA2D2); CYP2C19 phenotyping for N-DMC accumulation",
        "cacna2d2_note": "Effective adjunct for myoclonic + tonic seizures. Caution: hypotonia worsens with CLB → aspirate risk. CYP2C19 poor metabolisers accumulate N-DMC → excess sedation. Start low, titrate slowly.",
    },
    {
        "drug": "Vigabatrin (VGB) — Sabril (Level A infantile spasms — SHARE REMS mandatory)",
        "level": "Level A (infantile spasms; UKISS 2004 Lancet; FDA SHARE REMS)",
        "dose": "100–150 mg/kg/day in 2 divided doses (infantile spasms); 40–80 mg/kg/day (adjunct older children); adults 1–3 g/day",
        "moa": "GABA-T irreversible inhibitor → ↑ synaptic GABA; reduces epileptiform activity via GABA-B and GABA-A enhancement. Irreversibly binds GABA-T (requires new enzyme synthesis for recovery).",
        "efficacy": "50–60% EEG response for infantile spasms (similar to ACTH in UKISS — both ~55–70% at 14 days); NOT effective as sole AED for GTCS/myoclonic in CACNA2D2",
        "monitoring": "VISUAL FIELD TESTING: formal Goldman/Humphrey VF + ERG q3M (VGB VFD risk — 30–40% of patients, IRREVERSIBLE); formal VF baseline BEFORE starting VGB; SHARE REMS enrollment (USA); MRI signal changes (T2/FLAIR globus pallidus/brainstem in infants — transient myelin vacuolation)",
        "cacna2d2_note": "Use ONLY for infantile spasms; discontinue once spasms controlled (do not continue long-term for other seizures — VFD risk outweighs benefit). MANDATORY: VF + ERG baseline before VGB start.",
    },
    {
        "drug": "Ketogenic Diet (KD 4:1 ratio) — Level B (Drug-Resistant Epilepsy)",
        "level": "Level B (ILAE Dietary Therapies Guidelines 2018; most effective DRE intervention in CACNA2D2)",
        "dose": "4:1 or 3:1 fat:protein+carbohydrate ratio; target BHB 2–4 mmol/L serum; dietitian-supervised; 3-month trial minimum before declaring non-response",
        "moa": "Multiple synergistic mechanisms in Ca²⁺ channelopathies: (1) reduces Cav2 channel activity via HCO3 reduction (acidosis) + ketone-body effects; (2) inhibits mTOR pathway; (3) elevates GABA via substrate competition; (4) anti-inflammatory (reduces IL-6/TNF-α in cerebellar microenvironment); (5) reduces neuronal excitability directly via KATP channel opening",
        "efficacy": "40–60% ≥50% seizure reduction in CACNA2D2 DEE (observational series); case reports of 3–6 month remission; best for myoclonic + tonic components; GTCS partially responsive",
        "monitoring": "Serum BHB q4W; glucose (hypoglycaemia in young children); micronutrients (Se/Zn/carnitine/biotin) q6M; renal stones (KD → hypercalciuria + hypocitraturia); lipid panel q12M; growth (height/weight centile) q6M; liver function q12M",
        "cacna2d2_note": "Best overall DRE option in CACNA2D2 — addresses cerebellar Ca²⁺ channel deficiency indirectly via KATP/mTOR. Modified Atkins diet (less restrictive, good adherence) as alternative. Start after 2 failed AEDs.",
    },
    {
        "drug": "Pyridoxine (Vitamin B6) — Level C (Trial for neonatal/early-onset refractory seizures)",
        "level": "Level C (empirical trial; no RCT; case reports of partial response; excludes pyridoxine-dependent epilepsy phenocopy)",
        "dose": "100 mg IV single bolus (diagnostic trial); if EEG response → pyridoxine 5–10 mg/kg/day oral maintenance; only continue if clear response",
        "moa": "Pyridoxal-5-phosphate (PLP) cofactor for GABA synthesis (GAD enzyme); in some CACNA2D2 patients, early pyridoxine trial excludes ALDH7A1 PDE and may have modest anticonvulsant effect via GAD enhancement",
        "efficacy": "Response in <15% of CACNA2D2 (mostly excludes PDE phenocopy); rarely a primary treatment; use as initial diagnostic exclusion in neonatal/early-infantile severe onset",
        "monitoring": "EEG during IV trial (15–30 min post-dose — EEG suppression = PDE response); peripheral neuropathy if >200 mg/day long-term",
        "cacna2d2_note": "Give once as diagnostic trial in any <6-month-onset severe neonatal/infantile seizure presentation BEFORE labelling as CACNA2D2 DEE. If clear EEG response → pursue ALDH7A1/PNPO testing. No response → do not continue.",
    },
    {
        "drug": "Riboflavin (Vitamin B2) — Level C (Investigational; case report benefit)",
        "level": "Level C (Shahar 2016 single case report; mechanism uncertain; no controlled trial)",
        "dose": "50–200 mg/day oral in divided doses; water-soluble; non-toxic at these doses; 3-month trial minimum",
        "moa": "Mitochondrial cofactor (FMN/FAD); may enhance residual α2-δ-2 function via mitochondrial energy coupling in cerebellar Purkinje cells; flavin-dependent enzyme support. Theoretical: CACNA2D2 LOF → Purkinje cell energy deficit → riboflavin supplementation → partial compensation.",
        "efficacy": "Single case report (Shahar 2016): striking seizure reduction in 1 CACNA2D2 patient on riboflavin; not replicated in larger series; anecdotal",
        "monitoring": "Urine turns fluorescent yellow (harmless); no monitoring required at these doses",
        "cacna2d2_note": "Consider as safe add-on after KD and standard AEDs exhausted. Well-tolerated, inexpensive, no interactions. Case report evidence only — communicate uncertainty to family.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS — 6 KEY
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Gabapentin / Pregabalin — HIGH RISK (Gabapentinoid paradox in α2-δ-2 LOF)",
        "risk": "HIGH RISK",
        "mechanism": "Gabapentinoids bind α2-δ subunits (primarily α2-δ-1, but also α2-δ-2/CACNA2D2). In CACNA2D2 LOF, α2-δ-2 is already non-functional → gabapentinoids at this site have unpredictable/reduced efficacy. Critically: gabapentinoids WORSEN CEREBELLAR ATAXIA (primary side effect: dizziness, ataxia, vertigo) — severely worsens existing cerebellar baseline. Avoid in CACNA2D2 unless no alternative; extreme caution if used.",
        "alternative": "VPA or LEV for seizures; avoid gabapentinoids for neuropathic pain in CACNA2D2 patients — use alternative analgesics (TCAs, duloxetine, opioids if necessary)"
    },
    {
        "drug": "Phenytoin (PHT) — HIGH RISK (Cerebellar toxicity compounding CACNA2D2 atrophy)",
        "risk": "HIGH RISK",
        "mechanism": "PHT causes cerebellar toxicity with chronic use (Purkinje cell loss, chronic cerebellar atrophy), even at therapeutic levels in some patients. CACNA2D2 patients already have cerebellar atrophy/hypoplasia — PHT COMPOUNDS PRE-EXISTING CEREBELLAR NEURONAL LOSS. Risk of irreversible cerebellar worsening with PHT. Use only as acute IV rescue (status epilepticus) if no alternative, NOT as maintenance AED.",
        "alternative": "VPA / LEV for maintenance. IV lorazepam → IV VPA (or IV LEV) for status epilepticus; reserve IV PHT only if both fail"
    },
    {
        "drug": "Tiagabine (TGB) — ABSOLUTE CI (Non-convulsive status epilepticus risk)",
        "risk": "ABSOLUTE CI",
        "mechanism": "TGB (GABA reuptake inhibitor — GAT-1) → non-convulsive status epilepticus (NCSE) in genetic epilepsies; highest NCSE risk of any AED in DEE spectrum. CACNA2D2 DEE patients at extreme risk. NEVER prescribe.",
        "alternative": "CLB / VPA / LEV for adjunctive therapy instead of TGB"
    },
    {
        "drug": "Valproic Acid + POLG1 Pathogenic Variant — ABSOLUTE CI (Alpers hepatotoxicity)",
        "risk": "ABSOLUTE CI",
        "mechanism": "VPA in POLG1 pathogenic variant carriers (Alpers-Huttenlocher syndrome risk) → fatal hepatotoxicity + lactic acidosis; liver failure within weeks. POLG1 testing MUST be completed before VPA initiation. If POLG1 positive or result unavailable urgently → use LEV instead.",
        "alternative": "LEV is the safe alternative in POLG1-uncertain or POLG1-positive situations"
    },
    {
        "drug": "Carbamazepine / Oxcarbazepine — HIGH RISK (Myoclonus worsening + cerebellar)",
        "risk": "HIGH RISK",
        "mechanism": "CBZ/OXC are Na⁺ channel stabilisers — narrow-spectrum, ineffective for myoclonic seizures; can WORSEN myoclonus (Na-channel activation paradox in myoclonic DEE). Long-term CBZ also associated with cerebellar toxicity (compound to CACNA2D2 cerebellar atrophy). Additionally HLA-B*15:02 SJS/TEN risk (Asian ancestry — CPIC Level A).",
        "alternative": "VPA / LEV / CLB for CACNA2D2. If focal seizures predominate and myoclonus is absent, LCM or LTG may be used with caution."
    },
    {
        "drug": "Vigabatrin (VGB) long-term maintenance — HIGH RISK (Visual Field Defect — IRREVERSIBLE)",
        "risk": "HIGH RISK for long-term use",
        "mechanism": "VGB causes irreversible visual field defects (VFD) in 30–40% of patients with prolonged use. Use is justified ONLY for infantile spasms (limited course); NOT as long-term maintenance AED. VFD is particularly devastating if patient also has cerebellar oculomotor dysfunction (common in CACNA2D2).",
        "alternative": "After spasms controlled with VGB + ACTH, transition to VPA/LEV/KD for maintenance. Discontinue VGB once spasms remit."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING — 14 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG1 testing before VPA initiation", "frequency": "Once (before first VPA dose)", "rationale": "Fatal hepatotoxicity in POLG1 pathogenic variant carriers; results must be available before starting VPA. If unavailable within 48h → use LEV instead."},
    {"item": "VPA TDM (therapeutic drug monitoring) trough", "frequency": "q3M steady state; q4W during dose changes", "rationale": "Target 50–100 mg/L; CACNA2D2 patients often malnourished/hypotonic → altered VD; monitor closely in first 6 months"},
    {"item": "LFT + FBC + ammonia (VPA monitoring)", "frequency": "q3M (steady state); q4W first 6 months", "rationale": "VPA hepatotoxicity (ALT/AST >3× ULN → hold VPA); thrombocytopaenia (FBC); hyperammonaemia (ammonia >80 μmol/L → carnitine + dose reduction)"},
    {"item": "Neuroimaging (MRI brain + cerebellum)", "frequency": "At diagnosis; q12–24M (cerebellar atrophy progression)", "rationale": "Baseline cerebellar volume; monitor progression; PHT exposure → additional atrophy monitoring; diffusion tensor imaging for white matter tract integrity"},
    {"item": "Seizure diary + video capture", "frequency": "Daily diary; video for each new seizure type", "rationale": "CACNA2D2 phenotype evolves (infantile spasms → myoclonus → tonic/GTCS); video EEG correlation needed at each phenotype transition"},
    {"item": "Neurodevelopmental assessment (Griffiths/Bayley/VABS)", "frequency": "q6M (infants); q12M (children)", "rationale": "Track developmental trajectory; CACNA2D2 DEE typically causes severe GDD; early intensive therapy improves functional outcomes"},
    {"item": "VGB visual field + ERG monitoring (SHARE REMS)", "frequency": "Baseline before VGB start; q3M during VGB use", "rationale": "VFD 30–40% with prolonged VGB; ERG (ring-ring ERG amplitude reduction precedes VF loss by months); formal Goldmann perimetry when feasible; SHARE REMS enrollment mandatory in USA"},
    {"item": "KD monitoring (BHB, glucose, micronutrients)", "frequency": "BHB q4W; glucose daily first month; Se/Zn/carnitine/biotin q6M; lipid panel q12M", "rationale": "Ensure adequate ketosis (BHB 2–4 mmol/L); prevent hypoglycaemia in infants; micronutrient deficiency common in KD-restricted diet; renal stone risk"},
    {"item": "Respiratory / aspiration assessment (hypotonia)", "frequency": "At each visit; chest physio q3M; swallowing assessment annually", "rationale": "CACNA2D2 hypotonia → pharyngeal dysfunction → aspiration pneumonia risk (leading cause of hospitalisations); videofluoroscopy swallowing study annually; chest physio protocol"},
    {"item": "Cardiac monitoring (hypotonia → respiratory compromise)", "frequency": "Holter 24h at diagnosis; ECG annually", "rationale": "Hypotonia-related bradycardia/arrhythmia; CLB respiratory compromise monitoring; autonomic dysregulation in severe DEE"},
    {"item": "VPPP (Valproate Pregnancy Prevention Programme)", "frequency": "Annual risk form + contraceptive review for females ≥12y on VPA", "rationale": "MHRA 2021 mandatory; VPA teratogenicity (spina bifida 2–3%, NTD); neurodevelopmental risk child; Form completion mandatory UK/EU; switch females pre-conception to LEV/LTG"},
    {"item": "Orthopaedic / physiotherapy review (hypotonia + ataxia)", "frequency": "q6M physiotherapy; orthopaedic q12M (scoliosis, hip dislocation)", "rationale": "CACNA2D2 hypotonia + cerebellar ataxia → scoliosis, hip subluxation, contractures; early physio prevents deformity; AFO orthoses for ambulation support"},
    {"item": "SUDEP risk assessment and caregiver education", "frequency": "Annually; at every seizure type change", "rationale": "CACNA2D2 DEE has SUDEP risk similar to other severe DEE (3–7× general population); nocturnal GTCS are highest risk; nocturnal monitoring; co-sleeping contraindicated; SUDEP education mandatory"},
    {"item": "Ophthalmology (cerebellar oculomotor + VGB VFD)", "frequency": "At diagnosis; q12M; q3M if VGB use", "rationale": "Cerebellar atrophy → nystagmus, ocular motor apraxia, saccadic dysmetria; VGB VFD monitoring (ERG + VF); document pre-VGB ocular baseline"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE — 6 STAGES
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Neonatal/Pre-symptomatic (0–3 months)",
        "key_action": "If family history/prenatal diagnosis → baseline EEG, neonatal MRI, CACNA2D2 molecular confirmation. Pyridoxine IV diagnostic exclusion (PDE). Feeding assessment (hypotonia → NG tube early). Genetic counselling (AR — 25% sibling recurrence)."
    },
    {
        "stage": "Infantile (3–18 months) — Peak West Syndrome / Spasm Onset",
        "key_action": "ACTH Level-A + VGB Level-A for infantile spasms. Baseline VF/ERG before VGB. POLG1 exclusion before VPA. Monthly EEG surveillance. Commence early physiotherapy + speech therapy + occupational therapy. Genetic counselling + family cascade."
    },
    {
        "stage": "Early Childhood (18 months – 5 years) — Myoclonic/Tonic Evolution",
        "key_action": "Transition to GTCS/myoclonus/tonic phenotype. VPA ± LEV ± CLB optimisation. KD after 2 AED failures. MRI annually for cerebellar atrophy progression. Avoid PHT/gabapentinoids. SUDEP education. Specialist epilepsy school placement."
    },
    {
        "stage": "School Age (5–12 years) — Drug-Resistant Epilepsy Management",
        "key_action": "KD continuation or Modified Atkins. VNS trial (Level C) if KD insufficient. Riboflavin trial (Level C). Epilepsy surgery evaluation (rarely beneficial in CACNA2D2 — diffuse network disorder). Spasticity/ataxia rehabilitation. Augmentative communication (AAC) devices. SUDEP risk review."
    },
    {
        "stage": "Adolescence (12–25 years) — Transition + VPPP",
        "key_action": "VPPP mandatory for females on VPA (MHRA 2021 — risk form + contraception). Transition to adult epilepsy services. Driving DVLA notification (seizure-related). Social care package. Sexual health education (adapted). Scoliosis surgical review if progressive. Riboflavin continuation if prior response."
    },
    {
        "stage": "Adulthood (25+ years) — Long-term Maintenance + Care Planning",
        "key_action": "Long-term AED maintenance (most CACNA2D2 patients require lifelong AED). Advance care planning + seizure emergency plan. Annual SUDEP review. Cerebellar atrophy monitoring (MRI q24–36M). Respiratory/swallowing annual review. Genetic counselling for family planning (AR 25% recurrence)."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS — 15 KEY CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {"term": "CACNA2D2 (3p21.3)", "definition": "Voltage-gated calcium channel auxiliary subunit α2-δ-2 gene, chromosome 3p21.3. Encodes a 1145-aa protein (α2 component ~130 kDa + δ component ~28 kDa, post-translationally cleaved, GPI-anchored/disulfide-linked). Biallelic LOF causes EECAT (epileptic encephalopathy with cerebellar atrophy, hypotonia, and impaired intellectual development)."},
    {"term": "α2-δ-2 Auxiliary Subunit", "definition": "One of 4 α2-δ subunit isoforms (CACNA2D1–4). α2-δ subunits are essential for channel trafficking, gating modulation, and synaptic membrane stability of Cav1 and Cav2 α1 pore subunits. α2-δ-2 is enriched in cerebellum (Purkinje cells > cortex), making cerebellar features dominant in CACNA2D2 LOF."},
    {"term": "EECAT (Epileptic Encephalopathy with Cerebellar Atrophy and Hypotonia)", "definition": "Core CACNA2D2 syndrome: severe infantile-onset epileptic encephalopathy + cerebellar ataxia + intention tremor + global hypotonia + severe intellectual disability + cerebellar atrophy/hypoplasia on MRI. No approved precision therapy as of 2026. Resembles early-onset CACNA1A DEE but distinguishable by AR inheritance and α2-δ-2 biology."},
    {"term": "Gabapentinoid Paradox in CACNA2D2", "definition": "Gabapentin and Pregabalin bind α2-δ subunits (primarily α2-δ-1/CACNA2D1 for pain; also α2-δ-2/CACNA2D2). In CACNA2D2 LOF, α2-δ-2 is already non-functional → gabapentinoids targeting this subunit have unpredictable/reduced anticonvulsant efficacy. Additionally gabapentinoids worsen cerebellar ataxia (primary side effect) which is already severe in CACNA2D2. HIGH RISK: avoid gabapentinoids in CACNA2D2."},
    {"term": "Ducky Mouse Model (CACNA2D2 null)", "definition": "Spontaneous CACNA2D2 loss-of-function mouse mutant ('ducky' phenotype: severe ataxia, hindlimb clasping on tail suspension, SWD on EEG, Purkinje cell loss). First characterised by Barclay et al. (2001 Nat Neurosci). Established CACNA2D2 as epilepsy + cerebellar ataxia gene. CACNA2D2 knockout mice die early (nutritional failure + respiratory failure from ataxia)."},
    {"term": "Cerebellar Atrophy Progression in CACNA2D2", "definition": "CACNA2D2 LOF → reduced Cav2.1/Cav2.2 Ca²⁺ entry in Purkinje cells → impaired Ca²⁺-dependent signalling + mitochondrial function → progressive Purkinje cell loss. MRI shows progressive cerebellar cortical atrophy (superior vermis first → hemispheres). Serial MRI q12–24M mandatory. PHT/CBZ compound atrophy."},
    {"term": "PHT Cerebellar Toxicity in CACNA2D2", "definition": "Phenytoin causes cerebellar toxicity (Purkinje cell loss, cerebellar atrophy) with chronic use, even at therapeutic concentrations in susceptible patients. CACNA2D2 patients have pre-existing cerebellar atrophy/Purkinje cell vulnerability → PHT COMPOUNDS pre-existing damage. PHT is HIGH RISK for maintenance use; acceptable as acute IV status epilepticus rescue only."},
    {"term": "POLG1 / Alpers-Huttenlocher Syndrome", "definition": "POLG1 (polymerase gamma) pathogenic variants cause Alpers syndrome. VPA in POLG1 carriers → fatal hepatotoxicity (liver failure within weeks). POLG1 exclusion is MANDATORY before any VPA initiation. CACNA2D2 DEE can phenotypically overlap with early Alpers (both have cerebellar features). If POLG1 testing unavailable → use LEV instead of VPA."},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021 UK mandatory programme for all females aged ≥12 years prescribed valproate. Requirements: (1) annual risk acknowledgement form signed by patient and prescriber; (2) effective contraception documented; (3) specialist review; (4) patient card carried. Identical requirements applicable across EU (EMA 2018). VPA teratogenicity: NTD 2–3%, fetal valproate syndrome, neurodevelopmental impairment in offspring."},
    {"term": "Ketogenic Diet (KD) in Ca²⁺ Channelopathies", "definition": "KD in CACNA2D2 addresses multiple pathological pathways: reduces Cav2 channel activity via acidosis; inhibits mTOR pathway; enhances GABA synthesis; anti-inflammatory in cerebellar microenvironment; activates KATP channels reducing neuronal excitability. Most effective DRE intervention in CACNA2D2 DEE. 4:1 or Modified Atkins variants. 3-month minimum trial before declaring failure."},
    {"term": "VGB VFD (Visual Field Defect — Vigabatrin)", "definition": "Irreversible visual field defects occur in 30–40% of patients on prolonged vigabatrin. Concentric VFD, nasal field loss first. ERG changes (ring-ring amplitude reduction) precede perimetric loss by months. SHARE REMS (USA): mandatory enrollment, ERG q3M. In CACNA2D2 patients with cerebellar oculomotor dysfunction (nystagmus), baseline oculomotor assessment is critical before VGB."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Leading epilepsy-related death cause. SUDEP risk in CACNA2D2 DEE is 3–7× general epilepsy population (similar to SCN1A Dravet). Highest risk: nocturnal GTCS, prone position post-ictally, subtherapeutic AED, uncontrolled DRE. Prevention: nocturnal monitoring, supine position, AED optimisation, SUDEP education to caregivers (mandatory at first appointment and annually)."},
    {"term": "West Syndrome / Infantile Spasms in CACNA2D2", "definition": "65% of CACNA2D2 patients present with infantile spasms (onset 3–18 months) with hypsarrhythmia. ACTH Level A + VGB Level A first-line. Relapse rate after ACTH is high in CACNA2D2 (vs idiopathic West where remission is more durable). EEG evolution: hypsarrhythmia → multifocal + SWD → tonic/GTCS pattern by age 3–5 years."},
    {"term": "Riboflavin (B2) in CACNA2D2", "definition": "Investigational treatment reported in a single case (Shahar 2016). Proposed mechanism: mitochondrial energy support via flavin mononucleotide/flavin adenine dinucleotide (FMN/FAD) — may partially compensate α2-δ-2 deficiency in Purkinje cell mitochondria. Not replicated in series. Consider as safe, inexpensive, non-toxic adjunct after exhausting standard options. 50–200 mg/day oral."},
    {"term": "Autosomal Recessive CACNA2D2 Inheritance", "definition": "CACNA2D2 DEE follows strict AR inheritance (pLI ~0.01; gene tolerant to heterozygous LOF — carriers healthy). Both parents are obligate carriers (25% recurrence in each subsequent pregnancy). Consanguineous families overrepresented in published series. Prenatal/preimplantation genetic testing available. De novo dominant CACNA2D2 is NOT a recognised epilepsy mechanism as of 2026."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — 12 ACTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "POLG1 result unavailable within 48h before VPA needed urgently", "action": "Use LEV instead of VPA until POLG1 result available; do NOT start VPA empirically in CACNA2D2 DEE without POLG1 clearance"},
    {"threshold": "VPA ALT/AST >3× ULN or ammonia >80 μmol/L", "action": "Hold VPA immediately; hepatology review; if POLG1 positive on retrospective testing → permanent VPA discontinuation + L-carnitine 100 mg/kg/day IV"},
    {"threshold": "EEG non-response at Day 14 of ACTH (hypsarrhythmia persists)", "action": "Extend ACTH 2 additional weeks OR switch to VGB (if not already started); add VPA/LEV; escalate to KD consideration if 2 AEDs failed"},
    {"threshold": "Febrile illness (temp ≥37.5°C) in CACNA2D2 patient", "action": "Pre-emptive paracetamol 15 mg/kg q4–6h; ensure AED doses not missed; home rescue diazepam (rectal/intranasal) available; lower threshold for hospital admission vs non-DEE children"},
    {"threshold": "ERG ring amplitude reduction ≥50% from baseline (VGB use)", "action": "Discontinue VGB; urgent ophthalmology; formal Goldmann perimetry; switch to alternative AED (VPA/CLB); VFD is irreversible — do not delay action on ERG changes"},
    {"threshold": "BHB <1.5 mmol/L on KD (target 2–4 mmol/L)", "action": "Review dietary compliance; check hidden carbohydrates (medications — liquid formulations contain sorbitol/sucrose); adjust fat:protein ratio; dietitian review"},
    {"threshold": "KD BHB >6 mmol/L + metabolic acidosis symptoms (vomiting, lethargy)", "action": "Reduce fat:protein ratio; IV fluids if tolerating orally; check glucose; suspend KD until pH >7.35; resume at lower ratio"},
    {"threshold": "Seizure-free ≥3 years (AED taper consideration)", "action": "AED taper only after MDT review; minimum 12-month taper; EEG at baseline, mid-taper, completion; CACNA2D2 high relapse rate on withdrawal — clear consent discussion"},
    {"threshold": "PHT proposed for status epilepticus rescue in CACNA2D2 patient", "action": "Use IV VPA (25 mg/kg loading) as first IV AED for status; if VPA contraindicated (POLG1) → IV LEV 60 mg/kg; IV lacosamide 200 mg; IV PHT only as last resort (cerebellar toxicity risk documented in notes)"},
    {"threshold": "Progressive scoliosis Cobb angle >20° on X-ray", "action": "Refer orthopaedic surgery; physiotherapy intensive programme; TLSO brace fitting; surgical correction if Cobb >40° (monitor respiratory function concurrently)"},
    {"threshold": "CACNA2D2 female patient aged 12y starting or already on VPA", "action": "VPPP initiation: (1) MHRA risk form signed; (2) document contraception status; (3) arrange specialist review within 4 weeks; (4) provide patient card; consider switch to LEV/LTG if conception planned"},
    {"threshold": "Gabapentin or Pregabalin prescribed to CACNA2D2 patient", "action": "Intercept prescription; discuss with prescriber; document gabapentinoid HIGH RISK status in CACNA2D2 (α2-δ-2 LOF → worsened cerebellar ataxia + unpredictable efficacy); recommend VPA/LEV/CLB alternative"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS — 12 GUIDELINES
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE Classification 2022", "applies": "Seizure type + epilepsy syndrome classification; CACNA2D2 DEE within genetic DEE spectrum; epilepsy + encephalopathy joint classification"},
    {"name": "NICE NG217 (2022)", "applies": "Epilepsy management guidelines; AED first-line recommendations; referral criteria for infantile spasms; ketogenic diet commissioning criteria"},
    {"name": "UKISS 2004 (UK Infantile Spasms Study — Mackay Lancet)", "applies": "ACTH vs vigabatrin RCT for infantile spasms; both Level A; 14-day EEG response primary endpoint; primary evidence base for IS management"},
    {"name": "ILAE Dietary Therapies Guidelines 2018 (Kossoff et al.)", "applies": "Ketogenic diet indications, implementation, monitoring; Modified Atkins diet; MCT diet; seizure type response data; micronutrient monitoring"},
    {"name": "Barclay 2001 Nat Neurosci (ducky mouse CACNA2D2)", "applies": "Foundational animal model paper establishing CACNA2D2 LOF → epilepsy + cerebellar ataxia; α2-δ-2 biology established; gabapentin binding site characterised"},
    {"name": "Pippucci 2013 Epilepsia (first human CACNA2D2 DEE)", "applies": "First human case series (consanguineous family) establishing CACNA2D2 biallelic LOF → EECAT phenotype; genotype-phenotype correlation"},
    {"name": "CPIC Guideline POLG 2023", "applies": "Pharmacogenomics — VPA contraindicated in POLG1 pathogenic variant carriers; mandatory testing protocol before VPA initiation in DEE"},
    {"name": "MHRA VPPP 2021", "applies": "UK Valproate Pregnancy Prevention Programme — mandatory for females ≥12y on VPA; risk form + effective contraception required annually"},
    {"name": "FDA SHARE REMS (Vigabatrin / Sabril)", "applies": "VGB visual field monitoring programme — ERG q3M mandatory during VGB use; US program enrollment required for VGB prescription; baseline VF before starting"},
    {"name": "ACMG-AMP Variant Interpretation 2015 (Richards et al.)", "applies": "CACNA2D2 variant classification (pathogenic/likely pathogenic/VUS); AR LOF classification rules; functional assay evidence; segregation; population frequency (gnomAD)"},
    {"name": "CPIC CBZ HLA-B*15:02 2023", "applies": "HLA-B*15:02 testing mandatory before CBZ/OXC in Asian ancestry — SJS/TEN risk; applicable if CBZ is considered (HIGH RISK already for CACNA2D2 myoclonus)"},
    {"name": "WHO ICF 2019 (International Classification of Functioning)", "applies": "Comprehensive disability framework for CACNA2D2 EECAT — epilepsy + cerebellar ataxia + severe ID + hypotonia; functional goals beyond seizure control; participation-focused care plan"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES — 6 KEY
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"author": "Barclay J et al.", "year": 2001, "journal": "Nat Neurosci 4:769–775", "title": "Ducky mouse phenotype of epilepsy and ataxia is associated with mutations in the Cacna2d2 gene and decreased calcium channel current in cerebellar Purkinje cells", "pmid": "11477425"},
    {"author": "Pippucci T et al.", "year": 2013, "journal": "Epilepsia 54:1569–1573", "title": "Homozygous CACNA2D2 loss-of-function mutations in a patient with epileptic encephalopathy and cerebellar atrophy", "pmid": "23909637"},
    {"author": "Vergult S et al.", "year": 2015, "journal": "Epilepsy Res 115:115–118", "title": "CACNA2D2 mutations: a new gene for developmental delay, intellectual disability, and epilepsy", "pmid": "26302543"},
    {"author": "Shahar E et al.", "year": 2016, "journal": "Brain Dev 38:693–697", "title": "Riboflavin treatment in a child with severe epileptic encephalopathy associated with a CACNA2D2 mutation — a case report", "pmid": "26781843"},
    {"author": "Doohan PT et al.", "year": 2019, "journal": "Front Pharmacol 10:1461", "title": "Gene variants associated with psychiatric disorders: a review of CACNA2D2 and CACNA2D3 alpha2-delta calcium channel subunits", "pmid": "31866860"},
    {"author": "Lenk GM et al.", "year": 2020, "journal": "Elife 9:e54451", "title": "De novo mutations in CACNA2D2 cause epileptic encephalopathy with cerebellar atrophy and profound hypotonia", "pmid": "32510328"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC COHORT — 40 PATIENTS
# ─────────────────────────────────────────────────────────────────────────────
_etiology_pool = (
    ["Biallelic-LOF-Truncating-Severe-EECAT"] * 17 +
    ["Biallelic-LOF-Missense-Moderate-EECAT"] * 11 +
    ["Biallelic-LOF-CNV-Deletion-3p21"] * 6 +
    ["Compound-Heterozygous-LOF-Intermediate"] * 4 +
    ["Phenocopy-Panel-Negative"] * 2
)

_aed_pool = [
    ["VPA", "LEV"], ["ACTH", "VGB", "VPA"], ["ACTH", "LEV", "CLB"],
    ["VPA", "CLB"], ["LEV", "CLB", "KD"], ["ACTH", "VPA", "CLB"],
    ["VPA", "KD"], ["LEV", "KD"], ["ACTH", "LEV"],
    ["VPA", "LEV", "CLB", "KD"], ["CLB", "LEV"], ["ACTH", "CLB", "VPA"],
    ["LEV", "VPA", "CLB"], ["KD", "VPA"], ["ACTH", "VPA"],
]

_outcomes = ["seizure-free", "≥50%-reduction", "partial-response", "DRE"]
_outcome_wts = [0.08, 0.28, 0.34, 0.30]

_cohort = []
for i in range(40):
    et = _etiology_pool[i]
    is_truncating = "Truncating" in et
    is_cnv = "CNV" in et
    is_phenocopy = "Phenocopy" in et
    age = random.randint(2, 22)
    onset_months = random.randint(2, 18)
    aeds = _aed_pool[i % len(_aed_pool)]
    outcome = random.choices(_outcomes, weights=_outcome_wts)[0]
    has_is = random.random() < 0.65
    has_ataxia = not is_phenocopy
    has_cerebellar_atrophy = not is_phenocopy and random.random() < 0.85
    has_hypotonia = not is_phenocopy and random.random() < 0.92
    kd_use = "KD" in aeds
    acth_use = "ACTH" in aeds
    _cohort.append({
        "patient_id": f"P{i+1:03d}",
        "age_years": age,
        "onset_months": onset_months,
        "etiology": et,
        "variant_type": "truncating" if is_truncating else ("CNV-deletion" if is_cnv else ("compound-het" if "Compound" in et else ("missense" if "Missense" in et else "unknown"))),
        "seizure_free": outcome == "seizure-free",
        "dre": outcome == "DRE",
        "outcome": outcome,
        "aeds": aeds,
        "infantile_spasms_hx": has_is,
        "cerebellar_ataxia": has_ataxia,
        "cerebellar_atrophy_mri": has_cerebellar_atrophy,
        "hypotonia": has_hypotonia,
        "kd_use": kd_use,
        "acth_use": acth_use,
        "gabapentinoid_exposure": random.random() < 0.18,
        "pht_exposure": random.random() < 0.12,
        "myoclonus": random.random() < 0.72,
        "gtcs": random.random() < 0.80,
        "tonic": random.random() < 0.55,
    })


# ─────────────────────────────────────────────────────────────────────────────
# API RESPONSE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(_cohort)
    seizure_free = sum(1 for p in _cohort if p["seizure_free"])
    dre = sum(1 for p in _cohort if p["dre"])
    infantile_spasms = sum(1 for p in _cohort if p["infantile_spasms_hx"])
    ataxia = sum(1 for p in _cohort if p["cerebellar_ataxia"])
    atrophy = sum(1 for p in _cohort if p["cerebellar_atrophy_mri"])
    hypotonia = sum(1 for p in _cohort if p["hypotonia"])
    kd_use = sum(1 for p in _cohort if p["kd_use"])
    gabapentinoid_exp = sum(1 for p in _cohort if p["gabapentinoid_exposure"])
    myoclonus = sum(1 for p in _cohort if p["myoclonus"])

    etiology_dist = {}
    for et_spec in ETIOLOGY_CATALOG:
        count = sum(1 for p in _cohort if p["etiology"] == et_spec["category"])
        etiology_dist[et_spec["category"]] = count

    seizure_summary = [
        {"type": s["type"], "frequency_pct": s["frequency_pct"]}
        for s in SEIZURE_TYPES
    ]
    treatment_summary = [
        {"drug": t["drug"].split(" —")[0].split(" (")[0][:45], "level": t["level"].split(" (")[0]}
        for t in TREATMENTS
    ]
    monitoring_summary = [
        {"item": m["item"].split(" (")[0][:55], "frequency": m["frequency"].split(";")[0]}
        for m in MONITORING[:8]
    ]
    lifecycle_summary = [
        {"stage": lc["stage"][:60], "key_action": lc["key_action"][:90]}
        for lc in LIFECYCLE
    ]

    return {
        "gene": "CACNA2D2",
        "locus": "3p21.3",
        "protein": "α2-δ-2 (alpha2-delta-2) — Voltage-Gated Ca²⁺ Channel Auxiliary Subunit",
        "channel_role": "Trafficking/modulation of Cav1 + Cav2 α1 pore subunits; GPI-anchored extracellular; thrombospondin synaptogenesis partner",
        "syndrome": "EECAT (Epileptic Encephalopathy with Cerebellar Atrophy, Hypotonia, and Impaired Intellectual Development)",
        "gene_omim": "*602843 CACNA2D2 gene",
        "inheritance": "Autosomal Recessive (AR) biallelic LOF; carriers healthy; 25% sibling recurrence",
        "color": "#1b5e20",
        "precision_tx": "None approved. KD (best DRE option). AVOID gabapentinoids + PHT (cerebellar toxicity).",
        "total_patients": n,
        "seizure_free_pct": round(seizure_free / n * 100, 1),
        "dre_pct": round(dre / n * 100, 1),
        "infantile_spasms_count": infantile_spasms,
        "cerebellar_ataxia_count": ataxia,
        "cerebellar_atrophy_mri_count": atrophy,
        "hypotonia_count": hypotonia,
        "kd_use_count": kd_use,
        "gabapentinoid_exposure_count": gabapentinoid_exp,
        "myoclonus_count": myoclonus,
        "etiology_distribution": etiology_dist,
        "seizure_summary": seizure_summary,
        "treatments_summary": treatment_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": lifecycle_summary,
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [
            {"drug": ci_["drug"].split(" —")[0].split(" (")[0][:50], "risk": ci_["risk"]}
            for ci_ in CONTRAINDICATIONS[:5]
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
            "gene": "CACNA2D2",
            "full_name": "Calcium Voltage-Gated Channel Auxiliary Subunit Alpha2 Delta-2",
            "chromosome": "3p21.3",
            "protein": "α2-δ-2 (alpha2-delta-2) auxiliary subunit of voltage-gated calcium channels",
            "size": "1145 aa total (α2 ~130 kDa + δ ~28 kDa, disulfide-linked post-translational processing)",
            "function": "Promotes Cav1/Cav2 α1 subunit forward trafficking (ER→Golgi→plasma membrane); modulates channel kinetics; supports active zone organisation; thrombospondin receptor (synapse formation via TSP1/TSP2)",
            "expression": "Cerebellum DOMINANT (Purkinje cells > granule cells); cortex; hippocampus; presynaptic glutamatergic + GABAergic terminals",
            "animal_model": "Ducky mouse (CACNA2D2 null): spontaneous ataxia + spike-wave discharges + Purkinje cell loss (Barclay 2001 Nat Neurosci) — established α2-δ-2 epilepsy biology",
            "inheritance": "AR biallelic LOF; pLI ~0.01 (gene tolerates heterozygous LOF in carriers); 25% recurrence per pregnancy; consanguineous families overrepresented",
            "gene_omim": "OMIM *602843 CACNA2D2 gene",
            "syndrome": "EECAT — Epileptic Encephalopathy with Cerebellar Atrophy, Hypotonia, and Impaired Intellectual Development",
            "key_distinction": "α2-δ-2 is GABAPENTINOID BINDING SITE — LOF already impairs α2-δ-2; gabapentinoids worsen cerebellar ataxia; avoid in CACNA2D2. PHT is HIGH RISK (cerebellar toxicity compounds pre-existing atrophy). No precision blocker available as of 2026 — symptomatic treatment + KD is best DRE approach.",
            "gabapentinoid_paradox": "Gabapentin/Pregabalin bind α2-δ subunits. In CACNA2D2 LOF, α2-δ-2 is non-functional → reduced gabapentinoid efficacy at this site + worsened cerebellar ataxia (primary side effect). HIGH RISK: AVOID gabapentinoids in CACNA2D2.",
            "cacna2_subfamily": "α2-δ-1/CACNA2D1 (primary analgesic gabapentinoid target, neuropathic pain) · α2-δ-2/CACNA2D2 (cerebellar/epilepsy, EECAT) · α2-δ-3/CACNA2D3 (pain/anxiety) · α2-δ-4/CACNA2D4 (retinal photoreceptors)",
            "absolute_ci": "TGB (NCSE) · VPA+POLG1 (Alpers-fatal hepatotoxicity) · Gabapentinoids (cerebellar toxicity + α2-δ-2 LOF paradox) · PHT maintenance (Purkinje cell loss compound) · VGB long-term (VFD irreversible)",
        },
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
