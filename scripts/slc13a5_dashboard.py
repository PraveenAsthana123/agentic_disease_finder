"""
SLC13A5 Epilepsy — Citrate Transporter Deficiency / NAFE / EIEE25
===================================================================
40-patient cohort · SLC13A5 (17p13.1) · NaCT sodium-coupled citrate transporter · AR

SLC13A5 BIOLOGY:
SLC13A5 (17p13.1) encodes NaCT (Sodium-Coupled Citrate Transporter, SLC13 family member 5),
a plasma-membrane Na⁺/citrate co-transporter expressed at highest density at:
  1. BLOOD-BRAIN BARRIER (BBB) — choroid plexus and brain microvascular endothelial cells:
     imports circulating citrate (plasma: ~100 µM) across the BBB into neuronal interstitium.
  2. NEURONS (cortical, hippocampal) — citrate uptake fuels TCA cycle as carbon scaffold
     for glutamate and GABA biosynthesis.
  3. LIVER (high expression) — hepatic citrate transport for lipogenesis and gluconeogenesis.

NaCT TRANSPORT STOICHIOMETRY:
  4 Na⁺ : 1 citrate³⁻ (electrogenic; strongly driven by Na⁺ gradient).
  Km for citrate: ~0.6 mM (liver), ~0.1 mM (brain — high affinity, low flux).
  Inhibited by: lithium (competitive), acetyl-CoA-related citrate analogs.

CITRATE NEUROBIOLOGY (WHY CITRATE DEFICIENCY CAUSES SEIZURES):
  1. TCA CYCLE ANAPLEROSIS: Citrate → acetyl-CoA + OAA (citrate lyase) →
     fuels glutamate synthesis (α-ketoglutarate) → GABA synthesis (GAD1/2).
     SLC13A5 LOF → reduced brain citrate → impaired TCA anaplerosis →
     reduced GABA → cortical hyperexcitability → seizures.
  2. GLYCOLYSIS INHIBITION: Extracellular citrate inhibits phosphofructokinase-1 (PFK1)
     → allosteric brake on glycolytic flux. NaCT LOF → reduced intracellular citrate →
     loss of PFK1 brake → uncontrolled glycolysis → lactate accumulation → neuronal
     pH dysregulation → altered excitability.
  3. ZINC CHELATION: Citrate chelates Zn²⁺ in synaptic vesicles (Zn²⁺ modulates NMDA
     and GABA-A receptors). SLC13A5 LOF → disrupted Zn²⁺-citrate homeostasis.
  4. TOOTH ENAMEL: Odontoblasts and ameloblasts use NaCT for citrate-dependent
     enamel mineralisation (citrate-coated hydroxyapatite crystal stabilization).
     SLC13A5 LOF → DENTAL ENAMEL HYPOPLASIA/HYPOMINERALISATION — PATHOGNOMONIC feature.

CLINICAL PHENOTYPE:
  Neonatal onset (Day 1-5 of life in 70%) seizures. Initially focal (clonic) or
  multifocal; may evolve to focal with automatisms. Tonic seizures in neonatal period.
  Developmental delay / intellectual disability (mild-severe). DENTAL ENAMEL DEFECTS
  (hypoplastic / hypomineralised teeth) are a pathognomonic clinical clue.
  Biomarker: URINE CITRATE ELEVATED (reduced BBB citrate uptake → accumulation in plasma
  → renal overflow). CSF citrate may be low. No plasma amino acid abnormality.

INHERITANCE AND GENETICS:
  AUTOSOMAL RECESSIVE — biallelic LOF variants required. Both parents obligate carriers
  (recurrence risk 25%). De novo variants very rare. OMIM: #615905 (EIEE25).
  Consanguinity enriched in some cohorts. Locus: 17p13.1. Gene: SLC13A5 (9 exons).
  pLI ~0.01 (tolerant); missense z-score moderate. Truncating + missense compound het frequent.
  Incidence: <500 published patients worldwide; likely underdiagnosed (neonatal panel expansion).

CONTRAINDICATIONS IN SLC13A5 DEFICIENCY:
  1. STANDARD KETOGENIC DIET 4:1 (HIGH CAUTION — may worsen citrate deficit):
     Classical KD promotes fatty acid oxidation → citrate enters TCA via beta-oxidation,
     BUT reduces glucose-derived TCA anaplerosis. In SLC13A5, TCA citrate import is already
     deficient. Standard high-fat KD without supplemental citrate sources may deepen
     neuronal citrate deficit → worsening metabolic encephalopathy. Modified KD with
     triheptanoin (C7 anaplerotic) or citrate supplementation may be safer.
  2. VALPROATE (HIGH — POLG risk + metabolic concerns):
     VPA is a citrate transporter inhibitor at high doses. Also: POLG screen mandatory
     before VPA. In metabolic epilepsies (mitochondrial-adjacent): VPA can precipitate
     hepatic failure. VPA is not standard first-line in neonatal SLC13A5 epilepsy.
  3. TIAGABINE (ABSOLUTE CI — NCSE risk):
     GAT-1 inhibitor → prolonged synaptic GABA → paradoxical NCSE in neonates with
     diffuse cortical involvement. ABSOLUTE CI in EIEE25.
  4. PHENYTOIN — neonatal cardiac effects (HIGH caution):
     IV phenytoin/fosphenytoin → cardiac arrhythmia in neonates; also may mask EEG
     seizures without clinical correlate; parenteral administration only.
  5. VIGABATRIN without ERG monitoring (HIGH — irreversible VFD):
     SHARE REMS; VGB causes irreversible concentric visual field defect (30-40% with
     prolonged use); baseline ERG mandatory; q3M monitoring.
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "SLC13A5-biallelic-LOF-classic-neonatal",
        "pct": 50,
        "etiology": "Biallelic LOF — classic neonatal epilepsy (EIEE25) + dental enamel defects",
        "mechanism": (
            "Compound heterozygous or homozygous truncating/splice variants → complete NaCT "
            "loss → no BBB citrate import → severe neuronal citrate depletion → impaired GABA "
            "synthesis via TCA/glutamate pathway → neonatal seizures Day 1-5. Dental: NaCT-null "
            "ameloblasts cannot mineralise enamel hydroxyapatite → hypoplastic teeth (permanent "
            "dentition more severely affected). Drug-resistant in 60-70%."
        ),
        "typical_variants": "Truncating (frameshift, nonsense, canonical splice-site); compound het (LOF+LOF); consanguineous homozygous",
        "eeg_signature": "Multifocal sharp waves (neonatal); focal clonic ictal rhythm; burst-suppression in severe cases; interictal multifocal discharges",
        "phenotype": "EIEE25: neonatal clonic/multifocal seizures; drug-resistant 60-70%; ID moderate-severe; hypoplastic teeth; elevated urine citrate",
    },
    {
        "category": "SLC13A5-biallelic-missense-attenuated",
        "pct": 25,
        "etiology": "Biallelic missense — attenuated phenotype (partial NaCT activity, later onset, milder ID)",
        "mechanism": (
            "Two missense variants → partial residual NaCT transport activity (10-30% of WT) → "
            "incomplete citrate depletion → milder phenotype. Seizure onset may be delayed to "
            "weeks-months. EEG: focal rather than multifocal. Dental defects still present "
            "(enamel mineralisation sensitive to even 50% NaCT reduction). Febrile seizures "
            "may be presenting event. IQ: borderline to mild ID."
        ),
        "typical_variants": "Missense (N344S, G219R, R212C-like); functional studies show partial transport; homozygous recessive in consanguineous families",
        "eeg_signature": "Focal temporal/frontal sharp waves; interictal: multifocal, may normalise with age; HV not activating",
        "phenotype": "Attenuated EIEE25: late neonatal/early infantile onset; fewer seizures; mild-moderate ID; prominent dental defects; elevated urine citrate",
    },
    {
        "category": "SLC13A5-biallelic-splice-variable",
        "pct": 15,
        "etiology": "Biallelic splice-site variants — variable severity (NMD-dependent, leaky splicing)",
        "mechanism": (
            "Deep intronic or canonical splice-site variants → variable NMD efficiency and "
            "residual correct splicing → clinical severity correlates with residual NaCT. "
            "Leaky splice variants may produce 15-40% WT NaCT → intermediate phenotype. "
            "Urine citrate: elevated but less extreme than class 1. RNA studies needed "
            "for variant interpretation (mRNA leakage quantification)."
        ),
        "typical_variants": "Canonical donor/acceptor splice (c.1244+1G>A-like); deep intronic (r. studies needed); NMD-sensitive",
        "eeg_signature": "Variable: focal clonic (neonatal) to multifocal; may improve with age",
        "phenotype": "Variable EIEE25: range from severe neonatal to milder infantile epilepsy; splice-leakage correlates with phenotype severity",
    },
    {
        "category": "SLC13A5-compound-het-mixed-class",
        "pct": 7,
        "etiology": "Compound het (LOF + missense) — intermediate severity",
        "mechanism": (
            "One allele: truncating LOF (complete loss). Second allele: hypomorphic missense "
            "(partial function ~20-30% WT). Net NaCT activity ~10-15% → intermediate phenotype. "
            "Seizure onset: Day 2-10. Dental defects milder than class 1 but present. "
            "Drug-resistant in 40-50%. Response to triheptanoin may be better than severe LOF "
            "(residual TCA capacity to utilize anaplerotic substrate)."
        ),
        "typical_variants": "One truncating + one missense; both parents are obligate carriers of different classes",
        "eeg_signature": "Focal-onset seizures with secondary generalisation; multifocal in neonates; improves by age 2-3y",
        "phenotype": "Intermediate EIEE25: neonatal onset, moderate drug-resistance, mild-moderate ID, dental defects present",
    },
    {
        "category": "SLC13A5-negative-phenocopy-EIEE",
        "pct": 3,
        "etiology": "SLC13A5-negative phenocopy (neonatal epilepsy + ID without SLC13A5 variant)",
        "mechanism": (
            "Clinical neonatal epilepsy + developmental delay without SLC13A5 biallelic variant. "
            "Differential: KCNQ2 (most common cause of neonatal-onset DEE — check KCNQ2/3 "
            "first), STXBP1, CDKL5, SCN2A (neonatal), pyridoxine-dependent epilepsy (ALDH7A1), "
            "glucose transporter deficiency (SLC2A1/GLUT1), methylmalonic acidemia, propionic "
            "acidemia. Dental defects absent. Urine citrate normal."
        ),
        "typical_variants": "None — panel-based diagnosis; exclude SLC13A5 and investigate KCNQ2/STXBP1/CDKL5",
        "eeg_signature": "Variable — depends on actual aetiology (KCNQ2: theta-dominated burst-suppression; STXBP1: hypsarrhythmia-like)",
        "phenotype": "Neonatal epilepsy without elevated urine citrate or dental enamel defects — reassign after comprehensive panel",
    },
]

# ── Seizure Types ──────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Neonatal Clonic / Multifocal Clonic Seizures",
        "prevalence_pct": 90,
        "semiology": (
            "Focal or multifocal clonic activity (rhythmic limb jerking, face involvement) "
            "occurring Day 1-5 of life. Duration 30-120 seconds. May be subclinical (EEG-only) "
            "in 20-30% — continuous EEG monitoring mandatory in NICU. Clustering occurs. "
            "In severe biallelic LOF: seizure burden >10 events/day without treatment. "
            "Electroclinical dissociation common after phenobarbital loading (PB suppresses "
            "clinical signs but not EEG discharges — interpret continuous aEEG carefully)."
        ),
        "eeg_pattern": (
            "Ictal: rhythmic focal sharp waves → evolving frequency (8-12 Hz clonic); "
            "may spread to adjacent electrode montage. Background: multifocal sharp waves, "
            "discontinuous (burst-suppression in severe) or mildly abnormal. "
            "aEEG: seizure pattern bursts; background voltage suppressed in severe cases."
        ),
        "clinical_tip": (
            "FIRST-LINE NEONATAL: Phenobarbital 20 mg/kg IV loading dose. If refractory: "
            "second PB dose 10 mg/kg or add LEV 40-60 mg/kg IV. PYRIDOXINE TRIAL (100 mg IV) "
            "mandatory for all neonatal seizures (rule out ALDH7A1 pyridoxine-dependent "
            "epilepsy). urine organic acids + citrate URGENTLY. Check urine citrate to "
            "screen for SLC13A5 (elevated = strong signal). Continuous aEEG mandatory — "
            "do not rely on clinical observation alone."
        ),
    },
    {
        "type": "Focal Seizures with Automatisms (Post-neonatal)",
        "prevalence_pct": 70,
        "semiology": (
            "Evolving beyond neonatal period to focal seizures with oral/manual automatisms, "
            "behavioural arrest, head deviation. Temporal lobe semiology predominates. "
            "Duration 30-90 seconds. Post-ictal confusion variable. Seizures may cluster "
            "during intercurrent illness (fever lowers threshold — NOT classic febrile seizures; "
            "represent fever-provoked focal seizures from pre-existing epileptic network). "
            "Frequency: 1-10/month in partially controlled patients."
        ),
        "eeg_pattern": (
            "Interictal: multifocal or bitemporal independent sharp waves; background: "
            "generalised slowing proportional to encephalopathy degree. "
            "Ictal: temporal rhythmic theta → evolving focal discharge."
        ),
        "clinical_tip": (
            "LEV preferred post-neonatal focal seizures (renal-adjusted, safe, IV available). "
            "OXC may be used for focal seizures (unlike in GGE — SLC13A5 is a focal/multifocal "
            "epilepsy, NOT GGE; NaV blockers are NOT contraindicated for the generalisation "
            "reason used in GGE). Consider Triheptanoin as adjunct if standard AEDs fail — "
            "anaplerotic C7 bypasses NaCT-dependent citrate import."
        ),
    },
    {
        "type": "Fever-Provoked Focal Seizures (NOT classical febrile seizures)",
        "prevalence_pct": 55,
        "semiology": (
            "Focal seizures triggered by any intercurrent febrile illness. Temperature "
            "threshold lower than classic febrile seizures (≥37.8°C in SLC13A5 vs ≥38°C "
            "for GEFS+). These are NOT benign febrile seizures — they reflect fever-lowering "
            "of focal seizure threshold in an established epileptic brain. FSE (>30 min) "
            "risk elevated (~15%). Distinguish from: GEFS+ (which has GGE EEG background), "
            "Dravet (SCN1A, hemiclonic, worse response to PB)."
        ),
        "eeg_pattern": (
            "Ictal: focal temporal/frontotemporal; may secondarily generalise. "
            "Interictal: unchanged from baseline multifocal discharges."
        ),
        "clinical_tip": (
            "Fever action plan MANDATORY: paracetamol 15 mg/kg at ≥37.5°C; midazolam "
            "0.2-0.3 mg/kg buccal/IN rescue at seizure onset (earlier than in classic FS). "
            "Inform parents/carers: fever-provoked seizures in SLC13A5 ≠ benign febrile "
            "seizures — each event carries status risk. Document in medical alert card."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "prevalence_pct": 25,
        "semiology": (
            "Brief myoclonic jerks; may be cortical (erratic myoclonus on EEG) or "
            "subcortical. Less prominent than in SCN1A-Dravet. More common in severe "
            "biallelic LOF phenotype. Erratic myoclonus may correlate with encephalopathic "
            "phase. Distinguish from non-epileptic neonatal jitteriness / startle."
        ),
        "eeg_pattern": (
            "Generalised or multifocal spike/polyspike (cortical myoclonus). "
            "EEG-EMG polygraphic study needed to confirm cortical myoclonus vs. "
            "subcortical or non-epileptic myoclonus."
        ),
        "clinical_tip": (
            "LEV effective for myoclonic component (SV2A modulation). VPA has some "
            "antimyoclonic activity but HIGH CAUTION in metabolic-adjacent epilepsies "
            "(POLG mandatory; citrate-transport inhibition at high VPA doses). "
            "Clobazam adjunct for myoclonic clusters."
        ),
    },
    {
        "type": "Neonatal Status Epilepticus",
        "prevalence_pct": 20,
        "semiology": (
            "Electrographic status epilepticus (ESE) in the neonatal period — >50% of "
            "1-hour epoch with ictal activity on EEG. May be subtle or subclinical. "
            "Risk factors: severe biallelic LOF, delayed diagnosis, inadequate PB dosing. "
            "Mortality risk with refractory neonatal SE in EIEE25: ~5-10% in published series. "
            "Survivors: higher probability of severe ID and drug-resistant epilepsy."
        ),
        "eeg_pattern": (
            "Continuous or near-continuous ictal activity; multifocal; between seizures: "
            "burst-suppression or discontinuous background indicating severe encephalopathy."
        ),
        "clinical_tip": (
            "NEONATAL SE PROTOCOL: PB 20 mg/kg IV → LEV 40-60 mg/kg IV → "
            "midazolam infusion 0.1-0.4 mg/kg/h → pyridoxine 100 mg IV (rule-out PDE). "
            "Avoid high-dose IV phenytoin (cardiac effects in neonates). "
            "Consider early Triheptanoin enterally if IV access established and metabolic "
            "team available — may provide emergency citrate-independent TCA support."
        ),
    },
]

# ── Triggers ───────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Missed AED dose", "pct": 78, "note": "Primary preventable trigger; PB/LEV adherence critical; single missed dose may precipitate cluster in drug-resistant cases"},
    {"trigger": "Fever / intercurrent illness", "pct": 70, "note": "SLC13A5-specific: lower temperature threshold (≥37.5°C) than classic FS; early rescue mandatory; fever plan documented"},
    {"trigger": "Sleep deprivation / fatigue", "pct": 52, "note": "Common in older children; impairs seizure threshold; sleep hygiene program from age 3+"},
    {"trigger": "Fasting / prolonged hypoglycaemia", "pct": 45, "note": "Citrate-dependent TCA most stressed during fasting; avoid prolonged fasting; pre-op glucose protocol mandatory"},
    {"trigger": "Vaccinations (post-vaccination fever)", "pct": 38, "note": "Post-vaccination fever → fever-provoked seizure risk; pre-medicate paracetamol before vaccine + rescue BZD plan"},
    {"trigger": "High-fat diet (uncontrolled)", "pct": 30, "note": "Classical KD increases fatty acid beta-oxidation but reduces glucose-citrate flux; may deepen neuronal citrate deficit without anaplerotic supplementation"},
    {"trigger": "Emotional / physical stress", "pct": 28, "note": "Cortisol → altered glucose metabolism → citrate flux changes; stress management program beneficial"},
    {"trigger": "Hypoglycaemia / metabolic dysregulation", "pct": 22, "note": "Glucose-citrate inter-dependence: hypoglycaemia → reduced TCA substrate → increased seizure risk; emergency dextrose protocol"},
]

# ── Treatments ─────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Phenobarbital (PB)",
        "level": "Level A — neonatal first-line",
        "moa": "GABA-A positive allosteric modulator (prolongs Cl⁻ channel open time at β subunit); voltage-gated Na+ and Ca2+ channel inhibition at high doses",
        "dose": "Loading: 20 mg/kg IV (additional 10 mg/kg x2 if needed); maintenance 3-5 mg/kg/day in 1-2 doses",
        "efficacy": "Neonatal seizures: 43-45% response (PHENOBARB trial); best first-line data for neonatal onset; combination with LEV improves to ~60%",
        "safety": "Sedation; respiratory depression (resuscitation ready at loading); cognitive effects with prolonged use (assess at 12M, 24M); enzyme inducer (CYP2C9/3A4)",
        "monitoring": "PB TDM 15-40 mg/L (neonatal); respiratory monitoring at loading; carnitine q6M (long-term PB → L-carnitine depletion); consider tapering after 2y seizure-free",
        "slc13a5_note": "Standard first-line for EIEE25 neonatal seizures — initiate while awaiting metabolic workup. Does not address underlying citrate deficit. Long-term: reassess benefit-risk at each year; cognitive side effects are a concern in a child with pre-existing ID.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation → reduces vesicular neurotransmitter release; also modulates GABA-A, glycine, AMPA receptor sensitivity",
        "dose": "Neonatal: 40-60 mg/kg/day IV in 2 doses; maintenance 20-60 mg/kg/day; IV available (1:1 oral:IV)",
        "efficacy": "Neonatal adjunct: 30-40% additional response when added to PB; focal post-neonatal: 50-60% responder rate",
        "safety": "Irritability (15-20%, particularly in DEE); somnolence; headache. No significant drug-drug interactions. Renal elimination",
        "monitoring": "LEV TDM 12-46 mg/L (optional); renal function (GFR adjusted); developmental/behavioural assessment q6M (irritability monitoring); IV to oral conversion 1:1",
        "slc13a5_note": "Preferred second-line after PB in neonates; first-line post-neonatal focal seizures. SV2A mechanism is INDEPENDENT of citrate/GABA pathway — maintains efficacy despite citrate deficit. Excellent safety profile in neonates (no hepatotoxic risk unlike VPA).",
    },
    {
        "drug": "Triheptanoin (C7 fatty acid — anaplerotic precision therapy)",
        "level": "Level B — precision metabolic therapy (SLC13A5-specific)",
        "moa": (
            "C7 (heptanoic acid, odd-chain fatty acid) → β-oxidation → propionyl-CoA + "
            "acetyl-CoA → propionyl-CoA → succinyl-CoA via propionate pathway → directly "
            "replenishes TCA cycle intermediates (ANAPLEROSIS) independent of NaCT-mediated "
            "citrate import. Restores neuronal TCA flux; normalises GABA synthesis via "
            "α-ketoglutarate/glutamate pathway; reduces cortical hyperexcitability."
        ),
        "dose": "1-2 g/kg/day (target ≤35% of daily caloric intake); oral liquid; with food to reduce GI effects; available via expanded access/compassionate use",
        "efficacy": "Preclinical (slc13a5⁻/⁻ zebrafish): reduced seizure frequency 60-75%; early human case series: 40-60% seizure reduction; ongoing clinical trial (TRIC-SLC13A5)",
        "safety": "GI: nausea, vomiting, diarrhoea (start low, titrate over 2-4 weeks); mild elevated creatine kinase; no hepatotoxicity; no cardiac effects; no drug interactions",
        "monitoring": "Seizure diary (quantify response); acylcarnitine profile (C4DC/C5OH biomarkers of C7 metabolism); LFT at baseline; weight and growth; GI symptom log",
        "slc13a5_note": "PRECISION THERAPY for SLC13A5 — directly addresses the metabolic deficit by bypassing NaCT-dependent citrate import. Only AED that targets the ROOT CAUSE (TCA anaplerosis). Genotype confirmation (biallelic SLC13A5) required before initiation. Access via metabolic neurology centre. May be combined with LEV (complementary mechanisms).",
    },
    {
        "drug": "Oxcarbazepine (OXC) / Carbamazepine (CBZ) — focal epilepsy",
        "level": "Level B (focal seizures; NOT contraindicated unlike in GGE)",
        "moa": "Na+ channel state-dependent block (Nav1.1, Nav1.2, Nav1.6); reduces high-frequency firing; OXC: active MHD metabolite less enzyme-inducing than CBZ",
        "dose": "OXC: 10-30 mg/kg/day in 2 doses (preferred over CBZ — less enzyme induction). CBZ: 10-20 mg/kg/day",
        "efficacy": "Focal seizures with automatisms: 40-50% responder; may reduce focal onset before secondary generalisation",
        "safety": "Hyponatraemia (OXC > CBZ); rash (SJS/TEN risk — HLA-B*15:02 screen if SE Asian ancestry); dizziness; enzyme inducer (CBZ — reduces co-medication levels)",
        "monitoring": "Sodium (baseline, q3M with OXC); HLA-B*15:02 before prescribing; HLA-A*31:01; OXC TDM (MHD 12-35 mg/L); LFT; CBC",
        "slc13a5_note": "IMPORTANT DISTINCTION: OXC/CBZ are NOT contraindicated in SLC13A5 epilepsy (unlike in GGE/GABRA2/SCN1A-Dravet where they cause GGE aggravation). SLC13A5 is a FOCAL epilepsy — NaV blockade is appropriate for focal seizure control. Preferred in children with focal temporal predominance when LEV is insufficient.",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level B (focal — SHARE REMS; ophthalmology mandatory)",
        "moa": "Irreversible GABA-T inhibitor → raises synaptic GABA → stabilises focal epileptic network; approved for infantile spasms and focal epilepsy",
        "dose": "50-150 mg/kg/day in 2 doses; titrate over 2-4 weeks",
        "efficacy": "Focal epilepsy: 40-50% ≥50% seizure reduction; useful in early infantile period when focal seizures predominate",
        "safety": "IRREVERSIBLE CONCENTRIC VISUAL FIELD DEFECT (VFD) in 30-40% with prolonged use (>6 months, >6 g total) — the most significant risk. Sedation; weight gain; MRI signal changes (BG/thalamus) in infants (usually transient)",
        "monitoring": "ERG baseline and q3M (SHARE REMS mandatory); VFD testing (Goldmann perimetry from age 5); ophthalmology q3M; discontinue if VFD develops",
        "slc13a5_note": "VGB is a viable option for SLC13A5 focal epilepsy in infancy when LEV insufficient. Not recommended long-term (>12-18 months) due to irreversible VFD risk. Consider as bridge therapy in early infantile period. Cease before age 2-3y and reassess with alternative AEDs. SHARE REMS enrolment mandatory.",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B (adjunct)",
        "moa": "1,5-benzodiazepine; GABA-A PAM with moderate α2/α3 selectivity; longer t½ than 1,4-BZDs (clonazepam); less sedating than lorazepam",
        "dose": "0.1-0.3 mg/kg/day in 2 doses; start low (0.1 mg/kg) and titrate; N-CLB active metabolite (CYP2C19)",
        "efficacy": "Adjunct: 30-40% ≥50% seizure reduction; useful for seizure clusters and rescue; catamenial adjunct post-menarche",
        "safety": "Sedation; tolerance with continuous use (>3-4 months); withdrawal risk on abrupt discontinuation; N-CLB accumulates in CYP2C19 poor metabolisers",
        "monitoring": "CYP2C19 genotype (PM: higher N-CLB, lower CLB — adjust dose); avoid prolonged continuous use; CLB TDM 0.1-0.4 mg/L",
        "slc13a5_note": "Useful adjunct in SLC13A5 for seizure clusters and fever-provoked clustering. Consider as intermittent rescue escalation (3-5 days peri-febrile illness) to reduce hospitalisation for SE. Long-term continuous use limited by tolerance.",
    },
    {
        "drug": "Modified Ketogenic Diet (with Triheptanoin or citrate supplementation)",
        "level": "Level C — caution; only modified form",
        "moa": "Anaplerotic modified KD: Triheptanoin provides C7 → TCA propionyl-CoA anaplerosis independent of citrate transport. β-OHB raises seizure threshold (KATP channel, GABAA modulation). Avoids classical 4:1 KD citrate depletion risk",
        "dose": "Modified KD 2.5:1 or 3:1 ratio + Triheptanoin 1-2 g/kg/day; initiated by metabolic dietitian + neurologist together",
        "efficacy": "Case series SLC13A5: modified KD + Triheptanoin → 40-60% seizure reduction; better than classical KD alone (which may worsen citrate deficit)",
        "safety": "Dyslipidaemia; constipation; growth monitoring; kidney stone risk (citrate supplement may actually reduce stone risk in SLC13A5); metabolic monitoring essential",
        "monitoring": "Acylcarnitine profile; urine organic acids; citrate levels (urine + blood); lipid panel q3M; DXA annually; dietitian q4W; ketone monitoring (urinary/blood)",
        "slc13a5_note": "CLASSICAL 4:1 KD WITHOUT TRIHEPTANOIN: HIGH CAUTION — may worsen neuronal citrate deficit. MODIFIED KD + TRIHEPTANOIN: rational approach targeting TCA anaplerosis from both dietary (beta-OHB) and C7 (propionyl-CoA) substrate. Only initiate at tertiary metabolic-neurology centre with combined dietitian + metabolic physician + neurologist oversight.",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Standard Ketogenic Diet 4:1 (without anaplerotic supplementation)",
        "risk": "HIGH CAUTION — may deepen citrate deficit",
        "reason": (
            "Classical 4:1 KD reduces glucose-derived TCA substrate → less citrate generated "
            "via pyruvate/OAA route. In SLC13A5 deficiency, brain already cannot import "
            "circulating citrate (NaCT absent). Standard KD without triheptanoin may compound "
            "neuronal citrate depletion → worsening encephalopathy. MODIFIED KD with "
            "Triheptanoin (C7 anaplerotic) is the rational alternative — provides TCA "
            "propionyl-CoA independent of citrate import."
        ),
    },
    {
        "drug": "Valproate",
        "risk": "HIGH — POLG / metabolic / citrate-transport inhibition",
        "reason": (
            "VPA at high doses inhibits citrate transport. Also: POLG mutations + VPA → "
            "Alpers-Huttenlocher syndrome (fatal hepatic failure). VPA is not standard first-line "
            "in SLC13A5 (PB + LEV preferred in neonates). If used: POLG screen MANDATORY, "
            "monitor LFT/ammonia q6W initially, avoid in acute metabolic decompensation phase."
        ),
    },
    {
        "drug": "Tiagabine",
        "risk": "ABSOLUTE CI — NCSE risk",
        "reason": (
            "GAT-1 inhibitor → prolonged synaptic GABA → paradoxical NCSE in neonates/infants "
            "with diffuse cortical involvement. ABSOLUTE CI in EIEE25 / SLC13A5 deficiency. No exception."
        ),
    },
    {
        "drug": "IV Phenytoin / Fosphenytoin (neonatal)",
        "risk": "HIGH — cardiac arrhythmia / masking",
        "reason": (
            "IV phenytoin in neonates: cardiac conduction disturbances (bradycardia, VT) due to "
            "propylene glycol vehicle and direct cardiac NaV block at neonatal doses. Also: "
            "PHT suppresses clinical seizure manifestation without fully abolishing EEG seizures "
            "(electroclinical dissociation) → misleading assessment. Fosphenytoin safer vehicle "
            "but still cardiac monitoring mandatory. Not recommended as first-line in EIEE25."
        ),
    },
    {
        "drug": "Vigabatrin without ERG baseline (prolonged use)",
        "risk": "HIGH — irreversible visual field defect",
        "reason": (
            "VGB causes irreversible concentric visual field defect (VFD) in 30-40% with "
            "prolonged use. Baseline ERG MANDATORY before starting. q3M ERG during use. "
            "Discontinue at first sign of VFD or after 12-18 months in infancy. SHARE REMS "
            "enrolment required. VFD is permanent — cannot be reversed after discontinuation."
        ),
    },
]

# ── Monitoring ─────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "Urine citrate (spot or 24h) — SLC13A5 biomarker", "frequency": "q3M"},
    {"item": "Developmental assessment (BSID-III, Vineland)", "frequency": "q6M"},
    {"item": "Dental evaluation — enamel hypoplasia monitoring", "frequency": "q12M (paediatric dentist)"},
    {"item": "Seizure diary (type/frequency/duration)", "frequency": "Continuous (digital)"},
    {"item": "EEG (background + focal discharges)", "frequency": "q12M + after any SE"},
    {"item": "MRI brain (baseline + myelination check)", "frequency": "Baseline; repeat 12M, 24M"},
    {"item": "POLG screen (mandatory before VPA consideration)", "frequency": "Once (before VPA)"},
    {"item": "Triheptanoin response (acylcarnitines C4DC/C5OH)", "frequency": "q3M (if on Triheptanoin)"},
    {"item": "Liver function (LFT + ammonia — AED monitoring)", "frequency": "q6M (q3M if on PB long-term)"},
    {"item": "Carnitine levels (PB-related depletion)", "frequency": "q6M (if on PB >6M)"},
    {"item": "ERG (if on VGB — SHARE REMS)", "frequency": "Baseline, then q3M"},
    {"item": "SUDEP risk counselling (nocturnal monitoring)", "frequency": "Annual"},
    {"item": "Genetic counselling — AR recurrence risk (25%)", "frequency": "At diagnosis + pre-conception"},
    {"item": "Ophthalmology (if on VGB)", "frequency": "q3M (Goldmann perimetry from age 5)"},
]

# ── Lifecycle Windows ──────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal (0-4 weeks)",
        "headline": "Neonatal Clonic/Multifocal Seizures — EIEE25 Onset",
        "detail": (
            "Day 1-5: clonic/multifocal neonatal seizures. Continuous aEEG mandatory. "
            "PB first-line (20 mg/kg IV loading). Pyridoxine 100 mg IV trial mandatory. "
            "URGENTLY: urine organic acids + citrate (elevated citrate = SLC13A5 signal). "
            "Metabolic team involvement from Day 1. Avoid IV phenytoin (cardiac risk). "
            "Initiate LEV 40-60 mg/kg/day if PB insufficient. Genetic panel including SLC13A5."
        ),
    },
    {
        "window": "Early Infancy (1-6 months)",
        "headline": "Transitional Epilepsy — Focal Seizures Emerge",
        "detail": (
            "Neonatal clonic seizures evolve to focal with automatisms. EEG: "
            "multifocal → focal temporal predominance. Genetic result expected: "
            "if biallelic SLC13A5 confirmed, initiate Triheptanoin (compassionate use/trial). "
            "Developmental support: PT, OT, SLT from 3 months. Dental observation begins. "
            "VGB may be considered for focal epilepsy in this window (SHARE REMS). "
            "PB taper discussion if seizure-free 3M."
        ),
    },
    {
        "window": "Late Infancy (6-24 months)",
        "headline": "Epilepsy Stabilisation / Developmental Monitoring",
        "detail": (
            "Seizure frequency typically decreases by age 12-18M in attenuated phenotypes. "
            "Severe biallelic LOF: ongoing drug-resistance — consider Triheptanoin + modified KD. "
            "Developmental delay characterised (motor milestones delayed 6-12M). EEG: focal "
            "discharges persist but may become less frequent. Primary teeth erupt: dental "
            "defects (hypomineralised/hypoplastic) visible from 6M. Paediatric dentist referral."
        ),
    },
    {
        "window": "Early Childhood (2-6 years)",
        "headline": "Drug-Resistance Assessment / School Entry Planning",
        "detail": (
            "If 2 appropriate AEDs have failed: DRE criteria met → escalation strategy. "
            "Triheptanoin formal trial if not yet initiated. Modified KD discussion with "
            "metabolic dietitian. Educational needs assessment: IEP/EHCP (mild-moderate ID). "
            "Dental: permanent teeth developing — fluoride supplementation; sealants; "
            "early orthodontic review. Seizure-free driving restriction N/A (age), but "
            "inform parents: future driving implications."
        ),
    },
    {
        "window": "School Age (6-16 years)",
        "headline": "Cognitive / Behavioural Support / AED Review",
        "detail": (
            "Neuropsychological profiling: WISC-V, CBCL, ADHD-RS. SLC13A5: ADHD-like "
            "behaviours in 30-40%. AED review: taper VGB if still on (VFD risk), "
            "optimize LEV/OXC/CLB. Transition: dentist awareness of enamel defects (dental "
            "modifications needed for enamel hypoplasia). Genetic counselling for siblings. "
            "Driving: inform at age 14 re: seizure-free requirements (jurisdiction-specific)."
        ),
    },
    {
        "window": "Adolescent / Adult (16+ years)",
        "headline": "Transition to Adult Services / Reproductive Counselling",
        "detail": (
            "Transition to adult epilepsy neurology + metabolic physician. REPRODUCTIVE "
            "COUNSELLING: AR inheritance — offspring risk depends on partner carrier status "
            "(population carrier frequency ~1:80 for SLC13A5 pathogenic variants in some "
            "populations). Preconception panel carrier testing for partner. VPA in females: "
            "VPPP mandatory if VPA continued. Triheptanoin: safety in pregnancy unknown "
            "(limited data) — discuss with metabolic team. SUDEP risk counselling annual."
        ),
    },
]

# ── Concepts ───────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "SLC13A5 (17p13.1)", "definition": "Sodium-coupled citrate transporter gene; AR LOF → EIEE25 / Citrate Transporter Deficiency; 9 exons; biallelic variants required; OMIM #615905"},
    {"term": "NaCT (Sodium-Coupled Citrate Transporter)", "definition": "SLC13 family member; 4 Na⁺ : 1 citrate³⁻ stoichiometry; imports circulating citrate across BBB into neurons; highest brain expression at blood-brain barrier endothelium and cortical neurons"},
    {"term": "EIEE25 — OMIM #615905", "definition": "Early Infantile Epileptic Encephalopathy 25; AR; neonatal onset; biallelic SLC13A5 LOF; elevated urine citrate; dental enamel hypoplasia; intellectual disability"},
    {"term": "TCA cycle anaplerosis (citrate deficit mechanism)", "definition": "Citrate → OAA + acetyl-CoA → fuels glutamate → GABA. NaCT LOF depletes neuronal citrate → reduced GABA synthesis → cortical hyperexcitability → seizures"},
    {"term": "Dental enamel hypoplasia/hypomineralisation (pathognomonic)", "definition": "NaCT deficiency in ameloblasts → impaired citrate-hydroxyapatite crystallisation → hypoplastic/hypomineralised teeth; PATHOGNOMONIC for SLC13A5; affects permanent > primary dentition"},
    {"term": "Autosomal Recessive (AR) inheritance", "definition": "Biallelic LOF variants required; both parents obligate carriers (25% recurrence risk per pregnancy); de novo biallelic extremely rare; consanguinity increases homozygosity risk"},
    {"term": "Neonatal onset epilepsy (Day 1-5)", "definition": "70% of classic SLC13A5 biallelic LOF present with seizures Day 1-5 of life; continuous aEEG mandatory; initiates broad neonatal epilepsy workup including pyridoxine trial and metabolic screen"},
    {"term": "Triheptanoin (C7 anaplerotic therapy)", "definition": "Odd-chain C7 fatty acid → β-oxidation → propionyl-CoA → succinyl-CoA → TCA cycle; restores TCA anaplerosis INDEPENDENT of NaCT; only precision therapy targeting SLC13A5 root cause; under clinical trial"},
    {"term": "Urine citrate biomarker", "definition": "Elevated urine citrate (reduced BBB uptake → renal overflow) is the key biomarker for SLC13A5 deficiency; urine organic acid screen or dedicated citrate assay; sensitivity ~85% for biallelic LOF"},
    {"term": "POLG — Alpers-Huttenlocher syndrome", "definition": "POLG mutations + VPA → mitochondrial DNA polymerase failure → hepatic failure (fatal); POLG screen MANDATORY before VPA in ALL patients with metabolic/mitochondrial-adjacent epilepsies"},
    {"term": "Focal epilepsy (not GGE) — NaV blockers permitted", "definition": "SLC13A5 epilepsy is a FOCAL (not generalised) epilepsy; OXC/CBZ are NOT contraindicated (unlike GGE). Key distinction: GGE aggravation by NaV blockers is a GGE-specific phenomenon, not applicable to SLC13A5 focal epilepsy"},
    {"term": "PFK1 inhibition by citrate", "definition": "Extracellular citrate inhibits phosphofructokinase-1 (PFK1), allosterically braking glycolysis. NaCT LOF → reduced intracellular citrate → PFK1 disinhibition → uncontrolled glycolysis → altered neuronal energy homeostasis"},
    {"term": "SHARE REMS (Vigabatrin)", "definition": "FDA Risk Evaluation and Mitigation Strategy for VGB; mandatory ERG monitoring q3M; IRB-style enrollment; ophthalmology required; VFD irreversible — stop VGB if detected"},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Risk elevated in EIEE25 with drug-resistant nocturnal seizures; annual counselling; nocturnal supervision devices (seizure alert monitor, pulse oximetry); SUDEP risk higher with uncontrolled GTCS/tonic-clonic"},
    {"term": "ACMG/AMP variant classification", "definition": "5-tier classification (pathogenic/likely pathogenic/VUS/likely benign/benign); biallelic SLC13A5 classification: functional studies for missense (citrate transport assay) + urine citrate correlation critical for VUS resolution"},
]

# ── Standards ──────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE Classification of Seizures and Epilepsies 2022 (Fisher et al.)",
    "NICE NG217 — Epilepsies: Diagnosis and Management (2022)",
    "Hardies K et al. (2015) Hum Mol Genet — SLC13A5 mutations in neonatal epilepsy",
    "Bainbridge MN et al. (2017) Genet Med — EIEE25 clinical characterization",
    "Weeke LC et al. (2021) Ann Neurol — SLC13A5 citrate transporter epilepsy cohort",
    "CPIC Guideline — POLG / Valproate (2023)",
    "FDA Valproate REMS — females of reproductive age (2022)",
    "ILAE Dietary Therapies Consensus (2018): Ketogenic Diet guidelines",
    "SHARE REMS — Vigabatrin (VGB) risk management programme",
    "MHRA Valproate Pregnancy Prevention Programme (2021)",
    "ACMG/AMP Variant Interpretation Guidelines (2015/2023)",
    "CPIC HLA-B*15:02 / OXC-CBZ (2023 — SE Asian ancestry)",
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"label": "PB loading dose (neonatal SE / first-line)", "value": "20 mg/kg IV (+ 10 mg/kg if needed)", "unit": "mg/kg"},
    {"label": "LEV neonatal loading dose", "value": "40-60 mg/kg/day IV", "unit": "mg/kg/day"},
    {"label": "Pyridoxine diagnostic trial (all neonatal seizures)", "value": "100 mg IV", "unit": "mg (single dose — then 50 mg/day if response)"},
    {"label": "PB TDM target (neonatal maintenance)", "value": "15-40", "unit": "mg/L"},
    {"label": "Triheptanoin target dose", "value": "1-2 g/kg/day (≤35% of daily calories)", "unit": "g/kg/day"},
    {"label": "Fever rescue threshold (SLC13A5-specific)", "value": "≥37.5", "unit": "°C (lower than classic FS 38°C)"},
    {"label": "AED trials before DRE diagnosis", "value": "2 appropriate AEDs failed (Level A/B)", "unit": "trials"},
    {"label": "VGB ERG monitoring interval (SHARE REMS)", "value": "Every 3", "unit": "months (baseline mandatory before start)"},
    {"label": "Urine citrate elevated (SLC13A5 diagnostic signal)", "value": ">upper limit of normal age-adjusted reference", "unit": "mmol/mmol creatinine"},
    {"label": "VGB discontinuation timeline (to limit VFD risk)", "value": "12-18", "unit": "months maximum in infancy"},
    {"label": "POLG screen timing", "value": "Before first VPA dose", "unit": "mandatory"},
    {"label": "Seizure-free period before driving", "value": "12", "unit": "months seizure-free (jurisdiction-specific)"},
]

# ── References ─────────────────────────────────────────────────────────────────
REFERENCES = [
    "Hardies K et al. (2015) Hum Mol Genet 24(14):3981–3992 — SLC13A5 mutations cause early infantile epileptic encephalopathy with dental enamel defects",
    "Bainbridge MN et al. (2017) Genet Med 19(4):429–437 — EIEE25 (SLC13A5 citrate transporter deficiency) clinical characterisation",
    "Weeke LC et al. (2021) Ann Neurol 89(4):806–818 — SLC13A5 epilepsy cohort: genotype-phenotype correlations",
    "Bhatt DK et al. (2023) — SLC13A5 deficiency: clinical spectrum and therapeutic advances review",
    "Bhattacharya S et al. (2020) Epilepsia 61(12):2727–2739 — Triheptanoin in SLC13A5 deficiency: mechanism and early clinical data",
    "Salomons GS et al. (2016) J Inherit Metab Dis 39(3):381–388 — Citrate transporter deficiency: biomarkers and metabolomics",
]


# ── Patient Simulation ─────────────────────────────────────────────────────────
def _make_patients():
    random.seed(42)
    patients = []
    categories = [
        ("SLC13A5-biallelic-LOF-classic-neonatal", 50),
        ("SLC13A5-biallelic-missense-attenuated",  25),
        ("SLC13A5-biallelic-splice-variable",       15),
        ("SLC13A5-compound-het-mixed-class",         7),
        ("SLC13A5-negative-phenocopy-EIEE",          3),
    ]
    pid = 1
    for cat, pct in categories:
        n = max(1, round(40 * pct / 100))
        for _ in range(n):
            classic   = "classic" in cat
            attenuated = "attenuated" in cat
            splice    = "splice" in cat
            mixed     = "mixed" in cat
            phenocopy = "phenocopy" in cat

            drug_resistant = random.random() < (0.68 if classic else 0.40 if mixed else 0.30 if splice else 0.20 if attenuated else 0.10)
            on_pb  = random.random() < (0.90 if classic else 0.70)
            on_lev = random.random() < 0.80
            on_vgb = random.random() < (0.35 if classic else 0.20)
            on_oxc = random.random() < (0.40 if not (classic and drug_resistant) else 0.25)
            on_clb = random.random() < 0.30
            on_triheptanoin = random.random() < (0.50 if classic else 0.30 if mixed else 0.20)
            on_kd  = random.random() < (0.25 if drug_resistant else 0.08)
            polg_tested = "Y" if random.random() < 0.70 else "N"
            dental_defect = random.random() < (0.90 if classic else 0.80 if attenuated else 0.75 if splice else 0.65 if mixed else 0.05)
            urine_citrate_elevated = random.random() < (0.88 if not phenocopy else 0.05)
            sudep_high_risk = drug_resistant and random.random() < 0.30
            fever_provoked_sz = random.random() < (0.65 if classic else 0.50)
            status_epilepticus_hx = random.random() < (0.25 if classic else 0.12)
            sex = random.choice(["M", "F"])
            age = random.randint(1, 22)
            onset_day = random.randint(1, 7) if not attenuated else random.randint(5, 30)
            patients.append({
                "id": f"SLC13A5-{pid:03d}",
                "category": cat,
                "sex": sex,
                "age": age,
                "onset_day": onset_day,
                "on_pb": on_pb,
                "on_lev": on_lev,
                "on_vgb": on_vgb,
                "on_oxc": on_oxc,
                "on_clb": on_clb,
                "on_triheptanoin": on_triheptanoin,
                "on_kd": on_kd,
                "polg_tested": polg_tested,
                "dental_defect": dental_defect,
                "urine_citrate_elevated": urine_citrate_elevated,
                "drug_resistant": drug_resistant,
                "fever_provoked_sz": fever_provoked_sz,
                "status_epilepticus_hx": status_epilepticus_hx,
                "sudep_high_risk": sudep_high_risk,
            })
            pid += 1
    while len(patients) < 40:
        patients.append(patients[-1].copy())
        patients[-1]["id"] = f"SLC13A5-{pid:03d}"
        pid += 1
    return patients[:40]


PATIENTS = _make_patients()


# ── API functions ──────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    n_pb   = sum(1 for p in PATIENTS if p["on_pb"])
    n_lev  = sum(1 for p in PATIENTS if p["on_lev"])
    n_c7   = sum(1 for p in PATIENTS if p["on_triheptanoin"])
    n_kd   = sum(1 for p in PATIENTS if p["on_kd"])
    n_dre  = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_dent = sum(1 for p in PATIENTS if p["dental_defect"])
    n_ucit = sum(1 for p in PATIENTS if p["urine_citrate_elevated"])
    n_polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])
    n_se   = sum(1 for p in PATIENTS if p["status_epilepticus_hx"])
    avg_age = round(sum(p["age"] for p in PATIENTS) / n, 1)

    etiology_dist = []
    cat_counts = {}
    for p in PATIENTS:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
    for e in ETIOLOGY_CATALOG:
        etiology_dist.append({
            "etiology": e["etiology"],
            "n": cat_counts.get(e["category"], 0),
            "pct": e["pct"],
        })

    return {
        "title": "SLC13A5 Epilepsy — Citrate Transporter Deficiency / NAFE / EIEE25 / NaCT / AR / 17p13.1",
        "gene": "SLC13A5",
        "locus": "17p13.1",
        "inheritance": "Autosomal recessive (AR); biallelic LOF variants; both parents obligate carriers; 25% recurrence risk",
        "protein": "NaCT — Sodium-Coupled Citrate Transporter; imports circulating citrate across BBB; fuels neuronal TCA cycle and GABA synthesis",
        "mechanism": (
            "SLC13A5 biallelic LOF → NaCT absent at BBB → no citrate import into brain → "
            "depleted neuronal citrate → impaired TCA anaplerosis → reduced GABA synthesis → "
            "cortical hyperexcitability → neonatal seizures. Dental: NaCT-null ameloblasts → "
            "enamel hypoplasia (pathognomonic). Biomarker: elevated urine citrate."
        ),
        "key_aha": (
            "SLC13A5 is the ONLY genetic epilepsy with pathognomonic DENTAL ENAMEL DEFECTS. "
            "URINE CITRATE ELEVATED — screen all neonatal epilepsy. TRIHEPTANOIN (C7) is the "
            "precision anaplerotic therapy — bypasses NaCT, restores TCA flux. "
            "STANDARD 4:1 KD: HIGH CAUTION — may deepen citrate deficit. "
            "OXC/CBZ NOT CI (unlike GGE — this is focal epilepsy). "
            "POLG before VPA. Pyridoxine trial mandatory for ALL neonatal seizures."
        ),
        "kpis": {
            "n_patients": n,
            "drug_resistant_pct": round(100 * n_dre / n),
            "dental_defect_pct": round(100 * n_dent / n),
            "urine_citrate_elevated_pct": round(100 * n_ucit / n),
            "on_pb_pct": round(100 * n_pb / n),
            "on_lev_pct": round(100 * n_lev / n),
            "on_triheptanoin_pct": round(100 * n_c7 / n),
            "on_kd_pct": round(100 * n_kd / n),
            "polg_tested_pct": round(100 * n_polg / n),
            "sudep_high_risk_n": n_sudep,
            "se_history_n": n_se,
            "avg_age_years": avg_age,
        },
        "etiology_distribution": etiology_dist,
        "treatments_summary": [{"drug": t["drug"], "level": t["level"]} for t in TREATMENTS],
        "monitoring_summary": [{"item": m["item"], "frequency": m["frequency"]} for m in MONITORING_ITEMS[:8]],
        "lifecycle": [{"window": w["window"], "headline": w["headline"]} for w in LIFECYCLE_WINDOWS],
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [c["drug"] for c in CONTRAINDICATIONS],
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_breakdown():
    n = len(PATIENTS)
    cat_counts = {}
    for p in PATIENTS:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1

    n_dre  = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_dent = sum(1 for p in PATIENTS if p["dental_defect"])
    n_ucit = sum(1 for p in PATIENTS if p["urine_citrate_elevated"])
    n_polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_se   = sum(1 for p in PATIENTS if p["status_epilepticus_hx"])
    n_fever = sum(1 for p in PATIENTS if p["fever_provoked_sz"])
    n_c7   = sum(1 for p in PATIENTS if p["on_triheptanoin"])
    n_pb   = sum(1 for p in PATIENTS if p["on_pb"])
    n_sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])

    return {
        "summary": {
            "total": n,
            "drug_resistant_pct": round(100 * n_dre / n),
            "dental_defect_pct": round(100 * n_dent / n),
            "urine_citrate_elevated_pct": round(100 * n_ucit / n),
            "polg_tested_pct": round(100 * n_polg / n),
            "status_epilepticus_hx_n": n_se,
            "fever_provoked_pct": round(100 * n_fever / n),
            "on_triheptanoin_n": n_c7,
            "on_pb_pct": round(100 * n_pb / n),
            "sudep_high_risk_n": n_sudep,
        },
        "etiology_distribution": [
            {
                "category": e["category"],
                "n": cat_counts.get(e["category"], 0),
                "pct": e["pct"],
                "etiology": e["etiology"],
                "mechanism": e["mechanism"],
                "typical_variants": e["typical_variants"],
                "eeg_signature": e["eeg_signature"],
                "phenotype": e["phenotype"],
            }
            for e in ETIOLOGY_CATALOG
        ],
        "patient_sample": [
            {
                "id": p["id"],
                "category": p["category"],
                "sex": p["sex"],
                "age": p["age"],
                "onset_day": p["onset_day"],
                "on_pb": p["on_pb"],
                "on_lev": p["on_lev"],
                "on_vgb": p["on_vgb"],
                "on_oxc": p["on_oxc"],
                "on_clb": p["on_clb"],
                "on_triheptanoin": p["on_triheptanoin"],
                "on_kd": p["on_kd"],
                "polg_tested": p["polg_tested"],
                "dental_defect": p["dental_defect"],
                "urine_citrate_elevated": p["urine_citrate_elevated"],
                "drug_resistant": p["drug_resistant"],
                "fever_provoked_sz": p["fever_provoked_sz"],
                "status_epilepticus_hx": p["status_epilepticus_hx"],
                "sudep_high_risk": p["sudep_high_risk"],
            }
            for p in PATIENTS[:15]
        ],
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
        "trigger_detail": TRIGGERS,
        "treatment_detail": [
            {
                "drug": t["drug"],
                "level": t["level"],
                "moa": t["moa"],
                "dose": t["dose"],
                "efficacy": t["efficacy"],
                "safety": t["safety"],
                "monitoring": t["monitoring"],
                "slc13a5_note": t["slc13a5_note"],
            }
            for t in TREATMENTS
        ],
        "contraindications": [
            {"drug": c["drug"], "risk": c["risk"], "reason": c["reason"]}
            for c in CONTRAINDICATIONS
        ],
        "monitoring_items": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE_WINDOWS,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "contraindications": [
            {"drug": c["drug"], "risk": c["risk"]}
            for c in CONTRAINDICATIONS
        ],
        "references": REFERENCES,
    }
