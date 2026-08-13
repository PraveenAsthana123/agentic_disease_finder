"""
PCDH19 Clustering Epilepsy (PCDH19-CE) Dashboard
===================================================
41-patient cohort · PCDH19 (Xq22.1) · Protocadherin-19 · Cadherin Superfamily
PCDH19 clustering epilepsy: pathogenic loss-of-function (LOF) variants in PCDH19 (Xq22.1,
encoding Protocadherin-19) causing the hallmark seizure-cluster syndrome that uniquely
affects FEMALES — heterozygous females are affected, while hemizygous males are unaffected
carriers (the Cellular Interference Paradox). The ILAE 2022 classification renamed the
condition "PCDH19 Clustering Epilepsy" (PCDH19-CE), superseding the historical name
EFMR (Epilepsy and Mental Retardation limited to Females; OMIM 300088).

CELLULAR INTERFERENCE MODEL: PCDH19 encodes Protocadherin-19, a delta2-protocadherin
cell-cell adhesion molecule critical for cortical lamination and GABAergic interneuron
circuit formation. In heterozygous females, X-inactivation creates a mosaic of
PCDH19-expressing and PCDH19-null cells. These two cell populations cannot adhere
normally to each other — the "cellular interference" disrupts cortical network integrity
and creates an epileptogenic substrate. In hemizygous males, all cells uniformly lack
PCDH19 → no mosaic → no cellular interference → no epilepsy. Exception: somatic mosaic
males (post-zygotic PCDH19 mutation) CAN develop PCDH19-CE.

HALLMARK SEIZURE CLUSTERS: Abrupt clusters of seizures (3–20+ seizures/day for 1–7 days),
predominantly fever-triggered, interspersed with seizure-free periods (weeks to months).
Onset: 6 months to 3 years (median ~12 months). Seizures: focal (temporal > frontal),
secondary FBTCS. Cognitive outcomes variable: intellectual disability in ~40%, behavioural
problems (autism spectrum features) in ~60%.

FENFLURAMINE FDA-APPROVED (2022): Fenfluramine (Fintepla), a serotonin-releasing agent
and 5-HT2C receptor agonist, received FDA/EMA approval (2022) for PCDH19-CE as an
expanded indication (initially approved for Dravet 2020). Mechanism: serotonin-dependent
reduction of seizure cluster frequency. MANDATORY cardiac monitoring: echocardiography
q6 months to detect cardiac valvulopathy (rare but serious; incidence <1% at low doses).
Prescribers must enrol in the FINTEPLA REMS program (USA).

CLB CLUSTER PROTOCOL: Clobazam (CLB) is the cornerstone of cluster management in
PCDH19-CE — both as daily maintenance and as a PRN rescue dose escalation during fever/
cluster. Written "fever action plan" is mandatory: start CLB/buccal diazepam at first
febrile sign (≥38°C) before seizures begin. Evidence: CLB provides 50–80% reduction
in cluster frequency in retrospective series and is recommended Level B.

BROMIDE (KBr): Potassium bromide was historically used for PCDH19-CE before modern
AEDs, and remains effective as adjunct (Level C). Mechanism: bromide substitutes for
Cl⁻ in GABA-A channels, hyperpolarising neurons. Serum TDM mandatory (1.0–2.5 mmol/L).
Adverse effects: sedation, acneiform rash (bromoderma), cognitive dulling.

CATAMENIAL PATTERN: Perimenstrual exacerbation of seizure clusters occurs in ~28% of
post-pubertal women with PCDH19-CE. Progesterone (or norethisterone) adjunct is used
in case series; hormonal contraception may also modulate cluster frequency.
"""

import random
from datetime import datetime

SEED = 9179  # dashboard 179
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "Pathogenic PCDH19 truncating / frameshift (de novo, heterozygous female)",
        "n": 17, "pct": 42,
        "category": "De-novo-PCDH19-truncating-frameshift",
        "mechanism": (
            "Most prevalent PCDH19-CE variant class (~42%): de novo nonsense, frameshift, or large "
            "deletion variants causing truncation of the Protocadherin-19 extracellular cadherin (EC) "
            "repeats (EC1–EC6) or intracellular domain. These LOF variants abolish PCDH19 protein "
            "expression in ~50% of cortical cells (X-inactivation mosaic), creating the cellular "
            "interference substrate. Severe clinical phenotype: seizure onset <12 months, dense "
            "clusters (5–20 seizures/day), higher rate of intellectual disability (~55%) and autism "
            "spectrum features (~65%). Diagnostic approach: trio exome + PCDH19 deletion/duplication "
            "panel; some truncating variants detectable on gene panel but exome/genome preferred for "
            "full spectrum. De novo confirmed by parental testing — heterozygous father = CARRIER male."
        ),
        "eeg_signature": (
            "Frontal or frontotemporal focal ictal discharge (rhythmic theta 5–7 Hz evolving to "
            "alpha); may generalise. Ictal onset often subtle — bilateral synchronous FBTCS less "
            "common than focal. Interictal: multifocal IEDs, frontotemporal predominance; EEG may "
            "be normal between clusters. During cluster: nearly continuous interictal spiking."
        ),
        "mri": "Usually normal (>90%). Occasional hippocampal signal change in prolonged status. No structural malformations expected.",
        "clinical_note": (
            "If father is unaffected carrier male (hemizygous), ALL daughters are at 50% risk of "
            "being heterozygous (affected); ALL sons at 50% risk of being hemizygous carriers "
            "(unaffected). PCDH19 testing of the father is MANDATORY to determine inheritance mode "
            "and recurrence risk. Somatic mosaic father (blood-level mosaicism <10%) may be missed — "
            "skin biopsy or saliva testing may improve detection. Clinic: fever action plan at first visit."
        ),
    },
    {
        "etiology": "Pathogenic PCDH19 missense (de novo, heterozygous female)",
        "n": 14, "pct": 34,
        "category": "De-novo-PCDH19-missense",
        "mechanism": (
            "De novo PCDH19 missense variants (~34%): amino acid substitutions affecting the "
            "extracellular cadherin (EC1–EC6) calcium-binding interfaces (particularly EC1, EC2 "
            "calcium-coordination residues) or the intracellular PDZ-binding domain. These variants "
            "cause cellular interference via dominant-negative disruption of PCDH19 homophilic "
            "adhesion rather than haploinsufficiency. Pathogenic missense variants cluster at "
            "conserved EC-domain residues (DXD and DXNDN calcium-binding loops). ClinGen PCDH19 "
            "Expert Panel review required for VUS classification — many missense variants are reclassified "
            "to likely pathogenic after functional studies (PCDH19 EC-domain expression assays). "
            "Clinical phenotype: similar to truncating variants; cluster frequency and severity "
            "variable by variant position."
        ),
        "eeg_signature": (
            "Focal frontotemporal or temporal ictal onset; ictal: rhythmic beta/alpha, 10–14 Hz "
            "high-amplitude; may secondarily generalise to FBTCS. Interictal: focal or multifocal "
            "IEDs; EEG background generally preserved between clusters."
        ),
        "mri": "Normal in >90%; occasional white matter signal changes in prolonged cluster/SE episodes.",
        "clinical_note": (
            "Missense variant pathogenicity requires careful interpretation: ACMG PS3 (functional "
            "data) + PP3 (in silico tools: REVEL >0.7 supportive) + PM2 (absent from gnomAD) + "
            "PM1 (critical EC domain position). Variants in non-conserved regions with no functional "
            "data may remain VUS — functional PCDH19 cell-aggregation assay is available in "
            "specialised labs. Family testing of mother and paternal side essential."
        ),
    },
    {
        "etiology": "Pathogenic PCDH19 splice site (de novo, heterozygous female)",
        "n": 4, "pct": 10,
        "category": "De-novo-PCDH19-splice-site",
        "mechanism": (
            "Splice-site variants (~10%): canonical ±1/2 or deep-intronic splice variants causing "
            "exon skipping, cryptic splice activation, or intron retention. Effect: partial or "
            "complete LOF of one PCDH19 allele. RT-PCR on RNA from lymphoblasts or fibroblasts "
            "required to confirm aberrant splicing. Phenotype generally intermediate — less severe "
            "than truncating if partial LOF; similar if complete exon skipping. Deep-intronic "
            "variants (e.g., affecting the PCDH19 splicing regulatory elements) may be missed on "
            "standard exome sequencing — whole-genome sequencing (WGS) + RNA sequencing recommended "
            "for PCDH19-negative females with clustering epilepsy phenotype."
        ),
        "eeg_signature": "Variable; frontotemporal focal as above; cluster EEG morphology indistinguishable from truncating variants.",
        "mri": "Normal.",
        "clinical_note": (
            "WGS is superior to exome for detecting deep-intronic PCDH19 splice variants. If "
            "standard exome is negative in a female with classic clustering epilepsy phenotype, "
            "request trio WGS + RNA seq. Consider PCDH19 mRNA expression analysis in available "
            "tissue. Family testing essential (father blood + saliva for mosaicism detection)."
        ),
    },
    {
        "etiology": "Pathogenic PCDH19 deletion / duplication (CNV, heterozygous female)",
        "n": 3, "pct": 8,
        "category": "De-novo-PCDH19-CNV-deletion",
        "mechanism": (
            "Chromosomal copy number variants (~8%): intragenic deletions (single or multi-exon) "
            "or whole-gene deletions of PCDH19 (Xq22.1). May include flanking genes (deletion "
            "syndromes) — requires microarray analysis or WGS for full characterisation. "
            "Multi-exon deletions cause complete haploinsufficiency — most severe cellular "
            "interference phenotype expected. Duplications of PCDH19 in females are typically "
            "pathogenic only if in a cellular-interference context (heterozygous duplication in "
            "a mosaic background). Intragenic duplications may cause aberrant splicing."
        ),
        "eeg_signature": "Frontal/frontotemporal focal; indistinguishable from other LOF classes during cluster.",
        "mri": "Normal; co-deleted genes may add additional phenotypic features if large deletion.",
        "clinical_note": (
            "Array CGH / chromosomal microarray is first-tier for CNV detection. If intragenic "
            "deletion suspected but not confirmed, MLPA (Multiplex Ligation-dependent Probe "
            "Amplification) for PCDH19 exon-level resolution. Whole-gene deletion: cascade testing "
            "for carrier father and recurrence risk quantification essential."
        ),
    },
    {
        "etiology": "Clinical PCDH19-negative (PCDH19-CE phenocopy, female clustering epilepsy)",
        "n": 3, "pct": 6,
        "category": "Clinical-PCDH19-negative-phenocopy",
        "mechanism": (
            "Female patients with the clustering epilepsy phenotype (fever-triggered clusters, "
            "focal seizures, characteristic age of onset) but no pathogenic PCDH19 variant on "
            "comprehensive sequencing (~6%). Alternative diagnostic considerations: GABRA1, GABRB3, "
            "GABRG2 (GABAergic) mutations; SCN1A (Dravet overlap — febrile clusters); KCNQ2 (later "
            "cluster onset); SLC6A1 (MAE with clusters); chromosomal mosaicism below detection; "
            "somatic PCDH19 mosaicism in tissue (brain) not represented in blood. Somatic mosaic "
            "PCDH19 must be considered — brain-limited mosaicism would not be detectable on blood "
            "exome. Skin biopsy → fibroblasts → RNA expression is alternative. Empiric CLB cluster "
            "protocol is reasonable pending genetic diagnosis."
        ),
        "eeg_signature": "Clustering focal EEG pattern — indistinguishable from PCDH19-positive on routine EEG.",
        "mri": "Usually normal; Dravet (SCN1A-negative) may show hippocampal sclerosis in later life.",
        "clinical_note": (
            "Comprehensive epilepsy gene panel (250–500 genes) or trio genome sequencing recommended. "
            "If still negative, consider somatic mosaicism (skin fibroblasts, hair follicles, saliva "
            "vs blood discordance). PCDH19-CE management (CLB, fenfluramine fever plan) is appropriate "
            "empirically while awaiting genetic diagnosis. Fenfluramine is specifically FDA-labelled "
            "for PCDH19-CE — off-label use for phenocopy should be discussed with expert center."
        ),
    },
]

# ── Seizure Types (4 primary, N=41 cohort) ──────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Fever-triggered focal seizure clusters (PCDH19-CE hallmark)",
        "prevalence_pct": 95,
        "onset_age": "6 months – 3 years (median ~12 months)",
        "eeg_correlate": (
            "Hallmark PCDH19-CE EEG pattern: abrupt focal ictal discharge (frontotemporal or "
            "temporal onset) at the start of cluster, recurring multiple times per day for 1–7 days. "
            "Ictal: rhythmic theta (5–7 Hz) or alpha (8–10 Hz) evolving to higher amplitude, focal "
            "or rapid bilateral spread. Between clusters: interictal EEG may be completely normal "
            "or show multifocal frontotemporal IEDs. During febrile cluster: near-continuous "
            "spiking; clinical seizure frequency 5–20+ per day."
        ),
        "clinical_tip": (
            "PCDH19-CE cluster management requires a WRITTEN FEVER ACTION PLAN issued at the first "
            "clinic visit: start CLB dose escalation or buccal diazepam/midazolam at first febrile "
            "sign (T ≥38°C) BEFORE the first seizure — do not wait for the cluster to establish. "
            "Prompt antipyretic use (paracetamol + ibuprofen alternation) is adjunct. Parent "
            "education: cluster recognition, cluster rescue, when to call emergency services. "
            "Request 24h ambulatory EEG during a cluster for characterisation."
        ),
    },
    {
        "type": "Focal to bilateral tonic-clonic (FBTCS)",
        "prevalence_pct": 72,
        "onset_age": "Variable — within clusters; any age after onset",
        "eeg_correlate": (
            "Rapid bilateral spread from frontotemporal ictal focus — clonic phase with generalised "
            "high-amplitude polyspike-slow wave. FBTCS duration typically <3 min in PCDH19-CE "
            "(shorter than Dravet). Postictal suppression followed by resumption of cluster focal "
            "seizures. Generalised spike-wave interictal activity may appear during dense clusters."
        ),
        "clinical_tip": (
            "FBTCS within a PCDH19-CE cluster does NOT indicate Dravet syndrome (though overlap "
            "exists: temperature sensitivity, onset age, female). Distinguish: PCDH19-CE does NOT "
            "typically show the persistent hemi-clonic status or prolonged initial status of "
            "Dravet. SCN1A sequencing warranted if >3 prolonged febrile status episodes — Dravet "
            "vs PCDH19-CE has different treatment implications (Na blockers OK in PCDH19; NOT in Dravet)."
        ),
    },
    {
        "type": "Atypical absence / staring seizures",
        "prevalence_pct": 38,
        "onset_age": "Predominantly 18 months – 6 years",
        "eeg_correlate": (
            "Diffuse or bifrontal slow spike-wave (2–2.5 Hz) atypical absence; often within "
            "cluster periods. May be misidentified as focal impairment of consciousness if "
            "bifrontal slow wave not prominent. Response to hyperventilation is variable — "
            "unlike childhood absence epilepsy (CAE), PCDH19-CE atypical absences do not "
            "reliably activate with HV. Associated with higher cognitive load during clusters."
        ),
        "clinical_tip": (
            "Atypical absences in PCDH19-CE increase cognitive burden during clusters. CLB "
            "covers both focal and absence-type seizures. Do NOT use ethosuximide alone — not "
            "effective for the focal cluster component. VPA broad-spectrum adjunct covers atypical "
            "absence component; POLG exclusion mandatory before VPA initiation."
        ),
    },
    {
        "type": "Myoclonic seizures",
        "prevalence_pct": 22,
        "onset_age": "Toddler to early school age",
        "eeg_correlate": (
            "Brief high-amplitude generalised polyspike-wave burst (50–200 ms) associated with "
            "sudden bilateral upper limb jerk. Occurs within clusters; inter-ictal myoclonus less "
            "common. EEG: generalised or bifrontal polyspike. Distinguish from hyperekplexia "
            "(startle myoclonus) by EEG coupling and fever-trigger pattern."
        ),
        "clinical_tip": (
            "Myoclonic component in PCDH19-CE generally responds to VPA or LEV. Myoclonus within "
            "cluster may be a sign of escalating seizure activity and warrants cluster rescue "
            "protocol initiation. Clonazepam (long-acting BDZ) as maintenance adjunct may be "
            "considered for persistent myoclonic component; CLB preferred for cluster rescue."
        ),
    },
]

# ── Triggers (8) ──────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / febrile illness",
        "rate_pct": 95,
        "mechanism": (
            "Primary trigger in PCDH19-CE (95% of patients): fever lowers seizure threshold via "
            "temperature-dependent increase in neuronal excitability and disruption of PCDH19-dependent "
            "GABAergic interneuron circuits. The cellular interference substrate is exquisitely "
            "temperature-sensitive — fever creates a perfect storm of reduced inhibitory tone + "
            "disrupted cell-adhesion signalling. Even low-grade fever (38°C) is sufficient to "
            "trigger a cluster in most patients; clusters typically begin within 12–24h of fever onset. "
            "Antipyretics (paracetamol + ibuprofen alternating) reduce but do not eliminate cluster risk."
        ),
        "management": (
            "Written FEVER ACTION PLAN: ① Temperature ≥38°C → paracetamol 15 mg/kg + ibuprofen "
            "10 mg/kg alternating q4h. ② Simultaneously: start CLB rescue dose (1–2 mg/kg extra "
            "daily dose, or per written plan). ③ If cluster begins: buccal midazolam 0.2–0.5 mg/kg "
            "or rectal diazepam 0.5 mg/kg. ④ If >5 seizures in 24h: ED review. ⑤ Pre-plan: all "
            "daycare/school carers trained in seizure recognition + rescue medicine administration."
        ),
    },
    {
        "trigger": "Hot weather / environmental hyperthermia",
        "rate_pct": 73,
        "mechanism": (
            "Ambient high temperature (>28°C), hot baths, physical exertion in heat — raises core "
            "body temperature sufficiently to trigger cluster without true febrile illness. "
            "Mechanism: same temperature-dependent threshold lowering as fever. "
            "Summer months and hot climates associated with increased cluster frequency. "
            "Heat exposure management is an important practical consideration for families."
        ),
        "management": "Avoid prolonged hot-weather exposure; no hot baths (lukewarm only); air-conditioned home during heat waves. Written plan covers heat-related hyperthermia same as fever.",
    },
    {
        "trigger": "Vaccination (7–10 days post-immunisation)",
        "rate_pct": 61,
        "mechanism": (
            "Post-vaccination inflammatory response (7–10 days after MMR, DTP, etc.) generates "
            "low-grade fever sufficient to trigger cluster. Important: vaccination must NOT be "
            "withheld — risk of vaccine-preventable illness (measles, pertussis) is far greater "
            "than cluster risk. Pre-vaccine CLB dose escalation for 48–72h is standard management. "
            "Live attenuated vaccines (MMR) more commonly associated than inactivated vaccines."
        ),
        "management": "Pre-vaccinate planning: ① Notify GP/pediatrician of PCDH19-CE. ② Start CLB dose escalation 24h before vaccination, continue 72h post. ③ Monitor temperature q4h for 10 days post MMR. ④ Written plan in vaccine record.",
    },
    {
        "trigger": "Intercurrent illness (without fever)",
        "rate_pct": 58,
        "mechanism": (
            "Systemic illness (viral gastroenteritis, URI) can trigger clusters even without "
            "significant fever. Inflammatory cytokines (IL-6, TNF-α) may lower seizure threshold "
            "independently of temperature. Dehydration + electrolyte disturbance during illness "
            "adds additional risk. AED absorption disrupted by vomiting → breakthrough clusters. "
            "Written illness management plan essential."
        ),
        "management": "Illness plan: oral rehydration + CLB dose escalation during GI illness; if vomiting prevents oral AEDs → ED for IV medication. Alert team to any illness episode.",
    },
    {
        "trigger": "Emotional stress / excitement",
        "rate_pct": 47,
        "mechanism": (
            "Acute emotional arousal (excitement, anxiety, fear) can precipitate cluster onset — "
            "likely via cortisol/adrenaline surge altering GABAergic tone in the PCDH19-deficient "
            "cortical mosaic. Social situations (birthday parties, school transitions, family events) "
            "associated with increased cluster risk in retrospective surveys. "
            "Psychosocial support for families is an important management component."
        ),
        "management": "Behavioural strategies: predictable routines; pre-event CLB dose adjustment with specialist guidance; psychological support for child and family.",
    },
    {
        "trigger": "Missed AED dose",
        "rate_pct": 42,
        "mechanism": (
            "Missed doses of CLB or VPA reduce protective GABAergic tone acutely. CLB has a "
            "relatively short effective half-life at therapeutic doses — missed dose → >30% "
            "reduction in GABAergic protection within 12h. AED adherence is critical during "
            "high-risk periods (febrile illness, vaccination)."
        ),
        "management": "Adherence support: written AED schedule; electronic reminders; school nurse administration plan; community pharmacy blister packs for complex regimens.",
    },
    {
        "trigger": "Sleep deprivation",
        "rate_pct": 35,
        "mechanism": (
            "Sleep deprivation reduces seizure threshold in cortical networks by disrupting "
            "slow-wave sleep recovery of GABAergic function. Effect synergistic with fever — "
            "febrile illness + sleep deprivation is the highest cluster-risk combination. "
            "School-age children with PCDH19-CE often have poor sleep quality due to "
            "behavioural comorbidities; sleep hygiene is a modifiable risk factor."
        ),
        "management": "Regular sleep schedule; limit screen time; treat comorbid sleep disorder (referral to sleep clinic if persistent insomnia or parasomnias). Educate carers on sleep-risk interaction.",
    },
    {
        "trigger": "Menstrual / hormonal fluctuation (catamenial)",
        "rate_pct": 28,
        "mechanism": (
            "Perimenstrual exacerbation in post-pubertal women with PCDH19-CE (~28%): fluctuation "
            "in progesterone/oestrogen across the menstrual cycle alters GABAergic tone. "
            "Progesterone withdrawal (late luteal phase) increases cortical excitability — "
            "catamenial pattern. PCDH19 protein expression may be regulated by steroid hormones "
            "in glial and neuronal cells (emerging evidence). Hormonal contraception (progesterone-"
            "containing pill) may stabilise seizure frequency by reducing hormonal variability."
        ),
        "management": "Menstrual diary + seizure diary for 3+ cycles to confirm catamenial pattern. Options: norethisterone (progesterone) perimenstrually; progesterone-containing hormonal contraception; CLB dose escalation in luteal phase. Consult gynaecology and epilepsy specialist jointly.",
    },
]

# ── Treatments (8) ──────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Clobazam (CLB)",
        "evidence_level": "Level B (Class III–IV evidence; ILAE 2022; NICE NG217)",
        "role": "First-line — cluster maintenance + PRN rescue dose escalation",
        "dose": (
            "Maintenance: 0.1–0.3 mg/kg/day in 1–2 divided doses (max 40 mg/day). "
            "Cluster rescue protocol (fever action plan): additional 0.1–0.2 mg/kg/dose "
            "at fever onset, repeat q12h for duration of cluster (max 3–5 days escalation). "
            "Buccal CLB or buccal midazolam as acute rescue for cluster seizures >5 min."
        ),
        "moa": (
            "Positive allosteric modulator of GABA-A receptor at the benzodiazepine binding site "
            "(1,5-benzodiazepine: less sedating than 1,4-benzodiazepines). Enhances Cl⁻ conductance "
            "→ neuronal hyperpolarisation → reduced cortical excitability. Active metabolite: "
            "N-desmethylclobazam (norclobazam) — longer half-life, GABA-A agonist activity."
        ),
        "efficacy": "50–80% cluster frequency reduction in retrospective series (Specchio 2019, Kolc 2019). Best evidence for cluster prevention during fever. Cluster rescue: highly effective for aborting individual seizures within cluster.",
        "monitoring": "Norclobazam serum level (50–300 ng/mL); sedation scale; tolerance assessment at 6-monthly review; LFT q6M.",
        "safety": "Tolerance (BDZ class): dose-escalation protocol minimises tolerance risk. Sedation: dose-related. Respiratory depression: rare at therapeutic doses; risk with concomitant opioids.",
    },
    {
        "drug": "Fenfluramine (FFA; Fintepla)",
        "evidence_level": "Level B (FDA/EMA approved 2022 for PCDH19-CE; Phase 3 RCT data)",
        "role": "Second-line adjunct — PCDH19-CE cluster frequency reduction",
        "dose": (
            "Starting: 0.1 mg/kg/day in 2 divided doses. Titrate weekly by 0.1 mg/kg/day. "
            "Maximum: 0.35 mg/kg/day (max 17 mg/day with stiripentol; max 26 mg/day without). "
            "PCDH19-CE: typically 0.2–0.35 mg/kg/day. Maintenance dose determined by cluster "
            "response and echocardiographic monitoring."
        ),
        "moa": (
            "Serotonin releasing agent: promotes 5-HT release from presynaptic terminals via "
            "reverse transport; direct 5-HT2C receptor agonism → activation of descending "
            "serotonergic anti-epileptic pathways. Sigma-1 receptor agonism modulates Ca²⁺ "
            "signalling in ER. In PCDH19-CE: hypothesised to restore serotonin-dependent "
            "stabilisation of GABAergic interneuron networks disrupted by cellular interference."
        ),
        "efficacy": "Phase 3 open-label trial (Lagae 2022, Epilepsia): ≥50% reduction in monthly cluster days in 74% of patients vs baseline. Cluster-free periods extended significantly. First pharmacological agent with prospective trial data specifically for PCDH19-CE.",
        "monitoring": (
            "⚠️ MANDATORY CARDIAC MONITORING: Echocardiography q6M (cardiac valvulopathy risk — "
            "low at ≤0.35 mg/kg/day but mandated by FINTEPLA REMS). Baseline echo before starting. "
            "If new valvulopathy detected: dose reduction; if progressive: discontinue. "
            "Weight weekly during titration (anorexia effect). Blood pressure monthly."
        ),
        "safety": "⚠️ FINTEPLA REMS PROGRAM (USA): mandatory prescriber enrolment, patient registry, and cardiac monitoring. Cardiac valvulopathy: <1% at low doses. Anorexia/weight loss: common (20–30%). Somnolence: mild. Pulmonary arterial hypertension: rare at therapeutic doses.",
    },
    {
        "drug": "Potassium Bromide (KBr)",
        "evidence_level": "Level C (historical case series; PCDH19-CE specific: Scheffer 2008, Marini 2010)",
        "role": "Adjunct — effective in some PCDH19-CE patients when CLB insufficient",
        "dose": (
            "20–40 mg/kg/day in 2 divided doses (oral solution). Dose titrated to serum TDM "
            "target: 1.0–2.5 mmol/L. Therapeutic effect gradual (2–4 weeks to steady state "
            "due to long half-life ~12 days). Add with food to minimise GI irritation."
        ),
        "moa": (
            "Bromide (Br⁻) is taken up via neuronal chloride transporters and substitutes for Cl⁻ "
            "in GABA-A receptor ion channels, enhancing Cl⁻-mediated hyperpolarisation at lower "
            "GABA concentrations. Net effect: raised seizure threshold. Mechanism is GABAergic "
            "but distinct from BDZ (acts at the channel pore rather than the allosteric BDZ site) "
            "— therefore additive benefit with CLB rather than competing."
        ),
        "efficacy": "Effective in ~40–60% of PCDH19-CE patients as CLB adjunct (Marini 2010 retrospective; Scheffer 2008 series). May provide >50% cluster reduction in CLB-partial responders. Historical use predates CLB — now adjunct role.",
        "monitoring": "Serum bromide level q3M (target 1.0–2.5 mmol/L; toxic >3 mmol/L). Renal function q6M. Acne/rash check (bromoderma). Cognitive/sedation screening at each visit.",
        "safety": "Bromoderma (acneiform rash): 10–20%; dose-related. Sedation/cognitive dulling: dose-related. GI irritation: take with food. Bromism (toxicity): tremor, confusion, ataxia at >3 mmol/L — reduce dose.",
    },
    {
        "drug": "Valproate (VPA)",
        "evidence_level": "Level B (broad-spectrum; retrospective series; POLG EXCLUSION MANDATORY)",
        "role": "Adjunct — broad-spectrum coverage including atypical absence + myoclonic component",
        "dose": (
            "15–30 mg/kg/day in 2–3 divided doses (oral). Dose adjusted to TDM (target 50–100 mg/L). "
            "Slow titration recommended (2–4 weeks) to minimise GI side effects. IV VPA available "
            "for cluster status management (20 mg/kg loading over 5–10 min)."
        ),
        "moa": (
            "Multi-mechanistic: (1) Na⁺ channel inactivation — reduces high-frequency firing; "
            "(2) GABA transaminase inhibition → increased brain GABA levels; (3) T-type Ca²⁺ "
            "channel blockade — reduces thalamocortical absence oscillations; (4) HDAC inhibition — "
            "epigenetic neuroprotective effects. Effective for focal, generalised (absence, myoclonic), "
            "and broad-spectrum coverage — particularly useful for the absence + myoclonic component."
        ),
        "efficacy": "~50–60% ≥50% cluster reduction as CLB adjunct in PCDH19-CE retrospective data. Best for atypical absence + myoclonic seizure components within clusters.",
        "monitoring": "⚠️ POLG EXCLUSION MANDATORY: POLG gene panel before initiation. VPA TDM q3M (target 50–100 mg/L). LFT q3M (hepatotoxicity risk). Coagulation q6M. Ammonia if encephalopathy suspected. Weight monitoring (obesity risk). Folic acid supplementation.",
        "safety": "⚠️ POLG: fatal hepatic failure in POLG-mutant patients — ABSOLUTE contraindication pending POLG result. Teratogen: Neural tube defect risk; VPA NOT for reproductive-age females without contraception + folic acid 5 mg/day. Hyperammonaemia. Weight gain.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence_level": "Level C (adjunct; variable response in PCDH19-CE)",
        "role": "Adjunct — broad-spectrum; IV formulation useful for cluster status",
        "dose": (
            "20–40 mg/kg/day in 2 divided doses (oral; max 60 mg/kg/day). "
            "IV LEV for cluster status: 20–40 mg/kg over 15 min (can repeat ×1). "
            "Titrate oral dose over 2–4 weeks."
        ),
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation → reduces vesicle fusion and neurotransmitter exocytosis at synapses. Net anti-epileptic effect via presynaptic glutamate release reduction.",
        "efficacy": "Variable response in PCDH19-CE (~30–40% ≥50% reduction as adjunct). IV LEV useful in cluster status when oral CLB/BDZ fails. Not first-line as maintenance but useful as IV bridge during cluster episodes requiring hospitalisation.",
        "monitoring": "Behavioural side effects (irritability, aggression): more prominent in children; review at 3M; consider dose reduction or switch if severe. Renal function q12M (renally cleared).",
        "safety": "Behavioural side effects: irritability, aggression, anxiety (10–15% of children — significant overlap with PCDH19-CE behavioural comorbidities). Relatively safe teratogenically compared to VPA.",
    },
    {
        "drug": "Ketogenic Diet (KD 4:1)",
        "evidence_level": "Level C (case series; PCDH19-CE specific data emerging)",
        "role": "DRE adjunct — after ≥2 AED failures + CLB + fenfluramine",
        "dose": (
            "KD 4:1 fat:CHO+protein ratio initiated by paediatric metabolic dietitian. "
            "Caloric target per age/weight. Maintain BHB (beta-hydroxybutyrate) 2–4 mmol/L. "
            "Micronutrient supplementation: selenium, zinc, carnitine, multivitamins. "
            "Modify classic KD to Modified Atkins Diet (MAD) in older children/adolescents."
        ),
        "moa": "Ketone bodies (BHB, acetoacetate) inhibit vesicular glutamate transport + activate KATP channels → neuronal hyperpolarisation. Anti-inflammatory and neuro-protective mechanisms also contribute. BHB directly inhibits NLRP3 inflammasome — potentially relevant to fever-triggered cluster mechanism in PCDH19-CE.",
        "efficacy": "Case series (n=8–12 patients): 50% ≥50% seizure reduction; cluster frequency reduced in majority. BHB KATP pathway may specifically reduce fever-triggered threshold. ILAE Dietary Therapies 2018 recommends KD for DRE after 2+ AED failures.",
        "monitoring": "BHB 2–4 mmol/L q3M. Lipid profile annually. Micronutrients (Se, Zn, carnitine) q6M. Renal calculi ultrasound annually. Growth monitoring. Dental hygiene (acid diet).",
        "safety": "Initiation risks: hypoglycaemia, acidosis — hospitalised initiation recommended. GI intolerance: constipation, nausea. Renal calculi: 3–5%. Long-term: hyperlipidaemia (monitor). Contraindicated: fatty acid oxidation disorders, pyruvate carboxylase deficiency.",
    },
    {
        "drug": "Progesterone / Norethisterone (catamenial adjunct)",
        "evidence_level": "Level C (case series; catamenial PCDH19-CE only)",
        "role": "Catamenial-pattern adjunct — perimenstrual CLB escalation alternative",
        "dose": (
            "Norethisterone 5 mg/day days 14–28 of cycle (luteal phase). "
            "Or combined oral contraceptive (progesterone-predominant). "
            "Natural progesterone (oral micronised progesterone 200–400 mg/day at night) is an "
            "alternative — less androgenic. Duration: minimum 3 cycles to assess efficacy."
        ),
        "moa": "Progesterone and its neuroactive metabolite allopregnanolone (ALLO) are positive allosteric modulators of GABA-A receptors — augmenting inhibitory tone. Sustained luteal-phase progesterone prevents the perimenstrual withdrawal drop in ALLO that triggers catamenial seizure exacerbation.",
        "efficacy": "Case series (n=5–8 PCDH19-CE catamenial): 40–60% reduction in perimenstrual cluster days. Menstrual diary + seizure diary for ≥3 cycles required before initiating. Hormonal contraception also reduces cycle variability.",
        "monitoring": "Menstrual diary + seizure diary. Blood pressure monthly. Review at 3 cycles. Consider bone density if long-term use (DEXA at 2 years).",
        "safety": "Mood changes: depression risk. Breakthrough bleeding. Thrombotic risk: low with progesterone-only; slightly higher with combined OCP. Not teratogenic at standard doses.",
    },
    {
        "drug": "ACTH / Corticosteroids (cluster status protocol)",
        "evidence_level": "Level C (acute cluster status; PCDH19-CE expert opinion)",
        "role": "Acute — short-course for prolonged refractory cluster / cluster status",
        "dose": (
            "Short course: prednisolone 2 mg/kg/day (max 60 mg/day) for 7–14 days, then taper. "
            "ACTH (if cluster status): 0.5–1 mg/kg/day IM for 7–14 days. "
            "Use only for cluster status unresponsive to BDZ + IV AED (not for routine clusters)."
        ),
        "moa": "Anti-inflammatory and immunomodulatory: reduces cortical inflammatory cytokine burden (IL-6, TNF-α) that lowers seizure threshold during febrile illness. May also directly modulate GABA-A receptor expression via steroid receptor pathways.",
        "efficacy": "Expert opinion: short-course steroids can terminate prolonged cluster status in PCDH19-CE refractory to standard AED/BDZ rescue. Not for routine cluster management — reserved for cluster status (>24h continuous clustering with functional impairment).",
        "monitoring": "Blood pressure, glucose daily during ACTH. Infection surveillance (immunosuppression). Cushingoid features on prolonged courses. Avoid live vaccines during course.",
        "safety": "Infection risk. Hypertension. Hyperglycaemia. Immunosuppression: avoid live vaccines during treatment + 3 months after. Adrenal suppression with prolonged use — gradual taper.",
    },
]

# ── AED Monitoring (8 items) ──────────────────────────────────────────────
AED_MONITORING = [
    {
        "item": "Fenfluramine echocardiogram q6M (FINTEPLA REMS mandatory)",
        "scope": "All patients on fenfluramine (FFA/Fintepla)",
        "mechanism": (
            "Fenfluramine's serotonin-releasing mechanism can cause valvular heart disease via "
            "5-HT2B receptor activation on cardiac valve interstitial cells — historically seen "
            "at high doses in adults (Pondimin era); risk is low at PCDH19-CE doses (≤0.35 mg/kg/day) "
            "but mandated monitoring is required by FINTEPLA REMS."
        ),
        "action": "Baseline TTE (transthoracic echocardiogram) before starting FFA. Repeat q6M. If new mild valvulopathy: dose review + cardiology consult. Moderate/severe: discontinue FFA.",
        "evidence": "FINTEPLA REMS Program (FDA 2020, expanded PCDH19-CE 2022); Lagae 2022 safety data.",
    },
    {
        "item": "CLB norclobazam TDM 50–300 ng/mL",
        "scope": "All patients on clobazam maintenance",
        "mechanism": (
            "N-desmethylclobazam (norclobazam) is the active metabolite of CLB with GABA-A agonist "
            "activity. Norclobazam TDM is preferred over CLB TDM (norclobazam has longer half-life "
            "and is the primary active species at steady state). CYP2C19 poor metabolisers accumulate "
            "higher norclobazam levels — PGx testing warranted if unexpected toxicity or poor response."
        ),
        "action": "Norclobazam level at 4–6 weeks post-initiation (steady state), then q6M. If >400 ng/mL: sedation risk — reduce dose. CYP2C19 genotyping if variable response. Drug interaction: clobazam + stiripentol significantly elevates norclobazam (reduce CLB dose by 25%).",
        "evidence": "ILAE TDM Committee 2022; Coupez 2010 pharmacokinetics data.",
    },
    {
        "item": "VPA TDM 50–100 mg/L + LFT q3M",
        "scope": "All patients on valproate",
        "mechanism": (
            "VPA TDM ensures therapeutic range (50–100 mg/L). LFT monitoring for hepatotoxicity — "
            "particularly important in young children (highest risk of POLG-undetected hepatotoxicity). "
            "Coagulation monitoring: VPA inhibits platelet aggregation and fibrinogen synthesis. "
            "Ammonia level if encephalopathy suspected (VPA-induced hyperammonaemia — distinct from "
            "hepatotoxicity)."
        ),
        "action": "VPA level (trough) q3M. LFT + coagulation q3M. Ammonia if encephalopathy. Weight monitoring q3M (obesity risk). Folic acid 5 mg/day supplementation.",
        "evidence": "ILAE TDM Committee 2022; NICE NG217.",
    },
    {
        "item": "Bromide TDM 1.0–2.5 mmol/L + renal function",
        "scope": "All patients on potassium bromide",
        "mechanism": (
            "Bromide has a narrow therapeutic range (1.0–2.5 mmol/L). Toxicity (bromism) occurs at "
            ">3 mmol/L: tremor, confusion, ataxia, psychosis in severe cases. Bromide is renally "
            "cleared — renal impairment causes accumulation and toxicity. Competes with Cl⁻ — high "
            "NaCl intake (salt-rich diet) increases Br⁻ renal clearance → reduced levels."
        ),
        "action": "Serum bromide q3M at steady state (long half-life ~12 days). Renal function q6M. Dietary salt counselling — avoid sudden diet changes. Bromoderma skin check at each visit.",
        "evidence": "Clinical pharmacology review — bromide; PCDH19-CE expert consensus.",
    },
    {
        "item": "Developmental assessment Bayley-III / WISC q6–12M",
        "scope": "All PCDH19-CE patients",
        "mechanism": (
            "Cognitive outcomes in PCDH19-CE range from normal (30%) to moderate intellectual "
            "disability (40%). Behavioural comorbidities (ASD features 60%, ADHD 35%) are common "
            "and impact functional outcome independent of seizure control. Regular developmental "
            "assessment enables early intervention, educational planning, and outcome tracking."
        ),
        "action": "Bayley-III (language, motor, cognitive) q6M until age 3. WPPSI/WISC age 4–16. ADOS/ADI-R if ASD features. ADHD rating scale q12M. Speech pathology referral if language delay >1SD.",
        "evidence": "PCDH19 Alliance guidelines; Kolc 2019 Genetics in Medicine neurodevelopmental data.",
    },
    {
        "item": "EEG / VEEG during cluster and q12M baseline",
        "scope": "All PCDH19-CE patients",
        "mechanism": (
            "Cluster EEG characterisation is essential for: diagnosis confirmation, seizure type "
            "classification (focal vs generalised component), assessment of subclinical seizure "
            "burden, and monitoring treatment response. Ambulatory EEG during cluster is preferred "
            "for capturing natural cluster morphology."
        ),
        "action": "Baseline 24h ambulatory EEG at diagnosis. VEEG during a cluster (hospitalized or ambulatory). Annual 24h EEG for monitoring. Repeat VEEG if seizure semiology changes or new seizure type suspected.",
        "evidence": "ACNS EEG Standards 2021; PCDH19 Task Force 2019 (Specchio).",
    },
    {
        "item": "Fever Action Plan review q6M",
        "scope": "All PCDH19-CE patients and caregivers",
        "mechanism": (
            "Fever is the primary trigger in 95% of PCDH19-CE. The fever action plan (FAP) "
            "is the single most important management tool. Plans become outdated as patient weight "
            "changes (dose recalculation needed), AED regimen changes, or new rescue medications "
            "are added. Review at every clinic visit."
        ),
        "action": "Review FAP at every clinic visit (at minimum q6M). Update doses for current weight. Verify caregiver competency in rescue medication administration (buccal midazolam/diazepam technique). Provide to school nurse, daycare, emergency contact card. Update after any medication change.",
        "evidence": "PCDH19 Alliance Fever Action Plan; NICE NG217 rescue medication guidance.",
    },
    {
        "item": "Cardiac monitoring (FFA): weight + BP monthly during titration",
        "scope": "Patients on fenfluramine titration phase",
        "mechanism": (
            "Fenfluramine causes anorexia (serotonin-mediated appetite suppression) → weight loss "
            "during titration. In children, weight loss can affect dosing (mg/kg target dose) and "
            "growth. Blood pressure monitoring: serotonin-mediated vasoconstriction at high doses. "
            "Pulmonary arterial hypertension: rare at ≤0.35 mg/kg/day but mandated monitoring."
        ),
        "action": "Weight weekly during titration (first 12 weeks). BP monthly during titration. If weight loss >5% from baseline: diet counselling + dietitian referral; consider slower titration. Pulmonary pressure assessment via Doppler echo q6M.",
        "evidence": "FINTEPLA REMS; Knupp 2022 PCDH19-CE trial safety data.",
    },
]

# ── Absolute Contraindications (4) ────────────────────────────────────────────
ABSOLUTE_CI = [
    {
        "drug": "Lamotrigine (LTG) — Relative contraindication / Use with caution in PCDH19-CE",
        "scope": "PCDH19-CE patients with fever-triggered clusters",
        "mechanism": (
            "LTG is a sodium channel blocker with pro-epileptic effects reported in several "
            "PCDH19-CE patients (case reports: Marini 2010, Kolc 2019). Proposed mechanism: "
            "LTG may paradoxically worsen fever-triggered clusters in PCDH19-CE by altering "
            "cortical synchrony in the cellular-interference mosaic. Evidence is limited "
            "(case reports/series) but consistent enough to warrant avoiding LTG as first-line "
            "and considering an alternative if LTG is already prescribed and clusters worsen."
        ),
        "action": "Do NOT start LTG as first-line in PCDH19-CE. If patient is on LTG and clusters are worsening: consider tapering and switching. If LTG was started before PCDH19-CE diagnosis and seizures are controlled: careful monitoring rather than immediate withdrawal.",
        "evidence": "Marini 2010 Brain (case series); Kolc 2019 Genetics in Medicine (cohort data, LTG worsening reported); PCDH19-CE expert consensus.",
    },
    {
        "drug": "VPA — POLG exclusion MANDATORY before initiation",
        "scope": "All PCDH19-CE patients (infants, toddlers — highest POLG hepatotoxicity risk)",
        "mechanism": (
            "VPA causes fatal mitochondrial hepatopathy in patients with POLG (polymerase gamma, "
            "POLG1/POLG2) mutations — VPA inhibits mtDNA replication, causing acute liver failure. "
            "Risk is highest in children under 2 years with DEE phenotype, particularly if multiple "
            "organ involvement (Alpers syndrome). PCDH19-CE onset age (6–36M) overlaps perfectly "
            "with the highest-risk age for POLG hepatotoxicity."
        ),
        "action": "POLG gene panel (POLG1 + POLG2) MUST be reported as NEGATIVE before starting VPA. If POLG result pending → do NOT start VPA. If POLG positive → VPA is ABSOLUTELY contraindicated (use CLB, LEV, KD instead).",
        "evidence": "EAN 2014 Consensus on Mitochondrial Disease; NICE NG217; Alpers-Huttenlocher Syndrome OMIM 203700.",
    },
    {
        "drug": "Fenfluramine without FINTEPLA REMS enrolment (USA) / echo monitoring",
        "scope": "All patients on fenfluramine (FFA/Fintepla) in the USA",
        "mechanism": (
            "The FDA FINTEPLA REMS (Risk Evaluation and Mitigation Strategy) program requires "
            "mandatory prescriber certification, patient enrolment, and pharmacy participation "
            "before dispensing fenfluramine. Echocardiography monitoring q6M is a core REMS "
            "requirement. Dispensing FFA without REMS enrolment is a federal dispensing violation "
            "and exposes the patient to unmonitored cardiac valvulopathy risk."
        ),
        "action": "Enrol prescriber + patient in FINTEPLA REMS before first prescription. Confirm pharmacy is REMS-enrolled. Document baseline echo report in medical record before first dispensing. Set echo reminder q6M.",
        "evidence": "FDA FINTEPLA REMS program (2020, updated 2022); FDA label for Fintepla (fenfluramine).",
    },
    {
        "drug": "Hospital NPO without cluster rescue plan + AED continuation",
        "scope": "All PCDH19-CE patients during any hospital procedure / surgery requiring NPO",
        "mechanism": (
            "Nil per os (NPO) interrupts oral CLB and VPA — abrupt discontinuation can precipitate "
            "severe cluster status. CLB interruption for >12h → acute cluster breakthrough. "
            "Perioperative and procedure NPO planning must include: IV CLB equivalent (IV midazolam "
            "infusion) or buccal midazolam as bridge, IV VPA loading if VPA is maintenance, IV LEV "
            "as broad-spectrum adjunct, and notification of surgical/anaesthesia team of PCDH19-CE "
            "diagnosis and cluster risk."
        ),
        "action": "Pre-operative planning at every surgery/procedure requiring NPO: ① Alert anaesthesiology team. ② Plan IV midazolam infusion as CLB bridge. ③ IV VPA if VPA maintenance. ④ Written perioperative AED plan in surgical notes. ⑤ Post-op: resume oral AEDs as soon as tolerating oral intake; buccal midazolam PRN for any post-op cluster.",
        "evidence": "ILAE Dietary Therapies 2018 (perioperative); EAN Neonatal SE 2019; NICE NG217 peri-operative guidance.",
    },
]

# ── Lifecycle Windows (6) ────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Infancy — cluster onset (6–36M)",
        "age_range": "6 months – 3 years",
        "focus": "Fever action plan, molecular diagnosis, CLB initiation, developmental surveillance",
        "key_action": "Trio exome / PCDH19 panel STAT; written fever action plan at first visit; CLB initiation; developmental baseline (Griffiths/Bayley-III); parental education on rescue medicine.",
    },
    {
        "window": "Toddler years — cluster characterisation (2–5Y)",
        "age_range": "2 – 5 years",
        "focus": "Fenfluramine trial, cluster frequency optimisation, speech/language therapy, ASD screening",
        "key_action": "ADOS if ASD features; speech pathology referral; fenfluramine initiation if CLB-partial responder; echo baseline; update fever action plan for weight change.",
    },
    {
        "window": "Early childhood — DRE optimisation (5–8Y)",
        "age_range": "5 – 8 years",
        "focus": "KD trial if DRE, educational planning (EHCP/IEP), behaviour management",
        "key_action": "KD referral if ≥2 AED failures + CLB + fenfluramine; neuropsychological testing (WISC); educational needs assessment; bromide trial if KD deferred; ADHD screening.",
    },
    {
        "window": "School age — educational integration (8–12Y)",
        "age_range": "8 – 12 years",
        "focus": "Seizure management in school, cognitive support, ADHD treatment, sibling genetic testing",
        "key_action": "School seizure action plan (SSEAP); annual EEG; ADHD medication if indicated; sibling testing (father + sisters). Review medication regimen for weight-based dose adjustment.",
    },
    {
        "window": "Adolescence — hormonal + transition (12–18Y)",
        "age_range": "12 – 18 years",
        "focus": "Catamenial pattern, hormonal contraception, reproductive counselling, transition to adult services",
        "key_action": "Menstrual + seizure diary; catamenial management if perimenstrual cluster; reproductive counselling (X-linked inheritance, 50% daughter risk); transition planning to adult epilepsy service; driving safety counselling.",
    },
    {
        "window": "Adulthood — reproductive + chronic management (18Y+)",
        "age_range": "18 years and above",
        "focus": "Reproductive planning, VPA teratology counselling, ongoing cluster management, SUDEP",
        "key_action": "VPA: mandatory contraception + folic acid 5 mg/day if reproductive age; pre-conception counselling; carrier testing of partner; long-term fenfluramine cardiac monitoring continuation; SUDEP annual counselling; life insurance/disability planning support.",
    },
]

# ── Key Concepts (14) ──────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "PCDH19 / Protocadherin-19 (Xq22.1)",
     "definition": "X-linked gene encoding Protocadherin-19, a delta2-protocadherin cell-cell adhesion molecule of the cadherin superfamily. Expressed in cortical layer 2/3 pyramidal neurons and GABAergic interneurons; critical for cortical lamination, dendritic arborisation, and inhibitory circuit formation. LOF variants cause PCDH19 Clustering Epilepsy (OMIM 300088)."},
    {"term": "Cellular Interference Model (PCDH19-CE paradox)",
     "definition": "The mechanism explaining why heterozygous females (two populations: PCDH19+ and PCDH19-null cells) are affected while hemizygous males (all cells uniformly PCDH19-null) are UNAFFECTED carriers. The MOSAIC mixture of adhesion-competent and adhesion-deficient cells cannot form normal cell-cell contacts — this mismatch disrupts cortical GABAergic circuit assembly and creates an epileptogenic substrate. Uniformly null males lack the mismatch → no cellular interference → no epilepsy. Discovery: Dibbens 2008 NatGenet; model: Depienne 2009 AJHG."},
    {"term": "PCDH19-CE (PCDH19 Clustering Epilepsy)",
     "definition": "ILAE 2022 official syndrome name. Key features: (1) female-predominant (X-linked cellular interference); (2) onset 6–36 months; (3) fever-triggered seizure clusters (3–20+ seizures over 1–7 days); (4) focal (frontotemporal) seizures ± secondary generalisation; (5) cluster-free intervals of weeks to months; (6) variable cognitive outcome (normal to moderate ID); (7) ASD features in ~60%; (8) catamenial pattern in ~28% post-puberty. Formerly: EFMR (Epilepsy and Mental Retardation limited to Females)."},
    {"term": "X-linked Female Predominance (PCDH19-CE)",
     "definition": "Unique X-linked inheritance pattern: heterozygous females AFFECTED (cellular interference); hemizygous males UNAFFECTED carriers. All daughters of a carrier father have 50% risk of heterozygous PCDH19-CE; all sons of a carrier father have 50% risk of being carriers. Somatic mosaic males: rare exception — post-zygotic PCDH19 mutation creates mosaic (similar to females) → can develop PCDH19-CE."},
    {"term": "Fever Action Plan (FAP) — PCDH19-CE Mandatory",
     "definition": "Written emergency plan issued at the first clinic visit (mandatory). Contents: ① Temperature trigger threshold (≥38°C); ② Antipyretics: paracetamol + ibuprofen alternating; ③ CLB PRN rescue dose; ④ Buccal midazolam/diazepam dose + administration technique; ⑤ Trigger to call ambulance (>5 seizures, >5 min seizure, cluster status); ⑥ Emergency contacts; ⑦ School/daycare copy. Review at every clinic visit — update dose for weight change."},
    {"term": "Fenfluramine (Fintepla) — FDA/EMA approved 2022 for PCDH19-CE",
     "definition": "First pharmacological agent with prospective trial data specifically for PCDH19-CE (Lagae 2022). Mechanism: 5-HT releasing agent + 5-HT2C agonist + sigma-1 receptor. FDA approved 2022 as expanded indication (initially Dravet 2020). FINTEPLA REMS program: mandatory echocardiography q6M, prescriber certification, patient enrolment. Dose: ≤0.35 mg/kg/day (max 26 mg/day without stiripentol; 17 mg/day with stiripentol). Efficacy: 74% of patients achieved ≥50% reduction in monthly cluster days."},
    {"term": "Somatic Mosaic Males (PCDH19)",
     "definition": "Rare exception to the female-predominance rule: males with post-zygotic (somatic) PCDH19 mutations develop a mosaic of PCDH19+ and PCDH19-null cells — analogous to heterozygous females → cellular interference → PCDH19-CE. Blood-level mosaicism may be below standard sequencing detection threshold (<10–15%). Diagnostic approach: skin biopsy/fibroblasts + hair follicle RNA testing to detect low-level mosaicism missed on blood exome."},
    {"term": "POLG Exclusion before VPA",
     "definition": "POLG (polymerase gamma, POLG1/POLG2) mutations cause Alpers-Huttenlocher syndrome: fatal mitochondrial hepatopathy with VPA. Children with DEE onset <2 years are at risk. PCDH19-CE onset (6–36M) precisely overlaps the highest-risk window. POLG panel MUST be reported negative before VPA initiation. POLG positive → VPA ABSOLUTELY contraindicated. Use CLB, LEV, KD instead."},
    {"term": "Catamenial Epilepsy (PCDH19-CE)",
     "definition": "Perimenstrual exacerbation of seizure clusters in ~28% of post-pubertal women with PCDH19-CE. Mechanism: progesterone withdrawal (late luteal phase) reduces allopregnanolone (ALLO, a GABA-A positive allosteric modulator) → reduced inhibitory tone → cluster threshold lowered. Management: menstrual + seizure diary (≥3 cycles), progesterone/norethisterone luteal-phase supplementation, or combined OCP to reduce hormonal variability."},
    {"term": "Bromide (KBr) — PCDH19-CE adjunct",
     "definition": "Potassium bromide: one of the earliest AEDs (1857) that retains a role as CLB adjunct in PCDH19-CE. Mechanism: Br⁻ substitutes for Cl⁻ in GABA-A channels, augmenting inhibitory tone. TDM target: 1.0–2.5 mmol/L. ADRs: bromoderma (acneiform rash), sedation. Diet interaction: high-salt diet reduces bromide levels. Used in some European centres as first- or second-line PCDH19-CE agent where fenfluramine unavailable."},
    {"term": "SUDEP — PCDH19-CE",
     "definition": "Sudden Unexpected Death in Epilepsy. Risk exists in PCDH19-CE (DRE with nocturnal FBTCS) though lower than KCNT1-EIMFS or SCN8A-DEE. Annual SUDEP counselling is mandatory. Risk reduction: nocturnal supervision, SpO2 monitoring if nocturnal FBTCS, avoiding prone sleeping, optimising cluster control, SUDEP-safe home environments. PCDH19 Alliance provides SUDEP family resources."},
    {"term": "PCDH19 Alliance / International PCDH19 Collaborative (I2PC)",
     "definition": "Patient advocacy organisation for PCDH19-CE. Resources: genetic counselling, clinical trial registry, fever action plan templates, family support network. International PCDH19 Collaborative (I2PC) coordinates research between academic centres. Website: pcdh19alliance.com. Registry: PCDH19 International Patient Registry (IPRPCDH19)."},
    {"term": "Intellectual Disability and ASD in PCDH19-CE",
     "definition": "Cognitive outcomes are variable: normal IQ (30%), mild ID (30%), moderate ID (30%), severe ID (10%). Autism spectrum disorder (ASD) features in ~60%; ADHD in ~35%. Behavioural problems are often more disabling than seizure frequency. Early speech pathology, ABA therapy for ASD, and ADHD pharmacotherapy significantly improve functional outcomes. Cognitive outcome correlates with variant class (truncating > missense for ID prevalence) and early seizure control."},
    {"term": "ASO / Gene Therapy — PCDH19 (Preclinical)",
     "definition": "Restoration of PCDH19 expression is the hypothetical gene therapy goal, though the cellular interference model complicates therapeutic strategy (restoring PCDH19 to null cells might paradoxically worsen the mosaic imbalance). Research directions: (1) allele-specific silencing of the variant allele (equalising cells to null); (2) downstream pathway targets (GABAergic circuit restoration); (3) antisense oligonucleotides targeting aberrant splice products. No Phase 1 human trial open as of 2026."},
]

# ── Evidence Standards (8) ──────────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE-2022", "title": "ILAE Classification and Definition of Epilepsy Syndromes (2022)", "relevance": "PCDH19 Clustering Epilepsy (PCDH19-CE) syndrome definition and classification; DEE diagnostic framework."},
    {"standard": "NICE-NG217", "title": "Epilepsies: diagnosis and management (NG217, 2022)", "relevance": "AED selection, CLB first-line, fever action plan, rescue medication, SUDEP counselling, genetic testing pathway."},
    {"standard": "FDA-FINTEPLA-REMS-2022", "title": "FDA FINTEPLA REMS — Fenfluramine (2020, PCDH19 expanded 2022)", "relevance": "Mandatory prescriber certification, patient enrolment, echocardiography q6M, pharmacy requirements for fenfluramine."},
    {"standard": "ILAE-Dietary-Therapies-2018", "title": "ILAE Dietary Therapies Consensus (2018)", "relevance": "KD 4:1 initiation protocol, monitoring parameters, perioperative AED bridge management."},
    {"standard": "ACMG-AMP-2015", "title": "ACMG/AMP Standards for Variant Interpretation (2015)", "relevance": "PCDH19 LOF variant classification; ClinGen PCDH19 Expert Panel functional evidence framework."},
    {"standard": "ACNS-EEG-2021", "title": "ACNS Standardised EEG Terminology (2021)", "relevance": "PCDH19-CE cluster EEG characterisation; focal ictal discharge classification; ambulatory EEG standards."},
    {"standard": "EAN-CatamenialEpilepsy-2022", "title": "EAN Guideline on Catamenial Epilepsy and Reproductive Issues (2022)", "relevance": "Catamenial PCDH19-CE management; hormonal contraception; progesterone adjunct evidence grading."},
    {"standard": "Specchio-PCDH19-TaskForce-2019", "title": "Specchio et al. 2019 — International PCDH19 Task Force Consensus", "relevance": "First international expert consensus on PCDH19-CE management: CLB protocol, bromide use, diagnostic workup, genetic counselling."},
]

# ── Thresholds (10) ────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Onset 6–36M + fever clusters in FEMALE → PCDH19 molecular diagnosis",
     "action": "Request trio exome + PCDH19-specific deletion/duplication panel STAT. Test father for carrier status. If exome negative → WGS + RNA sequencing."},
    {"threshold": "CLB norclobazam TDM 50–300 ng/mL",
     "action": "Below 50: insufficient — review dose, check compliance, consider CYP2C19 genotyping. Above 400: sedation risk — reduce dose."},
    {"threshold": "FFA dose ≤0.35 mg/kg/day (max 26 mg/day without stiripentol)",
     "action": "Do not exceed. Weight change requires dose recalculation. Titrate weekly from 0.1 mg/kg/day. Adjust for concomitant stiripentol (lower max)."},
    {"threshold": "FFA echocardiogram q6M — new valvulopathy",
     "action": "Mild new valvulopathy: cardiology consult + dose review. Moderate/severe: discontinue FFA. Document in FINTEPLA REMS registry."},
    {"threshold": "VPA TDM 50–100 mg/L + POLG negative before start",
     "action": "POLG positive → stop VPA immediately (hepatic failure risk). VPA level below 50: sub-therapeutic — increase dose. Above 120: toxicity risk."},
    {"threshold": "Bromide TDM 1.0–2.5 mmol/L",
     "action": "Below 1.0: sub-therapeutic — increase dose or review diet (high salt reduces level). Above 3.0: bromism — reduce dose urgently; hold if symptomatic."},
    {"threshold": "POLG panel NEGATIVE before VPA",
     "action": "VPA is contraindicated until POLG result is negative. Pending POLG result = do NOT start VPA."},
    {"threshold": "Cluster ≥5 seizures in 24h → cluster rescue protocol",
     "action": "Initiate written fever action plan rescue arm: CLB dose escalation + buccal midazolam/diazepam. If >10 seizures in 24h or cluster >5 days → ED review."},
    {"threshold": "2 AED failures → fenfluramine + KD evaluation",
     "action": "After CLB monotherapy failure and 1 adjunct failure: initiate fenfluramine (REMS enrolment + echo) AND/OR KD referral. Do not defer to last resort."},
    {"threshold": "Menstrual cycle + seizure diary ≥3 cycles → catamenial pattern confirmation",
     "action": "If ≥2 of 3 cycles show perimenstrual cluster exacerbation (2× average seizure rate): initiate catamenial management (progesterone/hormonal contraception)."},
]

# ── References (6) ─────────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "Dibbens 2008 NatGenet",
     "title": "X-linked protocadherin 19 mutations cause female-limited epilepsy and cognitive impairment",
     "relevance": "Discovery paper identifying PCDH19 as the X-linked gene causing female-limited epilepsy (EFMR); established the X-linked but female-affected paradox. First description of cellular interference concept."},
    {"ref": "Depienne 2009 AJHG",
     "title": "Sporadic infantile epileptic encephalopathy caused by mutations in PCDH19 resembles Dravet syndrome but mainly affects females",
     "relevance": "Expanded the PCDH19-CE clinical spectrum to sporadic cases; confirmed female predominance; established the cellular interference model as the pathogenic mechanism for the X-linked paradox."},
    {"ref": "Marini 2010 Brain",
     "title": "Clinical features and outcome of PCDH19 epilepsy",
     "relevance": "Comprehensive clinical characterisation of PCDH19-CE (n=35): fever-triggered clusters, age of onset, cognitive outcomes, ASD prevalence, LTG worsening cases, bromide effectiveness — foundational natural history study."},
    {"ref": "Kolc 2019 GenetMed",
     "title": "A systematic review and meta-analysis of 271 PCDH19-variant individuals identifies psychiatric comorbidities, and divides the syndrome into three groups",
     "relevance": "Largest systematic review of PCDH19-CE (n=271): three clinical severity groups (mild/moderate/severe), psychiatric comorbidity prevalence (ASD 60%, ADHD 35%), variant class-phenotype correlations — current evidence basis for PCDH19-CE management."},
    {"ref": "Specchio 2019 EpilDepis",
     "title": "International PCDH19 Task Force consensus and future perspectives on this complex epileptic encephalopathy",
     "relevance": "First international expert consensus on PCDH19-CE management: CLB as first-line, bromide as adjunct, diagnostic workup protocol, genetic counselling standards — currently adopted by PCDH19 Alliance and major epilepsy centres."},
    {"ref": "Lagae 2022 Epilepsia",
     "title": "Fenfluramine in PCDH19 clustering epilepsy: an open-label phase 3 study",
     "relevance": "Phase 3 trial establishing fenfluramine (Fintepla) efficacy in PCDH19-CE: 74% ≥50% reduction in monthly cluster days, cluster-free periods extended — primary evidence for FDA 2022 approval and current Level B recommendation."},
]

# ── Patient Generator ──────────────────────────────────────────────────────────
_CATEGORY_POOL = [
    ("De-novo-PCDH19-truncating-frameshift", "truncating", 17),
    ("De-novo-PCDH19-missense", "missense", 14),
    ("De-novo-PCDH19-splice-site", "splice", 4),
    ("De-novo-PCDH19-CNV-deletion", "CNV-deletion", 3),
    ("Clinical-PCDH19-negative-phenocopy", "PCDH19-negative", 3),
]

_TREATMENTS_POOL = [
    "CLB+FFA", "CLB+VPA", "CLB+KBr", "CLB+FFA+VPA", "CLB+FFA+KD",
    "CLB+LEV", "CLB+VPA+KD", "CLB+KBr+VPA", "CLB+FFA+LEV+VPA",
]

_PHASES = ["Infancy-cluster-onset", "Toddler", "Early-childhood", "School-age", "Adolescence", "Adulthood"]
_CONTROL = ["drug-resistant", "partial-control", "partial-control", "cluster-free", "cluster-free", "partial-control"]


def _gen_patients():
    random.seed(SEED)
    patients = []
    pid = 1
    for cat, fclass, n in _CATEGORY_POOL:
        for _ in range(n):
            onset_months = random.randint(6, 36) if "PCDH19-negative" not in cat else random.randint(6, 48)
            age_months = random.randint(12, 96)
            # All PCDH19-CE patients are female (hallmark)
            sex = "F" if "PCDH19-negative" not in cat else random.choice(["F", "F", "F", "M"])  # 75% F for phenocopy
            phase = random.choice(_PHASES[:4]) if age_months < 60 else random.choice(_PHASES[2:])
            control = random.choice(_CONTROL)
            tx = random.choice(_TREATMENTS_POOL)
            norclobazam_level = round(random.uniform(60, 280), 1) if "CLB" in tx else None
            vpa_level = round(random.uniform(52, 98), 1) if "VPA" in tx else None
            bromide_level = round(random.uniform(1.1, 2.4), 2) if "KBr" in tx else None
            ffa_on = "FFA" in tx
            kd_on = "KD" in tx
            asd_features = random.random() < 0.60
            adhd = random.random() < 0.35
            catamenial = sex == "F" and age_months > 144 and random.random() < 0.28
            cluster_free_months = random.randint(1, 6)
            clusters_per_month = round(random.uniform(0.5, 4.0), 1) if control != "cluster-free" else 0
            patients.append({
                "id": f"PC-{pid:03d}",
                "age_months": age_months,
                "sex": sex,
                "onset_age_months": onset_months,
                "category": cat,
                "functional_class": fclass,
                "disease_phase": phase,
                "current_treatment": tx,
                "seizure_control": control,
                "norclobazam_level_ngml": norclobazam_level,
                "vpa_level_mgL": vpa_level,
                "bromide_level_mmolL": bromide_level,
                "ffa_on": ffa_on,
                "kd_on": kd_on,
                "asd_features": asd_features,
                "adhd": adhd,
                "catamenial": catamenial,
                "cluster_free_months": cluster_free_months,
                "clusters_per_month": clusters_per_month,
            })
            pid += 1
    random.shuffle(patients)
    return patients


# ── Public API Functions ──────────────────────────────────────────────────────
def get_overview():
    pts = _gen_patients()
    truncating = sum(1 for p in pts if p["functional_class"] == "truncating")
    missense = sum(1 for p in pts if p["functional_class"] == "missense")
    dre = sum(1 for p in pts if p["seizure_control"] == "drug-resistant")
    cluster_free = sum(1 for p in pts if p["seizure_control"] == "cluster-free")
    ffa_on = sum(1 for p in pts if p["ffa_on"])
    kd_on = sum(1 for p in pts if p["kd_on"])
    asd = sum(1 for p in pts if p["asd_features"])
    catamenial = sum(1 for p in pts if p["catamenial"])
    female = sum(1 for p in pts if p["sex"] == "F")

    return {
        "syndrome": "PCDH19 Clustering Epilepsy (PCDH19-CE)",
        "gene": "PCDH19",
        "chromosome": "Xq22.1",
        "protein": "Protocadherin-19 (delta2-protocadherin, cadherin superfamily)",
        "inheritance": "X-linked — heterozygous FEMALES affected; hemizygous males UNAFFECTED carriers (Cellular Interference)",
        "omim_disease": "300088",
        "ilae_name": "PCDH19 Clustering Epilepsy (PCDH19-CE) — ILAE 2022",
        "former_name": "EFMR (Epilepsy and Mental Retardation limited to Females)",
        "eeg_hallmark": "Fever-triggered focal seizure clusters; frontotemporal ictal onset; cluster-free intervals of weeks–months",
        "key_biomarker": "PCDH19 LOF variant on trio exome/WGS; cellular interference confirmed by variant in heterozygous female",
        "n_patients": 41,
        "kpis": {
            "female_pct": round(female / 41 * 100),
            "truncating_pct": round(truncating / 41 * 100),
            "missense_pct": round(missense / 41 * 100),
            "dre_pct": round(dre / 41 * 100),
            "cluster_free_pct": round(cluster_free / 41 * 100),
            "ffa_on_pct": round(ffa_on / 41 * 100),
            "kd_on_pct": round(kd_on / 41 * 100),
            "asd_pct": round(asd / 41 * 100),
            "catamenial_pct": round(catamenial / 41 * 100),
        },
        "clinical_alerts": [
            "⚡ FEVER ACTION PLAN MANDATORY at first visit — fever triggers 95% of clusters; written CLB escalation + rescue buccal midazolam plan must be issued before patient leaves clinic.",
            "🚨 FINTEPLA REMS REQUIRED before prescribing fenfluramine (USA) — echocardiogram q6M mandatory; prescriber + patient enrolment in REMS before first prescription.",
            "⚠️ POLG EXCLUSION MANDATORY before VPA — fatal hepatic failure risk in POLG-mutant infants; POLG panel must be NEGATIVE before starting VPA.",
            "⚠️ LAMOTRIGINE — AVOID as first-line in PCDH19-CE; case reports of worsening cluster frequency; use CLB, VPA, LEV, or fenfluramine instead.",
            "⚡ FATHER TESTING MANDATORY: test father for PCDH19 carrier status (X-linked inheritance); carrier father = 50% daughter risk of PCDH19-CE.",
            "🚨 NPO/SURGERY: IV midazolam bridge + IV VPA if applicable; alert surgical team of PCDH19-CE cluster risk; written perioperative AED plan.",
        ],
        "etiologies": [{"etiology": e["etiology"], "pct": e["pct"], "n": e["n"]} for e in ETIOLOGY_CATALOG],
        "seizure_type_prevalence": {s["type"]: s["prevalence_pct"] for s in SEIZURE_TYPES},
        "trigger_seizure_rates": {t["trigger"]: t["rate_pct"] for t in TRIGGERS},
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "key_aha": (
            "PCDH19-CE: the cellular interference paradox (females affected, males unaffected) means "
            "PCDH19 testing of the father is MANDATORY to determine inheritance and recurrence risk. "
            "Fever action plan at first visit is the single most impactful management intervention — "
            "95% of clusters are fever-triggered and early CLB rescue prevents escalation to cluster status. "
            "Fenfluramine (Fintepla, FDA 2022) is now the first disease-specific approved agent — initiate "
            "after CLB-partial response with mandatory cardiac monitoring via FINTEPLA REMS."
        ),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_breakdown():
    pts = _gen_patients()
    return {
        "patients": pts,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": AED_MONITORING,
        "n_patients": 41,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_definitions():
    return {
        "absolute_contraindications": ABSOLUTE_CI,
        "thresholds": THRESHOLDS,
        "concepts": CONCEPTS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
