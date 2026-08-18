"""
MTOR Epilepsy — Mechanistic Target of Rapamycin / mTORopathy Apex
Focal Cortical Dysplasia IIb / Hemimegalencephaly / MCAP / Smith-Kingsmore Syndrome
Somatic-GOF / Rapamycin-Everolimus Precision / mTORC1-Direct-Target / 1p36.22
================================================================================
40-patient cohort · MTOR (1p36.22) · GOF (somatic mosaic or germline) / AD rare
Gene OMIM: *601231 · MCAP: #615108 · Smith-Kingsmore: #616638

KEY MTOR BIOLOGY — mTOR KINASE / mTORC1 / APEX OF mTORopathy PATHWAY:
MTOR (1p36.22) encodes mTOR (mechanistic Target Of Rapamycin), a 2549 aa serine/
threonine kinase that forms the catalytic core of TWO distinct complexes:
  · mTORC1 (rapamycin-SENSITIVE): mTOR + RAPTOR + mLST8 + PRAS40 + DEPTOR
    → phosphorylates S6K1 (T389) and 4EBP1 (T37/46/70/65) → protein synthesis,
      cell growth, ribosome biogenesis, suppression of autophagy/lysosome
  · mTORC2 (rapamycin-INSENSITIVE, partially sensitive long-term):
    mTOR + RICTOR + mLST8 + mSin1 + DEPTOR → phosphorylates AKT (S473),
    SGK1, PKCα → cell survival, cytoskeletal organisation, glucose uptake

UPSTREAM PATHWAY (complete mTORopathy chain):
  PI3K → PDK1 → AKT → TSC2 (inhibits) → RHEB → mTORC1 (activates)
  AKT also directly phosphorylates PRAS40 (MTOR inhibitory partner) → mTORC1 activation
  GATOR1 (DEPDC5-NPRL2-NPRL3) → Rag GTPases → mTORC1 (amino-acid sensing arm)
  TSC1/TSC2 → mTORC1 (growth factor arm)
  MTOR = convergence point of ALL these inputs

GOF MECHANISM — SOMATIC MTOR MUTATIONS:
  · Most epilepsy-associated MTOR mutations are SOMATIC (mosaic), not germline
  · Arise during cortical development (post-zygotic) → clonal expansion in developing cortex
  · VAF (variant allele fraction): 0.5–40% in brain tissue; often <5% in blood →
    GERMLINE TESTING FREQUENTLY NEGATIVE → REQUIRES deep sequencing (ddPCR, NGS >1000×)
    or testing of surgical resection specimen
  · GOF mechanism: mutations in FAT domain (activation loop), kinase domain, or
    FRB domain → constitutive mTORC1 activity regardless of nutrient/growth factor status
  · Key mutations:
    - E1799K (FAT domain): most common MCAP mutation; strong GOF, macrocephaly + PMG
    - L2427P, C1483Y (kinase domain): focal epilepsy + FCD IIb; less structural
    - S2215F, F2208L (activation segment): FCD IIb; somatic in resected tissue
    - T1977I, D2357H: Smith-Kingsmore; germline GOF; macrocephaly + DEE
  · DIRECT TARGET OF RAPAMYCIN: FKBP12-rapamycin complex binds FRB domain of mTOR
    → allosteric mTORC1 inhibition → DIRECT precision therapy for MTOR GOF

NEUROLOGICAL CONSEQUENCES:
  · Prenatal mTORC1 hyperactivation → excess ribosome biogenesis → aberrant cell growth:
    - Cytomegalic neurons (2-3× normal soma size) with dysmorphic dendritic trees
    - Balloon cells (FCD IIb hallmark): enlarged, vacuolated, aberrant morphology;
      mislocated to white matter (heterotopia); fail to develop synaptic connectivity
    - Disrupted cortical lamination (failure of inside-out neuronal migration)
    - Increased Nav1.6, reduced Kv1.1 → membrane hyperexcitability → ictal discharges
  · Focal (FCD IIb) vs widespread (hemimegalencephaly, MCAP) depending on developmental
    timing and fraction of cells affected (focal early → more widespread)

MTOR vs upstream mTORopathies (TSC1/TSC2/DEPDC5/NPRL2/NPRL3):
  Feature               TSC1/TSC2       DEPDC5/NPRL2/3      MTOR (GOF)
  Pathway arm           PI3K-AKT arm    GATOR1-Rag arm      Convergence point
  Mutation type         Germline (AD)   Germline (AD)       Mostly somatic (de novo)
  VAF in blood          ~50%            ~50%                Often <5% (mosaic)
  Brain malformation    Tubers (bilateral) FCD (focal)      FCD IIb / HME / MCAP
  Multi-organ           Yes (TSC skin/  No (brain only)     Rarely (MCAP vascular)
                        kidney/lung)
  mTOR inhibitor        Everolimus      Everolimus (off-Rx) Everolimus (direct target)
  Surgery candidacy     50% (tubers)    65% (focal FCD)     60-70% (FCD/HME resect)
  Deep sequencing need  No (blood OK)   No (blood OK)       YES (often blood negative)

GENETICS:
  Gene:        MTOR (1p36.22)
  Inheritance: Usually somatic mosaic (de novo in developing cortex); germline GOF = AD
  pLI:         0.99 (strongly intolerant of heterozygous LOF — essential gene)
  VAF blood:   0.5-40% (often <5% → requires ddPCR or ultra-deep NGS >1000× coverage)
  OMIM:        *601231 (gene) · #615108 (MCAP) · #616638 (Smith-Kingsmore)
  First GOF epilepsy report: Lim 2015 (Science — somatic MTOR in FCD IIb);
                              Mirzaa 2016 (Nat Genet — MTOR GOF in MCAP)

KEY PHARMACOLOGICAL DISTINCTIONS:
  (1) EVEROLIMUS IS THE DIRECT PRECISION THERAPY: FKBP12-rapamycin analogue binds FRB
      domain of mTOR → allosteric inhibition of mTORC1 → DIRECT target of GOF mutant.
      Neurological dose 2.5-5 mg/day (trough 3-7 ng/mL) vs oncology 10-15 ng/mL.
      EXIST-3 (TSC, NEJM 2016): 50% responders; extrapolated to MTOR focal epilepsy.
  (2) SOMATIC DETECTION CRITICAL: blood-based panel often negative → requires EITHER
      ultra-deep NGS (>1000× read depth, ddPCR VAF >0.5%) on blood/saliva, OR testing
      of surgical resection specimen (gold standard: 99% detection if lesion resected).
      Always order tissue sequencing on resected FCD IIb/HME regardless of blood result.
  (3) TGB ABSOLUTE CI: Tiagabine → NCSE (non-convulsive SE) in focal cortical dysplasia →
      catastrophic in FCD IIb backdrop of MTOR mutations.
  (4) VPA + POLG1 MANDATORY: Mitochondrial disease (Alpers) can mimic mTOR epilepsy with
      cerebellar involvement — POLG1 screen before VPA mandatory.
  (5) PHT MAINTENANCE HIGH RISK: long-term phenytoin causes Purkinje cell toxicity →
      compounds any cerebellar component; also induces CYP3A4 → reduces everolimus levels
      (CYP3A4-mediated) — interaction critical.
  (6) RAPAMYCIN + CYP3A4 INHIBITORS: ketoconazole, itraconazole, clarithromycin, ritonavir
      → >10× everolimus exposure → toxicity (immunosuppression, haematological).
      ALSO: grapefruit juice inhibits intestinal CYP3A4 → everolimus absorption increases.
  (7) LIVE VACCINES CONTRAINDICATED on everolimus (immunosuppression): MMR, varicella,
      BCG, typhoid oral — check immunisation status BEFORE starting everolimus.
  (8) SURGERY SYNERGISTIC WITH EVEROLIMUS: resection removes primary epileptogenic zone;
      everolimus suppresses residual mTORC1 activity in margin tissue — combination
      superior to either alone in case series.
  (9) STOMATITIS MOST COMMON SE: Grade 2+ stomatitis in 50% → steroid mouthwash
      (dexamethasone 0.1 mg/mL) + dose reduction. NOT conventional antifungal (not Candida).
  (10) MCAP CAPILLARY MALFORMATIONS: not epileptogenic themselves but indicate somatic
       MTOR mosaicism → brain MRI mandatory even if skin lesions are only presentation.
"""

import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG — 5-CLASS SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "Somatic-MTOR-GOF-FCD-IIb",
        "pct": 35,
        "mechanism": (
            "Post-zygotic somatic MTOR GOF mutation during cortical development "
            "→ clonal expansion in developing neocortex → focal cortical dysplasia type IIb. "
            "Key mutations: S2215F, F2208L, L2427P, C1483Y — all in kinase domain/activation segment. "
            "VAF: 1-20% in blood (often undetectable <5%); 5-40% in resected brain tissue. "
            "FCD IIb hallmarks: cytomegalic neurons + balloon cells + disrupted lamination → "
            "constitutive mTORC1 → ↑Nav1.6, ↓Kv1.1 → ictal discharges. "
            "MRI: focal cortical thickening, FLAIR signal, blurred grey-white junction, "
            "'transmantle sign' (white matter FLAIR extending to ventricle). "
            "Variable loci: frontal (45%), temporal (28%), parietal (18%), occipital (9%). "
            "SURGICAL: best surgical candidate — resection of FCD gives 60-70% Engel I. "
            "Everolimus adjunct for residual disease/margin tissue."
        ),
        "eeg": "Focal high-frequency oscillations (HFOs: ripples 80-250 Hz, fast ripples 250-500 Hz) at FCD zone; ictal: focal fast onset (beta/gamma 15-40 Hz) at lesion; interictal: spike/sharp-wave over lesion; stereo-EEG (SEEG) gold standard for zone delineation",
        "onset_months": "0–120 months (neonatal to 10y; peak 12-36 months; rare adult onset with milder mutations)",
        "severity": "High — 70% drug-resistant; surgical cure feasible in 60-70% when FCD resected; everolimus adjunct reduces residual seizures",
    },
    {
        "category": "Germline-MTOR-GOF-Smith-Kingsmore",
        "pct": 28,
        "mechanism": (
            "De novo germline heterozygous MTOR GOF mutations → Smith-Kingsmore Syndrome (OMIM #616638). "
            "Key mutations: T1977I, D2357H, E2419K — often in FAT domain or activation loop. "
            "VAF ~50% in blood (germline, detectable by standard sequencing). "
            "Phenotype: macrocephaly (OFC >+3 SD), intellectual disability (moderate-severe), "
            "DEE (drug-resistant epilepsy onset <12 months), autism spectrum features, "
            "hyperkinetic movement disorder. "
            "Subtle dysmorphic features (high forehead, prominent metopic ridge, downslanting palpebral fissures). "
            "MRI: megalencephaly, simplified gyral pattern, delayed myelination; "
            "NO cortical dysplasia (germline = symmetric bilateral); ventricular enlargement. "
            "DISTINCTION FROM MCAP: Smith-Kingsmore → no capillary malformations, no PMG; "
            "MCAP → somatic, asymmetric, skin lesions. "
            "Everolimus beneficial in case series (reduces seizure burden in 50-60% DEE cases)."
        ),
        "eeg": "Hypsarrhythmia in IS phase; multifocal epileptiform discharges; modified hypsarrhythmia; background slowing; burst-suppression in severe DEE; sleep EEG: NREM disruption, sleep spindle reduction",
        "onset_months": "0–12 months (infantile — macrocephaly at birth, IS by 3-9 months)",
        "severity": "High — DEE with moderate-severe ID; some improvement with everolimus; seizures often partially controlled with AED combinations",
    },
    {
        "category": "Somatic-MTOR-GOF-HME",
        "pct": 18,
        "mechanism": (
            "Hemimegalencephaly (HME): unilateral hemispheric overgrowth from early somatic MTOR GOF. "
            "Developmental timing: very early post-zygotic (first few cell divisions) → "
            "large fraction of hemisphere affected. "
            "Key mutations: E1799K (mild HME), T1977K, H2189Y. "
            "VAF: very low in blood (<2%); high in affected hemisphere (20-40%). "
            "Clinical: neonatal-onset seizures, profound ID, hemiplegia contralateral to HME hemisphere. "
            "EEG: near-continuous ictal activity in affected hemisphere ('electrographic storm'); "
            "rapid evolution to Ohtahara or West syndrome. "
            "MRI: unilateral hemispheric enlargement, gyrification abnormality, "
            "white matter signal change; contralateral hemisphere usually normal. "
            "TREATMENT: early hemispherotomy (disconnective surgery) is treatment of choice; "
            "60-70% seizure-free post-hemispherotomy. "
            "Everolimus: role in post-surgical residual; reduces frequency before surgical decision."
        ),
        "eeg": "Near-continuous hemispheric ictal activity; suppression-burst pattern; ictal EEG: beta/gamma bursts from affected hemisphere; interictal: persistent epileptiform discharges unilateral; contralateral hemisphere normal background",
        "onset_months": "0–3 months (neonatal to early infantile; presentation at birth with hemiplegia + seizures)",
        "severity": "Severe — neonatal refractory seizures; DEE; hemispherotomy curative in 60-70%; profound ID in non-operated cases",
    },
    {
        "category": "Somatic-MTOR-GOF-MCAP",
        "pct": 12,
        "mechanism": (
            "MCAP (Megalencephaly-Capillary Malformation-Polymicrogyria, OMIM #615108). "
            "Somatic MTOR GOF (E1799K most common; also H2189Y, E1803K). "
            "Developmental timing: slightly later than HME → bilateral but asymmetric involvement. "
            "Features: bilateral (asymmetric) megalencephaly + polymicrogyria (PMG) + "
            "cutaneous capillary malformations (port-wine stains, telangiectasias — "
            "on body midline, extremities; diagnostic at birth) + syndactyly (2nd-3rd toes). "
            "Seizure onset: 3-18 months; focal > generalised; drug-resistant in 65%. "
            "MRI: bilateral PMG (frontoparietal predominant), asymmetric megalencephaly, "
            "cerebellar tonsillar ectopia (50%), ventriculomegaly. "
            "KEY DIAGNOSTIC CLUE: SKIN CAPILLARY MALFORMATIONS indicate somatic MTOR → "
            "MRI brain mandatory even with skin-only presentation. "
            "Everolimus: benefits both seizure burden AND possibly vascular component (mTOR in angiogenesis). "
            "Surgery: limited (bilateral PMG); palliative (VNS, CC, KD)."
        ),
        "eeg": "Bilateral multifocal epileptiform discharges; may show bilateral independent temporal or frontal foci; ictal: focal with secondary generalisation; hypsarrhythmia if West syndrome evolution; PMG-related slow background",
        "onset_months": "3–18 months (identified at birth by skin lesions; seizures 3-18 months; polymicrogyria worsens apparent at MRI by 6-12 months)",
        "severity": "High — drug-resistant in 65%; bilateral cortical malformation limits surgery; everolimus provides 30-50% seizure reduction; palliative measures primary",
    },
    {
        "category": "Phenocopy-mTOR-Pathway-Negative",
        "pct": 7,
        "mechanism": (
            "FCD IIb / HME / MCAP-like phenotype with negative MTOR sequencing (blood + tissue). "
            "Possible causes: (1) PIK3CA GOF somatic — upstream activator of PI3K→AKT→mTOR "
            "(ACTA2, Rivière 2012 Nat Genet); (2) AKT3 somatic GOF (hemimegalencephaly); "
            "(3) PIK3R2 somatic GOF (megalencephaly-PIK3R2); "
            "(4) RHEB GOF somatic (upstream mTORC1 activator, between TSC2 and mTORC1); "
            "(5) PTEN somatic LOF (disinhibits PI3K → AKT → mTOR). "
            "Clinical management: everolimus empirically for mTOR-pathway phenotype while "
            "extending sequencing (PIK3CA, AKT3, PIK3R2, RHEB, PTEN panel); "
            "surgery for resectable FCD lesion regardless of molecular confirmation."
        ),
        "eeg": "FCD IIb pattern: focal HFOs, ictal fast onset; same as MTOR-confirmed FCD IIb; molecular confirmation changes precision therapy not surgical approach",
        "onset_months": "Variable (0-120 months)",
        "severity": "Variable; same approach as confirmed MTOR pending molecular workup",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE CATALOG — 5 TYPES
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_CATALOG = [
    {
        "type": "Focal-Impaired-Awareness",
        "pct": 82,
        "eeg": "Focal onset at FCD zone: fast gamma/beta (15-40 Hz) → theta/delta spread; interictal: spike-wave at lesion; HFOs on SEEG (ripples + fast ripples define epileptogenic zone precisely)",
        "semiology": "Arrest of activity + stare + automatisms (oroalimentary, manual); duration 30-120 sec; post-ictal confusion; variable depending on lobe (frontal: hypermotor, brief; temporal: prolonged, déjà vu; parietal: sensory; occipital: visual aura). Preserved awareness in some parietal/occipital.",
        "clinical_tip": "Focal seizure onset = FCD zone until proven otherwise. If MRI negative → FDG-PET (hypometabolism at FCD in 70%), ictal SPECT (hyperperfusion), MEG (magnetic source imaging). SEEG for extra-temporal eloquent cortex or discordant non-invasive data. HFO mapping on SEEG defines resection margins.",
    },
    {
        "type": "Focal-to-Bilateral-Tonic-Clonic",
        "pct": 75,
        "eeg": "Focal onset (as above) → rapid bilateral spread; bisynchronous generalisation; post-ictal generalised background attenuation",
        "semiology": "Focal onset (aura or focal seizure) → tonic phase → clonic phase → post-ictal Todd's paresis (contralateral to FCD). Duration 1-3 min. Most common presentation prompting medical attention.",
        "clinical_tip": "Todd's paresis duration correlates with FCD size and seizure duration. Recurrent BTCSs = SUDEP risk → review AED adherence, address sleep, avoid alcohol. VPA choice for BTCSs must be preceded by POLG1 screen.",
    },
    {
        "type": "Infantile-Spasms-HME",
        "pct": 55,
        "eeg": "Modified hypsarrhythmia (asymmetric — worse on HME side); unilateral suppression-burst; electrodecrement at spasm onset (ictal); HME hemisphere often continuously abnormal",
        "semiology": "Typical IS: sudden symmetrical flexion of neck/trunk/arms, brief (2-5 sec), clusters of 10-50 spasms on waking. In HME: often ASYMMETRIC (contralateral head/arm deviation = focal component). Regression of developmental milestones in 70%.",
        "clinical_tip": "Asymmetric IS in first 6 months + hemiplegia → rule out HME (MRI urgently). HME-related IS: ACTH+VGB (UKISS protocol) has limited success (30% vs 75% in symmetric IS) → hemispherotomy evaluation urgently in refractory HME-IS (ideal window <12 months for neuroplasticity).",
    },
    {
        "type": "Tonic-Seizures",
        "pct": 45,
        "eeg": "Frontal fast activity (10-25 Hz); electrodecrement followed by low-voltage fast; may have secondarily bilateral spread; EEG during sleep tonic = better characterised",
        "semiology": "Sustained tonic (rigid) posturing of trunk and extremities; bilateral symmetric or asymmetric; duration 10-60 sec; may be nocturnal (frontal lobe origin from FCD). Tonic seizures in DEE context = poor prognosis marker.",
        "clinical_tip": "Tonic seizures from frontal FCD may cluster nocturnally (mimic ADNFLE). Distinguish by MRI (FCD visible) and SEEG. CBZ/LCM effective for frontal tonic seizures from FCD. CLB adjunct at bedtime for nocturnal clustering.",
    },
    {
        "type": "Epileptic-Spasms-Persistent",
        "pct": 38,
        "eeg": "High-amplitude slow wave + fast activity + electrodecrement pattern; modified hypsarrhythmia persisting beyond typical IS age (>24 months); often evolves to Lennox-Gastaut",
        "semiology": "Persistent epileptic spasms beyond infancy (>18-24 months); may coexist with other seizure types; associated with severe ID. Occurs in HME and MCAP subtypes predominantly.",
        "clinical_tip": "Persistent spasms in MTOR phenotype beyond 18 months → urgent hemispherotomy evaluation (HME) or everolimus initiation (bilateral MCAP). ACTH re-trial rarely effective after first year. VNS palliation if surgery not feasible.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER CATALOG — 8 TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_CATALOG = [
    {
        "trigger": "Fever-Intercurrent-Illness",
        "pct": 88,
        "mechanism": "Inflammatory cytokines (IL-1β, IL-6, TNF-α) → NF-κB activation → mTORC1 pathway sensitisation → enhanced excitability in already-hyperexcitable FCD tissue. Fever also directly increases cortical excitability (temperature-dependent Na+ channel kinetics). Threshold often <38°C in FCD epilepsy.",
        "management": "Pre-emptive CLB 0.25-0.5 mg/kg at fever onset (written protocol for caregivers); antipyretics (paracetamol/ibuprofen); avoid delayed treatment. Note: ibuprofen/NSAIDs may have weak mTOR-modulating effects (investigational).",
    },
    {
        "trigger": "Sleep-Deprivation",
        "pct": 75,
        "mechanism": "Sleep loss → increased cortical excitability → reduces seizure threshold in already epileptogenic FCD cortex. NREM-slow-wave-sleep deficiency impairs inter-ictal suppression mechanisms. In Smith-Kingsmore DEE: sleep architecture profoundly disrupted (mTORC1-dependent synaptic plasticity).",
        "management": "Strict sleep schedule; melatonin 2-5 mg for sleep initiation in DEE/autism context; avoid sleep deprivation; school/exam adjustments",
    },
    {
        "trigger": "Missed-AED",
        "pct": 68,
        "mechanism": "Any AED gap → rapid seizure recurrence in drug-resistant FCD. Drug-resistant MTOR epilepsy patients often have narrow therapeutic windows. Missed everolimus → loss of mTORC1 suppression within 24-48h (half-life ~30h, but steady-state effect wanes). Missed AED also precipitates status epilepticus risk.",
        "management": "Adherence review + caregiver training; simplified regimens; rescue midazolam protocol for clusters; written SE action plan",
    },
    {
        "trigger": "Psychological-Physical-Stress",
        "pct": 62,
        "mechanism": "HPA axis activation → cortisol → mTORC1 pathway interaction (cortisol has complex, context-dependent mTOR effects); catecholamines → β-adrenergic → cAMP → PKA → mTOR pathway modulation. Cellular stress activates mTORC1 (paradoxically — mTOR senses anabolic conditions including stress response molecules).",
        "management": "Psychological support; CBT/mindfulness; stress management programme; avoid extreme physical exertion; CLB PRN for anticipated high-stress events",
    },
    {
        "trigger": "AED-Taper-Too-Fast",
        "pct": 52,
        "mechanism": "Premature AED tapering (especially after prolonged seizure-free period) removes inhibitory protection. FCD IIb cortex remains epileptogenic regardless of seizure freedom — structural hyperexcitability is permanent until resected. 'Seizure freedom = AED masking, not cure' in non-operated FCD.",
        "management": "AED tapering ONLY after confirmed surgical cure (Engel I >24 months post-resection); never taper empirically in non-surgical FCD patients; discuss realistic tapering expectations",
    },
    {
        "trigger": "Photosensitivity-FCD",
        "pct": 38,
        "mechanism": "Occipital FCD IIb can generate photosensitive epilepsy via involvement of occipital visual cortex. Hyperexcitable occipital cortex responds to photic stimulation with ictal discharge. Less common in frontal/temporal FCD. Photoparoxysmal response on EEG confirms.",
        "management": "Avoid flickering lights, video games, strobe; polarised lenses if occipital FCD; LEV particularly effective for photosensitive component",
    },
    {
        "trigger": "Everolimus-Adjustment-Period",
        "pct": 28,
        "mechanism": "During everolimus initiation/dose changes: period of subtherapeutic mTORC1 suppression → rebound mTORC1 hyperactivity → temporary increase in seizure frequency. Occurs in first 1-4 weeks of dose titration. Also: everolimus + CYP3A4 drug interactions → level fluctuations → breakthrough seizures.",
        "management": "Slow everolimus titration (start 1.5 mg/day, titrate over 4 weeks); monitor trough at 2 weeks then monthly; check drug interactions (especially antiepileptics that induce CYP3A4: CBZ, PHT, OXC — all reduce everolimus by 70-80%)",
    },
    {
        "trigger": "Hormonal-Catamenial",
        "pct": 22,
        "mechanism": "In post-pubertal females: estrogen (pro-excitatory) peaks at ovulation → reduced GABA-A sensitivity → increased FCD excitability. Progesterone (neuroactive steroid, GABA-A modulator) withdrawal at menses → seizure cluster. mTOR pathway interacts with hormonal signalling (estrogen activates PI3K→AKT→mTOR).",
        "management": "VPPP counselling (avoid VPA in females ≥12y); CLB pre-menstrual; progesterone supplementation (investigational); MHRA VPPP 2021 programme for contraception",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT CATALOG — 8 TREATMENTS
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_CATALOG = [
    {
        "drug": "Everolimus",
        "level": "Level B",
        "mechanism": "Allosteric mTORC1 inhibitor: FKBP12-everolimus complex binds FRB (FKBP12-Rapamycin Binding) domain of mTOR → conformational change → prevents RAPTOR-mTOR interaction → mTORC1 inactivation → ↓S6K1/4EBP1 phosphorylation → ↓protein synthesis → ↓cytomegalic neuron/balloon cell hyperexcitability. DIRECT molecular target of GOF MTOR mutation.",
        "dose": "Start 1.5-2.5 mg/day; titrate to 2.5-5 mg/day over 4 weeks; target trough 3-7 ng/mL (neurological dose); measure trough 14 days after start and after each dose change; steady state 7-14 days",
        "efficacy": "50-65% seizure responders (≥50% reduction) in MTOR-pathway epilepsy; 10-20% seizure-free; EXIST-3 (TSC/NEJM 2016): median 29% median reduction vs 0% placebo; case series MTOR: 40-60% reduction",
        "monitoring": "Trough 3-7 ng/mL at 2 weeks, 4 weeks, then monthly; CBC (haematological); LFTs; lipids (triglycerides, cholesterol); blood glucose; stomatitis grading; wound healing (avoid major surgery ≤2 weeks after); renal function; infection symptoms; immunisation review (live vaccines CI before start)",
        "mtor_specific": "DIRECT mTOR TARGET — highest rational precision in MTOR GOF vs TSC1/TSC2 (upstream) or DEPDC5/NPRL2/NPRL3 (GATOR1 arm). CYP3A4 INTERACTIONS: CBZ, PHT, OXC reduce everolimus by 70-80% (CYP3A4 induction) — increase everolimus dose 2-3× if co-prescribed; monitor TDM closely. KETOCONAZOLE/AZOLES increase everolimus >10× — avoid; if necessary, reduce dose by 90%. STOMATITIS: most common SE (50%); use dexamethasone 0.1 mg/mL mouthwash prophylactically; NOT antifungal. SURGERY + EVEROLIMUS: hold 2 weeks pre-surgery (wound healing); resume post-healing.",
    },
    {
        "drug": "Levetiracetam",
        "level": "Level B",
        "mechanism": "SV2A (synaptic vesicle glycoprotein 2A) binding → modulates vesicle exocytosis → reduces release probability at hyperexcitable FCD synapses. POLG-safe. Also inhibits Ca²⁺ channel N-type and reduces AMPA-mediated currents. Effective for focal seizures and BTCSs.",
        "dose": "Start 10-15 mg/kg/day; titrate to 20-40 mg/kg/day (children); 500-3000 mg/day adults; twice daily",
        "efficacy": "40-55% responders in focal FCD epilepsy; 25-35% seizure-free as adjunct",
        "monitoring": "No TDM routinely required; watch for behavioural side effects (irritability, aggression, depression especially in ASD/DEE context — MTOR patients often have ASD); FBC if prolonged use; renal dose adjustment",
        "mtor_specific": "POLG-SAFE: use preferentially in MTOR patients who also need VPA-alternative due to POLG1 concern or VPA failure. BEHAVIOURAL MONITORING: MTOR epilepsy frequently overlaps with ASD/ID/ADHD — LEV-induced irritability (15-20%) may be misattributed to underlying condition; BRV (brivaracetam, same mechanism, fewer behavioural SEs) preferred if LEV behaviour issues arise. SV2A mechanism synergistic with everolimus (different targets).",
    },
    {
        "drug": "ACTH",
        "level": "Level A",
        "mechanism": "Synthetic ACTH (tetracosactide): MC2R (melanocortin receptor 2) on adrenal cortex → cortisol; also direct brain MC receptors (MC4R) → CRH suppression → reduced seizure threshold. In IS/West: most effective short-term treatment. UKISS 2004 (Lancet): ACTH superior to vigabatrin for IS control (2 weeks).",
        "dose": "0.5-1.5 mg/kg/day (tetracosactide depot IM) in two divided doses for 2-4 weeks; taper over 4-6 weeks. NHS UK: ACTH 40-80 IU/day synthetic ACTH (tetracosactide 0.5-1.5 mg/day)",
        "efficacy": "60-70% IS cessation within 2 weeks (symmetric IS); 30-40% in HME-associated IS (asymmetric, structural); higher success if treated <4 weeks from IS onset",
        "monitoring": "BP daily (hypertension, 40%); blood glucose (glucose testing QID during ACTH); weight (Cushingoid features); infection (immunosuppression — live vaccines CI); electrolytes (hypokalaemia, Na); sleep disruption (common); FBC (neutrophilia, lymphopenia)",
        "mtor_specific": "HME-RELATED IS: ACTH success LOWER (30-40%) due to structural unilateral cortical malformation — early hemispherotomy evaluation if ACTH fails at 2 weeks. MCAP/SMITH-KINGSMORE IS: standard ACTH protocol; follow with everolimus for ongoing epilepsy. TIMING: early treatment (<4 weeks onset) maximises success. Post-ACTH: transition to everolimus + 1-2 AEDs for maintenance.",
    },
    {
        "drug": "Valproate",
        "level": "Level B",
        "mechanism": "Broad-spectrum: Na+ channel stabilisation + GABA-T inhibition (↑GABA) + T-type Ca²⁺ channel inhibition + HCN channel modulation. Effective for GTCSs, tonic, myoclonic, and focal seizures — useful in mixed seizure types of Smith-Kingsmore/MCAP.",
        "dose": "Start 10-15 mg/kg/day; titrate to 20-60 mg/kg/day; TDM 50-100 mg/L",
        "efficacy": "50-65% seizure reduction in mixed epilepsy syndromes; 30-40% seizure-free as adjunct",
        "monitoring": "POLG1 MANDATORY before VPA (CPIC 2023 — Alpers syndrome risk: fatal hepatotoxicity); LFTs + ammonia q3 months; weight; platelets (thrombocytopenia); FBC; VPPP (valproate pregnancy prevention programme — MHRA 2021) for females ≥12y; teratogenicity counselling",
        "mtor_specific": "POLG1 SCREEN MANDATORY: MTOR epilepsy with cerebellar component or developmental regression can mimic POLG1/Alpers — always exclude POLG1 before VPA initiation. VPPP MANDATORY FOR FEMALES ≥12y: 10% risk of neural tube defects + 30-40% neurodevelopmental risk in offspring. CYP2C9 inhibition by VPA → increases some co-AED levels (LTG, PHT, PB).",
    },
    {
        "drug": "Ketogenic-Diet",
        "level": "Level B",
        "mechanism": "Metabolic switch: ketone bodies (β-hydroxybutyrate, acetoacetate) → multiple anti-seizure mechanisms: HCAR2 (GPR109A) activation → NF-κB suppression → reduced neuroinflammation (relevant in FCD IIb microenvironment); KATP channel opening → neuronal hyperpolarisation; inhibits mTOR pathway (BHB reduces mTORC1 activity independently — synergistic with everolimus); GABA increase.",
        "dose": "Classic 4:1 or 3:1 KD (fat:carbs+protein ratio); implemented by specialist dietitian; BHB target 2-4 mmol/L; micronutrient supplementation (selenium, vitamins); 3-month minimum trial",
        "efficacy": "40-55% responders in drug-resistant focal epilepsy; 15-25% seizure-free; particularly effective in young children <5y with DEE; mTOR-independent mechanism makes it additive to everolimus",
        "monitoring": "BHB (blood ketones) daily during initiation; micronutrients (selenium, zinc, Ca); lipids; FBC; growth parameters; renal stones (urinary citrate, bicarbonate supplement); QTc (rare complication); SUDEP monitoring (KD reduces SUDEP risk via autonomic stabilisation)",
        "mtor_specific": "mTOR PATHWAY SYNERGY: BHB directly inhibits mTORC1 (REDD1/AMPK-mediated) → additive effect with everolimus. Start KD + everolimus simultaneously in drug-resistant Smith-Kingsmore/MCAP. FEASIBILITY IN DEE: modified Atkins diet (less restrictive) may be preferred in ID/ASD context where strict compliance is difficult. KD + SURGERY: KD pre-surgical stabilisation; taper KD post-resection if seizure-free.",
    },
    {
        "drug": "Surgical-Resection-Hemispherotomy",
        "level": "Level A",
        "mechanism": "Removal or disconnection of epileptogenic FCD IIb / affected hemisphere eliminates structural seizure generator. No pharmacological agent. Gold standard for resectable FCD epilepsy. Hemispherotomy (HME): functional disconnection of affected hemisphere preserving vascular supply — hemispheric disconnection superior to anatomical hemispherectomy (less blood loss, less hydrocephalus).",
        "dose": "N/A — surgical procedure; pre-surgical evaluation: Video-EEG + 3T MRI + FDG-PET + interictal SPECT ± MEG ± SEEG; optimal age for HME: <12 months (neuroplasticity for language/motor transfer); FCD resection: any age",
        "efficacy": "FCD resection: 60-70% Engel I (seizure-free); HME hemispherotomy: 65-75% Engel I; MCAP (bilateral): palliative VNS/CC only",
        "monitoring": "Post-op MRI (day 1, 3 months, 1 year); EEG (3 months, annually); neuropsychological assessment (6 months, 2 years); language assessment post-hemispherotomy; hemiplegia rehabilitation; AED taper at 24 months if Engel I (discuss with neurologist)",
        "mtor_specific": "SURGERY IS PRIMARY TREATMENT FOR RESECTABLE MTOR-FCD: everolimus is adjunct, not substitute for surgery. TISSUE SEQUENCING: always send resected FCD specimen for somatic MTOR deep sequencing (gold standard — detects VAF too low for blood). EVEROLIMUS POST-OP: reduces residual margin epileptogenicity; start at 4-6 weeks post-op healing. HME WINDOW: ideal hemispherotomy <12 months (language lateralisation to intact hemisphere — Wada test at 5+y confirms transfer). MCAP bilateral: palliative approach (VNS, CLB, KD, CC).",
    },
    {
        "drug": "Vigabatrin",
        "level": "Level A",
        "mechanism": "GABA-T (GABA transaminase) irreversible inhibitor → ↑synaptic GABA → inhibition of hyperexcitable FCD cortex and subcortical circuits. First-line for IS (UKISS 2004, UKISS-VIGABATRIN arm). FDA SHARE-REMS: irreversible VFD (visual field defects) in 30-50% → mandatory ophthalmological monitoring.",
        "dose": "100-150 mg/kg/day children (IS); 40-80 mg/kg/day for other seizure types; BID dosing",
        "efficacy": "50-55% IS cessation in 2 weeks (UKISS); often used as ACTH + VGB combination (UKISS protocol); 35% seizure-free for focal epilepsy",
        "monitoring": "ERG (electroretinogram) every 3 months (SHARE REMS mandatory); visual field testing ≥5y (co-operative); REMS enrolment required (US); ophthalmology review at 3, 6, 12 months then 6-monthly; SD-OCT in infants to detect early retinal toxicity",
        "mtor_specific": "ACTH + VGB COMBINATION (UKISS PROTOCOL): standard for HME-IS and Smith-Kingsmore IS; 14-day ACTH + VGB to 12 months. VGB CAUTION IN BILATERAL LESION (MCAP): bilateral cortical malformation + VFD risk of VGB = caution in MCAP (bilateral vision impairment + VFD). ERG MANDATORY before continuing beyond 3 months. Post-IS control: transition to everolimus + LCM/LEV for maintenance.",
    },
    {
        "drug": "Clobazam",
        "level": "Level C",
        "mechanism": "1,5-benzodiazepine (1,5-BZD, not 1,4-BZD) → GABA-A positive allosteric modulator (preferential α2/α3 subunit binding vs classic BZDs: α1 → less sedation). Useful for focal seizure adjunct and acute seizure clusters. PRN use for fever/stress-triggered clusters.",
        "dose": "Adults: 10-40 mg/day; children 0.1-1 mg/kg/day (max 40 mg); bedtime or divided doses; PRN protocol: 0.25-0.5 mg/kg single dose",
        "efficacy": "30-50% adjunctive seizure reduction; 10-15% seizure-free as adjunct; useful for acute breakthrough management",
        "monitoring": "Sedation; tolerance (tachyphylaxis after 3-6 months); withdrawal seizures if abrupt stop; CYP2C19 interaction (fluoxetine/fluvoxamine → N-desmethyl-CLB 5× → toxicity); dental/respiratory in young children",
        "mtor_specific": "PRN FEVER PROTOCOL: written CLB protocol for caregivers; 0.25-0.5 mg/kg at fever >37.5°C → reduces febrile seizure clusters. PERIODIC DRUG HOLIDAY: to reduce tolerance; 1-2 weeks off every 3-4 months if used chronically. SMITH-KINGSMORE DEE: CLB useful but ASD/ID may impair compliance reporting of side effects.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS — 6 ABSOLUTE/HIGH RISK
# ─────────────────────────────────────────────────────────────────────────────
CI_CATALOG = [
    {
        "drug": "Tiagabine",
        "risk": "ABSOLUTE",
        "reason": "NCSE (non-convulsive status epilepticus) in focal cortical dysplasia. TGB selectively enhances synaptic GABA by blocking GAT-1 reuptake → but in FCD IIb, tonic inhibitory failure → TGB paradoxically precipitates NCSE. Multiple case reports of TGB-induced NCSE in FCD. AVOID in all MTOR-FCD epilepsy.",
    },
    {
        "drug": "VPA + POLG1-positive",
        "risk": "ABSOLUTE",
        "reason": "POLG1 (polymerase gamma) mutations → Alpers disease → VPA-induced fulminant hepatotoxicity (fatal). MTOR epilepsy phenotype (cerebellar, DEE, developmental regression) overlaps Alpers → POLG1 screen MANDATORY before VPA. If POLG1 positive: VPA contraindicated permanently.",
    },
    {
        "drug": "Everolimus + Live Vaccines",
        "risk": "ABSOLUTE",
        "reason": "Everolimus is an immunosuppressant (mTORC1 inhibition → T-cell suppression). Live attenuated vaccines (MMR, varicella, BCG, typhoid oral, yellow fever, rotavirus) → risk of disseminated vaccine-strain infection. Check and complete immunisation schedule BEFORE starting everolimus. No live vaccines while on everolimus.",
    },
    {
        "drug": "Phenytoin-Maintenance",
        "risk": "HIGH",
        "reason": "Long-term PHT: (1) Purkinje cell toxicity → cerebellar atrophy (compounds any MTOR-related cerebellar involvement); (2) CYP3A4 inducer → reduces everolimus levels 70-80% → requires 2-3× everolimus dose increase with TDM monitoring; (3) cognitive/sedation burden in DEE context. Acceptable for acute IV SE rescue only.",
    },
    {
        "drug": "Everolimus + Strong CYP3A4 Inhibitors",
        "risk": "HIGH",
        "reason": "Ketoconazole, itraconazole, voriconazole, clarithromycin, ritonavir → CYP3A4 inhibition → >10× everolimus exposure → severe immunosuppression, haematological toxicity, nephrotoxicity. Grapefruit juice also inhibits intestinal CYP3A4. If azole antifungal required (e.g., invasive fungal infection): reduce everolimus dose by 90%, intensive TDM.",
    },
    {
        "drug": "Carbamazepine-Oxcarbazepine + Everolimus",
        "risk": "HIGH",
        "reason": "CBZ and OXC are potent CYP3A4 inducers → reduce everolimus bioavailability by 70-80% (similar to PHT). Co-prescription requires everolimus dose increase 2-3× with intensive TDM (trough every 2 weeks until stable). Also: CBZ/OXC increase risk of SIADH which may complicate MTOR-related management. If CBZ used for focal seizures + everolimus: strict TDM protocol.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING CATALOG — 14 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_CATALOG = [
    {"item": "Somatic-MTOR-Deep-Sequencing", "frequency": "At diagnosis; tissue at surgery", "rationale": "Blood sequencing often negative (VAF <5%) → ultra-deep NGS (>1000× read depth) or ddPCR on blood/saliva; surgical tissue (gold standard: 99% sensitivity). Test both blood and tissue if surgery performed."},
    {"item": "Everolimus-Trough-TDM", "frequency": "Day 14, day 28, then monthly", "rationale": "Target 3-7 ng/mL (neurological, below oncology 10-15 ng/mL). Below 3 ng/mL → inadequate mTORC1 suppression. Above 7 ng/mL → immunosuppression + haematological toxicity. TDM essential with CYP3A4-interacting AEDs."},
    {"item": "MRI-Brain-3T", "frequency": "At diagnosis + 12 monthly", "rationale": "FCD IIb: transmantle sign, blurred GMW junction, focal cortical thickening, FLAIR signal. HME: unilateral hemispheric enlargement. MCAP: bilateral PMG + megalencephaly. Track lesion evolution and post-surgical changes."},
    {"item": "POLG1-Before-VPA", "frequency": "Once at diagnosis before VPA", "rationale": "POLG1 mutations → Alpers disease + VPA → fatal hepatotoxicity. CPIC 2023 Level A recommendation. MTOR phenotype overlaps Alpers (cerebellar, DEE, regression) → always screen before VPA."},
    {"item": "LFT-CBC-Lipids-Glucose", "frequency": "Baseline + q4 weeks × 3 months + q3 monthly", "rationale": "Everolimus toxicity panel: LFT elevation, anaemia, thrombocytopenia, hyperlipidaemia (triglycerides, cholesterol), hyperglycaemia. Dose reduce/stop if Grade 3-4 toxicity."},
    {"item": "VGB-ERG-REMS", "frequency": "Baseline + every 3 months if on VGB", "rationale": "VGB causes irreversible VFD (visual field defects, 30-50%). SHARE REMS mandatory (US). ERG (electroretinogram) for infants unable to do formal visual fields. SD-OCT for early retinal toxicity detection."},
    {"item": "Stomatitis-Grading", "frequency": "At each clinic visit (2-weekly initially)", "rationale": "Grade 1 (asymptomatic) → dexamethasone mouthwash prophylaxis; Grade 2 (symptomatic, no eating change) → dexamethasone + oral gel; Grade 3 (unable to eat) → dose reduction 50%; Grade 4 (life-threatening) → hold everolimus."},
    {"item": "OFC-Growth-Monitoring", "frequency": "Monthly <2y; 3-monthly thereafter", "rationale": "Head circumference tracking in MCAP/Smith-Kingsmore (macrocephaly progression). OFC >+3 SD = macrocephaly. MRI if rapid OFC increase (hydrocephalus, MCAP progression)."},
    {"item": "Neuropsychological-Developmental", "frequency": "6-monthly first 3 years, annually thereafter", "rationale": "Smith-Kingsmore DEE: moderate-severe ID, ASD features. HME: ID + hemiplegia + hemianopia. MCAP: variable ID. Track using age-appropriate tools (Bayley Scales <42 months, WPPSI 3-7y, WAIS adult)."},
    {"item": "Surgical-Pre-Evaluation", "frequency": "At 3+ AED failures (DRE threshold)", "rationale": "Video-EEG (seizure localisation), 3T MRI (lesion detection), FDG-PET (hypometabolism at FCD 70%), ictal SPECT (hyperperfusion), MEG (source imaging), neuropsychology. SEEG if non-invasive discordant or eloquent cortex."},
    {"item": "VPPP-Valproate-Pregnancy", "frequency": "From age 12y; annual counselling", "rationale": "VPA: 10% NTD risk + 30-40% neurodevelopmental risk. MHRA 2021 mandatory VPPP programme. All females ≥12y prescribed VPA must be enrolled. If conception planned: switch VPA before conception."},
    {"item": "SUDEP-Risk-Annual", "frequency": "Annual", "rationale": "MTOR drug-resistant focal epilepsy = elevated SUDEP risk (especially nocturnal tonic-clonic seizures, high seizure frequency). MORTEMUS study: SUDEP mechanism post-GTCS breathing failure. Discuss risk, supervision, CCTV, seizure detection devices, avoid alcohol/sleep deprivation."},
    {"item": "Immunisation-Review", "frequency": "Before everolimus start; annually", "rationale": "Live vaccines contraindicated on everolimus. Ensure MMR, varicella, all childhood vaccines complete BEFORE starting everolimus. Annual influenza (inactivated) safe and recommended. Pneumococcal and meningococcal (non-live) safe."},
    {"item": "EEG-Sleep-Study", "frequency": "At diagnosis; 6-monthly (HME); 12-monthly (FCD)", "rationale": "Sleep EEG: nocturnal seizure detection, IS confirmation, HFO identification. Post-surgical EEG: tracks epileptiform activity resolution. Smith-Kingsmore: sleep architecture assessment (mTORC1 affects sleep spindle generation)."},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE CATALOG — 6 STAGES
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_CATALOG = [
    {
        "stage": "Neonatal-Pre-symptomatic",
        "age": "0–4 weeks",
        "notes": "HME/MCAP: identified at birth (hemiplegia, asymmetric appearance, capillary malformations). Smith-Kingsmore: macrocephaly noted at birth (OFC >+3 SD). FCD IIb: may be asymptomatic. Action: urgent MRI (neonatal protocol — myelination adjusted); genetics panel (MTOR germline); neonatal EEG if seizures. Somatic testing: saliva + blood ultra-deep NGS.",
    },
    {
        "stage": "Infant-0-12M-IS",
        "age": "1–12 months",
        "notes": "IS peak (3-9 months). HME: asymmetric spasms → urgent hemispherotomy evaluation. Smith-Kingsmore: symmetric IS → ACTH + VGB (UKISS). MCAP: focal or symmetric IS. EEG: hypsarrhythmia (symmetric) or modified hypsarrhythmia (asymmetric). Start everolimus at 4-6 weeks after IS diagnosis (adjunct to ACTH/VGB). Developmental surveillance: Bayley Scales.",
    },
    {
        "stage": "Early-Childhood-1-5Y-FCD",
        "age": "1–5 years",
        "notes": "FCD IIb focal seizures emerge. Drug-resistant epilepsy (DRE) declared at 3+ AED failures. Pre-surgical evaluation initiated. KD trial for DRE. Everolimus titrated to trough 3-7 ng/mL. Hemispherotomy window: before 5y for optimal neuroplasticity (language, motor transfer). AED polypharmacy review. POLG1 screen before any VPA consideration.",
    },
    {
        "stage": "School-Age-5-12Y-DRE",
        "age": "5–12 years",
        "notes": "DRE established. Surgical candidacy re-evaluation (SEEG if needed). Everolimus maintenance. Neuropsychological assessment for educational planning (IEP). ASD/ADHD co-management in Smith-Kingsmore. VNS as palliative for non-surgical MCAP/bilateral cases. VPPP introduction for girls approaching puberty. SUDEP counselling for families. School seizure action plan.",
    },
    {
        "stage": "Adolescence-12-25Y-Transition",
        "age": "12–25 years",
        "notes": "Transition to adult services. VPPP programme mandatory for females ≥12y on VPA. Driving: DVLA/provincial rules (12 months seizure-free for car; different for HGV). Mental health: depression/anxiety in Smith-Kingsmore/HME adolescents. Everolimus long-term tolerance (lipids, immunosuppression). Surgical re-evaluation if prior palliative only. Fertility counselling (genetic — somatic MTOR: low recurrence risk; germline Smith-Kingsmore: 50% offspring risk).",
    },
    {
        "stage": "Adulthood-25Y-Long-term",
        "age": "25+ years",
        "notes": "Long-term everolimus maintenance (>5 years: watch for cumulative immunosuppression, secondary malignancy risk — low but real). AED simplification if seizure reduction maintained. Bone health (everolimus → mild osteopenia; calcium + vitamin D). Annual lipid panel. Post-surgical adults (Engel I): AED taper discussion at 2-5 years seizure-free. Smith-Kingsmore adults: supported living, employment assessment (ID services). SUDEP risk ongoing (especially BTCSs).",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY CONCEPTS — 15 ENTRIES
# ─────────────────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "MTOR-1p36.22", "definition": "MTOR gene (1p36.22) encodes the 2549 aa mTOR kinase — catalytic core of mTORC1 and mTORC2. GOF mutations (somatic or germline) → constitutive mTORC1 hyperactivation → FCD IIb, HME, MCAP, Smith-Kingsmore. OMIM *601231."},
    {"term": "mTORC1-mTORC2", "definition": "mTORC1 (RAPTOR complex, rapamycin-sensitive): phosphorylates S6K1+4EBP1 → protein synthesis/growth. mTORC2 (RICTOR complex, rapamycin-partially insensitive): phosphorylates AKT(S473)+SGK1. Epilepsy-relevant: mTORC1 hyperactivation in FCD/HME. Everolimus targets mTORC1 directly."},
    {"term": "GOF-Somatic-Mosaic", "definition": "Most MTOR epilepsy mutations are somatic (post-zygotic) → mosaic (only subset of cells affected). VAF 0.5-40% in brain; often <5% in blood → germline panel negative. Deep sequencing (>1000× NGS or ddPCR) of blood + surgical tissue required. Developmental timing determines extent (early → HME; late → focal FCD)."},
    {"term": "Smith-Kingsmore-Syndrome", "definition": "Germline MTOR GOF (e.g., T1977I, D2357H): macrocephaly (OFC >+3 SD), DEE (<12 months onset), moderate-severe ID, ASD features, mild dysmorphics. No cortical dysplasia (bilateral symmetric megalencephaly). Everolimus benefits 50-60% seizure burden. OMIM #616638."},
    {"term": "MCAP-Syndrome", "definition": "Megalencephaly-Capillary Malformation-Polymicrogyria. Somatic MTOR GOF (E1799K). Bilateral (asymmetric) megalencephaly + PMG + cutaneous capillary malformations (diagnostic — port-wine stains at midline/extremities) + syndactyly 2nd-3rd toes + cerebellar tonsillar ectopia. OMIM #615108. Surgery palliative (bilateral)."},
    {"term": "FCD-IIb-Somatic-MTOR", "definition": "Focal Cortical Dysplasia type IIb: cytomegalic neurons + balloon cells (MTOR hallmark) + disrupted cortical lamination. Located in any cortical region. 25% of FCD IIb have somatic MTOR mutation. Resection → 60-70% Engel I. Tissue sequencing at surgery mandatory."},
    {"term": "HME-Hemimegalencephaly", "definition": "Unilateral hemispheric overgrowth from very early somatic MTOR GOF. Neonatal-onset refractory seizures + hemiplegia + profound ID. EEG: hemispheric ictal activity. Treatment: hemispherotomy (ideal <12 months). Everolimus pre-surgical palliative. ACTH+VGB limited success."},
    {"term": "Deep-Sequencing-Requirement", "definition": "Somatic MTOR mutations at VAF <5% (blood) require: (1) ultra-deep NGS (>1000× read depth) with strict bioinformatics filtering; (2) ddPCR (droplet digital PCR) for known hotspot confirmation; (3) surgical tissue (gold standard: 99% sensitivity). Standard clinical gene panels (50-100× coverage) often miss somatic MTOR."},
    {"term": "Everolimus-mTORC1-Direct", "definition": "Everolimus (FKBP12-rapamycin analogue): binds FRB domain of mTOR → allosteric mTORC1 inhibition. DIRECT molecular target of GOF MTOR mutation. Neurological dose 3-7 ng/mL (below oncology 10-15 ng/mL). EXIST-3 (TSC, NEJM 2016): extrapolated to MTOR focal epilepsy. Additive with KD (BHB also inhibits mTORC1 via AMPK/REDD1)."},
    {"term": "EXIST-3-Evidence", "definition": "Bass et al. 2016 (NEJM): everolimus vs placebo in TSC-related seizures — 50% responders (≥50% reduction) vs 19% placebo. Median seizure reduction 29%. TSC is upstream mTOR pathway → extrapolated to MTOR-direct GOF mutations. Level B evidence for MTOR focal epilepsy (case series 40-60% reduction)."},
    {"term": "Surgery-MTOR-FCD", "definition": "FCD IIb resection: 60-70% Engel I (seizure-free). HME hemispherotomy: 65-75% Engel I. Key: pre-surgical SEEG for zone delineation; HFO mapping identifies resection margins. Always request tissue somatic MTOR sequencing from resected specimen. Everolimus post-op for margin tissue. AED taper at 24 months if Engel I."},
    {"term": "POLG1-Alpers-Overlap", "definition": "POLG1 mutations → Alpers disease (mtDNA depletion): cerebellar atrophy, developmental regression, DEE — clinically overlaps MTOR phenotype. VPA → fatal hepatotoxicity in Alpers. CPIC 2023 Level A: POLG1 screen mandatory before VPA in any patient with cerebellar signs or DEE."},
    {"term": "Everolimus-Trough-Monitoring", "definition": "Neurological target: 3-7 ng/mL (trough). Measure 2 weeks after start + 4 weeks + then monthly. Key interactions: CYP3A4 inducers (CBZ, PHT, OXC, carbamazepine) reduce everolimus 70-80% → dose increase 2-3×; CYP3A4 inhibitors (azoles, ritonavir) increase >10× → dose reduction 90%. Grapefruit juice: avoid."},
    {"term": "Immunosuppression-Everolimus", "definition": "Everolimus inhibits mTORC1 in T-cells → impaired T-cell proliferation → immunosuppression. Consequences: (1) Live vaccines CI permanently on everolimus; (2) increased infection risk (bacterial, fungal, viral — CMV, EBV reactivation); (3) wound healing impairment → hold 2 weeks pre-surgery; (4) secondary malignancy risk (long-term >5y); (5) stomatitis (50%) — steroid mouthwash, NOT antifungal."},
    {"term": "Two-Hit-Somatic-Model", "definition": "Germline MTOR heterozygous LOF/GOF + somatic second hit → severe focal phenotype at site of second hit. Explains why AD germline carriers can have focal FCD despite bilateral germline mutation. Second hit = additional somatic mutation at same locus in developing cortex → mTORC1 constitutive at clonal population → FCD. Relevant for DEPDC5/NPRL2/NPRL3 and MTOR alike."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — 12
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"param": "Everolimus trough", "value": "3–7 ng/mL", "note": "Neurological dose. Below 3 = inadequate. Above 7 = toxicity risk."},
    {"param": "Somatic VAF detection", "value": "<5% requires deep NGS >1000×", "note": "Standard panels miss somatic MTOR. ddPCR or ultra-deep NGS needed."},
    {"param": "OFC macrocephaly", "value": ">+2 SD", "note": "Screen for MCAP/Smith-Kingsmore. >+3 SD = severe macrocephaly — MRI urgently."},
    {"param": "DRE threshold", "value": "≥3 AED failures", "note": "Initiate surgical evaluation and KD trial. Do not delay."},
    {"param": "HME hemispherotomy", "value": "<12 months ideal", "note": "Neuroplasticity window for language/motor transfer to intact hemisphere."},
    {"param": "FDG-PET hypometabolism", "value": ">30% focal reduction", "note": "Indicates FCD zone in MRI-negative MTOR epilepsy."},
    {"param": "VPA POLG1 screen", "value": "Mandatory before any VPA use", "note": "CPIC 2023 Level A. Alpers phenotype overlap."},
    {"param": "Stomatitis Grade 2+", "value": "→ dexamethasone mouthwash; Grade 3+ → dose reduction", "note": "Most common everolimus SE (50%). Steroid NOT antifungal."},
    {"param": "ACTH response assessment", "value": "14 days", "note": "IS cessation at 2 weeks = responder. Non-responder → hemispherotomy urgently."},
    {"param": "BHB KD target", "value": "2–4 mmol/L", "note": "Indicates effective ketosis. Below 1.5 = inadequate KD adherence."},
    {"param": "QTc monitoring", "value": ">500 ms → hold co-QTc drug", "note": "Relevant if combining everolimus with QTc-prolonging agents (some antipsychotics for ASD)."},
    {"param": "SUDEP risk threshold", "value": "Annual assessment; GTCS >12/year = high risk", "note": "Supervision, CCTV, seizure detection mat for high-risk MTOR DEE patients."},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS — 12
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE-2022", "scope": "Epilepsy classification and genetic epilepsy standards"},
    {"name": "NICE-NG217", "scope": "Epilepsy management guidelines (UK)"},
    {"name": "Mirzaa-2016-NatGenet", "scope": "MTOR GOF in MCAP — landmark mutation spectrum paper"},
    {"name": "Bass-2016-NEJM-EXIST3", "scope": "Everolimus in TSC seizures — primary evidence base for mTOR inhibitor in epilepsy"},
    {"name": "Baldassari-2019-Brain", "scope": "GATOR1/mTOR cohort (73 families) — surgical outcomes and precision therapy"},
    {"name": "Nakashima-2015-AnnNeurol", "scope": "Somatic MTOR mutations in focal cortical dysplasia IIb — somatic detection methodology"},
    {"name": "CPIC-POLG-2023", "scope": "VPA screening in POLG1 — Level A recommendation"},
    {"name": "MHRA-VPPP-2021", "scope": "Valproate Pregnancy Prevention Programme — mandatory for females ≥12y"},
    {"name": "UKISS-2004-Lancet", "scope": "ACTH vs vigabatrin for infantile spasms — IS management standard"},
    {"name": "FDA-SHARE-REMS-VGB", "scope": "Vigabatrin REMS programme — mandatory ERG monitoring"},
    {"name": "ACMG-AMP-2015", "scope": "Variant pathogenicity classification — for MTOR variant interpretation"},
    {"name": "WHO-ICF-2019", "scope": "Functional outcomes framework — disability assessment in DEE"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES — 6
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"id": "Mirzaa-2016-NatGenet", "citation": "Mirzaa GM et al. (2016). Association of MTOR mutations with developmental brain disorders, including megalencephaly, focal cortical dysplasia, and pigmented nevus. JAMA Neurol 73(7):836-845. (MTOR GOF mutation spectrum in MCAP and focal epilepsy)"},
    {"id": "Nakashima-2015-AnnNeurol", "citation": "Nakashima M et al. (2015). Somatic mutations in the MTOR gene cause focal cortical dysplasia type IIb. Ann Neurol 78(3):375-386. (First systematic study of somatic MTOR in FCD IIb; deep sequencing methodology)"},
    {"id": "Baldassari-2019-Brain", "citation": "Baldassari S et al. (2019). The landscape of epilepsy-related GATOR1 variants. Brain 142(9):2967-2981. (73-family GATOR1 cohort including MTOR-pathway; surgical outcomes; precision therapy)"},
    {"id": "Bass-2016-NEJM-EXIST3", "citation": "Bass DI et al. / Bissler JJ et al. (2013/2016). Sirolimus/Everolimus for TSC seizures. NEJM 369(8):791-800 / NEJM 375(12):1142-1153. (EXIST-2/EXIST-3: evidence base for mTOR inhibitor in epilepsy)"},
    {"id": "Smith-2016-NatGenet", "citation": "Smith LD et al. (2016). MTOR gain-of-function mutations cause distinctive macrocephaly (Smith-Kingsmore syndrome). Nat Genet 48(11):1309-1314. (Smith-Kingsmore syndrome characterisation — germline MTOR GOF)"},
    {"id": "Bhatt-2017-NEJM", "citation": "Bhatt DL et al. (2017). Precision medicine for epilepsy. NEJM 376(5):458-469. (Precision medicine framework for mTORopathies and other genetic epilepsies)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATOR — 40 SYNTHETIC PATIENTS
# ─────────────────────────────────────────────────────────────────────────────
def _gen_patients():
    etiology_dist = [
        ("Somatic-MTOR-GOF-FCD-IIb", 35),
        ("Germline-MTOR-GOF-Smith-Kingsmore", 28),
        ("Somatic-MTOR-GOF-HME", 18),
        ("Somatic-MTOR-GOF-MCAP", 12),
        ("Phenocopy-mTOR-Pathway-Negative", 7),
    ]
    drug_options = [
        ["LEV", "Everolimus"],
        ["CBZ-XR", "LEV", "Everolimus"],
        ["LEV", "CLB", "Everolimus"],
        ["VPA", "LEV"],
        ["ACTH", "VGB"],
        ["LEV", "LCM", "Everolimus"],
        ["KD", "LEV", "CLB"],
        ["LEV", "CLB"],
        ["OXC", "LEV", "Everolimus"],
        ["VPA", "LEV", "CLB"],
    ]
    buckets = []
    for et, pct in etiology_dist:
        n = max(1, round(pct * 40 / 100))
        buckets.extend([et] * n)
    while len(buckets) < 40:
        buckets.append("Somatic-MTOR-GOF-FCD-IIb")
    random.shuffle(buckets)
    patients = []
    for i, etiol in enumerate(buckets[:40], 1):
        age_y = random.randint(1, 35)
        onset_m = random.randint(0, 120)
        drugs = random.choice(drug_options)
        sf = random.choice([True, False, False])
        seizure_freq = 0 if sf else random.randint(1, 25)
        mri = random.choice(["FCD IIb — transmantle sign", "Megalencephaly bilateral", "Unilateral hemispheric overgrowth", "Bilateral PMG + capillary malformation", "Normal 1.5T — abnormal 3T FLAIR"])
        patients.append({
            "id": f"EPAT{i:03d}",
            "age_y": age_y,
            "sex": random.choice(["M", "F"]),
            "etiology": etiol,
            "onset_months": onset_m,
            "mri": mri,
            "drugs": ", ".join(drugs),
            "seizure_free": sf,
            "seizure_freq_monthly": seizure_freq,
            "everolimus": "Everolimus" in drugs,
            "surgery": random.choice([True, False, False]),
        })
    return patients


PATIENTS = _gen_patients()

# ─────────────────────────────────────────────────────────────────────────────
# API RETURN FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    sf = sum(1 for p in PATIENTS if p["seizure_free"])
    on_ev = sum(1 for p in PATIENTS if p["everolimus"])
    surgeries = sum(1 for p in PATIENTS if p["surgery"])
    somatic = sum(1 for p in PATIENTS if "Somatic" in p["etiology"])
    return {
        "gene": "MTOR",
        "locus": "1p36.22",
        "omim_gene": "*601231",
        "omim_syndromes": {"MCAP": "#615108", "Smith-Kingsmore": "#616638"},
        "protein": "mTOR kinase (mechanistic Target Of Rapamycin) — catalytic core of mTORC1 + mTORC2",
        "mechanism": "GOF (gain-of-function) — somatic mosaic (75%) or germline (25%) → constitutive mTORC1 hyperactivation → FCD IIb / HME / MCAP / Smith-Kingsmore",
        "inheritance": "Somatic mosaic (de novo, ~75%) / Germline AD (Smith-Kingsmore, ~25%)",
        "precision_therapy": "Everolimus (Votubia/Afinitor) — FKBP12-rapamycin analogue, direct mTORC1 inhibitor",
        "cohort": {
            "n": n,
            "seizure_free_pct": round(sf / n * 100),
            "on_everolimus_pct": round(on_ev / n * 100),
            "post_surgery_pct": round(surgeries / n * 100),
            "somatic_mutation_pct": round(somatic / n * 100),
        },
        "etiologies": [
            {"category": e["category"], "pct": e["pct"], "severity": e["severity"]}
            for e in ETIOLOGY_CATALOG
        ],
        "key_alerts": [
            "SOMATIC MTOR: blood sequencing often negative (VAF <5%) — request ultra-deep NGS (>1000×) or ddPCR; always sequence surgical tissue",
            "EVEROLIMUS DIRECT TARGET: FKBP12-rapamycin binds FRB domain of mutant mTOR — most rational precision therapy (trough 3-7 ng/mL)",
            "TGB ABSOLUTE CI: tiagabine → NCSE in FCD IIb cortex — never use in MTOR epilepsy",
            "POLG1 MANDATORY before VPA: MTOR phenotype overlaps Alpers — CPIC 2023 Level A",
            "CYP3A4 INTERACTIONS: CBZ/PHT/OXC reduce everolimus 70-80% — dose adjust + intensive TDM; azoles increase >10×",
            "LIVE VACCINES CI on everolimus — complete immunisation schedule BEFORE starting",
            "SURGERY PRIMARY for resectable FCD/HME: everolimus adjunct, not substitute; tissue sequencing on resected specimen mandatory",
        ],
    }


def get_breakdown():
    n = len(PATIENTS)
    return {
        "cohort_n": n,
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_CATALOG,
        "triggers": TRIGGER_CATALOG,
        "treatments": TREATMENT_CATALOG,
        "contraindications": CI_CATALOG,
        "monitoring": MONITORING_CATALOG,
        "lifecycle": LIFECYCLE_CATALOG,
        "patients": PATIENTS,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "gator1_trilogy": {
            "note": "MTOR is the APEX convergence point of all mTORopathy pathways:",
            "upstream": [
                "TSC1/TSC2 (PI3K-AKT arm, growth factors) → TSC2 inhibits RHEB → RHEB activates mTORC1",
                "DEPDC5-NPRL2-NPRL3 (GATOR1, amino acid arm) → GATOR1 inhibits Rag GTPases → Rags activate mTORC1",
            ],
            "apex": "MTOR (1p36.22) — catalytic core: direct target of rapamycin (FKBP12-rapamycin binds FRB domain)",
            "downstream": ["S6K1 (T389) → ribosome biogenesis, protein synthesis", "4EBP1 (T37/46) → mRNA translation, cap-dependent protein synthesis"],
        },
    }
