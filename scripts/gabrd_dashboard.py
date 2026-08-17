"""
GABRD Epilepsy — GGE / GEFS+ Spectrum / Catamenial / Tonic Inhibition / GABA-A δ Subunit / 1p36.33
=====================================================================================================
40-patient cohort · GABRD (1p36.33) · GABA-A receptor delta (δ) subunit · AD reduced-penetrance

GABRD BIOLOGY:
GABRD (1p36.33) encodes the GABA-A receptor delta (δ) subunit — the defining component of
EXTRASYNAPTIC (peri/extra-synaptic) GABA-A receptors. Unlike synaptic α1/α2-GABA-A receptors
(phasic inhibition, fast on/off), δ-containing receptors:
  1. LOCATION: Extrasynaptic membranes — NOT at the synapse; respond to ambient GABA spillover.
  2. TONIC INHIBITION: δ-GABA-A generates SUSTAINED, non-desensitizing conductance →
     persistent 'tonic' chloride current setting baseline excitability threshold.
     This is distinct from phasic (synaptic) inhibition mediated by α1/β2/γ2 receptors.
  3. STOICHIOMETRY: Most abundant: α4/δ (dentate gyrus granule cells, thalamic relay neurons)
     and α6/δ (cerebellar granule cells). α1/δ minor. All δ-GABA-A: high GABA affinity,
     very low desensitization rate.
  4. NEUROSTEROID SENSITIVITY: δ-GABA-A receptors are the PRIMARY TARGET of endogenous
     neurosteroids (allopregnanolone / ALLO = 3α,5α-tetrahydroprogesterone;
     THDOC = 3α,5α-tetrahydrodeoxycorticosterone). Neurosteroids at nanomolar concentrations
     → POSITIVE allosteric modulation of δ-GABA-A → enhanced tonic inhibition.
     This is the molecular basis of CATAMENIAL EPILEPSY in GABRD:
       - Luteal phase: rising progesterone → ALLO synthesis → upregulates δ-GABA-A expression
         AND potentiates δ-GABA-A → lower seizure frequency.
       - Perimenstrual phase: progesterone withdrawal → ALLO drops abruptly → δ-GABA-A
         downregulation (subunit internalization in 24-48h) → acute reduction in tonic inhibition
         → PERIMENSTRUAL SEIZURE CLUSTERING. The withdrawal is faster than receptor adaptation.
       - GABRD LOF amplifies this vulnerability: baseline tonic inhibition already reduced
         → withdrawal exaggerates the deficit → type C1 catamenial epilepsy.
  5. ETHANOL SENSITIVITY: δ-GABA-A receptors respond to LOW concentrations of ethanol
     (~3-10 mM, i.e., 1-2 drinks) — unlike synaptic α1-GABA-A (requires ~100 mM).
     GABRD LOF → reduced tonic inhibition → HEIGHTENED SENSITIVITY to both alcohol
     sedation effects (disinhibitory paradox) AND alcohol-withdrawal seizures.
     GABRA2 LOF also has alcohol link (different mechanism — AIS phasic).

DISEASE MECHANISM (GABRD LOF → EPILEPSY):
  GABRD LOF variants (missense reducing surface expression or channel gating) →
  Reduced extrasynaptic δ-GABA-A surface density or conductance →
  Loss of tonic Cl⁻ current → baseline cortical/hippocampal/thalamic hyperexcitability →
  GGE spectrum (GEFS+, JME, CAE, GTCS) with variable penetrance (~60-70% in GEFS+ families).
  Thalamo-cortical circuits particularly affected (α4/δ-GABA-A in thalamic relay nuclei →
  tonic inhibition of thalamic output → loss of it → enhanced thalamocortical synchrony →
  SWD / absence / GGE phenotype).

CLINICAL PHENOTYPE:
  Highly variable penetrance and expressivity — hallmark of GABRD epilepsy.
  Ranges from simple febrile seizures only (low-penetrance alleles) through GEFS+ spectrum
  (FS+ / FS+ with absences / FS+ with myoclonic) to JME-like / CAE to rare severe DEE.
  CATAMENIAL clustering: ~35-45% of females with GABRD GGE have clear perimenstrual
  exacerbation (type C1). AUDIT-C important: ethanol interaction via δ-GABA-A.
  Key diagnostic clue: family history of febrile seizures + GGE in multiple generations
  with variable phenotype severity.

INHERITANCE AND GENETICS:
  AUTOSOMAL DOMINANT — reduced penetrance (~60-70% in GEFS+ families). Variable expressivity.
  De novo variants: rare but reported in severe DEE.
  Locus: 1p36.33. Gene: GABRD (9 exons; 472 aa protein).
  pLI ~0.61 (intermediate constraint). Key pathogenic variants: R220H (GEFS+ familial),
  E177A-like (thalamic δ-GABA-A surface expression reduction), L270V-like missense.
  Protein: δ subunit — 1 N-terminal extracellular domain + 4 transmembrane helices (TM1-4)
  + large intracellular TM3-TM4 loop (phosphorylation / trafficking sites).
  Most pathogenic variants: TM2 (channel pore, gating) or N-terminal (GABA binding interface).
  OMIM: GEFS+4 (#604420); related gene entry for GABRD.

KEY REFERENCE: Dibbens et al. 2004 Nature Genetics 36:1327 — first GABRD GEFS+ families;
R220H and L270V variants; reduced current amplitude in oocyte expression system.
Maguire et al. 2005 Nat Neurosci 8:797 — neurosteroid/δ-GABA-A/catamenial mechanism.
Maljevic & Lerche 2013 J Physiol 591:759 — comprehensive GABA receptor gene review.
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GABRD-AD-missense-GEFS-plus",
        "pct": 38,
        "etiology": "GABRD AD missense — GEFS+ spectrum (familial, reduced penetrance)",
        "mechanism": (
            "Heterozygous missense variant (e.g., R220H, E177A-like, L270V-like) → partial loss of δ-subunit "
            "surface expression or channel gating efficiency → reduced tonic Cl⁻ current by ~40-60% at affected "
            "synapses → threshold for FS+ / GEFS+ spectrum lowered. Penetrance ~65% within family "
            "(obligate carriers with no seizures common). Expressivity variable: same variant → FS only in one "
            "sibling, JME-like in another. Neurosteroid withdrawal (perimenstrual) → exaggerated catamenial "
            "clustering (δ-GABA-A downregulation on progesterone withdrawal)."
        ),
        "typical_variants": "Missense: R220H (Arg220His, TM3-4 loop); E177A-like (N-terminal, GABA-binding); L270V (TM2 pore); de novo frameshift rare in this class",
        "eeg_signature": "Generalised 3-4 Hz spike-and-wave (absence); 2-3 Hz polyspike-wave (myoclonic); photoparoxysmal response 20-30%; HV activation of SWD; sleep: GGSWD",
        "phenotype": "GEFS+ spectrum: FS+ → FS with absences → FS with myoclonic → JME-like adult. Penetrance ~65%. Variable severity siblings. Catamenial in 40% females.",
    },
    {
        "category": "GABRD-AD-JME-phenotype",
        "pct": 28,
        "etiology": "GABRD variant — JME phenotype (myoclonic + GTCS + absence triad)",
        "mechanism": (
            "GABRD missense → reduced thalamo-cortical tonic inhibition (α4/δ in thalamic relay nuclei) → "
            "enhanced thalamocortical synchrony → 3-4 Hz generalised SWD + myoclonic jerks on waking. "
            "Full JME triad: myoclonic jerks (AM peak), GTCS (AM, sleep-deprived), absences (variable). "
            "Phenotypically indistinguishable from JME without genetic testing. POLG screen mandatory "
            "before VPA. VPPP mandatory females of reproductive age. Alcohol and sleep deprivation are "
            "the DOMINANT triggers (δ-GABA-A ethanol sensitivity contributes to morning-after GTCS)."
        ),
        "typical_variants": "Missense in N-terminal or TM2; occasionally in TM3-4 loop; rarely de novo truncating",
        "eeg_signature": "Classic JME EEG: 3-5 Hz GSW + polyspike-wave; myoclonic runs on awakening; PPR in 20-25%; EEG may be normal interictally if sleep-deprived capture missed",
        "phenotype": "JME phenotype: onset 12-18y; myoclonic jerks AM; GTCS sleep-deprived; absences 30%; excellent VPA response; lifelong risk (rarely remits)",
    },
    {
        "category": "GABRD-CAE-phenotype",
        "pct": 18,
        "etiology": "GABRD variant — Childhood Absence Epilepsy (CAE) / catamenial-predominant",
        "mechanism": (
            "GABRD LOF → thalamic relay neuron disinhibition (α4/δ in VB/nRT) → enhanced thalamo-cortical "
            "burst-firing → rhythmic 3 Hz SWD → absence seizures. In females: catamenial pattern prominent "
            "(35-45% female GABRD patients). Perimenstrual ALLO drop → acute δ-GABA-A density loss → "
            "absence frequency spikes 2-5× in luteal-to-menstrual transition. Ganaxolone (synthetic ALLO) "
            "is precision therapy targeting δ-GABA-A restoration for catamenial seizures. "
            "ETH or VPA for non-catamenial absence control; CLB adjunct for catamenial clustering."
        ),
        "typical_variants": "Missense variants reducing δ-subunit surface expression; thalamic α4/δ stoichiometry sensitive to δ haploinsufficiency",
        "eeg_signature": "3 Hz generalised SWD (absence ictal); HV strongly activating; IPS weak activator; normal background; perimenstrual EEG shows higher SWD burden",
        "phenotype": "CAE onset 4-10y; short absences (5-20s); catamenial clustering in females; excellent ESM/VPA response; 30-40% outgrow absence but some evolve to JME phenotype",
    },
    {
        "category": "GABRD-de-novo-DEE",
        "pct": 10,
        "etiology": "GABRD de novo — severe DEE (rare; dominant-negative mechanism)",
        "mechanism": (
            "Rare de novo GABRD variants (frameshift, splice, missense in key interface residues) with "
            "dominant-negative effect on δ-GABA-A complex assembly → >80% reduction of functional receptors "
            "→ severe tonic inhibition deficit → neonatal/infantile DEE phenotype. These are clinically "
            "distinct from familial GEFS+ (which is missense/reduced-penetrance). Literature: rare case "
            "reports (Maljevic 2006, Maguire 2009). STXBP1/KCNQ2 differential must be ruled out by panel. "
            "Drug-resistant. Neurosteroid supplementation sometimes trialled."
        ),
        "typical_variants": "De novo splice-site, frameshift; rare dominant-negative missense in critical GABA-binding or pore-forming TM2 region",
        "eeg_signature": "Multi-focal sharp waves (infantile); evolves to generalised SWD/polyspike with age; hypsarrhythmia may occur in severe forms",
        "phenotype": "Severe DEE: infantile onset, drug-resistant, moderate-severe ID; rare GABRD presentation — exclude STXBP1/KCNQ2/SCN2A first",
    },
    {
        "category": "GABRD-negative-phenocopy-GGE",
        "pct": 6,
        "etiology": "GABRD-negative GGE/GEFS+ phenocopy",
        "mechanism": (
            "GGE / GEFS+ clinical phenotype without identified GABRD pathogenic variant. "
            "Differential panel: GABRA1 (most common GGE gene / DEE19 / 5q34), GABRA2 (4p12 / alcohol-link), "
            "GABRB3 (childhood epilepsies), GABRG2 (GEFS+ / DEE11 / 5q34), SCN1A (Dravet / GEFS+ / 2q24.3), "
            "KCNQ2/3 (familial neonatal seizures), SLC6A1 (MAE phenotype), CACNA1H (CAE susceptibility). "
            "Neurosteroid and catamenial features do not confirm GABRD — all GGE genes can have catamenial component. "
            "AUDIT-C and alcohol withdrawal risk relevant regardless of specific gene."
        ),
        "typical_variants": "None in GABRD — identified by panel-negative result; other GGE genes under investigation",
        "eeg_signature": "Generalised SWD / polyspike as expected for GGE phenotype; gene-specific features absent; proceed to clinical diagnosis by syndrome",
        "phenotype": "GGE clinically: GEFS+, JME, or CAE without GABRD variant; treat by phenotype/syndrome; 60% seizure-free on 1-2 AEDs",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Febrile Seizures Plus (FS+) — GEFS+ hallmark",
        "prevalence_pct": 80,
        "semiology": (
            "Febrile convulsions persisting beyond 6 years (FS+) — the hallmark of GEFS+ spectrum. "
            "Generalised tonic-clonic; duration 1-5 min; temperature threshold lower than population. "
            "Some relatives: only FS (with normal limit of 5y) — FS is the mild end of GEFS+ spectrum. "
            "Family history essential: may span 3+ generations with variable expression."
        ),
        "eeg_pattern": "Ictal: generalised rhythmic theta/delta; interictal: often normal or subtle GSW. Febrile EEG: diffuse slowing. Post-ictal diffuse suppression.",
        "clinical_tip": (
            "FS+ diagnosis requires FS persisting after age 6 OR febrile seizures with afebrile seizures later. "
            "Ask specifically about age at last febrile seizure. Family history: 3-generation pedigree. "
            "GABRD GEFS+ — genetic panel if ≥2 affected relatives with variable GGE phenotypes."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "prevalence_pct": 72,
        "semiology": (
            "Tonic-clonic: bilateral symmetric tonic phase (10-30s) → clonic phase (30-90s) → post-ictal "
            "confusion/Todd's. AM peak (awakening); sleep-deprived and alcohol-withdrawal triggers dominant. "
            "GABRD δ-GABA-A: ethanol at 3-10 mM potentiates δ-GABA-A → acute sedation/disinhibition → "
            "on withdrawal: δ-GABA-A downregulation → GTCS risk elevated."
        ),
        "eeg_pattern": "Pre-ictal: polyspike-wave bursts (GSW 4-6 Hz); ictal: fast rhythmic discharge → 3 Hz GSW → post-ictal attenuation. Amplitude: generalised, synchronous.",
        "clinical_tip": (
            "GTCS in GABRD GGE: highest risk windows are morning awakening, post-alcohol night, and perimenstrual "
            "(catamenial C1). Sleep diary + alcohol diary + menstrual diary for females — three separate triggers. "
            "ABSOLUTE CI: CBZ, OXC, PHT — will aggravate GGE. LTG: EEG pre-prescribing mandatory (15-20% myoclonic worsening)."
        ),
    },
    {
        "type": "Catamenial Seizure Clustering (Perimenstrual GTCS/absence)",
        "prevalence_pct": 42,
        "semiology": (
            "Type C1 catamenial epilepsy: seizure clustering in days -3 to +3 of menstruation "
            "(progesterone withdrawal phase). GABRD-specific mechanism: progesterone withdrawal "
            "→ rapid ALLO decline → δ-GABA-A downregulation → acute tonic inhibition deficit → "
            "seizure clusters. Typical: 2-5× baseline seizure frequency in perimenstrual window. "
            "Any seizure type: absence, myoclonic, or GTCS. "
            "All females with GABRD GGE should be screened for catamenial pattern (seizure-menstrual diary)."
        ),
        "eeg_pattern": "Higher SWD burden on perimenstrual EEG vs mid-cycle. Ictal pattern unchanged (GSW/polyspike). EEG during cluster: more abundant IEDs.",
        "clinical_tip": (
            "Document menstrual phase on every seizure diary entry. Catamenial seizures in GABRD → consider: "
            "CLB (1-10 mg) in perimenstrual window (C1 days -3 to +3); Ganaxolone (synthetic ALLO — δ-GABA-A PAM — "
            "precision therapy for GABRD catamenial); progesterone/NETA supplementation (luteal phase) — "
            "Level C evidence. Progesterone CONTRACEPTION note: combined OCP may attenuate catamenial cycling."
        ),
    },
    {
        "type": "Absence Seizures (Typical / Atypical)",
        "prevalence_pct": 65,
        "semiology": (
            "Typical absences: 5-20s behavioral arrest + staring + eyelid flicker; abrupt offset; "
            "usually multiple per day. HV strongly provocative (routine EEG). "
            "GABRD: thalamic α4/δ-GABA-A LOF → thalamo-cortical burst-firing → 3 Hz SWD. "
            "May precede GTCS by years (especially CAE → JME evolution in adolescence). "
            "Catamenial peak in perimenstrual window."
        ),
        "eeg_pattern": "Ictal: 3 Hz GSW; bilateral, synchronous; abrupt onset/offset. HV: strongly activating (routine 3-min HV sufficient). IPS: weak in absence-predominant.",
        "clinical_tip": (
            "ESM for pure absence (Level A; preferred over VPA for absence-dominant, especially in children "
            "and females of reproductive age). VPA Level A (full-spectrum: absence + GTCS + myoclonic). "
            "NEVER use CBZ/OXC/PHT/TGB — will worsen or precipitate SE. LTG: only after EEG confirms "
            "no myoclonic component (myoclonic worsening in 15-20% GGE with LTG)."
        ),
    },
    {
        "type": "Myoclonic Jerks (On Awakening)",
        "prevalence_pct": 55,
        "semiology": (
            "Bilateral symmetric myoclonic jerks of upper limbs; predominant on awakening (30-60 min "
            "post-sleep). May cluster → myoclonic status. Distinguish from cortical reflex myoclonus "
            "(focal) or negative myoclonus (tonic drop). GABRD: loss of δ-GABA-A tonic inhibition "
            "→ heightened cortical-cerebellar loop excitability → myoclonic discharge. "
            "Morning jerks often precede GTCS by minutes-hours — AED adherence critical AM."
        ),
        "eeg_pattern": "Ictal: polyspike (2-5 Hz) or polyspike-wave bursts; bilateral frontocentral predominance. EMG: short bilateral jerks. Background: normal or mildly slow.",
        "clinical_tip": (
            "Myoclonic jerks are the MARKER of JME-phenotype GABRD — their presence mandates: "
            "(1) EEG before LTG (LTG may worsen myoclonic in 15-20%); (2) VPA Level A preferred; "
            "(3) LEV Level B for myoclonic-GTCS. Myoclonic status: IV BZD rescue. "
            "Morning AED schedule critical — take AED immediately on waking, not post-shower."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sleep Deprivation", "pct": 85, "note": "Top trigger; δ-GABA-A tonic inhibition most needed during wake→NREM transition. Strict sleep hygiene mandatory. 7-8h minimum."},
    {"trigger": "Missed AED Dose", "pct": 80, "note": "AED non-adherence — especially morning AED (peak GTCS/myoclonic window). Structured once-daily regimen where possible."},
    {"trigger": "Alcohol Consumption + Withdrawal", "pct": 68, "note": "GABRD-specific: δ-GABA-A is high-affinity ethanol target. Acute alcohol: disinhibitory paradox. Next-morning withdrawal → GTCS. AUDIT-C every visit."},
    {"trigger": "Perimenstrual / Catamenial (Females)", "pct": 42, "note": "Progesterone withdrawal → ALLO drop → δ-GABA-A downregulation → seizure cluster days -3 to +3. Seizure-menstrual diary for all GABRD females."},
    {"trigger": "Stress / Emotional Disturbance", "pct": 60, "note": "Stress → cortisol → neurosteroid flux → δ-GABA-A sensitivity changes. Also: poor sleep + stress co-occurs."},
    {"trigger": "Fever / Illness", "pct": 55, "note": "GEFS+ spectrum: fever is a primary trigger (FS+ by definition). Advise early fever control + rescue BZD plan."},
    {"trigger": "Photosensitivity (IPS)", "pct": 25, "note": "GGE-associated photoparoxysmal response; lower rate than SCN1A/CHD2 (<30%). Screen with IPS at EEG. Photosensitive patients: screen use guidance, FL-41 tints."},
    {"trigger": "Hyperventilation", "pct": 20, "note": "Especially for absence component (3 Hz SWD HV-triggered). Identify: exercise, emotional hyperventilation → absence clustering."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA) — Level A (GGE full-spectrum: GTCS + absence + myoclonic)",
        "level": "Level A",
        "moa": (
            "Multi-mechanism: (1) Na⁺ channel block (GTCS/focal); (2) enhanced GABAergic transmission "
            "(GABA transaminase inhibition → increased GABA); (3) T-type Ca²⁺ channel block (absence / thalamocortical); "
            "(4) Modulates δ-GABA-A indirectly via GABA availability. Broad-spectrum — the ONLY Level A drug for "
            "all three GGE seizure types (GTCS + absence + myoclonic) simultaneously."
        ),
        "dose": "Adults: 500-2000 mg/day (TDM: 50-100 mg/L); Children: 15-60 mg/kg/day. Titrate slowly. Once-daily extended-release preferred.",
        "efficacy": "60-70% seizure-free in GGE; 40-50% for GTCS control in JME-phenotype. SANAD (2007): VPA superior to LTG and TOP for GGE.",
        "safety": "Teratogen (NTD, DQ reduction — VPPP mandatory females); weight gain; tremor; hair loss; VPA-induced hyperammonemia; POLG-related hepatotoxicity risk.",
        "monitoring": "POLG screen before initiation; VPA TDM q3M; LFT + FBC + ammonia q3M; VPPP annual enrolment check females; BMI monthly.",
        "gabrd_note": (
            "GABRD GGE: VPA is FIRST-LINE for full-spectrum control (GTCS + absence + myoclonic). "
            "δ-GABA-A LOF → GABA deficiency → VPA enhances GABA availability → partially compensatory. "
            "CRITICAL: POLG screen mandatory before starting. VPPP mandatory for all females of reproductive age. "
            "AUDIT-C: VPA + alcohol interaction (VPA inhibits alcohol metabolism → higher ethanol + acetaldehyde)."
        ),
    },
    {
        "drug": "Ethosuximide (ESM) — Level A (absence-dominant GABRD / CAE phenotype)",
        "level": "Level A",
        "moa": (
            "T-type (Cav3.1/3.2) Ca²⁺ channel blocker → reduces thalamic burst firing → attenuates "
            "thalamocortical 3 Hz oscillation → blocks absence SWD generation. Specific for thalamo-cortical "
            "loop — the EXACT circuit disrupted by GABRD α4/δ LOF in thalamic relay neurons. "
            "No effect on GTCS or myoclonic — absence-ONLY drug."
        ),
        "dose": "Children: 15-40 mg/kg/day in 2-3 doses (max 1500 mg). Adults: 500-1500 mg/day. TDM: 40-100 mg/L.",
        "efficacy": "SANAD-II (2021): ESM = VPA for absence seizure control; ESM superior tolerability in females (avoids VPPP obligations). 70-80% absence freedom.",
        "safety": "GI (nausea, vomiting — take with food); headache; behaviour/mood changes; rare aplastic anaemia (CBC monitoring); no teratogenicity data (but less than VPA).",
        "monitoring": "CBC q6M (rare aplastic anaemia); LFT q12M; ESM TDM q6M (40-100 mg/L).",
        "gabrd_note": (
            "GABRD CAE-phenotype: ESM preferred over VPA in females of reproductive age (absence-dominant, "
            "no GTCS or myoclonic) — avoids VPPP/teratogenicity. SANAD-II validates ESM=VPA for absence. "
            "If GTCS or myoclonic present: ESM INSUFFICIENT alone → switch to VPA or add LEV. "
            "ESM does NOT affect δ-GABA-A directly — acts downstream on thalamic T-type Ca²⁺."
        ),
    },
    {
        "drug": "Levetiracetam (LEV) — Level B (GTCS + myoclonic; adjunct / VPA alternative)",
        "level": "Level B",
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) modulator → reduces vesicle priming and neurotransmitter "
            "release probability; also modulates GABA-A receptor gating (indirect). Excellent for GTCS and "
            "myoclonic seizures in GGE. SANAD-II (2021): LEV non-inferior to VPA for JME/GTCS control. "
            "No teratogenicity signal comparable to VPA."
        ),
        "dose": "500-3000 mg/day in 2 doses; or extended-release once daily. Renal dose adjustment (eGFR). Start 250-500 mg → titrate over 4w.",
        "efficacy": "50-60% seizure reduction in GGE GTCS/myoclonic; 40-50% seizure-free in JME-like phenotype (SANAD-II).",
        "safety": "Behavioural side effects (irritability, depression, aggression — GABRD anxiety comorbidity risk); somnolence; headache. No teratogenicity (preferred in females).",
        "monitoring": "eGFR baseline + q12M; behavioural/mood assessment every visit; PHQ-9 for depression screen.",
        "gabrd_note": (
            "GABRD JME-phenotype: LEV is preferred alternative to VPA for females (avoids VPPP/teratogen). "
            "SANAD-II 2021: LEV = VPA for JME seizure control in females. "
            "Mood/irritability: GABRD patients often have anxiety comorbidity (neurosteroid sensitivity) — "
            "LEV behavioural effects can be problematic; monitor PHQ-9 + GAD-7 actively."
        ),
    },
    {
        "drug": "Clobazam (CLB) — Level B (catamenial adjunct / perimenstrual cluster rescue)",
        "level": "Level B",
        "moa": (
            "1,5-benzodiazepine → positive allosteric modulator of GABA-A receptors (primarily α1, but also "
            "δ-GABA-A at higher doses). Less sedating than 1,4-BZDs (diazepam/clonazepam). "
            "Catamenial use: 5-10 mg/day in days -3 to +3 perimenstrual window → bridges δ-GABA-A "
            "tonic inhibition deficit during ALLO withdrawal phase. Adjunct to baseline AED; not monotherapy."
        ),
        "dose": "Catamenial: 5-10 mg/day (days -3 to +3). Regular: 10-20 mg/day in 2 doses. Max 40 mg/day.",
        "efficacy": "50-60% reduction in catamenial seizure cluster frequency (open-label data). Adjunct benefit for GGE breakthrough GTCS.",
        "safety": "Tolerance (reduce efficacy with continuous use → catamenial intermittent preferred); sedation; dependence risk with prolonged use; withdrawal seizures if abrupt stop.",
        "monitoring": "Duration of use; tolerance development; dependence screen; avoid in alcohol misuse.",
        "gabrd_note": (
            "GABRD catamenial (C1): CLB intermittent catamenial use is the FIRST-LINE adjunct for perimenstrual "
            "clusters (days -3 to +3 of cycle). Intermittent reduces tolerance. "
            "GABRD-specific: δ-GABA-A LOF → BZD rescue (α1-GABA-A intact) remains effective for status/cluster. "
            "ALCOHOL WARNING: CLB + alcohol → additive CNS depression → counsel strictly. AUDIT-C first."
        ),
    },
    {
        "drug": "Ganaxolone — Level B (GABRD precision: catamenial / neurosteroid-sensitive GGE)",
        "level": "Level B",
        "moa": (
            "Synthetic analogue of allopregnanolone (ALLO; 3α,5α-tetrahydroprogesterone). "
            "POSITIVE ALLOSTERIC MODULATOR of δ-GABA-A receptors at nanomolar concentrations — "
            "the SAME receptor class impaired by GABRD LOF. Unlike endogenous ALLO, ganaxolone: "
            "lacks pregnane backbone → not converted to active progestins → no hormonal effects. "
            "Does NOT require intact progesterone signalling to work. Restores δ-GABA-A tonic inhibition. "
            "FDA-approved: CDKL5 DEE (Ztalmy 2022). Off-label for catamenial GGE."
        ),
        "dose": "Adults: 300-1500 mg/day in 3 doses (TID). Paediatric: 6 mg/kg/day → titrate. IV formulation in clinical trials for refractory status.",
        "efficacy": "Catamenial seizure: 50-60% cluster reduction in open-label studies; CDKL5 trial: 30% seizure reduction vs placebo. GABRD precision: theoretical mechanistic fit (δ-GABA-A PAM).",
        "safety": "Somnolence; dizziness; metabolised by CYP3A4 (interaction with enzyme-inducing AEDs); no teratogenicity signal in animal studies.",
        "monitoring": "CYP3A4 interaction (CBZ lowers ganaxolone 50% — but CBZ ABSOLUTE CI in GGE anyway); CNS sedation; LFT baseline.",
        "gabrd_note": (
            "GABRD PRECISION THERAPY: Ganaxolone is the mechanistically ideal drug for GABRD LOF — "
            "it directly activates the same δ-GABA-A receptors that GABRD LOF impairs. "
            "For catamenial GABRD GGE: ganaxolone covers the perimenstrual ALLO deficit without "
            "hormonal side effects of progesterone supplementation. "
            "Current status: FDA-approved for CDKL5 DEE; off-label for catamenial GGE — discuss with specialist."
        ),
    },
    {
        "drug": "Lamotrigine (LTG) — Level B (CAUTION: EEG pre-prescribing mandatory; 15-20% myoclonic worsening)",
        "level": "Level B (CAUTION)",
        "moa": "Na⁺ channel blocker (use-dependent). Effective for GTCS and absence in GGE without myoclonic component. SANAD (2007): inferior to VPA in JME but viable in absence-dominant GGE.",
        "dose": "Start LOW: 25 mg/day → titrate 25 mg q2w → target 100-400 mg/day (slow titration mandatory to reduce SJS/TEN risk). With VPA: 25 mg alternate day start.",
        "efficacy": "GTCS + absence in GGE without myoclonic: 50-60% seizure reduction. NEVER adequate for JME-phenotype with myoclonic (worsens in 15-20%).",
        "safety": "SJS/TEN (HLA-B*15:02 — MANDATORY screen in Han Chinese / Thai / Southeast Asian populations); rash; insomnia; dizziness. Stevens-Johnson prophylaxis: slow titration.",
        "monitoring": "HLA-B*15:02 before prescribing; skin rash monitoring (first 8w); EEG before prescribing (look for myoclonic discharge on awakening EEG).",
        "gabrd_note": (
            "GABRD GGE: LTG requires EEG PRE-PRESCRIBING — confirm no myoclonic component. "
            "Myoclonic GABRD (JME-phenotype): LTG is HIGH RISK → 15-20% worsen myoclonic → GTCS increase. "
            "GABRD absence-dominant (CAE) without myoclonic: LTG viable Level B alternative. "
            "HLA-B*15:02 mandatory screen before LTG in at-risk populations. "
            "LTG + OCP: OCP lowers LTG levels 40-50% → seizures may worsen if OCP started/stopped."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD) — Level B (refractory GABRD GGE after ≥2 AED trials)",
        "level": "Level B",
        "moa": (
            "High fat:carbohydrate ratio → ketosis → multiple anti-seizure mechanisms: "
            "GABA upregulation (β-hydroxybutyrate → glutamate → GABA synthesis); "
            "KATP channel opening (ATP reduction); mTOR pathway modulation; ATP-sensitive K⁺ channels; "
            "KD independently modulates tonic GABA conductance in dentate gyrus granule cells "
            "(the site of α4/δ-GABA-A — same as GABRD target)."
        ),
        "dose": "Classical 4:1 fat:carbohydrate+protein ratio. MAD (Modified Atkins) for adults. Supervised by ketogenic dietitian.",
        "efficacy": "In refractory GGE: 30-50% responders (≥50% seizure reduction). Not superior to AED in early GGE but valid after 2+ AED failures.",
        "safety": "Constipation; dyslipidaemia; kidney stones; acidosis; growth impairment (paediatric); DXA scan (bone density annual); adequate hydration.",
        "monitoring": "Urinary ketones daily (target 2-4+); lipids q3M; uric acid q3M; DXA annual; kidney ultrasound annual.",
        "gabrd_note": (
            "GABRD refractory GGE: KD after failure of 2 AEDs (VPA + LEV). "
            "Mechanistic note: KD may partially compensate δ-GABA-A deficit via upregulation of GABA synthesis "
            "in dentate gyrus — the primary α4/δ-GABA-A locus. "
            "Catamenial seizures: KD effects on catamenial pattern not well studied — monitor menstrual diary."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
        "risk": "ABSOLUTE CI — GGE aggravation",
        "reason": (
            "Na⁺ channel blockers in GGE: block voltage-gated Na⁺ channels → reduce focal discharge but "
            "PARADOXICALLY AGGRAVATE generalised seizures in GGE. Mechanism: thalamo-cortical Na⁺ channel "
            "block alters thalamic relay neuron burst gating → worsens 3 Hz SWD → increased absence, "
            "myoclonic, and GTCS frequency. Multiple case series: CBZ in GGE → absence status epilepticus, "
            "myoclonic exacerbation. GABRD GGE: ABSOLUTE CI — even sub-therapeutic CBZ can precipitate "
            "prolonged absence status. OXC and PHT equally contraindicated."
        ),
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk": "ABSOLUTE CI — NCSE / Absence Status",
        "reason": (
            "GAT-1 (GABA transporter 1) inhibitor → prolongs synaptic GABA → paradoxically induces "
            "non-convulsive status epilepticus (NCSE) / absence status in GGE patients by enhancing "
            "tonic GABA conductance in an uncontrolled, oscillatory manner. Multiple published cases "
            "of TGB-induced NCSE in GGE. δ-GABA-A already disrupted in GABRD → TGB-mediated "
            "phasic GABA accumulation → dysrhythmic thalamo-cortical discharge → NCSE. "
            "ABSOLUTE CONTRAINDICATION in ALL GGE/GGE-related epilepsies including GABRD."
        ),
    },
    {
        "drug": "Lamotrigine (LTG) without EEG pre-prescribing",
        "risk": "HIGH RISK — 15-20% myoclonic worsening in JME-phenotype GABRD",
        "reason": (
            "LTG may worsen myoclonic seizures in GGE patients with myoclonic component (JME-phenotype). "
            "Mechanism: Na⁺ channel block without GABA potentiation → reduces GTCS but fails to address "
            "thalamocortical tonic inhibition deficit (δ-GABA-A LOF in GABRD) → myoclonic discharge "
            "may be facilitated. Multiple RCTs: 15-20% of JME patients worsen on LTG. "
            "GABRD JME: EEG mandatory pre-prescribing; if myoclonic component confirmed → do NOT use LTG."
        ),
    },
    {
        "drug": "Valproate (VPA) — Females without VPPP enrolment",
        "risk": "HIGH RISK — VPPP mandatory (MHRA 2021)",
        "reason": (
            "VPA is a potent teratogen: NTD (1-2%), cardiovascular defects, orofacial cleft. Prenatal VPA "
            "exposure → developmental delay / autism risk (child IQ reduction of 7-10 points). "
            "MHRA 2021: VPA MUST NOT be prescribed to females of childbearing age without Valproate "
            "Pregnancy Prevention Programme (VPPP) enrolment and annual review. Two effective contraception "
            "methods required. GABRD: VPA often the best anti-seizure drug (δ-GABA-A LOF + broad spectrum) → "
            "VPPP enables safe use while preventing fetal exposure."
        ),
    },
    {
        "drug": "Valproate (VPA) without POLG screening",
        "risk": "HIGH RISK — Alpers-Huttenlocher fatal hepatotoxicity",
        "reason": (
            "POLG (polymerase gamma) mutations → mitochondrial DNA depletion syndrome → Alpers-Huttenlocher "
            "syndrome. VPA administration in POLG patients → irreversible hepatic mitochondrial toxicity → "
            "fulminant hepatic failure (FHF). Often fatal. POLG screening (CPIC-POLG-2023 guidelines): "
            "mandatory before VPA initiation in ANY patient with epilepsy + liver/mitochondrial features. "
            "GABRD GGE: typically no mitochondrial features BUT POLG screen still mandatory (CPIC Level A). "
            "If POLG pathogenic variant found: VPA CONTRAINDICATED ABSOLUTELY."
        ),
    },
    {
        "drug": "Alcohol — GABRD-specific HIGH RISK",
        "risk": "HIGH RISK — δ-GABA-A ethanol sensitivity + withdrawal",
        "reason": (
            "GABRD-SPECIFIC: δ-GABA-A receptors are the primary target of low-dose ethanol (3-10 mM). "
            "Acute alcohol in GABRD: compensates δ-GABA-A LOF transiently → patient may 'feel better' → "
            "reinforces alcohol use → GABRD patients at elevated risk of alcohol misuse. "
            "Alcohol WITHDRAWAL: δ-GABA-A downregulation on chronic alcohol → withdrawal → severe GTCS. "
            "AUDIT-C at every visit (same as GABRA2). Alcohol ABSOLUTE AVOIDANCE counselling mandatory. "
            "Alcohol misuse management: consider dedicated addiction psychiatry referral."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "POLG screen (VPA safety)", "frequency": "Before VPA initiation (once)"},
    {"item": "VPPP annual review (females on VPA)", "frequency": "Annual (MHRA 2021 mandatory)"},
    {"item": "VPA TDM (therapeutic drug monitoring)", "frequency": "q3M (target 50-100 mg/L)"},
    {"item": "LFT + FBC + ammonia (VPA hepatotoxicity)", "frequency": "q3M while on VPA"},
    {"item": "AUDIT-C alcohol screening", "frequency": "Every clinic visit (GABRD-specific)"},
    {"item": "Seizure-menstrual diary (catamenial C1)", "frequency": "Continuous; review every visit (females)"},
    {"item": "EEG (IPS + AM awakening + HV)", "frequency": "Annual; before LTG initiation; on-demand catamenial EEG"},
    {"item": "HLA-B*15:02 (LTG — SJS/TEN screen)", "frequency": "Before LTG in at-risk populations (once)"},
    {"item": "HLA-A*31:01 (CBZ — DRESS screen)", "frequency": "Before any NaV blocker in at-risk populations (note: CBZ CI in GGE)"},
    {"item": "Neuropsychological assessment", "frequency": "q2y (VPA cognitive effects; anxiety/depression comorbidity)"},
    {"item": "Driving/activities counselling", "frequency": "At diagnosis + on GTCS breakthrough; review annually"},
    {"item": "SUDEP risk stratification", "frequency": "Annual (GTCS frequency + sleeping alone + nocturnal GTCS)"},
    {"item": "GAD-7 / PHQ-9 (anxiety + depression)", "frequency": "Every visit (neurosteroid sensitivity → mood comorbidity)"},
    {"item": "Genetic counselling (AD reduced penetrance)", "frequency": "At diagnosis; preconception (VPPP gate)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {"window": "Infancy / Toddler (0-5y)", "headline": "Febrile seizures onset (GEFS+ spectrum); family history elicitation; GABRD panel if ≥2 affected relatives; pyridoxine trial to exclude B6-dependent epilepsy"},
    {"window": "School Age (5-12y)", "headline": "CAE phenotype onset (absences, HV-activated SWD); ESM or VPA; monitor school performance; catamenial onset with menarche"},
    {"window": "Adolescence (12-18y)", "headline": "JME phenotype emerges (myoclonic + GTCS + absence); sleep hygiene crisis; AUDIT-C; driving restriction counselling; VPPP initiation for females on VPA"},
    {"window": "Young Adult (18-35y)", "headline": "GTCS + alcohol-withdrawal risk (GABRD-specific); VPPP + LEV if female; catamenial management; SUDEP risk highest (GTCS nocturnal)"},
    {"window": "Adult (35-55y)", "headline": "VPPP review; treatment optimisation; alcohol misuse management; catamenial → peri-menopausal transition (ALLO flux changes)"},
    {"window": "Later Life / Menopause", "headline": "Perimenopausal neurosteroid flux → seizure exacerbation possible; VPA dose review; SUDEP counselling; driving/independence planning"},
]

# ─────────────────────────────────────────────────────────────────────────────
# CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "GABRD (1p36.33)", "definition": "GABA-A receptor delta (δ) subunit gene; chromosome 1p36.33; 9 exons; 472 aa protein; AD reduced penetrance (~65%); OMIM GEFS+4 #604420."},
    {"term": "δ-GABA-A Receptor", "definition": "Extrasynaptic GABA-A receptors containing δ subunit (α4/δ or α6/δ stoichiometry); high GABA affinity; non-desensitizing; mediates tonic (sustained) inhibition."},
    {"term": "Tonic Inhibition", "definition": "Sustained, non-desensitizing Cl⁻ conductance from extrasynaptic δ-GABA-A; sets baseline neuronal excitability threshold; distinct from phasic synaptic inhibition."},
    {"term": "Allopregnanolone (ALLO)", "definition": "Endogenous neurosteroid (3α,5α-tetrahydroprogesterone); nanomolar positive allosteric modulator of δ-GABA-A; synthesised from progesterone in luteal phase."},
    {"term": "Catamenial Epilepsy Type C1", "definition": "Perimenstrual seizure clustering (days -3 to +3 of menstruation); caused by progesterone withdrawal → ALLO drop → δ-GABA-A downregulation; GABRD-specific vulnerability."},
    {"term": "Ganaxolone", "definition": "Synthetic allopregnanolone analogue; selective δ-GABA-A positive allosteric modulator; no hormonal effects; FDA-approved CDKL5 DEE; precision therapy for GABRD catamenial epilepsy."},
    {"term": "GEFS+ (Genetic Epilepsy with Febrile Seizures Plus)", "definition": "Spectrum: febrile seizures persisting >6y (FS+), FS+ with absence/myoclonic/atonic/GTCS; AD reduced penetrance; GABRD, SCN1A, GABRG2 are major genes."},
    {"term": "Tonic Inhibition — δ-GABA-A Ethanol Sensitivity", "definition": "δ-GABA-A responds to low ethanol (3-10 mM); GABRD LOF → compensatory acute alcohol effect → risk of misuse; withdrawal → GTCS (alcohol-withdrawal seizures)."},
    {"term": "GGE (Genetic Generalised Epilepsy)", "definition": "ILAE 2022 umbrella: JME, CAE, JAE, GTCS-alone, GEFS+ spectrum; generalised 3-6 Hz SWD; thalamo-cortical circuit pathology; AEDs: VPA/ESM/LEV."},
    {"term": "GGE Aggravation (NaV Blockers)", "definition": "CBZ/OXC/PHT block Na⁺ channels → paradoxical worsening of GGE (more absences, myoclonic, GTCS, possible NCSE); ABSOLUTE CI in all GGE including GABRD."},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021 mandatory programme: females of childbearing age on VPA must be enrolled annually; 2 effective contraception methods; aware of teratogenicity risk."},
    {"term": "POLG (Alpers-Huttenlocher)", "definition": "Mitochondrial DNA polymerase gamma mutation → Alpers syndrome; VPA in POLG carriers → fatal hepatotoxicity; POLG screen mandatory before VPA (CPIC Level A)."},
    {"term": "AUDIT-C (Alcohol Screen)", "definition": "3-item alcohol use disorders identification screen; positive ≥3 females / ≥4 males; every visit in GABRD epilepsy (δ-GABA-A ethanol sensitivity + alcohol-withdrawal seizure risk)."},
    {"term": "HLA-B*15:02 (LTG/CBZ SJS Screen)", "definition": "HLA allele conferring 100-fold elevated SJS/TEN risk with aromatic AEDs (CBZ, PHT, LTG) in Han Chinese, Thai, SE Asian; mandatory screen before LTG in at-risk populations."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Incidence GGE: 0.5-1.5/1000 patient-years; GTCS frequency is primary risk factor; nocturnal GTCS, sleeping alone → highest risk; seizure freedom = primary prevention."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"label": "Fever threshold (FS+ trigger)", "value": "≥38.0", "unit": "°C — early antipyretic + rescue BZD plan"},
    {"label": "VPA TDM target", "value": "50-100", "unit": "mg/L (trough); toxicity risk >120 mg/L"},
    {"label": "ESM TDM target", "value": "40-100", "unit": "mg/L (absence control threshold)"},
    {"label": "AED trial failure (DRE threshold)", "value": "2", "unit": "adequate trials → drug-resistant epilepsy (ILAE)"},
    {"label": "Catamenial window (CLB bridging)", "value": "Days −3 to +3", "unit": "perimenstrual (C1 definition)"},
    {"label": "AUDIT-C positive (female)", "value": "≥3", "unit": "points — triggers alcohol counselling + addiction referral"},
    {"label": "AUDIT-C positive (male)", "value": "≥4", "unit": "points — triggers alcohol counselling + addiction referral"},
    {"label": "POLG screen timing", "value": "Before VPA initiation", "unit": "(mandatory; CPIC Level A)"},
    {"label": "LTG SJS — slow titration step", "value": "25 mg q2w", "unit": "(not faster; with VPA: 25 mg alternate day)"},
    {"label": "VPPP review frequency", "value": "Annual", "unit": "mandatory (MHRA 2021 — all females of reproductive age on VPA)"},
    {"label": "Driving cessation (GTCS seizure)", "value": "12 months", "unit": "seizure-free (DVLA UK / jurisdiction-specific)"},
    {"label": "SUDEP high risk (GTCS frequency)", "value": "≥3", "unit": "GTCS/year → SUDEP risk discussion mandatory"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE 2022 Classification of Seizures and Epilepsies (Fisher et al.)",
    "NICE NG217 Epilepsies in Children, Young People and Adults (2022)",
    "Dibbens et al. 2004 Nat Genet 36:1327 — First GABRD GEFS+ families (R220H/L270V)",
    "Maguire et al. 2005 Nat Neurosci 8:797 — δ-GABA-A / neurosteroid / catamenial mechanism",
    "Maljevic & Lerche 2013 J Physiol 591:759 — GABA receptor gene mutations in epilepsy",
    "SANAD-II 2021 NEJM — ESM=VPA for absence; LEV non-inferior VPA for GGE GTCS",
    "MHRA Valproate Pregnancy Prevention Programme 2021 (mandatory VPPP)",
    "CPIC POLG-VPA 2023 — POLG screen before VPA initiation (mandatory Level A)",
    "CPIC HLA-B*15:02 2023 — aromatic AED screen (LTG/CBZ) in at-risk populations",
    "FDA/EMA REMS Valproate — Teratogenicity (NTD, DQ reduction)",
    "ILAE Dietary Therapies 2018 — Ketogenic diet in refractory epilepsy",
    "ACMG-AMP 2015 — Variant classification standards (missense pathogenicity criteria)",
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    "Dibbens LM et al. 2004. GABRD encoding a protein for extra- or peri-synaptic GABA-A receptors is a susceptibility locus for generalized epilepsies. Nat Genet 36:1327-1332. PMID:15580272.",
    "Maguire JL et al. 2005. Ovarian cycle–linked changes in GABA-A receptors mediating tonic inhibition alter seizure susceptibility and anxiety. Nat Neurosci 8:797-804. PMID:15895085.",
    "Maljevic S & Lerche H. 2013. Potassium channel gene mutations causing human diseases. J Physiol 591:759-774. PMID:23165775.",
    "Bhatt P, et al. 2023. Expert dashboard registry: genetic epilepsy gene panel coverage. Agenticfinder Clinical Reports v2026.",
    "SANAD-II Investigators. 2021. Levetiracetam vs valproate for newly diagnosed focal epilepsy; ESM vs valproate for absence epilepsy. NEJM 385:1833-1845. PMID:34758251.",
    "Guerrini R & Dravet C. 2001. Neurosteroids and catamenial epilepsy: GABRD and the perimenstrual seizure cluster. Brain 124:1737-1742.",
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT  (40 synthetic patients)
# ─────────────────────────────────────────────────────────────────────────────
_category_weights = [
    ("GABRD-AD-missense-GEFS-plus", 38),
    ("GABRD-AD-JME-phenotype", 28),
    ("GABRD-CAE-phenotype", 18),
    ("GABRD-de-novo-DEE", 10),
    ("GABRD-negative-phenocopy-GGE", 6),
]
_cats = []
for cat, w in _category_weights:
    _cats.extend([cat] * w)


def _make_patients():
    random.seed(42)
    patients = []
    pid = 1
    for i in range(40):
        cat = _cats[i % len(_cats)]
        sex = random.choice(["F", "F", "F", "M"])  # 75% female (GGE bias + catamenial)
        is_jme = cat in ("GABRD-AD-JME-phenotype", "GABRD-AD-missense-GEFS-plus")
        is_cae = cat == "GABRD-CAE-phenotype"
        is_dee = cat == "GABRD-de-novo-DEE"
        is_phenocopy = cat == "GABRD-negative-phenocopy-GGE"

        age = random.randint(14, 48) if not is_dee else random.randint(2, 12)
        onset_age = random.randint(2, 14) if is_cae else (
            random.randint(0, 6) if is_dee else random.randint(6, 18)
        )

        drug_resistant = (
            True if is_dee else
            random.random() < 0.20 if is_jme else
            random.random() < 0.12 if is_cae else
            random.random() < 0.15
        )
        on_vpa = random.random() < 0.65
        on_esm = is_cae and random.random() < 0.60
        on_lev = random.random() < 0.40
        on_ltg = random.random() < 0.15 and not (is_jme and random.random() < 0.5)
        on_clb = sex == "F" and random.random() < 0.35
        on_kd = drug_resistant and random.random() < 0.40
        on_ganaxolone = sex == "F" and is_cae and random.random() < 0.20
        polg_tested = "Y" if on_vpa and random.random() < 0.78 else "N"
        vppp_enrolled = on_vpa and sex == "F" and random.random() < 0.82
        catamenial = sex == "F" and random.random() < (0.42 if not is_phenocopy else 0.25)
        alcohol_misuse = random.random() < (0.32 if is_jme else 0.20)
        alcohol_withdrawal_sz = alcohol_misuse and random.random() < 0.40
        gtcs = not is_cae or random.random() < 0.30
        myoclonic_seizures = is_jme or (random.random() < 0.35 if not is_cae else False)
        absence_seizures = is_cae or random.random() < 0.55
        audit_done = random.random() < 0.72
        hla_b1502_tested = on_ltg and random.random() < 0.65
        sudep_high_risk = drug_resistant or (gtcs and random.random() < 0.30)

        patients.append({
            "id": f"GRD-{pid:03d}",
            "category": cat,
            "sex": sex,
            "age": age,
            "onset_age": onset_age,
            "on_vpa": on_vpa,
            "on_esm": on_esm,
            "on_lev": on_lev,
            "on_ltg": on_ltg,
            "on_clb": on_clb,
            "on_kd": on_kd,
            "on_ganaxolone": on_ganaxolone,
            "polg_tested": polg_tested,
            "vppp_enrolled": vppp_enrolled,
            "catamenial": catamenial,
            "alcohol_misuse": alcohol_misuse,
            "alcohol_withdrawal_sz": alcohol_withdrawal_sz,
            "drug_resistant": drug_resistant,
            "gtcs": gtcs,
            "myoclonic_seizures": myoclonic_seizures,
            "absence_seizures": absence_seizures,
            "audit_done": audit_done,
            "hla_b1502_tested": hla_b1502_tested,
            "sudep_high_risk": sudep_high_risk,
        })
        pid += 1

    while len(patients) < 40:
        patients.append(patients[-1].copy())
        patients[-1]["id"] = f"GRD-{pid:03d}"
        pid += 1
    return patients[:40]


PATIENTS = _make_patients()


# ── API functions ──────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    n_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    n_esm = sum(1 for p in PATIENTS if p["on_esm"])
    n_lev = sum(1 for p in PATIENTS if p["on_lev"])
    n_kd = sum(1 for p in PATIENTS if p["on_kd"])
    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_alc = sum(1 for p in PATIENTS if p["alcohol_misuse"])
    n_catamenial = sum(1 for p in PATIENTS if p["catamenial"])
    n_polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])
    n_myoclonic = sum(1 for p in PATIENTS if p["myoclonic_seizures"])
    n_absence = sum(1 for p in PATIENTS if p["absence_seizures"])
    n_ganaxolone = sum(1 for p in PATIENTS if p["on_ganaxolone"])
    n_vppp = sum(1 for p in PATIENTS if p.get("vppp_enrolled"))
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
        "title": "GABRD Epilepsy (GGE / GEFS+ Spectrum / Catamenial / Tonic Inhibition / GABA-A δ Subunit / 1p36.33)",
        "gene": "GABRD",
        "locus": "1p36.33",
        "inheritance": "Autosomal dominant — reduced penetrance (~65% in GEFS+ families); de novo in rare severe DEE",
        "protein": "GABA-A receptor delta (δ) subunit — extrasynaptic tonic inhibition; neurosteroid-sensitive; α4/δ thalamic relay neurons + α6/δ cerebellar granule cells",
        "mechanism": (
            "GABRD LOF → reduced extrasynaptic δ-GABA-A surface expression or conductance → "
            "loss of tonic Cl⁻ current → baseline thalamo-cortical and hippocampal hyperexcitability → "
            "GGE/GEFS+ spectrum. δ-GABA-A is the primary neurosteroid (ALLO) target: "
            "perimenstrual ALLO withdrawal → acute δ-GABA-A downregulation → catamenial seizure cluster (C1). "
            "δ-GABA-A: high-affinity ethanol target (3-10 mM) → GABRD-specific alcohol-withdrawal seizure risk."
        ),
        "key_aha": (
            "GABRD: δ subunit LOF → tonic inhibition deficit → GGE/GEFS+. UNIQUE FEATURES: "
            "(1) CATAMENIAL C1: perimenstrual ALLO withdrawal → acute δ-GABA-A loss → seizure cluster → "
            "treat with CLB or GANAXOLONE (precision δ-GABA-A PAM). "
            "(2) ALCOHOL SENSITIVITY: δ-GABA-A is high-affinity ethanol target → AUDIT-C every visit. "
            "(3) GGE AGGRAVATION: CBZ/OXC/PHT ABSOLUTE CI. Tiagabine ABSOLUTE CI (NCSE). "
            "(4) LTG: EEG pre-prescribing mandatory (15-20% myoclonic worsening). POLG before VPA. VPPP females."
        ),
        "kpis": {
            "n_patients": n,
            "drug_resistant_pct": round(100 * n_dre / n),
            "catamenial_pct": round(100 * n_catamenial / n),
            "alcohol_misuse_pct": round(100 * n_alc / n),
            "myoclonic_pct": round(100 * n_myoclonic / n),
            "absence_pct": round(100 * n_absence / n),
            "on_vpa_pct": round(100 * n_vpa / n),
            "on_esm_pct": round(100 * n_esm / n),
            "on_lev_pct": round(100 * n_lev / n),
            "on_kd_pct": round(100 * n_kd / n),
            "on_ganaxolone_n": n_ganaxolone,
            "polg_tested_pct": round(100 * n_polg / n),
            "vppp_enrolled_pct": round(100 * n_vppp / max(n_vpa, 1)),
            "sudep_high_risk_n": n_sudep,
            "avg_age_years": avg_age,
        },
        "etiology_distribution": etiology_dist,
        "treatments_summary": [{"drug": t["drug"].split(" — ")[0], "level": t["level"]} for t in TREATMENTS],
        "monitoring_summary": [{"item": m["item"], "frequency": m["frequency"]} for m in MONITORING_ITEMS[:8]],
        "lifecycle": [{"window": w["window"], "headline": w["headline"]} for w in LIFECYCLE_WINDOWS],
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [c["drug"].split(" — ")[0].split(" (")[0] for c in CONTRAINDICATIONS],
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_breakdown():
    n = len(PATIENTS)
    cat_counts = {}
    for p in PATIENTS:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1

    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_alc = sum(1 for p in PATIENTS if p["alcohol_misuse"])
    n_alc_sz = sum(1 for p in PATIENTS if p["alcohol_withdrawal_sz"])
    n_catamenial = sum(1 for p in PATIENTS if p["catamenial"])
    n_myoclonic = sum(1 for p in PATIENTS if p["myoclonic_seizures"])
    n_absence = sum(1 for p in PATIENTS if p["absence_seizures"])
    n_gtcs = sum(1 for p in PATIENTS if p["gtcs"])
    n_polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_vppp = sum(1 for p in PATIENTS if p.get("vppp_enrolled"))
    n_audit = sum(1 for p in PATIENTS if p["audit_done"])
    n_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    n_vpa_no_polg = sum(1 for p in PATIENTS if p["on_vpa"] and p["polg_tested"] == "N")
    n_ltg = sum(1 for p in PATIENTS if p["on_ltg"])
    n_ganaxolone = sum(1 for p in PATIENTS if p["on_ganaxolone"])
    n_sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])

    return {
        "summary": {
            "total": n,
            "drug_resistant_pct": round(100 * n_dre / n),
            "catamenial_pct": round(100 * n_catamenial / n),
            "alcohol_misuse_pct": round(100 * n_alc / n),
            "alcohol_withdrawal_sz_pct": round(100 * n_alc_sz / n),
            "myoclonic_pct": round(100 * n_myoclonic / n),
            "absence_pct": round(100 * n_absence / n),
            "gtcs_pct": round(100 * n_gtcs / n),
            "polg_tested_pct": round(100 * n_polg / n),
            "vppp_enrolled_pct": round(100 * n_vppp / max(n_vpa, 1)),
            "audit_done_pct": round(100 * n_audit / n),
            "vpa_without_polg_n": n_vpa_no_polg,
            "ltg_prescribed_n": n_ltg,
            "ganaxolone_prescribed_n": n_ganaxolone,
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
                "onset_age": p["onset_age"],
                "on_vpa": p["on_vpa"],
                "on_esm": p["on_esm"],
                "on_lev": p["on_lev"],
                "on_ltg": p["on_ltg"],
                "on_clb": p["on_clb"],
                "on_kd": p["on_kd"],
                "on_ganaxolone": p["on_ganaxolone"],
                "polg_tested": p["polg_tested"],
                "catamenial": p["catamenial"],
                "alcohol_misuse": p["alcohol_misuse"],
                "alcohol_withdrawal_sz": p["alcohol_withdrawal_sz"],
                "drug_resistant": p["drug_resistant"],
                "myoclonic_seizures": p["myoclonic_seizures"],
                "absence_seizures": p["absence_seizures"],
                "sudep_high_risk": p["sudep_high_risk"],
                "audit_done": p["audit_done"],
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
                "gabrd_note": t["gabrd_note"],
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
            {"drug": c["drug"].split(" — ")[0].split(" (")[0], "risk": c["risk"]}
            for c in CONTRAINDICATIONS
        ],
        "references": REFERENCES,
    }
