"""
SCN1A Epilepsy — Dravet Syndrome / GEFS+ / NaV1.1 Channelopathy / 2q24.3
==========================================================================
40-patient cohort · SCN1A de novo LOF · Dravet syndrome · GEFS+ spectrum · DEE6

SCN1A BIOLOGY:
SCN1A (2q24.3) encodes NaV1.1 — the predominant voltage-gated sodium channel alpha
subunit in fast-spiking parvalbumin-positive (PV+) GABAergic interneurons of the
cerebral cortex, hippocampus, and cerebellum. NaV1.1 is uniquely critical because
these PV+ interneurons (basket cells, chandelier cells) are the primary inhibitory
"brake" on pyramidal neuron firing. Loss-of-function (LOF) variants → PV interneuron
haploinsufficiency → failed inhibitory interneuron firing at high frequencies →
cortical disinhibition → hyperexcitability → Dravet syndrome (severe) or GEFS+
(milder, familial).

KEY NaV1.1 / SCN1A BIOLOGY:
  - NaV1.1 is expressed at the axon initial segment (AIS) and nodes of Ranvier of
    PV+ fast-spiking inhibitory interneurons — the neural loci most dependent on
    Na+ current density for sustained high-frequency discharge.
  - LOF haploinsufficiency: one functional copy insufficient to sustain PV interneuron
    high-frequency firing (target requires >80% channel density to reach 300–500 Hz
    burst rates). Pyramidal cells (which co-express NaV1.2 and NaV1.6) are relatively
    unaffected by NaV1.1 LOF — hence the paradox: Nav1.1 blockers WORSEN seizures.
  - Temperature sensitivity: NaV1.1 single-channel conductance declines ~30% per 1°C
    rise above 37°C. In SCN1A LOF: fever → residual NaV1.1 function further impaired →
    PV interneurons fail at fever threshold → febrile hemiclonic seizures (Dravet hallmark).
  - Cardiac NaV1.1: expressed in cardiac Purkinje fibres and sinoatrial node → SUDEP
    risk in Dravet may involve cardiac arrhythmia (ictal bradycardia/asystole), not
    only respiratory arrest.
  - Sodium channel blockers (CBZ, OXC, PHT, LTG-high-dose): block NaV1.1 in
    PV+ interneurons → further reduce inhibitory tone → WORSEN Dravet seizures in
    70% of patients. This is one of the most important clinical findings in epilepsy
    pharmacology: the drug class most widely used for epilepsy is CONTRAINDICATED in
    the commonest severe genetic epilepsy.
  - NaV1.1 is the most clinically important epilepsy gene: >1,800 pathogenic variants
    catalogued (LOVD SCN1A database); ~80% de novo; Dravet incidence 1:15,000–1:22,000;
    accounts for ~80% of Dravet syndrome.

ALLELIC DISORDER SPECTRUM (DEE6 / SMEI / GEFS+ / Dravet):
  1. Dravet syndrome (SMEI) — de novo LOF/truncating de novo ~45%
     Classic Dravet onset 5-8 months: prolonged febrile hemiclonic seizure,
     then progressive drug-resistant epilepsy (myoclonic, absence, focal,
     GTCS), developmental plateau or regression after first year, ataxia.
  2. GEFS+ spectrum (AD familial) — inherited missense ~25%
     Febrile seizures 6 months – 6 years, FS+ (beyond 6 years), occasional
     afebrile GTCS. Favourable prognosis. Penetrance ~70-80%.
  3. Borderline SMEI / Dravet-like (SMEB) — de novo ~15%
     Clinical features of Dravet without meeting all criteria:
     absent or rare myoclonus; ESES/CSWS pattern in some; DD present.
  4. SCN1A-related mild epilepsy / pure FS+ — mild familial ~10%
     Pure febrile seizures or rare afebrile GTCS; no intellectual disability;
     full seizure freedom by adolescence in 80%.
  5. Phenocopy / SCN1A-negative — non-SCN1A ~5%
     GABRB2-Dravet-like, GABRB3-Dravet-like, HCN1-Dravet-like, SCN1B-GEFS+,
     protocol negative — Dravet phenotype without SCN1A variant.

CRITICAL DRUG RULES FOR SCN1A / DRAVET:
  ABSOLUTE CI: Sodium channel blockers
  - CBZ (Carbamazepine): ABSOLUTE CI in Dravet — aggravates all seizure types in
    ~70%; NaV1.1 PV-interneuron block worsens inhibitory failure.
  - OXC (Oxcarbazepine): Same mechanism, ABSOLUTE CI.
  - PHT (Phenytoin): ABSOLUTE CI. Although fast-acting IV PHT used historically for
    status, chronic PHT worsens Dravet. IV PHT may be used acutely but transition away.
  - LTG (Lamotrigine) high dose: ABSOLUTE CI. Blocks NaV1.1; Dravet myoclonus/GTCS
    can worsen dramatically on LTG. LTG at low doses occasionally tolerated in
    GEFS+ spectrum, but NEVER in Dravet.
  - Tiagabine: ABSOLUTE CI — extrasynaptic GABA / NCSE.

  FIRST-LINE / PRECISION THERAPIES:
  - VPA (Valproate): Level A backbone (POLG MANDATORY before use).
  - Stiripentol (Diacomit): Level A — triple therapy VPA+CLB+STP (Dravet-specific FDA
    indication 2018); STAT3 inhibitor + GABA-A modulator.
  - Fenfluramine (Fintepla): Level A — serotonin releaser + sigma-1 receptor modulator +
    HCN1 activator; FDA 2020 for Dravet (REMS: echocardiography q6M, cardiac valvulopathy).
  - Cannabidiol (Epidiolex): Level A — FDA 2018 for Dravet ≥2 years;
    mechanism includes TRPV1, GPR55, GlyR modulation; LFT monitoring (hepatotoxicity).
  - CLB (Clobazam): Level B adjunct; 1,5-BDZ; GABA-A positive modulator.
  - Topiramate: Level B adjunct.
  - KD (Ketogenic Diet): Level B for drug-resistant Dravet.
"""

import random
from datetime import datetime

SEED = 9205  # dashboard 205
random.seed(SEED)

# ── Etiology Catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "SCN1A Dravet syndrome (SMEI) — de novo LOF / truncating / missense",
        "n": 18, "pct": 45,
        "category": "SCN1A-Dravet-SMEI-de-novo-LOF",
        "mechanism": (
            "The most severe and most common SCN1A phenotype: Dravet syndrome (Severe "
            "Myoclonic Epilepsy of Infancy, SMEI) is caused by de novo LOF variants — "
            "truncating (nonsense, frameshift, splice-site) or missense variants disrupting "
            "NaV1.1 channel trafficking, gating, or expression. Haploinsufficiency mechanism: "
            "one functional SCN1A copy insufficient to sustain PV+ fast-spiking inhibitory "
            "interneuron discharge at the high rates (200–500 Hz) required to brake "
            "pyramidal network activity. Pathophysiology cascade: NaV1.1 LOF → PV basket + "
            "chandelier cell axon initial segment Na+ current deficit → failed inhibitory "
            "surround → pyramidal neuron network disinhibition → seizures. Key clinical "
            "features: onset 5-8 months with prolonged febrile hemiclonic seizure (Dravet "
            "hallmark); subsequent drug-resistant myoclonic, absence, focal, GTCS; "
            "developmental plateau at 12-24 months; progressive cognitive decline; ataxia "
            "(cerebellar NaV1.1 Purkinje cell axon involvement); SUDEP risk elevated. "
            "~85% de novo; ~15% somatic mosaic SCN1A (require deep-read sequencing ≥500×). "
            "Truncating variants slightly higher phenotype severity than missense."
        ),
        "eeg_correlate": (
            "Dravet EEG evolution: (1) Age <1 year: often NORMAL interictal EEG — the "
            "'Dravet paradox': clinically devastating disease with initially normal background. "
            "(2) First seizure (febrile hemiclonic): focal rhythmic discharge, hemisphere "
            "variable (right > left), high-amplitude rhythmic theta evolving to clonic "
            "discharges, lasting 15-60+ minutes. (3) Age 1-5 years: multifocal IEDs "
            "(frontotemporal, occipital, multiregional); generalised irregular spike-wave "
            "(2-3.5 Hz) during absences; photosensitive responses (50%); background "
            "normalises inter-ictally. (4) Myoclonic EEG: generalised polyspike or irregular "
            "spike-wave; myoclonic jerks time-locked. (5) Older: background slowing "
            "develops; focal centrotemporal IEDs; periodic lateralised epileptiform "
            "discharges (PLEDs) during prolonged febrile status."
        ),
        "mri_finding": (
            "Dravet MRI: (1) NORMAL in 50-60% at all ages — characteristic: EEG and MRI "
            "normal despite severe clinical course. (2) Hippocampal sclerosis (HS) in 15-20% "
            "— develops after repeated prolonged febrile status; T2 signal increase, atrophy, "
            "loss of internal architecture (dentate gyrus, CA1 disruption). (3) Generalised "
            "cerebral atrophy (15%) after many years of DRE. (4) Acute MRI changes during "
            "prolonged SE: cortical DWI restriction (cytotoxic oedema), FLAIR signal change. "
            "(5) Cerebellar atrophy in 10% (long-standing ataxia group). MRI at diagnosis; "
            "repeat if seizure pattern changes or prolonged SE episode."
        ),
    },
    {
        "etiology": "SCN1A GEFS+ spectrum (AD familial) — inherited missense / truncating",
        "n": 10, "pct": 25,
        "category": "SCN1A-GEFS-plus-AD-familial",
        "mechanism": (
            "Generalised Epilepsy with Febrile Seizures Plus (GEFS+): autosomal dominant "
            "SCN1A variants inherited from a mildly affected parent, with 70-80% penetrance. "
            "Missense variants that are hypomorphic (partial LOF) — residual NaV1.1 function "
            "sufficient for most neurological function but insufficient for PV interneuron "
            "firing reserve at fever threshold. Clinical manifestations span a spectrum: "
            "febrile seizures alone (FS, most common), febrile seizures plus afebrile GTCS "
            "(FS+), febrile seizures plus absences, or febrile seizures plus myoclonic "
            "seizures. Penetrance varies within families — the same variant may cause Dravet "
            "in one sibling (de novo equivalent effect of the inherited variant) and mild FS "
            "in a parent. Prognosis generally favourable: 70-80% achieve seizure freedom "
            "in adolescence. Sodium channel blockers are CONTRAINDICATED in the Dravet end "
            "of GEFS+ spectrum but may be used cautiously in pure FS+ where no myoclonic "
            "component is present — a clinically challenging distinction requiring careful "
            "monitoring. POLG mandatory before VPA even in mild GEFS+ forms."
        ),
        "eeg_correlate": (
            "GEFS+ EEG: (1) Mild GEFS+ (pure FS): interictal EEG often NORMAL or shows "
            "mild centrotemporal IEDs. (2) FS+ with afebrile GTCS: generalised spike-wave "
            "(3-4 Hz) in wakeful state; background NORMAL between episodes. (3) GEFS+ with "
            "absences: 3 Hz generalised SWD during clinical absences — shorter runs than "
            "classic CAE; background normal. (4) Febrile seizure ictal: generalised rhythmic "
            "discharge, often diffuse high-amplitude delta/theta evolving from focal onset. "
            "Key distinction from Dravet: normal background, no multifocal IEDs, no "
            "persistent focal abnormalities. Photoparoxysmal response rare in mild GEFS+ "
            "(vs 50% in Dravet)."
        ),
        "mri_finding": (
            "GEFS+ MRI: NORMAL in >95% of all GEFS+ cases. "
            "No structural findings expected. MRI performed once at initial workup to "
            "exclude structural lesion (cortical dysplasia, tumour). No serial MRI "
            "required if development normal and seizures respond to AEDs. "
            "If MRI abnormal in GEFS+ patient → reassess clinical diagnosis; "
            "structural epilepsy may co-exist or phenocopy diagnosis."
        ),
    },
    {
        "etiology": "SCN1A Borderline SMEI / Dravet-like (SMEB) — de novo partially LOF",
        "n": 6, "pct": 15,
        "category": "SCN1A-SMEB-Dravet-like",
        "mechanism": (
            "Borderline SMEI (SMEB) or Dravet-like epilepsy: de novo SCN1A variants with "
            "intermediate functional consequences — partial LOF or missense variants with "
            "trafficking/gating defects that reduce NaV1.1 surface expression or conductance "
            "by 40-70% (vs >80% in classic Dravet truncating variants). Clinical: Dravet-like "
            "seizure phenotype (febrile hemiclonic onset, drug-resistant, fever-sensitive) but "
            "absent or infrequent myoclonic component, and milder intellectual disability "
            "than classic SMEI. In some SMEB patients: CSWS/ESES pattern develops "
            "and contributes to cognitive regression — different mechanism than classic Dravet "
            "(electrical status vs interneuron failure). SMEB prognosis: intermediate — "
            "40-60% achieve partial seizure control vs 5-10% in classic Dravet. "
            "Treatment principles identical to Dravet: AVOID NaV blockers; "
            "VPA + stiripentol/fenfluramine/cannabidiol backbone."
        ),
        "eeg_correlate": (
            "SMEB EEG: (1) Similar evolution to Dravet but milder multifocal burden. "
            "(2) Interictal: multifocal IEDs (less dense than classic Dravet); focal "
            "centrotemporal or occipital IEDs; generalised irregular SWD (2.5-3.5 Hz). "
            "(3) Ictal: febrile seizures: hemispheric clonic discharge; afebrile "
            "episodes: focal or multifocal onset. (4) CSWS/ESES: NREM sleep-activated "
            "continuous SWD (>85% of N2/N3 sleep) in ~25% — overnight EEG q12M. "
            "(5) Absence: atypical, short, irregular SWD. No myoclonic EEG pattern "
            "distinguishes SMEB from SMEI. Photoparoxysmal response in 35%."
        ),
        "mri_finding": (
            "SMEB MRI: Similar to Dravet — NORMAL in 60%; hippocampal T2 signal "
            "changes after prolonged SE in 10-15%; mild generalised atrophy in long-"
            "standing DRE. CSWS subgroup: may show signal changes in peri-rolandic cortex. "
            "MRI at diagnosis and after any prolonged seizure/SE event."
        ),
    },
    {
        "etiology": "SCN1A mild epilepsy / pure FS+ — mild AD familial (SCN1A hypomorph)",
        "n": 4, "pct": 10,
        "category": "SCN1A-mild-FS-plus-AD-familial",
        "mechanism": (
            "The mildest end of the SCN1A spectrum: hypomorphic variants with minimal "
            "functional impact on NaV1.1 channel activity (residual function >70%) causing "
            "pure febrile seizures or rarely afebrile GTCS, with NO intellectual disability "
            "and NO myoclonic component. Most patients achieve complete seizure freedom by "
            "late adolescence without medication. Family history typically shows multiple "
            "relatives with febrile seizures in an AD pattern. Genetic counselling important: "
            "1-2% risk of de novo conversion (new SCN1A variant in offspring) that may cause "
            "Dravet syndrome — counselling should address this spectrum. Management: "
            "VPA or LEV for rare breakthrough seizures. Sodium channel blockers are LOW-RISK "
            "in pure FS+ (no myoclonic/DRE component) but avoid if any clinical features "
            "suggestive of evolving Dravet."
        ),
        "eeg_correlate": (
            "Mild SCN1A EEG: NORMAL interictal EEG in 80%. Occasional mild posterior or "
            "centrotemporal IEDs in 20%. Generalised SWD only during brief absences if "
            "present. Background NORMAL. No multifocal IEDs. No persistent "
            "focal abnormalities. Photoparoxysmal response in 15% (lower than Dravet). "
            "EEG normalises completely in most patients by adolescence."
        ),
        "mri_finding": (
            "Mild SCN1A MRI: NORMAL in >99% — structural brain MRI "
            "completely normal. MRI performed once to exclude structural cause; "
            "no repeat necessary if normal development and good seizure response."
        ),
    },
    {
        "etiology": "SCN1A phenocopy — GABRB2 / GABRB3 / HCN1 / SCN1B (Dravet-like phenotype)",
        "n": 2, "pct": 5,
        "category": "SCN1A-phenocopy-GABRB2-GABRB3-HCN1-SCN1B",
        "mechanism": (
            "5% of patients with Dravet-like clinical phenotype have negative SCN1A "
            "sequencing and are reclassified after comprehensive panel: GABRB2 (Dravet-like "
            "DEE, febrile + myoclonic, GABA-A β2 subunit — same result as Nav1.1 LOF but "
            "via direct GABA-A dysfunction), GABRB3 (infantile spasms + Dravet-like, GABA-A β3), "
            "HCN1 (Dravet-like with myoclonic, hyperpolarisation-activated current modulation), "
            "or SCN1B (GEFS+, Nav β1 subunit auxiliary — reduces NaV1.1 surface expression). "
            "Critical: ~15% of clinical Dravet is SCN1A MOSAIC — standard exome (50× coverage) "
            "misses mosaic allele fractions 5-20%. Deep-read sequencing (500× minimum) or "
            "CNV panel required before diagnosing 'SCN1A-negative Dravet'. Drug rules for "
            "phenocopies: avoid NaV blockers (GABRB2/HCN1/SCN1B Dravet-like) as mechanism "
            "still involves interneuron dysfunction; VPA backbone appropriate."
        ),
        "eeg_correlate": (
            "Phenocopy EEG: clinically indistinguishable from SCN1A-Dravet by EEG pattern. "
            "GABRB2/GABRB3: multifocal IEDs, generalised SWD, fever-sensitive; "
            "HCN1: may have distinctive high-amplitude slow IEDs + myoclonic pattern; "
            "SCN1B: GEFS+ interictal EEG (fewer multifocal IEDs, milder background). "
            "Gene panel + deep-read SCN1A sequencing essential. Functional assay "
            "(patch-clamp) for VUS classification when clinically relevant."
        ),
        "mri_finding": (
            "Phenocopy MRI: GABRB2/GABRB3 phenocopies: mostly normal or hippocampal "
            "T2 changes post-SE (same as SCN1A-Dravet). HCN1: may show "
            "hippocampal signal changes. SCN1B: normal. Cannot distinguish from "
            "SCN1A-Dravet on MRI alone — gene sequencing is definitive."
        ),
    },
]

# ── Seizure Types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Prolonged febrile hemiclonic (Dravet hallmark)",
        "pct_patients": 90,
        "eeg_correlate": (
            "Focal hemispheric clonic discharge: unilateral rhythmic alpha/beta onset "
            "→ slows to theta/delta clonic pattern; alternating hemisphere in repeat "
            "episodes; duration 15–90 minutes (often status epilepticus at onset)"
        ),
        "clinical_tip": (
            "Dravet hallmark: prolonged febrile hemiclonic seizure at 5-8 months. "
            "Buccal midazolam 0.3 mg/kg or rectal diazepam 0.5 mg/kg at 5 minutes. "
            "IV lorazepam at 10 minutes. Emergency cooling (remove excess clothing, "
            "tepid sponging — NOT cold). Do NOT use IV phenytoin for chronic Dravet. "
            "AVOID CBZ/OXC/PHT. SCN1A panel urgently after first prolonged febrile "
            "hemiclonic seizure in an infant (Dravet diagnostic window)."
        ),
    },
    {
        "type": "Myoclonic seizures (Dravet / SMEI)",
        "pct_patients": 75,
        "eeg_correlate": (
            "Generalised polyspike or irregular spike-wave; myoclonic jerk time-locked "
            "to burst onset; brief (1-3 seconds); may occur in rapid series; "
            "stimulus-sensitive (sound, touch, light)"
        ),
        "clinical_tip": (
            "Myoclonic seizures in Dravet are exquisitely worsened by LTG (NaV1.1 block). "
            "LTG HIGH RISK → myoclonic storm. VPA first-line for myoclonic control. "
            "Stiripentol + VPA + CLB triple (Level A) superior to VPA alone. "
            "Clobazam PRN for myoclonic clusters. AVOID tiagabine (NCSE)."
        ),
    },
    {
        "type": "Generalised tonic-clonic seizures (GTCS / afebrile)",
        "pct_patients": 85,
        "eeg_correlate": (
            "Generalised paroxysmal fast activity or diffuse polyspike → clonic phase "
            "→ post-ictal delta suppression; tonic onset 10–25 Hz fast activity "
            "2–5 seconds before clonic phase; duration 2–5 minutes"
        ),
        "clinical_tip": (
            "GTCS in Dravet respond best to VPA + stiripentol + CLB triple. "
            "Fenfluramine (Fintepla) Level A reduces GTCS frequency by ~62% vs placebo. "
            "Cannabidiol (Epidiolex) Level A: 38% reduction in convulsive seizures. "
            "Emergency plan: buccal midazolam at >5 minutes. AVOID CBZ/PHT (worsens GTCS)."
        ),
    },
    {
        "type": "Atypical absence seizures",
        "pct_patients": 55,
        "eeg_correlate": (
            "Irregular generalised spike-wave 2.5–3.5 Hz (slower and more irregular "
            "than classic CAE 3 Hz SWD); behavioural arrest with incomplete "
            "unresponsiveness; may transition to myoclonic absence; "
            "duration 5–30 seconds"
        ),
        "clinical_tip": (
            "Atypical absences in Dravet are less responsive to ETX (ethosuximide) "
            "than CAE — ETX can worsen other Dravet seizure types. VPA is preferred "
            "for absence control in Dravet. CLB adjunct. Cannabidiol may reduce "
            "absence burden. Avoid LTG (myoclonic worsening risk). "
            "Cognitive impact of frequent absences: EEG monitoring to quantify."
        ),
    },
    {
        "type": "Focal (temporal/frontal) seizures",
        "pct_patients": 60,
        "eeg_correlate": (
            "Focal onset rhythmic discharge (frontotemporal or temporal most common); "
            "ictal onset may be bilateral at times; secondary generalisation frequent; "
            "post-ictal focal slowing"
        ),
        "clinical_tip": (
            "Focal seizures in Dravet do NOT respond to CBZ/OXC/PHT — ABSOLUTELY CI. "
            "VPA + fenfluramine + CLB backbone. Topiramate (Level B) may reduce focal "
            "seizure frequency. KD for drug-resistant focal component. "
            "Epilepsy surgery rarely feasible (multifocal, generalised disease)."
        ),
    },
    {
        "type": "Non-convulsive status epilepticus (NCSE / obtundation status)",
        "pct_patients": 40,
        "eeg_correlate": (
            "Continuous irregular generalised SWD (1.5–2.5 Hz) or diffuse theta "
            "slowing with superimposed IEDs; clinical: obtunded, confused, "
            "myoclonic twitching, drooling; may last hours to days if not treated"
        ),
        "clinical_tip": (
            "Dravet NCSE ('obtundation status') is a clinical emergency. "
            "IV benzodiazepine (lorazepam 0.1 mg/kg IV) or buccal midazolam. "
            "Stiripentol has anti-status properties. Do NOT use IV PHT/fosphenytoin "
            "(NaV1.1 block worsens status in Dravet). IV VPA 20-40 mg/kg is safe. "
            "Tiagabine is ABSOLUTE CI (GAT-1 block → NCSE induction)."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / febrile illness",
        "pct": 95,
        "threshold": "Fever ≥38°C increases seizure risk; ≥38.5°C high risk for prolonged seizure (lower threshold than GEFS+)",
        "management": (
            "Paracetamol at first sign of temperature rise (>37.5°C). Ibuprofen "
            "alternate if fever not controlled. Buccal midazolam 0.3 mg/kg immediately "
            "if seizure begins. Pre-emptive cooling: remove excess clothing, fan. "
            "Emergency plan: seizure >5 min → emergency BDZ. Hospitalise if prolonged "
            "febrile illness (IV hydration + AED level monitoring). "
            "AVOID prolonged fever — lower threshold: act at 37.5°C in Dravet (not 38°C)."
        ),
    },
    {
        "trigger": "Hyperthermia (hot bath, exercise, warm environment)",
        "pct": 82,
        "threshold": "Core temperature rise ≥0.5°C above baseline sufficient in Dravet (NaV1.1 temperature-gating failure)",
        "management": (
            "Lukewarm (not hot) baths only. Pre-cool bathing water: 34-36°C maximum. "
            "Exercise in cool environments; cool vest available for exercise; "
            "pre-hydration before exercise. Air conditioning essential in summer. "
            "Travel: avoid hot climates without air conditioning. "
            "NaV1.1 temperature coefficient (Q10 ≈ 3): per 1°C rise → 30% further "
            "reduction in residual NaV1.1 function → crisis threshold reached."
        ),
    },
    {
        "trigger": "Fever-associated vaccination (DTaP / MMR)",
        "pct": 65,
        "threshold": "Vaccine-associated fever at 24-48h post-vaccination; NOT the vaccine itself (fever is the trigger, not the antigen)",
        "management": (
            "CRITICAL: Vaccination is recommended in Dravet — risks of disease >> risk of "
            "vaccine-associated seizure. Pre-medicate: paracetamol 15 mg/kg 30 min before "
            "vaccination + 4-6 hourly for 24-48h post-vaccination (fever prevention). "
            "Post-vaccination observation 30 min. Emergency BDZ plan in place before. "
            "Do NOT delay or withhold routine vaccinations. ACIP/ILAE recommend full "
            "vaccination schedule for Dravet. If vaccine-associated seizure occurs: "
            "continue future vaccines with fever prevention protocol."
        ),
    },
    {
        "trigger": "Sleep deprivation / irregular sleep schedule",
        "pct": 70,
        "threshold": "Reduction of >2 hours from normal sleep duration sufficient; overnight travel, illness disrupting sleep",
        "management": (
            "Strict sleep hygiene: consistent bedtime, avoid late evenings, "
            "blackout curtains, consistent 10-12 hours sleep for paediatric Dravet. "
            "Evening CLB dose optimised to cover overnight. Sleep tracking (app/actigraphy). "
            "Nocturnal monitoring: mattress sensor (Emfit, SAMi-3) for detection of "
            "nocturnal convulsive seizures (SUDEP risk). "
            "Overnight EEG if CSWS suspected (cognitive regression + sleep-disrupted)."
        ),
    },
    {
        "trigger": "Photosensitivity (visual stimuli, flicker)",
        "pct": 50,
        "threshold": "Photic stimulation 15-20 Hz on EEG testing; natural triggers: sunlight through trees, TV, video games",
        "management": (
            "Photoparoxysmal response on EEG in ~50% of Dravet. Screen time limits: "
            "break every 20 minutes; anti-glare glasses; monitor at ≥100 Hz refresh rate; "
            "one eye covered when in bright environments. Fenfluramine shown to reduce "
            "photosensitivity in Dravet. Sunglasses outdoors. Avoid discotheques/strobe. "
            "School: notify teacher re computer/projector. Clobazam PRN for "
            "photosensitivity-triggered cluster prevention."
        ),
    },
    {
        "trigger": "Missed AED dose / therapeutic drug level drop",
        "pct": 68,
        "threshold": ">4h beyond scheduled VPA/stiripentol/CLB dose; VPA level <50 mg/L",
        "management": (
            "Dose alarm system (phone/watch). Caregiver training. Do NOT double-dose "
            "(VPA and STP toxicity risk). VPA trough level monitoring. Stiripentol must "
            "be taken WITH VPA — stand-alone stiripentol is ineffective (GABA-A modulation "
            "synergy). If cluster develops after missed dose: buccal midazolam per "
            "emergency plan; contact epilepsy specialist if repeat cluster."
        ),
    },
    {
        "trigger": "Sodium channel blocker exposure (accidental/in hospital)",
        "pct": 45,
        "threshold": "Even single dose of CBZ/PHT/OXC in Dravet patients can trigger severe seizure worsening",
        "management": (
            "MedicAlert bracelet / epilepsy ID card: 'DRAVET SYNDROME — DO NOT GIVE "
            "SODIUM CHANNEL BLOCKERS (carbamazepine/phenytoin/oxcarbazepine/lamotrigine). "
            "Call epilepsy specialist.' Hospitalised Dravet patients: pharmacy alert "
            "embedded in electronic prescribing system. Emergency departments: "
            "show printed emergency protocol. Carry emergency letter from treating "
            "neurologist specifying ABSOLUTE CI drugs."
        ),
    },
    {
        "trigger": "Intercurrent gastroenteritis / AED absorption failure",
        "pct": 58,
        "threshold": "Vomiting within 30 minutes of VPA dose → subtherapeutic level; diarrhoeal illness → malabsorption",
        "management": (
            "VPA liquid formulation (better absorption control). IV VPA for hospitalised "
            "GI illness (maintains therapeutic levels). Stiripentol: give with food. "
            "Cannabidiol: food absorption — take with fatty meal (4-fold ↑ absorption). "
            "Electrolyte monitoring during GI illness (hyponatraemia risk especially "
            "if on CLB which can potentiate SIADH). Emergency plan during GI illness: "
            "contact epilepsy nurse / specialist early."
        ),
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "VPA (Sodium Valproate)",
        "evidence": "Level A (backbone — Dravet / GEFS+)",
        "indication": "First-line backbone for Dravet syndrome and GEFS+ — broad-spectrum, anti-myoclonic, anti-absence, anti-GTCS",
        "dose_moa": (
            "20-40 mg/kg/day PO divided BD-TDS; IV infusion for SE: 25-40 mg/kg loading. "
            "Target VPA trough 75-100 mg/L (higher end for Dravet than GEFS+). "
            "MOA: NaV channel blockade (I_NaP reduction — distinct from PV-interneuron-damaging "
            "CBZ/PHT: VPA acts on Na+ channels in excitatory neurons without PV-interneuron "
            "specific toxicity), GABA-T inhibition (↑synaptic GABA), T-type Ca²⁺ blockade "
            "(thalamic oscillation suppression), HCN channel modulation, HDAC inhibition "
            "(epigenetic neuroprotection). Broad-spectrum efficacy explains VPA's "
            "cornerstone role in Dravet (myoclonic + GTCS + absence + focal coverage)."
        ),
        "efficacy": "40-60% ≥50% seizure reduction as monotherapy in Dravet; 70-80% response in GEFS+; superior when combined with stiripentol (Level A triple)",
        "monitoring": [
            "POLG sequencing MANDATORY before initiation (fatal Alpers-Huttenlocher in POLG carriers)",
            "LFTs + ammonia: baseline, 4w, 12w, 6-monthly (hepatotoxicity, hyperammonaemia)",
            "FBC + platelets: baseline, 3-monthly (thrombocytopaenia)",
            "VPA trough level (target 75-100 mg/L for Dravet); ammonia >80 µmol/L → L-carnitine 50 mg/kg/day",
            "Weight (VPA weight gain), pancreatitis signs (serum lipase/amylase if symptomatic)",
            "VPPP mandatory for females of childbearing age (NTD teratogenicity 2-10%; MHRA 2021)",
        ],
        "scn1a_note": (
            "VPA backbone of all Dravet treatment. Does NOT block NaV1.1 in PV interneurons "
            "(mechanism distinct from CBZ/PHT — VPA's Na+ channel effect is on excitatory "
            "neurons at concentrations used clinically). POLG MANDATORY — Dravet infants on "
            "the cusp of initiating VPA; POLG carrier frequency ~1:200. VPPP for females "
            "from puberty: teratogenicity cannot be minimised in DRE requiring VPA."
        ),
    },
    {
        "drug": "Stiripentol (Diacomit)",
        "evidence": "Level A (Dravet-specific; always with VPA + CLB triple)",
        "indication": "Dravet syndrome — triple therapy VPA + CLB + stiripentol (Level A FDA 2018, EMA 2007); NOT effective as monotherapy",
        "dose_moa": (
            "50 mg/kg/day PO divided BD-TDS WITH food (suspension or capsules). "
            "Always combined with VPA and CLB — must be given as triple. "
            "MOA: (1) GABA-A positive allosteric modulator (distinct site from BDZ); "
            "(2) CYP2C19 inhibition → ↑CLB active metabolite N-desmethylclobazam level "
            "(pharmacokinetic synergy); (3) STAT3 transcription factor inhibition "
            "(anti-inflammatory); (4) Inhibits lactate dehydrogenase (metabolic). "
            "The STICLO trials (NEJM 2000) showed 71% ≥50% seizure reduction vs placebo "
            "when added to VPA+CLB in Dravet."
        ),
        "efficacy": "71% ≥50% seizure reduction in STICLO trial (Dravet); 67% convulsive seizure reduction (NEJM 2000); GTCS + tonic-clonic most responsive",
        "monitoring": [
            "VPA + CLB levels (stiripentol inhibits CYP2C19 → CLB and N-CLB levels rise — reduce CLB dose by 30-50% on initiation)",
            "LFTs (combined VPA + STP: monitor hepatotoxicity synergy)",
            "Appetite, weight (anorexia, nausea — take with food, may improve over weeks)",
            "Neutrophil count (transient neutropaenia reported at initiation)",
            "Neurological side effects: ataxia, sedation (especially in first weeks — titrate slowly)",
        ],
        "scn1a_note": (
            "Stiripentol is Dravet-SPECIFIC (EMA Dravet-only indication). Must be prescribed as "
            "triple VPA+CLB+STP. Reduce CLB dose at STP initiation (CYP2C19 inhibition ↑CLB → "
            "sedation). Do NOT use stiripentol as monotherapy (ineffective). Combination available "
            "through Biocodex patient support programme. Mechanism: GABAergic synergy with CLB "
            "through both pharmacodynamic (GABA-A allosteric) and pharmacokinetic (CLB level "
            "boosting) effects."
        ),
    },
    {
        "drug": "Fenfluramine (Fintepla)",
        "evidence": "Level A (FDA 2020; EMA 2020 for Dravet ≥2 years; REMS required)",
        "indication": "Dravet syndrome ≥2 years — adjunct therapy; FDA-approved 2020; prescription through REMS (cardiac monitoring required)",
        "dose_moa": (
            "0.1 mg/kg/day divided BD initially; titrate to 0.2 mg/kg/day by Week 2; "
            "max 0.7 mg/kg/day (max 26 mg/day; max 17 mg/day if on stiripentol). "
            "Available as 2.2 mg/mL oral solution (Fintepla). "
            "MOA: (1) Serotonin releaser and reuptake inhibitor → 5-HT1D / 5-HT2C "
            "receptor activation → inhibitory modulation; (2) Sigma-1 receptor "
            "modulation (neuroprotective); (3) HCN1 channel activation (increases "
            "hyperpolarisation-activated current — stabilises neuronal excitability); "
            "(4) Indirect SERT-independent mechanism. Pooled analysis: 54% reduction "
            "in convulsive seizure days (NEJM 2022)."
        ),
        "efficacy": "62% ≥50% reduction in convulsive seizure frequency (Dravet phase III trials); 54% vs 6% placebo (NEJM 2022); seizure-free: 26% vs 3% placebo",
        "monitoring": [
            "Echocardiogram: MANDATORY before initiation, then every 6 months (cardiac valvulopathy risk — FDA REMS requirement)",
            "Blood pressure and heart rate: each visit (serotonergic cardiovascular effects)",
            "Growth / weight: monthly (anorexia, weight loss)",
            "Serotonin syndrome risk: avoid concurrent serotonergic drugs (SSRIs, SNRIs, MAOIs, triptans)",
            "LFT (mild enzyme elevation reported)",
            "REMS enrolment: prescriber + pharmacy + patient/caregiver (Fintepla REMS programme mandatory)",
        ],
        "scn1a_note": (
            "Fenfluramine: highly effective Dravet-specific therapy. FDA REMS mandatory — cardiac "
            "valvulopathy risk from serotonin pathway (known from historical high-dose 'fen-phen' use; "
            "low-dose Fintepla trials show minimal signal but mandatory monitoring). "
            "Dose reduction required if combining with stiripentol (max 17 mg/day; stiripentol "
            "potentiates serotonergic effects). Do NOT combine with MAOIs or serotonergic drugs "
            "(serotonin syndrome). HCN1 activation mechanism supports use in HCN1-Dravet "
            "phenocopy."
        ),
    },
    {
        "drug": "Cannabidiol (Epidiolex)",
        "evidence": "Level A (FDA 2018 for Dravet ≥2 years and LGS ≥2 years)",
        "indication": "Dravet syndrome and LGS ≥2 years; plant-derived purified CBD (not THC); oral solution",
        "dose_moa": (
            "5 mg/kg/day divided BD initially; titrate to 10 mg/kg/day by Week 2; "
            "max 20 mg/kg/day. Take with fatty food (4-fold ↑ absorption). "
            "MOA: not fully elucidated; proposed mechanisms: "
            "(1) TRPV1 (transient receptor potential vanilloid) modulation; "
            "(2) GPR55 receptor antagonism (endocannabinoid orphan receptor); "
            "(3) GlyR (glycine receptor) potentiation; (4) inhibition of adenosine "
            "reuptake; (5) T-type Ca²⁺ channel inhibition. "
            "Dravet trial (NEJM 2017): 38.9% reduction in convulsive seizure frequency vs "
            "13.3% placebo; seizure freedom in 5% vs 0%."
        ),
        "efficacy": "38-39% reduction in convulsive seizure frequency (Dravet); 5% seizure-free (NEJM 2017); additive with stiripentol (GWPCARE5 trial)",
        "monitoring": [
            "LFTs MANDATORY: baseline, then 2w, 4w, 12w, then 3-monthly (dose-dependent transaminase elevation; especially with VPA)",
            "Reduce VPA dose if combined (CBD inhibits VPA glucuronidation → VPA levels rise 20-30%)",
            "Sedation: especially when combined with CLB (CBD inhibits CLB CYP metabolism → N-CLB accumulation)",
            "Reduce CLB dose 25-50% at CBD initiation",
            "Appetite, diarrhoea, weight loss",
            "AED interactions: CBD is a CYP3A4 and CYP2C19 inhibitor",
        ],
        "scn1a_note": (
            "Cannabidiol (Epidiolex) is a plant-derived purified CBD — not equivalent to "
            "cannabis/marijuana/THC and does not cause intoxication. FDA-approved for Dravet "
            "syndrome 2018. LFT monitoring mandatory — VPA + CBD → VPA level rise (reduce VPA "
            "dose). CBD + CLB → N-CLB accumulation (reduce CLB dose on initiation). Take with "
            "fatty meal (absorption). GWPCARE5: CBD + stiripentol + standard therapy shows "
            "additional benefit over stiripentol alone."
        ),
    },
    {
        "drug": "CLB (Clobazam)",
        "evidence": "Level B (adjunct — Dravet and GEFS+)",
        "indication": "Adjunct for Dravet (in triple with VPA+STP) and GEFS+ GTCS/absence",
        "dose_moa": (
            "0.1-0.3 mg/kg/day PO divided BD; max 20 mg/day in paediatrics; 40 mg/day adults. "
            "Evening loading dose for nocturnal seizure predominance. "
            "MOA: 1,5-benzodiazepine GABA-A positive allosteric modulator; "
            "less sedating than 1,4-BDZ (diazepam); active metabolite N-desmethylclobazam "
            "(prolonged t½ 30-60h); CLB is the 'C' in the VPA+CLB+STP triple. "
            "Tolerance develops in 20-30% after 3-12 months."
        ),
        "efficacy": "40-50% ≥50% reduction in Dravet seizures (as adjunct); essential component of VPA+CLB+STP triple (Level A evidence for the triple)",
        "monitoring": [
            "Sedation and cognitive effects (especially in first weeks; improves with time)",
            "CLB level monitoring if adding stiripentol or CBD (both raise N-CLB level)",
            "Tolerance: reassess every 6 months; pulse dosing for catamenial pattern",
            "Withdrawal: do NOT stop abruptly (benzodiazepine withdrawal seizures); taper ≥4 weeks",
        ],
        "scn1a_note": (
            "CLB is the essential 'C' in the VPA+CLB+STP triple — DO NOT omit when adding stiripentol. "
            "CYP2C19 phenotyping: poor metabolisers have higher N-CLB levels (more effective but more "
            "sedating). When adding stiripentol: reduce CLB by 30-50% (CYP2C19 inhibition). "
            "When adding CBD: reduce CLB by 25% (similar mechanism). CLB pulse dosing "
            "for catamenial Dravet (10 days/month perimenstrual — reduce catamenial seizure clusters)."
        ),
    },
    {
        "drug": "Topiramate (TPM)",
        "evidence": "Level B (adjunct — Dravet and GEFS+)",
        "indication": "Adjunct for drug-resistant Dravet — GTCS and focal seizure reduction; no Dravet-specific FDA indication",
        "dose_moa": (
            "1-3 mg/kg/day PO divided BD; titrate slowly (max 400 mg/day). "
            "MOA: (1) NaV channel blockade (use-dependent, preferentially high-frequency "
            "firing — IMPORTANT: different selectivity from CBZ/PHT; less NaV1.1-PV "
            "interneuron toxicity at clinical doses); (2) GABA-A potentiation; "
            "(3) AMPA/kainate glutamate receptor antagonism; (4) Carbonic anhydrase "
            "inhibition (mild). Broad-spectrum profile. "
            "Evidence: Dravet open-label series show 50% reduction GTCS in ~40% patients."
        ),
        "efficacy": "~40% ≥50% seizure reduction as adjunct in Dravet (open-label); better for GTCS than myoclonic; Level B evidence",
        "monitoring": [
            "Kidney stones (metabolic acidosis + carbonic anhydrase inhibition — hydration mandatory)",
            "Cognitive effects / word-finding difficulty ('dopamax') — titrate slowly",
            "Weight loss (useful co-benefit in VPA-induced weight gain)",
            "Hyperthermia risk (carbonic anhydrase inhibition → reduced sweating; avoid exercise heat stress in Dravet)",
            "VPA + TPM: hyperammonaemia risk (monitor ammonia)",
            "Eye: acute closed-angle glaucoma (rare — alert parents to eye pain/visual change)",
        ],
        "scn1a_note": (
            "Topiramate is a useful adjunct in Dravet when VPA+stiripentol+CLB+fenfluramine combination "
            "still insufficient. NaV channel blockade at clinical TPM doses appears to have less "
            "NaV1.1 PV-interneuron selectivity than CBZ/PHT — but caution in myoclonic-predominant "
            "Dravet (monitor). IMPORTANT: carbonic anhydrase inhibition → hyperthermia risk "
            "(reduced sweating) — significant in Dravet where fever is the primary trigger: "
            "educate families; avoid co-prescription with acetazolamide."
        ),
    },
    {
        "drug": "KD (Ketogenic Diet)",
        "evidence": "Level B (drug-resistant Dravet / GEFS+)",
        "indication": "Drug-resistant Dravet after failure of ≥3 AEDs; earlier consideration than other epilepsies given DRE severity",
        "dose_moa": (
            "Classic KD: 3:1 or 4:1 (fat:protein+carbohydrate) ratio; "
            "or MCT (medium-chain triglyceride) diet. "
            "Initiation: fasting protocol or gradual (both validated). "
            "MOA: ketone bodies (β-hydroxybutyrate) → KATP channel opening in "
            "neuronal membranes → hyperpolarisation; also: alters GABA/glutamate "
            "balance, reduces reactive oxygen species, enhances mitochondrial "
            "biogenesis, inhibits HDAC. Multiple anti-epileptic mechanisms "
            "independent of NaV channel effects — hence effective in NaV1.1-dependent "
            "Dravet."
        ),
        "efficacy": "50% ≥50% reduction in Dravet seizures (case series meta-analysis); 14% seizure-free (Dravet-specific KD review, Epilepsia 2016)",
        "monitoring": [
            "Lipid profile (LDL, HDL, triglycerides — baseline, 3M, 6M, 12M)",
            "Kidney stones (urate, oxalate — urine screening quarterly; urinalysis)",
            "Growth (calories restricted — anthropometry monthly in children)",
            "Ketone levels (urine or blood β-HB; target 2-5 mmol/L for seizure control)",
            "Selenium, zinc, selenium supplementation (micronutrient deficiency on KD)",
            "Electrolytes, bicarbonate (metabolic acidosis, hyponatraemia on KD)",
            "Bone density (DEXA every 2 years — acidosis-related bone demineralisation)",
        ],
        "scn1a_note": (
            "KD earlier initiation in Dravet than most epilepsies — given severity and DRE nature. "
            "Avoid concurrent POLG (if VPA discontinued for KD — POLG becomes less relevant). "
            "Fever management: KD reduces infection-provoked seizures in some Dravet "
            "(possibly via anti-inflammatory mechanism). KD + fenfluramine: both reduce appetite — "
            "caloric intake monitoring critical. KD + stiripentol: no major pharmacokinetic "
            "interactions. KD + topiramate: add carbonic anhydrase inhibitor → acidosis risk; "
            "monitor bicarbonate."
        ),
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "CBZ / OXC / PHT (Carbamazepine / Oxcarbazepine / Phenytoin)",
        "risk": "ABSOLUTE CI in Dravet syndrome",
        "reason": (
            "Sodium channel blockers preferentially block NaV1.1 in PV+ fast-spiking "
            "inhibitory interneurons → further reduce the already-haploinsufficient "
            "NaV1.1 function → PV interneuron failure → paradoxical seizure worsening. "
            "Clinical evidence: ~70% of Dravet patients worsen on CBZ/PHT; prolonged "
            "GTCS, increased seizure frequency, and status epilepticus documented after "
            "CBZ initiation. MECHANISM: these drugs bind preferentially to inactivated "
            "NaV channels — in high-frequency PV interneurons the channels are "
            "predominantly inactivated → preferential block of inhibitory neurons. "
            "NEVER use CBZ/OXC/PHT as chronic therapy in any Dravet patient."
        ),
        "alternative": "VPA backbone; stiripentol triple; fenfluramine; cannabidiol; CLB; topiramate; KD",
    },
    {
        "drug": "LTG (Lamotrigine) — high dose / Dravet",
        "risk": "ABSOLUTE CI in Dravet syndrome; HIGH RISK in GEFS+ with myoclonic component",
        "reason": (
            "LTG blocks NaV1.1 in a use-dependent manner. In Dravet: same PV-interneuron "
            "failure mechanism as CBZ/PHT — LTG dramatically worsens myoclonic seizures in "
            "Dravet, often causing myoclonic storm. Well-documented in clinical series: LTG "
            "initiation in Dravet → acute severe myoclonic worsening within days. "
            "The NaV1.1-PV interneuron toxicity of LTG is equipotent to CBZ at clinical doses. "
            "CAUTION: LTG may be tolerated at very low doses in GEFS+ patients with pure GTCS "
            "and NO myoclonic component — but extreme caution and monitoring required. "
            "NEVER initiate LTG in confirmed Dravet."
        ),
        "alternative": "VPA + stiripentol + CLB triple; fenfluramine; cannabidiol (no NaV1.1 mechanism)",
    },
    {
        "drug": "Tiagabine",
        "risk": "ABSOLUTE CI — NCSE risk in all Dravet forms",
        "reason": (
            "GAT-1 GABA reuptake inhibitor → extrasynaptic GABA accumulation → tonic "
            "GABA-A receptor activation (extrasynaptic α5-subunit containing GABA-A) → "
            "neuronal clamping → NCSE (non-convulsive status epilepticus). "
            "In Dravet: compromised inhibitory architecture makes extrasynaptic GABA "
            "toxicity catastrophic. Documented NCSE induction by tiagabine across all "
            "epileptic encephalopathies. NEVER use in Dravet."
        ),
        "alternative": "CLB (1,5-BDZ) or LEV for adjunct GABAergic/vesicular modulation",
    },
    {
        "drug": "VPA without POLG sequencing",
        "risk": "HIGH RISK — Alpers-Huttenlocher fatal hepatic failure in POLG carriers",
        "reason": (
            "POLG (polymerase gamma) mutations → VPA-induced fatal progressive hepatic failure "
            "(Alpers-Huttenlocher syndrome). POLG carrier frequency ~1:200 in general population; "
            "higher in epileptic encephalopathy cohorts. VPA + POLG = irreversible hepatic "
            "necrosis, often fatal. Dravet infants are commonly started on VPA before genetic "
            "panel complete — POLG should be run BEFORE VPA, or interim VPA accepted ONLY if "
            "non-VPA options exhausted and transition plan in place when POLG result returns."
        ),
        "alternative": "LEV + CLB bridge while awaiting POLG result in urgent situations",
    },
    {
        "drug": "Fenfluramine without cardiac monitoring / REMS enrolment",
        "risk": "HIGH RISK — cardiac valvulopathy, serotonin syndrome",
        "reason": (
            "Fenfluramine (Fintepla) REMS (Risk Evaluation and Mitigation Strategy) mandatory: "
            "echocardiography before initiation and every 6 months (cardiac valvulopathy "
            "signal from historical high-dose 'fen-phen' — low-dose Fintepla trials show "
            "minimal signal, but monitoring is mandatory per FDA). Serotonin syndrome risk "
            "with concurrent SSRIs, SNRIs, triptans, MAOIs — ABSOLUTE CI combination. "
            "Do NOT prescribe outside REMS programme."
        ),
        "alternative": "Cannabidiol or stiripentol as Level A alternatives without cardiac monitoring requirement",
    },
]

# ── Monitoring ────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG sequencing MANDATORY before VPA", "frequency": "Once (before VPA initiation)", "rationale": "Fatal Alpers-Huttenlocher hepatic failure in POLG carriers"},
    {"item": "Echocardiogram (Fenfluramine/Fintepla REMS)", "frequency": "Before initiation, then every 6 months", "rationale": "Cardiac valvulopathy risk — FDA REMS mandatory requirement for Fintepla"},
    {"item": "LFTs + VPA level + ammonia", "frequency": "Baseline, 4w, 12w, then 6-monthly", "rationale": "VPA hepatotoxicity, hyperammonaemia; CBD co-administration raises VPA level 20-30%"},
    {"item": "LFTs for Cannabidiol (Epidiolex)", "frequency": "Baseline, 2w, 4w, 12w, then 3-monthly", "rationale": "Dose-dependent transaminase elevation especially with concurrent VPA"},
    {"item": "CLB + N-CLB levels", "frequency": "At initiation and when adding stiripentol or CBD", "rationale": "Stiripentol + CBD both inhibit CLB metabolism → N-CLB accumulation → sedation"},
    {"item": "Prolonged EEG (overnight) if CSWS suspected", "frequency": "Every 12 months in Dravet; immediately if cognitive regression", "rationale": "CSWS in 20-25% SMEB/Dravet; NREM electrical status → cognitive regression"},
    {"item": "SUDEP counselling + nocturnal monitoring", "frequency": "Annual from Dravet diagnosis", "rationale": "SUDEP risk elevated (1:1000/year Dravet); nocturnal monitoring device recommended (Emfit, SAMi-3)"},
    {"item": "VPPP (Valproate Pregnancy Prevention Programme)", "frequency": "Annual review from puberty for all females on VPA", "rationale": "VPA teratogenicity NTD 2-10%; MHRA 2021 VPPP mandatory; contraception counselling"},
    {"item": "Neurodevelopmental assessment", "frequency": "Every 12 months from diagnosis", "rationale": "Dravet: developmental plateau 12-24 months then regression; cognitive + motor + behavioural trajectory"},
    {"item": "Ketone monitoring (if on KD)", "frequency": "Urine daily or blood β-HB weekly; clinic 3-monthly", "rationale": "KD: target blood β-HB 2-5 mmol/L for seizure control; avoid excessive ketosis"},
],

# ── Lifecycle ─────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {"window": "Prenatal", "ages": "Conception – birth", "events": "No prenatal phenotype in Dravet; if prior child with Dravet + confirmed de novo SCN1A: 15-20% recurrence risk (parental mosaicism); prenatal diagnosis available"},
    {"window": "Infantile Onset (Dravet)", "ages": "5-12 months", "events": "First prolonged febrile hemiclonic seizure (Dravet hallmark); SCN1A panel urgently; fever action plan; buccal midazolam prescription; avoid CBZ/PHT; VPA + POLG initiation"},
    {"window": "Early Childhood", "ages": "1-5 years", "events": "Polyphasic seizures (myoclonic, absence, focal, GTCS); VPA+CLB+stiripentol triple; fenfluramine/cannabidiol add; developmental plateau; KD if ≥2 AED failure; SUDEP monitoring"},
    {"window": "School Age", "ages": "5-12 years", "events": "DRE management; cognitive support (learning disability, attention); MedicAlert; school seizure protocol; photosensitivity management; CSWS monitoring (SMEB); seizure diary; quality of life"},
    {"window": "Adolescent", "ages": "12-18 years", "events": "Gradual reduction of febrile sensitivity in some; VPPP for females on VPA; transition to adult services; SUDEP risk counselling; driving (not permitted until 5 years seizure-free; Dravet rarely qualifies); vocational planning"},
    {"window": "Adult", "ages": "18 years+", "events": "Ongoing DRE in most Dravet; GEFS+ often remits; multidisciplinary adult epilepsy care; independent living assessment; employment support; continued SUDEP counselling; fenfluramine/cannabidiol continuation"},
]

# ── Concepts ──────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "SCN1A (2q24.3)", "definition": "Gene encoding NaV1.1 voltage-gated Na+ channel alpha subunit; expressed in PV+ fast-spiking inhibitory interneurons; LOF → Dravet/GEFS+; most important epilepsy gene; >1,800 variants in LOVD"},
    {"term": "NaV1.1 / PV interneuron haploinsufficiency", "definition": "NaV1.1 preferentially expressed in parvalbumin fast-spiking inhibitory interneurons; haploinsufficiency → interneuron failure at high firing rates → cortical disinhibition → seizures"},
    {"term": "Dravet syndrome (SMEI)", "definition": "Severe Myoclonic Epilepsy of Infancy; de novo SCN1A LOF; onset 5-8 months febrile hemiclonic; drug-resistant; intellectual disability; SUDEP risk; 1:15,000–1:22,000"},
    {"term": "GEFS+ (Generalised Epilepsy with FS+)", "definition": "Autosomal dominant SCN1A partial LOF; febrile seizures ± FS+; 70-80% penetrance; generally favourable prognosis; spectrum from mild FS to GEFS+ overlap with Dravet"},
    {"term": "SMEB (Borderline SMEI)", "definition": "Dravet-like clinical phenotype without classic myoclonic component; de novo SCN1A; intermediate prognosis; CSWS in 25%; drug-resistant"},
    {"term": "Stiripentol (Diacomit)", "definition": "GABA-A allosteric modulator + CYP2C19 inhibitor; Dravet-specific Level A; VPA+CLB+STP triple therapy; STICLO trial 71% responders; not effective as monotherapy"},
    {"term": "Fenfluramine (Fintepla)", "definition": "Serotonin releaser + HCN1 activator; FDA 2020 Dravet; Level A; REMS echocardiogram q6M mandatory; 62% GTCS reduction; max 0.7 mg/kg/day"},
    {"term": "Cannabidiol (Epidiolex)", "definition": "Purified plant-derived CBD (not THC); FDA 2018 Dravet; Level A; 39% convulsive seizure reduction; LFT monitoring mandatory; interacts with VPA and CLB"},
    {"term": "NaV1.1 PV-interneuron / sodium channel blocker paradox", "definition": "CBZ/OXC/PHT/LTG block NaV1.1 preferentially in high-frequency firing PV interneurons → further inhibitory failure → paradoxical seizure worsening in Dravet — explains ABSOLUTE CI"},
    {"term": "SUDEP risk in Dravet", "definition": "Sudden Unexpected Death in Epilepsy; elevated 1:1000/year in Dravet (vs 1:4500 general epilepsy); cardiac NaV1.1 involvement; nocturnal monitoring recommended"},
    {"term": "SCN1A mosaic Dravet", "definition": "~15% of Dravet is SCN1A somatic mosaic (5-20% allele fraction); missed on standard exome (50× coverage); requires deep-read sequencing ≥500× or targeted assay"},
    {"term": "POLG-Alpers-Huttenlocher", "definition": "POLG (mitochondrial DNA polymerase gamma) mutation + VPA = fatal progressive hepatic failure (Alpers syndrome); POLG MANDATORY before VPA in ALL SCN1A patients"},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021 mandatory: females on VPA annual review + effective contraception + specialist counselling; NTD risk 2-10%; spina bifida most common"},
    {"term": "CSWS / ESES", "definition": "Continuous Spike-Wave in Slow Sleep / Electrical Status Epilepticus of Sleep; in SMEB ~25%; causes developmental regression; overnight EEG q12M; CLB pulse may treat"},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    "POLG sequencing MANDATORY before VPA in any SCN1A patient (Dravet + GEFS+ + SMEB)",
    "CBZ / OXC / PHT / LTG: ABSOLUTE CI in Dravet syndrome — NaV1.1 PV-interneuron block → paradoxical seizure worsening",
    "Fever action threshold: ≥37.5°C in Dravet → antipyretic + BDZ standby (lower than standard ≥38°C threshold)",
    "Fenfluramine REMS: echocardiogram MANDATORY before initiation and every 6 months (cardiac valvulopathy monitoring)",
    "Fenfluramine dose max: 0.7 mg/kg/day (max 26 mg/day); if combined with stiripentol: max 0.4 mg/kg/day (max 17 mg/day)",
    "Cannabidiol: LFT monitoring baseline, 2w, 4w, 12w, 3-monthly; reduce VPA dose when adding CBD (VPA level ↑20-30%)",
    "Stiripentol: ALWAYS combined with VPA+CLB triple — reduce CLB dose 30-50% at stiripentol initiation (CYP2C19 inhibition)",
    "VPA target trough 75-100 mg/L (Dravet); ammonia >80 µmol/L → L-carnitine 50 mg/kg/day",
    "SCN1A mosaic screen: if clinical Dravet with negative standard exome → deep-read sequencing ≥500× (15% are mosaic)",
    "SUDEP: annual counselling from diagnosis; nocturnal monitoring device (Emfit/SAMi-3) recommended",
    "VPPP: annual review for ALL females on VPA from puberty (MHRA 2021 mandatory)",
    "CLB dose reduction: reduce by 30-50% when adding stiripentol; by 25% when adding cannabidiol",
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE 2022 Classification of the Epilepsies",
    "NICE NG217 (Epilepsies in children, young people and adults, 2022)",
    "Wirrell EC et al. — ILAE Task Force on Dravet syndrome management (2022 Epilepsia)",
    "STICLO Trials (Chiron C et al., NEJM 2000) — stiripentol Level A for Dravet",
    "Aquino D et al. (NEJM 2022) — Fenfluramine Level A Dravet Phase III",
    "Devinsky O et al. (NEJM 2017) — Cannabidiol Level A Dravet Phase III",
    "MHRA VPPP 2021 (Valproate Pregnancy Prevention Programme — mandatory)",
    "ACMG-AMP 2015 (Variant classification standards)",
    "FDA Fintepla REMS (2020) — cardiac monitoring for fenfluramine",
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Claes L et al. (2001) — De novo mutations in the sodium-channel gene SCN1A cause severe myoclonic epilepsy of infancy. Am J Hum Genet.",
    "Chiron C et al. (2000) — Stiripentol in severe myoclonic epilepsy in infancy (STICLO). Lancet (NEJM 2000 reprint).",
    "Devinsky O et al. (2017) — Trial of cannabidiol for drug-resistant seizures in the Dravet syndrome. NEJM.",
    "Lagae L et al. (2019) — Fenfluramine hydrochloride for the treatment of seizures in Dravet syndrome. Lancet.",
    "Wirrell EC et al. (2022) — Optimizing the treatment of Dravet syndrome — ILAE Task Force recommendations. Epilepsia.",
    "Bhatt DL et al. (2023) — SCN1A-related epilepsy spectrum: systematic review. Epilepsia.",
]


# ── Patient Generator ─────────────────────────────────────────────────────────
def _generate_patients(n: int = 40):
    random.seed(SEED)
    patients = []
    # Distribute etiologies per catalog
    etiol_pool = []
    for e in ETIOLOGY_CATALOG:
        etiol_pool.extend([e["etiology"]] * e["n"])
    random.shuffle(etiol_pool)

    for i in range(n):
        etiology = etiol_pool[i]
        is_dravet = "Dravet" in etiology and "phenocopy" not in etiology.lower()
        is_smeb = "SMEB" in etiology or "Borderline" in etiology
        is_gefs = "GEFS+" in etiology and "Dravet" not in etiology
        is_mild = "mild" in etiology.lower() and "pure FS+" in etiology
        is_phenocopy = "phenocopy" in etiology.lower()

        # Onset months: Dravet: 5-8 mo typical; GEFS+: 6-24 mo; mild: 12-36 mo
        if is_dravet:
            onset = random.randint(3, 12)
        elif is_smeb:
            onset = random.randint(4, 18)
        elif is_gefs:
            onset = random.randint(6, 24)
        elif is_mild:
            onset = random.randint(12, 36)
        else:
            onset = random.randint(4, 18)

        seizure_free = random.random() < (
            0.65 if is_mild else
            0.35 if is_gefs else
            0.10 if is_smeb else
            0.04 if is_dravet else
            0.15
        )
        drug_resistant = (
            not seizure_free and
            random.random() < (
                0.90 if is_dravet else
                0.50 if is_smeb else
                0.20 if is_gefs else
                0.05 if is_mild else
                0.60
            )
        )
        on_vpa = random.random() < (0.85 if is_dravet else 0.70 if is_smeb else 0.60 if is_gefs else 0.40)
        polg_tested = "Y" if random.random() < 0.80 else "N"
        vpa_without_polg = on_vpa and polg_tested == "N"
        on_stiripentol = is_dravet and random.random() < 0.68
        on_fenfluramine = (is_dravet or is_smeb) and random.random() < 0.38
        on_cbd = (is_dravet or is_smeb) and random.random() < 0.32
        on_clo = random.random() < (0.75 if is_dravet else 0.45 if is_smeb else 0.30)
        on_kd = drug_resistant and random.random() < 0.35
        has_sudep_device = (is_dravet or is_smeb) and random.random() < 0.48
        csws = (is_smeb or is_dravet) and random.random() < (0.25 if is_smeb else 0.12)
        navblocker_exposure = random.random() < (0.18 if is_dravet else 0.08)  # accidental exposure

        patients.append({
            "id": f"SCN1A-{i+1:03d}",
            "etiology": etiology,
            "onset_months": onset,
            "sex": random.choice(["M", "F"]),
            "seizure_free": seizure_free,
            "drug_resistant": drug_resistant,
            "on_vpa": on_vpa,
            "polg_tested": polg_tested,
            "vpa_without_polg": vpa_without_polg,
            "on_stiripentol": on_stiripentol,
            "on_fenfluramine": on_fenfluramine,
            "on_cbd": on_cbd,
            "on_clo": on_clo,
            "on_kd": on_kd,
            "has_sudep_device": has_sudep_device,
            "csws": csws,
            "navblocker_exposure": navblocker_exposure,
        })
    return patients


def get_overview():
    """Return SCN1A overview dict."""
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    on_vpa = sum(1 for p in pts if p["on_vpa"])
    polg_done = sum(1 for p in pts if p["polg_tested"] == "Y")
    vpa_without_polg = sum(1 for p in pts if p["vpa_without_polg"])
    on_stiripentol = sum(1 for p in pts if p["on_stiripentol"])
    on_fenfluramine = sum(1 for p in pts if p["on_fenfluramine"])
    on_cbd = sum(1 for p in pts if p["on_cbd"])
    on_kd = sum(1 for p in pts if p["on_kd"])
    sudep_device = sum(1 for p in pts if p["has_sudep_device"])
    csws_n = sum(1 for p in pts if p["csws"])
    navblocker_exp = sum(1 for p in pts if p["navblocker_exposure"])

    return {
        "gene": "SCN1A",
        "locus": "2q24.3",
        "inheritance": "De novo (Dravet ~85% de novo; ~15% mosaic) or autosomal dominant (GEFS+ familial; ~70-80% penetrance)",
        "protein": "NaV1.1 — Voltage-Gated Sodium Channel Alpha Subunit 1; PV+ fast-spiking inhibitory interneuron; haploinsufficiency → Dravet / GEFS+",
        "mechanism": (
            "SCN1A LOF → NaV1.1 haploinsufficiency in parvalbumin-positive (PV+) "
            "fast-spiking inhibitory interneurons → PV interneuron cannot sustain "
            "high-frequency inhibitory firing → cortical disinhibition → seizures. "
            "PARADOX: sodium channel blockers (CBZ/PHT/OXC/LTG) further impair "
            "NaV1.1 in PV interneurons → worsen Dravet seizures → ABSOLUTE CI."
        ),
        "key_aha": (
            "SCN1A (2q24.3) — NaV1.1 PV-interneuron haploinsufficiency. "
            "CBZ / OXC / PHT / LTG: ABSOLUTE CI in Dravet (NaV1.1 block → PV interneuron failure). "
            "Tiagabine ABSOLUTE CI. POLG MANDATORY before VPA. "
            "Stiripentol (Level A triple VPA+CLB+STP), Fenfluramine (Level A REMS echocardiogram), "
            "Cannabidiol (Level A LFT monitoring). SUDEP risk elevated — nocturnal monitoring. "
            "VPPP mandatory for females on VPA."
        ),
        "n_patients": n,
        "seizure_free_pct": round(100 * seizure_free / n),
        "drug_resistant_pct": round(100 * drug_resistant / n),
        "on_vpa_pct": round(100 * on_vpa / n),
        "polg_done_pct": round(100 * polg_done / n),
        "vpa_without_polg": vpa_without_polg,
        "on_stiripentol_pct": round(100 * on_stiripentol / n),
        "on_fenfluramine_pct": round(100 * on_fenfluramine / n),
        "on_cbd_pct": round(100 * on_cbd / n),
        "on_kd_pct": round(100 * on_kd / n),
        "sudep_device_pct": round(100 * sudep_device / n),
        "csws_n": csws_n,
        "navblocker_exposure_n": navblocker_exp,
        "cbl_alert": "CBZ / OXC / PHT: ABSOLUTE CI in Dravet — NaV1.1 PV-interneuron block → paradoxical seizure worsening in ~70% of Dravet patients",
        "ltg_alert": "LTG (Lamotrigine): ABSOLUTE CI in Dravet — NaV1.1 block → myoclonic storm; HIGH RISK in GEFS+ with myoclonic component",
        "tiagabine_alert": "Tiagabine: ABSOLUTE CI — GAT-1 block → extrasynaptic GABA → NCSE",
        "polg_alert": "POLG MANDATORY before VPA — fatal Alpers-Huttenlocher hepatic failure in POLG carriers",
        "fenfluramine_alert": "Fenfluramine (Fintepla): REMS mandatory — echocardiogram before initiation and every 6 months (cardiac valvulopathy)",
        "sudep_alert": "SUDEP risk elevated in Dravet (~1:1000/year). Nocturnal monitoring device recommended. Annual counselling mandatory.",
        "contraindications_summary": [
            "CBZ / OXC / PHT — ABSOLUTE CI in Dravet: NaV1.1 PV-interneuron block → paradoxical worsening (~70% patients worsen)",
            "LTG — ABSOLUTE CI in Dravet: myoclonic storm risk; HIGH RISK in GEFS+ with myoclonic component",
            "Tiagabine — ABSOLUTE CI: NCSE in any Dravet/encephalopathy (GAT-1 → extrasynaptic GABA)",
            "VPA without POLG — HIGH RISK: Alpers-Huttenlocher fatal hepatic failure in POLG carriers",
            "Fenfluramine without REMS/echocardiogram — HIGH RISK: cardiac valvulopathy + serotonin syndrome",
        ],
        "thresholds": THRESHOLDS,
        "references": [r.split(" — ")[0] for r in REFERENCES],
    }


def get_breakdown():
    """Return SCN1A breakdown dict."""
    pts = _generate_patients()
    n = len(pts)

    # Sample patients (first 15)
    sample = []
    for p in pts[:15]:
        sample.append({
            "id": p["id"],
            "etiology_short": p["etiology"].split(" — ")[0] if " — " in p["etiology"] else p["etiology"].split(" (")[0],
            "onset_months": p["onset_months"],
            "sex": p["sex"],
            "seizure_free": p["seizure_free"],
            "drug_resistant": p["drug_resistant"],
            "on_vpa": p["on_vpa"],
            "polg": p["polg_tested"],
            "vpa_no_polg": p["vpa_without_polg"],
            "on_stp": p["on_stiripentol"],
            "on_fen": p["on_fenfluramine"],
            "on_cbd": p["on_cbd"],
            "on_kd": p["on_kd"],
            "sudep_device": p["has_sudep_device"],
            "csws": p["csws"],
        })

    return {
        "n_patients": n,
        "etiology_distribution": [
            {"etiology": e["etiology"], "n": e["n"], "pct": e["pct"],
             "category": e["category"],
             "mechanism_summary": e["mechanism"][:300] + "…",
             "eeg": e["eeg_correlate"][:200] + "…",
             "mri": e["mri_finding"][:200] + "…"}
            for e in ETIOLOGY_CATALOG
        ],
        "seizure_types": [
            {"type": st["type"], "pct": st["pct_patients"],
             "eeg": st["eeg_correlate"], "tip": st["clinical_tip"]}
            for st in SEIZURE_TYPES
        ],
        "triggers": [
            {"trigger": t["trigger"], "pct": t["pct"],
             "threshold": t["threshold"], "mgmt": t["management"]}
            for t in TRIGGERS
        ],
        "treatments": [
            {"drug": t["drug"], "evidence": t["evidence"],
             "indication": t["indication"],
             "efficacy": t["efficacy"],
             "scn1a_note": t["scn1a_note"]}
            for t in TREATMENTS
        ],
        "contraindications": CONTRAINDICATIONS,
        "monitoring": [m for m in MONITORING[0]],
        "lifecycle": LIFECYCLE,
        "sample_patients": sample,
    }


def get_definitions():
    """Return SCN1A definitions dict."""
    return {
        "gene": "SCN1A",
        "concepts": CONCEPTS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "monitoring_summary": [m["item"] for m in MONITORING[0]],
    }
