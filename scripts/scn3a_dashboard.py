"""
SCN3A Epilepsy (DEE67 / NaV1.3 Channelopathy / Focal Epilepsy of Infancy / 2q24.3)
======================================================================================
40-patient cohort · SCN3A GOF/LOF · Focal Epilepsy of Infancy · West/LGS spectrum · DEE67

SCN3A BIOLOGY:
SCN3A (2q24.3) encodes NaV1.3 — the voltage-gated sodium channel alpha subunit 3.
NaV1.3 is uniquely characterised by its developmental expression pattern: maximally
expressed in fetal and early neonatal brain, then normally downregulated postnatally
as NaV1.6 (SCN8A) is upregulated. In adult brain, NaV1.3 is re-expressed after
axotomy, injury, or recurrent seizures — creating a pro-epileptic feed-forward loop
(epilepsy begets more NaV1.3 re-expression). De novo GOF variants in SCN3A cause
DEE67 — a spectrum from severe neonatal encephalopathy to focal epilepsy of infancy
with high seizure rates.

KEY NaV1.3 BIOLOGY:
  - NaV1.3 has uniquely fast repriming (recovery from inactivation): τ_repriming ≈ 2-5 ms
    vs NaV1.2 (τ ≈ 15-20 ms) — makes NaV1.3-expressing neurons capable of sustained
    high-frequency burst firing
  - SCN3A R357Q: voltage sensor domain S4 — severe GOF causing both epilepsy AND
    polymicrogyria (cortical malformation from aberrant neuronal migration during
    fetal NaV1.3 peak expression period — channelopathy causing structural malformation)
  - GOF mechanisms: (1) right-shifted fast inactivation (channels remain open at
    physiological voltages), (2) slowed inactivation kinetics, (3) enhanced persistent
    Na+ current (I_NaP), (4) hyperpolarised activation threshold — all increase
    net depolarising Na+ flux → burst epileptic firing
  - NaV1.3 in thalamic relay neurons: implicated in thalamo-cortical burst firing
    → generalised spike-wave propagation in severe GOF

ALLELIC DISORDER SPECTRUM (DEE67, OMIM #619288):
  1. Severe GOF de novo — DEE67 / West → LGS evolution (25%)
     Neonatal or infantile onset, spasms, hypsarrhythmia; evolution to LGS;
     universal intellectual disability; treatment-resistant.
  2. Moderate GOF de novo — Focal Epilepsy of Infancy with High-Rate Seizures (35%)
     Most distinctive SCN3A phenotype: neonatal focal motor seizures (high rate,
     often >20/day in first weeks), autonomic features; may partially remit but
     often leaves cognitive sequelae. NaV1.3 peak expression = fetal/neonatal
     → explains striking neonatal predominance.
  3. Polymicrogyria-associated (R357Q) — severe (10%)
     The R357Q variant specifically causes bilateral perisylvian polymicrogyria
     (abnormal cortical folding from disrupted radial neuronal migration during
     fetal period when NaV1.3 is maximally expressed) + DEE.
  4. AD familial (GEFS+-like) — mild (20%)
     Autosomal dominant familial: febrile seizures ± afebrile GTCS, less severe
     cognitive phenotype. Partial penetrance.
  5. Phenocopy / panel-negative (10%)
     SCN1A (Dravet/GEFS+), SCN2A, SCN8A, SCN1B presenting with clinically similar
     infantile focal epilepsy or GEFS+ spectrum.

CRITICAL DRUG RULES FOR SCN3A:
  NaV blockers (CBZ/OXC/PHT/LTG): CONDITIONALLY beneficial in GOF but with caveats:
  - CBZ/OXC theoretically reduce NaV1.3 persistent current → may help focal/GEFS+ forms
  - However: AVOID in West syndrome / infantile spasms (NaV1.1 PV-interneuron suppression
    → spasm aggravation, same mechanism as SCN1A-Dravet)
  - LTG: MODERATE RISK if myoclonic component present
  - PHT: may be considered in GOF-dominant focal epilepsy but cardiac monitoring required

  TIAGABINE: ABSOLUTE CI in DEE67 / encephalopathy forms — NCSE risk.
  VPA: POLG MANDATORY. Good broad-spectrum coverage for spasm + GTCS component.
  QUINIDINE: Investigational for SCN3A GOF (NaV persistent current blocker) —
  used in SCN8A/SCN2A GOF; limited SCN3A-specific data; QT monitoring mandatory.
  KD: Level B for drug-resistant DEE67; KATP-independent mechanism via β-HB.
  ACTH: Level A for West syndrome (infantile spasms), regardless of SCN3A status.
"""

import random
from datetime import datetime

SEED = 9204  # dashboard 204
random.seed(SEED)

# ── Etiology Catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "SCN3A severe GOF de novo (DEE67 / West → LGS)",
        "n": 10, "pct": 25,
        "category": "SCN3A-GOF-DEE67-West-LGS",
        "mechanism": (
            "Approximately 25% of the SCN3A cohort carry severe de novo gain-of-function "
            "variants — the most clinically severe end of the SCN3A spectrum (DEE67, "
            "OMIM #619288). Severe GOF mechanisms include: right-shifted fast inactivation "
            "(channels remain open at near-resting voltages), markedly slowed inactivation "
            "kinetics, and substantially enhanced persistent Na+ current (I_NaP). In the "
            "neonatal brain — where NaV1.3 is maximally expressed in pyramidal neurons, "
            "thalamic relay neurons, and GABAergic interneurons — this produces catastrophic "
            "Na+-mediated depolarisation imbalance: sustained burst firing, thalamo-cortical "
            "hypersynchrony, and progressive cortical disinhibition. Clinical presentation: "
            "infantile spasms onset 3-8 months (hypsarrhythmia on EEG), evolving to "
            "Lennox-Gastaut syndrome (multiple seizure types, slow spike-wave <2.5 Hz, "
            "intellectual disability). ACTH Level A for West. VPA backbone. KD for "
            "DRE. Universal intellectual disability; SUDEP risk high."
        ),
        "eeg_correlate": (
            "Severe GOF DEE67 EEG: (1) Spasm phase: hypsarrhythmia — chaotic high-voltage "
            "mixed delta/theta with multifocal IEDs, voltage suppression after spasm cluster "
            "(modified hypsarrhythmia pattern in some). Spasm clusters on waking. "
            "(2) Post-ACTH: background improvement; persistence of focal IEDs (frontal). "
            "(3) LGS phase: slow generalised spike-wave <2.5 Hz, polyspike bursts during "
            "tonic seizures (frontal onset), background disorganisation, paroxysmal fast "
            "activity 10-20 Hz during tonic seizures. (4) Sleep: CSWS pattern (continuous "
            "spike-wave in slow sleep) in 30%. Background: severe diffuse slowing throughout."
        ),
        "mri_finding": (
            "Severe DEE67 MRI: (1) Usually normal in first months. (2) Progressive: diffuse "
            "cerebral volume loss, thin corpus callosum, deep white matter T2 hyperintensity "
            "after prolonged SE or recurrent seizure clusters. (3) MRS: reduced NAA/Cr in "
            "frontal cortex. (4) DWI: cortical restricted diffusion during SE (transient). "
            "3T MRI annually. Exclude R357Q polymicrogyria by reviewing cortical folding "
            "on T1 volumetric sequence (MPRAGE)."
        ),
    },
    {
        "etiology": "SCN3A moderate GOF de novo (Focal Epilepsy of Infancy / high-rate)",
        "n": 14, "pct": 35,
        "category": "SCN3A-GOF-FEI-high-rate",
        "mechanism": (
            "The most numerically common SCN3A phenotype (35%) — de novo missense variants "
            "with moderate GOF causing the distinctive 'Focal Epilepsy of Infancy with "
            "High-Rate Seizures' syndrome. Key features: onset in first 2 weeks to 4 months, "
            "focal motor seizures with high frequency (often 5-50 seizures/day in the acute "
            "phase, sometimes >100/day in the neonatal period), prominent autonomic features "
            "(flushing, apnoea, tachycardia), and transient tonic posturing. The extraordinary "
            "seizure frequency reflects NaV1.3's unique property of fast repriming (τ ≈ 2-5 ms "
            "vs τ ≈ 20 ms for NaV1.2) — neurons can sustain high-frequency burst firing with "
            "minimal refractory period. This neonatal-predominant presentation reflects the "
            "developmental expression peak of NaV1.3 in fetal/neonatal cortex: as postnatal "
            "NaV1.6 upregulation occurs by 2-3 months, NaV1.3 contribution diminishes and "
            "seizure frequency often falls spontaneously. However, permanent cognitive sequelae "
            "persist in 60% even after seizure reduction. CBZ/OXC may reduce high-frequency "
            "neonatal focal seizures (NaV1.3 persistent current inhibition). AVOID in "
            "spasm/West evolution. PHT IV for acute neonatal status. VPA + POLG screen."
        ),
        "eeg_correlate": (
            "Focal Epilepsy of Infancy SCN3A EEG: (1) Ictal: focal rhythmic beta/gamma "
            "discharge (10-40 Hz) in frontotemporal region, often right-sided, with "
            "electrodecrement at onset then high-frequency focal discharge evolving to "
            "regional rhythmic delta. Very short seizure duration (5-30 seconds) but "
            "extremely high frequency (serial clusters). (2) Interictal: focal IEDs "
            "(spike-slow wave) right frontotemporal; may switch hemisphere. Background: "
            "mildly slow for age in affected hemisphere. (3) High-rate phase: near-continuous "
            "focal IEDs between seizures in most severe phase. (4) EEG normalisation: "
            "partial background recovery by 6-12 months as NaV1.3 naturally downregulated. "
            "Continuous EEG monitoring mandatory in acute high-rate phase."
        ),
        "mri_finding": (
            "FEI-SCN3A MRI: (1) 60-70% normal on initial MRI. (2) 15-20%: ipsilateral "
            "cortical signal changes after prolonged focal SE (DWI restriction → FLAIR "
            "hyperintensity → cortical laminar necrosis in severe cases). (3) Atrophy: "
            "ipsilateral frontotemporal volume loss in drug-resistant cases. (4) Exclude "
            "polymicrogyria (R357Q variant) on T1 volumetric. MRI at diagnosis; repeat "
            "at 12M and if seizure pattern changes."
        ),
    },
    {
        "etiology": "SCN3A R357Q Polymicrogyria-Associated DEE (structural channelopathy)",
        "n": 4, "pct": 10,
        "category": "SCN3A-R357Q-PMG",
        "mechanism": (
            "The SCN3A R357Q variant occupies a unique niche in human channelopathy genetics: "
            "a single missense variant (Arg357→Gln in the S4 voltage sensor of domain II) "
            "causes both epilepsy AND bilateral perisylvian polymicrogyria — a structural "
            "cortical malformation characterised by abnormal cortical folding, irregular "
            "cortical surface, and microgyral pattern. The mechanism links the developmental "
            "expression timing of NaV1.3: R357Q creates a severe GOF channel that is "
            "maximally active during the fetal peak of NaV1.3 expression (embryonic weeks "
            "10-24), when radial neuronal migration and cortical organisation are occurring. "
            "Pathological NaV1.3-mediated depolarisation during this critical period disrupts "
            "the radial migration of neurons from the germinal zone to the cortical plate, "
            "producing cortical malformation. In postnatal life: bilateral perisylvian PMG "
            "produces pseudobulbar palsy (drooling, dysarthria, dysphagia) + polymicrogyria "
            "epilepsy (focal/multifocal, drug-resistant) + the ongoing GOF SCN3A-DEE. "
            "This group has the most complex treatment needs: surgical epilepsy evaluation "
            "(MST or resection rarely feasible for bilateral PMG), VNS Level B, KD Level B."
        ),
        "eeg_correlate": (
            "PMG-SCN3A R357Q EEG: (1) Bilateral perisylvian PMG pattern: continuous spike "
            "and slow wave over bilateral centrotemporal/perisylvian regions (CSWS-like), "
            "activated in NREM sleep. (2) Ictal: bilateral centrotemporal or perisylvian "
            "onset focal seizures; may secondarily generalise. (3) Atypical absence: "
            "diffuse slow spike-wave (2-3 Hz) in some. Background: bilateral perisylvian "
            "theta slowing. (4) During pseudobulbar episodes: oromotor seizure discharges "
            "(bilateral centrotemporal high-amplitude spike bursts). EEG + MRI combination "
            "essential for structural + functional localisation."
        ),
        "mri_finding": (
            "PMG R357Q: (1) Bilateral perisylvian polymicrogyria — unmistakable on 3T MRI: "
            "irregular cortical surface with over-folding, thickened cortex, sylvian fissure "
            "extension, and abnormal cortical grey-white junction. (2) Perisylvian extension "
            "from central opercular to posterior temporal/parietal operculum bilaterally. "
            "(3) Thin corpus callosum (isthmus/splenium hypoplasia) in 50%. (4) T2 cortical "
            "hyperintensity within PMG zones. 3T MRI with thin-slice T1 volumetric (MPRAGE "
            "1mm³ isotropic) + FLAIR + T2 GRE essential for PMG characterisation. "
            "SPECIFIC NOTE: if SCN3A R357Q detected on gene panel → mandatory 3T MRI "
            "to exclude/confirm PMG regardless of clinical severity."
        ),
    },
    {
        "etiology": "SCN3A AD familial (GEFS+-like / mild)",
        "n": 8, "pct": 20,
        "category": "SCN3A-AD-GEFS-like",
        "mechanism": (
            "20% carry autosomal dominant familial SCN3A variants — partial LOF or mild GOF "
            "variants with ~70% penetrance causing a GEFS+-like spectrum: febrile seizures "
            "(onset 6M-6Y), febrile seizures plus (FS extending beyond 6Y), and occasional "
            "afebrile GTCS or focal seizures in adulthood. NaV1.3 LOF in the familial context "
            "is paradoxical: reduced NaV1.3 current → loss of the Na+/K+ homeostatic "
            "feedback mechanism → increased network excitability under certain conditions "
            "(fever, fatigue, sodium channel turnover states). Mild variant carriers often "
            "undiagnosed until a more severely affected family member (de novo DEE67) triggers "
            "cascade family testing. Prognosis: generally favourable — most achieve seizure "
            "freedom by adolescence. CBZ/OXC useful for breakthrough afebrile focal seizures. "
            "Avoid NaV blockers during febrile seizure phase (risk of masking NaV reserve)."
        ),
        "eeg_correlate": (
            "AD familial SCN3A EEG: (1) Febrile seizure phase: normal interictal EEG. "
            "Ictal EEG (rarely captured): generalised or hemispheric rhythmic delta during "
            "febrile convulsion. (2) GEFS+ / afebrile phase: normal interictal EEG in 70%; "
            "mild centrotemporal IEDs in 20%; generalised spike-wave in 10%. (3) Background: "
            "NORMAL — distinguishes familial mild SCN3A from DEE67. (4) Photoparoxysmal "
            "response absent (unlike GABRB2-GEFS+). Normal sleep architecture."
        ),
        "mri_finding": (
            "Familial SCN3A GEFS+-like: Brain MRI NORMAL in all cases. "
            "No structural findings. MRI performed once at initial diagnostic workup to "
            "exclude structural lesion. No serial MRI required if development is normal "
            "and seizures respond to AEDs. Key distinction: if MRI shows PMG → R357Q "
            "variant subtype (must be excluded by direct sequencing of exon 9)."
        ),
    },
    {
        "etiology": "SCN3A phenocopy (SCN1A / SCN2A / SCN8A / SCN1B panel-negative)",
        "n": 4, "pct": 10,
        "category": "SCN3A-phenocopy",
        "mechanism": (
            "10% of patients initially referred with clinical suspicion of SCN3A-related "
            "epilepsy have negative SCN3A sequencing and are reclassified as SCN1A (Dravet "
            "syndrome / GEFS+ — SCN1A mosaic missed on standard sequencing), SCN2A (neonatal "
            "focal seizures with high frequency resembling FEI-SCN3A), SCN8A (DEE13 — GOF, "
            "neonatal/infantile onset, similar NaV channelopathy), or SCN1B (GEFS+ familial). "
            "Clinical distinction requires comprehensive gene panel (at minimum SCN1A/2A/3A/8A/"
            "1B) plus deep-read sequencing to detect mosaic SCN1A variants (5-20% mosaic "
            "allele fraction, missed on standard exome at 50× coverage). Phenocopy rate is "
            "highest in the mild familial GEFS+-like presentation (mimicked by SCN1B-GEFS+) "
            "and in severe neonatal focal epilepsy (mimicked by SCN2A-FEI or SCN8A-DEE13)."
        ),
        "eeg_correlate": (
            "SCN3A phenocopy EEG: Indistinguishable from true SCN3A by EEG alone. "
            "(1) SCN2A phenocopy: neonatal focal discharge (often right temporal), high-"
            "rate clusters, autonomic features — identical to FEI-SCN3A. (2) SCN8A phenocopy: "
            "diffuse epileptiform abnormalities, paroxysmal tonic phase, movement-related "
            "seizures (extensor posturing). (3) SCN1A/GEFS+ phenocopy: centrotemporal IEDs, "
            "normal background. Gene panel + extended sequencing is the ONLY definitive "
            "distinguishing test — EEG alone cannot differentiate NaV channelopathies."
        ),
        "mri_finding": (
            "Phenocopy MRI: Normal in GEFS+/FEI phenocopy SCN1A/SCN1B cases. "
            "SCN8A phenocopy: may show basal ganglia T1 shortening (motor predominance). "
            "Key: if MRI shows PMG → SCN3A R357Q specific (not a phenocopy); immediate "
            "targeted R357Q sequencing. If MRI normal: comprehensive gene panel required."
        ),
    },
]

# ── Seizure Types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal motor / clonic (neonatal high-rate)",
        "pct_patients": 80,
        "eeg_correlate": "Focal rhythmic beta-gamma (10-40 Hz) → regional delta evolution; right frontotemporal most common",
        "clinical_tip": (
            "High rate (5-50/day in acute phase) is the SCN3A hallmark. "
            "CBZ/OXC IV first-line for high-rate neonatal focal. PB for breakthrough. "
            "Continuous EEG monitoring mandatory. Expect partial frequency reduction "
            "by 2-3 months as NaV1.3 naturally downregulated by NaV1.6."
        ),
    },
    {
        "type": "Infantile spasms / West syndrome",
        "pct_patients": 30,
        "eeg_correlate": "Hypsarrhythmia; electrodecrement + high-amplitude muscle artefact on spasm; background chaotic high-voltage",
        "clinical_tip": (
            "ACTH Level A (UK-ISSS protocol) — 2 weeks high-dose synthetic ACTH or "
            "prednisolone. Day 14 EEG mandatory. VGB Level A second-line (SHARE REMS, "
            "ERG q6M, ≤16 weeks). AVOID CBZ/OXC during spasms (NaV1.1 PV-interneuron "
            "suppression aggravates spasms — same mechanism as SCN1A-Dravet)."
        ),
    },
    {
        "type": "Tonic seizures (LGS evolution)",
        "pct_patients": 25,
        "eeg_correlate": "Paroxysmal fast activity (10-25 Hz frontal) + electrodecrement at tonic onset; slow SWD interictal",
        "clinical_tip": (
            "Rufinamide Level B for LGS tonic/atonic seizures. VPA + CLB backbone. "
            "Corpus callosotomy Level A for drop attacks. Perampanel Level B adjunct. "
            "AVOID VGB in LGS phase (ineffective; prior VGB exposure for spasms may "
            "have already occurred with REMS monitoring requirement)."
        ),
    },
    {
        "type": "Atonic (drop attacks / LGS)",
        "pct_patients": 18,
        "eeg_correlate": "Generalised polyspike or spike-wave → atonia; 2-3 Hz SWD or fast polyspike at atonic onset",
        "clinical_tip": (
            "Protective helmet mandatory for drop attack patients. Corpus callosotomy "
            "Level A (>80% drop attack reduction). Rufinamide or CLB adjunct. "
            "VNS Level B if surgery not feasible. AVOID PHT (ineffective for atonic, "
            "potential toxicity in DEE). Emergency seizure plan for clusters."
        ),
    },
    {
        "type": "Focal autonomic (apnoea / tachycardia / flushing)",
        "pct_patients": 45,
        "eeg_correlate": "Frontoinsular or temporal focal discharge; may appear as EEG-only ictal without visible motor signs",
        "clinical_tip": (
            "Autonomic focal seizures (apnoea, colour change, HR irregularity) in "
            "neonates/infants are easily misdiagnosed as cardiorespiratory events. "
            "Continuous EEG monitoring + pulse oximetry correlation mandatory. "
            "IV phenobarbital or LEV for acute management. CPAP/O₂ for apnoeic episodes. "
            "SCN3A panel if neonatal unexplained apnoea + EEG seizures."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / intercurrent illness",
        "pct": 78,
        "threshold": "Fever >37.8°C (lower than standard 38°C threshold) sufficient for FEI-SCN3A; >37.5°C in DEE67",
        "management": (
            "Antipyretic at first sign of fever (paracetamol/ibuprofen). Emergency "
            "benzodiazepine plan (diastat/buccal midazolam) for prolonged seizure. "
            "Lower temperature action threshold than standard GEFS+ protocols."
        ),
    },
    {
        "trigger": "Neonatal physiological activation (feeding, handling, crying)",
        "pct": 72,
        "threshold": "Any arousal stimulus in acute neonatal phase",
        "management": (
            "Clustered care protocol — minimise unnecessary handling in acute high-rate "
            "phase. NaV1.3 channels are exquisitely activation-dependent in the neonatal "
            "context. Feeding via NG tube during acute phase if oropharyngeal seizures. "
            "Dimmed lighting, minimal noise stimulation in NICU."
        ),
    },
    {
        "trigger": "Sleep-wake transition",
        "pct": 65,
        "threshold": "Stage N2 onset and post-arousal from N2/N3 (major transitions)",
        "management": (
            "Adequate sleep hygiene — regular sleep schedule. Evening AED dosing "
            "optimisation to cover peak transition window. Consider evening CLB dose "
            "for nocturnal cluster prevention. CSWS monitoring: overnight EEG q6-12M "
            "in DEE67 (30% CSWS rate)."
        ),
    },
    {
        "trigger": "Missed AED dose",
        "pct": 62,
        "threshold": ">4 hours beyond scheduled dose",
        "management": (
            "Dose alarm + caregivers trained on rescue protocol. Do NOT double the "
            "next dose (toxicity risk). If cluster occurs after missed dose: buccal "
            "midazolam per emergency plan."
        ),
    },
    {
        "trigger": "Hyperthermia (non-infectious: hot bath, exercise)",
        "pct": 45,
        "threshold": "Core temp rise >0.8°C — particularly relevant for FEI and GEFS+ variants",
        "management": (
            "Avoid prolonged hot baths. Exercise in cool environment. Monitor "
            "temperature during exercise sessions. Pre-treat with paracetamol "
            "before anticipated heat exposure. NaV1.3 temperature-sensitivity "
            "(Q10 ≈ 2-3 for gating kinetics): GOF worsens with temperature."
        ),
    },
    {
        "trigger": "Sodium channel drug interactions (CBZ + PHT level fluctuations)",
        "pct": 38,
        "threshold": "CBZ level drop >20% below therapeutic range",
        "management": (
            "CBZ level monitoring mandatory (target 4-12 mg/L). Drug interactions: "
            "PHT + CBZ → CBZ-epoxide accumulation (neurotoxicity). Enzyme-inducing "
            "AEDs reduce CBZ levels. Therapeutic drug monitoring at each clinic visit "
            "and after any drug change."
        ),
    },
    {
        "trigger": "Intercurrent metabolic stress (hyponatraemia, hypoglycaemia)",
        "pct": 32,
        "threshold": "Na+ <133 mmol/L or glucose <3.5 mmol/L → seizure breakthrough",
        "management": (
            "OXC-SIADH risk: serum Na+ monitoring mandatory on OXC (baseline, Day 3, "
            "Day 7, Day 14, monthly). Supplement sodium if hyponatraemia. "
            "KD patients: maintain adequate glucose intake (hypoglycaemia risk). "
            "Electrolyte panel at each visit."
        ),
    },
    {
        "trigger": "Social/emotional stress (older children/adolescents with mild SCN3A)",
        "pct": 25,
        "threshold": "Acute stress, exam, emotional arousal",
        "management": (
            "Stress management referral. CBT for anxiety-associated seizure frequency. "
            "Ensure adequate sleep during stressful periods. Rescue plan for breakthrough."
        ),
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "ACTH (Tetracosactide / Synacthen)",
        "evidence": "Level A",
        "indication": "Infantile spasms / West syndrome (regardless of SCN3A GOF status)",
        "dose_moa": (
            "UK-ISSS Protocol: High-dose synthetic ACTH 40-60 IU/m²/day IM × 14 days, "
            "then taper over 2 weeks. Alternative: oral prednisolone 10 mg QDS × 14 days. "
            "MOA: anti-inflammatory + direct GABAergic effect + normalise ACTH-CRH axis "
            "dysregulation in infantile spasms. DAY 14 EEG mandatory — hypsarrhythmia "
            "resolution = treatment success. If not resolved → switch treatment protocol."
        ),
        "efficacy": "Spasm cessation in 55-70% by Day 14 (infantile spasms UK-ISSS); EEG response (hypsarrhythmia resolution) in 60%",
        "monitoring": [
            "Blood pressure (hypertension — daily in first week)",
            "Blood glucose (steroid hyperglycaemia, check QDS)",
            "Electrolytes (hypokalaemia, Na+ — every 48h)",
            "Infection screen (immunosuppression — any fever → septic workup)",
            "Weight / fluid balance (Cushingoid effects)",
            "Day 14 EEG (mandatory — hypsarrhythmia resolution endpoint)",
        ],
        "scn3a_note": (
            "SCN3A GOF + West syndrome: ACTH is first-line (Level A) — the GOF mechanism "
            "does not diminish the anti-spasmodic ACTH response. AVOID NaV blockers (CBZ/OXC) "
            "during the infantile spasm phase — risk of spasm aggravation via NaV1.1 "
            "fast-spiking interneuron suppression."
        ),
    },
    {
        "drug": "VPA (Sodium Valproate)",
        "evidence": "Level B (SCN3A-GEFS+ / DEE67)",
        "indication": "Broad-spectrum AED for SCN3A GTCS, absence, spasms, DEE67",
        "dose_moa": (
            "20-40 mg/kg/day PO/IV divided BD-TDS. Target level 50-100 mg/L (trough). "
            "MOA: NaV channel blockade (I_NaP reduction), GABA transaminase inhibition "
            "(↑brain GABA), T-type Ca²⁺ channel blockade (thalamic oscillations), HCN "
            "channel modulation. Broad-spectrum coverage including spasms, absence, "
            "GTCS, myoclonic, focal components."
        ),
        "efficacy": "50-60% ≥50% seizure reduction in SCN3A-DEE67; better response in GEFS+-like forms (70-80%)",
        "monitoring": [
            "POLG sequencing MANDATORY before use (fatal Alpers hepatotoxicity in POLG carriers)",
            "LFTs + ammonia at baseline, 4 weeks, 12 weeks, then 6-monthly",
            "FBC + platelets (thrombocytopaenia, VPA-induced)",
            "VPA trough level (50-100 mg/L; ammonia >80 µmol/L → L-carnitine 50 mg/kg/day)",
            "VPPP mandatory in females of childbearing age (teratogenicity NTD 2-10%)",
            "Weight / pancreatitis signs (annual amylase/lipase if symptoms)",
        ],
        "scn3a_note": (
            "VPA is appropriate backbone for SCN3A-GEFS+ (focal + GTCS component) and for "
            "DEE67 (broad-spectrum). POLG MANDATORY — the SCN3A DEE67 cohort includes "
            "infants where VPA + POLG mutation = catastrophic Alpers hepatic failure. "
            "VPPP mandatory for all females of childbearing age regardless of SCN3A subtype."
        ),
    },
    {
        "drug": "CBZ / OXC (Carbamazepine / Oxcarbazepine)",
        "evidence": "Level B (SCN3A GOF focal — conditional; AVOID West/spasms)",
        "indication": "SCN3A GOF focal epilepsy of infancy / GEFS+-like (moderate/mild forms); NOT for West/spasms",
        "dose_moa": (
            "CBZ: 10-20 mg/kg/day PO BD-TDS (level 4-12 mg/L). "
            "OXC: 15-30 mg/kg/day PO BD. "
            "MOA: NaV channel use-dependent block → reduces persistent Na+ current (I_NaP) "
            "and high-frequency firing. In SCN3A GOF: CBZ/OXC theoretically targets the "
            "underlying GOF mechanism (enhanced I_NaP) by increasing fast inactivation rate. "
            "Evidence: SCN3A case series showing CBZ response in focal epilepsy of infancy."
        ),
        "efficacy": "30-50% significant reduction in focal seizure frequency in moderate GOF SCN3A-FEI (case series); limited controlled data",
        "monitoring": [
            "HLA-B*15:02 MANDATORY before CBZ/OXC in Asian ancestry (CPIC Level A — SJS/TEN risk)",
            "CBZ level (4-12 mg/L); OXC: MHD level 12-35 mg/L",
            "Serum Na+ on OXC (SIADH): baseline, Day 3, Day 7, Day 14, monthly",
            "LFTs (hepatotoxicity, rare)",
            "FBC (aplastic anaemia risk with CBZ, rare — monitoring per BNF)",
            "Rash screening (carbamazepine hypersensitivity)",
        ],
        "scn3a_note": (
            "CRITICAL SCN3A RULE: CBZ/OXC conditionally appropriate for focal epilepsy of "
            "infancy (GOF, no spasm/West component) but ABSOLUTELY AVOID during infantile "
            "spasm phase — NaV1.1 PV-interneuron suppression aggravates spasms (same "
            "mechanism as SCN1A-Dravet). Resume consideration after spasm remission only. "
            "HLA-B*15:02 mandatory in any Asian-ancestry patient before first CBZ/OXC dose."
        ),
    },
    {
        "drug": "LEV (Levetiracetam)",
        "evidence": "Level B (adjunct)",
        "indication": "SCN3A DEE67 adjunct; neonatal seizures (IV LEV Level B); GTCS component",
        "dose_moa": (
            "20-60 mg/kg/day PO/IV divided BD. IV LEV: loading 40-60 mg/kg over 15 min. "
            "MOA: SV2A synaptic vesicle protein 2A binding → impaired vesicular NT release "
            "(glutamate + GABA) — atypical mechanism not directly targeting NaV channels. "
            "Good IV formulation for acute neonatal/infantile seizure management."
        ),
        "efficacy": "35-50% adjunct response in SCN3A DEE67 and neonatal focal seizures; often partial",
        "monitoring": [
            "Behavioural monitoring (irritability, aggression — SV2A mechanism, common in infants)",
            "Renal function (LEV is renally cleared; dose adjust if eGFR <50)",
            "Baseline cognitive assessment (repeat annually in DEE67)",
        ],
        "scn3a_note": (
            "LEV IV is a safe first-choice for acute neonatal seizure management in SCN3A "
            "pending gene confirmation (no organ toxicity, no QT, available IV). "
            "Behavioural side effects (irritability, aggression) are dose-dependent and "
            "particularly prominent in infants/toddlers with DEE — monitor closely."
        ),
    },
    {
        "drug": "CLB (Clobazam)",
        "evidence": "Level B (adjunct, LGS / clusters)",
        "indication": "SCN3A LGS evolution; acute seizure cluster prevention; tonic/atonic adjunct",
        "dose_moa": (
            "0.1-0.3 mg/kg/day PO (max 1 mg/kg/day) in BD dosing. "
            "MOA: 1,5-benzodiazepine GABA-A allosteric positive modulator (γ2-containing "
            "receptor benzodiazepine site) — less sedation than 1,4-BDZ (diazepam, "
            "clonazepam) due to lower α1/α2 ratio. Reduces tonic/atonic cluster frequency "
            "in LGS phase. Catamenial pulse dosing in adolescent females."
        ),
        "efficacy": "50-60% ≥50% reduction in drop attack frequency as adjunct in LGS (FDA-approved 2011 for LGS)",
        "monitoring": [
            "Sedation / cognitive assessment (monthly initially)",
            "Saliva / secretion increase (may worsen swallowing in PMG patients)",
            "Tolerance assessment (loss of effect after 3-6 months in 30%)",
            "Weight (appetite stimulation)",
        ],
        "scn3a_note": (
            "CLB is FDA-approved for LGS (2011) — appropriate for SCN3A evolving to LGS "
            "phenotype (DEE67 severe group). In R357Q polymicrogyria with pseudobulbar palsy: "
            "monitor swallowing carefully — secretion increase from CLB may worsen aspiration "
            "risk in patients with pre-existing swallowing difficulties."
        ),
    },
    {
        "drug": "KD (Ketogenic Diet)",
        "evidence": "Level B (drug-resistant DEE67 / LGS / focal)",
        "indication": "SCN3A DEE67 drug-resistant; ≥2 AED failure; LGS phase",
        "dose_moa": (
            "Classical KD 4:1 ratio initiated under dietitian supervision with anticonvulsant "
            "β-hydroxybutyrate (β-HB) target 2-5 mmol/L. "
            "MOA: β-HB → KATP channel activation (independent of NaV mechanism) + vesicular "
            "glutamate transporter inhibition (VGLUT2) + reduced glucose-derived "
            "acetyl-CoA → reduced mTOR signalling → anticonvulsant. Mechanism independent "
            "of NaV channel — effective even in SCN3A GOF where NaV blockers may "
            "be contraindicated (West/spasms phase)."
        ),
        "efficacy": "50% achieve ≥50% seizure reduction in DEE/LGS (Cochrane 2018); 10-15% seizure-free on KD",
        "monitoring": [
            "β-OHB target 2-5 mmol/L (urine or serum ketones)",
            "Glucose (hypoglycaemia risk — maintain >3.5 mmol/L)",
            "Growth parameters (monthly — KD restricts calories)",
            "Electrolytes + pH (metabolic acidosis; base-supplement PRN)",
            "Lipids (annual TG, LDL, HDL)",
            "Renal ultrasound (kidney stones risk 5-10% — adequate hydration)",
            "AVOID high-ratio KD (4:1) + Topiramate concurrently (additive acidosis)",
        ],
        "scn3a_note": (
            "KD is particularly appropriate for SCN3A DEE67 evolving to LGS phase (≥2 "
            "AED failure) — the KATP/VGLUT2 mechanism is entirely NaV-independent, so "
            "KD can be safely combined with CBZ/OXC (if used for focal component) or "
            "VPA. AVOID high-ratio KD + Topiramate (additive metabolic acidosis in infants)."
        ),
    },
    {
        "drug": "Quinidine (investigational for SCN3A GOF)",
        "evidence": "Investigational / Case Report",
        "indication": "SCN3A GOF persistent current — investigational (limited data); used in SCN8A/SCN2A GOF",
        "dose_moa": (
            "Quinidine 15-30 mg/kg/day PO (cardiac dosing; serum level 2-5 mg/L). "
            "MOA: Class Ia antiarrhythmic → NaV channel open-state blocker targeting "
            "persistent (non-inactivating) Na+ current (I_NaP). More selective for I_NaP "
            "vs transient I_NaT at low concentrations. Evidence base: documented efficacy "
            "in SCN8A GOF (DEE13) and SCN2A GOF (neonatal); SCN3A GOF: 2-3 published "
            "case reports/series showing partial response. "
            "QTc monitoring MANDATORY (quinidine is a Class Ia drug with QT-prolonging "
            "and pro-arrhythmic risk — requires 12-lead ECG + telemetry at initiation)."
        ),
        "efficacy": "Case reports: 30-60% focal seizure reduction in SCN3A GOF (FEI); not randomised controlled; use in specialist centres only",
        "monitoring": [
            "12-lead ECG at baseline (QTc <440 ms required before initiation)",
            "QTc monitoring: at 48h, Day 7, Day 14, monthly (STOP if QTc >500 ms or >60 ms increase)",
            "Serum quinidine level (target 2-5 mg/L; toxicity >8 mg/L)",
            "BP monitoring (quinidine alpha-blockade → hypotension)",
            "Thrombocytopaenia check (quinidine-immune platelet destruction — monthly FBC)",
            "Drug interactions: quinidine inhibits CYP2D6 — monitor all co-medications metabolised by CYP2D6",
        ],
        "scn3a_note": (
            "INVESTIGATIONAL ONLY — not a standard-of-care AED for SCN3A. Use ONLY in "
            "specialist centres with paediatric cardiology support (QT monitoring). "
            "Most relevant for moderate GOF FEI-SCN3A phenotype with documented I_NaP "
            "on patch-clamp functional validation. Contrast with CBZ/OXC (use-dependent "
            "NaV block) — quinidine targets I_NaP more selectively. Do NOT use quinidine "
            "if patient has structural cardiac disease, AV block, or concurrent QT-prolonging "
            "drugs (azithromycin, haloperidol, etc)."
        ),
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Tiagabine",
        "risk": "ABSOLUTE CI",
        "reason": (
            "GAT-1 GABA reuptake inhibitor → extrasynaptic GABA accumulation → tonic "
            "GABA-A receptor activation → NCSE (non-convulsive status epilepticus). "
            "In SCN3A-DEE67 and encephalopathic forms: the already-compromised inhibitory "
            "architecture makes extrasynaptic GABA toxicity catastrophic. NEVER use in any "
            "SCN3A encephalopathy (DEE67, LGS-evolution, R357Q-PMG)."
        ),
        "alternative": "CLB (1,5-BDZ) or LEV for adjunct GABAergic/vesicular modulation",
    },
    {
        "drug": "CBZ / OXC during West syndrome / infantile spasms phase",
        "risk": "HIGH RISK — spasm aggravation",
        "reason": (
            "NaV1.1 PV-interneuron suppression: CBZ/OXC block NaV1.1-expressing "
            "fast-spiking parvalbumin interneurons → reduced cortical inhibitory tone "
            "→ spasm aggravation. Same mechanism as in SCN1A-Dravet and KCNQ3-West. "
            "CONDITIONAL: CBZ/OXC may be reintroduced after successful spasm treatment "
            "for focal seizure residual component."
        ),
        "alternative": "ACTH (Level A) + VGB (Level A, ≤16w) for infantile spasms",
    },
    {
        "drug": "VPA without POLG sequencing",
        "risk": "HIGH RISK — Alpers-Huttenlocher fatal hepatic failure",
        "reason": (
            "POLG (polymerase gamma) mutations → VPA hepatotoxicity (Alpers-Huttenlocher "
            "syndrome): fatal progressive hepatic failure in POLG carriers exposed to VPA. "
            "POLG carrier frequency ~1:200. In SCN3A-DEE67 infants who may require VPA: "
            "POLG panel (point mutations + deletions + CNV) MANDATORY before first dose. "
            "If urgent VPA needed: interim use only pending POLG result; switch if positive."
        ),
        "alternative": "LEV + CLB bridge while awaiting POLG result if urgently needed",
    },
    {
        "drug": "LTG (Lamotrigine) with myoclonic component",
        "risk": "MODERATE-HIGH RISK — myoclonic aggravation",
        "reason": (
            "LTG: NaV1.1 PV-interneuron suppression → myoclonic aggravation in epilepsies "
            "with GABAergic interneuron dependence. In SCN3A DEE67 evolving to LGS with "
            "myoclonic component (myoclonic-atonic, myoclonic-absence): LTG can dramatically "
            "worsen myoclonic frequency. If no myoclonic component: LTG is lower risk for "
            "GTCS/focal adjunct — but screen for myoclonia before initiating."
        ),
        "alternative": "VPA or CLB if myoclonic component present; consider LTG only in pure GTCS/focal form",
    },
    {
        "drug": "Quinidine without cardiac monitoring and specialist oversight",
        "risk": "HIGH RISK — QT prolongation, torsades de pointes, pro-arrhythmia",
        "reason": (
            "Quinidine Class Ia antiarrhythmic: QT-prolonging, pro-arrhythmic (torsades de "
            "pointes risk), alpha-blocking (hypotension), CYP2D6 inhibitor. Use ONLY under "
            "paediatric cardiology oversight with continuous ECG monitoring at initiation. "
            "Contraindicated in: QTc >440 ms at baseline, congenital long QT syndrome, "
            "AV block, concurrent QT-prolonging medications, structural cardiac disease."
        ),
        "alternative": "CBZ/OXC for GOF focal epilepsy; quinidine reserved for specialist setting only",
    },
]

# ── Monitoring ────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG sequencing MANDATORY before VPA", "frequency": "Once (before VPA initiation)", "rationale": "Fatal Alpers-Huttenlocher hepatic failure in POLG carriers"},
    {"item": "Day 14 EEG (hypsarrhythmia check)", "frequency": "Day 14 after ACTH initiation for infantile spasms", "rationale": "ACTH response endpoint (hypsarrhythmia resolution = success)"},
    {"item": "HLA-B*15:02 before CBZ/OXC (Asian ancestry)", "frequency": "Once before CBZ/OXC initiation", "rationale": "Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis — CPIC Level A"},
    {"item": "Serum Na+ on OXC (SIADH monitoring)", "frequency": "Baseline, Day 3, Day 7, Day 14, monthly", "rationale": "OXC-SIADH: symptomatic hyponatraemia in 10-25%"},
    {"item": "VPA trough level + LFTs + ammonia", "frequency": "Baseline, 4w, 12w, then 6-monthly", "rationale": "VPA hepatotoxicity, hyperammonaemia, thrombocytopaenia"},
    {"item": "QTc monitoring (Quinidine initiation)", "frequency": "Baseline, 48h, Day 7, Day 14, monthly", "rationale": "Quinidine QT prolongation / pro-arrhythmia risk"},
    {"item": "R357Q MRI (3T, thin-slice T1 volumetric)", "frequency": "At diagnosis if SCN3A R357Q detected", "rationale": "Bilateral perisylvian polymicrogyria — structural malformation requiring dedicated 3T MRI"},
    {"item": "Neurodevelopmental assessment", "frequency": "Every 12 months", "rationale": "DEE67 developmental trajectory; cognitive + motor + language sequelae"},
    {"item": "SUDEP counselling + nocturnal monitoring", "frequency": "Annual (from age 2 in DEE67)", "rationale": "SUDEP risk elevated in DEE67 / LGS phase; nocturnal sensor (Emfit/mattress) recommended"},
    {"item": "VGB ERG (electroretinogram) if VGB used", "frequency": "Every 6 months during VGB (SHARE REMS requirement)", "rationale": "Irreversible visual field constriction (VGB); REMS-mandated monitoring"},
],

# ── Lifecycle ─────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {"window": "Prenatal / Fetal", "ages": "Conception – birth", "events": "NaV1.3 peak expression (fetal cortex weeks 10-30); R357Q → polymicrogyria during radial migration; fetal MRI if R357Q detected in prenatal sequencing"},
    {"window": "Neonatal Acute", "ages": "Birth – 4 weeks", "events": "High-rate focal motor seizures (FEI onset); autonomic seizures (apnoea, tachycardia); continuous EEG monitoring; IV LEV/PB acute management; SCN3A panel send"},
    {"window": "Infantile", "ages": "1 month – 12 months", "events": "West syndrome / infantile spasms (ACTH Level A); NaV1.3 natural downregulation begins (NaV1.6 upregulation); seizure frequency may partially fall; VPA + POLG; CBZ consider post-spasms"},
    {"window": "Toddler / Preschool", "ages": "1 – 5 years", "events": "LGS evolution in severe DEE67 (tonic/atonic drop attacks, slow SWD); corpus callosotomy consideration; KD initiation if ≥2 AED failure; neurodevelopmental support"},
    {"window": "School age", "ages": "5 – 12 years", "events": "Persistent focal or multifocal epilepsy; cognitive / behavioural support; CSWS monitoring (overnight EEG q6-12M); seizure diary; quality of life assessment"},
    {"window": "Adolescent / Adult", "ages": "12 years+", "events": "Mild GEFS+-like: seizure freedom common; VPPP in females (VPA); severe DEE67: multidisciplinary; transition to adult neurology; SUDEP annual counselling; vocational / independent living planning"},
]

# ── Concepts ──────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "SCN3A (2q24.3)", "definition": "Gene encoding NaV1.3 voltage-gated Na+ channel alpha subunit; fetal peak expression; fast repriming; DEE67 de novo GOF; focal epilepsy of infancy; GEFS+-like familial"},
    {"term": "NaV1.3 fast repriming", "definition": "Recovery from inactivation τ ≈ 2-5 ms (vs NaV1.2 τ ≈ 20 ms) — enables sustained high-frequency burst firing; explains SCN3A high-rate neonatal seizures"},
    {"term": "DEE67 (OMIM #619288)", "definition": "Developmental and Epileptic Encephalopathy 67; SCN3A de novo GOF; severe; West → LGS evolution; intellectual disability; SUDEP risk"},
    {"term": "Focal Epilepsy of Infancy (FEI)", "definition": "SCN3A moderate GOF: neonatal-onset focal motor seizures, high rate (>5-50/day); autonomic features; natural partial remission as NaV1.6 upregulates postnatally"},
    {"term": "R357Q Polymicrogyria", "definition": "SCN3A R357Q (S4 voltage sensor domain II) — severe GOF during fetal NaV1.3 peak → bilateral perisylvian PMG (cortical malformation); pseudobulbar palsy; DEE"},
    {"term": "NaV1.3 persistent current (I_NaP)", "definition": "Persistent (non-inactivating) Na+ current enhanced by GOF variants; mediates sustained depolarisation; target for quinidine (investigational) and CBZ/OXC"},
    {"term": "NaV1.3 developmental switch", "definition": "NaV1.3 peaks fetal-neonatal; downregulated postnatally as NaV1.6 (SCN8A) upregulates; explains neonatal predominance and partial spontaneous improvement at 2-3 months"},
    {"term": "ACTH for infantile spasms", "definition": "Tetracosactide (synthetic ACTH) Level A for West syndrome; UK-ISSS protocol; Day 14 EEG mandatory; anti-inflammatory + GABAergic mechanism"},
    {"term": "POLG-Alpers-Huttenlocher", "definition": "POLG (polymerase gamma) mutation carriers → VPA exposure → fatal progressive hepatic failure (Alpers syndrome); POLG MANDATORY before VPA in any SCN3A patient"},
    {"term": "HLA-B*15:02 SJS/TEN", "definition": "CPIC Level A: HLA-B*15:02 allele in Asian ancestry → CBZ/OXC exposure → Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis; mandatory genotyping before CBZ/OXC"},
    {"term": "OXC-SIADH", "definition": "Oxcarbazepine → SIADH (syndrome of inappropriate ADH) → hyponatraemia; Na+ monitoring: baseline, Day 3, Day 7, Day 14, monthly"},
    {"term": "Quinidine (NaV I_NaP blocker)", "definition": "Class Ia antiarrhythmic; persistent Na+ current blocker; investigational for SCN3A GOF; QT monitoring mandatory; used in SCN8A/SCN2A GOF — limited SCN3A data"},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021: all females on VPA must be on VPPP (annual review + contraception + counselling); NTD teratogenicity 2-10%; SCN3A DEE67 females from puberty"},
    {"term": "CSWS (Continuous Spike-Wave in Slow Sleep)", "definition": "Electrical status epilepticus in NREM sleep; in SCN3A DEE67 (30%); causes developmental regression; overnight EEG q6-12M monitoring"},
    {"term": "SUDEP risk (DEE67)", "definition": "Sudden Unexpected Death in Epilepsy; elevated in SCN3A DEE67 / LGS phase; annual counselling; nocturnal seizure monitoring (Emfit/mattress); seizure diary"},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    "POLG sequencing MANDATORY before VPA in any SCN3A patient (DEE67 + GEFS+ + FEI)",
    "Day 14 EEG MANDATORY after ACTH initiation (hypsarrhythmia resolution = success endpoint)",
    "HLA-B*15:02 MANDATORY before CBZ/OXC in any patient of Asian ancestry (CPIC Level A)",
    "OXC serum Na+: baseline → Day 3 → Day 7 → Day 14 → monthly (SIADH prevention)",
    "VPA target trough 50-100 mg/L; ammonia >80 µmol/L → L-carnitine 50 mg/kg/day",
    "Quinidine: QTc >440 ms baseline → do NOT initiate; QTc >500 ms on treatment → STOP",
    "Fever action threshold: 37.8°C for FEI-SCN3A; 37.5°C for DEE67 (antipyretic + emergency BDZ plan)",
    "VGB: maximum 16 weeks for West syndrome (SHARE REMS); ERG every 6 months (irreversible VF constriction)",
    "R357Q variant detected on panel: 3T MRI with thin-slice T1 volumetric MANDATORY (PMG exclusion/confirmation)",
    "CSWS screening: overnight EEG every 6-12 months in DEE67 (30% CSWS rate → developmental regression)",
    "SUDEP annual counselling from age 2 in DEE67; nocturnal seizure monitoring device recommended",
    "VPPP annual review for all females on VPA from puberty (MHRA 2021 mandatory requirement)",
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE 2022 Classification of the Epilepsies",
    "NICE NG217 (Epilepsies in children, young people and adults, 2022)",
    "UK-ISSS (United Kingdom Infantile Spasms Study) — ACTH Level A for West syndrome",
    "CPIC HLA-B*15:02 guideline 2023 (Level A — CBZ/OXC in Asian ancestry)",
    "SHARE REMS (Vigabatrin Risk Evaluation and Mitigation Strategy — visual field)",
    "MHRA VPPP 2021 (Valproate Pregnancy Prevention Programme — mandatory)",
    "ACMG-AMP 2015 (Variant classification standards)",
    "Bhatt DL et al. (2023) — SCN3A DEE67 review, Epilepsia",
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Holland KD et al. (2008) — SCN3A mutation in focal cortical dysplasia and epilepsy. Ann Neurol.",
    "Veeramah KR et al. (2012) — Exome sequencing reveals new causal mutations in SCN3A. Nat Genet.",
    "Zaman T et al. (2019) — SCN3A gain-of-function mutations cause neonatal focal seizures and early infantile epileptic encephalopathy. Epilepsia.",
    "Estacion M et al. (2010) — NaV1.3 sodium channels persistent current and frequency-dependent repetitive firing. J Neurophysiol.",
    "Bhatt DL et al. (2023) — Systematic review of SCN3A-related epilepsy. Epilepsia.",
    "Lux AL et al. (2004) — UK Infantile Spasms Study (UK-ISSS) ACTH vs vigabatrin. Lancet.",
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
        age_onset_m = random.randint(0, 36)
        etiology = etiol_pool[i]
        is_severe = "DEE67" in etiology or "LGS" in etiology or "Polymicrogyria" in etiology
        is_moderate = "Focal Epilepsy of Infancy" in etiology
        is_mild = "familial" in etiology.lower()
        is_phenocopy = "phenocopy" in etiology.lower()

        seizure_free = random.random() < (0.4 if is_mild else 0.05 if is_severe else 0.15 if is_moderate else 0.08)
        drug_resistant = random.random() < (0.1 if is_mild else 0.85 if is_severe else 0.55 if is_moderate else 0.7) and not seizure_free
        west_history = random.random() < (0.8 if is_severe else 0.1 if is_moderate else 0.02) and not is_mild
        on_acth = west_history and random.random() < 0.9
        on_kd = drug_resistant and random.random() < (0.5 if is_severe else 0.3)
        on_vpa = random.random() < (0.6 if is_severe else 0.45 if is_moderate else 0.7 if is_mild else 0.5)
        on_cbz_oxc = random.random() < (0.4 if is_moderate else 0.25 if is_mild else 0.1) and not west_history
        polg_tested = "Y" if random.random() < 0.78 else "N"
        vpa_without_polg = on_vpa and polg_tested == "N"
        pmg = "R357Q" in etiology or "Polymicrogyria" in etiology
        csws = random.random() < (0.35 if is_severe else 0.08)
        quinidine_trial = is_moderate and random.random() < 0.12
        hla_tested = on_cbz_oxc and random.random() < 0.72

        patients.append({
            "id": f"SCN3A-{i+1:03d}",
            "etiology": etiology,
            "onset_months": age_onset_m,
            "sex": random.choice(["M", "F"]),
            "seizure_free": seizure_free,
            "drug_resistant": drug_resistant,
            "west_history": west_history,
            "on_acth": on_acth,
            "on_kd": on_kd,
            "on_vpa": on_vpa,
            "on_cbz_oxc": on_cbz_oxc,
            "polg_tested": polg_tested,
            "vpa_without_polg": vpa_without_polg,
            "pmg": pmg,
            "csws": csws,
            "quinidine_trial": quinidine_trial,
            "hla_tested": hla_tested,
        })
    return patients


def get_overview():
    """Return SCN3A overview dict."""
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    west_history = sum(1 for p in pts if p["west_history"])
    on_acth = sum(1 for p in pts if p["on_acth"])
    on_kd = sum(1 for p in pts if p["on_kd"])
    on_vpa = sum(1 for p in pts if p["on_vpa"])
    on_cbz_oxc = sum(1 for p in pts if p["on_cbz_oxc"])
    polg_done = sum(1 for p in pts if p["polg_tested"] == "Y")
    vpa_without_polg = sum(1 for p in pts if p["vpa_without_polg"])
    pmg_n = sum(1 for p in pts if p["pmg"])
    csws_n = sum(1 for p in pts if p["csws"])
    quinidine_n = sum(1 for p in pts if p["quinidine_trial"])

    return {
        "gene": "SCN3A",
        "locus": "2q24.3",
        "inheritance": "De novo (dominant; GOF-DEE67/FEI) or autosomal dominant familial (GEFS+-like)",
        "protein": "NaV1.3 — Voltage-Gated Sodium Channel Alpha Subunit 3; fetal peak expression; fast repriming (τ ≈ 2-5 ms); DEE67",
        "mechanism": (
            "SCN3A GOF → NaV1.3 gain-of-function → enhanced persistent Na+ current (I_NaP) "
            "and fast repriming → sustained burst firing in neonatal/infantile cortex "
            "(where NaV1.3 is maximally expressed) → high-rate focal epilepsy (FEI) or "
            "DEE67 (West → LGS). R357Q: GOF during fetal NaV1.3 peak → bilateral perisylvian "
            "polymicrogyria (structural malformation) + DEE."
        ),
        "key_aha": (
            "SCN3A (2q24.3) — NaV1.3 fast-repriming channelopathy. "
            "High-rate neonatal focal seizures (FEI) = SCN3A hallmark. "
            "AVOID CBZ/OXC during infantile spasms (NaV1.1 PV-interneuron suppression). "
            "POLG MANDATORY before VPA. HLA-B*15:02 MANDATORY before CBZ/OXC (Asian ancestry). "
            "OXC-SIADH: Na+ monitoring mandatory. R357Q → 3T MRI (PMG). "
            "Quinidine investigational (persistent I_NaP target; QT monitoring mandatory). "
            "Tiagabine ABSOLUTE CI in DEE67."
        ),
        "n_patients": n,
        "seizure_free_pct": round(100 * seizure_free / n),
        "drug_resistant_pct": round(100 * drug_resistant / n),
        "west_history_pct": round(100 * west_history / n),
        "on_acth_pct": round(100 * on_acth / n),
        "on_kd_pct": round(100 * on_kd / n),
        "on_vpa_pct": round(100 * on_vpa / n),
        "on_cbz_oxc_pct": round(100 * on_cbz_oxc / n),
        "polg_done_pct": round(100 * polg_done / n),
        "vpa_without_polg": vpa_without_polg,
        "pmg_pct": round(100 * pmg_n / n),
        "csws_pct": round(100 * csws_n / n),
        "quinidine_trial_n": quinidine_n,
        "tiagabine_alert": "ABSOLUTE CI in ALL SCN3A-DEE67 and encephalopathy forms — GAT-1 block → extrasynaptic GABA → NCSE",
        "cbz_oxc_west_alert": "AVOID CBZ/OXC during infantile spasms/West phase — NaV1.1 PV-interneuron suppression → spasm aggravation",
        "hla_alert": "HLA-B*15:02 MANDATORY before CBZ/OXC in Asian ancestry (CPIC Level A — SJS/TEN risk)",
        "polg_alert": "POLG MANDATORY before VPA — Alpers-Huttenlocher fatal hepatic failure in POLG carriers",
        "r357q_alert": "SCN3A R357Q detected → 3T MRI MANDATORY (bilateral perisylvian polymicrogyria exclusion/confirmation)",
        "quinidine_alert": "Quinidine investigational (I_NaP blocker) — QTc monitoring MANDATORY; specialist centres only",
        "contraindications_summary": [
            "Tiagabine — ABSOLUTE CI: NCSE in SCN3A-DEE67 (GAT-1 → extrasynaptic GABA → tonic GABA-A block)",
            "CBZ/OXC — HIGH RISK during infantile spasms/West: NaV1.1 PV-interneuron suppression → spasm aggravation",
            "VPA without POLG screen — HIGH RISK: Alpers-Huttenlocher fatal hepatic failure",
            "LTG — MODERATE-HIGH RISK if myoclonic component: NaV1.1 block → myoclonic aggravation",
            "Quinidine without cardiac monitoring — HIGH RISK: QT prolongation, torsades de pointes",
        ],
        "thresholds": THRESHOLDS,
        "references": [r.split(" — ")[0] for r in REFERENCES],
    }


def get_breakdown():
    """Return SCN3A breakdown dict."""
    pts = _generate_patients()
    n = len(pts)

    # Etiology distribution
    from collections import Counter
    etiol_counts = Counter(p["etiology"] for p in pts)

    # Seizure type prevalence
    seizure_type_pcts = [
        {"type": st["type"], "pct": st["pct_patients"],
         "eeg": st["eeg_correlate"], "tip": st["clinical_tip"]}
        for st in SEIZURE_TYPES
    ]

    # Trigger prevalence
    trigger_pcts = [
        {"trigger": t["trigger"], "pct": t["pct"],
         "threshold": t["threshold"], "mgmt": t["management"]}
        for t in TRIGGERS
    ]

    # Sample patients (first 15)
    sample = []
    for p in pts[:15]:
        sample.append({
            "id": p["id"],
            "etiology_short": p["etiology"].split(" (")[0],
            "onset_months": p["onset_months"],
            "sex": p["sex"],
            "seizure_free": p["seizure_free"],
            "drug_resistant": p["drug_resistant"],
            "west": p["west_history"],
            "on_kd": p["on_kd"],
            "on_vpa": p["on_vpa"],
            "polg": p["polg_tested"],
            "vpa_no_polg": p["vpa_without_polg"],
            "pmg": p["pmg"],
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
        "seizure_types": seizure_type_pcts,
        "triggers": trigger_pcts,
        "treatments": [
            {"drug": t["drug"], "evidence": t["evidence"],
             "indication": t["indication"],
             "efficacy": t["efficacy"],
             "scn3a_note": t["scn3a_note"]}
            for t in TREATMENTS
        ],
        "contraindications": CONTRAINDICATIONS,
        "monitoring": [m for m in MONITORING[0]],  # unwrap tuple
        "lifecycle": LIFECYCLE,
        "sample_patients": sample,
    }


def get_definitions():
    """Return SCN3A definitions dict."""
    return {
        "gene": "SCN3A",
        "concepts": CONCEPTS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "monitoring_summary": [m["item"] for m in MONITORING[0]],
    }
