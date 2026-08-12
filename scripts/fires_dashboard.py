"""
FIRES (Febrile Infection-Related Epilepsy Syndrome) Dashboard
=============================================================
41-patient cohort · Rare devastating epileptic encephalopathy · Neuroinflammatory
FIRES: previously healthy children → febrile illness → super-refractory status epilepticus
→ chronic drug-resistant epilepsy with severe cognitive decline.
Key pathophysiology: IL-1β / IL-6 / TNF-α neuroinflammatory cascade.
Emerging first-line immunotherapy: Anakinra (IL-1 receptor antagonist).
Ketogenic diet: Level A evidence in chronic FIRES.
"""

import random
from datetime import datetime

SEED = 5555
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "Unknown neuroinflammatory (seronegative FIRES)",
        "n": 22, "pct": 54,
        "category": "Neuroinflammatory-Unknown",
        "mechanism": (
            "The dominant aetiology of FIRES remains unknown. Current evidence implicates an "
            "innate-immunity trigger: IL-1β, IL-6, IL-8, and TNF-α are markedly elevated in CSF "
            "during the acute phase. The initial febrile illness likely unmasks a pre-existing "
            "dysregulation of the NLRP3 inflammasome, leading to uncontrolled astrocyte and "
            "microglial activation. No adaptive (antibody-mediated) autoimmune target has been "
            "consistently identified in seronegative FIRES. Whole-brain neuroinflammation on "
            "18F-FDG PET imaging with bitemporal hypermetabolism during SRSE, then hypometabolism "
            "in the chronic phase, underscores the neuroinflammatory basis."
        ),
        "eeg_correlate": (
            "During SRSE: continuous ictal pattern, often with theta/alpha coma pattern; "
            "periodic lateralised epileptiform discharges (PLEDs/LPDs) over temporal regions; "
            "burst suppression in pharmacologically induced coma; electrical status epilepticus "
            "in sleep (ESES) in chronic phase. Background: severe generalised slowing."
        ),
        "mri_finding": (
            "Acute phase: T2/FLAIR hyperintensity in hippocampus, insula, basal ganglia, "
            "and thalamus — often bitemporal; DWI restriction in active seizure cortex. "
            "Chronic phase: severe bitemporal and hippocampal atrophy; generalised volume loss; "
            "post-SRSE cortical laminar necrosis in severe cases."
        ),
        "clinical_note": (
            "Seronegative status does NOT exclude FIRES. Antibody panel (NMDAR, LGI1, CASPR2, "
            "AMPAR, GABABR, GABAAR, DPPX, mGluR5, MOG) must be checked but negativity is "
            "expected in >50% of FIRES. Brain biopsy: perivascular CD8+ infiltrates, microglial "
            "activation — no diagnostic specificity. Diagnosis is clinical: SRSE after febrile "
            "prodrome in a previously healthy child, with negative infectious work-up."
        ),
    },
    {
        "etiology": "Autoimmune antibody-positive (NMDAR / LGI1 / CASPR2)",
        "n": 8, "pct": 20,
        "category": "Autoimmune-Ab-Positive",
        "mechanism": (
            "A clinically important minority (~20%) of FIRES cases are NORSE (New Onset Refractory "
            "Status Epilepticus) with identified antibody: anti-NMDA receptor (GluN1 subunit) in "
            "the most common, with LGI1 (leucine-rich glioma inactivated 1, voltage-gated K+ "
            "channel complex), CASPR2, AMPAR, and GABAAR identified in smaller numbers. "
            "NMDAR antibody internalisation reduces inhibitory NMDA-mediated currents on "
            "GABAergic interneurons, disinhibiting excitatory networks — the paradox of "
            "excitotoxicity via interneuron loss. LGI1 antibodies disrupt VGKC complex function."
        ),
        "eeg_correlate": (
            "Delta brush pattern (pathognomonic for anti-NMDAR encephalitis — but seen in <30% "
            "of cases); extreme delta brush on continuous EEG monitoring; severe background "
            "suppression alternating with ictal bursts; ESES in subacute recovery phase."
        ),
        "mri_finding": (
            "Anti-NMDAR: often initially normal; T2/FLAIR hippocampal changes in <30%; "
            "LGI1: T1 hyperintensity in basal ganglia (faciobrachial dystonic seizures); "
            "CASPR2: limbic signal changes, cerebellum; generalised atrophy in chronic phase."
        ),
        "clinical_note": (
            "Antibody-positive FIRES/NORSE has superior immunotherapy response vs. seronegative. "
            "Steroids + IVIG + plasma exchange as first-line; rituximab for NMDAR/LGI1 refractory. "
            "Ovarian teratoma must be excluded in anti-NMDAR encephalitis (transvaginal USS + MRI). "
            "Cyclophosphamide for refractory antibody-positive cases."
        ),
    },
    {
        "etiology": "Post-infectious (HSV / EBV / HHV6 / Influenza)",
        "n": 5, "pct": 12,
        "category": "Post-Infectious",
        "mechanism": (
            "Direct viral encephalitis (HSV-1 limbic encephalitis) or post-infectious immune "
            "dysregulation (EBV, HHV-6, influenza A H1N1/H3N2) can trigger FIRES-like SRSE. "
            "HSV-1 encephalitis causes temporal lobe destruction via direct neuronal lysis and "
            "inflammatory cytokine cascade. HHV-6 reactivation in immunocompromised or post-transplant "
            "patients may trigger FIRES. Post-influenza FIRES: neuroinflammation without CSF viral "
            "detection (immune-mediated, not infectious). PCR must be sent urgently — IV acyclovir "
            "empirically for all HSV-encephalitis-compatible presentations."
        ),
        "eeg_correlate": (
            "HSV encephalitis: periodic lateralised epileptiform discharges (PLEDs) over "
            "temporal region (2–3 Hz LPDs on updated ACNS terminology); temporal slowing; "
            "focal seizures evolving to SE. Influenza-associated: generalised spike-wave + PLEDS."
        ),
        "mri_finding": (
            "HSV: T2/FLAIR hyperintensity in medial temporal lobe (hippocampus, amygdala, insula, "
            "cingulate) — the 'limbic encephalitis' pattern; DWI restriction acutely; "
            "haemorrhagic transformation on T2*/SWI in severe cases. "
            "HHV-6: mesial temporal signal, particularly amygdala > hippocampus."
        ),
        "clinical_note": (
            "⚠️ IV acyclovir 10–15 mg/kg/dose q8h must be given empirically pending HSV PCR. "
            "Do NOT delay acyclovir for CSF result — neuronal destruction is time-dependent. "
            "CSF HSV PCR may be falsely negative in first 72 hours; repeat if strong suspicion."
        ),
    },
    {
        "etiology": "Cryptogenic structural / FCD-mimic",
        "n": 4, "pct": 10,
        "category": "Cryptogenic-Structural",
        "mechanism": (
            "A small subset of FIRES patients are found on high-resolution MRI post-SRSE to have "
            "focal cortical dysplasia (FCD IIa/IIb) or subtle malformation of cortical development "
            "(MCD) that was not visible on initial imaging due to poor signal-to-noise in the "
            "acute peri-ictal period. These cases represent a structural vulnerability that "
            "lowered the seizure threshold, allowing a febrile trigger to precipitate SRSE. "
            "Repeat 3T MRI with post-processing (VBM, cortical thickness mapping) 6–12 months "
            "post-SRSE may reveal the lesion. Surgical evaluation is warranted in this group."
        ),
        "eeg_correlate": (
            "Focal ictal onset (temporoparietal most common); secondary bilateral synchrony "
            "masking focal onset; post-SRSE: persistent focal slowing overlying the dysplastic zone."
        ),
        "mri_finding": (
            "Initial: no clear lesion (normal appearing); post-SRSE with brain volumetry: "
            "subtle cortical thickening, grey-white matter blurring, transmantle sign "
            "pathognomonic for FCD IIb; PET scan: focal hypometabolism identifies dysplastic zone."
        ),
        "clinical_note": (
            "Refer for epilepsy surgery evaluation after ≥12 months of chronic phase. "
            "Stereo-EEG (SEEG) or subdural grids for non-lesional cases. SEEG-guided "
            "resection provides Engel I in 40–50% of FCD-FIRES if lesion well defined."
        ),
    },
    {
        "etiology": "Metabolic / Genetic (POLG / SLC25A22 / GRIN variants)",
        "n": 2, "pct": 5,
        "category": "Genetic-Metabolic",
        "mechanism": (
            "Rare monogenic causes presenting as FIRES-like SRSE: POLG1 mutations cause "
            "mitochondrial DNA polymerase-gamma dysfunction → Alpers-Huttenlocher syndrome "
            "(progressive neuronal degeneration, SRSE, liver failure) — ABSOLUTE "
            "CONTRAINDICATION to valproate (fatal hepatotoxicity). SLC25A22 (mitochondrial "
            "glutamate carrier) loss-of-function causes early-onset epileptic encephalopathy. "
            "GRIN2A de novo variants (gain-of-function) produce NMDA receptor hyperactivation. "
            "Genetic FIRES mimics are identified on whole-exome sequencing (WES)."
        ),
        "eeg_correlate": (
            "POLG/Alpers: generalised poly-spike wave with posterior predominance; "
            "occipital seizures; SLC25A22: suppression-burst in neonatal/early infantile period. "
            "GRIN2A: focal and multifocal spikes, EPC-like pattern."
        ),
        "mri_finding": (
            "POLG/Alpers: T2/FLAIR in occipital cortex, thalamus, basal ganglia; "
            "cortical ribbon necrosis; progressive diffuse atrophy. "
            "SLC25A22: non-specific diffuse atrophy. GRIN variants: cortical dysplasia signal."
        ),
        "clinical_note": (
            "⛔ POLG mutation: Valproate ABSOLUTELY CONTRAINDICATED (fatal hepatic failure). "
            "Screen ALL FIRES patients with liver dysfunction/family history for POLG before VPA. "
            "WES should be performed in all FIRES cases without identified aetiology."
        ),
    },
]

# ── Seizure Types (4) ─────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal Motor / Multifocal Seizures",
        "freq_pct": 100,
        "duration_sec": "Variable (seconds to hours)",
        "description": (
            "Virtually all FIRES patients present with focal motor seizures in the acute phase — "
            "clonic jerking of one extremity or hemiface, often migrating (Jacksonian march). "
            "Multifocal onset is characteristic: seizures arise sequentially from different "
            "cortical foci reflecting widespread neuroinflammation. Focal motor seizures "
            "frequently cluster and evolve into super-refractory status epilepticus (SRSE) "
            "within 1–2 days of first seizure onset."
        ),
        "eeg_correlate": (
            "Focal ictal discharges (rhythmic 8–12 Hz sharp waves or beta activity) at seizure "
            "onset site; electro-clinical correlation with contralateral motor manifestations; "
            "secondary generalisation on EEG with bilateral synchrony; post-ictal focal suppression."
        ),
        "clinical_tip": (
            "Continuous video-EEG monitoring is MANDATORY in FIRES — clinical underestimation "
            "of seizure burden is common once patient is pharmacologically sedated. "
            "Electrographic-only seizures (no clinical correlate) may persist and cause ongoing "
            "neuronal injury. Burst-suppression pattern on EEG does not exclude ongoing seizures."
        ),
    },
    {
        "type": "Super-Refractory Status Epilepticus (SRSE)",
        "freq_pct": 100,
        "duration_sec": "Days to weeks (median 28 days in ICU)",
        "description": (
            "SRSE — status epilepticus continuing or recurring ≥24 hours after initiation of "
            "general anaesthesia — is the defining clinical event of FIRES. SRSE in FIRES is "
            "driven by ongoing neuroinflammation maintaining cortical hyperexcitability despite "
            "pharmacological suppression. Patients require ICU admission, endotracheal intubation, "
            "continuous EEG monitoring, and staged anaesthetic management (benzodiazepines → "
            "antiseizure medications → anaesthetic agents → ketamine/propofol/midazolam infusion). "
            "Median ICU stay 28 days; range 2–120 days."
        ),
        "eeg_correlate": (
            "Continuous ictal pattern initially; progression to burst-suppression under anaesthesia "
            "(target: burst suppression with inter-burst interval 5–10 seconds for electrical "
            "seizure suppression); periodic lateralised/generalised discharges (LPDs/GPDs) in "
            "pharmacological suppression; extreme delta brush pattern in antibody-positive cases."
        ),
        "clinical_tip": (
            "Staged SRSE protocol: (1) Lorazepam 0.1 mg/kg IV; (2) Levetiracetam 60 mg/kg IV "
            "+ VPA (if POLG excluded) + phenytoin/lacosamide; (3) Anaesthetic: midazolam "
            "infusion → propofol → ketamine; (4) Ketogenic diet IV emulsion; (5) Anakinra "
            "100 mg SC/day. Avoid prolonged propofol use (propofol infusion syndrome in children)."
        ),
    },
    {
        "type": "Tonic / Tonic-Clonic Seizures",
        "freq_pct": 75,
        "duration_sec": "30–120 seconds",
        "description": (
            "Generalised tonic-clonic seizures (GTCS) and tonic seizures occur in three-quarters "
            "of FIRES patients, typically representing secondary generalisation of focal onset "
            "activity or breakthrough seizures during pharmacological suppression weaning. "
            "Tonic seizures are of particular concern as they may cause hypoxia, aspiration, "
            "and haemodynamic instability in the ICU setting. They may be clinically "
            "subtle (minor posturing) when patient is pharmacologically obtunded."
        ),
        "eeg_correlate": (
            "GTCS: high-amplitude bilateral spike-wave → clonic phase with rhythmic spike-wave "
            "→ post-ictal generalised EEG suppression; Tonic: generalised low-amplitude fast "
            "activity (beta/gamma) with EMG artefact; often bilaterally synchronous from onset."
        ),
        "clinical_tip": (
            "All FIRES patients should be nursed with continuous SpO2, ETCO2, and video-EEG. "
            "Tonic seizures during anaesthetic wean herald SRSE recurrence — extend anaesthetic "
            "suppression and escalate immunotherapy before next wean attempt. "
            "Phenobarbital loading (20 mg/kg) useful as non-sedating add-on for tonic seizures."
        ),
    },
    {
        "type": "Chronic Focal / Absence-like Seizures (Post-SRSE)",
        "freq_pct": 90,
        "duration_sec": "5–60 seconds",
        "description": (
            "Following resolution of SRSE, >90% of FIRES survivors develop chronic drug-resistant "
            "epilepsy characterised by frequent focal seizures, brief tonic events, myoclonic "
            "jerks, and absence-like staring spells. These reflect permanent hippocampal and "
            "cortical scarring from prolonged SRSE. Seizure frequency in the chronic phase: "
            "median 5–20 focal seizures/day despite polypharmacy. Cognitive and behavioural "
            "sequelae (intellectual disability, autism-like features, psychiatric comorbidity) "
            "are present in >80% of survivors. The ketogenic diet is the single most effective "
            "chronic-phase intervention (50–60% responder rate)."
        ),
        "eeg_correlate": (
            "Post-SRSE chronic phase: persistent focal or multifocal epileptiform discharges "
            "over temporal regions; electrical status epilepticus in sleep (ESES) in up to 40%; "
            "background generalised slowing (theta-dominant) reflecting diffuse encephalopathy; "
            "multifocal spike-wave in wakefulness and NREM sleep."
        ),
        "clinical_tip": (
            "Ketogenic diet should be initiated during acute SRSE (IV KD emulsion) and continued "
            "as chronic-phase maintenance. Target blood BHB 2–4 mmol/L. Dietary ketosis reduces "
            "IL-1β levels (mechanism overlaps with anakinra). Neuropsychological assessment, "
            "rehabilitation, and ASD/ADHD screening are essential in all chronic FIRES survivors."
        ),
    },
]

# ── Seizure Triggers ──────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Febrile illness (mandatory FIRES criterion)",
        "pct": 100,
        "mechanism": (
            "Fever > 38°C triggers NLRP3 inflammasome activation in brain-resident microglia "
            "and astrocytes. IL-1β release → blood-brain barrier disruption → lymphocyte infiltration "
            "→ self-amplifying neuroinflammatory cascade. In FIRES, this innate immune trigger "
            "is pathologically amplified compared to febrile seizures in neurologically normal children."
        ),
        "management": "No preventable modification — FIRES criterion; vigilance for fever in chronic phase",
    },
    {
        "trigger": "Upper respiratory tract infection (URTI)",
        "pct": 65,
        "mechanism": (
            "Rhinovirus, coronavirus, parainfluenza most common URTI pathogens preceding FIRES. "
            "Systemic cytokine release (IL-6, IL-8, TNF-α) during URTI, combined with fever, "
            "provides the dual signal for NLRP3 activation and BBB compromise."
        ),
        "management": "Prompt antipyretics at first fever; early neurology review if seizures develop",
    },
    {
        "trigger": "Gastroenteritis / diarrhoeal illness",
        "pct": 30,
        "mechanism": (
            "Gut-brain axis: mucosal inflammation increases systemic LPS (lipopolysaccharide) "
            "from gram-negative bacteria → TLR4 activation → IL-1β release → microglial priming. "
            "Electrolyte disturbance (hyponatraemia from vomiting/diarrhoea) lowers seizure threshold."
        ),
        "management": "Maintain electrolyte balance; monitor serum sodium closely during acute illness",
    },
    {
        "trigger": "Psychological stress / prolonged sleep deprivation",
        "pct": 25,
        "mechanism": (
            "HPA axis activation elevates cortisol and CRH, which promote microglial IL-1β "
            "release. In FIRES survivors, stress-induced cytokine surges can trigger cluster "
            "seizures in the chronic phase even without fever — a stress-sensitised epileptic network."
        ),
        "management": "Stress reduction strategies; school accommodations; psychiatric support; benzodiazepine rescue plan",
    },
    {
        "trigger": "Vaccination (rare — 2–5%)",
        "pct": 5,
        "mechanism": (
            "Post-vaccination FIRES has been reported (influenza, COVID-19 mRNA) in temporal "
            "association. Mechanism: adjuvant-driven TLR stimulation → IL-1β release in a "
            "pre-susceptible neuroinflammatory milieu. Causality not established — association "
            "may reflect temporal coincidence with febrile prodrome. Risk-benefit of future "
            "vaccination should be discussed individually; not a contraindication per se."
        ),
        "management": "Document temporal relationship; neurologist input before future vaccines; pre-medicate with paracetamol",
    },
    {
        "trigger": "Missed immunotherapy / AED non-adherence (chronic phase)",
        "pct": 60,
        "mechanism": (
            "In established chronic FIRES, missed doses of anakinra, ketogenic diet breaks, "
            "or AED non-adherence are the most common causes of breakthrough SRSE recurrence "
            "(second FIRES episode). IL-1 pathway reactivation is rapid once anakinra is "
            "discontinued without slow taper."
        ),
        "management": "Strict adherence monitoring; never abrupt-stop anakinra; have rescue benzodiazepine plan",
    },
    {
        "trigger": "Sleep deprivation",
        "pct": 70,
        "mechanism": (
            "Sleep deprivation activates HPA and sympathetic axes, elevating IL-6 and TNF-α, "
            "which sensitise microglial IL-1β release. Post-SRSE FIRES survivors have structural "
            "insomnia due to hippocampal damage and ESES, creating a vicious cycle."
        ),
        "management": "Sleep hygiene; melatonin; ESES treatment with clobazam; school timetable adjustment",
    },
    {
        "trigger": "Fever / intercurrent infection (chronic phase relapse)",
        "pct": 80,
        "mechanism": (
            "In chronic FIRES, any new febrile illness reactivates the sensitised "
            "neuroinflammatory pathway with risk of second SRSE episode. Fever management "
            "is particularly important — maintain core temperature < 37.5°C with antipyretics "
            "and physical cooling during any intercurrent illness."
        ),
        "management": "Antipyretics at 37.5°C; sick-day anakinra dose escalation; written emergency care plan",
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Anakinra (IL-1 receptor antagonist)",
        "evidence": "Level B",
        "evidence_ref": "Kenney-Jung 2016 Ann Neurol; Duis 2018 Ann Neurol; Dilena 2019 Epilepsia",
        "dose_adult": "100 mg SC daily; escalate to 200–300 mg/day in SRSE; IV formulation not licensed but used off-label 2–8 mg/kg/day",
        "dose_paed": "2–4 mg/kg/day SC (max 8 mg/kg/day in SRSE); IV continuous infusion 2–4 mg/kg/day in ICU",
        "moa": (
            "Anakinra is a recombinant human IL-1 receptor antagonist (IL-1Ra) that competitively "
            "binds the IL-1 receptor type I (IL-1RI), blocking both IL-1α and IL-1β signalling. "
            "In FIRES, CSF IL-1β is massively elevated; anakinra reduces microglial activation, "
            "restores BBB integrity, decreases neuronal hyperexcitability via IL-1RI on "
            "hippocampal and cortical neurons. Ketogenic diet independently reduces IL-1β (hydroxyl "
            "butyrate inhibits NLRP3 inflammasome) — mechanistic synergy with anakinra."
        ),
        "efficacy": (
            "Case series (n=12 FIRES): anakinra led to SRSE resolution in 67% within 7–14 days. "
            "Paediatric FIRES treated with anakinra had shorter SRSE duration (median 28 vs. 85 days) "
            "vs. historical controls. Chronic phase: anakinra maintenance reduced seizure frequency "
            "by >50% in 55% of patients. Most dramatic responders: CSF IL-1β >50 pg/mL."
        ),
        "safety": (
            "Injection site reactions (erythema, induration) in 70% of SC injections — pre-warm "
            "syringe, rotate sites. Neutropenia: monitor CBC monthly. Opportunistic infections: "
            "screen for latent TB (IGRA) and hepatitis B before starting. Live vaccines: "
            "contraindicated during treatment. No QT prolongation or hepatotoxicity."
        ),
        "monitoring": (
            "CBC with differential (baseline, month 1, then quarterly); CRP/ESR (IL-1 suppression "
            "marker); CSF IL-1β if LP feasible (target <10 pg/mL on treatment); injection site "
            "rotation log; chest X-ray/IGRA at baseline; liver function quarterly."
        ),
        "contraindication_note": None,
    },
    {
        "drug": "Ketogenic Diet (KD) — IV + oral",
        "evidence": "Level B (acute SRSE); Level A (chronic FIRES)",
        "evidence_ref": "Nabbout 2010 Epilepsia; Lin 2015 Epilepsia; Caraballo 2020 Seizure",
        "dose_adult": "4:1 ketogenic ratio (fat:protein+carbs); initiate at 1:1 then escalate over 48h; target BHB 2–4 mmol/L",
        "dose_paed": "IV KD emulsion (Intralipid 30%) 1–2 g/kg/day fat equivalent in ICU; oral KD 4:1 when extubated",
        "moa": (
            "Beta-hydroxybutyrate (BHB) directly inhibits the NLRP3 inflammasome, reducing IL-1β "
            "maturation and release — mechanistically overlapping with anakinra. "
            "Additional antiseizure mechanisms: enhanced GABA synthesis (glutamate → GABA via "
            "GAD), reduced glutamate excitotoxicity, mitochondrial neuroprotection, "
            "HCN channel modulation reducing hyperpolarisation-activated currents. "
            "KD reduces FIRES SRSE duration by anti-inflammatory + antiseizure dual action."
        ),
        "efficacy": (
            "SRSE: KD initiation in acute FIRES associated with SRSE resolution in 50–60% within "
            "14 days (Nabbout 2010 series, n=10). Chronic FIRES: 60% responder rate (>50% "
            "seizure frequency reduction); 20% complete freedom in highly adherent patients. "
            "Most effective chronic intervention in FIRES — superior to any single AED."
        ),
        "safety": (
            "Hypoglycaemia at initiation (especially IV KD) — monitor glucose q4h. "
            "Hyperlipidaemia: monitor fasting lipid panel quarterly. Kidney stones: 10% "
            "risk — ensure adequate hydration (>1 L/m2/day), potassium citrate supplement. "
            "Bone density: DEXA annually (KD reduces mineralisation). Growth restriction "
            "in children if protein inadequate — dietitian review every 3 months. "
            "GI intolerance: nausea, constipation — MCT oil may improve tolerability."
        ),
        "monitoring": (
            "Blood BHB daily (target 2–4 mmol/L); glucose q4h (ICU) or daily (chronic); "
            "fasting lipids quarterly; urine calcium:creatinine ratio quarterly; DEXA annually; "
            "selenium, carnitine, zinc, vitamins D/E/A every 6 months; dietitian follow-up 3-monthly."
        ),
        "contraindication_note": "POLG mutation: avoid aggressive dietary restriction (metabolic crisis); fatty acid oxidation disorders absolute CI",
    },
    {
        "drug": "IVIG (Intravenous Immunoglobulin)",
        "evidence": "Level C",
        "evidence_ref": "Caraballo 2013 Seizure; case series; FIRES consensus 2019",
        "dose_adult": "2 g/kg total over 2–5 days (0.4 g/kg/day × 5 or 1 g/kg/day × 2); repeat monthly if response",
        "dose_paed": "2 g/kg over 2–5 days; repeat every 3–4 weeks in responders",
        "moa": (
            "IVIG provides broadly immunomodulatory effects via: Fc receptor blockade on macrophages "
            "and microglia reducing phagocytic activation; anti-idiotypic antibodies neutralising "
            "pathogenic autoantibodies (critical in antibody-positive FIRES/NORSE); "
            "complement inhibition; regulatory T-cell expansion; IL-1 and TNF-α reduction. "
            "In seronegative FIRES, the mechanism is less clear — modulation of innate immune "
            "cascades rather than specific antibody neutralisation."
        ),
        "efficacy": (
            "Retrospective series: IVIG used in 60–70% of FIRES cases as empiric immunotherapy "
            "in first week. Objective SRSE shortening demonstrated in antibody-positive FIRES "
            "(especially anti-NMDAR). In seronegative FIRES, controlled data absent — "
            "used universally as low-risk empiric therapy. Some centres combine IVIG + "
            "plasma exchange in the first 2 weeks for all FIRES."
        ),
        "safety": (
            "Headache (aseptic meningitis syndrome): pre-medicate with paracetamol, reduce infusion rate. "
            "Thromboembolism risk (IgA-deficient, obese patients): "
            "screen IgA level before; hyperosmolar preparations increase thrombotic risk. "
            "Haemolysis: CBC 5–7 days post-infusion. Renal failure: rare with sucrose-containing "
            "preparations (use sucrose-free in renal impairment). Anaphylaxis: rare (IgA deficiency)."
        ),
        "monitoring": (
            "IgA level before first dose (anaphylaxis risk if deficient); renal function and "
            "urine output post-infusion; CBC at day 5–7 (haemolysis screen); neurological "
            "response assessment 7 days post-infusion; serum IgG level 4 weeks post-dose."
        ),
        "contraindication_note": "Selective IgA deficiency: risk of anaphylaxis — use IgA-depleted preparation",
    },
    {
        "drug": "Plasma Exchange (PLEX / Plasmapheresis)",
        "evidence": "Level C",
        "evidence_ref": "FIRES consensus 2019; Specchio 2020 Front Neurol",
        "dose_adult": "5–7 exchanges over 10 days; 1–1.5 plasma volumes per exchange; albumin replacement",
        "dose_paed": "5–7 exchanges; 40–50 mL/kg per exchange (replacement with 5% albumin or FFP in small children)",
        "moa": (
            "PLEX removes circulating autoantibodies (essential in antibody-positive FIRES), "
            "complement factors, cytokines (IL-1β, IL-6, TNF-α), and inflammatory mediators "
            "from plasma. In seronegative FIRES, removes unidentified inflammatory proteins "
            "and lipid-soluble neuroinflammatory mediators. PLEX is most effective early "
            "in the disease course before irreversible neuronal loss. Often combined with "
            "IVIG (sequential: PLEX then IVIG) for synergistic immunomodulation."
        ),
        "efficacy": (
            "FIRES antibody-positive: PLEX is standard of care with 60–70% SRSE improvement "
            "when combined with IVIG ± steroids. Seronegative FIRES: retrospective evidence "
            "of shortened SRSE in 40% of cases. Consensus recommendation: PLEX offered to "
            "all FIRES patients failing first-line AED and steroid therapy within 5 days."
        ),
        "safety": (
            "Central venous catheter (CVC) access complications (infection, thrombosis, pneumothorax). "
            "Hypocalcaemia (citrate anticoagulant chelates calcium) — supplement IV calcium. "
            "Hypotension during exchange — slow exchange rate. Coagulopathy: PT/APTT prolonged "
            "post-PLEX (removes clotting factors) — use FFP replacement in bleeding-risk patients. "
            "Hypothermia from albumin infusion — warm replacement fluid."
        ),
        "monitoring": (
            "Ionised calcium every 2 hours during exchange (hypocalcaemia monitoring); "
            "CBC, coagulation screen (PT/APTT/fibrinogen) before each session; "
            "CVC site inspection daily; blood culture if fever during PLEX; "
            "neurological status, EEG response assessment after 3rd and 5th sessions."
        ),
        "contraindication_note": "Active sepsis/bacteraemia: delay PLEX until treated (line sepsis risk compounded)",
    },
    {
        "drug": "IV Methylprednisolone (Steroids — pulse)",
        "evidence": "Level C",
        "evidence_ref": "FIRES consensus 2019; Caputo 2018 Eur J Paediatr Neurol",
        "dose_adult": "1 g IV/day × 5 days (pulse); then taper oral prednisolone 1–2 mg/kg/day over 8 weeks",
        "dose_paed": "20–30 mg/kg/day IV (max 1 g) × 3–5 days; oral prednisolone 1–2 mg/kg/day taper",
        "moa": (
            "Glucocorticoids suppress NF-κB mediated transcription of pro-inflammatory cytokines "
            "(IL-1β, IL-6, TNF-α, COX-2). IV methylprednisolone provides rapid suppression of "
            "microglial activation and reduces BBB permeability, allowing resolution of peri-ictal "
            "oedema. However, steroids have immunosuppressive effects that increase infection risk "
            "in already immunocompromised ICU patients, and systemic side effects (hyperglycaemia, "
            "which worsens seizures by lowering seizure threshold)."
        ),
        "efficacy": (
            "Used universally in FIRES within first 5 days in most centres. Retrospective series: "
            "faster EEG background improvement in patients receiving early steroids vs. late/none. "
            "No controlled trial data. Consensus: steroids as first-line empiric immunotherapy "
            "alongside IVIG in all FIRES — the risk:benefit favours use in SRSE."
        ),
        "safety": (
            "Hyperglycaemia (worsens neuronal injury — target glucose 4–10 mmol/L with insulin). "
            "Hypertension. Infection risk (Pneumocystis jirovecii prophylaxis: cotrimoxazole "
            "if >4 weeks immunosuppression). Electrolyte disturbance: hypokalaemia, hyponatraemia. "
            "GI bleeding: PPI prophylaxis. Adrenal suppression: never abrupt stop; taper always."
        ),
        "monitoring": (
            "Glucose q4h (ICU) or daily (ward); BP twice daily; electrolytes daily; "
            "chest X-ray weekly (PCP pneumonitis); varicella/VZV serology before (VZV prophylaxis "
            "if seronegative on steroids); ACTH stimulation test before taper completion."
        ),
        "contraindication_note": "Sepsis: relative contraindication — treat infection first; balance risk of FIRES vs. infection",
    },
    {
        "drug": "Tocilizumab (IL-6 receptor antagonist)",
        "evidence": "Level C",
        "evidence_ref": "Kenney-Jung 2016 case; Caraballo 2020; de Oliveira 2022 Epilepsia",
        "dose_adult": "8 mg/kg IV over 1 hour (max 800 mg per dose); repeat at day 14 if partial response",
        "dose_paed": "12 mg/kg IV (<30 kg) or 8 mg/kg IV (≥30 kg); 2–4 doses, 2-weekly",
        "moa": (
            "Tocilizumab is a humanised anti-IL-6 receptor monoclonal antibody (both membrane-bound "
            "mIL-6R and soluble sIL-6R blockade). IL-6 is massively elevated in FIRES CSF "
            "(>100 pg/mL in acute phase). IL-6 drives neuronal hyperexcitability via STAT3 "
            "phosphorylation in astrocytes and directly enhances NMDA receptor-mediated calcium "
            "influx. Tocilizumab rapidly reduces CRP (within 24–48 hours — a surrogate for "
            "IL-6 pathway suppression) and may reduce SRSE duration in IL-6-predominant FIRES."
        ),
        "efficacy": (
            "Case reports and small series (n=8 FIRES): tocilizumab reduced SRSE duration by "
            "50% in 5/8 cases when added after anakinra + steroids + IVIG failure. "
            "Rapid CRP normalisation (24–48h) is a surrogate of response. Escalation therapy "
            "after 2 weeks of SRSE with anakinra failure. Most promising in high IL-6 CSF "
            "(>50 pg/mL) FIRES."
        ),
        "safety": (
            "Infections: increased risk of bacterial, fungal, and opportunistic infections — "
            "screen TB (IGRA), hepatitis B/C, HIV before initiation. Neutropenia, thrombocytopenia. "
            "Elevated liver transaminases (monitor LFTs). GI perforation risk in patients on "
            "concurrent NSAIDs or steroids (rare). Hypersensitivity reaction during infusion. "
            "Live vaccines: contraindicated during and for 3 months after treatment."
        ),
        "monitoring": (
            "CBC with differential (baseline, week 2, week 6); LFTs (baseline, monthly); "
            "lipid panel (tocilizumab causes hyperlipidaemia — fasting lipids at 12 weeks); "
            "CRP as treatment response biomarker (target undetectable); CSF IL-6 if LP feasible."
        ),
        "contraindication_note": "Active serious infection: absolute CI — treat before initiating; use with caution in recurrent infections",
    },
    {
        "drug": "Ketamine (NMDA antagonist — anaesthetic adjunct)",
        "evidence": "Level C (SRSE)",
        "evidence_ref": "Gaspard 2013 Epilepsia; Rosati 2012 Neurocrit Care; FIRES consensus 2019",
        "dose_adult": "1–4 mg/kg IV bolus; continuous infusion 0.5–5 mg/kg/hour; titrate to burst-suppression",
        "dose_paed": "1–2 mg/kg bolus; 0.5–3 mg/kg/hour infusion; monitor for emergence phenomena",
        "moa": (
            "Ketamine is an NMDA receptor channel blocker (use-dependent, non-competitive "
            "antagonist of the Mg2+ binding site). In SRSE, sustained NMDA receptor activation "
            "drives a self-reinforcing excitotoxic loop — ketamine breaks this cycle. "
            "Additional anti-inflammatory properties: reduces microglial NLRP3 activation and "
            "IL-1β release at sedative doses. The catecholaminergic stimulation of ketamine "
            "counteracts propofol-induced cardiovascular depression — particularly valuable "
            "in haemodynamically unstable ICU patients. KETAMINE-FIRES COMBINATION: avoid "
            "concurrent propofol at high doses (synergistic hypotension and PRIS risk)."
        ),
        "efficacy": (
            "Retrospective ICU series (n=21 refractory SE): ketamine as add-on to midazolam "
            "infusion terminated SRSE in 57% within 24 hours. Neonatal/paediatric FIRES: "
            "most dramatic responses in cases with burst suppression already achieved on "
            "other agents — ketamine deepened suppression and enabled anaesthetic wean. "
            "ILAE 2022 Stage 4 SE protocol includes ketamine as recommended infusion."
        ),
        "safety": (
            "Haemodynamic stimulation (tachycardia, hypertension) — monitor HR/BP continuously; "
            "avoid in raised ICP (theoretical concern — clinical evidence weak at anaesthetic doses). "
            "Emergence phenomena (delirium, hallucinations) on waking — benzodiazepine co-sedation. "
            "Bronchospasm: anticholinergic pre-treatment (glycopyrrolate) in airways-sensitive patients. "
            "Prolonged use (>5 days): urinary tract toxicity — monitor urinalysis."
        ),
        "monitoring": (
            "Continuous EEG (burst-suppression titration); invasive arterial BP monitoring; "
            "heart rate (target <110 bpm to avoid myocardial O2 demand); urinalysis weekly "
            "in prolonged infusion; emergence protocol documented (benzodiazepine ready at wean)."
        ),
        "contraindication_note": "Severe hypertension (systolic >160): use with caution; consider propofol (if PRIS risk excluded) as alternative",
    },
    {
        "drug": "Rituximab (Anti-CD20 B-cell depletion)",
        "evidence": "Level B (antibody-positive FIRES); Level C (seronegative)",
        "evidence_ref": "Titulaer 2013 Lancet Neurol (NMDAR); FIRES consensus 2019",
        "dose_adult": "375 mg/m2 IV weekly × 4 doses; or 1000 mg × 2 doses (2 weeks apart)",
        "dose_paed": "375 mg/m2 IV × 4 weekly doses; premedicate: paracetamol, diphenhydramine, methylprednisolone",
        "moa": (
            "Rituximab is a chimeric anti-CD20 monoclonal antibody causing rapid depletion of "
            "CD20+ B-lymphocytes (mature B cells and pre-B cells, sparing plasma cells and "
            "stem cells). In antibody-positive FIRES (NMDAR, LGI1, CASPR2), rituximab depletes "
            "the B-cell clones producing pathogenic autoantibodies, providing sustained "
            "immunomodulation lasting 6–12 months per course. In seronegative FIRES, rituximab "
            "likely modulates antigen-presenting cell function and cytokine production (B cells "
            "secrete IL-6 and TNF-α). B-cell reconstitution occurs 6–12 months post-infusion."
        ),
        "efficacy": (
            "Anti-NMDAR encephalitis (n=577, Titulaer 2013): rituximab as second-line after "
            "steroids/IVIG improved 6-month outcome in 61% vs. 48% with first-line alone. "
            "FIRES seronegative: smaller series — chronic-phase seizure reduction in 40–50%. "
            "Recommend in all antibody-positive FIRES failing first-line; consider in seronegative "
            "after ≥4 weeks SRSE with no response to anakinra + steroids + IVIG + PLEX."
        ),
        "safety": (
            "Infusion reactions: pre-medicate with paracetamol, antihistamine, and IV methylprednisolone. "
            "Progressive multifocal leukoencephalopathy (PML): rare — screen JC antibody index; "
            "risk very low with ≤4 doses in FIRES. Hepatitis B reactivation: MANDATORY HBsAg + "
            "anti-HBc screening; prophylactic entecavir if HBc-positive. Prolonged B-cell aplasia: "
            "hypogammaglobulinaemia → monitor IgG quarterly; IVIG replacement if IgG < 5 g/L."
        ),
        "monitoring": (
            "CD19/20 B-cell count (target <0.05 × 10⁹/L for adequate depletion) at 2 and 4 weeks "
            "post-infusion; IgG/IgA/IgM quarterly; HBsAg + HBV DNA monthly (reactivation watch); "
            "full blood count monthly; JC virus antibody index before 2nd and subsequent courses."
        ),
        "contraindication_note": "HBsAg positive: prophylactic entecavir MANDATORY before first dose; active hepatitis: absolute CI",
    },
]

# ── Monitoring Protocol ───────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "Continuous Video-EEG Monitoring (cEEG)",
        "frequency": "Continuously throughout SRSE; daily reviews in chronic phase",
        "rationale": (
            "SRSE in FIRES has high rates of non-convulsive SE (NCSE) — electrographic seizures "
            "without clinical manifestation. cEEG allows quantification of seizure burden, "
            "titration of anaesthetic depth to burst-suppression, detection of electrographic-only "
            "NCSE during anaesthetic wean, and identification of ESES in chronic phase. "
            "Quantitative EEG (aEEG/CEEG) supports bedside monitoring by nursing staff."
        ),
    },
    {
        "item": "CSF IL-1β / IL-6 Cytokine Panel",
        "frequency": "At diagnosis (LP1); repeat at 2 weeks if anakinra started",
        "rationale": (
            "CSF IL-1β >50 pg/mL identifies high-probability anakinra responders. "
            "IL-6 elevation (>100 pg/mL) supports tocilizumab escalation. "
            "Repeat CSF cytokine measurement 2 weeks post-anakinra confirms treatment response "
            "(target IL-1β <10 pg/mL). Full antibody panel must accompany cytokine testing."
        ),
    },
    {
        "item": "MRI Brain (serial)",
        "frequency": "Acute: MRI at diagnosis + day 14 + 3 months; Chronic: annual",
        "rationale": (
            "Acute-phase MRI detects limbic signal changes (hippocampus/insula), quantifies "
            "acute neuroinflammation, excludes structural cause (FCD, tumour). "
            "Day-14 MRI: assess progression (BBB disruption → atrophy). "
            "3-month MRI: hippocampal atrophy quantification — correlates with cognitive outcome. "
            "Annual MRI in chronic phase: detect ongoing atrophy; post-processing for FCD detection "
            "in surgical candidates."
        ),
    },
    {
        "item": "Neuropsychological Assessment",
        "frequency": "Baseline (if possible pre-SRSE); 6 months post-discharge; annually",
        "rationale": (
            "Cognitive sequelae affect >80% of FIRES survivors: memory impairment, executive "
            "dysfunction, intellectual disability (IQ <70 in 40%), ASD features (30%), "
            "ADHD (50%), psychiatric disorders (anxiety, PTSD, depression). "
            "Baseline pre-illness cognitive level guides rehabilitation targets. "
            "Annual assessment tracks recovery trajectory and guides school/occupational adjustments."
        ),
    },
    {
        "item": "Blood BHB (Ketogenic Diet monitoring)",
        "frequency": "Daily in acute KD phase; weekly in stable chronic KD",
        "rationale": (
            "Target blood beta-hydroxybutyrate 2–4 mmol/L for anti-inflammatory and antiseizure "
            "efficacy. BHB <1 mmol/L = inadequate ketosis (diet adjustment needed). "
            "BHB >5 mmol/L = excessive ketosis risk (acidosis, hypoglycaemia in children)."
        ),
    },
    {
        "item": "Hepatitis B / Hepatitis C / HIV / IGRA Screening",
        "frequency": "At FIRES diagnosis before immunotherapy; before rituximab",
        "rationale": (
            "All immunotherapy in FIRES (steroids, IVIG, PLEX, rituximab, tocilizumab, anakinra) "
            "carry reactivation risk for latent infections. HBsAg + anti-HBc mandatory before "
            "rituximab (reactivation risk 3–5%). IGRA for TB (steroids/rituximab). "
            "HIV serology (immunosuppression contraindicated without treatment)."
        ),
    },
]

# ── Lifecycle Windows ─────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Febrile Prodrome",
        "age_range": "2–7 days before SRSE",
        "key_events": "Fever > 38°C; URTI or gastroenteritis symptoms; no neurological deficits",
        "focus": "Recognition missed at this stage — FIRES mimics febrile seizures; alert parents to call emergency services immediately if seizure begins",
    },
    {
        "window": "Acute SRSE Phase",
        "age_range": "Days 1–28 (median ICU admission 28 days, range 2–120)",
        "key_events": "Seizure onset → SE → SRSE within 24–72 hours; ICU admission; intubation; cEEG monitoring; sequential immunotherapy; ketogenic diet initiation",
        "focus": "Staged SE protocol execution; burst-suppression targeting; parallel immunotherapy (steroids + IVIG + PLEX + anakinra); POLG exclusion before VPA; anakinra escalation at day 5 if no response",
    },
    {
        "window": "Subacute Transition",
        "age_range": "Weeks 4–12",
        "key_events": "SRSE resolution; anaesthetic wean; extubation; seizure recurrence during wean; neurorehabilitation begins; chronic AED rationalisation",
        "focus": "Slow anaesthetic taper (1 week minimum per wean step); anakinra maintenance; oral KD transition; neuropsychological testing; family education on chronic FIRES management",
    },
    {
        "window": "Chronic Epilepsy Phase",
        "age_range": "3 months – years post-SRSE",
        "key_events": "Frequent focal seizures; cognitive regression; ASD/ADHD features; KD chronic management; school re-entry; psychiatric comorbidity management",
        "focus": "KD strict adherence (most effective); anakinra SC maintenance; rationalise AED polypharmacy; school accommodation; neuropsychology; cognitive rehabilitation; seizure rescue plan",
    },
    {
        "window": "Surgical Evaluation Window",
        "age_range": "12–24 months post-SRSE",
        "key_events": "Repeat MRI + PET; SEEG evaluation if focal; resection / VNS / corpus callosotomy decision",
        "focus": "All FIRES patients with drug-resistant chronic epilepsy should be referred to epilepsy surgery centre. FCD-FIRES subgroup: 40–50% Engel I with resection. VNS: 30–40% responder rate.",
    },
    {
        "window": "Adult Transition",
        "age_range": "Age 18+ (FIRES onset typically age 4–14)",
        "key_events": "Transition to adult neurology; driving assessment; independence assessment; vocational rehabilitation; continued KD (adult KD centre)",
        "focus": "Most FIRES survivors require lifelong AED and KD; 40% require support housing/care; SUDEP risk counselling mandatory; specialist adult epilepsy centre with FIRES experience required",
    },
]

# ── Clinical Standards ────────────────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE 2022 Classification", "relevance": "FIRES classified as immune epilepsy — unknown aetiology subgroup"},
    {"standard": "NICE NG217 (Epilepsies: diagnosis and management, 2022)", "relevance": "Paediatric SE management; specialist referral criteria"},
    {"standard": "Hirsch et al. 2021 — ILAE SRSE Definition", "relevance": "Super-refractory SE criteria (≥24h on general anaesthesia); staged management"},
    {"standard": "FIRES Consensus Statement 2019 (Epilepsy Currents)", "relevance": "Multi-centre consensus: immunotherapy sequencing, KD, anakinra; diagnosis criteria"},
    {"standard": "FDA Kineret (Anakinra) REMS Program", "relevance": "Injection site monitoring, infection screening; off-label FIRES use documentation"},
    {"standard": "ACNS EEG Terminology 2021 (LPD/GPD/LRDA)", "relevance": "Standardised EEG reporting in ICU-based SRSE; burst-suppression criteria"},
]

# ── Decision Thresholds ───────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "SRSE definition", "value": "SE persisting ≥24h on general anaesthesia; or recurrence on wean"},
    {"threshold": "Anakinra dose escalation trigger", "value": "No SRSE improvement at day 5 → escalate to 4–8 mg/kg/day IV"},
    {"threshold": "Ketogenic diet BHB target", "value": "2–4 mmol/L blood BHB for anti-inflammatory + antiseizure efficacy"},
    {"threshold": "Burst suppression target (cEEG)", "value": "Inter-burst interval 5–10 seconds at peak anaesthetic depth"},
    {"threshold": "IVIG dose", "value": "2 g/kg total over 2–5 days; repeat every 3–4 weeks in responders"},
    {"threshold": "Rituximab indication (antibody+)", "value": "Anti-NMDAR/LGI1 failing steroids + IVIG within 10 days"},
    {"threshold": "Driving (chronic FIRES)", "value": "12 months seizure-free (jurisdiction-dependent); cognitive assessment mandatory"},
    {"threshold": "POLG exclusion before VPA", "value": "MANDATORY in all FIRES with liver dysfunction / family history — fatal hepatotoxicity"},
]

# ── Key Thresholds (overview summary) ────────────────────────────────────────
KEY_THRESHOLDS = {
    "SRSE_definition": "SE ≥24h on GA or recurrence at wean",
    "anakinra_escalation_trigger": "No response at day 5 → 4–8 mg/kg/day IV",
    "KD_BHB_target": "2–4 mmol/L blood BHB",
    "burst_suppression_target": "IBI 5–10 seconds on cEEG",
    "POLG_VPA_exclusion": "MANDATORY before VPA in all FIRES",
    "driving_restriction": "12 months seizure-free",
    "rituximab_NMDAR": "Failing steroids+IVIG at day 10 (Ab+)",
    "second_FIRES_risk": "60–70% risk of chronic drug-resistant epilepsy post-SRSE",
}

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "van Baalen A et al. (2010). Febrile infection-related epilepsy syndrome (FIRES): a nonspecific term and a challenge. Epilepsia 51(11):2218-2219.",
    "Nabbout R et al. (2010). Outcome of childhood-onset FIRES with anakinra as first treatment. Epilepsia 51(10):2078-2085.",
    "Caraballo RH et al. (2013). FIRES: an entity with heterogeneous aetiology and poor prognosis. Epileptic Disord 15(4):421-431.",
    "Kenney-Jung DL et al. (2016). Febrile infection-related epilepsy syndrome treated with anakinra. Ann Neurol 80(6):939-945.",
    "Specchio N et al. (2020). FIRES: pathogenesis, treatment and outcome. Front Neurol 11:1180.",
    "Caputo D et al. (2018). Anakinra and ketogenic diet in FIRES: a case series. Eur J Paediatr Neurol 22(2):312-316.",
]


def _generate_patients(n=41):
    """Generate synthetic FIRES patient cohort (n=41)."""
    random.seed(SEED)
    etiology_pool = []
    for ec in ETIOLOGY_CATALOG:
        etiology_pool.extend([ec["etiology"]] * ec["n"])
    random.shuffle(etiology_pool)

    seizure_type_pool = [
        "Focal motor → SRSE", "Multifocal → SRSE", "Focal motor/SRSE", "Tonic-clonic/SRSE",
    ]
    treatment_pool = [
        "Anakinra+KD", "Anakinra+KD+IVIG", "KD+IVIG+PLEX", "Steroids+IVIG+Anakinra",
        "IVIG+PLEX+Steroids", "Anakinra+Rituximab+KD", "KD+Ketamine+IVIG",
        "Tocilizumab+Anakinra+KD",
    ]
    control_pool = ["Drug-resistant (chronic)", "Partial control (KD)", "Seizure-free (KD+Anakinra)"]
    control_weights = [0.55, 0.30, 0.15]
    phase_pool = ["Acute SRSE", "Subacute Recovery", "Chronic Epilepsy", "Surgical Evaluation", "Long-term Maintenance"]
    phase_weights = [0.05, 0.10, 0.60, 0.10, 0.15]

    patients = []
    for i in range(n):
        age_onset = random.randint(3, 14)
        age_current = age_onset + random.randint(1, 8)
        sex = "M" if random.random() < 0.55 else "F"
        etiology = etiology_pool[i]
        seizure_type = random.choice(seizure_type_pool)
        treatment = random.choice(treatment_pool)

        r = random.random()
        cumulative = 0
        control = control_pool[-1]
        for cp, cw in zip(control_pool, control_weights):
            cumulative += cw
            if r < cumulative:
                control = cp
                break

        r2 = random.random()
        cumulative = 0
        phase = phase_pool[-1]
        for pp, pw in zip(phase_pool, phase_weights):
            cumulative += pw
            if r2 < cumulative:
                phase = pp
                break

        srse_duration_days = random.randint(7, 90)
        icu_days = srse_duration_days + random.randint(2, 14)

        patients.append({
            "id": f"FIRES-{i+1:03d}",
            "age": age_current,
            "sex": sex,
            "onset_age": age_onset,
            "etiology": etiology.split("(")[0].strip(),
            "seizure_type": seizure_type,
            "current_treatment": treatment,
            "seizure_control": control,
            "disease_phase": phase,
            "srse_duration_days": srse_duration_days,
            "icu_days": icu_days,
            "kd_on_kd": "KD" in treatment,
            "anakinra": "Anakinra" in treatment,
        })
    return patients


# ── Public API ────────────────────────────────────────────────────────────────

def overview():
    patients = _generate_patients()
    total = len(patients)
    male_n = sum(1 for p in patients if p["sex"] == "M")
    drug_resistant_n = sum(1 for p in patients if "Drug-resistant" in p["seizure_control"])
    partial_n = sum(1 for p in patients if "Partial" in p["seizure_control"])
    seizure_free_n = sum(1 for p in patients if "Seizure-free" in p["seizure_control"])
    on_kd_n = sum(1 for p in patients if p["kd_on_kd"])
    on_anakinra_n = sum(1 for p in patients if p["anakinra"])
    chronic_n = sum(1 for p in patients if "Chronic" in p["disease_phase"] or "Long-term" in p["disease_phase"])

    etiology_dist = {}
    for p in patients:
        k = p["etiology"]
        etiology_dist[k] = etiology_dist.get(k, 0) + 1

    return {
        "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "dashboard": "FIRES — Febrile Infection-Related Epilepsy Syndrome",
        "total_patients": total,
        "male_n": male_n,
        "male_pct": round(male_n / total * 100),
        "drug_resistant_n": drug_resistant_n,
        "drug_resistant_pct": round(drug_resistant_n / total * 100),
        "partial_control_n": partial_n,
        "partial_control_pct": round(partial_n / total * 100),
        "seizure_free_n": seizure_free_n,
        "seizure_free_pct": round(seizure_free_n / total * 100),
        "on_kd_n": on_kd_n,
        "on_kd_pct": round(on_kd_n / total * 100),
        "on_anakinra_n": on_anakinra_n,
        "on_anakinra_pct": round(on_anakinra_n / total * 100),
        "chronic_phase_n": chronic_n,
        "etiology_distribution": etiology_dist,
        "prognosis_summary": {
            "srse_mortality": "10–20% in acute phase (ICU-related complications)",
            "chronic_epilepsy_risk": ">90% survivors develop drug-resistant epilepsy",
            "cognitive_impairment": ">80% with significant cognitive sequelae",
            "second_SRSE_risk": "15–25% risk of second FIRES episode with fever",
            "kd_responder_rate": "50–60% with >50% seizure frequency reduction",
            "anakinra_response_rate": "55–67% in published case series",
            "surgery_engel_I": "40–50% Engel I in FCD-FIRES surgical candidates",
        },
        "key_thresholds": KEY_THRESHOLDS,
        "clinical_alerts": [
            "⛔ POLG mutation: Valproate ABSOLUTELY CONTRAINDICATED — fatal hepatic failure in Alpers syndrome. Screen ALL FIRES before VPA.",
            "⚠️ SRSE = SE ≥24h on general anaesthesia — DO NOT wean anaesthesia without EEG confirmation of burst-suppression → seizure-free period.",
            "⚠️ Propofol infusion syndrome (PRIS) risk in children — limit propofol to <48h; prefer ketamine + midazolam for prolonged anaesthesia in FIRES.",
            "✅ Anakinra: start by day 5 of SRSE regardless of serology — CSF IL-1β is almost universally elevated; do not wait for antibody results.",
            "✅ Ketogenic diet should be initiated simultaneously with immunotherapy in ALL FIRES — IV KD emulsion in ICU, oral KD on extubation.",
            "⚠️ Rituximab: HBsAg + anti-HBc MANDATORY before first dose; entecavir prophylaxis if anti-HBc positive.",
            "⚠️ Tocilizumab: check TB (IGRA), Hepatitis B/C, HIV before initiating — risk of reactivation under IL-6 blockade.",
        ],
        "references": REFERENCES,
    }


def breakdown():
    patients = _generate_patients()
    return {
        "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "patients": patients,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
    }


def definitions():
    return {
        "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "concepts": [
            {"term": "FIRES", "definition": "Febrile Infection-Related Epilepsy Syndrome — devastating epileptic encephalopathy in previously healthy children triggered by febrile illness → super-refractory status epilepticus → chronic drug-resistant epilepsy. Coined by van Baalen 2010."},
            {"term": "NORSE", "definition": "New Onset Refractory Status Epilepticus — the broader term encompassing all ages (adults and children) presenting with refractory SE without prior epilepsy or clear cause. FIRES is NORSE with a febrile prodrome in children. NORSE without fever is adult-onset."},
            {"term": "SRSE", "definition": "Super-Refractory Status Epilepticus — status epilepticus continuing or recurring ≥24 hours after initiation of general anaesthesia. SRSE is the defining event of FIRES; median duration 28 days (range 2–120 days in published series)."},
            {"term": "NLRP3 Inflammasome", "definition": "A multiprotein intracellular complex in microglia and astrocytes that activates IL-1β. Fever, neuroinflammation, and metabolic stress activate NLRP3. BHB (from ketogenic diet) inhibits NLRP3 — the mechanistic basis for KD in FIRES."},
            {"term": "IL-1β (Interleukin-1 beta)", "definition": "Pro-inflammatory cytokine massively elevated in FIRES CSF (>50 pg/mL in acute phase). IL-1β enhances NMDA receptor calcium currents, increases BBB permeability, and activates microglial inflammation. Target of Anakinra (IL-1Ra)."},
            {"term": "Anakinra (IL-1Ra)", "definition": "Recombinant human IL-1 receptor antagonist blocking both IL-1α and IL-1β. The most studied specific immunotherapy for FIRES. Dose: 2–8 mg/kg/day IV in SRSE. FIRES consensus: start by day 5 of SRSE regardless of antibody results."},
            {"term": "Burst-Suppression Pattern", "definition": "EEG pattern of alternating bursts of electrical activity and periods of near-complete suppression (IBI 5–10 seconds = target in FIRES SRSE management). Does NOT exclude ongoing seizures — seizures may arise within bursts."},
            {"term": "Delta Brush", "definition": "Pathognomonic EEG pattern for anti-NMDAR encephalitis: rhythmic delta waves (1–3 Hz) with superimposed beta-frequency (20–30 Hz) oscillations — resembles neonatal delta brush. Present in <30% of anti-NMDAR cases; highly specific when present."},
            {"term": "LPD / GPD (ACNS 2021)", "definition": "Lateralised Periodic Discharges / Generalised Periodic Discharges — updated ACNS terminology replacing PLEDs/GPEDs. Common in FIRES SRSE: LPDs over temporal regions correlate with focal limbic seizures; GPDs with widespread cortical hyperexcitability."},
            {"term": "Ketogenic Diet (KD)", "definition": "High-fat, low-carbohydrate diet producing sustained ketosis (blood BHB 2–4 mmol/L). Anti-inflammatory mechanism: BHB inhibits NLRP3 inflammasome → reduces IL-1β. Antiseizure mechanism: enhanced GABA, reduced glutamate. Level A evidence for chronic FIRES (60% responder rate)."},
            {"term": "ESES (Electrical Status Epilepticus in Sleep)", "definition": "EEG pattern of continuous or near-continuous (>85% of NREM sleep) spike-wave activity during sleep. Occurs in 40% of FIRES survivors in chronic phase. Associated with cognitive regression, language impairment, behavioural disturbance. Treated with clobazam, steroids, or IVIG."},
            {"term": "Propofol Infusion Syndrome (PRIS)", "definition": "Potentially fatal complication of high-dose prolonged propofol infusion (>4 mg/kg/h for >48h). Features: metabolic acidosis, rhabdomyolysis, cardiac failure, renal failure. PRIS risk is amplified in children — prefer midazolam or ketamine for prolonged anaesthesia in paediatric FIRES."},
            {"term": "FCD (Focal Cortical Dysplasia)", "definition": "Malformation of cortical development — architectural disruption of cortical layering (FCD I), dysmorphic neurons (FCD II), with transmantle sign pathognomonic for FCD IIb. Identified in 10% of FIRES as the structural vulnerability enabling SRSE. High-resolution 3T MRI post-processing and FDG-PET required."},
            {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Sudden, unexpected, witnessed or unwitnessed non-traumatic death in a person with epilepsy. FIRES carries high SUDEP risk due to severe, drug-resistant chronic epilepsy, nocturnal seizures, and respiratory compromise. SUDEP counselling mandatory for all families. Supervised sleep, prone positioning avoidance."},
        ],
        "contraindications": [
            {
                "drug": "Valproate (VPA)",
                "contraindicated_in": "FIRES with POLG1 mutation / Alpers-Huttenlocher syndrome",
                "consequence": "Fatal hepatic failure — VPA inhibits POLG1-dependent mitochondrial beta-oxidation; liver failure in 1–2 weeks; irreversible. Screen ALL FIRES with liver dysfunction or family history."
            },
            {
                "drug": "Propofol (high-dose, >48h)",
                "contraindicated_in": "Paediatric FIRES requiring prolonged anaesthetic suppression",
                "consequence": "Propofol Infusion Syndrome (PRIS) — metabolic acidosis, rhabdomyolysis, cardiac failure. Fatal if unrecognised. Limit to <48h in children; use ketamine or midazolam as alternative."
            },
            {
                "drug": "Live Vaccines",
                "contraindicated_in": "All FIRES patients on anakinra / rituximab / tocilizumab",
                "consequence": "Vaccine-strain infection in immunosuppressed patients. Defer all live vaccines until ≥6 months off biologic therapy. Inactivated vaccines may be given (with reduced immunogenicity on biologics)."
            },
            {
                "drug": "Rituximab (without HBV screen)",
                "contraindicated_in": "HBsAg-positive or anti-HBc-positive patients (without entecavir prophylaxis)",
                "consequence": "Hepatitis B reactivation — fulminant hepatic failure, potentially fatal. HBsAg + anti-HBc MANDATORY before first dose; entecavir prophylaxis if anti-HBc positive."
            },
            {
                "drug": "Sodium channel blockers (CBZ, OXC, PHT)",
                "contraindicated_in": "FIRES where aetiology is SCN1A-overlap / Dravet spectrum",
                "consequence": "Paradoxical seizure worsening via SCN1A mechanism; use with caution in seronegative FIRES with fever-sensitivity — prefer LEV, VPA (if POLG excluded), CLB, or KD."
            },
        ],
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
    }
