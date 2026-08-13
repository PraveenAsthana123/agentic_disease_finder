"""
POLG Epilepsy — Alpers-Huttenlocher Syndrome / POLG-DEE / Mitochondrial Epilepsy
==================================================================================
41-patient cohort · POLG (15q26.1) · Mitochondrial DNA Polymerase Gamma (catalytic alpha subunit)

POLG BIOLOGY:
POLG (Polymerase Gamma, 15q26.1) encodes the catalytic alpha subunit of the mitochondrial DNA
polymerase (Pol-γ), the sole DNA polymerase responsible for replication and repair of the
mitochondrial genome (mtDNA, ~16,569 bp). Pathogenic POLG variants impair mtDNA replication
fidelity and/or processivity, leading to:
  ① mtDNA depletion (quantitative loss, usually >70% before symptomatic) — Alpers/childhood-onset
  ② mtDNA multiple deletions (qualitative — accumulate with age/cell division) — PEO/SANDO adult onset
  → Progressive respiratory chain (RC) Complex I/III/IV deficiency in high-energy tissues (neuron, liver)
  → Neuronal energy crisis: ATP deficiency + reactive oxygen species (ROS) accumulation + calcium
     dysregulation → excitotoxicity → epilepsia partialis continua (EPC) + cortical necrosis
  → Alpers hepatopathy: hepatocyte energy failure + VPA-accelerated mtDNA depletion → ALF

PRECISION MECHANISM — WHY VPA IS ABSOLUTELY CONTRAINDICATED:
① VPA depletes mitochondrial glutathione (GSH) → increased oxidative mtDNA damage → accelerates
   mtDNA depletion in POLG-deficient hepatocytes (already mtDNA-depleted) → fatal hepatocyte failure
② VPA inhibits mtDNA polymerase gamma (POLG2 beta subunit) directly → compounds POLG1 deficiency
③ VPA → carnitine depletion → long-chain fatty acid beta-oxidation failure → lactic acidosis
④ VPA → mitochondrial permeability transition pore (mPTP) opening → cytochrome c release → apoptosis
Net result: VPA in Alpers = ALF within weeks-months. Time to ALF after VPA exposure: median 2-6 months.
MORTALITY: >80% of POLG-AHS patients exposed to VPA die from VPA-induced liver failure.

EPC (EPILEPSIA PARTIALIS CONTINUA) — THE HALLMARK:
EPC = continuous or semi-continuous rhythmic unilateral focal motor seizures (usually arm > face)
lasting >1 hour, without loss of consciousness. In POLG-AHS:
  • EPC represents focal cortical necrosis/depletion of cortical neurons in sensorimotor strip
  • Highly refractory: EPC in POLG is often completely drug-resistant (even benzodiazepines fail)
  • EEG: periodic lateralised discharges (PLEDs/LPDs) + focal rhythmic delta + embedded fast activity
  • MRI correlate: T2/FLAIR cortical signal change in Sylvian/perilandic cortex (evolves to atrophy)
  • MANAGEMENT: treat with levetiracetam + clobazam + ketamine infusion; phenobarbitone bridge acceptable
    — DO NOT give VPA even in status epilepticus

ALPERS HEPATOPATHY — LIVER FAILURE RISK:
  • Biochemistry: ALT/AST ≥10× ULN; bilirubin rising; PT↑; albumin↓ — acute or subacute pattern
  • Liver biopsy: panlobular hepatocyte dropout + microvesicular steatosis + bile duct proliferation +
    reduced mtDNA copy number (quantitative PCR from liver)
  • VPA triggers/accelerates; also: fever, surgery, intercurrent illness, valproate-like drugs
  • Liver transplantation: NOT recommended — neurological disease progresses independently
  • Management: supportive — NAC infusion, N-acetylcysteine (repletes GSH); avoid triggers

INHERITANCE SPECTRUM:
  AR (most DEE/AHS):   biallelic POLG variants — compound heterozygous most common
  AD (adult PEO):      heterozygous POLG (dominant-negative, haploinsufficiency)
  Sporadic:            de novo — rare in paediatric onset

MOST COMMON EUROPEAN VARIANTS:
  p.Ala467Thr (c.1399G>A)  — most prevalent European allele (~30-35% alleles); protein misfolding
  p.Trp748Ser (c.2243G>C)  — compound-het with Ala467Thr most common UK genotype
  p.Gly848Ser (c.2542G>A)  — severe phenotype; common Scandinavian
  p.Arg853Gln (c.2558G>A)  — hepatopathy predisposing

KEY SAFETY PEARLS:
• VPA ABSOLUTE CONTRAINDICATION — before ANY AED in a child with undiagnosed epilepsy + liver disease
  + developmental regression: POLG MUST be excluded. POLG sequencing BEFORE VPA in ALL children with:
  unexplained refractory focal epilepsy + liver enzyme elevation + developmental regression.
• Phenobarbitone: HIGH RISK — hepatotoxic in mitochondrial disease; use only as last resort in SE
• Topiramate: CAUTION — inhibits carbonic anhydrase → increased lactic acidosis risk
• Ketogenic diet: CAUTION — requires intact fatty acid oxidation; safe if LCHAD/VLCAD excluded; often
  used cautiously in POLG with specialist oversight
• POLG2 co-depletion: always sequence POLG2 (accessory beta subunit) and TWNK (Twinkle helicase)
  as phenocopy — clinically identical AHS but different gene, same VPA CI applies
"""

import random
from datetime import datetime

SEED = 9187  # dashboard 187
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ───────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": (
            "POLG biallelic AR — classic Alpers-Huttenlocher syndrome "
            "(compound heterozygous / homozygous)"
        ),
        "n": 18, "pct": 44,
        "category": "POLG-biallelic-AR-Alpers",
        "functional_class": "AR-mtDNA-depletion-Alpers",
        "mechanism": (
            "Most prevalent class (~44%): biallelic POLG1 loss-of-function variants (most commonly "
            "compound heterozygous p.Ala467Thr/p.Trp748Ser or p.Ala467Thr/p.Gly848Ser), producing "
            "severe mtDNA polymerase dysfunction → progressive mtDNA depletion (>70% reduction in "
            "hepatocytes and neurons) → Complex I/III/IV respiratory chain failure → ATP crisis in "
            "high-energy neurons → Alpers-Huttenlocher syndrome (AHS). AHS triad: (1) "
            "intractable focal epilepsy/EPC; (2) progressive psychomotor regression; (3) liver disease "
            "(Alpers hepatopathy). Onset: 2-4 years (range: neonatal to adolescence). Invariably fatal "
            "within 2-12 years — liver failure or SE. ACMG: PVS1/PS3 + PM2 → Pathogenic in trans."
        ),
        "eeg_signature": (
            "Alpers EEG triad: (1) HIGH-VOLTAGE SLOW (HVS) pattern — continuous or near-continuous "
            "high-amplitude (>200 µV) polymorphic delta/theta, maximal posteriorly; (2) FOCAL FAST "
            "ACTIVITY — intermittent runs of 12-16 Hz superimposed on slow background (ictal correlate "
            "of EPC); (3) PLEDs/LPDs — periodic lateralised epileptiform discharges, unilateral, "
            "correlating with focal cortical necrosis. Ictal: EPC produces near-continuous focal "
            "rhythmic delta (1-3 Hz) with embedded faster ictal activity in contralateral motor strip. "
            "Background progressively deteriorates — evolves from posterior slowing → continuous delta "
            "→ burst-suppression → iso-electric as disease advances."
        ),
        "clinical_note": (
            "Diagnostic algorithm: (1) Serum lactate + CK (lactate >2 mmol/L 60-70% sensitivity). "
            "(2) CSF lactate >3.0 mmol/L — present in 80% AHS. (3) MRI: T2/FLAIR cortical/subcortical "
            "signal change (thalami, occipital, perirolandic) — often asymmetric, evolves to focal "
            "atrophy. (4) POLG full gene sequencing + MLPA (deletion panel). (5) Respiratory chain "
            "enzymology (muscle biopsy: COX-deficient fibers, ragged-red fibers, reduced Complex I/IV). "
            "VPA MUST NOT be given while awaiting diagnosis. Use CBZ/LEV bridging."
        ),
    },
    {
        "etiology": (
            "POLG biallelic AR — infantile-onset severe DEE "
            "(neonatal / early infantile mtDNA depletion)"
        ),
        "n": 12, "pct": 29,
        "category": "POLG-biallelic-AR-infantile-DEE",
        "functional_class": "AR-mtDNA-depletion-infantile-severe",
        "mechanism": (
            "Severe homozygous or compound-heterozygous POLG variants causing near-complete loss of "
            "Pol-γ activity → profound neonatal/infantile mtDNA depletion → multi-organ failure: "
            "severe hypotonia, liver failure, refractory neonatal seizures, lactic acidosis. "
            "Presentation within days-weeks of life. Rapid progression to death within months. EEG: "
            "burst-suppression with suppression intervals → iso-electric. Distinction from classic "
            "AHS: earlier onset, faster progression, often no EPC (insufficient cortical organisation). "
            "Associated with p.Gly848Ser homozygous, or p.Arg853Gln (hepatopathy predisposing)."
        ),
        "eeg_signature": (
            "Burst-suppression (BS) pattern — high-amplitude bursts of mixed polyspike/slow activity "
            "(100-300 µV, duration 1-10 seconds) alternating with periods of near-suppression (<20 µV). "
            "BS does NOT respond to AEDs including B6 (unlike PDE-ALDH7A1). Subsequent evolution to "
            "continuous suppression/iso-electric pattern as mtDNA depletion progresses. Myoclonic jerks "
            "may be EEG-confirmed (occasional fragmented spikes preceding jerks). Ictal: brief tonic "
            "or clonic runs with limited post-ictal suppression (high background load)."
        ),
        "clinical_note": (
            "Neonatal POLG-DEE: differentiate from ALDH7A1-PDE (B6 trial first), SCN2A-DEE, "
            "STXBP1-DEE, and metabolic causes (non-ketotic hyperglycinaemia, pyridoxine deficiency). "
            "Emergency investigations: blood/urine amino acids, organic acids, lactate, ammonia, "
            "LFTs (ALT/AST/bilirubin), coagulation. CSF: lactate, glycine, neurotransmitters. "
            "POLG sequencing (blood): 5-10 working days. Interim: CBZ/phenobarbitone (NOT VPA). "
            "Palliative care discussion early — very high mortality."
        ),
    },
    {
        "etiology": (
            "POLG biallelic AR — juvenile/adolescent Alpers variant "
            "(late-onset, slower progression)"
        ),
        "n": 5, "pct": 12,
        "category": "POLG-biallelic-AR-juvenile",
        "functional_class": "AR-mtDNA-depletion-juvenile",
        "mechanism": (
            "Biallelic POLG with one or both missense alleles retaining partial Pol-γ activity → slower "
            "mtDNA depletion rate → AHS with later onset (8-20 years). Clinical phenotype: epilepsy "
            "(focal, EPC episodes, GTCS) + ataxia + visual failure (cortical) ± liver disease. Some "
            "have ataxia-neuropathy spectrum (MIRAS/SANDO) rather than full AHS. Progression over "
            "5-20 years rather than months. Often misdiagnosed as JME, focal cortical dysplasia, or "
            "autoimmune encephalitis before POLG identified. p.Ala467Thr compound-het most common."
        ),
        "eeg_signature": (
            "In juvenile-onset: initially focal posterior slowing (occipital theta/delta) + focal IEDs "
            "with generalisation tendency. EPC may be episodic rather than continuous. High-voltage "
            "slow evolves slowly. Photoparoxysmal response in ~30%. Background EEG may be near-normal "
            "between episodes early in disease. Posterior dominant rhythm preserved until later stages. "
            "Ictal: focal onset with rapid secondary generalisation or prolonged focal status. PLEDs "
            "emerge during disease exacerbations."
        ),
        "clinical_note": (
            "Mimics: FIRES (febrile-triggered — POLG can relapse with fever), Rasmussen encephalitis, "
            "autoimmune epilepsy (check NMDAR, LGI1, CASPR2), progressive myoclonic epilepsy (Unverricht, "
            "MERRF), mitochondrial disease (MELAS — mtDNA m.3243A>G). Key distinguisher: POLG has liver "
            "involvement (even subclinical — check LFTs yearly), ataxia-neuropathy, and VPA CI. "
            "Mitochondrial genetic panel (blood mtDNA common deletions, POLG/TWNK/RRM2B) before VPA."
        ),
    },
    {
        "etiology": (
            "POLG heterozygous AD — PEO/SANDO with focal epilepsy "
            "(dominant de novo or familial)"
        ),
        "n": 4, "pct": 10,
        "category": "POLG-heterozygous-AD-PEO-SANDO",
        "functional_class": "AD-mtDNA-deletions-PEO",
        "mechanism": (
            "Heterozygous POLG variants causing dominant-negative or haploinsufficiency effect → "
            "accumulation of mtDNA multiple deletions (qualitative defect) in post-mitotic tissues. "
            "Slower progression than AR. Syndromes: PEO (progressive external ophthalmoplegia), "
            "SANDO (sensory ataxic neuropathy, dysarthria, ophthalmoparesis), MEMSA (myopathy, "
            "encephalopathy, mental retardation, stroke-like episodes with Alpers). Epilepsy in "
            "AD-POLG is focal, often temporal/occipital, generally more responsive to AEDs than AR. "
            "VPA CI applies equally — still risk of hepatotoxicity + mitochondrial worsening."
        ),
        "eeg_signature": (
            "AD-POLG focal epilepsy: temporal (TLE-like) or occipital IEDs. Background may be "
            "near-normal with intermittent focal slow. Less severe than AR — no EPC in most AD cases. "
            "Encephalopathic changes during disease exacerbations. EMG/NCS: axonal sensory neuropathy "
            "commonly co-present (50-60%). Muscle biopsy: COX-deficient fibers (subset), ragged-red "
            "fibers (less common than AR). mtDNA deletions detectable in blood (lower level) and "
            "muscle (higher level — Southern blot or qPCR)."
        ),
        "clinical_note": (
            "Family history: PEO (drooping eyelids, bilateral ptosis), cerebellar ataxia, "
            "sensorineural hearing loss (SNHL), peripheral neuropathy in first-degree relatives "
            "→ consider dominant mitochondrial disorder. Genetic counselling: AD-POLG → 50% "
            "transmission risk per pregnancy. Variants: p.Tyr955His (most common AD), p.Arg943His. "
            "Phenotypic variability within same family — from asymptomatic to PEO to SANDO."
        ),
    },
    {
        "etiology": (
            "POLG-negative mitochondrial epilepsy phenocopy "
            "(TWNK/RRM2B/DGUOK/SUCLA2 — same AHS/EPC phenotype)"
        ),
        "n": 2, "pct": 5,
        "category": "POLG-negative-mitochondrial-phenocopy",
        "functional_class": "mtDNA-depletion-non-POLG",
        "mechanism": (
            "Clinically identical to AHS/POLG-DEE but caused by other mtDNA maintenance genes: "
            "TWNK (10q24.31, Twinkle helicase, mtDNA replication unwinding), RRM2B (8q22.3, "
            "p53R2, dNTP pool → mtDNA synthesis), DGUOK (2p13.1, deoxyguanosine kinase, pyrimidine "
            "salvage), SUCLA2 (13q14.2, succinate-CoA ligase, Krebs/mtDNA). All share mtDNA depletion "
            "syndrome spectrum. Critical: VPA CI applies to ALL mtDNA depletion syndromes regardless "
            "of gene — ALL can develop VPA-triggered acute liver failure. POLG negative genetic "
            "testing → next-tier panel: TWNK/RRM2B/DGUOK/SUCLA2."
        ),
        "eeg_signature": (
            "Indistinguishable from POLG-AHS: posterior high-voltage slow, PLEDs/LPDs, EPC pattern, "
            "progressive background deterioration. EEG signature reflects cortical mtDNA depletion "
            "regardless of the nuclear gene responsible for the replication defect. TWNK (Twinkle) "
            "can produce infantile-onset EPC indistinguishable from POLG p.Gly848Ser homozygous."
        ),
        "clinical_note": (
            "Management is identical to POLG-AHS: VPA ABSOLUTE CI, supportive mitochondrial therapy, "
            "genetic counselling for at-risk relatives. Liver transplantation equally not indicated "
            "for non-POLG mtDNA depletion syndromes (neurological progression continues post-LTx). "
            "If muscle biopsy shows mtDNA depletion + normal POLG sequencing → extended mitochondrial "
            "maintenance gene panel or WES (whole exome sequencing)."
        ),
    },
]

# ── Patient Roster (N=41) ─────────────────────────────────────────────────────
def _make_patients():
    patients = []
    pid = 1
    specs = [
        # (cat, func_class, n, age_range_mo, sex_bias, onset_range_y, typical_control)
        ("POLG-biallelic-AR-Alpers",              "AR-mtDNA-depletion-Alpers",     18, (24, 84),  "MF", (2.0, 6.0),  "drug-resistant"),
        ("POLG-biallelic-AR-infantile-DEE",       "AR-mtDNA-depletion-infantile-severe", 12, (3, 24), "MF", (0.1, 1.5), "drug-resistant"),
        ("POLG-biallelic-AR-juvenile",            "AR-mtDNA-depletion-juvenile",    5, (96, 240), "MF", (8.0, 18.0), "partially-controlled"),
        ("POLG-heterozygous-AD-PEO-SANDO",       "AD-mtDNA-deletions-PEO",         4, (180, 480), "MF", (20.0, 45.0),"partially-controlled"),
        ("POLG-negative-mitochondrial-phenocopy", "mtDNA-depletion-non-POLG",       2, (12, 60),  "MF", (1.0, 5.0),  "drug-resistant"),
    ]
    phases_map = {
        "POLG-biallelic-AR-Alpers":              ["diagnostic","seizure-onset","EPC-phase","hepatopathy","palliative"],
        "POLG-biallelic-AR-infantile-DEE":       ["neonatal-crisis","early-infancy","progressive-decline","palliative"],
        "POLG-biallelic-AR-juvenile":            ["seizure-onset","EPC-phase","plateau","progressive-decline"],
        "POLG-heterozygous-AD-PEO-SANDO":        ["PEO-onset","focal-epilepsy","stable-maintenance","late-progression"],
        "POLG-negative-mitochondrial-phenocopy": ["diagnostic","seizure-onset","EPC-phase","palliative"],
    }
    sex_pool = ["M","F"]
    seizure_controls = {
        "drug-resistant":       ("drug-resistant", "#dc3545"),
        "partially-controlled": ("partially-controlled","#fd7e14"),
    }
    for cat, func_class, n, age_range, _, onset_range, control_type in specs:
        phases = phases_map[cat]
        for _ in range(n):
            sex = random.choice(sex_pool)
            age_months = random.randint(*age_range)
            onset_years = round(random.uniform(*onset_range), 1)
            phase = random.choice(phases)
            # VPA exposed? (critically: some exposed before diagnosis → liver)
            vpa_exposed = (cat == "POLG-biallelic-AR-Alpers" and random.random() < 0.45) or \
                          (cat == "POLG-biallelic-AR-infantile-DEE" and random.random() < 0.30)
            liver_injury = vpa_exposed and random.random() < 0.78
            cbl = round(random.uniform(3.1, 12.4) if vpa_exposed else random.uniform(0.4, 2.9), 1)
            alt_x_uln = round(random.uniform(3.5, 25.0) if liver_injury else random.uniform(0.8, 2.5), 1)
            lactate = round(random.uniform(3.0, 8.5) if cat != "POLG-heterozygous-AD-PEO-SANDO" else random.uniform(1.5, 4.2), 1)
            epc_present = (cat in ("POLG-biallelic-AR-Alpers","POLG-negative-mitochondrial-phenocopy") and
                           random.random() < 0.75)
            # Current treatment
            if epc_present:
                txs = [random.choice(["LEV+CLB","LEV+KET+CLB","OXC+CLB+LEV","CBZ+LEV+CLB"])]
            elif cat == "POLG-heterozygous-AD-PEO-SANDO":
                txs = [random.choice(["CBZ","OXC","LTG","LEV"])]
            else:
                txs = [random.choice(["LEV+CLB","CBZ+LEV","OXC+LEV","KD+LEV"])]
            patients.append({
                "id": f"POLG-{pid:03d}",
                "age_months": age_months,
                "sex": sex,
                "onset_years": onset_years,
                "category": cat,
                "functional_class": func_class,
                "disease_phase": phase,
                "current_treatment": txs[0],
                "seizure_control": control_type,
                "epc_present": epc_present,
                "vpa_exposed": vpa_exposed,
                "liver_injury_vpa": liver_injury,
                "csf_lactate_mmol": lactate,
                "alt_x_uln": alt_x_uln,
                "mtdna_depletion_pct": round(random.uniform(55, 92)) if cat != "POLG-heterozygous-AD-PEO-SANDO" else None,
                "polg_variant_1": random.choice(["p.Ala467Thr","p.Trp748Ser","p.Gly848Ser","p.Arg853Gln","p.Tyr955Cys"]),
                "polg_variant_2": random.choice(["p.Trp748Ser","p.Gly848Ser","p.Arg853Gln","p.Ala467Thr",None]) if cat.startswith("POLG-biallelic") else None,
                "mito_cofactors": random.choice([True, True, False]),
                "kd_trialed": random.random() < 0.35,
            })
            pid += 1
    return patients

PATIENTS = _make_patients()

# ── Seizure Types (4 core types) ───────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Epilepsia Partialis Continua (EPC)",
        "prevalence_pct": 78,
        "onset_age": "2-8 years (peak in Alpers), can be any age",
        "eeg_correlate": (
            "Near-continuous focal rhythmic delta (1-3 Hz) with superimposed fast activity in "
            "contralateral sensorimotor cortex. PLEDs/LPDs often co-present. EPC is semi-continuous "
            "by definition: persists for hours-days-weeks with brief interruptions. EEG background "
            "shows progressive high-voltage slow (HVS) pattern. Ictal onset zone: perirolandic / "
            "occipital most common in POLG."
        ),
        "clinical_tip": (
            "EPC in POLG is almost always refractory to AEDs. Do NOT escalate to VPA for EPC — "
            "give LEV 60 mg/kg/day IV loading then maintenance, CLB 0.5-1 mg/kg/day, consider "
            "ketamine infusion (0.3-1.5 mg/kg/h) for burst-suppression in super-refractory SE. "
            "Phenobarbitone only as last resort (hepatotoxic risk). Family counselling: EPC in "
            "POLG indicates active cortical necrosis — disease is progressing."
        ),
    },
    {
        "type": "Focal Seizures (occipital/perirolandic onset)",
        "prevalence_pct": 90,
        "onset_age": "Variable — mirrors POLG disease onset",
        "eeg_correlate": (
            "Focal IEDs: occipital/posterior > frontal > temporal. High-voltage slow complexes "
            "(posterior maximal). During ictal: rhythmic focal delta with/without secondary "
            "generalisation. Occipital seizures: visual phenomena (phosphenes, ictal blindness), "
            "versive head/eye movements, nausea. Perirolandic: jacksonian march, EPC initiation. "
            "Post-ictal: Todd's paresis common with perirolandic focus."
        ),
        "clinical_tip": (
            "Occipital focal seizures in a child with liver disease + developmental regression = "
            "POLG until proven otherwise. Differential: MELAS (mtDNA m.3243A>G — MRI stroke-like), "
            "MERRF (myoclonus-ataxia-seizures — muscle biopsy ragged-red), FIRES (fever-triggered "
            "SE — no liver disease). CBZ 10-20 mg/kg/day is best first-line for focal POLG seizures "
            "(hepatic enzyme induction — monitor LFTs; switch to OXC if hepatotoxicity concern)."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "prevalence_pct": 62,
        "onset_age": "Usually secondary to focal-onset; pure GTCS uncommon",
        "eeg_correlate": (
            "Generalised paroxysmal fast activity evolving to rhythmic clonic discharge then "
            "post-ictal attenuation. Often preceded by focal preictal slow in occipital/parietal "
            "channels. Myoclonic jerks may precede GTCS (PME-like pattern in some). Background: "
            "diffuse slowing proportional to disease burden — can be severe even between seizures."
        ),
        "clinical_tip": (
            "GTCS in POLG-AHS: treat with LEV (broad-spectrum, hepatically safe) + CLB adjunct. "
            "Avoid VPA (absolute CI), LTG (can worsen myoclonus in PME-like cases), CBZ "
            "(may worsen myoclonus/GTCS in some mitochondrial presentations). Ketogenic diet "
            "shows modest benefit (30% seizure reduction) in drug-resistant POLG if fatty acid "
            "oxidation intact — check LCHAD/VLCAD before commencing KD."
        ),
    },
    {
        "type": "Focal Status Epilepticus / Super-refractory SE",
        "prevalence_pct": 55,
        "onset_age": "Any age — peak risk during febrile illness or disease exacerbation",
        "eeg_correlate": (
            "EEG-confirmed focal SE: persistent focal ictal pattern >30 min without clinical "
            "improvement. Background: generalised periodic discharges (GPDs) with triphasic "
            "morphology in hepatic encephalopathy co-presence. Burst-suppression in "
            "anaesthesia-treated SE (ketamine/propofol/midazolam). Burst-suppression TARGET: "
            "1-3 bursts/min, then wean slowly after 24-48h."
        ),
        "clinical_tip": (
            "POLG SE management: (1) IV LEV 60 mg/kg loading; (2) IV CLB/MDZ; (3) IV phenobarb "
            "only if LEV/CLB fail (CAUTION — hepatotoxic, monitor LFTs q24h); (4) IV ketamine "
            "0.3-1.5 mg/kg/h (no hepatotoxicity, mild NMDA receptor antagonism); (5) IV propofol "
            "for burst-suppression induction (24-48h only — propofol infusion syndrome risk). "
            "NEVER give VPA IV even in SE — mortality risk unacceptably high in POLG."
        ),
    },
]

# ── Seizure Triggers (8 core) ─────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Febrile illness / intercurrent infection",
        "rate_pct": 88,
        "note": (
            "The single most potent trigger for POLG-AHS exacerbations. Fever → metabolic demand "
            "↑ → mtDNA-depleted neurons cannot meet ATP demand → acute cortical energy crisis → "
            "EPC onset or GTCS cluster. Each febrile exacerbation causes irreversible cortical "
            "damage. MANAGEMENT: (1) Written 'sick day plan' given to ALL POLG families — escalate "
            "AEDs at first fever; (2) Rescue CLB 0.5-1 mg/kg + IV fluids at fever onset; "
            "(3) Avoid dehydration — maintain glucose-electrolyte balance."
        ),
    },
    {
        "trigger": "Valproate / VPA exposure (iatrogenic)",
        "rate_pct": 45,
        "note": (
            "ABSOLUTE CONTRAINDICATION — but 45% of cohort received VPA BEFORE POLG was diagnosed "
            "(pre-diagnosis exposure). Of those exposed to VPA: 78% developed acute liver injury "
            "(ALT >3× ULN); 32% progressed to acute liver failure requiring ICU admission. "
            "VPA-induced Alpers hepatopathy mortality: >80% without liver transplant. Even patients "
            "without liver disease showed accelerated neurological decline after VPA exposure. "
            "MANDATORY: POLG testing before VPA in any child with unexplained focal epilepsy + "
            "liver enzyme elevation + developmental regression."
        ),
    },
    {
        "trigger": "Missed AED doses / sudden AED withdrawal",
        "rate_pct": 72,
        "note": (
            "Abrupt AED withdrawal is a major POLG exacerbation trigger — particularly dangerous "
            "because POLG seizures are refractory, and breakthrough seizures can initiate EPC that "
            "is then self-sustaining. Patients/carers must be counselled: NEVER stop AEDs abruptly. "
            "Sick day plan: if unable to swallow oral AEDs → IV/IM/buccal route mandated. Seizure "
            "diary + wearable seizure alert recommended for all POLG patients."
        ),
    },
    {
        "trigger": "Surgical procedures / general anaesthesia",
        "rate_pct": 42,
        "note": (
            "Surgery/anaesthesia triggers: (1) starvation (glucose deprivation → mitochondrial energy "
            "crisis); (2) hypothermia → impaired RC function; (3) propofol infusion (POLG-safe if "
            "<48h, but propofol inhibits Complex I — use cautiously); (4) suxamethonium (myopathy "
            "risk). Pre-operative POLG protocol: (a) minimum fasting; (b) glucose-saline IV throughout; "
            "(c) avoid halothane/enflurane; (d) ketamine or sevoflurane preferred; (e) avoid "
            "neuromuscular blockers if myopathy present; (f) post-operative EEG monitoring 48h."
        ),
    },
    {
        "trigger": "Medication exposures (non-VPA mitochondrial toxins)",
        "rate_pct": 35,
        "note": (
            "Other mitochondrially-toxic drugs that can exacerbate POLG: (1) Topiramate — carbonic "
            "anhydrase inhibition → lactic acidosis amplification (CAUTION — use lowest effective "
            "dose); (2) Metformin — Complex I inhibitor (ABSOLUTE CI in mtDNA disease); (3) Statins "
            "— myotoxic in mitochondrial myopathy; (4) Aminoglycoside antibiotics — mitochondrial "
            "ribosome inhibition → worsens pre-existing RCE; (5) Linezolid — mitochondrial protein "
            "synthesis inhibition → lactic acidosis; (6) Chloramphenicol — CI. Use hepatically-safe "
            "antibiotics (amoxicillin/cephalosporins) for POLG patients."
        ),
    },
    {
        "trigger": "Metabolic decompensation (dehydration, hypoglycaemia, fasting)",
        "rate_pct": 68,
        "note": (
            "Glucose is the primary fuel for POLG-depleted neurons (impaired fatty acid oxidation "
            "in some). Hypoglycaemia (<3.0 mmol/L) acutely worsens mitochondrial ATP production "
            "→ seizure threshold drops dramatically. MANAGEMENT: (1) glucose-electrolyte solution "
            "for sick days; (2) never fast >4h (neonatal/infant: >2h); (3) emergency glucose gel "
            "and IV dextrose at home; (4) hospital admission protocol at first sign of dehydration. "
            "Sick day letter: provide to school, GP, emergency department — states VPA CI and glucose "
            "supplementation protocol."
        ),
    },
    {
        "trigger": "Disease progression / new MRI lesions",
        "rate_pct": 60,
        "note": (
            "POLG-AHS is progressive — each neurological exacerbation (EPC episode, SE event) "
            "produces irreversible focal cortical necrosis. New T2/FLAIR MRI lesions (cortical "
            "signal change in perirolandic, occipital, insular cortex) correlate with new EPC "
            "foci. Disease staging by MRI: Stage I (focal T2 change, asymmetric) → Stage II "
            "(bilateral posterior > anterior) → Stage III (generalised cortical atrophy). "
            "Quarterly MRI recommended during active phase to guide AED escalation and family "
            "counselling re: prognosis."
        ),
    },
    {
        "trigger": "Psychological stress / sleep deprivation",
        "rate_pct": 40,
        "note": (
            "Chronic stress → cortisol → mitochondrial ROS → enhanced excitotoxicity in POLG "
            "neurons. Sleep deprivation reduces seizure threshold in all epilepsies and is "
            "particularly impactful in POLG (mitochondria regenerate ATP during sleep). "
            "MANAGEMENT: (1) seizure precautions — sleep safety protocol; (2) supervised bathing "
            "only; (3) SUDEP counselling (risk elevated in POLG GTCS); (4) school adjustments "
            "for fatigue management; (5) carer respite support — primary carer burnout common."
        ),
    },
]

# ── Treatments (8 items) ──────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Levetiracetam (LEV) — FIRST-LINE (hepatically safe)",
        "evidence": "Level B — first-line in POLG (hepatic safety, broad-spectrum, no drug interactions)",
        "indication": "First-line for all POLG seizure types — focal, GTCS, EPC, SE; hepatically safe",
        "dose": (
            "IV loading: 60 mg/kg (max 3g) over 15 min in SE. Maintenance: 30-60 mg/kg/day BD "
            "(paediatric) or 1000-3000 mg/day BD (adult). Titrate by 500-1000 mg/day every 2w. "
            "No dose adjustment required in liver disease (renally cleared). Level: not routinely "
            "needed (clinical response-guided)."
        ),
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) modulator → reduces presynaptic neurotransmitter "
            "release, particularly glutamate. Broad-spectrum mechanism: also inhibits high-voltage "
            "activated Ca²⁺ channels and GABA transaminase. No hepatic metabolism (>90% renal "
            "elimination) — IDEAL for POLG hepatopathy."
        ),
        "efficacy": (
            "Reduces seizure frequency in 50-65% of POLG focal epilepsy cases. EPC response: partial "
            "in 40-50% (reduces EPC intensity/duration but rarely eliminates). SE: IV loading achieves "
            "seizure cessation in 40% of focal SE within 30 min."
        ),
        "safety": (
            "POLG-safe: no hepatotoxicity. Side effects: behavioural (irritability, aggression — "
            "10-15%); somnolence (dose-dependent); rare: leukopenia (FBC q6M). Renal function "
            "monitoring q6M. Drug interactions: minimal — no CYP interactions."
        ),
        "monitoring": "LFTs not required (renal clearance); FBC q6M; behavioural questionnaire at each visit",
        "contraindications": "Renal failure (dose reduce); prior LEV behavioural intolerance",
    },
    {
        "drug": "Clobazam (CLB) — ADJUNCT (EPC/focal, safe in liver disease)",
        "evidence": "Level C — adjunct for EPC and focal seizures; rescue and maintenance roles",
        "indication": "EPC adjunct, focal seizure adjunct, rescue CLB (sick day protocol)",
        "dose": (
            "Maintenance: 0.3-1.0 mg/kg/day (paediatric) or 10-30 mg/day (adult) BD-TDS. "
            "Rescue: 0.3-0.5 mg/kg buccal/oral at seizure onset. Titrate slowly to avoid "
            "sedation. Tolerance develops in 20-30% — drug holiday (4-6 weeks) may restore effect. "
            "In hepatic impairment: reduce dose by 50%, monitor sedation."
        ),
        "moa": (
            "GABA-A positive allosteric modulator (1,5-benzodiazepine) — less sedating than "
            "1,4-benzodiazepines (diazepam). Binds γ2-containing GABA-A receptors. Hepatically "
            "metabolised to active N-desmethylclobazam (norCLB) — accumulates in liver disease "
            "(reduce dose, monitor sedation). Active metabolite half-life 40-100h."
        ),
        "efficacy": (
            "EPC: partial response in 50-60% (reduces EPC frequency/intensity, rarely eliminates). "
            "Rescue: buccal CLB 0.3 mg/kg terminates focal SE within 10 min in 55-70%. Sick day "
            "protocol CLB reduces hospitalisation rate by 40% vs no-rescue cohort."
        ),
        "safety": (
            "Sedation (dose-dependent); tolerance; respiratory depression at high IV doses. "
            "In hepatic impairment: norCLB accumulates → prolonged sedation. Monitor LFTs "
            "and sedation score at each visit. Dependence risk with chronic high-dose."
        ),
        "monitoring": "LFTs q3M (accumulation in liver disease); sedation scoring; norCLB TDM if hepatopathy",
        "contraindications": "Severe hepatic impairment (relative — reduce dose); respiratory failure",
    },
    {
        "drug": "Carbamazepine (CBZ) — First-line focal (caution LFT monitoring)",
        "evidence": "Level B — focal seizures in POLG; induces CYP3A4 (drug interactions)",
        "indication": "Focal seizures (occipital, perirolandic) in POLG; not for EPC monotherapy",
        "dose": (
            "Paediatric: 10-30 mg/kg/day TDS-QDS. Adult: 400-1600 mg/day BD (CR). Start low "
            "(100 mg/day), titrate over 4-6 weeks. TDM: 4-12 mg/L. Note: CYP3A4 inducer — "
            "may reduce levels of co-medications. HLA-B*1502 testing before initiating (SJS/TEN "
            "risk in East/South-East Asian populations — CPIC Level A)."
        ),
        "moa": (
            "Voltage-gated Na⁺ channel blocker — preferential action on fast-firing neurons. "
            "Reduces repetitive neuronal firing without blocking normal (low-frequency) neuronal "
            "activity. HEPATIC CAUTION: CBZ is an auto-inducer (induces its own metabolism via "
            "CYP3A4) — monitor LFTs in POLG hepatopathy. Hepatotoxic (rare, idiosyncratic) — "
            "monitor LFTs baseline + 6w + 3M + 6M."
        ),
        "efficacy": (
            "Focal seizures in POLG: 55-65% ≥50% reduction. Best for occipital/perirolandic "
            "focal onset. Not effective for EPC (may worsen via sodium channel effects on "
            "partially depolarised neurons in cortical necrosis zones). Also not for GTCS "
            "secondary to EPC — LEV preferred."
        ),
        "safety": (
            "Hepatotoxicity (idiosyncratic, rare — 1:10,000 but higher risk in mitochondrial "
            "disease with pre-existing liver dysfunction). POLG-hepatopathy: use with caution, "
            "LFTs q4-6w initially. Hyponatraemia (SIADH — common at higher doses); rash (10%). "
            "SJS/TEN (rare, HLA-B*1502 risk); aplastic anaemia (rare)."
        ),
        "monitoring": "LFTs + FBC baseline then q6w × 3, then q3M; TDM 4-12 mg/L; Na+ q3M; HLA-B*1502 before start",
        "contraindications": "HLA-B*1502 positive (SJS risk); severe hepatic failure; concurrent MAOI; POLG with ALT >3× ULN (switch to OXC/LEV)",
    },
    {
        "drug": "Ketamine infusion — Super-refractory SE / refractory EPC",
        "evidence": "Level C — emerging evidence for super-refractory SE and EPC in mitochondrial disease",
        "indication": "Super-refractory SE, refractory EPC unresponsive to LEV+CLB+phenobarb",
        "dose": (
            "IV loading: 1.5 mg/kg bolus (over 10 min). Maintenance infusion: 0.3-5 mg/kg/h "
            "(titrate to burst-suppression 1-3 bursts/min on cEEG). Wean after 24-48h of "
            "seizure control. Can be combined with midazolam infusion (additive). Oral/enteral "
            "ketamine not effective for SE."
        ),
        "moa": (
            "NMDA receptor open-channel blocker → reduces excitotoxicity during sustained "
            "seizure activity. MITOCHONDRIALLY SAFE: ketamine does not inhibit mitochondrial "
            "respiration (unlike propofol which inhibits Complex I). Also: dissociative "
            "anaesthetic (airway reflexes preserved at infusion doses). Hepatically safe "
            "(CYP2B6/CYP3A4 metabolism — dose-reduce in hepatic failure)."
        ),
        "efficacy": (
            "Super-refractory SE (SRSE): case series report 60-80% burst-suppression induction "
            "with ketamine infusion after barbiturate failure. EPC: reduces EPC frequency and "
            "intensity in 40-55% of POLG EPC cases in case series. Combination LEV+CLB+ketamine "
            "is current preferred POLG-safe SE protocol (no hepatotoxic agents)."
        ),
        "safety": (
            "Haemodynamic: hypertension (usually manageable), tachycardia. Psychotomimetic: "
            "emergence reactions (give midazolam concurrent). Respiratory: airway reflexes "
            "preserved (advantage over propofol). Hepatotoxicity: NOT reported. Propofol "
            "infusion syndrome risk: not applicable to ketamine. POLG-safe."
        ),
        "monitoring": "Continuous EEG (cEEG) during infusion; HR/BP/SpO2 continuous; LFTs q48h in hepatopathy",
        "contraindications": "Uncontrolled hypertension; intracranial hypertension (relative); known ketamine allergy",
    },
    {
        "drug": "Mitochondrial cofactors — Riboflavin (B2) + CoQ10 + L-carnitine",
        "evidence": "Level C — rational metabolic support; no RCT evidence; widely used in clinical practice",
        "indication": "All POLG patients — metabolic support to maximise residual RC function",
        "dose": (
            "Riboflavin (B2): 100-400 mg/day (Complex II FADH2 support). "
            "CoQ10: 10-30 mg/kg/day (max 1200 mg/day) — divide TDS. "
            "L-carnitine: 50-100 mg/kg/day (max 3g/day) — essential for fatty acid transport "
            "into mitochondria; replenishes VPA-depleted carnitine if prior VPA exposure. "
            "Folinic acid 1-5 mg/day — CSF folate supplementation."
        ),
        "moa": (
            "Riboflavin: electron carrier in Complex I/II — bypasses partial RC deficiency. "
            "CoQ10 (ubiquinol): mobile electron shuttle Complex I→III; antioxidant — scavenges "
            "mitochondrial ROS. L-carnitine: FA β-oxidation and mtDNA protection; depleted by "
            "VPA → mandatory after VPA exposure. Folinic acid: replenishes mitochondrial one-"
            "carbon metabolism. Note: no RCT evidence for POLG specifically but mechanistically "
            "rational and widely used in mitochondrial disease centres."
        ),
        "efficacy": (
            "No RCT evidence for seizure reduction in POLG specifically. Observational: "
            "mitochondrial cofactor cocktail associated with slower disease progression in some "
            "retrospective cohorts (Parikh 2009, DiMauro 2013 — mixed mitochondrial populations). "
            "Broad consensus among mitochondrial disease experts to supplement — minimal risk, "
            "potential benefit."
        ),
        "safety": "Riboflavin: harmless (orange urine). CoQ10: GI upset (give with food). L-carnitine: fishy odour (dose-dependent). No hepatotoxicity. All POLG-safe.",
        "monitoring": "Plasma carnitine (free/total) q6M; CoQ10 plasma level (target >2.5 µg/mL) q12M; clinical response tracking",
        "contraindications": "Selenium: theoretical risk in mitochondrial disease (limited data); avoid megadose antioxidants (can paradoxically impair RC electron flow)",
    },
    {
        "drug": "Ketogenic Diet (KD) — Drug-resistant POLG focal/GTCS",
        "evidence": "Level B — effective in DRE; CAUTION in POLG (fatty acid oxidation screen required)",
        "indication": "Drug-resistant POLG epilepsy after ≥2 AED failures; requires metabolic work-up",
        "dose": (
            "Classical KD: 4:1 ratio (fat:carb+protein by weight). Modified Atkins (MAD): "
            "10-20g/day net carbs. Medium-chain triglyceride (MCT) oil: 20-50% calories as MCT. "
            "POLG pre-KD screen: plasma LCHAD/VLCAD enzyme activity; plasma acylcarnitine profile "
            "(rule out MCAD, VLCAD) — long-chain FA oxidation defect is ABSOLUTE CI for KD. "
            "Monitor: beta-OHB 2-4 mmol/L; glucose >3.0 mmol/L at all times."
        ),
        "moa": (
            "Ketosis: beta-hydroxybutyrate (BHB) → alternative fuel for POLG neurons (bypasses "
            "Complex I glucose oxidation bottleneck partially) → mild seizure-suppressive effect. "
            "BHB activates TREK-1/TREK-2 K⁺ channels → neuronal membrane hyperpolarisation. "
            "Reduces glycolysis → less ROS generation from glucose oxidation → mild mitochondrial "
            "protection. Note: long-chain FA β-oxidation requires intact LCHAD/VLCAD — screen first."
        ),
        "efficacy": (
            "Retrospective POLG/mitochondrial disease series (Kang 2007, Lee 2008): 40-50% "
            "≥50% seizure reduction in DRE. EPC: variable response (20-35%). Best for generalised "
            "POLG GTCS. KD does not stop disease progression but can meaningfully reduce seizure "
            "burden. Aim 3-month ketosis trial before assessing response."
        ),
        "safety": (
            "POLG-specific risks: (1) hypoglycaemia → acute mitochondrial crisis (monitor glucose "
            "continuously during initiation); (2) carnitine depletion (supplement L-carnitine on KD); "
            "(3) worsening lactic acidosis in some (lactic acid from FA oxidation intermediates); "
            "(4) hepatic steatosis (rare but monitor LFTs). ABSOLUTE CI: VLCAD, LCHAD deficiency "
            "(long-chain FA oxidation defect — KD forces long-chain FA oxidation → energy crisis)."
        ),
        "monitoring": "BHB 2-4 mmol/L; glucose >3 mmol/L; lactate q4w initially; LFTs q3M; plasma acylcarnitines; lipid profile; selenium (KD depletes)",
        "contraindications": "VLCAD/LCHAD deficiency; MCAD deficiency; pyruvate carboxylase deficiency; severe hepatic failure; known dyslipidaemia",
    },
    {
        "drug": "Phenobarbitone (PB) — Last-resort SE (CAUTION — hepatotoxic risk)",
        "evidence": "Level C — use ONLY if LEV+CLB+ketamine fail in SE; HIGH RISK in POLG hepatopathy",
        "indication": "Last-resort in SE unresponsive to LEV+CLB+ketamine; very high caution",
        "dose": (
            "IV loading: 20 mg/kg (max 1g) over 30 min (monitoring). Maintenance: 3-5 mg/kg/day "
            "in POLG (reduce dose — slower metabolism in hepatopathy). TDM: 20-40 mg/L. "
            "In hepatic impairment: extend dosing interval, reduce maintenance. Stop as soon as "
            "alternative options available — do not continue as chronic AED in POLG."
        ),
        "moa": (
            "GABA-A positive allosteric modulator (barbiturate site) + Na⁺ channel blockade. "
            "Potent CNS depressant. HEPATIC CONCERNS in POLG: (1) hepatotoxic (idiosyncratic, "
            "higher risk in mitochondrial liver disease); (2) cytochrome P450 inducer — reduces "
            "CoQ10 levels (counter-productive in mito disease). Long half-life (80-120h) — "
            "accumulates in hepatic impairment."
        ),
        "efficacy": (
            "SE termination: 40-60% when added after LEV+BZD failure. EPC: modest. "
            "NOT recommended for chronic AED use in POLG — hepatotoxicity and CoQ10 depletion "
            "worsen disease. Emergency use only, planned discontinuation once SE controlled."
        ),
        "safety": (
            "POLG CAUTION: (1) hepatotoxic (idiosyncratic, 1:10,000 general population but higher "
            "in pre-existing liver disease); (2) CoQ10 depletion via CYP induction; "
            "(3) sedation/respiratory depression (ventilator support required at IV doses). "
            "NEVER use as first-line or chronic AED in POLG — reserve for SE only."
        ),
        "monitoring": "LFTs q48h during IV PB; TDM 20-40 mg/L; respiratory monitoring; CoQ10 levels q3M if continued",
        "contraindications": "ALT >3× ULN in POLG (HIGH RISK — use ketamine instead); acute hepatic porphyria; chronic use in POLG",
    },
    {
        "drug": "Valproate (VPA) — ABSOLUTE CONTRAINDICATION IN ALL POLG",
        "evidence": "Level A ABSOLUTE CI — VPA in POLG = acute liver failure, >80% mortality",
        "indication": "NEVER — contraindicated in ALL POLG patients regardless of disease stage",
        "dose": (
            "DO NOT USE. Any dose, any route (oral/IV/suppository), any indication including "
            "status epilepticus. VPA IV (e.g. Epilim IV) is ESPECIALLY DANGEROUS — rapid hepatic "
            "delivery at high concentration to POLG-diseased hepatocytes → accelerated ALF. "
            "VPA must be on the allergy/contraindication list in the POLG patient's medical record "
            "and all medical alert systems."
        ),
        "moa": (
            "VPA → POLG hepatotoxicity mechanism (quadruple hit): (1) mtDNA polymerase gamma "
            "inhibition (POLG2/accessory subunit binding) → compounds POLG1 deficiency → "
            "accelerates mtDNA depletion in hepatocytes already at ~30-40% normal mtDNA; "
            "(2) mitochondrial GSH depletion → ROS → mtDNA oxidative damage; (3) carnitine "
            "depletion → fatty acid β-oxidation failure → hepatocyte energy crisis; "
            "(4) mPTP opening → cytochrome c → apoptosis. ALL FOUR MECHANISMS simultaneously → "
            "rapid, irreversible hepatocyte loss → ALF."
        ),
        "efficacy": "N/A — DO NOT USE",
        "safety": (
            "MORTAL RISK: VPA in POLG → ALF in 32-45% of cases exposed, median time-to-ALF "
            "6 weeks (range 2 weeks–6 months). ALF in POLG: liver transplantation NOT indicated "
            "(brain disease continues post-LTx) → palliative/comfort care. Published case series: "
            "Naviaux 1999 (first POLG-VPA description); Wolf 2009 (liver failure in mitochondrial "
            "hepatopathy); Rahman 2012 (VPA CI in mitochondrial disease — consensus statement). "
            "VPA must NEVER be given even as temporary bridge while awaiting POLG result."
        ),
        "monitoring": "N/A — DO NOT USE",
        "contraindications": "ALL POLG PATIENTS — no exceptions. Document in red in medical record + electronic prescribing alert",
    },
]

# ── Monitoring Panel (8 items) ────────────────────────────────────────────────
MONITORING = [
    {
        "item": "Liver function tests (LFTs) — ALT/AST/Bili/Albumin/PT",
        "target": "ALT/AST <2× ULN; PT/INR normal",
        "frequency": "Every 3M; monthly if ALT rising; WEEKLY during disease exacerbations",
        "rationale": "Alpers hepatopathy detection — VPA-exposed: more frequent; rising ALT signals hepatic decompensation",
    },
    {
        "item": "Plasma/CSF lactate",
        "target": "Plasma <2.0 mmol/L; CSF <3.0 mmol/L",
        "frequency": "Every 3M; acute if suspected metabolic crisis; CSF at diagnosis then annually",
        "rationale": "RC failure biomarker — rising lactate indicates worsening mtDNA depletion; CSF lactate >3 mmol/L supports POLG diagnosis",
    },
    {
        "item": "POLG genetic testing (full sequencing + MLPA)",
        "target": "Two pathogenic variants in trans (AR) or one (AD)",
        "frequency": "Once — at diagnosis (blood DNA); family cascade testing offered to first-degree relatives",
        "rationale": "Diagnostic confirmation + genetic counselling; identifies at-risk siblings (AR: 25% risk per pregnancy)",
    },
    {
        "item": "Respiratory chain enzymology (muscle biopsy)",
        "target": "Complex I/III/IV activities; mtDNA copy number by qPCR",
        "frequency": "Once at diagnosis (muscle biopsy); repeat if mtDNA depletion % needed for disease staging",
        "rationale": "Confirms mtDNA depletion (% depletion); COX-deficient fiber count; ragged-red fibers (Gomori-trichrome)",
    },
    {
        "item": "Continuous EEG (cEEG) monitoring",
        "target": "EPC characterisation; SE monitoring; burst-suppression titration",
        "frequency": "During all SE events (minimum 24-48h); EPC monitoring; q6M outpatient EEG for disease staging",
        "rationale": "PLEDs/LPDs localise cortical necrosis foci; EPC monitoring; response to ketamine/phenobarb titration",
    },
    {
        "item": "MRI brain — T1/T2/FLAIR/DWI",
        "target": "No new T2/FLAIR cortical lesions; stable or absent cortical necrosis zones",
        "frequency": "Annually (stable); quarterly during active EPC phases; after each SE event",
        "rationale": "Cortical necrosis staging (T2/FLAIR perirolandic/occipital signal change correlates with EPC foci); disease progression monitoring",
    },
    {
        "item": "Plasma carnitine (free/total/acylcarnitine profile)",
        "target": "Free carnitine >20 µmol/L; acylcarnitine profile normal",
        "frequency": "Every 6M; monthly if VPA recently stopped (carnitine repleted); before KD initiation",
        "rationale": "VPA depletes carnitine → supplement L-carnitine; KD requires adequate carnitine for FA β-oxidation; mitochondrial disease baseline",
    },
    {
        "item": "Neuropsychological assessment + developmental milestones",
        "target": "Stable or plateau cognitive function; early detection of regression",
        "frequency": "Every 12M; after each SE or major EPC episode (regression common); educational report q12M",
        "rationale": "Regression tracking — POLG-AHS causes progressive cognitive decline; early occupational therapy/school adjustments improve quality of life",
    },
]

# ── Lifecycle Windows (6) ─────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Pre-symptomatic genetic",
        "age_range": "Birth – symptoms",
        "focus": "Sibling/family cascade testing",
        "key_action": "POLG sequencing of at-risk siblings (AR: 25% risk). Genetic counselling. Sick day plan pre-drafted. VPA added to contraindication record at birth if POLG confirmed.",
    },
    {
        "window": "Neonatal / early infantile onset",
        "age_range": "0 – 18 months",
        "focus": "Diagnosis, exclude treatable causes, bridging AED",
        "key_action": "Emergency metabolic screen (lactate, B6 trial, AASA, NKH screen). POLG sequencing priority. NEVER give VPA — use LEV + CBZ/PB bridge. Family counselling re: fatal prognosis.",
    },
    {
        "window": "Classic AHS (Alpers) — seizure onset phase",
        "age_range": "2 – 6 years",
        "focus": "AED optimisation, EPC prevention, metabolic support",
        "key_action": "LEV + CLB + mitochondrial cofactors. Sick day plan (CLB rescue + glucose). LFTs monthly. MRI staging. VPA CI documented. School support + SUDEP counselling.",
    },
    {
        "window": "EPC and disease exacerbation phase",
        "age_range": "Variable — typically 3 – 8 years in classic AHS",
        "focus": "EPC management, SE prevention, palliative planning",
        "key_action": "IV LEV + CLB + ketamine infusion for EPC/SE. cEEG monitoring. MRI quarterly. Ketogenic diet trial if tolerating. Palliative care team introduction. Goals-of-care meeting.",
    },
    {
        "window": "Stabilisation / plateau",
        "age_range": "Juvenile/adolescent POLG — months to years",
        "focus": "Chronic AED, metabolic, quality of life",
        "key_action": "Stable AED regimen (LEV ± OXC ± CLB). Annual MRI. Annual LFTs. Neuropsychological review. Social/educational support. Seizure alert wearable. Transition to adult neurology.",
    },
    {
        "window": "Late / adult-onset AD-POLG",
        "age_range": "18+ years",
        "focus": "PEO, ataxia, neuropathy management alongside epilepsy",
        "key_action": "Focal AED (CBZ/OXC/LTG). Annual LFT (lower hepatopathy risk in AD). Ophthalmology (PEO — prism glasses/ptosis surgery). Physio for ataxia. Genetic counselling (50% AD transmission).",
    },
]

# ── Clinical Alerts (4) ───────────────────────────────────────────────────────
ALERTS = [
    "🚨 VPA ABSOLUTE CI IN ALL POLG: Valproate in ANY POLG patient → acute liver failure (ALF) in 32-45%; mortality >80%. Document VPA as ALLERGY in all medical records and prescribing systems immediately at diagnosis.",
    "⚠️ POLG SEQUENCING BEFORE VPA: In any child with unexplained focal epilepsy + elevated LFTs + developmental regression — POLG must be excluded BEFORE initiating VPA. Use LEV as bridging AED while awaiting result (3-5 days expedited).",
    "⚡ EPC IS POLG UNTIL PROVEN OTHERWISE: Epilepsia partialis continua (continuous unilateral focal motor activity) in a child with liver disease + regression = Alpers-Huttenlocher syndrome. Start POLG workup immediately; use LEV + CLB; NEVER VPA even for EPC.",
    "🔴 SICK DAY PLAN MANDATORY: Every POLG patient must have a written sick-day protocol — CLB rescue 0.5 mg/kg at fever onset, IV glucose-saline if unable to feed, hospital threshold: any febrile seizure or EPC duration >10 min. Protocol to school, GP, A&E team.",
]

# ── Definitions (14 key concepts) ─────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "POLG / Polymerase Gamma",
        "definition": (
            "POLG (Polymerase Gamma, 15q26.1) encodes the catalytic alpha subunit (p140) of the "
            "mitochondrial DNA polymerase (Pol-γ), the only polymerase capable of replicating and "
            "repairing the mitochondrial genome (mtDNA, ~16,569 bp circle). Pol-γ operates as a "
            "trimer: one alpha (POLG1, catalytic) + two beta (POLG2, processivity) subunits. "
            "Pathogenic POLG variants reduce mtDNA replication fidelity and/or speed → depletion "
            "(quantitative) or deletion accumulation (qualitative). > 300 pathogenic POLG variants "
            "identified (ClinVar)."
        ),
    },
    {
        "term": "Alpers-Huttenlocher Syndrome (AHS)",
        "definition": (
            "Severe clinical phenotype of POLG-related disease: (1) intractable focal epilepsy ± "
            "EPC; (2) progressive psychomotor regression; (3) Alpers hepatopathy (liver disease). "
            "Onset: 2-4 years (neonatal to adolescence). Invariably fatal: median survival 2-5 "
            "years from symptom onset. Caused by biallelic (AR) POLG variants producing severe "
            "mtDNA depletion. Named: Alpers (1931, neuropathologist, described progressive "
            "neuronal loss + spongy degeneration) + Huttenlocher (1976, confirmed hepatic "
            "involvement as diagnostic criterion)."
        ),
    },
    {
        "term": "Epilepsia Partialis Continua (EPC)",
        "definition": (
            "Continuous or semi-continuous unilateral focal motor seizures (arm > face > leg) "
            "lasting >1 hour without loss of consciousness. In POLG-AHS, EPC represents ongoing "
            "cortical necrosis in sensorimotor cortex (perirolandic). EPC is highly refractory — "
            "rarely eliminated by AEDs; management goal is to reduce intensity and prevent "
            "SE. EEG: continuous or near-continuous focal rhythmic delta ± PLEDs in contralateral "
            "sensorimotor cortex. EPC is a clinical neurological emergency in POLG — signals "
            "active disease progression."
        ),
    },
    {
        "term": "mtDNA Depletion Syndrome (MDS)",
        "definition": (
            "Group of disorders characterised by severe reduction in mtDNA copy number in affected "
            "tissues (usually >70% depletion before symptomatic). Caused by mutations in nuclear-"
            "encoded mtDNA maintenance genes (POLG, TWNK, RRM2B, DGUOK, SUCLA2, SUCLG1, MPV17, "
            "TK2, others). Tissue distribution of depletion determines phenotype: "
            "hepato-encephalopathic (POLG, DGUOK — liver + brain); encephalomyopathic (SUCLA2, "
            "RRM2B — muscle + brain); myopathic (TK2 — muscle alone). All MDS carry VPA CI."
        ),
    },
    {
        "term": "VPA-Induced Alpers Hepatopathy",
        "definition": (
            "Fatal acute or subacute liver failure triggered by valproate in POLG-AHS patients. "
            "Mechanism: VPA depletes mitochondrial GSH + inhibits POLG2 + depletes carnitine + "
            "opens mPTP → hepatocyte mtDNA depletion accelerated → RC failure → ALF. Onset after "
            "VPA exposure: median 6 weeks (range 2 weeks–6 months). Biochemistry: rapidly rising "
            "ALT/AST, bilirubin, PT, declining albumin. Liver Biopsy: panlobular hepatocyte dropout "
            "+ microvesicular steatosis + mtDNA depletion (quantitative PCR). Liver transplant: "
            "NOT indicated — brain disease continues post-LTx."
        ),
    },
    {
        "term": "p.Ala467Thr (POLG c.1399G>A)",
        "definition": (
            "Most common European POLG pathogenic variant (~30-35% of mutant alleles in Europeans). "
            "Results in protein misfolding → impaired Pol-γ catalytic activity (5-10% residual). "
            "Classic genotype: p.Ala467Thr/p.Trp748Ser compound heterozygous (most common UK "
            "genotype) → AHS or MIRAS (mitochondrial recessive ataxia syndrome). p.Ala467Thr "
            "homozygous → variable phenotype (moderate to severe). ACMG classification: Pathogenic "
            "(PP3 + PS3 + PM3 + PM2). Detected on standard POLG sequencing panel."
        ),
    },
    {
        "term": "Periodic Lateralised Epileptiform Discharges (PLEDs / LPDs)",
        "definition": (
            "Lateralised periodic discharges (LPDs) — EEG pattern of periodic sharp waves, spikes, "
            "or complexes recurring every 0.5-4 seconds, maximal unilaterally. In POLG-AHS: "
            "LPDs localise to the hemisphere with active cortical necrosis (perirolandic, occipital). "
            "LPDs are associated with ongoing excitotoxicity and cortical injury. They may or may "
            "not have ictal significance (some are 'interictal' correlates of damaged cortex). "
            "Combined with high-voltage slow (HVS) background: highly characteristic of Alpers EEG."
        ),
    },
    {
        "term": "Respiratory Chain (RC) / OXPHOS Complexes",
        "definition": (
            "Mitochondrial electron transport chain: Complex I (NADH dehydrogenase, 45 subunits — "
            "7 mtDNA-encoded), Complex II (succinate dehydrogenase, all nuclear), Complex III "
            "(cytochrome bc1), Complex IV (cytochrome c oxidase / COX, 3 mtDNA-encoded), "
            "Complex V (ATP synthase). In POLG-AHS: mtDNA depletion → reduced Complex I/III/IV "
            "subunit production → RC failure → ATP deficit in neurons and hepatocytes. Enzymology "
            "from muscle biopsy quantifies RC enzyme activities; reduced Complex I + IV with "
            "normal Complex II = classical mtDNA depletion pattern."
        ),
    },
    {
        "term": "Twinkle Helicase (TWNK / C10orf2)",
        "definition": (
            "TWNK (10q24.31) encodes Twinkle, the mitochondrial DNA helicase that unwinds the "
            "mtDNA double helix ahead of the replication fork (partners with POLG in the minimal "
            "mtDNA replisome). Dominant TWNK mutations → mtDNA multiple deletions → PEO/CPEO "
            "(progressive external ophthalmoplegia). Recessive TWNK mutations → mtDNA depletion "
            "→ AHS-like phenotype clinically identical to POLG-AHS. VPA CI applies equally. "
            "TWNK sequencing is the first next-tier test after negative POLG in suspected AHS."
        ),
    },
    {
        "term": "COX-Deficient Fibres / Ragged-Red Fibres (RRF)",
        "definition": (
            "COX-deficient fibres: muscle fibres showing absent/reduced cytochrome c oxidase "
            "(Complex IV) activity on histochemical staining — appear blue on combined SDH/COX "
            "stain (COX-negative, SDH-positive). In POLG-AHS: scattered COX-deficient fibres "
            "reflect reduced Complex IV from mtDNA depletion. Ragged-red fibres (RRF): "
            "subsarcolemmal mitochondrial proliferation seen on Gomori trichrome stain as red "
            "aggregates — compensatory upregulation of mitochondrial biogenesis in depleted cells. "
            "RRF less common in POLG-AHS than in mitochondrial deletion disorders (PEO)."
        ),
    },
    {
        "term": "Mitochondrial Recessive Ataxia Syndrome (MIRAS)",
        "definition": (
            "Adult-onset POLG-related syndrome: spinocerebellar ataxia + peripheral neuropathy "
            "+ psychiatric features ± epilepsy; primarily p.Ala467Thr/p.Trp748Ser compound het. "
            "Onset 12-40 years. Milder than AHS — slower progression. Epilepsy: focal "
            "(temporal/occipital) in ~40%; EPC rare. VPA CI applies equally to MIRAS — liver "
            "disease may be subclinical but VPA can still trigger ALF. MIRAS is part of the "
            "AR-POLG spectrum: MIRAS → SANDO → MEMSA → AHS (increasing severity)."
        ),
    },
    {
        "term": "POLG Spectrum Disorders",
        "definition": (
            "POLG-related disorders span a wide clinical spectrum from neonatal lethality to "
            "asymptomatic adult carriers: (1) AHS (most severe AR — infant/child); (2) "
            "mtDNA depletion syndrome-hepatoencephalopathic (MDS-HE, AR, infant); (3) MIRAS "
            "(AR adult ataxia-neuropathy); (4) SANDO (AR sensory ataxia-neuropathy-dysarthria-"
            "ophthalmoparesis); (5) MEMSA (AR myopathy-encephalopathy-MELAS-like, SE); "
            "(6) CPEO/PEO (AD adult, opthalmoplegia ± ataxia); (7) late-onset ataxia neuropathy "
            "syndrome. All carry VPA CI — POLG genotype correlates imperfectly with phenotype."
        ),
    },
    {
        "term": "Sick Day Plan — Mitochondrial Disease Protocol",
        "definition": (
            "Written emergency management plan given to ALL POLG families: (1) Fever >38°C: "
            "give CLB rescue 0.3-0.5 mg/kg + paracetamol (NOT aspirin — salicylate disrupts "
            "mitochondrial oxidative phosphorylation); (2) Unable to feed: oral glucose polymer + "
            "electrolyte solution every 2h; hospital if not tolerating; (3) Seizure >5 min: "
            "buccal midazolam/CLB + call emergency services; (4) Any EPC: immediate hospital "
            "admission for IV LEV; (5) Hospital letter: states VPA CONTRAINDICATED IN POLG — "
            "do not give even in SE; includes treating neurologist direct contact number."
        ),
    },
    {
        "term": "N-acetylcysteine (NAC) — Alpers Hepatopathy Treatment",
        "definition": (
            "IV NAC is the principal intervention for VPA-induced Alpers hepatopathy (and "
            "general mitochondrial hepatopathy) — mechanism: NAC replenishes hepatic and "
            "mitochondrial glutathione (GSH), the primary mtDNA antioxidant defence. Protocol: "
            "IV NAC 150 mg/kg over 60 min loading, then 50 mg/kg over 4h, then 100 mg/kg over "
            "16h (repeat cycles in ALF). Oral NAC 600 mg TDS for ongoing GSH support. Evidence: "
            "NAC reduces VPA-induced mitochondrial oxidative stress in in vitro POLG models "
            "(Luft 1994, Naviaux 1999). No RCT in POLG specifically. Used as standard of care "
            "at mitochondrial disease centres for POLG hepatopathy."
        ),
    },
]

# ── Evidence Standards (8) ────────────────────────────────────────────────────
STANDARDS = [
    {
        "standard": "ILAE-2022",
        "title": "ILAE 2022 classification of the epilepsies / genetic epilepsy nomenclature",
        "relevance": "POLG-DEE classification; EPC definition; treatment-resistant epilepsy thresholds",
    },
    {
        "standard": "NICE-NG217",
        "title": "NICE Epilepsies NG217 (2022) — diagnosis and management",
        "relevance": "AED choice in genetic epilepsies; referral thresholds; VPA risk documentation; POLG testing indications",
    },
    {
        "standard": "EUROMIT-Consensus-2020",
        "title": "European Mitochondrial Disease Consensus Statement (EUROMIT 2020)",
        "relevance": "VPA CI in mitochondrial disease; mitochondrial cofactor supplementation; sick day plan; SE management",
    },
    {
        "standard": "Rahman-2012-Arch-Dis-Child",
        "title": "Rahman & Copeland (2012) — POLG disease and its many faces",
        "relevance": "Comprehensive POLG phenotype spectrum; VPA hepatotoxicity mechanism; diagnostic criteria",
    },
    {
        "standard": "Parikh-2015-Mol-Genet-Metab",
        "title": "Parikh et al (2015) — Diagnosis and Management of Mitochondrial Disease",
        "relevance": "Mitochondrial disease diagnostic pathway; biochemical testing; mitochondrial cofactors",
    },
    {
        "standard": "ACMG-AMP-2015",
        "title": "ACMG/AMP variant classification (Richards 2015, Am J Hum Genet)",
        "relevance": "POLG variant pathogenicity classification; PVS1, PP3, PM3, PM2 criteria",
    },
    {
        "standard": "Naviaux-1999-AnnNeurol",
        "title": "Naviaux & Nguyen (1999) — POLG mutations cause mitochondrial DNA depletion",
        "relevance": "First molecular characterisation of POLG mutations causing AHS; VPA CI mechanism",
    },
    {
        "standard": "ACNS-EEG-2021",
        "title": "American Clinical Neurophysiology Society — Guideline for cEEG (2021)",
        "relevance": "EPC monitoring protocol; cEEG in SE; LPD/PLED classification (ACNS 2.0 terminology)",
    },
]

# ── Monitoring Thresholds (10) ────────────────────────────────────────────────
THRESHOLDS = [
    {
        "threshold": "VPA-POLG: ABSOLUTE CI — ZERO DOSE",
        "action": "No valproate in any POLG patient, any age, any indication — document as allergy",
    },
    {
        "threshold": "ALT >3× ULN",
        "action": "Stop CBZ if on it; switch to OXC/LEV; escalate LFT monitoring to q2w; hepatology consult",
    },
    {
        "threshold": "CSF lactate >3.0 mmol/L",
        "action": "Supports AHS diagnosis; initiate POLG sequencing urgently; do not delay AED change awaiting result",
    },
    {
        "threshold": "EPC >10 minutes",
        "action": "Hospital admission, IV LEV loading 60 mg/kg, IV CLB rescue; cEEG monitoring; ketamine infusion if LEV+CLB fail",
    },
    {
        "threshold": "mtDNA depletion >70%",
        "action": "Confirms clinically significant depletion; maximise mitochondrial cofactors; family counselling re: prognosis",
    },
    {
        "threshold": "POLG testing before VPA (MANDATORY)",
        "action": "Any child with unexplained focal epilepsy + elevated LFTs + regression: POLG sequencing BEFORE VPA (use LEV bridge)",
    },
    {
        "threshold": "KD pre-screen: VLCAD/LCHAD activity",
        "action": "Normal VLCAD/LCHAD → proceed to KD. Deficient LCHAD/VLCAD → ABSOLUTE CI for ketogenic diet",
    },
    {
        "threshold": "LEV loading 60 mg/kg in SE",
        "action": "First-line IV AED for SE in POLG (hepatically safe); administer within 10 min of SE onset",
    },
    {
        "threshold": "Liver transplant in POLG-ALF: NOT indicated",
        "action": "Inform family: liver transplant does not prevent neurological progression; palliative goals-of-care discussion",
    },
    {
        "threshold": "SUDEP counselling — annually from seizure onset",
        "action": "POLG-GTCS: SUDEP risk elevated; wearable seizure monitors; safe sleeping; nocturnal supervision",
    },
]

# ── Key References (6) ────────────────────────────────────────────────────────
REFERENCES = [
    {
        "ref": "Naviaux-1999-AnnNeurol",
        "title": "Naviaux & Nguyen (1999) — POLG mutations cause mitochondrial DNA depletion and lead to the Alpers syndrome. Ann Neurol.",
        "relevance": "First molecular identification of POLG as AHS gene; establishes VPA hepatotoxicity mechanism in POLG",
    },
    {
        "ref": "Rahman-2012-ArchDisChild",
        "title": "Rahman & Copeland (2012) — POLG disease and its many faces. Arch Dis Child.",
        "relevance": "Comprehensive POLG clinical spectrum (AHS to MIRAS); diagnostic algorithm; VPA CI rationale",
    },
    {
        "ref": "Darin-2003-AnnNeurol",
        "title": "Darin et al (2003) — High frequency of nuclear-encoded mtDNA mutations in childhood-onset epilepsy. Ann Neurol.",
        "relevance": "Establishes POLG prevalence in childhood epilepsy; justifies early POLG testing in unexplained focal epilepsy",
    },
    {
        "ref": "Tzoulis-2006-Brain",
        "title": "Tzoulis et al (2006) — The spectrum of clinical disease caused by the A467T and W748S POLG mutations. Brain.",
        "relevance": "Defines genotype-phenotype correlations for most common POLG variants; p.Ala467Thr/p.Trp748Ser phenotype spectrum",
    },
    {
        "ref": "Wolf-2009-JChildNeurol",
        "title": "Wolf & Smeitink (2009) — Mitochondrial disorders: a proposal for consensus diagnostic criteria in infants and children. J Child Neurol.",
        "relevance": "Diagnostic criteria for mitochondrial disease in paediatric population; scoring system",
    },
    {
        "ref": "Parikh-2015-MolGenetMetab",
        "title": "Parikh et al (2015) — Diagnosis and Management of Mitochondrial Disease: A Consensus Statement. Mol Genet Metab.",
        "relevance": "Comprehensive mitochondrial disease management guidelines; mitochondrial cofactors; VPA CI; sick day plan",
    },
]


# ── Public API ─────────────────────────────────────────────────────────────────
def get_overview():
    """POLG Epilepsy (Alpers-Huttenlocher Syndrome / POLG-DEE) — overview endpoint."""
    total = sum(e["n"] for e in ETIOLOGY_CATALOG)
    epc_n = sum(1 for p in PATIENTS if p.get("epc_present"))
    dre_n = sum(1 for p in PATIENTS if p.get("seizure_control") == "drug-resistant")
    vpa_exp = sum(1 for p in PATIENTS if p.get("vpa_exposed"))
    liver_n = sum(1 for p in PATIENTS if p.get("liver_injury_vpa"))
    kd_n = sum(1 for p in PATIENTS if p.get("kd_trialed"))
    cofactor_n = sum(1 for p in PATIENTS if p.get("mito_cofactors"))
    return {
        "syndrome": "POLG Epilepsy — Alpers-Huttenlocher Syndrome (POLG-DEE / mtDNA Depletion)",
        "gene": "POLG — 15q26.1 — Mitochondrial DNA Polymerase Gamma alpha subunit (Pol-γ)",
        "inheritance": "Autosomal recessive (AHS/DEE — biallelic); Autosomal dominant (PEO/SANDO — heterozygous)",
        "eeg_hallmark": "High-voltage slow (HVS) + PLEDs/LPDs + EPC (epilepsia partialis continua) — Alpers EEG triad",
        "key_biomarker": "CSF lactate >3.0 mmol/L + mtDNA depletion >70% (muscle qPCR) + biallelic POLG variants",
        "precision_therapy": "VPA ABSOLUTE CI — LEV + CLB + mitochondrial cofactors (Riboflavin/CoQ10/L-carnitine)",
        "n_patients": total,
        "kpis": {
            "epc_pct": round(epc_n / total * 100),
            "dre_pct": round(dre_n / total * 100),
            "vpa_exposed_pct": round(vpa_exp / total * 100),
            "vpa_liver_injury_pct": round(liver_n / total * 100) if vpa_exp else 0,
            "kd_trialed_pct": round(kd_n / total * 100),
            "cofactors_pct": round(cofactor_n / total * 100),
        },
        "etiologies": [
            {"etiology": e["etiology"][:70], "n": e["n"], "pct": e["pct"]}
            for e in ETIOLOGY_CATALOG
        ],
        "seizure_type_prevalence": {s["type"][:50]: s["prevalence_pct"] for s in SEIZURE_TYPES},
        "trigger_seizure_rates": {t["trigger"][:50]: t["rate_pct"] for t in TRIGGERS},
        "lifecycle_windows": LIFECYCLE,
        "clinical_alerts": ALERTS,
        "key_aha": (
            "In any child with unexplained focal epilepsy + elevated liver enzymes + developmental "
            "regression: POLG sequencing BEFORE VPA. Valproate in POLG = acute liver failure in "
            "32-45% → mortality >80%. EPC (epilepsia partialis continua) is the hallmark — treat "
            "with LEV + CLB + ketamine. Never phenobarbitone as first-line. Never VPA even in SE."
        ),
        "generated_at": datetime.now().isoformat(),
        "dashboard_id": 187,
    }


def get_breakdown():
    """POLG Epilepsy — breakdown endpoint (full clinical detail)."""
    return {
        "patients": PATIENTS,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "alerts": ALERTS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }


def get_definitions():
    """POLG Epilepsy — definitions endpoint (14 key concepts + contraindications + thresholds)."""
    return {
        "syndrome": "POLG Epilepsy — Alpers-Huttenlocher Syndrome (POLG-DEE / mtDNA Depletion)",
        "definitions": DEFINITIONS,
        "absolute_contraindications": [
            {
                "drug": "Valproate (VPA) — ABSOLUTE CONTRAINDICATION IN ALL POLG",
                "scope": "ALL POLG patients — any age, any indication, any route (oral/IV/suppository/enteral)",
                "mechanism": (
                    "VPA depletes mitochondrial GSH + inhibits POLG2 accessory subunit + depletes "
                    "carnitine + opens mPTP → accelerated mtDNA depletion in POLG-diseased hepatocytes "
                    "→ acute liver failure (Alpers hepatopathy). Onset: median 6 weeks after VPA. "
                    "Mortality: >80% without liver transplant. Liver transplant does not help — "
                    "brain disease continues post-LTx."
                ),
                "action": (
                    "Document VPA as ALLERGY in all medical records (EMR red allergy flag). "
                    "Verbal and written VPA CI at every clinic visit. Give allergy bracelet/card. "
                    "Alert school nurse, GP, emergency department. Never give VPA IV in SE — use "
                    "IV LEV 60 mg/kg + IV CLB + ketamine infusion."
                ),
                "evidence": (
                    "Naviaux 1999 AnnNeurol; Rahman 2012 ArchDisChild; Wolf 2009 JChildNeurol; "
                    "EUROMIT 2020 Consensus; NICE NG217 2022; EAN 2019 — ALL agree VPA is absolutely "
                    "contraindicated in ALL POLG-related disorders."
                ),
            },
            {
                "drug": "Metformin — ABSOLUTE CONTRAINDICATION in mtDNA disease",
                "scope": "ALL mitochondrial disease patients including POLG",
                "mechanism": (
                    "Metformin is a Complex I inhibitor → compounds RC deficiency in mtDNA-depleted "
                    "POLG cells → severe lactic acidosis → metabolic crisis. Risk of fatal metformin-"
                    "associated lactic acidosis (MALA) is dramatically elevated in mitochondrial disease."
                ),
                "action": (
                    "Exclude from prescribing in all POLG patients. If POLG patient requires glucose "
                    "control: insulin preferred. Alert endocrinology/diabetes team. Document as "
                    "contraindication in medical record."
                ),
                "evidence": "EUROMIT 2020 Consensus; Parikh 2015 MolGenetMetab; BNF mitochondrial disease CI list",
            },
            {
                "drug": "Liver transplantation for POLG-ALF",
                "scope": "POLG-induced acute liver failure — NOT INDICATED",
                "mechanism": (
                    "Liver transplant in POLG-ALF does not treat the underlying mtDNA depletion in "
                    "the brain (neurons, not hepatocytes, are responsible for neurological decline). "
                    "Post-LTx: neurological deterioration continues — EPC, regression, death from "
                    "neurological causes within months. Published series: 100% neurological "
                    "progression post-LTx in POLG-AHS."
                ),
                "action": (
                    "Goals-of-care meeting with family before any discussion of LTx. Refer to "
                    "palliative care team simultaneously with hepatology. Provide written prognostic "
                    "information. Document family's informed choice."
                ),
                "evidence": "Kayler 2002 Transplantation; Delarue 2014 Hepatology — series showing no neurological benefit post-LTx in AHS",
            },
            {
                "drug": "Withholding POLG testing before VPA in at-risk children",
                "scope": "MANDATORY: POLG testing before VPA in unexplained focal epilepsy + LFTs + regression",
                "mechanism": (
                    "Failure to exclude POLG before initiating VPA = potentially preventable VPA-"
                    "induced ALF. Clinical scenario triggering MANDATORY POLG exclusion: (1) focal "
                    "epilepsy + (2) ALT/AST >1.5× ULN or bilirubin rise + (3) developmental "
                    "regression or (4) sibling with AHS. Expedited POLG sequencing (3-5 business "
                    "days). Bridging AED: IV/oral LEV — hepatically safe."
                ),
                "action": (
                    "Send POLG sequencing (blood EDTA 5 mL) before VPA initiation. Bridging AED: "
                    "LEV 30 mg/kg/day oral or IV. Document in medical record: 'VPA withheld pending "
                    "POLG result — reason: unexplained focal epilepsy + elevated LFTs + regression.' "
                    "Expedite processing: label sample URGENT-POLG."
                ),
                "evidence": "NICE NG217; Rahman 2012; EUROMIT 2020 — POLG exclusion protocol before VPA",
            },
        ],
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }
