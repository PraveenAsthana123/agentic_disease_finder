#!/usr/bin/env python3
"""ALDH5A1 (Succinic Semialdehyde Dehydrogenase Deficiency / 4-Hydroxybutyric Aciduria / SSADHA)
Epilepsy Dashboard.

ALDH5A1 encodes Succinic Semialdehyde Dehydrogenase (SSADH), a mitochondrial matrix NAD+-dependent
aldehyde dehydrogenase (~535 aa, ~59 kDa, ALDH superfamily class 5). SSADH is the FINAL STEP in
GABA catabolism, converting succinic semialdehyde (SSA) to succinate (entering TCA cycle).

GABA CATABOLISM PATHWAY (ALDH5A1 position):
  GABA (γ-aminobutyric acid)
    ↓  GABA-T (GABA transaminase, encoded by ABAT) — transamination with α-ketoglutarate
  Succinic Semialdehyde (SSA) + glutamate
    ↓  SSADH (encoded by ALDH5A1) — NAD+-dependent oxidation [THIS STEP]
  Succinate → TCA cycle → energy

SSADH DEFICIENCY (ALDH5A1 LOF):
  SSA cannot be converted to succinate → SSA + GHB BOTH ACCUMULATE.
  Alternate SSA reduction pathway (SSAR / AKR7A2 enzyme) converts excess SSA → GHB (4-hydroxybutyric acid).
  GHB is neuroactive: agonist at GABA-B receptors + specific GHB receptors in striatum/cortex.
  SSA is toxic: impairs mitochondrial function, increases ROS, activates glutamate release.

KEY BIOMARKER:
  Urine 4-hydroxybutyric acid (GHB): markedly elevated (10–100× normal) — PATHOGNOMONIC.
  Plasma SSA: elevated (harder to measure, unstable).
  CSF GABA: elevated (GABA degradation blocked upstream).
  CSF GHB: elevated (crosses BBB via MCT transporters).
  Succinic acid: can be elevated (TCA cycle perturbation).

VGB (VIGABATRIN) IS ABSOLUTE CI — UNIQUE MECHANISM:
  VGB inhibits GABA-T (the enzyme UPSTREAM of SSADH in GABA catabolism).
  In SSADH deficiency: SSA is ALREADY failing to be cleared downstream.
  VGB → further GABA-T inhibition → MORE SSA accumulates upstream.
  MORE SSA → MORE GHB (via SSAR/AKR7A2 alternate reduction).
  RESULT: GHB accumulation worsens dramatically → increased GABA-B activation →
  paradoxical worsening of EEG spike burden + increased absence/tonic seizures.
  Clinical reports: VGB in SSADH causes ACUTE SEIZURE WORSENING within 48-72h.
  VGB is the ONLY AED that worsens SSADH by mechanism (not just a general PME concern).
  COMPARE: VGB is Absolute CI for retinal NCL (VGB retinal toxicity); in SSADH it is CI
  by metabolic mechanism — a COMPLETELY DIFFERENT ABSOLUTE CI RATIONALE.

EPIDEMIOLOGY:
  ~350-500 cases reported worldwide (2026); possibly underdiagnosed.
  Incidence estimated ~1:500,000 (Europe); higher in consanguineous populations.
  AR biallelic LOF; ALDH5A1 gene at 6p22.3 (OMIM *610045 gene / #271980 disease).
  First described by Gibson 1983. Pattern of SSADH deficiency in leukocytes.
  Danish/Turkish/Saudi founders described; p.Gly409Glu Danish founder.

CLINICAL PHENOTYPE:
  Intellectual disability: 100% (mild to severe); language delay prominent (often first symptom).
  Epilepsy: ~50% (various types; often treatment-resistant).
  Ataxia: ~60% (cerebellar, gait instability).
  Behavioral: ADHD-like (75%), autism spectrum (50%), OCD traits (40%), aggression/self-injury (35%).
  Hypotonia: ~70% (neonatal/infantile).
  Sleep disturbance: ~65% (non-restorative; GHB is a natural hypnotic at low doses, but elevated GHB → fragmented sleep architecture).
  Seizure types: absence 40%, myoclonic 30%, GTCS 25%, tonic 15%, focal 20%.
  EEG: generalized spike-wave (3 Hz and slow), centrotemporal discharges, excess theta/delta.
  Drug resistance: ~45% of those with epilepsy.
  MRI: T2 hyperintensity in globi pallidi (80% — PATHOGNOMONIC pattern), cerebellar white matter,
       posterior limb internal capsule, cortical atrophy in severe cases.
"""

import random
from datetime import datetime

SEED = 20260819
rng = random.Random(SEED)

def _rng_choice(items): return rng.choice(items)
def _rng_int(lo, hi): return rng.randint(lo, hi)
def _rng_float(lo, hi, dec=2): return round(rng.uniform(lo, hi), dec)

# Colour for ALDH5A1 — deep amber-orange (GABA catabolism / mitochondrial / warm metabolic)
COLOUR = "#e65100"  # deep orange — GABA catabolism / SSA / GHB accumulation / mitochondrial

N = 40  # 40-patient cohort

ETIOLOGIES = [
    {"etiology": "Biallelic LOF — compound het missense+missense (most common European)", "pct": 35, "n": 14},
    {"etiology": "Biallelic LOF — homozygous missense consanguineous (Middle East/South Asia)", "pct": 25, "n": 10},
    {"etiology": "Biallelic LOF — compound het missense+truncating", "pct": 20, "n": 8},
    {"etiology": "Biallelic LOF — p.Gly409Glu homozygous Danish founder", "pct": 10, "n": 4},
    {"etiology": "Biallelic LOF — deep intronic/regulatory (WGS required)", "pct": 7, "n": 3},
    {"etiology": "Single het + CNV (haploinsufficiency incomplete penetrance)", "pct": 3, "n": 1},
]

SEIZURE_TYPES = [
    {"type": "Absence (typical 3 Hz + slow spike-wave)", "pct": 40, "eeg": "3 Hz GSW; slow (<2.5 Hz) GSW in severe; atypical absence in LGS-like",
     "semiology": "Brief 5–30 s staring; subtle automatisms; no post-ictal; frequently missed; VIDEO-EEG critical",
     "tips": "Standard absence EEG (3 Hz GSW) but ALSO see slow < 2.5 Hz GSW; VPA first-line; ETX alone insufficient in SSADH"},
    {"type": "Myoclonic (action + reflex + nocturnal)", "pct": 30, "eeg": "Poly-spike GSW; giant SEP if cortical myoclonus; back-averaging positive",
     "semiology": "Predominantly action myoclonus (writing, eating); morning jerks; sleep-onset myoclonus",
     "tips": "GHB at elevated levels → GABA-B agonism → hyperpolarisation rebound (low-threshold Ca2+ IT) → thalamo-cortical oscillations → myoclonus; VPA + LEV or CLB combination"},
    {"type": "GTCS (secondary generalised)", "pct": 25, "eeg": "Generalized poly-spike then slow wave; post-ictal attenuation",
     "semiology": "Tonic-clonic from generalised onset; fever sensitive; nocturnal predominance",
     "tips": "Drug resistant GTCS in SSADH → consider KD early; VGB ABSOLUTE CI — worsens GHB accumulation"},
    {"type": "Tonic (sleep-related)", "pct": 15, "eeg": "Beta recruitment; EMG artifact; brief tonic bursts in NREM",
     "semiology": "Sudden tonic posturing during NREM sleep; rare diurnal; falls if ambulatory",
     "tips": "CLB or VPA + CLB; KD reduces tonic burden in SSADH (ketone bodies may reduce GHB synthesis)"},
    {"type": "Focal aware / impaired awareness (occipital/temporal)", "pct": 20, "eeg": "Posterior theta/delta; occipital spikes; temporal involvement",
     "semiology": "Visual aura (GHB effect on V1); temporal lobe semiology; rarely secondary GTCS",
     "tips": "Focal semiology in SSADH mimics temporal/occipital epilepsy; WES mandatory in ALL focal epilepsy + ID; GHB globus pallidus MRI clue"},
]

TRIGGERS = [
    {"trigger": "Sleep deprivation", "pct": 72, "mechanism": "GHB sensitisation amplified by sleep disruption; fragmented sleep architecture worsens GHB oscillatory burden"},
    {"trigger": "Fever / intercurrent illness", "pct": 65, "mechanism": "Fever → increased GABA turnover → more SSA → more GHB → threshold drop; VPA hepatotoxicity risk during fever — monitor LFT"},
    {"trigger": "Missed AED dose", "pct": 58, "mechanism": "GHB oscillatory cycle disrupted by AED withdrawal; VPA serum level drop → breakthrough"},
    {"trigger": "Voluntary movement (action myoclonus)", "pct": 48, "mechanism": "Elevated GHB → GABA-B hyper-activation → IT rebound → thalamo-cortical myoclonic loop; action myoclonus specifically triggered by intentional movement"},
    {"trigger": "Stress / emotional arousal", "pct": 45, "mechanism": "Amygdala GHB receptor activation → increased limbic excitability; aggression/anxiety pre-ictal in ~30%"},
    {"trigger": "Glucose fluctuation / fasting", "pct": 38, "mechanism": "TCA cycle perturbation (succinate depletion) worsens during fasting; GHB synthesis from SSA increased when TCA flux falls"},
    {"trigger": "Alcohol exposure (adolescent/adult)", "pct": 22, "mechanism": "Ethanol metabolised to acetaldehyde → SSA accumulation via shared aldehyde dehydrogenase competition; GHB synthesis amplified; EXTREME HAZARD"},
    {"trigger": "VGB or GABA-T inhibitor exposure", "pct": 100, "mechanism": "VGB → GABA-T inhibition → SSA accumulates upstream → MORE GHB via SSAR → acute worsening within 48-72h — ABSOLUTE CONTRAINDICATION"},
]

TREATMENTS = [
    {
        "drug": "Valproic Acid (VPA)",
        "level": "Level A — First-Line Backbone",
        "dose": "30–60 mg/kg/day div q8-12h; target TDM 80–120 mg/L; start low titrate over 4-6 weeks",
        "moa": "Na+-channel + Ca2+ T-type block + GABA-T inhibition (at UPSTREAM step — BUT VPA's GABA-T effect is weaker than VGB and its Na+/Ca2+ effects dominate to net benefit); increases GABA levels but SSADH substrate SSA not dramatically worsened at therapeutic doses",
        "efficacy": "50-60% ≥50% seizure reduction in SSADH (absence + myoclonic); gold standard backbone despite theoretical GABA-T concern",
        "monitoring": "LFT every 3 months (SSADH has baseline mitochondrial vulnerability; VPA mitotoxicity risk elevated; NH3 quarterly; VPA TDM monthly until stable; POLG1 exclusion MANDATORY pre-VPA (CPIC-A)",
        "ssadh_note": "VPA vs VGB: VPA has WEAK GABA-T inhibition at therapeutic doses; clinical benefit outweighs theoretical SSA accumulation in most patients; VGB has IRREVERSIBLE STRONG GABA-T inhibition — categorically different. POLG1 MANDATORY before VPA."
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — First-Line Adjunct",
        "dose": "20–60 mg/kg/day div q12h; target TDM 20–40 mg/L; IV 40 mg/kg SE bolus",
        "moa": "SV2A modulator; reduces vesicular GABA/glutamate release without affecting GABA catabolism pathway; no SSA/GHB interaction",
        "efficacy": "40-50% ≥50% reduction; myoclonic and GTCS; does not worsen SSADH pathophysiology",
        "monitoring": "FBC monthly (thrombocytopenia rare); serum creatinine (renal clearance); behaviour assessment monthly (irritability 15–25% in ID patients — use brivaracetam if behavioural toxicity)",
        "ssadh_note": "SV2A mechanism is safe in SSADH — no GABA-T or aldehyde dehydrogenase interaction. Preferred IV agent for SE over fosphenytoin."
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B — Adjunct (Absence + Myoclonic + Tonic)",
        "dose": "0.1–0.5 mg/kg/day div q12-24h; max 40 mg/day adult; nocturnal dosing for sleep-related tonic",
        "moa": "GABA-A positive allosteric modulator (α2/α3 selective); enhances chloride conductance; reduces tonic inhibition via cortical and striatal GABA-A",
        "efficacy": "50-65% ≥50% reduction; particularly effective tonic (nocturnal) + myoclonic; tolerance develops at 12-18 months (drug holiday protocol)",
        "monitoring": "Sedation/cognition monthly; tolerance assessment 6-monthly; taper slowly (10%/week) to avoid withdrawal seizures",
        "ssadh_note": "GABA-A enhancement is distinct from GABA catabolism pathway — CLB does not affect SSADH or GABA-T; safe mechanistic choice in SSADH."
    },
    {
        "drug": "Ethosuximide (ETX) — Absence Adjunct",
        "level": "Level B — Absence Add-On",
        "dose": "15–40 mg/kg/day div q8-12h; target TDM 40–100 mg/L",
        "moa": "T-type Ca2+ channel blocker (Cav3.1/3.2/3.3); reduces thalamo-cortical oscillatory burst firing — mechanism of absence generation",
        "efficacy": "30-40% ≥50% absence reduction; adjunct to VPA; not monotherapy (SSADH has mixed seizure types beyond absence)",
        "monitoring": "FBC (aplastic anaemia rare but serious); GI tolerance (take with food); LFT 3-monthly; TDM monthly until stable",
        "ssadh_note": "IT channel block is upstream of GABA catabolism; ETX safe in SSADH; complements VPA for absence; add when absence persists despite VPA"
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — Early DRE (4:1 or MAD)",
        "dose": "4:1 ratio (fat:carb+protein) or Modified Atkins Diet 20 g/day carb; minimum 6-month trial",
        "moa": "Ketone bodies (β-hydroxybutyrate, acetoacetate) reduce SSA/GHB cycling: (1) BHB inhibits SSAR (AKR7A2) reducing GHB synthesis from SSA; (2) BHB is an HDAC inhibitor modulating ALDH5A1 expression and compensatory pathways; (3) ketone bodies restore TCA cycle carbon skeletons (succinate supplementation effect)",
        "efficacy": "55% ≥50% reduction in open-label SSADH series (Böhringer 2018; Pearl 2021); reduces globus pallidus GHB signal on MRS; reduces urinary GHB excretion",
        "monitoring": "BHB target 2–4 mmol/L fasting; lipid panel 3-monthly; renal stone screen (uric acid, calcium, citrate); growth 6-monthly; liver enzymes 3-monthly",
        "ssadh_note": "KD is DISEASE-SPECIFIC in SSADH beyond general anticonvulsant effect: BHB inhibits SSAR/AKR7A2 → reduces GHB biosynthesis from SSA → directly lowers pathogenic GHB. KD is mechanistically rational as PRECISION treatment, not just last-resort anticonvulsant."
    },
    {
        "drug": "N-Acetyl-L-cysteine (NAC) — Experimental",
        "level": "Level C — Experimental Antioxidant (Off-Label)",
        "dose": "600–2400 mg/day adult; 30–70 mg/kg/day child; oral q8-12h",
        "moa": "Glutathione precursor; SSA accumulation → increased ROS + mitochondrial oxidative stress; NAC restores glutathione → reduces SSA/GHB-induced oxidative damage; may reduce neuroinflammation",
        "efficacy": "Limited data; case series suggest behavioural improvement (ADHD-like features, aggression); minimal anticonvulsant effect alone; adjunct to VPA+KD",
        "monitoring": "GI tolerance (nausea common); serum cysteine; no serious toxicity at standard doses; avoid in renal failure",
        "ssadh_note": "Mechanistic rationale: elevated SSA → mitochondrial oxidative stress (SSA is a reactive aldehyde directly damaging ETC complex I/II); NAC addresses pathogenic oxidative component, not just seizure suppression."
    },
    {
        "drug": "Taurine — Experimental",
        "level": "Level C — Experimental",
        "dose": "50–100 mg/kg/day; oral q8h",
        "moa": "GHB/GABA-B receptor antagonism at high concentrations; endogenous inhibitory amino acid; may reduce GHB receptor activation by competitive mechanism at striatal GHB receptors",
        "efficacy": "Animal model data (SSADH knockout mouse: Gupta 2004); human case reports anecdotal improvement in ataxia + myoclonus; no RCT",
        "monitoring": "Generally well-tolerated; GI tolerance; no specific monitoring required",
        "ssadh_note": "GHB receptors in striatum/basal ganglia are pharmacologically distinct from GABA-B; taurine may competitively reduce GHB receptor activation; experimental only — requires dedicated SSADH trial."
    },
    {
        "drug": "Vigabatrin (VGB) — ABSOLUTE CONTRAINDICATION",
        "level": "ABSOLUTE CI — NEVER USE",
        "dose": "CONTRAINDICATED — DO NOT PRESCRIBE",
        "moa": "VGB irreversibly inhibits GABA-T; in SSADH deficiency: MORE SSA accumulates upstream → MORE GHB via SSAR → ACUTE SEIZURE WORSENING within 48-72h; permanent GABA-T inhibition cannot be reversed",
        "efficacy": "HARMFUL — worsens GHB accumulation; case reports of ACUTE CATASTROPHIC WORSENING in SSADH deficiency",
        "monitoring": "NOT APPLICABLE — CONTRAINDICATED",
        "ssadh_note": "VGB ABSOLUTE CI UNIQUE MECHANISM IN SSADHA: NOT about retinal toxicity (which is the VGB CI in NCL diseases). In SSADH: VGB → GABA-T block → SSA accumulates → SSAR (AKR7A2) diverts MORE SSA → GHB → acute worsening. Medical alert must list VGB alongside contraindicated drugs. ER note MANDATORY: 'VGB ABSOLUTE CI — SSADH deficiency'. This is the ONLY AED where the CI is due to direct metabolic pathway aggravation."
    },
]

CONTRAINDICATIONS = [
    {"drug": "VGB (Vigabatrin)", "level": "ABSOLUTE CI — METABOLIC WORSENING",
     "reason": "Inhibits GABA-T → SSA accumulates → SSAR diverts MORE SSA to GHB → acute seizure worsening within 48–72h; UNIQUE metabolic CI mechanism (NOT retinal toxicity CI); emergency care plan MANDATORY"},
    {"drug": "PHT/Fosphenytoin", "level": "RELATIVE CI — SE Management",
     "reason": "Na+ channel blocker; not first-line for SSADH myoclonus; IV fosphenytoin is CI in SE — use IV LEV 60 mg/kg instead; PHT worsens myoclonic + tonic seizures in mixed SSADH phenotype"},
    {"drug": "CBZ / OXC / LTG (Na-channel blockers)", "level": "HIGH RISK — Myoclonic Worsening",
     "reason": "Na-channel blockers worsen myoclonic + absence seizures; SSADH has myoclonic component in 30%; CBZ/OXC/LTG documented to worsen generalised myoclonus; AVOID if myoclonic feature present"},
    {"drug": "Alcohol", "level": "ABSOLUTE HAZARD",
     "reason": "Ethanol competes with SSA for aldehyde dehydrogenase → aldehyde backpressure → more GHB synthesis; even small alcohol doses → acute GHB surge; medical alert: alcohol ABSOLUTELY PROHIBITED"},
    {"drug": "GHB / Sodium Oxybate (Xyrem)", "level": "ABSOLUTE CI",
     "reason": "GHB accumulation is the pathogenic mechanism in SSADH; exogenous GHB supplementation (used for narcolepsy) would directly augment toxic GHB levels; absolutely contraindicated"},
    {"drug": "Phenobarbital (monotherapy)", "level": "RELATIVE CI — Sedation + Behaviour",
     "reason": "PB excessive sedation in SSADH patients with baseline cognitive impairment; behaviour worsening reported; VPA preferred if barbiturate class needed; PB acceptable for severe refractory SE as last resort"},
    {"drug": "VPA (if POLG1+ found)", "level": "ABSOLUTE CI (POLG1+ patients)",
     "reason": "POLG1/MERRF must be excluded before VPA in all SSADH patients; biallelic POLG1 + VPA = Alpers-Huttenlocher fatal hepatotoxicity; turnaround 7–14 days; bridge with LEV + CLB"},
]

MONITORING = [
    {"parameter": "ALDH5A1 WES/gene panel (biallelic variants)", "frequency": "At diagnosis (confirmatory)", "target": "Biallelic pathogenic/likely pathogenic variants; ACMG classification; functional studies if VUS"},
    {"parameter": "Urine 4-hydroxybutyric acid (GHB) — GC-MS", "frequency": "Every 6 months", "target": "Baseline + on treatment; GHB normalisation correlates with clinical improvement; KD can reduce by 40-60%"},
    {"parameter": "Plasma/urine amino acids (succinic acid)", "frequency": "Every 6 months", "target": "Succinic acid elevation supports diagnosis; alanine for mitochondrial function"},
    {"parameter": "CSF GABA + GHB (if clinically indicated)", "frequency": "At diagnosis; repeat if clinical worsening", "target": "CSF GABA >1000 nmol/L + CSF GHB elevation confirms SSADH; LP sedation risk — plan carefully"},
    {"parameter": "EEG (routine + video-EEG)", "frequency": "Annual + urgent for clinical change", "target": "GSW frequency; absence burden; focal features; back-averaging if action myoclonus suspected"},
    {"parameter": "MRI brain (3T) with GRE/SWI", "frequency": "At diagnosis; every 2–3 years", "target": "Globus pallidus T2 hyperintensity (80% SSADH — pathognomonic); cerebellar WM; cortical atrophy progression"},
    {"parameter": "Brain MRS (1H spectroscopy)", "frequency": "At diagnosis; annually on KD", "target": "GHB peak at 2.41 ppm (pathognomonic in vivo GHB biomarker); succinate at 2.4 ppm; KD reduces GHB MRS signal"},
    {"parameter": "POLG1 WES / MERRF exclusion pre-VPA", "frequency": "MANDATORY before VPA initiation", "target": "Biallelic POLG1 variants → VPA ABSOLUTE CI; MERRF (m.8344A>G) exclusion; 7–14 day turnaround"},
    {"parameter": "VPA TDM + LFT + NH3", "frequency": "Monthly (initiation); every 3 months (stable)", "target": "VPA TDM 80–120 mg/L; ALT/AST <3× ULN; NH3 <50 µmol/L; SSADH mitochondrial vulnerability increases VPA hepatotoxicity risk"},
    {"parameter": "Cognitive/behavioural assessment (Vineland ABAS)", "frequency": "Annual", "target": "Adaptive behaviour; ID severity; ADHD-like features; ASD screening; aggression/SIB frequency"},
    {"parameter": "Sleep study (polysomnography)", "frequency": "At diagnosis if sleep disturbance; annually if severe", "target": "Sleep architecture (GHB disrupts NREM/REM); circadian rhythm; NREM tonic seizures identified"},
    {"parameter": "KD monitoring (BHB, lipids, growth, renal stone)", "frequency": "Monthly (initiation 3 months); every 3 months (stable)", "target": "BHB 2–4 mmol/L fasting; LDL <200 mg/dL; no kidney stone (uric acid, Ca-oxalate); growth z-score stable"},
    {"parameter": "Urine GHB (on KD — efficacy monitor)", "frequency": "Every 3 months on KD", "target": "GHB reduction ≥30% from baseline at 6 months = KD responsive; GHB MRS reduction parallels clinical improvement"},
    {"parameter": "SUDEP risk assessment", "frequency": "Annual", "target": "DRE patients + nocturnal seizures + sleep-related tonic = highest SUDEP risk; nocturnal alarm systems (NightWatch, SAMi-3); prone sleep position avoided"},
]

LIFECYCLE = [
    {"stage": "Prenatal / Newborn", "age": "Prenatal–0 months", "description": "Neonatal screening not routine for SSADH; sibling of known case → amniocyte SSADH enzyme assay or fetal DNA; hypotonia may be first sign; elevated urine GHB on expanded NBS organic acid screen (GC-MS) if ordered"},
    {"stage": "Infancy / Early Language Delay", "age": "0–18 months", "description": "Hypotonia (70%); language delay first red flag (average first word: 18–24 months vs 12); motor milestones mildly delayed; rarely seizures at this age; early referral to metabolic + neurology"},
    {"stage": "Toddler / Pre-school (Behavioural Phase)", "age": "18 months–5 years", "description": "ADHD-like behaviour dominates (75%); ASD features emerge (50%); aggression/SIB (35%); OCD traits (40%); sleep disorder worsens; epilepsy onset in ~30% at this stage; frequent misdiagnosis as ADHD or autism only"},
    {"stage": "School Age (Epilepsy + Cognitive Phase)", "age": "5–12 years", "description": "Epilepsy onset in majority (absence + myoclonic); cognitive plateau or mild decline; ASD consolidates; DRE emerges in ~45% of epileptic patients; KD initiation most effective at this stage; SSADH diagnosis often made here"},
    {"stage": "Adolescence (Drug Resistance Phase)", "age": "12–18 years", "description": "DRE management; puberty may alter seizure frequency; alcohol/substance exposure HAZARD; behavioural challenges peak (aggression, depression, SIB); SUDEP risk monitoring begins; KD adherence challenge"},
    {"stage": "Adulthood (Chronic Management)", "age": "18+ years", "description": "Chronic DRE management; community integration (supported); psychiatric comorbidities; medication adjustment (renal/hepatic changes); genetic counselling (25% sibling recurrence); research participation (gene therapy trials planned)"},
]

CONCEPTS = [
    {"term": "ALDH5A1 / SSADH / 6p22.3", "definition": "Succinic Semialdehyde Dehydrogenase; mitochondrial matrix NAD+-dependent oxidase; FINAL STEP of GABA catabolism; LOF → SSA + GHB accumulate; AR biallelic LOF; OMIM *610045 gene / #271980 disease"},
    {"term": "4-Hydroxybutyric Acid (GHB)", "definition": "Pathogenic metabolite in SSADH deficiency; produced by SSAR (AKR7A2) from SSA; GABA-B receptor agonist + specific striatal GHB receptor agonist; neuroactive; urine GHB is the PATHOGNOMONIC biomarker (GC-MS)"},
    {"term": "Succinic Semialdehyde (SSA)", "definition": "Intermediate in GABA catabolism; accumulates in SSADH deficiency; reactive aldehyde; mitochondrial toxin; increases ROS; impairs ETC complex I/II; drives GHB synthesis via SSAR"},
    {"term": "VGB ABSOLUTE CI — UNIQUE METABOLIC MECHANISM", "definition": "VGB inhibits GABA-T (upstream enzyme); MORE SSA accumulates; SSAR diverts MORE SSA → GHB; acute worsening within 48–72h; NOT retinal toxicity; DIFFERENT from VGB CI in NCL diseases"},
    {"term": "Globus Pallidus T2 Hyperintensity", "definition": "80% of SSADH patients; most specific MRI finding; GHB accumulates in striatum/GP via specific GHB receptors; pallidal signal on T2/FLAIR; pathognomonic when combined with urine GHB elevation"},
    {"term": "Brain MRS GHB Peak (2.41 ppm)", "definition": "In vivo GHB signal by proton MRS at 2.41 ppm; pathognomonic for SSADH deficiency; reduces with KD treatment; non-invasive monitoring of GHB brain burden; LP-free treatment response assessment"},
    {"term": "GABA-B vs GHB Receptor — Dual Mechanism", "definition": "GHB activates GABA-B receptors (Gi/o-coupled; presynaptic inhibition) AND specific GHB receptors in striatum (positively coupled to cAMP); GABA-B activation → IT current rebound → thalamo-cortical oscillations → absence/myoclonus"},
    {"term": "Ketogenic Diet — Mechanistic Precision", "definition": "BHB inhibits SSAR (AKR7A2) → reduces GHB synthesis from SSA; BHB restores TCA cycle carbons (succinate-sparing); MRS shows GHB peak reduction on KD; disease-specific mechanism beyond general anticonvulsant effect"},
    {"term": "SSAR / AKR7A2 — Alternative SSA Reductase", "definition": "Succinic Semialdehyde Reductase; aldo-keto reductase; converts SSA → GHB; physiological minor pathway; becomes DOMINANT in SSADH deficiency when SSA cannot be cleared; KD inhibits SSAR"},
    {"term": "POLG1 Exclusion Before VPA", "definition": "Biallelic POLG1 coexistence possible; POLG1 + VPA → Alpers fatal hepatotoxicity; SSADH patients have mitochondrial vulnerability (SSA-induced ROS); POLG1/MERRF exclusion MANDATORY (CPIC-A)"},
    {"term": "Alcohol Absolute Hazard", "definition": "Ethanol competes with SSA for aldehyde dehydrogenase (including ALDH2/ALDH1A1); SSA aldehyde backpressure → more SSAR diversion → acute GHB surge; even small doses → acute worsening; medical alert: alcohol PROHIBITED"},
    {"term": "Sleep Disruption — GHB Paradox", "definition": "GHB at pharmacological doses is a hypnotic (Xyrem for narcolepsy); in SSADH, endogenous GHB at toxic levels → paradoxically FRAGMENTED sleep (NREM suppression at high concentrations); nocturnal tonic seizures in NREM 2-3"},
    {"term": "NAC — Antioxidant Rationale", "definition": "SSA is reactive aldehyde → mitochondrial oxidative stress; NAC restores glutathione; addresses oxidative component; behavioural improvement reported; experimental adjunct"},
    {"term": "SSADH Gene Therapy — Horizon", "definition": "AAV9-ALDH5A1 intrathecal/IV gene replacement under development (SSADH mouse model data); SSADH KO mouse = well-validated model; human trials not yet initiated 2026; contact SSADH research registries"},
    {"term": "Diagnostic Pipeline", "definition": "Suspected ID + epilepsy + ataxia + behaviour: (1) Urine organic acids GC-MS → GHB peak; (2) SSADH enzyme assay leukocytes; (3) ALDH5A1 gene sequencing; (4) Brain MRS for GHB peak; (5) MRI GP T2"},
]

THRESHOLDS = [
    {"parameter": "Urine GHB (normal)", "value": "<1 mmol/mol creatinine", "clinical": "Normal GABA catabolism; SSADH intact"},
    {"parameter": "Urine GHB (SSADH deficiency)", "value": "100–1000 mmol/mol creatinine", "clinical": "10–100× normal; pathognomonic elevation; diagnosis confirmed"},
    {"parameter": "SSADH enzyme activity (leukocytes)", "value": "<20% residual", "clinical": "Classic SSADH deficiency; rare attenuated 20-50%"},
    {"parameter": "CSF GABA (normal)", "value": "<150 nmol/L", "clinical": "Normal GABA catabolism"},
    {"parameter": "CSF GABA (SSADH deficiency)", "value": ">500 nmol/L (often >1000)", "clinical": "GABA catabolism blocked; accumulation in CSF"},
    {"parameter": "VPA TDM target", "value": "80–120 mg/L", "clinical": "Seizure suppression; above 120 → toxicity risk in SSADH (mitochondrial vulnerability)"},
    {"parameter": "KD BHB target (fasting)", "value": "2–4 mmol/L", "clinical": "Ketosis adequate for SSAR inhibition; >4 mmol/L not more effective"},
    {"parameter": "Globus Pallidus T2 signal (MRI)", "value": "Bilateral symmetric hyperintensity", "clinical": "Pathognomonic SSADH pattern (80%); correlates with GHB accumulation"},
    {"parameter": "Brain MRS GHB peak", "value": "2.41 ppm (pathognomonic)", "clinical": "In vivo GHB confirmation; reduction under KD = treatment response"},
    {"parameter": "POLG1 variant alleles (pre-VPA)", "value": "Zero biallelic pathogenic variants", "clinical": "Safe to initiate VPA; if biallelic POLG1 found → VPA ABSOLUTE CI"},
    {"parameter": "NH3 on VPA (threshold for action)", "value": ">80 µmol/L", "clinical": "Hyperammonaemia from VPA; carnitine supplement + VPA dose reduction; >120 → consider VPA discontinuation"},
    {"parameter": "Seizure-free target", "value": "12 months seizure-free for IEP/driving", "clinical": "Standard epilepsy threshold; DRE patients unlikely to reach without KD+polytherapy"},
]

def _patient_profile(i):
    pat_id = f"SSADH-{i+1:03d}"
    sex = "F" if rng.random() < 0.5 else "M"
    onset_age_mo = _rng_int(12, 96)  # 1-8 years
    etiology = _rng_choice(ETIOLOGIES)["etiology"]
    has_epilepsy = rng.random() < 0.50
    seizure_type = _rng_choice(SEIZURE_TYPES)["type"] if has_epilepsy else None
    dre = has_epilepsy and rng.random() < 0.45
    has_ataxia = rng.random() < 0.60
    has_id = True  # 100%
    id_severity = _rng_choice(["Mild", "Moderate", "Severe"])
    has_asd = rng.random() < 0.50
    has_adhd_like = rng.random() < 0.75
    on_vpa = rng.random() < 0.80
    on_kd = rng.random() < 0.35
    on_lev = rng.random() < 0.60
    on_clb = rng.random() < 0.45
    urine_ghb = _rng_float(80, 900, 0)  # mmol/mol creatinine
    polg1_screened = rng.random() < 0.90
    confidence = _rng_float(0.65, 0.99)
    return {
        "patient_id": pat_id,
        "sex": sex,
        "onset_age_months": onset_age_mo,
        "etiology": etiology[:60],
        "has_epilepsy": has_epilepsy,
        "seizure_type": seizure_type,
        "drug_resistant": dre,
        "ataxia": has_ataxia,
        "id_severity": id_severity,
        "asd": has_asd,
        "adhd_like": has_adhd_like,
        "on_vpa": on_vpa,
        "on_kd": on_kd,
        "on_lev": on_lev,
        "on_clb": on_clb,
        "urine_ghb_mmol_cr": float(urine_ghb),
        "polg1_screened": polg1_screened,
        "confidence": confidence,
    }

PATIENTS = [_patient_profile(i) for i in range(N)]

def get_overview():
    n_epilepsy = sum(1 for p in PATIENTS if p["has_epilepsy"])
    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_ataxia = sum(1 for p in PATIENTS if p["ataxia"])
    n_asd = sum(1 for p in PATIENTS if p["asd"])
    n_adhd = sum(1 for p in PATIENTS if p["adhd_like"])
    n_on_kd = sum(1 for p in PATIENTS if p["on_kd"])
    n_polg1_screened = sum(1 for p in PATIENTS if p["polg1_screened"])
    avg_ghb = round(sum(p["urine_ghb_mmol_cr"] for p in PATIENTS) / N, 1)
    return {
        "dashboard": "ALDH5A1 Epilepsy — Succinic Semialdehyde Dehydrogenase Deficiency (SSADHA) / 4-Hydroxybutyric Aciduria",
        "gene": "ALDH5A1 (6p22.3) — Succinic Semialdehyde Dehydrogenase; mitochondrial NAD+-dependent aldehyde dehydrogenase; 535 aa ~59 kDa; ALDH superfamily class 5; final step of GABA catabolism (SSA → succinate); PTS1-like import signal for mitochondrial matrix localisation",
        "inheritance": "Autosomal Recessive (AR) biallelic LOF; pLI~0.0; consanguinity enriched; ~350-500 cases worldwide 2026",
        "omim": "*610045 (ALDH5A1 gene); #271980 (Succinic Semialdehyde Dehydrogenase Deficiency / 4-Hydroxybutyric Aciduria)",
        "cohort_size": N,
        "female_n": sum(1 for p in PATIENTS if p["sex"] == "F"),
        "female_pct": round(100 * sum(1 for p in PATIENTS if p["sex"] == "F") / N),
        "n_epilepsy": n_epilepsy,
        "epilepsy_pct": round(100 * n_epilepsy / N),
        "n_dre": n_dre,
        "dre_pct": round(100 * n_dre / N),
        "n_ataxia": n_ataxia,
        "ataxia_pct": round(100 * n_ataxia / N),
        "n_asd": n_asd,
        "asd_pct": round(100 * n_asd / N),
        "n_adhd_like": n_adhd,
        "adhd_like_pct": round(100 * n_adhd / N),
        "n_on_kd": n_on_kd,
        "on_kd_pct": round(100 * n_on_kd / N),
        "n_polg1_screened": n_polg1_screened,
        "polg1_screened_pct": round(100 * n_polg1_screened / N),
        "avg_urine_ghb": avg_ghb,
        "etiologies": ETIOLOGIES,
        "per_patient_kpis": sorted(
            PATIENTS,
            key=lambda p: p["urine_ghb_mmol_cr"],
            reverse=True
        ),
    }


def get_breakdown():
    seizure_counts = {}
    for p in PATIENTS:
        if p["seizure_type"]:
            t = p["seizure_type"].split(" ")[0]
            seizure_counts[t] = seizure_counts.get(t, 0) + 1

    id_dist = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in PATIENTS:
        id_dist[p["id_severity"]] += 1

    ghb_ranges = [(0, 200, "0–200"), (200, 400, "200–400"), (400, 600, "400–600"), (600, 900, "600–900")]
    ghb_hist = []
    for lo, hi, label in ghb_ranges:
        cnt = sum(1 for p in PATIENTS if lo <= p["urine_ghb_mmol_cr"] < hi)
        ghb_hist.append({"range": f"{label} mmol/mol Cr", "n": cnt, "pct": round(100 * cnt / N)})

    onset_ranges = [(0, 24, "0–24 mo"), (24, 48, "24–48 mo"), (48, 72, "48–72 mo"), (72, 120, "72+ mo")]
    onset_hist = []
    for lo, hi, label in onset_ranges:
        cnt = sum(1 for p in PATIENTS if lo <= p["onset_age_months"] < hi)
        onset_hist.append({"range": label, "n": cnt, "pct": round(100 * cnt / N)})

    treatment_counts = {
        "VPA": sum(1 for p in PATIENTS if p["on_vpa"]),
        "LEV": sum(1 for p in PATIENTS if p["on_lev"]),
        "CLB": sum(1 for p in PATIENTS if p["on_clb"]),
        "KD": sum(1 for p in PATIENTS if p["on_kd"]),
    }

    return {
        "seizure_type_distribution": SEIZURE_TYPES,
        "trigger_distribution": TRIGGERS,
        "id_severity_distribution": [
            {"severity": k, "n": v, "pct": round(100 * v / N)} for k, v in id_dist.items()
        ],
        "urine_ghb_histogram": ghb_hist,
        "onset_age_histogram": onset_hist,
        "treatment_counts": treatment_counts,
        "per_patient_profiles": PATIENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "title": "ALDH5A1 / Succinic Semialdehyde Dehydrogenase Deficiency — Definitions, Pathway, Pharmacology",
        "gene_card": {
            "gene": "ALDH5A1",
            "locus": "6p22.3",
            "protein": "Succinic Semialdehyde Dehydrogenase (SSADH)",
            "size": "535 aa, ~59 kDa",
            "family": "ALDH superfamily, class 5",
            "structure": "Mitochondrial matrix NAD+-dependent aldehyde dehydrogenase; homotetrameric; Rossmann fold",
            "reaction": "Succinic Semialdehyde (SSA) + NAD+ → Succinate + NADH + H+ (irreversible in vivo)",
            "localisation": "Mitochondrial matrix (N-terminal MTS targeting sequence)",
            "omim_gene": "*610045",
            "omim_disease": "#271980",
            "inheritance": "AR biallelic LOF",
        },
        "pathway": {
            "name": "GABA Catabolism Pathway",
            "steps": [
                {"step": 1, "enzyme": "GAD (GAD1/GAD2)", "gene": "GAD1/GAD2", "reaction": "Glutamate → GABA + CO₂ (decarboxylation)", "clinical": "GAD1/GAD2 LOF → GABA synthesis defect; distinct from SSADH"},
                {"step": 2, "enzyme": "GABA Transaminase (GABA-T)", "gene": "ABAT", "reaction": "GABA + α-ketoglutarate → SSA + glutamate (transamination)", "clinical": "ABAT LOF → GABA ↑↑; VGB inhibits GABA-T (therapeutic in ABAT LOF; CI in SSADH)"},
                {"step": 3, "enzyme": "SSADH (THIS ENZYME)", "gene": "ALDH5A1", "reaction": "SSA + NAD+ → Succinate + NADH", "clinical": "ALDH5A1 LOF → SSA + GHB accumulate; SSADH deficiency"},
                {"step": 4, "enzyme": "SSAR (alternate)", "gene": "AKR7A2", "reaction": "SSA + NADPH → GHB (4-hydroxybutyric acid) — alternate reduction pathway", "clinical": "SSAR becomes dominant in SSADH deficiency; GHB is the PATHOGENIC METABOLITE; KD inhibits SSAR"},
            ]
        },
        "biomarkers": [
            {"marker": "Urine 4-hydroxybutyric acid (GHB)", "method": "GC-MS organic acid screen", "reference_range": "<1 mmol/mol creatinine", "ssadh_range": "100–1000 mmol/mol Cr", "notes": "PATHOGNOMONIC; 10–100× normal; first-line screening; non-invasive; affected by KD (GHB reduces ~40–60%)"},
            {"marker": "SSADH enzyme activity (leukocytes)", "method": "Spectrophotometric NADH generation assay", "reference_range": "Normal range lab-dependent", "ssadh_range": "<20% residual activity", "notes": "Confirmatory enzyme assay; available at metabolic reference labs; not available everywhere"},
            {"marker": "Brain MRS GHB peak", "method": "1H proton MRS at 2.41 ppm", "reference_range": "Not detectable", "ssadh_range": "Detectable peak 2.41 ppm", "notes": "In vivo pathognomonic; non-invasive CSF GHB substitute; reduces with KD — treatment response monitoring"},
            {"marker": "CSF GABA", "method": "LC-MS/MS", "reference_range": "<150 nmol/L", "ssadh_range": ">500 nmol/L (often >1000)", "notes": "Requires LP; GABA accumulates (catabolism blocked); not first-line (urine GHB simpler)"},
        ],
        "key_concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "treatments": TREATMENTS,
        "references": [
            "Gibson KM et al. Am J Hum Genet 1983 — First human SSADH deficiency description",
            "Pearl PL et al. Ann Neurol 2003 — SSADH deficiency comprehensive review",
            "Böhringer J et al. J Inherit Metab Dis 2018 — Ketogenic diet in SSADH deficiency",
            "Pearl PL et al. JIMD 2021 — Natural history + treatment outcomes",
            "Cortez MA et al. Ann Neurol 2004 — SSADH knockout mouse model",
            "Gupta M et al. J Pharmacol Exp Ther 2004 — Taurine in SSADH model",
            "OMIM #271980 — Succinic Semialdehyde Dehydrogenase Deficiency",
            "SSADH Deficiency Registry — Patient International Collaborative Network",
        ],
        "differential_diagnosis": [
            {"condition": "ABAT deficiency (GABA-T deficiency)", "distinction": "ABAT LOF: GABA ↑↑ but GHB NORMAL (SSADH intact); SSADH: GABA ↑ + GHB ↑↑; gene sequencing mandatory; VGB is TREATMENT in ABAT, ABSOLUTE CI in SSADH — OPPOSITE pharmacology"},
            {"condition": "Nonketotic Hyperglycinemia (GLDC/AMT/GCSH)", "distinction": "NKH: glycine ↑↑ CSF/plasma; GHB NORMAL; burst-suppression EEG neonatal; SSADH: GHB ↑↑; urine organic acids differentiate immediately"},
            {"condition": "GM1/GM2 Gangliosidosis (GLB1/HEXA/HEXB)", "distinction": "Gangliosidoses: cherry red macular spot + bone changes + vacuolated lymphocytes; GHB NORMAL; lysosomal enzyme assays; SSADH: GHB ↑↑; globus pallidus MRI; no cherry red spot"},
            {"condition": "Mitochondrial disease (MERRF/POLG)", "distinction": "Mitochondrial: lactate ↑, mtDNA mutation or POLG variants; GHB NORMAL; SSADH: GHB ↑↑, succinate ↑; both cause ID + epilepsy + ataxia — VPA CI overlap (POLG; also SSADH has mitochondrial vulnerability)"},
            {"condition": "Angelman syndrome (UBE3A)", "distinction": "Angelman: happy disposition, absent speech, EEG 'notched-delta', 15q11.2 deletion/UPD/mutation; GHB NORMAL; SSADH: GHB ↑↑; CMA/microarray + UBE3A methylation differentiate"},
        ],
    }
