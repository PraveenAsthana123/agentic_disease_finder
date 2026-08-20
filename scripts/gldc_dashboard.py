#!/usr/bin/env python3
"""GLDC (Non-Ketotic Hyperglycinemia / NKH — Classic Type)
Epilepsy Dashboard.

GLDC encodes the P-protein (Glycine Decarboxylase) of the mitochondrial Glycine Cleavage System
(GCS). The GCS is a four-component multi-enzyme complex housed in the mitochondrial matrix:
  • P-protein (GLDC, 1020 aa): pyridoxal phosphate (PLP)-dependent; oxidatively decarboxylates
    glycine; transfers the aminomethyl moiety to the H-protein via a methylamine intermediate
  • H-protein (GCSH, 125 aa): lipoic acid-bearing carrier; accepts aminomethyl group from P-protein
    and shuttles it to T-protein
  • T-protein (AMT, 403 aa): aminomethyltransferase; transfers NH3 from H-protein to THF →
    5,10-methyleneTHF + NH4+ (plus CO2 released at P-protein step)
  • L-protein (GCSL / DLD, shared with pyruvate/alpha-KG DH complexes): lipoamide
    dehydrogenase; regenerates reduced H-protein lipoic acid via NAD+

GCS NET REACTION:
  Glycine + THF + NAD+  →  5,10-methyleneTHF + CO2 + NH4+ + NADH
  (2 molecules of glycine feed 1-carbon units into folate cycle)

GLDC LOF (75–80% of NKH):
  GCS cannot decarboxylate glycine → glycine accumulates in ALL compartments:
  plasma, CSF, urine, brain interstitial fluid.

PATHOPHYSIOLOGY — DUAL GLYCINE RECEPTOR PARADOX:
  Glycine receptor (GlyR, GLRA1/GLRB): Cl− channel (inhibitory) in brainstem / spinal cord /
  reticular formation. In neonates, GlyR predominates brainstem inhibitory tone.

  NMDA receptor (NMDAr, GluN1/GluN2): requires BOTH glutamate (GluN2 site) AND glycine (GluN1
  glycine-binding site, Km ~0.5–5 µM). Glycine is the OBLIGATE CO-AGONIST. At saturating
  glycine concentrations (NKH: CSF glycine often >500 µM vs normal 5–10 µM), the GluN1 site is
  fully occupied → MAXIMUM NMDAr activity → excitotoxic cascade → burst-suppression → seizures.

  NEONATAL PARADOX: Excess glycine first activates brainstem GlyR → PROFOUND HYPOTONIA + APNEA
  (the 'floppy baby' + ventilator requirement). Then, as NMDAr excitotoxicity dominates cortex →
  BURST-SUPPRESSION EEG → seizures escalate.

  HICCUPS PATHOGNOMONIC CLUE:
  The phrenic nerve nucleus (C3–C5, anterior horn) relies on inhibitory glycinergic interneurons.
  Excess glycine → EXCESSIVE GlyR activation at phrenic nucleus → rhythmic, uncontrolled
  diaphragm activation = HICCUPS. Persistent neonatal hiccups + hypotonia + apnea = NKH until
  proven otherwise.

DIAGNOSTIC BIOMARKER — CSF:PLASMA GLYCINE RATIO:
  The RATIO is mandatory — plasma glycine alone insufficient (elevated in many organic acidemias).
  Normal CSF:plasma glycine ratio: <0.02 (often 0.01–0.02)
  NKH: ratio >0.08; severe classic: 0.10–0.60 (often 0.15–0.40)
  Threshold ≥0.08 is highly specific for NKH.
  Simultaneous LP + blood draw (fasting) MANDATORY. CSF and plasma MUST be processed together.

TREATMENT PILLARS:
  1. Sodium Benzoate: conjugates glycine + benzoate → hippuric acid → renal excretion
     Targets: plasma glycine <500 µmol/L (ideally <300 µmol/L)
     Mechanism: depletes the glycine pool available for CNS penetration
  2. Dextromethorphan (DXM): uncompetitive NMDAr antagonist at the Mg2+-binding channel site
     Blocks NMDAr over-activation regardless of glycine concentration
     NOTE: Not a glycine-site antagonist — DXM blocks the channel open state
  3. Felbamate: additional NMDAr antagonist (similar mechanism to DXM) — adjunct for refractory
  4. Standard AEDs: LEV (SV2A modulator — safe, no glycine interaction) + CLB (GABA-A PAM)

EPIDEMIOLOGY:
  ~1:60,000–1:76,000 live births (Europe); ~1:12,000 (Finland – founder effect);
  ~1,200–2,000 cases active worldwide 2026.
  AR biallelic LOF; GLDC gene at 9p24.1 (OMIM *238300 gene / #605899 disease).
  GLDC: ~75–80% of NKH. AMT (~15%). GCSH (~1%). DLD (very rare; not classic NKH phenotype).

CLINICAL PHENOTYPES:
  Classic Neonatal (~80–85%): neonatal onset hours-days; hypotonia + apnea + hiccups;
    burst-suppression EEG → hypsarrhythmia; severe ID in survivors; mortality 30% neonatal.
  Attenuated (~15%): infantile/childhood onset; milder ID; seizures manageable; choreoathetosis;
    p.Gly761Arg (c.2281G>A) founder allele (European) is the commonest attenuated allele.
  Transient (~rare): neonatal glycine elevation normalises spontaneously; enzyme immaturity.

KEY DISTINGUISHER FROM SSADHA: glycine ↑↑ (NKH) vs GHB ↑↑ (SSADH). Completely different
metabolites. Urine organic acids + CSF:plasma glycine ratio differentiate immediately.
"""

import random
from datetime import datetime

SEED = 20260820
rng = random.Random(SEED)

def _rng_choice(items): return rng.choice(items)
def _rng_int(lo, hi): return rng.randint(lo, hi)
def _rng_float(lo, hi, dec=2): return round(rng.uniform(lo, hi), dec)

# Colour: deep indigo-blue — glycine/NMDA/amino acid metabolism
COLOUR = "#283593"  # dark indigo — GCS/glycine/NMDAr biology

N = 40  # 40-patient cohort

ETIOLOGIES = [
    {"etiology": "Classic Neonatal — null/null (both alleles truncating/frameshift)", "pct": 35, "n": 14,
     "csf_plasma_ratio": "0.20–0.50", "outcome": "Severe ID; DRE; many non-ambulatory"},
    {"etiology": "Classic Neonatal — null/severe missense (one allele partial)", "pct": 25, "n": 10,
     "csf_plasma_ratio": "0.15–0.35", "outcome": "Severe neonatal; IS + multifocal; DRE"},
    {"etiology": "Attenuated — compound het with p.Gly761Arg (common European partial)", "pct": 20, "n": 8,
     "csf_plasma_ratio": "0.08–0.15", "outcome": "Mild-moderate ID; seizures manageable; ambulatory possible"},
    {"etiology": "Attenuated — homozygous p.Gly761Arg (Finnish founder — most attenuated)", "pct": 10, "n": 4,
     "csf_plasma_ratio": "0.08–0.12", "outcome": "Mildest phenotype; IQ 40–60; walks; speaks"},
    {"etiology": "Classic Neonatal — consanguineous homozygous missense (Middle East/South Asia)", "pct": 7, "n": 3,
     "csf_plasma_ratio": "0.18–0.45", "outcome": "Severe; neonatal ICU; high mortality"},
    {"etiology": "Transient NKH — neonatal glycine normalises (enzyme immaturity)", "pct": 3, "n": 1,
     "csf_plasma_ratio": "0.09–0.12 (normalising)", "outcome": "Generally benign; monitor closely"},
]

PHENOTYPE_CLASSES = [
    {
        "name": "Classic Neonatal (Severe)", "pct": 60,
        "description": "Onset hours–2 days after birth. Profound hypotonia, apnea (often requiring mechanical ventilation), absent Moro reflex, absent suck reflex, HICCUPS (pathognomonic). EEG: burst-suppression → evolves to hypsarrhythmia. Mortality ~30%. Survivors: severe ID, refractory epilepsy, no meaningful development. GLDC null/null or null/severe-missense.",
        "eeg": "Burst-suppression (neonatal) → hypsarrhythmia → multifocal spikes → electrical SE",
        "seizure_types": "Myoclonic, focal clonic, tonic, electrical SE — very frequent",
        "outcome": "Severe ID in all survivors; DRE in >80%; non-ambulatory; non-verbal",
        "csf_ratio": ">0.15 (often 0.20–0.50)"
    },
    {
        "name": "Attenuated (Mild-Moderate)", "pct": 15,
        "description": "Later onset (weeks–years). Less severe ID (IQ range 20–80+). Seizures in ~50% — manageable. Chorea/choreoathetosis in ~40%. Some ambulation and speech possible. Common with p.Gly761Arg allele (European) or other partial-function alleles. CSF:plasma ratio typically 0.08–0.15.",
        "eeg": "Multifocal or generalised spikes; may be subtle in mild attenuated; no burst-suppression",
        "seizure_types": "Focal, myoclonic, absence — lower frequency than classic; sodium benzoate + DXM often sufficient",
        "outcome": "Variable; mildest (homozygous p.Gly761Arg): walks, communicates, IQ ~40–60",
        "csf_ratio": "0.08–0.15 (borderline-moderate elevation)"
    },
    {
        "name": "Transient NKH (Rare)", "pct": 5,
        "description": "Neonatal glycine elevation that spontaneously normalises by 2–8 weeks. Enzyme immaturity hypothesised. Outcome generally benign but must monitor — some develop late classic NKH phenotype. Re-test CSF:plasma ratio at 8 weeks minimum.",
        "eeg": "Transient burst-suppression → normalises",
        "seizure_types": "Transient neonatal seizures; resolve as glycine normalises",
        "outcome": "Generally benign; some have subtle developmental delay; long-term seizure-free possible",
        "csf_ratio": "Elevated initially (>0.08) → falls <0.02 by 8 weeks"
    },
]

SEIZURE_TYPES = [
    {"type": "Electrical Status Epilepticus (neonatal SE)", "pct": 60,
     "eeg": "Burst-suppression → continuous spike-wave; electrical SE without motor manifestation",
     "semiology": "Neonatal — EEG-confirmed SE without clinical correlate due to profound hypotonia; easily missed without continuous video-EEG; treat as SE",
     "tips": "LEV IV 60 mg/kg loading dose for SE; phenobarbital second-line; DXM adjunct reduces NMDAr-driven ictal burden; sodium benzoate initiated simultaneously — metabolic approach + AED synergistic"},
    {"type": "Myoclonic seizures (neonatal/infantile)", "pct": 70,
     "eeg": "Poly-spike GSW or poly-spike with burst-suppression background; high-amplitude; back-averaging + in IS period",
     "semiology": "Sudden jerks — often occurring in burst-suppression 'bursts'; myoclonic during IS period; bilateral synchronous in early life",
     "tips": "IS: ACTH Level A; myoclonic component: VPA HIGH RISK — prefer LEV + CLB; avoid CBZ/PHT/OXC (worsens myoclonus); DXM reduces burst-suppression-linked myoclonic burden"},
    {"type": "Tonic seizures (bilateral)", "pct": 40,
     "eeg": "Beta recruitment EMG artifact; brief tonic bursts in NREM; bilateral synchronous",
     "semiology": "Rigid posturing; opisthotonic; bilateral; often nocturnal; falls if ambulatory in attenuated phenotype",
     "tips": "CLB + VPA-alternatives; KD for refractory tonic; avoid PHT/CBZ/OXC"},
    {"type": "Focal clonic (cortical)", "pct": 35,
     "eeg": "Focal spike/sharp-wave; may be multifocal; rhythmic clonic delta",
     "semiology": "Face/limb clonic; may secondarily generalise; often multifocal in neonatal phase — reflects diffuse cortical NMDAr excitotoxicity",
     "tips": "Multifocal focal = metabolic cause; mandates plasma amino acid + CSF glycine; LEV first-line focal; sodium benzoate reduces glycine load driving cortical excitability"},
    {"type": "Infantile Spasms (IS/West Syndrome)", "pct": 45,
     "eeg": "Hypsarrhythmia — classical or modified (inter-spasm suppression + chaotic background + multifocal spikes)",
     "semiology": "Spasm clusters; salaam movements; regressive milestones; peak onset 4–8 months in neonatal survivors",
     "tips": "NKH-associated IS: ACTH Level A; vigabatrin AVOID (VGB raises glycine via GABAergic mechanism — see pharmacology notes); KD adjunct for ACTH-refractory IS in NKH"},
    {"type": "Generalised Tonic-Clonic (GTCS)", "pct": 30,
     "eeg": "Generalised poly-spike then slow-wave; post-ictal attenuation",
     "semiology": "Tonic-clonic from generalised onset; often in attenuated NKH during illness/fever; less common in classic severe",
     "tips": "LEV + CLB combination; sodium benzoate optimisation; KD if GTCS persist; avoid sodium channel blockers"},
]

TRIGGERS = [
    {"trigger": "Intercurrent illness / fever", "pct": 70,
     "mechanism": "Fever → catabolic state → increased protein/amino acid turnover → more glycine generated; glycine clearance limited by impaired GCS → acute glycine surge; risk of SE"},
    {"trigger": "Protein-rich meal / increased amino acid load", "pct": 55,
     "mechanism": "Serine → glycine (interconvertible via SHMT); dietary protein catabolism → glycine load exceeds clearance; plasma glycine spike → CSF glycine rises within hours; pre-meal sodium benzoate dosing important"},
    {"trigger": "Sleep deprivation / disrupted sleep", "pct": 48,
     "mechanism": "NMDAr sensitivity heightened by sleep deprivation; burst-suppression risk increased in NREM sleep; circadian modulation of glycine transport amplifies nighttime glycine CNS burden"},
    {"trigger": "Missed sodium benzoate dose", "pct": 65,
     "mechanism": "Sodium benzoate primary glycine-depleting mechanism; missed dose → glycine reaccumulates within 12–24h → plasma glycine rises → CSF glycine follows → NMDAr reactivation → breakthrough seizures"},
    {"trigger": "VPA / valproate exposure", "pct": 55,
     "mechanism": "VPA inhibits GLDC (the deficient enzyme) at therapeutic concentrations via direct enzyme competition AND raises CSF glycine by impeding GCS residual activity; also elevates plasma glycine by inhibiting N-methylglycine (sarcosine) pathway; clinically worsens seizure control in NKH"},
    {"trigger": "Fasting / prolonged NPO", "pct": 38,
     "mechanism": "Fasting → muscle protein catabolism → serine/glycine released; gluconeogenesis uses glycine carbon skeleton; net glycine liberation exceeds GCS clearance; IV dextrose + sodium benzoate coverage mandatory for NPO in NKH"},
    {"trigger": "Anesthesia / sedation (GlyR interactions)", "pct": 30,
     "mechanism": "Several anesthetic agents modulate glycine and NMDAr; propofol and barbiturates potentiate GlyR; ketamine (NMDAr antagonist) may be BENEFICIAL sedation agent in NKH — pre-operative discussion with metabolic team mandatory"},
]

TREATMENTS = [
    {
        "drug": "Sodium Benzoate",
        "level": "Level A — First-Line (glycine-lowering backbone)",
        "dose": "250–750 mg/kg/day (neonates: start 250 mg/kg/day, titrate); divided q6-8h; IV 500 mg/kg loading in SE; oral maintenance",
        "moa": "Benzoate conjugates with glycine in mitochondria (via glycine N-acyltransferase, GLYAT) → hippuric acid → renal excretion. Each benzoate molecule removes one glycine. Net effect: reduces plasma glycine pool, limiting CNS penetration.",
        "efficacy": "Reduces plasma glycine by 50–80% at therapeutic doses. TARGET: plasma glycine <500 µmol/L (ideally <300 µmol/L; normal <260 µmol/L). DOES NOT correct enzyme defect — ongoing requirement lifelong.",
        "monitoring": "Plasma glycine every 3 months (target <500 µmol/L); annual CSF:plasma ratio; serum carnitine (benzoate conjugation depletes carnitine — supplement carnitine 50–100 mg/kg/day); liver function (hepatotoxicity risk at >500 mg/kg/day); quarterly NH3 (hyperammonaemia risk)",
        "nkh_note": "Sodium benzoate is NKH-SPECIFIC treatment — not a general anticonvulsant. Must be started simultaneously with AED therapy in neonatal NKH. IV formulation for SE management. Carnitine co-supplementation MANDATORY to prevent secondary carnitine depletion from conjugation."
    },
    {
        "drug": "Dextromethorphan (DXM)",
        "level": "Level B — NMDAr Antagonist (NKH-Specific Adjunct)",
        "dose": "Neonates: 5–10 mg/kg/day div q6-8h. Infants/children: 2–15 mg/kg/day. Start low, titrate to EEG response. Maximum 35 mg/kg/day in some series.",
        "moa": "Uncompetitive NMDAr channel blocker at the Mg2+-binding site within the open channel. Blocks NMDAr hyperactivation driven by excess glycine co-agonism at GluN1. Does NOT lower glycine levels — complements sodium benzoate by reducing NMDAr downstream effect.",
        "efficacy": "Reduces burst-suppression burden; reduces IS frequency; improves EEG background in 40–60% of classic neonatal NKH. Best evidence for IS + burst-suppression reduction. Less effective as sole agent; synergistic with sodium benzoate.",
        "monitoring": "Level monitoring not standard (DXM converted to dextrorphan by CYP2D6 — CYP2D6 phenotyping if poor response); sedation assessment; respiratory depression risk in neonates — monitor O2 sat; discontinue if no benefit at 4 weeks",
        "nkh_note": "DXM is the KEY NMDA antagonist adjunct in NKH — its use is NKH-specific (not routine epilepsy practice). CYP2D6 polymorphism affects DXM→dextrorphan conversion: poor metabolizers may have impaired effect; ultra-rapid metabolizers may have toxicity. DXM is a prodrug; dextrorphan is the active NMDAr antagonist."
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — First-Line AED (SE + Maintenance)",
        "dose": "20–60 mg/kg/day div q12h; IV 60 mg/kg loading dose for SE; oral maintenance",
        "moa": "SV2A modulator; reduces presynaptic neurotransmitter release (glutamate + GABA vesicle cycling). No glycine interaction. Safe mechanistic choice in NKH.",
        "efficacy": "30–50% ≥50% seizure reduction; preferred IV agent for SE over phenobarbital (less respiratory depression); myoclonic + focal reduction",
        "monitoring": "FBC (thrombocytopenia rare); serum creatinine (renal clearance); behaviour assessment monthly (irritability 15–25% in ID patients)",
        "nkh_note": "LEV has NO interaction with glycine metabolism or GCS. Preferred over phenobarbital (PB increases respiratory depression on top of GlyR-mediated hypotonia in neonatal NKH). First-line IV agent for SE in NKH."
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B — Adjunct (Myoclonic + Tonic + Focal)",
        "dose": "0.1–0.5 mg/kg/day div q12-24h; max 40 mg/day adult; nocturnal dosing for tonic",
        "moa": "GABA-A PAM (α2/α3-selective benzodiazepine). Enhances Cl− conductance. No glycine catabolism interaction — safe in NKH.",
        "efficacy": "40–55% ≥50% reduction (myoclonic + tonic); tolerance develops at 12–18 months; drug holiday protocol",
        "monitoring": "Sedation/cognition monthly; tolerance at 6 months; taper slowly 10%/week",
        "nkh_note": "CLB GABA-A mechanism is orthogonal to glycine GlyR — does not affect disease pathophysiology but reduces seizure frequency via cortical inhibition. Safe adjunct."
    },
    {
        "drug": "ACTH (Corticotropin) — IS Management",
        "level": "Level A — Infantile Spasms in NKH",
        "dose": "High-dose ACTH: 150 IU/m²/day IM (natural ACTH) or synthetic tetracosactide 0.5–0.75 mg/day x4 weeks then taper",
        "moa": "Reduces ACTH-driven neuroinflammatory cascade in developing brain; reduces hypsarrhythmia; mechanism not fully elucidated for IS but evidence-based",
        "efficacy": "60–70% spasm cessation in IS; NKH-associated IS: combined ACTH + sodium benzoate + DXM — best outcomes. ACTH preferred over VGB in NKH (VGB raises glycine — CI).",
        "monitoring": "BP (hypertension); glucose (hyperglycaemia); electrolytes (hypokalaemia); infection risk; cushingoid features; growth suppression",
        "nkh_note": "ACTH is PREFERRED over VGB for IS in NKH because VGB raises glycine (GABA-T inhibition → GABA↑ → SSADH diversion → NOT the same mechanism but VGB also independently raises CSF glycine in some reports via GABA-glycine co-transport — and VGB-IS evidence is VGB-specific, not NKH-specific). ACTH + benzoate + DXM = NKH IS triple combination."
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — DRE Adjunct (Limited NKH Data)",
        "dose": "4:1 ratio or MAD (20 g carb/day); minimum 3-month trial; ketosis BHB 2–4 mmol/L",
        "moa": "Reduces amino acid catabolism (protein-restricted component); reduces serine→glycine flux (Ser hydroxymethyltransferase, SHMT, requires C1-units from 5,10-methyleneTHF — KD may alter 1C flux); BHB may modulate NMDAr directly",
        "efficacy": "Limited NKH-specific data; case series: 30–40% seizure reduction in DRE NKH; useful adjunct when sodium benzoate + DXM + LEV insufficient",
        "monitoring": "BHB monthly; lipids 3-monthly; growth; renal stone screen; glycine TDM (KD may lower glycine modestly via reduced serine catabolism)",
        "nkh_note": "KD in NKH: modest glycine-lowering by reducing serine→glycine flux (serine contributes ~50% of brain glycine via SHMT). Not as disease-specific as KD in SSADH. Use as last-resort adjunct for DRE when other measures insufficient."
    },
    {
        "drug": "Felbamate (FBM) — NMDAr Adjunct",
        "level": "Level C — Experimental NMDAr Adjunct (Refractory NKH)",
        "dose": "15–45 mg/kg/day div q6-8h; very slow titration (aplastic anaemia risk)",
        "moa": "NMDAr antagonist at glycine-binding site (strychnine-insensitive) + NMDAr channel blocker. DUAL NMDAr mechanism: like DXM (channel) + glycine-site block. Theoretically superior to DXM alone.",
        "efficacy": "Limited NKH data; case reports of improvement in burst-suppression burden; use only after DXM failure due to serious adverse effect profile",
        "monitoring": "MANDATORY: FBC weekly x6 months then monthly (aplastic anaemia — rare but potentially fatal; Black Box Warning); LFT (hepatotoxicity); drug interactions (PHT/CBZ reduce FBM levels)",
        "nkh_note": "FBM glycine-site NMDAr antagonism is mechanistically attractive in NKH but restricted by toxicity profile. Only use under specialist metabolic epilepsy unit after failing DXM + sodium benzoate optimisation. INFORM FAMILY of aplastic anaemia risk."
    },
    {
        "drug": "Valproic Acid (VPA) — HIGH RISK",
        "level": "HIGH RISK — Metabolic Worsening + Standard Contraindications",
        "dose": "AVOID unless no alternative (see notes)",
        "moa": "VPA inhibits residual GLDC activity; raises plasma glycine by impairing GCS-mediated clearance; also inhibits N-methylglycine (sarcosine) pathway (reduces an alternative glycine disposal route); net: WORSENS glycine accumulation",
        "efficacy": "Classically used empirically for seizure suppression, but worsens underlying metabolic defect → increased NMDAr activation → may PARADOXICALLY worsen seizures or encephalopathy",
        "monitoring": "NOT RECOMMENDED. If used (refractory SE, no alternatives): maximum monitoring (LFT weekly; NH3 weekly; plasma glycine monthly; POLG1 exclusion MANDATORY)",
        "nkh_note": "VPA HIGH RISK IN NKH — distinct mechanism from carnitine/POLG CI: VPA directly inhibits GLDC (partial competitive inhibition demonstrated in vitro) AND reduces alternative glycine disposal routes → glycine burden worsens. Avoid in NKH — use LEV + CLB + DXM + benzoate first. If SE uncontrolled: IV LEV → IV PB → IV ketamine (NMDAr antagonist — potentially beneficial) rather than IV VPA."
    },
    {
        "drug": "VGB (Vigabatrin) — HIGH RISK for IS",
        "level": "HIGH RISK — VGB raises CSF glycine in NKH; prefer ACTH for IS",
        "dose": "AVOID for IS in NKH — use ACTH",
        "moa": "VGB inhibits GABA-T → GABA accumulates. GABA and glycine share inhibitory co-transport mechanisms (GlyT2, SLC6A5). VGB-associated GABA elevation may increase CSF glycine by competitive upregulation of glycine reuptake transporters, raising extracellular glycine.",
        "efficacy": "VGB efficacious for IS in general but DISEASE-SPECIFIC RISK in NKH. ACTH is preferred and equally or more effective in IS.",
        "monitoring": "RETINAL TOXICITY (VGB): mandatory visual field + OCT (ERG in pre-verbal) every 3 months regardless; if NKH IS: use ACTH first — avoid VGB",
        "nkh_note": "VGB HIGH RISK IN NKH for IS: mechanism differs from VGB CI in SSADH (where it accumulates SSA→GHB); in NKH concern is VGB → GABA↑ → possible glycine transporter upregulation → CSF glycine rises → worsens NMDAr excitotoxicity. Clinical reports of glycine worsening with VGB exist. ACTH preferred for IS. VGB retinal toxicity is ALSO a concern (independent of glycine). DOUBLE RISK."
    },
]

CONTRAINDICATIONS = [
    {"drug": "VPA (Valproic Acid)", "level": "HIGH RISK — METABOLIC WORSENING",
     "reason": "VPA inhibits GLDC (residual GCS activity) + impairs alternative glycine disposal → RAISES plasma and CSF glycine → worsens NMDAr excitotoxicity → paradoxical seizure worsening or encephalopathy. Also: POLG1 MANDATORY (VPA hepatotoxicity). Standard CI for POLG1+ adds to disease-specific CI.",
     "alternative": "LEV + CLB + DXM + sodium benzoate. IV LEV for SE. Ketamine IV if SE refractory."},
    {"drug": "VGB (Vigabatrin) for IS", "level": "HIGH RISK — PREFER ACTH",
     "reason": "VGB may raise CSF glycine via GABA-glycine transporter interaction. GABA↑ from GABA-T inhibition → competitive pressure on glycine reuptake → extracellular glycine rises. Also: VGB retinal toxicity applies regardless.",
     "alternative": "ACTH (Level A for IS); ACTH + DXM + benzoate combination for NKH-IS."},
    {"drug": "CBZ / OXC / LTG / PHT (Na-channel blockers)", "level": "RELATIVE CI — Myoclonic Worsening",
     "reason": "Na-channel blockers worsen myoclonic seizures in generalised epilepsies. NKH has major myoclonic component (neonatal + IS period). CBZ/PHT documented to worsen generalised myoclonus. Avoid if any myoclonic feature.",
     "alternative": "LEV, CLB, VPA (if unavoidable despite glycine risk), KD for refractory myoclonus."},
    {"drug": "Phenobarbital (PB) — Neonatal / Maintenance", "level": "CAUTION — Respiratory Depression + GlyR Additive",
     "reason": "PB potentiates GABA-A and GlyR. In neonatal NKH: excess glycine already activates GlyR brainstem inhibition causing apnea/hypotonia. PB adds to GlyR-mediated brainstem depression → excessive respiratory depression risk. Use ONLY if LEV failed as second-line for SE.",
     "alternative": "IV LEV 60 mg/kg for SE (primary). IV ketamine (NMDAr antagonist — BENEFICIAL mechanism in NKH) as third-line SE."},
    {"drug": "Protein Restriction (extreme)", "level": "CAUTION — Nutritional",
     "reason": "Extreme protein restriction below RDA impairs growth and neurodevelopment; does not lower glycine sufficiently; sodium benzoate is far more effective for glycine reduction. Protein restriction alone is inadequate and harmful if excessive.",
     "alternative": "Sodium benzoate (metabolic clearance); standard RDA protein intake; carnitine co-supplementation."},
    {"drug": "Fasting / NPO without coverage", "level": "CAUTION — Glycine Surge",
     "reason": "Fasting → muscle catabolism → glycine release → glycine surge without GCS clearance → seizure risk. IV dextrose + sodium benzoate MANDATORY for any NPO period in NKH patients.",
     "alternative": "IV 10% dextrose + oral/IV sodium benzoate coverage; target plasma glycine <500 µmol/L throughout perioperative period."},
]

MONITORING = [
    {"parameter": "GLDC gene sequencing (WES/targeted panel)", "frequency": "At diagnosis (confirmatory)", "target": "Biallelic pathogenic variants; ACMG classification; functional GLDC enzyme assay if VUS; AMT/GCSH if GLDC negative"},
    {"parameter": "CSF:Plasma glycine ratio (simultaneous)", "frequency": "At diagnosis; annually on treatment", "target": "Diagnostic: ratio ≥0.08; target on sodium benzoate: ratio <0.04; plasma glycine target <500 µmol/L (ideally <300 µmol/L)"},
    {"parameter": "Plasma amino acid quantitative (glycine focus)", "frequency": "Every 3 months (sodium benzoate titration)", "target": "Glycine <500 µmol/L; serine, threonine, alanine (natual history markers); folate/methylene-THF perturbation"},
    {"parameter": "Urine organic acids (glycine, hippurate)", "frequency": "Every 3 months on sodium benzoate", "target": "Hippuric acid excretion confirms sodium benzoate effect; glycylglycine absent (GLDC-specific); excludes other organic acidemias"},
    {"parameter": "Serum carnitine (free + acylcarnitine)", "frequency": "Every 3 months", "target": "Free carnitine >20 µmol/L; sodium benzoate → glycyl-benzoate → hippurate depletes carnitine stores; supplement L-carnitine 50–100 mg/kg/day if low"},
    {"parameter": "EEG (video-EEG — continuous in neonatal SE)", "frequency": "Continuous neonatal phase; annual + urgent for clinical change", "target": "Burst-suppression burden; IS detection (EEG-only IS in NKH very common); hypsarrhythmia resolution on ACTH; SE detection"},
    {"parameter": "MRI brain (3T + DWI in acute)", "frequency": "At diagnosis; 6 months; annually in first 3 years", "target": "Periventricular WM hypomyelination; thin/absent corpus callosum (especially splenium); cerebellar hypoplasia; DWI restriction in acute NKH (cytotoxic edema in cortex/basal ganglia)"},
    {"parameter": "POLG1 WES / MERRF exclusion (pre-VPA)", "frequency": "MANDATORY before VPA initiation", "target": "Biallelic POLG1 → VPA ABSOLUTE CI (Alpers); NKH metabolic vulnerability adds to standard POLG1 rule"},
    {"parameter": "Liver function tests + NH3 (sodium benzoate monitoring)", "frequency": "Monthly (first 6 months); quarterly (stable)", "target": "ALT/AST <3× ULN; NH3 <80 µmol/L (benzoate → hippurate consumes carnitine → affects urea cycle at high doses); GGT, ALP"},
    {"parameter": "Developmental / cognitive assessment (Bayley/Vineland)", "frequency": "Every 6 months in first 3 years; annual thereafter", "target": "Motor milestones; language development; adaptive behaviour; Vineland-3 adaptive composite; assess impact of glycine control on developmental trajectory"},
    {"parameter": "Visual assessment (retinal + cortical visual)", "frequency": "Annual", "target": "Cortical visual impairment (CVI) from NMDAr excitotoxicity; ERG if concerned; OCT in older attenuated patients; visual field in ambulatory patients"},
    {"parameter": "CYP2D6 pharmacogenomics (DXM metabolism)", "frequency": "Once (baseline before DXM start)", "target": "CYP2D6 poor metabolizer → DXM accumulates (adjust dose down); ultra-rapid → reduced effect; intermediate: standard dosing"},
    {"parameter": "Growth anthropometry + nutrition (carnitine/folate)", "frequency": "Every 3 months (first 2 years); 6-monthly thereafter", "target": "Weight/height z-score; head circumference; folate levels (GCS is one-carbon metabolism — 5,10-methyleneTHF depleted in NKH); folate supplementation often needed"},
    {"parameter": "SUDEP risk assessment", "frequency": "Annual in DRE patients", "target": "Nocturnal seizures + IS + SE history = highest SUDEP risk in NKH; nocturnal alarm; prone position avoidance; side-car sleeping"},
]

LIFECYCLE = [
    {"stage": "Prenatal / Sibling Screening", "age": "Prenatal", "description": "Known GLDC family: chorionic villus sampling or amniocentesis for GLDC genotyping; GLDC enzyme assay on chorionic villi. Expanded NBS (tandem MS): does NOT detect NKH reliably (glycine not routinely on NBS panels). Prenatal diagnosis essential for known carrier parents."},
    {"stage": "Neonatal Crisis (Classic)", "age": "0–7 days", "description": "NICU admission hours after birth. Apnea (mechanical ventilation), profound hypotonia, absent Moro/suck, HICCUPS (pathognomonic). CSF:plasma glycine ratio STAT. EEG: burst-suppression immediately. START: IV sodium benzoate + IV LEV + DXM oral/NG tube. HIGH MORTALITY RISK in first 2 weeks — aggressive management."},
    {"stage": "Post-acute Neonatal Stabilisation", "age": "2–8 weeks", "description": "Wean ventilator (if survived). Burst-suppression persists → transitions to multifocal epilepsy/hypsarrhythmia. Oral sodium benzoate titration. DXM dose optimisation. MRI: WM hypomyelination, thin CC. POLG1 exclusion. Genetic counselling."},
    {"stage": "Infantile Phase (IS / Epilepsy)", "age": "3–18 months", "description": "IS onset in survivors (40–50%): hypsarrhythmia + spasms. ACTH x4 weeks. NKH-IS: ACTH + benzoate + DXM triple therapy. Developmental regression. Feeding difficulties (NG/PEG). Physiotherapy, OT, speech therapy. Seizure types expand: myoclonic + tonic."},
    {"stage": "Early Childhood (DRE Management)", "age": "18 months–6 years", "description": "DRE management for classic phenotype; attenuated: epilepsy manageable; school/rehabilitation planning; cognitive plateau vs slow gains; KD initiation if DRE; respite care; home seizure care plan; nasal midazolam rescue; annual MRI."},
    {"stage": "School Age / Adolescence (Attenuated)", "age": "6–18 years", "description": "Classic NKH: specialized care facility; attenuated NKH: mainstream school with support; mood disorders (depression, anxiety 30–40%); puberty-related seizure frequency change; VPA CI continues; sodium benzoate compliance monitoring; genetics re-review with updated variant interpretation."},
    {"stage": "Adulthood (Chronic Management)", "age": "18+ years", "description": "Classic NKH: specialized residential care; attenuated: supported independent living possible (for mildest); renal monitoring (sodium benzoate → hippurate → renal load long-term); carnitine; genetic counselling (25% sibling recurrence AR); clinical trial enrolment (GCS replacement, mRNA therapy in pipeline 2026)."},
]

CONCEPTS = [
    {"term": "GLDC / P-protein / 9p24.1", "definition": "Glycine Decarboxylase; P-protein (pyridoxal phosphate-dependent); 1020 aa, ~114 kDa; mitochondrial matrix; decarboxylates glycine + transfers aminomethyl moiety to H-protein (GCSH). AR biallelic LOF → NKH. OMIM *238300 gene / #605899 disease."},
    {"term": "GCS (Glycine Cleavage System) — 4-Protein Complex", "definition": "Mitochondrial matrix multi-enzyme complex: P-protein (GLDC, decarboxylase), H-protein (GCSH, lipoic acid carrier), T-protein (AMT, aminomethyltransferase), L-protein (DLD, dihydrolipoamide dehydrogenase, shared). GCS net reaction: Glycine + THF + NAD+ → 5,10-methyleneTHF + CO2 + NH4+ + NADH. LOF → glycine accumulates."},
    {"term": "CSF:Plasma Glycine Ratio — DIAGNOSTIC THRESHOLD", "definition": "SIMULTANEOUS collection mandatory. Normal: <0.02. NKH: ≥0.08 (often 0.15–0.50). Threshold 0.08 is highly specific. Plasma glycine alone insufficient. Must be fasting. LP sedation should avoid GlyR-potentiating agents (ketamine preferred for sedation in NKH)."},
    {"term": "NMDAr Co-Agonism — Core Excitotoxic Mechanism", "definition": "GluN1 subunit of NMDAr has obligate glycine-binding site (Km ~0.5–5 µM, distinct from glutamate site). In NKH: CSF glycine often >500 µM → glycine site SATURATED → NMDAr in maximum activation state when glutamate also present → excitotoxic seizures. DXM/felbamate block this by channel antagonism."},
    {"term": "HICCUPS — Pathognomonic NKH Clue", "definition": "Phrenic nerve nucleus (C3–C5 anterior horn) uses glycinergic inhibitory interneurons. Excess glycine → excessive GlyR activation at phrenic nucleus → rhythmic uncontrolled diaphragm contractions = hiccups. Persistent neonatal hiccups + hypotonia + apnea = NKH until proven otherwise. Not seen in other metabolic epilepsies."},
    {"term": "Burst-Suppression EEG — NKH Neonatal Signature", "definition": "Burst-suppression within hours-days of birth = hallmark neonatal NKH EEG. Bursts of high-amplitude polyspikes alternating with isoelectric suppression. This pattern reflects severe cortical dysfunction from NMDAr excitotoxicity. Transitions to hypsarrhythmia by 3–6 months."},
    {"term": "Sodium Benzoate — Glycine Conjugation Mechanism", "definition": "Benzoate + glycine → hippuric acid (N-benzoylglycine) via glycine N-acyltransferase (GLYAT) in mitochondria. Hippurate excreted in urine. Removes glycine from available pool. Net: each benzoate molecule = one glycine eliminated. Rate limited by GLYAT capacity and carnitine availability. Carnitine co-supplementation mandatory."},
    {"term": "Dextromethorphan (DXM) — NMDAr Channel Antagonist", "definition": "DXM (prodrug) → dextrorphan (active, by CYP2D6) → uncompetitive NMDAr channel blocker at Mg2+ site. Blocks the open NMDAr channel regardless of glycine concentration. CYP2D6 phenotyping clinically relevant — poor metabolizers: DXM accumulates; ultra-rapid: reduced effect from rapid conversion."},
    {"term": "VPA HIGH RISK — GLDC Inhibition Mechanism", "definition": "VPA directly inhibits GLDC (P-protein) at therapeutic concentrations. Also inhibits N-methylglycine (sarcosine) metabolism pathway (an alternative glycine disposal route). Net: VPA raises plasma and CSF glycine in NKH — directly worsening the disease. Distinct mechanism from VPA carnitine depletion/POLG CI — an ADDITIONAL NKH-specific risk."},
    {"term": "VGB HIGH RISK for IS in NKH", "definition": "VGB inhibits GABA-T → GABA↑. GABA-glycine co-transport interactions may raise extracellular glycine. Clinical risk: VGB for IS in NKH may worsen glycine-driven NMDAr burden. ACTH preferred for NKH-IS. VGB retinal toxicity is ALSO a concern (independent of NKH mechanism)."},
    {"term": "p.Gly761Arg (c.2281G>A) — Attenuated Founder Allele", "definition": "Most common European/worldwide hypomorphic GLDC allele. Partial function (~20–30% residual GLDC activity). Homozygous p.Gly761Arg → mildest NKH (some walk, communicate, IQ 40–60). Compound het with null allele → moderate NKH. Key genotype-phenotype correlation anchor."},
    {"term": "Carnitine Depletion — Sodium Benzoate Complication", "definition": "Benzoate conjugation with glycine consumes CoA and indirectly depletes carnitine stores (glycine-benzoate conjugation shifts acyl-CoA balance, reducing free carnitine). Plasma carnitine falls on high-dose benzoate therapy. L-carnitine 50–100 mg/kg/day supplementation MANDATORY with sodium benzoate."},
    {"term": "Transient NKH vs Classic NKH — Critical Distinction", "definition": "Transient NKH: neonatal glycine elevation that normalises by 8 weeks; enzyme immaturity; generally benign. MUST re-test CSF:plasma ratio at 8 weeks minimum — some 'transient' cases evolve to classic NKH. Gene sequencing distinguishes: if biallelic pathogenic GLDC variants → not truly transient. Close neurodevelopmental follow-up required."},
    {"term": "Thin/Absent Corpus Callosum — NKH MRI Signature", "definition": "Agenesis or hypoplasia of the corpus callosum (especially splenium) in classic NKH. Due to NMDAr excitotoxicity during critical axonal development window (periventricular WM). Periventricular T2 WM hyperintensity also characteristic. DWI: cytotoxic edema in acute phase. Cerebellar hypoplasia in severe cases."},
    {"term": "5,10-MethyleneTHF Depletion — One-Carbon Metabolism Impact", "definition": "GCS produces 5,10-methyleneTHF (one-carbon unit donor for thymidylate synthesis + serine synthesis + methionine cycle via MTHFR). GLDC LOF → GCS not running → 5,10-methyleneTHF production from glycine pathway reduced. Folate supplementation often needed in NKH. Monitoring: plasma folate, methionine, homocysteine."},
]

THRESHOLDS = [
    {"parameter": "CSF:Plasma glycine ratio (normal)", "value": "<0.02", "clinical": "Normal GCS function; no NKH"},
    {"parameter": "CSF:Plasma glycine ratio (NKH diagnostic)", "value": "≥0.08", "clinical": "Highly specific for NKH; classic severe: 0.15–0.50"},
    {"parameter": "Plasma glycine (normal)", "value": "150–260 µmol/L", "clinical": "Normal range (laboratory-dependent)"},
    {"parameter": "Plasma glycine (NKH untreated)", "value": "600–3000 µmol/L", "clinical": "10–20× normal; severity correlates with ratio not absolute level"},
    {"parameter": "Target plasma glycine on sodium benzoate", "value": "<500 µmol/L (ideally <300 µmol/L)", "clinical": "Glycine-lowering target; TDM every 3 months"},
    {"parameter": "CSF glycine (normal)", "value": "3–10 µmol/L", "clinical": "Normal range"},
    {"parameter": "CSF glycine (NKH)", "value": ">100 µmol/L (often >300 µmol/L)", "clinical": "Severe classic: 300–2000 µmol/L; correlates with excitotoxic burden"},
    {"parameter": "Serum carnitine (target on benzoate)", "value": ">20 µmol/L free carnitine", "clinical": "Supplement if <20 µmol/L; benzoate depletes carnitine via conjugation pathway"},
    {"parameter": "NH3 on sodium benzoate (threshold)", "value": "<80 µmol/L", "clinical": "Hyperammonaemia risk at high benzoate doses (disrupts urea cycle); >80 → benzoate dose reduction"},
    {"parameter": "DXM CYP2D6 poor metabolizer", "value": "DXM levels >200 ng/mL", "clinical": "Accumulation risk in PM — reduce dose 50%; consider dextrorphan direct if available"},
    {"parameter": "MRI CC (corpus callosum)", "value": "CC thickness at genu/splenium <2 SD", "clinical": "CC hypoplasia/agenesis in classic NKH; splenium typically most affected"},
]

def _patient_profile(i):
    pat_id = f"NKH-{i+1:03d}"
    sex = "F" if rng.random() < 0.5 else "M"
    etiology = _rng_choice(ETIOLOGIES)
    # Phenotype from etiology
    if "null/null" in etiology["etiology"] or "Classic" in etiology["etiology"] or "consanguineous" in etiology["etiology"]:
        phenotype_class = "Classic Neonatal"
        onset_age_days = _rng_int(0, 4)
    elif "Attenuated" in etiology["etiology"]:
        phenotype_class = "Attenuated"
        onset_age_days = _rng_int(90, 365)
    else:
        phenotype_class = "Transient"
        onset_age_days = _rng_int(0, 7)
    has_epilepsy = phenotype_class != "Transient" or rng.random() < 0.3
    has_is = phenotype_class == "Classic Neonatal" and rng.random() < 0.45
    has_bs = phenotype_class == "Classic Neonatal" and rng.random() < 0.60
    dre = has_epilepsy and (phenotype_class == "Classic Neonatal" and rng.random() < 0.80 or
                            phenotype_class == "Attenuated" and rng.random() < 0.25)
    on_benzoate = rng.random() < 0.90
    on_dxm = rng.random() < 0.75
    on_lev = rng.random() < 0.70
    on_clb = rng.random() < 0.45
    on_acth = has_is and rng.random() < 0.80
    on_kd = dre and rng.random() < 0.30
    plasma_gly = _rng_float(180, 600, 0) if on_benzoate else _rng_float(600, 2500, 0)
    csf_plasma_ratio = _rng_float(0.09, 0.45, 3) if phenotype_class == "Classic Neonatal" else _rng_float(0.08, 0.15, 3)
    carnitine_ok = rng.random() < 0.75
    confidence = _rng_float(0.72, 0.99)
    return {
        "patient_id": pat_id,
        "sex": sex,
        "phenotype_class": phenotype_class,
        "onset_age_days": int(onset_age_days),
        "etiology": etiology["etiology"][:70],
        "has_epilepsy": has_epilepsy,
        "has_infantile_spasms": has_is,
        "has_burst_suppression": has_bs,
        "drug_resistant": dre,
        "on_benzoate": on_benzoate,
        "on_dxm": on_dxm,
        "on_lev": on_lev,
        "on_clb": on_clb,
        "on_acth": on_acth,
        "on_kd": on_kd,
        "plasma_glycine_umol": float(plasma_gly),
        "csf_plasma_ratio": float(csf_plasma_ratio),
        "carnitine_normal": carnitine_ok,
        "confidence": confidence,
    }


PATIENTS = [_patient_profile(i) for i in range(N)]


def get_overview():
    n_epilepsy = sum(1 for p in PATIENTS if p["has_epilepsy"])
    n_is = sum(1 for p in PATIENTS if p["has_infantile_spasms"])
    n_bs = sum(1 for p in PATIENTS if p["has_burst_suppression"])
    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_classic = sum(1 for p in PATIENTS if p["phenotype_class"] == "Classic Neonatal")
    n_attenuated = sum(1 for p in PATIENTS if p["phenotype_class"] == "Attenuated")
    n_transient = sum(1 for p in PATIENTS if p["phenotype_class"] == "Transient")
    n_benzoate = sum(1 for p in PATIENTS if p["on_benzoate"])
    n_dxm = sum(1 for p in PATIENTS if p["on_dxm"])
    avg_plasma_gly = round(sum(p["plasma_glycine_umol"] for p in PATIENTS) / N, 1)
    avg_ratio = round(sum(p["csf_plasma_ratio"] for p in PATIENTS) / N, 3)
    return {
        "dashboard": "GLDC Epilepsy — Non-Ketotic Hyperglycinemia (NKH) / Glycine Encephalopathy",
        "gene": "GLDC (9p24.1) — Glycine Decarboxylase; P-protein of GCS; 1020 aa ~114 kDa; mitochondrial matrix; PLP-dependent decarboxylase; first step of the GCS; ~75–80% of NKH",
        "inheritance": "Autosomal Recessive (AR) biallelic LOF; pLI~0.0; ~1:60,000–76,000 live births (Europe); ~1:12,000 (Finland — founder effect); ~1,200–2,000 cases worldwide 2026",
        "omim_gene": "238300",
        "omim_disease": "605899",
        "locus": "9p24.1",
        "cohort_size": N,
        "female_n": sum(1 for p in PATIENTS if p["sex"] == "F"),
        "female_pct": round(100 * sum(1 for p in PATIENTS if p["sex"] == "F") / N),
        "n_epilepsy": n_epilepsy,
        "epilepsy_pct": round(100 * n_epilepsy / N),
        "n_infantile_spasms": n_is,
        "is_pct": round(100 * n_is / N),
        "n_burst_suppression": n_bs,
        "burst_suppression_pct": round(100 * n_bs / N),
        "n_dre": n_dre,
        "dre_pct": round(100 * n_dre / N),
        "n_classic_neonatal": n_classic,
        "classic_pct": round(100 * n_classic / N),
        "n_attenuated": n_attenuated,
        "attenuated_pct": round(100 * n_attenuated / N),
        "n_transient": n_transient,
        "transient_pct": round(100 * n_transient / N),
        "n_on_benzoate": n_benzoate,
        "benzoate_pct": round(100 * n_benzoate / N),
        "n_on_dxm": n_dxm,
        "dxm_pct": round(100 * n_dxm / N),
        "avg_plasma_glycine": avg_plasma_gly,
        "avg_csf_plasma_ratio": avg_ratio,
        "phenotype_classes": PHENOTYPE_CLASSES,
        "etiologies": ETIOLOGIES,
        "key_concepts": [c["term"] for c in CONCEPTS[:8]],
        "standards": [
            "Hamosh A & Johnston MV. Nonketotic hyperglycinemia. OMIM #605899.",
            "Kikuchi G et al. Glycine cleavage system. Mol Cell Biochem 1978.",
            "Van Hove JLK et al. NKH — long-term outcome. J Inherit Metab Dis 2006.",
            "Toone JR et al. Glycine decarboxylase deficiency. Mol Genet Metab 2003.",
            "Korman SH et al. Disparate clinical outcome in siblings with NKH. J Inherit Metab Dis 2004.",
            "Swanson MA et al. Attenuated NKH outcomes. Dev Med Child Neurol 2015.",
            "ACMG/AMP — Variant interpretation standards for NKH/GLDC.",
            "ILAE — Guidelines for neonatal metabolic epilepsy (2024).",
            "NKH International Family Network — Clinical care guidelines 2023.",
        ],
        "per_patient_kpis": sorted(PATIENTS, key=lambda p: p["csf_plasma_ratio"], reverse=True),
    }


def get_breakdown():
    classic = sum(1 for p in PATIENTS if p["phenotype_class"] == "Classic Neonatal")
    attenuated = sum(1 for p in PATIENTS if p["phenotype_class"] == "Attenuated")
    transient = sum(1 for p in PATIENTS if p["phenotype_class"] == "Transient")

    ratio_ranges = [(0.0, 0.08, "<0.08 (not NKH/transient)"), (0.08, 0.15, "0.08–0.15 (attenuated)"),
                    (0.15, 0.30, "0.15–0.30 (moderate)"), (0.30, 1.0, ">0.30 (severe classic)")]
    ratio_hist = []
    for lo, hi, label in ratio_ranges:
        cnt = sum(1 for p in PATIENTS if lo <= p["csf_plasma_ratio"] < hi)
        ratio_hist.append({"range": label, "n": cnt, "pct": round(100 * cnt / N)})

    gly_ranges = [(0, 300, "<300 µmol/L (on-target)"), (300, 500, "300–500 (partially controlled)"),
                  (500, 1000, "500–1000 (subtherapeutic)"), (1000, 5000, ">1000 (uncontrolled)")]
    gly_hist = []
    for lo, hi, label in gly_ranges:
        cnt = sum(1 for p in PATIENTS if lo <= p["plasma_glycine_umol"] < hi)
        gly_hist.append({"range": label, "n": cnt, "pct": round(100 * cnt / N)})

    treatment_counts = {
        "Sodium Benzoate": sum(1 for p in PATIENTS if p["on_benzoate"]),
        "DXM": sum(1 for p in PATIENTS if p["on_dxm"]),
        "LEV": sum(1 for p in PATIENTS if p["on_lev"]),
        "CLB": sum(1 for p in PATIENTS if p["on_clb"]),
        "ACTH": sum(1 for p in PATIENTS if p["on_acth"]),
        "KD": sum(1 for p in PATIENTS if p["on_kd"]),
    }

    return {
        "phenotype_class_distribution": [
            {"class": "Classic Neonatal", "n": classic, "pct": round(100 * classic / N), "colour": "#b71c1c"},
            {"class": "Attenuated", "n": attenuated, "pct": round(100 * attenuated / N), "colour": "#e65100"},
            {"class": "Transient", "n": transient, "pct": round(100 * transient / N), "colour": "#1565c0"},
        ],
        "seizure_type_distribution": SEIZURE_TYPES,
        "trigger_distribution": TRIGGERS,
        "csf_plasma_ratio_histogram": ratio_hist,
        "plasma_glycine_histogram": gly_hist,
        "treatment_counts": treatment_counts,
        "per_patient_profiles": PATIENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "title": "GLDC / Non-Ketotic Hyperglycinemia — Definitions, GCS Pathway, Pharmacology",
        "gene_card": {
            "gene": "GLDC",
            "locus": "9p24.1",
            "protein": "Glycine Decarboxylase (P-protein)",
            "size": "1020 aa, ~114 kDa",
            "family": "Pyridoxal phosphate (PLP)-dependent amino acid decarboxylases",
            "structure": "Mitochondrial matrix; homomeric; PLP-Schiff base at active site Lys754; decarboxylates glycine → aminomethyl intermediate transferred to H-protein (GCSH)",
            "cofactor": "Pyridoxal phosphate (PLP, vitamin B6) — covalently bound; NOT B6-responsive disease (B6 supplementation does not correct LOF)",
            "localisation": "Mitochondrial matrix (N-terminal MTS)",
            "omim_gene": "*238300",
            "omim_disease": "#605899",
            "inheritance": "AR biallelic LOF",
            "cause_of_nkh": "~75–80% of NKH; AMT ~15%; GCSH ~1%; DLD very rare",
        },
        "pathway": {
            "name": "Glycine Cleavage System (GCS) — 4-Protein Complex",
            "steps": [
                {"step": 1, "enzyme": "P-protein (GLDC)", "gene": "GLDC", "cofactor": "PLP",
                 "reaction": "Glycine + H-protein(oxidised) → CO₂ + aminomethyl-H-protein",
                 "clinical": "GLDC LOF (75–80% of NKH): P-protein absent/dysfunctional → glycine cannot enter GCS → accumulates in all compartments"},
                {"step": 2, "enzyme": "H-protein (GCSH)", "gene": "GCSH", "cofactor": "Lipoic acid",
                 "reaction": "Aminomethyl-H-protein shuttles to T-protein",
                 "clinical": "GCSH LOF (1% of NKH): H-protein cannot carry aminomethyl group → same glycine accumulation; biochemically identical to GLDC NKH"},
                {"step": 3, "enzyme": "T-protein (AMT)", "gene": "AMT", "cofactor": "THF",
                 "reaction": "Aminomethyl-H-protein + THF → 5,10-methyleneTHF + NH₄⁺ + H-protein(reduced)",
                 "clinical": "AMT LOF (15% of NKH): T-protein absent → same NKH; 5,10-methyleneTHF not produced from this step"},
                {"step": 4, "enzyme": "L-protein (DLD)", "gene": "DLD", "cofactor": "NAD+/FAD",
                 "reaction": "H-protein(reduced) + NAD⁺ → H-protein(oxidised) + NADH",
                 "clinical": "DLD is shared with pyruvate DH and alpha-KG DH complexes; DLD LOF → combined dehydrogenase defect + glycine accumulation (rare, distinct from classic NKH)"},
            ],
            "net_reaction": "Glycine + THF + NAD⁺ → 5,10-methyleneTHF + CO₂ + NH₄⁺ + NADH",
            "clinical_consequence": "GLDC/AMT/GCSH LOF → GCS inoperable → glycine cannot be catabolised → accumulates in plasma/CSF/brain"
        },
        "biomarkers": [
            {"marker": "CSF:Plasma glycine ratio (simultaneous)", "method": "Quantitative plasma amino acids (LC-MS/MS) + CSF amino acids",
             "reference_range": "<0.02", "nkh_range": "≥0.08 (classic: 0.15–0.50)",
             "notes": "PRIMARY DIAGNOSTIC TEST. Simultaneous collection mandatory. Highly specific ≥0.08. Cannot diagnose from plasma glycine alone. Fasting preferred."},
            {"marker": "Plasma glycine (quantitative)", "method": "Quantitative amino acid panel (LC-MS/MS)",
             "reference_range": "150–260 µmol/L", "nkh_range": "600–3000+ µmol/L (untreated)",
             "notes": "Sensitive but not specific — elevated in many organic acidemias. Ratio is diagnostic; plasma level is monitoring tool. Target on benzoate: <500 µmol/L."},
            {"marker": "Urine hippuric acid (on sodium benzoate)", "method": "Urine organic acid GC-MS",
             "reference_range": "Variable (benzoate-dependent)", "nkh_range": "High on therapy — confirms benzoate metabolism",
             "notes": "Treatment monitoring: hippuric acid excretion confirms benzoate conjugation is working; absent hippuric acid despite benzoate intake = poor adherence or GLYAT saturation."},
            {"marker": "GLDC enzyme activity (lymphocytes/liver)", "method": "14C-glycine oxidation assay",
             "reference_range": "Normal range lab-dependent", "nkh_range": "<5% residual (classic); 20–30% (attenuated)",
             "notes": "Confirmatory enzyme assay; technically demanding; available at specialist reference labs; not widely available; gene sequencing usually preferred for diagnosis in 2026."},
        ],
        "key_concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "treatments": TREATMENTS,
        "references": [
            "Hamosh A & Johnston MV. Nonketotic hyperglycinemia. In: OMIM #605899. 2024.",
            "Kikuchi G, Motokawa Y, Yoshida T. Glycine cleavage system. Mol Cell Biochem 1978.",
            "Van Hove JLK et al. Long-term outcome and management of NKH. J Inherit Metab Dis 2006.",
            "Toone JR et al. Molecular genetic and potential biochemical characteristics. Mol Genet Metab 2003.",
            "Korman SH et al. Disparate clinical outcome in siblings with NKH. J Inherit Metab Dis 2004.",
            "Swanson MA et al. Attenuated NKH outcomes in older patients. Dev Med Child Neurol 2015.",
            "García-Cazorla A et al. NKH clinical spectrum. Orphanet J Rare Dis 2022.",
            "NKH International Family Network — Clinical care guidelines. 2023.",
        ],
        "differential_diagnosis": [
            {"condition": "SSADH deficiency (ALDH5A1)", "distinction": "SSADH: GHB ↑↑ in urine (not glycine); glycine NORMAL; globus pallidus T2 hyperintensity; VGB ABSOLUTE CI (different mechanism); no burst-suppression neonatal; NKH: glycine ↑↑, GHB NORMAL"},
            {"condition": "D-glyceric aciduria (GLYCTK)", "distinction": "D-glyceric aciduria: glycine ↑ plasma, L-2-hydroxy-3-oxoadipic acid elevated; GCS activity NORMAL; gene: GLYCTK; rare; NKH: GCS absent, CSF:plasma glycine ratio ≥0.08"},
            {"condition": "Propionic acidemia (PCCA/PCCB)", "distinction": "Propionic acidemia: GLYCINE ↑ (methylamine accumulates → glycine conjugation pathway saturated — 'ketotic hyperglycinemia'); organic acids: 3-OH-propionate/propionylglycine/methylcitrate; GCS normal; ACYLCARNITINE C3 elevated; NEVER diagnose NKH without organic acids"},
            {"condition": "Methylmalonic acidemia (MUT/MMAA etc.)", "distinction": "MMA: glycine ↑ (same 'ketotic hyperglycinemia' mechanism as propionic); MMA ↑↑ urine; C3 acylcarnitine; B12 responsive forms; organic acids mandatory to exclude"},
            {"condition": "Isovaleric acidemia (IVD)", "distinction": "IVA: glycine ↑ (glycine conjugation with isovaleric acid); isovalerylglycine in urine organic acids; C5 acylcarnitine; distinctive 'sweaty feet' odour; NKH: glycine primary elevation without isovalerylglycine"},
            {"condition": "Hyperglycinuria (SLC6A18/renal)", "distinction": "Isolated hyperglycinuria (renal aminoaciduria, SLC6A18 Hartnup variant): glycine ↑ URINE but PLASMA normal and CSF:plasma ratio NORMAL (<0.02); benign; NKH: plasma + CSF both elevated; ratio ≥0.08"},
        ],
    }
