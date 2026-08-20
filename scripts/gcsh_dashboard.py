#!/usr/bin/env python3
"""GCSH (Glycine Cleavage System H-protein) Epilepsy Dashboard.

GCSH encodes the H-protein — the CENTRAL CARRIER of the mitochondrial Glycine Cleavage System
(GCS). H-protein is the rarest cause of Non-Ketotic Hyperglycinemia (NKH / Glycine Encephalopathy),
accounting for ~1% of all NKH cases (GLDC P-protein ~75–80%, AMT T-protein ~15%, GCSH H-protein ~1%).

GCS — 4-PROTEIN COMPLEX (mitochondrial matrix):
  • P-protein (GLDC, 1020 aa): PLP-dependent; oxidatively decarboxylates glycine; transfers the
    aminomethyl moiety to the H-protein via its lipoamide arm; releases CO₂
  • H-protein (GCSH, 125 aa): lipoic acid-bearing CENTRAL CARRIER; accepts aminomethyl group
    from P-protein; shuttles it to T-protein (AMT); H-protein cycles between aminomethyl-loaded
    (reduced lipoamide) and unloaded (oxidised lipoamide) states — IT IS THE SWINGING ARM
  • T-protein (AMT, 403 aa): aminomethyltransferase; accepts aminomethyl group from loaded
    H-protein; transfers it to THF → 5,10-methyleneTHF + NH₄⁺; simultaneously regenerates
    oxidised H-protein to cycle GCS
  • L-protein (DLD, shared): lipoamide dehydrogenase; regenerates oxidised H-protein lipoic
    acid via NAD+ (shared with pyruvate DH and alpha-KG DH complexes)

GCS NET REACTION:
  Glycine + THF + NAD⁺ → 5,10-methyleneTHF + CO₂ + NH₄⁺ + NADH

GCSH — THE H-PROTEIN: CENTRAL HUB OF THE GCS SWINGING ARM MECHANISM
  H-protein is a UNIQUE scaffold: 125 aa, ~16 kDa; lipoic acid covalently attached to Lys59
  via an amide bond. The lipoyl domain undergoes a ~14 Å conformational swing that physically
  transports the aminomethyl group from P-protein to T-protein.
  GCSH has NO catalytic activity of its own — it is a CARRIER ONLY.

GCSH LOF — THE CENTRAL BOTTLENECK (UNIQUE VS GLDC/AMT LOF):
  When H-protein is absent/dysfunctional:
  (1) P-protein (GLDC) CANNOT transfer its aminomethyl intermediate → no H-protein to receive it
      → P-protein stalls at Step 1 → GCS blocked UPSTREAM
  (2) T-protein (AMT) has NO SUBSTRATE → no loaded H-protein to accept aminomethyl from
      → T-protein stalls at Step 3 → GCS blocked DOWNSTREAM simultaneously
  Net result: ENTIRE GCS is blocked at BOTH upstream and downstream simultaneously — the most
  complete GCS block of any NKH type. Glycine accumulates IDENTICALLY to GLDC-NKH and AMT-NKH.
  5,10-methyleneTHF is NOT produced (identical folate perturbation to GLDC-NKH and AMT-NKH).

PATHOPHYSIOLOGY — SAME DUAL GLYCINE RECEPTOR PARADOX AS GLDC/AMT NKH:
  GlyR (GLRA1/GLRB): Cl⁻ channel (inhibitory) — brainstem/spinal cord/reticular formation.
  Neonates: excess glycine → GlyR OVER-ACTIVATION → profound hypotonia + apnea + HICCUPS
  (phrenic nucleus GlyR — C3–C5).
  NMDAr (GluN1/GluN2): GluN1 obligate glycine co-agonist site (Km ~0.5–5 µM). CSF glycine in
  NKH >500 µM → GluN1 site SATURATED → maximum NMDAr excitotoxicity → burst-suppression → SE.

DIAGNOSTIC BIOMARKER — IDENTICAL TO GLDC-NKH AND AMT-NKH:
  CSF:plasma glycine ratio (SIMULTANEOUS): normal <0.02; NKH ≥0.08. GCSH-NKH: same ratio
  range as GLDC-NKH and AMT-NKH. GCSH CANNOT be distinguished from GLDC or AMT biochemically —
  gene panel mandatory (AMT + GLDC + GCSH + DLD).

GENETICS:
  Gene: GCSH at 16q23.2; 5 exons; 125 amino acids; ~16 kDa; mitochondrial matrix; lipoamide
  carrier (no catalytic activity); lipoic acid covalently attached to Lys59.
  AR biallelic LOF. OMIM gene *238330.
  ~40–60 GCSH pathogenic variants reported (all rare/private — NO pan-ethnic founder allele).
  KEY REPORTED VARIANTS:
    • p.Gly47Arg (c.139G>A): disrupts lipoyl domain folding; near Lys59 lipoic acid attachment;
      severe neonatal phenotype; one of most commonly cited GCSH variants
    • p.Pro189Leu (c.566C>T): C-terminal domain; severe; classic neonatal
    • p.Ala32Val (c.95C>T): N-terminal region; disrupts MTS or early folding; severe
    • p.Arg228Gln (c.683G>A): lipoyl domain interaction surface; moderate
    • Homozygous frameshift/nonsense: null alleles → classic severe neonatal; most severe

EPIDEMIOLOGY:
  ~1% of NKH cases; NKH overall ~1:60,000–76,000 (Europe); GCSH-NKH ~1:6,000,000–7,600,000.
  ~20–50 GCSH-NKH cases worldwide 2026 — the rarest GCS gene defect causing NKH.
  AR biallelic LOF; 16q23.2. No geographic enrichment.

TREATMENT — IDENTICAL TO GLDC-NKH AND AMT-NKH (same disease, different carrier gene):
  1. Sodium Benzoate (Level A): conjugates glycine → hippuric acid; depletes glycine pool
     MANDATORY: L-carnitine co-supplementation (benzoate conjugation depletes carnitine)
     GCSH-NOTE: benzoate acts via GLYAT — completely independent of H-protein; equally effective
  2. Dextromethorphan / DXM (Level B): NMDAr channel antagonist; CYP2D6 prodrug
  3. LEV (Level B): first-line AED; SV2A; no glycine interaction; IV for SE
  4. CLB (Level B): GABA-A PAM; adjunct for myoclonic + tonic
  5. ACTH (Level A): IS management; preferred over VGB (which raises glycine)
  6. KD (Level B): DRE adjunct; reduces serine→glycine flux (SHMT pathway)
  7. Felbamate (Level C): NMDAr glycine-site antagonist; aplastic anaemia risk

HIGH RISK DRUGS (same profile as GLDC-NKH and AMT-NKH):
  VPA: HIGH RISK — VPA raises glycine via multiple pathways. In GCSH-NKH: VPA cannot directly
  inhibit H-protein (no enzymatic activity to inhibit), but raises glycine via secondary mechanisms
  (GCS disequilibrium, sarcosine pathway inhibition, carnitine depletion). Avoid in all NKH.
  VGB: HIGH RISK for IS — GABA↑ → GABA-glycine co-transporter upregulation → CSF glycine rises;
  prefer ACTH for IS in all NKH types. VGB retinal toxicity is additional independent CI.
"""

import random
from datetime import datetime

SEED = 20260821
rng = random.Random(SEED)

def _rng_choice(items): return rng.choice(items)
def _rng_int(lo, hi): return rng.randint(lo, hi)
def _rng_float(lo, hi, dec=2): return round(rng.uniform(lo, hi), dec)

# Colour: dark amber-brown — H-protein lipoamide carrier / lipoic acid sulfur chemistry
COLOUR = "#5d4037"  # dark brown — H-protein / lipoamide / lipoic acid / central carrier

N = 40  # 40-patient cohort

ETIOLOGIES = [
    {"etiology": "Classic Neonatal — homozygous null (truncating/frameshift GCSH biallelic)", "pct": 35, "n": 14,
     "csf_plasma_ratio": "0.18–0.55", "outcome": "Severe ID; DRE; non-ambulatory; mechanical ventilation neonatal"},
    {"etiology": "Classic Neonatal — compound het null/missense (e.g. null + p.Gly47Arg)", "pct": 25, "n": 10,
     "csf_plasma_ratio": "0.12–0.45", "outcome": "Severe-to-moderate neonatal; IS + multifocal; DRE in >75%"},
    {"etiology": "Attenuated — compound het with partial-function GCSH missense (H-protein partial lipoamide function)", "pct": 20, "n": 8,
     "csf_plasma_ratio": "0.08–0.18", "outcome": "Moderate phenotype; ambulatory possible; seizures manageable with benzoate + DXM"},
    {"etiology": "Classic Neonatal — consanguineous homozygous missense (near Lys59 lipoyl site)", "pct": 12, "n": 5,
     "csf_plasma_ratio": "0.20–0.50", "outcome": "Severe; neonatal ICU; high early mortality; DRE in survivors"},
    {"etiology": "Attenuated — homozygous partial-function missense (residual H-protein carrier activity)", "pct": 5, "n": 2,
     "csf_plasma_ratio": "0.08–0.14", "outcome": "Mildest GCSH-NKH; intellectual disability variable; benzoate + DXM often sufficient"},
    {"etiology": "Transient NKH — neonatal GCSH glycine elevation normalises (H-protein immaturity or VUS)", "pct": 3, "n": 1,
     "csf_plasma_ratio": "0.08–0.11 (normalising)", "outcome": "Generally benign; GCSH gene still pathogenic — monitor closely"},
]

PHENOTYPE_CLASSES = [
    {
        "name": "Classic Neonatal (Severe)", "pct": 70,
        "description": "Onset within hours–2 days of birth. Profound hypotonia, apnea (mechanical ventilation often required), absent Moro/suck, HICCUPS (pathognomonic — phrenic nucleus GlyR). EEG: burst-suppression → hypsarrhythmia. Mortality ~25–35% neonatal. Survivors: severe ID, refractory epilepsy. GCSH null/null or null/missense near Lys59. Biochemically IDENTICAL to GLDC-NKH and AMT-NKH — gene panel distinguishes. GCSH represents most complete GCS block (both GLDC upstream AND AMT downstream simultaneously inoperable).",
        "eeg": "Burst-suppression (neonatal) → hypsarrhythmia → multifocal spikes → electrical SE",
        "seizure_types": "Myoclonic, focal clonic, tonic, electrical SE (EEG-confirmed without motor due to hypotonia)",
        "outcome": "Severe ID in all survivors; DRE >80%; non-ambulatory; non-verbal",
        "csf_ratio": ">0.12 (often 0.20–0.55)"
    },
    {
        "name": "Attenuated (Mild-Moderate)", "pct": 25,
        "description": "Later onset (weeks–years). Milder ID. Seizures in ~50–60% — manageable with benzoate + DXM + LEV. Chorea/choreoathetosis in ~30%. Some ambulation and speech. Associated with partial-function GCSH missense variants that retain partial lipoamide carrier activity (~10–25% residual H-protein function). CSF:plasma ratio typically 0.08–0.18. No known founder allele unlike AMT (p.Arg320His) — must identify on gene panel.",
        "eeg": "Multifocal or generalised spikes; no burst-suppression; may be subtle",
        "seizure_types": "Focal, myoclonic, absence — lower frequency; benzoate + DXM often sufficient",
        "outcome": "Variable; some ambulatory + communicative; IQ 30–65 in mildest forms",
        "csf_ratio": "0.08–0.18 (borderline-moderate elevation)"
    },
    {
        "name": "Transient NKH (Rare)", "pct": 5,
        "description": "Neonatal glycine elevation that normalises by ~8 weeks. H-protein immaturity hypothesised — very rare in GCSH-NKH. IMPORTANT: gene sequencing still identifies pathogenic GCSH variants. Re-test CSF:plasma ratio at 8 weeks minimum. GCSH transient cases are even rarer than GLDC/AMT transient NKH given the ultra-rare disease. Neurodevelopmental follow-up mandatory.",
        "eeg": "Transient burst-suppression → normalises by weeks 4–8",
        "seizure_types": "Transient neonatal seizures; resolve as glycine normalises",
        "outcome": "Generally benign; subtle developmental delay common; long-term seizure-free possible",
        "csf_ratio": "Elevated (>0.08) initially → <0.02 by 8 weeks"
    },
]

SEIZURE_TYPES = [
    {"type": "Electrical Status Epilepticus (neonatal SE, EEG-confirmed)", "pct": 60,
     "eeg": "Burst-suppression → continuous electrographic SE; no motor correlate (hypotonia masks)",
     "semiology": "EEG-only SE in context of profound hypotonia; easily missed without continuous video-EEG; treat as SE; burst-suppression background confirms metabolic cause",
     "tips": "LEV IV 60 mg/kg loading; phenobarbital second-line (CAUTION — GlyR additive); IV ketamine third-line for refractory SE (NMDAr antagonist — BENEFICIAL in NKH); simultaneous IV sodium benzoate + glycine lowering"},
    {"type": "Myoclonic seizures (neonatal/infantile)", "pct": 70,
     "eeg": "Poly-spike GSW or poly-spike with burst-suppression background; high-amplitude bursts",
     "semiology": "Sudden jerks in burst-suppression 'bursts'; myoclonic + IS combination; bilateral synchronous",
     "tips": "IS: ACTH Level A; myoclonic: VPA HIGH RISK — LEV + CLB preferred; avoid CBZ/PHT/OXC (myoclonic worsening); DXM reduces burst-suppression-linked myoclonic burden via NMDAr blockade"},
    {"type": "Infantile Spasms (IS / West Syndrome)", "pct": 45,
     "eeg": "Hypsarrhythmia — classical or modified (inter-spasm suppression + multifocal spikes)",
     "semiology": "Spasm clusters; salaam movements; regressive milestones; peak onset 4–8 months",
     "tips": "ACTH Level A; VGB AVOID in NKH (raises glycine — see pharmacology); combined ACTH + sodium benzoate + DXM = NKH-IS triple combination; benzoate optimisation before ACTH improves IS response"},
    {"type": "Focal clonic (cortical, multifocal)", "pct": 35,
     "eeg": "Multifocal spikes/sharp-waves; reflects diffuse NMDAr cortical excitotoxicity",
     "semiology": "Face/limb clonic; secondary generalisation; multifocal pattern in neonatal → metabolic cause",
     "tips": "Multifocal focal = mandates plasma amino acids + CSF glycine ratio; sodium benzoate lowers glycine → reduces cortical excitability; LEV first-line"},
    {"type": "Tonic seizures (bilateral)", "pct": 40,
     "eeg": "Beta recruitment EMG artifact; bilateral tonic bursts in NREM",
     "semiology": "Rigid posturing; opisthotonic; bilateral; nocturnal predilection; falls risk if ambulatory",
     "tips": "CLB + KD for refractory tonic; avoid PHT/CBZ/OXC; sodium benzoate glycine-lowering reduces tonic seizure frequency in GCSH-NKH"},
    {"type": "Generalised Tonic-Clonic (GTCS)", "pct": 25,
     "eeg": "Generalised poly-spike then slow-wave; post-ictal attenuation",
     "semiology": "Tonic-clonic from generalised onset; less common in classic severe",
     "tips": "LEV + CLB; sodium benzoate optimisation; KD if persistent GTCS; avoid sodium channel blockers (CBZ/PHT/OXC)"},
]

TRIGGERS = [
    {"trigger": "Intercurrent illness / fever", "pct": 75,
     "mechanism": "Fever → catabolic state → protein catabolism → glycine release → H-protein absent (GCSH LOF) → BOTH GLDC and AMT simultaneously blocked → acute glycine surge; SE risk during febrile illness"},
    {"trigger": "Missed sodium benzoate dose", "pct": 65,
     "mechanism": "Sodium benzoate is primary glycine-depleting mechanism; missed dose → glycine reaccumulates 12–24h → plasma glycine rises → CSF glycine follows → NMDAr reactivation → breakthrough seizures"},
    {"trigger": "Protein-rich meal / amino acid load", "pct": 58,
     "mechanism": "Serine → glycine (SHMT interconversion); dietary protein → glycine liberation; GCSH-blocked GCS cannot clear → plasma glycine spike → CSF glycine rises within hours; pre-meal benzoate timing important"},
    {"trigger": "VPA / valproate exposure", "pct": 50,
     "mechanism": "VPA raises glycine via multiple secondary pathways in GCSH-NKH: GCS flux disequilibrium; sarcosine/N-methylglycine alternative disposal pathway inhibited; carnitine depletion reduces benzoate conjugation efficacy; net: glycine worsens"},
    {"trigger": "Sleep deprivation / disrupted sleep", "pct": 45,
     "mechanism": "NMDAr sensitivity heightened by sleep deprivation; NREM burst-suppression risk increased; circadian glycine transport modulation amplifies nighttime CSF glycine burden in NKH"},
    {"trigger": "Fasting / prolonged NPO", "pct": 42,
     "mechanism": "Fasting → muscle protein catabolism → serine/glycine release; gluconeogenesis uses glycine carbon; net glycine liberation exceeds GCS clearance (blocked at H-protein in GCSH-NKH); IV dextrose + benzoate coverage mandatory"},
    {"trigger": "Anesthesia / sedation (glycine/NMDAr interactions)", "pct": 30,
     "mechanism": "GlyR-potentiating anesthetics (propofol, barbiturates) add to GlyR brainstem depression (excess glycine already active); ketamine (NMDAr antagonist) may be BENEFICIAL; pre-operative metabolic team consultation mandatory"},
]

TREATMENTS = [
    {
        "drug": "Sodium Benzoate",
        "level": "Level A — First-Line (glycine-lowering backbone; identical to GLDC-NKH and AMT-NKH)",
        "dose": "250–750 mg/kg/day divided q6-8h; neonates: start 250 mg/kg/day; IV 500 mg/kg loading in SE; oral maintenance",
        "moa": "Benzoate conjugates glycine in mitochondria (GLYAT enzyme) → hippuric acid → renal excretion. Each benzoate removes one glycine. Targets the glycine pool INDEPENDENTLY of which GCS protein is defective (GLDC, AMT, or GCSH).",
        "efficacy": "Reduces plasma glycine 50–80% at therapeutic doses. Target: plasma glycine <500 µmol/L (ideally <300 µmol/L). Mechanism is GCS-independent — equally effective in GCSH-NKH as in GLDC-NKH and AMT-NKH.",
        "monitoring": "Plasma glycine every 3 months; annual CSF:plasma ratio; serum carnitine (benzoate DEPLETES carnitine — supplement 50–100 mg/kg/day MANDATORY); LFT quarterly; NH3 (hyperammonaemia at high doses)",
        "gcsh_note": "Sodium benzoate mechanism bypasses the blocked GCS entirely — acts via GLYAT-mediated conjugation (NOT GCS-dependent). H-protein is NOT involved in benzoate conjugation. Identical efficacy in GCSH-NKH vs GLDC-NKH and AMT-NKH. Carnitine co-supplementation EQUALLY MANDATORY."
    },
    {
        "drug": "Dextromethorphan (DXM)",
        "level": "Level B — NMDAr Antagonist (NKH-Specific Adjunct; identical to GLDC-NKH and AMT-NKH)",
        "dose": "Neonates: 5–10 mg/kg/day div q6-8h. Infants/children: 2–15 mg/kg/day. Maximum 35 mg/kg/day in refractory cases.",
        "moa": "Uncompetitive NMDAr channel blocker at Mg²⁺-binding site. Blocks NMDAr overactivation caused by excess glycine at GluN1 co-agonist site. Does NOT lower glycine — complements benzoate by reducing downstream NMDAr excitotoxicity.",
        "efficacy": "Reduces burst-suppression burden; IS frequency reduction; EEG background improvement in 40–60% of classic GCSH-NKH. Synergistic with sodium benzoate.",
        "monitoring": "CYP2D6 phenotyping if poor response (DXM → dextrorphan by CYP2D6; poor metabolizers accumulate DXM); sedation; respiratory depression in neonates; discontinue if no benefit at 4 weeks",
        "gcsh_note": "DXM mechanism is GCS-independent — blocks NMDAr regardless of which GCS gene is defective. CYP2D6 pharmacogenomics applicable in GCSH-NKH identically to GLDC-NKH and AMT-NKH. No population-specific CYP2D6 enrichment in GCSH-NKH (no founder allele)."
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — First-Line AED (SE + Maintenance; identical to GLDC-NKH and AMT-NKH)",
        "dose": "20–60 mg/kg/day div q12h; IV 60 mg/kg loading for SE; oral maintenance",
        "moa": "SV2A modulator; reduces presynaptic neurotransmitter release. No glycine interaction. Mechanistically safe in NKH.",
        "efficacy": "30–50% ≥50% seizure reduction; preferred IV agent for SE; myoclonic + focal reduction",
        "monitoring": "FBC (thrombocytopenia rare); serum creatinine (renal clearance); behaviour assessment (irritability 15–25% in ID patients)",
        "gcsh_note": "LEV has NO interaction with glycine metabolism or GCS regardless of which protein is defective. First-line IV agent for SE in GCSH-NKH — preferred over phenobarbital (PB potentiates GlyR brainstem depression on top of excess glycine)."
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B — Adjunct (Myoclonic + Tonic + Focal; identical to GLDC-NKH and AMT-NKH)",
        "dose": "0.1–0.5 mg/kg/day div q12-24h; maximum 40 mg/day adult; nocturnal dosing for tonic",
        "moa": "GABA-A PAM (α2/α3-selective benzodiazepine). Enhances Cl⁻ conductance. No glycine catabolism interaction.",
        "efficacy": "40–55% ≥50% reduction (myoclonic + tonic); tolerance develops 12–18 months; drug holiday protocol",
        "monitoring": "Sedation/cognition monthly; tolerance at 6 months; slow taper 10%/week",
        "gcsh_note": "CLB GABA-A mechanism orthogonal to GlyR — does not affect disease pathophysiology but reduces seizure frequency. Safe adjunct in GCSH-NKH."
    },
    {
        "drug": "ACTH (Corticotropin) — IS Management",
        "level": "Level A — Infantile Spasms in GCSH-NKH (identical preference to GLDC-NKH and AMT-NKH)",
        "dose": "High-dose ACTH 150 IU/m²/day IM or synthetic tetracosactide 0.5–0.75 mg/day x4 weeks then taper",
        "moa": "Reduces neuroinflammatory cascade in developing brain; reduces hypsarrhythmia. Evidence-based for IS regardless of etiology.",
        "efficacy": "60–70% spasm cessation; NKH-IS: combined ACTH + benzoate + DXM — best outcomes. ACTH preferred over VGB (VGB raises glycine — avoid in all NKH).",
        "monitoring": "BP (hypertension); glucose (hyperglycaemia); electrolytes (hypokalaemia); infection risk; cushingoid features",
        "gcsh_note": "ACTH preferred over VGB for IS in GCSH-NKH for same reason as GLDC-NKH and AMT-NKH: VGB raises glycine via GABA-T inhibition → GABA-glycine co-transporter upregulation. ACTH + benzoate + DXM triple combination = GCSH-NKH IS standard of care."
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — DRE Adjunct (identical to GLDC-NKH and AMT-NKH; modest glycine-lowering)",
        "dose": "4:1 ratio or MAD (20 g carb/day); minimum 3-month trial; ketosis BHB 2–4 mmol/L",
        "moa": "Reduces serine→glycine flux (SHMT pathway); protein-restricted component limits glycine substrate; BHB may modulate NMDAr",
        "efficacy": "30–40% seizure reduction in GCSH-NKH DRE; modest plasma glycine lowering (~10–15%); case series level evidence",
        "monitoring": "BHB monthly; lipids 3-monthly; growth; renal stone screen; glycine TDM",
        "gcsh_note": "KD in GCSH-NKH: reduces serine substrate for SHMT → limits glycine synthesis; modest GCS-independent glycine lowering additive to benzoate. Use as DRE adjunct after benzoate + DXM optimised."
    },
    {
        "drug": "Felbamate (FBM) — NMDAr Adjunct",
        "level": "Level C — Experimental NMDAr Adjunct (Refractory NKH only)",
        "dose": "15–45 mg/kg/day div q6-8h; very slow titration (aplastic anaemia risk mandates slow initiation)",
        "moa": "NMDAr glycine-site antagonist (strychnine-insensitive) + NMDAr channel blocker. DUAL NMDAr mechanism: glycine-site block (theoretically superior to DXM alone in NKH where glycine site is saturated).",
        "efficacy": "Limited GCSH-NKH data; case reports of burst-suppression reduction; use only after DXM failure",
        "monitoring": "FBC weekly x6 months then monthly (aplastic anaemia — Black Box Warning); LFT (hepatotoxicity); drug interactions (CYP450 induction)",
        "gcsh_note": "FBM glycine-site NMDAr antagonism is mechanistically attractive in GCSH-NKH but restricted by toxicity. Only under specialist metabolic epilepsy unit after failing DXM + optimised benzoate."
    },
    {
        "drug": "Valproic Acid (VPA) — HIGH RISK",
        "level": "HIGH RISK — Metabolic Worsening + Standard CIs",
        "dose": "AVOID — glycine worsening in all NKH types",
        "moa": "VPA raises glycine in GCSH-NKH via secondary pathways: (1) GCS flux disequilibrium — even though H-protein is absent, VPA disrupts residual GCS equilibrium in heterozygous or partial-function GCSH; (2) N-methylglycine (sarcosine) alternative disposal pathway inhibited; (3) carnitine depletion → reduces benzoate conjugation efficiency. GCSH unique: VPA cannot inhibit H-protein directly (H-protein has no enzyme activity), but secondary glycine-worsening is clinically equivalent to GLDC-NKH and AMT-NKH.",
        "efficacy": "May suppress some seizure types transiently but worsens underlying glycine accumulation — NOT recommended in any NKH genotype",
        "monitoring": "NOT RECOMMENDED. If used in extremis: LFT weekly; NH3 weekly; plasma glycine monthly; POLG1 exclusion MANDATORY",
        "gcsh_note": "VPA HIGH RISK IN GCSH-NKH: H-protein has no catalytic activity (carrier only) — VPA cannot inhibit it directly (unlike GLDC-NKH where VPA directly inhibits P-protein enzyme). Nevertheless, VPA raises glycine via secondary mechanisms (GCS equilibrium disruption, sarcosine pathway inhibition, carnitine depletion) with EQUIVALENT clinical risk. AVOID VPA in all NKH genotypes."
    },
    {
        "drug": "VGB (Vigabatrin) — HIGH RISK for IS",
        "level": "HIGH RISK — VGB raises CSF glycine in NKH; prefer ACTH for IS",
        "dose": "AVOID for IS in GCSH-NKH — use ACTH",
        "moa": "VGB inhibits GABA-T → GABA↑. GABA and glycine share inhibitory co-transport (GlyT2, SLC6A5). Elevated GABA may upregulate glycine reuptake via co-transporter competition → extracellular glycine rises.",
        "efficacy": "VGB effective for IS in general but disease-specific glycine risk in GCSH-NKH as in all NKH types. ACTH equally or more effective for IS.",
        "monitoring": "RETINAL TOXICITY mandatory (visual field + OCT or ERG pre-verbal every 3 months); if NKH: use ACTH first",
        "gcsh_note": "VGB HIGH RISK in GCSH-NKH for IS: mechanism is the same GABA-glycine co-transporter concern as GLDC-NKH and AMT-NKH. VGB retinal toxicity is ALSO a separate concern. DOUBLE RISK regardless of which GCS gene is defective."
    },
]

CONTRAINDICATIONS = [
    {"drug": "VPA (Valproic Acid)", "level": "HIGH RISK — METABOLIC WORSENING",
     "reason": "VPA raises glycine in GCSH-NKH via secondary pathways (GCS equilibrium disruption, sarcosine pathway inhibition, carnitine depletion reducing benzoate efficacy). H-protein has no enzymatic activity — VPA cannot directly inhibit it. Nevertheless secondary glycine-worsening is clinically equivalent to GLDC-NKH and AMT-NKH. POLG1 MANDATORY before any VPA use.",
     "alternative": "LEV + CLB + DXM + sodium benzoate. IV LEV for SE. IV ketamine (NMDAr antagonist — beneficial mechanism) for refractory SE."},
    {"drug": "VGB (Vigabatrin) for IS", "level": "HIGH RISK — PREFER ACTH",
     "reason": "VGB may raise CSF glycine via GABA-glycine co-transporter upregulation (GABA↑ from GABA-T inhibition → extracellular glycine rises). VGB retinal toxicity is additional independent risk. Both risks apply regardless of GLDC vs AMT vs GCSH genotype.",
     "alternative": "ACTH (Level A for IS in all NKH genotypes); combined ACTH + benzoate + DXM for NKH-IS."},
    {"drug": "CBZ / OXC / PHT (Na-channel blockers)", "level": "RELATIVE CI — Myoclonic Worsening",
     "reason": "Sodium channel blockers worsen myoclonic seizures in generalised epilepsies. GCSH-NKH has major myoclonic component (neonatal + IS period). CBZ/PHT documented to worsen generalised myoclonus.",
     "alternative": "LEV, CLB for myoclonus; KD for refractory myoclonus."},
    {"drug": "Phenobarbital (PB) — Neonatal/Maintenance", "level": "CAUTION — GlyR Additive + Respiratory",
     "reason": "PB potentiates GABA-A and GlyR. In neonatal GCSH-NKH: excess glycine already activates GlyR brainstem inhibition → apnea/hypotonia. PB ADDS to GlyR-mediated brainstem depression → excessive respiratory depression. Use ONLY if LEV failed as second-line for SE.",
     "alternative": "IV LEV 60 mg/kg (primary SE agent). IV ketamine (NMDAr antagonist — BENEFICIAL in NKH) as third-line."},
    {"drug": "Fasting / NPO without coverage", "level": "CAUTION — Glycine Surge",
     "reason": "Fasting → muscle catabolism → glycine release → glycine surge without GCS clearance (H-protein blocked in GCSH-NKH → BOTH GLDC and AMT stall). IV dextrose + sodium benzoate MANDATORY for any NPO period.",
     "alternative": "IV 10% dextrose + oral/IV sodium benzoate; target plasma glycine <500 µmol/L perioperatively."},
]

MONITORING = [
    {"parameter": "GCSH gene sequencing (WES/targeted GCS panel: GCSH + AMT + GLDC + DLD)", "frequency": "At diagnosis (confirmatory)", "target": "Biallelic pathogenic GCSH variants; ACMG classification; GCSH enzyme assay not clinically available (H-protein has no catalytic activity — carrier only); biochemically cannot distinguish GCSH from GLDC or AMT — gene panel mandatory"},
    {"parameter": "CSF:Plasma glycine ratio (simultaneous)", "frequency": "At diagnosis; annually on treatment", "target": "Diagnostic: ratio ≥0.08; treatment target: ratio <0.04; plasma glycine <500 µmol/L (ideally <300 µmol/L); identical target to GLDC-NKH and AMT-NKH"},
    {"parameter": "Plasma amino acid quantitative (glycine focus)", "frequency": "Every 3 months (sodium benzoate titration)", "target": "Glycine <500 µmol/L; serine (SHMT substrate); folate metabolites (identical folate perturbation to GLDC-NKH: 5,10-methyleneTHF not produced since full GCS cycle blocked)"},
    {"parameter": "Urine organic acids (glycine, hippurate)", "frequency": "Every 3 months on sodium benzoate", "target": "Hippuric acid excretion confirms benzoate conjugation; exclude 'ketotic hyperglycinemia' (propionic/MMA — organic acids differentiate)"},
    {"parameter": "Serum carnitine (free + acylcarnitine)", "frequency": "Every 3 months", "target": "Free carnitine >20 µmol/L; benzoate conjugation depletes carnitine; supplement L-carnitine 50–100 mg/kg/day"},
    {"parameter": "EEG (video-EEG — continuous neonatal)", "frequency": "Continuous neonatal phase; annual + urgent for clinical change", "target": "Burst-suppression burden; EEG-only IS detection (GCSH-NKH: IS without motor manifestation common due to hypotonia); hypsarrhythmia on ACTH resolution; SE detection"},
    {"parameter": "MRI brain (3T + DWI in acute)", "frequency": "At diagnosis; 6 months; annually first 3 years", "target": "Periventricular WM hypomyelination; thin/absent corpus callosum (especially splenium); DWI restriction acute GCSH-NKH; cerebellar hypoplasia severe; identical pattern to GLDC-NKH and AMT-NKH"},
    {"parameter": "Plasma folate + homocysteine + methionine", "frequency": "Every 6 months", "target": "GCSH LOF blocks full GCS → 5,10-methyleneTHF not produced from glycine (same as GLDC-NKH and AMT-NKH); low plasma folate + elevated homocysteine → folate supplementation"},
    {"parameter": "POLG1 WES exclusion (mandatory pre-VPA)", "frequency": "MANDATORY before VPA", "target": "Biallelic POLG1 → VPA ABSOLUTE CI (Alpers); GCSH-NKH metabolic vulnerability adds to POLG1 standard CI"},
    {"parameter": "Liver function + NH3 (sodium benzoate monitoring)", "frequency": "Monthly first 6 months; quarterly stable", "target": "ALT/AST <3× ULN; NH3 <80 µmol/L; GGT, ALP"},
    {"parameter": "Developmental / cognitive assessment (Bayley/Vineland)", "frequency": "Every 6 months first 3 years; annual thereafter", "target": "Motor milestones; language; adaptive behaviour; Vineland-3 composite; track impact of glycine control on trajectory"},
    {"parameter": "CYP2D6 pharmacogenomics (DXM metabolism)", "frequency": "Once at baseline before DXM", "target": "CYP2D6 poor metabolizer → DXM accumulates; ultra-rapid → reduced effect. No population-specific enrichment in GCSH-NKH (no founder allele; pan-ethnic rare)"},
    {"parameter": "Growth anthropometry + nutrition", "frequency": "Every 3 months first 2 years; 6-monthly thereafter", "target": "Weight/height z-score; head circumference; folate levels; serine levels; carnitine; folate supplementation as needed"},
]

LIFECYCLE = [
    {"stage": "Prenatal / Sibling Screening", "age": "Prenatal", "description": "Known GCSH family: CVS or amniocentesis for GCSH genotyping. Standard NBS (tandem MS): does NOT detect NKH (glycine not on standard NBS panels). GCSH-NKH is ultra-rare — prenatal genetic diagnosis essential for known GCSH biallelic carrier parents."},
    {"stage": "Neonatal Crisis (Classic GCSH-NKH)", "age": "0–7 days", "description": "NICU admission hours after birth. Apnea (MV required), profound hypotonia, absent Moro/suck, HICCUPS (pathognomonic — phrenic GlyR). CSF:plasma glycine ratio STAT (simultaneous). EEG: burst-suppression immediately. START: IV sodium benzoate + IV LEV + DXM oral/NG. POLG1 exclusion initiated. GCSH-NKH biochemically identical to GLDC-NKH and AMT-NKH at this stage — gene panel (GCSH + GLDC + AMT + DLD) sent simultaneously."},
    {"stage": "Post-acute Neonatal / Gene Identification", "age": "2–8 weeks", "description": "Wean ventilator (if survived). Gene panel result → confirms GCSH (not GLDC or AMT). Burst-suppression persists → transitions to multifocal/hypsarrhythmia. Oral benzoate titration. DXM optimisation. MRI: WM hypomyelination, thin CC. CYP2D6 phenotyping for DXM. GCSH pathogenic variants catalogued — ultra-rare, likely novel or private."},
    {"stage": "Infantile Phase (IS / Epilepsy)", "age": "3–18 months", "description": "IS onset in survivors (45–50%): hypsarrhythmia + spasms. ACTH x4 weeks (NOT VGB). GCSH-NKH IS: ACTH + benzoate + DXM triple combination. Developmental regression. Feeding difficulties. PT, OT, speech therapy. Seizure types evolve: myoclonic + tonic. Folate supplementation initiated if homocysteine elevated."},
    {"stage": "Early Childhood (DRE Management)", "age": "18 months–6 years", "description": "DRE management (classic GCSH-NKH); attenuated GCSH-NKH: epilepsy may be manageable; KD initiation for DRE; school/rehabilitation planning; cognitive plateau vs slow gains; home seizure care plan; annual MRI; plasma folate/homocysteine monitoring."},
    {"stage": "School Age / Adolescence", "age": "6–18 years", "description": "Classic GCSH-NKH: specialized care. Attenuated: mainstream school with support (partial H-protein function: better outcomes). Mood disorders (depression/anxiety 30–40%); puberty-related seizure change; VPA CI continues; benzoate compliance monitoring; GCSH variant re-interpretation with updated databases."},
    {"stage": "Adulthood (Chronic Management)", "age": "18+ years", "description": "Classic GCSH-NKH: specialized residential care. Attenuated: supported independent living possible. Renal monitoring (long-term benzoate → hippurate → renal load). Genetic counselling (25% sibling recurrence AR). Clinical trial enrolment — GCS carrier replacement / H-protein mRNA therapy / AAV9-GCSH gene therapy in early research 2026."},
]

CONCEPTS = [
    {"term": "GCSH / H-protein / 16q23.2", "definition": "Glycine Cleavage System H-protein; CENTRAL CARRIER of GCS; 125 aa, ~16 kDa; mitochondrial matrix; lipoic acid covalently attached to Lys59 (amide bond via lipoate ligase); NO catalytic activity — CARRIER ONLY. Physically shuttles aminomethyl group from P-protein (GLDC) to T-protein (AMT) via 'swinging arm' conformational change (~14 Å). AR biallelic LOF → NKH. OMIM *238330. ~1% of all NKH."},
    {"term": "GCSH LOF — DUAL UPSTREAM AND DOWNSTREAM GCS BLOCK", "definition": "When H-protein is absent, the GCS is blocked at TWO points simultaneously: (1) P-protein (GLDC) CANNOT transfer its aminomethyl intermediate — no H-protein lipoamide to receive it → Step 1 stalls. (2) T-protein (AMT) has NO SUBSTRATE — no loaded H-protein to donate aminomethyl → Step 3 stalls. Contrast: GLDC-NKH = upstream block only; AMT-NKH = downstream block only + upstream secondary. GCSH-NKH = complete dual block. Net glycine accumulation is IDENTICAL across all three."},
    {"term": "Lipoyl Domain — Lys59 Attachment", "definition": "H-protein contains a lipoyl domain: lipoic acid (a dithiolane fatty acid) is covalently linked via amide bond to ε-amino group of Lys59 by mitochondrial lipoate ligase (LIAS). The dithiolane ring cycles between: (1) oxidised (disulfide, unloaded) state: can accept aminomethyl from GLDC; (2) reduced/aminomethyl-loaded state (after GLDC step): presents aminomethyl to AMT. GCSH LOF → no H-protein lipoyl domain → GCS cannot cycle."},
    {"term": "GCS Biochemical Identity — GCSH vs GLDC vs AMT", "definition": "NKH caused by GCSH (~1%), GLDC (75–80%), or AMT (15%) is BIOCHEMICALLY IDENTICAL: same CSF:plasma glycine ratio elevation (≥0.08), same plasma glycine range, same CSF glycine range. Gene panel (GCSH + GLDC + AMT + DLD) is MANDATORY. Treatment is also identical. GCSH rarest: ~20–50 cases worldwide 2026."},
    {"term": "CSF:Plasma Glycine Ratio — DIAGNOSTIC THRESHOLD", "definition": "SIMULTANEOUS collection mandatory (fasting; LP sedation with ketamine preferred). Normal: <0.02. GCSH-NKH: ≥0.08 (classic: 0.20–0.55; attenuated: 0.08–0.18). Plasma glycine alone insufficient (elevated in propionic/MMA/IVA — 'ketotic hyperglycinemia'). Ratio ≥0.08 is highly specific for NKH (all three types: GLDC/AMT/GCSH)."},
    {"term": "HICCUPS — Pathognomonic NKH Clue (All Types)", "definition": "Phrenic nerve nucleus (C3–C5 anterior horn) uses glycinergic inhibitory interneurons. Excess glycine → GlyR over-activation at phrenic nucleus → rhythmic uncontrolled diaphragm contractions = HICCUPS. Applies to GCSH-NKH identically to GLDC-NKH and AMT-NKH. Persistent neonatal hiccups + hypotonia + apnea = NKH (any GCS gene) until proven otherwise."},
    {"term": "VPA HIGH RISK — GCSH-NKH No-Direct-Inhibition But Equivalent Clinical Risk", "definition": "Unlike GLDC-NKH (VPA directly inhibits P-protein enzyme) or AMT-NKH (VPA secondary mechanisms), in GCSH-NKH the H-protein has NO catalytic activity — VPA cannot inhibit it directly. However, VPA raises glycine via: (1) GCS equilibrium disruption; (2) sarcosine pathway inhibition; (3) carnitine depletion → reduced benzoate efficacy. Clinical glycine-worsening is EQUIVALENT. AVOID VPA in all NKH (GLDC/AMT/GCSH)."},
    {"term": "Sodium Benzoate — GCS-Independent Glycine Removal (GCSH)", "definition": "Sodium benzoate works via GLYAT (glycine N-acyltransferase) — an enzyme STRUCTURALLY INDEPENDENT of GCS and H-protein. Benzoate conjugates glycine → hippuric acid → renal excretion. H-protein plays NO role in benzoate conjugation. Therefore, sodium benzoate has IDENTICAL efficacy in GCSH-NKH vs GLDC-NKH and AMT-NKH. Carnitine depletion via CoA pathway is the same concern."},
    {"term": "Burst-Suppression EEG — GCSH-NKH Neonatal Signature", "definition": "Burst-suppression within hours-days of birth is hallmark neonatal NKH EEG regardless of GCS gene. In GCSH-NKH: identical EEG pattern to GLDC-NKH and AMT-NKH. NMDAr excitotoxicity driven by glycine accumulation (GCSH LOF → same glycine burden as GLDC LOF). Continuous video-EEG mandatory in neonatal period — electrical SE without motor manifestation (hypotonia masks clinical SE)."},
    {"term": "GCSH Gene Therapy / H-Protein Replacement — 2026 Pipeline", "definition": "AAV9-GCSH gene therapy: early preclinical stage 2026; H-protein (125 aa, small) is ideal AAV cargo. mRNA therapy (lipid nanoparticle-GCSH-mRNA): hepatocyte-targeted; H-protein mRNA small and well-suited for LNP. Protein replacement: H-protein is soluble (not membrane-bound) — potential for mitochondria-targeted delivery via MTS fusion proteins. GCS enzyme replacement complex not feasible for the full 4-protein complex, but H-protein alone might rescue function if GLDC and AMT are intact."},
    {"term": "No GCSH Founder Allele — Pan-Ethnic Ultra-Rare", "definition": "Unlike AMT-NKH (p.Arg320His semi-founder in East Asian/Japanese) and GLDC-NKH (p.Gly761Arg European founder for attenuated), GCSH-NKH has NO identified founder allele. All GCSH pathogenic variants reported to date are private/family-specific. GCSH-NKH occurs pan-ethnically at ultra-low frequency. This means CYP2D6 population-specific concerns do not apply, and attenuated GCSH-NKH is identified only by sequencing — not clinical clustering."},
]

THRESHOLDS = [
    {"parameter": "CSF:Plasma glycine ratio (normal)", "value": "<0.02", "clinical": "Normal GCS function; no NKH regardless of gene"},
    {"parameter": "CSF:Plasma glycine ratio (GCSH-NKH diagnostic)", "value": "≥0.08", "clinical": "Highly specific for NKH (all types GLDC/AMT/GCSH); classic GCSH: 0.20–0.55"},
    {"parameter": "Plasma glycine (normal)", "value": "150–260 µmol/L", "clinical": "Normal (lab-dependent); GCSH-NKH: same range as GLDC-NKH and AMT-NKH"},
    {"parameter": "Plasma glycine (GCSH-NKH untreated)", "value": "600–3000 µmol/L", "clinical": "Similar range to GLDC-NKH and AMT-NKH; classic GCSH may be higher (dual block)"},
    {"parameter": "Target plasma glycine on sodium benzoate", "value": "<500 µmol/L (ideally <300 µmol/L)", "clinical": "Same glycine-lowering target as GLDC-NKH and AMT-NKH; TDM every 3 months"},
    {"parameter": "CSF glycine (GCSH-NKH)", "value": ">100 µmol/L (often 150–2000 µmol/L)", "clinical": "Identical range to GLDC-NKH and AMT-NKH; confirms NKH pathophysiology"},
    {"parameter": "Serum carnitine (target on benzoate)", "value": ">20 µmol/L free carnitine", "clinical": "Supplement L-carnitine 50–100 mg/kg/day; benzoate depletes carnitine identically in GCSH-NKH"},
    {"parameter": "Plasma homocysteine (folate monitoring)", "value": "<10 µmol/L", "clinical": "GCSH LOF → full GCS blocked → 5,10-methyleneTHF not produced → folate cycle perturbation; supplement if elevated"},
    {"parameter": "Plasma folate", "value": ">6 ng/mL", "clinical": "GCS block (any gene: GLDC/AMT/GCSH) → folate cycle perturbation; supplement if low"},
    {"parameter": "NH3 on sodium benzoate (threshold)", "value": "<80 µmol/L", "clinical": "Hyperammonaemia risk at high benzoate doses; >80 → dose reduction; monitor as in GLDC-NKH and AMT-NKH"},
    {"parameter": "H-protein enzyme activity", "value": "Not clinically applicable", "clinical": "H-protein has NO catalytic activity — carrier only. No enzymatic assay clinically available. Diagnosis is by gene sequencing only."},
]


def _patient_profile(i):
    pat_id = f"GCSH-{i+1:03d}"
    sex = "F" if rng.random() < 0.5 else "M"
    etiology = _rng_choice(ETIOLOGIES)
    if "Classic" in etiology["etiology"] or "homozygous null" in etiology["etiology"] or "consanguineous" in etiology["etiology"]:
        phenotype_class = "Classic Neonatal"
        onset_age_days = _rng_int(0, 3)
    elif "Attenuated" in etiology["etiology"]:
        phenotype_class = "Attenuated"
        onset_age_days = _rng_int(60, 365)
    else:
        phenotype_class = "Transient"
        onset_age_days = _rng_int(0, 5)
    has_epilepsy = phenotype_class != "Transient" or rng.random() < 0.3
    has_is = phenotype_class == "Classic Neonatal" and rng.random() < 0.45
    has_bs = phenotype_class == "Classic Neonatal" and rng.random() < 0.62
    dre = has_epilepsy and (phenotype_class == "Classic Neonatal" and rng.random() < 0.80 or
                            phenotype_class == "Attenuated" and rng.random() < 0.20)
    on_benzoate = rng.random() < 0.92
    on_dxm = rng.random() < 0.78
    on_lev = rng.random() < 0.72
    on_clb = rng.random() < 0.42
    on_acth = has_is and rng.random() < 0.80
    on_kd = dre and rng.random() < 0.28
    # No founder allele in GCSH-NKH
    plasma_gly = _rng_float(180, 580, 0) if on_benzoate else _rng_float(650, 3000, 0)
    csf_plasma_ratio = (_rng_float(0.12, 0.55, 3) if phenotype_class == "Classic Neonatal"
                        else _rng_float(0.08, 0.18, 3) if phenotype_class == "Attenuated"
                        else _rng_float(0.08, 0.12, 3))
    carnitine_ok = rng.random() < 0.72
    folate_low = rng.random() < 0.38  # GCS fully blocked → folate deficit
    confidence = _rng_float(0.74, 0.99)
    return {
        "patient_id": pat_id,
        "sex": sex,
        "phenotype_class": phenotype_class,
        "onset_age_days": int(onset_age_days),
        "etiology": etiology["etiology"][:75],
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
        "plasma_glycine_umol": int(plasma_gly),
        "csf_plasma_ratio": csf_plasma_ratio,
        "carnitine_normal": carnitine_ok,
        "folate_low": folate_low,
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
    n_folate_low = sum(1 for p in PATIENTS if p["folate_low"])
    avg_plasma_gly = round(sum(p["plasma_glycine_umol"] for p in PATIENTS) / N, 1)
    avg_ratio = round(sum(p["csf_plasma_ratio"] for p in PATIENTS) / N, 3)
    return {
        "dashboard": "GCSH Epilepsy — Non-Ketotic Hyperglycinemia (NKH) / H-protein (Glycine Cleavage System H-protein) Deficiency",
        "gene": "GCSH (16q23.2) — H-protein; Central Carrier of GCS; 125 aa ~16 kDa; mitochondrial matrix; lipoamide swinging arm; NO catalytic activity; ~1% of NKH",
        "inheritance": "Autosomal Recessive (AR) biallelic LOF; ~1% of NKH; NKH overall ~1:60,000–76,000; GCSH-NKH ~1:6,000,000–7,600,000; ~20–50 cases worldwide 2026",
        "omim_gene": "238330",
        "omim_disease": "605899",
        "locus": "16q23.2",
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
        "n_folate_low": n_folate_low,
        "folate_low_pct": round(100 * n_folate_low / N),
        "avg_plasma_glycine": avg_plasma_gly,
        "avg_csf_plasma_ratio": avg_ratio,
        "phenotype_classes": PHENOTYPE_CLASSES,
        "etiologies": ETIOLOGIES,
        "key_concepts": [c["term"] for c in CONCEPTS[:8]],
        "high_risk_drugs": ["VPA (valproic acid) — secondary glycine-raising; H-protein no enzymatic activity but clinically equivalent risk",
                            "VGB (vigabatrin) for IS — GABA-glycine co-transporter; prefer ACTH",
                            "CBZ/PHT/OXC — relative CI (myoclonic worsening)",
                            "Phenobarbital — CAUTION (GlyR additive brainstem depression + respiratory)"],
        "pathognomonic_sign": "HICCUPS + profound hypotonia + apnea in neonates = NKH until proven otherwise (all GCS genes: GLDC/AMT/GCSH)",
        "unique_mechanism": "GCSH LOF = DUAL BLOCK: P-protein (GLDC) stalls upstream (no H-protein to receive aminomethyl) AND T-protein (AMT) stalls downstream (no loaded H-protein substrate) — most complete GCS block of all NKH types",
        "diagnostic_biomarker": "CSF:plasma glycine ratio ≥0.08 (simultaneous) — identical to GLDC-NKH and AMT-NKH; gene panel (GCSH+GLDC+AMT+DLD) mandatory to identify causative gene",
        "founder_allele": "NONE — GCSH-NKH is pan-ethnic ultra-rare; all variants private/family-specific; contrast AMT (p.Arg320His East Asian) and GLDC (p.Gly761Arg European)",
        "worldwide_cases_2026": "~20–50 GCSH-NKH cases (rarest NKH gene; fewest reported worldwide)",
        "standards": [
            "Kure S et al. Nonketotic hyperglycinemia: biochemical, molecular, and neurological aspects. J Inherit Metab Dis 1997.",
            "Nanao K et al. Identification of H-protein gene mutations in non-ketotic hyperglycinemia. J Inherit Metab Dis 1994.",
            "Hamosh A & Johnston MV. Nonketotic hyperglycinemia. OMIM #605899. 2024.",
            "Van Hove JLK et al. Long-term outcome and management of NKH. J Inherit Metab Dis 2006.",
            "García-Cazorla A et al. NKH: current state and research perspectives. Orphanet J Rare Dis 2022.",
            "ACMG/AMP — Variant interpretation standards for NKH/GCSH 2024.",
        ],
        "per_patient_kpis": sorted(PATIENTS, key=lambda p: p["csf_plasma_ratio"], reverse=True),
    }


def get_breakdown():
    classic = sum(1 for p in PATIENTS if p["phenotype_class"] == "Classic Neonatal")
    attenuated = sum(1 for p in PATIENTS if p["phenotype_class"] == "Attenuated")
    transient = sum(1 for p in PATIENTS if p["phenotype_class"] == "Transient")

    ratio_ranges = [(0.0, 0.08, "<0.08 (not NKH / transient resolving)"),
                    (0.08, 0.18, "0.08–0.18 (attenuated GCSH)"),
                    (0.18, 0.35, "0.18–0.35 (moderate classic)"),
                    (0.35, 1.0, ">0.35 (severe classic)")]
    ratio_hist = []
    for lo, hi, label in ratio_ranges:
        cnt = sum(1 for p in PATIENTS if lo <= p["csf_plasma_ratio"] < hi)
        ratio_hist.append({"range": label, "n": cnt, "pct": round(100 * cnt / N)})

    gly_ranges = [(0, 300, "<300 µmol/L (on-target)"),
                  (300, 500, "300–500 (partially controlled)"),
                  (500, 1000, "500–1000 (subtherapeutic)"),
                  (1000, 5000, ">1000 (uncontrolled)")]
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
        "title": "GCSH / Non-Ketotic Hyperglycinemia — Definitions, GCS Pathway, H-Protein Central Carrier, Pharmacology",
        "gene_card": {
            "gene": "GCSH",
            "locus": "16q23.2",
            "protein": "H-protein (Glycine Cleavage System H-protein / Hydrogen carrier protein)",
            "size": "125 aa, ~16 kDa",
            "family": "Glycine cleavage system H-protein; lipoate-bearing carrier; swinging arm scaffold; NO catalytic activity",
            "structure": "Mitochondrial matrix; lipoic acid covalently attached to Lys59 (amide bond); undergoes ~14 Å conformational swing to transfer aminomethyl group from P-protein (GLDC) to T-protein (AMT); cycles between oxidised (unloaded) and reduced/aminomethyl-loaded states",
            "cofactor": "Lipoic acid (covalently attached; not free cofactor); lipoate ligase (LIAS) installs lipoic acid on Lys59",
            "localisation": "Mitochondrial matrix (N-terminal MTS signal)",
            "omim_gene": "*238330",
            "omim_disease": "#605899",
            "inheritance": "AR biallelic LOF",
            "cause_of_nkh": "~1% of NKH; GLDC ~75–80%; AMT ~15%; GCSH ~1%; DLD very rare",
        },
        "pathway": {
            "name": "Glycine Cleavage System (GCS) — 4-Protein Complex; H-Protein (GCSH) Is The Central Carrier (Step 2)",
            "steps": [
                {"step": 1, "enzyme": "P-protein (GLDC)", "gene": "GLDC", "cofactor": "PLP",
                 "reaction": "Glycine + H-protein(oxidised) → CO₂ + aminomethyl-H-protein",
                 "clinical": "GLDC LOF (75–80% NKH): P-protein absent → GCS cannot start. IN GCSH-NKH: P-protein structurally INTACT but CANNOT function (no H-protein lipoamide to receive aminomethyl group → Step 1 stalls)."},
                {"step": 2, "enzyme": "H-protein (GCSH) — THE DEFECTIVE CARRIER IN GCSH-NKH", "gene": "GCSH", "cofactor": "Lipoic acid (Lys59)",
                 "reaction": "Aminomethyl-H-protein shuttles aminomethyl group from P-protein to T-protein (swinging arm, ~14 Å conformational change)",
                 "clinical": "GCSH LOF (1% NKH): H-protein ABSENT → (1) P-protein cannot transfer aminomethyl (no H-protein) → Step 1 stalls; (2) T-protein has no substrate (no loaded H-protein) → Step 3 stalls. DUAL UPSTREAM+DOWNSTREAM BLOCK — most complete GCS block of all NKH types."},
                {"step": 3, "enzyme": "T-protein (AMT)", "gene": "AMT", "cofactor": "THF",
                 "reaction": "Aminomethyl-H-protein + THF → 5,10-methyleneTHF + NH₄⁺ + H-protein(oxidised)",
                 "clinical": "AMT LOF (15% NKH): T-protein absent → aminomethyl cannot discharge to THF → H-protein remains loaded → upstream block. IN GCSH-NKH: T-protein INTACT but has NO SUBSTRATE (no loaded H-protein)."},
                {"step": 4, "enzyme": "L-protein (DLD)", "gene": "DLD", "cofactor": "NAD+/FAD",
                 "reaction": "H-protein(reduced) + NAD⁺ → H-protein(oxidised) + NADH",
                 "clinical": "DLD is shared with pyruvate DH + alpha-KG DH. IN GCSH-NKH: DLD cannot act on absent H-protein. DLD LOF → combined dehydrogenase defect + glycine accumulation (distinct from classic NKH)."},
            ],
            "net_reaction": "Glycine + THF + NAD⁺ → 5,10-methyleneTHF + CO₂ + NH₄⁺ + NADH",
            "gcsh_lof_consequence": "GCSH LOF blocks Step 2 carrier → P-protein (GLDC) stalls upstream (no H-protein) AND T-protein (AMT) stalls downstream (no substrate) → entire GCS inoperable at TWO points → glycine accumulates identically to GLDC-NKH and AMT-NKH + 5,10-methyleneTHF not produced",
        },
        "biomarkers": [
            {"marker": "CSF:Plasma glycine ratio (simultaneous)", "method": "Quantitative plasma + CSF amino acids (LC-MS/MS)",
             "reference_range": "<0.02", "nkh_range": "≥0.08 (classic GCSH: 0.20–0.55; attenuated: 0.08–0.18)",
             "notes": "PRIMARY DIAGNOSTIC TEST for NKH. GCSH-NKH is biochemically IDENTICAL to GLDC-NKH and AMT-NKH. Gene panel (GCSH+GLDC+AMT+DLD) required to identify causative gene. Simultaneous collection mandatory."},
            {"marker": "Plasma glycine (quantitative)", "method": "Quantitative amino acid panel (LC-MS/MS)",
             "reference_range": "150–260 µmol/L", "nkh_range": "650–3000+ µmol/L (untreated GCSH-NKH)",
             "notes": "Same range as GLDC-NKH and AMT-NKH. Not specific — elevated in propionic/MMA/IVA (ketotic hyperglycinemia). Ratio is diagnostic; plasma level is monitoring. Target on benzoate: <500 µmol/L."},
            {"marker": "Plasma folate + homocysteine + serine", "method": "Plasma metabolomics / amino acids + folate panel",
             "reference_range": "Folate >6 ng/mL; homocysteine <10 µmol/L; serine 65–150 µmol/L",
             "nkh_range": "GCSH-NKH: folate may be low; homocysteine may be elevated (5,10-methyleneTHF deficit — full GCS blocked)",
             "notes": "GCSH LOF → full GCS blocked → 5,10-methyleneTHF not produced (same as GLDC-NKH and AMT-NKH). Monitor folate cycle. Folate supplementation if homocysteine elevated."},
            {"marker": "H-protein functional assay", "method": "NOT CLINICALLY APPLICABLE",
             "reference_range": "N/A — H-protein has no catalytic activity", "nkh_range": "N/A",
             "notes": "H-protein is a CARRIER ONLY — no enzymatic activity to measure. Diagnosis is exclusively by gene sequencing. Contrast with GLDC (enzyme activity measurable in lymphocytes/liver) and AMT (enzyme activity assay, technically demanding). GCSH diagnosis = WES or targeted GCS gene panel."},
        ],
        "key_concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "treatments": TREATMENTS,
        "references": [
            "Kure S et al. Biochemical and molecular analysis of the glycine cleavage system H-protein. J Inherit Metab Dis 1997.",
            "Nanao K et al. Identification of H-protein gene mutations causing non-ketotic hyperglycinemia. J Inherit Metab Dis 1994.",
            "Hamosh A & Johnston MV. Nonketotic hyperglycinemia. OMIM #605899. 2024.",
            "Van Hove JLK et al. Long-term outcome and management of NKH. J Inherit Metab Dis 2006.",
            "García-Cazorla A et al. NKH: current state and research perspectives. Orphanet J Rare Dis 2022.",
            "NKH International Family Network — Clinical care guidelines 2023.",
        ],
        "differential_diagnosis": [
            {"condition": "GLDC-NKH (P-protein deficiency)", "distinction": "Biochemically IDENTICAL — CSF:plasma ratio ≥0.08, same glycine range. Only gene panel distinguishes GCSH from GLDC. GLDC is ~75–80% of NKH. Treatment identical. GLDC has p.Gly761Arg European founder (attenuated); GCSH has NO founder allele."},
            {"condition": "AMT-NKH (T-protein deficiency)", "distinction": "Biochemically IDENTICAL — same ratio range, same glycine levels. AMT ~15% of NKH. Gene panel mandatory. Treatment identical. AMT has p.Arg320His East Asian semi-founder; GCSH has NO founder allele."},
            {"condition": "SSADH deficiency (ALDH5A1)", "distinction": "SSADH: GHB ↑↑ in urine/plasma (NOT glycine); glycine NORMAL. CSF:plasma ratio NORMAL. Globus pallidus T2 hyperintensity. VGB ABSOLUTE CI (different mechanism). No burst-suppression neonatal."},
            {"condition": "Propionic acidemia (PCCA/PCCB) — 'Ketotic hyperglycinemia'", "distinction": "Propionic acidemia: glycine ↑ plasma BUT via methylamine-glycine conjugation saturation. Organic acids: 3-OH-propionate + propionylglycine + methylcitrate. NEVER diagnose NKH without urine organic acids. CSF:plasma ratio NORMAL in propionic acidemia."},
            {"condition": "Methylmalonic acidemia (MUT/MMAA etc.) — 'Ketotic hyperglycinemia'", "distinction": "MMA: glycine ↑; methylmalonic acid ↑↑ urine; C3 acylcarnitine; B12-responsive forms exist. Urine organic acids differentiate immediately from GCSH-NKH."},
            {"condition": "DLD deficiency (L-protein) — Combined NKH + dehydrogenase", "distinction": "DLD LOF: glycine ↑ (NKH component) + pyruvate/lactate ↑ + alpha-KG ↑ (combined dehydrogenase deficit). COMBINED metabolic profile distinguishes DLD from GCSH. DLD shares L-protein with pyruvate DH and alpha-KG DH complexes."},
        ],
    }
