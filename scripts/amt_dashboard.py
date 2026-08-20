#!/usr/bin/env python3
"""AMT (Aminomethyltransferase — T-protein of GCS) Epilepsy Dashboard.

AMT encodes the T-protein (Aminomethyltransferase) of the mitochondrial Glycine Cleavage System
(GCS) — the SECOND most common cause of Non-Ketotic Hyperglycinemia (NKH / Glycine Encephalopathy),
accounting for ~15% of all NKH cases (GLDC P-protein ~75–80%, AMT T-protein ~15%, GCSH H-protein ~1%).

GCS — 4-PROTEIN COMPLEX (mitochondrial matrix):
  • P-protein (GLDC, 1020 aa): PLP-dependent; oxidatively decarboxylates glycine → aminomethyl
    intermediate transferred to lipoamide of H-protein; releases CO₂
  • H-protein (GCSH, 125 aa): lipoic acid-bearing carrier; accepts aminomethyl group from P-protein,
    shuttles it to T-protein (AMT); H-protein cycles between aminomethyl-loaded (reduced) and
    unloaded (oxidised) states
  • T-protein (AMT, 403 aa): aminomethyltransferase; accepts aminomethyl group from loaded
    H-protein; transfers it to THF → 5,10-methyleneTHF + NH₄⁺; simultaneously regenerates
    oxidised H-protein so GCS can cycle again
  • L-protein (DLD, shared): lipoamide dehydrogenase; regenerates H-protein lipoic acid via NAD+
    (shared with pyruvate DH and alpha-KG DH complexes)

GCS NET REACTION:
  Glycine + THF + NAD⁺ → 5,10-methyleneTHF + CO₂ + NH₄⁺ + NADH

AMT LOF — THE T-PROTEIN BOTTLENECK:
  When AMT is absent/dysfunctional, the aminomethyl group CANNOT be transferred from H-protein
  to THF. Consequence: H-protein remains permanently LOADED (aminomethyl-H-protein cannot discharge).
  Backed-up H-protein → P-protein (GLDC) has NO free H-protein to accept its aminomethyl
  intermediate → GLDC reaction stalls → entire GCS is blocked upstream.
  Net result: IDENTICAL to GLDC-NKH biochemically — glycine accumulates in ALL compartments
  (plasma, CSF, urine, brain interstitial fluid).
  ADDITIONAL DEFICIT: 5,10-methyleneTHF is NOT produced from the glycine → THF step →
  folate one-carbon metabolism perturbation is EQUAL TO or GREATER THAN in GLDC-NKH
  (GLDC produces the aminomethyl intermediate; AMT converts it; both are required for
  5,10-methyleneTHF production from glycine).

PATHOPHYSIOLOGY — SAME DUAL GLYCINE RECEPTOR PARADOX AS GLDC-NKH:
  GlyR (GLRA1/GLRB): Cl⁻ channel (inhibitory) — brainstem/spinal cord/reticular formation.
  Neonates: excess glycine → GlyR OVER-ACTIVATION → profound hypotonia + apnea + HICCUPS
  (phrenic nucleus GlyR — C3–C5).
  NMDAr (GluN1/GluN2): GluN1 obligate glycine co-agonist site (Km ~0.5–5 µM). CSF glycine in
  NKH >500 µM → GluN1 site SATURATED → maximum NMDAr excitotoxicity → burst-suppression → SE.

DIAGNOSTIC BIOMARKER — IDENTICAL TO GLDC-NKH:
  CSF:plasma glycine ratio (SIMULTANEOUS): normal <0.02; NKH ≥0.08. AMT-NKH: same ratio
  range as GLDC-NKH. AMT cannot be distinguished from GLDC biochemically — gene panel required.

GENETICS:
  Gene: AMT at 3p21.2; 9 exons; 403 amino acids; ~45 kDa; mitochondrial matrix; THF-binding
  aminomethyltransferase; PLP-independent (unlike GLDC P-protein); accepts aminomethyl from
  reduced lipoamide-GCSH. AR biallelic LOF. OMIM gene *238310.
  ~100–200 AMT pathogenic variants reported (many private/family-specific — no pan-ethnic founder).
  KEY RECURRENT VARIANTS:
    • p.Arg320His (c.959G>A): commonest recurrent AMT variant; over-represented in East Asian
      (especially Japanese) NKH — semi-founder in this population; compound het with null allele
      → moderate/classic phenotype; homozygous → attenuated; partial residual T-protein function
    • p.Gly47Arg (c.139G>A): European; associated with classic neonatal phenotype; near-null
    • p.Gly199Ser (c.595G>A): reported multiple families; moderate phenotype
    • p.Ile308Asn (c.923T>A): several families; moderate-to-attenuated
    • p.Ser150Arg (c.450T>G): reported European; classic neonatal; null-equivalent function

EPIDEMIOLOGY:
  ~15% of NKH cases; NKH overall ~1:60,000–76,000 (Europe); AMT-NKH ~1:400,000–500,000.
  East Asian (Japanese) over-representation due to p.Arg320His semi-founder allele.
  ~200–300 AMT-NKH cases worldwide 2026. AR biallelic LOF; 3p21.2.

TREATMENT — IDENTICAL TO GLDC-NKH (same disease, different gene):
  1. Sodium Benzoate (Level A): conjugates glycine → hippuric acid; depletes glycine pool
     MANDATORY: L-carnitine co-supplementation (benzoate conjugation depletes carnitine)
  2. Dextromethorphan / DXM (Level B): NMDAr channel antagonist; CYP2D6 prodrug
  3. LEV (Level B): first-line AED; SV2A; no glycine interaction; IV for SE
  4. CLB (Level B): GABA-A PAM; adjunct for myoclonic + tonic
  5. ACTH (Level A): IS management; preferred over VGB (which raises glycine)
  6. KD (Level B): DRE adjunct; reduces serine→glycine flux (SHMT pathway)
  7. Felbamate (Level C): NMDAr glycine-site antagonist; aplastic anaemia risk

HIGH RISK DRUGS (same profile as GLDC-NKH):
  VPA: HIGH RISK — VPA raises glycine via multiple pathways (impairs residual GCS; inhibits
  sarcosine/N-methylglycine pathway; carnitine depletion; POLG1 CI mandatory).
  NOTE: Unlike GLDC-NKH where VPA directly inhibits the deficient P-protein, in AMT-NKH
  VPA raises glycine via SECONDARY pathways (GCS disequilibrium; alternative disposal
  inhibition) — but the CLINICAL RISK IS EQUALLY HIGH. Avoid VPA in all NKH types.
  VGB: HIGH RISK for IS — GABA↑ → possible glycine-glycine transporter upregulation → CSF
  glycine rises; prefer ACTH for IS in NKH. VGB retinal toxicity additional independent CI.
"""

import random
from datetime import datetime

SEED = 20260820
rng = random.Random(SEED)

def _rng_choice(items): return rng.choice(items)
def _rng_int(lo, hi): return rng.randint(lo, hi)
def _rng_float(lo, hi, dec=2): return round(rng.uniform(lo, hi), dec)

# Colour: teal-green — aminomethyltransferase / folate one-carbon / T-protein
COLOUR = "#00695c"  # deep teal — T-protein / THF / AMT

N = 40  # 40-patient cohort

ETIOLOGIES = [
    {"etiology": "Classic Neonatal — homozygous null (truncating/frameshift AMT biallelic)", "pct": 30, "n": 12,
     "csf_plasma_ratio": "0.18–0.50", "outcome": "Severe ID; DRE; non-ambulatory; mechanical ventilation neonatal"},
    {"etiology": "Classic Neonatal — compound het null/missense (e.g. null + p.Gly47Arg or p.Ser150Arg)", "pct": 25, "n": 10,
     "csf_plasma_ratio": "0.12–0.40", "outcome": "Severe-to-moderate neonatal; IS + multifocal; DRE in >75%"},
    {"etiology": "Attenuated — compound het with p.Arg320His (East Asian semi-founder; partial T-protein)", "pct": 15, "n": 6,
     "csf_plasma_ratio": "0.08–0.16", "outcome": "Moderate phenotype; ambulatory possible; seizures manageable"},
    {"etiology": "Attenuated — homozygous p.Arg320His (Japanese cohort — mildest AMT phenotype)", "pct": 10, "n": 4,
     "csf_plasma_ratio": "0.08–0.13", "outcome": "Mildest; some walk/communicate; IQ 35–65; benzoate + DXM sufficient"},
    {"etiology": "Classic Neonatal — consanguineous homozygous missense (Middle East/South Asia)", "pct": 13, "n": 5,
     "csf_plasma_ratio": "0.16–0.45", "outcome": "Severe; neonatal ICU; high early mortality; DRE in survivors"},
    {"etiology": "Transient NKH — neonatal AMT glycine elevation normalises (enzyme immaturity)", "pct": 7, "n": 3,
     "csf_plasma_ratio": "0.09–0.12 (normalising)", "outcome": "Generally benign; AMT gene still pathogenic — monitor closely"},
]

PHENOTYPE_CLASSES = [
    {
        "name": "Classic Neonatal (Severe)", "pct": 65,
        "description": "Onset within hours–2 days of birth. Profound hypotonia, apnea (mechanical ventilation often required), absent Moro/suck, HICCUPS (pathognomonic — phrenic nucleus GlyR). EEG: burst-suppression → hypsarrhythmia. Mortality ~25–30% neonatal. Survivors: severe ID, refractory epilepsy. AMT null/null or null/missense. Biochemically IDENTICAL to GLDC-NKH — gene panel distinguishes.",
        "eeg": "Burst-suppression (neonatal) → hypsarrhythmia → multifocal spikes → electrical SE",
        "seizure_types": "Myoclonic, focal clonic, tonic, electrical SE (EEG-confirmed without motor due to hypotonia)",
        "outcome": "Severe ID in all survivors; DRE >80%; non-ambulatory; non-verbal",
        "csf_ratio": ">0.12 (often 0.18–0.50)"
    },
    {
        "name": "Attenuated (Mild-Moderate)", "pct": 28,
        "description": "Later onset (weeks–years). Milder ID. Seizures in ~50–60% — manageable with benzoate + DXM + LEV. Chorea/choreoathetosis in ~35%. Some ambulation and speech. Enriched in p.Arg320His carriers (East Asian) and other partial-function AMT alleles. CSF:plasma ratio typically 0.08–0.16. Residual T-protein function (10–30%) sufficient to partially maintain GCS flux.",
        "eeg": "Multifocal or generalised spikes; no burst-suppression; may be subtle",
        "seizure_types": "Focal, myoclonic, absence — lower frequency; benzoate + DXM often sufficient",
        "outcome": "Variable; homozygous p.Arg320His (mildest): walks, communicates, IQ 35–65",
        "csf_ratio": "0.08–0.16 (borderline-moderate elevation)"
    },
    {
        "name": "Transient NKH (Rare)", "pct": 7,
        "description": "Neonatal glycine elevation that normalises by ~8 weeks. AMT enzyme immaturity hypothesised in early neonatal life — even with pathogenic biallelic AMT variants, some residual T-protein function transiently compensates. IMPORTANT: gene sequencing still identifies pathogenic AMT variants — not truly 'normal'. Re-test CSF:plasma ratio at 8 weeks minimum. Neurodevelopmental follow-up mandatory.",
        "eeg": "Transient burst-suppression → normalises by weeks 4–8",
        "seizure_types": "Transient neonatal seizures; resolve as glycine normalises",
        "outcome": "Generally benign; subtle developmental delay common; long-term seizure-free possible; AMT pathogenic variants persist",
        "csf_ratio": "Elevated (>0.08) initially → <0.02 by 8 weeks"
    },
]

SEIZURE_TYPES = [
    {"type": "Electrical Status Epilepticus (neonatal SE, EEG-confirmed)", "pct": 58,
     "eeg": "Burst-suppression → continuous electrographic SE; no motor correlate (hypotonia masks)",
     "semiology": "EEG-only SE in context of profound hypotonia; easily missed without continuous video-EEG; treat as SE; burst-suppression background confirms metabolic cause",
     "tips": "LEV IV 60 mg/kg loading; phenobarbital second-line (CAUTION — GlyR additive); IV ketamine third-line for refractory SE (NMDAr antagonist — BENEFICIAL in NKH); simultaneous IV sodium benzoate + glycine lowering"},
    {"type": "Myoclonic seizures (neonatal/infantile)", "pct": 68,
     "eeg": "Poly-spike GSW or poly-spike with burst-suppression background; high-amplitude bursts",
     "semiology": "Sudden jerks in burst-suppression 'bursts'; myoclonic + IS combination; bilateral synchronous",
     "tips": "IS: ACTH Level A; myoclonic: VPA HIGH RISK — LEV + CLB preferred; avoid CBZ/PHT/OXC (myoclonic worsening); DXM reduces burst-suppression-linked myoclonic burden via NMDAr blockade"},
    {"type": "Infantile Spasms (IS / West Syndrome)", "pct": 43,
     "eeg": "Hypsarrhythmia — classical or modified (inter-spasm suppression + multifocal spikes)",
     "semiology": "Spasm clusters; salaam movements; regressive milestones; peak onset 4–8 months",
     "tips": "ACTH Level A; VGB AVOID in NKH (raises glycine — see pharmacology); combined ACTH + sodium benzoate + DXM = NKH-IS triple combination; benzoate optimisation before ACTH improves IS response"},
    {"type": "Focal clonic (cortical, multifocal)", "pct": 33,
     "eeg": "Multifocal spikes/sharp-waves; may reflect diffuse NMDAr cortical excitotoxicity",
     "semiology": "Face/limb clonic; secondary generalisation; multifocal pattern in neonatal → metabolic cause",
     "tips": "Multifocal focal = mandates plasma amino acids + CSF glycine ratio; sodium benzoate lowers glycine → reduces cortical excitability; LEV first-line"},
    {"type": "Tonic seizures (bilateral)", "pct": 38,
     "eeg": "Beta recruitment EMG artifact; bilateral tonic bursts in NREM",
     "semiology": "Rigid posturing; opisthotonic; bilateral; nocturnal predilection; falls risk if ambulatory",
     "tips": "CLB + KD for refractory tonic; avoid PHT/CBZ/OXC; sodium benzoate glycine-lowering reduces tonic seizure frequency in AMT-NKH"},
    {"type": "Generalised Tonic-Clonic (GTCS)", "pct": 28,
     "eeg": "Generalised poly-spike then slow-wave; post-ictal attenuation",
     "semiology": "Tonic-clonic from generalised onset; attenuated AMT-NKH during illness/fever; less common in classic severe",
     "tips": "LEV + CLB; sodium benzoate optimisation; KD if persistent GTCS; avoid sodium channel blockers (CBZ/PHT/OXC)"},
]

TRIGGERS = [
    {"trigger": "Intercurrent illness / fever", "pct": 72,
     "mechanism": "Fever → catabolic state → protein catabolism → glycine release → AMT T-protein cannot transfer aminomethyl → H-protein backed up → GCS stalls → acute glycine surge; SE risk during febrile illness"},
    {"trigger": "Missed sodium benzoate dose", "pct": 63,
     "mechanism": "Sodium benzoate is primary glycine-depleting mechanism; missed dose → glycine reaccumulates 12–24h → plasma glycine rises → CSF glycine follows → NMDAr reactivation → breakthrough seizures"},
    {"trigger": "Protein-rich meal / amino acid load", "pct": 55,
     "mechanism": "Serine → glycine (SHMT interconversion); dietary protein → glycine liberation; AMT-blocked GCS cannot clear → plasma glycine spike → CSF glycine rises within hours; pre-meal benzoate timing important"},
    {"trigger": "VPA / valproate exposure", "pct": 50,
     "mechanism": "VPA raises glycine via multiple secondary pathways: disrupts GCS flux equilibrium (even in AMT-NKH via GCSH cycling perturbation); inhibits sarcosine/N-methylglycine alternative disposal; carnitine depletion reduces benzoate conjugation efficacy; net: glycine worsens"},
    {"trigger": "Sleep deprivation / disrupted sleep", "pct": 45,
     "mechanism": "NMDAr sensitivity heightened by sleep deprivation; NREM burst-suppression risk increased; circadian glycine transport modulation amplifies nighttime CSF glycine burden in NKH"},
    {"trigger": "Fasting / prolonged NPO", "pct": 40,
     "mechanism": "Fasting → muscle protein catabolism → serine/glycine release; gluconeogenesis uses glycine carbon; net glycine liberation exceeds GCS clearance (blocked at T-protein in AMT-NKH); IV dextrose + benzoate coverage mandatory"},
    {"trigger": "Anesthesia / sedation (glycine/NMDAr interactions)", "pct": 28,
     "mechanism": "GlyR-potentiating anesthetics (propofol, barbiturates) add to GlyR brainstem depression (excess glycine already active); ketamine (NMDAr antagonist) may be BENEFICIAL; pre-operative metabolic team consultation mandatory"},
]

TREATMENTS = [
    {
        "drug": "Sodium Benzoate",
        "level": "Level A — First-Line (glycine-lowering backbone; identical to GLDC-NKH)",
        "dose": "250–750 mg/kg/day divided q6-8h; neonates: start 250 mg/kg/day; IV 500 mg/kg loading in SE; oral maintenance",
        "moa": "Benzoate conjugates glycine in mitochondria (GLYAT enzyme) → hippuric acid → renal excretion. Each benzoate removes one glycine. Targets the glycine pool INDEPENDENTLY of which GCS protein is defective (GLDC, AMT, or GCSH).",
        "efficacy": "Reduces plasma glycine 50–80% at therapeutic doses. Target: plasma glycine <500 µmol/L (ideally <300 µmol/L). Mechanism is GCS-independent — equally effective in AMT-NKH as in GLDC-NKH.",
        "monitoring": "Plasma glycine every 3 months; annual CSF:plasma ratio; serum carnitine (benzoate DEPLETES carnitine — supplement 50–100 mg/kg/day MANDATORY); LFT quarterly; NH3 (hyperammonaemia at high doses)",
        "amt_note": "Sodium benzoate mechanism bypasses the blocked GCS entirely — acts via GLYAT-mediated conjugation (NOT GCS-dependent). Therefore, identical efficacy in AMT-NKH compared to GLDC-NKH. Carnitine co-supplementation is EQUALLY MANDATORY."
    },
    {
        "drug": "Dextromethorphan (DXM)",
        "level": "Level B — NMDAr Antagonist (NKH-Specific Adjunct; identical to GLDC-NKH)",
        "dose": "Neonates: 5–10 mg/kg/day div q6-8h. Infants/children: 2–15 mg/kg/day. Maximum 35 mg/kg/day in refractory cases.",
        "moa": "Uncompetitive NMDAr channel blocker at Mg²⁺-binding site. Blocks NMDAr overactivation caused by excess glycine at GluN1 co-agonist site. Does NOT lower glycine — complements benzoate by reducing downstream NMDAr excitotoxicity.",
        "efficacy": "Reduces burst-suppression burden; IS frequency reduction; EEG background improvement in 40–60% of classic AMT-NKH. Synergistic with sodium benzoate.",
        "monitoring": "CYP2D6 phenotyping if poor response (DXM → dextrorphan by CYP2D6; poor metabolizers accumulate DXM); sedation; respiratory depression in neonates; discontinue if no benefit at 4 weeks",
        "amt_note": "DXM mechanism is GCS-independent — blocks NMDAr regardless of which GCS gene is defective. CYP2D6 pharmacogenomics applicable in AMT-NKH identically to GLDC-NKH. NOTE: p.Arg320His carriers (East Asian) may have different CYP2D6 allele frequency vs European GLDC-NKH patients — CYP2D6 phenotyping particularly important."
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — First-Line AED (SE + Maintenance; identical to GLDC-NKH)",
        "dose": "20–60 mg/kg/day div q12h; IV 60 mg/kg loading for SE; oral maintenance",
        "moa": "SV2A modulator; reduces presynaptic neurotransmitter release. No glycine interaction. Mechanistically safe in NKH.",
        "efficacy": "30–50% ≥50% seizure reduction; preferred IV agent for SE; myoclonic + focal reduction",
        "monitoring": "FBC (thrombocytopenia rare); serum creatinine (renal clearance); behaviour assessment (irritability 15–25% in ID patients)",
        "amt_note": "LEV has NO interaction with glycine metabolism or GCS regardless of which protein is defective. First-line IV agent for SE in AMT-NKH — preferred over phenobarbital (PB potentiates GlyR brainstem depression on top of excess glycine)."
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B — Adjunct (Myoclonic + Tonic + Focal; identical to GLDC-NKH)",
        "dose": "0.1–0.5 mg/kg/day div q12-24h; maximum 40 mg/day adult; nocturnal dosing for tonic",
        "moa": "GABA-A PAM (α2/α3-selective benzodiazepine). Enhances Cl⁻ conductance. No glycine catabolism interaction.",
        "efficacy": "40–55% ≥50% reduction (myoclonic + tonic); tolerance develops 12–18 months; drug holiday protocol",
        "monitoring": "Sedation/cognition monthly; tolerance at 6 months; slow taper 10%/week",
        "amt_note": "CLB GABA-A mechanism orthogonal to GlyR — does not affect disease pathophysiology but reduces seizure frequency. Safe adjunct in AMT-NKH."
    },
    {
        "drug": "ACTH (Corticotropin) — IS Management",
        "level": "Level A — Infantile Spasms in AMT-NKH (identical preference to GLDC-NKH)",
        "dose": "High-dose ACTH 150 IU/m²/day IM or synthetic tetracosactide 0.5–0.75 mg/day x4 weeks then taper",
        "moa": "Reduces neuroinflammatory cascade in developing brain; reduces hypsarrhythmia. Evidence-based for IS regardless of etiology.",
        "efficacy": "60–70% spasm cessation; NKH-IS: combined ACTH + benzoate + DXM — best outcomes. ACTH preferred over VGB (VGB raises glycine — avoid in all NKH).",
        "monitoring": "BP (hypertension); glucose (hyperglycaemia); electrolytes (hypokalaemia); infection risk; cushingoid features",
        "amt_note": "ACTH preferred over VGB for IS in AMT-NKH for same reason as GLDC-NKH: VGB raises glycine via GABA-T inhibition → GABA-glycine co-transporter upregulation. ACTH + benzoate + DXM triple combination = AMT-NKH IS standard of care."
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — DRE Adjunct (identical to GLDC-NKH; modest glycine-lowering)",
        "dose": "4:1 ratio or MAD (20 g carb/day); minimum 3-month trial; ketosis BHB 2–4 mmol/L",
        "moa": "Reduces serine→glycine flux (SHMT, which requires 5,10-methyleneTHF from GCS — KD may alter one-carbon flux); protein-restricted component limits glycine substrate; BHB may modulate NMDAr",
        "efficacy": "30–40% seizure reduction in AMT-NKH DRE; modest plasma glycine lowering (~10–15%); case series only",
        "monitoring": "BHB monthly; lipids 3-monthly; growth; renal stone screen; glycine TDM",
        "amt_note": "KD in AMT-NKH: SHMT produces serine from 5,10-methyleneTHF; AMT LOF reduces 5,10-methyleneTHF production from glycine → SHMT may actually run LESS efficiently → KD may have slightly MORE glycine-lowering in AMT-NKH than GLDC-NKH (theoretical; clinical data sparse). Use as DRE adjunct."
    },
    {
        "drug": "Felbamate (FBM) — NMDAr Adjunct",
        "level": "Level C — Experimental NMDAr Adjunct (Refractory NKH only)",
        "dose": "15–45 mg/kg/day div q6-8h; very slow titration (aplastic anaemia risk mandates slow initiation)",
        "moa": "NMDAr glycine-site antagonist (strychnine-insensitive) + NMDAr channel blocker. DUAL NMDAr mechanism: glycine-site block (theoretically superior to DXM alone in NKH where glycine site is saturated).",
        "efficacy": "Limited AMT-NKH data; case reports of burst-suppression reduction; use only after DXM failure",
        "monitoring": "FBC weekly x6 months then monthly (aplastic anaemia — Black Box Warning); LFT (hepatotoxicity); drug interactions (CYP450 induction; PHT/CBZ reduce FBM levels)",
        "amt_note": "FBM glycine-site NMDAr antagonism is mechanistically attractive in AMT-NKH but restricted by toxicity. Only under specialist metabolic epilepsy unit after failing DXM + optimised benzoate."
    },
    {
        "drug": "Valproic Acid (VPA) — HIGH RISK",
        "level": "HIGH RISK — Metabolic Worsening + Standard CIs",
        "dose": "AVOID — glycine worsening regardless of GCS gene affected",
        "moa": "VPA raises glycine via multiple secondary pathways in AMT-NKH: (1) GCS flux disequilibrium — GCSH H-protein cycling perturbation worsens; (2) N-methylglycine (sarcosine) alternative disposal pathway inhibited; (3) carnitine depletion → reduces benzoate conjugation efficiency; net: plasma glycine rises, CSF glycine rises",
        "efficacy": "May suppress some seizure types transiently but worsens underlying glycine accumulation — paradoxical worsening or encephalopathy risk; NOT recommended in any NKH genotype",
        "monitoring": "NOT RECOMMENDED. If used in extremis: LFT weekly; NH3 weekly; plasma glycine monthly; POLG1 exclusion MANDATORY",
        "amt_note": "VPA HIGH RISK IN AMT-NKH: Unlike GLDC-NKH where VPA directly inhibits the P-protein (GLDC enzyme), in AMT-NKH the P-protein (GLDC) is intact — however VPA still raises glycine via secondary mechanisms (GCS equilibrium disruption, sarcosine pathway inhibition, carnitine depletion). The CLINICAL RISK IS EQUIVALENT. Avoid VPA in all NKH genotypes regardless of which GCS gene is defective."
    },
    {
        "drug": "VGB (Vigabatrin) — HIGH RISK for IS",
        "level": "HIGH RISK — VGB raises CSF glycine in NKH; prefer ACTH for IS",
        "dose": "AVOID for IS in AMT-NKH — use ACTH",
        "moa": "VGB inhibits GABA-T → GABA↑. GABA and glycine share inhibitory co-transport (GlyT2, SLC6A5). Elevated GABA may upregulate glycine reuptake via co-transporter competition → extracellular glycine rises.",
        "efficacy": "VGB effective for IS in general but disease-specific glycine risk in AMT-NKH as in GLDC-NKH. ACTH equally or more effective for IS.",
        "monitoring": "RETINAL TOXICITY mandatory (visual field + OCT or ERG pre-verbal every 3 months); if NKH: use ACTH first",
        "amt_note": "VGB HIGH RISK in AMT-NKH for IS: mechanism is the same GABA-glycine co-transporter concern as GLDC-NKH. VGB retinal toxicity is ALSO a separate concern. DOUBLE RISK regardless of which GCS gene is defective."
    },
]

CONTRAINDICATIONS = [
    {"drug": "VPA (Valproic Acid)", "level": "HIGH RISK — METABOLIC WORSENING",
     "reason": "VPA raises glycine in AMT-NKH via secondary pathways (GCS disequilibrium, sarcosine pathway inhibition, carnitine depletion reducing benzoate efficacy). Unlike GLDC-NKH where VPA directly inhibits P-protein, in AMT-NKH it operates via secondary mechanisms — but the glycine-worsening effect is clinically equivalent. POLG1 MANDATORY before any VPA use.",
     "alternative": "LEV + CLB + DXM + sodium benzoate. IV LEV for SE. IV ketamine (NMDAr antagonist — beneficial mechanism) for refractory SE."},
    {"drug": "VGB (Vigabatrin) for IS", "level": "HIGH RISK — PREFER ACTH",
     "reason": "VGB may raise CSF glycine via GABA-glycine co-transporter upregulation (GABA↑ from GABA-T inhibition → extracellular glycine rises). VGB retinal toxicity is additional independent risk. Both risks apply regardless of GLDC vs AMT vs GCSH genotype.",
     "alternative": "ACTH (Level A for IS in all NKH genotypes); combined ACTH + benzoate + DXM for NKH-IS."},
    {"drug": "CBZ / OXC / PHT (Na-channel blockers)", "level": "RELATIVE CI — Myoclonic Worsening",
     "reason": "Sodium channel blockers worsen myoclonic seizures in generalised epilepsies. AMT-NKH has major myoclonic component (neonatal + IS period). CBZ/PHT documented to worsen generalised myoclonus. Avoid if any myoclonic feature present.",
     "alternative": "LEV, CLB, VPA (avoided in NKH), KD for refractory myoclonus."},
    {"drug": "Phenobarbital (PB) — Neonatal/Maintenance", "level": "CAUTION — GlyR Additive + Respiratory",
     "reason": "PB potentiates GABA-A and GlyR. In neonatal AMT-NKH: excess glycine already activates GlyR brainstem inhibition → apnea/hypotonia. PB ADDS to GlyR-mediated brainstem depression → excessive respiratory depression. Use ONLY if LEV failed as second-line for SE.",
     "alternative": "IV LEV 60 mg/kg (primary SE agent). IV ketamine (NMDAr antagonist — BENEFICIAL in NKH) as third-line."},
    {"drug": "Fasting / NPO without coverage", "level": "CAUTION — Glycine Surge",
     "reason": "Fasting → muscle catabolism → glycine release → glycine surge without GCS clearance (T-protein blocked in AMT-NKH). IV dextrose + sodium benzoate MANDATORY for any NPO period.",
     "alternative": "IV 10% dextrose + oral/IV sodium benzoate; target plasma glycine <500 µmol/L perioperatively."},
    {"drug": "Protein Restriction (extreme)", "level": "CAUTION — Nutritional",
     "reason": "Extreme protein restriction below RDA impairs growth/neurodevelopment; sodium benzoate is far more effective for glycine reduction. Protein restriction alone is inadequate and harmful if excessive.",
     "alternative": "Sodium benzoate (metabolic clearance); maintain RDA protein; carnitine co-supplementation."},
]

MONITORING = [
    {"parameter": "AMT gene sequencing (WES/targeted GCS panel: AMT + GLDC + GCSH + DLD)", "frequency": "At diagnosis (confirmatory)", "target": "Biallelic pathogenic AMT variants; ACMG classification; AMT enzyme assay if VUS; biochemically cannot distinguish AMT from GLDC or GCSH — gene panel mandatory"},
    {"parameter": "CSF:Plasma glycine ratio (simultaneous)", "frequency": "At diagnosis; annually on treatment", "target": "Diagnostic: ratio ≥0.08; treatment target: ratio <0.04; plasma glycine <500 µmol/L (ideally <300 µmol/L); identical target to GLDC-NKH"},
    {"parameter": "Plasma amino acid quantitative (glycine focus)", "frequency": "Every 3 months (sodium benzoate titration)", "target": "Glycine <500 µmol/L; serine (SHMT substrate — monitors one-carbon flux); folate metabolites (5,10-methyleneTHF production from AMT step — monitor plasma folate/homocysteine)"},
    {"parameter": "Urine organic acids (glycine, hippurate)", "frequency": "Every 3 months on sodium benzoate", "target": "Hippuric acid excretion confirms benzoate conjugation; exclude 'ketotic hyperglycinemia' (propionic/MMA — organic acids differentiate)"},
    {"parameter": "Serum carnitine (free + acylcarnitine)", "frequency": "Every 3 months", "target": "Free carnitine >20 µmol/L; benzoate conjugation depletes carnitine; supplement L-carnitine 50–100 mg/kg/day; check acylcarnitine profile (C3 elevation → propionic acidemia differential)"},
    {"parameter": "EEG (video-EEG — continuous neonatal)", "frequency": "Continuous neonatal phase; annual + urgent for clinical change", "target": "Burst-suppression burden; EEG-only IS detection (AMT-NKH: IS without motor manifestation common due to hypotonia); hypsarrhythmia on ACTH resolution; SE detection"},
    {"parameter": "MRI brain (3T + DWI in acute)", "frequency": "At diagnosis; 6 months; annually first 3 years", "target": "Periventricular WM hypomyelination; thin/absent corpus callosum (especially splenium); DWI restriction acute AMT-NKH; cerebellar hypoplasia severe; identical pattern to GLDC-NKH"},
    {"parameter": "Plasma folate + homocysteine + methionine", "frequency": "Every 6 months", "target": "AMT LOF → 5,10-methyleneTHF NOT produced from glycine → folate one-carbon cycle perturbation; low plasma folate + elevated homocysteine indicates one-carbon deficit; folate supplementation as needed"},
    {"parameter": "POLG1 WES exclusion (mandatory pre-VPA)", "frequency": "MANDATORY before VPA", "target": "Biallelic POLG1 → VPA ABSOLUTE CI (Alpers); AMT-NKH metabolic vulnerability adds to POLG1 standard CI"},
    {"parameter": "Liver function + NH3 (sodium benzoate monitoring)", "frequency": "Monthly first 6 months; quarterly stable", "target": "ALT/AST <3× ULN; NH3 <80 µmol/L (high-dose benzoate → urea cycle perturbation); GGT, ALP"},
    {"parameter": "Developmental / cognitive assessment (Bayley/Vineland)", "frequency": "Every 6 months first 3 years; annual thereafter", "target": "Motor milestones; language; adaptive behaviour; Vineland-3 composite; track impact of glycine control on trajectory"},
    {"parameter": "CYP2D6 pharmacogenomics (DXM metabolism)", "frequency": "Once at baseline before DXM", "target": "CYP2D6 poor metabolizer → DXM accumulates; ultra-rapid → reduced effect. NOTE: p.Arg320His allele enriched in East Asian (Japanese) — CYP2D6 poor metabolizer frequency differs in this population (PM ~5% vs European ~8%); still clinically relevant"},
    {"parameter": "Growth anthropometry + nutrition", "frequency": "Every 3 months first 2 years; 6-monthly thereafter", "target": "Weight/height z-score; head circumference; folate levels; serine levels; carnitine; folate supplementation often needed in AMT-NKH (5,10-methyleneTHF deficit)"},
    {"parameter": "SUDEP risk assessment", "frequency": "Annual in DRE patients", "target": "Neonatal SE + IS + DRE = highest SUDEP risk; nocturnal alarm; prone position avoidance; side-car sleeping; sodium benzoate adherence monitoring"},
]

LIFECYCLE = [
    {"stage": "Prenatal / Sibling Screening", "age": "Prenatal", "description": "Known AMT family: CVS or amniocentesis for AMT genotyping. AMT enzyme assay on chorionic villi (technically demanding; gene sequencing preferred 2026). Standard NBS (tandem MS): does NOT detect NKH (glycine not on standard NBS panels). Prenatal genetic diagnosis essential for known AMT biallelic carrier parents. p.Arg320His East Asian families: cascade testing recommended."},
    {"stage": "Neonatal Crisis (Classic AMT-NKH)", "age": "0–7 days", "description": "NICU admission hours after birth. Apnea (MV required), profound hypotonia, absent Moro/suck, HICCUPS (pathognomonic — phrenic GlyR). CSF:plasma glycine ratio STAT (simultaneous). EEG: burst-suppression immediately. START: IV sodium benzoate + IV LEV + DXM oral/NG. POLG1 exclusion initiated. AMT-NKH biochemically identical to GLDC-NKH at this stage — gene panel sent simultaneously."},
    {"stage": "Post-acute Neonatal / Gene Identification", "age": "2–8 weeks", "description": "Wean ventilator (if survived). Gene panel result → confirms AMT (not GLDC). Burst-suppression persists → transitions to multifocal/hypsarrhythmia. Oral benzoate titration. DXM optimisation. MRI: WM hypomyelination, thin CC. CYP2D6 phenotyping for DXM. p.Arg320His carriers identified → counselling re: East Asian AMT-NKH."},
    {"stage": "Infantile Phase (IS / Epilepsy)", "age": "3–18 months", "description": "IS onset in survivors (40–50%): hypsarrhythmia + spasms. ACTH x4 weeks (NOT VGB). AMT-NKH IS: ACTH + benzoate + DXM triple combination. Developmental regression. Feeding difficulties. PT, OT, speech therapy. Seizure types evolve: myoclonic + tonic. Folate supplementation initiated if homocysteine elevated."},
    {"stage": "Early Childhood (DRE Management)", "age": "18 months–6 years", "description": "DRE management (classic AMT-NKH); attenuated AMT-NKH: epilepsy manageable; KD initiation for DRE; school/rehabilitation planning; cognitive plateau vs slow gains; home seizure care plan; annual MRI; plasma folate/homocysteine monitoring."},
    {"stage": "School Age / Adolescence", "age": "6–18 years", "description": "Classic AMT-NKH: specialized care. Attenuated: mainstream school with support (p.Arg320His East Asian cohort: better outcomes — some ambulatory + communicative). Mood disorders (depression/anxiety 30–40%); puberty-related seizure change; VPA CI continues; benzoate compliance monitoring; AMT variant re-interpretation with updated databases."},
    {"stage": "Adulthood (Chronic Management)", "age": "18+ years", "description": "Classic AMT-NKH: specialized residential care. Attenuated (p.Arg320His): supported independent living possible for mildest. Renal monitoring (long-term benzoate → hippurate → renal load). Genetic counselling (25% sibling recurrence AR). Clinical trial enrolment — GCS enzyme replacement / mRNA therapy / AAV9-AMT gene therapy in pipeline 2026."},
]

CONCEPTS = [
    {"term": "AMT / T-protein / 3p21.2", "definition": "Aminomethyltransferase; T-protein of GCS; 403 aa, ~45 kDa; mitochondrial matrix; THF-binding enzyme; accepts aminomethyl group from loaded H-protein (GCSH); transfers to THF → 5,10-methyleneTHF + NH₄⁺; simultaneously regenerates oxidised H-protein to cycle GCS. AR biallelic LOF → NKH. OMIM *238310. ~15% of all NKH."},
    {"term": "H-protein (GCSH) Back-up — AMT LOF Bottleneck", "definition": "When AMT is absent, aminomethyl-loaded H-protein (reduced GCSH) CANNOT discharge its aminomethyl group to THF. H-protein accumulates in loaded state → no free oxidised H-protein available → P-protein (GLDC) cannot transfer its aminomethyl intermediate to H-protein → GCS stalls at STEP 1. Upstream blockade of entire GCS. Net result: IDENTICAL glycine accumulation to GLDC-NKH despite P-protein being structurally intact in AMT-NKH."},
    {"term": "5,10-MethyleneTHF Deficit — AMT-NKH Folate Impact", "definition": "AMT specifically catalyses the step that produces 5,10-methyleneTHF from the aminomethyl group and THF. AMT LOF → this THF-dependent step fails → 5,10-methyleneTHF is NOT produced from glycine. 5,10-methyleneTHF is required for: (1) thymidylate synthesis (dTMP); (2) serine synthesis (via SHMT reverse); (3) methionine cycle (via MTHFR). Folate supplementation and monitoring (plasma folate, homocysteine, serine) is clinically important in AMT-NKH — arguably more so than GLDC-NKH since AMT is the direct 5,10-methyleneTHF-generating step."},
    {"term": "p.Arg320His — East Asian AMT Semi-Founder", "definition": "c.959G>A; p.Arg320His: the most common recurrent AMT variant globally; over-represented in East Asian (especially Japanese) NKH. Arginine 320 is in the THF-binding domain — p.Arg320His reduces aminomethyl transfer efficiency (partial T-protein function ~15–25% residual). Homozygous p.Arg320His → attenuated AMT-NKH (mildest; ambulatory; IQ 35–65). Compound het with null allele → moderate-to-classic phenotype. NOT a pan-ethnic founder — European/Middle Eastern AMT-NKH more often private variants."},
    {"term": "GCS Biochemical Identity — AMT vs GLDC vs GCSH", "definition": "NKH caused by GLDC (75–80%), AMT (15%), or GCSH (1%) is BIOCHEMICALLY IDENTICAL: same CSF:plasma glycine ratio elevation, same plasma glycine range, same CSF glycine range. Gene panel (AMT + GLDC + GCSH) is MANDATORY to identify the causative gene — biochemistry alone cannot distinguish. Treatment is also identical. 2026 context: WES is first-line diagnostic in metabolic epilepsy — gene identification is rapid."},
    {"term": "CSF:Plasma Glycine Ratio — DIAGNOSTIC THRESHOLD", "definition": "SIMULTANEOUS collection mandatory (fasting; LP sedation with ketamine preferred). Normal: <0.02. AMT-NKH: ≥0.08 (classic: 0.15–0.50; attenuated: 0.08–0.16). Plasma glycine alone insufficient (elevated in propionic/MMA/IVA — 'ketotic hyperglycinemia'). Ratio ≥0.08 is highly specific for NKH (all three types: GLDC/AMT/GCSH)."},
    {"term": "HICCUPS — Pathognomonic NKH Clue (All Types)", "definition": "Phrenic nerve nucleus (C3–C5 anterior horn) uses glycinergic inhibitory interneurons. Excess glycine → GlyR over-activation at phrenic nucleus → rhythmic uncontrolled diaphragm contractions = HICCUPS. Applies to AMT-NKH identically to GLDC-NKH. Persistent neonatal hiccups + hypotonia + apnea = NKH (any GCS gene) until proven otherwise."},
    {"term": "VPA HIGH RISK — AMT-NKH Secondary Pathway Mechanism", "definition": "Unlike GLDC-NKH (where VPA directly inhibits P-protein/GLDC), in AMT-NKH the P-protein is intact but VPA still raises glycine via: (1) GCS equilibrium disruption (GCSH cycling backed up when AMT absent — VPA adds metabolic stress to already strained GCS); (2) Sarcosine/N-methylglycine pathway inhibition (alternative glycine disposal reduced); (3) Carnitine depletion → reduces sodium benzoate conjugation efficiency. Clinical glycine worsening is EQUIVALENT across NKH types on VPA. AVOID VPA in all NKH (GLDC/AMT/GCSH)."},
    {"term": "Sodium Benzoate — GCS-Independent Glycine Removal", "definition": "Sodium benzoate works via GLYAT (glycine N-acyltransferase) in mitochondria — this enzyme is STRUCTURALLY INDEPENDENT of GCS (different mitochondrial enzyme). Benzoate conjugates glycine → hippuric acid → renal excretion. This pathway functions NORMALLY in AMT-NKH (AMT has no role in benzoate conjugation). Therefore, sodium benzoate has IDENTICAL efficacy in AMT-NKH vs GLDC-NKH. Carnitine depletion via CoA pathway is the same concern."},
    {"term": "Burst-Suppression EEG — AMT-NKH Neonatal Signature", "definition": "Burst-suppression within hours-days of birth is hallmark neonatal NKH EEG regardless of GCS gene. In AMT-NKH: identical EEG pattern to GLDC-NKH. NMDAr excitotoxicity driven by glycine accumulation (AMT LOF → same glycine burden as GLDC LOF). Continuous video-EEG mandatory in neonatal period — electrical SE without motor manifestation (hypotonia masks clinical SE)."},
    {"term": "CYP2D6 and DXM in p.Arg320His East Asian Cohort", "definition": "DXM is a CYP2D6 prodrug → dextrorphan (active). In East Asian (especially Japanese) patients enriched for p.Arg320His: CYP2D6 poor metabolizer (PM) frequency ~5% vs European ~8% (slightly lower PM frequency but still clinically significant). CYP2D6 intermediate metabolizer (IM) frequency is higher in East Asian populations. CYP2D6 phenotyping particularly valuable in p.Arg320His AMT-NKH patients where DXM response may differ."},
    {"term": "AMT Gene Therapy / GCS Replacement — 2026 Pipeline", "definition": "AAV9-AMT gene therapy: preclinical stage 2026; AAV9 targets CNS and liver; AMT transgene delivery aims to restore T-protein function in hepatocytes (reduces peripheral glycine) and neurons (reduces CNS glycine). mRNA therapy (lipid nanoparticle-AMT-mRNA): hepatocyte-targeted; periodic dosing avoids genomic integration. GCS enzyme replacement not feasible (multi-protein complex; mitochondrial; no M6P route). Clinical trial anticipated 2027–2028."},
]

THRESHOLDS = [
    {"parameter": "CSF:Plasma glycine ratio (normal)", "value": "<0.02", "clinical": "Normal GCS function; no NKH regardless of gene"},
    {"parameter": "CSF:Plasma glycine ratio (AMT-NKH diagnostic)", "value": "≥0.08", "clinical": "Highly specific for NKH (all types GLDC/AMT/GCSH); classic AMT: 0.15–0.50"},
    {"parameter": "Plasma glycine (normal)", "value": "150–260 µmol/L", "clinical": "Normal (lab-dependent); AMT-NKH: same range as GLDC-NKH"},
    {"parameter": "Plasma glycine (AMT-NKH untreated)", "value": "600–2800 µmol/L", "clinical": "Similar range to GLDC-NKH; severity correlates with ratio not absolute level"},
    {"parameter": "Target plasma glycine on sodium benzoate", "value": "<500 µmol/L (ideally <300 µmol/L)", "clinical": "Same glycine-lowering target as GLDC-NKH; TDM every 3 months"},
    {"parameter": "CSF glycine (AMT-NKH)", "value": ">100 µmol/L (often 150–1800 µmol/L)", "clinical": "Identical range to GLDC-NKH; confirms NKH pathophysiology; no biochemical distinction possible"},
    {"parameter": "Serum carnitine (target on benzoate)", "value": ">20 µmol/L free carnitine", "clinical": "Supplement L-carnitine 50–100 mg/kg/day; benzoate depletes carnitine identically in AMT-NKH"},
    {"parameter": "Plasma homocysteine (folate monitoring)", "value": "<10 µmol/L", "clinical": "Elevated homocysteine → 5,10-methyleneTHF deficit (AMT LOF directly impairs this step); folate supplementation if elevated"},
    {"parameter": "Plasma folate", "value": ">6 ng/mL", "clinical": "AMT LOF → 5,10-methyleneTHF not produced from glycine → folate cycle perturbation; supplement if low"},
    {"parameter": "NH3 on sodium benzoate (threshold)", "value": "<80 µmol/L", "clinical": "Hyperammonaemia risk at high benzoate doses; >80 → dose reduction; monitor in AMT-NKH as in GLDC-NKH"},
    {"parameter": "Residual AMT enzyme activity (p.Arg320His homozygous)", "value": "~15–25% residual", "clinical": "Attenuated phenotype correlates with residual T-protein activity; homozygous p.Arg320His ~15–25% → mildest AMT-NKH"},
]


def _patient_profile(i):
    pat_id = f"AMT-{i+1:03d}"
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
    has_is = phenotype_class == "Classic Neonatal" and rng.random() < 0.43
    has_bs = phenotype_class == "Classic Neonatal" and rng.random() < 0.58
    dre = has_epilepsy and (phenotype_class == "Classic Neonatal" and rng.random() < 0.78 or
                            phenotype_class == "Attenuated" and rng.random() < 0.22)
    on_benzoate = rng.random() < 0.92
    on_dxm = rng.random() < 0.76
    on_lev = rng.random() < 0.70
    on_clb = rng.random() < 0.42
    on_acth = has_is and rng.random() < 0.82
    on_kd = dre and rng.random() < 0.28
    # p.Arg320His enriched in attenuated
    is_arg320his = phenotype_class == "Attenuated" and rng.random() < 0.55
    plasma_gly = _rng_float(180, 580, 0) if on_benzoate else _rng_float(600, 2400, 0)
    csf_plasma_ratio = (_rng_float(0.12, 0.50, 3) if phenotype_class == "Classic Neonatal"
                        else _rng_float(0.08, 0.16, 3) if phenotype_class == "Attenuated"
                        else _rng_float(0.08, 0.13, 3))
    carnitine_ok = rng.random() < 0.73
    folate_low = rng.random() < 0.35  # AMT-NKH has meaningful folate deficit
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
        "is_arg320his_carrier": is_arg320his,
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
    n_arg320his = sum(1 for p in PATIENTS if p["is_arg320his_carrier"])
    n_folate_low = sum(1 for p in PATIENTS if p["folate_low"])
    avg_plasma_gly = round(sum(p["plasma_glycine_umol"] for p in PATIENTS) / N, 1)
    avg_ratio = round(sum(p["csf_plasma_ratio"] for p in PATIENTS) / N, 3)
    return {
        "dashboard": "AMT Epilepsy — Non-Ketotic Hyperglycinemia (NKH) / T-protein (Aminomethyltransferase) Deficiency",
        "gene": "AMT (3p21.2) — Aminomethyltransferase; T-protein of GCS; 403 aa ~45 kDa; mitochondrial matrix; THF-dependent; accepts aminomethyl from H-protein (GCSH); ~15% of NKH",
        "inheritance": "Autosomal Recessive (AR) biallelic LOF; ~15% of NKH; NKH overall ~1:60,000–76,000; AMT-NKH ~1:400,000–500,000; ~200–300 cases worldwide 2026",
        "omim_gene": "238310",
        "omim_disease": "605899",
        "locus": "3p21.2",
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
        "n_arg320his_carriers": n_arg320his,
        "arg320his_pct": round(100 * n_arg320his / N),
        "n_folate_low": n_folate_low,
        "folate_low_pct": round(100 * n_folate_low / N),
        "avg_plasma_glycine": avg_plasma_gly,
        "avg_csf_plasma_ratio": avg_ratio,
        "phenotype_classes": PHENOTYPE_CLASSES,
        "etiologies": ETIOLOGIES,
        "key_concepts": [c["term"] for c in CONCEPTS[:8]],
        "standards": [
            "Kure S et al. Cloning and expression of cDNA encoding human T-protein. J Biol Chem 1991.",
            "Nanao K et al. Identification of the mutations in the T-protein of the glycine cleavage system. J Inherit Metab Dis 1994.",
            "Hamosh A & Johnston MV. Nonketotic hyperglycinemia. OMIM #605899. 2024.",
            "Van Hove JLK et al. NKH long-term outcome and management. J Inherit Metab Dis 2006.",
            "Toone JR et al. Molecular characteristics of non-ketotic hyperglycinemia. Mol Genet Metab 2003.",
            "Tada K et al. Non-ketotic hyperglycinemia: clinical variants. Brain Dev 1980.",
            "García-Cazorla A et al. NKH clinical spectrum review. Orphanet J Rare Dis 2022.",
            "NKH International Family Network — Clinical care guidelines 2023.",
            "ACMG/AMP — Variant interpretation standards for NKH/AMT 2024.",
        ],
        "per_patient_kpis": sorted(PATIENTS, key=lambda p: p["csf_plasma_ratio"], reverse=True),
    }


def get_breakdown():
    classic = sum(1 for p in PATIENTS if p["phenotype_class"] == "Classic Neonatal")
    attenuated = sum(1 for p in PATIENTS if p["phenotype_class"] == "Attenuated")
    transient = sum(1 for p in PATIENTS if p["phenotype_class"] == "Transient")

    ratio_ranges = [(0.0, 0.08, "<0.08 (not NKH / transient resolving)"),
                    (0.08, 0.16, "0.08–0.16 (attenuated AMT)"),
                    (0.16, 0.30, "0.16–0.30 (moderate classic)"),
                    (0.30, 1.0, ">0.30 (severe classic)")]
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
        "title": "AMT / Non-Ketotic Hyperglycinemia — Definitions, GCS Pathway, T-Protein Bottleneck, Pharmacology",
        "gene_card": {
            "gene": "AMT",
            "locus": "3p21.2",
            "protein": "Aminomethyltransferase (T-protein)",
            "size": "403 aa, ~45 kDa",
            "family": "Aminomethyltransferases; glycine cleavage system T-protein; THF-binding; PLP-independent",
            "structure": "Mitochondrial matrix; accepts aminomethyl group from loaded H-protein (aminomethyl-lipoamide-GCSH); transfers to THF → 5,10-methyleneTHF + NH₄⁺; regenerates oxidised H-protein to cycle GCS",
            "cofactor": "Tetrahydrofolate (THF) — substrate, not prosthetic group; pyridoxal phosphate NOT required (contrast with P-protein GLDC)",
            "localisation": "Mitochondrial matrix (N-terminal MTS signal)",
            "omim_gene": "*238310",
            "omim_disease": "#605899",
            "inheritance": "AR biallelic LOF",
            "cause_of_nkh": "~15% of NKH; GLDC ~75–80%; GCSH ~1%; DLD very rare",
        },
        "pathway": {
            "name": "Glycine Cleavage System (GCS) — 4-Protein Complex; T-Protein (AMT) Is Step 3",
            "steps": [
                {"step": 1, "enzyme": "P-protein (GLDC)", "gene": "GLDC", "cofactor": "PLP",
                 "reaction": "Glycine + H-protein(oxidised) → CO₂ + aminomethyl-H-protein",
                 "clinical": "GLDC LOF (75–80% NKH): P-protein absent → glycine cannot enter GCS → accumulates. IN AMT-NKH: P-protein structurally INTACT but cannot function (H-protein backed up as loaded)."},
                {"step": 2, "enzyme": "H-protein (GCSH)", "gene": "GCSH", "cofactor": "Lipoic acid",
                 "reaction": "Aminomethyl-H-protein shuttles aminomethyl group to T-protein (AMT)",
                 "clinical": "GCSH LOF (1% NKH): H-protein cannot carry aminomethyl group. IN AMT-NKH: H-protein structurally intact but REMAINS LOADED — cannot unload to absent AMT."},
                {"step": 3, "enzyme": "T-protein (AMT) — THE DEFECTIVE STEP IN AMT-NKH", "gene": "AMT", "cofactor": "THF",
                 "reaction": "Aminomethyl-H-protein + THF → 5,10-methyleneTHF + NH₄⁺ + H-protein(oxidised)",
                 "clinical": "AMT LOF (15% NKH): T-protein absent → aminomethyl group CANNOT transfer to THF → H-protein remains loaded → P-protein cannot act (no free H-protein) → ENTIRE GCS blocked. 5,10-methyleneTHF NOT produced."},
                {"step": 4, "enzyme": "L-protein (DLD)", "gene": "DLD", "cofactor": "NAD+/FAD",
                 "reaction": "H-protein(reduced) + NAD⁺ → H-protein(oxidised) + NADH",
                 "clinical": "DLD is shared with pyruvate DH + alpha-KG DH. DLD LOF → combined dehydrogenase defect + glycine accumulation (distinct from classic NKH). IN AMT-NKH: DLD cannot act on H-protein that never reaches reduced state."},
            ],
            "net_reaction": "Glycine + THF + NAD⁺ → 5,10-methyleneTHF + CO₂ + NH₄⁺ + NADH",
            "amt_lof_consequence": "AMT LOF blocks step 3 → H-protein backed up (loaded) → step 1 (GLDC) and step 2 stall → GCS completely inoperable → glycine accumulates identically to GLDC-NKH + 5,10-methyleneTHF not produced",
        },
        "biomarkers": [
            {"marker": "CSF:Plasma glycine ratio (simultaneous)", "method": "Quantitative plasma + CSF amino acids (LC-MS/MS)",
             "reference_range": "<0.02", "nkh_range": "≥0.08 (classic AMT: 0.15–0.50; attenuated: 0.08–0.16)",
             "notes": "PRIMARY DIAGNOSTIC TEST for NKH. AMT-NKH is biochemically IDENTICAL to GLDC-NKH at this step. Gene panel required to identify AMT vs GLDC vs GCSH. Simultaneous collection mandatory."},
            {"marker": "Plasma glycine (quantitative)", "method": "Quantitative amino acid panel (LC-MS/MS)",
             "reference_range": "150–260 µmol/L", "nkh_range": "600–2800+ µmol/L (untreated AMT-NKH)",
             "notes": "Same range as GLDC-NKH. Not specific — elevated in propionic/MMA/IVA (ketotic hyperglycinemia). Ratio is diagnostic; plasma level is monitoring. Target on benzoate: <500 µmol/L."},
            {"marker": "Plasma folate + homocysteine + serine", "method": "Plasma metabolomics / amino acids + folate panel",
             "reference_range": "Folate >6 ng/mL; homocysteine <10 µmol/L; serine 65–150 µmol/L",
             "nkh_range": "AMT-NKH: folate may be low; homocysteine may be elevated (5,10-methyleneTHF deficit)",
             "notes": "AMT LOF directly impairs 5,10-methyleneTHF production from glycine. Monitor folate cycle. Folate supplementation if homocysteine elevated. This monitoring is SPECIFIC to AMT-NKH (not routinely needed in GLDC-NKH to the same degree)."},
            {"marker": "AMT enzyme activity (lymphocytes/liver)", "method": "Aminomethyltransferase activity assay (C1-unit transfer)",
             "reference_range": "Lab-dependent", "nkh_range": "<5% residual (null); ~15–25% (p.Arg320His homozygous)",
             "notes": "AMT enzyme assay is technically demanding and not widely available. WES/gene sequencing preferred in 2026. Enzyme assay useful for VUS classification or atypical presentations."},
        ],
        "key_concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "treatments": TREATMENTS,
        "references": [
            "Kure S et al. Cloning and expression of a cDNA encoding human T-protein of the glycine cleavage system. J Biol Chem 1991;266:11257.",
            "Nanao K et al. Identification of the mutations in the T-protein gene causing non-ketotic hyperglycinemia. J Inherit Metab Dis 1994;17:70.",
            "Hamosh A & Johnston MV. Nonketotic hyperglycinemia. OMIM #605899. 2024.",
            "Van Hove JLK et al. Long-term outcome and management of NKH. J Inherit Metab Dis 2006;29:531.",
            "Toone JR et al. Molecular genetics of NKH with identification of mutations in 60 families. Mol Genet Metab 2003;79:164.",
            "Tada K et al. Hyperglycemia and non-ketotic hyperglycinemia — clinical variants. Brain Dev 1980.",
            "García-Cazorla A et al. NKH: current state and research perspectives. Orphanet J Rare Dis 2022.",
            "NKH International Family Network — Clinical care guidelines 2023.",
        ],
        "differential_diagnosis": [
            {"condition": "GLDC-NKH (P-protein deficiency)", "distinction": "Biochemically IDENTICAL — CSF:plasma ratio ≥0.08, same glycine range. Only gene panel distinguishes AMT from GLDC. GLDC is ~75–80%, AMT ~15%. Treatment identical. GLDC has p.Gly761Arg European founder (attenuated); AMT has p.Arg320His East Asian semi-founder."},
            {"condition": "GCSH-NKH (H-protein deficiency)", "distinction": "Biochemically IDENTICAL — same ratio range, same glycine levels. GCSH ~1% of NKH. Gene panel mandatory. Treatment identical. Even rarer than AMT-NKH."},
            {"condition": "SSADH deficiency (ALDH5A1)", "distinction": "SSADH: GHB ↑↑ in urine/plasma (NOT glycine); glycine NORMAL. CSF:plasma ratio NORMAL. Globus pallidus T2 hyperintensity. VGB ABSOLUTE CI (different mechanism from NKH). No burst-suppression neonatal."},
            {"condition": "Propionic acidemia (PCCA/PCCB) — 'Ketotic hyperglycinemia'", "distinction": "Propionic acidemia: glycine ↑ plasma BUT via methylamine-glycine conjugation saturation. Organic acids: 3-OH-propionate + propionylglycine + methylcitrate. NEVER diagnose NKH without urine organic acids. CSF:plasma ratio NORMAL in propionic acidemia."},
            {"condition": "Methylmalonic acidemia (MUT/MMAA etc.) — 'Ketotic hyperglycinemia'", "distinction": "MMA: glycine ↑ (same mechanism as propionic); methylmalonic acid ↑↑ urine; C3 acylcarnitine; B12-responsive forms exist. Urine organic acids differentiate immediately from AMT-NKH."},
            {"condition": "D-glyceric aciduria (GLYCTK)", "distinction": "D-glyceric aciduria: glycine ↑ plasma; GCS activity NORMAL; gene GLYCTK; L-2-hydroxy-3-oxoadipic acid in urine organic acids; CSF:plasma ratio NORMAL (<0.02). AMT-NKH: GCS absent, ratio ≥0.08."},
        ],
    }
