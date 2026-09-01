#!/usr/bin/env python3
"""POLG Alpers-Huttenlocher Syndrome Dashboard.

Alpers-Huttenlocher Syndrome (AHS) = OMIM #203700.
Progressive epileptic encephalopathy + hepatopathy + psychomotor regression.
Biallelic AR POLG mutations → mtDNA depletion in liver and brain → OXPHOS failure.

POLG (DNA Polymerase Gamma, Catalytic Subunit, 1240 aa, 15q25.1) is the sole
mitochondrial DNA polymerase. Biallelic LOF → mtDNA depletion → AHS.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — most critical drug warning in mito medicine
     VPA + POLG = acute liver failure → death; documented in hundreds of published cases
  2. Epilepsy (100%) — intractable; occipital onset; EPC (epilepsia partialis continua) 60%
  3. Hepatopathy (80%) — transaminase elevation; acute liver failure 30-40%
  4. mtDNA depletion DIAGNOSTIC — liver + brain <30% normal copy number
  5. Founder variants: p.Ala467Thr (European) + p.Trp748Ser (European)
     Most AR Alpers = p.[Ala467Thr];[Trp748Ser] compound het
  6. LEV + phenobarb + clonazepam — preferred AEDs; NO VPA, NO high-dose IV PHT
  7. Liver transplant DOES NOT cure neurological disease
  8. PHT caution — IV fosphenytoin at toxic doses inhibits Complex I; avoid for SE rescue
  9. Propofol AVOID — PRIS risk in any mitochondrial disease
 10. NG/PEG early — dysphagia 70%; high-carb; no prolonged fasting
 11. AR biallelic; Alpers 1931; POLG linkage Naviaux 2004 AJHG
 12. Deoxynucleoside supplementation (dAMP+dCMP) — investigational

POLG BIOLOGY:
POLG (1240 amino acids, 15q25.1) forms heterotrimer: POLG catalytic + 2x POLG2.
Replicates circular mtDNA (~16.6 kb) by strand displacement synthesis.

Domain architecture:
  Exonuclease domain (aa1-440): 3'→5' proofreading; p.Ala467Thr in linker
    aa467 disrupts interdomain folding; ~90% loss of both activities
  Spacer/linker (aa441-815): POLG2 interaction; p.Trp748Ser reduces processivity
  Polymerase domain (aa816-1240): DNA synthesis; YGDTDS catalytic motif;
    p.Gly848Ser disrupts YGDTDS → near-complete polymerase inactivity

Why VPA causes fatal hepatotoxicity in POLG:
  Mechanism 1: VPA directly inhibits POLG polymerase (dNTP competition) →
    complete mtDNA depletion in already-depleted hepatocytes → necrosis
  Mechanism 2: VPA → propionyl-CoA → CoA sequestration → FAO collapse
    → microvesicular steatosis → acute hepatic failure
  Mechanism 3: VPA 4-en-VPA epoxide → direct hepatocyte necrosis
  Combined: lethal in most POLG patients; latency 3wk-9mo; irreversible

PATHOGENIC VARIANT DISTRIBUTION (biallelic AR, n=40, seed-547):
  p.Ala467Thr / p.Trp748Ser compound het: ~40% — most common AHS genotype
  p.Ala467Thr homozygous: ~30%
  p.Gly848Ser / null compound: ~10% — near-complete loss; worst prognosis
  p.Trp748Ser / severe null compound: ~10%
  Other biallelic (p.Arg627Gln, p.Gln497Arg, p.Gly517Val): ~10%
"""

import random
from datetime import date

SEED = 547  # 40-patient cohort seed


def get_overview() -> dict:
    """POLG Alpers-Huttenlocher — overview for /api/polg/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Alpers-Huttenlocher Syndrome (AHS) / POLG-Related mtDNA Depletion Syndrome",
        "gene": "POLG; DNA Polymerase Gamma Catalytic Subunit; mtDNA Replicase; 1240 aa; heterotrimer with POLG2",
        "chromosome": "15q25.1",
        "omim_gene": "174763",
        "omim_disease": "203700",
        "inheritance": "Autosomal Recessive (biallelic POLG) for Alpers-Huttenlocher; Autosomal Dominant (monoallelic) for adPEO/PEO1 — separate milder spectrum",
        "prevalence": "~1:100,000 overall POLG disorders; Alpers-Huttenlocher ~1:250,000; p.Ala467Thr European carrier frequency ~1:200",
        "protein": "POLG 1240 aa; Exonuclease-domain(aa1-440)-Spacer-POLG2-binding(aa441-815)-Polymerase-domain(aa816-1240); mitochondrial matrix",
        "category": "mtDNA Depletion Syndrome / Mitochondrial DNA Maintenance / POLG-Related Disorder",
        "first_described": "Alpers 1931 (progressive sclerosing poliodystrophy); POLG linkage: Naviaux & Nguyen 2004 AJHG",
        "kpis": {
            "epilepsy_pct": 100,
            "epc_pct": 60,
            "hepatopathy_pct": 80,
            "regression_pct": 100,
            "visual_pct": 70,
            "acute_liver_failure_pct": 35,
            "vpa_risk": "ABSOLUTE CONTRAINDICATION — lethal hepatotoxicity",
            "mtdna_depletion": "DIAGNOSTIC — liver + brain <30% normal copy number",
        },
        "clinical_highlights": [
            "VPA (Valproate) = ABSOLUTE CONTRAINDICATION — the most critical drug safety warning in mitochondrial medicine; VPA + POLG = acute liver failure → death; documented in hundreds of cases; no safe dose; even topical VPA is contraindicated",
            "Epilepsy (100%) — CARDINAL, INTRACTABLE; occipital onset → EPC (epilepsia partialis continua) in 60%; status epilepticus >50%; refractory to 3+ AEDs in 80%; EPC is the hallmark seizure type",
            "Hepatopathy (80%) — transaminase elevation; acute liver failure 30-40%; VPA is the most common precipitant (65% of liver failure); spontaneous liver failure also occurs",
            "Psychomotor regression (100%) — initially normal development; rapid regression after first seizures; language loss 90%; ambulation loss 70%; progressive to minimal consciousness",
            "mtDNA depletion DIAGNOSTIC — liver biopsy or post-mortem: <30% normal mtDNA copy number; brain MRS: reduced NAA + lactate peak in occipital cortex",
            "Founder variants — p.Ala467Thr + p.Trp748Ser compound het most common; screen these first; p.Ala467Thr carrier ~1/200 European; 25% sibling recurrence risk",
            "Liver transplant DOES NOT cure AHS — brain mtDNA depletion is primary; neurological disease continues and often accelerates post-transplant",
            "LEV (levetiracetam) preferred AED — IV loading 20-40 mg/kg for SE; no hepatic metabolism; no mito toxicity; no ammonia effect; renal excretion",
            "PHT (phenytoin) CAUTION — IV fosphenytoin at supratherapeutic doses inhibits Complex I; avoid as SE rescue; use midazolam + IV LEV + IV lacosamide instead",
            "Propofol AVOID — PRIS risk in mitochondrial disease; use ketamine + sevoflurane for anaesthesia",
            "MRI — DWI restriction occipital cortex + thalami at presentation; cortical ribbon-like T2 (laminar necrosis); posterior >> anterior atrophy; thalamic degeneration",
            "Deoxynucleoside supplementation (dAMP + dCMP) — investigational; no licensed therapy; compassionate use in European centres; must not delay VPA cessation",
        ],
        "contraindications": [
            {
                "drug": "VPA (Valproate / Valproic Acid / Sodium Valproate / Divalproex)",
                "level": "ABSOLUTE CONTRAINDICATION — DO NOT USE UNDER ANY CIRCUMSTANCES",
                "reason": "Three synergistic lethal mechanisms: (1) Direct POLG inhibition — VPA competes with dNTPs suppressing residual mtDNA replication → complete hepatocyte mtDNA depletion → necrosis; (2) CoA sequestration via propionyl-CoA → fatty acid oxidation collapse → microvesicular steatosis; (3) VPA 4-en-VPA epoxide → direct hepatocyte necrosis. Published evidence: 65-70% of POLG acute hepatic failure had VPA exposure; latency 3wk–9mo; irreversible once established. Emergency action if POLG diagnosed while on VPA: switch immediately to IV levetiracetam; document VPA as permanently contraindicated in all medical records, allergy systems, and emergency letters.",
            },
            {
                "drug": "IV Fosphenytoin / Phenytoin (high-dose for status epilepticus)",
                "level": "CAUTION — avoid as first-line SE rescue; use midazolam + IV LEV instead",
                "reason": "IV fosphenytoin at supratherapeutic concentrations (common in RSE management) inhibits mitochondrial Complex I, exacerbating OXPHOS failure in POLG-deficient neurons. SE rescue protocol without VPA/PHT: (1) Buccal midazolam 0.3 mg/kg; (2) IV LEV 20-40 mg/kg loading; (3) IV lacosamide 200 mg; (4) IV phenobarbitone 15-20 mg/kg; (5) General anaesthesia — thiopentone (NOT propofol — PRIS risk).",
            },
            {
                "drug": "Propofol (anaesthesia / sedation)",
                "level": "AVOID — PRIS (Propofol Infusion Syndrome) risk",
                "reason": "Propofol inhibits mitochondrial Complex I and uncouples OXPHOS. In POLG patients with pre-existing OXPHOS failure, propofol infusion → PRIS (lactic acidosis + rhabdomyolysis + cardiac failure). Use ketamine + volatile agents (sevoflurane/desflurane) for induction and maintenance. Alert anaesthesia team to POLG diagnosis at every procedural encounter.",
            },
            {
                "drug": "Prolonged fasting / NPO protocols",
                "level": "CAUTION — IV 10% dextrose MANDATORY if NPO >4h",
                "reason": "Fasting forces catabolism; POLG-deficient hepatocytes cannot sustain gluconeogenesis adequately. Protocol: oral glucose drinks during illness; if vomiting, IV 10% dextrose GIR 6-8 mg/kg/min; perioperative: IV dextrose from evening before surgery; glucose 2-hourly intra-op.",
            },
            {
                "drug": "Liver transplantation (as curative therapy expectation)",
                "level": "CLINICAL CAUTION — OLT corrects hepatic depletion but NOT brain depletion",
                "reason": "Liver transplant prevents liver-failure death but does NOT reverse brain mtDNA depletion or neurological progression. Post-transplant, seizures and encephalopathy continue and often accelerate (perioperative stress + immunosuppressants worsen neurological state). OLT considered only in: severe but not terminal liver failure + neurological function still present + family/team consensus on goals. Most centres do NOT offer OLT in late-stage AHS. Family must understand: transplant does not restore brain function.",
            },
        ],
        "thresholds": [
            {"marker": "POLG gene panel / WES", "cutoff": "2 pathogenic variants in trans (compound het or homozygous)", "interpretation": "Diagnostic for POLG-related disorder. p.Ala467Thr + p.Trp748Ser = highest AR Alpers risk (~10% residual activity). Classify Alpers-Huttenlocher if: onset <4yr + epilepsy + regression + hepatopathy. Screen p.Ala467Thr and p.Trp748Ser first (common European founder mutations) before full panel."},
            {"marker": "ALT / AST (serum transaminases)", "cutoff": ">3× ULN", "interpretation": "STOP VPA IMMEDIATELY if present. Review all hepatotoxic medications. Repeat LFTs weekly at 3× ULN. At 10× ULN or rising bilirubin: coagulation studies, INR, ammonia, liver team. Quarterly LFT monitoring mandatory in all POLG patients."},
            {"marker": "Serum lactate (resting)", "cutoff": ">3.0 mmol/L persistent", "interpretation": "OXPHOS failure marker. Persistent >3 = metabolic decompensation → admit + IV dextrose + review precipitants. Lactate:pyruvate ratio >20:1 = mitochondrial (not hypoxic) cause. Measure post-prandial and fasting."},
            {"marker": "mtDNA copy number (liver biopsy qPCR)", "cutoff": "<30% of age-matched controls", "interpretation": "Diagnostic for mtDNA depletion syndrome. ND1 or ND4 probe vs nuclear housekeeping. Liver <30% = significant depletion consistent with POLG. Muscle biopsy may be false-negative in early Alpers — liver and brain are primary."},
            {"marker": "MRI — DWI restriction occipital cortex / thalami", "cutoff": "Any DWI restriction in occipital cortex or thalami in child with epilepsy + regression", "interpretation": "High alert for POLG/Alpers-Huttenlocher. Occipital DWI = active cortical neuronal injury from OXPHOS failure. Thalamic DWI = relay nuclei damage → rapid secondary generalisation. In any child with occipital seizures + DWI restriction: POLG sequencing MANDATORY before any AED decision involving VPA."},
            {"marker": "CSF lactate", "cutoff": ">2.2 mmol/L", "interpretation": "CSF lactate elevation = cerebral lactic acidosis from neuronal OXPHOS failure. CSF > serum lactate ratio = CNS-primary production. Also measure CSF amino acids (elevated alanine = indirect marker). CSF protein mildly elevated in POLG."},
        ],
        "ddx_table": [
            {
                "disease": "SCN1A Dravet Syndrome",
                "shared": "Febrile-triggered refractory epilepsy; regression post-SE",
                "distinguishing": "Dravet: SCN1A mutation; fever-triggered; generalised; photosensitive; normal liver; normal MRI early; VPA IS FIRST-LINE in Dravet (vs absolute CI in POLG). POLG: occipital onset; EPC; liver disease; mtDNA depletion; MRI posterior. Screen POLG before VPA in any child with febrile-triggered refractory epilepsy + liver abnormality.",
            },
            {
                "disease": "TMEM70 Complex V Deficiency (3-MGA Type VI)",
                "shared": "Mitochondrial disease; VPA absolute CI",
                "distinguishing": "TMEM70: 3-MGA-uria (100%) + neonatal lactic acidosis (pH <7.1) + hyperammonemia + DCM. POLG/Alpers: NO 3-MGA; epilepsy dominant; hepatopathy; occipital MRI; onset 2mo–4yr not neonatal. Different mechanism; POLG = mtDNA depletion; TMEM70 = Complex V assembly.",
            },
            {
                "disease": "CLPB MGCA7 (3-MGA Type VII)",
                "shared": "Mitochondrial disease; VPA caution",
                "distinguishing": "CLPB: cataracts PATHOGNOMONIC + 3-MGA-uria + neutropenia + NO hepatopathy + NO occipital EPC. VPA = MODERATE CAUTION in CLPB (vs ABSOLUTE CI in POLG). POLG: hepatopathy dominant; EPC; occipital MRI; no cataracts; no neutropenia. CRITICAL: POLG must be ruled out before prescribing VPA in ANY mitochondrial disease — CLPB protocol explicitly states this.",
            },
            {
                "disease": "MELAS (m.3243A>G most common)",
                "shared": "Mitochondrial disease; stroke-like episodes; occipital DWI; lactic acidosis",
                "distinguishing": "MELAS: maternal inheritance (mitochondrial DNA); m.3243A>G mtDNA point mutation; stroke-like episodes (not EPC); ragged-red fibres + COX-normal; L-arginine treatment; tRNA mutation not POLG. POLG: AR biallelic; EPC; hepatopathy; liver mtDNA depletion; POLG nuclear gene.",
            },
            {
                "disease": "Herpes Simplex Encephalitis (HSE)",
                "shared": "Temporal/occipital DWI restriction; fever + seizures; regression",
                "distinguishing": "HSE: CSF HSV PCR positive; DWI restriction temporal > occipital; acute onset (days not months); responds to aciclovir. POLG: subacute-chronic months; liver disease; mtDNA depletion; negative viral PCR. CRITICAL: start aciclovir empirically in any acute encephalitis — do NOT wait for POLG result.",
            },
            {
                "disease": "SURF1 Leigh Syndrome (COX deficiency)",
                "shared": "Mitochondrial disease; progressive encephalopathy; lactic acidosis",
                "distinguishing": "SURF1/Leigh: bilateral basal ganglia + brainstem T2 (PATHOGNOMONIC); COX-absent on muscle; elevated succinate; no hepatopathy as primary. POLG/Alpers: occipital cortex + thalami (not basal ganglia Leigh); prominent hepatopathy; EPC; no basal ganglia lesions early.",
            },
        ],
    }


def get_breakdown() -> dict:
    """POLG Alpers-Huttenlocher — patient breakdown for /api/polg/breakdown."""
    rng = random.Random(SEED)
    n = 40

    phenotype_groups = [
        ("Classic AHS: onset <12mo, EPC + hepatopathy + rapid regression", 20),
        ("Typical AHS: onset 12-36mo, occipital epilepsy + liver involvement + regression", 14),
        ("Attenuated AHS / MCHS overlap: onset 2-4yr, milder regression, liver predominant", 6),
    ]
    assert sum(c for _, c in phenotype_groups) == n

    variant_dist = [
        {"variant": "p.Ala467Thr / p.Trp748Ser compound heterozygous (European founder)", "n_alleles": 32, "pct": 40, "effect": "Most common AHS genotype; ~10% residual POLG activity; severe infantile Alpers; EPC onset <18mo; hepatopathy 75%; rapid regression; p.Ala467Thr carrier ~1/200 European; founder effect Northern European + British"},
        {"variant": "p.Ala467Thr homozygous (European)", "n_alleles": 24, "pct": 30, "effect": "Second most common; slightly more residual function than compound het; onset 12-24mo typical; hepatopathy 60%; slower progression in first 2yr then accelerates; prognosis similar to compound het by age 5"},
        {"variant": "p.Gly848Ser / null compound heterozygous (severe)", "n_alleles": 8, "pct": 10, "effect": "Near-complete polymerase loss; p.Gly848Ser disrupts YGDTDS catalytic motif; neonatal onset in some; worst prognosis; median survival 18mo from onset; severe brain + liver depletion at post-mortem"},
        {"variant": "p.Trp748Ser / severe null compound", "n_alleles": 8, "pct": 10, "effect": "p.Trp748Ser reduces POLG2 binding → processivity failure; with null second allele → severe AHS; alone (homozygous) → milder ANS/SANDO; genotype prediction requires knowing BOTH alleles"},
        {"variant": "Other biallelic missense (p.Arg627Gln, p.Gln497Arg, p.Gly517Val)", "n_alleles": 8, "pct": 10, "effect": "Heterogeneous; >200 known pathogenic POLG variants; activity 5-30% depending on variant; Alpers phenotype if combined activity <10%; POLG mutation database + functional assay for VUS classification"},
    ]

    treatment_dist = [
        {"treatment": "Levetiracetam (LEV) — oral and IV", "n": 38, "pct": 95, "indication": "Level A — preferred first-line AED; no hepatic metabolism; no mito toxicity; no ammonia; no P450 induction; IV loading for SE: 20-40 mg/kg; oral maintenance 30-50 mg/kg/day divided 2-3; renal excretion"},
        {"treatment": "NG or PEG feeding", "n": 32, "pct": 80, "indication": "Level A — dysphagia 70%; high-carbohydrate enteral formula; NG from first sign of dysphagia; PEG if >6-week requirement; avoid prolonged fasting; GIR 6-8 mg/kg/min; nutrition team from diagnosis"},
        {"treatment": "IV Dextrose (sick-day + perioperative)", "n": 40, "pct": 100, "indication": "Level A — any intercurrent illness + NPO: IV 10% dextrose GIR 6-8 mg/kg/min; prevent catabolism → lactic acidosis; perioperative: start IV dextrose evening before; glucose 2-hourly intra-op"},
        {"treatment": "Buccal / IV Midazolam (acute seizure rescue)", "n": 36, "pct": 90, "indication": "Level A — first-line acute rescue; buccal 0.3 mg/kg (max 10 mg) for seizures >5min; IV infusion for SE; families trained in buccal midazolam; preferred over rectal diazepam"},
        {"treatment": "Clonazepam", "n": 30, "pct": 75, "indication": "Level B — GABA-A positive; effective in EPC + myoclonic components; no hepatotoxicity; buccal or oral; maintenance 0.05-0.2 mg/kg/day; tolerance can develop"},
        {"treatment": "Phenobarbitone (PB)", "n": 26, "pct": 65, "indication": "Level B — broad-spectrum; IV loading 15-20 mg/kg for SE (after BZD + LEV); oral 3-5 mg/kg/day; not hepatotoxic in POLG unlike VPA; sedation + respiratory depression monitoring"},
        {"treatment": "Lacosamide (IV + oral)", "n": 14, "pct": 35, "indication": "Level B — second-line IV AED for SE after BZD + LEV; slow sodium channel inactivation; not hepatically metabolised; IV 200-400 mg loading; gaining use in POLG centres as VPA-safe IV option"},
        {"treatment": "Physiotherapy + occupational therapy", "n": 38, "pct": 95, "indication": "Level A — motor regression; spasticity; contracture prevention; positioning; adaptive equipment; from diagnosis; intensity increases as regression progresses"},
        {"treatment": "Palliative / goals-of-care team", "n": 40, "pct": 100, "indication": "Level A — AHS is progressive, fatal (median survival 2-4yr from onset); palliative care from diagnosis; goals-of-care discussion; hospice planning; symptom management in terminal phase"},
        {"treatment": "Riboflavin (B2) + CoQ10 empirical", "n": 20, "pct": 50, "indication": "Level D — empirical mito cofactor supplementation; no controlled evidence in POLG; generally safe; does not alter disease course; used as low-risk supportive therapy in many centres"},
        {"treatment": "Felbamate", "n": 10, "pct": 25, "indication": "Level C — refractory EPC not controlled on LEV + PB + CLZ; NMDA antagonism useful in cortical hyperexcitability; risk: aplastic anaemia + hepatotoxicity (monitoring); specialist centres only"},
        {"treatment": "Deoxynucleoside supplementation (dAMP+dCMP) — investigational", "n": 4, "pct": 10, "indication": "Level D (investigational) — replenish dNTP pool depleted by POLG dysfunction; oral dAMP + dCMP; compassionate use European centres; no RCT data; families must understand experimental status"},
    ]

    seizure_profile = [
        {"type": "EPC (Epilepsia Partialis Continua)", "n": 24, "pct": 60, "desc": "Continuous focal motor seizure >1h; jerk/tremor one body part; often hand/face after occipital onset; highly resistant; hallmark of AHS; correlates with contralateral cortical neuronal loss"},
        {"type": "Focal occipital onset (visual aura)", "n": 32, "pct": 80, "desc": "Visual disturbance + flashing lights + hemianopia + ictal blindness; primary occipital cortex degeneration; earliest seizure type; EEG: occipital spike-wave"},
        {"type": "Generalised tonic-clonic (secondary)", "n": 28, "pct": 70, "desc": "Secondary generalisation from occipital focus; thalamic relay → bilateral spread; often prolonged; high SE risk"},
        {"type": "Status epilepticus (SE)", "n": 22, "pct": 55, "desc": "Often refractory (RSE/SRSE); most common precipitant of acute neurological deterioration; fever or VPA the key triggers; manages: midazolam → IV LEV → PB IV → ICU (thiopentone, NOT propofol)"},
        {"type": "Myoclonic seizures", "n": 16, "pct": 40, "desc": "Action or stimulus-sensitive; cortical origin; late feature; correlates with cortical hyperexcitability in degenerating occipital-parietal cortex; clonazepam most effective"},
        {"type": "Absence-like / CSWS", "n": 16, "pct": 40, "desc": "Continuous spike-wave during slow-wave sleep (CSWS) in 40%; contributes to nocturnal cognitive decline; night-time EEG monitoring important for CSWS detection"},
    ]

    hepatic_outcomes = [
        {"outcome": "Transaminase elevation (>3× ULN at any point)", "n": 32, "pct": 80, "notes": "Most common hepatic finding; insidious or precipitous; quarterly LFT mandatory; stop ALL hepatotoxic drugs at 3× ULN"},
        {"outcome": "Acute liver failure (fulminant)", "n": 14, "pct": 35, "notes": "VPA-precipitated in 65%; spontaneous in 35%; INR >1.5 + jaundice + encephalopathy = liver failure; mortality 80% without transplant"},
        {"outcome": "Coagulopathy (INR >1.3)", "n": 16, "pct": 40, "notes": "Synthetic liver function failure; FFP + vitamin K; holds elective procedures; emergency anaesthesia risk"},
        {"outcome": "Jaundice / hyperbilirubinemia", "n": 10, "pct": 25, "notes": "Direct hyperbilirubinemia; ursodeoxycholic acid for cholestasis; late hepatic sign"},
        {"outcome": "Hepatic fibrosis / cirrhosis (biopsy/post-mortem)", "n": 20, "pct": 50, "notes": "Microvesicular steatosis → lobular inflammation → bridging fibrosis → cirrhosis; mtDNA depletion on qPCR confirms diagnosis"},
    ]

    biomarker_summary = {
        "epilepsy_pct": 100,
        "epc_pct": 60,
        "status_epilepticus_pct": 55,
        "hepatopathy_pct": 80,
        "acute_liver_failure_pct": 35,
        "vpa_exposure_in_hepatic_failure_pct": 65,
        "regression_pct": 100,
        "language_loss_pct": 90,
        "ambulation_loss_pct": 70,
        "visual_involvement_pct": 70,
        "cortical_blindness_pct": 30,
        "ataxia_neuropathy_pct": 60,
        "mtdna_depletion_liver_pct": 90,
        "mri_occipital_dri_pct": 75,
        "lactate_elevated_pct": 80,
        "csf_lactate_elevated_pct": 70,
        "median_onset_months": 12,
        "median_diagnosis_delay_months": 8,
        "median_survival_from_onset_months": 36,
    }

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "phenotype_groups": [{"group": g, "n": c, "pct": round(c / n * 100)} for g, c in phenotype_groups],
        "variant_distribution": variant_dist,
        "treatment_distribution": treatment_dist,
        "seizure_profile": seizure_profile,
        "hepatic_outcomes": hepatic_outcomes,
        "biomarker_summary": biomarker_summary,
        "outcomes": {
            "median_survival_months_from_onset": 36,
            "vpa_liver_failure_risk_pct": 65,
            "refractory_epilepsy_pct": 80,
            "language_loss_by_2yr_pct": 90,
            "ambulation_loss_by_3yr_pct": 70,
            "liver_transplant_considered_pct": 25,
            "cortical_blindness_pct": 30,
            "palliative_care_enrolled_pct": 100,
            "median_diagnosis_delay_months": 8,
        },
    }


def get_definitions() -> dict:
    """POLG Alpers-Huttenlocher — definitions for /api/polg/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Alpers-Huttenlocher Syndrome (AHS) / POLG-Related mtDNA Depletion Syndrome",
        "gene": "POLG",
        "omim_gene": "174763",
        "omim_disease": "203700",
        "definitions": [
            {
                "term": "POLG — DNA Polymerase Gamma and mtDNA Replication",
                "definition": "POLG (DNA Polymerase Gamma, Catalytic Subunit; 1240 aa; 15q25.1) is the sole enzyme responsible for replicating the circular 16.6 kb mitochondrial genome. It functions as a heterotrimer: one POLG catalytic subunit + two POLG2 processivity-clamp subunits. POLG carries both 5'→3' DNA synthesis (polymerase domain, aa816-1240) and 3'→5' proofreading exonuclease (exonuclease domain, aa1-440). Each human cell contains 100–10,000 mtDNA copies; high-energy-demand tissues (neurons, hepatocytes) require ≥70% normal copy number for adequate OXPHOS. Biallelic POLG LOF → mtDNA replication rate <30% normal → depletion below threshold → 13 mtDNA-encoded OXPHOS subunits (7 Complex I + 1 CIII + 3 CIV + 2 CV) insufficient → ATP production collapses → neuronal + hepatocyte death in Alpers-Huttenlocher.",
                "relevance": "POLG activity can be measured in patient fibroblasts (in vitro DNA synthesis assay). Residual activity predicts severity: <10% = Alpers; 10-30% = MCHS/MEMSA; >30% = ANS/PEO. Fibroblast mtDNA copy number (qPCR: ND1 probe vs nuclear housekeeping) measurable in any clinical genetics lab. Muscle biopsy: Gomori trichrome (ragged-red fibres) + COX/SDH histochemistry. Muscle may be early-normal in Alpers — brain and liver have earlier depletion. WES/WGS with POLG panel is standard diagnostic approach. p.Ala467Thr and p.Trp748Ser should be screened first in any European patient with suspected POLG disease.",
            },
            {
                "term": "Why VPA is Absolutely Contraindicated in POLG — Mechanism and Evidence",
                "definition": "Valproate (VPA) is the most dangerous drug for POLG-mutation carriers. Three synergistic lethal mechanisms: (1) POLG Direct Inhibition: VPA and its metabolite 4-en-VPA directly inhibit POLG polymerase by competing with dNTPs at the polymerase active site. In a patient with already-impaired POLG, VPA can reduce residual polymerase function a further 30-70%, causing complete mtDNA depletion in hepatocytes → necrosis within weeks. (2) CoA Sequestration: VPA is metabolised to propionyl-CoA, which sequesters mitochondrial Coenzyme A. CoA depletion disrupts fatty acid oxidation + urea cycle + TCA simultaneously → microvesicular steatosis + hyperammonemia. In POLG-deficient hepatocytes with pre-existing OXPHOS failure, this is rapidly fatal. (3) Reactive Metabolite: VPA 4-en-VPA epoxide is hepatotoxic through direct covalent protein modification — normally detoxified by glucuronidation but this pathway fails in mitochondrially impaired liver. Clinical evidence: 65-70% of POLG acute hepatic failure patients had VPA exposure; latency 3wk–9mo; irreversible; fatal in 80% without transplant.",
                "relevance": "NEVER prescribe VPA to: (1) any child with unexplained refractory epilepsy + liver disease; (2) any known or suspected mitochondrial disease; (3) any POLG patient; (4) any sibling of a POLG patient before genetic testing. VPA must be excluded BEFORE POLG sequencing is completed when epilepsy is refractory — not after. Emergency protocol if POLG diagnosed while on VPA: switch immediately to IV levetiracetam (20 mg/kg loading + 30-50 mg/kg/day) and document VPA as permanently contraindicated in ALL medical records, allergy systems, and emergency letters. This is the most important prescribing safety action in mitochondrial medicine.",
            },
            {
                "term": "EPC (Epilepsia Partialis Continua) — Hallmark Seizure of Alpers-Huttenlocher",
                "definition": "Epilepsia Partialis Continua (EPC, Kojevnikov's syndrome) = continuous focal motor seizure (repetitive rhythmic jerking of a body part) lasting >1 hour without loss of consciousness. EPC occurs in 60% of Alpers patients and is the hallmark seizure type. In AHS, EPC typically involves the hands or face (somatosensory-motor cortex) following initial visual/occipital seizures. Mechanism: massive occipital and parietal cortical neuron loss → chronic cortical hyperexcitability → persistent focal motor discharge. EPC is highly refractory — responds partially to LEV, clonazepam, felbamate, IV lacosamide. EPC episodes last hours, days, or weeks continuously. Unlike typical focal seizures, EPC can occur while the child appears conscious (though impaired). EPC correlates with neuroimaging: DWI restriction in motor/parietal cortex contralateral to affected limb during active EPC.",
                "relevance": "EPC in a child with prior developmental regression + liver disease = POLG/Alpers until proven otherwise. EEG during EPC: continuous spike-wave at 2-4 Hz in contralateral centro-parietal-occipital region; may generalise. Treatment: (1) IV levetiracetam: 20-40 mg/kg loading — most effective available; (2) IV midazolam: 0.15 mg/kg bolus + infusion; (3) IV lacosamide: 200-400 mg loading; (4) IV phenobarbitone if above fail; (5) Felbamate oral; (6) Ketamine anaesthesia as last resort. NEVER use fosphenytoin at supratherapeutic doses for EPC — Complex I inhibition accelerates cortical injury.",
            },
            {
                "term": "POLG Disease Spectrum — Alpers to PEO (Severity Axis and Prescribing Implications)",
                "definition": "POLG mutations cause a clinical spectrum from most severe (Alpers-Huttenlocher, infantile, AR biallelic) to mildest (isolated chronic progressive external ophthalmoplegia, adult onset, AD monoallelic). Intermediate: MCHS (Myocerebrohepatopathy Spectrum, early childhood), MEMSA (Myoclonic Epilepsy Myopathy Sensory Ataxia, childhood/adolescence), ANS / SANDO / MIRAS (Ataxia-Neuropathy Spectrum, adulthood). Severity correlates with residual POLG polymerase activity: <10% = Alpers; 10-20% = MCHS/MEMSA; 20-35% = ANS/SANDO/MIRAS; >35% = arPEO/adPEO. Genotype-phenotype: p.Ala467Thr + p.Trp748Ser compound het (~10% activity) → Alpers. p.Trp748Ser homozygous (~20% activity) → MIRAS (ataxia, neuropathy — NO childhood epilepsy). Dominant POLG (p.Tyr955Cys) → ptosis + PEO in adults — NOT Alpers.",
                "relevance": "VPA absolute CI applies to ALL biallelic POLG (Alpers + MCHS + MEMSA + ANS/SANDO/MIRAS). Dominant heterozygous POLG carrier parents do NOT need VPA restriction — they have one normal POLG allele and normal activity. Genetic counselling: AR = 25% sibling recurrence; prenatal diagnosis (CVS/amnio) available; preimplantation genetic testing possible. An adult with SANDO or MIRAS who needs AED for seizures: VPA is still absolutely contraindicated — the biallelic AR genotype is the same whether the presentation is Alpers or SANDO.",
            },
            {
                "term": "MRI Findings in POLG/Alpers — Posterior-Predominant Progressive Pattern",
                "definition": "MRI in Alpers-Huttenlocher evolves with disease: Early phase (months 1-6 from symptom onset): DWI restriction in occipital cortex + posterior thalami = cytotoxic oedema from acute neuronal OXPHOS failure; may mimic stroke-like episodes (DDx MELAS) but follows sulcal-gyral pattern not vascular territory. Intermediate phase (6-24 months): T2/FLAIR cortical ribbon-like hyperintensity in occipital > parietal; cortical laminar necrosis pattern; thalamic T2 changes bilateral; progressive occipital volume loss. Late phase (>24 months): generalised cortical atrophy; posterior >> anterior; cerebellar atrophy; white matter T2 signal periventricular; corpus callosum thinning. MRS: lactate peak in affected cortex (OXPHOS failure); reduced NAA in occipital cortex (neuronal loss). SPECT/PET: occipital hypometabolism correlated with seizure focus.",
                "relevance": "Serial MRI every 6-12 months tracks disease progression. DWI restriction is the earliest sensitive marker — start MRI at diagnosis; if normal, repeat at 3-6 months. Occipital DWI + seizures + regression = emergency POLG sequencing + metabolic genetics same day. MRI also guides AED choice: thalamic involvement predicts rapid secondary generalisation → aggressive SE management protocol. Key DDx: MELAS tends posterior-parietal-occipital too — differentiate by maternal inheritance (MELAS) vs AR biallelic POLG; m.3243A>G mtDNA mutation (MELAS) vs POLG nuclear gene; hepatopathy (POLG) vs stroke-like episodes (MELAS). Herpes simplex encephalitis: temporal predominant + acute onset → always empiric aciclovir pending HSV PCR before establishing POLG diagnosis.",
            },
            {
                "term": "Emergency Management: SE in POLG Without VPA — Protocol",
                "definition": "Managing status epilepticus (SE) in POLG WITHOUT VPA requires specific protocol: Stage 1 (0-5 min): Buccal midazolam 0.3 mg/kg (max 10 mg) at home/first-responder; lorazepam 0.1 mg/kg IV if IV access. Stage 2 (5-20 min, hospital): IV levetiracetam 20-40 mg/kg over 15 min; IV midazolam infusion 0.1-0.2 mg/kg/h; IV lacosamide 200 mg (adults) / 6-8 mg/kg (children) if refractory. Stage 3 (20-60 min, RSE): IV phenobarbitone 15-20 mg/kg; intubation team standby. Stage 4 (>60 min, SRSE): GA — thiopentone or ketamine infusion (NOT propofol — PRIS risk); continuous EEG; ICU. Liver protection throughout: IV dextrose 10% GIR 8 mg/kg/min; LFTs + lactate + ammonia + glucose 6-hourly; stop ALL hepatotoxic drugs.",
                "relevance": "The most preventable cause of death in POLG/Alpers is VPA administration by a non-specialist team in the emergency setting. Every POLG family must hold an emergency letter stating: POLG ALPERS SYNDROME — VALPROATE (VPA) IS ABSOLUTELY CONTRAINDICATED — DO NOT GIVE VPA IN ANY FORM. Letter should be registered in local ED systems, hospital allergy flags, ambulance service medical ID. Every AHS family should have: (1) home buccal midazolam with written protocol; (2) emergency VPA contraindication card; (3) direct metabolic team out-of-hours contact. SE in POLG = metabolic emergency — each SE episode causes irreversible neuronal loss from acute OXPHOS failure.",
            },
            {
                "term": "mtDNA Depletion Diagnosis — Tissues, Methods, and Clinical Use",
                "definition": "mtDNA depletion defined as <30% of age-matched normal control mtDNA copy number in affected tissue. In POLG/Alpers: most severe in brain (occipital > frontal) and liver. Muscle depletion less reliable — may be absent or mild in early Alpers, making muscle biopsy non-diagnostic unlike CPEO (where muscle is primary). Diagnosis methods: (1) Liver biopsy: qPCR with ND1 or ND4 probe vs nuclear housekeeping gene (GAPDH or β2-microglobulin); histology: microvesicular steatosis + lobular inflammation + bridging fibrosis. (2) Post-mortem brain: occipital > frontal cortical neuron loss + reactive astrogliosis = 'progressive sclerosing poliodystrophy'. (3) Fibroblasts: mtDNA copy number + POLG activity assay + yeast complementation for VUS classification. MRS: lactate peak + reduced NAA in occipital cortex; early changes precede conventional MRI abnormality.",
                "relevance": "Liver biopsy in POLG patients with ALT/AST >5× ULN, coagulopathy, or uncertain VUS: confirm INR <1.5, platelets >80 before proceeding; transjugular route if coagulopathy. Do NOT delay diagnosis waiting for liver biopsy if clear biallelic POLG pathogenic variants are confirmed and phenotype is typical — genetic diagnosis sufficient; biopsy adds functional confirmation but must not delay VPA cessation. Muscle biopsy is NOT the primary diagnostic tissue in Alpers — may show ragged-red fibres + COX-negative fibres in later disease but is frequently normal early. Brain imaging (MRS) is a non-invasive surrogate for brain mtDNA depletion.",
            },
        ],
    }
