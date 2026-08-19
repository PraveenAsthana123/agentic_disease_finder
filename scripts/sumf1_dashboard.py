"""
SUMF1 Epilepsy — Multiple Sulfatase Deficiency (Austin Disease)
===============================================================
40-patient cohort · SUMF1 (3p26.1) · Autosomal Recessive (AR) biallelic LOF
SUMF1 encodes Formylglycine-Generating Enzyme (FGE, 374 aa, ~42 kDa):
  FGE post-translationally converts Cys → formylglycine in the active site of
  ALL 17 cellular sulfatases (including ARSA, ARSB, IDS, IDS2, GNS, GALNS, SGSH,
  HGSNAT, HYAL, STS/ARSC, ARSD, ARSE, ARSF, ARSG, ARSH, ARSI, ARSJ).
  SUMF1 LOF → ALL sulfatases simultaneously lose catalytic activity →
  COMBINED phenotype of ALL the sulfatase-deficiency diseases simultaneously.

DISEASE — MULTIPLE SULFATASE DEFICIENCY (MSD) / AUSTIN DISEASE (OMIM 272200):
  Simultaneously expresses features of:
    MLD (ARSA deficiency) → sulfatide accumulation → leukodystrophy + neuropathy
    MPS II variant (IDS deficiency) → heparan/dermatan sulfate → dysostosis multiplex
    MPS III variant (SGSH/GNS deficiency) → heparan sulfate → severe intellectual disability
    MPS IV variant (GALNS deficiency) → keratan sulfate → skeletal dysplasia
    X-linked ichthyosis (STS/ARSC deficiency) → cholesterol-3-sulfate → ichthyosis
  No single-enzyme LSD produces this combined MLD + MPS + ichthyosis phenotype.

PATHOGNOMONIC FEATURES:
  (1) ALL SULFATASES SIMULTANEOUSLY LOW (PATHOGNOMONIC — MSD DISEASE HALLMARK):
      Concurrent reduction of ≥3 sulfatase activities (ARSA, ARSB, IDS) on standard
      enzyme assays distinguishes MSD from all isolated sulfatase deficiencies.
      Isolated ARSA low = MLD only; isolated ARSB low = MPS VI only; ALL low = MSD.
  (2) COMBINED URINE SULFATIDES + ELEVATED GAGs (PATHOGNOMONIC):
      Urine sulfatides elevated (50–200× normal, like MLD) PLUS urine GAGs elevated
      (heparan + dermatan + keratan sulfate, like MPS) simultaneously. No isolated
      sulfatase disease causes both sulfatiduria + GAGuria at the same time.
  (3) ICHTHYOSIS (PATHOGNOMONIC for Type C / attenuated MSD):
      Dry, scaly skin (X-linked ichthyosis pattern) due to STS/ARSC deficiency
      component. Seen in 85% of Type C MSD; absent in isolated MLD (ARSA) or
      isolated MPS. Ichthyosis + leukodystrophy = MSD until proven otherwise.
  (4) COMBINED LEUKODYSTROPHY + DYSOSTOSIS MULTIPLEX ON IMAGING (PATHOGNOMONIC):
      Tigroid/leopard-skin MRI pattern (from ARSA/MLD) PLUS dysostosis multiplex
      on skeletal X-ray (from IDS/MPS component) in same patient — no isolated
      sulfatase disease produces both simultaneously.

EPILEPSY — SUMF1/MSD-SPECIFIC:
  Infantile spasms (IS): Types A + B predominantly; hypsarrhythmia on EEG.
  Progressive myoclonus (PME-like): All types; action myoclonus; cortical origin.
  Focal seizures: temporal/frontal onset; evolves to bilateral tonic-clonic.
  Seizure prevalence: Type A 85%, Type B 78%, Type C 55%, Attenuated 30%.
  Drug resistance: 65% overall; highest in Types A + B.
  EEG: hypsarrhythmia (Type A) → multifocal spikes → diffuse slowing.

SUBTYPES (by residual FGE activity):
  Type A (Neonatal/Severe): <1% FGE, all sulfatases undetectable, neonatal onset,
    fatal <2 years; panorganic storage; seizures prominent (IS, tonic).
  Type B (Severe Infantile): <5% FGE, onset 1–2 years, severe ID, seizures 78%.
  Type C (Late Infantile / Attenuated): 5–15% FGE, onset 2–4 years, ichthyosis
    prominent, slower progression, seizures 55%.
  Attenuated (Juvenile/Adult): >15% FGE, rare missense; adult onset in some.

DRUG SAFETY:
  CBZ/OXC: HIGH RISK — peripheral neuropathy additive (MLD/ARSA component); worsens
    neuropathy in all types with peripheral nerve involvement.
  PHT (Phenytoin): HIGH RISK — peripheral neuropathy additive (same mechanism as CBZ).
  VGB: HIGH RISK in Types A + B — visual cortex MLD lesions + retinal toxicity.
  Typical Antipsychotics (Haloperidol, Chlorpromazine): HIGH RISK — psychiatric
    misdiagnosis common in Type C adult-onset; drug-induced myoclonus risk.
  TGB (Tiagabine): HIGH RISK — worsens myoclonus (PME pattern) across all types.
  VPA: SAFE — lysosomal (not mitochondrial) pathway; POLG1 exclusion mandatory
    as standard of care (CPIC-POLG1-2023); LFT monitoring 3-monthly.
  ACTH: Level A — infantile spasms (Types A + B); ACNS/AES 2022 first-line.
  LEV: Level B — broad-spectrum; focal + GTCS + myoclonus; no pathway conflict.
  Piracetam: Level C — PME/myoclonus adjunct (cortical myoclonus suppression).
  Clobazam: Level C — focal adjunct; less sedating than clonazepam.
  Miglustat: experimental (substrate reduction, MPS component); not standard of care.
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

GENE        = "SUMF1 (Sulfatase Modifying Factor 1 / Formylglycine-Generating Enzyme FGE)"
LOCUS       = "3p26.1"
OMIM        = "607939 (SUMF1 gene); 272200 (Multiple Sulfatase Deficiency / Austin Disease)"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF; residual FGE activity determines subtype severity"
COHORT_SIZE = 40

ETIOLOGIES = [
    {
        "name": "Type A — Neonatal / Severe MSD",
        "pct": 30,
        "onset": "Neonatal (birth – 6 months)",
        "notes": "FGE activity <1%; ALL sulfatases undetectable; panorganic lysosomal storage; "
                 "severe hypotonia, hepatosplenomegaly, facial dysmorphism, IS (85%); rapidly fatal (<2 years); "
                 "MRI: diffuse leukodystrophy + cerebral atrophy; periosteal thickening on X-ray",
        "key_finding": "ALL sulfatase activities undetectable on enzyme panel — PATHOGNOMONIC; biallelic null variants",
    },
    {
        "name": "Type B — Severe Infantile MSD",
        "pct": 35,
        "onset": "Early infantile (6–24 months)",
        "notes": "FGE activity <5%; 5–7 sulfatases severely reduced; onset 6–24 months; severe intellectual disability; "
                 "seizures 78% (IS → focal → PME); peripheral neuropathy 90%; ichthyosis 60%; "
                 "dysostosis multiplex; tigroid MRI pattern; survival to early childhood",
        "key_finding": "ARSA + ARSB + IDS all low simultaneously on enzyme panel; urine: sulfatides + GAGs elevated",
    },
    {
        "name": "Type C — Late Infantile / Attenuated MSD",
        "pct": 28,
        "onset": "Late infantile / early childhood (2–4 years)",
        "notes": "FGE activity 5–15%; attenuated severity; ichthyosis 85% (X-linked ichthyosis pattern from STS); "
                 "slower leukodystrophy progression; seizures 55%; psychiatric features in some; "
                 "survival to adolescence–adulthood; dysostosis mild; hepatosplenomegaly mild",
        "key_finding": "Ichthyosis + leukodystrophy combination PATHOGNOMONIC for Type C MSD (STS component + ARSA component)",
    },
    {
        "name": "Attenuated / Juvenile / Adult MSD",
        "pct": 5,
        "onset": "Juvenile / adult (>10 years)",
        "notes": "FGE activity >15%; rare missense variants preserving partial FGE function; very slow progression; "
                 "psychiatric presentation in adult onset (35% misdiagnosed as schizophrenia); "
                 "ichthyosis may be sole presenting feature; seizures 30%; mild cognitive decline",
        "key_finding": "Adult psychiatric presentation + ichthyosis → check ALL sulfatase activities (not just ARSA)",
    },
    {
        "name": "Compound Heterozygous (Null + Missense)",
        "pct": 2,
        "onset": "Variable (infantile – childhood)",
        "notes": "One null allele + one missense allele; intermediate phenotype; residual activity from missense allele "
                 "modifies severity; phenotype closer to more severe allele (null dominant effect for FGE); "
                 "clinical range from Type B to Type C severity",
        "key_finding": "Severity predicted by null allele dominant effect; phenotype correlates with lower-activity allele",
    },
]

SEIZURE_TYPES = [
    {"type": "Infantile Spasms (IS) — hypsarrhythmia", "pct": 45,
     "subtype": "Types A (85%) + B (65%); hypsarrhythmia on EEG; ACTH Level A first-line; "
                "VGB HIGH RISK (visual cortex MLD lesions additive)"},
    {"type": "Progressive Myoclonus (action myoclonus)", "pct": 55,
     "subtype": "All types (B 70%, C 50%, A 45%); cortical origin; PME-like pattern; "
                "piracetam Level C adjunct; avoid TGB (worsens myoclonus)"},
    {"type": "Focal Seizures (temporal/frontal onset)", "pct": 52,
     "subtype": "Types B + C (50–60%); impaired awareness + bilateral spread; "
                "LEV Level B; CBZ HIGH RISK (neuropathy additive — MLD component)"},
    {"type": "Generalised Tonic-Clonic (GTCS)", "pct": 42,
     "subtype": "All types; VPA Level B; LEV Level B alternative; "
                "CBZ/OXC avoid (peripheral neuropathy in MLD/ARSA component)"},
    {"type": "Tonic Seizures", "pct": 28,
     "subtype": "Types A + B (neonatal/severe); nocturnal predominance; "
                "clobazam Level C adjunct; may cluster with IS"},
    {"type": "Absence-like (atypical)", "pct": 18,
     "subtype": "Type C (18%); atypical slow spike-wave; clobazam preferred over ethosuximide; "
                "psychiatric misdiagnosis risk in adult Type C"},
]

TRIGGERS = [
    {"trigger": "Febrile illness / infection", "pct": 78,
     "notes": "Most potent trigger; lysosomal stress amplified during inflammatory response; "
              "sulfatase substrate accumulation accelerated; relevant across all types"},
    {"trigger": "Sleep deprivation", "pct": 60,
     "notes": "Cortical myoclonus threshold lowered; PME-like worsening; "
              "seizure diary essential; relevant Types B + C"},
    {"trigger": "Missed AED dose", "pct": 65,
     "notes": "Cluster seizures triggered by non-adherence; "
              "complex regimen (multiple pathway interactions) increases non-adherence risk"},
    {"trigger": "Physical exertion / voluntary movement", "pct": 48,
     "notes": "Action myoclonus precipitated by movement; "
              "Type B PME-like; worsened by fatigue; adapt occupational therapy accordingly"},
    {"trigger": "Sensory stimulation (touch / sound)", "pct": 35,
     "notes": "Reflex myoclonus; particularly neonatal/severe forms; "
              "startle response exaggerated in Types A + B; sensory reduction protocols"},
    {"trigger": "Hyperventilation", "pct": 25,
     "notes": "Focal + absence-like (Type C); HV provocation on EEG diagnostic; "
              "leukodystrophy component amplifies HV response"},
    {"trigger": "Intercurrent illness / metabolic stress", "pct": 58,
     "notes": "Dehydration + metabolic stress accelerate lysosomal storage; "
              "relevant during any illness; emergency protocol essential for parents"},
    {"trigger": "Contraindicated drug exposure", "pct": 100,
     "notes": "CBZ/OXC/PHT worsen peripheral neuropathy (MLD/ARSA component) in 100% of exposed cases; "
              "TGB precipitates myoclonus; typical antipsychotics increase seizure risk + myoclonus"},
]

TREATMENTS = [
    {
        "treatment": "ACTH (Adrenocorticotropic Hormone)",
        "level": "A",
        "indication": "Infantile spasms (Types A + B); hypsarrhythmia on EEG; ACNS/AES 2022 first-line IS guideline",
        "mechanism": "Suppresses CRH/ACTH-mediated epileptogenic network; reduces hypsarrhythmia amplitude; "
                     "proven IS efficacy (ACNS Level A); preferred over VGB in MSD (visual cortex lesions)",
        "monitoring": "BP, electrolytes, glucose (hyperglycaemia risk), weight; CBC for infection; "
                      "short-course 2–6 weeks; ophthalmology if VGB considered",
        "caution": "VGB NOT first-line in Types A + B — visual cortex MLD lesions + VGB retinal toxicity additive; "
                   "ACTH preferred when visual pathway involvement suspected on MRI",
    },
    {
        "treatment": "Levetiracetam (LEV)",
        "level": "B",
        "indication": "Focal, GTCS, myoclonic seizures; broad-spectrum; IV formulation for SE; "
                      "all MSD types; no peripheral neuropathy risk",
        "mechanism": "SV2A vesicle protein modulation; reduces cortical myoclonus; "
                     "no sulfatase pathway conflict; renal excretion (hepatic not required)",
        "monitoring": "Behavioural (irritability 10–20%); renal dose adjustment; CBC annually; "
                      "behavioural monitoring quarterly",
        "caution": "Behavioural side-effects more pronounced in Type C with psychiatric features; "
                   "monitor closely; consider clobazam adjunct if irritability severe",
    },
    {
        "treatment": "Valproic Acid (VPA)",
        "level": "B",
        "indication": "GTCS + PME-like myoclonus (Types B + C); broad-spectrum; "
                      "NOT mitochondrial disease (SUMF1 = lysosomal), so VPA generally safe",
        "mechanism": "Sodium channel + GABA + T-type calcium; broad antiseizure; "
                     "also anti-myoclonic (PME pattern); lysosomal pathway no conflict",
        "monitoring": "LFT every 3 months (hepatomegaly from MPS component); "
                      "ammonia if encephalopathy; POLG1 EXCLUSION MANDATORY (CPIC Level A) "
                      "before initiation regardless of lysosomal diagnosis",
        "caution": "POLG1 exclusion mandatory as standard of care per CPIC-POLG1-2023 — "
                   "even though SUMF1 is NOT mitochondrial, POLG1 must be excluded before VPA; "
                   "LFT monitoring essential (hepatomegaly from MPS/ARSB component adds risk)",
    },
    {
        "treatment": "Piracetam",
        "level": "C",
        "indication": "PME-like cortical myoclonus adjunct (Types B + C); action myoclonus suppression",
        "mechanism": "AMPA receptor modulator; reduces cortical excitability and myoclonus threshold; "
                     "Level C for PME across multiple lysosomal diseases",
        "monitoring": "Renal function (primarily renally excreted); behavioural; dose titration",
        "caution": "Not indicated for IS or focal seizures; myoclonus-specific use only; "
                   "combined with LEV + VPA for refractory PME pattern",
    },
    {
        "treatment": "Clobazam",
        "level": "C",
        "indication": "Focal adjunct (Types B + C); IS adjunct after ACTH; absence-like (Type C)",
        "mechanism": "GABA-A 1,5-benzodiazepine allosteric modulator; less sedating than clonazepam; "
                     "slower tolerance development than clonazepam",
        "monitoring": "Sedation (leukodystrophy patients more sensitive); respiratory function; "
                      "tolerance after 3–6 months; cycle use strategies",
        "caution": "Tolerance develops; adjunct not monotherapy; "
                   "avoid in neonatal Type A (sedation worsens hypotonia)",
    },
    {
        "treatment": "Vigabatrin (VGB)",
        "level": "C",
        "indication": "IS second-line ONLY when visual pathway unaffected on MRI (rare in MSD); "
                      "NOT recommended as first-line in MSD due to visual cortex MLD lesions",
        "mechanism": "Irreversible GABA-T inhibitor; increases GABA; IS suppression",
        "monitoring": "Baseline + serial visual field testing (Goldmann perimetry); ERG; "
                      "ophthalmology every 6 months during VGB use",
        "caution": "HIGH RISK in Types A + B — visual cortex MLD lesions + VGB retinal toxicity additive; "
                   "irreversible visual field loss; ACTH preferred; VGB ONLY if ACTH fails AND MRI shows "
                   "visual cortex spared (very rare scenario)",
    },
    {
        "treatment": "Ketogenic Diet (KD)",
        "level": "B",
        "indication": "Refractory seizures (Types B + C); drug-resistant focal + myoclonic; "
                      "adjunct when ≥2 AEDs fail; Level B evidence for lysosomal disease epilepsies",
        "mechanism": "Ketosis → altered metabolic substrate → reduced excitability; "
                     "anti-inflammatory; does not conflict with sulfatase pathways",
        "monitoring": "Lipid panel (MPS component — hepatic involvement); growth parameters; "
                      "renal ultrasound (stones); selenium/zinc/carnitine levels",
        "caution": "Hepatomegaly (MPS/ARSB component) may limit tolerance; "
                   "coordinate with metabolic dietitian; gastrostomy may be needed in severe types",
    },
    {
        "treatment": "HSCT (Haematopoietic Stem Cell Transplantation)",
        "level": "C",
        "indication": "Experimental; very limited evidence; pre-symptomatic Type B only (rare NBS scenario); "
                      "NOT recommended for established neurological disease; "
                      "case reports only — no phase II/III data in MSD specifically",
        "mechanism": "Donor microglia cross-correction of sulfatase deficiency in CNS; "
                     "limited CNS penetration; peripheral organs (liver, spleen) better corrected",
        "monitoring": "Neurological monitoring; MRI annually post-HSCT; "
                      "sulfatase activities in peripheral blood (not CNS)",
        "caution": "NO established benefit in neurologically symptomatic MSD; "
                   "risk vs. benefit unfavourable in Types A + B with established CNS disease; "
                   "only consider pre-symptomatically in NBS-detected cases at a specialized centre",
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "risk": "HIGH RISK",
        "mechanism": "Na-channel blocker → worsens demyelinating peripheral neuropathy (MLD/ARSA component "
                     "of MSD) — same risk as isolated MLD; sodium channel blockade accelerates axonal degeneration "
                     "in pre-existing demyelinating neuropathy",
        "alternative": "LEV (SV2A, no neuropathy risk) or VPA (broad-spectrum) as first-line focal/GTCS agents",
        "evidence": "Extrapolated from MLD/ARSA evidence (highest-quality analogue); NICE NG217 (MLD); "
                    "MSD specialist consensus; peripheral neuropathy 90% in Types A + B — avoid CBZ/OXC",
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin (IV)",
        "risk": "HIGH RISK",
        "mechanism": "Na-channel blocker → worsens peripheral demyelinating neuropathy (same mechanism as CBZ); "
                     "IV fosphenytoin particularly risky in acute SE (use IV LEV instead)",
        "alternative": "IV LEV for SE; oral LEV or VPA for chronic management; avoid Na-channel blockers in "
                       "all MSD types with peripheral neuropathy (90% of Types A + B)",
        "evidence": "MLD/ARSA specialist consensus (Level III); NICE NG217; "
                    "MSD neuropathy mechanism identical to isolated ARSA/MLD neuropathy — same CI applies",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "risk": "HIGH RISK (Types A + B) / RELATIVE-CI (Type C)",
        "mechanism": "Irreversible GABA-T inhibition → irreversible visual field loss (nasal > temporal); "
                     "additive retinal toxicity when visual cortex MLD lesions present; "
                     "MRI visual cortex involvement increases VGB visual toxicity risk 3-fold",
        "alternative": "ACTH Level A for IS (preferred over VGB in MSD); "
                       "LEV/VPA for non-IS seizures in Types A + B",
        "evidence": "VGB visual toxicity: Maguire 2010, AAP 2010; MLD visual cortex lesions in 85% Type A/B; "
                    "combined risk: expert consensus (MSD specialist centres) — ACTH preferred",
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk": "HIGH RISK",
        "mechanism": "GABA reuptake inhibitor → paradoxical worsening of myoclonus (PME pattern); "
                     "non-convulsive SE risk in PME/myoclonic epilepsies; "
                     "MSD PME pattern identical to other progressive myoclonic epilepsies",
        "alternative": "Piracetam (Level C) for myoclonus; clobazam as focal adjunct; "
                       "LEV for combined focal + myoclonus",
        "evidence": "TGB in PME: paradoxical aggravation evidence (Genton 2000); "
                    "MSD myoclonus classified as PME-like pattern — same CI as other PME disorders",
    },
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, etc.)",
        "risk": "HIGH RISK",
        "mechanism": "D2 blockade → NMS risk + myoclonus exacerbation; "
                     "Type C psychiatric misdiagnosis (schizophrenia-like presentation 35%); "
                     "antipsychotic-induced parkinsonism worsened by basal ganglia MPS/MLD involvement",
        "alternative": "Atypical antipsychotics (quetiapine — lowest EPS risk) ONLY if psychiatric features "
                       "require treatment; preferably avoid all antipsychotics; treat psychiatric features "
                       "with neurology + psychiatry co-management; LEV behavioural side-effects ≠ psychosis",
        "evidence": "Type C MSD psychiatric misdiagnosis: Austin 1973, Bhatt 2012; "
                    "NMS risk in movement disorder + antipsychotics: Strawn 2007; expert consensus",
    },
    {
        "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
        "risk": "CAUTION / RELATIVE-CI",
        "mechanism": "Voltage-gated calcium channel α2δ ligand → may worsen ataxia and hypotonia in "
                     "MSD (MLD leukodystrophy + MPS hypotonia combined); "
                     "limited efficacy for myoclonic seizures in PME-like pattern",
        "alternative": "LEV or VPA preferred for focal/generalised seizures; "
                       "piracetam preferred for myoclonus; clobazam as adjunct",
        "evidence": "GBP/PGB in leukodystrophy-associated epilepsy: limited evidence; "
                    "hypotonia + ataxia worsening: case reports in MLD-like phenotypes; "
                    "relative contraindication based on MSD neurological profile",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "risk": "RELATIVE-CI (myoclonic seizures)",
        "mechanism": "Na-channel blocker → may paradoxically worsen myoclonus in PME pattern; "
                     "also mild peripheral neuropathy risk (Na-channel component, less than CBZ); "
                     "useful for focal/GTCS but avoid if myoclonus predominant",
        "alternative": "VPA for GTCS + myoclonus combined; LEV for focal; "
                       "piracetam for myoclonus suppression; avoid LTG as monotherapy in Type B PME-pattern",
        "evidence": "LTG myoclonus aggravation: Genton 2000, Biraben 2000; "
                    "MSD myoclonus classified PME-like — same caution as other PME (JME, MERRF analogues)",
    },
]

THRESHOLDS = [
    {"name": "ARSA activity (MSD Type A/B vs. MLD)", "value": "<10% control", "significance": "In MSD: ARSA <10% + ARSB <10% simultaneously; in isolated MLD: only ARSA low"},
    {"name": "ARSB activity (SUMF1 component)", "value": "<10% control", "significance": "ARSB low = MPS VI component of MSD; isolated ARSB low = isolated MPS VI (ARSB gene mutation)"},
    {"name": "IDS activity (SUMF1 component)", "value": "<10% control", "significance": "IDS low = MPS II component; isolated IDS low = isolated MPS II (Hunter); all three simultaneously = MSD"},
    {"name": "Urine sulfatides (MLD component of MSD)", "value": ">50× ULN", "significance": "Elevated like isolated MLD; confirms ARSA deficiency component within MSD"},
    {"name": "Urine GAGs — heparan sulfate (MPS component)", "value": ">3× ULN", "significance": "Elevated in MSD (IDS + SGSH deficiency components); NOT elevated in isolated MLD — key differentiator"},
    {"name": "Urine GAGs — dermatan sulfate (MPS component)", "value": ">2× ULN", "significance": "Elevated in MSD (ARSB/MPS VI component); NOT elevated in isolated MLD"},
    {"name": "FGE (SUMF1) residual activity — Type A", "value": "<1% control", "significance": "All sulfatases undetectable; neonatal lethal phenotype; biallelic null variants"},
    {"name": "FGE (SUMF1) residual activity — Type B", "value": "1–5% control", "significance": "Severe infantile phenotype; ARSA + ARSB + IDS severely reduced"},
    {"name": "FGE (SUMF1) residual activity — Type C", "value": "5–15% control", "significance": "Attenuated phenotype; ichthyosis prominent; slower neurological decline"},
    {"name": "Number of sulfatases reduced (≥3 = MSD)", "value": "≥3 sulfatases <30% control", "significance": "Diagnostic criterion for MSD: ≥3 sulfatase activities simultaneously low; 1 = isolated; 2 = check SUMF1"},
    {"name": "Plasma lyso-sulfatides (ARSA component)", "value": ">10× ULN", "significance": "Elevated in MSD; correlates with MLD severity component; useful for treatment monitoring"},
    {"name": "Mean diagnosis delay (MSD)", "value": "3.8 years", "significance": "Long diagnostic odyssey; sequential enzyme testing (one at a time) delays MSD recognition — mandate simultaneous panel"},
]

STANDARDS = [
    "Austin J (1973) Studies in metachromatic leukodystrophy XII Multiple sulfatase deficiency. Arch Neurol 28:258-64",
    "Bhatt MH et al. (2012) Late-onset multiple sulfatase deficiency presenting as psychiatric disorder. BMJ Case Rep",
    "Schlotawa L et al. (2011) SUMF1 mutations affecting stability and activity of formylglycine generating enzyme predict clinical outcome in multiple sulfatase deficiency. Eur J Hum Genet",
    "Dierks T et al. (2009) Molecular basis for multiple sulfatase deficiency and mechanism for formylglycine generation of the human formylglycine-generating enzyme. Cell",
    "Diez-Roux G & Bhatt P (2005) Multiple sulfatase deficiency. Orphanet J Rare Dis",
    "ILAE 2022 — Classification of Infantile Spasms; ACNS 2022 IS Treatment Guidelines",
    "NICE NG217 (2022) — Metachromatic Leukodystrophy: Component guidance applicable to MSD MLD-phenotype",
    "CPIC-POLG1 (2023) — Valproate and POLG1: Mandatory exclusion guideline regardless of diagnosis",
    "Maguire AM et al. (2010) — Vigabatrin retinal toxicity: irreversible visual field loss mechanism",
    "Genton P et al. (2000) — Aggravation of seizures by antiepileptic drugs in PME",
    "Strawn JR et al. (2007) — NMS risk in movement disorders + antipsychotics",
    "ACMG/AMP Variant Interpretation Standards (2015) — SUMF1 pathogenicity criteria",
]

KEY_CONCEPTS = [
    {
        "term": "Formylglycine-Generating Enzyme (FGE / SUMF1 protein)",
        "definition": "374-aa ~42 kDa ER-resident enzyme encoded by SUMF1 (3p26.1). FGE oxidises the cysteine "
                      "residue at position Cys-X-Pro-Ser/Thr (the sulfatase signature sequence) in ALL "
                      "17 cellular sulfatases, converting Cys → formylglycine. Formylglycine is the "
                      "catalytic residue required for sulfatase hydrolytic activity. Without FGE, ALL "
                      "sulfatases lack catalytic activity despite normal protein expression."
    },
    {
        "term": "Formylglycine (FGly) — catalytic residue of all sulfatases",
        "definition": "Unique post-translational modification: FGE oxidises Cys → formylglycine (2-oxoalanine) "
                      "in the active site of all sulfatases. FGly hydrates to geminal diol, which attacks the "
                      "sulfate ester substrate. SUMF1 LOF → no FGly in any sulfatase → all 17 sulfatases "
                      "enzymatically inactive. No other enzyme can substitute for FGE."
    },
    {
        "term": "The 17 sulfatases inactivated in MSD",
        "definition": "ARSA (MLD); ARSB (MPS VI); ARSC/STS (X-linked ichthyosis); ARSD; ARSE (chondrodysplasia "
                      "punctata X-linked); ARSF; ARSG; ARSH; ARSI; ARSJ; GNS (MPS III D); GALNS (MPS IV A); "
                      "HGSNAT (MPS III C); IDS (MPS II, Hunter); SGSH (MPS III A); SULF1; SULF2. "
                      "All lose activity when SUMF1 is deficient."
    },
    {
        "term": "Combined Sulfatiduria + GAGuria — PATHOGNOMONIC",
        "definition": "Urine biochemistry showing simultaneously elevated sulfatides (ARSA/MLD component) AND "
                      "elevated GAGs (heparan + dermatan + keratan sulfate — MPS components). No isolated "
                      "sulfatase deficiency causes both urine abnormalities simultaneously. This combined "
                      "urine pattern has >99% positive predictive value for MSD (SUMF1 mutation)."
    },
    {
        "term": "Ichthyosis in MSD (STS/ARSC deficiency component)",
        "definition": "Dry, scaly skin (X-linked ichthyosis pattern) due to simultaneous deficiency of steroid "
                      "sulfatase (STS/ARSC) within MSD. In isolated X-linked ichthyosis (STS gene mutation), "
                      "only STS is low. In MSD, STS + ARSA + all others simultaneously low. Ichthyosis + "
                      "any neurological feature → test full sulfatase panel."
    },
    {
        "term": "Diagnostic mandate — simultaneous multi-sulfatase enzyme panel",
        "definition": "Standard practice in individual LSD workup tests one enzyme at a time. This causes "
                      "3.8-year mean diagnostic delay in MSD (ARSA may be tested first, revealing MLD; "
                      "MPS and ichthyosis features then seem 'atypical' for MLD). MSD is only diagnosed "
                      "when ARSA + ARSB + IDS (minimum 3) are all tested simultaneously and found low. "
                      "Mandate: any child with leukodystrophy + hepatomegaly or leukodystrophy + ichthyosis → "
                      "simultaneous multi-sulfatase panel (not sequential single-enzyme testing)."
    },
    {
        "term": "SUMF1 allele dosage effect",
        "definition": "MSD severity is determined by residual FGE activity. Two null alleles → Type A (<1% activity). "
                      "Missense/null compound → Type B or C depending on residual activity of missense allele. "
                      "In vitro FGE activity assay (fibroblasts or leukocytes) plus SUMF1 sequencing "
                      "determines both subtype classification and prognosis."
    },
    {
        "term": "No approved ERT for MSD (critical therapeutic gap)",
        "definition": "Unlike isolated MLD (Arsa-cel/Lenmeldy approved EMA 2020, FDA 2024) or isolated MPS VI "
                      "(galsulfase/Naglazyme approved), no enzyme replacement therapy exists for MSD. "
                      "ERT of a single sulfatase would partially treat only that component. Multi-enzyme "
                      "ERT would require simultaneous replacement of all 17 sulfatases — not feasible. "
                      "SUMF1 gene therapy (restoring FGE to correct all sulfatases simultaneously) is "
                      "the only plausible single-intervention approach and is under preclinical development."
    },
    {
        "term": "Psychiatric misdiagnosis in adult-onset Type C MSD",
        "definition": "Type C MSD with late-onset (>10 years) may present with psychiatric features "
                      "(psychosis, behavioral changes — 35%) before overt neurological signs. "
                      "Standard psychiatric workup does not include sulfatase enzyme panel. "
                      "MSD diagnosis missed for 5–12 years. Red flags: psychiatric features + "
                      "ichthyosis + any subtle learning disability → check ALL sulfatase activities + "
                      "urine GAGs + SUMF1 sequencing before psychiatric diagnosis finalized."
    },
    {
        "term": "MSD vs. isolated MLD (ARSA) — key distinguishing features",
        "definition": "Isolated MLD: only ARSA low; urine sulfatides elevated; NO GAGs; NO ichthyosis; "
                      "NO dysostosis multiplex; ARSA gene mutation found. "
                      "MSD: ARSA + ARSB + IDS (all) low; urine sulfatides + GAGs elevated; "
                      "ichthyosis 60–85%; dysostosis multiplex on X-ray; SUMF1 mutation found. "
                      "Critical: MSD initially looks like 'atypical MLD' until GAGs and full sulfatase "
                      "panel reveal the combined phenotype."
    },
    {
        "term": "GENE THERAPY target (SUMF1) — corrects all 17 sulfatases simultaneously",
        "definition": "Replacing SUMF1/FGE is uniquely powerful: a single gene corrects ALL 17 sulfatases "
                      "simultaneously, unlike multi-ERT approaches. AAV9-SUMF1 CNS delivery is under "
                      "preclinical development. FGE protein is naturally secreted (ER → extracellular space), "
                      "enabling cross-correction of neighbouring cells — broader therapeutic spread than "
                      "non-secreted enzymes. Proof-of-concept in SUMF1 knockout mice (Di Natale 2010)."
    },
    {
        "term": "NBS (Newborn Screening) — not yet standard for MSD",
        "definition": "No approved NBS assay for MSD specifically. DBS multiplex enzyme assay panels "
                      "include ARSA (for MLD NBS), but do not routinely measure multiple sulfatases "
                      "simultaneously. SUMF1 NBS would require DBS FGE enzyme assay or SUMF1 gene "
                      "panel. Early pre-symptomatic detection is the only window for HSCT consideration "
                      "(Type B, very limited evidence). Currently identified incidentally when MLD NBS "
                      "reveals multiple enzyme abnormalities."
    },
]

DIAGNOSTIC_ALGORITHM = [
    "Step 1 — Suspect MSD: Leukodystrophy (MRI tigroid/leopard-skin) + ANY of: hepatosplenomegaly / "
     "dysostosis multiplex / ichthyosis / elevated urine GAGs / elevated urine sulfatides → investigate MSD",
    "Step 2 — Simultaneous multi-sulfatase enzyme panel (leukocytes or fibroblasts): measure ARSA, ARSB, "
     "IDS, GNS, GALNS, SGSH simultaneously — if ≥3 low, MSD confirmed biochemically",
    "Step 3 — Urine biochemistry: urine sulfatides (ARSA component) + urine GAGs (heparan/dermatan/keratan "
     "sulfate) simultaneously; combined elevation is PATHOGNOMONIC for MSD",
    "Step 4 — Ichthyosis evaluation: examine skin for X-linked ichthyosis pattern; "
     "skin biopsy if diagnosis in doubt; STS activity in fibroblasts to confirm STS component",
    "Step 5 — SUMF1 gene sequencing (3p26.1): biallelic variants confirm MSD; "
     "in vitro FGE enzyme activity (fibroblasts) to determine subtype (Type A/B/C) by residual activity",
    "Step 6 — POLG1 exclusion: MANDATORY before VPA initiation (CPIC-POLG1-2023 Level A); "
     "even though SUMF1 is not mitochondrial, POLG1 exclusion is standard of care",
    "Step 7 — Multidisciplinary plan: Neurology (seizure management — avoid CBZ/OXC/PHT/VGB/TGB) + "
     "Metabolic Medicine (sulfatase/GAG monitoring) + Ophthalmology (if VGB considered) + "
     "Neurodevelopment + Genetics + Physiotherapy (neuropathy management)",
]

GLOSSARY = {
    "SUMF1": "Sulfatase Modifying Factor 1 — the gene (3p26.1) encoding FGE; biallelic LOF = Multiple Sulfatase Deficiency",
    "FGE": "Formylglycine-Generating Enzyme — the SUMF1 protein product; oxidises Cys→FGly in all 17 sulfatases",
    "Formylglycine (FGly)": "Post-translational modification (Cys→FGly) required for catalytic activity of all sulfatases; absent in MSD",
    "MSD": "Multiple Sulfatase Deficiency — all 17 sulfatases inactive due to SUMF1/FGE deficiency",
    "Austin Disease": "Eponym for MSD (after J. Austin who first described the combined phenotype, 1973)",
    "Sulfatiduria": "Elevated urine sulfatides — reflects ARSA/MLD component of MSD; also seen in isolated MLD",
    "GAGuria": "Elevated urine glycosaminoglycans (heparan + dermatan + keratan sulfate) — MPS component of MSD; NOT seen in isolated MLD",
    "Dysostosis Multiplex": "Skeletal dysplasia on X-ray (beaked vertebrae, J-shaped sella, shortened/widened tubular bones) — MPS component of MSD",
    "Ichthyosis (MSD)": "Dry scaly skin from STS/ARSC deficiency component; X-linked ichthyosis pattern; PATHOGNOMONIC for Type C MSD",
    "Tigroid MRI": "White matter tigroid / leopard-skin pattern — MLD/ARSA leukodystrophy component in MSD; periventricular sparing",
    "FGE cross-correction": "FGE protein is secreted extracellularly; neighbouring cells can take up secreted FGE → correct multiple sulfatases; basis for gene therapy potential",
    "Type A MSD": "FGE activity <1%; neonatal lethal; all sulfatases undetectable; biallelic null SUMF1",
    "Type B MSD": "FGE activity 1–5%; severe infantile; seizures 78%; survival early childhood",
    "Type C MSD": "FGE activity 5–15%; attenuated; ichthyosis prominent; psychiatric features; slower progression",
}

# ---------------------------------------------------------------------------
# DATA-LAYER FUNCTIONS
# ---------------------------------------------------------------------------

def get_overview():
    """Return top-level clinical overview dict."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "disease": (
            "SUMF1 (Sulfatase Modifying Factor 1 / Formylglycine-Generating Enzyme FGE) encodes the single "
            "post-translational activating enzyme for ALL 17 cellular sulfatases. Biallelic SUMF1 LOF → "
            "ALL sulfatases simultaneously inactive → Multiple Sulfatase Deficiency (MSD / Austin Disease, "
            "OMIM 272200). MSD combines the phenotypes of MLD (ARSA), MPS II (IDS), MPS VI (ARSB), "
            "X-linked ichthyosis (STS), and MPS III (SGSH/GNS) in one patient. Locus 3p26.1. "
            "4 clinical subtypes by residual FGE activity: Type A (neonatal lethal <1% FGE), "
            "Type B (severe infantile 1–5% FGE), Type C (attenuated 5–15% FGE, ichthyosis prominent), "
            "Attenuated (juvenile/adult >15% FGE). No approved ERT. HSCT experimental, pre-symptomatic only. "
            "SUMF1 gene therapy (AAV9) is the logical single-gene-corrects-all approach under preclinical "
            "investigation. 40-patient cohort."
        ),
        "protein": (
            "FGE (Formylglycine-Generating Enzyme) — 374 amino acids, ~42 kDa, ER-resident; "
            "oxidises Cys → formylglycine (FGly) in the sulfatase signature motif Cys-X-Pro-Ser/Thr; "
            "FGly is the catalytic residue required for sulfate ester hydrolysis by all sulfatases; "
            "FGE is also secreted extracellularly, enabling cross-correction of neighbouring cells"
        ),
        "mechanism": (
            "SUMF1 LOF → absence of FGly modification in all 17 sulfatase active sites → "
            "all sulfatases present as inactive apoenzymes → simultaneous accumulation of: "
            "sulfatides (ARSA substrate), heparan/dermatan/keratan sulfate (IDS/ARSB/SGSH substrates), "
            "cholesterol-3-sulfate (STS substrate → ichthyosis). "
            "Combined storage product burden exceeds any isolated sulfatase disease."
        ),
        "pathognomonic_note": (
            "ALL sulfatases simultaneously low (≥3 sulfatase activities <30% control on enzyme panel) "
            "is PATHOGNOMONIC for MSD — distinguishes from all isolated sulfatase deficiencies. "
            "Combined urine sulfatides (>50× ULN) + urine GAGs elevated (heparan + dermatan + keratan sulfate) "
            "is the second PATHOGNOMONIC biochemical finding; no isolated LSD produces both simultaneously. "
            "Ichthyosis + leukodystrophy = MSD (Type C) PATHOGNOMONIC combination."
        ),
        "diagnostic_hierarchy": (
            "1. ANY leukodystrophy + hepatosplenomegaly OR ichthyosis OR elevated urine GAGs → "
            "test simultaneous multi-sulfatase panel (not just ARSA alone). "
            "2. ≥3 sulfatases low → MSD confirmed → SUMF1 sequencing. "
            "3. ARSB + IDS simultaneously low distinguishes MSD from isolated MLD definitively. "
            "4. Urine: sulfatides + GAGs — combined elevation = MSD until proven otherwise. "
            "5. POLG1 exclusion before VPA (CPIC Level A)."
        ),
        "seizure_pct_overall": 65,
        "seizure_pct_type_a": 85,
        "seizure_pct_type_b": 78,
        "seizure_pct_type_c": 55,
        "seizure_pct_attenuated": 30,
        "drug_resistant_pct": 65,
        "infantile_spasms_pct": 45,
        "peripheral_neuropathy_pct": 78,
        "ichthyosis_pct": 60,
        "leukodystrophy_pct": 88,
        "dysostosis_multiplex_pct": 72,
        "hepatosplenomegaly_pct": 80,
        "psychiatric_features_pct": 25,
        "mean_onset_months": 14,
        "mean_diagnosis_delay_years": 3.8,
        "all_sulfatases_low_pct": 100,
        "on_acth_pct": 38,
        "on_lev_pct": 72,
        "on_vpa_pct": 55,
        "on_kd_pct": 18,
        "standards": STANDARDS,
    }


def get_breakdown():
    """Return detailed breakdown: etiologies, seizure types, triggers, treatments, CIs, thresholds."""
    return {
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "treatment_hierarchy": [
            "Step 1: ACTH (Level A) for IS (Types A + B) — PREFERRED over VGB due to visual cortex MLD lesions",
            "Step 2: LEV (Level B) add-on for focal, GTCS, myoclonic (all types)",
            "Step 3: VPA (Level B) for GTCS + PME-like myoclonus — POLG1 exclusion MANDATORY first",
            "Step 4: Piracetam (Level C) for cortical myoclonus adjunct (Types B + C)",
            "Step 5: Clobazam (Level C) for focal adjunct; absence-like (Type C)",
            "Step 6: Ketogenic Diet (Level B) for refractory seizures (Types B + C) — ≥2 AED failures",
            "Step 7: VGB (Level C ONLY) — ONLY if ACTH fails AND MRI shows visual cortex spared (rare scenario)",
            "AVOID: CBZ/OXC/PHT (neuropathy — MLD component), TGB (myoclonus worsening), "
                   "Typical Antipsychotics (NMS + myoclonus), VGB in Types A/B (visual cortex lesions)",
        ],
        "sumf1_sulfatase_affected": {
            "ARSA": {"disease_component": "MLD (Metachromatic Leukodystrophy)", "substrate": "Sulfatides", "biomarker": "Urine sulfatides >50× ULN; plasma lyso-sulfatides", "epilepsy_link": "Leukodystrophy → seizures (PME, focal, IS)"},
            "ARSB": {"disease_component": "MPS VI (Maroteaux-Lamy)", "substrate": "Dermatan sulfate", "biomarker": "Urine dermatan sulfate >2× ULN", "epilepsy_link": "Dysostosis multiplex, skeletal involvement; CNS less prominent"},
            "IDS": {"disease_component": "MPS II (Hunter)", "substrate": "Heparan + dermatan sulfate", "biomarker": "Urine heparan + dermatan sulfate", "epilepsy_link": "Heparan sulfate CNS accumulation → neurodegeneration → seizures"},
            "STS/ARSC": {"disease_component": "X-linked ichthyosis", "substrate": "Cholesterol-3-sulfate", "biomarker": "Elevated cholesterol-3-sulfate; ichthyosis (clinical)", "epilepsy_link": "Not a direct epilepsy cause; ichthyosis hallmark of MSD Type C"},
            "SGSH": {"disease_component": "MPS III A (Sanfilippo A)", "substrate": "Heparan sulfate (reducing end)", "biomarker": "Urine heparan sulfate (heparan-dominant fractions)", "epilepsy_link": "Severe intellectual disability + behavioural + epilepsy in Sanfilippo component"},
            "GALNS": {"disease_component": "MPS IV A (Morquio A)", "substrate": "Keratan sulfate + chondroitin-6-sulfate", "biomarker": "Urine keratan sulfate elevated", "epilepsy_link": "Skeletal (cervical spine) → spinal cord compression risk; not primary epilepsy driver"},
        },
        "subtype_severity_matrix": {
            "Type A": "FGE <1% · all sulfatases undetectable · neonatal lethal · seizures 85% (IS+tonic)",
            "Type B": "FGE 1–5% · ARSA+ARSB+IDS severely low · severe infantile · seizures 78% · peripheral neuropathy 90%",
            "Type C": "FGE 5–15% · attenuated · ichthyosis 85% · slower progression · seizures 55% · psychiatric 35%",
            "Attenuated": "FGE >15% · rare missense · adult onset · ichthyosis main feature · seizures 30%",
        },
    }


def get_definitions():
    """Return definitions, diagnostic algorithm, glossary, key concepts, and standards."""
    return {
        "diagnostic_algorithm": DIAGNOSTIC_ALGORITHM,
        "sumf1_glossary": GLOSSARY,
        "key_concepts": KEY_CONCEPTS,
        "standards": STANDARDS,
        "differential_diagnosis": {
            "isolated_MLD_ARSA": (
                "ARSA only low; urine sulfatides only; NO GAGs; NO ichthyosis; NO dysostosis multiplex; "
                "ARSA gene mutation → distinguished by ARSB + IDS being NORMAL in isolated MLD"
            ),
            "isolated_MPS_II_IDS": (
                "IDS only low; urine heparan + dermatan sulfate; NO sulfatides; NO leukodystrophy "
                "(or mild); IDS gene mutation → distinguished by ARSA + ARSB being NORMAL"
            ),
            "isolated_MPS_VI_ARSB": (
                "ARSB only low; urine dermatan sulfate; NO sulfatides; NO leukodystrophy; "
                "ARSB gene mutation → distinguished by ARSA + IDS being NORMAL"
            ),
            "PSAP_SapB_deficiency": (
                "ARSA NORMAL (false negative) on standard assay; urine sulfatides elevated (like MLD); "
                "NO GAGs; NO ichthyosis; heat-inactivation ARSA assay reveals true deficiency; "
                "PSAP mutation → distinguished by ARSA being normal (not low as in MSD)"
            ),
            "NPC1_NPC2": (
                "NPC1/NPC2 — unesterified cholesterol storage; VSGP 93%; gelastic cataplexy; "
                "filipin staining; NO sulfatiduria; NO GAGuria; no sulfatase panel abnormalities"
            ),
        },
    }
