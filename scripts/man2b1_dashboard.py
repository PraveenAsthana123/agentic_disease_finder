"""
MAN2B1 Epilepsy — Alpha-Mannosidosis (Lysosomal Alpha-Mannosidase Deficiency)
===============================================================================
40-patient cohort · MAN2B1 (19p13.2) · Autosomal Recessive (AR) biallelic LOF
MAN2B1 encodes Lysosomal Acid Alpha-Mannosidase (LAMAN, 1011 aa, ~107 kDa precursor):
  LAMAN cleaves terminal α-mannosyl residues from N-linked glycoproteins in lysosomes.
  MAN2B1 LOF → LAMAN deficiency → accumulation of mannose-rich oligosaccharides
  (Man2–Man5-GlcNAc2 oligosaccharides) in lysosomes of all cell types, especially
  neurons, hepatocytes, Kupffer cells, kidney tubular cells, and leukocytes.

DISEASE — ALPHA-MANNOSIDOSIS (OMIM 248500):
  Pan-European founder effect (p.His72Tyr in Scandinavian patients ~24%);
  Incidence ~1:500,000 live births; >200 pathogenic MAN2B1 variants described.
  Progressive intellectual disability (100%); recurrent infections (85%);
  hearing loss (>90%); coarse facial features (80%); skeletal dysplasia (75%);
  ataxia (60%); psychiatric symptoms (30–50%); epilepsy (30–50%).
  Pan-ethnic (Scandinavian, Middle Eastern, and South Asian populations enriched).

PATHOGNOMONIC FEATURES:
  (1) URINE MANNOSE-RICH OLIGOSACCHARIDES (PATHOGNOMONIC — FIRST-LINE SCREEN):
      Quantitative urine TLC reveals Man2–Man5-GlcNAc2 oligosaccharides (retention
      time unique to alpha-mannosidosis). Sensitivity ~95%; specificity >99%.
      Fastest, cheapest, most widely available LSD screen — send urine from any lab.
  (2) VACUOLATED LYMPHOCYTES ON BLOOD SMEAR (PATHOGNOMONIC — BEDSIDE DIAGNOSIS):
      PAS-positive, large, clear cytoplasmic vacuoles in lymphocytes (virtually 100%
      of patients). Detectable on routine peripheral blood film. Fastest bedside clue.
      Distinguishes from most LSD phenocopies (vacuolation absent in GM2, NCL, NPC).
  (3) LEUKOCYTE LAMAN ENZYME ACTIVITY <10% CONTROL (CONFIRMATORY):
      Leukocyte (or fibroblast) alpha-mannosidase assay using 4-methylumbelliferyl-
      alpha-D-mannoside substrate; <10% residual = diagnostic; DBS assay less reliable.
  (4) MAN2B1 BIALLELIC VARIANTS (MOLECULAR CONFIRMATION):
      WES/WGS or targeted panel; >200 pathogenic variants described; p.His72Tyr
      founder variant (Scandinavian); genotype–phenotype correlation weak.

EPILEPSY — MAN2B1-SPECIFIC:
  Prevalence: 30–50% lifetime; onset typically adolescence–adulthood (later than
  most other LSDs); milder overall seizure burden than sphingolipidoses.
  Seizure types: GTCS (70%), myoclonic (30%), focal (25%), absence-like (15%).
  Drug resistance: 30% (lower than most other LSD epilepsies).
  Precipitants: Febrile illness (infections prominent due to immunodeficiency),
  sleep deprivation, missed AED; psychiatric medication changes.
  EEG: Generalised spike-wave; may show theta slowing (encephalopathy component);
  photosensitivity uncommon (unlike NCL/sphingolipidoses).

ENZYME REPLACEMENT THERAPY (ERT):
  Velmanase alfa (Lamzede, Chiesi): recombinant human LAMAN (CHO cell expression);
  M6P receptor-mediated lysosomal targeting; dose 1 mg/kg IV weekly.
  EMA approved 2018 (EU/UK); FDA approved July 2023 (USA) — FIRST and ONLY ERT.
  CNS penetration: LIMITED (mannose-6-phosphate receptor deficiency on CNS endothelium
  limits BBB penetration); ERT primarily addresses somatic manifestations (infections,
  functional ability, hepatosplenomegaly, urinary oligosaccharides). Neurological
  improvement is modest (attenuates progression, does not reverse established damage).

DRUG SAFETY (AED-SPECIFIC):
  CBZ/OXC: GENERALLY SAFE — alpha-mannosidosis lacks demyelinating peripheral
    neuropathy (unlike MLD/ARSA/GALC where CBZ/OXC are ABSOLUTE CIs). Standard
    monitoring applies (hyponatraemia, HLA-B*15:02 in SE Asian).
  VPA: SAFE (lysosomal, not mitochondrial pathway); POLG1 exclusion MANDATORY
    (CPIC Level A) as standard of care; hepatic monitoring (hepatomegaly uncommon).
  VGB: NOT A SPECIFIC CI — no retinal NCL, no visual cortex lesion component
    (unlike NCL diseases where VGB is ABSOLUTE CI); standard VGB visual monitoring.
  Typical Antipsychotics: CAUTION — psychiatric misdiagnosis 30–50%; extrapyramidal
    side effects (EPS) worsen pre-existing ataxia; QTc monitoring required.
  Benzodiazepines: standard anxiolytic/seizure-abort use; respiratory caution in
    severe forms with reduced respiratory reserve.
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

GENE        = "MAN2B1 (Lysosomal Acid Alpha-Mannosidase / LAMAN)"
LOCUS       = "19p13.2"
OMIM        = "609458 (MAN2B1 gene); 248500 (Alpha-Mannosidosis)"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF; genotype–phenotype correlation weak"
COHORT_SIZE = 40

ETIOLOGIES = [
    {
        "name": "Compound Heterozygous — Missense + Missense",
        "pct": 35,
        "onset": "Childhood / early adolescence (3–10 years)",
        "notes": (
            "Two different missense alleles; most common genotype in non-consanguineous populations; "
            "residual LAMAN activity 1–5%; moderate phenotype; intellectual disability moderate (IQ 40–70); "
            "hearing loss >90%; ataxia onset mid-childhood; seizures 40%; psychiatric features 35%; "
            "Scandinavian founder p.His72Tyr often in compound het configurations"
        ),
        "key_finding": "Two missense MAN2B1 variants; leukocyte LAMAN 1–5% control; urine oligosaccharides moderate",
    },
    {
        "name": "Homozygous Missense (Consanguineous / Founder)",
        "pct": 28,
        "onset": "Infantile / early childhood (1–5 years)",
        "notes": (
            "Consanguineous families (Middle Eastern, South Asian, Turkish, Norwegian) with founder variants; "
            "p.His72Tyr homozygous in Scandinavian patients (~24% of all European alleles); "
            "moderate-to-severe phenotype; LAMAN activity near-zero to 2%; "
            "recurrent infections prominent from infancy (B-cell dysfunction, IgG class-switching defect); "
            "seizures 50%; coarse facial features prominent"
        ),
        "key_finding": "Homozygous MAN2B1 missense; Middle Eastern/Scandinavian ancestry; vacuolated lymphocytes 100%",
    },
    {
        "name": "Compound Heterozygous — Null + Missense",
        "pct": 20,
        "onset": "Infantile (severe) / childhood (moderate) — depends on missense severity",
        "notes": (
            "One truncating allele (frameshift, nonsense, splice-site) + one missense allele; "
            "severity driven by null allele (dominant loss-of-function effect); "
            "phenotype ranges from severe infantile (null-dominant) to moderate childhood; "
            "LAMAN activity near-zero from null allele; seizures 55%; drug-resistant epilepsy 40%; "
            "immunodeficiency prominent; earlier ERT/HSCT consideration"
        ),
        "key_finding": "Null + missense compound het; null allele dominant severity; LAMAN ~0%; consider early HSCT",
    },
    {
        "name": "Homozygous Null / Biallelic Truncating",
        "pct": 12,
        "onset": "Severe infantile (birth – 2 years)",
        "notes": (
            "Biallelic frameshift, nonsense, or splice-site variants; LAMAN activity undetectable (<1%); "
            "severe intellectual disability (IQ typically <40); severe organomegaly; "
            "frequent bacterial infections from infancy (encapsulated organisms: Streptococcus pneumoniae, "
            "Neisseria meningitidis); severe ataxia; seizures 60%; non-ambulatory by adolescence; "
            "HSCT most urgently indicated (pre-symptomatic detection via NBS or family cascade)"
        ),
        "key_finding": "Biallelic null; LAMAN undetectable; severe infantile LSD; earliest HSCT candidacy",
    },
    {
        "name": "Attenuated / Late-Onset (Mild Type I Phenotype)",
        "pct": 5,
        "onset": "Adolescence / adulthood (10–30 years)",
        "notes": (
            "Missense variants preserving 5–15% residual LAMAN activity; mild intellectual disability "
            "(borderline IQ 60–80); hearing loss present; psychiatric features prominent (40% — first "
            "presentation as psychosis, depression, or behavioral disorder in teenage years); "
            "seizures less frequent (20%); employment possible with support; "
            "most vulnerable to psychiatric misdiagnosis and delayed MAN2B1 diagnosis"
        ),
        "key_finding": "Residual LAMAN 5–15%; adult psychiatric presentation; urine oligosaccharides may be mild",
    },
]

SEIZURE_TYPES = [
    {
        "type": "Generalised Tonic-Clonic (GTCS)",
        "pct": 70,
        "subtype": (
            "Most common seizure type; onset adolescence–adulthood in most; well-controlled with VPA/LEV; "
            "fever-triggered flares common (immunodeficiency → frequent infections); "
            "EEG: generalised spike-wave or polyspike-wave; background theta slowing reflects encephalopathy"
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 30,
        "subtype": (
            "Action myoclonus (cortical) in moderate-severe phenotypes; "
            "PME-like pattern in <10% (unlike sphingolipid PMEs); "
            "LEV and VPA most effective; piracetam Level C adjunct; "
            "ataxia + myoclonus combination may mimic cerebellar PME"
        ),
    },
    {
        "type": "Focal Seizures (with/without impaired awareness)",
        "pct": 25,
        "subtype": (
            "Temporal and frontal origin; oligosaccharide accumulation in hippocampal interneurons; "
            "CBZ/OXC generally SAFE in alpha-mannosidosis (no demyelinating neuropathy); "
            "LEV first-line alternative; evolve to bilateral tonic-clonic in 60%"
        ),
    },
    {
        "type": "Absence-like (Atypical Absence)",
        "pct": 15,
        "subtype": (
            "Atypical absence; slow spike-wave (<3 Hz); cognitive fluctuations; "
            "clobazam adjunct; may be difficult to distinguish from behavioral fluctuations "
            "of intellectual disability without EEG confirmation"
        ),
    },
    {
        "type": "Febrile Seizures / Infection-Triggered Clusters",
        "pct": 45,
        "subtype": (
            "Febrile seizures unusually common due to immunodeficiency (recurrent pneumonia, CMV, EBV); "
            "infection → oligosaccharide accumulation acutely worsens → lowered seizure threshold; "
            "rescue plan essential; IV immunoglobulin (IVIG) for immunodeficiency may reduce infective triggers"
        ),
    },
]

TRIGGERS = [
    {
        "trigger": "Febrile illness / bacterial/viral infection",
        "pct": 82,
        "notes": (
            "MOST POTENT TRIGGER — unique to alpha-mannosidosis: immunodeficiency (B-cell dysfunction, "
            "defective IgG class-switching due to oligosaccharide accumulation in B-cells) → "
            "recurrent respiratory tract infections (Streptococcus pneumoniae, H. influenzae), "
            "sinusitis, otitis media, CMV/EBV reactivation → fever → lowered seizure threshold; "
            "velmanase alfa reduces infective episodes 30%; IVIG for severe immunodeficiency"
        ),
    },
    {
        "trigger": "Sleep deprivation",
        "pct": 65,
        "notes": (
            "Sleep disruption from recurrent ear infections (hearing loss + otitis media); "
            "cortical excitability threshold lowered; action myoclonus worsened; "
            "behavioral sleep problems common in intellectual disability + ataxia"
        ),
    },
    {
        "trigger": "Missed AED dose",
        "pct": 60,
        "notes": (
            "Intellectual disability + complex medication regimens (AED + velmanase alfa infusion + "
            "IVIG + supplements) → non-adherence risk; seizure clusters on dose omission; "
            "simplest possible regimen preferred; pill dispensers; carer involvement"
        ),
    },
    {
        "trigger": "Psychiatric medication changes",
        "pct": 40,
        "notes": (
            "30–50% of patients receive antipsychotics for psychiatric features (psychosis, behavioral); "
            "initiation, dose change, or cessation of antipsychotics (especially typical antipsychotics) → "
            "seizure threshold lowering; atypical antipsychotics preferred (lower seizure risk); "
            "EPS from typical antipsychotics worsens ataxia — seizure falls injury risk"
        ),
    },
    {
        "trigger": "Physical exertion / fatigue",
        "pct": 35,
        "notes": (
            "Action myoclonus triggered by voluntary movement and fatigue; ataxia compounds fall risk; "
            "occupational therapy and physiotherapy essential; activity modification without restriction"
        ),
    },
    {
        "trigger": "Dehydration / metabolic stress",
        "pct": 30,
        "notes": (
            "Diarrhoea + vomiting from recurrent infections → dehydration → electrolyte disturbance → "
            "seizure threshold lowering; CBZ/OXC hyponatraemia risk on background dehydration; "
            "emergency protocol: oral rehydration first; hospital for IV fluid if severe"
        ),
    },
    {
        "trigger": "Photosensitivity",
        "pct": 10,
        "notes": (
            "Photosensitivity UNCOMMON in alpha-mannosidosis (unlike NCL, sphingolipidoses where >70%); "
            "if present, check EEG for photo-paroxysmal response; standard VPA photosensitivity protocol; "
            "absence of photosensitivity is a distinguishing feature from NCL/sphingolipid LSD"
        ),
    },
    {
        "trigger": "Contraindicated drug exposure",
        "pct": 25,
        "notes": (
            "Typical antipsychotics (seizure threshold lowering + EPS worsening ataxia); "
            "VGB (standard retinal monitoring applies — not CI in alpha-mannosidosis but requires monitoring); "
            "NOTE: CBZ/OXC NOT contraindicated (no demyelinating neuropathy unlike MLD/GALC/ARSA)"
        ),
    },
]

TREATMENTS = [
    {
        "treatment": "Levetiracetam (LEV)",
        "level": "B",
        "indication": (
            "GTCS, focal, myoclonic seizures; broad-spectrum; IV formulation for SE; all alpha-mannosidosis "
            "phenotypes; no pathway conflict; preferred first-line in severe intellectual disability (simplest "
            "dosing, no TDM required routinely)"
        ),
        "mechanism": (
            "SV2A synaptic vesicle protein modulation; reduces cortical myoclonus and generalised bursts; "
            "renal excretion (hepatic metabolism not required — safe in hepatomegaly if present)"
        ),
        "monitoring": (
            "Behavioural (irritability 10–20%; more pronounced in intellectual disability + psychiatric co-morbidity); "
            "renal dose adjustment; CBC annually; psychiatric monitoring quarterly in at-risk patients"
        ),
        "caution": (
            "Behavioural side-effects more pronounced in patients with pre-existing behavioral challenges; "
            "consider clobazam co-prescription if severe behavioural dysregulation with LEV"
        ),
    },
    {
        "treatment": "Valproic Acid (VPA)",
        "level": "B",
        "indication": (
            "GTCS + myoclonic seizures; broad-spectrum; POLG1 exclusion MANDATORY before initiation "
            "(CPIC Level A); safe in lysosomal (non-mitochondrial) disease with appropriate monitoring"
        ),
        "mechanism": (
            "Sodium channel + GABA augmentation + T-type calcium channel; anti-myoclonic; "
            "broad antiseizure spectrum; lysosomal oligosaccharide pathway — no mechanistic conflict with VPA"
        ),
        "monitoring": (
            "LFT every 3 months (hepatomegaly uncommon in alpha-mannosidosis but monitor); "
            "ammonia if encephalopathy; POLG1 exclusion MANDATORY (CPIC-POLG1-2023 Level A standard of care); "
            "VPPP monitoring in females ≥12 years (teratogenicity, polycystic ovary syndrome)"
        ),
        "caution": (
            "POLG1 exclusion before VPA — mandatory even though MAN2B1 is lysosomal (not mitochondrial); "
            "MERRF (PME + mitochondrial, VPA CI) is clinical phenocopy; "
            "VPPP compliance essential in females of reproductive age"
        ),
    },
    {
        "treatment": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "level": "B",
        "indication": (
            "Focal seizures; GTCS — SAFE in alpha-mannosidosis (unlike MLD/GALC/ARSA where ABSOLUTE CI); "
            "alpha-mannosidosis lacks demyelinating peripheral neuropathy; standard focal seizure choice"
        ),
        "mechanism": (
            "Sodium channel stabilisation; effective for focal onset and generalised tonic-clonic; "
            "CBZ-XR preferred (steadier levels, fewer side effects)"
        ),
        "monitoring": (
            "HLA-B*15:02 MANDATORY before CBZ/OXC in SE Asian ancestry (SJS/TEN risk); "
            "sodium (SIADH — monitor electrolytes particularly in dehydration during infections); "
            "CBZ-TDM 4–12 mg/L; CYP3A4 drug interactions (azole antifungals used in recurrent infections)"
        ),
        "caution": (
            "HLA-B*15:02 testing MANDATORY in SE Asian patients — SJS/TEN fatal risk; "
            "SIADH risk on background of dehydration from recurrent infections and diarrhoea; "
            "CYP3A4 autoinducer (reduces own levels + reduces itraconazole, fluconazole if used for infections); "
            "KEY DISTINCTION: CBZ/OXC SAFE in alpha-mannosidosis — NO demyelinating neuropathy unlike MLD/ARSA/GALC"
        ),
    },
    {
        "treatment": "Velmanase alfa (Lamzede) — Enzyme Replacement Therapy",
        "level": "A",
        "indication": (
            "ALL confirmed alpha-mannosidosis patients (FDA 2023 / EMA 2018); 1 mg/kg IV weekly; "
            "reduces infections, improves functional status, normalises urine oligosaccharides; "
            "modest CNS benefit (attenuates progression, limited BBB penetration)"
        ),
        "mechanism": (
            "Recombinant human LAMAN (CHO cell expression, M6P-tagged); mannose-6-phosphate receptor "
            "(M6PR)-mediated lysosomal targeting in peripheral tissues; "
            "cleaves mannose-rich oligosaccharides in lysosomes; CNS penetration LIMITED by M6PR "
            "deficiency on CNS endothelium — somatic benefit predominates over neurological"
        ),
        "monitoring": (
            "Urine mannose-rich oligosaccharides (primary biomarker — normalize with treatment); "
            "infusion-related reactions (pre-medicate with antihistamine if prior reactions); "
            "anti-velmanase antibodies (neutralising — check if efficacy wanes); "
            "infection rate tracking (immunological endpoints)"
        ),
        "caution": (
            "CNS penetration LIMITED — seizures and neurological features may not improve with ERT alone; "
            "ERT + HSCT combination under investigation; ERT does NOT replace HSCT for CNS disease; "
            "weekly IV infusion — adherence requires specialist infusion centre; "
            "not a cure — disease progression continues (ERT = maintenance, not reversal)"
        ),
    },
    {
        "treatment": "HSCT (Haematopoietic Stem Cell Transplantation)",
        "level": "B",
        "indication": (
            "Severe / early-onset alpha-mannosidosis (biallelic null, homozygous null); "
            "BEST EVIDENCE among lysosomal diseases for cognitive stabilisation; "
            "ideally pre-symptomatic (NBS-detected) or early symptomatic; "
            "stabilises CNS disease better than ERT; immunodeficiency correction"
        ),
        "mechanism": (
            "Donor microglia replace host LAMAN-deficient microglia; cross-correction of CNS neurons "
            "by secreted LAMAN from donor macrophages; immunological reconstitution (B-cell function); "
            "CNS penetration superior to ERT (cells cross BBB); "
            "Reduces oligosaccharide accumulation in brain over 1–2 years post-HSCT"
        ),
        "monitoring": (
            "Post-HSCT: neurological assessments (cognitive, motor) 6-monthly; "
            "urine oligosaccharides (LAMAN activity in leukocytes post-HSCT); "
            "MRI annually (white matter, atrophy); "
            "AED continuation post-HSCT (seizures may persist despite stabilisation)"
        ),
        "caution": (
            "Transplant-related morbidity/mortality 5–15% — reserve for severe phenotypes; "
            "does NOT reverse established neurological damage — window of opportunity is pre-symptomatic; "
            "AED management continues post-HSCT; ERT can bridge pre-HSCT; "
            "ERT + HSCT combination may be complementary (ongoing trials)"
        ),
    },
    {
        "treatment": "Clobazam",
        "level": "C",
        "indication": (
            "Focal adjunct; absence-like seizures; rescue benzodiazepine for cluster seizures; "
            "bedtime dosing for nocturnal seizures; less sedating than clonazepam"
        ),
        "mechanism": (
            "GABA-A 1,5-benzodiazepine allosteric modulator; slower tolerance development than "
            "clonazepam; effective focal + myoclonic adjunct"
        ),
        "monitoring": "Sedation; tolerance (3–6 months); respiratory in severe phenotypes",
        "caution": "Tolerance develops; cyclic use strategies; avoid in severe respiratory insufficiency",
    },
    {
        "treatment": "IVIG (Intravenous Immunoglobulin)",
        "level": "B",
        "indication": (
            "Severe immunodeficiency with recurrent bacterial infections (Streptococcus pneumoniae, "
            "H. influenzae, meningococcus); IgG <4 g/L or >4 serious infections per year; "
            "INDIRECT seizure benefit: reduces infective trigger for seizure clusters"
        ),
        "mechanism": (
            "Passive IgG replacement; corrects B-cell class-switching defect from oligosaccharide "
            "accumulation in B-lymphocytes; reduces encapsulated bacterial infections; "
            "reduces febrile seizure triggers"
        ),
        "monitoring": "IgG trough levels (target >6 g/L); infusion reactions; renal function (IVIG nephropathy)",
        "caution": "Not standard for all patients — only severe immunodeficiency; IgA deficiency → anaphylaxis risk",
    },
    {
        "treatment": "Piracetam",
        "level": "C",
        "indication": "Cortical myoclonus adjunct (action myoclonus component); PME-like patterns if present",
        "mechanism": "AMPA receptor modulator; reduces cortical myoclonus threshold; Level C for LSD myoclonus",
        "monitoring": "Renal function (primarily renal excretion); behavioural; dose titration",
        "caution": "Myoclonus-specific; not effective for GTCS or focal; combine with LEV/VPA for refractory myoclonus",
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, Fluphenazine)",
        "risk": "CAUTION — HIGH RISK in alpha-mannosidosis",
        "mechanism": (
            "Extrapyramidal side effects (EPS) — particularly parkinsonism and akathisia — "
            "worsen pre-existing cerebellar ataxia (60% of patients); "
            "fall risk dramatically increased; D2 blockade lowers seizure threshold; "
            "NMS risk in patients with intellectual disability who cannot report prodromal symptoms; "
            "tardive dyskinesia may be misinterpreted as worsening ataxia"
        ),
        "alternative": (
            "Atypical antipsychotics preferred (quetiapine, aripiprazole — lower EPS risk); "
            "for behavioral: clonazepam, melatonin, SSRIs; "
            "psychiatry review before initiation; MAN2B1 diagnosis must be known to treating psychiatrist"
        ),
        "evidence": (
            "Intellectual disability + ataxia + EPS = severe falls; "
            "general LSD consensus (no alpha-mannosidosis-specific RCT); "
            "NICE NG117 atypical antipsychotics preferred in intellectual disability + movement disorder"
        ),
    },
    {
        "drug": "VGB (Vigabatrin) — STANDARD MONITORING REQUIRED, NOT AN ABSOLUTE CI",
        "risk": "CAUTION — Visual field monitoring mandatory (standard VGB protocol)",
        "mechanism": (
            "IRREVERSIBLE GABA-T inhibition → peripheral retinal toxicity (visual field constriction); "
            "alpha-mannosidosis does NOT have retinal NCL (unlike CLN1–CLN8 where VGB = ABSOLUTE CI) "
            "and does NOT have retinal lysosomal storage primary pathology; "
            "standard VGB retinal risk applies (not amplified by disease); "
            "hearing loss in alpha-mannosidosis does not interact with VGB retinal risk"
        ),
        "alternative": (
            "LEV or VPA preferred for IS/generalised seizures; VGB only for IS if ACTH fails; "
            "if used: ophthalmology baseline + ERG + visual field 3-monthly; "
            "KEY DISTINCTION: VGB not ABSOLUTE CI in alpha-mannosidosis unlike NCL diseases"
        ),
        "evidence": (
            "Alpha-mannosidosis: no retinal storage disease; VGB retinal risk applies as in general population; "
            "NCL (CLN1–CLN8): VGB ABSOLUTE CI (retinal NCL + VGB = catastrophic additive blindness); "
            "distinction critical in differential diagnosis of alpha-mannosidosis vs NCL"
        ),
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin (IV)",
        "risk": "RELATIVE CI — prefer IV LEV for SE; PHT for chronic use requires justification",
        "mechanism": (
            "Sodium channel blocker — no specific alpha-mannosidosis contraindication; "
            "IV fosphenytoin has local tissue toxicity risk (purple glove syndrome); "
            "PHT narrow therapeutic index; non-linear kinetics; cerebellar toxicity worsens ataxia; "
            "drug interactions with azole antifungals used in recurrent infections (CYP2C9)"
        ),
        "alternative": (
            "IV LEV 60 mg/kg for SE (equivalent efficacy, safer in encephalopathic patients); "
            "oral LEV or VPA for chronic management; "
            "avoid PHT if significant ataxia (cerebellar toxicity compound)"
        ),
        "evidence": (
            "ESETT trial (IV LEV vs PHT vs VPA — equivalent SE efficacy); "
            "PHT cerebellar toxicity worsening in pre-existing ataxia — general neurology principle; "
            "CYP2C9/2C19 interactions with triazole antifungals used for recurrent fungal infections"
        ),
    },
    {
        "drug": "Carbamazepine / Oxcarbazepine — NOTE: SAFE, NOT CI (unlike MLD/GALC/ARSA)",
        "risk": "MONITOR — HLA-B*15:02 mandatory; SIADH in dehydration; CYP interactions",
        "mechanism": (
            "IMPORTANT DISTINCTION: CBZ/OXC are ABSOLUTE CIs in MLD, GALC, ARSA (demyelinating neuropathy); "
            "BUT in alpha-mannosidosis there is NO demyelinating peripheral neuropathy — "
            "oligosaccharide storage does not cause the same axonal/myelin vulnerability as sulfatide/psychosine; "
            "CBZ/OXC standard focal seizure use applies; monitor SIADH + HLA-B*15:02 + CYP interactions"
        ),
        "alternative": "No need to avoid — use with standard monitoring protocols",
        "evidence": (
            "MLD: CBZ → Na-channel blockade worsens demyelinating neuropathy → ABSOLUTE CI; "
            "GALC: psychosine demyelination + CBZ → axonal loss → ABSOLUTE CI; "
            "Alpha-mannosidosis: oligosaccharide storage does not cause demyelinating neuropathy → "
            "CBZ/OXC may be used with standard monitoring"
        ),
    },
]

THRESHOLDS = [
    "LAMAN leukocyte activity <10% control = diagnostic for alpha-mannosidosis",
    "Urine oligosaccharides TLC — Man2–5-GlcNAc2 bands = pathognomonic pattern",
    "Hearing loss >25 dB = significant — hearing aids; ENT review every 6 months",
    "Velmanase alfa 1 mg/kg IV weekly = approved ERT dose (EMA 2018/FDA 2023)",
    "IVIG threshold: IgG <4 g/L OR >4 serious bacterial infections per year",
    "CBZ TDM target 4–12 mg/L; recheck at 6–8 weeks (autoinduction); check Na if infections",
    "VPA LFT every 3 months; POLG1 exclusion before initiation (CPIC Level A mandatory)",
    "HLA-B*15:02 test BEFORE CBZ/OXC in any SE Asian patient (SJS/TEN fatality risk)",
    "HSCT best outcomes: pre-symptomatic or <6 years of age (developmental window)",
    "Velmanase alfa neutralising antibodies: check if clinical efficacy declines after 6 months",
    "Anti-velmanase IgE: check if infusion reactions occur (anaphylaxis risk)",
    "MRI: white matter changes in severe phenotype; atrophy pattern; annual in progressive cases",
]

KEY_CONCEPTS = [
    {
        "term": "MAN2B1 and Lysosomal Alpha-Mannosidase (LAMAN)",
        "definition": (
            "MAN2B1 (19p13.2) encodes lysosomal acid alpha-mannosidase (LAMAN, EC 3.2.1.24), a 1011-aa "
            "~107 kDa precursor glycoprotein processed to a mature ~70 kDa active form. LAMAN cleaves "
            "terminal alpha-mannosyl residues from N-linked oligosaccharides (Man2–Man5-GlcNAc2) in "
            "lysosomes. Without LAMAN, mannose-rich oligosaccharides from glycoprotein catabolism "
            "accumulate in all lysosomes — neurons, hepatocytes, B-lymphocytes, connective tissue cells."
        ),
    },
    {
        "term": "Urine oligosaccharides — pathognomonic first-line screen",
        "definition": (
            "Quantitative urine TLC reveals Man2–Man5-GlcNAc2 oligosaccharides: a specific ladder pattern "
            "at retention times unique to alpha-mannosidosis. Sensitivity ~95%, specificity >99%. "
            "This is the fastest, cheapest LSD urine screen — performable in most metabolic labs worldwide. "
            "Oligosaccharide bands normalise with velmanase alfa ERT — useful treatment biomarker."
        ),
    },
    {
        "term": "Vacuolated lymphocytes — bedside PAS-positive pathognomonic finding",
        "definition": (
            "Peripheral blood smear: PAS-positive large clear cytoplasmic vacuoles in lymphocytes, "
            "virtually 100% of alpha-mannosidosis patients. Detectable on routine complete blood count smear "
            "examination. Fastest bedside clue — minutes to result from any haematology lab. "
            "Distinguished from CLN3 vacuoles (CLN3 = PAS-negative electron-dense granular deposits "
            "vs alpha-mannosidosis = clear PAS-positive vacuoles)."
        ),
    },
    {
        "term": "Velmanase alfa (Lamzede) — first and only approved ERT",
        "definition": (
            "Recombinant human LAMAN produced in CHO cells; M6P-tagged for lysosomal targeting; "
            "EMA approved 2018 (EU), FDA approved July 2023 (USA). Dose: 1 mg/kg IV weekly. "
            "Primary benefit: reduces infections (30% fewer), improves functional motor ability, "
            "normalises urine oligosaccharides. CNS penetration LIMITED — M6PR expression deficient "
            "on CNS blood-brain barrier endothelium reduces enzyme uptake in brain. "
            "Somatic benefits predominate; neurological benefit is attenuation of progression, "
            "not reversal. Not a cure."
        ),
    },
    {
        "term": "Immunodeficiency — unique seizure trigger in alpha-mannosidosis",
        "definition": (
            "Oligosaccharide accumulation in B-lymphocytes impairs IgG class-switching → "
            "recurrent bacterial infections (Streptococcus pneumoniae, H. influenzae, Neisseria meningitidis, "
            "CMV, EBV). Recurrent infections → recurrent fever → recurrent seizure clusters. "
            "This immunological seizure trigger is unique to alpha-mannosidosis among LSD epilepsies. "
            "Management: prophylactic antibiotics, vaccines, IVIG for severe cases, velmanase alfa "
            "reduces infection rate 30%."
        ),
    },
    {
        "term": "CBZ/OXC SAFE (unlike MLD/GALC/ARSA where ABSOLUTE CI)",
        "definition": (
            "KEY PHARMACOLOGICAL DISTINCTION: In MLD (ARSA), Krabbe (GALC), and ARSA/SUMF1, "
            "CBZ/OXC are ABSOLUTE CIs because Na-channel blockade worsens demyelinating peripheral neuropathy. "
            "Alpha-mannosidosis does NOT cause demyelinating peripheral neuropathy — oligosaccharide storage "
            "causes primarily CNS and somatic cell disease, not peripheral nerve demyelination. "
            "Therefore, CBZ/OXC can be used for focal seizures in alpha-mannosidosis with standard monitoring "
            "(HLA-B*15:02, SIADH, TDM). This is a critical differential prescribing point."
        ),
    },
    {
        "term": "HSCT — optimal CNS treatment window",
        "definition": (
            "HSCT corrects alpha-mannosidosis CNS disease by replacing LAMAN-deficient microglia with "
            "donor cells that continuously supply LAMAN to neurons via cross-correction. "
            "Best outcomes: pre-symptomatic (NBS-detected) or <6 years with minimal neurological damage. "
            "Cognitive stabilisation in 70–80% of early-HSCT cases; does NOT reverse established damage. "
            "ERT (velmanase alfa) does not match HSCT CNS efficacy due to limited BBB penetration. "
            "ERT may bridge to HSCT in pre-transplant period."
        ),
    },
    {
        "term": "p.His72Tyr — Scandinavian founder variant (~24% of European alleles)",
        "definition": (
            "The most common MAN2B1 pathogenic variant in Northern European populations, particularly "
            "Scandinavian (Norwegian, Danish, Swedish). Disrupts His72 which coordinates zinc in LAMAN "
            "active site → near-complete loss of enzyme activity. Homozygous p.His72Tyr = moderate phenotype "
            "(not neonatal lethal). Compound heterozygous with null allele = moderate-to-severe. "
            "This variant alone accounts for ~24% of all European pathogenic alleles."
        ),
    },
    {
        "term": "Psychiatric misdiagnosis — delayed diagnosis in attenuated phenotype",
        "definition": (
            "Mild alpha-mannosidosis (Type I / attenuated) presents in adolescence/adulthood with: "
            "learning difficulties, behavioral problems, depression, psychosis-like features — "
            "BEFORE clear neurological signs. Standard psychiatric workup does not include urine "
            "oligosaccharide testing. Mean diagnostic delay in mild cases: 8–15 years. "
            "RED FLAGS: intellectual disability + hearing loss + psychiatric features in any patient "
            "→ urine TLC for oligosaccharides + leukocyte LAMAN enzyme assay."
        ),
    },
    {
        "term": "VGB not absolute CI (unlike NCL diseases)",
        "definition": (
            "In neuronal ceroid lipofuscinoses (CLN1–CLN8), VGB is an ABSOLUTE CI because retinal NCL "
            "(lysosomal storage in photoreceptors) + VGB retinal toxicity = irreversible catastrophic blindness. "
            "Alpha-mannosidosis does NOT have retinal NCL; oligosaccharide storage does not primarily affect "
            "photoreceptors; VGB standard retinal risk applies (as in general population). "
            "ERG + visual field monitoring is required if VGB is used — but it is not an absolute contraindication."
        ),
    },
    {
        "term": "M6P receptor targeting and CNS limitations of velmanase alfa",
        "definition": (
            "Velmanase alfa is tagged with mannose-6-phosphate (M6P) groups for lysosomal targeting via "
            "M6P receptors on cell surfaces. Peripheral tissues (liver, spleen, muscle) express high levels "
            "of M6PR → good ERT uptake → somatic benefit. CNS blood-brain barrier endothelium expresses "
            "LOW M6PR → poor ERT transit across BBB → limited CNS enzyme delivery. This is why HSCT "
            "(which delivers cells that directly enter CNS) is superior to ERT for neurological outcomes."
        ),
    },
    {
        "term": "Differential diagnosis from NCL (neuronal ceroid lipofuscinosis)",
        "definition": (
            "Alpha-mannosidosis vs NCL (most common diagnostic confusion): "
            "NCL: progressive visual failure (ABSENT in alpha-mannosidosis), retinal degeneration on ERG, "
            "neuronal ceroid on EM (PATHOGNOMONIC granular/curvilinear/fingerprint), enzyme assay (CLN1=PPT1, "
            "CLN2=TPP1), VGB ABSOLUTE CI. "
            "Alpha-mannosidosis: hearing loss (NOT primary visual failure), vacuolated lymphocytes (PAS+), "
            "mannose oligosaccharides on urine TLC, LAMAN enzyme low, VGB NOT absolute CI. "
            "Fastest distinguisher: urine TLC (alpha-mannosidosis specific) vs CLN1/2 DBS enzyme assay."
        ),
    },
]

STANDARDS = [
    "Malm-1999-J-Inherit-Metab-Dis (natural history, cognitive)",
    "Nilssen-1997-Am-J-Hum-Genet (MAN2B1 genotype-phenotype)",
    "Mynarek-2012-Bone-Marrow-Transplant (HSCT evidence)",
    "Borgwardt-2013-Orphanet-J-Rare-Dis (clinical endpoints ERT)",
    "Borgwardt-2015-J-Inherit-Metab-Dis (velmanase alfa Phase III)",
    "Lamzede-EMA-2018-EPAR (European approval dossier velmanase alfa)",
    "FDA-Lamzede-2023-NDA (US approval velmanase alfa July 2023)",
    "ILAE-2022-Epilepsy-Genetics (LSD epilepsy classification)",
    "NICE-NG217-MLD (AED contraindications in LSD — CBZ principle)",
    "CPIC-POLG1-2023 (VPA contraindication in mitochondrial disease)",
    "CPIC-HLA-B-CBZ-2023 (HLA-B*15:02 mandatory before CBZ/OXC)",
    "ACMG-AMP-2015 (variant classification — MAN2B1 pathogenic criteria)",
]

DIAGNOSTIC_ALGORITHM = [
    "Step 1 — Suspect alpha-mannosidosis: Intellectual disability + hearing loss + recurrent bacterial "
    "infections + coarse facial features → ANY 3 of 4 = investigate; OR: intellectual disability + "
    "psychiatric features in adolescent/adult + ANY suggestive feature → screen",
    "Step 2 — Peripheral blood smear: PAS-stain for vacuolated lymphocytes (15 min, any haematology lab); "
    "if positive → confirms lysosomal storage → next step; if negative → alpha-mannosidosis less likely "
    "(but proceed if clinical suspicion high)",
    "Step 3 — Urine TLC oligosaccharides: mannose-rich Man2–Man5-GlcNAc2 bands = PATHOGNOMONIC for "
    "alpha-mannosidosis (24–48h, metabolic lab); negative result makes alpha-mannosidosis unlikely",
    "Step 4 — Leukocyte LAMAN enzyme assay: alpha-mannosidase activity <10% control = DIAGNOSTIC; "
    "collect in EDTA (fresh cells within 4h); cannot use DBS reliably for LAMAN — fresh leukocytes required",
    "Step 5 — MAN2B1 gene sequencing (19p13.2): biallelic pathogenic variants confirm molecular diagnosis; "
    "check for p.His72Tyr (Scandinavian/European); WES if targeted panel negative (atypical cases)",
    "Step 6 — POLG1 exclusion: MANDATORY before VPA initiation (CPIC-POLG1-2023 Level A); "
    "MERRF (PME + mitochondrial, cherry-red spot rare) is phenocopy — VPA fatal if POLG1+",
    "Step 7 — HLA-B*15:02 typing: MANDATORY before CBZ/OXC in any SE Asian patient (SJS/TEN risk); "
    "note: alpha-mannosidosis does NOT have neuropathy CI for CBZ/OXC unlike MLD/ARSA/GALC",
    "Step 8 — Multidisciplinary plan: Neurology (AED — VPA/LEV/CBZ-with-monitoring) + Metabolic Medicine "
    "(velmanase alfa ERT weekly) + Immunology (IVIG if severe immunodeficiency) + Audiology "
    "(hearing aids — >90% hearing loss) + ENT + Genetics + Psychiatry + Physiotherapy",
]

GLOSSARY = {
    "MAN2B1": "Mannosidase Alpha Class 2B Member 1 — gene (19p13.2) encoding lysosomal acid alpha-mannosidase (LAMAN); biallelic LOF = alpha-mannosidosis",
    "LAMAN": "Lysosomal Acid Alpha-Mannosidase — MAN2B1 protein product; cleaves terminal alpha-mannosyl residues from N-linked glycoproteins in lysosomes",
    "Alpha-Mannosidosis": "OMIM 248500; AR LAMAN deficiency; mannose-rich oligosaccharide accumulation; ID + infections + hearing loss + epilepsy",
    "Mannose-rich oligosaccharides": "Man2–Man5-GlcNAc2 — lysosomal storage substrate in alpha-mannosidosis; quantified by urine TLC (diagnostic); normalise with velmanase alfa ERT",
    "Vacuolated lymphocytes": "PAS-positive clear cytoplasmic vacuoles in peripheral blood lymphocytes (~100% of patients); fastest bedside diagnostic clue",
    "Velmanase alfa (Lamzede)": "Recombinant human LAMAN; EMA 2018 / FDA 2023; 1 mg/kg IV weekly; somatic benefit predominates (limited CNS penetration via M6PR pathway)",
    "M6P receptor (M6PR)": "Mannose-6-phosphate receptor — targets velmanase alfa to lysosomes; high in peripheral tissues; low on CNS BBB endothelium → limits ERT CNS penetration",
    "p.His72Tyr": "Most common European MAN2B1 pathogenic variant (~24% of all European alleles); Scandinavian founder; disrupts zinc-coordinating His72 in LAMAN active site",
    "Immunodeficiency": "B-cell IgG class-switching defect from oligosaccharide accumulation in B-lymphocytes; recurrent bacterial infections → seizure trigger unique to alpha-mannosidosis",
    "POLG1 exclusion": "Mandatory before VPA — CPIC Level A; MERRF is alpha-mannosidosis phenocopy; fatal VPA hepatotoxicity if POLG1 mutation present",
    "HLA-B*15:02": "Pharmacogenomic test required before CBZ/OXC in SE Asian ancestry; SJS/TEN fatal risk; alpha-mannosidosis does NOT have CBZ neuropathy CI (unlike MLD/GALC)",
    "CBZ safe (not CI)": "KEY DISTINCTION: CBZ/OXC SAFE in alpha-mannosidosis — no demyelinating neuropathy unlike MLD(ARSA)/Krabbe(GALC)/SUMF1 where CBZ/OXC = ABSOLUTE CI",
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
            "MAN2B1 (Lysosomal Acid Alpha-Mannosidase / LAMAN, 19p13.2) biallelic LOF → "
            "LAMAN deficiency → accumulation of mannose-rich oligosaccharides (Man2–Man5-GlcNAc2) "
            "in lysosomes of neurons, hepatocytes, B-lymphocytes, and connective tissue cells. "
            "Alpha-Mannosidosis (OMIM 248500): incidence ~1:500,000; progressive intellectual disability (100%); "
            "recurrent bacterial infections due to B-cell immunodeficiency (85%); sensorineural hearing loss (>90%); "
            "coarse facial features (80%); skeletal dysplasia/ataxia (60–75%); psychiatric features (30–50%); "
            "epilepsy (30–50%, onset typically adolescence–adulthood). "
            "Pan-European founder variant p.His72Tyr (~24% of European alleles). "
            "FIRST/ONLY ERT: Velmanase alfa (Lamzede) — EMA 2018, FDA 2023, 1 mg/kg weekly IV. "
            "HSCT: CNS-effective if pre-symptomatic or early. 40-patient cohort."
        ),
        "protein": (
            "LAMAN (Lysosomal Acid Alpha-Mannosidase) — 1011 amino acids, ~107 kDa precursor; "
            "processed in lysosomes to mature ~70 kDa active form; zinc metalloenzyme (GH38 family); "
            "cleaves alpha-1,2; 1,3; 1,6 mannosyl linkages from N-linked glycoproteins; "
            "M6P-tagged for lysosomal targeting via mannose-6-phosphate receptor pathway"
        ),
        "mechanism": (
            "MAN2B1 LOF → absent LAMAN → N-linked glycoprotein catabolism blocked at terminal mannose step → "
            "Man2–Man5-GlcNAc2 oligosaccharide accumulation in lysosomes of ALL cell types → "
            "lysosomal swelling → cellular dysfunction → neurodegeneration, B-cell IgG class-switching defect, "
            "connective tissue abnormalities, skeletal changes, hepatomegaly. "
            "Oligosaccharide accumulation in B-lymphocytes → immunodeficiency → recurrent infections → "
            "fever → seizure trigger (unique immunological epilepsy mechanism in alpha-mannosidosis)."
        ),
        "pathognomonic_note": (
            "URINE MANNOSE-RICH OLIGOSACCHARIDES (Man2–Man5-GlcNAc2): quantitative TLC = PATHOGNOMONIC, "
            "fastest cheapest non-invasive screen from any metabolic lab. "
            "VACUOLATED LYMPHOCYTES (PAS-positive): ~100% of patients on peripheral blood smear = "
            "fastest bedside diagnostic clue (15 minutes). "
            "LAMAN LEUKOCYTE ENZYME ACTIVITY <10% CONTROL: confirmatory biochemical test. "
            "MAN2B1 BIALLELIC VARIANTS: molecular confirmation."
        ),
        "diagnostic_hierarchy": (
            "1. ID + hearing loss + infections + coarse features → ANY 3 → screen. "
            "2. Blood smear PAS vacuolated lymphocytes (fastest bedside). "
            "3. Urine TLC oligosaccharides (Man2-5-GlcNAc2 bands = pathognomonic). "
            "4. Leukocyte LAMAN <10% control (fresh cells — cannot use DBS). "
            "5. MAN2B1 WES/panel → biallelic pathogenic variants. "
            "6. POLG1 exclusion before VPA; HLA-B*15:02 before CBZ/OXC in SE Asian."
        ),
        "seizure_pct_overall": 40,
        "seizure_pct_severe": 60,
        "seizure_pct_moderate": 40,
        "seizure_pct_mild": 20,
        "drug_resistant_pct": 30,
        "gtcs_pct": 70,
        "myoclonic_pct": 30,
        "focal_pct": 25,
        "febrile_cluster_pct": 45,
        "hearing_loss_pct": 92,
        "infections_pct": 85,
        "intellectual_disability_pct": 100,
        "psychiatric_features_pct": 40,
        "ataxia_pct": 60,
        "vacuolated_lymphocytes_pct": 98,
        "on_ert_pct": 55,
        "on_lev_pct": 68,
        "on_vpa_pct": 50,
        "on_cbz_pct": 28,
        "on_ivig_pct": 22,
        "mean_onset_years": 4.5,
        "mean_diagnosis_delay_years": 8.2,
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
            "Step 1: Velmanase alfa (Level A ERT) — start ALL confirmed patients (EMA 2018/FDA 2023); 1 mg/kg IV weekly",
            "Step 2: IVIG — if severe immunodeficiency (IgG <4 g/L or >4 serious infections/year) — reduces seizure triggers",
            "Step 3: LEV (Level B) — GTCS + focal + myoclonic; broad-spectrum; IV for SE; first AED in severe phenotype",
            "Step 4: VPA (Level B) — GTCS + myoclonic; POLG1 exclusion MANDATORY first; broad-spectrum",
            "Step 5: CBZ/OXC (Level B) — SAFE for focal seizures; HLA-B*15:02 BEFORE in SE Asian; SIADH monitor",
            "Step 6: Clobazam (Level C) — focal adjunct; absence-like; rescue cluster seizures",
            "Step 7: HSCT — severe/early-onset; BEST CNS option; pre-symptomatic preferred; ERT bridge pre-HSCT",
            "AVOID: Typical antipsychotics (EPS worsens ataxia); PHT chronic (ataxia, interactions); "
            "VGB without ophthalmology monitoring (not CI but requires surveillance)",
            "KEY DISTINCTION: CBZ/OXC NOT contraindicated in alpha-mannosidosis (unlike MLD/GALC/ARSA where ABSOLUTE CI)",
        ],
        "ert_vs_hsct_comparison": {
            "velmanase_alfa": {
                "route": "1 mg/kg IV weekly (lifelong)",
                "cns_penetration": "LIMITED (M6PR low on BBB endothelium)",
                "somatic_benefit": "Reduces infections 30%; improves function; normalises urine oligosaccharides",
                "neurological_benefit": "Attenuates progression; does not reverse established CNS damage",
                "eligibility": "All confirmed alpha-mannosidosis (EMA/FDA approved)",
                "practical": "Weekly infusion centre; anti-drug antibody monitoring",
            },
            "hsct": {
                "route": "Haematopoietic stem cell transplant (once); allogeneic",
                "cns_penetration": "EXCELLENT (donor microglia cross BBB; continuous CNS enzyme supply)",
                "somatic_benefit": "Immunological reconstitution; hepatosplenomegaly correction",
                "neurological_benefit": "Cognitive stabilisation 70–80% in pre-symptomatic; superior to ERT for CNS",
                "eligibility": "Severe/early phenotype; pre-symptomatic or <6 years preferred",
                "practical": "One-time transplant; significant transplant morbidity 5–15%; specialist centre",
            },
            "combination": "ERT bridges to HSCT in pre-transplant period; ERT continues post-HSCT for somatic maintenance",
        },
    }


def get_definitions():
    """Return definitions, diagnostic algorithm, glossary, key concepts, and standards."""
    return {
        "diagnostic_algorithm": DIAGNOSTIC_ALGORITHM,
        "man2b1_glossary": GLOSSARY,
        "key_concepts": KEY_CONCEPTS,
        "standards": STANDARDS,
        "differential_diagnosis": {
            "NCL_CLN1_CLN8": (
                "NCL: progressive visual failure + retinal degeneration (ERG pathological) ABSENT in alpha-mannosidosis; "
                "NCL EM = PATHOGNOMONIC granular/curvilinear/fingerprint inclusions; "
                "NCL: VGB ABSOLUTE CI (retinal NCL + VGB = catastrophic blindness); "
                "alpha-mannosidosis: hearing loss predominates, NOT primary visual failure; "
                "urine TLC distinguishes (oligosaccharide bands = alpha-mannosidosis, NOT NCL)"
            ),
            "MLD_ARSA": (
                "MLD: ARSA enzyme low (not LAMAN); urine sulfatides (not oligosaccharides); "
                "peripheral demyelinating neuropathy (ABSENT in alpha-mannosidosis); "
                "CBZ/OXC ABSOLUTE CI in MLD (NOT CI in alpha-mannosidosis — key prescribing distinction); "
                "white matter leukodystrophy more prominent in MLD"
            ),
            "GM2_Gangliosidosis_TaySachs_Sandhoff": (
                "HEXA/HEXB: cherry-red macular spot (ABSENT in alpha-mannosidosis); "
                "hexosaminidase A/B enzyme low (not LAMAN); "
                "urine: GM2 ganglioside/oligosaccharides (different from mannose oligosaccharides); "
                "no vacuolated lymphocytes in GM2 gangliosidosis"
            ),
            "Fucosidosis_FUCA1": (
                "FUCA1: alpha-fucosidase deficiency; urine fucose-rich oligosaccharides (distinguish by TLC pattern); "
                "similar clinical phenotype (ID + coarse features + infections); "
                "angiokeratoma present in fucosidosis (absent or rare in alpha-mannosidosis); "
                "vacuolated lymphocytes in both — urine TLC distinguishes"
            ),
            "Pompe_GAA": (
                "Pompe: GAA enzyme (not LAMAN); no ID typically; cardiomyopathy + myopathy prominent; "
                "urine TLC: glucose oligosaccharides (not mannose oligosaccharides); no vacuolated lymphocytes"
            ),
        },
    }
