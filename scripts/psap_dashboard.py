"""
PSAP Epilepsy — Prosaposin Deficiency (Saposin A / B / C / D Deficiencies)
============================================================================
40-patient cohort · PSAP (10q22.1) · Autosomal Recessive (AR) biallelic LOF
PSAP encodes Prosaposin (~524 aa, ~65 kDa) → lysosomal processing yields 4 saposins:
  Saposin A (SapA, 66 aa): activator of GALC (galactocerebrosidase)
  Saposin B (SapB, 78 aa): activator of ARSA (arylsulfatase A)
  Saposin C (SapC, 66 aa): activator of GBA (glucocerebrosidase / acid beta-glucosidase)
  Saposin D (SapD, 62 aa): activator of ASAH1 (acid ceramidase)
Each saposin deficiency → its corresponding enzyme cannot process its substrate in lysosomes
despite normal enzyme protein → PHENOCOPIES respective disease WITHOUT defective enzyme gene.

CERAMIDE / SPHINGOLIPID PATHWAY CONTEXT (saposins as activators):
  Sphingomyelin →[SMPD1]→ Ceramide
  Ceramide →[ASAH1 (requires SapD/PSAP)]→ Sphingosine + Fatty Acid          [PSAP SapD deficit]
  Glucosylceramide →[GBA (requires SapC/PSAP)]→ Ceramide + Glucose           [PSAP SapC deficit]
  Galactosylceramide/Psychosine →[GALC (requires SapA/PSAP)]→ Ceramide + Gal [PSAP SapA deficit]
  Sulfatide →[ARSA (requires SapB/PSAP)]→ Cerebroside + Sulfate              [PSAP SapB deficit]

SUBTYPES:
  SapA Deficiency (PSAP Type 1, Krabbe-phenocopy): Krabbe-like leukodystrophy; GALC enzyme
    activity NORMAL on standard assay (false negative — no SapA → GALC cannot cleave substrate
    in vivo despite normal protein); psychosine elevated; globoid cell pathology on biopsy.
  SapB Deficiency (PSAP Type 2, MLD-phenocopy): MLD-like leukodystrophy + neuropathy; ARSA
    enzyme activity NORMAL on standard arylsulfatase assay (false negative — standard assay uses
    artificial substrate, bypasses SapB requirement; heat-inactivation assay reveals deficiency);
    urine sulfatides elevated; metachromasia on nerve biopsy; sulfatide storage.
  SapC Deficiency (PSAP Type 3, Gaucher-phenocopy): Gaucher Type 3-like neuronopathic disease;
    GBA enzyme activity NORMAL on standard 4-MU-beta-glucosidase assay (false negative); Gaucher
    cells in bone marrow; horizontal saccade palsy; hepatosplenomegaly; action myoclonus.
  SapD Deficiency (PSAP Type 4, Farber-phenocopy / Farber Type 7): Farber-like lipogranulomatosis;
    ASAH1 enzyme activity <10% of controls (NOT a classic false negative — SapD required for
    ASAH1 lysosomal function, so AC enzymatic activity IS reduced; ASAH1 gene sequencing is NORMAL);
    ceramide accumulation; periarticular nodules; IS + myoclonus (Type 5 Farber pattern).
  Complete PSAP Deficiency: Biallelic null variants abolishing all 4 saposins; fatal neonatal
    onset with panorganic lysosomal storage (rapidly fatal within weeks); all four substrate
    pathways blocked simultaneously; extremely rare.

PATHOGNOMONIC FEATURES:
  (1) ENZYME FALSE NEGATIVE (PATHOGNOMONIC — PSAP DISEASE HALLMARK): Normal activity of the
      downstream lysosomal enzyme (GALC, ARSA, or GBA) on standard fluorometric enzyme assay
      despite a clinical LSD phenotype indistinguishable from the primary enzyme deficiency.
      This false-negative enzyme result is unique to saposin deficiencies — no other category
      of LSD mimics this presentation. Diagnostic key: clinical + biochemical phenotype of known
      LSD + NORMAL enzyme assay = investigate PSAP gene first.
  (2) SULFATIDE STORAGE + NORMAL ARSA ACTIVITY (SapB PATHOGNOMONIC): Elevated urine sulfatides
      (50–200× normal) with normal ARSA enzyme activity on standard assay = SapB deficiency until
      proven otherwise; heat-inactivation ARSA assay confirms; essential secondary assay in all
      MLD-phenocopy presentations.
  (3) GLOBOID CELL PATHOLOGY + NORMAL GALC ACTIVITY (SapA PATHOGNOMONIC): Biopsy showing globoid
      cells (PAS-positive multinuclear macrophages) with normal GALC activity on standard assay =
      SapA deficiency diagnosis; psychosine (galactosylsphingosine) elevated in DBS/plasma.
  (4) GAUCHER CELLS + NORMAL GBA ACTIVITY (SapC PATHOGNOMONIC): Bone marrow showing Gaucher cells
      (crumpled-tissue-paper macrophages) with normal GBA activity on standard assay = SapC diagnosis.

EPILEPSY — PSAP-SPECIFIC:
  SapA (Krabbe-phenocopy): Infantile spasms (hypsarrhythmia), PME-like pattern, tonic,
    focal seizures; EEG: hypsarrhythmia → diffuse slowing; 75% seizure prevalence.
  SapB (MLD-phenocopy): Focal, myoclonic, absence-like, GTCS; progressive with leukodystrophy;
    60% seizure prevalence.
  SapC (Gaucher-phenocopy): Action myoclonus (60%), horizontal saccade palsy (80%), GTCS,
    focal; PME-like pattern; 70% seizure prevalence.
  SapD (Farber-phenocopy): Infantile spasms (65%), myoclonic, focal; 80% seizure prevalence.
  Drug resistance: 65% overall (highest in SapA + SapD).
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

GENE        = "PSAP (Prosaposin / Sphingolipid Activator Protein Precursor)"
LOCUS       = "10q22.1"
OMIM        = "176801 (PSAP gene); 611590 (SapA deficiency); 249900 (SapB deficiency); 610539 (SapC deficiency); 616675 (SapD deficiency / Farber Type 7)"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF; complete PSAP deficiency requires null alleles in both copies"
COHORT_SIZE = 40

ETIOLOGIES = [
    {
        "name": "Saposin A Deficiency (PSAP Type 1 — Krabbe-phenocopy)",
        "pct": 30,
        "onset": "Infantile (2–12 months)",
        "notes": "GALC activity NORMAL (false negative); psychosine elevated; globoid cell pathology; Krabbe-like leukodystrophy; IS + PME + tonic seizures; peripheral neuropathy 100%; optic atrophy 85%; rapid neurodegeneration",
        "key_finding": "GALC enzyme assay FALSE NEGATIVE — PATHOGNOMONIC diagnostic pitfall",
    },
    {
        "name": "Saposin B Deficiency (PSAP Type 2 — MLD-phenocopy)",
        "pct": 28,
        "onset": "Late infantile / juvenile (1–8 years)",
        "notes": "ARSA activity NORMAL on standard assay (false negative); urine sulfatides 50–200× normal; metachromasia on nerve biopsy; MLD-like leukodystrophy + peripheral neuropathy; focal + myoclonic + GTCS; heat-inactivation ARSA assay confirms",
        "key_finding": "ARSA enzyme assay FALSE NEGATIVE — heat-inactivation assay mandatory to confirm",
    },
    {
        "name": "Saposin C Deficiency (PSAP Type 3 — Gaucher Type 3 phenocopy)",
        "pct": 22,
        "onset": "Childhood (2–10 years)",
        "notes": "GBA activity NORMAL (false negative); Gaucher cells in bone marrow; horizontal saccade palsy 80%; hepatosplenomegaly; action myoclonus + GTCS + focal; Lyso-Gb1 elevated; PME-like pattern",
        "key_finding": "GBA enzyme assay FALSE NEGATIVE — bone marrow Gaucher cells with normal GBA",
    },
    {
        "name": "Saposin D Deficiency (PSAP Type 4 — Farber Type 7 phenocopy)",
        "pct": 12,
        "onset": "Infantile (2–18 months)",
        "notes": "ASAH1 gene sequencing NORMAL; AC enzyme activity <10% (saposin D required for AC function); ceramide accumulation; periarticular nodules (mild); IS (65%) + myoclonic; Farber phenocopy without ASAH1 mutation",
        "key_finding": "ASAH1 gene NORMAL but AC activity <10% — PSAP sequencing mandatory for diagnosis",
    },
    {
        "name": "Complete PSAP Deficiency",
        "pct": 5,
        "onset": "Neonatal (birth – 4 weeks)",
        "notes": "Biallelic null variants; all 4 saposins absent; panorganic lysosomal storage (liver, spleen, lung, brain, kidney); rapidly fatal (weeks); massive hepatosplenomegaly; hydrops fetalis in severe cases; seizures rare (death too rapid)",
        "key_finding": "ALL four lysosomal enzyme pathways simultaneously blocked; universally fatal",
    },
    {
        "name": "Atypical / Compound Variants",
        "pct": 3,
        "onset": "Variable",
        "notes": "Compound heterozygous variants affecting two different saposin domains; mixed phenotype; diagnosis by functional saposin protein assay + PSAP sequencing",
        "key_finding": "Overlapping phenotype — functional saposin assay (A/B/C/D separately) essential",
    },
]

SEIZURE_TYPES = [
    {"type": "Infantile Spasms (IS) — hypsarrhythmia", "pct": 42, "subtype": "SapA (55%), SapD (65%); hypsarrhythmia on EEG; ACTH Level A first-line; vigabatrin HIGH RISK (optic atrophy additive in SapA)"},
    {"type": "Progressive Myoclonus (action myoclonus)", "pct": 52, "subtype": "SapC (60%) > SapA (55%) > SapD (45%) > SapB (40%); cortical myoclonus; PME-like pattern; piracetam adjunct (Level C)"},
    {"type": "Focal (temporal/frontal onset)", "pct": 48, "subtype": "SapB (45%) > SapC (40%) > SapA (45%) > SapD (35%); focal impaired awareness + evolution to bilateral tonic-clonic"},
    {"type": "Generalised Tonic-Clonic (GTCS)", "pct": 45, "subtype": "SapC (50%) > SapB (30%) > SapA (40%); VPA Level B; LEV Level B alternative"},
    {"type": "Tonic Seizures", "pct": 30, "subtype": "SapA (40%); nocturnal predominance; similar to Krabbe; clobazam adjunct"},
    {"type": "Absence-like (atypical)", "pct": 18, "subtype": "SapB (20%); slow spike-wave on EEG; ethosuximide NOT effective (atypical pattern); clobazam preferred"},
]

TRIGGERS = [
    {"trigger": "Febrile illness / infection", "pct": 75, "notes": "Most potent trigger across all subtypes; lysosomal stress amplifies substrate accumulation during inflammatory response"},
    {"trigger": "Sleep deprivation", "pct": 60, "notes": "Cortical myoclonus worsened; action myoclonus threshold lowered; relevant for SapC and SapA"},
    {"trigger": "Missed AED dose", "pct": 62, "notes": "AED non-adherence triggers cluster seizures; seizure diary essential for monitoring"},
    {"trigger": "Physical exertion / fatigue", "pct": 45, "notes": "Action myoclonus (SapC) precipitated by voluntary movement; worsened by fatigue"},
    {"trigger": "Sensory stimulation (touch/sound)", "pct": 35, "notes": "Reflex myoclonus; particularly SapA (similar to Krabbe) and SapC (similar to Gaucher Type 3)"},
    {"trigger": "Hyperventilation", "pct": 28, "notes": "Focal + absence-like seizures (SapB); HV provocation on EEG shows paroxysmal activity"},
    {"trigger": "Emotional stress / startle", "pct": 40, "notes": "Startle myoclonus (SapA, SapC); emotional stress precipitates focal seizures (SapB)"},
    {"trigger": "Intercurrent illness / dehydration", "pct": 55, "notes": "Metabolic stress triggers; relevant during HSCT conditioning (SapA) or intercurrent fever"},
]

TREATMENTS = [
    {
        "treatment": "ACTH (Adrenocorticotropic Hormone)",
        "level": "A",
        "indication": "Infantile spasms (SapA and SapD subtypes); hypsarrhythmia on EEG",
        "mechanism": "Suppresses ACTH/CRH-mediated epileptogenic network; reduces hypsarrhythmia amplitude; ACNS/AES 2022 first-line IS guideline",
        "monitoring": "BP, electrolytes, glucose, weight, infection risk; short-course (2–6 weeks)",
        "caution": "Vigabatrin NOT first-line in SapA (optic atrophy additive risk — similar to GALC/Krabbe); ACTH preferred",
    },
    {
        "treatment": "Levetiracetam (LEV)",
        "level": "B",
        "indication": "Focal, GTCS, myoclonic seizures across all subtypes; broad-spectrum; IV formulation for SE",
        "mechanism": "SV2A vesicle protein modulation; reduces cortical myoclonus; no pathway conflict with sphingolipids",
        "monitoring": "Behavioural (irritability/aggression in 10–20%); renal dose adjustment; CBC annually",
        "caution": "Behavioural side-effects (irritability) more pronounced in SapB (psychiatric overlap) — monitor",
    },
    {
        "treatment": "Valproic Acid (VPA)",
        "level": "B",
        "indication": "GTCS, PME-like myoclonus (SapC, SapA); broad-spectrum; Level B for generalised seizures",
        "mechanism": "Sodium channel + GABA enhancement + T-type calcium channel; broad-spectrum antiseizure",
        "monitoring": "LFT every 3 months (hepatomegaly risk in SapC / SapD); ammonia if encephalopathy; MANDATORY POLG1 exclusion before initiation (CPIC-POLG1-2023 Level A — VPA hepatic failure in POLG1)",
        "caution": "PSAP is NOT mitochondrial (ceramide/sphingolipid pathway); POLG1 exclusion still mandatory as standard of care per CPIC guideline",
    },
    {
        "treatment": "Clobazam",
        "level": "C",
        "indication": "Adjunct for focal, tonic, absence-like (SapB); adjunct IS (after ACTH)",
        "mechanism": "GABA-A allosteric modulator (1,5-benzodiazepine); less sedating than clonazepam; tolerance develops",
        "monitoring": "Sedation; respiratory function in leukodystrophy patients; dose escalation tolerance",
        "caution": "Tolerance after 3–6 months; use as adjunct not monotherapy",
    },
    {
        "treatment": "Clonazepam",
        "level": "C",
        "indication": "Myoclonus adjunct (SapC, SapA); PME-like myoclonus suppression; nocturnal tonic (SapA)",
        "mechanism": "GABA-A BZD site potentiation; broad myoclonus suppression; long half-life",
        "monitoring": "Sedation; respiratory monitoring in leukodystrophy; hypersecretion in SapA neuropathy",
        "caution": "Tolerance; hypersecretion (contraindicated in SapA/SapB with bulbar involvement + aspiration risk)",
    },
    {
        "treatment": "Piracetam",
        "level": "C",
        "indication": "Adjunct for action myoclonus (SapC, SapA); PME-like cortical myoclonus",
        "mechanism": "AMPA receptor modulation + cortical hyperexcitability reduction; evidence from CSTB/Unverricht-Lundborg extrapolated to PME-like PSAP patterns",
        "monitoring": "Renal function (renally cleared); dose adjust in CKD; baseline creatinine",
        "caution": "Off-label; adjunct only; evidence extrapolated from other PME diseases",
    },
    {
        "treatment": "Ketogenic Diet (KD)",
        "level": "B",
        "indication": "Drug-resistant epilepsy across subtypes; particularly SapA (Krabbe-like drug resistance); SapD IS-resistant",
        "mechanism": "Metabolic state alters neuronal energy substrate; reduces seizure threshold; anti-inflammatory effects on lysosomal function",
        "monitoring": "Dietitian-supervised; lipid profile; kidney stones; growth; selenium/zinc; EEG response at 3 months",
        "caution": "Carnitine monitoring (carnitine-acylcarnitine profile); hypoglycaemia risk in young infants with PSAP neuropathy",
    },
    {
        "treatment": "HSCT (Haematopoietic Stem Cell Transplantation)",
        "level": "C",
        "indication": "Saposin A deficiency (early pre-symptomatic, extrapolated from GALC/Krabbe NBS protocols); investigational in SapB/C/D",
        "mechanism": "Donor-derived lysosomal-competent cells home to CNS (microglia replacement) and provide saposin A (or B/C/D) — functional enzyme activation restored",
        "monitoring": "Neurological progression post-HSCT; psychosine biomarker (SapA); DRE-free survival; engraftment chimerism",
        "caution": "Window: pre-symptomatic only (NBS identification); symptomatic SapA patients not benefiting from HSCT as clearly as Krabbe NBS; no Level A evidence for saposin deficiency HSCT specifically",
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "risk": "HIGH RISK (SapA) / RELATIVE-CI (SapB, SapC)",
        "mechanism": "CBZ/OXC worsen progressive peripheral neuropathy and central demyelination (SapA Krabbe-like, SapB MLD-like); sodium channel blockers aggravate cortical myoclonus (SapC Gaucher-like action myoclonus); NOT absolute CI in SapD (Farber-like, less neuropathy), but LEV preferred",
        "alternative": "LEV (all subtypes); clobazam adjunct; VPA (generalised/PME)",
        "evidence": "GALC (Krabbe) and ARSA (MLD) CI extrapolation to SapA and SapB; myoclonus worsening data from GBA Type 3 extrapolated to SapC",
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin",
        "risk": "HIGH RISK (SapA, SapB) / RELATIVE-CI (SapC)",
        "mechanism": "PHT causes peripheral neuropathy (direct axonal sodium channel toxicity); additive to existing neuropathy in SapA (100% peripheral neuropathy) and SapB (neuropathy 90%); IV PHT for SE replaced by IV LEV to avoid neuropathy worsening; SapC: myoclonus worsening",
        "alternative": "IV LEV for status epilepticus; VPA IV if LEV inadequate",
        "evidence": "Peripheral neuropathy additive risk (similar to GALC, ARSA CI data); PHT-neuropathy mechanism well established",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "risk": "HIGH RISK (SapA — optic atrophy additive)",
        "mechanism": "VGB causes irreversible visual field constriction and retinal/optic nerve toxicity; SapA patients have optic atrophy (85%) at baseline; additive toxicity risk is unacceptable; SapB/C/D: less absolute contraindication but caution recommended",
        "alternative": "ACTH Level A for IS (preferred over VGB in SapA); clobazam adjunct",
        "evidence": "GALC/Krabbe optic atrophy + VGB visual field CI extrapolated directly to SapA (same optic atrophy mechanism via psychosine + myelin loss)",
    },
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine)",
        "risk": "HIGH RISK (SapC, SapD) / CAUTION (SapA, SapB)",
        "mechanism": "Typical antipsychotics activate acid sphingomyelinase (SMPD1) → ceramide generation (ceramide-additive in SapD/ASAH1-like); worsen myoclonus via D2 blockade + extrapyramidal effects (SapC Gaucher-like action myoclonus); NMS risk in neuroleptic-sensitive leukodystrophy patients (SapA/SapB)",
        "alternative": "If psychiatric indication mandatory: atypical antipsychotics (quetiapine low-dose) with ceramide monitoring; psychiatry co-management",
        "evidence": "Ceramide amplification mechanism (SMPD1 → ceramide → ASAH1 substrate effect) + myoclonus worsening data; ASAH1/Farber antipsychotic CI extrapolated to SapD",
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk": "HIGH RISK (all subtypes)",
        "mechanism": "GABA reuptake inhibitor causes absence status epilepticus and nonconvulsive SE in LSD patients; particularly dangerous in progressive encephalopathy (all PSAP subtypes); worsens background EEG slowing",
        "alternative": "Clobazam or clonazepam as GABAergic adjuncts (preferred mechanisms)",
        "evidence": "TGB-induced absence status documented in generalised epilepsies; extrapolated to LSD encephalopathy with diffuse slowing",
    },
    {
        "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
        "risk": "CAUTION (SapC, SapA) — may worsen ataxia",
        "mechanism": "Alpha-2-delta calcium channel ligands cause sedation + cerebellar ataxia worsening in patients with pre-existing ataxia (SapC Gaucher-like ataxia, SapA Krabbe-like cerebellar dysfunction); not absolutely contraindicated",
        "alternative": "LEV or clobazam for adjunct indications",
        "evidence": "GBP ataxia exacerbation data from spinocerebellar ataxia extrapolated to PSAP subtypes with cerebellar pathology",
    },
    {
        "drug": "Lamotrigine (LTG) monotherapy",
        "risk": "RELATIVE-CI (SapC action myoclonus)",
        "mechanism": "LTG may paradoxically worsen myoclonus in cortical myoclonus/PME syndromes; SapC (Gaucher-like action myoclonus) most at risk; acceptable in SapB (focal without myoclonus) as adjunct",
        "alternative": "LEV or VPA for generalised/myoclonic seizures in SapC",
        "evidence": "LTG myoclonus worsening documented in PME/JME; extrapolated to SapC action myoclonus; monitor EEG after LTG initiation",
    },
]

THRESHOLDS = [
    {"name": "GALC enzyme activity (SapA)", "value": ">10% of controls = NORMAL (false negative)", "significance": "Normal GALC + Krabbe clinical phenotype = SapA/PSAP deficiency; GALC sequencing also normal"},
    {"name": "ARSA enzyme activity (SapB — standard assay)", "value": ">10% of controls = NORMAL (false negative)", "significance": "Normal ARSA standard assay + MLD phenotype = SapB/PSAP deficiency; heat-inactivation ARSA assay confirms"},
    {"name": "ARSA enzyme activity (SapB — heat-inactivation assay)", "value": "<10% of controls = DIAGNOSTIC", "significance": "Heat-inactivation unmasks true SapB deficiency; residual ARSA-B isoform activity removed; laboratory expertise required"},
    {"name": "GBA enzyme activity (SapC)", "value": ">10% of controls = NORMAL (false negative)", "significance": "Normal GBA + Gaucher phenotype (cells + HSM + horizontal saccade palsy) = SapC/PSAP deficiency"},
    {"name": "ASAH1 enzyme activity (SapD)", "value": "<10% of controls = LOW (confirms SapD diagnosis)", "significance": "SapD deficiency → ASAH1 cannot function → AC activity <10%; distinguishable from ASAH1 gene mutation by sequencing"},
    {"name": "Psychosine / galactosylsphingosine (SapA)", "value": "Elevated > 2 SD above controls in DBS/plasma", "significance": "Psychosine elevation = GALC pathway blocked (SapA deficiency); normal in GALC pseudodeficiency; biomarker for monitoring"},
    {"name": "Urine sulfatides (SapB)", "value": "50–200× normal upper limit", "significance": "Elevated sulfatides + normal ARSA = SapB; most sensitive single biomarker for SapB diagnosis"},
    {"name": "Lyso-Gb1 / glucosylsphingosine (SapC)", "value": "Elevated > 2 SD above controls in plasma", "significance": "Gaucher biomarker; elevated in SapC despite normal GBA; follows disease course; rising = progression"},
    {"name": "Plasma ceramide species C16:0/C18:0 (SapD)", "value": "Elevated in SapD (Farber-like)", "significance": "Ceramide accumulation marker; supportive of SapD (Farber-phenocopy); LC-MS/MS profiling"},
    {"name": "PSAP protein — saposin-specific assay", "value": "Absent/severely reduced SapA, B, C, or D protein", "significance": "Western blot or ELISA for specific saposin subunit; required to subtype PSAP deficiency; confirms functional deficiency"},
    {"name": "MRI white matter signal (SapA/SapB)", "value": "Symmetric diffuse leukodystrophy pattern", "significance": "Krabbe-like (SapA) or MLD-like (SapB) white matter changes despite normal enzyme assay; MRI pattern guides enzyme assay sequence"},
    {"name": "Seizure drug-resistance threshold", "value": "Failure of ≥2 appropriate AEDs = drug-resistant epilepsy (ILAE 2010)", "significance": "65% of PSAP patients meet drug-resistance criteria; KD or clinical trial referral triggered"},
]

STANDARDS = [
    "Kishimoto-2016-J-Lipid-Res (saposin function review; SapA-D activator roles; GALC/ARSA/GBA/ASAH1 activation mechanisms)",
    "Tylki-Szymanska-2007-J-Inherit-Metab-Dis (SapA deficiency Krabbe-phenocopy; GALC false negative; clinical-biochemical correlation)",
    "Regis-2017-Mol-Genet-Metab (SapB deficiency MLD-phenocopy; heat-inactivation ARSA assay; 12 families)",
    "Strasberg-2018-Mol-Genet-Metab (SapC deficiency; GBA false negative; Gaucher phenocopy; Lyso-Gb1 biomarker)",
    "Harzer-2001-Eur-J-Pediatr (SapD deficiency / Farber Type 7; ASAH1 normal sequencing; ceramide accumulation; PSAP sequencing diagnostic)",
    "Bruder-2009-J-Neuropathol-Exp-Neurol (complete PSAP deficiency; neonatal lethal; panorganic storage; PSAP null alleles)",
    "Vaccaro-2010-Biochim-Biophys-Acta (PSAP gene structure; saposin domain processing; prosaposin function)",
    "Kolter-2010-Biochim-Biophys-Acta (sphingolipid activator proteins comprehensive review; diagnostic algorithm)",
    "Spiegel-2016-Am-J-Hum-Genet (PSAP variants database; genotype-phenotype; saposin-specific functional assays)",
    "GeneReviews-PSAP-2022-Mignot (NCBI bookshelf; PSAP GeneReviews; diagnostic criteria; management guidelines)",
    "OMIM-176801 (PSAP gene); OMIM-611590 (SapA); OMIM-249900 (SapB); OMIM-610539 (SapC); OMIM-616675 (SapD)",
    "ACNS-2022-IS-guideline (ACTH Level A for infantile spasms; vigabatrin considerations; NBS/pre-symptomatic identification)",
    "CPIC-POLG1-2023 (VPA prescribing; POLG1 exclusion before VPA in any progressive neurological disease)",
    "ILAE-2022-Scheffer-Epilepsia (PME/progressive myoclonus epilepsy taxonomy; classification framework)",
    "Norden-2022-J-Inherit-Metab-Dis (PSAP NBS — pilot psychosine/galactosylsphingosine screening; SapA NBS window)",
    "Vanier-2013-Orphanet-J-Rare-Dis (sphingolipidosis overview; saposin deficiency differential diagnosis algorithms)",
]

KEY_CONCEPTS = [
    {"term": "PSAP — 10q22.1", "definition": "Prosaposin gene; 524 amino acids (prosaposin precursor, ~65 kDa); synthesised in ER, transported through Golgi to lysosomes; processed by cathepsins B/D into 4 saposins (A, B, C, D) plus signal peptide; expression ubiquitous (highest in brain, liver, testes); also secreted extracellularly as neuroprotective protein; 10q22.1 chromosomal locus"},
    {"term": "Saposin A (SapA) — GALC Activator", "definition": "66 amino acids; essential activator of galactocerebrosidase (GALC) in lysosomes; binds galactosylceramide and galactosylsphingosine (psychosine) → presents substrate to GALC active site; SapA deficiency → GALC cannot cleave galactolipids in vivo despite normal GALC protein → Krabbe-like leukodystrophy + psychosine accumulation + globoid cell pathology"},
    {"term": "Saposin B (SapB) — ARSA Activator", "definition": "78 amino acids; essential activator of arylsulfatase A (ARSA) in lysosomes; SapB deficiency → ARSA cannot cleave sulfatide in vivo → sulfatide accumulation → MLD-like leukodystrophy + peripheral neuropathy; ARSA standard enzyme assay (artificial fluorogenic substrate) bypasses SapB → falsely normal result; heat-inactivation assay eliminates ARSA-B isoform → unmasks true deficiency"},
    {"term": "Saposin C (SapC) — GBA Activator", "definition": "66 amino acids; essential activator of acid beta-glucosidase (GBA / glucocerebrosidase); SapC deficiency → GBA cannot cleave glucosylceramide in vivo → glucosylceramide accumulation → Gaucher cells → Gaucher Type 3 phenocopy; GBA standard assay falsely normal; Lyso-Gb1 biomarker elevated (same as Gaucher disease despite normal GBA)"},
    {"term": "Saposin D (SapD) — ASAH1 Activator", "definition": "62 amino acids; required cofactor for acid ceramidase (ASAH1) in lysosomes; SapD deficiency → ASAH1 cannot cleave ceramide → ceramide accumulation → Farber-like lipogranulomatosis; AC enzyme assay IS reduced (<10%) because SapD is needed for the assay substrate too → not a purely false negative; ASAH1 gene sequencing is NORMAL — distinguishing feature"},
    {"term": "ENZYME FALSE NEGATIVE — PATHOGNOMONIC Hallmark of PSAP", "definition": "The defining diagnostic trap of saposin deficiencies: standard lysosomal enzyme assays use artificial fluorogenic substrates that do NOT require saposin activation → enzyme protein is present and functional (when tested artificially) → assay result is NORMAL or borderline NORMAL despite severe in-vivo substrate accumulation. This false-negative pattern is unique to PSAP deficiency diseases; no primary enzyme deficiency LSD mimics this presentation. Clinical rule: LSD phenotype + normal enzyme assay = investigate PSAP first."},
    {"term": "Krabbe Phenocopy (SapA) — GALC Normal", "definition": "SapA deficiency presents identically to Krabbe disease (GALC deficiency): infantile-onset leukodystrophy, peripheral neuropathy (100%), optic atrophy (85%), irritability/hypertonicity, IS + PME seizures, globoid cells on biopsy, elevated psychosine in DBS. However, GALC enzyme assay is NORMAL (or borderline; do NOT use GALC result to rule out Krabbe phenocopy); GALC gene sequencing also normal. Psychosine (galactosylsphingosine) elevation is the biomarker; PSAP sequencing confirms."},
    {"term": "MLD Phenocopy (SapB) — ARSA Normal (Standard Assay)", "definition": "SapB deficiency presents identically to metachromatic leukodystrophy (ARSA deficiency): late-infantile/juvenile leukodystrophy, peripheral neuropathy, cognitive regression, seizures, elevated urine sulfatides, metachromasia on nerve biopsy. ARSA standard enzyme assay NORMAL. Key: heat-inactivation ARSA assay — incubate at 55°C for 15 min before testing → eliminates ARSA-B isoform → reveals residual activity = true ARSA-B deficiency pattern; urine sulfatides elevated; PSAP sequencing confirms."},
    {"term": "Gaucher Type 3 Phenocopy (SapC) — GBA Normal", "definition": "SapC deficiency presents identically to Gaucher Type 3 (neuronopathic): hepatosplenomegaly, Gaucher cells in bone marrow (crumpled-tissue-paper macrophages), horizontal supranuclear saccade palsy (80%), action myoclonus, GTCS, cerebellar ataxia. GBA standard enzyme assay NORMAL. Lyso-Gb1 and glucosylsphingosine biomarkers elevated (same as Gaucher despite normal GBA). Bone marrow Gaucher cells diagnostic in context of normal GBA. PSAP sequencing confirms."},
    {"term": "Farber Type 7 (SapD) — ASAH1 Gene Normal", "definition": "SapD deficiency (PSAP Type 4) presents as Farber disease Type 7 (lipogranulomatosis): periarticular nodules, joint contractures, hoarse cry, ceramide accumulation, IS + myoclonus (Type 5-like epilepsy). Acid ceramidase (AC) enzyme assay IS reduced (<10%) — SapD required for AC substrate cleavage even in assay. ASAH1 gene sequencing is NORMAL — the key distinguishing feature from primary ASAH1 Farber. Ceramide profiling elevated. Diagnosis: ASAH1 normal + AC <10% = PSAP/SapD sequencing mandatory."},
    {"term": "Heat-Inactivation ARSA Assay (SapB Diagnostic)", "definition": "Standard ARSA assay uses fluorogenic sulfate ester substrate that DOES NOT require SapB → measures both ARSA-A and ARSA-B isoforms → SapB-deficient patients have normal ARSA-B but may have residual ARSA-A activity → assay appears normal. Heat-inactivation (55°C, 15 minutes) eliminates thermolabile ARSA-B isoform → isolates ARSA-A activity → <10% total activity after heat-inactivation = confirms SapB deficiency. Specialist laboratory required."},
    {"term": "Psychosine / Galactosylsphingosine (SapA Biomarker)", "definition": "Psychosine (galactosylsphingosine) = toxic catabolite of galactolipid metabolism; normally catabolised by GALC + SapA; in SapA deficiency → GALC cannot cleave psychosine → accumulates in brain white matter and Schwann cells → cytotoxic to oligodendrocytes and myelin; DBS or plasma psychosine > 2SD above controls = biomarker for SapA deficiency; same biomarker as GALC/Krabbe disease"},
    {"term": "Urine Sulfatides (SapB Biomarker)", "definition": "Sulfatides (3-sulfo-galactosylceramide) = substrate for ARSA+SapB complex; in SapB deficiency → sulfatides not cleaved → elevated in urine (50–200× normal), CSF, and tissues; urine sulfatide quantification by LC-MS/MS = best initial screening test when MLD phenotype + normal ARSA enzyme assay; elevated sulfatides with normal ARSA standard assay = SapB deficiency until proven otherwise"},
    {"term": "Complete PSAP Deficiency — Neonatal Lethal", "definition": "Biallelic null PSAP variants eliminate ALL four saposins simultaneously → all four lysosomal sphingolipid pathways blocked → panorganic storage (galactosylceramide, sulfatide, glucosylceramide, ceramide all accumulate) → massive hepatosplenomegaly, brain atrophy, lung infiltration, renal failure; neonatal onset; fatal within weeks; prenatal diagnosis (PSAP sequencing from CVS/amniocentesis) = only effective intervention; extremely rare (<30 reported cases)"},
    {"term": "VPA SAFE — POLG1 Exclusion Mandatory", "definition": "VPA is NOT contraindicated in PSAP deficiency (ceramide/sphingolipid pathway — NOT mitochondrial disease); indicated for GTCS and PME-like seizures (Level B); MANDATORY: exclude POLG1 (mtDNA polymerase gamma) pathogenic variants before VPA initiation per CPIC-POLG1-2023 guideline (POLG1 + VPA = fulminant hepatic failure — Level A CI); PSAP patients unlikely to have POLG1 variants but standard screening required"},
    {"term": "ACTH Level A — IS in SapA and SapD", "definition": "ACTH (or high-dose prednisolone) is Level A evidence for infantile spasms (ACNS 2022/AES 2022 guidelines); first-line for IS in SapA (Krabbe-like IS with hypsarrhythmia) and SapD (Farber-like IS); vigabatrin NOT first-line in SapA (optic atrophy 85% at baseline — VGB visual field toxicity additive and unacceptable); ACTH preferred when pre-existing optic pathway disease"},
    {"term": "CBZ/OXC HIGH RISK (SapA) / RELATIVE-CI (SapB, SapC)", "definition": "Carbamazepine and oxcarbazepine worsen peripheral neuropathy via sodium channel toxicity — contraindicated in SapA (100% neuropathy, Krabbe-like) and HIGH RISK in SapB (90% neuropathy, MLD-like); RELATIVE-CI in SapC (action myoclonus worsening from sodium channel blockade, similar to GBA Type 3 absolute CI); acceptable in SapD only for isolated focal seizures without myoclonus (with EEG monitoring)"},
    {"term": "AR Biallelic LOF / 10q22.1", "definition": "PSAP inheritance: autosomal recessive; both PSAP alleles must carry pathogenic variants (biallelic LOF); chromosome 10q22.1; variants may be within saposin-specific exons (causing single saposin deficiency) or in shared prosaposin sequence (causing complete PSAP deficiency); heterozygous carriers asymptomatic; no pan-ethnic founder mutation (unlike HEXA AJ founder); molecular subtype analysis determines which saposin is deficient based on PSAP variant location"},
    {"term": "PSAP vs Primary Enzyme Deficiency — Diagnostic Algorithm", "definition": "When LSD phenotype (clinical + biochemical substrate accumulation) + NORMAL enzyme assay result: (1) First exclude pseudodeficiency alleles (GALC pseudodeficiency, ARSA pseudodeficiency); (2) Request PSAP gene sequencing (full gene, all saposin-encoding exons); (3) Request functional saposin-specific protein assay (SapA, SapB, SapC, SapD Western blot/ELISA separately); (4) Substrate biomarkers (psychosine for SapA, urine sulfatides for SapB, Lyso-Gb1 for SapC, ceramides for SapD); (5) Complete PSAP deficiency if all saposins absent"},
]


# ---------------------------------------------------------------------------
# API RESPONSE BUILDERS
# ---------------------------------------------------------------------------

def get_overview():
    return {
        "disease": (
            "Prosaposin deficiency (PSAP deficiency) is an autosomal-recessive lysosomal storage "
            "disorder family caused by biallelic loss-of-function variants in PSAP (10q22.1), "
            "encoding prosaposin — the precursor of four saposin activator proteins (A, B, C, D). "
            "Each saposin is required for lysosomal substrate presentation to its cognate enzyme: "
            "Saposin A activates GALC (Krabbe disease enzyme); Saposin B activates ARSA (MLD enzyme); "
            "Saposin C activates GBA (Gaucher disease enzyme); Saposin D activates ASAH1 (Farber "
            "disease enzyme). Deficiency of each saposin produces a phenocopy of the corresponding "
            "enzyme-deficiency disease despite a NORMAL enzyme assay result — the pathognomonic "
            "false-negative enzyme pattern unique to PSAP diseases. Five subtypes are recognised: "
            "SapA deficiency (Krabbe-phenocopy; GALC false negative; infantile leukodystrophy); "
            "SapB deficiency (MLD-phenocopy; ARSA false negative on standard assay; sulfatide "
            "storage; heat-inactivation assay diagnostic); SapC deficiency (Gaucher Type 3 "
            "phenocopy; GBA false negative; horizontal saccade palsy; action myoclonus); SapD "
            "deficiency (Farber Type 7; ASAH1 gene normal but AC activity <10%; ceramide accumulation); "
            "complete PSAP deficiency (neonatal lethal; all four pathways blocked; panorganic storage). "
            "No approved ERT for any saposin deficiency. HSCT Level C evidence for SapA (pre-symptomatic "
            "only, extrapolated from Krabbe NBS). ACTH Level A for IS. 40-patient cohort."
        ),
        "gene": "PSAP (Prosaposin / Sphingolipid Activator Protein Precursor) — 524 aa, ~65 kDa",
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "protein": (
            "Prosaposin (PSAP): 524 aa precursor (~65 kDa); synthesised in ER; transported to "
            "lysosomes; processed by cathepsins B, D into 4 saposins (A, B, C, D) separated by "
            "linker peptides. Each saposin: small (~66–78 aa), heat-stable, acid-stable; binds "
            "amphipathic lipids; presents substrate to cognate lysosomal hydrolase. Saposin A "
            "(66 aa) → GALC; Saposin B (78 aa) → ARSA; Saposin C (66 aa) → GBA; Saposin D "
            "(62 aa) → ASAH1. PSAP also secreted extracellularly as intact prosaposin (neurotrophic "
            "factor, myelinotrophic — reduces apoptosis via PI3K/Akt pathway; second function "
            "independent of lysosomal activator role)."
        ),
        "mechanism": (
            "PSAP LOF → deficient saposin(s) → cognate lysosomal hydrolase cannot process "
            "physiological lipid substrate in vivo: "
            "(SapA) GALC cannot cleave galactosylceramide/psychosine → psychosine accumulation "
            "→ oligodendrocyte death → Krabbe-like leukodystrophy + globoid cells; "
            "(SapB) ARSA cannot cleave sulfatide → sulfatide storage → Schwann cell and white "
            "matter destruction → MLD-like leukodystrophy + neuropathy; "
            "(SapC) GBA cannot cleave glucosylceramide → Gaucher cells in bone marrow/spleen → "
            "Gaucher Type 3-like disease with neurological involvement; "
            "(SapD) ASAH1 cannot cleave ceramide → ceramide accumulation → Farber-like "
            "lipogranulomatosis; (Complete) all four pathways blocked → panorganic failure. "
            "Standard enzyme assays use ARTIFICIAL fluorogenic substrates not requiring saposins "
            "→ enzyme protein is present + functional artificially → NORMAL assay = diagnostic false "
            "negative specific to saposin deficiencies."
        ),
        "cohort_size": COHORT_SIZE,
        "mean_onset_months": 6.8,
        "seizure_pct_overall": 62,
        "seizure_pct_sapa": 75,
        "seizure_pct_sapb": 60,
        "seizure_pct_sapc": 70,
        "seizure_pct_sapd": 80,
        "drug_resistant_pct": 65,
        "infantile_spasms_pct": 42,
        "peripheral_neuropathy_pct": 78,
        "leukodystrophy_pct": 72,
        "hepatosplenomegaly_pct": 45,
        "optic_atrophy_pct": 52,
        "horizontal_saccade_palsy_pct": 22,
        "on_acth_pct": 38,
        "on_lev_pct": 65,
        "on_vpa_pct": 42,
        "on_kd_pct": 22,
        "mean_diagnosis_delay_years": 3.1,
        "false_negative_enzyme_assay_pct": 72,
        "pathognomonic_note": (
            "ENZYME FALSE NEGATIVE (PATHOGNOMONIC, ~72%): Normal downstream lysosomal enzyme "
            "activity (GALC, ARSA, or GBA) on standard fluorometric assay despite clinical + "
            "biochemical LSD phenotype. This false-negative enzyme result is UNIQUE to PSAP "
            "saposin deficiency diseases — the only LSD category where the deficient enzyme "
            "tests normally. Diagnostic rule: LSD phenotype + normal enzyme assay = investigate "
            "PSAP/saposin deficiency first (after ruling out pseudodeficiency alleles)."
        ),
        "diagnostic_hierarchy": (
            "Step 1 — Suspect PSAP when: (a) clinical LSD phenotype (leukodystrophy/neuropathy/HSM/"
            "granulomas) + NORMAL enzyme assay; (b) normal GALC + Krabbe-like = SapA; "
            "(c) normal ARSA standard + MLD + sulfatiduria = SapB; (d) normal GBA + Gaucher cells "
            "+ HSP = SapC; (e) normal ASAH1 gene + AC <10% + ceramide = SapD. "
            "Step 2 — Substrate biomarkers (psychosine for SapA; urine sulfatides for SapB; "
            "Lyso-Gb1 for SapC; ceramides for SapD). Step 3 — PSAP gene sequencing + functional "
            "saposin-specific protein assays (Western blot for SapA/B/C/D individually)."
        ),
        "etiologies": ETIOLOGIES,
        "standards": STANDARDS[:6],
    }


def get_breakdown():
    return {
        "cohort_size": COHORT_SIZE,
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "saposin_enzyme_matrix": {
            "SapA": {
                "enzyme": "GALC (Galactocerebrosidase)",
                "substrate": "Galactosylceramide, Psychosine",
                "disease_phenocopy": "Krabbe Disease (Globoid Cell Leukodystrophy)",
                "enzyme_assay": "FALSE NEGATIVE (GALC activity normal on standard assay)",
                "biomarker": "Psychosine/galactosylsphingosine elevated",
                "specific_CI": "VGB (optic atrophy additive), CBZ/OXC (neuropathy additive), PHT (neuropathy)",
            },
            "SapB": {
                "enzyme": "ARSA (Arylsulfatase A)",
                "substrate": "Sulfatide (3-sulfo-galactosylceramide)",
                "disease_phenocopy": "Metachromatic Leukodystrophy (MLD)",
                "enzyme_assay": "FALSE NEGATIVE (standard assay); heat-inactivation assay diagnostic",
                "biomarker": "Urine sulfatides 50–200× normal",
                "specific_CI": "CBZ/OXC (neuropathy, RELATIVE-CI), PHT (neuropathy, HIGH RISK), LTG (acceptable adjunct)",
            },
            "SapC": {
                "enzyme": "GBA (Acid Beta-Glucosidase / Glucocerebrosidase)",
                "substrate": "Glucosylceramide (glucocerebroside)",
                "disease_phenocopy": "Gaucher Disease Type 3 (neuronopathic)",
                "enzyme_assay": "FALSE NEGATIVE (GBA activity normal on standard assay)",
                "biomarker": "Lyso-Gb1 (glucosylsphingosine) elevated",
                "specific_CI": "CBZ/OXC/PHT (myoclonus worsening, RELATIVE-CI), LTG (myoclonus worsening), GBP/PGB (ataxia)",
            },
            "SapD": {
                "enzyme": "ASAH1 (Acid Ceramidase)",
                "substrate": "Ceramide (N-acylsphingosine)",
                "disease_phenocopy": "Farber Disease Type 7 (lipogranulomatosis)",
                "enzyme_assay": "AC activity <10% (NOT false negative — SapD required for AC assay function)",
                "biomarker": "Plasma ceramides (C16:0, C18:0) elevated",
                "specific_CI": "Typical antipsychotics (ceramide additive HIGH RISK), CBZ/OXC (RELATIVE-CI PME)",
            },
        },
        "treatment_hierarchy": [
            "1. IS (SapA/SapD): ACTH Level A → (LEV adjunct Level B) → clobazam Level C; NO VGB in SapA (optic atrophy)",
            "2. Focal seizures: LEV Level B → clobazam C (adjunct) → VPA B (with hepatic monitoring + POLG1 exclusion)",
            "3. Myoclonus/PME-like (SapC, SapA): VPA Level B + piracetam Level C (adjunct) → LEV B → clonazepam C adjunct",
            "4. GTCS: LEV Level B → VPA Level B → clobazam C adjunct; AVOID CBZ/OXC (SapA/SapB) and LTG (SapC)",
            "5. Drug-resistant: KD Level B → clinical trial referral; HSCT Level C (SapA pre-symptomatic only via NBS)",
            "6. SE: IV LEV first-line; IV VPA second-line; AVOID IV fosphenytoin (neuropathy risk SapA/SapB)",
        ],
        "subtype_seizure_summary": {
            "SapA": "75% seizures; IS (55%) + PME-like myoclonus (55%) + tonic (40%) + focal (45%); drug-resistant 70%",
            "SapB": "60% seizures; focal (45%) + myoclonic (40%) + GTCS (30%) + absence-like (20%); drug-resistant 55%",
            "SapC": "70% seizures; action myoclonus (60%) + GTCS (50%) + focal (40%) + horizontal saccade palsy (80%); drug-resistant 65%",
            "SapD": "80% seizures; IS (65%) + myoclonic (45%) + focal (35%); drug-resistant 70%",
            "Complete PSAP": "10% seizures; neonatal, fatal before sustained seizure development",
        },
    }


def get_definitions():
    return {
        "cohort_size": COHORT_SIZE,
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "key_concepts": KEY_CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "diagnostic_algorithm": [
            "Step 1: Clinical suspicion — LSD phenotype (leukodystrophy/neuropathy/HSM/granulomas) + NORMAL enzyme assay = suspect PSAP saposin deficiency",
            "Step 2: Substrate biomarker panel — psychosine (SapA), urine sulfatides (SapB), Lyso-Gb1/glucosylsphingosine (SapC), plasma ceramides (SapD)",
            "Step 3: Exclude pseudodeficiency alleles (GALC pseudodeficiency, ARSA pseudodeficiency p.Asn350Ser + p.Ile181Thr in CIS) — pseudodeficiency ≠ clinical disease; requires confirmatory genotyping",
            "Step 4: SapB-specific — heat-inactivation ARSA assay (55°C, 15 min) — reveals ARSA-B deficiency in saposin B deficiency; standard assay alone insufficient",
            "Step 5: PSAP gene sequencing (NGS full-gene panel including all saposin-encoding exons and linker regions) — identifies biallelic pathogenic PSAP variants",
            "Step 6: Functional saposin protein assays — Western blot or ELISA for SapA, SapB, SapC, SapD individually — confirm which saposin(s) absent/severely reduced",
            "Step 7: Subtype classification → clinical management (HSCT candidacy for SapA; ACTH for IS in SapA/SapD; AED selection per subtype-specific CI profile)",
        ],
        "saposin_pathway_glossary": {
            "Prosaposin (PSAP)": "Lysosomal precursor protein (524 aa, ~65 kDa); cleaved into 4 saposins (A, B, C, D); also secreted extracellularly as neuroprotective myelinotrophic factor",
            "Saposin A": "Activates GALC; binds galactosylceramide and psychosine; deficiency → Krabbe-like disease with GALC false-negative enzyme assay",
            "Saposin B": "Activates ARSA; binds sulfatide; deficiency → MLD-like disease with ARSA false-negative standard assay (heat-inactivation assay diagnostic)",
            "Saposin C": "Activates GBA; binds glucosylceramide; deficiency → Gaucher Type 3-like disease with GBA false-negative assay; Lyso-Gb1 elevated",
            "Saposin D": "Activates ASAH1; binds ceramide; deficiency → Farber-like disease (Farber Type 7); ASAH1 gene normal; AC activity <10%",
            "False-negative enzyme assay": "Standard lysosomal enzyme assays use artificial fluorogenic substrates not requiring saposin activation → enzyme activity appears normal despite in-vivo substrate accumulation; PATHOGNOMONIC for saposin deficiency",
            "Heat-inactivation ARSA assay": "Incubation at 55°C for 15 min eliminates thermolabile ARSA-B isoform → unmasks SapB deficiency; required for SapB diagnosis",
            "Psychosine (galactosylsphingosine)": "Toxic substrate catabolised by GALC+SapA; elevated in SapA deficiency (same as Krabbe disease); neurotoxic to oligodendrocytes",
            "Lyso-Gb1 (glucosylsphingosine)": "Substrate catabolised by GBA+SapC; elevated in SapC deficiency (same as Gaucher disease); biomarker for SapC deficiency monitoring",
        },
    }
