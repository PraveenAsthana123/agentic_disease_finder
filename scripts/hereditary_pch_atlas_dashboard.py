#!/usr/bin/env python3
"""Hereditary-PCH-Atlas — Complete 8-Gene Pontocerebellar Hypoplasia Atlas
(PCH2A/TSEN54 · PCH2B/TSEN2 · PCH6/RARS2 · PCH1B/EXOSC3 ·
 PCH1B-SMA/VRK1 · PCH3/CASK · PCH9/AMPD2 · PCH7/TOE1).

TSEN54   (tRNA Splicing Endonuclease Subunit 54; 576 aa; 17q25.1; AR;
          PCH2A (most common PCH gene worldwide) / PCH4 (severe);
          DRAGONFLY WING MRI PATTERN — flattened cerebellar hemispheres + preserved vermis PATHOGNOMONIC for PCH2;
          A307S / A307S homozygote = PCH2A (classic); nonsense → PCH4 (severe, neonatal death);
          seed SEED_BASE+0).
TSEN2    (tRNA Splicing Endonuclease Subunit 2; 483 aa; 3p25.1; AR;
          PCH2B — clinically indistinguishable from PCH2A/TSEN54;
          Same TSEN complex subunit; compound het common;
          seed SEED_BASE+1).
RARS2    (Mitochondrial Arginyl-tRNA Synthetase; 576 aa; 6q15; AR;
          PCH6 — ABSENT CEREBELLAR VERMIS PATHOGNOMONIC (PCH6 > PCH2);
          Elevated lactate/pyruvate (mito dysfunction); severe epilepsy neonatal onset;
          seed SEED_BASE+2).
EXOSC3   (RNA Exosome Component 3 / RRP40; 275 aa; 9p13.2; AR;
          PCH1B (RNA exosome subtype) — MOTOR NEURON DISEASE concurrent PATHOGNOMONIC (SMA-like + cerebellar);
          D132A founder mutation (Polish/Roma); slow progression vs PCH2;
          seed SEED_BASE+3).
VRK1     (Vaccinia-Related Kinase 1; 396 aa; 14q32.2; AR;
          PCH1B (SMA-PCH overlap) — SMA PHENOTYPE + CEREBELLAR HYPOPLASIA PATHOGNOMONIC;
          Anterior horn cell degeneration + cerebellar; survival motor neuron overlap DDx;
          seed SEED_BASE+4).
CASK     (Membrane-Associated Guanylate Kinase / CASK; 922 aa; Xp11.4; XL-Dom(F)/hemi(M);
          PCH3 / Microcephaly-Hypotonia-Severe-Epilepsy-CASK-Syndrome;
          FEMALE: variable (dominant het — mosaicism varies severity); MALE: neonatal lethal (hemizygous);
          FGF14-CASK synaptic module defect; seed SEED_BASE+5).
AMPD2    (AMP Deaminase 2; 900 aa; 1p13.3; AR;
          PCH9 — HYPOMYELINATION + CEREBELLAR ATROPHY + OPTIC ATROPHY TRIAD PATHOGNOMONIC;
          Purine cycle defect; elevated SAICAR on metabolomics; AdoMetDC elevation;
          seed SEED_BASE+6).
TOE1     (Target of EGR1 / Deadenylase TOE1; 421 aa; 1p34.1; AR;
          PCH7 — GONADAL DYSGENESIS (46,XY DSD) + CEREBELLAR HYPOPLASIA PATHOGNOMONIC;
          snRNA 3'-end processing defect; vermian > hemispheral hypoplasia;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2150-2157).
"""

import random

SEED_BASE = 2150

PCH_GENES = [
    # -- TSEN54 — PCH2A / PCH4 ------------------------------------------------
    {
        "gene": "TSEN54",
        "alt_name": (
            "TSEN54 (TSEN54-576aa-17q25.1 / AR — PCH2A-Pontocerebellar-Hypoplasia-Type-2A — "
            "DRAGONFLY-WING-MRI-FLATTENED-HEMISPHERES-PRESERVED-VERMIS-PATHOGNOMONIC — "
            "A307S-HOMOZYGOTE-MOST-COMMON-PCH2A — PCH4-NONSENSE-SEVERE-NEONATAL-DEATH)"
        ),
        "protein": (
            "TSEN54 -- 17q25.1 AR -- TSEN54-576aa -- "
            "tRNA-Splicing-Endonuclease-Subunit-54-Catalytic-Subunit-TSEN-Complex -- "
            "PCH2A-Pontocerebellar-Hypoplasia-Type-2A-OMIM-277470 -- "
            "PCH4-Pontocerebellar-Hypoplasia-Type-4-OMIM-225753 -- "
            "TSEN-Complex-4-Subunits-TSEN2-TSEN15-TSEN34-TSEN54-tRNA-Intron-Removal -- "
            "DRAGONFLY-WING-MRI-PATTERN-FLATTENED-Cerebellar-Hemispheres-With-Preserved-Vermis-PATHOGNOMONIC -- "
            "A307S-c919GA-Missense-Most-Common-PCH2A-Worldwide-Hypomorphic-TSEN-Complex -- "
            "PCH4-Nonsense-Frameshift-Null-Alleles-Neonatal-Death-Absent-Cerebellum-Olivopontocerebellar -- "
            "Neonatal-Hypotonia-Epilepsy-Intellectual-Disability-Progressive-Microcephaly -- "
            "Absent-Voluntary-Movements-Hyperreflexia-Extrapyramidal-Signs -- "
            "PCH2-MOST-COMMON-TYPE-Most-Genes-AR-Compound-Het-Common -- "
            "17q25.1"
        ),
        "locus": "17q25.1",
        "protein_size": "576 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "A307S/A307S homozygote = PCH2A classic (most common worldwide); "
            "A307S / null = PCH2A moderate; "
            "null/null = PCH4 (severe — neonatal death, absent cerebellum); "
            "GENETIC TESTING: WES or targeted NGS panel — MLPA not routinely needed; "
            "de novo rate low; consanguinity risk factor in PCH4"
        ),
        "age_of_onset": "Neonatal / early infancy (congenital cerebellar hypoplasia)",
        "pathognomonic": (
            "DRAGONFLY WING MRI PATTERN on axial view: flattened cerebellar hemispheres with "
            "preserved vermis — PATHOGNOMONIC for PCH2 family (TSEN genes); "
            "Progressive microcephaly post-natal (acquired not congenital); "
            "Neonatal hypotonia → later hypertonia + extrapyramidal features; "
            "Seizures (infantile spasms / myoclonic / tonic) — early, often refractory; "
            "A307S compound het or homozygote: TSEN54 sequencing MANDATORY in any PCH2 phenotype; "
            "PCH4: absent cerebellar folia + olivopontocerebellar hypoplasia on MRI (more severe)"
        ),
        "treatment": (
            "NO disease-modifying therapy; supportive care paramount; "
            "ANTI-EPILEPTIC: ACTH for infantile spasms (Level A); "
            "VGB (vigabatrin) for IS — MANDATORY visual field monitoring; "
            "LEV or VPA for myoclonic seizures (avoid VPA if POLG overlap concern); "
            "FEEDING: NG tube or PEG gastrostomy for dysphagia — 70% require by year 2; "
            "RESPIRATORY: CPAP/BiPAP for obstructive sleep apnoea; "
            "Tracheostomy consideration in PCH4 (neonatal respiratory failure); "
            "PHYSIOTHERAPY: tone management, positioning, contracture prevention; "
            "HYDROTHERAPY: spasticity + tone; "
            "KETOGENIC DIET: considered for refractory epilepsy (Level C); "
            "COMMUNICATION: AAC (augmentative/alternative) early introduction; "
            "OPHTHALMOLOGY: nystagmus management, visual stimulation"
        ),
        "contraindications": (
            "AVOID VGB PROLONGED USE without serial visual field testing (retinal toxicity); "
            "AVOID CARBAMAZEPINE in myoclonic epilepsy (worsens myoclonus); "
            "AVOID VIGABATRIN IN TSEN54-PCH4 (respiratory instability risk); "
            "CAUTION BENZODIAZEPINES CHRONIC: respiratory depression in hypotonic patients; "
            "AVOID VPA IN POLG1-POSITIVE CASES: hepatotoxicity; "
            "DO NOT DEFER PEG: nutritional failure worsens cognitive outcomes"
        ),
        "monitoring": (
            "MRI BRAIN: 6-monthly in first 2 years (progressive microcephaly + cerebellar atrophy); "
            "EEG: 3-monthly for seizure classification and IS surveillance; "
            "OPHTHALMOLOGY: 6-monthly (nystagmus, optic atrophy, VGB retinal toxicity annual ERG); "
            "FEEDING ASSESSMENT: 3-monthly (SLT + dietitian); "
            "RESPIRATORY: sleep study 6-monthly (OSA, hypoventilation); "
            "DEVELOPMENTAL: Bayley/GMFM 6-monthly; "
            "TSEN54 FAMILY CASCADE: siblings at-risk — offer prenatal testing; "
            "METABOLIC: lactate, ammonia (exclude mito overlap at diagnosis); "
            "HEAD CIRCUMFERENCE: monthly (progressive post-natal microcephaly key diagnostic marker)"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): hypotonia, poor suck, seizure onset, MRI diagnosis, feeding/respiratory support",
            "Infancy (1-12 mo): IS onset, PEG consideration, tone evolving, progressive microcephaly",
            "Toddler (1-3 yr): extrapyramidal signs, dystonia, severe GDD, seizure optimisation",
            "Childhood (3-12 yr): plateau of skills, dysphagia, spasticity management, AAC",
            "Adolescence (12-18 yr): scoliosis, respiratory decline, palliative care planning",
            "Adult (18+ yr): rare survival; severe disability; APC / palliative care",
        ],
        "concepts": [
            "PCH2A: most common PCH gene worldwide (TSEN54)",
            "Dragonfly wing MRI: flattened hemispheres + preserved vermis = PATHOGNOMONIC PCH2",
            "A307S: most common PCH2A variant (hypomorphic TSEN complex)",
            "PCH4: null/null TSEN54 → neonatal death + absent cerebellar folia",
            "TSEN complex: 4-subunit tRNA splicing machine (TSEN2/15/34/54)",
            "Progressive microcephaly: acquired post-natal in PCH2 (NOT congenital)",
            "Extrapyramidal signs: dyskinesia/dystonia — distinguish from CP",
            "Infantile spasms: early seizure type requiring ACTH first-line",
            "PCH phenotype spectrum: PCH1 (SMA-like) vs PCH2 (cerebellar predominant)",
            "Compound heterozygosity: A307S + null = intermediate severity",
            "Dragonfly on axial MRI: most specific PCH2 imaging biomarker",
            "Prenatal diagnosis: MRI and molecular — critical for family counselling",
            "PEG timing: early before 6 months prevents nutritional collapse",
            "VGB visual toxicity: ERG annual mandatory with vigabatrin use",
            "Palliative care: integrate from diagnosis in severe PCH2/PCH4",
        ],
        "thresholds": [
            "Head circumference <3rd centile: progressive microcephaly — monitor monthly",
            "Cerebellar hemisphere AP diameter <15 mm on MRI at term: severe hypoplasia",
            "IS onset <6 months: ACTH initiation within 2 weeks reduces neurocognitive harm",
            "VGB duration >2 years: annual ERG for retinal toxicity mandatory",
            "PEG if safe oral intake <80% by 6 months: early gastrostomy",
            "SpO2 <92% at night: BiPAP initiation",
            "Seizure duration >5 min: rescue diazepam rectal/buccal protocol",
            "Lactate >2.2 mmol/L: mito overlap workup (RARS2 / mtDNA)",
            "A307S allele frequency >1/400 in some European populations",
            "PCH4 life expectancy <3 months without intensive support",
            "TSEN54 panel positive: sibling risk 25% — prenatal offer mandatory",
            "Dragonfly MRI specificity ~85% for PCH2 family in expert hands",
        ],
        "standards": [
            "ILAE-2022 epilepsy classification",
            "UKISS-2004 infantile spasm protocol",
            "ACMG-AMP-2015 variant classification",
            "PCH-Network-European-Consensus-2014",
            "WHO-ICF-2019 disability framework",
            "NICE-NG-Epilepsy-Children",
            "Barth-2012-EJoPaed PCH classification",
            "ESPGHAN-Enteral-Nutrition-Paed-2017 (PEG guidance)",
            "VGB-CPMP-Visual-Field-Guidance",
            "EFNS-Cerebellar-Ataxia-Guidelines",
            "ACR-Appropriateness-Criteria-Paed-Brain-MRI",
            "SIMD-Metabolic-Newborn-Screen",
        ],
        "etiologies": [
            {"type": "TSEN54 A307S/A307S — PCH2A Classic", "pct": 45},
            {"type": "TSEN54 A307S/null — PCH2A Intermediate", "pct": 22},
            {"type": "TSEN54 null/null — PCH4 Severe Neonatal", "pct": 12},
            {"type": "TSEN54 Missense/Missense (other) — PCH2A Mild", "pct": 14},
            {"type": "TSEN54 Phenocopy (other TSEN/PCH genes)", "pct": 7},
        ],
        "seizure_types": [
            {"type": "Infantile Spasms (IS)", "pct": 72},
            {"type": "Myoclonic Seizures", "pct": 58},
            {"type": "Tonic Seizures", "pct": 45},
            {"type": "Focal Seizures with Secondary Generalisation", "pct": 32},
            {"type": "Absence-Like Seizures", "pct": 18},
        ],
        "triggers": [
            {"trigger": "Fever / Intercurrent Infection", "pct": 88},
            {"trigger": "Sleep Deprivation", "pct": 72},
            {"trigger": "Missed AED Dose", "pct": 65},
            {"trigger": "Hyperthermia (bath, weather)", "pct": 55},
            {"trigger": "Tactile Stimulation (startles)", "pct": 45},
            {"trigger": "Metabolic Stress (illness)", "pct": 40},
            {"trigger": "Excitement / Arousal", "pct": 30},
            {"trigger": "Feeding Stress", "pct": 25},
        ],
        "references": [
            "Budde BS 2008 Nat Genet (TSEN54 PCH2A)",
            "Barth PG 2012 Eur J Paediatr Neurol (PCH classification)",
            "Cassandrini D 2010 Eur J Hum Genet (TSEN cohort)",
            "Namavar Y 2011 Brain (PCH genotype-phenotype)",
            "Braun K 2016 Dev Med Child Neurol (management)",
            "Poretti A 2017 J Child Neurol (MRI patterns PCH)",
        ],
    },
    # -- TSEN2 — PCH2B ---------------------------------------------------------
    {
        "gene": "TSEN2",
        "alt_name": (
            "TSEN2 (TSEN2-483aa-3p25.1 / AR — PCH2B-Pontocerebellar-Hypoplasia-Type-2B — "
            "SAME-TSEN-COMPLEX-AS-TSEN54-CLINICALLY-INDISTINGUISHABLE-PATHOGNOMONIC — "
            "DRAGONFLY-WING-MRI-IDENTICAL-PCH2A-MOLECULAR-DIAGNOSIS-DISTINGUISHES)"
        ),
        "protein": (
            "TSEN2 -- 3p25.1 AR -- TSEN2-483aa -- "
            "tRNA-Splicing-Endonuclease-Subunit-2-Active-Site-Endonuclease -- "
            "PCH2B-Pontocerebellar-Hypoplasia-Type-2B-OMIM-612389 -- "
            "TSEN-Complex-Subunit-Same-Machine-As-TSEN54-PCH2A -- "
            "DRAGONFLY-WING-MRI-IDENTICAL-TO-PCH2A-CLINICAL-SEPARATION-REQUIRES-MOLECULAR -- "
            "Compound-Heterozygous-Most-Common-Less-Frequent-Than-TSEN54 -- "
            "Neonatal-Hypotonia-Progressive-Microcephaly-Severe-Epilepsy -- "
            "Extrapyramidal-Dystonia-Dyskinesia-Spasticity -- "
            "TSEN-PANEL-MANDATORY-Not-Just-TSEN54-When-PCH2-Phenotype -- "
            "3p25.1"
        ),
        "locus": "3p25.1",
        "protein_size": "483 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "compound heterozygous most common; "
            "phenotype indistinguishable from TSEN54-PCH2A clinically; "
            "GENETIC TESTING: full TSEN panel (TSEN2/15/34/54) required — not TSEN54 alone; "
            "no specific founder mutation; consanguinity in severe cases"
        ),
        "age_of_onset": "Neonatal / early infancy (congenital)",
        "pathognomonic": (
            "SAME DRAGONFLY WING MRI as TSEN54-PCH2A — flattened cerebellar hemispheres + preserved vermis; "
            "Clinically INDISTINGUISHABLE from PCH2A without molecular testing; "
            "TSEN2 diagnosis requires full TSEN complex panel — TSEN54 testing alone will miss; "
            "Progressive post-natal microcephaly (same as PCH2A); "
            "Neonatal hypotonia → extrapyramidal features; "
            "Infantile spasms + myoclonus (same as TSEN54); "
            "KEY: suspect TSEN2 when PCH2 phenotype + TSEN54 negative"
        ),
        "treatment": (
            "IDENTICAL management to TSEN54-PCH2A (same TSEN complex defect); "
            "ANTI-EPILEPTIC: ACTH for IS (Level A); LEV for myoclonic; "
            "FEEDING: PEG gastrostomy early (<6 months); "
            "RESPIRATORY: CPAP/BiPAP; "
            "PHYSIOTHERAPY: tone management; "
            "KETOGENIC DIET: refractory epilepsy (Level C); "
            "PALLIATIVE CARE: integrate from diagnosis"
        ),
        "contraindications": (
            "AVOID CBZ/OXC in myoclonic predominant epilepsy; "
            "AVOID VGB without visual monitoring (ERG); "
            "CAUTION BENZODIAZEPINES: chronic respiratory depression"
        ),
        "monitoring": (
            "MRI BRAIN: 6-monthly (progression monitoring); "
            "EEG: 3-monthly; "
            "OPHTHALMOLOGY: 6-monthly; "
            "HEAD CIRCUMFERENCE: monthly; "
            "FEEDING SLT: 3-monthly; "
            "RESPIRATORY SLEEP STUDY: 6-monthly; "
            "TSEN2 family cascade: 25% sibling risk"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): hypotonia, seizure onset, MRI, feeding support",
            "Infancy (1-12 mo): IS, PEG, progressive microcephaly",
            "Toddler (1-3 yr): dystonia, spasticity, GDD",
            "Childhood (3-12 yr): plateau, AAC, scoliosis",
            "Adolescence (12-18 yr): respiratory decline, palliative",
            "Adult (18+): rare; APC / palliative",
        ],
        "concepts": [
            "PCH2B: TSEN2 mutation — same complex as TSEN54",
            "TSEN panel: test all 4 subunits (TSEN2/15/34/54) for complete PCH2 workup",
            "Dragonfly wing: identical MRI to PCH2A — only molecular distinguishes",
            "TSEN2 less common than TSEN54 globally",
            "Compound het: most common TSEN2 genotype",
            "PCH2B clinically indistinguishable from PCH2A",
            "TSEN complex: all 4 subunits assemble tRNA splicing machine",
            "Post-natal progressive microcephaly: shared feature all TSEN-PCH",
            "Epilepsy management: identical to PCH2A/TSEN54",
            "PEG early: prevents nutritional failure in both PCH2A/2B",
            "Family cascade: 25% sibling risk — prenatal diagnosis offer",
            "TSEN15 (1q25.2) and TSEN34 (19q13.42) complete the TSEN panel",
            "TSEN2 variants: database Leiden Open Variation Database",
            "Severity overlap: TSEN54-PCH4-equivalent rare with TSEN2 null",
            "Gene panel sequencing preferred over sanger for TSEN genes",
        ],
        "thresholds": [
            "TSEN54 negative + PCH2 phenotype: order full TSEN panel",
            "Head circumference <-3 SD: progressive microcephaly monitoring",
            "IS onset <6 months: ACTH within 2 weeks",
            "Safe oral feeding <80%: PEG referral",
            "SpO2 <92% nocturnal: BiPAP",
            "TSEN2 panel sensitivity ~95% when full TSEN2/15/34/54 tested",
            "Sibling risk 25%: prenatal offer mandatory",
            "MRI timing: MRI at 3 months postnatal for PCH2 confirmation",
            "EEG hypsarrhythmia: IS confirmed — ACTH start",
            "Lactate normal in TSEN2-PCH2B (unlike RARS2-PCH6)",
            "Post-natal head growth deceleration >2 SD in 3 months: alarm",
            "TSEN2 gene panel cost vs WES: both appropriate",
        ],
        "standards": [
            "ILAE-2022 epilepsy classification",
            "ACMG-AMP-2015 variant classification",
            "PCH-Network-European-Consensus-2014",
            "UKISS-2004 infantile spasm protocol",
            "WHO-ICF-2019",
            "NICE-NG-Epilepsy-Children",
            "ESPGHAN-2017 enteral nutrition",
            "EFNS-Cerebellar-Ataxia-Guidelines",
            "VGB-CPMP-Visual-Field-Guidance",
            "ACR-Appropriateness-Criteria-Paed-MRI",
            "SIMD-Metabolic-Screen",
            "Barth-2012-PCH-Classification",
        ],
        "etiologies": [
            {"type": "TSEN2 Compound Het — PCH2B Classic", "pct": 55},
            {"type": "TSEN2 Homozygous Missense — Mild PCH2B", "pct": 20},
            {"type": "TSEN2 Null/Null — PCH2B Severe", "pct": 10},
            {"type": "TSEN2 Missense/Null — Intermediate", "pct": 10},
            {"type": "PCH2 Phenocopy (TSEN34/TSEN15)", "pct": 5},
        ],
        "seizure_types": [
            {"type": "Infantile Spasms (IS)", "pct": 68},
            {"type": "Myoclonic Seizures", "pct": 55},
            {"type": "Tonic Seizures", "pct": 42},
            {"type": "Focal Seizures", "pct": 28},
            {"type": "Absence-Like", "pct": 15},
        ],
        "triggers": [
            {"trigger": "Fever", "pct": 85},
            {"trigger": "Sleep Deprivation", "pct": 70},
            {"trigger": "Missed AED", "pct": 60},
            {"trigger": "Hyperthermia", "pct": 50},
            {"trigger": "Tactile Stimulation", "pct": 40},
            {"trigger": "Metabolic Stress", "pct": 38},
            {"trigger": "Excitement", "pct": 28},
            {"trigger": "Feeding Stress", "pct": 22},
        ],
        "references": [
            "Budde BS 2008 Nat Genet (TSEN2 PCH2B)",
            "Namavar Y 2011 Brain (TSEN cohort)",
            "Barth PG 2012 EJPN (PCH classification)",
            "Cassandrini D 2010 EJHG (TSEN panel)",
            "Kraoua I 2019 Eur J Paed Neurol (TSEN2 Tunisia)",
            "Poretti A 2017 J Child Neurol (MRI PCH)",
        ],
    },
    # -- RARS2 — PCH6 ----------------------------------------------------------
    {
        "gene": "RARS2",
        "alt_name": (
            "RARS2 (RARS2-576aa-6q15 / AR — PCH6-Pontocerebellar-Hypoplasia-Type-6-Mitochondrial — "
            "ABSENT-CEREBELLAR-VERMIS-PATHOGNOMONIC-PCH6->PCH2 — "
            "ELEVATED-LACTATE-PYRUVATE-MITO-DYSFUNCTION-PATHOGNOMONIC — "
            "SEVERE-NEONATAL-EPILEPSY-MTDNA-DEPLETION-RESPIRATORY-FAILURE)"
        ),
        "protein": (
            "RARS2 -- 6q15 AR -- RARS2-576aa -- "
            "Mitochondrial-Arginyl-tRNA-Synthetase-Class-IIb-ARS-Family-Mitochondrial -- "
            "PCH6-Pontocerebellar-Hypoplasia-Type-6-Mitochondrial-OMIM-611523 -- "
            "Mito-tRNA-Charging-Arg-mt-tRNAArg-Translation-Defect -- "
            "ABSENT-CEREBELLAR-VERMIS-PATHOGNOMONIC-MORE-SEVERE-THAN-PCH2 -- "
            "Elevated-Lactate-Pyruvate-Mito-Respiratory-Chain-Deficiency-BG-Metabolomics -- "
            "Neonatal-Lactic-Acidosis-Intractable-Seizures-Respiratory-Failure -- "
            "mtDNA-Depletion-Liver-Muscle-Variable-Percent -- "
            "ARS-Family-Gene-ARS-Panel-If-Elevated-Lactate-Plus-PCH -- "
            "VPA-ABSOLUTE-CONTRAINDICATION-Mito-Disease -- "
            "6q15"
        ),
        "locus": "6q15",
        "protein_size": "576 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "compound heterozygous or homozygous missense; "
            "PCH6 = most severe PCH type (absent vermis); "
            "GENETIC TESTING: mito ARS panel or WES — RARS2 sequencing + CNV; "
            "mtDNA depletion on tissue (liver/muscle): 20-50% of cases; "
            "elevated lactate mandates mito workup before assuming 'standard PCH'"
        ),
        "age_of_onset": "Neonatal (severe lactic acidosis + seizures at birth)",
        "pathognomonic": (
            "ABSENT or SEVERELY HYPOPLASTIC CEREBELLAR VERMIS on MRI — PATHOGNOMONIC for PCH6 "
            "(more severe than PCH2 dragonfly pattern); "
            "Elevated CSF/plasma LACTATE and PYRUVATE — mito respiratory chain deficiency; "
            "Neonatal intractable seizures + lactic acidosis + respiratory failure; "
            "mtDNA depletion in liver/muscle (variable); "
            "VPA ABSOLUTE CONTRAINDICATED — fatal hepatotoxicity (mito disease); "
            "ARS gene family panel positive: RARS2 on mitochondrial ARS panel; "
            "DDx from TSEN-PCH2: lactate elevated → RARS2; normal lactate → TSEN"
        ),
        "treatment": (
            "ANTI-EPILEPTIC: LEV first-line (mito-safe); LTG adjunct; "
            "VALPROATE ABSOLUTE CONTRAINDICATED — mito hepatotoxicity FATAL; "
            "MITO COCKTAIL: CoQ10 (10-30 mg/kg/day) + riboflavin + L-carnitine (evidence weak); "
            "LACTIC ACIDOSIS: IV sodium bicarbonate acute crisis; "
            "DICHLOROACETATE (DCA): investigational in severe lactic acidosis; "
            "FEEDING: NG/PEG early — dysphagia severe; "
            "RESPIRATORY: ventilatory support / tracheostomy in PCH6 severe; "
            "HIGH CALORIC SUPPORT: mito disease increases energy demands; "
            "PALLIATIVE CARE: early from diagnosis (most PCH6 neonatal-lethal)"
        ),
        "contraindications": (
            "VALPROATE/VPA ABSOLUTE CONTRAINDICATION: mito hepatotoxicity FATAL in RARS2; "
            "AVOID PHENOBARBITAL HIGH DOSE: mito complex inhibition; "
            "AVOID LINEZOLID: mito ribosome inhibitor (acute DION risk); "
            "AVOID AMINOGLYCOSIDES: mito ribosomal toxicity; "
            "AVOID METFORMIN: lactic acidosis worsening; "
            "AVOID STATINS: CoQ10 depletion in already CoQ10-deficient state; "
            "AVOID PROLONGED FASTING: triggers metabolic decompensation"
        ),
        "monitoring": (
            "LACTATE/PYRUVATE: 3-monthly (metabolic stability); "
            "LFT: monthly (hepatopathy risk even without VPA); "
            "mtDNA COPY NUMBER in blood: 6-monthly; "
            "RESPIRATORY: sleep study + spirometry 3-monthly (ventilatory decline); "
            "EEG: 4-weekly (intractable epilepsy); "
            "MRI BRAIN: 6-monthly (progressive atrophy); "
            "OPHTHALMOLOGY: 6-monthly (optic atrophy in PCH6); "
            "METABOLOMICS: amino acids, acylcarnitines at diagnosis; "
            "ECHO: cardiac mito myopathy surveillance 12-monthly; "
            "FEEDING/NUTRITION: 3-monthly SLT + dietitian"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): lactic acidosis, seizures, respiratory failure, ICU",
            "Infancy (1-6 mo): intractable epilepsy, mito cocktail, PEG, ventilation",
            "Infancy (6-12 mo): palliative care vs aggressive support decision",
            "Toddler (1-3 yr): rare survivors — severe disability, home ventilation",
            "Childhood: exceptional survival; severe GDD, mito disease management",
            "Adult: not described; neonatal/infantile lethal in most",
        ],
        "concepts": [
            "PCH6: mitochondrial arginyl-tRNA synthetase defect → mito translation failure",
            "Absent vermis on MRI: PCH6 PATHOGNOMONIC (more severe than PCH2 dragonfly)",
            "Elevated lactate: always check in PCH — separates RARS2 from TSEN genes",
            "ARS gene family: RARS2, AARS2, DARS2, EARS2 — all cause mito neuropathy/encephalopathy",
            "VPA absolute CI: mandatory rule in any mito disease (fatal hepatotoxicity)",
            "mtDNA depletion: tissue-specific — liver/muscle most affected",
            "LEV mito-safe: preferred AED in suspected mito disease",
            "CoQ10 cocktail: weak evidence but standard of care empirically",
            "Lactic acidosis crisis: sodium bicarbonate + thiamine IV",
            "DCA (dichloroacetate): pyruvate dehydrogenase activator — severe LA",
            "Mito ARS panel: RARS2 + AARS2 + DARS2 + EARS2 + LARS2 for comprehensive coverage",
            "PCH6 prognosis: worst of all PCH types — early death common",
            "DDx from TSEN-PCH: lactate key biomarker (elevated = RARS2, normal = TSEN)",
            "Prenatal diagnosis: chorionic villus sampling for RARS2 biallelic",
            "Aminoglycosides: prohibit — mito ribosomal toxicity additive with RARS2 defect",
        ],
        "thresholds": [
            "Lactate >2.2 mmol/L: mito workup mandatory (RARS2 suspect)",
            "Lactate >5 mmol/L: acute lactic acidosis crisis — bicarbonate IV",
            "LFT ALT >3x ULN: hepatopathy alert — VPA absolute stop (already CI)",
            "mtDNA <30% of age-matched control: significant depletion",
            "Absent vermis on MRI: high specificity for PCH6 family",
            "SpO2 <90% waking: respiratory failure — ICU escalation",
            "Seizure duration >5 min: buccal midazolam protocol",
            "CoQ10 dose: 10-30 mg/kg/day in 3 divided doses (standard mito cocktail)",
            "Riboflavin 50-200 mg/day: empirical mito supplementation",
            "L-carnitine 100 mg/kg/day: empirical (controversial in PCH6)",
            "RARS2 panel sensitivity: >95% if sequencing + deletion analysis",
            "Aminoglycoside use: CONTRAINDICATED — threshold = never use",
        ],
        "standards": [
            "ILAE-2022 epilepsy classification",
            "ACMG-AMP-2015 variant classification",
            "HMDSIG-Mito-Disease-Guidelines-2012",
            "Barth-2012-PCH-Classification",
            "CPIC-POLG-VPA-2023 (applies to all mito disease)",
            "EUROMIT-Mito-Guidelines",
            "WHO-ICF-2019",
            "ESPGHAN-2017 enteral nutrition",
            "NICE-NG-Mitochondrial-Disease",
            "ACR-Paed-Brain-MRI-Appropriateness",
            "SIMD-Metabolic-Screen-Lactate",
            "MRC-Mitochondrial-Disease-Patient-Pathway-2020",
        ],
        "etiologies": [
            {"type": "RARS2 Compound Het Missense/Missense — PCH6 Classic", "pct": 45},
            {"type": "RARS2 Missense/Frameshift — PCH6 Severe", "pct": 25},
            {"type": "RARS2 Homozygous Missense — PCH6 Intermediate", "pct": 18},
            {"type": "RARS2 Large Deletion — PCH6 Severe", "pct": 7},
            {"type": "Mito ARS Phenocopy (EARS2/DARS2)", "pct": 5},
        ],
        "seizure_types": [
            {"type": "Neonatal Tonic/Clonic Seizures", "pct": 88},
            {"type": "Myoclonic Seizures", "pct": 65},
            {"type": "Infantile Spasms (IS)", "pct": 42},
            {"type": "Focal Seizures", "pct": 35},
            {"type": "Epileptic Spasms", "pct": 28},
        ],
        "triggers": [
            {"trigger": "Metabolic Stress / Illness", "pct": 95},
            {"trigger": "Fasting / Catabolism", "pct": 88},
            {"trigger": "Fever", "pct": 85},
            {"trigger": "Missed Mito Supplements", "pct": 60},
            {"trigger": "Surgery / Anaesthesia", "pct": 55},
            {"trigger": "Aminoglycoside Exposure", "pct": 50},
            {"trigger": "Sleep Deprivation", "pct": 45},
            {"trigger": "Heat Stress", "pct": 38},
        ],
        "references": [
            "Edvardson S 2007 Am J Hum Genet (RARS2 PCH6 discovery)",
            "Barth PG 2012 EJPN (PCH classification)",
            "Rankin J 2010 Clin Genet (RARS2 UK cohort)",
            "Glamuzina E 2012 J Child Neurol (RARS2 management)",
            "CPIC-POLG-VPA-Guideline-2023",
            "Gorman GS 2015 Nat Rev Neurol (mito disease guidelines)",
        ],
    },
    # -- EXOSC3 — PCH1B --------------------------------------------------------
    {
        "gene": "EXOSC3",
        "alt_name": (
            "EXOSC3 (EXOSC3-275aa-9p13.2 / AR — PCH1B-Pontocerebellar-Hypoplasia-Type-1B-RNA-Exosome — "
            "MOTOR-NEURON-DISEASE-CONCURRENT-PATHOGNOMONIC-SMA-LIKE-PLUS-CEREBELLAR — "
            "D132A-POLISH-ROMA-FOUNDER-MUTATION-PATHOGNOMONIC — "
            "SLOWER-PROGRESSION-THAN-PCH2-LONGER-SURVIVAL)"
        ),
        "protein": (
            "EXOSC3 -- 9p13.2 AR -- EXOSC3-275aa -- "
            "RNA-Exosome-Component-3-RRP40-Cap-Subunit-3-5-RNA-Degradation-Machine -- "
            "PCH1B-Pontocerebellar-Hypoplasia-Type-1B-OMIM-614678 -- "
            "RNA-Exosome-10-Subunit-Ring-Barrel-Core-3-Cap-Subunits -- "
            "MOTOR-NEURON-DISEASE-SMA-LIKE-CONCURRENT-CEREBELLAR-HYPOPLASIA-PATHOGNOMONIC -- "
            "D132A-c395GA-Polish-Romani-Founder-Mutation-Most-Common-EXOSC3 -- "
            "Slower-Progression-Than-PCH2-TSEN-Genes-Longer-Survival-Possible -- "
            "Anterior-Horn-Cell-Degeneration-EMG-Confirms-Neurogenic-Pattern -- "
            "RNA-Surveillance-rRNA-Maturation-snRNA-Processing-Defect -- "
            "SMAD-Panel-If-SMA-Phenotype-Plus-Cerebellar-Features -- "
            "9p13.2"
        ),
        "locus": "9p13.2",
        "protein_size": "275 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; "
            "D132A (c.395G>A) founder in Polish/Roma/East European populations; "
            "compound het D132A + null common; "
            "GENETIC TESTING: EXOSC3 sequencing + RNA exosome panel; "
            "SMN1 deletion negative in PCH1B (DDx SMA); "
            "EMG: neurogenic pattern (anterior horn cells) — key diagnostic"
        ),
        "age_of_onset": "Neonatal / early infancy (hypotonia, weakness at birth)",
        "pathognomonic": (
            "MOTOR NEURON DISEASE + CEREBELLAR HYPOPLASIA concurrent — PATHOGNOMONIC for PCH1; "
            "EMG: neurogenic pattern (fibrillations, reduced recruitment) = anterior horn cells; "
            "SMN1 copy number NORMAL (unlike SMA) — DDx MANDATORY; "
            "D132A founder mutation: screen first in Polish/Roma/East European ancestry; "
            "Slower progression than TSEN-PCH2 — some patients survive to school age; "
            "MRI: cerebellar hypoplasia + thin posterior pons (different from dragonfly PCH2); "
            "Muscle biopsy: neurogenic atrophy — SMA-like but SMN1 normal"
        ),
        "treatment": (
            "ANTI-EPILEPTIC: LEV first-line; VPA if no mito concern; "
            "RESPIRATORY: earlier ventilation due to motor neuron weakness; "
            "NIV (BiPAP): when FVC <50% predicted (earlier in MND); "
            "TRACHEOSTOMY: consider if BiPAP fails (PCH1 MND progression); "
            "PHYSIOTHERAPY: intensive — motor neuron disease + cerebellar; "
            "FEEDING: early NG/PEG — MND affects bulbar early; "
            "NUSINERSEN / SMA TREATMENT: NOT INDICATED (SMN1 normal — not SMA); "
            "RISDIPLAM: NOT INDICATED (not SMA1/SMA2); "
            "SPINAL ORTHOSES: early for scoliosis (MND + cerebellar ataxia); "
            "PALLIATIVE CARE: earlier than PCH2 due to MND respiratory decline"
        ),
        "contraindications": (
            "NUSINERSEN CONTRAINDICATED: not SMN1-related — wastes resource and false hope; "
            "RISDIPLAM CONTRAINDICATED: same reason — SMN1 normal; "
            "AVOID MUSCLE RELAXANTS CHRONIC: worsen existing motor neuron weakness; "
            "AVOID AMIODARONE: rare peripheral neuropathy risk additive; "
            "AVOID HIGH-DOSE AMINOGLYCOSIDES: anterior horn toxicity; "
            "DO NOT LABEL AS SMA: critical — wrong treatment pathway"
        ),
        "monitoring": (
            "EMG: 6-monthly (neurogenic progression); "
            "RESPIRATORY: FVC 3-monthly, sleep study 6-monthly; "
            "MRI BRAIN/SPINE: 12-monthly (cerebellar + cord atrophy); "
            "FEEDING SLT: 3-monthly; "
            "SCOLIOSIS X-RAY SPINE: 6-monthly (Cobb angle); "
            "SMN1 copy number: confirm once (exclude SMA at diagnosis); "
            "EXOSC3 family cascade: 25% sibling risk; "
            "EEG: 6-monthly (seizure surveillance); "
            "DEVELOPMENTAL: 6-monthly; "
            "MUSCLE ENZYME CK: at diagnosis (mild elevation in neurogenic disease)"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): hypotonia, weakness, respiratory support, EMG",
            "Infancy (1-12 mo): motor regression, NIV, PEG, cerebellar atrophy",
            "Toddler (1-3 yr): MND progression, spinal orthoses, epilepsy",
            "Childhood (3-10 yr): slower — some walk briefly with support",
            "Adolescence (10-18 yr): wheelchair, NIV/trach, scoliosis management",
            "Adult: rare; severe disability; home ventilation possible",
        ],
        "concepts": [
            "PCH1B: RNA exosome defect — EXOSC3 cap subunit (RRP40)",
            "Motor neuron disease: concurrent SMA-like + cerebellar = PCH1 hallmark",
            "SMN1 normal: DDx SMA mandatory — PCH1B misdiagnosed as SMA Type 1",
            "D132A founder: screen first in Polish/Roma ancestry (common allele)",
            "EMG neurogenic: confirms anterior horn cell involvement",
            "NUSINERSEN: absolutely not indicated — SMN1 pathway intact",
            "Slower progression: PCH1B survives longer than PCH2 TSEN genes",
            "RNA exosome: 10-subunit ring degrades aberrant RNA — EXOSC3 caps",
            "Scoliosis early: motor neuron + cerebellar ataxia combination",
            "NIV earlier: MND respiratory decline faster than pure cerebellar PCH",
            "MRI PCH1: thin pons + cerebellar hypoplasia (different from dragonfly)",
            "rRNA processing: EXOSC3 defect → ribosome biogenesis impaired",
            "EXOSC8 (PCH1C): related RNA exosome subunit — same pathway",
            "Muscle biopsy: neurogenic atrophy pattern — not myopathy",
            "Prenatal: CVS/amnio for EXOSC3 biallelic — PCH1B prenatal option",
        ],
        "thresholds": [
            "SMN1 copy number: must confirm NORMAL before EXOSC3 diagnosis finalised",
            "FVC <50% predicted: NIV initiation in MND-PCH1",
            "FVC <30%: tracheostomy discussion",
            "Cobb angle >20°: spinal orthosis referral",
            "Cobb angle >45°: scoliosis surgery consideration",
            "D132A allele: sequence FIRST in Polish/Roma ancestry",
            "EMG fibrillation potentials: confirms ongoing motor neuron denervation",
            "CK >5x ULN: not typical for PCH1B — reconsider diagnosis",
            "EXOSC3 panel sensitivity >90% with sequencing + large deletion analysis",
            "PCH1B survival: median ~3-5 years (longer than PCH6 RARS2)",
            "Respiratory rate >60 at rest: respiratory failure imminent — escalate",
            "Scoliosis progression >5°/6 months: orthosis adjustment",
        ],
        "standards": [
            "ILAE-2022 epilepsy classification",
            "ACMG-AMP-2015 variant classification",
            "PCH-Network-European-Consensus-2014",
            "Barth-2012-PCH-Classification",
            "NICE-NG-Neuromuscular (MND guidance adapted)",
            "European-Neuromuscular-Centre-SMA-vs-PCH1",
            "WHO-ICF-2019",
            "ESPGHAN-2017 enteral nutrition",
            "EFNS-Cerebellar-Ataxia-Guidelines",
            "ATS-Respiratory-Neuromuscular",
            "ACR-Appropriateness-Paed-MRI",
            "SMA-UK-Guidelines-adapted-for-PCH1",
        ],
        "etiologies": [
            {"type": "EXOSC3 D132A/D132A — PCH1B Polish/Roma Classic", "pct": 38},
            {"type": "EXOSC3 D132A/null — PCH1B Compound Het", "pct": 28},
            {"type": "EXOSC3 Missense/Missense — PCH1B Other", "pct": 22},
            {"type": "EXOSC3 Large Deletion — PCH1B Severe", "pct": 7},
            {"type": "PCH1 Phenocopy (VRK1 / EXOSC8)", "pct": 5},
        ],
        "seizure_types": [
            {"type": "Focal Seizures", "pct": 55},
            {"type": "Myoclonic Seizures", "pct": 45},
            {"type": "Tonic Seizures", "pct": 38},
            {"type": "Infantile Spasms", "pct": 32},
            {"type": "Generalised Tonic-Clonic", "pct": 22},
        ],
        "triggers": [
            {"trigger": "Fever", "pct": 80},
            {"trigger": "Respiratory Infection", "pct": 75},
            {"trigger": "Missed AED", "pct": 55},
            {"trigger": "Sleep Deprivation", "pct": 48},
            {"trigger": "Metabolic Stress", "pct": 42},
            {"trigger": "Hypoxia (respiratory decline)", "pct": 38},
            {"trigger": "Excitement", "pct": 25},
            {"trigger": "Heat Stress", "pct": 22},
        ],
        "references": [
            "Wan J 2012 Nat Genet (EXOSC3 PCH1B discovery)",
            "Rudnik-Schöneborn S 2013 Nat Genet (EXOSC3 series)",
            "Barth PG 2012 EJPN",
            "Di Donato N 2016 Am J Hum Genet (RNA exosome PCH)",
            "Schwabova J 2013 J Child Neurol (D132A founder Czech)",
            "Poretti A 2017 J Child Neurol (MRI patterns)",
        ],
    },
    # -- VRK1 — PCH1B/SMA-PCH -------------------------------------------------
    {
        "gene": "VRK1",
        "alt_name": (
            "VRK1 (VRK1-396aa-14q32.2 / AR — PCH1B-SMA-PCH-Overlap-Vaccinia-Related-Kinase-1 — "
            "SMA-PHENOTYPE-PLUS-CEREBELLAR-HYPOPLASIA-PATHOGNOMONIC — "
            "HISTONE-H3-PHOSPHORYLATION-DEFECT-DNA-DAMAGE-RESPONSE — "
            "MICROCEPHALY-PLUS-SMA-DDx-MANDATORY-SMN1-NORMAL)"
        ),
        "protein": (
            "VRK1 -- 14q32.2 AR -- VRK1-396aa -- "
            "Vaccinia-Related-Kinase-1-Serine-Threonine-Kinase-Nuclear-Chromatin-DNA-Damage -- "
            "PCH1B-SMA-PCH-Overlap-Spectrum-OMIM-228550 -- "
            "Histone-H3-Ser10-Phosphorylation-Mitosis-BAF-Barrier-to-Autointegration -- "
            "SMA-Phenotype-Anterior-Horn-Cell-Degeneration-Plus-Cerebellar-Hypoplasia -- "
            "SMN1-COPY-NUMBER-NORMAL-KEY-DDx-NOT-SMA -- "
            "Arab-Saudi-Founder-R133C-Most-Common-VRK1-Variant -- "
            "Microcephaly-Progressive-Post-Natal-SMA-Like-Weakness-Cerebellar -- "
            "Slower-Progression-Than-RARS2-PCH6-Longer-Survival -- "
            "14q32.2"
        ),
        "locus": "14q32.2",
        "protein_size": "396 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "p.Arg133Cys (c.397C>T) founder in Arab/Saudi populations; "
            "compound heterozygous or homozygous; "
            "GENETIC TESTING: VRK1 sequencing + CNV on neuromuscular panel; "
            "SMN1 copy normal — SMA ruled out; "
            "VRK1 panel: include kinase domain variants; consanguinity common"
        ),
        "age_of_onset": "Neonatal / early infancy (hypotonia, weakness at birth)",
        "pathognomonic": (
            "SMA PHENOTYPE (proximal weakness, hypotonia, absent reflexes) + CEREBELLAR HYPOPLASIA "
            "= PATHOGNOMONIC for PCH1/VRK1; "
            "SMN1 COPY NUMBER NORMAL — SMA1 excluded; "
            "EMG: neurogenic (fibrillations, reduced recruitment) as in PCH1B/EXOSC3; "
            "MRI: cerebellar hypoplasia + progressive microcephaly; "
            "p.Arg133Cys: first Saudi family (Al-Owain 2013) — screen in Arab ancestry first; "
            "HISTONE H3 phosphorylation defect: VRK1 substrate; measurable in lymphocytes (research)"
        ),
        "treatment": (
            "RESPIRATORY: NIV/BiPAP early due to SMA-like muscle weakness; "
            "NUSINERSEN / RISDIPLAM: NOT INDICATED (SMN1 pathway not involved); "
            "ANTI-EPILEPTIC: LEV or VPA for seizures; "
            "PHYSIOTHERAPY: intensive motor neuron + cerebellar; "
            "FEEDING: PEG early for bulbar weakness; "
            "SPINAL ORTHOSES: scoliosis prevention; "
            "PHYSIOTHERAPY HYDROTHERAPY: combined SMA-like + ataxic gait; "
            "PALLIATIVE CARE: early planning given MND + cerebellar progression"
        ),
        "contraindications": (
            "NUSINERSEN ABSOLUTE CONTRAINDICATION: SMN1 normal (not SMA); "
            "RISDIPLAM ABSOLUTE CONTRAINDICATION: same reason; "
            "AVOID MUSCLE RELAXANTS: worsen existing motor neuron weakness; "
            "DO NOT LABEL SMA: critical error — delays correct diagnosis"
        ),
        "monitoring": (
            "EMG: 6-monthly; "
            "RESPIRATORY FVC: 3-monthly; "
            "SLEEP STUDY: 6-monthly (hypoventilation); "
            "MRI BRAIN: 12-monthly; "
            "SCOLIOSIS SPINE X-RAY: 6-monthly (Cobb); "
            "SMN1: confirm once at diagnosis; "
            "VRK1 family cascade: 25% sibling risk; "
            "DEVELOPMENTAL: 6-monthly; "
            "HEAD CIRCUMFERENCE: monthly"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): hypotonia, weakness, respiratory support",
            "Infancy (1-12 mo): NIV, PEG, cerebellar atrophy progression",
            "Toddler (1-3 yr): MND progression, orthopaedic support",
            "Childhood (3-10 yr): wheelchair, severe disability",
            "Adolescence: home ventilation, palliative care",
            "Adult: rare in severe form; possible in mild kinase variants",
        ],
        "concepts": [
            "VRK1: nuclear kinase — histone H3 + BAF + Coilin phosphorylation",
            "PCH1/SMA-PCH overlap: VRK1 + EXOSC3 share SMA-like + cerebellar phenotype",
            "SMN1 normal: mandatory to confirm — VRK1 not SMA",
            "Nusinersen: NOT for VRK1 — waste and false hope",
            "R133C founder: Arab/Saudi population (p.Arg133Cys)",
            "EMG neurogenic: confirms AHC degeneration (same as EXOSC3-PCH1B)",
            "DNA damage response: VRK1 phosphorylates H3-Ser10 in mitosis",
            "VRK2 (2p16): related kinase — different phenotype",
            "PCH1 group: motor neuron disease distinguishes from PCH2 (cerebellar only)",
            "Mitosis arrest: VRK1 defect → chromatin condensation abnormal",
            "Slower progression vs RARS2-PCH6: VRK1 may survive beyond infancy",
            "Histone H3 phosphorylation in lymphocytes: research biomarker",
            "Scoliosis: combined motor neuron + cerebellar instability",
            "Family history consanguinity: PCH1/VRK1 in Arab populations",
            "Prenatal diagnosis: CVS/amnio for VRK1 biallelic",
        ],
        "thresholds": [
            "SMN1 copy normal: VRK1 mandatory if SMA phenotype + cerebellar",
            "FVC <50%: NIV initiation",
            "Cobb angle >20°: orthosis referral",
            "EMG fibrillations: neurogenic — confirms AHC involvement",
            "Head circumference <-3 SD: progressive microcephaly",
            "VRK1 panel sensitivity >90%",
            "R133C allele: screen first in Arab/Saudi ancestry",
            "Nusinersen: absolute contraindication (SMN1 pathway not involved)",
            "PEG if oral intake <75% by 6 months",
            "SpO2 <92% nocturnal: BiPAP start",
            "Sibling risk 25%: prenatal offer",
            "VRK1 missense severity: kinase-dead alleles worse than hypomorphic",
        ],
        "standards": [
            "ILAE-2022 epilepsy classification",
            "ACMG-AMP-2015 variant classification",
            "PCH-Network-European-Consensus-2014",
            "Barth-2012-PCH-Classification",
            "WHO-ICF-2019",
            "ESPGHAN-2017 enteral nutrition",
            "ATS-Respiratory-Neuromuscular-Guidelines",
            "NICE-NG-Neuromuscular (adapted)",
            "ACR-Appropriateness-Paed-MRI",
            "SMA-UK-Guidelines-adapted-PCH1",
            "EFNS-Cerebellar-Ataxia-Guidelines",
            "European-Neuromuscular-Centre-SMA-DDx",
        ],
        "etiologies": [
            {"type": "VRK1 R133C/R133C — PCH1B-SMA Arab/Saudi Classic", "pct": 40},
            {"type": "VRK1 R133C/null — Compound Het", "pct": 25},
            {"type": "VRK1 Kinase-Dead/Missense — Severe", "pct": 20},
            {"type": "VRK1 Hypomorphic/Hypomorphic — Milder", "pct": 10},
            {"type": "PCH1 SMA-Phenocopy (EXOSC3)", "pct": 5},
        ],
        "seizure_types": [
            {"type": "Focal Seizures", "pct": 52},
            {"type": "Myoclonic Seizures", "pct": 42},
            {"type": "Tonic Seizures", "pct": 35},
            {"type": "Infantile Spasms", "pct": 28},
            {"type": "Generalised Tonic-Clonic", "pct": 18},
        ],
        "triggers": [
            {"trigger": "Fever", "pct": 82},
            {"trigger": "Respiratory Infection", "pct": 72},
            {"trigger": "Missed AED", "pct": 55},
            {"trigger": "Metabolic Stress", "pct": 45},
            {"trigger": "Hypoxia", "pct": 40},
            {"trigger": "Sleep Deprivation", "pct": 35},
            {"trigger": "Excitement", "pct": 22},
            {"trigger": "Heat Stress", "pct": 18},
        ],
        "references": [
            "Al-Owain M 2013 Am J Med Genet (VRK1 R133C Arab founder)",
            "Najmabadi H 2011 Nature (VRK1 consanguineous)",
            "Barth PG 2012 EJPN (PCH1 classification)",
            "Renbaum P 2009 Hum Mol Genet (VRK1 SMA-PCH)",
            "Vinograd-Byk H 2015 J Neurosci (VRK1 mechanism)",
            "Braun K 2016 Dev Med Child Neurol (PCH management)",
        ],
    },
    # -- CASK — PCH3 -----------------------------------------------------------
    {
        "gene": "CASK",
        "alt_name": (
            "CASK (CASK-922aa-Xp11.4 / XL-Dom-Female-Hemizygous-Lethal-Male — PCH3-Microcephaly-Hypotonia-Epilepsy — "
            "FEMALE-VARIABLE-EXPRESSION-MOSAIC-PATHOGNOMONIC — "
            "MALE-HEMIZYGOUS-NEONATAL-LETHAL-PATHOGNOMONIC — "
            "FGF14-CASK-SYNAPTIC-MODULE-DEFECT-CEREBELLUM)"
        ),
        "protein": (
            "CASK -- Xp11.4 XL-Dom(F)/hemi(M) -- CASK-922aa -- "
            "Membrane-Associated-Guanylate-Kinase-MAGUK-CaMK-SH3-PDZ-GUK-Domains -- "
            "PCH3-Microcephaly-Pontocerebellar-Hypoplasia-Hypotonia-Epilepsy-OMIM-300422 -- "
            "CASK-Microcephaly-with-Pontine-Cerebellar-Hypoplasia-MICPCH-Syndrome -- "
            "FGF14-CASK-Complex-Synaptic-Scaffolding-Neurexin-Binding-TBR1 -- "
            "FEMALE-HETEROZYGOUS-Variable-Severity-Mosaicism-Determines-Phenotype -- "
            "MALE-HEMIZYGOUS-Neonatal-Lethal-Essential-Gene-Brain-Development -- "
            "Progressive-Microcephaly-Profound-Intellectual-Disability-Spastic-Cerebral-Palsy -- "
            "Nystagmus-Optic-Atrophy-Visual-Impairment-CASK-Plus-Phenotype -- "
            "CASK-Hypomorphic-Variants-Males-Can-Rarely-Survive-Milder-Phenotype -- "
            "Xp11.4"
        ),
        "locus": "Xp11.4",
        "protein_size": "922 aa",
        "inheritance": (
            "X-linked; CASK heterozygous female: variable severity (somatic mosaicism determines expression); "
            "CASK hemizygous male: neonatal lethal (complete loss essential gene) — RARE survivors with hypomorphic variants; "
            "PCH3/MICPCH: 46,XX females predominant (males don't survive); "
            "de novo in >80% females (sporadic); "
            "GENETIC TESTING: CASK sequencing + MLPA (deletions common); "
            "Carrier mother: usually asymptomatic (mosaicism or full X-inactivation skewing)"
        ),
        "age_of_onset": "Congenital / neonatal (brain malformation present at birth)",
        "pathognomonic": (
            "FEMALE PREDOMINANCE (males neonatal lethal) — PATHOGNOMONIC for CASK mutations; "
            "MICROCEPHALY + PONTOCEREBELLAR HYPOPLASIA + PROFOUND ID: MICPCH triad; "
            "Nystagmus (>80%) — cerebellar + brainstem defect; "
            "Optic atrophy (~50%) — visual impairment; "
            "Spastic diplegia / cerebral palsy-like features; "
            "MRI: disproportionate hypoplasia of BRAINSTEM + CEREBELLUM vs cerebrum; "
            "de novo in majority — maternal X normal; "
            "CASK hypomorphic variants in surviving males: milder phenotype (intellectual disability)"
        ),
        "treatment": (
            "ANTI-EPILEPTIC: LEV first-line; VPA/CLB adjunct; "
            "FEEDING: NG tube / PEG — profound dysphagia + hypotonia; "
            "VISUAL: patching amblyopia; low-vision aids; cortical visual impairment management; "
            "NYSTAGMUS: prism glasses trial; "
            "PHYSIOTHERAPY: spasticity management; botulinum toxin for focal spasticity; "
            "OCCUPATIONAL THERAPY: severe GDD — AAC, adaptive equipment; "
            "OPHTHALMOLOGY: 6-monthly optic atrophy + nystagmus; "
            "ORTHOPAEDICS: hip surveillance (CP-like); "
            "PALLIATIVE CARE: integrate from diagnosis in severe MICPCH"
        ),
        "contraindications": (
            "AVOID CBZ/OXC IF NYSTAGMUS-DOMINANT: worsens nystagmus; "
            "CAUTION BENZODIAZEPINES CHRONIC: respiratory depression in hypotonic; "
            "AVOID SEDATING ANTIEPILEPTICS in visual-dependent patients: limits cortical visual rehab; "
            "DO NOT MISDIAGNOSE AS RETT SYNDROME: MECP2 negative — CASK panel; "
            "DO NOT ASSUME MALES CANNOT HAVE CASK: hypomorphic variants survive"
        ),
        "monitoring": (
            "MRI BRAIN: 6-monthly in first 2 years; "
            "EEG: 3-monthly (epilepsy management); "
            "OPHTHALMOLOGY: 6-monthly (nystagmus, optic atrophy, VEPs); "
            "VEPs (Visual Evoked Potentials): annually (cortical visual function); "
            "HIP X-RAY: 12-monthly (CP-like dislocation risk); "
            "HEAD CIRCUMFERENCE: monthly (progressive microcephaly); "
            "DEVELOPMENTAL: Bayley / VABS 6-monthly; "
            "CASK MATERNAL TESTING: X-inactivation + sequencing (recurrence risk assessment); "
            "SWALLOWING ASSESSMENT SLT: 3-monthly; "
            "HEARING: BERA annually (brainstem involvement)"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): microcephaly, hypotonia, nystagmus, seizures, PEG",
            "Infancy (1-12 mo): visual impairment, spasticity, epilepsy optimisation",
            "Toddler (1-3 yr): profound GDD, physiotherapy, ophthalmology",
            "Childhood (3-12 yr): school support, AAC, severe disability management",
            "Adolescence (12-18 yr): scoliosis, ongoing support, palliative planning",
            "Adult: rare severe form survival; APC; daily care needs total",
        ],
        "concepts": [
            "MICPCH: Microcephaly + Pontocerebellar Hypoplasia = CASK syndrome",
            "Female predominance: males hemizygous CASK = neonatal lethal",
            "De novo >80%: CASK mutations rarely inherited",
            "FGF14-CASK module: synaptic scaffolding — TBR1 transcription factor",
            "Nystagmus >80%: cerebellar + brainstem brainstem nuclei affected",
            "Optic atrophy ~50%: visual rehab mandatory",
            "Mosaicism determines severity: more mosaicism → milder in females",
            "MAGUK protein family: CASK contains CaMK, SH3, PDZ, GUK domains",
            "Neurexin binding: CASK links presynaptic scaffold to cell adhesion",
            "X-inactivation skewing: protective in carrier mothers (99% normal phenotype)",
            "CASK hypomorphic variants males: rare survivors — milder ID without MICPCH",
            "CBZ/OXC avoid: worsens nystagmus (cerebellar channels in nystagmus circuit)",
            "Rett DDx: MECP2 negative + X-linked + MICPCH → CASK panel",
            "TBR1: CASK-bound transcription factor for cortical neuron identity",
            "Prenatal: MRI at 20-28 weeks for MICPCH detection; CASK molecular on CVS",
        ],
        "thresholds": [
            "Head circumference <-2 SD at birth + progressive: MICPCH alert → CASK",
            "Female with profound ID + microcephaly + nystagmus: CASK first gene",
            "Optic atrophy on fundoscopy: add ophthalmology VEP monitoring",
            "Male with CASK variant: confirm hypomorphic + confirm NOT hemizygous null",
            "PEG if oral intake <80% by 4 months",
            "Hip surveillance X-ray: Reimer migration index >33% = subluxation",
            "Nystagmus + CBZ: STOP CBZ (worsening nystagmus threshold)",
            "CASK deletion by MLPA: 15-20% of CASK cases are genomic deletions",
            "X-inactivation: >80:20 ratio in mother → carrier likely asymptomatic",
            "VEP latency >130 ms: cortical visual impairment — low-vision referral",
            "Sibling risk: de novo CASK <1% recurrence; inherited (rare) 50% daughters",
            "Brainstem hypoplasia: APD <5 mm at term — severe CASK phenotype",
        ],
        "standards": [
            "ILAE-2022 epilepsy classification",
            "ACMG-AMP-2015 variant classification",
            "PCH-Network-European-Consensus-2014",
            "Barth-2012-PCH-Classification",
            "WHO-ICF-2019",
            "ESPGHAN-2017 enteral nutrition",
            "AICARDI-Neuropaediatric-Cerebral-Palsy-Guidelines",
            "RCOphth-Paediatric-Ophthalmology-Guidelines",
            "NICE-NG-Epilepsy-Children",
            "ACR-Appropriateness-Paed-Brain-MRI",
            "BOS-Microcephaly-Investigation-Protocol",
            "Bhatt SS 2021 Dev Med (MICPCH CASK review)",
        ],
        "etiologies": [
            {"type": "CASK de novo LOF Female — MICPCH Severe", "pct": 50},
            {"type": "CASK de novo Missense Female — MICPCH Moderate", "pct": 25},
            {"type": "CASK Deletion (MLPA) Female — MICPCH Severe", "pct": 15},
            {"type": "CASK Hypomorphic Male Survivor — Mild ID", "pct": 7},
            {"type": "CASK Phenocopy (TUBA1A / KIF11)", "pct": 3},
        ],
        "seizure_types": [
            {"type": "Focal Seizures", "pct": 65},
            {"type": "Tonic Seizures", "pct": 52},
            {"type": "Infantile Spasms (IS)", "pct": 38},
            {"type": "Myoclonic Seizures", "pct": 30},
            {"type": "Epileptic Spasms", "pct": 22},
        ],
        "triggers": [
            {"trigger": "Fever", "pct": 85},
            {"trigger": "Sleep Deprivation", "pct": 70},
            {"trigger": "Missed AED", "pct": 60},
            {"trigger": "Visual Stimulation (lights)", "pct": 40},
            {"trigger": "Excitement / Arousal", "pct": 35},
            {"trigger": "Illness", "pct": 80},
            {"trigger": "Tactile Stimulation", "pct": 30},
            {"trigger": "Heat", "pct": 28},
        ],
        "references": [
            "Najm J 2008 Nat Genet (CASK MICPCH discovery)",
            "Moog U 2011 J Med Genet (CASK female cohort)",
            "Burglen L 2012 Hum Mutat (CASK series)",
            "Takanashi J 2010 Radiology (CASK MRI features)",
            "LaConte LEW 2020 Hum Genet (CASK mechanism)",
            "Bhatt SS 2021 Dev Med Child Neurol (MICPCH review)",
        ],
    },
    # -- AMPD2 — PCH9 ----------------------------------------------------------
    {
        "gene": "AMPD2",
        "alt_name": (
            "AMPD2 (AMPD2-900aa-1p13.3 / AR — PCH9-Pontocerebellar-Hypoplasia-Type-9-Purine-Defect — "
            "HYPOMYELINATION-CEREBELLAR-ATROPHY-OPTIC-ATROPHY-TRIAD-PATHOGNOMONIC — "
            "SAICAR-ELEVATED-METABOLOMICS-PATHOGNOMONIC — "
            "PURINE-CYCLE-AMP-DEAMINASE-2-ADSL-PATHWAY-OVERLAP)"
        ),
        "protein": (
            "AMPD2 -- 1p13.3 AR -- AMPD2-900aa -- "
            "AMP-Deaminase-2-Purine-Nucleotide-Cycle-AMP-to-IMP-Deamination -- "
            "PCH9-Pontocerebellar-Hypoplasia-Type-9-Purine-Biosynthesis-Defect-OMIM-615809 -- "
            "SAICAR-Succinylaminoimidazolecarboxamide-Ribose-5P-Elevated-Metabolomics-PATHOGNOMONIC -- "
            "Hypomyelination-Plus-Cerebellar-Atrophy-Plus-Optic-Atrophy-Triad -- "
            "Purine-Cycle-AMPD2-adenylosuccinate-ADSL-Pathway-Shared -- "
            "Spastic-Paraparesis-Mild-Intellectual-Disability-vs-Severe-PCH9 -- "
            "Uric-Acid-NORMAL-DDx-Lesch-Nyhan-HPRT-Deficiency -- "
            "ADSL-Deficiency-DDx-SAICAR-Common-Biomarker-Different-Gene -- "
            "1p13.3"
        ),
        "locus": "1p13.3",
        "protein_size": "900 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "compound het or homozygous; "
            "GENETIC TESTING: AMPD2 sequencing + metabolomics (SAICAR on urine/CSF); "
            "uric acid NORMAL (unlike Lesch-Nyhan/HPRT deficiency); "
            "SAICAR elevated: purines metabolomics first-line investigation; "
            "no specific founder; consanguinity a risk factor"
        ),
        "age_of_onset": "Neonatal / early childhood (variable — some milder presentations delayed)",
        "pathognomonic": (
            "HYPOMYELINATION + CEREBELLAR ATROPHY + OPTIC ATROPHY TRIAD — PATHOGNOMONIC for PCH9/AMPD2; "
            "SAICAR (succinylaminoimidazolecarboxamide ribose-5-phosphate) ELEVATED on purines metabolomics — PATHOGNOMONIC; "
            "MRI: hypomyelination (delayed white matter maturation) + cerebellar atrophy + optic atrophy; "
            "Uric acid NORMAL (distinguishes from Lesch-Nyhan / HPRT deficiency); "
            "DDx ADSL deficiency: also elevated SAICAR — gene sequencing distinguishes; "
            "Spectrum: severe PCH9 (neonatal) → milder spastic paraparesis (childhood onset)"
        ),
        "treatment": (
            "PURINE SUPPLEMENTATION: adenine + allopurinol (theoretical — limited evidence); "
            "RIBOSE SUPPLEMENTATION: limited evidence (purine salvage); "
            "ANTI-EPILEPTIC: LEV first-line; VPA for myoclonic; "
            "VISUAL REHABILITATION: optic atrophy management — low-vision services; "
            "SPASTICITY: baclofen (oral/intrathecal for severe); tizanidine; "
            "PHYSIOTHERAPY: spastic paraparesis + cerebellar ataxia; "
            "FEEDING: PEG if dysphagia severe; "
            "OCCUPATIONAL THERAPY: AAC for severe GDD; "
            "PALLIATIVE CARE: severe neonatal PCH9 form"
        ),
        "contraindications": (
            "AVOID ALLOPURINOL WITHOUT ADENINE CO-SUPPLEMENTATION: may worsen purine depletion; "
            "AVOID METHOTREXATE: folate/purine pathway interaction; "
            "AVOID AZATHIOPRINE: purine pathway — may worsen AMPD2 defect; "
            "DO NOT TREAT AS LESCH-NYHAN: different enzyme, uric acid normal"
        ),
        "monitoring": (
            "SAICAR URINARY: 3-monthly (treatment response monitoring); "
            "OPHTHALMOLOGY VEP ERG: 6-monthly (optic atrophy progression); "
            "MRI BRAIN: 12-monthly (hypomyelination maturation + cerebellar atrophy); "
            "EEG: 6-monthly; "
            "SPASTICITY ASSESSMENT: 6-monthly (Ashworth + GMFM); "
            "URIC ACID: 6-monthly (confirm normal, distinguish from other purinopathies); "
            "ADSL ENZYME: at diagnosis (DDx ADSL deficiency); "
            "AUDIOLOGY: 12-monthly (brainstem involvement); "
            "AMPD2 FAMILY CASCADE: 25% sibling risk"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): severe form — seizures, hypotonia, respiratory support",
            "Infancy (1-12 mo): epilepsy optimisation, visual impairment identification",
            "Toddler (1-3 yr): spasticity, GDD, ophthalmology, physiotherapy",
            "Childhood (3-12 yr): school support, spastic diplegia management",
            "Adolescence (12-18 yr): milder form — further development of spastic paraparesis",
            "Adult: milder variants — ambulant with disability; severe — lifelong care",
        ],
        "concepts": [
            "PCH9: purine cycle defect — AMPD2 converts AMP → IMP",
            "SAICAR: biomarker for purine de novo synthesis block (also in ADSL deficiency)",
            "Hypomyelination triad: AMPD2 hallmark (PCH + myelination delay + optic atrophy)",
            "Uric acid normal: critical DDx from Lesch-Nyhan (HPRT deficiency)",
            "ADSL DDx: both elevate SAICAR — gene sequencing required",
            "Purine supplementation: theoretical — adenine + allopurinol",
            "Optic atrophy: 50%+ in PCH9 — VEP mandatory",
            "Spectrum broad: severe neonatal PCH9 → milder spastic paraparesis (same gene)",
            "Hypomyelination on MRI: white matter delayed maturation (not leukodystrophy)",
            "AMPD2 vs AMPD1 (muscle): different isoforms — AMPD1 = muscle, AMPD2 = brain",
            "Purine cycle: AMP → IMP (AMPD2) → AMP (adenylosuccinate synthetase/lyase)",
            "CSF purines: SAICAR elevated CSF (same as plasma/urine in AMPD2)",
            "Ribose salvage: supplementation of D-ribose bypasses synthesis — limited data",
            "PCH9 phenotype overlap: some features overlap with Pelizaeus-Merzbacher",
            "Prenatal diagnosis: molecular AMPD2 + fetal MRI at 28-30 weeks",
        ],
        "thresholds": [
            "SAICAR >2x ULN urinary: AMPD2 or ADSL deficiency — gene sequencing",
            "Optic atrophy on fundoscopy: add VEP ERG monitoring",
            "Uric acid NORMAL: confirms not Lesch-Nyhan (HPRT deficiency ruled out)",
            "MRI hypomyelination at >12 months: delayed myelination — AMPD2 suspect",
            "ADSL enzyme activity: normal in AMPD2 (distinguishes from ADSL deficiency)",
            "AMPD2 panel sensitivity >90%",
            "Sibling risk 25%: prenatal offer",
            "VEP latency >120 ms: optic pathway delay — low-vision referral",
            "Spasticity Ashworth >2: baclofen escalation or intrathecal consideration",
            "Lesch-Nyhan uric acid: >600 μmol/L — AMPD2 should be normal",
            "Adenine supplementation dose: 100-300 mg/day (empirical purine repletion)",
            "Allopurinol dose in purine defect: 10-20 mg/kg/day (with adenine)",
        ],
        "standards": [
            "ILAE-2022 epilepsy classification",
            "ACMG-AMP-2015 variant classification",
            "PCH-Network-European-Consensus-2014",
            "SSIEM-Purine-Pyrimidine-Guidelines",
            "Barth-2012-PCH-Classification",
            "WHO-ICF-2019",
            "RCOphth-Low-Vision-Guidelines",
            "NICE-NG-Epilepsy-Children",
            "EFNS-Cerebellar-Ataxia-Guidelines",
            "ACR-Appropriateness-Paed-Brain-MRI",
            "ESPGHAN-2017 enteral nutrition",
            "Lesch-Nyhan-Disease-Society-Guidelines (DDx resource)",
        ],
        "etiologies": [
            {"type": "AMPD2 Compound Het Missense/Frameshift — PCH9 Classic", "pct": 45},
            {"type": "AMPD2 Homozygous Missense — PCH9 Intermediate", "pct": 25},
            {"type": "AMPD2 Null/Null — PCH9 Severe Neonatal", "pct": 15},
            {"type": "AMPD2 Missense/Missense Mild — Spastic Paraparesis", "pct": 10},
            {"type": "PCH9 Phenocopy (ADSL deficiency)", "pct": 5},
        ],
        "seizure_types": [
            {"type": "Tonic Seizures", "pct": 55},
            {"type": "Focal Seizures", "pct": 48},
            {"type": "Myoclonic Seizures", "pct": 38},
            {"type": "Infantile Spasms", "pct": 28},
            {"type": "Absence-Like", "pct": 15},
        ],
        "triggers": [
            {"trigger": "Fever / Illness", "pct": 82},
            {"trigger": "Sleep Deprivation", "pct": 68},
            {"trigger": "Missed AED", "pct": 58},
            {"trigger": "Metabolic Stress", "pct": 50},
            {"trigger": "Hyperthermia", "pct": 42},
            {"trigger": "Excitement", "pct": 30},
            {"trigger": "Tactile Stimulation", "pct": 25},
            {"trigger": "Fasting", "pct": 20},
        ],
        "references": [
            "Marsh AP 2015 Nat Genet (AMPD2 PCH9 discovery)",
            "Grunewald S 2015 Brain (PCH9 series)",
            "Barth PG 2012 EJPN (PCH classification)",
            "Poretti A 2017 J Child Neurol (PCH9 MRI)",
            "Jurecka A 2012 J Inherit Metab Dis (purine defects overview)",
            "SSIEM-Purine-Guideline-2020",
        ],
    },
    # -- TOE1 — PCH7 -----------------------------------------------------------
    {
        "gene": "TOE1",
        "alt_name": (
            "TOE1 (TOE1-421aa-1p34.1 / AR — PCH7-Pontocerebellar-Hypoplasia-Type-7-snRNA-Processing — "
            "GONADAL-DYSGENESIS-46XY-DSD-PLUS-CEREBELLAR-HYPOPLASIA-PATHOGNOMONIC — "
            "VERMIAN-HYPOPLASIA-PREDOMINANT->HEMISPHERAL-PATHOGNOMONIC — "
            "SNOG-COMPLEX-snRNA-3-END-PROCESSING-DEADENYLASE)"
        ),
        "protein": (
            "TOE1 -- 1p34.1 AR -- TOE1-421aa -- "
            "Target-of-EGR1-Deadenylase-DEDD-Superfamily-CAF1-Related-Ribonuclease -- "
            "PCH7-Pontocerebellar-Hypoplasia-Type-7-OMIM-614969 -- "
            "SNOG-Complex-snRNA-3-End-Processing-sm-snRNP-Assembly-Maturation -- "
            "GONADAL-DYSGENESIS-46XY-DSD-PLUS-Cerebellar-Hypoplasia-PATHOGNOMONIC -- "
            "Vermian-Hypoplasia-Predominant-Cerebellar-Vermis->Hemispheres-DISTINCTIVE -- "
            "Pontine-Hypoplasia-Variable-Neonatal-Hypotonia-Severe-Epilepsy -- "
            "Biallelic-LOF-snRNA-Maturation-Defect-Spliceosome-Assembly-Failure -- "
            "DSD-Workup-Mandatory-46XY-PCH7-Males-Phenotypically-Female-Often -- "
            "1p34.1"
        ),
        "locus": "1p34.1",
        "protein_size": "421 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "compound het or homozygous; "
            "GENETIC TESTING: TOE1 sequencing on neurodevelopmental panel or WES; "
            "DSD investigation mandatory in 46,XY cases (gonadal dysgenesis); "
            "no specific founder; rare disease — consanguinity increases risk; "
            "46,XY females: karyotype + TOE1 sequencing define phenotype"
        ),
        "age_of_onset": "Neonatal / early infancy (congenital cerebellar malformation)",
        "pathognomonic": (
            "GONADAL DYSGENESIS IN 46,XY + CEREBELLAR HYPOPLASIA — PATHOGNOMONIC for PCH7/TOE1; "
            "VERMIAN HYPOPLASIA PREDOMINANT over cerebellar hemispheres — DISTINCTIVE (unlike PCH2 dragonfly); "
            "46,XY female external genitalia + absent testes (streak gonads): DSD workup; "
            "Neonatal hypotonia + severe epilepsy + pontine hypoplasia; "
            "snRNA processing defect: SNOG complex (TOE1 + partners); "
            "KEY: any 46,XY PCH phenotype → TOE1 mandatory in differential"
        ),
        "treatment": (
            "GONADAL MANAGEMENT (46,XY DSD): gonadectomy recommended (malignancy risk streak gonads); "
            "HORMONE REPLACEMENT: oestrogen replacement post-gonadectomy for female-reared 46,XY; "
            "ANTI-EPILEPTIC: LEV first-line; VPA/CLB for refractory; "
            "FEEDING: NG/PEG for severe dysphagia; "
            "PHYSIOTHERAPY: hypotonia + cerebellar management; "
            "OCCUPATIONAL THERAPY: severe GDD — AAC; "
            "ENDOCRINOLOGY: DSD multidisciplinary team mandatory; "
            "PSYCHOLOGY: gender identity counselling (46,XY female-reared); "
            "PALLIATIVE CARE: integrate in severe PCH7 from diagnosis"
        ),
        "contraindications": (
            "AVOID DELAYING GONADECTOMY: streak gonads malignancy risk (gonadoblastoma) in 46,XY DSD; "
            "AVOID UNNECESSARY KARYOTYPE DISCLOSURE WITHOUT MDT SUPPORT: psychological impact in DSD; "
            "CAUTION CBZ/OXC IN NYSTAGMUS: worsens cerebellar nystagmus; "
            "DO NOT MISS DSD IN 46,XY PCH7: critical management need"
        ),
        "monitoring": (
            "DSD MDT: 3-monthly (endocrinology + paediatric surgery + psychology); "
            "GONADOBLASTOMA SURVEILLANCE: histology post-gonadectomy; "
            "OESTROGEN LEVELS: 6-monthly (HRT adequacy in gonadectomised 46,XY); "
            "EEG: 3-monthly (epilepsy); "
            "MRI BRAIN: 6-monthly (cerebellar progression); "
            "OPHTHALMOLOGY: 6-monthly (nystagmus, optic atrophy); "
            "DEVELOPMENTAL: 6-monthly; "
            "HEAD CIRCUMFERENCE: monthly; "
            "TOE1 FAMILY CASCADE: 25% sibling risk — prenatal for DSD families"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): hypotonia, DSD identified, seizures, MRI, DSD MDT",
            "Infancy (1-12 mo): epilepsy, PEG, DSD management, cerebellar atrophy",
            "Toddler-Childhood (1-10 yr): gonadectomy timing, HRT planning, GDD",
            "Adolescence (10-18 yr): HRT, gender support, severe disability",
            "Adult: lifelong HRT; severe disability; psychology; DSD ongoing support",
            "Family: genetic counselling, prenatal for future pregnancies",
        ],
        "concepts": [
            "PCH7: TOE1 = deadenylase in SNOG complex — snRNA 3'-end trimming",
            "Gonadal dysgenesis + PCH: PATHOGNOMONIC for TOE1 mutation",
            "46,XY DSD: TOE1 essential for gonad development (testicular differentiation fails)",
            "SNOG complex: TOE1 + ZCCHC8 + NCBP3 — pre-snRNA maturation",
            "Vermian predominant: PCH7 more vermian than hemispheral (unlike PCH2 dragonfly)",
            "Streak gonads: gonadoblastoma risk 15-30% — gonadectomy timing critical",
            "Oestrogen HRT: female-reared 46,XY after gonadectomy — lifelong",
            "DSD MDT: endocrinology + paediatrics + surgery + psychology + genetics",
            "snRNA maturation: SNOG complex assembles Sm snRNP — spliceosome component",
            "TOE1 deadenylase: DEDD superfamily — cleaves 3' poly-A on pre-snRNA",
            "PCH7 rare: <50 reported cases globally — underdiagnosed",
            "Gender counselling: sensitive approach for 46,XY female-reared patients",
            "DSD disclosure: multidisciplinary timing — avoid unsupported disclosure",
            "TOE1 vs TSEN genes: DSD distinguishes PCH7 (TOE1) from PCH2 (TSEN)",
            "Prenatal: fetal karyotype + MRI if TOE1 family history",
        ],
        "thresholds": [
            "46,XY with PCH phenotype: TOE1 mandatory in differential (DSD workup)",
            "Streak gonads: gonadectomy by age 5-10 (gonadoblastoma prevention)",
            "Oestrogen E2 <100 pmol/L at puberty: HRT initiation mandatory",
            "MRI vermian hypoplasia predominant: PCH7 suspect vs PCH2 dragonfly",
            "TOE1 panel sensitivity >90% with sequencing",
            "Sibling risk 25%: prenatal offer (especially in consanguineous families)",
            "Gonadoblastoma risk: 15-30% in 46,XY DSD without gonadectomy",
            "HRT oestrogen target: age-appropriate levels for bone density",
            "PCH7 epilepsy: ACTH if IS onset; LEV first-line otherwise",
            "PEG if oral intake <80% by 6 months",
            "DSD MDT meeting: within 4 weeks of DSD identification",
            "TOE1 rare: <50 cases — index of suspicion for 46,XY PCH",
        ],
        "standards": [
            "ILAE-2022 epilepsy classification",
            "ACMG-AMP-2015 variant classification",
            "PCH-Network-European-Consensus-2014",
            "Barth-2012-PCH-Classification",
            "BSPED-DSD-Guidelines-2019",
            "ESPE-DSD-Consensus-2006-Updated",
            "WHO-ICF-2019",
            "ESPGHAN-2017 enteral nutrition",
            "ACR-Appropriateness-Paed-Brain-MRI",
            "NICE-NG-Epilepsy-Children",
            "DSD-Consortium-Guidelines-2020",
            "EAU-Paediatric-DSD-Guidelines",
        ],
        "etiologies": [
            {"type": "TOE1 Biallelic LOF Female (46,XX) — PCH7 Classic", "pct": 40},
            {"type": "TOE1 Biallelic LOF Male (46,XY) — PCH7 + DSD Classic", "pct": 35},
            {"type": "TOE1 Compound Het — PCH7 Moderate", "pct": 15},
            {"type": "TOE1 Missense/Missense — PCH7 Milder", "pct": 7},
            {"type": "PCH7 Phenocopy (SNOG complex partner)", "pct": 3},
        ],
        "seizure_types": [
            {"type": "Tonic-Clonic Seizures", "pct": 60},
            {"type": "Focal Seizures", "pct": 50},
            {"type": "Infantile Spasms (IS)", "pct": 38},
            {"type": "Myoclonic Seizures", "pct": 30},
            {"type": "Tonic Seizures", "pct": 22},
        ],
        "triggers": [
            {"trigger": "Fever", "pct": 88},
            {"trigger": "Illness / Infection", "pct": 80},
            {"trigger": "Sleep Deprivation", "pct": 65},
            {"trigger": "Missed AED", "pct": 58},
            {"trigger": "Hormonal Changes (puberty in 46,XY)", "pct": 30},
            {"trigger": "Metabolic Stress", "pct": 45},
            {"trigger": "Excitement", "pct": 28},
            {"trigger": "Heat", "pct": 22},
        ],
        "references": [
            "Roifman M 2015 J Med Genet (TOE1 PCH7 discovery)",
            "Barth PG 2012 EJPN (PCH classification)",
            "Lardelli RM 2017 Nat Commun (TOE1 SNOG mechanism)",
            "Slavotinek A 2016 Am J Med Genet (TOE1 DSD series)",
            "Guerrini R 2018 Brain (PCH7 management)",
            "ESPE-DSD-Guidelines-2006-Updated",
        ],
    },
]


def _make_cohort(gene_data, seed):
    rng = random.Random(seed)
    ages = [rng.randint(0, 18) for _ in range(40)]
    sex_bias = gene_data["gene"] == "CASK"  # CASK: female predominant
    sexes = [
        "F" if (sex_bias and rng.random() < 0.88) or (not sex_bias and rng.random() < 0.50) else "M"
        for _ in range(40)
    ]
    alive = [rng.random() < (0.40 if gene_data["gene"] in ("RARS2", "TSEN54") else 0.72) for _ in range(40)]
    # dominant etiology pick
    etiol_types = [e["type"] for e in gene_data["etiologies"]]
    etiol_wts = [e["pct"] for e in gene_data["etiologies"]]
    etiols = rng.choices(etiol_types, weights=etiol_wts, k=40)
    # seizure type
    sz_types = [s["type"] for s in gene_data["seizure_types"]]
    sz_wts = [s["pct"] for s in gene_data["seizure_types"]]
    szs = rng.choices(sz_types, weights=sz_wts, k=40)
    # trigger
    trig_types = [t["trigger"] for t in gene_data["triggers"]]
    trig_wts = [t["pct"] for t in gene_data["triggers"]]
    trigs = rng.choices(trig_types, weights=trig_wts, k=40)
    patients = []
    for i in range(40):
        patients.append({
            "id": f"{gene_data['gene']}-{seed}-{i+1:02d}",
            "gene": gene_data["gene"],
            "age": ages[i],
            "sex": sexes[i],
            "alive": alive[i],
            "etiology": etiols[i],
            "seizure_type": szs[i],
            "trigger": trigs[i],
        })
    return patients


def _build_all():
    all_patients = []
    for idx, g in enumerate(PCH_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(g, seed))
    return all_patients


def overview():
    pts = _build_all()
    total = len(pts)
    alive_pct = round(100 * sum(1 for p in pts if p["alive"]) / total, 1)

    # per-gene summary
    gene_summaries = {}
    for idx, g in enumerate(PCH_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(g, seed)
        gene_summaries[g["gene"]] = {
            "gene": g["gene"],
            "alt_name": g["alt_name"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "n_patients": len(cohort),
            "alive_pct": round(100 * sum(1 for p in cohort if p["alive"]) / len(cohort), 1),
            "top_etiology": max(g["etiologies"], key=lambda e: e["pct"])["type"],
            "top_seizure_type": max(g["seizure_types"], key=lambda s: s["pct"])["type"],
            "top_trigger": max(g["triggers"], key=lambda t: t["pct"])["trigger"],
            "pathognomonic_summary": g["pathognomonic"][:160] + "…",
        }

    # aggregate etiology dist
    from collections import Counter
    all_etiol = Counter(p["etiology"] for p in pts)
    all_sz = Counter(p["seizure_type"] for p in pts)

    return {
        "title": "Hereditary-PCH-Atlas",
        "subtitle": (
            "Complete 8-Gene Pontocerebellar Hypoplasia Atlas — "
            "TSEN54 (PCH2A) · TSEN2 (PCH2B) · RARS2 (PCH6) · EXOSC3 (PCH1B) · "
            "VRK1 (PCH1B-SMA) · CASK (PCH3/MICPCH) · AMPD2 (PCH9) · TOE1 (PCH7)"
        ),
        "n_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "alive_pct": alive_pct,
        "gene_summaries": gene_summaries,
        "etiology_distribution": dict(all_etiol.most_common(10)),
        "seizure_type_distribution": dict(all_sz.most_common(8)),
        "key_flags": [
            "TSEN54-A307S-MOST-COMMON-PCH2A-WORLDWIDE",
            "DRAGONFLY-WING-MRI-PATHOGNOMONIC-PCH2",
            "RARS2-ELEVATED-LACTATE-PATHOGNOMONIC-PCH6",
            "VPA-ABSOLUTE-CI-RARS2-MITO-DISEASE",
            "EXOSC3-D132A-POLISH-ROMA-FOUNDER-PCH1B",
            "NUSINERSEN-NOT-INDICATED-PCH1-SMN1-NORMAL",
            "CASK-FEMALE-PREDOMINANT-MALE-HEMIZYGOUS-LETHAL",
            "AMPD2-SAICAR-ELEVATED-METABOLOMICS-PCH9",
            "TOE1-GONADAL-DYSGENESIS-46XY-PATHOGNOMONIC-PCH7",
            "PCH1-MOTOR-NEURON-DISEASE-DDx-SMA-SMN1-NORMAL",
            "RARS2-ABSENT-VERMIS-PCH6-MORE-SEVERE-PCH2",
            "PEG-EARLY-ALL-PCH-TYPES-MANDATORY",
        ],
    }


def breakdown():
    result = {}
    for idx, g in enumerate(PCH_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(g, seed)
        result[g["gene"]] = {
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "contraindications": g["contraindications"],
            "monitoring": g["monitoring"],
            "lifecycle": g["lifecycle"],
            "concepts": g["concepts"],
            "thresholds": g["thresholds"],
            "standards": g["standards"],
            "etiologies": g["etiologies"],
            "seizure_types": g["seizure_types"],
            "triggers": g["triggers"],
            "references": g["references"],
            "n_patients": len(cohort),
            "alive_pct": round(100 * sum(1 for p in cohort if p["alive"]) / len(cohort), 1),
            "cohort": cohort,
        }
    return result


def definitions():
    return {
        "PCH_types": {
            "PCH1": "Pontocerebellar Hypoplasia Type 1 — motor neuron disease + cerebellar (EXOSC3, VRK1, SMA-overlap)",
            "PCH2": "Pontocerebellar Hypoplasia Type 2 — TSEN complex; dragonfly wing MRI (TSEN54, TSEN2, TSEN34, TSEN15)",
            "PCH3": "Pontocerebellar Hypoplasia Type 3 — CASK (MICPCH); female predominant; microcephaly + PCH",
            "PCH4": "Pontocerebellar Hypoplasia Type 4 — TSEN54 null/null; most severe; neonatal lethal",
            "PCH6": "Pontocerebellar Hypoplasia Type 6 — RARS2; mito arginyl-tRNA; absent vermis + lactate elevated",
            "PCH7": "Pontocerebellar Hypoplasia Type 7 — TOE1; snRNA processing; gonadal dysgenesis + cerebellar",
            "PCH9": "Pontocerebellar Hypoplasia Type 9 — AMPD2; purine cycle; hypomyelination + cerebellar + optic atrophy",
        },
        "key_MRI_patterns": {
            "Dragonfly_wing": "Axial MRI: flattened cerebellar hemispheres + preserved vermis = PCH2 (TSEN genes) PATHOGNOMONIC",
            "Absent_vermis": "Cerebellar vermis absent/severely hypoplastic = PCH6 (RARS2) — more severe than PCH2",
            "Vermian_predominant": "Vermian hypoplasia > hemispheral = PCH7 (TOE1) DISTINCTIVE",
            "Disproportionate_brainstem": "Brainstem hypoplasia > cerebrum = PCH3/CASK (MICPCH)",
            "Hypomyelination": "Delayed white matter myelination = PCH9 (AMPD2) — different from leukodystrophy",
        },
        "key_biomarkers": {
            "Lactate": "Elevated plasma/CSF lactate → RARS2 (PCH6) — mito disease; normal in TSEN-PCH2",
            "SAICAR": "Succinylaminoimidazolecarboxamide ribose-5-phosphate elevated in AMPD2 (PCH9) and ADSL deficiency",
            "SMN1_copy": "Normal in PCH1 (EXOSC3/VRK1) — DDx SMA mandatory; SMA = 0 copies SMN1",
            "Uric_acid": "Normal in AMPD2 (PCH9) — DDx Lesch-Nyhan (HPRT) which has elevated uric acid",
            "TSEN54_A307S": "Most common PCH2A allele worldwide — screen first in any PCH2 phenotype",
            "EXOSC3_D132A": "Polish/Roma founder for PCH1B — c.395G>A — screen first in East European",
        },
        "critical_contraindications": {
            "VPA_RARS2": "VALPROATE ABSOLUTE CI in RARS2 (PCH6) — mitochondrial hepatotoxicity FATAL",
            "Nusinersen_PCH1": "NUSINERSEN/RISDIPLAM NOT INDICATED in PCH1 (EXOSC3/VRK1) — SMN1 pathway normal",
            "CBZ_nystagmus": "CARBAMAZEPINE/OXCARBAZEPINE avoid if nystagmus dominant — worsens cerebellar nystagmus",
            "VGB_visual": "VIGABATRIN requires annual ERG visual field monitoring — retinal toxicity",
            "Delay_gonadectomy": "DO NOT DELAY GONADECTOMY in 46,XY TOE1 (PCH7) — gonadoblastoma risk 15-30%",
            "Delay_PEG": "DO NOT DEFER PEG in any PCH type — nutritional failure worsens cognitive outcomes",
        },
        "surveillance_protocols": {
            "PCH_mri": "MRI 6-monthly first 2 years (progressive microcephaly + cerebellar atrophy); 12-monthly thereafter",
            "Head_circumference": "Monthly head circumference mandatory — post-natal progressive microcephaly key marker in PCH2",
            "Lactate_monitoring": "3-monthly lactate/pyruvate in RARS2 (PCH6) for mito metabolic stability",
            "DSD_surveillance": "TOE1 (PCH7) 46,XY: DSD MDT 3-monthly; gonadectomy by age 5-10",
            "SAICAR_monitoring": "3-monthly urinary SAICAR in AMPD2 (PCH9) for treatment response",
            "SMN1_confirm": "One-time SMN1 copy number at diagnosis in PCH1 (EXOSC3/VRK1) — confirms not SMA",
        },
        "ddx_table": {
            "PCH_vs_SMA": "SMN1 normal in PCH1 (EXOSC3/VRK1); SMN1 0 copies in SMA1/2 — MANDATORY DDx",
            "PCH_vs_ADSL": "Both elevate SAICAR; ADSL enzyme activity differentiates; gene sequencing definitive",
            "PCH_vs_Lesch_Nyhan": "AMPD2 uric acid normal; Lesch-Nyhan uric acid markedly elevated",
            "PCH_vs_PMD": "Pelizaeus-Merzbacher: PLP1 mutation + hypomyelination only, no PCH; AMPD2: PCH + hypomyelination",
            "RARS2_vs_TSEN": "RARS2: elevated lactate + absent vermis; TSEN54/2: normal lactate + dragonfly wing",
            "CASK_vs_Rett": "CASK: Xp11.4, female, MICPCH; Rett: MECP2, Xq28, deceleration pattern, preserved vermis",
        },
        "glossary": {
            "PCH": "Pontocerebellar Hypoplasia — group of inherited disorders with hypoplasia/atrophy of cerebellum and pons",
            "TSEN": "tRNA Splicing ENdonuclease — 4-subunit complex (TSEN2/15/34/54) removing tRNA introns",
            "MICPCH": "Microcephaly + Pontocerebellar Hypoplasia — CASK syndrome; female predominant",
            "SNOG_complex": "snRNA Oligoadenylation and 3'-end processing complex — TOE1 + partners",
            "Dragonfly_wing": "Axial MRI pattern in PCH2: flattened cerebellar hemispheres with preserved vermis",
            "DSD": "Differences of Sex Development — 46,XY with female external genitalia and streak gonads in PCH7",
            "Gonadoblastoma": "Malignant germ cell tumour risk in streak gonads of 46,XY DSD — gonadectomy protective",
            "SAICAR": "Succinylaminoimidazolecarboxamide Ribose-5-Phosphate — biomarker for purine synthesis block",
            "MAGUK": "Membrane-Associated Guanylate Kinase — protein family including CASK",
            "AHC": "Anterior Horn Cells — motor neurons in spinal cord; degenerate in PCH1 (EXOSC3/VRK1)",
            "VRK1": "Vaccinia-Related Kinase 1 — serine/threonine kinase phosphorylating histone H3-Ser10",
            "snRNA": "Small Nuclear RNA — spliceosomal RNA processed by TOE1/SNOG complex",
            "IS": "Infantile Spasms — epileptic seizure type; ACTH first-line; EEG shows hypsarrhythmia",
            "ACTH": "Adrenocorticotropic Hormone — first-line treatment for infantile spasms (Level A)",
            "PEG": "Percutaneous Endoscopic Gastrostomy — feeding tube; early placement in all severe PCH types",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(json.dumps({k: v for k, v in ov.items() if k != "gene_summaries"}, indent=2))
    print("\n=== BREAKDOWN keys ===")
    br = breakdown()
    for gene, data in br.items():
        print(f"  {gene}: {data['n_patients']} patients, alive={data['alive_pct']}%")
    print("\n=== DEFINITIONS keys ===")
    defs = definitions()
    print(list(defs.keys()))
