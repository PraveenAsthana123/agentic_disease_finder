"""Hereditary Vitamin Metabolism Atlas — 8-Gene Reference
BTD-HLCS-SLC19A3-MTHFR-SLC52A2-FLAD1-TCN2-AMN
320 patients (8 x 40), seeds 2638-2645.
Endpoints: /api/hereditary-vitamin-metabolism-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "BTD",
        "protein": (
            "BTD -- 3p25.1 AR -- 543aa -- Biotinidase-67kDa-"
            "Lysosomal-Biotinamide-Amidohydrolase-Biotin-Recycling-AR -- OMIM-Gene-609019-Disease-BiotinidaseDeficiency-253260"
        ),
        "locus": "3p25.1",
        "protein_size": "543 aa / 67 kDa",
        "inheritance": (
            "AR (biallelic BTD loss of function); Biotinidase deficiency; prevalence 1:60,000 (profound) to 1:40,000 (all forms); "
            "Profound deficiency (<10% activity): seizures, hypotonia, cutaneous signs, hearing loss, optic atrophy if untreated; "
            "Partial deficiency (10-30% activity): symptomatic with illness/metabolic stress; "
            "Onset: neonatal to early infancy in untreated profound; "
            "Newborn screening by fluorimetric biotinidase activity assay — universal in USA, UK, Canada, Australia; "
            "Biotin 5-10mg/day oral: curative if started early; symptoms fully preventable; "
            "Untreated: progressive metabolic acidosis, alopecia, perioral/perinasal dermatitis, hearing loss (40-75%), vision loss, neurodevelopmental regression"
        ),
        "disease_category": (
            "Biotinidase deficiency; inborn error of biotin recycling; water-soluble vitamin metabolism disorder; "
            "BTD encodes biotinidase — enzyme that cleaves biocytin (biotin-lysine) to recycle biotin from protein turnover; "
            "Loss of BTD → biotin deficiency secondary to inability to recycle free biotin → "
            "multiple carboxylase deficiency (PCC, MCC, ACC, PC) → impaired gluconeogenesis + fatty acid synthesis + amino acid catabolism; "
            "Organic aciduria: 3-methylcrotonylglycinuria, propionic aciduria, methylcitric aciduria on urine OA; "
            "p.Asp444His: most common severe allele worldwide; c.98_104del7ins3: common severe allele; "
            "p.Arg538Cys + p.Asp444His compound het: partial deficiency; "
            "BIOTIN 5-10mg/day: free water-soluble vitamin bypasses recycling defect; completely curative if pre-symptomatic; "
            "NBS has eliminated symptomatic BTD deficiency in screened populations"
        ),
        "disease_pathway": (
            "BTD encodes biotinidase, a serum and lysosomal enzyme that releases free biotin from biocytin "
            "(biotinyl-epsilon-lysine) generated during protein turnover and intestinal proteolysis of dietary biotinyl-proteins. "
            "Free biotin is then recycled by holocarboxylase synthetase (HLCS) to re-biotinylate four essential carboxylases: "
            "PYRUVATE CARBOXYLASE (PC): gluconeogenesis from pyruvate → oxaloacetate; "
            "PROPIONYL-CoA CARBOXYLASE (PCC): propionate → methylmalonyl-CoA; "
            "3-METHYLCROTONYL-CoA CARBOXYLASE (MCC): leucine catabolism; "
            "ACETYL-CoA CARBOXYLASE (ACC): fatty acid synthesis. "
            "Loss of BTD → biocytin accumulates → free biotin not recycled → progressive biotin deficiency → "
            "all four carboxylases become unactivated → "
            "METABOLIC ACIDOSIS: lactic acidosis + organic aciduria (3-methylcrotonylglycinuria, propionylcarnitine elevation); "
            "CNS: seizures (often infantile spasms), hypotonia, developmental regression, optic atrophy; "
            "SKIN: seborrheic dermatitis, alopecia totalis (biotin-related), periorificial dermatitis — classic triad; "
            "HEARING: sensorineural hearing loss (40-75%) — cochlear biotin-dependent enzyme deficiency; "
            "TREATMENT: oral free biotin 5-10mg/day completely bypasses recycling defect — curative."
        ),
        "pathognomonic": (
            "BTD DIAGNOSTIC CLUSTER: "
            "1) ALOPECIA + PERIORAL DERMATITIS TRIAD: alopecia totalis + seborrheic/periorificial dermatitis + "
            "conjunctivitis — biotin deficiency CLASSIC TRIAD; not all present simultaneously; "
            "2) SENSORINEURAL HEARING LOSS (40-75%): cochlear damage from biotin-dependent enzyme deficiency; "
            "may be permanent even with biotin treatment if established; mandatory audiologic follow-up; "
            "3) SEIZURES: onset 1-6 months in profound untreated; myoclonic or infantile spasms; "
            "anticonvulsants INEFFECTIVE — biotin treats underlying cause; "
            "4) LACTIC ACIDOSIS + ORGANIC ACIDURIA: 3-methylcrotonylglycinuria on urine organic acids; "
            "propionylcarnitine on acylcarnitine profile; "
            "5) BIOTINIDASE ENZYME ASSAY: <10% activity (profound), 10-30% (partial) in leukocytes/serum; "
            "6) NEWBORN SCREEN: fluorimetric BTD assay detects profound and partial deficiency; "
            "UNTREATED SEVERE: vision loss (optic atrophy), hearing loss, neurodevelopmental delay — all preventable with NBS + biotin; "
            "BIOTIN RESPONSE: dramatic clinical improvement within weeks — hallmark of diagnosis"
        ),
        "treatment": (
            "BIOTINIDASE DEFICIENCY TREATMENT: "
            "FREE BIOTIN 5-10 mg/day orally: bypasses recycling defect entirely; "
            "PROFOUND: biotin 10 mg/day — start IMMEDIATELY on NBS recall without waiting for confirmatory tests; "
            "PARTIAL: biotin 5-10 mg/day especially during febrile illness; "
            "OUTCOMES: pre-symptomatic treatment = completely normal development; "
            "POST-SYMPTOMATIC: seizures and metabolic symptoms resolve promptly; "
            "hearing loss and vision loss may be permanent — emphasises NBS importance; "
            "MONITORING: biotinidase activity (treated patients will not normalize — not a monitoring marker); "
            "urine organic acids + acylcarnitines to assess metabolic control; "
            "annual audiology (SNHL can progress despite biotin); "
            "PREGNANCY: maternal BTD deficiency managed with biotin throughout; "
            "fetal protection against BTD deficiency requires maternal biotin supplementation; "
            "DIET: no dietary restriction required; normal diet; "
            "DRUG INTERACTIONS: anticonvulsants (especially valproate, carbamazepine) reduce biotin — increase dose during antiepileptic use; "
            "RAW EGG WHITE: contains avidin which binds biotin — avoid; processed/cooked eggs safe"
        ),
        "key_features": [
            "BTD (Biotinidase deficiency): AR; biotinidase enzyme deficiency; biotin recycling defect; 1:60,000 profound",
            "Classic triad: alopecia + seborrheic/periorificial dermatitis + conjunctivitis (not always complete)",
            "SENSORINEURAL HEARING LOSS (40-75%) — cochlear damage; may be permanent even with treatment",
            "Seizures (myoclonic/infantile spasms): anticonvulsants INEFFECTIVE — only biotin treats underlying cause",
            "Lactic acidosis + 3-methylcrotonylglycinuria + propionylcarnitine on metabolic screen",
            "BIOTIN 5-10 mg/day orally: CURATIVE if started pre-symptomatically via NBS",
            "Newborn screening by fluorimetric biotinidase assay — universal in USA, UK, Canada, Australia",
            "p.Asp444His: most common severe allele worldwide (approximately 50% of severe alleles)",
        ],
        "key_ddx": [
            "HLCS (holocarboxylase synthetase) deficiency: neonatal/early onset; biotin-responsive; same metabolites; HLCS enzyme assay or molecular confirms",
            "Isolated propionic acidemia (PCCA/PCCB): propionylcarnitine elevated; NO alopecia/dermatitis; biotinidase assay normal",
            "Isolated MCC deficiency: 3-methylcrotonylglycinuria; often benign in isolation; no systemic features",
            "Biotin dietary deficiency (raw egg white, TPN): same biochemistry; no BTD mutation; history key",
            "SLC19A3 deficiency (BTBGD): stress-triggered encephalopathy; striatum lesions; BOTH biotin+thiamine required; no alopecia/dermatitis",
        ],
    },
    {
        "gene": "HLCS",
        "protein": (
            "HLCS -- 21q22.13 AR -- 726aa -- Holocarboxylase-Synthetase-82kDa-"
            "Mitochondrial-Cytoplasmic-Biotin-Ligase-All-4-Carboxylases-AR -- OMIM-Gene-609018-Disease-HLCS-Deficiency-253270"
        ),
        "locus": "21q22.13",
        "protein_size": "726 aa / 82 kDa",
        "inheritance": (
            "AR (biallelic HLCS loss of function); Holocarboxylase synthetase deficiency; "
            "Multiple carboxylase deficiency (MCD) — neonatal form; prevalence ~1:87,000; "
            "Earlier and more severe onset than BTD deficiency; "
            "Neonatal crisis: metabolic acidosis, hypotonia, vomiting, hyperammonaemia in first days of life; "
            "Skin: seborrheic/perioral dermatitis, alopecia (often milder than BTD); "
            "CNS: seizures, hypotonia, coma without treatment; "
            "Biotin-responsive: 10-40 mg/day — higher dose than BTD because defect is in enzyme (Km mutants); "
            "Some mutations: Km mutants respond to pharmacological biotin (overcome reduced affinity); "
            "Others: functionally null — treated with max tolerated biotin dose"
        ),
        "disease_category": (
            "HLCS deficiency; inborn error of biotin ligation; multiple carboxylase deficiency (neonatal form); "
            "HLCS encodes holocarboxylase synthetase — the enzyme that covalently attaches biotin (via epsilon-amino of lysine) "
            "to the apocarboxylase forms of all four mitochondrial/cytoplasmic biotin-dependent carboxylases: "
            "pyruvate carboxylase (PC), propionyl-CoA carboxylase (PCC), 3-methylcrotonyl-CoA carboxylase (MCC), "
            "acetyl-CoA carboxylase (ACC1 in cytoplasm, ACC2 in mitochondria). "
            "Loss of HLCS → all four carboxylases inactive regardless of biotin dietary availability → "
            "more severe/earlier presentation than BTD (where biotin recycling is defective but dietary biotin still incorporated); "
            "p.Leu216Arg: most common severe Km-mutant allele; c.1522C>T (p.Arg508Trp): common severe; "
            "BIOTIN 10-40 mg/day: Km mutants respond well; "
            "NBS: elevated 3-methylcrotonylglycine + propionylcarnitine on MSMS-NBS"
        ),
        "disease_pathway": (
            "HLCS encodes holocarboxylase synthetase, a bifunctional enzyme (mitochondrial and cytoplasmic forms) "
            "that catalyses the ATP-dependent covalent attachment of biotin to the epsilon-amino group of "
            "specific lysine residues in the biotin-dependent carboxylase apoenzymes. "
            "Without functional HLCS: all four carboxylases remain as inactive apoenzymes: "
            "1) PYRUVATE CARBOXYLASE: gluconeogenesis blocked → lactic acidosis; "
            "2) PROPIONYL-CoA CARBOXYLASE: propionate catabolism blocked → propionic acidaemia phenotype; "
            "3) 3-METHYLCROTONYL-CoA CARBOXYLASE: leucine catabolism blocked → 3-methylcrotonylglycinuria; "
            "4) ACETYL-CoA CARBOXYLASE: fatty acid synthesis impaired. "
            "METABOLIC CONSEQUENCES: severe lactic acidosis + hyperammonaemia (urea cycle overwhelmed by anaplerosis failure) + "
            "organic aciduria — detectable by urine OA and acylcarnitine profile (C5-OH elevated = 3-methylcrotonylcarnitine; "
            "C3 elevated = propionylcarnitine). "
            "BIOTIN PHARMACOLOGY: Km mutants have reduced affinity for biotin — high-dose biotin saturates despite low Km; "
            "functionally null variants: limited response; "
            "BIOTIN DOSE: 10-40 mg/day (vs 5-10 mg BTD) required to overcome Km defect. "
            "NBS: tandem mass spectrometry detects C5-OH (3-methylcrotonylcarnitine) elevation."
        ),
        "pathognomonic": (
            "HLCS DIAGNOSTIC CLUSTER: "
            "1) NEONATAL METABOLIC CRISIS: severe metabolic acidosis (pH <7.1) + hyperammonaemia + "
            "lactic acidosis in first 24-72 hours of life — earlier than BTD; "
            "2) TRIPLE ORGANIC ACID PATTERN on urine OA: 3-methylcrotonylglycinuria + lactic aciduria + "
            "methylcitric aciduria (propionate derived); "
            "3) ACYLCARNITINE PROFILE: C5-OH (3-methylcrotonylcarnitine) + C3 (propionylcarnitine) ELEVATED — "
            "NBS trigger; "
            "4) HYPERAMMONAEMIA: often >200 μmol/L in severe neonatal crisis; secondary urea cycle dysfunction; "
            "5) SKIN: seborrheic/perioral dermatitis + alopecia (less prominent than BTD but present); "
            "6) BIOTIN RESPONSE TEST: dramatic metabolic normalisation within 24-48 hours of 10-40 mg/day biotin; "
            "confirmatory if molecular/enzyme assay pending; "
            "7) HLCS ENZYME ACTIVITY: reduced in fibroblasts; Km assay demonstrates biotin-responsiveness; "
            "HOLCOARBOXYLASE assay: all four carboxylases low without biotin + normalise with high-dose biotin in Km mutants"
        ),
        "treatment": (
            "HLCS DEFICIENCY TREATMENT: "
            "FREE BIOTIN 10-40 mg/day orally: Km-mutant alleles (most common) respond well to pharmacological doses; "
            "Start IMMEDIATELY in neonatal crisis; do not wait for confirmation; "
            "ACUTE MANAGEMENT: glucose infusion (stop catabolism); protein restriction (temporary); "
            "bicarbonate if severe acidosis; ammonia scavengers if severe hyperammonaemia (Na benzoate/phenylacetate); "
            "carnitine supplementation (200-300 mg/kg/day) to prevent secondary carnitine deficiency; "
            "BIOTIN DOSE TITRATION: start 10 mg/day; increase to 20-40 mg/day in poor responders or functionally null variants; "
            "MAINTENANCE: lifelong biotin; do NOT stop or reduce on parent/physician discretion; "
            "MONITORING: urine organic acids + acylcarnitines every 3-6 months; "
            "plasma amino acids (hyperglycaemia from PC deficiency); "
            "plasma lactate; plasma ammonia if febrile; "
            "OUTCOMES: Km mutants with NBS and early biotin = normal development; "
            "functionally null variants: poorer outcome despite max biotin; neurological sequelae common; "
            "FEBRILE ILLNESS PROTOCOL: double biotin dose during illness; metabolic team contact"
        ),
        "key_features": [
            "HLCS (Holocarboxylase synthetase deficiency): AR; neonatal form of multiple carboxylase deficiency; 1:87,000",
            "Earlier and more severe than BTD: neonatal metabolic crisis (metabolic acidosis + hyperammonaemia) in first days",
            "TRIPLE ORGANIC ACID PATTERN: 3-methylcrotonylglycinuria + lactic aciduria + methylcitric aciduria",
            "NBS marker: C5-OH (3-methylcrotonylcarnitine) + C3 (propionylcarnitine) on tandem MS",
            "Biotin 10-40 mg/day (higher dose than BTD): Km-mutant alleles respond well to pharmacological biotin",
            "p.Leu216Arg: most common Km-mutant allele worldwide (~40% of alleles in European populations)",
            "Functionally null variants: poor biotin response; neurological sequelae common despite treatment",
            "HLCS enzyme assay in fibroblasts: all four carboxylases low; Km assay predicts biotin-responsiveness",
        ],
        "key_ddx": [
            "BTD (biotinidase) deficiency: older onset (1-6 months not neonatal); prominent alopecia/dermatitis; same metabolites; BTD enzyme assay distinguishes",
            "Propionic acidemia (PCCA/PCCB): C3 elevation; NO 3-methylcrotonylglycinuria; no biotin response; enzyme assay",
            "Isolated MCC (MCCA/MCCB) deficiency: C5-OH elevated; usually benign; no systemic features; no biotin response",
            "Methylmalonic acidemia (MUT/MMACHC): C3 elevated; methylmalonyl-CoA not propionyl-CoA disorder; no C5-OH",
            "Lactic acidosis syndromes (pyruvate carboxylase alone): no organic aciduria pattern; isolated PC deficiency",
        ],
    },
    {
        "gene": "SLC19A3",
        "protein": (
            "SLC19A3 -- 2q36.3 AR -- 500aa -- Thiamine-Transporter-2-SLC19A3-"
            "Plasma-Membrane-Thiamine-Uptake-56kDa-BTBGD-AR -- OMIM-Gene-606152-Disease-BTBGD-607483"
        ),
        "locus": "2q36.3",
        "protein_size": "500 aa / 56 kDa",
        "inheritance": (
            "AR (biallelic SLC19A3 loss of function); Biotin-thiamine responsive basal ganglia disease (BTBGD); "
            "also known as thiamine metabolism dysfunction syndrome 2 (THMD2); "
            "Prevalence: enriched in Middle East (Saudi Arabia, Bahrain); founder effect; "
            "Onset: subacute encephalopathy triggered by febrile illness or vaccination; "
            "BIPHASIC PRESENTATION: (1) acute/subacute crisis — confusion, seizures, movement disorder (dystonia); "
            "(2) inter-crisis period — subtle motor/cognitive impairment; "
            "MRI: BILATERAL SYMMETRIC BASAL GANGLIA LESIONS (caudate + putamen) + cortical ribbon-like T2 hyperintensity; "
            "PATHOGNOMONIC MRI pattern — differential narrows to BTBGD, Leigh syndrome, Wilson's, NPC; "
            "BIOTIN + THIAMINE TREATMENT: both required; thiamine alone insufficient; biotin alone insufficient"
        ),
        "disease_category": (
            "Biotin-thiamine responsive basal ganglia disease (BTBGD); inborn error of thiamine transport; "
            "SLC19A3 encodes thiamine transporter 2 (ThTr2), a plasma membrane transporter mediating "
            "high-affinity cellular uptake of free thiamine (vitamin B1) — particularly important in brain; "
            "Loss of SLC19A3 → impaired thiamine uptake in CNS → "
            "thiamine pyrophosphate (TPP) cofactor deficiency in brain → "
            "pyruvate dehydrogenase (PDH) and alpha-ketoglutarate dehydrogenase (KGDH) dysfunction → "
            "energy failure in high-metabolic-demand neurons (basal ganglia, cortex); "
            "BIOTIN component: biotin supplementation enhances SLC19A3 transcription via biotin-responsive elements; "
            "THIAMINE component: high-dose thiamine overcomes transport defect via SLC19A2 (alternative transporter); "
            "p.Glu320Gln (c.958G>C): Saudi founder variant (~60% of BTBGD alleles worldwide); "
            "TRIGGERS: febrile illness, vaccination — increased metabolic demand exposes transport insufficiency"
        ),
        "disease_pathway": (
            "SLC19A3 encodes thiamine transporter 2 (ThTr2), expressed at brain capillary endothelium and choroid plexus, "
            "which is the primary route for thiamine entry into the CNS. "
            "Thiamine (vitamin B1) → phosphorylated to thiamine pyrophosphate (TPP) inside cells → TPP is essential cofactor for: "
            "PYRUVATE DEHYDROGENASE (PDH): pyruvate → acetyl-CoA (mitochondrial entry); "
            "ALPHA-KETOGLUTARATE DEHYDROGENASE (KGDH): TCA cycle intermediate; "
            "TRANSKETOLASE: pentose phosphate pathway; "
            "BRANCHED-CHAIN KETOACID DEHYDROGENASE (BCKDH): BCAA catabolism. "
            "Loss of SLC19A3 → reduced CNS thiamine → TPP deficiency in neurons → "
            "ENERGY FAILURE: basal ganglia and cortex most vulnerable (high metabolic demand + limited glycolytic reserve); "
            "CRISIS TRIGGER: febrile illness → increased metabolic rate → overwhelms residual transport; "
            "BASAL GANGLIA: caudate, putamen, globus pallidus involvement → dystonia + chorea; "
            "CORTEX: ribbon-like T2 changes → seizures + confusion. "
            "TREATMENT MECHANISM: biotin upregulates SLC19A3 gene expression via biotin-responsive elements; "
            "high-dose thiamine uses alternative SLC19A2 transporter to partially compensate."
        ),
        "pathognomonic": (
            "BTBGD DIAGNOSTIC CLUSTER: "
            "1) BILATERAL SYMMETRIC BASAL GANGLIA MRI LESIONS: T2/FLAIR hyperintensity caudate + putamen + "
            "variable globus pallidus; DWI restriction in acute phase; "
            "cortical ribbon-like T2 changes (perirolandic/parieto-occipital) — KEY DDx FROM LEIGH SYNDROME; "
            "2) STRESS-TRIGGERED ACUTE ENCEPHALOPATHY: fever/vaccination → subacute onset confusion, ataxia, "
            "dystonia, seizures within days; partial or complete recovery between episodes; "
            "PATHOGNOMONIC pattern: recurrent stress-triggered crises with basal ganglia involvement; "
            "3) DYSTONIA + DYSARTHRIA: subacute onset; movement disorder during crisis; "
            "4) CSF: lactate may be mildly elevated; pyruvate normal or mildly elevated; "
            "5) BIOTIN + THIAMINE RESPONSE: dramatic clinical and MRI improvement within days to weeks; "
            "partial MRI resolution (signal may persist but shrinks); "
            "6) SLC19A3 MOLECULAR: biallelic pathogenic variants; p.Glu320Gln Saudi founder detectable on targeted panel; "
            "FOUNDER VARIANT TESTING: p.Glu320Gln first in Middle Eastern patients — positive confirms diagnosis; "
            "MRI PATTERN DISTINGUISHES: BTBGD = caudate/putamen + cortical; Leigh = dorsal midbrain + periaqueductal"
        ),
        "treatment": (
            "BTBGD TREATMENT: "
            "BIOTIN 5-10 mg/kg/day + THIAMINE 100-300 mg/day: BOTH required; neither alone sufficient; "
            "Start immediately in any child with symmetric basal ganglia lesions + encephalopathy; do not wait for molecular; "
            "ACUTE CRISIS: IV thiamine 100 mg TDS (thiamine in glucose — always precede glucose with thiamine); "
            "IV biotin if oral route compromised; "
            "MAINTENANCE: oral biotin + thiamine LIFELONG; never stop; "
            "FEVER MANAGEMENT: aggressive antipyretics; double biotin+thiamine during febrile illness; "
            "VACCINATION PROTOCOL: consider pre- and post-vaccine biotin+thiamine dose increase; "
            "discuss timing with metabolic team; "
            "SEIZURE MANAGEMENT: anticonvulsants as needed; treat underlying metabolic cause preferentially; "
            "MONITORING: MRI every 6-12 months initially (assess lesion evolution); "
            "Neuropsychological assessment annually; "
            "plasma thiamine + biotin levels if compliance uncertain; "
            "OUTCOMES: early treatment = good neurological outcome; delays lead to permanent basal ganglia damage; "
            "MRI lesions may persist but clinical outcome good with treatment; "
            "GENETIC COUNSELLING: Saudi Arabian families: test p.Glu320Gln founder first"
        ),
        "key_features": [
            "SLC19A3 (BTBGD): AR; thiamine transporter 2 deficiency; brain thiamine transport defect; Saudi founder variant p.Glu320Gln",
            "BILATERAL SYMMETRIC BASAL GANGLIA LESIONS on MRI: T2/DWI caudate+putamen + cortical ribbon changes PATHOGNOMONIC",
            "STRESS-TRIGGERED ACUTE ENCEPHALOPATHY: febrile illness/vaccination triggers crisis — distinguishes from Leigh",
            "BOTH BIOTIN + THIAMINE REQUIRED: biotin alone insufficient; thiamine alone insufficient; must combine",
            "Biotin 5-10 mg/kg/day + Thiamine 100-300 mg/day: start IMMEDIATELY on clinical suspicion",
            "Crisis management: IV thiamine BEFORE glucose; aggressive antipyretics to prevent trigger",
            "Saudi/Middle Eastern founder enrichment: p.Glu320Gln ~60% of BTBGD alleles worldwide",
            "MRI DDx: BTBGD = caudate/putamen + cortex; Leigh = dorsal midbrain/periaqueductal — critical distinction",
        ],
        "key_ddx": [
            "Leigh syndrome (mitochondrial): dorsal midbrain/periaqueductal gray + BG; does NOT respond to biotin+thiamine; mtDNA/nuclear OXPHOS variants",
            "Wernicke encephalopathy: acquired thiamine deficiency; mammillary bodies + thalami; no SLC19A3 mutation; malnutrition/alcohol history",
            "Wilson disease (ATP7B): BG involvement; KF rings; low ceruloplasmin; copper studies distinguish",
            "BTD/HLCS deficiency: biotin-responsive but no thiamine component; alopecia/dermatitis; organic aciduria",
            "Niemann-Pick C (NPC1): vertical supra-nuclear gaze palsy; cataplexy; filipin test/oxysterols; no BG MRI pattern",
        ],
    },
    {
        "gene": "MTHFR",
        "protein": (
            "MTHFR -- 1p36.22 AR -- 698aa -- Methylenetetrahydrofolate-Reductase-74kDa-"
            "Cytoplasmic-NADPH-Flavoenzyme-Homocysteine-Remethylation-AR-Severe -- OMIM-Gene-607093-Disease-Homocystinuria-MTHFR-236250"
        ),
        "locus": "1p36.22",
        "protein_size": "698 aa / 74 kDa",
        "inheritance": (
            "AR (biallelic MTHFR severe LOF); Severe MTHFR deficiency / homocystinuria type 4 / homocystinuria without methylmalonic aciduria; "
            "IMPORTANT: common MTHFR variants c.677C>T (p.Ala222Val/C677T) and c.1298A>C are thermolabile variants — "
            "NOT disease-causing; heterozygous and even homozygous C677T does NOT cause MTHFR deficiency disease; "
            "SEVERE MTHFR deficiency (rare biallelic null/severe mutations): "
            "early-onset severe neurological disease; apnoea, seizures, microcephaly, spasticity; "
            "severe hyperhomocysteinaemia (>100 μmol/L) with LOW-NORMAL or LOW methionine (KEY DDx from CBS deficiency); "
            "treatment: betaine + folate + methyl-B12 ± pyridoxine"
        ),
        "disease_category": (
            "Severe MTHFR deficiency; inborn error of folate-dependent homocysteine remethylation; "
            "MTHFR encodes methylenetetrahydrofolate reductase, which catalyses the irreversible reduction of "
            "5,10-methylenetetrahydrofolate (5,10-MTHF) to 5-methyltetrahydrofolate (5-MTHF); "
            "5-MTHF is the methyl donor for methionine synthase (MTR/MS) to remethylate homocysteine → methionine; "
            "Loss of MTHFR → 5-MTHF depleted → homocysteine cannot be remethylated → "
            "HYPERHOMOCYSTEINAEMIA + LOW METHIONINE (not elevated as in CBS deficiency — critical DDx); "
            "SAM (S-adenosyl-methionine) depleted → methylation reactions impaired → "
            "DNA methylation, myelin synthesis (phosphatidylcholine), catecholamine methylation all impaired; "
            "COMMON VARIANT CAVEAT: C677T/A1298C are thermolabile variants causing 30-70% activity reduction; "
            "they are NOT rare disease — they are population polymorphisms (C677T homozygous: 10-15% of populations); "
            "SEVERE DISEASE: rare biallelic null/severe pathogenic variants; p.Arg157Gln; Del exon 6"
        ),
        "disease_pathway": (
            "MTHFR catalyses the NADPH-dependent reduction of 5,10-methylenetetrahydrofolate to 5-methyltetrahydrofolate (5-MTHF), "
            "which is the primary methyl donor in the methionine cycle. "
            "METHIONINE CYCLE: Homocysteine + 5-MTHF → Methionine (via MTR/MS + B12 cofactor) → "
            "Methionine → SAM (S-adenosylmethionine) via MAT → "
            "SAM donates methyl group → SAH → Homocysteine (cycle completes). "
            "Severe MTHFR deficiency → 5-MTHF depleted → homocysteine remethylation impaired → "
            "HOMOCYSTEINE ACCUMULATES: vascular toxicity, neurological toxicity, disulfide bond interference; "
            "METHIONINE LOW: SAM depleted → hypomethylation: "
            "MYELIN: reduced phosphatidylcholine synthesis → demyelination; "
            "DNA: reduced methylation → chromatin dysregulation; "
            "CATECHOLAMINES: reduced neurotransmitter methylation → psychiatric symptoms; "
            "THROMBOSIS: homocysteine activates endothelium, impairs anticoagulant pathways; "
            "KEY BIOCHEMISTRY: low methionine + elevated homocysteine distinguishes from CBS (where methionine is ELEVATED). "
            "COMMON VARIANTS (C677T/A1298C): do NOT reproduce this severe phenotype — only cause mild homocysteine elevation."
        ),
        "pathognomonic": (
            "SEVERE MTHFR DEFICIENCY DIAGNOSTIC CLUSTER: "
            "1) HYPERHOMOCYSTEINAEMIA + LOW (or LOW-NORMAL) METHIONINE: PATHOGNOMONIC DDx FROM CBS DEFICIENCY "
            "(CBS = elevated methionine + elevated homocysteine); MTHFR = low methionine + elevated homocysteine; "
            "2) NEONATAL/INFANTILE NEUROLOGICAL PRESENTATION: apnoea, seizures, microcephaly, spasticity, "
            "profound hypotonia; can present as neonatal encephalopathy; "
            "3) MRI: periventricular white matter signal abnormality; delayed myelination; brain atrophy; "
            "4) MEGALOBLASTIC ANAEMIA (variable): folate trap effect; macrocytosis; "
            "5) URINE HOMOCYSTEINE: elevated total homocysteine (>100 μmol/L in severe forms); "
            "HOMOCYSTINURIA confirmed (qualitative/quantitative); "
            "6) PLASMA AMINO ACIDS: low methionine = KEY — differentiates from CBS and all other homocystinurias; "
            "7) MTHFR ENZYME ACTIVITY: reduced in fibroblasts; confirms diagnosis; "
            "THERMOLABILITY TEST: C677T homozygous shows thermolabile activity but NOT severe deficiency; "
            "COMMON VARIANT WARNING: report C677T/A1298C as polymorphisms — NOT diagnostic of MTHFR deficiency disease; "
            "never report these as disease-causing in a rare disease context without biallelic severe mutations"
        ),
        "treatment": (
            "SEVERE MTHFR DEFICIENCY TREATMENT: "
            "BETAINE 100-200 mg/kg/day: alternative remethylation donor (betaine→methionine via BHMT); "
            "raises methionine; reduces homocysteine; most effective agent; "
            "5-METHYLTETRAHYDROFOLATE (5-MTHF): bypass MTHFR defect with downstream product; "
            "HYDROXOCOBALAMIN (B12) IM or high-dose oral: cofactor for methionine synthase; "
            "PYRIDOXINE B6: not primary for MTHFR (unlike CBS) but cofactor; trial if response incomplete; "
            "METHIONINE SUPPLEMENTATION: if methionine critically low; "
            "FOLATE: 5 mg/day — replenish folate pool; "
            "MONITORING: plasma total homocysteine (target <50 μmol/L) + plasma methionine + MRI 6-12 monthly; "
            "ACUTE CRISIS: betaine push + B12 IV; "
            "OUTCOMES: early treatment limits neurological progression; "
            "established neurological damage (demyelination) may partially reverse with aggressive treatment; "
            "COMMON VARIANT MANAGEMENT: C677T/A1298C homozygous: "
            "mild hyperhomocysteinaemia only; treat with folic acid 400-800 μg/day + standard diet; "
            "NOT the same as rare MTHFR deficiency disease — do not manage with betaine/B12 protocols"
        ),
        "key_features": [
            "MTHFR (severe deficiency): AR; rare biallelic null/severe variants; DISTINCT from common C677T/A1298C polymorphisms",
            "LOW METHIONINE + elevated homocysteine: PATHOGNOMONIC DDx from CBS deficiency (where methionine is HIGH)",
            "COMMON VARIANTS C677T/A1298C are NOT disease: population polymorphisms; do NOT cause MTHFR deficiency disease",
            "Neonatal/infantile: apnoea, seizures, microcephaly, spasticity, hypotonia; white matter disease on MRI",
            "Megaloblastic anaemia (variable): folate trap; macrocytosis",
            "Treatment: betaine (primary) + 5-MTHF + hydroxocobalamin + folate",
            "Betaine: raises methionine via BHMT pathway; most effective agent for homocysteine reduction",
            "MTHFR enzyme assay: thermolabile variants reduced 30-70% — NOT same as severe deficiency",
        ],
        "key_ddx": [
            "CBS deficiency (classic homocystinuria): elevated methionine + homocysteine; marfanoid habitus; lens subluxation; B6-responsive; CBS assay",
            "cblC disease (MMACHC): homocystinuria + methylmalonic aciduria COMBINED; normal/low methionine; B12-responsive",
            "cblE/G disease (MTRR/MTR): homocystinuria without MMA; similar to MTHFR; normal methionine; low methyl-B12",
            "Transcobalamin II deficiency (TCN2): B12 deficiency state; megaloblastic anaemia; homocysteine mildly elevated; TCN2 assay",
            "MTHFR C677T/A1298C (common variants): mild homocysteine elevation; NOT rare disease; folate supplementation only",
        ],
    },
    {
        "gene": "SLC52A2",
        "protein": (
            "SLC52A2 -- 8q24.13 AR -- 460aa -- Riboflavin-Transporter-2-SLC52A2-"
            "Plasma-Membrane-Riboflavin-Uptake-50kDa-BVVL2-AR -- OMIM-Gene-607882-Disease-BVVL2-614707"
        ),
        "locus": "8q24.13",
        "protein_size": "460 aa / 50 kDa",
        "inheritance": (
            "AR (biallelic SLC52A2 loss of function); Riboflavin transporter deficiency type 2 (RTD2); "
            "Brown-Vialetto-Van Laere syndrome 2 (BVVL2); "
            "SLC52A3 (same phenotype = BVVL1, RTD1); "
            "Prevalence: rare; worldwide distribution; no founder enrichment; "
            "Onset: childhood to early adulthood (bimodal: infant-onset severe, childhood/adolescent onset); "
            "CARDINAL FEATURES: sensorineural hearing loss (often first symptom) + pontobulbar palsy (cranial nerve palsies) + "
            "sensory-predominant neuropathy (axonal); "
            "RESPIRATORY FAILURE: lower cranial nerve palsies → dysphagia/aspiration → respiratory failure; "
            "RIBOFLAVIN HIGH-DOSE TREATMENT: 10-40 mg/kg/day oral; dramatic reversal; "
            "can reverse established hearing loss, neuropathy, respiratory failure"
        ),
        "disease_category": (
            "Riboflavin transporter deficiency type 2 (RTD2); Brown-Vialetto-Van Laere syndrome 2; "
            "inborn error of riboflavin (vitamin B2) transport; "
            "SLC52A2 encodes riboflavin transporter 2 (RFT2/hRFT2), a plasma membrane transporter for cellular uptake of riboflavin; "
            "Riboflavin → FAD (flavin adenine dinucleotide) and FMN (flavin mononucleotide) — essential cofactors for: "
            "electron transport chain (Complex I/II), fatty acid β-oxidation, amino acid catabolism; "
            "Loss of SLC52A2 → impaired riboflavin uptake, especially in neural tissue → "
            "FAD/FMN deficiency → electron transport chain dysfunction + multiple acyl-CoA dehydrogenase deficiency (MADD) → "
            "cranial nerve nuclei/brainstem + peripheral nerve particularly vulnerable; "
            "BIOCHEMISTRY: plasma riboflavin low; urine organic acids: MADD-like (glutaric + ethylmalonic + C8/C10/C12 acylcarnitines); "
            "RIBOFLAVIN 10-40 mg/kg/day: high dose overcomes transport defect by mass action"
        ),
        "disease_pathway": (
            "SLC52A2 encodes RFT2 (riboflavin transporter 2), expressed highly in brain and peripheral nerve, "
            "responsible for high-affinity cellular riboflavin import. "
            "Riboflavin (vitamin B2) is phosphorylated to FMN → then adenylated to FAD inside cells. "
            "FAD/FMN are essential cofactors for: "
            "COMPLEX I (NADH dehydrogenase): FAD in subunits; "
            "COMPLEX II (succinate dehydrogenase): FAD; "
            "ACYL-CoA DEHYDROGENASES: VLCAD, LCAD, MCAD, SCAD — all require FAD → β-oxidation; "
            "MULTIPLE ACYL-CoA DEHYDROGENASE DEFICIENCY (MADD/GA2): when all FAD-dependent dehydrogenases fail; "
            "RIBOFLAVIN KINASE and FAD SYNTHASE: require riboflavin substrate. "
            "Loss of SLC52A2 → FAD/FMN depleted especially in high-metabolic CNS cells → "
            "BRAINSTEM CRANIAL NERVE NUCLEI: VII, VIII, IX, X, XII → facial palsy, SNHL, dysphagia, dysphonia, tongue atrophy; "
            "PERIPHERAL NERVE: axonal sensory > motor neuropathy; "
            "RESPIRATORY MUSCLES: bulbar + phrenic nerve → respiratory failure. "
            "HIGH-DOSE RIBOFLAVIN: mass action bypasses transporter defect — riboflavin enters via passive diffusion at high concentration."
        ),
        "pathognomonic": (
            "RTD2/BVVL2 DIAGNOSTIC CLUSTER: "
            "1) SENSORINEURAL HEARING LOSS: often FIRST symptom; bilateral; progressive; severe/profound; "
            "may precede pontobulbar palsy by months to years; audiometry mandatory; "
            "2) PONTOBULBAR PALSY: lower cranial nerves (VII, IX, X, XII) → facial weakness, dysphagia, dysphonia, "
            "tongue fasciculations/atrophy; bulbar signs DISTINCTIVE for RTD; "
            "3) AXONAL SENSORY NEUROPATHY: peripheral neuropathy sensory-predominant; nerve conduction studies; "
            "4) RESPIRATORY FAILURE: bulbar + respiratory muscle involvement → ventilator dependence if untreated; "
            "may present as unexplained respiratory failure in child; "
            "5) ACYLCARNITINE PROFILE: multiple acyl-CoA elevation (C6-OH, C8, C10, C12 dicarboxylyl) — MADD pattern; "
            "urine organic acids: ethylmalonic, glutaric, methylsuccinic, adipic acids; "
            "6) PLASMA RIBOFLAVIN: low in untreated patients; "
            "7) RIBOFLAVIN RESPONSE TEST: high-dose riboflavin (10-40 mg/kg/day) → remarkable recovery "
            "including reversal of established hearing loss, neuropathy improvement, respiratory weaning; "
            "HEARING LOSS + PONTOBULBAR PALSY COMBINATION: strongly suggests RTD until proven otherwise; "
            "start riboflavin empirically before molecular confirmation"
        ),
        "treatment": (
            "RTD2/BVVL2 TREATMENT: "
            "RIBOFLAVIN 10-40 mg/kg/day orally: CURATIVE in most cases if started promptly; "
            "start empirically in any child with SNHL + pontobulbar palsy — do not wait for molecular; "
            "DOSE: start 10 mg/kg/day; increase to 40 mg/kg/day if insufficient response; "
            "ACUTE/VENTILATED PATIENTS: IV riboflavin (5-phosphate) if parenteral route needed; "
            "RECOVERY: hearing loss may PARTIALLY REVERSE (remarkable — unique among SNHL causes); "
            "neuropathy improves over months; respiratory function recovers; swallowing improves; "
            "MAINTENANCE: lifelong high-dose riboflavin; never reduce without metabolic team review; "
            "RESPIRATORY SUPPORT: ventilatory support as bridging; wean as riboflavin takes effect; "
            "MONITORING: plasma riboflavin/FAD levels; urine organic acids (MADD pattern should normalise); "
            "audiometry every 6 months; nerve conduction studies annually; respiratory function tests; "
            "NUTRITION: dysphagia management; PEG if required in bulbar phase; "
            "GENETIC COUNSELLING: siblings: test immediately; start riboflavin pre-symptomatically; "
            "OUTCOMES: early treatment = excellent; "
            "late treatment = partial recovery (SNHL partial reversal still remarkable); "
            "untreated = progressive respiratory failure and death"
        ),
        "key_features": [
            "SLC52A2 (RTD2/BVVL2): AR; riboflavin transporter 2 deficiency; riboflavin uptake defect in CNS/nerve",
            "SENSORINEURAL HEARING LOSS + PONTOBULBAR PALSY: PATHOGNOMONIC COMBINATION — start riboflavin empirically",
            "Cranial nerve palsies: VII, IX, X, XII → facial weakness, dysphagia, dysphonia, tongue atrophy",
            "RESPIRATORY FAILURE: bulbar + phrenic nerve involvement → ventilator dependence if untreated",
            "MADD-like biochemistry: acylcarnitine profile C6-OH/C8/C10/C12; ethylmalonic/glutaric aciduria",
            "RIBOFLAVIN 10-40 mg/kg/day: CURATIVE; can REVERSE established SNHL — unique among treatable SNHL causes",
            "HEARING LOSS REVERSAL with riboflavin: diagnostic confirmation and distinctive treatment response",
            "SLC52A3 = BVVL1 (RTD1): same phenotype; same treatment; different transporter gene",
        ],
        "key_ddx": [
            "FLAD1 deficiency (FAD synthase): overlapping MADD biochemistry; no SNHL/pontobulbar; riboflavin-responsive; FLAD1 assay",
            "Classical MADD/GA2 (ETFA/ETFB/ETFDH): neonatal onset, cardiomyopathy, severe; some late-onset riboflavin-responsive (ETFDH)",
            "Isolated SNHL (MYO7A/CDH23/OTOF): no pontobulbar/neuropathy; no acylcarnitine changes; cochlear gene panel",
            "Guillain-Barré syndrome: acute; CSF albuminocytological dissociation; often post-infectious; no hearing loss",
            "Kennedy disease (SBMA): adult-onset; XLR; androgen receptor CAG expansion; gynecomastia; bulbar late",
        ],
    },
    {
        "gene": "FLAD1",
        "protein": (
            "FLAD1 -- 1q21.3 AR -- 644aa -- FAD-Synthase-72kDa-"
            "Bifunctional-Mitochondrial-Membrane-FMN-Adenylyltransferase-MADD-Like-AR -- OMIM-Gene-610595-Disease-MADD-Like-255120"
        ),
        "locus": "1q21.3",
        "protein_size": "644 aa / 72 kDa",
        "inheritance": (
            "AR (biallelic FLAD1 loss of function); FAD synthase deficiency; MADD-like riboflavin-responsive multiple acyl-CoA dehydrogenase deficiency; "
            "Overlap with MADD (glutaric aciduria type II); "
            "Onset: childhood to early adulthood; variable severity; "
            "CARDINAL FEATURES: exercise-induced myopathy + fatigue + muscle weakness; lipid storage in muscle; "
            "MADD biochemical profile on acylcarnitines; "
            "RIBOFLAVIN TREATMENT: 10-40 mg/kg/day — often dramatically effective; "
            "DISTINCTION from SLC52A2: no cranial nerve palsies/SNHL; peripheral myopathy dominant; "
            "milder overall course in many; some patients have severe neonatal form with cardiomyopathy"
        ),
        "disease_category": (
            "FLAD1 deficiency; FAD synthase deficiency; riboflavin-responsive MADD-like disorder; "
            "FLAD1 encodes FAD synthase (also called riboflavin kinase/FMN adenylyltransferase bifunctional enzyme), "
            "the enzyme responsible for the two-step synthesis of FAD from riboflavin: "
            "Step 1 (riboflavin kinase domain): riboflavin → FMN; "
            "Step 2 (FMN adenylyltransferase domain): FMN → FAD. "
            "FAD is the essential cofactor for ALL acyl-CoA dehydrogenases (VLCAD, LCAD, MCAD, SCAD, IVD, GCD, GCDH) + "
            "Complex I, Complex II (SDH) + DHODH + other flavoenzymes. "
            "Loss of FLAD1 → FAD synthesis impaired → all FAD-dependent enzymes dysfunctional → "
            "MADD phenotype (similar to ETFA/ETFB deficiency); "
            "RIBOFLAVIN THERAPY: exogenous riboflavin → FMN/FAD via residual FLAD1 activity (hypomorphic mutations) "
            "or alternative pathways; "
            "Specific mutations: p.Arg434Cys and domain-specific variants determine severity"
        ),
        "disease_pathway": (
            "FLAD1 encodes a bifunctional FAD synthase located in mitochondria and cytoplasm: "
            "N-terminal PTAN domain: FMN adenylyltransferase (FMN → FAD, ATP-dependent); "
            "C-terminal riboflavin kinase domain: riboflavin → FMN, ATP-dependent. "
            "Loss of FLAD1 → FAD not synthesised → all FAD-dependent flavoproteins lose cofactor: "
            "ELECTRON TRANSFER FLAVOPROTEIN (ETF): accepts electrons from acyl-CoA dehydrogenases → ETC; "
            "when ETF is FAD-deficient → all acyl-CoA dehydrogenases back up → "
            "FATTY ACID β-OXIDATION: VLCAD, LCAD, MCAD, SCAD — all blocked → C6-C18 acylcarnitine elevation; "
            "AMINO ACID CATABOLISM: IVD (isovaleryl-CoA), GCD (glutaryl-CoA) — blocked → organic aciduria; "
            "COMPLEX I: NADH:ubiquinone oxidoreductase FAD-containing subunits → respiratory chain failure; "
            "LIPID STORAGE MYOPATHY: lipid droplets in muscle fibres on biopsy (accumulation of unoxidised fatty acids); "
            "EXERCISE TRIGGER: metabolic demand reveals insufficient FAD → rhabdomyolysis risk. "
            "RIBOFLAVIN PHARMACOLOGY: high-dose riboflavin → FMN/FAD accumulate via residual FLAD1 activity "
            "or substrate-level overload of alternative pathway."
        ),
        "pathognomonic": (
            "FLAD1 DIAGNOSTIC CLUSTER: "
            "1) LIPID STORAGE MYOPATHY ON MUSCLE BIOPSY: oil red O staining — excessive lipid droplets in type I fibres; "
            "2) MADD BIOCHEMICAL PROFILE: acylcarnitines C6-C18 dicarboxylyl elevation + "
            "urine organic acids: ethylmalonic, glutaric, adipic, methylsuccinic, isovalerylglycine; "
            "3) EXERCISE INTOLERANCE + MYALGIA + PROXIMAL WEAKNESS: onset childhood-adulthood; "
            "rhabdomyolysis triggered by illness or fasting; CK elevation (variable); "
            "4) RIBOFLAVIN RESPONSE: high-dose riboflavin (10-40 mg/kg/day) → metabolic normalisation "
            "of acylcarnitines + improvement of muscle symptoms; "
            "DRAMATIC riboflavin response distinguishes FLAD1 from classical ETFA/ETFB MADD; "
            "5) MUSCLE HISTOLOGY: subsarcolemmal lipid accumulation; ragged-red fibres absent (unlike mitochondrial); "
            "6) ACYLCARNITINE PROFILE TIMING: normal between episodes; elevated during metabolic stress — "
            "repeat testing during illness/exercise stress; "
            "OVERLAP WITH SLC52A2: both riboflavin-responsive with MADD biochemistry; "
            "distinguish: FLAD1 = peripheral myopathy dominant; SLC52A2 = SNHL + pontobulbar palsy"
        ),
        "treatment": (
            "FLAD1 DEFICIENCY TREATMENT: "
            "RIBOFLAVIN 10-40 mg/kg/day orally: primary treatment; often dramatically effective; "
            "CARNITINE 100-200 mg/kg/day: prevents secondary carnitine deficiency from acylcarnitine export; "
            "COENZYME Q10 100-300 mg/day: adjunct; some benefit for mitochondrial and electron transport support; "
            "LOW-FAT HIGH-CARBOHYDRATE DIET: reduces fatty acid load during metabolic stress; "
            "FASTING AVOIDANCE: critical — avoid prolonged fasting; "
            "carbohydrate supplement before exercise; frequent meals; "
            "RHABDOMYOLYSIS PROTOCOL: IV glucose + hydration; "
            "avoid fasting; monitor CK + renal function; "
            "RIBOFLAVIN DOSE TITRATION: start 10 mg/kg/day; increase by 10 mg/kg increments until acylcarnitines normalise; "
            "MONITORING: acylcarnitine profile + urine organic acids 3-6 monthly; "
            "CK levels; plasma carnitine; renal function (rhabdomyolysis); "
            "muscle MRI at baseline and follow-up; "
            "OUTCOMES: riboflavin-responsive FLAD1: good long-term outcome; "
            "neonatal-onset severe form (cardiomyopathy + severe acidosis): poor prognosis despite treatment; "
            "GENETIC COUNSELLING: identify FLAD1 mutation class — hypomorphic vs null predicts response"
        ),
        "key_features": [
            "FLAD1 (FAD synthase deficiency): AR; riboflavin-responsive MADD-like; FAD synthesis defect",
            "LIPID STORAGE MYOPATHY on muscle biopsy: oil red O — excessive lipid droplets in type I fibres PATHOGNOMONIC for lipid storage",
            "MADD biochemical profile: C6-C18 acylcarnitines + ethylmalonic/glutaric/adipic aciduria",
            "Exercise intolerance + proximal myopathy + rhabdomyolysis risk (especially fasting/illness trigger)",
            "RIBOFLAVIN 10-40 mg/kg/day: often dramatically effective — normalises acylcarnitines and improves myopathy",
            "FASTING AVOIDANCE critical: fasting triggers metabolic crisis; frequent carbohydrate feeds essential",
            "Distinguished from SLC52A2 (BVVL): no SNHL or pontobulbar palsy; pure myopathy phenotype",
            "Severe neonatal form: cardiomyopathy + severe acidosis; poor prognosis even with riboflavin",
        ],
        "key_ddx": [
            "Classical MADD/GA2 (ETFA/ETFB): neonatal, severe, cardiomyopathy; NOT riboflavin-responsive (unlike ETFDH); acylcarnitine same pattern",
            "ETFDH deficiency (late-onset MADD): riboflavin-responsive; myopathy; same biochemistry; ETFDH assay",
            "SLC52A2 (BVVL2/RTD2): riboflavin-responsive MADD biochemistry; adds SNHL + pontobulbar palsy; SLC52A2 assay",
            "CPT2 deficiency: exercise-induced myolysis; long-chain acylcarnitines C16/C18; no organic aciduria pattern",
            "Pompe disease (GAA): lysosomal glycogen storage; myopathy + cardiomyopathy; acid maltase assay; no MADD pattern",
        ],
    },
    {
        "gene": "TCN2",
        "protein": (
            "TCN2 -- 22q12.2 AR -- 427aa -- Transcobalamin-II-46kDa-"
            "Plasma-B12-Transport-Glycoprotein-Megaloblastic-Anaemia-Neonatal-AR -- OMIM-Gene-613441-Disease-TCN2-Deficiency-275350"
        ),
        "locus": "22q12.2",
        "protein_size": "427 aa / 46 kDa",
        "inheritance": (
            "AR (biallelic TCN2 loss of function); Transcobalamin II deficiency; "
            "Prevalence: rare; worldwide; "
            "Onset: NEONATAL to early infancy (onset before 3 months typically); "
            "CARDINAL FEATURES: megaloblastic anaemia (often severe in first weeks of life) + "
            "failure to thrive + vomiting + diarrhoea; "
            "CNS: developmental delay, hypotonia, seizures if untreated; "
            "CRITICALLY IMPORTANT: serum B12 levels are NORMAL or HIGH despite severe deficiency — "
            "BECAUSE serum B12 is predominantly haptocorrin-bound (not transcobalamin-bound); "
            "TCN2 deficiency → B12 cannot be DELIVERED to cells even if serum B12 appears normal; "
            "TREATMENT: high-dose hydroxocobalamin IM bypasses transcobalamin — CURATIVE"
        ),
        "disease_category": (
            "Transcobalamin II (TCII) deficiency; inborn error of cobalamin (vitamin B12) cellular delivery; "
            "TCN2 encodes transcobalamin II, the primary plasma protein responsible for DELIVERING cobalamin to cells via TC-II receptor (CD320); "
            "Three cobalamin-binding proteins in plasma: "
            "Haptocorrin (TC-I): ~70-80% of serum B12; no role in cellular delivery; "
            "Transcobalamin I/TCII: 20-30% of serum B12 but BIOLOGICALLY ACTIVE B12 for cell delivery; "
            "Transcobalamin III (TC-III): from granulocytes; minor. "
            "Loss of TCN2 → no TC-II protein → cobalamin not delivered to cells via CD320 receptor → "
            "INTRACELLULAR COBALAMIN DEFICIENCY despite normal/high serum B12 (haptocorrin fraction intact); "
            "METABOLIC CONSEQUENCES: methylcobalamin (MeCbl) deficiency → methionine synthase (MTR) inactive → "
            "homocysteine accumulates; adenosylcobalamin (AdoCbl) deficiency → methylmalonyl-CoA mutase inactive → "
            "MMA accumulates; "
            "THYMIDYLATE SYNTHESIS impaired → megaloblastic changes"
        ),
        "disease_pathway": (
            "TCN2 encodes transcobalamin II (TCII), the predominant cobalamin-transport protein for INTRACELLULAR delivery. "
            "Dietary B12 → absorbed via intrinsic factor (GIF) + cubilin (CUBN) complex in terminal ileum → "
            "enters portal blood → released from IF → binds TCII in plasma → "
            "TCII-B12 complex binds CD320 (TC-II receptor) on cell surface → receptor-mediated endocytosis → "
            "lysosomal release → B12 converted to: "
            "METHYLCOBALAMIN (MeCbl): cytoplasm → methionine synthase (MTR) cofactor → homocysteine → methionine; "
            "ADENOSYLCOBALAMIN (AdoCbl): mitochondria → methylmalonyl-CoA mutase (MMUT) cofactor → propionyl-CoA catabolism. "
            "Loss of TCN2 → TCII protein absent → B12 absorbed normally but cannot be delivered to cells → "
            "PARADOX: serum B12 normal/elevated (haptocorrin still binds B12) but ALL cells B12-deficient; "
            "MEGALOBLASTIC ANAEMIA: thymidylate synthesis impaired → DNA replication error → macrocytes; "
            "METABOLIC: mild MMA + homocysteine elevation (not severe like cblC); "
            "HIGH-DOSE PARENTERAL B12: saturates passive diffusion pathway → bypasses TCII requirement."
        ),
        "pathognomonic": (
            "TCN2 DEFICIENCY DIAGNOSTIC CLUSTER: "
            "1) NEONATAL/EARLY INFANTILE MEGALOBLASTIC ANAEMIA: macrocytic anaemia in first weeks of life; "
            "hypersegmented neutrophils; thrombocytopenia; reticulocytopenia; pancytopaenia; "
            "2) NORMAL OR ELEVATED SERUM B12: PATHOGNOMONIC — B12 levels appear NORMAL despite severe B12 deficiency; "
            "this is the diagnostic trap: normal serum B12 does NOT exclude TCN2 deficiency; "
            "3) ELEVATED URINARY METHYLMALONIC ACID: elevated MMA on urine organic acids (mild-moderate); "
            "PLASMA TOTAL HOMOCYSTEINE: elevated (mild-moderate); "
            "4) SERUM TRANSCOBALAMIN II: absent or markedly reduced in TCN2 deficiency; "
            "HOLOTRANSCOBALAMIN (active B12 = B12 bound to TC-II): LOW or undetectable — most sensitive marker; "
            "5) FAILURE TO THRIVE + VOMITING + DIARRHOEA: poor feeding in neonate; "
            "6) CNS: hypotonia, developmental delay, seizures if untreated; "
            "DIAGNOSTIC CLUE: child with megaloblastic anaemia + NORMAL B12 — always consider TCN2; "
            "HOLOTRANSCOBALAMIN ASSAY: most sensitive screening test for functional B12 deficiency"
        ),
        "treatment": (
            "TCN2 DEFICIENCY TREATMENT: "
            "HYDROXOCOBALAMIN IM 1 mg every 2-3 days initially: bypasses TCII requirement — passive diffusion at high serum levels; "
            "MAINTENANCE: hydroxocobalamin IM 1 mg weekly to monthly after haematological stabilisation; "
            "SUBCUTANEOUS administration acceptable in some; "
            "HIGH-DOSE ORAL B12: not adequately absorbed without TCII in most forms — parenteral preferred; "
            "HAEMATOLOGICAL RESPONSE: reticulocyte count rises in 72 hours; "
            "anaemia corrects in 1-2 weeks; "
            "METABOLIC RESPONSE: MMA and homocysteine normalise; "
            "MONITORING: full blood count 3-6 monthly; "
            "holotranscobalamin levels (remains low in TCN2 deficiency — not a monitoring marker); "
            "serum MMA + homocysteine every 6 months; "
            "developmental assessment annually; "
            "DOSE MAINTENANCE: lifelong parenteral B12; NEVER stop; "
            "FOLINIC ACID 5 mg/day: adjunct (methionine cycle support); "
            "OUTCOMES: pre-symptomatic treatment = normal development; "
            "delayed treatment = neurodevelopmental sequelae; "
            "FAMILY SCREENING: siblings test immediately — start parenteral B12 if confirmed; "
            "ANTENATAL: maternal B12 supplementation during pregnancy if known TCN2 deficiency family"
        ),
        "key_features": [
            "TCN2 (Transcobalamin II deficiency): AR; B12 cellular delivery defect; neonatal megaloblastic anaemia",
            "NORMAL/HIGH SERUM B12 DESPITE SEVERE B12 DEFICIENCY: PATHOGNOMONIC DIAGNOSTIC TRAP — always measure holotranscobalamin",
            "HOLOTRANSCOBALAMIN LOW: most sensitive marker for functional B12 deficiency; standard B12 misses TCN2 deficiency",
            "Neonatal presentation: megaloblastic anaemia + pancytopaenia + FTT + vomiting — first weeks of life",
            "Mild MMA + homocysteine elevation: B12 metabolic consequences; NOT severe like cblC disease",
            "Hydroxocobalamin IM 1 mg every 2-3 days initially: bypasses TCN2 via passive diffusion at high serum levels",
            "LIFELONG parenteral B12: oral B12 not reliably absorbed without TCII carrier; parenteral route mandatory",
            "DDx trap: normal serum B12 leads to missed diagnosis; holotranscobalamin is the diagnostic key",
        ],
        "key_ddx": [
            "cblC disease (MMACHC): combined MMA + homocystinuria; presentation similar; BUT higher MMA + MMA:homocysteine ratio; MMACHC molecular",
            "Intrinsic factor deficiency (juvenile pernicious anaemia): B12 LOW (not normal); anti-IF antibodies; responds to IM B12 but different mechanism",
            "Imerslund-Gräsbeck syndrome (AMN/CUBN): selective B12 malabsorption; B12 low; proteinuria 50%; intestinal biopsy + Schilling test",
            "Dietary B12 deficiency (vegan/breastfed): B12 LOW; maternal dietary history; no molecular cause",
            "cblE/cblG disease (MTRR/MTR): homocystinuria without MMA; low holotranscobalamin; responds to B12; enzyme assay",
        ],
    },
    {
        "gene": "AMN",
        "protein": (
            "AMN -- 14q32.32 AR -- 453aa -- Amnionless-Protein-50kDa-"
            "Cubilin-Trafficking-Intestinal-Cobalamin-Absorption-Co-receptor-AR -- OMIM-Gene-605799-Disease-IGS2-261100"
        ),
        "locus": "14q32.32",
        "protein_size": "453 aa / 50 kDa",
        "inheritance": (
            "AR (biallelic AMN loss of function); Imerslund-Gräsbeck syndrome type 2 (IGS-2); "
            "Selective intestinal cobalamin malabsorption + mild proteinuria; "
            "CUBN mutations = IGS-1 (same phenotype, cubilin gene on 10p13); "
            "Prevalence: rare worldwide; Norwegian/Finnish founder (Fin-Major CUBN) for IGS-1; "
            "Onset: early childhood (onset 1-5 years typically); slower onset than TCN2; "
            "CARDINAL FEATURES: megaloblastic anaemia + failure to thrive + PROTEINURIA (50% of cases) — "
            "proteinuria is LOW-GRADE, tubular (urine protein electrophoresis shows β2-microglobulin/LMWP); "
            "TREATMENT: IM hydroxocobalamin CURATIVE; no specific renal treatment for proteinuria; "
            "SELECTIVE MALABSORPTION: oral B12 NOT effective; intrinsic-factor/B12 complex not absorbed"
        ),
        "disease_category": (
            "Imerslund-Gräsbeck syndrome (IGS); selective intestinal cobalamin malabsorption; "
            "inborn error of the cubilin-amnionless (IF-B12 receptor complex) endocytic pathway; "
            "AMN encodes amnionless (AMNL), a type I transmembrane protein that forms a stable heterodimeric complex "
            "with cubilin (CUBN) to form the cubam complex; "
            "CUBAM COMPLEX FUNCTION: "
            "1) INTESTINAL ABSORPTION: ileal cubam binds intrinsic factor (IF)-cobalamin complex → "
            "receptor-mediated endocytosis → B12 released into portal blood; "
            "2) RENAL REABSORPTION: proximal tubular cubam binds albumin, low-molecular-weight proteins (β2-microglobulin) → "
            "endocytosis and reabsorption; "
            "Loss of AMN → cubam complex not trafficked to cell surface (AMN acts as trafficking chaperone for CUBN) → "
            "IF-B12 complex not absorbed → B12 deficiency; "
            "RENAL PHENOTYPE: cubam also mediates tubular protein reabsorption → loss → mild low-grade proteinuria "
            "(β2-microglobulin, RBP — tubular markers, NOT glomerular protein)"
        ),
        "disease_pathway": (
            "AMN (amnionless) is required for cubilin (CUBN) cell-surface expression; without AMN, "
            "cubilin is retained in endoplasmic reticulum and not trafficked to the apical membrane of "
            "ileal enterocytes and renal proximal tubular cells. "
            "INTESTINAL PATHWAY: dietary B12 → binds haptocorrin in stomach → pancreatic proteases cleave haptocorrin → "
            "B12 binds intrinsic factor (GIF from parietal cells) → GIF-B12 complex traverses small bowel → "
            "ileal cubam binds GIF-B12 at pH 7 (calcium-dependent) → endocytosis → "
            "lysosomal release → B12 enters portal blood bound to TCII. "
            "Loss of AMN → CUBN not on ileal surface → GIF-B12 complex PASSES THROUGH UNABSORBED → "
            "B12 MALABSORPTION (selective — other nutrients absorbed normally): "
            "serum B12 LOW; MCV elevated; megaloblastic anaemia; homocysteine elevated; mild MMA elevated. "
            "RENAL PATHWAY: cubam on proximal tubular brush border binds and reabsorbs filtered albumin + LMWP → "
            "loss of AMN → LMWP not reabsorbed → TUBULAR PROTEINURIA (β2-microglobulin, RBP, alpha1-microglobulin); "
            "NOT nephrotic syndrome; urine dipstick may be trace; urine ACR normal; urine protein electrophoresis shows LMWP band."
        ),
        "pathognomonic": (
            "IGS-2 (AMN) DIAGNOSTIC CLUSTER: "
            "1) MEGALOBLASTIC ANAEMIA + LOW SERUM B12: early childhood onset (contrast TCN2: normal B12); "
            "hypersegmented neutrophils; macrocytic red cells; pancytopaenia; "
            "2) LOW-GRADE TUBULAR PROTEINURIA in 50%: β2-microglobulin + RBP (retinol-binding protein) + "
            "alpha1-microglobulin on URINE PROTEIN ELECTROPHORESIS — tubular pattern; "
            "NOT nephrotic-range; urine albumin:creatinine ratio may be normal; "
            "total urine protein only mildly elevated; dipstick may be negative; "
            "PATHOGNOMONIC: megaloblastic anaemia + selective B12 malabsorption + tubular proteinuria = IGS until proven otherwise; "
            "3) SCHILLING TEST (historical): abnormal Stage 1 (no IF); "
            "corrected with IF in Stage 2 — NO, NOT corrected (malabsorption is post-IF binding stage); "
            "DOES NOT correct with IF — distinguishes from pernicious anaemia (where IF corrects Schilling); "
            "4) ORAL B12 ABSORPTION STUDY: B12 not absorbed even with added IF — confirmatory; "
            "5) AMN/CUBN MOLECULAR: biallelic pathogenic variants in AMN (IGS-2) or CUBN (IGS-1); "
            "6) RENAL BIOPSY (if performed): proximal tubular changes; cubam absent from apical membrane; "
            "REMEMBER: IGS = LOW B12 (unlike TCN2 where B12 normal); proteinuria in 50% (unique to IGS)"
        ),
        "treatment": (
            "IGS-2 (AMN DEFICIENCY) TREATMENT: "
            "HYDROXOCOBALAMIN IM 1 mg every 2-3 days initially: bypasses intestinal absorption entirely; "
            "MAINTENANCE: hydroxocobalamin IM 1 mg weekly to monthly; "
            "ORAL B12: NOT effective — intestinal absorption mechanism is broken; parenteral route required; "
            "HAEMATOLOGICAL RESPONSE: rapid (reticulocytosis 72 hours; full correction 2-4 weeks); "
            "METABOLIC RESPONSE: MMA and homocysteine normalise with adequate B12 replacement; "
            "RENAL PROTEINURIA: no specific treatment; low-grade tubular proteinuria persistent; "
            "renal function usually PRESERVED long-term; "
            "annual monitoring of renal function (eGFR) and proteinuria (urine protein electrophoresis); "
            "ACEi/ARB not routinely indicated for tubular proteinuria; "
            "MONITORING: serum B12 + holotranscobalamin every 6 months; "
            "MCV; plasma MMA + homocysteine; "
            "urine protein electrophoresis annually; renal function annually; "
            "NEUROLOGICAL: developmental assessment; B12 supplementation reverses early neurological changes; "
            "LIFELONG parenteral B12; never stop; "
            "OUTCOMES: excellent with adequate B12 replacement; "
            "renal tubular proteinuria persistent but benign; "
            "FAMILY SCREENING: siblings — check B12 + holotranscobalamin immediately"
        ),
        "key_features": [
            "AMN (IGS-2 / Imerslund-Gräsbeck syndrome type 2): AR; amnionless = cubilin trafficking chaperone; selective intestinal B12 malabsorption",
            "LOW SERUM B12 + MEGALOBLASTIC ANAEMIA: contrast TCN2 (normal B12); early childhood onset (1-5 years)",
            "TUBULAR PROTEINURIA in 50%: β2-microglobulin + RBP + alpha1-microglobulin — PATHOGNOMONIC combination with megaloblastic anaemia",
            "ORAL B12 NOT ABSORBED even with IF: ileal cubam complex absent — does NOT correct with intrinsic factor (unlike pernicious anaemia)",
            "SCHILLING TEST: NOT corrected by addition of IF — distinguishes IGS from pernicious anaemia (IF-correctable)",
            "Hydroxocobalamin IM 1 mg weekly-monthly: CURATIVE; bypasses intestinal absorption; lifelong",
            "Renal tubular proteinuria persistent but renal function usually preserved long-term",
            "CUBN mutations (10p13) = IGS-1 same phenotype; Norwegian/Finnish Fin-Major founder variant",
        ],
        "key_ddx": [
            "TCN2 deficiency: neonatal onset; NORMAL serum B12 (not low); no tubular proteinuria; holotranscobalamin low",
            "Juvenile pernicious anaemia (anti-GIF antibodies): low B12; Schilling CORRECTS with IF addition — unlike IGS; anti-IF antibodies positive",
            "Dietary B12 deficiency: low B12; maternal vegan history if infant; responds to oral B12",
            "cblC disease (MMACHC): early onset; combined MMA+homocystinuria; B12 variable; responds to OHCbl; MMACHC molecular",
            "Nephronophthisis (NPHP genes): renal tubular disease; no megaloblastic anaemia; no B12 deficiency",
        ],
    },
]

def _seed_patients(gene_idx: int, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]
    gene_name = gene["gene"]

    patients = []
    for i in range(n):
        age_onset = _gene_onset(gene_name, rng)
        features = _gene_features(gene_name, rng)
        patients.append({
            "patient_id": f"{gene_name}-{seed}-{i+1:02d}",
            "gene": gene_name,
            "age_onset_yr": round(age_onset, 1),
            "sex": rng.choice(["M", "F"]),
            "features": features,
            "treated": rng.random() < 0.88,
            "outcome": rng.choice(["stable", "improved", "mild_sequelae", "severe_sequelae"]) if rng.random() < 0.95 else "lost_to_followup",
        })
    return patients


def _gene_onset(gene: str, rng: random.Random) -> float:
    ranges = {
        "BTD":    (0.1, 1.5),
        "HLCS":   (0.01, 0.3),
        "SLC19A3":(0.5, 8.0),
        "MTHFR":  (0.02, 0.5),
        "SLC52A2":(1.0, 20.0),
        "FLAD1":  (2.0, 25.0),
        "TCN2":   (0.02, 0.25),
        "AMN":    (0.5, 5.0),
    }
    lo, hi = ranges.get(gene, (1.0, 10.0))
    return round(rng.uniform(lo, hi), 2)


def _gene_features(gene: str, rng: random.Random) -> dict:
    if gene == "BTD":
        return {
            "alopecia": rng.random() < 0.72,
            "perioral_dermatitis": rng.random() < 0.68,
            "sensorineural_hearing_loss": rng.random() < 0.55,
            "seizures": rng.random() < 0.60,
            "lactic_acidosis": rng.random() < 0.78,
            "organic_aciduria": rng.random() < 0.82,
            "vision_loss": rng.random() < 0.28,
            "biotin_responsive": rng.random() < 0.94,
        }
    if gene == "HLCS":
        return {
            "neonatal_metabolic_crisis": rng.random() < 0.90,
            "hyperammonaemia": rng.random() < 0.75,
            "lactic_acidosis": rng.random() < 0.92,
            "triple_organic_acid_pattern": rng.random() < 0.88,
            "alopecia": rng.random() < 0.50,
            "seizures": rng.random() < 0.65,
            "biotin_responsive": rng.random() < 0.80,
            "km_mutant": rng.random() < 0.68,
        }
    if gene == "SLC19A3":
        return {
            "basal_ganglia_mri_lesions": rng.random() < 0.95,
            "stress_triggered_encephalopathy": rng.random() < 0.88,
            "dystonia": rng.random() < 0.80,
            "seizures": rng.random() < 0.72,
            "saudi_founder_variant": rng.random() < 0.60,
            "biotin_thiamine_responsive": rng.random() < 0.85,
            "cortical_ribbon_t2": rng.random() < 0.68,
        }
    if gene == "MTHFR":
        return {
            "megaloblastic_anaemia": rng.random() < 0.55,
            "hyperhomocysteinaemia": rng.random() < 0.98,
            "low_methionine": rng.random() < 0.92,
            "white_matter_disease": rng.random() < 0.70,
            "seizures": rng.random() < 0.60,
            "thrombosis": rng.random() < 0.35,
            "betaine_responsive": rng.random() < 0.78,
        }
    if gene == "SLC52A2":
        return {
            "sensorineural_hearing_loss": rng.random() < 0.90,
            "pontobulbar_palsy": rng.random() < 0.82,
            "axonal_neuropathy": rng.random() < 0.75,
            "respiratory_failure": rng.random() < 0.60,
            "madd_acylcarnitines": rng.random() < 0.85,
            "riboflavin_responsive": rng.random() < 0.90,
            "facial_palsy": rng.random() < 0.70,
        }
    if gene == "FLAD1":
        return {
            "lipid_storage_myopathy": rng.random() < 0.88,
            "exercise_intolerance": rng.random() < 0.92,
            "madd_acylcarnitines": rng.random() < 0.85,
            "rhabdomyolysis": rng.random() < 0.55,
            "cardiomyopathy": rng.random() < 0.25,
            "riboflavin_responsive": rng.random() < 0.80,
            "neonatal_severe": rng.random() < 0.18,
        }
    if gene == "TCN2":
        return {
            "megaloblastic_anaemia": rng.random() < 0.98,
            "normal_serum_b12": rng.random() < 0.90,
            "low_holotranscobalamin": rng.random() < 0.96,
            "failure_to_thrive": rng.random() < 0.85,
            "mild_mma_elevation": rng.random() < 0.78,
            "mild_homocysteine_elevation": rng.random() < 0.82,
            "neurodevelopmental_delay": rng.random() < 0.45,
            "hydroxocobalamin_responsive": rng.random() < 0.95,
        }
    if gene == "AMN":
        return {
            "megaloblastic_anaemia": rng.random() < 0.96,
            "low_serum_b12": rng.random() < 0.94,
            "tubular_proteinuria": rng.random() < 0.52,
            "failure_to_thrive": rng.random() < 0.78,
            "mild_mma_elevation": rng.random() < 0.65,
            "no_oral_b12_absorption": rng.random() < 0.98,
            "hydroxocobalamin_responsive": rng.random() < 0.96,
        }
    return {}


SEEDS = list(range(2638, 2646))   # 8 seeds for 8 genes


def _build_cohort():
    all_patients = []
    for idx, seed in enumerate(SEEDS):
        all_patients.extend(_seed_patients(idx, seed))
    return all_patients


def generate_overview():
    cohort = _build_cohort()
    gene_summaries = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        pts = [p for p in cohort if p["gene"] == gene]
        avg_onset = round(sum(p["age_onset_yr"] for p in pts) / len(pts), 2) if pts else 0
        hearing_loss_pct = round(100 * sum(1 for p in pts if p["features"].get("sensorineural_hearing_loss", False)) / len(pts)) if pts else 0
        metabolic_crisis_pct = round(100 * sum(1 for p in pts if any(p["features"].get(k, False) for k in ["neonatal_metabolic_crisis", "stress_triggered_encephalopathy", "lactic_acidosis"])) / len(pts)) if pts else 0
        responsive_pct = round(100 * sum(1 for p in pts if any(p["features"].get(k, False) for k in ["biotin_responsive", "riboflavin_responsive", "biotin_thiamine_responsive", "betaine_responsive", "hydroxocobalamin_responsive"])) / len(pts)) if pts else 0
        gene_summaries.append({
            "gene": gene,
            "locus": gene_info["locus"],
            "n_patients": len(pts),
            "avg_onset_age": avg_onset,
            "hearing_loss_pct": hearing_loss_pct,
            "metabolic_crisis_pct": metabolic_crisis_pct,
            "vitamin_treatment_responsive_pct": responsive_pct,
        })

    return {
        "atlas": "Hereditary Vitamin Metabolism Atlas",
        "total_patients": len(cohort),
        "n_genes": len(ATLAS_GENES),
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "disease_classes": [
            "BTD — Biotinidase Deficiency: AR; biotin recycling defect; alopecia + dermatitis + SNHL + seizures; biotin 5-10mg/day curative",
            "HLCS — Holocarboxylase Synthetase Deficiency: AR; neonatal MCD; biotin ligation defect; hyperammonaemia; biotin 10-40mg/day",
            "SLC19A3 — BTBGD: AR; thiamine transporter 2 deficiency; stress-triggered encephalopathy; bilateral BG MRI lesions; biotin+thiamine",
            "MTHFR — Severe MTHFR Deficiency: AR; homocystinuria type 4; low methionine + elevated Hcy; betaine+5-MTHF+B12",
            "SLC52A2 — RTD2/BVVL2: AR; riboflavin transporter 2 deficiency; SNHL + pontobulbar palsy; riboflavin 10-40mg/kg/day",
            "FLAD1 — FAD Synthase Deficiency: AR; riboflavin-responsive MADD-like; lipid storage myopathy; riboflavin 10-40mg/kg/day",
            "TCN2 — Transcobalamin II Deficiency: AR; B12 cellular delivery defect; megaloblastic anaemia; NORMAL serum B12 TRAP; parenteral B12",
            "AMN — IGS-2/Imerslund-Gräsbeck: AR; selective B12 malabsorption; tubular proteinuria 50%; IM B12 curative; oral B12 ineffective",
        ],
        "aggregate_stats": {
            "overall_hearing_loss_pct": round(sum(gs["hearing_loss_pct"] for gs in gene_summaries) / len(gene_summaries)),
            "overall_metabolic_crisis_pct": round(sum(gs["metabolic_crisis_pct"] for gs in gene_summaries) / len(gene_summaries)),
            "overall_vitamin_responsive_pct": round(sum(gs["vitamin_treatment_responsive_pct"] for gs in gene_summaries) / len(gene_summaries)),
        },
        "gene_summaries": gene_summaries,
        "key_clinical_distinctions": [
            "BTD vs HLCS: BTD older onset (1-6mo) + prominent alopecia/dermatitis; HLCS earlier (neonatal) + hyperammonaemia; same metabolites; enzyme assay distinguishes",
            "TCN2 vs AMN: TCN2 = NORMAL serum B12 (cell delivery defect); AMN = LOW serum B12 (intestinal absorption defect); both require parenteral B12",
            "SLC52A2 vs FLAD1: both riboflavin-responsive MADD biochemistry; SLC52A2 = SNHL + pontobulbar palsy; FLAD1 = pure peripheral myopathy; no cranial neuropathy",
            "MTHFR severe vs CBS: MTHFR = LOW methionine + elevated Hcy; CBS = HIGH methionine + elevated Hcy; critical DDx biochemical distinction",
            "BTD/HLCS vs SLC19A3: BTD/HLCS = organic aciduria + alopecia; SLC19A3 = bilateral BG MRI lesions + stress-triggered crises; BOTH thiamine",
            "AMN vs pernicious anaemia (anti-GIF): AMN = Schilling NOT corrected by IF; pernicious anaemia = Schilling CORRECTED by IF; tubular proteinuria in AMN",
            "TCN2 DIAGNOSTIC TRAP: serum B12 normal/high → diagnosis missed; always measure holotranscobalamin in neonatal megaloblastic anaemia",
            "SLC52A2 REVERSIBLE SNHL: riboflavin can REVERSE established SNHL — unique treatment response not seen in other SNHL causes",
        ],
    }


def generate_breakdown():
    cohort = _build_cohort()
    gene_breakdowns = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        pts = [p for p in cohort if p["gene"] == gene]
        gene_breakdowns.append({
            "gene": gene,
            "locus": gene_info["locus"],
            "protein_size": gene_info["protein_size"],
            "inheritance": gene_info["inheritance"],
            "disease_category": gene_info["disease_category"],
            "pathognomonic": gene_info["pathognomonic"],
            "treatment": gene_info["treatment"],
            "key_features": gene_info["key_features"],
            "key_ddx": gene_info["key_ddx"],
            "n_patients": len(pts),
            "avg_onset_age": round(sum(p["age_onset_yr"] for p in pts) / len(pts), 2) if pts else 0,
            "feature_rates": {
                k: round(100 * sum(1 for p in pts if p["features"].get(k, False)) / len(pts))
                for k in (pts[0]["features"].keys() if pts else [])
            },
        })
    return {"gene_breakdowns": gene_breakdowns, "total_cohort": len(cohort)}


def generate_definitions():
    gene_entries = {}
    for g in ATLAS_GENES:
        gene_entries[g["gene"]] = {
            "disease_name": g["inheritance"].split(";")[1].strip() if ";" in g["inheritance"] else g["gene"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
        }

    return {
        "gene_entries": gene_entries,
        "vitamin_metabolism_glossary": {
            "BIOTIN (Vitamin B7)": (
                "Water-soluble vitamin; cofactor for four essential carboxylases: pyruvate carboxylase (PC), "
                "propionyl-CoA carboxylase (PCC), 3-methylcrotonyl-CoA carboxylase (MCC), acetyl-CoA carboxylase (ACC). "
                "DEFICIENCY SOURCES: dietary, biotinidase deficiency (BTD), holocarboxylase synthetase deficiency (HLCS). "
                "TREATMENT: free oral biotin 5-40 mg/day depending on defect type."
            ),
            "THIAMINE (Vitamin B1)": (
                "Water-soluble vitamin; cofactor as thiamine pyrophosphate (TPP) for: "
                "pyruvate dehydrogenase (PDH), alpha-ketoglutarate dehydrogenase (KGDH), transketolase, BCKDH. "
                "CNS particularly dependent on thiamine: basal ganglia vulnerable (BTBGD, Wernicke). "
                "SLC19A3 (ThTr2) and SLC19A2 (ThTr1) mediate cellular uptake; SLC19A3 deficiency = BTBGD."
            ),
            "RIBOFLAVIN (Vitamin B2)": (
                "Water-soluble vitamin; precursor to FAD and FMN cofactors. "
                "FAD essential for: Complex I, Complex II, all acyl-CoA dehydrogenases (VLCAD-SCAD), "
                "ETF, DHODH, FLAD1 (FAD synthase). "
                "Transport defects: SLC52A1/A2/A3 (riboflavin transporters); deficiency = RTD/BVVL. "
                "FAD synthesis defect: FLAD1; riboflavin-responsive MADD-like. "
                "High-dose riboflavin (10-40 mg/kg/day) can reverse SNHL in SLC52A2 deficiency."
            ),
            "COBALAMIN (Vitamin B12)": (
                "Water-soluble vitamin; cofactor for: methionine synthase (MTR — as methylcobalamin) and "
                "methylmalonyl-CoA mutase (MMUT — as adenosylcobalamin). "
                "Absorption pathway: dietary B12 → IF (GIF) → ileal cubam (CUBN+AMN) → portal blood → "
                "TCII carries to cells → CD320 receptor → cellular uptake. "
                "Defects: AMN/CUBN (intestinal absorption, IGS); TCN2 (cellular delivery); "
                "MMACHC/cblC (intracellular processing). "
                "DIAGNOSTIC TRAP: TCN2 deficiency = NORMAL serum B12; holotranscobalamin is functional B12 marker."
            ),
            "FOLATE (Vitamin B9)": (
                "Water-soluble vitamin; 5-methyltetrahydrofolate (5-MTHF) = methyl donor for homocysteine remethylation. "
                "MTHFR converts 5,10-MTHF → 5-MTHF. "
                "MTHFR deficiency (severe biallelic) = hyperhomocysteinaemia + LOW methionine (KEY DDx from CBS). "
                "COMMON VARIANT CAVEAT: MTHFR C677T/A1298C are polymorphisms, NOT rare disease — "
                "do not cause MTHFR deficiency disease; treated only with folic acid 400-800 μg/day."
            ),
            "MULTIPLE CARBOXYLASE DEFICIENCY (MCD)": (
                "Syndrome of simultaneous deficiency of all four biotin-dependent carboxylases: PC, PCC, MCC, ACC. "
                "Causes: BTD deficiency (biotin recycling defect) or HLCS deficiency (biotin ligation defect). "
                "Metabolic signature: 3-methylcrotonylglycinuria + propionic acidaemia + lactic acidosis. "
                "Acylcarnitine: C5-OH (3-methylcrotonylcarnitine) + C3 (propionylcarnitine) elevated on NBS. "
                "Treatment: biotin; HLCS needs higher dose (10-40 mg) vs BTD (5-10 mg)."
            ),
            "MADD / GLUTARIC ACIDURIA TYPE II (GA2)": (
                "Multiple acyl-CoA dehydrogenase deficiency — all FAD-dependent acyl-CoA dehydrogenases fail simultaneously. "
                "Classical MADD: ETFA/ETFB mutations; neonatal severe; non-riboflavin-responsive. "
                "RIBOFLAVIN-RESPONSIVE MADD: SLC52A2 (RTD), FLAD1 (FAD synthase), ETFDH mutations — "
                "high-dose riboflavin often dramatically effective. "
                "Acylcarnitine signature: C6-C18 dicarboxylyl + C3; organic acids: ethylmalonic + glutaric + adipic."
            ),
            "TRANSCOBALAMIN vs HAPTOCORRIN (B12 FRACTIONS)": (
                "Serum B12 is predominantly (70-80%) bound to HAPTOCORRIN (TC-I) — biologically INACTIVE for cell delivery. "
                "Only 20-30% is HOLOTRANSCOBALAMIN (B12 bound to TCII) — the ACTIVE fraction for cellular delivery. "
                "DIAGNOSTIC IMPLICATION: TCN2 deficiency (TCII absent) = serum B12 NORMAL (haptocorrin fraction intact) "
                "but holotranscobalamin ABSENT — all cells B12-deficient. "
                "Always measure holotranscobalamin in unexplained megaloblastic anaemia or homocysteinaemia."
            ),
            "IMERSLUND-GRÄSBECK SYNDROME (IGS)": (
                "Selective intestinal cobalamin (B12) malabsorption due to defective cubam complex (CUBN+AMN). "
                "Two genes: CUBN (10p13) = IGS-1; AMN (14q32.32) = IGS-2. "
                "Distinguishing features: (1) low serum B12 (contrast TCN2 where B12 is normal); "
                "(2) tubular proteinuria 50% (β2-microglobulin/RBP — LOW-GRADE, not nephrotic); "
                "(3) oral B12 NOT absorbed (even with added IF); "
                "(4) Schilling test Stage 2 NOT corrected (contrast pernicious anaemia). "
                "Treatment: lifelong IM hydroxocobalamin."
            ),
        },
    }
