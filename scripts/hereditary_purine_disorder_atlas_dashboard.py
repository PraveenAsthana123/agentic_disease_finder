"""Hereditary Purine Disorder Atlas — 8-Gene Reference
HPRT1-ADSL-ATIC-ADA-PNP-APRT-DGUOK-PRPS1
320 patients (8 x 40), seeds 2702-2709.
Endpoints: /api/hereditary-purine-disorder-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "HPRT1",
        "seed_base": 2702,
        "protein": (
            "HPRT1 -- Xq26.2 XLR -- 218aa -- Hypoxanthine-Guanine-Phosphoribosyltransferase-"
            "25kDa-Homotetramer-Purine-Salvage-Hypoxanthine+Guanine→IMP+GMP-"
            "OMIM-Gene-308000-Disease-Lesch-Nyhan-300322"
        ),
        "locus": "Xq26.2",
        "protein_size": "218 aa / 25 kDa (homotetramer; cytoplasmic purine salvage)",
        "inheritance": (
            "X-LINKED RECESSIVE — hemizygous LOF in males; carrier females asymptomatic "
            "(functional mosaicism from X-inactivation); "
            "HPRT1 encodes hypoxanthine-guanine phosphoribosyltransferase (HGPRT): "
            "salvages hypoxanthine → IMP and guanine → GMP using PRPP as pyrophosphate donor; "
            "HPRT1 LOF → hypoxanthine + guanine NOT salvaged → PRPP accumulates → "
            "de novo purine synthesis accelerated → uric acid overproduction (gout-like); "
            "dopaminergic neuron dysfunction (HGPRT highest in basal ganglia); "
            "DISEASE SPECTRUM: "
            "  Complete LOF → Lesch-Nyhan Disease (LND): self-injurious behaviour + uric acid overproduction; "
            "  Partial LOF → Lesch-Nyhan Variants (LNV): hyperuricaemia without self-mutilation; "
            "PREVALENCE: ~1 in 380,000 live male births; no ethnic predilection"
        ),
        "disease_category": (
            "LESCH-NYHAN DISEASE (OMIM 300322); "
            "CLINICAL TRIAD: "
            "  1. URIC ACID OVERPRODUCTION: urolithiasis (urate crystals, orange crystals on nappy); "
            "     tophi; gout; haematuria; renal colic; renal failure if untreated; "
            "  2. NEUROLOGICAL: dystonia (most prominent); choreoathetosis; spasticity; "
            "     hypotonia → hypertonia progression; all patients wheelchair-bound; "
            "     INTELLECTUAL DISABILITY: severe in classic LND (IQ typically 40-70); "
            "     dysarthria; dysphagia; "
            "  3. SELF-INJURIOUS BEHAVIOUR (SIB): compulsive — lip/finger biting, head banging; "
            "     PATHOGNOMONIC for classic LND; begins 2-3 years; "
            "     contra-willful: patients experience SIB as dystonic, request restraints; "
            "     MECHANISM: dopaminergic denervation of basal ganglia → impaired impulse control; "
            "ALLOPURINOL: controls hyperuricaemia but DOES NOT TREAT NEUROLOGICAL/BEHAVIOURAL symptoms; "
            "DENTAL EXTRACTION: prevents lip biting when restraints insufficient; "
            "BACLOFEN/BENZODIAZEPINE: reduce dystonia; intrathecal baclofen used; "
            "GABAPENTIN/CARBAMAZEPINE: SIB and dystonia; "
            "GENE THERAPY: lentiviral HPRT1 correction in HSCs — Phase I/II (Comet Therapeutics 2024)"
        ),
        "disease_pathway": (
            "PURINE SALVAGE PATHWAY — HGPRT: "
            "Hypoxanthine + PRPP → IMP + PPi (HGPRT reaction); "
            "Guanine + PRPP → GMP + PPi (HGPRT reaction); "
            "HPRT1 LOF PATHOMECHANISM: "
            "  Hypoxanthine/guanine NOT salvaged → excess substrate for XDH; "
            "  Xanthine dehydrogenase (XDH): hypoxanthine → xanthine → uric acid (urate); "
            "  PRPP accumulates (not consumed by HGPRT) → "
            "    de novo purine synthesis maximally stimulated → more AMP/GMP → more degradation → "
            "    net uric acid overproduction (10-20x normal excretion); "
            "  DOPAMINERGIC NEURONS: HGPRT normally highest in basal ganglia; "
            "    HPRT1 LOF → dopamine synthesis impaired (IMP deficiency affects BH4 recycling) → "
            "    DAT expression abnormal → D1/D2 receptor supersensitivity → impaired basal ganglia circuits; "
            "TREATMENT RATIONALE: "
            "  ALLOPURINOL: xanthine oxidase inhibitor → reduces uric acid; "
            "  FEBUXOSTAT: alternative XO inhibitor; "
            "  RASBURICASE: urate oxidase (acute crisis only); "
            "  INTRATHECAL BACLOFEN: for severe dystonia; "
            "  HEMATOPOIETIC GENE THERAPY: HPRT1 correction in HSCs → restore HGPRT in marrow; "
            "  NOTE: allopurinol does NOT penetrate CNS effectively for neurological benefit"
        ),
        "pathognomonic": (
            "SELF-INJURIOUS BEHAVIOUR (SIB) IN YOUNG MALE WITH HYPERURICAEMIA: "
            "  combination PATHOGNOMONIC for Lesch-Nyhan Disease; "
            "  no other condition causes compulsive self-biting + uric acid overproduction; "
            "ORANGE CRYSTALS ON NAPPY/DIAPER: urate deposits in neonatal urine — "
            "  first clinical clue in many cases before SIB appears; "
            "ERYTHROCYTE HGPRT ACTIVITY <0.5 nmol/h/mg protein: "
            "  normal 90-210 nmol/h/mg; classic LND <1% normal; "
            "  VARIANTS: 1-10% residual activity → Kelley-Seegmiller variant (hyperuricaemia, mild neuro, no SIB); "
            "URIC ACID EXCRETION: urine UA/creatinine ratio >2.0 (normal <0.6) in children; "
            "MRI BRAIN: caudate + putamen atrophy; reduced DAT signal on DATscan; "
            "GENOTYPE-PHENOTYPE: missense mutations with >1% residual activity → variants; "
            "  complete deletion/frameshift/nonsense → classic LND; "
            "  splice mutations variable depending on residual transcript"
        ),
        "treatment": (
            "1. ALLOPURINOL: XO inhibitor; 10-15 mg/kg/day; controls gout/urolithiasis; "
            "   urine alkalinisation (sodium citrate) co-administered; "
            "   DOES NOT IMPROVE NEUROLOGICAL OR BEHAVIOURAL SYMPTOMS; "
            "2. PHYSICAL RESTRAINTS: paradoxically requested by patients; prevent SIB; "
            "   padded arm splints; dental guards; IMPORTANT: removal → patient distress (compulsive drive); "
            "3. DENTAL EXTRACTION/FILING: removes ability to self-bite; used when restraints insufficient; "
            "4. BACLOFEN (oral or intrathecal): reduces spasticity and dystonia; "
            "   intrathecal preferred for severe cases (oral: 30-120 mg/day); "
            "5. BENZODIAZEPINES: diazepam for acute dystonic crises; "
            "6. GABAPENTIN: SIB modulation (case series evidence); "
            "7. CARBAMAZEPINE: SIB and dystonia (moderate evidence); "
            "8. CARBIDOPA/LEVODOPA: limited benefit for dopamine pathway; "
            "9. S-ADENOSYLMETHIONINE (SAM): investigational; purine metabolism support; "
            "10. GENE THERAPY: lentiviral HPRT1 in HSCs — Phase I/II trials 2024 (Comet Therapeutics); "
            "11. AVOID: aspirin (displaces urate, worsens gout); high-purine foods; "
            "AVOID IN ALLOPURINOL USE: 6-mercaptopurine/azathioprine → fatal myelosuppression (XO metabolises both)"
        ),
    },
    {
        "gene": "ADSL",
        "seed_base": 2703,
        "protein": (
            "ADSL -- 22q13.1 AR -- 484aa -- Adenylosuccinate-Lyase-54kDa-Homotetramer-"
            "De-Novo-Purine+AMP-Synthesis-Two-Steps-SAICAR→AICAR+AMP→AMP-"
            "OMIM-Gene-608222-Disease-ADSL-Deficiency-103050"
        ),
        "locus": "22q13.1",
        "protein_size": "484 aa / 54 kDa (homotetramer; cytoplasmic; two-reaction enzyme)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in ADSL; "
            "ADSL catalyses TWO reactions: "
            "  1. De novo purine synthesis: SAICAR → AICAR + fumarate (step 8 of 10); "
            "  2. AMP biosynthesis: adenylosuccinate → AMP + fumarate; "
            "ADSL LOF → succinylaminoimidazole carboxamide ribotide (SAICAR) and "
            "adenylosuccinate accumulate → toxic to neurons; "
            "MECHANISM: succinylated purines (SAICAr + S-Ado) accumulate; "
            "SAICAr TOXIC: inhibits adenylosuccinate synthetase + phosphoribosylaminoimidazole synthetase; "
            "CSF/PLASMA S-Ado:SAICAr RATIO <1 = PATHOGNOMONIC for ADSL deficiency; "
            "PREVALENCE: ~100 cases worldwide; Belgian population >30% of cases (founder effect)"
        ),
        "disease_category": (
            "ADSL DEFICIENCY (OMIM 103050); "
            "THREE CLINICAL PHENOTYPES: "
            "  1. NEONATAL LETHAL (Type I severe): respiratory failure at birth; "
            "     severe hypotonia; no developmental milestones; early death; "
            "  2. PSYCHOMOTOR RETARDATION FORM (Type II): most common; "
            "     profound intellectual disability; autistic features (60-80%); "
            "     epilepsy (multifocal/generalized seizures, infantile spasms); "
            "     hypotonia → spasticity; no language; "
            "  3. MILD/MODERATE FORM (Type III): moderate ID; autism; seizures variable; "
            "     some independent ambulation; "
            "GENETICS: severe alleles (early frameshift/nonsense) → Type I/II; "
            "  mild missense (Arg426His — Belgian founder) → Type III; "
            "HALLMARKS: autism-like behaviour in ~60%; growth retardation; "
            "  wasting syndrome in severe forms; brain MRI: cerebral atrophy, white matter signal; "
            "BIOCHEMISTRY: urine succinylaminoimidazole carboxamide ribose (SAICAr) + "
            "  adenylosuccinate (S-Ado) ELEVATED — detected by purine chromatography; "
            "  CSF SAICAr/S-Ado ratio: S-Ado predominance in severe; SAICAr in milder"
        ),
        "disease_pathway": (
            "DE NOVO PURINE SYNTHESIS — ADSL (STEPS 8 + AMP SYNTHESIS): "
            "Step 8 (de novo): SAICAR → AICAR + fumarate; "
            "AMP synthesis: adenylosuccinate → AMP + fumarate; "
            "ADSL LOF PATHOMECHANISM: "
            "  SAICAR accumulates (step 8 block) → excreted as SAICAr (dephosphorylated); "
            "  Adenylosuccinate accumulates (AMP synthesis block) → excreted as S-Ado; "
            "  SAICAr toxic: inhibits adenylosuccinate synthetase → amplifies AMP deficit; "
            "  AMP depletion → ATP/AMP ratio falls → energy deficit in neurons; "
            "  Fumarate deficiency: both reactions produce fumarate → TCA cycle anaplerosis impaired; "
            "  SUCCINYLATED PURINES: cross BBB; direct CNS toxicity mechanism confirmed in yeast models; "
            "RIBOSE-5-PHOSPHATE / AICAR: AICAR normally activates AMPK; "
            "  ADSL → AICAR production impaired → reduced AMPK activation → "
            "  possible impairment of autophagy, glucose metabolism in neurons; "
            "TREATMENT RATIONALE: "
            "  No curative therapy; supportive AED + developmental support; "
            "  ALLOPURINOL: reduces de novo purine flux (theoretical — reduces SAICAR production); "
            "  RIBOSE SUPPLEMENTATION: investigational; "
            "  SUBSTRATE REDUCTION: experimental ADSL bypass strategies"
        ),
        "pathognomonic": (
            "URINE/CSF SAICAr + S-Ado ELEVATED (succinylpurine screen): "
            "  PATHOGNOMONIC for ADSL deficiency; "
            "  detected by HPLC or cation-exchange chromatography (succinylpurine screening); "
            "CSF S-Ado:SAICAr RATIO: "
            "  S-Ado >> SAICAr → severe phenotype; "
            "  SAICAr >> S-Ado → milder phenotype; "
            "  ratio <1 (S-Ado ≤ SAICAr) PATHOGNOMONIC; "
            "AUTISM + EPILEPSY + GROWTH RETARDATION TRIAD: "
            "  in male infant with elevated succinylpurines → ADSL until proven otherwise; "
            "BRAIN MRI: generalised atrophy; cerebellar hypoplasia; periventricular white matter changes; "
            "PLASMA/URINE PURINES: normal uric acid (contrast HPRT1 Lesch-Nyhan); "
            "  succinylpurines NOT on standard metabolic screens — MUST request specifically; "
            "ADSL ENZYME ASSAY: erythrocytes (both reactions measurable); "
            "  activity <5% of normal in Type I/II; 5-30% in Type III"
        ),
        "treatment": (
            "1. ANTIEPILEPTIC DRUGS: "
            "   Infantile spasms → ACTH or vigabatrin (standard IS protocol); "
            "   Focal/multifocal seizures → levetiracetam (first-line); valproate (caution — hepatotoxic risk); "
            "   Refractory seizures → ketogenic diet (evidence in ADSL: metabolic shift reduces purine burden); "
            "2. ALLOPURINOL: 10 mg/kg/day; reduces de novo purine synthesis flux → less SAICAR production; "
            "   biochemical improvement in some (SAICAr reduces); clinical benefit limited/inconsistent; "
            "3. RIBOSE SUPPLEMENTATION: investigational; 0.5-1 g/kg/day; theoretical: bypasses AMP deficit; "
            "4. DEVELOPMENTAL SUPPORT: physiotherapy; speech/language therapy; autism intervention; "
            "5. KETOGENIC DIET: reduces pyruvate → less oxaloacetate → less aspartate → less adenylosuccinate; "
            "   anecdotal seizure improvement; metabolic rationale; "
            "6. GENE THERAPY: no clinical trials (2026); target gene small; HSC/CNS approaches theoretical; "
            "7. AVOID: high-protein + high-purine diet (increases substrate load); "
            "   succinylcholine in anaesthesia (myopathy risk); "
            "8. NEWBORN SCREENING: not standard; succinylpurine screen only if clinically suspected"
        ),
    },
    {
        "gene": "ATIC",
        "seed_base": 2704,
        "protein": (
            "ATIC -- 2q35 AR -- 592aa -- AICAR-Transformylase/IMP-Cyclohydrolase-"
            "65kDa-Homodimer-De-Novo-Purine-Synthesis-Steps-9-10-"
            "OMIM-Gene-601731-Disease-AICA-Ribosiduria-608688"
        ),
        "locus": "2q35",
        "protein_size": "592 aa / 65 kDa (homodimer; cytoplasmic; bifunctional steps 9-10)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in ATIC; "
            "ATIC bifunctional enzyme catalyses last two steps of de novo purine synthesis: "
            "  Step 9: AICAR + 10-formyl-THF → FAICAR + THF (AICAR transformylase domain); "
            "  Step 10: FAICAR → IMP + H2O (IMP cyclohydrolase domain); "
            "ATIC LOF → AICAR (AICA-ribotide) accumulates → AICAR ribosiduria; "
            "AICAR exported from cell → dephosphorylated to AICA-riboside (acadesine) in plasma; "
            "AICAR pathological at high concentrations: "
            "  inhibits AMP deaminase → AMP/adenosine accumulate; "
            "  inhibits adenylsuccinate lyase (ADSL) — autoinhibitory feedback; "
            "  zebrafish atic morphants: severe developmental defects; "
            "PREVALENCE: EXTREMELY RARE — ~5 cases reported worldwide (2026); ultraorphan"
        ),
        "disease_category": (
            "AICA-RIBOSIDURIA (OMIM 608688); "
            "SINGLE REPORTED PHENOTYPE (severe): "
            "  Profound INTELLECTUAL DISABILITY; "
            "  EPILEPSY: refractory multifocal seizures; West syndrome; "
            "  CONGENITAL BLINDNESS: optic atrophy; cortical visual impairment; "
            "  AXIAL HYPOTONIA → limb spasticity; "
            "  DYSMORPHIC FEATURES: not consistent — not a recognised dysmorphic syndrome; "
            "  BRAIN MRI: progressive cerebral atrophy; periventricular white matter signal; "
            "  REGRESSION: developmental plateau followed by loss of milestones; "
            "BIOCHEMISTRY: "
            "  urine AICAR (AICA-ribotide monophosphate metabolites) MASSIVELY ELEVATED; "
            "  plasma AICA-riboside elevated; "
            "  urine organic acids: AICAR not detected on standard OA — SPECIFIC PURINE SCREEN required; "
            "  serum uric acid: NORMAL (purine synthesis blocked before IMP → reduced urate); "
            "OUTCOME: poor prognosis; all reported cases severely affected"
        ),
        "disease_pathway": (
            "DE NOVO PURINE SYNTHESIS — ATIC (STEPS 9 AND 10): "
            "Step 9 (AICAR transformylase): AICAR + 10-formyl-THF → FAICAR + THF; "
            "  consumes one molecule of 10-formyl-THF (folate cycle); "
            "Step 10 (IMP cyclohydrolase): FAICAR → IMP + H2O (ring closure); "
            "ATIC LOF PATHOMECHANISM: "
            "  AICAR accumulates massively (steps 9-10 blocked); "
            "  IMP NOT produced → AMP/GMP deficit; "
            "  AICAR toxicity: "
            "    1. Inhibits AMP deaminase → adenosine accumulates → A1/A2 receptor overstimulation; "
            "    2. Inhibits adenylosuccinate lyase (ADSL) — indirect toxic effect on AMP synthesis; "
            "    3. 10-formyl-THF NOT consumed → folate cycle altered; "
            "    4. AMPK activation by AICAR at physiological concentrations is beneficial; "
            "       at PATHOLOGICAL concentrations → chronic AMPK activation + protein synthesis suppression; "
            "ACADESINE (AICAR): paradoxically studied as cancer therapy + cardioprotectant at low doses; "
            "  at very high concentrations (as in ATIC deficiency) → neurotoxic; "
            "TREATMENT RATIONALE: "
            "  IMP + AMP deficit → adenine or hypoxanthine supplementation theoretical; "
            "  reduce AICAR accumulation: unknown how; "
            "  no proven therapy"
        ),
        "pathognomonic": (
            "URINE AICAR/AICA-RIBOSIDE MASSIVELY ELEVATED: "
            "  PATHOGNOMONIC; detected only by targeted purine metabolite HPLC/MS-MS; "
            "  NOT detected on standard urinary organic acid GC-MS or amino acid screens; "
            "  specific purine chromatography or metabolomics required; "
            "SERUM URIC ACID NORMAL/LOW: "
            "  contrasts with HPRT1 Lesch-Nyhan (uric acid HIGH); "
            "  ATIC blocks synthesis BEFORE IMP → less urate production; "
            "SEVERE REFRACTORY EPILEPSY + INTELLECTUAL DISABILITY + BLINDNESS TRIAD: "
            "  in neonate/infant with NORMAL uric acid + elevated urinary AICAR → ATIC; "
            "BRAIN MRI: progressive atrophy from infancy; white matter signal; optic tract atrophy; "
            "ATIC ENZYME ACTIVITY: fibroblasts/leukocytes; bifunctional assay; "
            "GENETIC CONFIRMATION: WES/WGS required given ultraorphan status; "
            "ONLY ~5 CASES WORLDWIDE: high clinical suspicion needed; consider in unexplained "
            "  refractory epilepsy + blindness + ID with normal metabolic workup when uric acid normal"
        ),
        "treatment": (
            "1. ANTIEPILEPTIC DRUGS: "
            "   West syndrome → ACTH first-line; "
            "   Refractory focal → polypharmacy (LEV + VPA + CLB); "
            "   Ketogenic diet: theoretical metabolic benefit (reduces purine synthesis substrate); "
            "2. ADENINE/HYPOXANTHINE SUPPLEMENTATION: investigational; theoretical bypass of IMP deficit; "
            "   no controlled evidence; single case reports only; "
            "3. ALLOPURINOL: NOT indicated (uric acid normal; may worsen by reducing PRPP consumption); "
            "4. FOLATE SUPPLEMENTATION: step 9 consumes 10-formyl-THF; supplementation theoretical; "
            "5. VISUAL REHABILITATION: dark adaptation training; cortical visual impairment programs; "
            "6. SUPPORTIVE: physio; NG/PEG feeding; anti-spasticity (baclofen); "
            "7. GENE THERAPY: no trials; gene small (592aa CDS); theoretical HSC approach; "
            "8. AVOID: high-purine diet (increases purine synthesis flux → more AICAR); "
            "   drugs inhibiting AICAR metabolism further (methotrexate inhibits folate cycle — "
            "   would reduce 10-formyl-THF for step 9 even further — ABSOLUTELY CONTRAINDICATED); "
            "PROGNOSIS: poor; all reported patients severely affected at last follow-up"
        ),
    },
    {
        "gene": "ADA",
        "seed_base": 2705,
        "protein": (
            "ADA -- 20q13.12 AR -- 363aa -- Adenosine-Deaminase-41kDa-Monomer-"
            "Purine-Catabolism-Adenosine+dAdenosine→Inosine+dInosine-"
            "OMIM-Gene-608958-Disease-ADA-SCID-102700"
        ),
        "locus": "20q13.12",
        "protein_size": "363 aa / 41 kDa (monomer; cytoplasmic + extracellular; zinc metalloenzyme)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in ADA; "
            "ADA deaminates adenosine → inosine and deoxyadenosine → deoxyinosine: "
            "  Adenosine + H2O → inosine + NH3; "
            "  dAdenosine + H2O → dInosine + NH3; "
            "ADA LOF → ADENOSINE and dADENOSINE accumulate: "
            "  dAdenosine → phosphorylated to dATP (by nucleoside kinases); "
            "  dATP massive accumulation → inhibits ribonucleotide reductase (RRM) → "
            "  DNA synthesis impaired → S-phase arrest → apoptosis in lymphocytes; "
            "  LYMPHOCYTE SELECTIVITY: lymphocytes cannot inactivate dATP via degradation (unlike other cells); "
            "  T, B, NK cells all depleted → combined immunodeficiency; "
            "DISEASE: ADA-SCID — one of the first diseases treated with gene therapy (1990); "
            "PREVALENCE: 1 in 500,000-1,000,000 births; ~15% of all SCID cases"
        ),
        "disease_category": (
            "ADA-SCID (OMIM 102700) + ADA NEUROLOGICAL FEATURES; "
            "PRIMARY PRESENTATION — SEVERE COMBINED IMMUNODEFICIENCY (SCID): "
            "  Complete lymphocyte depletion: T cells, B cells, NK cells all absent/very low; "
            "  Recurrent severe infections from birth: Pneumocystis, CMV, aspergillus, rotavirus; "
            "  Failure to thrive; "
            "  Without treatment: death in infancy (infections); "
            "NEUROLOGICAL FEATURES (underrecognised): "
            "  present in 40-60% of ADA-SCID patients DESPITE IMMUNE RECONSTITUTION; "
            "  SENSORINEURAL HEARING LOSS (SNHL): 35-50%; "
            "  COGNITIVE IMPAIRMENT: borderline-moderate ID in some; "
            "  AUTISM SPECTRUM: 15-20%; "
            "  ATTENTION DEFICIT: 20-30%; "
            "  MECHANISM: deoxyadenosine accumulation in CNS neurons (ADA normally high in brain); "
            "  neurological features NOT reversed by HSCT or gene therapy → intrinsic neuronal injury; "
            "PARTIAL ADA DEFICIENCY: "
            "  late/adult onset; lymphopenia but not full SCID; autoimmune features; "
            "  elevated dATP; immune dysregulation without infectious SCID phenotype"
        ),
        "disease_pathway": (
            "PURINE CATABOLISM — ADENOSINE DEAMINASE: "
            "Adenosine → inosine + NH3 (ADA); "
            "dAdenosine → dInosine + NH3 (ADA); "
            "ADA LOF PATHOMECHANISM: "
            "  dAdenosine → dAMP → dADP → dATP (by TK + NDP kinase); "
            "  dATP MASSIVELY ELEVATED (10-50x normal in lymphocytes); "
            "  dATP inhibits ribonucleotide reductase (RRM1/RRM2): "
            "    RRM converts NDP → dNDP; inhibition → all dNDP/dNTP depleted; "
            "    DNA replication HALT → S-phase arrest → lymphocyte apoptosis; "
            "  ADENOSINE accumulation: "
            "    ecto-ADA on RBC surface absent → plasma adenosine elevated; "
            "    A1/A2A/A3 receptor chronic stimulation → lymphocyte signal transduction altered; "
            "    adenosine → raises cAMP via A2A → immunosuppression; "
            "  CNS ADA NORMALLY HIGH: "
            "    ADA absent in neurons → dAdenosine toxic directly; "
            "    S-adenosylhomocysteine hydrolase (SAHH) irreversibly inhibited by dATP; "
            "    SAHH inhibition → SAM/methylation reactions impaired; "
            "TREATMENT RATIONALE: "
            "  PEG-ADA: exogenous ADA replaces enzyme; dATP normalises; "
            "  HSCT: restores lymphocytes; curative for immunity; "
            "  GENE THERAPY (ADA-SCID GT): first approved human gene therapy (EMA 2016, Strimvelis)"
        ),
        "pathognomonic": (
            "ERYTHROCYTE dATP ELEVATED: "
            "  normal <1 nmol/10^8 RBC; ADA-SCID >100 nmol/10^8; "
            "  PATHOGNOMONIC for ADA deficiency; "
            "RBC ADA ACTIVITY <0.01 nmol/h/mg Hb: normal 0.3-0.7; "
            "ABSOLUTE LYMPHOPENIA (<300/µL) FROM BIRTH + T, B, NK cell depletion: "
            "  SCID pattern PATHOGNOMONIC for ADA when RBC dATP elevated; "
            "URINE DEOXYADENOSINE/DEOXYINOSINE ELEVATED: mass spec; "
            "S-ADENOSYLHOMOCYSTEINE (SAH) ELEVATED in plasma: "
            "  ADA LOF → SAHH inhibited by dATP → SAH accumulates; "
            "  plasma SAH/SAM ratio raised; "
            "NEWBORN SCREENING (NBS): "
            "  TREC (T-cell receptor excision circles) low → SCID trigger → ADA enzyme in DBS; "
            "  US NBS since 2018; catching presymptomatic cases; "
            "SENSORINEURAL HEARING LOSS IN TREATED PATIENT: "
            "  suggests ADA neurological features despite immune reconstitution; "
            "  audiological monitoring MANDATORY annually in all ADA-SCID patients"
        ),
        "treatment": (
            "1. PEG-ADA (ELAPEGADEMASE, REVCOVI): "
            "   PEGylated bovine ADA; 0.4 mg/kg IM twice weekly; "
            "   restores immune function; bridges to HSCT/GT; "
            "   maintains dATP normalisation; lifelong if HSCT/GT not possible; "
            "   DOES NOT FULLY REVERSE NEUROLOGICAL MANIFESTATIONS; "
            "2. GENE THERAPY (STRIMVELIS, EMA 2016): "
            "   autologous HSC retrovirally transduced with ADA; one-time treatment; "
            "   >85% immune reconstitution sustained at 3 years; "
            "   preferred over HSCT in matched sibling unavailability; "
            "   BBMT 2024 long-term data: 88% event-free survival 10 years; "
            "3. HEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "   HLA-matched sibling preferred; 90%+ survival; "
            "   mismatched HSCT lower survival → GT often preferred; "
            "4. PROPHYLAXIS (infection prevention): "
            "   TMP-SMX (PCP prophylaxis); antifungal; IVIG; "
            "   LIVE VACCINES CONTRAINDICATED until immune reconstitution confirmed; "
            "5. NEUROLOGICAL MONITORING: "
            "   annual audiology (SNHL common); neurodevelopmental assessment; "
            "   hearing aids early if SNHL confirmed; "
            "6. STOP PEG-ADA: discontinue 30-60 days before gene therapy (antibody interference)"
        ),
    },
    {
        "gene": "PNP",
        "seed_base": 2706,
        "protein": (
            "PNP -- 14q11.2 AR -- 289aa -- Purine-Nucleoside-Phosphorylase-32kDa-Homotrimer-"
            "Inosine+Guanosine+dInosine+dGuanosine→Hypoxanthine+Guanine+Ribose-1-P-"
            "OMIM-Gene-164050-Disease-PNP-Deficiency-613179"
        ),
        "locus": "14q11.2",
        "protein_size": "289 aa / 32 kDa (homotrimer; cytoplasmic purine salvage/catabolism)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in PNP; "
            "PNP phosphorolyses purine nucleosides: "
            "  Inosine + Pi → hypoxanthine + ribose-1-phosphate; "
            "  Guanosine + Pi → guanine + ribose-1-phosphate; "
            "  dInosine + Pi → hypoxanthine + deoxyribose-1-phosphate; "
            "  dGuanosine + Pi → guanine + deoxyribose-1-phosphate; "
            "PNP LOF → dGUANOSINE (most toxic) and inosine accumulate; "
            "dGuo → dGTP (by TK + NDP kinase); "
            "dGTP massively elevated in T lymphocytes → ribonucleotide reductase inhibited → "
            "DNA synthesis blocked → T-cell apoptosis; "
            "B CELLS SPARED (dGTP not as toxic to B cells) → "
            "SELECTIVE T-CELL IMMUNODEFICIENCY (unlike ADA-SCID which depletes T+B+NK); "
            "NEUROLOGICAL FEATURES in 60-70%: "
            "  developmental regression; spastic diplegia/quadriplegia; "
            "  mechanism: dGTP/guanosine toxic to neurons; CNS PNP activity high normally; "
            "PREVALENCE: ~70 cases worldwide; extremely rare"
        ),
        "disease_category": (
            "PNP DEFICIENCY (OMIM 613179); "
            "SELECTIVE T-CELL IMMUNODEFICIENCY + NEUROLOGICAL TRIAD: "
            "  1. T-CELL IMMUNODEFICIENCY: "
            "     CD4+ and CD8+ T cells severely depleted; "
            "     B cells normal/elevated (characteristic; contrasts ADA-SCID); "
            "     NK cells variable; "
            "     Recurrent viral infections: VZV, CMV, EBV (T-cell pathogens); "
            "     Opportunistic infections: PCP, candida; "
            "     AUTOIMMUNITY paradoxically common (haemolytic anaemia, ITP) — "
            "       abnormal B-cell regulation without T-cell oversight; "
            "  2. NEUROLOGICAL FEATURES (60-70%): "
            "     spastic diplegia; hypotonia → spasticity; "
            "     intellectual disability (moderate-severe in most); "
            "     ATAXIA (cerebellar, uncommon); "
            "     developmental regression after normal early period; "
            "  3. AUTOIMMUNE HAEMOLYTIC ANAEMIA (Coombs positive): 50%; "
            "     ITP (immune thrombocytopenia): 20%; "
            "BIOCHEMISTRY: "
            "  serum URIC ACID VERY LOW (HYPOURICAEMIA): "
            "    PNP produces hypoxanthine/guanine for XDH → uric acid; "
            "    PNP LOF → less substrate for XDH → very low uric acid; "
            "    HYPOURICAEMIA PATHOGNOMONIC for PNP deficiency; "
            "  INOSINE + GUANOSINE elevated in urine; "
            "  dGTP elevated in T lymphocytes"
        ),
        "disease_pathway": (
            "PURINE SALVAGE/CATABOLISM — PURINE NUCLEOSIDE PHOSPHORYLASE: "
            "Inosine → hypoxanthine + R1P (PNP reaction; R1P = ribose-1-phosphate); "
            "Guanosine → guanine + R1P; "
            "dInosine → hypoxanthine + dR1P; "
            "dGuanosine → guanine + dR1P; "
            "PNP LOF PATHOMECHANISM: "
            "  dGuanosine most toxic: accumulates → TK2/cytosolic TK → dGDP → dGTP; "
            "  dGTP MASSIVELY ELEVATED in T cells: "
            "    dGTP inhibits ribonucleotide reductase → dNTP pool collapse → T cell death; "
            "  Hypoxanthine production reduced → HPRT1 cannot salvage → PRPP accumulates; "
            "  Guanine production reduced → de novo purine synthesis compensates; "
            "  URIC ACID: hypoxanthine/xanthine → uric acid (XDH); "
            "    PNP LOF → less hypoxanthine → less xanthine → very low uric acid; "
            "  CNS: neurons high in PNP normally; "
            "    dGuanosine/guanosine accumulate → direct neuronal toxicity; "
            "    myelin abnormalities on MRI; "
            "TREATMENT RATIONALE: "
            "  HSCT restores PNP-competent T cells; "
            "  gene therapy: in development; "
            "  PNP inhibitor drugs (forodesine) paradoxically designed to INHIBIT PNP in cancer T-cells — "
            "  CONTRAINDICATED in PNP deficiency (would worsen)"
        ),
        "pathognomonic": (
            "HYPOURICAEMIA (serum uric acid <1.0 mg/dL, often <0.5 mg/dL): "
            "  PATHOGNOMONIC for PNP deficiency; "
            "  Normal uric acid: 2.5-7.0 mg/dL; PNP deficiency: often undetectable; "
            "  CRITICAL DISTINCTION FROM ADA-SCID: ADA-SCID uric acid normal; PNP very low; "
            "URINE INOSINE + GUANOSINE + dGUANOSINE ELEVATED: "
            "  purine metabolite chromatography (HPLC); "
            "T-CELL LYMPHOPENIA WITH NORMAL/ELEVATED B CELLS: "
            "  selective T-cell deficiency PATHOGNOMONIC for PNP; "
            "  contrasts ADA-SCID (T+B+NK depleted); "
            "PNP ENZYME ACTIVITY <1% in erythrocytes: "
            "  normal 14-36 nmol/h/mg Hb; PNP deficiency: <0.1; "
            "AUTOIMMUNE HAEMOLYTIC ANAEMIA in T-cell immunodeficient patient: "
            "  raises PNP deficiency on differential; "
            "NEUROLOGICAL REGRESSION + SPASTIC DIPLEGIA in immunodeficient infant: "
            "  particularly with low uric acid → PNP deficiency until proven otherwise; "
            "FORODESINE (PNP inhibitor): CONTRAINDICATED — would worsen dGTP accumulation"
        ),
        "treatment": (
            "1. HEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "   only curative option for immune reconstitution; "
            "   HLA-matched sibling or unrelated donor; "
            "   T-cell reconstitution: 3-6 months; "
            "   NEUROLOGICAL FEATURES: may NOT reverse with HSCT (intrinsic neuronal injury); "
            "   early HSCT before extensive neurological damage improves outcome; "
            "2. PEG-PNP: pegylated PNP enzyme replacement — investigational; no approved product; "
            "3. INFECTION PROPHYLAXIS: "
            "   TMP-SMX (PCP); acyclovir (VZV); IVIG; antifungal; "
            "   LIVE VACCINES CONTRAINDICATED (vaccine-strain viral disease reported); "
            "   CMV surveillance; "
            "4. AUTOIMMUNE MANAGEMENT: "
            "   haemolytic anaemia → steroids + IVIG; transfusion if severe; "
            "   ITP → IVIG; steroid-sparing agents; "
            "5. NEUROLOGICAL SUPPORT: physiotherapy; SALT; spasticity management (baclofen); "
            "6. GENE THERAPY: ADA-SCID GT success makes PNP-GT plausible; no approved product 2026; "
            "7. AVOID: forodesine (PNP inhibitor — used in T-cell lymphoma; absolutely CI in PNP deficiency); "
            "   live vaccines; high-purine diet; "
            "8. NEUROLOGICAL MONITORING: serial MRI; neurodevelopmental assessment; "
            "   neurological features may progress even after immune reconstitution"
        ),
    },
    {
        "gene": "APRT",
        "seed_base": 2707,
        "protein": (
            "APRT -- 16q24.3 AR -- 180aa -- Adenine-Phosphoribosyltransferase-20kDa-Homodimer-"
            "Adenine-Salvage-Adenine+PRPP→AMP+PPi-"
            "OMIM-Gene-102600-Disease-2-8-Dihydroxyadenine-Urolithiasis-614723"
        ),
        "locus": "16q24.3",
        "protein_size": "180 aa / 20 kDa (homodimer; cytoplasmic; adenine salvage only)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in APRT; "
            "APRT salvages adenine: adenine + PRPP → AMP + PPi; "
            "APRT LOF → adenine NOT salvaged → adenine degraded by xanthine oxidase: "
            "  adenine → 8-hydroxyadenine → 2,8-dihydroxyadenine (DHA); "
            "  DHA is EXTREMELY INSOLUBLE in urine (solubility <0.3 mg/dL); "
            "  DHA PRECIPITATES in renal tubules and bladder → urolithiasis; "
            "  KIDNEY: DHA deposits cause tubular damage + interstitial nephritis; "
            "  progressive renal failure if untreated; "
            "TYPE I (most common worldwide): complete LOF (frameshift/nonsense/missense); "
            "TYPE II (Japan): partial LOF (p.Met136Thr); "
            "  Japanese APRT deficiency prevalence 1:1,000 (highest worldwide); "
            "  TYPE II: lower DHA excretion; milder nephropathy; "
            "PREVALENCE: Japan ~1:27,000; worldwide <200 reported cases (2026); "
            "IMPORTANT: APRT deficiency is curable — allopurinol + low-adenine diet"
        ),
        "disease_category": (
            "2,8-DIHYDROXYADENINE UROLITHIASIS (OMIM 614723); "
            "CLINICAL MANIFESTATIONS: "
            "  1. UROLITHIASIS: DHA stones; may appear in neonate, infant, child, or adult; "
            "     stones RADIOLUCENT on plain X-ray (common diagnostic pitfall — mistaken for uric acid); "
            "     CT-KUB: radiopaque on CT; characteristic density; "
            "     STONE COMPOSITION: 2,8-DHA (confirmed by infrared spectroscopy or mass spec); "
            "  2. PROGRESSIVE CHRONIC KIDNEY DISEASE (CKD): "
            "     DHA crystal deposits → interstitial nephritis; tubular damage; "
            "     untreated: ESRD (20-50 years if onset childhood); "
            "     early treatment prevents CKD progression; "
            "  3. RENAL COLIC: recurrent episodes from infancy; "
            "  4. URINARY TRACT INFECTIONS: secondary to stone obstruction; "
            "  5. HAEMATURIA; microscopic or gross; "
            "NEUROLOGICAL: NONE (contrast HPRT1, ADA, PNP) — purely renal disease; "
            "IMMUNOLOGICAL: NONE; "
            "SERUM URIC ACID: usually NORMAL (adenine → DHA via XO, not urate pathway); "
            "  DISTINGUISH FROM GOUT: APRT = DHA stones, normal UA; gout = urate stones, high UA; "
            "DIAGNOSIS OFTEN DELAYED: DHA stones radiolucent; DHA not on standard stone panels"
        ),
        "disease_pathway": (
            "ADENINE SALVAGE — APRT: "
            "Adenine + PRPP → AMP + PPi (APRT reaction); "
            "APRT LOF PATHOMECHANISM: "
            "  Dietary adenine NOT salvaged → excess adenine substrate; "
            "  Adenine oxidised by xanthine oxidase (XO): "
            "    Adenine → 8-hydroxyadenine (XO reaction 1); "
            "    8-hydroxyadenine → 2,8-dihydroxyadenine (XO reaction 2); "
            "  DHA: solubility in urine <0.3 mg/dL (pH 5-7); "
            "    glomerular filtration → tubular concentration → PRECIPITATION; "
            "    tubular DHA deposits → interstitial nephritis → CKD; "
            "    bladder DHA → stone crystallization; "
            "  AMP deficit from salvage block → compensated by de novo synthesis; "
            "    de novo synthesis not impaired → no immunological/neurological consequences; "
            "TREATMENT RATIONALE: "
            "  ALLOPURINOL: inhibits XO → blocks adenine → 8-hydroxyadenine → DHA conversion; "
            "    adenine accumulates (benign — not as toxic as DHA); "
            "    DHA production ceases → stones stop forming; kidneys protected; "
            "  LOW-ADENINE DIET: reduces substrate load; "
            "  ALKALINISATION: minimal benefit (DHA solubility pH-independent compared to urate)"
        ),
        "pathognomonic": (
            "2,8-DIHYDROXYADENINE STONES: "
            "  PATHOGNOMONIC for APRT deficiency; "
            "  brownish-purple stones (contrast: uric acid = yellow-orange); "
            "  infrared spectroscopy or mass spectrometry of stone = diagnosis; "
            "  RADIOLUCENT on plain X-ray — commonly mistaken for uric acid stones; "
            "  CT density 100-700 HU — opaque on CT; "
            "URINE DHA CRYSTALS: "
            "  brown round crystals with 'Maltese cross' birefringence; "
            "  urine microscopy; but easily missed; "
            "URINE DHA SPOT TEST: purple colour with ammoniacal silver nitrate; not widely available; "
            "APRT ENZYME ACTIVITY <1% in erythrocytes: "
            "  normal 18-30 nmol/h/mg Hb; APRT deficiency: undetectable; "
            "SERUM URIC ACID NORMAL: "
            "  critical DDx — differentiates from HPRT1 Lesch-Nyhan (high UA) and "
            "  PNP deficiency (low UA); APRT = normal UA; "
            "DIAGNOSIS COMMONLY DELAYED 5-20 YEARS: "
            "  stones labelled 'uric acid'; patients started on allopurinol empirically → "
            "  coincidentally correct treatment (XO inhibition); "
            "STONE ANALYSIS: MANDATORY in all urolithiasis — infrared spectroscopy is definitive"
        ),
        "treatment": (
            "1. ALLOPURINOL: xanthine oxidase inhibitor; FIRST-LINE; "
            "   10-20 mg/kg/day children; 300-600 mg/day adults; "
            "   inhibits adenine → 8-hydroxyadenine → DHA; "
            "   DHA excretion ceases → stones stop; kidney disease arrested; "
            "   LIFELONG treatment required; "
            "2. LOW-PURINE / LOW-ADENINE DIET: "
            "   reduce adenine substrate: avoid organ meats, sardines, yeast, beer; "
            "   adjunct to allopurinol; alone insufficient; "
            "3. FEBUXOSTAT: alternative XO inhibitor if allopurinol not tolerated; "
            "   limited evidence in APRT deficiency; "
            "4. HIGH FLUID INTAKE: 2-3 L/day; dilutes urine; reduces DHA concentration; "
            "5. RENAL TRANSPLANTATION: for ESRD; "
            "   CONTINUE ALLOPURINOL POST-TRANSPLANT: DHA nephropathy recurs in transplanted kidney; "
            "   allopurinol-calcineurin interaction: allopurinol + azathioprine → "
            "     FATAL MYELOSUPPRESSION (XO metabolises 6-MP/azathioprine); "
            "     switch azathioprine to mycophenolate in transplanted APRT patients on allopurinol; "
            "6. UROLOGICAL: lithotripsy or ureteroscopy for obstructing stones; "
            "7. UTI TREATMENT: standard antibiotics; "
            "IMPORTANT: APRT deficiency is potentially CURABLE with early allopurinol; "
            "KEY PITFALL: azathioprine + allopurinol in transplant → myelosuppression — ALWAYS avoid"
        ),
    },
    {
        "gene": "DGUOK",
        "seed_base": 2708,
        "protein": (
            "DGUOK -- 2p13.1 AR -- 277aa -- Deoxyguanosine-Kinase-30kDa-Monomer-"
            "Mitochondrial-Matrix-dGuo+dAdo→dGMP+dAMP-Mitochondrial-dNTP-Pool-"
            "OMIM-Gene-601465-Disease-Hepatocerebral-MDDS3-251880"
        ),
        "locus": "2p13.1",
        "protein_size": "277 aa / 30 kDa (monomer; mitochondrial matrix targeting sequence N-terminal)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in DGUOK; "
            "DGUOK phosphorylates PURINE deoxyribonucleosides inside mitochondria: "
            "  dGuanosine + ATP → dGMP + ADP (DGUOK reaction, rate-limiting for dGTP); "
            "  dAdenosine + ATP → dAMP + ADP (DGUOK reaction, also dATP); "
            "DGUOK LOF → mitochondrial dGTP + dATP DEFICIENT: "
            "  mtDNA replication requires all 4 dNTPs from MITOCHONDRIAL pools; "
            "  cytoplasmic dNTP cannot efficiently enter mitochondria in post-mitotic cells; "
            "  dGTP/dATP deficit → mtDNA replication stalls → mtDNA DEPLETION (<10-30% normal); "
            "ORGANS MOST AFFECTED: liver (high mtDNA requirement) and brain (neurons); "
            "HEPATOCEREBRAL SYNDROME = most common DGUOK phenotype; "
            "CONTRAST TK2: TK2 → PYRIMIDINE dNTP (dTTP+dCTP) deficient → MYOPATHIC MDDS; "
            "  DGUOK → PURINE dNTP (dGTP+dATP) deficient → HEPATOCEREBRAL MDDS; "
            "PREVALENCE: most common hepatocerebral MDDS; hundreds of cases worldwide"
        ),
        "disease_category": (
            "HEPATOCEREBRAL mtDNA DEPLETION SYNDROME TYPE 3 (DGUOK-MDDS3, OMIM 251880); "
            "TWO MAJOR PHENOTYPES: "
            "  1. HEPATOCEREBRAL FORM (neonatal/infantile, most common): "
            "     LIVER: neonatal cholestasis → progressive liver failure; "
            "       jaundice; hepatomegaly; coagulopathy; "
            "       elevated transaminases + GGT; "
            "       liver histology: steatosis + mtDNA depletion + respiratory chain deficiency; "
            "       50-70% die from liver failure first year without transplant; "
            "     BRAIN: neurological progression after liver transplant or independent: "
            "       nystagmus (rotatory, early feature); "
            "       psychomotor regression; hypotonia; "
            "       pontine abnormalities on MRI; "
            "       seizures; "
            "     KEY PROGNOSTIC FACTOR: neurological involvement BEFORE liver transplant → "
            "       poor post-transplant neurological outcome; "
            "  2. ISOLATED HEPATIC FORM (rarer, milder): "
            "     liver disease only; neurological sparing; "
            "     some patients survive to adulthood with liver disease alone; "
            "BIOCHEMISTRY: "
            "  lactate ELEVATED (plasma + CSF); pyruvate elevated; L:P ratio raised; "
            "  liver transaminases elevated; coagulopathy; hypoglycaemia; "
            "  mtDNA depletion in liver biopsy (quantitative PCR) — DEFINITIVE; "
            "  respiratory chain complex deficiencies: CI + CIII + CIV (all mtDNA-encoded subunits)"
        ),
        "disease_pathway": (
            "MITOCHONDRIAL PURINE DEOXYRIBONUCLEOSIDE SALVAGE — DGUOK: "
            "dGuo → dGMP → dGDP → dGTP (DGUOK + NMP/NDP kinases); "
            "dAdo → dAMP → dADP → dATP (DGUOK + NMP/NDP kinases); "
            "DGUOK LOF PATHOMECHANISM: "
            "  Mitochondrial dGTP and dATP SEVERELY DEPLETED; "
            "  mtDNA polymerase gamma (POLG) requires all 4 mitochondrial dNTPs; "
            "  dGTP/dATP depletion → POLG stalls at G/A insertion → mtDNA replication incomplete; "
            "  mtDNA copy number falls to <10-30% normal in affected tissues; "
            "  Respiratory chain: CI, CIII, CIV, CV all require mtDNA-encoded subunits; "
            "    all 13 mtDNA-encoded OXPHOS subunits lost → COMBINED RESPIRATORY CHAIN DEFICIENCY; "
            "    affected tissues: liver (highest mtDNA turnover + post-mitotic hepatocytes), brain; "
            "  LIVER: hepatocyte energy failure → apoptosis → liver failure; "
            "  BRAIN: neuronal energy failure → regression; "
            "TREATMENT RATIONALE: "
            "  No deoxyribonucleoside therapy for DGUOK (contrast TK2): "
            "    dGuo supplementation tried → may worsen (dGuo → dGTP inhibits RRM in cytoplasm); "
            "  Liver transplantation: corrects hepatic disease; neurological outcome variable"
        ),
        "pathognomonic": (
            "NEONATAL CHOLESTASIS + ROTATORY NYSTAGMUS + LACTIC ACIDOSIS TRIAD: "
            "  PATHOGNOMONIC combination for DGUOK hepatocerebral MDDS; "
            "  nystagmus distinguishes DGUOK from other neonatal hepatopathies; "
            "mtDNA DEPLETION IN LIVER BIOPSY: "
            "  <20% of age-matched controls by quantitative PCR; DEFINITIVE; "
            "COMBINED RESPIRATORY CHAIN DEFICIENCY: "
            "  CI+CIII+CIV all deficient in liver (CI+CIII+CIV = all use mtDNA-encoded subunits); "
            "  CII NORMAL (fully nuclear-encoded); "
            "  CII normal + CI/III/IV deficient = mtDNA depletion pattern; "
            "PLASMA LACTATE >5 mmol/L + ELEVATED TRANSAMINASES + COAGULOPATHY: "
            "  in neonate → hepatocerebral MDDS including DGUOK; "
            "MRI BRAIN: bilateral symmetric T2 hyperintensities in basal ganglia + brainstem; "
            "  periventricular white matter changes; cerebellar; "
            "CONTRAST POLG: POLG → Alpers syndrome (cortex + liver); DGUOK → brainstem + liver; "
            "CONTRAST TK2: TK2 = MUSCLE (COX-negative fibres); DGUOK = LIVER + BRAIN; "
            "LIVER BIOPSY mtDNA DEPLETION: mandatory for definitive diagnosis pre-transplant"
        ),
        "treatment": (
            "1. LIVER TRANSPLANTATION: "
            "   orthotopic liver transplantation (OLT) for isolated hepatic form; "
            "   CONTRAINDICATED if neurological features present BEFORE transplant → "
            "     neurological progression continues despite liver transplant; "
            "   neurological assessment MANDATORY before listing; "
            "   EEG + MRI brain + developmental milestones review pre-OLT; "
            "2. NUTRITIONAL SUPPORT: "
            "   glucose infusion (prevents hypoglycaemia); "
            "   vitamin K + FFP (coagulopathy); "
            "   MCFA formula (bypass mitochondrial fatty acid beta-oxidation impairment); "
            "3. ANTIEPILEPTIC DRUGS: "
            "   AVOID VALPROATE — hepatotoxic in MDDS; absolute contraindication; "
            "   preferred: levetiracetam; phenobarbital; benzodiazepines; "
            "4. RIBOFLAVIN (B2): 100-300 mg/day; some CI/CIII patients respond; "
            "   limited DGUOK-specific evidence; "
            "5. UBIQUINONE (CoQ10): 5-10 mg/kg/day; mitochondrial support; "
            "6. DEOXYRIBONUCLEOSIDE THERAPY: "
            "   NOT recommended for DGUOK (contrast TK2); "
            "   dGuo supplementation may worsen via cytoplasmic dGTP elevation; "
            "7. GENE THERAPY: theoretical AAV-based liver gene therapy; "
            "   neonatal hepatocyte transduction — preclinical only (2026); "
            "8. AVOID: valproate (absolute CI); metformin (CI complex I inhibitor); "
            "   fasting (ketogenesis adds metabolic stress); hepatotoxic drugs"
        ),
    },
    {
        "gene": "PRPS1",
        "seed_base": 2709,
        "protein": (
            "PRPS1 -- Xq22.3 XLR -- 318aa -- Phosphoribosyl-Pyrophosphate-Synthetase-1-"
            "34kDa-Hexamer-Purine+Pyrimidine-de-Novo-Synthesis-PRPP-Production-"
            "OMIM-Gene-311850-Disease-Arts-Syndrome-301835-CMTX5-311070"
        ),
        "locus": "Xq22.3",
        "protein_size": "318 aa / 34 kDa (hexamer; cytoplasmic; produces PRPP from R5P + ATP)",
        "inheritance": (
            "X-LINKED RECESSIVE — hemizygous LOF in males (gain-of-function variants cause different disease); "
            "PRPS1 converts ribose-5-phosphate + ATP → PRPP (phosphoribosyl pyrophosphate) + AMP; "
            "PRPP is the central substrate donor for ALL purine + pyrimidine de novo synthesis and salvage: "
            "  HGPRT (HPRT1): hypoxanthine/guanine + PRPP → IMP/GMP; "
            "  APRT: adenine + PRPP → AMP; "
            "  OPRT (UMPS step 5): orotic acid + PRPP → OMP (pyrimidine); "
            "PRPS1 LOF → PRPP DEFICIENCY → ALL purine + pyrimidine synthesis impaired; "
            "MULTIPLE PRPS ISOFORMS: PRPS1, PRPS2, PRPS1L1 partially compensate; "
            "LOF disease spectrum (males): "
            "  Severe LOF → ARTS SYNDROME (OMIM 301835): "
            "    intellectual disability + ataxia + sensorineural hearing loss (SNHL) + muscle hypotonia; "
            "    optic atrophy + peripheral neuropathy; early immunodeficiency; severe; "
            "  Milder LOF → CMTX5 (Charcot-Marie-Tooth X-linked type 5): "
            "    peripheral neuropathy + hearing loss + optic atrophy; "
            "  GAIN-OF-FUNCTION PRPS1: superactivity → gout + SNHL (different disease)"
        ),
        "disease_category": (
            "ARTS SYNDROME (OMIM 301835) — Severe PRPS1 LOF; "
            "CMTX5 (OMIM 311070) — Mild/Moderate PRPS1 LOF; "
            "ARTS SYNDROME CLINICAL FEATURES (males): "
            "  1. INTELLECTUAL DISABILITY: profound; "
            "  2. ATAXIA: cerebellar; progressive; truncal + gait; "
            "  3. SENSORINEURAL HEARING LOSS (SNHL): severe-profound; early childhood; "
            "  4. MUSCLE HYPOTONIA: generalised from birth; "
            "  5. OPTIC ATROPHY: progressive visual loss; "
            "  6. PERIPHERAL NEUROPATHY: demyelinating + axonal; "
            "  7. RECURRENT INFECTIONS: T-cell immunodeficiency (purine synthesis impaired in lymphocytes); "
            "  PROGNOSIS: most severely affected males die in childhood from infections; "
            "  HETEROZYGOUS FEMALES: mild-moderate features (SNHL + mild neuropathy); "
            "    X-inactivation skewing determines severity in females; "
            "CMTX5 CLINICAL FEATURES: "
            "  peripheral neuropathy; SNHL; optic atrophy; "
            "  intellectual function relatively preserved; "
            "  survival to adulthood; "
            "BIOCHEMISTRY: "
            "  low erythrocyte PRPP synthetase activity; "
            "  uric acid LOW-NORMAL (less de novo purine synthesis); "
            "  adenine + orotic acid may be mildly elevated (compensation)"
        ),
        "disease_pathway": (
            "PRPP PRODUCTION — PRPS1 (CENTRAL METABOLITE): "
            "Ribose-5-phosphate + ATP → PRPP + AMP (PRPS1 reaction); "
            "PRPP used by: HPRT1 + APRT (salvage); OPRT (pyrimidine de novo); "
            "  glutamine PRPP amidotransferase (PPAT): first step de novo purine synthesis; "
            "PRPS1 LOF PATHOMECHANISM: "
            "  PRPP DEFICIENCY → ALL pathways requiring PRPP reduced: "
            "  1. PURINE DE NOVO: PPAT reduced → less IMP → less AMP/GMP → AMP/GMP deficit; "
            "  2. PURINE SALVAGE: HPRT1 reaction reduced (less PRPP available); "
            "  3. PYRIMIDINE DE NOVO: OPRT step reduced (less PRPP for OMP synthesis); "
            "  NET EFFECT: ALL nucleotide pools reduced → affects DNA/RNA synthesis in ALL proliferating cells; "
            "  NEURONS: non-proliferating but high purine turnover for neurotransmission (ATP → ADP → AMP); "
            "    purine deficit → ATP for Na/K-ATPase reduced → neuronal excitability abnormal; "
            "  LYMPHOCYTES: proliferating rapidly → PRPP deficiency → immunodeficiency; "
            "  INNER EAR HAIR CELLS: extremely high purine turnover for mechanoelectric transduction; "
            "    PRPP deficiency → hair cell degeneration → SNHL; "
            "  OPTIC NERVE: axonal energy failure; "
            "TREATMENT RATIONALE: "
            "  Purine supplementation (adenine + inosine): theoretical bypass; "
            "  S-adenosylmethionine (SAM): methionine-derived; may partially restore PRPP through alternative"
        ),
        "pathognomonic": (
            "MALE WITH PROFOUND ID + CEREBELLAR ATAXIA + SEVERE SNHL + OPTIC ATROPHY + IMMUNODEFICIENCY: "
            "  PATHOGNOMONIC PENTAD for Arts syndrome; "
            "  no other single diagnosis causes all 5 features together in hemizygous males; "
            "ERYTHROCYTE PRPP SYNTHETASE ACTIVITY <5% NORMAL: "
            "  normal 0.5-2.5 nmol/h/mg Hb; Arts syndrome: unmeasurable or <0.05; "
            "SERUM URIC ACID NORMAL/LOW-NORMAL: "
            "  less purine de novo synthesis → less urate; "
            "  contrast PRPS1 SUPERACTIVITY (gain-of-function) → HIGH uric acid + gout; "
            "  LOF vs GOF distinction CRITICAL — opposite directions of uric acid; "
            "AUDIOMETRY: severe-profound bilateral sensorineural loss; flat audiogram; "
            "NERVE CONDUCTION STUDIES: demyelinating + axonal peripheral neuropathy; "
            "VISUAL EVOKED POTENTIALS: prolonged/absent; optic atrophy on fundoscopy; "
            "MRI BRAIN: cerebellar atrophy; cortical atrophy in severe cases; "
            "LYMPHOCYTE COUNT: low T-cell counts; reduced lymphocyte proliferation to mitogens; "
            "HETEROZYGOUS FEMALE RELATIVES: may show mild SNHL + neuropathy — "
            "  family testing important; X-inactivation ratio directs severity prediction"
        ),
        "treatment": (
            "1. INFECTION PROPHYLAXIS AND MANAGEMENT: "
            "   TMP-SMX (PCP prophylaxis); IVIG; antifungal; "
            "   bacterial infections → aggressive antibiotics; "
            "   LIVE VACCINES CONTRAINDICATED; "
            "2. HEARING REHABILITATION: "
            "   bilateral cochlear implants (SNHL profound); early implantation preferred; "
            "   hearing aids if residual hearing; "
            "   intensive speech + language therapy post-implant; "
            "3. VISUAL SUPPORT: "
            "   low vision aids; anti-VEGF if neovascular component; "
            "   genetic counselling for family; "
            "4. ANTIEPILEPTIC DRUGS (if seizures): "
            "   levetiracetam; avoid valproate (mitochondrial sensitivity); "
            "5. ADENINE + INOSINE SUPPLEMENTATION: "
            "   investigational; provides purine salvage substrates bypassing PRPP deficit; "
            "   limited evidence; case reports only; "
            "6. S-ADENOSYLMETHIONINE (SAM): investigational; purine support; "
            "7. GENE THERAPY: X-linked; lentiviral PRPS1 in HSCs — theoretical; no trials 2026; "
            "8. PHYSIOTHERAPY: ataxia management; gait training; orthotics; "
            "9. HEMATOPOIETIC STEM CELL TRANSPLANTATION: "
            "   corrects immunodeficiency; does NOT correct neurological/hearing/visual features; "
            "   consider in severe immunodeficiency if HLA match available; "
            "10. AVOID: allopurinol (reduces PRPP consumption by XO pathway → may help; paradoxical); "
            "    high-purine diet; nephrotoxic drugs; "
            "PROGNOSIS: Arts syndrome — poor in males; most die in childhood; females milder"
        ),
    },
]


def _generate_patients(gene_idx, n=40, seed=2702):
    """Generate realistic synthetic patient cohort for one gene."""
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]
    g = gene["gene"]
    patients = []

    # Gene-specific phenotype rates
    RATES = {
        "HPRT1": {
            "self_injurious": 0.75, "hyperuricaemia": 0.98, "dystonia": 0.90,
            "id_severe": 0.80, "seizures": 0.35, "renal_stones": 0.65,
            "immunodeficiency": 0.0, "hepatopathy": 0.0, "snhl": 0.15,
            "ataxia": 0.30, "optic_atrophy": 0.0, "neuropathy": 0.20,
            "lactic_acidosis": 0.0, "cardiomyopathy": 0.0,
            "mean_onset": 1.2, "sd_onset": 0.8,
        },
        "ADSL": {
            "self_injurious": 0.0, "hyperuricaemia": 0.0, "dystonia": 0.35,
            "id_severe": 0.80, "seizures": 0.75, "renal_stones": 0.0,
            "immunodeficiency": 0.0, "hepatopathy": 0.0, "snhl": 0.10,
            "ataxia": 0.30, "optic_atrophy": 0.05, "neuropathy": 0.10,
            "lactic_acidosis": 0.10, "cardiomyopathy": 0.05,
            "mean_onset": 0.4, "sd_onset": 0.5,
        },
        "ATIC": {
            "self_injurious": 0.0, "hyperuricaemia": 0.0, "dystonia": 0.30,
            "id_severe": 0.95, "seizures": 0.90, "renal_stones": 0.0,
            "immunodeficiency": 0.0, "hepatopathy": 0.0, "snhl": 0.20,
            "ataxia": 0.25, "optic_atrophy": 0.60, "neuropathy": 0.20,
            "lactic_acidosis": 0.15, "cardiomyopathy": 0.05,
            "mean_onset": 0.1, "sd_onset": 0.1,
        },
        "ADA": {
            "self_injurious": 0.0, "hyperuricaemia": 0.0, "dystonia": 0.10,
            "id_severe": 0.35, "seizures": 0.15, "renal_stones": 0.0,
            "immunodeficiency": 0.98, "hepatopathy": 0.05, "snhl": 0.45,
            "ataxia": 0.10, "optic_atrophy": 0.05, "neuropathy": 0.10,
            "lactic_acidosis": 0.05, "cardiomyopathy": 0.05,
            "mean_onset": 0.3, "sd_onset": 0.3,
        },
        "PNP": {
            "self_injurious": 0.0, "hyperuricaemia": 0.0, "dystonia": 0.30,
            "id_severe": 0.60, "seizures": 0.25, "renal_stones": 0.05,
            "immunodeficiency": 0.92, "hepatopathy": 0.10, "snhl": 0.15,
            "ataxia": 0.35, "optic_atrophy": 0.10, "neuropathy": 0.35,
            "lactic_acidosis": 0.05, "cardiomyopathy": 0.05,
            "mean_onset": 0.8, "sd_onset": 0.6,
        },
        "APRT": {
            "self_injurious": 0.0, "hyperuricaemia": 0.0, "dystonia": 0.0,
            "id_severe": 0.0, "seizures": 0.0, "renal_stones": 0.98,
            "immunodeficiency": 0.0, "hepatopathy": 0.05, "snhl": 0.0,
            "ataxia": 0.0, "optic_atrophy": 0.0, "neuropathy": 0.0,
            "lactic_acidosis": 0.0, "cardiomyopathy": 0.0,
            "mean_onset": 12.0, "sd_onset": 12.0,
        },
        "DGUOK": {
            "self_injurious": 0.0, "hyperuricaemia": 0.0, "dystonia": 0.30,
            "id_severe": 0.60, "seizures": 0.45, "renal_stones": 0.0,
            "immunodeficiency": 0.10, "hepatopathy": 0.92, "snhl": 0.10,
            "ataxia": 0.35, "optic_atrophy": 0.15, "neuropathy": 0.20,
            "lactic_acidosis": 0.88, "cardiomyopathy": 0.15,
            "mean_onset": 0.15, "sd_onset": 0.2,
        },
        "PRPS1": {
            "self_injurious": 0.0, "hyperuricaemia": 0.0, "dystonia": 0.25,
            "id_severe": 0.70, "seizures": 0.30, "renal_stones": 0.0,
            "immunodeficiency": 0.55, "hepatopathy": 0.05, "snhl": 0.92,
            "ataxia": 0.80, "optic_atrophy": 0.70, "neuropathy": 0.65,
            "lactic_acidosis": 0.10, "cardiomyopathy": 0.05,
            "mean_onset": 0.5, "sd_onset": 0.5,
        },
    }

    rates = RATES.get(g, {})
    xlinked = g in ("HPRT1", "PRPS1")

    for i in range(n):
        sex = "M" if xlinked else rng.choice(["M", "F"])
        onset = max(0.0, rng.gauss(rates.get("mean_onset", 2.0), rates.get("sd_onset", 2.0)))
        age_now = max(onset + 0.5, rng.uniform(onset + 1, onset + 20))
        age_now = round(min(age_now, 45.0), 1)
        onset = round(onset, 2)

        patients.append({
            "patient_id": f"{g}-{seed:04d}-{i+1:02d}",
            "gene": g,
            "sex": sex,
            "age_onset_years": onset,
            "age_current_years": age_now,
            "self_injurious": rng.random() < rates.get("self_injurious", 0),
            "hyperuricaemia": rng.random() < rates.get("hyperuricaemia", 0),
            "dystonia": rng.random() < rates.get("dystonia", 0),
            "id_severe": rng.random() < rates.get("id_severe", 0),
            "seizures": rng.random() < rates.get("seizures", 0),
            "renal_stones": rng.random() < rates.get("renal_stones", 0),
            "immunodeficiency": rng.random() < rates.get("immunodeficiency", 0),
            "hepatopathy": rng.random() < rates.get("hepatopathy", 0),
            "snhl": rng.random() < rates.get("snhl", 0),
            "ataxia": rng.random() < rates.get("ataxia", 0),
            "optic_atrophy": rng.random() < rates.get("optic_atrophy", 0),
            "peripheral_neuropathy": rng.random() < rates.get("neuropathy", 0),
            "lactic_acidosis": rng.random() < rates.get("lactic_acidosis", 0),
            "cardiomyopathy": rng.random() < rates.get("cardiomyopathy", 0),
        })

    return patients


def generate_overview():
    all_patients = []
    gene_summaries = []
    seeds = list(range(2702, 2710))

    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=seeds[idx])
        all_patients.extend(patients)
        n = len(patients)

        sib_n = sum(1 for p in patients if p["self_injurious"])
        ua_n = sum(1 for p in patients if p["hyperuricaemia"])
        dys_n = sum(1 for p in patients if p["dystonia"])
        id_n = sum(1 for p in patients if p["id_severe"])
        seiz_n = sum(1 for p in patients if p["seizures"])
        stone_n = sum(1 for p in patients if p["renal_stones"])
        immune_n = sum(1 for p in patients if p["immunodeficiency"])
        hep_n = sum(1 for p in patients if p["hepatopathy"])
        snhl_n = sum(1 for p in patients if p["snhl"])
        atax_n = sum(1 for p in patients if p["ataxia"])
        optic_n = sum(1 for p in patients if p["optic_atrophy"])
        neuro_n = sum(1 for p in patients if p["peripheral_neuropathy"])
        lactic_n = sum(1 for p in patients if p["lactic_acidosis"])
        mean_onset = round(sum(p["age_onset_years"] for p in patients) / n, 1)

        gene_summaries.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "mean_onset_years": mean_onset,
            "self_injurious_pct": round(100 * sib_n / n, 1),
            "hyperuricaemia_pct": round(100 * ua_n / n, 1),
            "dystonia_pct": round(100 * dys_n / n, 1),
            "id_severe_pct": round(100 * id_n / n, 1),
            "seizures_pct": round(100 * seiz_n / n, 1),
            "renal_stones_pct": round(100 * stone_n / n, 1),
            "immunodeficiency_pct": round(100 * immune_n / n, 1),
            "hepatopathy_pct": round(100 * hep_n / n, 1),
            "snhl_pct": round(100 * snhl_n / n, 1),
            "ataxia_pct": round(100 * atax_n / n, 1),
            "optic_atrophy_pct": round(100 * optic_n / n, 1),
            "neuropathy_pct": round(100 * neuro_n / n, 1),
            "lactic_acidosis_pct": round(100 * lactic_n / n, 1),
        })

    pathway_categories = [
        "PURINE-SALVAGE-COMPLETE-BLOCK — HPRT1: hypoxanthine/guanine NOT salvaged → PRPP accumulates → "
        "de novo purine overproduction → URIC ACID EXCESS + dopaminergic deficit → LESCH-NYHAN self-injurious + dystonia + gout",
        "DE-NOVO-PURINE-SYNTHESIS-STEP-8+AMP-SYNTHESIS — ADSL: SAICAR+adenylosuccinate accumulate → "
        "succinylpurine toxicity → AUTISM+EPILEPSY+ID; CSF S-Ado:SAICAr PATHOGNOMONIC",
        "DE-NOVO-PURINE-SYNTHESIS-STEPS-9-10 — ATIC: AICAR massively accumulates → AICA-ribosiduria → "
        "severe refractory epilepsy + blindness + ID; ULTRAORPHAN (<5 cases worldwide)",
        "ADENOSINE-DEAMINASE-DEFICIENCY — ADA: dATP massively elevated → RRM inhibited → lymphocyte apoptosis → "
        "T+B+NK depleted (ADA-SCID); FIRST GENE THERAPY 1990; STRIMVELIS EMA 2016",
        "PURINE-NUCLEOSIDE-PHOSPHORYLASE-DEFICIENCY — PNP: dGTP elevated → SELECTIVE T-CELL IMMUNODEFICIENCY + "
        "spastic diplegia; HYPOURICAEMIA PATHOGNOMONIC; B cells normal (contrast ADA-SCID)",
        "ADENINE-SALVAGE-DEFICIENCY — APRT: adenine → 2,8-dihydroxyadenine (DHA) via XO → "
        "RADIOLUCENT DHA UROLITHIASIS + CKD; PURELY RENAL; NO NEURO/IMMUNE; ALLOPURINOL CURATIVE",
        "MITOCHONDRIAL-PURINE-DEOXYRIBONUCLEOSIDE-KINASE — DGUOK: mitochondrial dGTP+dATP deficient → "
        "mtDNA depletion in LIVER+BRAIN → HEPATOCEREBRAL MDDS; ROTATORY NYSTAGMUS EARLY HALLMARK; VPA ABSOLUTELY CI",
        "PRPP-SYNTHETASE-LOF — PRPS1: PRPP deficiency → ALL purine+pyrimidine synthesis impaired → "
        "ARTS SYNDROME pentad (ID+ataxia+SNHL+optic atrophy+immunodeficiency) in males; CMTX5 milder",
    ]

    critical_distinctions = [
        "URIC ACID AXIS: HPRT1 → VERY HIGH UA (overproduction); PNP → VERY LOW UA (underproduction); "
        "APRT → NORMAL UA (2,8-DHA not urate); PRPS1 LOF → LOW-NORMAL; PRPS1 GOF (superactivity) → HIGH UA + GOUT",
        "ADA vs PNP: ADA-SCID = T+B+NK depleted; PNP = SELECTIVE T-cell only + B cells NORMAL; "
        "ADA = high dATP; PNP = high dGTP; ADA uric acid normal; PNP uric acid very low",
        "LESCH-NYHAN SIB: HPRT1 ONLY — self-injurious behaviour COMPULSIVE and PATHOGNOMONIC; "
        "patients WANT restraints (contra-willful); NO other purine disorder causes SIB",
        "HEPATOCEREBRAL vs MYOPATHIC MDDS: DGUOK = LIVER+BRAIN (purine dNTP); TK2 = MUSCLE-ONLY (pyrimidine dNTP); "
        "ROTATORY NYSTAGMUS distinguishes DGUOK from other neonatal hepatopathies",
        "DHA UROLITHIASIS (APRT): RADIOLUCENT on X-ray (mistaken for uric acid stones) but OPAQUE on CT; "
        "NORMAL uric acid; ALLOPURINOL curative; azathioprine + allopurinol → FATAL myelosuppression",
        "PRPS1 LOF (Arts/CMTX5) vs GOF (superactivity): LOF → uric acid LOW + immunodeficiency + SNHL; "
        "GOF → uric acid VERY HIGH + gout + SNHL; same gene, OPPOSITE uric acid direction",
        "ADSL SUCCINYLPURINES: NOT on standard metabolic screens; must REQUEST specific succinylpurine HPLC; "
        "Belgian founder Arg426His → milder Type III; neonatal lethal alleles → no residual ADSL activity",
        "VPA CONTRAINDICATIONS IN PURINE DISORDERS: DGUOK (hepatocerebral MDDS) — hepatotoxic → ABSOLUTE CI; "
        "ADSL — seizures common but VPA caution (hepatotoxic risk); prefer LEV/PB in all mtDNA disorders",
    ]

    return {
        "atlas": "Hereditary Purine Disorder Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": len(all_patients),
        "seeds": seeds,
        "pathway_categories": pathway_categories,
        "critical_distinctions": critical_distinctions,
        "gene_summaries": gene_summaries,
    }


def generate_breakdown():
    breakdown = []
    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=2702 + idx)
        n = len(patients)

        breakdown.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "n_patients": n,
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "patients": patients,
        })

    return {"atlas": "Hereditary Purine Disorder Atlas", "genes": breakdown}


def generate_definitions():
    glossary = {
        "PRPP (Phosphoribosyl Pyrophosphate)": (
            "Central metabolite produced by PRPS1 from ribose-5-phosphate + ATP; "
            "required substrate for ALL purine + pyrimidine de novo synthesis and salvage; "
            "PRPS1 LOF → PRPP deficiency → ALL nucleotide pathways impaired"
        ),
        "HGPRT (Hypoxanthine-Guanine Phosphoribosyltransferase)": (
            "Encoded by HPRT1; salvages hypoxanthine → IMP and guanine → GMP using PRPP; "
            "highest expression in basal ganglia neurons; LOF → Lesch-Nyhan disease; "
            "complete LOF → SIB + dystonia + hyperuricaemia; partial LOF → Kelley-Seegmiller variant"
        ),
        "dNTP Pool Imbalance — Purine": (
            "DGUOK LOF → mitochondrial dGTP+dATP deficient (hepatocerebral MDDS); "
            "ADA LOF → cytoplasmic dATP massively elevated (ADA-SCID lymphocyte apoptosis); "
            "PNP LOF → dGTP elevated in T cells (selective T-cell immunodeficiency); "
            "each causes distinct organ-selective disease based on cell-type dNTP dependency"
        ),
        "Succinylpurines (SAICAr + S-Ado)": (
            "ADSL deficiency biomarkers: succinylaminoimidazole carboxamide riboside (SAICAr) + "
            "adenylosuccinate (S-Ado); detected by succinylpurine HPLC/MS; "
            "NOT on standard organic acid/amino acid screens — specific request required; "
            "CSF S-Ado > SAICAr → severe phenotype; SAICAr > S-Ado → milder"
        ),
        "AICAR (5-Aminoimidazole-4-Carboxamide Ribonucleotide)": (
            "Intermediate in de novo purine synthesis step 9 (ATIC substrate); "
            "ATIC LOF → AICAR accumulates → dephosphorylated to AICA-riboside in plasma; "
            "at physiological doses: AMPK activator (cardioprotective + anti-cancer, acadesine); "
            "at pathological doses (ATIC deficiency): neurotoxic; inhibits AMP deaminase + ADSL"
        ),
        "2,8-Dihydroxyadenine (DHA) Urolithiasis": (
            "APRT deficiency: adenine → 8-hydroxyadenine → 2,8-DHA (via xanthine oxidase); "
            "DHA extremely insoluble → precipitates in renal tubules + bladder → stones + CKD; "
            "RADIOLUCENT on plain X-ray (mistaken for uric acid); OPAQUE on CT; "
            "allopurinol CURATIVE — blocks XO → DHA production ceases"
        ),
        "Selective T-Cell vs Combined Immunodeficiency": (
            "ADA-SCID: T cells + B cells + NK cells ALL depleted (dATP toxic to all lymphocytes); "
            "PNP deficiency: ONLY T cells depleted (dGTP selectively toxic to T cells; B cells normal); "
            "Distinction: ADA → dATP; PNP → dGTP; ADA → combined; PNP → T-cell only; "
            "PNP: B-cell excess → autoimmune features (haemolytic anaemia, ITP) without T-cell regulation"
        ),
        "Arts Syndrome vs CMTX5 (PRPS1)": (
            "Same gene (PRPS1), different LOF severity: "
            "Arts syndrome (severe LOF): profound ID + cerebellar ataxia + SNHL + optic atrophy + immunodeficiency; "
            "CMTX5 (milder LOF): peripheral neuropathy + SNHL + optic atrophy; intelligence relatively preserved; "
            "CONTRAST: PRPS1 SUPERACTIVITY (gain-of-function) → gout + SNHL → opposite direction"
        ),
        "Hepatocerebral mtDNA Depletion (DGUOK)": (
            "DGUOK LOF → mitochondrial dGTP+dATP deficit → mtDNA depletion in LIVER + BRAIN; "
            "ROTATORY NYSTAGMUS early feature (distinguishes from other neonatal hepatopathies); "
            "Combined CI+CIII+CIV deficiency with CII NORMAL = mtDNA depletion pattern; "
            "LIVER TRANSPLANT: corrects hepatic disease; CONTRAINDICATED if neurological already present; "
            "VPA ABSOLUTELY CONTRAINDICATED in all hepatocerebral mtDNA depletion disorders"
        ),
        "Purine Pathway — De Novo vs Salvage": (
            "DE NOVO (10 steps, cytoplasmic): "
            "  PPAT (step 1, PRPP→PRA) → ... → ADSL (step 8, SAICAR→AICAR) → ATIC (steps 9-10, AICAR→IMP); "
            "  IMP → AMP (via ADSL+ADSS) or GMP; "
            "SALVAGE: "
            "  HPRT1: hypoxanthine/guanine + PRPP → IMP/GMP; "
            "  APRT: adenine + PRPP → AMP; "
            "  ADA → PNP: adenosine/inosine/guanosine catabolism; "
            "MITOCHONDRIAL SALVAGE: DGUOK (dGuo+dAdo→dGMP+dAMP); TK2 (dThd+dCyd, pyrimidine)"
        ),
    }

    return {
        "atlas": "Hereditary Purine Disorder Atlas",
        "gene_entries": {g["gene"]: {
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
        } for g in ATLAS_GENES},
        "glossary": glossary,
    }
