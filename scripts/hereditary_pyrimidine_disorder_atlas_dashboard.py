"""Hereditary Pyrimidine Disorder Atlas — 8-Gene Reference
TYMP-DPYD-DPYS-UPB1-CAD-DHODH-RRM2B-TK2
320 patients (8 x 40), seeds 2694-2701.
Endpoints: /api/hereditary-pyrimidine-disorder-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "TYMP",
        "seed_base": 2694,
        "protein": (
            "TYMP -- 22q13.33 AR -- 482aa -- Thymidine-Phosphorylase-54kDa-Homodimer-"
            "Pyrimidine-Nucleoside-Catabolism-Thymidine+Deoxyuridine-Accumulate-"
            "OMIM-Gene-131222-Disease-MNGIE-603041"
        ),
        "locus": "22q13.33",
        "protein_size": "482 aa / 54 kDa (homodimer; cytoplasmic thymidine/deoxyuridine phosphorylase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in TYMP; "
            "TYMP encodes thymidine phosphorylase (TP), which catabolises thymidine → thymine + 2-deoxyribose-1-phosphate "
            "and deoxyuridine → uracil + 2-deoxyribose-1-phosphate in pyrimidine salvage; "
            "TP LOF → thymidine (dThd) and deoxyuridine (dUrd) massively accumulate in plasma and urine; "
            "dThd/dUrd imbalance corrupts the mitochondrial dNTP pool → mtDNA multiple deletions + depletion → "
            "mitochondrial dysfunction in post-mitotic tissues (gut smooth muscle, neurons); "
            "DISEASE: MNGIE — Mitochondrial NeuroGastroIntestinal Encephalomyopathy (OMIM 603041); "
            "PREVALENCE: ~120 patients worldwide reported; consanguineous families; no ethnic predilection; "
            "KEY MUTATIONS: frameshift + nonsense most common; missense p.Tyr95Cys (founder, Japanese); "
            "p.Ser111Pro (Portuguese/Spanish); p.Ala289Val; complete null alleles → most severe; "
            "DISEASE ONSET: typically 2nd-3rd decade (range 5 months–35 years); "
            "PLASMA dThd >3 µmol/L + plasma dUrd >5 µmol/L = PATHOGNOMONIC biochemical diagnosis"
        ),
        "disease_category": (
            "MNGIE — MITOCHONDRIAL NEUROGASTROINTESTINAL ENCEPHALOMYOPATHY (OMIM 603041); "
            "CLINICAL HALLMARKS (PENTAD): "
            "  1. GASTROINTESTINAL DYSMOTILITY: borborygmi, early satiety, postprandial emesis, diarrhoea, "
            "     intestinal pseudo-obstruction — due to smooth muscle mitochondrial dysfunction; "
            "  2. CACHEXIA: extreme weight loss (BMI often <14); malnutrition; "
            "  3. PERIPHERAL NEUROPATHY: sensorimotor demyelinating neuropathy (NCS slowed); "
            "  4. LEUKOENCEPHALOPATHY: T2 hyperintensity white matter (MRI brain); symmetric periventricular; "
            "  5. EXTERNAL OPHTHALMOPLEGIA + PTOSIS: CPEO (chronic progressive) — due to mtDNA defects in EOMs; "
            "ADDITIONAL: hearing loss; GERD; sensorineural; muscle weakness; retinal pigmentary changes; "
            "BIOCHEMISTRY: thymidine phosphorylase activity <10% normal in buffy coat (diagnostic); "
            "GI histology: vacuolation of smooth muscle; disrupted ICC networks; sparse mitochondria; "
            "PROGNOSIS: median survival ~35 years; most die from GI complications (aspiration, malnutrition); "
            "HISTOPATHOLOGY: intestinal muscle biopsy — electron microscopy: mitochondrial abnormalities + ragged red fibres"
        ),
        "disease_pathway": (
            "PYRIMIDINE NUCLEOSIDE CATABOLISM — THYMIDINE PHOSPHORYLASE: "
            "Thymidine (dThd) → thymine + 2-deoxyribose-1-phosphate (TP reaction); "
            "Deoxyuridine (dUrd) → uracil + 2-deoxyribose-1-phosphate (TP reaction); "
            "MNGIE PATHOMECHANISM: "
            "  TYMP LOF → dThd and dUrd accumulate in plasma/urine/tissues; "
            "  dThd → intracellularly phosphorylated to dTTP (by TK1/TK2); "
            "  dTTP excess → inhibits RRM via dTTP feedback inhibition → dCDP↓ → dCTP↓; "
            "  mitochondrial dNTP pool severely imbalanced (dTTP:dCTP ratio >100x normal); "
            "  imbalanced dNTP pool → mtDNA replication errors → mtDNA multiple deletions + depletion; "
            "  post-mitotic tissues (smooth muscle, neurons) most vulnerable (limited mtDNA repair capacity); "
            "TREATMENT RATIONALE: "
            "  HEMODIALYSIS: removes dThd/dUrd; partial benefit; "
            "  PLATELET INFUSIONS: provide exogenous TP; transient effect; "
            "  HEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): restores TP activity; "
            "    only curative option; registry evidence shows metabolic correction (dThd normalises); "
            "    clinical stabilisation or improvement in GI + neurological; "
            "    best outcomes: early transplant before severe GI damage; "
            "  ENZYME REPLACEMENT THERAPY (PEGylated-TP): Phase 2 trials (Enzon)"
        ),
        "pathognomonic": (
            "PLASMA dThd >3 µmol/L + dUrd >5 µmol/L: PATHOGNOMONIC combined elevation; "
            "measured by HPLC-MS/MS; diagnostic for MNGIE without genetic testing; "
            "distinguishes MNGIE from other mitochondrial GI disorders (MELAS, POLG); "
            "THYMIDINE PHOSPHORYLASE ACTIVITY <10%: in buffy coat leukocytes; "
            "normal: 14-130 nmol/h/mg protein; MNGIE: often undetectable; "
            "GI PSEUDO-OBSTRUCTION + CACHEXIA + LEUKOENCEPHALOPATHY + CPEO PENTAD: "
            "any 3 of 5 hallmarks in young adult → MNGIE until proven otherwise; "
            "BRAIN MRI — SYMMETRIC T2 LEUKOENCEPHALOPATHY: diffuse periventricular white matter signal; "
            "  spares cortex; does NOT correlate with clinical severity (can be severe with minor symptoms)"
        ),
        "treatment": (
            "1. HSCT (ALLOGENEIC BONE MARROW TRANSPLANT): only curative option for MNGIE; "
            "   restores leukocyte TP activity → plasma dThd/dUrd normalise within months; "
            "   GI dysmotility may improve over 12-24 months post-HSCT; "
            "   TIMING: before severe intestinal fibrosis/irreversible damage; "
            "   MORTALITY: 13-20% transplant-related in early series; improved with reduced-intensity conditioning; "
            "2. NUTRITIONAL SUPPORT: PEG/jejunostomy; parenteral nutrition if severe; "
            "3. HEMODIALYSIS: 3x/week reduces plasma dThd/dUrd ~60%; partial benefit; bridge to HSCT; "
            "4. PLATELET TRANSFUSIONS: provide TP; transient (hours); investigational; "
            "5. PEGylated-TP ENZYME REPLACEMENT: Phase 1/2 trials; metabolic normalisation; "
            "6. GI MANAGEMENT: prokinetics (metoclopramide, erythromycin); antiemetics; "
            "7. AVOID: thymidine analogues (AZT, d4T, zalcitabine) — worsen dNTP imbalance"
        ),
    },
    {
        "gene": "DPYD",
        "seed_base": 2695,
        "protein": (
            "DPYD -- 1p21.3 AR -- 1025aa -- Dihydropyrimidine-Dehydrogenase-111kDa-Homodimer-"
            "Rate-Limiting-Pyrimidine-Catabolism-Uracil+Thymine→DihydroUracil+DihydroThymine-"
            "OMIM-Gene-612779-Disease-DPD-Deficiency-274270"
        ),
        "locus": "1p21.3",
        "protein_size": "1025 aa / 111 kDa (homodimer; cytoplasmic NADPH-dependent oxidoreductase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (complete deficiency — rare, clinically severe); "
            "HETEROZYGOUS PATHOGENIC VARIANTS: pharmacogenomics relevance — 5-FU/capecitabine toxicity; "
            "DPYD encodes dihydropyrimidine dehydrogenase (DPD), which catalyses: "
            "  uracil + NADPH → dihydrouracil; thymine + NADPH → dihydrothymine (rate-limiting step); "
            "CPIC DPYD gene variants: "
            "  *2A (IVS14+1G>A, c.1905+1G>A): exon 14 skipping → null; ~0.5% European carriers; "
            "    ACTIVITY SCORE (AS) = 0 (null); GRADE A RECOMMENDATION: avoid fluoropyrimidines; "
            "  c.2846A>T (p.Asp949Val, *13): AS = 0.5; CPIC Grade A: 50% dose reduction; "
            "  HapB3 (c.1236G>A/HapB3): AS = 0.5; Grade A; "
            "  c.1679T>G (p.Ile560Ser, *12): null; rare; "
            "COMPLETE DPD DEFICIENCY (biallelic): 1 in 10,000 births; homozygous or compound het; "
            "PRESENTS: neonatal/infantile seizures, intellectual disability, autism features, motor delay; "
            "URINE: uracil >80 µmol/mmol creatinine + thymine >60 µmol/mmol creatinine = PATHOGNOMONIC; "
            "ADULT ONSET: very rare complete deficiency presenting only with 5-FU toxicity"
        ),
        "disease_category": (
            "DPD DEFICIENCY (COMPLETE) — OMIM 274270: "
            "CLINICAL PHENOTYPE (complete deficiency): "
            "  NEONATAL/INFANTILE: seizures (first weeks-months); hypotonia; intellectual disability; "
            "  AUTISM SPECTRUM FEATURES: social withdrawal, repetitive behaviours; "
            "  MOTOR DELAY: delayed milestones; hypotonia; spasticity in some; "
            "  MICROCEPHALY (variable); "
            "  GROWTH RETARDATION; "
            "ASYMPTOMATIC COMPLETE DEFICIENCY: reported — subset have no neurological features despite zero DPD; "
            "BIOCHEMISTRY: urine uracil markedly elevated; thymine elevated; no dihydrouracil/thymine; "
            "PHARMACOGENOMICS PHENOTYPE (partial deficiency, heterozygous): "
            "  CAPECITABINE/5-FU SEVERE TOXICITY: mucositis, myelosuppression, diarrhoea, hand-foot syndrome; "
            "  RISK: Grade 3-4 toxicity in ~30% of heterozygous *2A carriers given standard dose; "
            "  FATAL TOXICITY: documented in homozygous receiving full 5-FU dose; "
            "PRE-TREATMENT DPYD GENOTYPING: CPIC Grade A recommendation; "
            "EUROPEAN GUIDELINES (EMSO/DPYD consortium): mandatory DPYD testing before fluoropyrimidines"
        ),
        "disease_pathway": (
            "PYRIMIDINE CATABOLISM — DPD PATHWAY: "
            "Uracil → dihydrouracil (DPD, rate-limiting) → ureidopropionic acid (DPYS) → beta-alanine + CO2 + NH3 (UPB1); "
            "Thymine → dihydrothymine (DPD) → ureidoisobutyric acid (DPYS) → beta-aminoisobutyric acid + CO2 + NH3 (UPB1); "
            "DPD DEFICIENCY CONSEQUENCES: "
            "  uracil + thymine accumulate → competitively inhibit downstream salvage; "
            "  uracil may enter CNS → excitatory (GABA synthesis competition?); mechanism of seizures unclear; "
            "  NO uridine deficiency (de novo synthesis intact); "
            "5-FU PHARMACOLOGY: "
            "  5-fluorouracil (5-FU) → DPD degrades ~80-85% of administered 5-FU → dihydrofluorouracil; "
            "  DPD activity determines 5-FU half-life (high DPD → fast clearance → less activity; "
            "  Low DPD → slow clearance → 5-FU accumulates → severe toxicity); "
            "  Active 5-FU metabolites: FdUMP (TS inhibitor) + FUTP (RNA incorporation) — these cause toxicity/efficacy"
        ),
        "pathognomonic": (
            "URINE URACIL >80 µmol/mmol creatinine (normal <10): PATHOGNOMONIC for DPD deficiency; "
            "URINE THYMINE >60 µmol/mmol creatinine: concurrent elevation; "
            "ABSENCE OF DIHYDROURACIL IN URINE: substrate accumulates without product = enzyme block confirmed; "
            "DPYD *2A HETEROZYGOSITY + GRADE 4 MUCOSITIS AFTER 5-FU: "
            "  clinical pharmacogenomics scenario; CPIC Grade A; "
            "DPD ENZYME ACTIVITY IN PBMCs <10%: in complete deficiency; "
            "URACIL:DIHYDROURACIL RATIO IN PLASMA >6: suggested pharmacogenomics screening marker "
            "(plasma uracil >16 ng/mL → suggests reduced DPD phenotype before 5-FU)"
        ),
        "treatment": (
            "COMPLETE DPD DEFICIENCY: "
            "1. SEIZURE MANAGEMENT: standard AEDs (phenobarbital, VPA, LEV); "
            "   VPA CAUTION: may reduce DPD activity further — theoretical; monitor closely; "
            "2. SUPPORTIVE: early intervention, physiotherapy, educational support; "
            "3. NO SPECIFIC METABOLITE REPLACEMENT (uracil is NOT deficient — it accumulates); "
            "PHARMACOGENOMICS MANAGEMENT (partial DPD deficiency): "
            "4. DPYD *2A homozygous: AVOID 5-FU/capecitabine; alternative: raltitrexed, irinotecan; "
            "5. DPYD *2A heterozygous (AS=1): 50% dose reduction of 5-FU/capecitabine; "
            "6. DPYD c.2846A>T or HapB3 heterozygous (AS=1.5): 25-50% dose reduction; "
            "7. URIDINE TRIACETATE (Vistogard): FDA-approved antidote for 5-FU overdose or early-onset toxicity; "
            "   competes with 5-FU metabolites; give within 96h of last 5-FU dose; "
            "8. THERAPEUTIC DRUG MONITORING: 5-FU AUC-guided dosing (target AUC 20-30 mg*h/L) — reduces toxicity"
        ),
    },
    {
        "gene": "DPYS",
        "seed_base": 2696,
        "protein": (
            "DPYS -- 8q22.3 AR -- 414aa -- Dihydropyrimidinase-46kDa-Homotrimer-"
            "Second-Step-Pyrimidine-Catabolism-Ring-Opening-DihydroUracil+DihydroThymine→"
            "UreidopropiOnicAcid+UreidoisobutyricAcid-"
            "OMIM-Gene-126010-Disease-Dihydropyrimidinase-Deficiency-222748"
        ),
        "locus": "8q22.3",
        "protein_size": "414 aa / 46 kDa (homotrimer; zinc metalloenzyme; dihydropyrimidine amidohydrolase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in DPYS; "
            "DPYS encodes dihydropyrimidinase (DHP), the second enzyme in pyrimidine catabolism: "
            "  dihydrouracil → ureidopropionic acid (ring opening); "
            "  dihydrothymine → ureidoisobutyric acid (ring opening); "
            "DPYS LOF → dihydrouracil + dihydrothymine accumulate in urine and plasma; "
            "VERY RARE: fewer than 50 patients reported worldwide; "
            "MUTATIONS: missense predominantly; "
            "  p.Leu7Pro (Japanese founder); p.Arg302Gln (common in East Asian); "
            "  p.Thr404Ala; p.Gly400Asp; null alleles → severe; "
            "RESIDUAL ACTIVITY CORRELATES IMPERFECTLY with phenotype — partial enzyme activity "
            "may be clinically asymptomatic (detected incidentally or via urine OA screening); "
            "CLINICAL PENETRANCE VARIABLE: many homozygous individuals asymptomatic or mildly affected; "
            "NEONATAL SCREENING: detected by GCMS urine OA (dihydrouracil peak) in some programmes"
        ),
        "disease_category": (
            "DIHYDROPYRIMIDINASE DEFICIENCY — OMIM 222748: "
            "CLINICAL FEATURES (variable expressivity): "
            "  INTELLECTUAL DISABILITY: mild to moderate (in symptomatic patients); "
            "  EPILEPSY: febrile seizures; generalised seizures; "
            "  GASTROINTESTINAL: feeding difficulties; cyclic vomiting; "
            "  AUTISTIC FEATURES: social difficulties; repetitive behaviours; "
            "ASYMPTOMATIC CASES: substantial fraction; incidentally detected by urine OA; "
            "BIOCHEMISTRY HALLMARKS: "
            "  URINE DIHYDROURACIL markedly elevated (normally <1 µmol/mmol creatinine; "
            "    disease: >100 µmol/mmol creatinine); "
            "  URINE DIHYDROTHYMINE elevated; "
            "  URACIL: normal or mild elevation (DPD is intact, producing substrate normally); "
            "  UREIDOPROPIONIC ACID: absent or low (block at DPYS prevents its production); "
            "IMPORTANT: DIHYDROURACIL in urine can also be elevated after 5-FU exposure "
            "(5-FU → 5-FU-DH by DPD → 5,6-dihydrofluorouracil — different compound but peaks overlap on GC); "
            "DISTINGUISH: medication history essential; DHP enzyme assay on erythrocytes"
        ),
        "disease_pathway": (
            "PYRIMIDINE CATABOLISM — SECOND STEP (DPYS): "
            "Step 1: Uracil → Dihydrouracil (DPD/DPYD); Thymine → Dihydrothymine (DPD/DPYD); "
            "Step 2 [BLOCKED in DPYS deficiency]: "
            "  Dihydrouracil → Ureidopropionic acid (N-carbamoyl-beta-alanine) + H2O [DPYS, ring opening]; "
            "  Dihydrothymine → Ureidoisobutyric acid (N-carbamoyl-beta-aminoisobutyric acid) + H2O [DPYS]; "
            "Step 3 (downstream): "
            "  Ureidopropionic acid → beta-alanine + CO2 + NH3 (UPB1); "
            "  Ureidoisobutyric acid → beta-aminoisobutyric acid + CO2 + NH3 (UPB1); "
            "CONSEQUENCES OF DPYS BLOCK: "
            "  dihydrouracil + dihydrothymine accumulate; "
            "  no uridine/uracil pool deficiency (de novo synthesis intact); "
            "  potential neurotoxicity from dihydropyrimidine accumulation unclear (GABA metabolism?); "
            "TREATMENT RATIONALE: no substrate reduction possible; no downstream product deficiency; "
            "  management is purely symptomatic; "
            "PHARMACOGENOMICS NOTE: Patients with DPYS deficiency accumulate dihydrouracil normally; "
            "  5-FU metabolism — DPYS deficiency does NOT alter 5-FU toxicity (DPD step upstream is intact); "
            "  however DPYS variants may modify capecitabine metabolite clearance subtly"
        ),
        "pathognomonic": (
            "URINE DIHYDROURACIL >100 µmol/mmol creatinine: highly characteristic; "
            "  normally absent or trace; dramatically elevated in DPYS deficiency; "
            "CONCURRENT DIHYDROTHYMINE ELEVATION: distinguishes from DPD deficiency "
            "  (in DPD deficiency: uracil + thymine elevated, not dihydro-forms); "
            "UREIDOPROPIONIC ACID ABSENT: confirms block at DPYS not downstream (UPB1 produces it); "
            "DHP ENZYME ACTIVITY IN ERYTHROCYTES <10%: confirmatory; "
            "GC-MS URINE OA PROFILE: dihydrouracil peak (retention time characteristic) + dihydrothymine; "
            "NO 5-FU PHARMACOGENOMICS RISK (DPD intact): important clinical distinction from DPYD deficiency"
        ),
        "treatment": (
            "1. SYMPTOMATIC MANAGEMENT: no disease-specific treatment; "
            "2. SEIZURE CONTROL: standard AEDs (LEV, VPA, LTG based on seizure type); "
            "3. EDUCATIONAL SUPPORT: early intervention for cognitive/developmental delay; "
            "4. GI MANAGEMENT: antiemetics for cyclic vomiting; dietary advice for feeding difficulties; "
            "5. SURVEILLANCE: periodic neurodevelopmental assessment; EEG monitoring; "
            "6. GENETIC COUNSELLING: recurrence risk 25%; prenatal diagnosis available via DPYS sequencing; "
            "7. PROGNOSIS: variable — asymptomatic subset requires no intervention; "
            "   symptomatic subset benefits from early support but no cure; "
            "8. RESEARCH: beta-alanine supplementation theoretical (downstream product deficient) — "
            "   not proven effective in published cases; "
            "9. 5-FU SAFETY: DPYS deficiency does NOT increase 5-FU toxicity — DPD intact; "
            "   counsel patients regarding cancer treatment misconceptions"
        ),
    },
    {
        "gene": "UPB1",
        "seed_base": 2697,
        "protein": (
            "UPB1 -- 22q11.23 AR -- 404aa -- Beta-Ureidopropionase-44kDa-Homotrimer-"
            "Third-Step-Pyrimidine-Catabolism-UreidopropionicAcid→BetaAlanine+CO2+NH3-"
            "OMIM-Gene-613161-Disease-Beta-Ureidopropionase-Deficiency-613162"
        ),
        "locus": "22q11.23",
        "protein_size": "404 aa / 44 kDa (homotrimer; pyrimidine-specific amidohydrolase; PLP-independent)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in UPB1; "
            "UPB1 encodes beta-ureidopropionase (BUP-1), the final step of pyrimidine catabolism: "
            "  N-carbamoyl-beta-alanine (ureidopropionic acid) → beta-alanine + CO2 + NH3; "
            "  N-carbamoyl-beta-aminoisobutyric acid (ureidoisobutyric acid) → beta-aminoisobutyric acid + CO2 + NH3; "
            "UPB1 LOF → ureidopropionic acid + ureidoisobutyric acid accumulate; "
            "  ADDITIONALLY: beta-alanine is NOT produced → beta-alanine deficiency; "
            "EXTREMELY RARE: <20 cases published; "
            "MUTATIONS: missense + frameshift; "
            "  p.Arg326Cys (reported multiple times); p.Gln382Arg; "
            "CLINICAL: neurological features including epilepsy, intellectual disability, hypotonia; "
            "IMPORTANT LINK: beta-alanine is a competitive inhibitor of GABA transaminase → "
            "  beta-alanine DEFICIENCY → reduced GABA inhibitory tone → seizures (mechanism proposed); "
            "NEWBORN SCREENING: not routinely detected; OA profile required"
        ),
        "disease_category": (
            "BETA-UREIDOPROPIONASE DEFICIENCY — OMIM 613162: "
            "CLINICAL FEATURES: "
            "  EPILEPSY: neonatal/infantile seizures; febrile seizures; refractory in some; "
            "  INTELLECTUAL DISABILITY: moderate to severe; "
            "  HYPOTONIA: neonatal hypotonia; delayed motor milestones; "
            "  AUTISM SPECTRUM FEATURES: in some cases; "
            "  MICROCEPHALY: variable; "
            "BIOCHEMISTRY: "
            "  URINE UREIDOPROPIONIC ACID greatly elevated (normally absent); "
            "  URINE UREIDOISOBUTYRIC ACID elevated; "
            "  DIHYDROURACIL MILD elevation (backlog from UPB1 block); "
            "  BETA-ALANINE: reduced in urine (decreased production due to block); "
            "UNIQUE PATHOMECHANISM: beta-alanine deficiency → reduced inhibitory neurotransmission; "
            "  beta-alanine is a partial GABA-A agonist + competitive inhibitor of GABA transaminase; "
            "  low beta-alanine → increased GABA catabolism → net GABAergic tone reduction → epilepsy; "
            "TREATMENT HYPOTHESIS: beta-alanine supplementation may restore GABAergic tone; "
            "  small case reports suggest clinical improvement with beta-alanine"
        ),
        "disease_pathway": (
            "PYRIMIDINE CATABOLISM — THIRD STEP (UPB1): "
            "Step 1: Uracil → Dihydrouracil (DPD); Thymine → Dihydrothymine (DPD); "
            "Step 2: Dihydrouracil → Ureidopropionic acid (DPYS); Dihydrothymine → Ureidoisobutyric acid (DPYS); "
            "Step 3 [BLOCKED in UPB1 deficiency]: "
            "  Ureidopropionic acid → beta-ALANINE + CO2 + NH3 [UPB1]; "
            "  Ureidoisobutyric acid → beta-AMINOISOBUTYRIC ACID + CO2 + NH3 [UPB1]; "
            "CONSEQUENCES OF UPB1 BLOCK: "
            "  ureidopropionic acid + ureidoisobutyric acid accumulate; "
            "  beta-alanine NOT PRODUCED → deficiency; "
            "  beta-aminoisobutyric acid NOT PRODUCED; "
            "BETA-ALANINE FUNCTION: "
            "  GABA-A receptor partial agonist (low affinity); "
            "  inhibits GABA-T (GABA transaminase) → prevents GABA catabolism → net GABAergic effect; "
            "  carnosine synthesis precursor (beta-Ala + His → carnosine in muscle/brain); "
            "  pantothenate synthesis via beta-Ala + pantoate; "
            "BETA-ALANINE DEFICIENCY → reduced GABAergic tone → EPILEPSY (proposed mechanism); "
            "TREATMENT RATIONALE: beta-alanine supplementation to replace deficient product"
        ),
        "pathognomonic": (
            "URINE UREIDOPROPIONIC ACID ELEVATED (normally absent): characteristic marker; "
            "URINE UREIDOISOBUTYRIC ACID ELEVATED: concurrent; "
            "BETA-ALANINE ABSENT/LOW IN URINE: confirms product deficiency (unlike DPYS where beta-alanine intact); "
            "GC-MS URINE OA PROFILE: ureidopropionic acid + ureidoisobutyric acid peaks; "
            "UPB1 ENZYME ACTIVITY IN LIVER/LYMPHOCYTES <5%: confirmatory; "
            "DISTINGUISH FROM DPYS DEFICIENCY: "
            "  DPYS: dihydrouracil + dihydrothymine elevated; ureidopropionic acid ABSENT; "
            "  UPB1: ureidopropionic acid + ureidoisobutyric acid elevated; dihydrouracil mild/absent; "
            "BETA-ALANINE SUPPLEMENTATION TRIAL: clinical response supports UPB1 diagnosis"
        ),
        "treatment": (
            "1. BETA-ALANINE SUPPLEMENTATION: 100-200 mg/kg/day orally; "
            "   rationale: replaces deficient end-product; restores GABAergic inhibitory tone; "
            "   case reports: seizure improvement + developmental gains; NOT proven in RCT; "
            "2. ANTIEPILEPTIC DRUGS: LEV, VPA (caution: VPA may inhibit GABA-T — complex interaction); "
            "   LTG, CLB; ketogenic diet considered; "
            "3. NUTRITIONAL SUPPORT: carnosine (beta-Ala + His) — supplementation theoretical; "
            "4. PHYSIOTHERAPY + OCCUPATIONAL THERAPY: for motor/developmental delay; "
            "5. SURVEILLANCE: EEG, neuroimaging (MRI), neurodevelopmental assessments; "
            "6. GENETIC COUNSELLING: 25% recurrence; prenatal diagnosis via UPB1 sequencing; "
            "7. NO APPROVED THERAPY: all treatment is off-label and supportive; "
            "8. DISTINGUISH FROM DPYD/DPYS: critical — 5-FU pharmacogenomics does not apply to UPB1"
        ),
    },
    {
        "gene": "CAD",
        "seed_base": 2698,
        "protein": (
            "CAD -- 2p23.3 AR -- 2225aa -- CAD-Multifunctional-Enzyme-CPS2+ATCase+DHOase-243kDa-Hexamer-"
            "De-Novo-Pyrimidine-Biosynthesis-Cytoplasmic-First-Three-Steps-"
            "OMIM-Gene-114010-Disease-CAD-Deficiency-616457"
        ),
        "locus": "2p23.3",
        "protein_size": "2225 aa / 243 kDa (hexamer; trifunctional: CPS2 + ATCase + DHOase domains)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in CAD; "
            "CAD encodes a trifunctional enzyme catalysing the first three steps of de novo pyrimidine biosynthesis: "
            "  Domain 1 (CPS2 — carbamoyl phosphate synthetase 2): "
            "    Glutamine + 2 ATP + HCO3- → carbamoyl phosphate (cytoplasmic; distinct from CPS1 in urea cycle); "
            "  Domain 2 (ATCase — aspartate transcarbamylase): "
            "    Carbamoyl phosphate + aspartate → N-carbamoyl-aspartate; "
            "  Domain 3 (DHOase — dihydroorotase): "
            "    N-carbamoyl-aspartate → dihydroorotate; "
            "CAD LOF → severe impairment of de novo UMP synthesis → pyrimidine deficiency; "
            "VERY RARE: discovered in 2015; approximately 10-15 patients described (as of 2024); "
            "MUTATIONS: predominantly missense; p.Arg2024Cys (recurrent); p.Pro1110Leu; "
            "DISEASE ONSET: first months of life; severe early-onset epileptic encephalopathy; "
            "KEY: uridine supplementation is immediately effective — highly specific treatment response; "
            "DIAGNOSIS: uridine responsiveness PATHOGNOMONIC; confirmed by urinary orotic acid measurement; "
            "  plasma: elevated dihydroorotate + orotic acid (downstream enzymes working on limited substrate?"
        ),
        "disease_category": (
            "CAD DEFICIENCY — OMIM 616457: "
            "CLINICAL HALLMARKS: "
            "  EARLY-ONSET EPILEPTIC ENCEPHALOPATHY: onset first 3-6 months; "
            "    seizure types: myoclonic, infantile spasms, generalised tonic-clonic, focal; "
            "    EEG: hypsarrhythmia (IS phenotype); multifocal epileptiform; "
            "  PROFOUND INTELLECTUAL DISABILITY (if untreated); "
            "  SEVERE GLOBAL DEVELOPMENTAL DELAY: pre-treatment; "
            "  MICROCEPHALY: progressive if untreated; "
            "  BLOOD FILM: hypersegmented neutrophils (megaloblastic-like haematopoiesis — pyrimidine deficiency); "
            "  ANAEMIA: macrocytic anaemia (megaloblastic — pyrimidine-starved erythropoiesis); "
            "URIDINE RESPONSE: dramatic — seizures cease within days of uridine treatment; "
            "  development resumes; head circumference recovers; anaemia resolves; "
            "  URIDINE SUPPLEMENTATION MUST BE LIFELONG (pyrimidine de novo synthesis cannot self-correct); "
            "BIOCHEMISTRY: "
            "  PLASMA DIHYDROOROTATE ELEVATED; "
            "  URINE OROTIC ACID: variable (may be low — substrate starvation); "
            "  URIDINE: low; "
            "DISTINCT FROM UMPS (orotic aciduria): CAD is upstream; UMPS is downstream"
        ),
        "disease_pathway": (
            "DE NOVO PYRIMIDINE BIOSYNTHESIS — FIRST 3 STEPS (CAD): "
            "Step 1 [CPS2 domain]: Glutamine + 2ATP + HCO3- → Carbamoyl phosphate (cytoplasmic); "
            "  Note: CPS1 is mitochondrial (urea cycle) — entirely different gene/function; "
            "Step 2 [ATCase domain]: Carbamoyl phosphate + Aspartate → N-carbamoyl-aspartate; "
            "Step 3 [DHOase domain]: N-carbamoyl-aspartate → Dihydroorotate; "
            "Downstream steps (other enzymes): "
            "  Step 4: DHODH — Dihydroorotate → Orotate (mitochondrial inner membrane); "
            "  Step 5: UMPS (OPRT domain) — Orotate + PRPP → OMP; "
            "  Step 6: UMPS (ODCase domain) — OMP → UMP; "
            "  UMP → UDP → UTP → CTP (+ dUMP → dTMP via TS); "
            "CAD BLOCK CONSEQUENCES: "
            "  NO carbamoyl phosphate for ATCase → all downstream pyrimidines from de novo STARVED; "
            "  salvage pathway (from diet/cell turnover) insufficient for rapidly dividing cells; "
            "  proliferating cells (neurons, erythroid precursors) most vulnerable; "
            "  TREATMENT: exogenous uridine → enters via UPB (salvage) → UMP → all pyrimidines produced"
        ),
        "pathognomonic": (
            "URIDINE SUPPLEMENTATION RESPONSE: dramatic seizure cessation within 48-72h PATHOGNOMONIC; "
            "  no other epileptic encephalopathy responds this specifically to uridine; "
            "BLOOD FILM — HYPERSEGMENTED NEUTROPHILS: pyrimidine starvation of haematopoiesis; "
            "MACROCYTIC ANAEMIA RESPONSIVE TO URIDINE (not B12/folate): PATHOGNOMONIC combination; "
            "PLASMA DIHYDROOROTATE ELEVATED: substrate accumulates before DHODH step; "
            "NEONATAL EEG — HYPSARRHYTHMIA + MYOCLONUS responding to uridine NOT ACTH/VGB: key alert; "
            "URINE OROTIC ACID: may be low or normal (substrate starvation prevents orotate production); "
            "  CONTRAST UMPS (orotic aciduria): UMPS has HIGH urine orotic acid; CAD has LOW; "
            "EEG NORMALISATION WITHIN 2 WEEKS of uridine: specific marker of CAD vs other DEE"
        ),
        "treatment": (
            "1. URIDINE SUPPLEMENTATION: IMMEDIATE LIFE-CHANGING TREATMENT; "
            "   oral uridine 100-200 mg/kg/day divided TID; "
            "   available as dietary supplement (TriAcetyluridine = Nucleoside supplement — better bioavailability); "
            "   SEIZURES CEASE WITHIN 48-72h; development resumes; anaemia resolves; "
            "2. TRIACETYL URIDINE (TAU): uridine prodrug; 3x better oral bioavailability vs uridine; "
            "   dose: 60 mg/kg/day TAU ≈ 200 mg/kg/day uridine; preferred; "
            "3. LIFELONG TREATMENT: pyrimidine de novo synthesis permanently impaired; "
            "   never discontinue; acute deterioration on withdrawal; "
            "4. SUPPORTIVE AEDs: while awaiting uridine response; valproate, LEV; "
            "5. NEURODEVELOPMENTAL REHABILITATION: intensive post-treatment; "
            "6. EARLY DIAGNOSIS CRITICAL: irreversible cortical damage if untreated > 6 months; "
            "7. FAMILY SCREENING: 25% sibling risk; diagnose before symptom onset → prevent encephalopathy; "
            "8. BLOOD FILM MONITORING: hypersegmented neutrophils resolve with adequate uridine dose → compliance check"
        ),
    },
    {
        "gene": "DHODH",
        "seed_base": 2699,
        "protein": (
            "DHODH -- 16q22.2 AR -- 395aa -- Dihydroorotate-Dehydrogenase-43kDa-Monomer-"
            "Inner-Mitochondrial-Membrane-Coenzyme-Q-Linked-De-Novo-Pyrimidine-Step4-"
            "OMIM-Gene-126064-Disease-Miller-Syndrome-263750"
        ),
        "locus": "16q22.2",
        "protein_size": "395 aa / 43 kDa (monomer; inner mitochondrial membrane; FAD + CoQ10 cofactors)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in DHODH; "
            "DHODH encodes dihydroorotate dehydrogenase, step 4 of de novo pyrimidine biosynthesis: "
            "  dihydroorotate + CoQ → orotate + CoQH2 (on inner mitochondrial membrane); "
            "  DHODH is the ONLY mitochondrially located enzyme in de novo pyrimidine pathway; "
            "  DHODH is also the pharmacological target of leflunomide (DHODH inhibitor → immunosuppression); "
            "DISEASE: MILLER SYNDROME — Postaxial Acrofacial Dysostosis (POADS, OMIM 263750); "
            "  a syndrome of limb + craniofacial defects NOT a metabolic disease in the traditional sense; "
            "  LIMB DEFECTS: postaxial (ulnar/fibular) limb hypoplasia — 4th + 5th digit aplasia/hypoplasia; "
            "  CRANIOFACIAl DEFECTS: malar hypoplasia, downslanting palpebral fissures, micrognathia, "
            "    cleft palate/lip, coloboma of eyelids; "
            "MUTATIONS: compound het most common; "
            "  p.Arg346Glu (most frequent pathogenic allele); p.Glu264Lys; p.Thr385Pro; p.Leu253Val; "
            "PREVALENCE: extremely rare; ~70 families reported; "
            "PATHOMECHANISM: DHODH LOF → transient pyrimidine deficiency during critical developmental windows "
            "(6-10 weeks gestation) → impaired proliferation of specific neural crest and limb bud cell populations"
        ),
        "disease_category": (
            "MILLER SYNDROME (POSTAXIAL ACROFACIAL DYSOSTOSIS, POADS) — OMIM 263750: "
            "HALLMARK FEATURES: "
            "  POSTAXIAL LIMB DEFECTS (ulnar/fibular side = 4th+5th rays): "
            "    absent or hypoplastic 4th and 5th metacarpals/phalanges + metatarsals; "
            "    absent or hypoplastic ulna (forearm shortening); fibula hypoplasia; "
            "    supernumerary or absent digits on ulnar/fibular side; "
            "  CRANIOFACIAL: "
            "    malar hypoplasia (underdeveloped cheekbones); "
            "    micrognathia (small jaw); "
            "    downslanting palpebral fissures; "
            "    lower eyelid coloboma; "
            "    cleft palate +/- cleft lip; "
            "    cup-shaped external ears; "
            "    choanal atresia (some cases); "
            "DIFFERENTIAL DIAGNOSIS: "
            "  Treacher Collins syndrome (TCOF1/POLR1C/POLR1D): craniofacial-only, no limb defects; "
            "  Nager syndrome (SF3B4): preaxial (radial/tibial) limb defects + craniofacial; "
            "INTELLECTUAL DEVELOPMENT: typically NORMAL — pyrimidine deficiency transient (embryonic only); "
            "  adult DHODH-deficient individuals have NO metabolic disease postnatally "
            "  (diet provides sufficient pyrimidines via salvage for maintenance); "
            "THERAPY: surgical reconstruction of limb/craniofacial defects; "
            "  LEFLUNOMIDE CONTRAINDICATED: would further inhibit residual DHODH activity"
        ),
        "disease_pathway": (
            "DE NOVO PYRIMIDINE BIOSYNTHESIS — STEP 4 (DHODH): "
            "Step 4 [DHODH, inner mitochondrial membrane]: "
            "  Dihydroorotate + CoQ10 (oxidised) → Orotate + CoQ10H2 (reduced); "
            "  electrons transferred to CoQ → respiratory chain; "
            "  DHODH is functionally linked to mitochondrial electron transport chain; "
            "Steps 5-6 (UMPS, cytoplasmic): "
            "  Orotate + PRPP → OMP (OPRT domain); OMP → UMP (ODCase domain); "
            "  UMP → UDP → UTP → CTP; "
            "EMBRYONIC vs POSTNATAL PYRIMIDINE REQUIREMENTS: "
            "  EMBRYO (high proliferation): primarily dependent on de novo synthesis; "
            "    DHODH deficiency → transient pyrimidine starvation at critical windows; "
            "    postaxial limb bud mesenchyme + neural crest cells = highly sensitive populations; "
            "  ADULT (low proliferation): salvage pathway from dietary/recycled pyrimidines sufficient; "
            "    DHODH patients have NO metabolic disease as adults; "
            "LEFLUNOMIDE (DHODH INHIBITOR) MECHANISM: "
            "  teriflunomide (active metabolite) inhibits DHODH → pyrimidine starvation of proliferating "
            "  lymphocytes → immunosuppression (rheumatoid arthritis, multiple sclerosis)"
        ),
        "pathognomonic": (
            "POSTAXIAL LIMB DEFECTS (4th+5th rays) + LOWER EYELID COLOBOMA + MALAR HYPOPLASIA: "
            "  this triad in a newborn → MILLER SYNDROME until proven otherwise; "
            "PREAXIAL LIMB DEFECTS ABSENT: distinguishes from Nager syndrome (SF3B4 — radial/tibial); "
            "NORMAL INTELLIGENCE: unlike most skeletal dysplasias with structural brain involvement; "
            "DHODH BIALLELIC MUTATIONS: molecular confirmation; "
            "NO METABOLIC ABNORMALITY IN POSTNATAL PERIOD: "
            "  plasma uridine, orotate, dihydroorotate NORMAL in postnatal life; "
            "  DISTINGUISHES from CAD/UMPS (metabolic disease with measurable OA abnormalities); "
            "LEFLUNOMIDE TERATOGENICITY PARALLEL: leflunomide causes similar limb defects in animal models "
            "  (pharmacological DHODH inhibition during embryogenesis); "
            "TREACHER COLLINS DDx: no limb defects in Treacher Collins; TCOF1 gene; "
            "NAGER DDx: radial-side (thumb-side) defects in Nager, not ulnar; SF3B4 gene"
        ),
        "treatment": (
            "MILLER SYNDROME — SURGICAL AND MULTIDISCIPLINARY MANAGEMENT: "
            "1. CRANIOFACIAL SURGERY: cleft palate repair (3-6 months); maxillary advancement; jaw distraction; "
            "   lower eyelid reconstruction; choanal atresia repair (neonatal emergency if bilateral); "
            "2. LIMB SURGERY: prosthetics; finger reconstruction; stabilisation of hypoplastic ulna; "
            "   collaboration between paediatric orthopaedic + craniofacial + hand surgery; "
            "3. HEARING ASSESSMENT: sensorineural/conductive hearing loss — ENT; hearing aids; "
            "4. OPHTHALMOLOGY: eyelid coloboma management; amblyopia risk monitoring; "
            "5. FEEDING SUPPORT: nasogastric/G-tube if severe micrognathia; "
            "6. URIDINE SUPPLEMENTATION: investigated (rationale: replaces embryonically deficient pyrimidines); "
            "   NOT needed postnatally (diet sufficient); THEORETICAL prenatal benefit (maternal supplementation); "
            "7. LEFLUNOMIDE + TERIFLUNOMIDE: ABSOLUTELY CONTRAINDICATED (DHODH inhibitor — identical target); "
            "8. GENETIC COUNSELLING: prenatal diagnosis; preimplantation genetic testing (PGT); "
            "9. PROGNOSIS: normal intellectual development; quality of life dependent on surgical access"
        ),
    },
    {
        "gene": "RRM2B",
        "seed_base": 2700,
        "protein": (
            "RRM2B -- 8q23.1 AR-AD -- 351aa -- Ribonucleotide-Reductase-M2B-Small-Subunit-p53R2-40kDa-"
            "p53-Inducible-Non-S-Phase-dNTP-Synthesis-mtDNA-Repair-Maintenance-"
            "OMIM-Gene-604712-Disease-MDDS8A-612075-MDDS8B-612075"
        ),
        "locus": "8q23.1",
        "protein_size": "351 aa / 40 kDa (dimer with R1; p53-inducible; iron-tyrosyl radical active site)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic) — severe multi-system mtDNA depletion; OR "
            "AUTOSOMAL DOMINANT (heterozygous) — late-onset PEO/CPEO + mtDNA multiple deletions; "
            "RRM2B encodes p53R2 (p53-inducible small ribonucleotide reductase subunit), which provides: "
            "  dNTPs for mtDNA repair in G1/G0 non-dividing cells; "
            "  works with R1 subunit (RRM1) to generate dATP/dCTP/dGTP/dTTP; "
            "  p53R2 is induced by DNA damage via p53 → crucial for post-replicative mtDNA maintenance; "
            "  RRM2 (canonical S-phase subunit) supplies dNTPs during cell division; "
            "  p53R2 supplants RRM2 in non-dividing/post-mitotic cells; "
            "AR (biallelic) MUTATIONS → MDDS (mtDNA depletion syndrome type 8): "
            "  severe multi-system: neonatal/infantile encephalomyopathy + renal tubular acidosis + "
            "  liver failure + cardiomyopathy; very severe phenotype; early lethality; "
            "AD (heterozygous) MUTATIONS → PEO (progressive external ophthalmoplegia): "
            "  adult-onset; ocular + skeletal muscle; mtDNA multiple deletions (not depletion); "
            "  slowly progressive; LOF dominant-negative haploinsufficiency mechanism"
        ),
        "disease_category": (
            "RRM2B-RELATED MTDNA DEPLETION/DELETION SYNDROME (OMIM 612075): "
            "AR FORM (biallelic LOF) — MDDS Type 8: "
            "  ONSET: neonatal/infantile (1-12 months); "
            "  MUSCLE: severe hypotonia, myopathy, respiratory failure; "
            "  ENCEPHALOPATHY: global developmental delay, regression; seizures; "
            "  KIDNEY: renal tubular acidosis (Fanconi syndrome); "
            "  LIVER: transaminitis, hepatomegaly, liver failure in severe cases; "
            "  CARDIOMYOPATHY (hypertrophic or dilated); "
            "  GASTROINTESTINAL: feeding difficulties, failure to thrive; "
            "  LACTIC ACIDOSIS: blood lactate >5 mmol/L; "
            "  PROGNOSIS: severe; many die in first 5 years; "
            "AD FORM (heterozygous): "
            "  ONSET: 2nd-5th decade; "
            "  CPEO (chronic progressive external ophthalmoplegia) + ptosis; "
            "  EXERCISE INTOLERANCE; proximal limb weakness (variable); "
            "  HEARING LOSS (sensorineural, variable); "
            "  SLOW PROGRESSION; near-normal lifespan in most"
        ),
        "disease_pathway": (
            "MITOCHONDRIAL dNTP POOL MAINTENANCE — p53R2 (RRM2B): "
            "Ribonucleotide reductase (RNR) reaction: "
            "  NDP (ADP/CDP/GDP/UDP) → dNDP → dNTP (substrate for DNA replication/repair); "
            "  R1 subunit (RRM1) = large catalytic subunit (always expressed); "
            "  R2 subunit (RRM2) = small subunit, S-phase specific → dNTPs for nuclear DNA replication; "
            "  p53R2 (RRM2B) = small subunit, p53-inducible → dNTPs for MTDNA REPAIR in G0/G1 cells; "
            "RRM2B DEFICIENCY CONSEQUENCES: "
            "  post-mitotic cells (neurons, muscle) cannot maintain mtDNA copy number; "
            "  mtDNA depletion (AR) or multiple deletions (AD): "
            "    DEPLETION: <30% normal mtDNA copies → severe mitochondrial dysfunction; "
            "    DELETIONS: single large or multiple small deletions → partial mtDNA loss → milder; "
            "PYRIMIDINE CONNECTION: "
            "  p53R2 produces dCTP and dTTP (pyrimidine dNTPs) for mtDNA; "
            "  deficiency → dCTP + dTTP shortage in mitochondrial matrix; "
            "  imbalanced dNTP pool → mtDNA replication errors → depletion/deletions"
        ),
        "pathognomonic": (
            "MTDNA DEPLETION <30% ON MUSCLE/LIVER BIOPSY (AR form): PATHOGNOMONIC for MDDS; "
            "RAGGED-RED FIBRES + COX-NEGATIVE FIBRES ON MUSCLE BIOPSY: mitochondrial myopathy evidence; "
            "LACTIC ACIDOSIS (blood lactate >5 mmol/L) + EARLY MULTI-ORGAN INVOLVEMENT: "
            "  AR form: kidney + liver + heart + brain simultaneously → mtDNA depletion syndrome; "
            "MTDNA MULTIPLE DELETIONS ON SOUTHERN BLOT (AD form): "
            "  adult PEO + multiple mtDNA deletions → RRM2B (or POLG, C10orf2, POLG2); "
            "RESPIRATORY CHAIN COMPLEXES: COX (Complex IV) ± Combined Complex I+IV deficiency; "
            "p53R2 PROTEIN ABSENT ON WESTERN BLOT: confirmatory for AR biallelic null; "
            "DISTINGUISH FROM POLG PEO: POLG causes PEO + peripheral neuropathy + parkinsonism; "
            "  RRM2B AD: less neuropathy; similar ophthalmoplegia; Southern blot pattern similar"
        ),
        "treatment": (
            "AR FORM (MDDS8 — severe): "
            "1. SUPPORTIVE INTENSIVE CARE: respiratory support (mechanical ventilation for respiratory failure); "
            "2. COENZYME Q10: 10-30 mg/kg/day; limited evidence; antioxidant + electron carrier support; "
            "3. RIBOFLAVIN + B VITAMINS: cofactor supplementation; "
            "4. RENAL TUBULAR ACIDOSIS: bicarbonate supplementation; potassium replacement; "
            "5. LIVER TRANSPLANT: considered if isolated liver failure — but multi-organ disease limits benefit; "
            "6. LIVER TRANSPLANT (limited role): does NOT correct muscle/brain mtDNA depletion; "
            "AD FORM (PEO — milder): "
            "7. PTOSIS SURGERY: frontalis suspension or levator resection for functional ptosis; "
            "8. STRABISMUS SURGERY: restricted; beware anaesthesia (mitochondrial sensitivity); "
            "9. EXERCISE PROGRAMME: aerobic conditioning; avoid high-intensity → lactic acidosis; "
            "10. AVOID: valproate (mitochondrial toxicity — POLG overlap), statins (CoQ10 depletion), "
            "    alcohol, aminoglycosides; "
            "11. ANAESTHESIA: avoid succinylcholine; use regional; monitor for malignant hyperthermia-like response"
        ),
    },
    {
        "gene": "TK2",
        "seed_base": 2701,
        "protein": (
            "TK2 -- 16q21 AR -- 234aa -- Thymidine-Kinase-2-26kDa-Homodimer-"
            "Mitochondrial-Pyrimidine-Nucleoside-Salvage-dThd+dCyd-Phosphorylation-mtDNA-dTTP-dCTP-"
            "OMIM-Gene-188250-Disease-MDDS4A-B-C-610708"
        ),
        "locus": "16q21",
        "protein_size": "234 aa / 26 kDa (homodimer; mitochondrial matrix; pyrimidine nucleoside kinase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in TK2; "
            "TK2 encodes mitochondrial thymidine kinase 2, which phosphorylates: "
            "  thymidine (dThd) → dTMP (→ dTTP) in mitochondrial pyrimidine salvage; "
            "  deoxycytidine (dCyd) → dCMP (→ dCTP) in mitochondrial pyrimidine salvage; "
            "  EXCLUSIVELY for mitochondrial dNTP pool (distinct from cytoplasmic TK1); "
            "TK2 LOF → severe dTTP + dCTP deficiency in mitochondrial matrix → "
            "  mtDNA cannot be replicated/repaired → mtDNA depletion (myopathic MDDS); "
            "DISEASE: TK2 DEFICIENCY (MTDNA DEPLETION SYNDROME TYPE 4, MDDS4) OMIM 610708; "
            "MUTATIONS: "
            "  p.His90Asn (recurrent, >30% alleles; disrupts kinase active site); "
            "  p.Ile212Asn; p.Arg161Cys; p.Thr77Met; "
            "  homozygous H90N: 5% residual TK2 activity; severe neonatal/infantile onset; "
            "PHENOTYPE SPECTRUM: "
            "  SEVERE INFANTILE: onset 1-12 months; profound hypotonia; respiratory failure; early death; "
            "  CHILDHOOD: onset 2-10 years; progressive limb-girdle weakness; most common; "
            "  JUVENILE/ADULT: onset teens-40s; CPEO + limb weakness; milder course; "
            "  MNGIE-LIKE: rare GI + neuropathy overlap"
        ),
        "disease_category": (
            "TK2 DEFICIENCY — MTDNA DEPLETION SYNDROME TYPE 4 (MDDS4, OMIM 610708): "
            "PRIMARY PHENOTYPE: PROGRESSIVE MYOPATHY (dominant clinical feature): "
            "  PROXIMAL > DISTAL weakness; "
            "  RESPIRATORY FAILURE: earliest life-threatening complication; "
            "    diaphragm weakness → FVC decline → ventilatory failure; "
            "    CPAP/BiPAP → tracheostomy as disease progresses; "
            "  FACIAL WEAKNESS: ptosis; ophthalmoplegia (in childhood-adult forms); "
            "  LIMB GIRDLE WEAKNESS: shoulder + hip girdle; "
            "  DYSPHAGIA (late); "
            "SERUM CREATINE KINASE: typically ELEVATED (200-2000 U/L); "
            "SERUM LACTATE: elevated (mitochondrial myopathy); "
            "EMG: myopathic (short duration, polyphasic potentials); "
            "MUSCLE BIOPSY: ragged red fibres; COX-negative fibres; electron microscopy — "
            "  mitochondrial proliferation, paracrystalline inclusions, abnormal morphology; "
            "MTDNA COPY NUMBER IN MUSCLE: typically 10-30% normal (severe depletion); "
            "RESPIRATORY CHAIN: complex IV (COX) deficiency ± combined I+IV; "
            "CNS: generally SPARED (unlike AR RRM2B form) — distinguishes TK2 from encephalomyopathic MDDS"
        ),
        "disease_pathway": (
            "MITOCHONDRIAL PYRIMIDINE SALVAGE — TK2: "
            "Pyrimidine salvage in mitochondria (distinct from de novo synthesis in cytoplasm): "
            "  dThd (cytoplasmic) → crosses inner mitochondrial membrane → "
            "    TK2 phosphorylates: dThd → dTMP → dTDP → dTTP (for mtDNA); "
            "  dCyd (cytoplasmic) → crosses mitochondrial membrane → "
            "    TK2 phosphorylates: dCyd → dCMP → dCDP → dCTP (for mtDNA); "
            "PARALLEL CYTOPLASMIC PATHWAY: "
            "  TK1 (cytoplasmic) phosphorylates dThd → dTMP → nuclear DNA dTTP; "
            "  TK1 is S-phase regulated; TK2 is constitutively expressed; "
            "TK2 DEFICIENCY CONSEQUENCES: "
            "  mitochondrial dTTP + dCTP severely depleted; "
            "  imbalanced mitochondrial dNTP pool → mtDNA replication stalls → depletion; "
            "  skeletal muscle particularly vulnerable (high mtDNA demand, post-mitotic); "
            "DEOXYNUCLEOSIDE THERAPY RATIONALE: "
            "  exogenous dThd + dCyd provided orally → bypass TK2 → "
            "  cells use downstream kinases (dTMP kinase, CMPK1) to reach dTTP/dCTP; "
            "  PRECLINICAL (Tk2-/- mouse): dramatic reversal of myopathy with dThd+dCyd treatment; "
            "  HUMAN TRIALS: EAP (expanded access) + Phase 1/2 — significant clinical improvement"
        ),
        "pathognomonic": (
            "PROGRESSIVE MYOPATHY + RESPIRATORY FAILURE IN CHILD WITH ELEVATED CK: "
            "  myopathic EMG + COX-NEGATIVE fibres → mitochondrial myopathy; "
            "  TK2 deficiency in differential alongside POLG, RRM2B, TWNK, TYMP; "
            "MUSCLE MTDNA COPY NUMBER <30% (by qPCR): MDDS confirmed; "
            "  TK2 is MOST COMMON CAUSE of MYOPATHIC MDDS (vs encephalomyopathic for others); "
            "RAGGED RED FIBRES + COX-NEGATIVE FIBRES ON MUSCLE BIOPSY: "
            "  hallmarks of mitochondrial myopathy; classic pattern in TK2; "
            "CNS SPARING: BRAIN MRI NORMAL — distinguishes TK2 myopathic form from POLG/RRM2B "
            "  encephalomyopathy (where T2 changes / cortical atrophy); "
            "TK2 p.His90Asn MUTATION: recurrent; >30% of pathogenic alleles; if homozygous → severe infant form; "
            "DEOXYNUCLEOSIDE THERAPY RESPONSE: clinical improvement in strength + FVC → "
            "  strongly suggests TK2 (pharmacogenomically specific response)"
        ),
        "treatment": (
            "1. DEOXYNUCLEOSIDE THERAPY (dThd + dCyd): DISEASE-MODIFYING TREATMENT; "
            "   dThd 200 mg/kg/day + dCyd 200 mg/kg/day divided TID; "
            "   oral administration; bypasses TK2 block using downstream phosphorylation; "
            "   EVIDENCE: EAP patients show FVC stabilisation + strength improvement; regression of disability; "
            "   ACCELERATED APPROVAL: FDA approval in progress (orphan drug); "
            "   EARLY TREATMENT: initiate before severe muscle loss for best outcome; "
            "2. RESPIRATORY SUPPORT: NIV (BiPAP) early; plan tracheostomy when FVC <50%; "
            "   RESPIRATORY ASSESSMENT Q3-6 MONTHS (FVC, NIF, ABG); "
            "3. PHYSIOTHERAPY: respiratory physiotherapy; limb physiotherapy; "
            "4. GASTROSTOMY: if dysphagia / nutritional compromise; "
            "5. COENZYME Q10 + RIBOFLAVIN: cofactor support (limited evidence); "
            "6. AVOID: statin myopathy risk (compound muscle toxicity); valproate (mitochondrial toxicity); "
            "   nucleoside analogues (AZT, d4T) — compete with TK2 substrates → worsen dNTP imbalance; "
            "7. ANAESTHESIA: avoid suxamethonium; volatile agents cautiously; regional preferred; "
            "8. SURVEILLANCE: FVC monthly in rapid decline phase; CK, lactate; muscle MRI for tracking"
        ),
    },
]


def _generate_patients(gene_idx, n=40, seed=2694):
    """Generate deterministic patient cohort for a pyrimidine disorder gene."""
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]
    g = gene["gene"]
    patients = []

    for i in range(n):
        age_dx = rng.uniform(0.1, 45) if g in ("TYMP", "RRM2B", "TK2") else rng.uniform(0.1, 25)
        age_onset = age_dx - rng.uniform(0, 5)
        age_onset = max(0.05, age_onset)

        # Gene-specific phenotype parameters
        if g == "TYMP":
            gi_dysmot = rng.random() < 0.95
            cachexia = rng.random() < 0.88
            leukoenceph = rng.random() < 0.78
            neuropathy = rng.random() < 0.85
            cpeo = rng.random() < 0.72
            plasma_dthd = round(rng.uniform(4, 120), 1)
            lactic_acid = rng.random() < 0.55
            hsct_candidate = rng.random() < 0.40

        elif g == "DPYD":
            gi_dysmot = False
            cachexia = rng.random() < 0.12
            leukoenceph = rng.random() < 0.08
            neuropathy = rng.random() < 0.10
            cpeo = False
            plasma_dthd = 0.0
            lactic_acid = False
            seizures = rng.random() < 0.72
            id_severe = rng.random() < 0.55
            fivefu_toxicity = rng.random() < 0.82
            hsct_candidate = False

        elif g == "DPYS":
            gi_dysmot = rng.random() < 0.35
            cachexia = False
            leukoenceph = False
            neuropathy = False
            cpeo = False
            plasma_dthd = 0.0
            lactic_acid = False
            seizures = rng.random() < 0.58
            id_severe = rng.random() < 0.42
            fivefu_toxicity = False
            hsct_candidate = False
            asymptomatic = rng.random() < 0.28

        elif g == "UPB1":
            gi_dysmot = rng.random() < 0.22
            cachexia = False
            leukoenceph = False
            neuropathy = False
            cpeo = False
            plasma_dthd = 0.0
            lactic_acid = False
            seizures = rng.random() < 0.78
            id_severe = rng.random() < 0.65
            fivefu_toxicity = False
            hsct_candidate = False
            balalanine_resp = rng.random() < 0.52

        elif g == "CAD":
            gi_dysmot = False
            cachexia = False
            leukoenceph = False
            neuropathy = False
            cpeo = False
            plasma_dthd = 0.0
            lactic_acid = False
            seizures = rng.random() < 0.92
            id_severe = rng.random() < 0.45  # with treatment, improves
            fivefu_toxicity = False
            hsct_candidate = False
            uridine_response = rng.random() < 0.95
            anaemia = rng.random() < 0.80

        elif g == "DHODH":
            gi_dysmot = False
            cachexia = False
            leukoenceph = False
            neuropathy = False
            cpeo = False
            plasma_dthd = 0.0
            lactic_acid = False
            seizures = rng.random() < 0.05
            id_severe = rng.random() < 0.04
            fivefu_toxicity = False
            hsct_candidate = False
            postaxial_limb = rng.random() < 0.98
            craniofacial = rng.random() < 0.95

        elif g == "RRM2B":
            gi_dysmot = rng.random() < 0.25
            cachexia = False
            leukoenceph = rng.random() < 0.35
            neuropathy = rng.random() < 0.45
            cpeo = rng.random() < 0.65
            plasma_dthd = 0.0
            lactic_acid = rng.random() < 0.60
            seizures = rng.random() < 0.30
            id_severe = rng.random() < 0.28
            fivefu_toxicity = False
            hsct_candidate = False

        elif g == "TK2":
            gi_dysmot = rng.random() < 0.15
            cachexia = rng.random() < 0.45
            leukoenceph = rng.random() < 0.08
            neuropathy = rng.random() < 0.20
            cpeo = rng.random() < 0.40
            plasma_dthd = 0.0
            lactic_acid = rng.random() < 0.70
            seizures = rng.random() < 0.08
            id_severe = rng.random() < 0.05
            fivefu_toxicity = False
            hsct_candidate = False
            resp_failure = rng.random() < 0.75
            dnucleoside_rx = rng.random() < 0.55

        pt = {
            "patient_id": f"{g}-{seed + i:04d}",
            "age_onset_years": round(age_onset, 1),
            "age_dx_years": round(age_dx, 1),
            "gi_dysmotility": gi_dysmot,
            "cachexia": cachexia,
            "leukoencephalopathy": leukoenceph,
            "peripheral_neuropathy": neuropathy,
            "cpeo_ptosis": cpeo,
            "lactic_acidosis": lactic_acid,
            "seizures": locals().get("seizures", False),
            "id_moderate_severe": locals().get("id_severe", False),
            "resp_failure": locals().get("resp_failure", False),
            "dnucleoside_treatment": locals().get("dnucleoside_rx", False),
            "uridine_response": locals().get("uridine_response", False),
            "fivefu_toxicity": locals().get("fivefu_toxicity", False),
            "postaxial_limb": locals().get("postaxial_limb", False),
            "craniofacial": locals().get("craniofacial", False),
            "hsct_done": hsct_candidate if g == "TYMP" else False,
            "plasma_dthd_umolL": plasma_dthd if g == "TYMP" else None,
        }
        patients.append(pt)

    return patients


def generate_overview():
    all_patients = []
    gene_summaries = []
    seeds = list(range(2694, 2702))

    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=seeds[idx])
        all_patients.extend(patients)
        n = len(patients)

        gi_n = sum(1 for p in patients if p["gi_dysmotility"])
        cachex_n = sum(1 for p in patients if p["cachexia"])
        leuko_n = sum(1 for p in patients if p["leukoencephalopathy"])
        neuro_n = sum(1 for p in patients if p["peripheral_neuropathy"])
        cpeo_n = sum(1 for p in patients if p["cpeo_ptosis"])
        lactic_n = sum(1 for p in patients if p["lactic_acidosis"])
        seiz_n = sum(1 for p in patients if p["seizures"])
        id_n = sum(1 for p in patients if p["id_moderate_severe"])
        resp_n = sum(1 for p in patients if p["resp_failure"])
        fivefu_n = sum(1 for p in patients if p["fivefu_toxicity"])
        mean_onset = round(sum(p["age_onset_years"] for p in patients) / n, 1)

        gene_summaries.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "mean_onset_years": mean_onset,
            "gi_dysmotility_pct": round(100 * gi_n / n, 1),
            "cachexia_pct": round(100 * cachex_n / n, 1),
            "leukoenceph_pct": round(100 * leuko_n / n, 1),
            "neuropathy_pct": round(100 * neuro_n / n, 1),
            "cpeo_ptosis_pct": round(100 * cpeo_n / n, 1),
            "lactic_acidosis_pct": round(100 * lactic_n / n, 1),
            "seizures_pct": round(100 * seiz_n / n, 1),
            "id_pct": round(100 * id_n / n, 1),
            "resp_failure_pct": round(100 * resp_n / n, 1),
            "fivefu_toxicity_pct": round(100 * fivefu_n / n, 1),
        })

    pathway_categories = [
        "MNGIE-THYMIDINE-PHOSPHORYLASE-DEFICIENCY — TYMP: thymidine+deoxyuridine accumulate → mtDNA depletion/deletions → GI dysmotility+cachexia+leukoencephalopathy+neuropathy+CPEO pentad",
        "DPD-DEFICIENCY — DPYD: uracil+thymine accumulate in pyrimidine catabolism (rate-limiting step) → seizures+ID in complete deficiency; 5-FU pharmacogenomics in partial — CPIC GRADE A",
        "DIHYDROPYRIMIDINASE-DEFICIENCY — DPYS: dihydrouracil+dihydrothymine accumulate (step 2 block) → variable seizures+ID; asymptomatic subset; NO 5-FU risk",
        "BETA-UREIDOPROPIONASE-DEFICIENCY — UPB1: ureidopropionic acid accumulates + beta-alanine DEFICIENT (step 3 block) → seizures+ID; beta-alanine supplementation may help",
        "CAD-MULTIFUNCTIONAL-ENZYME-DEFICIENCY — CAD: de novo pyrimidine biosynthesis block (steps 1-3) → uridine deficiency → epileptic encephalopathy + megaloblastic anaemia; URIDINE SUPPLEMENTATION CURATIVE",
        "MILLER-SYNDROME — DHODH: pyrimidine de novo step 4 block embryonically → postaxial limb + craniofacial defects; NO postnatal metabolic disease; LEFLUNOMIDE ABSOLUTELY CONTRAINDICATED",
        "RRM2B-MTDNA-DEPLETION — RRM2B: p53R2 deficiency → mitochondrial dNTP pool imbalance → mtDNA depletion (AR severe multi-system) or deletions (AD late-onset PEO)",
        "TK2-MYOPATHIC-MDDS — TK2: mitochondrial thymidine kinase deficiency → dTTP+dCTP pool depletion in muscle → myopathic MDDS; DEOXYNUCLEOSIDE THERAPY (dThd+dCyd) disease-modifying",
    ]

    critical_distinctions = [
        "TYMP vs POLG vs TK2: all cause mtDNA pathology; TYMP = GI dysmotility PENTAD + plasma dThd/dUrd elevated; POLG = encephalopathy + liver + neuropathy; TK2 = pure MYOPATHY + brain SPARED",
        "DPYD vs DPYS vs UPB1: DPYD = uracil+thymine elevated (early catabolism); DPYS = dihydrouracil+dihydrothymine elevated (step 2); UPB1 = ureidopropionic acid + beta-alanine ABSENT (step 3)",
        "CAD vs UMPS: both de novo pyrimidine; CAD urine orotic acid LOW/NORMAL (substrate starvation); UMPS urine orotic acid VERY HIGH PATHOGNOMONIC; both respond to uridine",
        "URIDINE SUPPLEMENTATION: CURATIVE in CAD deficiency; USEFUL in UMPS; NOT indicated in DPYD/DPYS/UPB1 (catabolism disorders where uracil accumulates — not deficient)",
        "DHODH Miller Syndrome: ONLY pyrimidine disorder with structural birth defects + NO postnatal metabolic disease; postnatal uridine/orotate normal; LEFLUNOMIDE (DHODH inhibitor) ABSOLUTELY CONTRAINDICATED",
        "CPIC DPYD GRADE A: heterozygous *2A (c.1905+1G>A) → 5-FU/capecitabine 50% dose reduction; homozygous → AVOID fluoropyrimidines; PLASMA URACIL >16 ng/mL = phenotypic DPD deficiency screening",
        "MNGIE HSCT: ONLY curative treatment; dThd/dUrd normalise post-HSCT; GI may improve 12-24 months; best outcomes before irreversible intestinal smooth muscle fibrosis",
        "TK2 DEOXYNUCLEOSIDE THERAPY: dThd+dCyd 200 mg/kg/day EACH — bypasses TK2 block via downstream phosphorylation; disease-modifying evidence; AVOID in TYMP (dThd worsens MNGIE) — opposite effect",
    ]

    return {
        "atlas": "Hereditary Pyrimidine Disorder Atlas",
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
        patients = _generate_patients(idx, n=40, seed=2694 + idx)
        n = len(patients)

        gi_n = sum(1 for p in patients if p["gi_dysmotility"])
        cachex_n = sum(1 for p in patients if p["cachexia"])
        leuko_n = sum(1 for p in patients if p["leukoencephalopathy"])
        neuro_n = sum(1 for p in patients if p["peripheral_neuropathy"])
        cpeo_n = sum(1 for p in patients if p["cpeo_ptosis"])
        lactic_n = sum(1 for p in patients if p["lactic_acidosis"])
        seiz_n = sum(1 for p in patients if p["seizures"])
        id_n = sum(1 for p in patients if p["id_moderate_severe"])
        resp_n = sum(1 for p in patients if p["resp_failure"])
        fivefu_n = sum(1 for p in patients if p["fivefu_toxicity"])
        postaxial_n = sum(1 for p in patients if p["postaxial_limb"])
        uridine_n = sum(1 for p in patients if p["uridine_response"])
        dnucl_n = sum(1 for p in patients if p["dnucleoside_treatment"])

        breakdown.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"][:350],
            "disease_category": gene_data["disease_category"][:400],
            "pathognomonic": gene_data["pathognomonic"][:400],
            "treatment_summary": gene_data["treatment"][:400],
            "patients_n": n,
            "gi_dysmotility_pct": round(100 * gi_n / n, 1),
            "cachexia_pct": round(100 * cachex_n / n, 1),
            "leukoenceph_pct": round(100 * leuko_n / n, 1),
            "neuropathy_pct": round(100 * neuro_n / n, 1),
            "cpeo_ptosis_pct": round(100 * cpeo_n / n, 1),
            "lactic_acidosis_pct": round(100 * lactic_n / n, 1),
            "seizures_pct": round(100 * seiz_n / n, 1),
            "id_pct": round(100 * id_n / n, 1),
            "resp_failure_pct": round(100 * resp_n / n, 1),
            "fivefu_toxicity_pct": round(100 * fivefu_n / n, 1),
            "postaxial_limb_pct": round(100 * postaxial_n / n, 1),
            "uridine_response_pct": round(100 * uridine_n / n, 1),
            "dnucleoside_treatment_pct": round(100 * dnucl_n / n, 1),
            "sample_patients": patients[:5],
        })

    return {"atlas": "Hereditary Pyrimidine Disorder Atlas", "breakdown": breakdown}


def generate_definitions():
    glossary = {
        "MNGIE (Mitochondrial NeuroGastroIntestinal Encephalomyopathy)": (
            "AR TYMP deficiency → thymidine phosphorylase absent → plasma dThd >3 µmol/L + dUrd >5 µmol/L; "
            "pentad: GI dysmotility + cachexia + peripheral neuropathy + leukoencephalopathy + CPEO/ptosis; "
            "pathomechanism: dThd/dUrd → imbalanced mitochondrial dNTP pool → mtDNA depletion/deletions; "
            "TREATMENT: HSCT only curative (restores TP activity → normalises metabolites); "
            "AVOID: thymidine analogues (AZT, d4T) — worsen dNTP imbalance"
        ),
        "Dihydropyrimidine Dehydrogenase (DPD/DPYD) Deficiency": (
            "AR complete deficiency: seizures + ID + autism features; urine uracil >80 µmol/mmol Cr; "
            "PHARMACOGENOMICS: DPYD *2A (c.1905+1G>A) heterozygous → 5-FU/capecitabine severe toxicity; "
            "CPIC Grade A: *2A heterozygous → 50% dose reduction; homozygous → AVOID 5-FU; "
            "ANTIDOTE: uridine triacetate (Vistogard) for 5-FU overdose (within 96h); "
            "PLASMA URACIL >16 ng/mL: phenotypic screening marker for reduced DPD activity"
        ),
        "DPD vs DHP Deficiency (DPYD vs DPYS)": (
            "DPYD (DPD, step 1): uracil + thymine accumulate; early catabolism block; "
            "  5-FU PHARMACOGENOMICS RISK; "
            "DPYS (DHP, step 2): dihydrouracil + dihydrothymine accumulate; mid-catabolism block; "
            "  NO 5-FU pharmacogenomics risk (DPD intact); "
            "  asymptomatic subset; variable penetrance; "
            "CRITICAL: urine OA profile distinguishes (GC-MS peak retention time differs)"
        ),
        "Beta-Ureidopropionase (UPB1) Deficiency": (
            "AR, step 3 pyrimidine catabolism; ureidopropionic acid accumulates; "
            "beta-alanine DEFICIENT (downstream product not produced); "
            "seizures + ID + hypotonia from neonatal period; "
            "TREATMENT: beta-alanine supplementation (100-200 mg/kg/day) — case reports of seizure improvement; "
            "MECHANISM: beta-alanine is GABA-A partial agonist + GABA-T inhibitor → deficiency → reduced GABAergic tone; "
            "DISTINGUISH FROM DPYS: UPB1 has ureidopropionic acid elevated (not dihydrouracil)"
        ),
        "CAD Deficiency": (
            "AR deficiency of trifunctional enzyme (CPS2 + ATCase + DHOase = steps 1-3 of de novo pyrimidine synthesis); "
            "onset first 3-6 months; epileptic encephalopathy + megaloblastic anaemia + hypersegmented neutrophils; "
            "URIDINE SUPPLEMENTATION: immediate, dramatic seizure cessation (days); PATHOGNOMONIC RESPONSE; "
            "urine orotic acid LOW/NORMAL (substrate starvation — CONTRAST UMPS with HIGH orotic acid); "
            "plasma dihydroorotate ELEVATED; "
            "lifelong uridine required; triacetyl uridine (TAU) preferred (3x better bioavailability)"
        ),
        "Miller Syndrome (DHODH Deficiency)": (
            "AR DHODH deficiency → transient pyrimidine starvation during embryogenesis (6-10 weeks); "
            "POSTAXIAL limb defects: 4th+5th digit/ray aplasia; forearm/fibula hypoplasia; "
            "craniofacial: malar hypoplasia + lower eyelid coloboma + micrognathia + cleft palate + downslanting PF; "
            "NORMAL INTELLIGENCE: postnatal pyrimidine supply from salvage is sufficient; "
            "LEFLUNOMIDE/TERIFLUNOMIDE: ABSOLUTELY CONTRAINDICATED (same DHODH target); "
            "DDx: Nager (SF3B4, preaxial/radial defects); Treacher Collins (TCOF1, no limbs)"
        ),
        "RRM2B-Related mtDNA Depletion Syndrome (MDDS8)": (
            "p53R2 (RRM2B) provides dNTPs for mtDNA REPAIR in non-dividing cells; "
            "AR biallelic: severe neonatal MDDS — multi-organ (muscle+kidney+liver+heart+brain); early lethality; "
            "AD heterozygous: adult-onset PEO + mtDNA multiple DELETIONS; slowly progressive; near-normal lifespan; "
            "distinguish AR from AD by age of onset + severity + mtDNA depletion (AR) vs deletions (AD); "
            "AVOID: valproate (mitochondrial toxicity); succinylcholine anaesthesia"
        ),
        "TK2 Deficiency (Myopathic MDDS4)": (
            "AR mitochondrial thymidine kinase 2 deficiency → dTTP+dCTP shortage in mitochondria → mtDNA depletion in muscle; "
            "MYOPATHIC PHENOTYPE: progressive proximal weakness + respiratory failure (diaphragm); CNS SPARED; "
            "p.His90Asn recurrent mutation (>30% alleles); "
            "DEOXYNUCLEOSIDE THERAPY: dThd 200 mg/kg/day + dCyd 200 mg/kg/day → bypasses TK2 → disease-modifying; "
            "CONTRAST TYMP (MNGIE): TK2 benefits FROM dThd; TYMP is HARMED by dThd (opposite mechanisms)"
        ),
        "Mitochondrial DNA Depletion Syndrome (MDDS)": (
            "Group of AR disorders with mtDNA copy number <30% normal in affected tissue; "
            "caused by defects in: mtDNA replication (POLG, TWNK, POLG2); "
            "  mitochondrial dNTP supply (TK2, RRM2B, SUCLA2, SUCLG1, DGUOK); "
            "  maintenance factors; "
            "TYMP causes mtDNA depletion/DELETIONS (dNTP pool imbalance — different from depletion per se); "
            "TISSUE-SPECIFIC: TK2 → muscle; DGUOK → liver; SUCLA2 → brain+muscle; RRM2B → multi-system"
        ),
        "Pyrimidine Catabolism Pathway": (
            "3-enzyme pyrimidine degradation: "
            "Uracil →[DPD/DPYD]→ Dihydrouracil →[DPYS/DHP]→ Ureidopropionic acid →[UPB1]→ Beta-alanine + CO2 + NH3; "
            "Thymine →[DPD]→ Dihydrothymine →[DPYS]→ Ureidoisobutyric acid →[UPB1]→ Beta-AIBA + CO2 + NH3; "
            "Diagnosis: each enzyme deficiency produces specific accumulating substrate; "
            "urine OA panel (GC-MS) identifies block location"
        ),
        "CPIC DPYD Pharmacogenomics": (
            "CPIC Gene-Drug guideline for DPYD and fluoropyrimidines (5-FU, capecitabine, tegafur); "
            "Activity Score (AS) system: *2A=0, *13=0 (null); c.2846A>T=0.5; HapB3=0.5; "
            "AS=0 (null): AVOID 5-FU/capecitabine — use alternative (raltitrexed, irinotecan); "
            "AS=0.5: 50% dose reduction + TDM; "
            "AS=1.5: 25-50% dose reduction; "
            "EMSO/DPYD consortium: mandatory pre-treatment DPYD testing in Europe; "
            "PLASMA URACIL TEST (phenotypic): uracil >16 ng/mL suggests low DPD → dose modify"
        ),
        "De Novo Pyrimidine Biosynthesis (CAD → DHODH → UMPS)": (
            "6-step pathway producing UMP from glutamine + HCO3- + aspartate: "
            "Steps 1-3 (CAD, cytoplasmic): carbamoyl phosphate → N-carbamoyl-aspartate → dihydroorotate; "
            "Step 4 (DHODH, inner mitochondrial membrane): dihydroorotate → orotate; "
            "Steps 5-6 (UMPS, cytoplasmic): orotate → OMP → UMP; "
            "UMP → UDP → UTP → CTP → RNA + dTMP synthesis; "
            "EMBRYONIC DEPENDENCE: de novo essential for rapid proliferation; "
            "POSTNATAL: salvage sufficient for maintenance (except in UMPS/CAD deficiency)"
        ),
    }

    return {
        "atlas": "Hereditary Pyrimidine Disorder Atlas",
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
