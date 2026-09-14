"""Hereditary Bile Acid Synthesis & Sterol Biosynthesis Atlas — 8-Gene Reference
DHCR7-CYP27A1-HSD3B7-AKR1D1-CYP7B1-AMACR-SC5D-EBP
320 patients (8 x 40), seeds 2678-2685.
Endpoints: /api/hereditary-bile-acid-sterol-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "DHCR7",
        "protein": (
            "DHCR7 -- 11q13.4 AR -- 475aa -- 7-Dehydrocholesterol-Reductase-54kDa-9TM-ER-Membrane-Enzyme-"
            "Catalyses-Final-Step-Cholesterol-Synthesis-7-DHC-to-Cholesterol-FAD-Dependent-"
            "OMIM-Gene-602858-Disease-SLO-270400"
        ),
        "locus": "11q13.4",
        "protein_size": "475 aa / 54 kDa (9-transmembrane ER membrane enzyme)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "DHCR7 encodes the final enzyme in post-squalene cholesterol biosynthesis: "
            "  7-dehydrocholesterol (7-DHC) → cholesterol (Δ7 double bond reduction); "
            "DHCR7 loss → 7-DHC accumulates (toxic) + cholesterol deficient; "
            "7-DHC DUAL TOXICITY: (1) membrane fluidity disruption; "
            "  (2) photodegradation products (oxysterols) highly cytotoxic; "
            "  (3) 7-DHC:cholesterol ratio in cell membranes determines severity; "
            "MOST COMMON STEROL SYNTHESIS DEFECT: 1:15,000-1:30,000 live births (varies by population); "
            "MOST COMMON MUTATION: IVS8-1G>C (splice site) — ~25-30% of European alleles; "
            "  T93M (p.Thr93Met) — ~10% European alleles; "
            "  W151X (p.Trp151X) — severe null; "
            "CARRIER FREQUENCY: ~1:30 in European populations (high heterozygote frequency); "
            "GENOTYPE-PHENOTYPE: null/null → severe (SLOS type II, holoprosencephaly); "
            "  missense/missense or null/missense → variable (SLOS type I, classic); "
            "  IVS8-1G>C/missense → typical phenotype; "
            "PRENATAL: maternal low unconjugated oestriol on triple screen (7-DHC cannot convert to oestriol precursor); "
            "  elevated amniotic fluid 7-DHC + DHCR7 sequencing for prenatal diagnosis"
        ),
        "disease_category": (
            "SMITH-LEMLI-OPITZ SYNDROME (SLOS) — OMIM 270400; "
            "BIOCHEMICAL HALLMARK: elevated plasma 7-dehydrocholesterol (7-DHC) — PATHOGNOMONIC; "
            "  normal range: 7-DHC <5 µg/mL; SLOS: 7-DHC 10-1500 µg/mL; "
            "  7-DHC:cholesterol ratio >0.1 = diagnostic; "
            "CHOLESTEROL LOW: total cholesterol often <100 mg/dL in classic SLOS; "
            "CLINICAL FEATURES: "
            "  2nd-3rd toe syndactyly (Y-shaped skin bridge) — PATHOGNOMONIC, present in >97%; "
            "  anteverted nares + broad nasal bridge; "
            "  intellectual disability (mild to severe); "
            "  behavioural disorder: autism spectrum features in 60-70%; "
            "  self-injurious behaviour + light hypersensitivity (7-DHC photodegradation); "
            "  genital ambiguity in 46XY males (cholesterol required for testosterone synthesis); "
            "  microcephaly; ptosis; cataracts; "
            "  cleft palate; cardiac defects; "
            "SEVERE (TYPE II): holoprosencephaly, limb reduction, lethal; "
            "NBS: NOT detected by standard NBS; "
            "  expanded NBS: 7-DHC measurable but not universally implemented; "
            "PRENATAL CLUE: maternal low unconjugated oestriol (triple screen) — should trigger 7-DHC testing; "
            "DIAGNOSIS: plasma 7-DHC + DHCR7 sequencing"
        ),
        "disease_pathway": (
            "CHOLESTEROL BIOSYNTHESIS (POST-SQUALENE PATHWAY): "
            "Acetyl-CoA → (HMG-CoA reductase, statin target) → mevalonate → squalene → "
            "lanosterol → (multiple steps) → 7-dehydrocholesterol → (DHCR7) → CHOLESTEROL; "
            "CHOLESTEROL FUNCTIONS DISRUPTED IN SLOS: "
            "  1. Cell membrane structural component (membrane fluidity, lipid rafts, caveolae); "
            "  2. Myelin sheath component (CNS development); "
            "  3. Bile acid precursor (bile acid synthesis starts with cholesterol); "
            "  4. Steroid hormone precursor (cortisol, aldosterone, sex hormones, vitamin D3); "
            "  5. Hedgehog signalling ligand binding — Sonic Hedgehog (SHH) requires cholesterol; "
            "     SHH pathway disruption → brain and limb patterning defects (holoprosencephaly); "
            "7-DHC ACCUMULATION: "
            "  7-DHC is photosensitive → UV light → toxic oxysterol derivatives; "
            "  explains light-induced behavioural worsening; "
            "  explains skin photosensitivity; "
            "TREATMENT RATIONALE: "
            "  cholesterol supplementation (egg yolk, dietary) restores plasma cholesterol; "
            "  reduce UV light exposure (photoprotection); "
            "  avoid plant sterols (compete with cholesterol absorption); "
            "  statins: CONTRAINDICATED (further reduce already-deficient cholesterol synthesis); "
            "SIMVASTATIN/STATINS ABSOLUTE CI: worsen cholesterol depletion + increase 7-DHC"
        ),
        "pathognomonic": (
            "7-DEHYDROCHOLESTEROL ELEVATED PATHOGNOMONIC: "
            "7-DHC:cholesterol ratio >0.1 on plasma sterol profile = diagnostic; "
            "CLINICAL PATHOGNOMONIC: 2nd-3rd toe syndactyly in >97% — virtually universal; "
            "LOW MATERNAL UNCONJUGATED OESTRIOL on triple screen: "
            "  7-DHC cannot be converted to 16-hydroxy-DHEA (oestriol precursor requires cholesterol intermediates); "
            "  low uE3 (<0.5 MoM) → immediate reflex 7-DHC testing; "
            "AUTISM-LIGHT HYPERSENSITIVITY COMBINATION: in any child with ASD + photosensitivity → SLOS screen; "
            "MALE GENITAL AMBIGUITY + DYSMORPHIC FEATURES: testosterone requires cholesterol → "
            "46XY SLOS may have undervirilised genitalia; "
            "BEHAVIOUR: self-injurious behaviour worsened by UV light (7-DHC oxidation product toxicity) — "
            "specific to SLOS among sterol disorders"
        ),
        "treatment": (
            "CHOLESTEROL SUPPLEMENTATION: "
            "  dietary cholesterol: eggs (1 egg yolk = ~200 mg cholesterol), meat; "
            "  pharmaceutical: cholesterol powder 30-100 mg/kg/day in formula/feeds; "
            "  target: plasma cholesterol >100 mg/dL; "
            "  BENEFIT: improves growth, behaviour, reduces 7-DHC levels; "
            "  DOES NOT correct all features (prenatal developmental defects irreversible); "
            "PHOTOPROTECTION: "
            "  UVA/UVB sunscreen + protective clothing; "
            "  reduces 7-DHC photoproduct burden; "
            "  improves behaviour + reduces self-injury; "
            "STATINS: ABSOLUTE CONTRAINDICATION — inhibit HMG-CoA reductase → "
            "  further reduces cholesterol + increases 7-DHC accumulation; "
            "PLANT STEROLS: AVOID — phytosterols compete with cholesterol absorption; "
            "DIETARY: avoid plant sterol-enriched foods (margarine, fortified products); "
            "SIMETHICONE/ANTACIDS: sometimes used for GI symptoms; "
            "BEHAVIOURAL SUPPORTS: ABA + pharmacological (if severe self-injury); "
            "PRENATAL CHOLESTEROL: some evidence for maternal cholesterol supplementation in pregnancy; "
            "GENETIC COUNSELLING: AR; 25% recurrence risk per pregnancy; "
            "PREIMPLANTATION GENETIC TESTING: available"
        ),
    },
    {
        "gene": "CYP27A1",
        "protein": (
            "CYP27A1 -- 2q35 AR -- 531aa -- Sterol-27-Hydroxylase-CYP27A1-60kDa-Mitochondrial-Inner-Membrane-"
            "Cytochrome-P450-Initiates-Bile-Acid-Synthesis-Alternative-Pathway-27-Hydroxylation-"
            "Oxidises-27-Position-Cholesterol-to-27-Hydroxycholesterol-then-to-3-alpha-7-alpha-dihydroxy-5-beta-cholestanoic-acid-"
            "OMIM-Gene-606530-Disease-CTX-213700"
        ),
        "locus": "2q35",
        "protein_size": "531 aa / 60 kDa (mitochondrial inner membrane CYP450)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "CYP27A1 encodes sterol 27-hydroxylase: initiates the alternative (acidic) pathway of bile acid synthesis; "
            "MAIN REACTION: cholesterol → 27-hydroxycholesterol → 3α,7α-dihydroxy-5β-cholestanoic acid (DHCA); "
            "  also: 26-hydroxycholesterol (oxysterol) oxidation → downstream bile acid precursors; "
            "CYP27A1 LOSS: "
            "  alternative bile acid synthesis pathway blocked → cholestanol accumulates; "
            "  cholestanol (5α-dihydrocholesterol) is an abnormal sterol; "
            "  cholestanol deposits in tendons, brain white matter, lens; "
            "  primary bile acid production via alternative pathway impaired → "
            "  compensatory upregulation of 7-alpha-hydroxylase pathway; "
            "POPULATION GENETICS: "
            "  Moroccan Jewish founder: pGln403Arg (p.Q403R) — ~1:108 carrier frequency; "
            "  panethnic otherwise; incidence 1:50,000-70,000; "
            "DIAGNOSTIC DELAY: average 15-20 years from first symptom to diagnosis; "
            "TREATABLE IF CAUGHT EARLY: CDCA reverses neurological progression if started before neurodegeneration"
        ),
        "disease_category": (
            "CEREBROTENDINOUS XANTHOMATOSIS (CTX) — OMIM 213700; "
            "BIOCHEMICAL HALLMARK: elevated plasma cholestanol — PATHOGNOMONIC; "
            "  cholestanol:cholesterol ratio >0.01 (normal <0.003); "
            "  urine bile alcohols elevated (glucuronides of 5β-cholestane-3α,7α,12α-triol); "
            "CLINICAL TRIAD (classic): "
            "  1. Achilles tendon xanthomas (cholestanol deposits) — PATHOGNOMONIC; "
            "     may not appear until 2nd-3rd decade; "
            "  2. Cataracts (infantile-onset, often first sign in 1st decade); "
            "  3. Neurological deterioration (cognitive decline, cerebellar ataxia, spastic paraplegia, epilepsy); "
            "EARLY FEATURES (often missed): "
            "  infantile-onset diarrhoea + failure to thrive (bile acid deficiency); "
            "  cataracts in childhood (cholestanol lens deposits); "
            "BRAIN MRI: "
            "  bilateral dentate nucleus T2/FLAIR hyperintensities — PATHOGNOMONIC for CTX; "
            "  cerebellar + pyramidal + white matter lesions in advanced disease; "
            "PSYCHIATRIC: depression, psychosis in ~50% (often years before neurological diagnosis); "
            "SEIZURES: in 40-50% of patients with advanced neurological disease; "
            "NBS: NOT detected (cholestanol not measured on standard NBS); "
            "TREATMENT WINDOW: start CDCA before neurological disease onset for full benefit"
        ),
        "disease_pathway": (
            "BILE ACID SYNTHESIS — ALTERNATIVE (ACIDIC) PATHWAY: "
            "Cholesterol → (CYP27A1: 27-hydroxylase) → 27-hydroxycholesterol → (CYP7B1: 7α-hydroxylase) → "
            "→ primary bile acid precursors → chenodeoxycholic acid (CDCA) + cholic acid (CA); "
            "CLASSIC (NEUTRAL) PATHWAY: "
            "Cholesterol → (CYP7A1: 7α-hydroxylase, rate-limiting) → 7α-hydroxycholesterol → "
            "→ cholic acid (CA) + chenodeoxycholic acid (CDCA); "
            "CYP27A1 LOSS CONSEQUENCES: "
            "  1. Alternative pathway blocked → upstream 27-OH-cholesterol cannot proceed; "
            "  2. Cholesterol → cholestanol shunted (abnormal reduction pathway); "
            "  3. Cholestanol accumulates in: tendons (xanthomas), lens, brain white matter; "
            "  4. Bile acid deficiency → fat-soluble vitamin malabsorption; "
            "CDCA TREATMENT MECHANISM: "
            "  exogenous CDCA → activates FXR (nuclear receptor) → "
            "  suppresses CYP7A1 (rate-limiting classic pathway enzyme) → "
            "  reduces cholestanol synthesis (via 7-dehydrocholesterol shunt) → "
            "  plasma cholestanol normalises; "
            "  ALSO: CDCA partially restores bile acid pool; "
            "  REVERSES neurological progression if started before irreversible damage; "
            "MONITORING: plasma cholestanol (target normalisation) + MRI + neuropsychological testing"
        ),
        "pathognomonic": (
            "CHOLESTANOL ELEVATED PATHOGNOMONIC: "
            "plasma cholestanol >5 µg/mL (normal <2 µg/mL) + cholestanol:cholesterol ratio >0.01; "
            "ACHILLES TENDON XANTHOMAS PATHOGNOMONIC: "
            "  cholestanol deposits in tendon collagen; "
            "  imaging: MRI or ultrasound of Achilles tendons; "
            "  absence does NOT exclude CTX in early cases; "
            "DENTATE NUCLEUS T2-HYPERINTENSITY: bilateral symmetric — PATHOGNOMONIC on MRI; "
            "INFANTILE DIARRHOEA + CATARACTS COMBINATION: "
            "  any child with both findings → cholestanol testing immediately; "
            "  cataracts alone: 1st decade; diarrhoea: infancy; "
            "PSYCHIATRIC SYMPTOMS + ATAXIA IN YOUNG ADULT: "
            "  depression/psychosis + cerebellar ataxia in 20s-30s → CTX must be excluded; "
            "MOROCCAN JEWISH ORIGIN: 1:108 carrier frequency → lower threshold for testing; "
            "CDCA RESPONSE AS DIAGNOSTIC CONFIRMATION: "
            "  cholestanol normalisation after CDCA therapy confirms diagnosis"
        ),
        "treatment": (
            "CHENODEOXYCHOLIC ACID (CDCA) — PRIMARY TREATMENT: "
            "  dose: 750 mg/day (250 mg TDS) in adults; 15 mg/kg/day in children; "
            "  mechanism: FXR activation → suppresses CYP7A1 → reduces cholestanol synthesis; "
            "  REVERSES neurological deterioration if started before severe damage; "
            "  stabilises MRI lesions; improves cognitive function; "
            "  must be continued lifelong; "
            "  CAUTION: ursodeoxycholic acid (UDCA) — ineffective for CTX (does not activate FXR sufficiently); "
            "STATINS (adjunct): "
            "  reduce cholesterol substrate for cholestanol synthesis; "
            "  ADD-ON to CDCA (not replacement); "
            "BILE ACID SUPPLEMENTATION: "
            "  concurrent cholic acid (CA) sometimes added; "
            "FAT-SOLUBLE VITAMINS: A, D, E, K supplementation (bile acid deficiency → malabsorption); "
            "SEIZURE MANAGEMENT: LEV, LTG, ZNS (avoid VPA — hepatotoxicity risk); "
            "PSYCHIATRIC: standard management; "
            "MONITORING: "
            "  plasma cholestanol every 3-6 months; "
            "  LFTs (CDCA hepatotoxicity rare but monitor); "
            "  MRI brain every 2-3 years; "
            "  ophthalmology annually; "
            "NEWBORN FAMILY SCREENING: siblings of CTX patients → cholestanol testing immediately; "
            "GENETIC COUNSELLING: AR; 25% recurrence"
        ),
    },
    {
        "gene": "HSD3B7",
        "protein": (
            "HSD3B7 -- 16p11.2 AR -- 369aa -- 3-Beta-Hydroxy-Delta5-C27-Steroid-Oxidoreductase-42kDa-"
            "Microsomal-ER-Membrane-NAD-Dependent-Oxidoreductase-Catalyses-Step-1-Bile-Acid-Synthesis-"
            "After-7-Alpha-Hydroxylation-Oxidises-3-beta-OH-Position-on-C27-sterols-"
            "OMIM-Gene-606875-Disease-CBAS1-607765"
        ),
        "locus": "16p11.2",
        "protein_size": "369 aa / 42 kDa (microsomal ER membrane, NAD-dependent)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "HSD3B7 = 3β-hydroxy-Δ5-C27-steroid oxidoreductase: "
            "  catalyses the second reaction in the classic bile acid synthesis pathway; "
            "  converts 7α-hydroxycholesterol → 7α-hydroxy-4-cholesten-3-one (via oxidoreduction); "
            "  NAD-dependent; distinct from HSD3B1/HSD3B2 (adrenal/gonadal 3β-HSD); "
            "HSD3B7 LOSS: "
            "  3β-hydroxy-Δ5 bile acid intermediates accumulate; "
            "  3β-hydroxy-Δ5 cholenoic acids excreted in urine (PATHOGNOMONIC); "
            "  normal/low primary bile acid synthesis → fat-soluble vitamin deficiency; "
            "  fat-soluble vitamin malabsorption → rickets (D), coagulopathy (K), haemolytic anaemia (E); "
            "INCIDENCE: rare, <100 cases worldwide; "
            "RESPONDS TO: oral cholic acid (primary bile acid replacement) — excellent prognosis if treated; "
            "GGT: characteristically NORMAL (unusual for cholestatic liver disease) — key DDx clue; "
            "URSODEOXYCHOLIC ACID (UDCA): NOT effective — does not correct the substrate accumulation"
        ),
        "disease_category": (
            "CONGENITAL BILE ACID SYNTHESIS DEFECT TYPE 1 (CBAS1) — OMIM 607765; "
            "PRESENTATION: neonatal cholestasis or infantile liver disease; "
            "BIOCHEMICAL: "
            "  3β-hydroxy-Δ5 cholenoic acids in urine (PATHOGNOMONIC) — detected by LSIMS or FAB-MS; "
            "  elevated transaminases (AST, ALT); "
            "  conjugated hyperbilirubinaemia; "
            "  prolonged prothrombin time (vitamin K malabsorption); "
            "  NORMAL serum GGT — UNUSUAL for cholestasis (distinguishing feature); "
            "  normal serum bile acids (or low) — measured levels do not reflect abnormal intermediates; "
            "CLINICAL: "
            "  fat-soluble vitamin deficiency: rickets (vitamin D), coagulopathy (K), haemolytic anaemia (E); "
            "  failure to thrive; "
            "  hepatomegaly ± splenomegaly; "
            "  WITHOUT TREATMENT: progressive liver disease → cirrhosis; "
            "  WITH TREATMENT (cholic acid): near-complete normalisation of liver function; "
            "DIAGNOSIS: "
            "  urine bile acid analysis (LSIMS/FAB-MS/ESI-MS/MS) — key investigation; "
            "  normal serum bile acid profile does NOT exclude CBAS (abnormal intermediates not measured); "
            "  HSD3B7 sequencing"
        ),
        "disease_pathway": (
            "CLASSIC (NEUTRAL) BILE ACID SYNTHESIS — HSD3B7 POSITION: "
            "Cholesterol → (CYP7A1: step 1, rate-limiting, 7α-hydroxylation) → 7α-hydroxycholesterol "
            "→ (HSD3B7: step 2, 3β-oxidation/isomerisation) → 7α-hydroxy-4-cholesten-3-one "
            "→ (CYP8B1: step 3, 12α-hydroxylation toward cholic acid) or "
            "→ (AKR1D1: step 4, 5β-reduction) → 5β-cholestane-3α,7α-diol "
            "→ (CYP27A1: step 5) → primary bile acids (cholic acid, CDCA); "
            "HSD3B7 BLOCK: "
            "  7α-hydroxycholesterol → cannot proceed → accumulates as 3β-OH-Δ5 intermediates; "
            "  3β-hydroxy-5-cholenoic acids: 3β-hydroxy-5-cholenoic acid + 3β,7α-dihydroxy-5-cholenoic acid; "
            "  these are measurable in urine by LSIMS/mass spectrometry; "
            "CHOLIC ACID REPLACEMENT RATIONALE: "
            "  oral CA → activates FXR → suppresses CYP7A1 → "
            "  reduces flux through defective pathway; "
            "  simultaneously restores bile acid pool for fat-soluble vitamin absorption; "
            "  net: reduces toxic intermediate accumulation + restores bile acid function; "
            "WHY UDCA FAILS: "
            "  UDCA does not suppress CYP7A1 (not an FXR agonist at therapeutic doses); "
            "  does not replace normal bile acids; "
            "  toxic intermediates continue to accumulate"
        ),
        "pathognomonic": (
            "3BETA-HYDROXY-DELTA5 BILE ACID PRECURSORS IN URINE PATHOGNOMONIC: "
            "3β-hydroxy-5-cholenoic acid + 3β,7α-dihydroxy-5-cholenoic acid on urine LSIMS/FAB-MS; "
            "these are NOT detectable on standard serum bile acid profiles — "
            "urine mass spectrometry is mandatory for diagnosis; "
            "NORMAL GGT WITH CHOLESTASIS: "
            "  GGT is elevated in most causes of neonatal cholestasis; "
            "  NORMAL GGT in a cholestatic neonate → bile acid synthesis defect (CBAS1-4) or PFIC type 1/2; "
            "  key discriminating investigation; "
            "VITAMIN K-DEPENDENT COAGULOPATHY AT PRESENTATION: "
            "  prolonged PT in neonatal cholestasis → exclude CBAS urgently (treatable); "
            "  intracranial haemorrhage risk if vitamin K replacement delayed; "
            "CHOLIC ACID RESPONSE: "
            "  biochemical normalisation within weeks on CA therapy confirms diagnosis; "
            "  failure to improve on UDCA supports CBAS over biliary atresia"
        ),
        "treatment": (
            "CHOLIC ACID (PRIMARY BILE ACID REPLACEMENT) — CURATIVE TREATMENT: "
            "  dose: 5-15 mg/kg/day orally (RARE orphan drug, may require pharmacy compounding); "
            "  mechanism: replaces deficient primary bile acid + FXR activation suppresses upstream flux; "
            "  EXCELLENT PROGNOSIS if started before cirrhosis; "
            "  liver function normalises within weeks-months; "
            "  must continue lifelong (enzyme defect persists); "
            "FAT-SOLUBLE VITAMINS (pre-treatment): "
            "  vitamin K: IV/IM phytomenadione urgently if coagulopathy; "
            "  vitamin D: cholecalciferol supplementation; "
            "  vitamin E: oral α-tocopherol; "
            "  vitamin A: supplementation; "
            "UDCA: NOT RECOMMENDED — does not correct pathophysiology; "
            "  should not substitute for cholic acid; "
            "LIVER TRANSPLANTATION: "
            "  effective (replaces defective enzyme); "
            "  generally avoided if CA therapy can be started promptly; "
            "  reserved for established cirrhosis before diagnosis; "
            "MONITORING: "
            "  serum transaminases + bilirubin monthly initially; "
            "  urine bile acid profile (to confirm intermediate clearance); "
            "  fat-soluble vitamin levels every 6 months; "
            "GENETIC COUNSELLING: AR; 25% recurrence; HSD3B7 sequencing of family"
        ),
    },
    {
        "gene": "AKR1D1",
        "protein": (
            "AKR1D1 -- 7q33 AR -- 326aa -- Delta4-3-Oxosteroid-5-Beta-Reductase-37kDa-"
            "Cytoplasmic-NADPH-Dependent-Aldo-Keto-Reductase-Catalyses-5-Beta-Reduction-"
            "Steroid-A-Ring-Critical-for-Primary-Bile-Acid-Conjugation-Solubility-"
            "OMIM-Gene-604741-Disease-CBAS2-235555"
        ),
        "locus": "7q33",
        "protein_size": "326 aa / 37 kDa (cytoplasmic, NADPH-dependent aldo-keto reductase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "AKR1D1 = Δ4-3-oxosteroid 5β-reductase: "
            "  catalyses 5β-reduction of the steroid A-ring (step 4 of classic bile acid synthesis); "
            "  converts 7α-hydroxy-4-cholesten-3-one → 5β-cholestan-7α-ol-3-one; "
            "  essential for generating the 5β (cis-A/B ring) configuration of primary bile acids; "
            "AKR1D1 LOSS: "
            "  5α-reduced (allo) bile acids accumulate instead of normal 5β-bile acids; "
            "  allo-bile acids (5α-configuration) are far less efficient at micellar solubilisation; "
            "  accumulate as hepatotoxic intermediates; "
            "  SEVERE hepatitis and liver failure in neonatal period; "
            "INCIDENCE: extremely rare (<50 cases); "
            "SEVERITY: CBAS2 generally more severe than CBAS1 (HSD3B7); "
            "RESPONDS TO: oral cholic acid — may arrest progression but liver disease often more severe; "
            "AKR1D1 ALSO IMPORTANT FOR: "
            "  cortisol and other steroid hormone metabolism (5β-reduction of cortisol → tetrahydrocortisol); "
            "  bile acid-independent role in adrenal steroid catabolism"
        ),
        "disease_category": (
            "CONGENITAL BILE ACID SYNTHESIS DEFECT TYPE 2 (CBAS2) — OMIM 235555; "
            "NEONATAL PRESENTATION: severe neonatal hepatitis; "
            "BIOCHEMICAL: "
            "  allo-bile acids (5α-reduced stereoisomers) in urine — PATHOGNOMONIC; "
            "  3α,7α-dihydroxy-5α-cholestanoic acid + related allo-acids on LSIMS/MS/MS; "
            "  elevated transaminases; coagulopathy; "
            "  NORMAL or LOW GGT (as in CBAS1); "
            "CLINICAL: "
            "  severe neonatal hepatitis → cirrhosis and liver failure if untreated; "
            "  fat-soluble vitamin deficiency; "
            "  WORSE PROGNOSIS than CBAS1 without early treatment; "
            "ALLO-BILE ACID TOXICITY: "
            "  5α-stereoisomers are hepatotoxic (membrane-disrupting, cannot form normal micelles); "
            "  accumulate in hepatocytes → mitochondrial dysfunction → hepatocellular necrosis; "
            "DIAGNOSIS: "
            "  urine mass spectrometry (LSIMS or ESI-MS/MS) — allo-bile acid profile; "
            "  AKR1D1 sequencing; "
            "  CANNOT be diagnosed from serum bile acid profile alone"
        ),
        "disease_pathway": (
            "CLASSIC BILE ACID SYNTHESIS — AKR1D1 POSITION: "
            "7α-hydroxy-4-cholesten-3-one (from HSD3B7 product) "
            "→ (CYP8B1: optional 12α-hydroxylation → toward cholic acid) "
            "→ 7α,12α-dihydroxy-4-cholesten-3-one "
            "→ (AKR1D1: 5β-reduction) → 5β-cholestane-3α,7α,12α-triol "
            "→ (CYP27A1: side-chain oxidation) → bile acid conjugates; "
            "5-BETA vs 5-ALPHA CONFIGURATION: "
            "  normal primary bile acids (cholic acid, CDCA) = 5β configuration (cis A/B ring junction); "
            "  5β configuration → proper bile acid geometry for intestinal absorption + micellar formation; "
            "  AKR1D1 loss → 5α-reduction by an alternative enzyme → allo-bile acids (5α configuration); "
            "  5α bile acids are poorly absorbed, hepatotoxic, accumulate in liver; "
            "VALPROATE INTERACTION: "
            "  VPA inhibits mitochondrial beta-oxidation AND impairs AKR1D1 steroid metabolism; "
            "  CONTRAINDICATED in ANY suspected bile acid synthesis defect; "
            "  worsens hepatic injury; "
            "CHOLIC ACID MECHANISM: "
            "  CA bypasses defective step (already has correct 5β-configuration); "
            "  suppresses CYP7A1 → reduces flux through blocked pathway; "
            "  reduces allo-bile acid production"
        ),
        "pathognomonic": (
            "ALLO-BILE ACIDS (5-ALPHA-REDUCED STEREOISOMERS) IN URINE PATHOGNOMONIC: "
            "3α,7α-dihydroxy-5α-cholestanoic acid detected on urine LSIMS/MS/MS; "
            "NOT detectable on routine serum bile acid assays; "
            "specialist mass spectrometry laboratory required; "
            "NORMAL GGT WITH NEONATAL HEPATITIS: "
            "  GGT-normal cholestasis in neonate → CBAS until proven otherwise; "
            "  urgently request urine mass spectrometry; "
            "SEVERE NEONATAL LIVER FAILURE: "
            "  more severe than CBAS1 at presentation; "
            "  neonatal coagulopathy + jaundice + hepatomegaly; "
            "  if rapidly progressive → liver transplant may be needed before diagnosis; "
            "VPA CONTRAINDICATION IDENTIFICATION: "
            "  any seizures in the context of neonatal liver disease → VPA absolutely avoided; "
            "  phenobarbitone or levetiracetam only"
        ),
        "treatment": (
            "CHOLIC ACID — FIRST-LINE: "
            "  dose: 5-15 mg/kg/day; as in CBAS1 but may require higher doses; "
            "  mechanism: suppresses CYP7A1 → reduces flux through defective AKR1D1 pathway; "
            "  reduces allo-bile acid production; "
            "  improves liver function + fat absorption; "
            "  PROGNOSIS: worse than CBAS1 if cirrhosis established before treatment; "
            "LIVER TRANSPLANTATION: "
            "  more likely required than in CBAS1 (disease more severe); "
            "  restores normal AKR1D1 activity (provides donor enzyme); "
            "  post-transplant cholic acid NOT required; "
            "VPA: ABSOLUTELY CONTRAINDICATED — hepatotoxic + worsens AKR1D1 pathway; "
            "FAT-SOLUBLE VITAMINS: K urgently (coagulopathy); D, E, A supplementation; "
            "PHENOBARBITONE: caution (avoid in severe hepatic dysfunction); "
            "LEVETIRACETAM: preferred for seizures; "
            "URSODEOXYCHOLIC ACID: NOT effective; does not bypass or suppress defective step; "
            "MONITORING: "
            "  urine allo-bile acid profile (confirm clearance on CA therapy); "
            "  LFTs + coagulation weekly initially; "
            "  fat-soluble vitamin levels monthly"
        ),
    },
    {
        "gene": "CYP7B1",
        "protein": (
            "CYP7B1 -- 8q12.3 AR -- 506aa -- Oxysterol-7-Alpha-Hydroxylase-57kDa-Microsomal-ER-"
            "Cytochrome-P450-Hydroxylates-25-Hydroxycholesterol-and-27-Hydroxycholesterol-"
            "Alternative-Bile-Acid-Pathway-Also-Expressed-Brain-Hippocampus-Neurosteroid-7-Alpha-Hydroxylation-"
            "OMIM-Gene-603711-Disease-CBAS3-613812-SPG5-270800"
        ),
        "locus": "8q12.3",
        "protein_size": "506 aa / 57 kDa (microsomal ER CYP450)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "DUAL-PHENOTYPE GENE — same biallelic LOF → TWO DISTINCT DISEASES: "
            "  1. CBAS3 (early-onset, severe LOF): neonatal severe liver failure + cholestasis; "
            "  2. SPG5 (Hereditary Spastic Paraplegia type 5, later onset): "
            "     pure or complex HSP in young adults; "
            "     progressive spastic paraplegia + cerebellar ataxia; "
            "CYP7B1 FUNCTION: "
            "  alternative bile acid synthesis pathway: hydroxylates 27-hydroxycholesterol (from CYP27A1) → "
            "  7α,27-dihydroxycholesterol → downstream CDCA synthesis; "
            "  also: 25-hydroxycholesterol → 7α-hydroxy-25-hydroxycholesterol; "
            "  neuronal role: DHEA + other neurosteroids 7α-hydroxylation in brain; "
            "OXYSTEROL ACCUMULATION: "
            "  25-OH-cholesterol + 27-OH-cholesterol accumulate in CYP7B1 deficiency; "
            "  oxysterols are pro-apoptotic to neurons (explains SPG5 neurodegeneration); "
            "SPG5 MECHANISM: "
            "  oxysterol neurotoxicity → corticospinal tract degeneration; "
            "  CDCA treatment: reduces oxysterol accumulation via FXR-CYP7A1 suppression"
        ),
        "disease_category": (
            "DUAL DISEASE: "
            "CBAS3 (CONGENITAL BILE ACID SYNTHESIS DEFECT TYPE 3) — OMIM 613812: "
            "  severe neonatal liver failure; fat-soluble vitamin deficiency; "
            "  25-OH-cholesterol + related oxysterols in urine (mass spectrometry); "
            "  NORMAL or LOW GGT; "
            "  progressive without treatment; "
            "  responds to CDCA (preferred over cholic acid for CYP7B1 as CDCA is the end-product); "
            "SPG5 (HEREDITARY SPASTIC PARAPLEGIA TYPE 5A) — OMIM 270800: "
            "  onset: typically 2nd-4th decade; "
            "  pure HSP: progressive lower limb spasticity + hyperreflexia; "
            "  complex HSP in some: cerebellar ataxia + peripheral neuropathy + cognitive changes; "
            "  elevated plasma 25-hydroxycholesterol + 27-hydroxycholesterol — DIAGNOSTIC BIOMARKER; "
            "  white matter hyperintensities on brain MRI; "
            "  NO liver disease in SPG5 (liver disease age-dependent — resolves or not symptomatic in adults?); "
            "  CDCA treatment being investigated — early data suggests slows progression; "
            "GENOTYPE-PHENOTYPE CORRELATION: "
            "  same biallelic null mutations → either CBAS3 or SPG5 (modifier genes/environment); "
            "  family members with identical mutations may have different presentations"
        ),
        "disease_pathway": (
            "CYP7B1 IN BILE ACID AND OXYSTEROL METABOLISM: "
            "ALTERNATIVE BILE ACID PATHWAY: "
            "Cholesterol → (CYP27A1) → 27-hydroxycholesterol → (CYP7B1: 7α-hydroxylation) → "
            "7α,27-dihydroxycholesterol → downstream → CDCA (chenodeoxycholic acid); "
            "Cholesterol → (CYP27A1) → 25-hydroxycholesterol → (CYP7B1) → "
            "7α,25-dihydroxycholesterol → downstream bile acid intermediates; "
            "CYP7B1 BLOCK: "
            "  25-OH-cholesterol + 27-OH-cholesterol ACCUMULATE; "
            "  oxysterols are potent pro-apoptotic signals via LXR activation + mitochondrial pathway; "
            "  neuronal CYP7B1 role: brain oxysterol clearance → "
            "  loss → oxysterol neurotoxicity → corticospinal tract degeneration (SPG5); "
            "NEUROSTEROID PATHWAY: "
            "  CYP7B1 also metabolises DHEA, pregnenolone in brain; "
            "  loss → neurosteroid accumulation (possible additional neurotoxicity); "
            "CDCA MECHANISM IN CBAS3/SPG5: "
            "  CDCA activates FXR → suppresses CYP7A1 → reduces oxysterol precursor flux; "
            "  reduces 25-OH and 27-OH-cholesterol accumulation; "
            "  liver disease and neurodegeneration both potentially responsive"
        ),
        "pathognomonic": (
            "ELEVATED PLASMA 25-HYDROXYCHOLESTEROL + 27-HYDROXYCHOLESTEROL: "
            "oxysterol panel on plasma (specialist lab) — detects both CBAS3 and SPG5; "
            "CYP7B1 IS THE ONLY HSP-CAUSING GENE WHERE AN OXYSTEROL BIOMARKER EXISTS; "
            "DUAL PHENOTYPE FROM SAME MUTATIONS: "
            "  CYP7B1 mutations in an HSP family → biochemical testing mandatory; "
            "  CDCA treatment available (unlike most HSPs); "
            "SPG5 TREATMENT OPPORTUNITY: "
            "  only pure HSP with a potentially disease-modifying treatment (CDCA/statins); "
            "  diagnosis = therapeutic opportunity; "
            "NORMAL GGT IN NEONATAL CHOLESTASIS: "
            "  same as CBAS1/CBAS2 — key discriminator from biliary obstruction; "
            "OXYSTEROL-DRIVEN NEURODEGENERATION: "
            "  plasma oxysterol ratio > normal → explains white matter lesions on MRI; "
            "  biomarker for monitoring treatment response"
        ),
        "treatment": (
            "CHENODEOXYCHOLIC ACID (CDCA) — PREFERRED FOR CBAS3 (end-product of blocked pathway): "
            "  dose: 15 mg/kg/day (children); 250 mg TDS (adults); "
            "  rationale: CDCA is the downstream product of CYP7B1 — more physiological replacement; "
            "  also reduces oxysterol accumulation via FXR-CYP7A1 suppression; "
            "SPG5 TREATMENT (emerging): "
            "  CDCA: reduces plasma oxysterols; small cohort data shows neurological stabilisation; "
            "  simvastatin (add-on): reduces cholesterol substrate → less oxysterol generation; "
            "  antioxidants (vitamin E, N-acetylcysteine): neuroprotective rationale; "
            "  physiotherapy + spasticity management (baclofen, botulinum toxin for focal spasticity); "
            "  mobility aids as disease progresses; "
            "FAT-SOLUBLE VITAMINS (CBAS3): K, D, E, A urgently; "
            "VPA: AVOID in CBAS3 (hepatotoxic); "
            "LIVER TRANSPLANT (CBAS3): reserved for established cirrhosis; "
            "MONITORING: "
            "  plasma oxysterol panel every 6-12 months (treatment response); "
            "  MRI spine/brain every 2-3 years (SPG5); "
            "  walking speed + SARA score (SPG5 progression); "
            "GENETIC COUNSELLING: "
            "  all families: oxysterol panel on all at-risk siblings → treat early"
        ),
    },
    {
        "gene": "AMACR",
        "protein": (
            "AMACR -- 5p13.2-q11.1 AR -- 382aa -- 2-Methylacyl-CoA-Racemase-42kDa-"
            "Peroxisomal-Mitochondrial-Enzyme-Converts-2R-Methyl-Fatty-Acids-to-2S-Configuration-"
            "Required-for-Peroxisomal-Beta-Oxidation-of-Pristanic-Acid-and-C27-Bile-Acid-Intermediates-"
            "ALSO-Prostate-Cancer-Biomarker-Overexpressed-OMIM-Gene-604489-Disease-CBAS4-214950"
        ),
        "locus": "5p13.2-q11.1",
        "protein_size": "382 aa / 42 kDa (peroxisomal and mitochondrial dual-localisation)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "AMACR = 2-methylacyl-CoA racemase: "
            "  interconverts 2R and 2S stereoisomers of 2-methyl branched-chain fatty acids; "
            "  enables β-oxidation of: "
            "    pristanic acid (from phytanic acid breakdown — from dietary chlorophyll); "
            "    C27 bile acid intermediates (di- and trihydroxycholestanoic acids: DHCA + THCA); "
            "  DUAL COMPARTMENT: peroxisomal (main) + mitochondrial; "
            "AMACR LOSS: "
            "  pristanic acid accumulates (phytanic acid precursor cannot be fully catabolised); "
            "  THCA (trihydroxycholestanoic acid) + DHCA accumulate → urine and plasma; "
            "  C27 bile acid intermediates accumulate; "
            "AMACR AS PROSTATE CANCER BIOMARKER: "
            "  distinct from deficiency disease — AMACR is OVEREXPRESSED in prostate cancer; "
            "  used as IHC diagnostic marker for prostate cancer (AMACR/P504S antibody); "
            "  LOF causes disease; GOF/overexpression is oncogenic (different mechanism); "
            "LATE-ONSET: AMACR deficiency typically presents in adults (unlike CBAS1-3)"
        ),
        "disease_category": (
            "CONGENITAL BILE ACID SYNTHESIS DEFECT TYPE 4 (CBAS4) — OMIM 214950; "
            "PRESENTATION: ADULT-ONSET sensorimotor neuropathy ± cholestasis; "
            "  youngest presentations: 2nd decade; most adults 3rd-5th decade; "
            "  UNLIKE CBAS1-3: no neonatal liver disease; "
            "BIOCHEMICAL: "
            "  pristanic acid elevated in plasma; "
            "  THCA (trihydroxycholestanoic acid) + DHCA elevated in plasma; "
            "  C27 bile acid intermediates in urine (mass spectrometry); "
            "  phytanic acid: may be mildly elevated (pristanic acid precursor backup); "
            "  VLCFA: NORMAL (unlike X-ALD or ZSD — KEY DDx); "
            "CLINICAL: "
            "  sensorimotor neuropathy (axonal, progressive); "
            "  cerebellar ataxia; "
            "  retinitis pigmentosa (in some cases — similar to Refsum disease overlap); "
            "  liver disease: variable (some have chronic cholestasis); "
            "  severe vitamin K-independent coagulopathy in hepatic cases; "
            "DDx FROM REFSUM DISEASE: "
            "  Refsum: elevated phytanic acid (PHYH deficiency); "
            "  AMACR: elevated PRISTANIC acid + THCA/DHCA (phytanic acid usually normal); "
            "  both: neuropathy + retinitis pigmentosa"
        ),
        "disease_pathway": (
            "AMACR IN BILE ACID AND BRANCHED-CHAIN FATTY ACID METABOLISM: "
            "PATHWAY 1 — C27 BILE ACID INTERMEDIATES: "
            "Cholesterol → (CYP27A1, CYP7B1 etc.) → 3α,7α-dihydroxy-5β-cholestanoyl-CoA (DHCA-CoA) [2R form] "
            "→ (AMACR: R→S racemisation) → 2S-DHCA-CoA "
            "→ (peroxisomal ACOX2 β-oxidation) → CDCA-CoA → CDCA; "
            "  WITHOUT AMACR: THCA + DHCA cannot be β-oxidised → accumulate; "
            "PATHWAY 2 — PRISTANIC ACID: "
            "Phytanic acid (dietary) → (PHYH: α-oxidation) → pristanic acid [2R form] "
            "→ (AMACR: R→S racemisation) → 2S-pristanic acid "
            "→ (ACOX2 β-oxidation) → propionyl-CoA + acetyl-CoA; "
            "  WITHOUT AMACR: pristanic acid accumulates → neurotoxic; "
            "DIETARY MODIFICATION: "
            "  reducing dietary phytanic acid precursors → reduces pristanic acid load; "
            "  phytanic acid sources: dairy fat, ruminant meat fat, certain fish (cod, tuna); "
            "  chlorophyll (phytol) → phytanic acid (gut bacteria); "
            "  low-phytol/low-phytanic diet reduces substrate flux; "
            "BILE ACID REPLACEMENT: "
            "  cholic acid + CDCA: suppress CYP7A1 → reduce THCA/DHCA production; "
            "  reduce accumulation of toxic C27 intermediates"
        ),
        "pathognomonic": (
            "PRISTANIC ACID + THCA/DHCA ELEVATED: "
            "pristanic acid >1.0 µg/mL plasma (normal <0.3) — detectable on plasma bile acid panel; "
            "THCA + DHCA elevated on specialised C27 bile acid analysis; "
            "VLCFA NORMAL — CRITICAL DDx from X-ALD and ZSD: "
            "  X-ALD and ZSD: VLCFA elevated; "
            "  AMACR deficiency: VLCFA NORMAL (peroxisomal β-oxidation of VLCFA intact); "
            "  phytanic acid profile overlap with Refsum disease but pristanic acid distinguishes; "
            "ADULT-ONSET NEUROPATHY WITH RETINITIS PIGMENTOSA: "
            "  any adult with this combination → include AMACR/PHYH/ABCDs in diagnostic panel; "
            "AMACR OVEREXPRESSION IN PROSTATE DDx: "
            "  AMACR deficiency disease vs AMACR overexpression in cancer — opposite mechanisms; "
            "  AMACR IHC in biopsy = prostate cancer marker; "
            "  AMACR gene sequencing = rare metabolic disease diagnosis"
        ),
        "treatment": (
            "DIETARY MODIFICATION — PHYTANIC/PRISTANIC ACID RESTRICTION: "
            "  avoid: dairy fat (butter, cream, full-fat milk, cheese), ruminant fat (lamb, beef fat); "
            "  avoid: certain fish (cod, tuna, haddock high in phytol); "
            "  target plasma pristanic acid normalisation; "
            "BILE ACID REPLACEMENT (IF LIVER DISEASE): "
            "  cholic acid 5-15 mg/kg/day; CDCA 250-750 mg/day; "
            "  reduces C27 intermediate accumulation via FXR/CYP7A1 suppression; "
            "VITAMIN E: oral α-tocopherol (antioxidant neuroprotection); "
            "FAT-SOLUBLE VITAMINS: A, D, E, K if cholestasis present; "
            "PHYSIOTHERAPY: "
            "  gait rehabilitation; "
            "  balance training for ataxia; "
            "  occupational therapy for fine motor; "
            "RETINAL MONITORING: "
            "  annual ERG + fundal examination; "
            "  low vision aids if retinitis pigmentosa advanced; "
            "PROGNOSIS: "
            "  dietary modification slows progression; "
            "  disease less rapidly progressive than CBAS1-3; "
            "  neuropathy may stabilise with treatment; "
            "MONITORING: "
            "  plasma pristanic + THCA/DHCA annually; "
            "  EMG/nerve conduction every 2-3 years"
        ),
    },
    {
        "gene": "SC5D",
        "protein": (
            "SC5D -- 11q23.3 AR -- 299aa -- Sterol-C5-Desaturase-Lathosterol-Oxidase-35kDa-"
            "ER-Membrane-Non-Heme-Iron-Enzyme-FAD-Dependent-Catalyses-C5-Desaturation-"
            "Lathosterol-to-7-Dehydrocholesterol-Post-Squalene-Cholesterol-Synthesis-"
            "OMIM-Gene-602630-Disease-Lathosterolosis-607330"
        ),
        "locus": "11q23.3",
        "protein_size": "299 aa / 35 kDa (ER membrane, FAD-dependent non-heme iron enzyme)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "SC5D = sterol-C5-desaturase (lathosterol oxidase): "
            "  catalyses C5-desaturation of lathosterol → 7-dehydrocholesterol (7-DHC); "
            "  UPSTREAM of DHCR7 in cholesterol synthesis: "
            "  lathosterol → (SC5D: C5 desaturation) → 7-DHC → (DHCR7) → cholesterol; "
            "SC5D LOSS: "
            "  lathosterol accumulates (PATHOGNOMONIC); "
            "  7-DHC AND cholesterol BOTH depleted; "
            "  DISTINCT from DHCR7/SLO: different substrate accumulates; "
            "EXTREME RARITY: <20 cases reported worldwide (2026); "
            "SEVERITY: severe, overlapping with SLO but with LATHOSTEROL as marker (not 7-DHC); "
            "SC5D also expressed in: "
            "  brain (critical for CNS myelination during development); "
            "  liver; adrenal gland; skin"
        ),
        "disease_category": (
            "LATHOSTEROLOSIS — OMIM 607330; "
            "EXTREMELY RARE: fewer than 20 cases confirmed worldwide; "
            "BIOCHEMICAL HALLMARK: lathosterol elevated in plasma — PATHOGNOMONIC; "
            "  lathosterol:cholesterol ratio >0.05 (normal: <0.003); "
            "  7-DHC: low (SC5D is upstream of DHCR7 — 7-DHC cannot form); "
            "  total cholesterol: LOW (both 7-DHC and cholesterol deficient); "
            "CLINICAL FEATURES: "
            "  microcephaly (severe, progressive); "
            "  intellectual disability (moderate-severe); "
            "  liver disease: hepatomegaly + neonatal cholestasis; "
            "  cleft palate; "
            "  cataracts; "
            "  limb abnormalities (reported in severe cases); "
            "OVERLAP WITH SLO: microcephaly + liver + intellectual disability; "
            "KEY DIFFERENCE FROM SLO: "
            "  SLO: 7-DHC elevated; cholesterol low; lathosterol NORMAL; "
            "  Lathosterolosis: lathosterol elevated; 7-DHC LOW (not elevated); cholesterol low; "
            "DIAGNOSIS: "
            "  plasma sterol profile (GC-MS) — lathosterol elevated; "
            "  SC5D sequencing; "
            "  CANNOT distinguish from SLO on cholesterol level alone; "
            "TREATMENT: same rationale as SLO (cholesterol supplementation) — very limited data"
        ),
        "disease_pathway": (
            "POST-SQUALENE CHOLESTEROL SYNTHESIS — SC5D POSITION (UPSTREAM OF DHCR7): "
            "Squalene → lanosterol → zymosterol → lathosterol "
            "→ (SC5D: C5 desaturation, introduces Δ5 double bond) → 7-dehydrocholesterol (Δ5,7) "
            "→ (DHCR7: Δ7 reduction) → cholesterol (Δ5 only); "
            "SC5D BLOCK: "
            "  lathosterol (no double bonds in B-ring) accumulates; "
            "  7-DHC cannot form (SC5D is prerequisite for DHCR7 substrate); "
            "  cholesterol synthesis COMPLETELY blocked at this step; "
            "  both 7-DHC AND cholesterol depleted simultaneously; "
            "COMPARISON WITH DHCR7 DEFICIENCY (SLO): "
            "  SLO: SC5D intact → lathosterol → 7-DHC (SC5D works); DHCR7 blocked → 7-DHC accumulates; "
            "  Lathosterolosis: SC5D blocked → lathosterol accumulates; 7-DHC cannot form; "
            "  both diseases: cholesterol-deficient; "
            "CHOLESTEROL FUNCTIONS DISRUPTED: same as SLO "
            "(membrane, myelin, steroid hormones, hedgehog signalling, bile acids); "
            "LATHOSTEROL TOXICITY: less studied than 7-DHC; "
            "  lathosterol itself may have membrane-disruptive properties; "
            "  lacks Δ5 double bond → different membrane fluidity vs cholesterol; "
            "TREATMENT RATIONALE: "
            "  cholesterol supplementation (same as SLO); "
            "  no evidence for statins (further reduce already-depleted precursors); "
            "  photoprotection: lathosterol also photosensitive (though less than 7-DHC)"
        ),
        "pathognomonic": (
            "LATHOSTEROL ELEVATED PATHOGNOMONIC: "
            "lathosterol:cholesterol ratio >0.05 on plasma GC-MS sterol profile; "
            "LATHOSTEROL IS NOT MEASURED ON ROUTINE LIPID PANELS — "
            "  full plasma sterol profile (GC-MS) required; "
            "  standard cholesterol panel will just show low cholesterol (non-specific); "
            "DDx FROM SLO: "
            "  SLO: 7-DHC HIGH, lathosterol NORMAL; "
            "  Lathosterolosis: lathosterol HIGH, 7-DHC LOW; "
            "  both: total cholesterol LOW — not distinguishable by total cholesterol alone; "
            "EXTREME RARITY CONSIDERATION: "
            "  <20 cases worldwide → this diagnosis is almost never considered initially; "
            "  lathosterol in sterol profile is definitive; "
            "  SC5D sequencing confirms; "
            "SHARED FEATURES WITH SLO THAT PROMPT TESTING: "
            "  microcephaly + liver disease + intellectual disability + cleft palate → "
            "  full plasma sterol profile (not just 7-DHC); "
            "PRENATAL: "
            "  lathosterol in amniotic fluid if SC5D suspected"
        ),
        "treatment": (
            "CHOLESTEROL SUPPLEMENTATION (SAME RATIONALE AS SLO): "
            "  pharmaceutical cholesterol: 30-100 mg/kg/day; "
            "  dietary: egg yolk, meat; "
            "  evidence base extremely limited (<20 cases); "
            "  target: plasma cholesterol normalisation; "
            "PHOTOPROTECTION: "
            "  UV protection (lathosterol may also generate toxic photoproducts, though less studied); "
            "LIVER DISEASE MANAGEMENT: "
            "  fat-soluble vitamins: K, D, E, A; "
            "  ursodeoxycholic acid (UDCA) may reduce cholestasis symptoms; "
            "STATINS: AVOID — further reduce cholesterol synthesis below SC5D block; "
            "PHENYLKETONURIA DIET PARALLEL: "
            "  plant sterol-free diet (avoid phytosterol-containing products); "
            "CHOLESTEROL BIOAVAILABILITY: "
            "  plant sterols compete with cholesterol absorption (avoid margarine, fortified products); "
            "NEUROLOGICAL SUPPORT: "
            "  early intervention developmental therapy; "
            "  seizure management (LEV preferred); "
            "PROGNOSIS: "
            "  very limited data from <20 patients; "
            "  microcephaly not reversible with postnatal cholesterol; "
            "  treatment initiated prenatally (if identified) may modify outcome; "
            "MONITORING: "
            "  plasma lathosterol + cholesterol every 3-6 months; "
            "  developmental milestones; "
            "GENETIC COUNSELLING: AR; 25% recurrence"
        ),
    },
    {
        "gene": "EBP",
        "protein": (
            "EBP -- Xp11.23 XLD -- 230aa -- Emopamil-Binding-Protein-Delta8-Delta7-Sterol-Isomerase-"
            "25kDa-ER-Membrane-Protein-4-Transmembrane-Helices-Isomerises-Delta8-Cholesterol-Intermediates-"
            "to-Delta7-Configuration-Enables-Further-Cholesterol-Synthesis-Steps-"
            "OMIM-Gene-300205-Disease-CDPX2-302960"
        ),
        "locus": "Xp11.23",
        "protein_size": "230 aa / 25 kDa (4-transmembrane ER membrane protein)",
        "inheritance": (
            "X-LINKED DOMINANT — monoallelic mutations in females; "
            "MALES USUALLY LETHAL IN UTERO — virtually all live-born affected individuals are FEMALE; "
            "EXCEPTIONS: rare live-born males with somatic mosaicism or Klinefelter (47,XXY) survive; "
            "EBP = emopamil-binding protein (Δ8-Δ7 sterol isomerase): "
            "  isomerises Δ8 double bond → Δ7 configuration in post-squalene cholesterol synthesis; "
            "  zymosterol (Δ8) → 7-dehydrodesmosterol (Δ7, via EBP) → lathosterol → 7-DHC → cholesterol; "
            "  CRITICAL MID-PATHWAY STEP: EBP loss blocks conversion at Δ8 stage; "
            "EBP LOSS: "
            "  8-dehydrocholesterol (Δ8) + 8(9)-dehydrocholesterol accumulate; "
            "  3β-hydroxy-Δ8-cholesterol derivatives accumulate; "
            "  all downstream cholesterol synthesis halted; "
            "SOMATIC MOSAICISM: "
            "  X-inactivation (lyonisation) → patchy normal/affected cells in females; "
            "  explains Blaschko's lines in skin (following clonal boundaries of X-inactivation); "
            "  mosaicism also EXPLAINS SURVIVAL of some hemizygous males (mosaic for lethal mutation)"
        ),
        "disease_category": (
            "CHONDRODYSPLASIA PUNCTATA TYPE 2 (CDPX2) — OMIM 302960; "
            "X-LINKED DOMINANT; virtually all survivors are FEMALE; "
            "CLASSIC TRIAD: "
            "  1. Ichthyosis — skin changes following Blaschko's lines (streaky/patchy): "
            "     erythroderma + scaling at birth → evolves to follicular atrophoderma in childhood; "
            "  2. Stippled epiphyses (chondrodysplasia punctata) — calcification of cartilaginous epiphyses; "
            "     RESOLVES with age (NOT permanent — important for DDx); "
            "     calcium-EBP product deposits in cartilage; "
            "  3. Cataracts (sectoral or total) — unilateral or bilateral; "
            "     lens cholesterol requirement → inadequate; "
            "ADDITIONAL FEATURES: "
            "  asymmetric limb shortening; "
            "  craniofacial: short nose, mid-face hypoplasia; "
            "  intellectual disability: variable (mosaicism-dependent); "
            "  vertebral defects; "
            "BIOCHEMICAL: "
            "  8-dehydrocholesterol + 8(9)-dehydrocholesterol elevated on plasma sterol profile; "
            "  cholesterol: often LOW-NORMAL (some residual flux); "
            "  VLCFA: NORMAL (DDx from RCDP/ZSD); "
            "DDx: "
            "  RCDP (X-linked: EBP; AR: PEX7) → punctate calcifications similar; "
            "  warfarin embryopathy → stippled epiphyses (from warfarin in 1st trimester); "
            "  HAPPLE syndrome (another X-linked dominant sterol disorder)"
        ),
        "disease_pathway": (
            "EBP IN POST-SQUALENE CHOLESTEROL SYNTHESIS: "
            "Squalene → lanosterol → zymosterol (Δ8) "
            "→ (EBP: Δ8→Δ7 isomerisation) → 7-dehydrodesmosterol "
            "→ (DHCR24: Δ24 reduction) or → lathosterol (SC5D pathway) → 7-DHC → cholesterol; "
            "EBP BLOCK: "
            "  zymosterol (Δ8) and downstream Δ8-intermediates accumulate; "
            "  8-dehydrocholesterol (8-DHC) accumulates (PATHOGNOMONIC sterol); "
            "  3β-hydroxy-Δ8-cholestenol derivatives detectable in plasma/urine; "
            "SPATIAL PATTERNING (BLASCHKO'S LINES): "
            "  EBP is X-linked → females are mosaics (normal X / mutant X after lyonisation); "
            "  clones of normal cells + clones of EBP-null cells in skin; "
            "  skin lesions follow Blaschko's lines (lines of clonal epidermal spread during embryogenesis); "
            "  normal patches: normal cholesterol synthesis; "
            "  mutant patches: 8-DHC accumulates → ichthyosis; "
            "STIPPLED EPIPHYSES MECHANISM: "
            "  calcium deposits in cartilaginous epiphyses during fetal development; "
            "  cholesterol-deficient cartilage → calcium-EBP product accumulation; "
            "  resolves postnatally as cholesterol supply normalises in some tissues; "
            "MALE LETHALITY: "
            "  hemizygous males: ALL cells lack EBP → total cholesterol synthesis block → "
            "  embryonic lethality (cholesterol required for Hedgehog signalling, membrane function); "
            "  exceptions: mosaic males (somatic mutation after fertilisation) or 47,XXY survive"
        ),
        "pathognomonic": (
            "8-DEHYDROCHOLESTEROL ELEVATED PATHOGNOMONIC: "
            "8-DHC on plasma GC-MS sterol profile; "
            "3β-hydroxy-Δ8-intermediates distinctive pattern; "
            "ICHTHYOSIS FOLLOWING BLASCHKO'S LINES: "
            "  skin lesions in linear/patchy distribution following Blaschko's lines — "
            "  virtually pathognomonic for X-linked mosaic disorder; "
            "  normal adjacent skin confirms mosaicism; "
            "  biopsy of affected skin shows histological abnormality; "
            "STIPPLED EPIPHYSES THAT RESOLVE: "
            "  calcifications visible on neonatal/infant skeletal survey → "
            "  RESOLVE with age (distinguishes CDPX2 from permanent epiphyseal dysplasias); "
            "  radiograph at birth: punctate calcifications in patella, hips, shoulder epiphyses; "
            "FEMALE PREDOMINANCE OF SURVIVORS: "
            "  virtually no live-born hemizygous males → female patient → XLD diagnosis; "
            "  male fetus with CDPX2 features → likely miscarriage/stillbirth; "
            "WARFARIN EMBRYOPATHY DDx: "
            "  warfarin in 1st trimester → stippled epiphyses + nasal hypoplasia (similar to CDPX2); "
            "  KEY DDx: maternal warfarin history; sterol profile normal in warfarin embryopathy; "
            "RCDP TYPE 1 (PEX7) DDx: "
            "  also has stippled epiphyses but: VLCFA elevated; plasma phytanic elevated; AR (not XLD)"
        ),
        "treatment": (
            "CHOLESTEROL SUPPLEMENTATION: "
            "  rationale: similar to SLO (downstream cholesterol depleted); "
            "  evidence base: very limited (XLD disease, heterogeneous mosaicism); "
            "  some case reports: improved growth + skin lesions on cholesterol supplementation; "
            "STATINS: AVOID (as in SLO — reduce upstream cholesterol synthesis further); "
            "SKIN CARE: "
            "  emollients + keratolytics for ichthyosis; "
            "  Blaschko-line distribution: targeted treatment; "
            "  evolves to follicular atrophoderma → may improve spontaneously in some patches; "
            "ORTHOPAEDIC: "
            "  limb length discrepancy → orthopaedic monitoring + orthotics; "
            "  scoliosis surveillance; "
            "  stippled epiphyses: observe — typically resolve; no specific orthopaedic intervention for calcifications; "
            "OPHTHALMOLOGY: "
            "  cataract extraction if visual impairment; "
            "  early surgery to prevent amblyopia (unilateral cataract especially); "
            "DEVELOPMENTAL: "
            "  early intervention if intellectual disability; "
            "  mosaicism-dependent variability in cognitive outcome; "
            "MALE FETUS: "
            "  CDPX2 in male fetus → counsel re: likely lethality / very poor prognosis; "
            "  unless somatic mosaicism (rare); "
            "PRENATAL: "
            "  maternal sterol profile may be normal; EBP gene analysis; "
            "  fetal skeletal survey (stippled epiphyses); "
            "GENETIC COUNSELLING: "
            "  X-linked dominant; 50% risk per pregnancy of affected female; "
            "  male conceptuses with full mutation: likely spontaneous loss; "
            "  PGT-XLD available"
        ),
    },
]


def _generate_patients(gene_idx: int, n: int = 40, seed: int = 0):
    rng = random.Random(seed)
    gene_data = ATLAS_GENES[gene_idx]
    g = gene_data["gene"]

    # Per-gene clinical parameters
    params = {
        "DHCR7":  dict(onset_range=(0, 0.1), cholesterol_low=True,  liver_pct=0.45,
                       neuro_pct=0.90, nbs_pct=0.05, treatment_resp_pct=0.65,
                       pathognomonic_pct=0.98),
        "CYP27A1":dict(onset_range=(2, 35),  cholesterol_low=False, liver_pct=0.30,
                       neuro_pct=0.95, nbs_pct=0.00, treatment_resp_pct=0.80,
                       pathognomonic_pct=0.92),
        "HSD3B7": dict(onset_range=(0, 0.5), cholesterol_low=False, liver_pct=1.00,
                       neuro_pct=0.10, nbs_pct=0.00, treatment_resp_pct=0.90,
                       pathognomonic_pct=0.98),
        "AKR1D1": dict(onset_range=(0, 0.1), cholesterol_low=False, liver_pct=1.00,
                       neuro_pct=0.05, nbs_pct=0.00, treatment_resp_pct=0.70,
                       pathognomonic_pct=0.98),
        "CYP7B1": dict(onset_range=(0, 30),  cholesterol_low=False, liver_pct=0.60,
                       neuro_pct=0.80, nbs_pct=0.00, treatment_resp_pct=0.65,
                       pathognomonic_pct=0.88),
        "AMACR":  dict(onset_range=(15, 55), cholesterol_low=False, liver_pct=0.40,
                       neuro_pct=0.95, nbs_pct=0.00, treatment_resp_pct=0.60,
                       pathognomonic_pct=0.95),
        "SC5D":   dict(onset_range=(0, 0.1), cholesterol_low=True,  liver_pct=0.80,
                       neuro_pct=0.85, nbs_pct=0.00, treatment_resp_pct=0.40,
                       pathognomonic_pct=0.95),
        "EBP":    dict(onset_range=(0, 0.1), cholesterol_low=False, liver_pct=0.15,
                       neuro_pct=0.40, nbs_pct=0.00, treatment_resp_pct=0.55,
                       pathognomonic_pct=0.95),
    }
    p = params[g]
    lo, hi = p["onset_range"]

    patients = []
    for i in range(n):
        onset = round(rng.uniform(lo, hi), 2)
        chol = round(rng.uniform(50, 95) if p["cholesterol_low"] else rng.uniform(100, 200), 1)
        patients.append({
            "id": f"{g}-{seed}-{i+1:03d}",
            "gene": g,
            "onset_years": onset,
            "cholesterol_mg_dL": chol,
            "liver_involvement": rng.random() < p["liver_pct"],
            "neurological": rng.random() < p["neuro_pct"],
            "nbs_detected": rng.random() < p["nbs_pct"],
            "treatment_responsive": rng.random() < p["treatment_resp_pct"],
            "pathognomonic_marker_present": rng.random() < p["pathognomonic_pct"],
            "fat_soluble_vitamin_deficiency": rng.random() < (p["liver_pct"] * 0.85),
            "ggt_normal": g in ("HSD3B7", "AKR1D1", "CYP7B1"),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx in range(len(ATLAS_GENES)):
        all_patients.extend(_generate_patients(idx, n=40, seed=2678 + idx))

    summary = {}
    for p in all_patients:
        g = p["gene"]
        if g not in summary:
            summary[g] = {
                "gene": g, "n": 0,
                "liver_pct": 0, "neuro_pct": 0, "nbs_detected_pct": 0,
                "treatment_resp_pct": 0, "pathognomonic_pct": 0,
                "fat_soluble_vit_pct": 0, "mean_onset": 0.0, "mean_chol": 0.0,
                "ggt_normal": p["ggt_normal"],
            }
        s = summary[g]
        s["n"] += 1
        s["liver_pct"] += int(p["liver_involvement"])
        s["neuro_pct"] += int(p["neurological"])
        s["nbs_detected_pct"] += int(p["nbs_detected"])
        s["treatment_resp_pct"] += int(p["treatment_responsive"])
        s["pathognomonic_pct"] += int(p["pathognomonic_marker_present"])
        s["fat_soluble_vit_pct"] += int(p["fat_soluble_vitamin_deficiency"])
        s["mean_onset"] += p["onset_years"]
        s["mean_chol"] += p["cholesterol_mg_dL"]

    gene_summaries = []
    for g, s in summary.items():
        n = s["n"]
        gene_summaries.append({
            "gene": g,
            "n": n,
            "liver_involvement_pct": round(100 * s["liver_pct"] / n, 1),
            "neurological_pct": round(100 * s["neuro_pct"] / n, 1),
            "nbs_detected_pct": round(100 * s["nbs_detected_pct"] / n, 1),
            "treatment_responsive_pct": round(100 * s["treatment_resp_pct"] / n, 1),
            "pathognomonic_marker_pct": round(100 * s["pathognomonic_pct"] / n, 1),
            "fat_soluble_vitamin_deficiency_pct": round(100 * s["fat_soluble_vit_pct"] / n, 1),
            "mean_onset_years": round(s["mean_onset"] / n, 2),
            "mean_cholesterol_mg_dL": round(s["mean_chol"] / n, 1),
            "ggt_normal_in_cholestasis": s["ggt_normal"],
        })

    return {
        "atlas": "Hereditary Bile Acid Synthesis & Sterol Biosynthesis Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": len(all_patients),
        "seeds": list(range(2678, 2686)),
        "gene_summaries": gene_summaries,
        "pathway_classification": [
            "CHOLESTEROL SYNTHESIS DEFECTS (sterol accumulates upstream of DHCR7): "
            "DHCR7 (SLO — 7-DHC elevated, most common), SC5D (Lathosterolosis — lathosterol elevated, ultra-rare), "
            "EBP (CDPX2 — 8-DHC elevated, X-linked dominant, female-predominant)",
            "PRIMARY BILE ACID SYNTHESIS DEFECTS (C27 intermediates accumulate): "
            "HSD3B7 (CBAS1 — 3β-hydroxy-Δ5 precursors, neonatal cholestasis, cholic acid cures), "
            "AKR1D1 (CBAS2 — allo-bile acids, severe neonatal hepatitis), "
            "CYP7B1 (CBAS3/SPG5 — oxysterols, dual phenotype neonatal liver+adult spastic paraplegia), "
            "AMACR (CBAS4 — pristanic+THCA/DHCA, adult-onset neuropathy)",
            "BILE ACID SYNTHESIS AND OXYSTEROL METABOLISM: "
            "CYP27A1 (CTX — cholestanol elevated PATHOGNOMONIC, Achilles xanthomas, dentate nucleus T2, CDCA treats)",
        ],
        "nbs_detected_none": ["CYP27A1", "HSD3B7", "AKR1D1", "CYP7B1", "AMACR", "SC5D", "EBP"],
        "nbs_partially_detected": ["DHCR7 (expanded NBS only, not universal)"],
        "ggt_normal_in_cholestasis": ["HSD3B7", "AKR1D1", "CYP7B1"],
        "dual_phenotype_genes": ["CYP7B1 (CBAS3 neonatal + SPG5 adult)"],
        "primary_treatment": {
            "DHCR7": "Cholesterol supplementation + photoprotection; statins ABSOLUTELY CI",
            "CYP27A1": "CDCA 750 mg/day (FXR activation → suppresses cholestanol synthesis)",
            "HSD3B7": "Cholic acid 5-15 mg/kg/day (FXR suppression of upstream flux)",
            "AKR1D1": "Cholic acid; VPA ABSOLUTELY CI; liver transplant if cirrhosis",
            "CYP7B1": "CDCA (preferred over CA for CBAS3) + oxysterol reduction",
            "AMACR": "Dietary pristanic/phytanic acid restriction + bile acid replacement",
            "SC5D": "Cholesterol supplementation (very limited evidence, <20 cases)",
            "EBP": "Cholesterol supplementation + skin emollients; statins AVOID",
        },
        "male_lethal_gene": "EBP (X-linked dominant — hemizygous males virtually always die in utero)",
        "most_common": "DHCR7 (SLO) — 1:15,000-1:30,000 (most common cholesterol synthesis defect)",
    }


def generate_breakdown():
    all_entries = []
    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=2678 + idx)
        g = gene_data["gene"]
        n = len(patients)
        all_entries.append({
            "gene": g,
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance_summary": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "patients_n": n,
            "mean_onset_years": round(sum(p["onset_years"] for p in patients) / n, 2),
            "mean_cholesterol_mg_dL": round(sum(p["cholesterol_mg_dL"] for p in patients) / n, 1),
            "liver_involvement_pct": round(100 * sum(p["liver_involvement"] for p in patients) / n, 1),
            "neurological_pct": round(100 * sum(p["neurological"] for p in patients) / n, 1),
            "nbs_detected_pct": round(100 * sum(p["nbs_detected"] for p in patients) / n, 1),
            "treatment_responsive_pct": round(100 * sum(p["treatment_responsive"] for p in patients) / n, 1),
            "pathognomonic_marker_pct": round(100 * sum(p["pathognomonic_marker_present"] for p in patients) / n, 1),
        })
    return {"atlas": "Hereditary Bile Acid Synthesis & Sterol Biosynthesis Atlas", "gene_entries": all_entries}


def generate_definitions():
    glossary = {
        "7-Dehydrocholesterol (7-DHC)": (
            "PATHOGNOMONIC BIOMARKER for DHCR7/SLO: "
            "penultimate precursor in cholesterol synthesis (Δ5,7-sterol); "
            "normally present at trace levels (<5 µg/mL plasma); "
            "elevated (>10 µg/mL) in SLO — diagnostic; "
            "photosensitive → UV light → toxic oxysterols → behavioural worsening; "
            "measured by GC-MS plasma sterol profile"
        ),
        "Cholestanol": (
            "PATHOGNOMONIC BIOMARKER for CYP27A1/CTX: "
            "5α-dihydrocholesterol (cholesterol reduced at C5); "
            "normally at trace levels (<2 µg/mL); "
            "elevated in CTX (accumulates when bile acid synthesis blocked); "
            "deposits in tendons (xanthomas), brain, lens; "
            "measured by plasma sterol profile or dedicated GC-MS"
        ),
        "Lathosterol": (
            "PATHOGNOMONIC BIOMARKER for SC5D/Lathosterolosis: "
            "sterol with no double bonds in B-ring (Δ8 intermediate before SC5D); "
            "elevated lathosterol:cholesterol ratio >0.05 = diagnostic; "
            "normally traces only; distinct from 7-DHC (DHCR7) and 8-DHC (EBP)"
        ),
        "8-Dehydrocholesterol (8-DHC)": (
            "PATHOGNOMONIC BIOMARKER for EBP/CDPX2: "
            "3β-hydroxy-Δ8-cholesterol — accumulates when Δ8→Δ7 isomerisation blocked; "
            "detected on GC-MS plasma sterol profile; "
            "distinct from 7-DHC (one double bond difference, different position)"
        ),
        "Allo-Bile Acids": (
            "PATHOGNOMONIC for AKR1D1/CBAS2: "
            "5α-reduced stereoisomers of primary bile acids (5α = trans A/B ring junction); "
            "normal primary bile acids are 5β (cis A/B junction); "
            "allo-bile acids: less soluble, hepatotoxic, accumulate in hepatocytes; "
            "detected by urine LSIMS or ESI-MS/MS (specialist laboratory)"
        ),
        "3β-Hydroxy-Δ5 Bile Acid Precursors": (
            "PATHOGNOMONIC for HSD3B7/CBAS1: "
            "3β-hydroxy-5-cholenoic acid + 3β,7α-dihydroxy-5-cholenoic acid in urine; "
            "accumulate when HSD3B7 cannot oxidise the 3β-hydroxyl group; "
            "detected by LSIMS or FAB-MS (not visible on routine serum bile acid assay)"
        ),
        "THCA / DHCA": (
            "C27 BILE ACID INTERMEDIATES — elevated in AMACR/CBAS4: "
            "THCA = trihydroxycholestanoic acid (3α,7α,12α-trihydroxy-5β-cholestanoic acid); "
            "DHCA = dihydroxycholestanoic acid; "
            "cannot be β-oxidised without AMACR racemisation; "
            "accumulate in AMACR deficiency; measurable in plasma + urine"
        ),
        "Pristanic Acid": (
            "ELEVATED in AMACR/CBAS4 (also in Refsum if PHYH deficient): "
            "branched-chain fatty acid derived from phytanic acid (α-oxidation); "
            "requires AMACR to racemise 2R → 2S before peroxisomal β-oxidation; "
            "dietary sources: dairy fat, ruminant fat, certain fish; "
            "restriction reduces substrate flux in AMACR deficiency"
        ),
        "Smith-Lemli-Opitz (SLO)": (
            "Disease caused by DHCR7 biallelic LOF; 1:15,000-1:30,000; "
            "features: 2nd-3rd toe syndactyly (PATHOGNOMONIC, >97%), microcephaly, "
            "ASD, photosensitivity, genital ambiguity in 46XY; "
            "low cholesterol + high 7-DHC = biochemical signature; "
            "IVS8-1G>C most common European mutation"
        ),
        "Cerebrotendinous Xanthomatosis (CTX)": (
            "Disease caused by CYP27A1 biallelic LOF; 1:50,000-70,000; "
            "triad: Achilles xanthomas + cataracts + neurodegeneration; "
            "cholestanol elevated PATHOGNOMONIC; dentate nucleus T2 PATHOGNOMONIC; "
            "Moroccan Jewish founder pGln403Arg; CDCA treats and reverses neurodegeneration"
        ),
        "CBAS 1-4": (
            "Congenital Bile Acid Synthesis Defects — 4 types from 4 enzymes: "
            "CBAS1: HSD3B7 (3β-hydroxy-Δ5 precursors; neonatal cholestasis; normal GGT; cholic acid cures); "
            "CBAS2: AKR1D1 (allo-bile acids; severe neonatal hepatitis; VPA absolutely CI); "
            "CBAS3: CYP7B1 (oxysterols; neonatal liver ± adult SPG5; CDCA treats); "
            "CBAS4: AMACR (pristanic+THCA/DHCA; adult-onset neuropathy; dietary restriction); "
            "SHARED CBAS FEATURES: normal GGT (CBAS1-3), bile acid analysis mandatory, respond to bile acid replacement"
        ),
        "CDPX2 (Chondrodysplasia Punctata Type 2)": (
            "Disease caused by EBP monoallelic LOF in females (X-linked dominant); "
            "virtually all survivors are female (hemizygous males lethal in utero); "
            "triad: ichthyosis (Blaschko's lines) + stippled epiphyses (resolves!) + cataracts; "
            "8-DHC elevated on sterol profile; normal VLCFA; "
            "DDx: warfarin embryopathy (same stippling but normal sterol profile)"
        ),
        "Lathosterolosis": (
            "Disease caused by SC5D biallelic LOF; fewer than 20 cases worldwide; "
            "lathosterol elevated PATHOGNOMONIC (not 7-DHC); cholesterol low; "
            "features overlap SLO: microcephaly + liver + cleft palate + cataracts; "
            "DDx from SLO: lathosterol HIGH in lathosterolosis; 7-DHC HIGH in SLO"
        ),
        "Chenodeoxycholic Acid (CDCA)": (
            "PRIMARY TREATMENT for CTX (CYP27A1) and CBAS3 (CYP7B1): "
            "activates FXR (farnesoid X receptor) → suppresses CYP7A1 (rate-limiting bile acid synthesis) → "
            "reduces cholestanol/oxysterol accumulation; "
            "reverses neurological progression in CTX if started early; "
            "DISTINCT ACTION FROM CHOLIC ACID (cholic acid also treats CBAS1/2 but via same FXR mechanism)"
        ),
        "FXR (Farnesoid X Receptor)": (
            "Nuclear bile acid receptor — activated by bile acids (CDCA > cholic acid): "
            "FXR activation → suppresses CYP7A1 (rate-limiting classic pathway) → "
            "reduces upstream flux → reduces toxic intermediate accumulation; "
            "THERAPEUTIC TARGET in all CBAS types and CTX; "
            "explains why oral bile acid replacement works in these disorders"
        ),
        "GGT-Normal Cholestasis": (
            "KEY DISCRIMINATOR for CBAS1-3 (HSD3B7, AKR1D1, CYP7B1): "
            "almost all causes of neonatal cholestasis raise GGT (biliary epithelial marker); "
            "CBAS disorders + PFIC1 + PFIC2 = NORMAL GGT in cholestasis; "
            "normal GGT with conjugated hyperbilirubinaemia → urgent bile acid mass spectrometry; "
            "if GGT elevated → more likely biliary atresia, neonatal hepatitis, PFIC3, other"
        ),
        "Statins (Contraindication in Sterol Synthesis Defects)": (
            "STATINS ABSOLUTELY CONTRAINDICATED in DHCR7/SLO and SC5D/Lathosterolosis: "
            "statins inhibit HMG-CoA reductase (upstream rate-limiting step) → "
            "further reduce cholesterol synthesis in already-deficient pathway → "
            "worsen cholesterol depletion AND increase toxic precursor accumulation; "
            "STATINS ARE ADJUNCT (not contraindicated) in CTX (CYP27A1): "
            "add to CDCA to further reduce cholesterol substrate"
        ),
        "Blaschko's Lines": (
            "Linear patterns of skin lesion distribution following embryonic clonal development paths: "
            "X-linked mosaic disorders (EBP/CDPX2, NEMO) → Blaschko pattern as X-inactivation divides "
            "clones of normal vs mutant cells; "
            "skin lesions = cells where mutant X is active; "
            "normal adjacent skin = cells where normal X is active; "
            "NOT dermatomes (not nerve distribution) and NOT Langer's lines"
        ),
        "Smith-Lemli-Opitz IVS8-1G>C": (
            "Most common SLO mutation in European populations (~25-30% of alleles): "
            "splice site mutation at intron 8 → exon 9 skipping → truncated protein → null; "
            "compound heterozygotes: IVS8-1G>C + missense → typical SLO (not severe); "
            "homozygous null (IVS8-1G>C/IVS8-1G>C): variable, may be severe"
        ),
        "Maternal Low Unconjugated Oestriol (uE3)": (
            "PRENATAL CLUE FOR SLO (DHCR7 deficiency): "
            "normal triple screen includes uE3 (unconjugated oestriol); "
            "7-DHC cannot be converted to 16-hydroxy-DHEA (oestriol precursor requires normal sterol); "
            "low uE3 (<0.5 MoM) on maternal triple screen → reflex 7-DHC on amniotic fluid + DHCR7 sequencing; "
            "also low in: X-linked ichthyosis (STS deficiency), Smith-Magenis, placental aromatase deficiency"
        ),
    }

    return {
        "atlas": "Hereditary Bile Acid Synthesis & Sterol Biosynthesis Atlas",
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
