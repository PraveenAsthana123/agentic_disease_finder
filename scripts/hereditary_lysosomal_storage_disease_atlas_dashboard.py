#!/usr/bin/env python3
"""Hereditary-Lysosomal-Storage-Disease-Atlas -- Complete 8-Gene Reference
Classic lysosomal storage diseases with neurological and visceral manifestations
(Gaucher/GBA, Niemann-Pick-A-B/SMPD1, Niemann-Pick-C/NPC1, Tay-Sachs/HEXA,
 Krabbe/GALC, MLD/ARSA, GM1-Morquio-B/GLB1, Farber/ASAH1):

GBA    (Glucocerebrosidase; 497aa; 1q22; AR;
         Gaucher disease; most common LSD worldwide;
         Splenomegaly + thrombocytopenia + bone crises TRIAD;
         ERT imiglucerase/velaglucerase alfa FDA-approved;
         SRT eliglustat FDA2014/miglustat;
         Parkinson risk 5x in heterozygous carriers;
         seed SEED_BASE+0).
SMPD1  (Acid sphingomyelinase; 629aa; 11p15.4; AR;
         Niemann-Pick Type A (severe neuronopathic) + Type B (visceral non-neuronopathic);
         ASMD acid sphingomyelinase deficiency;
         Olipudase alfa XENPOZYME FDA2022 Type B ONLY;
         Sea-blue histiocytes on BM biopsy PATHOGNOMONIC;
         seed SEED_BASE+1).
NPC1   (Niemann-Pick Type C1 protein; 1278aa; 18q11.2; AR;
         Niemann-Pick Type C; intracellular cholesterol trafficking defect;
         Vertical Supranuclear Gaze Palsy VSGP PATHOGNOMONIC;
         Gelastic cataplexy PATHOGNOMONIC;
         Filipin staining unesterified cholesterol in fibroblasts PATHOGNOMONIC;
         Miglustat slows progression;
         seed SEED_BASE+2).
HEXA   (Hexosaminidase A alpha subunit; 529aa; 15q23; AR;
         Tay-Sachs disease; GM2 gangliosidosis;
         Cherry-red spot PATHOGNOMONIC on fundoscopy;
         Ashkenazi Jewish founder mutations (carrier 1/30);
         Infantile lethal; no disease-modifying therapy;
         seed SEED_BASE+3).
GALC   (Galactocerebrosidase; 669aa; 14q31.3; AR;
         Krabbe disease; globoid cell leukodystrophy;
         Psychosine toxic metabolite PATHOGNOMONIC biomarker;
         HSCT effective ONLY presymptomatic NBS-detected;
         Irritability + stiffness + developmental regression TRIAD;
         seed SEED_BASE+4).
ARSA   (Arylsulfatase A; 507aa; 22q13.33; AR;
         Metachromatic Leukodystrophy MLD;
         Libmeldy (atidarsagene autotemcel) EMA2020 FDA2024 presymptomatic/early-symptomatic only;
         Sulfatide accumulation; metachromatically-staining granules PATHOGNOMONIC nerve biopsy;
         seed SEED_BASE+5).
GLB1   (Beta-galactosidase 1; 677aa; 3p22.3; AR;
         GM1 gangliosidosis AND Morquio B (MPS IVB) SAME GENE different mutations;
         Cherry-red spot + dysostosis TRIAD GM1 infantile PATHOGNOMONIC;
         Keratan sulfate elevated in urine;
         seed SEED_BASE+6).
ASAH1  (Acid ceramidase; 395aa; 8p22; AR;
         Farber lipogranulomatosis / Farber disease;
         Painful subcutaneous nodules + joint contractures + hoarseness TRIAD PATHOGNOMONIC;
         Ceramide accumulation;
         HSCT improves joints/nodules but NOT neurological;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3798-3805)
"""
import random

SEED_BASE = 3798

ATLAS_GENES = [
    {
        "gene": "GBA",
        "protein": (
            "GBA -- 1q22 AR lysosomal LOF -- 497aa -- "
            "Glucocerebrosidase-beta-glucosidase-lysosomal-glycolipid-"
            "Gaucher-disease-OMIM-230800-most-common-LSD-worldwide-"
            "ERT-imiglucerase-velaglucerase-alfa-FDA-SRT-eliglustat-FDA2014-miglustat-"
            "splenomegaly-bone-crises-Parkinson-risk-heterozygotes"
        ),
        "locus": "1q22",
        "protein_size": (
            "497 aa / ~59 kDa / 1q22 GBA glucocerebrosidase molecular context: "
            "STRUCTURE: "
            "  497 aa; lysosomal acid beta-glucosidase; cleaves glucosylceramide -> glucose + ceramide; "
            "  Deficiency -> glucosylceramide accumulation in macrophage lysosomes; "
            "  AR inheritance; biallelic loss-of-function variants; "
            "  N370S: most common Ashkenazi variant; mild/non-neuronopathic Type 1; "
            "  L444P: more severe; neuronopathic risk; common non-Jewish; "
            "  84GG insertion: severe; Type 2/3; "
            "  >400 variants identified; genotype-phenotype correlation imperfect; "
            "GAUCHER DISEASE -- THREE TYPES: "
            "  TYPE 1 (most common, 94% of Gaucher): NON-NEURONOPATHIC; "
            "    SPLENOMEGALY: massive; hepatomegaly; Gaucher cells (crumpled tissue-paper macrophages); "
            "    THROMBOCYTOPENIA: secondary to hypersplenism + marrow infiltration; "
            "    BONE DISEASE: infarcts, avascular necrosis (femoral head), Erlenmeyer-flask deformity; "
            "    BONE CRISES: sudden severe pain mimicking osteomyelitis; "
            "    Splenomegaly + thrombocytopenia + bone crises = GAUCHER TRIAD; "
            "    Anemia; fatigue; bleeding tendency; "
            "  TYPE 2 (acute neuronopathic): severe neurological, death <2yr; "
            "  TYPE 3 (chronic neuronopathic): Norrbottnian type; supranuclear gaze palsy; "
            "PARKINSON DISEASE RISK (CRITICAL): "
            "  GBA HETEROZYGOUS CARRIERS: 5x increased Parkinson risk vs population; "
            "  Mechanism: lysosomal dysfunction, alpha-synuclein accumulation; "
            "  GBA-Parkinson: earlier onset, faster cognitive decline than idiopathic PD; "
            "  Biallelic GBA Gaucher patients: further increased PD risk; "
            "DIAGNOSIS: "
            "  Leukocyte glucocerebrosidase activity: low/absent in affected; "
            "  GBA gene sequencing: MANDATORY (>400 variants); "
            "  Plasma glucosylsphingosine (lyso-GL1): biomarker; elevated, correlates with disease; "
            "  Bone marrow biopsy: Gaucher cells (crumpled-paper macrophages) PATHOGNOMONIC; "
            "  Chitotriosidase: elevated in plasma; treatment monitoring; "
            "TREATMENT: "
            "  Imiglucerase (Cerezyme): standard ERT 60 U/kg biweekly; FDA1994; "
            "  Velaglucerase alfa (VPRIV): ERT 60 U/kg biweekly; FDA2010; plant-derived; "
            "  Taliglucerase alfa (Elelyso): FDA2012; carrot-derived ERT; "
            "  Eliglustat (Cerdelga): SRT oral; FDA2014; substrate reduction; "
            "    Inhibits glucosylceramide synthase; first-line Type 1 adults eligible; "
            "    CYP2D6 metabolizer status REQUIRED before prescribing; "
            "  Miglustat (Zavesca): SRT oral; older agent; more side effects (diarrhea); "
            "    Not first-line; use if ERT/eliglustat unsuitable; "
            "  SRT ONLY after ERT failure or mild/moderate Type 1 Gaucher; "
            "  Splenectomy: AVOIDED now (ERT effective; splenomegaly reverses)"
        ),
        "syndrome": (
            "Gaucher-disease / AR-lysosomal-glucosylceramide-storage / "
            "splenomegaly + thrombocytopenia + bone-crises-TRIAD / "
            "Type1-non-neuronopathic-94pct / ERT-imiglucerase-velaglucerase / "
            "SRT-eliglustat-FDA2014 / Parkinson-risk-5x-heterozygotes"
        ),
        "inheritance": "AR (biallelic GBA mutations)",
        "key_therapy": "Imiglucerase/velaglucerase ERT 60 U/kg biweekly — eliglustat SRT FDA2014 — CYP2D6 testing mandatory",
        "pathognomonic": (
            "Gaucher cells (crumpled tissue-paper macrophages) in BM biopsy PATHOGNOMONIC; "
            "Erlenmeyer-flask femoral deformity on X-ray PATHOGNOMONIC; "
            "Splenomegaly + thrombocytopenia + bone crises TRIAD PATHOGNOMONIC; "
            "Plasma glucosylsphingosine (lyso-GL1) elevated = disease biomarker"
        ),
        "key_drugs_ci": [
            "Imiglucerase-ERT-60Ukg-biweekly-FDA1994",
            "Velaglucerase-alfa-ERT-FDA2010",
            "Eliglustat-SRT-oral-FDA2014-CYP2D6-mandatory",
            "Miglustat-SRT-second-line-diarrhea",
            "SRT-ONLY-after-ERT-failure-or-mild-moderate-Type1",
            "SPLENECTOMY-AVOID-ERT-reverses-splenomegaly",
        ],
    },
    {
        "gene": "SMPD1",
        "protein": (
            "SMPD1 -- 11p15.4 AR lysosomal LOF -- 629aa -- "
            "Acid-sphingomyelinase-sphingomyelin-phosphodiesterase-1-"
            "Niemann-Pick-Type-A-B-ASMD-OMIM-257200-257220-"
            "Olipudase-alfa-XENPOZYME-FDA2022-Type-B-ONLY-"
            "sea-blue-histiocytes-bone-marrow-PATHOGNOMONIC"
        ),
        "locus": "11p15.4",
        "protein_size": (
            "629 aa / ~70 kDa / 11p15.4 SMPD1 acid sphingomyelinase molecular context: "
            "STRUCTURE: "
            "  629 aa; lysosomal acid sphingomyelinase (ASM); cleaves sphingomyelin -> ceramide + phosphocholine; "
            "  Deficiency -> sphingomyelin accumulation in lysosomes of histiocytes/macrophages; "
            "  AR inheritance; biallelic SMPD1 variants; "
            "  R496L, L302P, fsP330: common variants; "
            "  Residual enzyme activity determines Type A vs Type B: "
            "    Type A: <1% residual ASM; severe CNS; "
            "    Type B: 1-10% residual ASM; primarily visceral; "
            "NIEMANN-PICK TYPE A (CLASSIC INFANTILE NEURONOPATHIC): "
            "  Onset: first months of life; "
            "  Massive hepatosplenomegaly + failure to thrive; "
            "  CHERRY-RED SPOT on fundoscopy: ~50% of Type A; "
            "  Rapidly progressive neurological deterioration: "
            "    Hypotonia -> spasticity; developmental regression; loss of milestones; "
            "  Death typically by age 2-3yr; "
            "  No effective disease-modifying therapy for Type A CNS disease; "
            "NIEMANN-PICK TYPE B (CHRONIC NON-NEURONOPATHIC): "
            "  Visceral disease dominant; NO neurological involvement; "
            "  Hepatosplenomegaly + lung infiltration (ILD pattern); "
            "  Thrombocytopenia + dyslipidemia (low HDL, high LDL/TG); "
            "  Coronary artery disease risk significantly elevated; "
            "  Chronic respiratory insufficiency (ILD/alveolar proteinosis); "
            "  Survival into adulthood common; "
            "SEA-BLUE HISTIOCYTES (PATHOGNOMONIC): "
            "  BM biopsy: macrophages with blue-green granular cytoplasm on Giemsa stain; "
            "  Due to sphingomyelin + other lipid accumulation; "
            "  PATHOGNOMONIC for SMPD1/Niemann-Pick Type B; "
            "  Also seen (less specific): Gaucher disease, some other LSDs; "
            "DIAGNOSIS: "
            "  Leukocyte acid sphingomyelinase activity: reduced (<50% normal for Type B); "
            "  SMPD1 gene sequencing: confirms variants; "
            "  Bone marrow biopsy: sea-blue histiocytes PATHOGNOMONIC; "
            "  Lyso-sphingomyelin: elevated biomarker in plasma; "
            "  Lung HRCT: ground-glass opacities (ILD pattern Type B); "
            "TREATMENT: "
            "  Olipudase alfa (XENPOZYME): ERT for ACID SPHINGOMYELINASE DEFICIENCY; "
            "    FDA approved 2022; TYPE B (non-neuronopathic) ONLY; "
            "    Type A: olipudase not indicated (CNS not reached by IV ERT); "
            "  Lipid management: statins/fibrates for dyslipidemia; "
            "  Pulmonary: bronchodilators, supplemental O2 for ILD; "
            "  Type A: supportive only; palliative"
        ),
        "syndrome": (
            "Niemann-Pick-Type-A-B / ASMD-acid-sphingomyelinase-deficiency / "
            "Type-A: severe-neuronopathic-infantile-fatal / "
            "Type-B: visceral-non-neuronopathic-chronic / "
            "sea-blue-histiocytes-BM-PATHOGNOMONIC / "
            "Olipudase-alfa-XENPOZYME-FDA2022-Type-B-ONLY"
        ),
        "inheritance": "AR (biallelic SMPD1 mutations)",
        "key_therapy": "Olipudase alfa (XENPOZYME) ERT FDA2022 — TYPE B ONLY — Type A: supportive only",
        "pathognomonic": (
            "Sea-blue histiocytes on BM biopsy PATHOGNOMONIC (Giemsa stain blue-green macrophages); "
            "Cherry-red spot fundoscopy (~50% Type A) PATHOGNOMONIC; "
            "Lyso-sphingomyelin elevated in plasma = disease biomarker"
        ),
        "key_drugs_ci": [
            "Olipudase-alfa-XENPOZYME-FDA2022-TYPE-B-ONLY",
            "Type-A-NO-disease-modifying-therapy-CNS-not-reached",
            "Statin-dyslipidemia-Type-B-coronary-risk",
            "Supplemental-O2-ILD-pulmonary-Type-B",
        ],
    },
    {
        "gene": "NPC1",
        "protein": (
            "NPC1 -- 18q11.2 AR lysosomal LOF -- 1278aa -- "
            "Niemann-Pick-Type-C1-protein-cholesterol-trafficking-regulator-"
            "Niemann-Pick-Type-C-OMIM-257220-VSGP-cataplexy-filipin-PATHOGNOMONIC-"
            "miglustat-progression-slowing-neonatal-cholestasis"
        ),
        "locus": "18q11.2",
        "protein_size": (
            "1278 aa / ~142 kDa / 18q11.2 NPC1 cholesterol trafficking molecular context: "
            "STRUCTURE: "
            "  1278 aa; large transmembrane protein; 13 transmembrane domains; "
            "  Expressed in lysosomal/late-endosomal membrane; "
            "  Function: transports unesterified cholesterol out of late endosomes/lysosomes; "
            "  NPC2 (smaller protein) works in tandem with NPC1 for cholesterol transfer; "
            "  Deficiency -> unesterified cholesterol + sphingolipids trap in lysosomes; "
            "  AR; ~85% Niemann-Pick C cases from NPC1; ~15% from NPC2 (19p13.11); "
            "NIEMANN-PICK TYPE C -- NEUROLOGICAL PROGRESSION: "
            "  VERTICAL SUPRANUCLEAR GAZE PALSY (VSGP): "
            "    PATHOGNOMONIC for NPC1 -- impaired voluntary vertical eye movements; "
            "    Preserved oculocephalic (Doll's eye) reflex; "
            "    Present in >80% symptomatic NPC patients; "
            "    Earliest: difficulty with downward gaze; "
            "  GELASTIC CATAPLEXY: "
            "    PATHOGNOMONIC for NPC1 -- sudden muscle atonia triggered by laughter/emotion; "
            "    Distinct from Niemann-Pick A/B; "
            "    Present in ~30-40% NPC patients; "
            "  PROGRESSIVE NEURODEGENERATION: "
            "    Cerebellar ataxia + dysarthria + dysphagia; "
            "    Cognitive decline and dementia (progressive); "
            "    Seizures (multiple types); "
            "    Progressive spastic paraparesis; "
            "  PSYCHIATRIC: "
            "    Psychosis, schizophrenia-like presentation -- NPC misdiagnosed as schizophrenia; "
            "    Any young-onset psychiatric disease + neurological = SCREEN FOR NPC; "
            "VISCERAL DISEASE: "
            "  Hepatosplenomegaly: common but variable; "
            "  Neonatal cholestatic jaundice (transient): resolves spontaneously; "
            "    IMPORTANT: NPC1 neonatal cholestasis = SCREEN AT AGE 3-5yr for neurological NPC; "
            "  Adult presentation can lack visible splenomegaly; "
            "FILIPIN STAINING (DIAGNOSTIC GOLD STANDARD): "
            "  Fibroblast culture + filipin staining: "
            "    Filipin binds unesterified cholesterol -> intense perinuclear fluorescence; "
            "    PATHOGNOMONIC in classic NPC -- massive perinuclear cholesterol accumulation; "
            "  Variants with variant filipin patterns: follow up with NPC1/NPC2 sequencing; "
            "BIOMARKER: "
            "  Plasma oxysterols (7-ketocholesterol, 25-hydroxycholesterol): elevated; "
            "  Plasma N-palmitoyl-O-phosphocholineserine (PPCS): specific NPC biomarker; "
            "  Lysosphingomyelin-509: sensitive and specific plasma biomarker; "
            "TREATMENT: "
            "  Miglustat (Zavesca): substrate reduction for neurological form; "
            "    Slows neurological progression (delays ambulation loss); "
            "    NOT curative; NOT available in all countries for NPC; "
            "  Arimoclomol: heat shock protein co-inducer; clinical trials NPC; "
            "  Hydroxypropyl-beta-cyclodextrin: experimental; intrathecal/IV; "
            "  Supportive: antiepileptics, physiotherapy, communication aids"
        ),
        "syndrome": (
            "Niemann-Pick-Type-C / AR-cholesterol-trafficking-NPC1 / "
            "VSGP-PATHOGNOMONIC / gelastic-cataplexy-PATHOGNOMONIC / "
            "filipin-staining-fibroblasts-PATHOGNOMONIC / "
            "cerebellar-ataxia-dementia-progression / miglustat"
        ),
        "inheritance": "AR (biallelic NPC1 mutations; ~15% NPC2 gene)",
        "key_therapy": "Miglustat (slows neurological progression) — arimoclomol clinical trials — supportive",
        "pathognomonic": (
            "VSGP (vertical supranuclear gaze palsy) = PATHOGNOMONIC NPC1; "
            "Gelastic cataplexy (laugh-triggered atonia) = PATHOGNOMONIC NPC1; "
            "Filipin staining: massive perinuclear unesterified cholesterol fluorescence in fibroblasts PATHOGNOMONIC; "
            "Plasma lysosphingomyelin-509 elevated = specific biomarker"
        ),
        "key_drugs_ci": [
            "Miglustat-oral-substrate-reduction-neurological-progression-slowing",
            "Arimoclomol-clinical-trials-NPC",
            "Filipin-staining-DIAGNOSTIC-before-gene-sequencing",
            "Oxysterols-plasma-biomarker-screening",
            "Screen-NPC-any-young-onset-ataxia-plus-VSGP",
        ],
    },
    {
        "gene": "HEXA",
        "protein": (
            "HEXA -- 15q23 AR lysosomal LOF -- 529aa -- "
            "Hexosaminidase-A-alpha-subunit-GM2-ganglioside-hydrolysis-"
            "Tay-Sachs-disease-GM2-gangliosidosis-OMIM-272800-"
            "cherry-red-spot-fundoscopy-PATHOGNOMONIC-Ashkenazi-Jewish-founder-"
            "infantile-lethal-no-ERT-available"
        ),
        "locus": "15q23",
        "protein_size": (
            "529 aa / ~60 kDa / 15q23 HEXA hexosaminidase A alpha subunit molecular context: "
            "STRUCTURE: "
            "  529 aa; alpha subunit of beta-hexosaminidase A (HexA = alpha+beta heterodimer); "
            "  HexA specifically degrades GM2 ganglioside in lysosomes; "
            "  HexB (beta-beta homodimer) degrades other substrates but NOT GM2; "
            "  HEXA deficiency -> selective HexA absence -> GM2 ganglioside accumulates in neurons; "
            "  HEXB mutations: Sandhoff disease (HexA + HexB both absent); "
            "  AR inheritance; biallelic HEXA mutations; "
            "  Infantile: <5% residual HexA; "
            "ASHKENAZI JEWISH FOUNDER MUTATIONS: "
            "  Carrier frequency: ~1/30 Ashkenazi Jews (vs ~1/300 general population); "
            "  c.1274_1277dupTATC (1278insTATC): most common Ashkenazi frameshift; "
            "  c.1421+1G>C (IVS12+1G>C): second common Ashkenazi; "
            "  p.G269S: adult/late-onset Tay-Sachs; "
            "  Carrier screening programs (Dor Yeshorim): highly effective in Ashkenazi communities; "
            "INFANTILE TAY-SACHS (CLASSIC): "
            "  Onset: 3-6 months; normal development then regression; "
            "  Startle response to sound (hyperacusis): EARLY SIGN; "
            "  Progressive loss of milestones: motor, cognitive; "
            "  CHERRY-RED SPOT ON FUNDOSCOPY: "
            "    PATHOGNOMONIC -- normal macula surrounded by pale lipid-laden ganglion cells; "
            "    Seen in ~100% infantile Tay-Sachs; "
            "    Also: Niemann-Pick A, GM1 gangliosidosis (GLB1); "
            "  Seizures (multiple types: myoclonic, tonic-clonic); "
            "  Macrocephaly (cerebral GM2 accumulation); "
            "  Decerebrate posturing; vegetative state; "
            "  Death typically by age 2-5yr; "
            "LATE-ONSET TAY-SACHS (LOTS): "
            "  p.G269S or compound heterozygote with null allele; "
            "  Spinocerebellar ataxia + proximal muscle weakness; "
            "  Psychiatric manifestations (psychosis, depression); "
            "  Onset adolescence-adulthood; "
            "  Cognitive decline late; "
            "DIAGNOSIS: "
            "  Serum/leukocyte hexosaminidase A activity: severely reduced in affected; "
            "  Hexosaminidase A/total ratio: <10% in affected; "
            "  Carrier: ~50% HexA activity; "
            "  HEXA gene sequencing: confirms specific variant; "
            "  Fundoscopy: cherry-red spot (infantile form); "
            "TREATMENT: "
            "  NO disease-modifying therapy currently FDA-approved; "
            "  Gene therapy clinical trials underway; "
            "  Supportive: antiepileptics, nutrition, physiotherapy; "
            "  Palliative care for infantile form"
        ),
        "syndrome": (
            "Tay-Sachs / GM2-gangliosidosis / HEXA-HexA-alpha-subunit / "
            "cherry-red-spot-fundoscopy-PATHOGNOMONIC / "
            "Ashkenazi-Jewish-1in30-carrier / infantile-lethal / "
            "no-ERT-available / late-onset-LOTS-ataxia"
        ),
        "inheritance": "AR (biallelic HEXA mutations)",
        "key_therapy": "No disease-modifying therapy — supportive + palliative — gene therapy clinical trials",
        "pathognomonic": (
            "Cherry-red spot on fundoscopy PATHOGNOMONIC (macula surrounded by pale lipid-laden ganglion cells); "
            "Hyperacusis (exaggerated startle to sound) early sign infantile; "
            "HexA activity severely reduced / HexA:total ratio <10% diagnostic; "
            "Ashkenazi carrier frequency 1/30 PATHOGNOMONIC context"
        ),
        "key_drugs_ci": [
            "NO-ERT-available-Tay-Sachs",
            "Gene-therapy-clinical-trials-ongoing",
            "Carrier-screening-Ashkenazi-1in30-MANDATORY",
            "Antiepileptics-seizure-management",
            "Palliative-care-infantile-form",
        ],
    },
    {
        "gene": "GALC",
        "protein": (
            "GALC -- 14q31.3 AR lysosomal LOF -- 669aa -- "
            "Galactocerebrosidase-galactosylceramide-galactosylsphingosine-"
            "Krabbe-disease-globoid-cell-leukodystrophy-OMIM-245200-"
            "psychosine-PATHOGNOMONIC-HSCT-presymptomatic-NBS-critical-"
            "irritability-stiffness-developmental-regression-TRIAD"
        ),
        "locus": "14q31.3",
        "protein_size": (
            "669 aa / ~80 kDa / 14q31.3 GALC galactocerebrosidase molecular context: "
            "STRUCTURE: "
            "  669 aa; lysosomal galactocerebrosidase; "
            "  Cleaves galactosylceramide and galactosylsphingosine (psychosine); "
            "  Deficiency -> galactosylceramide + PSYCHOSINE accumulation in CNS myelin; "
            "  Psychosine = galactosylsphingosine: extremely toxic to oligodendrocytes; "
            "  AR inheritance; biallelic GALC variants; "
            "  30-kb deletion: most common Western Caucasian variant (~45% alleles); "
            "KRABBE DISEASE (GLOBOID CELL LEUKODYSTROPHY): "
            "  EARLY-INFANTILE (most common, ~85%): "
            "    Onset: 3-6 months; "
            "    TRIAD: irritability + stiffness (hypertonia) + developmental regression; "
            "    Stage 1: extreme irritability; feeding difficulties; hypertonicity; "
            "    Stage 2: progressive spastic quadriplegia; optic atrophy; "
            "    Stage 3: vegetative state; death typically by age 2yr; "
            "  LATE-INFANTILE / JUVENILE: "
            "    Onset: 6 months - 3 years; slower progression; "
            "    Ataxia; vision loss; regression; "
            "  ADULT-ONSET: "
            "    Rare; primarily motor neuron-like; "
            "PSYCHOSINE TOXICITY (PATHOGNOMONIC): "
            "  Psychosine (galactosylsphingosine): no ceramide moiety = cannot be stored; "
            "  Cytotoxic to oligodendrocytes = demyelination; "
            "  GLOBOID CELLS: multinucleated macrophages engulfing galactosylceramide = PATHOGNOMONIC; "
            "  Psychosine blood/CSF level: elevated; NBS screening biomarker; "
            "MRI: "
            "  White matter changes: periventricular T2 hyperintensity; "
            "  Basal ganglia T2 signal abnormality; "
            "  Corticospinal tract involvement; "
            "NBS AND HSCT (CRITICAL TIMING): "
            "  HSCT effective ONLY presymptomatic: "
            "    NBS detected (dried blood spot GALC enzyme + psychosine); "
            "    Cord blood transplant before symptom onset: significantly better outcomes; "
            "    After symptom onset: HSCT does NOT prevent neurological decline; "
            "  New York State NBS program: Krabbe included since 2006; "
            "DIAGNOSIS: "
            "  Leukocyte GALC activity: reduced; "
            "  GALC gene sequencing; "
            "  Brain MRI: white matter changes; "
            "  NCV: peripheral neuropathy; "
            "  CSF protein: elevated; "
            "TREATMENT: "
            "  HSCT: ONLY presymptomatic benefit; NBS critical; "
            "  Symptomatic: antiepileptics, physiotherapy; "
            "  Gene therapy clinical trials"
        ),
        "syndrome": (
            "Krabbe-disease / globoid-cell-leukodystrophy / GALC-galactocerebrosidase / "
            "psychosine-toxic-PATHOGNOMONIC / "
            "HSCT-presymptomatic-ONLY-NBS-critical / "
            "irritability-stiffness-regression-TRIAD-early-infantile"
        ),
        "inheritance": "AR (biallelic GALC mutations)",
        "key_therapy": "HSCT only presymptomatic (NBS critical) — gene therapy trials — symptomatic support",
        "pathognomonic": (
            "Psychosine (galactosylsphingosine) elevated in blood/CSF = PATHOGNOMONIC Krabbe; "
            "Globoid cells (multinucleated macrophages) in CNS white matter PATHOGNOMONIC; "
            "TRIAD: extreme irritability + hypertonia + developmental regression in early infantile; "
            "White matter changes on MRI (periventricular T2 hyperintensity)"
        ),
        "key_drugs_ci": [
            "HSCT-presymptomatic-ONLY-NBS-detected-benefit",
            "HSCT-after-symptom-onset-NO-neurological-benefit",
            "NBS-GALC-enzyme-plus-psychosine-mandatory-where-available",
            "Gene-therapy-clinical-trials-ongoing",
        ],
    },
    {
        "gene": "ARSA",
        "protein": (
            "ARSA -- 22q13.33 AR lysosomal LOF -- 507aa -- "
            "Arylsulfatase-A-sulfatide-lysosomal-hydrolase-"
            "Metachromatic-Leukodystrophy-MLD-OMIM-250100-"
            "Libmeldy-atidarsagene-autotemcel-EMA2020-FDA2024-"
            "presymptomatic-early-symptomatic-only-"
            "metachromatically-staining-granules-nerve-biopsy-PATHOGNOMONIC"
        ),
        "locus": "22q13.33",
        "protein_size": (
            "507 aa / ~58 kDa / 22q13.33 ARSA arylsulfatase A molecular context: "
            "STRUCTURE: "
            "  507 aa; lysosomal arylsulfatase A; cleaves sulfate from galactosylceramide-3-sulfate (sulfatide); "
            "  Deficiency -> sulfatide (galactosylceramide-3-sulfate) accumulation in myelin; "
            "  BOTH peripheral AND central nervous system involvement (unique among LSDs); "
            "  AR inheritance; biallelic ARSA mutations; "
            "  Pseudodeficiency allele (ARSA*PD): reduced enzyme activity with NO disease; "
            "    CRITICAL: ARSA*PD carrier + affected allele = carrier; NOT affected; "
            "    Enzyme assay alone cannot diagnose -- gene sequencing MANDATORY; "
            "METACHROMATIC LEUKODYSTROPHY -- FORMS: "
            "  LATE-INFANTILE MLD (most common, ~50%): "
            "    Onset: 1-2yr; gait abnormality first; peripheral neuropathy; "
            "    Progressive regression: loss of walking -> wheelchair -> loss of speech; "
            "    Dementia; seizures; death in several years; "
            "  JUVENILE MLD (~25%): "
            "    Onset: 3-16yr; behavioral/school problems first; "
            "    Peripheral neuropathy + cerebellar ataxia; cognitive decline; "
            "  ADULT MLD (~25%): "
            "    Onset: >16yr; psychiatric first (schizophrenia-like); "
            "    Peripheral neuropathy; progressive dementia; "
            "PATHOGNOMONIC FINDINGS: "
            "  NERVE BIOPSY: metachromatically-staining granules (sulfatide deposits); "
            "    Metachromatic = brown/orange with toluidine blue (not blue); PATHOGNOMONIC; "
            "  MRI BRAIN: confluent T2 hyperintensity white matter (tigroid pattern); "
            "    Frontal predominance + periventricular; sparing subcortical U-fibres initially; "
            "  NERVE CONDUCTION: slowed/absent NCV -- peripheral neuropathy early; "
            "  Urine sulfatides: elevated; "
            "LIBMELDY (ATIDARSAGENE AUTOTEMCEL): "
            "  Ex-vivo gene therapy: HSC transduced with lentiviral ARSA; "
            "  EMA approved December 2020; FDA approved June 2024; "
            "  Eligible: "
            "    Pre-symptomatic LATE-INFANTILE or JUVENILE MLD (NBS detected); "
            "    Early-symptomatic late-infantile before loss of ambulation; "
            "    Early-symptomatic juvenile before cognitive decline; "
            "  NOT eligible: symptomatic with significant neurological impairment; "
            "  TIMING IS CRITICAL: treat as early as possible; "
            "DIAGNOSIS: "
            "  ARSA enzyme activity: reduced (confirm with genotyping -- pseudodeficiency); "
            "  ARSA gene sequencing: identifies true affected vs pseudodeficiency; "
            "  Urine sulfatides: elevated in affected; "
            "  Brain MRI: white matter changes; "
            "  NCV: peripheral neuropathy; "
            "TREATMENT: "
            "  Libmeldy: EMA2020/FDA2024 -- presymptomatic/early-symptomatic ONLY; "
            "  HSCT: older option; inferior to Libmeldy for eligible patients; "
            "  Symptomatic: antiepileptics, physiotherapy, communication"
        ),
        "syndrome": (
            "Metachromatic-Leukodystrophy-MLD / ARSA-arylsulfatase-A / "
            "sulfatide-accumulation-PNS-AND-CNS / "
            "metachromatically-staining-granules-nerve-biopsy-PATHOGNOMONIC / "
            "Libmeldy-EMA2020-FDA2024-presymptomatic-early-symptomatic-ONLY / "
            "pseudodeficiency-gene-sequencing-mandatory"
        ),
        "inheritance": "AR (biallelic ARSA mutations; exclude pseudodeficiency allele)",
        "key_therapy": "Libmeldy (atidarsagene autotemcel) EMA2020/FDA2024 — EARLY TIMING CRITICAL — symptomatic support",
        "pathognomonic": (
            "Metachromatically-staining granules on nerve biopsy PATHOGNOMONIC (brown/orange toluidine blue); "
            "MRI: confluent T2 white matter hyperintensity (tigroid pattern) PATHOGNOMONIC; "
            "Urine sulfatides elevated = disease biomarker; "
            "Pseudodeficiency allele: ARSA enzyme low but NO disease -- gene sequencing MANDATORY"
        ),
        "key_drugs_ci": [
            "Libmeldy-atidarsagene-autotemcel-EMA2020-FDA2024",
            "Libmeldy-ONLY-presymptomatic-early-symptomatic",
            "ARSA-pseudodeficiency-gene-sequencing-MANDATORY",
            "HSCT-inferior-to-Libmeldy-for-eligible",
            "NBS-enables-presymptomatic-diagnosis-and-treatment",
        ],
    },
    {
        "gene": "GLB1",
        "protein": (
            "GLB1 -- 3p22.3 AR lysosomal LOF -- 677aa -- "
            "Beta-galactosidase-1-GM1-ganglioside-galactosylceramide-keratan-sulfate-"
            "GM1-gangliosidosis-AND-Morquio-B-MPS-IVB-SAME-GENE-"
            "OMIM-230500-253010-cherry-red-spot-dysostosis-TRIAD-GM1-infantile-"
            "keratan-sulfate-urine-elevated"
        ),
        "locus": "3p22.3",
        "protein_size": (
            "677 aa / ~75 kDa / 3p22.3 GLB1 beta-galactosidase 1 molecular context: "
            "STRUCTURE: "
            "  677 aa; lysosomal beta-galactosidase; "
            "  Cleaves: GM1 ganglioside (CNS) + keratan sulfate (cartilage) + galactosylceramide; "
            "  AR inheritance; biallelic GLB1 variants; "
            "  SAME GENE -- TWO DISTINCT DISEASES depending on mutation type: "
            "    GLB1 NULL mutations (complete loss): GM1 gangliosidosis; "
            "    GLB1 MISSENSE mutations (partial activity, residual keratan sulfatase): Morquio B; "
            "    Genotype determines phenotype -- all GLB1 panels must cover both; "
            "  Protein forms complex with protective protein/cathepsin A (PPCA); "
            "GM1 GANGLIOSIDOSIS -- THREE TYPES: "
            "  TYPE 1 (INFANTILE): "
            "    Onset: neonatal/first months; "
            "    TRIAD: CHERRY-RED SPOT (fundoscopy) + DYSOSTOSIS MULTIPLEX + NEURODEGENERATION PATHOGNOMONIC; "
            "    Facial coarsening; hepatosplenomegaly; macroglossia; "
            "    Macular cherry-red spot: ~50% Type 1 infantile; "
            "    Gibbus deformity (vertebral); periosteal new bone; "
            "    Death typically 1-2yr; "
            "  TYPE 2 (LATE-INFANTILE/JUVENILE): "
            "    Onset: 1-3yr; motor decline; ataxia; seizures; "
            "    NO cherry-red spot; less dysostosis; "
            "  TYPE 3 (ADULT/CHRONIC): "
            "    Extrapyramidal signs; dystonia; cognitive sparing relative to motor; "
            "MORQUIO B (MPS IVB): "
            "  Mutation: missense with residual beta-galactosidase; specific keratan sulfatase deficiency; "
            "  SKELETAL DISEASE DOMINANT: "
            "    Short stature; odontoid hypoplasia (C1-C2 instability CATASTROPHIC); "
            "    Genu valgum; pectus carinatum; "
            "    Dysostosis WITHOUT neurological involvement (in contrast to GM1); "
            "  Corneal clouding; "
            "  Urine keratan sulfate: elevated PATHOGNOMONIC for Morquio B; "
            "DIAGNOSIS: "
            "  Beta-galactosidase activity: reduced in both GM1 and Morquio B; "
            "  GLB1 gene sequencing: MANDATORY to distinguish GM1 vs Morquio B; "
            "  Urine oligosaccharides: galactosyl-oligomers (GM1); keratan sulfate (Morquio B); "
            "  Brain MRI: white matter changes, basal ganglia (GM1); "
            "  Skeletal X-ray: dysostosis multiplex (GM1 Type 1; Morquio B); "
            "TREATMENT: "
            "  GM1: no approved disease-modifying therapy; gene therapy trials; "
            "  Morquio B: elosulfase alfa (Vimizim) for MPS IVA (GALNS gene) NOT GLB1/Morquio B; "
            "    No ERT for Morquio B (GLB1); supportive; C1-C2 surgery if instability; "
            "  GM1 + Gene therapy: ongoing clinical trials"
        ),
        "syndrome": (
            "GM1-gangliosidosis-AND-Morquio-B-SAME-GENE / GLB1-beta-galactosidase / "
            "GM1-infantile: cherry-red-spot + dysostosis + neurodegeneration TRIAD PATHOGNOMONIC / "
            "Morquio-B: skeletal-dominant NO-CNS keratan-sulfate-urine-PATHOGNOMONIC / "
            "genotype-determines-phenotype-null-vs-missense"
        ),
        "inheritance": "AR (biallelic GLB1 mutations; null -> GM1; missense partial -> Morquio B)",
        "key_therapy": "GM1: no approved therapy — gene therapy trials — Morquio B: supportive + C1-C2 surveillance",
        "pathognomonic": (
            "GM1 infantile TRIAD: cherry-red spot + dysostosis multiplex + neurodegeneration PATHOGNOMONIC; "
            "Morquio B: urine keratan sulfate elevated PATHOGNOMONIC; "
            "SAME GENE GLB1 causes both GM1 AND Morquio B depending on mutation type; "
            "Odontoid hypoplasia Morquio B: C1-C2 instability surveillance MANDATORY"
        ),
        "key_drugs_ci": [
            "GM1-no-approved-ERT-gene-therapy-trials",
            "Morquio-B-NO-ERT-available-GLB1",
            "C1-C2-instability-surgery-Morquio-B-MANDATORY-surveillance",
            "Elosulfase-alfa-Vimizim-for-MPS-IVA-GALNS-NOT-GLB1",
            "GLB1-panel-MUST-cover-both-GM1-and-Morquio-B",
        ],
    },
    {
        "gene": "ASAH1",
        "protein": (
            "ASAH1 -- 8p22 AR lysosomal LOF -- 395aa -- "
            "Acid-ceramidase-ceramide-hydrolysis-N-acylsphingosine-amidohydrolase-"
            "Farber-lipogranulomatosis-Farber-disease-OMIM-228000-"
            "painful-nodules-joint-contractures-hoarseness-TRIAD-PATHOGNOMONIC-"
            "ceramide-accumulation-HSCT-somatic-NOT-CNS"
        ),
        "locus": "8p22",
        "protein_size": (
            "395 aa / ~44 kDa / 8p22 ASAH1 acid ceramidase molecular context: "
            "STRUCTURE: "
            "  395 aa; lysosomal acid ceramidase; cleaves ceramide -> sphingosine + fatty acid; "
            "  Precursor protein: 53 kDa; autocatalytic cleavage -> 13kDa alpha + 40kDa beta subunit; "
            "  Deficiency -> ceramide accumulation in lysosomes of all tissues; "
            "  AR inheritance; biallelic ASAH1 variants; "
            "  >50 variants described; missense most common; "
            "FARBER DISEASE -- CLASSIC TRIAD (PATHOGNOMONIC): "
            "  1. PAINFUL SUBCUTANEOUS NODULES (Farber granulomata): "
            "     Periarticular + subcutaneous; tender; histiocytic granulomata; "
            "     Progressive: multiple joints, tendons, digits; "
            "     CERAMIDE-CONTAINING CURVILINEAR BODIES (EM) in histiocytes PATHOGNOMONIC; "
            "  2. PROGRESSIVE JOINT CONTRACTURES: "
            "     Arthropathy; fixed flexion deformities; "
            "     Resembles JIA initially -- Farber MISDIAGNOSED as JIA; "
            "  3. HOARSENESS / STRIDOR: "
            "     Laryngeal granulomata -> voice changes in infancy; "
            "     Can cause respiratory compromise; "
            "  ALL THREE = FARBER TRIAD PATHOGNOMONIC; "
            "NEUROLOGICAL (TYPE-DEPENDENT): "
            "  Classic Type 1: severe neurological involvement; "
            "    Psychomotor retardation; seizures; death by 2-3yr; "
            "  Type 2/3: intermediate; somatic dominant; longer survival; "
            "  Type 5: primarily neurological (Faber SMA-PME); "
            "    ASAH1 mutations: spinal muscular atrophy-progressive myoclonic epilepsy (SMA-PME); "
            "    Distinct clinical syndrome; "
            "CERAMIDE ACCUMULATION IN TISSUES: "
            "  Histiocytic granulomata: liver, lungs, lymph nodes, CNS; "
            "  Foam cells in liver; hepatomegaly; "
            "DIAGNOSIS: "
            "  Acid ceramidase activity: reduced in leukocytes/fibroblasts; "
            "  ASAH1 gene sequencing; "
            "  Skin/subcutaneous nodule biopsy: "
            "    Histiocytic granulomata + ceramide curvilinear bodies on EM PATHOGNOMONIC; "
            "  Plasma ceramide: elevated (but less specific); "
            "TREATMENT: "
            "  HSCT: "
            "    IMPROVES: nodules, joint contractures, respiratory; "
            "    Does NOT improve: neurological disease; "
            "    CRITICAL: HSCT before severe neurological damage maximizes benefit; "
            "    HSCT in Types 2/3 (minimal CNS): good somatic outcomes; "
            "  Symptomatic: anti-inflammatory (temporary), physiotherapy; "
            "  No approved ERT for ASAH1; clinical trials ongoing; "
            "  SRT/substrate reduction: experimental"
        ),
        "syndrome": (
            "Farber-lipogranulomatosis / Farber-disease / ASAH1-acid-ceramidase / "
            "painful-nodules + joint-contractures + hoarseness-TRIAD-PATHOGNOMONIC / "
            "ceramide-curvilinear-bodies-EM-PATHOGNOMONIC / "
            "HSCT-improves-somatic-NOT-CNS / SMA-PME-variant"
        ),
        "inheritance": "AR (biallelic ASAH1 mutations)",
        "key_therapy": "HSCT (improves somatic disease — NOT neurological) — timing critical before CNS damage",
        "pathognomonic": (
            "FARBER TRIAD: painful periarticular nodules + progressive joint contractures + hoarseness = PATHOGNOMONIC; "
            "Ceramide-containing curvilinear bodies in histiocytes on EM PATHOGNOMONIC; "
            "HSCT improves nodules/joints/respiratory but NOT neurological disease; "
            "ASAH1 also causes SMA-PME (spinal muscular atrophy-progressive myoclonic epilepsy)"
        ),
        "key_drugs_ci": [
            "HSCT-improves-somatic-NOT-CNS-ASAH1",
            "HSCT-Types-2-3-minimal-CNS-best-outcomes",
            "No-ERT-approved-ASAH1-clinical-trials",
            "Farber-misdiagnosed-as-JIA-TRIAD-distinguishes",
        ],
    },
]


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _gene_cohort(gene_def, seed):
    rng = _rng(seed)
    gene = gene_def["gene"]

    # Gene-specific clinical profiles
    profiles = {
        "GBA": dict(
            age_range=(5, 50), cardiac_prob=0.05, neuro_prob=0.15,
            renal_prob=0.00, myopathy_prob=0.10, metabolic_crisis_prob=0.10,
            ert_treated_prob=0.70, transplant_prob=0.02,
            lsd_type="Gaucher-Type1-non-neuronopathic",
        ),
        "SMPD1": dict(
            age_range=(2, 45), cardiac_prob=0.20, neuro_prob=0.50,
            renal_prob=0.00, myopathy_prob=0.00, metabolic_crisis_prob=0.15,
            ert_treated_prob=0.35, transplant_prob=0.03,
            lsd_type="Niemann-Pick-Type-A-B-ASMD",
        ),
        "NPC1": dict(
            age_range=(5, 40), cardiac_prob=0.00, neuro_prob=0.98,
            renal_prob=0.00, myopathy_prob=0.00, metabolic_crisis_prob=0.00,
            ert_treated_prob=0.00, transplant_prob=0.00,
            lsd_type="Niemann-Pick-Type-C-neurological",
        ),
        "HEXA": dict(
            age_range=(0, 5), cardiac_prob=0.00, neuro_prob=0.99,
            renal_prob=0.00, myopathy_prob=0.00, metabolic_crisis_prob=0.00,
            ert_treated_prob=0.00, transplant_prob=0.00,
            lsd_type="Tay-Sachs-infantile",
        ),
        "GALC": dict(
            age_range=(0, 3), cardiac_prob=0.00, neuro_prob=0.98,
            renal_prob=0.00, myopathy_prob=0.00, metabolic_crisis_prob=0.00,
            ert_treated_prob=0.00, transplant_prob=0.00,
            lsd_type="Krabbe-globoid-cell-leukodystrophy",
        ),
        "ARSA": dict(
            age_range=(1, 35), cardiac_prob=0.00, neuro_prob=0.98,
            renal_prob=0.00, myopathy_prob=0.10, metabolic_crisis_prob=0.00,
            ert_treated_prob=0.30, transplant_prob=0.00,
            lsd_type="MLD-metachromatic-leukodystrophy",
        ),
        "GLB1": dict(
            age_range=(0, 20), cardiac_prob=0.00, neuro_prob=0.60,
            renal_prob=0.00, myopathy_prob=0.00, metabolic_crisis_prob=0.00,
            ert_treated_prob=0.00, transplant_prob=0.00,
            lsd_type="GM1-gangliosidosis-Morquio-B",
        ),
        "ASAH1": dict(
            age_range=(0, 10), cardiac_prob=0.00, neuro_prob=0.60,
            renal_prob=0.00, myopathy_prob=0.00, metabolic_crisis_prob=0.00,
            ert_treated_prob=0.00, transplant_prob=0.00,
            lsd_type="Farber-lipogranulomatosis",
        ),
    }
    p = profiles[gene]

    ages = sorted([
        rng.uniform(p["age_range"][0], p["age_range"][1])
        for _ in range(40)
    ])
    mean_age = round(sum(ages) / 40, 1)

    cardiac_n = sum(1 for _ in range(40) if rng.random() < p["cardiac_prob"])
    neuro_n = sum(1 for _ in range(40) if rng.random() < p["neuro_prob"])
    renal_n = sum(1 for _ in range(40) if rng.random() < p["renal_prob"])
    myopathy_n = sum(1 for _ in range(40) if rng.random() < p["myopathy_prob"])
    metabolic_crisis_n = sum(1 for _ in range(40) if rng.random() < p["metabolic_crisis_prob"])
    ert_treated_n = sum(1 for _ in range(40) if rng.random() < p["ert_treated_prob"])
    transplant_n = sum(1 for _ in range(40) if rng.random() < p["transplant_prob"])

    # Gene-specific phenotypes
    splenomegaly_n = 0
    bone_crisis_n = 0
    parkinson_n = 0
    sea_blue_histiocytes_n = 0
    lung_infiltrates_n = 0
    vsgp_n = 0
    cataplexy_n = 0
    cherry_red_spot_n = 0
    psychosine_n = 0
    hsct_n = 0
    libmeldy_n = 0
    peripheral_neuropathy_n = 0
    keratan_sulfate_n = 0
    nodules_n = 0
    joint_contractures_n = 0
    hoarseness_n = 0

    if gene == "GBA":
        splenomegaly_n = sum(1 for _ in range(40) if rng.random() < 0.85)
        bone_crisis_n = sum(1 for _ in range(40) if rng.random() < 0.45)
        parkinson_n = sum(1 for _ in range(40) if rng.random() < 0.10)
    elif gene == "SMPD1":
        sea_blue_histiocytes_n = sum(1 for _ in range(40) if rng.random() < 0.80)
        lung_infiltrates_n = sum(1 for _ in range(40) if rng.random() < 0.65)
    elif gene == "NPC1":
        vsgp_n = sum(1 for _ in range(40) if rng.random() < 0.85)
        cataplexy_n = sum(1 for _ in range(40) if rng.random() < 0.35)
    elif gene == "HEXA":
        cherry_red_spot_n = sum(1 for _ in range(40) if rng.random() < 0.90)
    elif gene == "GALC":
        psychosine_n = sum(1 for _ in range(40) if rng.random() < 0.90)
        hsct_n = sum(1 for _ in range(40) if rng.random() < 0.25)
    elif gene == "ARSA":
        libmeldy_n = sum(1 for _ in range(40) if rng.random() < 0.20)
        peripheral_neuropathy_n = sum(1 for _ in range(40) if rng.random() < 0.85)
    elif gene == "GLB1":
        cherry_red_spot_n = sum(1 for _ in range(40) if rng.random() < 0.40)
        keratan_sulfate_n = sum(1 for _ in range(40) if rng.random() < 0.70)
    elif gene == "ASAH1":
        nodules_n = sum(1 for _ in range(40) if rng.random() < 0.95)
        joint_contractures_n = sum(1 for _ in range(40) if rng.random() < 0.92)
        hoarseness_n = sum(1 for _ in range(40) if rng.random() < 0.88)
        hsct_n = sum(1 for _ in range(40) if rng.random() < 0.30)

    return {
        "gene": gene,
        "n": 40,
        "mean_age_dx": mean_age,
        "cardiac_n": cardiac_n,
        "neuropathy_n": neuro_n,
        "renal_n": renal_n,
        "myopathy_n": myopathy_n,
        "metabolic_crisis_n": metabolic_crisis_n,
        "ert_treated_n": ert_treated_n,
        "transplant_n": transplant_n,
        "splenomegaly_n": splenomegaly_n,
        "bone_crisis_n": bone_crisis_n,
        "parkinson_n": parkinson_n,
        "sea_blue_histiocytes_n": sea_blue_histiocytes_n,
        "lung_infiltrates_n": lung_infiltrates_n,
        "vsgp_n": vsgp_n,
        "cataplexy_n": cataplexy_n,
        "cherry_red_spot_n": cherry_red_spot_n,
        "psychosine_n": psychosine_n,
        "hsct_n": hsct_n,
        "libmeldy_n": libmeldy_n,
        "peripheral_neuropathy_n": peripheral_neuropathy_n,
        "keratan_sulfate_n": keratan_sulfate_n,
        "nodules_n": nodules_n,
        "joint_contractures_n": joint_contractures_n,
        "hoarseness_n": hoarseness_n,
        "lsd_type": p["lsd_type"],
    }


def generate_overview():
    genes = [g["gene"] for g in ATLAS_GENES]
    per_gene = {}
    for i, gd in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        cohort = _gene_cohort(gd, seed)
        per_gene[gd["gene"]] = cohort

    total_patients = sum(c["n"] for c in per_gene.values())

    return {
        "atlas": "Hereditary-Lysosomal-Storage-Disease-Atlas",
        "subtitle": (
            "Complete 8-Gene Reference: GBA/Gaucher · SMPD1/Niemann-Pick-A-B · NPC1/Niemann-Pick-C · "
            "HEXA/Tay-Sachs · GALC/Krabbe · ARSA/MLD · GLB1/GM1-Morquio-B · ASAH1/Farber"
        ),
        "genes": genes,
        "total_patients": total_patients,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "per_gene_summary": per_gene,
        "key_clinical_themes": {
            "ert_available": ["GBA (imiglucerase/velaglucerase)", "SMPD1-TypeB (olipudase alfa FDA2022)"],
            "srt_available": ["GBA (eliglustat FDA2014 / miglustat)", "NPC1 (miglustat progression slowing)"],
            "gene_therapy_approved": ["ARSA (Libmeldy EMA2020/FDA2024 presymptomatic/early-symptomatic)"],
            "hsct_effective_presymptomatic": ["GALC (NBS critical)", "ASAH1 (somatic only)"],
            "no_disease_modifying_therapy": ["HEXA (Tay-Sachs)", "GLB1-GM1 (gene therapy trials)"],
            "pathognomonic_eye_findings": [
                "HEXA (cherry-red spot fundoscopy)",
                "SMPD1-TypeA (cherry-red spot ~50%)",
                "GLB1-GM1 (cherry-red spot infantile)",
                "NPC1 (VSGP PATHOGNOMONIC)",
            ],
            "parkinson_risk": ["GBA heterozygotes (5x increased risk)"],
            "same_gene_two_diseases": ["GLB1 (GM1 gangliosidosis + Morquio B)"],
        },
    }


def generate_breakdown():
    gene_data = {}
    for i, gd in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        gene_data[gd["gene"]] = _gene_cohort(gd, seed)

    return {
        "atlas": "Hereditary-Lysosomal-Storage-Disease-Atlas",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "lsd_type_map": {g["gene"]: _gene_cohort(g, SEED_BASE + i)["lsd_type"]
                         for i, g in enumerate(ATLAS_GENES)},
        "inheritance_map": {g["gene"]: g["inheritance"] for g in ATLAS_GENES},
        "therapy_curative": {
            "GBA": "ERT (imiglucerase/velaglucerase 60 U/kg biweekly) — eliglustat SRT FDA2014 (CYP2D6 required)",
            "SMPD1": "Olipudase alfa FDA2022 TYPE B ONLY — Type A: supportive/palliative",
            "NPC1": "Miglustat slows neurological progression — not curative",
            "HEXA": "No disease-modifying therapy — supportive/palliative — gene therapy trials",
            "GALC": "HSCT ONLY presymptomatic NBS-detected — gene therapy trials",
            "ARSA": "Libmeldy EMA2020/FDA2024 presymptomatic/early-symptomatic ONLY",
            "GLB1": "GM1: no approved therapy — Morquio B: supportive + C1-C2 surveillance",
            "ASAH1": "HSCT improves somatic (nodules/joints) NOT neurological — no ERT",
        },
    }


def generate_definitions():
    return {
        "atlas": "Hereditary-Lysosomal-Storage-Disease-Atlas",
        "definitions": {
            "Gaucher disease": (
                "AR GBA deficiency; glucosylceramide accumulation in macrophage lysosomes; "
                "Type 1 (94%): splenomegaly + thrombocytopenia + bone crises TRIAD; "
                "Erlenmeyer-flask deformity radiograph PATHOGNOMONIC; "
                "ERT (imiglucerase/velaglucerase) reverses splenomegaly; eliglustat SRT FDA2014; "
                "GBA HETEROZYGOTES: 5x Parkinson risk; GBA most common genetic PD risk factor"
            ),
            "Niemann-Pick Type A/B (ASMD)": (
                "AR SMPD1/acid sphingomyelinase deficiency; sphingomyelin accumulation in histiocytes; "
                "Type A: severe neuronopathic (<1% ASM); infantile fatal; cherry-red spot ~50%; "
                "Type B: visceral non-neuronopathic (1-10% ASM); hepatosplenomegaly + ILD + dyslipidemia; "
                "Sea-blue histiocytes BM biopsy PATHOGNOMONIC; "
                "Olipudase alfa (XENPOZYME) ERT FDA2022 for TYPE B ONLY"
            ),
            "Niemann-Pick Type C": (
                "AR NPC1 deficiency; intracellular cholesterol trafficking defect; "
                "VSGP (vertical supranuclear gaze palsy) = PATHOGNOMONIC; "
                "Gelastic cataplexy (laugh-triggered atonia) = PATHOGNOMONIC; "
                "Filipin staining fibroblasts: massive perinuclear cholesterol fluorescence PATHOGNOMONIC; "
                "Progressive cerebellar ataxia + dementia + psychiatric; "
                "Miglustat slows neurological progression; arimoclomol trials"
            ),
            "Tay-Sachs disease": (
                "AR HEXA deficiency; HexA absent; GM2 ganglioside accumulates selectively in neurons; "
                "Cherry-red spot on fundoscopy PATHOGNOMONIC (~100% infantile); "
                "Hyperacusis (exaggerated sound startle) early sign; "
                "Ashkenazi Jewish carrier frequency 1/30; "
                "No ERT available; supportive/palliative; gene therapy trials; "
                "Late-onset LOTS: ataxia + psychiatric + proximal weakness"
            ),
            "Krabbe disease": (
                "AR GALC deficiency; psychosine (galactosylsphingosine) toxic metabolite accumulates; "
                "Psychosine = oligodendrocyte cytotoxin = demyelination; "
                "TRIAD: extreme irritability + hypertonia + developmental regression PATHOGNOMONIC; "
                "Globoid cells (multinucleated macrophages) in white matter PATHOGNOMONIC; "
                "HSCT effective ONLY presymptomatic (NBS critical); "
                "After symptom onset: HSCT has NO neurological benefit"
            ),
            "Metachromatic Leukodystrophy (MLD)": (
                "AR ARSA deficiency; sulfatide (galactosylceramide-3-sulfate) accumulates in PNS + CNS; "
                "Metachromatically-staining granules on nerve biopsy PATHOGNOMONIC (brown/orange toluidine blue); "
                "MRI: confluent T2 white matter hyperintensity (tigroid pattern); "
                "Pseudodeficiency allele: low enzyme with NO disease -- gene sequencing MANDATORY; "
                "Libmeldy (atidarsagene autotemcel) EMA2020/FDA2024 "
                "presymptomatic/early-symptomatic ONLY -- timing critical"
            ),
            "GM1 gangliosidosis / Morquio B": (
                "AR GLB1 (beta-galactosidase 1) deficiency; SAME GENE causes two distinct diseases: "
                "GM1 gangliosidosis (null mutations): TRIAD cherry-red spot + dysostosis multiplex + neurodegeneration; "
                "Morquio B / MPS IVB (missense with residual activity): skeletal dominant, NO CNS, keratan sulfate elevated; "
                "Odontoid hypoplasia Morquio B: C1-C2 instability surveillance MANDATORY; "
                "No approved ERT for either; gene therapy clinical trials"
            ),
            "Farber disease": (
                "AR ASAH1/acid ceramidase deficiency; ceramide accumulates in lysosomes; "
                "PATHOGNOMONIC TRIAD: painful periarticular subcutaneous nodules + progressive joint contractures "
                "+ hoarseness (laryngeal granulomata); "
                "Farber routinely misdiagnosed as JIA; "
                "Ceramide curvilinear bodies on EM PATHOGNOMONIC; "
                "HSCT improves nodules/joints/respiratory but NOT neurological disease; "
                "ASAH1 also causes SMA-PME (spinal muscular atrophy-progressive myoclonic epilepsy)"
            ),
            "Lysosomal Storage Disease (LSD)": (
                "Group of metabolic disorders from lysosomal enzyme deficiency -> substrate accumulation in lysosomes; "
                "~70 recognized LSDs; combined incidence ~1/7000 births; "
                "Presentation: organomegaly, neurological, skeletal, ophthalmological findings; "
                "Newborn screening (NBS) critical for treatable LSDs: Gaucher, Krabbe, MLD, Pompe, Fabry; "
                "Enzyme assay + gene sequencing: complementary diagnostic tools"
            ),
            "Filipin staining": (
                "Filipin: fluorescent compound that binds unesterified free cholesterol; "
                "Used on cultured fibroblasts from NPC patients; "
                "Classic NPC1: massive perinuclear fluorescence = unesterified cholesterol trapped in lysosomes; "
                "Variant NPC: partial staining; must follow up with NPC1/NPC2 sequencing; "
                "PATHOGNOMONIC for NPC1/NPC2 diagnosis"
            ),
            "Psychosine (galactosylsphingosine)": (
                "Toxic metabolite accumulating in GALC (Krabbe) deficiency; "
                "Lacks ceramide moiety: cannot be stored, directly cytotoxic; "
                "Destroys oligodendrocytes -> demyelination -> globoid cell reaction; "
                "Elevated in blood/DBS at birth in NBS programs; "
                "Plasma psychosine: prognostic biomarker (level correlates with disease pace)"
            ),
            "Enzyme Replacement Therapy (ERT) in LSD": (
                "IV infusion of recombinant lysosomal enzyme; "
                "Available for: Gaucher (imiglucerase/velaglucerase/taliglucerase), "
                "Pompe (alglucosidase/avalglucosidase), Fabry (agalsidase alfa/beta), "
                "Niemann-Pick B/ASMD (olipudase alfa FDA2022); "
                "Does NOT cross blood-brain barrier: no CNS benefit; "
                "Must be given before irreversible organ damage for maximum benefit"
            ),
        },
        "critical_clinical_rules": {
            "GBA-SPLENOMEGALY-TRIAD-PATHOGNOMONIC": (
                "Gaucher disease (GBA): splenomegaly + thrombocytopenia + bone crises = TRIAD PATHOGNOMONIC; "
                "ERT reverses splenomegaly, thrombocytopenia, bone marrow infiltration; "
                "SPLENECTOMY: now AVOIDED -- ERT effective; splenectomy removes Gaucher cell pool; "
                "SRT (eliglustat): CYP2D6 metabolizer status MANDATORY before prescribing; "
                "GBA HETEROZYGOTES: 5x Parkinson risk -- SCREEN for PD in all GBA carriers"
            ),
            "SMPD1-OLIPUDASE-TYPE-B-ONLY": (
                "Niemann-Pick B (SMPD1): olipudase alfa (XENPOZYME) FDA2022 = TYPE B NON-NEURONOPATHIC ONLY; "
                "Sea-blue histiocytes BM biopsy = PATHOGNOMONIC (always obtain BM if suspected ASMD); "
                "Type A: NO disease-modifying therapy -- IV ERT does NOT reach CNS; "
                "Coronary artery disease risk markedly elevated in Type B: aggressive lipid management mandatory; "
                "Lyso-sphingomyelin: plasma biomarker for diagnosis and monitoring"
            ),
            "NPC1-VSGP-CATAPLEXY-PATHOGNOMONIC": (
                "Niemann-Pick Type C (NPC1): VSGP (vertical supranuclear gaze palsy) = PATHOGNOMONIC; "
                "Gelastic cataplexy (laugh-triggered sudden atonia) = PATHOGNOMONIC NPC1; "
                "ANY young patient with ataxia + psychiatric symptoms + VSGP = NPC1 UNTIL PROVEN OTHERWISE; "
                "Filipin staining fibroblasts = DIAGNOSTIC GOLD STANDARD; "
                "Plasma oxysterols (7-ketocholesterol) and lysosphingomyelin-509 = NPC biomarkers; "
                "Miglustat: slows but does NOT stop neurological progression"
            ),
            "HEXA-NO-ERT-CHERRY-RED-SPOT": (
                "Tay-Sachs (HEXA): NO disease-modifying ERT available; supportive/palliative only; "
                "Cherry-red spot on fundoscopy PATHOGNOMONIC in infantile (~100%); "
                "Ashkenazi Jewish carrier frequency 1/30 -- carrier screening MANDATORY; "
                "HexA/total hexosaminidase ratio <10% = diagnostic; "
                "Distinguish from Sandhoff (HEXB): both HexA and HexB absent; similar phenotype"
            ),
            "GALC-HSCT-PRESYMPTOMATIC-ONLY": (
                "Krabbe disease (GALC): HSCT effective ONLY presymptomatic; "
                "NBS detection (GALC enzyme + psychosine DBS) CRITICAL; "
                "Cord blood transplant before symptom onset: significantly better motor/neurological outcomes; "
                "After symptom onset: HSCT does NOT prevent neurological decline -- too late; "
                "Psychosine elevated at birth: DBS psychosine most sensitive NBS marker"
            ),
            "ARSA-LIBMELDY-TIMING-CRITICAL": (
                "MLD (ARSA): Libmeldy (atidarsagene autotemcel) EMA2020/FDA2024 = "
                "PRESYMPTOMATIC or EARLY-SYMPTOMATIC ONLY; "
                "TIMING IS CRITICAL: treat as early as possible before neurological damage; "
                "Pseudodeficiency allele (ARSA*PD): LOW ENZYME BUT NO DISEASE -- "
                "gene sequencing MANDATORY before labeling as 'affected'; "
                "Metachromatically-staining granules nerve biopsy (brown/orange toluidine blue) PATHOGNOMONIC"
            ),
            "GLB1-SAME-GENE-TWO-DISEASES": (
                "GLB1: SAME GENE causes GM1 gangliosidosis (null mutations) AND Morquio B (missense partial activity); "
                "GM1 infantile TRIAD: cherry-red spot + dysostosis multiplex + neurodegeneration = PATHOGNOMONIC; "
                "Morquio B: skeletal dominant; urine keratan sulfate elevated PATHOGNOMONIC; "
                "C1-C2 instability Morquio B: SURVEILLANCE MANDATORY (odontoid hypoplasia -> cord compression); "
                "Elosulfase alfa (Vimizim): for MPS IVA (GALNS gene) -- NOT for Morquio B (GLB1)"
            ),
            "ASAH1-FARBER-TRIAD-HSCT-SOMATIC-ONLY": (
                "Farber disease (ASAH1): PATHOGNOMONIC TRIAD = painful nodules + joint contractures + hoarseness; "
                "Farber routinely MISDIAGNOSED as JIA or other arthritis -- TRIAD distinguishes; "
                "HSCT: improves nodules, joint contractures, respiratory disease; "
                "HSCT: does NOT improve neurological disease; "
                "TIMING: HSCT before severe neurological damage for maximum somatic benefit; "
                "ASAH1 also causes SMA-PME (distinct from classic Farber -- check ASAH1 in all unexplained SMA-PME)"
            ),
            "LSD-NEWBORN-SCREENING-CRITICAL": (
                "Newborn screening for LSDs: enables presymptomatic treatment for treatable conditions; "
                "Key NBS-actionable LSDs: Gaucher (GBA), Krabbe (GALC, psychosine DBS), MLD (ARSA), "
                "Pompe (GAA), Fabry (GLA); "
                "NBS positive: CONFIRM with enzyme assay + gene sequencing before treating; "
                "Pseudodeficiency (ARSA, GLA in females): NBS can be positive -- gene sequencing mandatory; "
                "Missing NBS window (no NBS program): late diagnosis often after irreversible damage"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Lysosomal Storage Disease Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(
            f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, "
            f"neuro={d['neuropathy_n']}/40, "
            f"ert={d['ert_treated_n']}/40"
        )
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
