#!/usr/bin/env python3
"""Hereditary-TSC-mTOR-Pathway-Atlas — Complete 8-Gene TSC/mTOR Pathway Atlas
TSC1   (hamartin; 1164 aa; 9q34.13; AD LOF;
         Tuberous Sclerosis Complex type 1;
         CORTICAL TUBERS + SUBEPENDYMAL NODULES + ANGIOMYOLIPOMATA PATHOGNOMONIC triad;
         Epilepsy 85%; SEGA 5-15%; mTORC1 hyperactivation;
         HAMARTIN/TUBERIN heterodimer inhibits mTORC1;
         seed SEED_BASE+0) ·
TSC2   (tuberin; 1807 aa; 16p13.3; AD LOF;
         Tuberous Sclerosis Complex type 2;
         MORE SEVERE than TSC1 — more tubers, more refractory epilepsy, lower IQ;
         POLYCYSTIC KIDNEYS + RENAL AML; EVEROLIMUS FIRST-LINE for SEGA + LAM;
         Cardiac rhabdomyomata neonatal;
         seed SEED_BASE+1) ·
DEPDC5 (DEP domain containing 5; 1604 aa; 22q12.3; AD LOF;
         GATOR1 complex component;
         FAMILIAL FOCAL EPILEPSY WITH VARIABLE FOCI (FFEVF);
         SUDDEN UNEXPECTED DEATH IN EPILEPSY (SUDEP) RISK ELEVATED;
         somatic 2nd hit in resected tissue;
         seed SEED_BASE+2) ·
NPRL2  (nitrogen permease regulator-like 2; 441 aa; 17p13.1; AD LOF;
         GATOR1 complex component; Focal epilepsy;
         FAMILIAL FOCAL EPILEPSY; Autism 30%; SUDEP risk;
         seed SEED_BASE+3) ·
NPRL3  (nitrogen permease regulator-like 3; 489 aa; 16p13.3; AD LOF;
         GATOR1 complex component;
         AUTOSOMAL DOMINANT NOCTURNAL FRONTAL LOBE EPILEPSY (ADNFLE)-like;
         SUDEP risk; co-localised with TSC2 16p13.3;
         seed SEED_BASE+4) ·
MTOR   (mechanistic target of rapamycin; 2549 aa; 1p36.22; SOMATIC GOF brain only;
         Focal Cortical Dysplasia type IIb (FCD IIb);
         BALLOON CELLS PATHOGNOMONIC on histology;
         DEEP SEQUENCING 500x MANDATORY;
         seed SEED_BASE+5) ·
PIK3R2 (phosphoinositide-3-kinase regulatory subunit beta; 724 aa; 19p13.11;
         SOMATIC GOF brain only mosaic;
         MEGALENCEPHALY-CAPILLARY MALFORMATION-POLYMICROGYRIA (MCAP);
         HEMISPHERE ASYMMETRY PATHOGNOMONIC on MRI;
         seed SEED_BASE+6) ·
AKT3   (AKT serine/threonine kinase 3; 479 aa; 1q44; SOMATIC GOF brain only mosaic;
         HEMIMEGALENCEPHALY (HME);
         MEGALENCEPHALY-POLYMICROGYRIA-POLYDACTYLY-HYDROCEPHALUS (MPPH);
         HEMISPHERE ASYMMETRY MOST EXTREME FORM PATHOGNOMONIC;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 3062-3069)
"""
import random

SEED_BASE = 3062

ATLAS_GENES = [
    {
        "gene": "TSC1",
        "protein": (
            "TSC1 -- 9q34.13 Autosomal-Dominant-LOF -- 1164aa -- Hamartin-"
            "TSC1-TSC2-Heterodimer-mTORC1-Suppressor-Tuberous-Sclerosis-Complex-1-"
            "CORTICAL-TUBERS-SUBEPENDYMAL-NODULES-AML-PATHOGNOMONIC-TRIAD-SEGA-EVEROLIMUS-OMIM-605284"
        ),
        "locus": "9q34.13",
        "protein_size": (
            "1164 aa / 130 kDa (TSC1; hamartin; "
            "STRUCTURE: coiled-coil domain (C-terminal) + ezrin/radixin/moesin (ERM)-like N-terminal domain; "
            "FUNCTION: "
            "  TSC1 forms obligate heterodimer with TSC2 (tuberin) — HAMARTIN/TUBERIN complex; "
            "  Complex inhibits RHEB (Ras homolog enriched in brain) via GAP (GTPase activating protein) activity of TSC2; "
            "  RHEB-GDP (inactive) → does NOT activate mTORC1; "
            "  Loss of TSC1 → RHEB-GTP constitutively active → mTORC1 hyperactivation; "
            "  mTORC1 hyperactivation → excessive protein synthesis, cell growth, proliferation; "
            "  TSC1 also regulates autophagy (via mTORC1) and cell polarity; "
            "TUBEROUS SCLEROSIS COMPLEX TYPE 1 (TSC1): "
            "  PREVALENCE: ~1:6,000 live births (combined TSC1+TSC2); TSC1 ~30% of cases; "
            "  CLASSIC PATHOGNOMONIC TRIAD: "
            "    (1) CORTICAL TUBERS: focal malformations of cortical development; "
            "      Dysplastic cells (giant cells, dysmorphic neurons); "
            "      Epileptogenic zone — correlate with epilepsy severity; "
            "      Present in 80-95% TSC; "
            "    (2) SUBEPENDYMAL NODULES (SEN): calcified nodules lining lateral ventricles; "
            "      May transform to SEGA; present in 80-95%; "
            "    (3) ANGIOMYOLIPOMATA (AML): renal tumours (fat/smooth muscle/blood vessels); "
            "      Present in 55-75%; haemorrhage risk if >3 cm (Wunderlich syndrome); "
            "  EPILEPSY: 85% — most common presenting feature; "
            "    Infantile spasms (West syndrome) — most common onset; "
            "    Focal seizures; Lennox-Gastaut; drop attacks; "
            "    Drug-resistant in majority; "
            "  SUBEPENDYMAL GIANT CELL ASTROCYTOMA (SEGA): "
            "    5-15% of TSC patients; "
            "    Growth of SEN near foramen of Monro → obstructive hydrocephalus; "
            "    EVEROLIMUS FDA APPROVED for SEGA — disease-modifying (tumour reduction); "
            "    Surgical resection if symptomatic; "
            "  INTELLECTUAL DISABILITY: 50% — range from normal IQ to severe; "
            "  AUTISM: 50% of TSC with ID; "
            "  TSC-ASSOCIATED NEUROPSYCHIATRIC DISORDERS (TAND): "
            "    Autism, ADHD, anxiety, sleep disorders, learning disability; "
            "  SKIN MANIFESTATIONS: "
            "    Hypomelanotic macules (ash-leaf spots): 90% — UV light required; FIRST sign; "
            "    Facial angiofibromas: 75% — red papules nasolabial folds; "
            "    Shagreen patches: connective tissue naevi (lumbosacral); "
            "    Periungual fibromas (Koenen tumours): fingers/toes; "
            "  PULMONARY: "
            "    Lymphangioleiomyomatosis (LAM): females predominantly; cystic lung destruction; "
            "    Everolimus FDA approved for LAM; "
            "  CARDIAC: "
            "    Rhabdomyomata: 50% neonates — usually regress spontaneously; "
            "    May cause arrhythmia or outflow tract obstruction; "
            "    Echo MANDATORY in neonates with TSC; "
            "  RENAL: "
            "    AML — everolimus reduces haemorrhage risk; embolisation if acute bleed; "
            "    Polycystic kidney disease (PKD): uncommon in TSC1 (more TSC2); "
            "TSC1 vs TSC2 SEVERITY: "
            "  TSC1 MILDER overall: fewer/smaller tubers; better cognitive outcomes; "
            "  Less refractory epilepsy than TSC2; "
            "  LAM rare in TSC1 females (vs common TSC2); "
            "TREATMENT: "
            "  EVEROLIMUS (mTORC1 inhibitor): "
            "    SEGA: FDA approved 2010 — reduces SEGA volume; "
            "    AML: FDA approved 2012 — reduces renal AML volume; "
            "    LAM: FDA approved 2012; "
            "    Off-label: epilepsy (significant evidence — EXIST-3 trial); "
            "  VIGABATRIN: FIRST-LINE for TSC infantile spasms (specific recommendation); "
            "  Rapamycin (sirolimus): alternative mTORC1 inhibitor; similar efficacy; "
            "SURVEILLANCE PROTOCOL: "
            "  Brain MRI: 1-3 yearly (SEGA monitoring); "
            "  Renal USS: annually; "
            "  EEG: as clinically indicated; "
            "  ECG/Echo: neonatal; "
            "  Ophthalmology: retinal hamartomata; "
            "  Skin: dermatology; "
            "  Neuropsychology: TAND assessment annually; "
            "TESTING: TSC1 + TSC2 sequencing + MLPA; NGS panel; "
            "  ~15% TSC: no mutation found (NMF) — likely deep intronic/mosaic; "
        ),
        "inheritance": (
            "Autosomal dominant LOF (de novo ~60%; familial ~40%); "
            "TSC1 9q34.13; germline + somatic mosaicism documented; "
            "Two-hit model: germline TSC1 LOF + somatic second hit in tumour/tuber cells; "
            "Variable expressivity: same family members may differ greatly; "
        ),
        "disease_category": "Tuberous Sclerosis Complex type 1 / mTORopathy",
        "key_mutations": [
            "p.Arg692Ter (common nonsense)",
            "p.Arg611Ter (nonsense)",
            "p.Gln801Ter (nonsense coiled-coil)",
            "p.Leu939Pro (missense)",
            "Large deletions (exon 1-23 — MLPA)",
            "c.1525+1G>A (splice site)",
            "p.Ser584Leu (missense ERM domain)",
        ],
        "clinical_keys": [
            "CORTICAL TUBERS + SUBEPENDYMAL NODULES + RENAL AML PATHOGNOMONIC TRIAD",
            "Vigabatrin FIRST-LINE for TSC infantile spasms (specific recommendation)",
            "SEGA (5-15%): EVEROLIMUS FDA approved — disease-modifying tumour reduction",
            "Everolimus also approved for renal AML + LAM",
            "Brain MRI 1-3 yearly for SEGA monitoring — foramen of Monro location",
            "TSC1 MILDER than TSC2 — fewer tubers, better IQ, less refractory epilepsy",
            "Ash-leaf spots UV light (Wood's lamp) — FIRST skin sign in neonates",
            "TAND (TSC-associated neuropsychiatric disorders): autism + ADHD — screen annually",
        ],
    },
    {
        "gene": "TSC2",
        "protein": (
            "TSC2 -- 16p13.3 Autosomal-Dominant-LOF -- 1807aa -- Tuberin-"
            "GAP-RHEB-mTORC1-Suppressor-Tuberous-Sclerosis-Complex-2-"
            "MORE-SEVERE-MORE-TUBERS-REFRACTORY-EPILEPSY-POLYCYSTIC-KIDNEYS-AML-EVEROLIMUS-OMIM-191092"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "1807 aa / 200 kDa (TSC2; tuberin; "
            "STRUCTURE: N-terminal coiled-coil domain + central HEAT repeats + C-terminal GAP domain (Rap1-GAP homology); "
            "FUNCTION: "
            "  GAP activity directed at RHEB — converts RHEB-GTP to RHEB-GDP (inactive); "
            "  TSC2 is the catalytic subunit of the hamartin/tuberin complex; "
            "  Loss of TSC2 → RHEB permanently GTP-loaded → constitutive mTORC1 activation; "
            "  TSC2 also interacts with calmodulin, AKT, AMPK — integrates energy/growth signals; "
            "  TSC2 more critical than TSC1 for mTORC1 suppression (GAP domain on TSC2); "
            "TUBEROUS SCLEROSIS COMPLEX TYPE 2 (TSC2): "
            "  PREVALENCE: TSC2 accounts for ~70% of TSC cases; "
            "  MORE SEVERE THAN TSC1 — clinically and molecularly: "
            "    More and larger cortical tubers; "
            "    More refractory epilepsy; "
            "    Lower IQ (higher rate of intellectual disability); "
            "    Higher rate of infantile spasms; "
            "    More severe autism and TAND; "
            "  SAME TRIAD AS TSC1 BUT MORE SEVERE EXPRESSION: "
            "    CORTICAL TUBERS: more numerous; larger; higher epileptogenic burden; "
            "    SUBEPENDYMAL NODULES: more numerous; higher SEGA risk; "
            "    RENAL AML: bilateral; larger; more bleeding events; "
            "  POLYCYSTIC KIDNEYS: "
            "    TSC2 gene is adjacent to PKD1 on 16p13.3; "
            "    Contiguous gene deletion (TSC2+PKD1): SEVERE POLYCYSTIC KIDNEY DISEASE; "
            "    Screen renal USS annually; cystic kidneys in addition to AML; "
            "  CARDIAC RHABDOMYOMATA: "
            "    50%+ of TSC neonates — especially TSC2; "
            "    Majority regress spontaneously by 2 years; "
            "    Echo neonatal MANDATORY; "
            "    If large: outflow obstruction (subaortic/pulmonary); "
            "  LAM (LYMPHANGIOLEIOMYOMATOSIS): "
            "    Predominantly TSC2 females; "
            "    Progressive cystic lung destruction; "
            "    Spontaneous pneumothorax risk; "
            "    Everolimus FDA approved — stabilises lung function; "
            "  EPILEPSY IN TSC2: "
            "    More refractory than TSC1; "
            "    Infantile spasms higher rate; "
            "    Drug-resistant majority; "
            "    Vigabatrin first-line for infantile spasms; "
            "    Everolimus (EXIST-3 trial): ~40% seizure reduction adjunctive; "
            "  RETINAL HAMARTOMATA: 30-50% (phakomas); "
            "  BRAIN MRI ADDITIONAL: "
            "    Radial migration lines; subependymal nodules larger; "
            "    Cerebellar tubers (15%); "
            "    White matter lesions; "
            "GENOTYPE-PHENOTYPE: "
            "  TSC2 missense mutations (GAP domain): variable severity; "
            "  TSC2 truncating/LOF: typically severe; "
            "  TSC2+PKD1 contiguous deletion: severe polycystic kidney; "
            "  De novo mutations: typically severe (selected against in familial); "
            "TREATMENT: "
            "  EVEROLIMUS: "
            "    SEGA: first-line — FDA approved (EXIST-1 trial); "
            "    AML: first-line (>3 cm or growing): FDA approved (EXIST-2 trial); "
            "    LAM: first-line (EXIST-LAM trial); "
            "    Epilepsy: adjunctive (EXIST-3) — significant seizure reduction; "
            "  VIGABATRIN: TSC infantile spasms — FIRST-LINE specific recommendation; "
            "  Sirolimus (rapamycin): alternative to everolimus; similar mechanism; "
            "TESTING: TSC1 + TSC2 panel sequencing + MLPA; "
            "  TSC2 adjacent to PKD1 — check for contiguous deletion; "
            "  Skin biopsy for mosaicism if germline negative; "
        ),
        "inheritance": (
            "Autosomal dominant LOF (de novo ~75-80% of TSC2; familial ~20-25%); "
            "TSC2 16p13.3; de novo rate higher than TSC1 (more severe → less reproductively fit); "
            "Adjacent to PKD1 — contiguous deletion possible; "
            "Somatic mosaicism: 10-15% of TSC; "
        ),
        "disease_category": "Tuberous Sclerosis Complex type 2 / mTORopathy",
        "key_mutations": [
            "p.Arg611Ter (common nonsense — severe)",
            "p.Arg905Gln (missense GAP domain)",
            "p.Arg1772Ter (nonsense C-terminal)",
            "p.Glu1558Lys (GAP domain missense)",
            "Large exon deletions (MLPA — contiguous PKD1+TSC2)",
            "c.5024-2A>G (splice site — severe)",
            "p.Leu1624Pro (missense)",
        ],
        "clinical_keys": [
            "MORE SEVERE than TSC1 — more tubers, lower IQ, more refractory epilepsy",
            "EVEROLIMUS FIRST-LINE for SEGA + AML + LAM (three FDA approvals)",
            "POLYCYSTIC KIDNEYS if TSC2+PKD1 contiguous deletion on 16p13.3",
            "Cardiac rhabdomyomata neonates — echo MANDATORY; majority self-resolve",
            "LAM predominantly TSC2 females — everolimus stabilises lung function",
            "EXIST-3 trial: everolimus ~40% seizure reduction adjunctive to AEDs",
            "Vigabatrin first-line for TSC infantile spasms",
            "TSC2 de novo rate higher than TSC1 — more severe selection pressure",
        ],
    },
    {
        "gene": "DEPDC5",
        "protein": (
            "DEPDC5 -- 22q12.3 Autosomal-Dominant-LOF -- 1604aa -- DEP-Domain-Containing-5-"
            "GATOR1-Complex-mTORC1-Suppressor-FFEVF-FAMILIAL-FOCAL-EPILEPSY-VARIABLE-FOCI-"
            "SUDEP-RISK-ELEVATED-SOMATIC-2ND-HIT-RESECTED-TISSUE-OMIM-614191"
        ),
        "locus": "22q12.3",
        "protein_size": (
            "1604 aa / 175 kDa (DEPDC5; DEP domain containing 5; "
            "STRUCTURE: DEP domain (N-terminal) + SHEN domain + C-terminal regulatory region; "
            "FUNCTION: "
            "  Component of GATOR1 complex (DEPDC5 + NPRL2 + NPRL3); "
            "  GATOR1 is a GTPase activating protein (GAP) for RAG GTPases; "
            "  RAG GTPases control mTORC1 localisation to lysosomal surface; "
            "  GATOR1 inhibits RAG-A/B → prevents mTORC1 activation by amino acids; "
            "  Loss of DEPDC5 → GATOR1 non-functional → RAG GTPases constitutively active; "
            "  → mTORC1 hyperactivation (amino acid sensing pathway); "
            "  DEPDC5 most commonly mutated GATOR1 gene; "
            "FAMILIAL FOCAL EPILEPSY WITH VARIABLE FOCI (FFEVF): "
            "  PREVALENCE: DEPDC5 most common genetic cause of familial focal epilepsy; "
            "    ~10% of familial focal epilepsy; ~1% of all epilepsy; "
            "  HALLMARK: VARIABLE FOCI WITHIN SAME FAMILY: "
            "    Different family members may have seizures from different cortical regions; "
            "    Frontal, temporal, occipital, parietal — variable; "
            "    Same gene → different semiology in family members; "
            "    This variability is DIAGNOSTIC CLUE for DEPDC5; "
            "  SEIZURE TYPES: "
            "    Focal seizures (various types): nocturnal frontal lobe seizures common; "
            "    Hypermotor seizures; "
            "    Focal to bilateral tonic-clonic; "
            "    Typically nocturnal (sleep-related) — especially frontal lobe type; "
            "  PENETRANCE: ~60-70% (incomplete) — family members may be gene carriers without epilepsy; "
            "  SEVERITY: wide range — some well-controlled; others drug-resistant; "
            "SUDEP RISK: "
            "  ELEVATED vs sporadic focal epilepsy — multiple publications confirm; "
            "  Mechanism: nocturnal seizures + sleep-related respiratory suppression; "
            "  MONITORING: "
            "    Safety counselling MANDATORY at diagnosis; "
            "    Nocturnal seizure monitoring (bed sensor/camera); "
            "    Avoid sleeping alone; "
            "    Prone sleep position discouraged; "
            "    Consider seizure detection device (Emfit/Embrace); "
            "SOMATIC SECOND HIT: "
            "  Germline DEPDC5 LOF + somatic second hit in resected brain tissue; "
            "  Two-hit model (like TSC1/TSC2); "
            "  Somatic mutations in resected epileptogenic cortex → mTORC1 hyperactivation locally; "
            "  DEEP SEQUENCING of resected tissue (>200x): detects somatic 2nd hit; "
            "  Identifies focal mTOR dysregulation explaining drug resistance; "
            "FOCAL CORTICAL DYSPLASIA: "
            "  DEPDC5 LOF (germline) + somatic 2nd hit → focal cortical dysplasia (FCD); "
            "  FCD type Ia/Ib on histology; "
            "  Surgical candidacy: if MRI visible lesion or interictal EEG focus identified; "
            "  Pre-surgical evaluation: video-EEG + 3T MRI + MEG + SEEG if needed; "
            "TREATMENT: "
            "  Standard focal epilepsy AEDs: lamotrigine, levetiracetam, carbamazepine, lacosamide; "
            "  mTOR inhibitors: evidence emerging — everolimus/rapamycin may reduce seizures; "
            "  Surgical resection: good outcomes if localised FCD identified; "
            "  Ketogenic diet: reasonable for drug-resistant cases; "
            "  VAGAL NERVE STIMULATOR (VNS): palliative option; "
            "TESTING: DEPDC5 sequencing (germline); "
            "  Deep sequencing resected tissue (somatic 2nd hit — research/clinical labs); "
            "  Family testing: penetrance ~60-70% (unaffected carriers common); "
        ),
        "inheritance": (
            "Autosomal dominant LOF (germline); incomplete penetrance ~60-70%; "
            "DEPDC5 22q12.3; familial in majority (vs de novo TSC); "
            "Somatic second hit in brain tissue: two-hit model; "
            "Family members may be obligate carriers without epilepsy; "
        ),
        "disease_category": "FFEVF / GATOR1-mTORopathy / Familial Focal Epilepsy",
        "key_mutations": [
            "p.Arg239Ter (common nonsense — severe)",
            "p.Arg422Ter (nonsense)",
            "p.Gln488Ter (nonsense)",
            "c.1130+1G>A (splice site)",
            "p.Arg1430Ter (C-terminal nonsense)",
            "p.Ala395Ser (missense DEP domain)",
            "Large deletions (gene-level — array CGH)",
        ],
        "clinical_keys": [
            "VARIABLE FOCI WITHIN SAME FAMILY — different seizure origins in family members DIAGNOSTIC",
            "SUDEP RISK ELEVATED vs sporadic focal epilepsy — safety counselling MANDATORY",
            "Nocturnal seizure monitoring recommended (camera/sensor) + avoid prone sleep",
            "Somatic 2nd hit in resected brain tissue — deep sequencing identifies mTOR dysregulation",
            "Incomplete penetrance ~60-70% — unaffected carrier family members common",
            "mTOR inhibitors (everolimus): emerging evidence for seizure reduction",
            "Surgical resection: good outcomes if FCD visible on 3T MRI",
            "DEPDC5: most commonly mutated GATOR1 gene (~10% familial focal epilepsy)",
        ],
    },
    {
        "gene": "NPRL2",
        "protein": (
            "NPRL2 -- 17p13.1 Autosomal-Dominant-LOF -- 441aa -- Nitrogen-Permease-Regulator-Like-2-"
            "GATOR1-Complex-Component-mTORC1-RAG-GTPase-Suppressor-FAMILIAL-FOCAL-EPILEPSY-"
            "AUTISM-30pct-SUDEP-RISK-OMIM-607072"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "441 aa / 49 kDa (NPRL2; nitrogen permease regulator-like 2; "
            "STRUCTURE: longin domain + roadblock domain — similar to NPRL3; "
            "FUNCTION: "
            "  GATOR1 subunit (together with DEPDC5 and NPRL3); "
            "  NPRL2 and NPRL3 form heterodimer — scaffold for DEPDC5; "
            "  GATOR1 complex: GAP for RAG GTPases (RAGA/RAGB); "
            "  Loss of NPRL2 → destabilises entire GATOR1 complex → mTORC1 hyperactivation; "
            "  mTORC1 hyperactivation → amino acid sensing pathway dysregulated; "
            "  NPRL2 mutations rarer than DEPDC5 but same pathway; "
            "NPRL2-RELATED FOCAL EPILEPSY: "
            "  PREVALENCE: rare; ~50-80 families reported; "
            "  FAMILIAL FOCAL EPILEPSY: "
            "    Focal seizures — frontal lobe predominant; "
            "    Temporal and multilobar also reported; "
            "    NOCTURNAL frontal lobe seizures common (ADNFLE-like in some); "
            "    Variable penetrance: 60-70% (similar to DEPDC5); "
            "  AUTISM SPECTRUM DISORDER: ~30% of NPRL2 LOF carriers; "
            "    Higher autism rate than DEPDC5 (distinguishing feature); "
            "    ID in subset; "
            "  SEVERITY: variable — mild to drug-resistant; "
            "  SEIZURE TYPES: "
            "    Focal aware/impaired awareness; "
            "    Focal to bilateral tonic-clonic; "
            "    Hypermotor (frontal); "
            "  EEG: focal frontotemporal discharges; sometimes normal interictal; "
            "SUDEP RISK: "
            "  Elevated (similar to DEPDC5) — nocturnal seizures + respiratory; "
            "  SAFETY MONITORING: same protocol as DEPDC5; "
            "  Nocturnal camera/sensor; avoid sleeping alone; seizure alert device; "
            "GATOR1 PANEL TESTING: "
            "  When testing DEPDC5 negative in familial focal epilepsy: test NPRL2 + NPRL3; "
            "  GATOR1 panel: DEPDC5 + NPRL2 + NPRL3 together; "
            "  May account for additional 2-3% familial focal epilepsy; "
            "TREATMENT: "
            "  Standard focal epilepsy AEDs: lamotrigine, levetiracetam, carbamazepine, oxcarbazepine; "
            "  mTOR inhibitors: emerging evidence (same pathway as TSC/DEPDC5); "
            "  Surgery: if localised FCD + visible on MRI; "
            "  Ketogenic diet: reasonable option; "
            "TESTING: NPRL2 sequencing; GATOR1 panel (DEPDC5+NPRL2+NPRL3); "
            "  Family testing mandatory (penetrance incomplete); "
        ),
        "inheritance": (
            "Autosomal dominant LOF; incomplete penetrance ~60-70%; "
            "NPRL2 17p13.1; familial predominant (similar to DEPDC5); "
            "De novo rare; carrier family members may be unaffected; "
        ),
        "disease_category": "NPRL2-Focal Epilepsy / GATOR1-mTORopathy",
        "key_mutations": [
            "p.Arg34Ter (nonsense longin domain)",
            "p.Gln295Ter (nonsense)",
            "p.Ala58Val (missense longin domain)",
            "p.Ile372Asn (missense roadblock domain)",
            "c.453+1G>T (splice site)",
            "p.Arg262Cys (missense)",
            "Exon deletions (MLPA/panel)",
        ],
        "clinical_keys": [
            "GATOR1 complex member — same mTORC1 pathway as DEPDC5 and NPRL3",
            "AUTISM 30% — higher autism rate than DEPDC5 (distinguishing feature)",
            "SUDEP RISK elevated — nocturnal safety monitoring MANDATORY",
            "Test GATOR1 panel (DEPDC5+NPRL2+NPRL3) together in familial focal epilepsy",
            "Incomplete penetrance ~60-70% — unaffected obligate carriers common",
            "Frontal lobe nocturnal seizures predominant pattern",
            "mTOR inhibitors: emerging evidence — same pathway as TSC/DEPDC5",
            "Family cascade testing: identify unaffected carriers for SUDEP counselling",
        ],
    },
    {
        "gene": "NPRL3",
        "protein": (
            "NPRL3 -- 16p13.3 Autosomal-Dominant-LOF -- 489aa -- Nitrogen-Permease-Regulator-Like-3-"
            "GATOR1-Complex-Component-mTORC1-Suppressor-ADNFLE-LIKE-NOCTURNAL-FRONTAL-LOBE-EPILEPSY-"
            "SUDEP-RISK-16p13.3-CO-LOCALISED-TSC2-DUAL-DELETION-OMIM-600928"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "489 aa / 54 kDa (NPRL3; nitrogen permease regulator-like 3; "
            "STRUCTURE: longin domain + roadblock domain; NPRL2 paralogue; "
            "FUNCTION: "
            "  GATOR1 subunit (together with DEPDC5 and NPRL2); "
            "  NPRL3 heterodimerises with NPRL2 → structural scaffold for DEPDC5; "
            "  Loss of NPRL3 → destabilises GATOR1 complex → mTORC1 hyperactivation; "
            "  Amino acid sensing pathway: GATOR1 senses amino acid sufficiency; "
            "  Loss → cells perceive amino acid starvation signals wrongly → mTORC1 stays ON; "
            "IMPORTANT CHROMOSOMAL LOCATION: "
            "  NPRL3 maps to 16p13.3 — SAME REGION AS TSC2; "
            "  Larger 16p13.3 deletions may encompass BOTH TSC2 + NPRL3 simultaneously; "
            "  DUAL DELETION (TSC2 + NPRL3): "
            "    Two mTOR pathway genes lost simultaneously; "
            "    May produce combined/additive phenotype; "
            "    CMA/chromosomal microarray or MLPA panels must check BOTH; "
            "    When 16p13.3 deletion identified: CHECK TSC2 AND NPRL3 BOTH; "
            "NPRL3-RELATED FOCAL EPILEPSY: "
            "  FAMILIAL FOCAL EPILEPSY: "
            "    ADNFLE-LIKE PHENOTYPE: "
            "      Autosomal dominant nocturnal frontal lobe epilepsy (ADNFLE) semiology; "
            "      Hypermotor nocturnal seizures; "
            "      Often misdiagnosed as parasomnias (sleepwalking) before EEG; "
            "      ADNFLE historically attributed to CHRNA4/CHRNB2/CHRNA2 (nicotinic) — "
            "      now NPRL3 recognised as additional cause; "
            "    Variable foci (like DEPDC5): frontal predominant but multilobar possible; "
            "    Incomplete penetrance: ~65%; "
            "  SEVERITY: variable — mild (rare seizures) to drug-resistant; "
            "  SUDEP RISK: elevated (nocturnal seizures + GATOR1 pathway); "
            "  ID/AUTISM: less common than NPRL2 (~15-20% autism); "
            "MISDIAGNOSIS: "
            "  Nocturnal hypermotor seizures misdiagnosed as parasomnias; "
            "  EEG may be NORMAL interictal; "
            "  Video-polysomnography to differentiate from REM sleep behaviour disorder; "
            "TREATMENT: "
            "  Standard focal epilepsy AEDs: carbamazepine/oxcarbazepine effective in some; "
            "  Lacosamide, lamotrigine, levetiracetam; "
            "  Nocturnal dose of AED at bedtime: effective strategy; "
            "  mTOR inhibitors: emerging (same rationale as TSC/DEPDC5); "
            "  Surgical assessment if localised FCD; "
            "TESTING: NPRL3 sequencing; GATOR1 panel; "
            "  CMA/MLPA: check BOTH TSC2 and NPRL3 on 16p13.3 region; "
            "  Family testing (incomplete penetrance); "
        ),
        "inheritance": (
            "Autosomal dominant LOF; incomplete penetrance ~65%; "
            "NPRL3 16p13.3 — co-localised with TSC2; "
            "Larger 16p13.3 deletions: may include TSC2 + NPRL3 simultaneously; "
            "Family members may carry without epilepsy; "
        ),
        "disease_category": "NPRL3-Focal Epilepsy / GATOR1-mTORopathy / ADNFLE-like",
        "key_mutations": [
            "p.Arg114Ter (nonsense longin domain)",
            "p.Gln267Ter (nonsense)",
            "p.Ser385Phe (missense roadblock)",
            "c.1073+2T>C (splice site)",
            "p.Arg421Ter (nonsense C-terminal)",
            "16p13.3 deletion (CMA — includes TSC2 region)",
            "p.Ala73Val (missense)",
        ],
        "clinical_keys": [
            "16p13.3 CO-LOCALISED with TSC2 — check BOTH when 16p13.3 region deleted",
            "ADNFLE-LIKE nocturnal hypermotor seizures — often misdiagnosed as parasomnias",
            "SUDEP RISK elevated — nocturnal monitoring + safety counselling MANDATORY",
            "GATOR1 panel: test DEPDC5+NPRL2+NPRL3 together in familial focal epilepsy",
            "Incomplete penetrance ~65% — unaffected carrier family members common",
            "EEG may be normal interictal — video-polysomnography for diagnosis",
            "Carbamazepine/oxcarbazepine effective for nocturnal frontal lobe seizures",
            "CMA/MLPA must cover full 16p13.3 region to detect dual TSC2+NPRL3 deletion",
        ],
    },
    {
        "gene": "MTOR",
        "protein": (
            "MTOR -- 1p36.22 SOMATIC-GOF-Brain-Only -- 2549aa -- Mechanistic-Target-Of-Rapamycin-"
            "PI3K-Related-Kinase-mTORC1-mTORC2-Catalytic-Subunit-FCD-IIb-FOCAL-CORTICAL-DYSPLASIA-"
            "BALLOON-CELLS-PATHOGNOMONIC-DEEP-SEQUENCING-500x-MANDATORY-EPILEPSY-SURGERY-OMIM-601231"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "2549 aa / 289 kDa (MTOR; mechanistic target of rapamycin; "
            "STRUCTURE: HEAT repeats (N-terminal) + FAT domain + FRB domain (rapamycin-binding) + "
            "kinase domain (PI3K-related) + FATC domain (C-terminal); "
            "FUNCTION: "
            "  Central kinase of mTOR pathway; forms two complexes: "
            "    mTORC1 (with RAPTOR): controls protein synthesis, autophagy, cell growth; "
            "    mTORC2 (with RICTOR): controls AKT activation, cytoskeleton; "
            "  mTORC1 phosphorylates S6K1 and 4E-BP1 → protein synthesis; "
            "  mTOR integrates: growth factors (PI3K/AKT), amino acids (GATOR1/LYSOSOME), energy (AMPK); "
            "  Somatic GOF mutations in MTOR: constitutive activation independent of upstream signals; "
            "  Rapamycin/everolimus: bind FKBP12 → FKBP12-rapamycin complex inhibits mTORC1 (FRB domain); "
            "FOCAL CORTICAL DYSPLASIA TYPE IIb (FCD IIb): "
            "  SOMATIC GOF MTOR mutations cause FCD IIb; "
            "  MECHANISM: somatic mutation in single progenitor cell → clonal expansion; "
            "  Dysplastic cells accumulate in one cortical region; "
            "  FCD IIb PATHOGNOMONIC HISTOLOGY: "
            "    BALLOON CELLS: large, dysmorphic cells with abundant cytoplasm; "
            "      Balloon cells contain both neuronal and glial markers; "
            "      PATHOGNOMONIC for FCD IIb (not present in FCD IIa); "
            "    Dysmorphic neurons: enlarged abnormal-shaped neurons; "
            "    Disrupted cortical layering; "
            "    Phospho-S6 immunohistochemistry: confirms mTOR hyperactivation in lesion; "
            "CRITICAL DIAGNOSTIC CHALLENGE: "
            "  SOMATIC VARIANTS — present in small fraction of brain cells; "
            "  STANDARD NGS (100x): MISSES somatic MTOR variants (variant allele fraction <5%); "
            "  DEEP SEQUENCING MANDATORY: "
            "    Minimum 500x depth required on resected tissue; "
            "    Variant allele fraction may be 1-5% (not detectable at 100x); "
            "    Blood DNA: variant usually undetectable (brain-only somatic); "
            "    Must request DEEP/ULTRA-DEEP sequencing from diagnostic lab; "
            "  BRAIN TISSUE: resected epilepsy surgery specimen ideal source; "
            "  Saliva/buccal: may contain low-level somatic if early embryonic hit; "
            "EPILEPSY SURGERY — MOST EFFECTIVE TREATMENT: "
            "  FCD IIb surgical resection: CURE RATE ~70% seizure-free; "
            "  Best outcomes in all FCD types — balloon cells = radiologically visible on 3T; "
            "  PRE-SURGICAL EVALUATION: "
            "    3T MRI with dedicated FCD protocol (FLAIR, double inversion recovery); "
            "    Video-EEG monitoring (define ictal onset zone); "
            "    FDG-PET: hypometabolism in tuber/FCD area; "
            "    MEG: magnetic source imaging; "
            "    SEEG if MRI-negative or MRI/EEG discordant; "
            "  MRI NEGATIVE FCD IIb: ~20% — deep sequencing of tissue may identify MTOR; "
            "  SURGEON + PATHOLOGIST: phospho-S6 staining mandatory on resected tissue; "
            "mTOR INHIBITORS: "
            "  Everolimus/rapamycin: may reduce seizures preoperatively or if non-surgical; "
            "  CASE SERIES: significant seizure reduction with everolimus in MTOR-FCD IIb; "
            "  Not FDA approved for this indication; compassionate/trial use; "
            "TESTING: "
            "  Blood DNA: usually negative (somatic); do NOT reassure on negative blood test; "
            "  Resected tissue: 500x deep sequencing (mandatory); "
            "  Saliva panel if early-embryonic suspected; "
        ),
        "inheritance": (
            "Somatic GOF (brain only — not germline); "
            "MTOR 1p36.22; de novo somatic during embryonic cortical development; "
            "Blood DNA: variant usually NOT detectable; "
            "Non-heritable — recurrence risk negligible; "
            "Deep sequencing of brain tissue mandatory to detect; "
        ),
        "disease_category": "FCD IIb / MTOR-mTORopathy / Somatic-GOF Epilepsy",
        "key_mutations": [
            "p.Glu1799Lys (most common somatic GOF — kinase domain)",
            "p.Cys1483Tyr (somatic GOF)",
            "p.Leu2427Pro (somatic GOF FRB region)",
            "p.Ser2215Tyr (somatic GOF — kinase activation loop)",
            "p.Arg1859Gln (somatic GOF)",
            "p.Val1636Ile (somatic GOF — HEAT repeat)",
            "p.Thr1977Ile (somatic GOF)",
        ],
        "clinical_keys": [
            "BALLOON CELLS on histology PATHOGNOMONIC for FCD IIb — request phospho-S6 staining",
            "SOMATIC — STANDARD 100x NGS MISSES: DEEP SEQUENCING 500x MANDATORY on tissue",
            "Blood DNA: variant usually undetectable — do NOT exclude MTOR on negative blood test",
            "Epilepsy surgery MOST EFFECTIVE — ~70% seizure-free for FCD IIb",
            "3T MRI dedicated FCD protocol (FLAIR/DIR) + FDG-PET for lesion localisation",
            "Everolimus/rapamycin: emerging evidence for seizure reduction in MTOR-FCD IIb",
            "Non-heritable (somatic) — recurrence risk negligible; reassure family",
            "SEEG if MRI-negative or MRI/EEG discordant — FCD may be invisible on standard MRI",
        ],
    },
    {
        "gene": "PIK3R2",
        "protein": (
            "PIK3R2 -- 19p13.11 SOMATIC-GOF-Brain-Mosaic -- 724aa -- Phosphoinositide-3-Kinase-Regulatory-Subunit-Beta-"
            "PI3K-p85beta-AKT-mTOR-Pathway-Activator-MCAP-MEGALENCEPHALY-CAPILLARY-MALFORMATION-POLYMICROGYRIA-"
            "HEMISPHERE-ASYMMETRY-PATHOGNOMONIC-DEEP-SEQUENCING-MANDATORY-ALPELISIB-INVESTIGATIONAL-OMIM-603157"
        ),
        "locus": "19p13.11",
        "protein_size": (
            "724 aa / 83 kDa (PIK3R2; phosphoinositide-3-kinase regulatory subunit beta; p85beta; "
            "STRUCTURE: SH3 domain + proline-rich region + nSH2 domain + iSH2 domain + cSH2 domain; "
            "FUNCTION: "
            "  Regulatory subunit of class IA PI3K (heterodimer: p85beta/PIK3R2 + p110); "
            "  Normally inhibits p110 catalytic activity; "
            "  GOF mutations: disrupt inhibitory contact between nSH2 and p110 → constitutive PI3K activity; "
            "  PI3K → PIP2 → PIP3 → PDK1 → AKT activation → mTORC1 activation; "
            "  Somatic PIK3R2 GOF: focal mTOR pathway hyperactivation in brain mosaic; "
            "  Same pathway as PTEN (tumour suppressor: dephosphorylates PIP3); "
            "MEGALENCEPHALY-CAPILLARY MALFORMATION-POLYMICROGYRIA (MCAP): "
            "  MCAP SYNDROME: "
            "    MEGALENCEPHALY: brain overgrowth — head circumference >+4 SD; "
            "    CAPILLARY MALFORMATIONS: cutaneous, midline, philtrum, glabella; "
            "    POLYMICROGYRIA: cortical malformation (overgrowth of cortex); "
            "    HEMISPHERE ASYMMETRY (PATHOGNOMONIC): one hemisphere larger than other; "
            "      Asymmetry may be subtle or dramatic; "
            "      MRI: asymmetric megalencephaly + polymicrogyria + abnormal cortical gyration; "
            "    Ventriculomegaly; syndactyly (mild); "
            "    Connective tissue dysplasia; "
            "  ADDITIONAL FEATURES: "
            "    Cerebellar tonsillar ectopia (Chiari-like); "
            "    Hydrocephalus (communicating); "
            "    Intellectual disability: mild to severe; "
            "    Epilepsy: 80%+ — focal/multifocal; "
            "    Autism features; "
            "CRITICAL DIAGNOSTIC CHALLENGE (SOMATIC): "
            "  Brain-restricted somatic mosaic GOF; "
            "  STANDARD NGS (100x): MISSES (allele fraction 1-10%); "
            "  DEEP SEQUENCING MANDATORY: "
            "    Blood: may detect if early embryonic; ~500x on blood; "
            "    Saliva/buccal: sometimes higher allele fraction; "
            "    Brain tissue (surgical/post-mortem): highest yield; "
            "  REQUEST panel including PIK3R2, PIK3CA, AKT1, AKT3, PTEN for megalencephaly; "
            "ALPELISIB (PI3K INHIBITOR) — INVESTIGATIONAL: "
            "  Alpelisib (BYL719): selective PI3K-alpha inhibitor; "
            "  FDA approved for PIK3CA-related PROS (2022); "
            "  PIK3R2-MCAP: clinical trials and compassionate use; "
            "  MECHANISM: inhibits p110alpha → reduces PIP3 → reduces AKT/mTOR activation; "
            "  Evidence: case series + phase 2 trials ongoing; "
            "  ADVERSE EFFECTS: hyperglycaemia, rash, GI; monitor glucose; "
            "mTOR INHIBITORS: "
            "  Everolimus/sirolimus: target downstream mTORC1; evidence for MCAP/megalencephaly; "
            "  Rationale: PIK3R2 GOF → AKT → mTORC1; mTOR inhibitors suppress downstream; "
            "TREATMENT: "
            "  Epilepsy: standard focal AEDs + everolimus investigational; "
            "  Hydrocephalus: VP shunt if symptomatic; "
            "  Megalencephaly: neuropsychology support; no direct treatment; "
            "  Capillary malformations: laser/pulsed dye for cosmesis; "
            "TESTING: "
            "  Megalencephaly/MCAP panel: PIK3R2, PIK3CA, AKT1, AKT3, PTEN; "
            "  Deep sequencing mandatory (500x); "
            "  Blood + saliva + brain tissue if available; "
        ),
        "inheritance": (
            "Somatic GOF (brain mosaic — typically not germline); "
            "PIK3R2 19p13.11; somatic during embryonic brain development; "
            "Blood DNA: may detect if early embryonic hit (low allele fraction); "
            "Recurrence risk low (somatic); germline PIK3R2 GOF reported rarely; "
        ),
        "disease_category": "MCAP Syndrome / PIK3R2-mTORopathy / Somatic-GOF Megalencephaly",
        "key_mutations": [
            "p.Gly373Arg (most common somatic GOF — nSH2 domain)",
            "p.Trp583Ter (somatic GOF nSH2)",
            "p.Asp560Tyr (somatic GOF — iSH2 domain)",
            "p.Glu453Lys (somatic GOF nSH2)",
            "p.Gly538Arg (somatic GOF)",
            "p.Asn564Asp (somatic missense nSH2)",
            "p.Tyr464His (somatic GOF)",
        ],
        "clinical_keys": [
            "HEMISPHERE ASYMMETRY PATHOGNOMONIC on MRI — megalencephaly + polymicrogyria",
            "SOMATIC — STANDARD 100x NGS MISSES: DEEP SEQUENCING 500x MANDATORY",
            "MCAP triad: Megalencephaly + Capillary Malformation + Polymicrogyria",
            "Alpelisib (PI3K inhibitor): investigational for PIK3R2-MCAP; trials ongoing",
            "Megalencephaly panel: PIK3R2 + PIK3CA + AKT1 + AKT3 + PTEN — test together",
            "Blood + saliva + brain tissue: test all samples to maximise detection",
            "Everolimus downstream mTORC1 inhibition: evidence for PIK3R2/AKT megalencephaly",
            "VP shunt for hydrocephalus; cerebellar ectopia monitoring (Chiari-like)",
        ],
    },
    {
        "gene": "AKT3",
        "protein": (
            "AKT3 -- 1q44 SOMATIC-GOF-Brain-Mosaic -- 479aa -- AKT-Serine-Threonine-Kinase-3-"
            "PKB-gamma-PI3K-AKT-mTOR-Pathway-Effector-HEMIMEGALENCEPHALY-HME-MPPH-"
            "HEMISPHERE-ASYMMETRY-MOST-EXTREME-PATHOGNOMONIC-FUNCTIONAL-HEMISPHEROTOMY-"
            "DEEP-SEQUENCING-BLOOD-SALIVA-BIOPSY-MANDATORY-OMIM-611223"
        ),
        "locus": "1q44",
        "protein_size": (
            "479 aa / 56 kDa (AKT3; AKT serine/threonine kinase 3; PKB-gamma; "
            "STRUCTURE: PH domain (N-terminal, binds PIP3) + kinase domain + regulatory domain (C-terminal); "
            "FUNCTION: "
            "  AKT3 (PKB-gamma) is brain-predominant AKT isoform; "
            "  AKT1 ubiquitous; AKT2 liver/muscle; AKT3 brain predominant; "
            "  PIP3 (from PI3K) recruits AKT3 to membrane → PDK1 phosphorylates T305; "
            "  mTORC2 phosphorylates S472 (full activation); "
            "  Active AKT3 → phosphorylates TSC1/TSC2 → relieves RHEB inhibition → mTORC1 active; "
            "  Also phosphorylates: GSK3beta, FOXO, BAD, MDM2 (survival signals); "
            "  GOF mutations: PH domain (constitutive membrane localisation) or kinase (constitutive activity); "
            "  Somatic AKT3 GOF: extreme brain overgrowth — most severe mTOR phenotype; "
            "HEMIMEGALENCEPHALY (HME): "
            "  DEFINITION: hamartomatous overgrowth of one entire cerebral hemisphere; "
            "  AKT3 most common cause of isolated HME; "
            "  PATHOGNOMONIC FEATURES: "
            "    HEMISPHERE ASYMMETRY MOST EXTREME FORM: one hemisphere dramatically larger; "
            "    MRI: grossly enlarged hemisphere with dysmorphic gyration; "
            "    Ipsilateral ventriculomegaly; "
            "    Dysmorphic cortex: pachygyria, polymicrogyria, heterotopia; "
            "    White matter signal abnormality; "
            "    Contralateral hemisphere: normal; "
            "  PRESENTATION: "
            "    NEONATAL SEIZURES: 80%+ — onset first days-weeks of life; "
            "    Drug-resistant epilepsy: virtually universal; "
            "    Profound developmental disability (contralateral hemiparesis/hemiplegia); "
            "    Head circumference: asymmetric (ipsilateral enlargement); "
            "  EPILEPSY: "
            "    Continuous epileptiform discharges from affected hemisphere; "
            "    EEG: suppression-burst pattern; high-amplitude polymorphic slowing + spikes; "
            "    SEVERELY drug-resistant — rarely controlled medically; "
            "MEGALENCEPHALY-POLYMICROGYRIA-POLYDACTYLY-HYDROCEPHALUS (MPPH): "
            "  AKT3 also causes MPPH syndrome (bilateral megalencephaly); "
            "  MPPH: "
            "    Megalencephaly (bilateral, >+4 SD); "
            "    Polymicrogyria (bilateral, perisylvian); "
            "    Postaxial polydactyly; "
            "    Hydrocephalus; "
            "    LESS severe than HME (bilateral = more cells spared); "
            "  AKT3 vs PIK3R2 vs PIK3CA in MPPH/MCAP: overlap; test as panel; "
            "FUNCTIONAL HEMISPHEROTOMY — MOST EFFECTIVE TREATMENT: "
            "  For HME: functional hemispherotomy (disconnection of affected hemisphere); "
            "  SEIZURE-FREE rate: 50-70% (best available for HME); "
            "  TIMING: earlier surgery = better neurodevelopmental outcome; "
            "  PRE-SURGICAL: "
            "    MRI (volumetric): hemisphere size, cortical morphology; "
            "    Video-EEG: confirm unilateral hemisphere onset; "
            "    fMRI/Wada: language lateralisation (rarely applicable in infants); "
            "    Neuropsychology: developmental level; "
            "  POST-SURGICAL: contralateral hemiplegia persists/may worsen initially; "
            "  Rehabilitation: intensive PT/OT/SLT post-surgery; "
            "DEEP SEQUENCING — ALL THREE SOURCES MANDATORY: "
            "  AKT3 somatic mosaic: variant allele fraction 1-15% depending on source; "
            "  BLOOD: 2-10% VAF (early embryonic); test with 500x; "
            "  SALIVA/BUCCAL: may be higher than blood (neural crest contribution); "
            "  BRAIN BIOPSY (surgical specimen): highest yield (30-50% VAF in affected cortex); "
            "  Test ALL THREE when HME/MPPH suspected — different sources give different yield; "
            "  Standard blood NGS (100x): may miss low-level somatic; "
            "mTOR INHIBITORS: "
            "  Everolimus/sirolimus: target mTORC1 downstream of AKT3; "
            "  Evidence: seizure reduction in some HME cases pre-/post-surgery; "
            "  Alpelisib: PI3K inhibitor (upstream); investigational; "
            "TESTING: "
            "  Megalencephaly/HME panel: AKT3, PIK3CA, PIK3R2, AKT1, PTEN; "
            "  Deep sequencing (500x) on blood + saliva + biopsy; "
        ),
        "inheritance": (
            "Somatic GOF (brain mosaic); "
            "AKT3 1q44; somatic during embryonic neurogenesis; "
            "Germline AKT3 GOF: reported (non-mosaic — severe bilateral megalencephaly); "
            "Blood DNA: detectable at low allele fraction if early embryonic; "
            "Non-heritable in mosaic form; germline very rare; "
        ),
        "disease_category": "HME / MPPH / AKT3-mTORopathy / Somatic-GOF Extreme Megalencephaly",
        "key_mutations": [
            "p.Glu17Lys (most common somatic GOF — PH domain, constitutive membrane binding)",
            "p.Gly171Asp (somatic GOF kinase domain)",
            "p.Trp80Arg (somatic GOF PH domain)",
            "p.Gln59Leu (somatic GOF PH domain)",
            "p.His280Tyr (somatic GOF kinase domain)",
            "1q44 duplication/amplification (somatic)",
            "p.Met263Ile (somatic GOF)",
        ],
        "clinical_keys": [
            "HEMISPHERE ASYMMETRY MOST EXTREME FORM — one entire hemisphere enlarged PATHOGNOMONIC",
            "FUNCTIONAL HEMISPHEROTOMY most effective treatment — seizure-free 50-70%",
            "DEEP SEQUENCING BLOOD+SALIVA+BIOPSY MANDATORY — test all three sources",
            "Standard 100x NGS misses low-allele-fraction somatic AKT3 — request 500x",
            "AKT3 brain-predominant isoform (AKT1 ubiquitous; AKT2 liver/muscle)",
            "MPPH: Megalencephaly-Polymicrogyria-Polydactyly-Hydrocephalus (bilateral, milder)",
            "Neonatal drug-resistant epilepsy + asymmetric head circumference = HME until proven otherwise",
            "Megalencephaly/HME panel: AKT3 + PIK3CA + PIK3R2 + AKT1 + PTEN — test together",
        ],
    },
]


def _generate_patients_for_gene(gene: str, seed: int) -> list:
    """Generate 40 synthetic patients for a single TSC/mTOR pathway gene."""
    rng = random.Random(seed)
    patients = []

    gene_params = {
        "TSC1": {
            "iq_range": (40, 85), "epilepsy_rate": 0.85, "speech_absent_rate": 0.20,
            "walk_rate": 0.85, "sega_rate": 0.10, "aml_rate": 0.65,
            "autism_rate": 0.45, "skin_lesions_rate": 0.92, "cardiac_rate": 0.35,
            "dx_months": (3, 24), "severity_dist": (0.25, 0.45, 0.30),
            "refractory_epilepsy_rate": 0.45, "lam_rate": 0.10,
            "mutations": ["p.Arg692Ter", "p.Arg611Ter", "p.Gln801Ter", "p.Leu939Pro",
                          "Large del exon 1-23", "c.1525+1G>A", "p.Ser584Leu"],
            "sudep_risk": 0.02,
        },
        "TSC2": {
            "iq_range": (25, 70), "epilepsy_rate": 0.90, "speech_absent_rate": 0.35,
            "walk_rate": 0.70, "sega_rate": 0.15, "aml_rate": 0.75,
            "autism_rate": 0.55, "skin_lesions_rate": 0.95, "cardiac_rate": 0.50,
            "dx_months": (1, 18), "severity_dist": (0.40, 0.40, 0.20),
            "refractory_epilepsy_rate": 0.60, "lam_rate": 0.20,
            "mutations": ["p.Arg611Ter", "p.Arg905Gln", "p.Arg1772Ter", "p.Glu1558Lys",
                          "Large del TSC2+PKD1", "c.5024-2A>G", "p.Leu1624Pro"],
            "sudep_risk": 0.03,
        },
        "DEPDC5": {
            "iq_range": (55, 100), "epilepsy_rate": 0.70, "speech_absent_rate": 0.05,
            "walk_rate": 0.95, "sega_rate": 0.00, "aml_rate": 0.00,
            "autism_rate": 0.15, "skin_lesions_rate": 0.02, "cardiac_rate": 0.02,
            "dx_months": (12, 120), "severity_dist": (0.10, 0.40, 0.50),
            "refractory_epilepsy_rate": 0.30, "lam_rate": 0.00,
            "mutations": ["p.Arg239Ter", "p.Arg422Ter", "p.Gln488Ter", "c.1130+1G>A",
                          "p.Arg1430Ter", "p.Ala395Ser", "Large del CGH"],
            "sudep_risk": 0.12,
        },
        "NPRL2": {
            "iq_range": (50, 100), "epilepsy_rate": 0.65, "speech_absent_rate": 0.05,
            "walk_rate": 0.95, "sega_rate": 0.00, "aml_rate": 0.00,
            "autism_rate": 0.30, "skin_lesions_rate": 0.02, "cardiac_rate": 0.02,
            "dx_months": (12, 120), "severity_dist": (0.10, 0.40, 0.50),
            "refractory_epilepsy_rate": 0.25, "lam_rate": 0.00,
            "mutations": ["p.Arg34Ter", "p.Gln295Ter", "p.Ala58Val", "p.Ile372Asn",
                          "c.453+1G>T", "p.Arg262Cys", "Exon del"],
            "sudep_risk": 0.10,
        },
        "NPRL3": {
            "iq_range": (55, 100), "epilepsy_rate": 0.65, "speech_absent_rate": 0.04,
            "walk_rate": 0.96, "sega_rate": 0.00, "aml_rate": 0.00,
            "autism_rate": 0.18, "skin_lesions_rate": 0.02, "cardiac_rate": 0.02,
            "dx_months": (12, 120), "severity_dist": (0.08, 0.40, 0.52),
            "refractory_epilepsy_rate": 0.25, "lam_rate": 0.00,
            "mutations": ["p.Arg114Ter", "p.Gln267Ter", "p.Ser385Phe", "c.1073+2T>C",
                          "p.Arg421Ter", "16p13.3 del", "p.Ala73Val"],
            "sudep_risk": 0.10,
        },
        "MTOR": {
            "iq_range": (30, 80), "epilepsy_rate": 0.98, "speech_absent_rate": 0.30,
            "walk_rate": 0.70, "sega_rate": 0.00, "aml_rate": 0.00,
            "autism_rate": 0.35, "skin_lesions_rate": 0.05, "cardiac_rate": 0.02,
            "dx_months": (3, 48), "severity_dist": (0.40, 0.45, 0.15),
            "refractory_epilepsy_rate": 0.70, "lam_rate": 0.00,
            "mutations": ["p.Glu1799Lys", "p.Cys1483Tyr", "p.Leu2427Pro", "p.Ser2215Tyr",
                          "p.Arg1859Gln", "p.Val1636Ile", "p.Thr1977Ile"],
            "sudep_risk": 0.05,
            "balloon_cells": 0.95,
            "surgical_cure_rate": 0.70,
        },
        "PIK3R2": {
            "iq_range": (30, 75), "epilepsy_rate": 0.82, "speech_absent_rate": 0.25,
            "walk_rate": 0.72, "sega_rate": 0.00, "aml_rate": 0.00,
            "autism_rate": 0.40, "skin_lesions_rate": 0.60, "cardiac_rate": 0.05,
            "dx_months": (1, 24), "severity_dist": (0.35, 0.45, 0.20),
            "refractory_epilepsy_rate": 0.55, "lam_rate": 0.00,
            "mutations": ["p.Gly373Arg", "p.Trp583Ter", "p.Asp560Tyr", "p.Glu453Lys",
                          "p.Gly538Arg", "p.Asn564Asp", "p.Tyr464His"],
            "sudep_risk": 0.05,
            "hemisphere_asymmetry": 0.95,
        },
        "AKT3": {
            "iq_range": (15, 55), "epilepsy_rate": 0.95, "speech_absent_rate": 0.55,
            "walk_rate": 0.45, "sega_rate": 0.00, "aml_rate": 0.00,
            "autism_rate": 0.50, "skin_lesions_rate": 0.05, "cardiac_rate": 0.03,
            "dx_months": (0, 6), "severity_dist": (0.60, 0.30, 0.10),
            "refractory_epilepsy_rate": 0.85, "lam_rate": 0.00,
            "mutations": ["p.Glu17Lys", "p.Gly171Asp", "p.Trp80Arg", "p.Gln59Leu",
                          "p.His280Tyr", "1q44 dup", "p.Met263Ile"],
            "sudep_risk": 0.07,
            "hemisphere_asymmetry": 0.98,
        },
    }

    p = gene_params.get(gene, gene_params["TSC1"])
    sev_w = p["severity_dist"]

    # somatic genes: both sexes, germline dominant genes: both sexes
    for i in range(40):
        iq = rng.randint(*p["iq_range"])
        severity = rng.choices(["severe", "moderate", "mild"], weights=sev_w, k=1)[0]
        # TSC: slight male/female equal; MTOR/PIK3R2/AKT3 somatic: both sexes
        sex = rng.choice(["F", "M"])
        dx_mo = rng.randint(*p["dx_months"])
        mutation = rng.choice(p["mutations"])

        # Gene-specific fields
        sega = rng.random() < p["sega_rate"]
        aml = rng.random() < p["aml_rate"]
        lam = rng.random() < p["lam_rate"]
        balloon_cells = rng.random() < p.get("balloon_cells", 0.05)
        hemisphere_asymmetry = rng.random() < p.get("hemisphere_asymmetry", 0.05)
        surgical_candidate = rng.random() < (0.70 if gene in ("MTOR", "AKT3") else
                                              0.50 if gene in ("DEPDC5", "NPRL2", "NPRL3", "PIK3R2") else
                                              0.25)
        everolimus_eligible = sega or aml or lam or (gene in ("TSC1", "TSC2") and rng.random() < 0.35)
        deep_seq_required = gene in ("MTOR", "PIK3R2", "AKT3")
        sudep_risk = rng.random() < p["sudep_risk"]
        nocturnal_seizures = rng.random() < (0.70 if gene in ("DEPDC5", "NPRL2", "NPRL3") else 0.25)

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "sex": sex,
            "iq_estimate": iq,
            "severity": severity,
            "age_at_diagnosis_mo": dx_mo,
            "mutation": mutation,
            "epilepsy": rng.random() < p["epilepsy_rate"],
            "refractory_epilepsy": rng.random() < p["refractory_epilepsy_rate"],
            "speech_absent": rng.random() < p["speech_absent_rate"],
            "independent_walk": rng.random() < p["walk_rate"],
            "sega": sega,
            "renal_aml": aml,
            "lam": lam,
            "autism_features": rng.random() < p["autism_rate"],
            "skin_lesions": rng.random() < p["skin_lesions_rate"],
            "cardiac_rhabdomyoma": rng.random() < p["cardiac_rate"],
            "balloon_cells_histology": balloon_cells,
            "hemisphere_asymmetry_mri": hemisphere_asymmetry,
            "surgical_candidate": surgical_candidate,
            "everolimus_eligible": everolimus_eligible,
            "deep_seq_required": deep_seq_required,
            "sudep_risk_flag": sudep_risk,
            "nocturnal_seizures": nocturnal_seizures,
            "somatic_variant": gene in ("MTOR", "PIK3R2", "AKT3"),
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-TSC-mTOR-Pathway-Atlas."""
    return {
        "atlas":          "Hereditary-TSC-mTOR-Pathway-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary TSC/mTOR Pathway Atlas "
            "(TSC1-TSC2-DEPDC5-NPRL2-NPRL3-MTOR-PIK3R2-AKT3)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "TSC1":   "AD LOF 9q34.13 (Tuberous Sclerosis Complex type 1; CORTICAL TUBERS + SUBEPENDYMAL NODULES + AML PATHOGNOMONIC TRIAD; SEGA 5-15% EVEROLIMUS FDA; MILDER than TSC2; vigabatrin infantile spasms)",
            "TSC2":   "AD LOF 16p13.3 (Tuberous Sclerosis Complex type 2; MORE SEVERE than TSC1 — more tubers, refractory epilepsy, lower IQ; POLYCYSTIC KIDNEYS adjacent PKD1; EVEROLIMUS FIRST-LINE SEGA+AML+LAM; cardiac rhabdomyomata neonatal)",
            "DEPDC5": "AD LOF 22q12.3 (GATOR1 complex; FFEVF FAMILIAL FOCAL EPILEPSY VARIABLE FOCI; SUDEP RISK ELEVATED vs sporadic; somatic 2nd hit resected tissue; incomplete penetrance ~60-70%)",
            "NPRL2":  "AD LOF 17p13.1 (GATOR1 complex; Familial focal epilepsy; AUTISM 30% higher than DEPDC5; SUDEP risk; nocturnal frontal lobe seizures; GATOR1 panel DEPDC5+NPRL2+NPRL3)",
            "NPRL3":  "AD LOF 16p13.3 (GATOR1 complex; ADNFLE-like nocturnal hypermotor seizures; SUDEP risk; 16p13.3 CO-LOCALISED with TSC2 — CHECK BOTH on 16p13.3 deletion; misdiagnosed as parasomnias)",
            "MTOR":   "Somatic GOF 1p36.22 brain only (FCD IIb; BALLOON CELLS PATHOGNOMONIC histology; DEEP SEQUENCING 500x MANDATORY — standard 100x MISSES; epilepsy surgery ~70% cure rate; non-heritable)",
            "PIK3R2": "Somatic GOF 19p13.11 brain mosaic (MCAP — Megalencephaly+Capillary Malformation+Polymicrogyria; HEMISPHERE ASYMMETRY PATHOGNOMONIC; DEEP SEQUENCING MANDATORY; alpelisib investigational)",
            "AKT3":   "Somatic GOF 1q44 brain mosaic (HME HEMIMEGALENCEPHALY; MPPH; HEMISPHERE ASYMMETRY MOST EXTREME FORM PATHOGNOMONIC; FUNCTIONAL HEMISPHEROTOMY most effective; DEEP SEQUENCING BLOOD+SALIVA+BIOPSY MANDATORY)",
        },
        "key_clinical_rules": [
            "TSC1/TSC2: mTOR pathway — EVEROLIMUS (mTORC1 inhibitor) is DISEASE-MODIFYING for SEGA + AML + LAM (three separate FDA approvals); also adjunctive for epilepsy (EXIST-3 ~40% reduction)",
            "TSC2 MORE SEVERE than TSC1 — more tubers, lower IQ, more refractory epilepsy, higher SEGA rate; TSC2+PKD1 contiguous 16p13.3 deletion = severe polycystic kidney disease",
            "DEPDC5/NPRL2/NPRL3: GATOR1 complex — SUDEP RISK ELEVATED vs sporadic focal epilepsy; nocturnal safety monitoring + counselling MANDATORY for all three; avoid sleeping alone",
            "MTOR/PIK3R2/AKT3: SOMATIC — STANDARD 100x NGS MISSES low allele-fraction variants; DEEP SEQUENCING 500x on brain tissue/blood/saliva MANDATORY; do NOT exclude on negative blood 100x NGS",
            "MTOR somatic FCD IIb: surgical resection PRIMARY treatment — ~70% seizure-free; BALLOON CELLS on histology PATHOGNOMONIC; request phospho-S6 immunostaining on resected tissue",
            "NPRL3 co-localised with TSC2 on 16p13.3 — when 16p13.3 chromosomal region deleted, CHECK BOTH TSC2 AND NPRL3 independently by MLPA/CMA; dual pathway hit possible",
            "Rapamycin/everolimus works DOWNSTREAM of entire TSC1-TSC2-DEPDC5-NPRL2-NPRL3 pathway: RHEB → mTORC1; also works downstream of MTOR-PIK3R2-AKT3 (mTORC1 is convergence point)",
            "Vigabatrin is FIRST-LINE AED specifically for TSC infantile spasms (higher evidence than general IS); everolimus may be added adjunctively for drug-resistant TSC epilepsy",
            "AKT3 HME: test BLOOD + SALIVA + BRAIN BIOPSY all three — different sources give different allele fractions; early surgical hemispherotomy (before 2 years) gives best developmental outcome",
            "GATOR1 panel (DEPDC5+NPRL2+NPRL3) together in familial focal epilepsy — incomplete penetrance means unaffected obligate carrier family members require SUDEP counselling too",
            "PIK3R2/AKT3/PIK3CA megalencephaly panel: test together as these somatic mTOR pathway genes cause overlapping MCAP/HME/MPPH phenotypes with similar hemisphere asymmetry pattern",
            "TSC surveillance protocol: brain MRI 1-3 yearly (SEGA), renal USS annually (AML), ECG/echo neonatal (rhabdomyomata), annual TAND (neuropsychiatric) assessment — all four mandatory",
        ],
        "gene_panel_note": (
            "TSC/mTOR pathway panel (2024): TSC1, TSC2, DEPDC5, NPRL2, NPRL3, MTOR, PIK3R2, AKT3; "
            "Extended: PIK3CA, AKT1, PTEN, RHEB, RPS6KB1 for mTOR-related overgrowth; "
            "Testing strategy: "
            "  TSC: panel sequencing + MLPA (TSC1 + TSC2); ~15% NMF (skin biopsy for mosaic); "
            "  GATOR1 (familial focal epilepsy): DEPDC5+NPRL2+NPRL3 panel; "
            "  Somatic (FCD/HME/megalencephaly): deep sequencing 500x brain + saliva + blood; "
            "Everolimus (FDA approvals): SEGA 2010, AML 2012, LAM 2012 — three separate indications; "
            "Alpelisib (FDA 2022): PROS (PIK3CA-related); PIK3R2 investigational; "
            "SUDEP monitoring: all GATOR1 gene carriers + drug-resistant TSC epilepsy; "
            "mTOR inhibitors downstream convergence: relevant to all 8 genes in this atlas"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-TSC-mTOR-Pathway-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)

        epilepsy_n          = sum(1 for p in patients if p["epilepsy"])
        refractory_n        = sum(1 for p in patients if p["refractory_epilepsy"])
        speech_absent_n     = sum(1 for p in patients if p["speech_absent"])
        walk_n              = sum(1 for p in patients if p["independent_walk"])
        sega_n              = sum(1 for p in patients if p["sega"])
        aml_n               = sum(1 for p in patients if p["renal_aml"])
        lam_n               = sum(1 for p in patients if p["lam"])
        autism_n            = sum(1 for p in patients if p["autism_features"])
        skin_n              = sum(1 for p in patients if p["skin_lesions"])
        cardiac_n           = sum(1 for p in patients if p["cardiac_rhabdomyoma"])
        balloon_n           = sum(1 for p in patients if p["balloon_cells_histology"])
        hemi_asym_n         = sum(1 for p in patients if p["hemisphere_asymmetry_mri"])
        surgical_n          = sum(1 for p in patients if p["surgical_candidate"])
        everolimus_n        = sum(1 for p in patients if p["everolimus_eligible"])
        deep_seq_n          = sum(1 for p in patients if p["deep_seq_required"])
        sudep_n             = sum(1 for p in patients if p["sudep_risk_flag"])
        nocturnal_n         = sum(1 for p in patients if p["nocturnal_seizures"])
        somatic_n           = sum(1 for p in patients if p["somatic_variant"])
        severe_n            = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n          = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n              = sum(1 for p in patients if p["severity"] == "mild")
        mean_iq             = round(sum(p["iq_estimate"] for p in patients) / n, 1)
        mean_age            = round(sum(p["age_at_diagnosis_mo"] for p in patients) / n, 1)
        mutations_seen      = list({p["mutation"] for p in patients})

        genes_data.append({
            "gene":                     gene,
            "locus":                    gene_info["locus"],
            "n_patients":               n,
            "severe_pct":               round(severe_n / n * 100, 1),
            "moderate_pct":             round(moderate_n / n * 100, 1),
            "mild_pct":                 round(mild_n / n * 100, 1),
            "mean_iq":                  mean_iq,
            "epilepsy_pct":             round(epilepsy_n / n * 100, 1),
            "refractory_epilepsy_pct":  round(refractory_n / n * 100, 1),
            "speech_absent_pct":        round(speech_absent_n / n * 100, 1),
            "independent_walk_pct":     round(walk_n / n * 100, 1),
            "sega_pct":                 round(sega_n / n * 100, 1),
            "renal_aml_pct":            round(aml_n / n * 100, 1),
            "lam_pct":                  round(lam_n / n * 100, 1),
            "autism_pct":               round(autism_n / n * 100, 1),
            "skin_lesions_pct":         round(skin_n / n * 100, 1),
            "cardiac_rhabdomyoma_pct":  round(cardiac_n / n * 100, 1),
            "balloon_cells_pct":        round(balloon_n / n * 100, 1),
            "hemisphere_asymmetry_pct": round(hemi_asym_n / n * 100, 1),
            "surgical_candidate_pct":   round(surgical_n / n * 100, 1),
            "everolimus_eligible_pct":  round(everolimus_n / n * 100, 1),
            "deep_seq_required_pct":    round(deep_seq_n / n * 100, 1),
            "sudep_risk_pct":           round(sudep_n / n * 100, 1),
            "nocturnal_seizures_pct":   round(nocturnal_n / n * 100, 1),
            "somatic_variant_pct":      round(somatic_n / n * 100, 1),
            "mean_age_dx_mo":           mean_age,
            "sample_mutations":         mutations_seen[:4],
            "protein":                  gene_info["protein"],
            "inheritance":              gene_info["inheritance"][:200],
            "disease_category":         gene_info["disease_category"],
        })
    return {
        "atlas": "Hereditary-TSC-mTOR-Pathway-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-TSC-mTOR-Pathway-Atlas."""
    definitions = [
        {
            "term": "TSC/mTOR Pathway Atlas: Classification, Pathway Architecture, and Diagnostic Framework",
            "genes": ["TSC1", "TSC2", "DEPDC5", "NPRL2", "NPRL3", "MTOR", "PIK3R2", "AKT3"],
            "definition": (
                "HEREDITARY TSC/mTOR PATHWAY ATLAS — OVERVIEW AND CLASSIFICATION: "
                "mTOR PATHWAY ARCHITECTURE (top to bottom): "
                "  UPSTREAM (growth factor arm): "
                "    PTEN (phosphatase) → suppresses PIP3; "
                "    PI3K (PIK3R2 regulatory + PIK3CA catalytic) → generates PIP3; "
                "    AKT (AKT1/AKT2/AKT3) → activated by PIP3; "
                "    AKT → phosphorylates TSC2 → inhibits TSC1/TSC2 complex; "
                "  AMINO ACID SENSING ARM: "
                "    GATOR1 (DEPDC5+NPRL2+NPRL3) → GAP for RAG GTPases → suppresses mTORC1; "
                "    Loss of GATOR1 → RAG constitutively active → mTORC1 to lysosome; "
                "  CONVERGENCE POINT: "
                "    TSC1/TSC2 complex → inhibits RHEB; "
                "    RHEB → activates mTORC1 (only when TSC1/2 released); "
                "    mTORC1 → S6K1, 4E-BP1 → protein synthesis, cell growth; "
                "  RAPAMYCIN/EVEROLIMUS TARGET: mTORC1 (FKBP12 binding site on MTOR protein); "
                "8-GENE CLASSIFICATION: "
                "  GROUP 1 — GERMLINE AD LOF (classic TSC): TSC1, TSC2; "
                "    Systemic disease: brain + kidneys + lungs + skin + heart; "
                "    Hamartin/tuberin heterodimer: direct mTORC1 suppressor via RHEB; "
                "  GROUP 2 — GERMLINE AD LOF (GATOR1 focal epilepsy): DEPDC5, NPRL2, NPRL3; "
                "    Predominantly epilepsy; little systemic disease; "
                "    Incomplete penetrance; SUDEP risk shared feature; "
                "  GROUP 3 — SOMATIC GOF (brain-only mosaic): MTOR, PIK3R2, AKT3; "
                "    Brain malformations: FCD IIb, megalencephaly, HME; "
                "    Somatic = NOT heritable; DEEP SEQUENCING mandatory; "
                "    Standard NGS MISSES low allele fraction somatic variants; "
                "KEY CROSS-GENE CLINICAL RULES: "
                "  EVEROLIMUS: works downstream of ALL Group 1+2 genes (mTORC1 suppression); "
                "    Also works downstream of Group 3 (MTOR/PIK3R2/AKT3 → mTORC1); "
                "  SUDEP RISK: Group 2 (GATOR1) elevated; TSC drug-resistant also elevated; "
                "  SOMATIC TESTING: Groups 3 ALWAYS requires deep sequencing — never standard NGS alone; "
                "  16p13.3 LOCUS: TSC2 and NPRL3 both here — check both on any 16p13.3 deletion; "
                "DIAGNOSTIC ALGORITHM: "
                "  STEP 1: Clinical TSC features (tubers + SEN + AML)? → TSC1/TSC2 panel + MLPA; "
                "  STEP 2: Familial focal epilepsy? → GATOR1 panel (DEPDC5+NPRL2+NPRL3); "
                "  STEP 3: FCD on MRI (drug-resistant)? → Deep sequencing tissue 500x for MTOR; "
                "  STEP 4: Megalencephaly/HME? → Somatic panel (PIK3R2+AKT3+PIK3CA+AKT1+PTEN) 500x; "
                "TREATMENT SUMMARY: "
                "  TSC1/TSC2: everolimus (SEGA/AML/LAM/epilepsy) + vigabatrin (IS); "
                "  GATOR1: standard focal AEDs + SUDEP safety protocol; mTOR investigational; "
                "  MTOR-FCD IIb: epilepsy surgery first; everolimus adjunctive; "
                "  PIK3R2-MCAP: alpelisib investigational; everolimus downstream; "
                "  AKT3-HME: functional hemispherotomy FIRST; early timing critical"
            ),
        },
        {
            "term": "Tuberous Sclerosis Complex (TSC1 and TSC2) — Diagnosis, Surveillance, and Everolimus Protocol",
            "genes": ["TSC1", "TSC2"],
            "definition": (
                "TUBEROUS SCLEROSIS COMPLEX — COMPLETE CLINICAL PROTOCOL: "
                "DIAGNOSTIC CRITERIA (revised 2012 — Northrup-Krueger): "
                "  DEFINITE TSC: 2 major OR 1 major + 2 minor features; "
                "  MAJOR FEATURES (11 listed): "
                "    Hypomelanotic macules (≥3, ≥5mm): EARLIEST SIGN — Wood's lamp in neonates; "
                "    Facial angiofibromas (≥3) OR fibrous cephalic plaque; "
                "    Ungual fibromas (≥2 Koenen tumours); "
                "    Shagreen patch (connective tissue naevus); "
                "    Multiple retinal hamartomata; "
                "    Cortical dysplasias (tubers + radial migration lines); "
                "    Subependymal nodules (SEN); "
                "    Subependymal giant cell astrocytoma (SEGA); "
                "    Cardiac rhabdomyoma; "
                "    Lymphangioleiomyomatosis (LAM — females); "
                "    Angiomyolipomata (≥2); "
                "TSC1 vs TSC2 SEVERITY COMPARISON: "
                "  TSC1: ~30% cases; milder; fewer tubers; better IQ; less LAM; "
                "  TSC2: ~70% cases; more severe; more tubers; lower IQ; higher SEGA rate; "
                "  TSC2+PKD1 (adjacent 16p13.3): SEVERE polycystic kidney disease — "
                "    Presents in infancy/childhood (unlike adult ADPKD); renal function decline; "
                "EPILEPSY MANAGEMENT IN TSC: "
                "  VIGABATRIN: FIRST-LINE for TSC infantile spasms (specific guideline); "
                "    Dose: 50-150 mg/kg/day in 2 divided doses; visual field monitoring; "
                "  EVEROLIMUS (EXIST-3): adjunctive for drug-resistant TSC epilepsy; "
                "    4.5 mg/m2/day orally; trough level 5-15 ng/mL; "
                "    40% ≥50% seizure reduction; "
                "  Ketogenic diet: reasonable option for drug-resistant TSC; "
                "  Epilepsy surgery: if SEGA or tuber clearly epileptogenic; "
                "SEGA MANAGEMENT — EVEROLIMUS PROTOCOL: "
                "  INDICATION: growing SEGA near foramen of Monro OR symptomatic; "
                "  EVEROLIMUS: "
                "    Start: 4.5 mg/m2/day; titrate to trough 5-10 ng/mL; "
                "    Monitoring: CBC, metabolic panel, lipids, creatinine 4-weekly initially; "
                "    Stomatitis: commonest side effect — oral hygiene; "
                "    Immunosuppression: avoid live vaccines; infection surveillance; "
                "    Duration: INDEFINITE (discontinuation → regrowth); "
                "  SURGICAL RESECTION: if acute hydrocephalus or everolimus failure; "
                "RENAL AML MANAGEMENT: "
                "  EVEROLIMUS: first-line if ≥3 cm AML or growing; "
                "  EMBOLISATION: acute haemorrhage (Wunderlich syndrome — retroperitoneal bleed); "
                "  SURVEILLANCE: renal USS annually; MRI every 3 years; "
                "SURVEILLANCE PROTOCOL (ANNUAL MINIMUM): "
                "  Brain MRI: SEGA + tuber burden; new lesions; 1-3 yearly; "
                "  Renal USS: AML + cysts; annually; "
                "  Skin: dermatology review; angiofibromas (laser treatment); "
                "  Ophthalmology: retinal hamartomata; "
                "  TAND (TSC-associated neuropsychiatric): autism/ADHD/anxiety/sleep annually; "
                "  ECG: arrhythmia screening; "
                "  Chest HRCT: LAM in females (from age 18 every 5-10 years or if symptomatic); "
                "  Neuropsychology: annual cognitive assessment; "
                "CARDIAC RHABDOMYOMATA: "
                "  Echo MANDATORY in all neonates + foetal echo if TSC known prenatally; "
                "  MAJORITY REGRESS SPONTANEOUSLY by age 2; "
                "  If large: outflow obstruction → intervention (everolimus may shrink); "
                "  Arrhythmia: Wolff-Parkinson-White associated; ECG; "
                "GENETIC COUNSELLING: "
                "  AD inheritance: 50% risk to offspring; "
                "  De novo: ~60% TSC1; ~75-80% TSC2; "
                "  Mosaicism: ~10-15% negative germline → skin biopsy for mosaic testing; "
                "  Prenatal: foetal echo + brain MRI offer"
            ),
        },
        {
            "term": "GATOR1 Complex Epilepsies (DEPDC5, NPRL2, NPRL3) — SUDEP Risk Protocol and mTOR Connection",
            "genes": ["DEPDC5", "NPRL2", "NPRL3"],
            "definition": (
                "GATOR1 COMPLEX EPILEPSIES — DEPDC5, NPRL2, NPRL3 CLINICAL PROTOCOL: "
                "GATOR1 COMPLEX STRUCTURE AND FUNCTION: "
                "  GATOR1 = DEPDC5 (scaffold) + NPRL2 (structural) + NPRL3 (structural); "
                "  GATOR1 function: GAP (GTPase activating protein) for RAG GTPases (RAGA/RAGB); "
                "  RAG GTPases: gate mTORC1 access to lysosomal surface (amino acid sensing); "
                "  GATOR1 OFF (LOF mutation) → RAG constitutively active → mTORC1 ON; "
                "  mTORC1 hyperactivation → excessive protein synthesis → neuronal excitability ↑; "
                "  This is the FOCAL form of mTOR pathway epilepsy (cf TSC = systemic); "
                "CLINICAL SPECTRUM (all three genes similar): "
                "  FAMILIAL FOCAL EPILEPSY: "
                "    DEPDC5: VARIABLE FOCI (FFEVF) — different foci in family members DIAGNOSTIC; "
                "    NPRL2: frontal lobe predominant; autism 30% (higher than DEPDC5); "
                "    NPRL3: ADNFLE-like nocturnal hypermotor; misdiagnosed as parasomnias; "
                "  PENETRANCE: incomplete 60-70% for all three; "
                "    Unaffected obligate carrier family members: NOT EXEMPT from SUDEP counselling; "
                "    Carriers may have subclinical seizures or sporadic unprovoked events; "
                "SUDEP RISK PROTOCOL (MANDATORY FOR ALL THREE GENES): "
                "  ELEVATED SUDEP RISK vs sporadic focal epilepsy (confirmed multiple cohorts); "
                "  Mechanism: nocturnal seizures → respiratory suppression → cardiac arrhythmia; "
                "  SAFETY INTERVENTIONS: "
                "    (1) Nocturnal seizure monitoring: bed sensor (Emfit) OR camera; "
                "    (2) Avoid sleeping alone — partner/carer awareness; "
                "    (3) Prone sleep position DISCOURAGED (especially post-ictal); "
                "    (4) Seizure detection wearable (Embrace/GW watch); "
                "    (5) Rescue medication (buccal midazolam/nasal diazepam) readily available; "
                "    (6) Swimming/bathing supervision; "
                "    (7) Driving restrictions per local regulation (standard); "
                "  FAMILY CARRIERS: SAME SUDEP COUNSELLING even if currently seizure-free; "
                "    Breakthrough seizure may occur especially if febrile/sleep-deprived; "
                "SOMATIC SECOND HIT IN GATOR1 GENES: "
                "  Germline DEPDC5/NPRL2/NPRL3 LOF + somatic 2nd hit in focal dysplastic cortex; "
                "  Two-hit model: explains why penetrance is incomplete AND some have focal FCD; "
                "  Somatic 2nd hit detected by deep sequencing (200-500x) resected tissue; "
                "  Those with somatic 2nd hit: more likely to have visible FCD on MRI; "
                "  Those without somatic 2nd hit: normal MRI; diffuse/variable epilepsy; "
                "mTOR INHIBITOR EVIDENCE IN GATOR1: "
                "  Everolimus/sirolimus: case series + preclinical evidence for seizure reduction; "
                "  Rationale: GATOR1 LOF → mTORC1 hyperactive → everolimus inhibits mTORC1; "
                "  Not FDA approved for this indication; consider in drug-resistant GATOR1 epilepsy; "
                "TESTING STRATEGY: "
                "  GATOR1 PANEL (DEPDC5+NPRL2+NPRL3): test together; "
                "  Family testing: all first-degree relatives — identify carriers for SUDEP counselling; "
                "  Unaffected carrier parents: STILL require SUDEP safety counselling; "
                "  Deep sequencing of resected tissue: if surgical candidate; "
                "TREATMENT: "
                "  DEPDC5: standard focal AEDs (lamotrigine, levetiracetam, carbamazepine, lacosamide); "
                "  NPRL2/NPRL3: carbamazepine/oxcarbazepine often effective for frontal lobe type; "
                "  Surgical evaluation: if MRI visible FCD + electroclinical concordance; "
                "  Ketogenic diet: reasonable for drug-resistant cases; "
                "  VNS: palliative option if non-surgical; "
                "NPRL3-SPECIFIC: 16p13.3 CO-LOCALISATION WITH TSC2: "
                "  NPRL3 maps to 16p13.3 (same chromosomal band as TSC2); "
                "  Larger 16p13.3 deletions may encompass BOTH genes; "
                "  CMA report showing 16p13.3 deletion: confirm whether TSC2 AND NPRL3 both deleted; "
                "  Combined TSC2+NPRL3 deletion: combined phenotype (TSC features + GATOR1 epilepsy)"
            ),
        },
        {
            "term": "Somatic mTOR Pathway Epilepsies (MTOR, PIK3R2, AKT3) — Deep Sequencing and Surgical Protocol",
            "genes": ["MTOR", "PIK3R2", "AKT3"],
            "definition": (
                "SOMATIC mTOR PATHWAY BRAIN MALFORMATIONS — DIAGNOSTIC AND SURGICAL PROTOCOL: "
                "WHY SOMATIC VARIANTS REQUIRE SPECIAL ATTENTION: "
                "  Standard clinical NGS (100x depth): designed for germline variants (VAF ~50%); "
                "  Somatic variants in mosaic brain: VAF 1-10% in blood; 1-5% in some brain cells; "
                "  At 100x depth: minimum detectable VAF ~5-10% (many somatic variants missed); "
                "  At 500x depth: detectable VAF ~1-2%; captures brain-restricted somatic mosaicism; "
                "  CRITICAL: NEGATIVE BLOOD 100x NGS DOES NOT EXCLUDE SOMATIC mTOR VARIANT; "
                "THREE-GENE SOMATIC SPECTRUM: "
                "  MTOR (1p36.22): FCD IIb (focal); seizures main feature; "
                "  PIK3R2 (19p13.11): MCAP (hemispheric/bilateral megalencephaly); "
                "  AKT3 (1q44): HME (extreme hemispheric overgrowth) + MPPH (bilateral); "
                "MTOR — FOCAL CORTICAL DYSPLASIA IIb PROTOCOL: "
                "  HISTOLOGY: "
                "    BALLOON CELLS: large pale cells with abundant cytoplasm; "
                "      Both neuronal AND glial markers on IHC; "
                "      PATHOGNOMONIC for FCD IIb; distinguishes from FCD IIa (no balloon cells); "
                "    Dysmorphic neurons: large, misshapen nuclei; "
                "    Disrupted cortical layering (loss of normal 6-layer structure); "
                "    Phospho-S6K1 immunostaining: bright positive in lesion; confirms mTOR activation; "
                "  SURGICAL EVALUATION: "
                "    3T MRI with FCD protocol: FLAIR + DIR + 1mm isotropic; "
                "      FCD IIb MRI: transmantle sign (linear FLAIR hyperintensity to ventricle); "
                "      Bottom-of-sulcus FCD: easily missed — re-read with neuroradiology; "
                "    Video-EEG: ictal onset zone; "
                "    FDG-PET: hypometabolism; "
                "    MEG/source imaging; "
                "    SEEG (stereoelectroencephalography): if MRI negative/discordant; "
                "  SURGICAL OUTCOME: ~70% seizure-free (best in all FCD types); "
                "  POST-RESECTION: deep sequencing of resected tissue MANDATORY (confirms diagnosis); "
                "PIK3R2 — MCAP SYNDROME PROTOCOL: "
                "  DIAGNOSTIC FEATURES: "
                "    Megalencephaly (head circumference >+4 SD): "
                "    Capillary malformations: midline, philtrum, glabella (cutaneous vascular); "
                "    Polymicrogyria: bilateral, perisylvian predominant; "
                "    HEMISPHERE ASYMMETRY (PATHOGNOMONIC): one side larger; "
                "    Ventriculomegaly; cerebellar tonsillar ectopia; "
                "    Mild syndactyly (2/3 toes); connective tissue laxity; "
                "  TESTING: megalencephaly panel 500x (PIK3R2 + PIK3CA + AKT1 + AKT3 + PTEN); "
                "  TREATMENT: alpelisib (PI3K inhibitor) — trials ongoing; everolimus downstream; "
                "AKT3 — HEMIMEGALENCEPHALY (HME) PROTOCOL: "
                "  SEVERITY: most extreme mTOR brain malformation; "
                "  CLINICAL: neonatal drug-resistant epilepsy (80%+ seizure-free impossible medically); "
                "  MRI: one hemisphere grossly enlarged, dysmorphic, pachygyric; "
                "  EEG: continuous high-amplitude epileptiform activity from affected hemisphere; "
                "  FUNCTIONAL HEMISPHEROTOMY: "
                "    PROCEDURE: disconnect epileptic hemisphere while preserving blood supply; "
                "    Various techniques: lateral (Rasmussen), peri-insular (Villemure), endoscopic; "
                "    Timing: EARLY (< 2 years) = better developmental sparing of contralateral; "
                "    Seizure-free rate: 50-70%; "
                "    Hemiplegia: contralateral; pre-existing in many (may worsen acutely); "
                "    Language: transfer to ipsilateral hemisphere if early enough; "
                "  DEEP SEQUENCING PROTOCOL (AKT3): "
                "    BLOOD 500x: VAF 2-10% if early embryonic; "
                "    SALIVA/BUCCAL 500x: may be higher than blood (ectoderm contribution); "
                "    BRAIN BIOPSY (surgical specimen) 500x: highest yield 30-50% in affected cortex; "
                "    Request ALL THREE sources simultaneously; "
                "MEGALENCEPHALY PANEL — WHEN TO REQUEST: "
                "  Head circumference >+3 SD (megalencephaly); "
                "  Asymmetric head circumference; "
                "  FCD on MRI (drug-resistant); "
                "  Neonatal seizures + brain overgrowth; "
                "  Panel: AKT3 + PIK3CA + PIK3R2 + AKT1 + PTEN; all at 500x depth; "
                "  BLOOD + SALIVA + TISSUE (when available) simultaneously"
            ),
        },
        {
            "term": "Everolimus and mTOR Inhibitor Therapy — Mechanism, Dosing, and Cross-Gene Indications",
            "genes": ["TSC1", "TSC2", "MTOR", "PIK3R2", "AKT3"],
            "definition": (
                "mTOR INHIBITOR THERAPY ACROSS THE TSC/mTOR PATHWAY ATLAS: "
                "MECHANISM — WHY EVEROLIMUS/RAPAMYCIN WORKS FOR MULTIPLE GENES: "
                "  mTORC1 is the CONVERGENCE POINT of the entire pathway; "
                "  ALL 8 genes in this atlas ultimately hyperactivate mTORC1 when mutated; "
                "  Everolimus + FKBP12 → inhibits FRB domain on MTOR kinase → mTORC1 OFF; "
                "  Downstream: S6K1 dephosphorylated → protein synthesis reduced; "
                "    4E-BP1 dephosphorylated → cap-dependent translation reduced; "
                "    Autophagy restored (mTORC1 no longer inhibits ULK1); "
                "  This mechanism is GENE-AGNOSTIC: works whether hyperactivation is via: "
                "    Loss of TSC1/TSC2 (RHEB arm); "
                "    Loss of GATOR1 (RAG arm); "
                "    Gain-of-function MTOR (direct kinase); "
                "    Gain-of-function PIK3R2/AKT3 (upstream PI3K/AKT arm); "
                "FDA APPROVALS FOR EVEROLIMUS: "
                "  (1) SEGA in TSC: approved 2010 (EXIST-1 trial); "
                "    Evidence: 35% of patients ≥50% volume reduction; some complete response; "
                "  (2) Renal AML in TSC: approved 2012 (EXIST-2 trial); "
                "    Evidence: AML volume reduction; reduced haemorrhage risk; "
                "  (3) LAM (lung): approved 2012 (EXIST-LAM trial); "
                "    Evidence: stabilisation of lung function; slowed FEV1 decline; "
                "  (4) EXIST-3 (epilepsy): approved as adjunctive; "
                "    Evidence: 40% of patients ≥50% seizure reduction; "
                "DOSING AND MONITORING PROTOCOL: "
                "  STARTING DOSE: 4.5 mg/m2/day orally (adjust per weight/BSA tables); "
                "  TARGET TROUGH: 5-15 ng/mL (SEGA/AML); 5-10 ng/mL (epilepsy); "
                "  MONITORING: "
                "    CBC (myelosuppression): every 4 weeks first 3 months; 3-monthly thereafter; "
                "    LFTs: every 4 weeks; "
                "    Lipids (hypertriglyceridaemia): baseline + quarterly; "
                "    Glucose (hyperglycaemia): fasting baseline + quarterly; "
                "    Creatinine: baseline + 4-weekly; "
                "    Stomatitis: most common — daily oral hygiene; topical steroids; "
                "    Infections: avoid live vaccines; screen at each visit; "
                "  DRUG INTERACTIONS: "
                "    Strong CYP3A4 inhibitors (azole antifungals, erythromycin): increase levels; "
                "    Strong CYP3A4 inducers (rifampicin, carbamazepine, phenytoin): decrease levels; "
                "    NOTE: carbamazepine induction → everolimus level may be sub-therapeutic; "
                "SIROLIMUS (RAPAMYCIN) — ALTERNATIVE: "
                "  Same mechanism as everolimus; parent compound; "
                "  Sirolimus used in some centres for TSC/LAM (longer-established data); "
                "  Everolimus: easier dosing (daily); better paediatric evidence; "
                "EMERGING mTOR INHIBITOR INDICATIONS (NON-FDA-APPROVED): "
                "  GATOR1 epilepsy (DEPDC5/NPRL2/NPRL3): "
                "    Rationale: GATOR1 LOF → mTORC1 → seizures; "
                "    Case series + preclinical models: seizure reduction; "
                "  MTOR-FCD IIb: "
                "    Case series: everolimus reduces seizure frequency pre/post-surgery; "
                "  PIK3R2-MCAP/AKT3-HME: "
                "    Everolimus targets downstream mTORC1; some seizure benefit; "
                "    Alpelisib (PI3K inhibitor) addresses upstream PI3K for PIK3R2/AKT3; "
                "CARBAMAZEPINE INTERACTION WITH EVEROLIMUS: "
                "  Carbamazepine is a strong CYP3A4 inducer; "
                "  Dramatically reduces everolimus bioavailability; "
                "  If carbamazepine + everolimus co-prescribed: monitor levels; may need dose increase; "
                "  PREFER: levetiracetam, lamotrigine, lacosamide, clobazam (no CYP3A4 induction)"
            ),
        },
    ]
    return {
        "atlas": "Hereditary-TSC-mTOR-Pathway-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }
