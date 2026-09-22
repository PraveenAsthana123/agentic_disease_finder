#!/usr/bin/env python3
"""Hereditary-HCC-Liver-Cancer-Predisposition-Atlas — Complete 8-Gene Reference
FAH    (Fumarylacetoacetate Hydrolase; 419aa; 15q25.1; AR LOF;
         Hereditary Tyrosinemia Type 1 (HT1);
         HCC 37%+ by age 2 without NTBC — MOST EXTREME PEDIATRIC HCC RISK;
         Succinylacetone (urine/plasma) PATHOGNOMONIC;
         NTBC + low-Tyr/Phe diet standard; liver Tx curative;
         seed SEED_BASE+0) ·
ABCB11 (Bile Salt Export Pump BSEP; 1321aa; 2q31.1; AR LOF;
         Progressive Familial Intrahepatic Cholestasis Type 2 (PFIC2);
         HCC/CCA in children <5yr — HIGHEST PEDIATRIC LIVER CANCER RISK;
         BSEP-null IHC PATHOGNOMONIC;
         UDCA/4-PBA; biliary diversion; liver Tx;
         seed SEED_BASE+1) ·
HFE    (Homeostatic Iron Regulator; 343aa; 6p21.3; AR C282Y/H63D;
         Hereditary Hemochromatosis Type 1 (HH1);
         C282Y homozygote HCC 20-200x — HIGHEST ADULT IRON-OVERLOAD HCC;
         MRI T2* liver iron quantification; phlebotomy prevents if pre-cirrhosis;
         seed SEED_BASE+2) ·
ATP7B  (Copper-Transporting ATPase Beta; 1465aa; 13q14.3; AR LOF;
         Wilson's Disease (WD);
         Kayser-Fleischer rings PATHOGNOMONIC (slit-lamp MANDATORY);
         HCC 2-5x RR; D-penicillamine / trientine / zinc maintenance;
         seed SEED_BASE+3) ·
SERPINA1 (Alpha-1 Antitrypsin; 418aa; 14q32.13; AR Pi*ZZ;
         Alpha-1 Antitrypsin Deficiency (AATD);
         PASD-positive globules on liver biopsy PATHOGNOMONIC;
         HCC 5-20x RR (Pi*ZZ); lung augmentation; liver Tx curative;
         seed SEED_BASE+4) ·
APC    (Adenomatous Polyposis Coli; 2843aa; 5q22.2; AD LOF;
         FAP / Gardner Syndrome;
         Hepatoblastoma 750-7500x RR — HIGHEST hepatoblastoma gene;
         Liver USS MANDATORY from birth to age 10 (often missed);
         CHRPE PATHOGNOMONIC; prophylactic colectomy 20-25yr;
         seed SEED_BASE+5) ·
TSC2   (Tuberin; 1807aa; 16p13.3; AD LOF;
         Tuberous Sclerosis Complex (TSC2);
         Hepatic angiomyolipomas 75% bilateral/multiple — surveillance MANDATORY;
         Everolimus FDA-approved for AML regression;
         seed SEED_BASE+6) ·
SMAD4  (SMAD Family Member 4; 552aa; 18q21.2; AD LOF;
         Juvenile Polyposis Syndrome (JPS) / HHT overlap;
         SMAD4-null IHC PATHOGNOMONIC; gastric hamartomas;
         Hepatic AVMs in HHT-overlap (bevacizumab);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3190-3197)
"""
import random

SEED_BASE = 3190

ATLAS_GENES = [
    {
        "gene": "FAH",
        "protein": (
            "FAH -- 15q25.1 Autosomal-Recessive-LOF -- 419aa -- "
            "FAH-46kDa-Fumarylacetoacetase-Final-Tyrosine-Catabolism-Enzyme-"
            "HT1-Hereditary-Tyrosinemia-Type1-HCC-EXTREME-PEDIATRIC-NTBC-Standard-OMIM-276700"
        ),
        "locus": "15q25.1",
        "protein_size": (
            "419 aa / 15q25.1 FAH encodes Fumarylacetoacetate Hydrolase (FAH): "
            "STRUCTURE: "
            "  46 kDa; cytoplasmic enzyme; final step of tyrosine catabolism pathway; "
            "  FAH cleaves fumarylacetoacetate → fumarate + acetoacetate (TCA + ketone body); "
            "  LOF → fumarylacetoacetate + maleylacetoacetate accumulate → hepatotoxic alkylating agents; "
            "  Succinylacetone (SA) = FAA metabolite — PATHOGNOMONIC biomarker in urine + plasma; "
            "  SA inhibits delta-aminolevulinic acid dehydratase (ALA-D) → porphyria-like crises; "
            "HCC RISK — MOST EXTREME PEDIATRIC: "
            "  HCC risk 37%+ without treatment by age 2 — MOST EXTREME PEDIATRIC HCC risk; "
            "  Even with NTBC: residual HCC risk ~2% per year; surveillance MANDATORY for life; "
            "  HCC median onset age <3yr in untreated; with NTBC onset delayed to teen/young adult; "
            "  AFP elevation in HT1: can be VERY HIGH even without HCC (biochemical confound); "
            "  MRI liver 3-monthly + AFP correction for NTBC-treated patients (Lindstedt 1992); "
            "NTBC (NITISINONE): "
            "  NTBC (2-(2-nitro-4-trifluoromethylbenzoyl)-1,3-cyclohexanedione): "
            "  Inhibits 4-hydroxyphenylpyruvate dioxygenase → prevents FAA/SA formation; "
            "  FDA/EMA approved; reduces HCC risk >90% vs untreated; started at diagnosis; "
            "DIAGNOSIS: "
            "  NBS: succinylacetone in dried blood spot — most reliable NBS marker; "
            "  Plasma succinylacetone + urine delta-ALA elevated; "
            "  Liver/kidney dysfunction neonatal/infantile; "
            "  Molecular: FAH sequencing; IVS12+5GA splice variant common in Quebec; "
            "LIVER TRANSPLANTATION: "
            "  Curative for liver disease and HCC risk; "
            "  Indication: HCC, NTBC non-responder, cirrhosis with portal hypertension; "
            "  Post-Tx: NTBC can be discontinued; renal tubular dysfunction may persist"
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF required; de novo extremely rare; consanguinity increases risk",
        "cancer_risk": "HCC 37%+ by age 2 (untreated) — EXTREME; NTBC-treated: ~2%/yr residual; Hepatoblastoma rare",
        "pathognomonic": "Succinylacetone in urine/plasma PATHOGNOMONIC; delta-ALA elevation; NBS DBS spot SA = HT1 until proven otherwise",
        "surveillance_key": "MRI liver 3-monthly + AFP from diagnosis (NTBC-era); lifelong surveillance even post-NTBC normalisation; liver Tx if HCC/non-response",
        "key_distinctions": [
            "SUCCINYLACETONE-PATHOGNOMONIC-URINE-PLASMA",
            "HCC-37PCT-BY-AGE2-EXTREME-PEDIATRIC",
            "NTBC-STANDARD-TREATMENT",
            "AFP-CONFOUNDED-BY-HT1-BIOCHEMISTRY",
            "MRI-LIVER-3-MONTHLY-LIFELONG",
            "LIVER-TX-CURATIVE-HCC-RISK",
        ],
    },
    {
        "gene": "ABCB11",
        "protein": (
            "ABCB11 -- 2q31.1 Autosomal-Recessive-LOF -- 1321aa -- "
            "ABCB11-146kDa-Bile-Salt-Export-Pump-BSEP-Canalicular-Membrane-ABC-Transporter-"
            "PFIC2-HCC-CCA-Children-HIGHEST-BSEP-null-IHC-PATHOGNOMONIC-OMIM-605479"
        ),
        "locus": "2q31.1",
        "protein_size": (
            "1321 aa / 2q31.1 ABCB11 encodes Bile Salt Export Pump (BSEP): "
            "STRUCTURE: "
            "  146 kDa; ABC transporter superfamily (ABCB subfamily); canalicular membrane; "
            "  12 transmembrane helices; 2 nucleotide-binding domains (NBD1/2); "
            "  Responsible for >80% of bile acid export from hepatocyte to bile canaliculus; "
            "  LOF → bile salt retention within hepatocytes → detergent injury → cirrhosis; "
            "HCC/CCA RISK — HIGHEST PEDIATRIC LIVER CANCER: "
            "  HCC in PFIC2: onset <5yr — HIGHEST PEDIATRIC liver cancer risk of any gene; "
            "  CCA (cholangiocarcinoma) also reported in children and young adults with PFIC2; "
            "  Bile salt retention activates FXR-independent oncogenic pathways; "
            "  Without transplant: HCC median age 2-5yr; after Tx: no recurrence if successful; "
            "PFIC2 CLINICAL: "
            "  Progressive cholestasis from neonatal period; pruritus severe; "
            "  GGT characteristically NORMAL (DDx PFIC1: GGT also low; PFIC3: GGT HIGH); "
            "  ALT/AST markedly elevated; conjugated bilirubin elevated; "
            "  Growth failure; rickets (fat-soluble vitamin deficiency); "
            "BSEP-NULL IHC — PATHOGNOMONIC: "
            "  BSEP immunohistochemistry on liver biopsy: null/absent = PFIC2 PATHOGNOMONIC; "
            "  Contrast with PFIC1 (ATP8B1): BSEP present but mislocalised; "
            "  Contrast with PFIC3 (ABCB4): BSEP present; MDR3 absent; "
            "TREATMENT: "
            "  UDCA (ursodeoxycholic acid): symptom relief; not curative; "
            "  4-PBA (4-phenylbutyric acid): partial BSEP rescue for trafficking mutants; "
            "  Maralixibat / odevixibat (ileal bile acid transporter inhibitors): reduce pruritus; "
            "  Biliary diversion (partial internal/external): delays transplant; "
            "  Liver transplantation: definitive — curative for PFIC2 liver disease and HCC risk; "
            "  Post-Tx HCC surveillance: continue 2yr post-Tx (residual HCC risk)"
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF; ABCB11 heterozygous carriers: drug-induced cholestasis risk (e.g. hormonal, antibiotics)",
        "cancer_risk": "HCC/CCA <5yr — HIGHEST PEDIATRIC; even partial PFIC2 (heterozygous+modifier): HCC risk elevated; post-Tx: low risk",
        "pathognomonic": "BSEP-null IHC on liver biopsy PATHOGNOMONIC; GGT characteristically normal in PFIC2 (DDx: PFIC3 GGT very high; PFIC1 GGT low but BSEP present)",
        "surveillance_key": "6-monthly liver MRI + AFP from diagnosis; biliary diversion if possible pre-Tx; liver Tx if HCC/cirrhosis/non-response; post-Tx surveillance 2yr",
        "key_distinctions": [
            "BSEP-NULL-IHC-PATHOGNOMONIC",
            "HCC-CCA-CHILDREN-HIGHEST-PEDIATRIC",
            "GGT-NORMAL-PATHOGNOMONIC-PFIC2-vs-PFIC3",
            "MARALIXIBAT-ODEVIXIBAT-IBAT-INHIBITORS",
            "LIVER-TX-CURATIVE",
            "HETEROZYGOUS-DRUG-INDUCED-CHOLESTASIS-RISK",
        ],
    },
    {
        "gene": "HFE",
        "protein": (
            "HFE -- 6p21.3 Autosomal-Recessive-C282Y-H63D -- 343aa -- "
            "HFE-37kDa-MHC-Class1-Like-Hepcidin-Regulator-"
            "HH1-Hereditary-Hemochromatosis-C282Y-Homozygote-HCC-20-200x-HIGHEST-IRON-OVERLOAD-OMIM-235200"
        ),
        "locus": "6p21.3",
        "protein_size": (
            "343 aa / 6p21.3 HFE encodes Hereditary Hemochromatosis Protein (HFE): "
            "STRUCTURE: "
            "  37 kDa; MHC class I-like structure; non-classical HLA; associated with beta-2-microglobulin; "
            "  Interacts with transferrin receptor (TfR1) to sense body iron; "
            "  HFE-TfR1 interaction signals hepatic hepcidin production; "
            "  LOF → impaired hepcidin upregulation → unregulated intestinal iron absorption → iron overload; "
            "VARIANTS: "
            "  p.C282Y (c.845GA): most common pathogenic — disrupts disulfide bond, MHC fold lost; "
            "    C282Y/C282Y homozygote: ~0.5% Northern European; penetrance ~30-50% in males; "
            "  p.H63D (c.187CA): second most common; mild; C282Y/H63D compound: intermediate; "
            "  p.S65C: minor modifier variant; rarely pathogenic alone; "
            "HCC RISK — HIGHEST ADULT IRON-OVERLOAD HCC: "
            "  C282Y/C282Y homozygote HCC: 20-200x RR — HIGHEST HCC risk from iron overload; "
            "  Absolute lifetime risk: ~6-10% in males with cirrhosis; females lower (oestrogen protective); "
            "  HCC risk driven by cirrhosis AND direct iron-mediated DNA damage; "
            "  HCC can arise WITHOUT cirrhosis in HFE (unique vs other liver diseases); "
            "  HCC surveillance: 6-monthly USS ± AFP in ALL C282Y/C282Y with F3-4 fibrosis; "
            "MANAGEMENT: "
            "  Phlebotomy: weekly initially; target ferritin <50 mcg/L and transferrin sat <50%; "
            "  PREVENTS HCC if started PRE-CIRRHOSIS — key counselling point; "
            "  MRI T2* (IDEAL sequence): liver iron concentration quantification (non-invasive); "
            "  Ferritin >1000 mcg/L → liver biopsy for fibrosis staging (or FibroScan); "
            "  Erythrocytapheresis: faster iron removal in high-ferritin; "
            "  Chelation (deferoxamine/deferasirox): second-line only; "
            "FAMILY SCREENING: "
            "  All first-degree relatives of C282Y/C282Y: HFE genotype + transferrin saturation + ferritin; "
            "  Paediatric screening deferred to adulthood (low penetrance in minors)"
        ),
        "inheritance": "Autosomal Recessive (AR) for clinical hemochromatosis; C282Y/C282Y homozygote or C282Y/H63D compound; incomplete penetrance (~30-50% males)",
        "cancer_risk": "HCC 20-200x RR (C282Y/C282Y + cirrhosis) — HIGHEST; HCC without cirrhosis possible; hepatocellular adenoma rare",
        "pathognomonic": "C282Y/C282Y homozygosity + high transferrin saturation (>45%) + elevated ferritin; MRI T2* diffuse hepatic iron; Perl's Prussian blue GRADE 3-4 on biopsy",
        "surveillance_key": "Phlebotomy to ferritin <50 mcg/L; MRI T2* liver iron staging; biopsy if ferritin >1000; 6-monthly USS + AFP if cirrhosis; family first-degree genotyping",
        "key_distinctions": [
            "C282Y-HOMOZYGOTE-HCC-20-200X-HIGHEST-ADULT",
            "PHLEBOTOMY-PREVENTS-HCC-IF-PRE-CIRRHOSIS",
            "MRI-T2STAR-LIVER-IRON-QUANTIFICATION",
            "HCC-WITHOUT-CIRRHOSIS-UNIQUE-HFE",
            "FERRITIN-1000-BIOPSY-THRESHOLD",
            "TRANSFERRIN-SATURATION-SCREEN-FIRST-DEGREE",
        ],
    },
    {
        "gene": "ATP7B",
        "protein": (
            "ATP7B -- 13q14.3 Autosomal-Recessive-LOF -- 1465aa -- "
            "ATP7B-160kDa-Copper-Transporting-P-Type-ATPase-Trans-Golgi-Network-Canalicular-"
            "Wilson-Disease-WD-Kayser-Fleischer-PATHOGNOMONIC-HCC-2-5x-D-Penicillamine-Trientine-OMIM-277900"
        ),
        "locus": "13q14.3",
        "protein_size": (
            "1465 aa / 13q14.3 ATP7B encodes Copper-Transporting ATPase Beta (ATP7B): "
            "STRUCTURE: "
            "  160 kDa; P-type ATPase; 8 transmembrane helices; trans-Golgi network to canalicular membrane; "
            "  6 copper-binding CXXC motifs in N-terminal metal-binding domain; "
            "  LOF → copper accumulates in liver → hepatotoxicity → overflow to brain, kidney, cornea; "
            "  Hepatic copper deposition → oxidative stress → mitochondrial dysfunction → cirrhosis; "
            "HCC RISK: "
            "  HCC in Wilson's Disease: 2-5x RR; HCC can precede clinical diagnosis; "
            "  HCC rare in adequately treated WD; risk concentrated in cirrhotic patients; "
            "  Copper-mediated HCC: direct DNA damage + oxidative mutagenesis; "
            "  Annual USS + AFP in cirrhotic WD patients; "
            "KAYSER-FLEISCHER RINGS — PATHOGNOMONIC: "
            "  Copper deposits in Descemet's membrane of peripheral cornea; "
            "  KF rings = PATHOGNOMONIC when present — but absent in 50% of hepatic WD; "
            "  KF rings absent in hepatic-only presentation — SLIT-LAMP MANDATORY; "
            "  Slit-lamp by experienced ophthalmologist: gold standard; "
            "  KF rings disappear with chelation (improvement marker); "
            "CLINICAL PRESENTATIONS: "
            "  Hepatic WD (40%): ALF, chronic hepatitis, cirrhosis — any age >5yr; "
            "  Neuropsychiatric WD (50%): dysarthria, tremor, dystonia, psychiatric; "
            "  Combined: severe; KF rings more common in neurological; "
            "DIAGNOSIS: "
            "  Ceruloplasmin <0.1 g/L + KF rings = WD until proven otherwise; "
            "  24h urine copper >100 mcg/24h (>200 with D-pen challenge); "
            "  Liver copper >250 mcg/g dry weight; "
            "  Leipzig Score ≥4 = WD (ceruloplasmin + KF + urine Cu + histology + molecular); "
            "TREATMENT: "
            "  D-penicillamine (chelator): first-line; SE: nephrotoxicity, lupus, skin changes; "
            "  Trientine (TETA): better tolerated; chelator; preferred in neurological WD; "
            "  Zinc: maintenance phase; blocks intestinal copper absorption; "
            "  Tetrathiomolybdate (TTM): investigational; neuroprotective; "
            "  Liver Tx: ALF or decompensated cirrhosis — curative; liver disease resolved post-Tx; "
            "  Neurological WD post-Tx: incomplete recovery — Tx prevents further deterioration"
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF; p.His1069Gln (c.3207CA) most common in European (30-50%); Arg778Leu most common in East Asian",
        "cancer_risk": "HCC 2-5x RR; risk confined to cirrhotic patients; adequately treated WD: low HCC risk; KF rings DDx: primary biliary cholangitis (PBC), neonatal cholestasis",
        "pathognomonic": "Kayser-Fleischer rings PATHOGNOMONIC (slit-lamp); ceruloplasmin <0.1 g/L; 24h urine copper >100 mcg; liver copper >250 mcg/g dry weight",
        "surveillance_key": "Slit-lamp annually (KF rings treatment response); 24h urine copper 3-monthly during treatment; annual USS + AFP if cirrhosis; ceruloplasmin + LFTs 6-monthly",
        "key_distinctions": [
            "KAYSER-FLEISCHER-RINGS-PATHOGNOMONIC-SLIT-LAMP-MANDATORY",
            "HCC-2-5X-CIRRHOTIC-WD",
            "KF-ABSENT-50PCT-HEPATIC-WD-SLIT-LAMP-STILL-MANDATORY",
            "D-PENICILLAMINE-FIRST-LINE-CHELATOR",
            "TRIENTINE-PREFERRED-NEUROLOGICAL-WD",
            "LIVER-TX-ALF-CURATIVE",
        ],
    },
    {
        "gene": "SERPINA1",
        "protein": (
            "SERPINA1 -- 14q32.13 Autosomal-Recessive-Pi*ZZ -- 418aa -- "
            "SERPINA1-52kDa-Alpha1-Antitrypsin-Serine-Protease-Inhibitor-"
            "AATD-PiZZ-PASD-Globules-Liver-Biopsy-PATHOGNOMONIC-HCC-5-20x-Augmentation-Liver-Tx-OMIM-613490"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "418 aa / 14q32.13 SERPINA1 encodes Alpha-1 Antitrypsin (AAT): "
            "STRUCTURE: "
            "  52 kDa secreted serine protease inhibitor (serpin); "
            "  Major serum protease inhibitor; neutralises neutrophil elastase in lung; "
            "  Pi*Z allele (p.Glu342Lys, c.1024GA): Z-AAT misfolds → polymerises → ER retention; "
            "  Pi*ZZ homozygote: Z-AAT accumulates in hepatocyte ER as periodic globules; "
            "  ER stress → mitochondrial dysfunction → hepatocyte death → cirrhosis; "
            "  Lung: reduced circulating AAT → uninhibited elastase → emphysema; "
            "PASD GLOBULES — PATHOGNOMONIC: "
            "  Periodic Acid-Schiff with Diastase (PASD): diastase removes glycogen; "
            "  Z-AAT globules resist diastase digestion → PASD-positive globules REMAIN; "
            "  Periportal PASD-positive eosinophilic globules = AATD PATHOGNOMONIC; "
            "  Absent in Pi*MZ heterozygotes; quantity correlates with disease severity; "
            "HCC RISK: "
            "  Pi*ZZ: HCC 5-20x RR; absolute lifetime risk ~2-5% in males with cirrhosis; "
            "  HCC risk mainly in cirrhotic AATD; some HCC without cirrhosis reported; "
            "  Surveillance: 6-monthly USS + AFP in all Pi*ZZ with F3-4 fibrosis; "
            "  Sorafenib/lenvatinib: standard HCC SYSTEMIC THERAPY; "
            "LUNG DISEASE: "
            "  Panacinar emphysema basal-predominant; onset 30-40yr smokers, 50-60yr non-smokers; "
            "  Lung function: FEV1/FVC obstructive defect; TLCO reduced; "
            "AUGMENTATION THERAPY: "
            "  IV purified AAT (ARALAST, ZEMAIRA, PROLASTIN-C): weekly infusions; "
            "  Replaces deficient circulating AAT → NE inhibition in lung; "
            "  Slows lung decline (RAPID trial); does NOT address liver disease; "
            "LIVER TRANSPLANTATION: "
            "  Curative for AATD liver disease AND HCC risk; "
            "  Post-Tx: recipient produces donor (normal) AAT — AATD lung risk reduced; "
            "  Eligibility: cirrhosis + portal HTN, refractory symptoms; "
            "EMERGING THERAPIES: "
            "  Fazirsiran (ARO-AAT): RNAi silencing of Z-AAT production (Phase 3); "
            "  Gene therapy: AAV-based liver-directed correction (clinical trials); "
            "FAMILY SCREENING: "
            "  Pi phenotyping / SERPINA1 genotyping of first-degree relatives; "
            "  Neonatal diagnosis changes outcome → NBS in Pi*ZZ families"
        ),
        "inheritance": "Autosomal Recessive for severe disease (Pi*ZZ); Pi*MZ intermediate (carrier + increased risk); phenotypic heterogeneity; Pi*ZZ = 1 in 2000-5000 in European",
        "cancer_risk": "HCC 5-20x RR (Pi*ZZ + cirrhosis); HCC without cirrhosis possible; hepatocellular carcinoma in transplant recipients: rare (donor AAT normal post-Tx)",
        "pathognomonic": "PASD-positive periportal globules on liver biopsy PATHOGNOMONIC; serum AAT <0.57 g/L (Pi*ZZ); Pi phenotyping (isoelectric focusing) or SERPINA1 genotyping",
        "surveillance_key": "6-monthly USS + AFP if cirrhosis/F3-4; spirometry + DLCO annually (lung); augmentation infusions for lung disease; liver Tx if cirrhosis/HCC; fazirsiran Phase 3 enrollment",
        "key_distinctions": [
            "PASD-GLOBULES-PATHOGNOMONIC-LIVER-BIOPSY",
            "HCC-5-20X-Pi-ZZ-CIRRHOSIS",
            "AUGMENTATION-ADDRESSES-LUNG-NOT-LIVER",
            "LIVER-TX-CURATIVE-AATD",
            "FAZIRSIRAN-RNAi-EMERGING-THERAPY",
            "NEONATAL-DIAGNOSIS-CHANGES-OUTCOME",
        ],
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "APC-310kDa-WNT-Pathway-Gatekeeper-Beta-Catenin-Destruction-Complex-"
            "FAP-Gardner-Hepatoblastoma-750-7500x-HIGHEST-CHRPE-PATHOGNOMONIC-Prophylactic-Colectomy-OMIM-175100"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 5q22.2 APC encodes Adenomatous Polyposis Coli Protein (APC): "
            "STRUCTURE: "
            "  310 kDa; scaffolding protein; forms beta-catenin destruction complex with AXIN1/2, CK1, GSK3B; "
            "  APC targets beta-catenin for ubiquitin-proteasome degradation; "
            "  LOF → free beta-catenin enters nucleus → TCF/LEF transcription → WNT target genes → proliferation; "
            "  'Two-hit' mechanism: germline + somatic second-hit; second allele loss drives adenoma; "
            "HEPATOBLASTOMA — HIGHEST RISK: "
            "  APC germline: hepatoblastoma 750-7500x RR — HIGHEST hepatoblastoma risk of ANY gene; "
            "  Absolute risk: ~1.5-2% lifetime in FAP (0-15yr); "
            "  Onset typically 0-10yr; peak age 18 months; AFP markedly elevated; "
            "  Liver USS from BIRTH to age 10yr — often MISSED in FAP families; "
            "  HCC in adults with FAP: less common (~1.5x RR); "
            "CHRPE — PATHOGNOMONIC: "
            "  Congenital Hypertrophy of the Retinal Pigment Epithelium (CHRPE); "
            "  Multiple bilateral CHRPE (>4 lesions) = APC germline PATHOGNOMONIC; "
            "  Present from birth; ophthalmoscopy pre-genetic test in classic FAP families; "
            "  CHRPE absent in attenuated FAP (AFAP) — genotype/phenotype correlation; "
            "FAP EXTRACOLONIC: "
            "  Desmoid tumours: 10-20%; mesenteric > abdominal wall; APC codon 1310-1400 mutation risk; "
            "  Duodenal/periampullary polyps: surveillance MANDATORY (Spigelman staging); "
            "  Papillary thyroid cancer: 1-2%; annual thyroid USS from 20yr; "
            "  Gastric fundic gland polyps (benign, but surveillance gastroscopy); "
            "  Osteomas (Gardner triad); supernumerary teeth; epidermoid cysts; "
            "MANAGEMENT: "
            "  Prophylactic colectomy: 20-25yr (or earlier if dense polyposis/cancer); "
            "  IRA (ileorectal anastomosis) vs IPAA (pouch): APC codon location influences rectal sparing; "
            "  Sulindac/celecoxib: polyp number reduction (not prevention); adjunct only; "
            "  Erlotinib+sulindac: EGFR + COX2 inhibition; desmoid trials; "
            "  Hepatoblastoma: cisplatin-based chemotherapy + resection; liver Tx if unresectable"
        ),
        "inheritance": "Autosomal Dominant (AD); 25-30% de novo; penetrance ~100% for polyps; 5q22.2 deletion: MLPA required when sequencing negative",
        "cancer_risk": "Hepatoblastoma 750-7500x RR (children 0-10yr) — HIGHEST; CRC ~100% lifetime without colectomy; Duodenal cancer 4-12%; Desmoid 10-20%; Thyroid 1-2%",
        "pathognomonic": "Multiple bilateral CHRPE (≥4 bilateral lesions) PATHOGNOMONIC; colonic polyposis (>100 adenomas) classic; desmoid + polyposis = Gardner Syndrome",
        "surveillance_key": "Liver USS from birth to age 10 (hepatoblastoma); AFP baseline; colonoscopy annually from 12-15yr; prophylactic colectomy 20-25yr; Spigelman duodenal staging; annual thyroid USS from 20yr",
        "key_distinctions": [
            "HEPATOBLASTOMA-750-7500X-HIGHEST-CHILDREN-0-10YR",
            "LIVER-USS-FROM-BIRTH-MANDATORY-OFTEN-MISSED",
            "CHRPE-MULTIPLE-BILATERAL-PATHOGNOMONIC",
            "PROPHYLACTIC-COLECTOMY-20-25YR",
            "DESMOID-CODON-1310-1400-RISK",
            "SPIGELMAN-DUODENAL-STAGING-MANDATORY",
        ],
    },
    {
        "gene": "TSC2",
        "protein": (
            "TSC2 -- 16p13.3 Autosomal-Dominant-LOF -- 1807aa -- "
            "TSC2-200kDa-Tuberin-GAP-Rheb-mTORC1-Inhibitor-"
            "TSC-Tuberous-Sclerosis-Complex-Hepatic-Angiomyolipomas-75pct-Bilateral-Everolimus-FDA-OMIM-613254"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "1807 aa / 16p13.3 TSC2 encodes Tuberin (TSC2): "
            "STRUCTURE: "
            "  200 kDa; GTPase-activating protein (GAP) for Rheb GTPase; "
            "  Forms heterodimer with Hamartin (TSC1) — TSC1-TSC2 complex; "
            "  TSC1/TSC2 complex inhibits Rheb → inhibits mTORC1; "
            "  LOF → constitutive mTORC1 activation → uncontrolled cell growth and proliferation; "
            "HEPATIC ANGIOMYOLIPOMAS — SURVEILLANCE MANDATORY: "
            "  Hepatic AMLs: 75% of TSC patients; bilateral, multiple; female predominant; "
            "  Large AMLs (>3cm): haemorrhage risk — spontaneous life-threatening haemorrhage; "
            "  Annual liver MRI: size monitoring (>3cm = intervention threshold); "
            "  Haemorrhage predictors: fat-poor AML > fat-rich; aneurysm within AML; "
            "  Surveillance: baseline MRI at diagnosis; annual if AML present; "
            "EVEROLIMUS — FDA APPROVED: "
            "  mTOR inhibitor (rapamycin analogue); FDA-approved for hepatic AMLs in TSC (2012); "
            "  AML regression: ~50% reduction in volume with everolimus; "
            "  On discontinuation: AMLs regrow — maintenance dosing; "
            "  Also FDA-approved: TSC-SEGA (subependymal giant cell astrocytoma); renal AML; SLAML; "
            "  Sirolimus (rapamycin): older mTORi; LAM-specific (sirolimus for LAM non-TSC); "
            "OTHER TSC MANIFESTATIONS: "
            "  Renal AMLs: 80%; bilateral; >3cm → everolimus or embolisation; RCC: 2-4% (mainly clear cell); "
            "  Pulmonary LAM (lymphangioleiomyomatosis): females >20yr; progressive dyspnoea; "
            "  SEGA: subependymal giant cell astrocytoma; Foramen of Monro → hydrocephalus; "
            "  Facial angiofibromas (adenoma sebaceum): PATHOGNOMONIC; "
            "  Shagreen patch (connective tissue hamartoma): PATHOGNOMONIC; "
            "  Cortical tubers: epilepsy in 85%; cognitive impairment; "
            "  Dental enamel pits (≥3): PATHOGNOMONIC; "
            "TSC1 vs TSC2: "
            "  TSC2 mutations: more severe phenotype; higher SEGA, renal AML, LAM, cognitive impairment; "
            "  TSC1 mutations: milder; TSC2 de novo rate ~60-70% (vs TSC1 ~10-15%)"
        ),
        "inheritance": "Autosomal Dominant (AD); LOF; TSC2 de novo ~60-70%; TSC1/TSC2 somatic mosaicism in ~15% of apparently de novo TSC (low-level somatic — WGS-level sensitivity required)",
        "cancer_risk": "Hepatic AML (benign, haemorrhage risk not malignancy); RCC 2-4% (clear cell); SEGA (locally aggressive); LAM (pulmonary); HCC/HCA rare; biliary hamartoma rare",
        "pathognomonic": "Facial angiofibromas (adenoma sebaceum) PATHOGNOMONIC; Shagreen patch PATHOGNOMONIC; ≥3 dental enamel pits PATHOGNOMONIC; cortical tubers on MRI; bilateral renal AMLs on imaging",
        "surveillance_key": "Annual liver MRI (AML size; >3cm = everolimus); annual renal MRI (AML/RCC); annual spirometry for LAM females; MRI brain biennial for SEGA; ophthalmology for retinal hamartoma; everolimus for AMLs >3cm",
        "key_distinctions": [
            "HEPATIC-AML-75PCT-BILATERAL-SURVEILLANCE-MANDATORY",
            "EVEROLIMUS-FDA-APPROVED-AML-REGRESSION",
            "AML-GREATER-3CM-HEMORRHAGE-RISK-THRESHOLD",
            "FACIAL-ANGIOFIBROMAS-PATHOGNOMONIC",
            "TSC2-MORE-SEVERE-THAN-TSC1",
            "DE-NOVO-60-70PCT-TSC2",
        ],
    },
    {
        "gene": "SMAD4",
        "protein": (
            "SMAD4 -- 18q21.2 Autosomal-Dominant-LOF -- 552aa -- "
            "SMAD4-60kDa-Common-Mediator-SMAD-TGFbeta-BMP-Signal-Transducer-"
            "JPS-Juvenile-Polyposis-HHT-Overlap-SMAD4-null-IHC-PATHOGNOMONIC-Gastric-Hamartomas-Hepatic-AVMs-OMIM-175050"
        ),
        "locus": "18q21.2",
        "protein_size": (
            "552 aa / 18q21.2 SMAD4 encodes SMAD Family Member 4 (SMAD4): "
            "STRUCTURE: "
            "  60 kDa; MH1 domain (DNA binding); MH2 domain (SMAD complex formation); "
            "  'Common mediator SMAD' (Co-SMAD): shared signal transducer for TGF-beta AND BMP; "
            "  TGF-beta → pSMAD2/3 complex with SMAD4 → nucleus → anti-proliferative targets; "
            "  BMP → pSMAD1/5/9 complex with SMAD4 → nucleus → HJV/hepcidin regulation; "
            "  LOF → loss of TGF-beta tumour suppression → colonic hamartoma/adenoma → CRC; "
            "HHT OVERLAP — HEPATIC AVMs: "
            "  SMAD4 mutations in ~15-20% of Juvenile Polyposis Syndrome (JPS); "
            "  SMAD4-JPS: combined JPS+HHT phenotype (HHT type 5, JPS-HHT); "
            "  Hepatic AVMs (hepatic arteriovenous malformations): hepatic HHT manifestation; "
            "  Hepatic AVMs → high-output cardiac failure; biliary ischaemia; portal HTN; "
            "  Bevacizumab (anti-VEGF): for severe hepatic HHT AVMs (reduce cardiac failure); "
            "  Liver Tx: definitive for severe hepatic HHT (SMAD4-JPS); "
            "SMAD4-NULL IHC — PATHOGNOMONIC: "
            "  SMAD4 immunohistochemistry on colonic/gastric polyp tissue: "
            "    Null/absent staining = JPS with SMAD4 germline mutation PATHOGNOMONIC; "
            "  Contrast with BMPR1A-JPS: SMAD4 IHC intact; "
            "  Cascade: SMAD4-null polyp IHC → germline testing MANDATORY; "
            "GASTRIC HAMARTOMAS — JPS-GASTRIC: "
            "  SMAD4 mutations associated with DIFFUSE GASTRIC POLYPOSIS (JPS-gastric subtype); "
            "  Severe gastric polyposis → protein-losing enteropathy; "
            "  Gastrectomy may be required for severe JPS-gastric (SMAD4 genotype); "
            "JPS MANAGEMENT: "
            "  Colonoscopy + gastroscopy annually from diagnosis (age 15 or earlier if symptomatic); "
            "  Polypectomy for hamartomas >10mm or dysplastic; "
            "  Colectomy/gastrectomy if unmanageable polyp burden; "
            "  CRC risk: 39-68% lifetime; gastric cancer risk: 21% (SMAD4 JPS-gastric); "
            "HHT MANAGEMENT: "
            "  HHES epistaxis: tranexamic acid; tamponade; laser/septoplasty; bevacizumab IV; "
            "  Pulmonary AVMs: embolisation (first-line); "
            "  Cerebral AVMs: neurosurgical / GK radiosurgery; "
            "  Hepatic AVMs: bevacizumab 5mg/kg q2w (reduces cardiac output); liver Tx if severe; "
            "  Echocardiography + CT chest for PAVMs at diagnosis"
        ),
        "inheritance": "Autosomal Dominant (AD); LOF; SMAD4 germline mutation in ~15-20% of JPS families; de novo ~20%; BMPR1A: other 25-40% of JPS (same phenotype, IHC SMAD4 intact)",
        "cancer_risk": "CRC 39-68% lifetime (JPS); Gastric cancer 21% (SMAD4-JPS-gastric subtype); Duodenal/small bowel cancer 5-10%; hepatic AVM (non-malignant, cardiac failure risk); pancreatic cancer 2-5%",
        "pathognomonic": "SMAD4-null IHC on colonic/gastric polyp PATHOGNOMONIC for SMAD4-JPS; JPS + HHT features (epistaxis, telangiectasias, PAVMs) = JPS-HHT/SMAD4 syndrome until proven otherwise",
        "surveillance_key": "Annual colonoscopy + gastroscopy from diagnosis; echocardiography + CT chest for PAVMs; hepatic MRI for hepatic AVMs; bevacizumab for hepatic HHT; colectomy/gastrectomy if polyp burden unmanageable; liver Tx for severe hepatic HHT",
        "key_distinctions": [
            "SMAD4-NULL-IHC-PATHOGNOMONIC-JPS",
            "JPS-HHT-OVERLAP-SMAD4-SPECIFIC",
            "HEPATIC-AVMs-BEVACIZUMAB-ANTI-VEGF",
            "GASTRIC-POLYPOSIS-GASTRECTOMY-SMAD4",
            "CRC-39-68PCT-GASTRIC-21PCT",
            "LIVER-TX-SEVERE-HEPATIC-HHT",
        ],
    },
]


def _make_patients(gene_entry):
    """Deterministic synthetic cohort: 40 patients per gene."""
    rng  = random.Random(gene_entry["locus"] + gene_entry["gene"])
    seed = SEED_BASE + ATLAS_GENES.index(gene_entry)
    rng  = random.Random(seed)

    gene = gene_entry["gene"]
    # Age-of-onset distributions tailored per gene
    age_params = {
        "FAH":      (2,  4),
        "ABCB11":   (3,  4),
        "HFE":      (52, 10),
        "ATP7B":    (28, 12),
        "SERPINA1": (48, 10),
        "APC":      (38, 12),
        "TSC2":     (32, 12),
        "SMAD4":    (35, 12),
    }
    mu, sigma = age_params.get(gene, (45, 12))

    # Liver cancer event rates
    liver_cancer_rates = {
        "FAH":      0.92,  # HCC/HBL: extreme without NTBC; near universal
        "ABCB11":   0.85,  # HCC/CCA children: very high in cohort
        "HFE":      0.72,  # C282Y/C282Y + cirrhosis: high in affected cohort
        "ATP7B":    0.48,  # HCC: moderate in cirrhotic WD
        "SERPINA1": 0.58,  # HCC: Pi*ZZ cirrhosis
        "APC":      0.62,  # hepatoblastoma + HCC: high in children
        "TSC2":     0.25,  # hepatic AMLs not HCC; HCC rare
        "SMAD4":    0.38,  # hepatic AVMs not HCC; CRC/gastric dominant
    }
    hcc_rate = liver_cancer_rates.get(gene, 0.5)

    patients = []
    for i in range(40):
        age       = max(0, round(rng.gauss(mu, sigma), 1))
        hcc_event = rng.random() < hcc_rate
        patients.append({
            "id":        f"{gene}-{i+1:02d}",
            "age_onset": age,
            "liver_ca":  hcc_event,
            "seed":      seed,
        })
    return patients


def generate_overview():
    rows = []
    for g in ATLAS_GENES:
        pts = _make_patients(g)
        liver_n = sum(1 for p in pts if p["liver_ca"])
        rows.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "liver_ca_n":       liver_n,
            "liver_ca_pct":     round(liver_n / len(pts) * 100, 1),
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
            "inheritance":      g["inheritance"],
            "cancer_risk":      g["cancer_risk"],
            "protein":          g["protein"],
        })

    total_pts    = sum(r["n"]         for r in rows)
    total_lca_n  = sum(r["liver_ca_n"] for r in rows)
    highest_gene = max(rows, key=lambda r: r["liver_ca_pct"])

    return {
        "atlas":             "Hereditary-HCC-Liver-Cancer-Predisposition-Atlas",
        "seed_range":        f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes_n":           len(ATLAS_GENES),
        "total_patients":    total_pts,
        "liver_ca_total_n":  total_lca_n,
        "liver_ca_total_pct": round(total_lca_n / total_pts * 100, 1),
        "highest_risk_gene": highest_gene["gene"],
        "highest_risk_pct":  highest_gene["liver_ca_pct"],
        "gene_summary":      rows,
        "genes_detail": [
            {
                "gene":        g["gene"],
                "inheritance": g["inheritance"],
                "cancer_risk": g["cancer_risk"],
                "pathognomonic": g["pathognomonic"],
                "surveillance_key": g["surveillance_key"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_breakdown():
    breakdown = []
    for g in ATLAS_GENES:
        pts      = _make_patients(g)
        seed_idx = ATLAS_GENES.index(g)
        lca_n    = sum(1 for p in pts if p["liver_ca"])
        breakdown.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "seed":             SEED_BASE + seed_idx,
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "liver_ca_n":       lca_n,
            "liver_ca_pct":     round(lca_n / len(pts) * 100, 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
        })
    return {"atlas": "Hereditary-HCC-Liver-Cancer-Predisposition-Atlas", "breakdown": breakdown}


def generate_definitions():
    defs = [
        {
            "term": "FAH / HT1 / HCC-37%-AGE2-EXTREME / SUCCINYLACETONE-PATHOGNOMONIC / NTBC-STANDARD",
            "definition": (
                "FAH — 419aa / 46 kDa / 15q25.1 / AR LOF\n"
                "HT1 — HCC 37%+ by age 2 (untreated) EXTREME; succinylacetone PATHOGNOMONIC; NTBC standard.\n\n"
                "SUCCINYLACETONE — PATHOGNOMONIC:\n"
                "  Succinylacetone in urine OR plasma = HT1 PATHOGNOMONIC.\n"
                "  NBS dried blood spot (DBS) succinylacetone = most reliable NBS marker for HT1.\n"
                "  SA inhibits ALA-D → delta-ALA elevation → porphyria-like neurological crises.\n\n"
                "HCC — MOST EXTREME PEDIATRIC RISK:\n"
                "  HCC: 37%+ by age 2 WITHOUT NTBC = MOST EXTREME pediatric liver cancer risk.\n"
                "  Even with NTBC: ~2%/yr residual HCC risk; MRI liver 3-monthly + AFP lifelong.\n"
                "  AFP elevated in HT1 biochemically: confound — AFP elevation ≠ HCC without imaging.\n\n"
                "NTBC TREATMENT:\n"
                "  NTBC (nitisinone) + low-Tyr/Phe diet: standard of care from NBS diagnosis.\n"
                "  NTBC >90% reduction in HCC risk vs untreated; started at birth in NBS programmes.\n"
                "  Liver Tx: curative for HCC, NTBC non-responders, end-stage cirrhosis."
            ),
        },
        {
            "term": "ABCB11 / PFIC2 / HCC-CCA-CHILDREN-HIGHEST / BSEP-NULL-IHC-PATHOGNOMONIC / GGT-NORMAL",
            "definition": (
                "ABCB11 (BSEP) — 1321aa / 146 kDa / 2q31.1 / AR LOF\n"
                "PFIC2 — HCC/CCA children <5yr HIGHEST pediatric; BSEP-null IHC PATHOGNOMONIC; GGT normal.\n\n"
                "BSEP-NULL IHC — PATHOGNOMONIC:\n"
                "  BSEP immunohistochemistry on liver biopsy: absent = PFIC2 PATHOGNOMONIC.\n"
                "  Contrast: PFIC1 (ATP8B1): GGT normal but BSEP present (different IHC pattern).\n"
                "  Contrast: PFIC3 (ABCB4): GGT VERY HIGH; MDR3 absent IHC; BSEP present.\n\n"
                "GGT NORMAL — DDx CLUE:\n"
                "  PFIC1 + PFIC2: GGT characteristically NORMAL (bile salt transport defects).\n"
                "  PFIC3: GGT markedly ELEVATED — key DDx from PFIC1/2.\n\n"
                "HCC/CCA IN CHILDREN:\n"
                "  HCC/CCA onset <5yr in PFIC2 = HIGHEST PEDIATRIC liver cancer risk.\n"
                "  Without liver Tx: HCC median 2-5yr; urgent liver Tx when possible.\n"
                "  Odevixibat / maralixibat (IBAT inhibitors): reduce pruritus; delay Tx."
            ),
        },
        {
            "term": "HFE / HH1 / C282Y-HOMOZYGOTE-HCC-20-200X / PHLEBOTOMY-PREVENTS / MRI-T2STAR",
            "definition": (
                "HFE — 343aa / 37 kDa / 6p21.3 / AR (C282Y/C282Y)\n"
                "HH1 — C282Y/C282Y HCC 20-200x HIGHEST adult iron overload; phlebotomy prevents pre-cirrhosis.\n\n"
                "C282Y HOMOZYGOTE — HIGHEST ADULT IRON-OVERLOAD HCC:\n"
                "  C282Y/C282Y: HCC 20-200x RR — HIGHEST of ALL iron-overload syndromes.\n"
                "  HCC WITHOUT cirrhosis: HFE unique — iron-mediated direct DNA mutagenesis.\n"
                "  Surveillance: 6-monthly USS + AFP in ALL C282Y/C282Y with F3-F4 fibrosis.\n\n"
                "PHLEBOTOMY — PREVENTS IF PRE-CIRRHOSIS:\n"
                "  Phlebotomy to ferritin <50 mcg/L + transferrin sat <50% = PREVENTS HCC if pre-cirrhosis.\n"
                "  Once cirrhosis established: HCC risk persists despite iron depletion.\n\n"
                "MRI T2* LIVER IRON:\n"
                "  MRI T2* (or IDEAL): non-invasive liver iron concentration (LIC) quantification.\n"
                "  LIC >250 mcg/g (>4.5 mg/g) = severe overload; biopsy for fibrosis if ferritin >1000.\n"
                "  Replaces liver biopsy for iron staging in most patients."
            ),
        },
        {
            "term": "ATP7B / WILSON'S DISEASE / KAYSER-FLEISCHER-PATHOGNOMONIC / SLIT-LAMP-MANDATORY / HCC-2-5X",
            "definition": (
                "ATP7B — 1465aa / 160 kDa / 13q14.3 / AR LOF\n"
                "Wilson's Disease — Kayser-Fleischer PATHOGNOMONIC; slit-lamp MANDATORY; HCC 2-5x.\n\n"
                "KAYSER-FLEISCHER RINGS — PATHOGNOMONIC:\n"
                "  Copper in Descemet's membrane peripheral cornea = WD PATHOGNOMONIC.\n"
                "  BUT absent in 50% of hepatic WD presentation — SLIT-LAMP STILL MANDATORY.\n"
                "  KF rings disappear with chelation — treatment response marker.\n"
                "  Slit-lamp by experienced ophthalmologist (not hand-held torch).\n\n"
                "HCC IN WILSON'S:\n"
                "  HCC: 2-5x RR; confined to cirrhotic WD patients.\n"
                "  Adequately treated WD: low HCC risk — early treatment PREVENTS cirrhosis AND HCC.\n\n"
                "TREATMENT HIERARCHY:\n"
                "  Acute: D-penicillamine or trientine (chelation); zinc for maintenance.\n"
                "  Neurological WD: trientine preferred (D-pen can worsen neuro acutely — start LOW).\n"
                "  ALF: liver Tx — D-pen CANNOT chelate fast enough for WD ALF."
            ),
        },
        {
            "term": "SERPINA1 / AATD / Pi*ZZ / PASD-GLOBULES-PATHOGNOMONIC / HCC-5-20X / AUGMENTATION-LUNG",
            "definition": (
                "SERPINA1 — 418aa / 52 kDa / 14q32.13 / AR (Pi*ZZ)\n"
                "AATD — Pi*ZZ; PASD-positive globules PATHOGNOMONIC; HCC 5-20x; augmentation for LUNG not liver.\n\n"
                "PASD GLOBULES — PATHOGNOMONIC:\n"
                "  Periodic acid-Schiff with diastase (PASD) staining: Z-AAT globules resist diastase.\n"
                "  Periportal PASD-positive eosinophilic globules = AATD PATHOGNOMONIC on liver biopsy.\n"
                "  Quantity correlates with disease severity.\n\n"
                "AUGMENTATION = LUNG, NOT LIVER:\n"
                "  IV AAT augmentation (ARALAST, ZEMAIRA): addresses lung emphysema, NOT liver disease.\n"
                "  Z-AAT accumulation in liver ER: augmentation does NOT reduce hepatic Z-AAT load.\n"
                "  Liver Tx: CURATIVE for AATD liver disease AND HCC risk (donor hepatocytes produce normal AAT).\n\n"
                "EMERGING THERAPY:\n"
                "  Fazirsiran (ARO-AAT): RNAi silencing of hepatic Z-AAT synthesis — Phase 3.\n"
                "  Gene therapy (AAV-SERPINA1): liver-directed correction — clinical trials."
            ),
        },
        {
            "term": "APC / FAP / HEPATOBLASTOMA-750-7500X-HIGHEST / LIVER-USS-FROM-BIRTH / CHRPE-PATHOGNOMONIC",
            "definition": (
                "APC — 2843aa / 310 kDa / 5q22.2 / AD LOF\n"
                "FAP/Gardner — hepatoblastoma 750-7500x HIGHEST; liver USS from birth; CHRPE PATHOGNOMONIC.\n\n"
                "HEPATOBLASTOMA — HIGHEST RISK (APC):\n"
                "  APC germline: hepatoblastoma 750-7500x RR = HIGHEST hepatoblastoma gene.\n"
                "  Absolute risk: ~1.5-2% FAP lifetime (0-15yr); peak 18 months.\n"
                "  LIVER USS FROM BIRTH TO AGE 10: mandatory in FAP families — OFTEN MISSED.\n"
                "  AFP: marked elevation with hepatoblastoma; serial AFP monitoring.\n\n"
                "CHRPE — PATHOGNOMONIC:\n"
                "  ≥4 bilateral CHRPE lesions = APC germline PATHOGNOMONIC (present from birth).\n"
                "  CHRPE absent in AFAP — genotype-phenotype: codon 157-1309 (attenuated = 5' end).\n"
                "  5q22.2 deletion in sequencing-negative FAP: MLPA MANDATORY.\n\n"
                "FAP MANAGEMENT:\n"
                "  Prophylactic colectomy: 20-25yr (IRA or IPAA based on rectal polyp burden).\n"
                "  Sulindac/celecoxib: polyp reduction adjunct; does not prevent CRC.\n"
                "  Spigelman duodenal staging: endoscopic surveillance for periampullary cancer."
            ),
        },
        {
            "term": "TSC2 / TSC / HEPATIC-AML-75%-BILATERAL / EVEROLIMUS-FDA-APPROVED / AML-GREATER-3CM-THRESHOLD",
            "definition": (
                "TSC2 (Tuberin) — 1807aa / 200 kDa / 16p13.3 / AD LOF\n"
                "TSC — hepatic AML 75% bilateral; everolimus FDA approved; AML >3cm = intervention threshold.\n\n"
                "HEPATIC ANGIOMYOLIPOMAS (AMLs):\n"
                "  75% of TSC patients; bilateral, multiple; predominantly female; benign (not HCC).\n"
                "  Haemorrhage risk: large AMLs (>3cm), fat-poor AMLs, intralesional aneurysms.\n"
                "  Annual liver MRI: size monitoring — MANDATORY in TSC.\n\n"
                "EVEROLIMUS (mTORi) — FDA APPROVED:\n"
                "  FDA-approved 2012 for hepatic AMLs in TSC; ~50% volume reduction.\n"
                "  ON DISCONTINUATION: AMLs REGROW — maintenance dosing required.\n"
                "  Also FDA-approved: SEGA (brain), renal AML, pulmonary LAM.\n\n"
                "TSC2 vs TSC1:\n"
                "  TSC2: more severe (SEGA, cognitive impairment, renal AML, LAM more common).\n"
                "  TSC2 de novo: ~60-70% (vs TSC1 ~10-15%) — somatic mosaicism in ~15% de novo."
            ),
        },
        {
            "term": "SMAD4 / JPS-HHT / SMAD4-NULL-IHC-PATHOGNOMONIC / HEPATIC-AVMs / BEVACIZUMAB / GASTRIC-HAMARTOMAS",
            "definition": (
                "SMAD4 — 552aa / 60 kDa / 18q21.2 / AD LOF\n"
                "JPS-HHT — SMAD4-null IHC PATHOGNOMONIC; hepatic AVMs (bevacizumab); gastric hamartomas.\n\n"
                "SMAD4-NULL IHC — PATHOGNOMONIC:\n"
                "  SMAD4 IHC on colonic/gastric polyp tissue: null = SMAD4-JPS PATHOGNOMONIC.\n"
                "  Contrast with BMPR1A-JPS: SMAD4 IHC intact — DDx from IHC alone.\n"
                "  SMAD4-null polyp IHC → germline SMAD4 testing MANDATORY.\n\n"
                "JPS-HHT — HEPATIC AVMs:\n"
                "  SMAD4-JPS: combined JPS + HHT phenotype in ~15-20% of JPS families.\n"
                "  Hepatic AVMs → high-output cardiac failure; biliary ischaemia; portal HTN.\n"
                "  Bevacizumab 5mg/kg q2w: anti-VEGF for hepatic HHT — reduces cardiac output.\n"
                "  Liver Tx: definitive for severe hepatic HHT (SMAD4-JPS variant).\n\n"
                "GASTRIC POLYPOSIS (SMAD4-JPS-GASTRIC):\n"
                "  Severe diffuse gastric polyposis → protein-losing enteropathy — SMAD4 specific.\n"
                "  Gastrectomy may be required; annual gastroscopy from diagnosis.\n"
                "  Gastric cancer risk 21% in SMAD4-JPS (much higher than BMPR1A-JPS)."
            ),
        },
        {
            "term": "CASCADE TESTING — Hereditary HCC & Primary Liver Cancer Predisposition",
            "definition": (
                "CASCADE TESTING PRIORITIES for Hereditary HCC & Primary Liver Cancer Predisposition:\n\n"
                "TIER 1 — EXTREME/HIGHEST PEDIATRIC LIVER CANCER RISK:\n"
                "  FAH (HT1): ALL first-degree relatives; succinylacetone NBS DBS or plasma.\n"
                "    → NTBC from birth; MRI liver 3-monthly + AFP from diagnosis; liver Tx if HCC.\n"
                "  ABCB11 (PFIC2): family cascade (AR: sibling risk 25%); BSEP IHC on liver biopsy.\n"
                "    → GGT normal DDx: PFIC1/PFIC2 vs PFIC3; liver Tx if HCC/severe PFIC2.\n"
                "  APC (FAP): ALL first-degree relatives; hepatoblastoma 750-7500x.\n"
                "    → Liver USS from BIRTH to age 10 (MANDATORY — often missed);\n"
                "    → Colonoscopy annually 12-15yr; prophylactic colectomy 20-25yr.\n\n"
                "TIER 2 — HIGH ADULT HCC RISK:\n"
                "  HFE (HH1): all C282Y/C282Y first-degree relatives; transferrin sat + ferritin screen.\n"
                "    → Phlebotomy to ferritin <50 mcg/L — PREVENTS HCC if pre-cirrhosis.\n"
                "    → MRI T2* liver iron; 6-monthly USS + AFP if F3-F4 fibrosis.\n"
                "  SERPINA1 (AATD): Pi phenotyping/genotyping all first-degree relatives of Pi*ZZ.\n"
                "    → Fazirsiran Phase 3 enrollment if eligible; liver Tx if HCC/cirrhosis.\n"
                "  ATP7B (WD): all first-degree relatives; ceruloplasmin + 24h urine Cu.\n"
                "    → D-penicillamine/trientine/zinc; early treatment PREVENTS cirrhosis AND HCC.\n\n"
                "TIER 3 — HAMARTOMAL LIVER RISK:\n"
                "  TSC2: all first-degree relatives; baseline liver MRI; everolimus for AML >3cm.\n"
                "    → NOT HCC risk; AML haemorrhage risk; annual liver MRI.\n"
                "  SMAD4 (JPS-HHT): first-degree relatives; SMAD4-null IHC on polyp tissue.\n"
                "    → Annual colonoscopy+gastroscopy; echocardiography + CT chest (PAVMs).\n\n"
                "PATHOGNOMONIC POINTERS:\n"
                "  Neonatal/infantile liver disease + succinylacetone in urine → FAH HT1 MANDATORY.\n"
                "  Paediatric cholestasis + GGT NORMAL + BSEP-null IHC → PFIC2 (ABCB11).\n"
                "  Adult fatigue + high transferrin saturation + C282Y/C282Y → HFE HH1.\n"
                "  KF rings + hepatic + neuropsychiatric → ATP7B Wilson's (slit-lamp MANDATORY).\n"
                "  PASD-positive periportal globules on liver biopsy → SERPINA1 AATD Pi*ZZ.\n"
                "  Child <10yr + AFP elevation + FAP family → APC hepatoblastoma (USS immediately).\n"
                "  Bilateral hepatic AMLs + epilepsy + skin findings → TSC2 (everolimus if >3cm).\n"
                "  JPS + HHT features (epistaxis, telangiectasias) → SMAD4 JPS-HHT cascade.\n"
            ),
        },
    ]

    return {
        "atlas":       "Hereditary-HCC-Liver-Cancer-Predisposition-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": defs,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("genes_detail",)}, indent=2))
    print("\n=== BREAKDOWN summary ===")
    br = generate_breakdown()
    for row in br["breakdown"]:
        print(f"  {row['gene']:10s} n={row['n']} mean_age={row['mean_age_onset']} "
              f"liver_ca_n={row['liver_ca_n']} ({row['liver_ca_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term'][:80]}")
