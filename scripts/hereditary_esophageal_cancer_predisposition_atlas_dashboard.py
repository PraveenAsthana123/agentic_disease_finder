#!/usr/bin/env python3
"""Hereditary-Esophageal-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
RHBDF2  (iRhom2 / Inactive Rhomboid 2; 315aa; 17q25.1; AD GOF;
         Tylosis with Esophageal Cancer (TOC) — Howel-Evans Syndrome;
         Focal palmoplantar keratoderma PATHOGNOMONIC + esophageal SCC 95% lifetime;
         MOST SPECIFIC hereditary esophageal predisposition gene known;
         seed SEED_BASE+0) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome — esophageal SCC/adenocarcinoma 3-5%;
         AVOID RADIATION ABSOLUTELY — radiation-induced sarcoma risk;
         WB-MRI Toronto Protocol ANNUALLY MANDATORY;
         seed SEED_BASE+1) .
CDH1    (E-cadherin; 882aa; 16q22.1; AD LOF;
         Hereditary Diffuse Gastric Cancer (HDGC) — Barrett's/GEJ adenocarcinoma risk;
         Prophylactic total gastrectomy age 20-30yr MANDATORY in CDH1 carriers;
         Lobular breast cancer 42-56% lifetime (females) — annual MRI from 30yr;
         seed SEED_BASE+2) .
BRCA2   (Breast Cancer Gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC — esophageal SCC + adenocarcinoma 2-3x relative risk;
         HRD — PARP inhibitors (olaparib) active; cisplatin preferred over carboplatin;
         seed SEED_BASE+3) .
ATM     (Ataxia-Telangiectasia Mutated; 3056aa; 11q22.3; AR/AD;
         Esophageal 2-3x (heterozygous carriers); radiation hypersensitivity even heterozygotes;
         Reduce RT dose 20-30% if unavoidable; prefer surgery for esophageal resection;
         seed SEED_BASE+4) .
MLH1    (MutL Homolog 1; 756aa; 3p22.2; AD LOF;
         Lynch syndrome type 1 — esophageal cancer rare 0.4-1% (adenocarcinoma, NOT SCC);
         MSI-H PATHOGNOMONIC; pembrolizumab FDA2017 (MSI-H/dMMR); aspirin CAPP2 50% risk reduction;
         seed SEED_BASE+5) .
MSH2    (MutS Homolog 2; 934aa; 2p21; AD LOF;
         Lynch syndrome type 2 — Muir-Torre sebaceous neoplasms PATHOGNOMONIC;
         Esophageal 0.5%; EPCAM 3-prime deletion — MSH2 promoter methylation; MLPA MANDATORY;
         seed SEED_BASE+6) .
PALB2   (Partner and Localiser of BRCA2 / FANCN; 1186aa; 16p12.2; AD LOF;
         Esophageal cancer 1.8x emerging; HRD — PARP inhibitor sensitivity;
         Breast cancer 53% lifetime (females) — dominant clinical phenotype;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3462-3469)
"""
import random

SEED_BASE = 3462

ATLAS_GENES = [
    {
        "gene": "RHBDF2",
        "protein": (
            "RHBDF2 -- 17q25.1 Autosomal-Dominant-GOF -- 315aa -- "
            "iRhom2-Inactive-Rhomboid-Pseudoprotease-35kDa-"
            "TOC-Tylosis-Howel-Evans-Syndrome-Focal-PPK-PATHOGNOMONIC-"
            "Esophageal-SCC-95pct-Lifetime-MOST-SPECIFIC-Esophageal-Gene-"
            "ADAM17-TACE-Hyperactivation-EGFR-Ligand-Shedding-OMIM-614248"
        ),
        "locus": "17q25.1",
        "protein_size": (
            "315 aa / 35 kDa / 17q25.1 RHBDF2 esophageal cancer molecular context: "
            "STRUCTURE: "
            "  315 aa / 35 kDa; 7-TM inactive rhomboid pseudoprotease — catalytically dead; "
            "  iRhom homology domain (iRhom2 HD): ADAM17/TACE activation via ER export; "
            "  GOF mutations cluster in cytoplasmic N-terminal (aa 45-120) — TOC hotspot region; "
            "  Activates ADAM17 (TACE): hyperactivated metalloprotease sheds EGFR ligands; "
            "  EGF, TGF-α, HB-EGF, amphiregulin: all shed by activated RHBDF2-ADAM17 complex; "
            "MOLECULAR MECHANISM OF CANCER: "
            "  ADAM17 hyperactivation → autocrine EGFR signalling → hyperproliferative esophageal epithelium; "
            "  Focal PPK: same mechanism drives palmoplantar keratinocyte hyperproliferation; "
            "  SCC-specific: squamous esophageal epithelium selectively transformed (not Barrett's/adenocarcinoma); "
            "  Cetuximab rationale: EGFR pathway hyperactivation — anti-EGFR directly targets the mechanism; "
            "KEY MUTATIONS (TOC hotspot): "
            "  p.Ile186Thr (c.557T>C): most common UK families — cytoplasmic iRhom2 HD; "
            "  p.Tyr186Cys (c.557A>G): Irish/Australian families — same codon, different amino acid; "
            "  p.Phe214Ile (c.640T>A): German families; "
            "  p.Pro189Leu (c.566C>T): Dutch/Belgian families; "
            "  p.Val197Phe (c.589G>T): sporadic de novo; "
            "CANCER RISKS: "
            "  ESOPHAGEAL SCC: 95% cumulative lifetime — HIGHEST esophageal predisposition of any known gene; "
            "  ONSET: median esophageal cancer diagnosis 45-55yr (vs sporadic 65-70yr); "
            "  ORAL CAVITY SCC: elevated (2-5x); NOT adenocarcinoma — squamous only; "
            "KEY MANAGEMENT: "
            "  Annual upper GI endoscopy from age 20yr MANDATORY — high-quality chromoendoscopy; "
            "  Narrow-band imaging (NBI): detects early SCC; Lugol iodine spray: unmasks dysplasia; "
            "  Focal PPK onset: age 5-15yr — PATHOGNOMONIC marker; fingernails/toenails also affected; "
            "  Prophylactic esophagectomy: NOT recommended (95% lifetime risk does not mean immediate); "
            "  Cetuximab: on-mechanism EGFR inhibitor — preferred over platinum if systemic needed"
        ),
        "syndrome": "Tylosis with Esophageal Cancer (TOC) / Howel-Evans Syndrome — AD GOF RHBDF2",
        "inheritance": "AD GOF (autosomal dominant gain-of-function)",
        "esophageal_risk": "Esophageal SCC: 95% cumulative lifetime — highest hereditary esophageal risk of any gene",
        "pathognomonic": "Focal palmoplantar keratoderma (PPK) + esophageal SCC family history = TOC PATHOGNOMONIC; fingernail/toenail thickening",
        "key_avoid": "No specific absolute contraindication — cetuximab preferred over platinum (on-mechanism EGFR targeting for RHBDF2-driven SCC)",
        "key_rule": "Annual upper GI endoscopy with Lugol iodine/NBI from age 20yr MANDATORY; focal PPK onset age 5-15yr is the surveillance trigger; cetuximab preferred over platinum for RHBDF2-SCC",
        "surveillance": "Annual OGD with Lugol iodine chromoendoscopy + NBI from age 20yr; annual dermatology (focal PPK monitoring); annual oral exam (SCC risk); chest X-ray annually from age 30yr",
        "targeted_rx": "Cetuximab (EGFR inhibitor — on-mechanism for RHBDF2 GOF); pembrolizumab (anti-PD-1 SCC); nivolumab (CheckMate-648 esophageal SCC); fluorouracil + cisplatin (esophageal SCC standard); paclitaxel/carboplatin (if cisplatin intolerant)",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "Tumour-Suppressor-43kDa-Guardian-of-Genome-"
            "LFS-Esophageal-3-5pct-SCC-Adenocarcinoma-Both-"
            "AVOID-RADIATION-ABSOLUTELY-Radiation-Induced-Sarcoma-"
            "WB-MRI-Toronto-Protocol-ANNUALLY-MANDATORY-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 esophageal cancer molecular context: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; N-terminal transactivation domain (aa 1-67): MDM2 binding; "
            "  DNA-binding domain (aa 102-292): hotspot mutations (R175H, R248W, R273H, G245S); "
            "  Tetramerisation domain (aa 323-356): homo-tetramer (active form); "
            "  Regulatory domain (aa 363-393): acetylation/ubiquitination; "
            "ESOPHAGEAL CANCER IN LFS: "
            "  Esophageal cancer: 3-5% cumulative risk — SCC and adenocarcinoma both elevated; "
            "  LFS-esophageal: onset median age 30yr vs sporadic 65-70yr; "
            "  TP53 somatic mutation: most common driver in sporadic esophageal SCC (50-80%); "
            "  Barrett's: TP53 LOF appears in Barrett's → adenocarcinoma progression; "
            "RADIATION RULE: "
            "  AVOID RADIATION ABSOLUTELY for curative intent — radiation-induced sarcoma 30% in LFS; "
            "  Esophageal RT (CROSS/FLOT-based chemoradiotherapy): substitute surgery + chemotherapy; "
            "  If RT unavoidable (metastatic/palliative): informed consent re secondary sarcoma; "
            "KEY MANAGEMENT: "
            "  WB-MRI Toronto Protocol ANNUALLY from birth (LFS surveillance standard); "
            "  Annual upper GI endoscopy from age 20yr; "
            "  Annual breast MRI women 20-65yr; "
            "  Eprenetapopt/APR-246 (reactivates mutant p53): esophageal SCC investigational trial"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) — esophageal SCC + adenocarcinoma 3-5%",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "esophageal_risk": "Esophageal SCC + adenocarcinoma: 3-5% cumulative lifetime; onset age 30yr (vs sporadic 65yr)",
        "pathognomonic": "TP53 germline variant + early-onset esophageal cancer + LFS family pedigree (sarcoma/breast/brain/adrenal)",
        "key_avoid": "RADIATION ABSOLUTELY CI for curative esophageal treatment — omit chemoradiotherapy (CROSS protocol CI in LFS); use surgery + cisplatin/5-FU chemotherapy only",
        "key_rule": "WB-MRI Toronto Protocol ANNUALLY from birth MANDATORY; avoid esophageal RT (CROSS protocol); annual OGD from age 20yr; cetuximab or immunotherapy preferred for EGFR+ / MSI-H subtypes",
        "surveillance": "Annual WB-MRI (whole-body MRI); annual OGD from age 20yr; annual breast MRI women 20-65yr; annual abdominal USS; annual dermatology; annual colonoscopy from age 25yr",
        "targeted_rx": "APR-246/eprenetapopt (reactivates R175H/R248W — phase II esophageal); Pembrolizumab (tumour-agnostic MSI-H/TMB-H); Nivolumab (CheckMate-648 esophageal SCC); Cetuximab (LFS SCC — no RT); Cisplatin/5-FU (without RT)",
    },
    {
        "gene": "CDH1",
        "protein": (
            "CDH1 -- 16q22.1 Autosomal-Dominant-LOF -- 882aa -- "
            "E-Cadherin-97kDa-Epithelial-Adhesion-Molecule-"
            "HDGC-Hereditary-Diffuse-Gastric-Cancer-Diffuse-Type-Gastric-70-83pct-"
            "Barrett-Esophagus-GEJ-Adenocarcinoma-2-3x-Risk-"
            "Prophylactic-Total-Gastrectomy-20-30yr-MANDATORY-"
            "Lobular-Breast-42-56pct-Females-OMIM-192090"
        ),
        "locus": "16q22.1",
        "protein_size": (
            "882 aa / 97 kDa / 16q22.1 CDH1 esophageal/GEJ cancer molecular context: "
            "STRUCTURE: "
            "  882 aa / 97 kDa (precursor, signal peptide cleaved); "
            "  5 extracellular cadherin (EC) repeat domains: homophilic adhesion; "
            "  EC3 domain calcium-binding: Ca2+ stabilises EC rigid conformation; "
            "  Transmembrane domain + cytoplasmic tail (aa 780-882): β-catenin binding; "
            "  β-catenin/p120-catenin interactions: WNT pathway crosstalk; "
            "  LOF: E-cadherin loss → epithelial-mesenchymal transition (EMT); "
            "ESOPHAGEAL/GEJ RISK IN HDGC: "
            "  Barrett's esophagus: 2-3x elevated in CDH1 carriers (GEJ / cardia adenocarcinoma); "
            "  GEJ (gastroesophageal junction) adenocarcinoma: falls within HDGC spectrum; "
            "  Mechanism: E-cadherin loss at GEJ promotes columnar metaplasia (Barrett's); "
            "  Esophageal SCC: NOT elevated — CDH1 risk is adenocarcinoma-specific; "
            "PRIMARY RISKS: "
            "  Diffuse gastric cancer: males 67-70% lifetime; females 56-83% lifetime; "
            "  Lobular breast cancer: females 42-56% lifetime — second dominant phenotype; "
            "  Cleft lip/palate: 15-20% penetrance (craniofacial development role); "
            "KEY MANAGEMENT: "
            "  Prophylactic total gastrectomy: recommended age 20-30yr MANDATORY once mutation confirmed; "
            "  Pre-gastrectomy OGD with random biopsies (Cambridge protocol): 56 biopsies; "
            "  OGD CANNOT exclude diffuse gastric cancer reliably — gastrectomy still recommended; "
            "  Annual breast MRI from age 30yr (females); "
            "  Annual OGD with Barrett's surveillance protocol from age 25yr until gastrectomy"
        ),
        "syndrome": "Hereditary Diffuse Gastric Cancer (HDGC) — CDH1 germline LOF",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "esophageal_risk": "Barrett's/GEJ adenocarcinoma: 2-3x; primary risk is diffuse gastric cancer 67-83% and lobular breast 42-56% (females)",
        "pathognomonic": "Diffuse (signet ring cell) gastric cancer + lobular breast cancer family pedigree = HDGC/CDH1 PATHOGNOMONIC; cleft lip/palate association",
        "key_avoid": "No specific drug contraindication; avoid delaying prophylactic gastrectomy beyond age 30yr — OGD alone insufficient to exclude gastric cancer; do NOT reassure on negative OGD",
        "key_rule": "Prophylactic total gastrectomy age 20-30yr MANDATORY; Cambridge 56-biopsy protocol pre-gastrectomy; annual breast MRI females from age 30yr; annual OGD with Barrett's surveillance until gastrectomy",
        "surveillance": "Annual OGD (Cambridge protocol 56 biopsies) until gastrectomy age 20-30yr; annual breast MRI women from age 30yr; annual colonoscopy (colorectal risk minor elevation); annual dermatology (melanoma minor elevation)",
        "targeted_rx": "Trastuzumab (if HER2+ gastric/GEJ — ToGA data); Pembrolizumab (MSI-H or PD-L1 CPS≥10); Ramucirumab (anti-VEGFR2 gastric 2nd line); Nivolumab (CheckMate-649 gastric); Larotrectinib (if NTRK fusion)",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "FANCD1-HR-Mediator-384kDa-"
            "HBOC2-Esophageal-2-3x-Squamous-Adenocarcinoma-Both-"
            "HRD-Cisplatin-Preferred-Over-Carboplatin-"
            "Olaparib-FDA2020-SOLO2-PROfound-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 esophageal cancer molecular context: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; 8 BRC repeats (aa 1002-2085): RAD51 binding (8 copies); "
            "  DNA binding domain (aa 2396-3190): ssDNA/dsDNA binding; "
            "  OB-folds (aa 2396-3190): RPA displacement from ssDNA; "
            "  C-terminal NLS + BRCA1-binding domain; "
            "  FANCD1 = biallelic BRCA2 LOF — Fanconi anaemia type D1; "
            "ESOPHAGEAL CANCER IN BRCA2 CARRIERS: "
            "  Esophageal cancer: 2-3x relative risk — BOTH SCC and adenocarcinoma elevated; "
            "  Adenocarcinoma: 2-3x (HRD promotes chromosomal instability in Barrett's → adenocarcinoma); "
            "  SCC: 2x (oxidative damage repair impairment in squamous epithelium); "
            "  ONSET: median BRCA2-esophageal age 52yr (vs sporadic 65yr); "
            "  Primary risks dominate: breast cancer females 69-72%; ovarian cancer 17%; "
            "  Male breast cancer 6%; pancreatic cancer 3-5x; prostate 3-8x; "
            "HRD THERAPEUTIC IMPLICATIONS: "
            "  Cisplatin preferred over carboplatin (creates ICL — BRCA2 null cells cannot repair); "
            "  PARP inhibitors: olaparib (FDA2020 — BRCA-mutant solid tumours PROfound/OlympiAD); "
            "  Rucaparib, niraparib: also active in BRCA2 tumours; "
            "KEY MANAGEMENT: "
            "  Annual OGD from age 50yr with Barrett's protocol; "
            "  Annual breast MRI from age 25yr (females); "
            "  Annual prostate PSA from age 40yr (males); "
            "  Pancreatic MRI/EUS annually from age 50yr"
        ),
        "syndrome": "HBOC syndrome (Hereditary Breast and Ovarian Cancer) type 2 — BRCA2/FANCD1",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); biallelic = Fanconi Anaemia type D1",
        "esophageal_risk": "Esophageal SCC + adenocarcinoma: 2-3x relative risk; BOTH histological subtypes elevated",
        "pathognomonic": "Breast + ovarian + pancreatic + esophageal cancer family pedigree = BRCA2 HBOC phenotype; biallelic = FA-D1 with Wilms/medulloblastoma in childhood",
        "key_avoid": "CARBOPLATIN: cisplatin PREFERRED over carboplatin for BRCA2-esophageal (superior ICL-forming activity in HRD context)",
        "key_rule": "Cisplatin preferred over carboplatin for BRCA2-esophageal treatment; PARP inhibitors (olaparib) for maintenance after platinum response; annual OGD from age 50yr; annual breast MRI females from age 25yr",
        "surveillance": "Annual breast MRI from age 25yr (females); annual OGD from age 50yr (Barrett's protocol); annual prostate PSA from age 40yr (males); pancreatic MRI/EUS from age 50yr; annual colonoscopy from age 50yr",
        "targeted_rx": "Olaparib (PARP inhibitor — FDA2020 BRCA-mutant solid tumours); Rucaparib (BRCA-mutant); Niraparib (BRCA-mutant); Cisplatin/5-FU (HRD — preferred platinum); Pembrolizumab (MSI-H or TMB-H); Nivolumab (esophageal SCC/adenocarcinoma)",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AR-Biallelic=AT / AD-Heterozygous=2-3xEsophageal -- 3056aa -- "
            "PI3K-Like-Kinase-350kDa-DSB-Master-Sensor-"
            "AT-Biallelic-Cerebellar-Ataxia-Telangiectasia-IgA-Deficiency-PATHOGNOMONIC-"
            "RADIATION-HYPERSENSITIVITY-Even-Heterozygotes-Reduce-RT-20-30pct-"
            "Esophageal-2-3x-Olaparib-Emerging-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM esophageal cancer molecular context: "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; FAT domain (aa 1981-2566): protein-protein interactions; "
            "  Kinase domain (aa 2712-2962): PI3K-like — phosphorylates H2AX (γH2AX), CHEK2, BRCA1; "
            "  FATC domain (aa 3024-3056): essential for kinase activity; "
            "  MRN complex (MRE11-RAD50-NBN) recruits ATM to DSBs; "
            "ESOPHAGEAL CANCER RISK: "
            "  Heterozygous carriers: esophageal cancer 2-3x RR — BOTH SCC and adenocarcinoma elevated; "
            "  Biallelic (A-T): esophageal cancer elevated (dominant risks are lymphoma/leukaemia 35-40%); "
            "  IgA deficiency in A-T: esophageal/pharyngeal mucosal vulnerability; "
            "  ATM somatic mutation: frequent in sporadic Barrett's → adenocarcinoma progression; "
            "RADIATION HYPERSENSITIVITY (KEY CLINICAL RULE): "
            "  Even heterozygotes: G2/M checkpoint impairment — measurable radiation sensitivity; "
            "  ESOPHAGEAL chemoradiotherapy (CROSS, FLOT-RT): reduce radiation dose 20-30%; "
            "  Preferably avoid radical RT in ATM carriers; surgery-first for esophageal resection; "
            "  Biallelic A-T: esophageal RT equivalent to ABSOLUTELY CI; "
            "KEY MANAGEMENT: "
            "  Annual OGD with Barrett's surveillance from age 40yr; "
            "  Reduce esophageal RT dose 20-30% if unavoidable; prefer surgery; "
            "  Immunoglobulin replacement in biallelic A-T (IgA-IgG deficiency)"
        ),
        "syndrome": "Ataxia-Telangiectasia (biallelic) / ATM carrier syndrome — esophageal 2-3x elevated",
        "inheritance": "AR biallelic = A-T; AD heterozygous = esophageal/breast/prostate risk",
        "esophageal_risk": "Heterozygous: esophageal 2-3x; biallelic: elevated (lymphoma dominant), IgA deficiency affects GI mucosal surveillance",
        "pathognomonic": "Cerebellar ataxia + telangiectasia + IgA deficiency = A-T biallelic PATHOGNOMONIC; esophageal risk elevated in heterozygous carriers",
        "key_avoid": "RADIATION: reduce esophageal RT dose 20-30% even in heterozygotes; avoid radical chemoradiotherapy (CROSS protocol) in ATM carriers — use surgery-first approach",
        "key_rule": "Radiation sensitivity applies to HETEROZYGOUS ATM carriers for esophageal RT — reduce dose 20-30%; annual OGD from age 40yr; immunoglobulin levels mandatory pre-surgery biallelic",
        "surveillance": "Annual OGD from age 40yr (Barrett's + SCC protocol); immunoglobulin levels annually biallelic; neurological assessment annually biallelic; lymphoma surveillance biallelic (annual CT/PET)",
        "targeted_rx": "Olaparib (ATM-mutant tumours — PROfound/JAVELIN Lung data, emerging esophageal); Pembrolizumab (esophageal SCC/adenocarcinoma); Nivolumab (esophageal); Cetuximab; Immunoglobulin replacement (biallelic A-T)",
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MutL-Alpha-MMR-85kDa-Lynch-Type-1-"
            "Esophageal-0.4-1pct-ADENOCARCINOMA-NOT-SCC-"
            "MSI-H-PATHOGNOMONIC-Pembrolizumab-FDA2017-"
            "Constitutional-Methylation-NOT-Inherited-EPIMUTATION-"
            "CAPP2-Aspirin-600mg-50pct-CRC-Risk-Reduction-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 esophageal cancer molecular context: "
            "STRUCTURE: "
            "  756 aa / 85 kDa; N-terminal ATPase domain (aa 1-340): ATP hydrolysis drives mismatch excision; "
            "  Dimerisation domain (aa 492-756): PMS2 binding — forms MutLα heterodimer; "
            "  MutLα (MLH1-PMS2): latent endonuclease — strand discrimination and nick introduction; "
            "  PCNA interaction via C-terminal PIP box; "
            "ESOPHAGEAL CANCER IN LYNCH SYNDROME: "
            "  Esophageal cancer: 0.4-1% cumulative lifetime — RARE Lynch manifestation; "
            "  ADENOCARCINOMA ONLY: Lynch/MLH1-esophageal is adenocarcinoma (NOT SCC); "
            "  Mechanism: MSI-H adenocarcinoma in Barrett's or GEJ; "
            "  ONSET: Lynch-esophageal median age 57yr; typical Lynch tumor spectrum: CRC, endometrial, ovarian dominant; "
            "MSI-H DIAGNOSTIC AND THERAPEUTIC: "
            "  MSI-H PATHOGNOMONIC for Lynch in esophageal adenocarcinoma; "
            "  PEMBROLIZUMAB FDA2017: first tumour-agnostic approval for MSI-H/dMMR solid tumours; "
            "  Response rate MSI-H esophageal: 40-57% (Keynote-158); "
            "CONSTITUTIONAL METHYLATION (EPIMUTATION): "
            "  5% of Lynch-MLH1: MLH1 promoter constitutional methylation — NOT heritable in Mendelian sense; "
            "  Sporadic silencing event — family members have different risk; "
            "  MLPA or MS-MLPA mandatory to distinguish germline variant vs epimutation; "
            "KEY MANAGEMENT: "
            "  Annual upper GI endoscopy with Barrett's protocol from age 45yr (Lynch specific); "
            "  Aspirin 600mg daily (CAPP2): 50% CRC risk reduction — likely esophageal benefit too; "
            "  Annual colonoscopy from age 25yr (dominant Lynch cancer)"
        ),
        "syndrome": "Lynch Syndrome type 1 (MLH1 — CRC + endometrial dominant; esophageal rare adenocarcinoma)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); constitutional epimutation variant in 5%",
        "esophageal_risk": "Esophageal adenocarcinoma: 0.4-1% cumulative lifetime; RARE Lynch manifestation — adenocarcinoma not SCC; MSI-H PATHOGNOMONIC",
        "pathognomonic": "MSI-H esophageal adenocarcinoma in Lynch family = MLH1 germline or constitutional methylation; Amsterdam II criteria CRC pedigree",
        "key_avoid": "No specific drug contraindication for esophageal; MLPA essential to distinguish germline LOF from constitutional methylation (management differs — methylation not fully heritable)",
        "key_rule": "MLPA mandatory to distinguish germline LOF vs constitutional methylation; pembrolizumab for MSI-H/dMMR esophageal adenocarcinoma; aspirin 600mg CAPP2 protocol; annual OGD from age 45yr",
        "surveillance": "Annual colonoscopy from age 25yr; annual OGD with Barrett's surveillance from age 45yr; annual gynaecological surveillance women 25-65yr; annual urinalysis (urothelial Lynch risk 10-14%); annual dermatology (sebaceous — Muir-Torre if MLH1+MSH2 overlap)",
        "targeted_rx": "Pembrolizumab (FDA2017 MSI-H/dMMR — KEYNOTE-158 40-57% ORR esophageal); Dostarlimab (FDA2021 MSI-H); Nivolumab (nivolumab + ipilimumab: CheckMate-142 MSI-H); Aspirin 600mg CAPP2 (chemoprevention); FOLFOX/CAPOX (Lynch-CRC and esophageal)",
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutS-Alpha-MMR-MutS-Homolog-104kDa-Lynch-Type-2-"
            "Muir-Torre-Sebaceous-Neoplasms-PATHOGNOMONIC-"
            "Esophageal-0.5pct-Squamous-Adenocarcinoma-"
            "EPCAM-3prime-Deletion-MSH2-Methylation-MLPA-MANDATORY-"
            "Urothelial-14pct-HIGHEST-Lynch-Gene-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 104 kDa / 2p21 MSH2 esophageal cancer molecular context: "
            "STRUCTURE: "
            "  934 aa / 104 kDa; mismatch-binding domain (aa 1-127): mismatch recognition; "
            "  Connector domain (aa 127-295): conformational change; "
            "  Lever domain (aa 295-432): ATPase coupling; "
            "  Clamp domain (aa 432-509): DNA binding; "
            "  ATPase domain (aa 509-934): ATP hydrolysis drives MMR; "
            "  MSH2 dimerises with MSH6 (MutSα — single/dinucleotide mismatches) or "
            "  MSH3 (MutSβ — small insertions/deletions); "
            "EPCAM-MSH2 EPIGENETIC SILENCING: "
            "  EPCAM 3-prime deletion → read-through transcription → MSH2 promoter methylation; "
            "  Affects MSH2 expression WITHOUT MSH2 point mutation — MLPA MANDATORY; "
            "  EPCAM deletion: urothelial cancer dominant phenotype; "
            "MUIR-TORRE SYNDROME: "
            "  MSH2 (and MLH1) LOF: sebaceous adenoma, sebaceous epithelioma, sebaceoma, sebaceous carcinoma; "
            "  Keratoacanthoma: Lynch-spectrum variant; "
            "  Sebaceous neoplasm = PATHOGNOMONIC marker for Lynch/Muir-Torre; "
            "ESOPHAGEAL CANCER: "
            "  Esophageal cancer: 0.5% cumulative — BOTH SCC and adenocarcinoma slightly elevated; "
            "  Lynch-esophageal MSH2: MSI-H adenocarcinoma more common than SCC in Lynch; "
            "  Dominant Lynch-MSH2 risks: CRC (50-80%), urothelial (10-14% HIGHEST of any Lynch gene), "
            "    endometrial (25-60%); esophageal very rare; "
            "KEY MANAGEMENT: "
            "  MLPA MANDATORY: test for EPCAM 3-prime deletion if no MSH2 point mutation found; "
            "  Annual cystoscopy + urine cytology from age 25yr (urothelial Lynch 10-14%); "
            "  Annual colonoscopy from age 25yr; "
            "  Annual OGD with Barrett's surveillance from age 45yr"
        ),
        "syndrome": "Lynch Syndrome type 2 + Muir-Torre Syndrome — MSH2 germline LOF; EPCAM 3-prime deletion variant",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); EPCAM 3-prime deletion → epigenetic MSH2 silencing",
        "esophageal_risk": "Esophageal SCC + adenocarcinoma: 0.5% cumulative — RARE Lynch manifestation; MSI-H predominant in adenocarcinoma type",
        "pathognomonic": "Sebaceous neoplasms (sebaceous adenoma/carcinoma/keratoacanthoma) = Muir-Torre/MSH2 PATHOGNOMONIC; urothelial cancer 10-14% HIGHEST among Lynch genes",
        "key_avoid": "EPCAM deletion must be EXCLUDED by MLPA before concluding no MSH2 pathogenic variant — failure to MLPA = diagnostic miss",
        "key_rule": "MLPA MANDATORY to detect EPCAM 3-prime deletion silencing MSH2; annual urine cytology/cystoscopy from age 25yr (urothelial 10-14%); sebaceous neoplasm biopsy triggers Lynch evaluation; pembrolizumab for MSI-H esophageal",
        "surveillance": "Annual colonoscopy from age 25yr; annual urine cytology + cystoscopy from age 25yr; annual OGD from age 45yr; annual gynaecological surveillance women 25-65yr; annual dermatology (Muir-Torre sebaceous surveillance)",
        "targeted_rx": "Pembrolizumab (MSI-H/dMMR FDA2017); Dostarlimab (MSI-H FDA2021); Nivolumab + ipilimumab (MSI-H); Aspirin 600mg CAPP2 (chemoprevention); FOLFOX/CAPOX (Lynch-spectrum cancers)",
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "FANCN-BRCA1-BRCA2-Bridge-131kDa-"
            "Esophageal-1.8x-Emerging-Evidence-HRD-"
            "Breast-Cancer-53pct-Females-Dominant-Phenotype-"
            "TBCRC048-Olaparib-82pct-ORR-Germline-PALB2-Breast-"
            "PARP-Inhibitor-Sensitivity-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 esophageal cancer molecular context: "
            "STRUCTURE: "
            "  1186 aa / 131 kDa; N-terminal coiled-coil (aa 1-100): BRCA1 binding (direct interaction); "
            "  Central domain (aa 200-900): self-oligomerisation; chromatin association; "
            "  WD40 repeat domain (C-terminal, aa 853-1186): BRCA2 binding — PALB2 bridges BRCA1 and BRCA2; "
            "  FANCN = biallelic PALB2 LOF — Fanconi anaemia type N; "
            "  Function: tethers BRCA2 to BRCA1 at DSB sites (PALB2 = Partner and Localiser of BRCA2); "
            "ESOPHAGEAL CANCER RISK (EMERGING EVIDENCE): "
            "  Esophageal cancer: 1.8x relative risk — modest elevation, emerging evidence base; "
            "  MECHANISM: HRD (homologous recombination deficiency) from PALB2 LOF — promotes chromosomal instability; "
            "  Predominantly adenocarcinoma (HRD → Barrett's → adenocarcinoma pathway); "
            "  Larger cohort data from Consortium (PALB2 Interest Group): esophageal 1.5-2.0x pooled; "
            "DOMINANT CLINICAL PHENOTYPE: "
            "  Breast cancer: 53% lifetime risk (females) — moderate-high risk approaching BRCA2 level; "
            "  Ovarian cancer: 3-5% (lower than BRCA1/2); "
            "  Pancreatic cancer: 2-6x; "
            "HRD THERAPEUTIC IMPLICATIONS: "
            "  PARP inhibitors: olaparib active in PALB2-mutant tumours — TBCRC048 82% ORR in PALB2-breast; "
            "  Cisplatin sensitivity via ICL formation similar to BRCA2 context; "
            "  PARP inhibitor maintenance: emerging data in PALB2-esophageal after platinum response; "
            "KEY MANAGEMENT: "
            "  Annual breast MRI from age 30yr (females) — dominant phenotype management; "
            "  Annual OGD from age 50yr with Barrett's surveillance; "
            "  Annual pancreatic MRI/EUS from age 50yr; "
            "  PARP inhibitor consideration for PALB2-esophageal after platinum response"
        ),
        "syndrome": "PALB2/FANCN hereditary cancer syndrome — breast cancer dominant; esophageal 1.8x emerging evidence",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); biallelic = Fanconi Anaemia type N (FA-N)",
        "esophageal_risk": "Esophageal adenocarcinoma: 1.8x relative risk (emerging evidence); HRD promotes Barrett's → adenocarcinoma pathway",
        "pathognomonic": "Breast cancer <50yr + pancreatic cancer family history + PALB2 germline LOF = PALB2 syndrome; biallelic = FA-N (FA core overlap)",
        "key_avoid": "No absolute contraindication for esophageal treatment; prefer cisplatin over carboplatin in HRD context (PALB2 shares BRCA2 mechanistic pathway)",
        "key_rule": "PARP inhibitor (olaparib) maintenance consideration after platinum response in PALB2-esophageal; annual breast MRI from age 30yr females (dominant phenotype); annual OGD from age 50yr",
        "surveillance": "Annual breast MRI from age 30yr (females); annual OGD with Barrett's surveillance from age 50yr; annual pancreatic MRI/EUS from age 50yr; annual ovarian USS + CA-125 from age 35yr (females)",
        "targeted_rx": "Olaparib (PARP inhibitor — TBCRC048 82% ORR PALB2-breast; esophageal emerging); Rucaparib (BRCA/PALB2-mutant); Cisplatin (HRD — platinum preferred); Pembrolizumab (MSI-H or TMB-H esophageal); Nivolumab (esophageal SCC/adenocarcinoma)",
    },
]

# ── Variants (5 per gene, seed-based) ────────────────────────────────────────
VARIANTS = {
    "RHBDF2": [
        ("c.557T>C", "p.Ile186Thr", "exon 6", "cytoplasmic iRhom2 HD — UK families most common", 0.35),
        ("c.557A>G", "p.Tyr186Cys", "exon 6", "same codon — Irish/Australian families", 0.22),
        ("c.640T>A", "p.Phe214Ile", "exon 6", "iRhom2 HD cytoplasmic — German families", 0.18),
        ("c.566C>T", "p.Pro189Leu", "exon 6", "Dutch/Belgian families GOF", 0.14),
        ("c.589G>T", "p.Val197Phe", "exon 6", "sporadic de novo — ADAM17 activation surface", 0.11),
    ],
    "TP53": [
        ("c.817C>T", "p.Arg273Cys", "exon 8", "DNA contact hotspot — dominant negative", 0.30),
        ("c.524G>A", "p.Arg175His", "exon 5", "structural hotspot — partial GOF", 0.25),
        ("c.742C>T", "p.Arg248Trp", "exon 7", "DNA contact hotspot — dominant negative", 0.20),
        ("c.733G>A", "p.Gly245Ser", "exon 7", "structural hotspot — partial GOF", 0.15),
        ("c.1010G>T", "p.Arg337Leu", "exon 9", "tetramerisation domain — Brazilian founder", 0.10),
    ],
    "CDH1": [
        ("c.1901C>T", "p.Thr634Ile", "exon 13", "EC4 calcium-binding — adhesion loss", 0.28),
        ("c.1137G>A", "p.Trp379*", "exon 8", "EC2 truncation — null allele severe", 0.24),
        ("c.2509C>T", "p.Arg837*", "exon 15", "cytoplasmic tail truncation — β-catenin binding lost", 0.20),
        ("c.1565+1G>A", "p.?", "IVS12", "splice donor — EC5 skip, adhesion abolished", 0.16),
        ("c.832G>A", "p.Asp278Asn", "exon 6", "EC2-EC3 hinge — Ca2+ coordination disrupted", 0.12),
    ],
    "BRCA2": [
        ("c.5946delT", "p.Ser1982fs", "exon 11", "BRC repeat 5 frameshift — null allele", 0.30),
        ("c.8488-1G>C", "p.?", "IVS20", "splice acceptor — exon 21 skip, OB-fold disrupted", 0.24),
        ("c.3847_3848del", "p.Asn1283Lysfs", "exon 11", "Icelandic founder 999del5 region", 0.20),
        ("c.9976A>T", "p.Lys3326*", "exon 27", "C-terminal truncation — NLS loss (mild)", 0.14),
        ("c.6174delT", "p.Ser2058fs", "exon 11", "Ashkenazi founder frameshift — null", 0.12),
    ],
    "ATM": [
        ("c.7271T>G", "p.Val2424Gly", "exon 50", "kinase domain PI3K-like — common panel", 0.28),
        ("c.1066-6T>G", "p.?", "IVS10", "splice — partial exon 11 skip, kinase impaired", 0.22),
        ("c.2572T>C", "p.Ser858Pro", "exon 18", "MRN interaction surface", 0.20),
        ("c.8147T>C", "p.Phe2716Ser", "exon 55", "FATC domain — activation surface", 0.16),
        ("c.3161C>T", "p.Thr1054Met", "exon 22", "FAT domain — BRCA1 interaction", 0.14),
    ],
    "MLH1": [
        ("c.1852_1853delinsGC", "p.Lys618Ala", "exon 17", "PMS2 binding domain disrupted — MutLα loss", 0.30),
        ("c.454-2A>G", "p.?", "IVS4", "splice acceptor — exon 5 skip — ATPase impaired", 0.24),
        ("c.199G>A", "p.Val67Met", "exon 2", "ATPase N-terminal — common European founder", 0.20),
        ("c.943C>T", "p.Arg315*", "exon 10", "connector domain nonsense — null", 0.15),
        ("c.306+5G>A", "p.?", "IVS3", "splice — partial exon 4 inclusion — hypomorphic", 0.11),
    ],
    "MSH2": [
        ("c.943+3A>T", "p.?", "IVS5", "splice — partial exon 5 skip — mismatch binding impaired", 0.30),
        ("c.1906G>C", "p.Ala636Pro", "exon 12", "lever domain — conformational change blocked", 0.24),
        ("c.211+1G>A", "p.?", "IVS2", "splice donor — exon 3 skip — ATPase impaired", 0.20),
        ("c.2635-2A>G", "p.?", "IVS15", "splice acceptor — exon 16 skip — dimerisation affected", 0.15),
        ("c.388_389delCA", "p.His130Thrfs", "exon 3", "mismatch-binding domain frameshift — null", 0.11),
    ],
    "PALB2": [
        ("c.3113G>A", "p.Trp1038*", "exon 10", "WD40 repeat — BRCA2 binding lost", 0.32),
        ("c.1592delT", "p.Leu531fs", "exon 4", "central domain frameshift — null", 0.25),
        ("c.172_175delTTGT", "p.Leu58fs", "exon 2", "coiled-coil BRCA1 binding — null", 0.20),
        ("c.2816T>G", "p.Leu939Arg", "exon 9", "WD40 core — BRCA2 interface disruption", 0.14),
        ("c.1240C>T", "p.Arg414*", "exon 4", "central domain nonsense — early truncation", 0.09),
    ],
}

# ── Phenotype generators ──────────────────────────────────────────────────────
ESO_SITES    = ["lower esophagus / GEJ", "mid esophagus", "upper esophagus / hypopharynx", "GEJ / gastric cardia"]
STAGE_OPT    = ["I", "II", "III", "IVA", "IVB"]
HIST_OPT     = ["SCC", "adenocarcinoma", "adenosquamous", "undifferentiated"]
STATUS_OPTS  = ["active surveillance", "esophageal cancer treated", "Barrett's surveillance", "prophylactic gastrectomy done", "remission"]


def _patients_for_gene(gene_dict: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    syndrome = gene_dict["syndrome"]
    variants = VARIANTS[gene]

    gene_ages = {
        "RHBDF2": (35, 65), "TP53": (22, 58), "CDH1": (30, 65),
        "BRCA2":  (42, 72), "ATM":  (40, 70), "MLH1": (45, 72),
        "MSH2":   (40, 70), "PALB2": (38, 68),
    }
    age_lo, age_hi = gene_ages.get(gene, (30, 70))

    # Histology probabilities per gene
    hist_weights = {
        "RHBDF2": [0.90, 0.05, 0.03, 0.02],  # SCC dominant
        "TP53":   [0.50, 0.40, 0.07, 0.03],
        "CDH1":   [0.10, 0.82, 0.05, 0.03],  # adenocarcinoma dominant (GEJ/diffuse)
        "BRCA2":  [0.40, 0.50, 0.07, 0.03],
        "ATM":    [0.45, 0.48, 0.05, 0.02],
        "MLH1":   [0.10, 0.85, 0.03, 0.02],  # adenocarcinoma (MSI-H)
        "MSH2":   [0.30, 0.62, 0.05, 0.03],
        "PALB2":  [0.25, 0.65, 0.07, 0.03],
    }

    cancer_prob = {
        "RHBDF2": 0.55,  # 95% lifetime — high proportion already affected in registry
        "TP53":   0.32,
        "CDH1":   0.40,  # gastric+esophageal combined
        "BRCA2":  0.28,
        "ATM":    0.25,
        "MLH1":   0.22,
        "MSH2":   0.22,
        "PALB2":  0.20,
    }

    patients = []
    for i in range(n):
        age = rng.randint(age_lo, age_hi)
        sex = rng.choice(["M", "F"])
        var = rng.choices(variants, weights=[v[4] for v in variants])[0]

        has_cancer = rng.random() < cancer_prob.get(gene, 0.25)
        cancer_site = rng.choice(ESO_SITES) if has_cancer else None
        cancer_stage = rng.choice(STAGE_OPT) if has_cancer else None
        histology = rng.choices(HIST_OPT, weights=hist_weights.get(gene, [0.50, 0.40, 0.07, 0.03]))[0] if has_cancer else None

        barrett  = rng.random() < 0.30 if gene in {"CDH1", "BRCA2", "PALB2", "ATM"} else rng.random() < 0.15
        ppk      = rng.random() < 0.85 if gene == "RHBDF2" else False
        msi_h    = rng.random() < 0.75 if gene in {"MLH1", "MSH2"} and has_cancer else False

        patients.append({
            "patient_id":    f"{gene}-{seed}-{i+1:03d}",
            "gene":          gene,
            "syndrome":      syndrome,
            "age":           age,
            "sex":           sex,
            "variant":       var[0],
            "effect":        var[1],
            "exon":          var[2],
            "domain":        var[3],
            "has_cancer":    has_cancer,
            "cancer_site":   cancer_site,
            "cancer_stage":  cancer_stage,
            "histology":     histology,
            "has_barrett":   barrett,
            "has_ppk":       ppk,
            "msi_h":         msi_h,
            "status":        rng.choice(STATUS_OPTS),
        })
    return patients


def _all_patients() -> list:
    all_pts = []
    for gene_dict in ATLAS_GENES:
        idx = [g["gene"] for g in ATLAS_GENES].index(gene_dict["gene"])
        all_pts.extend(_patients_for_gene(gene_dict, SEED_BASE + idx))
    return all_pts


# ── API generators ────────────────────────────────────────────────────────────
def generate_overview() -> dict:
    pts = _all_patients()
    gene_counts = {}
    for p in pts:
        gene_counts.setdefault(p["gene"], {"gene": p["gene"], "n": 0, "cancer": 0, "barrett": 0, "msi_h": 0})
        gene_counts[p["gene"]]["n"] += 1
        if p["has_cancer"]:  gene_counts[p["gene"]]["cancer"] += 1
        if p["has_barrett"]: gene_counts[p["gene"]]["barrett"] += 1
        if p["msi_h"]:       gene_counts[p["gene"]]["msi_h"] += 1

    cancer_total = sum(1 for p in pts if p["has_cancer"])
    barrett_total = sum(1 for p in pts if p["has_barrett"])
    ppk_total = sum(1 for p in pts if p["has_ppk"])
    msi_h_total = sum(1 for p in pts if p["msi_h"])

    return {
        "atlas":             "Hereditary-Esophageal-Cancer-Predisposition-Atlas",
        "atlas_id":          "hereditary-esophageal-cancer-predisposition-atlas",
        "subtitle":          "Complete 8-Gene RHBDF2-TP53-CDH1-BRCA2-ATM-MLH1-MSH2-PALB2 Reference",
        "total_patients":    len(pts),
        "gene_cohorts":      len(ATLAS_GENES),
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "esophageal_cancer_cases": cancer_total,
        "cancer_rate_pct":   round(100 * cancer_total / len(pts), 1),
        "barrett_cases":     barrett_total,
        "ppk_cases":         ppk_total,
        "msi_h_cases":       msi_h_total,
        "gene_summary":      list(gene_counts.values()),
        "key_clinical_rules": [
            "RHBDF2 (TOC/Howel-Evans): 95% lifetime esophageal SCC — annual OGD with Lugol iodine/NBI from age 20yr MANDATORY; focal PPK onset age 5-15yr is the sentinel marker",
            "TP53 (LFS): AVOID RADIATION ABSOLUTELY — omit CROSS chemoradiotherapy protocol; use surgery + chemotherapy only; WB-MRI Toronto annually",
            "CDH1 (HDGC): prophylactic total gastrectomy age 20-30yr MANDATORY — OGD alone cannot exclude diffuse gastric cancer; Barrett's/GEJ adenocarcinoma 2-3x elevated",
            "BRCA2/PALB2 (HRD): cisplatin preferred over carboplatin (ICL formation exploits HRD); olaparib PARP inhibitor maintenance after platinum response",
            "ATM heterozygotes: reduce esophageal RT dose 20-30% if unavoidable — G2/M checkpoint impairment even in carriers; prefer surgery-first",
            "MLH1/MSH2 (Lynch): esophageal cancer RARE (0.4-1%); MSI-H adenocarcinoma — pembrolizumab FDA2017 active; aspirin 600mg CAPP2 50% risk reduction",
        ],
        "histology_summary": {
            "RHBDF2_note": "SQUAMOUS CELL CARCINOMA ONLY — TOC/RHBDF2 does NOT predispose to adenocarcinoma",
            "CDH1_note":   "ADENOCARCINOMA (GEJ/gastric cardia) — NOT squamous; Barrett's mechanism",
            "Lynch_note":  "MSI-H ADENOCARCINOMA predominantly — Lynch esophageal is adenocarcinoma not SCC",
            "BRCA2_PALB2_ATM": "BOTH SCC and adenocarcinoma elevated — HRD drives chromosomal instability in both histologies",
        },
    }


def generate_breakdown() -> dict:
    pts = _all_patients()
    per_gene = {}
    for g in ATLAS_GENES:
        gene = g["gene"]
        gpts = [p for p in pts if p["gene"] == gene]
        ca_pts = [p for p in gpts if p["has_cancer"]]
        hist_dist = {}
        for p in ca_pts:
            if p["histology"]:
                hist_dist[p["histology"]] = hist_dist.get(p["histology"], 0) + 1
        site_dist = {}
        for p in ca_pts:
            if p["cancer_site"]:
                site_dist[p["cancer_site"]] = site_dist.get(p["cancer_site"], 0) + 1
        stage_dist = {}
        for p in ca_pts:
            if p["cancer_stage"]:
                stage_dist[p["cancer_stage"]] = stage_dist.get(p["cancer_stage"], 0) + 1
        per_gene[gene] = {
            "gene":            gene,
            "syndrome":        g["syndrome"],
            "inheritance":     g["inheritance"],
            "locus":           g["locus"],
            "n":               len(gpts),
            "cancer_n":        len(ca_pts),
            "cancer_pct":      round(100 * len(ca_pts) / len(gpts), 1) if gpts else 0,
            "esophageal_risk": g["esophageal_risk"],
            "histology_distribution": hist_dist,
            "site_distribution":      site_dist,
            "stage_distribution":     stage_dist,
            "key_avoid":       g["key_avoid"],
            "key_rule":        g["key_rule"],
            "targeted_rx":     g["targeted_rx"],
            "surveillance":    g["surveillance"],
            "top_variants": [
                {"variant": v[0], "effect": v[1], "exon": v[2], "domain": v[3], "freq": round(v[4], 2)}
                for v in VARIANTS[gene]
            ],
        }

    all_hist = {}
    for p in pts:
        if p["histology"]:
            all_hist[p["histology"]] = all_hist.get(p["histology"], 0) + 1

    return {
        "per_gene": list(per_gene.values()),
        "histology_distribution": all_hist,
        "hrd_specific": {
            "cisplatin_rule":    "BRCA2/PALB2/ATM: cisplatin PREFERRED over carboplatin — superior ICL formation exploits HRD",
            "parp_inhibitor":    "Olaparib (BRCA2), rucaparib (BRCA1/2/PALB2): PARP inhibitor maintenance after platinum response",
            "radiation_rule":    "TP53: AVOID RT ABSOLUTELY (LFS); ATM: reduce RT dose 20-30% even heterozygotes; CDH1/RHBDF2: no specific RT contraindication",
        },
        "lynch_msi_summary": {
            "msi_h_pembrolizumab": "Pembrolizumab FDA2017: first tumour-agnostic approval for MSI-H/dMMR (MLH1 or MSH2 LOF); ORR 40-57% esophageal",
            "aspirin_capp2":       "Aspirin 600mg daily: CAPP2 50% CRC risk reduction; likely esophageal benefit in Lynch carriers",
            "mlpa_mandatory":      "EPCAM 3-prime deletion (MSH2 silencing) detected ONLY by MLPA — not by sequencing alone",
        },
        "surveillance_by_risk": {
            "annual_ogi_from_20yr": "RHBDF2 (TOC/Howel-Evans): Lugol iodine + NBI — highest priority, 95% lifetime risk",
            "annual_ogi_from_25yr": "TP53 LFS, CDH1 HDGC (Cambridge 56-biopsy protocol until gastrectomy)",
            "annual_ogi_from_45yr": "MLH1/MSH2 Lynch: Barrett's protocol focus (adenocarcinoma subtype)",
            "annual_ogi_from_50yr": "BRCA2, ATM, PALB2: Barrett's surveillance protocol",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Esophageal-Cancer-Predisposition-Atlas",
        "genes": [
            {
                "gene":                  g["gene"],
                "locus":                 g["locus"],
                "protein_function":      g["protein"],
                "protein_size":          g["protein_size"],
                "syndrome":              g["syndrome"],
                "inheritance":           g["inheritance"],
                "esophageal_risk":       g["esophageal_risk"],
                "pathognomonic_features":g["pathognomonic"],
                "absolutely_avoid":      g["key_avoid"],
                "mandatory_rule":        g["key_rule"],
                "surveillance_protocol": g["surveillance"],
                "targeted_therapies":    g["targeted_rx"],
                "variants": [
                    {"variant": v[0], "protein_effect": v[1], "location": v[2], "domain_impact": v[3]}
                    for v in VARIANTS[g["gene"]]
                ],
            }
            for g in ATLAS_GENES
        ],
        "key_clinical_concepts": {
            "rhbdf2_toc_mechanism": (
                "RHBDF2 GOF mutations activate ADAM17 (TACE) via ER export — ADAM17 hyperactivation "
                "sheds EGFR ligands (EGF, TGF-α, HB-EGF, amphiregulin) creating autocrine EGFR "
                "signalling in esophageal squamous epithelium. Focal PPK is the same mechanism in "
                "palmoplantar keratinocytes. Cetuximab is on-mechanism (anti-EGFR) for RHBDF2-driven SCC. "
                "95% lifetime esophageal SCC risk — HIGHEST hereditary esophageal predisposition known."
            ),
            "tp53_radiation_rule": (
                "LFS (TP53 LOF): G1 checkpoint absent — radiotherapy induces new tumours in radiation field. "
                "Radiation-induced sarcoma 30% in treated LFS cases. For esophageal cancer: "
                "OMIT CROSS chemoradiotherapy protocol; substitute surgery + cisplatin/5-FU without RT. "
                "WB-MRI Toronto Protocol annually from birth is the LFS surveillance cornerstone."
            ),
            "cdh1_gastrectomy_rule": (
                "CDH1 LOF (HDGC): E-cadherin loss drives diffuse-type (signet ring cell) gastric cancer "
                "and lobular breast cancer. OGD with 56-biopsy Cambridge protocol is mandatory pre-gastrectomy "
                "but CANNOT reliably exclude early diffuse gastric cancer — prophylactic total gastrectomy "
                "age 20-30yr MANDATORY regardless of negative endoscopy. Barrett's/GEJ adenocarcinoma "
                "2-3x elevated via E-cadherin loss at the squamocolumnar junction."
            ),
            "hrd_cisplatin_preference": (
                "BRCA2, PALB2, ATM: homologous recombination deficiency (HRD) — "
                "cisplatin creates interstrand crosslinks (ICLs) that HRD-cells cannot repair. "
                "CISPLATIN PREFERRED OVER CARBOPLATIN in HRD context for esophageal treatment. "
                "PARP inhibitors (olaparib, rucaparib): trap PARP1/2 at SSBs → collapsed replication "
                "forks → DSBs that HRD cells cannot repair — PARP inhibitor maintenance after platinum."
            ),
            "atm_radiation_sensitivity": (
                "ATM heterozygous carriers: G2/M checkpoint impairment — measurable radiation sensitivity "
                "even with 50% ATM protein level. For esophageal chemoradiotherapy (CROSS protocol): "
                "REDUCE RADIATION DOSE 20-30% in ATM carriers; prefer surgery-first (Ivor-Lewis/McKeown). "
                "Biallelic A-T: esophageal RT = ABSOLUTELY CONTRAINDICATED (equivalent to FA alkylating CI)."
            ),
            "lynch_msi_h_esophageal": (
                "MLH1/MSH2 Lynch esophageal cancer: RARE (0.4-1% lifetime); predominantly ADENOCARCINOMA "
                "with MSI-H (microsatellite instability high) — NOT squamous cell carcinoma. "
                "Pembrolizumab FDA2017: KEYNOTE-158 ORR 40-57% in MSI-H esophageal — first-line preferred "
                "in MSI-H/dMMR context. MLPA essential: EPCAM 3-prime deletion silences MSH2 epigenetically "
                "and is NOT detected by standard sequencing."
            ),
        },
        "abbreviations": {
            "TOC":    "Tylosis with Esophageal Cancer (Howel-Evans Syndrome)",
            "PPK":    "Palmoplantar Keratoderma",
            "LFS":    "Li-Fraumeni Syndrome",
            "HDGC":   "Hereditary Diffuse Gastric Cancer",
            "HBOC":   "Hereditary Breast and Ovarian Cancer",
            "HRD":    "Homologous Recombination Deficiency",
            "MSI-H":  "Microsatellite Instability High",
            "dMMR":   "Deficient Mismatch Repair",
            "ICL":    "Interstrand Crosslink",
            "GEJ":    "Gastroesophageal Junction",
            "SCC":    "Squamous Cell Carcinoma",
            "NBI":    "Narrow-Band Imaging",
            "OGD":    "Oesophagogastroduodenoscopy",
            "CAPP2":  "Colorectal Adenoma/Carcinoma Prevention Programme 2 (aspirin RCT)",
            "WB-MRI": "Whole-Body MRI (Toronto Protocol, LFS surveillance)",
            "MLPA":   "Multiplex Ligation-dependent Probe Amplification",
            "ADAM17": "A Disintegrin and Metalloproteinase Domain-Containing Protein 17 (TACE)",
        },
    }
