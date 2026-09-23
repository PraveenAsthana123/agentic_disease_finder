#!/usr/bin/env python3
"""Hereditary-Colorectal-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
APC    (APC regulator of WNT signalling pathway; 2843aa; 5q22.2; AD LOF;
        FAP — 100% CRC lifetime; CHRPE PATHOGNOMONIC; prophylactic colectomy 20-25yr;
        seed SEED_BASE+0) .
MUTYH  (MutY DNA glycosylase; 546aa; 1p34.1; AR LOF;
        MAP — Y179C/G396D founders; biallelic required; attenuated polyposis;
        seed SEED_BASE+1) .
MLH1   (MutL homolog 1; 756aa; 3p22.2; AD LOF;
        Lynch1 — MSI-H PATHOGNOMONIC; BRAF V600E absent confirms germline;
        aspirin CAPP2 50% risk reduction; somatic methylation 90% sporadics;
        seed SEED_BASE+2) .
MSH2   (MutS homolog 2; 934aa; 2p21; AD LOF;
        Lynch2 — Muir-Torre sebaceous PATHOGNOMONIC; urothelial 14% HIGHEST;
        EPCAM 3'-deletion MLPA MANDATORY; Muir-Torre sebaceous gland neoplasms;
        seed SEED_BASE+3) .
MSH6   (MutS homolog 6; 1360aa; 2p16.3; AD LOF;
        Lynch3 — endometrial 71% HIGHEST single MMR gene; MSI-L 30% false negative;
        attenuated CRC 10-22%; IHC ALWAYS the primary test;
        seed SEED_BASE+4) .
PMS2   (PMS1 homolog 2; 862aa; 7p22.1; AD LOF;
        Lynch4 — LOWEST penetrance CRC 15-20%; 4 pseudogenes MLPA MANDATORY;
        biallelic CMMRD — childhood café-au-lait, brain tumour, hematologic;
        seed SEED_BASE+5) .
STK11  (Serine/threonine kinase 11; 433aa; 19p13.3; AD LOF;
        PJS — perioral pigmentation PATHOGNOMONIC; SCTAT ovarian PATHOGNOMONIC;
        GI intussusception EMERGENCY; gastric 29%; pancreatic 36%;
        seed SEED_BASE+6) .
SMAD4  (SMAD family member 4; 552aa; 18q21.2; AD LOF;
        JPS-HHT overlap — juvenile polyposis + telangiectasia + aortic dilatation;
        protein-losing enteropathy; gastrectomy may be required;
        seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3398-3405)
"""
import random

SEED_BASE = 3398

ATLAS_GENES = [
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "WNT-Gatekeeper-310kDa-Beta-Catenin-Destruction-Complex-"
            "FAP-100pct-CRC-Lifetime-HIGHEST-Hereditary-CRC-Gene-"
            "CHRPE-PATHOGNOMONIC-Desmoid-Gardner-Turcot-Type2-"
            "Prophylactic-Colectomy-20-25yr-MANDATORY-OMIM-175100"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 310 kDa / 5q22.2 APC colorectal cancer molecular context: "
            "STRUCTURE: "
            "  2843 aa / 310 kDa; armadillo repeats (aa 453-767): protein-protein interactions; "
            "  15-aa repeats (aa 1020-1169): β-catenin binding 3 repeats; "
            "  20-aa repeats (aa 1342-1427): β-catenin binding + phosphorylation by GSK-3β; "
            "  EB1/ASEF/IQGAP binding domains; nuclear export signal; "
            "  Mutation cluster region (MCR, aa 1250-1464): classic FAP — codon 1309 most common; "
            "CANCER RISKS (CRC FOCUS): "
            "  CRC: 100% lifetime — prophylactic colectomy MANDATORY 20-25yr; "
            "  Duodenal/ampullary adenomas: Spigelman score guides surveillance; periampullary 4-12%; "
            "  Gastric fundic gland polyps (FGP): universal in FAP — 0.2% malignant; "
            "  Hepatoblastoma: 1.5% childhood FAP — AFP surveillance in children; "
            "  Thyroid: papillary cribriform-morular PTC PATHOGNOMONIC 1-12%; "
            "  Desmoid: 10-30% FAP; codon >1310 highest risk; mesenteric location most dangerous; "
            "GENOTYPE-PHENOTYPE (CRITICAL): "
            "  Codon <200 or >1400: attenuated FAP (AFAP) — fewer polyps, later onset; "
            "  Codon 1250-1464 (MCR): classic FAP 1000s polyps; "
            "  Codon 1309 hotspot: dense polyposis + early onset CRC; "
            "  Codon 1328-1580: desmoid-prone FAP; "
            "  Codon 457-1444: CHRPE most prominent; "
            "KEY MANAGEMENT: "
            "  Sigmoidoscopy/colonoscopy from 12-14yr annual until colectomy; "
            "  Prophylactic colectomy: 20-25yr (or when unmanageable polyps); "
            "  IRA vs TPC-IPAA decision based on rectal polyp burden; "
            "  Duodenoscopy with side-viewing scope: from 25yr, frequency based on Spigelman stage; "
            "  CELECOXIB: reduces polyp burden — NOT substitute for colectomy; "
            "  SULINDAC: adjunct for rectal remnant surveillance; "
            "  Desmoid: progressive symptoms = first-line sulindac/celecoxib; imatinib/sorafenib refractory"
        ),
        "syndrome": "Familial Adenomatous Polyposis (FAP) / Attenuated FAP / Gardner syndrome / Turcot type 2",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo 25%",
        "crc_risk": "100% lifetime CRC if unscreened; prophylactic colectomy eliminates most CRC risk",
        "pathognomonic": "CHRPE (congenital hypertrophy of retinal pigment epithelium) — bilateral pigmented fundus lesions; cribriform-morular thyroid PTC",
        "key_avoid": "NEVER delay prophylactic colectomy beyond unmanageable polyposis; NEVER skip duodenal surveillance (periampullary 4-12%); NEVER use CELECOXIB as colectomy substitute",
        "key_rule": "100% CRC = colonoscopy from 12-14yr + prophylactic colectomy 20-25yr MANDATORY. Desmoid = codon >1310 highest risk. CHRPE = classic FAP codon 311-1444.",
        "surveillance": ["Colonoscopy annual from 12-14yr until colectomy", "Duodenoscopy side-viewing from 25yr (Spigelman staging)", "Thyroid ultrasound annual from 20yr", "AFP surveillance children to age 5yr (hepatoblastoma)", "Desmoid MRI if abdominal symptoms"],
        "targeted_rx": "Sulindac/celecoxib (polyp reduction); imatinib/sorafenib desmoid; no systemic chemotherapy for FAP",
    },
    {
        "gene": "MUTYH",
        "protein": (
            "MUTYH -- 1p34.1 Autosomal-Recessive-LOF -- 546aa -- "
            "DNA-Glycosylase-60kDa-Removes-OG-A-Mispairs-BER-"
            "MAP-Y179C-G396D-Founders-Northern-European-"
            "BIALLELIC-REQUIRED-Attenuated-to-Dense-Polyposis-"
            "Somatic-Transversion-CC-GT-Signature-PATHOGNOMONIC-OMIM-604933"
        ),
        "locus": "1p34.1",
        "protein_size": (
            "546 aa / 60 kDa / 1p34.1 MUTYH colorectal cancer molecular context: "
            "STRUCTURE: "
            "  546 aa / 60 kDa; MutT-like domain (aa 1-65): OG binding; "
            "  Catalytic domain (aa 66-350): adenine removal from OG:A mispair; "
            "  C-terminal domain (aa 351-546): PCNA interaction; "
            "  Recognises 8-oxoguanine:adenine mispairs — removes adenine (incorrect); "
            "CANCER RISKS (CRC FOCUS): "
            "  CRC: biallelic carriers 40-100x risk; 43-75% CRC lifetime; "
            "  COLORECTAL POLYPOSIS: oligopolyposis (10-100 adenomas) to attenuated (LT100); "
            "  Rare: dense polyposis phenocopying FAP; "
            "  CRC WITHOUT polyposis: possible — MAP-CRC may have few/no adenomas; "
            "  Endometrial: 2-3x monoallelic; "
            "  Ovarian: modest elevation monoallelic; "
            "  Duodenal: 4% duodenal polyps (milder than FAP); "
            "SOMATIC SIGNATURE PATHOGNOMONIC: "
            "  MUTYH-MAP tumours: CC>TT, CC>GT transversions; somatic APC G>T mutations specific pattern; "
            "  IHC: MMR proteins RETAINED (unlike Lynch) — KEY DISTINCTION; "
            "  MSS (microsatellite stable) — not MSI-H; "
            "BIALLELIC REQUIREMENT: "
            "  MONOALLELIC MUTYH: modest elevation only — NOT MAP; "
            "  PARTNER TESTING MANDATORY when proband biallelic — carrier counselling; "
            "  Y179C + G396D: compound heterozygous most common Northern Europeans; "
            "  Y179C: more severe phenotype — higher CRC risk than G396D; "
            "KEY MANAGEMENT: "
            "  Colonoscopy from 18-25yr every 2yr; "
            "  Polypectomy to clear adenomas; "
            "  Colectomy when unmanageable (>100 adenomas); "
            "  Upper GI: duodenoscopy from 30yr; "
            "  APC testing FIRST in dense polyposis — MUTYH if APC negative"
        ),
        "syndrome": "MUTYH-Associated Polyposis (MAP) — attenuated to dense adenomatous polyposis",
        "inheritance": "AR LOF (autosomal recessive); BIALLELIC required for MAP; monoallelic = modest risk",
        "crc_risk": "Biallelic: 40-100x RR; 43-75% lifetime CRC; oligopolyposis typical",
        "pathognomonic": "CC>GT somatic transversion signature in tumour; MMR-retained / MSS tumours (unlike Lynch); Y179C + G396D compound het most common",
        "key_avoid": "NEVER diagnose MAP from monoallelic MUTYH alone (partner testing MANDATORY); NEVER assume MSI-H (MAP tumours are MSS); NEVER skip APC testing first in dense polyposis",
        "key_rule": "MAP = BIALLELIC MUTYH only. MMR IHC retained = NOT Lynch. Y179C more severe than G396D. Partner testing when biallelic found.",
        "surveillance": ["Colonoscopy from 18-25yr every 1-2yr", "Colectomy when polyposis unmanageable (>100 adenomas)", "Duodenoscopy from 30-35yr", "Partner testing MANDATORY for reproductive counselling", "Monoallelic carriers: colonoscopy from 40yr every 3-5yr"],
        "targeted_rx": "Endoscopic polypectomy; colectomy when necessary; no specific targeted systemic therapy; sulindac adjunct post-colectomy",
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MMR-MutL-Alpha-85kDa-Dimerises-PMS2-"
            "Lynch1-CRC-40-80pct-MSI-H-PATHOGNOMONIC-"
            "BRAF-V600E-Absent-Excludes-Somatic-Methylation-"
            "Aspirin-CAPP2-50pct-Risk-Reduction-"
            "Constitutional-Methylation-Misses-Standard-Sequencing-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 colorectal cancer molecular context: "
            "STRUCTURE: "
            "  756 aa / 85 kDa; N-terminal ATPase (aa 1-335): ATP binding + hydrolysis; "
            "  Linker (aa 336-500): connects ATPase to C-terminal; "
            "  C-terminal interaction domain (aa 501-756): PMS2 heterodimer interface (MutLα); "
            "  MLH1-PMS2 = MutLα: mediates MMR after mismatch recognition by MutSα/MutSβ; "
            "CANCER RISKS (CRC FOCUS): "
            "  CRC: 40-80% lifetime Lynch1 — onset typically 40-60yr; right-sided predominance; "
            "  Endometrial: 30-60% female carriers — HIGHEST Lynch endometrial risk; "
            "  Ovarian: 8-13% — ENDOMETRIOID NOT HGSOC (Lynch ovarian distinct from BRCA); "
            "  Gastric: 6-13%; urothelial: 10-14%; brain: 1-4% Turcot glioblastoma; "
            "SOMATIC METHYLATION TRAP: "
            "  90% MLH1-deficient sporadic CRC: somatic MLH1 promoter methylation — NOT Lynch; "
            "  BRAF V600E presence = somatic methylation (NOT germline) — excludes Lynch CRC; "
            "  BRAF V600E ABSENT + IHC MLH1 loss = GERMLINE MLH1 PROBABLE; "
            "  Constitutional methylation (epimutation): MLH1 promoter methylated in germline — "
            "    missed by standard sequencing; epigenetic test required if IHC loss + sequencing negative; "
            "ASPIRIN CAPP2: "
            "  600mg aspirin daily for 2yr: 50% CRC risk reduction in Lynch — CAPP2 RCT evidence; "
            "  Mechanism: cyclooxygenase + mismatch repair induction; "
            "  Dose debate: 150mg may be sufficient (lower GI side-effects); "
            "KEY MANAGEMENT: "
            "  Colonoscopy every 1-2yr from 25yr; "
            "  Aspirin 100-600mg daily RECOMMENDED (CAPP2 evidence Level A); "
            "  Endometrial surveillance: annual transvaginal ultrasound + CA-125 from 30-35yr; "
            "  Pembrolizumab FDA2017: ANY MSI-H tumour — MLH1-Lynch colorectal included; "
            "  Durvalumab/nivolumab: second-line dMMR CRC options"
        ),
        "syndrome": "Lynch syndrome type 1 (Hereditary Non-Polyposis CRC — HNPCC type 1)",
        "inheritance": "AD LOF (autosomal dominant); constitutional methylation (epimutation) also exists",
        "crc_risk": "40-80% CRC lifetime; right-sided predominance; onset 40-60yr",
        "pathognomonic": "MSI-H + MLH1-IHC loss + BRAF V600E absent = Lynch CRC; BRAF V600E present = sporadic methylation (excludes Lynch)",
        "key_avoid": "NEVER call MLH1 IHC loss 'Lynch' without BRAF V600E exclusion; NEVER skip aspirin counselling (CAPP2 50% reduction); NEVER miss constitutional methylation with sequencing alone",
        "key_rule": "BRAF V600E absent + MLH1-IHC loss = germline probable. Aspirin CAPP2 Level A. Right-sided CRC + MSI-H + young onset = Lynch until proven otherwise.",
        "surveillance": ["Colonoscopy every 1-2yr from 25yr", "Aspirin 100-600mg daily (CAPP2 Level A)", "Transvaginal US + CA-125 from 30-35yr (endometrial/ovarian)", "Urinalysis + urine cytology from 30yr (urothelial)", "Gastric: H. pylori test/treat; gastroscopy from 30-35yr in high-risk families"],
        "targeted_rx": "Pembrolizumab FDA2017 (any MSI-H); aspirin chemoprevention; standard CRC chemotherapy; immune checkpoint inhibitor first-line metastatic dMMR CRC",
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MMR-MutS-Alpha-MutS-Beta-Scaffold-105kDa-"
            "Lynch2-CRC-40-80pct-Urothelial-14pct-HIGHEST-Lynch-"
            "Muir-Torre-Sebaceous-PATHOGNOMONIC-"
            "EPCAM-3prime-Deletion-MLPA-MANDATORY-"
            "MSI-H-PATHOGNOMONIC-Pembrolizumab-FDA2017-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 105 kDa / 2p21 MSH2 colorectal cancer molecular context: "
            "STRUCTURE: "
            "  934 aa / 105 kDa; MutS-related domains; forms MutSα (MSH2-MSH6) and MutSβ (MSH2-MSH3); "
            "  MutSα: recognises single-base mismatches and 1-nt insertions/deletions; "
            "  MutSβ: recognises 2-6 nt insertion/deletion loops; "
            "  MSH2 is the obligate heterodimer partner for both MSH6 and MSH3; "
            "CANCER RISKS (CRC FOCUS): "
            "  CRC: 40-80% lifetime Lynch2 — right-sided predominance; onset 40-60yr; "
            "  UROTHELIAL 14%: HIGHEST Lynch gene for urothelial (bladder + ureter) — annual surveillance; "
            "  Endometrial: 40-60% female carriers; ovarian: 8-13%; gastric: 9-19%; "
            "  Sebaceous neoplasms (Muir-Torre): sebaceoma, sebaceous carcinoma — PATHOGNOMONIC; "
            "  Keratoacanthoma: associated Muir-Torre — NOT always sebaceous; "
            "  Small bowel: 2-4% (HIGHEST Lynch gene for small bowel); "
            "EPCAM CRITICAL: "
            "  EPCAM gene (upstream 2p21): 3'-end deletions silence MSH2 via promoter methylation; "
            "  EPCAM deletions: 1-3% of all Lynch — MISSED by MSH2 sequencing and deletion analysis; "
            "  EPCAM: MSH2 IHC loss + MSH2 sequencing negative = TEST EPCAM MLPA MANDATORY; "
            "  EPCAM Lynch: urothelial cancer risk LOWER than germline MSH2 point mutations; "
            "MUIR-TORRE SYNDROME: "
            "  MSH2 (most common) or MLH1; sebaceous gland neoplasms + internal malignancy; "
            "  Sebaceoma at ANY AGE = germline MMR testing MANDATORY; "
            "KEY MANAGEMENT: "
            "  Colonoscopy every 1-2yr from 25yr; "
            "  Annual urine cytology + consideration cystoscopy from 30yr (urothelial 14%); "
            "  Annual skin exam from diagnosis (sebaceous neoplasm); "
            "  EPCAM MLPA when MSH2 sequencing negative; "
            "  Aspirin CAPP2 applies to all Lynch including MSH2"
        ),
        "syndrome": "Lynch syndrome type 2 (HNPCC type 2) / Muir-Torre syndrome",
        "inheritance": "AD LOF; EPCAM 3'-deletion upstream → methylation silences MSH2",
        "crc_risk": "40-80% CRC lifetime; urothelial 14% HIGHEST Lynch gene",
        "pathognomonic": "Muir-Torre: sebaceoma/sebaceous carcinoma + internal cancer; EPCAM deletion = MSH2 silencing missed by sequencing",
        "key_avoid": "NEVER skip EPCAM MLPA when MSH2 sequencing negative + IHC loss; NEVER skip annual urine cytology (urothelial 14% HIGHEST); NEVER miss sebaceous neoplasms (Muir-Torre PATHOGNOMONIC)",
        "key_rule": "MSH2 = Lynch2: urothelial HIGHEST (14%) = annual urine cytology MANDATORY. Sebaceoma at any age = MMR testing MANDATORY. EPCAM MLPA if MSH2 sequencing negative.",
        "surveillance": ["Colonoscopy every 1-2yr from 25yr", "Annual urine cytology from 30yr (urothelial 14% HIGHEST)", "Annual skin exam (sebaceous/keratoacanthoma — Muir-Torre)", "EPCAM MLPA if MSH2 sequencing negative + IHC loss", "Aspirin 100-600mg daily CAPP2 Level A"],
        "targeted_rx": "Pembrolizumab FDA2017 (MSI-H); aspirin chemoprevention; standard CRC therapy; immune checkpoint inhibitor metastatic dMMR CRC",
    },
    {
        "gene": "MSH6",
        "protein": (
            "MSH6 -- 2p16.3 Autosomal-Dominant-LOF -- 1360aa -- "
            "MMR-MutS-Alpha-160kDa-Single-Base-Mismatch-Recogniser-"
            "Lynch3-Endometrial-71pct-ABSOLUTE-HIGHEST-Single-MMR-Gene-"
            "MSI-L-30pct-False-Negative-KEY-DISTINCTION-"
            "Attenuated-CRC-10-22pct-LOWER-Than-MLH1-MSH2-OMIM-600678"
        ),
        "locus": "2p16.3",
        "protein_size": (
            "1360 aa / 160 kDa / 2p16.3 MSH6 colorectal cancer molecular context: "
            "STRUCTURE: "
            "  1360 aa / 160 kDa; PWWP domain (aa 1-129): histone binding; "
            "  MH2 domain (aa 130-360): essential for heterodimer formation with MSH2; "
            "  PCNA interacting peptide (PIP box): tethers MutSα to replication foci; "
            "  ATPase domain (aa 950-1130): ADP/ATP cycling for mismatch verification; "
            "  Forms MutSα with MSH2 — recognises single-base mismatches and 1-nt indels; "
            "CANCER RISKS (CRC FOCUS): "
            "  CRC: 10-22% lifetime — ATTENUATED vs MLH1/MSH2 40-80%; LOWER PENETRANCE CRITICAL; "
            "  ENDOMETRIAL: 71% ABSOLUTE HIGHEST lifetime — dominates MSH6 phenotype; "
            "  Ovarian: 10-17%; gastric: 4-9%; urothelial: 7-11%; "
            "MSI-L FALSE NEGATIVE CRITICAL: "
            "  MSH6-deficient CRC: 30% are MSI-L (not MSI-H) — microsatellite instability is LOW; "
            "  Standard 5-marker panel (BAT25, BAT26, D2S123, D5S346, D17S250): misses MSH6-Lynch; "
            "  IHC MSH6 loss is PRIMARY TEST — MSI testing can be negative in MSH6-Lynch; "
            "  Extended marker panel or IHC resolves this — MSI testing ALONE not sufficient; "
            "  MSH6-Lynch families: CRC may not be the presenting cancer — endometrial is; "
            "LATE ONSET: "
            "  MSH6 Lynch: CRC onset often 60-70yr (later than MLH1/MSH2); "
            "  Endometrial onset: 50-60yr; "
            "  Women with MSH6: endometrial surveillance DOMINANT priority; "
            "KEY MANAGEMENT: "
            "  Colonoscopy every 1-2yr from 30yr (later start than MLH1/MSH2 per attenuated CRC risk); "
            "  Annual transvaginal US + endometrial sampling from 35yr MANDATORY (endometrial 71%); "
            "  IHC always primary test — NOT MSI alone; "
            "  Aspirin CAPP2 applies; pembrolizumab for MSI-H subset MSH6"
        ),
        "syndrome": "Lynch syndrome type 3 (HNPCC type 3) — endometrial phenotype dominant",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "crc_risk": "10-22% CRC lifetime (ATTENUATED); endometrial 71% DOMINATES phenotype",
        "pathognomonic": "MSI-L/MSS in MSH6-deficient CRC (30%) — IHC is primary test; endometrial 71% = HIGHEST single MMR gene for endometrial",
        "key_avoid": "NEVER rely on MSI testing alone for MSH6-Lynch (30% MSI-L false negative); NEVER delay endometrial surveillance (71% HIGHEST); NEVER start colonoscopy at 25yr (attenuated — start 30yr per risk)",
        "key_rule": "MSH6 = endometrial phenotype dominant (71%). MSI-L in 30% = IHC primary ALWAYS. Attenuated CRC (10-22%) NOT 40-80%. Late onset 60-70yr.",
        "surveillance": ["Colonoscopy every 1-2yr from 30-35yr (attenuated CRC)", "Annual transvaginal US + endometrial sampling from 35yr MANDATORY (endometrial 71%)", "IHC testing as primary MMR test (not MSI alone)", "Ovarian: CA-125 + TVUS from 35yr", "Urine cytology from 35yr (urothelial 7-11%)"],
        "targeted_rx": "Pembrolizumab for MSI-H subset; standard CRC therapy; endometrial: lenvatinib+pembrolizumab or carboplatin+paclitaxel; aspirin CAPP2",
    },
    {
        "gene": "PMS2",
        "protein": (
            "PMS2 -- 7p22.1 Autosomal-Dominant-LOF -- 862aa -- "
            "MMR-MutL-Alpha-96kDa-Endonuclease-PMS2-"
            "Lynch4-LOWEST-Lynch-Penetrance-CRC-15-20pct-"
            "4-Pseudogenes-MLPA-MANDATORY-SEQUENCING-DIFFICULT-"
            "Biallelic-CMMRD-Constitutional-MMR-Deficiency-"
            "Childhood-Brain-Haematologic-CRC-Cafe-au-Lait-OMIM-600259"
        ),
        "locus": "7p22.1",
        "protein_size": (
            "862 aa / 96 kDa / 7p22.1 PMS2 colorectal cancer molecular context: "
            "STRUCTURE: "
            "  862 aa / 96 kDa; N-terminal ATPase (aa 1-362): ATP hydrolysis; "
            "  Linker domain (aa 363-540); "
            "  C-terminal endonuclease (aa 541-862): METAL-DEPENDENT ENDONUCLEASE — unique to PMS2; "
            "  PMS2 is the endonuclease in MutLα (MLH1-PMS2): incises strand during MMR; "
            "  MLH1 loss → PMS2 loss on IHC (MLH1 stabilises PMS2); "
            "  PMS2 loss with MLH1 RETAINED = isolated PMS2 defect (Lynch4) — NOT MLH1 Lynch; "
            "CANCER RISKS (CRC FOCUS): "
            "  CRC: 15-20% lifetime — LOWEST Lynch penetrance; onset 60-70yr typically; "
            "  Endometrial: 15-26% — LOWEST Lynch endometrial; "
            "  Ovarian: 5%; urothelial: 5-6%; gastric: rare; "
            "PSEUDOGENE PROBLEM (CRITICAL): "
            "  4 PMS2 pseudogenes (PMS2CL, PMS2P1-3): all on chromosome 7 near PMS2; "
            "  Standard sequencing: pseudogene homology causes false results; "
            "  MLPA MANDATORY for PMS2 deletion/duplication analysis — standard MLPA misses pseudogene; "
            "  PMS2-specific MLPA probe set (e.g. SALSA MLPA P008-D1) required; "
            "  Long-range PCR + sequencing: required for exon 11-15 pseudo region; "
            "CMMRD (BIALLELIC PMS2): "
            "  Constitutional MMR Deficiency: biallelic PMS2 most common CMMRD gene; "
            "  Childhood onset: brain tumours (glioblastoma), hematologic malignancy (T-ALL, lymphoma), Lynch-spectrum CRC early; "
            "  Café-au-lait macules: NF1-like phenocopy — CMMRD misdiagnosed as NF1; "
            "  CMMRD: hypermutation → immunotherapy response EXCEPTIONAL; "
            "  CMMRD screening: children with multiple cancers + café-au-lait + family history; "
            "KEY MANAGEMENT: "
            "  Colonoscopy every 2yr from 35yr (attenuated CRC 15-20%); "
            "  MLPA mandatory — PMS2-specific probe set; "
            "  Endometrial: annual TVUS from 40yr; "
            "  CMMRD children: annual MRI brain + haematology"
        ),
        "syndrome": "Lynch syndrome type 4 (HNPCC type 4) / CMMRD (biallelic PMS2)",
        "inheritance": "AD LOF (Lynch4); AR biallelic (CMMRD — Constitutional MMR Deficiency)",
        "crc_risk": "15-20% CRC lifetime (LOWEST Lynch); onset 60-70yr; biallelic CMMRD = childhood CRC",
        "pathognomonic": "IHC: PMS2 loss with MLH1 retained = isolated PMS2 Lynch; Café-au-lait NF1-phenocopy in children = CMMRD until proven otherwise",
        "key_avoid": "NEVER use standard MLPA for PMS2 (4 pseudogenes — PMS2-specific MLPA mandatory); NEVER miss CMMRD in children with NF1-like features; NEVER start colonoscopy at 25yr (attenuated — start 35yr)",
        "key_rule": "PMS2 = LOWEST Lynch (15-20% CRC). 4 pseudogenes = PMS2-specific MLPA MANDATORY. IHC PMS2 loss + MLH1 retained = isolated Lynch4. CMMRD = biallelic childhood emergency.",
        "surveillance": ["Colonoscopy every 2yr from 35yr (attenuated CRC)", "Annual TVUS endometrial sampling from 40yr", "MLPA PMS2-specific probe set MANDATORY", "CMMRD: annual MRI brain + haematology + oncology from childhood", "Urinalysis/cytology from 40yr"],
        "targeted_rx": "Pembrolizumab for MSI-H (Lynch4/CMMRD); CMMRD: exceptional immunotherapy response hypermutated; standard CRC chemotherapy; aspirin CAPP2",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-AMPK-Master-Kinase-48kDa-"
            "PJS-Perioral-Pigmentation-PATHOGNOMONIC-"
            "SCTAT-Ovarian-PATHOGNOMONIC-"
            "GI-Intussusception-EMERGENCY-"
            "Gastric-29pct-Pancreatic-36pct-"
            "Adenoma-Malignum-Cervix-PATHOGNOMONIC-OMIM-175200"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 colorectal cancer molecular context: "
            "STRUCTURE: "
            "  433 aa / 48 kDa; N-terminal kinase domain (aa 49-309): LKB1 catalytic; "
            "  STRAD-binding: forms complex with STRAD + MO25 for cytoplasmic activation; "
            "  Downstream: AMPK pathway — energy sensing, mTORC1 inhibition; "
            "  Farnesylation site C-terminal: membrane targeting; "
            "CANCER RISKS (CRC/GI FOCUS): "
            "  CRC: 35-40% lifetime PJS — hamartomatous (NOT adenomatous); polyps large but malignant rare per polyp; "
            "  GASTRIC: 29% lifetime HIGHEST non-hereditary single-gene gastric risk; "
            "  PANCREATIC: 36% lifetime HIGHEST single gene hereditary pancreatic risk; "
            "  SMALL BOWEL: 13% lifetime — surveillance critical (enteroscopy); "
            "  BREAST: 50% HIGHEST non-BRCA single-gene hereditary breast risk; "
            "  OVARIAN: 21% — SCTAT (sex cord tumour with annular tubules) PATHOGNOMONIC; "
            "  CERVICAL: 27% — adenoma malignum (minimal deviation adenocarcinoma) PATHOGNOMONIC; "
            "  LUNG: 7-17% (KRAS co-mutation — immunotherapy resistance); "
            "PATHOGNOMONIC FEATURES: "
            "  Perioral pigmentation: mucocutaneous macules lips/buccal/fingers — PATHOGNOMONIC PJS; "
            "  Present in infancy; fades after puberty — diagnosis window childhood; "
            "  SCTAT: benign sex cord tumour ovary — PJS hallmark; can calcify; "
            "  Adenoma malignum: minimal deviation cervical adenocarcinoma — most PJS-associated; "
            "INTUSSUSCEPTION EMERGENCY: "
            "  Large hamartomatous polyps: lead point for intussusception at any age; "
            "  GI endoscopy from 8yr MANDATORY — clear polyps >1-1.5 cm to prevent intussusception; "
            "  Recurrent intussusceptions from childhood: classic PJS presentation; "
            "KEY MANAGEMENT: "
            "  Upper and lower GI endoscopy + video capsule from 8yr; "
            "  Clear polyps >1-1.5 cm during endoscopy; "
            "  Pancreatic EUS/MRI from 35yr annual; "
            "  Breast MRI from 25yr annual; "
            "  Cervical smear annual + colposcopy if abnormal; "
            "  Ovarian annual TVUS"
        ),
        "syndrome": "Peutz-Jeghers Syndrome (PJS) — GI hamartomatous polyposis + mucocutaneous pigmentation",
        "inheritance": "AD LOF (autosomal dominant); de novo 25%",
        "crc_risk": "35-40% CRC lifetime; gastric 29%, pancreatic 36%, small bowel 13% also elevated",
        "pathognomonic": "Perioral/buccal/digital mucocutaneous macules (present in infancy); SCTAT ovarian; adenoma malignum cervix — ALL PATHOGNOMONIC PJS",
        "key_avoid": "NEVER delay GI endoscopy beyond 8yr (intussusception risk); NEVER miss SCTAT or adenoma malignum (PJS PATHOGNOMONIC); NEVER skip pancreatic surveillance (36% HIGHEST hereditary)",
        "key_rule": "PJS: GI endoscopy from 8yr to clear polyps MANDATORY. Perioral pigmentation = PJS until proven. Pancreatic 36% = EUS/MRI from 35yr. SCTAT + adenoma malignum = cervical/ovarian PATHOGNOMONIC.",
        "surveillance": ["Upper + lower GI endoscopy + capsule from 8yr every 2-3yr", "Pancreatic EUS/MRI annual from 35yr (36% risk)", "Breast MRI annual from 25yr (50% risk)", "Annual cervical smear + colposcopy", "Ovarian TVUS annual + SCTAT awareness"],
        "targeted_rx": "mTOR inhibitors (investigational for PJS polyps); no standard targeted therapy; selumetinib for KRAS-coMut lung (limited PJS data)",
    },
    {
        "gene": "SMAD4",
        "protein": (
            "SMAD4 -- 18q21.2 Autosomal-Dominant-LOF -- 552aa -- "
            "TGF-Beta-Signalling-Mediator-60kDa-"
            "JPS-HHT-Overlap-Juvenile-Polyposis-Telangiectasia-"
            "Aortic-Dilatation-PATHOGNOMONIC-"
            "Protein-Losing-Enteropathy-Gastrectomy-May-Be-Required-"
            "Gastric-Polyposis-SMAD4-Specific-Phenotype-OMIM-174900"
        ),
        "locus": "18q21.2",
        "protein_size": (
            "552 aa / 60 kDa / 18q21.2 SMAD4 colorectal cancer molecular context: "
            "STRUCTURE: "
            "  552 aa / 60 kDa; MH1 domain (aa 1-130): DNA binding; "
            "  Linker (aa 131-260); "
            "  MH2 domain (aa 261-552): R-SMAD interaction + transcriptional activation; "
            "  SMAD4 is the central mediator (co-SMAD): transduces TGF-β/BMP signals; "
            "  All R-SMAD complexes (SMAD1/2/3/5/8/9) require SMAD4 to enter nucleus; "
            "  LOF: loss of TGF-β tumour suppression + loss of BMP-mediated differentiation; "
            "CANCER RISKS (CRC/GI FOCUS): "
            "  CRC: 40-70% lifetime in JPS (combined SMAD4 + BMPR1A); "
            "  GASTRIC: 15-20% SMAD4-JPS — gastric polyposis SMAD4-specific (not BMPR1A); "
            "  Upper GI: gastric polyp burden can be massive in SMAD4 — gastrectomy required; "
            "  Small bowel: diffuse polyposis; hepatic: rare; "
            "JPS-HHT OVERLAP (SMAD4-SPECIFIC): "
            "  SMAD4 JPS only (NOT BMPR1A): 20-25% carriers have concurrent HHT; "
            "  BMPR1A JPS: NO HHT — HHT is SMAD4-specific; "
            "  HHT features: epistaxis, telangiectasias, pulmonary AVM, hepatic AVM, brain AVM; "
            "  Annual echocardiogram + bubble contrast echo: pulmonary AVM screening SMAD4 carriers; "
            "AORTIC DILATATION: "
            "  25-30% SMAD4 JPS carriers: aortic root dilatation; "
            "  Mechanism: SMAD4 loss in aortic smooth muscle — TGF-β signalling disrupted; "
            "  Annual cardiac echo + aortic root measurement MANDATORY in SMAD4 carriers; "
            "  SURGERY THRESHOLD: aortic root >50mm or rapid progression; "
            "PROTEIN-LOSING ENTEROPATHY: "
            "  Diffuse SMAD4 gastric/small bowel polyposis: massive protein loss; "
            "  Hypoalbuminaemia, oedema, malnutrition; can be life-limiting if uncontrolled; "
            "  Gastrectomy + polypectomy may be required; "
            "KEY MANAGEMENT: "
            "  Colonoscopy every 1-3yr from 15yr (JPS polyposis); "
            "  Upper GI from 15yr annual (gastric polyposis SMAD4 — aggressive); "
            "  Annual cardiac echo + aortic root measurement; "
            "  Pulmonary AVM: bubble contrast echo SMAD4 carriers; "
            "  HHT features: ENT review (epistaxis); hepatic + brain AVM screening"
        ),
        "syndrome": "Juvenile Polyposis Syndrome (JPS) — SMAD4 subtype + HHT overlap + aortic dilatation",
        "inheritance": "AD LOF (autosomal dominant); de novo variant in ~25%",
        "crc_risk": "40-70% CRC lifetime; gastric polyposis SMAD4-specific (not BMPR1A); protein-losing enteropathy risk",
        "pathognomonic": "HHT + juvenile polyposis = SMAD4 (not BMPR1A); aortic dilatation 25-30%; diffuse gastric polyposis SMAD4-specific",
        "key_avoid": "NEVER assume JPS is BMPR1A without testing SMAD4 (HHT only in SMAD4); NEVER miss aortic surveillance (25-30% dilatation — surgical threshold risk); NEVER delay upper GI (gastric polyposis SMAD4-specific)",
        "key_rule": "JPS + HHT = SMAD4 (not BMPR1A). Aortic echo MANDATORY. Gastric polyposis SMAD4-specific — annual upper GI. Protein-losing enteropathy may require gastrectomy.",
        "surveillance": ["Colonoscopy every 1-3yr from 15yr", "Annual upper GI endoscopy from 15yr (gastric polyposis SMAD4-specific)", "Annual cardiac echo + aortic root from diagnosis (25-30% dilatation)", "Bubble contrast echo + chest CT pulmonary AVM (HHT overlap)", "ENT review: epistaxis management (HHT telangiectasias)"],
        "targeted_rx": "No specific systemic targeted therapy; polypectomy; surgery for unmanageable polyposis or aortic dilatation; standard CRC chemotherapy",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

_TUMOUR_TYPES = {
    "APC":   ["Right-sided CRC (cecum/ascending)", "Left-sided CRC (sigmoid/descending)", "Rectal CRC", "Periampullary adenocarcinoma", "Papillary thyroid (cribriform-morular)", "Desmoid tumour (non-malignant)"],
    "MUTYH": ["Right-sided CRC", "Left-sided CRC", "Rectal CRC", "CRC without polyposis", "Duodenal adenocarcinoma"],
    "MLH1":  ["Right-sided CRC (MSI-H)", "Left-sided CRC (MSI-H)", "Endometrial carcinoma", "Ovarian endometrioid carcinoma", "Gastric carcinoma", "Urothelial carcinoma"],
    "MSH2":  ["Right-sided CRC (MSI-H)", "Left-sided CRC (MSI-H)", "Urothelial carcinoma (ureter/bladder)", "Endometrial carcinoma", "Sebaceous carcinoma (Muir-Torre)", "Small bowel adenocarcinoma"],
    "MSH6":  ["Endometrial carcinoma", "Right-sided CRC (MSI-H)", "Left-sided CRC (MSI-L/MSS)", "Rectal CRC (attenuated)", "Ovarian carcinoma", "Urothelial carcinoma"],
    "PMS2":  ["CRC (MSI-H/MSI-L)", "Endometrial carcinoma", "CMMRD brain tumour (glioblastoma)", "CMMRD T-ALL / lymphoma", "Attenuated Lynch CRC"],
    "STK11": ["CRC (hamartomatous)", "Gastric adenocarcinoma", "Pancreatic ductal adenocarcinoma", "Small bowel adenocarcinoma", "Breast carcinoma", "Cervical adenoma malignum"],
    "SMAD4": ["CRC with juvenile polyposis", "Gastric carcinoma (polyposis)", "Small bowel adenocarcinoma", "Upper GI (JPS-related)", "HHT-related complications"],
}

_VARIANTS_BY_GENE = {
    "APC":   ["c.3927_3931del (codon 1309 — dense FAP)", "c.4348C>T (Arg1450Ter — classic FAP)", "c.1A>G (attenuated FAP)", "Large deletion exons 14-15", "c.3183_3187del (MCR dense)"],
    "MUTYH": ["p.Tyr179Cys (Y179C — severe, Northern European)", "p.Gly396Asp (G396D — moderate, Northern European)", "Y179C/G396D compound heterozygous", "p.Glu480del (Mediterranean)", "Large deletion exon 4"],
    "MLH1":  ["p.Val384Asp (frequent missense)", "c.1038G>A (splice — exon 11)", "Large deletion exon 16", "Constitutional methylation (epimutation)", "c.350C>T (Arg117Ter frameshift)"],
    "MSH2":  ["p.Glu923Ter (truncating — common)", "EPCAM exon 8-9 deletion (3'- silences MSH2)", "c.1216C>T (Arg406Ter)", "Large deletion exons 1-6", "c.942+3A>T (splice — Muir-Torre)"],
    "MSH6":  ["p.Cys1518Tyr (PWWP domain)", "c.3261dupC (frameshift — endometrial dominant)", "p.Thr1219Ile (ATPase)", "c.3439_3440del (late onset)", "p.Ser144Ile (early missense)"],
    "PMS2":  ["c.1A>G (Met1Val — severe)", "p.Arg20Ter (biallelic CMMRD)", "Pseudogene conversion exon 14", "Large deletion exon 9 (MLPA required)", "p.Gly244Val (low penetrance)"],
    "STK11": ["p.Leu67Pro (kinase — severe polyposis)", "p.Phe354Leu (C-terminal)", "Large deletion exon 4-7", "c.921+3A>T (splice)", "de novo frameshift exon 6"],
    "SMAD4": ["p.Arg361His (MH2 domain — aortic dilatation)", "p.Asp351His (MH2 — HHT overlap)", "Large deletion (JPS diffuse)", "c.1081C>T (protein-losing enteropathy)", "p.Ala406Val (missense gastric dominant)"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "APC":   ["Prophylactic colectomy IRA vs TPC-IPAA 20-25yr", "Sulindac/celecoxib polyp reduction (NOT colectomy substitute)", "Duodenoscopy + polypectomy Spigelman staging", "Imatinib/sorafenib desmoid refractory", "Standard CRC chemotherapy (FOLFOX/FOLFIRI/bevacizumab)"],
    "MUTYH": ["Colonoscopy polypectomy every 1-2yr", "Colectomy when >100 adenomas unmanageable", "Standard CRC chemotherapy", "No specific targeted therapy for MAP", "Duodenoscopy + polypectomy from 30yr"],
    "MLH1":  ["Aspirin 100-600mg daily CAPP2 Level A (50% risk reduction)", "Pembrolizumab/dostarlimab MSI-H Lynch CRC (FDA approved)", "FOLFOXIRI + bevacizumab (MMR-proficient subset)", "Endometrial: carboplatin+paclitaxel or lenvatinib+pembrolizumab", "Annual colonoscopy surveillance"],
    "MSH2":  ["Aspirin CAPP2 (applies all Lynch)", "Pembrolizumab MSI-H urothelial (FDA approved Lynch urothelial)", "Annual urine cytology + cystoscopy (urothelial 14%)", "Standard CRC chemotherapy Lynch", "Sebaceous excision Muir-Torre"],
    "MSH6":  ["Aspirin CAPP2 (applies MSH6 Lynch3)", "Pembrolizumab MSI-H subset (30% MSI-L miss)", "Endometrial-directed therapy dominant (71% risk)", "Lenvatinib + pembrolizumab MMR-d endometrial", "IHC primary surveillance not MSI alone"],
    "PMS2":  ["Aspirin CAPP2 (attenuated Lynch4)", "Pembrolizumab for MSI-H CRC", "CMMRD: anti-PD1 exceptional response (hypermutated)", "PMS2-specific MLPA probe MANDATORY diagnostics", "Conservative surveillance (attenuated 15-20% CRC)"],
    "STK11": ["GI polypectomy to prevent intussusception (from 8yr)", "Pancreatic EUS/MRI annual from 35yr surveillance", "Breast MRI annual from 25yr surveillance", "mTOR inhibitors investigational (PJS polyp reduction)", "Standard CRC/pancreatic chemotherapy"],
    "SMAD4": ["Annual upper GI endoscopy + polypectomy", "Colectomy/gastrectomy when unmanageable polyposis", "Aortic root surgery if >50mm or rapid progression", "Pulmonary AVM embolisation (HHT overlap)", "Standard CRC chemotherapy (JPS-CRC)"],
}

SURVEILLANCE_BY_GENE = {
    "APC":   ["Colonoscopy annual from 12-14yr until colectomy", "Duodenoscopy side-viewing from 25yr (Spigelman)", "Thyroid ultrasound annual from 20yr", "Desmoid MRI if abdominal symptoms", "AFP children to 5yr (hepatoblastoma)"],
    "MUTYH": ["Colonoscopy every 1-2yr from 18-25yr (biallelic)", "Duodenoscopy from 30-35yr", "Monoallelic: colonoscopy from 40yr every 3-5yr", "Partner testing MANDATORY reproductive counselling", "APC testing first in dense polyposis"],
    "MLH1":  ["Colonoscopy every 1-2yr from 25yr", "Aspirin CAPP2 daily", "Transvaginal US + CA-125 from 30-35yr", "Urinalysis + urine cytology from 30yr", "H. pylori test/treat; gastroscopy from 30-35yr"],
    "MSH2":  ["Colonoscopy every 1-2yr from 25yr", "Annual urine cytology from 30yr (urothelial 14%)", "Annual skin exam (sebaceous — Muir-Torre)", "EPCAM MLPA if MSH2 sequencing negative", "Aspirin CAPP2 daily"],
    "MSH6":  ["Colonoscopy every 1-2yr from 30-35yr (attenuated)", "Annual transvaginal US + endometrial sampling from 35yr MANDATORY", "IHC primary — not MSI alone", "CA-125 + TVUS ovarian from 35yr", "Urine cytology from 35yr (7-11%)"],
    "PMS2":  ["Colonoscopy every 2yr from 35yr (attenuated 15-20%)", "Annual TVUS + endometrial sampling from 40yr", "MLPA PMS2-specific MANDATORY", "CMMRD: annual MRI brain + haematology + oncology", "Urinalysis from 40yr"],
    "STK11": ["Upper + lower GI endoscopy + capsule from 8yr every 2-3yr", "Pancreatic EUS/MRI annual from 35yr (36% risk)", "Breast MRI annual from 25yr (50% risk)", "Annual cervical smear + colposcopy", "Ovarian TVUS annual (SCTAT + ovarian 21%)"],
    "SMAD4": ["Colonoscopy every 1-3yr from 15yr", "Annual upper GI from 15yr (gastric polyposis SMAD4-specific)", "Annual cardiac echo + aortic root (25-30% dilatation)", "Bubble contrast echo + chest CT (pulmonary AVM HHT)", "ENT review epistaxis (HHT telangiectasias)"],
}


def _make_patients(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)
    tumours = _TUMOUR_TYPES.get(gene, ["CRC NOS"])
    variants = _VARIANTS_BY_GENE.get(gene, ["Pathogenic variant"])
    pts = []
    for i in range(n):
        age = rng.randint(18, 80)
        pts.append({
            "patient_id": f"{gene[:4]}-HCRCA-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "tumour_type": rng.choice(tumours),
            "variant": rng.choice(variants),
            "stage": rng.choice(["Local", "Regional", "Metastatic", "Surveillance"]),
            "msi_status": rng.choice(["MSI-H", "MSS", "MSI-L"]) if gene in ("MLH1","MSH2","MSH6","PMS2") else "MSS",
            "immunotherapy_eligible": rng.random() < (0.72 if gene in ("MLH1","MSH2") else 0.50 if gene in ("MSH6","PMS2") else 0.05),
            "prophylactic_procedure": rng.random() < (0.75 if gene == "APC" else 0.45 if gene == "MUTYH" else 0.20),
            "aspirin_prescribed": rng.random() < (0.65 if gene in ("MLH1","MSH2","MSH6","PMS2") else 0.10),
            "relapse": rng.random() < 0.38,
        })
    return pts


def generate_overview() -> dict:
    cohorts = {}
    for i, g in enumerate(ATLAS_GENES):
        gene = g["gene"]
        pts = _make_patients(gene, SEED_BASE + i)
        cohorts[gene] = pts

    total = sum(len(v) for v in cohorts.values())
    gene_counts = {g: len(pts) for g, pts in cohorts.items()}

    msi_h_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["msi_status"] == "MSI-H") / total, 1
    )
    immunotherapy_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["immunotherapy_eligible"]) / total, 1
    )
    prophylactic_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["prophylactic_procedure"]) / total, 1
    )
    aspirin_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["aspirin_prescribed"]) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )
    metastatic_pct = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["stage"] == "Metastatic") / total, 1
    )

    return {
        "atlas": "Hereditary-Colorectal-Cancer-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference · APC-MUTYH-MLH1-MSH2-MSH6-PMS2-STK11-SMAD4",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "msi_h_rate_pct": msi_h_rate,
        "immunotherapy_eligible_pct": immunotherapy_rate,
        "prophylactic_procedure_pct": prophylactic_rate,
        "aspirin_prescribed_pct": aspirin_rate,
        "mean_age_at_dx": mean_age,
        "metastatic_pct": metastatic_pct,
        "key_facts": [
            "APC: FAP — 100% CRC lifetime; prophylactic colectomy 20-25yr MANDATORY; CHRPE PATHOGNOMONIC; codon 1309 = dense polyposis",
            "MUTYH: MAP — biallelic REQUIRED; Y179C + G396D Northern European founders; MMR RETAINED (MSS unlike Lynch); APC test first in dense polyposis",
            "MLH1: Lynch1 — MSI-H PATHOGNOMONIC; BRAF V600E absent = germline (not somatic methylation); aspirin CAPP2 50% reduction Level A",
            "MSH2: Lynch2 — urothelial 14% HIGHEST Lynch gene; Muir-Torre sebaceous PATHOGNOMONIC; EPCAM MLPA MANDATORY if sequencing negative",
            "MSH6: Lynch3 — endometrial 71% ABSOLUTE HIGHEST single MMR gene; MSI-L 30% false negative (IHC primary NOT MSI); attenuated CRC 10-22%",
            "PMS2: Lynch4 — LOWEST Lynch (15-20% CRC); 4 pseudogenes = PMS2-specific MLPA MANDATORY; biallelic CMMRD = childhood cancer emergency",
            "STK11: PJS — perioral pigmentation PATHOGNOMONIC; GI endoscopy from 8yr (intussusception); pancreatic 36% HIGHEST hereditary risk",
            "SMAD4: JPS — HHT overlap SMAD4-ONLY (not BMPR1A); aortic dilatation 25-30% PATHOGNOMONIC; annual cardiac echo MANDATORY",
        ],
    }


def generate_breakdown() -> dict:
    breakdown = {}
    from collections import Counter
    for i, g in enumerate(ATLAS_GENES):
        gene = g["gene"]
        pts = _make_patients(gene, SEED_BASE + i)
        tumour_counts = Counter(p["tumour_type"] for p in pts)
        variant_counts = Counter(p["variant"] for p in pts)
        top_tumours = tumour_counts.most_common(3)
        top_variants = variant_counts.most_common(3)
        breakdown[gene] = {
            "gene_info": {
                "gene": gene,
                "protein": g["protein"],
                "locus": g["locus"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "crc_risk": g["crc_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "msi_h_pct": round(100 * sum(1 for p in pts if p["msi_status"] == "MSI-H") / len(pts), 1),
            "immunotherapy_pct": round(100 * sum(1 for p in pts if p["immunotherapy_eligible"]) / len(pts), 1),
            "prophylactic_pct": round(100 * sum(1 for p in pts if p["prophylactic_procedure"]) / len(pts), 1),
            "aspirin_pct": round(100 * sum(1 for p in pts if p["aspirin_prescribed"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "top_tumour_types": [{"type": t, "count": c} for t, c in top_tumours],
            "top_variants": [{"variant": v, "count": c} for v, c in top_variants],
            "treatment_protocols": TREATMENT_PROTOCOLS_BY_GENE[gene],
            "surveillance_protocols": SURVEILLANCE_BY_GENE[gene],
        }
    return {"breakdown": breakdown, "genes": _GENE_LIST}


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Colorectal-Cancer-Predisposition-Atlas",
        "definitions": {
            "apc_fap_chrpe": (
                "APC FAP: WNT gatekeeper 2843aa — 100% CRC lifetime if unscreened; "
                "CHRPE (bilateral congenital hypertrophy of retinal pigment epithelium): PATHOGNOMONIC FAP codon 311-1444; "
                "Codon 1309: DENSE polyposis — prophylactic colectomy earliest (adolescence); "
                "Desmoid: codon >1310 — mesenteric desmoid most dangerous; imatinib/sorafenib refractory; "
                "COLECTOMY NOT OPTIONAL: prophylactic IRA or TPC-IPAA 20-25yr MANDATORY — delay = 100% CRC"
            ),
            "mutyh_map_vs_lynch": (
                "MUTYH MAP CRITICAL DISTINCTION from Lynch: "
                "MAP: AR BIALLELIC — monoallelic alone does NOT confer MAP risk; "
                "MAP TUMOURS: MSS/MMR-RETAINED — IHC shows no MMR loss; "
                "Lynch: MSI-H + MMR IHC loss; MAP: MSS + MMR retained; "
                "SOMATIC SIGNATURE: CC>GT transversions (OG:A → C:G corrected) — MAP-specific; "
                "Y179C (more severe CRC risk) vs G396D (moderate risk) — phenotype dictates surveillance intensity"
            ),
            "mlh1_braf_methylation": (
                "MLH1 Lynch vs somatic methylation CRITICAL GATE: "
                "90% MLH1-deficient CRC: somatic MLH1 promoter methylation — NOT germline Lynch; "
                "BRAF V600E present: somatic methylation → EXCLUDES Lynch CRC (nearly absolute rule); "
                "BRAF V600E absent + MLH1 IHC loss: germline MLH1 probable → CASCADE FAMILY; "
                "Constitutional methylation (epimutation): germline MLH1 methylation — missed by DNA sequencing; "
                "CAPP2 aspirin 600mg 2yr: 50% CRC risk reduction Lynch1 — Level A evidence"
            ),
            "msh2_epcam_urothelial": (
                "MSH2 Lynch2 dual-site risk: "
                "UROTHELIAL 14%: HIGHEST Lynch gene for bladder/ureter — annual urine cytology MANDATORY from 30yr; "
                "EPCAM TRAP: 3'-EPCAM deletion epigenetically silences MSH2 (promoter methylation) — "
                "MISSED by MSH2 sequencing AND standard MLPA; EPCAM MLPA probe required; "
                "Muir-Torre: sebaceoma or sebaceous carcinoma at ANY AGE = MMR test MANDATORY; "
                "MSH2 = sebaceous + urothelial DUAL obligation beyond CRC — DO NOT FOCUS ONLY ON CRC"
            ),
            "msh6_msi_l_pitfall": (
                "MSH6 Lynch3 MSI-L false negative CRITICAL: "
                "30% MSH6-deficient CRC: MSI-L (low instability) NOT MSI-H; "
                "Standard 5-marker panel designed for MLH1/MSH2 — MISSES MSH6; "
                "IHC MSH6 is PRIMARY TEST — clinician must ORDER MSH6 IHC specifically; "
                "MSH6 PHENOTYPE: endometrial DOMINATES (71%) not CRC (10-22%); "
                "LATE ONSET: CRC 60-70yr, endometrial 50-60yr — DIFFERENT from MLH1/MSH2 younger onset"
            ),
            "pms2_pseudogenes": (
                "PMS2 Lynch4 pseudogene problem: "
                "4 PSEUDOGENES (PMS2CL, PMS2P1-P3): all chromosome 7, share homology with PMS2; "
                "Standard sequencing: pseudogene sequence contaminates — false variants reported; "
                "Standard MLPA: pseudogene probes give false positive/negative copy number; "
                "PMS2-SPECIFIC MLPA ONLY (e.g. SALSA MLPA P008) + long-range PCR exons 11-15; "
                "CMMRD (biallelic PMS2): childhood cancer emergency — café-au-lait NF1-phenocopy; brain tumour + haematologic + CRC"
            ),
            "stk11_pjs_pathognomonic": (
                "STK11 PJS pathognomonic features: "
                "PERIORAL PIGMENTATION: mucocutaneous macules (lips, buccal, fingertips) — PATHOGNOMONIC; "
                "Present in infancy/childhood; fades after puberty — diagnosis window is childhood; "
                "SCTAT (sex cord tumour with annular tubules): benign ovarian PJS tumour — PATHOGNOMONIC; "
                "ADENOMA MALIGNUM: minimal deviation cervical adenocarcinoma — PATHOGNOMONIC PJS; "
                "INTUSSUSCEPTION: large hamartomas = lead point — GI endoscopy from 8yr CLEARS >1.5cm polyps; "
                "PANCREATIC 36%: HIGHEST hereditary risk single gene — EUS/MRI annual from 35yr MANDATORY"
            ),
            "smad4_jps_hht": (
                "SMAD4 JPS-HHT overlap CRITICAL DISTINCTION: "
                "SMAD4 JPS: 20-25% concurrent HHT (hereditary haemorrhagic telangiectasia); "
                "BMPR1A JPS: NO HHT — HHT is SMAD4-SPECIFIC; "
                "HHT features: epistaxis, telangiectasias, pulmonary AVM, brain AVM, hepatic AVM; "
                "AORTIC DILATATION: 25-30% SMAD4 carriers — annual echo MANDATORY (surgical if >50mm); "
                "GASTRIC POLYPOSIS: SMAD4-specific diffuse — can cause protein-losing enteropathy + gastrectomy; "
                "SMAD4 vs BMPR1A distinction: genotype determines HHT screening obligation"
            ),
            "cascade_testing_crc": (
                "CASCADE TESTING Hereditary CRC Predisposition: "
                "APC: cascade first-degree from diagnosis (100% risk); colonoscopy from 12-14yr; "
                "MUTYH: partner testing MANDATORY (biallelic proband); siblings 25% MAP risk; "
                "MLH1/MSH2/MSH6/PMS2: cascade all first-degree; colonoscopy + gynaecological surveillance; "
                "STK11: first-degree from 8yr GI endoscopy (intussusception risk); "
                "SMAD4: first-degree cardiac echo from diagnosis; upper GI early; "
                "UNIVERSAL: aspirin CAPP2 for all Lynch (MLH1/MSH2/MSH6/PMS2); "
                "PEMBROLIZUMAB: FDA approved any MSI-H regardless of gene — applies Lynch CRC"
            ),
        },
        "key_clinical_distinctions": [
            "APC 100% CRC: prophylactic colectomy 20-25yr MANDATORY — delay = certainty of CRC; CHRPE PATHOGNOMONIC; codon 1309 = dense",
            "MUTYH MAP: BIALLELIC only = MAP risk; MSS tumours (NOT MSI-H); APC first in dense polyposis; partner testing MANDATORY",
            "MLH1 BRAF gate: BRAF V600E present = somatic methylation (NOT Lynch); absent + IHC loss = germline; aspirin CAPP2 50% reduction Level A",
            "MSH2 dual obligation: urothelial 14% HIGHEST Lynch = annual urine cytology; EPCAM MLPA if sequencing negative; sebaceous = Muir-Torre PATHOGNOMONIC",
            "MSH6 endometrial 71%: NOT a CRC-dominant gene — endometrial DOMINATES; MSI-L 30% = IHC primary always; attenuated CRC 10-22%",
            "PMS2 pseudogene trap: 4 pseudogenes = PMS2-specific MLPA MANDATORY; LOWEST Lynch (15-20%); CMMRD biallelic = childhood cancer café-au-lait NF1-phenocopy",
            "STK11 PJS: perioral pigmentation PATHOGNOMONIC; GI from 8yr MANDATORY; intussusception emergency; pancreatic 36% HIGHEST hereditary",
            "SMAD4 HHT exclusive: HHT + JPS = SMAD4 only (never BMPR1A); aortic dilatation annual echo MANDATORY; gastric polyposis SMAD4-specific",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
