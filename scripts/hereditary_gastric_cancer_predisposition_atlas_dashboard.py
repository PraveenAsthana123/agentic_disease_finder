#!/usr/bin/env python3
"""Hereditary-Gastric-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
CDH1    (E-cadherin; 882aa; 16q22.1; AD LOF;
         Hereditary Diffuse Gastric Cancer (HDGC);
         gastric cancer 67-83% lifetime -- HIGHEST single gene;
         prophylactic total gastrectomy MANDATORY age 20-30;
         lobular breast cancer 42% lifetime (women);
         CDH1 p.T340A Portuguese/Newfoundland founder;
         seed SEED_BASE+0) .
CTNNA1  (Alpha-E-catenin; 906aa; 5q31.2; AD LOF;
         HDGC-like (CTNNA1-associated);
         diffuse gastric cancer moderate risk;
         annual gastroscopy with Cambridge protocol;
         no prophylactic gastrectomy consensus yet;
         seed SEED_BASE+1) .
APC     (Adenomatous polyposis coli; 2843aa; 5q22.2; AD LOF;
         FAP / Gardner Syndrome;
         gastric fundic gland polyposis PATHOGNOMONIC;
         gastric cancer 0.5-2% FAP (attenuated 5-10%);
         upper GI surveillance from age 25 MANDATORY;
         prophylactic colectomy 20-25yr;
         seed SEED_BASE+2) .
SMAD4   (SMAD family member 4; 552aa; 18q21.2; AD LOF;
         Juvenile Polyposis Syndrome (JPS) / JPS-HHT overlap;
         gastric juvenile polyps PATHOGNOMONIC;
         gastric cancer 21-34% lifetime -- HIGHEST JPS gene;
         HHT (Hereditary Haemorrhagic Telangiectasia) overlap SMAD4;
         epistaxis + telangiectasiae + AVM (pulmonary/cerebral) PATHOGNOMONIC HHT;
         annual screening CT chest/brain for AVMs MANDATORY;
         seed SEED_BASE+3) .
STK11   (Serine/threonine kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome (PJS);
         gastric cancer 29% lifetime; hamartomatous polyps;
         mucocutaneous melanin macules PATHOGNOMONIC;
         GI endoscopy from age 8yr;
         seed SEED_BASE+4) .
MLH1    (MutL homologue 1; 756aa; 3p22.2; AD LOF;
         Lynch Syndrome type 1 / CMMRD (biallelic);
         gastric cancer 6-13% lifetime; 6-8x elevated;
         MSI-H PATHOGNOMONIC; pembrolizumab FDA 2017;
         aspirin 600mg CAPP2 daily;
         seed SEED_BASE+5) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome;
         gastric cancer elevated in LFS (undifferentiated diffuse);
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+6) .
BRCA2   (BRCA2 homologous recombination scaffold; 3418aa; 13q12.3; AD LOF;
         HBOC / Fanconi Anaemia D1 (biallelic);
         gastric cancer 2-3x elevated monoallelic;
         platinum-based (HRD sensitivity); olaparib FDA 2019 POLO;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3350-3357)
"""
import random

SEED_BASE = 3350

ATLAS_GENES = [
    {
        "gene": "CDH1",
        "protein": (
            "CDH1 -- 16q22.1 Autosomal-Dominant-LOF -- 882aa -- "
            "E-Cadherin-97kDa-Epithelial-Adhesion-HDGC-"
            "Gastric-67-83pct-HIGHEST-Single-Gene-"
            "Prophylactic-Gastrectomy-MANDATORY-Age-20-30-"
            "Lobular-Breast-42pct-Lifetime-OMIM-192090"
        ),
        "locus": "16q22.1",
        "protein_size": (
            "882 aa / 97 kDa / 16q22.1 CDH1 encodes E-cadherin (epithelial cadherin): "
            "STRUCTURE: "
            "  882 aa / 97 kDa; single-pass type I transmembrane glycoprotein; "
            "  Signal peptide (aa 1-22); pro-domain (aa 23-154); "
            "  5 extracellular cadherin (EC) repeats (EC1-EC5): calcium-binding; cell-cell adhesion; "
            "  EC1-EC2 interface: homophilic binding; CDH1 LOF → E-cadherin loss → disrupted adherens junctions; "
            "  Transmembrane domain (aa 714-737); "
            "  Intracellular domain (aa 738-882): binds beta-catenin and p120-catenin; "
            "  CDH1 LOF → free beta-catenin → nuclear WNT signalling → invasion; "
            "  Signet ring cells: intracytoplasmic mucin vacuole displaces nucleus — CDH1 loss PATHOGNOMONIC; "
            "HEREDITARY DIFFUSE GASTRIC CANCER (HDGC): "
            "  GASTRIC CANCER RISK: 67-83% lifetime (female); male risk similar; "
            "  Histology: ALWAYS diffuse-type (signet ring cell carcinoma); "
            "  Signet ring cells on prophylactic gastrectomy in 50-70% of CDH1 carriers at age 20-30yr; "
            "  PROPHYLACTIC TOTAL GASTRECTOMY MANDATORY: recommended age 20-30yr (or within 5yr of youngest relative onset); "
            "  Delay beyond 30yr substantially increases infiltrative carcinoma detection; "
            "  Cambridge protocol: annual gastroscopy with 28+ targeted biopsies in Cambridge protocol; "
            "  NOT reliable as primary surveillance — annual gastroscopy misses signet ring foci; "
            "LOBULAR BREAST CANCER CONCURRENT: "
            "  Lobular breast cancer: 42% lifetime (women CDH1); "
            "  Lobular breast cancer (CDH1 IHC loss in breast tumour PATHOGNOMONIC CDH1 germline); "
            "  Annual breast MRI + mammography from age 30 (women); "
            "  BRCA1/BRCA2 not elevated — lobular specifically linked; "
            "KEY VARIANTS: "
            "  p.T340A: Portuguese/Newfoundland founder (most studied); "
            "  p.W409Ter, p.E748Ter: protein truncating — unequivocally pathogenic; "
            "  Splice site variants (IVS11+1G>T, IVS6+1G>A): highly penetrant; "
            "  Missense variants: functional assay required (CDH1-specific); "
            "SURVEILLANCE CDH1: "
            "  Annual gastroscopy Cambridge protocol (28+ targeted biopsies) until gastrectomy; "
            "  Annual breast MRI + mammography from age 30 (women); "
            "  Genetic cascade testing all first-degree relatives; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Hereditary Diffuse Gastric Cancer (HDGC)",
        "gc_risk": "Gastric cancer 67-83% lifetime — HIGHEST single gene; diffuse signet ring type ALWAYS; gastrectomy by 20-30yr",
        "pathognomonic": "Signet ring cell carcinoma (diffuse type) PATHOGNOMONIC CDH1; lobular breast 42% concurrent; E-cadherin IHC loss",
        "key_avoid": "DELAY GASTRECTOMY — prophylactic total gastrectomy must be done age 20-30yr; gastroscopy alone is unreliable",
        "surveillance": "Annual gastroscopy Cambridge 28+ biopsies until gastrectomy; annual breast MRI from 30 (women); cascade testing",
        "targeted_rx": "Prophylactic total gastrectomy (curative intent); laparoscopic preferred; FLOT/ECF chemotherapy locally advanced CDH1 gastric",
        "key_rule": "PROPHYLACTIC TOTAL GASTRECTOMY MANDATORY age 20-30yr — annual gastroscopy UNRELIABLE for CDH1; 67-83% lifetime risk",
    },
    {
        "gene": "CTNNA1",
        "protein": (
            "CTNNA1 -- 5q31.2 Autosomal-Dominant-LOF -- 906aa -- "
            "Alpha-E-Catenin-100kDa-Actin-Bridge-HDGC-Like-"
            "Diffuse-Gastric-Cancer-Moderate-Risk-"
            "Annual-Gastroscopy-Cambridge-Protocol-"
            "No-Prophylactic-Gastrectomy-Consensus-Yet-OMIM-116805"
        ),
        "locus": "5q31.2",
        "protein_size": (
            "906 aa / 100 kDa / 5q31.2 CTNNA1 encodes alpha-E-catenin (alpha-catenin 1): "
            "STRUCTURE: "
            "  906 aa / 100 kDa; cytoskeletal scaffold protein; "
            "  N-terminal domain (aa 1-264): beta-catenin / vinculin binding; "
            "  M-domain (aa 265-636): dimerisation; "
            "  C-terminal actin-binding domain (aa 637-906): links adherens junction to actin cytoskeleton; "
            "  CTNNA1 bridges E-cadherin/beta-catenin complex to actin via vinculin; "
            "  CTNNA1 LOF → disrupted cell-cell adhesion → increased cell motility → diffuse growth pattern; "
            "  IHC: alpha-catenin loss in tumour (vs CDH1 IHC loss) — different IHC pattern from CDH1; "
            "CTNNA1-ASSOCIATED HDGC-LIKE: "
            "  GASTRIC CANCER RISK: moderate elevated lifetime (estimate 30-50%); less data than CDH1; "
            "  Histology: diffuse-type (signet ring cell carcinoma); identical pathology to CDH1-HDGC; "
            "  CTNNA1 is the second identified HDGC gene after CDH1; "
            "  Expert consensus: annual gastroscopy Cambridge protocol (28+ targeted biopsies); "
            "  Prophylactic gastrectomy: no consensus yet (insufficient outcome data vs CDH1); "
            "  Decision shared between patient + expert centre; consider after age 30 with elevated familial history; "
            "  Lobular breast cancer risk: emerging data (CTNNA1-associated lobular breast also reported); "
            "FAMILIAL ASSESSMENT: "
            "  CTNNA1 inherited: family history of diffuse gastric cancer required for full risk stratification; "
            "  De novo CTNNA1 variants: penetrance potentially lower; "
            "  Cambridge registry / IGCLC protocol recommended for all CTNNA1 carriers; "
            "SURVEILLANCE CTNNA1: "
            "  Annual gastroscopy Cambridge protocol (28+ biopsies); "
            "  Annual breast MRI from age 30-35 (women — lobular breast risk emerging data); "
            "  Referral to HDGC specialist centre (IGCLC); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "HDGC-like (CTNNA1-associated Diffuse Gastric Cancer)",
        "gc_risk": "Diffuse gastric cancer moderate elevated (est. 30-50% lifetime); second HDGC gene after CDH1; less outcome data",
        "pathognomonic": "Diffuse-type signet ring carcinoma (as CDH1-HDGC); alpha-catenin IHC loss in tumour (distinct from E-cadherin loss)",
        "key_avoid": "SAME-DAY GASTRECTOMY DECISION — no consensus yet; engage IGCLC specialist centre before prophylactic surgery",
        "surveillance": "Annual gastroscopy Cambridge 28+ biopsies; annual breast MRI from 30-35 (women); IGCLC specialist referral",
        "targeted_rx": "FLOT/ECF for locally advanced; no CTNNA1-specific targeted therapy; platinum-based empirically; specialist centre",
        "key_rule": "ANNUAL GASTROSCOPY CAMBRIDGE PROTOCOL MANDATORY — no prophylactic gastrectomy consensus yet; IGCLC referral required",
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "APC-310kDa-WNT-Scaffold-FAP-Gardner-"
            "Fundic-Gland-Polyposis-PATHOGNOMONIC-"
            "Gastric-Cancer-0.5-2pct-FAP-AFAP-5-10pct-"
            "Upper-GI-Surveillance-Age-25-MANDATORY-Colectomy-20-25yr-OMIM-611731"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 310 kDa / 5q22.2 APC encodes adenomatous polyposis coli protein: "
            "STRUCTURE: "
            "  2843 aa / 310 kDa; tumour suppressor scaffold; multidomain protein; "
            "  Oligomerisation domain (aa 1-62): APC-APC homodimerisation; "
            "  ARM repeat domain (aa 453-767): beta-catenin binding; "
            "  20-aa repeats (aa 1020-2100): 3× beta-catenin binding; axin binding; "
            "  SAMP motifs: axin binding — destruction complex assembly; "
            "  C-terminal domain: microtubule + EB1 binding; "
            "  APC LOF → destruction complex failure → beta-catenin nuclear accumulation → WNT target genes; "
            "  Mutation cluster region (MCR, aa 1286-1513): genotype-phenotype correlation for polyp density; "
            "FAP / AFAP GASTRIC FEATURES: "
            "  Fundic gland polyps (FGP): PATHOGNOMONIC FAP (multiple FGPs in young patient); "
            "  FGPs: foveolar cell hyperplasia + cystically dilated fundic glands; "
            "  FGPs in FAP: low malignant potential but require surveillance; "
            "  Gastric cancer in FAP: 0.5-2% lifetime (classical FAP); attenuated FAP (AFAP) 5-10%; "
            "  Duodenal/ampullary adenomas: Spigelman staging mandatory; duodenal cancer 4-10% lifetime; "
            "  Gastric adenomas (antrum-predominant in Asian FAP): higher malignant risk; "
            "  UPPER GI SURVEILLANCE FROM AGE 25 MANDATORY: gastroscopy + duodenoscopy; "
            "  Side-viewing duodenoscope for ampulla inspection; "
            "COLORECTAL DOMINANT RISK: "
            "  CRC: 100% lifetime without colectomy (classical FAP); "
            "  Prophylactic colectomy age 20-25yr (colectomy timing guided by polyp burden); "
            "  Celecoxib/sulindac: polyp reduction (not curative); "
            "  CHRPE (congenital hypertrophy of retinal pigment epithelium): PATHOGNOMONIC FAP (>4 bilateral); "
            "GARDNER SYNDROME FEATURES: "
            "  Desmoid tumours: 10-30% FAP; aggressive desmoids can be life-threatening (abdominal wall); "
            "  Osteomas (jaw, skull): PATHOGNOMONIC Gardner; "
            "  Epidermoid cysts; supernumerary teeth; "
            "SURVEILLANCE APC: "
            "  Upper GI (gastroscopy + duodenoscopy) from age 25yr; "
            "  Colonoscopy annually from age 12yr; prophylactic colectomy 20-25yr; "
            "  Annual thyroid USS (papillary thyroid cancer cribriform morular variant PATHOGNOMONIC); "
            "  Desmoid screening: abdominal MRI if prior surgery, family history of desmoid; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Familial Adenomatous Polyposis (FAP) / Attenuated FAP (AFAP) / Gardner Syndrome",
        "gc_risk": "Fundic gland polyposis PATHOGNOMONIC FAP; gastric cancer 0.5-2% (FAP); AFAP 5-10%; duodenal cancer 4-10% (dominant upper GI)",
        "pathognomonic": "Multiple fundic gland polyps PATHOGNOMONIC FAP; CHRPE bilateral >4 PATHOGNOMONIC; cribriform morular PTC PATHOGNOMONIC FAP",
        "key_avoid": "SKIP UPPER GI SURVEILLANCE — gastroscopy + duodenoscopy from age 25 MANDATORY; duodenal cancer 4-10% lifetime FAP",
        "surveillance": "Upper GI (gastro+duodeno) from 25yr; colonoscopy annual from 12yr; colectomy 20-25yr; annual thyroid USS",
        "targeted_rx": "Colectomy (curative CRC prevention); sulindac/celecoxib (polyp reduction); FOLFIRINOX for desmoid-eligible gastric",
        "key_rule": "UPPER GI SURVEILLANCE FROM AGE 25 MANDATORY — duodenal/ampullary cancer 4-10% lifetime; gastroscopy side-viewing scope",
    },
    {
        "gene": "SMAD4",
        "protein": (
            "SMAD4 -- 18q21.2 Autosomal-Dominant-LOF -- 552aa -- "
            "SMAD4-60kDa-TGF-Beta-Signal-Transducer-JPS-HHT-Overlap-"
            "Gastric-Juvenile-Polyps-PATHOGNOMONIC-"
            "Gastric-Cancer-21-34pct-HIGHEST-JPS-Gene-"
            "HHT-Epistaxis-AVM-PATHOGNOMONIC-Annual-CT-Chest-Brain-MANDATORY-OMIM-600993"
        ),
        "locus": "18q21.2",
        "protein_size": (
            "552 aa / 60 kDa / 18q21.2 SMAD4 encodes SMAD family member 4 (co-SMAD): "
            "STRUCTURE: "
            "  552 aa / 60 kDa; central signal transducer of TGF-beta superfamily; "
            "  MH1 domain (aa 1-142): DNA binding; "
            "  Linker region (aa 143-282): regulatory phosphorylation; "
            "  MH2 domain (aa 283-552): SMAD-SMAD interactions + receptor-SMAD binding; "
            "  SMAD4 = co-SMAD: partners R-SMADs (SMAD1/2/3/5/8) → nuclear complex; "
            "  SMAD4 LOF → TGF-beta tumour suppression lost → epithelial-mesenchymal transition; "
            "  18q21 deletion (somatic): present in 50-55% of sporadic pancreatic cancers; "
            "JUVENILE POLYPOSIS SYNDROME (JPS) — SMAD4: "
            "  GASTRIC CANCER: 21-34% lifetime — HIGHEST gastric cancer risk of all JPS genes; "
            "  Gastric juvenile polyps (hamartomatous): pathognomonic JPS; "
            "  Gastric juvenile polyposis (GJP): predominantly gastric polyps — SMAD4 exclusive feature; "
            "  Upper GI endoscopy from age 15yr (gastric polyp surveillance); "
            "  Colonic JPS: colorectal cancer 39-68% lifetime; "
            "  Colonoscopy from age 15yr annually; "
            "JPS-HHT OVERLAP (SMAD4 ONLY — KEY DISTINCTION FROM BMPR1A): "
            "  SMAD4 JPS (not BMPR1A JPS): 15-22% of SMAD4 carriers develop HHT features; "
            "  HHT (Hereditary Haemorrhagic Telangiectasia): epistaxis PATHOGNOMONIC + cutaneous telangiectasiae; "
            "  Pulmonary AVM: arteriovenous malformation — paradoxical embolism → stroke risk; "
            "  Cerebral AVM: haemorrhage → potentially catastrophic; "
            "  Hepatic AVM: liver shunting → high-output heart failure; "
            "  ANNUAL CT CHEST MANDATORY in SMAD4 carriers: pulmonary AVM screening; "
            "  MRI brain: cerebral AVM baseline; repeat every 5yr; "
            "  SMAD4-HHT: more severe AVMs than ENG/ACVRL1 HHT; "
            "  Bevacizumab (anti-VEGF): systemic AVM management in severe HHT; "
            "SMAD4 GASTRIC CANCER: "
            "  PDAC concurrent risk (somatic SMAD4 in PDAC; germline SMAD4 confers pancreatic risk); "
            "  Annual EUS/MRI pancreas from age 40 (emerging data); "
            "SURVEILLANCE SMAD4: "
            "  Upper GI endoscopy from age 15yr (gastric juvenile polyposis); "
            "  Colonoscopy from age 15yr annually; "
            "  Annual CT chest (pulmonary AVM) MANDATORY from diagnosis; "
            "  MRI brain at diagnosis then every 5yr (cerebral AVM); "
            "  Annual clinical assessment for HHT (epistaxis, telangiectasiae, anaemia); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Juvenile Polyposis Syndrome (JPS) / JPS-HHT Overlap (SMAD4)",
        "gc_risk": "Gastric cancer 21-34% lifetime — HIGHEST JPS gene; gastric juvenile polyposis exclusively SMAD4 (not BMPR1A)",
        "pathognomonic": "Gastric juvenile polyps PATHOGNOMONIC JPS-SMAD4; HHT epistaxis + telangiectasiae PATHOGNOMONIC; pulmonary AVM → stroke risk",
        "key_avoid": "MISS PULMONARY AVM — annual CT chest MANDATORY in SMAD4 carriers; pulmonary AVM → paradoxical embolism → stroke",
        "surveillance": "Upper GI from 15yr; colonoscopy from 15yr; annual CT chest MANDATORY (AVM); MRI brain at dx then 5-yearly",
        "targeted_rx": "Bevacizumab (anti-VEGF) for systemic HHT-AVM; FOLFIRINOX for gastric/pancreatic cancer; polyp surveillance + polypectomy",
        "key_rule": "ANNUAL CT CHEST MANDATORY — SMAD4 HHT pulmonary AVM → paradoxical embolism → stroke; gastric cancer 21-34% lifetime",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-48kDa-AMPK-Kinase-PJS-"
            "Gastric-Cancer-29pct-Lifetime-"
            "Mucocutaneous-Melanin-Macules-PATHOGNOMONIC-"
            "SCTAT-Ovary-PATHOGNOMONIC-GI-Endoscopy-Age-8yr-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 encodes serine/threonine kinase 11 (LKB1): "
            "STRUCTURE: "
            "  433 aa / 48 kDa; master serine/threonine kinase; AMPK kinase family; "
            "  N-terminal regulatory domain; kinase domain (aa 49-309); C-terminal farnesylation signal; "
            "  STK11 activates AMPK → mTORC1 suppression → metabolic checkpoint; "
            "  STK11 LOF → AMPK not activated → mTOR constitutive → proliferation; "
            "PEUTZ-JEGHERS SYNDROME (PJS): "
            "  GASTRIC CANCER: 29% lifetime (PJS) — elevated via gastric hamartomatous polyps + diffuse risk; "
            "  Hamartomatous polyps of stomach, small bowel, colon: benign but with malignant risk; "
            "  Mucocutaneous melanin macules (lips, buccal mucosa, digits): PATHOGNOMONIC PJS onset 1-2yr; "
            "  GI endoscopy (upper + lower) from age 8yr MANDATORY; "
            "  Small bowel capsule endoscopy + MRI enterography from age 8yr; "
            "  SCTAT (sex cord tumour with annular tubules): ovary PATHOGNOMONIC PJS; "
            "  Cervical adenoma malignum (minimal deviation adenocarcinoma): PATHOGNOMONIC PJS; "
            "  Breast cancer: 45-50% lifetime (women); "
            "  Pancreatic cancer: 11-36% lifetime (see separate atlas); "
            "STK11 GASTRIC CANCER: "
            "  Gastric cancer 29% lifetime: hamartomatous polyps rarely transform + diffuse-type increased; "
            "  Annual gastroscopy from age 8yr; "
            "  Polypectomy when polyps >10-15mm; "
            "  No specific targeted therapy; standard chemotherapy for advanced disease; "
            "SURVEILLANCE STK11: "
            "  Annual gastroscopy + upper GI endoscopy from age 8yr; "
            "  Annual colonoscopy from age 8yr; "
            "  Small bowel MRI/capsule endoscopy from age 8yr; "
            "  Annual gynaecologic USS (SCTAT ovary, cervical adenoma malignum); "
            "  Annual breast MRI + mammography from age 25 (women); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Peutz-Jeghers Syndrome (PJS)",
        "gc_risk": "Gastric cancer 29% lifetime (PJS); hamartomatous polyps stomach; endoscopy from age 8yr MANDATORY",
        "pathognomonic": "Mucocutaneous melanin macules lips/buccal PATHOGNOMONIC (onset 1-2yr); SCTAT ovary PATHOGNOMONIC; cervical adenoma malignum",
        "key_avoid": "DELAY ENDOSCOPY — GI endoscopy (upper + lower + small bowel) from age 8yr MANDATORY; intussusception risk in childhood",
        "surveillance": "Annual gastroscopy + colonoscopy from 8yr; small bowel MRI/capsule from 8yr; annual gynaecologic USS; breast MRI from 25",
        "targeted_rx": "Polypectomy when >10-15mm; standard chemotherapy (FLOT/ECF) gastric; no STK11-specific targeted therapy",
        "key_rule": "GI ENDOSCOPY FROM AGE 8yr MANDATORY — PJS intussusception risk childhood; gastric cancer 29% lifetime; breast 45-50%",
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MutL-Homologue1-85kDa-MMR-Scaffold-Lynch-Syndrome-Type1-CMMRD-Biallelic-"
            "Gastric-6-13pct-Lifetime-6-8x-Elevated-MSI-H-PATHOGNOMONIC-"
            "Pembrolizumab-FDA2017-ANY-MSI-H-Aspirin-600mg-CAPP2-"
            "Muir-Torre-Sebaceous-PATHOGNOMONIC-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 encodes MutL homologue 1: "
            "STRUCTURE: "
            "  756 aa / 85 kDa; ATP-dependent endonuclease scaffold protein; "
            "  N-terminal ATPase domain (aa 1-335); C-terminal PMS2 dimerisation domain (aa 502-756); "
            "  MLH1 + PMS2 = MutLα (major repair complex); MLH1 LOF → MSI; "
            "  MMR loss → thousands of frameshifts → neoantigen generation; "
            "LYNCH SYNDROME TYPE 1 — GASTRIC CANCER SPECIFIC: "
            "  GASTRIC CANCER: 6-13% lifetime (MLH1 Lynch); 6-8x elevated vs general population; "
            "  Gastric cancer is the third most common Lynch cancer (after CRC and endometrial); "
            "  Histology: intestinal-type predominantly (vs CDH1 diffuse-type); "
            "  MSI-H PATHOGNOMONIC Lynch gastric tumours; "
            "  BRAF V600E absent in MSI-H Lynch gastric (DDx sporadic MLH1 methylation); "
            "  ANNUAL GASTROSCOPY from age 30-35yr MANDATORY in Lynch MLH1; "
            "  H. pylori eradication: MLH1 Lynch carriers H. pylori + → 2-3x additional gastric risk; "
            "  H. PYLORI ERADICATION MANDATORY at Lynch diagnosis; "
            "PEMBROLIZUMAB ELIGIBILITY — KEY RULE: "
            "  PEMBROLIZUMAB FDA 2017: first tumour-agnostic approval — any MSI-H/dMMR solid tumour; "
            "  Lynch gastric cancer: MSI-H → pembrolizumab eligible; "
            "  Dostarlimab FDA 2021: alternative dMMR approved; "
            "  ASPIRIN 600mg daily: CAPP2 trial — reduces Lynch cancer risk ~50% at 10yr; "
            "SURVEILLANCE MLH1: "
            "  Colonoscopy every 1-2yr from age 25yr; "
            "  Annual gastroscopy from age 30-35yr; "
            "  H. pylori test + treat at Lynch diagnosis; "
            "  Annual endometrial biopsy from age 35yr (women); "
            "  Aspirin 600mg daily CAPP2 (after excluding existing polyps); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Lynch Syndrome type 1 / CMMRD (biallelic)",
        "gc_risk": "Gastric cancer 6-13% lifetime Lynch; 6-8x elevated; intestinal-type MSI-H; H. pylori eradication MANDATORY at diagnosis",
        "pathognomonic": "MSI-H PATHOGNOMONIC Lynch gastric; BRAF V600E absent (DDx sporadic MLH1 methylation); Muir-Torre sebaceous PATHOGNOMONIC",
        "key_avoid": "H. PYLORI UNTREATED — MLH1 + H. pylori → 2-3x additional gastric risk; eradicate MANDATORY at Lynch diagnosis",
        "surveillance": "Annual gastroscopy from 30-35yr; H. pylori test+treat; colonoscopy 1-2yr from 25yr; aspirin 600mg CAPP2",
        "targeted_rx": "Pembrolizumab FDA2017 any MSI-H (Lynch gastric eligible); dostarlimab; aspirin 600mg CAPP2; FLOT/ECF standard chemo",
        "key_rule": "H. PYLORI ERADICATION MANDATORY AT LYNCH DIAGNOSIS — additional 2-3x gastric risk; pembrolizumab FDA any MSI-H gastric",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-"
            "Gastric-Undifferentiated-Diffuse-LFS-Elevated-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-"
            "APR-246-Eprenetapopt-p53-Reactivator-MDM2i-Investigational-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; homotetrameric transcription factor; the genome guardian; "
            "  N-terminal transactivation domain (aa 1-42): MDM2 binding; "
            "  DNA-binding domain (aa 94-292): hotspot mutations R175H/G245S/R248W/R248Q/R273H/R273C/R282W; "
            "  Tetramerisation domain (aa 323-356): required for functional tetramer; "
            "  TP53 activates CDKN1A (p21) → G1/S arrest; PUMA/NOXA → apoptosis; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  Near 100% penetrance by age 70; ~20% de novo germline; "
            "  Core cancers: soft tissue sarcoma 50-60%; premenopausal breast <45yr 30%; "
            "  Brain (glioma, CPC) 15%; osteosarcoma; adrenocortical carcinoma child PATHOGNOMONIC; "
            "TP53 GASTRIC CANCER: "
            "  Gastric cancer elevated in LFS (particularly undifferentiated/diffuse-type); "
            "  TP53 somatic loss: present in >50% of sporadic gastric cancers (Lauren intestinal + diffuse); "
            "  Germline TP53 + gastric: EBV-positive gastric cancer reported in LFS families; "
            "  R337H Brazilian founder variant: moderately elevated gastric risk alongside adrenal and pancreatic; "
            "  Annual gastroscopy from age 35yr (LFS with gastric family history); "
            "AVOID RADIATION ABSOLUTELY: "
            "  TP53 germline LOF → impaired G1/S checkpoint → radiation → secondary malignancy in field; "
            "  PREFER SURGERY over RT consolidation for gastric or any cancer in LFS; "
            "  MRI-based surveillance (avoid CT radiation where possible); "
            "EMERGING THERAPIES: "
            "  APR-246 / eprenetapopt: p53 reactivator; "
            "  MDM2 inhibitors (milademetan, idasanutlin): investigational; "
            "SURVEILLANCE LFS: "
            "  WBMRI Toronto protocol annually; brain MRI annually; "
            "  Annual breast MRI from age 20-25yr (women); "
            "  Annual gastroscopy from age 35yr (LFS + gastric family history); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Li-Fraumeni Syndrome (LFS)",
        "gc_risk": "Gastric cancer elevated in LFS (undifferentiated/diffuse-type); R337H Brazilian founder → pancreatic + adrenal + gastric",
        "pathognomonic": "ACC child PATHOGNOMONIC LFS; CPC child PATHOGNOMONIC; WBMRI Toronto protocol detects early LFS-spectrum cancers",
        "key_avoid": "RADIATION — AVOID RADIATION ABSOLUTELY in germline TP53; use MRI not CT for surveillance; surgery preferred over RT",
        "surveillance": "WBMRI Toronto annually; annual brain MRI; annual breast MRI from 20-25; annual gastroscopy from 35 (family history)",
        "targeted_rx": "Surgery preferred over RT; APR-246/eprenetapopt p53 reactivator investigational; MDM2i investigational; FLOT standard gastric",
        "key_rule": "AVOID RADIATION ABSOLUTELY — TP53 germline + radiation → secondary sarcoma/carcinoma; gastroscopy from 35 (LFS + GC history)",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-HBOC-FA-D1-"
            "Gastric-Cancer-2-3x-Elevated-Monoallelic-"
            "Platinum-Sensitive-HRD-Olaparib-FDA2019-POLO-"
            "Annual-Gastroscopy-Age-50-Monoallelic-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes breast cancer susceptibility protein 2: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; nuclear scaffold for homologous recombination (HR); "
            "  8 BRC repeats (aa 1002-2082): RAD51 binding; BRCA2 loads RAD51 onto ssDNA; "
            "  OB folds (aa 2402-3186): ssDNA binding; C-terminal RAD51 binding motif + NLS; "
            "  BRCA2 LOF → RAD51 fails to load → HR repair failure → NHEJ errors → chromosomal instability; "
            "  PALB2 binding domain (aa 10-40): anchors BRCA2 to chromatin at DSB sites; "
            "HBOC (BRCA2 monoallelic — germline LOF) GASTRIC: "
            "  Gastric cancer: 2-3x elevated lifetime risk (monoallelic BRCA2); "
            "  Gastric cancer risk in BRCA2: approximately 3-5% lifetime (vs 1.5% general); "
            "  H. PYLORI ERADICATION: monoallelic BRCA2 + H. pylori → compounded gastric risk; "
            "  Annual gastroscopy from age 50 in BRCA2 carriers with gastric family history; "
            "  Intestinal-type gastric cancer (NOT diffuse-type as in CDH1); "
            "BREAST/OVARIAN DOMINANT RISK: "
            "  Breast cancer (female): 69-85% lifetime; annual breast MRI + mammography from 30; "
            "  Ovarian cancer: 18-27% lifetime; RRSO (risk-reducing salpingo-oophorectomy) age 40-45; "
            "  Pancreatic cancer: 5-7% lifetime (see separate atlas); "
            "BRCA2 GASTRIC CANCER TREATMENT: "
            "  Platinum-based (HRD sensitivity: gemcitabine + cisplatin or FLOT + cisplatin); "
            "  OLAPARIB: FDA 2019 POLO trial (pancreatic BRCA2); gastric BRCA2 investigational; "
            "  TRASTUZUMAB: only if HER2+ gastric (BRCA2 does not predict HER2 status); "
            "  MSI testing: BRCA2 gastric can be MSI-H (test separately — pembrolizumab eligible if MSI-H); "
            "FANCONI ANAEMIA D1 (biallelic BRCA2): "
            "  Biallelic → FA-D1: childhood medulloblastoma, Wilms tumour, AML; "
            "  AVOID ALKYLATING AGENTS in biallelic FA-D1; cisplatin preferred; "
            "SURVEILLANCE BRCA2: "
            "  Annual gastroscopy from age 50 if gastric family history; "
            "  Annual breast MRI + mammography from age 30 (women); "
            "  Annual prostate PSA from age 40 (men); "
            "  Annual pancreatic MRI/EUS from age 50; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "HBOC (Hereditary Breast–Ovarian Cancer) / Fanconi Anaemia D1 (biallelic)",
        "gc_risk": "Gastric cancer 2-3x elevated monoallelic; ~3-5% lifetime; intestinal-type; H. pylori eradication recommended",
        "pathognomonic": "FA-D1 biallelic: childhood medulloblastoma + Wilms + AML PATHOGNOMONIC; BRCA2 monoallelic gastric intestinal-type",
        "key_avoid": "ALKYLATING AGENTS in biallelic FA-D1; miss H. pylori eradication — compounds 2-3x gastric risk in BRCA2 monoallelic",
        "surveillance": "Gastroscopy from 50 (if GC family history); annual breast MRI from 30; annual prostate PSA from 40; pancreatic MRI/EUS from 50",
        "targeted_rx": "Platinum-based (HRD sensitivity); olaparib FDA2019 (pancreatic BRCA2); FLOT ± cisplatin for BRCA2 gastric; HER2 test separately",
        "key_rule": "H. PYLORI ERADICATION — BRCA2 + H. pylori compounds 2-3x gastric risk; platinum-based preferred for HRD sensitivity",
    },
]

# Tumour types per gene
TUMOUR_TYPES_BY_GENE: dict = {
    "CDH1":   ["Hereditary Diffuse Gastric Cancer (HDGC) Signet Ring", "HDGC Pre-Prophylactic Gastrectomy Focus", "Lobular Breast Cancer (CDH1)", "HDGC After Delayed Gastrectomy", "HDGC + Lobular Breast Concurrent"],
    "CTNNA1": ["HDGC-Like Diffuse Gastric Carcinoma (CTNNA1)", "Diffuse Gastric Cancer Young Onset", "Signet Ring Cell Variant CTNNA1", "HDGC-Like Family Aggregation", "CTNNA1 + Lobular Breast Emerging"],
    "APC":    ["Gastric Fundic Gland Polyposis FAP", "Duodenal/Ampullary Adenocarcinoma FAP", "Gastric Adenocarcinoma AFAP", "Colorectal Cancer FAP (dominant)", "Desmoid Tumour Gardner (abdominal)"],
    "SMAD4":  ["Gastric Juvenile Polyposis Cancer SMAD4", "Colorectal Cancer JPS SMAD4", "Pancreatic Cancer SMAD4-LOF", "Gastric Cancer JPS-HHT Overlap", "AVM Haemorrhage Complication HHT"],
    "STK11":  ["Gastric Cancer PJS (hamartoma-associated)", "Small Bowel Cancer PJS", "Pancreatic Cancer PJS (11-36%)", "Breast Cancer PJS (45-50%)", "SCTAT Ovary PJS (PATHOGNOMONIC)"],
    "MLH1":   ["Gastric Cancer Lynch (intestinal-type MSI-H)", "Colorectal Cancer Lynch (dominant)", "Endometrial Cancer Lynch", "Gastric + Pembrolizumab Response MSI-H", "Muir-Torre Variant MLH1"],
    "TP53":   ["Gastric Cancer Undifferentiated LFS", "Adrenocortical Carcinoma Child (LFS)", "Osteosarcoma (LFS)", "Breast Cancer Premenopausal LFS", "Brain Tumour (CPC/Glioma) LFS"],
    "BRCA2":  ["Gastric Cancer Intestinal-Type BRCA2", "Breast Cancer HBOC BRCA2", "Ovarian Cancer HGSOC", "Pancreatic Cancer BRCA2 (3.5-10x)", "Gastric + Platinum Sensitive HRD"],
}

# Pathogenic variants per gene
VARIANTS_BY_GENE: dict = {
    "CDH1":   ["p.T340A (Portuguese/Newfoundland founder)", "p.W409Ter (protein truncating)", "p.E748Ter (truncating)", "IVS11+1G>T (splice donor)", "IVS6+1G>A (splice)", "c.2195G>A p.R732H (missense — functional assay required)"],
    "CTNNA1": ["p.Arg58Ter (truncating)", "c.1666del (frameshift)", "p.Gln849Ter (truncating)", "5q31.2 deletion (MLPA)", "p.Glu227Ter (truncating exon 7)", "c.1015_1016del (frameshift)"],
    "APC":    ["p.Glu1309Asp (codon 1309 — classic FAP)", "p.Gln1414Ter (MCR truncating)", "p.Arg876Ter (attenuated AFAP proximal)", "5q22.2 deletion (MLPA)", "c.3927_3931del (5-bp del — common)", "p.Ala1309Thr"],
    "SMAD4":  ["p.Arg445His (MH2 domain — haploinsufficiency)", "p.Arg361Ter (linker truncating)", "18q21.2 deletion (MLPA)", "c.868C>T p.Arg290Ter", "p.Tyr353Cys (MH2 missense)", "c.1141-1G>A (splice acceptor)"],
    "STK11":  ["p.Gln369Ter (kinase domain stop)", "p.Asp194Tyr (activation loop)", "STK11 exon 1-10 deletion (MLPA)", "c.1062+1G>A (splice)", "p.Phe354Leu (kinase missense)", "p.Glu170Lys"],
    "MLH1":   ["p.Val384Asp (missense pathogenic)", "p.Arg265Cys (MMR defect)", "3p22.2 deletion (MLPA)", "c.1852_1854del (in-frame del)", "c.676C>T p.Arg226Ter", "c.1852G>A splice promoter"],
    "TP53":   ["p.Arg175His (DBD hotspot GOF)", "p.Arg337His (Brazilian founder)", "p.Arg248Trp (hotspot)", "p.Gly245Ser (hotspot)", "p.Arg273His (hotspot)", "c.559+1G>T (splice IVS5)"],
    "BRCA2":  ["p.Trp3189Ter (protein truncating)", "p.Ser1982Arg (BRC repeat 7)", "p.Lys3326Ter (founder-like, attenuated)", "13q12.3 large deletion (MLPA)", "c.9382del (frameshift exon 23)", "p.Glu1308Ter"],
}

# Treatment protocols per gene
TREATMENT_PROTOCOLS_BY_GENE: dict = {
    "CDH1":   ["Prophylactic total gastrectomy MANDATORY age 20-30yr", "FLOT (fluorouracil + leucovorin + oxaliplatin + docetaxel) locally advanced", "Gastroscopy Cambridge protocol (28+ biopsies) until gastrectomy", "Annual breast MRI + mammography from age 30 (lobular breast)", "Genetic cascade testing first-degree relatives", "Perioperative chemotherapy if delayed presentation advanced disease"],
    "CTNNA1": ["Annual gastroscopy Cambridge protocol (28+ biopsies)", "IGCLC specialist centre referral MANDATORY", "FLOT for locally advanced CTNNA1 gastric", "Annual breast MRI from 30-35 (emerging lobular breast risk)", "No prophylactic gastrectomy consensus — shared decision specialist", "Platinum-based chemotherapy for advanced disease"],
    "APC":    ["Prophylactic colectomy age 20-25yr (curative CRC prevention)", "Upper GI surveillance (gastroscopy + duodenoscopy) from age 25yr annually", "FLOT for gastric cancer locally advanced FAP", "Sulindac/celecoxib (polyp reduction — not curative)", "Annual thyroid USS (cribriform morular PTC)", "Desmoid: celecoxib + tamoxifen; vinblastine + methotrexate (unresectable)"],
    "SMAD4":  ["Annual CT chest (pulmonary AVM) MANDATORY from diagnosis", "Upper GI endoscopy from age 15yr annually", "Colonoscopy from age 15yr annually", "Bevacizumab (anti-VEGF) systemic HHT-AVM severe", "FLOT for gastric cancer; MRI brain baseline + 5-yearly (cerebral AVM)", "Prophylactic embolisation pulmonary AVM at diagnosis if >2-3mm feeding artery"],
    "STK11":  ["GI endoscopy (upper + lower + small bowel) from age 8yr MANDATORY", "Annual gastroscopy polypectomy if >10-15mm", "FLOT for advanced gastric PJS", "Annual gynaecologic USS (SCTAT ovary; cervical adenoma malignum)", "Annual breast MRI + mammography from age 25 (women)", "Standard chemotherapy only for advanced disease (no targeted STK11 gastric)"],
    "MLH1":   ["H. pylori test + treat MANDATORY at Lynch diagnosis", "Pembrolizumab FDA 2017 any MSI-H/dMMR (Lynch gastric eligible)", "Dostarlimab FDA 2021 dMMR solid tumours", "Aspirin 600mg daily CAPP2 trial (~50% Lynch cancer reduction)", "Annual gastroscopy from age 30-35yr", "Colonoscopy every 1-2yr from age 25yr"],
    "TP53":   ["Surgery preferred over RT (AVOID RADIATION ABSOLUTELY)", "APR-246/eprenetapopt p53 reactivator investigational", "MDM2 inhibitors (milademetan, idasanutlin) investigational", "WBMRI Toronto protocol annually", "Annual gastroscopy from age 35yr (LFS + gastric family history)", "Annual breast MRI from 20-25 (women)"],
    "BRCA2":  ["H. pylori eradication MANDATORY (monoallelic BRCA2 + H. pylori 2-3x gastric risk)", "Platinum-based chemo (FLOT ± cisplatin; HRD sensitivity)", "Olaparib FDA 2019 (pancreatic BRCA2 primary; gastric investigational)", "Annual gastroscopy from age 50 (if gastric family history)", "Annual breast MRI + mammography from age 30 (women)", "Annual prostate PSA from age 40 (men)"],
}

# Surveillance protocols per gene
SURVEILLANCE_BY_GENE: dict = {
    "CDH1":   ["Annual gastroscopy Cambridge protocol (28+ targeted biopsies) until prophylactic gastrectomy", "Annual breast MRI + mammography from age 30 (women — lobular 42%)", "Prophylactic total gastrectomy age 20-30yr MANDATORY", "Genetic cascade testing first-degree relatives at diagnosis", "Annual gynaecologic review (cervical cancer rare CDH1)", "Family variant testing at 18yr or pre-gastrectomy planning"],
    "CTNNA1": ["Annual gastroscopy Cambridge protocol (28+ biopsies)", "Annual breast MRI from 30-35 (women — lobular breast emerging)", "IGCLC specialist centre referral MANDATORY", "Cascade genetic testing first-degree relatives", "Shared-decision prophylactic gastrectomy — no consensus yet", "Annual clinical review at specialist centre"],
    "APC":    ["Annual gastroscopy + duodenoscopy from age 25yr (side-viewing scope)", "Annual colonoscopy from age 12yr; prophylactic colectomy 20-25yr", "Annual thyroid USS (cribriform morular PTC surveillance)", "Desmoid screening: abdominal MRI if prior surgery or family desmoid history", "Annual upper GI review with Spigelman staging duodenal adenomas", "Genetic cascade testing first-degree relatives"],
    "SMAD4":  ["Annual CT chest MANDATORY from diagnosis (pulmonary AVM)", "MRI brain at diagnosis then every 5yr (cerebral AVM)", "Upper GI endoscopy from age 15yr annually", "Colonoscopy from age 15yr annually", "Annual clinical assessment (epistaxis, telangiectasiae, anaemia)", "Annual gastroscopy from age 15yr (gastric juvenile polyposis)"],
    "STK11":  ["Annual gastroscopy + upper GI endoscopy from age 8yr", "Annual colonoscopy from age 8yr", "Small bowel MRI/capsule endoscopy from age 8yr", "Annual gynaecologic USS (SCTAT ovary, cervical adenoma malignum)", "Annual breast MRI + mammography from age 25 (women)", "Annual testicular exam (Sertoli cell tumour risk males)"],
    "MLH1":   ["H. pylori test + treat at Lynch diagnosis", "Annual gastroscopy from age 30-35yr", "Colonoscopy every 1-2yr from age 25yr", "Annual endometrial biopsy from age 35yr (women)", "Aspirin 600mg daily CAPP2 (after excluding existing polyps)", "Annual urinary cytology from age 30-35yr (urothelial Lynch)"],
    "TP53":   ["WBMRI Toronto protocol annually", "Annual brain MRI (glioma/CPC surveillance)", "Annual breast MRI from age 20-25yr (women)", "Annual gastroscopy from age 35yr (LFS + gastric family history)", "Annual thyroid USS", "Annual abdominopelvic USS every 3-4 months <18yr (adrenocortical)"],
    "BRCA2":  ["Annual gastroscopy from age 50 (if gastric family history)", "Annual breast MRI + mammography from age 30 (women)", "Annual prostate PSA from age 40 (men)", "Annual ovarian USS + CA-125 (until RRSO age 40-45 women)", "Annual pancreatic MRI/EUS from age 50", "H. pylori test + treat at BRCA2 diagnosis"],
}

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]


def _make_patient(gene_idx: int, patient_idx: int) -> dict:
    seed = SEED_BASE + gene_idx + patient_idx * len(ATLAS_GENES)
    rng = random.Random(seed)
    gene = _GENE_LIST[gene_idx]
    gene_info = ATLAS_GENES[gene_idx]
    tumour_types = TUMOUR_TYPES_BY_GENE[gene]
    variants = VARIANTS_BY_GENE[gene]
    treatments = TREATMENT_PROTOCOLS_BY_GENE[gene]

    age_at_dx = rng.randint(24, 74)
    tumour_type = rng.choice(tumour_types)
    variant = rng.choice(variants)
    treatment = rng.choice(treatments)
    cr = rng.random() < 0.51
    radiation = rng.random() < (0.03 if gene in ("TP53", "CDH1") else 0.19)
    relapse = rng.random() < 0.40 if cr else rng.random() < 0.63

    return {
        "patient_id": f"HGCP-{gene}-{patient_idx:03d}",
        "gene": gene,
        "syndrome": gene_info["syndrome"],
        "age_at_dx": age_at_dx,
        "tumour_type": tumour_type,
        "variant": variant,
        "treatment": treatment,
        "cr": cr,
        "radiation": radiation,
        "relapse": relapse,
    }


def _generate_cohort() -> list:
    patients = []
    for gi in range(len(ATLAS_GENES)):
        for pi in range(40):
            patients.append(_make_patient(gi, pi))
    return patients


def generate_overview() -> dict:
    cohort = _generate_cohort()
    n = len(cohort)
    cr_n = sum(1 for p in cohort if p["cr"])
    rad_n = sum(1 for p in cohort if p["radiation"])
    relapse_n = sum(1 for p in cohort if p["relapse"])
    mean_age = round(sum(p["age_at_dx"] for p in cohort) / n, 1)

    gene_summary = {}
    for g in _GENE_LIST:
        pts = [p for p in cohort if p["gene"] == g]
        gene_summary[g] = {
            "n": len(pts),
            "cr_pct": round(100 * sum(1 for p in pts if p["cr"]) / len(pts), 1),
            "radiation_pct": round(100 * sum(1 for p in pts if p["radiation"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
        }

    return {
        "atlas": "Hereditary-Gastric-Cancer-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Gastric Cancer Predisposition Reference — CDH1-CTNNA1-APC-SMAD4-STK11-MLH1-TP53-BRCA2",
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "total_patients": n,
        "cr_pct": round(100 * cr_n / n, 1),
        "radiation_pct": round(100 * rad_n / n, 1),
        "relapse_pct": round(100 * relapse_n / n, 1),
        "mean_age_at_dx": mean_age,
        "genes": _GENE_LIST,
        "gene_summary": gene_summary,
        "key_rules": [
            "PROPHYLACTIC TOTAL GASTRECTOMY MANDATORY age 20-30yr (CDH1) — gastroscopy UNRELIABLE; 67-83% lifetime risk",
            "ANNUAL CT CHEST MANDATORY from diagnosis (SMAD4) — pulmonary AVM → paradoxical embolism → stroke",
            "H. PYLORI ERADICATION MANDATORY at Lynch diagnosis (MLH1) — 2-3x additional gastric risk",
            "UPPER GI SURVEILLANCE FROM AGE 25 MANDATORY (APC) — duodenal cancer 4-10% lifetime FAP",
            "PEMBROLIZUMAB FDA 2017 ANY MSI-H (MLH1) — test MSI on every gastric tumour Lynch",
            "GI ENDOSCOPY FROM AGE 8yr MANDATORY (STK11) — PJS intussusception; gastric 29% lifetime",
            "AVOID RADIATION ABSOLUTELY (TP53) — secondary malignancy; WBMRI Toronto annually",
            "H. PYLORI ERADICATION (BRCA2) — BRCA2 + H. pylori compounds 2-3x gastric risk; platinum HRD sensitive",
        ],
    }


def generate_breakdown() -> dict:
    cohort = _generate_cohort()
    breakdown = {}
    for gi, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        pts = [p for p in cohort if p["gene"] == gene]
        tumour_counts: dict = {}
        for p in pts:
            tumour_counts[p["tumour_type"]] = tumour_counts.get(p["tumour_type"], 0) + 1
        top_tumours = sorted(tumour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        variant_counts: dict = {}
        for p in pts:
            variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1
        top_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        breakdown[gene] = {
            "gene": gene,
            "protein": gene_info["protein"],
            "locus": gene_info["locus"],
            "syndrome": gene_info["syndrome"],
            "inheritance": gene_info["inheritance"],
            "gc_risk": gene_info["gc_risk"],
            "pathognomonic": gene_info["pathognomonic"],
            "key_avoid": gene_info["key_avoid"],
            "key_rule": gene_info["key_rule"],
            "surveillance": gene_info["surveillance"],
            "targeted_rx": gene_info["targeted_rx"],
            "n_patients": len(pts),
            "cr_pct": round(100 * sum(1 for p in pts if p["cr"]) / len(pts), 1),
            "radiation_pct": round(100 * sum(1 for p in pts if p["radiation"]) / len(pts), 1),
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
        "atlas": "Hereditary-Gastric-Cancer-Predisposition-Atlas",
        "definitions": {
            "hdgc_cdh1": (
                "HDGC (Hereditary Diffuse Gastric Cancer): CDH1 germline LOF; "
                "Gastric cancer 67-83% lifetime — HIGHEST single gene; always diffuse-type signet ring; "
                "PROPHYLACTIC TOTAL GASTRECTOMY MANDATORY age 20-30yr — gastroscopy Cambridge UNRELIABLE alone; "
                "Signet ring foci found in 50-70% at prophylactic gastrectomy; "
                "Lobular breast cancer 42% lifetime concurrent (annual breast MRI from 30 women). "
                "CDH1 p.T340A: Portuguese/Newfoundland founder."
            ),
            "hdgc_like_ctnna1": (
                "HDGC-like (CTNNA1-associated): CTNNA1 germline LOF — alpha-E-catenin; "
                "Diffuse gastric cancer moderate elevated (est. 30-50% lifetime); second HDGC gene after CDH1; "
                "Annual gastroscopy Cambridge protocol (28+ biopsies); "
                "No prophylactic gastrectomy consensus yet — IGCLC specialist centre referral MANDATORY; "
                "Lobular breast cancer risk emerging. Alpha-catenin IHC loss (vs E-cadherin loss in CDH1)."
            ),
            "fap_apc": (
                "FAP (Familial Adenomatous Polyposis): APC germline LOF; "
                "Fundic gland polyposis PATHOGNOMONIC (multiple FGPs in young patient); "
                "Gastric cancer 0.5-2% (classical FAP); attenuated AFAP 5-10%; "
                "Duodenal/ampullary cancer 4-10% lifetime — upper GI surveillance MANDATORY from 25yr; "
                "CRC 100% lifetime without colectomy; prophylactic colectomy 20-25yr. "
                "Cribriform morular PTC PATHOGNOMONIC FAP. CHRPE bilateral >4 PATHOGNOMONIC."
            ),
            "jps_hht_smad4": (
                "JPS-HHT (Juvenile Polyposis Syndrome + Hereditary Haemorrhagic Telangiectasia overlap): SMAD4 germline LOF; "
                "Gastric juvenile polyposis PATHOGNOMONIC; gastric cancer 21-34% lifetime — HIGHEST JPS gene; "
                "HHT overlap SMAD4-specific (NOT BMPR1A): epistaxis + telangiectasiae + AVM PATHOGNOMONIC; "
                "ANNUAL CT CHEST MANDATORY — pulmonary AVM → paradoxical embolism → stroke; "
                "Cerebral AVM: MRI brain at diagnosis then every 5yr; bevacizumab for systemic HHT."
            ),
            "pjs_stk11": (
                "Peutz-Jeghers Syndrome (PJS): STK11 germline LOF; "
                "Gastric cancer 29% lifetime; hamartomatous polyps of stomach/small bowel/colon; "
                "GI endoscopy (upper + lower + small bowel) from age 8yr MANDATORY; "
                "Mucocutaneous melanin macules lips/buccal PATHOGNOMONIC (onset 1-2yr); "
                "SCTAT ovary PATHOGNOMONIC; breast 45-50% lifetime; pancreatic 11-36% lifetime."
            ),
            "lynch_mlh1": (
                "Lynch Syndrome type 1 (MLH1): MLH1 germline LOF → MSI-H/dMMR tumours; "
                "Gastric cancer 6-13% lifetime (6-8x elevated); intestinal-type MSI-H; "
                "H. PYLORI ERADICATION MANDATORY at Lynch diagnosis — 2-3x additional gastric risk; "
                "PEMBROLIZUMAB FDA 2017 — first tumour-agnostic approval: any MSI-H/dMMR solid tumour; "
                "ASPIRIN 600mg daily CAPP2 reduces Lynch cancer risk ~50% at 10yr; "
                "Annual gastroscopy from age 30-35yr MANDATORY."
            ),
            "lfs_tp53": (
                "Li-Fraumeni Syndrome (LFS): TP53 germline LOF; "
                "Gastric cancer elevated in LFS (undifferentiated diffuse-type); "
                "AVOID RADIATION ABSOLUTELY — secondary sarcoma/carcinoma in RT field; use MRI not CT; "
                "WBMRI Toronto protocol annually; ACC child PATHOGNOMONIC; CPC child PATHOGNOMONIC; "
                "APR-246/eprenetapopt p53 reactivator investigational; MDM2i investigational."
            ),
            "hboc_brca2": (
                "HBOC BRCA2 (Hereditary Breast–Ovarian Cancer): BRCA2 germline LOF; "
                "Gastric cancer 2-3x elevated monoallelic (~3-5% lifetime); intestinal-type; "
                "H. PYLORI ERADICATION — BRCA2 + H. pylori compounds gastric risk; "
                "Platinum-based chemo (HRD sensitivity); olaparib FDA 2019 POLO (pancreatic primary); "
                "Annual gastroscopy from 50 (gastric family history); breast MRI from 30; prostate PSA from 40 (men)."
            ),
            "cascade_testing": (
                "CASCADE TESTING — Hereditary Gastric Cancer: "
                "1. CDH1: prophylactic gastrectomy 20-30yr MANDATORY; Cambridge gastroscopy until surgery; lobular breast 42%; "
                "2. CTNNA1: annual gastroscopy Cambridge; IGCLC specialist centre; no gastrectomy consensus yet; "
                "3. APC: upper GI from 25yr; colectomy 20-25yr; duodenal Spigelman staging; "
                "4. SMAD4: annual CT chest AVM MANDATORY; upper GI from 15yr; MRI brain at dx; "
                "5. STK11: GI endoscopy from 8yr MANDATORY; small bowel MRI from 8yr; SCTAT ovary; "
                "6. MLH1: H. pylori eradicate; pembrolizumab MSI-H; aspirin 600mg CAPP2; gastroscopy 30-35yr; "
                "7. TP53: WBMRI Toronto annually; avoid radiation; gastroscopy 35yr if GC history; "
                "8. BRCA2: H. pylori eradicate; platinum HRD; gastroscopy 50yr (GC history); breast MRI from 30."
            ),
        },
        "key_clinical_rules": [
            {
                "rule": "PROPHYLACTIC TOTAL GASTRECTOMY MANDATORY age 20-30yr (CDH1)",
                "gene": "CDH1",
                "rationale": "CDH1 germline LOF confers 67-83% lifetime diffuse gastric cancer risk; signet ring foci found in 50-70% at prophylactic gastrectomy at age 20-30yr; annual gastroscopy Cambridge protocol is unreliable as primary surveillance — misses multifocal signet ring foci",
                "consequence": "Delay beyond 30yr → infiltrative carcinoma detection in 15-30% at time of (now therapeutic) gastrectomy; preventable mortality",
            },
            {
                "rule": "ANNUAL CT CHEST MANDATORY from diagnosis (SMAD4)",
                "gene": "SMAD4",
                "rationale": "SMAD4 germline LOF causes JPS-HHT overlap in 15-22% of carriers; pulmonary AVM → right-to-left shunting → paradoxical embolism → stroke or cerebral abscess; AVM may be silent until catastrophic event; annual CT chest identifies PAVMs for embolisation",
                "consequence": "Undetected PAVM → paradoxical embolism → stroke; AVMs ≥2-3mm feeding artery need prophylactic embolisation before complication",
            },
            {
                "rule": "H. PYLORI ERADICATION MANDATORY at Lynch diagnosis (MLH1)",
                "gene": "MLH1",
                "rationale": "H. pylori co-infection in Lynch MLH1 carriers compounds gastric cancer risk 2-3x above baseline Lynch gastric risk; eradication reduces this additional risk; annual gastroscopy alone insufficient without eradication",
                "consequence": "Untreated H. pylori in Lynch MLH1 → substantially elevated gastric cancer risk; preventable inflammatory mucosal damage superimposed on MMR deficiency",
            },
            {
                "rule": "UPPER GI SURVEILLANCE FROM AGE 25 MANDATORY (APC)",
                "gene": "APC",
                "rationale": "FAP duodenal/ampullary adenomas progress to cancer in 4-10% lifetime; standard upper GI endoscopy must include side-viewing duodenoscope for ampulla; gastric fundic gland polyposis also requires gastroscopy; colectomy prevents CRC but not upper GI cancer",
                "consequence": "No upper GI surveillance in FAP → advanced duodenal/ampullary cancer; the second leading cancer death in post-colectomy FAP patients",
            },
            {
                "rule": "GI ENDOSCOPY FROM AGE 8yr MANDATORY (STK11)",
                "gene": "STK11",
                "rationale": "Peutz-Jeghers polyps cause childhood intussusception requiring emergency surgery; small bowel polyps and gastric polyps from early childhood; gastric cancer 29% lifetime; endoscopy from 8yr catches intussusception risk AND early malignant transformation",
                "consequence": "Delayed endoscopy → intussusception emergency in childhood; missed gastric polyp malignant transformation; preventable surgery and cancer",
            },
            {
                "rule": "PEMBROLIZUMAB FDA 2017 ANY MSI-H (MLH1)",
                "gene": "MLH1",
                "rationale": "First tumour-agnostic FDA approval: pembrolizumab works in any MSI-H/dMMR solid tumour regardless of histology; Lynch MLH1 gastric cancer is MSI-H and pembrolizumab-sensitive; MSI testing is MANDATORY on all gastric tumours in Lynch families",
                "consequence": "Not testing MSI in Lynch gastric cancer → missed pembrolizumab eligibility; major therapeutic opportunity foregone; preventable progression",
            },
            {
                "rule": "AVOID RADIATION ABSOLUTELY (TP53)",
                "gene": "TP53",
                "rationale": "TP53 germline LOF → impaired G1/S checkpoint → ionising radiation causes secondary sarcoma/carcinoma in RT field; secondary LFS-spectrum cancers documented in prior RT fields; prefer surgery-first approach for gastric and all cancers",
                "consequence": "Secondary malignancy in RT field within 5-10yr; accelerated carcinogenesis in LFS; gastric RT → secondary sarcoma in field",
            },
            {
                "rule": "H. PYLORI ERADICATION + PLATINUM-BASED CHEMO (BRCA2)",
                "gene": "BRCA2",
                "rationale": "BRCA2 monoallelic carriers: H. pylori co-infection compounds 2-3x gastric risk above baseline; eradication at BRCA2 diagnosis reduces this; BRCA2-null gastric cancer has HRD phenotype (platinum-sensitive); platinum-based regimens preferred over standard FLOT alone",
                "consequence": "Untreated H. pylori in BRCA2 → compounded 6-9x overall gastric risk; non-platinum chemo foregoes HRD sensitivity advantage",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}")
    print(f"Genes: {ov['genes']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"CR%: {ov['cr_pct']}")
    print(f"Mean age dx: {ov['mean_age_at_dx']}")
    print("\n=== BREAKDOWN KEYS ===")
    bd = generate_breakdown()
    for g in bd["genes"]:
        print(f"  {g}: n={bd['breakdown'][g]['n_patients']}, CR={bd['breakdown'][g]['cr_pct']}%")
    print("\n=== DEFINITIONS ===")
    df = generate_definitions()
    for k in list(df["definitions"].keys())[:3]:
        print(f"  {k}: {df['definitions'][k][:60]}...")
