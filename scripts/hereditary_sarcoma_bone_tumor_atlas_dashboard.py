#!/usr/bin/env python3
"""Hereditary-Sarcoma-Bone-Tumor-Predisposition-Atlas — Complete 8-Gene Hereditary Sarcoma & Bone Tumour Atlas
TP53   (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome (LFS);
         OSTEOSARCOMA most common LFS cancer in adolescents (28-30%);
         rhabdomyosarcoma, undifferentiated pleomorphic sarcoma, Ewing-like sarcoma;
         AVOID RADIATION ABSOLUTELY — radiation-field sarcoma in LFS survivors documented;
         WBMRI annually (Toronto protocol) for LFS surveillance;
         seed SEED_BASE+0) ·
RB1    (Retinoblastoma 1; 928aa; 13q14.2; AD LOF;
         Hereditary Retinoblastoma;
         BILATERAL retinoblastoma PATHOGNOMONIC for germline;
         secondary osteosarcoma 30-40% lifetime — LEADING cause of death post-cure;
         AVOID RADIATION ABSOLUTELY — radiation-field sarcoma (osteosarcoma) documented;
         TRILATERAL retinoblastoma (pineoblastoma) PATHOGNOMONIC;
         seed SEED_BASE+1) ·
DICER1 (Dicer 1 Ribonuclease; 1922aa; 14q32.13; AD LOF;
         DICER1 Tumour Predisposition Syndrome / FAPOL;
         Pleuropulmonary Blastoma (PPB) Type I/II/III PATHOGNOMONIC in childhood;
         embryonal rhabdomyosarcoma cervix, Sertoli-Leydig cell tumour ovary;
         thyroid nodular disease; cystic nephroma; Wilms; pineoblastoma;
         PPB Type I = cystic lung, favourable; Type III = solid, aggressive;
         seed SEED_BASE+2) ·
EXT1   (Exostosin Glycosyltransferase 1; 858aa; 8q24.11; AD LOF;
         Hereditary Multiple Exostoses Type 1 (HME1);
         MULTIPLE OSTEOCHONDROMAS PATHOGNOMONIC — epiphyses of long bones;
         secondary chondrosarcoma transformation 1-5% lifetime;
         EXT1 carries HIGHER malignant risk than EXT2;
         heparan sulfate proteoglycan (HSPG) biosynthesis pathway;
         seed SEED_BASE+3) ·
EXT2   (Exostosin Glycosyltransferase 2; 718aa; 11p11.2; AD LOF;
         Hereditary Multiple Exostoses Type 2 (HME2);
         MILDER than EXT1; fewer lesions; lower malignant risk ~1%;
         analogous HSPG biosynthesis defect — IHC EXT1 protein lost in EXT2 tumours too;
         same surveillance protocol as EXT1 but lower threshold for alarm;
         seed SEED_BASE+4) ·
RECQL4 (RecQ Like Helicase 4; 1208aa; 8q24.12; AR LOF;
         Rothmund-Thomson Syndrome Type 2 (RTS2);
         OSTEOSARCOMA 30% lifetime — highest osteosarcoma risk of any hereditary syndrome;
         poikiloderma PATHOGNOMONIC — congenital; skin atrophy + telangiectasia + hypo/hyperpigmentation;
         RECQL4 helicase required for DNA repair and replication;
         bone marrow failure; premature ageing features;
         seed SEED_BASE+5) ·
WRN    (Werner Syndrome RecQ Like Helicase; 1432aa; 8p12; AR LOF;
         Werner Syndrome (WS) / Adult Progeria;
         soft tissue sarcoma and osteosarcoma at early adult age (30s-40s);
         bilateral cataracts by age 30 PATHOGNOMONIC of WS;
         diabetes mellitus; hypogonadism; premature atherosclerosis;
         RecQ helicase — Werner protein unwinds aberrant DNA structures at stalled replication forks;
         seed SEED_BASE+6) ·
NF1    (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis Type 1 (NF1);
         MALIGNANT PERIPHERAL NERVE SHEATH TUMOUR (MPNST) 8-13% lifetime — most common hereditary sarcoma;
         MPNST in NF1 arise from plexiform neurofibromas — internal plexiform = high risk;
         FDG-PET MANDATORY for suspected MPNST (SUVmax >3.5 diagnostic);
         AVOID RADIATION — radiation-induced MPNST in field documented;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3118-3125)
"""
import random

SEED_BASE = 3118

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-Tumour-Suppressor-Transcription-Factor-Guardian-of-Genome-"
            "Li-Fraumeni-Syndrome-LFS-Osteosarcoma-RMS-STS-AVOID-RADIATION-ABSOLUTELY-"
            "WBMRI-Annual-Toronto-Protocol-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 17p13.1 TP53 encodes tumour protein p53, the 'guardian of the genome': "
            "STRUCTURE: N-terminal transactivation domain (TAD1/TAD2); proline-rich region; "
            "  central DNA-binding domain (DBD; most mutations here — hotspots R175H/R248W/R248Q/R273H/G245S); "
            "  C-terminal tetramerisation domain; regulatory domain; "
            "  Functions as homo-tetramer; binds p53 response elements → activates CDKN1A (p21), MDM2, PUMA, BAX; "
            "  TP53 LOF → loss of cell-cycle arrest at G1/S and G2/M → unrepaired DNA replication → cancer; "
            "LI-FRAUMENI SYNDROME (LFS) CANCER SPECTRUM: "
            "  CORE cancers: osteosarcoma (28-30% lifetime in paediatric LFS), "
            "    rhabdomyosarcoma, adrenocortical carcinoma (ACC — often first cancer if <5yr), "
            "    brain tumours (GBM/medulloblastoma/choroid plexus carcinoma); "
            "  ACC in children <5yr is DIAGNOSTIC of LFS — germline TP53 testing mandatory; "
            "  Soft tissue sarcomas: undifferentiated pleomorphic sarcoma, leiomyosarcoma; "
            "  Breast cancer (premenopausal); colorectal; lung; leukemia; "
            "SARCOMA BIOLOGY: "
            "  Osteosarcoma: TP53 somatic mutation in 30% sporadic; germline = LFS defines syndrome; "
            "  Peak sarcoma risk age 10-30yr in LFS (adolescent/young adult peak); "
            "  Radiation-induced sarcoma: LFS osteosarcoma arising in radiation field documented — AVOID ALL RADIATION; "
            "WBMRI TORONTO PROTOCOL: "
            "  Annual whole-body MRI from diagnosis (or age 18, whichever earlier); "
            "  Brain MRI (with gadolinium) annual; "
            "  Abdominal ultrasound 6-monthly (ACC screen, especially children); "
            "  Annual CBC, CMP; "
            "  Breast MRI annual from age 20 (women); avoid mammography until >30yr (radiation risk); "
            "GERMLINE TP53 TESTING INDICATIONS: "
            "  Sarcoma age <45yr; ACC at any age; choroid plexus carcinoma; multiple primary cancers; "
            "  Pathogenic somatic TP53 at age <30yr (may have missed germline); "
            "  Brazil R337H: common founder allele in southern Brazil — 1:300 prevalence; moderate penetrance; "
        ),
        "inheritance": (
            "AD LOF 17p13.1 — TP53. 1:5,000–20,000 live births. "
            "~30% de novo mutations. "
            "Dominant-negative: most missense variants titrate out wild-type p53 (heterozygous effect > simple LOF). "
            "Germline LOF (frameshift, stop, splice) → typical LFS; "
            "Gain-of-function (GOF) missense (R175H, R248W, R248Q, R273H, R249S, G245S) → "
            "  amplified LFS with earlier onset, more diverse tumour spectrum; "
            "CLONAL HAEMATOPOIESIS: TP53 germline → increased CH with TP53 somatic second hit — "
            "  mimic of therapy-related MDS; distinguish from true therapy-related MDS."
        ),
        "surveillance_key": "WBMRI annually (Toronto); brain MRI annual; abdominal USS 6-monthly; avoid ALL radiation; breast MRI from age 20 (women)",
        "pathognomonic": "adrenocortical carcinoma in child <5yr PATHOGNOMONIC for LFS; radiation-field sarcoma in LFS survivors",
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "Retinoblastoma-Protein-pRB-E2F-Repressor-Tumour-Suppressor-"
            "Hereditary-Retinoblastoma-BILATERAL-PATHOGNOMONIC-Secondary-Osteosarcoma-30-40pct-"
            "TRILATERAL-Pineoblastoma-PATHOGNOMONIC-AVOID-RADIATION-ABSOLUTELY-OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 13q14.2 RB1 encodes retinoblastoma protein (pRB): "
            "STRUCTURE: Pocket domain (A/B pockets) binds E2F transcription factors; "
            "  LXCXE motif-binding cleft binds viral oncoproteins (HPV E7, SV40 T-antigen); "
            "  Hypophosphorylated pRB (active) binds/represses E2F-target genes (CCNA, CCNE, DHFR); "
            "  CDK4/6-CyclinD phosphorylates pRB → releases E2F → G1/S progression; "
            "  RB1 LOF → constitutive E2F release → uncontrolled S-phase entry; "
            "HEREDITARY RETINOBLASTOMA: "
            "  BILATERAL retinoblastoma = germline until proven otherwise; "
            "  Unilateral multifocal = germline until proven otherwise; "
            "  Age of onset: bilateral onset 12-18 months; unilateral 24 months (mean); "
            "  White pupillary reflex (leukocoria) + strabismus = CARDINAL presentation; "
            "  Ophthalmological surveillance: monthly from birth for affected families; "
            "  Treatment: focal therapies (transpupillary thermotherapy, cryotherapy, laser); "
            "    Intra-arterial chemotherapy (IAC) for larger tumours; "
            "    Systemic carboplatin + vincristine + etoposide for bilateral advanced; "
            "    Enucleation: for unilateral advanced refractory disease; "
            "TRILATERAL RETINOBLASTOMA: "
            "  Intracranial primitive neuroectodermal tumour (PNET) — usually pineoblastoma; "
            "  PATHOGNOMONIC of bilateral germline retinoblastoma; "
            "  Rare (~3-5% bilateral); usually age 2-3yr; "
            "  Outcome: poor unless detected early on MRI surveillance; "
            "SECONDARY CANCERS (lifetime risk — MAJOR SURVIVORSHIP CONCERN): "
            "  OSTEOSARCOMA: 30-40% lifetime secondary cancer — leading cause of death in cured retinoblastoma; "
            "    In-field osteosarcoma (orbital RT field): extremely high dose-dependent risk; "
            "    Out-of-field osteosarcoma (13q LOH in mesenchymal cells): still elevated ~25% lifetime; "
            "  Other: soft tissue sarcoma, melanoma, bladder cancer, lung cancer, brain tumour; "
            "  AVOID RADIATION ABSOLUTELY: radiation-field osteosarcoma dramatically amplified; "
            "  Surveillance: annual CBC/CMP; MRI of common sarcoma sites from age 15; WBMRI; "
        ),
        "inheritance": (
            "AD LOF 13q14.2 — RB1. ~1:15,000-20,000 live births. "
            "~45% de novo mutations. "
            "Knudson two-hit model: hereditary = one germline hit + one somatic hit; "
            "Penetrance: ~90% for bilateral disease; lower for unilateral (modifier effects). "
            "Somatic mosaicism (~10%): mosaic patients have unilateral/unifocal disease with less systemic risk. "
            "Molecular testing: sequencing + large deletion analysis (MLPA) for del13q14; "
            "Full family screening: siblings tested at birth; parents screened."
        ),
        "surveillance_key": "bilateral retinoblastoma = germline; ophthalmology monthly from birth; WBMRI from age 15 for secondary sarcoma; AVOID RADIATION ABSOLUTELY",
        "pathognomonic": "bilateral retinoblastoma PATHOGNOMONIC germline; trilateral retinoblastoma (pineoblastoma) PATHOGNOMONIC; secondary osteosarcoma 30-40% lifetime",
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-RNase-III-MicroRNA-Processing-miRNA-Biogenesis-"
            "FAPOL-Pleuropulmonary-Blastoma-PATHOGNOMONIC-Childhood-"
            "Embryonal-RMS-Sertoli-Leydig-Cystic-Nephroma-Thyroid-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 14q32.13 DICER1 encodes a RNase III endoribonuclease: "
            "STRUCTURE: N-terminal helicase domain; Platform-PAZ domain (binds RNA 2-nt 3'-overhang); "
            "  Two RNase III catalytic domains (RNase IIIa and IIIb) — each cleaves one RNA strand; "
            "  Dsrm (double-stranded RNA-binding domain); "
            "  DICER1 cleaves pre-microRNA hairpins → mature miRNA ~21-23nt; "
            "  miRNAs repress translation/destabilise mRNAs; "
            "  DICER1 LOF → impaired miRNA biogenesis → de-repression of oncogenes; "
            "  Somatic second hit: RNase IIIb hotspot mutations (E1705, D1709, G1809 etc.) — missense in 'metal finger'; "
            "    These retain partial function but with biased miRNA processing; "
            "    Two-hit mechanism: germline LOF + somatic hotspot missense = complete miRNA dysregulation; "
            "PLEUROPULMONARY BLASTOMA (PPB) — PATHOGNOMONIC: "
            "  FAPOL (DICER1-associated tumour predisposition syndrome); "
            "  Type I (cystic): lung cysts in infancy; favourable, 85% 5-year OS if detected early; "
            "    SCREEN: chest CT at 3-monthly intervals in first 3yr of life for DICER1 families; "
            "  Type II (mixed): regress from cystic → mixed cystic-solid; "
            "  Type III (solid): aggressive; <50% 5-year OS; cisplatin-based chemotherapy; "
            "  HIGH SUSPICION: any child with multiple/recurrent pneumothorax or large pulmonary cyst; "
            "EMBRYONAL RHABDOMYOSARCOMA (ERMS) OF CERVIX: "
            "  Botryoid ERMS in cervix of DICER1 carriers — PATHOGNOMONIC of DICER1 in female adolescents; "
            "  Fertility-sparing surgery preferred when feasible; "
            "SERTOLI-LEYDIG CELL TUMOUR (SLCT) OVARY: "
            "  Moderately/poorly differentiated SLCT in young women → DICER1 test; "
            "  Virilisation/androgen excess; ovarian mass; "
            "  Somatic RNase IIIb hotspot in tumour + germline LOF; "
            "THYROID: "
            "  Multinodular goiter (70% of carriers); differentiated thyroid cancer in some; "
            "  Annual thyroid USS from adolescence; "
            "CYSTIC NEPHROMA / WILMS: "
            "  Cystic nephroma in DICER1 families → DICER1 germline; annual renal USS in children; "
            "OTHER TUMOURS: ciliary body medulloepithelioma; pineoblastoma; intestinal hamartoma; "
        ),
        "inheritance": (
            "AD LOF 14q32.13 — DICER1. ~1:10,000. "
            "Penetrance: ~13% for PPB in DICER1 families — incomplete penetrance means most carriers never develop PPB; "
            "BUT family history + early surveillance saves lives in high-risk families. "
            "De novo: ~50% PPB cases have de novo germline DICER1. "
            "Two-hit model: germline LOF + somatic RNase IIIb hotspot (unusual second hit mechanism — not LOH); "
            "Mosaic DICER1: some PPB families have mosaic germline — lower VAF → targeted deep sequencing."
        ),
        "surveillance_key": "chest CT 3-monthly in first 3yr for known DICER1 families; annual thyroid USS from adolescence; annual renal USS; pelvic MRI for ERMS if symptomatic",
        "pathognomonic": "pleuropulmonary blastoma Type I/II/III PATHOGNOMONIC; botryoid ERMS cervix PATHOGNOMONIC; DICER1 somatic RNase IIIb hotspot in tumour",
    },
    {
        "gene": "EXT1",
        "protein": (
            "EXT1 -- 8q24.11 Autosomal-Dominant-LOF -- 858aa -- "
            "Exostosin-Glycosyltransferase-1-Heparan-Sulfate-Biosynthesis-"
            "Hereditary-Multiple-Exostoses-Type1-HME1-Multiple-Osteochondromas-"
            "Chondrosarcoma-1-5pct-HIGHER-RISK-Than-EXT2-OMIM-608177"
        ),
        "locus": "8q24.11",
        "protein_size": (
            "858 aa / 8q24.11 EXT1 encodes exostosin glycosyltransferase 1: "
            "STRUCTURE: N-terminal region; C-terminal dual-function glycosyltransferase domain; "
            "  EXT1 forms heterodimer with EXT2 (EXT complex) in Golgi apparatus; "
            "  Complex polymerises heparan sulphate (HS) chains on core proteins; "
            "  HS = essential component of heparan sulphate proteoglycans (HSPGs) on cell surface; "
            "  HSPGs bind and modulate Hedgehog, FGF, Wnt, and BMP signalling gradients; "
            "  EXT1 LOF → truncated/absent HS → aberrant morphogen gradients → osteochondroma; "
            "HEREDITARY MULTIPLE EXOSTOSES (HME1) — TYPE 1: "
            "  Multiple osteochondromas arising from cartilage cap of epiphysis of long bones; "
            "  Distribution: distal femur, proximal tibia, proximal humerus (metaphysis > epiphysis); "
            "  Number: ranges from a few to >100; EXT1 > EXT2 for lesion burden; "
            "  Growth: osteochondromas grow during childhood → quiescent after skeletal maturity; "
            "  PAIN: mechanical impingement; nerve/vessel compression; bursitis; "
            "  DEFORMITY: forearm (radio-ulnar discordance); valgus knee; scoliosis; "
            "MALIGNANT TRANSFORMATION — CHONDROSARCOMA: "
            "  Lifetime risk: 1-5% for EXT1 (EXT1 > EXT2); "
            "  Warning signs: growth of pre-existing lesion after skeletal maturity; "
            "    NEW pain in previously painless lesion; soft-tissue mass over lesion; "
            "  Imaging: cartilage cap >2cm on MRI after skeletal maturity = suspicious; "
            "    FDG-PET for metabolic activity if concerned; "
            "  Histology: peripheral chondrosarcoma grade 1-3; resection only (chemo/RT ineffective grade 1); "
            "SURVEILLANCE: "
            "  Clinical review + targeted X-ray annually; "
            "  Baseline skeletal survey at diagnosis; "
            "  MRI for suspicious lesions (cartilage cap thickness); "
            "  Lesion growth after skeletal maturity → urgent surgical assessment; "
            "MOLECULAR: "
            "  EXT1 truncating > EXT1 missense; del/dup by MLPA common; "
            "  Somatic second hit (LOH 8q24.11) in osteochondroma cartilage cap cells; "
        ),
        "inheritance": (
            "AD LOF 8q24.11 — EXT1. 1:50,000. "
            "High penetrance (~95%); 10-15% de novo. "
            "EXT1 and EXT2 together account for >90% of HME families. "
            "Genotype-phenotype: EXT1 → more exostoses, higher malignant risk; EXT2 → fewer, lower risk. "
            "Expressivity variable even within family. "
            "Molecular: sequencing + MLPA; large deletions in ~10%."
        ),
        "surveillance_key": "annual clinical review; MRI if cartilage cap >2cm after skeletal maturity; FDG-PET if malignant transformation suspected; surgical resection for growing lesions post-maturity",
        "pathognomonic": "multiple osteochondromas at metaphyses of long bones PATHOGNOMONIC; cartilage cap >2cm after skeletal maturity = malignant transformation alarm",
    },
    {
        "gene": "EXT2",
        "protein": (
            "EXT2 -- 11p11.2 Autosomal-Dominant-LOF -- 718aa -- "
            "Exostosin-Glycosyltransferase-2-Heparan-Sulfate-Biosynthesis-"
            "Hereditary-Multiple-Exostoses-Type2-HME2-MILDER-Than-EXT1-"
            "Chondrosarcoma-1pct-Same-Surveillance-Protocol-OMIM-608210"
        ),
        "locus": "11p11.2",
        "protein_size": (
            "718 aa / 11p11.2 EXT2 encodes exostosin glycosyltransferase 2: "
            "STRUCTURE: N-terminal region; C-terminal catalytic domain; "
            "  EXT2 forms obligate heterodimer with EXT1 → co-dependent for heparan sulphate biosynthesis; "
            "  EXT2 alone has weak HS polymerisation activity; "
            "  EXT2 LOF → same HS deficiency mechanism as EXT1 → osteochondroma; "
            "HEREDITARY MULTIPLE EXOSTOSES TYPE 2 (HME2): "
            "  Clinically identical to HME1 — multiple osteochondromas; "
            "  MILDER: fewer lesions on average; less deformity; "
            "  Malignant transformation ~1% (vs 1-5% EXT1); "
            "  Osteochondroma distribution: same as EXT1 — distal femur, proximal tibia, humerus; "
            "IHC PARADOX: "
            "  In osteochondroma chondrocytes: BOTH EXT1 and EXT2 protein are lost — "
            "    even in EXT2-mutated tumours, EXT1 expression lost (complex interdependence); "
            "  This means IHC cannot distinguish EXT1 vs EXT2 lesion — molecular testing essential; "
            "DEFORMITY MANAGEMENT: "
            "  Forearm shortening/bowing: radioulnar discordance from distal ulnar exostosis; "
            "  Osteotomy + ulnar lengthening if functional impairment; "
            "  Valgus deformity knee: stapling or corrective osteotomy; "
            "  Spinal cord compression (rare): urgent laminectomy; "
            "EXOSTOSIS RESECTION INDICATIONS: "
            "  Symptomatic (pain, compression); cosmetically unacceptable; functional limitation; "
            "  Growing lesion after skeletal maturity; "
            "  DO NOT resect asymptomatic lesions prophylactically — recurrence risk low; "
            "CANCER RISK — SAME RULES AS EXT1: "
            "  Cartilage cap >2cm = alarm; pain after maturity = urgent; FDG-PET if concerned; "
            "  Annual clinical review; MRI for suspicious lesions; "
        ),
        "inheritance": (
            "AD LOF 11p11.2 — EXT2. 1:50,000 (shared prevalence with EXT1). "
            "~10-15% de novo. "
            "EXT2 cases phenotypically milder than EXT1 on average; "
            "Incomplete penetrance in rare families; "
            "MLPA for deletions; point mutation sequencing; EXT2 del common."
        ),
        "surveillance_key": "annual clinical review; same protocol as EXT1 but lower malignant risk; MRI if cartilage cap suspicious; surgical resection for symptomatic/growing lesions",
        "pathognomonic": "multiple osteochondromas; EXT2 milder than EXT1; IHC cannot distinguish EXT1 vs EXT2 — molecular testing essential",
    },
    {
        "gene": "RECQL4",
        "protein": (
            "RECQL4 -- 8q24.12 Autosomal-Recessive-LOF -- 1208aa -- "
            "RecQ-Like-Helicase-4-DNA-Repair-Replication-Initiation-"
            "Rothmund-Thomson-Syndrome-Type2-RTS2-OSTEOSARCOMA-30pct-"
            "Poikiloderma-PATHOGNOMONIC-Congenital-Bone-Marrow-Failure-OMIM-603780"
        ),
        "locus": "8q24.12",
        "protein_size": (
            "1208 aa / 8q24.12 RECQL4 encodes the RecQ4 DNA helicase: "
            "STRUCTURE: N-terminal Sld2/RecQL4 domain (DNA replication initiation function); "
            "  Central RecQ helicase domain (DEXH box); "
            "  C-terminal domain unique to RECQL4; "
            "  RECQL4 functions: "
            "    1. DNA replication initiation: N-terminal domain analogous to yeast Sld2 — fires origins of replication; "
            "    2. Base excision repair (BER) and double-strand break repair; "
            "    3. Mitochondrial DNA maintenance; "
            "  RECQL4 LOF → replication initiation failure + impaired DNA repair → genomic instability; "
            "ROTHMUND-THOMSON SYNDROME TYPE 2 (RTS2): "
            "  POIKILODERMA — PATHOGNOMONIC: "
            "    Onset 3-6 months of age; erythema → oedema → telangiectasia → atrophy → mottled pigmentation; "
            "    Distribution: face (butterfly pattern) → extremities; spares trunk initially; "
            "    CONGENITAL poikiloderma in infant = consider RTS2; "
            "  Sparse/absent scalp hair, eyelashes, eyebrows by childhood; "
            "  Short stature (~85% RTS2); "
            "  Skeletal abnormalities: radial ray hypoplasia; absent/hypoplastic thumb; "
            "  Cataracts (juvenile, bilateral); dental abnormalities; "
            "OSTEOSARCOMA RISK — HIGHEST OF ANY HEREDITARY SYNDROME: "
            "  30% lifetime risk — highest osteosarcoma predisposition of all hereditary syndromes; "
            "  Onset: median age 11yr (range 4-22yr) — childhood/adolescent peak; "
            "  Distribution: same as sporadic (distal femur > proximal tibia > proximal humerus); "
            "  SURVEILLANCE: annual MRI of lower limbs from age 5; FDG-PET if suspicious; "
            "  Treatment: standard osteosarcoma chemotherapy (MAP: methotrexate + doxorubicin + cisplatin); "
            "BONE MARROW FAILURE: "
            "  Cytopenia (mild to moderate); not classic Fanconi anaemia phenotype; "
            "  Annual CBC; transfusion support if needed; "
            "DIFFERENTIAL DIAGNOSIS: "
            "  RTS1 (no RECQL4; due to ANAMORSIN/C16orf57 mutation): poikiloderma WITHOUT osteosarcoma; "
            "  Baller-Gerold syndrome (craniosynostosis + radial aplasia): RECQL4 mutations subset; "
            "  RAPADILINO syndrome (Finland): radial aplasia + patella aplasia: RECQL4 truncating; "
        ),
        "inheritance": (
            "AR LOF 8q24.12 — RECQL4. "
            "Rare: <500 cases reported worldwide. "
            "Compound heterozygous or homozygous loss-of-function; "
            "Truncating variants → classic RTS2; missense → milder (Baller-Gerold / RAPADILINO spectrum); "
            "Founder mutations: RAPADILINO Finnish founder c.1390+2T>C (IVS11+2T>C); "
            "Carrier frequency low; prenatal/pre-implantation testing available for known families."
        ),
        "surveillance_key": "annual MRI lower limbs from age 5 for osteosarcoma; annual CBC; FDG-PET if suspicious; standard MAP chemotherapy for osteosarcoma",
        "pathognomonic": "congenital poikiloderma PATHOGNOMONIC (butterfly facial erythema in infancy); osteosarcoma 30% highest hereditary risk",
    },
    {
        "gene": "WRN",
        "protein": (
            "WRN -- 8p12 Autosomal-Recessive-LOF -- 1432aa -- "
            "Werner-Syndrome-Helicase-RecQ-Helicase-Exonuclease-"
            "Werner-Syndrome-Adult-Progeria-Sarcoma-Early-Adult-Onset-"
            "Bilateral-Cataracts-30yr-PATHOGNOMONIC-Diabetes-Atherosclerosis-OMIM-604611"
        ),
        "locus": "8p12",
        "protein_size": (
            "1432 aa / 8p12 WRN encodes the Werner syndrome RecQ helicase: "
            "STRUCTURE: N-terminal exonuclease domain (3'→5' exonuclease; unique among RecQ helicases); "
            "  DEXH helicase core domain (3'→5' DNA helicase + ATPase); "
            "  Central winged-helix domain; RQC (RecQ C-terminal) domain; HRDC (helicase RNase D C-terminal); "
            "  C-terminal nuclear localisation signal; "
            "  WRN functions: "
            "    1. Resolves G-quadruplexes and D-loops at stalled replication forks; "
            "    2. 3'→5' exonuclease activity processes DNA DSBs and aids NER; "
            "    3. Interacts with Ku70/80 (NHEJ), RPA, PCNA, FEN1; "
            "  WRN LOF → accumulation of stalled replication forks → genomic instability → premature ageing + cancer; "
            "WERNER SYNDROME (WS) — ADULT PROGERIA: "
            "  Hallmarks of premature ageing beginning in 2nd-3rd decade: "
            "  BILATERAL CATARACTS by age 30 — PATHOGNOMONIC (often first clinical sign); "
            "  Thin limbs with truncal obesity ('bird-like' habitus); "
            "  Grey hair + alopecia; short stature; "
            "  Prematurely aged skin; leg ulcers (often calcified); "
            "  Diabetes mellitus type 2 (hypo/insulin-dependent); "
            "  Hypogonadism; infertility; "
            "  Premature atherosclerosis; myocardial infarction / CVA in 40s-50s; "
            "CANCER RISK — SARCOMA PREDOMINANT: "
            "  Lifetime cancer risk ~10x general population; "
            "  Sarcomas predominate (vs common epithelial cancers in normal ageing): "
            "    Osteosarcoma; soft tissue sarcomas (leiomyosarcoma, fibrosarcoma, undifferentiated); "
            "    Thyroid carcinoma; meningioma; melanoma; hematological malignancies; "
            "  Early onset: sarcoma median age 40yr (vs 60yr in general population); "
            "MANAGEMENT: "
            "  No disease-modifying therapy proven; supportive/symptomatic; "
            "  Cataracts: surgical extraction; "
            "  Diabetes: standard antidiabetics; "
            "  Wound management: leg ulcers require specialist wound care; "
            "  Cancer surveillance: annual MRI body; skin cancer surveillance; thyroid USS; "
            "  Japanese registry: highest documented WS population (founder mutations common in Japan); "
            "WS FOUNDER MUTATIONS (Japan): "
            "  c.3139-1G>C (IVS25-1G>C) splice; p.Leu1074Pro; "
            "  Prevalence ~1:20,000–200,000 in Japan; higher than Western populations; "
        ),
        "inheritance": (
            "AR LOF 8p12 — WRN. Rare globally; enriched in Japan and Sardinia. "
            "Compound heterozygous or homozygous loss-of-function; "
            "Carrier frequency ~1:150 in Japan (founder mutations); "
            "Heterozygous carriers: no classic WS; possible mild cancer risk increase; "
            "WRN is one of five human RecQ helicases; BLM (Bloom), RecQL3 (no human disease), "
            "  RecQL1 (no human disease), RECQL4 (RTS), WRN (Werner) — all genome stability functions."
        ),
        "surveillance_key": "annual MRI body from age 30; cataract watch from age 25; annual CBC/CMP/lipids; skin cancer surveillance; thyroid USS; wound care for leg ulcers",
        "pathognomonic": "bilateral cataracts by age 30 PATHOGNOMONIC; sarcoma in 3rd-4th decade; bird-like habitus (thin limbs + truncal obesity); premature atherosclerosis",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-1-RAS-GAP-GTPase-Activating-Protein-"
            "Malignant-Peripheral-Nerve-Sheath-Tumour-MPNST-8-13pct-"
            "Most-Common-Hereditary-Sarcoma-FDG-PET-MANDATORY-AVOID-RADIATION-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 17q11.2 NF1 encodes neurofibromin-1, the largest RAS-GAP: "
            "STRUCTURE: Central GRD (GAP-related domain) catalyses RAS-GTP → RAS-GDP; "
            "  PH domain; IRA-like domain; Sec14-like domain; cysteine-serine rich region; "
            "  NF1 LOF → sustained RAS-GTP → hyperactivated MAPK/ERK and PI3K/Akt pathways; "
            "  Tumour formation requires second hit (LOH at 17q11.2 in tumour); "
            "NF1 AND MALIGNANT PERIPHERAL NERVE SHEATH TUMOUR (MPNST): "
            "  MPNST risk 8-13% lifetime in NF1 — MOST COMMON HEREDITARY SARCOMA; "
            "  MPNST arises from Schwann cells of plexiform neurofibromas (internal > cutaneous); "
            "  RISK FACTORS for MPNST in NF1: "
            "    Large plexiform neurofibroma (especially internal/paraspinal/mediastinal); "
            "    Prior diagnosis of plexiform; "
            "    Whole-gene NF1 deletion (microdeletion 17q11.2 — higher MPNST risk); "
            "    Prior radiation exposure in field (AVOID ALL RADIATION); "
            "    Associated CDKN2A deletion (somatic); "
            "FDG-PET FOR MPNST — MANDATORY: "
            "  Gold standard for detecting malignant transformation in plexiform neurofibromas; "
            "  SUVmax >3.5: high specificity for MPNST transformation; "
            "  Dual-time FDG-PET (1hr + 2hr): improves specificity; "
            "  MRI whole body: detects internal plexiform + monitors size change; "
            "  Annual full-body MRI for patients with known large internal plexiform; "
            "  MPNST warning signs: rapid growth; new/worsening pain; firmness in previously soft neurofibroma; "
            "  Any such change → FDG-PET within weeks (not months); "
            "MPNST TREATMENT: "
            "  Wide surgical excision with clear margins (R0) = primary treatment; "
            "  Adjuvant radiation: CONTROVERSIAL in NF1 (avoids radiation-induced secondary tumours); "
            "    If used: proton therapy preferred in young patients; "
            "  Chemotherapy: doxorubicin + ifosfamide (response rate ~20-30%); "
            "  MEK inhibitors (selumetinib, mirdametinib): trials for NF1 MPNST; "
            "  5-year OS: NF1-MPNST 20-40% (worse than sporadic MPNST due to diagnostic delay); "
            "NF1 BROAD CANCER SPECTRUM: "
            "  Paediatric: optic pathway glioma (15-20%); JMML (juvenile myelomonocytic leukaemia); "
            "  Adult: glioma; pheochromocytoma (2-5%); gastrointestinal stromal tumour (GIST, rare); "
            "  Rhabdomyosarcoma (embryonal): children with NF1; "
        ),
        "inheritance": (
            "AD LOF 17q11.2 — NF1. 1:3,000 live births (most common AD disorder). "
            "50% de novo mutations. "
            "NF1 gene = 350kb; one of the largest human genes; del/dup/complex rearrangements common (~5-10%). "
            "MLPA + sequencing required for comprehensive testing. "
            "Somatic mosaic NF1 (~10%): segmental NF1; reduced severity; can transmit full NF1 to offspring. "
            "Whole-gene deletion (microdeletion 17q11.2): ~5% NF1; associated with higher MPNST risk + "
            "  cognitive impairment + more tumours."
        ),
        "surveillance_key": "FDG-PET MANDATORY for suspected MPNST (SUVmax >3.5); annual full-body MRI for large internal plexiform; selumetinib for paediatric plexiform; AVOID RADIATION",
        "pathognomonic": "MPNST from plexiform neurofibroma; café-au-lait ≥6 + Lisch nodules; plexiform neurofibroma → rapid growth = MPNST alarm",
    },
]


def _gene_stats(seed: int, gene_config: dict) -> dict:
    """Generate per-gene statistics for one gene using a fixed seed."""
    rng = random.Random(seed)
    gene = gene_config["gene"]

    # Base rates vary by gene
    base = {
        "TP53":   {"lfs_cancer": 92, "sarcoma": 28, "breast": 32, "brain_tumour": 18, "acc": 12, "radiation_sarcoma": 8},
        "RB1":    {"bilateral_rb": 62, "secondary_osteosarcoma": 34, "trilateral": 4, "sarcoma_in_field": 25, "other_secondary": 15},
        "DICER1": {"ppb_type_i": 55, "ppb_type_ii_iii": 18, "thyroid_nodule": 72, "slct_ovary": 22, "erms_cervix": 8, "cystic_nephroma": 12},
        "EXT1":   {"multiple_exostoses": 100, "chondrosarcoma": 3, "forearm_deformity": 42, "height_loss": 28, "pain": 65},
        "EXT2":   {"multiple_exostoses": 100, "chondrosarcoma": 1, "forearm_deformity": 28, "height_loss": 18, "pain": 52},
        "RECQL4": {"poikiloderma": 98, "osteosarcoma": 30, "short_stature": 85, "cataracts": 22, "bone_marrow_failure": 18},
        "WRN":    {"bilateral_cataracts": 96, "sarcoma": 12, "diabetes": 72, "atherosclerosis": 55, "leg_ulcers": 38},
        "NF1":    {"cafe_au_lait": 98, "plexiform": 42, "mpnst": 10, "optic_glioma": 17, "jmml": 2, "fdg_pet_positive": 75},
    }.get(gene, {})

    n = 40
    age_mean = {
        "TP53": 16, "RB1": 2, "DICER1": 3, "EXT1": 8,
        "EXT2": 10, "RECQL4": 7, "WRN": 32, "NF1": 18,
    }.get(gene, 15)

    stats = {
        "gene": gene,
        "n": n,
        "seed": seed,
        "mean_age_diagnosis": round(age_mean + rng.gauss(0, 2.5), 1),
        "female_pct": round(rng.uniform(45, 65), 1),
    }

    for feature, base_rate in base.items():
        rate = max(0, min(100, base_rate + rng.gauss(0, 4)))
        stats[f"{feature}_pct"] = round(rate, 1)

    stats["genetic_testing_positive_pct"] = round(rng.uniform(88, 99), 1)
    stats["surveillance_adherent_pct"] = round(rng.uniform(65, 85), 1)
    stats["family_history_positive_pct"] = round(rng.uniform(40, 65), 1)
    stats["de_novo_pct"] = round(rng.uniform(35, 60), 1)
    return stats


def generate_overview() -> dict:
    """Overview data for Hereditary-Sarcoma-Bone-Tumor-Predisposition-Atlas."""
    return {
        "atlas":          "Hereditary-Sarcoma-Bone-Tumor-Predisposition-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Sarcoma and Bone Tumour Predisposition Atlas "
            "(TP53-RB1-DICER1-EXT1-EXT2-RECQL4-WRN-NF1)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "TP53": (
                "AD LOF 17p13.1 (p53; 393aa; Li-Fraumeni Syndrome; "
                "osteosarcoma 28-30% adolescent LFS cancers; ACC in child <5yr PATHOGNOMONIC; "
                "AVOID RADIATION ABSOLUTELY — radiation-field sarcoma documented; "
                "WBMRI annually Toronto protocol; 50% de novo; dominant-negative hotspot variants)"
            ),
            "RB1": (
                "AD LOF 13q14.2 (pRB; 928aa; Hereditary Retinoblastoma; "
                "BILATERAL retinoblastoma PATHOGNOMONIC germline; "
                "secondary osteosarcoma 30-40% lifetime — leading cause of death post-cure; "
                "trilateral retinoblastoma (pineoblastoma) PATHOGNOMONIC; "
                "AVOID RADIATION ABSOLUTELY — radiation-field osteosarcoma amplified)"
            ),
            "DICER1": (
                "AD LOF 14q32.13 (DICER1 helicase; 1922aa; FAPOL; "
                "Pleuropulmonary Blastoma PPB Type I/II/III PATHOGNOMONIC childhood; "
                "embryonal RMS cervix; Sertoli-Leydig cell tumour ovary; thyroid multinodular; "
                "somatic RNase IIIb hotspot as second hit — unusual mechanism)"
            ),
            "EXT1": (
                "AD LOF 8q24.11 (Exostosin-1; 858aa; HME Type 1; "
                "multiple osteochondromas PATHOGNOMONIC; chondrosarcoma 1-5% (HIGHER than EXT2); "
                "cartilage cap >2cm after skeletal maturity = malignant transformation alarm; "
                "more severe than EXT2 — more lesions, higher malignant risk)"
            ),
            "EXT2": (
                "AD LOF 11p11.2 (Exostosin-2; 718aa; HME Type 2; "
                "multiple osteochondromas; MILDER than EXT1; chondrosarcoma ~1%; "
                "same surveillance protocol; IHC cannot distinguish EXT1 vs EXT2 — use molecular; "
                "EXT1 + EXT2 together >90% of all HME families)"
            ),
            "RECQL4": (
                "AR LOF 8q24.12 (RECQL4 helicase; 1208aa; Rothmund-Thomson Syndrome Type 2; "
                "OSTEOSARCOMA 30% — HIGHEST hereditary osteosarcoma risk; "
                "congenital poikiloderma PATHOGNOMONIC (butterfly facial erythema in infancy); "
                "annual MRI lower limbs from age 5; rare globally)"
            ),
            "WRN": (
                "AR LOF 8p12 (Werner helicase; 1432aa; Werner Syndrome / Adult Progeria; "
                "BILATERAL CATARACTS by age 30 PATHOGNOMONIC; sarcoma dominant cancer in 40s; "
                "premature atherosclerosis; diabetes; leg ulcers; "
                "Japanese founder mutations common; heterozygotes unaffected)"
            ),
            "NF1": (
                "AD LOF 17q11.2 (Neurofibromin-1; 2839aa; NF Type 1; 1:3000 most common AD; "
                "MPNST 8-13% — MOST COMMON HEREDITARY SARCOMA — from plexiform neurofibromas; "
                "FDG-PET MANDATORY for suspected MPNST (SUVmax >3.5); "
                "AVOID RADIATION — radiation-induced MPNST in field documented; "
                "selumetinib FDA2020 for paediatric plexiform)"
            ),
        },
        "key_clinical_rules": [
            "TP53/LFS: AVOID ALL RADIATION — radiation-field sarcoma in LFS survivors is a known, preventable iatrogenic tragedy",
            "TP53/LFS: WBMRI annually (Toronto protocol) from diagnosis/age 18; adrenocortical carcinoma in child <5yr = germline TP53 test immediately",
            "RB1: BILATERAL retinoblastoma = germline RB1 until proven otherwise — test ALL bilateral cases regardless of family history",
            "RB1: AVOID RADIATION ABSOLUTELY — in-field osteosarcoma risk extremely high; IAC (intra-arterial chemotherapy) preferred over external beam",
            "RB1: secondary osteosarcoma surveillance — WBMRI + annual MRI from age 15 in all hereditary retinoblastoma survivors",
            "DICER1: PPB Type I (lung cysts in infant) → DICER1 germline test + chest CT 3-monthly in first 3yr for family members",
            "DICER1: botryoid ERMS of cervix in adolescent female = DICER1 germline until proven otherwise",
            "EXT1/EXT2: cartilage cap >2cm on MRI after skeletal maturity = alarm for chondrosarcoma transformation → urgent surgical referral",
            "EXT1/EXT2: osteochondroma growing after skeletal maturity + new pain = MALIGNANT TRANSFORMATION — do not observe without imaging",
            "RECQL4: congenital poikiloderma + skeletal abnormalities → test RECQL4; annual MRI lower limbs from age 5yr for osteosarcoma surveillance",
            "WRN: bilateral cataracts age <30yr in young adult + thin limbs + truncal obesity = Werner Syndrome — test WRN; sarcoma surveillance annual MRI from age 30",
            "NF1: FDG-PET MANDATORY for any plexiform neurofibroma showing rapid growth, new pain, or firmness change — SUVmax >3.5 is MPNST until proven otherwise",
            "NF1: AVOID RADIATION in NF1 — radiation-induced MPNST in radiation field documented; selumetinib for symptomatic inoperable plexiform (FDA2020)",
            "SARCOMA PANEL MINIMUM: TP53 + RB1 + NF1 for any adolescent/young adult sarcoma; add DICER1 for childhood sarcoma; add RECQL4 + WRN for poikiloderma/progeria",
        ],
        "gene_panel_note": (
            "Hereditary Sarcoma and Bone Tumour Predisposition panel (clinical 2024): "
            "HIGH-PENETRANCE SARCOMA PREDISPOSITION: "
            "  TP53 (LFS): osteosarcoma, RMS, STS, ACC — adolescent/young adult; "
            "  RB1: bilateral retinoblastoma → secondary osteosarcoma; "
            "  NF1: MPNST from plexiform neurofibromas — most common hereditary sarcoma; "
            "CHILDHOOD SARCOMA/TUMOUR PREDISPOSITION: "
            "  DICER1: PPB (lung), ERMS (cervix), SLCT (ovary), cystic nephroma; "
            "  RB1: retinoblastoma in first 2yr of life; "
            "BONE TUMOUR PREDISPOSITION (BENIGN WITH MALIGNANT RISK): "
            "  EXT1/EXT2: multiple osteochondromas → chondrosarcoma; "
            "  RECQL4: osteosarcoma 30% — highest hereditary osteosarcoma risk; "
            "PROGERIA-ASSOCIATED SARCOMA: "
            "  WRN: Werner syndrome — adult-onset sarcoma + premature ageing; "
            "  RECQL4: Rothmund-Thomson Type 2 — childhood osteosarcoma + poikiloderma; "
            "RADIATION AVOIDANCE MANDATE (CRITICAL): "
            "  TP53 (LFS): AVOID ALL RADIATION — radiation-field sarcoma documented; "
            "  RB1: AVOID ALL RADIATION — in-field osteosarcoma amplified; "
            "  NF1: AVOID RADIATION — radiation-induced MPNST in field documented; "
            "CLINICAL DECISION TREE: "
            "  Adolescent osteosarcoma → TP53 first; then RB1; then RECQL4; "
            "  Bilateral retinoblastoma → RB1 germline; secondary sarcoma surveillance; "
            "  Infant with lung cysts/pneumothorax → DICER1 (PPB Type I); "
            "  Multiple hard bony swellings metaphyses → EXT1/EXT2 (HME); "
            "  Congenital facial poikiloderma → RECQL4 (RTS2); "
            "  Young adult cataracts + thin limbs + sarcoma → WRN (Werner); "
            "  Plexiform neurofibroma + rapid growth → NF1 + FDG-PET (MPNST alarm); "
            "SURVEILLANCE TIERS: "
            "  All TP53/LFS: annual WBMRI Toronto protocol; "
            "  All RB1 hereditary: WBMRI from age 15 for secondary sarcoma; "
            "  All NF1: annual exam + FDG-PET for any plexiform change; "
            "  All RECQL4: annual MRI lower limbs from age 5"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Sarcoma-Bone-Tumor-Predisposition-Atlas."""
    genes_data = []
    for i, gene_cfg in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        stats = _gene_stats(seed, gene_cfg)
        stats["protein_summary"] = gene_cfg["protein"]
        stats["locus"] = gene_cfg["locus"]
        stats["inheritance"] = gene_cfg["inheritance"]
        stats["surveillance_key"] = gene_cfg["surveillance_key"]
        stats["pathognomonic"] = gene_cfg["pathognomonic"]
        genes_data.append(stats)
    return {
        "atlas":          "Hereditary-Sarcoma-Bone-Tumor-Predisposition-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "n_genes":        len(genes_data),
        "total_patients": 320,
        "genes":          genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Sarcoma-Bone-Tumor-Predisposition-Atlas."""
    definitions = [
        {
            "term": "TP53-Li-Fraumeni-Sarcoma-Surveillance-Protocol",
            "definition": (
                "TP53 / Li-Fraumeni Syndrome Sarcoma Management (Toronto Protocol 2024): "
                "SARCOMA SPECTRUM IN LFS: "
                "  Osteosarcoma: most common LFS cancer in adolescents (28-30% of LFS cancers in <20yr); "
                "  Rhabdomyosarcoma (embryonal): children 0-5yr; "
                "  Undifferentiated pleomorphic sarcoma (UPS): adults; "
                "  Leiomyosarcoma: uterine / retroperitoneal; "
                "  Other soft tissue sarcomas; "
                "AVOID RADIATION — ABSOLUTE RULE: "
                "  Radiation-field sarcoma in LFS survivors: documented; "
                "  LFS patients treated with radiotherapy for primary cancer → sarcoma in field within 5-15yr; "
                "  When radiation-free alternatives exist (surgery, proton therapy), use them; "
                "  If radiation unavoidable: discuss with patient/family; smaller fields; proton preferred; "
                "WBMRI TORONTO PROTOCOL (annual): "
                "  Sequence: T1 + STIR; whole-body coverage (vertex → toes); "
                "  Annual from time of diagnosis / age 18 (whichever earlier); "
                "  Detects: musculoskeletal tumours, adrenal masses, lymphoma, brain lesions; "
                "  Brain MRI (gadolinium): annually; "
                "  Abdominal MRI / USS: 6-monthly in children (ACC risk); "
                "  Breast MRI: annually from age 20 (women); NO mammography <30yr (radiation dose); "
                "  Annual CBC + CMP; "
                "OSTEOSARCOMA TREATMENT IN LFS: "
                "  Same as sporadic osteosarcoma: MAP (methotrexate + doxorubicin + cisplatin); "
                "  Surgery: wide resection ± limb salvage; "
                "  POST-TREATMENT SURVEILLANCE: "
                "    Continue WBMRI for surveillance of second primaries; "
                "    LFS patients have multiple successive primaries — treat each as potentially curable; "
                "GENETIC COUNSELLING: "
                "  Autosomal dominant: 50% recurrence to children; "
                "  De novo rate ~30%; "
                "  Genotyping: sequencing + del/dup; "
                "  If somatic TP53 identified at age <30yr without WBMRI finding: "
                "    Consider germline testing — somatic-germline distinction requires germline DNA; "
                "CLONAL HAEMATOPOIESIS CAVEAT: "
                "  TP53 germline LOF patients have higher CH rates; "
                "  CBC pancytopenia in LFS → bone marrow biopsy to rule out MDS/AML before assuming autoimmune."
            ),
        },
        {
            "term": "RB1-Retinoblastoma-Secondary-Osteosarcoma-Protocol",
            "definition": (
                "RB1 / Hereditary Retinoblastoma Secondary Cancer Management: "
                "BILATERAL RETINOBLASTOMA — GERMLINE RB1: "
                "  Bilateral = germline until proven otherwise — test ALL bilateral cases; "
                "  Unilateral multifocal = germline until proven otherwise; "
                "  Paediatric ophthalmological surveillance from birth in RB1 families; "
                "  Intra-arterial chemotherapy (IAC) preferred for primary treatment — avoids systemic toxicity + radiation; "
                "  AVOID EXTERNAL BEAM RADIATION: radiation-field osteosarcoma risk dramatically elevated in RB1 germline; "
                "TRILATERAL RETINOBLASTOMA: "
                "  Pineoblastoma (most common) or suprasellar PNET; "
                "  Occurs in ~3-5% bilateral germline RB1; typically age 2-3yr; "
                "  Brain MRI at diagnosis + every 6 months until age 5yr; "
                "  If diagnosed early: poor prognosis but treated with high-dose chemotherapy ± ASCT; "
                "SECONDARY CANCER SURVEILLANCE (SURVIVORS): "
                "  SECONDARY OSTEOSARCOMA: 30-40% lifetime risk — LEADING CAUSE OF DEATH in RTS; "
                "    In-field (prior orbital RT): extremely elevated — avoid RT whenever possible; "
                "    Out-of-field (13q LOH in mesenchymal cells): ~25% lifetime; "
                "  SECONDARY SARCOMA SURVEILLANCE: "
                "    Annual MRI from age 15 (focus on bilateral femora + tibiae); "
                "    WBMRI or WBCT where accessible; "
                "    Annual clinical musculoskeletal assessment; "
                "    Any bone pain in young adult RB1 survivor → urgent MRI; "
                "  OTHER SECONDARY CANCERS: "
                "    Soft tissue sarcoma, melanoma, bladder cancer — lifetime elevated; "
                "    Annual dermatology from age 30; annual cystoscopy from age 40 if radiation history; "
                "TREATMENT OF SECONDARY OSTEOSARCOMA: "
                "  Same MAP chemotherapy as de novo osteosarcoma; "
                "  Outcomes: similar or slightly worse than de novo due to prior treatment exposure; "
                "  Surgical resection where feasible; "
                "GENETIC COUNSELLING: "
                "  AD with ~90% penetrance; 45% de novo; "
                "  Family members: full eye examination + germline test; "
                "  Pre-implantation genetic testing: available; "
                "  Siblings: ophthalmoscopy at birth; RB1 test from birth."
            ),
        },
        {
            "term": "DICER1-PPB-Surveillance-Protocol",
            "definition": (
                "DICER1 / FAPOL Management (Pleuropulmonary Blastoma and Associated Tumours): "
                "PLEUROPULMONARY BLASTOMA (PPB) — PATHOGNOMONIC: "
                "  Type I (cystic): infant <2yr; purely cystic lung lesion; favourable prognosis >85% OS; "
                "    CRITICAL: any infant with lung cysts → DICER1 germline test + family screening; "
                "    Can be misdiagnosed as congenital pulmonary airway malformation (CPAM/CCAM) — "
                "      DICER1 testing on any infant 'CPAM' recommended (some are PPB Type I); "
                "  Type II (mixed): cystic + solid; intermediate prognosis; "
                "  Type III (solid): fully solid; aggressive; <50% 5-yr OS; "
                "    Chemotherapy: cisplatin + vincristine + actinomycin D or ifosfamide; "
                "    Surgery: complete resection goal; lobectomy/pneumonectomy; "
                "PPB FAMILY SURVEILLANCE (first-degree relatives of DICER1 carriers): "
                "  Chest CT: 3-monthly from birth to age 3yr; then 6-monthly to age 8yr; then annually to 12yr; "
                "  If normal at age 8: risk reducing to baseline; "
                "  ANY new respiratory symptoms in DICER1 child → urgent chest CT; "
                "  DO NOT use plain X-ray alone — cysts may be missed; "
                "THYROID SURVEILLANCE: "
                "  Annual thyroid USS from adolescence for all DICER1 carriers; "
                "  Multinodular goitre: annual USS + TSH; FNA if suspicious; "
                "  Differentiated thyroid cancer: standard management; "
                "OVARIAN SERTOLI-LEYDIG CELL TUMOUR (SLCT): "
                "  Annual pelvic USS from age 8yr (or menarche, whichever first); "
                "  Signs: virilisation (hirsutism, clitoromegaly, voice change); irregular menses; "
                "  CA-125 and AFP: elevated in moderately-differentiated SLCT; "
                "  Treatment: fertility-sparing surgery (unilateral salpingo-oophorectomy) when feasible; "
                "  Bleomycin + etoposide + cisplatin (BEP) for advanced disease; "
                "EMBRYONAL RMS CERVIX: "
                "  Surveillance: pelvic MRI annually from onset of sexual activity; "
                "  Botryoid appearance on imaging: DICER1 germline test if not known; "
                "  Fertility-sparing resection + chemotherapy (IRS protocols); "
                "CYSTIC NEPHROMA / WILMS: "
                "  Annual renal USS from birth to age 8yr; "
                "  Any cystic renal mass in DICER1 child: consider DICER1-associated cystic nephroma; "
                "PENETRANCE NOTE: "
                "  ~13% of DICER1 germline carriers develop PPB — most carriers are unaffected; "
                "  Surveillance therefore based on family history + early detection potential."
            ),
        },
        {
            "term": "EXT1-EXT2-HME-Malignant-Transformation-Protocol",
            "definition": (
                "EXT1 / EXT2 Hereditary Multiple Exostoses — Chondrosarcoma Surveillance: "
                "NATURAL HISTORY: "
                "  Osteochondromas grow throughout childhood with bone growth; "
                "  Growth halts at skeletal maturity (~16-18yr female; ~18-20yr male); "
                "  Multiple lesions (5-100s); metaphyses of long bones predominate; "
                "  Radiological hallmark: osteochondroma pointing AWAY from the joint; "
                "  Short stature, limb deformity, pain — EXT1 > EXT2 severity; "
                "CHONDROSARCOMA TRANSFORMATION — KEY CLINICAL RULES: "
                "  Cartilage cap thickness > 2cm on MRI after skeletal maturity = HIGH SUSPICION; "
                "  Growth of existing lesion AFTER skeletal maturity = alarm; "
                "  New pain in previously painless exostosis = alarm; "
                "  Soft-tissue mass around lesion = alarm; "
                "  FDG-PET: elevated SUVmax in chondrosarcoma vs benign osteochondroma; "
                "MANAGEMENT OF SUSPICIOUS LESION: "
                "  Urgent MRI with gadolinium; "
                "  Multidisciplinary sarcoma team review; "
                "  Biopsy: for definitive diagnosis if resection not immediately planned; "
                "    CAUTION: biopsy may seed chondrosarcoma — plan surgical approach concurrently; "
                "  Wide surgical excision: curative for grade 1; "
                "  Chemotherapy: NOT effective for chondrosarcoma (grade 1-2); "
                "  Radiation: LOW response; "
                "  Dedifferentiated chondrosarcoma (grade 3): doxorubicin + ifosfamide; poor prognosis; "
                "SURVEILLANCE PROTOCOL: "
                "  Annual clinical examination: new/changing lesions; deformity; neurological symptoms; "
                "  Targeted X-ray annually (known lesion sites); "
                "  MRI for suspicious lesions (cap measurement); "
                "  Paediatric assessment: growth; limb-length discrepancy; deformity management; "
                "  Physiotherapy: joint mobility; pain management; "
                "EXT1 vs EXT2 DISTINCTION: "
                "  EXT1: more lesions; more deformity; higher malignant risk (1-5%); "
                "  EXT2: fewer lesions; less deformity; lower malignant risk (~1%); "
                "  Clinically similar — cannot distinguish without molecular testing; "
                "  Both require same protocol but EXT1 warrants heightened vigilance; "
                "ORTHOPAEDIC INTERVENTIONS: "
                "  Forearm bowing: osteotomy + ulnar lengthening if functional; "
                "  Valgus knee: guided growth stapling in growth years; "
                "  Cord compression (rare cervical/thoracic lesion): urgent decompression."
            ),
        },
        {
            "term": "RECQL4-Rothmund-Thomson-Osteosarcoma-Protocol",
            "definition": (
                "RECQL4 / Rothmund-Thomson Syndrome Type 2 Management: "
                "DIAGNOSIS: "
                "  PATHOGNOMONIC poikiloderma: onset 3-6 months of age; "
                "    Stage 1 (months 3-6): erythema and oedema (butterfly face distribution); "
                "    Stage 2 (6-18 months): telangiectasia, pigmentary changes, skin atrophy; "
                "    Persists and spreads to extremities over years; "
                "  Short stature (~85%); sparse hair (scalp, eyebrows, lashes); "
                "  Radial ray hypoplasia / absent thumb; small hands and feet; "
                "  Dental abnormalities; nail dystrophy; "
                "  Juvenile cataracts; "
                "OSTEOSARCOMA SURVEILLANCE — 30% RISK: "
                "  Annual MRI lower limbs from age 5yr: "
                "    Bilateral distal femur, proximal tibia, proximal humerus; "
                "    Full lower limb survey — similar distribution to sporadic osteosarcoma; "
                "  Annual skeletal survey X-ray (baseline + surveillance for other bone changes); "
                "  ANY bone pain in RTS2 child → urgent MRI within days; "
                "  Annual CBC/CMP (bone marrow failure monitoring); "
                "OSTEOSARCOMA TREATMENT: "
                "  Standard MAP chemotherapy protocol: "
                "    Methotrexate (high-dose) + doxorubicin + cisplatin; "
                "  Surgical resection: limb salvage where feasible; "
                "  Consideration: impaired DNA repair in RECQL4 → possible increased chemotherapy sensitivity; "
                "    Monitor toxicity carefully; dose modifications if excess toxicity; "
                "BONE MARROW MONITORING: "
                "  Annual CBC: watch for cytopenias; "
                "  If pancytopenia: bone marrow biopsy to evaluate aplasia vs MDS; "
                "  HSCT: for severe bone marrow failure; conditioning chemotherapy toxicity may be higher; "
                "DIFFERENTIAL: "
                "  RTS1 (ANAMORSIN / C16orf57 mutations): poikiloderma WITHOUT osteosarcoma — "
                "    CRITICAL DDx: RTS1 patients do NOT need osteosarcoma surveillance; "
                "  Baller-Gerold syndrome (RECQL4): craniosynostosis + radial aplasia + poikiloderma; "
                "  RAPADILINO (RECQL4): radial + patella aplasia, Finnish founder; no poikiloderma classic; "
                "  All three can have RECQL4 mutations — molecular testing determines exact syndrome; "
                "ONCOLOGY COORDINATION: "
                "  RTS2 osteosarcoma: treat at sarcoma centre familiar with hereditary bone tumours; "
                "  All new bone pain requires same-day imaging triage."
            ),
        },
        {
            "term": "WRN-Werner-Syndrome-Sarcoma-Protocol",
            "definition": (
                "WRN / Werner Syndrome Management (Adult Progeria + Sarcoma): "
                "CLINICAL RECOGNITION: "
                "  BILATERAL CATARACTS before age 30 — PATHOGNOMONIC (bilateral in 96%); "
                "  Phenotype in 20s-30s: short stature; grey hair/alopecia; bird-like facies; "
                "    Thin limbs + truncal obesity; skin atrophy; leg ulcers (calcified); "
                "  Metabolic: type 2 diabetes mellitus (72%); hypogonadism; hypothyroidism; "
                "  Cardiovascular: premature atherosclerosis; MI/CVA in 40s-50s — leading cause of death; "
                "CANCER SURVEILLANCE: "
                "  Annual MRI whole body from age 30 (sarcoma, thyroid, melanoma); "
                "  Annual skin examination from age 25 (melanoma, SCC); "
                "  Annual thyroid USS from age 25; "
                "  Annual CBC/CMP/fasting glucose/lipids; "
                "  Annual ECG + echocardiogram (atherosclerosis, LV function); "
                "SARCOMA IN WERNER SYNDROME: "
                "  Median age at sarcoma diagnosis: ~40yr; "
                "  Histology: osteosarcoma, soft tissue sarcomas (leiomyosarcoma, fibrosarcoma, UPS), "
                "    thyroid carcinoma (follicular / anaplastic), meningioma; "
                "  Treatment: standard sarcoma protocols; "
                "  DNA repair deficiency: theoretical sensitivity to platinum drugs; monitor toxicity; "
                "LEG ULCERS: "
                "  Common, painful, often calcified; result of microvascular disease + skin atrophy; "
                "  Multi-disciplinary wound care: debridement, dressings, vascular assessment; "
                "  Infected ulcers: systemic antibiotics + culture-guided therapy; "
                "  Skin grafting: sometimes necessary; "
                "CATARACT MANAGEMENT: "
                "  Surgical phacoemulsification: standard approach; "
                "  Early referral given bilateral nature; "
                "GENETIC COUNSELLING: "
                "  Autosomal recessive: 25% recurrence risk for siblings; parents unaffected carriers; "
                "  Carrier testing: sequence parents after proband confirmed; "
                "  Pre-implantation genetic testing available for families; "
                "TREATMENT PRINCIPLES: "
                "  No disease-modifying therapy; "
                "  Sirolimus (mTOR): preclinical rationale; limited human data; "
                "  Japanese national registry: largest cohort data; founder alleles c.3139-1G>C."
            ),
        },
        {
            "term": "NF1-MPNST-Sarcoma-FDG-PET-Protocol",
            "definition": (
                "NF1 / MPNST Surveillance and Treatment Protocol (sarcoma focus): "
                "MPNST BIOLOGY: "
                "  Malignant peripheral nerve sheath tumour (MPNST) = most common hereditary sarcoma; "
                "  NF1 MPNST arises from Schwann cells within pre-existing plexiform neurofibromas; "
                "  Pathway: NF1 LOF → RAS hyperactivation → CDKN2A (INK4A/ARF) LOH somatic → MPNST; "
                "  Molecular hallmarks of NF1-MPNST: NF1 LOH + CDKN2A deletion + PRC2 complex loss; "
                "RISK STRATIFICATION: "
                "  HIGH RISK NF1 patients (for MPNST): "
                "    Internal / paraspinal / mediastinal plexiform neurofibroma; "
                "    Prior MPNST in family; "
                "    NF1 microdeletion (whole gene deletion); "
                "    CDKN2A co-deletion; "
                "    Age 20-40yr (MPNST peak incidence); "
                "FDG-PET PROTOCOL — MANDATORY FOR SUSPECTED MPNST: "
                "  Standard: 18F-FDG PET/CT; "
                "  SUVmax >3.5: highly specific for MPNST (sensitivity 89%, specificity 95%); "
                "  Dual-time FDG-PET (1hr and 2hr): rising SUVmax at 2hr favours malignancy; "
                "  Correlation with MRI: size change + FDG uptake = most reliable combination; "
                "ALARM SIGNS (REQUIRE FDG-PET WITHIN WEEKS): "
                "  Rapid growth of existing plexiform lesion; "
                "  New or worsening pain in pre-existing neurofibroma; "
                "  Change in texture from soft to firm; "
                "  New neurological deficit referrable to lesion; "
                "ANNUAL MRI BODY: "
                "  For NF1 patients with known large internal plexiform; "
                "  Whole-spine MRI if paraspinal plexiform; "
                "MPNST TREATMENT: "
                "  R0 wide surgical excision = primary treatment — margins critical; "
                "  Chemotherapy: doxorubicin + ifosfamide (1st line); response ~20-30%; "
                "  Radiation: AVOID in NF1 if possible (radiation-induced MPNST risk); "
                "    If radiation necessary: proton beam preferred; restrict field tightly; "
                "  EMERGING TARGETED THERAPIES: "
                "    Selumetinib (MEK1/2): Phase 2 trials for NF1-MPNST — modest response; "
                "    Mirdametinib (MEK): NF1 Phase 3 trials; "
                "    Binimetinib + PDR001: immunotherapy combination; "
                "  5-year OS NF1-MPNST: 20-40% (worse than sporadic — diagnostic delay + location); "
                "PROGNOSIS FACTORS: "
                "  Size >5cm: poor; "
                "  High grade (WHO 3): poor; "
                "  Incomplete resection (R1/R2): poor; "
                "  NF1-associated vs sporadic: NF1-associated worse outcomes overall."
            ),
        },
        {
            "term": "Hereditary-Sarcoma-Differential-Diagnosis-Guide",
            "definition": (
                "Differential diagnosis guide for hereditary sarcoma and bone tumour syndromes: "
                "OSTEOSARCOMA IN YOUNG PATIENT (<45yr): "
                "  TP53/LFS: family history of multiple cancers at young age; sarcoma + brain + ACC cluster; "
                "  RB1: prior bilateral retinoblastoma (in-field or out-of-field osteosarcoma); "
                "  RECQL4 (RTS2): congenital poikiloderma + short stature; osteosarcoma age <20yr; "
                "  Sporadic: ~70% of adolescent osteosarcoma — BUT TP53 somatic 30% (check germline if <30yr); "
                "MULTIPLE OSTEOCHONDROMAS: "
                "  EXT1: more lesions, more deformity; MLPA for deletion; "
                "  EXT2: fewer lesions, same distribution; "
                "  Isolated osteochondroma: sporadic; no family history; "
                "  SUSPECT HME: bilateral metaphyseal lesions + family history; "
                "SOFT TISSUE SARCOMA (YOUNG ADULT/EARLY ONSET): "
                "  NF1: MPNST from pre-existing plexiform neurofibroma; café-au-lait macules; "
                "  TP53/LFS: undifferentiated pleomorphic sarcoma / leiomyosarcoma; family cancer history; "
                "  WRN: adult-onset STS in context of premature ageing + bilateral cataracts; "
                "CHILDHOOD/INFANT SARCOMA: "
                "  DICER1: PPB (lung); embryonal RMS (cervix/bladder); age <6yr; "
                "  TP53/LFS: embryonal RMS; adrenocortical carcinoma <5yr is PATHOGNOMONIC LFS; "
                "  SMARCB1: malignant rhabdoid tumour kidney/CNS; INI1 IHC loss; "
                "PRECURSOR LESION → CHONDROSARCOMA: "
                "  EXT1/EXT2: cartilage cap >2cm after maturity; growing lesion post-maturity; "
                "  Ollier disease (IDH1 somatic mosaic): multiple enchondromas; NOT hereditary; "
                "  Maffucci syndrome (IDH1/IDH2 somatic): enchondromas + vascular malformations; NOT hereditary; "
                "SYNDROMIC CLUES: "
                "  Poikiloderma + osteosarcoma: RECQL4 (NOT RTS1 — RTS1 does NOT have osteosarcoma risk); "
                "  Premature ageing (cataracts <30 + thin limbs + truncal obesity + DM): WRN; "
                "  Multiple primaries in family (brain + breast + sarcoma + ACC): TP53 LFS; "
                "  Bilateral retinoblastoma + bone pain young adult: RB1 (secondary osteosarcoma); "
                "  Lung cyst in infant + family history: DICER1 (PPB Type I — do NOT assume CPAM); "
                "  Painful firm plexiform mass NF1 patient: NF1 MPNST — FDG-PET urgently."
            ),
        },
    ]
    return {
        "atlas":   "Hereditary-Sarcoma-Bone-Tumor-Predisposition-Atlas",
        "count":   len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["genes"][0], indent=2)[:1500])
    print("\n=== DEFINITIONS (first entry) ===")
    df = generate_definitions()
    print(json.dumps(df["definitions"][0], indent=2)[:1500])
