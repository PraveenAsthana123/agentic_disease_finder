#!/usr/bin/env python3
"""Hereditary-Esophageal-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
RHBDF2  (iRhom2 / Tylosis-Esophageal-Cancer / Howel-Evans Syndrome; 817aa; 17q25.1; AD GOF;
         OMIM 614321 TOC; palmoplantar keratoderma (diffuse non-epidermolytic PPK) PATHOGNOMONIC;
         ESCC nearly 100% penetrance by age 65yr in affected kindreds;
         Annual Lugol iodine endoscopy from age 30yr MANDATORY;
         AVOID TOBACCO/ALCOHOL ABSOLUTELY -- synergistic co-risk;
         seed SEED_BASE+0) .
TP53    (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome (LFS);
         ESCC/EAC elevated risk (p53 aberrant serous/squamous);
         AVOID RADIATION ABSOLUTELY -- radiation-induced secondary malignancies RULE 1;
         WBMRI annually Toronto Protocol; R337H Brazilian founder;
         Sarcoma 50-60% PRIMARY;
         seed SEED_BASE+1) .
CDKN2A  (Cyclin-Dependent Kinase Inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMM -- Familial Atypical Multiple Mole Melanoma;
         9p21.3 homozygous deletion common in ESCC and EAC;
         Cutaneous melanoma 25-36% PRIMARY; pancreatic 20x elevated;
         CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib);
         seed SEED_BASE+2) .
ATM     (Ataxia-Telangiectasia Mutated; 3056aa; 11q22.3; AD/AR LOF;
         Ataxia-Telangiectasia biallelic;
         RADIOSENSITIVITY ABSOLUTE in biallelic A-T;
         Monoallelic ATM: upper GI/esophageal 2-3x elevated;
         Ceralasertib ATRi + olaparib clinical trials;
         Breast cancer 15-25% monoallelic;
         seed SEED_BASE+3) .
BRCA2   (Breast Cancer Gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC -- Hereditary Breast-Ovarian Cancer syndrome;
         EAC 2-3x elevated risk;
         Cisplatin/carboplatin sensitivity (HR deficiency); Olaparib PARP inhibitor;
         Fanconi Anemia type D1 (biallelic) -- medulloblastoma/Wilms/AML MOST SEVERE FA;
         seed SEED_BASE+4) .
FANCA   (Fanconi Anemia Complementation Group A; 1455aa; 16q24.3; AR LOF;
         Fanconi Anemia type A -- most common FA complementation group ~60%;
         ESCC highest solid tumour risk in FA (10-15% lifetime ~400x RR vs general population);
         AVOID aldehyde/alcohol exposure ABSOLUTELY;
         Biallelic = classic FA (bone marrow failure, radial ray anomalies, VACTERL);
         OMIM FANCA 607139;
         seed SEED_BASE+5) .
MLH1    (MutL Homolog 1; 756aa; 3p22.2; AD LOF;
         Lynch Syndrome type 1 (HNPCC);
         EAC/ESCC 2-3x elevated;
         Pembrolizumab FDA-approved MSI-H tumours ALL HISTOLOGIES;
         Aspirin 600mg/day CAPP2 trial 50% CRC risk reduction LEVEL A evidence;
         seed SEED_BASE+6) .
PALB2   (Partner and Localiser of BRCA2; 1186aa; 16p12.2; AD LOF;
         HBOC-2;
         Breast 53% lifetime;
         Emerging upper GI/EAC 2-3x elevated data;
         Olaparib TBCRC048 82% ORR breast;
         FA-N biallelic; BRCA2-bridge protein;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3214-3221)
"""
import random

SEED_BASE = 3214

ATLAS_GENES = [
    {
        "gene": "RHBDF2",
        "protein": (
            "RHBDF2 -- 17q25.1 Autosomal-Dominant-GOF -- 817aa -- "
            "iRhom2-92kDa-RHBDF2-Inactive-Rhomboid-Protease-Howel-Evans-TOC-ESCC-100pct-Penetrance-"
            "PPK-PATHOGNOMONIC-Lugol-Iodine-Annual-Endoscopy-AVOID-TOBACCO-ALCOHOL-OMIM-614321"
        ),
        "locus": "17q25.1",
        "protein_size": (
            "817 aa / 92 kDa / 17q25.1 RHBDF2 encodes iRhom2 (Inactive Rhomboid Protease 2): "
            "STRUCTURE: "
            "  817 aa / 92 kDa; ER-resident inactive rhomboid pseudoprotease; "
            "  N-terminal cytoplasmic domain (aa 1-~100); "
            "  7 transmembrane helices (rhomboid-family topology); "
            "  Catalytic residues mutated -- INACTIVE protease (iRhom = inactive rhomboid); "
            "  iRhom2 function: chaperones TACE (ADAM17/ADAM10 metalloprotease) through ER; "
            "  iRhom2 regulates TACE maturation -> EGFR ligand shedding (AREG, EREG, EGF); "
            "  RHBDF2 GOF mutations (TOC-associated) -> INCREASED iRhom2/TACE activity; "
            "  GOF -> hyperactivated EGFR pathway -> esophageal keratinocyte proliferation; "
            "  RHBDF2 GOF: Ile186Thr, Pro189Leu, Asp188Glu (TOC hotspot mutations 17q25.1); "
            "TYLOSIS ESOPHAGEAL CANCER (TOC) -- HOWEL-EVANS SYNDROME: "
            "  OMIM 614321 (TOC); autosomal dominant GOF; 17q25.1; "
            "  Tylosis = palmoplantar keratoderma (PPK): diffuse non-epidermolytic PPK = PATHOGNOMONIC; "
            "  PPK: diffuse thickening of palms and soles -- onset childhood age 5-15yr; "
            "  Leukoplakia of oral mucosa: present in ~80% of carriers; "
            "  ESCC (esophageal squamous cell carcinoma): nearly 100% penetrance by age 65yr; "
            "  TOC-Liverpool/London/Finnish families: 3 well-characterised multi-generation pedigrees; "
            "  ESCC risk: extraordinarily high -- near certainty if carrier survives to age 65yr; "
            "PPK (PALMOPLANTAR KERATODERMA) -- PATHOGNOMONIC: "
            "  PPK = PATHOGNOMONIC for RHBDF2 TOC; non-epidermolytic = no skin fragility (contrast EBS); "
            "  PPK precedes ESCC by decades -- all PPK patients in affected kindreds = germline RHBDF2; "
            "  PPK + esophageal cancer family history = RHBDF2 until proven otherwise; "
            "  Oral leukoplakia (non-cancerous): additional mucosal involvement; "
            "ESCC SURVEILLANCE (TOC): "
            "  Annual endoscopy with Lugol iodine solution from age 30yr = MANDATORY; "
            "  Lugol iodine: stains normal glycogen-rich mucosa brown; ESCC and dysplasia appear UNSTAINED; "
            "  Lugol chromoendoscopy: superior to white-light for early ESCC detection in TOC; "
            "  Biopsy all Lugol-unstained areas; "
            "  Surveillance from age 30yr: high-risk window begins 3rd decade; "
            "AVOID TOBACCO/ALCOHOL -- ABSOLUTE CRITICAL CO-RISK: "
            "  Tobacco: ABSOLUTE risk factor in ESCC; germline RHBDF2 + tobacco = extreme synergistic risk; "
            "  Alcohol: acetaldehyde (ethanol metabolite) is direct ESCC carcinogen; "
            "  All RHBDF2 TOC carriers: AVOID tobacco and alcohol ABSOLUTELY; "
            "  EGFR pathway hyperactivation (GOF) + tobacco/alcohol carcinogens -> ESCC acceleration; "
            "SURVEILLANCE (TOC COMPLETE): "
            "  Annual Lugol iodine endoscopy from age 30yr (esophageal); "
            "  Dermatology monitoring for PPK management and oral leukoplakia; "
            "  Family cascade testing of first-degree relatives; "
            "  Tobacco/alcohol avoidance counselling MANDATORY; "
            "  Nutritional monitoring (esophageal dysmotility risk with surveillance)"
        ),
        "inheritance": "Autosomal Dominant (AD); germline GOF; OMIM 614321 (TOC); near 100% penetrance ESCC by age 65yr; PPK onset childhood; Howel-Evans Syndrome families; family cascade mandatory",
        "cancer_risk": "ESCC (esophageal squamous cell carcinoma): nearly 100% penetrance by age 65yr = HIGHEST/NEAR-CERTAIN; PPK PATHOGNOMONIC precedes ESCC; oral leukoplakia in 80%; no other major organ system elevated",
        "pathognomonic": "PPK (palmoplantar keratoderma, diffuse non-epidermolytic) PATHOGNOMONIC for RHBDF2-TOC; Lugol iodine chromoendoscopy unstained areas = ESCC/dysplasia; TOC triad: PPK + oral leukoplakia + ESCC family history = RHBDF2 until proven",
        "surveillance_key": "Annual Lugol iodine endoscopy from age 30yr MANDATORY; AVOID tobacco/alcohol ABSOLUTELY; PPK dermatology; oral leukoplakia surveillance; family cascade testing all first-degree relatives; EGFR pathway GOF = no targeted inhibitor approved yet",
        "key_distinctions": [
            "PPK-DIFFUSE-NON-EPIDERMOLYTIC-PATHOGNOMONIC",
            "ESCC-100PCT-PENETRANCE-AGE-65-HIGHEST",
            "LUGOL-IODINE-CHROMOENDOSCOPY-ANNUAL-MANDATORY",
            "AVOID-TOBACCO-ALCOHOL-ABSOLUTELY-SYNERGISTIC",
            "EGFR-TACE-ADAM17-GOF-MECHANISM",
            "TOC-HOWEL-EVANS-LIVERPOOL-LONDON-KINDREDS",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-ESCC-EAC-Elevated-AVOID-RADIATION-ABSOLUTELY-"
            "WBMRI-Toronto-R337H-Brazilian-Sarcoma-50-60pct-PRIMARY-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes Tumour Protein p53 (Guardian of the Genome): "
            "STRUCTURE: "
            "  393 aa / 43 kDa; tetrameric transcription factor; "
            "  N-terminal transactivation domain (TAD1: aa 1-40; TAD2: aa 40-67); "
            "  Proline-rich domain (aa 67-98); "
            "  DBD (DNA-binding domain): aa 94-292 -- CONTAINS ALL HOTSPOT RESIDUES (R175, G245, R248, R249, R273, R282); "
            "  Tetramerisation domain (aa 325-356) -- forms functional homotetramers; "
            "  C-terminal regulatory domain (aa 356-393): acetylation, ubiquitination; "
            "  p53 activates transcription of: p21 (CDKN1A -- G1 arrest), BAX (apoptosis), PUMA, NOXA, MDM2; "
            "  p53 responds to: DNA damage, oncogene activation, hypoxia, oxidative stress; "
            "  GOF hotspot mutations (R175H, R248W, R273H): gain oncogenic function -> invasion; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; AD LOF; de novo ~25%; highly penetrant LFS (~80-100% lifetime cancer); "
            "  Sarcoma: 50-60% LFS (soft tissue sarcoma + osteosarcoma) = PRIMARY cancer type; "
            "  Breast cancer: 30-40% (early-onset <40yr); "
            "  Brain tumour: 10-15% (astrocytoma, choroid plexus carcinoma PATHOGNOMONIC in children); "
            "  Adrenocortical carcinoma (ACC): 10-15% (children: R337H-associated in Brazil); "
            "ESCC/EAC IN LFS: "
            "  ESCC and EAC: elevated risk in TP53 germline carriers; "
            "  p53 somatic mutations: most frequent in ESCC (~80-90% sporadic ESCC) and EAC; "
            "  TP53-null IHC or TP53-diffuse overexpression: PATHOGNOMONIC for TP53-aberrant esophageal tumour; "
            "  ESCC/EAC: moderate elevated risk with germline TP53 (sarcoma overwhelmingly dominant PRIMARY); "
            "AVOID RADIATION -- ABSOLUTE RULE: "
            "  LFS: AVOID THERAPEUTIC RADIATION ABSOLUTELY -- RULE 1 -- no exceptions; "
            "  Radiation-induced secondary sarcomas in TP53 LOF carriers: extreme latency risk; "
            "  Historical cases: radiation to breast cancer -> fatal radiation-induced sarcoma; "
            "  Replace CT surveillance with MRI (non-ionising) in ALL LFS patients; "
            "  Even diagnostic X-rays: minimise; prefer MRI/ultrasound at all times; "
            "TORONTO WBMRI PROTOCOL: "
            "  Whole-Body MRI (WBMRI): annually in LFS -- Toronto Protocol; "
            "  WBMRI detects sarcoma, breast, ACC, brain, colorectal in one examination; "
            "  Replaces CT-based surveillance -- NO RADIATION; "
            "R337H BRAZILIAN FOUNDER MUTATION: "
            "  TP53 R337H: Brazilian founder mutation, frequency ~1/300 southern Brazil; "
            "  Associated with paediatric ACC predominantly; lower penetrance than classic LFS; "
            "p53 IHC ABERRANT PATTERNS: "
            "  p53-null (complete loss): PATHOGNOMONIC for TP53 LOF mutation; "
            "  p53-overexpression (diffuse strong): PATHOGNOMONIC for missense GOF mutation"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 191170/151623; ~25% de novo; highly penetrant LFS (~80-100% lifetime cancer); GOF hotspot mutations possible; family cascade mandatory",
        "cancer_risk": "Sarcoma (STS + osteosarcoma): 50-60% PRIMARY LFS; breast cancer 30-40%; brain tumour 10-15%; ACC 10-15% children; ESCC/EAC elevated (p53 aberrant esophageal histology common); R337H Brazilian paediatric ACC",
        "pathognomonic": "AVOID RADIATION ABSOLUTELY (LFS Rule 1); R337H Brazilian founder 1/300 south Brazil paediatric ACC; p53-null or p53-overexpression IHC = aberrant p53 PATHOGNOMONIC in ESCC/EAC; choroid plexus carcinoma in children PATHOGNOMONIC LFS",
        "surveillance_key": "WBMRI annually Toronto Protocol (no radiation); brain MRI annual; abdominal US 6-monthly children; AVOID radiation absolutely -- MRI preferred; R337H Brazilian screen; endoscopy in ESCC family history; TP53 germline all early-onset sarcoma/ACC",
        "key_distinctions": [
            "AVOID-RADIATION-ABSOLUTELY-LFS-RULE-1",
            "WBMRI-TORONTO-ANNUALLY-NO-CT",
            "SARCOMA-50-60PCT-PRIMARY-LFS",
            "R337H-BRAZILIAN-FOUNDER-1-IN-300",
            "P53-ABERRANT-IHC-ESCC-EAC-PATHOGNOMONIC",
            "CHOROID-PLEXUS-CARCINOMA-PATHOGNOMONIC-CHILDREN",
        ],
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-16kDa-CDK4-6-Inhibitor-FAMM-Melanoma-25-36pct-"
            "Pancreatic-20x-9p21-Deletion-ESCC-EAC-CDK4-6i-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 16 kDa / 9p21.3 CDKN2A encodes two tumour suppressors via alternate reading frames: "
            "DUAL PRODUCT LOCUS: "
            "  CDKN2A locus (9p21.3): encodes TWO functionally distinct proteins via ARF: "
            "    p16-INK4A (exons 1alpha, 2, 3): 156 aa / 16 kDa; ankyrin repeat domain; "
            "    p14-ARF (exon 1beta + exon 2, alternate reading frame): 132 aa / 14 kDa; "
            "  SAME exon 2 shared, different frames -> completely different proteins; "
            "p16-INK4A FUNCTION: "
            "  p16-INK4A: CDK4/6 inhibitor -- binds CDK4 and CDK6; "
            "  CDK4/CDK6-cyclin D complex: phosphorylates Rb -> releases E2F -> S-phase entry; "
            "  p16-INK4A LOF -> CDK4/6 unimpeded -> Rb hyperphosphorylation -> unchecked G1-S transition; "
            "  Downstream: uncontrolled proliferation -> melanoma, pancreatic cancer, ESCC, EAC; "
            "p14-ARF FUNCTION: "
            "  p14-ARF: MDM2 antagonist -- binds MDM2 -> prevents MDM2-mediated p53 ubiquitination; "
            "  p14-ARF LOF: MDM2 free -> p53 degradation -> p53 pathway OFF; "
            "  CDKN2A deletion knocks out BOTH Rb (via p16) AND p53 (via ARF) pathways; "
            "FAMM SYNDROME: "
            "  FAMM = Familial Atypical Multiple Mole Melanoma; FAMMM-PC: FAMM + Pancreatic Carcinoma; "
            "  Cutaneous melanoma: 25-36% lifetime = PRIMARY hereditary CDKN2A indication; "
            "  Pancreatic cancer: 20x elevated risk = dominant co-risk after melanoma; "
            "9p21 HOMOZYGOUS DELETION -- ESCC AND EAC: "
            "  9p21.3 homozygous deletion (CDKN2A/p16): highly common in ESCC (50-80% sporadic); "
            "  CDKN2A deletion in EAC (esophageal adenocarcinoma): also frequent; "
            "  Barrett's esophagus progression to EAC: early CDKN2A deletion event; "
            "  Germline CDKN2A carriers: moderate elevated ESCC/EAC risk beyond melanoma/pancreatic; "
            "CDK4/6 INHIBITORS -- THERAPEUTIC: "
            "  CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib): FDA-approved HR+ breast cancer; "
            "  CDKN2A-deleted tumours: CDK4/6 inhibitor rationale (unimpeded CDK4/6 activity); "
            "  Clinical trials: CDK4/6 inhibitors in CDKN2A-deleted upper GI and esophageal cancers; "
            "SURVEILLANCE (CDKN2A): "
            "  Annual whole-body skin exam + dermoscopy (melanoma); "
            "  Pancreatic MRI/MRCP from age 40yr (CAPS consortium); "
            "  Endoscopic surveillance (EAC/Barrett's risk) in CDKN2A carriers with GI symptoms; "
            "  Photosensitivity counselling; SPF50+ photoprotection"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 600160 (FAMM); high penetrance melanoma + pancreatic; CDKN2A exon 1beta (ARF) or exon 2 mutations affect both p16 and p14-ARF; family cascade mandatory",
        "cancer_risk": "Cutaneous melanoma 25-36% lifetime PRIMARY; pancreatic cancer 20x elevated (FAMMM-PC); ESCC/EAC moderate elevated; 9p21 deletion very common in sporadic ESCC (50-80%) and EAC (Barrett's progression); CDK4/6i targeted",
        "pathognomonic": "FAMM (multiple atypical nevi + melanoma family history) = CDKN2A until proven otherwise; FAMMM-PC = pancreatic cancer co-segregating melanoma family; 9p21.3 homozygous deletion = most frequent CDKN2A alteration in ESCC/EAC somatic",
        "surveillance_key": "Annual whole-body skin exam + dermoscopy (melanoma); pancreatic MRI/MRCP from age 40yr; CDKN2A FISH dual-product locus; CDK4/6 inhibitors for CDKN2A-deleted tumours; endoscopic surveillance EAC/Barrett's; photosensitivity counselling SPF50+",
        "key_distinctions": [
            "9P21-DELETION-ESCC-EAC-50-80PCT-SPORADIC",
            "MELANOMA-25-36PCT-FAMM-PRIMARY",
            "PANCREATIC-20X-FAMMM-PC",
            "CDK4-6-INHIBITORS-CDKN2A-DELETED-TUMOURS",
            "P16-INK4A-P14-ARF-DUAL-PRODUCT-SAME-LOCUS",
            "BARRETT-ESOPHAGUS-CDKN2A-EARLY-EAC-EVENT",
        ],
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-LOF-Monoallelic-Biallelic-AR-AT -- 3056aa -- "
            "ATM-350kDa-PI3K-Like-Kinase-A-T-Cerebellar-Ataxia-Telangiectasia-"
            "RADIOSENSITIVITY-ABSOLUTE-Biallelic-Upper-GI-2-3x-Monoallelic-Ceralasertib-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM encodes Ataxia-Telangiectasia Mutated Kinase: "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; PI3K-related kinase (PIKK family); "
            "  HEAT repeat domain (aa 1-1960): protein-protein interactions; "
            "  FATC domain (aa 3024-3056): regulatory C-terminal; "
            "  FAT domain (aa 1961-2566): stabilises kinase domain; "
            "  Kinase domain (KD: aa 2713-3013); "
            "  ATM activated by MRN (MRE11-RAD50-NBS1) complex at DNA DSBs; "
            "  ATM auto-phosphorylates Ser1981 -> activation; "
            "  ATM substrates: H2AX-Ser139 (gammaH2AX), BRCA1-Ser1524, CHK2-Thr68, p53-Ser15; "
            "  ATM activates: CHK2 -> CDC25 degradation -> G1/S and G2/M checkpoints; "
            "ATAXIA-TELANGIECTASIA (BIALLELIC A-T): "
            "  Biallelic ATM LOF = A-T; OMIM 208900; AR; "
            "  Cerebellar ataxia: progressive from age 1-2yr; wheelchair-bound by age 10yr; "
            "  Oculocutaneous telangiectasia: conjunctival; onset 3-5yr; "
            "  Immune deficiency: IgA/IgG; recurrent sinopulmonary infections; "
            "  RADIOSENSITIVITY ABSOLUTE -- BIALLELIC A-T: "
            "    AVOID ALL THERAPEUTIC RADIATION -- LETHAL if given full-dose radiation; "
            "    ATM LOF -> failed DSB repair after radiation -> chromosomal catastrophe; "
            "    A-T cancer: leukaemia/lymphoma 80-100x elevated; T-cell leukaemia; "
            "MONOALLELIC ATM -- UPPER GI/ESOPHAGEAL RISK: "
            "  Monoallelic ATM LOF: moderate cancer predisposition; "
            "  Upper GI (esophageal/gastric): 2-3x elevated risk monoallelic ATM; "
            "  Breast cancer: 15-25% lifetime (monoallelic); "
            "  Pancreatic: 5-10x elevated; prostate 2-4x; "
            "  Mechanism: HRD-like state -> genomic instability -> GI cancer risk; "
            "CERALASERTIB (ATRi) + OLAPARIB: "
            "  ATM LOF -> HRD-like -> PARP inhibitor active (BRCAness); "
            "  Olaparib: active in ATM-mutant tumours; "
            "  Ceralasertib (AZD6738; ATRi): ATR inhibitor; "
            "  Ceralasertib + olaparib: clinical trials in ATM-deficient solid tumours; "
            "  Rationale: ATM LOF -> increased ATR reliance -> ATRi synthetic lethal; "
            "SURVEILLANCE (MONOALLELIC ATM): "
            "  Annual MRI breast from age 30yr; "
            "  Pancreatic MRI/MRCP from age 50yr; "
            "  PSA prostate from age 40yr; "
            "  Upper GI endoscopy in ATM carriers with GI symptoms or family history; "
            "  ATM monoallelic: avoid excessive radiation; A-T biallelic: AVOID ALL therapeutic radiation"
        ),
        "inheritance": "Autosomal Dominant (monoallelic LOF) / Autosomal Recessive (biallelic LOF -- A-T); OMIM 607585/208900; monoallelic: moderate risk; biallelic: A-T severe; family cascade mandatory",
        "cancer_risk": "Monoallelic: breast 15-25%, pancreatic 5-10x, upper GI/esophageal 2-3x, prostate 2-4x; biallelic (A-T): leukaemia/lymphoma 80-100x, cerebellar ataxia, telangiectasia, immune deficiency",
        "pathognomonic": "RADIOSENSITIVITY ABSOLUTE (biallelic A-T -- avoid radiation LETHAL); cerebellar ataxia + oculocutaneous telangiectasia = A-T PATHOGNOMONIC combination; ceralasertib (ATRi) + olaparib synthetic lethality ATM-deficient tumours",
        "surveillance_key": "Monoallelic: annual MRI breast age 30yr; pancreatic MRI age 50yr; prostate PSA age 40yr; olaparib PARP inhibitor ATM-mutant; ceralasertib ATRi + olaparib trials; upper GI endoscopy family history; biallelic A-T: avoid radiation ABSOLUTELY",
        "key_distinctions": [
            "RADIOSENSITIVITY-ABSOLUTE-BIALLELIC-AT-LETHAL",
            "CEREBELLAR-ATAXIA-TELANGIECTASIA-AT-PATHOGNOMONIC",
            "UPPER-GI-ESOPHAGEAL-2-3X-MONOALLELIC-ATM",
            "CERALASERTIB-ATRi-OLAPARIB-SYNTHETIC-LETHALITY",
            "BRCANESS-OLAPARIB-ATM-MUTANT",
            "BIALLELIC-LEUKAEMIA-LYMPHOMA-80-100X",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-HBOC-EAC-2-3x-"
            "Olaparib-PARP-Cisplatin-Sensitive-FA-D1-Biallelic-MOST-SEVERE-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes Breast Cancer Type 2 Susceptibility Protein: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; largest hereditary breast cancer gene product; "
            "  N-terminal transactivation domain; 8 BRC repeats (aa 1002-2085) -- each binds one RAD51; "
            "  DBD (DNA-binding domain): OB folds + Tower domain (aa 2402-3190); "
            "  C-terminal RAD51-binding motif (aa 3265-3330); "
            "  NLS (aa 3263-3269); nuclear scaffolding of RAD51 at DSBs; "
            "  BRCA2 function: HR scaffold -- loads RAD51 onto ssDNA at resected DSBs; "
            "  BRCA2 LOF -> HR deficiency (HRD) -> NHEJ-dependent repair -> chromosomal instability; "
            "  BRCA2 interacts with PALB2 (BRCA2-PALB2-BRCA1 module) at DSBs; "
            "EAC RISK (BRCA2): "
            "  Monoallelic BRCA2 LOF: 2-3x elevated EAC risk (esophageal adenocarcinoma); "
            "  EAC mechanism: HRD -> genomic instability; Barrett's esophagus context; "
            "  BRCA2-associated EAC: enhanced cisplatin/olaparib sensitivity (HRD); "
            "  Growing evidence base from BCAC and GCAC consortia; "
            "CISPLATIN/CARBOPLATIN SENSITIVITY: "
            "  BRCA2 LOF -> HRD -> PLATINATING AGENTS highly active; "
            "  Cisplatin-based regimens preferred in HRD-positive EAC/gastric; "
            "OLAPARIB (PARP INHIBITOR): "
            "  BRCA2 LOF -> HRD -> PARP inhibitor synthetic lethality; "
            "  Olaparib FDA-approved BRCA1/2-germline ovarian, breast, pancreatic, prostate; "
            "  BRCA2-related EAC: HRD-directed therapy approach; "
            "FANCONI ANEMIA TYPE D1 -- BIALLELIC -- MOST SEVERE FA: "
            "  BRCA2 = FANCD1 gene; biallelic BRCA2 LOF = Fanconi Anemia type D1; "
            "  FA-D1 = MOST SEVERE FA phenotype: early childhood cancers; "
            "  FA-D1 malignancies: medulloblastoma (brain), Wilms tumour (kidney), AML (leukaemia); "
            "  FA-D1 children: avoid mitomycin C/cross-linking agents; "
            "HBOC SURVEILLANCE (BRCA2): "
            "  Annual MRI breast + mammography from age 25yr; "
            "  BSO at age 40-45yr; PSA from age 40yr; "
            "  Pancreatic MRI/MRCP from age 50yr; "
            "  EAC/upper GI surveillance in BRCA2 carriers with Barrett's or reflux history"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 600185/612555; high penetrance breast/ovarian; biallelic = FA-D1 (MOST SEVERE FA); family cascade mandatory",
        "cancer_risk": "Breast (female 70%, male 8%), ovarian 15-25%, pancreatic 5-10%, prostate 5-8x; EAC 2-3x elevated; FA-D1 biallelic: medulloblastoma/Wilms/AML childhood cancers",
        "pathognomonic": "BRCA2 LOF -> HRD -> cisplatin/olaparib sensitivity; FA-D1 biallelic = MOST SEVERE FA (medulloblastoma/Wilms/AML PATHOGNOMONIC combination); BRCA2-mutant EAC enhanced platinum/PARP sensitivity",
        "surveillance_key": "Annual MRI breast + mammography age 25yr; BSO age 40-45yr; pancreatic MRI age 50yr; prostate PSA age 40yr; olaparib FDA BRCA2 ovarian/breast/pancreatic; FA-D1 biallelic: avoid cross-linking agents; EAC/Barrett's surveillance",
        "key_distinctions": [
            "EAC-2-3X-ELEVATED-BRCA2",
            "CISPLATIN-CARBOPLATIN-SENSITIVITY-HRD",
            "OLAPARIB-PARP-SYNTHETIC-LETHALITY",
            "FA-D1-BIALLELIC-MOST-SEVERE-FA",
            "FA-D1-MEDULLOBLASTOMA-WILMS-AML-TRIAD",
            "BRCA2-PALB2-BRCA1-HR-MODULE",
        ],
    },
    {
        "gene": "FANCA",
        "protein": (
            "FANCA -- 16q24.3 Autosomal-Recessive-LOF -- 1455aa -- "
            "FANCA-163kDa-FA-Core-Complex-Type-A-Most-Common-60pct-"
            "ESCC-400x-RR-HIGHEST-Solid-Tumour-AVOID-ALDEHYDE-ABSOLUTELY-BMF-Radial-Ray-OMIM-607139"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "1455 aa / 163 kDa / 16q24.3 FANCA encodes Fanconi Anemia Complementation Group A Protein: "
            "STRUCTURE: "
            "  1455 aa / 163 kDa; largest structural FA core complex subunit; "
            "  Leucine-rich repeats (LRR): N-terminal; "
            "  HEAT repeat domain: scaffold for FA core complex assembly; "
            "  Nuclear export signal (NES): cytoplasmic-nuclear shuttling; "
            "  NLS: nuclear localisation; FANCA nuclear localisation required for function; "
            "  FANCA forms FANCA-FANCG heterodimer (nuclear); "
            "  FANCA/G/C: three subunits critical for FA core complex nuclear import; "
            "  FA core complex: FANCA + FANCB + FANCC + FANCE + FANCF + FANCG + FANCL + FANCM + accessory; "
            "  FA core complex monoubiquitinates FANCD2 (Lys561) and FANCI (Lys523) at DNA ICLs; "
            "  FANCD2-Ub + FANCI-Ub: loads downstream effectors (BRCA2/FANCD1, RAD51C, BRCA1) for ICL repair; "
            "  FANCA LOF -> FA core complex dysfunction -> no FANCD2/FANCI monoubiquitination -> ICL repair failure; "
            "FANCONI ANEMIA TYPE A (MOST COMMON FA): "
            "  FA-A: most common FA complementation group -- ~60% of all FA patients; "
            "  OMIM 607139 (FANCA gene) / 227650 (Fanconi Anemia); AR biallelic LOF; "
            "  Classic FA triad: bone marrow failure (BMF) + radial ray anomalies + cancer predisposition; "
            "  BMF: typically presents age 5-10yr; progressive pancytopenia; "
            "  Radial ray anomalies: absent/hypoplastic thumbs, radial aplasia (50-75% FA-A); "
            "  VACTERL association (Vertebral-Anal-Cardiac-TracheoEsophageal-Renal-Limb anomalies): some FA; "
            "  Skin: cafe-au-lait spots, hyperpigmentation, hypopigmented patches; "
            "  Short stature, microcephaly; endocrine: insulin resistance, DM, hypothyroidism; "
            "ESCC RISK -- HIGHEST SOLID TUMOUR IN FA: "
            "  ESCC: highest solid tumour risk in FA patients; "
            "  ESCC lifetime risk in FA: 10-15% (vs ~0.04% general population = ~400x relative risk); "
            "  ESCC in FA: median onset age ~26yr (vs ~70yr general population) -- VERY EARLY ONSET; "
            "  Mechanism: FA pathway deficiency -> failure to repair acetaldehyde (ethanol metabolite)-induced ICLs; "
            "  Aldehydes (acetaldehyde, formaldehyde): directly damage FA pathway -- DNA ICL induction; "
            "AVOID ALDEHYDE/ALCOHOL EXPOSURE -- ABSOLUTE RULE: "
            "  FA patients: AVOID alcohol ABSOLUTELY (ethanol -> acetaldehyde via ADH); "
            "  Formaldehyde: occupational exposure AVOID ABSOLUTELY in FA; "
            "  Acetaldehyde directly damages FA pathway cells: ICL induction in FANCA-null cells lethal; "
            "  Clinical implication: alcohol abstinence + formaldehyde occupational exposure avoidance; "
            "MONOALLELIC FANCA CARRIERS: "
            "  Monoallelic FANCA LOF (carriers): mild elevated cancer risk (less characterised); "
            "  ESCC risk in monoallelic carriers: smaller elevation than biallelic FA; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "  HSCT: curative for BMF in FA-A (RIC -- Reduced Intensity Conditioning MANDATORY); "
            "  RIC-HSCT: myeloablative conditioning AVOIDED -- excessive toxicity in FA DNA repair deficiency; "
            "  HSCT does NOT prevent solid tumours (ESCC risk persists post-HSCT); "
            "DEB TEST (DIAGNOSTIC): "
            "  Diepoxybutane (DEB) challenge: diagnostic FA chromosome fragility test; "
            "  DEB -> DNA ICLs -> FA pathway required for repair; "
            "  DEB test: increased chromosomal breaks in FA patient cells = PATHOGNOMONIC FA diagnosis; "
            "SURVEILLANCE (FA-A): "
            "  Annual esophageal endoscopy from age 16yr (esophageal surveillance); "
            "  Regular haematology (CBC, bone marrow assessment); "
            "  ENT surveillance (head/neck squamous); "
            "  Gynaecological surveillance (vulvar/cervical HPV-related) in females; "
            "  AVOID tobacco, alcohol, formaldehyde ABSOLUTELY"
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF = classic FA-A; OMIM 607139/227650; monoallelic = carrier (mild elevated risk); most common FA ~60% of FA patients; family carrier testing mandatory",
        "cancer_risk": "ESCC: 10-15% lifetime in FA (400x RR vs general population) = HIGHEST SOLID TUMOUR; AML/MDS (haematologic): 25-30x; squamous cell carcinoma (head/neck, vulva, cervix): highly elevated; HSCT does not prevent solid tumours",
        "pathognomonic": "DEB (diepoxybutane) chromosomal fragility test PATHOGNOMONIC for FA; ESCC 400x RR HIGHEST solid tumour in FA; AVOID ALDEHYDE/ALCOHOL ABSOLUTELY (aldehydes directly damage FA pathway); BMF + radial ray anomalies + cafe-au-lait = FA classic triad",
        "surveillance_key": "Annual esophageal endoscopy from age 16yr; AVOID alcohol/aldehyde/formaldehyde ABSOLUTELY; RIC-HSCT for BMF (NOT full myeloablative); DEB test diagnosis; CBC/bone marrow monitoring; HPV vaccination; ENT head/neck surveillance; ESCC risk PERSISTS post-HSCT",
        "key_distinctions": [
            "ESCC-400X-RR-HIGHEST-SOLID-TUMOUR-FA",
            "AVOID-ALDEHYDE-ALCOHOL-ABSOLUTELY-FA-PATHWAY",
            "FA-A-MOST-COMMON-60PCT-ALL-FA",
            "DEB-TEST-CHROMOSOMAL-FRAGILITY-PATHOGNOMONIC",
            "BMF-RADIAL-RAY-CAFE-AU-LAIT-FA-TRIAD",
            "HSCT-RIC-MANDATORY-BMF-ESCC-PERSISTS",
        ],
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MLH1-90kDa-MutL-Homolog1-MMR-Lynch1-CRC-40-50pct-"
            "EAC-ESCC-2-3x-MSI-H-Pembrolizumab-ALL-HISTOLOGIES-Aspirin-CAPP2-50pct-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 90 kDa / 3p22.2 MLH1 encodes MutL Homolog 1 (DNA Mismatch Repair Protein): "
            "STRUCTURE: "
            "  756 aa / 90 kDa; MutL family ATP-dependent endonuclease; "
            "  N-terminal ATPase domain (HATPase-c fold): aa 1-336; "
            "  C-terminal dimerisation domain (MLH1-CTD): aa 506-756; "
            "  MLH1 obligately heterodimerises: MutLalpha = MLH1 + PMS2 (primary MMR); "
            "  MutLbeta = MLH1 + PMS1; MutLgamma = MLH1 + MLH3 (meiotic MMR); "
            "  MLH1 recruits PMS2 endonuclease latent activity -> nicks DNA strand for excision; "
            "  MLH1 LOF -> MutLalpha dysfunction -> MMR failure -> MSI-H phenotype; "
            "LYNCH SYNDROME TYPE 1 (MLH1): "
            "  OMIM 120435; AD LOF; most common Lynch gene causing methylation-spectrum; "
            "  Colorectal cancer (CRC): 40-50% lifetime = PRIMARY Lynch MLH1 risk; "
            "  Endometrial cancer: 40-50% lifetime (highest MLH1 endometrial risk); "
            "  Ovarian cancer: 10-12%; gastric cancer: 6-8%; small bowel; urinary tract; "
            "EAC/ESCC IN LYNCH: "
            "  EAC and ESCC: 2-3x elevated risk in MLH1/Lynch carriers; "
            "  Upper GI MLH1 association: gastric and esophageal surveillance warranted; "
            "  MSI-H EAC: responds to pembrolizumab immunotherapy; "
            "  Lynch-associated EAC/ESCC: check MMR IHC on all resected esophageal specimens; "
            "MLH1 HYPERMETHYLATION -- SOMATIC (NOT LYNCH): "
            "  MLH1 promoter hypermethylation: sporadic MSI-H CRC in elderly = NOT Lynch germline; "
            "  Somatic MLH1 methylation + BRAF V600E = sporadic NOT Lynch; "
            "MSI-H IHC -- PATHOGNOMONIC: "
            "  MLH1 protein loss on IHC = PATHOGNOMONIC for MLH1 LOF; "
            "  4-gene MMR IHC panel (MLH1/MSH2/MSH6/PMS2): loss of any = dMMR; "
            "PEMBROLIZUMAB (MSI-H ALL HISTOLOGIES): "
            "  Pembrolizumab (anti-PD1): FDA-approved MSI-H/dMMR solid tumours ALL HISTOLOGIES; "
            "  MLH1-mutant EAC/ESCC: pembrolizumab preferred if MSI-H confirmed; "
            "ASPIRIN CAPP2 TRIAL: "
            "  CAPP2 trial: aspirin 600mg/day -> 50% CRC risk reduction in Lynch; "
            "  LEVEL A evidence for aspirin chemoprevention in Lynch; "
            "SURVEILLANCE (MLH1/LYNCH): "
            "  Colonoscopy 1-2yr from age 25yr; "
            "  Annual gynaecological surveillance endometrial; "
            "  Upper endoscopy (esophageal/gastric) in Lynch upper GI risk carriers; "
            "  Aspirin 600mg/day CAPP2 chemoprevention"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 120436/120435; moderate-high penetrance CRC + endometrial; MLH1 hypermethylation = somatic NOT germline Lynch; family cascade mandatory",
        "cancer_risk": "CRC 40-50% PRIMARY Lynch1; endometrial 40-50%; EAC/ESCC 2-3x elevated; ovarian 10-12%; gastric 6-8%; small bowel; urinary tract; MSI-H tumours pembrolizumab-eligible ALL histologies",
        "pathognomonic": "MLH1 protein loss on IHC PATHOGNOMONIC MMR-deficient; MSI-H = dMMR pembrolizumab-eligible; MLH1 hypermethylation + BRAF V600E = sporadic NOT Lynch (key DDx); CAPP2 aspirin 50% CRC reduction LEVEL A evidence",
        "surveillance_key": "Colonoscopy 1-2yr from age 25yr; annual gynaecological endometrial surveillance; upper endoscopy esophageal/gastric Lynch upper GI risk; MSI-H IHC all CRC/endometrial/EAC universal screening; pembrolizumab MSI-H all histologies; aspirin 600mg/day CAPP2",
        "key_distinctions": [
            "MSI-H-IHC-PATHOGNOMONIC-MLH1-LOSS",
            "PEMBROLIZUMAB-MSI-H-ALL-HISTOLOGIES",
            "ASPIRIN-CAPP2-50PCT-CRC-REDUCTION-LEVEL-A",
            "EAC-ESCC-2-3X-LYNCH-MLH1",
            "MLH1-HYPERMETHYLATION-SOMATIC-NOT-LYNCH",
            "CRC-40-50PCT-ENDOMETRIAL-40-50PCT-LYNCH1",
        ],
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "PALB2-131kDa-WD40-BRCA2-Bridge-HBOC2-Breast-53pct-"
            "Upper-GI-EAC-2-3x-Olaparib-TBCRC048-82pct-ORR-FA-N-Biallelic-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 encodes Partner and Localiser of BRCA2: "
            "STRUCTURE: "
            "  1186 aa / 131 kDa; scaffold protein linking BRCA1-BRCA2 at DNA damage sites; "
            "  N-terminal coiled-coil domain (aa 1-~100): binds BRCA1 BRCT2 domain; "
            "  WD40 repeat domain (C-terminal aa ~850-1186): binds BRCA2 and RAD51/RAD51C; "
            "  PALB2 forms ternary BRCA1-PALB2-BRCA2 complex at DSBs; "
            "  PALB2 function: bridges BRCA1 (DSB sensing/signalling) to BRCA2 (RAD51 loading); "
            "  PALB2 LOF -> BRCA1-BRCA2 module disruption -> HR deficiency (HRD); "
            "  PALB2 is FA complementation group N (FANCN): biallelic = Fanconi Anemia type N; "
            "  PALB2/FANCN pathway: BRCA1-PALB2-BRCA2-RAD51 axis; "
            "HBOC-2 (PALB2): "
            "  PALB2 germline LOF: HBOC-2 syndrome (distinct from BRCA1/BRCA2-HBOC); "
            "  OMIM 610355; moderate-high penetrance; "
            "  Breast cancer (female): 53% lifetime = DOMINANT PALB2 risk; "
            "  Ovarian cancer: 3-5x elevated; "
            "  Pancreatic cancer: 2-3x elevated; "
            "  PALB2 breast cancer: similar to BRCA2 HR-deficient phenotype; "
            "UPPER GI / EAC EMERGING DATA (PALB2): "
            "  Upper GI and EAC: 2-3x elevated risk -- EMERGING data from GCAC/BCAC consortia; "
            "  HRD mechanism: PALB2 LOF -> BRCA1-BRCA2 axis disruption -> ICL/DSB repair failure; "
            "  PALB2-related upper GI: cisplatin/platinum sensitivity expected (HRD); "
            "  EAC and gastric: PALB2 germline carriers increasing evidence; "
            "OLAPARIB (PARP INHIBITOR) -- TBCRC048: "
            "  PALB2 LOF -> HRD -> PARP inhibitor synthetic lethality; "
            "  TBCRC048 trial: olaparib in PALB2 germline breast cancer; "
            "  TBCRC048: 82% ORR = HIGHEST reported PARP inhibitor ORR in germline breast cancer; "
            "  Olaparib: off-label consideration PALB2 tumours with HRD; "
            "FA-N -- FANCONI ANEMIA TYPE N (BIALLELIC): "
            "  PALB2 = FANCN; biallelic PALB2 LOF = Fanconi Anemia type N; "
            "  FA-N phenotype: severe; similar to FA-D1 (BRCA2 biallelic); "
            "  FA-N malignancies: medulloblastoma, Wilms tumour, leukaemia (biallelic childhood); "
            "  FA-N children: AVOID cross-linking agents / DEB diagnostic sensitivity test; "
            "BRCA2-BRIDGE FUNCTION: "
            "  PALB2 = molecular bridge between BRCA1 (upstream) and BRCA2 (downstream) in HR; "
            "  PALB2 mutations disrupt the entire BRCA1-PALB2-BRCA2-RAD51 axis; "
            "  PALB2-null cells: HRD phenotype functionally equivalent to BRCA2-null; "
            "SURVEILLANCE (PALB2): "
            "  Annual MRI breast + mammography from age 30yr; "
            "  BSO consideration age 40-50yr (ovarian cancer); "
            "  Pancreatic MRI/MRCP from age 50yr; "
            "  Upper GI endoscopy in carriers with GI symptoms or family EAC history; "
            "  Olaparib TBCRC048 evidence for breast cancer PARP inhibition"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 610355; moderate-high penetrance breast (53%); biallelic = FA-N (severe, similar FA-D1); family cascade mandatory",
        "cancer_risk": "Breast (female): 53% lifetime DOMINANT HBOC-2; ovarian 3-5x; pancreatic 2-3x; upper GI/EAC 2-3x EMERGING; FA-N biallelic: medulloblastoma/Wilms/AML childhood; olaparib TBCRC048 82% ORR HIGHEST",
        "pathognomonic": "PALB2 LOF -> HRD -> olaparib/cisplatin sensitivity (BRCA2-bridge protein); TBCRC048 82% ORR = HIGHEST PARP inhibitor ORR breast cancer; FA-N biallelic = Fanconi Anemia type N (severe biallelic); BRCA1-PALB2-BRCA2-RAD51 axis disruption",
        "surveillance_key": "Annual MRI breast + mammography age 30yr; BSO consideration age 40-50yr; pancreatic MRI age 50yr; olaparib TBCRC048 82% ORR breast; cisplatin platinum HRD sensitivity; upper GI endoscopy family EAC history; FA-N biallelic: avoid cross-linking agents",
        "key_distinctions": [
            "BREAST-53PCT-LIFETIME-DOMINANT-PALB2",
            "OLAPARIB-TBCRC048-82PCT-ORR-HIGHEST",
            "UPPER-GI-EAC-2-3X-EMERGING-DATA",
            "FA-N-BIALLELIC-SEVERE-MEDULLOBLASTOMA-WILMS",
            "BRCA2-BRIDGE-BRCA1-PALB2-BRCA2-RAD51",
            "HRD-CISPLATIN-PLATINUM-SENSITIVITY-PALB2",
        ],
    },
]


def _make_patients(gene_entry):
    """Deterministic synthetic cohort: 40 patients per gene."""
    seed = SEED_BASE + ATLAS_GENES.index(gene_entry)
    rng  = random.Random(seed)

    gene = gene_entry["gene"]
    age_params = {
        "RHBDF2": (52, 10),
        "TP53":   (28, 14),
        "CDKN2A": (40, 12),
        "ATM":    (48, 12),
        "BRCA2":  (52, 12),
        "FANCA":  (35, 10),
        "MLH1":   (44, 12),
        "PALB2":  (50, 12),
    }
    mu, sigma = age_params.get(gene, (40, 12))

    severe_rates = {
        "RHBDF2": 0.78,
        "TP53":   0.75,
        "CDKN2A": 0.62,
        "ATM":    0.52,
        "BRCA2":  0.55,
        "FANCA":  0.70,
        "MLH1":   0.58,
        "PALB2":  0.54,
    }
    sev_rate = severe_rates.get(gene, 0.5)

    patients = []
    for i in range(40):
        age       = max(5, round(rng.gauss(mu, sigma), 1))
        sev_event = rng.random() < sev_rate
        patients.append({
            "id":        f"{gene}-{i+1:02d}",
            "age_onset": age,
            "severe":    sev_event,
            "seed":      seed,
        })
    return patients


def generate_overview():
    rows = []
    for g in ATLAS_GENES:
        pts   = _make_patients(g)
        sev_n = sum(1 for p in pts if p["severe"])
        rows.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
            "inheritance":      g["inheritance"],
            "cancer_risk":      g["cancer_risk"],
            "protein":          g["protein"],
        })

    total_pts  = sum(r["n"]        for r in rows)
    total_sev  = sum(r["severe_n"] for r in rows)
    highest    = max(rows, key=lambda r: r["severe_pct"])

    return {
        "atlas":              "Hereditary-Esophageal-Cancer-Predisposition-Atlas",
        "seed_range":         f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes_n":            len(ATLAS_GENES),
        "total_patients":     total_pts,
        "severe_total_n":     total_sev,
        "severe_total_pct":   round(total_sev / total_pts * 100, 1),
        "highest_risk_gene":  highest["gene"],
        "highest_risk_pct":   highest["severe_pct"],
        "gene_summary":       rows,
        "genes_detail": [
            {
                "gene":             g["gene"],
                "inheritance":      g["inheritance"],
                "cancer_risk":      g["cancer_risk"],
                "pathognomonic":    g["pathognomonic"],
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
        sev_n    = sum(1 for p in pts if p["severe"])
        breakdown.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "seed":             SEED_BASE + seed_idx,
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
        })
    return {"atlas": "Hereditary-Esophageal-Cancer-Predisposition-Atlas", "breakdown": breakdown}


def generate_definitions():
    defs = [
        {
            "term": "RHBDF2 / TYLOSIS-HOWEL-EVANS / PPK-ESCC-PATHOGNOMONIC / LUGOL-IODINE-ENDOSCOPY / AVOID-TOBACCO-ALCOHOL-ABSOLUTELY",
            "definition": (
                "RHBDF2 -- 817aa / 92 kDa / 17q25.1 / AD GOF\n"
                "Tylosis Esophageal Cancer (TOC); PPK PATHOGNOMONIC; ESCC 100% penetrance; Lugol iodine annual endoscopy.\n\n"
                "TOC CANCER SPECTRUM:\n"
                "  ESCC: nearly 100% penetrance by age 65yr in affected kindreds = HIGHEST/NEAR-CERTAIN risk.\n"
                "  PPK (palmoplantar keratoderma, diffuse non-epidermolytic) = PATHOGNOMONIC onset childhood.\n"
                "  Oral leukoplakia: ~80% of RHBDF2 carriers; non-cancerous mucosal involvement.\n\n"
                "LUGOL IODINE CHROMOENDOSCOPY -- ANNUAL MANDATORY:\n"
                "  Annual endoscopy with Lugol iodine from age 30yr = MANDATORY surveillance.\n"
                "  Lugol stains normal glycogen-rich mucosa brown; ESCC/dysplasia = UNSTAINED areas.\n"
                "  Lugol chromoendoscopy superior to white-light for early ESCC in TOC kindreds.\n"
                "  Biopsy all Lugol-unstained areas; surveillance window begins 3rd decade.\n\n"
                "AVOID TOBACCO/ALCOHOL -- ABSOLUTE CRITICAL CO-RISK:\n"
                "  Tobacco: ABSOLUTE esophageal risk factor; RHBDF2 GOF + tobacco = synergistic extreme ESCC.\n"
                "  Alcohol: acetaldehyde (ethanol metabolite) = direct ESCC carcinogen.\n"
                "  ALL RHBDF2 TOC carriers: AVOID tobacco and alcohol ABSOLUTELY.\n\n"
                "MECHANISM:\n"
                "  RHBDF2 GOF -> hyperactivated TACE (ADAM17) -> EGFR ligand shedding (AREG/EREG/EGF).\n"
                "  EGFR hyperactivation + tobacco/alcohol carcinogens -> ESCC acceleration.\n\n"
                "SURVEILLANCE:\n"
                "  Annual Lugol iodine endoscopy from age 30yr (esophageal).\n"
                "  Dermatology (PPK management); oral leukoplakia monitoring.\n"
                "  Family cascade testing first-degree relatives mandatory."
            ),
        },
        {
            "term": "TP53 / LFS / AVOID-RADIATION-ABSOLUTELY / WBMRI-TORONTO / R337H-BRAZILIAN-FOUNDER",
            "definition": (
                "TP53 -- 393aa / 43 kDa / 17p13.1 / AD LOF\n"
                "Li-Fraumeni Syndrome; AVOID RADIATION ABSOLUTELY; WBMRI Toronto Protocol; R337H Brazilian founder.\n\n"
                "LFS CANCER SPECTRUM:\n"
                "  Sarcoma (STS + osteosarcoma): 50-60% = PRIMARY LFS cancer.\n"
                "  Breast cancer: 30-40% (early-onset <40yr).\n"
                "  Brain tumour: 10-15% (choroid plexus carcinoma in children PATHOGNOMONIC).\n"
                "  ACC: 10-15% children; ESCC/EAC elevated (p53 most common somatic mutation in ESCC).\n\n"
                "AVOID RADIATION -- ABSOLUTE RULE 1:\n"
                "  LFS: NO THERAPEUTIC RADIATION -- radiation-induced secondary sarcoma lethal risk.\n"
                "  Replace CT surveillance with MRI in ALL LFS patients.\n"
                "  Even diagnostic X-rays: minimise; prefer MRI/ultrasound.\n\n"
                "TORONTO WBMRI PROTOCOL:\n"
                "  WBMRI: annually -- detects sarcoma, breast, ACC, brain, CRC in one examination.\n"
                "  No ionising radiation; brain MRI annual (with gadolinium).\n\n"
                "R337H BRAZILIAN FOUNDER:\n"
                "  TP53 R337H: frequency ~1/300 southern Brazil.\n"
                "  Associated with paediatric ACC predominantly.\n\n"
                "p53 IHC ABERRANT PATTERNS (ESCC/EAC):\n"
                "  p53-null (complete loss): LOF mutation PATHOGNOMONIC.\n"
                "  p53-overexpression (diffuse strong): GOF missense PATHOGNOMONIC.\n"
                "  p53 most commonly mutated gene in ESCC (~80-90% sporadic)."
            ),
        },
        {
            "term": "CDKN2A / FAMM / MELANOMA-25-36PCT / 9P21-DELETION-ESCC-EAC / CDK4-6-INHIBITORS",
            "definition": (
                "CDKN2A -- 156aa / 16 kDa / 9p21.3 / AD LOF\n"
                "FAMM; melanoma 25-36% PRIMARY; 9p21 deletion common ESCC and EAC; CDK4/6 inhibitors.\n\n"
                "FAMM + FAMMM-PC:\n"
                "  FAMM: familial atypical multiple mole melanoma.\n"
                "  Cutaneous melanoma: 25-36% lifetime = PRIMARY hereditary CDKN2A indication.\n"
                "  Pancreatic cancer: 20x elevated = FAMMM-PC co-risk.\n\n"
                "9P21 DELETION -- ESCC AND EAC:\n"
                "  9p21.3 homozygous deletion (CDKN2A/p16): 50-80% sporadic ESCC.\n"
                "  CDKN2A deletion in EAC: common early event in Barrett's -> EAC progression.\n"
                "  Germline CDKN2A: moderate elevated ESCC/EAC risk.\n\n"
                "DUAL PRODUCT -- p16-INK4A + p14-ARF:\n"
                "  Same 9p21.3 locus: two proteins via alternate reading frames.\n"
                "  p16-INK4A: CDK4/6 inhibitor -> Rb pathway.\n"
                "  p14-ARF: MDM2 antagonist -> p53 pathway.\n"
                "  CDKN2A deletion knocks out BOTH Rb AND p53 pathways simultaneously.\n\n"
                "CDK4/6 INHIBITORS:\n"
                "  Palbociclib, ribociclib, abemaciclib: FDA-approved HR+ breast cancer.\n"
                "  CDKN2A-deleted upper GI / esophageal cancers: CDK4/6 inhibitor trials.\n\n"
                "SURVEILLANCE:\n"
                "  Annual whole-body skin exam + dermoscopy (melanoma).\n"
                "  Pancreatic MRI/MRCP from age 40yr (CAPS consortium guidelines).\n"
                "  Endoscopic surveillance (EAC/Barrett's risk) in CDKN2A carriers with GI symptoms."
            ),
        },
        {
            "term": "ATM / A-T-BIALLELIC / RADIOSENSITIVITY-ABSOLUTE / UPPER-GI-2-3X / CERALASERTIB-ATRi-OLAPARIB",
            "definition": (
                "ATM -- 3056aa / 350 kDa / 11q22.3 / AD LOF (monoallelic) / AR biallelic A-T\n"
                "Ataxia-Telangiectasia biallelic; radiosensitivity ABSOLUTE; upper GI 2-3x monoallelic; ceralasertib ATRi.\n\n"
                "ATAXIA-TELANGIECTASIA (BIALLELIC A-T):\n"
                "  Cerebellar ataxia: progressive from age 1-2yr; wheelchair-bound age 10yr.\n"
                "  Telangiectasia: conjunctival/oculocutaneous characteristic.\n"
                "  Immune deficiency: IgA/IgG; sinopulmonary infections.\n\n"
                "RADIOSENSITIVITY ABSOLUTE -- BIALLELIC A-T:\n"
                "  A-T: AVOID ALL THERAPEUTIC RADIATION -- LETHAL.\n"
                "  ATM LOF -> failed DSB repair after radiation -> chromosomal catastrophe.\n"
                "  Use MRI/US alternatives; even diagnostic X-rays: minimise.\n"
                "  A-T leukaemia/lymphoma: 80-100x elevated.\n\n"
                "MONOALLELIC ATM -- UPPER GI:\n"
                "  Upper GI (esophageal/gastric): 2-3x elevated (monoallelic).\n"
                "  Breast cancer: 15-25% lifetime; pancreatic 5-10x; prostate 2-4x.\n\n"
                "CERALASERTIB (ATRi) + OLAPARIB:\n"
                "  ATM LOF -> HRD-like -> PARP inhibitor synthetic lethality (olaparib).\n"
                "  Ceralasertib (ATR inhibitor) + olaparib: clinical trials ATM-deficient solid tumours.\n"
                "  Rationale: ATM LOF -> increased ATR reliance -> ATRi synthetic lethal."
            ),
        },
        {
            "term": "BRCA2 / HBOC / EAC-2-3X / OLAPARIB-PARP / FA-D1-BIALLELIC-MOST-SEVERE",
            "definition": (
                "BRCA2 -- 3418aa / 384 kDa / 13q12.3 / AD LOF\n"
                "HBOC; EAC 2-3x elevated; olaparib PARP inhibitor; FA-D1 biallelic MOST SEVERE FA.\n\n"
                "EAC RISK (BRCA2):\n"
                "  Monoallelic BRCA2 LOF: 2-3x elevated EAC risk.\n"
                "  HRD -> genomic instability; Barrett's esophagus context; enhanced cisplatin sensitivity.\n\n"
                "OLAPARIB (PARP INHIBITOR):\n"
                "  BRCA2 LOF -> HRD -> PARP inhibitor synthetic lethality.\n"
                "  Olaparib FDA-approved: BRCA2-germline ovarian, breast, pancreatic, prostate.\n"
                "  BRCA2-mutant EAC: HRD-directed therapy approach.\n\n"
                "FA-D1 BIALLELIC -- MOST SEVERE FA:\n"
                "  BRCA2 = FANCD1; biallelic = Fanconi Anemia type D1.\n"
                "  FA-D1 = MOST SEVERE FA: medulloblastoma + Wilms + AML in early childhood.\n"
                "  FA-D1 children: avoid mitomycin C + cross-linking agents.\n\n"
                "HBOC SURVEILLANCE:\n"
                "  Annual MRI breast + mammography from age 25yr.\n"
                "  BSO at age 40-45yr; PSA from 40yr; pancreatic MRI from 50yr.\n"
                "  EAC/upper GI surveillance in BRCA2 carriers with Barrett's or reflux history."
            ),
        },
        {
            "term": "FANCA / FA-TYPE-A / ESCC-400X-RR-HIGHEST-SOLID-TUMOUR / AVOID-ALDEHYDE-ABSOLUTELY / BMF-RADIAL-RAY",
            "definition": (
                "FANCA -- 1455aa / 163 kDa / 16q24.3 / AR LOF\n"
                "FA type A most common 60%; ESCC 400x RR HIGHEST solid tumour in FA; AVOID ALDEHYDE ABSOLUTELY.\n\n"
                "FA-A CANCER SPECTRUM:\n"
                "  ESCC: 10-15% lifetime in FA = HIGHEST SOLID TUMOUR RISK in FA patients.\n"
                "  ESCC relative risk: ~400x vs general population (10-15% vs ~0.04%).\n"
                "  ESCC onset in FA: median age ~26yr (vs ~70yr general population) -- VERY EARLY.\n"
                "  AML/MDS (haematologic): 25-30x elevated; squamous head/neck: highly elevated.\n"
                "  HSCT cures BMF but does NOT prevent solid tumours (ESCC risk PERSISTS post-HSCT).\n\n"
                "AVOID ALDEHYDE/ALCOHOL ABSOLUTELY:\n"
                "  FA patients: AVOID alcohol ABSOLUTELY (ethanol -> acetaldehyde via ADH).\n"
                "  Formaldehyde: occupational exposure AVOID ABSOLUTELY in FA.\n"
                "  Acetaldehyde/aldehydes: directly damage FA pathway cells (ICL induction).\n"
                "  FA cells: HYPERSENSITIVE to aldehyde-induced DNA interstrand crosslinks.\n\n"
                "CLASSIC FA TRIAD:\n"
                "  BMF (bone marrow failure): onset age 5-10yr; progressive pancytopenia.\n"
                "  Radial ray anomalies: absent/hypoplastic thumbs (50-75% FA-A).\n"
                "  Cancer predisposition: ESCC 400x, AML 25-30x.\n\n"
                "DEB TEST -- DIAGNOSTIC:\n"
                "  Diepoxybutane (DEB) challenge: chromosomal fragility = PATHOGNOMONIC FA diagnosis.\n"
                "  DEB -> DNA ICLs -> FA pathway required for repair -> FA cells: increased breaks.\n\n"
                "SURVEILLANCE:\n"
                "  Annual esophageal endoscopy from age 16yr; AVOID tobacco/alcohol/formaldehyde ABSOLUTELY.\n"
                "  RIC-HSCT for BMF (NOT full myeloablative -- excessive toxicity in FA).\n"
                "  CBC/bone marrow monitoring; HPV vaccination; ENT head/neck surveillance."
            ),
        },
        {
            "term": "MLH1 / LYNCH1 / MSI-H-PATHOGNOMONIC / PEMBROLIZUMAB-ALL-HISTOLOGIES / ASPIRIN-CAPP2-50PCT",
            "definition": (
                "MLH1 -- 756aa / 90 kDa / 3p22.2 / AD LOF\n"
                "Lynch Syndrome type 1; MSI-H IHC PATHOGNOMONIC; EAC/ESCC 2-3x elevated; pembrolizumab all histologies.\n\n"
                "LYNCH CRC + ENDOMETRIAL PRIMARY:\n"
                "  CRC: 40-50% lifetime = PRIMARY Lynch1 indication.\n"
                "  Endometrial: 40-50% lifetime; EAC/ESCC: 2-3x elevated Lynch risk.\n"
                "  Ovarian 10-12%; gastric 6-8%.\n\n"
                "MSI-H IHC -- PATHOGNOMONIC:\n"
                "  MLH1 protein loss on IHC = PATHOGNOMONIC MMR-deficient.\n"
                "  4-gene MMR IHC (MLH1/MSH2/MSH6/PMS2): loss of any = dMMR.\n"
                "  MSI-H by PCR/NGS correlates with IHC loss.\n\n"
                "MLH1 HYPERMETHYLATION -- SOMATIC NOT GERMLINE:\n"
                "  MLH1 methylation + BRAF V600E = sporadic NOT Lynch.\n"
                "  Distinguish from germline Lynch before family counselling.\n\n"
                "PEMBROLIZUMAB (MSI-H ALL HISTOLOGIES):\n"
                "  Pembrolizumab FDA-approved: MSI-H/dMMR solid tumours ALL histologies.\n"
                "  MLH1-mutant EAC/ESCC (MSI-H): pembrolizumab preferred immunotherapy.\n\n"
                "ASPIRIN CAPP2 -- LEVEL A:\n"
                "  Aspirin 600mg/day: 50% CRC risk reduction in Lynch = LEVEL A evidence.\n\n"
                "SURVEILLANCE:\n"
                "  Colonoscopy 1-2yr from age 25yr.\n"
                "  Annual gynaecological surveillance (endometrial).\n"
                "  Upper endoscopy (esophageal/gastric) in Lynch upper GI risk carriers."
            ),
        },
        {
            "term": "PALB2 / HBOC2 / BREAST-53PCT / OLAPARIB-TBCRC048-82PCT-ORR / FA-N-BIALLELIC",
            "definition": (
                "PALB2 -- 1186aa / 131 kDa / 16p12.2 / AD LOF\n"
                "HBOC-2; breast 53% lifetime; upper GI/EAC 2-3x emerging; olaparib TBCRC048 82% ORR HIGHEST; FA-N biallelic.\n\n"
                "PALB2 / HBOC-2 CANCER SPECTRUM:\n"
                "  Breast cancer (female): 53% lifetime = DOMINANT PALB2 risk.\n"
                "  Ovarian: 3-5x; pancreatic: 2-3x; upper GI/EAC: 2-3x EMERGING.\n"
                "  PALB2 LOF = HRD (BRCA2-bridge protein disrupted) -> cisplatin/platinum sensitivity.\n\n"
                "OLAPARIB TBCRC048 -- 82% ORR HIGHEST:\n"
                "  TBCRC048 trial: olaparib in PALB2 germline breast cancer.\n"
                "  82% ORR = HIGHEST reported PARP inhibitor ORR in germline breast cancer.\n"
                "  Olaparib mechanism: PALB2 LOF -> HRD -> PARP inhibitor synthetic lethality.\n\n"
                "FA-N BIALLELIC -- FANCONI ANEMIA TYPE N:\n"
                "  PALB2 = FANCN; biallelic = Fanconi Anemia type N.\n"
                "  FA-N: severe, similar to FA-D1 (BRCA2 biallelic).\n"
                "  FA-N malignancies: medulloblastoma, Wilms, leukaemia (childhood).\n"
                "  FA-N children: AVOID cross-linking agents.\n\n"
                "BRCA2-BRIDGE FUNCTION:\n"
                "  PALB2 bridges BRCA1 (upstream DSB sensing) to BRCA2 (RAD51 loading).\n"
                "  PALB2-null: equivalent HRD phenotype to BRCA2-null.\n\n"
                "CASCADE TESTING -- HEREDITARY ESOPHAGEAL CANCER PANEL:\n"
                "  RHBDF2 (primary): PPK + ESCC family history -> RHBDF2 sequencing; Lugol iodine annual.\n"
                "  FANCA: DEB test + FA panel in early-onset ESCC with BMF/radial ray anomalies.\n"
                "  TP53: p53 aberrant IHC on ESCC/EAC -> germline TP53; AVOID radiation.\n"
                "  CDKN2A: FAMM + ESCC/EAC -> CDKN2A sequencing; CDK4/6i targeted.\n"
                "  ATM/BRCA2/PALB2: HRD ESCC/EAC -> platinum/olaparib sensitivity.\n"
                "  MLH1/MMR: dMMR EAC/ESCC -> pembrolizumab MSI-H all histologies.\n\n"
                "TIER 1 -- MOST ACTIONABLE:\n"
                "  RHBDF2: Lugol iodine endoscopy annual; AVOID tobacco/alcohol absolutely.\n"
                "  FANCA: esophageal endoscopy from age 16yr; AVOID aldehyde absolutely.\n"
                "  TP53: WBMRI Toronto annually; AVOID radiation absolutely.\n\n"
                "TIER 2 -- DNA REPAIR (HRD):\n"
                "  BRCA2/PALB2/ATM: cisplatin/olaparib/ceralasertib sensitivity.\n"
                "  CDKN2A: CDK4/6 inhibitor trials (9p21 deletion).\n\n"
                "TIER 3 -- IMMUNE:\n"
                "  MLH1 dMMR: pembrolizumab MSI-H all histologies.\n"
                "  Aspirin CAPP2 50% CRC reduction LEVEL A Lynch."
            ),
        },
    ]

    return {
        "atlas":       "Hereditary-Esophageal-Cancer-Predisposition-Atlas",
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
              f"severe_n={row['severe_n']} ({row['severe_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term'][:80]}")
