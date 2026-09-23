#!/usr/bin/env python3
"""Hereditary-Pancreatic-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
BRCA2   (FANCD1 / RAD51 Loader; 3418aa; 13q12.3; AD LOF;
         HBOC-type2; pancreatic cancer 3-6% lifetime, ~30-40x RR;
         HRD cisplatin preferred over carboplatin; olaparib POLO trial HR 0.53;
         seed SEED_BASE+0) .
CDKN2A  (p16-INK4a / p14-ARF; 156aa; 9p21.3; AD LOF;
         FAMMM; pancreatic 15-17% lifetime HIGHEST SINGLE gene;
         annual MRI + EUS from age 40yr MANDATORY;
         seed SEED_BASE+1) .
ATM     (Ataxia-Telangiectasia Mutated; 3056aa; 11q22.3; AD LOF;
         A-T; pancreatic 5-10x RR; olaparib POLO HR 0.72;
         RT reduce 20-30% even heterozygotes;
         seed SEED_BASE+2) .
PALB2   (Partner and Localiser of BRCA2; 1186aa; 16p12.2; AD LOF;
         FANCN; pancreatic 2-4% lifetime emerging; olaparib POLO;
         seed SEED_BASE+3) .
STK11   (Serine-Threonine Kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome; pancreatic 11-36% lifetime HIGHEST SYNDROME;
         perioral pigmentation PATHOGNOMONIC;
         GI endoscopy 8yr MANDATORY; MRI/EUS 25yr MANDATORY;
         seed SEED_BASE+4) .
BRCA1   (RING E3 Ligase / FANCS; 1863aa; 17q21.31; AD LOF;
         HBOC-type1; pancreatic 2-3x RR (lower than BRCA2);
         HRD cisplatin preferred; olaparib POLO HR 0.82;
         seed SEED_BASE+5) .
MLH1    (MutL Homolog 1; 756aa; 3p22.2; AD LOF;
         Lynch Syndrome type 1; pancreatic 9-11x RR;
         MSI-H pembrolizumab FDA2017 tumor-agnostic;
         BRAF V600E absent confirms germline; MLPA MANDATORY;
         seed SEED_BASE+6) .
TP53    (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome; pancreatic 7-10x RR;
         AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annually;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3478-3485)
"""
import random

SEED_BASE = 3478

ATLAS_GENES = [
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "FANCD1-HR-Mediator-384kDa-RAD51-Loader-"
            "HBOC-Type2-Pancreatic-3-6pct-Lifetime-HIGHEST-"
            "HRD-Cisplatin-Preferred-Carboplatin-"
            "Olaparib-POLO-HR-0.53-Rucaparib-Niraparib-"
            "OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 pancreatic cancer molecular context: "
            "STRUCTURE: 3418 aa / 384 kDa; largest BRCA protein; "
            "8 BRC repeats (aa 1002-2085): RAD51 binding; "
            "OB folds (aa 2402-3190): ssDNA binding; "
            "MECHANISM: LOF -> HR deficiency (HRD) -> genomic instability -> PDAC; "
            "HRD signature (Sig 3); cisplatin exploits HRD cross-links -> DSBs; "
            "POLO trial: olaparib maintenance HR 0.53 vs placebo in germline BRCA1/2 mPDAC; "
            "CANCER RISKS: Pancreatic 3-6% lifetime (30-40x RR); Breast 69-72%; Ovarian 11-17%; "
            "KEY MANAGEMENT: Annual MRI pancreas from age 50yr; EUS alternating; "
            "Cisplatin preferred (HCRN-GI016); olaparib maintenance after platinum response"
        ),
        "key_mutations": [
            {"variant": "p.Glu1308Ter (c.3922G>T)", "protein_effect": "BRC repeat region truncation", "location": "Exon 11", "phenotype": "PDAC + HBOC; Ashkenazi-enriched"},
            {"variant": "c.5946delT (6174delT)", "protein_effect": "Frameshift complete LOF", "location": "Exon 11", "phenotype": "Ashkenazi Jewish founder; PDAC + HBOC"},
            {"variant": "p.Lys3326Ter (c.9976A>T)", "protein_effect": "C-terminal truncation near OB fold", "location": "Exon 27", "phenotype": "Lower penetrance; moderate PC risk"},
        ],
        "surveillance": [
            "Annual pancreatic MRI/MRCP from age 50yr (or 10yr before index case)",
            "EUS alternating with MRI every 6-12 months (specialist centres)",
            "Annual mammogram + MRI breast from age 30yr",
            "RRSO at 40-45yr (ovarian risk 11-17%)",
            "Germline BRCA2 testing: olaparib-eligible if mPDAC after platinum",
        ],
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4a-CDK4-6-Inhibitor-15kDa-p14-ARF-MDM2-Inhibitor-"
            "FAMMM-Familial-Atypical-Multiple-Mole-Melanoma-"
            "Pancreatic-15-17pct-Lifetime-HIGHEST-SINGLE-GENE-"
            "Annual-MRI-EUS-40yr-MANDATORY-"
            "Melanoma-Co-Predisposition-"
            "OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 15 kDa / 9p21.3 CDKN2A pancreatic cancer molecular context: "
            "STRUCTURE: 156 aa / 15 kDa; 4 ankyrin repeats; CDK4/6 inhibitor; "
            "p14-ARF (alternate reading frame): MDM2 inhibitor -> p53 stabilisation; "
            "MECHANISM: LOF -> CDK4/6 constitutive -> Rb hyperphosphorylation -> G1/S release -> PDAC; "
            "p14-ARF loss: MDM2 unrestrained -> p53 degradation; "
            "CDKN2A homozygous deletion most common somatic event in PDAC (~95%); "
            "CANCER RISKS: Pancreatic 15-17% lifetime (HIGHEST single gene); Melanoma 28-67%; "
            "KEY MANAGEMENT: Annual pancreatic MRI + EUS from age 40yr MANDATORY; "
            "Lifelong annual dermatological exam; photoprotection"
        ),
        "key_mutations": [
            {"variant": "p.Ala148Thr (c.442G>A)", "protein_effect": "Ankyrin repeat 3 disruption", "location": "Exon 2", "phenotype": "FAMMM; melanoma + pancreatic"},
            {"variant": "p.Gly101Trp (c.301G>T)", "protein_effect": "CDK4-binding surface disruption", "location": "Exon 2", "phenotype": "High-penetrance PDAC + melanoma"},
            {"variant": "p.Arg58Stop (c.172C>T)", "protein_effect": "Truncation complete LOF", "location": "Exon 1", "phenotype": "FAMMM; high melanoma + PC risk"},
        ],
        "surveillance": [
            "Annual pancreatic MRI/MRCP + EUS from age 40yr — MANDATORY",
            "Whole-body skin exam by dermatologist annually",
            "Full body photography baseline at first assessment",
            "CA 19-9 monitoring q6-monthly (adjunctive; low sensitivity/specificity alone)",
            "Family cascade testing for all FAMMM families",
        ],
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-LOF -- 3056aa -- "
            "PIK3-Like-Kinase-350kDa-DSB-Sensor-"
            "Ataxia-Telangiectasia-A-T-"
            "Pancreatic-5-10x-RR-2-3pct-Lifetime-"
            "Olaparib-POLO-Trial-HR-0.72-"
            "RT-Reduce-20-30pct-Even-Heterozygotes-Surgery-First-"
            "OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM pancreatic cancer molecular context: "
            "STRUCTURE: 3056 aa / 350 kDa; PI3K-like kinase family; "
            "FAT domain (aa 1960-2566): regulatory HEAT repeats; "
            "Kinase domain (aa 2712-2962); FATC (aa 2962-3056): redox sensor; "
            "MECHANISM: ATM senses DSBs -> autophosphorylation Ser1981 -> activates H2AX CHEK2 BRCA1; "
            "Moderate HRD vs BRCA2 (ATM = DDR kinase not direct HR mediator); "
            "CANCER RISKS: Pancreatic 5-10x RR (2-3% lifetime); Breast 2-4x RR; CLL elevated; "
            "RADIATION SENSITIVITY: heterozygotes RT dose reduce 20-30%; surgery-first preferred; "
            "KEY MANAGEMENT: Annual MRI from 50yr; olaparib after platinum (POLO HR 0.72)"
        ),
        "key_mutations": [
            {"variant": "p.Ile2991Val (c.8971A>G)", "protein_effect": "Kinase domain structural change", "location": "Exon 61", "phenotype": "Pancreatic + breast risk"},
            {"variant": "c.7271T>G (p.Val2424Gly)", "protein_effect": "FAT domain destabilisation", "location": "Exon 50", "phenotype": "Elevated PDAC + CLL risk"},
            {"variant": "p.Ser49Cys (c.146C>G)", "protein_effect": "FAT domain HEAT repeat disruption", "location": "Exon 3", "phenotype": "Moderate penetrance PDAC"},
        ],
        "surveillance": [
            "Annual pancreatic MRI/MRCP from age 50yr (or 10yr before index)",
            "Annual breast MRI from age 40yr (female carriers)",
            "Avoid high-dose RT; reduce 20-30% if RT mandatory",
            "Surgery-first strategy for resectable PDAC",
            "Olaparib consideration if mPDAC + platinum response (POLO HR 0.72)",
        ],
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "FANCN-WD40-BRCA1-BRCA2-Bridge-131kDa-"
            "Pancreatic-2-4pct-Lifetime-Emerging-"
            "HRD-Olaparib-POLO-TBCRC048-82pct-ORR-"
            "Breast-53pct-DOMINATES-"
            "OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 pancreatic cancer molecular context: "
            "STRUCTURE: 1186 aa / 131 kDa; WD40 domain (aa 853-1186); "
            "N-terminal coiled-coil (aa 1-45): BRCA1 binding; "
            "C-terminal WD40 (aa 853-1186): BRCA2 binding; "
            "PALB2 = partner and localiser of BRCA2 — bridges BRCA1-BRCA2; "
            "MECHANISM: LOF -> BRCA2 mislocalised -> impaired RAD51 loading -> HRD; "
            "HRD signature in PALB2-mutated PDAC similar to BRCA2; "
            "TBCRC048: olaparib 82% ORR in germline PALB2 mBC; "
            "CANCER RISKS: Breast 53% lifetime (DOMINATES); Pancreatic 2-4%; Ovarian 3-5%; "
            "KEY MANAGEMENT: Breast MRI from 30yr; Pancreatic MRI from 50yr; HRD testing mandatory"
        ),
        "key_mutations": [
            {"variant": "p.Leu939Trp (c.2816T>G)", "protein_effect": "WD40 domain structural disruption", "location": "Exon 10", "phenotype": "Breast + PDAC + ovarian"},
            {"variant": "c.3113+1G>A", "protein_effect": "Splice donor Exon 12 exon skipping", "location": "Intron 12", "phenotype": "FANCN; moderate PDAC risk"},
            {"variant": "p.Tyr551Stop (c.1653T>A)", "protein_effect": "Central domain truncation", "location": "Exon 6", "phenotype": "Breast-dominant; PDAC secondary"},
        ],
        "surveillance": [
            "Annual breast MRI from age 30yr (breast risk 53% dominates)",
            "Annual pancreatic MRI from age 50yr (or 10yr before index case)",
            "EUS every 1-2 years at specialist centres",
            "HRD testing of tumour at PDAC diagnosis",
            "Olaparib/niraparib consideration for mPDAC (TBCRC048 data)",
        ],
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-AMPK-Kinase-48kDa-"
            "Peutz-Jeghers-Syndrome-PJS-"
            "Pancreatic-11-36pct-Lifetime-HIGHEST-PC-SYNDROME-"
            "Perioral-Pigmentation-PATHOGNOMONIC-"
            "GI-Endoscopy-8yr-MANDATORY-MRI-EUS-25yr-MANDATORY-"
            "SCTAT-Ovarian-PATHOGNOMONIC-Adenoma-Malignum-Cervix-PATHOGNOMONIC-"
            "OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 pancreatic cancer molecular context: "
            "STRUCTURE: 433 aa / 48 kDa; serine-threonine kinase; "
            "Kinase domain (aa 44-309): catalytic DFG motif Asp194; "
            "STK11 activates AMPK -> mTORC1 inhibition -> cellular energy sensor; "
            "MECHANISM: LOF -> mTORC1 constitutive -> PI3K/AKT/mTOR -> PDAC; "
            "KRAS co-mutation accelerates PDAC in STK11-deficient background; "
            "CANCER RISKS: Pancreatic 11-36% lifetime HIGHEST SYNDROME; CRC 39%; Breast 32-54%; "
            "Gastric 29%; SCTAT ovarian PATHOGNOMONIC; Adenoma malignum cervix PATHOGNOMONIC; "
            "KEY MANAGEMENT: GI endoscopy from 8yr MANDATORY; MRI/EUS pancreas from 25yr MANDATORY"
        ),
        "key_mutations": [
            {"variant": "p.Phe354Leu (c.1060T>C)", "protein_effect": "Kinase domain catalytic disruption", "location": "Exon 7", "phenotype": "Classic PJS; highest PC risk"},
            {"variant": "p.Trp239Stop (c.717G>A)", "protein_effect": "Kinase domain truncation", "location": "Exon 5", "phenotype": "PJS full syndrome; PDAC 36% lifetime"},
            {"variant": "Large deletion exon 1-10", "protein_effect": "Complete gene LOF; MLPA detects", "location": "Multi-exon", "phenotype": "Severe PJS; early polyp onset"},
        ],
        "surveillance": [
            "GI endoscopy from age 8yr (gastric + small bowel + colon) — MANDATORY",
            "Annual MRI small bowel / video capsule from age 8yr",
            "Pancreatic MRI + EUS from age 25yr — MANDATORY annual",
            "Annual gynaecological exam + transvaginal US from age 18yr",
            "Annual breast MRI from age 25yr (breast risk 32-54%)",
        ],
    },
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "FANCS-RING-E3-Ligase-208kDa-HR-Scaffold-"
            "HBOC-Type1-Pancreatic-2-3x-Lower-Than-BRCA2-"
            "HRD-Cisplatin-Preferred-"
            "Olaparib-POLO-HR-0.82-Weaker-Than-BRCA2-"
            "OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 208 kDa / 17q21.31 BRCA1 pancreatic cancer molecular context: "
            "STRUCTURE: 1863 aa / 208 kDa; "
            "RING domain (aa 1-109): E3 ubiquitin ligase with BARD1; "
            "BRCT domains (aa 1646-1863): phosphopeptide-binding scaffold; "
            "BRCA1 vs BRCA2 IN PDAC: BRCA1 RR 2-3x (lower than BRCA2 30-40x); "
            "POLO trial: BRCA1 HR 0.82 (marginal) vs BRCA2 HR 0.53 (meaningful); "
            "BRCA1 = scaffold/signalling (indirect HRD); BRCA2 = direct RAD51 loader; "
            "CANCER RISKS: Breast 72% lifetime DOMINATES; Ovarian 39-44% RRSO 35-40yr; "
            "Pancreatic 1-2% lifetime (2-3x RR present but lower); "
            "KEY MANAGEMENT: Annual pancreatic MRI from 50yr; RRSO 35-40yr primary intervention"
        ),
        "key_mutations": [
            {"variant": "c.5266dupC (5382insC)", "protein_effect": "BRCT domain frameshift complete LOF", "location": "Exon 20", "phenotype": "Ashkenazi founder; breast/ovarian dominant; PDAC 2-3x"},
            {"variant": "c.68_69del (185delAG)", "protein_effect": "RING domain frameshift", "location": "Exon 2", "phenotype": "Ashkenazi founder; HBOC classic"},
            {"variant": "p.Arg1699Trp (c.5095C>T)", "protein_effect": "BRCT phosphopeptide binding disrupted", "location": "Exon 18", "phenotype": "BRCT variant; breast/ovarian + PDAC secondary"},
        ],
        "surveillance": [
            "Annual MRI pancreas from age 50yr (BRCA1 moderate PC risk — lower priority than BRCA2)",
            "Annual mammogram + breast MRI from age 30yr",
            "RRSO at 35-40yr (ovarian risk 39-44%)",
            "Olaparib maintenance for mPDAC if platinum-responsive (POLO HR 0.82 — weaker)",
            "Germline testing of all PDAC index cases for platinum eligibility",
        ],
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MutL-Homolog-1-85kDa-MMR-MutLalpha-Dimerises-PMS2-"
            "Lynch-Syndrome-Type1-"
            "Pancreatic-9-11x-RR-"
            "MSI-H-Pembrolizumab-FDA2017-Tumor-Agnostic-"
            "BRAF-V600E-Absent-Confirms-Germline-"
            "MLPA-MANDATORY-EPCAM-"
            "OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 pancreatic cancer molecular context: "
            "STRUCTURE: 756 aa / 85 kDa; MutL homolog; "
            "N-terminal ATPase (aa 1-340); C-terminal dimerisation (aa 500-756): PMS2 binding; "
            "MLH1-PMS2 = MutLalpha: endonuclease complex incises mismatch-containing strand; "
            "MECHANISM: MMR loss -> MSI-H -> hypermutation -> neoantigens -> immunotherapy-responsive; "
            "BRAF V600E gate: if present = somatic methylation (NOT Lynch germline); "
            "Germline MLH1: BRAF V600E absent + somatic methylation absent + MLPA mandatory; "
            "CANCER RISKS: CRC 80-85% Lynch1 DOMINANT; Endometrial 40-60%; "
            "Pancreatic 9-11x RR (3.7% lifetime); Ovarian 8-13% (endometrioid); "
            "KEY MANAGEMENT: Colonoscopy q1-2yr from 25yr; aspirin 600mg CAPP2; pembrolizumab MSI-H"
        ),
        "key_mutations": [
            {"variant": "p.Val384Asp (c.1151T>A)", "protein_effect": "C-terminal dimerisation domain disruption", "location": "Exon 11", "phenotype": "Lynch1; CRC/endometrial dominant; PDAC 9-11x"},
            {"variant": "Large deletion exon 16 (Scandinavian founder)", "protein_effect": "Frameshift complete LOF; MLPA detects", "location": "Exon 16", "phenotype": "Scandinavian Lynch1 founder"},
            {"variant": "p.Arg659Stop (c.1975C>T)", "protein_effect": "C-terminal truncation MutLalpha lost", "location": "Exon 17", "phenotype": "Lynch1; elevated CRC + PC risk"},
        ],
        "surveillance": [
            "Colonoscopy every 1-2yr from age 25yr",
            "Annual pancreatic MRI from age 50yr in Lynch1 families with PDAC history",
            "Annual endometrial biopsy from age 35yr (female carriers)",
            "Aspirin 600mg CAPP2 regimen (Lynch CRC risk reduction 50%)",
            "Pembrolizumab for MSI-H PDAC (FDA2017 tumor-agnostic)",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-Guardian-Genome-"
            "Li-Fraumeni-Syndrome-LFS-"
            "Pancreatic-7-10x-RR-"
            "AVOID-RADIATION-ABSOLUTELY-"
            "WB-MRI-Toronto-Protocol-Annually-"
            "OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 pancreatic cancer molecular context: "
            "STRUCTURE: 393 aa / 43 kDa; "
            "N-terminal TAD1 (aa 1-40): MDM2 binding; TAD2 (aa 40-67): BRCA1 interaction; "
            "DNA-binding domain (aa 102-292): HOTSPOT mutations; "
            "Tetramerisation (aa 323-356); C-terminal regulatory (aa 364-393); "
            "MECHANISM: TP53 activated downstream ATM/CHK2 at DSBs; "
            "LFS germline LOF + somatic LOH -> biallelic p53 loss in PDAC; "
            "AVOID RADIATION ABSOLUTELY: RT-induced second primaries 20-fold in LFS; "
            "CANCER RISKS: Pancreatic 7-10x RR; Sarcoma 15-30%; Brain tumour 5-10%; "
            "Breast 30-50% early onset; Adrenocortical 80% childhood ACC; "
            "KEY MANAGEMENT: WB-MRI Toronto Protocol annually; AVOID ALL RADIATION"
        ),
        "key_mutations": [
            {"variant": "p.Arg248Trp (c.742C>T)", "protein_effect": "DNA-binding hotspot dominant negative GOF", "location": "Exon 7", "phenotype": "LFS high penetrance; PDAC + multiple primaries"},
            {"variant": "p.Arg175His (c.524G>A)", "protein_effect": "Structural hotspot dominant negative", "location": "Exon 5", "phenotype": "LFS classic; sarcoma + PDAC"},
            {"variant": "p.Arg273His (c.818G>A)", "protein_effect": "Contact hotspot DNA-binding lost", "location": "Exon 8", "phenotype": "LFS; GOF oncogenic properties"},
        ],
        "surveillance": [
            "Annual whole-body MRI (Toronto Protocol) — REPLACES CT in LFS",
            "Pancreatic MRI from age 40yr (or 10yr before youngest index case)",
            "Annual brain MRI from age 18yr",
            "Annual abdominal ultrasound in childhood (adrenocortical)",
            "AVOID ALL RADIATION in clinical and surveillance decisions ABSOLUTELY",
        ],
    },
]


def _patients_for_gene(gene_dict: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    pts = []
    for i in range(40):
        age = int(rng.gauss(
            {"BRCA2": 57, "CDKN2A": 54, "ATM": 59, "PALB2": 56,
             "STK11": 48, "BRCA1": 60, "MLH1": 61, "TP53": 52}[gene],
            {"BRCA2": 10, "CDKN2A": 10, "ATM": 10, "PALB2": 10,
             "STK11": 9,  "BRCA1": 10, "MLH1": 10, "TP53": 12}[gene]
        ))
        age = max(25, min(80, age))

        pc_rate = {"BRCA2": 0.55, "CDKN2A": 0.60, "ATM": 0.40, "PALB2": 0.38,
                   "STK11": 0.70, "BRCA1": 0.25, "MLH1": 0.42, "TP53": 0.45}[gene]
        has_pdac = rng.random() < pc_rate

        breast_rate = {"BRCA2": 0.70, "CDKN2A": 0.0,  "ATM": 0.40, "PALB2": 0.60,
                       "STK11": 0.45, "BRCA1": 0.75, "MLH1": 0.0,  "TP53": 0.50}[gene]
        has_breast = rng.random() < breast_rate

        ovarian_rate = {"BRCA2": 0.15, "CDKN2A": 0.0,  "ATM": 0.0,  "PALB2": 0.05,
                        "STK11": 0.25, "BRCA1": 0.42, "MLH1": 0.12, "TP53": 0.05}[gene]
        has_ovarian = rng.random() < ovarian_rate

        melanoma_rate = {"BRCA2": 0.02, "CDKN2A": 0.55, "ATM": 0.0,  "PALB2": 0.0,
                         "STK11": 0.0,  "BRCA1": 0.02, "MLH1": 0.0,  "TP53": 0.05}[gene]
        has_melanoma = rng.random() < melanoma_rate

        crc_rate = {"BRCA2": 0.05, "CDKN2A": 0.05, "ATM": 0.05, "PALB2": 0.03,
                    "STK11": 0.39, "BRCA1": 0.03,  "MLH1": 0.80, "TP53": 0.05}[gene]
        has_crc = rng.random() < crc_rate

        msi_h = gene in ("MLH1",) and has_pdac and rng.random() < 0.85
        hrd_flag = gene in ("BRCA2", "BRCA1", "PALB2", "ATM") and has_pdac and rng.random() < 0.80
        radiation_ci = gene in ("TP53",)
        radiation_sensitivity = gene in ("ATM",)
        pjs_polyps = gene == "STK11" and rng.random() < 0.92

        pts.append({
            "gene": gene,
            "age": age,
            "has_pdac": has_pdac,
            "has_breast": has_breast,
            "has_ovarian": has_ovarian,
            "has_melanoma": has_melanoma,
            "has_crc": has_crc,
            "msi_h": msi_h,
            "hrd_flag": hrd_flag,
            "radiation_ci": radiation_ci,
            "radiation_sensitivity": radiation_sensitivity,
            "pjs_polyps": pjs_polyps,
        })
    return pts


def _all_patients() -> list:
    all_pts = []
    for idx, gene_dict in enumerate(ATLAS_GENES):
        all_pts.extend(_patients_for_gene(gene_dict, SEED_BASE + idx))
    return all_pts


# ── API generators ────────────────────────────────────────────────────────────
def generate_overview() -> dict:
    pts = _all_patients()
    gene_counts: dict = {}
    for p in pts:
        g = p["gene"]
        if g not in gene_counts:
            gene_counts[g] = {
                "gene": g, "n": 0, "pdac": 0, "breast": 0, "ovarian": 0,
                "melanoma": 0, "crc": 0, "msi_h": 0, "hrd": 0, "mean_age": 0
            }
        gene_counts[g]["n"] += 1
        if p["has_pdac"]:     gene_counts[g]["pdac"] += 1
        if p["has_breast"]:   gene_counts[g]["breast"] += 1
        if p["has_ovarian"]:  gene_counts[g]["ovarian"] += 1
        if p["has_melanoma"]: gene_counts[g]["melanoma"] += 1
        if p["has_crc"]:      gene_counts[g]["crc"] += 1
        if p["msi_h"]:        gene_counts[g]["msi_h"] += 1
        if p["hrd_flag"]:     gene_counts[g]["hrd"] += 1
        gene_counts[g]["mean_age"] += p["age"]

    for g in gene_counts:
        n = gene_counts[g]["n"]
        gene_counts[g]["mean_age"] = round(gene_counts[g]["mean_age"] / n, 1)
        gene_counts[g]["pdac_pct"] = round(100 * gene_counts[g]["pdac"] / n, 1)
        gene_counts[g]["hrd_pct"]  = round(100 * gene_counts[g]["hrd"] / n, 1)

    pdac_total  = sum(1 for p in pts if p["has_pdac"])
    hrd_total   = sum(1 for p in pts if p["hrd_flag"])
    msi_total   = sum(1 for p in pts if p["msi_h"])
    rt_ci_total = sum(1 for p in pts if p["radiation_ci"])
    rt_s_total  = sum(1 for p in pts if p["radiation_sensitivity"])
    pjs_total   = sum(1 for p in pts if p["pjs_polyps"])

    return {
        "atlas":           "Hereditary-Pancreatic-Cancer-Predisposition-Atlas",
        "atlas_id":        "hereditary-pancreatic-cancer-predisposition-atlas",
        "subtitle":        "Complete 8-Gene BRCA2-CDKN2A-ATM-PALB2-STK11-BRCA1-MLH1-TP53 Reference",
        "total_patients":  len(pts),
        "gene_cohorts":    len(ATLAS_GENES),
        "seeds":           f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "pdac_cases":      pdac_total,
        "pdac_rate_pct":   round(100 * pdac_total / len(pts), 1),
        "hrd_cases":       hrd_total,
        "hrd_rate_pct":    round(100 * hrd_total / len(pts), 1),
        "msi_h_cases":     msi_total,
        "radiation_ci_cases": rt_ci_total,
        "radiation_sensitivity_cases": rt_s_total,
        "pjs_polyp_cases": pjs_total,
        "gene_summary":    list(gene_counts.values()),
        "key_clinical_rules": [
            "BRCA2 (HBOC-2): pancreatic 3-6% lifetime (30-40x RR) — highest BRCA gene for PC; cisplatin preferred (HRD cisplatin > oxaliplatin); olaparib POLO HR 0.53 after platinum response",
            "CDKN2A (FAMMM): pancreatic 15-17% lifetime — HIGHEST single gene hereditary PC; annual MRI + EUS from age 40yr MANDATORY; melanoma co-predisposition 28-67%",
            "STK11 (Peutz-Jeghers): pancreatic 11-36% lifetime — ABSOLUTE HIGHEST hereditary PC syndrome; perioral pigmentation PATHOGNOMONIC; GI endoscopy from 8yr MANDATORY; MRI/EUS pancreas 25yr MANDATORY",
            "ATM: olaparib POLO HR 0.72 (modest); RT reduce 20-30% even heterozygotes; surgery-first preferred for resectable PDAC",
            "MLH1 (Lynch1): MSI-H PDAC pembrolizumab FDA2017 tumor-agnostic; BRAF V600E absent confirms germline; MLPA MANDATORY (large deletions 20-40%)",
            "TP53 (Li-Fraumeni): AVOID RADIATION ABSOLUTELY; WB-MRI Toronto Protocol annually; secondary malignancy 20-fold post-RT; pancreatic 7-10x RR",
        ],
        "pancreatic_risk_hierarchy": {
            "STK11_PJS":    "11-36% lifetime (HIGHEST syndrome)",
            "CDKN2A_FAMMM": "15-17% lifetime (HIGHEST single gene)",
            "BRCA2_HBOC2":  "3-6% lifetime (30-40x RR; HRD cisplatin/olaparib)",
            "MLH1_Lynch1":  "3.7% lifetime (9-11x RR; MSI-H pembrolizumab)",
            "ATM":          "2-3% lifetime (5-10x RR; moderate HRD; RT-sensitive)",
            "PALB2_FANCN":  "2-4% lifetime (emerging HRD; olaparib TBCRC048)",
            "TP53_LFS":     "7-10x RR (RT absolutely CI; WB-MRI mandatory)",
            "BRCA1_HBOC1":  "2-3x RR (lower than BRCA2; HRD scaffold)",
        },
    }


def generate_breakdown() -> dict:
    pts = _all_patients()
    per_gene = []
    for gene_dict in ATLAS_GENES:
        gene = gene_dict["gene"]
        gpts = [p for p in pts if p["gene"] == gene]
        n = len(gpts)
        pdac_n = sum(1 for p in gpts if p["has_pdac"])
        hrd_n  = sum(1 for p in gpts if p["hrd_flag"])
        msi_n  = sum(1 for p in gpts if p["msi_h"])
        pjs_n  = sum(1 for p in gpts if p["pjs_polyps"])
        mean_age = round(sum(p["age"] for p in gpts) / n, 1)

        key_avoid = {
            "BRCA2": "Carboplatin if cisplatin available (HRD cisplatin > oxaliplatin data)",
            "CDKN2A": "Delayed pancreatic surveillance — 40yr is the mandatory start",
            "ATM": "High-dose RT — reduce 20-30% even heterozygotes",
            "PALB2": "Skipping HRD testing — olaparib-eligible if platinum-responsive",
            "STK11": "Delayed GI surveillance — endoscopy from 8yr; intussusception emergency",
            "BRCA1": "Substituting oxaliplatin for cisplatin (cisplatin preferred in HRD BRCA1)",
            "MLH1": "Missing MSI-H testing on PDAC — pembrolizumab opportunity lost",
            "TP53": "RADIATION ABSOLUTELY — 20-fold secondary malignancy risk in LFS",
        }[gene]

        key_rule = {
            "BRCA2": "Cisplatin-gemcitabine first-line; olaparib maintenance if platinum response (POLO)",
            "CDKN2A": "Annual MRI + EUS from age 40yr MANDATORY; dermatology annually",
            "ATM": "Surgery-first for resectable PDAC; olaparib after platinum (POLO HR 0.72)",
            "PALB2": "HRD testing mandatory at PDAC diagnosis; olaparib if HRD-positive (TBCRC048)",
            "STK11": "GI endoscopy from 8yr MANDATORY; MRI/EUS pancreas from 25yr MANDATORY annually",
            "BRCA1": "Annual pancreatic MRI from age 50yr; RRSO 35-40yr dominates management",
            "MLH1": "MSI testing on ALL PDAC; pembrolizumab FDA2017 tumor-agnostic for MSI-H",
            "TP53": "WB-MRI Toronto Protocol annually; AVOID ALL RADIATION ABSOLUTELY",
        }[gene]

        per_gene.append({
            "gene": gene,
            "syndrome": {
                "BRCA2": "HBOC Type-2 (FANCD1/RAD51 Loader)",
                "CDKN2A": "FAMMM (Familial Atypical Multiple Mole Melanoma)",
                "ATM": "Ataxia-Telangiectasia (A-T; DDR kinase)",
                "PALB2": "FANCN (BRCA1-BRCA2 Bridge)",
                "STK11": "Peutz-Jeghers Syndrome (PJS)",
                "BRCA1": "HBOC Type-1 (FANCS/RING E3 Ligase)",
                "MLH1": "Lynch Syndrome Type-1 (MutLα)",
                "TP53": "Li-Fraumeni Syndrome (LFS)",
            }[gene],
            "n": n,
            "pdac_n": pdac_n,
            "pdac_pct": round(100 * pdac_n / n, 1),
            "hrd_n": hrd_n,
            "hrd_pct": round(100 * hrd_n / n, 1),
            "msi_h_n": msi_n,
            "pjs_polyp_n": pjs_n,
            "mean_age": mean_age,
            "key_avoid": key_avoid,
            "key_rule": key_rule,
        })

    brca1_vs_brca2 = {
        "BRCA2_pancreatic_RR":       "30-40x (3-6% lifetime) — HIGHER and more actionable",
        "BRCA1_pancreatic_RR":       "2-3x (1-2% lifetime) — LOWER; HRD indirect",
        "BRCA2_POLO_HR":             "0.53 (clinically meaningful maintenance benefit)",
        "BRCA1_POLO_HR":             "0.82 (marginal; BRCA1 = scaffold not direct HR mediator)",
        "cisplatin_preference":      "Both benefit from cisplatin; BRCA2 evidence stronger",
        "clinical_bottom_line":      "BRCA2 = primary actionable target for hereditary PDAC",
    }

    hrd_vs_msi = {
        "HRD_genes":        "BRCA2 / BRCA1 / PALB2 / ATM — platinum + PARPi strategy",
        "MSI_H_gene":       "MLH1 (Lynch1) — pembrolizumab tumor-agnostic FDA2017",
        "HRD_testing":      "Tumour genomic scars (LOH, TAI, LST) + germline BRCA2",
        "MSI_H_testing":    "IHC MMR proteins (MLH1/PMS2 loss) + PCR MSI panel",
        "mutual_exclusivity": "HRD and MSI-H rarely co-occur in same PDAC",
        "clinical_implication": "Test ALL PDAC: germline panel + tumour MSI + tumour HRD score",
    }

    stk11_perioral = {
        "perioral_pigmentation": "Lips + buccal mucosa + digits PATHOGNOMONIC — diagnoses PJS",
        "appearance":            "1-5yr childhood; fade with age (mucosal persist)",
        "GI_surveillance_start": "8yr — gastric + small bowel capsule + colonoscopy MANDATORY",
        "pancreatic_start":      "25yr — annual MRI + EUS MANDATORY",
        "SCTAT_ovarian":         "PATHOGNOMONIC — small multilobular gonadal tumour in PJS females",
        "adenoma_malignum":      "PATHOGNOMONIC cervix — villoglandular adenoCA; Ki67/CEA IHC",
    }

    return {
        "per_gene":                  per_gene,
        "brca1_vs_brca2_comparison": brca1_vs_brca2,
        "hrd_vs_msi_h_treatment":    hrd_vs_msi,
        "stk11_pjs_key_rules":       stk11_perioral,
    }


def generate_definitions() -> dict:
    return {
        "genes": [
            {
                "gene": gd["gene"],
                "full_name": {
                    "BRCA2": "Breast Cancer Susceptibility 2 (FANCD1 / RAD51 Loader)",
                    "CDKN2A": "Cyclin-Dependent Kinase Inhibitor 2A (p16-INK4a / p14-ARF)",
                    "ATM": "Ataxia-Telangiectasia Mutated (PIK3-like DSB sensor kinase)",
                    "PALB2": "Partner and Localiser of BRCA2 (FANCN / WD40 scaffold)",
                    "STK11": "Serine-Threonine Kinase 11 / LKB1 (AMPK activator)",
                    "BRCA1": "Breast Cancer Susceptibility 1 (FANCS / RING E3 Ligase)",
                    "MLH1": "MutL Homolog 1 (MutLalpha MMR endonuclease scaffold)",
                    "TP53": "Tumour Protein p53 (Guardian of the Genome)",
                }[gd["gene"]],
                "protein_context": gd["protein_size"],
                "variants": gd["key_mutations"],
                "surveillance": gd["surveillance"],
            }
            for gd in ATLAS_GENES
        ],
        "key_clinical_concepts": {
            "POLO_trial_olaparib": (
                "POLO trial: germline BRCA1/2 mPDAC after >= 16 weeks platinum maintained on olaparib "
                "achieved HR 0.53 vs placebo (PFS). FDA approved 2019. BRCA2 HR 0.53 (meaningful); "
                "BRCA1 HR 0.82 (marginal — scaffold not direct HR mediator). "
                "PARPi trapping mechanism: PARP1 trapped on DNA -> collapsed replication forks -> lethal DSBs in HRD cells."
            ),
            "CDKN2A_FAMMM_dual_protein": (
                "CDKN2A locus encodes TWO proteins: p16-INK4a (CDK4/6 inhibitor; G1/S arrest) "
                "and p14-ARF (MDM2 inhibitor; p53 stabilisation) from alternate reading frames. "
                "FAMMM (familial atypical multiple mole melanoma): annual WB skin exam + "
                "annual pancreatic MRI + EUS from age 40yr MANDATORY. "
                "PC lifetime risk 15-17% — HIGHEST single gene. Melanoma 28-67% lifetime."
            ),
            "STK11_PJS_highest_syndrome": (
                "Peutz-Jeghers syndrome (STK11/LKB1 AD LOF): pancreatic 11-36% — "
                "HIGHEST of all hereditary PC syndromes. "
                "Perioral pigmentation (lips/buccal mucosa/digits) PATHOGNOMONIC — appears 1-5yr. "
                "GI endoscopy from 8yr MANDATORY. MRI+EUS pancreas from 25yr MANDATORY. "
                "SCTAT (sex-cord tumour annular tubules) PATHOGNOMONIC in PJS females. "
                "Adenoma malignum cervix PATHOGNOMONIC (Ki67/CEA IHC)."
            ),
            "ATM_radiation_sensitivity": (
                "ATM heterozygous carriers: ~2-fold increased radiation sensitivity. "
                "Reduce RT dose 20-30% if RT absolutely necessary. Surgery-first preferred. "
                "POLO trial: HR 0.72 (modest — ATM = DDR kinase, not direct HR mediator like BRCA2)."
            ),
            "MLH1_BRAF_gate": (
                "MLH1 somatic promoter methylation -> sporadic MSI-H (NOT Lynch). "
                "BRAF V600E present -> somatic methylation (acquired) -> NOT Lynch germline. "
                "Germline MLH1: BRAF V600E absent + MLH1 protein IHC loss + MLPA mandatory. "
                "MSI-H PDAC: 1-2% of all PDAC; pembrolizumab FDA2017 tumor-agnostic."
            ),
            "TP53_LFS_radiation_rule": (
                "LFS (TP53 AD LOF): AVOID ALL RADIATION ABSOLUTELY. "
                "RT-induced second primaries 20-fold in LFS. "
                "WB-MRI Toronto Protocol replaces CT for annual surveillance. "
                "Pancreatic MRI from age 40yr. Pancreatic 7-10x RR."
            ),
            "HRD_cisplatin_preference": (
                "In BRCA2/BRCA1/PALB2 germline PDAC: cisplatin preferred over oxaliplatin. "
                "Cisplatin = intrastrand (GpG) + interstrand cross-links requiring HR repair. "
                "Evidence strongest for cisplatin-gemcitabine in germline HRD PDAC. "
                "Olaparib: after >= 16 weeks platinum stabilisation/response (POLO criteria)."
            ),
        },
        "abbreviations": {
            "PDAC": "Pancreatic Ductal Adenocarcinoma",
            "HRD": "Homologous Recombination Deficiency",
            "MSI-H": "Microsatellite Instability-High",
            "PJS": "Peutz-Jeghers Syndrome",
            "FAMMM": "Familial Atypical Multiple Mole Melanoma",
            "LFS": "Li-Fraumeni Syndrome",
            "HBOC": "Hereditary Breast and Ovarian Cancer",
            "MMR": "Mismatch Repair",
            "DSB": "Double-Strand Break",
            "HR": "Hazard Ratio (POLO) / Homologous Recombination (HRD)",
            "POLO": "Pancreatic Olaparib trial (germline BRCA1/2 mPDAC maintenance)",
            "RRSO": "Risk-Reducing Salpingo-Oophorectomy",
            "EUS": "Endoscopic Ultrasound",
            "MRCP": "Magnetic Resonance Cholangiopancreatography",
            "SCTAT": "Sex-Cord Tumour With Annular Tubules (STK11-PJS PATHOGNOMONIC)",
            "WB-MRI": "Whole-Body MRI (Toronto Protocol; LFS/TP53)",
            "PARPi": "PARP Inhibitor (olaparib, niraparib, rucaparib)",
            "TBCRC048": "Translational Breast Cancer Research Consortium 048 (PALB2 olaparib trial; 82% ORR)",
        },
    }
