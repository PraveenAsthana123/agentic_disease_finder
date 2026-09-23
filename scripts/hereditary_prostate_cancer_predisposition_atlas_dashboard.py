#!/usr/bin/env python3
"""Hereditary-Prostate-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
BRCA2   (FANCD1 / RAD51 Loader; 3418aa; 13q12.3; AD LOF;
         HBOC-type2; prostate cancer 15% in aggressive/metastatic;
         olaparib PROfound HR 0.34; rucaparib FDA2020;
         seed SEED_BASE+0) .
BRCA1   (FANCS / RING E3 Ligase; 1863aa; 17q21.31; AD LOF;
         HBOC-type1; prostate cancer 2-3x RR;
         HRD; olaparib PROfound HR 0.35;
         seed SEED_BASE+1) .
ATM     (ATM kinase; 3056aa; 11q22.3; AD LOF;
         A-T monoallelic; prostate 4-8x HIGH GRADE;
         olaparib PROfound Cohort B; RT reduce 20-30%;
         seed SEED_BASE+2) .
PALB2   (BRCA2 anchor; 1186aa; 16p12.2; AD LOF;
         FANCN; prostate 2-3x emerging; HRD olaparib;
         seed SEED_BASE+3) .
MLH1    (MutLalpha; 756aa; 3p22.2; AD LOF;
         Lynch Syndrome type 1; prostate 3-8x RR;
         MSI-H pembrolizumab FDA2017;
         seed SEED_BASE+4) .
MSH2    (MutSalpha; 936aa; 2p21; AD LOF;
         Lynch Syndrome type 2; prostate HIGHEST Lynch 11-14x;
         EPCAM silencing MLPA MANDATORY;
         seed SEED_BASE+5) .
HOXB13  (AR coregulator; 283aa; 17q21.2; AD LOF;
         G84E N-European founder 3.5%; prostate 3-6x;
         PROSTATE-SPECIFIC gene;
         seed SEED_BASE+6) .
CHEK2   (ATM effector kinase; 543aa; 22q12.1; AD LOF;
         moderate-risk DDR; prostate 2-3x;
         I157T eastern; 1100delC western founder;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3486-3493)
"""
import random

SEED_BASE = 3486

ATLAS_GENES = [
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "FANCD1-HR-Mediator-384kDa-RAD51-Loader-"
            "HBOC-Type2-Prostate-15pct-Aggressive-Metastatic-"
            "Olaparib-PROfound-HR-0.34-rucaparib-FDA2020-"
            "PSA-Screening-40yr-MANDATORY-IMPACT-Trial-"
            "OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 prostate cancer molecular context: "
            "STRUCTURE: 3418 aa / 384 kDa; largest BRCA protein; "
            "8 BRC repeats (aa 1002-2085): RAD51 binding; "
            "OB folds (aa 2402-3190): ssDNA binding; "
            "MECHANISM: LOF -> HR deficiency (HRD) -> genomic instability -> aggressive PCa; "
            "HRD signature (Sig 3); BRCA2 carriers: higher Gleason grade; higher stage at diagnosis; "
            "PROfound trial: olaparib vs enzalutamide/abiraterone in mCRPC; "
            "Cohort A (BRCA2/BRCA1): HR 0.34 for rPFS; rucaparib also FDA2020; "
            "CANCER RISKS: ~15% of metastatic mCRPC carry BRCA2 (germline or somatic); "
            "Prostate RR 3-8x (lifetime risk elevated vs population); "
            "Breast 69-72% (dominates clinical management); Ovarian 11-17%; "
            "KEY MANAGEMENT: PSA + DRE from age 40yr IMPACT trial; "
            "MRI prostate if PSA rising; olaparib/rucaparib if mCRPC + HRD"
        ),
        "key_mutations": [
            {"variant": "p.Glu1308Ter (c.3922G>T)", "protein_effect": "BRC repeat region truncation", "location": "Exon 11", "phenotype": "HBOC + aggressive PCa; Ashkenazi-enriched"},
            {"variant": "c.5946delT (6174delT)", "protein_effect": "Frameshift complete LOF", "location": "Exon 11", "phenotype": "Ashkenazi Jewish founder; prostate + breast/ovarian"},
            {"variant": "p.Lys3326Ter (c.9976A>T)", "protein_effect": "C-terminal near-OB fold truncation", "location": "Exon 27", "phenotype": "Lower penetrance; PCa risk retained"},
        ],
        "surveillance": [
            "PSA + DRE annually from age 40yr (BRCA2 carriers — IMPACT trial protocol)",
            "Annual MRI prostate if PSA >3 ng/mL or rising rapidly",
            "Prostate biopsy threshold lowered: PSA ≥3 in BRCA2 (IMPACT data)",
            "Annual mammogram + breast MRI from age 30yr",
            "RRSO at 40-45yr (ovarian risk 11-17%)",
        ],
    },
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "FANCS-RING-E3-Ligase-208kDa-HR-Scaffold-"
            "HBOC-Type1-Prostate-2-3x-RR-"
            "Olaparib-PROfound-Cohort-A-HR-0.35-"
            "Breast-72pct-Ovarian-39-44pct-DOMINATES-"
            "OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 208 kDa / 17q21.31 BRCA1 prostate cancer molecular context: "
            "STRUCTURE: 1863 aa / 208 kDa; "
            "RING domain (aa 1-109): E3 ubiquitin ligase with BARD1; "
            "BRCT domains (aa 1646-1863): phosphopeptide-binding scaffold; "
            "BRCA1 vs BRCA2 IN PCa: BRCA1 RR 2-3x (moderate); BRCA2 RR 3-8x (higher); "
            "PROfound trial Cohort A: BRCA1 HR 0.35 (similar to BRCA2 HR 0.34 — olaparib works); "
            "BRCA1 = scaffold/signalling; BRCA2 = direct RAD51 loader; "
            "CANCER RISKS: Breast 72% lifetime DOMINATES; Ovarian 39-44% — RRSO 35-40yr; "
            "Prostate 2-3x RR (present but lower priority than breast/ovarian); "
            "KEY MANAGEMENT: Annual breast MRI from 30yr; RRSO 35-40yr; PSA from 45yr; olaparib if mCRPC"
        ),
        "key_mutations": [
            {"variant": "c.5266dupC (5382insC)", "protein_effect": "BRCT domain frameshift complete LOF", "location": "Exon 20", "phenotype": "Ashkenazi founder; breast/ovarian dominant; PCa 2-3x"},
            {"variant": "c.68_69del (185delAG)", "protein_effect": "RING domain frameshift", "location": "Exon 2", "phenotype": "Ashkenazi founder; HBOC classic"},
            {"variant": "p.Arg1699Trp (c.5095C>T)", "protein_effect": "BRCT phosphopeptide binding disrupted", "location": "Exon 18", "phenotype": "BRCT variant; breast/ovarian primary; PCa secondary"},
        ],
        "surveillance": [
            "PSA + DRE annually from age 45yr (BRCA1 carriers — IMPACT trial)",
            "Annual mammogram + breast MRI from age 30yr (breast 72% dominates)",
            "RRSO at 35-40yr (ovarian risk 39-44%)",
            "Olaparib consideration for mCRPC with HRD (PROfound Cohort A HR 0.35)",
            "Germline testing of all mCRPC patients for treatment selection",
        ],
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-LOF -- 3056aa -- "
            "PIK3-Like-Kinase-350kDa-DSB-Sensor-"
            "Ataxia-Telangiectasia-Monoallelic-"
            "Prostate-4-8x-RR-HIGH-GRADE-Gleason-8-10-"
            "Olaparib-PROfound-Cohort-B-HR-0.45-"
            "RT-Reduce-20-30pct-Even-Heterozygotes-"
            "OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM prostate cancer molecular context: "
            "STRUCTURE: 3056 aa / 350 kDa; PI3K-like kinase family; "
            "FAT domain (aa 1960-2566): regulatory HEAT repeats; "
            "Kinase domain (aa 2712-2962); FATC (aa 2962-3056): redox sensor; "
            "MECHANISM: ATM senses DSBs -> autophosphorylation Ser1981 -> activates H2AX/CHEK2/BRCA1; "
            "Moderate HRD — ATM monoallelic somatic loss second hit required; "
            "ATM PCa: HIGH GRADE predilection — Gleason 8-10 association well-documented; "
            "PROfound trial Cohort B (ATM + 12 other HRR genes): HR 0.45 (olaparib); "
            "RADIATION SENSITIVITY: heterozygotes must have RT dose reduced 20-30%; "
            "CANCER RISKS: Prostate 4-8x RR (HIGH GRADE); Breast 2-4x; CLL elevated; "
            "KEY MANAGEMENT: PSA from 40yr; early biopsy; avoid high-dose RT"
        ),
        "key_mutations": [
            {"variant": "p.Val2424Gly (c.7271T>G)", "protein_effect": "FAT domain destabilisation", "location": "Exon 50", "phenotype": "High-grade PCa + breast + CLL"},
            {"variant": "p.Ile2991Val (c.8971A>G)", "protein_effect": "Kinase domain structural change", "location": "Exon 61", "phenotype": "Moderate PCa + breast risk"},
            {"variant": "p.Arg2032His (c.6095G>A)", "protein_effect": "Kinase domain catalytic disruption", "location": "Exon 42", "phenotype": "High-grade PCa; RT-sensitive"},
        ],
        "surveillance": [
            "PSA + DRE annually from age 40yr",
            "Low threshold for MRI prostate and biopsy (high-grade Gleason 8-10 predilection)",
            "Avoid high-dose RT; reduce 20-30% if RT mandatory even as heterozygote",
            "Olaparib consideration in mCRPC (PROfound Cohort B HR 0.45)",
            "Annual breast MRI from age 40yr (female carriers: 2-4x RR)",
        ],
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "FANCN-WD40-BRCA1-BRCA2-Bridge-131kDa-"
            "Prostate-2-3x-RR-Emerging-"
            "HRD-Olaparib-Emerging-Evidence-"
            "Breast-53pct-DOMINATES-Clinical-Management-"
            "OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 prostate cancer molecular context: "
            "STRUCTURE: 1186 aa / 131 kDa; WD40 domain (aa 853-1186); "
            "N-terminal coiled-coil (aa 1-45): BRCA1 binding; "
            "C-terminal WD40 (aa 853-1186): BRCA2 binding; "
            "PALB2 = partner and localiser of BRCA2 — bridges BRCA1-BRCA2 in HR; "
            "MECHANISM: LOF -> BRCA2 mislocalised -> impaired RAD51 loading -> HRD; "
            "PALB2 PCa: emerging evidence; HRD signature; olaparib rationale from HRD mechanism; "
            "CANCER RISKS: Breast 53% lifetime DOMINATES; Prostate 2-3x RR; Ovarian 3-5%; "
            "KEY MANAGEMENT: Breast MRI from 30yr primary; PSA from 45yr; HRD testing at mCRPC diagnosis"
        ),
        "key_mutations": [
            {"variant": "p.Leu939Trp (c.2816T>G)", "protein_effect": "WD40 domain structural disruption", "location": "Exon 10", "phenotype": "Breast + PCa + ovarian secondary"},
            {"variant": "c.3113+1G>A", "protein_effect": "Splice donor Exon 12 exon skipping", "location": "Intron 12", "phenotype": "FANCN; moderate PCa risk"},
            {"variant": "p.Tyr551Stop (c.1653T>A)", "protein_effect": "Central domain truncation", "location": "Exon 6", "phenotype": "Breast-dominant; PCa emerging secondary"},
        ],
        "surveillance": [
            "Annual breast MRI from age 30yr (breast risk 53% dominates clinical priorities)",
            "PSA + DRE annually from age 45yr",
            "HRD testing of tumour at mCRPC diagnosis",
            "Olaparib consideration for mCRPC if HRD-positive (emerging data)",
            "Annual pancreatic MRI from age 50yr (PALB2 PDAC secondary risk)",
        ],
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MutL-Homolog-1-85kDa-MMR-MutLalpha-"
            "Lynch-Syndrome-Type1-"
            "Prostate-3-8x-RR-MSI-H-"
            "Pembrolizumab-FDA2017-Tumor-Agnostic-"
            "BRAF-V600E-Absent-Confirms-Germline-"
            "MLPA-MANDATORY-EPCAM-"
            "OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 prostate cancer molecular context: "
            "STRUCTURE: 756 aa / 85 kDa; MutL homolog; "
            "N-terminal ATPase (aa 1-340); C-terminal dimerisation (aa 500-756): PMS2 binding; "
            "MLH1-PMS2 = MutLalpha: corrects mismatches in newly synthesised DNA; "
            "MMR-deficient PCa: MSI-H or high TMB -> immunotherapy-responsive; "
            "MSI-H PCa approximately 3-5% of all PCa; Lynch-germline enriched; "
            "CANCER RISKS: CRC 80-85% Lynch1 DOMINANT; Endometrial 40-60% (female); "
            "Prostate 3-8x RR (3.7% lifetime); Urological cancers elevated; "
            "KEY MANAGEMENT: Colonoscopy q1-2yr from 25yr; pembrolizumab FDA2017 for MSI-H PCa"
        ),
        "key_mutations": [
            {"variant": "p.Val384Asp (c.1151T>A)", "protein_effect": "C-terminal dimerisation disrupted", "location": "Exon 11", "phenotype": "Lynch1; CRC dominant; PCa 3-8x"},
            {"variant": "Large exon deletion (MLPA required)", "protein_effect": "Complete exon LOF; WES misses", "location": "Multi-exon", "phenotype": "Lynch1; MLPA essential — WES false-negative risk"},
            {"variant": "p.Arg659Stop (c.1975C>T)", "protein_effect": "C-terminal truncation MutLalpha lost", "location": "Exon 17", "phenotype": "Lynch1; CRC + PCa + urological elevated"},
        ],
        "surveillance": [
            "Colonoscopy every 1-2yr from age 25yr (CRC 80-85% DOMINATES Lynch1)",
            "PSA + DRE from age 40yr; MSI testing of PCa at diagnosis",
            "Pembrolizumab FDA2017 for MSI-H PCa (tumor-agnostic approval)",
            "Aspirin 600mg CAPP2 regimen (Lynch CRC risk 50% reduction)",
            "Annual endometrial biopsy from age 35yr (female carriers)",
        ],
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 936aa -- "
            "MutS-Homolog-2-105kDa-MutSalpha-MutSbeta-"
            "Lynch-Syndrome-Type2-Muir-Torre-"
            "Prostate-HIGHEST-Lynch-11-14x-RR-"
            "MSI-H-Pembrolizumab-FDA2017-"
            "EPCAM-3prime-Deletion-MSH2-Silencing-MLPA-MANDATORY-"
            "Sebaceous-Carcinoma-Muir-Torre-PATHOGNOMONIC-"
            "OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "936 aa / 105 kDa / 2p21 MSH2 prostate cancer molecular context: "
            "STRUCTURE: 936 aa / 105 kDa; MutS homolog; "
            "MutSalpha (MSH2-MSH6): primary mismatch recognition complex; "
            "MutSbeta (MSH2-MSH3): insertion-deletion loop recognition complex; "
            "MSH2 = obligate partner of both MSH6 and MSH3; "
            "EPCAM 3-prime deletion: silences adjacent MSH2 via promoter methylation; MLPA detects; "
            "Muir-Torre Syndrome: MSH2-predominant; sebaceous carcinoma PATHOGNOMONIC; "
            "MSH2 gives HIGHEST Lynch prostate cancer risk (11-14x RR); "
            "CANCER RISKS: CRC 80% Lynch2; Endometrial 40%; "
            "Prostate 11-14x RR (HIGHEST Lynch gene for PCa); Urological elevated; "
            "Sebaceous carcinoma (Muir-Torre PATHOGNOMONIC); "
            "KEY MANAGEMENT: PSA from 40yr; MLPA MANDATORY; MSI testing; pembrolizumab MSI-H PCa"
        ),
        "key_mutations": [
            {"variant": "p.Arg524Pro (c.1571G>C)", "protein_effect": "DNA-mismatch binding interface disrupted", "location": "Exon 10", "phenotype": "Lynch2; prostate 11-14x; Muir-Torre"},
            {"variant": "EPCAM 3-prime deletion (epigenetic silencing)", "protein_effect": "MSH2 epigenetic silencing by promoter methylation", "location": "EPCAM-MSH2 locus", "phenotype": "Lynch2; WES MISSES — MLPA MANDATORY; PCa 11-14x"},
            {"variant": "p.Ala636Pro (c.1906G>C)", "protein_effect": "MutSalpha dimerisation disrupted", "location": "Exon 12", "phenotype": "Lynch2 high penetrance; PCa HIGHEST"},
        ],
        "surveillance": [
            "PSA + DRE from age 40yr (MSH2 HIGHEST Lynch PCa — 11-14x RR)",
            "MRI prostate at PSA >2.5 ng/mL or any rapid rise",
            "MLPA mandatory in all Lynch2 families (EPCAM deletions missed by WES)",
            "MSI testing on ALL Lynch2 prostate cancers at diagnosis",
            "Pembrolizumab FDA2017 for MSI-H PCa; KEYNOTE-365 olaparib+pembrolizumab active",
        ],
    },
    {
        "gene": "HOXB13",
        "protein": (
            "HOXB13 -- 17q21.2 Autosomal-Dominant-LOF -- 283aa -- "
            "Homeodomain-TF-31kDa-AR-Coregulator-"
            "G84E-Founder-North-European-3.5pct-Carrier-Freq-"
            "Prostate-3-6x-RR-HEREDITARY-PROSTATE-SPECIFIC-"
            "Early-Onset-Lt60yr-No-Breast-Ovarian-CRC-Risk-"
            "OMIM-604607"
        ),
        "locus": "17q21.2",
        "protein_size": (
            "283 aa / 31 kDa / 17q21.2 HOXB13 prostate cancer molecular context: "
            "STRUCTURE: 283 aa / 31 kDa; homeobox transcription factor; "
            "Homeodomain (aa 209-268): DNA binding ANTP class; "
            "HOXB13 = androgen receptor coregulator expressed in prostate epithelium; "
            "MECHANISM: G84E disrupts homeodomain structure -> altered AR co-regulation -> PCa initiation; "
            "HOXB13 = PROSTATE-SPECIFIC HEREDITARY GENE — minimal cancer risk outside prostate; "
            "G84E founder: 3.5% carrier frequency in Northern European (Swedish/Finnish/Norwegian); "
            "Carries 3-6x PCa RR; early-onset (<60yr); strong familial clustering; "
            "CANCER RISKS: Prostate 3-6x RR — PROSTATE-SPECIFIC (most specific hereditary PCa gene); "
            "Breast: NOT elevated (HOXB13 = prostate-specific tissue expression); "
            "KEY MANAGEMENT: PSA from age 40yr; cascade family testing; standard AS if screen-detected"
        ),
        "key_mutations": [
            {"variant": "p.Gly84Glu (c.251G>A)", "protein_effect": "Homeodomain structural disruption", "location": "Exon 1", "phenotype": "Founder allele 3.5% N-European; PCa 3-6x; early-onset <60yr"},
            {"variant": "p.Arg243His (c.728G>A)", "protein_effect": "Homeodomain DNA-binding disrupted", "location": "Exon 1", "phenotype": "Rare; hereditary PCa; less characterised than G84E"},
            {"variant": "p.Ala129Val (c.386C>T)", "protein_effect": "N-terminal regulatory disruption", "location": "Exon 1", "phenotype": "Rare; prostate-specific risk"},
        ],
        "surveillance": [
            "PSA + DRE from age 40yr MANDATORY (HOXB13 G84E carriers)",
            "Annual PSA surveillance with low threshold for MRI prostate",
            "Cascade family testing strongly recommended (G84E 3.5% N-European freq)",
            "Active surveillance standard criteria for screen-detected low-risk PCa",
            "NOTE: No breast/ovarian/colorectal risk elevation — prostate-specific gene",
        ],
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 Autosomal-Dominant-LOF -- 543aa -- "
            "Checkpoint-Kinase-2-60kDa-ATM-Effector-DDR-"
            "Moderate-Risk-Prostate-2-3x-RR-"
            "I157T-Central-Eastern-European-Founder-"
            "1100delC-Western-European-Founder-"
            "Breast-2-3x-RR-CRC-2x-RR-"
            "OMIM-604373"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / 60 kDa / 22q12.1 CHEK2 prostate cancer molecular context: "
            "STRUCTURE: 543 aa / 60 kDa; checkpoint kinase effector; "
            "SQ/TQ cluster domain (aa 1-69): ATM phosphorylation targets; "
            "FHA domain (aa 113-175): phosphopeptide recognition; "
            "Kinase domain (aa 210-486): downstream DDR effectors CDC25; "
            "MECHANISM: ATM activates CHEK2 (Thr68) -> CHEK2 phosphorylates BRCA1/CDC25C/p53; "
            "CHEK2 LOF = impaired G2/M checkpoint -> genomic instability -> moderate cancer risk; "
            "I157T (missense FHA domain): central/eastern European; moderate risk; ~1% carrier freq; "
            "1100delC (frameshift): western European; moderate-higher risk; ~0.5-1% carrier freq; "
            "CANCER RISKS: Prostate 2-3x RR (I157T enriched in PCa); "
            "Breast 2-3x RR (female carriers); CRC 2x; Kidney 2x; "
            "KEY MANAGEMENT: PSA from 40yr; breast MRI female; no PARPi indication currently"
        ),
        "key_mutations": [
            {"variant": "p.Ile157Thr (c.470T>C)", "protein_effect": "FHA domain phosphopeptide binding impaired", "location": "Exon 3", "phenotype": "Central/eastern European founder; PCa 2-3x; breast 2x"},
            {"variant": "c.1100delC (p.Thr367MetfsTer15)", "protein_effect": "Frameshift kinase domain truncation", "location": "Exon 10", "phenotype": "Western European founder; PCa 2-3x; breast 2-3x"},
            {"variant": "p.Ser428Phe (c.1283C>T)", "protein_effect": "Kinase domain catalytic disruption", "location": "Exon 11", "phenotype": "Less common; moderate PCa + breast risk"},
        ],
        "surveillance": [
            "PSA + DRE annually from age 40yr",
            "Annual breast MRI from age 40yr (female carriers: 2-3x RR)",
            "Annual colonoscopy from age 45yr (CRC 2x RR)",
            "No PARPi indication currently — HRD assay insufficient for olaparib without confirmation",
            "Cascade family testing recommended given founder allele frequencies",
        ],
    },
]


def _patients_for_gene(gene_dict: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    pts = []
    for i in range(40):
        age = int(rng.gauss(
            {"BRCA2": 56, "BRCA1": 60, "ATM": 61, "PALB2": 59,
             "MLH1": 64, "MSH2": 62, "HOXB13": 55, "CHEK2": 63}[gene],
            {"BRCA2": 10, "BRCA1": 10, "ATM": 10, "PALB2": 10,
             "MLH1": 10, "MSH2": 10, "HOXB13": 9,  "CHEK2": 10}[gene]
        ))
        age = max(35, min(82, age))

        pc_rate = {
            "BRCA2": 0.65, "BRCA1": 0.45, "ATM": 0.55, "PALB2": 0.40,
            "MLH1": 0.45,  "MSH2": 0.60, "HOXB13": 0.70, "CHEK2": 0.50,
        }[gene]
        has_pca = rng.random() < pc_rate

        high_grade_rate = {
            "BRCA2": 0.62, "BRCA1": 0.45, "ATM": 0.65, "PALB2": 0.42,
            "MLH1": 0.30,  "MSH2": 0.28, "HOXB13": 0.50, "CHEK2": 0.35,
        }[gene]
        high_grade = has_pca and rng.random() < high_grade_rate

        metastatic_rate = {
            "BRCA2": 0.42, "BRCA1": 0.30, "ATM": 0.40, "PALB2": 0.28,
            "MLH1": 0.22,  "MSH2": 0.20, "HOXB13": 0.30, "CHEK2": 0.25,
        }[gene]
        is_metastatic = has_pca and rng.random() < metastatic_rate

        msi_h = gene in ("MLH1", "MSH2") and has_pca and rng.random() < 0.82
        hrd_flag = gene in ("BRCA2", "BRCA1", "ATM", "PALB2") and has_pca and rng.random() < 0.75

        breast_rate = {
            "BRCA2": 0.72, "BRCA1": 0.75, "ATM": 0.35, "PALB2": 0.55,
            "MLH1": 0.0,   "MSH2": 0.0,  "HOXB13": 0.0, "CHEK2": 0.28,
        }[gene]
        has_breast = rng.random() < breast_rate * 0.5

        crc_rate = {
            "BRCA2": 0.05, "BRCA1": 0.03, "ATM": 0.05, "PALB2": 0.03,
            "MLH1": 0.82,  "MSH2": 0.80, "HOXB13": 0.0, "CHEK2": 0.18,
        }[gene]
        has_crc = rng.random() < crc_rate

        radiation_sensitivity = gene in ("ATM",)
        early_onset = gene == "HOXB13" and age < 60 and has_pca
        founder_allele = gene in ("CHEK2", "HOXB13") and rng.random() < 0.75

        pts.append({
            "gene": gene,
            "age": age,
            "has_pca": has_pca,
            "high_grade": high_grade,
            "is_metastatic": is_metastatic,
            "msi_h": msi_h,
            "hrd_flag": hrd_flag,
            "has_breast": has_breast,
            "has_crc": has_crc,
            "radiation_sensitivity": radiation_sensitivity,
            "early_onset": early_onset,
            "founder_allele": founder_allele,
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
                "gene": g, "n": 0, "pca": 0, "high_grade": 0,
                "metastatic": 0, "msi_h": 0, "hrd": 0,
                "early_onset": 0, "mean_age": 0
            }
        gene_counts[g]["n"] += 1
        if p["has_pca"]:       gene_counts[g]["pca"] += 1
        if p["high_grade"]:    gene_counts[g]["high_grade"] += 1
        if p["is_metastatic"]: gene_counts[g]["metastatic"] += 1
        if p["msi_h"]:         gene_counts[g]["msi_h"] += 1
        if p["hrd_flag"]:      gene_counts[g]["hrd"] += 1
        if p["early_onset"]:   gene_counts[g]["early_onset"] += 1
        gene_counts[g]["mean_age"] += p["age"]

    for g in gene_counts:
        n = gene_counts[g]["n"]
        gene_counts[g]["mean_age"] = round(gene_counts[g]["mean_age"] / n, 1)
        gene_counts[g]["pca_pct"] = round(100 * gene_counts[g]["pca"] / n, 1)
        gene_counts[g]["hrd_pct"] = round(100 * gene_counts[g]["hrd"] / n, 1)

    pca_total    = sum(1 for p in pts if p["has_pca"])
    hg_total     = sum(1 for p in pts if p["high_grade"])
    meta_total   = sum(1 for p in pts if p["is_metastatic"])
    msi_total    = sum(1 for p in pts if p["msi_h"])
    hrd_total    = sum(1 for p in pts if p["hrd_flag"])
    early_total  = sum(1 for p in pts if p["early_onset"])
    rt_sens_total = sum(1 for p in pts if p["radiation_sensitivity"])

    return {
        "atlas":             "Hereditary-Prostate-Cancer-Predisposition-Atlas",
        "atlas_id":          "hereditary-prostate-cancer-predisposition-atlas",
        "subtitle":          "Complete 8-Gene BRCA2-BRCA1-ATM-PALB2-MLH1-MSH2-HOXB13-CHEK2 Reference",
        "total_patients":    len(pts),
        "gene_cohorts":      len(ATLAS_GENES),
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "pca_cases":         pca_total,
        "pca_rate_pct":      round(100 * pca_total / len(pts), 1),
        "high_grade_cases":  hg_total,
        "metastatic_cases":  meta_total,
        "msi_h_cases":       msi_total,
        "hrd_cases":         hrd_total,
        "hrd_rate_pct":      round(100 * hrd_total / len(pts), 1),
        "early_onset_cases": early_total,
        "rt_sensitivity_cases": rt_sens_total,
        "gene_summary":      list(gene_counts.values()),
        "key_clinical_rules": [
            "BRCA2 (HBOC-2): ~15% of mCRPC carry BRCA2; olaparib PROfound HR 0.34; rucaparib FDA2020; PSA from 40yr IMPACT trial; biopsy threshold PSA ≥3",
            "ATM: HIGH GRADE predilection Gleason 8-10; olaparib PROfound Cohort B HR 0.45; RT reduce 20-30% even heterozygotes; surgery-first preferred",
            "MSH2 (Lynch2): HIGHEST Lynch PCa 11-14x RR; EPCAM 3-prime deletion silences MSH2 — MLPA MANDATORY (WES misses); MSI-H pembrolizumab FDA2017",
            "HOXB13 G84E: PROSTATE-SPECIFIC gene; 3.5% carrier freq N-European; PCa 3-6x; early onset <60yr; NO breast/ovarian/CRC elevation",
            "CHEK2 I157T: central/eastern European founder; CHEK2 1100delC: western European; PCa 2-3x moderate; no PARPi indication currently",
            "MSI-H PCa (MLH1/MSH2): pembrolizumab FDA2017 tumor-agnostic; ~3-5% of all PCa MSI-H; screen ALL PCa with IHC MMR + MSI PCR",
        ],
        "prostate_risk_hierarchy": {
            "MSH2_Lynch2":    "11-14x RR (HIGHEST Lynch PCa; EPCAM MLPA MANDATORY; MSI-H pembrolizumab)",
            "HOXB13_G84E":    "3-6x RR (prostate-specific; G84E 3.5% N-European; early-onset <60yr)",
            "BRCA2_HBOC2":    "3-8x RR (15% mCRPC; olaparib PROfound HR 0.34; HIGH grade)",
            "ATM":            "4-8x RR HIGH GRADE (Gleason 8-10; PROfound Cohort B; RT-sensitive)",
            "MLH1_Lynch1":    "3-8x RR (MSI-H pembrolizumab; colonoscopy from 25yr dominates)",
            "BRCA1_HBOC1":    "2-3x RR (PROfound Cohort A HR 0.35; olaparib active; breast dominates)",
            "PALB2_FANCN":    "2-3x RR (emerging HRD; olaparib rationale; breast 53% dominates)",
            "CHEK2":          "2-3x RR (moderate; I157T/1100delC founders; no PARPi yet)",
        },
        "profound_trial_summary": {
            "design":      "Phase III: olaparib 300mg BD vs enzalutamide/abiraterone in mCRPC + HRR mutations",
            "cohort_A":    "BRCA1/BRCA2 — HR 0.34 rPFS; FDA2020 approval olaparib + rucaparib",
            "cohort_B":    "ATM + 12 other HRR genes — HR 0.45 rPFS; significant",
            "FDA_approval": "Olaparib FDA2020 mCRPC BRCA1/2; rucaparib FDA2020 mCRPC BRCA1/2",
            "germline_testing": "ALL mCRPC patients: germline HRR panel + somatic HRR + MSI testing",
        },
    }


def generate_breakdown() -> dict:
    pts = _all_patients()
    per_gene = []
    for gene_dict in ATLAS_GENES:
        gene = gene_dict["gene"]
        gpts = [p for p in pts if p["gene"] == gene]
        n = len(gpts)
        pca_n   = sum(1 for p in gpts if p["has_pca"])
        hg_n    = sum(1 for p in gpts if p["high_grade"])
        meta_n  = sum(1 for p in gpts if p["is_metastatic"])
        hrd_n   = sum(1 for p in gpts if p["hrd_flag"])
        msi_n   = sum(1 for p in gpts if p["msi_h"])
        early_n = sum(1 for p in gpts if p["early_onset"])
        mean_age = round(sum(p["age"] for p in gpts) / n, 1)

        key_avoid = {
            "BRCA2":  "Delaying germline testing — all mCRPC need HRR panel; olaparib opportunity lost",
            "BRCA1":  "Assuming BRCA1 lower olaparib benefit — PROfound HR 0.35 is still significant",
            "ATM":    "High-dose RT without reduction — 20-30% reduction mandatory even heterozygotes",
            "PALB2":  "Skipping HRD testing — olaparib rationale depends on HRD status confirmation",
            "MLH1":   "Missing MSI testing on PCa — pembrolizumab FDA2017 opportunity in MSI-H",
            "MSH2":   "Skipping MLPA — EPCAM 3-prime deletions silencing MSH2 MISSED by WES alone",
            "HOXB13": "Assuming breast/ovarian risk — HOXB13 is PROSTATE-SPECIFIC; no other organ risk",
            "CHEK2":  "Offering PARPi empirically — CHEK2 HRD insufficient for olaparib without HRD assay confirmation",
        }[gene]

        key_rule = {
            "BRCA2":  "PSA from 40yr IMPACT trial; biopsy PSA ≥3; olaparib/rucaparib FDA-approved mCRPC",
            "BRCA1":  "PSA from 45yr; olaparib PROfound Cohort A; breast MRI from 30yr (dominates)",
            "ATM":    "Low biopsy threshold (high-grade Gleason 8-10); avoid RT (RT-sensitive); olaparib PROfound Cohort B",
            "PALB2":  "Breast MRI 30yr primary; PSA 45yr; HRD assay at mCRPC diagnosis",
            "MLH1":   "Colonoscopy q1-2yr from 25yr; MSI testing all PCa; pembrolizumab MSI-H",
            "MSH2":   "MLPA MANDATORY for EPCAM deletion; PSA from 40yr; pembrolizumab MSI-H PCa",
            "HOXB13": "PSA + DRE from 40yr MANDATORY; cascade family testing; prostate-specific — no other organ surveillance needed",
            "CHEK2":  "PSA from 40yr; breast MRI 40yr female; colonoscopy 45yr; founder freq high in N-European ancestry",
        }[gene]

        per_gene.append({
            "gene": gene,
            "syndrome": {
                "BRCA2":  "HBOC Type-2 (FANCD1 / RAD51 Loader)",
                "BRCA1":  "HBOC Type-1 (FANCS / RING E3 Ligase)",
                "ATM":    "Ataxia-Telangiectasia monoallelic (DDR kinase; RT-sensitive)",
                "PALB2":  "FANCN (BRCA1-BRCA2 Bridge; WD40 scaffold)",
                "MLH1":   "Lynch Syndrome Type-1 (MutLalpha MMR)",
                "MSH2":   "Lynch Syndrome Type-2 / Muir-Torre (MutSalpha)",
                "HOXB13": "Hereditary Prostate Cancer (G84E N-European Founder; prostate-specific)",
                "CHEK2":  "CHEK2-associated cancer predisposition (moderate DDR risk)",
            }[gene],
            "n": n,
            "pca_n": pca_n,
            "pca_pct": round(100 * pca_n / n, 1),
            "high_grade_n": hg_n,
            "high_grade_pct": round(100 * hg_n / n, 1),
            "metastatic_n": meta_n,
            "metastatic_pct": round(100 * meta_n / n, 1),
            "hrd_n": hrd_n,
            "hrd_pct": round(100 * hrd_n / n, 1),
            "msi_h_n": msi_n,
            "early_onset_n": early_n,
            "mean_age": mean_age,
            "key_avoid": key_avoid,
            "key_rule": key_rule,
        })

    profound_cohort_a = {
        "genes":           "BRCA1 and BRCA2",
        "HR_rPFS":         0.34,
        "HR_OS":           0.55,
        "ORR_pct":         33,
        "clinical_takeaway": "olaparib 300mg BD approved FDA2020; rucaparib also FDA2020 for BRCA1/2 mCRPC",
        "note":            "Cohort A: most actionable — high HR benefit; BRCA2 primary target",
    }
    profound_cohort_b = {
        "genes":           "ATM + 12 other HRR genes (CDK12, CHEK2, FANCA, BRIP1, etc.)",
        "HR_rPFS":         0.45,
        "ORR_pct":         12,
        "clinical_takeaway": "Significant but weaker than Cohort A; ATM highest benefit in Cohort B",
        "note":            "Cohort B: ATM most clinically actionable; CHEK2 benefit modest",
    }
    msi_vs_hrd = {
        "HRD_genes":         "BRCA2 / BRCA1 / ATM / PALB2 — PARPi (olaparib/rucaparib) strategy",
        "MSI_H_genes":       "MLH1 / MSH2 — pembrolizumab tumor-agnostic FDA2017",
        "HRD_testing":       "Tumour genomic scar score (LOH/LST/TAI) + germline BRCA panel",
        "MSI_H_testing":     "IHC MMR proteins + PCR MSI panel (all PCa at diagnosis)",
        "mutual_exclusivity": "HRD and MSI-H rarely co-occur in the same PCa",
        "clinical_guidance": "Germline HRR panel + somatic HRR + MSI testing: ALL mCRPC patients",
    }
    hoxb13_specific = {
        "G84E_founder_freq":   "3.5% in Northern European population (Swedish/Finnish/Norwegian enriched)",
        "PCa_RR":              "3-6x lifetime prostate cancer risk",
        "early_onset":         "Predominantly <60yr onset in G84E carriers",
        "prostate_specific":   "NO elevated breast/ovarian/colorectal risk — prostate-only gene",
        "clinical_management": "PSA + DRE from 40yr; low MRI/biopsy threshold; cascade family testing",
        "active_surveillance": "Standard AS criteria still apply for screen-detected low-risk PCa",
    }
    return {
        "per_gene":                 per_gene,
        "profound_trial_cohort_a":  profound_cohort_a,
        "profound_trial_cohort_b":  profound_cohort_b,
        "msi_h_vs_hrd_treatment":   msi_vs_hrd,
        "hoxb13_g84e_key_rules":    hoxb13_specific,
    }


def generate_definitions() -> dict:
    return {
        "genes": [
            {
                "gene":          gd["gene"],
                "full_name": {
                    "BRCA2":  "Breast Cancer Susceptibility 2 (FANCD1 / RAD51 Loader)",
                    "BRCA1":  "Breast Cancer Susceptibility 1 (FANCS / RING E3 Ligase)",
                    "ATM":    "Ataxia-Telangiectasia Mutated (PIK3-like DSB sensor kinase)",
                    "PALB2":  "Partner and Localiser of BRCA2 (FANCN / WD40 scaffold)",
                    "MLH1":   "MutL Homolog 1 (MutLalpha MMR endonuclease scaffold)",
                    "MSH2":   "MutS Homolog 2 (MutSalpha / MutSbeta — MMR mismatch sensor)",
                    "HOXB13": "Homeobox B13 (androgen receptor coregulator; prostate-specific)",
                    "CHEK2":  "Checkpoint Kinase 2 (ATM effector; G2/M checkpoint enforcer)",
                }[gd["gene"]],
                "locus":        gd["locus"],
                "protein":      gd["protein"],
                "protein_size": gd["protein_size"],
                "key_mutations": gd["key_mutations"],
                "surveillance": gd["surveillance"],
            }
            for gd in ATLAS_GENES
        ],
        "key_concepts": {
            "PROfound_trial": (
                "Phase III RCT olaparib 300mg BD vs enzalutamide/abiraterone in mCRPC + HRR mutations. "
                "Cohort A (BRCA1/BRCA2): HR 0.34 rPFS; Cohort B (ATM+12 genes): HR 0.45. "
                "Led to FDA2020 approval olaparib + rucaparib for BRCA1/2 mCRPC."
            ),
            "HRD_in_prostate": (
                "Homologous Recombination Deficiency in prostate cancer: "
                "approximately 27% of mCRPC carry HRR somatic mutation; approximately 12% germline. "
                "BRCA2 most common germline HRR gene in PCa. "
                "HRD defines PARPi eligibility — all mCRPC must be tested for HRR."
            ),
            "MSI_H_prostate": (
                "MSI-High in prostate cancer: approximately 3-5% of all PCa; enriched in Lynch syndrome (MLH1/MSH2). "
                "MSI-H PCa responds to pembrolizumab (FDA2017 tumor-agnostic). "
                "Screen ALL PCa with IHC MMR + MSI PCR at diagnosis."
            ),
            "HOXB13_G84E": (
                "G84E (p.Gly84Glu) founder allele in Northern European ancestry. "
                "3.5% carrier frequency in Swedish/Finnish/Norwegian populations. "
                "PCa 3-6x RR; predominantly less than 60yr early onset. "
                "PROSTATE-SPECIFIC: no elevated breast/ovarian/CRC risk."
            ),
            "CHEK2_founders": (
                "I157T (missense): central/eastern European (Poland, Baltic states); approximately 1% carrier. "
                "1100delC (frameshift): western European; approximately 0.5-1% carrier. "
                "PCa 2-3x RR (moderate); breast 2-3x (female). "
                "No PARPi indication currently."
            ),
            "EPCAM_MSH2_silencing": (
                "EPCAM 3-prime deletion: silences adjacent MSH2 via promoter methylation. "
                "Presents as Lynch2 phenotype (MSH2/MSH6 IHC loss). "
                "MISSED by WES/gene panels — MLPA MANDATORY for all Lynch2 families."
            ),
            "cascade_testing": (
                "First-degree relatives of all 8 genes: 50% carrier probability. "
                "HOXB13 G84E: all male first-degree relatives should be tested. "
                "CHEK2/HOXB13 founder alleles: targeted testing cost-effective in N-European ancestry."
            ),
        },
        "atlas_metadata": {
            "total_genes":             len(ATLAS_GENES),
            "total_patients":          len(ATLAS_GENES) * 40,
            "seeds":                   f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
            "hrd_genes":               ["BRCA2", "BRCA1", "ATM", "PALB2"],
            "msi_genes":               ["MLH1", "MSH2"],
            "prostate_specific_genes": ["HOXB13"],
            "moderate_risk_genes":     ["CHEK2"],
            "primary_targeted_therapy": "Olaparib/Rucaparib (FDA2020 BRCA1/2 mCRPC) + Pembrolizumab (FDA2017 MSI-H PCa)",
        },
    }
