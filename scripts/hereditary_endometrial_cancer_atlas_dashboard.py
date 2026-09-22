#!/usr/bin/env python3
"""Hereditary-Endometrial-Cancer-Atlas — Complete 8-Gene Hereditary Endometrial Cancer Atlas
MLH1   (MutL Homolog 1; 793aa; 3p22.2; AD LOF;
         Lynch Syndrome Type 1 — MMR gene;
         Endometrial cancer 40-50% lifetime; highest overall Lynch cancer burden;
         MSI-H → Pembrolizumab; Aspirin 600mg CAPP2 50% risk reduction;
         seed SEED_BASE+0) ·
MSH6   (MutS Homolog 6; 1360aa; 2p16.3; AD LOF;
         Lynch Syndrome Type 3 — MutSα component;
         ENDOMETRIAL 40-71% — HIGHEST endometrial Lynch risk; CRC 25-40% only;
         MSI-L 30% false-negative confound on MSI-PCR;
         seed SEED_BASE+1) ·
PTEN   (Phosphatase and Tensin Homolog; 403aa; 10q23.31; AD LOF;
         Cowden Syndrome / PHTS;
         Endometrial cancer 28-44% lifetime — DOMINANT cancer in female PHTS;
         Macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC;
         Everolimus mTOR FDA; Selumetinib;
         seed SEED_BASE+2) ·
POLE   (DNA Polymerase Epsilon; 2286aa; 12q24.33; AD GOF exonuclease;
         POLE-Associated Polyposis / PPAP;
         Ultra-hypermutated endometrial Ca (TMB >100 mut/Mb MSS);
         EXCEPTIONAL Pembrolizumab response — complete remissions documented;
         MSS but TMB >100 — do NOT rely on MSI-PCR alone;
         seed SEED_BASE+3) ·
TP53   (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome / serous endometrial Ca;
         p53-abnormal endometrial Ca = distinct TCGA molecular subtype;
         Serous-like, TP53 missense GOF somatic > germline for endometrial;
         AVOID RADIATION ABSOLUTELY in germline TP53 LFS;
         seed SEED_BASE+4) ·
BRCA1  (Breast Cancer Gene 1; 1863aa; 17q21.31; AD LOF;
         HBOC — serous-like / non-endometrioid endometrial Ca;
         Endometrial 2-3x RR; serous uterine Ca in BRCA1 carriers;
         Olaparib/PARP inhibitor sensitivity; BSO recommended by age 40;
         seed SEED_BASE+5) ·
STK11  (Serine-Threonine Kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome;
         Sex cord tumour with annular tubules (SCTAT) PATHOGNOMONIC in females;
         Endometrial Ca 9-12x RR; combined oral contraceptive REDUCES risk;
         Perioral pigmentation PATHOGNOMONIC;
         seed SEED_BASE+6) ·
MSH2   (MutS Homolog 2; 934aa; 2p21; AD LOF;
         Lynch Syndrome Type 2 — MutSα/MutSβ shared subunit;
         Endometrial 40-60% lifetime; EPCAM 3' deletion → MSH2 epigenetic silencing;
         MUIR-TORRE sebaceous neoplasm PATHOGNOMONIC;
         Universal MMR IHC on all endometrial Ca biopsies;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3126-3133)
"""
import random

SEED_BASE = 3126

ATLAS_GENES = [
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 793aa -- "
            "MutL-Homolog-1-MutLalpha-ATPase-Coordinator-"
            "Lynch-Syndrome-Type1-HNPCC1-Endometrial-40-50pct-"
            "MSI-H-Pembrolizumab-Aspirin-600mg-CAPP2-50pct-Reduction-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "793 aa / 3p22.2 MLH1 encodes MutL Homolog 1 (MLH1): "
            "STRUCTURE: N-terminal ATPase domain (binds/hydrolyses ATP); "
            "  C-terminal dimerisation domain (obligate heterodimerisation with PMS2); "
            "  MLH1-PMS2 = MutLα (major repair heterodimer); "
            "  MLH1 also dimerises with PMS1 (MutLβ) and MLH3 (MutLγ — meiotic); "
            "  MutLα functions: "
            "    1. Recruited by MutSα/MutSβ after mismatch recognition; "
            "    2. PMS2 endonuclease nicks strand containing mismatch; "
            "    3. Coordinates EXO1 excision, RPA protection, re-synthesis; "
            "  MLH1 LOF → MMR failure → MSI-High phenotype; "
            "LYNCH SYNDROME TYPE 1 — ENDOMETRIAL: "
            "  MLH1 = most common Lynch gene (~35-40% Lynch families); "
            "  ENDOMETRIAL CANCER RISK: 40-50% lifetime (second cancer in MLH1 after CRC); "
            "    Onset: median age 48yr (vs 63yr sporadic endometrial Ca); "
            "    Histology: endometrioid grade 1-2 most common (NOT serous in Lynch); "
            "    MSI-H in tumour: >95% MLH1-Lynch endometrial Ca; "
            "  COLORECTAL: 75-85% lifetime (dominant cancer); "
            "  SOMATIC MLH1 METHYLATION (NOT germline): "
            "    Sporadic MSI-H endometrial Ca: MLH1 promoter METHYLATION (not Lynch); "
            "    BRAF V600E in sporadic CRC MSI-H: NOT Lynch (BRAF unmutated → Lynch work-up); "
            "    Methylation-specific PCR on tumour: distinguishes somatic from germline; "
            "ENDOMETRIAL CA SURVEILLANCE: "
            "  Annual pelvic US + CA-125 from age 35 (controversial — low sensitivity); "
            "  ANNUAL ENDOMETRIAL BIOPSY from age 35: most effective Lynch endometrial surveillance; "
            "  Hysteroscopy: if biopsy inconclusive; "
            "  Risk-reducing surgery: "
            "    TOTAL HYSTERECTOMY + BILATERAL SALPINGO-OOPHORECTOMY (TH+BSO): "
            "    Recommended after child-bearing complete or age 40-45; "
            "    Eliminates endometrial Ca risk AND reduces ovarian Ca risk; "
            "    Discuss individual risk tolerance and fertility plans; "
            "IMMUNOTHERAPY: "
            "  MSI-H endometrial Ca → Pembrolizumab (KEYNOTE-158, KEYNOTE-775 + lenvatinib); "
            "  Lenvatinib + pembrolizumab (Lenvima+Keytruda): FDA2021 — ALL endometrial Ca (not only MSI-H); "
            "    Greater benefit in pMMR (non-MSI-H); MSI-H → pembrolizumab monotherapy preferred; "
            "  Dostarlimab (Jemperli): FDA2021 — dMMR endometrial Ca; "
            "    RUBY trial: dostarlimab + carboplatin + paclitaxel → 1st-line advanced endometrial Ca; "
            "ASPIRIN CHEMOPREVENTION: "
            "  600mg/day aspirin (CAPP2 trial): 50% reduction in Lynch cancer incidence; "
            "  Minimum 2 years recommended; standard of care in Lynch surveillance."
        ),
        "inheritance": (
            "AD LOF 3p22.2 — MLH1. ~1:500-1:3,000 (Lynch syndrome all genes 1:400 combined). "
            "MLH1 = most common Lynch gene (~35-40% families). "
            "Large deletions/duplications: ~20% MLH1 pathogenic — MLPA mandatory. "
            "Somatic MLH1 methylation: sporadic MSI-H — NOT germline Lynch; methylation assay on tumour. "
            "CMMRD (biallelic MLH1): haematological malignancy + brain tumour + café-au-lait in childhood. "
            "Cascade testing: all first-degree relatives. "
            "Pre-implantation genetic testing (PGT) available."
        ),
        "surveillance_key": "annual endometrial biopsy from age 35; TH+BSO after childbearing; aspirin 600mg/day; MSI-H → pembrolizumab; distinguish somatic MLH1 methylation from germline Lynch",
        "pathognomonic": "Lynch syndrome: MSI-H endometrial Ca at age <50yr; BRAF V600E somatic is sporadic (NOT Lynch); Muir-Torre sebaceous neoplasms PATHOGNOMONIC Lynch",
    },
    {
        "gene": "MSH6",
        "protein": (
            "MSH6 -- 2p16.3 Autosomal-Dominant-LOF -- 1360aa -- "
            "MutS-Homolog-6-MutSalpha-Mononucleotide-Mismatch-Binding-"
            "Lynch-Syndrome-Type3-ENDOMETRIAL-40-71pct-HIGHEST-Lynch-Risk-"
            "MSI-L-30pct-False-Negative-Confounds-Testing-OMIM-600678"
        ),
        "locus": "2p16.3",
        "protein_size": (
            "1360 aa / 2p16.3 MSH6 encodes MutS Homolog 6 (MSH6): "
            "STRUCTURE: N-terminal PCNA-binding domain; "
            "  Mismatch-binding domain — MBD (MSH6 provides mismatch contact residues); "
            "  Connector/lever/clamp domains; "
            "  C-terminal ATPase domain; "
            "  MSH6 obligate heterodimerises with MSH2 → MutSα (MutSα = MSH2-MSH6); "
            "    MutSα recognises: single base-base mismatches + short IDLs (1-2 nt); "
            "  MSH6 LOF → MutSα failure → loss of single-base mismatch repair; "
            "  Note: MSH3 (pairs with MSH2 as MutSβ) still functions → "
            "    preserves some IDL repair → MSI-L or MSS in 30% MSH6 tumours; "
            "MSH6 AND ENDOMETRIAL CANCER — DOMINANT PHENOTYPE: "
            "  ENDOMETRIAL CANCER: 40-71% lifetime — HIGHEST endometrial risk of all Lynch genes; "
            "    Onset: median 55yr (later than MLH1/MSH2 endometrial Ca); "
            "    Endometrioid histology most common; "
            "    MSI testing pitfall: 30% MSH6 endometrial Ca = MSI-Low or MSS on PCR; "
            "      → Use IHC (MSH6 protein loss) as PRIMARY test; NOT MSI-PCR alone; "
            "  COLORECTAL: 25-40% (attenuated — later onset ~50-60yr); "
            "  OVARIAN: 10-17% lifetime; "
            "MSI TESTING PITFALL IN MSH6: "
            "  Standard 5-marker MSI panel (mononucleotide repeats): "
            "    BAT25, BAT26 — STABLE in MSH6 tumours (30% of time); "
            "    PentaD, PentaE (dinucleotide) — may be unstable; "
            "  CORRECT APPROACH: "
            "    IHC first: MSH6 loss on IHC → MSH6 Lynch regardless of MSI status; "
            "    Expanded dinucleotide MSI panel: more sensitive for MSH6; "
            "    NGS-based MSI: most sensitive — identifies MSH6 MSI-L/MSS tumours; "
            "IHC INTERPRETATION: "
            "  Isolated MSH6 loss (MSH2 intact): MSH6 germline mutation; "
            "  MSH2 + MSH6 both lost: MSH2 germline (MSH2 LOF destabilises MSH6); "
            "ENDOMETRIAL SURVEILLANCE AND RISK REDUCTION: "
            "  Annual endometrial biopsy from age 35-40; "
            "  TH+BSO: after childbearing complete — eliminates endometrial + reduces ovarian risk; "
            "  COMBINED ORAL CONTRACEPTIVE (COC): reduces endometrial Ca risk in Lynch carriers; "
            "    Progestogen dominant COC or IUS (levonorgestrel) — chemoprotective; "
            "  Aspirin 600mg/day (CAPP2); "
            "IMMUNOTHERAPY: "
            "  MSH6-Lynch endometrial Ca: MSI-H on IHC → pembrolizumab eligible; "
            "  But: 30% MSH6 endometrial Ca MSS by PCR → NGS-MSI or IHC for eligibility; "
        ),
        "inheritance": (
            "AD LOF 2p16.3 — MSH6. ~1:2,000-1:10,000. "
            "MSH6 = ~18% Lynch families; commoner in endometrial-dominant families. "
            "Large deletions: ~5% MSH6 pathogenic — MLPA. "
            "MSI pitfall: 30% MSH6 Lynch endometrial Ca MSI-L/MSS — IHC more sensitive. "
            "Attenuated CRC phenotype: later onset, lower penetrance than MLH1/MSH2. "
            "Biallelic MSH6: CMMRD — less well-characterised than biallelic MLH1/MSH2."
        ),
        "surveillance_key": "IHC MSH6 as PRIMARY endometrial test (not MSI-PCR — 30% false negative); annual endometrial biopsy from age 35; TH+BSO after childbearing; aspirin 600mg/day",
        "pathognomonic": "isolated MSH6 IHC loss PATHOGNOMONIC MSH6 Lynch; MSI-L/MSS on PCR does NOT exclude MSH6 Lynch — IHC is more sensitive; endometrial dominant > colorectal phenotype",
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "Phosphatase-Tensin-Homolog-Lipid-Protein-Phosphatase-PI3K-Akt-mTOR-Brake-"
            "Cowden-Syndrome-PHTS-ENDOMETRIAL-28-44pct-DOMINANT-Female-Cancer-"
            "Macrocephaly-PATHOGNOMONIC-Lhermitte-Duclos-PATHOGNOMONIC-Everolimus-OMIM-601728"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 10q23.31 PTEN encodes Phosphatase and Tensin Homolog (PTEN): "
            "STRUCTURE: N-terminal phosphatase domain — "
            "  dual-specificity lipid (dephosphorylates PIP3→PIP2) + protein phosphatase; "
            "  C2 domain (membrane localisation); "
            "  C-terminal PDZ-binding motif; "
            "  PTEN is the dominant brake on PI3K-Akt-mTOR pathway: "
            "    PI3K phosphorylates PIP2→PIP3 → Akt activation; "
            "    PTEN dephosphorylates PIP3→PIP2 → Akt OFF; "
            "  PTEN LOF → constitutive PI3K-Akt-mTOR → proliferation; "
            "  mTOR → mTORC1 phosphorylates S6K1 + 4EBP1 → protein synthesis → growth; "
            "  PTEN LOF also impairs NHEJ DNA repair (nuclear PTEN function); "
            "COWDEN SYNDROME / PHTS (PTEN Hamartoma Tumour Syndrome): "
            "  Prevalence: 1:200,000-1:250,000 (may be underdiagnosed due to clinical variability); "
            "  PATHOGNOMONIC FEATURES: "
            "    MACROCEPHALY (≥97th centile OFC): >95% PHTS — most consistent sign; "
            "    Lhermitte-Duclos disease (dysplastic gangliocytoma cerebellum): PATHOGNOMONIC; "
            "    Trichilemmomas: benign adnexal skin tumours (face, neck); "
            "    Papillomatous papules (oral mucosa, skin); "
            "    Macular pigmentation glans penis; "
            "ENDOMETRIAL CANCER IN PHTS — DOMINANT FEMALE CANCER: "
            "  28-44% lifetime endometrial cancer — higher than many realise; "
            "  Onset earlier than sporadic (mean age 44-50yr vs 62yr sporadic); "
            "  Histology: endometrioid (grade 1-2 most common, like Lynch); "
            "  MSI: pMMR (microsatellite stable) — PTEN tumours are NOT MSI-H; "
            "  MOLECULAR: "
            "    Somatic PTEN loss common in ENDOMETRIAL Ca (50-80% somatic): "
            "      precursor lesion (endometrial intraepithelial neoplasia); "
            "    Germline PTEN → higher endometrial Ca lifetime risk; "
            "ENDOMETRIAL SURVEILLANCE: "
            "  Annual endometrial biopsy/transvaginal USS from age 35-40; "
            "  TH+BSO: risk-reducing surgery option after childbearing; "
            "  Levonorgestrel IUS: chemoprotective (limited PTEN-specific data); "
            "BREAST CANCER: "
            "  BRCA1/2-equivalent or higher breast risk: 85% lifetime (NCCN PHTS); "
            "  Annual breast MRI + mammography from age 30-35; "
            "  Risk-reducing mastectomy option; "
            "mTOR INHIBITOR THERAPY: "
            "  Everolimus (Afinitor): FDA-approved for PTEN-loss tumours (advanced RCC, breast, PNET); "
            "    Endometrial Ca: trials ongoing (GOG-0248 everolimus + letrozole); "
            "  Selumetinib + everolimus: dual PI3K-MAPK block (preclinical); "
        ),
        "inheritance": (
            "AD LOF 10q23.31 — PTEN. ~1:200,000; possibly underdiagnosed. "
            "Penetrance: 85% breast; 28-44% endometrial; 34% renal; thyroid 35%. "
            "Somatic mosaicism: ~10% PHTS — mosaic PTEN → segmental features + lower cancer risk. "
            "Clinical diagnosis: International Cowden Consortium (ICC) criteria: "
            "  Pathognomonic (Lhermitte-Duclos or trichilemmomas): diagnostic alone; "
            "  Major (macrocephaly, breast Ca, thyroid Ca, endometrial Ca): ≥3 or 2 major + 3 minor. "
            "PTEN VUS: common (PTEN missense VUS) — functional assays (luciferase PIP3 assay) helpful. "
            "Testing via clinical suspicion: macrocephaly + tumour → PTEN."
        ),
        "surveillance_key": "macrocephaly is KEY screening sign; annual endometrial biopsy from age 35; TH+BSO after childbearing; annual breast MRI from age 30; Lhermitte-Duclos = diagnostic",
        "pathognomonic": "macrocephaly (≥97th centile OFC) PATHOGNOMONIC PHTS; Lhermitte-Duclos (cerebellar gangliocytoma) PATHOGNOMONIC PHTS; trichilemmomas on face PATHOGNOMONIC; endometrial Ca dominant female cancer 28-44%",
    },
    {
        "gene": "POLE",
        "protein": (
            "POLE -- 12q24.33 Autosomal-Dominant-GOF-Exonuclease -- 2286aa -- "
            "DNA-Polymerase-Epsilon-Catalytic-Proofreading-Exonuclease-"
            "POLE-Associated-Polyposis-PPAP-ULTRA-HYPERMUTATED-TMB-gt100-MSS-"
            "EXCEPTIONAL-Pembrolizumab-Complete-Remissions-NOT-MSI-OMIM-174762"
        ),
        "locus": "12q24.33",
        "protein_size": (
            "2286 aa / 12q24.33 POLE encodes the catalytic subunit of DNA Polymerase ε (Pol ε): "
            "STRUCTURE: N-terminal exonuclease domain (3'→5' proofreading — PATHOGENIC variants here); "
            "  Pol domain (polymerisation); "
            "  C-terminal domain (regulatory, PCNA binding); "
            "  POLE exonuclease domain key residues: Asp275, Glu277 (D275-E277 motif — catalytic); "
            "  POLE proofreads errors made during leading-strand synthesis; "
            "POLE PATHOGENIC VARIANTS — ULTRA-HYPERMUTATED PHENOTYPE: "
            "  Exonuclease domain mutations (pathogenic site): P286R, V411L (most common); "
            "    Also: L424V, S297F, L424I, A456P, M444K, D368V; "
            "  POLE exonuclease LOF: "
            "    Cannot excise misincorporated nucleotides during leading-strand synthesis; "
            "    Errors accumulate → ultra-hypermutated phenotype; "
            "    TMB: >100 mut/Mb (often 200-800 mut/Mb) — highest of any cancer class; "
            "    Mutation signature: C>A transversions; TCT>TAT context; "
            "    MSI STATUS: MICROSATELLITE STABLE (MSS) — NOT MSI-H; "
            "    CRITICAL: POLE ultra-hypermutated ≠ MSI-High — different assay needed; "
            "POLE AND ENDOMETRIAL CANCER: "
            "  POLE germline (P286R, V411L in exonuclease): "
            "    POLE-Associated Polyposis (PPAP): endometrial + colorectal Ca; "
            "    Onset: endometrial Ca median 50-60yr; CRC younger; "
            "  SOMATIC POLE exonuclease: "
            "    7-10% of all endometrial Ca (very high proportion); "
            "    POLE somatic endometrial Ca: best prognosis of all molecular subtypes (TCGA); "
            "    Despite being grade 3 high-grade: excellent prognosis if POLE ultra-mutated; "
            "    Chemotherapy: may confer resistance paradoxically (high mutation burden → resistance?); "
            "PEMBROLIZUMAB — EXCEPTIONAL RESPONSE IN POLE ENDOMETRIAL Ca: "
            "  TMB >10 mut/Mb: FDA2020 approval for pembrolizumab (tumour-agnostic); "
            "  POLE ultra-hypermutated (TMB >100): exceptional pembrolizumab response; "
            "    COMPLETE REMISSIONS documented in advanced POLE endometrial Ca; "
            "    Studies: 90-100% response rates in POLE-mut advanced EC on PD-1/L1; "
            "    Some centres: neoadjuvant pembrolizumab → potential surgery avoidance; "
            "HOW TO TEST FOR POLE: "
            "  NOT detected by MSI-PCR (MSS) — PCR/IHC will MISS POLE; "
            "  Requires: "
            "    Somatic POLE hotspot sequencing (Foundation Medicine CDx / AmpliSeq Colon-Lung panel); "
            "    OR: NGS-based TMB measurement (>10 mut/Mb triggers further workup); "
            "    OR: WES/WGS signature analysis; "
            "  IHC: POLE antibodies not routinely validated for clinical use; "
            "  Germline POLE: include in hereditary endometrial Ca panel if age <50 or family history; "
        ),
        "inheritance": (
            "AD GOF (exonuclease LOF) 12q24.33 — POLE. Germline POLE exonuclease = rare. "
            "PPAP (POLE-Associated Polyposis): AD; adenomatous colonic polyps + endometrial + CRC. "
            "Somatic POLE exonuclease mutations: much more common (7-10% all endometrial Ca); "
            "Germline POLE: ~0.5-1% all endometrial Ca patients. "
            "KEY: somatic POLE confers excellent prognosis; germline POLE = hereditary syndrome. "
            "POLD1 (POLE paralogue): similar exonuclease-domain GOF mutations → PPAP with similar management."
        ),
        "surveillance_key": "POLE endometrial Ca is MSS — do NOT use MSI-PCR; test TMB by NGS; exceptional pembrolizumab response (complete remissions); POLE somatic = best endometrial Ca prognosis TCGA",
        "pathognomonic": "POLE ultra-hypermutated endometrial Ca (TMB >100 mut/Mb) is MSS — NOT detected by MSI-PCR; EXCEPTIONAL pembrolizumab response including complete remissions; POLE somatic = ultra-hypermutated TCGA subtype best prognosis",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF/GOF -- 393aa -- "
            "Tumour-Protein-p53-Transcription-Factor-"
            "Li-Fraumeni-Syndrome-LFS-Serous-Endometrial-Ca-p53-Abnormal-Subtype-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 17p13.1 TP53 encodes tumour protein p53 ('guardian of the genome'): "
            "STRUCTURE: N-terminal transactivation domains (TAD1, TAD2); "
            "  Proline-rich region; "
            "  Central DNA-binding domain (DBD) — most pathogenic variants cluster here; "
            "    Hotspot missense: R175H, R248W, R248Q, R273H, R249S, G245S; "
            "  Tetramerisation domain; "
            "  C-terminal regulatory domain; "
            "  Active as homotetramer; binds p53 response elements → CDKN1A/p21/BAX/PUMA; "
            "  LOF → unchecked G1/S checkpoint → unrepaired DNA replication; "
            "TP53 AND ENDOMETRIAL CANCER: "
            "  Germline TP53 (Li-Fraumeni Syndrome): "
            "    Endometrial cancer UNCOMMON in classic LFS (not a core LFS cancer); "
            "    BUT: serous endometrial Ca can occur in LFS; "
            "    When endometrial Ca in LFS context: usually serous/high-grade; "
            "  SOMATIC TP53 IN ENDOMETRIAL CA: "
            "    p53-abnormal endometrial Ca = one of four TCGA molecular subtypes: "
            "      1. POLE ultra-mutated (best prognosis); "
            "      2. MMR-deficient/MSI-H (Lynch or sporadic); "
            "      3. Copy-number high (p53-abnormal) — WORST prognosis; "
            "      4. NSMP (no specific molecular profile); "
            "    p53-abnormal: predominantly serous histology; TP53 somatic missense dominant; "
            "    p53-abn endometrial Ca: 25-30% all endometrial Ca; aggressive; "
            "    Treatment: carboplatin + paclitaxel ± trastuzumab (if HER2+); bevacizumab trials; "
            "  IHC P53: "
            "    Normal p53 IHC: diffuse weak → pMMR/NSMP; "
            "    Aberrant p53 IHC: "
            "      Overexpression (strong diffuse block): GOF missense TP53 somatic; "
            "      Complete absence: LOF TP53 (truncating/deletion); "
            "      Cytoplasmic: rare localisation mutation; "
            "    p53 IHC on ALL endometrial Ca biopsies: IMPORTANT for molecular classification; "
            "LFS SURVEILLANCE (IF GERMLINE TP53 + ENDOMETRIAL CA): "
            "  AVOID RADIATION ABSOLUTELY (radiation-field sarcoma documented); "
            "  WBMRI annually (Toronto protocol); "
            "  Breast MRI annually from age 20 (women); "
        ),
        "inheritance": (
            "AD LOF 17p13.1 — TP53. ~1:5,000-1:20,000 (LFS). "
            "30% de novo LFS mutations. "
            "Dominant-negative: GOF missense variants (R248W, R273H) titrate wild-type p53. "
            "Germline TP53 vs somatic TP53 in endometrial Ca: clonal haematopoiesis with TP53 somatic in blood "
            "can mimic germline — germline confirmation requires fibroblast/buccal DNA. "
            "Brazil R337H: 1:300 Southern Brazil — moderate LFS penetrance. "
            "Testing: germline TP53 for serous endometrial Ca at young age or LFS phenotype."
        ),
        "surveillance_key": "p53 IHC on ALL endometrial Ca biopsies (molecular classification); germline TP53 LFS — AVOID RADIATION ABSOLUTELY; WBMRI annually; serous endometrial Ca age <50 → consider TP53 germline testing",
        "pathognomonic": "p53-abnormal IHC on endometrial Ca = serous/high-grade molecular subtype (WORST prognosis TCGA); germline TP53 LFS → adrenocortical carcinoma child <5yr PATHOGNOMONIC; radiation-field sarcoma in LFS",
    },
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "BRCA1-RING-BARD1-E3-Ligase-BRCT-Phospho-Binding-"
            "Hereditary-Breast-Ovarian-Cancer-Serous-Endometrial-2-3x-RR-"
            "BSO-Recommended-Age-40-Olaparib-Platinum-Sensitivity-OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 17q21.31 BRCA1 encodes the BRCA1 tumour suppressor: "
            "STRUCTURE: N-terminal RING domain (E3 ubiquitin ligase; heterodimerises with BARD1); "
            "  Coiled-coil domain (binds PALB2); "
            "  Two BRCT repeats (phospho-Ser/Thr binding — activated by ATM-phospho at DSBs); "
            "  BRCA1 functions: "
            "    1. DSB end resection initiation (with MRN, CtIP, BLM); "
            "    2. RAD51 filament loading (via PALB2-BRCA2 axis); "
            "    3. Cell cycle checkpoint (S-phase, G2/M via CHK1); "
            "    4. Transcription regulation; "
            "  BRCA1 LOF → HRD; PARP inhibitor synthetic lethality; "
            "BRCA1 AND ENDOMETRIAL CANCER: "
            "  Endometrial Ca relative risk: 2-3x vs general population (moderate); "
            "  PHENOTYPE: SEROUS-LIKE endometrial Ca (non-endometrioid, papillary serous); "
            "    BRCA1 endometrial Ca clusters with high-grade, p53-abn molecular subtype; "
            "    NOT the usual Lynch-associated endometrioid Ca; "
            "  Lifetime risk: ~7-9% (vs 3% background — modest absolute elevation); "
            "  Clinical significance: "
            "    Hysterectomy at time of BSO (bilateral salpingo-oophorectomy) reduces risk; "
            "    MOST BRCA1 endometrial Ca arises in post-menopausal period; "
            "BILATERAL SALPINGO-OOPHORECTOMY (BSO) IN BRCA1: "
            "  Recommended age 35-40 (after childbearing) — eliminates ovarian Ca risk; "
            "  CONCOMITANT HYSTERECTOMY: "
            "    Removes uterus at same procedure — eliminates endometrial Ca risk; "
            "    Reduces tamoxifen-associated endometrial Ca risk (if taking tamoxifen); "
            "    Avoids ongoing cervical smear requirement; "
            "  Risk/benefit: add ~30-45 min to BSO; very low peri-operative risk; "
            "  Most expert centres now recommend TH+BSO (not BSO alone) for BRCA1; "
            "BREAST CANCER — DOMINANT RISK: "
            "  Breast: 72% lifetime; ovarian: 44% lifetime; "
            "  Annual breast MRI + mammography from age 25-30; "
            "  Risk-reducing mastectomy: 90% risk reduction; "
            "PARP INHIBITORS FOR BRCA1: "
            "  Olaparib: approved for BRCA1-mutated ovarian Ca (SOLO1/SOLO2); "
            "  Breast, fallopian, peritoneal: approved; "
            "  Endometrial Ca with BRCA1: limited trial data; HRD-sensitive tumours; "
        ),
        "inheritance": (
            "AD LOF 17q21.31 — BRCA1. ~1:400 population. "
            "Penetrance: breast 72%; ovarian 44%; endometrial 7-9% (modest). "
            "Ashkenazi founders: 185delAG (1:100) + 5382insC (1:41). "
            "BRCA1-associated endometrial Ca: serous/p53-abn phenotype — different from Lynch endometrial Ca. "
            "Concomitant hysterectomy at time of BSO: now standard recommendation at most expert centres. "
            "Biallelic BRCA1: essentially lethal embryonically in most contexts — no FANCN biallelic clinical cases."
        ),
        "surveillance_key": "BSO recommended age 35-40 after childbearing; consider concomitant hysterectomy (TH+BSO) to remove endometrial Ca risk + avoid tamoxifen endometrial risk; annual breast MRI from age 25",
        "pathognomonic": "no endometrial-specific pathognomonic sign; BRCA1 endometrial Ca is serous/high-grade (not endometrioid); TH+BSO now preferred over BSO alone to eliminate both ovarian and endometrial Ca risks",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "Serine-Threonine-Kinase-11-LKB1-AMPK-mTOR-Cell-Polarity-"
            "Peutz-Jeghers-Syndrome-PJS-Perioral-Pigmentation-PATHOGNOMONIC-"
            "SCTAT-PATHOGNOMONIC-Endometrial-9-12x-RR-Intussusception-Emergency-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 19p13.3 STK11 encodes Serine-Threonine Kinase 11 (LKB1): "
            "STRUCTURE: N-terminal regulatory domain; "
            "  Kinase domain (activates AMPK and 12 other AMPK-related kinases); "
            "  C-terminal regulatory domain; "
            "  STK11/LKB1 functions: "
            "    1. Activates AMPK → inhibits mTORC1 → metabolic checkpoint; "
            "    2. Cell polarity (via MARK kinases — Par-1 homologues); "
            "    3. Cell migration suppression; "
            "    4. Wnt pathway suppression; "
            "  STK11 LOF → AMPK failure → mTOR constitutive activation → proliferation; "
            "  Somatic STK11 loss in ~3-5% lung adenocarcinoma (non-small cell) — not hereditary; "
            "PEUTZ-JEGHERS SYNDROME (PJS): "
            "  PERIORAL MUCOCUTANEOUS PIGMENTATION — PATHOGNOMONIC: "
            "    Dark spots on lips, buccal mucosa, perioral skin; hands/feet; "
            "    Appear in infancy; fade with age (but oral spots persist); "
            "  Gastrointestinal hamartomatous polyps (small bowel, colon, stomach); "
            "  INTUSSUSCEPTION: leading paediatric emergency in PJS; "
            "    Small bowel polyp acts as lead point → obstruction; "
            "    ANY PJS child with abdominal pain → urgent assessment; "
            "    Treatment: double-balloon enteroscopy polypectomy + surgery if obstructed; "
            "CANCER RISKS IN PJS: "
            "  OVERALL cancer lifetime risk: 93% by age 70 (highest among polyposis syndromes); "
            "  GI: CRC 40%, small bowel 13%, gastric 30%, pancreatic 11%; "
            "  GYNAECOLOGICAL: "
            "    Endometrial Ca: 9x RR; lifetime risk ~9-12%; "
            "    Cervical Ca (minimal deviation adenocarcinoma = adenoma malignum): rare but specific; "
            "    SCTAT (Sex Cord Tumour with Annular Tubules): PATHOGNOMONIC — "
            "      Small ovarian tumour in PJS women; bilateral/multifocal in PJS (vs unilateral sporadic); "
            "      Pathognomonic: <1cm multifocal bilateral ovarian calcifications + SCTAT = PJS; "
            "    Ovarian adenoma malignum (minimal deviation cervical adenocarcinoma): PATHOGNOMONIC; "
            "    Breast: 45-50% lifetime; "
            "  Male: Sertoli cell testicular tumour (large cell calcifying) — PATHOGNOMONIC in PJS; "
            "ENDOMETRIAL SURVEILLANCE PJS: "
            "  Annual pelvic USS + CA-125 from age 25-30; "
            "  Endometrial biopsy: from age 35 (some centres); "
            "  Consider TH+BSO after childbearing (high cumulative risk); "
            "  Combined oral contraceptive: possibly reduces endometrial risk; "
            "mTOR INHIBITOR — RATIONALE: "
            "  Everolimus (mTOR inhibitor): LKB1 LOF → mTOR constitutive → everolimus target; "
            "  Trial data in PJS polyposis: PARP inhibitor trials ongoing; "
        ),
        "inheritance": (
            "AD LOF 19p13.3 — STK11. ~1:50,000-1:200,000. "
            "30-70% de novo mutations (high de novo rate). "
            "Large deletions: ~10% STK11 pathogenic variants — MLPA. "
            "Penetrance: very high cancer risk (93% by age 70). "
            "Somatic STK11: common in lung adenocarcinoma, cervical Ca — NOT hereditary. "
            "Female SCTAT surveillance from puberty; annual breast MRI from age 25."
        ),
        "surveillance_key": "intussusception in PJS child = emergency; SCTAT bilateral multifocal ovarian calcifications PATHOGNOMONIC; endometrial 9x RR; annual breast MRI from age 25; double-balloon enteroscopy polyp surveillance",
        "pathognomonic": "perioral mucocutaneous pigmentation PATHOGNOMONIC PJS; SCTAT (bilateral multifocal ovarian calcifications) PATHOGNOMONIC PJS female; large-cell calcifying Sertoli cell testicular tumour PATHOGNOMONIC PJS male; intussusception in child",
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutS-Homolog-2-MutSalpha-MutSbeta-Obligate-Partner-"
            "Lynch-Syndrome-Type2-Muir-Torre-EPCAM-Silencing-"
            "Endometrial-40-60pct-Universal-MMR-IHC-PATHOGNOMONIC-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 2p21 MSH2 encodes MutS Homolog 2: "
            "STRUCTURE: N-terminal MBD (mismatch-binding domain — binds MSH6 in MutSα); "
            "  Connector/lever/clamp domains; "
            "  C-terminal ATPase domain; "
            "  MSH2 obligate heterodimerises with: "
            "    MSH6 → MutSα (single-base + 1-2 IDL recognition); "
            "    MSH3 → MutSβ (2-16 nt IDL recognition); "
            "  MSH2 is the scaffolding partner in both complexes; "
            "  MSH2 LOF → BOTH MutSα AND MutSβ failure → MSI-High phenotype (full MMR deficiency); "
            "    Unlike MSH6 LOF: MSH2 LOF generates full MSI-High (not partial); "
            "LYNCH SYNDROME TYPE 2 — ENDOMETRIAL COMPONENT: "
            "  ENDOMETRIAL CA: 40-60% lifetime in MSH2 female carriers; "
            "    Second only to MSH6 for endometrial cancer risk in Lynch genes; "
            "    Onset: median 50-55yr (earlier than sporadic); "
            "    Histology: endometrioid predominantly; MSI-H tumours; "
            "  CRC: 50-70% lifetime; "
            "  UROTHELIAL (urinary tract): 25-28% lifetime — HIGHEST Lynch urinary tract risk; "
            "    Annual urine cytology + urological imaging (CT urography) from age 35; "
            "MUIR-TORRE SYNDROME: "
            "  SEBACEOUS NEOPLASM — PATHOGNOMONIC: "
            "    Sebaceous adenoma; sebaceous carcinoma; sebaceoma; keratoacanthoma (sebaceous type); "
            "    ANY sebaceous tumour → MSH2 (and MLH1) Lynch testing; "
            "    Muir-Torre = Lynch with sebaceous skin manifestation — same gene, same management; "
            "  MSH2 most common Muir-Torre gene (~50% Muir-Torre); "
            "EPCAM 3' DELETION → MSH2 EPIGENETIC SILENCING: "
            "  EPCAM (Epithelial Cell Adhesion Molecule, 2p21): adjacent to MSH2; "
            "  3' deletion of EPCAM → read-through transcription into MSH2 → "
            "    MSH2 promoter methylation → silencing of MSH2 protein; "
            "  IHC: MSH2 + MSH6 both lost (same as MSH2 LOF); EPCAM IHC: NORMAL (EPCAM LOF partial); "
            "  CRITICAL: EPCAM 3' deletions detected by MLPA (standard sequencing misses); "
            "    MLPA MSH2 panel must include EPCAM 3' region; "
            "  Tissue specificity: EPCAM silencing predominantly in epithelial tissues (CRC, endometrial); "
            "    Germline DNA: MSH2 intact; MLPA EPCAM essential; "
            "ENDOMETRIAL SURVEILLANCE: "
            "  Annual endometrial biopsy from age 35; "
            "  TH+BSO after childbearing complete; "
            "  Annual urine cytology + CT urography from age 35 (urothelial risk); "
            "  Aspirin 600mg/day (CAPP2); "
        ),
        "inheritance": (
            "AD LOF 2p21 — MSH2. ~1:3,000-1:5,000 Lynch gene. "
            "MSH2 = ~40% Lynch families (second most common after MLH1). "
            "Large deletions: ~20% MSH2 pathogenic — MLPA mandatory (includes EPCAM region). "
            "EPCAM 3' deletion: ~10-15% MSH2-Lynch families — check MLPA EPCAM. "
            "MSH2 loss on IHC: MSH2 + MSH6 co-loss → MSH2 mutation primary. "
            "Cascade testing all first-degree relatives."
        ),
        "surveillance_key": "Muir-Torre sebaceous neoplasm PATHOGNOMONIC Lynch MSH2; MLPA EPCAM region mandatory; annual urine cytology + CT urography (highest Lynch urothelial risk); endometrial biopsy annual from age 35",
        "pathognomonic": "Muir-Torre sebaceous neoplasms PATHOGNOMONIC for MSH2 Lynch; MSH2+MSH6 co-loss on IHC = MSH2 germline; EPCAM 3' deletion → MSH2 silencing — detected by MLPA only (NOT sequencing)",
    },
]


def _gene_stats(seed: int, gene_config: dict) -> dict:
    """Generate per-gene statistics for one gene using a fixed seed."""
    rng = random.Random(seed)
    gene = gene_config["gene"]

    base = {
        "MLH1":  {"endometrial_ca": 45, "crc": 78, "ovarian": 10, "msi_high": 95, "immunotherapy_response": 42, "surgery_prophylactic": 35},
        "MSH6":  {"endometrial_ca": 55, "crc": 30, "ovarian": 14, "msi_high": 70, "immunotherapy_response": 38, "surgery_prophylactic": 32},
        "PTEN":  {"endometrial_ca": 36, "breast": 82, "thyroid": 35, "msi_high": 5, "everolimus_trial": 28, "macrocephaly": 96},
        "POLE":  {"endometrial_ca": 15, "crc": 8, "tmb_high": 98, "msi_high": 5, "immunotherapy_response": 90, "complete_remission": 35},
        "TP53":  {"endometrial_ca": 8, "serous": 65, "p53_abn_ihc": 92, "breast": 30, "radiation_avoid": 100, "wbmri_surveillance": 82},
        "BRCA1": {"endometrial_ca": 8, "breast": 72, "ovarian": 44, "serous": 60, "parp_response": 38, "bso_completed": 55},
        "STK11": {"endometrial_ca": 10, "crc": 40, "breast": 48, "intussusception": 42, "sctat": 18, "perioral_pigmentation": 98},
        "MSH2":  {"endometrial_ca": 48, "crc": 60, "urothelial": 26, "msi_high": 95, "muir_torre": 15, "surgery_prophylactic": 38},
    }.get(gene, {})

    n = 40
    age_mean = {
        "MLH1": 48, "MSH6": 54, "PTEN": 46, "POLE": 54,
        "TP53": 42, "BRCA1": 52, "STK11": 44, "MSH2": 50,
    }.get(gene, 50)

    stats = {
        "gene": gene,
        "n": n,
        "seed": seed,
        "mean_age_diagnosis": round(age_mean + rng.gauss(0, 3), 1),
        "female_pct": round(rng.uniform(95, 100), 1),  # endometrial Ca — female dominant
    }

    for feature, base_rate in base.items():
        rate = max(0, min(100, base_rate + rng.gauss(0, 4)))
        stats[f"{feature}_pct"] = round(rate, 1)

    stats["genetic_testing_positive_pct"] = round(rng.uniform(88, 99), 1)
    stats["surveillance_adherent_pct"] = round(rng.uniform(60, 82), 1)
    stats["family_history_positive_pct"] = round(rng.uniform(35, 68), 1)
    stats["de_novo_pct"] = round(rng.uniform(5, 35), 1)
    return stats


def generate_overview() -> dict:
    return {
        "atlas":          "Hereditary-Endometrial-Cancer-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Endometrial Cancer Atlas "
            "(MLH1-MSH6-PTEN-POLE-TP53-BRCA1-STK11-MSH2)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "MLH1": (
                "AD LOF 3p22.2 (MutLα anchor; 793aa; Lynch Syndrome Type 1; "
                "endometrial 40-50% — 2nd highest overall Lynch burden; "
                "aspirin 600mg CAPP2 50% risk reduction; MSI-H → pembrolizumab; "
                "distinguish from sporadic MLH1 methylation by BRAF V600E + methylation PCR)"
            ),
            "MSH6": (
                "AD LOF 2p16.3 (MutSα mismatch-binding; 1360aa; Lynch Syndrome Type 3; "
                "ENDOMETRIAL 40-71% — HIGHEST Lynch endometrial risk; CRC only 25-40% attenuated; "
                "MSI-L 30% false-negative: USE IHC not MSI-PCR as primary test; "
                "isolated MSH6 IHC loss PATHOGNOMONIC)"
            ),
            "PTEN": (
                "AD LOF 10q23.31 (PI3K-Akt-mTOR brake; 403aa; Cowden/PHTS; "
                "ENDOMETRIAL 28-44% — dominant female cancer; macrocephaly PATHOGNOMONIC; "
                "Lhermitte-Duclos PATHOGNOMONIC; breast 85%; mTOR inhibitor everolimus)"
            ),
            "POLE": (
                "AD GOF exonuclease 12q24.33 (Pol ε proofreading; 2286aa; PPAP; "
                "ULTRA-HYPERMUTATED TMB >100 mut/Mb but MICROSATELLITE STABLE; "
                "NOT detected by MSI-PCR — test TMB by NGS; "
                "EXCEPTIONAL pembrolizumab response — complete remissions documented)"
            ),
            "TP53": (
                "AD LOF/GOF 17p13.1 (p53 guardian; 393aa; Li-Fraumeni Syndrome; "
                "somatic TP53 = p53-abnormal TCGA subtype — serous/high-grade WORST prognosis; "
                "germline TP53 LFS: AVOID ALL RADIATION; WBMRI annually Toronto; "
                "p53 IHC on ALL endometrial Ca biopsies for molecular classification)"
            ),
            "BRCA1": (
                "AD LOF 17q21.31 (RING-BRCT HR repair; 1863aa; HBOC; "
                "endometrial 2-3x RR — serous-like phenotype; "
                "BSO recommended age 35-40 with concomitant hysterectomy (TH+BSO) to remove endometrial risk; "
                "Olaparib/PARP inhibitor sensitivity in HRD tumours)"
            ),
            "STK11": (
                "AD LOF 19p13.3 (LKB1/AMPK kinase; 433aa; Peutz-Jeghers Syndrome; "
                "perioral pigmentation PATHOGNOMONIC; SCTAT bilateral multifocal calcifications PATHOGNOMONIC; "
                "endometrial 9-12x RR; intussusception paediatric EMERGENCY; "
                "overall Ca risk 93% by age 70)"
            ),
            "MSH2": (
                "AD LOF 2p21 (MutSα/MutSβ shared; 934aa; Lynch Syndrome Type 2; "
                "endometrial 40-60%; UROTHELIAL 25-28% HIGHEST Lynch urinary risk; "
                "Muir-Torre sebaceous neoplasms PATHOGNOMONIC; "
                "EPCAM 3' deletion → MSH2 silencing — MLPA EPCAM region mandatory)"
            ),
        },
        "key_clinical_rules": [
            "MSH6: USE IHC (not MSI-PCR) as primary endometrial test — 30% MSH6 endometrial Ca is MSI-L/MSS, causing false negatives",
            "POLE: ultra-hypermutated endometrial Ca (TMB >100) is MICROSATELLITE STABLE — MSI-PCR MISSES POLE; test by NGS-TMB",
            "POLE: EXCEPTIONAL pembrolizumab response including COMPLETE REMISSIONS — test ALL endometrial Ca for POLE somatic variants",
            "PTEN/PHTS: macrocephaly (≥97th centile) is the MOST CONSISTENT sign — all patients with macrocephaly + any cancer → PTEN test",
            "PTEN/PHTS: Lhermitte-Duclos disease (cerebellar dysplastic gangliocytoma) = PATHOGNOMONIC for PHTS — test PTEN immediately",
            "TP53: p53 IHC on ALL endometrial Ca biopsies enables TCGA molecular classification (POLE > MMR-d > NSMP > p53-abn worst prognosis)",
            "TP53 germline (LFS): AVOID ALL RADIATION — radiation-field sarcoma documented; WBMRI annually (Toronto protocol)",
            "MLH1: distinguish somatic MLH1 methylation (sporadic MSI-H) from germline Lynch using BRAF V600E (positive = sporadic) + MLH1 methylation PCR",
            "MSH2: EPCAM 3' deletion causes MSH2 silencing — NOT detected by sequencing; MLPA including EPCAM region is mandatory for all MSH2 Lynch work-up",
            "MSH2: Muir-Torre sebaceous neoplasm on ANY patient → Lynch work-up (MSH2 most common Muir-Torre gene)",
            "BRCA1: TH+BSO (total hysterectomy + BSO) now preferred over BSO alone — removes endometrial Ca risk + tamoxifen endometrial risk simultaneously",
            "STK11/PJS: intussusception in PJS child = ABDOMINAL EMERGENCY — any abdominal pain → urgent assessment; double-balloon enteroscopy for polyp surveillance",
            "ALL endometrial Ca at diagnosis: universal MMR IHC (MLH1/MSH2/MSH6/PMS2) on tumour — clinical standard for Lynch detection + immunotherapy eligibility",
        ],
        "gene_panel_note": (
            "Hereditary Endometrial Cancer Germline Panel (clinical 2024): "
            "MMR/LYNCH GENES (MSI-H → immunotherapy): "
            "  MLH1: 40-50% lifetime endometrial; CRC dominant; aspirin CAPP2; somatic methylation pitfall; "
            "  MSH6: 40-71% lifetime — HIGHEST Lynch endometrial; MSI-L/MSS 30% pitfall — IHC primary; "
            "  MSH2: 40-60% lifetime; urothelial 25-28% HIGHEST Lynch; Muir-Torre; EPCAM MLPA; "
            "COWDEN/mTOR PATHWAY: "
            "  PTEN: 28-44% endometrial (dominant female cancer); macrocephaly; Lhermitte-Duclos; breast 85%; "
            "ULTRA-HYPERMUTATED (IMMUNOTHERAPY — NOT MSI): "
            "  POLE (P286R, V411L): TMB >100 mut/Mb MSS; EXCEPTIONAL pembrolizumab response; "
            "    COMPLETE REMISSIONS documented; PPAP polyposis syndrome; "
            "TP53 PATHWAY: "
            "  TP53 germline (LFS): rare for endometrial Ca specifically; serous-like phenotype; avoid radiation; "
            "  TP53 somatic: p53-abnormal TCGA subtype — WORST prognosis; carboplatin/paclitaxel ± trastuzumab; "
            "HBOC: "
            "  BRCA1: 2-3x endometrial RR; serous-like; TH+BSO recommended; "
            "POLYPOSIS: "
            "  STK11/PJS: 9-12x endometrial RR; SCTAT pathognomonic; 93% overall Ca by 70yr; "
            "UNIVERSAL TUMOUR TESTING: "
            "  MMR IHC on ALL endometrial Ca (NCCN 2024, ESGO 2023): "
            "    Identifies Lynch syndrome; "
            "    Directs immunotherapy (MSI-H or POLE → pembrolizumab/dostarlimab); "
            "    Enables TCGA molecular classification (prognosis + treatment guidance); "
            "  POLE somatic testing: FoundationOne CDx / AmpliSeq panel / NGS-TMB; "
            "  p53 IHC: all endometrial Ca biopsies for molecular classification"
        ),
    }


def generate_breakdown() -> dict:
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
        "atlas":          "Hereditary-Endometrial-Cancer-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "n_genes":        len(genes_data),
        "total_patients": 320,
        "genes":          genes_data,
    }


def generate_definitions() -> dict:
    definitions = [
        {
            "term": "MSH6-MSI-L-Pitfall-Endometrial-Lynch-Protocol",
            "definition": (
                "MSH6 Lynch Syndrome — MSI-L Pitfall and Correct Testing Protocol: "
                "THE PROBLEM — MSI-PCR FALSE NEGATIVES IN MSH6 ENDOMETRIAL CA: "
                "  Standard 5-marker MSI panel (BAT25, BAT26, NR-21, NR-24, Mono-27): "
                "    These markers are MONONUCLEOTIDE repeats; "
                "    MutSα (MSH2-MSH6) repairs single-base mismatches; "
                "    MutSβ (MSH2-MSH3) repairs IDLs at dinucleotide repeats; "
                "    MSH6 LOF → MutSα absent but MutSβ (MSH2-MSH3) INTACT → "
                "      Dinucleotide microsatellites still repaired → panel appears STABLE (MSS or MSI-L); "
                "    FALSE NEGATIVE RATE: ~30% MSH6 Lynch endometrial Ca are MSI-L or MSS by PCR; "
                "CORRECT TESTING APPROACH: "
                "  Step 1: IHC FIRST — MLH1, PMS2, MSH2, MSH6 simultaneously; "
                "    Isolated MSH6 IHC loss (MSH2 intact) = MSH6 Lynch (confirmed without PCR); "
                "  Step 2: If IHC equivocal → NGS-based MSI (Foundation Medicine / MSI-Sensor); "
                "    NGS-MSI is more sensitive than PCR for MSH6 tumours; "
                "  Step 3: Germline MSH6 sequencing + MLPA if IHC shows MSH6 loss; "
                "IHC INTERPRETATION: "
                "  Isolated MSH6 loss (MSH2 normal): MSH6 germline (Lynch 3); "
                "    Exception: somatic MSH6 second hit only → confirm germline; "
                "  MSH2 + MSH6 co-loss: MSH2 germline (MSH2 LOF destabilises MSH6); "
                "  MLH1 + PMS2 co-loss: MLH1 germline or somatic methylation (sporadic); "
                "  PMS2 alone: PMS2 germline; "
                "IMMUNOTHERAPY ELIGIBILITY: "
                "  MSH6 Lynch endometrial Ca MSI-H on IHC → pembrolizumab eligible; "
                "  MSI-L/MSS by PCR but MSH6 IHC loss → still pembrolizumab eligible (Lynch-proven); "
                "  Submit claim as dMMR (deficient mismatch repair by IHC) if MSI-L by PCR. "
            ),
        },
        {
            "term": "POLE-Ultra-Hypermutated-Pembrolizumab-Protocol",
            "definition": (
                "POLE-Mutated Endometrial Cancer — Immunotherapy Protocol and MSS Distinction: "
                "POLE ULTRA-HYPERMUTATED PHENOTYPE: "
                "  Mutation burden: >100 mut/Mb (often 200-800 mut/Mb); "
                "  Dominant mutations: C>A transversions in TCT→TAT context; "
                "  Microsatellite STATUS: MICROSATELLITE STABLE (MSS) — "
                "    POLE proofreads leading-strand synthesis: "
                "      POLE exonuclease LOF → single-nucleotide mismatches; "
                "      Short tandem repeats (microsatellites) are repaired by MutSβ (MSH2-MSH3); "
                "      Microsatellites remain STABLE in POLE mutated tumours; "
                "  CRITICAL: DO NOT USE MSI-PCR to rule in/rule out POLE; "
                "HOW TO DETECT POLE IN ENDOMETRIAL CA: "
                "  NGS sequencing of POLE exonuclease domain hotspots: P286R, V411L, L424V, S297F; "
                "  TMB measurement by NGS (Foundation Medicine CDx, MSK-IMPACT, Tempus): "
                "    TMB >10 mut/Mb: FDA pembrolizumab approval (tumour-agnostic FDA2020); "
                "    TMB >100 mut/Mb: POLE ultra-mutated — exceptional response range; "
                "  Germline POLE: for PPAP (colonic polyps + endometrial Ca family); "
                "  Somatic POLE: 7-10% all endometrial Ca — much more common than germline; "
                "PEMBROLIZUMAB IN POLE ENDOMETRIAL CA: "
                "  EXCEPTIONAL response rates documented: "
                "    90-100% response in POLE exonuclease-mutated advanced endometrial Ca on PD-1/L1; "
                "    COMPLETE REMISSIONS: durable complete responses including in stage IV disease; "
                "    NEOADJUVANT pembrolizumab: some centres using pre-operative pembrolizumab → "
                "      pathological complete response → possible surgery avoidance (experimental); "
                "  Rationale: "
                "    Ultra-high TMB → abundant neoantigens → T-cell recognition; "
                "    High PD-L1 expression in POLE tumours; "
                "    IFN-γ-rich tumour microenvironment; "
                "POLE SOMATIC PROGNOSIS: "
                "  TCGA POLE-ultramutated endometrial Ca = BEST prognosis molecular subtype; "
                "  Even grade 3 poorly differentiated POLE endometrial Ca: excellent OS; "
                "  Chemotherapy: may not add benefit over immunotherapy; evolving data; "
                "CLINICAL MANAGEMENT: "
                "  Pembrolizumab 200mg Q3W (or 400mg Q6W) for advanced POLE endometrial Ca; "
                "  Combination: lenvatinib + pembrolizumab (KEYNOTE-775) — greater benefit in pMMR; "
                "    POLE tumours: pembrolizumab monotherapy likely adequate (exceptional responders); "
                "  Surgery: standard for resectable POLE endometrial Ca; adjuvant pembrolizumab trials."
            ),
        },
        {
            "term": "PTEN-Cowden-Endometrial-Macrocephaly-Protocol",
            "definition": (
                "PTEN / Cowden Syndrome (PHTS) — Endometrial Cancer and Diagnosis Protocol: "
                "CLINICAL DIAGNOSIS — PTEN HAMARTOMA TUMOUR SYNDROME (PHTS): "
                "  International Cowden Consortium (ICC) 2013 criteria — PATHOGNOMONIC features: "
                "    Lhermitte-Duclos disease (adult-onset): PATHOGNOMONIC — test PTEN immediately; "
                "    Trichilemmoma (histologically confirmed): face/neck skin — PATHOGNOMONIC; "
                "    Mucosal papillomatosis (cobblestone mucosal papules): PATHOGNOMONIC; "
                "  MAJOR criteria: macrocephaly; breast Ca; thyroid Ca (non-medullary); endometrial Ca; "
                "  MINOR criteria: autism spectrum; colon polyps; fibrocystic breast; lipoma; others; "
                "MACROCEPHALY — KEY SCREENING SIGN: "
                "  Occipitofrontal circumference (OFC) ≥97th centile (≥58 cm women; ≥60 cm men); "
                "  Present in >95% PHTS patients — most sensitive clinical feature; "
                "  Measure OFC on ALL patients presenting with: "
                "    Multiple cancers; endometrial + breast; young endometrial Ca (<50yr); "
                "    Thyroid + endometrial combination; "
                "  If macrocephaly present + one major criterion → germline PTEN testing; "
                "ENDOMETRIAL CANCER IN PHTS: "
                "  28-44% lifetime endometrial Ca (DOMINANT cancer in female PHTS); "
                "  Onset earlier: mean 44-50yr (vs 62yr sporadic); "
                "  Histology: endometrioid grade 1-2 most common (MSS unlike Lynch); "
                "  PRECURSOR: endometrial intraepithelial neoplasia (EIN): "
                "    PTEN somatic loss in 50-80% precursor + invasive endometrial Ca; "
                "    PTEN LOF → PI3K-Akt-mTOR → endometrial proliferation; "
                "SURVEILLANCE: "
                "  Annual pelvic USS + endometrial biopsy from age 35; "
                "  CA-125 + biopsy annually from age 35-40; "
                "  TH+BSO after childbearing: recommended — eliminates endometrial + ovarian risk; "
                "  Annual thyroid USS: 35% thyroid Ca risk in PHTS; follicular most common; "
                "  Annual breast MRI + mammography from age 30-35 (85% breast Ca risk); "
                "mTOR INHIBITOR: "
                "  Everolimus (mTOR): rationale — PTEN LOF → mTOR constitutive activation; "
                "  Endometrial Ca: GOG-0248 trial (everolimus + letrozole); limited data; "
                "  PTEN/PIK3CA + everolimus: some responses in mTOR-activated endometrial Ca. "
            ),
        },
        {
            "term": "Universal-MMR-IHC-Endometrial-Ca-Protocol",
            "definition": (
                "Universal MMR IHC on Endometrial Cancer — Standard of Care Protocol: "
                "RATIONALE — UNIVERSAL TESTING: "
                "  Lynch syndrome accounts for ~3-5% of all endometrial Ca (underdiagnosed); "
                "  Selective testing by family history/age misses >50% Lynch endometrial Ca; "
                "  Universal MMR IHC is standard of care (NCCN 2024; ESGO 2023; SGO 2021); "
                "  Also identifies POLE + p53-abn for prognosis and treatment (TCGA classification); "
                "IHC PANEL: MLH1 + PMS2 + MSH2 + MSH6 (all four on same block): "
                "INTERPRETATION LOGIC: "
                "  All 4 proteins intact: MMR-proficient (pMMR); consider POLE testing + p53 IHC; "
                "  MLH1 + PMS2 co-loss: "
                "    MOST COMMON MMR IHC loss pattern; "
                "    Reflex BRAF V600E: positive → sporadic methylation (NOT Lynch); "
                "    BRAF negative → reflex MLH1 promoter methylation PCR; "
                "    Methylation negative → germline MLH1 testing; "
                "  MSH2 + MSH6 co-loss: MSH2 germline most likely (+ EPCAM MLPA); "
                "  Isolated MSH6 loss: MSH6 germline; "
                "  Isolated PMS2 loss: PMS2 germline; "
                "  MSH2 + MSH6 + MLH1 + PMS2 all lost: CMMRD (biallelic or extreme alleles); "
                "MOLECULAR CLASSIFICATION (TCGA FRAMEWORK): "
                "  Step 1: POLE exonuclease sequencing (P286R, V411L hotspots) — if positive: POLE subtype; "
                "  Step 2: MMR IHC (if POLE negative) — if dMMR: MMR-deficient subtype; "
                "  Step 3: p53 IHC (if POLE negative + pMMR): "
                "    p53 aberrant: p53-abnormal subtype (WORST prognosis); "
                "    p53 normal: NSMP (no specific molecular profile) subtype; "
                "TREATMENT IMPLICATIONS: "
                "  POLE ultra-mutated: immunotherapy exceptional response; consider pembrolizumab; "
                "  dMMR/MSI-H: pembrolizumab (KEYNOTE-158) + dostarlimab (RUBY); lenvatinib+pembro; "
                "  p53-abnormal: carboplatin + paclitaxel ± trastuzumab (if HER2+); "
                "  NSMP: carboplatin + paclitaxel; progestin for Grade 1 localised."
            ),
        },
        {
            "term": "STK11-PJS-Endometrial-SCTAT-Protocol",
            "definition": (
                "STK11 / Peutz-Jeghers Syndrome — Gynaecological Cancer Surveillance: "
                "GYNAECOLOGICAL CANCER RISK IN PJS: "
                "  Endometrial Ca: 9x RR (lifetime ~9-12%); onset younger; "
                "  Cervical Ca (minimal deviation adenocarcinoma = adenoma malignum): "
                "    Rare but PJS-specific; well-differentiated cervical adenocarcinoma; "
                "    Often mucin-rich; standard IHC may show normal appearance; "
                "    HPV NEGATIVE (unlike sporadic cervical Ca); "
                "  Ovarian: "
                "    SCTAT (Sex Cord Tumour with Annular Tubules) — PATHOGNOMONIC: "
                "      Bilateral multifocal calcifications on transvaginal USS; "
                "      Small (<1cm); oestrogen-secreting (irregular menses); "
                "      PATHOGNOMONIC for PJS when multifocal bilateral; "
                "      Sporadic SCTAT: unilateral; usually benign; "
                "      PJS SCTAT: can be malignant (~20%); "
                "    Ovarian granulosa cell tumour: elevated risk; "
                "GYNAECOLOGICAL SURVEILLANCE IN PJS: "
                "  From puberty: annual pelvic USS (SCTAT detection); "
                "  Annual cervical smear: glandular cells (adenocarcinoma surveillance); "
                "    Liquid-based cytology for glandular abnormalities; "
                "  Annual endometrial biopsy/USS from age 30-35; "
                "  Serum CA-125 + inhibin B (SCTAT marker) annually; "
                "  Consider TH+BSO after childbearing (cumulative risk); "
                "  Combined oral contraceptive: endometrial + ovarian protection; "
                "GI EMERGENCIES — INTUSSUSCEPTION: "
                "  ANY PJS patient with abdominal pain/vomiting → urgent assessment; "
                "  Plain AXR + USS: lead-point intussusception visible; "
                "  Small bowel: most common site; ileocolic; "
                "  Management: "
                "    Air/hydrostatic enema reduction: for ileocolic in children; "
                "    Surgery: open or laparoscopic reduction + polypectomy; "
                "    Intraoperative enteroscopy: survey entire small bowel at laparotomy; "
                "SMALL BOWEL SURVEILLANCE: "
                "  Double-balloon enteroscopy (DBE) + video capsule from age 8-10yr; "
                "  DBE: every 3yr; remove polyps >1.5cm (lead-point risk); "
                "  Colonoscopy 2-3 yearly; gastroscopy 2-3 yearly; "
                "CANCER OVERALL RISK: "
                "  93% cumulative by age 70 — management by multidisciplinary team essential."
            ),
        },
        {
            "term": "BRCA1-Endometrial-TH-BSO-Protocol",
            "definition": (
                "BRCA1 / Endometrial Cancer — TH+BSO Recommendation and Serous Phenotype: "
                "ENDOMETRIAL CANCER IN BRCA1: "
                "  Relative risk: 2-3x vs general population (moderate elevation); "
                "  Absolute lifetime risk: ~7-9% (vs 3% background); "
                "  PHENOTYPE: serous/non-endometrioid histology; "
                "    HER2 overexpression common in serous endometrial Ca; "
                "    p53-abnormal molecular subtype in many BRCA1 endometrial Ca; "
                "    Clinically similar to BRCA1-associated serous ovarian Ca; "
                "  Mechanism: BRCA1-associated endometrial Ca — HRD tumours; "
                "    Platinum chemotherapy + olaparib sensitivity; "
                "BSO vs TH+BSO IN BRCA1: "
                "  BSO alone (original recommendation): "
                "    Eliminates ovarian Ca risk; reduces breast Ca risk; "
                "    Does NOT remove uterus; endometrial Ca risk remains; "
                "  TH+BSO (updated recommendation — most expert centres): "
                "    Removes uterus + cervix + both tubes + both ovaries; "
                "    Eliminates endometrial Ca risk; "
                "    Eliminates need for tamoxifen-associated endometrial Ca monitoring "
                "      (if taking tamoxifen for breast Ca risk reduction); "
                "    Eliminates ongoing cervical smear requirement; "
                "    Operative time: +30-45 min (minimal additional risk); "
                "    Morbidity: similar to BSO alone (laparoscopic approach); "
                "  HORMONES AFTER TH+BSO: "
                "    HRT after BSO/TH+BSO for BRCA1 premenopausal: safe + indicated; "
                "    Reduces post-BSO menopausal symptoms (hot flushes, bone loss); "
                "    Continuous combined (progestogen + oestrogen): minimal endometrial risk post-hysterectomy; "
                "    Duration: until natural menopause age (typically 51yr); "
                "TIMING OF TH+BSO: "
                "  Age 35-40 after childbearing complete; "
                "  Individualised: some patients with BRCA1 choose earlier (32-35) for ovarian Ca risk; "
                "  Later: up to 45 if childbreaking desires pending; ovarian Ca risk increases >40; "
                "PARP INHIBITOR FOR BRCA1 ENDOMETRIAL Ca: "
                "  Limited endometrial Ca-specific data; "
                "  HRD test (Myriad myChoice): if positive → olaparib trials; "
                "  Ovarian Ca data (SOLO1/SOLO2) supports BRCA1 HRD sensitivity."
            ),
        },
        {
            "term": "MLH1-Methylation-Sporadic-vs-Germline-Lynch-Protocol",
            "definition": (
                "MLH1 Promoter Methylation — Distinguishing Sporadic MSI-H from Lynch Syndrome: "
                "THE PROBLEM: MLH1 + PMS2 CO-LOSS ON IHC — IS IT LYNCH OR SPORADIC? "
                "  MLH1+PMS2 co-loss = most common MMR IHC loss pattern in endometrial Ca; "
                "  Two causes: "
                "    1. GERMLINE MLH1 mutation (Lynch syndrome) — HEREDITARY; "
                "    2. SOMATIC MLH1 PROMOTER METHYLATION — SPORADIC (NOT Lynch); "
                "  Frequency: "
                "    Sporadic methylation: ~80-90% of MLH1+PMS2 co-loss endometrial Ca; "
                "    Lynch MLH1: ~10-20% of MLH1+PMS2 co-loss endometrial Ca; "
                "STEP-BY-STEP DISTINCTION PROTOCOL: "
                "  Step 1: MMR IHC shows MLH1 + PMS2 co-loss; "
                "  Step 2: Check BRAF V600E on tumour (more relevant for CRC): "
                "    In endometrial Ca: BRAF V600E less useful (somatic BRAF rare in endometrial Ca); "
                "    Focus on MLH1 methylation instead; "
                "  Step 3: MLH1 PROMOTER METHYLATION PCR on tumour DNA: "
                "    Methylated → SPORADIC MSI-H (reassure patient — low Lynch risk); "
                "    Unmethylated + MLH1+PMS2 loss on IHC → "
                "      Germline MLH1 testing MANDATORY; "
                "      OR: somatic biallelic MLH1 LOF without methylation (less common); "
                "  Step 4: If methylation negative + IHC loss: germline MLH1 sequencing + MLPA; "
                "IMMUNOTHERAPY ELIGIBILITY (REGARDLESS OF LYNCH STATUS): "
                "  MLH1+PMS2 loss (sporadic or Lynch) = dMMR → pembrolizumab eligible; "
                "  Germline Lynch testing does NOT affect immunotherapy eligibility; "
                "  BUT: germline status affects cascade family testing and cancer surveillance; "
                "CAPP2 ASPIRIN: "
                "  Germline Lynch MSH2: aspirin 600mg/day → 50% CRC risk reduction (CAPP2); "
                "  Consider in all Lynch carriers who can tolerate aspirin; "
                "  Minimum 2 years; assess GI tolerance; check for aspirin contraindications."
            ),
        },
        {
            "term": "Hereditary-Endometrial-Cancer-Differential-Diagnosis-Guide",
            "definition": (
                "Differential Diagnosis — Hereditary Endometrial Cancer Gene Selection: "
                "WHEN TO SUSPECT HEREDITARY ENDOMETRIAL CANCER: "
                "  Endometrial Ca age <50yr; "
                "  Personal or family history of Lynch-spectrum cancers (CRC, gastric, urothelial, ovarian); "
                "  Synchronous endometrial + ovarian Ca (Lynch-like synchronous tumours); "
                "  PTEN-associated features: macrocephaly, thyroid Ca, breast Ca, Lhermitte-Duclos; "
                "  Personal history of hamartomatous polyposis or perioral pigmentation (PJS); "
                "  Serous endometrial Ca at young age (BRCA1/TP53); "
                "  High-grade endometrial Ca with exceptional immunotherapy response (POLE); "
                "GENE SELECTION BY CLINICAL SCENARIO: "
                "  Endometrial Ca + CRC family history: MLH1/MSH2/MSH6 Lynch first; "
                "  Endometrial Ca + multiple sebaceous tumours: MSH2 (Muir-Torre) MANDATORY; "
                "  Endometrial Ca + macrocephaly + thyroid Ca: PTEN (Cowden) first; "
                "  Endometrial Ca + high TMB on NGS (MSS): POLE exonuclease sequencing; "
                "  Endometrial Ca + serous histology + young age: BRCA1 + TP53; "
                "  Endometrial Ca + CRC + small bowel polyps: STK11/PJS; "
                "ENDOMETRIAL Ca MOLECULAR SUBTYPE → TREATMENT: "
                "  POLE (ultra-mutated MSS, TMB >100): pembrolizumab exceptional response; "
                "  dMMR/MSI-H (Lynch or sporadic): pembrolizumab + lenvatinib (KEYNOTE-775); "
                "  p53-abnormal (serous): carboplatin/paclitaxel ± trastuzumab (HER2+); "
                "  NSMP: carboplatin/paclitaxel standard; hormone therapy for grade 1; "
                "ACTIVE HEREDITARY SURVEILLANCE VS POPULATION RISK: "
                "  MSH6 Lynch: endometrial 40-71% — annual biopsy from age 35 is standard; "
                "  PTEN/Cowden: 28-44% — annual biopsy + macrocephaly screening trigger; "
                "  MLH1/MSH2 Lynch: annual biopsy from 35 + TH+BSO after childbearing; "
                "  BRCA1: 7-9% (modest) — consider TH+BSO at time of BSO for completeness; "
                "  STK11/PJS: 9-12x RR — annual pelvic surveillance from puberty + biopsy from 35. "
            ),
        },
    ]
    return {
        "atlas":   "Hereditary-Endometrial-Cancer-Atlas",
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
