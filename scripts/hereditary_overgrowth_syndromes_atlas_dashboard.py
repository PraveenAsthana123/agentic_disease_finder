#!/usr/bin/env python3
"""Hereditary-Overgrowth-Syndromes-Atlas — Complete 8-Gene Overgrowth Genetics Atlas
NSD1    (nuclear receptor binding SET domain protein 1; 1992 aa; 5q35.3; AD de novo LOF;
         Sotos syndrome; cerebral gigantism; MOST COMMON overgrowth syndrome ~1:14,000;
         macrocephaly + tall stature + characteristic face + ID (variable);
         seed SEED_BASE+0) ·
EZH2    (enhancer of zeste homolog 2; 746 aa; 7q36.1; AD de novo LOF;
         Weaver syndrome; ACCELERATED OSSEOUS MATURATION PATHOGNOMONIC;
         AML/ALL predisposition; Sotos-like but clinically distinct face;
         seed SEED_BASE+1) ·
GPC3    (glypican 3; 580 aa; Xq26.2; X-linked recessive LOF;
         Simpson-Golabi-Behmel syndrome type 1 (SGBS1);
         SUPERNUMERARY NIPPLES PATHOGNOMONIC; Wilms tumour 10%;
         polydactyly; X-linked males severely affected;
         seed SEED_BASE+2) ·
CDKN1C  (cyclin-dependent kinase inhibitor 1C; 316 aa; 11p15.4; maternal imprinting;
         Beckwith-Wiedemann syndrome type 3 (BWS-CDKN1C-LOF);
         OMPHALOCELE + MACROGLOSSIA + MACROSOMIA TRIAD PATHOGNOMONIC;
         Wilms tumour 7-9%; hemihypertrophy; methylation testing mandatory;
         seed SEED_BASE+3) ·
PTEN    (phosphatase and tensin homolog; 403 aa; 10q23.31; AD LOF;
         PTEN Hamartoma Tumour Syndrome (PHTS); Bannayan-Riley-Ruvalcaba (BRR);
         MACROCEPHALY >2SD ABOVE MEAN triggers PTEN testing; ASD 20%;
         breast cancer lifetime risk 85%; thyroid carcinoma 35%;
         seed SEED_BASE+4) ·
SETD2   (SET domain containing 2; 2564 aa; 3p21.31; AD LOF;
         Luscan-Lumish syndrome; Sotos-like: tall stature + macrocephaly + ID + autism;
         SETD2 H3K36me3 histone methyltransferase; renal clear cell carcinoma risk;
         seed SEED_BASE+5) ·
NFIX    (nuclear factor I X; 391 aa; 19p13.3; AD LOF/GOF;
         Malan syndrome (SOTOS-2) [LOF] / Marshall-Smith syndrome [GOF];
         Sotos-like with intellectual disability and tall stature;
         Marshall-Smith: accelerated bone age + severe ID + distinctive face;
         seed SEED_BASE+6) ·
PIK3CA  (phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha; 1068 aa;
         3q26.32; somatic GOF mosaic;
         PROS (PIK3CA-Related Overgrowth Spectrum): MCAP/CLOVES/hemimegalencephaly/
         limb overgrowth/isolated cerebellar overgrowth; ALPELISIB treatment;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 3030-3037)
"""
import random

SEED_BASE = 3030

ATLAS_GENES = [
    {
        "gene": "NSD1",
        "protein": (
            "NSD1 -- 5q35.3 AD-de-novo-LOF -- 1992aa -- Nuclear-Receptor-Binding-"
            "SET-Domain-Protein-1-Sotos-Syndrome-Cerebral-Gigantism-MOST-COMMON-"
            "Overgrowth-1in14000-Macrocephaly-Tall-ID-OMIM-117550"
        ),
        "locus": "5q35.3",
        "protein_size": (
            "1992 aa / 231 kDa (NSD1; histone methyltransferase; SET domain H3K36me1/2 + H4K20me1; "
            "STRUCTURE: PHD fingers × 5 (chromatin reading) + SET domain (catalytic) + AWS domain; "
            "FUNCTION: "
            "  NSD1 dimethylates H3K36 → active chromatin mark → gene transcription regulation; "
            "  NSD1 also monomethylates H4K20; "
            "  NSD1 is a transcriptional coactivator of nuclear receptors (RAR, RXR, SF1); "
            "  Critical for embryonic growth and development; "
            "SOTOS SYNDROME — CLINICAL: "
            "  PREVALENCE: ~1:14,000 — MOST COMMON overgrowth syndrome; "
            "  TRIAD: Macrocephaly (OFC > 2 SD) + Tall stature + Characteristic facial features; "
            "  CHARACTERISTIC FACE: "
            "    High/broad forehead; downslanting palpebral fissures; "
            "    Long chin; high-arched palate; sparse frontoparietal hair; "
            "  OVERGROWTH: prenatal onset; birth length/weight > 2 SD; postnatal tall stature; "
            "  INTELLECTUAL DISABILITY: "
            "    VARIABLE: mild-moderate 90%; severe 10%; "
            "    Learning difficulties universal even without formal ID; "
            "    Expressive language more impaired than receptive; "
            "  ADVANCED BONE AGE: 1-3 years ahead; "
            "  BEHAVIOURAL: ADHD-like, ASD features in some; poor coordination; "
            "  CARDIAC: CHD 15-20% (ASD, VSD, PDA); "
            "  TUMOUR RISK: WILMS (nephroblastoma) 2-3%; sacrococcygeal teratoma; "
            "    AFP surveillance; "
            "JAPANESE MICRODELETION: "
            "  5q35 microdeletion (~60% of Japanese Sotos) vs intragenic mutation (~70% Europe); "
            "  MLPA ESSENTIAL: large deletions/duplications in 5-15%; "
            "GENE: NSD1; encoded 5q35.3; OMIM gene 606681; Sotos syndrome #117550"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — NSD1 / SOTOS SYNDROME: "
            "  DE NOVO: >95% cases; paternal age effect documented; "
            "  FAMILIAL: <5% with variable expressivity (mildly affected parent); "
            "  RECURRENCE: <1% empiric for de novo; 50% if inherited; "
            "  MOSAICISM: rare mosaic NSD1 → milder somatic Sotos; "
            "TESTING ALGORITHM: "
            "  NSD1 sequencing + MLPA (deletions 5-15% of cases); "
            "  Japanese 5q35 deletion panel if Japanese ancestry; "
            "  WES if NSD1 negative + strong Sotos phenotype → NFIX/SETD2/EZH2"
        ),
        "disease_category": (
            "NSD1-SOTOS-SYNDROME-CEREBRAL-GIGANTISM: "
            "  MOST COMMON overgrowth syndrome (~1:14,000); "
            "  TRIAD: Macrocephaly + tall stature + intellectual disability (variable); "
            "  FACIAL: high forehead + downslant palpebral fissures + long chin CHARACTERISTIC; "
            "  TUMOUR RISK: Wilms 2-3% → AFP + renal USS surveillance; "
            "  MLPA MANDATORY: 5-15% large deletions missed by sequencing alone"
        ),
    },
    {
        "gene": "EZH2",
        "protein": (
            "EZH2 -- 7q36.1 AD-de-novo-LOF -- 746aa -- Enhancer-of-Zeste-Homolog-2-"
            "Weaver-Syndrome-ACCELERATED-OSSEOUS-MATURATION-PATHOGNOMONIC-"
            "AML-ALL-Predisposition-OMIM-277590"
        ),
        "locus": "7q36.1",
        "protein_size": (
            "746 aa / 85 kDa (EZH2; histone methyltransferase; catalytic subunit of PRC2 complex; "
            "STRUCTURE: SET domain (H3K27me1/2/3) + CXC domain + SANT domains; "
            "FUNCTION: "
            "  EZH2 is the catalytic subunit of PRC2 (Polycomb Repressive Complex 2); "
            "  EZH2 trimethylates H3K27 → H3K27me3 → gene SILENCING; "
            "  PRC2: EZH2 + SUZ12 + EED + RbAp46/48; "
            "  EZH2 LOF (Weaver): reduced H3K27me3 → derepression of growth/developmental genes; "
            "  NOTE: EZH2 GOF = Diffuse Large B-Cell Lymphoma (cancer, same gene opposite direction); "
            "WEAVER SYNDROME — CLINICAL: "
            "  ACCELERATED OSSEOUS MATURATION: bone age 2-3 years ahead PATHOGNOMONIC; "
            "    Wrist X-ray; carpal/tarsal fusion; "
            "  OVERGROWTH: prenatal onset; macrocephaly; tall stature; large hands/feet; "
            "  FACIAL: "
            "    OCULAR HYPERTELORISM; broad forehead; round face; "
            "    Prominent ears; wide philtrum; Sotos-like but ROUND face vs long chin; "
            "  INTELLECTUAL DISABILITY: mild-moderate; variable; "
            "  CAMPTODACTYLY: flexion contracture fingers (variable); "
            "  UMBILICAL HERNIA; excessive loose skin; "
            "  HAEMATOLOGICAL MALIGNANCY PREDISPOSITION: "
            "    AML (acute myeloid leukaemia): elevated risk; "
            "    ALL (acute lymphoblastic leukaemia): elevated risk; "
            "    Annual FBC surveillance; IMMEDIATE haematology if fatigue/pallor/bruising; "
            "  CARDIAC: CHD occasional; "
            "SOTOS vs WEAVER — KEY DIFFERENTIAL: "
            "  FACE: Weaver = ROUND + ocular hypertelorism; Sotos = LONG + downslant eyes; "
            "  BONE AGE: both advanced but Weaver often more pronounced; "
            "  GENE: Weaver = EZH2; Sotos = NSD1; "
            "GENE: EZH2; encoded 7q36.1; OMIM gene 601573; Weaver syndrome #277590"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — EZH2 / WEAVER SYNDROME: "
            "  DE NOVO: majority; paternal age effect; "
            "  FAMILIAL: rare; variable expressivity (mildly affected parent); "
            "  GONADAL MOSAICISM: reported; "
            "TESTING: EZH2 sequencing + deletion testing; WES if negative strong phenotype; "
            "HAEMATOLOGY SURVEILLANCE: "
            "  Annual FBC from diagnosis; "
            "  Low threshold for bone marrow biopsy if cytopenias; "
            "MANAGEMENT: "
            "  Physiotherapy for camptodactyly/tone; "
            "  Neurodevelopmental support; "
            "  Ophthalmology surveillance (strabismus, refractive)"
        ),
        "disease_category": (
            "EZH2-WEAVER-SYNDROME-ACCELERATED-BONE-AGE: "
            "  ACCELERATED OSSEOUS MATURATION PATHOGNOMONIC — wrist X-ray at diagnosis; "
            "  AML/ALL PREDISPOSITION — annual FBC surveillance mandatory; "
            "  ROUND FACE + hypertelorism vs Sotos LONG FACE — key Sotos DDx; "
            "  EZH2 PRC2 H3K27me3 enzyme — LOF in Weaver, GOF in lymphoma (same gene!)"
        ),
    },
    {
        "gene": "GPC3",
        "protein": (
            "GPC3 -- Xq26.2 XLR-LOF -- 580aa -- Glypican-3-"
            "Simpson-Golabi-Behmel-Syndrome-Type1-SGBS1-"
            "SUPERNUMERARY-NIPPLES-PATHOGNOMONIC-Wilms-10pct-Polydactyly-XL-OMIM-312870"
        ),
        "locus": "Xq26.2",
        "protein_size": (
            "580 aa / 65 kDa (GPC3; glypican 3; cell-surface heparan sulphate proteoglycan; "
            "GPI-anchored; "
            "STRUCTURE: core protein + heparan sulphate chains × 2 (Ser495, Ser509); "
            "  furin cleavage at Arg358 in ER → GPC3α (N-term) + GPC3β (C-term, GPI-anchored); "
            "FUNCTION: "
            "  GPC3 binds and INHIBITS Wnt signalling → controls cell proliferation; "
            "  GPC3 also inhibits Hedgehog (SHH) signalling; "
            "  GPC3 LOF → UNRESTRICTED Wnt/HH → overgrowth + Wilms tumour susceptibility; "
            "SGBS TYPE 1 — CLINICAL: "
            "  X-LINKED: males SEVERELY affected; females MILDLY affected (lyonization); "
            "  PRENATAL MACROSOMIA: large-for-gestational-age; polyhydramnios; "
            "  POSTNATAL OVERGROWTH: tall stature; macrocephaly; large hands/feet; "
            "  SUPERNUMERARY NIPPLES: PATHOGNOMONIC — 1-8 extra nipples below normal pair; "
            "    POLYTHELIA distinguishes SGBS from other overgrowth syndromes; "
            "  POLYDACTYLY: postaxial polydactyly most common; duplication distal phalanges; "
            "  VISCERAL ANOMALIES: hepatosplenomegaly; intestinal malrotation; "
            "    diaphragmatic hernia; "
            "  CARDIAC: CHD in 40%; VSD/ASD; "
            "  RENAL: kidney abnormalities; "
            "  WILMS TUMOUR (NEPHROBLASTOMA): "
            "    Risk ~10% in males; "
            "    Renal surveillance ultrasound MANDATORY: every 3 months until age 8; "
            "  FACIES: coarse face; broad forehead; wide mouth; macroglossia; "
            "  INTELLECTUAL DISABILITY: mild-moderate in males; "
            "  NEONATAL HYPOGLYCAEMIA: due to macrosomia; "
            "SGBS TYPE 2: OFD1 gene (different gene, different syndrome); "
            "FEMALES: carrier females usually unaffected or mildly affected; "
            "  Rare severely affected female (skewed X-inactivation); "
            "GENE: GPC3; encoded Xq26.2; OMIM gene 300037; SGBS1 #312870"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE LOF — GPC3 / SIMPSON-GOLABI-BEHMEL SYNDROME TYPE 1: "
            "  HEMIZYGOUS MALES: severely affected; "
            "  HETEROZYGOUS FEMALES: carrier; usually mild/unaffected; "
            "  RECURRENCE: 50% of male offspring affected; 50% females carriers; "
            "  PRENATAL: fetal macrosomia on USS; maternal polyhydramnios; "
            "TESTING: GPC3 sequencing + deletion/duplication (MLPA); "
            "WILMS TUMOUR SURVEILLANCE: "
            "  Renal USS every 3 months: birth to age 8 years; "
            "  AFP (alphafetoprotein) every 3 months (Wilms AFP-secreting); "
            "MANAGEMENT: "
            "  Echocardiogram at diagnosis (40% CHD); "
            "  Neonatal glucose monitoring; "
            "  Ophthalmology (coloboma); "
            "  Physiotherapy + speech therapy"
        ),
        "disease_category": (
            "GPC3-SGBS1-SUPERNUMERARY-NIPPLES-WILMS-10pct: "
            "  SUPERNUMERARY NIPPLES (POLYTHELIA) PATHOGNOMONIC — no other overgrowth syndrome; "
            "  WILMS TUMOUR 10% → renal USS every 3 months to age 8 MANDATORY; "
            "  X-LINKED: males severely affected; females mildly/unaffected; "
            "  GPC3 LOF → unrestricted Wnt/HH → overgrowth + tumour susceptibility"
        ),
    },
    {
        "gene": "CDKN1C",
        "protein": (
            "CDKN1C -- 11p15.4 Maternal-Imprinting-LOF -- 316aa -- Cyclin-Dependent-Kinase-"
            "Inhibitor-1C-Beckwith-Wiedemann-Type3-OMPHALOCELE-MACROGLOSSIA-MACROSOMIA-"
            "TRIAD-PATHOGNOMONIC-Wilms-7-9pct-Hemihypertrophy-OMIM-130650"
        ),
        "locus": "11p15.4",
        "protein_size": (
            "316 aa / 35 kDa (CDKN1C; p57KIP2; cyclin-dependent kinase inhibitor; "
            "STRUCTURE: KIP domain (CDK binding) + PCNA-binding motif + nuclear localisation; "
            "FUNCTION: "
            "  CDKN1C/p57KIP2 inhibits cyclin-CDK complexes → G1 ARREST → anti-proliferative; "
            "  IMPRINTING: MATERNALLY expressed; PATERNALLY imprinted (silenced); "
            "  11p15.5 imprinting domain: CDKN1C on maternal chromosome is ACTIVE; "
            "    Paternal allele silenced by KvDMR1 methylation; "
            "  CDKN1C LOF (mutation of maternal copy OR IC2 methylation gain): "
            "    Loss of growth inhibition → fetal overgrowth; "
            "BECKWITH-WIEDEMANN SYNDROME (BWS) OVERVIEW: "
            "  BWS is a clinical diagnosis with MOLECULAR SUBTYPE classification; "
            "  H19/IGF2 methylation gain (~50%): most common; Wilms 7.9%; "
            "  CDKN1C LOF mutations (~10% sporadic, ~50% familial BWS): "
            "    CLASSICAL PHENOTYPE: MOST SEVERE; "
            "    OMPHALOCELE (abdominal wall defect): ~60% of CDKN1C-BWS; "
            "    MACROGLOSSIA: large protruding tongue; airway management; "
            "    MACROSOMIA: birth weight > 2 SD; large for gestational age; "
            "    HEMIHYPERTROPHY (hemihyperplasia): asymmetric body growth; "
            "    HYPOGLYCAEMIA: neonatal; hyperinsulinism; "
            "    EAR PITS/CREASES: auricular pits/post-auricular creases; "
            "    ADRENAL CORTICAL CYTOMEGALY; "
            "  WILMS TUMOUR: 7-9% lifetime risk (highest in CDKN1C subtype); "
            "  HEPATOBLASTOMA: 1% overall BWS; "
            "  IMAPs/methylation testing: "
            "    MANDATORY first-line test: 11p15.5 methylation array; "
            "    CDKN1C sequencing if IC2 methylation abnormal or family history; "
            "CLINICAL NOTE: "
            "  MACROGLOSSIA → speech therapy + tongue reduction if severe; "
            "  Neonatal hypoglycaemia → glucose IV; octreotide if persistent; "
            "GENE: CDKN1C; encoded 11p15.4; OMIM gene 600856; BWS #130650"
        ),
        "inheritance": (
            "MATERNAL IMPRINTING — CDKN1C / BECKWITH-WIEDEMANN SYNDROME TYPE 3: "
            "  CDKN1C is MATERNALLY EXPRESSED — mutation of MATERNAL allele causes BWS; "
            "  FAMILIAL BWS: CDKN1C point mutations in ~50% familial BWS; "
            "  PATERNAL TRANSMISSION: no effect (paternal CDKN1C is IMPRINTED = silenced); "
            "  De novo: less common than H19/IGF2 methylation errors; "
            "  ART/IVF: associated with increased BWS risk (epigenetic mechanism); "
            "TESTING: 11p15.5 methylation analysis FIRST; then CDKN1C sequencing; "
            "SURVEILLANCE: "
            "  Abdominal USS every 3 months (age 0-8): Wilms + hepatoblastoma; "
            "  AFP every 3 months (hepatoblastoma); "
            "  Blood glucose monitoring neonatal period; "
            "  Asymmetry tracking (hemihypertrophy)"
        ),
        "disease_category": (
            "CDKN1C-BWS-OMPHALOCELE-MACROGLOSSIA-MACROSOMIA: "
            "  CLASSICAL BWS TRIAD PATHOGNOMONIC: OMPHALOCELE + MACROGLOSSIA + MACROSOMIA; "
            "  MATERNAL IMPRINTING: only maternal CDKN1C mutations cause BWS; "
            "  METHYLATION TESTING MANDATORY: 11p15.5 methylation first; then sequencing; "
            "  WILMS 7-9% → abdominal USS every 3 months to age 8 MANDATORY"
        ),
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 AD-LOF -- 403aa -- Phosphatase-And-Tensin-Homolog-"
            "PTEN-Hamartoma-Tumour-Syndrome-Bannayan-Riley-Ruvalcaba-"
            "MACROCEPHALY-gt2SD-TESTING-TRIGGER-ASD-20pct-Breast-85pct-Thyroid-35pct-OMIM-158350"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 47 kDa (PTEN; phosphatase and tensin homolog; lipid + protein phosphatase; "
            "STRUCTURE: N-terminal phosphatase domain + C2 domain + C-terminal tail; "
            "CATALYTIC: dephosphorylates PIP3 → PIP2; "
            "FUNCTION: "
            "  PTEN is the primary negative regulator of the PI3K/AKT/mTOR pathway; "
            "  PTEN dephosphorylates PIP3 → reduces AKT activation → ANTI-PROLIFERATIVE; "
            "  PTEN LOF → CONSTITUTIVE AKT/mTOR activation → overgrowth + cancer; "
            "  NUCLEAR PTEN also has chromatin stability and DNA repair functions; "
            "PTEN HAMARTOMA TUMOUR SYNDROME (PHTS): "
            "  UMBRELLA TERM for PTEN LOF germline mutations: "
            "  COWDEN SYNDROME (CS): adults; multiple hamartomas; cancer risk; "
            "  BANNAYAN-RILEY-RUVALCABA (BRR): children; macrocephaly + lipomas + pigmented penile macules; "
            "  ADULT LHERMITTE-DUCLOS (LDDS): dysplastic cerebellar gangliocytoma; "
            "PTEN MACROCEPHALY — KEY TESTING TRIGGER: "
            "  Macrocephaly > 2 SD (98th centile) → PTEN TESTING recommended; "
            "  Macrocephaly > 2.5 SD → PTEN testing strongly indicated; "
            "  Macrocephaly + ASD = 17-20% PTEN mutation rate; "
            "AUTISM SPECTRUM DISORDER: 20% of PTEN PHTS; "
            "  Highest rate among single-gene ASD causes in macrocephalic patients; "
            "CANCER RISKS (LIFETIME): "
            "  BREAST CANCER: 85% lifetime risk → annual MRI from age 30-35; "
            "  THYROID CANCER (non-medullary, follicular/papillary): 35%; "
            "  ENDOMETRIAL CANCER: 28%; "
            "  COLORECTAL CANCER: 9%; "
            "  RENAL CELL CARCINOMA: 33%; "
            "  MELANOMA: elevated; "
            "LIPOMATOSIS: multiple subcutaneous lipomas characteristic of BRR; "
            "VASCULAR ANOMALIES: arteriovenous malformations (AVMs); "
            "PENILE MACULES: pigmented freckling penile shaft — BRR PATHOGNOMONIC in males; "
            "GENE: PTEN; encoded 10q23.31; OMIM gene 601728; Cowden #158350; BRR #153480"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — PTEN / PTEN HAMARTOMA TUMOUR SYNDROME: "
            "  DE NOVO: ~10-15% (rest familial AD); "
            "  FAMILIAL: AD with variable expressivity and incomplete penetrance; "
            "  VARIABLE EXPRESSION: within same family CS vs BRR phenotype; "
            "TESTING: "
            "  PTEN sequencing + large deletion analysis; "
            "  MACROCEPHALY + ASD: PTEN first (highest yield single gene); "
            "  Phospholipid phosphatase activity assay for VUS; "
            "SURVEILLANCE — ADULT: "
            "  BREAST: annual MRI from age 30-35; "
            "  THYROID: annual ultrasound from diagnosis; "
            "  ENDOMETRIUM: annual USS/biopsy from 30-35; "
            "  COLORECTAL: colonoscopy from 35; "
            "  RENAL: renal imaging every 1-2 years from 40; "
            "  SKIN: annual dermatology; "
            "SURVEILLANCE — PAEDIATRIC: "
            "  Annual thyroid USS from diagnosis; "
            "  Macrocephaly monitoring; neurodevelopmental assessment"
        ),
        "disease_category": (
            "PTEN-PHTS-MACROCEPHALY-CANCER-SURVEILLANCE: "
            "  MACROCEPHALY >2SD → PTEN TESTING TRIGGER (most actionable macrocephaly gene); "
            "  BREAST CANCER 85% LIFETIME → annual MRI surveillance mandatory from age 30-35; "
            "  ASD 20%: PTEN highest-yield single-gene in macrocephalic ASD patients; "
            "  AKT/mTOR pathway master regulator: LOF = overgrowth + cancer"
        ),
    },
    {
        "gene": "SETD2",
        "protein": (
            "SETD2 -- 3p21.31 AD-LOF -- 2564aa -- SET-Domain-Containing-2-"
            "Luscan-Lumish-Syndrome-Sotos-Like-Tall-Macrocephaly-ID-Autism-"
            "H3K36me3-Histone-Methyltransferase-Renal-Clear-Cell-Carcinoma-Risk-OMIM-616831"
        ),
        "locus": "3p21.31",
        "protein_size": (
            "2564 aa / 287 kDa (SETD2; SET domain containing 2; histone methyltransferase; "
            "STRUCTURE: AWA domain + SET domain (catalytic H3K36me3) + WW domain + SRI domain; "
            "FUNCTION: "
            "  SETD2 is the SOLE enzyme responsible for H3K36 TRIMETHYLATION (H3K36me3); "
            "  H3K36me3 = active transcription mark; "
            "  SETD2 + NSD1 both modify H3K36 but: "
            "    NSD1: H3K36me1 + H3K36me2 (intermediate steps); "
            "    SETD2: H3K36me3 (final step, most active mark); "
            "  SETD2 H3K36me3 is critical for: "
            "    DNA mismatch repair (MMR) — recruits MSH6 via PWWP domain; "
            "    RNA splicing fidelity; "
            "    Transcriptional elongation; "
            "LUSCAN-LUMISH SYNDROME — CLINICAL: "
            "  SOTOS-LIKE PHENOTYPE: tall stature + macrocephaly + intellectual disability; "
            "  AUTISM SPECTRUM DISORDER: prominent (~50%); "
            "  INTELLECTUAL DISABILITY: moderate; "
            "  BEHAVIOURAL: aggressive behaviour; hyperactivity; "
            "  MACROCEPHALY: OFC often > 2 SD; "
            "  TALL STATURE: prenatal overgrowth; advanced bone age; "
            "  CHARACTERISTIC FACE (distinct from Sotos): "
            "    Broad forehead; deep-set eyes; broad/flat nasal bridge; "
            "  NO specific cancer surveillance published yet (renal CCC risk from somatic SETD2); "
            "    Renal USS every 2-3 years reasonable given SETD2 tumour suppressor biology; "
            "SOMATIC SETD2: "
            "  Somatic SETD2 LOF: most common in renal clear cell carcinoma (ccRCC): 8-13%; "
            "    Also: leukemia, endometrial, lung; "
            "  Germline SETD2 LOF (Luscan-Lumish): rare; distinct from somatic oncology; "
            "GENE: SETD2; encoded 3p21.31; OMIM gene 612778; Luscan-Lumish syndrome #616831"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — SETD2 / LUSCAN-LUMISH SYNDROME: "
            "  DE NOVO: majority; "
            "  VARIABLE: mild parental phenotype possible (familial rare); "
            "TESTING: SETD2 sequencing; part of overgrowth panel; "
            "MANAGEMENT: "
            "  Neurodevelopmental support: speech, OT, behavioural therapy; "
            "  Autism management: structured ABA/EIBI as appropriate; "
            "  Macrocephaly: neuroimaging if severe (screen for hydrocephalus); "
            "  Renal imaging: reasonable surveillance given SETD2 tumour biology; "
            "  Ophthalmology: strabismus common"
        ),
        "disease_category": (
            "SETD2-LUSCAN-LUMISH-SOTOS-LIKE-AUTISM: "
            "  SOTOS-LIKE: tall + macrocephaly + ID — differentiate by EZH2/NSD1 first then SETD2; "
            "  AUTISM ~50%: prominent in Luscan-Lumish; ASD management core; "
            "  H3K36me3 sole enzyme: SETD2 bridges NSD1 (H3K36me2) → final methylation step; "
            "  RENAL ccRCC risk (somatic SETD2 very common): surveillance reasonable"
        ),
    },
    {
        "gene": "NFIX",
        "protein": (
            "NFIX -- 19p13.3 AD-LOF/GOF -- 391aa -- Nuclear-Factor-I-X-"
            "Malan-Syndrome-SOTOS-2-Sotos-Like-Intellectual-Disability-Tall-Stature-"
            "Marshall-Smith-Syndrome-GOF-Accelerated-Bone-Age-OMIM-613389"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "391 aa / 45 kDa (NFIX; nuclear factor I X; transcription factor; NFI family; "
            "STRUCTURE: N-terminal DNA-binding domain (CTF/NF-I type) + proline-rich transactivation; "
            "  Homodimerises and heterodimerises with other NFI members (NFIA, NFIB, NFIC); "
            "FUNCTION: "
            "  NFIX regulates muscle and CNS development; "
            "  NFIX controls chondrocyte differentiation and bone development; "
            "  LOF (Malan syndrome): haploinsufficiency → SOTOS-2; "
            "  GOF (Marshall-Smith syndrome): NFIX mutations that escape NMD → "
            "    aberrant/dominant-negative → distinct severe phenotype; "
            "MALAN SYNDROME (SOTOS-2) — NFIX LOF: "
            "  SOTOS-LIKE: tall stature + macrocephaly + intellectual disability; "
            "  FACIES: similar to Sotos but MILDER; "
            "    Long face; prominent forehead; deep-set eyes; pointed chin; "
            "  INTELLECTUAL DISABILITY: mild-moderate; "
            "  TALL STATURE: childhood; may normalise adult height; "
            "  BEHAVIOURAL: anxiety; ASD features; "
            "  ADVANCED BONE AGE: mild; "
            "MARSHALL-SMITH SYNDROME — NFIX GOF: "
            "  SEVERE/NEONATAL PHENOTYPE: "
            "  MARKEDLY ACCELERATED OSSEOUS MATURATION: carpal bones advanced several years; "
            "  SEVERE INTELLECTUAL DISABILITY; "
            "  DISTINCTIVE FACE: prominent eyes; large forehead; failure to thrive; "
            "  RESPIRATORY FAILURE: upper airway obstruction; neonatal death in severe cases; "
            "  CHOANAL ATRESIA; "
            "  NFIX GOF vs LOF phenotype DIVERGENCE: clinically distinguishable; "
            "GENE: NFIX; encoded 19p13.3; OMIM gene 164005; Malan syndrome #613389; Marshall-Smith #602535"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — NFIX / MALAN SYNDROME (LOF) / MARSHALL-SMITH (GOF): "
            "  MALAN (LOF): de novo haploinsufficiency; "
            "  MARSHALL-SMITH (GOF): de novo; mutations escape NMD → dominant-negative; "
            "  Both de novo; "
            "TESTING: NFIX sequencing; classify LOF vs GOF for syndrome prediction; "
            "MANAGEMENT — MALAN: "
            "  Neurodevelopmental support; physiotherapy; speech; "
            "  Anxiety/behavioural intervention; "
            "  Advanced bone age: orthopaedic monitoring; "
            "MANAGEMENT — MARSHALL-SMITH: "
            "  Respiratory: tracheotomy/NIV if upper airway obstruction; "
            "  Airway surveillance; gastrostomy feeding; "
            "  Intensive neurodevelopmental support"
        ),
        "disease_category": (
            "NFIX-MALAN-SYNDROME-SOTOS2-SOTOS-LIKE: "
            "  LOF = Malan syndrome (SOTOS-2): Sotos-like phenotype, milder than NSD1; "
            "  GOF = Marshall-Smith syndrome: SEVERE — accelerated bone age + respiratory failure; "
            "  SAME GENE OPPOSITE PHENOTYPES: LOF mild-moderate; GOF severe/neonatal; "
            "  DISTINGUISH by mutation type: LOF (haploinsufficiency) vs GOF (NMD-escape)"
        ),
    },
    {
        "gene": "PIK3CA",
        "protein": (
            "PIK3CA -- 3q26.32 Somatic-GOF-Mosaic -- 1068aa -- PI3K-Catalytic-Subunit-Alpha-"
            "PROS-PIK3CA-Related-Overgrowth-Spectrum-MCAP-CLOVES-Hemimegalencephaly-"
            "Limb-Overgrowth-ALPELISIB-Treatment-OMIM-616973"
        ),
        "locus": "3q26.32",
        "protein_size": (
            "1068 aa / 120 kDa (PIK3CA; phosphatidylinositol-4,5-bisphosphate 3-kinase p110α; "
            "STRUCTURE: ras-binding domain (RBD) + C2 domain + helical domain + kinase domain; "
            "  Forms heterodimer with regulatory subunit p85 (PIK3R1); "
            "FUNCTION: "
            "  PIK3CA phosphorylates PIP2 → PIP3; "
            "  PIP3 recruits and activates PDK1 → AKT; "
            "  PIK3CA GOF → constitutive PIP3/AKT/mTOR → LOCALISED OVERGROWTH; "
            "  SOMATIC MOSAIC = post-zygotic mutation → explains tissue-specific/segmental distribution; "
            "PROS — PIK3CA-RELATED OVERGROWTH SPECTRUM: "
            "  MCAP SYNDROME (Megalencephaly-Capillary Malformation-Polymicrogyria): "
            "    Bilateral megalencephaly; capillary malformations; polymicrogyria; "
            "    Hydrocephalus; seizures; "
            "  CLOVES SYNDROME: "
            "    CONGENITAL LIPOMATOUS Overgrowth + Vascular malformations + "
            "    Epidermal naevi + Scoliosis/Skeletal/Spinal; "
            "    Lymphatic/venous malformations; limb asymmetry; "
            "  HEMIMEGALENCEPHALY: unilateral brain hemisphere overgrowth → intractable seizures; "
            "  ISOLATED CEREBELLAR OVERGROWTH; "
            "  FIBROADIPOSE OVERGROWTH (FAO): limb adipose overgrowth; "
            "  FACEOM: focal asymmetric cortical overgrowth of the brain; "
            "DETECTION: "
            "  MOSAIC — LOW ALLELE FREQUENCY: "
            "    Standard NGS MISSES mosaic PIK3CA (<1-5% allele frequency); "
            "    DEEP SEQUENCING (>500-1000×) or droplet digital PCR required; "
            "    Affected tissue biopsy > blood for detection; "
            "  PIK3CA hotspot mutations: p.His1047Arg (most common), p.His1047Leu, "
            "    p.Glu545Lys, p.Glu542Lys; "
            "ALPELISIB (BYL719) TREATMENT: "
            "  PIK3CA-specific inhibitor; FDA approved for PROS (2022); "
            "  Targets mutant PIK3CA specifically; "
            "  Response in vascular and overgrowth components; "
            "  Monitoring: glucose (hyperglycaemia); rash; stomatitis; "
            "GENE: PIK3CA; encoded 3q26.32; OMIM gene 171834; MCAP #602501; CLOVES/PROS #616973"
        ),
        "inheritance": (
            "SOMATIC MOSAIC GOF — PIK3CA / PROS: "
            "  POST-ZYGOTIC SOMATIC MUTATION: not inherited; recurrence risk VERY LOW; "
            "  GERMLINE PIK3CA GOF: not viable (embryonic lethal for most activating mutations); "
            "  DETECTION: affected tissue + deep sequencing (>500×); ddPCR; "
            "  ALLELE FREQUENCY: 1-15% in blood; higher in affected tissue; "
            "MANAGEMENT: "
            "  ALPELISIB: PIK3CA inhibitor FDA-approved for PROS (Vijoice™); "
            "  Vascular malformations: sclerotherapy + surgery + sirolimus; "
            "  Seizures: anti-epileptic drugs + hemispherectomy in severe hemimegalencephaly; "
            "  Orthopaedic: limb asymmetry; epiphysiodesis; "
            "  Regular surveillance: brain MRI (megalencephaly); USS (retroperitoneal lipomas)"
        ),
        "disease_category": (
            "PIK3CA-PROS-SOMATIC-MOSAIC-ALPELISIB: "
            "  SOMATIC MOSAIC: post-zygotic → tissue-specific/segmental overgrowth; "
            "  MCAP/CLOVES/hemimegalencephaly: major PROS phenotypes; "
            "  ALPELISIB (Vijoice™) FDA-APPROVED: first targeted treatment for PIK3CA-PROS; "
            "  DEEP SEQUENCING MANDATORY: standard NGS MISSES low-allele-frequency mosaic PIK3CA"
        ),
    },
]


def _generate_patients_for_gene(gene: str, seed: int) -> list:
    """Generate 40 synthetic patients for one overgrowth syndrome gene."""
    rng = random.Random(seed)

    severity_params = {
        "NSD1":    {"severe_pct": 0.15, "moderate_pct": 0.60, "mild_pct": 0.25,
                    "age_min": 0, "age_max": 36, "iq_mean": 78, "iq_sd": 15},
        "EZH2":    {"severe_pct": 0.15, "moderate_pct": 0.55, "mild_pct": 0.30,
                    "age_min": 0, "age_max": 30, "iq_mean": 80, "iq_sd": 14},
        "GPC3":    {"severe_pct": 0.30, "moderate_pct": 0.50, "mild_pct": 0.20,
                    "age_min": 0, "age_max": 24, "iq_mean": 74, "iq_sd": 16},
        "CDKN1C":  {"severe_pct": 0.35, "moderate_pct": 0.45, "mild_pct": 0.20,
                    "age_min": 0, "age_max": 6, "iq_mean": 86, "iq_sd": 14},
        "PTEN":    {"severe_pct": 0.10, "moderate_pct": 0.35, "mild_pct": 0.55,
                    "age_min": 0, "age_max": 60, "iq_mean": 92, "iq_sd": 13},
        "SETD2":   {"severe_pct": 0.20, "moderate_pct": 0.55, "mild_pct": 0.25,
                    "age_min": 0, "age_max": 30, "iq_mean": 72, "iq_sd": 14},
        "NFIX":    {"severe_pct": 0.25, "moderate_pct": 0.50, "mild_pct": 0.25,
                    "age_min": 0, "age_max": 36, "iq_mean": 76, "iq_sd": 15},
        "PIK3CA":  {"severe_pct": 0.40, "moderate_pct": 0.40, "mild_pct": 0.20,
                    "age_min": 0, "age_max": 24, "iq_mean": 84, "iq_sd": 16},
    }
    p = severity_params.get(gene, {"severe_pct": 0.25, "moderate_pct": 0.50, "mild_pct": 0.25,
                                    "age_min": 0, "age_max": 30, "iq_mean": 80, "iq_sd": 14})

    features_map = {
        "NSD1":    [["tall-stature-macrocephaly-high-forehead"],
                    ["downslant-palpebral-fissures-long-chin"],
                    ["mild-id-advanced-bone-age", "adhd-features"],
                    ["wilms-tumour-2-3pct", "sacrococcygeal-teratoma"]],
        "EZH2":    [["accelerated-osseous-maturation-wrist-xray-pathognomonic"],
                    ["round-face-hypertelorism-camptodactyly"],
                    ["aml-predisposition-annual-fbc"],
                    ["sotos-like-advanced-bone-age-2-3yr"]],
        "GPC3":    [["supernumerary-nipples-pathognomonic", "polydactyly-postaxial"],
                    ["macrosomia-hepatosplenomegaly", "coarse-facies"],
                    ["wilms-tumour-renal-USS-3mo", "cardiac-chd-40pct"],
                    ["neonatal-hypoglycaemia", "intestinal-malrotation"]],
        "CDKN1C":  [["omphalocele-abdominal-wall-defect", "macroglossia-airway"],
                    ["macrosomia-hemihypertrophy", "ear-creases-pits"],
                    ["neonatal-hypoglycaemia-hyperinsulinism"],
                    ["wilms-tumour-7-9pct", "hepatoblastoma"]],
        "PTEN":    [["macrocephaly-gt2SD-ASD-20pct", "lipomatosis-multiple"],
                    ["breast-cancer-85pct-mri-surveillance"],
                    ["thyroid-cancer-35pct-annual-USS"],
                    ["penile-macules-BRR-pathognomonic", "vascular-AVM"]],
        "SETD2":   [["sotos-like-tall-macrocephaly", "autism-50pct-prominent"],
                    ["moderate-id-aggressive-behaviour"],
                    ["broad-flat-nasal-bridge-deep-set-eyes"],
                    ["renal-surveillance-h3k36me3-biology"]],
        "NFIX":    [["malan-sotos2-tall-macrocephaly-mild-id"],
                    ["marshall-smith-accelerated-bone-severe-respiratory"],
                    ["anxiety-asd-features", "advanced-bone-age"],
                    ["pointed-chin-long-face-deep-set-eyes"]],
        "PIK3CA":  [["mcap-megalencephaly-capillary-malformation-polymicrogyria"],
                    ["cloves-lipomatous-overgrowth-vascular-epidermal"],
                    ["hemimegalencephaly-seizures-intractable"],
                    ["limb-overgrowth-asymmetry-fibroadipose"]],
    }
    feature_options = features_map.get(gene, [["overgrowth-tall-stature"]])

    treatment_map = {
        "NSD1":    ["neurodevelopmental-support-speech-OT",
                    "wilms-surveillance-AFP-renal-USS",
                    "cardiac-monitoring-CHD",
                    "advanced-bone-age-orthopaedic-monitoring"],
        "EZH2":    ["annual-FBC-haematology-AML-surveillance",
                    "physiotherapy-camptodactyly",
                    "neurodevelopmental-support",
                    "bone-age-xray-monitoring"],
        "GPC3":    ["renal-USS-every-3mo-wilms-surveillance",
                    "AFP-3-monthly",
                    "echocardiogram-CHD-40pct",
                    "neonatal-glucose-monitoring"],
        "CDKN1C":  ["abdominal-USS-3mo-wilms-hepatoblastoma",
                    "AFP-surveillance",
                    "neonatal-glucose-IV-octreotide",
                    "macroglossia-speech-therapy-tongue-reduction"],
        "PTEN":    ["annual-breast-MRI-from-30-35",
                    "annual-thyroid-USS",
                    "endometrial-surveillance-from-30",
                    "neurodevelopmental-ASD-management",
                    "renal-imaging-every-1-2yr"],
        "SETD2":   ["neurodevelopmental-ABA-EIBI-autism",
                    "renal-USS-surveillance",
                    "speech-OT-physiotherapy",
                    "neuroimaging-macrocephaly"],
        "NFIX":    ["neurodevelopmental-support-malan",
                    "respiratory-tracheotomy-NIV-marshall-smith",
                    "physiotherapy-bone-age-monitoring",
                    "anxiety-management-CBT"],
        "PIK3CA":  ["alpelisib-PIK3CA-inhibitor-FDA-approved",
                    "sclerotherapy-vascular-malformations",
                    "antiepileptic-AED-seizures",
                    "hemispherectomy-hemimegalencephaly-severe",
                    "epiphysiodesis-limb-length-discrepancy"],
    }
    treatments = treatment_map.get(gene, ["neurodevelopmental-support"])

    mutation_map = {
        "NSD1":    ["p.Arg1984Ter", "p.Ser2326Ter", "c.5005del-frameshift",
                    "5q35-microdeletion-MLPA", "p.Arg1146Cys-SET-domain",
                    "c.6274C>T-exon20"],
        "EZH2":    ["p.Tyr646Cys", "p.Arg679His", "p.Ala677Thr-SET-domain",
                    "p.Arg685His", "c.2053C>T-exon16",
                    "p.Ile714Val"],
        "GPC3":    ["c.1686del-frameshift", "p.Arg373Ter",
                    "Xq26.2-deletion-MLPA", "p.Cys424Arg-HS-chain",
                    "c.747+1G>A-splice"],
        "CDKN1C":  ["p.Arg279Ter-maternal", "p.Leu204Pro-KIP-domain",
                    "p.Gln249Ter-maternal", "c.452del-frameshift-maternal",
                    "11p15-IC2-methylation-gain"],
        "PTEN":    ["p.Arg130Ter", "p.Arg173Cys", "p.Asp268Tyr-catalytic",
                    "c.388C>T-exon5", "10q23-deletion-MLPA",
                    "p.Ile101Thr"],
        "SETD2":   ["p.Arg1625Ter-SET-domain", "c.6478C>T-exon14",
                    "p.Leu1822Pro", "c.3576del-frameshift",
                    "p.Arg2510Ter"],
        "NFIX":    ["p.Arg391Ter-LOF-Malan", "c.969+1G>A-splice-LOF",
                    "p.Ala314Thr-GOF-Marshall-Smith",
                    "p.Arg388Cys-LOF", "c.741del-frameshift-Malan"],
        "PIK3CA":  ["p.His1047Arg-mosaic-hotspot", "p.His1047Leu-mosaic",
                    "p.Glu545Lys-mosaic", "p.Glu542Lys-mosaic",
                    "p.Gln546Lys-mosaic", "p.Cys420Arg-mosaic"],
    }
    mutations = mutation_map.get(gene, ["unknown"])

    patients = []
    for i in range(40):
        rand = rng.random()
        if rand < p["severe_pct"]:
            severity = "severe"
        elif rand < p["severe_pct"] + p["moderate_pct"]:
            severity = "moderate"
        else:
            severity = "mild"
        age_dx = rng.randint(p["age_min"], p["age_max"])
        iq = max(40, int(rng.normalvariate(p["iq_mean"], p["iq_sd"])))
        # Gene-specific complication flags
        wilms      = gene in ("NSD1", "GPC3", "CDKN1C") and rng.random() < (
            0.03 if gene == "NSD1" else 0.10 if gene == "GPC3" else 0.08)
        cardiac    = gene in ("GPC3", "CDKN1C", "NSD1") and rng.random() < (
            0.40 if gene == "GPC3" else 0.20 if gene == "CDKN1C" else 0.18)
        aml_risk   = gene == "EZH2" and rng.random() < 0.08
        autism_ftr = gene in ("PTEN", "SETD2", "NSD1") and rng.random() < (
            0.20 if gene == "PTEN" else 0.50 if gene == "SETD2" else 0.12)
        advanced_ba = gene in ("NSD1", "EZH2", "NFIX", "GPC3", "CDKN1C") and rng.random() < 0.75
        raised_icp  = rng.random() < (0.20 if severity == "severe" else 0.05)
        treatment   = rng.choice(treatments)
        mutation    = rng.choice(mutations)
        features    = rng.choice(feature_options)
        patients.append({
            "id":                  f"{gene}-{seed}-{i+1:03d}",
            "gene":                gene,
            "age_at_diagnosis_mo": age_dx * 12 if age_dx < 5 else age_dx,
            "severity":            severity,
            "iq_estimate":         iq,
            "associated_features": features,
            "wilms_tumour":        wilms,
            "cardiac_defect":      cardiac,
            "aml_risk":            aml_risk,
            "autism_feature":      autism_ftr,
            "advanced_bone_age":   advanced_ba,
            "raised_icp":          raised_icp,
            "treatment":           treatment,
            "mutation":            mutation,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Overgrowth-Syndromes-Atlas."""
    return {
        "atlas":          "Hereditary-Overgrowth-Syndromes-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Overgrowth Syndromes Atlas "
            "(NSD1-EZH2-GPC3-CDKN1C-PTEN-SETD2-NFIX-PIK3CA)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "NSD1":   "AD LOF de novo (Sotos syndrome; cerebral gigantism; most common overgrowth ~1:14,000)",
            "EZH2":   "AD LOF de novo (Weaver syndrome; accelerated osseous maturation PATHOGNOMONIC; AML/ALL risk)",
            "GPC3":   "X-linked recessive LOF (SGBS1; supernumerary nipples PATHOGNOMONIC; Wilms 10%; males severely affected)",
            "CDKN1C": "Maternal imprinting LOF (BWS type 3; omphalocele+macroglossia+macrosomia triad; Wilms 7-9%)",
            "PTEN":   "AD LOF (PHTS/BRR; macrocephaly >2SD = testing trigger; ASD 20%; breast 85%; thyroid 35%)",
            "SETD2":  "AD LOF (Luscan-Lumish syndrome; Sotos-like + autism 50%; H3K36me3 sole enzyme)",
            "NFIX":   "AD LOF=Malan/Sotos-2 (tall+macrocephaly+ID) / GOF=Marshall-Smith (severe+accelerated BA+respiratory)",
            "PIK3CA": "Somatic GOF mosaic (PROS: MCAP/CLOVES/hemimegalencephaly; alpelisib FDA-approved; deep sequencing mandatory)",
        },
        "key_clinical_rules": [
            "SOTOS (NSD1): MOST COMMON overgrowth ~1:14,000; TRIAD: macrocephaly+tall+ID (variable); MLPA mandatory (5-15% large deletions)",
            "WEAVER (EZH2): ACCELERATED OSSEOUS MATURATION PATHOGNOMONIC — wrist X-ray at diagnosis; AML/ALL surveillance annual FBC",
            "SGBS1 (GPC3): SUPERNUMERARY NIPPLES PATHOGNOMONIC — no other overgrowth syndrome; Wilms 10% → renal USS every 3 months to age 8",
            "BWS (CDKN1C): METHYLATION TESTING FIRST (11p15.5); CDKN1C = classical BWS + omphalocele; MATERNAL mutation only causes disease",
            "PTEN: MACROCEPHALY >2SD → PTEN TESTING TRIGGER; breast cancer 85% lifetime → annual MRI from age 30-35",
            "PTEN + ASD: macrocephalic ASD patients → PTEN has highest single-gene yield (20% positive rate)",
            "SETD2 (Luscan-Lumish): SOTOS-LIKE + AUTISM 50%; differentiate from NSD1/EZH2 by molecular testing",
            "NFIX: LOF = Malan (mild-moderate Sotos-like); GOF = Marshall-Smith (SEVERE + respiratory failure); same gene opposite phenotypes",
            "PIK3CA PROS: SOMATIC MOSAIC → deep sequencing >500× mandatory; standard NGS MISSES; alpelisib (Vijoice™) FDA-approved 2022",
            "OVERGROWTH DDx: NSD1 (long face+downslant) vs EZH2 (round face+hypertelorism) vs NFIX vs SETD2 — molecular testing required",
            "SURVEILLANCE: all Wilms-risk genes (NSD1/GPC3/CDKN1C) → abdominal USS + AFP every 3 months to age 8",
            "ART/IVF: increased BWS risk — epigenetic mechanism; counsel families undergoing assisted reproduction",
        ],
        "gene_panel_note": (
            "Comprehensive overgrowth gene panel (2024): NSD1, EZH2, GPC3, CDKN1C, PTEN, SETD2, NFIX, PIK3CA, "
            "AKT1 (mosaic Proteus), AKT3, MTOR, H3F3A/B, NF1 (overgrowth+tumour), BRCA1/2 (Cowden DDx), "
            "methylation analysis (11p15.5 for BWS), CNV/MLPA (NSD1 5q35 deletion)"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Overgrowth-Syndromes-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)
        severe_n   = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n     = sum(1 for p in patients if p["severity"] == "mild")
        mean_iq    = round(sum(p["iq_estimate"] for p in patients) / n, 1)
        wilms_n    = sum(1 for p in patients if p["wilms_tumour"])
        cardiac_n  = sum(1 for p in patients if p["cardiac_defect"])
        aml_n      = sum(1 for p in patients if p["aml_risk"])
        autism_n   = sum(1 for p in patients if p["autism_feature"])
        adv_ba_n   = sum(1 for p in patients if p["advanced_bone_age"])
        raised_icp_n = sum(1 for p in patients if p["raised_icp"])
        mean_age   = round(sum(p["age_at_diagnosis_mo"] for p in patients) / n, 1)
        mutations_seen = list({p["mutation"] for p in patients})
        genes_data.append({
            "gene":             gene,
            "locus":            gene_info["locus"],
            "n_patients":       n,
            "severe_pct":       round(severe_n / n * 100, 1),
            "moderate_pct":     round(moderate_n / n * 100, 1),
            "mild_pct":         round(mild_n / n * 100, 1),
            "mean_iq":          mean_iq,
            "wilms_pct":        round(wilms_n / n * 100, 1),
            "cardiac_pct":      round(cardiac_n / n * 100, 1),
            "aml_pct":          round(aml_n / n * 100, 1),
            "autism_pct":       round(autism_n / n * 100, 1),
            "adv_bone_age_pct": round(adv_ba_n / n * 100, 1),
            "raised_icp_pct":   round(raised_icp_n / n * 100, 1),
            "mean_age_dx_mo":   mean_age,
            "sample_mutations": mutations_seen[:4],
            "protein":          gene_info["protein"],
            "inheritance":      gene_info["inheritance"][:200],
            "disease_category": gene_info["disease_category"],
        })
    return {
        "atlas": "Hereditary-Overgrowth-Syndromes-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Overgrowth-Syndromes-Atlas."""
    definitions = [
        {
            "term": "Overgrowth Syndromes — Classification and Diagnostic Approach",
            "genes": ["NSD1", "EZH2", "GPC3", "CDKN1C", "PTEN", "SETD2", "NFIX", "PIK3CA"],
            "definition": (
                "HEREDITARY OVERGROWTH SYNDROMES — OVERVIEW: "
                "DEFINITION: overgrowth = growth > 2 SD above mean for age/sex in one or more parameters "
                "  (height, OFC, weight, organ size); "
                "PREVALENCE: "
                "  Sotos (NSD1): ~1:14,000 — most common; "
                "  BWS (11p15 including CDKN1C): ~1:10,000-13,000; "
                "  PTEN PHTS: prevalence uncertain (~1:200,000 Cowden clinical); "
                "  Others rarer; "
                "CLASSIFICATION BY MECHANISM: "
                "  CHROMATIN/EPIGENETIC: NSD1 (H3K36me1/2), EZH2 (H3K27me3), SETD2 (H3K36me3); "
                "  GROWTH FACTOR SIGNALLING: PIK3CA (PI3K/AKT/mTOR), PTEN (negative regulator of same); "
                "  HEPARAN SULPHATE/WNT: GPC3 (glypican 3, Wnt/HH); "
                "  CELL CYCLE CONTROL: CDKN1C (p57KIP2, CDK inhibitor); "
                "  TRANSCRIPTION FACTOR: NFIX (NFI family); "
                "DIAGNOSTIC APPROACH: "
                "  Step 1: quantify — OFC centile, height centile, weight; "
                "  Step 2: identify syndromic features (see gene-specific rules); "
                "  Step 3: order appropriate molecular test: "
                "    MACROCEPHALY ONLY: PTEN first (highest cancer risk); "
                "    MACROCEPHALY + TALL + ID: NSD1 → EZH2 → SETD2 → NFIX panel; "
                "    MACROCEPHALY + ASD: PTEN first; "
                "    NEONATAL MACROSOMIA + OMPHALOCELE: 11p15.5 methylation → CDKN1C; "
                "    SEGMENTAL/ASYMMETRIC overgrowth: PIK3CA mosaic (deep sequencing); "
                "    SUPERNUMERARY NIPPLES + male: GPC3 (SGBS1); "
                "  Step 4: METHYLATION TESTING: mandatory in all BWS-phenotype (GPC3/CDKN1C DDx); "
                "  Step 5: TUMOUR SURVEILLANCE based on gene; "
                "TUMOUR RISKS SUMMARY: "
                "  NSD1: Wilms 2-3%; sacrococcygeal teratoma; "
                "  EZH2: AML/ALL; "
                "  GPC3: Wilms 10%; "
                "  CDKN1C: Wilms 7-9%; hepatoblastoma 1%; "
                "  PTEN: breast 85%, thyroid 35%, endometrial 28%, RCC 33%; "
                "  SETD2: renal CCC (somatic); surveillance reasonable"
            ),
        },
        {
            "term": "Sotos vs Weaver vs Luscan-Lumish vs Malan — Differential Diagnosis",
            "genes": ["NSD1", "EZH2", "SETD2", "NFIX"],
            "definition": (
                "SOTOS SYNDROME SPECTRUM — MOLECULAR DIFFERENTIAL: "
                "SOTOS SYNDROME (NSD1): "
                "  FACE: LONG face; HIGH/broad forehead; DOWNSLANTING palpebral fissures; LONG CHIN; "
                "  MACROCEPHALY: OFC > 2 SD; sparse frontoparietal hair; "
                "  BEHAVIOUR: expressive language worse than receptive; ASD features 10%; "
                "  BONE AGE: advanced 1-2 years; "
                "  TUMOUR: Wilms 2-3%; AFP surveillance; "
                "  PREVALENCE: ~1:14,000 MOST COMMON; "
                "WEAVER SYNDROME (EZH2): "
                "  FACE: ROUND face; ocular HYPERTELORISM; camptodactyly; large ears; "
                "  KEY DDx: ROUND vs LONG face distinguishes Weaver from Sotos; "
                "  BONE AGE: MARKEDLY advanced 2-3 years (more than Sotos); "
                "  AML/ALL: predisposition — annual FBC; "
                "  SAME gene (EZH2): GOF in lymphoma, LOF in Weaver; "
                "LUSCAN-LUMISH SYNDROME (SETD2): "
                "  FACE: broad flat nasal bridge; deep-set eyes; "
                "  AUTISM: PROMINENT ~50% (higher than Sotos/Weaver); "
                "  BEHAVIOUR: aggression; hyperactivity; "
                "  MOLECULAR: SETD2 is H3K36me3 enzyme (final step after NSD1); "
                "MALAN SYNDROME / SOTOS-2 (NFIX LOF): "
                "  FACE: milder Sotos-like; pointed chin; long face; "
                "  PHENOTYPE: milder than classic Sotos; anxiety prominent; "
                "  NFIX GOF = MARSHALL-SMITH: SEVERE + accelerated BA + respiratory failure; "
                "SUMMARY TABLE: "
                "  FACE: Weaver (round+hypertelorism) ≠ Sotos/Luscan (long) ≠ Malan (mild-long); "
                "  BONE AGE: Weaver > Sotos ≈ NFIX-MS > Luscan; "
                "  AML: EZH2 only; "
                "  AUTISM: SETD2 > PTEN > NSD1; "
                "RECOMMENDATION: if clinically Sotos but NSD1 negative: panel NSD1→EZH2→SETD2→NFIX"
            ),
        },
        {
            "term": "Beckwith-Wiedemann Syndrome — Molecular Subtypes and Tumour Surveillance",
            "genes": ["CDKN1C"],
            "definition": (
                "BECKWITH-WIEDEMANN SYNDROME — MOLECULAR CLASSIFICATION: "
                "GENETIC HETEROGENEITY: BWS is a CLINICAL DIAGNOSIS with multiple molecular causes; "
                "  All mechanisms affect 11p15.5 imprinting region: "
                "  IC1 (H19/IGF2 DMR): "
                "    H19/IGF2 methylation GAIN (IC1 gain): ~7% sporadic; highest Wilms risk ~24%; "
                "    PATERNAL UPD of 11p15 (~20%): IC1 gain + IC2 loss simultaneously; "
                "  IC2 (KvDMR1 LIT1): "
                "    IC2 methylation LOSS (~50% sporadic): commonest cause; Wilms 5%; "
                "    CDKN1C mutation (~10% sporadic; ~50% familial BWS): classical severe phenotype; "
                "CDKN1C-BWS (TYPE 3) — MOST SEVERE: "
                "  CLASSICAL TRIAD: OMPHALOCELE + MACROGLOSSIA + MACROSOMIA; "
                "  IMPRINTING RULE: mutation must be on MATERNAL allele (maternally expressed gene); "
                "    PATERNAL CDKN1C mutation → NO BWS (paternal allele is silenced); "
                "  OMPHALOCELE: present in ~60% of CDKN1C-BWS (vs ~20% H19 gain); "
                "  NEONATAL HYPOGLYCAEMIA: hyperinsulinism; IV glucose; octreotide if persistent; "
                "  HEMIHYPERTROPHY: limb/body asymmetry; "
                "  WILMS TUMOUR: 7-9%; "
                "  HEPATOBLASTOMA: ~1%; "
                "TESTING ALGORITHM: "
                "  FIRST: 11p15.5 methylation analysis (CpG microarray or MS-MLPA); "
                "  SECOND: CDKN1C sequencing (if family history or methylation abnormal); "
                "  UPD analysis if methylation pattern complex; "
                "SURVEILLANCE PROTOCOL: "
                "  ABDOMINAL USS: every 3 months from birth to age 8 (Wilms + hepatoblastoma); "
                "  AFP: every 3 months to age 4 (hepatoblastoma); "
                "  Blood glucose: neonatal intensive monitoring; "
                "  Growth: hemihypertrophy documentation; "
                "  MACROGLOSSIA: SALT referral + orthodontics; tongue reduction if severe"
            ),
        },
        {
            "term": "PTEN Hamartoma Tumour Syndrome — Macrocephaly, ASD, and Cancer Surveillance",
            "genes": ["PTEN"],
            "definition": (
                "PTEN HAMARTOMA TUMOUR SYNDROME (PHTS) — COMPLETE GUIDE: "
                "TESTING TRIGGER: "
                "  MACROCEPHALY > 2 SD: PTEN testing recommended; "
                "  MACROCEPHALY > 2.5 SD: PTEN testing strongly indicated; "
                "  MACROCEPHALY + ASD: PTEN yield 17-20% (highest single-gene in this combination); "
                "  Multiple hamartomas: skin, thyroid, GI polyps; "
                "  Family history Cowden/BRR; "
                "CLINICAL SUBTYPES: "
                "  COWDEN SYNDROME (CS): "
                "    Trichilemmomas (skin around nose/ears); oral papillomas; acral keratoses; "
                "    Adult onset; cancer predominant presentation; "
                "  BANNAYAN-RILEY-RUVALCABA (BRR): "
                "    Childhood: macrocephaly + lipomatosis + PENILE MACULES; "
                "    PENILE MACULES = pigmented freckling penile shaft (PATHOGNOMONIC for BRR); "
                "    Vascular malformations (AVMs); intestinal polyps; "
                "  ADULT LHERMITTE-DUCLOS: dysplastic cerebellar gangliocytoma; MRI; "
                "CANCER RISKS AND SURVEILLANCE: "
                "  BREAST CANCER: 85% lifetime → annual breast MRI from 30-35 years; "
                "  THYROID CANCER (follicular/papillary): 35% → annual thyroid USS from diagnosis; "
                "  ENDOMETRIAL CANCER: 28% → annual USS/biopsy from 30-35; "
                "  COLORECTAL: 9% → colonoscopy from 35, every 5 years; "
                "  RENAL CELL CARCINOMA: 33% → renal imaging every 1-2 years from 40; "
                "  MELANOMA: elevated; annual dermatology; "
                "  BREAST: risk-reducing mastectomy option (discuss >age 25 with significant density); "
                "PI3K/AKT/mTOR PATHWAY: "
                "  PTEN dephosphorylates PIP3 → stops AKT → stops mTOR; "
                "  PTEN LOF = constitutive mTOR → growth + cancer; "
                "  MTOR INHIBITORS: everolimus used off-label for severe PTEN PHTS manifestations; "
                "PAEDIATRIC SURVEILLANCE: "
                "  Annual thyroid USS from diagnosis; "
                "  Macrocephaly + developmental assessment; "
                "  Neuroimaging if Lhermitte-Duclos suspected (progressive cerebellar signs)"
            ),
        },
        {
            "term": "PIK3CA PROS — Mosaic Testing, MCAP/CLOVES, and Alpelisib Treatment",
            "genes": ["PIK3CA"],
            "definition": (
                "PIK3CA-RELATED OVERGROWTH SPECTRUM (PROS) — COMPLETE GUIDE: "
                "MECHANISM: "
                "  PIK3CA somatic GAIN-OF-FUNCTION mosaic mutation; "
                "  Post-zygotic timing → mosaic distribution → TISSUE-SPECIFIC overgrowth; "
                "  PIK3CA → PIP3 → AKT → mTOR ACTIVATION → localised growth; "
                "  Mutation burden/timing determines phenotype spectrum; "
                "PROS PHENOTYPES: "
                "  MCAP (Megalencephaly-Capillary Malformation-Polymicrogyria): "
                "    Bilateral MEGALENCEPHALY; cutaneous capillary malformations (port-wine stain); "
                "    POLYMICROGYRIA (brain cortex); hydrocephalus; SEIZURES; "
                "    Somatic growth acceleration; toe/finger overgrowth; "
                "  CLOVES SYNDROME: "
                "    Congenital Lipomatous Overgrowth; Vascular malformations; Epidermal naevi; "
                "    Scoliosis/skeletal anomalies/Spinal anomalies; "
                "    Large retroperitoneal fatty masses at birth; "
                "  HEMIMEGALENCEPHALY (HME): "
                "    Unilateral brain hemisphere overgrowth; INTRACTABLE SEIZURES; "
                "    Hemispherectomy may be required; "
                "  FIBROADIPOSE OVERGROWTH (FAO): "
                "    Segmental/limb fatty and soft tissue overgrowth; "
                "  OTHER: isolated cerebellar overgrowth; FACEOM; "
                "DIAGNOSIS — DEEP SEQUENCING MANDATORY: "
                "  Standard NGS (20-30× depth): MISSES low-allele-frequency mosaic; "
                "  DEEP SEQUENCING (500-1000×): needed; "
                "  DROPLET DIGITAL PCR (ddPCR): highest sensitivity; "
                "  TISSUE SELECTION: biopsied affected tissue > peripheral blood; "
                "  COMMON MUTATIONS: p.His1047Arg (most), p.His1047Leu, p.Glu545Lys; "
                "ALPELISIB (VIJOICE™) TREATMENT: "
                "  FDA-APPROVED 2022 for PROS in patients ≥2 years; "
                "  Mechanism: selective PIK3CA/p110α inhibitor; "
                "  RESPONSE: reduction in vascular malformations; improvement in overgrowth; "
                "  MONITORING: blood glucose (hyperglycaemia risk); skin rash; stomatitis; LFTs; "
                "  DOSING: weight-based; specialist centre; "
                "  Previously: sirolimus (mTOR inhibitor) used off-label; "
                "MANAGEMENT: "
                "  Vascular malformations: sclerotherapy + surgery; "
                "  Seizures: AEDs + epilepsy surgery (hemispherectomy if HME); "
                "  Limb discrepancy: orthopaedic epiphysiodesis; "
                "  Retroperitoneal lipomas (CLOVES): surgical debulking"
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Overgrowth-Syndromes-Atlas",
        "count":       len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:800])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    for g in bd["genes"]:
        print(f"  {g['gene']}: n={g['n_patients']}, severe={g['severe_pct']}%, "
              f"iq={g['mean_iq']}, wilms={g['wilms_pct']}%, cardiac={g['cardiac_pct']}%")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Definition entries: {df['count']}")
    for d in df["definitions"]:
        print(f"  {d['term'][:60]}")
