#!/usr/bin/env python3
"""Hereditary-Chromatinopathy-Epigenetic-Atlas — Complete 8-Gene Chromatinopathy & Epigenetic Syndrome Atlas
KMT2D  (histone H3K4 methyltransferase; 5768 aa; 19p13.11; AD de novo LOF;
         Kabuki syndrome 1 (KS1); MOST COMMON — 60-75% all Kabuki;
         ARCHED EYEBROWS + PERSISTENT FETAL FINGERTIP PADS PATHOGNOMONIC;
         cardiac CHD 31-50%; short stature; intellectual disability moderate;
         seed SEED_BASE+0) ·
KDM6A  (lysine demethylase 6A; 1401 aa; Xp11.3; X-linked LOF;
         Kabuki syndrome 2 (KS2); MILDER IN HETEROZYGOUS FEMALES (X-inactivation);
         HEMIZYGOUS MALES MORE SEVERELY AFFECTED;
         5% of all Kabuki cases; seed SEED_BASE+1) ·
CREBBP (CREB binding protein; 2442 aa; 16p13.3; AD de novo LOF;
         Rubinstein-Taybi syndrome 1 (RTS1); MOST COMMON RTS — 50-70%;
         BROAD THUMBS AND BROAD HALLUCES PATHOGNOMONIC;
         malignancy risk 10-15% (leukaemia, brain tumours);
         seed SEED_BASE+2) ·
EP300  (E1A binding protein p300; 2161 aa; 22q13.2; AD de novo LOF;
         Rubinstein-Taybi syndrome 2 (RTS2); MILDER than CREBBP;
         less malignancy; similar broad thumbs less severe;
         seed SEED_BASE+3) ·
ARID1B (AT-rich interaction domain 1B; 2285 aa; 6q25.3; AD de novo LOF;
         Coffin-Siris syndrome 1 (CSS1); MOST COMMON CSS gene ~50%;
         ABSENT OR HYPOPLASTIC 5th FINGERNAIL/TOENAIL PATHOGNOMONIC;
         coarse facies; mild-moderate ID; seed SEED_BASE+4) ·
EHMT1  (euchromatic histone lysine methyltransferase 1; 1210 aa; 9q34.3; AD de novo LOF;
         Kleefstra syndrome; 9q34.3 microdeletion (most common) or EHMT1 intragenic;
         HYPOTONIA + BRACHYCEPHALY + COARSE FACIES triad; autism 30%;
         cardiac CHD 35%; friendly behaviour; seed SEED_BASE+5) ·
KAT6B  (lysine acetyltransferase 6B; 2073 aa; 10q22.2; AD de novo LOF;
         Say-Barber-Biesecker-Young-Simpson (SBBYS) / Genitopatellar syndrome;
         ABSENT OR HYPOPLASTIC PATELLA PATHOGNOMONIC;
         agenesis corpus callosum 80%; genital anomalies;
         seed SEED_BASE+6) ·
KANSL1 (KAT8 regulatory NSL complex subunit 1; 1119 aa; 17q21.31; AD de novo LOF;
         Koolen-de Vries syndrome (KdVS);
         FRIENDLY/SOCIABLE BEHAVIOUR PATHOGNOMONIC (affects almost all);
         17q21.31 microdeletion ~75% of cases; epilepsy 50%; CHD 30%;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 3046-3053)
"""
import random

SEED_BASE = 3046

ATLAS_GENES = [
    {
        "gene": "KMT2D",
        "protein": (
            "KMT2D -- 19p13.11 AD-de-novo-LOF -- 5768aa -- Lysine-Methyltransferase-2D-"
            "H3K4me1-me2-Enhancer-Activator-Kabuki-Syndrome-1-ARCHED-EYEBROWS-PERSISTENT-"
            "FETAL-FINGERTIP-PADS-PATHOGNOMONIC-OMIM-602113"
        ),
        "locus": "19p13.11",
        "protein_size": (
            "5768 aa / 593 kDa (KMT2D; lysine methyltransferase 2D; "
            "histone H3K4 mono- and di-methyltransferase at active enhancers; "
            "STRUCTURE: PHD domains x3 + FYRN + FYRC + SET domain (catalytic) + HMG box; "
            "FUNCTION: "
            "  KMT2D deposits H3K4me1/me2 at ENHANCERS → gene activation; "
            "  KMT2D is a component of the COMPASS-like complex (with WDR5, RBBP5, ASH2L); "
            "  KMT2D is the MAJOR H3K4 monomethyltransferase at distal enhancers; "
            "  KMT2D loss → reduced enhancer activity → impaired tissue-specific gene expression; "
            "  KMT2D regulates adipogenesis, neural differentiation, cardiac morphogenesis; "
            "KABUKI SYNDROME 1 (KS1) — CLINICAL: "
            "  PREVALENCE: ~1:32,000; MOST COMMON cause of Kabuki = 60-75% of all cases; "
            "  FACIAL FEATURES (PATHOGNOMONIC combination): "
            "    LONG PALPEBRAL FISSURES; "
            "    ARCHED EYEBROWS with lateral third sparse/absent (PATHOGNOMONIC); "
            "    BROAD NASAL TIP; "
            "    LARGE PROMINENT EARS; "
            "    PERSISTENT FETAL FINGERTIP PADS (PATHOGNOMONIC — highly specific); "
            "  INTELLECTUAL DISABILITY: "
            "    Mild-moderate ID universal; IQ typically 50-75 range; "
            "    Wide range; some in mild range (IQ 70-85); "
            "  SHORT STATURE: postnatal growth retardation; GH therapy may help; "
            "  SKELETAL: joint hypermobility 90%; brachydactyly; 5th finger clinodactyly; "
            "  CARDIAC CHD: 31-50%; VSD, ASD, coarctation, AVSD; "
            "  FEEDING: neonatal feeding difficulties 70-80%; NG tube infancy common; "
            "  IMMUNE: hypogammaglobulinaemia 50%; recurrent otitis media; IgA deficiency; "
            "  RENAL: structural anomalies 30%; horseshoe kidney; duplex collecting system; "
            "  DENTAL: hypodontia; delayed eruption; enamel hypoplasia; "
            "  VISION: ptosis; strabismus; "
            "  HEARING: sensorineural/conductive loss 40%; "
            "  BEHAVIOURAL: friendly, social; anxiety; autistic traits 30%; "
            "  GENOTYPE-PHENOTYPE: "
            "    Truncating/LOF variants (frameshift, nonsense, splice) = most cases; "
            "    Missense variants in SET domain = more severe; "
            "    C-terminal SET domain truncations (STOP codon SET or C-terminal) = typical; "
            "DIAGNOSIS: "
            "  CLINICAL CRITERIA (Niikawa/van der Burgt criteria): ≥3 of 5 cardinal features; "
            "  MOLECULAR: WES/WGS + exon array (intragenic deletion); "
            "  CMA: intragenic deletion/duplication in minority; "
            "GENE: KMT2D; 19p13.11; OMIM gene 602113; KS1 disease OMIM #147920"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — KMT2D / KABUKI SYNDROME 1: "
            "  DE NOVO: >95% of cases; "
            "  FAMILIAL: rare; variable expressivity within families; "
            "  MOSAIC: parental somatic/germline mosaicism in 1-2%; "
            "TESTING ALGORITHM: "
            "  STEP 1: Clinical scoring (van der Burgt Kabuki score); "
            "  STEP 2: KMT2D sequencing (frameshift, nonsense, splice, missense SET domain); "
            "  STEP 3: KMT2D exon array / MLPA (intragenic deletion/duplication); "
            "  STEP 4: If negative → KDM6A (X-linked KS2); "
            "  WES/WGS panel including both KMT2D + KDM6A recommended first-line; "
            "MANAGEMENT: "
            "  Cardiac echo at diagnosis + surveillance; "
            "  GH therapy: consider for short stature (GH deficiency NOT universal); "
            "  Immunology: IgG levels; IVIG if recurrent infections; "
            "  Audiology: hearing aids if SNHL/conductive HL; "
            "  Dental: orthodontist early; "
            "  Renal USS: structural anomalies; "
            "  Ophthalmology: ptosis correction; strabismus; "
            "  Developmental support: ASD intervention if ASD criteria"
        ),
        "disease_category": (
            "KMT2D-KABUKI-ARCHED-EYEBROWS-FINGERTIP-PADS: "
            "  ARCHED EYEBROWS (lateral third sparse) + PERSISTENT FETAL FINGERTIP PADS PATHOGNOMONIC; "
            "  MOST COMMON KABUKI cause — 60-75% of all Kabuki syndrome; "
            "  CARDIAC CHD 31-50% — echo at diagnosis MANDATORY; "
            "  IMMUNE: IgA/IgG deficiency — immunological workup at diagnosis"
        ),
    },
    {
        "gene": "KDM6A",
        "protein": (
            "KDM6A -- Xp11.3 X-linked-LOF -- 1401aa -- Lysine-Demethylase-6A-"
            "H3K27me3-me2-Demethylase-Kabuki-Syndrome-2-MILDER-FEMALES-MORE-SEVERE-MALES-"
            "OMIM-300128"
        ),
        "locus": "Xp11.3",
        "protein_size": (
            "1401 aa / 154 kDa (KDM6A; lysine demethylase 6A; UTX; "
            "H3K27me2/me3 histone demethylase (Jumonji C domain); "
            "STRUCTURE: tetratricopeptide repeats (TPRs) + JmjC catalytic domain; "
            "FUNCTION: "
            "  KDM6A demethylates H3K27me2/me3 → H3K27me1/me0 → gene ACTIVATION; "
            "  KDM6A antagonises PRC2 complex (EZH2 H3K27me3 writer) → switch from silenced to active; "
            "  KDM6A is part of the KMT2D COMPASS-like complex as a non-enzymatic scaffold; "
            "  KDM6A regulates cardiac morphogenesis, neural differentiation, X-chromosome dosage; "
            "  ESCAPE from X-inactivation: KDM6A partially escapes Xi inactivation in females → "
            "    female heterozygotes have SOME functional KDM6A from both X chromosomes; "
            "    male hemizygotes lose ALL KDM6A function → MORE SEVERELY AFFECTED; "
            "KABUKI SYNDROME 2 (KS2) — CLINICAL: "
            "  PREVALENCE: ~5% of all Kabuki cases; "
            "  FEMALE HETEROZYGOTES (MILDER): "
            "    Mild intellectual disability; facial features LESS pronounced; "
            "    Short stature; cardiac CHD 30%; feeding difficulties; "
            "    SIMILAR to KS1 but systematically MILDER phenotype; "
            "    X-inactivation skewing can modify severity; "
            "  MALE HEMIZYGOTES (MORE SEVERE): "
            "    Moderate-severe ID; "
            "    More severe facial features; "
            "    More severe cardiac involvement; "
            "    MULTIPLE ANOMALIES — more organ systems affected; "
            "    Rare — most XL conditions have male excess in severity; "
            "  SHARED WITH KS1: "
            "    Arched eyebrows; persistent fingertip pads; broad nasal tip; large ears; "
            "    Short stature; joint hypermobility; feeding difficulties; cardiac CHD; "
            "  INTELLECTUAL DISABILITY: "
            "    Females: mild (IQ 60-80); "
            "    Males: moderate-severe (IQ 40-60); "
            "  CARDIAC: CHD in 30-40%; similar spectrum to KS1; "
            "  IMMUNE: similar immune dysfunction to KS1; "
            "  CANCER: KDM6A somatic mutations common in bladder cancer, myeloid malignancies; "
            "    Germline: surveillance for haematological malignancy uncertain but advised; "
            "DIAGNOSIS: "
            "  After KMT2D negative: KDM6A sequencing + copy number; "
            "  Females: heterozygous variants (missense, truncating); "
            "  Males: hemizygous truncating/missense; "
            "  MLPA/array for deletions including KDM6A; "
            "GENE: KDM6A; Xp11.3; OMIM gene 300128; KS2 disease OMIM #300867"
        ),
        "inheritance": (
            "X-LINKED LOF — KDM6A / KABUKI SYNDROME 2: "
            "  FEMALES: heterozygous LOF — MILDER (partial X-inactivation escape); "
            "  MALES: hemizygous LOF — MORE SEVERE; "
            "  DE NOVO: majority in both sexes; "
            "  FAMILIAL: X-linked; carrier mothers; obligate carrier testing; "
            "TESTING: "
            "  Sequence KDM6A if KMT2D negative; "
            "  Include in gene panel for Kabuki phenotype; "
            "  Males: hemizygous status confirms; females: heterozygous; "
            "MANAGEMENT: same as KS1 with attention to X-linked severity differences in males"
        ),
        "disease_category": (
            "KDM6A-KABUKI2-XLINKED-SEX-SPECIFIC-SEVERITY: "
            "  HEMIZYGOUS MALES MORE SEVERE than heterozygous females — X-inactivation escape; "
            "  5% of Kabuki syndrome — test after KMT2D negative; "
            "  SAME PHENOTYPE as KS1 but systematically MILDER in females; "
            "  KDM6A somatic mutations: bladder cancer + myeloid malignancies — biological link"
        ),
    },
    {
        "gene": "CREBBP",
        "protein": (
            "CREBBP -- 16p13.3 AD-de-novo-LOF -- 2442aa -- CREB-Binding-Protein-HAT-"
            "CBP-Rubinstein-Taybi-Syndrome-1-BROAD-THUMBS-BROAD-HALLUCES-PATHOGNOMONIC-"
            "MALIGNANCY-10-15pct-OMIM-600140"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "2442 aa / 265 kDa (CREBBP; CREB binding protein; CBP; "
            "histone acetyltransferase (HAT; KAT3A); "
            "STRUCTURE: KIX domain + TAZ1/2 + CH1/CH3 + RID + HAT domain + bromo domain; "
            "FUNCTION: "
            "  CREBBP acetylates histones H3K18, H3K27 → gene ACTIVATION; "
            "  CREBBP is a MASTER transcriptional co-activator; "
            "  CREBBP binds phospho-CREB, STAT proteins, NF-κB, p53, Rb → integrates signals; "
            "  CREBBP is a TUMOUR SUPPRESSOR — somatic loss in haematological malignancies; "
            "  CREBBP LOF → impaired H3K18/K27 acetylation → reduced enhancer activity; "
            "RUBINSTEIN-TAYBI SYNDROME 1 (RTS1) — CLINICAL: "
            "  PREVALENCE: ~1:125,000; MOST COMMON RTS cause — 50-70% of all RTS; "
            "  CARDINAL FEATURES (PATHOGNOMONIC): "
            "    BROAD THUMBS AND BROAD GREAT TOES (HALLUCES): "
            "      PATHOGNOMONIC — wide distal phalanx, angulated thumb tip; "
            "      Spatula-shaped broad flat thumb; "
            "      Broad great toes; sometimes duplicated/bifid distal phalanx; "
            "    BROAD-BASED NOSE WITH BEAKING: "
            "      Columella below ala nasi; "
            "    GRIMACING SMILE: characteristic facial expression; "
            "    DOWNWARD-SLANTING PALPEBRAL FISSURES; "
            "    HIGHLY ARCHED PALATE; "
            "  INTELLECTUAL DISABILITY: "
            "    Moderate ID typical (IQ 35-65); "
            "    Severe ID: ~25%; "
            "    Mild: ~10-15%; "
            "    Language delayed; expressive > receptive impairment; "
            "  MALIGNANCY: "
            "    10-15% lifetime risk — KEY SURVEILLANCE POINT; "
            "    Leukaemia (ALL, AML), brain tumours (meningioma, medulloblastoma), "
            "    Neuroblastoma; pilomatrixoma (benign skin tumour — common); "
            "    Annual surveillance: FBC, clinical; "
            "  BEHAVIOURAL: "
            "    Friendly, social, engaging CHARACTERISTIC; "
            "    Repetitive behaviours; OCD features; anxiety; "
            "    Aggression: some patients, situational; "
            "  GROWTH: short stature; adult height typically ≤150 cm males; "
            "  CARDIAC: CHD 24-38% (ASD, PDA, VSD, aortic anomalies); "
            "  RENAL: anomalies 27%; "
            "  OPHTHALMOLOGICAL: ptosis; strabismus; cataracts; coloboma; "
            "  DERMATOLOGICAL: keloid formation; pilomatrixoma; "
            "  ENDOCRINE: obesity tendency adolescence; "
            "DIAGNOSIS: "
            "  Clinical criteria (Hennekam): broad thumbs + intellectual disability + facial; "
            "  MOLECULAR: CREBBP sequencing (LOF most common); "
            "  MLPA: deletion/duplication (10-15% of cases); "
            "  WES/WGS recommended; EP300 after CREBBP negative; "
            "GENE: CREBBP; 16p13.3; OMIM gene 600140; RTS1 disease OMIM #180849"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — CREBBP / RUBINSTEIN-TAYBI 1: "
            "  DE NOVO: >95%; "
            "  FAMILIAL: rare; variable expressivity; "
            "  DELETION: MLPA/CMA — 10-15% large deletions/translocations; "
            "TESTING: "
            "  STEP 1: CREBBP sequencing (frameshift, nonsense, splice, missense HAT domain); "
            "  STEP 2: CREBBP MLPA/array (deletions/duplications); "
            "  STEP 3: If negative → EP300 (RTS2); "
            "MALIGNANCY SURVEILLANCE: annual FBC; clinical review; "
            "  Pilomatrixoma: benign; excision if large/symptomatic"
        ),
        "disease_category": (
            "CREBBP-RTS1-BROAD-THUMBS-HALLUCES-MALIGNANCY: "
            "  BROAD THUMBS + BROAD GREAT TOES PATHOGNOMONIC — check digit width at diagnosis; "
            "  MALIGNANCY 10-15% — annual FBC + clinical surveillance MANDATORY; "
            "  MOST COMMON RTS cause (50-70%); CREBBP before EP300 in testing; "
            "  PILOMATRIXOMA (benign skin calcified tumour) — frequent, NOT malignant"
        ),
    },
    {
        "gene": "EP300",
        "protein": (
            "EP300 -- 22q13.2 AD-de-novo-LOF -- 2161aa -- E1A-Binding-Protein-p300-HAT-"
            "KAT3B-Rubinstein-Taybi-Syndrome-2-MILDER-than-CREBBP-SIMILAR-BROAD-THUMBS-"
            "OMIM-602700"
        ),
        "locus": "22q13.2",
        "protein_size": (
            "2161 aa / 265 kDa (EP300; E1A-binding protein p300; KAT3B; "
            "histone acetyltransferase (HAT; KAT3B); "
            "STRUCTURE: essentially PARALOGUE of CREBBP (~60% overall, ~90% HAT domain identity); "
            "  KIX + CH1/2/3 + TAZ1/2 + RING + HAT + bromo domain; "
            "FUNCTION: "
            "  EP300 acetylates histones H3K18, H3K27, H3K56 → gene ACTIVATION; "
            "  EP300 is the PARALOGUE of CREBBP — same essential function; "
            "  EP300 co-activates same transcription factors as CREBBP (CREB, p53, NF-κB); "
            "  EP300 and CREBBP are partially redundant — explains why EP300 LOF is MILDER; "
            "  EP300 somatic mutations in 60-70% of cancers (bladder, colorectal, gastric); "
            "RUBINSTEIN-TAYBI SYNDROME 2 (RTS2) — CLINICAL: "
            "  PREVALENCE: ~30-50% of all RTS after CREBBP excluded; "
            "  PHENOTYPE — SIMILAR TO RTS1 but MILDER: "
            "    BROAD THUMBS AND GREAT TOES: PRESENT but LESS SEVERE than CREBBP; "
            "      Angulation less pronounced; some patients borderline; "
            "    FACIAL FEATURES: similar to RTS1 but LESS STRIKING; "
            "    INTELLECTUAL DISABILITY: "
            "      Mild-moderate ID (IQ 50-80); systematically MILDER than CREBBP; "
            "      Better language development; "
            "  MALIGNANCY: "
            "    LESS frequent than RTS1 — exact risk uncertain; "
            "    Annual FBC + clinical review STILL RECOMMENDED; "
            "    Pilomatrixoma: less common than CREBBP; "
            "  CARDIAC: CHD ~25% (ASD, VSD); "
            "  BEHAVIOURAL: "
            "    Friendly, social — similar to RTS1; "
            "    Less anxiety than CREBBP in some series; "
            "  GROWTH: short stature — similar to RTS1; "
            "  OBESITY TENDENCY: milder than CREBBP; "
            "EP300 vs CREBBP: "
            "  EP300 systematically MILDER in ID, digit anomalies, malignancy; "
            "  CANNOT be distinguished clinically without genetic testing; "
            "  Test CREBBP FIRST; EP300 if CREBBP negative in RTS phenotype; "
            "DIAGNOSIS: "
            "  Molecular: EP300 sequencing + MLPA; "
            "  WES panel including both CREBBP and EP300; "
            "GENE: EP300; 22q13.2; OMIM gene 602700; RTS2 disease OMIM #613684"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — EP300 / RUBINSTEIN-TAYBI 2: "
            "  DE NOVO: >95%; "
            "  FAMILIAL: rare; "
            "TESTING: "
            "  Test CREBBP first; EP300 second in RTS phenotype; "
            "  EP300 sequencing + MLPA; WES/WGS includes both; "
            "MANAGEMENT: same structure as RTS1 but malignancy risk lower (still surveil)"
        ),
        "disease_category": (
            "EP300-RTS2-MILDER-RTS1-SAME-MECHANISM: "
            "  MILDER than CREBBP/RTS1 — systematically less severe ID and digit anomalies; "
            "  SAME broad thumbs/halluces phenotype — less pronounced; "
            "  TEST CREBBP FIRST — EP300 second in Rubinstein-Taybi phenotype; "
            "  PARALOGUE of CREBBP — redundancy explains milder phenotype"
        ),
    },
    {
        "gene": "ARID1B",
        "protein": (
            "ARID1B -- 6q25.3 AD-de-novo-LOF -- 2285aa -- AT-Rich-Interactive-Domain-1B-"
            "SWI-SNF-Subunit-Coffin-Siris-Syndrome-1-ABSENT-5th-FINGERNAIL-TOENAIL-PATHOGNOMONIC-"
            "MOST-COMMON-CSS-OMIM-614556"
        ),
        "locus": "6q25.3",
        "protein_size": (
            "2285 aa / 250 kDa (ARID1B; AT-rich interactive domain 1B; BAF250b; "
            "SWI/SNF chromatin remodelling complex subunit; "
            "STRUCTURE: ARID (AT-rich interaction) domain + multiple protein interaction domains; "
            "FUNCTION: "
            "  ARID1B is a subunit of the BAF (BRG1/BRM-associated factor) complex; "
            "  BAF complex remodels nucleosome positioning → gene regulation; "
            "  ARID1B directs BAF complex to specific genomic regions via ARID domain; "
            "  ARID1B is mutually exclusive with ARID1A in BAF sub-complexes; "
            "  ARID1B LOF → impaired chromatin accessibility at enhancers; "
            "  ARID1B somatic mutations: most common in ovarian clear cell cancer; "
            "COFFIN-SIRIS SYNDROME 1 (CSS1) — CLINICAL: "
            "  PREVALENCE: CSS overall ~1:50,000; ARID1B = ~50% of all CSS cases; "
            "  CARDINAL FEATURE (PATHOGNOMONIC): "
            "    ABSENT OR HYPOPLASTIC 5th FINGERNAIL AND/OR 5th TOENAIL: "
            "      Absent or severely dysplastic nail of 5th finger + 5th toe; "
            "      May affect 4th finger/toe in some; "
            "      PATHOGNOMONIC — highly specific for CSS spectrum; "
            "  INTELLECTUAL DISABILITY: "
            "    Mild-moderate in ARID1B (milder than other CSS genes); "
            "    IQ range 50-80 typical for ARID1B; "
            "    Language: expressive language impairment; "
            "  COARSE FACIES: "
            "    Full/thick lips; wide mouth; full cheeks; "
            "    Flat nasal bridge; thick nasal alae; "
            "    Depressed nasal bridge; bushy/synophrys; "
            "  HAIR: coarse, thick scalp hair; hypertrichosis; low hairline; "
            "  HYPOTONIA: neonatal; feeding difficulties; "
            "  GROWTH: growth retardation; short stature; "
            "  BEHAVIOURAL: "
            "    Anxiety; ASD features 30-40%; friendly baseline character; "
            "    Repetitive behaviours; "
            "  BRAIN MRI: corpus callosum anomalies 30%; Dandy-Walker; "
            "  CARDIAC: CHD 15-20%; "
            "  5th FINGER/TOE HYPOPLASIA: in addition to nail — short 5th digit; "
            "  ARID1B CSS1 vs OTHER CSS GENES (ARID1A, SMARCB1, SMARCE1, etc.): "
            "    ARID1B = MILDEST among CSS genes; "
            "    Other SWI/SNF genes (SMARCB1) = more severe (Coffin-Siris + schwannomatosis risk); "
            "DIAGNOSIS: "
            "  Clinical: absent/hypoplastic 5th nail + coarse facies + ID + hypotonia; "
            "  WES or gene panel (SWI/SNF complex genes: ARID1B, ARID1A, SMARCB1, etc.); "
            "GENE: ARID1B; 6q25.3; OMIM gene 614556; CSS1 disease OMIM #135900"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — ARID1B / COFFIN-SIRIS 1: "
            "  DE NOVO: >95%; "
            "  FAMILIAL: rare; "
            "TESTING: "
            "  STEP 1: SWI/SNF gene panel: ARID1B, ARID1A, SMARCB1, SMARCE1, SMARCA4, SMARCA2, SOX11; "
            "  STEP 2: WES if panel negative; "
            "  CSS spectrum — test SWI/SNF complex broadly; "
            "MANAGEMENT: "
            "  Nail: cosmetic management; "
            "  ASD therapy if ASD criteria met; "
            "  Brain MRI at diagnosis (corpus callosum anomalies 30%); "
            "  Cardiac echo; ophthalmology; audiology"
        ),
        "disease_category": (
            "ARID1B-CSS1-ABSENT-5th-NAIL-COARSE-FACIES: "
            "  ABSENT/HYPOPLASTIC 5th FINGERNAIL/TOENAIL PATHOGNOMONIC — check nails at diagnosis; "
            "  MILDEST CSS gene (ARID1B) — milder ID than SMARCB1, SMARCE1; "
            "  SWI/SNF COMPLEX GENE — BAF complex chromatin remodelling; "
            "  MOST COMMON CSS GENE (~50%) — screen ARID1B first in CSS phenotype"
        ),
    },
    {
        "gene": "EHMT1",
        "protein": (
            "EHMT1 -- 9q34.3 AD-de-novo-LOF -- 1210aa -- Euchromatic-Histone-Lysine-"
            "Methyltransferase-1-GLP-G9a-Like-Protein-H3K9me1-me2-Kleefstra-Syndrome-"
            "HYPOTONIA-BRACHYCEPHALY-COARSE-FACIES-PATHOGNOMONIC-9q34.3-DELETION-OMIM-607001"
        ),
        "locus": "9q34.3",
        "protein_size": (
            "1210 aa / 136 kDa (EHMT1; euchromatic histone lysine methyltransferase 1; GLP; G9a-like protein; "
            "histone H3K9 mono- and di-methyltransferase (heterochromatin); "
            "STRUCTURE: ankyrin repeats + pre-SET + SET domain + post-SET; "
            "FUNCTION: "
            "  EHMT1 (GLP) functions as OBLIGATE HETERODIMER with EHMT2 (G9a); "
            "  EHMT1-EHMT2 dimer deposits H3K9me1/me2 → gene SILENCING at euchromatin; "
            "  EHMT1 is required for DNA methylation maintenance; "
            "  EHMT1 regulates neuronal differentiation and synaptic plasticity; "
            "  EHMT1 LOF → reduced H3K9me2 → impaired gene silencing → de-repression of developmental genes; "
            "KLEEFSTRA SYNDROME — CLINICAL: "
            "  PREVALENCE: ~1:200,000 (estimated); "
            "  CAUSE: 9q34.3 microdeletion (~75%) OR EHMT1 intragenic variant (~25%); "
            "  CARDINAL FEATURES: "
            "    HYPOTONIA: neonatal hypotonia UNIVERSAL — feeding difficulties; "
            "    BRACHYCEPHALY: short head; "
            "    COARSE FACIES: "
            "      Synophrys (fused eyebrows); wide forehead; upslanting palpebral fissures; "
            "      Broad/flat nasal tip; everted upper lip; prominent jaw; "
            "    INTELLECTUAL DISABILITY: "
            "      Moderate-severe ID; IQ typically 35-60; "
            "      Non-verbal or limited speech majority; "
            "    AUTISM: 30-40%; social communication impairment; "
            "    BEHAVIOUR: "
            "      FRIENDLY, SOCIABLE BEHAVIOUR in young children — CHARACTERISTIC; "
            "      Behavioural deterioration adolescence/adulthood (aggression, regression); "
            "  CARDIAC: CHD 35% (ASD, VSD, PDA); "
            "  BRAIN MRI: corpus callosum anomalies 25%; cerebellar vermis anomalies; "
            "  HEARING: SNHL 25%; "
            "  VISION: strabismus; ptosis; "
            "  RENAL: structural anomalies 30%; "
            "  GENITOURINARY: males — cryptorchidism, micropenis 40%; "
            "  RESPIRATORY: recurrent infections due to hypotonia + aspiration; "
            "  PSYCHIATRIC: psychosis in adolescence/adulthood (unusual for NDD); "
            "  REGRESSION: behavioural regression documented in adolescence; "
            "9q34.3 DELETION: "
            "  Most common cause (CMA detects); "
            "  Size: 100 kb – 9 Mb; EHMT1 haploinsufficiency drives phenotype; "
            "  9q34.3 also contains NOTCH1 (if larger deletion — cardiac more severe); "
            "  Smaller deletion = often EHMT1 + few flanking genes; "
            "DIAGNOSIS: "
            "  CMA first (9q34.3 deletion); "
            "  If CMA negative: EHMT1 sequencing; "
            "GENE: EHMT1; 9q34.3; OMIM gene 607001; Kleefstra syndrome OMIM #610253"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — EHMT1 / KLEEFSTRA SYNDROME: "
            "  DE NOVO: >95%; "
            "  9q34.3 DELETION: chromosomal microarray; "
            "  EHMT1 INTRAGENIC: gene sequencing; "
            "TESTING: "
            "  STEP 1: CMA (9q34.3 deletion — most common cause); "
            "  STEP 2: EHMT1 sequencing (intragenic LOF in 25%); "
            "MANAGEMENT: "
            "  Cardiac echo at diagnosis (CHD 35%); "
            "  Audiology; ophthalmology; "
            "  Behavioural support: ASD intervention; PBS for adolescent aggression; "
            "  Renal USS; "
            "  Psychiatry: psychosis surveillance adolescence/adulthood"
        ),
        "disease_category": (
            "EHMT1-KLEEFSTRA-HYPOTONIA-BRACHYCEPHALY-FRIENDLY-BEHAVIOUR: "
            "  HYPOTONIA (neonatal universal) + BRACHYCEPHALY + COARSE FACIES — Kleefstra triad; "
            "  9q34.3 MICRODELETION most common (~75%) — CMA FIRST; "
            "  FRIENDLY BEHAVIOUR in children → REGRESSION/AGGRESSION in adolescence; "
            "  CARDIAC 35% + RENAL 30% — mandatory organ surveys at diagnosis"
        ),
    },
    {
        "gene": "KAT6B",
        "protein": (
            "KAT6B -- 10q22.2 AD-de-novo-LOF -- 2073aa -- Lysine-Acetyltransferase-6B-"
            "MOZ2-MORF-MYST4-H3K9-H3K14-HAT-Say-Barber-Biesecker-Young-Simpson-Genitopatellar-"
            "Syndrome-ABSENT-PATELLA-PATHOGNOMONIC-OMIM-605880"
        ),
        "locus": "10q22.2",
        "protein_size": (
            "2073 aa / 228 kDa (KAT6B; lysine acetyltransferase 6B; MYST4; MORF; MOZ2; "
            "histone H3K9/H3K14 acetyltransferase (MYST family); "
            "STRUCTURE: MYST domain (zinc finger + acetyl-CoA binding + catalytic) + chromobarrel; "
            "FUNCTION: "
            "  KAT6B acetylates H3K9, H3K14 at promoters and enhancers → gene ACTIVATION; "
            "  KAT6B is closely related to KAT6A (paralogue; MYST family); "
            "  KAT6B recruits p53 and regulates Hox gene expression; "
            "  KAT6B is required for limb development, corpus callosum formation, genital development; "
            "  GENOTYPE-PHENOTYPE: "
            "    TRUNCATING variants DISTAL to HAT domain = SBBYS (milder); "
            "    TRUNCATING variants PROXIMAL to HAT domain / in HAT = Genitopatellar (more severe); "
            "    This LOCATION OF TRUNCATION determines phenotype — KEY GENOTYPE-PHENOTYPE RULE; "
            "SBBYS SYNDROME — CLINICAL: "
            "  Say-Barber-Biesecker-Young-Simpson syndrome: "
            "  INTELLECTUAL DISABILITY: moderate-severe; "
            "  HYPOTHYROIDISM: 50-70%; "
            "  HEARING LOSS: sensorineural 40%; "
            "  HYPOTONIA: universal; "
            "  ABSENT/HYPOPLASTIC PATELLA: "
            "    PATHOGNOMONIC — knee X-ray shows absent or tiny patella; "
            "    May not be clinically apparent — imaging REQUIRED at diagnosis; "
            "  CORPUS CALLOSUM AGENESIS: 80%; severe neurodevelopmental impact; "
            "  MICROCEPHALY: frequent; "
            "  FEEDING: difficulties; NG tube; "
            "GENITOPATELLAR SYNDROME — CLINICAL (more severe alleles): "
            "  ABSENT PATELLA (as above); "
            "  GENITAL ANOMALIES: "
            "    Males: micropenis, cryptorchidism; "
            "    Females: hypoplastic labia; "
            "  CORPUS CALLOSUM AGENESIS: 90%+; "
            "  MORE SEVERE ID than SBBYS; "
            "  POLYHYDRAMNIOS: fetal; "
            "  KIDNEY: anomalies (duplex, horseshoe) 30%; "
            "CLINICAL CLUE: ABSENT PATELLA on knee X-ray = KAT6B until proven otherwise; "
            "  Patellae absent on newborn X-ray → immediate KAT6B sequencing; "
            "GENE: KAT6B; 10q22.2; OMIM gene 605880; SBBYS OMIM #603736; Genitopatellar OMIM #606170"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — KAT6B / SBBYS + GENITOPATELLAR: "
            "  DE NOVO: >99%; "
            "  FAMILIAL: not described; "
            "  GENOTYPE-PHENOTYPE RULE: "
            "    Truncating in exons 16-18 (distal to HAT) → SBBYS (milder); "
            "    Truncating in exons 1-15 (HAT domain/proximal) → Genitopatellar (more severe); "
            "TESTING: "
            "  KAT6B sequencing — note location of truncation (HAT domain proximity); "
            "  Knee X-ray in all suspected cases (absent patella); "
            "  Brain MRI (corpus callosum agenesis 80-90%); "
            "MANAGEMENT: "
            "  Brain MRI at diagnosis; "
            "  Knee X-ray (absent patella imaging); "
            "  Thyroid function tests (hypothyroidism 50-70%); "
            "  Audiology; "
            "  Renal USS; "
            "  Genital anomaly assessment + urology"
        ),
        "disease_category": (
            "KAT6B-SBBYS-GENITOPATELLAR-ABSENT-PATELLA: "
            "  ABSENT/HYPOPLASTIC PATELLA PATHOGNOMONIC — KNEE X-RAY MANDATORY at diagnosis; "
            "  CORPUS CALLOSUM AGENESIS 80-90% — brain MRI mandatory; "
            "  HYPOTHYROIDISM 50-70% — thyroid function tests mandatory; "
            "  GENOTYPE-PHENOTYPE: truncation location determines SBBYS vs Genitopatellar severity"
        ),
    },
    {
        "gene": "KANSL1",
        "protein": (
            "KANSL1 -- 17q21.31 AD-de-novo-LOF -- 1119aa -- KAT8-Regulatory-NSL-Complex-"
            "Subunit-1-KANSL1-MBD5-Like-H4K16-Acetylation-Koolen-de-Vries-Syndrome-"
            "FRIENDLY-SOCIABLE-BEHAVIOUR-PATHOGNOMONIC-17q21.31-MICRODELETION-75pct-OMIM-612452"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1119 aa / 124 kDa (KANSL1; KAT8 regulatory NSL complex subunit 1; "
            "scaffold subunit of the NSL (non-specific lethal) histone acetyltransferase complex; "
            "STRUCTURE: MBD (methyl-CpG binding) domain-like + KAT8-interaction + scaffold domains; "
            "FUNCTION: "
            "  KANSL1 is a scaffold for the NSL complex; "
            "  NSL complex: KANSL1 + KAT8 (MOF; H4K16 acetyltransferase) + other subunits; "
            "  NSL complex deposits H4K16ac at active promoters + H3K9ac; "
            "  H4K16 acetylation: disrupts chromatin compaction → gene ACTIVATION; "
            "  KANSL1 LOF → reduced H4K16ac → impaired transcription of target genes; "
            "  KANSL1 regulates neuronal transcription + synaptic plasticity; "
            "  17q21.31 MICRODELETION MECHANISM: "
            "    Low-copy repeats (LCRs) at 17q21.31 → NAHR (non-allelic homologous recombination); "
            "    ~500 kb deletion (most common) encompasses KANSL1 + CRHR1 + MAPT + others; "
            "    MAPT (microtubule-associated protein tau) included in most deletions; "
            "    KANSL1 haploinsufficiency is the primary driver of KdVS phenotype; "
            "KOOLEN-DE VRIES SYNDROME (KdVS) — CLINICAL: "
            "  PREVALENCE: ~1:16,000 estimated; "
            "  CAUSE: 17q21.31 microdeletion (~75%) OR KANSL1 intragenic variant (~25%); "
            "  CARDINAL FEATURE (PATHOGNOMONIC): "
            "    FRIENDLY/SOCIABLE BEHAVIOUR: "
            "      AFFABLE, AMIABLE, COOPERATIVE personality AFFECTS ESSENTIALLY ALL; "
            "      Even with severe ID — patients UNIVERSALLY cooperative and sociable; "
            "      This is the single most recognised clinical feature; "
            "      DISTINGUISH: Kleefstra (friendly early, regresses), KdVS (persistent friendly); "
            "  INTELLECTUAL DISABILITY: "
            "    Mild-moderate ID; IQ typically 50-75; "
            "    Language: moderate delay; expressive > receptive; "
            "    ASD features: LESS than Kleefstra; "
            "  EPILEPSY: 50% — various types; often controlled; "
            "  CARDIAC CHD: 30% (ASD, PDA, pulmonary stenosis, VSD, AVSD); "
            "  FACIAL FEATURES: "
            "    High forehead; broad nose; bulbous nasal tip; "
            "    Long face; telecanthus; down-slanting palpebral fissures; "
            "    Low-set ears; everted upper lip; open mouth; "
            "  HYPOTONIA: moderate; feeding difficulties infancy; "
            "  SKELETAL: joint hypermobility 80%; talipes; kyphoscoliosis; "
            "  RENAL: anomalies 30% (duplex, horseshoe, reflux); "
            "  VISION: strabismus; hyperopia; "
            "  HEARING: SNHL 30%; conductive HL; "
            "  BRAIN MRI: corpus callosum thin 30%; cerebellar vermis; "
            "17q21.31 vs KANSL1 intragenic: "
            "  17q21.31 microdeletion: also haploinsufficient for MAPT, CRHR1 — rarely affects phenotype; "
            "  KANSL1 intragenic: phenotype essentially identical; "
            "GENE: KANSL1; 17q21.31; OMIM gene 612452; KdVS disease OMIM #610443"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — KANSL1 / KOOLEN-DE VRIES SYNDROME: "
            "  DE NOVO: >99%; "
            "  17q21.31 MICRODELETION: NAHR-mediated ~500 kb deletion (~75% of cases); "
            "  KANSL1 INTRAGENIC: sequencing (~25% of cases); "
            "TESTING: "
            "  STEP 1: CMA (17q21.31 microdeletion — 75% of cases); "
            "  STEP 2: KANSL1 sequencing (intragenic LOF in 25%); "
            "MANAGEMENT: "
            "  Cardiac echo at diagnosis (CHD 30%); "
            "  Epilepsy: standard AED if seizures; "
            "  Audiology; ophthalmology; "
            "  Renal USS; "
            "  Physiotherapy (hypotonia/scoliosis); "
            "  Behavioural: friendly disposition makes ABA/behavioural programmes easier"
        ),
        "disease_category": (
            "KANSL1-KOOLEN-DE-VRIES-FRIENDLY-SOCIABLE-BEHAVIOUR: "
            "  FRIENDLY/SOCIABLE BEHAVIOUR PATHOGNOMONIC — affects virtually all KdVS patients; "
            "  17q21.31 MICRODELETION most common (75%) — CMA FIRST; "
            "  EPILEPSY 50% + CARDIAC CHD 30% — mandatory organ surveys at diagnosis; "
            "  DISTINGUISH from Kleefstra: KdVS = PERSISTENT friendly; Kleefstra = friendly early then regresses"
        ),
    },
]


def _generate_patients_for_gene(gene: str, seed: int) -> list:
    """Generate 40 synthetic patients for one chromatinopathy gene."""
    rng = random.Random(seed)

    severity_params = {
        "KMT2D":  {"severe_pct": 0.10, "moderate_pct": 0.65, "mild_pct": 0.25,
                   "age_min": 0,  "age_max": 48, "iq_mean": 65, "iq_sd": 12},
        "KDM6A":  {"severe_pct": 0.25, "moderate_pct": 0.55, "mild_pct": 0.20,
                   "age_min": 0,  "age_max": 48, "iq_mean": 60, "iq_sd": 13},
        "CREBBP": {"severe_pct": 0.25, "moderate_pct": 0.55, "mild_pct": 0.20,
                   "age_min": 0,  "age_max": 36, "iq_mean": 50, "iq_sd": 13},
        "EP300":  {"severe_pct": 0.10, "moderate_pct": 0.60, "mild_pct": 0.30,
                   "age_min": 0,  "age_max": 48, "iq_mean": 62, "iq_sd": 12},
        "ARID1B": {"severe_pct": 0.10, "moderate_pct": 0.55, "mild_pct": 0.35,
                   "age_min": 0,  "age_max": 60, "iq_mean": 68, "iq_sd": 12},
        "EHMT1":  {"severe_pct": 0.35, "moderate_pct": 0.50, "mild_pct": 0.15,
                   "age_min": 0,  "age_max": 36, "iq_mean": 48, "iq_sd": 12},
        "KAT6B":  {"severe_pct": 0.40, "moderate_pct": 0.50, "mild_pct": 0.10,
                   "age_min": 0,  "age_max": 24, "iq_mean": 42, "iq_sd": 12},
        "KANSL1": {"severe_pct": 0.10, "moderate_pct": 0.60, "mild_pct": 0.30,
                   "age_min": 0,  "age_max": 48, "iq_mean": 63, "iq_sd": 12},
    }
    p = severity_params.get(gene, {"severe_pct": 0.25, "moderate_pct": 0.55, "mild_pct": 0.20,
                                    "age_min": 0, "age_max": 36, "iq_mean": 55, "iq_sd": 13})

    features_map = {
        "KMT2D":  [["arched-eyebrows-lateral-sparse", "persistent-fetal-fingertip-pads"],
                   ["cardiac-CHD-31-50pct-VSD-ASD", "short-stature-postnatal"],
                   ["joint-hypermobility-90pct", "IgA-deficiency-recurrent-otitis"],
                   ["feeding-difficulties-neonatal", "dental-hypodontia-enamel-hypoplasia"]],
        "KDM6A":  [["arched-eyebrows-fingertip-pads-milder", "cardiac-CHD-30pct"],
                   ["short-stature-feeding-difficulties", "X-linked-males-more-severe"],
                   ["immune-dysfunction-IgA-IgG", "joint-hypermobility"],
                   ["milder-ID-than-KMT2D", "hearing-loss-30pct"]],
        "CREBBP": [["broad-thumbs-broad-halluces-pathognomonic", "malignancy-10-15pct-ALL-AML"],
                   ["broad-beaked-nose-grimacing-smile", "pilomatrixoma-benign-skin"],
                   ["downslanting-palpebral-fissures", "cardiac-CHD-24-38pct"],
                   ["moderate-severe-ID-IQ-35-65", "friendly-sociable-OCD-anxiety"]],
        "EP300":  [["broad-thumbs-halluces-less-severe", "milder-than-CREBBP-ID"],
                   ["cardiac-CHD-25pct", "malignancy-less-frequent"],
                   ["friendly-behaviour-similar-RTS1", "short-stature"],
                   ["obesity-tendency-milder", "facial-features-less-striking"]],
        "ARID1B": [["absent-5th-fingernail-toenail-pathognomonic", "coarse-facies"],
                   ["thick-full-lips-wide-mouth", "coarse-scalp-hair-hypertrichosis"],
                   ["corpus-callosum-anomaly-30pct", "ASD-features-30-40pct"],
                   ["mild-moderate-ID-mildest-CSS", "short-stature-hypotonia"]],
        "EHMT1":  [["hypotonia-neonatal-universal", "brachycephaly"],
                   ["coarse-facies-synophrys-everted-lip", "9q34.3-deletion-CMA"],
                   ["friendly-behaviour-childhood-aggression-adolescent", "autism-30-40pct"],
                   ["cardiac-CHD-35pct", "renal-anomalies-30pct"]],
        "KAT6B":  [["absent-hypoplastic-patella-pathognomonic", "corpus-callosum-agenesis-80pct"],
                   ["hypothyroidism-50-70pct", "SNHL-40pct"],
                   ["genital-anomalies-males", "renal-anomalies-30pct"],
                   ["moderate-severe-ID-non-verbal", "feeding-difficulties-NG-tube"]],
        "KANSL1": [["friendly-sociable-behaviour-pathognomonic", "17q21.31-microdeletion-75pct"],
                   ["epilepsy-50pct", "cardiac-CHD-30pct"],
                   ["joint-hypermobility-80pct", "renal-anomalies-30pct"],
                   ["mild-moderate-ID-expressive-impaired", "down-slanting-palpebral-fissures"]],
    }
    feature_options = features_map.get(gene, [["neurodevelopmental-delay"]])

    treatment_map = {
        "KMT2D":  ["cardiac-echo-surveillance", "immunology-IgG-IVIG",
                   "GH-therapy-short-stature", "audiology-hearing-aids", "dental-orthodontist"],
        "KDM6A":  ["cardiac-echo-surveillance", "audiology-hearing-aids",
                   "ASD-intervention", "GH-therapy", "immunology-workup"],
        "CREBBP": ["malignancy-surveillance-annual-FBC", "cardiac-echo",
                   "physiotherapy-broad-thumbs", "psychiatry-anxiety-OCD", "renal-USS"],
        "EP300":  ["cardiac-echo", "malignancy-surveillance",
                   "physiotherapy", "educational-support", "renal-USS"],
        "ARID1B": ["ASD-intervention-30-40pct", "brain-MRI-corpus-callosum",
                   "cardiac-echo", "audiology", "SWI-SNF-panel-sequencing"],
        "EHMT1":  ["cardiac-echo-35pct-CHD", "audiology-25pct-SNHL",
                   "PBS-aggression-adolescent", "ASD-intervention", "renal-USS"],
        "KAT6B":  ["knee-Xray-absent-patella", "brain-MRI-corpus-callosum-agenesis",
                   "thyroid-function-50-70pct", "audiology-40pct", "gastrostomy-feeding"],
        "KANSL1": ["cardiac-echo-30pct-CHD", "epilepsy-AED",
                   "audiology-30pct-SNHL", "physiotherapy-hypotonia-scoliosis", "renal-USS"],
    }
    treatments = treatment_map.get(gene, ["neurodevelopmental-support"])

    mutation_map = {
        "KMT2D":  ["p.Arg2835Ter", "c.8497delC-frameshift", "p.Gln4183Ter-SET-adjacent",
                   "p.Trp2935Ter", "19p13.11-deletion-intragenic", "p.Arg3392Ter"],
        "KDM6A":  ["p.Arg1195Ter", "c.3584delA-frameshift", "Xp11.3-deletion-intragenic",
                   "p.Gln1255Ter", "p.Arg1023Ter-JmjC", "p.Leu1315Ter"],
        "CREBBP": ["16p13.3-deletion-CMA", "p.Arg1446Ter", "c.4331del-frameshift",
                   "p.Glu1994Lys-HAT-domain", "p.Arg2302Ter", "p.Gln1296Ter"],
        "EP300":  ["p.Arg1347Ter", "p.Gln1750Ter", "c.4051del-frameshift",
                   "p.Leu1777Ter-HAT", "22q13.2-deletion", "p.Arg2006Ter"],
        "ARID1B": ["p.Arg1362Ter", "6q25.3-deletion-CMA", "c.4084del-frameshift",
                   "p.Trp1736Ter", "p.Arg1779Ter-ARID-domain", "p.Gln1060Ter"],
        "EHMT1":  ["9q34.3-deletion-CMA-500kb", "p.Arg471Ter", "c.1412del-frameshift",
                   "p.Arg1005Ter-SET-domain", "9q34.3-deletion-1Mb", "p.Gln728Ter"],
        "KAT6B":  ["p.Arg1699Ter-exon18-SBBYS", "p.Arg1123Ter-exon14-GenitoPatellar",
                   "c.5097del-frameshift-distal", "p.Gln1456Ter-exon16",
                   "p.Arg615Ter-proximal-HAT-GenitoPatellar", "c.3748del-frameshift"],
        "KANSL1": ["17q21.31-microdeletion-500kb-NAHR", "p.Arg792Ter", "c.2375del-frameshift",
                   "p.Gln1044Ter", "17q21.31-microdeletion-750kb", "p.Arg530Ter"],
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
        iq = max(20, int(rng.normalvariate(p["iq_mean"], p["iq_sd"])))
        # Gene-specific flags
        cardiac      = gene in ("KMT2D", "KDM6A", "CREBBP", "EP300", "EHMT1", "KAT6B", "KANSL1") and rng.random() < (
            0.40 if gene == "KMT2D" else 0.35 if gene == "KDM6A" else
            0.31 if gene == "CREBBP" else 0.25 if gene == "EP300" else
            0.35 if gene == "EHMT1" else 0.30 if gene == "KAT6B" else 0.30)
        malignancy   = gene in ("CREBBP", "EP300") and rng.random() < (
            0.12 if gene == "CREBBP" else 0.05)
        absent_nail  = gene == "ARID1B" and rng.random() < 0.90
        absent_patella = gene == "KAT6B" and rng.random() < 0.90
        corpus_cal   = gene in ("KAT6B", "KANSL1", "EHMT1", "ARID1B") and rng.random() < (
            0.85 if gene == "KAT6B" else 0.30 if gene == "KANSL1" else
            0.25 if gene == "EHMT1" else 0.30)
        epilepsy     = gene in ("KANSL1", "EHMT1", "KAT6B") and rng.random() < (
            0.50 if gene == "KANSL1" else 0.30 if gene == "EHMT1" else 0.35)
        friendly_beh = gene in ("KANSL1", "KMT2D", "EHMT1", "CREBBP") and rng.random() < (
            0.95 if gene == "KANSL1" else 0.70 if gene == "KMT2D" else
            0.60 if gene == "EHMT1" else 0.70)
        hypothyroid  = gene == "KAT6B" and rng.random() < 0.60
        deletion_9q  = gene == "EHMT1" and rng.random() < 0.75
        deletion_17q = gene == "KANSL1" and rng.random() < 0.75
        treatment    = rng.choice(treatments)
        mutation     = rng.choice(mutations)
        features     = rng.choice(feature_options)
        patients.append({
            "id":                   f"{gene}-{seed}-{i+1:03d}",
            "gene":                 gene,
            "age_at_diagnosis_mo":  age_dx * 12 if age_dx < 5 else age_dx,
            "severity":             severity,
            "iq_estimate":          iq,
            "associated_features":  features,
            "cardiac_defect":       cardiac,
            "malignancy_risk":      malignancy,
            "absent_nail":          absent_nail,
            "absent_patella":       absent_patella,
            "corpus_callosum":      corpus_cal,
            "epilepsy":             epilepsy,
            "friendly_behaviour":   friendly_beh,
            "hypothyroidism":       hypothyroid,
            "deletion_9q34":        deletion_9q,
            "deletion_17q21":       deletion_17q,
            "treatment":            treatment,
            "mutation":             mutation,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Chromatinopathy-Epigenetic-Atlas."""
    return {
        "atlas":          "Hereditary-Chromatinopathy-Epigenetic-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Chromatinopathy & Epigenetic Syndrome Atlas "
            "(KMT2D-KDM6A-CREBBP-EP300-ARID1B-EHMT1-KAT6B-KANSL1)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "KMT2D":  "AD LOF de novo (Kabuki 1; ARCHED EYEBROWS + PERSISTENT FETAL FINGERTIP PADS PATHOGNOMONIC; 60-75% all Kabuki; cardiac 31-50%; immune IgA; H3K4me1/me2 enhancer)",
            "KDM6A":  "X-linked LOF (Kabuki 2; MILDER in heterozygous females; MORE SEVERE in hemizygous males; 5% of Kabuki; H3K27me3 demethylase)",
            "CREBBP": "AD LOF de novo (RTS1; BROAD THUMBS + BROAD HALLUCES PATHOGNOMONIC; 50-70% RTS; MALIGNANCY 10-15% ALL/AML; HAT acetyltransferase)",
            "EP300":  "AD LOF de novo (RTS2; MILDER than CREBBP; same broad thumbs less severe; less malignancy; paralogue CREBBP same HAT function)",
            "ARID1B": "AD LOF de novo (CSS1; ABSENT/HYPOPLASTIC 5th FINGERNAIL/TOENAIL PATHOGNOMONIC; ~50% CSS; mildest CSS gene; SWI/SNF BAF complex)",
            "EHMT1":  "AD LOF de novo (Kleefstra; 9q34.3 MICRODELETION 75%; HYPOTONIA+BRACHYCEPHALY+COARSE FACIES triad; cardiac 35%; friendly→regression; H3K9me2)",
            "KAT6B":  "AD LOF de novo (SBBYS/Genitopatellar; ABSENT PATELLA PATHOGNOMONIC; corpus callosum agenesis 80%; hypothyroidism 50-70%; HAT proximal truncation=severe)",
            "KANSL1": "AD LOF de novo (Koolen-de Vries; 17q21.31 MICRODELETION 75%; FRIENDLY/SOCIABLE BEHAVIOUR PATHOGNOMONIC; epilepsy 50%; CHD 30%; H4K16ac NSL complex)",
        },
        "key_clinical_rules": [
            "KMT2D (Kabuki 1): ARCHED EYEBROWS (lateral third sparse) + PERSISTENT FETAL FINGERTIP PADS PATHOGNOMONIC; cardiac echo MANDATORY (CHD 31-50%); IgA/IgG workup",
            "KDM6A (Kabuki 2): HEMIZYGOUS MALES MORE SEVERE than heterozygous females — X-inactivation escape; test AFTER KMT2D negative; 5% of all Kabuki",
            "CREBBP (RTS1): BROAD THUMBS + BROAD HALLUCES PATHOGNOMONIC — check digit width at diagnosis; MALIGNANCY 10-15% (ALL, AML, brain tumours) — annual FBC MANDATORY",
            "CREBBP vs EP300: CREBBP FIRST (most common RTS); EP300 MILDER phenotype — cannot distinguish clinically; test CREBBP → EP300; pilomatrixoma more CREBBP",
            "ARID1B (CSS1): ABSENT/HYPOPLASTIC 5th FINGERNAIL/TOENAIL PATHOGNOMONIC — examine nails; MILDEST CSS gene; brain MRI (corpus callosum 30%); SWI/SNF panel",
            "EHMT1 (Kleefstra): 9q34.3 MICRODELETION most common (75%) — CMA FIRST; hypotonia universal; FRIENDLY CHILDHOOD → AGGRESSION/REGRESSION ADOLESCENCE",
            "KAT6B (SBBYS/Genitopatellar): ABSENT PATELLA — KNEE X-RAY MANDATORY at diagnosis; corpus callosum agenesis 80% — brain MRI mandatory; hypothyroidism 50-70%",
            "KAT6B GENOTYPE-PHENOTYPE: truncation DISTAL to HAT domain = SBBYS (milder); truncation PROXIMAL/IN HAT = Genitopatellar (more severe) — NOTE POSITION",
            "KANSL1 (Koolen-de Vries): FRIENDLY/SOCIABLE BEHAVIOUR PATHOGNOMONIC — persists lifelong; 17q21.31 microdeletion 75% — CMA FIRST; epilepsy 50%; CHD 30%",
            "KdVS vs Kleefstra: KANSL1 = PERSISTENT friendly behaviour; EHMT1 = friendly early then REGRESSES adolescence — KEY DISTINGUISHING FEATURE",
            "CHROMATINOPATHY PANEL: all 8 genes cover Kabuki, Rubinstein-Taybi, Coffin-Siris, Kleefstra, SBBYS, Koolen-de Vries — common in developmental paediatrics + genetics",
            "9q34.3 or 17q21.31 microdeletion on CMA → direct diagnosis of Kleefstra or KdVS — CMA FIRST in both; intragenic variants need sequencing",
        ],
        "gene_panel_note": (
            "Chromatinopathy / epigenetic syndrome panel (2024): KMT2D, KDM6A, CREBBP, EP300, ARID1B, EHMT1, KAT6B, KANSL1, "
            "plus NIPBL/SMC1A/SMC3/RAD21/HDAC8 (Cornelia de Lange), NSD1 (Sotos), EZH2 (Weaver), "
            "SETD5 (MRD23), MBD5 (2q23.1 deletion syndrome), DNMT3A (Tatton-Brown-Rahman); "
            "CMA + gene sequencing complementary — CMA detects 9q34.3 (Kleefstra) + 17q21.31 (KdVS); "
            "WES/WGS increasingly first-line for complex NDD phenotypes"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Chromatinopathy-Epigenetic-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)
        severe_n       = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n     = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n         = sum(1 for p in patients if p["severity"] == "mild")
        mean_iq        = round(sum(p["iq_estimate"] for p in patients) / n, 1)
        cardiac_n      = sum(1 for p in patients if p["cardiac_defect"])
        malignancy_n   = sum(1 for p in patients if p["malignancy_risk"])
        absent_nail_n  = sum(1 for p in patients if p["absent_nail"])
        absent_pat_n   = sum(1 for p in patients if p["absent_patella"])
        corpus_n       = sum(1 for p in patients if p["corpus_callosum"])
        epilepsy_n     = sum(1 for p in patients if p["epilepsy"])
        friendly_n     = sum(1 for p in patients if p["friendly_behaviour"])
        hypothyroid_n  = sum(1 for p in patients if p["hypothyroidism"])
        del9q_n        = sum(1 for p in patients if p["deletion_9q34"])
        del17q_n       = sum(1 for p in patients if p["deletion_17q21"])
        mean_age       = round(sum(p["age_at_diagnosis_mo"] for p in patients) / n, 1)
        mutations_seen = list({p["mutation"] for p in patients})
        genes_data.append({
            "gene":              gene,
            "locus":             gene_info["locus"],
            "n_patients":        n,
            "severe_pct":        round(severe_n / n * 100, 1),
            "moderate_pct":      round(moderate_n / n * 100, 1),
            "mild_pct":          round(mild_n / n * 100, 1),
            "mean_iq":           mean_iq,
            "cardiac_pct":       round(cardiac_n / n * 100, 1),
            "malignancy_pct":    round(malignancy_n / n * 100, 1),
            "absent_nail_pct":   round(absent_nail_n / n * 100, 1),
            "absent_patella_pct": round(absent_pat_n / n * 100, 1),
            "corpus_cal_pct":    round(corpus_n / n * 100, 1),
            "epilepsy_pct":      round(epilepsy_n / n * 100, 1),
            "friendly_pct":      round(friendly_n / n * 100, 1),
            "hypothyroid_pct":   round(hypothyroid_n / n * 100, 1),
            "deletion_9q34_pct": round(del9q_n / n * 100, 1),
            "deletion_17q21_pct": round(del17q_n / n * 100, 1),
            "mean_age_dx_mo":    mean_age,
            "sample_mutations":  mutations_seen[:4],
            "protein":           gene_info["protein"],
            "inheritance":       gene_info["inheritance"][:200],
            "disease_category":  gene_info["disease_category"],
        })
    return {
        "atlas": "Hereditary-Chromatinopathy-Epigenetic-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Chromatinopathy-Epigenetic-Atlas."""
    definitions = [
        {
            "term": "Chromatinopathies — Chromatin Modifier Syndromes: Classification and Diagnostic Approach",
            "genes": ["KMT2D", "KDM6A", "CREBBP", "EP300", "ARID1B", "EHMT1", "KAT6B", "KANSL1"],
            "definition": (
                "HEREDITARY CHROMATINOPATHY & EPIGENETIC SYNDROMES — OVERVIEW: "
                "DEFINITION: Chromatinopathies = monogenic disorders caused by LOF of CHROMATIN MODIFIER genes "
                "→ impaired histone modification → disrupted gene expression during development; "
                "CLASSIFICATION BY HISTONE MODIFICATION: "
                "  H3K4 METHYLATION (COMPASS-like): KMT2D (Kabuki 1); KDM6A partly scaffolds; "
                "  H3K27 DEMETHYLATION (JmjC): KDM6A (Kabuki 2); "
                "  H3K9/H3K14 ACETYLATION (HAT): CREBBP (RTS1); EP300 (RTS2); KAT6B (SBBYS); "
                "  H3K9me2 METHYLATION (SET domain): EHMT1 (Kleefstra); "
                "  BAF COMPLEX REMODELLING (SWI/SNF): ARID1B (CSS1); "
                "  H4K16 ACETYLATION (NSL complex): KANSL1 (Koolen-de Vries); "
                "CLINICAL SYNDROMES COVERED: "
                "  KABUKI SYNDROME (KS1/KS2): KMT2D + KDM6A; arched eyebrows + fingertip pads; "
                "  RUBINSTEIN-TAYBI (RTS1/RTS2): CREBBP + EP300; broad thumbs/halluces; "
                "  COFFIN-SIRIS (CSS1): ARID1B; absent 5th nail; SWI/SNF; "
                "  KLEEFSTRA: EHMT1; 9q34.3 deletion; hypotonia + brachycephaly; "
                "  SBBYS/GENITOPATELLAR: KAT6B; absent patella; corpus callosum agenesis; "
                "  KOOLEN-DE VRIES (KdVS): KANSL1; 17q21.31 deletion; friendly behaviour; "
                "DIAGNOSTIC ALGORITHM: "
                "  STEP 1: CMA — detects 9q34.3 (Kleefstra 75%) + 17q21.31 (KdVS 75%); "
                "  STEP 2: Chromatinopathy gene panel: KMT2D, KDM6A, CREBBP, EP300, ARID1B, EHMT1, KAT6B, KANSL1; "
                "  STEP 3: WES/WGS if panel negative — broader NDD/chromatinopathy genes; "
                "PATHOGNOMONIC SIGNS SUMMARY: "
                "  KMT2D: arched eyebrows + persistent fetal fingertip pads; "
                "  CREBBP: broad thumbs + broad halluces; "
                "  ARID1B: absent/hypoplastic 5th fingernail; "
                "  KAT6B: absent patella on knee X-ray; "
                "  KANSL1: persistent friendly/sociable behaviour; "
                "SHARED FEATURES ACROSS CHROMATINOPATHIES: "
                "  Intellectual disability (mild-severe); "
                "  Congenital heart disease (variable rate: 24-50%); "
                "  Feeding difficulties / hypotonia neonatal; "
                "  Short stature; growth retardation; "
                "  Friendly/sociable behaviour (especially KMT2D, KANSL1, CREBBP, EHMT1 early)"
            ),
        },
        {
            "term": "Kabuki Syndrome (KMT2D vs KDM6A) — Differential Diagnosis and X-linked Severity",
            "genes": ["KMT2D", "KDM6A"],
            "definition": (
                "KABUKI SYNDROME — KS1 vs KS2 DIFFERENTIAL: "
                "KABUKI SYNDROME 1 (KMT2D): "
                "  PREVALENCE: 60-75% of all Kabuki cases; "
                "  CAUSE: KMT2D de novo LOF (frameshift, nonsense, splice, missense SET domain); "
                "  CARDINAL FEATURES: arched eyebrows (lateral third sparse) + persistent fetal fingertip pads "
                "    + long palpebral fissures + broad nasal tip + large prominent ears; "
                "  SEVERITY: mild-moderate ID (IQ 50-75); "
                "  CARDIAC: 31-50%; VSD, ASD, coarctation, AVSD; "
                "  IMMUNE: IgA deficiency 50%; IgG reduced; recurrent otitis media; "
                "  TESTING: KMT2D sequencing FIRST; MLPA if negative (intragenic deletions 10%); "
                "KABUKI SYNDROME 2 (KDM6A): "
                "  PREVALENCE: ~5% of all Kabuki; "
                "  CAUSE: KDM6A LOF on Xp11.3; "
                "  SEX-DEPENDENT SEVERITY: "
                "    HETEROZYGOUS FEMALES: MILDER — KDM6A PARTIALLY ESCAPES X-INACTIVATION; "
                "      Some functional KDM6A from Xi → partial compensation; "
                "      Mild ID; less severe facial features; shorter palpebral fissures; "
                "    HEMIZYGOUS MALES: MORE SEVERE — NO functional KDM6A at all; "
                "      Moderate-severe ID; more organ involvement; "
                "      More severe cardiac; more severe facial features; "
                "  TESTING: KDM6A after KMT2D negative; "
                "CLINICAL DISTINCTION KS1 vs KS2: "
                "  Cannot be distinguished clinically without sequencing; "
                "  Males with Kabuki + more severe ID → KDM6A hemizygous; "
                "  Testing: WES panel including BOTH KMT2D + KDM6A simultaneously; "
                "MANAGEMENT SHARED KS1+KS2: "
                "  Cardiac echo at diagnosis; "
                "  IgG/IgA: IVIG if recurrent severe infections; "
                "  GH therapy: consider (not all are GH-deficient); "
                "  Dental: orthodontist early — hypodontia; delayed eruption; "
                "  Renal USS (anomalies 30%); ophthalmology; audiology"
            ),
        },
        {
            "term": "Rubinstein-Taybi Syndrome (CREBBP vs EP300) — Broad Thumb Protocol and Malignancy Surveillance",
            "genes": ["CREBBP", "EP300"],
            "definition": (
                "RUBINSTEIN-TAYBI SYNDROME — RTS1 vs RTS2: "
                "BROAD THUMBS/HALLUCES — CLINICAL ASSESSMENT: "
                "  Measure distal phalanx width of thumbs and great toes; "
                "  Broad = distal phalanx visibly wider than middle phalanx (angulated); "
                "  CREBBP (RTS1): more pronounced broadening; angulated ('hitchhiker') thumbs in some; "
                "  EP300 (RTS2): broadening present but less severe — borderline; "
                "  Bilateral x-rays of thumbs + great toes at diagnosis; "
                "MALIGNANCY SURVEILLANCE — RTS1 (CREBBP): "
                "  LIFETIME RISK 10-15%: "
                "    Haematological: ALL, AML (highest risk in childhood-adolescence); "
                "    CNS: medulloblastoma, meningioma; "
                "    Other: neuroblastoma; leukaemia; "
                "  PILOMATRIXOMA: calcified benign skin tumour — common (up to 25%+); "
                "    NOT malignant; consider excision if large/symptomatic; "
                "  SURVEILLANCE PROTOCOL: "
                "    Annual FBC with differential (haematological malignancy); "
                "    Clinical examination for skin lumps (pilomatrixoma); "
                "    CNS: MRI if neurological symptoms; "
                "    Threshold to investigate should be LOW; "
                "MALIGNANCY SURVEILLANCE — RTS2 (EP300): "
                "  RISK LOWER than RTS1 — exact incidence uncertain; "
                "  STILL RECOMMEND annual FBC + clinical review; "
                "  Pilomatrixoma: less frequent; "
                "TREATMENT: "
                "  NO disease-modifying therapy (HAT replacement); "
                "  HDAC inhibitors (e.g. vorinostat): investigated in mouse models; not clinical; "
                "BEHAVIOURAL: "
                "  OCD features: CBT + SSRIs; "
                "  Anxiety: environmental modification + CBT; "
                "  FRIENDLY baseline: facilitates ABA/positive behaviour support; "
                "DIFFERENTIAL RTS vs Kabuki: "
                "  Both: ID, facial features, short stature; "
                "  BROAD THUMBS → RTS (Kabuki has fingertip pads, not broad thumbs per se); "
                "  FINGERTIP PADS → Kabuki (not RTS)"
            ),
        },
        {
            "term": "Coffin-Siris Syndrome (ARID1B) — 5th Nail Protocol and SWI/SNF Complex",
            "genes": ["ARID1B"],
            "definition": (
                "COFFIN-SIRIS SYNDROME 1 (ARID1B) — CLINICAL PROTOCOL: "
                "5th NAIL ASSESSMENT: "
                "  Examine 5th finger and 5th toe nails at every clinical encounter; "
                "  ABSENT: nail completely absent — skin only at nail bed; "
                "  HYPOPLASTIC: nail very small, thin, dysplastic; "
                "  May affect 4th digit in some patients; "
                "  PATHOGNOMONIC — absence/hypoplasia in the context of ID + coarse facies = CSS; "
                "SWI/SNF COMPLEX — WHY MULTIPLE GENES CAUSE CSS: "
                "  BAF complex: ARID1B + ARID1A + SMARCB1 + SMARCE1 + SMARCA4 + SMARCA2 + BRG1; "
                "  All BAF complex subunits → Coffin-Siris spectrum when mutated; "
                "  ARID1B is the MOST COMMON (50%); "
                "  SMARCB1 causes CSS + SCHWANNOMATOSIS (malignancy risk) — important distinction; "
                "  SMARCE1 causes meningioma predisposition; "
                "  Test WHOLE SWI/SNF panel if CSS phenotype (not ARID1B alone); "
                "SEVERITY SPECTRUM ARID1B vs other CSS genes: "
                "  ARID1B: MILDEST — mild-moderate ID; "
                "  SMARCB1: more severe; schwannomatosis risk; "
                "  SMARCE1: moderate-severe; meningioma; "
                "  Test ARID1B FIRST; full panel second; "
                "CSS1 MANAGEMENT: "
                "  Brain MRI (corpus callosum anomalies 30%); "
                "  Cardiac echo (CHD 15-20%); "
                "  Ophthalmology; audiology; "
                "  ASD therapy (30-40%); "
                "  Nail: no surgical intervention required; "
                "  Learning support: language therapy"
            ),
        },
        {
            "term": "Kleefstra vs Koolen-de Vries — Friendly Behaviour Differential and Microdeletion Testing",
            "genes": ["EHMT1", "KANSL1"],
            "definition": (
                "KLEEFSTRA (EHMT1/9q34.3) vs KOOLEN-DE VRIES (KANSL1/17q21.31) — KEY DIFFERENTIAL: "
                "FRIENDLY BEHAVIOUR: "
                "  BOTH syndromes have friendly/sociable behaviour; "
                "  KEY DIFFERENCE: "
                "    KLEEFSTRA: friendly in CHILDHOOD → REGRESSION + AGGRESSION in adolescence/adulthood; "
                "    KOOLEN-DE VRIES: friendly/sociable PERSISTS across lifespan — PATHOGNOMONIC for KdVS; "
                "  This temporal pattern is the best clinical differentiator; "
                "MICRODELETION TESTING: "
                "  9q34.3 deletion (Kleefstra): "
                "    CMA detects — ~75% of Kleefstra cases; "
                "    Size: 100 kb – several Mb; EHMT1 haploinsufficiency drives phenotype; "
                "    If CMA normal: EHMT1 sequencing; "
                "  17q21.31 deletion (Koolen-de Vries): "
                "    CMA detects — ~75% of KdVS cases; "
                "    Size: ~500 kb NAHR-mediated deletion; "
                "    MAPT included (most deletions) — not relevant clinically in childhood; "
                "    If CMA normal: KANSL1 sequencing; "
                "KLEEFSTRA SURVEILLANCE: "
                "  Cardiac echo (CHD 35%); audiology (SNHL 25%); renal USS (30%); "
                "  Psychiatry surveillance adolescence (psychosis documented in some); "
                "  PBS for behavioural deterioration; "
                "KdVS SURVEILLANCE: "
                "  Cardiac echo (CHD 30%); epilepsy management (50%); audiology (SNHL 30%); "
                "  Renal USS (anomalies 30%); scoliosis monitoring (joint hypermobility); "
                "BOTH: no disease-modifying therapy; supportive/surveillance standard"
            ),
        },
    ]
    return {
        "atlas": "Hereditary-Chromatinopathy-Epigenetic-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }
