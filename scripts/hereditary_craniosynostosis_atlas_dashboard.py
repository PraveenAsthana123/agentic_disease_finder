#!/usr/bin/env python3
"""Hereditary-Craniosynostosis-Atlas — Complete 8-Gene Craniosynostosis & Craniofacial Genetics Atlas
FGFR2   (fibroblast growth factor receptor 2; 821 aa; 10q26.13; AD GOF;
         Crouzon/Apert/Pfeiffer/Beare-Stevenson/Jackson-Weiss syndromes;
         Apert S252W/P253R — MITTEN HAND PATHOGNOMONIC; Crouzon C342R/W;
         most common FGFR gene in syndromic craniosynostosis (~50%);
         seed SEED_BASE+0) ·
FGFR1   (fibroblast growth factor receptor 1; 822 aa; 8p11.23; AD GOF;
         Pfeiffer syndrome type 1 (P252R); broad thumbs/toes PATHOGNOMONIC;
         osteoglophonic dysplasia (Y372C); Kallmann syndrome context;
         seed SEED_BASE+1) ·
FGFR3   (fibroblast growth factor receptor 3; 806 aa; 4p16.3; AD GOF;
         Muenke syndrome (P250R ONLY — most common single-gene craniosynostosis ~1:30,000);
         unicoronal > bicoronal; SNHL 30%; INCOMPLETE PENETRANCE ~60%;
         SAME GENE different mutation = achondroplasia (G380R) — CRITICAL DDx;
         seed SEED_BASE+2) ·
TWIST1  (Twist-related protein 1; 202 aa; 7p21.1; AD LOF haploinsufficiency;
         Saethre-Chotzen syndrome; UNICORONAL + PTOSIS + LOW-SET HAIRLINE PATHOGNOMONIC;
         soft-tissue digit 2/3 syndactyly; breast cancer risk elevated (TWIST1 EMT);
         seed SEED_BASE+3) ·
TCF12   (transcription factor 12; 598 aa; 15q21.3; AD LOF;
         coronal craniosynostosis, Saethre-Chotzen-like; intellectual disability 35%;
         MISSED BY OLDER GENE PANELS — must be specifically included;
         seed SEED_BASE+4) ·
EFNB1   (ephrin-B1; 346 aa; Xq12; X-linked;
         craniofrontonasal syndrome (CFNS);
         PARADOX: heterozygous FEMALES more severely affected than hemizygous males;
         HYPERTELORISM + CORONAL synostosis + BIFID NASAL TIP PATHOGNOMONIC triad;
         cellular interference mechanism (X-inactivation mosaicism);
         seed SEED_BASE+5) ·
ERF     (ETS2 repressor factor; 548 aa; 19q13.2; AD LOF;
         multi-suture craniosynostosis; Crouzon-like without FGFR2 mutation;
         CEREBELLAR TONSILLAR HERNIATION 40% — MRI mandatory;
         somatic mosaicism common — enhanced sequencing sensitivity critical;
         seed SEED_BASE+6) ·
RAB23   (RAB23, member RAS oncogene family; 237 aa; 6p12.1; AR;
         Carpenter syndrome type 1; MULTISUTURE + PREAXIAL POLYSYNDACTYLY + obesity;
         CARDIAC DEFECTS 50% (VSD/PDA/PS); intellectual disability all;
         biallelic LOF; 20q13 region Hedgehog pathway regulator;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 3014-3021)
"""
import random

SEED_BASE = 3014

ATLAS_GENES = [
    {
        "gene": "FGFR2",
        "protein": (
            "FGFR2 -- 10q26.13 AD-GOF -- 821aa -- Fibroblast-Growth-Factor-Receptor-2-"
            "Crouzon-Apert-Pfeiffer-Beare-Stevenson-Craniosynostosis-"
            "Apert-MITTEN-HAND-PATHOGNOMONIC-OMIM-176943"
        ),
        "locus": "10q26.13",
        "protein_size": (
            "821 aa / 88 kDa (FGFR2; type I transmembrane receptor tyrosine kinase; "
            "STRUCTURE: signal peptide + 3 Ig-like extracellular domains (D1-D2-D3) + "
            "TM domain + split intracellular TK domain; "
            "D2-D3 = ligand-binding domain; pathogenic mutations cluster in D2-D3 linker; "
            "GAIN-OF-FUNCTION MECHANISM: "
            "  Cysteine mutations (C342R, C342W, C342Y, C342S in D3) → unpaired cysteine → "
            "  aberrant disulfide bond → ligand-independent receptor dimerisation → "
            "  constitutive FGFR2 activation → RAS-MAPK, PI3K-AKT, STAT3 hyperactivation; "
            "  S252W (Apert) and P253R (Apert): GAIN-OF-FUNCTION via broadened ligand specificity; "
            "  Apert mutations: linker between D2 and D3 → recruit FGF2 with higher affinity; "
            "CLINICAL SYNDROMES (allele-specific): "
            "  CROUZON SYNDROME (C342R/W/Y, W290C, Y340C): "
            "    Premature fusion of CORONAL + LAMBDOID + SAGITTAL sutures; brachycephaly; "
            "    Exophthalmos/proptosis (shallow orbits); hypertelorism; "
            "    NORMAL LIMBS (no hand/foot anomaly — key DDx from Apert); "
            "    Normal intelligence; acanthosis nigricans with FGFR3 A391E mutation; "
            "  APERT SYNDROME (S252W ~66%, P253R ~33%): "
            "    Bicoronal synostosis; turribrachycephaly; "
            "    MITTEN HAND: complex osseous syndactyly fingers 2-3-4 PATHOGNOMONIC; "
            "    Toes 2-3-4 syndactyly; "
            "    Intellectual disability in 50% (more common than Crouzon); "
            "    Midface hypoplasia; choanal stenosis; "
            "    S252W > P253R: similar cranial; P253R slightly milder hand; "
            "  PFEIFFER SYNDROME TYPE 2/3 (C342R, S351C, others): "
            "    Type 2: CLOVERLEAF skull (kleeblattschaedel) PATHOGNOMONIC; elbow ankylosis; "
            "    Type 3: turribrachycephaly + BROAD THUMBS/TOES (short broad distal phalanges); "
            "    Severe; respiratory compromise; ELBOW ANKYLOSIS distinguishes from Crouzon; "
            "  BEARE-STEVENSON CUTIS GYRATA (Y375C, S372C): "
            "    CUTIS GYRATA (ridged corrugated skin on scalp/face) PATHOGNOMONIC; "
            "    Craniosynostosis + choanal atresia + anogenital anomalies; "
            "  JACKSON-WEISS: C342R allele with foot anomalies; "
            "encoded 10q26.13; OMIM gene 176943; Crouzon #123500; Apert #101200"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (GOF) — FGFR2 / CRANIOSYNOSTOSIS SYNDROMES: "
            "  DE NOVO in Apert (>95% cases — high paternal age effect); "
            "  INHERITED in Crouzon (autosomal dominant; variable expressivity); "
            "  GENOTYPE-PHENOTYPE CORRELATION: "
            "    C342R/W/Y/S → Crouzon or Pfeiffer type 2/3 depending on other modifiers; "
            "    S252W → Apert (66%); P253R → Apert (33%); "
            "    SAME MUTATION can produce different phenotypes in families; "
            "  SOMATIC MOSAICISM: rare; milder phenotype in mosaic parent; "
            "  FGFR2 EXON IIIa/IIIc ALTERNATIVE SPLICING: "
            "    Exon IIIa: mesenchyme/cranial suture form (pathogenic mutations here); "
            "    Exon IIIc: epithelial form; "
            "DIAGNOSIS: "
            "  Cranial CT: suture fusion pattern; 3D reconstruction; "
            "  FGFR2 hotspot panel: C342, W290, Y340, S252, P253 codons screened first; "
            "  Full FGFR2 sequencing if hotspot negative; "
            "  Molecular confirmation before surgical planning; "
            "SURGICAL MANAGEMENT: "
            "  Crouzon/Apert: fronto-orbital advancement (FOA) age 6-12 months; "
            "  Le Fort III / distraction osteogenesis for midface: age 5-7 years; "
            "  Apert hands: finger separation staged surgeries age 1-3 years; "
            "  Ventriculoperitoneal shunt if raised ICP"
        ),
        "disease_category": (
            "FGFR2-CRANIOSYNOSTOSIS-GOF-APERT-CROUZON-PFEIFFER: "
            "  KEY RULE: APERT = MITTEN HAND (osseous 2-3-4 syndactyly) — no other craniosynostosis has this; "
            "  CROUZON = NO limb anomaly (key DDx vs Apert and Pfeiffer); "
            "  PFEIFFER BROAD THUMBS/TOES = pathognomonic (all 3 types); TYPE 2 = cloverleaf = most severe; "
            "  BEARE-STEVENSON: ridged scalp skin (cutis gyrata) pathognomonic; "
            "  SURGICAL TIMING: FOA 6-12 months; raised ICP → urgent; "
            "  DE NOVO APERT: paternal age effect (father age >35-40 = 3× risk)"
        ),
    },
    {
        "gene": "FGFR1",
        "protein": (
            "FGFR1 -- 8p11.23 AD-GOF -- 822aa -- Fibroblast-Growth-Factor-Receptor-1-"
            "Pfeiffer-Type1-P252R-Broad-Thumb-Big-Toe-PATHOGNOMONIC-OMIM-136350"
        ),
        "locus": "8p11.23",
        "protein_size": (
            "822 aa / 91 kDa (FGFR1; structurally homologous to FGFR2; same 3-Ig-like domain architecture; "
            "FGFR1 vs FGFR2 DISTINCTION: "
            "  FGFR1: P252R = D2-D3 linker mutation (analogous to FGFR2 S252W in Apert); "
            "  Gain-of-function: broadened FGF ligand specificity; constitutive signalling; "
            "CLINICAL SYNDROME — PFEIFFER TYPE 1 (P252R): "
            "  MILDEST of three Pfeiffer types (types 1-3); "
            "  CRANIOSYNOSTOSIS: unicoronal or bicoronal; brachycephaly; "
            "  BROAD THUMBS (short, wide): PATHOGNOMONIC triad feature; "
            "  BROAD BIG TOES: PATHOGNOMONIC; "
            "  Normal intelligence in most; "
            "  Normal life expectancy; "
            "PFEIFFER TYPE COMPARISON: "
            "  Type 1 (FGFR1 P252R): broad thumbs/toes + brachycephaly; NO elbow ankylosis; "
            "  Type 2 (FGFR2): CLOVERLEAF skull + broad thumbs/toes + elbow ankylosis; SEVERE; "
            "  Type 3 (FGFR2): turri/acro + broad thumbs/toes + elbow ankylosis; SEVERE; "
            "  ELBOW ANKYLOSIS: absent in type 1; common in types 2/3 (FGFR2 mutations); "
            "OSTEOGLOPHONIC DYSPLASIA (FGFR1 Y372C/F268C): "
            "  Very rare; CRANIOSYNOSTOSIS + RHIZOMELIC SHORT STATURE + NANISM; "
            "  Prominent supraorbital ridges; choanal atresia; "
            "FGFR1 IN KALLMANN SYNDROME: "
            "  FGFR1 LOF (KAL2): hypogonadotropic hypogonadism + anosmia; "
            "  DIFFERENT from GOF craniosynostosis mutations; "
            "  Does NOT cause craniosynostosis; "
            "encoded 8p11.23; OMIM gene 136350; Pfeiffer #101600"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (GOF) — FGFR1 / PFEIFFER TYPE 1: "
            "  P252R: de novo (~25%) or familial; "
            "  VARIABLE EXPRESSIVITY: mild to moderate phenotype; "
            "  PENETRANCE: near complete for P252R; "
            "  MOLECULAR TESTING: "
            "    FGFR1 P252R hotspot (codon 252) — targeted first; "
            "    Full FGFR1 sequencing if osteoglophonic phenotype; "
            "  SURGICAL: "
            "    Fronto-orbital advancement age 6-12 months for synostosis; "
            "    Digit surgery NOT typically required (no syndactyly); "
            "    Broad thumb may be cosmetically significant; functional usually preserved"
        ),
        "disease_category": (
            "FGFR1-PFEIFFER-TYPE1-P252R-GOF: "
            "  KEY RULE: Pfeiffer TYPE 1 (FGFR1) = mildest Pfeiffer; broad thumbs/toes; NO elbow ankylosis; "
            "  BROAD THUMBS/TOES without syndactyly = Pfeiffer (all types); "
            "  TYPE 1 vs 2/3: type 1 FGFR1-P252R; types 2/3 FGFR2; types 2/3 WORSE (elbow, cloverleaf); "
            "  OSTEOGLOPHONIC DYSPLASIA: same gene (FGFR1), different mutation (Y372C), skeletal dysplasia; "
            "  KALLMANN context: FGFR1 LOF = Kallmann (KAL2); NOT craniosynostosis — CRITICAL DDx"
        ),
    },
    {
        "gene": "FGFR3",
        "protein": (
            "FGFR3 -- 4p16.3 AD-GOF -- 806aa -- Fibroblast-Growth-Factor-Receptor-3-"
            "Muenke-Syndrome-P250R-MOST-COMMON-SINGLE-GENE-Craniosynostosis-"
            "Incomplete-Penetrance-60pct-SNHL-30pct-OMIM-134934"
        ),
        "locus": "4p16.3",
        "protein_size": (
            "806 aa / 88 kDa (FGFR3; same 3-Ig-like domain RTK family; "
            "MUENKE SYNDROME — P250R (c.749C>G): "
            "  THE ONLY PATHOGENIC ALLELE for Muenke syndrome; "
            "  MOST COMMON single-gene cause of craniosynostosis: prevalence ~1:30,000; "
            "  POSITION: D2-D3 linker (analogous to FGFR1 P252R and FGFR2 S252W); "
            "  GOF: broadened FGF ligand specificity; increased signalling; "
            "CLINICAL FEATURES: "
            "  CRANIOSYNOSTOSIS: unicoronal (most common) > bicoronal; "
            "  VARIABLE EXPRESSION: may present as only hearing loss or only intellectual disability; "
            "  INCOMPLETE PENETRANCE: ~40% of P250R carriers have NO craniosynostosis; "
            "  SENSORINEURAL HEARING LOSS (SNHL): 30% of affected individuals; "
            "  MIDFACE HYPOPLASIA: mild; "
            "  MILD INTELLECTUAL DISABILITY: subset; "
            "  CARPAL/TARSAL FUSION: present in some; "
            "CRITICAL DISTINCTION — FGFR3 MUTATION-PHENOTYPE TABLE: "
            "  P250R (c.749C>G): MUENKE SYNDROME — craniosynostosis + SNHL + variable ID; "
            "  G380R (c.1138G>A): ACHONDROPLASIA — short limbs, NO craniosynostosis; "
            "  N540K/K650E: Hypochondroplasia / TD; "
            "  A391E: Crouzon with acanthosis nigricans (with FGFR3, NOT FGFR2 — TRAP); "
            "  SAME GENE, COMPLETELY DIFFERENT MUTATION, COMPLETELY DIFFERENT PHENOTYPE; "
            "SCREENING IMPLICATIONS: "
            "  Standard ACH panel (G380R) will NOT detect Muenke P250R; "
            "  Craniosynostosis panel must specifically include FGFR3 P250R; "
            "  Audiological surveillance for all FGFR3 P250R carriers; "
            "encoded 4p16.3; OMIM gene 134934; Muenke syndrome #602849"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (GOF) — FGFR3 / MUENKE SYNDROME: "
            "  P250R: de novo or familial; "
            "  INCOMPLETE PENETRANCE (~60% show clinical features): "
            "    A carrier parent may appear unaffected but carry P250R; "
            "    FAMILY HISTORY: ask specifically about hearing loss in apparently unaffected parents; "
            "  RECURRENCE: 50% for each child of affected parent; "
            "  DIAGNOSIS: "
            "    Targeted FGFR3 P250R testing; "
            "    Cannot be detected by standard FGFR3 hotspot panel for ACH (different codon); "
            "    Molecular FGFR3 sequencing of codon 250 specific; "
            "HEARING SURVEILLANCE: "
            "  Annual audiometry from diagnosis; "
            "  SNHL: sensorineural (cochlear); requires hearing aids or CI; "
            "  30% develop significant hearing loss — most important non-cranial manifestation"
        ),
        "disease_category": (
            "FGFR3-MUENKE-P250R-MOST-COMMON-SINGLE-GENE-CRANIOSYNOSTOSIS: "
            "  KEY RULE: P250R = MUENKE SYNDROME (MOST COMMON genetic craniosynostosis ~1:30,000); "
            "  INCOMPLETE PENETRANCE 40% — parent may appear normal, still passes on; "
            "  SNHL 30% — annual audiometry mandatory; "
            "  CRITICAL DDx: G380R = achondroplasia (same gene, COMPLETELY different phenotype); "
            "  ACH PANEL misses Muenke — must use craniosynostosis-specific panel; "
            "  A391E = Crouzon+acanthosis nigricans — FGFR3, NOT FGFR2 (common trap)"
        ),
    },
    {
        "gene": "TWIST1",
        "protein": (
            "TWIST1 -- 7p21.1 AD-LOF-Haploinsufficiency -- 202aa -- TWIST-Related-Protein-1-"
            "Saethre-Chotzen-Syndrome-UNICORONAL-PTOSIS-LOW-SET-HAIRLINE-PATHOGNOMONIC-OMIM-601622"
        ),
        "locus": "7p21.1",
        "protein_size": (
            "202 aa / 22 kDa (TWIST1; bHLH transcription factor; forms E-protein heterodimers; "
            "FUNCTION: "
            "  Master regulator of cranial suture development; "
            "  Expressed in cranial mesenchyme → promotes osteoblast differentiation boundary; "
            "  TWIST1 LOF → premature osteoblast differentiation → suture fusion; "
            "  EMT (epithelial-mesenchymal transition) regulator — cancer relevance; "
            "SAETHRE-CHOTZEN SYNDROME: "
            "  PREVALENCE: 1:25,000-50,000; "
            "  CRANIOSYNOSTOSIS: "
            "    UNICORONAL most common → unilateral coronal = FACIAL ASYMMETRY + plagiocephaly; "
            "    Bicoronal: brachycephaly; "
            "    Sagittal/metopic: less common; "
            "  PATHOGNOMONIC TRIAD: "
            "    (1) PTOSIS: upper eyelid ptosis (levator palpebrae involvement); "
            "    (2) LOW-SET HAIRLINE: downward displacement of hairline onto forehead; "
            "    (3) FACIAL ASYMMETRY: from unicoronal synostosis; "
            "  ADDITIONAL: "
            "    Soft-tissue syndactyly fingers 2-3 (skin webbing; NO osseous fusion — DDx Apert); "
            "    Brachydactyly; clinodactyly; "
            "    Ear helix abnormalities (prominent crura); "
            "    Hallux valgus; "
            "BREAST CANCER ASSOCIATION: "
            "  TWIST1 promotes EMT and cancer metastasis; "
            "  TWIST1 germline carriers: elevated breast cancer risk (10% lifetime risk increase); "
            "  Surveillance: mammography from age 35-40; "
            "  This is NOT hereditary breast cancer syndrome but cancer surveillance warranted; "
            "encoded 7p21.1; OMIM gene 601622; disease Saethre-Chotzen #101400"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF (HAPLOINSUFFICIENCY) — TWIST1 / SAETHRE-CHOTZEN: "
            "  75% FAMILIAL; 25% de novo; "
            "  VARIABLE EXPRESSIVITY: "
            "    Range from isolated plagiocephaly to full syndrome; "
            "    Carrier parent may appear mildly affected (low-set hairline alone); "
            "  PENETRANCE: high but variable expression; "
            "  MUTATION TYPES: "
            "    Point mutations (30%): missense/nonsense in bHLH domain; "
            "    Large deletions (30%): chromosomal 7p21.1 del — check for TWIST1 deletion by MLPA; "
            "    Insertions/frameshift (40%); "
            "  CRITICAL: large deletions missed by sequencing alone — add MLPA or array CGH; "
            "CANCER SURVEILLANCE: "
            "  Annual clinical breast exam from age 30; "
            "  Mammography/MRI from age 35-40; "
            "  Discuss risk with genetics/oncology; "
            "MOLECULAR: "
            "  TWIST1 sequencing + deletion analysis (MLPA); "
            "  7p21.1 deletion: may include contiguous gene deletion syndrome"
        ),
        "disease_category": (
            "TWIST1-SAETHRE-CHOTZEN-LOF-HAPLOINSUFFICIENCY: "
            "  KEY RULE: UNICORONAL + PTOSIS + LOW-SET HAIRLINE = Saethre-Chotzen (TWIST1); "
            "  SOFT-TISSUE syndactyly 2-3 fingers (NOT osseous — DDx Apert); "
            "  LARGE DELETION 30% — MLPA required (sequencing alone misses these); "
            "  BREAST CANCER RISK: TWIST1 EMT role → 10% elevated lifetime risk → surveillance age 35; "
            "  FAMILIAL 75% — examine parents for low-set hairline/mild ptosis"
        ),
    },
    {
        "gene": "TCF12",
        "protein": (
            "TCF12 -- 15q21.3 AD-LOF -- 598aa -- Transcription-Factor-12-HEB-"
            "Coronal-Craniosynostosis-ID-35pct-MISSED-BY-OLDER-PANELS-OMIM-600480"
        ),
        "locus": "15q21.3",
        "protein_size": (
            "598 aa / 65 kDa (TCF12; bHLH transcription factor; also called HEB (HeLa E-box binding); "
            "CLASS I bHLH: heterodimerizes with class II bHLH (e.g., TWIST1, HAND2); "
            "FUNCTION: "
            "  TCF12 + TWIST1 heterodimer: critical for cranial suture maintenance; "
            "  TCF12 LOF → suture premature fusion; mechanism analogous to TWIST1 haploinsufficiency; "
            "TCF12 vs TWIST1 CLINICAL COMPARISON: "
            "  SHARED: coronal craniosynostosis (often unicoronal or bicoronal); "
            "  DISTINGUISHING TCF12: "
            "    INTELLECTUAL DISABILITY: 35% in TCF12 (vs rare in TWIST1); "
            "    Less prominent ptosis; Less consistent low-set hairline; "
            "    Less prominent soft-tissue syndactyly; "
            "  DIAGNOSIS GAP: "
            "    Many craniosynostosis panels (pre-2012) did NOT include TCF12; "
            "    TCF12 discovered in 2013 (Sharma et al. Nat Genet); "
            "    Estimated 2-5% of unicoronal craniosynostosis = TCF12 mutations; "
            "    NEGATIVE FGFR2 + NEGATIVE TWIST1 + coronal synostosis → ADD TCF12; "
            "MUTATION SPECTRUM: "
            "  Predominantly truncating (nonsense, frameshift, splice); "
            "  Missense mutations in bHLH domain; "
            "  Penetrance: high but variable expression; "
            "INTELLECTUAL DISABILITY CONTEXT: "
            "  35% of TCF12 patients have cognitive impairment (mild-moderate); "
            "  CRITICAL for parental counselling: TCF12 not 'just cosmetic'; "
            "  Neurodevelopmental surveillance mandatory; "
            "encoded 15q21.3; OMIM gene 600480; coronal craniosynostosis #615193"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — TCF12 / CORONAL CRANIOSYNOSTOSIS: "
            "  De novo and familial; "
            "  VARIABLE PENETRANCE: some carriers may have only subtle features; "
            "  MUTATION TYPES: predominantly truncating; "
            "CLINICAL IMPLICATIONS: "
            "  Must now include TCF12 in all craniosynostosis gene panels; "
            "  NEGATIVE FGFR1/2/3 + NEGATIVE TWIST1 + coronal synostosis → TCF12 sequencing; "
            "  Family cascade testing after proband diagnosis; "
            "FOLLOW-UP: "
            "  Neurodevelopmental assessment all TCF12 patients (35% ID); "
            "  IQ/adaptive behaviour testing age 3-5 years; "
            "  Educational support planning early"
        ),
        "disease_category": (
            "TCF12-CORONAL-CRANIOSYNOSTOSIS-ID-35pct: "
            "  KEY RULE: NEGATIVE FGFR2 + NEGATIVE TWIST1 + coronal synostosis → test TCF12; "
            "  INTELLECTUAL DISABILITY 35% — critical for counselling (not purely cosmetic condition); "
            "  PANEL GAP: pre-2013 panels missed TCF12 — ensure current panel includes it; "
            "  TWIST1/TCF12 HETERODIMER: mechanistically related; clinically can overlap"
        ),
    },
    {
        "gene": "EFNB1",
        "protein": (
            "EFNB1 -- Xq12 X-linked -- 346aa -- Ephrin-B1-"
            "Craniofrontonasal-Syndrome-CFNS-"
            "HETEROZYGOUS-FEMALES-MORE-SEVERE-PARADOX-PATHOGNOMONIC-OMIM-300035"
        ),
        "locus": "Xq12",
        "protein_size": (
            "346 aa / 38 kDa (EFNB1; ephrin-B1; transmembrane ligand for EphB receptors; "
            "PDZ domain binding motif at C-terminus; "
            "FUNCTION: "
            "  Bidirectional Eph-ephrin signalling: EphB forward signalling; ephrin-B reverse signalling; "
            "  Critical for coronal suture development and frontal bone separation; "
            "CRANIOFRONTONASAL SYNDROME (CFNS) — X-LINKED PARADOX: "
            "  HETEROZYGOUS FEMALES: SEVERE phenotype "
            "    — coronal craniosynostosis + hypertelorism + bifid nasal tip + groove of nasal tip; "
            "    — Additional: sloping shoulders, thin curly hair, abnormal nails; "
            "    — BILATERAL coronal synostosis most common; "
            "  HEMIZYGOUS MALES: MILD phenotype "
            "    — hypertelorism only; "
            "    — NO craniosynostosis in most males; "
            "    — Duplicated halluces (bifid big toe); "
            "CELLULAR INTERFERENCE MECHANISM (explains paradox): "
            "  In heterozygous females: X-inactivation creates MOSAIC pattern; "
            "  Some cells express EFNB1 (active X), adjacent cells do not (inactive X); "
            "  BOUNDARY BETWEEN EFNB1+ and EFNB1- cells: creates ectopic EphB signalling; "
            "  This abnormal signalling INDUCES craniosynostosis; "
            "  In hemizygous males: ALL cells uniformly EFNB1 negative → no boundary → no craniosynostosis; "
            "  Analogy: skin cells in Incontinentia Pigmenti (IKBKG) — mosaic damage at boundaries; "
            "PATHOGNOMONIC CLINICAL TRIAD (in females): "
            "  (1) HYPERTELORISM: wide inter-orbital distance; "
            "  (2) CORONAL CRANIOSYNOSTOSIS: orbital asymmetry; "
            "  (3) BIFID NASAL TIP + GROOVED NASAL TIP: midline groove; "
            "encoded Xq12; OMIM gene 300035; CFNS #304110"
        ),
        "inheritance": (
            "X-LINKED — EFNB1 / CRANIOFRONTONASAL SYNDROME: "
            "  FEMALES (heterozygous = CARRIER = AFFECTED — paradox): "
            "    Severe CFNS phenotype (see above); "
            "  MALES (hemizygous): mild phenotype (hypertelorism ± duplicated halluces only); "
            "  INHERITANCE PATTERN: "
            "    Father to all daughters (affected daughters) — but mild in father; "
            "    Mother heterozygous to 50% daughters (affected) + 50% sons (mildly affected); "
            "  FAMILY HISTORY: mild-appearing father + severely affected daughter = CFNS; "
            "  IMPORTANT: do not dismiss mildly affected father as unaffected — examine for hypertelorism; "
            "MOLECULAR: "
            "  EFNB1 sequencing; "
            "  Deletion/duplication analysis; "
            "  X-inactivation studies in females can be informative but not diagnostic"
        ),
        "disease_category": (
            "EFNB1-CFNS-X-LINKED-PARADOX-FEMALES-WORSE: "
            "  KEY RULE: FEMALE heterozygotes SEVERELY affected; MALE hemizygotes mildly affected — PARADOX; "
            "  MECHANISM: cellular interference (X-inactivation mosaicism creates ectopic Eph signalling boundaries); "
            "  PATHOGNOMONIC TRIAD (females): hypertelorism + coronal synostosis + bifid/grooved nasal tip; "
            "  MALES: hypertelorism only + duplicated halluces — DO NOT DISMISS; "
            "  FAMILY: mild father + severely affected daughter = CFNS"
        ),
    },
    {
        "gene": "ERF",
        "protein": (
            "ERF -- 19q13.2 AD-LOF -- 548aa -- ETS2-Repressor-Factor-"
            "Multi-Suture-Craniosynostosis-Crouzon-Like-"
            "CEREBELLAR-TONSILLAR-HERNIATION-40pct-Somatic-Mosaicism-Common-OMIM-611888"
        ),
        "locus": "19q13.2",
        "protein_size": (
            "548 aa / 60 kDa (ERF; ETS2 repressor factor; ETS family transcription factor; "
            "ETS DNA-binding domain; phosphorylation by RAS-MAPK promotes nuclear export; "
            "FUNCTION: "
            "  ERF represses ETS2-driven gene transcription; "
            "  ERF acts downstream of FGFR signalling via RAS-MAPK pathway; "
            "  FGFR2 GOF → MAPK hyperactivation → ERF cytoplasmic sequestration → ERF LOF in nucleus; "
            "  ERF LOF (germline or functionally via upstream hyperactivation): "
            "    Loss of transcriptional repression → craniosynostosis; "
            "ERF CRANIOSYNOSTOSIS — CLINICAL: "
            "  MULTI-SUTURE CRANIOSYNOSTOSIS: coronal + sagittal + lambdoid; "
            "  Crouzon-like facial features WITHOUT FGFR1/2/3 or TWIST1 mutation; "
            "  COMPLEX CRANIOSYNOSTOSIS patients with negative standard panel → ERF; "
            "  CEREBELLAR TONSILLAR HERNIATION (Chiari I type): "
            "    Present in ~40% of ERF patients; "
            "    MANDATORY brain MRI (including posterior fossa); "
            "    Symptomatic Chiari → foramen magnum decompression; "
            "    MRI AT DIAGNOSIS + repeat with growth; "
            "  Intellectual disability: variable (10-30%); "
            "  Hydrocephalus: present in some; "
            "SOMATIC MOSAICISM — CRITICAL DIAGNOSTIC POINT: "
            "  ERF somatic mosaicism common (25-30% of ERF-positive patients); "
            "  LOW-LEVEL MOSAIC ALLELE: may be missed by standard NGS (10% threshold); "
            "  Implications: "
            "    Higher depth sequencing (>500×) needed for ERF; "
            "    Mosaicism in parent → milder/absent phenotype; "
            "    Recurrence risk for mosaic parent: LOW but non-zero (germline mosaicism possible); "
            "encoded 19q13.2; OMIM gene 611888; craniosynostosis 4 #600775"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — ERF / MULTI-SUTURE CRANIOSYNOSTOSIS: "
            "  DE NOVO (most common) or familial; "
            "  SOMATIC MOSAICISM: significant (25-30%); "
            "  RECURRENCE RISK: <1% for de novo germline; low but non-zero for somatic mosaic parent; "
            "DIAGNOSTIC APPROACH: "
            "  Multi-suture craniosynostosis + Crouzon-like + negative FGFR1/2/3/TWIST1/TCF12 → ERF; "
            "  High-sensitivity sequencing (≥500× depth) to detect somatic mosaicism; "
            "MANAGEMENT: "
            "  Brain MRI AT DIAGNOSIS: posterior fossa + look for Chiari I; "
            "  ANNUAL MRI for 5 years if Chiari present or borderline; "
            "  Neurosurgical referral if symptomatic (headache, dysphagia, cervical myelopathy); "
            "  Ophthalmology: papilloedema screening (raised ICP)"
        ),
        "disease_category": (
            "ERF-MULTI-SUTURE-CRANIOSYNOSTOSIS-CHIARI-40pct: "
            "  KEY RULE: MULTI-SUTURE + Crouzon-like + NEGATIVE standard panel → ERF; "
            "  CEREBELLAR TONSILLAR HERNIATION 40% → MRI MANDATORY at diagnosis (posterior fossa); "
            "  SOMATIC MOSAICISM 25-30% → requires HIGH-DEPTH sequencing (>500×) to detect; "
            "  ERF LOF = functional consequence of FGFR/RAS-MAPK hyperactivation → ERF sits in the same pathway"
        ),
    },
    {
        "gene": "RAB23",
        "protein": (
            "RAB23 -- 6p12.1 AR -- 237aa -- RAB23-RAS-Oncogene-Family-"
            "Carpenter-Syndrome-Type1-MULTISUTURE-POLYSYNDACTYLY-Obesity-"
            "CARDIAC-50pct-Intellectual-Disability-ALL-OMIM-606144"
        ),
        "locus": "6p12.1",
        "protein_size": (
            "237 aa / 27 kDa (RAB23; small GTPase RAS superfamily; endosomal trafficking; "
            "FUNCTION: "
            "  RAB23 regulates Hedgehog (HH) signalling pathway via vesicular trafficking; "
            "  RAB23 LOF → HH pathway UPREGULATION; "
            "  Sonic Hedgehog (SHH) pathway critical for limb/craniofacial/cardiac development; "
            "  Polydactyly is classic HH pathway dysregulation finding; "
            "CARPENTER SYNDROME TYPE 1: "
            "  MULTISUTURE CRANIOSYNOSTOSIS: "
            "    Coronal + sagittal + lambdoid → TOWER SKULL; brachycephaly or turricephaly; "
            "    Variable suture involvement; "
            "  PREAXIAL POLYSYNDACTYLY: "
            "    POLYDACTYLY: extra digit (preaxial = thumb/great toe side); "
            "    SYNDACTYLY: adjacent digit fusion (often toes); "
            "    PATHOGNOMONIC COMBINATION in feet; hands variable; "
            "  OBESITY: develops in childhood; central adiposity; "
            "  CARDIAC DEFECTS: 50% of patients; "
            "    VSD (ventricular septal defect): most common; "
            "    PDA (patent ductus arteriosus); "
            "    PS (pulmonary stenosis); "
            "    Complex CHD in some; "
            "    ECHOCARDIOGRAM MANDATORY at diagnosis; "
            "  INTELLECTUAL DISABILITY: present in ALL patients; "
            "    Range mild to moderate; "
            "    Early intervention essential; "
            "  GENITAL ANOMALIES: males — cryptorchidism, hypospadias; "
            "CARPENTER TYPE 2 vs TYPE 1: "
            "  Type 1 (RAB23): classic phenotype above; "
            "  Type 2 (MEGF8): similar but milder ID; situs inversus risk; "
            "  Differentiate by molecular testing; "
            "encoded 6p12.1; OMIM gene 606144; Carpenter syndrome #201000"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — RAB23 / CARPENTER SYNDROME TYPE 1: "
            "  BIALLELIC LOF mutations; "
            "  CONSANGUINITY: elevated; founder mutations in some populations; "
            "  RECURRENCE: 25% per pregnancy; "
            "  PRENATAL DIAGNOSIS: "
            "    Ultrasound: skull shape + limb anomalies (polysyndactyly) detectable second trimester; "
            "    Fetal cardiac echocardiography; "
            "    Molecular: CVS/amniocentesis if both parents carriers known; "
            "MANAGEMENT: "
            "  ECHO at diagnosis: 50% cardiac defects → cardiac surgery may be urgent; "
            "  Craniosynostosis surgery: multi-stage (complex); often total cranial vault remodelling; "
            "  Digit surgery: polysyndactyly correction staged; "
            "  Obesity management from childhood: diet + exercise + metabolic monitoring; "
            "  Neurodevelopmental: all patients ID → IEP, early therapy; "
            "  Endocrinology: insulin resistance monitoring with obesity"
        ),
        "disease_category": (
            "RAB23-CARPENTER-SYNDROME-AR-POLYSYNDACTYLY-CARDIAC-50pct: "
            "  KEY RULE: MULTISUTURE + PREAXIAL POLYSYNDACTYLY + CARDIAC DEFECTS = Carpenter syndrome; "
            "  ECHO MANDATORY: 50% cardiac defects (VSD/PDA/PS) — life-threatening if missed; "
            "  ALL PATIENTS have intellectual disability — critical for counselling; "
            "  AR: consanguinity risk; 25% recurrence; prenatal diagnosis available; "
            "  HH PATHWAY: RAB23 regulates SHH → explains polydactyly (HH pathway hallmark)"
        ),
    },
]


def _generate_patients_for_gene(gene: str, seed: int) -> list:
    """Generate 40 synthetic patients for one craniosynostosis gene."""
    rng = random.Random(seed)

    severity_params = {
        "FGFR2":  {"severe_pct": 0.45, "moderate_pct": 0.40, "mild_pct": 0.15,
                   "age_min": 0, "age_max": 8, "iq_mean": 88, "iq_sd": 15},
        "FGFR1":  {"severe_pct": 0.20, "moderate_pct": 0.50, "mild_pct": 0.30,
                   "age_min": 0, "age_max": 10, "iq_mean": 97, "iq_sd": 10},
        "FGFR3":  {"severe_pct": 0.20, "moderate_pct": 0.40, "mild_pct": 0.40,
                   "age_min": 0, "age_max": 12, "iq_mean": 96, "iq_sd": 12},
        "TWIST1": {"severe_pct": 0.25, "moderate_pct": 0.50, "mild_pct": 0.25,
                   "age_min": 0, "age_max": 10, "iq_mean": 96, "iq_sd": 12},
        "TCF12":  {"severe_pct": 0.25, "moderate_pct": 0.40, "mild_pct": 0.35,
                   "age_min": 0, "age_max": 10, "iq_mean": 89, "iq_sd": 14},
        "EFNB1":  {"severe_pct": 0.35, "moderate_pct": 0.45, "mild_pct": 0.20,
                   "age_min": 0, "age_max": 8, "iq_mean": 95, "iq_sd": 10},
        "ERF":    {"severe_pct": 0.40, "moderate_pct": 0.40, "mild_pct": 0.20,
                   "age_min": 0, "age_max": 12, "iq_mean": 90, "iq_sd": 14},
        "RAB23":  {"severe_pct": 0.60, "moderate_pct": 0.30, "mild_pct": 0.10,
                   "age_min": 0, "age_max": 6, "iq_mean": 65, "iq_sd": 12},
    }
    p = severity_params.get(gene, {"severe_pct": 0.30, "moderate_pct": 0.45, "mild_pct": 0.25,
                                    "age_min": 0, "age_max": 10, "iq_mean": 92, "iq_sd": 12})

    suture_map = {
        "FGFR2":  [["coronal-bicoronal-brachycephaly"], ["coronal-unicoronal-plagiocephaly"],
                   ["multi-suture-turricephaly"], ["lambdoid-posterior-plagiocephaly"]],
        "FGFR1":  [["coronal-unicoronal"], ["coronal-bicoronal"], ["sagittal-scaphocephaly"]],
        "FGFR3":  [["coronal-unicoronal-muenke"], ["coronal-bicoronal-muenke"],
                   ["metopic-trigonocephaly-muenke"]],
        "TWIST1": [["coronal-unicoronal-facial-asymmetry"], ["coronal-bicoronal"],
                   ["sagittal"]],
        "TCF12":  [["coronal-bicoronal"], ["coronal-unicoronal"], ["multi-suture"]],
        "EFNB1":  [["coronal-bilateral-CFNS-hypertelorism"], ["coronal-unilateral-CFNS"],
                   ["frontal-CFNS"]],
        "ERF":    [["multi-suture-coronal-sagittal"], ["multi-suture-all-three"],
                   ["coronal-lambdoid-multi"]],
        "RAB23":  [["multi-suture-coronal-sagittal-lambdoid-tower"],
                   ["multi-suture-complex-carpenter"]],
    }
    suture_options = suture_map.get(gene, [["coronal"]])

    features_map = {
        "FGFR2":  [["mitten-hand-syndactyly-apert", "midface-hypoplasia", "choanal-stenosis"],
                   ["exophthalmos-crouzon", "hypertelorism", "midface-hypoplasia"],
                   ["broad-thumb-toe-pfeiffer", "elbow-ankylosis-pfeiffer", "cloverleaf-skull-pfeiffer2"],
                   ["cutis-gyrata-beare-stevenson", "choanal-atresia"]],
        "FGFR1":  [["broad-thumb", "broad-big-toe", "normal-intellect"],
                   ["broad-thumb", "brachycephaly"]],
        "FGFR3":  [["snhl-sensorineural-hl-30pct", "unicoronal-plagiocephaly"],
                   ["mild-id", "carpal-tarsal-fusion", "unicoronal"],
                   ["hearing-loss-only-incomplete-penetrance"]],
        "TWIST1": [["ptosis-upper-lid", "low-set-hairline", "facial-asymmetry-unicoronal"],
                   ["soft-tissue-syndactyly-2-3-fingers", "ptosis"],
                   ["ear-helix-abnormality", "hallux-valgus"]],
        "TCF12":  [["unicoronal", "normal-intellect"], ["bicoronal", "mild-id"],
                   ["coronal-id-35pct"]],
        "EFNB1":  [["hypertelorism", "coronal-synostosis", "bifid-nasal-tip-grooved-pathognomonic"],
                   ["hypertelorism-male-only", "duplicated-halluces-male"]],
        "ERF":    [["chiari-I-cerebellar-herniation-40pct", "multi-suture"],
                   ["multi-suture-crouzon-like", "raised-icp"],
                   ["somatic-mosaic-milder"]],
        "RAB23":  [["preaxial-polysyndactyly-feet-pathognomonic", "vsd-cardiac",
                    "obesity-childhood", "intellectual-disability-all"],
                   ["multisuture", "pda-cardiac", "cryptorchidism",
                    "obesity", "intellectual-disability-all"]],
    }
    feature_options = features_map.get(gene, [["craniosynostosis"]])

    treatment_map = {
        "FGFR2":  ["fronto-orbital-advancement-FOA-6-12mo", "le-fort-III-distraction",
                   "finger-separation-staged-apert", "VP-shunt-raised-ICP",
                   "total-cranial-vault-remodelling"],
        "FGFR1":  ["fronto-orbital-advancement-6-12mo", "observation-mild",
                   "le-fort-III-midface"],
        "FGFR3":  ["fronto-orbital-advancement", "hearing-aids-SNHL",
                   "cochlear-implant-SNHL", "observation-mild"],
        "TWIST1": ["fronto-orbital-advancement", "le-fort-III-midface",
                   "syndactyly-release-soft-tissue", "breast-cancer-surveillance"],
        "TCF12":  ["fronto-orbital-advancement", "observation-mild",
                   "neurodevelopmental-intervention-ID", "IEP-school-support"],
        "EFNB1":  ["fronto-orbital-advancement-bilateral-females",
                   "orbital-hypertelorism-correction", "rhinoplasty-nasal-tip",
                   "observation-males-mild"],
        "ERF":    ["fronto-orbital-advancement", "foramen-magnum-decompression-chiari",
                   "VP-shunt-hydrocephalus", "total-cranial-vault-remodelling"],
        "RAB23":  ["cardiac-surgery-VSD-PDA", "total-cranial-vault-remodelling-complex",
                   "polysyndactyly-staged-surgery", "obesity-management",
                   "neurodevelopmental-IEP"],
    }
    treatments = treatment_map.get(gene, ["fronto-orbital-advancement"])

    mutation_map = {
        "FGFR2":  ["p.Ser252Trp-Apert", "p.Pro253Arg-Apert", "p.Cys342Arg-Crouzon",
                   "p.Cys342Trp-Pfeiffer", "p.Tyr340Cys-Crouzon", "p.Trp290Cys",
                   "p.Tyr375Cys-Beare-Stevenson"],
        "FGFR1":  ["p.Pro252Arg-Pfeiffer1", "p.Tyr372Cys-Osteoglophonic",
                   "p.Phe268Cys-Osteoglophonic", "p.Ala174Gly"],
        "FGFR3":  ["p.Pro250Arg-Muenke-ONLY-pathogenic-allele", "c.749C>G-P250R"],
        "TWIST1": ["p.Arg118Cys", "p.Gln119Ter", "p.Arg154Cys",
                   "7p21-large-deletion-MLPA", "p.Lys145Glu-bHLH"],
        "TCF12":  ["p.Arg470Ter", "p.Thr304fs", "p.Glu432Ter",
                   "c.1398+1G>A-splice", "p.Leu254Pro"],
        "EFNB1":  ["p.Cys66Ser", "p.Pro54Leu", "p.Trp95Ter",
                   "Xq12-deletion", "p.Arg66Trp"],
        "ERF":    ["p.Leu231Ter", "p.Arg242Ter", "p.Thr248fs",
                   "somatic-mosaic-p.Arg236Ter", "c.693+1G>A-splice"],
        "RAB23":  ["p.Leu145Pro-founder", "p.Ala202Val", "p.Trp166Ter",
                   "p.Arg82Ter", "c.557del-frameshift"],
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
        snhl = gene == "FGFR3" and rng.random() < 0.30
        cardiac = gene == "RAB23" and rng.random() < 0.50
        chiari = gene == "ERF" and rng.random() < 0.40
        raised_icp = rng.random() < (0.40 if severity == "severe" else 0.15)
        treatment = rng.choice(treatments)
        mutation = rng.choice(mutations)
        sutures = rng.choice(suture_options)
        features = rng.choice(feature_options)
        patients.append({
            "id":                   f"{gene}-{seed}-{i+1:03d}",
            "gene":                 gene,
            "age_at_diagnosis_mo":  age_dx * 12 if age_dx < 3 else age_dx,
            "severity":             severity,
            "iq_estimate":          iq,
            "sutures_fused":        sutures,
            "associated_features":  features,
            "snhl":                 snhl,
            "cardiac_defect":       cardiac,
            "chiari_herniation":    chiari,
            "raised_icp":           raised_icp,
            "treatment":            treatment,
            "mutation":             mutation,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Craniosynostosis-Atlas."""
    return {
        "atlas":          "Hereditary-Craniosynostosis-Atlas",
        "subtitle":       (
            "Complete 8-Gene Craniosynostosis & Craniofacial Genetics Atlas "
            "(FGFR2-FGFR1-FGFR3-TWIST1-TCF12-EFNB1-ERF-RAB23)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "FGFR2":  "AD GOF (Crouzon/Apert/Pfeiffer type 2-3/Beare-Stevenson — most common FGFR gene)",
            "FGFR1":  "AD GOF (Pfeiffer type 1, P252R; osteoglophonic dysplasia; broadened FGF specificity)",
            "FGFR3":  "AD GOF (Muenke syndrome P250R ONLY; most common single-gene; incomplete penetrance 60%)",
            "TWIST1": "AD LOF haploinsufficiency (Saethre-Chotzen; unicoronal+ptosis+low-set hairline PATHOGNOMONIC)",
            "TCF12":  "AD LOF (coronal craniosynostosis; ID 35%; MISSED by older panels — must include)",
            "EFNB1":  "X-linked LOF (CFNS; heterozygous FEMALES MORE severely affected than hemizygous males — PARADOX)",
            "ERF":    "AD LOF (multi-suture; Crouzon-like; Chiari herniation 40%; somatic mosaicism 25-30%)",
            "RAB23":  "AR biallelic LOF (Carpenter syndrome; multisuture+polysyndactyly+cardiac 50%+ID all)",
        },
        "key_clinical_rules": [
            "APERT: MITTEN HAND (osseous 2-3-4 syndactyly) PATHOGNOMONIC — no other craniosynostosis has this",
            "CROUZON: NO limb anomaly — key DDx vs Apert and Pfeiffer; all FGFR2",
            "PFEIFFER: BROAD THUMBS/TOES all 3 types; type 2 = cloverleaf skull = MOST SEVERE",
            "MUENKE (FGFR3-P250R): MOST COMMON single-gene craniosynostosis ~1:30,000; SNHL 30%",
            "FGFR3 trap: P250R = Muenke (craniosynostosis); G380R = achondroplasia — SAME GENE, DIFFERENT MUTATION",
            "SAETHRE-CHOTZEN (TWIST1): unicoronal + PTOSIS + LOW-SET HAIRLINE pathognomonic; large deletion 30% → MLPA needed",
            "TCF12: add to all panels; ID 35%; Saethre-Chotzen-like; MISSED by pre-2013 panels",
            "EFNB1 PARADOX: heterozygous FEMALES severely affected; hemizygous MALES mildly affected (cellular interference)",
            "ERF: multi-suture + CHIARI HERNIATION 40% → MRI MANDATORY at diagnosis (posterior fossa)",
            "RAB23 (Carpenter): PREAXIAL POLYSYNDACTYLY + multisuture + CARDIAC DEFECTS 50% → ECHO mandatory",
            "RAISED ICP: all craniosynostosis → annual fundoscopy; papilloedema → urgent neurosurgery",
            "SURGICAL TIMING: fronto-orbital advancement (FOA) 6-12 months standard; complex = staged",
        ],
        "gene_panel_note": (
            "Comprehensive craniosynostosis gene panel (2024): FGFR1, FGFR2, FGFR3, TWIST1, TCF12, EFNB1, ERF, RAB23, "
            "MEGF8, RECQL4, SKI, TGFBR1/2 (Loeys-Dietz), MSX2, ALX3, ALX4, FBN1 (Marfan)"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Craniosynostosis-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)
        severe_n = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n = sum(1 for p in patients if p["severity"] == "mild")
        mean_iq = round(sum(p["iq_estimate"] for p in patients) / n, 1)
        raised_icp_n = sum(1 for p in patients if p["raised_icp"])
        snhl_n = sum(1 for p in patients if p["snhl"])
        cardiac_n = sum(1 for p in patients if p["cardiac_defect"])
        chiari_n = sum(1 for p in patients if p["chiari_herniation"])
        mean_age = round(sum(p["age_at_diagnosis_mo"] for p in patients) / n, 1)
        mutations_seen = list({p["mutation"] for p in patients})
        genes_data.append({
            "gene":              gene,
            "locus":             gene_info["locus"],
            "n_patients":        n,
            "severe_pct":        round(severe_n / n * 100, 1),
            "moderate_pct":      round(moderate_n / n * 100, 1),
            "mild_pct":          round(mild_n / n * 100, 1),
            "mean_iq":           mean_iq,
            "raised_icp_pct":    round(raised_icp_n / n * 100, 1),
            "snhl_pct":          round(snhl_n / n * 100, 1),
            "cardiac_pct":       round(cardiac_n / n * 100, 1),
            "chiari_pct":        round(chiari_n / n * 100, 1),
            "mean_age_dx_mo":    mean_age,
            "sample_mutations":  mutations_seen[:4],
            "protein":           gene_info["protein"],
            "inheritance":       gene_info["inheritance"][:200],
            "disease_category":  gene_info["disease_category"],
        })
    return {
        "atlas": "Hereditary-Craniosynostosis-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Craniosynostosis-Atlas."""
    definitions = [
        {
            "term": "Craniosynostosis — Classification, Suture Patterns, and Diagnostic Approach",
            "genes": ["FGFR2", "FGFR1", "FGFR3", "TWIST1", "TCF12", "EFNB1", "ERF", "RAB23"],
            "definition": (
                "CRANIOSYNOSTOSIS — OVERVIEW: "
                "DEFINITION: premature fusion of one or more cranial sutures; "
                "  Prevalence: 1:2,000-2,500 live births (all forms); "
                "  Genetic cause: 20-25% have identifiable genetic mutation; "
                "  Majority: isolated/nonsyndromic (sagittal most common isolated suture); "
                "SUTURE PATTERNS AND PHENOTYPE: "
                "  SAGITTAL (55%): scaphocephaly (boat-shaped skull); boat-head shape; "
                "  CORONAL (20-25%): bicoronal = brachycephaly; unicoronal = plagiocephaly + facial asymmetry; "
                "  METOPIC (10-15%): trigonocephaly (triangular forehead); "
                "  LAMBDOID (2-4%): posterior plagiocephaly; DDx positional plagiocephaly; "
                "  MULTI-SUTURE: complex craniosynostosis → higher genetic yield → panel testing; "
                "GENETIC YIELD BY PATTERN: "
                "  ISOLATED sagittal: very low genetic yield (<5%); usually sporadic; "
                "  CORONAL (syndromic): FGFR1/2/3, TWIST1, TCF12, EFNB1 → panel; "
                "  MULTI-SUTURE: FGFR2, ERF, RAB23 → panel + MRI; "
                "  ANY SYNDROMIC feature (limb/facial/cardiac): gene panel urgently; "
                "DIAGNOSTIC ALGORITHM: "
                "  Step 1: Clinical phenotyping (limb, face, cardiac, cognitive); "
                "  Step 2: Cranial CT with 3D reconstruction (suture pattern); "
                "  Step 3: Brain MRI (ICP, Chiari, hydrocephalus); "
                "  Step 4: Gene panel (FGFR1/2/3 hotspots → full sequencing → TWIST1/TCF12/EFNB1/ERF/RAB23); "
                "  Step 5: Cardiac echo if syndromic or multi-suture; "
                "  Step 6: Audiometry if FGFR3 mutation or SNHL suspected; "
                "RAISED INTRACRANIAL PRESSURE (ICP): "
                "  Present in 15-20% of single suture; 40-60% of multi-suture; "
                "  Signs: papilloedema, headache, vomiting, irritability; "
                "  Annual fundoscopy; CT/MRI at diagnosis and follow-up; "
                "  Urgent: papilloedema → immediate neurosurgery referral"
            ),
        },
        {
            "term": "FGFR Syndromes — Apert vs Crouzon vs Pfeiffer vs Muenke Differential",
            "genes": ["FGFR2", "FGFR1", "FGFR3"],
            "definition": (
                "FGFR CRANIOSYNOSTOSIS SYNDROME DIFFERENTIAL DIAGNOSIS: "
                "APERT SYNDROME (FGFR2 S252W/P253R): "
                "  MITTEN HAND (OSSEOUS syndactyly 2-3-4): PATHOGNOMONIC — only craniosynostosis with this; "
                "  Bicoronal synostosis; turribrachycephaly; midface hypoplasia; "
                "  Intellectual disability 50%; choanal stenosis; "
                "CROUZON SYNDROME (FGFR2 C342R/W/Y, W290C): "
                "  Craniosynostosis (coronal+sagittal+lambdoid) + exophthalmos + hypertelorism; "
                "  NORMAL LIMBS — no hand/foot anomaly — KEY DDx from Apert and Pfeiffer; "
                "  Normal intelligence; "
                "PFEIFFER SYNDROME (FGFR2 types 2/3; FGFR1 P252R type 1): "
                "  BROAD THUMBS/TOES: present ALL types — PATHOGNOMONIC of Pfeiffer; "
                "  TYPE 1 (FGFR1 P252R): MILDEST; unicoronal/bicoronal; normal intelligence; NO elbow ankylosis; "
                "  TYPE 2 (FGFR2): CLOVERLEAF SKULL PATHOGNOMONIC; elbow ankylosis; SEVERE; "
                "  TYPE 3 (FGFR2): turri/acrocephaly; elbow ankylosis; SEVERE; "
                "  ELBOW ANKYLOSIS distinguishes type 2/3 from type 1; "
                "BEARE-STEVENSON (FGFR2 Y375C, S372C): "
                "  CUTIS GYRATA (ridged corrugated scalp/face skin) PATHOGNOMONIC; "
                "  Choanal atresia; anogenital anomalies; "
                "MUENKE SYNDROME (FGFR3 P250R): "
                "  MOST COMMON SINGLE-GENE craniosynostosis (~1:30,000); "
                "  Unicoronal > bicoronal; SNHL 30%; incomplete penetrance (~60%); "
                "  NORMAL LIMBS; "
                "  TRAP: G380R = achondroplasia (same FGFR3 gene, different mutation); "
                "CROUZON WITH ACANTHOSIS NIGRICANS (FGFR3 A391E): "
                "  Crouzon phenotype BUT mutation in FGFR3, NOT FGFR2 — common trap; "
                "  Acanthosis nigricans (dark velvety skin in skin folds); "
                "SUMMARY TABLE: "
                "  Limb: Apert (mitten hand) > Pfeiffer (broad thumbs) > Muenke/Crouzon (NORMAL); "
                "  Intelligence: Apert (ID 50%) > Muenke/ERF/TCF12 (variable) > Crouzon/Pfeiffer1 (normal); "
                "  Gene: mostly FGFR2; Muenke = FGFR3; Pfeiffer type1 = FGFR1"
            ),
        },
        {
            "term": "EFNB1 Cellular Interference — Why Females Are More Severely Affected",
            "genes": ["EFNB1"],
            "definition": (
                "CRANIOFRONTONASAL SYNDROME — EFNB1 X-LINKED PARADOX: "
                "CLASSICAL X-LINKED EXPECTATION (WRONG FOR EFNB1): "
                "  Expected: hemizygous males severely affected; heterozygous females mildly/not affected; "
                "  ACTUAL: the OPPOSITE — heterozygous FEMALES severely affected; hemizygous MALES mildly affected; "
                "CELLULAR INTERFERENCE MECHANISM: "
                "  X-INACTIVATION IN FEMALES: "
                "    Each cell randomly inactivates one X chromosome; "
                "    Result: mosaic tissue with EFNB1+ cells (active normal X) and EFNB1- cells (active mutant X); "
                "  BOUNDARY EFFECT: "
                "    Adjacent EFNB1+ and EFNB1- cells create a sharp cellular boundary; "
                "    EphB receptor signalling at these boundaries is ABERRANT (ectopic Eph activation); "
                "    This ectopic signalling DISRUPTS cranial suture development → craniosynostosis; "
                "  IN HEMIZYGOUS MALES: "
                "    ALL cells uniformly EFNB1 null (hemizygous); "
                "    NO boundary between EFNB1+ and EFNB1- cells; "
                "    NO ectopic Eph signalling → NO craniosynostosis; "
                "    Only mild features from EFNB1 deficiency itself (hypertelorism); "
                "ANALOGOUS EXAMPLES OF CELLULAR INTERFERENCE: "
                "  Incontinentia pigmenti (IKBKG/NEMO): females mosaic → skin blistering from NEMO+/NEMO- boundaries; "
                "  CFC mouse (Efnb1 knockout): X-inactivation mosaicism creates facial clefts; "
                "CLINICAL PHENOTYPE — FEMALES (heterozygous): "
                "  CORONAL CRANIOSYNOSTOSIS (bilateral); "
                "  HYPERTELORISM (wide inter-orbital distance); "
                "  BIFID/GROOVED NASAL TIP: midline groove or bifid appearance PATHOGNOMONIC triad; "
                "  Sloping shoulders; thin curly hair; abnormal nails; "
                "CLINICAL PHENOTYPE — MALES (hemizygous): "
                "  Mild hypertelorism; "
                "  Duplicated halluces (bifid great toe); "
                "  NO craniosynostosis in most; "
                "FAMILY PATTERN: "
                "  Mild father + severely affected daughter = CFNS (common presentation); "
                "  Do NOT dismiss mildly affected father as unrelated — examine for hypertelorism"
            ),
        },
        {
            "term": "Surgical Management of Craniosynostosis — Timing, Techniques, and Monitoring",
            "genes": ["FGFR2", "FGFR1", "FGFR3", "TWIST1", "TCF12", "EFNB1", "ERF", "RAB23"],
            "definition": (
                "CRANIOSYNOSTOSIS SURGICAL MANAGEMENT: "
                "STANDARD SURGICAL TIMING: "
                "  FRONTO-ORBITAL ADVANCEMENT (FOA): 6-12 months for single/bicoronal; "
                "    Releases fused suture; reshapes frontal bone + orbit; "
                "    Optimal timing: before brain growth peak (12-18 months); "
                "  COMPLEX/MULTI-SUTURE: staged operations; total cranial vault remodelling; "
                "  EMERGENCY: papilloedema → urgent cranial decompression regardless of age; "
                "MIDFACE SURGERY: "
                "  Le Fort III osteotomy or RED (rigid external distraction): age 5-8 years; "
                "  Advances midface (Crouzon/Apert exophthalmos correction); "
                "  Distraction osteogenesis: gradual advancement (~1 mm/day); "
                "APERT HAND SURGERY: "
                "  Staged finger separation: age 12-18 months (border digits first); "
                "  Goal: functional separation of all digits; multiple operations; "
                "PFEIFFER — THUMBS/TOES: "
                "  Broad thumbs usually functional; cosmetic surgery optional; "
                "RAB23 CARPENTER SYNDROME: "
                "  Cardiac surgery first if critical CHD (VSD/PDA); "
                "  Cranial surgery second; polysyndactyly staged; "
                "MONITORING: "
                "  Raised ICP: annual fundoscopy; ophthalmology surveillance; "
                "  Hearing: FGFR3 Muenke → annual audiometry; "
                "  Vision: strabismus/refractive error common in Crouzon/Apert; "
                "  Development: neuropsychological assessment age 3-5; "
                "  Chiari (ERF): annual MRI if herniation present; "
                "MULTIDISCIPLINARY TEAM: "
                "  Craniofacial surgery + neurosurgery + plastics + orthodontics + genetics + ophthalmology"
            ),
        },
        {
            "term": "8-Gene Craniosynostosis Differential Diagnosis Algorithm",
            "genes": ["FGFR2", "FGFR1", "FGFR3", "TWIST1", "TCF12", "EFNB1", "ERF", "RAB23"],
            "definition": (
                "8-GENE CRANIOSYNOSTOSIS DIAGNOSTIC ALGORITHM: "
                "STEP 1: SUTURE PATTERN: "
                "  Isolated sagittal: sporadic; low genetic yield; observation; "
                "  CORONAL (uni/bilateral): → FGFR1/2/3 + TWIST1 + TCF12 + EFNB1; "
                "  MULTI-SUTURE: → full panel including ERF + RAB23; brain MRI; "
                "STEP 2: LIMB ANOMALIES: "
                "  MITTEN HAND (osseous 2-3-4 syndactyly): → FGFR2 Apert (S252W/P253R); "
                "  BROAD THUMBS/TOES: → Pfeiffer (FGFR2 types2/3 or FGFR1 P252R type1); "
                "  PREAXIAL POLYSYNDACTYLY (feet): → RAB23 Carpenter; "
                "  SOFT TISSUE 2-3 finger syndactyly: → TWIST1 Saethre-Chotzen; "
                "  NO limb anomaly: → Crouzon (FGFR2) or Muenke (FGFR3) or TWIST1/TCF12; "
                "STEP 3: ADDITIONAL FEATURES: "
                "  PTOSIS + LOW-SET HAIRLINE: → TWIST1 (Saethre-Chotzen); "
                "  HYPERTELORISM + BIFID NASAL TIP (female): → EFNB1 (CFNS); "
                "  POLYSYNDACTYLY + CARDIAC + OBESITY + ID: → RAB23 (Carpenter); "
                "  CHIARI HERNIATION + multi-suture: → ERF; "
                "  SNHL + unicoronal + variable: → FGFR3 P250R (Muenke); "
                "  CUTIS GYRATA scalp: → FGFR2 Beare-Stevenson; "
                "STEP 4: MOLECULAR TESTING ALGORITHM: "
                "  FGFR2 hotspots (C342, S252, P253) → full FGFR2 → FGFR1 P252R → FGFR3 P250R; "
                "  → TWIST1 sequencing + MLPA (30% large deletion); "
                "  → TCF12 sequencing; "
                "  → EFNB1 (X-linked: females severely affected — test in severe females); "
                "  → ERF (high-depth ≥500× for mosaicism); "
                "  → RAB23 (AR: parental testing, consanguinity risk); "
                "BY INHERITANCE: "
                "  AD GOF: FGFR2 (most), FGFR1, FGFR3; "
                "  AD LOF: TWIST1, TCF12, ERF; "
                "  X-linked: EFNB1; "
                "  AR: RAB23"
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Craniosynostosis-Atlas",
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
              f"iq={g['mean_iq']}, chiari={g['chiari_pct']}%, cardiac={g['cardiac_pct']}%")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Definition entries: {df['count']}")
    for d in df["definitions"]:
        print(f"  {d['term'][:80]}")
