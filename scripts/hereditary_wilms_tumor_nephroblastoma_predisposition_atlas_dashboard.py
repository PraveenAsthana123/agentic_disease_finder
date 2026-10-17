#!/usr/bin/env python3
"""Hereditary-Wilms-Tumor-Nephroblastoma-Predisposition-Atlas -- Complete 8-Gene Reference
Inherited nephroblastoma predispositions: WT1 syndromes, WTX/AMER1, miRNA-biogenesis
(DROSHA/DGCR8), SIX1/SIX2 hotspot mutations, CTR9 familial WT, and CTNNB1 Wnt pathway.

WT1     (Wilms Tumour protein; 485aa; 11p13; AD LOF;
          WAGR=del11p13 (Wilms+Aniridia+GU anomalies+ID);
          Denys-Drash=R394W→DMS nephropathy+WT 90%+pseudohermaphroditism PATHOGNOMONIC;
          Frasier=IVS9 splice→FSGS+gonadoblastoma+WT 5-10%;
          Aniridia in child = WAGR screen PATHOGNOMONIC;
          seed SEED_BASE+0).
AMER1   (APC membrane recruitment protein 1 / WTX; 1135aa; Xq11.2; XLR LOF;
          Males: Wilms tumor (somatic 15-20% sporadic WT); OSCS in carrier females PATHOGNOMONIC;
          OSCS = Osteopathia Striata with Cranial Sclerosis -- striated bone X-ray PATHOGNOMONIC;
          seed SEED_BASE+1).
DROSHA  (Drosha ribonuclease III; 1374aa; 5p13.3; AD LOF;
          miRNA Microprocessor complex (DROSHA+DGCR8);
          E1147K somatic hotspot in 7-12% WT; germline → bilateral WT;
          Bilateral WT + miRNA gene = DROSHA/DGCR8 screen PATHOGNOMONIC;
          seed SEED_BASE+2).
DGCR8   (DiGeorge syndrome critical region gene 8; 773aa; 22q11.2; AD LOF;
          DROSHA Microprocessor partner; R1353H somatic hotspot ~5-8% WT;
          22q11.2 DiGeorge deletion overlap; germline isolated DGCR8 bilateral WT reported;
          seed SEED_BASE+3).
SIX1    (SIX homeobox 1; 284aa; 14q23.1; AD GOF;
          Q177R hotspot mutation 10-12% WT; prevents co-repressor interaction;
          Unfavorable histology (diffuse anaplasia) PATHOGNOMONIC SIX1 Q177R;
          Intensify therapy; avoid de-escalation;
          seed SEED_BASE+4).
SIX2    (SIX homeobox 2; 268aa; 2p21; AD GOF;
          Q177R hotspot 3-5% WT; anaplastic unfavorable histology;
          Earlier developmental arrest than SIX1; germline very rare;
          Parallel Q177R hotspot to SIX1 at cognate residue;
          seed SEED_BASE+5).
CTR9    (CTR9 homolog PAF1 complex subunit; 983aa; 11p15.3; AR LOF;
          PAF1 transcription elongation complex; biallelic LOF → familial WT (no syndrome);
          Perilobar nephrogenic rests PATHOGNOMONIC CTR9/CTNNB1-associated WT;
          Excellent prognosis despite familial predisposition;
          seed SEED_BASE+6).
CTNNB1  (Catenin beta-1; 781aa; 3p22.1; AD GOF;
          Beta-catenin Wnt pathway effector; exon 3 activating mutations;
          Germline GOF → intellectual disability+epilepsy+autism+WT/hepatoblastoma;
          Co-mutant with WT1 LOF classic two-hit WT;
          Perilobar nephrogenic rests (PLNR) enriched;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3710-3717)
"""
import random

SEED_BASE = 3710

ATLAS_GENES = [
    {
        "gene": "WT1",
        "protein": (
            "WT1 -- 11p13 AD LOF -- 485aa -- "
            "Wilms-Tumour-protein-WAGR-del11p13-Denys-Drash-Frasier-"
            "Aniridia-PATHOGNOMONIC-WAGR-screen-"
            "WT1-OMIM-607102"
        ),
        "locus": "11p13",
        "protein_size": (
            "485 aa / ~54 kDa / 11p13 WT1 Wilms Tumour Protein molecular context: "
            "STRUCTURE: "
            "  485 aa / ~54 kDa; zinc-finger transcription factor (4 Cys2-His2 zinc fingers); "
            "  Regulates genes involved in kidney and gonadal development; "
            "  Two major isoforms: WT1(-KTS) and WT1(+KTS) from alternative splicing of exon 9; "
            "  -KTS isoform: transcription factor; +KTS isoform: RNA splicing involvement; "
            "  Tumour suppressor: LOF → loss of G1 cell-cycle control in nephroblasts; "
            "  Constitutional WT1 mutations → syndrome-level nephro-gonadal dysplasia; "
            "THREE CLASSIC WT1 SYNDROMES: "
            "  (1) WAGR syndrome: chromosome 11p13 contiguous deletion (WT1 + PAX6 + other genes); "
            "    W=Wilms tumour (50% lifetime risk); A=Aniridia (PAX6 co-deletion PATHOGNOMONIC); "
            "    G=Genito-urinary anomalies; R=intellectual disabiity (Retardation old term); "
            "    PATHOGNOMONIC: Aniridia in child → WAGR screen with 11p13 FISH/chromosomal microarray; "
            "    WT risk 50%; bilateral WT 20% of WAGR; "
            "  (2) Denys-Drash syndrome: WT1 missense (R394W most common — exon 9 zinc finger); "
            "    Diffuse mesangial sclerosis (DMS): progressive nephropathy → ESRD by 3yr PATHOGNOMONIC; "
            "    46,XY disorder of sexual development (pseudohermaphroditism); "
            "    Wilms tumour risk 90% (highest of all WT1 syndromes); early-onset (<2yr); "
            "    PATHOGNOMONIC: DMS + WT + 46XY DSD = Denys-Drash (WT1 exon 8-9 zinc finger); "
            "  (3) Frasier syndrome: WT1 splice mutation in intron 9 (+KTS isoform lost); "
            "    Focal segmental glomerulosclerosis (FSGS): slowly progressive; "
            "    Gonadoblastoma (dysgerminoma) in streak gonads 40-60% PATHOGNOMONIC; "
            "    46,XY complete gonadal dysgenesis; male pseudohermaphroditism; "
            "    WT risk only 5-10% (lower than Denys-Drash); "
            "NEPHROGENIC RESTS: "
            "  Intralobar nephrogenic rests (ILNR): PATHOGNOMONIC for WT1/WT2-related WT; "
            "  ILNR = anaplastic risk higher; associated with bilateral/synchronous WT; "
            "  PLNR (perilobar rests): associated with CTNNB1/CTR9-related WT; "
            "TREATMENT CONSIDERATIONS: "
            "  Nephron-sparing surgery (NSS): preferred for bilateral WT (preserve function); "
            "  Pre-operative chemotherapy (Children Oncology Group prefers upfront surgery USA; "
            "    SIOP prefers pre-op chemo Europe — both acceptable); "
            "  ESRD in Denys-Drash: transplant after WT treatment; avoid WT1-mutation kidney for donor; "
            "SURVEILLANCE (germline WT1 carriers): "
            "  Abdominal USS every 3 months from birth to age 7; "
            "  Renal function monitoring (GFR, urine protein): annual from diagnosis; "
            "  Gonadal surveillance for Frasier (gonadoblastoma): prophylactic gonadectomy considered in XY; "
            "  Ophthalmology referral: PAX6 co-deletion (WAGR) → glaucoma/cataract monitoring"
        ),
        "syndrome": "WAGR-syndrome / Denys-Drash-syndrome / Frasier-syndrome / AD-LOF / WT1-R394W-Denys-Drash PATHOGNOMONIC / Aniridia-WAGR PATHOGNOMONIC",
        "inheritance": "AD LOF (haploinsufficiency or dominant-negative missense); 11p13 deletion (WAGR); 8-9 zinc-finger mutations (Denys-Drash); intron 9 splice (Frasier); ~10-15% hereditary WT = WT1",
        "bmf_risk": "No BMF; Wilms tumour (nephroblastoma) — solid tumour; ESRD from DMS (Denys-Drash) or FSGS (Frasier); non-hematologic; secondary BMF not expected",
        "pathognomonic": "Aniridia in child = WAGR screen PATHOGNOMONIC (11p13 deletion); DMS nephropathy + WT + 46XY DSD = Denys-Drash PATHOGNOMONIC; FSGS + gonadoblastoma + 46XY = Frasier PATHOGNOMONIC; ILNR on pathology = WT1/WT2-related Wilms",
        "key_rule": "Test WT1 germline in ALL bilateral WT, all WT with aniridia, all WT with DSD, all WT with nephropathy/ESRD; WAGR = 11p13 chromosomal microarray (not single-gene sequencing — contiguous deletion); Denys-Drash R394W = most common single mutation — test exons 8-9 zinc fingers first; prophylactic gonadectomy in Frasier 46XY (gonadoblastoma risk 40-60%)",
        "key_avoid": "Missing WAGR diagnosis (aniridia + WT = screen always); delaying renal surveillance in Denys-Drash (DMS aggressive — ESRD by 3yr); treating Frasier with aggressive chemotherapy alone without addressing FSGS/transplant planning; using WT1-germline-carrier kidney for transplant",
        "surveillance": "Abdominal USS every 3 months birth–age 7; annual renal function (GFR, urine protein); ophthalmology (PAX6/WAGR: glaucoma); gonadal monitoring (Frasier: prophylactic gonadectomy 46XY); cascade family testing (AD)",
        "targeted_rx": "Standard WT chemotherapy (vincristine + actinomycin D ± doxorubicin per COG/SIOP stage); nephron-sparing surgery bilateral WT; renal transplantation (Denys-Drash ESRD post-WT); no currently approved WT1-targeted therapy",
    },
    {
        "gene": "AMER1",
        "protein": (
            "AMER1/WTX -- Xq11.2 XLR LOF -- 1135aa -- "
            "APC-membrane-recruitment-protein-Wnt-antagonist-Wilms-males-15-20pct-somatic-sporadic-"
            "OSCS-females-striated-bone-PATHOGNOMONIC-"
            "AMER1-OMIM-300965"
        ),
        "locus": "Xq11.2",
        "protein_size": (
            "1135 aa / ~123 kDa / Xq11.2 AMER1 (WTX) Molecular context: "
            "STRUCTURE: "
            "  1135 aa / ~123 kDa; tumour suppressor; Wnt pathway antagonist; "
            "  Recruits APC (adenomatous polyposis coli) to the beta-catenin destruction complex; "
            "  Membrane-associated; contains APC-binding domain and PIP2-binding domain; "
            "  LOF → stabilises beta-catenin (same functional result as CTNNB1 GOF); "
            "  Wnt pathway dual suppression: AMER1 + APC both needed for full beta-catenin degradation; "
            "X-LINKED UNUSUAL GENETICS: "
            "  Males (hemizygous): single LOF = complete AMER1 loss → WT predisposition; "
            "  Females (heterozygous): X-inactivation skewing → mosaic expression → OSCS syndrome; "
            "  Complete LOF in female embryos likely lethal (never observed constitutional biallelic); "
            "SOMATIC AMER1 MUTATIONS IN SPORADIC WT: "
            "  Present in 15-20% sporadic (non-hereditary) Wilms tumours; "
            "  Somatic mutation (not germline) in these cases; "
            "  Most sporadic mutations truncating; co-occur with CTNNB1 activating mutations in some; "
            "OSTEOPATHIA STRIATA WITH CRANIAL SCLEROSIS (OSCS): "
            "  Germline heterozygous AMER1 LOF in females → OSCS PATHOGNOMONIC; "
            "  Radiographic: linear bone sclerosis (striations) in long bones PATHOGNOMONIC; "
            "  Cranial sclerosis (dense skull); macrocephaly; cleft palate 15-20%; "
            "  Intellectual disability in some; cardiac septal defects; "
            "  OSCS females: screen for Wilms tumour (risk lower than hemizygous males, but real); "
            "CLINICAL WILMS IN MALES: "
            "  Germline AMER1 LOF in males: VERY RARE germline (most somatic); "
            "  Family history of OSCS in mother/sister = screen male offspring for Wilms; "
            "  No specific syndrome name for isolated male Wilms + AMER1 germline; "
            "TREATMENT: "
            "  Standard WT chemotherapy per stage; "
            "  No AMER1-specific targeted therapy (Wnt pathway active — theoretical MEK/Wnt targets unvalidated); "
            "SURVEILLANCE: "
            "  Abdominal USS every 3 months for males from affected families (AMER1 LOF); "
            "  OSCS females: renal USS (less frequent), orthopaedic monitoring, cardiac echo"
        ),
        "syndrome": "OSCS-Osteopathia-Striata-Cranial-Sclerosis-females PATHOGNOMONIC / XLR-LOF / Wilms-tumour-males / Somatic-15-20pct-sporadic-WT / Wnt-pathway-AMER1-APC-beta-catenin",
        "inheritance": "XLR LOF (hemizygous males); X-linked dominant pattern in females (OSCS phenotype); germline AMER1 LOF extremely rare — most AMER1-mutant WT = somatic; X-inactivation determines female severity",
        "bmf_risk": "No BMF; Wilms tumour (males with germline LOF); OSCS connective/bone phenotype (females); non-hematologic",
        "pathognomonic": "OSCS in female (linear bone striations on X-ray + cranial sclerosis) = AMER1/WTX germline PATHOGNOMONIC; male child with maternal family history of OSCS = screen AMER1 for Wilms",
        "key_rule": "AMER1 somatic mutation (15-20% sporadic WT) is NOT hereditary — distinguish germline from somatic by testing blood DNA separately; OSCS diagnosis in female → screen male siblings/offspring for Wilms; most AMER1 in WT databases = somatic, not germline",
        "key_avoid": "Treating somatic AMER1 WT mutation as hereditary (blood DNA testing required); missing OSCS diagnosis in females (striations on X-ray specific sign); not screening male relatives of OSCS-affected females for Wilms",
        "surveillance": "Males from AMER1 families: abdominal USS every 3 months birth–age 7; females (OSCS): renal USS yearly, orthopaedic monitoring, cardiac echo; cascade family testing (X-linked)",
        "targeted_rx": "Standard WT chemotherapy per stage (vincristine, actinomycin D, doxorubicin); no AMER1-specific targeted therapy; theoretical Wnt pathway targets (XAV939, tankyrase inhibitors) not clinically validated in WT",
    },
    {
        "gene": "DROSHA",
        "protein": (
            "DROSHA -- 5p13.3 AD LOF -- 1374aa -- "
            "Drosha-ribonuclease-III-miRNA-Microprocessor-DROSHA+DGCR8-"
            "E1147K-somatic-hotspot-7-12pct-WT-bilateral-WT-germline-"
            "DROSHA-OMIM-608828"
        ),
        "locus": "5p13.3",
        "protein_size": (
            "1374 aa / ~159 kDa / 5p13.3 DROSHA Molecular context: "
            "STRUCTURE: "
            "  1374 aa / ~159 kDa; ribonuclease III enzyme; nuclear microprocessor complex; "
            "  Forms Microprocessor with DGCR8: DROSHA (catalytic RNase III) + DGCR8 (RNA-binding); "
            "  Cleaves primary miRNA transcripts (pri-miRNA) into precursor miRNAs (pre-miRNA); "
            "  Pre-miRNA exported to cytoplasm where DICER (see Rb atlas) completes miRNA maturation; "
            "  LOF → impaired miRNA biogenesis → failure to downregulate pro-proliferative mRNAs; "
            "  Multiple oncogenes (LIN28, IGF2, HMGA2) normally suppressed by let-7/miR-200 family; "
            "SOMATIC DROSHA MUTATIONS IN WT: "
            "  E1147K: most common somatic hotspot in ribonuclease III domain; "
            "  Present in 7-12% Wilms tumours (somatic); dominant-negative mechanism; "
            "  Co-occurs with SIX1/SIX2 Q177R hotspot mutations in blastema-predominant WT; "
            "  Associated with blastemal histology and unfavorable outcome if >66% blastema; "
            "GERMLINE DROSHA MUTATIONS: "
            "  Heterozygous germline LOF → predisposition to bilateral/multifocal WT; "
            "  No named clinical syndrome for germline DROSHA; "
            "  Bilateral WT + DROSHA/DGCR8 mutation = PATHOGNOMONIC for miRNA-pathway WT; "
            "CLINICAL IMPLICATIONS: "
            "  DROSHA-mutant WT: blastemal predominant histology → classify unfavorable if >66% blastema; "
            "  Increased intensity chemotherapy if blastemal unfavorable histology (doxorubicin/carboplatin); "
            "  Nephrogenic rests: variable — may show both ILNR and PLNR; "
            "DROSHA vs DGCR8 vs DICER1: "
            "  All three = miRNA pathway; DICER1 affects PPB/thyroid/SLCT (DICER1 syndrome); "
            "  DROSHA/DGCR8 = WT-specific miRNA pathway; "
            "  DICER1: two-hit (germline LOF + somatic hotspot); DROSHA: single hit mechanism (dominant); "
            "SURVEILLANCE (germline DROSHA): "
            "  Abdominal USS every 3 months birth–age 8; "
            "  Contralateral kidney surveillance after unilateral WT (bilateral risk elevated)"
        ),
        "syndrome": "Bilateral-WT-miRNA-Microprocessor-DROSHA-AD-LOF / E1147K-Somatic-Hotspot-7-12pct / Blastemal-Predominant-Histology-DROSHA-Mutant / DROSHA-DGCR8-miRNA-Biogenesis-Axis",
        "inheritance": "AD LOF (germline heterozygous predisposition); somatic dominant-negative E1147K hotspot more common than germline; ~7-12% sporadic WT have somatic DROSHA mutation; germline DROSHA rare",
        "bmf_risk": "No BMF; Wilms tumour bilateral/multifocal predisposition; non-hematologic; blastemal-predominant WT higher relapse risk",
        "pathognomonic": "Bilateral WT + miRNA Microprocessor gene mutation (DROSHA or DGCR8) = PATHOGNOMONIC miRNA-pathway Wilms; DROSHA E1147K + SIX1/SIX2 co-mutation = blastemal WT marker (unfavorable if >66% blastema)",
        "key_rule": "DROSHA E1147K somatic mutation ≠ germline predisposition (test blood DNA); germline DROSHA → bilateral WT surveillance; DROSHA-mutant blastemal-predominant WT → unfavorable histology management if >66% blastema; co-mutation SIX1/SIX2 + DROSHA = intensify therapy",
        "key_avoid": "Missing bilateral WT surveillance in germline DROSHA carriers; treating somatic DROSHA E1147K as hereditary without blood DNA confirmation; de-escalating therapy for blastemal-predominant DROSHA-mutant WT",
        "surveillance": "Germline DROSHA: abdominal USS every 3 months birth–age 8; post-treatment contralateral surveillance; cascade family testing",
        "targeted_rx": "Standard WT chemotherapy per stage + histology; blastemal unfavorable (>66%): intensified regimen (carboplatin, etoposide, doxorubicin per SIOP UH-1 or COG AREN0534); no DROSHA-specific targeted therapy",
    },
    {
        "gene": "DGCR8",
        "protein": (
            "DGCR8 -- 22q11.2 AD LOF -- 773aa -- "
            "DiGeorge-critical-region-8-Microprocessor-partner-DROSHA-R1353H-somatic-hotspot-"
            "bilateral-WT-22q11.2-DiGeorge-overlap-"
            "DGCR8-OMIM-609030"
        ),
        "locus": "22q11.2",
        "protein_size": (
            "773 aa / ~86 kDa / 22q11.2 DGCR8 Molecular context: "
            "STRUCTURE: "
            "  773 aa / ~86 kDa; double-stranded RNA-binding protein; Microprocessor complex; "
            "  Partners with DROSHA: DGCR8 recognises pri-miRNA hairpin, DROSHA cleaves; "
            "  Two dsRNA-binding domains (dsRBDs) and an RNA-binding heme domain (Rhed); "
            "  LOF: same functional consequence as DROSHA LOF (impaired pri-miRNA processing); "
            "  Expression in kidney development: high DGCR8 expression in renal progenitors; "
            "22q11.2 GENOMIC CONTEXT: "
            "  DGCR8 located within 22q11.2 DiGeorge critical region; "
            "  Large 22q11.2 deletions (DiGeorge / velocardiofacial syndrome) may encompass DGCR8; "
            "  DiGeorge: conotruncal cardiac defects + thymic hypoplasia + hypocalcemia + palatal anomalies; "
            "  Not all 22q11.2 deletion carriers develop WT (DGCR8 haploinsufficiency insufficient alone?); "
            "SOMATIC DGCR8 IN WT: "
            "  R1353H: somatic hotspot in ~5-8% Wilms tumours; "
            "  Associated with blastemal histology (same as DROSHA); "
            "  Co-mutation with SIX1/SIX2 Q177R possible; "
            "GERMLINE DGCR8 AND WT: "
            "  Isolated heterozygous DGCR8 germline LOF → bilateral WT reported (rare case series); "
            "  22q11.2 deletion syndrome patients have elevated but not high absolute WT risk; "
            "  Most reported Wilms in 22q11.2 deletion = coincidental somatic WT; "
            "DGCR8 vs DROSHA vs DICER1: "
            "  DGCR8 and DROSHA: same Microprocessor step — upstream of DICER1 in miRNA pathway; "
            "  DGCR8/DROSHA: primarily Wilms association; DICER1: pleuropulmonary blastoma + ovary; "
            "  Testing strategy: bilateral WT → sequence DROSHA, DGCR8, DICER1 (miRNA pathway panel); "
            "SURVEILLANCE: "
            "  22q11.2 deletion carriers: baseline renal USS + monitor; "
            "  Isolated germline DGCR8: abdominal USS every 3 months birth–age 8; "
            "  Cardiac assessment: 22q11.2 deletion cardiovascular surveillance"
        ),
        "syndrome": "Bilateral-WT-DGCR8-Microprocessor-AD-LOF / 22q11.2-DiGeorge-overlap / R1353H-Somatic-Hotspot-5-8pct / DGCR8-DROSHA-Parallel-miRNA-Axis",
        "inheritance": "AD LOF (germline heterozygous); 22q11.2 deletion (DiGeorge) may include DGCR8; isolated germline DGCR8 rare; somatic R1353H more common; ~5-8% sporadic WT have somatic DGCR8 mutation",
        "bmf_risk": "No BMF; Wilms tumour (bilateral predisposition germline); 22q11.2-related cardiac anomalies; non-hematologic for isolated DGCR8",
        "pathognomonic": "Bilateral WT + somatic DGCR8 R1353H = miRNA-pathway Wilms (co-mutation DROSHA E1147K or SIX1/SIX2 Q177R); 22q11.2 deletion + WT = test DGCR8 specifically within deletion",
        "key_rule": "Panel testing for bilateral WT: include DROSHA, DGCR8, DICER1 (all miRNA pathway); differentiate 22q11.2 deletion (DGCR8 haploinsufficient) from isolated DGCR8 mutation; R1353H somatic → not hereditary without blood DNA confirmation",
        "key_avoid": "Missing DGCR8 in miRNA-pathway bilateral WT panel; assuming all 22q11.2 deletion carriers have high Wilms risk (risk real but lower than WT1 Denys-Drash); conflating DGCR8 microprocessor function with DICER1 (different step, different tumour spectrum)",
        "surveillance": "Germline DGCR8: abdominal USS every 3 months birth–age 8; 22q11.2 deletion: cardiac echo, thyroid, calcium; cascade family testing",
        "targeted_rx": "Standard WT chemotherapy per stage/histology; no DGCR8-specific therapy; 22q11.2 cardiac: cardiothoracic surgery per defect; blastemal unfavorable WT: intensified (same DROSHA strategy)",
    },
    {
        "gene": "SIX1",
        "protein": (
            "SIX1 -- 14q23.1 AD GOF -- 284aa -- "
            "Sine-oculis-homeobox-1-Q177R-hotspot-10-12pct-WT-"
            "unfavorable-histology-diffuse-anaplasia-PATHOGNOMONIC-"
            "SIX1-OMIM-601205"
        ),
        "locus": "14q23.1",
        "protein_size": (
            "284 aa / ~32 kDa / 14q23.1 SIX1 Molecular context: "
            "STRUCTURE: "
            "  284 aa / ~32 kDa; homeodomain transcription factor; SIX family member; "
            "  Two domains: SIX domain (protein-protein interaction) + homeodomain (DNA binding); "
            "  Normally expressed in metanephric mesenchyme → turned off after kidney development; "
            "  Interacts with co-activator EYA1 and co-repressors DACH1/2; "
            "  Q177R mutation: impairs interaction with co-repressors DACH1/2 → constitutive activation; "
            "  GOF → persistent embryonic gene expression program → failure of nephroblast differentiation; "
            "Q177R HOTSPOT MUTATION: "
            "  The most clinically important SIX1 mutation in Wilms tumour; "
            "  Present in 10-12% of all Wilms tumours (somatic hotspot); "
            "  Germline Q177R reported in ~1-2% familial/bilateral WT cases (very rare germline); "
            "  Glutamine 177 in the SIX domain — co-repressor docking interface; "
            "  Same Q177 residue targeted in SIX2 (parallel evolution of hotspot); "
            "HISTOLOGY AND PROGNOSIS: "
            "  SIX1 Q177R: strongly associated with blastemal-predominant and anaplastic histology; "
            "  Diffuse anaplasia PATHOGNOMONIC poor prognosis marker in SIX1-mutant WT; "
            "  Avoid therapy de-escalation in SIX1 Q177R Wilms; "
            "  Co-mutation with DROSHA E1147K or Q177R DROSHA/SIX: worse prognosis; "
            "RENAL DEVELOPMENT RELEVANCE: "
            "  SIX1 essential for induction of the metanephric mesenchyme by the ureteric bud; "
            "  SIX1 null mice: severe kidney hypoplasia (bilateral renal agenesis in some); "
            "  GOF tumours: re-express SIX1 → embryonic kidney gene program = 'blastemal fate lock'; "
            "TREATMENT STRATEGY: "
            "  SIX1/SIX2 Q177R → blastemal unfavorable histology protocol if >66% blastema; "
            "  Intensified doxorubicin ± carboplatin/etoposide per unfavorable histology arm; "
            "  No approved targeted therapy for SIX1 directly; "
            "SURVEILLANCE (germline SIX1): "
            "  Abdominal USS every 3 months birth–age 7; "
            "  Contralateral surveillance post-unilateral WT (bilateral risk)"
        ),
        "syndrome": "Bilateral-WT-SIX1-Q177R-hotspot-AD-GOF / Blastemal-Predominant-Unfavorable-Histology-PATHOGNOMONIC / Diffuse-Anaplasia-SIX1-Q177R-PATHOGNOMONIC / SIX-family-developmental-transcription-factor",
        "inheritance": "AD GOF (somatic hotspot in 10-12% WT; germline Q177R very rare ~1-2% familial/bilateral WT); de novo germline possible; same Q177R as SIX2 (cognate residue)",
        "bmf_risk": "No BMF; Wilms tumour — blastemal-predominant unfavorable histology; diffuse anaplasia enriched; non-hematologic",
        "pathognomonic": "SIX1 Q177R + Wilms = unfavorable histology marker PATHOGNOMONIC — never de-escalate; SIX1 Q177R + DROSHA E1147K co-mutation = poorest blastemal Wilms subset",
        "key_rule": "SIX1 Q177R = intensify therapy (unfavorable histology protocol) regardless of apparent classic histology; test SIX1 and SIX2 as paired panel (cognate hotspot); germline Q177R rare but possible — test blood DNA in bilateral/familial WT; no de-escalation for SIX1 Q177R",
        "key_avoid": "Treating SIX1 Q177R Wilms as favorable-risk (unfavorable marker); missing SIX2 panel testing when SIX1 negative; de-escalating therapy without SIX1/SIX2 mutation status",
        "surveillance": "Germline SIX1: abdominal USS every 3 months birth–age 7; contralateral surveillance; cascade family testing",
        "targeted_rx": "Standard WT intensified regimen (blastemal unfavorable): carboplatin + etoposide + vincristine + actinomycin D + doxorubicin; no SIX1-specific approved targeted therapy; HDAC inhibitors (experimental — SIX1 repressor axis); vincristine + actinomycin D backbone retained",
    },
    {
        "gene": "SIX2",
        "protein": (
            "SIX2 -- 2p21 AD GOF -- 268aa -- "
            "Sine-oculis-homeobox-2-Q177R-hotspot-3-5pct-WT-"
            "parallel-hotspot-SIX1-anaplastic-unfavorable-"
            "SIX2-OMIM-604994"
        ),
        "locus": "2p21",
        "protein_size": (
            "268 aa / ~30 kDa / 2p21 SIX2 Molecular context: "
            "STRUCTURE: "
            "  268 aa / ~30 kDa; homeodomain transcription factor; SIX family; "
            "  Paralog of SIX1 with ~85% homeodomain identity; "
            "  Q177R: identical residue to SIX1 Q177R — parallel convergent hotspot; "
            "  SIX2 function: maintains nephron progenitor pool (cap mesenchyme self-renewal); "
            "  SIX2 expressed exclusively in cap mesenchyme; SIX1 broader mesenchyme expression; "
            "  GOF Q177R → loss of progenitor differentiation → blastemal fate persistence; "
            "PARALLEL Q177R HOTSPOT TO SIX1: "
            "  Both SIX1 and SIX2 have Q177R hotspot at cognate positions; "
            "  SIX2 Q177R less frequent than SIX1: ~3-5% vs 10-12% of Wilms; "
            "  SIX2 Q177R mechanism: identical (disrupts DACH1/2 co-repressor docking); "
            "  Tests should always include BOTH SIX1 and SIX2; "
            "CLINICAL DISTINCTIONS FROM SIX1: "
            "  SIX2: earlier developmental window (cap mesenchyme commitment); "
            "  SIX2 Q177R: slightly earlier age of onset than SIX1 Q177R (not clearly established); "
            "  SIX2 Q177R tumours: anaplastic/blastemal-predominant; same unfavorable implications; "
            "  SIX2 germline: very rare; case reports only of familial WT; "
            "  SIX2 heterozygous LOF in mouse: failure to maintain nephron progenitors → oligonephronia; "
            "BLASTEMAL SUBTYPE MOLECULAR CLUSTER: "
            "  DROSHA E1147K + SIX1 Q177R + SIX2 Q177R: blastemal WT molecular cluster; "
            "  Relapse-prone if treated on standard chemotherapy; "
            "  Blastemal >66% post-chemotherapy (SIOP) = unfavorable histology; "
            "  Any SIX1/SIX2/DROSHA mutation → heightened blastemal surveillance; "
            "TREATMENT AND SURVEILLANCE: "
            "  Same as SIX1 — intensified WT protocol; "
            "  Germline SIX2: abdominal USS every 3 months birth–age 7; "
            "  Bilateral WT: nephron-sparing surgery preference"
        ),
        "syndrome": "Bilateral-WT-SIX2-Q177R-hotspot-AD-GOF / Blastemal-Unfavorable-Histology / Parallel-Hotspot-SIX1-Cognate-Q177R / Cap-Mesenchyme-Progenitor-GOF",
        "inheritance": "AD GOF (somatic Q177R hotspot ~3-5% WT; germline very rare case reports only); de novo germline possible; parallel to SIX1 Q177R at cognate homeodomain residue",
        "bmf_risk": "No BMF; Wilms tumour — blastemal-predominant unfavorable histology; same implications as SIX1 Q177R; non-hematologic",
        "pathognomonic": "SIX2 Q177R + Wilms = unfavorable blastemal histology (same as SIX1); DROSHA + SIX1 + SIX2 co-mutation = worst blastemal subset; test SIX1 and SIX2 as mandatory paired panel",
        "key_rule": "Always test SIX1 and SIX2 as a pair — SIX2 Q177R less frequent but identical clinical implication as SIX1 Q177R; intensify therapy regardless of apparent conventional histology if Q177R present (either gene); germline SIX2 rare but trigger surveillance if found",
        "key_avoid": "Testing SIX1 alone without SIX2; equating SIX2 lower frequency with lower importance (identical unfavorable marker value); de-escalating therapy for SIX2 Q177R Wilms",
        "surveillance": "Germline SIX2: abdominal USS every 3 months birth–age 7; contralateral surveillance post-unilateral WT; cascade family testing",
        "targeted_rx": "Identical to SIX1: intensified WT regimen for blastemal unfavorable histology; vincristine + actinomycin D + doxorubicin + carboplatin + etoposide; no SIX2-specific approved therapy",
    },
    {
        "gene": "CTR9",
        "protein": (
            "CTR9 -- 11p15.3 AR LOF -- 983aa -- "
            "CTR9-PAF1-complex-subunit-PAF1-elongation-complex-biallelic-LOF-"
            "familial-WT-no-syndrome-PLNR-PATHOGNOMONIC-"
            "CTR9-OMIM-609366"
        ),
        "locus": "11p15.3",
        "protein_size": (
            "983 aa / ~111 kDa / 11p15.3 CTR9 Molecular context: "
            "STRUCTURE: "
            "  983 aa / ~111 kDa; scaffold subunit of the PAF1 transcription elongation complex; "
            "  PAF1 complex members: PAF1, CTR9, CDC73/HRPT2, LEO1, RTF1, WDR61; "
            "  PAF1 complex: recruits H3K4 and H3K36 methyltransferases → transcription elongation; "
            "  CTR9 is scaffold — bridges other PAF1 subunits; "
            "  LOF → dysregulated developmental gene transcription → nephroblast fate error; "
            "FAMILIAL WT WITHOUT NAMED SYNDROME: "
            "  Biallelic (compound heterozygous or homozygous) CTR9 LOF → familial WT; "
            "  No extrenal features — isolated renal predisposition syndrome; "
            "  Autosomal recessive (both copies must be lost); "
            "  Very rare: ~1-2% familial WT kindreds; "
            "  First described: Scott et al, 2012, Nature Genetics; "
            "PERILOBAR NEPHROGENIC RESTS (PLNR): "
            "  CTR9-associated Wilms: perilobar nephrogenic rests (PLNR) characteristic pattern; "
            "  PLNR: rests at periphery/subcapsular region of kidney; "
            "  Histological feature: multiple simultaneous or metachronous WT risk; "
            "  PATHOGNOMONIC distinction: ILNR (intralobar) = WT1/WT2-associated; "
            "    PLNR (perilobar) = CTR9/CTNNB1/CTNNB1+WT1-associated; "
            "  This histological finding should trigger CTR9 + CTNNB1 germline testing; "
            "PAF1 COMPLEX CDC73 (HRPT2) DISTINCTION: "
            "  CDC73/HRPT2 (another PAF1 complex subunit): causes Hyperparathyroidism-Jaw-Tumour; "
            "  CTR9: no parathyroid or jaw tumour phenotype — renal only; "
            "  Do not conflate CTR9 with CDC73/HRPT2 despite shared complex membership; "
            "PROGNOSIS: "
            "  Excellent prognosis despite familial predisposition (low-stage bilateral WT manageable); "
            "  Nephron-sparing surgery preferred for bilateral disease; "
            "  Standard histology (favourable, non-blastemal usually); "
            "SURVEILLANCE: "
            "  Biallelic CTR9 carriers (homozygous/compound het): abdominal USS every 3 months; "
            "  Siblings: test carrier status; "
            "  Obligate heterozygous parents: no WT risk (recessive); test grandchildren if sibling-sibling risk"
        ),
        "syndrome": "Familial-Wilms-Tumour-CTR9-No-Named-Syndrome / AR-LOF-Biallelic / PAF1-Transcription-Elongation-Complex / PLNR-PATHOGNOMONIC-CTR9 / Excellent-Prognosis-Despite-Familial",
        "inheritance": "AR LOF (biallelic required for WT predisposition); obligate heterozygous parents healthy; rare: ~1-2% familial WT kindreds; compound heterozygous or homozygous",
        "bmf_risk": "No BMF; isolated familial Wilms tumour predisposition; non-hematologic; excellent prognosis with treatment",
        "pathognomonic": "PLNR (perilobar nephrogenic rests) on pathology in familial WT = PATHOGNOMONIC → test CTR9 (and CTNNB1) germline; biallelic CTR9 = isolated familial WT, no extra-renal syndrome",
        "key_rule": "CTR9 is recessive — carriers (heterozygous parents) are healthy; PLNR histology + familial WT → CTR9 germline panel; test siblings for biallelic CTR9 when proband identified; CTR9 ≠ CDC73/HRPT2 (both PAF1, different phenotypes); excellent prognosis — nephron-sparing approach",
        "key_avoid": "Missing recessive inheritance (both copies must be mutated — single heterozygous = not at risk); conflating CTR9 with CDC73/HRPT2 (no parathyroid in CTR9); missing PLNR histology trigger for CTR9 testing",
        "surveillance": "Biallelic CTR9 proband: abdominal USS every 3 months birth–age 8; contralateral surveillance; siblings genotyped; cascade autosomal recessive pedigree analysis",
        "targeted_rx": "Standard WT chemotherapy per stage (vincristine + actinomycin D ± doxorubicin); bilateral WT: nephron-sparing surgery; no CTR9-specific targeted therapy; PAF1 complex: no approved CDK8/BRD4 inhibitors in WT",
    },
    {
        "gene": "CTNNB1",
        "protein": (
            "CTNNB1 -- 3p22.1 AD GOF -- 781aa -- "
            "beta-catenin-Wnt-pathway-effector-exon-3-activating-mutations-"
            "germline-GOF-ID+epilepsy+autism+WT-co-mutant-WT1-two-hit-PLNR-"
            "CTNNB1-OMIM-116806"
        ),
        "locus": "3p22.1",
        "protein_size": (
            "781 aa / ~92 kDa / 3p22.1 CTNNB1 Beta-Catenin Molecular context: "
            "STRUCTURE: "
            "  781 aa / ~92 kDa; multifunctional protein: Wnt signalling effector + adherens junction; "
            "  ARM repeat domain: central armadillo repeats (12 repeats) — interact with Wnt transcription; "
            "  N-terminal: phosphorylation domain (GSK3β + CK1 phosphorylation → ubiquitin degradation); "
            "  C-terminal: transcriptional activation domain (TAD); "
            "  Destruction complex: APC + AXIN1/2 + GSK3β + CK1 → phosphorylates exon 3 → βTrCP degradation; "
            "  Wnt signal → disrupts destruction complex → beta-catenin nuclear translocation → TCF/LEF target genes; "
            "EXON 3 ACTIVATING MUTATIONS: "
            "  Exon 3 encodes the phosphorylation degron (S33/S37/T41/S45); "
            "  Missense mutations at these residues: prevent phosphorylation → prevent degradation → GOF; "
            "  S45F/S45P: most common germline-associated mutations; "
            "  Exon 3 deletion (in-frame): seen in somatic hepatoblastoma and WT; "
            "  Somatic CTNNB1 in >15% sporadic Wilms; also somatic in >80% hepatoblastoma; "
            "GERMLINE CTNNB1 GOF PHENOTYPE: "
            "  Neurodevelopmental syndrome: intellectual disability + epilepsy + autism spectrum; "
            "  Hypotonia; facial dysmorphism; "
            "  Cancer risk: Wilms tumour and hepatoblastoma reported in germline GOF; "
            "  NOT the same as APC (FAP): CTNNB1 GOF is separate pathway activation; "
            "    FAP/APC: colorectal polyp risk; CTNNB1 GOF: neurodevelopmental + embryonal tumour; "
            "  APC mutations: also GOF Wnt (via APC LOF); distinguish genetically; "
            "WT1 + CTNNB1 TWO-HIT MODEL: "
            "  Classic Wilms tumour molecular biology: WT1 LOF (1st hit) + CTNNB1 GOF (2nd hit); "
            "  Co-mutation present in ~10-15% of WT1-mutant Wilms tumours; "
            "  PLNR: perilobar nephrogenic rests associated with CTNNB1 WT; "
            "  CTNNB1-mutant Wilms: typically blastemal/stromal mixed — not purely blastemal; "
            "HEPATOBLASTOMA OVERLAP: "
            "  Germline CTNNB1 GOF → hepatoblastoma risk (AFP surveillance): same embryonal tumour risk; "
            "  AFP + liver USS surveillance from birth; "
            "  Distinguish from FAP-associated hepatoblastoma (APC mutation, completely different gene); "
            "TREATMENT AND SURVEILLANCE: "
            "  Standard WT chemotherapy; "
            "  Germline CTNNB1: abdominal USS every 3 months birth–age 7 (Wilms + hepatoblastoma); "
            "  AFP every 3 months birth–age 4 (hepatoblastoma early detection); "
            "  Neurodevelopmental support (ID, epilepsy management); "
            "Wnt PATHWAY TARGETED THERAPY: "
            "  Beta-catenin: no approved direct inhibitor for WT; "
            "  CK1α activators (lenalidomide-related), tankyrase inhibitors: experimental; "
            "  Not validated in paediatric WT; standard chemotherapy backbone applies"
        ),
        "syndrome": "CTNNB1-Neurodevelopmental-Syndrome-AD-GOF / Intellectual-Disability+Epilepsy+Autism / Wilms-Hepatoblastoma-Risk / WT1+CTNNB1-Two-Hit-Classic-WT / PLNR-CTNNB1 / Exon-3-Phosphodegron-Mutations",
        "inheritance": "AD GOF (exon 3 activating mutations); de novo germline most common; somatic CTNNB1 >15% sporadic WT; somatic >80% hepatoblastoma; germline GOF = neurodevelopmental syndrome + embryonal tumour risk",
        "bmf_risk": "No BMF; Wilms tumour + hepatoblastoma embryonal cancer risk; neurodevelopmental syndrome (ID, epilepsy, autism) from germline GOF; non-hematologic",
        "pathognomonic": "PLNR (perilobar nephrogenic rests) on WT pathology = CTNNB1/CTR9 associated; WT1 LOF + CTNNB1 GOF co-mutation = classic two-hit WT molecular model; germline CTNNB1 GOF + ID + epilepsy + Wilms = CTNNB1 neurodevelopmental syndrome",
        "key_rule": "Germline CTNNB1 GOF = screen both Wilms AND hepatoblastoma (AFP + liver USS + abdominal USS); distinguish from APC (FAP): CTNNB1 ≠ colorectal polyp risk; PLNR histology → test CTR9 AND CTNNB1 germline; co-mutation WT1 + CTNNB1 = classic Wilms (common somatic model, rare germline); neurodevelopmental support mandatory for germline CTNNB1",
        "key_avoid": "Conflating CTNNB1 GOF with APC (FAP) — entirely different cancer spectrum; missing hepatoblastoma surveillance in germline CTNNB1 (AFP critical); treating CTNNB1 somatic WT mutation as hereditary without blood DNA confirmation",
        "surveillance": "Germline CTNNB1: AFP every 3 months birth–age 4 (hepatoblastoma); abdominal USS every 3 months birth–age 7 (Wilms + liver); neurology (epilepsy, ID management); cascade family testing (AD de novo mostly)",
        "targeted_rx": "Standard WT chemotherapy per stage; hepatoblastoma: cisplatin ± doxorubicin per stage; no approved Wnt/beta-catenin targeted therapy in WT; tankyrase inhibitors (XAV939) experimental; neurodevelopmental: antiepileptics per epilepsy type",
    },
]

DEFINITIONS = [
    {"term": "WAGR syndrome", "definition": "Contiguous 11p13 deletion encompassing WT1 and PAX6 genes; W=Wilms tumour (50% lifetime risk), A=Aniridia (PAX6 loss PATHOGNOMONIC), G=Genitourinary anomalies, R=intellectual disability (formerly Retardation). Aniridia in any child = mandatory 11p13 chromosomal microarray (WAGR screen)."},
    {"term": "Denys-Drash syndrome", "definition": "WT1 missense mutation (R394W most common, exon 9 zinc finger); triad: diffuse mesangial sclerosis (DMS nephropathy) → ESRD by age 3, 46,XY disorder of sexual development (pseudohermaphroditism), Wilms tumour (90% risk — highest of all WT1 syndromes). DMS + WT = Denys-Drash PATHOGNOMONIC."},
    {"term": "Frasier syndrome", "definition": "WT1 intron 9 splice mutation (+KTS isoform selectively lost); focal segmental glomerulosclerosis (FSGS, slowly progressive), 46,XY complete gonadal dysgenesis (streak gonads), gonadoblastoma risk 40-60% PATHOGNOMONIC. Wilms risk only 5-10% (lower than Denys-Drash). Prophylactic gonadectomy strongly considered for 46,XY Frasier."},
    {"term": "Intralobar vs perilobar nephrogenic rests (ILNR/PLNR)", "definition": "ILNR (intralobar): within the renal parenchyma; PATHOGNOMONIC for WT1/WT2-related Wilms; associated with anaplasia and bilateral WT. PLNR (perilobar): subcapsular/peripheral; associated with CTNNB1/CTR9 and WT2-imprinting disorders. Histological pattern guides germline testing direction."},
    {"term": "OSCS — Osteopathia Striata with Cranial Sclerosis", "definition": "AMER1/WTX heterozygous LOF in females; radiographic: linear striations in long bones + cranial sclerosis PATHOGNOMONIC; macrocephaly; cleft palate 15-20%; cardiac defects. Diagnosis triggers AMER1 testing in male relatives for Wilms tumour. Males hemizygous = WT risk; females heterozygous = OSCS."},
    {"term": "miRNA Microprocessor (DROSHA-DGCR8)", "definition": "Nuclear complex that processes primary miRNA transcripts (pri-miRNA) into precursor miRNAs (pre-miRNA). DROSHA (catalytic) + DGCR8 (recognition). LOF of either: impaired miRNA biogenesis → upregulated LIN28/HMGA2/IGF2 targets → nephroblast proliferation. Bilateral WT + miRNA biogenesis mutation = PATHOGNOMONIC axis."},
    {"term": "SIX1/SIX2 Q177R hotspot", "definition": "Glutamine-177 in the SIX domain of SIX1 (14q23.1) and SIX2 (2p21) — cognate residue at cognate position. GOF mutation prevents co-repressor DACH1/DACH2 docking → constitutive SIX1/SIX2 activation → blastemal fate persistence. Present in 10-12% (SIX1) and 3-5% (SIX2) of Wilms. Unfavorable histology marker — intensify therapy."},
    {"term": "Blastemal predominant Wilms tumour (unfavorable histology)", "definition": "Wilms tumour with >66% viable blastemal component after pre-operative chemotherapy (SIOP protocol). Associated with DROSHA E1147K, SIX1/SIX2 Q177R mutations. Classified as unfavorable histology → intensified chemotherapy (carboplatin, etoposide added). Never de-escalate treatment in blastemal-predominant WT."},
    {"term": "CTR9 PAF1 complex recessive WT", "definition": "Biallelic CTR9 LOF → familial Wilms tumour without any extra-renal syndrome features. PAF1 complex scaffold function. PLNR (perilobar nephrogenic rests) on pathology. Autosomal recessive: obligate heterozygous parents healthy. Very rare (~1-2% familial WT). Excellent prognosis."},
    {"term": "CTNNB1 exon 3 phosphodegron mutations", "definition": "Exon 3 of CTNNB1 encodes S33/S37/T41/S45 phosphorylation sites targeted by GSK3β/CK1 for ubiquitin-mediated degradation. Missense at these sites = GOF: beta-catenin not degraded → nuclear Wnt target gene activation. Somatic in >15% Wilms, >80% hepatoblastoma. Germline GOF: neurodevelopmental syndrome + Wilms/hepatoblastoma risk."},
    {"term": "Nephron-sparing surgery (NSS) in bilateral WT", "definition": "Preservation of functional nephrons in bilateral Wilms tumour by partial nephrectomy instead of total nephrectomy. Goal: avoid dialysis dependency. Pre-operative chemotherapy (SIOP preferred) shrinks tumours → enables NSS. COG: upfront surgery (partial if feasible) + post-surgical chemo. NSS: standard of care for bilateral WT regardless of germline gene."},
    {"term": "WT1-CDK inhibitor limitation", "definition": "Unlike RB1-null tumours where CDK4/6 inhibitors are inactive (no pRb target), WT1-mutant Wilms tumours retain an intact RB pathway. CDK4/6 inhibitors are not routinely used in WT; standard embryonal WT chemotherapy (vincristine-actinomycin D backbone) remains the approved approach regardless of WT1 status."},
    {"term": "Wilms surveillance protocol", "definition": "Standard screening for germline predisposition: abdominal ultrasound every 3 months from birth to age 7 (WT1, SIX1, SIX2, CTNNB1) or age 8 (DROSHA, DGCR8). AFP every 3 months birth–age 4 for CTNNB1 (hepatoblastoma risk). Screening ends at age 7-8 (risk drops sharply after nephrogenesis complete). Annual renal function monitoring continues lifelong."},
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_overview() -> dict:
    cohort_stats = []
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(0.5, rng.gauss(3.5, 2.0)), 1)
        # AMER1: X-linked, males predominantly
        if g["gene"] == "AMER1":
            pct_female = round(rng.uniform(15, 30), 1)
        elif g["gene"] == "CTR9":
            pct_female = round(rng.uniform(45, 55), 1)
        else:
            pct_female = round(rng.uniform(47, 53), 1)
        cohort_stats.append({
            "gene": g["gene"],
            "n": n,
            "mean_age": mean_age,
            "pct_female": pct_female,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
        })

    return {
        "atlas": "Hereditary-Wilms-Tumor-Nephroblastoma-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference — WT1 / AMER1 / DROSHA / DGCR8 / SIX1 / SIX2 / CTR9 / CTNNB1",
        "total_patients": 320,
        "gene_count": 8,
        "seeds": "3710-3717",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "cohort_stats": cohort_stats,
        "clinical_categories": {
            "wt1_syndromes": ["WT1"],
            "wnt_pathway": ["AMER1", "CTNNB1"],
            "mirna_biogenesis": ["DROSHA", "DGCR8"],
            "six_family_hotspot": ["SIX1", "SIX2"],
            "paf1_complex": ["CTR9"],
        },
        "key_clinical_rules": [
            "WT1: aniridia + child = WAGR screen PATHOGNOMONIC (11p13 microarray); bilateral WT 20%",
            "AMER1: OSCS in female = AMER1 germline PATHOGNOMONIC; screen male relatives for Wilms",
            "DROSHA: E1147K somatic hotspot 7-12% WT; germline → bilateral WT; blastemal unfavorable",
            "DGCR8: R1353H somatic 5-8% WT; 22q11.2 deletion overlap; miRNA axis parallel to DROSHA",
            "SIX1: Q177R hotspot 10-12% WT = unfavorable histology PATHOGNOMONIC — never de-escalate",
            "SIX2: Q177R hotspot 3-5% WT = same unfavorable implication as SIX1; test as paired panel",
            "CTR9: AR recessive; PLNR on pathology PATHOGNOMONIC; excellent prognosis; parents healthy",
            "CTNNB1: germline GOF = ID+epilepsy+WT+hepatoblastoma; AFP + USS birth–age 4 mandatory",
        ],
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(0.5, rng.gauss(3.5, 2.0)), 1)
        if g["gene"] == "AMER1":
            pct_f = round(rng.uniform(15, 30), 1)
        else:
            pct_f = round(rng.uniform(47, 53), 1)
        bilateral_n = rng.randint(12, 28) if g["gene"] in ("WT1", "DROSHA", "DGCR8") else rng.randint(3, 12)
        met_n = rng.randint(3, 12)
        # CTR9 excellent prognosis
        second_primary_n = rng.randint(1, 4) if g["gene"] == "CTR9" else rng.randint(2, 10)
        gene_data[g["gene"]] = {
            "n": n,
            "mean_age": mean_age,
            "pct_female": pct_f,
            "bilateral_n": bilateral_n,
            "metastatic_n": met_n,
            "second_primary_n": second_primary_n,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
        }

    return {
        "atlas": "Hereditary-Wilms-Tumor-Nephroblastoma-Predisposition-Atlas",
        "total_patients": 320,
        "seeds": "3710-3717",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "summary": {
            "wt1_syndrome_patients": 40,
            "amer1_patients": 40,
            "mirna_biogenesis_patients": 80,
            "six_family_patients": 80,
            "paf1_complex_patients": 40,
            "ctnnb1_wnt_patients": 40,
            "bilateral_wt_genes": ["WT1", "DROSHA", "DGCR8"],
            "unfavorable_histology_genes": ["SIX1", "SIX2", "DROSHA"],
            "surveillance_afp_required": ["CTNNB1"],
            "radiation_considerations": ["WT1-no-specific-CI", "all-standard-WT-protocol"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Wilms-Tumor-Nephroblastoma-Predisposition-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "WT1-ANIRIDIA-WAGR-PATHOGNOMONIC": "Aniridia in child = WAGR 11p13 microarray MANDATORY — never assume isolated aniridia without Wilms surveillance",
            "WT1-DMS-DENYS-DRASH-PATHOGNOMONIC": "Diffuse mesangial sclerosis + WT + 46XY DSD = Denys-Drash (R394W exon 9 zinc finger); transplant planning from diagnosis",
            "WT1-FRASIER-GONADOBLASTOMA": "Frasier IVS9 splice → streak gonads → gonadoblastoma 40-60%; prophylactic gonadectomy in 46,XY Frasier STRONGLY considered",
            "AMER1-OSCS-MALES-SCREEN": "OSCS diagnosis in female = screen male relatives for Wilms (AMER1 germline); striations on X-ray long bones = PATHOGNOMONIC OSCS",
            "DROSHA-SIX1-BLASTEMAL-UNFAVORABLE": "DROSHA E1147K or SIX1/SIX2 Q177R = blastemal unfavorable histology — intensify chemotherapy; NEVER de-escalate",
            "SIX1-SIX2-PAIRED-PANEL": "Always test SIX1 AND SIX2 as paired panel — both Q177R hotspot = identical unfavorable marker; testing one alone insufficient",
            "CTR9-PLNR-RECESSIVE": "PLNR on WT pathology → test CTR9 AND CTNNB1 germline; CTR9 = autosomal RECESSIVE (biallelic needed); parents healthy",
            "CTNNB1-AFP-HEPATOBLASTOMA": "Germline CTNNB1 GOF = AFP every 3 months birth–age 4 (hepatoblastoma risk) + abdominal USS for Wilms; NOT FAP (no colorectal risk)",
            "BILATERAL-WT-NSS-PRIORITY": "Bilateral Wilms (any gene) = nephron-sparing surgery priority; pre-op chemotherapy (SIOP) shrinks tumours; avoid total bilateral nephrectomy",
            "MIRNA-PANEL-BILATERAL": "Bilateral WT + no WT1/CTNNB1/CTR9 → screen DROSHA + DGCR8 + DICER1 (complete miRNA pathway panel); single gene testing insufficient",
        },
    }
