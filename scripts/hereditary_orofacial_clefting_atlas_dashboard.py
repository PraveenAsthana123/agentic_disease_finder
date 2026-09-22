#!/usr/bin/env python3
"""Hereditary-Orofacial-Clefting-Atlas — Complete 8-Gene Orofacial Clefting & Cleft Lip/Palate Genetics Atlas
IRF6    (interferon regulatory factor 6; 467 aa; 1q32.3; AD LOF;
         Van der Woude Syndrome VWS1 — OMIM #119300;
         most common hereditary CL/P gene; LIP PITS PATHOGNOMONIC;
         Popliteal Pterygium Syndrome (PPS) allelic; p.Arg84His/Cys founder;
         seed SEED_BASE+0) ·
GRHL3   (grainyhead-like transcription factor 3; 624 aa; 1p36.11; AD LOF;
         Van der Woude Syndrome type 2 VWS2 — OMIM #616653;
         also causes PVRL1-negative VWS; nonsense/frameshift LOF;
         seed SEED_BASE+1) ·
PVRL1   (nectin cell adhesion molecule 1 / NECTIN1; 517 aa; 11q23.3; AR;
         Clefting-Ectodermal Dysplasia Syndrome CLPED1 — OMIM #225060;
         CL/P + hypodontia + skin anomalies;
         Mediterranean founder c.296T>G (p.Val99Gly);
         seed SEED_BASE+2) ·
MSX1    (msh homeobox 1; 303 aa; 4p16.2; AD LOF;
         Orofacial Clefting type 5 OFC5 — OMIM #608874;
         TOOTH AGENESIS #2 + CL/P; p.Arg31Pro founder;
         seed SEED_BASE+3) ·
TBX22   (T-box transcription factor 22; 520 aa; Xq21.1; XLR;
         X-linked cleft palate with ankyloglossia CPX — OMIM #303400;
         ANKYLOGLOSSIA + cleft palate PATHOGNOMONIC combination;
         males fully affected, females mosaic; p.His213Asp;
         seed SEED_BASE+4) ·
SATB2   (special AT-rich binding protein 2; 733 aa; 2q33.1; AD de novo;
         SATB2-associated syndrome / Glass syndrome — OMIM #612313;
         severe intellectual disability + ABSENT speech + CL/P + behavioral;
         dental anomalies; de novo dominant;
         seed SEED_BASE+5) ·
TP63    (tumor protein p63; 680 aa; 3q28; AD GOF/DN;
         EEC syndrome Ectrodactyly-Ectodermal Dysplasia-Clefting — OMIM #604292;
         ECTRODACTYLY + ectodermal dysplasia + CL/P TRIAD PATHOGNOMONIC;
         also RHS Rapp-Hodgkin, AEC, LMS; allele-specific;
         seed SEED_BASE+6) ·
COL11A1 (collagen type XI alpha 1 chain; 1806 aa; 1p21.1; AD;
         Marshall syndrome / Stickler syndrome type 2 — OMIM #154780;
         CLEFT PALATE + HIGH MYOPIA + SNHL triad;
         Marshall: flat midface + large eye globes;
         COL11A1 vs COL2A1 Stickler 1 distinction;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 3022-3029)
"""
import random

SEED_BASE = 3022

ATLAS_GENES = [
    {
        "gene": "IRF6",
        "protein": (
            "IRF6 -- 1q32.3 AD-LOF -- 467aa -- Interferon-Regulatory-Factor-6-"
            "Van-der-Woude-Syndrome-VWS1-LIP-PITS-PATHOGNOMONIC-"
            "Most-Common-Hereditary-CLP-Gene-OMIM-607199"
        ),
        "locus": "1q32.3",
        "protein_size": (
            "467 aa / 53 kDa (IRF6; interferon regulatory factor 6; "
            "STRUCTURE: N-terminal DNA-binding domain (DBD) — HTH motif binding IFN stimulated response elements; "
            "C-terminal IRF association domain (IAD) — protein-protein interactions; "
            "FUNCTION: "
            "  IRF6 master regulator of keratinocyte proliferation and differentiation; "
            "  Critical for oral epithelium fusion during palate closure; "
            "  IRF6 activates GRHL3 in epithelial differentiation cascade; "
            "  IRF6 LOF → failure of oral epithelial differentiation → periderm persistence → clefting; "
            "VAN DER WOUDE SYNDROME TYPE 1 (VWS1): "
            "  OMIM #119300; prevalence 1:30,000-1:100,000; "
            "  MOST COMMON HEREDITARY CAUSE of cleft lip/palate (CL/P); "
            "  ~2% of all CL/P cases; "
            "  LIP PITS (paramedian lower lip pits): "
            "    PATHOGNOMONIC for VWS — bilateral symmetric pits on lower lip vermillion; "
            "    Can be subtle: minor depressions or blind-ending sinuses; "
            "    May be present without clefting in some family members; "
            "    LIP PITS alone = carrier; LIP PITS + CL/P = full VWS; "
            "  CLEFTING SPECTRUM: "
            "    CL only; CP only; CL+CP; bifid uvula; submucous cleft; "
            "    INTRA-FAMILY VARIABILITY: one member CL/P, another lip pits only; "
            "POPLITEAL PTERYGIUM SYNDROME (PPS): "
            "  ALLELIC to VWS (same IRF6 gene; different mutations); "
            "  Additional features: POPLITEAL WEBS (skin folds behind knees); "
            "    Intercrural pterygia; genital anomalies (cleft scrotum, labial hypoplasia); "
            "    Ankyloblepharon; eyelid fissures; "
            "  PPS mutations: mostly missense in DNA-binding domain; "
            "  VWS mutations: more mixed (missense, nonsense, frameshift); "
            "IRF6 MUTATIONS: "
            "  ~200 pathogenic variants documented; "
            "  HOT SPOTS: p.Arg84His, p.Arg84Cys (FOUNDER MUTATIONS — recurrent); "
            "  Mutation type: ~50% missense in DBD/IAD; ~30% nonsense/frameshift; "
            "  Large deletions: rare but possible; "
            "encoded 1q32.3; OMIM gene 607199; VWS1 #119300; PPS #119500"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF (HAPLOINSUFFICIENCY) — IRF6 / VAN DER WOUDE / PPS: "
            "  HIGH PENETRANCE (~96%) but VARIABLE EXPRESSIVITY; "
            "  INTRA-FAMILY VARIABILITY IS THE RULE: "
            "    Same mutation → one member full CL/P; another member lip pits only; "
            "    Cannot predict which phenotype in offspring; "
            "  DE NOVO: ~25% (sporadic new VWS); "
            "  FAMILIAL: ~75%; "
            "GENOTYPE-PHENOTYPE: "
            "  p.Arg84His/Cys: most commonly VWS; "
            "  Missense in DBD hotspot residues → higher PPS risk; "
            "  Full LOF (nonsense/frameshift) → VWS more common than PPS; "
            "CLINICAL RECOGNITION OF VWS: "
            "  Examine lower lip carefully in all CL/P families; "
            "  Lip pits may be tiny; may be unilateral; may be scarred from prior surgery; "
            "  Presence of lip pits in ANY family member = VWS until proven otherwise; "
            "SURVEILLANCE: "
            "  Cascade testing: test all first-degree relatives; "
            "  No cancer surveillance required (unlike TP63); "
            "  Dental: monitor for hypodontia (overlap with MSX1/PVRL1 phenotype); "
            "SURGICAL MANAGEMENT: "
            "  Cleft lip repair: 3-6 months; "
            "  Palate repair: 12-18 months; "
            "  Lip pit excision: if symptomatic (mucus discharge) — age 5+ years"
        ),
        "disease_category": (
            "IRF6-VWS1-LIP-PITS-PATHOGNOMONIC-MOST-COMMON-HEREDITARY-CLP: "
            "  KEY RULE: LIP PITS on lower lip = Van der Woude Syndrome (IRF6 or GRHL3) UNTIL PROVEN OTHERWISE; "
            "  IRF6 = MOST COMMON hereditary CL/P gene (~2% of all CL/P); "
            "  VARIABLE EXPRESSIVITY: same mutation → lip pits only or full CL/P in same family; "
            "  PPS ALLELIC: popliteal webs + genital anomalies + clefting = IRF6 missense in DBD; "
            "  p.Arg84His/Cys: RECURRENT FOUNDER MUTATIONS — most common IRF6 pathogenic variants"
        ),
    },
    {
        "gene": "GRHL3",
        "protein": (
            "GRHL3 -- 1p36.11 AD-LOF -- 624aa -- Grainyhead-Like-Transcription-Factor-3-"
            "Van-der-Woude-Syndrome-Type2-VWS2-PVRL1-Negative-VWS-OMIM-608317"
        ),
        "locus": "1p36.11",
        "protein_size": (
            "624 aa / 70 kDa (GRHL3; grainyhead-like transcription factor 3; "
            "Drosophila grainyhead homolog; "
            "STRUCTURE: N-terminal transcription activation domain; "
            "DNA-binding domain (CP2/LSF/GRH superfamily); "
            "dimerisation domain; "
            "FUNCTION: "
            "  GRHL3 regulates keratinocyte differentiation and epidermal barrier formation; "
            "  GRHL3 acts DOWNSTREAM of IRF6 in the oral epithelium fusion pathway; "
            "  IRF6 → activates GRHL3 → drives periderm differentiation → palate fusion; "
            "  GRHL3 LOF → same mechanistic consequence as IRF6 LOF → VWS phenotype; "
            "VAN DER WOUDE SYNDROME TYPE 2 (VWS2): "
            "  OMIM #616653; identified 2013 (Peyrard-Janvid et al.); "
            "  Clinically IDENTICAL to VWS1 (IRF6): "
            "    Lip pits PATHOGNOMONIC; "
            "    CL/P spectrum; variable expressivity; "
            "  PVRL1-NEGATIVE VWS: "
            "    GRHL3 also explains cases of VWS where PVRL1/NECTIN1 testing is negative; "
            "    Key for families with VWS2 phenotype (lip pits + cleft) but no IRF6 mutation; "
            "    After IRF6 negative result → test GRHL3; "
            "  ISOLATED CLEFT PALATE RISK: "
            "    GRHL3 mutations also found in isolated CP (without lip pits); "
            "    CPO (cleft palate only) cohorts: GRHL3 rare but real cause; "
            "MUTATION SPECTRUM: "
            "  Predominantly LOF: nonsense, frameshift, splice-site mutations; "
            "  Haploinsufficiency mechanism (similar to IRF6); "
            "  MISSENSE mutations: rare; require functional validation; "
            "  Large deletions: small 1p36 microdeletion can include GRHL3; "
            "GRHL FAMILY: "
            "  GRHL1: oral clefting in mice; rare human variants; "
            "  GRHL2: cleft palate in mice; human associations; "
            "  GRHL3: only confirmed human VWS gene in this family; "
            "encoded 1p36.11; OMIM gene 608317; VWS2 #616653"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF (HAPLOINSUFFICIENCY) — GRHL3 / VWS2: "
            "  HIGH PENETRANCE with VARIABLE EXPRESSIVITY (same as VWS1/IRF6); "
            "  DE NOVO and FAMILIAL; "
            "  TESTING STRATEGY: "
            "    VWS diagnosis (lip pits + cleft): test IRF6 FIRST (more common); "
            "    IRF6 NEGATIVE + VWS phenotype → test GRHL3 SECOND; "
            "    ~5% of VWS = GRHL3 mutations; "
            "    IRF6 + GRHL3 together account for majority of genetically-solved VWS; "
            "CLINICAL MANAGEMENT: "
            "  Identical to VWS1/IRF6: examine family; cascade test; "
            "  No additional surveillance beyond VWS1; "
            "  Lip pit excision and cleft repair as for VWS1; "
            "GENETIC COUNSELLING: "
            "  50% recurrence risk for offspring of affected parent; "
            "  Variable expressivity counselling identical to IRF6/VWS1; "
            "  Clinical panel: must include GRHL3 alongside IRF6 in orofacial clefting workup"
        ),
        "disease_category": (
            "GRHL3-VWS2-LOF-DOWNSTREAM-OF-IRF6-PATHWAY: "
            "  KEY RULE: IRF6 NEGATIVE + LIP PITS + CL/P → test GRHL3 (VWS2); "
            "  CLINICALLY IDENTICAL to VWS1 (IRF6): lip pits PATHOGNOMONIC; variable expressivity; "
            "  GRHL3 acts downstream of IRF6 in same periderm-differentiation pathway; "
            "  Also causes isolated CP (without lip pits) in rare cases; "
            "  PANEL NOTE: CL/P gene panel must include both IRF6 and GRHL3 for VWS workup"
        ),
    },
    {
        "gene": "PVRL1",
        "protein": (
            "PVRL1 -- 11q23.3 AR -- 517aa -- Nectin-Cell-Adhesion-Molecule-1-NECTIN1-"
            "CLPED1-Clefting-Ectodermal-Dysplasia-CLP-Hypodontia-"
            "Mediterranean-Founder-c296T>G-OMIM-600644"
        ),
        "locus": "11q23.3",
        "protein_size": (
            "517 aa / 55 kDa (PVRL1 / NECTIN1; poliovirus receptor-related 1; nectin-1; "
            "STRUCTURE: "
            "  Extracellular: 3 immunoglobulin-like domains (V-C2-C2); "
            "  Transmembrane domain; "
            "  Intracellular: PDZ-binding motif (interacts with afadin/AF-6); "
            "FUNCTION: "
            "  Cell-cell adhesion via nectin-afadin system; "
            "  Expressed in developing ectodermal tissue, palatal shelves, tooth germs; "
            "  PVRL1 homotypic (nectin1-nectin1) and heterotypic (nectin1-nectin3) trans-binding; "
            "  Critical for epithelial adhesion during palatal fusion; "
            "  Also expressed in peripheral nerve Schwann cells (herpes simplex virus receptor); "
            "CLPED1 — CLEFTING-ECTODERMAL DYSPLASIA SYNDROME (CLPED1): "
            "  OMIM #225060; "
            "  TRIAD: "
            "    (1) CLEFT LIP/PALATE: bilateral or unilateral CL/P; "
            "    (2) HYPODONTIA: missing teeth (often maxillary lateral incisors, second premolars); "
            "    (3) ECTODERMAL ANOMALIES: "
            "        Skin: palmar/plantar keratoderma; "
            "        Hair: fine, sparse, or absent in areas; "
            "        Nails: dysplastic nails; "
            "  COGNITIVE: NORMAL intelligence (unlike SATB2 or TP63 AEC/RHS); "
            "MEDITERRANEAN FOUNDER MUTATION: "
            "  c.296T>G (p.Val99Gly): "
            "    FOUNDER mutation common in Mediterranean populations (Italy, Greece, Lebanon, Turkey); "
            "    Frequent in CONSANGUINEOUS families from these regions; "
            "    Homozygous: full CLPED1 phenotype; "
            "    Heterozygous: rare; generally unaffected (AR inheritance); "
            "  Second Mediterranean founder: c.880C>T (p.Arg294Ter) in Israelis; "
            "PVRL1/NECTIN1 IN SIMPLEX HERPES: "
            "  NECTIN1 is HSV entry receptor; "
            "  PVRL1 null mice: resistant to HSV corneal infection; "
            "  Clinically: CLPED1 patients do not have increased susceptibility to HSV (compensatory); "
            "encoded 11q23.3; OMIM gene 600644; CLPED1 #225060"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — PVRL1/NECTIN1 / CLPED1: "
            "  BIALLELIC LOF mutations required for full CLPED1; "
            "  CONSANGUINITY: elevated in Mediterranean founders; "
            "  HETEROZYGOUS CARRIERS: generally asymptomatic; "
            "    Some reports of mild dental anomalies in carriers — need monitoring; "
            "  RECURRENCE: 25% per pregnancy; "
            "MUTATION TYPES: "
            "  Missense (especially founder p.Val99Gly); nonsense; frameshift; "
            "  Homozygous founder: consanguineous Mediterranean families; "
            "  Compound heterozygous: possible in outbred populations; "
            "DIAGNOSTIC APPROACH: "
            "  CL/P + hypodontia + ectodermal signs → test PVRL1 (especially Mediterranean origin); "
            "  Molecular confirmation: PVRL1 sequencing ± deletion analysis; "
            "  Parental carrier testing after proband identification; "
            "MANAGEMENT: "
            "  Cleft repair: standard protocol; "
            "  Dental restoration: prosthetic teeth for hypodontia (implants in adulthood); "
            "  Skin: emollients for keratoderma; "
            "  Ophthalmology: annual (corneal evaluation — NECTIN1 expressed in cornea); "
            "  Normal schooling expected (intelligence preserved)"
        ),
        "disease_category": (
            "PVRL1-NECTIN1-CLPED1-AR-CLP-HYPODONTIA-ECTODERMAL: "
            "  KEY RULE: CL/P + HYPODONTIA + ECTODERMAL SIGNS (keratoderma/nail dysplasia) = CLPED1 (PVRL1); "
            "  AR: must identify BIALLELIC mutations; consanguinity risk elevated; "
            "  MEDITERRANEAN FOUNDER p.Val99Gly: most common pathogenic variant worldwide; "
            "  NORMAL INTELLIGENCE: key DDx vs SATB2 (severe ID) and TP63-AEC (variable ID); "
            "  NECTIN1 = HSV entry receptor — not clinically significant for HSV susceptibility in CLPED1"
        ),
    },
    {
        "gene": "MSX1",
        "protein": (
            "MSX1 -- 4p16.2 AD-LOF -- 303aa -- MSH-Homeobox-1-"
            "Orofacial-Clefting-Type5-OFC5-TOOTH-AGENESIS-CLP-"
            "p.Arg31Pro-Founder-OMIM-142983"
        ),
        "locus": "4p16.2",
        "protein_size": (
            "303 aa / 31 kDa (MSX1; msh homeobox 1; NK-1 class homeodomain transcription factor; "
            "STRUCTURE: "
            "  N-terminal domain: transcriptional repression; "
            "  Homeodomain (HD): DNA binding to TAAT motif; "
            "  C-terminal tail: protein-protein interactions (partners: PAX9, BMPR1a); "
            "FUNCTION: "
            "  MSX1 represses transcription of target genes in tooth development; "
            "  Expressed in branchial arches, tooth germs (dental mesenchyme), palatal shelves; "
            "  MSX1-PAX9 interaction: cooperative regulation of tooth morphogenesis; "
            "  MSX1 LOF → failure of tooth induction → TOOTH AGENESIS; "
            "  Also required for palatal shelf elevation and fusion; "
            "OROFACIAL CLEFTING TYPE 5 (OFC5) — MSX1: "
            "  OMIM #608874; "
            "  COMBINED PHENOTYPE: TOOTH AGENESIS + CL/P; "
            "  Tooth agenesis pattern: "
            "    MAXILLARY PREMOLARS (#1 most common missing tooth type in MSX1); "
            "    MAXILLARY INCISORS (lateral incisors); "
            "    Can extend to SEVERE OLIGODONTIA (>6 missing teeth) including third molars; "
            "  CL/P: variable; CL ± CP; some CP only; "
            "  HYPODONTIA vs CLEFT: "
            "    Either can present alone; "
            "    Combined = CLASSIC MSX1 phenotype; "
            "MSX1 TOOTH AGENESIS SPECTRUM: "
            "  NON-SYNDROMIC TOOTH AGENESIS (autosomal dominant): "
            "    Isolated missing premolars/molars; no cleft; "
            "    OMIM #106600; "
            "    MSX1 = second most common non-syndromic tooth agenesis gene (after WNT10A); "
            "  OFC5 (with CL/P): more severe MSX1 mutations or modifier effects; "
            "MSX1 FOUNDER MUTATION: "
            "  p.Arg31Pro: well-documented European founder in isolated tooth agenesis cohorts; "
            "  p.Ile14Ser: family-specific; "
            "  Recurrent missense: p.Ala7Thr, p.Ser180Leu (homeodomain); "
            "encoded 4p16.2; OMIM gene 142983; OFC5 #608874; tooth agenesis #106600"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF (HAPLOINSUFFICIENCY) — MSX1 / OFC5: "
            "  HIGH PENETRANCE for tooth agenesis; LOWER PENETRANCE for CL/P; "
            "  INTRA-FAMILY VARIABILITY: "
            "    Same mutation → severe oligodontia in parent; isolated premolar agenesis in child; "
            "    Or tooth agenesis only in parent; child has tooth agenesis + cleft palate; "
            "  DE NOVO: rare; mostly familial; "
            "CLINICAL RECOGNITION: "
            "  Assess dental panoramic radiograph in all CL/P patients: missing teeth → test MSX1; "
            "  Tooth agenesis in parent/sibling of CL/P child → test MSX1 urgently; "
            "  MSX1 + PAX9: overlap phenotype — combined testing recommended; "
            "MANAGEMENT: "
            "  Cleft repair: standard (lip 3-6 mo; palate 12-18 mo); "
            "  Orthodontic planning: space management for missing teeth from age 5-6; "
            "  Prosthetic dental rehabilitation: bridges, dentures, implants at adulthood; "
            "  Alveolar bone graft for CL: facilitates later implant placement; "
            "  Genetic counselling: 50% risk; variable expressivity; "
            "  Screen MSX1 in all HYPODONTIA (>2 missing teeth) families"
        ),
        "disease_category": (
            "MSX1-OFC5-TOOTH-AGENESIS-CLP-LOF-HAPLOINSUFFICIENCY: "
            "  KEY RULE: TOOTH AGENESIS (premolars/incisors) + CL/P = MSX1 until proven otherwise; "
            "  MSX1 = #2 gene for non-syndromic tooth agenesis (after WNT10A); "
            "  INTRA-FAMILY VARIABILITY: same mutation → teeth only vs teeth + cleft; "
            "  PANORAMIC X-RAY mandatory in all CL/P families: missing teeth → MSX1 testing; "
            "  PAX9 overlap: MSX1 and PAX9 cooperate — test both when severe oligodontia + cleft"
        ),
    },
    {
        "gene": "TBX22",
        "protein": (
            "TBX22 -- Xq21.1 XLR -- 520aa -- T-Box-Transcription-Factor-22-"
            "X-Linked-Cleft-Palate-Ankyloglossia-CPX-"
            "ANKYLOGLOSSIA-Cleft-Palate-PATHOGNOMONIC-Combination-OMIM-300307"
        ),
        "locus": "Xq21.1",
        "protein_size": (
            "520 aa / 58 kDa (TBX22; T-box transcription factor 22; "
            "STRUCTURE: "
            "  T-box DNA-binding domain (TBD): binds T-element (AGGTGTGA); "
            "  N-terminal transactivation domain; "
            "  C-terminal repressor domain; "
            "FUNCTION: "
            "  TBX22 expressed in tongue musculature precursors and palatal mesenchyme; "
            "  Required for normal tongue development and palatal shelf elevation; "
            "  TBX22 LOF → tongue anchoring defect (ankyloglossia) AND palatal fusion failure; "
            "X-LINKED CLEFT PALATE WITH ANKYLOGLOSSIA (CPX): "
            "  OMIM #303400; X-linked recessive; "
            "  PATHOGNOMONIC COMBINATION: "
            "    ANKYLOGLOSSIA (tongue-tie): restricted tongue movement; "
            "      Short, tight lingual frenulum anchoring tongue to floor of mouth; "
            "      Causes feeding difficulties, speech articulation problems; "
            "    CLEFT PALATE (or submucous cleft palate): "
            "      Hard and soft palate; or submucous cleft (bifid uvula, palpable notch in posterior hard palate); "
            "    COMBINED ankyloglossia + cleft palate in males = CPX until proven otherwise; "
            "  SEX DIFFERENCES: "
            "    MALES (hemizygous): FULLY AFFECTED — cleft palate + ankyloglossia; "
            "    FEMALES (heterozygous): MOSAIC expression; "
            "      May have ANKYLOGLOSSIA ALONE without cleft palate; "
            "      Or submucous cleft only; "
            "      Generally milder than males; "
            "      IMPORTANT: female carriers may present with isolated tongue-tie; "
            "  BIFID UVULA: common in affected females; "
            "MUTATION SPECTRUM: "
            "  p.His213Asp: well-characterised pathogenic missense in T-box domain; "
            "  Missense (majority): T-box domain residues critical for DNA binding; "
            "  Nonsense/frameshift (minority): scattered throughout gene; "
            "  No founder mutation predominates globally; "
            "encoded Xq21.1; OMIM gene 300307; CPX #303400"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE — TBX22 / CPX: "
            "  MALES hemizygous: FULLY AFFECTED (cleft palate + ankyloglossia); "
            "  FEMALES heterozygous: CARRIER; variable expression (tongue-tie ± submucous cleft); "
            "  INHERITANCE PATTERNS: "
            "    Mother carrier → 50% of sons affected; 50% of daughters carriers; "
            "    Father affected → ALL daughters carriers (obligate carrier); ALL sons unaffected; "
            "  DE NOVO mutations in CPX: recognised but most are familial; "
            "  FAMILY HISTORY: examine mothers for tongue-tie or bifid uvula; "
            "  MOLECULAR: "
            "    TBX22 sequencing on X chromosome; "
            "    Males: hemizygous; females: heterozygous; "
            "CLINICAL RECOGNITION: "
            "  Male infant with CLEFT PALATE + TONGUE-TIE: test TBX22 immediately; "
            "  Female with isolated tongue-tie + family history CP in males: TBX22 carrier testing; "
            "MANAGEMENT: "
            "  Palate repair: 12-18 months; "
            "  Ankyloglossia: frenotomy/frenuloplasty — timing varies (if feeding impaired: early; "
            "    speech: age 2-4 years); "
            "  Speech therapy: essential post-palate repair; "
            "  Hearing: otological surveillance (glue ear common in cleft palate)"
        ),
        "disease_category": (
            "TBX22-CPX-XLR-ANKYLOGLOSSIA-CLEFT-PALATE-PATHOGNOMONIC-COMBINATION: "
            "  KEY RULE: MALE with CLEFT PALATE + ANKYLOGLOSSIA (tongue-tie) = CPX (TBX22) until proven otherwise; "
            "  X-LINKED RECESSIVE: males fully affected; females mosaic (tongue-tie ± submucous cleft); "
            "  FEMALE CARRIERS: may have ISOLATED TONGUE-TIE — examine mothers of affected males; "
            "  OBLIGATE CARRIERS: all daughters of affected males are carriers; "
            "  SUBMUCOUS CLEFT: palpate posterior palate in all tongue-tie cases"
        ),
    },
    {
        "gene": "SATB2",
        "protein": (
            "SATB2 -- 2q33.1 AD-De-Novo -- 733aa -- Special-AT-Rich-Binding-Protein-2-"
            "SATB2-Associated-Syndrome-Glass-Syndrome-"
            "ABSENT-SPEECH-Severe-ID-CLP-Behavioral-Dental-De-Novo-OMIM-608148"
        ),
        "locus": "2q33.1",
        "protein_size": (
            "733 aa / 82 kDa (SATB2; special AT-rich sequence-binding protein 2; "
            "STRUCTURE: "
            "  CUT domains (CUT1, CUT2): chromatin remodelling; AT-rich sequence binding; "
            "  Homeodomain: DNA binding; "
            "  Nuclear localisation signal; "
            "  Interacts with histone deacetylases (HDAC1/2) → transcriptional repression; "
            "FUNCTION: "
            "  SATB2: master regulator of cortical neuron identity and osteoblast differentiation; "
            "  In developing brain: corticospinal neuron specification (layer V), callosal neurons; "
            "  In craniofacial development: palatal and dental mesenchyme proliferation; "
            "  In bone: SATB2 activates osteocalcin, RUNX2 cofactor → mandible development; "
            "  SATB2 LOF → neurodevelopmental + craniofacial + dental abnormalities; "
            "SATB2-ASSOCIATED SYNDROME (SAS) / GLASS SYNDROME: "
            "  OMIM #612313; recognised 2008 (Glass et al.); named Glass syndrome 2012; "
            "  PREDOMINANTLY DE NOVO: >95% of cases are de novo; "
            "  SEVERE NEURODEVELOPMENTAL PHENOTYPE: "
            "    INTELLECTUAL DISABILITY: SEVERE; global developmental delay; "
            "    ABSENT OR SEVERELY IMPAIRED SPEECH: "
            "      PATHOGNOMONIC FEATURE: most patients never develop functional speech; "
            "      May have a few words or non-verbal communication only; "
            "      Augmentative and alternative communication (AAC) required; "
            "    BEHAVIORAL: autism spectrum disorder (40-50%); hyperactivity; aggression; "
            "      Self-injurious behavior; repetitive behaviors; "
            "  OROFACIAL FEATURES: "
            "    Cleft palate or high-arched palate; "
            "    CL/P in subset; submucous cleft; "
            "    DENTAL ANOMALIES: severe tooth agenesis; enamel hypoplasia; malocclusion; "
            "  SKELETAL: "
            "    Osteoporosis/osteopenia (DEXA scan recommended); "
            "    Short stature in some; "
            "    Dysplastic teeth (enamel hypomineralisation); "
            "  BRAIN MRI: "
            "    Thin/absent corpus callosum (partial agenesis): ~15%; "
            "    Cortical dysplasia: rare; "
            "MUTATION TYPES: "
            "  De novo missense (CUT domains, homeodomain): most common; "
            "  De novo nonsense/frameshift; "
            "  De novo splice-site; "
            "  2q33 deletion (includes SATB2): del 2q33 syndrome; "
            "encoded 2q33.1; OMIM gene 608148; SAS/Glass #612313"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT DE NOVO — SATB2 / GLASS SYNDROME: "
            "  >95% DE NOVO mutations: recurrence risk <1% for sibling (germline mosaicism risk ~1-2%); "
            "  FAMILIAL cases: rare; parent may be MOSAIC; "
            "  Recurrence risk for affected individual's offspring: 50% (AD); "
            "  DIAGNOSIS: "
            "    CMA (chromosomal microarray) first: detect 2q33 deletion; "
            "    SATB2 sequencing: detect point mutations/indels; "
            "    Trio sequencing (proband + parents) confirms de novo status; "
            "CLINICAL RECOGNITION: "
            "  Severe ID + ABSENT SPEECH + dental anomalies + palatal anomaly → test SATB2; "
            "  SATB2 should be in the differential for ALL children with severe ID + absent speech; "
            "MANAGEMENT: "
            "  Palate repair: standard (12-18 months); "
            "  AAC: augmentative and alternative communication — PRIORITY; "
            "  Behavioral support: ABA therapy, psychiatric input; "
            "  Dental: prosthodontic management for severe tooth agenesis; "
            "  Bone health: DEXA at baseline; calcium/vitamin D supplementation; "
            "    Bisphosphonates if severe osteoporosis; "
            "  Seizure monitoring: EEG if seizures suspected (occur in minority)"
        ),
        "disease_category": (
            "SATB2-GLASS-SYNDROME-DE-NOVO-ABSENT-SPEECH-SEVERE-ID-CLP-DENTAL: "
            "  KEY RULE: SEVERE ID + ABSENT SPEECH + DENTAL ANOMALIES + PALATAL ANOMALY = SATB2/Glass syndrome; "
            "  DE NOVO >95%: recurrence risk <1% for sibling; "
            "  ABSENT SPEECH: most pathognomonic feature — children rarely develop functional spoken language; "
            "  AAC (augmentative communication) is STANDARD OF CARE — not optional; "
            "  ASD 40-50%: behavioral support mandatory; "
            "  BONE HEALTH: DEXA recommended (SATB2 is osteoblast master regulator)"
        ),
    },
    {
        "gene": "TP63",
        "protein": (
            "TP63 -- 3q28 AD-GOF-DN -- 680aa -- Tumor-Protein-P63-"
            "EEC-Syndrome-Ectrodactyly-Ectodermal-Dysplasia-Clefting-"
            "ECTRODACTYLY-Ectodermal-Dysplasia-CLP-TRIAD-PATHOGNOMONIC-OMIM-603273"
        ),
        "locus": "3q28",
        "protein_size": (
            "680 aa / 70 kDa (TP63; tumor protein p63; p53 family member; "
            "ISOFORMS: "
            "  TAp63 (transactivating): N-terminal TA domain; transactivates p53 targets; "
            "  ΔNp63 (dominant-negative): lacks TA domain; dominant-negative over p53 and p73; "
            "  Additional: α, β, γ C-terminal isoforms; "
            "  ΔNp63α: major isoform in stratified epithelia (skin, oral mucosa); "
            "FUNCTION: "
            "  TP63: master regulator of stratified squamous epithelium development; "
            "  Required for ectoderm specification, limb bud ectoderm, apical ectodermal ridge (AER); "
            "  TP63 mutations → AER defect → ectrodactyly (split hand/foot malformation); "
            "  Critical for ectodermal appendage (hair, nails, sweat glands, teeth) specification; "
            "  p63 regulates IRF6 expression in palatal epithelium → TP63 → IRF6 pathway; "
            "EEC SYNDROME — ECTRODACTYLY-ECTODERMAL DYSPLASIA-CLEFTING: "
            "  OMIM #604292; most common TP63 syndrome; "
            "  PATHOGNOMONIC TRIAD: "
            "    (1) ECTRODACTYLY (split hand/foot malformation, SHFM): "
            "        Lobster-claw deformity; absent central rays (digits 2,3,4); "
            "        May be unilateral or bilateral; hands and/or feet; "
            "        PATHOGNOMONIC for TP63 EEC — no other orofacial clefting gene causes this; "
            "    (2) ECTODERMAL DYSPLASIA: "
            "        Sparse/absent hair (hypotrichosis); "
            "        Hypohidrosis (reduced sweating → heat intolerance); "
            "        Dental anomalies (anodontia, oligodontia, enamel hypoplasia); "
            "        Lacrimal duct stenosis/atresia → photophobia, tearing; "
            "    (3) CLEFT LIP/PALATE: "
            "        CL/P or CP alone; variable; "
            "  ADDITIONAL TP63 SYNDROMES (ALLELE-SPECIFIC): "
            "    RAPP-HODGKIN SYNDROME (RHS): ectodermal dysplasia + cleft; NO ectrodactyly; "
            "    AEC (ankyloblepharon-ectodermal dysplasia-cleft) / HAY-WELLS: "
            "      ANKYLOBLEPHARON (eyelid fusion strands) PATHOGNOMONIC for AEC; "
            "      Severe skin erosions at birth; "
            "    LIMB-MAMMARY SYNDROME (LMS): ectrodactyly + mammary gland hypoplasia; NO cleft; "
            "    ADULT SYNDROME: ectodermal dysplasia; lacrimal duct atresia; digit malformation; NO cleft; "
            "    TP63 BLADDER CANCER ASSOCIATION: separate somatic TP63 mutations in bladder carcinoma; "
            "ALLELE-SPECIFIC GENOTYPE-PHENOTYPE: "
            "  EEC: missense mutations in DNA-binding domain (R204W, R204Q, R227W, R279H, R304W, R304Q); "
            "  AEC: mutations in SAM (sterile alpha motif) domain; "
            "  RHS: overlaps with EEC domain; "
            "  LMS: isoleucine/leucine-rich domain; "
            "encoded 3q28; OMIM gene 603273; EEC #604292; AEC #106260; RHS #129400; LMS #603543"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GOF/DOMINANT-NEGATIVE — TP63 / EEC AND RELATED: "
            "  DOMINANT-NEGATIVE MECHANISM: "
            "    EEC mutations: ΔNp63α dominant-negative over wild-type p63; disrupts AER; "
            "    NOT simple haploinsufficiency — gain-of-abnormal-function interfering with normal p63 targets; "
            "  DE NOVO: ~90% of EEC is de novo; "
            "  FAMILIAL: rare; variable expressivity within families; "
            "  PENETRANCE: high for limb involvement; variable for cleft and ectodermal features; "
            "ALLELE-SPECIFIC TESTING: "
            "  DNA-binding domain hotspots (codons 204, 227, 279, 304) → EEC; "
            "  SAM domain mutations → AEC; "
            "  Full TP63 sequencing if phenotype suggests any TP63 syndrome; "
            "MANAGEMENT — EEC: "
            "  Cleft repair: standard timing; "
            "  Ectrodactyly: orthopaedic/plastic surgery staged correction; prosthetics as needed; "
            "  Ectodermal dysplasia: "
            "    Overheating risk: air conditioning; avoid heat exertion; school accommodation; "
            "    Lacrimal duct probing (if atresia) age 6-12 months; "
            "  Dental: comprehensive prosthetics from childhood; "
            "  Ophthalmology: annual (keratitis risk from lacrimal/corneal issues); "
            "  AEC: skin management team (severe erosions require specialist wound care)"
        ),
        "disease_category": (
            "TP63-EEC-ECTRODACTYLY-ECTODERMAL-DYSPLASIA-CLP-TRIAD-PATHOGNOMONIC: "
            "  KEY RULE: ECTRODACTYLY (split hand/foot) + ECTODERMAL DYSPLASIA + CL/P = EEC (TP63) — TRIAD PATHOGNOMONIC; "
            "  ALLELE-SPECIFIC: DNA-binding domain mutations → EEC; SAM domain → AEC (ankyloblepharon); "
            "  DOMINANT-NEGATIVE MECHANISM: ΔNp63α isoform interfering with wild-type p63 targets; "
            "  AEC TRAP: ankyloblepharon (eyelid fusions) distinguishes AEC from EEC — SAME GENE, different domain; "
            "  TP63 → IRF6: p63 regulates IRF6 → TP63 and IRF6/GRHL3 are in same pathway"
        ),
    },
    {
        "gene": "COL11A1",
        "protein": (
            "COL11A1 -- 1p21.1 AD -- 1806aa -- Collagen-Type-XI-Alpha-1-Chain-"
            "Marshall-Syndrome-Stickler-Syndrome-Type2-"
            "Cleft-Palate-HIGH-MYOPIA-SNHL-TRIAD-Flat-Midface-Large-Eye-Globes-OMIM-120280"
        ),
        "locus": "1p21.1",
        "protein_size": (
            "1806 aa / 200 kDa (COL11A1; collagen type XI alpha 1 chain; "
            "STRUCTURE: "
            "  Signal peptide; N-propeptide (alternative splicing determines cartilage vs vitreous forms); "
            "  Triple-helix domain: Gly-X-Y repeat region (1800+ aa); "
            "  C-propeptide; "
            "  COL11A1 + COL11A2 + COL2A1 → heterotrimer: type XI collagen; "
            "  Type XI collagen: fibril diameter regulator in vitreous humour and cartilage; "
            "FUNCTION: "
            "  Controls type II collagen fibril spacing/diameter in cartilage and vitreous; "
            "  Critical in nasal cartilage, palatal connective tissue, inner ear stria vascularis; "
            "  COL11A1 LOF → abnormal collagen fibril assembly → connective tissue fragility; "
            "MARSHALL SYNDROME vs STICKLER SYNDROME TYPE 2: "
            "  BOTH caused by COL11A1 mutations; SAME GENE; "
            "  CLINICAL DISTINCTION (historically debated): "
            "    MARSHALL: "
            "      FLAT MIDFACE (malar hypoplasia) — more pronounced than Stickler; "
            "      LARGE EYE GLOBES (increased axial length); "
            "      High myopia (typically >-10 dioptres); "
            "      Sensorineural hearing loss (more prominent than Stickler 2); "
            "      Midfacial hypoplasia more severe; "
            "    STICKLER TYPE 2 (COL11A1): "
            "      Vitreous: BEADED VITREOUS (type 2 vitreous anomaly — distinct from Stickler 1 optically empty); "
            "      Myopia; SNHL; cleft palate; "
            "  SHARED FEATURES (Marshall/Stickler 2 overlap): "
            "    CLEFT PALATE (or Robin sequence): frequent; "
            "    HIGH MYOPIA: retinal detachment risk (annual ophthalmology); "
            "    SENSORINEURAL HEARING LOSS: 50-80%; progressive; hearing aids; "
            "    Arthritis/hypermobility: joint laxity; early-onset arthropathy; "
            "  STICKLER TYPE 1 (COL2A1) vs TYPE 2 (COL11A1) DISTINCTION: "
            "    Stickler 1 (COL2A1): OPTICALLY EMPTY vitreous; more severe myopia; AD; "
            "    Stickler 2 (COL11A1): BEADED VITREOUS; mild-moderate myopia; AD; "
            "    VITREOUS EXAMINATION essential to distinguish; "
            "ROBIN SEQUENCE: "
            "  Micrognathia → tongue falls back → cleft palate obstruction → Robin sequence; "
            "  COL11A1/COL2A1/Stickler syndromes: important cause of Robin sequence; "
            "  Must exclude Stickler in all Robin sequence neonates (ophthalmology assessment); "
            "encoded 1p21.1; OMIM gene 120280; Marshall #154780; Stickler 2 #604841"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — COL11A1 / MARSHALL-STICKLER2: "
            "  HIGH PENETRANCE but VARIABLE EXPRESSIVITY; "
            "  DE NOVO and FAMILIAL; "
            "  GENOTYPE-PHENOTYPE: "
            "    Marshall: predominantly exon-skipping splice variants (especially exon 50); "
            "    Stickler 2: missense and LOF throughout; "
            "    Overlap exists — Marshall/Stickler2 boundary debated; "
            "  MUTATION TYPES: "
            "    Splice-site mutations (Marshall: exon 50 skip); "
            "    Missense (Gly-X-Y disruption); "
            "    Nonsense/frameshift; "
            "    Large deletions: rare; "
            "CLINICAL MANAGEMENT: "
            "  OPHTHALMOLOGY (PRIORITY): "
            "    Annual retinal examination for lattice degeneration and retinal detachment; "
            "    Retinal detachment risk: ~30-50% lifetime in high myopia Stickler; "
            "    Prophylactic laser retinopexy at first retinal tear; "
            "    Myopia correction (glasses/contact lens); "
            "  HEARING: "
            "    Annual audiometry from diagnosis; "
            "    Hearing aids for significant SNHL; "
            "  CLEFT/ROBIN: standard cleft repair; airway management for Robin; "
            "  JOINTS: physiotherapy; avoid high-impact sports with arthropathy; "
            "  Genetic counselling: 50% recurrence; examine parent for mild features"
        ),
        "disease_category": (
            "COL11A1-MARSHALL-STICKLER2-CLP-HIGH-MYOPIA-SNHL-TRIAD: "
            "  KEY RULE: CLEFT PALATE + HIGH MYOPIA + SNHL = Marshall/Stickler type 2 (COL11A1); "
            "  MARSHALL vs STICKLER2: same gene; Marshall has flat midface + large eye globes; "
            "  VITREOUS DISTINCTION: Stickler 1 (COL2A1) = optically empty; Stickler 2 (COL11A1) = beaded; "
            "  ROBIN SEQUENCE: COL11A1/COL2A1 must be excluded in all Robin sequence neonates; "
            "  ANNUAL OPHTHALMOLOGY MANDATORY: retinal detachment 30-50% lifetime risk; laser retinopexy at first tear; "
            "  EXON 50 SPLICE: classic Marshall mutation — exon-skipping alters N-propeptide"
        ),
    },
]


def _generate_patients_for_gene(gene: str, seed: int) -> list:
    """Generate 40 synthetic patients for one orofacial clefting gene."""
    rng = random.Random(seed)

    severity_params = {
        "IRF6":    {"severe_pct": 0.25, "moderate_pct": 0.50, "mild_pct": 0.25,
                    "age_min": 0, "age_max": 3, "iq_mean": 100, "iq_sd": 8},
        "GRHL3":   {"severe_pct": 0.20, "moderate_pct": 0.50, "mild_pct": 0.30,
                    "age_min": 0, "age_max": 3, "iq_mean": 100, "iq_sd": 8},
        "PVRL1":   {"severe_pct": 0.30, "moderate_pct": 0.45, "mild_pct": 0.25,
                    "age_min": 0, "age_max": 3, "iq_mean": 99, "iq_sd": 9},
        "MSX1":    {"severe_pct": 0.20, "moderate_pct": 0.45, "mild_pct": 0.35,
                    "age_min": 0, "age_max": 5, "iq_mean": 100, "iq_sd": 8},
        "TBX22":   {"severe_pct": 0.30, "moderate_pct": 0.45, "mild_pct": 0.25,
                    "age_min": 0, "age_max": 2, "iq_mean": 98, "iq_sd": 9},
        "SATB2":   {"severe_pct": 0.65, "moderate_pct": 0.30, "mild_pct": 0.05,
                    "age_min": 0, "age_max": 4, "iq_mean": 42, "iq_sd": 10},
        "TP63":    {"severe_pct": 0.40, "moderate_pct": 0.40, "mild_pct": 0.20,
                    "age_min": 0, "age_max": 2, "iq_mean": 96, "iq_sd": 10},
        "COL11A1": {"severe_pct": 0.30, "moderate_pct": 0.45, "mild_pct": 0.25,
                    "age_min": 0, "age_max": 5, "iq_mean": 99, "iq_sd": 8},
    }
    p = severity_params.get(gene, {"severe_pct": 0.30, "moderate_pct": 0.45, "mild_pct": 0.25,
                                    "age_min": 0, "age_max": 3, "iq_mean": 98, "iq_sd": 10})

    cleft_map = {
        "IRF6":    [["cleft-lip-bilateral-CL", "lip-pits-lower-bilateral"],
                    ["cleft-lip-palate-CLP", "lip-pits-lower"],
                    ["cleft-palate-CP-only", "lip-pits-minimal"],
                    ["lip-pits-only-no-cleft-carrier-phenotype"]],
        "GRHL3":   [["cleft-lip-palate-CLP", "lip-pits"],
                    ["cleft-palate-CP-only"],
                    ["isolated-lip-pits-VWS2"]],
        "PVRL1":   [["cleft-lip-palate-bilateral-CLP", "hypodontia", "palmar-keratoderma"],
                    ["cleft-palate-CP", "hypodontia", "nail-dysplasia"],
                    ["cleft-lip-unilateral-CL", "hypodontia"]],
        "MSX1":    [["cleft-palate-CP-only", "hypodontia-maxillary-premolars"],
                    ["cleft-lip-palate-CLP", "oligodontia-severe"],
                    ["hypodontia-isolated-no-cleft"],
                    ["cleft-palate-CP", "lateral-incisor-agenesis"]],
        "TBX22":   [["cleft-palate-CP-male", "ankyloglossia-tongue-tie"],
                    ["submucous-cleft-palate", "ankyloglossia"],
                    ["ankyloglossia-isolated-female-carrier"],
                    ["bifid-uvula", "ankyloglossia"]],
        "SATB2":   [["cleft-palate-CP", "severe-intellectual-disability", "absent-speech", "dental-anomalies"],
                    ["high-arched-palate", "severe-ID", "absent-speech", "autism-spectrum"],
                    ["cleft-palate", "absent-speech", "behavioral-problems", "osteoporosis"]],
        "TP63":    [["ectrodactyly-split-hand-foot", "ectodermal-dysplasia", "cleft-lip-palate-EEC-triad"],
                    ["ectrodactyly", "lacrimal-duct-atresia", "hypohidrosis", "cleft-palate"],
                    ["ankyloblepharon-eyelid-fusion-AEC", "skin-erosions", "cleft-palate"]],
        "COL11A1": [["cleft-palate-CP", "high-myopia-gt10D", "snhl-sensorineural"],
                    ["robin-sequence", "high-myopia", "snhl", "flat-midface-marshall"],
                    ["cleft-palate", "myopia", "joint-hypermobility-arthropathy"]],
    }
    cleft_options = cleft_map.get(gene, [["cleft-palate"]])

    treatment_map = {
        "IRF6":    ["cleft-lip-repair-3-6mo", "palate-repair-12-18mo",
                    "lip-pit-excision-5yr", "alveolar-bone-graft", "orthodontics"],
        "GRHL3":   ["cleft-lip-repair-3-6mo", "palate-repair-12-18mo",
                    "lip-pit-excision", "orthodontics"],
        "PVRL1":   ["cleft-lip-repair", "palate-repair-12-18mo",
                    "prosthetic-teeth-hypodontia", "dermatology-keratoderma", "dental-implants-adulthood"],
        "MSX1":    ["palate-repair-12-18mo", "cleft-lip-repair-3-6mo",
                    "space-maintenance-orthodontics", "prosthetic-teeth", "alveolar-bone-graft",
                    "dental-implants-adulthood"],
        "TBX22":   ["palate-repair-12-18mo", "frenotomy-frenuloplasty-ankyloglossia",
                    "speech-therapy", "otological-surveillance-glue-ear"],
        "SATB2":   ["palate-repair-12-18mo", "AAC-augmentative-communication",
                    "ABA-behavioral-therapy", "dental-prosthodontics",
                    "DEXA-bone-health", "bisphosphonates-osteoporosis"],
        "TP63":    ["cleft-lip-repair-3-6mo", "palate-repair-12-18mo",
                    "lacrimal-duct-probing", "ectrodactyly-orthopaedic-correction",
                    "ectodermal-dysplasia-management-overheating", "dental-prosthetics"],
        "COL11A1": ["palate-repair-12-18mo", "annual-ophthalmology-retinal-surveillance",
                    "laser-retinopexy-retinal-tear", "hearing-aids-SNHL",
                    "joint-physiotherapy-arthropathy"],
    }
    treatments = treatment_map.get(gene, ["cleft-lip-repair", "palate-repair"])

    mutation_map = {
        "IRF6":    ["p.Arg84His-founder", "p.Arg84Cys-founder", "p.Gly267Asp",
                    "p.Glu396Ter", "p.Arg252Ter", "c.1234del-frameshift",
                    "p.Phe252Leu-VWS1", "p.Pro409Ala"],
        "GRHL3":   ["p.Arg339Ter-VWS2", "p.Trp233Ter", "c.847+1G>T-splice",
                    "p.Gln376fs", "p.Thr349Met-LOF"],
        "PVRL1":   ["p.Val99Gly-Mediterranean-founder", "p.Arg294Ter-Israeli-founder",
                    "p.Gly65Asp", "p.Trp185Ter", "c.1018del-frameshift"],
        "MSX1":    ["p.Arg31Pro-founder", "p.Ile14Ser", "p.Ala7Thr",
                    "p.Ser180Leu-homeodomain", "p.Arg192Gly", "p.Pro147Leu"],
        "TBX22":   ["p.His213Asp-T-box", "p.Gln118Ter", "p.Arg185Trp",
                    "p.Tyr207Cys-T-box", "c.649G>C-splice"],
        "SATB2":   ["p.Arg389Cys-de-novo", "p.Gly292Ser-de-novo", "p.Arg446Ter-de-novo",
                    "2q33-deletion-de-novo", "p.Lys450Asn-CUT-domain",
                    "c.1049+1G>A-splice-de-novo"],
        "TP63":    ["p.Arg204Trp-EEC-DBD", "p.Arg204Gln-EEC-DBD", "p.Arg227Trp-EEC",
                    "p.Arg304Trp-EEC-DBD", "p.Arg304Gln-EEC-DBD",
                    "p.Asp541Asn-AEC-SAM", "p.Ile537Val-AEC-SAM"],
        "COL11A1": ["c.4004+1G>A-splice-exon50-Marshall", "p.Gly592Arg-Gly-XY",
                    "p.Gly1219Val-triple-helix", "c.3412G>T-Stickler2",
                    "p.Gly805Asp", "1p21-deletion"],
    }
    mutations = mutation_map.get(gene, ["unknown"])

    # Gene-specific feature flags
    snhl_prob = {
        "COL11A1": 0.65, "TBX22": 0.0, "SATB2": 0.0, "IRF6": 0.0,
        "GRHL3": 0.0, "PVRL1": 0.0, "MSX1": 0.0, "TP63": 0.0,
    }
    dental_prob = {
        "MSX1": 0.80, "PVRL1": 0.75, "SATB2": 0.85, "COL11A1": 0.40,
        "IRF6": 0.15, "GRHL3": 0.10, "TBX22": 0.10, "TP63": 0.70,
    }
    ectrodactyly_prob = {
        "TP63": 0.70, "IRF6": 0.0, "GRHL3": 0.0, "PVRL1": 0.0,
        "MSX1": 0.0, "TBX22": 0.0, "SATB2": 0.0, "COL11A1": 0.0,
    }
    myopia_prob = {
        "COL11A1": 0.90, "IRF6": 0.0, "GRHL3": 0.0, "PVRL1": 0.0,
        "MSX1": 0.0, "TBX22": 0.0, "SATB2": 0.0, "TP63": 0.0,
    }

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
        iq = max(25, int(rng.normalvariate(p["iq_mean"], p["iq_sd"])))
        snhl = rng.random() < snhl_prob.get(gene, 0.0)
        dental_anomaly = rng.random() < dental_prob.get(gene, 0.10)
        ectrodactyly = rng.random() < ectrodactyly_prob.get(gene, 0.0)
        high_myopia = rng.random() < myopia_prob.get(gene, 0.0)
        ankyloglossia = gene == "TBX22" and rng.random() < 0.70
        absent_speech = gene == "SATB2" and rng.random() < 0.80
        lip_pits = gene in ("IRF6", "GRHL3") and rng.random() < 0.75
        raised_icp = False  # not a primary feature in orofacial clefting
        treatment = rng.choice(treatments)
        mutation = rng.choice(mutations)
        cleft_features = rng.choice(cleft_options)
        patients.append({
            "id":                   f"{gene}-{seed}-{i+1:03d}",
            "gene":                 gene,
            "age_at_diagnosis_mo":  age_dx * 12 if age_dx < 3 else age_dx,
            "severity":             severity,
            "iq_estimate":          iq,
            "cleft_features":       cleft_features,
            "snhl":                 snhl,
            "dental_anomaly":       dental_anomaly,
            "ectrodactyly":         ectrodactyly,
            "high_myopia":          high_myopia,
            "ankyloglossia":        ankyloglossia,
            "absent_speech":        absent_speech,
            "lip_pits":             lip_pits,
            "raised_icp":           raised_icp,
            "treatment":            treatment,
            "mutation":             mutation,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Orofacial-Clefting-Atlas."""
    return {
        "atlas":          "Hereditary-Orofacial-Clefting-Atlas",
        "subtitle":       (
            "Complete 8-Gene Orofacial Clefting & Cleft Lip/Palate Genetics Atlas "
            "(IRF6-GRHL3-PVRL1-MSX1-TBX22-SATB2-TP63-COL11A1)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "IRF6":    "AD LOF (VWS1; lip pits PATHOGNOMONIC; most common hereditary CL/P gene; p.Arg84His/Cys founders; PPS allelic)",
            "GRHL3":   "AD LOF (VWS2; clinically identical to VWS1; IRF6-downstream pathway; nonsense/frameshift LOF)",
            "PVRL1":   "AR biallelic LOF (CLPED1; CL/P + hypodontia + ectodermal dysplasia; Mediterranean founder p.Val99Gly)",
            "MSX1":    "AD LOF (OFC5; tooth agenesis #2 gene + CL/P; premolar/incisor agenesis; p.Arg31Pro founder)",
            "TBX22":   "XLR (CPX; ankyloglossia + cleft palate PATHOGNOMONIC combination; males fully affected; females mosaic)",
            "SATB2":   "AD de novo LOF (Glass syndrome; severe ID + ABSENT SPEECH + CL/P + dental anomalies; >95% de novo)",
            "TP63":    "AD GOF/dominant-negative (EEC; ectrodactyly + ectodermal dysplasia + CL/P TRIAD; allele-specific syndromes)",
            "COL11A1": "AD LOF (Marshall/Stickler2; cleft palate + high myopia + SNHL triad; retinal detachment 30-50%)",
        },
        "key_clinical_rules": [
            "LIP PITS on lower lip = Van der Woude Syndrome (IRF6 or GRHL3) until proven otherwise — examine ALL CL/P families",
            "IRF6 NEGATIVE + LIP PITS + CL/P → test GRHL3 (VWS2) — second most common VWS gene",
            "PVRL1 CLPED1: CL/P + HYPODONTIA + ECTODERMAL SIGNS = AR; Mediterranean founder p.Val99Gly; normal intelligence",
            "MSX1: TOOTH AGENESIS (premolars) + CL/P → MSX1; panoramic X-ray mandatory in all CL/P families",
            "TBX22 CPX: MALE with CLEFT PALATE + ANKYLOGLOSSIA (tongue-tie) = CPX until proven otherwise",
            "TBX22: FEMALE CARRIERS may have ISOLATED TONGUE-TIE — examine all mothers of males with cleft palate",
            "SATB2/Glass: SEVERE ID + ABSENT SPEECH + DENTAL ANOMALIES + PALATAL ANOMALY — >95% de novo; AAC standard of care",
            "TP63 EEC TRIAD: ECTRODACTYLY + ECTODERMAL DYSPLASIA + CL/P = PATHOGNOMONIC — no other orofacial clefting gene causes ectrodactyly",
            "TP63 AEC: ANKYLOBLEPHARON (eyelid fusions) + ectodermal dysplasia + cleft = AEC — SAME TP63 gene, SAM domain mutations",
            "COL11A1: CLEFT PALATE + HIGH MYOPIA + SNHL = Marshall/Stickler2; ANNUAL OPHTHALMOLOGY mandatory (retinal detachment 30-50%)",
            "ROBIN SEQUENCE: exclude COL11A1/COL2A1 Stickler in ALL Robin sequence neonates — ophthalmology assessment at birth",
            "VARIABLE EXPRESSIVITY: IRF6/GRHL3/MSX1 — same mutation → lip pits only OR full CL/P in same family",
        ],
        "gene_panel_note": (
            "Comprehensive orofacial clefting gene panel (2024): IRF6, GRHL3, PVRL1/NECTIN1, MSX1, TBX22, SATB2, TP63, COL11A1, "
            "COL2A1 (Stickler 1), PAX9 (tooth agenesis), WNT10A (tooth agenesis/ectodermal dysplasia), FGFR1/2 (Kallmann/craniosynostosis overlap), "
            "KMT2D (Kabuki), ANKRD11 (KBG), KAT6A, PHF8 (X-linked ID with cleft)"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Orofacial-Clefting-Atlas."""
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
        snhl_n = sum(1 for p in patients if p["snhl"])
        dental_n = sum(1 for p in patients if p["dental_anomaly"])
        ectrodactyly_n = sum(1 for p in patients if p["ectrodactyly"])
        high_myopia_n = sum(1 for p in patients if p["high_myopia"])
        ankyloglossia_n = sum(1 for p in patients if p["ankyloglossia"])
        absent_speech_n = sum(1 for p in patients if p["absent_speech"])
        lip_pits_n = sum(1 for p in patients if p["lip_pits"])
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
            "snhl_pct":          round(snhl_n / n * 100, 1),
            "dental_pct":        round(dental_n / n * 100, 1),
            "ectrodactyly_pct":  round(ectrodactyly_n / n * 100, 1),
            "high_myopia_pct":   round(high_myopia_n / n * 100, 1),
            "ankyloglossia_pct": round(ankyloglossia_n / n * 100, 1),
            "absent_speech_pct": round(absent_speech_n / n * 100, 1),
            "lip_pits_pct":      round(lip_pits_n / n * 100, 1),
            "raised_icp_pct":    0.0,
            "cardiac_pct":       0.0,
            "chiari_pct":        0.0,
            "mean_age_dx_mo":    mean_age,
            "sample_mutations":  mutations_seen[:4],
            "protein":           gene_info["protein"],
            "inheritance":       gene_info["inheritance"][:200],
            "disease_category":  gene_info["disease_category"],
        })
    return {
        "atlas": "Hereditary-Orofacial-Clefting-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Orofacial-Clefting-Atlas."""
    definitions = [
        {
            "term": "Orofacial Clefting — Classification, Epidemiology, and Diagnostic Approach",
            "genes": ["IRF6", "GRHL3", "PVRL1", "MSX1", "TBX22", "SATB2", "TP63", "COL11A1"],
            "definition": (
                "OROFACIAL CLEFTING — OVERVIEW: "
                "DEFINITION: failure of fusion of facial processes during embryogenesis weeks 4-12; "
                "  CL ± CP: failure of medial nasal process + maxillary process fusion (weeks 5-7); "
                "  Isolated CP: failure of palatal shelf elevation and fusion (weeks 8-12); "
                "EPIDEMIOLOGY: "
                "  CL ± CP prevalence: ~1:700 live births (all ethnicities combined); "
                "  Isolated CP: ~1:1,500 live births; "
                "  CL/P + CP together: most common craniofacial birth defect; "
                "  GENETIC CAUSE: ~30% have identifiable genetic cause; 70% multifactorial; "
                "CLASSIFICATION: "
                "  CL (cleft lip without palate): unilateral > bilateral; "
                "  CL/P (cleft lip + palate): most common combined form; "
                "  CP (isolated cleft palate): different genetic aetiology from CL/P; "
                "  SUBMUCOUS CLEFT: bifid uvula + notch in posterior hard palate + diastasis of velum; "
                "  MICROFORM: vertical scar above lip (forme fruste of CL); "
                "GENETIC YIELD BY CONTEXT: "
                "  SYNDROMIC CL/P (+ other anomalies): high genetic yield → panel testing; "
                "  ISOLATED CL/P: 10-15% genetic cause; "
                "  ISOLATED CP: higher genetic yield (~20%) than isolated CL/P; "
                "  FAMILY HISTORY: first-degree relative with CL/P → panel urgently; "
                "DIAGNOSTIC ALGORITHM: "
                "  Step 1: Clinical phenotyping: lip pits, teeth, limbs, eyes, hearing, ID/speech; "
                "  Step 2: Family history: lip pits in relatives (VWS); tooth agenesis; males with tongue-tie; "
                "  Step 3: Dental panoramic X-ray: missing teeth → MSX1/PVRL1/SATB2; "
                "  Step 4: Gene panel: IRF6 + GRHL3 first (VWS); MSX1, TBX22, PVRL1; "
                "  Step 5: TP63 panel if ectrodactyly or ectodermal dysplasia; "
                "  Step 6: COL11A1/COL2A1 if myopia/SNHL (Robin sequence/Stickler); "
                "  Step 7: Trio exome/genome for syndromic with no panel diagnosis (SATB2 de novo); "
                "MULTIDISCIPLINARY TEAM: "
                "  Cleft surgeon + genetics + orthodontics + speech therapy + audiology + ophthalmology"
            ),
        },
        {
            "term": "Van der Woude Syndrome — IRF6 vs GRHL3 Differential and Lip Pit Recognition",
            "genes": ["IRF6", "GRHL3"],
            "definition": (
                "VAN DER WOUDE SYNDROME (VWS) — IRF6 AND GRHL3: "
                "DIAGNOSTIC HALLMARK: "
                "  LIP PITS: PATHOGNOMONIC for VWS; "
                "    Paramedian lower lip pits (bilateral, symmetric is classic); "
                "    Can be unilateral; can be minimally expressed (tiny depression); "
                "    May be scarred from prior surgical excision — check records; "
                "    LIP PITS alone (without cleft) = CARRIER/MILDLY AFFECTED family member; "
                "  Any CL/P patient with lip pits OR family member with lip pits: VWS UNTIL PROVEN OTHERWISE; "
                "IRF6 vs GRHL3: "
                "  IRF6 (VWS1): 70-80% of genetically solved VWS; test FIRST; "
                "  GRHL3 (VWS2): ~5% of VWS; test after IRF6 negative; "
                "  IDENTICAL CLINICAL PHENOTYPE: cannot distinguish IRF6 vs GRHL3 on clinical grounds; "
                "  MOLECULAR DISTINCTION ONLY; "
                "POPLITEAL PTERYGIUM SYNDROME (PPS) — IRF6 allelic: "
                "  PPS = VWS + popliteal webs + genital anomalies + ankyloblepharon; "
                "  PPS mutations: predominantly IRF6 missense in DNA-binding domain; "
                "  VWS mutations: missense + truncating throughout gene; "
                "  Same gene (IRF6) — different domain preference; "
                "VARIABLE EXPRESSIVITY (KEY COUNSELLING POINT): "
                "  SAME mutation in same family → "
                "    Member A: bilateral CL/P; "
                "    Member B: unilateral CL; "
                "    Member C: lip pits only; "
                "    Member D: apparently unaffected; "
                "  Cannot predict which phenotype for any given offspring; "
                "  All first-degree relatives should have lower lip examination; "
                "SURGICAL MANAGEMENT: "
                "  Cleft lip: millard/tennison repair 3-6 months; "
                "  Cleft palate: furlow/intravelar veloplasty 12-18 months; "
                "  Lip pit excision: only if symptomatic (mucus secretion/discharge); age 5+; "
                "  Speech therapy: VPI (velopharyngeal insufficiency) assessment post-repair; "
                "  Pharyngeal flap: if persistent VPI after palate repair"
            ),
        },
        {
            "term": "TP63 Syndromes — EEC vs AEC vs RHS vs LMS — Allele-Specific Genotype-Phenotype",
            "genes": ["TP63"],
            "definition": (
                "TP63 SYNDROME SPECTRUM — ALLELE-SPECIFIC GENOTYPE-PHENOTYPE: "
                "TP63 PROTEIN BIOLOGY: "
                "  P53 family: TP53, TP63, TP73; "
                "  TAp63 vs ΔNp63: alternative N-terminal transcription start; "
                "  ΔNp63α: dominant isoform in basal epithelium; dominant-negative over TP53/TP73; "
                "  Mutations → dominant-negative interference with AER (apical ectodermal ridge) + stratified epithelium; "
                "EEC SYNDROME (Ectrodactyly-Ectodermal Dysplasia-Clefting): "
                "  OMIM #604292; PATHOGNOMONIC TRIAD (all 3 must be present for full EEC): "
                "  (1) ECTRODACTYLY: split hand/foot; lobster-claw; absent central digits; "
                "  (2) ECTODERMAL DYSPLASIA: hypotrichosis, hypohidrosis, dental agenesis, lacrimal duct atresia; "
                "  (3) CLEFT LIP/PALATE: CL/P or CP; "
                "  MUTATION DOMAIN: DNA-BINDING DOMAIN missense (codons R204, R227, R279, R304); "
                "  DE NOVO: ~90%; familial rare; "
                "AEC SYNDROME (Ankyloblepharon-Ectodermal defects-Cleft): "
                "  OMIM #106260; also Hay-Wells syndrome; "
                "  MUTATION DOMAIN: SAM domain (C-terminal sterile alpha motif); "
                "  DISTINGUISHING FEATURES: "
                "    ANKYLOBLEPHARON: eyelid fusion strands (filiform bands); PATHOGNOMONIC for AEC; "
                "    SEVERE SKIN EROSIONS: scalp erosions at birth; chronic skin fragility; "
                "    Cleft palate; ectodermal dysplasia; "
                "    NO ectrodactyly (key DDx from EEC); "
                "RAPP-HODGKIN SYNDROME (RHS): "
                "  OMIM #129400; "
                "  Ectodermal dysplasia + cleft palate; "
                "  NO ectrodactyly; NO ankyloblepharon (DDx from EEC and AEC); "
                "  Mutations overlap EEC domain; "
                "LIMB-MAMMARY SYNDROME (LMS): "
                "  OMIM #603543; "
                "  Ectrodactyly + mammary gland hypoplasia/aplasia; "
                "  NO cleft palate (key DDx from EEC); "
                "  Isoleucine/leucine-rich domain mutations; "
                "ADULT SYNDROME: "
                "  Ectodermal dysplasia + lacrimal duct atresia + digital malformation; "
                "  Variable clefting; milder phenotype; "
                "ALLELE-GENOTYPE RULE: "
                "  p.Arg204, Arg227, Arg279, Arg304 mutations → EEC; "
                "  SAM domain (p.Asp541Asn, p.Ile537Val) → AEC; "
                "  SAME GENE, domain determines syndrome — critical for counselling; "
                "MANAGEMENT PRIORITIES: "
                "  EEC: ectrodactyly correction + ectodermal dysplasia (heat) + cleft repair; "
                "  AEC: skin wound care team (specialist) + eyelid separation + cleft repair; "
                "  ALL TP63: annual ophthalmology (keratitis, lacrimal); prosthetic dentistry"
            ),
        },
        {
            "term": "Marshall and Stickler Syndromes — COL11A1 vs COL2A1 and Vitreous Distinction",
            "genes": ["COL11A1"],
            "definition": (
                "COL11A1 / MARSHALL SYNDROME AND STICKLER TYPE 2 — COLLAGEN VITREORETINOPATHY: "
                "COLLAGEN XI BIOLOGY: "
                "  Type XI collagen: heterotrimer (COL11A1 + COL11A2 + COL2A1); "
                "  Function: regulates type II collagen fibril diameter in vitreous and cartilage; "
                "  COL11A1 LOF → abnormal vitreous collagen architecture; "
                "STICKLER SYNDROME TYPES: "
                "  TYPE 1 (COL2A1 — most common): "
                "    OPTICALLY EMPTY vitreous (membranous vitreous anomaly); "
                "    Most severe myopia (-6 to -30D); "
                "    Retinal detachment most common Stickler type; "
                "    AD; prevalent worldwide; "
                "  TYPE 2 (COL11A1): "
                "    BEADED VITREOUS (type 2 vitreous anomaly — strands/beads visible on slit-lamp); "
                "    Moderate myopia (-3 to -12D); "
                "    SNHL more prominent than Stickler 1; "
                "  TYPE 3 (COL11A2): NO ocular involvement (COL11A2 not in vitreous); deafness; "
                "  VITREOUS EXAMINATION: essential to distinguish types 1 and 2; "
                "MARSHALL SYNDROME vs STICKLER TYPE 2 (both COL11A1): "
                "  MARSHALL: "
                "    FLAT MIDFACE (malar hypoplasia) — pronounced; "
                "    LARGE EYE GLOBES (megalophthalmos): increased axial eye length visible clinically; "
                "    Very high myopia (>-10D); thick calvarium; "
                "    Mutation: predominantly exon 50 skipping → alters N-propeptide; "
                "  STICKLER 2: "
                "    Beaded vitreous; moderate myopia; SNHL; "
                "    Less distinctive facies than Marshall; "
                "    Mutations distributed throughout COL11A1; "
                "  CLINICAL SPECTRUM: considerable overlap; same family may show both; "
                "ROBIN SEQUENCE (MANDATORY RULE): "
                "  Micrognathia + cleft palate + airway obstruction = Pierre Robin sequence; "
                "  COL11A1 AND COL2A1 (Stickler) are IMPORTANT CAUSES of Robin sequence; "
                "  ALL Robin sequence neonates: ophthalmology assessment at birth (retinal exam + myopia); "
                "  If myopia or vitreous anomaly detected → Stickler confirmed → annual surveillance; "
                "  Missed diagnosis = catastrophic retinal detachment risk; "
                "RETINAL DETACHMENT MANAGEMENT: "
                "  Annual dilated fundus exam: look for lattice degeneration, retinal holes; "
                "  Prophylactic laser retinopexy: at first retinal break (do not wait for detachment); "
                "  Retinal detachment: vitreoretinal surgery; "
                "  Lifetime annual surveillance cannot be skipped — even mild Stickler cases"
            ),
        },
        {
            "term": "8-Gene Orofacial Clefting Differential Diagnosis Algorithm",
            "genes": ["IRF6", "GRHL3", "PVRL1", "MSX1", "TBX22", "SATB2", "TP63", "COL11A1"],
            "definition": (
                "8-GENE OROFACIAL CLEFTING DIAGNOSTIC ALGORITHM: "
                "STEP 1: LIP PITS? "
                "  YES → Van der Woude Syndrome: test IRF6 FIRST; if negative → GRHL3; "
                "  POPLITEAL WEBS too → PPS (IRF6 missense in DBD); "
                "STEP 2: ECTRODACTYLY (split hand/foot)? "
                "  YES → TP63 EEC syndrome; test DNA-binding domain hotspots (R204, R227, R279, R304); "
                "  Ectodermal dysplasia + cleft only (NO ectrodactyly) → TP63 RHS or AEC; "
                "  ANKYLOBLEPHARON (eyelid fusion strands) → TP63 AEC (SAM domain); "
                "STEP 3: TONGUE-TIE (ankyloglossia) in MALE with cleft palate? "
                "  YES → TBX22 CPX (X-linked); test TBX22; "
                "  Female relative with isolated tongue-tie → TBX22 carrier; "
                "STEP 4: SEVERE ID + ABSENT SPEECH + DENTAL ANOMALIES? "
                "  YES → SATB2/Glass syndrome; trio sequencing (de novo); CMA first (2q33 deletion); "
                "STEP 5: HIGH MYOPIA + SNHL + CLEFT PALATE? "
                "  YES → COL11A1 (Marshall/Stickler 2) or COL2A1 (Stickler 1); "
                "  Vitreous exam: optically empty = Stickler 1 (COL2A1); beaded = Stickler 2 (COL11A1); "
                "  Robin sequence neonate → ophthalmology IMMEDIATELY; "
                "STEP 6: TOOTH AGENESIS (missing premolars/incisors) + CL/P? "
                "  YES → MSX1 (OFC5); panoramic X-ray all family members; "
                "  Oligodontia + CL/P + ECTODERMAL SIGNS → PVRL1 (AR; Mediterranean); "
                "  OR TP63 (AD; ectrodactyly too); "
                "STEP 7: ISOLATED CL/P (no above features)? "
                "  → IRF6 hotspots (Arg84His/Cys); full IRF6 sequencing; "
                "  → GRHL3; MSX1; WNT10A; PAX9; "
                "BY INHERITANCE MODE: "
                "  AD LOF: IRF6 (VWS1), GRHL3 (VWS2), MSX1 (OFC5), SATB2 (de novo), COL11A1; "
                "  AD DN/GOF: TP63 (EEC/AEC/RHS); "
                "  X-linked recessive: TBX22 (CPX); "
                "  Autosomal recessive: PVRL1/NECTIN1 (CLPED1; Mediterranean founder)"
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Orofacial-Clefting-Atlas",
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
              f"iq={g['mean_iq']}, snhl={g['snhl_pct']}%, dental={g['dental_pct']}%")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Definition entries: {df['count']}")
    for d in df["definitions"]:
        print(f"  {d['term'][:80]}")
