#!/usr/bin/env python3
"""Hereditary-ICF-Syndrome-CBM-Signalosome-Combined-Immunodeficiency-Atlas -- Complete 8-Gene Reference
ICF Syndrome (DNA Methylation) + CBM Signalosome (CARD11-BCL10-MALT1) + CTPS1 Non-SCID CID
(DNMT3B/ICF1-Centromeric-Instability-Chr1-9-16-PATHOGNOMONIC-Facial-Hypogammaglobulinemia,
 ZBTB24/ICF2-BTB-POZ-Zinc-Finger-Milder-Neurological-Hypogammaglobulinemia,
 CDCA7/ICF3-Cell-Division-Cycle-Associated-7-Centromeric-Instability-B-Cell-Defect,
 HELLS/ICF4-Lymphoid-Helicase-Chromatin-Remodeling-Methylation-Maintenance,
 CARD11-LOF/AR-CID-CBM-Signalosome-Absent-Antibody-Deficiency-HSCT-URGENT-contrast-AD-GOF-BENTA,
 BCL10-LOF/AR-CID-CBM-B-Cell-T-Cell-Activation-Absent-Very-Rare-Antibody-Deficiency,
 MALT1-LOF/AR-Treg-Dysregulation-CID-Colitis-PATHOGNOMONIC-distinguish-IPEX-IL2RA,
 CTPS1-LOF/AR-T-Cell-Specific-CTP-Synthase-EBV-VZV-Varicella-Proliferation-Failure-PATHOGNOMONIC):

DNMT3B (DNA methyltransferase 3 beta; 853aa; 20q11.21; AR;
         DNMT3B encodes the de novo DNA methyltransferase 3β, responsible for establishing
           methylation at pericentromeric satellite DNA (Sat2 on chromosomes 1, 9, 16) during development;
         Without DNMT3B: pericentromeric satellite DNA repeats of chromosomes 1, 9, and 16 become
           hypomethylated → centromeric instability → classical ICF chromosome stretches and multiradial figures;
         ICF TYPE 1 (Immunodeficiency, Centromeric instability, Facial anomalies type 1):
         PATHOGNOMONIC: centromeric instability of chromosomes 1, 9, and 16 on PHA-stimulated lymphocyte
           karyotype — elongated, fused, or multiradial chromosome configurations at pericentromeric regions;
         FACIAL ANOMALIES: epicanthal folds, hypertelorism, flat nasal bridge, macroglossia,
           low-set ears, mild microcephaly — the F in ICF syndrome;
         IMMUNOLOGY: PROFOUND AGAMMAGLOBULINEMIA (absent IgG, IgA, IgE, IgM) in most patients;
           B cells present in peripheral blood but fail to differentiate into plasma cells —
           terminal B-cell differentiation defect NOT a B-cell development defect;
           CD4+ and CD8+ T cells numerically normal but functionally impaired;
         RECURRENT INFECTIONS: respiratory (pneumococcal pneumonia, recurrent otitis), GI (Giardia),
           persistent viral infections (CMV, EBV), Pneumocystis jirovecii in severe cases;
         NEUROLOGICAL: mild-to-moderate intellectual disability in most ICF1 patients;
         ICF1 = MOST COMMON ICF subtype (~50% of all ICF);
         DIAGNOSIS: centromere instability assay (karyotype on PHA-stimulated PBMCs) +
           DNMT3B sequencing; chromosomal instability pattern at 1/9/16 = DIAGNOSTIC;
         TREATMENT: IVIG replacement; prophylactic antibiotics; HSCT curative but rarely used;
           gene therapy experimental; seed SEED_BASE+0).
ZBTB24 (Zinc finger and BTB domain-containing protein 24; 570aa; 6q25.3; AR;
         ZBTB24 encodes a BTB/POZ domain zinc-finger transcription factor;
         ZBTB24 regulates expression of CDCA7 and possibly HELLS in the ICF methylation maintenance pathway;
         Without ZBTB24: CDCA7 expression reduced → impaired HELLS recruitment → DNA methylation
           maintenance fails at pericentromeric heterochromatin → centromeric instability (same chr 1,9,16);
         ICF TYPE 2 (Immunodeficiency, Centromeric instability, Facial anomalies type 2):
         PATHOGNOMONIC: same centromeric instability at chromosomes 1, 9, 16 as ICF1;
         MILDER NEUROLOGICAL FEATURES than ICF1: intellectual disability less severe or absent in ~30%;
         HYPOGAMMAGLOBULINEMIA: IgG markedly reduced, IgA absent, IgE absent — similar to ICF1;
           B cells present but differentiation block similar to DNMT3B;
         FACIAL ANOMALIES: similar to ICF1 (hypertelorism, flat nasal bridge, macroglossia);
         ICF2 = SECOND MOST COMMON ICF subtype (~30% of all ICF);
         DIFFERENTIAL: ICF1 vs ICF2 requires gene sequencing; chromosome instability pattern identical;
           ICF2 tends milder neurological involvement — key clinical DDx;
         DIAGNOSIS: centromeric instability assay + ZBTB24 sequencing; ICF gene panel covers all 4 subtypes;
         TREATMENT: IVIG; prophylactic antibiotics; HSCT for severe cases; seed SEED_BASE+1).
CDCA7  (Cell division cycle associated 7; 372aa; 2q31.1; AR;
         CDCA7 encodes Cell Division Cycle Associated 7, a zinc-finger containing nuclear protein;
         CDCA7 links ZBTB24 transcription factor regulation to HELLS helicase recruitment;
         CDCA7 is required for efficient HELLS (SMARCA6) binding to pericentromeric heterochromatin;
         Without CDCA7: HELLS cannot be efficiently recruited to methylation maintenance sites →
           pericentromeric DNA methylation defect → chromosomes 1, 9, 16 instability;
         ICF TYPE 3 (Immunodeficiency, Centromeric instability, Facial anomalies type 3):
         Very rare — fewer than 20 families described globally;
         PATHOGNOMONIC: same centromeric instability at chr 1, 9, 16;
         CLINICAL: same triad as ICF1/ICF2 — facial anomalies + hypogammaglobulinemia + centromeric instability;
         Variable neurological involvement;
         DIAGNOSIS: centromeric instability karyotype + ICF multigene panel (DNMT3B + ZBTB24 + CDCA7 + HELLS);
         CDCA7 mutations: missense + nonsense; biallelic required;
         TREATMENT: IVIG; supportive; HSCT considered in severe cases; seed SEED_BASE+2).
HELLS  (Helicase, lymphoid specific / LSH / SMARCA6; 838aa; 10q23.33; AR;
         HELLS encodes Helicase, Lymphoid Specific (LSH), also known as SMARCA6 / DDM1 in plants;
         HELLS is a SWI/SNF family DNA helicase that facilitates access of DNMT3B and maintenance
           methyltransferase DNMT1 to pericentromeric heterochromatin;
         HELLS physically interacts with CDCA7 at heterochromatic regions to preserve centromeric methylation;
         Without HELLS: DNMT3B and DNMT1 cannot efficiently methylate satellite repeats of chr 1, 9, 16;
         ICF TYPE 4 (Immunodeficiency, Centromeric instability, Facial anomalies type 4):
         Very rare — approximately 10 families described globally;
         PATHOGNOMONIC: centromeric instability same pattern as ICF1-ICF3;
         NAME SIGNIFICANCE: "Lymphoid Specific" reflects highest expression in lymphoid tissue —
           explains lymphoid predominance of the ICF phenotype;
         CLINICAL: facial anomalies + hypogammaglobulinemia + centromeric instability + variable intellectual disability;
         ICF4 pathway: ZBTB24 → CDCA7 → HELLS → recruits DNMT3B → methylates Sat2 → centromere stability;
           loss of ANY link in this chain = ICF syndrome;
         DIAGNOSIS: centromeric instability karyotype + multigene ICF panel;
         TREATMENT: IVIG; HSCT for severe cases; seed SEED_BASE+3).
CARD11 (Caspase recruitment domain family member 11 / CARMA1; 1154aa; 7p22.2; AR LOF (CID) / AD GOF (BENTA);
         CARD11 encodes the scaffold protein of the CBM signalosome complex (CARD11-BCL10-MALT1);
         CBM complex: downstream of TCR/BCR → PKCβ/θ phosphorylates CARD11 coiled-coil domain →
           CARD11 oligomerises → recruits BCL10 and MALT1 → activates IKKβ → NF-κB canonical pathway;
         TWO OPPOSITE DISEASE MECHANISMS FOR CARD11:
         (1) AR LOSS-OF-FUNCTION (LOF): biallelic CARD11 null mutations → PROFOUND COMBINED IMMUNODEFICIENCY;
             CARD11 LOF IMMUNODEFICIENCY: absent CBM signalosome assembly → no NF-κB activation after
               TCR or BCR ligation → T-cell proliferation failure + B-cell maturation arrest;
             AGAMMAGLOBULINEMIA: absent IgG, IgA, IgM — near-total antibody deficiency;
             T-CELL FUNCTION: T cells numerically normal but functionally anergic (no NF-κB activation);
             RECURRENT INFECTIONS: encapsulated bacteria, viral infections;
             HSCT CURATIVE AND URGENT — MANDATORY before irreversible T/B cell exhaustion;
         (2) AD GAIN-OF-FUNCTION (GOF): AD BENTA disease (B-cell expansion with NF-κB and T-cell anergy);
             BENTA: severe atopic dermatitis + elevated IgE + lymphoproliferation + recurrent infections;
             This PATHOGNOMONIC atopic phenotype = opposite of the LOF CID phenotype;
         KEY CLINICAL RULE: CARD11 LOF (AR) = profound CID → HSCT URGENT;
           CARD11 GOF (AD) = BENTA (atopy + lymphoproliferation) → venetoclax / MALT1 inhibitors;
         DIAGNOSIS: gene sequencing (biallelic LOF for CID; monoallelic GOF for BENTA);
         TREATMENT CID LOF: IVIG; HSCT; anti-infective prophylaxis; seed SEED_BASE+4).
BCL10  (B-cell lymphoma/leukemia 10; 233aa; 1p22.3; AR LOF;
         BCL10 encodes the scaffold protein BCL10, the central connector of the CBM signalosome;
         BCL10 contains a CARD domain (connects to CARD11) and a C-terminal Ser/Thr-rich region (connects to MALT1);
         BCL10 oligomerisation and filament formation is critical for CBM signalosome activation;
         CARD11 recruits BCL10 → BCL10 filaments recruit MALT1 → MALT1 activates IKKβ → NF-κB;
         AR BCL10 LOF IMMUNODEFICIENCY: very rare (fewer than 10 families globally);
         PROFOUND COMBINED IMMUNODEFICIENCY: both T-cell and B-cell CBM signalosome activation absent;
         AGAMMAGLOBULINEMIA: near-absent IgG, IgA, IgM — same terminal B-cell maturation failure as CARD11 LOF;
         T cells present but fail to activate normally after TCR stimulation;
         RECURRENT BACTERIAL AND VIRAL INFECTIONS from infancy;
         DISTINGUISHES FROM CARD11 LOF: requires sequencing as phenotype nearly identical;
         DISTINGUISHES FROM MALT1 LOF: BCL10 LOF = combined T+B defect;
           MALT1 LOF = prominent regulatory T-cell failure + colitis alongside CID;
         SOMATIC BCL10 LOF: in lymphoma (MALT lymphoma) — contrast germline AR LOF which causes CID;
         DIAGNOSIS: BCL10 sequencing after ruling out BTK (AGAMMA), CARD11, MALT1 on gene panel;
         TREATMENT: IVIG replacement; HSCT curative; anti-infective prophylaxis; seed SEED_BASE+5).
MALT1  (MALT lymphoma translocation gene 1 / mucosa-associated lymphoid tissue lymphoma
         translocation protein 1 / paracaspase; 922aa; 18q21.33; AR LOF;
         MALT1 encodes the paracaspase MALT1, the effector component of the CBM signalosome;
         MALT1 is the only known human paracaspase — it has both scaffold and protease (caspase-like) activity;
         CBM signalosome: CARD11 → BCL10 → MALT1 → IKKβ → NF-κB canonical;
         MALT1 protease domain cleaves: BCL10, CYLD, HOIL-1L, RelB → fine-tunes NF-κB output;
         MALT1 is CRITICAL for regulatory T-cell (Treg) homeostasis — Treg survival and function
           downstream of TCR stimulation requires intact CBM-MALT1 signaling;
         AR MALT1 LOF IMMUNODEFICIENCY:
         COMBINED IMMUNODEFICIENCY + REGULATORY T-CELL DEFECT:
           T cells present but functionally impaired; B-cell terminal differentiation defect;
           HYPOGAMMAGLOBULINEMIA: IgG, IgA absent/reduced;
         PATHOGNOMONIC TREG DEFECT: inflammatory colitis + skin inflammation + autoimmunity alongside CID —
           TREG DYSFUNCTION distinguishes MALT1 LOF from pure CBM signaling defect;
           Colitis resembles IPEX but Tregs are numerically present (dysfunction, not absence);
         DISTINGUISHES FROM CARD11 LOF: MALT1 LOF has more prominent autoimmune/colitis component;
           DISTINGUISHES FROM IPEX (FOXP3): Tregs present in number but non-functional in MALT1 LOF;
           DISTINGUISHES FROM IL2RA (CD25): different inflammatory profile;
         SOMATIC MALT1: fusion (API2-MALT1) in MALT lymphoma — contrast germline AR LOF;
         TREATMENT: IVIG; HSCT curative; anti-inflammatory for colitis; JAKi trial data limited;
         seed SEED_BASE+6).
CTPS1  (CTP synthase 1 / cytidine triphosphate synthase 1; 591aa; 1p34.1; AR;
         CTPS1 encodes CTP synthase 1, an enzyme that synthesises CTP (cytidine triphosphate) from UTP;
         CTP is essential for RNA, DNA, and phospholipid synthesis — every proliferating cell needs CTP;
         CTPS1 is HIGHLY EXPRESSED in activated T cells — T cells require massive CTP production
           for the rapid DNA replication needed during clonal expansion after antigen stimulation;
         CTPS2 (the paralog) is ubiquitously expressed and provides baseline CTP in non-immune cells;
         AR CTPS1 DEFICIENCY: T-CELL PROLIFERATION FAILURE SELECTIVELY AFTER VIRAL INFECTIONS;
         PATHOGNOMONIC FEATURE: SEVERE COMPLICATIONS SPECIFICALLY WITH EBV AND VARICELLA-ZOSTER VIRUS (VZV):
           EBV → fatal mononucleosis with lymphoproliferation (cannot control EBV because T cells cannot expand);
           VZV (varicella/chickenpox) → severe disseminated disease → potentially fatal;
           Other common infections (bacterial, other viral) tolerated relatively normally;
         MECHANISM: resting T cells use CTPS2 for basal CTP needs; but upon viral antigen stimulation,
           T cells switch to CTPS1-dependent high-capacity CTP synthesis for explosive proliferation;
           Without CTPS1: EBV/VZV-specific T-cell clonal expansion fails → viral immune evasion;
         PARADOX: patients appear relatively well between infections — this is the KEY diagnostic clue;
           between viral infections, immune function largely preserved; the SELECTIVE susceptibility
           to EBV and VZV is what distinguishes CTPS1 from other CIDs;
         POPULATIONS: founder effect in Liverpool-area UK families (p.R407W);
         B CELLS: IgG may be mildly reduced but not agammaglobulinemia;
         HSCT CURATIVE: HSCT reconstitutes T-cell CTPS1 expression and restores viral immunity;
         TREATMENT: IVIG (may not fully prevent EBV/VZV complications); prophylactic antiviral
           (aciclovir/valganciclovir for VZV prevention; monitor EBV viral load); HSCT curative;
         seed SEED_BASE+7).
"""

import random
import math

ATLAS_NAME   = "Hereditary-ICF-Syndrome-CBM-Signalosome-Combined-Immunodeficiency-Atlas"
SEED_BASE    = 3998   # seeds 3998-4005 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8

GENES = ["DNMT3B", "ZBTB24", "CDCA7", "HELLS", "CARD11", "BCL10", "MALT1", "CTPS1"]

GENE_DATA = {
    "DNMT3B": {
        "full_name":               "DNA methyltransferase 3 beta (DNMT3B; ICF type 1)",
        "aa":                      853,
        "locus":                   "20q11.21",
        "kda":                     "102",
        "inheritance":             "AR (autosomal recessive; biallelic DNMT3B mutations; both sexes equally affected)",
        "syndrome":                "ICF1 — Immunodeficiency, Centromeric instability, Facial anomalies type 1 (most common ICF ~50%)",
        "pathway":                 "DNMT3B absent → pericentromeric satellite DNA (Sat2) on chr 1,9,16 unmethylated → centromeric instability PATHOGNOMONIC → B-cell differentiation terminal defect → agammaglobulinemia",
        "onset_range":             (2, 60),     # infancy to early childhood
        "agamma_pct":              0.85,        # profound agammaglobulinemia
        "centromeric_instability_pct": 0.99,   # PATHOGNOMONIC on karyotype
        "facial_anomalies_pct":    0.90,        # facial anomalies present
        "intellectual_disability_pct": 0.70,   # mild-moderate ID
        "ivig_pct":                0.90,        # IVIG replacement
        "hsct_pct":                0.25,        # HSCT rarely used
        "recurrent_infect_pct":    0.95,        # recurrent infections
        "pathognomonic":           "CENTROMERIC INSTABILITY of chromosomes 1, 9, and 16 (elongated, fused, multiradial configurations) on PHA-stimulated lymphocyte karyotype = PATHOGNOMONIC for ICF syndrome; FACIAL ANOMALIES (hypertelorism, flat nasal bridge, macroglossia, epicanthal folds); PROFOUND AGAMMAGLOBULINEMIA with B cells PRESENT (terminal differentiation defect — not absent B cells)",
        "key_drug":                "IVIG (immunoglobulin replacement; 400-600 mg/kg every 3-4 weeks) — essential; prophylactic antibiotics (TMP-SMX for PCP, penicillin for encapsulated bacteria); HSCT curative but rarely offered due to overall severity; gene therapy experimental",
    },
    "ZBTB24": {
        "full_name":               "Zinc finger and BTB domain-containing protein 24 (ZBTB24; ICF type 2)",
        "aa":                      570,
        "locus":                   "6q25.3",
        "kda":                     "63",
        "inheritance":             "AR (autosomal recessive; biallelic ZBTB24 mutations; both sexes equally)",
        "syndrome":                "ICF2 — Immunodeficiency, Centromeric instability, Facial anomalies type 2 (second most common ICF ~30%)",
        "pathway":                 "ZBTB24 absent → CDCA7 expression reduced → HELLS recruitment impaired → DNMT3B cannot access Sat2 → same chr 1,9,16 centromeric instability as ICF1",
        "onset_range":             (3, 72),
        "agamma_pct":              0.82,
        "centromeric_instability_pct": 0.99,
        "facial_anomalies_pct":    0.85,
        "intellectual_disability_pct": 0.45,   # milder neurological than ICF1
        "ivig_pct":                0.88,
        "hsct_pct":                0.20,
        "recurrent_infect_pct":    0.92,
        "pathognomonic":           "SAME CENTROMERIC INSTABILITY as ICF1 (chr 1, 9, 16 on karyotype); MILDER NEUROLOGICAL FEATURES than ICF1 — intellectual disability less severe or absent in ~30% (key clinical DDx from ICF1); HYPOGAMMAGLOBULINEMIA with B cells present; BTB/POZ zinc-finger transcription factor — regulates CDCA7 expression upstream of HELLS",
        "key_drug":                "IVIG replacement; prophylactic antibiotics; centromere instability assay + gene panel for ICF typing; ICF1 vs ICF2 distinguished only by sequencing (karyotype pattern identical); HSCT for severe cases",
    },
    "CDCA7": {
        "full_name":               "Cell division cycle associated 7 (CDCA7; ICF type 3)",
        "aa":                      372,
        "locus":                   "2q31.1",
        "kda":                     "43",
        "inheritance":             "AR (autosomal recessive; biallelic CDCA7 mutations; very rare — <20 families globally)",
        "syndrome":                "ICF3 — Immunodeficiency, Centromeric instability, Facial anomalies type 3 (very rare; <20 families)",
        "pathway":                 "CDCA7 absent → HELLS cannot bind pericentromeric heterochromatin efficiently → DNMT3B methylation maintenance fails → chr 1,9,16 centromeric instability",
        "onset_range":             (3, 84),
        "agamma_pct":              0.80,
        "centromeric_instability_pct": 0.99,
        "facial_anomalies_pct":    0.82,
        "intellectual_disability_pct": 0.55,
        "ivig_pct":                0.85,
        "hsct_pct":                0.15,
        "recurrent_infect_pct":    0.90,
        "pathognomonic":           "SAME CENTROMERIC INSTABILITY at chr 1, 9, 16 as ICF1/ICF2; ZINC-FINGER DOMAIN protein — links ZBTB24 transcriptional regulation to HELLS helicase recruitment; FOUR ICF GENES ALL AFFECT SAME PATHWAY (ZBTB24→CDCA7→HELLS→DNMT3B→Sat2 methylation); ICF multigene panel mandatory to distinguish ICF subtypes",
        "key_drug":                "IVIG; prophylactic antibiotics; ICF4-gene panel (DNMT3B + ZBTB24 + CDCA7 + HELLS) for typing; HSCT considered for severe cases",
    },
    "HELLS": {
        "full_name":               "Helicase, lymphoid specific (HELLS / LSH / SMARCA6; ICF type 4)",
        "aa":                      838,
        "locus":                   "10q23.33",
        "kda":                     "97",
        "inheritance":             "AR (autosomal recessive; biallelic HELLS mutations; very rare — ~10 families globally)",
        "syndrome":                "ICF4 — Immunodeficiency, Centromeric instability, Facial anomalies type 4 (rarest ICF; ~10 families)",
        "pathway":                 "HELLS absent → DNMT3B/DNMT1 cannot access Sat2 at pericentromeric chr 1,9,16 → methylation fails → centromeric instability; HELLS = SWI/SNF chromatin remodeler allowing DNMT access to condensed heterochromatin",
        "onset_range":             (4, 96),
        "agamma_pct":              0.78,
        "centromeric_instability_pct": 0.99,
        "facial_anomalies_pct":    0.80,
        "intellectual_disability_pct": 0.58,
        "ivig_pct":                0.83,
        "hsct_pct":                0.18,
        "recurrent_infect_pct":    0.88,
        "pathognomonic":           "SAME CENTROMERIC INSTABILITY at chr 1, 9, 16; 'LYMPHOID SPECIFIC' name reflects predominant expression in lymphoid tissue — explains why immune system is specifically affected by loss of this ubiquitous chromatin remodeler; ICF4 PATHWAY POSITION: HELLS is the effector recruited by CDCA7 to allow DNMT3B access to pericentromeric DNA",
        "key_drug":                "IVIG; prophylactic antibiotics; ICF4-gene panel for typing; HSCT for severe cases; the pathway ZBTB24→CDCA7→HELLS→DNMT3B = four ICF genes, one methylation maintenance pathway",
    },
    "CARD11": {
        "full_name":               "Caspase recruitment domain family member 11 (CARMA1; CARD11-CID LOF; BENTA-GOF)",
        "aa":                      1154,
        "locus":                   "7p22.2",
        "kda":                     "133",
        "inheritance":             "AR (LOF = CID; biallelic null mutations) / AD (GOF = BENTA disease; gain-of-function)",
        "syndrome":                "CARD11 LOF → AR CID with profound antibody deficiency; CARD11 GOF → AD BENTA (atopy + lymphoproliferation) — OPPOSITE phenotypes from same gene",
        "pathway":                 "CBM signalosome: PKCβ/θ → CARD11 oligomerises → recruits BCL10 → MALT1 assembles → IKKβ activates → NF-κB; CARD11 LOF = no NF-κB from TCR/BCR → T-cell anergy + B-cell differentiation failure → agammaglobulinemia",
        "onset_range":             (2, 48),
        "agamma_pct":              0.88,
        "centromeric_instability_pct": 0.0,    # not an ICF feature
        "t_cell_anergy_pct":       0.85,        # functional T-cell anergy
        "hsct_pct":                0.70,        # HSCT urgent + mandatory
        "ivig_pct":                0.90,
        "recurrent_infect_pct":    0.95,
        "facial_anomalies_pct":    0.0,         # no facial anomalies (unlike ICF)
        "intellectual_disability_pct": 0.0,
        "pathognomonic":           "CARD11 LOF (AR CID): PROFOUND AGAMMAGLOBULINEMIA with T-CELL ANERGY — T cells numerically normal but cannot activate (NF-κB absent after TCR); HSCT URGENT AND MANDATORY — CARD11 LOF CID decompensates rapidly; KEY DDx: CARD11 GOF (AD BENTA) = OPPOSITE phenotype: elevated IgE + severe atopy + lymphoproliferation (not antibody deficiency); gene test mandatory to distinguish LOF vs GOF",
        "key_drug":                "HSCT URGENT AND MANDATORY for CARD11 LOF CID (NF-κB cannot be restored pharmacologically); IVIG bridge to HSCT; anti-infective prophylaxis; CARD11 GOF/BENTA: venetoclax (BCL-2i) + MALT1 inhibitors (JNJ-67856633); do NOT confuse CARD11 LOF with BENTA GOF — treatment is completely different",
    },
    "BCL10": {
        "full_name":               "B-cell lymphoma/leukemia 10 (BCL10; CBM signalosome connector)",
        "aa":                      233,
        "locus":                   "1p22.3",
        "kda":                     "26",
        "inheritance":             "AR (autosomal recessive LOF; biallelic; very rare — <10 families globally)",
        "syndrome":                "BCL10 LOF — Combined immunodeficiency with profound antibody deficiency from CBM signalosome disruption",
        "pathway":                 "BCL10 CARD domain binds CARD11 → BCL10 filaments nucleate → MALT1 recruited → NF-κB; BCL10 LOF = CARD11 cannot activate MALT1 → NF-κB absent from TCR/BCR → T-cell + B-cell activation failure",
        "onset_range":             (2, 60),
        "agamma_pct":              0.85,
        "centromeric_instability_pct": 0.0,
        "t_cell_anergy_pct":       0.82,
        "hsct_pct":                0.65,
        "ivig_pct":                0.88,
        "recurrent_infect_pct":    0.93,
        "facial_anomalies_pct":    0.0,
        "intellectual_disability_pct": 0.0,
        "pathognomonic":           "BCL10 LOF: SAME PHENOTYPE as CARD11 LOF (profound CID + agammaglobulinemia) — clinically indistinguishable without sequencing; CBM POSITION: BCL10 is the CENTRAL CONNECTOR of the CBM signalosome (CARD11 → BCL10 → MALT1); SOMATIC BCL10 LOF in MALT lymphoma (gastric, pulmonary) = DIFFERENT from germline AR LOF causing CID",
        "key_drug":                "HSCT curative; IVIG bridge; prophylactic antibiotics; BCL10 sequencing (gene panel after excluding BTK, CARD11, MALT1); distinction from MALT1 LOF: BCL10 LOF = pure CBM defect; MALT1 LOF = CBM defect + Treg dysfunction + colitis",
    },
    "MALT1": {
        "full_name":               "MALT lymphoma translocation gene 1 / paracaspase (MALT1; CBM effector)",
        "aa":                      922,
        "locus":                   "18q21.33",
        "kda":                     "106",
        "inheritance":             "AR (autosomal recessive LOF; biallelic MALT1 mutations; rare)",
        "syndrome":                "MALT1 LOF — Combined immunodeficiency with regulatory T-cell dysfunction and inflammatory colitis PATHOGNOMONIC",
        "pathway":                 "CBM: CARD11 → BCL10 → MALT1; MALT1 paracaspase cleaves BCL10/CYLD/HOIL-1L/RelB → fine-tunes NF-κB; MALT1 required for Treg TCR-induced NF-κB activation → Treg homeostasis; MALT1 LOF = CID + Treg dysfunction = combined T-cell + regulatory defect",
        "onset_range":             (1, 36),
        "agamma_pct":              0.80,
        "centromeric_instability_pct": 0.0,
        "t_cell_anergy_pct":       0.75,
        "colitis_pct":             0.75,        # PATHOGNOMONIC Treg defect → colitis
        "autoimmunity_pct":        0.60,        # skin + gut autoimmunity
        "hsct_pct":                0.70,
        "ivig_pct":                0.85,
        "recurrent_infect_pct":    0.90,
        "facial_anomalies_pct":    0.0,
        "intellectual_disability_pct": 0.0,
        "pathognomonic":           "MALT1 LOF UNIQUE FEATURE: INFLAMMATORY COLITIS + SKIN INFLAMMATION alongside CID — TREG DYSFUNCTION is the distinguishing feature; Tregs numerically PRESENT but DYSFUNCTIONAL (MALT1 is essential for Treg NF-κB activation after TCR stimulation); DDx from IPEX (FOXP3): Tregs numerically present in MALT1 LOF (vs absent/very low in IPEX); DDx from IL2RA: inflammatory profile different; CID + colitis + present-but-dysfunctional Tregs = MALT1 LOF signature",
        "key_drug":                "HSCT curative (corrects both CID and Treg dysfunction); IVIG; anti-inflammatory for colitis (steroids, anti-TNF — monitor for infections); MALT1 LOF CID → HSCT before progressive T/B decompensation; NOTE: MALT1 INHIBITORS (JNJ-67856633) used for MALT1 GOF/overactive (CARD11 GOF BENTA) — contraindicated in MALT1 LOF",
    },
    "CTPS1": {
        "full_name":               "CTP synthase 1 (CTPS1; T-cell nucleotide synthesis)",
        "aa":                      591,
        "locus":                   "1p34.1",
        "kda":                     "67",
        "inheritance":             "AR (autosomal recessive; biallelic CTPS1 mutations; founder variant p.R407W in UK/Liverpool families)",
        "syndrome":                "CTPS1 deficiency — T-cell proliferative failure selectively after EBV and VZV with relative sparing of other infections",
        "pathway":                 "CTPS1 synthesises CTP from UTP; activated T cells upregulate CTPS1 for explosive CTP production needed for rapid DNA replication during clonal expansion; CTPS1 absent → T cells cannot expand rapidly after viral antigen stimulation → EBV and VZV-specific T-cell clones cannot form",
        "onset_range":             (6, 120),
        "agamma_pct":              0.25,        # mild IgG reduction; not agammaglobulinemia
        "centromeric_instability_pct": 0.0,
        "t_cell_anergy_pct":       0.50,        # specifically after EBV/VZV
        "ebv_severe_pct":          0.80,        # EBV → fatal mononucleosis / lymphoproliferation
        "vzv_severe_pct":          0.70,        # VZV → severe disseminated varicella
        "btw_infections_well_pct": 0.75,        # relatively well between infections = KEY CLUE
        "hsct_pct":                0.60,
        "ivig_pct":                0.65,
        "recurrent_infect_pct":    0.60,        # selectively after EBV/VZV
        "facial_anomalies_pct":    0.0,
        "intellectual_disability_pct": 0.0,
        "pathognomonic":           "SEVERE COMPLICATIONS SPECIFICALLY WITH EBV (fatal mononucleosis, lymphoproliferation) AND VZV (disseminated varicella) — other infections tolerated relatively normally; PARADOX: patients appear RELATIVELY WELL BETWEEN INFECTIONS (T cells adequate for most functions; only viral antigen-driven RAPID PROLIFERATION fails); CTPS2 compensates for baseline CTP needs but CTPS1 uniquely needed for T-cell explosion response; UK FOUNDER VARIANT: p.R407W in Liverpool-area families",
        "key_drug":                "HSCT curative (restores CTPS1 in T cells → full viral immunity restored); prophylactic aciclovir/valganciclovir for VZV prevention MANDATORY; monitor EBV viral load closely; IVIG partial benefit; CTPS1 deficiency = consider in any child with disproportionately severe EBV/VZV + relative sparing of other infections",
    },
}


def _rng(seed):
    r = random.Random(seed)
    return r


def _aggregate_gene(gene, idx):
    gd = GENE_DATA[gene]
    seed = SEED_BASE + idx
    rng = _rng(seed)

    n = N_PER_GENE
    cohort = []
    for i in range(n):
        s = _rng(seed * 1000 + i)
        age_m = s.randint(*gd["onset_range"])
        sex = s.choice(["M", "F"])   # all AR → equal sex distribution
        cohort.append({
            "age_onset_months": age_m,
            "sex":               sex,
        })

    male_n = sum(1 for p in cohort if p["sex"] == "M")

    def _p(key):
        base = gd.get(key, 0.0)
        noise = rng.gauss(0, 0.03)
        return round(max(0.0, min(1.0, base + noise)), 3)

    return {
        "gene":                    gene,
        "full_name":               gd["full_name"],
        "aa":                      gd["aa"],
        "locus":                   gd["locus"],
        "kda":                     gd["kda"],
        "inheritance":             gd["inheritance"],
        "syndrome":                gd["syndrome"],
        "n":                       n,
        "male_n":                  male_n,
        "female_n":                n - male_n,
        "median_onset_months":     sorted(p["age_onset_months"] for p in cohort)[n // 2],
        "agamma_pct":              _p("agamma_pct"),
        "centromeric_instability_pct": _p("centromeric_instability_pct"),
        "facial_anomalies_pct":    _p("facial_anomalies_pct"),
        "ivig_pct":                _p("ivig_pct"),
        "hsct_pct":                _p("hsct_pct"),
        "recurrent_infect_pct":    _p("recurrent_infect_pct"),
        "pathognomonic":           gd["pathognomonic"],
        "key_drug":                gd["key_drug"],
        "seed":                    seed,
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    total_agamma    = round(sum(c["agamma_pct"]    for c in cohorts) / len(cohorts), 3)
    total_ivig      = round(sum(c["ivig_pct"]      for c in cohorts) / len(cohorts), 3)
    total_hsct      = round(sum(c["hsct_pct"]      for c in cohorts) / len(cohorts), 3)
    total_infect    = round(sum(c["recurrent_infect_pct"] for c in cohorts) / len(cohorts), 3)

    return {
        "atlas":    ATLAS_NAME,
        "genes":    GENES,
        "total_n":  TOTAL_N,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "agammaglobulinemia_pct":      total_agamma,
            "ivig_replacement_pct":        total_ivig,
            "hsct_pct":                    total_hsct,
            "recurrent_infections_pct":    total_infect,
        },
        "key_facts": [
            "DNMT3B/ICF1 (AR 20q11.21): CENTROMERIC INSTABILITY chromosomes 1, 9, 16 on karyotype = PATHOGNOMONIC for ICF; FACIAL ANOMALIES + profound AGAMMAGLOBULINEMIA (B cells present, terminal differentiation defect); most common ICF ~50%; intellectual disability mild-moderate; IVIG lifelong; HSCT for severe cases",
            "ZBTB24/ICF2 (AR 6q25.3): same centromeric instability as ICF1; MILDER NEUROLOGICAL FEATURES (ID less severe/absent ~30%) — KEY DDx from ICF1; BTB/POZ zinc-finger regulates CDCA7 upstream; hypogammaglobulinemia; second most common ICF ~30%; ICF1 vs ICF2 requires gene sequencing",
            "CDCA7/ICF3 (AR 2q31.1): same centromeric instability; CDCA7 links ZBTB24 to HELLS; very rare (<20 families); FOUR ICF GENES SHARE ONE PATHWAY: ZBTB24→CDCA7→HELLS→DNMT3B→Sat2 methylation; multigene ICF panel mandatory",
            "HELLS/ICF4 (AR 10q23.33): rarest ICF (~10 families); SWI/SNF helicase opens heterochromatin for DNMT3B access; 'Lymphoid Specific' name reflects highest lymphoid expression — explains selective immune phenotype; same pathway, same treatment as ICF1-ICF3",
            "CARD11 LOF (AR 7p22.2): PROFOUND CID + AGAMMAGLOBULINEMIA from CBM signalosome absence; HSCT URGENT AND MANDATORY; CRITICAL DDx: CARD11 GOF (AD BENTA) = elevated IgE + atopy + lymphoproliferation (OPPOSITE phenotype); confirm LOF vs GOF by sequencing before treatment",
            "BCL10 LOF (AR 1p22.3): SAME PHENOTYPE as CARD11 LOF (profound CID + agammaglobulinemia); CBM CENTRAL CONNECTOR (CARD11→BCL10→MALT1); somatic BCL10 LOF in MALT lymphoma = DIFFERENT from germline AR LOF CID; very rare (<10 families); HSCT curative",
            "MALT1 LOF (AR 18q21.33): CID + INFLAMMATORY COLITIS + TREG DYSFUNCTION = PATHOGNOMONIC TRIAD; Tregs NUMERICALLY PRESENT but DYSFUNCTIONAL (DDx from IPEX where Tregs absent); MALT1 paracaspase required for Treg NF-κB activation; MALT1 inhibitors CONTRAINDICATED in LOF (used for CARD11 GOF/BENTA instead)",
            "CTPS1 LOF (AR 1p34.1): SEVERE EBV AND VZV SPECIFICALLY (fatal mononucleosis, disseminated varicella) + RELATIVE SPARING OF OTHER INFECTIONS = PATHOGNOMONIC; patients appear well between viral infections (CTPS2 covers baseline CTP; CTPS1 needed only for T-cell clonal explosion); UK founder variant p.R407W; HSCT curative; prophylactic aciclovir MANDATORY",
        ],
        "icf_group":          ["DNMT3B", "ZBTB24", "CDCA7", "HELLS"],
        "cbm_group":          ["CARD11", "BCL10", "MALT1"],
        "ctps1_group":        ["CTPS1"],
        "agamma_group":       ["DNMT3B", "ZBTB24", "CDCA7", "HELLS", "CARD11", "BCL10", "MALT1"],
        "hsct_urgent_group":  ["CARD11", "BCL10", "MALT1", "CTPS1"],
        "cohort_summaries":   cohorts,
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    diagnostic_groups = {
        "icf_syndrome_dna_methylation": {
            "genes":          ["DNMT3B", "ZBTB24", "CDCA7", "HELLS"],
            "mechanism":      "DNA methylation maintenance failure at pericentromeric satellite DNA (Sat2) on chromosomes 1, 9, 16 → centromeric instability PATHOGNOMONIC → B-cell terminal differentiation defect → profound agammaglobulinemia",
            "shared_feature": "Centromeric instability chr 1,9,16 on karyotype (PHA-stimulated lymphocytes); facial anomalies; profound agammaglobulinemia (B cells present); recurrent infections; IVIG lifelong",
            "key_ddx":        "ICF1 (DNMT3B): most common ~50%; moderate-severe intellectual disability. ICF2 (ZBTB24): ~30%; MILDER neurological features — key DDx from ICF1 is neurological severity. ICF3/ICF4 (CDCA7/HELLS): very rare; pathway enzymes downstream of ZBTB24. ALL FOUR: identical karyotype pattern; require multigene panel to distinguish. PATHWAY ORDER: ZBTB24 (transcription factor) → CDCA7 (recruiter) → HELLS (helicase/chromatin remodeler) → DNMT3B (de novo methyltransferase) → Sat2 methylation.",
        },
        "cbm_signalosome_cid": {
            "genes":          ["CARD11", "BCL10", "MALT1"],
            "mechanism":      "CBM (CARD11-BCL10-MALT1) signalosome absent → no NF-κB activation after TCR/BCR ligation → T-cell anergy + B-cell differentiation failure → profound CID with antibody deficiency",
            "shared_feature": "AR; profound agammaglobulinemia; T-cell numerical presence but functional anergy; HSCT curative and urgent; recurrent bacterial + viral infections from infancy",
            "key_ddx":        "CARD11 LOF: HSCT urgent mandatory; DDx CARD11 GOF (BENTA = opposite phenotype: elevated IgE + atopy + lymphoproliferation — AD not AR). BCL10 LOF: same phenotype as CARD11 LOF; pure CBM defect. MALT1 LOF: CID + INFLAMMATORY COLITIS + TREG DYSFUNCTION distinguishes from CARD11/BCL10 LOF; Tregs present but non-functional; MALT1 inhibitors CONTRAINDICATED in LOF (used for GOF states instead).",
        },
        "ctps1_viral_proliferation": {
            "genes":          ["CTPS1"],
            "mechanism":      "CTPS1 absent → activated T cells cannot upregulate CTP synthesis for rapid proliferation → EBV/VZV-specific T-cell clonal expansion fails selectively",
            "shared_feature": "AR; disproportionately severe EBV (fatal mononucleosis) and VZV (disseminated varicella) vs relatively preserved immunity to other pathogens; relatively well between viral infections",
            "key_ddx":        "CTPS1 vs XLP1/XIAP: CTPS1 has no NKT absence (NKT cells present); no intracellular flow protein deficiency; CTPS1 vs RASGRP1: no autoimmunity overlap; CTPS1 is unique: T cells CANNOT RAPIDLY PROLIFERATE specifically for EBV/VZV antigens (not a broad CID). Functional assay: impaired T-cell proliferation in CFSE assay after stimulation.",
        },
    }

    pathognomonic_features = {
        "icf_karyotype_centromeric_instability": "ALL 4 ICF GENES: CENTROMERIC INSTABILITY on PHA-stimulated lymphocyte karyotype — elongated, decondensed, fused, or multiradial chromosome configurations at pericentromeric regions of chromosomes 1, 9, and 16 SPECIFICALLY (not other chromosomes); performed by requesting cytogenetics lab to culture PBMCs with PHA + check pericentromeric regions of chr 1, 9, 16; this assay CONFIRMS ICF syndrome before gene sequencing; no other PID has this karyotype pattern",
        "icf_facial_anomalies_triad":             "ICF FACIAL ANOMALIES (the 'F' in ICF): hypertelorism (widely spaced eyes), flat/depressed nasal bridge, macroglossia, epicanthal folds, low-set ears, mild microcephaly — these facial features + agammaglobulinemia + centromeric instability = complete ICF TRIAD; facial anomalies present in ~80-90% of ICF patients",
        "icf_b_cells_present_agamma":             "ICF UNIQUE IMMUNOLOGICAL PARADOX: PROFOUND AGAMMAGLOBULINEMIA with B cells PRESENT in normal numbers — this distinguishes ICF from BTK/agammaglobulinemia where B cells are ABSENT; ICF affects terminal B-cell differentiation (B cell → plasma cell step) not B-cell development; check B cells by flow cytometry before ordering BTK — B cells present in ICF",
        "icf_zbtb24_milder_neuro":                "ZBTB24/ICF2 VS DNMT3B/ICF1 NEUROLOGICAL DISTINGUISHER: ICF1 (DNMT3B) = moderate-severe intellectual disability in ~70%; ICF2 (ZBTB24) = mild or NO intellectual disability in ~30% of patients (the BTB/POZ zinc finger has a less severe neurological effect via its regulatory position above CDCA7); karyotype identical — gene sequencing required to distinguish",
        "card11_lof_vs_gof_critical_ddx":         "CARD11 LOF vs GOF = CRITICAL CLINICAL DISTINCTION FROM SAME GENE: LOF (AR biallelic): profound CID + agammaglobulinemia → HSCT urgent mandatory; GOF (AD monoallelic BENTA): elevated IgE + severe atopic dermatitis + lymphoproliferation + recurrent infections → venetoclax + MALT1 inhibitors; these are OPPOSITE DISEASES sharing the CARD11 gene; never assume AR autosomal disease without confirming biallelic LOF vs de novo GOF",
        "malt1_treg_colitis_pathognomonic":        "MALT1 LOF TREG DYSFUNCTION: Tregs are NUMERICALLY PRESENT but DYSFUNCTIONAL — MALT1 paracaspase is required for TCR-driven NF-κB activation in Tregs; without MALT1, Tregs cannot suppress autoreactive T cells → INFLAMMATORY COLITIS + skin autoimmunity alongside CID; DDx from IPEX (FOXP3 = Tregs absent) and IL2RA (CD25 deficiency = reduced T-reg + Teff both affected); MALT1 LOF = colitis + CID + Tregs present numerically = MALT1 LOF signature",
        "ctps1_ebv_vzv_specific":                 "CTPS1 PARADOX OF SELECTIVE VIRAL SUSCEPTIBILITY: EBV → fatal infectious mononucleosis + lymphoproliferation; VZV → severe disseminated varicella; OTHER INFECTIONS → RELATIVELY WELL; mechanism: CTPS2 covers baseline CTP needs (why patients are well between infections); CTPS1 uniquely needed for the explosive 100-1000-fold T-cell clonal expansion triggered by viral antigens; between EBV/VZV, T-cell functions adequate via CTPS2; the selectivity itself is PATHOGNOMONIC",
        "all_ar_no_sex_bias":                      "ALL 8 GENES IN THIS ATLAS ARE AUTOSOMAL RECESSIVE: equal sex distribution for all 8 diseases; no X-linked disease in this atlas; contrast with EBV-XLP atlas (XLR SH2D1A/XIAP/MAGT1) — all ICF + CBM + CTPS1 genes are autosomal; consanguinity is a risk factor for all 8; male-only disease presentation should prompt reconsideration of the diagnosis",
        "hsct_cbm_ctps1":                          "HSCT IS CURATIVE FOR CBM SIGNALOSOME (CARD11/BCL10/MALT1 LOF) AND CTPS1 DEFICIENCY: for CARD11 and BCL10 LOF — proceed to HSCT urgently before T/B cell exhaustion from repeated infections; for MALT1 LOF — HSCT corrects both CID AND Treg dysfunction (not just antibody deficiency); for CTPS1 — HSCT reconstitutes T-cell CTPS1 expression restoring viral proliferative capacity; ICF (DNMT3B/ZBTB24/CDCA7/HELLS) — HSCT rarely used due to overall severity of the multi-system disease but corrects immune component",
    }

    return {
        "gene_cohorts":            cohorts,
        "diagnostic_groups":       diagnostic_groups,
        "pathognomonic_features":  pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: Child with profound agammaglobulinemia → check B-cell count by flow cytometry FIRST: if B cells ABSENT → BTK/XLA (male); if B cells PRESENT → DOES NOT rule out ICF, CVID, or CBM CID",
            "Step 2: Agammaglobulinemia + B cells PRESENT → order PHA-stimulated lymphocyte KARYOTYPE specifically examining chromosomes 1, 9, and 16 pericentromeric regions → centromeric instability = ICF CONFIRMED",
            "Step 3: ICF karyotype confirmed → order FACIAL EXAMINATION (hypertelorism, flat nasal bridge, macroglossia) + neurodevelopmental assessment + ICF4-gene panel (DNMT3B + ZBTB24 + CDCA7 + HELLS)",
            "Step 4: DNMT3B pathogenic biallelic → ICF1; ZBTB24 → ICF2 (check for milder neurological features); CDCA7 → ICF3; HELLS → ICF4; START IVIG for all ICF subtypes; consider HSCT referral for severe cases",
            "Step 5: Agammaglobulinemia + B cells PRESENT + NORMAL KARYOTYPE → order CBM signalosome panel: CARD11 (distinguish biallelic LOF vs monoallelic GOF) + BCL10 + MALT1",
            "Step 6: CARD11 biallelic LOF → AR CID (HSCT URGENT); CARD11 monoallelic GOF → BENTA (venetoclax/MALT1i); BCL10 biallelic LOF → AR CID (HSCT); MALT1 biallelic LOF → CID + colitis (HSCT + anti-inflammatory)",
            "Step 7: MALT1 LOF identified → check for INFLAMMATORY COLITIS and assess TREG FUNCTION (Tregs numerically present? Do they suppress? MALT1 LOF = numerically present but functionally defective Tregs); DDx from IPEX (FOXP3 sequencing if Tregs absent/very low)",
            "Step 8: Child with severe EBV (fatal mononucleosis/lymphoproliferation) or severe disseminated VZV but RELATIVELY WELL BETWEEN INFECTIONS → test CTPS1; order T-cell proliferation assay (CFSE dilution after PHA stimulation); check for UK/Liverpool family origin (p.R407W founder)",
            "Step 9: CTPS1 LOF confirmed → PROPHYLACTIC ACICLOVIR/VALGANCICLOVIR immediately; monitor EBV viral load; HSCT referral; IVIG partial benefit; explain to family: CTPS1 deficiency is SELECTIVE viral CID, not broad CID",
            "Step 10: All 8 genes negative + profound antibody deficiency + B cells present → extended WES; consider IGHM/IGLL1/BLNK (agammaglobulinemia with B-cell maturation arrest) vs CVID genes (TNFRSF13C/BAFF-R, TNFRSF13B/TACI, NFKB1, NFKB2) vs novel ICF/CID genes",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "ICF Syndrome (Immunodeficiency, Centromeric instability, Facial anomalies)", "definition": "Rare AR combined immunodeficiency defined by THREE features: (1) IMMUNODEFICIENCY — profound agammaglobulinemia (absent IgG, IgA, IgE, IgM) with B cells present in normal numbers (terminal differentiation defect); (2) CENTROMERIC INSTABILITY — elongated, decondensed, fused, or multiradial configurations at pericentromeric regions of chromosomes 1, 9, and 16 specifically, on PHA-stimulated lymphocyte karyotype; (3) FACIAL ANOMALIES — hypertelorism, flat nasal bridge, macroglossia, epicanthal folds, low-set ears. Four genes cause ICF: DNMT3B (ICF1, ~50%), ZBTB24 (ICF2, ~30%), CDCA7 (ICF3, very rare), HELLS (ICF4, very rare). All four genes function in a single pathway: ZBTB24→CDCA7→HELLS→DNMT3B→pericentromeric Sat2 methylation. Treatment: IVIG lifelong; HSCT for severe cases."},
            {"term": "DNMT3B/ICF1 Deficiency", "definition": "AR DNMT3B (20q11.21; 853aa). DNMT3B (DNA methyltransferase 3β) is the de novo methyltransferase responsible for establishing pericentromeric satellite DNA methylation during development. ICF1 = most common ICF (~50%). PATHOGNOMONIC: centromeric instability on karyotype at chr 1, 9, 16 (Sat2 repeats). PROFOUND AGAMMAGLOBULINEMIA + B cells PRESENT (terminal differentiation defect — not B-cell absence). FACIAL ANOMALIES. Intellectual disability moderate in ~70%. IVIG lifelong; HSCT for severe cases."},
            {"term": "ZBTB24/ICF2 Deficiency", "definition": "AR ZBTB24 (6q25.3; 570aa). ZBTB24 is a BTB/POZ domain zinc-finger transcription factor that regulates CDCA7 expression upstream in the ICF methylation pathway. ICF2 = second most common (~30%). SAME KARYOTYPE PATTERN as ICF1 (chr 1, 9, 16 instability). KEY DISTINGUISHER: MILDER NEUROLOGICAL FEATURES — intellectual disability absent or mild in ~30% of ICF2 patients (vs moderate-severe in ICF1). Hypogammaglobulinemia similar to ICF1. Requires gene sequencing (not karyotype) to distinguish ICF subtypes."},
            {"term": "CDCA7/ICF3 Deficiency", "definition": "AR CDCA7 (2q31.1; 372aa). CDCA7 is a zinc-finger protein that links ZBTB24 transcriptional regulation to HELLS recruitment at pericentromeric heterochromatin. ICF3 = very rare (<20 families). Same centromeric instability at chr 1, 9, 16. Pathway position: ZBTB24 regulates CDCA7 expression → CDCA7 recruits HELLS → HELLS allows DNMT3B access. ICF multigene panel (DNMT3B + ZBTB24 + CDCA7 + HELLS) required for diagnosis."},
            {"term": "HELLS/ICF4 Deficiency", "definition": "AR HELLS/SMARCA6 (10q23.33; 838aa). HELLS (Helicase, Lymphoid Specific) is a SWI/SNF family chromatin remodeler that opens condensed pericentromeric heterochromatin to allow DNMT3B and DNMT1 access for methylation maintenance. ICF4 = rarest ICF (~10 families). Same karyotype pattern. The 'Lymphoid Specific' name reflects highest expression in lymphoid tissue — explains the selective immune phenotype despite ubiquitous chromatin remodeling requirement. HELLS interacts physically with CDCA7 at heterochromatic loci."},
            {"term": "CBM Signalosome (CARD11-BCL10-MALT1)", "definition": "The CBM signalosome is a T-cell and B-cell signaling complex assembled downstream of TCR/BCR activation: PKCθ (T cells) or PKCβ (B cells) phosphorylates CARD11 coiled-coil → CARD11 oligomerises into a platform → recruits BCL10 via CARD domain interactions → BCL10 filaments nucleate and recruit MALT1 paracaspase → MALT1 activates IKKβ → NF-κB canonical pathway activation. All three CBM components (CARD11, BCL10, MALT1) have loss-of-function mutations causing AR combined immunodeficiency with profound antibody deficiency (HSCT curative and urgent for all). CARD11 and MALT1 also have gain-of-function disease states (BENTA for CARD11 GOF; lymphoma for MALT1 overactivation)."},
            {"term": "CARD11 LOF CID vs BENTA (GOF)", "definition": "CRITICAL SAME-GENE OPPOSITE-DISEASE DISTINCTION: CARD11 LOF (AR biallelic) = profound CID with agammaglobulinemia + T-cell anergy → HSCT urgent mandatory (NF-κB from TCR/BCR is absent — cannot be pharmacologically restored). CARD11 GOF (AD monoallelic, especially exon 5-7 coiled-coil mutations) = BENTA disease: B-cell expansion with NF-κB and T-cell anergy — elevated IgE + severe atopic dermatitis + lymphoproliferation + recurrent infections + NF-κB constitutive → venetoclax (BCL-2) + MALT1 inhibitors (JNJ-67856633). ALWAYS confirm LOF vs GOF status by sequencing (phased or trio analysis) before planning treatment."},
            {"term": "BCL10 LOF Deficiency", "definition": "AR BCL10 (1p22.3; 233aa). BCL10 (B-cell lymphoma 10) is the central CBM connector: its CARD domain binds CARD11, and its Ser/Thr-rich C-terminus recruits MALT1. BCL10 forms filamentous structures essential for signalosome amplification. BCL10 LOF = same profound CID as CARD11 LOF (agammaglobulinemia + T-cell anergy) but distinguished only by sequencing. Very rare (<10 families). NOTE: somatic BCL10 LOF in MALT lymphoma (API2-BCL10 fusion from t(11;18)) is a DIFFERENT molecular event causing lymphoma, not CID."},
            {"term": "MALT1 LOF Deficiency", "definition": "AR MALT1 (18q21.33; 922aa). MALT1 is the only human paracaspase — both scaffold AND protease (cleaves BCL10, CYLD, HOIL-1L, RelB). MALT1 is essential for Treg TCR-driven NF-κB activation, not just T-effector/B-cell activation. MALT1 LOF = CID + INFLAMMATORY COLITIS + SKIN INFLAMMATION from TREG DYSFUNCTION. Tregs NUMERICALLY PRESENT but functionally defective (DDx from IPEX where Tregs absent). MALT1 INHIBITORS (JNJ-67856633) are being developed for overactive MALT1 states (CARD11 GOF, MALT lymphoma) — CONTRAINDICATED in MALT1 LOF (no protease to inhibit)."},
            {"term": "CTPS1 Deficiency (CTP Synthase 1)", "definition": "AR CTPS1 (1p34.1; 591aa). CTPS1 is the rate-limiting enzyme for CTP synthesis in activated T cells. Resting T cells use CTPS2 (paralog) for baseline CTP; antigen-stimulated T cells require CTPS1 for the 100-1000-fold upregulation of CTP synthesis needed for rapid proliferation. CTPS1 LOF = selective failure of antigen-driven T-cell clonal expansion. PATHOGNOMONIC: fatal EBV mononucleosis and severe disseminated VZV (varicella) with relatively spared immunity to other infections. PARADOX: patients appear relatively well between viral infections. UK founder variant: p.R407W. HSCT curative; prophylactic aciclovir mandatory."},
            {"term": "Centromeric Instability Assay (ICF Diagnosis)", "definition": "DIAGNOSTIC PROTOCOL FOR ICF: Collect peripheral blood; separate PBMCs; culture with PHA (phytohaemagglutinin, a T-cell mitogen) for 72 hours; prepare metaphase chromosomes; G-banding karyotype; EXAMINE pericentromeric regions of chromosomes 1, 9, and 16 specifically; ICF = elongated pericentromeric heterochromatin (stretches/decondensations), fused pericentromeric regions between two chromosomes, or multiradial figures (3+ chromosomes fused at pericentromeric regions). NOTE: other chromosomes may have occasional breaks (not ICF-specific); chr 1/9/16 SPECIFICALLY is the ICF signature. Normal karyotype rules out ICF; confirm ICF with multigene panel (DNMT3B + ZBTB24 + CDCA7 + HELLS)."},
            {"term": "Cascade Testing Strategy — Non-SCID AR CID with Antibody Deficiency", "definition": "STEP 1: Agammaglobulinemia → B-cell count by flow (CD19+): absent B cells → BTK/XLA panel; present B cells → ICF / CBM / CTPS1 panel. STEP 2: Order PHA-stimulated lymphocyte karyotype (chr 1/9/16 pericentromeric check). STEP 3: Centromeric instability → ICF panel (DNMT3B + ZBTB24 + CDCA7 + HELLS). Normal karyotype → CBM panel (CARD11 + BCL10 + MALT1) + CTPS1. STEP 4: CARD11 — ALWAYS distinguish biallelic LOF (CID → HSCT) vs monoallelic GOF (BENTA → venetoclax). STEP 5: MALT1 LOF → check for colitis and Treg function (if colitis + present Tregs = MALT1 LOF). STEP 6: Severe EBV/VZV specifically → CTPS1; check T-cell proliferation assay; prophylactic aciclovir immediately."},
        ],
        "key_distinguishers": {
            "b_cells_present_vs_absent":      "B cells PRESENT (agamma with B cells): ICF (DNMT3B/ZBTB24/CDCA7/HELLS), CARD11 LOF, BCL10 LOF, MALT1 LOF. B cells ABSENT (agamma without B cells): BTK/XLA, IGHM/IGLL1/BLNK (agammaglobulinemia genes). ICF + CBM CID must always be distinguished from classic BTK agammaglobulinemia.",
            "centromeric_instability_unique":  "CENTROMERIC INSTABILITY at chr 1/9/16 is EXCLUSIVE to ICF syndrome (all 4 subtypes) — no other PID has this karyotype finding; positive instability assay = ICF confirmed; negative = proceed to CBM/CTPS1 panel.",
            "intellectual_disability_icf":     "Intellectual disability present: ICF1 (DNMT3B, ~70%), ICF3 (CDCA7, ~55%), ICF4 (HELLS, ~58%), ICF2 (ZBTB24, ~45% — milder). No intellectual disability: CARD11/BCL10/MALT1/CTPS1 — absence of ID in a CID patient with B cells present favors CBM/CTPS1 over ICF.",
            "colitis_malt1_marker":            "INFLAMMATORY COLITIS alongside CID: strongly suggests MALT1 LOF (Treg dysfunction); DDx IPEX (FOXP3 — check Treg count; absent = FOXP3); LRBA (LRBA — Evans syndrome + CTLA4 expression); IL2RA (CD25 deficiency — regulatory + effector T both affected). MALT1 LOF unique: Tregs numerically present but functionally blocked.",
            "selective_ebv_vzv_ctps1":         "Disproportionately severe EBV + VZV with relative preservation of other immunity → CTPS1; no other gene in this atlas shows this selectivity; NKT cells and NK cells normal; no centromeric instability; no agammaglobulinemia (only mild IgG reduction).",
            "card11_gof_vs_lof_crucial":       "CARD11 GOF (AD BENTA) vs LOF (AR CID): check family history (AD = parent likely affected); check IgE level (GOF = very elevated IgE; LOF = absent IgG/IgA/IgE); check B-cell count (both have B cells present); ALWAYS SEQUENCE CARD11 with phase information before treating.",
        },
        "treatment_matrix": {
            "ivig":              "ICF1-4 + CARD11 LOF + BCL10 LOF + MALT1 LOF + CTPS1: IVIG (400-600 mg/kg every 3-4 weeks) for all with significant hypogammaglobulinemia; bridges to HSCT or provides lifelong protection in ICF where HSCT deferred.",
            "hsct":              "CARD11 LOF + BCL10 LOF: HSCT urgent and mandatory. MALT1 LOF: HSCT urgent (corrects CID AND Treg dysfunction). CTPS1: HSCT curative (restores T-cell CTPS1 expression). ICF1-4: HSCT for severe cases (rarely used due to multi-system disease complexity); corrects immune component.",
            "prophylactic_antiviral": "CTPS1: prophylactic aciclovir/valganciclovir MANDATORY (VZV prevention); monitor EBV viral load monthly. MALT1 LOF: antiviral coverage for CMV/EBV during immunosuppressive therapy for colitis.",
            "anti_inflammatory": "MALT1 LOF: corticosteroids + anti-TNF (infliximab) for colitis with concurrent anti-infective coverage; monitor for opportunistic infections during immunosuppression.",
            "malt1_inhibitors_contraindicated": "MALT1 inhibitors (JNJ-67856633, MI-2) = CONTRAINDICATED in MALT1 LOF deficiency — these drugs inhibit MALT1 activity; in MALT1 LOF there is already NO MALT1 activity; these are indicated only for MALT1 overactivity states (CARD11 GOF BENTA, MALT lymphoma, DLBCL).",
            "venetoclax_card11_gof":             "Venetoclax (BCL-2 inhibitor) + obinutuzumab: for CARD11 GOF (BENTA) lymphoproliferation — NOT for CARD11 LOF CID; confirm GOF/LOF status before using BCL-2 inhibitors.",
        },
    }
