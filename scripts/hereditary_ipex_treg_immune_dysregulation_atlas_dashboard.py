#!/usr/bin/env python3
"""Hereditary-IPEX-Treg-Immune-Dysregulation-Atlas -- Complete 8-Gene Reference
Regulatory T-Cell (Treg) Immune Dysregulation Diseases
(FOXP3/IPEX-Neonatal-Triad-HSCT-Curative-XLR-Treg-Absent,
 CTLA4/Haploinsufficiency-AD-LOF-Lymphoproliferation-CMV-EBV-Abatacept-First-Line,
 LRBA/AR-CTLA4-Lysosomal-Degradation-Phenocopy-CTLA4-Haploinsufficiency-Abatacept-Curative-Evans-Syndrome,
 STAT3-GOF/AD-GOF-Multi-Organ-Autoimmunity-JAK-Inhibitor-Ruxolitinib-Treg-Dysfunction,
 IL2RA/CD25-Deficiency-AR-Lymphoproliferation-Severe-IBD-Paradoxical-Expansion,
 AIRE/APECED-APS-1-AR-Mucocutaneous-Candidiasis-Addison-Hypoparathyroidism-Triad-anti-IFN-Autoantibodies,
 CARD11-GOF/BENTA-Disease-AD-GOF-Severe-Atopic-Dermatitis-Eosinophilia-IgE-Elevated-Dupilumab,
 BACH2/BRIDA-AD-LOF-Lymphoproliferation-Enteropathy-Hypogammaglobulinemia-Treg-Memory-B-Defect):

FOXP3  (Forkhead box P3; 256aa; Xp11.23; XLR;
         FOXP3 is the master transcription factor of regulatory T cells (Tregs);
         XLR → only males affected (classically); female carriers rarely manifesting;
         IPEX syndrome (Immune dysregulation Polyendocrinopathy Enteropathy X-linked);
         NEONATAL TRIAD PATHOGNOMONIC: (1) ENTEROPATHY (watery/bloody diarrhoea, severe malabsorption),
           (2) TYPE-1 DIABETES MELLITUS (neonatal autoimmune, day 1-weeks of life),
           (3) ECZEMA (severe atopic dermatitis from birth);
         Without treatment: DEATH before 2 years of age (infections, malnutrition, multi-organ failure);
         FOXP3-Treg absent/dysfunctional on flow cytometry (CD4+CD25+FoxP3+ Tregs markedly reduced);
         Additional features: thyroiditis, haemolytic anaemia, thrombocytopenia, nephropathy;
         HSCT IS CURATIVE — allogeneic HSCT the only definitive therapy; perform as early as possible;
         Bridge therapy: immunosuppression (sirolimus/tacrolimus preferred over steroids which damage gut);
         seed SEED_BASE+0).
CTLA4  (Cytotoxic T-lymphocyte antigen 4; 223aa; 2q33.2; AD-LOF (haploinsufficiency);
         CTLA4 is the inhibitory checkpoint receptor on T cells — constitutively expressed on Tregs;
         Normally CTLA4 removes CD80/CD86 from APCs by trans-endocytosis (reducing co-stimulation);
         AD-LOF haploinsufficiency → insufficient CTLA4 function → unrestrained T cell activation;
         CTLA4 Haploinsufficiency (also called CTLA4-associated regulatory T cell deficiency);
         Phenotype: lymphoproliferation (lymphadenopathy/splenomegaly/organomegaly) +
           AUTOIMMUNITY (cytopenias/IBD/thyroid/CNS/lung/joint) +
           CMV/EBV susceptibility (opportunistic infections despite adequate lymphocytes);
         ABATACEPT (CTLA4-Ig fusion protein) IS FIRST-LINE — exogenous CTLA4-Ig restores trans-endocytosis;
         Dramatically effective: shrinks organomegaly, resolves cytopenias, treats IBD;
         PIDD: combined primary immune deficiency + autoimmune disease;
         Treg-CTLA4 recycling defect: CTLA4 normally cycles to surface; LOF → depleted from membrane;
         seed SEED_BASE+1).
LRBA   (LPS-responsive beige-like anchor protein; 2863aa; 4q31.3; AR;
         LRBA encodes a vesicle trafficking protein essential for recycling CTLA4 from lysosomes back to cell surface;
         LRBA LOF → CTLA4 undergoes lysosomal degradation instead of recycling → CTLA4 surface expression reduced;
         Clinical consequence: phenocopies CTLA4 haploinsufficiency (same mechanism: insufficient surface CTLA4);
         LRBA deficiency: lymphoproliferation + autoimmunity + hypogammaglobulinemia + recurrent infections;
         EVANS SYNDROME HALLMARK: autoimmune haemolytic anaemia + immune thrombocytopenia in same patient;
         Additional: IBD, bronchiectasis, autoimmune hepatitis, diabetes;
         ABATACEPT CURATIVE: exogenous CTLA4-Ig bypasses need for intracellular CTLA4 recycling → restores inhibition;
         LRBA is the largest protein in this atlas (2863aa/319kDa) — largest known BEACH domain protein;
         AR inheritance — both parents carriers; consanguinity enriches;
         IVIG for hypogammaglobulinemia component; HSCT for refractory cases;
         seed SEED_BASE+2).
STAT3  (Signal transducer and activator of transcription 3 (GOF); 770aa; 17q21.2; AD-GOF;
         STAT3 is a transcription factor activated by cytokines (IL-6, IL-10, IL-21, JAK pathway);
         AD-GOF mutations → constitutively active STAT3 → hyperactivation of inflammatory cytokine pathways;
         STAT3-GOF syndrome: early-onset MULTI-ORGAN AUTOIMMUNITY (most severe combined autoimmune phenotype):
           TYPE-1 DIABETES (neonatal/infantile; most prominent), AUTOIMMUNE THYROIDITIS,
           AUTOIMMUNE CYTOPENIAS (haemolytic anaemia/thrombocytopenia), ENTEROPATHY, SHORT STATURE;
         Treg dysfunction: STAT3 GOF → IL-10/IL-21 overactivation → impairs Treg suppressive function;
         JAK INHIBITORS (ruxolitinib/baricitinib) ARE DRAMATICALLY EFFECTIVE — block JAK-STAT3 pathway;
         Multiple complete remissions reported with ruxolitinib;
         Also: lymphoproliferation, pulmonary disease, multi-organ autoimmunity from months of age;
         Distinguish from IPEX: multi-organ earlier; STAT3 panels for early-onset multi-organ autoimmune;
         seed SEED_BASE+3).
IL2RA  (Interleukin-2 receptor alpha chain / CD25; 272aa; 10p15.1; AR;
         IL2RA encodes CD25, the alpha chain of the high-affinity IL-2 receptor (IL-2Rα/CD25);
         IL-2 is the growth/survival factor for Tregs — IL-2 signalling essential for Treg development;
         IL2RA LOF → absent IL-2 signalling → Treg deficiency despite normal FOXP3 transcription;
         CD25 deficiency: combined immune deficiency + autoimmunity;
         Phenotype: LYMPHOPROLIFERATION (paradoxical — IL-2 expands effector cells not Tregs) +
           SEVERE INFLAMMATORY BOWEL DISEASE (refractory; colitis from infancy) +
           LUNG DISEASE (interstitial lung disease/bronchiectasis) +
           RECURRENT INFECTIONS (opportunistic: CMV/EBV/PCP);
         PARADOXICAL LYMPHOPROLIFERATION: IL-2 drives effector T cell expansion since Tregs absent;
         Very rare disease — fewer than 30 families worldwide reported;
         Flow cytometry: CD25 absent on T cells (diagnostic); FOXP3 Tregs reduced;
         Treatment: HSCT considered for severe cases; immunosuppression bridge;
         seed SEED_BASE+4).
AIRE   (Autoimmune regulator; 545aa; 21q22.3; AR;
         AIRE is a transcription factor expressed in medullary thymic epithelial cells (mTECs);
         AIRE drives promiscuous expression of tissue antigens in thymus → negative selection of autoreactive T cells;
         AIRE LOF → failure of central tolerance → escape of autoreactive T cells → organ-specific autoimmunity;
         APECED (Autoimmune Polyendocrinopathy Candidiasis Ectodermal Dystrophy) = APS-1;
         APECED TRIAD PATHOGNOMONIC: (1) CHRONIC MUCOCUTANEOUS CANDIDIASIS (earliest manifestation, appears first),
           (2) ADDISON DISEASE (autoimmune adrenal insufficiency — adrenal replacement MANDATORY),
           (3) HYPOPARATHYROIDISM (hypocalcaemia → tetany/seizures → supplement mandatory);
         Anti-IFN-ω and anti-IFN-α autoantibodies: present in >95% APECED patients;
           → BLOCK TYPE-I IFN RESPONSE → candidiasis (Th17 impaired) + COVID-19 SUSCEPTIBILITY (ICU risk);
         Additional features: alopecia, vitiligo, gonadal failure, diabetes, hepatitis, thyroiditis;
         Finnish founder (p.Arg257Stop) and Sardinian founders;
         ADRENAL REPLACEMENT MANDATORY — adrenal crisis life-threatening; cortisol + fludrocortisone;
         Treatment of candidiasis: fluconazole lifelong; IFN-alpha status affects COVID-19 vaccine response;
         seed SEED_BASE+5).
CARD11 (Caspase recruitment domain family member 11 (GOF); 1154aa; 7p22.2; AD-GOF;
         CARD11 is the scaffold of the CBM complex (CARD11-BCL10-MALT1) in lymphocyte NF-κB signalling;
         AD-GOF mutations → constitutively active CARD11 → BCL10-MALT1-NF-κB constitutive activation;
         BENTA disease (B cell expansion with NF-κB and T cell anergy):
           B cells expand (BCR-NF-κB constitutive) while T cells become anergic (TCR-NF-κB exhausted);
         SEVERE ATOPIC DERMATITIS PATHOGNOMONIC: most striking feature; refractory severe eczema from birth;
         EOSINOPHILIA prominent; IgE markedly elevated (>>10,000 IU/mL frequently);
         RECURRENT INFECTIONS: despite B-cell expansion, functional antibody deficient; Staph/viral susceptibility;
         DUPILUMAB PARTIALLY EFFECTIVE (IL-4/IL-13 blockade) — reduces atopic skin disease;
         T cell anergy contributes to infection susceptibility despite normal/elevated total lymphocytes;
         Distinguish from IPEX: no neonatal enteropathy/DM; prominent atopy + eosinophilia + IgE > Treg-primary;
         seed SEED_BASE+6).
BACH2  (BTB and CNC homolog 2; 641aa; 6q15; AD-LOF;
         BACH2 is a transcriptional repressor essential for Treg development and memory B cell differentiation;
         BACH2 LOF → impaired Treg generation + impaired memory B cell development → immune dysregulation;
         BRIDA (BACH2-related immunodeficiency and autoimmunity):
         Phenotype: LYMPHOPROLIFERATION (tissue infiltration; splenomegaly) +
           ENTEROPATHY (IBD-like; chronic diarrhoea; villous atrophy) +
           HYPOGAMMAGLOBULINEMIA (memory B cell defect → CVID-like antibody deficiency) +
           AUTOIMMUNITY (cytopenias/liver/lung);
         Rare disease — AD-LOF means single gene copy insufficient (haploinsufficiency-type mechanism);
         Recurrent infections from hypogammaglobulinemia; IVIG therapy required;
         Treatment: IVIG replacement; HSCT for severe cases; immunosuppression for autoimmune component;
         BACH2 connects Treg biology to memory B cell development — unique dual defect;
         seed SEED_BASE+7).
"""

import random
import math

ATLAS_NAME   = "Hereditary-IPEX-Treg-Immune-Dysregulation-Atlas"
SEED_BASE    = 3974   # seeds 3974-3981 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8

GENES = ["FOXP3", "CTLA4", "LRBA", "STAT3", "IL2RA", "AIRE", "CARD11", "BACH2"]

GENE_DATA = {
    "FOXP3": {
        "full_name":             "Forkhead box P3 (FOXP3 / IPEX master TF)",
        "aa":                    256,
        "locus":                 "Xp11.23",
        "kda":                   "47",
        "inheritance":           "XLR (X-linked recessive; males only classically)",
        "syndrome":              "IPEX syndrome (Immune dysregulation Polyendocrinopathy Enteropathy X-linked)",
        "pathway":               "Treg master TF absent → unchecked effector T cells → multi-organ autoimmunity",
        "onset_range":           (0, 2),    # neonatal/early infantile months
        "autoimmunity_pct":      0.99,
        "lymphoproliferation_pct": 0.60,
        "enteropathy_pct":       0.98,
        "hsct_pct":              0.80,      # HSCT curative
        "abatacept_pct":         0.05,      # not abatacept-treated
        "jaki_pct":              0.05,
        "candidiasis_pct":       0.10,
        "t1dm_pct":              0.90,      # neonatal type-1 DM
        "eczema_pct":            0.95,
        "pathognomonic":         "NEONATAL TRIAD: ENTEROPATHY + TYPE-1 DM (neonatal) + ECZEMA — FOXP3-Treg absent; DEATH <2yr without HSCT",
        "key_drug":              "HSCT CURATIVE (definitive); bridge: sirolimus/tacrolimus (preferred over steroids for gut)",
    },
    "CTLA4": {
        "full_name":             "Cytotoxic T-lymphocyte antigen 4 (CTLA4 haploinsufficiency)",
        "aa":                    223,
        "locus":                 "2q33.2",
        "kda":                   "25",
        "inheritance":           "AD-LOF (haploinsufficiency; 1 copy insufficient)",
        "syndrome":              "CTLA4 Haploinsufficiency (CTLA4-associated regulatory T cell deficiency)",
        "pathway":               "CTLA4 surface ↓ → unchecked CD80/86 co-stimulation → T cell hyperactivation + autoimmunity",
        "onset_range":           (12, 240),  # childhood to adult
        "autoimmunity_pct":      0.95,
        "lymphoproliferation_pct": 0.90,
        "enteropathy_pct":       0.65,
        "hsct_pct":              0.15,
        "abatacept_pct":         0.85,      # abatacept dramatically effective — first-line
        "jaki_pct":              0.10,
        "candidiasis_pct":       0.10,
        "t1dm_pct":              0.20,
        "eczema_pct":            0.25,
        "pathognomonic":         "Lymphoproliferation + CMV/EBV susceptibility + autoimmunity; ABATACEPT first-line (dramatically effective); Treg-CTLA4 recycling defect",
        "key_drug":              "Abatacept (CTLA4-Ig fusion; replaces missing surface CTLA4); dramatically shrinks organomegaly",
    },
    "LRBA": {
        "full_name":             "LPS-responsive beige-like anchor protein (LRBA; BEACH domain)",
        "aa":                    2863,
        "locus":                 "4q31.3",
        "kda":                   "319",
        "inheritance":           "AR (autosomal recessive; consanguinity enriches)",
        "syndrome":              "LRBA Deficiency (phenocopies CTLA4 haploinsufficiency; CTLA4 lysosomal degradation)",
        "pathway":               "LRBA absent → CTLA4 lysosomal degradation (not recycled to surface) → same as CTLA4-LOF",
        "onset_range":           (6, 120),
        "autoimmunity_pct":      0.92,
        "lymphoproliferation_pct": 0.88,
        "enteropathy_pct":       0.75,
        "hsct_pct":              0.25,
        "abatacept_pct":         0.80,      # abatacept CURATIVE — exogenous CTLA4-Ig bypasses recycling defect
        "jaki_pct":              0.08,
        "candidiasis_pct":       0.12,
        "t1dm_pct":              0.18,
        "eczema_pct":            0.30,
        "pathognomonic":         "EVANS SYNDROME (autoimmune haemolytic anaemia + ITP in same patient) hallmark; ABATACEPT CURATIVE; AR; CTLA4 lysosomal degradation",
        "key_drug":              "Abatacept (CTLA4-Ig; curative — bypasses LRBA recycling defect); IVIG for hypogammaglobulinemia",
    },
    "STAT3": {
        "full_name":             "Signal transducer and activator of transcription 3 (STAT3-GOF)",
        "aa":                    770,
        "locus":                 "17q21.2",
        "kda":                   "92",
        "inheritance":           "AD-GOF (gain-of-function; de novo common)",
        "syndrome":              "STAT3 Gain-of-Function (early-onset multi-organ autoimmunity)",
        "pathway":               "Constitutively active STAT3 → IL-10/IL-21/IL-6 overactivation → Treg dysfunction + multi-organ autoimmunity",
        "onset_range":           (0, 24),   # neonatal/early infantile
        "autoimmunity_pct":      0.98,
        "lymphoproliferation_pct": 0.75,
        "enteropathy_pct":       0.70,
        "hsct_pct":              0.30,
        "abatacept_pct":         0.15,
        "jaki_pct":              0.85,      # JAK inhibitors dramatically effective
        "candidiasis_pct":       0.15,
        "t1dm_pct":              0.80,      # prominent neonatal type-1 DM
        "eczema_pct":            0.50,
        "pathognomonic":         "Early-onset MULTI-ORGAN AUTOIMMUNITY: type-1 DM + thyroid + cytopenias + enteropathy + short stature; JAK INHIBITOR (ruxolitinib) dramatically effective",
        "key_drug":              "JAK inhibitors (ruxolitinib/baricitinib; dramatic response); multiple complete remissions reported",
    },
    "IL2RA": {
        "full_name":             "Interleukin-2 receptor alpha chain (CD25)",
        "aa":                    272,
        "locus":                 "10p15.1",
        "kda":                   "30",
        "inheritance":           "AR (very rare; <30 families worldwide)",
        "syndrome":              "CD25 Deficiency (absent IL-2 signalling; Treg deficiency + paradoxical lymphoproliferation)",
        "pathway":               "IL-2Rα absent → no IL-2 signalling → Treg depletion; IL-2 drives effector expansion (paradoxical)",
        "onset_range":           (0, 12),
        "autoimmunity_pct":      0.90,
        "lymphoproliferation_pct": 0.92,   # paradoxical lymphoproliferation
        "enteropathy_pct":       0.85,     # severe IBD
        "hsct_pct":              0.55,
        "abatacept_pct":         0.10,
        "jaki_pct":              0.15,
        "candidiasis_pct":       0.20,
        "t1dm_pct":              0.30,
        "eczema_pct":            0.35,
        "pathognomonic":         "PARADOXICAL LYMPHOPROLIFERATION (IL-2 expands effectors, not Tregs) + severe IBD + lung disease; CD25 absent by flow cytometry; very rare",
        "key_drug":              "HSCT for severe cases; immunosuppression bridge; CD25 absent on flow = diagnostic",
    },
    "AIRE": {
        "full_name":             "Autoimmune regulator (AIRE; mTEC central tolerance)",
        "aa":                    545,
        "locus":                 "21q22.3",
        "kda":                   "58",
        "inheritance":           "AR (Finnish founder p.Arg257Stop; Sardinian founder)",
        "syndrome":              "APECED / APS-1 (Autoimmune Polyendocrinopathy Candidiasis Ectodermal Dystrophy)",
        "pathway":               "AIRE absent → failed thymic central tolerance → organ-specific autoimmunity + anti-IFN autoantibodies",
        "onset_range":           (12, 180),  # childhood
        "autoimmunity_pct":      0.99,
        "lymphoproliferation_pct": 0.20,
        "enteropathy_pct":       0.35,
        "hsct_pct":              0.05,
        "abatacept_pct":         0.05,
        "jaki_pct":              0.05,
        "candidiasis_pct":       0.98,      # PATHOGNOMONIC first manifestation
        "t1dm_pct":              0.18,
        "eczema_pct":            0.15,
        "pathognomonic":         "APECED TRIAD PATHOGNOMONIC: CHRONIC MUCOCUTANEOUS CANDIDIASIS + ADDISON DISEASE + HYPOPARATHYROIDISM; anti-IFN autoantibodies → COVID-19 susceptibility; adrenal replacement MANDATORY",
        "key_drug":              "Fluconazole (lifelong candidiasis); cortisol + fludrocortisone (Addison MANDATORY); calcium + vitamin D (hypoparathyroidism)",
    },
    "CARD11": {
        "full_name":             "Caspase recruitment domain family member 11 (CARD11-GOF / BENTA)",
        "aa":                    1154,
        "locus":                 "7p22.2",
        "kda":                   "133",
        "inheritance":           "AD-GOF (constitutively active CBM-NF-κB complex)",
        "syndrome":              "BENTA disease (B cell expansion with NF-κB and T cell anergy)",
        "pathway":               "Constitutively active CARD11 → BCL10-MALT1-NF-κB → B cell expansion + T cell anergy",
        "onset_range":           (0, 12),   # from birth
        "autoimmunity_pct":      0.55,
        "lymphoproliferation_pct": 0.82,   # B cell expansion
        "enteropathy_pct":       0.30,
        "hsct_pct":              0.10,
        "abatacept_pct":         0.10,
        "jaki_pct":              0.15,
        "candidiasis_pct":       0.20,
        "t1dm_pct":              0.05,
        "eczema_pct":            0.99,     # PATHOGNOMONIC severe atopic dermatitis
        "pathognomonic":         "SEVERE ATOPIC DERMATITIS PATHOGNOMONIC + EOSINOPHILIA + IgE markedly elevated (>>10,000); BENTA disease; dupilumab partially effective",
        "key_drug":              "Dupilumab (IL-4/IL-13 blockade; partially effective for atopic skin); IVIG for infections; avoid live vaccines",
    },
    "BACH2": {
        "full_name":             "BTB and CNC homolog 2 (BACH2 / BRIDA; Treg + memory B cell TF)",
        "aa":                    641,
        "locus":                 "6q15",
        "kda":                   "72",
        "inheritance":           "AD-LOF (haploinsufficiency; single copy insufficient)",
        "syndrome":              "BRIDA (BACH2-related immunodeficiency and autoimmunity)",
        "pathway":               "BACH2 absent → impaired Treg development + impaired memory B cell differentiation → dual immune dysregulation",
        "onset_range":           (12, 360),  # variable childhood to adult
        "autoimmunity_pct":      0.80,
        "lymphoproliferation_pct": 0.78,
        "enteropathy_pct":       0.82,     # enteropathy prominent
        "hsct_pct":              0.20,
        "abatacept_pct":         0.10,
        "jaki_pct":              0.10,
        "candidiasis_pct":       0.08,
        "t1dm_pct":              0.12,
        "eczema_pct":            0.25,
        "pathognomonic":         "Lymphoproliferation + ENTEROPATHY + HYPOGAMMAGLOBULINEMIA (memory B cell defect); rare AD-LOF; BACH2 dual Treg + memory B cell development role",
        "key_drug":              "IVIG replacement (hypogammaglobulinemia); HSCT for severe cases; immunosuppression for autoimmune component",
    },
}


def _rng(seed):
    return random.Random(seed)


def _simulate_patient(gene, seed):
    rng = _rng(seed)
    gd  = GENE_DATA[gene]
    onset_lo, onset_hi = gd["onset_range"]
    onset_mo = rng.randint(onset_lo, max(onset_lo + 1, onset_hi))

    autoimmunity      = rng.random() < gd["autoimmunity_pct"]
    lymphoproliferation = rng.random() < gd["lymphoproliferation_pct"]
    enteropathy       = rng.random() < gd["enteropathy_pct"]
    hsct              = rng.random() < gd["hsct_pct"]
    abatacept         = rng.random() < gd["abatacept_pct"]
    jaki              = rng.random() < gd["jaki_pct"]
    candidiasis       = rng.random() < gd["candidiasis_pct"]
    t1dm              = rng.random() < gd["t1dm_pct"]
    eczema            = rng.random() < gd["eczema_pct"]

    return {
        "onset_months":      onset_mo,
        "autoimmunity":      autoimmunity,
        "lymphoproliferation": lymphoproliferation,
        "enteropathy":       enteropathy,
        "hsct":              hsct,
        "abatacept":         abatacept,
        "jaki":              jaki,
        "candidiasis":       candidiasis,
        "t1dm":              t1dm,
        "eczema":            eczema,
    }


def _aggregate_gene(gene, gene_idx):
    patients = [_simulate_patient(gene, SEED_BASE + gene_idx * N_PER_GENE + i) for i in range(N_PER_GENE)]
    n        = len(patients)

    def pct(k):
        return round(sum(1 for p in patients if p[k]) / n * 100, 1)

    mean_onset = round(sum(p["onset_months"] for p in patients) / n, 1)

    return {
        "gene":                 gene,
        "n":                    n,
        "syndrome":             GENE_DATA[gene]["syndrome"],
        "inheritance":          GENE_DATA[gene]["inheritance"],
        "locus":                GENE_DATA[gene]["locus"],
        "kda":                  GENE_DATA[gene]["kda"],
        "aa":                   GENE_DATA[gene]["aa"],
        "autoimmunity_pct":     pct("autoimmunity"),
        "lymphoproliferation_pct": pct("lymphoproliferation"),
        "enteropathy_pct":      pct("enteropathy"),
        "hsct_pct":             pct("hsct"),
        "abatacept_pct":        pct("abatacept"),
        "jaki_pct":             pct("jaki"),
        "candidiasis_pct":      pct("candidiasis"),
        "t1dm_pct":             pct("t1dm"),
        "eczema_pct":           pct("eczema"),
        "mean_onset_months":    mean_onset,
        "pathognomonic":        GENE_DATA[gene]["pathognomonic"],
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    total_autoimmunity        = round(sum(c["autoimmunity_pct"]        for c in cohorts) / len(cohorts), 1)
    total_lymphoproliferation = round(sum(c["lymphoproliferation_pct"] for c in cohorts) / len(cohorts), 1)
    total_enteropathy         = round(sum(c["enteropathy_pct"]         for c in cohorts) / len(cohorts), 1)
    total_hsct                = round(sum(c["hsct_pct"]                for c in cohorts) / len(cohorts), 1)
    total_abatacept           = round(sum(c["abatacept_pct"]           for c in cohorts) / len(cohorts), 1)
    total_jaki                = round(sum(c["jaki_pct"]                for c in cohorts) / len(cohorts), 1)
    total_candidiasis         = round(sum(c["candidiasis_pct"]         for c in cohorts) / len(cohorts), 1)
    total_t1dm                = round(sum(c["t1dm_pct"]                for c in cohorts) / len(cohorts), 1)
    total_eczema              = round(sum(c["eczema_pct"]              for c in cohorts) / len(cohorts), 1)

    return {
        "atlas":    ATLAS_NAME,
        "genes":    GENES,
        "total_n":  TOTAL_N,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "autoimmunity_pct":        total_autoimmunity,
            "lymphoproliferation_pct": total_lymphoproliferation,
            "enteropathy_pct":         total_enteropathy,
            "hsct_pct":                total_hsct,
            "abatacept_pct":           total_abatacept,
            "jaki_pct":                total_jaki,
            "candidiasis_pct":         total_candidiasis,
            "t1dm_pct":                total_t1dm,
            "eczema_pct":              total_eczema,
        },
        "key_facts": [
            "FOXP3 (IPEX): NEONATAL TRIAD PATHOGNOMONIC: enteropathy + type-1 DM + eczema; FOXP3-Treg absent; DEATH <2yr without HSCT — HSCT is CURATIVE; bridge with sirolimus/tacrolimus (NOT steroids for gut)",
            "CTLA4 (Haploinsufficiency): AD-LOF; lymphoproliferation + CMV/EBV susceptibility + autoimmunity; ABATACEPT FIRST-LINE (dramatically effective — shrinks organomegaly, resolves cytopenias); Treg-CTLA4 recycling defect",
            "LRBA Deficiency: AR; CTLA4 lysosomal degradation (not recycled to surface) → phenocopies CTLA4-haploinsufficiency; EVANS SYNDROME hallmark; ABATACEPT CURATIVE (exogenous CTLA4-Ig bypasses recycling defect)",
            "STAT3-GOF: AD-GOF; early-onset multi-organ autoimmunity (type-1 DM + thyroid + cytopenias + enteropathy + short stature); JAK INHIBITOR (ruxolitinib) dramatically effective; multiple complete remissions",
            "IL2RA (CD25 deficiency): AR; very rare (<30 families); PARADOXICAL LYMPHOPROLIFERATION (IL-2 expands effectors without Tregs); severe IBD + lung disease; CD25 absent on flow = diagnostic",
            "AIRE (APECED/APS-1): APECED TRIAD PATHOGNOMONIC: mucocutaneous CANDIDIASIS (first) + ADDISON (adrenal replacement MANDATORY) + HYPOPARATHYROIDISM; anti-IFN autoantibodies → COVID-19 susceptibility",
            "CARD11-GOF (BENTA): SEVERE ATOPIC DERMATITIS PATHOGNOMONIC; eosinophilia + IgE markedly elevated (>>10,000); B cell expansion + T cell anergy; dupilumab partially effective",
            "BACH2 (BRIDA): AD-LOF; lymphoproliferation + enteropathy + hypogammaglobulinemia (memory B cell defect); BACH2 essential for both Treg and memory B cell development — dual defect; IVIG required",
        ],
        "treg_foxp3_group":      ["FOXP3"],
        "ctla4_lrba_group":      ["CTLA4", "LRBA"],
        "jak_stat_group":        ["STAT3"],
        "il2_signalling_group":  ["IL2RA"],
        "central_tolerance_group": ["AIRE"],
        "nfkb_benta_group":      ["CARD11"],
        "treg_memory_b_group":   ["BACH2"],
        "cohort_summaries":      cohorts,
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    diagnostic_groups = {
        "foxp3_treg": {
            "genes":          ["FOXP3"],
            "mechanism":      "FOXP3 absent → no CD4+CD25+FoxP3+ Tregs → unchecked effector T cells → multi-organ autoimmunity",
            "shared_feature": "XLR; neonatal onset; IPEX triad; HSCT curative; Treg absent on flow",
            "key_ddx":        "IPEX vs STAT3-GOF: IPEX = XLR males, neonatal DM+enteropathy+eczema triad; STAT3-GOF = AD, broader multi-organ, JAKi responsive",
        },
        "ctla4_lrba": {
            "genes":          ["CTLA4", "LRBA"],
            "mechanism":      "Insufficient surface CTLA4 (CTLA4: haploinsufficiency; LRBA: recycling defect → lysosomal degradation) → CD80/CD86 unchecked → T cell hyperactivation",
            "shared_feature": "Both respond dramatically to abatacept (exogenous CTLA4-Ig); lymphoproliferation + autoimmunity + CMV/EBV",
            "key_ddx":        "CTLA4-haploinsufficiency: AD-LOF, dominant. LRBA deficiency: AR, recessive (consanguinity). Evans syndrome more specific to LRBA. Both: abatacept first-line.",
        },
        "stat3_gof": {
            "genes":          ["STAT3"],
            "mechanism":      "Constitutively active STAT3 → IL-10/IL-21/IL-6 overactivation → Treg dysfunction + multi-organ autoimmunity",
            "shared_feature": "AD-GOF; early-onset multi-organ; type-1 DM prominent; JAK inhibitors dramatically effective",
            "key_ddx":        "STAT3-GOF vs IPEX: STAT3-GOF = AD (de novo), multi-organ broader; IPEX = XLR, enteropathy+eczema triad. Both neonatal/infantile onset.",
        },
        "il2ra_cd25": {
            "genes":          ["IL2RA"],
            "mechanism":      "CD25 absent → no IL-2Rα chain → failed IL-2 signalling → Treg deficiency; IL-2 drives effector expansion (paradoxical lymphoproliferation)",
            "shared_feature": "AR; very rare; paradoxical lymphoproliferation; severe IBD; CD25 absent by flow (diagnostic)",
            "key_ddx":        "IL2RA vs FOXP3: FOXP3 XLR males; IL2RA AR both sexes. Both Treg deficiency. IL2RA: paradoxical lymphoproliferation distinctive.",
        },
        "aire_apeced": {
            "genes":          ["AIRE"],
            "mechanism":      "AIRE absent in mTECs → failed promiscuous antigen expression → no negative selection → organ-specific autoimmunity + anti-IFN autoantibodies",
            "shared_feature": "AR; APECED triad (candidiasis, Addison, hypoparathyroidism); anti-IFN-α/ω autoantibodies diagnostic; Finnish/Sardinian founders",
            "key_ddx":        "APECED vs other PIDs: candidiasis first (often age 1-4yr); Addison + hypoparathyroidism together distinguishes from CVID/other. anti-IFN-ω >95% sensitive.",
        },
        "card11_benta": {
            "genes":          ["CARD11"],
            "mechanism":      "CARD11-GOF → constitutively active BCL10-MALT1-NF-κB → B cell expansion (BCR pathway) + T cell anergy (TCR pathway exhausted)",
            "shared_feature": "AD-GOF; severe atopic dermatitis; eosinophilia; IgE >> elevated; BENTA disease",
            "key_ddx":        "BENTA vs IPEX atopy: BENTA = CARD11 GOF; IgE >>10,000; B cell expansion; NO neonatal DM/enteropathy triad. BENTA vs hyper-IgE: CARD11 GOF specific.",
        },
        "bach2_brida": {
            "genes":          ["BACH2"],
            "mechanism":      "BACH2-LOF → impaired Treg differentiation + impaired memory B cell generation → combined Treg + antibody deficiency",
            "shared_feature": "AD-LOF; lymphoproliferation + enteropathy + hypogammaglobulinemia; IVIG required; BACH2 unique dual role",
            "key_ddx":        "BRIDA vs CVID: BRIDA = BACH2 AD-LOF + lymphoproliferation + Treg dysfunction; CVID no clear Treg defect. Enteropathy + lymphoproliferation + hypo-Ig combo → BACH2 panel.",
        },
    }

    pathognomonic_features = {
        "ipex_neonatal_triad":        "FOXP3/IPEX: NEONATAL ENTEROPATHY + TYPE-1 DM + ECZEMA TRIAD = PATHOGNOMONIC; FOXP3-Treg absent on flow; HSCT curative; death <2yr without HSCT — most urgent in atlas",
        "ctla4_abatacept_response":   "CTLA4 haploinsufficiency: ABATACEPT (CTLA4-Ig) first-line and dramatically effective — organ response within weeks; lymphoproliferation + CMV/EBV susceptibility + autoimmunity",
        "lrba_evans_abatacept_cure":  "LRBA deficiency: EVANS SYNDROME (AIHA + ITP combined) hallmark + abatacept CURATIVE; AR; phenocopies CTLA4 haploinsufficiency — check LRBA on CTLA4-LOF phenotype with AR pattern",
        "stat3_gof_jaki_response":    "STAT3-GOF: JAK INHIBITOR (ruxolitinib) dramatically effective → complete remissions; early-onset multi-organ autoimmunity (DM+thyroid+cytopenias+enteropathy); AD-GOF de novo",
        "il2ra_paradoxical_lympho":   "IL2RA/CD25 deficiency: PARADOXICAL LYMPHOPROLIFERATION (IL-2 drives effectors without Tregs); CD25 absent by flow = diagnostic; very rare (<30 families worldwide)",
        "aire_apeced_triad":          "AIRE/APECED: CANDIDIASIS (first manifestation) + ADDISON (adrenal replacement MANDATORY — adrenal crisis kills) + HYPOPARATHYROIDISM = APECED TRIAD PATHOGNOMONIC",
        "aire_anti_ifn_covid":        "AIRE/APECED: anti-IFN-α/ω autoantibodies in >95% → block type-I IFN → CANDIDIASIS susceptibility + COVID-19 SEVERE RISK; check anti-IFN-ω when APECED suspected",
        "adrenal_replacement_mandatory": "AIRE/APECED: ADRENAL REPLACEMENT (hydrocortisone + fludrocortisone) MANDATORY — undiagnosed Addison = adrenal crisis → death; always check cortisol/ACTH in APECED",
        "card11_benta_severe_atopy":  "CARD11-GOF/BENTA: SEVERE ATOPIC DERMATITIS from birth PATHOGNOMONIC + eosinophilia + IgE >>10,000; B cell expansion + T cell anergy; dupilumab partial response",
        "bach2_dual_defect":          "BACH2/BRIDA: DUAL DEFECT — Treg development impaired + memory B cell impaired; lymphoproliferation + enteropathy + hypogammaglobulinemia; AD-LOF; IVIG mandatory",
        "abatacept_for_ctla4_lrba":   "ABATACEPT COVERS BOTH: CTLA4-haploinsufficiency and LRBA deficiency both respond to abatacept (CTLA4-Ig) — same downstream mechanism (insufficient surface CTLA4)",
        "hsct_foxp3_curative":        "FOXP3/IPEX: HSCT IS THE ONLY CURE — must be performed early; bridge with sirolimus (NOT high-dose steroids which worsen enteropathy gut inflammation)",
    }

    return {
        "gene_cohorts":           cohorts,
        "diagnostic_groups":      diagnostic_groups,
        "pathognomonic_features": pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: Neonatal boy with enteropathy + type-1 DM + eczema → FOXP3 sequencing URGENT; Treg flow cytometry; HSCT evaluation immediately",
            "Step 2: Early-onset multi-organ autoimmunity (DM + thyroid + cytopenias) → STAT3 GOF panel; start ruxolitinib trial if STAT3-GOF confirmed",
            "Step 3: Lymphoproliferation + autoimmunity + CMV/EBV susceptibility (any age) → CTLA4 sequencing; abatacept trial if CTLA4 AD-LOF confirmed",
            "Step 4: CTLA4-like phenotype with AR pattern (consanguinity/both sexes) → LRBA sequencing + CTLA4 flow (surface expression); Evans syndrome → LRBA first",
            "Step 5: Lymphoproliferation + severe IBD + lung disease (early onset) → IL2RA (CD25) flow cytometry; IL2RA sequencing if CD25 absent",
            "Step 6: Chronic mucocutaneous candidiasis (child, >4 sites) → APECED panel (AIRE sequencing); anti-IFN-ω autoantibodies; cortisol/ACTH; calcium/PTH",
            "Step 7: Confirmed APECED → adrenal replacement IMMEDIATELY; parathyroid hormone supplement; COVID-19 enhanced precautions (anti-IFN → severe risk)",
            "Step 8: Severe atopic dermatitis from birth + eosinophilia + IgE >>10,000 → CARD11 sequencing; BENTA disease panel",
            "Step 9: Lymphoproliferation + enteropathy + hypogammaglobulinemia → BACH2 sequencing (BRIDA); IVIG started empirically",
            "Step 10: All IPEX-spectrum diagnoses: FOXP3 Treg flow cytometry; CTLA4 surface expression; HSCT evaluation for FOXP3/FOXP3-like; abatacept for CTLA4/LRBA",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "IPEX Syndrome (FOXP3)", "definition": "Immune dysregulation Polyendocrinopathy Enteropathy X-linked. XLR — males only. FOXP3 = master Treg transcription factor; absent → no CD4+CD25+FoxP3+ Tregs → unchecked effector T cells attack every organ. NEONATAL TRIAD PATHOGNOMONIC: (1) Enteropathy (severe watery/bloody diarrhoea, failure to thrive), (2) Type-1 DM (neonatal autoimmune, first days-weeks of life), (3) Eczema (severe atopic dermatitis). Death <2 years of age without HSCT. HSCT IS CURATIVE. Bridge: sirolimus/tacrolimus (prefer over steroids which damage gut). FOXP3-Treg absent/markedly reduced on flow = diagnostic flow pattern."},
            {"term": "CTLA4 Haploinsufficiency", "definition": "AD-LOF CTLA4 (2q33.2; 223aa). CTLA4 is the inhibitory checkpoint on T cells; constitutively expressed on Tregs. Haploinsufficiency → insufficient CTLA4 function → unchecked CD80/CD86 co-stimulation on APCs → T cell hyperactivation. Phenotype: lymphoproliferation (lymphadenopathy/splenomegaly) + autoimmunity (cytopenias/IBD/thyroid/CNS/lung) + CMV/EBV susceptibility despite lymphocyte numbers adequate. ABATACEPT (CTLA4-Ig fusion protein) IS FIRST-LINE AND DRAMATICALLY EFFECTIVE — shrinks organomegaly within weeks, resolves cytopenias. Treg-CTLA4 recycling defect: CTLA4 cycles between intracellular vesicles and surface; LOF → depleted from membrane."},
            {"term": "LRBA Deficiency", "definition": "AR LRBA (4q31.3; 2863aa; largest BEACH domain protein). LRBA normally prevents CTLA4 from entering lysosomes by sorting it to recycling endosomes. LRBA LOF → CTLA4 routed to lysosomes for degradation → surface CTLA4 depleted → functionally same as CTLA4 haploinsufficiency. EVANS SYNDROME (autoimmune haemolytic anaemia + immune thrombocytopenia in same patient) = hallmark LRBA feature. Phenocopies CTLA4 haploinsufficiency: lymphoproliferation + autoimmunity + IBD + hypogammaglobulinemia. ABATACEPT CURATIVE: exogenous CTLA4-Ig bypasses need for intracellular CTLA4 recycling. AR = check parents/siblings; consanguinity enriches."},
            {"term": "STAT3 Gain-of-Function", "definition": "AD-GOF STAT3 (17q21.2; 770aa). Constitutively active STAT3 → IL-10/IL-21/IL-6 pathways overactive → Treg dysfunction (IL-10 feedback lost) + effector T cell hyperactivation. EARLY-ONSET MULTI-ORGAN AUTOIMMUNITY (most severe multi-autoimmune phenotype in atlas): Type-1 DM (prominent; neonatal/infantile onset), autoimmune thyroiditis, autoimmune cytopenias, enteropathy, SHORT STATURE. JAK INHIBITORS (ruxolitinib/baricitinib) DRAMATICALLY EFFECTIVE — block JAK-STAT3 constitutive activation; multiple complete remissions reported. De novo GOF mutations common. Panel: STAT3 for any early-onset multi-organ autoimmunity."},
            {"term": "CD25 Deficiency (IL2RA)", "definition": "AR IL2RA (10p15.1; 272aa; very rare <30 families worldwide). CD25 = alpha chain of high-affinity IL-2 receptor. IL-2 is the primary survival/growth signal for Tregs. CD25 absent → no IL-2 signalling → Treg deficiency despite intact FOXP3. PARADOXICAL LYMPHOPROLIFERATION: without Tregs, IL-2 drives effector T cell expansion (not Treg expansion) → lymphoproliferation despite Treg depletion. Phenotype: lymphoproliferation + SEVERE IBD (refractory; colitis from infancy) + interstitial lung disease + recurrent opportunistic infections (CMV/EBV/PCP). Flow: CD25 absent on T cells (diagnostic). HSCT for severe cases."},
            {"term": "APECED / APS-1 (AIRE)", "definition": "Autoimmune Polyendocrinopathy Candidiasis Ectodermal Dystrophy. AR AIRE (21q22.3; 545aa). AIRE absent in medullary thymic epithelial cells → fails to express tissue-specific antigens in thymus → autoreactive T cells escape negative selection → organ-specific autoimmunity. APECED TRIAD PATHOGNOMONIC: (1) CHRONIC MUCOCUTANEOUS CANDIDIASIS (earliest; appears first, age 1-4yr), (2) ADDISON DISEASE (autoimmune adrenal insufficiency; adrenal crisis = life-threatening; cortisol + fludrocortisone MANDATORY), (3) HYPOPARATHYROIDISM (hypocalcaemia → tetany/seizures). Anti-IFN-α/ω autoantibodies in >95% → block type-I IFN → candidiasis (Th17 impaired) + COVID-19 SEVERE SUSCEPTIBILITY (ICU mortality)."},
            {"term": "BENTA Disease (CARD11-GOF)", "definition": "B cell expansion with NF-κB and T cell anergy. AD-GOF CARD11 (7p22.2; 1154aa). CARD11 is the scaffold of CBM complex (CARD11-BCL10-MALT1). GOF → constitutively active CARD11 → BCL10-MALT1-NF-κB constitutively active: B cells expand (BCR-NF-κB constitutive activation), T cells become anergic (TCR-NF-κB pathway exhausted/desensitized). SEVERE ATOPIC DERMATITIS PATHOGNOMONIC from birth + eosinophilia + IgE markedly elevated (>>10,000 IU/mL). Recurrent infections (Staphylococcal/viral) despite B cell expansion (functional antibody defect). DUPILUMAB PARTIALLY EFFECTIVE (IL-4/IL-13 blockade reduces skin disease). Distinguish: no neonatal DM/enteropathy triad of IPEX."},
            {"term": "BRIDA (BACH2-LOF)", "definition": "BACH2-related immunodeficiency and autoimmunity. AD-LOF BACH2 (6q15; 641aa; BTB-CNC transcription factor). BACH2 controls differentiation of naïve T cells into Tregs (vs effectors) AND differentiation of B cells into memory B cells (vs plasma cells). BACH2-LOF → impaired Treg generation + impaired memory B cell formation → DUAL DEFECT. Phenotype: lymphoproliferation + ENTEROPATHY (IBD-like; chronic diarrhoea; villous atrophy) + HYPOGAMMAGLOBULINEMIA (memory B cell absent → CVID-like antibody deficiency) + autoimmunity (cytopenias/hepatitis/lung). IVIG replacement mandatory (hypogammaglobulinemia). Rare AD-LOF (haploinsufficiency mechanism)."},
            {"term": "Abatacept Mechanism (CTLA4-Ig)", "definition": "Abatacept = recombinant human CTLA4 fused to IgG1 Fc (CTLA4-Ig). Mechanism: CTLA4-Ig binds CD80 (B7-1) and CD86 (B7-2) on APCs competitively → blocks CD28 co-stimulation → T cell activation reduced. Relevance to CTLA4 haploinsufficiency/LRBA deficiency: exogenous CTLA4-Ig (abatacept) compensates for insufficient endogenous CTLA4 surface expression → restores trans-endocytosis capacity → reduces co-stimulation. DRAMATICALLY EFFECTIVE in CTLA4-haploinsufficiency and LRBA deficiency — organomegaly shrinks within weeks, cytopenias resolve, IBD improves. Higher doses/more frequent dosing often required vs rheumatology use."},
            {"term": "JAK Inhibitors in Immune Dysregulation", "definition": "JAK inhibitors (ruxolitinib/baricitinib/tofacitinib) block JAK1/JAK2/JAK3 → inhibit STAT1/STAT3/STAT5 phosphorylation → reduce cytokine-driven inflammation. PRIMARY USE IN ATLAS: STAT3-GOF — ruxolitinib directly blocks constitutively active STAT3 pathway; multiple complete remissions reported in case series. Also used off-label: IPEX (bridge before HSCT), LRBA (partial response), CTLA4 haploinsufficiency (adjunct). Mechanism insight: ruxolitinib in STAT3-GOF = targeted therapy addressing root cause (blocking the constitutively active kinase). Side effects: infections (JAK1/3 blockade → immunosuppression), cytopenias."},
            {"term": "Treg Biology", "definition": "Regulatory T cells (Tregs) = CD4+CD25+FoxP3+ T cells. Master TF: FOXP3. Key surface markers: CD25 (IL-2Rα), CTLA4, GITR, CD127-low. Mechanisms of suppression: (1) CTLA4 trans-endocytosis removes CD80/CD86 from APCs (reduces co-stimulation), (2) IL-10/TGF-β secretion (anti-inflammatory cytokines), (3) IL-2 consumption (deprives effectors). Treg deficiency → loss of self-tolerance → multi-organ autoimmunity. Genes regulating Tregs (this atlas): FOXP3 (master TF), CTLA4 (key Treg surface), LRBA (CTLA4 recycling), STAT3 (IL-10/IL-21 feedback), IL2RA (Treg survival signal), BACH2 (Treg differentiation)."},
            {"term": "Anti-IFN Autoantibodies (APECED)", "definition": "Anti-IFN-ω and anti-IFN-α autoantibodies: present in >95% of APECED/APS-1 patients. PATHOGENIC: block type-I interferon signalling → two consequences: (1) CANDIDIASIS susceptibility — type-I IFN normally supports Th17 responses which control Candida; blocked type-I IFN → impaired anti-fungal immunity, (2) COVID-19 SEVERE SUSCEPTIBILITY — type-I IFN is the primary anti-viral innate response; blocked by anti-IFN-α/ω → COVID-19 can progress to ICU/fatal in APECED without protective IFN. DIAGNOSTIC: anti-IFN-ω antibody highly sensitive/specific for APECED (>95%); use as a screening test when APECED suspected. Also found in severe COVID-19 non-APECED patients (10%) — different mechanism."},
            {"term": "Evans Syndrome (LRBA hallmark)", "definition": "Evans syndrome = autoimmune haemolytic anaemia (AIHA) + immune thrombocytopenia (ITP) occurring together in the same patient (not necessarily simultaneously). In general population: often idiopathic. IN PRIMARY IMMUNODEFICIENCY: Evans syndrome = strong LRBA deficiency marker — should prompt LRBA sequencing + CTLA4 flow cytometry. Mechanism: Treg dysfunction (CTLA4 surface depleted due to LRBA LOF) → loss of tolerance to erythrocytes + platelets simultaneously. Management: abatacept (treats underlying LRBA deficiency → resolves Evans); steroids/IVIG for acute haemolysis; rituximab for refractory AIHA component."},
        ],
        "key_distinguishers": {
            "treg_flow_pattern":       "FOXP3/IPEX: CD4+CD25+FoxP3+ Tregs absent/markedly reduced. CTLA4-haploinsufficiency/LRBA: Tregs present but CTLA4 surface expression reduced. IL2RA: CD25 absent on all T cells (not just Tregs — diagnostic). STAT3-GOF: Tregs present but functionally impaired.",
            "abatacept_vs_jaki":       "CTLA4 haploinsufficiency + LRBA deficiency: ABATACEPT first-line (dramatic). STAT3-GOF: JAK INHIBITOR (ruxolitinib) first-line (dramatic). FOXP3/IPEX: HSCT curative; abatacept/JAKi as bridge only.",
            "sex_pattern":             "FOXP3/IPEX: XLR — males only (classically; rare carrier females). CTLA4/AD-LOF: both sexes (dominant). LRBA/AR: both sexes. STAT3-GOF/AD: both sexes. IL2RA/AR: both sexes. AIRE/AR: both sexes. CARD11/AD-GOF: both sexes. BACH2/AD-LOF: both sexes.",
            "onset_age":               "FOXP3/IPEX: neonatal (days-weeks). STAT3-GOF: neonatal/infantile (months). IL2RA: infantile. CARD11: from birth. LRBA/CTLA4: childhood. AIRE/APECED: childhood (candidiasis first, age 1-4yr). BACH2: variable childhood-adult.",
            "candidiasis_clue":        "AIRE/APECED: mucocutaneous candidiasis FIRST MANIFESTATION (oral/nail/skin). Then Addison (median age 8-9yr). Then hypoparathyroidism. Candidiasis in a child → anti-IFN-ω antibodies → AIRE sequencing.",
            "hypogammaglobulinemia":   "LRBA deficiency: hypogammaglobulinemia (CVID-like component) + Treg dysfunction + lymphoproliferation. BACH2: hypogammaglobulinemia (memory B cell defect) + enteropathy + lymphoproliferation. CTLA4: less prominent hypogammaglobulinemia.",
            "ige_eosinophilia":        "CARD11-GOF/BENTA: IgE markedly elevated (>>10,000 IU/mL) + eosinophilia + severe atopy. Others: less prominent IgE elevation. High IgE + severe atopy + lymphoproliferation → CARD11 panel.",
        },
        "treatment_matrix": {
            "hsct_curative":    "FOXP3/IPEX: HSCT is definitive cure — perform early; bridge with sirolimus/tacrolimus. IL2RA: HSCT for severe. LRBA: HSCT for refractory-to-abatacept. BACH2: HSCT for severe cases.",
            "abatacept":        "CTLA4 haploinsufficiency: first-line, dramatic response. LRBA deficiency: CURATIVE (exogenous CTLA4-Ig bypasses recycling defect). FOXP3/IPEX: bridge use only.",
            "jaki_ruxolitinib": "STAT3-GOF: FIRST-LINE, dramatically effective, multiple complete remissions. Off-label: IPEX bridge, LRBA partial, CTLA4 adjunct.",
            "ivig":             "LRBA: hypogammaglobulinemia component. BACH2: mandatory (memory B cell defect → antibody deficiency). IL2RA: infections coverage. CARD11-BENTA: recurrent infections.",
            "fluconazole":      "AIRE/APECED: lifelong fluconazole for candidiasis; escalate to voriconazole/amphotericin for azole-resistant species.",
            "adrenal_replacement": "AIRE/APECED: hydrocortisone + fludrocortisone MANDATORY — adrenal crisis is life-threatening; never stop; stress dosing for illness/procedures.",
            "calcium_vit_d":    "AIRE/APECED: hypoparathyroidism → hypocalcaemia → oral calcium + calcitriol lifelong; monitor PTH/Ca2+ regularly.",
            "dupilumab":        "CARD11-GOF/BENTA: IL-4/IL-13 blockade → partial reduction of atopic skin disease. Not curative; does not address NF-κB constitutive activation.",
            "sirolimus_fk506":  "FOXP3/IPEX bridge (sirolimus preferred — mTOR inhibition also has Treg-sparing effects; tacrolimus secondary). NOT high-dose steroids (worsen enteropathy/gut damage).",
        },
    }


if __name__ == "__main__":
    import json
    ov = generate_overview()
    bk = generate_breakdown()
    df = generate_definitions()
    print(f"Overview genes: {ov['genes']}")
    print(f"Total N: {ov['total_n']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Gene cohorts: {len(bk['gene_cohorts'])}")
    print(f"Definitions: {len(df['definitions'])}")
    print("OK")
