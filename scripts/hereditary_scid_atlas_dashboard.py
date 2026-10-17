#!/usr/bin/env python3
"""Hereditary-SCID-Atlas -- Complete 8-Gene Reference
Severe Combined Immunodeficiency: T cell and B/NK cell enumeration by gene
(IL2RG/XL-SCID-T-B+NK--Most-Common-50pct-Gene-Therapy-First-Approved,
 ADA/T-B-NK--Enzyme-Replacement-Elapegademase-Gene-Therapy-Strimvelis,
 RAG1/T-B-NK+-Omenn-Syndrome-Spectrum-Leaky-Variants,
 RAG2/T-B-NK+-Omenn-Syndrome-Allelic-RAG1,
 DCLRE1C/Artemis-Radiosensitive-SCID-T-B-NK+-AVOID-RADIATION-ABSOLUTELY,
 JAK3/AR-T-B+NK--Phenocopy-XL-SCID-IL2RG-Distinguish-Genetics,
 IL7R/T-B+NK+-T-Cell-Differentiation-Blocked-Good-HSCT-Outcome,
 AK2/Reticular-Dysgenesis-Most-Severe-All-Lineages-Absent-Congenital-Deafness):

IL2RG  (Interleukin-2 receptor γ chain / common γ chain; 369aa; Xq13.1; XL;
         Most common SCID: 40-50% of all SCID cases worldwide;
         γc is shared by receptors for IL-2, IL-4, IL-7, IL-9, IL-15, IL-21;
         Without γc: signalling via all these cytokines blocked → T and NK cell failure;
         Immunophenotype: T-B+NK- (B cells present but non-functional; NK absent);
         Newborn screening: TREC (T-cell receptor excision circles) = 0 or severely low;
         Gene therapy (Retroviral vector): first gene therapy approved — Paris cohort 2000;
         HSCT: best outcomes with matched sibling donor (MSD) pre-conditioning omitted for infant;
         PCP prophylaxis (co-trimoxazole) + CMV/RSV surveillance MANDATORY from diagnosis;
         seed SEED_BASE+0).
ADA    (Adenosine deaminase; 363aa; 20q13.12; AR;
         ADA-SCID: 10-20% of all SCID; metabolic cause (purine salvage defect);
         Absent ADA → toxic accumulation of deoxyadenosine and dATP → T cell apoptosis (most sensitive);
         Immunophenotype: T-B-NK- (all lymphocyte lineages affected — most profound lymphopenia);
         ADA activity in red blood cells: absent or severely reduced (diagnostic test);
         Enzyme replacement: Elapegademase-lvlr (PEGylated bovine ADA) — bridges to curative therapy;
         Gene therapy: Strimvelis® (ex-vivo HSC gene therapy, EMA 2016) — FIRST GSK/ADA gene therapy;
         HSCT: excellent if MSD available; ERT stabilises until HSCT/gene therapy;
         Newborn screening TREC: profoundly low T cells → TREC = 0;
         seed SEED_BASE+1).
RAG1   (Recombination Activating Gene 1; 1043aa; 11p13; AR;
         RAG1 encodes the catalytic subunit of RAG1/2 recombinase — essential for V(D)J recombination;
         Complete LOF → no functional T or B cell receptors → T-B-NK+ SCID;
         Partial/hypomorphic RAG1 variants → Omenn syndrome: oligoclonal T cells, erythroderma, lymphadenopathy;
         Omenn syndrome TRIAD: generalised erythroderma + hypereosinophilia + lymphadenopathy + hepatosplenomegaly;
         Differential: Graft-vs-host disease in transfusion-naive infant (exclude maternal engraftment);
         Immunophenotype varies by variant severity: null = T-B-NK+; hypomorphic = Omenn (T+ oligoclonal);
         No HLA-matched graft needed for HSCT when B cell absent (no allorecognition);
         RAG1 and RAG2 on same chromosome 11p13 — ALWAYS sequence BOTH genes;
         seed SEED_BASE+2).
RAG2   (Recombination Activating Gene 2; 527aa; 11p13; AR;
         RAG2 is the DNA-binding component of the RAG1/2 recombinase complex (RAG1 catalytic);
         Complete LOF → T-B-NK+ SCID (allelic to RAG1 in immunophenotype);
         Hypomorphic RAG2 → Omenn syndrome (same spectrum as RAG1);
         RAG2 also associated with: combined immunodeficiency with granulomas, idiopathic CD4 lymphopenia;
         R229Q founder variant in North African populations → Omenn syndrome;
         RAG2 protein has a PHD finger domain that reads H3K4me3 chromatin mark for V(D)J opening;
         Both RAG1 and RAG2 testing MANDATORY when T-B-NK+ SCID detected — locus 11p13;
         HSCT curative: T-B-NK+ phenotype has best outcomes post-HLA-matched HSCT;
         seed SEED_BASE+3).
DCLRE1C (Artemis DNA repair endonuclease; 692aa; 10p13; AR;
          Artemis SCID = RADIOSENSITIVE SCID — critical therapy modification;
          Artemis opens hairpin intermediates created by RAG1/2 during V(D)J recombination;
          Absent Artemis → V(D)J recombination cannot complete → T-B-NK+ SCID;
          RADIOSENSITIVITY: Artemis is also required for DNA double-strand break repair (NHEJ pathway);
          ABSOLUTELY CONTRAINDICATED: ionising radiation (X-ray conditioning) and busulfan → fatal toxicity;
          REDUCED-INTENSITY CONDITIONING only for HSCT — avoid busulfan/TBI;
          Founder allele in Navajo (NA)/Apache (San Carlos) Native American populations;
          Immunophenotype: T-B-NK+ (NK cells present, T and B absent — identical to RAG1/2);
          Differentiate from RAG deficiency by radiosensitivity assay (colony survival or comet assay);
          seed SEED_BASE+4).
JAK3   (Janus kinase 3; 1124aa; 19p13.11; AR;
         JAK3 is the kinase that pairs exclusively with γc (IL2RG) to transduce cytokine signals;
         JAK3 LOF → identical immunophenotype to IL2RG/XL-SCID: T-B+NK-;
         AR inheritance: distinguishes JAK3-SCID from XL-SCID (IL2RG, males only) — JAK3 affects both sexes;
         CRITICAL DIAGNOSTIC PITFALL: T-B+NK- phenotype in a FEMALE → JAK3 not IL2RG (XL-SCID affects males);
         Also: T-B+NK- in male with negative IL2RG sequencing → sequence JAK3;
         JAK3 inhibitors (ruxolitinib, tofacitinib) mimic JAK3 deficiency phenotype — iatrogenic consideration;
         HSCT outcomes comparable to IL2RG-SCID: excellent with MSD early in life;
         Cytokine signalling cascade: IL2RG-JAK3-STAT5 pathway — JAK3 is the exclusive JAK3-γc partner;
         seed SEED_BASE+5).
IL7R   (Interleukin-7 receptor α chain; 459aa; 5p13.2; AR;
         IL-7 receptor is a heterodimer: IL7Rα (CD127) + γc (IL2RG);
         IL-7 is the non-redundant cytokine for human T-cell development in thymus;
         IL7R LOF → T-B+NK+ SCID (B and NK cells intact, only T cells absent);
         MOST SELECTIVE lymphopenia: pure T cell deficiency with normal B cells AND NK cells;
         T-B+NK+ phenotype distinguishes IL7R from XL-SCID/JAK3 (T-B+NK-) and RAG/Artemis (T-B-NK+);
         Good HSCT outcomes: T cell compartment selectively absent, reconstitution faster than pan-lymphopenic SCID;
         Functional T cells from maternal-fetal transfer may be detectable (maternal T cell engraftment — exclude graft-vs-host);
         IL7R also implicated in multiple sclerosis susceptibility via regulatory T cell development;
         seed SEED_BASE+6).
AK2    (Adenylate kinase 2; 239aa; 1p34.1; AR;
         AK2 deficiency = Reticular dysgenesis — MOST SEVERE and RAREST SCID;
         AK2 is the mitochondrial isoform of adenylate kinase — critical for energy homeostasis in early haematopoiesis;
         Complete LOF → arrest of haematopoiesis at the myeloid progenitor stage as well as lymphoid;
         Immunophenotype: T-B-NK- PLUS absent granulocytes (agranulocytosis) — NO innate OR adaptive immunity;
         CONGENITAL SENSORINEURAL DEAFNESS: AK2 expressed in inner ear stria vascularis — PATHOGNOMONIC bilateral deafness;
         Without hearing in a SCID infant → test AK2 first (pathognomonic association);
         Fatal infections in first weeks of life without HSCT — median survival without treatment: 3-6 weeks;
         HSCT curative for immune reconstitution but does NOT restore hearing (cochlear cells non-haematopoietic);
         Historical name: reticular dysgenesis = empty bone marrow on biopsy;
         seed SEED_BASE+7).
"""
import random
import math

SEED_BASE = 3854

ATLAS_GENES = [
    {
        "name": "IL2RG",
        "full_name": "Interleukin-2 receptor γ chain (common γ chain)",
        "aa": 369,
        "locus": "Xq13.1",
        "inheritance": "XL",
        "mechanism": "LOF",
        "syndrome": "XL-SCID T-B+NK- / Gene Therapy",
        "mw_kda": 42.3,
        "critical_rules": [
            "IL2RG-MOST-COMMON-SCID-40-50PCT-ALL-CASES",
            "IL2RG-T-MINUS-B-PLUS-NK-MINUS-PATHOGNOMONIC-PHENOTYPE",
            "IL2RG-GENE-THERAPY-FIRST-APPROVED-RETROVIRAL-PARIS-2000",
            "IL2RG-TREC-ZERO-NEWBORN-SCREENING-GOLD-STANDARD",
        ],
        "distinguishing_feature": "Most common SCID (40-50%); T-B+NK-; gene therapy first approved; TREC=0 on NBS",
        "first_line_treatment": "HSCT (MSD best, unrelated acceptable) ideally <3.5 months; gene therapy (lentiviral vector); PCP prophylaxis co-trimoxazole; CMV/RSV surveillance",
        "pathognomonic": "Male infant with T-B+NK- lymphopenia + absent TREC + recurrent severe infections + absent γc protein by flow cytometry",
        "onset_yrs_mean": 0.2,
        "penetrance_pct": 100,
        "t_cell_absent_pct": 100,
        "b_cell_absent_pct": 0,
        "nk_cell_absent_pct": 100,
        "radiosensitive_pct": 0,
        "deafness_pct": 0,
        "gene_therapy_eligible_pct": 80,
        "hsct_curative_pct": 90,
        "omenn_pct": 0,
        "maternal_engraftment_pct": 25,
        "dx_lag_yrs": 0.15,
    },
    {
        "name": "ADA",
        "full_name": "Adenosine deaminase",
        "aa": 363,
        "locus": "20q13.12",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "ADA-SCID T-B-NK- / Enzyme Replacement / Gene Therapy",
        "mw_kda": 40.8,
        "critical_rules": [
            "ADA-T-MINUS-B-MINUS-NK-MINUS-MOST-PROFOUND-LYMPHOPENIA",
            "ADA-ELAPEGADEMASE-ENZYME-REPLACEMENT-BRIDGES-TO-CURATIVE",
            "ADA-STRIMVELIS-GENE-THERAPY-EMA2016-FIRST-GSK-GENE-THERAPY",
            "ADA-RBC-ADENOSINE-DEAMINASE-ACTIVITY-ABSENT-DIAGNOSTIC",
        ],
        "distinguishing_feature": "T-B-NK- (all lineages); ADA activity absent in RBCs; Strimvelis gene therapy EMA 2016; enzyme replacement bridges to cure",
        "first_line_treatment": "Elapegademase-lvlr (PEGylated ADA) IM weekly until HSCT/gene therapy; Strimvelis (ex-vivo HSC lentiviral) if no MSD; HSCT if MSD available",
        "pathognomonic": "T-B-NK- lymphopenia + absent ADA enzyme activity in erythrocytes + deoxyadenosine toxicity + bone radiograph abnormalities (cupping/flaring of costochondral junctions)",
        "onset_yrs_mean": 0.3,
        "penetrance_pct": 100,
        "t_cell_absent_pct": 100,
        "b_cell_absent_pct": 100,
        "nk_cell_absent_pct": 100,
        "radiosensitive_pct": 0,
        "deafness_pct": 0,
        "gene_therapy_eligible_pct": 90,
        "hsct_curative_pct": 85,
        "omenn_pct": 0,
        "maternal_engraftment_pct": 10,
        "dx_lag_yrs": 0.25,
    },
    {
        "name": "RAG1",
        "full_name": "Recombination Activating Gene 1",
        "aa": 1043,
        "locus": "11p13",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "RAG1-SCID T-B-NK+ / Omenn Syndrome",
        "mw_kda": 119.0,
        "critical_rules": [
            "RAG1-T-MINUS-B-MINUS-NK-PLUS-COMPLETE-LOF",
            "RAG1-OMENN-HYPOMORPHIC-ERYTHRODERMA-EOSINOPHILIA-LYMPHADENOPATHY",
            "RAG1-RAG2-SAME-LOCUS-11P13-SEQUENCE-BOTH-ALWAYS",
            "RAG1-MATERNAL-T-CELL-ENGRAFTMENT-EXCLUDE-BEFORE-HSCT",
        ],
        "distinguishing_feature": "T-B-NK+; hypomorphic → Omenn syndrome (erythroderma+eosinophilia+LN); 11p13 with RAG2 — always sequence both",
        "first_line_treatment": "HSCT — T-B-NK+ phenotype: excellent outcomes with HLA-matched unrelated; immunosuppression for Omenn syndrome pre-HSCT (cyclosporine/steroids)",
        "pathognomonic": "T-B-NK+ SCID OR erythroderma + eosinophilia + lymphadenopathy + hepatosplenomegaly (Omenn) with absent RAG1 activity",
        "onset_yrs_mean": 0.25,
        "penetrance_pct": 100,
        "t_cell_absent_pct": 100,
        "b_cell_absent_pct": 100,
        "nk_cell_absent_pct": 0,
        "radiosensitive_pct": 0,
        "deafness_pct": 0,
        "gene_therapy_eligible_pct": 10,
        "hsct_curative_pct": 85,
        "omenn_pct": 40,
        "maternal_engraftment_pct": 30,
        "dx_lag_yrs": 0.2,
    },
    {
        "name": "RAG2",
        "full_name": "Recombination Activating Gene 2",
        "aa": 527,
        "locus": "11p13",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "RAG2-SCID T-B-NK+ / Omenn Syndrome",
        "mw_kda": 59.3,
        "critical_rules": [
            "RAG2-T-MINUS-B-MINUS-NK-PLUS-ALLELIC-RAG1-PHENOTYPE",
            "RAG2-R229Q-NORTH-AFRICAN-FOUNDER-OMENN-SYNDROME",
            "RAG2-PHD-FINGER-H3K4ME3-CHROMATIN-MARK-VDJ-RECOMBINATION",
            "RAG2-RAG1-SAME-LOCUS-ALWAYS-SEQUENCE-TOGETHER",
        ],
        "distinguishing_feature": "T-B-NK+; R229Q North African founder → Omenn; PHD chromatin reader; always test with RAG1; identical phenotype to RAG1",
        "first_line_treatment": "HSCT curative; immunosuppression for Omenn pre-HSCT (cyclosporine+steroids reduce oligoclonal T cell activation); co-trimoxazole PCP prophylaxis",
        "pathognomonic": "T-B-NK+ SCID or Omenn syndrome + North African ancestry or R229Q variant + absent RAG2 protein function",
        "onset_yrs_mean": 0.25,
        "penetrance_pct": 100,
        "t_cell_absent_pct": 100,
        "b_cell_absent_pct": 100,
        "nk_cell_absent_pct": 0,
        "radiosensitive_pct": 0,
        "deafness_pct": 0,
        "gene_therapy_eligible_pct": 10,
        "hsct_curative_pct": 85,
        "omenn_pct": 45,
        "maternal_engraftment_pct": 30,
        "dx_lag_yrs": 0.2,
    },
    {
        "name": "DCLRE1C",
        "full_name": "Artemis DNA repair endonuclease",
        "aa": 692,
        "locus": "10p13",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "Artemis Radiosensitive SCID T-B-NK+ / Navajo-Apache Founder",
        "mw_kda": 78.3,
        "critical_rules": [
            "DCLRE1C-RADIOSENSITIVE-SCID-BUSULFAN-TBI-ABSOLUTELY-CI",
            "DCLRE1C-T-MINUS-B-MINUS-NK-PLUS-IDENTICAL-RAG1-RAG2-PHENOTYPE",
            "DCLRE1C-NAVAJO-APACHE-NATIVE-AMERICAN-FOUNDER-ALLELE",
            "DCLRE1C-REDUCED-INTENSITY-CONDITIONING-ONLY-NO-RADIATION",
        ],
        "distinguishing_feature": "Radiosensitive SCID — radiation/busulfan ABSOLUTELY CONTRAINDICATED; Navajo/Apache founder; T-B-NK+; reduced-intensity conditioning only",
        "first_line_treatment": "HSCT with REDUCED-INTENSITY CONDITIONING ONLY (no busulfan, NO TBI); treosulfan/fludarabine-based conditioning; test for radiosensitivity before HSCT",
        "pathognomonic": "T-B-NK+ SCID + hypersensitivity to ionising radiation (colony survival assay <10% at 2Gy) + Navajo/Apache ancestry OR failed standard conditioning → test DCLRE1C",
        "onset_yrs_mean": 0.2,
        "penetrance_pct": 100,
        "t_cell_absent_pct": 100,
        "b_cell_absent_pct": 100,
        "nk_cell_absent_pct": 0,
        "radiosensitive_pct": 100,
        "deafness_pct": 0,
        "gene_therapy_eligible_pct": 5,
        "hsct_curative_pct": 75,
        "omenn_pct": 10,
        "maternal_engraftment_pct": 25,
        "dx_lag_yrs": 0.3,
    },
    {
        "name": "JAK3",
        "full_name": "Janus kinase 3",
        "aa": 1124,
        "locus": "19p13.11",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "JAK3-SCID T-B+NK- / AR Phenocopy of XL-SCID",
        "mw_kda": 124.8,
        "critical_rules": [
            "JAK3-T-MINUS-B-PLUS-NK-MINUS-PHENOCOPY-IL2RG-XL-SCID",
            "JAK3-AR-BOTH-SEXES-FEMALE-T-MINUS-B-PLUS-NK-MINUS-TEST-JAK3-NOT-IL2RG",
            "JAK3-EXCLUSIVE-PARTNER-OF-GAMMA-C-IL2RG-DOWNSTREAM-SIGNALLING",
            "JAK3-INHIBITORS-TOFACITINIB-RUXOLITINIB-CAUSE-IATROGENIC-SCID",
        ],
        "distinguishing_feature": "T-B+NK- identical to XL-SCID but AR; FEMALE with T-B+NK- → test JAK3 not IL2RG; JAK3 inhibitors cause iatrogenic SCID phenotype",
        "first_line_treatment": "HSCT (outcomes identical to IL2RG-SCID); early diagnosis via TREC NBS; PCP prophylaxis; no gene therapy approved (IL2RG gene therapy does not correct JAK3)",
        "pathognomonic": "T-B+NK- SCID in a FEMALE infant (or male with negative IL2RG sequencing) + absent JAK3 protein + TREC=0",
        "onset_yrs_mean": 0.2,
        "penetrance_pct": 100,
        "t_cell_absent_pct": 100,
        "b_cell_absent_pct": 0,
        "nk_cell_absent_pct": 100,
        "radiosensitive_pct": 0,
        "deafness_pct": 0,
        "gene_therapy_eligible_pct": 15,
        "hsct_curative_pct": 88,
        "omenn_pct": 0,
        "maternal_engraftment_pct": 20,
        "dx_lag_yrs": 0.2,
    },
    {
        "name": "IL7R",
        "full_name": "Interleukin-7 receptor α chain",
        "aa": 459,
        "locus": "5p13.2",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "IL7R-SCID T-B+NK+ / Selective T Cell Absent Best HSCT Outcome",
        "mw_kda": 50.0,
        "critical_rules": [
            "IL7R-T-MINUS-B-PLUS-NK-PLUS-PURE-T-CELL-DEFICIENCY",
            "IL7R-BEST-HSCT-OUTCOME-SELECTIVE-T-CELL-RECONSTITUTION",
            "IL7R-MATERNAL-T-CELL-ENGRAFTMENT-HIGH-RISK-EXCLUDE-GVH",
            "IL7R-IL7-NON-REDUNDANT-FOR-HUMAN-T-CELL-THYMIC-DEVELOPMENT",
        ],
        "distinguishing_feature": "T-B+NK+ (UNIQUE — only T absent); best HSCT outcomes of all SCID types; maternal T cell engraftment common (can cause rash/cytopenia in infant)",
        "first_line_treatment": "HSCT curative — selective T cell reconstitution achieves normal immunity; check for maternal T cell engraftment (GVH manifestations) before conditioning; PCP prophylaxis",
        "pathognomonic": "T-B+NK+ lymphopenia + TREC=0 + normal B and NK cell counts with absent T cells + IL7Rα absent on T cell precursors",
        "onset_yrs_mean": 0.25,
        "penetrance_pct": 100,
        "t_cell_absent_pct": 100,
        "b_cell_absent_pct": 0,
        "nk_cell_absent_pct": 0,
        "radiosensitive_pct": 0,
        "deafness_pct": 0,
        "gene_therapy_eligible_pct": 20,
        "hsct_curative_pct": 95,
        "omenn_pct": 5,
        "maternal_engraftment_pct": 40,
        "dx_lag_yrs": 0.2,
    },
    {
        "name": "AK2",
        "full_name": "Adenylate kinase 2 (mitochondrial)",
        "aa": 239,
        "locus": "1p34.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "Reticular Dysgenesis T-B-NK- + Agranulocytosis + Congenital Deafness",
        "mw_kda": 26.5,
        "critical_rules": [
            "AK2-RETICULAR-DYSGENESIS-MOST-SEVERE-RAREST-SCID",
            "AK2-T-MINUS-B-MINUS-NK-MINUS-PLUS-AGRANULOCYTOSIS-ALL-LINEAGES",
            "AK2-CONGENITAL-SENSORINEURAL-DEAFNESS-PATHOGNOMONIC-BILATERAL",
            "AK2-EMPTY-BONE-MARROW-BIOPSY-MYELOID-AND-LYMPHOID-ARREST",
        ],
        "distinguishing_feature": "Most severe SCID: T-B-NK- PLUS agranulocytosis; bilateral congenital sensorineural deafness PATHOGNOMONIC; empty bone marrow; fatal in weeks without HSCT",
        "first_line_treatment": "URGENT HSCT (survival without treatment: 3-6 weeks); HSCT restores immunity but NOT hearing (cochlear AK2 non-haematopoietic); hearing aids post-HSCT; strict isolation until HSCT",
        "pathognomonic": "SCID + bilateral congenital sensorineural deafness + agranulocytosis + empty bone marrow biopsy → reticular dysgenesis → AK2 sequencing",
        "onset_yrs_mean": 0.05,
        "penetrance_pct": 100,
        "t_cell_absent_pct": 100,
        "b_cell_absent_pct": 100,
        "nk_cell_absent_pct": 100,
        "radiosensitive_pct": 0,
        "deafness_pct": 100,
        "gene_therapy_eligible_pct": 0,
        "hsct_curative_pct": 70,
        "omenn_pct": 0,
        "maternal_engraftment_pct": 5,
        "dx_lag_yrs": 0.05,
    },
]


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _make_cohort(gene_data, seed):
    rng = _rng(seed)
    n = 40
    g = gene_data
    base_age = max(0.03, g["onset_yrs_mean"])

    onset_ages = sorted([max(0.0, rng.gauss(base_age, max(0.02, base_age * 0.4))) for _ in range(n)])
    dx_lag = [max(0.03, rng.gauss(g.get("dx_lag_yrs", 0.2), 0.1)) for _ in range(n)]
    dx_ages = [o + d for o, d in zip(onset_ages, dx_lag)]

    def pct_n(field, default=0):
        return round(g.get(field, default) / 100 * n)

    return {
        "gene": g["name"],
        "full_name": g["full_name"],
        "locus": g["locus"],
        "inheritance": g["inheritance"],
        "syndrome": g["syndrome"],
        "n": n,
        "seed": seed,
        "mean_onset_age": round(sum(onset_ages) / n, 3),
        "median_onset_age": round(sorted(onset_ages)[n // 2], 3),
        "mean_dx_age": round(sum(dx_ages) / n, 3),
        "mean_dx_lag_yrs": round(sum(dx_lag) / n, 3),
        "t_cell_absent_n": pct_n("t_cell_absent_pct"),
        "b_cell_absent_n": pct_n("b_cell_absent_pct"),
        "nk_cell_absent_n": pct_n("nk_cell_absent_pct"),
        "radiosensitive_n": pct_n("radiosensitive_pct"),
        "deafness_n": pct_n("deafness_pct"),
        "gene_therapy_eligible_n": pct_n("gene_therapy_eligible_pct"),
        "hsct_curative_n": pct_n("hsct_curative_pct"),
        "omenn_n": pct_n("omenn_pct"),
        "maternal_engraftment_n": pct_n("maternal_engraftment_pct"),
        "pathognomonic": g["pathognomonic"],
        "first_line_treatment": g["first_line_treatment"],
        "critical_rules": g["critical_rules"],
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    total_patients = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-SCID-Atlas",
        "subtitle": (
            "Complete 8-Gene Severe Combined Immunodeficiency Reference: "
            "IL2RG/XL-SCID-T-B+NK--Most-Common-Gene-Therapy · ADA/T-B-NK--Enzyme-Replacement-Strimvelis · "
            "RAG1/T-B-NK+-Omenn-Spectrum · RAG2/T-B-NK+-North-African-R229Q · "
            "DCLRE1C/Artemis-Radiosensitive-SCID-No-Radiation · JAK3/AR-T-B+NK--Phenocopy-XL-SCID · "
            "IL7R/T-B+NK+-Best-HSCT-Outcome · AK2/Reticular-Dysgenesis-Deafness-All-Lineages"
        ),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["name"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "cohort_size": 40,
        "cohorts": cohorts,
        "key_facts": {
            "il2rg_most_common": "IL2RG: most common SCID (40-50%); XL (males only); T-B+NK-; gene therapy first approved; TREC=0 NBS",
            "ada_enzyme": "ADA-SCID: T-B-NK- (all lineages); ADA enzyme absent in RBCs; Strimvelis EMA 2016 first ex-vivo gene therapy; elapegademase bridges to cure",
            "rag_omenn": "RAG1/RAG2 (11p13): T-B-NK+ SCID or Omenn syndrome (hypomorphic); always sequence both; NK cells preserved",
            "dclre1c_radiation": "DCLRE1C/Artemis: RADIOSENSITIVE SCID — busulfan and TBI ABSOLUTELY CONTRAINDICATED; Navajo/Apache founder; reduced-intensity conditioning ONLY",
            "jak3_female": "JAK3: T-B+NK- phenocopy of XL-SCID but AR; FEMALE with T-B+NK- → test JAK3 not IL2RG; JAK3 inhibitors cause iatrogenic SCID",
            "il7r_selective": "IL7R: T-B+NK+ (ONLY T cells absent); BEST HSCT outcome of all SCID; IL-7 non-redundant for human T-cell thymic development",
            "ak2_reticular": "AK2/Reticular dysgenesis: MOST SEVERE SCID — T-B-NK- plus agranulocytosis; bilateral congenital deafness PATHOGNOMONIC; empty BM; fatal weeks without HSCT",
            "trec_nbs": "TREC (T-cell receptor excision circles) newborn screening: detects ALL SCID types — TREC=0 or severely low is universal finding; save lives via early detection",
        },
        "diagnostic_approach": {
            "trec_zero_nbs": "TREC=0 on NBS → urgent immunophenotyping (T/B/NK by flow cytometry) + SCID gene panel within 48h",
            "t_minus_b_plus_nk_minus": "T-B+NK-: most likely IL2RG (male) or JAK3 (female or male with negative IL2RG); both share γc-JAK3 cytokine axis",
            "t_minus_b_minus_nk_plus": "T-B-NK+: RAG1, RAG2, or DCLRE1C (Artemis); test radiosensitivity to distinguish Artemis from RAG deficiency",
            "t_minus_b_minus_nk_minus_lymphocytes_only": "T-B-NK-: ADA (purine metabolites in plasma/urine; ADA enzyme in RBCs) or IL2RG (severe); check deoxyadenosine level",
            "t_minus_b_minus_nk_minus_plus_agranulocytosis": "T-B-NK- PLUS agranulocytosis + bilateral deafness → Reticular dysgenesis / AK2; empty bone marrow biopsy",
            "omenn_syndrome": "Erythroderma + eosinophilia + lymphadenopathy + hepatosplenomegaly in infant → Omenn (hypomorphic RAG1/RAG2 most common); skin biopsy + gene panel",
            "t_minus_b_plus_nk_plus": "T-B+NK+: IL7R (most common); also consider CD3 chain deficiencies (CD3E/D/G/Z); check IL7Rα on progenitors",
            "radiosensitivity_testing": "Colony survival assay (2Gy gamma radiation, <10% survival) = radiosensitive → DCLRE1C/Artemis → NO TBI, NO busulfan for HSCT",
        },
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    genes_out = []
    for i, (g, c) in enumerate(zip(ATLAS_GENES, cohorts)):
        genes_out.append({
            "rank": i + 1,
            "gene": g["name"],
            "full_name": g["full_name"],
            "aa": g["aa"],
            "mw_kda": g["mw_kda"],
            "locus": g["locus"],
            "inheritance": g["inheritance"],
            "mechanism": g["mechanism"],
            "syndrome": g["syndrome"],
            "seed": SEED_BASE + i,
            "n": c["n"],
            "mean_onset_age": c["mean_onset_age"],
            "mean_dx_lag_yrs": c["mean_dx_lag_yrs"],
            "t_cell_absent_n": c["t_cell_absent_n"],
            "b_cell_absent_n": c["b_cell_absent_n"],
            "nk_cell_absent_n": c["nk_cell_absent_n"],
            "radiosensitive_n": c["radiosensitive_n"],
            "deafness_n": c["deafness_n"],
            "gene_therapy_eligible_n": c["gene_therapy_eligible_n"],
            "hsct_curative_n": c["hsct_curative_n"],
            "omenn_n": c["omenn_n"],
            "critical_rules": g["critical_rules"],
            "pathognomonic": g["pathognomonic"],
            "first_line_treatment": g["first_line_treatment"],
        })
    return {"genes": genes_out}


def generate_definitions():
    return {
        "atlas": "Hereditary-SCID-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": {
            "SCID Overview — T-Cell Receptor Excision Circles (TREC) Newborn Screening": (
                "Severe Combined Immunodeficiency (SCID) is a group of primary immunodeficiencies characterised by "
                "profound T lymphopenia ± absent B and NK cells, rendering affected infants susceptible to "
                "life-threatening infections from viral, bacterial, fungal, and opportunistic pathogens within the first "
                "months of life. Without treatment (HSCT or gene therapy), median survival is <1 year. "
                "TREC newborn screening: T-cell receptor excision circles are DNA episomes generated during V(D)J recombination "
                "in the thymus — present in naive T cells, absent in SCID. TREC=0 on dried blood spot = universal SCID marker. "
                "All 50 US states + many countries now screen. Early detection (pre-symptomatic) dramatically improves HSCT outcomes: "
                "survival >90% when transplanted before 3.5 months and before serious infection. "
                "Immunophenotyping categories: T-B+NK- (XL-SCID/JAK3), T-B-NK+ (RAG1/RAG2/Artemis), "
                "T-B-NK- (ADA, severe IL2RG), T-B+NK+ (IL7R) guide gene panel selection."
            ),
            "IL2RG / XL-SCID — Most Common, Gene Therapy Pioneer": (
                "IL2RG (γc, common γ chain) is shared by receptors for IL-2, IL-4, IL-7, IL-9, IL-15, IL-21. "
                "XL-SCID: T lymphopenia + absent NK cells; B cells present but non-functional (cannot receive T cell help). "
                "Affects males (X-linked); female carriers typically asymptomatic. "
                "Immunophenotype: T-B+NK-. TREC=0 on NBS. Frequent PCP pneumonia, CMV pneumonitis, failure to thrive. "
                "GENE THERAPY: retroviral vector-mediated HSC gene transfer (Paris cohort, 2000) = first gene therapy to achieve "
                "clinical success. Updated lentiviral vectors (no insertional mutagenesis risk). "
                "HSCT: excellent outcomes before 3.5 months with MSD; unrelated/haplo HSCT also effective with reduced conditioning. "
                "PCP prophylaxis with co-trimoxazole from diagnosis. No live vaccines until immune reconstitution confirmed."
            ),
            "ADA-SCID — Metabolic, Enzyme Replacement, Strimvelis Gene Therapy": (
                "Adenosine deaminase (ADA) catalyses conversion of adenosine → inosine and deoxyadenosine → deoxyinosine. "
                "ADA deficiency → toxic deoxyadenosine and dATP accumulation → lymphocyte apoptosis (T cells most sensitive). "
                "Immunophenotype: T-B-NK- (most profound lymphopenia — all lineages affected). "
                "Diagnostic test: ADA enzyme activity in erythrocytes — absent or severely reduced. "
                "Purine metabolite screening (deoxyadenosine in plasma/urine) confirms diagnosis. "
                "Enzyme replacement therapy: Elapegademase-lvlr (Revcovi®) IM weekly — stabilises immune function. "
                "Strimvelis® (EMA 2016, AstraZeneca/Orchard): ex-vivo autologous HSC lentiviral gene therapy; "
                "curative without donor requirement. First ex-vivo HSC gene therapy to achieve EMA approval. "
                "Bone radiograph: cupping and flaring of costochondral junctions (metabolic bone disease from purine toxicity). "
                "HSCT: gold standard if MSD available; enzyme replacement bridges to definitive therapy."
            ),
            "RAG1 and RAG2 — V(D)J Recombinase, T-B-NK+, Omenn Syndrome": (
                "RAG1/RAG2 form the V(D)J recombinase that creates antigen receptor diversity in T and B cells. "
                "Complete LOF (biallelic null) → no TCR or BCR → T-B-NK+ SCID (NK cells intact). "
                "RAG1: 1043aa, catalytic subunit; RAG2: 527aa, DNA binding subunit with PHD finger reading H3K4me3. "
                "Hypomorphic variants (partial residual activity) → Omenn syndrome: oligoclonal T cells expand → "
                "TRIAD: generalised erythroderma + hypereosinophilia + lymphadenopathy + hepatosplenomegaly. "
                "Omenn is a leaky SCID — NOT complete immunodeficiency: skin biopsy shows T cell infiltrate. "
                "RAG1 and RAG2 are encoded on the SAME locus (chromosome 11p13) — ALWAYS sequence both genes together. "
                "R229Q (RAG2): North African founder allele → Omenn syndrome. "
                "HSCT: T-B-NK+ phenotype with excellent MRD outcomes; immunosuppression pre-HSCT for Omenn (cyclosporine+steroids)."
            ),
            "DCLRE1C / Artemis — Radiosensitive SCID, Navajo Founder, NO TBI/Busulfan": (
                "Artemis (DCLRE1C) is a nuclease that cleaves DNA hairpin intermediates during V(D)J recombination AND "
                "functions in DNA double-strand break repair (NHEJ pathway). "
                "DUAL FUNCTION → two consequences of LOF: (1) T-B-NK+ SCID (RAG-like phenotype); "
                "(2) cellular RADIOSENSITIVITY (NHEJ repair failure). "
                "CRITICAL: Artemis-SCID patients cannot tolerate standard HSCT conditioning: "
                "busulfan (alkylating agent), cyclophosphamide, and TBI (total body irradiation) cause fatal toxicity. "
                "Use ONLY reduced-intensity/treosulfan-fludarabine-based conditioning. Test radiosensitivity BEFORE HSCT. "
                "Colony survival assay: <10-15% survival at 2 Gy gamma radiation = radiosensitive → Artemis/Ligase4/XRCC4 panel. "
                "Founder alleles: Navajo Nation (del1459-1461 + IVS3+2T>C) and San Carlos Apache populations. "
                "Immunophenotype: T-B-NK+ (identical to RAG deficiency on flow cytometry — radiosensitivity assay distinguishes). "
                "HSCT outcomes: 75% with reduced-intensity conditioning; poor engraftment historically with standard conditioning."
            ),
            "JAK3 — AR T-B+NK-, Phenocopy of XL-SCID, Female Diagnostic Key": (
                "JAK3 is the tyrosine kinase that exclusively partners with γc (IL2RG) and transduces signals from "
                "IL-2, IL-4, IL-7, IL-9, IL-15, IL-21 receptors. JAK3 and IL2RG are functionally inseparable. "
                "JAK3 LOF → IDENTICAL immunophenotype to XL-SCID: T-B+NK-. "
                "CRITICAL DISTINCTION: JAK3-SCID is AUTOSOMAL RECESSIVE — affects BOTH sexes equally. "
                "DIAGNOSTIC KEY: T-B+NK- SCID in a FEMALE infant → JAK3 (NOT IL2RG). "
                "T-B+NK- in a MALE with negative IL2RG sequencing → TEST JAK3. "
                "Iatrogenic phenocopy: JAK3/JAK1 inhibitors (tofacitinib, ruxolitinib, baricitinib, upadacitinib) "
                "used in RA/IBD/dermatology cause lymphopenia and opportunistic infections — mimic JAK3-SCID. "
                "HSCT outcomes: comparable to IL2RG-SCID; early treatment before 3.5 months essential. "
                "Gene therapy: IL2RG gene therapy does NOT correct JAK3 deficiency (different gene, same pathway step)."
            ),
            "IL7R — T-B+NK+, Selective T Cell Deficiency, Best HSCT Outcome": (
                "IL-7 receptor is a heterodimer of IL7Rα (CD127, encoded by IL7R) and γc (IL2RG). "
                "IL-7 signalling via IL7Rα-γc-JAK1/3-STAT5 is NON-REDUNDANT for human T cell development in thymus. "
                "Unlike mice (where IL-7 deficiency spares some T cells), human IL7R LOF = complete T cell absence. "
                "Immunophenotype: T-B+NK+ — the ONLY SCID with both B AND NK cells intact. "
                "CLINICAL FEATURE: maternal T cell engraftment is more common (40%) — maternal T cells detectable "
                "in infant blood; can cause GVH manifestations (rash, anaemia, hepatitis) — exclude before HSCT. "
                "BEST HSCT OUTCOME: selective T cell reconstitution achieves full immune function; "
                "B cells already present (no B cell reconstitution needed); NK cells intact. "
                "Survival post-HSCT: >95% in specialised centres. PCP prophylaxis until T cell reconstitution confirmed."
            ),
            "AK2 / Reticular Dysgenesis — Most Severe SCID, Deafness, All Lineages Absent": (
                "Adenylate kinase 2 (AK2) is the mitochondrial isoform that interconverts AMP+ATP ↔ 2ADP — "
                "critical for mitochondrial energy homeostasis in rapidly dividing progenitor cells. "
                "AK2 expressed in: bone marrow haematopoietic progenitors AND inner ear stria vascularis. "
                "AK2 LOF → haematopoietic arrest at myeloid progenitor (AMP+ADP energy failure): "
                "T-B-NK- PLUS absent granulocytes (agranulocytosis) — no innate OR adaptive immunity. "
                "PATHOGNOMONIC: bilateral congenital sensorineural deafness (100% penetrance) — "
                "AK2 in stria vascularis essential for endocochlear potential maintenance. "
                "Bone marrow biopsy: EMPTY — reticular dysgenesis = absent myeloid AND lymphoid progenitors. "
                "Severity: fatal within 3-6 weeks without HSCT (combined innate + adaptive failure). "
                "HSCT: immune reconstitution achieved; hearing NOT restored (cochlear AK2 is non-haematopoietic). "
                "Post-HSCT: hearing aids; monitor thyroid (AK2 expressed in multiple glandular tissues)."
            ),
            "HSCT for SCID — Timing, Conditioning, and Gene Therapy Selection": (
                "HSCT outcomes in SCID are strongly time-dependent: survival >90% if transplanted <3.5 months and pre-infection. "
                "Donor hierarchy: (1) HLA-matched sibling (MSD); (2) matched unrelated (MUD); (3) haploidentical (haplo-HSCT); "
                "(4) mismatched unrelated. T-cell depleted haplo-HSCT acceptable for XL-SCID/ADA-SCID. "
                "CONDITIONING: "
                "• XL-SCID, JAK3: no conditioning needed for MSD (T cell reconstitution without conditioning); "
                "  reduced-intensity (busulfan-based) for full B cell reconstitution. "
                "• RAG1/RAG2: reduced-intensity acceptable; myeloablative for full reconstitution. "
                "• DCLRE1C/Artemis: NEVER myeloablative; treosulfan-fludarabine ONLY — no busulfan, NO TBI. "
                "• ADA-SCID: reduced-intensity; gene therapy (Strimvelis) if no MSD. "
                "• AK2: myeloablative often needed for granulocyte reconstitution; calculate RIC vs outcomes carefully. "
                "GENE THERAPY: IL2RG (lentiviral), ADA (Strimvelis/lentiviral), RAG1 (clinical trials) — not approved for others. "
                "Infection prophylaxis during HSCT: co-trimoxazole (PCP), aciclovir (HSV/VZV), fluconazole (fungal), "
                "monthly IVIG until IgG production confirmed post-HSCT."
            ),
        },
        "standards": [
            "ESID/EBMT SCID guidelines — Lankester AC et al. J Allergy Clin Immunol 2022",
            "TREC newborn screening outcomes — Pai SY et al. NEJM 2014",
            "ADA-SCID gene therapy Strimvelis — Cicalese MP et al. NEJM 2016",
            "IL2RG gene therapy lentiviral vector — Hacein-Bey Abina S et al. NEJM 2010",
            "Artemis/DCLRE1C radiosensitive SCID — Moshous D et al. Cell 2001",
            "TREC population screening — Routes JM et al. JAMA 2009",
            "Reticular dysgenesis AK2 — Lagresle-Peyrou C et al. Nat Genet 2009",
            "RAG1/RAG2 Omenn syndrome — Villa A et al. Cell 1998",
            "SCID HSCT outcomes European experience — Gennery AR et al. Blood 2010",
            "Primary immunodeficiency classification IUIS 2022",
        ],
    }
