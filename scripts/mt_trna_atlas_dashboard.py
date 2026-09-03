#!/usr/bin/env python3
"""MT-tRNA-Atlas — Complete 22-Gene Mitochondrial tRNA Genome Atlas
All 22 human mitochondrial tRNA genes | Genome-wide summary | Combined-CI+CIV CPEO-Myopathy-Spectrum

The human mitochondrial genome (rCRS 16,569 bp) encodes exactly 22 transfer RNA genes
that are required for translation of all 13 mt-encoded OXPHOS subunits. This Atlas
aggregates all 22 mt-tRNA genes into one cross-gene reference: genomic positions,
strand encoding, primary disease phenotypes, nuclear DDx, and universal drug CIs.

STRAND DISTRIBUTION (22 genes):
  H-strand (heavy, sense): 14 genes — MT-TF, MT-TV, MT-TL1, MT-TI, MT-TM, MT-TW,
                                       MT-TD, MT-TK, MT-TG, MT-TR, MT-TH, MT-TS2,
                                       MT-TL2, MT-TT
  L-strand (light, antisense): 8 genes — MT-TQ, MT-TA, MT-TN, MT-TC, MT-TY, MT-TS1,
                                           MT-TE, MT-TP
  L-strand genes require reverse-complement QC in standard NGS — mandatory separate
  analysis window; reported as "NGS pitfall" in individual gene dashboards.

OXPHOS DEFICIENCY PATTERN:
  Combined CI+CIV (mt-translation fingerprint): 20 of 22 genes → CPEO-Myopathy spectrum
  Distinctive MELAS (MT-TL1): CI+CIII+CIV pan-OXPHOS, stroke-like episodes HALLMARK
  Distinctive MERRF (MT-TK): CI+CIV+CV pan-OXPHOS, myoclonic epilepsy HALLMARK
  L-Strand cluster (MT-TA/TN/TC/TY): Combined CI+CIV, overlapping large-deletion risk

UNIVERSAL DRUG CONTRAINDICATIONS (ALL 22 mt-tRNA GENES):
  ABSOLUTE CI: Metformin, VPA (valproate), Propofol (PRIS), Linezolid, Chloramphenicol
  CONTRAINDICATED: Ketogenic Diet (KD)
  MANDATORY PROTOCOL: GIR 6–8 mg/kg/min (never fast), Thiamine B1 empiric, Biotin empiric
  PREFERRED AED: Levetiracetam (LEV)

WES LIMITATION (ALL 22 mt-tRNA GENES):
  Whole-Exome Sequencing MISSES all mt-tRNA genes — targeted mtDNA panel mandatory.
  L-strand genes additionally require dedicated L-strand QC window.

BTBGD EXCLUSION (ALL 22 mt-tRNA GENES):
  SLC19A3 (Biotin-Thiamine-Responsive Basal Ganglia Disease) must be excluded FIRST
  in all Leigh-like presentations — treatable mimic with identical MRI pattern.

COHORT: 22 × 40-patient gene-specific cohorts (880 total patient slots; seeds 799–831)
  Aggregate: 880 patients across all 22 genes (seed range 799–831, odd numbered)
"""

import random

SEED = 831
rng  = random.Random(SEED)

# ── All 22 mt-tRNA genes — authoritative table ───────────────────────────────
MT_TRNA_GENES = [
    {
        "gene": "MT-TF",  "aa": "Phe",    "anticodon": "GAA",
        "strand": "H",    "rcrs_start": 577,   "rcrs_end": 647,   "length_nt": 71,
        "omim_gene": 590070,
        "ordinal": 1,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "FIRST tRNA; D-loop adjacent; large deletions extend into D-loop / OH",
        "ddx_nuclear": "FARS2 — Neonatal Epileptic Encephalopathy (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.611T>C — Anticodon Loop/Stem Junction — CPEO+Myopathy ~30%",
        "ngs_pitfall": False,
        "seed": 803,
    },
    {
        "gene": "MT-TV",  "aa": "Val",    "anticodon": "UAC",
        "strand": "H",    "rcrs_start": 1602,  "rcrs_end": 1670,  "length_nt": 69,
        "omim_gene": 590105,
        "ordinal": 2,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / SNHL",
        "hallmark": "Between 12S-rRNA and 16S-rRNA; large deletions may impair mt-ribosome assembly",
        "ddx_nuclear": "VARS2 — Neonatal HCM + Lactic Acidosis (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.1624C>T — Anticodon Stem — CPEO+Myopathy ~30%",
        "ngs_pitfall": False,
        "seed": 805,
    },
    {
        "gene": "MT-TL1", "aa": "Leu(UUR)", "anticodon": "UAA",
        "strand": "H",    "rcrs_start": 3230,  "rcrs_end": 3304,  "length_nt": 75,
        "omim_gene": 590050,
        "ordinal": 3,
        "phenotype": "MELAS / MIDD — Pan-OXPHOS (CI+CIII+CIV) — Stroke-like Episodes HALLMARK",
        "hallmark": "MELAS: m.3243A>G most common mtDNA mutation; stroke-like episodes cross vascular territories; MIDD diabetes lean patients",
        "ddx_nuclear": "LARS2 — Perrault Syndrome + HVDRR (NOT MELAS); WES-detectable AR",
        "most_common_variant": "m.3243A>G — Anticodon Stem — MELAS+MIDD ~80% of MT-TL1 disease",
        "ngs_pitfall": False,
        "seed": 587,
    },
    {
        "gene": "MT-TI",  "aa": "Ile",    "anticodon": "GAU",
        "strand": "H",    "rcrs_start": 4263,  "rcrs_end": 4331,  "length_nt": 69,
        "omim_gene": 590045,
        "ordinal": 4,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy — Isolated HCM DISTINCTIVE (m.4300A>G)",
        "hallmark": "m.4300A>G: Isolated HCM without CPEO/myopathy — UNIQUE among all 22 mt-tRNAs; cardiac phenotype predominant",
        "ddx_nuclear": "IARS2 — Cavitary Optic Disc + GTPBP3-like (NOT isolated HCM); WES-detectable AR",
        "most_common_variant": "m.4300A>G — Acceptor Stem — Isolated HCM DISTINCTIVE ~35%",
        "ngs_pitfall": False,
        "seed": 809,
    },
    {
        "gene": "MT-TQ",  "aa": "Gln",    "anticodon": "UUG",
        "strand": "L",    "rcrs_start": 4329,  "rcrs_end": 4400,  "length_nt": 72,
        "omim_gene": 590030,
        "ordinal": 5,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "L-strand NGS pitfall; 3 nt overlap with MT-TI (rCRS 4329–4331) — shared boundary; QARS2 neonatal encephalopathy DDx",
        "ddx_nuclear": "QARS2 — Neonatal Encephalopathy + Microcephaly (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.4363A>G — Anticodon Stem — CPEO+Myopathy ~28%",
        "ngs_pitfall": True,
        "seed": 811,
    },
    {
        "gene": "MT-TM",  "aa": "Met",    "anticodon": "CAU",
        "strand": "H",    "rcrs_start": 4402,  "rcrs_end": 4469,  "length_nt": 68,
        "omim_gene": 590065,
        "ordinal": 6,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "CAU anticodon decodes both AUG (Met) and AUA (Ile) in mt-code via C34 modification; dual codon role",
        "ddx_nuclear": "MARS2 — ARSAL + Spastic Ataxia (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.4435A>G — Anticodon Stem — CPEO+Myopathy ~28%",
        "ngs_pitfall": False,
        "seed": 813,
    },
    {
        "gene": "MT-TW",  "aa": "Trp",    "anticodon": "UCA",
        "strand": "H",    "rcrs_start": 5512,  "rcrs_end": 5579,  "length_nt": 68,
        "omim_gene": 590095,
        "ordinal": 7,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy (CI>CIV asymmetry DISTINCTIVE)",
        "hallmark": "UGA recoding: mt-tRNA-Trp reads UGA as Trp (not Stop); CI-ND subunits more affected than CIV-COX (Trp-rich ND1-6/4L)",
        "ddx_nuclear": "WARS2 — Intellectual Disability + Parkinsonism (NOT mt-CPEO); WES-detectable AR",
        "most_common_variant": "m.5540G>A — Anticodon Loop — CPEO+Myopathy ~30%",
        "ngs_pitfall": False,
        "seed": 815,
    },
    {
        "gene": "MT-TA",  "aa": "Ala",    "anticodon": "UGC",
        "strand": "L",    "rcrs_start": 5587,  "rcrs_end": 5655,  "length_nt": 69,
        "omim_gene": 590000,
        "ordinal": 8,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "FIRST of four consecutive L-strand tRNAs (MT-TA/TN/TC/TY cluster rCRS 5587–5891); cluster L-strand QC mandatory",
        "ddx_nuclear": "AARS2 — Ovario-Leukodystrophy (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.5628C>T — Anticodon Loop — CPEO+Myopathy ~28%",
        "ngs_pitfall": True,
        "seed": 817,
    },
    {
        "gene": "MT-TN",  "aa": "Asn",    "anticodon": "GTT",
        "strand": "L",    "rcrs_start": 5657,  "rcrs_end": 5729,  "length_nt": 73,
        "omim_gene": 590010,
        "ordinal": 9,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "SECOND of four consecutive L-strand tRNAs (MT-TA/TN cluster); 2 nt gap from MT-TA; compound deletion risk",
        "ddx_nuclear": "NARS2 — Perrault Syndrome Type 6 SNHL+POI (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.5690A>G — Anticodon Stem — CPEO+Myopathy ~28%",
        "ngs_pitfall": True,
        "seed": 817,
    },
    {
        "gene": "MT-TC",  "aa": "Cys",    "anticodon": "GCA",
        "strand": "L",    "rcrs_start": 5761,  "rcrs_end": 5826,  "length_nt": 66,
        "omim_gene": 590020,
        "ordinal": 10,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "THIRD of four consecutive L-strand tRNAs (MT-TC cluster); compound deletion with MT-TY removes both Cys+Tyr mt-tRNAs",
        "ddx_nuclear": "CARS2 — Combined OXPHOS (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.5798C>T — Anticodon Loop — CPEO+Myopathy ~30%",
        "ngs_pitfall": True,
        "seed": 819,
    },
    {
        "gene": "MT-TY",  "aa": "Tyr",    "anticodon": "GUA",
        "strand": "L",    "rcrs_start": 5826,  "rcrs_end": 5891,  "length_nt": 66,
        "omim_gene": 590100,
        "ordinal": 11,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "FOURTH/FINAL consecutive L-strand tRNA; 0 nt gap shared with MT-TC (5826); transition L-strand→H-strand at rCRS 5904 (MT-CO1)",
        "ddx_nuclear": "YARS2 — MLASA2 Sideroblastic Anaemia + Myopathy (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.5877G>A — Acceptor Stem — CPEO+Myopathy ~28%",
        "ngs_pitfall": True,
        "seed": 821,
    },
    {
        "gene": "MT-TS1", "aa": "Ser(UCN)", "anticodon": "UGA",
        "strand": "L",    "rcrs_start": 7445,  "rcrs_end": 7516,  "length_nt": 72,
        "omim_gene": 590080,
        "ordinal": 12,
        "phenotype": "SNHL DOMINANT at low heteroplasmy (<40%) — CPEO+Myopathy at high heteroplasmy (>60%)",
        "hallmark": "m.7445A>G: DUAL-GENE BOUNDARY — simultaneously last nt MT-CO1 (p.Tyr544 stop) and first nt MT-TS1; SNHL at low heteroplasmy UNIQUE",
        "ddx_nuclear": "SARS2 — HUPRA Syndrome Hyperuricemia+Pulmonary+Renal+Alkalosis Neonatal (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.7445A>G — Dual Gene Boundary / Acceptor Stem — SNHL+CPEO ~28%",
        "ngs_pitfall": True,
        "seed": 823,
    },
    {
        "gene": "MT-TD",  "aa": "Asp",    "anticodon": "GUC",
        "strand": "H",    "rcrs_start": 7518,  "rcrs_end": 7585,  "length_nt": 68,
        "omim_gene": 590015,
        "ordinal": 13,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "0 nt gap shared boundary with MT-CO2 3'-end (rCRS 7585/7586); compound deletion spans MT-TS1+MT-TD (Ser+Asp dual loss)",
        "ddx_nuclear": "DARS2 — LBSL Leukoencephalopathy Brain Stem + Spinal Cord + Lactate (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.7543G>A — Variable Loop — CPEO+Myopathy ~28%",
        "ngs_pitfall": False,
        "seed": 825,
    },
    {
        "gene": "MT-TK",  "aa": "Lys",    "anticodon": "UUU",
        "strand": "H",    "rcrs_start": 8295,  "rcrs_end": 8364,  "length_nt": 70,
        "omim_gene": 590060,
        "ordinal": 14,
        "phenotype": "MERRF — Myoclonic Epilepsy + Ragged-Red Fibres — Pan-OXPHOS (CI+CIV+CV)",
        "hallmark": "m.8344A>G: MERRF hallmark — myoclonic epilepsy + RRF + SNHL + short stature; tau-m5U34 modification loss; MSL (multiple symmetrical lipomatosis) DISTINCTIVE",
        "ddx_nuclear": "KARS2 — Combined OXPHOS + Neurodegeneration (NOT MERRF); WES-detectable AR",
        "most_common_variant": "m.8344A>G — Anticodon Stem — MERRF+RRF ~80% of MT-TK disease",
        "ngs_pitfall": False,
        "seed": 563,
    },
    {
        "gene": "MT-TG",  "aa": "Gly",    "anticodon": "GCC",
        "strand": "H",    "rcrs_start": 9991,  "rcrs_end": 10058, "length_nt": 68,
        "omim_gene": 590035,
        "ordinal": 15,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "GCC anticodon decodes GGC (W-C) and GGU (G34-U wobble); GxxxG motifs in COX1/2/3 and ND subunits — CIV+CI both impaired",
        "ddx_nuclear": "GARS2 — Combined OXPHOS Neonatal (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.9997T>C — Acceptor Stem — CPEO+Myopathy ~28%",
        "ngs_pitfall": False,
        "seed": 827,
    },
    {
        "gene": "MT-TR",  "aa": "Arg",    "anticodon": "UCG",
        "strand": "H",    "rcrs_start": 10405, "rcrs_end": 10469, "length_nt": 65,
        "omim_gene": 590005,
        "ordinal": 16,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "UCG superwobble: decodes all four CGN codons (CGA/CGG/CGC/CGU); 0 nt gap both sides (ND3 + ND4L shared boundaries); KSS 4977bp deletion removes MT-TR+TH+TS2+TL2",
        "ddx_nuclear": "RARS2 — PCH6 Pontocerebellar Hypoplasia Neonatal (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.10450A>G — T-stem — CPEO+Myopathy ~28%",
        "ngs_pitfall": False,
        "seed": 829,
    },
    {
        "gene": "MT-TH",  "aa": "His",    "anticodon": "GUG",
        "strand": "H",    "rcrs_start": 12138, "rcrs_end": 12206, "length_nt": 69,
        "omim_gene": 590040,
        "ordinal": 17,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance — Leigh-like severe",
        "hallmark": "GUG anticodon decodes CAC+CAU (His codons); H-strand-encoded ND4-ND5 territory; KSS 4977bp deletion removes MT-TH+TS2+TL2+TR",
        "ddx_nuclear": "HARS2 — Perrault Syndrome Type 2 SNHL+Ovarian Failure (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.12183G>A — T-stem — CPEO+Myopathy ~28%",
        "ngs_pitfall": False,
        "seed": 833,
    },
    {
        "gene": "MT-TS2", "aa": "Ser(AGY)", "anticodon": "GCU",
        "strand": "H",    "rcrs_start": 12207, "rcrs_end": 12265, "length_nt": 59,
        "omim_gene": 590085,
        "ordinal": 18,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / SNHL",
        "hallmark": "SHORTEST mt-tRNA (59 nt) — lacks D-arm entirely; GCU anticodon decodes AGC+AGU (Ser codons); KSS 4977bp deletion removes MT-TS2+TL2+TH+TR",
        "ddx_nuclear": "SARS — Cytoplasmic isoform → CMT2N (NOT mt-disease); mt-isoform SARS2 different DDx from MT-TS1",
        "most_common_variant": "m.12246G>A — Anticodon Stem — CPEO+Myopathy+SNHL ~30%",
        "ngs_pitfall": False,
        "seed": 835,
    },
    {
        "gene": "MT-TL2", "aa": "Leu(CUN)", "anticodon": "UAG",
        "strand": "H",    "rcrs_start": 12266, "rcrs_end": 12336, "length_nt": 71,
        "omim_gene": 590055,
        "ordinal": 19,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "UAG anticodon decodes CUG+CUA+CUC+CUU (Leu CUN codons; distinct from MT-TL1 UUA/UUG codons); KSS 4977bp deletion removes MT-TL2+TS2+TH+TR",
        "ddx_nuclear": "LARS2 — Perrault Syndrome (also DDx for MT-TL1 — same synthetase family); WES-detectable AR",
        "most_common_variant": "m.12315G>A — Anticodon Loop — CPEO+Myopathy ~28%",
        "ngs_pitfall": False,
        "seed": 837,
    },
    {
        "gene": "MT-TE",  "aa": "Glu",    "anticodon": "UUC",
        "strand": "L",    "rcrs_start": 14674, "rcrs_end": 14742, "length_nt": 69,
        "omim_gene": 590025,
        "ordinal": 20,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy — MIDM Maternally-Inherited Diabetes DISTINCTIVE",
        "hallmark": "m.14709T>C: MIDM (Maternally Inherited Diabetes + Myopathy) — diabetes DOMINANT without CPEO UNIQUE; m.14674T>C: reversible infantile COX deficiency — recovery at 6-18 months UNIQUE",
        "ddx_nuclear": "EARS2 — LTBL Leukoencephalopathy Thalamus + Brainstem + Lactate (NOT adult CPEO); WES-detectable AR",
        "most_common_variant": "m.14709T>C — Acceptor Stem — MIDM Diabetes+Myopathy ~38%",
        "ngs_pitfall": True,
        "seed": 839,
    },
    {
        "gene": "MT-TT",  "aa": "Thr",    "anticodon": "UGU",
        "strand": "H",    "rcrs_start": 15888, "rcrs_end": 15953, "length_nt": 66,
        "omim_gene": 590094,
        "ordinal": 21,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy — HCM Cardiomyopathy MOST DISTINCTIVE (55–65%)",
        "hallmark": "HCM 55–65% HIGHEST cardiomyopathy prevalence of all 22 mt-tRNAs; WPW + AV block 20%; AMIODARONE ABSOLUTE CI (mt-OXPHOS inhibitor); annual Echo+Holter MANDATORY",
        "ddx_nuclear": "TARS2 — Combined OXPHOS less cardiac (NOT isolated HCM like m.4300A>G MT-TI); ANT1/SLC25A4 — adPEO+Cardiomyopathy AD",
        "most_common_variant": "m.15923A>G — Anticodon Loop position 34 — CPEO+Myopathy+HCM ~35%",
        "ngs_pitfall": False,
        "seed": 799,
    },
    {
        "gene": "MT-TP",  "aa": "Pro",    "anticodon": "UGG",
        "strand": "L",    "rcrs_start": 15956, "rcrs_end": 16023, "length_nt": 68,
        "omim_gene": 590016,
        "ordinal": 22,
        "phenotype": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance",
        "hallmark": "FINAL tRNA in mt-genome; D-loop adjacent (rCRS 16024+); L-strand NGS pitfall; large deletions extend into D-loop OH replication origin",
        "ddx_nuclear": "PARS2 — Combined CI+CIV (identical fingerprint to MT-TP; WES-detectable AR; contrast: WES detects PARS2, misses MT-TP)",
        "most_common_variant": "m.15990C>T — Variable Loop Boundary — CPEO+Myopathy ~32%",
        "ngs_pitfall": True,
        "seed": 801,
    },
]

N_PATIENTS_PER_GENE = 40
TOTAL_PATIENTS      = N_PATIENTS_PER_GENE * len(MT_TRNA_GENES)  # 880
H_STRAND_GENES      = [g for g in MT_TRNA_GENES if g["strand"] == "H"]
L_STRAND_GENES      = [g for g in MT_TRNA_GENES if g["strand"] == "L"]

UNIVERSAL_ABSOLUTE_CI = ["Metformin", "VPA (Valproate)", "Propofol (PRIS)", "Linezolid", "Chloramphenicol"]
UNIVERSAL_CONTRAINDICATED = ["Ketogenic Diet (KD)", "Fasting (GIR 6–8 mandatory)"]
UNIVERSAL_MANDATORY = ["Thiamine B1 (empiric)", "Biotin (empiric)", "BTBGD/SLC19A3 exclusion first-line"]
PREFERRED_AED = "Levetiracetam (LEV)"

# HALLMARK-UNIQUE genes (distinctive from standard CPEO/Myopathy pattern)
HALLMARK_GENES = {
    "MT-TL1":  "MELAS — stroke-like episodes cross vascular territories; m.3243A>G commonest pathogenic mtDNA mutation",
    "MT-TK":   "MERRF — myoclonic epilepsy + RRF + MSL (multiple symmetrical lipomatosis)",
    "MT-TI":   "Isolated HCM (m.4300A>G) — cardiomyopathy WITHOUT CPEO/myopathy — UNIQUE",
    "MT-TS1":  "SNHL at LOW heteroplasmy (<40%) — non-syndromic SNHL presentation — UNIQUE",
    "MT-TE":   "MIDM (maternally inherited diabetes + myopathy) — diabetes DOMINANT — UNIQUE; m.14674T>C reversible infantile COX deficiency",
    "MT-TT":   "HCM 55–65% HIGHEST cardiomyopathy rate; AMIODARONE ABSOLUTE CI; WPW/AV block 20%",
    "MT-TW":   "CI>CIV asymmetry (UGA recoding — Trp rich in ND subunits); UNIQUE asymmetric fingerprint",
}

# Large-deletion compound losses (KSS 4977bp deletion most common)
KSS_4977BP_GENES = ["MT-TR", "MT-TH", "MT-TS2", "MT-TL2"]
KSS_DELETION_REMARK = "Common 4977 bp deletion (rCRS 8470–13447) removes MT-TR+TH+TS2+TL2 — KSS compound multi-tRNA loss"


def get_overview() -> dict:
    h_genes = [g["gene"] for g in H_STRAND_GENES]
    l_genes = [g["gene"] for g in L_STRAND_GENES]
    ngs_pitfall_genes = [g["gene"] for g in MT_TRNA_GENES if g["ngs_pitfall"]]

    return {
        "title": "MT-tRNA Atlas — Complete 22-Gene Mitochondrial tRNA Genome",
        "subtitle": "All 22 human mt-tRNA genes | rCRS positions | OXPHOS phenotypes | Nuclear DDx | Drug CIs",
        "total_genes": len(MT_TRNA_GENES),
        "total_cohort_patients": TOTAL_PATIENTS,
        "strand_distribution": {
            "H_strand": {"count": len(H_STRAND_GENES), "genes": h_genes,
                          "note": "Standard NGS pipeline — no special QC window required"},
            "L_strand": {"count": len(L_STRAND_GENES), "genes": l_genes,
                          "note": "NGS pitfall — reverse-complement QC window MANDATORY for all 8 L-strand mt-tRNA genes"},
        },
        "genome_span": {
            "first_tRNA": {"gene": "MT-TF", "rcrs_start": 577,   "note": "D-loop adjacent (after rCRS 576)"},
            "last_tRNA":  {"gene": "MT-TP", "rcrs_end":   16023, "note": "D-loop adjacent (before rCRS 16024)"},
            "total_rCRS_span_bp": 16023 - 577,
            "note": "22 tRNA genes span ~15 kb of the 16,569 bp mt-genome",
        },
        "oxphos_patterns": {
            "combined_CI_CIV_CPEO_myopathy": {
                "genes": [g["gene"] for g in MT_TRNA_GENES
                           if "MELAS" not in g["phenotype"] and "MERRF" not in g["phenotype"]],
                "count": len([g for g in MT_TRNA_GENES
                              if "MELAS" not in g["phenotype"] and "MERRF" not in g["phenotype"]]),
                "description": "Standard mt-translation fingerprint: Combined CI+CIV deficiency; CII (SDH) normal; CPEO + Limb Myopathy",
            },
            "melas_pan_oxphos": {
                "genes": ["MT-TL1"],
                "description": "Pan-OXPHOS CI+CIII+CIV; stroke-like episodes cross vascular territories; HALLMARK of MELAS",
            },
            "merrf_pan_oxphos": {
                "genes": ["MT-TK"],
                "description": "Pan-OXPHOS CI+CIV+CV; myoclonic epilepsy + RRF + MSL; HALLMARK of MERRF",
            },
        },
        "hallmark_unique_phenotypes": HALLMARK_GENES,
        "ngs_pitfall_genes": {
            "count": len(ngs_pitfall_genes),
            "genes": ngs_pitfall_genes,
            "note": "L-strand encoded — all require dedicated reverse-complement QC window in NGS analysis",
        },
        "wes_limitation": "WES MISSES all 22 mt-tRNA genes — targeted mtDNA sequencing panel mandatory for any suspected mt-tRNA disease",
        "universal_absolute_ci": UNIVERSAL_ABSOLUTE_CI,
        "universal_contraindicated": UNIVERSAL_CONTRAINDICATED,
        "universal_mandatory": UNIVERSAL_MANDATORY,
        "preferred_aed": PREFERRED_AED,
        "kss_4977bp_deletion": {
            "removes_genes": KSS_4977BP_GENES,
            "remark": KSS_DELETION_REMARK,
        },
        "btbgd_slc19a3_exclusion": "Biotin-Thiamine-Responsive Basal Ganglia Disease (SLC19A3) — treatable Leigh-like mimic — MUST be excluded first in ALL mt-tRNA Leigh-like presentations",
    }


def get_breakdown() -> dict:
    gene_rows = []
    for g in MT_TRNA_GENES:
        hallmark_flag = g["gene"] in HALLMARK_GENES
        kss_flag      = g["gene"] in KSS_4977BP_GENES
        gene_rows.append({
            "ordinal":              g["ordinal"],
            "gene":                 g["gene"],
            "amino_acid":           g["aa"],
            "anticodon":            g["anticodon"],
            "strand":               g["strand"],
            "rcrs_start":           g["rcrs_start"],
            "rcrs_end":             g["rcrs_end"],
            "length_nt":            g["length_nt"],
            "omim_gene":            g["omim_gene"],
            "primary_phenotype":    g["phenotype"],
            "hallmark":             g["hallmark"],
            "nuclear_ddx":          g["ddx_nuclear"],
            "most_common_variant":  g["most_common_variant"],
            "ngs_pitfall":          g["ngs_pitfall"],
            "kss_4977bp_affected":  kss_flag,
            "hallmark_unique":      hallmark_flag,
            "hallmark_detail":      HALLMARK_GENES.get(g["gene"], None),
            "cohort_n":             N_PATIENTS_PER_GENE,
            "cohort_seed":          g["seed"],
            "dashboard_route":      f"/{g['gene'].lower().replace('-', '')}",
            "api_base":             f"/api/{g['gene'].lower().replace('-', '')}",
        })

    # Strand breakdown stats
    h_lengths = [g["length_nt"] for g in H_STRAND_GENES]
    l_lengths = [g["length_nt"] for g in L_STRAND_GENES]

    return {
        "gene_table": gene_rows,
        "summary_stats": {
            "total_genes": 22,
            "h_strand_count": len(H_STRAND_GENES),
            "l_strand_count": len(L_STRAND_GENES),
            "ngs_pitfall_count": len([g for g in MT_TRNA_GENES if g["ngs_pitfall"]]),
            "hallmark_unique_count": len(HALLMARK_GENES),
            "kss_4977bp_affected_count": len(KSS_4977BP_GENES),
            "mean_length_h_strand_nt": round(sum(h_lengths) / len(h_lengths), 1),
            "mean_length_l_strand_nt": round(sum(l_lengths) / len(l_lengths), 1),
            "shortest_gene": min(MT_TRNA_GENES, key=lambda x: x["length_nt"])["gene"],
            "shortest_length_nt": min(g["length_nt"] for g in MT_TRNA_GENES),
            "longest_gene": max(MT_TRNA_GENES, key=lambda x: x["length_nt"])["gene"],
            "longest_length_nt": max(g["length_nt"] for g in MT_TRNA_GENES),
            "total_nucleotides_all_trna": sum(g["length_nt"] for g in MT_TRNA_GENES),
        },
        "phenotype_breakdown": {
            "cpeo_myopathy_standard": {
                "genes": [g["gene"] for g in MT_TRNA_GENES
                           if "MELAS" not in g["phenotype"] and "MERRF" not in g["phenotype"]],
                "description": "Combined CI+CIV; CII normal; CPEO primary; Myopathy; Exercise Intolerance",
            },
            "melas_hallmark": {
                "genes": ["MT-TL1"],
                "description": "Pan-OXPHOS; Stroke-like Episodes cross vascular territories; MELAS/MIDD",
            },
            "merrf_hallmark": {
                "genes": ["MT-TK"],
                "description": "Pan-OXPHOS; Myoclonic Epilepsy; RRF; Multiple Symmetrical Lipomatosis",
            },
            "isolated_hcm_hallmark": {
                "genes": ["MT-TI"],
                "description": "m.4300A>G — Isolated HCM WITHOUT CPEO/myopathy — UNIQUE fingerprint",
            },
            "snhl_low_heteroplasmy": {
                "genes": ["MT-TS1"],
                "description": "Non-syndromic SNHL at <40% heteroplasmy — DISTINCTIVE low-threshold phenotype",
            },
            "midm_diabetes_hallmark": {
                "genes": ["MT-TE"],
                "description": "MIDM diabetes dominant; m.14674T>C reversible infantile COX deficiency",
            },
            "hcm_highest_prevalence": {
                "genes": ["MT-TT"],
                "description": "HCM 55–65% — HIGHEST cardiomyopathy rate of all 22 mt-tRNAs; AMIODARONE-ABSOLUTE-CI",
            },
        },
        "large_deletion_patterns": {
            "kss_common_4977bp": {
                "deletion": "rCRS 8470–13447 (4977 bp)",
                "affected_trna_genes": KSS_4977BP_GENES,
                "note": KSS_DELETION_REMARK,
            },
            "ltrna_cluster_deletions": {
                "cluster": "MT-TA/TN/TC/TY (rCRS 5587–5891)",
                "note": "Deletions spanning this 304 bp cluster cause compound multi-tRNA loss (Ala+Asn+Cys+Tyr)",
            },
        },
        "nuclear_ddx_synthetases": [
            {"gene": g["gene"], "ddx_nuclear": g["ddx_nuclear"]} for g in MT_TRNA_GENES
        ],
    }


def get_definitions() -> dict:
    return {
        "gene_definitions": [
            {
                "term": "Mitochondrial tRNA (mt-tRNA)",
                "definition": "Transfer RNA encoded by the mitochondrial genome; 22 mt-tRNA genes decode all 64 codons of the mt-genetic-code (reduced codon set via wobble); essential for translation of 13 mt-encoded OXPHOS subunits",
            },
            {
                "term": "H-strand (Heavy strand)",
                "definition": "Guanine-rich strand of mtDNA; encodes 14 of 22 mt-tRNA genes plus all 13 OXPHOS subunits, 2 rRNAs; standard NGS pipeline covers H-strand genes without special QC",
            },
            {
                "term": "L-strand (Light strand)",
                "definition": "Cytosine-rich complementary strand; encodes 8 of 22 mt-tRNA genes (MT-TQ/TA/TN/TC/TY/TS1/TE/TP) plus MT-ND6 protein; requires reverse-complement NGS QC window — standard pipelines may miss L-strand variants",
            },
            {
                "term": "Combined CI+CIV deficiency (mt-translation fingerprint)",
                "definition": "Biochemical OXPHOS pattern produced by most mt-tRNA mutations: Complex I (CI) and Complex IV (CIV) deficient; Complex II (CII/SDH) normal (nuclear-encoded only); indicates mt-translation failure affecting CI and CIV subunits simultaneously",
            },
            {
                "term": "Heteroplasmy",
                "definition": "Coexistence of mutant and wild-type mtDNA molecules within a cell or tissue; threshold for phenotype expression varies per gene (typically 50–80%); blood underestimates muscle heteroplasmy by 10–20%",
            },
            {
                "term": "L-strand NGS Pitfall",
                "definition": "L-strand mt-tRNA genes are read on the antisense strand; standard BWA/GATK pipelines may fail to call variants without a dedicated reverse-complement QC window; affects 8 of 22 mt-tRNA genes",
            },
            {
                "term": "Common 4977 bp deletion (KSS deletion)",
                "definition": "Most prevalent large-scale mtDNA deletion (rCRS 8470–13447); removes MT-TR, MT-TH, MT-TS2, MT-TL2 plus protein-coding genes; causes KSS (Kearns-Sayre Syndrome) — CPEO + pigmentary retinopathy + cardiac conduction defect + CSF protein >100 mg/dL",
            },
            {
                "term": "MELAS (MT-TL1 hallmark)",
                "definition": "Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke-like Episodes; m.3243A>G most common; stroke-like episodes cross vascular territories (NOT arterial ischaemic stroke); CI+CIII+CIV pan-OXPHOS deficiency",
            },
            {
                "term": "MERRF (MT-TK hallmark)",
                "definition": "Myoclonic Epilepsy with Ragged-Red Fibres; m.8344A>G most common; tau-m5U34 tRNA modification loss; pan-OXPHOS CI+CIV+CV deficiency; Multiple Symmetrical Lipomatosis (MSL) DISTINCTIVE",
            },
            {
                "term": "RRF (Ragged-Red Fibres)",
                "definition": "Mitochondrial proliferation seen on modified Gomori trichrome stain of muscle biopsy; hallmark of mt-tRNA disease at high heteroplasmy; COX-negative (blue on COX/SDH dual stain) when CIV severely affected",
            },
            {
                "term": "CPEO (Chronic Progressive External Ophthalmoplegia)",
                "definition": "Most common adult-onset phenotype of mt-tRNA disease: progressive bilateral ptosis + ophthalmoplegia; typically heteroplasmy >50–65% blood; combined CI+CIV deficiency; distinguished from nuclear CPEO (POLG/PEO1/SLC25A4) by maternal inheritance and muscle heteroplasmy",
            },
        ],
        "drug_definitions": [
            {"term": "Metformin",           "definition": "ABSOLUTE CI (ALL 22 mt-tRNA genes) — biguanide Complex I inhibitor; fatal lactic acidosis in mt-disease even at standard doses"},
            {"term": "VPA (Valproate)",     "definition": "ABSOLUTE CI (ALL 22 mt-tRNA genes) — inhibits β-oxidation and CI; use LEV instead for all mt-tRNA epilepsy"},
            {"term": "Propofol (PRIS)",     "definition": "ABSOLUTE CI (ALL 22 mt-tRNA genes) — propofol infusion syndrome: decouples OXPHOS; fatal myocardial failure; use ketamine or volatile agents"},
            {"term": "Linezolid",           "definition": "ABSOLUTE CI (ALL 22 mt-tRNA genes) — inhibits mt-23S-rRNA protein synthesis; reversible but severe mt-disease exacerbation"},
            {"term": "Chloramphenicol",     "definition": "ABSOLUTE CI (ALL 22 mt-tRNA genes) — inhibits mt-ribosome peptidyl transferase; severe mt-translation failure"},
            {"term": "Ketogenic Diet (KD)", "definition": "CONTRAINDICATED (ALL 22 mt-tRNA genes) — impairs mt-fatty acid oxidation and anaplerosis in mt-translation-deficient cells"},
            {"term": "Amiodarone",          "definition": "ABSOLUTE CI (MT-TT specifically) — mt-OXPHOS inhibitor; use beta-blocker for MT-TT cardiac disease"},
            {"term": "LEV (Levetiracetam)", "definition": "PREFERRED AED (ALL 22 mt-tRNA genes) — renal excretion; no CYP450 induction; no mitochondrial toxicity; Level C evidence"},
            {"term": "Thiamine (B1)",       "definition": "MANDATORY EMPIRIC (ALL 22 mt-tRNA genes) — cofactor for PDH and αKGDH; empiric treatment pending definitive diagnosis"},
            {"term": "Biotin",              "definition": "MANDATORY EMPIRIC (ALL 22 mt-tRNA genes) — BTBGD/SLC19A3 must be excluded; empiric biotin-thiamine until SLC19A3 ruled out"},
        ],
        "references": [
            {"ref": "DiMauro & Schon 2003 NEJM",        "citation": "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases — comprehensive mt-tRNA mutation review; NEJM 348:2656–2668"},
            {"ref": "Schaefer 2008 Ann Neurol",          "citation": "Schaefer AM et al. (2008) — Prevalence of mitochondrial disease in adults in the north east of England; Ann Neurol 63:35–43"},
            {"ref": "Gorman 2016 Nat Rev Dis Primers",   "citation": "Gorman GS et al. (2016) — Mitochondrial diseases — comprehensive review; Nat Rev Dis Primers 2:16080"},
            {"ref": "Boczonadi & Horvath 2014 IJBCB",    "citation": "Boczonadi V, Horvath R (2014) — Mitochondria: impaired mitochondrial translation causes neurological disorders; Int J Biochem Cell Biol 48:77–84"},
            {"ref": "Goto et al. 1990 Nature (MELAS)",  "citation": "Goto Y et al. (1990) — A mutation in the tRNA(Leu)(UUR) gene associated with the MELAS subgroup of mitochondrial encephalomyopathies; Nature 348:651–653"},
            {"ref": "Shoffner et al. 1990 Cell (MERRF)", "citation": "Shoffner JM et al. (1990) — Myoclonic epilepsy and ragged-red fiber disease (MERRF) is associated with a mitochondrial DNA tRNA(Lys) mutation; Cell 61:931–937"},
            {"ref": "Schon 1989 Science (4977bp del)",   "citation": "Schon EA et al. (1989) — A direct repeat is a hotspot for large-scale deletions of human mitochondrial DNA — common 4977 bp deletion; Science 244:346–349"},
            {"ref": "El-Hattab 2015 Biochim Biophys",   "citation": "El-Hattab AW, Scaglia F (2015) — Mitochondrial Cardiomyopathies; Front Cardiovasc Med 3:25 — comprehensive mt-tRNA cardiac phenotype review"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== MT-tRNA ATLAS OVERVIEW ===")
    ov = get_overview()
    print(json.dumps(ov, indent=2)[:3000])
    print("\n=== BREAKDOWN — first 3 genes ===")
    bd = get_breakdown()
    print(json.dumps(bd["gene_table"][:3], indent=2))
    print(f"\nTotal genes in table: {len(bd['gene_table'])}")
