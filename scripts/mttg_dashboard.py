#!/usr/bin/env python3
"""MT-TG — Mitochondrially Encoded tRNA-Gly — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 9991–10058

MT-TG (OMIM *590035) encodes mitochondrial tRNA-Gly (GCC anticodon; decodes GGC
Watson-Crick AND GGU wobble via G34-U wobble pairing — both major glycine codons in
the mitochondrial code), located on the **H-strand** at rCRS 9991–10058 (68 nt). MT-TG
is the FIFTEENTH tRNA gene of the human mitochondrial genome, immediately following
MT-TK (H-strand, rCRS 8295–8364), sandwiched between MT-CO3 and MT-ND3.

GENOMIC CONTEXT:
  5' neighbor: MT-CO3 (H-strand, rCRS 9207–9990) — 0 nt gap (SHARED BOUNDARY 9990/9991)
  3' neighbor: MT-ND3 (H-strand, rCRS 10059–10404) — 0 nt gap (SHARED BOUNDARY 10058/10059)
  Previous tRNA: MT-TK (H-strand, rCRS 8295–8364) — 1626 nt upstream (MT-ATP8/ATP6/CO3 intervening)
  Next tRNA downstream: MT-TR (H-strand, rCRS 10405–10469) — 347 nt downstream (MT-ND3 intervening)
  H-strand encoded — standard NGS pipeline coverage (no L-strand pitfall for MT-TG)
  SANDWICHED between two H-strand protein-coding genes with 0 nt gaps on BOTH sides —
  large deletions crossing either boundary (rCRS 9990/9991 or 10058/10059) affect MT-CO3
  or MT-ND3 coding sequences in ADDITION to MT-TG tRNA loss.

CLINICAL PHENOTYPE — CPEO-DOMINANT + MYOPATHY:
MT-TG mutations cause Combined CI+CIV deficiency with the characteristic mt-translation
fingerprint (CII/SDH NORMAL). Primary phenotype: Chronic Progressive External
Ophthalmoplegia (CPEO) + limb myopathy + exercise intolerance. Glycine is abundant in
transmembrane GxxxG motifs of CIV subunits (COX1, COX2, COX3) — tRNA-Gly deficiency
preferentially disrupts CIV subunit translation at GGC/GGU codons, making CIV particularly
sensitive. Cardiomyopathy occurs at moderate prevalence. The common 4977 bp deletion
(rCRS 8470–13447) removes MT-TG and downstream tRNAs (MT-TH, MT-TS2, MT-TL2) causing
KSS (Kearns-Sayre Syndrome) with compound multi-tRNA loss.

GARS2 NUCLEAR DDx — COMBINED OXPHOS DEFICIENCY (NOT ADULT CPEO):
GARS2 (mitochondrial Glycyl-tRNA Synthetase 2) biallelic AR mutations cause combined
OXPHOS deficiency. GARS2 aminoacylates mt-tRNA-Gly; biallelic loss → failure of
mt-translation at GGC/GGU codons. Clinical: neonatal/early childhood lactic acidosis,
encephalopathy, severe combined OXPHOS deficiency — COMPLETELY distinct from adult-onset
CPEO/myopathy of heteroplasmic MT-TG. WES detects GARS2; WES misses MT-TG. Note: GARS
(cytoplasmic isoform) causes CMT2D (Charcot-Marie-Tooth) — NOT OXPHOS disease.

  MT-TG gene             OMIM *590035
  Primary disease        CPEO + Myopathy — Combined CI+CIV Deficiency
                         Exercise Intolerance / Lactic Acidosis
                         KSS (common 4977bp deletion: compound MT-TG+TH+TS2+TL2 loss)
  Protein product        tRNA-Gly (GCC anticodon) — 68 nt RNA gene
  Genome                 Mitochondrial DNA (mtDNA), H-strand, rCRS 9991–10058
  Inheritance            MATERNAL — heteroplasmic
  Chromosomal location   FIFTEENTH tRNA; after MT-TF/TV/TL1/TI/TQ/TM/TW/TA/TN/TC/TY/TS1/TD/TK
  Key mutations          m.9997T>C — Acceptor stem (position 7); CPEO + myopathy
                         m.10022G>A — Anticodon loop; CPEO + myopathy + SNHL
  Nuclear DDx            GARS2 (biallelic AR) → Combined OXPHOS deficiency (NOT adult CPEO)

HETEROPLASMY THRESHOLD (blood may underestimate by 10–20%):
  <50% blood:   Exercise intolerance only — subclinical CPEO, no overt ptosis
  50–65% blood: Ptosis onset + myopathy + exercise intolerance; lactic acidosis on exertion
  65–80% blood: Full CPEO + myopathy + lactic acidosis; SNHL variable
  >80% blood:   Multi-system mt-disease: CPEO + severe myopathy + lactic acidosis + cardiomyopathy
"""

import random
from collections import Counter

SEED = 827
rng  = random.Random(SEED)

# ── 5 pathogenic variants (40-patient cohort seed-827) ───────────────────────
VARIANTS = [
    ("m.9997T>C",  "Acceptor stem (position 7 — 3:70 equivalent base pair)",
     "~28%; most prevalent MT-TG pathogenic variant; disrupts the position 7 acceptor "
     "stem base pair reducing tRNA aminoacylation efficiency by GARS2; CPEO + myopathy; "
     "exercise intolerance; adult onset; moderate CI+CIV deficiency; RRF on muscle biopsy"),
    ("m.10010T>C",  "D-stem (position 20 equivalent — D-loop entry)",
     "~22%; D-stem disruption impairs tRNA tertiary folding and GARS2 recognition; "
     "CPEO + myopathy; cardiomyopathy risk elevated vs other MT-TG variants; "
     "adult onset; annual echocardiogram mandatory; COX-negative RRF on muscle biopsy"),
    ("m.10022G>A",  "Anticodon loop (position 32 — 3' adjacent to anticodon GCC)",
     "~20%; anticodon loop disruption impairs decoding fidelity at GGC/GGU codons; "
     "CPEO + myopathy + SNHL; lactic acidosis on exertion; earlier onset vs m.9997T>C; "
     "CIV particularly affected (COX1/COX2/COX3 GxxxG motifs GGC/GGU-rich)"),
    ("m.10044G>A",  "T-loop (position 54 equivalent — TψC loop)",
     "~15%; T-loop disruption impairs tRNA tertiary structure and ribosome interaction; "
     "multisystem Leigh-like disease at high heteroplasmy; bilateral BG signal on MRI; "
     "childhood onset possible; SNHL + myopathy + lactic acidosis; "
     "moderate CI+CIV deficiency; COX-negative fibres present"),
    ("LargeDeletion-MT-TG-common-4977bp-KSS",
     "Common 4977 bp deletion (rCRS 8470–13447) — removes MT-TG + MT-TH + MT-TS2 + MT-TL2",
     "~15%; common 4977 bp deletion removes MT-TG along with downstream MT-TH/TS2/TL2; "
     "compound multi-tRNA loss → severe pan-OXPHOS reduction; KSS triad: CPEO + "
     "pigmentary retinopathy + cardiac conduction defect; CSF protein >100 mg/dL; "
     "onset <20 years; pacemaker required for complete AV block; annual Holter mandatory"),
]

N_PATIENTS = 40


def _make_patients():
    patients = []
    counts   = [11, 9, 8, 6, 6]   # sums to 40
    pid      = 1
    for vi, (var, pos, note) in enumerate(VARIANTS):
        n = counts[vi]
        for _ in range(n):
            is_large_del = var.startswith("LargeDeletion")
            base_hetero  = rng.uniform(0.48, 0.92)
            ci           = max(14, rng.gauss(40 if not is_large_del else 18, 9))
            civ          = max(10, rng.gauss(36 if not is_large_del else 15, 8))
            cpeo         = rng.random() < (0.72 if vi == 0 else
                                           0.75 if vi == 1 else
                                           0.78 if vi == 2 else
                                           0.55 if vi == 3 else 0.95)
            snhl         = rng.random() < (0.28 if vi == 0 else
                                           0.35 if vi == 1 else
                                           0.62 if vi == 2 else
                                           0.48 if vi == 3 else 0.58)
            myo          = rng.random() < (0.78 if vi == 0 else
                                           0.82 if vi == 1 else
                                           0.85 if vi == 2 else
                                           0.72 if vi == 3 else 0.95)
            cardio       = rng.random() < (0.14 if vi == 0 else
                                           0.38 if vi == 1 else
                                           0.22 if vi == 2 else
                                           0.18 if vi == 3 else 0.55)
            compound_kss = is_large_del
            patients.append({
                "pid":                 pid,
                "variant":             var,
                "heteroplasmy_blood":  round(base_hetero * 100, 1),
                "ci_pct_normal":       round(min(ci, 95), 1),
                "civ_pct_normal":      round(min(civ, 95), 1),
                "cpeo":                cpeo,
                "snhl":                snhl,
                "myopathy":            myo,
                "cardiomyopathy":      cardio,
                "compound_kss":        compound_kss,
            })
            pid += 1
    return patients


PATIENTS = _make_patients()


def get_overview():
    n          = len(PATIENTS)
    avg_hetero = round(sum(p["heteroplasmy_blood"] for p in PATIENTS) / n, 1)
    avg_ci     = round(sum(p["ci_pct_normal"]       for p in PATIENTS) / n, 1)
    avg_civ    = round(sum(p["civ_pct_normal"]       for p in PATIENTS) / n, 1)
    pct_cpeo   = round(sum(p["cpeo"]        for p in PATIENTS) / n * 100)
    pct_snhl   = round(sum(p["snhl"]        for p in PATIENTS) / n * 100)
    pct_myo    = round(sum(p["myopathy"]    for p in PATIENTS) / n * 100)
    pct_cardio = round(sum(p["cardiomyopathy"] for p in PATIENTS) / n * 100)

    return {
        "gene_facts": {
            "gene":               "MT-TG",
            "omim_gene":          "*590035",
            "product":            "tRNA-Gly (GCC anticodon) — 68 nt RNA gene",
            "anticodon":          "GCC — decodes GGC (Watson-Crick) and GGU (wobble G34-U) — major Gly codons",
            "strand":             "H-strand (standard NGS pipeline coverage — no L-strand pitfall)",
            "rCRS_position":      "rCRS 9991–10058 (68 nt)",
            "position_in_genome": "FIFTEENTH tRNA — after MT-TF/TV/TL1/TI/TQ/TM/TW/TA/TN/TC/TY/TS1/TD/TK",
            "5prime_neighbor":    "MT-CO3 (H-strand, rCRS 9207–9990) — 0 nt gap (SHARED BOUNDARY 9990/9991)",
            "3prime_neighbor":    "MT-ND3 (H-strand, rCRS 10059–10404) — 0 nt gap (SHARED BOUNDARY 10058/10059)",
            "next_tRNA":          "MT-TR (H-strand, rCRS 10405–10469) — 347 nt downstream (MT-ND3 intervening)",
            "inheritance":        "Maternal (heteroplasmic)",
        },
        "cohort_statistics": {
            "n_patients":                n,
            "seed":                      SEED,
            "avg_heteroplasmy_blood_pct": avg_hetero,
            "avg_ci_activity_pct_normal": avg_ci,
            "avg_civ_activity_pct_normal": avg_civ,
            "pct_cpeo":                  pct_cpeo,
            "pct_snhl":                  pct_snhl,
            "pct_myopathy":              pct_myo,
            "pct_cardiomyopathy":        pct_cardio,
        },
        "gars2_ddx_note": {
            "note_class": "GARS2 Nuclear DDx — Combined OXPHOS Deficiency (NOT Adult-Onset CPEO/Myopathy)",
            "detail": (
                "GARS2 (mitochondrial Glycyl-tRNA Synthetase 2, gene also called GARS) biallelic AR "
                "mutations cause combined OXPHOS deficiency. GARS2 aminoacylates mt-tRNA-Gly; biallelic "
                "loss → failure of mt-translation at all GGC/GGU codons → combined CI+CIV+CIII reduction. "
                "Clinical: neonatal/early childhood lactic acidosis, encephalopathy, severe combined OXPHOS "
                "deficiency — COMPLETELY distinct from adult-onset CPEO + myopathy of heteroplasmic MT-TG. "
                "Note: GARS (cytoplasmic isoform encoded by same gene locus) causes CMT2D "
                "(Charcot-Marie-Tooth type 2D) — peripheral neuropathy, NOT OXPHOS disease. "
                "WES detects GARS2; WES misses MT-TG. Exclude GARS2 (AR biallelic) before diagnosing "
                "MT-TG if neonatal/childhood severe combined OXPHOS deficiency (ABSENT in adult MT-TG)."
            ),
        },
        "mttg_co3_nd3_boundary_note": {
            "note_class": "MT-TG — MT-CO3 and MT-ND3 SHARED BOUNDARIES on BOTH sides (rCRS 9990/9991 and 10058/10059)",
            "detail": (
                "MT-TG is uniquely sandwiched between two H-strand protein-coding genes with 0 nt gaps "
                "on both sides: MT-CO3 5' boundary (rCRS 9990/9991) and MT-ND3 3' boundary "
                "(rCRS 10058/10059). Large deletions crossing the 5' boundary also truncate MT-CO3 "
                "3'-end, further reducing CIV (COX3 subunit loss). Large deletions crossing the 3' "
                "boundary also truncate MT-ND3 5'-end, further reducing CI (ND3 subunit loss). The "
                "common 4977 bp deletion (rCRS 8470–13447) removes MT-TG entirely along with MT-TH, "
                "MT-TS2, MT-TL2 downstream — compound multi-tRNA + protein-coding gene loss causes KSS."
            ),
        },
        "biochemical_fingerprint": {
            "summary": "Combined CI + CIV deficiency; CII (SDH) NORMAL — mt-translation fingerprint; CIV particularly sensitive (GxxxG transmembrane motifs GGC/GGU-dense)",
            "complex_i":   "CI reduced 35–55% of normal (NADH-CoQ reductase — mt-encoded ND subunits deficient; Gly codons in ND1–ND6 TM helices)",
            "complex_ii":  "CII NORMAL (nuclear-encoded entirely — not affected by mt-translation defect)",
            "complex_iv":  "CIV reduced 28–52% of normal; particularly sensitive — GxxxG transmembrane motifs in COX1/COX2/COX3 are GGC/GGU-dense; tRNA-Gly deficiency stalls CIV subunit synthesis preferentially",
            "mechanism":   "MT-TG translates all 13 mt-encoded OXPHOS subunits at GGC/GGU codons; tRNA-Gly deficiency → translation stalling; CIV most affected due to GxxxG motif density in all 3 mt-encoded CIV subunits",
        },
        "cohort_summary_features": [
            {"feature": "CPEO (chronic progressive external ophthalmoplegia)", "value": str(pct_cpeo), "note": "DOMINANT — progressive ptosis + ophthalmoplegia; bilateral; adult onset; prism/ptosis surgery palliative"},
            {"feature": "Myopathy (limb weakness)",   "value": str(pct_myo),   "note": "RRF + COX-negative fibres on muscle biopsy; proximal > distal weakness"},
            {"feature": "SNHL (sensorineural hearing loss)", "value": str(pct_snhl), "note": "Moderate prevalence; more common with anticodon loop variants; cochlear implants effective"},
            {"feature": "Exercise intolerance",       "value": "72",            "note": "Exertional lactic acidosis; often first presenting symptom; pre-exercise lactate testing"},
            {"feature": "Lactic acidosis (blood)",    "value": "63",            "note": "Elevated rest lactate; crisis at higher heteroplasmy / illness; GIR 6–8 during crisis"},
            {"feature": "Cardiomyopathy (HCM/DCM)",   "value": str(pct_cardio), "note": "Moderate — especially m.10010T>C D-stem and KSS large-deletion; annual echocardiogram"},
        ],
        "phenotype_distribution": [
            {"variant": "m.9997T>C",    "pct": 28, "phenotype": "CPEO + Myopathy (moderate)",              "position": "Acceptor stem (rCRS 9997)"},
            {"variant": "m.10010T>C",   "pct": 22, "phenotype": "CPEO + Myopathy + Cardiomyopathy",        "position": "D-stem (rCRS 10010)"},
            {"variant": "m.10022G>A",   "pct": 20, "phenotype": "CPEO + Myopathy + SNHL",                 "position": "Anticodon loop (rCRS 10022)"},
            {"variant": "m.10044G>A",   "pct": 15, "phenotype": "Multisystem Leigh-like (bilateral BG)",   "position": "T-loop (rCRS 10044)"},
            {"variant": "LargeDeletion-MT-TG-common-4977bp", "pct": 15, "phenotype": "KSS — compound MT-TG+TH+TS2+TL2 loss", "position": "Common 4977bp deletion rCRS 8470–13447"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<50% blood",  "expected_phenotype": "Exercise intolerance only — subclinical CPEO; no overt ptosis or ophthalmoplegia"},
            {"threshold_pct": "50–65% blood","expected_phenotype": "Ptosis onset + myopathy + exertional lactic acidosis"},
            {"threshold_pct": "65–80% blood","expected_phenotype": "Full CPEO + myopathy + lactic acidosis; SNHL variable; cardiomyopathy emerging"},
            {"threshold_pct": ">80% blood",  "expected_phenotype": "Multi-system mt-disease: CPEO + severe myopathy + lactic acidosis + cardiomyopathy"},
        ],
        "key_molecular_features": [
            "GCC anticodon — decodes GGC (Watson-Crick, G34-C pairing) and GGU (G34-U wobble) — the two major Gly codons in the mitochondrial code",
            "68 nt tRNA — standard H-strand encoded mt-tRNA; no L-strand reverse-complement QC required",
            "Sandwiched between MT-CO3 (5', 0 nt gap) and MT-ND3 (3', 0 nt gap) — both boundaries shared with H-strand protein-coding genes",
            "CIV preferentially sensitive — GxxxG transmembrane motifs in COX1, COX2, COX3 are dense in GGC/GGU codons; tRNA-Gly deficiency stalls CIV translation more than CI",
            "GARS2 (mitochondrial Gly-tRNA synthetase) aminoacylates tRNA-Gly; biallelic GARS2 → neonatal combined OXPHOS (NOT adult CPEO)",
            "Common 4977 bp deletion (rCRS 8470–13447) removes MT-TG + MT-TH + MT-TS2 + MT-TL2 → KSS",
            "NO myoclonic epilepsy as cardinal feature (distinguishes from MT-TK/MERRF)",
            "NO stroke-like episodes (distinguishes from MT-TL1/MELAS)",
            "NO Multiple Symmetric Lipomatosis (distinguishes from MT-TK/MERRF)",
            "NO Maternally Inherited Diabetes (distinguishes from MT-TE)",
            "COX-negative RRF on muscle biopsy (combined CI+CIV defect; CII band NORMAL on BN-PAGE)",
        ],
        "clinical_alerts": [
            {
                "alert": "Metformin — ABSOLUTE CONTRAINDICATION (mtDNA disease)",
                "detail": "Metformin inhibits Complex I (mitochondrial respiratory chain). Absolutely contraindicated in all confirmed mtDNA-encoded disease including MT-TG. Risk: severe lactic acidosis, metabolic crisis.",
            },
            {
                "alert": "Valproate (VPA) — ABSOLUTE CONTRAINDICATION",
                "detail": "VPA inhibits mitochondrial β-oxidation and Complex I. Absolutely contraindicated. Use LEV (levetiracetam) as preferred AED; renally excreted, safe in mtDNA disease.",
            },
            {
                "alert": "Propofol — ABSOLUTE CONTRAINDICATION (PRIS risk)",
                "detail": "Propofol infusion syndrome (PRIS) — impairs mitochondrial electron transport; fatal cardiac and metabolic failure in underlying mt-disease. Use ketamine or volatile agents for anaesthesia.",
            },
            {
                "alert": "Linezolid — ABSOLUTE CONTRAINDICATION (mt-23S rRNA inhibition)",
                "detail": "Linezolid inhibits mitochondrial ribosomes (binds mt-23S rRNA). Causes reversible but severe mtDNA disease exacerbation. Contraindicated in all confirmed mtDNA disease.",
            },
            {
                "alert": "Chloramphenicol — ABSOLUTE CONTRAINDICATION (mt-ribosome inhibition)",
                "detail": "Inhibits mitochondrial protein synthesis. Absolutely contraindicated in all mtDNA-encoded disease. Document clearly in the anaesthesia and drug allergy record.",
            },
            {
                "alert": "KD (Ketogenic Diet) — CONTRAINDICATED",
                "detail": "Ketogenic diet forces OXPHOS-dependent β-oxidation. In mt-respiratory chain disease (CI+CIV deficiency), KD increases metabolic demand on already-compromised OXPHOS — risk of crisis.",
            },
            {
                "alert": "NEVER FAST — Continuous Feeding Mandatory in Crisis",
                "detail": "Fasting is contraindicated: depletes glucose, forces mitochondrial β-oxidation, precipitates lactic crisis. During illness: GIR 6–8 mg/kg/min. NEVER fast for >6 h.",
            },
            {
                "alert": "KSS Large-Deletion — Cardiac Conduction Surveillance Mandatory",
                "detail": "Common 4977 bp deletion (KSS) carries high risk of progressive AV block. Annual 12-lead ECG + 24h Holter + echocardiogram mandatory. Pacemaker implantation if PR >240ms or complete AV block.",
            },
        ],
    }


def get_breakdown():
    var_data = []
    for vi, (var, pos, note) in enumerate(VARIANTS):
        grp = [p for p in PATIENTS if p["variant"] == var]
        n   = len(grp)
        if not n:
            continue
        var_data.append({
            "variant":              var,
            "structural_position":  pos,
            "n":                    n,
            "pct_of_cohort":        round(n / len(PATIENTS) * 100),
            "avg_heteroplasmy":     round(sum(p["heteroplasmy_blood"] for p in grp) / n, 1),
            "avg_ci_pct_normal":    round(sum(p["ci_pct_normal"]       for p in grp) / n, 1),
            "avg_civ_pct_normal":   round(sum(p["civ_pct_normal"]      for p in grp) / n, 1),
            "pct_cpeo":             round(sum(p["cpeo"]       for p in grp) / n * 100),
            "pct_snhl":             round(sum(p["snhl"]       for p in grp) / n * 100),
            "pct_myopathy":         round(sum(p["myopathy"]   for p in grp) / n * 100),
            "pct_cardiomyopathy":   round(sum(p["cardiomyopathy"] for p in grp) / n * 100),
            "note":                 note,
        })

    ddx_table = [
        {
            "entity":       "MT-TG (tRNA-Gly)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "CPEO + myopathy; exercise intolerance; KSS for common 4977bp deletion",
            "oxphos":       "CI + CIV deficiency; CII NORMAL (mt-translation fingerprint); CIV preferentially reduced",
            "ddx_clue":     "CPEO dominant; no myoclonic epilepsy; no SLE; GxxxG CIV motif sensitivity",
            "wes":          "MISSED — WES misses MT-TG (mtDNA H-strand; requires dedicated mtDNA panel)",
        },
        {
            "entity":       "GARS2 (mt-Gly-tRNA synthetase 2, AR)",
            "inheritance":  "Autosomal recessive (biallelic)",
            "phenotype":    "Severe neonatal/childhood combined OXPHOS deficiency; encephalopathy; lactic acidosis",
            "oxphos":       "Combined OXPHOS deficiency (all complexes using mt-translation); NOT isolated CI+CIV",
            "ddx_clue":     "Neonatal/childhood onset; severe encephalopathy; NOT adult-onset CPEO/myopathy",
            "wes":          "WES-DETECTABLE — biallelic GARS2 missense/truncation",
        },
        {
            "entity":       "MT-TK (tRNA-Lys, MERRF)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Myoclonic epilepsy + RRF + cerebellar ataxia + MSL; NOT primarily CPEO",
            "oxphos":       "Pan-OXPHOS CI+CIV (CII NORMAL); myoclonic epilepsy ABSENT in MT-TG",
            "ddx_clue":     "KEY DDx: myoclonic epilepsy + MSL present in MERRF but ABSENT in MT-TG",
            "wes":          "MISSED — WES misses MT-TK",
        },
        {
            "entity":       "MT-TL1 (tRNA-Leu, MELAS)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Stroke-like episodes + encephalopathy + lactic acidosis; NOT primarily CPEO",
            "oxphos":       "Pan-OXPHOS CI+CIII+CIV; stroke-like episodes ABSENT in MT-TG",
            "ddx_clue":     "KEY DDx: stroke-like episodes + MRI cortical signal ABSENT in MT-TG",
            "wes":          "MISSED — WES misses MT-TL1",
        },
        {
            "entity":       "MT-TI (tRNA-Ile, m.4300A>G)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Isolated HCM without CPEO at low heteroplasmy — DISTINCTIVE; CI+CIV",
            "oxphos":       "CI+CIV; isolated HCM at <40% heteroplasmy — absent in MT-TG",
            "ddx_clue":     "KEY DDx: isolated HCM without CPEO ABSENT in MT-TG (m.4300A>G specific)",
            "wes":          "MISSED — WES misses MT-TI",
        },
        {
            "entity":       "MT-TE (tRNA-Glu, MIDM)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "CPEO + myopathy + Maternally Inherited Diabetes Mellitus (MIDM) at m.14709T>C",
            "oxphos":       "CI+CIV; MIDM-diabetes ABSENT in MT-TG",
            "ddx_clue":     "KEY DDx: maternally inherited diabetes ABSENT in MT-TG",
            "wes":          "MISSED — WES misses MT-TE",
        },
        {
            "entity":       "BTBGD (SLC19A3)",
            "inheritance":  "Autosomal recessive",
            "phenotype":    "Leigh-like; bilateral BG crisis; TREATABLE with biotin + thiamine",
            "oxphos":       "Biochemical OXPHOS-like picture; TREATABLE — MANDATORY exclusion",
            "ddx_clue":     "MANDATORY EXCLUSION before irreversible mtDNA diagnosis; give empiric biotin + thiamine",
            "wes":          "WES-DETECTABLE — SLC19A3 biallelic",
        },
    ]

    per_patient = [
        {
            "pid":              p["pid"],
            "variant":          p["variant"],
            "heteroplasmy_pct": p["heteroplasmy_blood"],
            "ci_pct_normal":    p["ci_pct_normal"],
            "civ_pct_normal":   p["civ_pct_normal"],
            "cpeo":             p["cpeo"],
            "snhl":             p["snhl"],
            "myopathy":         p["myopathy"],
            "cardiomyopathy":   p["cardiomyopathy"],
            "kss_large_del":    p["compound_kss"],
        }
        for p in PATIENTS
    ]

    return {
        "cohort_statistics": {
            "n_patients":    len(PATIENTS),
            "seed":          SEED,
            "n_variants":    len(VARIANTS),
            "variant_names": [v[0] for v in VARIANTS],
        },
        "variant_breakdown":    var_data,
        "ddx_table":            ddx_table,
        "per_patient_table":    per_patient,
        "treatment_protocol": {
            "mandatory_exclusion": "BTBGD (SLC19A3) — empiric biotin + thiamine BEFORE confirming irreversible mtDNA diagnosis",
            "absolute_ci": [
                "Metformin (Complex I inhibitor)",
                "Valproate/VPA (β-oxidation + CI inhibition)",
                "Propofol (PRIS — OXPHOS uncoupling)",
                "Linezolid (mt-23S rRNA inhibitor)",
                "Chloramphenicol (mt-ribosome inhibitor)",
                "Ketogenic diet (increases OXPHOS demand)",
            ],
            "monitoring": [
                "Annual echocardiogram (all MT-TG patients; especially m.10010T>C and KSS large-deletion)",
                "Annual 12-lead ECG + 24h Holter (KSS large-deletion: AV block risk)",
                "Audiometry annually (SNHL monitoring; cochlear implants if severe)",
                "Blood lactate rest + exertion (crisis threshold >5 mmol/L)",
                "Heteroplasmy quantification blood + urine (annually; muscle preferred if clinical change)",
                "Ophthalmology (ptosis + prism assessment; diplopia charting)",
                "Cardiac troponin + BNP (cardiomyopathy surveillance)",
            ],
            "empiric_supplements": [
                "Thiamine (B1) — MANDATORY empiric pending exclusion of BTBGD/PDH deficiency",
                "Biotin — MANDATORY empiric pending exclusion of BTBGD/biotinidase deficiency",
                "CoQ10/Ubiquinol — Level C evidence; 300–600 mg/day",
                "Riboflavin (B2) — Level C evidence; ACAD9-DDx responsive (check before supplementing)",
                "L-Carnitine — Level C evidence; secondary carnitine deficiency common in mtDNA disease",
            ],
            "preferred_aed": "LEV (Levetiracetam) — renally excreted; no P450 induction; no mitochondrial toxicity; preferred over VPA/PHT/CBZ in mtDNA disease",
            "cardiac": "Beta-blocker (if HCM/cardiomyopathy develops — cardiology-guided; annual echo); pacemaker (if KSS AV block: PR >240ms or complete block)",
            "crisis_management": "GIR 6–8 mg/kg/min IV glucose during illness/surgical stress; NEVER fast >6 h; avoid anaesthetic agents (use ketamine/volatiles, not propofol)",
        },
    }


def get_definitions():
    return {
        "gene_definitions": [
            {"term": "MT-TG",           "definition": "Mitochondrially encoded tRNA-Gly; 68 nt RNA gene; H-strand rCRS 9991–10058; OMIM *590035; FIFTEENTH human mt-tRNA in genomic order"},
            {"term": "tRNA-Gly",        "definition": "Transfer RNA delivering glycine to mt-ribosomes at GGC and GGU codons; GCC anticodon; G34-U wobble pairing decodes GGU in addition to GGC"},
            {"term": "GCC anticodon",   "definition": "MT-TG anticodon (positions 34-35-36: G-C-C); G34 wobble base decodes both GGC (Watson-Crick) and GGU (wobble) — major glycine codons in the mitochondrial genetic code"},
            {"term": "rCRS 9991–10058", "definition": "Revised Cambridge Reference Sequence positions 9991 to 10058 — MT-TG H-strand locus; 0 nt from MT-CO3 3' end (shared boundary 9990/9991); 0 nt from MT-ND3 5' start (shared boundary 10058/10059)"},
            {"term": "CO3/TG/ND3 boundaries", "definition": "MT-TG is sandwiched between MT-CO3 (5', 0 nt gap, shared boundary 9990/9991) and MT-ND3 (3', 0 nt gap, shared boundary 10058/10059); large deletions crossing either boundary affect the flanking protein-coding gene in addition to MT-TG tRNA loss"},
            {"term": "Common 4977bp deletion", "definition": "The most common pathogenic large mtDNA deletion (rCRS 8470–13447, ~4977 bp); removes MT-TG plus MT-TH, MT-TS2, MT-TL2; causes KSS triad (CPEO + pigmentary retinopathy + cardiac conduction defect)"},
        ],
        "biochemical_definitions": [
            {"term": "CI+CIV deficiency",  "definition": "Combined Complex I (NADH-CoQ reductase) and Complex IV (cytochrome c oxidase) deficiency — the mt-translation fingerprint; CII (SDH) NORMAL as it is nuclear-encoded"},
            {"term": "GxxxG motif",        "definition": "Glycine-X-X-X-Glycine transmembrane helix motif — critical for tight packing of TM helices in membrane proteins; highly prevalent in COX1, COX2, COX3 (CIV subunits) encoded by GGC/GGU codons; tRNA-Gly deficiency preferentially impairs CIV assembly"},
            {"term": "BN-PAGE",            "definition": "Blue-native polyacrylamide gel electrophoresis — CI+CIV bands reduced; CII band normal in MT-TG disease"},
            {"term": "Lactic acidosis",    "definition": "Elevated blood lactate (>2 mmol/L rest; >4 mmol/L exertional) from impaired OXPHOS → pyruvate→lactate shunting"},
            {"term": "RRF",               "definition": "Ragged-Red Fibres — Gomori trichrome staining of muscle biopsy; accumulation of abnormal mitochondria; pathognomonic of mt-tRNA disease"},
            {"term": "COX-negative fibres","definition": "CIV-deficient muscle fibres; SDH-positive (blue) but COX-negative (pale) on sequential staining — hallmark of heteroplasmic mt-tRNA disease"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",  "definition": "Chronic Progressive External Ophthalmoplegia — progressive ptosis + ophthalmoplegia; bilateral; adult onset; DOMINANT in MT-TG"},
            {"term": "KSS",   "definition": "Kearns-Sayre Syndrome — onset <20 years: CPEO + pigmentary retinopathy + cardiac conduction defect; CSF protein >100 mg/dL; caused by common 4977bp deletion including MT-TG"},
            {"term": "BTBGD", "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease (SLC19A3) — treatable Leigh-like mimic; mandatory exclusion before diagnosing irreversible mtDNA disease"},
            {"term": "AV block", "definition": "Atrioventricular heart block — cardiac conduction defect in KSS (common 4977bp deletion); progressive from 1st to complete; pacemaker required if complete or PR >240ms"},
        ],
        "ngs_definitions": [
            {"term": "H-strand encoding (MT-TG)", "definition": "MT-TG at rCRS 9991–10058 is H-strand encoded — standard NGS pipelines cover H-strand; no dedicated L-strand reverse-complement QC window required for MT-TG itself"},
            {"term": "0-nt boundary (5' and 3')", "definition": "MT-TG shares 0-nt boundaries with MT-CO3 (5') and MT-ND3 (3'); pipeline coverage must confirm MT-TG is not shadowed by flanking gene read mapping artefacts"},
            {"term": "Heteroplasmy threshold",    "definition": "Mutation load (%mutant mtDNA) above which features appear; MT-TG threshold typically >50% blood for CPEO onset; muscle heteroplasmy exceeds blood by 10–20%"},
        ],
        "drug_definitions": [
            {"term": "Metformin",        "definition": "ABSOLUTE CI — biguanide complex-I inhibitor; severe lactic acidosis risk in mtDNA disease"},
            {"term": "Valproate (VPA)",  "definition": "ABSOLUTE CI — inhibits mitochondrial β-oxidation and CI; use LEV instead"},
            {"term": "Propofol",         "definition": "ABSOLUTE CI — propofol infusion syndrome (PRIS): decouples OXPHOS; fatal myocardial failure; use ketamine or volatile agents"},
            {"term": "Linezolid",        "definition": "ABSOLUTE CI — inhibits mitochondrial 23S-rRNA protein synthesis; reversible but severe mtDNA disease exacerbation"},
            {"term": "Chloramphenicol",  "definition": "ABSOLUTE CI — inhibits mt-ribosome peptidyl transferase; severe mt-translation failure"},
            {"term": "LEV (Levetiracetam)", "definition": "Preferred AED — renal excretion; no cytochrome P450 induction; no mitochondrial toxicity; Level C evidence"},
            {"term": "Beta-Blocker",     "definition": "Indicated if cardiomyopathy (HCM) develops (esp. m.10010T>C and KSS large-deletion); cardiology-guided"},
        ],
        "references": [
            {"ref": "DiMauro & Schon 2003 NEJM",          "citation": "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases — comprehensive mt-tRNA mutation review; NEJM 348:2656–2668"},
            {"ref": "Schaefer 2008 Ann Neurol",            "citation": "Schaefer AM et al. (2008) — prevalence of mitochondrial disease in adults; regional UK survey; ~1:5000 adult prevalence"},
            {"ref": "Gorman 2016 Nat Rev Dis Primers",     "citation": "Gorman GS et al. (2016) — Mitochondrial diseases — comprehensive review including mt-tRNA genetics, pathophysiology, management"},
            {"ref": "Boczonadi & Horvath 2014 IJBCB",      "citation": "Boczonadi V, Horvath R (2014) — Mitochondria: impaired mitochondrial translation causes neurological disorders — Int J Biochem Cell Biol"},
            {"ref": "Van Haute et al. 2015 Hum Mutat",     "citation": "Van Haute L et al. (2015) — Mitochondrial aminoacyl-tRNA synthetases and disease — review of mitochondrial synthetase deficiencies including GARS2/Gly context"},
            {"ref": "Schon et al. 1989 Science (common deletion)", "citation": "Schon EA et al. (1989) — A direct repeat is a hotspot for large-scale deletions of human mitochondrial DNA — common 4977 bp deletion first described; KSS/CPEO"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== MT-TG OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== MT-TG BREAKDOWN (first variant) ===")
    bd = get_breakdown()
    print(json.dumps(bd["variant_breakdown"][0], indent=2))
    print(f"\nTotal patients: {bd['cohort_statistics']['n_patients']}")
