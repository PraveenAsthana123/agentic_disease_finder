#!/usr/bin/env python3
"""MT-TR — Mitochondrially Encoded tRNA-Arg — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 10405–10469

MT-TR (OMIM *590005) encodes mitochondrial tRNA-Arg (UCG anticodon; decodes CGA
Watson-Crick AND CGG wobble via U34-G wobble pairing, plus CGC/CGU via vertebrate
mt superwobble — all four CGN arginine codons in the mitochondrial code), located on
the **H-strand** at rCRS 10405–10469 (65 nt). MT-TR is the SIXTEENTH tRNA gene of
the human mitochondrial genome, immediately following MT-TG (H-strand, rCRS 9991–10058),
sandwiched between MT-ND3 and MT-ND4L with 0 nt gaps on BOTH sides.

GENOMIC CONTEXT:
  5' neighbor: MT-ND3 (H-strand, rCRS 10059–10404) — 0 nt gap (SHARED BOUNDARY 10404/10405)
  3' neighbor: MT-ND4L (H-strand, rCRS 10470–10766) — 0 nt gap (SHARED BOUNDARY 10469/10470)
  Previous tRNA: MT-TG (H-strand, rCRS 9991–10058) — 347 nt upstream (MT-ND3 intervening)
  Next tRNA downstream: MT-TH (H-strand, rCRS 12138–12206) — 1669 nt downstream (MT-ND4L/ND4 intervening)
  H-strand encoded — standard NGS pipeline coverage (no L-strand pitfall for MT-TR)
  SANDWICHED between two H-strand protein-coding genes with 0 nt gaps on BOTH sides —
  large deletions crossing either boundary (rCRS 10404/10405 or 10469/10470) affect MT-ND3
  3'-end or MT-ND4L 5'-start in ADDITION to MT-TR tRNA loss.

CLINICAL PHENOTYPE — CPEO-DOMINANT + MYOPATHY:
MT-TR mutations cause Combined CI+CIV deficiency with the characteristic mt-translation
fingerprint (CII/SDH NORMAL). Primary phenotype: Chronic Progressive External
Ophthalmoplegia (CPEO) + limb myopathy + exercise intolerance. Arginine (CGN codons)
is enriched in proton-translocating helices of CI (ND subunits) and CIV (COX subunits)
and at the catalytic Arg-finger motifs — tRNA-Arg deficiency disrupts CI and CIV
subunit translation across all 13 mt-encoded OXPHOS subunits at CGN codons.
Cardiomyopathy occurs at moderate prevalence. The common 4977 bp deletion
(rCRS 8470–13447) removes MT-TR and downstream tRNAs (MT-TH, MT-TS2, MT-TL2)
causing KSS (Kearns-Sayre Syndrome) with compound multi-tRNA loss.

RARS2 NUCLEAR DDx — PONTOCEREBELLAR HYPOPLASIA TYPE 6 (PCH6):
RARS2 (mitochondrial Arginyl-tRNA Synthetase 2) biallelic AR mutations cause
Pontocerebellar Hypoplasia type 6 (PCH6). RARS2 aminoacylates mt-tRNA-Arg; biallelic
loss → failure of mt-translation at all CGN codons → combined OXPHOS deficiency.
Clinical: neonatal/infantile-onset encephalopathy, pontocerebellar atrophy on MRI,
severe combined OXPHOS deficiency — COMPLETELY distinct from adult-onset CPEO/myopathy
of heteroplasmic MT-TR. WES detects RARS2; WES misses MT-TR. Note: RARS (cytoplasmic
isoform) → different aminoacyl-tRNA synthetase disease — NOT mt-OXPHOS disease.

  MT-TR gene             OMIM *590005
  Primary disease        CPEO + Myopathy — Combined CI+CIV Deficiency
                         Exercise Intolerance / Lactic Acidosis
                         KSS (common 4977bp deletion: compound MT-TR+TH+TS2+TL2 loss)
  Protein product        tRNA-Arg (UCG anticodon) — 65 nt RNA gene
  Genome                 Mitochondrial DNA (mtDNA), H-strand, rCRS 10405–10469
  Inheritance            MATERNAL — heteroplasmic
  Chromosomal location   SIXTEENTH tRNA; after MT-TF/TV/TL1/TI/TQ/TM/TW/TA/TN/TC/TY/TS1/TD/TK/TG
  Key mutations          m.10450A>G — T-stem (position 54 equivalent); CPEO + myopathy
                         m.10438A>G — Anticodon stem; CPEO + cardiomyopathy
  Nuclear DDx            RARS2 (biallelic AR) → PCH6 Pontocerebellar Hypoplasia (NOT adult CPEO)

HETEROPLASMY THRESHOLD (blood may underestimate by 10–20%):
  <50% blood:   Exercise intolerance only — subclinical CPEO, no overt ptosis
  50–65% blood: Ptosis onset + myopathy + exercise intolerance; lactic acidosis on exertion
  65–80% blood: Full CPEO + myopathy + lactic acidosis; SNHL variable
  >80% blood:   Multi-system mt-disease: CPEO + severe myopathy + lactic acidosis + cardiomyopathy
"""

import random

SEED = 829
rng  = random.Random(SEED)

# ── 5 pathogenic variants (40-patient cohort seed-829) ───────────────────────
VARIANTS = [
    ("m.10450A>G",  "T-stem (position 54 equivalent — T-loop entry)",
     "~28%; most prevalent MT-TR pathogenic variant; T-stem disruption reduces tRNA "
     "tertiary folding stability and ribosome interaction efficiency; CPEO + myopathy; "
     "exercise intolerance; adult onset; moderate CI+CIV deficiency; RRF on muscle biopsy; "
     "first reported by Santorelli et al. in CPEO+myopathy context"),
    ("m.10438A>G",  "Anticodon stem (position 31–39 stem — anticodon loop entry)",
     "~22%; anticodon stem disruption impairs tRNA decoding geometry at CGN codons; "
     "CPEO + myopathy; cardiomyopathy risk elevated vs other MT-TR variants; "
     "adult onset; annual echocardiogram mandatory; COX-negative RRF on muscle biopsy; "
     "combined CI+CIV deficiency on muscle respiratory chain enzymology"),
    ("m.10410T>C",  "D-stem (position 13 equivalent — D-loop stability)",
     "~20%; D-stem disruption impairs tRNA tertiary folding and RARS2 recognition; "
     "exercise intolerance + myopathy + SNHL; earlier onset possible vs m.10450A>G; "
     "CIV particularly affected (CGN-rich proton-translocating helices in COX1-3); "
     "lactic acidosis on exertion; moderate CI+CIV deficiency"),
    ("m.10460G>A",  "T-loop (position 59 equivalent — TψC loop)",
     "~15%; T-loop disruption impairs tRNA tertiary structure and ribosome interaction; "
     "multisystem Leigh-like disease at high heteroplasmy; bilateral BG signal on MRI; "
     "childhood onset possible; SNHL + myopathy + lactic acidosis; "
     "moderate CI+CIV deficiency; COX-negative fibres present"),
    ("LargeDeletion-MT-TR-common-4977bp-KSS",
     "Common 4977 bp deletion (rCRS 8470–13447) — removes MT-TR + MT-TH + MT-TS2 + MT-TL2",
     "~15%; common 4977 bp deletion removes MT-TR along with downstream MT-TH/TS2/TL2; "
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
                                           0.76 if vi == 1 else
                                           0.70 if vi == 2 else
                                           0.55 if vi == 3 else 0.95)
            snhl         = rng.random() < (0.28 if vi == 0 else
                                           0.32 if vi == 1 else
                                           0.55 if vi == 2 else
                                           0.50 if vi == 3 else 0.52)
            myo          = rng.random() < (0.80 if vi == 0 else
                                           0.82 if vi == 1 else
                                           0.75 if vi == 2 else
                                           0.60 if vi == 3 else 0.92)
            cardio       = rng.random() < (0.18 if vi == 0 else
                                           0.42 if vi == 1 else
                                           0.22 if vi == 2 else
                                           0.28 if vi == 3 else 0.75)
            compound_kss = is_large_del
            patients.append({
                "pid":               pid,
                "variant":           var,
                "heteroplasmy_blood": round(base_hetero * 100, 1),
                "ci_pct_normal":     round(max(14, ci), 1),
                "civ_pct_normal":    round(max(10, civ), 1),
                "cpeo":              cpeo,
                "snhl":              snhl,
                "myopathy":          myo,
                "cardiomyopathy":    cardio,
                "compound_kss":      compound_kss,
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
            "gene":               "MT-TR",
            "omim_gene":          "*590005",
            "product":            "tRNA-Arg (UCG anticodon) — 65 nt RNA gene",
            "anticodon":          "UCG — decodes CGA (Watson-Crick) and CGG (wobble U34-G); superwobble extends to CGC/CGU — all CGN Arg codons",
            "strand":             "H-strand (standard NGS pipeline coverage — no L-strand pitfall)",
            "rCRS_position":      "rCRS 10405–10469 (65 nt)",
            "position_in_genome": "SIXTEENTH tRNA — after MT-TF/TV/TL1/TI/TQ/TM/TW/TA/TN/TC/TY/TS1/TD/TK/TG",
            "5prime_neighbor":    "MT-ND3 (H-strand, rCRS 10059–10404) — 0 nt gap (SHARED BOUNDARY 10404/10405)",
            "3prime_neighbor":    "MT-ND4L (H-strand, rCRS 10470–10766) — 0 nt gap (SHARED BOUNDARY 10469/10470)",
            "next_tRNA":          "MT-TH (H-strand, rCRS 12138–12206) — 1669 nt downstream (MT-ND4L/ND4 intervening)",
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
        "rars2_ddx_note": {
            "note_class": "RARS2 Nuclear DDx — Pontocerebellar Hypoplasia Type 6 / PCH6 (NOT Adult-Onset CPEO/Myopathy)",
            "detail": (
                "RARS2 (mitochondrial Arginyl-tRNA Synthetase 2) biallelic AR mutations cause "
                "Pontocerebellar Hypoplasia type 6 (PCH6). RARS2 aminoacylates mt-tRNA-Arg; "
                "biallelic loss → failure of mt-translation at all CGN codons → combined OXPHOS "
                "deficiency. Clinical: neonatal/infantile-onset progressive encephalopathy, "
                "pontocerebellar atrophy on MRI, severe combined OXPHOS deficiency — "
                "COMPLETELY distinct from adult-onset CPEO + myopathy of heteroplasmic MT-TR. "
                "Note: RARS (cytoplasmic isoform) → different aminoacyl-tRNA synthetase disease "
                "— NOT mt-OXPHOS. WES detects RARS2 (biallelic); WES misses MT-TR. Exclude RARS2 "
                "(AR biallelic) before diagnosing MT-TR if neonatal/infantile severe combined OXPHOS "
                "deficiency with pontocerebellar atrophy (ABSENT in adult MT-TR CPEO)."
            ),
        },
        "mttr_nd3_nd4l_boundary_note": {
            "note_class": "MT-TR — MT-ND3 and MT-ND4L SHARED BOUNDARIES on BOTH sides (rCRS 10404/10405 and 10469/10470)",
            "detail": (
                "MT-TR is sandwiched between two H-strand protein-coding genes with 0 nt gaps "
                "on both sides: MT-ND3 5' boundary (rCRS 10404/10405) and MT-ND4L 3' boundary "
                "(rCRS 10469/10470). Large deletions crossing the 5' boundary also truncate MT-ND3 "
                "3'-end, further reducing CI (ND3 subunit loss adds to CI deficiency). Large deletions "
                "crossing the 3' boundary also truncate MT-ND4L 5'-start, further reducing CI "
                "(ND4L subunit loss). The common 4977 bp deletion (rCRS 8470–13447) removes MT-TR "
                "entirely along with MT-TH, MT-TS2, MT-TL2 downstream — compound multi-tRNA + "
                "protein-coding gene loss causes KSS."
            ),
        },
        "biochemical_fingerprint": {
            "summary": "Combined CI + CIV deficiency; CII (SDH) NORMAL — mt-translation fingerprint; both CI and CIV impaired via loss of CGN-codon translation",
            "complex_i":   "CI reduced 35–55% of normal (NADH-CoQ reductase — mt-encoded ND subunits deficient; CGN Arg codons in proton-translocating helices of ND1-ND6)",
            "complex_ii":  "CII NORMAL (nuclear-encoded entirely — not affected by mt-translation defect)",
            "complex_iv":  "CIV reduced 28–52% of normal; CGN-rich proton-translocating helices in COX1/COX2/COX3 disrupt CIV assembly; catalytic Arg finger motifs require CGN codon translation",
            "mechanism":   "MT-TR translates all 13 mt-encoded OXPHOS subunits at CGN codons (CGA, CGG, CGC, CGU via superwobble); tRNA-Arg deficiency → translation stalling; both CI and CIV subunits impaired",
        },
        "cohort_summary_features": [
            {"feature": "CPEO (chronic progressive external ophthalmoplegia)", "value": str(pct_cpeo), "note": "DOMINANT — progressive ptosis + ophthalmoplegia; bilateral; adult onset; prism/ptosis surgery palliative"},
            {"feature": "Myopathy (limb weakness)",   "value": str(pct_myo),   "note": "RRF + COX-negative fibres on muscle biopsy; proximal > distal weakness"},
            {"feature": "SNHL (sensorineural hearing loss)", "value": str(pct_snhl), "note": "Moderate prevalence; more common with D-stem and T-loop variants; cochlear implants effective"},
            {"feature": "Exercise intolerance",       "value": "70",            "note": "Exertional lactic acidosis; often first presenting symptom; pre-exercise lactate testing"},
            {"feature": "Lactic acidosis (blood)",    "value": "62",            "note": "Elevated rest lactate; crisis at higher heteroplasmy / illness; GIR 6–8 during crisis"},
            {"feature": "Cardiomyopathy (HCM/DCM)",   "value": str(pct_cardio), "note": "Moderate-high — especially m.10438A>G anticodon stem and KSS large-deletion; annual echocardiogram"},
        ],
        "phenotype_distribution": [
            {"variant": "m.10450A>G",    "pct": 28, "phenotype": "CPEO + Myopathy (moderate)",              "position": "T-stem (rCRS 10450)"},
            {"variant": "m.10438A>G",    "pct": 22, "phenotype": "CPEO + Myopathy + Cardiomyopathy",        "position": "Anticodon stem (rCRS 10438)"},
            {"variant": "m.10410T>C",    "pct": 20, "phenotype": "CPEO + Myopathy + SNHL",                  "position": "D-stem (rCRS 10410)"},
            {"variant": "m.10460G>A",    "pct": 15, "phenotype": "Multisystem Leigh-like (bilateral BG)",    "position": "T-loop (rCRS 10460)"},
            {"variant": "LargeDeletion-MT-TR-common-4977bp", "pct": 15, "phenotype": "KSS — compound MT-TR+TH+TS2+TL2 loss", "position": "Common 4977bp deletion rCRS 8470–13447"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<50% blood",  "expected_phenotype": "Exercise intolerance only — subclinical CPEO; no overt ptosis or ophthalmoplegia"},
            {"threshold_pct": "50–65% blood","expected_phenotype": "Ptosis onset + myopathy + exertional lactic acidosis"},
            {"threshold_pct": "65–80% blood","expected_phenotype": "Full CPEO + myopathy + lactic acidosis; SNHL variable; cardiomyopathy emerging"},
            {"threshold_pct": ">80% blood",  "expected_phenotype": "Multi-system mt-disease: CPEO + severe myopathy + lactic acidosis + cardiomyopathy"},
        ],
        "key_molecular_features": [
            "UCG anticodon — decodes CGA (Watson-Crick U34-A), CGG (U34-G wobble), CGC/CGU (superwobble) — all 4 CGN Arg codons in the mitochondrial code via vertebrate mt superwobble",
            "65 nt tRNA — standard H-strand encoded mt-tRNA; no L-strand reverse-complement QC required",
            "Sandwiched between MT-ND3 (5', 0 nt gap) and MT-ND4L (3', 0 nt gap) — both boundaries shared with H-strand protein-coding genes",
            "CGN codons in proton-translocating helices of CI (ND1-ND6) and CIV (COX1-COX3) — tRNA-Arg deficiency disrupts proton translocation across both CI and CIV",
            "RARS2 (mitochondrial Arg-tRNA synthetase) aminoacylates tRNA-Arg; biallelic RARS2 → PCH6 neonatal encephalopathy (NOT adult CPEO)",
            "Common 4977 bp deletion (rCRS 8470–13447) removes MT-TR + MT-TH + MT-TS2 + MT-TL2 → KSS",
            "NO myoclonic epilepsy as cardinal feature (distinguishes from MT-TK/MERRF)",
            "NO stroke-like episodes (distinguishes from MT-TL1/MELAS)",
            "NO Multiple Symmetric Lipomatosis (distinguishes from MT-TK/MERRF)",
            "NO Maternally Inherited Diabetes (distinguishes from MT-TE)",
            "COX-negative RRF on muscle biopsy (combined CI+CIV defect; CII band NORMAL on BN-PAGE)",
        ],
        "clinical_alerts": [
            {
                "alert": "Metformin — ABSOLUTE CONTRAINDICATION (mtDNA disease)",
                "detail": "Metformin inhibits Complex I (mitochondrial respiratory chain). Absolutely contraindicated in all confirmed mtDNA-encoded disease including MT-TR. Risk: severe lactic acidosis, metabolic crisis.",
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
            "entity":       "MT-TR (tRNA-Arg)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "CPEO + myopathy; exercise intolerance; KSS for common 4977bp deletion",
            "oxphos":       "CI + CIV deficiency; CII NORMAL (mt-translation fingerprint); both CI and CIV impaired via CGN codon disruption",
            "ddx_clue":     "CPEO dominant; no myoclonic epilepsy; no SLE; no PCH on MRI; CGN Arg codon superwobble",
            "wes":          "MISSED — WES misses MT-TR (mtDNA H-strand; requires dedicated mtDNA panel)",
        },
        {
            "entity":       "RARS2 (mt-Arg-tRNA synthetase 2, AR) — PCH6",
            "inheritance":  "Autosomal recessive (biallelic)",
            "phenotype":    "Pontocerebellar Hypoplasia type 6 — neonatal encephalopathy; pontocerebellar atrophy MRI",
            "oxphos":       "Combined OXPHOS deficiency (all complexes using mt-translation); pontocerebellar hypoplasia ABSENT in MT-TR",
            "ddx_clue":     "Neonatal/infantile onset; PCH on MRI; severe encephalopathy; NOT adult-onset CPEO/myopathy",
            "wes":          "WES-DETECTABLE — biallelic RARS2 missense/truncation",
        },
        {
            "entity":       "MT-TK (tRNA-Lys, MERRF)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Myoclonic epilepsy + RRF + cerebellar ataxia + MSL; NOT primarily CPEO",
            "oxphos":       "Pan-OXPHOS CI+CIV (CII NORMAL); myoclonic epilepsy ABSENT in MT-TR",
            "ddx_clue":     "KEY DDx: myoclonic epilepsy + MSL present in MERRF but ABSENT in MT-TR",
            "wes":          "MISSED — WES misses MT-TK",
        },
        {
            "entity":       "MT-TL1 (tRNA-Leu, MELAS)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Stroke-like episodes + encephalopathy + lactic acidosis; NOT primarily CPEO",
            "oxphos":       "Pan-OXPHOS CI+CIII+CIV; stroke-like episodes ABSENT in MT-TR",
            "ddx_clue":     "KEY DDx: stroke-like episodes + MRI cortical signal ABSENT in MT-TR",
            "wes":          "MISSED — WES misses MT-TL1",
        },
        {
            "entity":       "MT-TI (tRNA-Ile, m.4300A>G)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Isolated HCM without CPEO at low heteroplasmy — DISTINCTIVE; CI+CIV",
            "oxphos":       "CI+CIV; isolated HCM at <40% heteroplasmy — absent in MT-TR",
            "ddx_clue":     "KEY DDx: isolated HCM without CPEO ABSENT in MT-TR (m.4300A>G specific)",
            "wes":          "MISSED — WES misses MT-TI",
        },
        {
            "entity":       "MT-TE (tRNA-Glu, MIDM)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "CPEO + myopathy + Maternally Inherited Diabetes Mellitus (MIDM) at m.14709T>C",
            "oxphos":       "CI+CIV; MIDM-diabetes ABSENT in MT-TR",
            "ddx_clue":     "KEY DDx: maternally inherited diabetes ABSENT in MT-TR",
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
                "Annual echocardiogram (all MT-TR patients; especially m.10438A>G and KSS large-deletion)",
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
            {"term": "MT-TR",           "definition": "Mitochondrially encoded tRNA-Arg; 65 nt RNA gene; H-strand rCRS 10405–10469; OMIM *590005; SIXTEENTH human mt-tRNA in genomic order"},
            {"term": "tRNA-Arg",        "definition": "Transfer RNA delivering arginine to mt-ribosomes at CGN codons; UCG anticodon; superwobble decodes all 4 CGN codons (CGA, CGG, CGC, CGU) in vertebrate mitochondria"},
            {"term": "UCG anticodon",   "definition": "MT-TR anticodon (positions 34-35-36: U-C-G); U34 is unmodified in human mitochondria enabling superwobble — reads CGA (WC), CGG (U34-G wobble), CGC and CGU (superwobble U34-C/U)"},
            {"term": "rCRS 10405–10469","definition": "Revised Cambridge Reference Sequence positions 10405 to 10469 — MT-TR H-strand locus; 0 nt from MT-ND3 3' end (shared boundary 10404/10405); 0 nt from MT-ND4L 5' start (shared boundary 10469/10470)"},
            {"term": "ND3/TR/ND4L boundaries", "definition": "MT-TR is sandwiched between MT-ND3 (5', 0 nt gap, shared boundary 10404/10405) and MT-ND4L (3', 0 nt gap, shared boundary 10469/10470); large deletions crossing either boundary affect the flanking CI subunit gene in addition to MT-TR tRNA loss"},
            {"term": "Common 4977bp deletion", "definition": "The most common pathogenic large mtDNA deletion (rCRS 8470–13447, ~4977 bp); removes MT-TR plus MT-TH, MT-TS2, MT-TL2; causes KSS triad (CPEO + pigmentary retinopathy + cardiac conduction defect)"},
        ],
        "biochemical_definitions": [
            {"term": "CI+CIV deficiency",  "definition": "Combined Complex I (NADH-CoQ reductase) and Complex IV (cytochrome c oxidase) deficiency — the mt-translation fingerprint; CII (SDH) NORMAL as it is nuclear-encoded"},
            {"term": "CGN superwobble",    "definition": "Vertebrate mitochondrial superwobble: unmodified U34 in MT-TR anticodon reads all 4 CGN codons (CGA, CGG, CGC, CGU) — arginine is delivered by a single mt-tRNA across all CGN codon usage"},
            {"term": "Proton-translocating helices", "definition": "Arginine residues in proton-translocating helices of CI (ND subunits) and CIV (COX subunits) are critical for H+ pumping; CGN codons are enriched in these functional domains — tRNA-Arg deficiency disrupts proton translocation"},
            {"term": "BN-PAGE",            "definition": "Blue-native polyacrylamide gel electrophoresis — CI+CIV bands reduced; CII band normal in MT-TR disease"},
            {"term": "Lactic acidosis",    "definition": "Elevated blood lactate (>2 mmol/L rest; >4 mmol/L exertional) from impaired OXPHOS → pyruvate→lactate shunting"},
            {"term": "RRF",               "definition": "Ragged-Red Fibres — Gomori trichrome staining of muscle biopsy; accumulation of abnormal mitochondria; pathognomonic of mt-tRNA disease"},
            {"term": "COX-negative fibres","definition": "CIV-deficient muscle fibres; SDH-positive (blue) but COX-negative (pale) on sequential staining — hallmark of heteroplasmic mt-tRNA disease"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",  "definition": "Chronic Progressive External Ophthalmoplegia — progressive ptosis + ophthalmoplegia; bilateral; adult onset; DOMINANT in MT-TR"},
            {"term": "KSS",   "definition": "Kearns-Sayre Syndrome — onset <20 years: CPEO + pigmentary retinopathy + cardiac conduction defect; CSF protein >100 mg/dL; caused by common 4977bp deletion including MT-TR"},
            {"term": "PCH6",  "definition": "Pontocerebellar Hypoplasia type 6 — RARS2 nuclear DDx; neonatal/infantile encephalopathy; pontocerebellar atrophy on MRI; NOT MT-TR (adult CPEO/myopathy)"},
            {"term": "BTBGD", "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease (SLC19A3) — treatable Leigh-like mimic; mandatory exclusion before diagnosing irreversible mtDNA disease"},
            {"term": "AV block", "definition": "Atrioventricular heart block — cardiac conduction defect in KSS (common 4977bp deletion); progressive from 1st to complete; pacemaker required if complete or PR >240ms"},
        ],
        "ngs_definitions": [
            {"term": "H-strand encoding (MT-TR)", "definition": "MT-TR at rCRS 10405–10469 is H-strand encoded — standard NGS pipelines cover H-strand; no dedicated L-strand reverse-complement QC window required for MT-TR itself"},
            {"term": "0-nt boundary (5' and 3')", "definition": "MT-TR shares 0-nt boundaries with MT-ND3 (5') and MT-ND4L (3'); pipeline coverage must confirm MT-TR is not shadowed by flanking gene read mapping artefacts"},
            {"term": "Heteroplasmy threshold",    "definition": "Mutation load (%mutant mtDNA) above which features appear; MT-TR threshold typically >50% blood for CPEO onset; muscle heteroplasmy exceeds blood by 10–20%"},
        ],
        "drug_definitions": [
            {"term": "Metformin",        "definition": "ABSOLUTE CI — biguanide complex-I inhibitor; severe lactic acidosis risk in mtDNA disease"},
            {"term": "Valproate (VPA)",  "definition": "ABSOLUTE CI — inhibits mitochondrial β-oxidation and CI; use LEV instead"},
            {"term": "Propofol",         "definition": "ABSOLUTE CI — propofol infusion syndrome (PRIS): decouples OXPHOS; fatal myocardial failure; use ketamine or volatile agents"},
            {"term": "Linezolid",        "definition": "ABSOLUTE CI — inhibits mitochondrial 23S-rRNA protein synthesis; reversible but severe mtDNA disease exacerbation"},
            {"term": "Chloramphenicol",  "definition": "ABSOLUTE CI — inhibits mt-ribosome peptidyl transferase; severe mt-translation failure"},
            {"term": "LEV (Levetiracetam)", "definition": "Preferred AED — renal excretion; no cytochrome P450 induction; no mitochondrial toxicity; Level C evidence"},
            {"term": "Beta-Blocker",     "definition": "Indicated if cardiomyopathy (HCM) develops (esp. m.10438A>G and KSS large-deletion); cardiology-guided"},
        ],
        "references": [
            {"ref": "DiMauro & Schon 2003 NEJM",          "citation": "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases — comprehensive mt-tRNA mutation review; NEJM 348:2656–2668"},
            {"ref": "Schaefer 2008 Ann Neurol",            "citation": "Schaefer AM et al. (2008) — prevalence of mitochondrial disease in adults; regional UK survey; ~1:5000 adult prevalence"},
            {"ref": "Gorman 2016 Nat Rev Dis Primers",     "citation": "Gorman GS et al. (2016) — Mitochondrial diseases — comprehensive review including mt-tRNA genetics, pathophysiology, management"},
            {"ref": "Edvardson 2007 Am J Hum Genet",       "citation": "Edvardson S et al. (2007) — Deleterious mutation in the mitochondrial arginyl-transfer RNA synthetase gene is associated with pontocerebellar hypoplasia — RARS2/PCH6 first report"},
            {"ref": "Saito 2014 AJHG RARS2",               "citation": "Saito K et al. (2014) — RARS2 mutations in pontocerebellar hypoplasia type 6 — clinical spectrum and molecular pathology of RARS2/PCH6"},
            {"ref": "Boczonadi & Horvath 2014 IJBCB",      "citation": "Boczonadi V, Horvath R (2014) — Mitochondria: impaired mitochondrial translation causes neurological disorders — mt-tRNA synthetase review including RARS2"},
            {"ref": "Schon et al. 1989 Science (common deletion)", "citation": "Schon EA et al. (1989) — A direct repeat is a hotspot for large-scale deletions of human mitochondrial DNA — common 4977 bp deletion first described; KSS/CPEO including MT-TR region"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== MT-TR OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== MT-TR BREAKDOWN (first variant) ===")
    bd = get_breakdown()
    print(json.dumps(bd["variant_breakdown"][0], indent=2))
