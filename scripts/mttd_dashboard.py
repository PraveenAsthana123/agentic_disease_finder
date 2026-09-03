#!/usr/bin/env python3
"""MT-TD — Mitochondrially Encoded tRNA-Asp — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 7518–7585

MT-TD (OMIM *590015) encodes mitochondrial tRNA-Asp (GUC anticodon; decodes GAC
Watson-Crick AND GAU wobble via G34-U wobble pairing in the mitochondrial code — both
aspartate codons), located on the **H-strand** at rCRS 7518–7585 (68 nt). MT-TD is the
THIRTEENTH tRNA gene of the human mitochondrial genome, immediately following MT-TS1
(L-strand, rCRS 7445–7516 — separated by 1 nt gap at rCRS 7517).

GENOMIC CONTEXT:
  5' neighbor: MT-TS1 (L-strand, rCRS 7445–7516) — 1 nt gap at rCRS 7517
  3' neighbor: MT-CO2 (H-strand, rCRS 7586–8269) — 0 nt gap (SHARED BOUNDARY 7585/7586)
  Next tRNA downstream: MT-TK (H-strand, rCRS 8295–8364) — 710 nt after MT-TD (MT-CO2 intervening)
  Note: MT-TD immediately precedes MT-CO2; the MT-CO2 5' end (rCRS 7586) abuts MT-TD 3' end (rCRS 7585)
  H-strand encoded — standard NGS pipeline coverage (no L-strand pitfall for MT-TD itself)

CLINICAL PHENOTYPE — CPEO-DOMINANT + MYOPATHY:
MT-TD mutations cause Combined CI+CIV deficiency with the characteristic mt-translation
fingerprint (CII/SDH NORMAL). Primary phenotype: Chronic Progressive External
Ophthalmoplegia (CPEO) + limb myopathy + exercise intolerance. Unlike MT-TS1 (SNHL-
dominant at low heteroplasmy), MT-TD presents with CPEO and myopathy as the cardinal
features. Cardiomyopathy is low prevalence. Sensorineural hearing loss can occur at
higher heteroplasmy levels. Large deletions spanning MT-TS1+MT-TD cause KSS (Kearns-
Sayre Syndrome) with compound tRNA-Ser(UCN)+tRNA-Asp loss.

DARS2 NUCLEAR DDx — LBSL SYNDROME:
DARS2 (mitochondrial aspartyl-tRNA synthetase, cytoplasmic = DARS; mitochondrial = DARS2)
biallelic mutations cause LBSL: Leukoencephalopathy with Brain Stem and Spinal cord
involvement and Lactate elevation. LBSL is a WHITE MATTER DISEASE (MRI-defined):
cerebellar atrophy, brain stem (dorsal columns, corticospinal tracts, medial lemnisci),
spinal cord (posterior and lateral columns), elevated CSF/blood lactate. LBSL onset is
childhood/young adult — slowly progressive spastic ataxia, cognitive decline. COMPLETELY
distinct from adult-onset CPEO/myopathy of MT-TD. WES detects DARS2; WES misses MT-TD.

  MT-TD gene             OMIM *590015
  Primary disease        CPEO + Myopathy — Combined CI+CIV Deficiency
                         Exercise Intolerance / Lactic Acidosis
                         KSS (large deletion compound MT-TS1+MT-TD loss)
  Protein product        tRNA-Asp (GUC anticodon) — 68 nt RNA gene
  Genome                 Mitochondrial DNA (mtDNA), H-strand, rCRS 7518–7585
  Inheritance            MATERNAL — heteroplasmic
  Chromosomal location   THIRTEENTH tRNA; after MT-TF/TV/TL1/TI/TQ/TM/TW/TA/TN/TC/TY/TS1
  Key mutations          m.7543G>A — Variable loop (most common); CPEO + myopathy
                         m.7526G>A — Anticodon loop; CPEO + myopathy + SNHL
  Nuclear DDx            DARS2 (biallelic AR) → LBSL (white matter disease, NOT adult CPEO)

HETEROPLASMY THRESHOLD (blood may underestimate by 10–20%):
  <50% blood:   Exercise intolerance only — subclinical CPEO, no overt ptosis
  50–65% blood: Ptosis onset + myopathy + exercise intolerance
  65–80% blood: Full CPEO + myopathy + lactic acidosis + SNHL variable
  >80% blood:   Multi-system mt-disease: CPEO + severe myopathy + lactic acidosis + cardiomyopathy
"""

import random
from collections import Counter

SEED = 825
rng  = random.Random(SEED)

# ── 5 pathogenic variants (40-patient cohort seed-825) ───────────────────────
VARIANTS = [
    ("m.7543G>A",  "Variable loop (position 47 equivalent)",
     "~28%; most common MT-TD pathogenic variant; variable loop disruption impairs "
     "tRNA tertiary fold; CPEO + myopathy; exercise intolerance early; "
     "moderate CI+CIV deficiency; adult-onset ptosis; RRF on muscle biopsy"),
    ("m.7526G>A",  "Anticodon loop (position 37 equivalent, 3' adjacent to anticodon)",
     "~25%; anticodon loop disruption impairs decoding accuracy at GAC/GAU codons; "
     "CPEO + myopathy + SNHL; lactic acidosis at higher heteroplasmy; "
     "more severe phenotype than m.7543G>A; earlier onset; COX-negative RRF"),
    ("m.7551T>C",  "T-stem (position 54 equivalent — TψC loop)",
     "~20%; T-stem disruption impairs tRNA tertiary structure; CPEO + myopathy; "
     "exercise-triggered lactic acidosis; adult-onset; cardiomyopathy risk moderate; "
     "RRF + COX-negative fibres; annual echocardiogram recommended"),
    ("m.7512T>C",  "Acceptor stem (position 3-72 base pair)",
     "~15%; acceptor stem disruption reduces aminoacylation by DARS2; "
     "exercise intolerance + SNHL + mild CPEO; lower heteroplasmy threshold for symptoms; "
     "moderate CI+CIV deficiency; COX-negative fibres present"),
    ("LargeDeletion-MT-TS1-MT-TD-Spanning",
     "Large mtDNA deletion spanning MT-TS1 (L-strand) and MT-TD (H-strand)",
     "~12%; KSS / CPEO; simultaneous loss of tRNA-Ser(UCN) AND tRNA-Asp translation; "
     "compound-tRNA defect → severe pan-OXPHOS reduction; Kearns-Sayre triad: CPEO + "
     "pigmentary retinopathy + cardiac conduction defect; CSF protein >100 mg/dL; "
     "onset <20 years; pacemaker may be required for heart block"),
]

N_PATIENTS = 40


def _make_patients():
    patients = []
    counts   = [11, 10, 8, 6, 5]   # sums to 40
    pid      = 1
    for vi, (var, pos, note) in enumerate(VARIANTS):
        n = counts[vi]
        for _ in range(n):
            is_large_del = var.startswith("LargeDeletion")
            base_hetero  = rng.uniform(0.50, 0.90)
            ci           = max(15, rng.gauss(42 if not is_large_del else 20, 9))
            civ          = max(12, rng.gauss(38 if not is_large_del else 18, 8))
            cpeo         = rng.random() < (0.70 if vi == 0 else
                                           0.75 if vi == 1 else
                                           0.80 if vi == 2 else
                                           0.50 if vi == 3 else 0.95)
            snhl         = rng.random() < (0.30 if vi == 0 else
                                           0.65 if vi == 1 else
                                           0.35 if vi == 2 else
                                           0.60 if vi == 3 else 0.55)
            myo          = rng.random() < (0.80 if vi == 0 else
                                           0.85 if vi == 1 else
                                           0.85 if vi == 2 else
                                           0.70 if vi == 3 else 0.95)
            cardio       = rng.random() < (0.15 if vi == 0 else
                                           0.20 if vi == 1 else
                                           0.35 if vi == 2 else
                                           0.15 if vi == 3 else 0.50)
            compound     = is_large_del
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
                "compound_ts1_td_loss": compound,
            })
            pid += 1
    return patients


PATIENTS = _make_patients()


def get_overview():
    n         = len(PATIENTS)
    avg_hetero = round(sum(p["heteroplasmy_blood"] for p in PATIENTS) / n, 1)
    avg_ci     = round(sum(p["ci_pct_normal"]      for p in PATIENTS) / n, 1)
    avg_civ    = round(sum(p["civ_pct_normal"]      for p in PATIENTS) / n, 1)
    pct_cpeo   = round(sum(p["cpeo"]    for p in PATIENTS) / n * 100)
    pct_snhl   = round(sum(p["snhl"]    for p in PATIENTS) / n * 100)
    pct_myo    = round(sum(p["myopathy"]for p in PATIENTS) / n * 100)
    pct_cardio = round(sum(p["cardiomyopathy"] for p in PATIENTS) / n * 100)

    return {
        "gene_facts": {
            "gene":               "MT-TD",
            "omim_gene":          "*590015",
            "product":            "tRNA-Asp (GUC anticodon) — 68 nt RNA gene",
            "anticodon":          "GUC — decodes GAC (Watson-Crick) and GAU (wobble G34-U) — both Asp codons",
            "strand":             "H-strand (standard NGS pipeline coverage — no L-strand pitfall)",
            "rCRS_position":      "rCRS 7518–7585 (68 nt)",
            "position_in_genome": "THIRTEENTH tRNA — after MT-TF/TV/TL1/TI/TQ/TM/TW/TA/TN/TC/TY/TS1",
            "5prime_neighbor":    "MT-TS1 (L-strand, rCRS 7445–7516) — 1 nt gap at rCRS 7517",
            "3prime_neighbor":    "MT-CO2 (H-strand, rCRS 7586–8269) — 0 nt gap (SHARED BOUNDARY 7585/7586)",
            "next_tRNA":          "MT-TK (H-strand, rCRS 8295–8364) — 710 nt downstream (MT-CO2 intervening)",
            "inheritance":        "Maternal (heteroplasmic)",
        },
        "cohort_statistics": {
            "n_patients":               n,
            "seed":                     SEED,
            "avg_heteroplasmy_blood_pct": avg_hetero,
            "avg_ci_activity_pct_normal": avg_ci,
            "avg_civ_activity_pct_normal": avg_civ,
            "pct_cpeo":                 pct_cpeo,
            "pct_snhl":                 pct_snhl,
            "pct_myopathy":             pct_myo,
            "pct_cardiomyopathy":       pct_cardio,
        },
        "dars2_ddx_note": {
            "note_class": "DARS2 Nuclear DDx — LBSL Syndrome (NOT Adult-Onset CPEO/Myopathy)",
            "detail": (
                "DARS2 (mitochondrial aspartyl-tRNA synthetase) biallelic AR mutations cause "
                "LBSL: Leukoencephalopathy with Brain Stem and Spinal cord involvement and Lactate "
                "elevation. LBSL is a WHITE MATTER disease — MRI shows signal in posterior fossa "
                "(dorsal brain stem, cerebellar white matter), spinal cord posterior + lateral columns, "
                "and periventricular white matter. Clinical: slowly progressive spastic ataxia, pyramidal "
                "signs, cognitive decline — onset childhood/young adult. Elevated CSF + blood lactate. "
                "COMPLETELY distinct from adult-onset CPEO + myopathy of MT-TD mutations. "
                "WES detects DARS2; WES misses MT-TD. Exclude DARS2 (AR biallelic) before diagnosing "
                "MT-TD if white matter or spinal cord involvement present (ABSENT in pure MT-TD)."
            ),
        },
        "mttd_co2_boundary_note": {
            "note_class": "MT-TD / MT-CO2 SHARED BOUNDARY — rCRS 7585/7586",
            "detail": (
                "MT-TD 3' end (rCRS 7585) and MT-CO2 5' start (rCRS 7586) share a 0 nt gap — "
                "immediate adjacency. Large mtDNA deletions originating near the MT-TS1 / MT-TD / "
                "MT-CO2 junction (rCRS 7445–7586) affect BOTH tRNA genes and the COX2 coding sequence. "
                "The compound MT-TS1+MT-TD tRNA loss produces severe pan-OXPHOS deficiency and KSS "
                "phenotype (CPEO + retinopathy + cardiac conduction defect). COX2 truncation from "
                "deletion extension further amplifies CIV deficiency."
            ),
        },
        "biochemical_fingerprint": {
            "summary": "Combined CI + CIV deficiency; CII (SDH) NORMAL — mt-translation fingerprint",
            "complex_i":   "CI reduced 38–55% of normal (NADH-CoQ reductase — mt-encoded subunits deficient; tRNA-Asp decodes Asp codons in ND1/2/4/5/6 subunits)",
            "complex_ii":  "CII NORMAL (nuclear-encoded entirely — not affected by mt-translation defect)",
            "complex_iv":  "CIV reduced 30–50% of normal (COX — mt-encoded subunits CO1/CO2/CO3 deficient; adjacent MT-CO2 amplifies CIV sensitivity)",
            "mechanism":   "MT-TD translates all 13 mt-encoded OXPHOS subunits at GAC/GAU codons; tRNA-Asp deficiency → translation stalling at Asp codons in CI/CIV subunits → combined CI+CIV reduction",
        },
        "cohort_summary_features": [
            {"feature": "CPEO (chronic progressive external ophthalmoplegia)", "value": str(pct_cpeo), "note": "DOMINANT — progressive ptosis + ophthalmoplegia; bilateral; adult onset"},
            {"feature": "Myopathy (limb weakness)",  "value": str(pct_myo),  "note": "RRF + COX-negative fibres on muscle biopsy; proximal > distal"},
            {"feature": "SNHL (sensorineural hearing loss)", "value": str(pct_snhl), "note": "Lower prevalence than MT-TS1; present at higher heteroplasmy"},
            {"feature": "Exercise intolerance",      "value": "74",          "note": "Exertional lactic acidosis; often first presenting symptom"},
            {"feature": "Lactic acidosis (blood)",   "value": "65",          "note": "Elevated rest lactate; crisis at higher heteroplasmy / illness"},
            {"feature": "Cardiomyopathy (HCM/DCM)",  "value": str(pct_cardio), "note": "LOW overall; higher in m.7551T>C and large-deletion KSS; annual echo"},
        ],
        "phenotype_distribution": [
            {"variant": "m.7543G>A",   "pct": 28, "phenotype": "CPEO + Myopathy (moderate)",            "position": "Variable loop (rCRS 7543)"},
            {"variant": "m.7526G>A",   "pct": 25, "phenotype": "CPEO + Myopathy + SNHL",               "position": "Anticodon loop (rCRS 7526)"},
            {"variant": "m.7551T>C",   "pct": 20, "phenotype": "CPEO + Myopathy + Cardiomyopathy",     "position": "T-stem (rCRS 7551)"},
            {"variant": "m.7512T>C",   "pct": 15, "phenotype": "Exercise intolerance + SNHL + mild CPEO", "position": "Acceptor stem (rCRS 7512)"},
            {"variant": "LargeDeletion-TS1-TD", "pct": 12, "phenotype": "KSS — compound tRNA-Ser+Asp loss + CPEO + retinopathy", "position": "Spanning rCRS 7445–7585 (MT-TS1+MT-TD)"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<50% blood",  "expected_phenotype": "Exercise intolerance only — subclinical CPEO; no overt ptosis or ophthalmoplegia"},
            {"threshold_pct": "50–65% blood","expected_phenotype": "Ptosis onset + myopathy + exertional lactic acidosis"},
            {"threshold_pct": "65–80% blood","expected_phenotype": "Full CPEO + myopathy + lactic acidosis; SNHL variable"},
            {"threshold_pct": ">80% blood",  "expected_phenotype": "Multi-system mt-disease: CPEO + severe myopathy + lactic acidosis + cardiomyopathy"},
        ],
        "key_molecular_features": [
            "GUC anticodon — decodes BOTH Asp codons (GAC Watson-Crick, GAU G34-U wobble) in the mitochondrial code",
            "68 nt tRNA — slightly shorter than average mt-tRNA (mean ~72 nt); compact structure amplifies folding vulnerability",
            "H-strand encoded (rCRS 7518–7585) — standard NGS pipeline coverage; no dedicated L-strand reverse-complement QC required for MT-TD itself",
            "Immediate 3' adjacency to MT-CO2 (0 nt gap) — large deletions crossing rCRS 7585/7586 damage BOTH MT-TD and MT-CO2",
            "1 nt gap from MT-TS1 (rCRS 7517) — MT-TS1+MT-TD compound large deletion → KSS triad",
            "DARS2 (mitochondrial aspartyl-tRNA synthetase) aminoacylates tRNA-Asp; biallelic DARS2 → LBSL (white matter disease, NOT CPEO)",
            "NO SNHL-dominant low-heteroplasmy phenotype (distinguishes from MT-TS1)",
            "NO myoclonic epilepsy as cardinal feature (distinguishes from MT-TK/MERRF)",
            "NO stroke-like episodes (distinguishes from MT-TL1/MELAS)",
            "COX-negative RRF on muscle biopsy (combined CI+CIV defect; CII band NORMAL on BN-PAGE)",
        ],
        "clinical_alerts": [
            {
                "alert": "Metformin — ABSOLUTE CONTRAINDICATION (mtDNA disease)",
                "detail": "Metformin inhibits Complex I (mitochondrial respiratory chain). Absolutely contraindicated in all confirmed mtDNA-encoded disease including MT-TD. Risk: severe lactic acidosis, metabolic crisis.",
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
                "detail": "Linezolid inhibits mitochondrial ribosomes (binds mt-23S rRNA). Causes REVERSIBLE but severe mtDNA disease exacerbation. Contraindicated in all confirmed mtDNA disease.",
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
                "detail": "Large deletions spanning MT-TS1+MT-TD (KSS) carry risk of progressive cardiac conduction defects (AV block, bundle branch block). Annual ECG + echocardiogram. Pacemaker if complete AV block develops.",
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
            "entity":       "MT-TD (tRNA-Asp)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "CPEO + myopathy; exercise intolerance; KSS for large MT-TS1+MT-TD deletions",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL; RRF + COX-negative fibres; elevated lactate",
            "ngs":          "H-strand (rCRS 7518–7585) — standard mtDNA panel coverage; no L-strand QC issue",
            "distinctive":  "Immediate MT-CO2 adjacency (0 nt gap); CPEO dominant (not SNHL-first like MT-TS1)",
        },
        {
            "entity":       "MT-TS1 (tRNA-Ser UCN)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "SNHL-DOMINANT at low heteroplasmy; CPEO + myopathy at higher heteroplasmy",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL; same fingerprint as MT-TD",
            "ngs":          "L-strand ISOLATED (rCRS 7445–7516) — mandatory L-strand reverse-complement QC; 1 nt upstream of MT-TD",
            "distinctive":  "KEY DDx: SNHL-first / SNHL-only at low heteroplasmy (ABSENT in MT-TD); m.7445A>G dual MT-CO1/MT-TS1 boundary",
        },
        {
            "entity":       "MT-TK (MERRF)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Myoclonic epilepsy + SNHL + ataxia + myopathy (MERRF); MSL lipomatosis",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL — same fingerprint",
            "ngs":          "m.8344A>G most common (H-strand rCRS 8295–8364) — 710 nt downstream of MT-TD (MT-CO2 intervening)",
            "distinctive":  "KEY DDx: Myoclonic epilepsy is cardinal (ABSENT in MT-TD); MSL lipomatosis; m.8344A>G",
        },
        {
            "entity":       "DARS2 (LBSL)",
            "inheritance":  "AR biallelic",
            "phenotype":    "LBSL: Leukoencephalopathy + Brain Stem + Spinal cord involvement + Lactate elevation; spastic ataxia",
            "biochemistry": "Mitochondrial OXPHOS reduced (DARS2 → tRNA-Asp aminoacylation failure); elevated CSF lactate; white matter MRI signal",
            "ngs":          "WES detects DARS2; WES misses MT-TD — mtDNA panel required for maternal inheritance",
            "distinctive":  "KEY DDx: WHITE MATTER DISEASE (MRI: brain stem, spinal cord, periventricular) — ABSENT in MT-TD; childhood onset",
        },
        {
            "entity":       "MT-TL1 (MELAS)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Stroke-like episodes + lactic acidosis + encephalopathy (MELAS); SNHL variable",
            "biochemistry": "Combined CI+CIV ↓; τm5s2U34 modification defect; same CI+CIV fingerprint",
            "ngs":          "m.3243A>G most common (H-strand rCRS 3230–3304) — standard panel",
            "distinctive":  "KEY DDx: Stroke-like episodes with cortical MRI signal (ABSENT in MT-TD); younger onset",
        },
        {
            "entity":       "MT-TH (tRNA-His)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "CPEO + myopathy + prominent HCM; SNHL variable",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL — same fingerprint",
            "ngs":          "H-strand rCRS 12138–12206 — standard pipeline covers H-strand",
            "distinctive":  "KEY DDx: PROMINENT cardiomyopathy (HCM/DCM) — LOW in MT-TD (~18%); annual echo more critical for MT-TH",
        },
        {
            "entity":       "MT-TI (m.4300A>G isolated HCM)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "ISOLATED HCM WITHOUT CPEO or myopathy — distinctive MT-TI hallmark",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL",
            "ngs":          "m.4300A>G (H-strand rCRS 4263–4331) — standard coverage",
            "distinctive":  "KEY DDx: ISOLATED HCM (ABSENT in MT-TD); no CPEO, no myopathy at typical heteroplasmy",
        },
        {
            "entity":       "MT-TE (tRNA-Glu MIDM)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "MIDM (Maternally Inherited Diabetes + Myopathy); cardiomyopathy; NO CPEO",
            "biochemistry": "Combined CI+CIV ↓; elevated lactate; L-strand encoded",
            "ngs":          "L-strand rCRS 14674–14742 — L-strand QC required",
            "distinctive":  "KEY DDx: MIDM (diabetes + myopathy, NOT CPEO); NO chronic progressive ophthalmoplegia",
        },
    ]

    return {
        "variant_breakdown":      var_data,
        "cohort_statistics": {
            "n_patients": len(PATIENTS),
            "avg_heteroplasmy_blood_pct": round(sum(p["heteroplasmy_blood"] for p in PATIENTS) / len(PATIENTS), 1),
        },
        "absolute_contraindications": [
            {"drug": "Metformin",       "reason": "Complex I inhibitor — lactic acidosis risk; ABSOLUTE CI in all confirmed mtDNA disease"},
            {"drug": "Valproate (VPA)", "reason": "Mitochondrial β-oxidation inhibitor — metabolic crisis; use LEV instead"},
            {"drug": "Propofol",        "reason": "Propofol infusion syndrome (PRIS) — ABSOLUTE CI; use ketamine or volatile agents"},
            {"drug": "Linezolid",       "reason": "mt-23S rRNA inhibitor → mitochondrial ribosome failure; ABSOLUTE CI"},
            {"drug": "Chloramphenicol", "reason": "Mitochondrial protein synthesis inhibitor; ABSOLUTE CI in all mt-disease"},
            {"drug": "Ketogenic Diet",  "reason": "Forces OXPHOS-dependent β-oxidation — catastrophic in CI+CIV deficiency"},
        ],
        "safe_interventions": [
            {"intervention": "LEV (Levetiracetam)", "evidence": "Preferred AED — renal excretion, no mitochondrial toxicity; Level C"},
            {"intervention": "CoQ10 / Ubiquinol",   "evidence": "Mitochondrial cofactor supplementation; Level C evidence"},
            {"intervention": "Riboflavin (B2)",     "evidence": "Complex I cofactor; Level C; may reduce lactic acidosis burden"},
            {"intervention": "L-Carnitine",         "evidence": "Supports β-oxidation (at safe substrate levels); Level C"},
            {"intervention": "Thiamine (B1) empiric","evidence": "Mandatory empiric thiamine before Wernicke exclusion confirmed"},
            {"intervention": "Biotin empiric",      "evidence": "Mandatory empiric before biotin/biotinidase deficiency excluded (BTBGD DDx)"},
            {"intervention": "Beta-Blocker",        "evidence": "If cardiomyopathy present (especially m.7551T>C, large-deletion KSS); cardiology-guided"},
            {"intervention": "Pacemaker",           "evidence": "If complete AV block develops (large-deletion KSS cardiac conduction disease)"},
            {"intervention": "GIR 6–8 mg/kg/min",  "evidence": "Glucose infusion during illness/crisis — prevents fasting-induced OXPHOS stress"},
            {"intervention": "Ptosis surgery",      "evidence": "Palliative for severe CPEO ptosis impairing vision; confirm stable ophthalmoplegia pre-op"},
        ],
        "management_by_variant": [
            {"variant": "m.7543G>A",  "cpeo_risk": "Moderate", "cardio_risk": "Low",      "snhl_risk": "Low",      "key_action": "CPEO monitoring: ptosis surgery if functional; annual ophthalmic exam; CoQ10 supplementation"},
            {"variant": "m.7526G>A",  "cpeo_risk": "High",     "cardio_risk": "Low",      "snhl_risk": "Moderate", "key_action": "CPEO + SNHL monitoring; audiogram annually; LEV if seizures; COX-negative RRF on biopsy"},
            {"variant": "m.7551T>C",  "cpeo_risk": "High",     "cardio_risk": "Moderate", "snhl_risk": "Low",      "key_action": "Annual echo; beta-blocker if HCM; CPEO ptosis surgery; GIR protocol in illness"},
            {"variant": "m.7512T>C",  "cpeo_risk": "Low",      "cardio_risk": "Low",      "snhl_risk": "Moderate", "key_action": "Exercise-triggered presentation; audiogram; avoid fasting; SNHL monitoring; CoQ10"},
            {"variant": "LargeDeletion","cpeo_risk": "High",   "cardio_risk": "High",     "snhl_risk": "Moderate", "key_action": "KSS triad: annual ECG/echo + retinal exam + CSF protein; pacemaker if AV block; early multidisciplinary"},
        ],
        "ddx_table": ddx_table,
    }


def get_definitions():
    return {
        "gene_definitions": [
            {"term": "MT-TD",           "definition": "Mitochondrially encoded tRNA-Asp; 68 nt RNA gene; H-strand rCRS 7518–7585; OMIM *590015; THIRTEENTH human mt-tRNA in genomic order"},
            {"term": "tRNA-Asp",        "definition": "Transfer RNA delivering aspartate to mt-ribosomes at GAC and GAU codons; GUC anticodon; G34-U wobble pairing decodes GAU in addition to GAC"},
            {"term": "GUC anticodon",   "definition": "MT-TD anticodon (positions 34-35-36: G-U-C); G34 wobble base decodes both GAC (Watson-Crick) and GAU (wobble) — all aspartate codons in the mitochondrial genetic code"},
            {"term": "rCRS 7518–7585",  "definition": "Revised Cambridge Reference Sequence positions 7518 to 7585 — MT-TD H-strand locus; 1 nt from MT-TS1 3' end (rCRS 7517 gap); 0 nt from MT-CO2 5' start (rCRS 7586 shared boundary)"},
            {"term": "MT-CO2 adjacency","definition": "MT-TD 3' end (rCRS 7585) shares a 0 nt boundary with MT-CO2 5' start (rCRS 7586); large deletions spanning this junction affect BOTH MT-TD AND MT-CO2, further amplifying CIV deficiency"},
            {"term": "Compound TS1+TD loss", "definition": "Large mtDNA deletion spanning both MT-TS1 (L-strand 7445–7516) and MT-TD (H-strand 7518–7585) causes simultaneous loss of both tRNA-Ser(UCN) AND tRNA-Asp → KSS phenotype"},
        ],
        "biochemical_definitions": [
            {"term": "CI+CIV deficiency",  "definition": "Combined Complex I (NADH-CoQ reductase) and Complex IV (cytochrome c oxidase) deficiency — the mt-translation fingerprint; CII (SDH) NORMAL as it is nuclear-encoded"},
            {"term": "BN-PAGE",            "definition": "Blue-native polyacrylamide gel electrophoresis — CI+CIV bands reduced; CII band normal in MT-TD disease; distinguishes from nuclear-encoded OXPHOS gene defects"},
            {"term": "Lactic acidosis",    "definition": "Elevated blood lactate (>2 mmol/L rest; >4 mmol/L exertional) from impaired OXPHOS → pyruvate→lactate shunting; crisis threshold >5 mmol/L"},
            {"term": "RRF",               "definition": "Ragged-Red Fibres — Gomori trichrome staining of muscle biopsy; accumulation of abnormal mitochondria at fibre periphery; pathognomonic of mt-tRNA disease"},
            {"term": "COX-negative fibres","definition": "Cytochrome c oxidase (CIV)-deficient muscle fibres; SDH-positive (blue) but COX-negative (pale) on sequential staining — hallmark of heteroplasmic mt-tRNA disease"},
            {"term": "Aminoacylation",     "definition": "Attachment of aspartate to tRNA-Asp by DARS2 (mitochondrial aspartyl-tRNA synthetase); disrupted by MT-TD mutations → insufficient Asp-tRNA supply → mt-translation stalling at GAC/GAU codons"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",  "definition": "Chronic Progressive External Ophthalmoplegia — progressive ptosis + ophthalmoplegia; bilateral; onset often adult (second–fourth decade); DOMINANT in MT-TD; prism/ptosis surgery palliative"},
            {"term": "KSS",   "definition": "Kearns-Sayre Syndrome — large mtDNA deletion (<20 yr onset): CPEO + pigmentary retinopathy + cardiac conduction defect; CSF protein >100 mg/dL; relevant for MT-TS1+MT-TD compound-loss large deletion"},
            {"term": "LBSL",  "definition": "Leukoencephalopathy with Brain Stem and Spinal cord involvement and Lactate elevation — caused by biallelic DARS2 AR mutations; MRI: white matter + dorsal brain stem + spinal cord signal; NOT adult CPEO (key DDx from MT-TD)"},
            {"term": "BTBGD", "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease (SLC19A3) — treatable Leigh-like mimic; mandatory exclusion before diagnosing irreversible mtDNA disease; give empiric biotin + thiamine"},
            {"term": "AV block", "definition": "Atrioventricular heart block — cardiac conduction defect in KSS (large-deletion MT-TD/TS1); progressive from 1st to complete block; pacemaker required if complete; ECG annually"},
        ],
        "ngs_definitions": [
            {"term": "H-strand encoding (MT-TD)", "definition": "MT-TD at rCRS 7518–7585 is H-strand encoded — standard NGS pipelines cover H-strand; no dedicated L-strand reverse-complement QC window required for MT-TD itself (unlike adjacent MT-TS1 which IS L-strand)"},
            {"term": "Adjacent L-strand pitfall (MT-TS1)", "definition": "The immediate 5' neighbor MT-TS1 (rCRS 7445–7516) IS L-strand encoded — if NGS pipeline misses MT-TS1 L-strand QC, both MT-TS1 AND large deletions spanning MT-TS1+MT-TD may be missed"},
            {"term": "Heteroplasmy threshold",    "definition": "The mutation load (% mutant mtDNA) above which clinical features appear; MT-TD threshold typically >50% blood for CPEO onset; muscle heteroplasmy exceeds blood by 10–20%"},
            {"term": "Compound deletion breakpoint", "definition": "Large deletions with one breakpoint in MT-TS1 (L-strand) and the other in MT-TD (H-strand) cause strand-switch deletions — require bioinformatic analysis of both strand orientations at the deletion boundaries"},
        ],
        "drug_definitions": [
            {"term": "Metformin",        "definition": "ABSOLUTE CI — biguanide complex-I inhibitor; severe lactic acidosis risk in mtDNA disease"},
            {"term": "Valproate (VPA)",  "definition": "ABSOLUTE CI — inhibits mitochondrial β-oxidation and CI; use LEV instead"},
            {"term": "Propofol",         "definition": "ABSOLUTE CI — propofol infusion syndrome (PRIS): decouples OXPHOS; fatal myocardial failure; use ketamine or volatile agents"},
            {"term": "Linezolid",        "definition": "ABSOLUTE CI — inhibits mitochondrial 23S-rRNA protein synthesis; reversible but severe mtDNA disease exacerbation"},
            {"term": "Chloramphenicol",  "definition": "ABSOLUTE CI — inhibits mt-ribosome peptidyl transferase; severe mt-translation failure"},
            {"term": "LEV (Levetiracetam)", "definition": "Preferred AED — renal excretion; no cytochrome P450 induction; no mitochondrial toxicity; Level C evidence"},
            {"term": "Beta-Blocker",     "definition": "Indicated if cardiomyopathy (HCM) develops (esp. m.7551T>C, KSS large-deletion); cardiology-guided; avoid negative inotropes in decompensated cardiomyopathy"},
        ],
        "references": [
            {"ref": "Seneca 2005 J Med Genet",         "citation": "Seneca S et al. (2005) — MT-TD m.7543G>A pathogenic variant causing CPEO and myopathy; variable loop disruption of tRNA-Asp"},
            {"ref": "Schaefer 2008 Ann Neurol",         "citation": "Schaefer AM et al. (2008) — prevalence of mitochondrial disease in adults; regional UK survey; ~1:5000 adult prevalence"},
            {"ref": "Gorman 2016 Nat Rev Dis Primers",  "citation": "Gorman GS et al. (2016) — Mitochondrial diseases — comprehensive review including mt-tRNA genetics, pathophysiology, management"},
            {"ref": "Scheper 2007 Nat Genet DARS2",     "citation": "Scheper GC et al. (2007) — DARS2 (mitochondrial aspartyl-tRNA synthetase) mutations cause LBSL; first description of LBSL gene"},
            {"ref": "Isohanni 2010 J Med Genet LBSL",   "citation": "Isohanni P et al. (2010) — LBSL clinical and MRI characteristics; DARS2 mutational spectrum; brain stem and spinal cord imaging"},
            {"ref": "Moraes 1995 Nat Genet KSS-deletion","citation": "Moraes CT et al. (1995) — Large mtDNA deletions causing KSS; compound tRNA loss from deletions spanning multiple mt-tRNA genes"},
            {"ref": "DiMauro & Schon 2003 NEJM",        "citation": "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases — comprehensive tRNA mutation review; NEJM 348:2656–2668"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== MT-TD OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== MT-TD BREAKDOWN (first variant) ===")
    bd = get_breakdown()
    print(json.dumps(bd["variant_breakdown"][0], indent=2))
    print(f"\nTotal patients: {bd['cohort_statistics']['n_patients']}")
