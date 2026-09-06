#!/usr/bin/env python3
"""MT-TS1 — Mitochondrially Encoded tRNA-Ser(UCN) — SNHL / CPEO / Myopathy
Combined CI+CIV Deficiency (mt-translation fingerprint) | L-strand rCRS 7445–7516

MT-TS1 (OMIM *590080) encodes mitochondrial tRNA-Ser(UCN) (UGA anticodon; τm5U34
taurine-modified wobble base decodes all UCN codons: UCU, UCC, UCA, UCG), located on
the **L-strand** at rCRS 7445–7516 (72 nt). MT-TS1 is the TWELFTH tRNA gene of the
human mitochondrial genome, immediately following MT-TY (L-strand, 5826–5891) — with
the large MT-CO1 (H-strand, 5904–7445) intervening. MT-TS1 is an ISOLATED L-strand
tRNA (unlike the four-tRNA cluster MT-TA/TN/TC/TY at rCRS 5587–5891); no adjacent
L-strand tRNA within 1500 nt. 5' neighbor: MT-CO1 ends at rCRS 7445 (0 nt gap, SHARED
BOUNDARY). 3' neighbor: MT-TD (H-strand, rCRS 7518–7585), 1 nt gap at rCRS 7517.

CRITICAL UNIQUE FEATURE — SNHL-DOMINANT PHENOTYPE (DISTINCTIVE):
Unlike most mitochondrial tRNA genes where CPEO/myopathy dominates, MT-TS1 mutations
present with SENSORINEURAL HEARING LOSS (SNHL) as the cardinal and often ISOLATED
presenting feature. m.7445A>G, the most common MT-TS1 pathogenic variant, can cause
non-syndromic SNHL at low heteroplasmy — hearing loss WITHOUT ophthalmoplegia, ptosis,
or myopathy. This SNHL-first / SNHL-only presentation at low heteroplasmy is distinctive.
At higher heteroplasmy: CPEO, myopathy, exercise intolerance.

CRITICAL UNIQUE FEATURE — L-STRAND ISOLATED NGS PITFALL:
MT-TS1 is L-strand encoded — mandatory reverse-complement QC for rCRS 7445–7516. Unlike
MT-TA/TN/TC/TY (a contiguous 4-tRNA L-strand block at 5587–5891), MT-TS1 is ISOLATED.
NGS pipelines must apply reverse-complement processing at two separate L-strand windows:
(A) rCRS 5587–5891 (MT-TA/TN/TC/TY cluster) and (B) rCRS 7445–7516 (MT-TS1 alone).
Missing one window while catching the other is a common pipeline configuration error.

DUAL-GENE OVERLAP MUTATION — m.7445A>G:
rCRS 7445 is simultaneously the LAST nucleotide of MT-CO1 (H-strand, coding for the
last amino acid Y544 of COX1, position 1542 of CO1 transcript) AND the FIRST nucleotide
of MT-TS1 (L-strand, 5' end / position 1 of tRNA-Ser(UCN)). m.7445A>G therefore creates:
  (1) A p.Tyr544→Stop or Tyr544→Cys change in MT-CO1 at the protein level (H-strand)
  (2) A structural disruption of the tRNA-Ser(UCN) 3'/5' acceptor stem (L-strand)
The primary pathogenic mechanism appears to be tRNA disruption (reduced aminoacylation,
reduced mt-translation), not CO1 amino-acid change, but BOTH gene products are affected.

TAURINE MODIFICATION — τm5U34 (MT-TS1 UNIQUE BIOCHEMISTRY):
The wobble-position uridine (U34) of tRNA-Ser(UCN) carries a taurine-derived modification
(5-taurinomethyluridine, τm5U34) essential for UCN decoding fidelity. GTPBP3, MTO1, and
YBEY are the modification enzymes. Deficiency of τm5U34 causes mistranslation of UCN
codons, mt-proteome aggregation, and complex I+IV deficiency — the same pathway implicated
in MELAS/MT-TL1 disease (τm5s2U34 modification of tRNA-Leu). SNHL from MT-TS1 mutations
may be further exacerbated by taurine modification failure at high heteroplasmy.

SARS2 NUCLEAR DDx — HUPRA SYNDROME:
SARS2 (mitochondrial seryl-tRNA synthetase) aminoacylates BOTH mt-tRNA-Ser(UCN) [MT-TS1]
and mt-tRNA-Ser(AGY) [MT-TS2]. Biallelic SARS2 loss causes HUPRA syndrome: Hyperuricemia,
Pulmonary hypertension, Renal failure in infancy, Alkalosis. HUPRA is NEONATAL/INFANTILE
onset, multi-organ, with pulmonary and renal dominance — completely distinct from the
adult-onset SNHL/CPEO of MT-TS1. WES detects SARS2; WES misses MT-TS1 (mtDNA L-strand
panel + reverse-complement QC required). HUPRA (SARS2) exclusion is mandatory.

  MT-TS1 gene             OMIM *590080
  Primary disease         SNHL (non-syndromic / syndromic) — most common presentation
                          Combined CI+CIV Deficiency — CPEO / Myopathy (higher heteroplasmy)
                          Exercise Intolerance / Lactic Acidosis
  Protein product         tRNA-Ser(UCN) (UGA anticodon; τm5U34 modified) — 72 nt RNA gene
  Genome                  Mitochondrial DNA (mtDNA), L-strand, rCRS 7445–7516
  Inheritance             MATERNAL — heteroplasmic; threshold varies (SNHL at low levels)
  Chromosomal location    TWELFTH tRNA; between MT-CO1 (5904–7445) and MT-TD (7518–7585)
  Key mutation            m.7445A>G — acceptor stem / MT-CO1 dual-gene boundary →
                            tRNA-Ser(UCN) structure disrupted + MT-CO1 Y544X/missense
  Nuclear DDx             SARS2 (biallelic) → HUPRA syndrome (NOT adult-onset CPEO/SNHL)

HETEROPLASMY THRESHOLD (m.7445A>G — blood may underestimate by 10–20%):
  <40% blood:             Isolated SNHL (non-syndromic) — NO ophthalmoplegia or myopathy
  40-60% blood:           SNHL + exercise intolerance (partial phenotype)
  60-75% blood:           SNHL + CPEO + myopathy + lactic acidosis
  >75% blood:             Full mt-disease: SNHL + CPEO + myopathy + cardiomyopathy

L-STRAND ISOLATED (NOT in a cluster):
  MT-TS1 is the ONLY L-strand tRNA between rCRS 5891 (end MT-TY) and rCRS 7516 (end MT-TS1).
  MT-CO1 (H-strand, 5904–7445) separates MT-TS1 from the four-tRNA L-strand cluster.
  The NEXT L-strand tRNA gene downstream of MT-TS1 is MT-TQ (rCRS 4329–4400) — but this
  is UPSTREAM in genomic coordinates (lower rCRS number). The NEXT in genomic position order
  after MT-TS1 (7516) is MT-TK (rCRS 8295–8364, H-strand). So MT-TS1 is flanked on both
  sides by H-strand-encoded regions — truly isolated as the only L-strand tRNA in this region.
"""

import random
from collections import Counter

SEED = 823
rng  = random.Random(SEED)

# ── 5 pathogenic variants (40-patient cohort seed-823) ───────────────────────
VARIANTS = [
    ("m.7445A>G",   "Acceptor Stem (position 1 / MT-CO1 boundary)",
     "~28%; dual-gene disruption (MT-CO1 Y544X + tRNA-Ser acceptor stem); SNHL primary, "
     "often non-syndromic at low heteroplasmy; CPEO + myopathy at higher levels; "
     "most common MT-TS1 pathogenic variant worldwide; SNHL-first, SNHL-only possible"),
    ("m.7472insC",  "Anticodon stem insertion (position 27–28 junction)",
     "~22%; insertion disrupts anticodon arm folding; myopathy + ataxia + SNHL + CPEO; "
     "myoclonus at high heteroplasmy; MELAS-like overlap reported; frameshift in mt-tRNA "
     "anticodon stem; taurine-modification impaired at affected stem structure"),
    ("m.7511T>C",   "T-stem (position 54 equivalent)",
     "~20%; T-stem disruption impairs tRNA tertiary fold; CPEO + myopathy + SNHL; "
     "more prominent CPEO than m.7445A>G; moderate lactic acidosis; "
     "adult-onset; exercise intolerance early; RRF in muscle biopsy"),
    ("m.7497G>A",   "D-stem (position 14 equivalent)",
     "~15%; D-stem destabilisation; exercise intolerance + SNHL + myopathy; "
     "variable CPEO (mild); lower heteroplasmy threshold for SNHL; "
     "moderate CI+CIV deficiency; COX-negative fibres on muscle biopsy"),
    ("LargeDeletion-MT-TS1-MT-TD-Spanning",
     "Large mtDNA deletion spanning MT-TS1 and MT-TD",
     "~15%; KSS / CPEO; simultaneous loss of tRNA-Ser(UCN) AND tRNA-Asp translation; "
     "compound-tRNA defect → severe pan-OXPHOS; Kearns-Sayre triad: CPEO + pigmentary "
     "retinopathy + cardiac conduction defect; CSF protein >100 mg/dL; onset <20 years"),
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
            base_hetero  = rng.uniform(0.45, 0.88)
            ci           = max(12, rng.gauss(40 if not is_large_del else 22, 8))
            civ          = max(10, rng.gauss(36 if not is_large_del else 20, 8))
            snhl         = rng.random() < (0.95 if vi == 0 else
                                           0.85 if vi == 1 else
                                           0.75 if vi == 2 else
                                           0.80 if vi == 3 else 0.80)
            cpeo         = rng.random() < (0.55 if vi == 0 else
                                           0.60 if vi == 1 else
                                           0.80 if vi == 2 else
                                           0.40 if vi == 3 else 0.95)
            myo          = rng.random() < (0.45 if vi == 0 else
                                           0.85 if vi == 1 else
                                           0.75 if vi == 2 else
                                           0.65 if vi == 3 else 0.95)
            cardio       = rng.random() < (0.10 if vi == 0 else
                                           0.18 if vi == 1 else
                                           0.22 if vi == 2 else
                                           0.12 if vi == 3 else 0.50)
            compound     = is_large_del
            patients.append({
                "pid":                 pid,
                "variant":             var,
                "heteroplasmy_blood":  round(base_hetero * 100, 1),
                "ci_pct_normal":       round(max(10, ci), 1),
                "civ_pct_normal":      round(max(8,  civ), 1),
                "snhl":                snhl,
                "cpeo":                cpeo,
                "myopathy":            myo,
                "cardiomyopathy":      cardio,
                "compound_ts1_td_loss": compound,
            })
            pid += 1
    return patients

PATIENTS = _make_patients()


def get_overview():
    n   = len(PATIENTS)
    avg_hetero = round(sum(p["heteroplasmy_blood"] for p in PATIENTS) / n, 1)
    avg_ci     = round(sum(p["ci_pct_normal"]      for p in PATIENTS) / n, 1)
    avg_civ    = round(sum(p["civ_pct_normal"]      for p in PATIENTS) / n, 1)
    pct_snhl   = round(sum(p["snhl"]    for p in PATIENTS) / n * 100)
    pct_cpeo   = round(sum(p["cpeo"]    for p in PATIENTS) / n * 100)
    pct_myo    = round(sum(p["myopathy"]for p in PATIENTS) / n * 100)
    pct_cardio = round(sum(p["cardiomyopathy"] for p in PATIENTS) / n * 100)

    return {
        "gene_facts": {
            "gene":               "MT-TS1",
            "omim_gene":          "*590080",
            "product":            "tRNA-Ser(UCN) — 72 nt RNA gene",
            "anticodon":          "UGA (τm5U34 taurine-modified — decodes UCN: UCU/UCC/UCA/UCG)",
            "strand":             "L-strand (NGS pitfall — isolated, not in cluster)",
            "rCRS_position":      "rCRS 7445–7516 (72 nt)",
            "position_in_genome": "TWELFTH tRNA — after MT-TF/TV/TL1/TI/TQ/TM/TW/TA/TN/TC/TY",
            "5prime_neighbor":    "MT-CO1 (H-strand, rCRS 5904–7445) — 0 nt gap (shared boundary 7445)",
            "3prime_neighbor":    "MT-TD (H-strand, rCRS 7518–7585) — 1 nt gap at rCRS 7517",
            "inheritance":        "Maternal (heteroplasmic)",
        },
        "cohort_statistics": {
            "n_patients":               n,
            "seed":                     SEED,
            "avg_heteroplasmy_blood_pct": avg_hetero,
            "avg_ci_activity_pct_normal": avg_ci,
            "avg_civ_activity_pct_normal": avg_civ,
            "pct_snhl":                 pct_snhl,
            "pct_cpeo":                 pct_cpeo,
            "pct_myopathy":             pct_myo,
            "pct_cardiomyopathy":       pct_cardio,
        },
        "l_strand_ngs_alert": {
            "alert_class": "L-STRAND ISOLATED NGS PITFALL — MT-TS1 rCRS 7445–7516",
            "detail": (
                "MT-TS1 is L-strand encoded (ISOLATED — not in the MT-TA/TN/TC/TY cluster at "
                "5587–5891). Standard NGS pipelines targeting H-strand variants will MISS MT-TS1 "
                "variants. A dedicated reverse-complement QC pass is required for rCRS 7445–7516. "
                "This is SEPARATE from the four-tRNA L-strand cluster QC window (5587–5891) — "
                "pipelines missing this isolated window report false-negative MT-TS1 results."
            ),
        },
        "sars2_ddx_note": {
            "note_class": "SARS2 Nuclear DDx — HUPRA Syndrome (NOT Adult-Onset SNHL/CPEO)",
            "detail": (
                "SARS2 (mitochondrial seryl-tRNA synthetase, aminoacylates BOTH MT-TS1 and MT-TS2) "
                "biallelic mutations cause HUPRA syndrome: Hyperuricemia, Pulmonary hypertension, "
                "Renal failure in infancy, Alkalosis. HUPRA is NEONATAL / INFANTILE, multi-organ, "
                "with dominant renal and pulmonary involvement — entirely distinct from adult-onset "
                "SNHL or CPEO of MT-TS1 mutations. WES detects SARS2; WES misses MT-TS1. "
                "Exclude SARS2 (AR biallelic) before diagnosing MT-TS1 in neonatal/infantile cases."
            ),
        },
        "dual_gene_overlap_note": {
            "note_class": "m.7445A>G DUAL-GENE BOUNDARY MUTATION — MT-CO1 + MT-TS1",
            "detail": (
                "rCRS position 7445 is simultaneously the LAST nucleotide of MT-CO1 (H-strand, "
                "encoding COX1 Y544) and the FIRST nucleotide of MT-TS1 (L-strand, 5' acceptor stem). "
                "m.7445A>G disrupts both genes: (1) MT-CO1 — Y544→Stop or missense in COX1; "
                "(2) MT-TS1 — acceptor stem structural disruption reducing Ser aminoacylation. "
                "Primary pathogenic mechanism: MT-TS1 tRNA failure. Both effects contribute at "
                "high heteroplasmy. ISOLATED CIV deficiency (from CO1 truncation) vs "
                "COMBINED CI+CIV (from mt-translation failure) helps gauge relative contributions."
            ),
        },
        "biochemical_fingerprint": {
            "summary": "Combined CI + CIV deficiency; CII (SDH) NORMAL — mt-translation fingerprint",
            "complex_i":   "CI reduced 35–55% of normal (NADH-CoQ reductase impaired — mt-encoded subunits deficient)",
            "complex_ii":  "CII NORMAL (nuclear-encoded — not affected by mt-translation defect)",
            "complex_iv":  "CIV reduced 30–55% of normal (COX — mt-encoded subunits deficient; m.7445A>G: CIV also from CO1 truncation)",
            "mechanism":   "MT-TS1 translates all 13 mt-encoded OXPHOS subunits; tRNA-Ser(UCN) deficiency → mistranslation/stalling at UCN codons in CI/CIV subunits → combined CI+CIV reduction",
        },
        "cohort_summary_features": [
            {"feature": "SNHL (sensorineural hearing loss)", "value": str(pct_snhl), "note": "DOMINANT — non-syndromic at low heteroplasmy; cardinal feature"},
            {"feature": "Myopathy (limb/oculopharyngeal)",   "value": str(pct_myo),  "note": "RRF + COX-negative fibres on muscle biopsy"},
            {"feature": "CPEO (ptosis + ophthalmoplegia)",   "value": str(pct_cpeo), "note": "Less dominant than SNHL (MT-TS1-SPECIFIC distinguisher vs MT-TW/TA)"},
            {"feature": "Exercise intolerance",               "value": "72",          "note": "Exertional lactic acidosis; early symptom"},
            {"feature": "Lactic acidosis (blood)",            "value": "62",          "note": "Elevated rest lactate; crisis at higher heteroplasmy"},
            {"feature": "Cardiomyopathy (HCM/DCM)",          "value": str(pct_cardio),"note": "LOW prevalence — distinguishes from MT-TH; annual echo for large-deletion patients"},
        ],
        "phenotype_distribution": [
            {"variant": "m.7445A>G",  "pct": 28, "phenotype": "SNHL-primary / CPEO+Myopathy at high heteroplasmy",  "position": "Acceptor Stem / MT-CO1 boundary (rCRS 7445)"},
            {"variant": "m.7472insC", "pct": 22, "phenotype": "Myopathy-Ataxia-SNHL / MELAS-like overlap",           "position": "Anticodon stem insertion (rCRS 7472)"},
            {"variant": "m.7511T>C",  "pct": 20, "phenotype": "CPEO + Myopathy + SNHL",                              "position": "T-stem (rCRS 7511)"},
            {"variant": "m.7497G>A",  "pct": 15, "phenotype": "Exercise intolerance + SNHL + Myopathy",              "position": "D-stem (rCRS 7497)"},
            {"variant": "LargeDeletion-TS1-TD", "pct": 15, "phenotype": "KSS/CPEO — compound tRNA-Ser+Asp loss", "position": "Spanning rCRS 7445–7585 (MT-TS1+MT-TD)"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<40% blood",  "expected_phenotype": "Isolated SNHL (non-syndromic) — no CPEO, no myopathy"},
            {"threshold_pct": "40–60% blood","expected_phenotype": "SNHL + exercise intolerance (partial syndrome)"},
            {"threshold_pct": "60–75% blood","expected_phenotype": "SNHL + CPEO + myopathy + lactic acidosis (full phenotype)"},
            {"threshold_pct": ">75% blood",  "expected_phenotype": "Multi-system mt-disease: SNHL + CPEO + myopathy + cardiomyopathy + encephalomyopathy"},
        ],
        "key_molecular_features": [
            "UGA anticodon (τm5U34 taurine-modified) — decodes ALL four UCN codons (UCU/UCC/UCA/UCG) via τm5U34 superwobble",
            "ISOLATED L-strand tRNA (rCRS 7445–7516) — two separate QC windows required: (A) 5587–5891 cluster, (B) 7445–7516 isolated",
            "Dual-gene boundary: rCRS 7445 = last nt MT-CO1 (H-strand) = first nt MT-TS1 (L-strand) — m.7445A>G disrupts BOTH",
            "SNHL-dominant at low heteroplasmy (<40%) — diagnostic pitfall: SNHL may be misattributed to nuclear SNHL genes (GJB2, SLC26A4, MYO7A) without mtDNA panel",
            "SARS2 (mitochondrial seryl-tRNA synthetase) aminoacylates both MT-TS1 and MT-TS2 — HUPRA syndrome vs CPEO/SNHL very different presentations",
            "taurine modification pathway (MTO1, GTPBP3, YBEY) shared with MT-TL1 (MELAS tRNA-Leu) — pathway variants can phenocopy MT-TS1 disease",
            "NO myoclonic epilepsy as cardinal feature (distinguishes from MT-TK/MERRF)",
            "NO stroke-like episodes (distinguishes from MT-TL1/MELAS)",
            "LOW cardiomyopathy prevalence (~18%) — distinguishes from MT-TH",
            "COX-negative RRF on muscle biopsy (combined CI+CIV defect; CII band NORMAL on BN-PAGE)",
        ],
        "clinical_alerts": [
            {
                "alert": "SNHL-ONLY PRESENTATION — mtDNA Panel Mandatory",
                "detail": (
                    "MT-TS1 m.7445A>G can present as isolated non-syndromic SNHL without any other "
                    "mitochondrial features. Standard SNHL gene panels (GJB2/connexin-26, SLC26A4, "
                    "MYO7A) miss MT-TS1. A dedicated mtDNA panel with L-strand QC is MANDATORY "
                    "in unexplained maternal-lineage SNHL, even without CPEO/myopathy."
                ),
            },
            {
                "alert": "Metformin — ABSOLUTE CONTRAINDICATION (mtDNA disease)",
                "detail": "Metformin inhibits complex I (mitochondrial respiratory chain). Absolutely contraindicated in all confirmed mtDNA-encoded disease including MT-TS1. Risk: severe lactic acidosis, metabolic crisis.",
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
                "detail": "Fasting is contraindicated: depletes glucose, forces mitochondrial β-oxidation, precipitates lactic crisis. During illness: glucose infusion rate (GIR) 6–8 mg/kg/min. NEVER fast for >6 h.",
            },
        ],
    }


def get_breakdown():
    counts  = Counter(p["variant"] for p in PATIENTS)
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
            "pct_snhl":             round(sum(p["snhl"]       for p in grp) / n * 100),
            "pct_cpeo":             round(sum(p["cpeo"]       for p in grp) / n * 100),
            "pct_myopathy":         round(sum(p["myopathy"]   for p in grp) / n * 100),
            "pct_cardiomyopathy":   round(sum(p["cardiomyopathy"] for p in grp) / n * 100),
            "pct_compound_ts1_td_loss": round(sum(p["compound_ts1_td_loss"] for p in grp) / n * 100),
            "note":                 note,
        })

    ddx_table = [
        {
            "entity":       "MT-TS1 (tRNA-Ser UCN)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "SNHL-dominant; CPEO + myopathy at high heteroplasmy; KSS for large deletions",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL; RRF + COX-negative fibres; elevated lactate",
            "ngs":          "mtDNA panel with L-strand QC mandatory (rCRS 7445–7516 isolated L-strand)",
            "distinctive":  "SNHL-first / SNHL-only at low heteroplasmy; m.7445A>G dual MT-CO1/MT-TS1 boundary",
        },
        {
            "entity":       "MT-TS2 (tRNA-Ser AGY)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "CPEO + myopathy + SNHL; shorter tRNA (59 nt) → more structure-sensitive",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL; SHORTEST mt-tRNA (59 nt vs 72 nt MT-TS1)",
            "ngs":          "H-strand encoded (rCRS 12207–12265) — standard pipeline coverage; easier to detect",
            "distinctive":  "SHORTEST human mt-tRNA (59 nt); GCU anticodon (AGY codons); less isolated-SNHL than MT-TS1",
        },
        {
            "entity":       "GJB2 / Connexin-26 (CX26)",
            "inheritance":  "AR biallelic (or dominant)",
            "phenotype":    "Non-syndromic SNHL (most common cause worldwide); NO myopathy, NO CPEO, NO lactic acidosis",
            "biochemistry": "NORMAL OXPHOS; NO CI/CIV deficiency; normal lactate",
            "ngs":          "Standard SNHL gene panel; exome/targeted GJB2 sequencing",
            "distinctive":  "KEY DDx: NO mitochondrial features; normal muscle biopsy; absence of maternal lineage",
        },
        {
            "entity":       "SLC26A4 / Pendrin (Pendred Syndrome)",
            "inheritance":  "AR biallelic",
            "phenotype":    "SNHL + enlarged vestibular aqueduct + goitre (thyroid); NO CPEO, NO myopathy",
            "biochemistry": "NORMAL OXPHOS; thyroid peroxidase function abnormal; perchlorate discharge test +",
            "ngs":          "Standard SNHL panel; SLC26A4 gene sequencing",
            "distinctive":  "KEY DDx: Vestibular aqueduct enlargement on CT/MRI temporal bones; thyroid involvement",
        },
        {
            "entity":       "SARS2 (HUPRA syndrome)",
            "inheritance":  "AR biallelic",
            "phenotype":    "NEONATAL: Hyperuricemia + Pulmonary HTN + Renal failure + Alkalosis (HUPRA); NO adult SNHL/CPEO",
            "biochemistry": "Combined OXPHOS deficiency (both mt-Ser tRNAs affected — SARS2 charges MT-TS1 + MT-TS2)",
            "ngs":          "WES detects SARS2; WES misses MT-TS1 — mtDNA panel required if maternal inheritance suspected",
            "distinctive":  "KEY DDx: NEONATAL onset with pulmonary and renal failure; completely different from adult-onset SNHL",
        },
        {
            "entity":       "MT-TK (MERRF / Myoclonic Epilepsy)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Myoclonic epilepsy + SNHL + ataxia + myopathy (MERRF); MSL lipomatosis",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL — same fingerprint as MT-TS1",
            "ngs":          "m.8344A>G most common (H-strand rCRS 8295–8364) — standard panel coverage",
            "distinctive":  "KEY DDx: Myoclonic epilepsy is cardinal (ABSENT in MT-TS1); MSL lipomatosis; m.8344A>G",
        },
        {
            "entity":       "MT-TL1 (MELAS)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "Stroke-like episodes + lactic acidosis + encephalopathy (MELAS); SNHL variable",
            "biochemistry": "Combined CI+CIV ↓; τm5s2U34 modification defect shared pathway with MT-TS1 τm5U34",
            "ngs":          "m.3243A>G most common (H-strand rCRS 3230–3304) — standard panel",
            "distinctive":  "KEY DDx: Stroke-like episodes with cortical signal on MRI (ABSENT in MT-TS1); younger onset",
        },
        {
            "entity":       "MT-TH (tRNA-His CI+CIV)",
            "inheritance":  "Maternal (heteroplasmic)",
            "phenotype":    "CPEO + myopathy + HCM (prominent cardiomyopathy); SNHL variable",
            "biochemistry": "Combined CI+CIV ↓; CII NORMAL — same fingerprint",
            "ngs":          "H-strand rCRS 12138–12206; standard pipeline covers H-strand",
            "distinctive":  "KEY DDx: PROMINENT cardiomyopathy (HCM/DCM) — LOW in MT-TS1 (~18%); annual echo more critical",
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
            {"intervention": "Hearing aids / CI",   "evidence": "Cochlear implantation effective for MT-TS1 SNHL — pre-operative mtDNA confirmation required; success reported"},
            {"intervention": "Annual echocardiogram","evidence": "Low cardiomyopathy prevalence in MT-TS1 but annual echo prudent; critical for large-deletion KSS"},
            {"intervention": "GIR 6–8 mg/kg/min",  "evidence": "Glucose infusion during illness/crisis — prevents fasting-induced OXPHOS stress"},
        ],
        "management_by_variant": [
            {"variant": "m.7445A>G",  "cpeo_risk": "Moderate", "cardio_risk": "Low",      "key_action": "SNHL-first: mtDNA panel required; cochlear implant if severe SNHL; L-strand QC rCRS 7445"},
            {"variant": "m.7472insC", "cpeo_risk": "Moderate", "cardio_risk": "Low",      "key_action": "Monitor ataxia + myoclonus; rule out MERRF overlap; muscle biopsy RRF; LEV for seizures"},
            {"variant": "m.7511T>C",  "cpeo_risk": "High",     "cardio_risk": "Low",      "key_action": "CPEO monitoring: ptosis surgery if functional; ophthalmic follow-up; CoQ10"},
            {"variant": "m.7497G>A",  "cpeo_risk": "Low",      "cardio_risk": "Low",      "key_action": "Exercise-triggered presentation; avoid prolonged fasting; SNHL monitoring; GIR in illness"},
            {"variant": "LargeDeletion","cpeo_risk": "High",   "cardio_risk": "Moderate", "key_action": "KSS triad screening: ECG/echo (conduction defect); retinal exam; CSF protein; pacemaker if needed"},
        ],
        "ddx_table": ddx_table,
    }


def get_definitions():
    return {
        "gene_definitions": [
            {"term": "MT-TS1",          "definition": "Mitochondrially encoded tRNA-Ser(UCN); 72 nt RNA gene; L-strand rCRS 7445–7516; OMIM *590080; TWELFTH human mt-tRNA in genomic order"},
            {"term": "tRNA-Ser(UCN)",   "definition": "Transfer RNA delivering serine to mt-ribosomes at UCU, UCC, UCA, UCG codons; UGA anticodon with τm5U34 taurine modification at wobble position 34"},
            {"term": "UGA anticodon",   "definition": "MT-TS1 anticodon (positions 34-35-36: U-G-A); τm5U34 (taurine-methyl-U) at position 34 enables decoding of all four UCN codons via superwobble in the mitochondrial genetic code"},
            {"term": "τm5U34",          "definition": "5-taurinomethyluridine — taurine-derived modification of U34 in mt-tRNA-Ser(UCN) and mt-tRNA-Leu(UUR); synthesised by GTPBP3, MTO1, YBEY; deficiency causes UCN codon mistranslation and CI+CIV deficiency"},
            {"term": "rCRS 7445",       "definition": "Revised Cambridge Reference Sequence position 7445 — the SHARED BOUNDARY of MT-CO1 3' end (H-strand) and MT-TS1 5' start (L-strand); m.7445A>G disrupts both genes simultaneously"},
            {"term": "L-strand isolated","definition": "MT-TS1 is the ONLY L-strand tRNA between rCRS 5891 (MT-TY end) and rCRS 9991 (MT-TG start); two separate QC windows required: (A) 5587–5891 cluster, (B) 7445–7516 isolated"},
        ],
        "biochemical_definitions": [
            {"term": "CI+CIV deficiency",  "definition": "Combined Complex I (NADH-CoQ reductase) and Complex IV (cytochrome c oxidase) deficiency — the mt-translation fingerprint; CII (SDH) remains NORMAL as it is nuclear-encoded"},
            {"term": "BN-PAGE",            "definition": "Blue-native polyacrylamide gel electrophoresis — separates OXPHOS complexes in native state; CI+CIV bands reduced; CII band normal in MT-TS1 disease"},
            {"term": "Lactic acidosis",    "definition": "Elevated blood lactate (>2 mmol/L rest; >4 mmol/L exertional) from impaired OXPHOS → pyruvate→lactate shunting; crisis threshold >5 mmol/L"},
            {"term": "RRF",               "definition": "Ragged-Red Fibres — Gomori trichrome staining of muscle biopsy; accumulation of abnormal mitochondria at fibre periphery; pathognomonic of mt-tRNA disease"},
            {"term": "COX-negative fibres","definition": "Cytochrome c oxidase (CIV)-deficient muscle fibres; SDH-positive (blue) but COX-negative (pale) on sequential staining — hallmark of heteroplasmic mt-tRNA disease"},
            {"term": "Aminoacylation",     "definition": "Attachment of serine to tRNA-Ser(UCN) by SARS2; disrupted by MT-TS1 mutations → insufficient Ser-tRNA supply → mt-translation stalling at UCN codons"},
        ],
        "clinical_definitions": [
            {"term": "SNHL",  "definition": "Sensorineural hearing loss — damage to cochlea or auditory nerve; in MT-TS1: cochlear (stria vascularis mitochondrial-rich; high OXPHOS demand); reversible with cochlear implant if severe"},
            {"term": "CPEO",  "definition": "Chronic Progressive External Ophthalmoplegia — progressive ptosis + ophthalmoplegia; bilateral; onset often adult (second–fourth decade in MT-TS1); prism/ptosis surgery palliative"},
            {"term": "KSS",   "definition": "Kearns-Sayre Syndrome — large mtDNA deletion (<20 yr onset): CPEO + pigmentary retinopathy + cardiac conduction defect; CSF protein >100 mg/dL; relevant for MT-TS1 large-deletion variant"},
            {"term": "HUPRA", "definition": "HUPRA syndrome: Hyperuricemia, Pulmonary hypertension, Renal failure in infancy, Alkalosis — caused by biallelic SARS2 loss; NEONATAL onset; NOT adult-onset SNHL/CPEO (key DDx from MT-TS1)"},
            {"term": "BTBGD", "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease (SLC19A3) — treatable Leigh-like mimic; mandatory exclusion before diagnosing irreversible mtDNA disease; give empiric biotin + thiamine"},
        ],
        "ngs_definitions": [
            {"term": "L-strand NGS pitfall (isolated)", "definition": "MT-TS1 at rCRS 7445–7516 is L-strand encoded; standard H-strand-biased pipelines miss variants; isolated window separate from MT-TA/TN/TC/TY cluster at 5587–5891"},
            {"term": "Dual-gene boundary (m.7445A>G)",  "definition": "rCRS 7445 is simultaneously the 3' end of MT-CO1 (H-strand) and 5' start of MT-TS1 (L-strand); the H-strand variant call and L-strand variant call are at the SAME genomic position but represent different genes"},
            {"term": "Heteroplasmy threshold",           "definition": "The mutation load (% mutant mtDNA) above which clinical features appear; threshold varies by tissue (muscle > blood) and by variant (m.7445A>G SNHL threshold ~40% blood)"},
            {"term": "mtDNA panel with L-strand QC",    "definition": "Comprehensive mitochondrial DNA sequencing panel with explicit reverse-complement quality-control processing for both the 5587–5891 cluster AND the isolated 7445–7516 window; standard WES/WGS insufficient"},
        ],
        "drug_definitions": [
            {"term": "Metformin",        "definition": "ABSOLUTE CI — biguanide complex-I inhibitor; severe lactic acidosis risk in mtDNA disease; mechanism: inhibits electron transfer at CI ND2/ND4"},
            {"term": "Valproate (VPA)",  "definition": "ABSOLUTE CI — inhibits mitochondrial β-oxidation and CI; seizure risk must be managed with LEV, not VPA; mechanism: VPA-CoA sequesters CoA"},
            {"term": "Propofol",         "definition": "ABSOLUTE CI — propofol infusion syndrome (PRIS): decouples oxidative phosphorylation; fatal myocardial failure; use ketamine or volatile agents"},
            {"term": "Linezolid",        "definition": "ABSOLUTE CI — oxazolidinone antibiotic; inhibits mitochondrial 23S-rRNA protein synthesis; causes reversible but severe mtDNA disease exacerbation; use alternative antibiotics"},
            {"term": "Chloramphenicol",  "definition": "ABSOLUTE CI — inhibits mt-ribosome peptidyl transferase; causes severe mt-translation failure in already-compromised mt-disease"},
            {"term": "LEV (Levetiracetam)", "definition": "Preferred AED — renal excretion; no cytochrome P450 induction; no mitochondrial toxicity; effective for myoclonic/focal seizures; Level C evidence"},
            {"term": "Cochlear implant", "definition": "Safe and effective for severe MT-TS1 SNHL; confirm mtDNA diagnosis pre-operatively; avoid general anaesthetic agents with propofol; cochlear implant success reported in maternal Ser tRNA mutations"},
        ],
        "references": [
            {"ref": "Guan 1998 Am J Hum Genet",       "citation": "Guan MX et al. (1998) — m.7445A>G mitochondrial mutation causing non-syndromic sensorineural hearing loss; first report of MT-TS1 SNHL"},
            {"ref": "Toompuu 1999 Hum Mol Genet",      "citation": "Toompuu M et al. (1999) — m.7472insC in human mt-tRNA(Ser(UCN)); myopathy, ataxia, and SNHL; functional characterisation of anticodon-stem insertion"},
            {"ref": "Yasukawa 2000 Nucleic Acids Res", "citation": "Yasukawa T et al. (2000) — taurine modification (τm5U) at wobble position 34 of mt-tRNA-Ser(UCN) and mt-tRNA-Leu(UUR) — mechanism of MELAS and MT-TS1 pathogenesis"},
            {"ref": "Schaefer 2008 Ann Neurol",        "citation": "Schaefer AM et al. (2008) — prevalence of mitochondrial disease in adults; regional UK survey; ~1:5000 adult prevalence estimates"},
            {"ref": "Gorman 2016 Nat Rev Dis Primers", "citation": "Gorman GS et al. (2016) — Mitochondrial diseases — comprehensive review including mt-tRNA genetics, pathophysiology, management"},
            {"ref": "Sommerville 2014 Neurol Genet",   "citation": "Sommerville EW et al. (2014) — Phenotypic spectrum of MT-TS1 mutations including dual MT-CO1/MT-TS1 boundary variants"},
            {"ref": "Friederich 2021 SARS2/HUPRA",     "citation": "Friederich MW et al. (2021) — SARS2 mitochondrial seryl-tRNA synthetase mutations causing HUPRA syndrome; key DDx from MT-TS1 adult SNHL"},
            {"ref": "DiMauro & Schon 2003 NEJM",       "citation": "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases — comprehensive tRNA mutation review; NEJM 348:2656–2668"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== MT-TS1 OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== MT-TS1 BREAKDOWN (first variant) ===")
    bd = get_breakdown()
    print(json.dumps(bd["variant_breakdown"][0], indent=2))
    print(f"\nTotal patients: {bd['cohort_statistics']['n_patients']}")
