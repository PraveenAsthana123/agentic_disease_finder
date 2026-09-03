#!/usr/bin/env python3
"""MT-TN — Mitochondrially Encoded tRNA-Asn — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | L-strand rCRS 5657–5729

MT-TN (OMIM *590010) encodes mitochondrial tRNA-Asn (GTT anticodon), located on the
**L-strand** at rCRS 5657–5729 (73 nt). MT-TN is the NINTH tRNA gene of the human
mitochondrial genome, following MT-TF (577–647), MT-TV (1602–1670), MT-TL1 (3230–3304),
MT-TI (4263–4331), MT-TQ (L-strand 4329–4400), MT-TM (4402–4469), MT-TW (5512–5579),
and MT-TA (L-strand 5587–5655). MT-TN is the SECOND consecutive L-strand tRNA in the
cluster: MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), MT-TY (5826–5891).

CRITICAL UNIQUE FEATURE — L-STRAND NGS PITFALL:
MT-TN is encoded on the L-strand (light/complementary strand). Standard NGS pipelines
optimised for H-strand (heavy/reference-strand) variant calling will MISS or MIS-CALL
MT-TN variants. The gene is read in the reverse-complement direction relative to the rCRS
reference sequence. This L-strand pitfall is shared by all four tRNAs in this cluster:
MT-TA, MT-TN, MT-TC, MT-TY — requiring mandatory reverse-complement QC processing for
the entire rCRS 5587–5891 region. MT-TN itself occupies rCRS 5657–5729 (2 nt gap from
MT-TA; 32 nt gap before MT-TC).

L-STRAND CLUSTER — SECOND OF FOUR:
MT-TN is the second (immediately following MT-TA) of the four consecutive L-strand tRNAs.
Immediately 5' is MT-TA (L-strand, 5587–5655) with a 2 nt gap at rCRS 5655–5657.
Immediately 3' is MT-TC (L-strand, 5761–5826) with a 32 nt gap at rCRS 5729–5761.
Large mtDNA deletions spanning the TA/TN junction may simultaneously remove MT-TA AND
MT-TN, producing compound tRNA-Ala + tRNA-Asn loss with compounded CI+CIV deficiency.

GTT ANTICODON — ASN CODON DECODING:
MT-TN has a GTT anticodon, reading AAC (Asn) codons by Watson-Crick pairing and AAU by
wobble (G:U at position 34). In human mitochondria, the two Asn codons (AAU, AAC) are
both decoded exclusively by MT-TN. Asn (AAC/AAU) residues occur throughout the hydrophilic
loops and matrix-facing domains of OXPHOS CI and CIV subunits. MT-TN mutations reduce
tRNA-Asn availability → impaired elongation at AAC/AAU codons → CI+CIV assembly defect.

NUCLEAR DDx — NARS2 (mt-Asparaginyl-tRNA Synthetase 2):
NARS2 biallelic mutations cause mitochondrial Asn-tRNA aminoacylation failure — same
biochemical level as MT-TN, but with a dramatically different clinical phenotype:
Perrault syndrome type 6 (OMIM #617717): sensorineural hearing loss + premature ovarian
failure in females (AR); or early-onset epileptic encephalopathy with regression. NARS2 is
AR, WES-detectable, and clinically distinguishable from MT-TN disease by: (1) SNHL + POF
in females (Perrault phenotype, NOT adult CPEO); (2) infantile epileptic encephalopathy
phenotype; (3) absence of maternal inheritance; (4) WES detection. MT-TN (maternal
heteroplasmic CPEO) does NOT cause Perrault syndrome or infantile encephalopathy.

  MT-TN gene              OMIM *590010
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Occasional Cardiomyopathy / SNHL / Leigh-like at high heteroplasmy
                          L-strand NGS pitfall: mandatory reverse-complement QC rCRS 5657–5729
  Protein product         tRNA-Asn (GTT anticodon) — 73 nucleotides; RNA gene
                          Decodes AAC (Watson-Crick) and AAU (wobble) — both Asn codons in mt code
  Genome                  Mitochondrial DNA (mtDNA), **L-strand**, rCRS 5657–5729
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    NINTH tRNA in mitochondrial genome
                          5′-adjacent to MT-TA (L-strand, 5587–5655), 2 nt gap
                          3′-adjacent to MT-TC (L-strand, 5761–5826), 32 nt gap
                          SECOND OF FOUR consecutive L-strand tRNAs (MT-TA/TN/TC/TY cluster)
"""

import random
import math

# ── Seed for reproducible 40-patient cohort ──────────────────────────────────
SEED = 817
RNG  = random.Random(SEED)


def _rand_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    v = RNG.gauss(mu, sigma)
    return round(max(lo, min(hi, v)), 1)


def _rand_int(lo: int, hi: int) -> int:
    return RNG.randint(lo, hi)


# ── 40-patient cohort generator ───────────────────────────────────────────────
def _build_cohort():
    """Generate deterministic 40-patient MT-TN cohort (seed-817)."""
    N = 40
    patients = []

    # Variant distribution (must sum to 40)
    variant_pool = (
        ["m.5690AG"]  * 11 +   # anticodon stem — CPEO + myopathy           ~28%
        ["m.5728AG"]  * 9  +   # acceptor stem  — CPEO + cardio              ~22%
        ["m.5703GA"]  * 8  +   # variable loop  — multisystem / Leigh-like   ~19% (≈20%)
        ["m.5692AG"]  * 6  +   # anticodon loop — exercise + SNHL            ~16% (≈15%)
        ["LargeDel"]  * 6      # TA-TN spanning — KSS/CPEO compound loss     ~15%
    )
    RNG.shuffle(variant_pool)

    for i in range(N):
        variant = variant_pool[i]

        # Heteroplasmy by variant
        if variant == "m.5690AG":
            hetero = _rand_normal(54, 15, 24, 87)
        elif variant == "m.5728AG":
            hetero = _rand_normal(57, 16, 22, 89)
        elif variant == "m.5703GA":
            hetero = _rand_normal(62, 16, 30, 91)
        elif variant == "m.5692AG":
            hetero = _rand_normal(49, 17, 20, 85)
        else:  # LargeDel
            hetero = _rand_normal(59, 17, 25, 90)

        # Enzyme activities — CI+CIV deficiency, CII NORMAL
        # Asn (AAC/AAU) in hydrophilic loops / matrix-domain of CI and CIV subunits
        ci_act  = _rand_normal(37 + (100 - hetero) * 0.31, 8,  9, 67)
        civ_act = _rand_normal(34 + (100 - hetero) * 0.29, 9,  7, 65)
        cii_act = _rand_normal(94, 4, 82, 104)  # CII nuclear-encoded → normal

        # Clinical features
        cpeo     = hetero > 35 or variant in ("m.5690AG", "m.5728AG", "LargeDel")
        myopathy = hetero > 32 or variant in ("m.5690AG", "m.5703GA")
        cardio   = variant == "m.5728AG" and hetero > 50 and RNG.random() < 0.38
        snhl     = variant in ("m.5692AG",) and RNG.random() < 0.50
        rrfs     = myopathy and RNG.random() < 0.74
        lactic   = hetero > 45 and RNG.random() < 0.62
        leigh_mri = (variant == "m.5703GA" or hetero > 72) and RNG.random() < 0.33
        exercise = hetero > 23 and RNG.random() < 0.83
        # Compound loss: LargeDel spanning TA-TN may also eliminate MT-TA
        compound_ta_loss = variant == "LargeDel" and RNG.random() < 0.60

        # Onset age
        if variant == "m.5703GA":
            onset = _rand_normal(15, 9, 5, 38)    # multisystem — younger
        elif variant == "LargeDel":
            onset = _rand_normal(18, 10, 6, 41)
        elif variant == "m.5692AG":
            onset = _rand_normal(23, 12, 8, 47)
        else:
            onset = _rand_normal(33, 13, 15, 62)

        patients.append({
            "id": i + 1,
            "variant": variant,
            "heteroplasmy_blood_pct": hetero,
            "ci_activity_pct_normal": ci_act,
            "civ_activity_pct_normal": civ_act,
            "cii_activity_pct_normal": cii_act,
            "cpeo": cpeo,
            "myopathy": myopathy,
            "cardiomyopathy": cardio,
            "snhl": snhl,
            "ragged_red_fibres": rrfs,
            "lactic_acidosis": lactic,
            "leigh_like_mri": leigh_mri,
            "exercise_intolerance": exercise,
            "compound_ta_loss": compound_ta_loss,
            "age_onset_yr": onset,
        })
    return patients


_COHORT = _build_cohort()


def _cohort_stats():
    c = _COHORT
    n = len(c)

    def pct(key): return round(sum(1 for p in c if p[key]) / n * 100, 1)
    def avg(key): return round(sum(p[key] for p in c) / n, 1)

    return {
        "n_patients":                        n,
        "avg_heteroplasmy_blood_pct":        avg("heteroplasmy_blood_pct"),
        "avg_ci_activity_pct_normal":        avg("ci_activity_pct_normal"),
        "avg_civ_activity_pct_normal":       avg("civ_activity_pct_normal"),
        "avg_cii_activity_pct_normal":       avg("cii_activity_pct_normal"),
        "pct_cpeo":                          pct("cpeo"),
        "pct_myopathy":                      pct("myopathy"),
        "pct_cardiomyopathy":                pct("cardiomyopathy"),
        "pct_snhl":                          pct("snhl"),
        "pct_ragged_red_fibres":             pct("ragged_red_fibres"),
        "pct_lactic_acidosis":               pct("lactic_acidosis"),
        "pct_leigh_like_mri":                pct("leigh_like_mri"),
        "pct_exercise_intolerance":          pct("exercise_intolerance"),
        "pct_compound_ta_loss":              pct("compound_ta_loss"),
        "avg_age_onset_yr":                  avg("age_onset_yr"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_overview():
    stats = _cohort_stats()
    return {
        "title":    "MT-TN — tRNA-Asn (GTT Anticodon)",
        "subtitle": "Combined CI+CIV Deficiency · CPEO · Myopathy · L-strand NGS Pitfall — NINTH tRNA in mt-genome — L-strand rCRS 5657–5729",
        "omim":     "OMIM *590010",
        "gene_facts": {
            "gene":             "MT-TN",
            "product":          "tRNA-Asn (Asparagine), 73 nt RNA gene",
            "anticodon":        "GTT (reads AAC by Watson-Crick; reads AAU by wobble at G34 — both Asn codons in human mt code)",
            "strand":           "L-strand — NGS PITFALL: reverse-complement QC MANDATORY for rCRS 5657–5729 AND the entire L-strand cluster (MT-TA/TN/TC/TY to rCRS 5891)",
            "rCRS_coordinates": "5657–5729",
            "length_nt":        73,
            "genome_position":  "NINTH tRNA in mitochondrial genome — SECOND of four consecutive L-strand tRNAs",
            "flanking_5prime":  "MT-TA (L-strand, 5587–5655) — 2 nt gap at rCRS 5655–5657",
            "flanking_3prime":  "MT-TC (L-strand, 5761–5826) — 32 nt gap at rCRS 5729–5761",
            "l_strand_cluster": "MT-TA (5587–5655) → MT-TN (5657–5729) → MT-TC (5761–5826) → MT-TY (5826–5891) — ALL L-strand encoded",
            "inheritance":      "Maternal (heteroplasmic)",
            "omim_gene":        "MT-TN *590010",
        },
        "l_strand_ngs_alert": {
            "alert_class": "L-STRAND NGS PITFALL — MANDATORY REVERSE-COMPLEMENT QC — FOUR-GENE L-strand CLUSTER",
            "detail": (
                "MT-TN (rCRS 5657–5729) is L-strand encoded. Standard NGS pipelines using H-strand "
                "(rCRS) reference calls WILL MISS or MIS-CALL MT-TN variants unless the pipeline "
                "applies reverse-complement processing. The same pitfall applies to all four tRNAs "
                "in this cluster: MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), "
                "MT-TY (5826–5891). Laboratories must apply L-strand QC from rCRS 5580 (immediately "
                "after MT-TW) through rCRS 5891 (end of MT-TY). "
                "This is the same pitfall class as MT-TQ, MT-TE, MT-TP, MT-ND6."
            ),
        },
        "nars2_ddx_note": {
            "note_class": "NARS2 NUCLEAR DDx — PERRAULT SYNDROME TYPE 6 vs ADULT CPEO",
            "detail": (
                "NARS2 (mt-Asn-tRNA synthetase 2) biallelic mutations cause Asn-tRNA aminoacylation "
                "failure — same biochemical level as MT-TN, but entirely different clinical phenotype. "
                "PERRAULT SYNDROME TYPE 6 (OMIM #617717): FEMALES: sensorineural hearing loss + "
                "premature ovarian failure (POF/POI); MALES: hearing loss only. "
                "Alternatively: severe early-onset epileptic encephalopathy with regression. "
                "NARS2 is AR, WES-detectable. MT-TN does NOT cause Perrault syndrome "
                "and does NOT produce infantile epileptic encephalopathy. Maternal CPEO + NO POI "
                "= MT-TN, NOT NARS2."
            ),
        },
        "biochemical_fingerprint": {
            "summary":   "Combined CI+CIV Deficiency (CII NORMAL — mt-translation fingerprint); symmetric CI≈CIV deficiency pattern",
            "complex_i": "Deficient (avg ~{:.0f}% normal)".format(stats["avg_ci_activity_pct_normal"]),
            "complex_ii": "NORMAL (nuclear-encoded; avg ~{:.0f}% normal)".format(stats["avg_cii_activity_pct_normal"]),
            "complex_iv": "Deficient (avg ~{:.0f}% normal)".format(stats["avg_civ_activity_pct_normal"]),
            "mechanism": (
                "Asn (AAC/AAU) residues appear in the hydrophilic loop regions and matrix-facing "
                "domains of CI subunits (MT-ND1, MT-ND2, MT-ND5, MT-ND6) and CIV subunits (MT-CO1, "
                "MT-CO2, MT-CO3). MT-TN mutations reduce tRNA-Asn availability → impaired elongation "
                "at AAC/AAU codons → global mt-translation defect affecting all 13 OXPHOS subunits, "
                "with CI and CIV showing the greatest enzyme deficiency. CII (succinate dehydrogenase) "
                "is entirely nuclear-encoded and remains intact. MT-TN produces a symmetric CI ≈ CIV "
                "deficiency pattern — similar to MT-TA but distinct from MT-TW (CIV > CI via UGA-recoding "
                "disruption in COX1)."
            ),
        },
        "cohort_statistics":       stats,
        "cohort_summary_features": [
            {"feature": "CPEO (ptosis + progressive ophthalmoplegia)",         "value": f"{stats['pct_cpeo']}%",                "note": "dominant phenotype; adult onset in most variants"},
            {"feature": "Myopathy (RRF, COX-negative fibres)",                 "value": f"{stats['pct_myopathy']}%",            "note": "SDH-positive, COX-negative ragged-red fibres"},
            {"feature": "Exercise Intolerance",                                 "value": f"{stats['pct_exercise_intolerance']}%", "note": "early symptom, often predates ophthalmoplegia"},
            {"feature": "Cardiomyopathy (m.5728AG subset)",                    "value": f"{stats['pct_cardiomyopathy']}%",      "note": "acceptor stem variant; annual echo required"},
            {"feature": "SNHL (m.5692AG subset)",                              "value": f"{stats['pct_snhl']}%",                "note": "anticodon loop variant; cochlear implants effective"},
            {"feature": "Lactic Acidosis",                                     "value": f"{stats['pct_lactic_acidosis']}%",     "note": "heteroplasmy-dependent; perioperative risk"},
            {"feature": "Leigh-like MRI (m.5703GA / high heteroplasmy)",       "value": f"{stats['pct_leigh_like_mri']}%",      "note": "bilateral BG + brainstem — BTBGD exclusion mandatory"},
            {"feature": "Ragged-Red Fibres (muscle biopsy)",                   "value": f"{stats['pct_ragged_red_fibres']}%",  "note": "SDH over-staining hallmark; COX-negative on dual stain"},
            {"feature": "Compound MT-TA loss (LargeDel subset)",               "value": f"{stats['pct_compound_ta_loss']}%",   "note": "LargeDel spanning 5587–5729: simultaneous MT-TA + MT-TN loss"},
        ],
        "phenotype_distribution": [
            {"variant": "m.5690A>G",    "pct": 28, "phenotype": "CPEO + Myopathy",              "position": "Anticodon stem (positions ~27–31/39–43 equivalent)"},
            {"variant": "m.5728A>G",    "pct": 22, "phenotype": "CPEO + Cardiomyopathy",        "position": "Acceptor stem — 3′ terminus region"},
            {"variant": "m.5703G>A",    "pct": 20, "phenotype": "Multisystem / Leigh-like",     "position": "Variable loop / T-stem junction"},
            {"variant": "m.5692A>G",    "pct": 15, "phenotype": "Exercise Intol + SNHL",        "position": "Anticodon loop (position ~34–36 adjacent)"},
            {"variant": "Large deletion","pct": 15, "phenotype": "KSS/CPEO — MT-TA+TN spanning", "position": "Spanning rCRS ~5587–5729 (compound loss)"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<30",  "expected_phenotype": "Subclinical — annual ophthalmology + exercise assessment; NARS2 exclusion if SNHL + POI in family"},
            {"threshold_pct": "30–50","expected_phenotype": "Exercise intolerance + early CPEO — OXPHOS surveillance; confirm L-strand NGS QC was used for this region"},
            {"threshold_pct": "50–70","expected_phenotype": "Full CPEO + myopathy + lactic acidosis; cardiomyopathy risk (m.5728AG); symmetric CI≈CIV deficiency"},
            {"threshold_pct": ">70",  "expected_phenotype": "Severe CPEO + myopathy; Leigh-like MRI risk; global mt-translation failure; compound loss if LargeDel"},
        ],
        "key_molecular_features": [
            "NINTH tRNA in the human mitochondrial genome (after MT-TF, MT-TV, MT-TL1, MT-TI, MT-TQ, MT-TM, MT-TW, MT-TA)",
            "SECOND of four consecutive L-strand tRNAs: MT-TA → MT-TN → MT-TC → MT-TY (all rCRS 5587–5891)",
            "L-strand NGS pitfall: reverse-complement QC MANDATORY — same pitfall class as MT-TA, MT-TQ, MT-TE, MT-TP, MT-ND6",
            "GTT anticodon — decodes AAC (Watson-Crick) and AAU (wobble G34:U) — both Asn codons in human mt code",
            "Asn (AAC/AAU) in hydrophilic loops and matrix-domain residues of CI (MT-ND1/ND2/ND5/ND6) and CIV (MT-CO1/CO2/CO3)",
            "Symmetric CI ≈ CIV deficiency — contrast with MT-TW (CIV > CI via UGA-recoding disruption in MT-CO1)",
            "2 nt gap 5′ from MT-TA (rCRS 5655–5657); 32 nt gap 3′ to MT-TC (rCRS 5729–5761)",
            "LargeDel may span MT-TA + MT-TN (2 nt gap makes TA-TN joint a common deletion breakpoint): compound tRNA-Ala + tRNA-Asn loss",
            "NARS2 nuclear DDx: Perrault syndrome type 6 (SNHL + POI) or infantile encephalopathy — NOT adult CPEO (MT-TN)",
        ],
        "clinical_alerts": [
            {"alert": "L-STRAND NGS PITFALL — FOUR-GENE CLUSTER — MANDATORY QC SWITCH",            "detail": "MT-TA, MT-TN, MT-TC, MT-TY (rCRS 5587–5891) ALL L-strand encoded; reverse-complement QC required from rCRS 5580 onward; H-strand-only pipelines MISS variants in all four genes"},
            {"alert": "NARS2 DDx — SNHL + POI in females distinguishes NARS2 from MT-TN",          "detail": "Perrault syndrome: SNHL + premature ovarian failure = NARS2 (AR, WES); adult CPEO + maternal inheritance = MT-TN (mtDNA panel, L-strand QC)"},
            {"alert": "COMPOUND LOSS — LargeDel spanning MT-TA + MT-TN",                            "detail": "2 nt gap between MT-TA and MT-TN (rCRS 5655–5657) makes the TA-TN junction a common deletion breakpoint; simultaneous tRNA-Ala + tRNA-Asn loss; compounded CI+CIV deficiency"},
            {"alert": "m.5728AG — acceptor stem — cardiomyopathy risk — annual echo",              "detail": "Acceptor stem 3′-terminus variant; cardiomyopathy in ~38% at heteroplasmy >50%; annual echocardiography + Holter; beta-blocker if HCM/DCM detected"},
            {"alert": "BTBGD (SLC19A3) — MANDATORY EXCLUSION",                                     "detail": "Treatable Leigh-like mimic; biotin+thiamine trial before attributing Leigh MRI to MT-TN at high heteroplasmy"},
            {"alert": "WES MISSES MT-TN",                                                           "detail": "WES does not sequence mtDNA adequately; dedicated mtDNA panel with L-strand QC required for MT-TN (L-strand, rCRS 5657–5729)"},
        ],
    }


def get_breakdown():
    stats = _cohort_stats()
    variants = {}
    for p in _COHORT:
        v = p["variant"]
        if v not in variants:
            variants[v] = {"variant": v, "n": 0, "pct_cpeo": [], "pct_myopathy": [], "pct_cardio": [],
                           "avg_hetero": [], "avg_ci": [], "avg_civ": [], "pct_compound": []}
        variants[v]["n"] += 1
        variants[v]["avg_hetero"].append(p["heteroplasmy_blood_pct"])
        variants[v]["avg_ci"].append(p["ci_activity_pct_normal"])
        variants[v]["avg_civ"].append(p["civ_activity_pct_normal"])
        if p["cpeo"]:            variants[v]["pct_cpeo"].append(1)
        if p["myopathy"]:        variants[v]["pct_myopathy"].append(1)
        if p["cardiomyopathy"]:  variants[v]["pct_cardio"].append(1)
        if p["compound_ta_loss"]: variants[v]["pct_compound"].append(1)

    breakdown = []
    for v, d in variants.items():
        n = d["n"]
        breakdown.append({
            "variant":                   v,
            "n":                         n,
            "pct_of_cohort":             round(n / 40 * 100, 1),
            "avg_heteroplasmy":          round(sum(d["avg_hetero"]) / n, 1),
            "avg_ci_pct_normal":         round(sum(d["avg_ci"]) / n, 1),
            "avg_civ_pct_normal":        round(sum(d["avg_civ"]) / n, 1),
            "pct_cpeo":                  round(len(d["pct_cpeo"]) / n * 100, 1),
            "pct_myopathy":              round(len(d["pct_myopathy"]) / n * 100, 1),
            "pct_cardiomyopathy":        round(len(d["pct_cardio"]) / n * 100, 1),
            "pct_compound_ta_loss":      round(len(d["pct_compound"]) / n * 100, 1),
        })

    return {
        "title":             "MT-TN Variant & Phenotype Breakdown — 40-patient cohort seed-817",
        "variant_breakdown": sorted(breakdown, key=lambda x: -x["n"]),
        "ddx_table": [
            {
                "entity":       "MT-TN pathogenic variant",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Adult CPEO + myopathy + exercise intolerance; cardiomyopathy in m.5728AG",
                "biochemistry": "CI+CIV deficiency (symmetric), CII NORMAL; mt-translation fingerprint",
                "ngs":          "L-strand mtDNA panel — WES MISSES; H-strand-only pipeline MISSES MT-TN",
                "distinctive":  "L-strand NGS pitfall; second of four L-strand cluster tRNAs (MT-TA/TN/TC/TY); GTT anticodon; 2 nt gap from MT-TA",
            },
            {
                "entity":       "NARS2 (mt-Asn-tRNA synthetase 2)",
                "inheritance":  "AR nuclear",
                "phenotype":    "Perrault syndrome type 6: SNHL + premature ovarian failure (females); OR early-onset epileptic encephalopathy",
                "biochemistry": "Combined OXPHOS deficiency; similar mt-translation impairment",
                "ngs":          "WES detects biallelic NARS2 mutations",
                "distinctive":  "Perrault (SNHL + POI): NOT seen in MT-TN; AR not maternal; WES-detectable; no adult CPEO",
            },
            {
                "entity":       "MT-TA (tRNA-Ala)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; GGC anticodon; first of L-strand cluster",
                "biochemistry": "CI+CIV deficiency (symmetric), CII NORMAL",
                "ngs":          "L-strand mtDNA panel — same pitfall cluster (2 nt gap before MT-TN)",
                "distinctive":  "Immediately 5′ of MT-TN (rCRS 5587–5655); AARS2 DDx (not NARS2); compound loss with MT-TN in large deletions",
            },
            {
                "entity":       "MT-TW (tRNA-Trp)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; UGA recoding; CIV > CI asymmetry",
                "biochemistry": "CI+CIV deficiency; CIV preferentially worse (UGA-Trp/COX1 disruption)",
                "ngs":          "H-strand mtDNA panel (MT-TW itself no L-strand pitfall; but L-strand cluster begins 3′)",
                "distinctive":  "UGA codon recoding; H-strand 5512–5579; immediately 5′ to the L-strand cluster",
            },
            {
                "entity":       "BTBGD / SLC19A3",
                "inheritance":  "AR nuclear",
                "phenotype":    "Biotin-thiamine-responsive Leigh-like — symmetric BG signals",
                "biochemistry": "Normal OXPHOS on treatment",
                "ngs":          "WES",
                "distinctive":  "TREATABLE — biotin + thiamine trial MANDATORY before attributing Leigh MRI to MT-TN",
            },
            {
                "entity":       "MT-TL1 (MELAS)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Stroke-like episodes, CPEO, ragged-red fibres",
                "biochemistry": "Pan-OXPHOS CI+CIII+CIV deficiency",
                "ngs":          "H-strand mtDNA panel — m.3243AG most common",
                "distinctive":  "STROKE-LIKE EPISODES and MELAS: NOT seen in MT-TN",
            },
        ],
        "management_by_variant": [
            {"variant": "m.5690AG",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Annual ophthalmology; ptosis surgery PRN; L-strand NGS QC verification; CoQ10 + riboflavin"},
            {"variant": "m.5728AG",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "Annual echo + Holter; acceptor stem — cardiomyopathy ~38% at >50% heteroplasmy; beta-blocker if HCM/DCM"},
            {"variant": "m.5703GA",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Full OXPHOS surveillance; BTBGD exclusion; Leigh MRI monitoring; early multidisciplinary referral"},
            {"variant": "m.5692AG",  "cpeo_risk": "Mod",  "cardio_risk": "Low",      "key_action": "Audiology + cochlear implant; anticodon-loop variant — SNHL in ~50%; watch OXPHOS progression"},
            {"variant": "LargeDel",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "KSS surveillance (cardiac + retina + endocrine); compound MT-TA+TN loss check; CSF lactate; L-strand QC"},
        ],
        "absolute_contraindications": [
            {"drug": "Metformin",       "reason": "Complex I inhibition → fatal lactic acidosis in CI-deficient patients"},
            {"drug": "Valproate (VPA)", "reason": "Inhibits mtDNA replication; depletes CoA; worsens hepatopathy"},
            {"drug": "Propofol",        "reason": "Propofol infusion syndrome (PRIS) — uncouples OXPHOS; fatal in mt disease"},
            {"drug": "Linezolid",       "reason": "Inhibits mitochondrial 23S rRNA translation → iatrogenic mt-translation failure"},
            {"drug": "Chloramphenicol", "reason": "mt-ribosome inhibitor; cumulative mt-translation toxicity"},
        ],
        "safe_interventions": [
            {"intervention": "CoQ10 (Ubiquinol)",   "evidence": "Level C — electron carrier support; 200–600 mg/day adult"},
            {"intervention": "Riboflavin (B2)",     "evidence": "Level C — Complex I/II cofactor; 100–400 mg/day"},
            {"intervention": "L-Carnitine",         "evidence": "Level C — acylcarnitine transport support"},
            {"intervention": "Thiamine (B1)",       "evidence": "MANDATORY empiric — BTBGD exclusion + mt-energy support"},
            {"intervention": "Biotin",              "evidence": "MANDATORY empiric — BTBGD exclusion; 5–10 mg/day"},
            {"intervention": "Levetiracetam (LEV)", "evidence": "Preferred AED — avoids VPA; safe OXPHOS profile"},
            {"intervention": "Beta-blocker",        "evidence": "First-line if cardiomyopathy (m.5728AG); replaces amiodarone"},
            {"intervention": "GIR 6–8 mg/kg/min",  "evidence": "MANDATORY perioperative — prevents catabolic crisis; never fast in mt disease"},
        ],
        "cohort_statistics": stats,
    }


def get_definitions():
    return {
        "title": "MT-TN — Clinical & Molecular Definitions",
        "gene_definitions": [
            {"term": "MT-TN",               "definition": "Mitochondrially encoded tRNA-Asn gene; L-strand, rCRS 5657–5729, 73 nt (OMIM *590010)"},
            {"term": "tRNA-Asn",            "definition": "Transfer RNA for Asparagine; GTT anticodon, decoding AAC (Watson-Crick) and AAU (wobble G34:U) — both Asn codons in human mt code"},
            {"term": "GTT anticodon",       "definition": "Anticodon of tRNA-Asn (positions 34–36 of the tRNA); G34 wobble position reads both AAC and AAU; Asn is exclusively decoded by MT-TN in human mitochondria"},
            {"term": "L-strand cluster",    "definition": "MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), MT-TY (5826–5891) — four consecutive L-strand tRNA genes; the entire 304 nt block rCRS 5587–5891 requires L-strand QC"},
            {"term": "rCRS",                "definition": "Revised Cambridge Reference Sequence — standard human mtDNA numbering (GenBank NC_012920.1)"},
            {"term": "Heteroplasmy",        "definition": "Mixture of wild-type and mutant mtDNA molecules within a cell/tissue; severity correlates with mutant load (%)"},
        ],
        "biochemical_definitions": [
            {"term": "CI deficiency",       "definition": "NADH:ubiquinone oxidoreductase (Complex I) deficiency — 7 mtDNA-encoded subunits; sensitive to mt-tRNA defects; Asn in hydrophilic loops of MT-ND1/ND2/ND5/ND6"},
            {"term": "CII (normal)",        "definition": "Succinate dehydrogenase — entirely nuclear-encoded; remains normal in mt-tRNA disease (diagnostic discriminator)"},
            {"term": "CIV deficiency",      "definition": "Cytochrome c oxidase (COX) deficiency — 3 mtDNA-encoded subunits including COX1/COX2/COX3; COX-negative fibres on histochemistry"},
            {"term": "Symmetric CI≈CIV",    "definition": "MT-TN produces symmetric CI ≈ CIV deficiency — contrast with MT-TW (CIV preferentially worse due to UGA-recoding disruption in MT-CO1)"},
            {"term": "mt-translation fingerprint", "definition": "Combined CI+CIV deficiency with CII NORMAL — characteristic of pathogenic mt-tRNA gene mutations"},
            {"term": "Ragged-Red Fibres",   "definition": "Subsarcolemmal mitochondrial accumulation on modified Gomori trichrome; SDH-positive/COX-negative on sequential histochemistry"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",                "definition": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive ophthalmoplegia; predominant MT-TN phenotype"},
            {"term": "KSS",                 "definition": "Kearns-Sayre Syndrome — CPEO + pigmentary retinopathy + cardiac conduction defect; often large mtDNA deletion spanning MT-TA+MT-TN"},
            {"term": "Perrault Syndrome Type 6", "definition": "OMIM #617717 — AR NARS2 biallelic mutations → sensorineural hearing loss + premature ovarian failure (females); hearing loss alone (males) — NOT adult CPEO; key NARS2 DDx"},
            {"term": "Premature Ovarian Failure (POF/POI)", "definition": "Key discriminator for NARS2 vs MT-TN: POI present = NARS2 (AR, WES); POI absent + maternal CPEO = MT-TN (mtDNA panel, L-strand QC)"},
            {"term": "Compound tRNA loss",  "definition": "Large deletions spanning MT-TA (L-strand 5587–5655) + MT-TN (L-strand 5657–5729): simultaneous loss of tRNA-Ala and tRNA-Asn decoding capacity; 2 nt gap between genes is common deletion breakpoint"},
            {"term": "BTBGD / SLC19A3",    "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease — AR treatable Leigh mimic; mandatory exclusion before attributing Leigh MRI to MT-TN"},
        ],
        "ngs_definitions": [
            {"term": "L-strand encoding",   "definition": "MT-TN (rCRS 5657–5729) is L-strand encoded; standard NGS H-strand calls MISS variants; reverse-complement pipeline required"},
            {"term": "L-strand NGS pitfall","definition": "H-strand-only NGS pipelines fail to detect L-strand tRNA variants; clinically identical pitfall to MT-TA (5587–5655), MT-TQ (4329–4400), MT-TE (14674–14742), MT-TP (15956–16023), MT-ND6 (14149–14673)"},
            {"term": "Four-gene L-strand cluster", "definition": "MT-TA, MT-TN, MT-TC, MT-TY occupy rCRS 5587–5891 — all L-strand; the longest consecutive L-strand tRNA block in the human mitochondrial genome; QC must cover the entire block"},
            {"term": "Variant allele fraction (VAF)", "definition": "Proportion of reads carrying the variant allele = heteroplasmy estimate; tissue-specific (blood < muscle < affected tissue)"},
        ],
        "drug_definitions": [
            {"term": "Absolute CI",         "definition": "Metformin, VPA, Propofol, Linezolid, Chloramphenicol — all worsen OXPHOS or deplete mtDNA; fatal in mt disease"},
            {"term": "PRIS",                "definition": "Propofol Infusion Syndrome — mitochondrial OXPHOS uncoupling; metabolic acidosis; rhabdomyolysis; high mortality in mt disease"},
            {"term": "GIR 6–8",             "definition": "Glucose Infusion Rate 6–8 mg/kg/min — mandatory during any perioperative fasting period; prevents catabolic crisis in mt disease"},
            {"term": "KD contraindication", "definition": "Ketogenic diet contraindicated in mt disease — forces OXPHOS-dependent fatty acid oxidation; stresses already-deficient ETC"},
        ],
        "references": [
            {"ref": "DiMauro-Schon-2003-NEJM",         "citation": "DiMauro S, Schon EA (2003) Mitochondrial respiratory-chain diseases. N Engl J Med 348:2656–2668"},
            {"ref": "Schaefer-2008-AnnNeurol",          "citation": "Schaefer AM et al. (2008) Prevalence of mitochondrial disease in adults. Ann Neurol 63:35–39"},
            {"ref": "Gorman-2016-NatRevDisPrimers",     "citation": "Gorman GS et al. (2016) Mitochondrial diseases. Nat Rev Dis Primers 2:16080"},
            {"ref": "Boczonadi-2018-NARS2-Perrault",    "citation": "Boczonadi V et al. (2018) NARS2 mutations cause Perrault syndrome with progressive encephalopathy. Brain 141:2285–2300"},
            {"ref": "Vanlander-2015-NARS2",             "citation": "Vanlander AV et al. (2015) Two siblings with homozygous pathogenic mutations in mitochondrial asparaginyl-tRNA synthetase. Hum Mutat 36:E2437–E2443"},
            {"ref": "Sprinzl-2005-NAR-tRNA",            "citation": "Sprinzl M & Vassilenko KS (2005) Compilation of tRNA sequences and sequences of tRNA genes. Nucleic Acids Res 33:D139–D140"},
            {"ref": "Rossmanith-1995-JBiolChem-tRNA",   "citation": "Rossmanith W et al. (1995) Processing of human mitochondrial tRNA precursors. J Biol Chem 270:12885–12891"},
        ],
    }
