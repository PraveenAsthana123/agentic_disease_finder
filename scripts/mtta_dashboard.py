#!/usr/bin/env python3
"""MT-TA — Mitochondrially Encoded tRNA-Ala — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | L-strand rCRS 5587–5655

MT-TA (OMIM *590000) encodes mitochondrial tRNA-Ala (GGC anticodon), located on the
**L-strand** at rCRS 5587–5655 (69 nt). MT-TA is the EIGHTH tRNA gene of the human
mitochondrial genome, following MT-TF (577–647), MT-TV (1602–1670), MT-TL1 (3230–3304),
MT-TI (4263–4331), MT-TQ (L-strand 4329–4400), MT-TM (4402–4469), and MT-TW (5512–5579).
MT-TA begins the consecutive L-strand tRNA cluster: MT-TA (5587–5655), MT-TN (5657–5729),
MT-TC (5761–5826), MT-TY (5826–5891) — four consecutive L-strand tRNAs in one genomic block.

CRITICAL UNIQUE FEATURE — L-STRAND NGS PITFALL:
MT-TA is encoded on the L-strand (light/complementary strand). Standard NGS pipelines
optimised for H-strand (heavy/reference-strand) variant calling will MISS or MIS-CALL
MT-TA variants. The gene is read in the reverse-complement direction relative to the rCRS
reference sequence. This L-strand pitfall is shared by MT-TQ, MT-TE, MT-TP, MT-ND6 —
but in the MT-TA region, ALL FOUR flanking tRNAs (MT-TA, MT-TN, MT-TC, MT-TY) are
L-strand encoded, making this a FOUR-GENE CONSECUTIVE L-strand block requiring mandatory
reverse-complement QC processing for the entire rCRS 5587–5891 region.

L-STRAND CLUSTER — FIRST OF FOUR:
MT-TA is the first (and most 5'-proximal in the L-strand reading direction) of the four
consecutive L-strand tRNAs. Immediately 5' is MT-TW (H-strand, 5512–5579) with a 7 nt gap.
Immediately 3' is MT-TN (L-strand, 5657–5729) with a 2 nt gap at rCRS 5655–5657.
Large mtDNA deletions spanning this region may simultaneously remove MT-TW (H-strand)
AND one or more of the L-strand cluster tRNAs — producing compound tRNA loss.

GGC ANTICODON — FOUR-FOLD CODON DEGENERACY:
MT-TA has a GGC anticodon, reading GCC (Ala) codons with standard Watson-Crick pairing.
In human mitochondria, additional wobble (G:U at position 34) allows MT-TA to read
GCU, GCA, and GCG — giving tRNA-Ala full four-fold degeneracy for the GCN box.
Ala (GCN) residues are abundant in hydrophobic transmembrane helices of OXPHOS subunits
(particularly MT-ND2, MT-ND4, MT-ND5, MT-CO1). MT-TA mutations reduce tRNA-Ala
availability → impaired elongation at GCN codons → CI+CIV assembly defect.

NUCLEAR DDx — AARS2 (mt-Alanyl-tRNA Synthetase 2):
AARS2 biallelic mutations cause mitochondrial Ala-tRNA aminoacylation failure — but with
a dramatically different clinical phenotype: in females, AARS2 causes ovario-leukodystrophy
(premature ovarian insufficiency + adult-onset progressive leukoencephalopathy, OMIM #615889);
in males, early-onset lethal cardiomyopathy with lactic acidosis. AARS2 is AR, WES-detectable,
and clinically distinguishable from MT-TA disease by: (1) ovarian insufficiency in females;
(2) white matter signal on MRI (not Leigh basal-ganglia/brainstem pattern); (3) absence of
maternal inheritance; (4) WES detection. MT-TA (maternal heteroplasmic CPEO) does NOT
cause ovarian insufficiency and does NOT produce leukodystrophy.

  MT-TA gene              OMIM *590000
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Occasional Cardiomyopathy / SNHL / Leigh-like at high heteroplasmy
                          L-strand NGS pitfall: mandatory reverse-complement QC for rCRS 5587–5655
  Protein product         tRNA-Ala (GGC anticodon) — 69 nucleotides; RNA gene
                          Four-fold degenerate: reads GCC/GCU/GCA/GCG (Ala) in human mt code
  Genome                  Mitochondrial DNA (mtDNA), **L-strand**, rCRS 5587–5655
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    EIGHTH tRNA in mitochondrial genome
                          5′-adjacent to MT-TW (H-strand, 5512–5579), 7 nt gap
                          3′-adjacent to MT-TN (L-strand, 5657–5729), 2 nt gap
                          FIRST OF FOUR consecutive L-strand tRNAs (MT-TA/TN/TC/TY cluster)
"""

import random
import math

# ── Seed for reproducible 40-patient cohort ──────────────────────────────────
SEED = 815
RNG  = random.Random(SEED)


def _rand_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    v = RNG.gauss(mu, sigma)
    return round(max(lo, min(hi, v)), 1)


def _rand_int(lo: int, hi: int) -> int:
    return RNG.randint(lo, hi)


# ── 40-patient cohort generator ───────────────────────────────────────────────
def _build_cohort():
    """Generate deterministic 40-patient MT-TA cohort (seed-815)."""
    N = 40
    patients = []

    # Variant distribution (must sum to 40)
    variant_pool = (
        ["m.5628CT"]  * 11 +   # D-stem — CPEO + myopathy           ~28%
        ["m.5655AG"]  * 9  +   # acceptor stem — CPEO + cardio       ~22%
        ["m.5644CT"]  * 8  +   # anticodon stem — multisystem        ~19%
        ["m.5650AG"]  * 6  +   # T-stem — exercise + SNHL            ~16%
        ["LargeDel"]  * 6      # MT-TW/TA-spanning KSS/CPEO del      ~15%
    )
    RNG.shuffle(variant_pool)

    for i in range(N):
        variant = variant_pool[i]

        # Heteroplasmy by variant
        if variant == "m.5628CT":
            hetero = _rand_normal(56, 15, 25, 88)
        elif variant == "m.5655AG":
            hetero = _rand_normal(58, 17, 22, 90)
        elif variant == "m.5644CT":
            hetero = _rand_normal(63, 16, 32, 92)
        elif variant == "m.5650AG":
            hetero = _rand_normal(50, 18, 20, 86)
        else:  # LargeDel
            hetero = _rand_normal(60, 18, 26, 91)

        # Enzyme activities — CI+CIV deficiency, CII NORMAL
        # Standard mt-tRNA fingerprint: Ala abundant in TM helices of CI/CIV subunits
        ci_act  = _rand_normal(36 + (100 - hetero) * 0.32, 8,  9, 68)
        civ_act = _rand_normal(33 + (100 - hetero) * 0.28, 9,  7, 66)
        cii_act = _rand_normal(94, 4, 82, 104)  # CII nuclear-encoded → normal

        # Clinical features
        cpeo     = hetero > 36 or variant in ("m.5628CT", "m.5655AG", "LargeDel")
        myopathy = hetero > 33 or variant in ("m.5628CT", "m.5644CT")
        cardio   = variant == "m.5655AG" and hetero > 50 and RNG.random() < 0.40
        snhl     = variant in ("m.5650AG",) and RNG.random() < 0.52
        rrfs     = myopathy and RNG.random() < 0.76
        lactic   = hetero > 46 and RNG.random() < 0.63
        leigh_mri = (variant == "m.5644CT" or hetero > 70) and RNG.random() < 0.35
        exercise = hetero > 24 and RNG.random() < 0.84
        # Compound loss: LargeDel may also eliminate MT-TW (H-strand, 5512–5579)
        compound_tw_loss = variant == "LargeDel" and RNG.random() < 0.55

        # Onset age
        if variant == "m.5644CT":
            onset = _rand_normal(16, 9, 5, 40)    # multisystem — younger
        elif variant == "LargeDel":
            onset = _rand_normal(19, 10, 7, 42)
        elif variant == "m.5650AG":
            onset = _rand_normal(22, 12, 7, 46)
        else:
            onset = _rand_normal(34, 13, 16, 63)

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
            "compound_tw_loss": compound_tw_loss,
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
        "pct_compound_tw_loss":              pct("compound_tw_loss"),
        "avg_age_onset_yr":                  avg("age_onset_yr"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_overview():
    stats = _cohort_stats()
    return {
        "title":    "MT-TA — tRNA-Ala (GGC Anticodon)",
        "subtitle": "Combined CI+CIV Deficiency · CPEO · Myopathy · L-strand NGS Pitfall — EIGHTH tRNA in mt-genome — L-strand rCRS 5587–5655",
        "omim":     "OMIM *590000",
        "gene_facts": {
            "gene":             "MT-TA",
            "product":          "tRNA-Ala (Alanine), 69 nt RNA gene",
            "anticodon":        "GGC (reads GCC, GCU, GCA, GCG — full four-fold Ala degeneracy in mt code)",
            "strand":           "L-strand — NGS PITFALL: reverse-complement QC MANDATORY for rCRS 5587–5655 AND the entire L-strand cluster (MT-TA/TN/TC/TY to rCRS 5891)",
            "rCRS_coordinates": "5587–5655",
            "length_nt":        69,
            "genome_position":  "EIGHTH tRNA in mitochondrial genome — FIRST of four consecutive L-strand tRNAs",
            "flanking_5prime":  "MT-TW (H-strand, 5512–5579) — 7 nt gap at rCRS 5580–5586",
            "flanking_3prime":  "MT-TN (L-strand, 5657–5729) — 2 nt gap at rCRS 5655–5657",
            "l_strand_cluster": "MT-TA (5587–5655) → MT-TN (5657–5729) → MT-TC (5761–5826) → MT-TY (5826–5891) — ALL L-strand encoded",
            "inheritance":      "Maternal (heteroplasmic)",
            "omim_gene":        "MT-TA *590000",
        },
        "l_strand_ngs_alert": {
            "alert_class": "L-STRAND NGS PITFALL — MANDATORY REVERSE-COMPLEMENT QC — FOUR-GENE L-strand CLUSTER",
            "detail": (
                "MT-TA (rCRS 5587–5655) is L-strand encoded. Standard NGS pipelines using H-strand "
                "(rCRS) reference calls WILL MISS or MIS-CALL MT-TA variants unless the pipeline "
                "applies reverse-complement processing. The same pitfall applies to MT-TN, MT-TC, "
                "and MT-TY — all immediately 3′, all L-strand, all within rCRS 5587–5891. "
                "Laboratories sequencing this region must apply L-strand QC from the moment the "
                "pipeline crosses MT-TW (rCRS 5579) until the end of MT-TY (rCRS 5891). "
                "This is the same pitfall class as MT-TQ, MT-TE, MT-TP, MT-ND6."
            ),
        },
        "aars2_ddx_note": {
            "note_class": "AARS2 NUCLEAR DDx — OVARIO-LEUKODYSTROPHY vs ADULT CPEO",
            "detail": (
                "AARS2 (mt-Ala-tRNA synthetase 2) biallelic mutations cause Ala-tRNA aminoacylation "
                "failure — same biochemical level as MT-TA, but entirely different clinical phenotype: "
                "FEMALES: premature ovarian insufficiency + progressive adult-onset leukoencephalopathy "
                "(white matter MRI — NOT Leigh BG/brainstem pattern). "
                "MALES: early-onset lethal cardiomyopathy + lactic acidosis. "
                "AARS2 is AR, WES-detectable (OMIM #615889). MT-TA does NOT cause ovarian "
                "insufficiency and does NOT produce leukodystrophy. Maternal inheritance + CPEO + "
                "NO ovarian insufficiency = MT-TA, NOT AARS2."
            ),
        },
        "biochemical_fingerprint": {
            "summary":   "Combined CI+CIV Deficiency (CII NORMAL — mt-translation fingerprint); standard symmetric CI≈CIV pattern",
            "complex_i": "Deficient (avg ~{:.0f}% normal)".format(stats["avg_ci_activity_pct_normal"]),
            "complex_ii": "NORMAL (nuclear-encoded; avg ~{:.0f}% normal)".format(stats["avg_cii_activity_pct_normal"]),
            "complex_iv": "Deficient (avg ~{:.0f}% normal)".format(stats["avg_civ_activity_pct_normal"]),
            "mechanism": (
                "Ala (GCN) residues are abundant in the transmembrane helices of CI subunits (MT-ND2, "
                "MT-ND4, MT-ND5) and CIV subunits (MT-CO1, MT-CO2). MT-TA mutations reduce tRNA-Ala "
                "availability → impaired elongation at GCN codons → global mt-translation defect "
                "affecting all 13 OXPHOS subunits, with CI and CIV showing the greatest enzyme "
                "deficiency. CII (succinate dehydrogenase) is entirely nuclear-encoded and remains "
                "intact. Unlike MT-TW (which has CIV > CI asymmetry via UGA-recoding disruption), "
                "MT-TA produces a symmetric CI ≈ CIV deficiency pattern."
            ),
        },
        "cohort_statistics":       stats,
        "cohort_summary_features": [
            {"feature": "CPEO (ptosis + progressive ophthalmoplegia)",        "value": f"{stats['pct_cpeo']}%",               "note": "dominant phenotype; adult onset in most variants"},
            {"feature": "Myopathy (RRF, COX-negative fibres)",                "value": f"{stats['pct_myopathy']}%",           "note": "SDH-positive, COX-negative ragged-red fibres"},
            {"feature": "Exercise Intolerance",                                "value": f"{stats['pct_exercise_intolerance']}%", "note": "early symptom, often predates ophthalmoplegia"},
            {"feature": "Cardiomyopathy (m.5655AG subset)",                   "value": f"{stats['pct_cardiomyopathy']}%",     "note": "acceptor stem variant; annual echo required"},
            {"feature": "SNHL (m.5650AG subset)",                              "value": f"{stats['pct_snhl']}%",               "note": "T-stem variant; cochlear implants effective"},
            {"feature": "Lactic Acidosis",                                     "value": f"{stats['pct_lactic_acidosis']}%",    "note": "heteroplasmy-dependent; perioperative risk"},
            {"feature": "Leigh-like MRI (m.5644CT / high heteroplasmy)",      "value": f"{stats['pct_leigh_like_mri']}%",     "note": "bilateral BG + brainstem — BTBGD exclusion mandatory"},
            {"feature": "Ragged-Red Fibres (muscle biopsy)",                  "value": f"{stats['pct_ragged_red_fibres']}%", "note": "SDH over-staining hallmark; COX-negative on dual stain"},
            {"feature": "Compound MT-TW loss (LargeDel subset)",              "value": f"{stats['pct_compound_tw_loss']}%",  "note": "LargeDel spanning 5512–5655: simultaneous MT-TW + MT-TA loss"},
        ],
        "phenotype_distribution": [
            {"variant": "m.5628C>T",    "pct": 28, "phenotype": "CPEO + Myopathy",            "position": "D-stem region"},
            {"variant": "m.5655A>G",    "pct": 22, "phenotype": "CPEO + Cardiomyopathy",      "position": "Acceptor stem — 3′ terminus region"},
            {"variant": "m.5644C>T",    "pct": 19, "phenotype": "Multisystem / Leigh-like",   "position": "Anticodon stem"},
            {"variant": "m.5650A>G",    "pct": 16, "phenotype": "Exercise Intol + SNHL",      "position": "T-stem region"},
            {"variant": "Large deletion","pct": 15, "phenotype": "KSS/CPEO — MT-TW+TA spanning","position": "Spanning rCRS ~5512–5655 (compound loss)"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<30",  "expected_phenotype": "Subclinical — annual ophthalmology + exercise assessment; AARS2 exclusion if leukodystrophy or POI in family"},
            {"threshold_pct": "30–50","expected_phenotype": "Exercise intolerance + early CPEO — OXPHOS surveillance; L-strand NGS pitfall: confirm mtDNA panel used L-strand QC"},
            {"threshold_pct": "50–70","expected_phenotype": "Full CPEO + myopathy + lactic acidosis; cardiomyopathy risk (m.5655AG); symmetric CI≈CIV deficiency"},
            {"threshold_pct": ">70",  "expected_phenotype": "Severe CPEO + myopathy; Leigh-like MRI risk; global mt-translation failure; compound loss if LargeDel"},
        ],
        "key_molecular_features": [
            "EIGHTH tRNA in the human mitochondrial genome (after MT-TF, MT-TV, MT-TL1, MT-TI, MT-TQ, MT-TM, MT-TW)",
            "FIRST of four consecutive L-strand tRNAs: MT-TA → MT-TN → MT-TC → MT-TY (all rCRS 5587–5891)",
            "L-strand NGS pitfall: reverse-complement QC MANDATORY — same pitfall class as MT-TQ, MT-TE, MT-TP, MT-ND6",
            "GGC anticodon — four-fold degenerate: decodes GCC, GCU, GCA, GCG (Ala) in human mt code",
            "Ala abundant in TM helices of CI (MT-ND2/ND4/ND5) and CIV (MT-CO1/CO2) subunits",
            "Symmetric CI ≈ CIV deficiency — contrast with MT-TW (CIV > CI asymmetry via UGA-recoding)",
            "7 nt gap 5′ from MT-TW (H-strand); 2 nt gap 3′ to MT-TN (L-strand)",
            "LargeDel may span MT-TW (H-strand) AND MT-TA (L-strand): compound tRNA-Trp + tRNA-Ala loss",
            "AARS2 nuclear DDx: ovario-leukodystrophy in females (POI + white matter) vs. adult CPEO (MT-TA)",
        ],
        "clinical_alerts": [
            {"alert": "L-STRAND NGS PITFALL — FOUR-GENE CLUSTER — MANDATORY QC SWITCH",           "detail": "MT-TA, MT-TN, MT-TC, MT-TY (rCRS 5587–5891) ALL L-strand encoded; reverse-complement QC required from rCRS 5580 onward; pipelines set to H-strand will MISS variants"},
            {"alert": "AARS2 DDx — OVARIAN INSUFFICIENCY in females distinguishes AARS2 from MT-TA","detail": "Premature ovarian insufficiency + white-matter MRI = AARS2 (AR, WES-detectable); adult CPEO + maternal inheritance = MT-TA; WES required if female with ovarian failure"},
            {"alert": "COMPOUND LOSS — LargeDel spanning MT-TW + MT-TA",                           "detail": "Large deletions in rCRS ~5480–5700 may remove both MT-TW (H-strand) and MT-TA (L-strand); expect combined tRNA-Trp + tRNA-Ala deficiency; KSS/CPEO phenotype"},
            {"alert": "m.5655AG — acceptor stem — cardiomyopathy risk — annual echo",              "detail": "Acceptor stem 3′-terminus variant; cardiomyopathy in ~40% at heteroplasmy >50%; annual echocardiography + Holter; beta-blocker if HCM/DCM detected"},
            {"alert": "BTBGD (SLC19A3) — MANDATORY EXCLUSION",                                    "detail": "Treatable Leigh-like mimic; biotin+thiamine trial before attributing Leigh MRI to MT-TA at high heteroplasmy"},
            {"alert": "WES MISSES MT-TA",                                                          "detail": "WES does not sequence mtDNA adequately; dedicated mtDNA panel with L-strand QC required for MT-TA (L-strand, rCRS 5587–5655)"},
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
        if p["compound_tw_loss"]: variants[v]["pct_compound"].append(1)

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
            "pct_compound_tw_loss":      round(len(d["pct_compound"]) / n * 100, 1),
        })

    return {
        "title":             "MT-TA Variant & Phenotype Breakdown — 40-patient cohort seed-815",
        "variant_breakdown": sorted(breakdown, key=lambda x: -x["n"]),
        "ddx_table": [
            {
                "entity":       "MT-TA pathogenic variant",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Adult CPEO + myopathy + exercise intolerance; cardiomyopathy in m.5655AG",
                "biochemistry": "CI+CIV deficiency (symmetric), CII NORMAL; mt-translation fingerprint",
                "ngs":          "L-strand mtDNA panel — WES MISSES; H-strand-only pipeline MISSES MT-TA",
                "distinctive":  "L-strand NGS pitfall; first of four L-strand cluster tRNAs (MT-TA/TN/TC/TY); GGC anticodon four-fold degenerate",
            },
            {
                "entity":       "AARS2 (mt-Ala-tRNA synthetase)",
                "inheritance":  "AR nuclear",
                "phenotype":    "Females: ovario-leukodystrophy (POI + leukoencephalopathy); Males: lethal cardiomyopathy + lactic acidosis",
                "biochemistry": "Combined OXPHOS deficiency; similar biochemistry",
                "ngs":          "WES detects biallelic AARS2 mutations",
                "distinctive":  "Ovarian insufficiency (NOT in MT-TA); white-matter MRI (NOT Leigh BG pattern); AR not maternal; POI = key discriminator",
            },
            {
                "entity":       "MT-TW (tRNA-Trp)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; CIV > CI asymmetry at high heteroplasmy",
                "biochemistry": "CI+CIV deficiency; CIV may exceed CI (UGA-recoding disruption)",
                "ngs":          "H-strand mtDNA panel (MT-TW itself no pitfall; but L-strand cluster follows 3′)",
                "distinctive":  "UGA codon recoding; sole tRNA decoding UGA as Trp; rCRS 5512–5579 (immediately 5′ to MT-TA)",
            },
            {
                "entity":       "MT-TN (tRNA-Asn)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; similar to MT-TA",
                "biochemistry": "CI+CIV deficiency, CII NORMAL",
                "ngs":          "L-strand mtDNA panel — same pitfall as MT-TA (immediately 3′ in same L-strand cluster)",
                "distinctive":  "Immediately 3′ of MT-TA (2 nt gap); QARS2-like DDx but NARS2; rCRS 5657–5729",
            },
            {
                "entity":       "BTBGD / SLC19A3",
                "inheritance":  "AR nuclear",
                "phenotype":    "Biotin-thiamine-responsive Leigh-like — symmetric BG signals",
                "biochemistry": "Normal OXPHOS on treatment",
                "ngs":          "WES",
                "distinctive":  "TREATABLE — biotin + thiamine trial MANDATORY before attributing Leigh MRI to MT-TA",
            },
            {
                "entity":       "MT-TL1 (MELAS)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Stroke-like episodes, CPEO, ragged-red fibres",
                "biochemistry": "Pan-OXPHOS CI+CIII+CIV deficiency",
                "ngs":          "H-strand mtDNA panel — m.3243AG most common",
                "distinctive":  "STROKE-LIKE EPISODES and MELAS: NOT seen in MT-TA",
            },
        ],
        "management_by_variant": [
            {"variant": "m.5628CT",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Annual ophthalmology; ptosis surgery PRN; L-strand NGS QC verification; CoQ10 + riboflavin"},
            {"variant": "m.5655AG",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "Annual echo + Holter; acceptor stem — cardiomyopathy ~40% at >50% heteroplasmy; beta-blocker if HCM/DCM"},
            {"variant": "m.5644CT",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Full OXPHOS surveillance; BTBGD exclusion; Leigh MRI monitoring; early referral multidisciplinary"},
            {"variant": "m.5650AG",  "cpeo_risk": "Mod",  "cardio_risk": "Low",      "key_action": "Audiology + cochlear implant; T-stem variant — SNHL in ~50%; watch OXPHOS progression"},
            {"variant": "LargeDel",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "KSS surveillance (cardiac + retina + endocrine); compound MT-TW+TA loss check; CSF lactate; L-strand QC"},
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
            {"intervention": "Beta-blocker",        "evidence": "First-line if cardiomyopathy (m.5655AG); replaces amiodarone"},
            {"intervention": "GIR 6–8 mg/kg/min",  "evidence": "MANDATORY perioperative — prevents catabolic crisis; never fast in mt disease"},
        ],
        "cohort_statistics": stats,
    }


def get_definitions():
    return {
        "title": "MT-TA — Clinical & Molecular Definitions",
        "gene_definitions": [
            {"term": "MT-TA",               "definition": "Mitochondrially encoded tRNA-Ala gene; L-strand, rCRS 5587–5655, 69 nt (OMIM *590000)"},
            {"term": "tRNA-Ala",            "definition": "Transfer RNA for Alanine; GGC anticodon, decoding GCC, GCU, GCA, GCG (four-fold degenerate Ala box in the mt genetic code)"},
            {"term": "GGC anticodon",       "definition": "Anticodon of tRNA-Ala (positions 34–36 of the tRNA); reads GCC by Watson-Crick; reads GCU/GCA/GCG by wobble in human mitochondria (G34 wobble)"},
            {"term": "L-strand cluster",    "definition": "MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), MT-TY (5826–5891) — four consecutive L-strand tRNA genes; the entire 304 nt block rCRS 5587–5891 requires L-strand QC"},
            {"term": "rCRS",                "definition": "Revised Cambridge Reference Sequence — standard human mtDNA numbering (GenBank NC_012920.1)"},
            {"term": "Heteroplasmy",        "definition": "Mixture of wild-type and mutant mtDNA molecules within a cell/tissue; severity correlates with mutant load (%)"},
        ],
        "biochemical_definitions": [
            {"term": "CI deficiency",       "definition": "NADH:ubiquinone oxidoreductase (Complex I) deficiency — 7 mtDNA-encoded subunits; sensitive to mt-tRNA defects; Ala-rich TM helices in MT-ND2/ND4/ND5"},
            {"term": "CII (normal)",        "definition": "Succinate dehydrogenase — entirely nuclear-encoded; remains normal in mt-tRNA disease (diagnostic discriminator)"},
            {"term": "CIV deficiency",      "definition": "Cytochrome c oxidase (COX) deficiency — 3 mtDNA-encoded subunits including COX1/COX2 (Ala-rich TM regions); COX-negative fibres on histochemistry"},
            {"term": "Symmetric CI≈CIV",    "definition": "MT-TA produces symmetric CI ≈ CIV deficiency — contrast with MT-TW (CIV preferentially worse due to UGA-recoding disruption)"},
            {"term": "mt-translation fingerprint", "definition": "Combined CI+CIV deficiency with CII NORMAL — characteristic of pathogenic mt-tRNA gene mutations"},
            {"term": "Ragged-Red Fibres",   "definition": "Subsarcolemmal mitochondrial accumulation on modified Gomori trichrome; SDH-positive/COX-negative on sequential histochemistry"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",                "definition": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive ophthalmoplegia; predominant MT-TA phenotype"},
            {"term": "KSS",                 "definition": "Kearns-Sayre Syndrome — CPEO + pigmentary retinopathy + cardiac conduction defect; often large mtDNA deletion spanning MT-TA"},
            {"term": "AARS2-ovario-leukodystrophy", "definition": "OMIM #615889 — AR AARS2 biallelic mutations → mt-Ala-tRNA aminoacylation failure → females: premature ovarian insufficiency + adult leukoencephalopathy; males: infantile cardiomyopathy — NOT adult CPEO"},
            {"term": "Premature Ovarian Insufficiency (POI)", "definition": "Key discriminator for AARS2 vs MT-TA: POI present = AARS2 (AR, WES); POI absent + maternal CPEO = MT-TA (mtDNA panel, L-strand QC)"},
            {"term": "Compound tRNA loss",  "definition": "Large deletions spanning MT-TW (H-strand 5512–5579) + MT-TA (L-strand 5587–5655): simultaneous loss of tRNA-Trp and tRNA-Ala decoding capacity"},
            {"term": "BTBGD / SLC19A3",    "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease — AR treatable Leigh mimic; mandatory exclusion before attributing Leigh MRI to MT-TA"},
        ],
        "ngs_definitions": [
            {"term": "L-strand encoding",   "definition": "MT-TA (rCRS 5587–5655) is L-strand encoded; standard NGS H-strand calls MISS variants; reverse-complement pipeline required"},
            {"term": "L-strand NGS pitfall","definition": "H-strand-only NGS pipelines fail to detect L-strand tRNA variants; clinically identical pitfall to MT-TQ (4329–4400), MT-TE (14674–14742), MT-TP (15956–16023), MT-ND6 (14149–14673)"},
            {"term": "Four-gene L-strand cluster", "definition": "MT-TA, MT-TN, MT-TC, MT-TY occupy rCRS 5587–5891 — all L-strand; the longest consecutive L-strand tRNA block in the human mitochondrial genome"},
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
            {"ref": "Dallabona-2014-BRAIN-AARS2",       "citation": "Dallabona C et al. (2014) Novel (ovario)leukodystrophy related to AARS2 mutations. Brain 137:2197–2209 (AARS2 ovario-leukodystrophy discovery)"},
            {"ref": "Lynch-2017-BRAIN-AARS2",           "citation": "Lynch DS et al. (2017) Ovarioleukodystrophy phenotype associated with biallelic mutations in AARS2. Brain 140:1261–1267"},
            {"ref": "Sprinzl-2005-NAR-tRNA",            "citation": "Sprinzl M & Vassilenko KS (2005) Compilation of tRNA sequences and sequences of tRNA genes. Nucleic Acids Res 33:D139–D140"},
            {"ref": "Rossmanith-1995-JBiolChem-tRNA",   "citation": "Rossmanith W et al. (1995) Processing of human mitochondrial tRNA precursors. J Biol Chem 270:12885–12891"},
        ],
    }
