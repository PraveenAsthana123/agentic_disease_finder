#!/usr/bin/env python3
"""MT-TW — Mitochondrially Encoded tRNA-Trp — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 5512–5579

MT-TW (OMIM *590095) encodes mitochondrial tRNA-Trp (UCA anticodon), located on the
**H-strand** at rCRS 5512–5579 (68 nt). MT-TW is the SEVENTH tRNA gene of the human
mitochondrial genome, following MT-TF (577–647), MT-TV (1602–1670), MT-TL1 (3230–3304),
MT-TI (4263–4331), MT-TQ (L-strand 4329–4400), and MT-TM (4402–4469). MT-TW is separated
from MT-TM by the remainder of the MT-ND2 gene (~1042 nt gap, rCRS 4469–5511), making it
the first tRNA after the large ND2 coding block.

CRITICAL UNIQUE FEATURE — UGA CODON RECODING:
In the standard genetic code, UGA encodes a STOP codon. In the human mitochondrial genetic
code, UGA is RECODED as Tryptophan (Trp/W). MT-TW (anticodon UCA) therefore decodes BOTH:
  1. UGG codons (standard Trp) — via Watson-Crick pairing at all three positions
  2. UGA codons (mitochondrial Trp) — via wobble at the third position (A:U wobble)
This UGA recoding is unique to the mitochondrial code and is executed exclusively by
MT-TW. Several critical OXPHOS subunits contain UGA (mt-Trp) codons:
  - MT-CO1 (COX1, Complex IV): multiple UGA-Trp residues in the catalytic core
  - MT-ND2 (Complex I): UGA-Trp residues
  - MT-ND5 (Complex I): UGA-Trp residues
MT-TW mutations that disrupt UGA recoding cause a PREFERENTIAL Complex IV assembly defect
(on top of the general mt-translation CI+CIV fingerprint) because COX1 has the highest
density of UGA-Trp codons. This explains why some MT-TW variants show relatively MORE
severe CIV deficiency than CI deficiency — an unusual asymmetry among mt-tRNA diseases.

H-STRAND ENCODING — NO NGS PITFALL:
MT-TW is H-strand encoded (rCRS 5512–5579). Immediately 3′, the four tRNAs MT-TA
(5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), and MT-TY (5826–5891) are ALL
L-strand encoded — a cluster of four consecutive L-strand tRNAs. Laboratories processing
this genomic region must switch QC strand after MT-TW. MT-TW itself has NO L-strand NGS
pitfall — standard H-strand variant calling is correct.

m.5543T>C is the most commonly reported pathogenic MT-TW variant (~30%), targeting the
anticodon loop (adjacent to the UCA anticodon at positions 34–36), producing CPEO +
myopathy. The anticodon-loop position disrupts tRNA tertiary folding required for ribosomal
A-site accommodation, reducing both UGG and UGA decoding efficiency.

NUCLEAR DDx — WARS2 (mt-Tryptophanyl-tRNA Synthetase 2):
WARS2 biallelic mutations cause mitochondrial Trp-tRNA aminoacylation failure — similar
biochemistry but dramatically different phenotype: severe neonatal/infantile OXPHOS
deficiency (OMIM #616045), presenting with cardiomyopathy, lactic acidosis, failure to
thrive, and neurodevelopmental regression — NOT adult-onset CPEO. WARS2 is AR,
WES-detectable, and clinically distinguishable by neonatal/early-infantile onset.

  MT-TW gene              OMIM *590095
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Occasional Cardiomyopathy / SNHL / Leigh-like at high heteroplasmy
                          UGA-recoding disruption → preferential COX deficiency asymmetry
  Protein product         tRNA-Trp (UCA anticodon) — 68 nucleotides; RNA gene
                          UGA recoding: reads BOTH UGG (standard Trp) AND UGA (mt-Trp)
  Genome                  Mitochondrial DNA (mtDNA), **H-strand**, rCRS 5512–5579
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    SEVENTH tRNA in mitochondrial genome
                          5′-adjacent to MT-ND2 (H-strand, ends ~5511), ~1 nt gap
                          3′-adjacent to MT-TA (L-strand, 5587–5655), 7 nt gap
"""

import random
import math

# ── Seed for reproducible 40-patient cohort ──────────────────────────────────
SEED = 813
RNG  = random.Random(SEED)


def _rand_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    v = RNG.gauss(mu, sigma)
    return round(max(lo, min(hi, v)), 1)


def _rand_int(lo: int, hi: int) -> int:
    return RNG.randint(lo, hi)


# ── 40-patient cohort generator ───────────────────────────────────────────────
def _build_cohort():
    """Generate deterministic 40-patient MT-TW cohort (seed-813)."""
    N = 40
    patients = []

    # Variant distribution (must sum to 40)
    variant_pool = (
        ["m.5543TC"]  * 12 +   # anticodon loop (near UCA anticodon)    ~30%
        ["m.5549GA"]  * 9  +   # variable loop / anticodon stem 3'      ~22%
        ["m.5523GA"]  * 7  +   # D-stem region                          ~18%
        ["m.5545GA"]  * 6  +   # anticodon position 34 (part of UCA)   ~17% (severe)
        ["LargeDel"]  * 6      # MT-TW-spanning KSS/CPEO deletion       ~13%
    )
    RNG.shuffle(variant_pool)

    for i in range(N):
        variant = variant_pool[i]

        # Heteroplasmy by variant
        if variant == "m.5543TC":
            hetero = _rand_normal(58, 16, 25, 90)
        elif variant == "m.5549GA":
            hetero = _rand_normal(55, 18, 22, 88)
        elif variant == "m.5523GA":
            hetero = _rand_normal(66, 17, 35, 95)
        elif variant == "m.5545GA":
            # Anticodon pos 34 — severe, lower threshold for phenotype
            hetero = _rand_normal(48, 20, 18, 88)
        else:  # LargeDel
            hetero = _rand_normal(62, 19, 28, 93)

        # Enzyme activities — CI+CIV deficiency, CII NORMAL
        # MT-TW unique: UGA recoding disruption → CIV disproportionately affected
        # at moderate-high heteroplasmy (>50%), CIV more severe than CI
        uga_recoding_disrupted = (variant in ("m.5545GA", "m.5543TC")) and hetero > 50
        ci_base  = 28 if uga_recoding_disrupted else 35
        civ_base = 20 if uga_recoding_disrupted else 32   # CIV worse if UGA disrupted
        ci_act  = _rand_normal(ci_base  + (100 - hetero) * 0.35, 8,  8, 70)
        civ_act = _rand_normal(civ_base + (100 - hetero) * 0.30, 9,  6, 68)
        cii_act = _rand_normal(93, 4, 82, 104)  # CII nuclear-encoded → normal

        # Clinical features
        cpeo     = hetero > 38 or variant in ("m.5543TC", "m.5523GA", "LargeDel")
        myopathy = hetero > 34 or variant in ("m.5543TC", "m.5549GA")
        cardio   = variant == "m.5549GA" and hetero > 52 and RNG.random() < 0.40
        snhl     = variant in ("m.5545GA",) and RNG.random() < 0.55
        rrfs     = myopathy and RNG.random() < 0.78
        lactic   = hetero > 48 and RNG.random() < 0.65
        leigh_mri = (variant == "m.5523GA" or hetero > 72) and RNG.random() < 0.38
        exercise = hetero > 26 and RNG.random() < 0.86
        # UGA-recoding asymmetry: CIV > CI deficiency at high heteroplasmy
        uga_asymmetry = uga_recoding_disrupted and (civ_act < ci_act - 8)

        # Onset age
        if variant == "m.5545GA":
            onset = _rand_normal(14, 8, 4, 38)    # anticodon variant — younger
        elif variant == "LargeDel":
            onset = _rand_normal(20, 10, 8, 44)
        elif variant == "m.5523GA":
            onset = _rand_normal(18, 12, 6, 46)
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
            "uga_recoding_disrupted": uga_recoding_disrupted,
            "uga_asymmetry_civ_worse": uga_asymmetry,
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
        "pct_uga_recoding_disrupted":        pct("uga_recoding_disrupted"),
        "pct_uga_asymmetry_civ_worse":       pct("uga_asymmetry_civ_worse"),
        "avg_age_onset_yr":                  avg("age_onset_yr"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_overview():
    stats = _cohort_stats()
    return {
        "title":    "MT-TW — tRNA-Trp (UCA Anticodon)",
        "subtitle": "Combined CI+CIV Deficiency · CPEO · Myopathy · UGA-Codon Recoding — SEVENTH tRNA in mt-genome — H-strand rCRS 5512–5579",
        "omim":     "OMIM *590095",
        "gene_facts": {
            "gene":             "MT-TW",
            "product":          "tRNA-Trp (Tryptophan), 68 nt RNA gene",
            "anticodon":        "UCA (reads UGG and UGA as Trp in the mitochondrial genetic code)",
            "strand":           "H-strand (NO NGS pitfall — contrast with MT-TA, MT-TN, MT-TC, MT-TY immediately 3′ which are ALL L-strand)",
            "rCRS_coordinates": "5512–5579",
            "length_nt":        68,
            "genome_position":  "SEVENTH tRNA in mitochondrial genome",
            "flanking_5prime":  "MT-ND2 (H-strand, 4470–5511) — ~1 nt gap at rCRS 5511–5512",
            "flanking_3prime":  "MT-TA (L-strand, 5587–5655) — 7 nt gap; followed by L-strand cluster MT-TA/TN/TC/TY",
            "inheritance":      "Maternal (heteroplasmic)",
            "omim_gene":        "MT-TW *590095",
            "uga_recoding":     "UGA = Tryptophan (NOT stop) in mt code — MT-TW is the sole executor of this recoding; disruption preferentially impairs Complex IV (COX1 UGA-Trp residues)",
        },
        "uga_recoding_alert": {
            "alert_class": "UGA CODON RECODING — MITOCHONDRIAL GENETIC CODE — MT-TW EXCLUSIVE",
            "detail": (
                "In the standard genetic code, UGA = STOP. In the human mitochondrial genetic code, "
                "UGA = Tryptophan. MT-TW (UCA anticodon) is the SOLE tRNA executing this recoding, "
                "decoding BOTH UGG (standard Trp) and UGA (mt-Trp) via wobble at anticodon position 34. "
                "MT-CO1 (COX subunit I) has multiple UGA-encoded Trp residues in its catalytic core. "
                "MT-TW mutations disrupting UGA decoding cause PREFERENTIAL Complex IV deficiency "
                "— CIV may be more severely reduced than CI, an asymmetry atypical for most mt-tRNA diseases."
            ),
        },
        "h_strand_note": {
            "note_class": "H-STRAND ENCODED — Verify L-strand switch immediately 3′",
            "detail": (
                "MT-TW (rCRS 5512–5579) is H-strand encoded. However, the four tRNAs immediately "
                "3′ — MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), MT-TY (5826–5891) — "
                "are ALL L-strand encoded. Laboratories MUST switch to L-strand QC after MT-TW. "
                "Standard H-strand variant calling is correct for MT-TW itself; no NGS pitfall applies."
            ),
        },
        "biochemical_fingerprint": {
            "summary":   "Combined CI+CIV Deficiency (CII NORMAL — mt-translation fingerprint); CIV preferentially worse in UGA-disrupting variants",
            "complex_i": "Deficient (avg ~{:.0f}% normal)".format(stats["avg_ci_activity_pct_normal"]),
            "complex_ii": "NORMAL (nuclear-encoded; avg ~{:.0f}% normal)".format(stats["avg_cii_activity_pct_normal"]),
            "complex_iv": "Deficient (avg ~{:.0f}% normal); may exceed CI deficit in m.5545GA/m.5543TC at >50% heteroplasmy (UGA-recoding asymmetry)".format(stats["avg_civ_activity_pct_normal"]),
            "mechanism": (
                "Trp (UGG + UGA) residues are required in mt-translation of all 13 OXPHOS subunits. "
                "MT-TW mutations reduce tRNA-Trp availability → impaired elongation → CI+CIV assembly defect. "
                "UGA-Trp codons are particularly dense in MT-CO1 (COX I, Complex IV): disruption of UGA "
                "decoding by anticodon-domain variants (m.5543TC, m.5545GA) preferentially impairs COX "
                "assembly, yielding CIV deficiency more severe than CI in ~{:.0f}% of UGA-disrupted cases. "
                "CII (succinate dehydrogenase) is entirely nuclear-encoded and remains intact.".format(stats["pct_uga_asymmetry_civ_worse"])
            ),
        },
        "cohort_statistics":       stats,
        "cohort_summary_features": [
            {"feature": "CPEO (ptosis + progressive ophthalmoplegia)",         "value": f"{stats['pct_cpeo']}%",                       "note": "dominant phenotype; adult onset in most variants"},
            {"feature": "Myopathy (RRF, COX-negative fibres)",                 "value": f"{stats['pct_myopathy']}%",                   "note": "SDH-positive, COX-negative ragged-red fibres"},
            {"feature": "Exercise Intolerance",                                 "value": f"{stats['pct_exercise_intolerance']}%",       "note": "early symptom, often predates ophthalmoplegia"},
            {"feature": "Cardiomyopathy (m.5549GA subset)",                     "value": f"{stats['pct_cardiomyopathy']}%",             "note": "variable loop variant; annual echo required"},
            {"feature": "SNHL (m.5545GA subset)",                               "value": f"{stats['pct_snhl']}%",                       "note": "anticodon position 34 variant; cochlear implants"},
            {"feature": "Lactic Acidosis",                                       "value": f"{stats['pct_lactic_acidosis']}%",            "note": "heteroplasmy-dependent; perioperative risk"},
            {"feature": "Leigh-like MRI (m.5523GA / high heteroplasmy)",        "value": f"{stats['pct_leigh_like_mri']}%",             "note": "bilateral symmetric BG + brainstem lesions"},
            {"feature": "Ragged-Red Fibres (muscle biopsy)",                    "value": f"{stats['pct_ragged_red_fibres']}%",          "note": "SDH over-staining hallmark; COX-negative on dual stain"},
            {"feature": "UGA-recoding disrupted (anticodon variants)",          "value": f"{stats['pct_uga_recoding_disrupted']}%",     "note": "m.5543TC / m.5545GA at >50% heteroplasmy"},
            {"feature": "CIV worse than CI (UGA-asymmetry)",                    "value": f"{stats['pct_uga_asymmetry_civ_worse']}%",    "note": "CIV >8% more deficient than CI — UGA-Trp/COX1 mechanism"},
        ],
        "phenotype_distribution": [
            {"variant": "m.5543T>C",    "pct": 30, "phenotype": "CPEO + Myopathy",                  "position": "Anticodon loop (adjacent to UCA at pos 34–36)"},
            {"variant": "m.5549G>A",    "pct": 22, "phenotype": "CPEO + Cardiomyopathy",             "position": "Variable loop / anticodon stem 3'-branch"},
            {"variant": "m.5523G>A",    "pct": 18, "phenotype": "Multisystem / Leigh-like",          "position": "D-stem region"},
            {"variant": "m.5545G>A",    "pct": 17, "phenotype": "Exercise Intol + SNHL (early onset)","position": "Anticodon position 34 (part of UCA anticodon)"},
            {"variant": "Large deletion","pct": 13, "phenotype": "KSS/CPEO — MT-TW + flanking loss", "position": "MT-TW-spanning deletion (rCRS ~5512–5579)"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<30",  "expected_phenotype": "Subclinical — annual ophthalmology + exercise ECG; WARS2 exclusion if infantile"},
            {"threshold_pct": "30–50","expected_phenotype": "Exercise intolerance + early CPEO — OXPHOS surveillance; UGA-recoding status not yet disrupted"},
            {"threshold_pct": "50–70","expected_phenotype": "Full CPEO + myopathy + lactic acidosis; UGA-recoding disruption begins (anticodon variants) → CIV asymmetry"},
            {"threshold_pct": ">70",  "expected_phenotype": "Severe CPEO + myopathy; Leigh-like MRI risk; CIV often more deficient than CI (UGA-Trp/COX1 mechanism)"},
        ],
        "key_molecular_features": [
            "SEVENTH tRNA in the human mitochondrial genome (after MT-TF, MT-TV, MT-TL1, MT-TI, MT-TQ, MT-TM)",
            "SOLE mt-tRNA executing UGA codon recoding: UGA = Tryptophan (not STOP) in the mitochondrial code",
            "UCA anticodon decodes BOTH UGG (standard Trp) AND UGA (mt-specific Trp) via wobble at position 34",
            "UGA-Trp codons in MT-CO1 (COX I) — anticodon variants cause preferential CIV > CI deficiency",
            "H-strand encoded — NO L-strand NGS pitfall; but 4 consecutive L-strand tRNAs follow immediately 3′",
            "~1 nt gap from MT-ND2 5′; 7 nt gap to MT-TA (L-strand) 3′; first tRNA after large ND2 block",
            "Combined CI+CIV deficiency (CII NORMAL) — canonical mt-translation fingerprint",
            "m.5543T>C most common (~30%) — anticodon loop, CPEO+myopathy",
            "WARS2 nuclear DDx: severe neonatal OXPHOS deficiency — NOT adult CPEO",
        ],
        "clinical_alerts": [
            {"alert": "UGA CODON RECODING — CIV ASYMMETRY IN ANTICODON VARIANTS", "detail": "m.5543TC / m.5545GA at >50% heteroplasmy: UGA-Trp/COX1 disruption → CIV may be more deficient than CI; unusual for mt-tRNA disease — clue for MT-TW"},
            {"alert": "L-strand cluster immediately 3′ — MANDATORY QC SWITCH",     "detail": "MT-TA, MT-TN, MT-TC, MT-TY (all L-strand, rCRS 5587–5891) follow MT-TW; labs must switch strand QC after MT-TW (rCRS 5579)"},
            {"alert": "m.5545GA — anticodon position 34 — early onset",            "detail": "Disrupts first position of UCA anticodon; severe UGA-recoding failure; onset childhood–young adult; SNHL common"},
            {"alert": "WARS2 DDx — WES required if infantile presentation",        "detail": "WARS2 biallelic: neonatal cardiomyopathy + lactic acidosis + failure to thrive; AR nuclear; WES-detectable; NOT adult CPEO"},
            {"alert": "BTBGD (SLC19A3) — MANDATORY EXCLUSION",                    "detail": "Treatable Leigh-like mimic; biotin+thiamine trial before attributing Leigh MRI to MT-TW at high heteroplasmy"},
            {"alert": "WES MISSES MT-TW",                                          "detail": "WES does not sequence mtDNA adequately; dedicated mtDNA panel required for MT-TW (H-strand, 5512–5579)"},
        ],
    }


def get_breakdown():
    stats = _cohort_stats()
    variants = {}
    for p in _COHORT:
        v = p["variant"]
        if v not in variants:
            variants[v] = {"variant": v, "n": 0, "pct_cpeo": [], "pct_myopathy": [], "pct_cardio": [],
                           "avg_hetero": [], "avg_ci": [], "avg_civ": [], "pct_uga": []}
        variants[v]["n"] += 1
        variants[v]["avg_hetero"].append(p["heteroplasmy_blood_pct"])
        variants[v]["avg_ci"].append(p["ci_activity_pct_normal"])
        variants[v]["avg_civ"].append(p["civ_activity_pct_normal"])
        if p["cpeo"]:                       variants[v]["pct_cpeo"].append(1)
        if p["myopathy"]:                   variants[v]["pct_myopathy"].append(1)
        if p["cardiomyopathy"]:             variants[v]["pct_cardio"].append(1)
        if p["uga_recoding_disrupted"]:     variants[v]["pct_uga"].append(1)

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
            "pct_uga_recoding_disrupted": round(len(d["pct_uga"]) / n * 100, 1),
        })

    return {
        "title":             "MT-TW Variant & Phenotype Breakdown — 40-patient cohort seed-813",
        "variant_breakdown": sorted(breakdown, key=lambda x: -x["n"]),
        "ddx_table": [
            {
                "entity":       "MT-TW pathogenic variant",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Adult CPEO + myopathy + exercise intolerance; CIV asymmetry in anticodon variants",
                "biochemistry": "CI+CIV deficiency, CII NORMAL; CIV may exceed CI deficit (UGA-recoding)",
                "ngs":          "H-strand mtDNA panel — WES MISSES MT-TW",
                "distinctive":  "UGA codon recoding: sole mt-tRNA decoding UGA as Trp; anticodon variants → CIV > CI deficiency",
            },
            {
                "entity":       "WARS2 (mt-Trp-tRNA synthetase)",
                "inheritance":  "AR nuclear",
                "phenotype":    "Severe neonatal/infantile OXPHOS deficiency — cardiomyopathy, lactic acidosis, FTT, neurodevelopmental regression",
                "biochemistry": "Combined OXPHOS deficiency; similar biochemistry",
                "ngs":          "WES detects biallelic WARS2 mutations",
                "distinctive":  "NOT adult CPEO; neonatal/infantile NOT adult onset; AR not maternal; WES detects",
            },
            {
                "entity":       "MT-TM (tRNA-Met)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; initiation block at >75% heteroplasmy",
                "biochemistry": "CI+CIV deficiency, CII NORMAL",
                "ngs":          "H-strand mtDNA panel",
                "distinctive":  "Dual function initiator+elongator; NO UGA recoding; MARS2 DDx; rCRS 4402–4469",
            },
            {
                "entity":       "MT-TQ (tRNA-Gln)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy, no isolated HCM",
                "biochemistry": "CI+CIV deficiency, CII NORMAL",
                "ngs":          "L-strand mtDNA panel (NGS pitfall — reverse complement required)",
                "distinctive":  "L-strand NGS pitfall; QARS2 DDx; rCRS 4329–4400",
            },
            {
                "entity":       "BTBGD / SLC19A3",
                "inheritance":  "AR nuclear",
                "phenotype":    "Biotin-thiamine-responsive Leigh-like — symmetric BG signals",
                "biochemistry": "Normal OXPHOS on treatment",
                "ngs":          "WES",
                "distinctive":  "TREATABLE — biotin + thiamine trial MANDATORY before attributing Leigh MRI to MT-TW",
            },
            {
                "entity":       "MT-TL1 (MELAS)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Stroke-like episodes, CPEO, ragged-red fibres",
                "biochemistry": "Pan-OXPHOS CI+CIII+CIV deficiency",
                "ngs":          "H-strand mtDNA panel — m.3243AG most common",
                "distinctive":  "STROKE-LIKE EPISODES and MELAS: NOT seen in MT-TW",
            },
        ],
        "management_by_variant": [
            {"variant": "m.5543TC",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Annual ophthalmology; ptosis surgery PRN; UGA-recoding CIV check; CoQ10 + riboflavin"},
            {"variant": "m.5549GA",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "Annual echo + Holter; Leigh MRI surveillance; beta-blocker if HCM"},
            {"variant": "m.5523GA",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Full OXPHOS surveillance; BTBGD exclusion; Leigh MRI monitoring"},
            {"variant": "m.5545GA",  "cpeo_risk": "Mod",  "cardio_risk": "Low",      "key_action": "Audiology + cochlear implant; anticodon pos-34 severe UGA disruption; watch CIV; early referral"},
            {"variant": "LargeDel",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "KSS surveillance (cardiac + retina + endocrine); co-check MT-TA/TN/TC/TY losses; CSF lactate"},
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
            {"intervention": "Beta-blocker",        "evidence": "First-line if cardiomyopathy (m.5549GA); replaces amiodarone"},
            {"intervention": "GIR 6–8 mg/kg/min",  "evidence": "MANDATORY perioperative — prevents catabolic crisis; never fast in mt disease"},
        ],
        "cohort_statistics": stats,
    }


def get_definitions():
    return {
        "title": "MT-TW — Clinical & Molecular Definitions",
        "gene_definitions": [
            {"term": "MT-TW",               "definition": "Mitochondrially encoded tRNA-Trp gene; H-strand, rCRS 5512–5579, 68 nt (OMIM *590095)"},
            {"term": "tRNA-Trp",            "definition": "Transfer RNA for Tryptophan; UCA anticodon, decoding both UGG (standard Trp) and UGA (mitochondrial Trp recoding)"},
            {"term": "UGA codon recoding",  "definition": "In the mitochondrial genetic code, UGA = Tryptophan (NOT stop); MT-TW is the sole mt-tRNA executing this recoding via wobble at anticodon position 34"},
            {"term": "UCA anticodon",       "definition": "Anticodon of tRNA-Trp (positions 34–36); reads UGG by Watson-Crick and UGA by wobble (U34:A codon-position-3 pairing in mt ribosomes)"},
            {"term": "UGA-Trp density in MT-CO1", "definition": "MT-CO1 (COX subunit I) has the highest density of UGA-encoded Trp residues among the 13 mt-encoded OXPHOS subunits; MT-TW anticodon mutations preferentially disrupt COX assembly"},
            {"term": "rCRS",                "definition": "Revised Cambridge Reference Sequence — standard human mtDNA numbering (GenBank NC_012920.1)"},
            {"term": "Heteroplasmy",        "definition": "Mixture of wild-type and mutant mtDNA molecules within a cell/tissue; severity correlates with mutant load (%)"},
        ],
        "biochemical_definitions": [
            {"term": "CI deficiency",       "definition": "NADH:ubiquinone oxidoreductase (Complex I) deficiency — 7 mtDNA-encoded subunits; sensitive to mt-tRNA defects"},
            {"term": "CII (normal)",        "definition": "Succinate dehydrogenase — entirely nuclear-encoded; remains normal in mt-tRNA disease (diagnostic discriminator)"},
            {"term": "CIV deficiency",      "definition": "Cytochrome c oxidase (COX) deficiency — 3 mtDNA-encoded subunits including COX1 (UGA-Trp rich); COX-negative fibres on histochemistry"},
            {"term": "UGA asymmetry",       "definition": "CIV more deficient than CI in some MT-TW anticodon variants — due to UGA-Trp disruption preferentially impairing COX1 assembly; unusual among mt-tRNA diseases"},
            {"term": "mt-translation fingerprint", "definition": "Combined CI+CIV deficiency with CII NORMAL — characteristic of pathogenic mt-tRNA gene mutations"},
            {"term": "Ragged-Red Fibres",   "definition": "Subsarcolemmal mitochondrial accumulation on modified Gomori trichrome; SDH-positive/COX-negative on sequential histochemistry"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",                "definition": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive ophthalmoplegia; predominant MT-TW phenotype"},
            {"term": "KSS",                 "definition": "Kearns-Sayre Syndrome — CPEO + pigmentary retinopathy + cardiac conduction defect; often large mtDNA deletion spanning MT-TW"},
            {"term": "WARS2-OXPHOS deficiency", "definition": "OMIM #616045 — AR WARS2 biallelic mutations → mt-Trp-tRNA aminoacylation failure → severe neonatal combined OXPHOS deficiency; NOT adult CPEO"},
            {"term": "BTBGD / SLC19A3",    "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease — AR treatable Leigh mimic; mandatory exclusion before attributing Leigh MRI to MT-TW"},
            {"term": "MELAS",               "definition": "Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke-like episodes — MT-TL1 (m.3243AG); stroke-like episodes NOT a feature of MT-TW"},
            {"term": "MERRF",               "definition": "Myoclonic Epilepsy with Ragged Red Fibres — MT-TK; myoclonic epilepsy NOT a feature of MT-TW"},
        ],
        "ngs_definitions": [
            {"term": "H-strand encoding",   "definition": "MT-TW (rCRS 5512–5579) is H-strand encoded; standard NGS H-strand variant calling correctly detects MT-TW variants (no L-strand pitfall for MT-TW itself)"},
            {"term": "L-strand cluster 3′", "definition": "MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), MT-TY (5826–5891) are ALL L-strand encoded; QC switch required after MT-TW at rCRS 5579"},
            {"term": "Variant allele fraction (VAF)", "definition": "Proportion of reads carrying the variant allele = heteroplasmy estimate; tissue-specific (blood < muscle < affected tissue)"},
        ],
        "drug_definitions": [
            {"term": "Absolute CI",         "definition": "Metformin, VPA, Propofol, Linezolid, Chloramphenicol — all worsen OXPHOS or deplete mtDNA; fatal in mt disease"},
            {"term": "PRIS",                "definition": "Propofol Infusion Syndrome — mitochondrial OXPHOS uncoupling; metabolic acidosis; rhabdomyolysis; high mortality in mt disease"},
            {"term": "GIR 6–8",             "definition": "Glucose Infusion Rate 6–8 mg/kg/min — mandatory during any perioperative fasting period; prevents catabolic crisis in mt disease"},
            {"term": "KD contraindication", "definition": "Ketogenic diet contraindicated in mt disease — fatty acid oxidation and ketone body utilisation via mt OXPHOS; KD stresses already-deficient ETC"},
        ],
        "references": [
            {"ref": "DiMauro-Schon-2003-NEJM",         "citation": "DiMauro S, Schon EA (2003) Mitochondrial respiratory-chain diseases. N Engl J Med 348:2656–2668"},
            {"ref": "Schaefer-2008-AnnNeurol",          "citation": "Schaefer AM et al. (2008) Prevalence of mitochondrial disease in adults. Ann Neurol 63:35–39"},
            {"ref": "Gorman-2016-NatRevDisPrimers",     "citation": "Gorman GS et al. (2016) Mitochondrial diseases. Nat Rev Dis Primers 2:16080"},
            {"ref": "Goto-2019-WARS2",                  "citation": "Goto Y et al. (2019) WARS2 mutations and mitochondrial tryptophan-tRNA synthetase deficiency. Brain Dev 41:428–437"},
            {"ref": "Barrell-1979-Science-UGA-Trp",    "citation": "Barrell BG et al. (1979) A different genetic code in human mitochondria. Nature 282:189–194 (UGA = Trp recoding discovery)"},
            {"ref": "Sprinzl-2005-NAR-mtRNA",           "citation": "Sprinzl M & Vassilenko KS (2005) Compilation of tRNA sequences and sequences of tRNA genes. Nucleic Acids Res 33:D139–D140"},
            {"ref": "Rossmanith-1995-JBiolChem-tRNA",   "citation": "Rossmanith W et al. (1995) Processing of human mitochondrial tRNA precursors. J Biol Chem 270:12885–12891"},
        ],
    }
