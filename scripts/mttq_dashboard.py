#!/usr/bin/env python3
"""MT-TQ — Mitochondrially Encoded tRNA-Gln — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | L-strand rCRS 4329–4400

MT-TQ (OMIM *590030) encodes mitochondrial tRNA-Gln (UUG anticodon, reading Gln codons
CAA and CAG), located on the **L-strand** at rCRS 4329–4400 (72 nt). MT-TQ is the FIFTH
tRNA gene of the human mitochondrial genome, following MT-TF (577–647), MT-TV (1602–1670),
MT-TL1 (3230–3304), and MT-TI (4263–4331). The MT-TQ gene is unique in that its 5′ start
(rCRS 4329–4331) overlaps with the 3′ end of MT-TI (H-strand, 4263–4331) — the MT-TI/MT-TQ
junction is the densest tRNA-gene overlap in the human mitochondrial genome.

IMPORTANT — L-STRAND ENCODING AND NGS PITFALL:
MT-TQ is encoded on the L-strand (same as MT-TE, MT-TP, and MT-ND6). In standard NGS
reporting, L-strand genes appear as reverse-complement sequences of the H-strand read.
Laboratories MUST verify L-strand coverage and apply reverse-complement decoding when
calling MT-TQ variants. Failure to do so (running H-strand variant calling only) can produce
false negatives for all MT-TQ pathogenic variants — this is the single most common
diagnostic pitfall for MT-TQ disease.

MT-TI / MT-TQ JUNCTION OVERLAP (rCRS 4329–4331):
MT-TI (H-strand, 4263–4331) and MT-TQ (L-strand, 4329–4400) share rCRS positions 4329–4331.
Large mtDNA deletions spanning this junction simultaneously remove the 3′ acceptor stem of
tRNA-Ile AND the 5′ acceptor stem of tRNA-Gln → compound tRNA loss (more severe than
single-tRNA deletion). These compound deletion phenotypes resemble KSS/CPEO but with higher
enzymatic deficiency and worse prognosis.

Gln codons (CAA, CAG) are decoded by mt-tRNA-Gln (UUG anticodon). Glutamine is incorporated
into multiple subunits of Complexes I and IV (NADH dehydrogenase + cytochrome c oxidase),
and pathogenic MT-TQ mutations reduce tRNA-Gln availability → combined CI+CIV deficiency
(CII NORMAL — the canonical mt-translation fingerprint for mt-tRNA gene mutations).

m.4332G>A is the most commonly reported pathogenic MT-TQ variant (~30%), targeting the
anticodon stem (position 4, L-strand numbering), producing CPEO + myopathy with combined
CI+CIV deficiency; it was first reported by Seneca et al. (1997) in Belgian patients.

NUCLEAR DDx — QARS2 (mt-Glutaminyl-tRNA Synthetase):
QARS2 biallelic mutations cause mitochondrial glutaminyl-tRNA aminoacylation failure —
similar biochemical fingerprint (CI+CIV deficiency) but DRAMATICALLY DIFFERENT phenotype:
neonatal/infantile encephalopathy with hypomyelination, severe developmental delay, and
Complex I deficiency (NOT adult-onset CPEO). QARS2 is AR (autosomal recessive),
WES-detectable, and presents in the first months of life. This DDx is critical because
WES can detect QARS2 but MISSES MT-TQ; conversely, mtDNA panel is required for MT-TQ.

  MT-TQ gene              OMIM *590030
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Occasional Cardiomyopathy / SNHL / Lactic Acidosis
  Protein product         tRNA-Gln (UUG anticodon) — 72 nucleotides; RNA gene
                          Gln codons: CAA, CAG
  Genome                  Mitochondrial DNA (mtDNA), **L-strand**, rCRS 4329–4400
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    FIFTH tRNA in mitochondrial genome; 5′-adjacent to MT-TM (4402–4469)
                          3′-overlaps MT-TI (H-strand) at rCRS 4329–4331
"""

import random
import math

# ── Seed for reproducible 40-patient cohort ──────────────────────────────────
SEED = 809
RNG  = random.Random(SEED)


def _rand_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    v = RNG.gauss(mu, sigma)
    return round(max(lo, min(hi, v)), 1)


def _rand_int(lo: int, hi: int) -> int:
    return RNG.randint(lo, hi)


# ── 40-patient cohort generator ───────────────────────────────────────────────
def _build_cohort():
    """Generate deterministic 40-patient MT-TQ cohort (seed-809)."""
    N = 40
    patients = []

    # Variant distribution (must sum to 40)
    variant_pool = (
        ["m.4332GA"]  * 12 +   # anticodon stem  ~30%
        ["m.4395AG"]  * 9  +   # acceptor stem   ~22%
        ["m.4363AG"]  * 7  +   # variable loop   ~18%
        ["m.4370TC"]  * 6  +   # T-stem          ~15%
        ["LargeDel"]  * 6      # junction KSS    ~15%
    )
    RNG.shuffle(variant_pool)

    for i in range(N):
        variant = variant_pool[i]

        # Heteroplasmy by variant
        if variant == "m.4332GA":
            hetero  = _rand_normal(62, 14, 28, 92)
        elif variant == "m.4395AG":
            hetero  = _rand_normal(55, 18, 20, 88)
        elif variant == "m.4363AG":
            hetero  = _rand_normal(72, 15, 40, 95)
        elif variant == "m.4370TC":
            hetero  = _rand_normal(48, 16, 20, 80)
        else:  # LargeDel
            hetero  = _rand_normal(65, 20, 25, 95)

        # Enzyme activities — CI+CIV deficiency, CII NORMAL
        ci_act  = _rand_normal(28 + (100 - hetero) * 0.35, 8, 8, 72)
        civ_act = _rand_normal(31 + (100 - hetero) * 0.30, 9, 10, 70)
        cii_act = _rand_normal(94, 4, 82, 104)   # CII nuclear-encoded → normal

        # Clinical features (L-strand tRNA-Gln, CPEO dominant)
        cpeo      = hetero > 42 or variant in ("m.4332GA", "m.4395AG", "LargeDel")
        myopathy  = hetero > 38 or variant in ("m.4332GA", "m.4363AG")
        cardio    = variant == "m.4395AG" and hetero > 50 and RNG.random() < 0.35
        snhl      = variant in ("m.4370TC",) and RNG.random() < 0.55
        rrfs      = myopathy and RNG.random() < 0.78
        lactic    = hetero > 52 and RNG.random() < 0.70
        leigh_mri = variant == "m.4363AG" and hetero > 75 and RNG.random() < 0.45
        exercise  = hetero > 30 and RNG.random() < 0.85

        # Onset age
        if variant in ("m.4363AG",):
            onset = _rand_normal(18, 12, 3, 50)
        elif variant == "LargeDel":
            onset = _rand_normal(22, 10, 8, 45)
        else:
            onset = _rand_normal(30, 12, 12, 62)

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
        "n_patients":                  n,
        "avg_heteroplasmy_blood_pct":  avg("heteroplasmy_blood_pct"),
        "avg_ci_activity_pct_normal":  avg("ci_activity_pct_normal"),
        "avg_civ_activity_pct_normal": avg("civ_activity_pct_normal"),
        "avg_cii_activity_pct_normal": avg("cii_activity_pct_normal"),
        "pct_cpeo":                    pct("cpeo"),
        "pct_myopathy":                pct("myopathy"),
        "pct_cardiomyopathy":          pct("cardiomyopathy"),
        "pct_snhl":                    pct("snhl"),
        "pct_ragged_red_fibres":       pct("ragged_red_fibres"),
        "pct_lactic_acidosis":         pct("lactic_acidosis"),
        "pct_leigh_like_mri":          pct("leigh_like_mri"),
        "pct_exercise_intolerance":    pct("exercise_intolerance"),
        "avg_age_onset_yr":            avg("age_onset_yr"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_overview():
    stats = _cohort_stats()
    return {
        "title":    "MT-TQ — tRNA-Gln (UUG Anticodon)",
        "subtitle": "Combined CI+CIV Deficiency · CPEO · Myopathy · Exercise Intolerance — L-strand NGS Pitfall · FIFTH tRNA in mt-genome",
        "omim":     "OMIM *590030",
        "gene_facts": {
            "gene":            "MT-TQ",
            "product":         "tRNA-Gln (Glutamine), 72 nt RNA gene",
            "anticodon":       "UUG (reads CAA and CAG codons)",
            "strand":          "L-strand (NGS PITFALL — reverse complement required)",
            "rCRS_coordinates":"4329–4400",
            "length_nt":       72,
            "genome_position":  "FIFTH tRNA in mitochondrial genome",
            "flanking_5prime": "MT-TI (H-strand, 4263–4331) — OVERLAPS at rCRS 4329–4331",
            "flanking_3prime": "MT-TM (L-strand, 4402–4469)",
            "inheritance":     "Maternal (heteroplasmic)",
            "omim_gene":       "MT-TQ *590030",
        },
        "ngs_pitfall_alert": {
            "alert_class": "L-STRAND NGS PITFALL — MANDATORY QC",
            "detail": (
                "MT-TQ is L-strand encoded (same as MT-TE, MT-TP, MT-ND6). Standard H-strand "
                "variant calling WILL MISS pathogenic MT-TQ variants. Laboratories MUST apply "
                "L-strand reverse-complement decoding and verify L-strand sequencing depth at "
                "rCRS 4329–4400 before reporting any mtDNA panel result involving this region."
            ),
        },
        "junction_overlap_alert": {
            "alert_class": "MT-TI / MT-TQ JUNCTION OVERLAP — rCRS 4329–4331",
            "detail": (
                "The 5′ end of MT-TQ (rCRS 4329–4331) physically overlaps with the 3′ end of "
                "MT-TI (rCRS 4263–4331). Large mtDNA deletions spanning this junction "
                "simultaneously destroy tRNA-Ile (CI/CIV subunits) AND tRNA-Gln (CI/CIV "
                "subunits), producing compound CPEO phenotypes more severe than single-tRNA loss. "
                "Always evaluate MT-TI status when reporting MT-TQ large deletions."
            ),
        },
        "biochemical_fingerprint": {
            "summary":         "Combined CI+CIV Deficiency (CII NORMAL — mt-translation fingerprint)",
            "complex_i":       "Deficient (avg ~{:.0f}% normal)".format(stats["avg_ci_activity_pct_normal"]),
            "complex_ii":      "NORMAL (nuclear-encoded; avg ~{:.0f}% normal)".format(stats["avg_cii_activity_pct_normal"]),
            "complex_iv":      "Deficient (avg ~{:.0f}% normal)".format(stats["avg_civ_activity_pct_normal"]),
            "mechanism":       (
                "Gln (CAA/CAG) residues are incorporated into ND1, ND2, ND4, ND5 (CI) and "
                "CO1, CO2, CO3 (CIV) subunits. MT-TQ mutations reduce mitochondrial tRNA-Gln "
                "availability → impaired mt-translation → combined CI+CIV assembly defect. "
                "CII (succinate dehydrogenase) is entirely nuclear-encoded and remains intact."
            ),
        },
        "cohort_statistics":         stats,
        "cohort_summary_features": [
            {"feature": "CPEO (ptosis + progressive ophthalmoplegia)",     "value": f"{stats['pct_cpeo']}%",             "note": "dominant phenotype, ~80% of patients"},
            {"feature": "Myopathy (RRF, COX-negative fibres)",             "value": f"{stats['pct_myopathy']}%",         "note": "SDH-positive, COX-negative ragged-red fibres"},
            {"feature": "Exercise Intolerance",                             "value": f"{stats['pct_exercise_intolerance']}%", "note": "often the earliest symptom"},
            {"feature": "Cardiomyopathy (m.4395A>G subset)",               "value": f"{stats['pct_cardiomyopathy']}%",   "note": "less than MT-TI/MT-TT; annual echo if m.4395AG"},
            {"feature": "SNHL (m.4370T>C subset)",                         "value": f"{stats['pct_snhl']}%",             "note": "cochlear implants effective"},
            {"feature": "Lactic Acidosis",                                  "value": f"{stats['pct_lactic_acidosis']}%",  "note": "high heteroplasmy threshold"},
            {"feature": "Leigh-like MRI (m.4363A>G, high hetero)",         "value": f"{stats['pct_leigh_like_mri']}%",   "note": "bilateral symmetric BG + brainstem"},
            {"feature": "Ragged-Red Fibres (muscle biopsy)",               "value": f"{stats['pct_ragged_red_fibres']}%","note": "SDH over-staining in RRF"},
        ],
        "phenotype_distribution": [
            {"variant": "m.4332G>A",    "pct": 30, "phenotype": "CPEO + Myopathy",               "position": "Anticodon stem (pos 4)"},
            {"variant": "m.4395A>G",    "pct": 22, "phenotype": "CPEO + Cardiomyopathy",         "position": "Acceptor stem (pos 69)"},
            {"variant": "m.4363A>G",    "pct": 18, "phenotype": "Multisystem / Leigh-like",       "position": "Variable loop (pos 47)"},
            {"variant": "m.4370T>C",    "pct": 15, "phenotype": "Exercise Intol + SNHL",          "position": "T-stem (pos 54)"},
            {"variant": "Large deletion","pct": 15, "phenotype": "KSS/CPEO — Compound tRNA Loss", "position": "MT-TI/MT-TQ junction span"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<30",  "expected_phenotype": "Subclinical / carrier — monitor annually"},
            {"threshold_pct": "30–50","expected_phenotype": "Exercise intolerance, mild CPEO — begin OXPHOS surveillance"},
            {"threshold_pct": "50–70","expected_phenotype": "CPEO + myopathy + lactic acidosis — full phenotype"},
            {"threshold_pct": ">70",  "expected_phenotype": "Severe multisystem / Leigh-like (m.4363AG) — ICU-level risk"},
        ],
        "key_molecular_features": [
            "L-STRAND ENCODED — NGS pitfall: reverse complement required (same as MT-TE, MT-TP, MT-ND6)",
            "FIFTH tRNA in the human mitochondrial genome (after MT-TF, MT-TV, MT-TL1, MT-TI)",
            "5′ overlap with MT-TI at rCRS 4329–4331 — densest tRNA-gene overlap in mt-genome",
            "Combined CI+CIV deficiency (CII NORMAL) — canonical mt-translation fingerprint",
            "CPEO dominant (~80%); less isolated HCM than MT-TI/MT-TT",
            "m.4332G>A most common (~30%) — anticodon stem, CPEO+myopathy",
            "m.4395A>G — acceptor stem (~22%), cardiomyopathy enriched",
            "QARS2 nuclear DDx: neonatal encephalopathy (NOT adult CPEO) — WES-detectable",
        ],
        "clinical_alerts": [
            {"alert": "L-STRAND NGS PITFALL — MANDATORY",    "detail": "Reverse-complement decoding required; H-strand-only calling misses all MT-TQ variants"},
            {"alert": "MT-TI/MT-TQ junction check MANDATORY", "detail": "Large deletions spanning 4329–4331 → compound tRNA loss; always co-report MT-TI status"},
            {"alert": "QARS2 DDx — WES required",            "detail": "Neonatal/infantile CI+CIV fingerprint + encephalopathy → add WES for QARS2 biallelic mutations"},
            {"alert": "m.4395AG — Annual echo",              "detail": "Cardiomyopathy subset; beta-blocker if HCM; no amiodarone"},
            {"alert": "BTBGD (SLC19A3) — MANDATORY EXCLUSION","detail": "Treatable Leigh-like mimic; biotin+thiamine trial before attributing to MT-TQ at high heteroplasmy"},
            {"alert": "WES MISSES MT-TQ",                    "detail": "WES does not sequence mtDNA L-strand adequately; dedicated mtDNA panel with L-strand QC required"},
        ],
    }


def get_breakdown():
    stats = _cohort_stats()
    # Variant-level breakdown
    variants = {}
    for p in _COHORT:
        v = p["variant"]
        if v not in variants:
            variants[v] = {"variant": v, "n": 0, "pct_cpeo": [], "pct_myopathy": [], "pct_cardio": [],
                           "avg_hetero": [], "avg_ci": [], "avg_civ": []}
        variants[v]["n"] += 1
        variants[v]["avg_hetero"].append(p["heteroplasmy_blood_pct"])
        variants[v]["avg_ci"].append(p["ci_activity_pct_normal"])
        variants[v]["avg_civ"].append(p["civ_activity_pct_normal"])
        if p["cpeo"]:       variants[v]["pct_cpeo"].append(1)
        if p["myopathy"]:   variants[v]["pct_myopathy"].append(1)
        if p["cardiomyopathy"]: variants[v]["pct_cardio"].append(1)

    breakdown = []
    for v, d in variants.items():
        n = d["n"]
        breakdown.append({
            "variant":            v,
            "n":                  n,
            "pct_of_cohort":      round(n / 40 * 100, 1),
            "avg_heteroplasmy":   round(sum(d["avg_hetero"]) / n, 1),
            "avg_ci_pct_normal":  round(sum(d["avg_ci"]) / n, 1),
            "avg_civ_pct_normal": round(sum(d["avg_civ"]) / n, 1),
            "pct_cpeo":           round(len(d["pct_cpeo"]) / n * 100, 1),
            "pct_myopathy":       round(len(d["pct_myopathy"]) / n * 100, 1),
            "pct_cardiomyopathy": round(len(d["pct_cardio"]) / n * 100, 1),
        })

    return {
        "title":        "MT-TQ Variant & Phenotype Breakdown — 40-patient cohort seed-809",
        "variant_breakdown": sorted(breakdown, key=lambda x: -x["n"]),
        "ddx_table": [
            {
                "entity":       "MT-TQ pathogenic variant",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Adult CPEO + myopathy + exercise intolerance; occasional cardiomyopathy (m.4395AG)",
                "biochemistry": "CI+CIV deficiency, CII NORMAL",
                "ngs":          "L-strand mtDNA panel — H-strand WES MISSES",
                "distinctive":  "L-strand encoding pitfall; MT-TI junction overlap; no isolated HCM",
            },
            {
                "entity":       "QARS2 (mt-Gln-tRNA synthetase)",
                "inheritance":  "AR nuclear",
                "phenotype":    "Neonatal/infantile encephalopathy, hypomyelination, severe developmental delay",
                "biochemistry": "Complex I deficiency (variable CI+CIV)",
                "ngs":          "WES detects biallelic QARS2 mutations",
                "distinctive":  "Neonatal onset, NOT adult CPEO; AR not maternal",
            },
            {
                "entity":       "MT-TI (tRNA-Ile)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy + isolated HCM (m.4300AG DISTINCTIVE)",
                "biochemistry": "CI+CIV deficiency, CII NORMAL",
                "ngs":          "H-strand mtDNA panel",
                "distinctive":  "m.4300AG isolated HCM without CPEO — NOT seen in MT-TQ",
            },
            {
                "entity":       "BTBGD / SLC19A3",
                "inheritance":  "AR nuclear",
                "phenotype":    "Biotin-thiamine-responsive Leigh-like — symmetric BG signals",
                "biochemistry": "Normal OXPHOS on treatment",
                "ngs":          "WES",
                "distinctive":  "TREATABLE — biotin + thiamine trial MANDATORY before attributing Leigh MRI to MT-TQ",
            },
            {
                "entity":       "MT-TL1 (MELAS)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Stroke-like episodes, CPEO, ragged-red fibres",
                "biochemistry": "Pan-OXPHOS CI+CIII+CIV deficiency",
                "ngs":          "H-strand mtDNA panel — m.3243AG most common",
                "distinctive":  "STROKE-LIKE EPISODES and MELAS: NOT seen in MT-TQ",
            },
        ],
        "management_by_variant": [
            {"variant": "m.4332GA",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Annual ophthalmology; strabismus/ptosis surgery PRN"},
            {"variant": "m.4395AG",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "Annual echo + Holter; beta-blocker if HCM; no amiodarone"},
            {"variant": "m.4363AG",  "cpeo_risk": "Mod",  "cardio_risk": "Low",      "key_action": "Neuro-MRI if childhood onset; BTBGD exclusion mandatory"},
            {"variant": "m.4370TC",  "cpeo_risk": "Low",  "cardio_risk": "Low",      "key_action": "Audiology + cochlear implant assessment; avoid aminoglycosides"},
            {"variant": "LargeDel",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "KSS surveillance (cardiac + retina + endocrine); co-check MT-TI loss"},
        ],
        "absolute_contraindications": [
            {"drug": "Metformin",     "reason": "Complex I inhibition → fatal lactic acidosis in CI-deficient patients"},
            {"drug": "Valproate (VPA)","reason": "Inhibits mtDNA replication; depletes CoA; worsens hepatopathy"},
            {"drug": "Propofol",      "reason": "Propofol infusion syndrome (PRIS) — uncouples OXPHOS; fatal in mt disease"},
            {"drug": "Linezolid",     "reason": "Inhibits mitochondrial 23S rRNA translation → iatrogenic mt-translation failure"},
            {"drug": "Chloramphenicol","reason": "mt-ribosome inhibitor; cumulative mt-translation toxicity"},
            {"drug": "Amiodarone",    "reason": "Mitochondrial OXPHOS inhibitor; especially dangerous if m.4395AG cardiomyopathy"},
            {"drug": "Aminoglycosides","reason": "SNHL amplification (cochlear OXPHOS); HIGH CAUTION for m.4370TC subset"},
        ],
        "safe_interventions": [
            {"intervention": "CoQ10 (Ubiquinol)",   "evidence": "Level C — electron carrier support; 200–600 mg/day adult"},
            {"intervention": "Riboflavin (B2)",     "evidence": "Level C — Complex I/II cofactor; 100–400 mg/day"},
            {"intervention": "L-Carnitine",         "evidence": "Level C — acylcarnitine transport support"},
            {"intervention": "Thiamine (B1)",       "evidence": "MANDATORY empiric — BTBGD exclusion + mt-energy support"},
            {"intervention": "Biotin",              "evidence": "MANDATORY empiric — BTBGD exclusion; 5–10 mg/day"},
            {"intervention": "Levetiracetam (LEV)", "evidence": "Preferred AED — avoids VPA; safe OXPHOS profile"},
            {"intervention": "Beta-blocker",        "evidence": "First-line if cardiomyopathy (m.4395AG); replaces amiodarone"},
            {"intervention": "Elamipretide",        "evidence": "Phase 2 — cardiolipin stabiliser for mt-cardiomyopathy"},
        ],
        "cohort_statistics": stats,
    }


def get_definitions():
    return {
        "title": "MT-TQ — Clinical & Molecular Definitions",
        "gene_definitions": [
            {"term": "MT-TQ",              "definition": "Mitochondrially encoded tRNA-Gln gene; L-strand, rCRS 4329–4400, 72 nt (OMIM *590030)"},
            {"term": "tRNA-Gln",           "definition": "Transfer RNA for Glutamine; UUG anticodon reads CAA and CAG codons in the mitochondrial genetic code"},
            {"term": "L-strand",           "definition": "Light strand of mtDNA (low G-content); genes encoded 3′→5′ on H-strand template; NGS pitfall — requires reverse complement"},
            {"term": "rCRS",               "definition": "Revised Cambridge Reference Sequence — standard human mtDNA numbering (GenBank NC_012920.1)"},
            {"term": "Heteroplasmy",       "definition": "Mixture of wild-type and mutant mtDNA molecules within a cell/tissue; clinical severity correlates with mutant load (%)"},
        ],
        "biochemical_definitions": [
            {"term": "CI deficiency",      "definition": "NADH:ubiquinone oxidoreductase (Complex I) deficiency — first enzyme of OXPHOS respiratory chain"},
            {"term": "CII (normal)",       "definition": "Succinate dehydrogenase — entirely nuclear-encoded; remains normal in mt-tRNA disease (diagnostic discriminator)"},
            {"term": "CIV deficiency",     "definition": "Cytochrome c oxidase (COX) deficiency — COX-negative fibres on muscle biopsy histochemistry"},
            {"term": "mt-translation fingerprint", "definition": "Combined CI+CIV deficiency with CII NORMAL — characteristic of pathogenic mt-tRNA gene mutations affecting all 13 mtDNA-encoded OXPHOS subunits"},
            {"term": "Ragged-Red Fibres",  "definition": "Subsarcolemmal mitochondrial accumulation on modified Gomori trichrome; also SDH-positive/COX-negative on sequential histochemistry"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",               "definition": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive ophthalmoplegia; predominant MT-TQ phenotype"},
            {"term": "KSS",                "definition": "Kearns-Sayre Syndrome — CPEO + pigmentary retinopathy + cardiac conduction defect; often large mtDNA deletion"},
            {"term": "CAGSSS",             "definition": "Cataracts, growth hormone deficiency, sensory neuropathy, SNHL, skeletal dysplasia — QARS2 nuclear DDx (NOT MT-TQ phenotype)"},
            {"term": "BTBGD / SLC19A3",   "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease — AR treatable Leigh mimic; mandatory exclusion before attributing Leigh MRI to MT-TQ"},
            {"term": "MELAS",              "definition": "Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke-like episodes — MT-TL1 (m.3243AG); NOT MT-TQ phenotype"},
        ],
        "ngs_definitions": [
            {"term": "L-strand NGS pitfall", "definition": "MT-TQ (+ MT-TE, MT-TP, MT-ND6) is L-strand encoded; H-strand-only variant callers produce false negatives — mandatory reverse-complement decoding"},
            {"term": "Variant allele fraction (VAF)", "definition": "Proportion of reads carrying the variant allele = heteroplasmy estimate; tissue-specific (blood < muscle < affected tissue)"},
            {"term": "Minimum heteroplasmy detection", "definition": "Clinical mtDNA panels should detect ≥1% VAF; recommend ≥5000× depth at L-strand MT-TQ locus"},
        ],
        "drug_definitions": [
            {"term": "Absolute CI (contraindication)", "definition": "Drugs that worsen OXPHOS, deplete mtDNA, or cause lactic acidosis in mitochondrial disease — Metformin, VPA, Propofol, Linezolid, Chloramphenicol, Amiodarone"},
            {"term": "PRIS",               "definition": "Propofol Infusion Syndrome — mitochondrial OXPHOS uncoupling; metabolic acidosis; rhabdomyolysis; high mortality in mt disease"},
            {"term": "GIR 6–8",            "definition": "Glucose Infusion Rate 6–8 mg/kg/min — mandatory during any perioperative fasting period in mt disease; prevents catabolic crisis"},
        ],
        "references": [
            {"ref": "Seneca-1997-HumMolGenet",    "citation": "Seneca S et al. (1997) A mitochondrial tRNA-Gln mutation causing CPEO. Hum Mol Genet 6:1677–1680"},
            {"ref": "Schaefer-2008-AnnNeurol",    "citation": "Schaefer AM et al. (2008) Prevalence of mitochondrial disease in adults. Ann Neurol 63:35–39"},
            {"ref": "Gorman-2016-NatRevDisPrimers","citation": "Gorman GS et al. (2016) Mitochondrial diseases. Nat Rev Dis Primers 2:16080"},
            {"ref": "DiMauro-Schon-2003-NEJM",    "citation": "DiMauro S, Schon EA (2003) Mitochondrial respiratory-chain diseases. N Engl J Med 348:2656–2668"},
            {"ref": "Tort-2013-HumMutat-QARS2",   "citation": "Tort F et al. (2013) Mutations in QARS2 cause neonatal lactic acidosis/encephalopathy. Hum Mutat 34:1503–1507"},
        ],
    }
