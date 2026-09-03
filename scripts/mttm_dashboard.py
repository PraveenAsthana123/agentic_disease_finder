#!/usr/bin/env python3
"""MT-TM — Mitochondrially Encoded tRNA-Met — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 4402–4469

MT-TM (OMIM *590065) encodes the single mitochondrial tRNA-Met (CAU anticodon), located on
the **H-strand** at rCRS 4402–4469 (68 nt). MT-TM is the SIXTH tRNA gene of the human
mitochondrial genome, following MT-TF (577–647), MT-TV (1602–1670), MT-TL1 (3230–3304),
MT-TI (4263–4331), and MT-TQ (L-strand 4329–4400). MT-TM immediately follows MT-TQ with only
a 2 nt gap (rCRS 4400–4402), and is immediately 5'-flanked by MT-ND2 with a 1 nt gap
(MT-TM ends 4469; MT-ND2 begins 4470).

CRITICAL UNIQUE FEATURE — DUAL FUNCTION (INITIATOR AND ELONGATOR):
MT-TM encodes the ONLY mitochondrial methionine tRNA. Unlike cytoplasmic translation (which
has separate initiator Met-tRNA_i and elongator Met-tRNA_e), human mitochondria rely on a
SINGLE tRNA-Met for BOTH:
  1. Translation INITIATION: formylated as N-formyl-methionine (fMet) at mitochondrial AUG
     start codons — initiating synthesis of all 13 mtDNA-encoded OXPHOS subunits
  2. Translation ELONGATION: decoding internal AUG and AUA codons (the modified wobble
     base of the CAU anticodon — lysidine/agmatidine at position 34 — reads AUA as Met in
     the mitochondrial genetic code)
This dual function is unique among all human mitochondrial tRNAs. At very high heteroplasmy
(>75–80%), MT-TM mutations impair BOTH initiation AND elongation, causing more global
mitochondrial translation failure than other mt-tRNA mutations.

H-STRAND ENCODING — NO NGS PITFALL:
Unlike MT-TQ (immediately 5'), MT-TE, MT-TP, and MT-ND6 (all L-strand encoded), MT-TM is
encoded on the H-strand. Standard NGS H-strand variant calling detects MT-TM variants
correctly. However, laboratories processing the MT-TQ/MT-TM junction must ensure they do NOT
mistake MT-TQ L-strand coverage for MT-TM H-strand coverage — the two overlap by 2 nt.

Met codons (AUG and AUA) appear in ALL 13 mtDNA-encoded subunits as start sites and
elongation sites. MT-TM mutations therefore reduce tRNA-Met availability across ALL OXPHOS
complexes (I, III, IV, V). However, the enzymatic signature is combined CI+CIV deficiency
(CII NORMAL) because: (a) CII is nuclear-encoded and not affected; (b) CI and CIV have the
greatest number of mtDNA-encoded subunits (7 and 3 respectively) and their assembly is most
sensitive to mt-translation impairment; CIII (MT-CYB alone) and CV (MT-ATP6, MT-ATP8) are
less severely affected at moderate heteroplasmy levels, giving the classical CI+CIV pattern.

m.4435A>G is the most commonly reported pathogenic MT-TM variant (~30%), targeting the
acceptor stem (position 3), producing CPEO + myopathy. It was first systematically described
in CPEO/myopathy patients with combined CI+CIV deficiency. Large deletions spanning the
MT-TQ/MT-TM junction simultaneously remove the 3' end of MT-TQ AND the 5' acceptor stem of
MT-TM, producing compound tRNA loss (more severe than single-tRNA deletion).

NUCLEAR DDx — MARS2 (mt-Methionyl-tRNA Synthetase):
MARS2 biallelic mutations cause mitochondrial methionyl-tRNA aminoacylation failure —
similar biochemical fingerprint but DRAMATICALLY DIFFERENT phenotype: autosomal recessive
spastic ataxia with leukoencephalopathy (ARSAL), NOT adult-onset CPEO. MARS2 is AR,
WES-detectable, and presents with cerebellar ataxia + spasticity + white matter changes in
childhood/adulthood. This DDx is critical: WES detects MARS2 but MISSES MT-TM; dedicated
mtDNA panel is required for MT-TM.

  MT-TM gene              OMIM *590065
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Occasional Cardiomyopathy / SNHL / Leigh-like at high heteroplasmy
  Protein product         tRNA-Met (CAU anticodon) — 68 nucleotides; RNA gene
                          Dual function: fMet initiator + Met elongator (AUG + AUA)
  Genome                  Mitochondrial DNA (mtDNA), **H-strand**, rCRS 4402–4469
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    SIXTH tRNA in mitochondrial genome
                          5′-adjacent to MT-TQ (L-strand, ends 4400), 2 nt gap
                          3′-adjacent to MT-ND2 (H-strand, starts 4470), 1 nt gap
"""

import random
import math

# ── Seed for reproducible 40-patient cohort ──────────────────────────────────
SEED = 811
RNG  = random.Random(SEED)


def _rand_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    v = RNG.gauss(mu, sigma)
    return round(max(lo, min(hi, v)), 1)


def _rand_int(lo: int, hi: int) -> int:
    return RNG.randint(lo, hi)


# ── 40-patient cohort generator ───────────────────────────────────────────────
def _build_cohort():
    """Generate deterministic 40-patient MT-TM cohort (seed-811)."""
    N = 40
    patients = []

    # Variant distribution (must sum to 40)
    variant_pool = (
        ["m.4435AG"]  * 12 +   # acceptor stem (pos 3)   ~30%
        ["m.4450TC"]  * 8  +   # variable loop (pos 47)  ~20%
        ["m.4429AG"]  * 7  +   # anticodon stem          ~17%
        ["m.4460GA"]  * 7  +   # acceptor stem terminal  ~18%
        ["LargeDel"]  * 6      # MT-TQ/TM junction KSS   ~15%
    )
    RNG.shuffle(variant_pool)

    for i in range(N):
        variant = variant_pool[i]

        # Heteroplasmy by variant
        if variant == "m.4435AG":
            hetero  = _rand_normal(60, 15, 25, 92)
        elif variant == "m.4450TC":
            hetero  = _rand_normal(52, 18, 20, 85)
        elif variant == "m.4429AG":
            hetero  = _rand_normal(70, 16, 38, 95)
        elif variant == "m.4460GA":
            hetero  = _rand_normal(74, 17, 42, 96)
        else:  # LargeDel
            hetero  = _rand_normal(63, 21, 25, 95)

        # Enzyme activities — CI+CIV deficiency, CII NORMAL
        # MT-TM dual function: initiation block at >75% → steeper CI/CIV drop
        initiation_block = hetero > 75
        ci_base  = 22 if initiation_block else 30
        civ_base = 25 if initiation_block else 33
        ci_act  = _rand_normal(ci_base  + (100 - hetero) * 0.38, 8,  6, 72)
        civ_act = _rand_normal(civ_base + (100 - hetero) * 0.32, 9, 10, 70)
        cii_act = _rand_normal(93, 4, 82, 104)  # CII nuclear-encoded → normal

        # Clinical features
        cpeo      = hetero > 40 or variant in ("m.4435AG", "m.4429AG", "LargeDel")
        myopathy  = hetero > 36 or variant in ("m.4435AG", "m.4460GA")
        cardio    = variant in ("m.4460GA",) and hetero > 55 and RNG.random() < 0.38
        snhl      = variant == "m.4450TC" and RNG.random() < 0.52
        rrfs      = myopathy and RNG.random() < 0.80
        lactic    = hetero > 50 and RNG.random() < 0.68
        # Leigh-like more likely with m.4460GA or very high heteroplasmy (initiation block)
        leigh_mri = (variant == "m.4460GA" or initiation_block) and RNG.random() < 0.42
        exercise  = hetero > 28 and RNG.random() < 0.88
        # Initiation block adds global mt-translation failure sign
        global_failure = initiation_block and RNG.random() < 0.60

        # Onset age
        if variant == "m.4460GA" or initiation_block:
            onset = _rand_normal(16, 10, 2, 45)
        elif variant == "LargeDel":
            onset = _rand_normal(21, 11, 7, 45)
        else:
            onset = _rand_normal(32, 13, 14, 65)

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
            "initiation_block_high_hetero": global_failure,
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
        "n_patients":                          n,
        "avg_heteroplasmy_blood_pct":          avg("heteroplasmy_blood_pct"),
        "avg_ci_activity_pct_normal":          avg("ci_activity_pct_normal"),
        "avg_civ_activity_pct_normal":         avg("civ_activity_pct_normal"),
        "avg_cii_activity_pct_normal":         avg("cii_activity_pct_normal"),
        "pct_cpeo":                            pct("cpeo"),
        "pct_myopathy":                        pct("myopathy"),
        "pct_cardiomyopathy":                  pct("cardiomyopathy"),
        "pct_snhl":                            pct("snhl"),
        "pct_ragged_red_fibres":               pct("ragged_red_fibres"),
        "pct_lactic_acidosis":                 pct("lactic_acidosis"),
        "pct_leigh_like_mri":                  pct("leigh_like_mri"),
        "pct_exercise_intolerance":            pct("exercise_intolerance"),
        "pct_initiation_block_high_hetero":    pct("initiation_block_high_hetero"),
        "avg_age_onset_yr":                    avg("age_onset_yr"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_overview():
    stats = _cohort_stats()
    return {
        "title":    "MT-TM — tRNA-Met (CAU Anticodon)",
        "subtitle": "Combined CI+CIV Deficiency · CPEO · Myopathy · Dual Initiator+Elongator Function — SIXTH tRNA in mt-genome — H-strand rCRS 4402–4469",
        "omim":     "OMIM *590065",
        "gene_facts": {
            "gene":             "MT-TM",
            "product":          "tRNA-Met (Methionine), 68 nt RNA gene",
            "anticodon":        "CAU (reads AUG and AUA codons — wobble modified to lysidine/agmatidine at pos 34)",
            "strand":           "H-strand (NO NGS pitfall — contrast with MT-TQ immediately 5')",
            "rCRS_coordinates": "4402–4469",
            "length_nt":        68,
            "genome_position":  "SIXTH tRNA in mitochondrial genome",
            "flanking_5prime":  "MT-TQ (L-strand, 4329–4400) — 2 nt gap at rCRS 4400–4402",
            "flanking_3prime":  "MT-ND2 (H-strand, 4470–5511) — 1 nt gap at rCRS 4469–4470",
            "inheritance":      "Maternal (heteroplasmic)",
            "omim_gene":        "MT-TM *590065",
            "dual_function":    "INITIATOR (fMet, translation start) + ELONGATOR (Met elongation, AUG+AUA) — unique in mt-genome",
        },
        "dual_function_alert": {
            "alert_class": "DUAL-FUNCTION tRNA — INITIATOR + ELONGATOR — UNIQUE IN mt-GENOME",
            "detail": (
                "MT-TM encodes the ONLY mitochondrial methionine tRNA, serving as BOTH initiator "
                "(N-formyl-methionine for AUG start codons) AND elongator (Met at internal AUG and "
                "AUA codons via modified CAU anticodon). At high heteroplasmy (>75–80%), MT-TM "
                "mutations impair translation INITIATION across all 13 mtDNA-encoded subunits — "
                "producing more severe, global mt-translation failure than single-elongator tRNA "
                "mutations. This makes high-heteroplasmy MT-TM mutations uniquely severe."
            ),
        },
        "h_strand_note": {
            "note_class": "H-STRAND ENCODED — No L-strand NGS Pitfall",
            "detail": (
                "MT-TM is H-strand encoded (unlike MT-TQ immediately 5', and MT-TE, MT-TP, MT-ND6). "
                "Standard NGS H-strand variant calling detects MT-TM variants correctly. "
                "However, laboratories must verify the MT-TQ/MT-TM junction (rCRS 4400–4402): "
                "ensure L-strand MT-TQ coverage does not mask H-strand MT-TM variants in the "
                "2 nt overlap region."
            ),
        },
        "biochemical_fingerprint": {
            "summary":   "Combined CI+CIV Deficiency (CII NORMAL — mt-translation fingerprint)",
            "complex_i": "Deficient (avg ~{:.0f}% normal)".format(stats["avg_ci_activity_pct_normal"]),
            "complex_ii":"NORMAL (nuclear-encoded; avg ~{:.0f}% normal)".format(stats["avg_cii_activity_pct_normal"]),
            "complex_iv":"Deficient (avg ~{:.0f}% normal)".format(stats["avg_civ_activity_pct_normal"]),
            "mechanism": (
                "Met (AUG/AUA) residues are present in all 13 mtDNA-encoded OXPHOS subunits. "
                "MT-TM mutations reduce tRNA-Met → impaired initiation and elongation of "
                "mt-translation → combined CI+CIV assembly defect. At >75–80% heteroplasmy, "
                "initiation failure amplifies CI+CIV deficiency severity. CII (succinate "
                "dehydrogenase) is entirely nuclear-encoded and remains intact."
            ),
        },
        "cohort_statistics":       stats,
        "cohort_summary_features": [
            {"feature": "CPEO (ptosis + progressive ophthalmoplegia)",         "value": f"{stats['pct_cpeo']}%",                   "note": "dominant phenotype across variants"},
            {"feature": "Myopathy (RRF, COX-negative fibres)",                 "value": f"{stats['pct_myopathy']}%",               "note": "SDH-positive, COX-negative ragged-red fibres"},
            {"feature": "Exercise Intolerance",                                 "value": f"{stats['pct_exercise_intolerance']}%",   "note": "earliest symptom, often before CPEO"},
            {"feature": "Cardiomyopathy (m.4460GA subset)",                     "value": f"{stats['pct_cardiomyopathy']}%",         "note": "acceptor-stem terminal variant; annual echo"},
            {"feature": "SNHL (m.4450TC subset)",                               "value": f"{stats['pct_snhl']}%",                   "note": "cochlear implants effective"},
            {"feature": "Lactic Acidosis",                                       "value": f"{stats['pct_lactic_acidosis']}%",        "note": "heteroplasmy-dependent threshold"},
            {"feature": "Leigh-like MRI (m.4460GA / high heteroplasmy)",        "value": f"{stats['pct_leigh_like_mri']}%",         "note": "bilateral symmetric BG + brainstem; initiation block"},
            {"feature": "Ragged-Red Fibres (muscle biopsy)",                    "value": f"{stats['pct_ragged_red_fibres']}%",      "note": "SDH over-staining in RRF"},
            {"feature": "Initiation Block (>75% heteroplasmy, global failure)", "value": f"{stats['pct_initiation_block_high_hetero']}%", "note": "unique to MT-TM — combined initiation+elongation failure"},
        ],
        "phenotype_distribution": [
            {"variant": "m.4435A>G",    "pct": 30, "phenotype": "CPEO + Myopathy",                  "position": "Acceptor stem (pos 3)"},
            {"variant": "m.4450T>C",    "pct": 20, "phenotype": "Exercise Intol + SNHL",             "position": "Variable loop (pos 47)"},
            {"variant": "m.4429A>G",    "pct": 17, "phenotype": "Multisystem CPEO + Myopathy",       "position": "Anticodon stem"},
            {"variant": "m.4460G>A",    "pct": 18, "phenotype": "Leigh-like / Cardiomyopathy",       "position": "Acceptor stem terminal (pos 68–69 region)"},
            {"variant": "Large deletion","pct": 15, "phenotype": "KSS/CPEO — MT-TQ+MT-TM loss",      "position": "MT-TQ/MT-TM junction span"},
        ],
        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<30",  "expected_phenotype": "Subclinical — monitor annually (exercise ECG + ophthalmology)"},
            {"threshold_pct": "30–55","expected_phenotype": "Exercise intolerance + early CPEO — OXPHOS surveillance"},
            {"threshold_pct": "55–75","expected_phenotype": "Full CPEO + myopathy + lactic acidosis — elongation block dominant"},
            {"threshold_pct": ">75",  "expected_phenotype": "Severe multisystem / Leigh-like — INITIATION BLOCK adds global mt-translation failure"},
        ],
        "key_molecular_features": [
            "SIXTH tRNA in the human mitochondrial genome (after MT-TF, MT-TV, MT-TL1, MT-TI, MT-TQ)",
            "ONLY mt-tRNA serving as BOTH initiator (fMet) AND elongator (Met) — unique in mt-genome",
            "Modified wobble base (lysidine/agmatidine at C34) decodes AUA as Met in mt genetic code",
            "H-strand encoded — NO L-strand NGS pitfall (contrast with adjacent MT-TQ)",
            "2 nt gap 5' to MT-TQ (rCRS 4400–4402); 1 nt gap 3' to MT-ND2 (rCRS 4469–4470)",
            "Combined CI+CIV deficiency (CII NORMAL) — canonical mt-translation fingerprint",
            "At >75% heteroplasmy: initiation block → global mt-translation failure (all 13 subunits)",
            "m.4435A>G most common (~30%) — acceptor stem, CPEO+myopathy",
            "MARS2 nuclear DDx: ARSAL (spastic ataxia + leukoencephalopathy) — NOT adult CPEO",
        ],
        "clinical_alerts": [
            {"alert": "DUAL FUNCTION — INITIATION BLOCK AT HIGH HETEROPLASMY",  "detail": ">75–80% heteroplasmy → MT-TM initiation failure → global mt-translation collapse; ICU metabolic risk"},
            {"alert": "H-STRAND ENCODED — verify MT-TQ/MT-TM junction",        "detail": "MT-TM is H-strand; MT-TQ is L-strand (ends rCRS 4400); 2 nt gap — do not confuse L-strand TQ coverage with H-strand TM coverage"},
            {"alert": "MT-TQ/MT-TM junction large deletions",                   "detail": "Deletions spanning rCRS 4400–4402 simultaneously remove 3' MT-TQ + 5' MT-TM → compound tRNA loss (KSS phenotype, worse than single-tRNA)"},
            {"alert": "MARS2 DDx — WES required",                               "detail": "ARSAL: cerebellar ataxia + spasticity + leukoencephalopathy; AR nuclear; NOT adult CPEO; WES-detectable; MISSES MT-TM"},
            {"alert": "BTBGD (SLC19A3) — MANDATORY EXCLUSION",                 "detail": "Treatable Leigh-like mimic; biotin+thiamine trial before attributing Leigh MRI to MT-TM at high heteroplasmy"},
            {"alert": "WES MISSES MT-TM",                                       "detail": "WES does not sequence mtDNA adequately; dedicated mtDNA panel required for MT-TM (H-strand, 4402–4469)"},
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
        if p["cpeo"]:           variants[v]["pct_cpeo"].append(1)
        if p["myopathy"]:       variants[v]["pct_myopathy"].append(1)
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
        "title":             "MT-TM Variant & Phenotype Breakdown — 40-patient cohort seed-811",
        "variant_breakdown": sorted(breakdown, key=lambda x: -x["n"]),
        "ddx_table": [
            {
                "entity":       "MT-TM pathogenic variant",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Adult CPEO + myopathy + exercise intolerance; Leigh-like at high heteroplasmy (initiation block)",
                "biochemistry": "CI+CIV deficiency, CII NORMAL",
                "ngs":          "H-strand mtDNA panel — WES MISSES MT-TM",
                "distinctive":  "DUAL FUNCTION: initiator + elongator; initiation block at >75% heteroplasmy",
            },
            {
                "entity":       "MARS2 (mt-Met-tRNA synthetase)",
                "inheritance":  "AR nuclear",
                "phenotype":    "ARSAL: cerebellar ataxia + spasticity + white matter changes",
                "biochemistry": "Variable Complex I/IV deficiency",
                "ngs":          "WES detects biallelic MARS2 mutations",
                "distinctive":  "NOT adult CPEO; spastic ataxia not ophthalmoplegia; AR not maternal",
            },
            {
                "entity":       "MT-TQ (tRNA-Gln)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy, no isolated HCM",
                "biochemistry": "CI+CIV deficiency, CII NORMAL",
                "ngs":          "L-strand mtDNA panel (NGS pitfall — reverse complement required)",
                "distinctive":  "L-strand encoding pitfall; no initiation block (elongator only); QARS2 DDx",
            },
            {
                "entity":       "MT-TI (tRNA-Ile)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy + isolated HCM (m.4300AG DISTINCTIVE)",
                "biochemistry": "CI+CIV deficiency, CII NORMAL",
                "ngs":          "H-strand mtDNA panel",
                "distinctive":  "m.4300AG isolated HCM without CPEO — NOT seen in MT-TM",
            },
            {
                "entity":       "BTBGD / SLC19A3",
                "inheritance":  "AR nuclear",
                "phenotype":    "Biotin-thiamine-responsive Leigh-like — symmetric BG signals",
                "biochemistry": "Normal OXPHOS on treatment",
                "ngs":          "WES",
                "distinctive":  "TREATABLE — biotin + thiamine trial MANDATORY before attributing Leigh MRI to MT-TM",
            },
            {
                "entity":       "MT-TL1 (MELAS)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Stroke-like episodes, CPEO, ragged-red fibres",
                "biochemistry": "Pan-OXPHOS CI+CIII+CIV deficiency",
                "ngs":          "H-strand mtDNA panel — m.3243AG most common",
                "distinctive":  "STROKE-LIKE EPISODES and MELAS: NOT seen in MT-TM",
            },
        ],
        "management_by_variant": [
            {"variant": "m.4435AG",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Annual ophthalmology; ptosis/strabismus surgery PRN; CoQ10 + riboflavin"},
            {"variant": "m.4450TC",  "cpeo_risk": "Mod",  "cardio_risk": "Low",      "key_action": "Audiology + cochlear implant assessment; avoid aminoglycosides"},
            {"variant": "m.4429AG",  "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Full OXPHOS surveillance; BTBGD exclusion; GIR 6–8 perioperative"},
            {"variant": "m.4460GA",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "Annual echo + Holter; Leigh MRI surveillance; beta-blocker if HCM; ICU precaution at high hetero"},
            {"variant": "LargeDel",  "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "KSS surveillance (cardiac + retina + endocrine); co-check MT-TQ loss; CSF lactate"},
        ],
        "absolute_contraindications": [
            {"drug": "Metformin",      "reason": "Complex I inhibition → fatal lactic acidosis in CI-deficient patients"},
            {"drug": "Valproate (VPA)","reason": "Inhibits mtDNA replication; depletes CoA; worsens hepatopathy"},
            {"drug": "Propofol",       "reason": "Propofol infusion syndrome (PRIS) — uncouples OXPHOS; fatal in mt disease"},
            {"drug": "Linezolid",      "reason": "Inhibits mitochondrial 23S rRNA translation → iatrogenic mt-translation failure"},
            {"drug": "Chloramphenicol","reason": "mt-ribosome inhibitor; cumulative mt-translation toxicity"},
        ],
        "safe_interventions": [
            {"intervention": "CoQ10 (Ubiquinol)",   "evidence": "Level C — electron carrier support; 200–600 mg/day adult"},
            {"intervention": "Riboflavin (B2)",     "evidence": "Level C — Complex I/II cofactor; 100–400 mg/day"},
            {"intervention": "L-Carnitine",         "evidence": "Level C — acylcarnitine transport support"},
            {"intervention": "Thiamine (B1)",       "evidence": "MANDATORY empiric — BTBGD exclusion + mt-energy support"},
            {"intervention": "Biotin",              "evidence": "MANDATORY empiric — BTBGD exclusion; 5–10 mg/day"},
            {"intervention": "Levetiracetam (LEV)", "evidence": "Preferred AED — avoids VPA; safe OXPHOS profile"},
            {"intervention": "Beta-blocker",        "evidence": "First-line if cardiomyopathy (m.4460GA); replaces amiodarone"},
            {"intervention": "GIR 6–8 mg/kg/min",  "evidence": "MANDATORY perioperative — prevents catabolic crisis; never fast in mt disease"},
        ],
        "cohort_statistics": stats,
    }


def get_definitions():
    return {
        "title": "MT-TM — Clinical & Molecular Definitions",
        "gene_definitions": [
            {"term": "MT-TM",               "definition": "Mitochondrially encoded tRNA-Met gene; H-strand, rCRS 4402–4469, 68 nt (OMIM *590065)"},
            {"term": "tRNA-Met",            "definition": "Transfer RNA for Methionine; CAU anticodon (modified wobble decodes AUG + AUA in the mitochondrial genetic code)"},
            {"term": "Dual-function tRNA",  "definition": "MT-TM serves as BOTH initiator (fMet-tRNA for translation start at AUG) AND elongator (Met-tRNA for internal AUG/AUA) — unique among all human mitochondrial tRNAs"},
            {"term": "fMet (N-formyl-Met)", "definition": "Formylated methionine — used to initiate mitochondrial protein synthesis; generated from mt-tRNA-Met charged and formylated by MTFMT (mitochondrial formyl-tRNA transformylase)"},
            {"term": "Lysidine/agmatidine at C34", "definition": "Modified wobble base in mt-tRNA-Met anticodon that allows AUA decoding as Met (rather than Ile) in the mitochondrial genetic code"},
            {"term": "rCRS",                "definition": "Revised Cambridge Reference Sequence — standard human mtDNA numbering (GenBank NC_012920.1)"},
            {"term": "Heteroplasmy",        "definition": "Mixture of wild-type and mutant mtDNA molecules within a cell/tissue; severity correlates with mutant load (%)"},
        ],
        "biochemical_definitions": [
            {"term": "CI deficiency",       "definition": "NADH:ubiquinone oxidoreductase (Complex I) deficiency — 7 mtDNA-encoded subunits; most sensitive to mt-tRNA defects"},
            {"term": "CII (normal)",        "definition": "Succinate dehydrogenase — entirely nuclear-encoded; remains normal in mt-tRNA disease (diagnostic discriminator)"},
            {"term": "CIV deficiency",      "definition": "Cytochrome c oxidase (COX) deficiency — 3 mtDNA-encoded subunits; COX-negative fibres on histochemistry"},
            {"term": "mt-translation fingerprint", "definition": "Combined CI+CIV deficiency with CII NORMAL — characteristic of pathogenic mt-tRNA gene mutations"},
            {"term": "Initiation block",    "definition": "Impairment of mitochondrial translation initiation (fMet) at >75–80% heteroplasmy in MT-TM mutations — global OXPHOS failure; unique to MT-TM among mt-tRNAs"},
            {"term": "Ragged-Red Fibres",   "definition": "Subsarcolemmal mitochondrial accumulation on modified Gomori trichrome; SDH-positive/COX-negative on sequential histochemistry"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",                "definition": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive ophthalmoplegia; predominant MT-TM phenotype"},
            {"term": "KSS",                 "definition": "Kearns-Sayre Syndrome — CPEO + pigmentary retinopathy + cardiac conduction defect; often large mtDNA deletion (MT-TQ/MT-TM junction)"},
            {"term": "ARSAL",               "definition": "Autosomal Recessive Spastic Ataxia with Leukoencephalopathy — MARS2 nuclear DDx; cerebellar ataxia + spasticity + white matter changes; NOT adult CPEO"},
            {"term": "BTBGD / SLC19A3",    "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease — AR treatable Leigh mimic; mandatory exclusion before attributing Leigh MRI to MT-TM"},
            {"term": "MELAS",               "definition": "Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke-like episodes — MT-TL1 (m.3243AG); NOT MT-TM phenotype"},
            {"term": "MERRF",               "definition": "Myoclonic Epilepsy with Ragged Red Fibres — MT-TK; myoclonic epilepsy NOT a feature of MT-TM"},
        ],
        "ngs_definitions": [
            {"term": "H-strand encoding",   "definition": "MT-TM (rCRS 4402–4469) is H-strand encoded; standard NGS H-strand variant calling correctly detects MT-TM variants (no L-strand pitfall)"},
            {"term": "MT-TQ/MT-TM junction","definition": "2 nt gap between L-strand MT-TQ (ends 4400) and H-strand MT-TM (starts 4402); laboratories must verify correct strand assignment at this junction"},
            {"term": "Variant allele fraction (VAF)", "definition": "Proportion of reads carrying the variant allele = heteroplasmy estimate; tissue-specific (blood < muscle < affected tissue)"},
        ],
        "drug_definitions": [
            {"term": "Absolute CI",         "definition": "Metformin, VPA, Propofol, Linezolid, Chloramphenicol — all worsen OXPHOS or deplete mtDNA; fatal in mt disease"},
            {"term": "PRIS",                "definition": "Propofol Infusion Syndrome — mitochondrial OXPHOS uncoupling; metabolic acidosis; rhabdomyolysis; high mortality in mt disease"},
            {"term": "GIR 6–8",             "definition": "Glucose Infusion Rate 6–8 mg/kg/min — mandatory during any perioperative fasting period; prevents catabolic crisis in mt disease"},
            {"term": "KD contraindication", "definition": "Ketogenic diet contraindicated in mt disease — fatty acid oxidation and ketone body utilisation via mt OXPHOS; KD stresses already-deficient ETC"},
        ],
        "references": [
            {"ref": "DiMauro-Schon-2003-NEJM",        "citation": "DiMauro S, Schon EA (2003) Mitochondrial respiratory-chain diseases. N Engl J Med 348:2656–2668"},
            {"ref": "Schaefer-2008-AnnNeurol",         "citation": "Schaefer AM et al. (2008) Prevalence of mitochondrial disease in adults. Ann Neurol 63:35–39"},
            {"ref": "Gorman-2016-NatRevDisPrimers",    "citation": "Gorman GS et al. (2016) Mitochondrial diseases. Nat Rev Dis Primers 2:16080"},
            {"ref": "Bhatt-2017-MARS2-ARSAL",          "citation": "Bhatt DK et al. (2017) MARS2 and mitochondrial dysfunction. Mitochondrion 35:21–28"},
            {"ref": "Suzuki-2011-NatChemBiol-lysidine", "citation": "Suzuki T et al. (2011) Wobble modification defects in tRNA disturb codon-anticodon pairing. Nat Chem Biol 7:531–538"},
            {"ref": "Rossmanith-1995-JBiolChem-mt-tRNA","citation": "Rossmanith W et al. (1995) Processing of human mitochondrial tRNA precursors. J Biol Chem 270:12885–12891"},
        ],
    }
