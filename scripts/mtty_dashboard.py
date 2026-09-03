#!/usr/bin/env python3
"""MT-TY — Mitochondrially Encoded tRNA-Tyr — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | L-strand rCRS 5826–5891

MT-TY (OMIM *590100) encodes mitochondrial tRNA-Tyr (GUA anticodon), located on the
**L-strand** at rCRS 5826–5891 (66 nt). MT-TY is the ELEVENTH tRNA gene of the human
mitochondrial genome, following MT-TF (577–647), MT-TV (1602–1670), MT-TL1 (3230–3304),
MT-TI (4263–4331), MT-TQ (L-strand 4329–4400), MT-TM (4402–4469), MT-TW (5512–5579),
MT-TA (L-strand 5587–5655), MT-TN (L-strand 5657–5729), and MT-TC (L-strand 5761–5826).
MT-TY is the FOURTH and FINAL consecutive L-strand tRNA in the cluster: MT-TA (5587–5655),
MT-TN (5657–5729), MT-TC (5761–5826), MT-TY (5826–5891). MT-TC ends at rCRS 5826 — the
same position where MT-TY begins — making MT-TC and MT-TY immediately adjacent with a
0 nt gap (shared boundary at 5826). MT-TY ends at 5891; MT-CO1 begins at 5904 (13 nt gap).

CRITICAL UNIQUE FEATURE — L-STRAND NGS PITFALL:
MT-TY is encoded on the L-strand (light/complementary strand). Standard NGS pipelines
optimised for H-strand (heavy/reference-strand) variant calling will MISS or MIS-CALL
MT-TY variants. The gene is read in the reverse-complement direction relative to the rCRS
reference sequence. This L-strand pitfall is shared by all four tRNAs in this cluster:
MT-TA, MT-TN, MT-TC, MT-TY — requiring mandatory reverse-complement QC processing for
the entire rCRS 5587–5891 region. MT-TY itself occupies rCRS 5826–5891 (0 nt gap from
MT-TC; 13 nt gap to MT-CO1 at rCRS 5904).

L-STRAND CLUSTER — FOURTH AND FINAL:
MT-TY is the FOURTH and FINAL of the four consecutive L-strand tRNAs.
Immediately 5' is MT-TC (L-strand, 5761–5826) with a 0 nt gap (shared boundary at 5826).
Immediately 3' is MT-CO1 (H-strand, 5904–7445) with a 13 nt non-coding gap (5892–5903).
Large mtDNA deletions spanning the TC/TY junction (rCRS ~5826) simultaneously remove
MT-TC AND MT-TY, producing compound tRNA-Cys + tRNA-Tyr loss with compounded CI+CIV
deficiency. MT-TY is the entry point into the MT-CO1 coding region — deletions extending
into MT-CO1 produce CIV-dominant or pan-OXPHOS deficiency depending on COX1 disruption.

GUA ANTICODON — TYR CODON DECODING:
MT-TY has a GUA anticodon, reading UAC (Tyr) codons by Watson-Crick pairing and UAU by
wobble (G:U at position 34). In human mitochondria, the two Tyr codons (UAU, UAC) are
decoded exclusively by MT-TY. Tyrosine residues occur throughout the OXPHOS structural
subunits including COX subunits (CIV) and ND subunits (CI). MT-TY mutations reduce
tRNA-Tyr availability → impaired elongation at UAU/UAC codons → CI+CIV assembly defect
(Tyr residues in structural domains of CI ND subunits and CIV COX subunits).

GUA ANTICODON — NOT UAA/UAG (STOP CODONS):
In the standard genetic code UAA and UAG are STOP codons; in human mitochondria UAA=STOP
and UAG=STOP. The GUA anticodon decodes UAY (UAC/UAU = Tyr) ONLY. MT-TY does NOT decode
stop codons; mt-release factors (mtRF1a) handle mt-stop codon recognition.

NUCLEAR DDx — YARS2 (mt-Tyrosyl-tRNA Synthetase 2):
YARS2 (OMIM *610957) biallelic mutations cause mitochondrial Tyr-tRNA aminoacylation
failure — same biochemical level as MT-TY, but with a dramatically different phenotype:
MLASA2 (Myopathy, Lactic Acidosis, and Sideroblastic Anemia type 2 — OMIM #613561).
Key distinguishing features of YARS2/MLASA2 vs MT-TY disease: (1) SIDEROBLASTIC ANEMIA
(ringed sideroblasts on bone marrow biopsy) — does NOT occur in MT-TY disease; (2) AR
inheritance (biallelic nuclear mutations) vs maternal heteroplasmic mtDNA; (3) earlier
onset with multi-system involvement including bone marrow failure; (4) WES-detectable
(YARS2 is nuclear), while MT-TY is missed by WES and requires dedicated mtDNA panel
with mandatory L-strand QC.

  MT-TY gene              OMIM *590100
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Occasional Cardiomyopathy / SNHL / Leigh-like at high heteroplasmy
                          L-strand NGS pitfall: mandatory reverse-complement QC rCRS 5826–5891
  Protein product         tRNA-Tyr (GUA anticodon) — 66 nucleotides; RNA gene
                          Decodes UAC (Watson-Crick) and UAU (wobble G34:U) — both Tyr codons
  Genome                  Mitochondrial DNA (mtDNA), **L-strand**, rCRS 5826–5891
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    ELEVENTH tRNA in mitochondrial genome
                          5′-adjacent to MT-TC (L-strand, 5761–5826), 0 nt gap (shared 5826)
                          3′-adjacent to MT-CO1 (H-strand, 5904–7445), 13 nt gap
                          FOURTH AND FINAL consecutive L-strand tRNA (MT-TA/TN/TC/TY cluster)
"""

import random
import math

# ── Seed for reproducible 40-patient cohort ──────────────────────────────────
SEED = 821
RNG  = random.Random(SEED)


def _rand_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    v = RNG.gauss(mu, sigma)
    return round(max(lo, min(hi, v)), 1)


def _rand_int(lo: int, hi: int) -> int:
    return RNG.randint(lo, hi)


# ── 40-patient cohort generator ───────────────────────────────────────────────
def _build_cohort():
    """Generate deterministic 40-patient MT-TY cohort (seed-821)."""
    N = 40
    patients = []

    # Variant distribution (must sum to 40)
    variant_pool = (
        ["m.5877GA"]  * 11 +   # Acceptor stem — CPEO + myopathy              ~28%
        ["m.5843AG"]  * 9  +   # Anticodon stem — CPEO + cardiomyopathy       ~22%
        ["m.5860CT"]  * 8  +   # Variable loop — multisystem / Leigh-like      ~20%
        ["m.5852AG"]  * 6  +   # T-loop — exercise intolerance + SNHL         ~15%
        ["LargeDel"]  * 6      # TC-TY spanning — KSS/CPEO compound loss      ~15%
    )
    RNG.shuffle(variant_pool)

    for i in range(N):
        variant = variant_pool[i]

        # Heteroplasmy by variant
        if variant == "m.5877GA":
            hetero = _rand_normal(52, 15, 22, 85)
        elif variant == "m.5843AG":
            hetero = _rand_normal(55, 16, 20, 87)
        elif variant == "m.5860CT":
            hetero = _rand_normal(67, 14, 43, 92)
        elif variant == "m.5852AG":
            hetero = _rand_normal(44, 13, 20, 73)
        else:  # LargeDeletion
            hetero = _rand_normal(37, 11, 17, 61)

        # OXPHOS activities
        if variant == "LargeDel":
            ci  = _rand_normal(27, 9, 13, 46)
            civ = _rand_normal(29, 9, 14, 48)
        elif variant == "m.5860CT":
            ci  = _rand_normal(30, 10, 14, 51)
            civ = _rand_normal(32, 10, 15, 53)
        else:
            ci  = _rand_normal(40, 12, 19, 64)
            civ = _rand_normal(42, 12, 21, 66)

        # Clinical flags
        cpeo     = variant in ("m.5877GA", "m.5843AG", "LargeDel") or RNG.random() < 0.64
        myo      = True
        cardio   = (variant == "m.5843AG") or (RNG.random() < 0.21)
        snhl     = (variant == "m.5852AG") or (RNG.random() < 0.18)
        compound = variant == "LargeDel"

        age_onset = _rand_normal(36, 12, 15, 63) if variant != "LargeDel" else _rand_normal(30, 9, 13, 49)

        patients.append({
            "id":              f"TY{i+1:02d}",
            "variant":         variant,
            "heteroplasmy":    hetero,
            "ci_pct_normal":   ci,
            "civ_pct_normal":  civ,
            "cpeo":            cpeo,
            "myopathy":        myo,
            "cardiomyopathy":  cardio,
            "snhl":            snhl,
            "compound_tc_ty_loss": compound,
            "age_onset":       age_onset,
        })
    return patients


def _cohort_stats(patients):
    N = len(patients)
    avg = lambda key: round(sum(p[key] for p in patients) / N, 1)
    pct = lambda key: round(100 * sum(1 for p in patients if p[key]) / N, 1)
    return {
        "n_patients":              N,
        "seed":                    SEED,
        "avg_heteroplasmy_blood_pct": avg("heteroplasmy"),
        "avg_ci_activity_pct_normal": avg("ci_pct_normal"),
        "avg_civ_activity_pct_normal": avg("civ_pct_normal"),
        "pct_cpeo":                pct("cpeo"),
        "pct_myopathy":            pct("myopathy"),
        "pct_cardiomyopathy":      pct("cardiomyopathy"),
        "pct_snhl":                pct("snhl"),
        "pct_compound_tc_ty_loss": pct("compound_tc_ty_loss"),
    }


# ── API response functions ────────────────────────────────────────────────────
def get_overview():
    patients = _build_cohort()
    stats    = _cohort_stats(patients)

    return {
        "title":  "MT-TY — tRNA-Tyr (GUA Anticodon) Dashboard",
        "gene":   "MT-TY",
        "omim":   "*590100",
        "strand": "L-strand",
        "rcrs":   "5826–5891",
        "nt_length": 66,
        "anticodon": "GUA",
        "tRNA_type": "tRNA-Tyr",
        "genome_order": "ELEVENTH tRNA — 11th in human mitochondrial genome",

        "gene_facts": {
            "gene":          "MT-TY (OMIM *590100)",
            "product":       "tRNA-Tyr — 66 nt; GUA anticodon; decodes UAC (Watson-Crick) & UAU (wobble G34:U)",
            "strand":        "L-strand (light / complementary strand) — NGS pitfall",
            "rCRS_position": "5826–5891",
            "cluster":       "FOURTH and FINAL consecutive L-strand tRNA (MT-TA/TN/TC/TY — rCRS 5587–5891)",
            "5prime_neighbor": "MT-TC (L-strand 5761–5826) — 0 nt gap (shared boundary at 5826)",
            "3prime_neighbor": "MT-CO1 (H-strand 5904–7445) — 13 nt non-coding gap",
            "inheritance":   "Maternal (heteroplasmic); heteroplasmy threshold determines severity",
            "diseases":      "CPEO · Myopathy · Exercise Intolerance · Cardiomyopathy · SNHL (threshold-dependent)",
            "nuclear_ddx":   "YARS2 — biallelic → MLASA2 (Myopathy + Lactic Acidosis + Sideroblastic Anemia), NOT adult CPEO",
        },

        "l_strand_ngs_alert": {
            "alert_class": "L-STRAND NGS PITFALL — MANDATORY Reverse-Complement QC",
            "detail": (
                "MT-TY (rCRS 5826–5891) is L-strand encoded. Standard H-strand NGS pipelines will MISS "
                "MT-TY variants. Mandatory reverse-complement processing required for rCRS 5826–5891. "
                "The entire L-strand cluster MT-TA (5587–5655) + MT-TN (5657–5729) + MT-TC (5761–5826) "
                "+ MT-TY (5826–5891) = rCRS 5587–5891 (304 nt block) requires dedicated L-strand QC. "
                "MT-TY shares its 5′ boundary with MT-TC at position 5826 — deletion calling across this "
                "0 nt junction requires phased read analysis. Same pitfall as MT-TQ, MT-TE, MT-TP, MT-ND6."
            ),
        },

        "yars2_ddx_note": {
            "note_class": "YARS2 (Nuclear DDx) — MLASA2 (Sideroblastic Anemia) — NOT Adult CPEO",
            "detail": (
                "YARS2 biallelic mutations (OMIM *610957) cause mt-Tyr-tRNA aminoacylation failure — "
                "same biochemical step as MT-TY but presenting as MLASA2: myopathy + lactic acidosis + "
                "sideroblastic anemia (ringed sideroblasts on bone marrow biopsy). AR inheritance; "
                "WES-detectable. CRITICAL DISCRIMINATOR: SIDEROBLASTIC ANEMIA does NOT occur in MT-TY "
                "disease. Distinguish by: bone marrow findings (MLASA2 has ringed sideroblasts; MT-TY "
                "does not), inheritance (AR vs maternal), onset (earlier in MLASA2). WES will detect "
                "YARS2 but MISS MT-TY — dedicated mtDNA panel with mandatory L-strand QC required."
            ),
        },

        "cohort_statistics": stats,

        "cohort_summary_features": [
            {"feature": "CPEO",           "value": str(stats["pct_cpeo"]),           "note": "Progressive ptosis + ophthalmoplegia"},
            {"feature": "Myopathy",       "value": str(stats["pct_myopathy"]),       "note": "RRF on biopsy; SDH+/COX− fibres"},
            {"feature": "Cardiomyopathy", "value": str(stats["pct_cardiomyopathy"]), "note": "HCM/DCM; annual echo"},
            {"feature": "SNHL",           "value": str(stats["pct_snhl"]),           "note": "Sensorineural hearing loss"},
            {"feature": "Compound TC/TY loss", "value": str(stats["pct_compound_tc_ty_loss"]), "note": "Large deletion spanning MT-TC & MT-TY"},
        ],

        "phenotype_distribution": [
            {"variant": "m.5877GA", "pct": 27.5, "phenotype": "CPEO + myopathy dominant",       "position": "Acceptor stem"},
            {"variant": "m.5843AG", "pct": 22.5, "phenotype": "CPEO + cardiomyopathy",          "position": "Anticodon stem"},
            {"variant": "m.5860CT", "pct": 20.0, "phenotype": "Multisystem / Leigh-like",        "position": "Variable loop"},
            {"variant": "m.5852AG", "pct": 15.0, "phenotype": "Exercise intolerance + SNHL",    "position": "T-loop"},
            {"variant": "LargeDel", "pct": 15.0, "phenotype": "KSS/CPEO — compound TC+TY loss", "position": "TC–TY spanning deletion (rCRS ~5826)"},
        ],

        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<20%",   "expected_phenotype": "Subclinical / carrier — no overt disease"},
            {"threshold_pct": "20–40%", "expected_phenotype": "Exercise intolerance; mild CPEO; SNHL (variable)"},
            {"threshold_pct": "40–60%", "expected_phenotype": "CPEO + myopathy; RRF on biopsy; fatigue"},
            {"threshold_pct": "60–80%", "expected_phenotype": "CPEO + myopathy + cardiomyopathy; elevated lactate"},
            {"threshold_pct": ">80%",   "expected_phenotype": "Multisystem; Leigh-like (basal ganglia); KSS features"},
        ],

        "biochemical_fingerprint": {
            "summary": "Combined CI+CIV deficiency with CII NORMAL — mt-translation fingerprint (tRNA-Tyr depletion → global mt-translation impairment at UAU/UAC codons in CI ND-subunits and CIV COX-subunits)",
            "complex_i":  "Reduced 35–64% of normal (NADH:ubiquinone oxidoreductase; 7 mtDNA-encoded subunits contain Tyr residues in critical structural domains including proton-pumping channels)",
            "complex_ii": "NORMAL (succinate dehydrogenase — entirely nuclear-encoded; diagnostic discriminator vs mitochondrial-disease mimics)",
            "complex_iv": "Reduced 36–66% of normal (CIV has 3 mtDNA-encoded subunits; COX1/COX2/COX3 contain conserved Tyr residues including Y244 in COX1 catalytic binuclear center)",
            "mechanism":  "tRNA-Tyr depletion reduces decoding of UAU/UAC codons → impaired elongation in Tyr-containing OXPHOS subunits → CI ND-subunit structural instability and CIV COX1 catalytic centre impairment",
        },

        "key_molecular_features": [
            "GUA anticodon decodes UAC (Watson-Crick) AND UAU (wobble G34:U) — BOTH Tyr codons in human mt code exclusively",
            "UAA and UAG are STOP codons in both mt and standard genetic code — GUA anticodon does NOT read stop codons",
            "L-strand gene: reverse-complement QC MANDATORY — same pitfall as MT-TA, MT-TN, MT-TC, MT-TQ, MT-TE, MT-TP, MT-ND6",
            "FOURTH AND FINAL of FOUR consecutive L-strand tRNAs (MT-TA → MT-TN → MT-TC → MT-TY): cluster rCRS 5587–5891",
            "0 nt gap 5′ — MT-TC ends at 5826 = MT-TY begins at 5826 (immediately adjacent, shared boundary)",
            "13 nt gap 3′ (from MT-TY 3′ end at 5891 to MT-CO1 5′ start at 5904) — non-coding control element",
            "Large deletions spanning MT-TC/MT-TY junction (0 nt gap at 5826) simultaneously ablate tRNA-Cys AND tRNA-Tyr",
            "Large deletions extending into MT-CO1 (5904+) produce additional CIV-dominant component from COX1 disruption",
            "COX1 (MT-CO1) immediately downstream: contains Y244 — key Tyr in binuclear catalytic center; CIV extremely sensitive to tRNA-Tyr depletion",
            "YARS2 (nuclear, AR) → MLASA2: same aminoacylation failure but sideroblastic anemia + earlier onset — KEY DDx by bone marrow biopsy",
        ],

        "clinical_alerts": [
            {
                "alert":  "L-STRAND NGS PITFALL — MT-TY variants MISSED by H-strand pipelines",
                "detail": "Dedicated mtDNA panel with mandatory reverse-complement QC for rCRS 5826–5891. Entire cluster 5587–5891 requires L-strand processing. 0 nt gap with MT-TC at 5826 requires phased read analysis for deletion calls.",
            },
            {
                "alert":  "COMPOUND DELETION RISK — MT-TC/TY shared boundary at rCRS 5826",
                "detail": "0 nt gap means any deletion spanning position 5826 simultaneously removes both tRNA-Cys AND tRNA-Tyr. Extended deletions into MT-CO1 (5904+) additionally disrupt COX1 causing CIV-dominant or pan-OXPHOS deficiency.",
            },
            {
                "alert":  "YARS2/MLASA2 vs MT-TY — SIDEROBLASTIC ANEMIA is the discriminator",
                "detail": "If sideroblastic anemia (ringed sideroblasts on bone marrow) is present → YARS2/MLASA2 (nuclear AR). MT-TY disease does NOT cause sideroblastic anemia. Both present with myopathy + lactic acidosis but MLASA2 is WES-detectable; MT-TY requires dedicated mtDNA panel.",
            },
            {
                "alert":  "ABSOLUTE CONTRAINDICATIONS — Metformin, VPA, Propofol, Linezolid, Chloramphenicol",
                "detail": "Same as all mt-tRNA diseases: these agents worsen OXPHOS, deplete mtDNA, or inhibit mt-ribosomes. Fatal in CI+CIV deficient patients.",
            },
        ],
    }


def get_breakdown():
    patients = _build_cohort()
    stats    = _cohort_stats(patients)

    # Per-variant breakdown
    variants = ["m.5877GA", "m.5843AG", "m.5860CT", "m.5852AG", "LargeDel"]
    vb = []
    for v in variants:
        grp = [p for p in patients if p["variant"] == v]
        if not grp:
            continue
        n = len(grp)
        avg = lambda k: round(sum(p[k] for p in grp) / n, 1)
        pct = lambda k: round(100 * sum(1 for p in grp if p[k]) / n, 1)
        vb.append({
            "variant":          v,
            "n":                n,
            "pct_of_cohort":    round(100 * n / 40, 1),
            "avg_heteroplasmy": avg("heteroplasmy"),
            "avg_ci_pct_normal":  avg("ci_pct_normal"),
            "avg_civ_pct_normal": avg("civ_pct_normal"),
            "pct_cpeo":         pct("cpeo"),
            "pct_myopathy":     pct("myopathy"),
            "pct_cardiomyopathy": pct("cardiomyopathy"),
            "pct_compound_tc_ty_loss": pct("compound_tc_ty_loss"),
        })

    return {
        "variant_breakdown": vb,
        "cohort_statistics": stats,

        "ddx_table": [
            {
                "entity":       "MT-TY (this gene)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Adult CPEO + myopathy; exercise intolerance; cardiomyopathy (m.5843AG); SNHL (m.5852AG)",
                "biochemistry": "CI+CIV deficiency; CII NORMAL; symmetric CI ≈ CIV; no sideroblastic anemia",
                "ngs":          "L-strand mtDNA panel — L-strand QC mandatory (rCRS 5826–5891)",
                "distinctive":  "L-strand NGS pitfall; GUA anticodon Tyr-only; YARS2 DDx MLASA2 (sideroblastic anemia)",
            },
            {
                "entity":       "YARS2 / MLASA2 (nuclear DDx)",
                "inheritance":  "AR nuclear — biallelic",
                "phenotype":    "Myopathy + lactic acidosis + SIDEROBLASTIC ANEMIA (ringed sideroblasts); earlier onset",
                "biochemistry": "Combined OXPHOS deficiency; mt-Tyr-tRNA aminoacylation failure; bone marrow failure",
                "ngs":          "WES detects YARS2 — does NOT detect MT-TY",
                "distinctive":  "SIDEROBLASTIC ANEMIA (MT-TY has NONE); AR not maternal; WES-detectable",
            },
            {
                "entity":       "MT-TC (tRNA-Cys, 5761–5826)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; cardiomyopathy (m.5763CT); SNHL (m.5793AG)",
                "biochemistry": "CI+CIV deficiency; CII NORMAL; symmetric CI ≈ CIV",
                "ngs":          "L-strand mtDNA panel — L-strand QC rCRS 5761–5826",
                "distinctive":  "THIRD L-strand tRNA (5761–5826); GCA anticodon Cys; CARS2 DDx neonatal OXPHOS",
            },
            {
                "entity":       "MT-TN (tRNA-Asn, 5657–5729)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; SNHL (m.5692AG); cardiomyopathy (m.5728AG)",
                "biochemistry": "CI+CIV deficiency; CII NORMAL",
                "ngs":          "L-strand mtDNA panel — L-strand QC rCRS 5657–5729",
                "distinctive":  "SECOND L-strand tRNA; GTT anticodon Asn; NARS2 DDx Perrault syndrome",
            },
            {
                "entity":       "MT-TA (tRNA-Ala, 5587–5655)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; exercise intolerance; cardiomyopathy (m.5655AG)",
                "biochemistry": "CI+CIV deficiency; CII NORMAL",
                "ngs":          "L-strand mtDNA panel — L-strand QC rCRS 5587–5655",
                "distinctive":  "FIRST L-strand tRNA; GGC anticodon Ala (fourfold degenerate); AARS2 DDx ovario-leukodystrophy",
            },
            {
                "entity":       "MT-TW (tRNA-Trp, 5512–5579)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; CIV-dominant asymmetry at anticodon variants",
                "biochemistry": "CI+CIV deficiency; CIV preferentially worse (UGA-Trp in COX1); CII NORMAL",
                "ngs":          "H-strand (no NGS pitfall for TW itself); L-strand cluster begins after TW",
                "distinctive":  "UGA-Trp recoding unique; CIV>CI asymmetry; H-strand but L-strand cluster immediately 3'",
            },
            {
                "entity":       "MT-TL1 / MELAS",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "MELAS: stroke-like episodes, encephalopathy, lactic acidosis; MIDD; CPEO",
                "biochemistry": "CI+CIV; pan-OXPHOS (high heteroplasmy); vascular endothelium mitochondrial angiopathy",
                "ngs":          "H-strand mtDNA panel (MT-TL1 is H-strand); no L-strand pitfall",
                "distinctive":  "STROKE-LIKE EPISODES — NOT present in MT-TY; m.3243AG most common; MIDD (diabetes)",
            },
            {
                "entity":       "BTBGD / SLC19A3",
                "inheritance":  "AR nuclear",
                "phenotype":    "Leigh-like MRI with basal ganglia signal; episodic encephalopathy; biotin/thiamine responsive",
                "biochemistry": "Normal OXPHOS activities; biotin-thiamine transporter deficiency; mimics Leigh MRI",
                "ngs":          "WES / targeted SLC19A3 sequencing",
                "distinctive":  "TREATABLE Leigh mimic — MANDATORY EXCLUSION before attributing Leigh MRI to MT-TY",
            },
        ],

        "management_by_variant": [
            {"variant": "m.5877GA", "cpeo_risk": "High",   "cardio_risk": "Low",      "key_action": "Annual ophthalmology + ptosis surgery as needed; annual echo (low risk)"},
            {"variant": "m.5843AG", "cpeo_risk": "High",   "cardio_risk": "Moderate", "key_action": "6-monthly cardiology echo; CPEO management; beta-blocker if HCM"},
            {"variant": "m.5860CT", "cpeo_risk": "Moderate","cardio_risk": "Moderate","key_action": "MRI brain for Leigh features at >60% heteroplasmy; metabolic team"},
            {"variant": "m.5852AG", "cpeo_risk": "Moderate","cardio_risk": "Low",     "key_action": "Audiology + cochlear implant assessment; exercise program modification"},
            {"variant": "LargeDel", "cpeo_risk": "High",   "cardio_risk": "Moderate", "key_action": "Deletion breakpoint mapping; compound tRNA-Cys + tRNA-Tyr loss; KSS protocol"},
        ],

        "absolute_contraindications": [
            {"drug": "Metformin",       "reason": "Inhibits CI directly; fatal lactic acidosis in mt-tRNA CI+CIV deficiency"},
            {"drug": "Valproic acid (VPA)", "reason": "Inhibits fatty acid oxidation; depletes carnitine; worsens mt-energy failure"},
            {"drug": "Propofol",        "reason": "Propofol Infusion Syndrome (PRIS): mitochondrial OXPHOS uncoupling; high mortality"},
            {"drug": "Linezolid",       "reason": "Inhibits mt 23S rRNA translation; worsens mt-tRNA translation defect at ribosomal level"},
            {"drug": "Chloramphenicol", "reason": "Inhibits mt 70S ribosome; compounds mt-translation failure from tRNA-Tyr depletion"},
            {"drug": "Ketogenic diet",  "reason": "Forces OXPHOS-dependent fatty acid oxidation; stresses already-deficient ETC"},
        ],

        "safe_interventions": [
            {"intervention": "Thiamine (B1) — Mandatory Empiric", "evidence": "Level B: empiric in any Leigh-like or lactic acidosis presentation until BTBGD excluded"},
            {"intervention": "Biotin — Mandatory Empiric",        "evidence": "Level B: empiric in any Leigh-like presentation until biotinidase/BTBGD excluded"},
            {"intervention": "LEV (Levetiracetam)",               "evidence": "Level B: preferred AED in mt-tRNA disease; avoids mitochondrial toxicity of VPA"},
            {"intervention": "CoQ10 / Ubiquinol",                 "evidence": "Level C: electron carrier supplementation; standard of care in mt disease"},
            {"intervention": "Riboflavin (B2)",                   "evidence": "Level C: FAD precursor; supports Complex I/II; some CI-deficiency benefit"},
            {"intervention": "L-Carnitine",                       "evidence": "Level C: supports fatty acid metabolism; replaces VPA-depleted carnitine stores"},
            {"intervention": "Beta-blocker (if HCM/cardiomyopathy)", "evidence": "Standard cardiology management: rate control + outflow obstruction reduction in HCM"},
            {"intervention": "GIR 6–8 mg/kg/min (perioperative)", "evidence": "MANDATORY: prevents catabolic crisis during fasting; never fast a mt-disease patient"},
        ],
    }


def get_definitions():
    return {
        "title": "MT-TY — Clinical & Molecular Definitions",
        "gene_definitions": [
            {"term": "MT-TY",               "definition": "Mitochondrially encoded tRNA-Tyr gene; L-strand, rCRS 5826–5891, 66 nt (OMIM *590100)"},
            {"term": "tRNA-Tyr",            "definition": "Transfer RNA for Tyrosine; GUA anticodon, decoding UAC (Watson-Crick) and UAU (wobble G34:U) — both Tyr codons in human mt code"},
            {"term": "GUA anticodon",       "definition": "Anticodon of tRNA-Tyr (positions 34–36 of the tRNA); G34 wobble position reads both UAC and UAU; Tyr is exclusively decoded by MT-TY in human mitochondria"},
            {"term": "UAY codon box",       "definition": "UAU and UAC both encode Tyrosine — the 'UAY' split of the UA codon box (UAA=STOP; UAG=STOP in mt code as in standard; UAU/UAC=Tyr by MT-TY)"},
            {"term": "L-strand cluster",    "definition": "MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), MT-TY (5826–5891) — four consecutive L-strand tRNA genes; entire 304 nt block rCRS 5587–5891 requires L-strand QC"},
            {"term": "rCRS",                "definition": "Revised Cambridge Reference Sequence — standard human mtDNA numbering (GenBank NC_012920.1)"},
            {"term": "Heteroplasmy",        "definition": "Mixture of wild-type and mutant mtDNA molecules within a cell/tissue; severity correlates with mutant load (%)"},
        ],
        "biochemical_definitions": [
            {"term": "CI deficiency",       "definition": "NADH:ubiquinone oxidoreductase (Complex I) deficiency — 7 mtDNA-encoded subunits; Tyr residues in ND2/ND4/ND5 structural proton channels are tRNA-Tyr dependent"},
            {"term": "CII (normal)",        "definition": "Succinate dehydrogenase — entirely nuclear-encoded; remains normal in mt-tRNA disease (diagnostic discriminator vs mitochondrial-disease mimics)"},
            {"term": "CIV deficiency",      "definition": "Cytochrome c oxidase (COX) deficiency — 3 mtDNA-encoded subunits; COX1 Y244 is critical catalytic Tyr in binuclear center; tRNA-Tyr depletion impairs COX1 elongation"},
            {"term": "Symmetric CI ≈ CIV",  "definition": "MT-TY produces symmetric CI ≈ CIV deficiency — contrast with MT-TW (CIV preferentially worse due to UGA-Trp recoding disruption in MT-CO1)"},
            {"term": "COX1 Y244",           "definition": "Tyrosine-244 in COX1 (MT-CO1) is located at the binuclear catalytic centre; tRNA-Tyr mutations reduce COX1 translation fidelity at UAY codons in MT-CO1"},
            {"term": "mt-translation fingerprint", "definition": "Combined CI+CIV deficiency with CII NORMAL — characteristic of pathogenic mt-tRNA gene mutations"},
            {"term": "Ragged-Red Fibres",   "definition": "Subsarcolemmal mitochondrial accumulation on modified Gomori trichrome; SDH-positive/COX-negative on sequential histochemistry"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",                "definition": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive ophthalmoplegia; predominant MT-TY phenotype"},
            {"term": "KSS",                 "definition": "Kearns-Sayre Syndrome — CPEO + pigmentary retinopathy + cardiac conduction defect; often large mtDNA deletion spanning MT-TY and adjacent tRNAs"},
            {"term": "MLASA2",              "definition": "Myopathy, Lactic Acidosis, and Sideroblastic Anemia type 2 (OMIM #613561) — caused by biallelic YARS2 mutations; CRITICAL DDx for MT-TY; sideroblastic anemia (ringed sideroblasts) distinguishes MLASA2 from MT-TY"},
            {"term": "YARS2 deficiency",    "definition": "OMIM *610957 — AR biallelic YARS2 mutations → mt-Tyr-tRNA aminoacylation failure; MLASA2 phenotype (myopathy + lactic acidosis + sideroblastic anemia); earlier onset than MT-TY CPEO"},
            {"term": "Sideroblastic anemia","definition": "Ringed sideroblasts on bone marrow trephine — pathognomonic of YARS2/MLASA2; completely ABSENT in MT-TY disease; key discriminating feature"},
            {"term": "Compound tRNA loss",  "definition": "Large deletions spanning MT-TC/MT-TY (0 nt junction at 5826): simultaneous loss of tRNA-Cys and tRNA-Tyr; extended deletions into MT-CO1 compound further with COX1 disruption"},
            {"term": "BTBGD / SLC19A3",    "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease — AR treatable Leigh mimic; mandatory exclusion before attributing Leigh MRI to MT-TY"},
        ],
        "ngs_definitions": [
            {"term": "L-strand encoding",   "definition": "MT-TY (rCRS 5826–5891) is L-strand encoded; standard NGS H-strand calls MISS variants; reverse-complement pipeline required"},
            {"term": "L-strand NGS pitfall","definition": "H-strand-only NGS pipelines fail to detect L-strand tRNA variants; clinically identical pitfall to MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), MT-TQ (4329–4400), MT-TE (14674–14742), MT-TP (15956–16023), MT-ND6 (14149–14673)"},
            {"term": "Four-gene L-strand cluster", "definition": "MT-TA, MT-TN, MT-TC, MT-TY occupy rCRS 5587–5891 — all L-strand; QC must cover the entire block; MT-TC/MT-TY shared 0 nt boundary at 5826 requires phased deletion calling; MT-TY/MT-CO1 transition at 5904 marks return to H-strand territory"},
            {"term": "MT-TY/MT-CO1 transition", "definition": "MT-TY ends at rCRS 5891; MT-CO1 begins at 5904 (13 nt gap); MT-CO1 is H-strand encoded — sequencing pipelines must switch from L-strand to H-strand mode at this boundary"},
            {"term": "Variant allele fraction (VAF)", "definition": "Proportion of reads carrying the variant allele = heteroplasmy estimate; tissue-specific (blood < muscle < affected tissue)"},
        ],
        "drug_definitions": [
            {"term": "Absolute CI",         "definition": "Metformin, VPA, Propofol, Linezolid, Chloramphenicol — all worsen OXPHOS or deplete mtDNA; fatal in mt disease"},
            {"term": "PRIS",                "definition": "Propofol Infusion Syndrome — mitochondrial OXPHOS uncoupling; metabolic acidosis; rhabdomyolysis; high mortality in mt disease"},
            {"term": "GIR 6–8",             "definition": "Glucose Infusion Rate 6–8 mg/kg/min — mandatory during any perioperative fasting period; prevents catabolic crisis in mt disease"},
            {"term": "KD contraindication", "definition": "Ketogenic diet contraindicated in mt disease — forces OXPHOS-dependent fatty acid oxidation; stresses already-deficient ETC"},
        ],
        "references": [
            {"ref": "DiMauro-Schon-2003-NEJM",       "citation": "DiMauro S, Schon EA (2003) Mitochondrial respiratory-chain diseases. N Engl J Med 348:2656–2668"},
            {"ref": "Schaefer-2008-AnnNeurol",        "citation": "Schaefer AM et al. (2008) Prevalence of mitochondrial disease in adults. Ann Neurol 63:35–39"},
            {"ref": "Gorman-2016-NatRevDisPrimers",   "citation": "Gorman GS et al. (2016) Mitochondrial diseases. Nat Rev Dis Primers 2:16080"},
            {"ref": "Riley-2010-MLASA2-YARS2",        "citation": "Riley LG et al. (2010) Mutation of the mitochondrial tyrosyl-tRNA synthetase gene YARS2 causes myopathy, lactic acidosis, and sideroblastic anemia — MLASA syndrome. Am J Hum Genet 87:52–59"},
            {"ref": "Sasarman-2012-YARS2",            "citation": "Sasarman F et al. (2012) The 3′ addition of CCA to mitochondrial tRNA-Ser(AGY) is specifically impaired in patients with mutations in the tRNA nucleotidyl transferase TRNT1. Hum Mol Genet 21:2841–2849 (context: MLASA2 biochemistry)"},
            {"ref": "Sprinzl-2005-NAR-tRNA",          "citation": "Sprinzl M & Vassilenko KS (2005) Compilation of tRNA sequences and sequences of tRNA genes. Nucleic Acids Res 33:D139–D140"},
            {"ref": "Barrell-1979-Science",           "citation": "Barrell BG et al. (1979) A different genetic code in human mitochondria. Nature 282:189–194"},
        ],
    }
