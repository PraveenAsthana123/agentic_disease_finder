#!/usr/bin/env python3
"""MT-TC — Mitochondrially Encoded tRNA-Cys — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | L-strand rCRS 5761–5826

MT-TC (OMIM *590020) encodes mitochondrial tRNA-Cys (GCA anticodon), located on the
**L-strand** at rCRS 5761–5826 (66 nt). MT-TC is the TENTH tRNA gene of the human
mitochondrial genome, following MT-TF (577–647), MT-TV (1602–1670), MT-TL1 (3230–3304),
MT-TI (4263–4331), MT-TQ (L-strand 4329–4400), MT-TM (4402–4469), MT-TW (5512–5579),
MT-TA (L-strand 5587–5655), and MT-TN (L-strand 5657–5729). MT-TC is the THIRD consecutive
L-strand tRNA in the cluster: MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826),
MT-TY (5826–5891). MT-TY begins at rCRS 5826 — the same position where MT-TC ends —
making MT-TC and MT-TY immediately adjacent with a 0 nt gap (shared boundary at 5826).

CRITICAL UNIQUE FEATURE — L-STRAND NGS PITFALL:
MT-TC is encoded on the L-strand (light/complementary strand). Standard NGS pipelines
optimised for H-strand (heavy/reference-strand) variant calling will MISS or MIS-CALL
MT-TC variants. The gene is read in the reverse-complement direction relative to the rCRS
reference sequence. This L-strand pitfall is shared by all four tRNAs in this cluster:
MT-TA, MT-TN, MT-TC, MT-TY — requiring mandatory reverse-complement QC processing for
the entire rCRS 5587–5891 region. MT-TC itself occupies rCRS 5761–5826 (32 nt gap from
MT-TN; 0 nt gap to MT-TY at position 5826).

L-STRAND CLUSTER — THIRD OF FOUR:
MT-TC is the third of the four consecutive L-strand tRNAs.
Immediately 5' is MT-TN (L-strand, 5657–5729) with a 32 nt gap at rCRS 5729–5761.
Immediately 3' is MT-TY (L-strand, 5826–5891) with a 0 nt gap (adjacent boundary at 5826).
Large mtDNA deletions spanning the TN/TC junction (rCRS ~5729–5761) simultaneously remove
MT-TN AND MT-TC, producing compound tRNA-Asn + tRNA-Cys loss with compounded CI+CIV
deficiency. Deletions spanning the TC/TY boundary simultaneously remove MT-TC AND MT-TY
(compound tRNA-Cys + tRNA-Tyr loss).

GCA ANTICODON — CYS CODON DECODING:
MT-TC has a GCA anticodon, reading UGC (Cys) codons by Watson-Crick pairing and UGU by
wobble (G:U at position 34). In human mitochondria, the two Cys codons (UGU, UGC) are
both decoded exclusively by MT-TC. Cysteine residues occur in structural cysteine clusters,
Fe-S cluster ligands of CI (FeS subunits N1–N7), and disulfide bonds of CIII and CIV
subunits. MT-TC mutations reduce tRNA-Cys availability → impaired elongation at UGU/UGC
codons → CI+CIV assembly defect (Fe-S cluster and structural Cys residues in CI/CIV).

IMPORTANT DISTINCTION — UGA vs UGY:
UGA is decoded as TRYPTOPHAN (Trp) by MT-TW (not as Cys or STOP) in the mitochondrial
genetic code. MT-TC decodes only UGY (UGU/UGC = Cys). UGA is not a Cys codon.
This is a critical codon-table distinction: MT-TC does NOT read UGA codons.

NUCLEAR DDx — CARS2 (mt-Cysteinyl-tRNA Synthetase 2):
CARS2 (OMIM *612800) biallelic mutations cause mitochondrial Cys-tRNA aminoacylation
failure — same biochemical level as MT-TC, but with a dramatically different phenotype:
severe combined OXPHOS deficiency with neonatal/infantile onset: epileptic encephalopathy,
lactic acidosis, severe combined CI+CIV+CIII deficiency. CARS2 is AR, WES-detectable, and
clinically distinguishable from MT-TC disease by: (1) neonatal/infantile onset (NOT adult
CPEO); (2) AR inheritance (NOT maternal); (3) severe multisystem disease early in life;
(4) WES detection. MT-TC (maternal heteroplasmic CPEO/myopathy) does NOT present in the
neonatal period with epileptic encephalopathy.

  MT-TC gene              OMIM *590020
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Occasional Cardiomyopathy / SNHL / Leigh-like at high heteroplasmy
                          L-strand NGS pitfall: mandatory reverse-complement QC rCRS 5761–5826
  Protein product         tRNA-Cys (GCA anticodon) — 66 nucleotides; RNA gene
                          Decodes UGC (Watson-Crick) and UGU (wobble G34:U) — both Cys codons
  Genome                  Mitochondrial DNA (mtDNA), **L-strand**, rCRS 5761–5826
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    TENTH tRNA in mitochondrial genome
                          5′-adjacent to MT-TN (L-strand, 5657–5729), 32 nt gap
                          3′-adjacent to MT-TY (L-strand, 5826–5891), 0 nt gap (shared 5826)
                          THIRD OF FOUR consecutive L-strand tRNAs (MT-TA/TN/TC/TY cluster)
"""

import random
import math

# ── Seed for reproducible 40-patient cohort ──────────────────────────────────
SEED = 819
RNG  = random.Random(SEED)


def _rand_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    v = RNG.gauss(mu, sigma)
    return round(max(lo, min(hi, v)), 1)


def _rand_int(lo: int, hi: int) -> int:
    return RNG.randint(lo, hi)


# ── 40-patient cohort generator ───────────────────────────────────────────────
def _build_cohort():
    """Generate deterministic 40-patient MT-TC cohort (seed-819)."""
    N = 40
    patients = []

    # Variant distribution (must sum to 40)
    variant_pool = (
        ["m.5814AG"]  * 11 +   # D-stem — CPEO + myopathy                 ~28%
        ["m.5763CT"]  * 9  +   # T-stem — CPEO + cardiomyopathy            ~22%
        ["m.5801GA"]  * 8  +   # variable loop — multisystem / Leigh-like   ~20%
        ["m.5793AG"]  * 6  +   # anticodon loop — exercise + SNHL           ~15%
        ["LargeDel"]  * 6      # TN-TC spanning — KSS/CPEO compound loss    ~15%
    )
    RNG.shuffle(variant_pool)

    for i in range(N):
        variant = variant_pool[i]

        # Heteroplasmy by variant
        if variant == "m.5814AG":
            hetero = _rand_normal(53, 15, 23, 86)
        elif variant == "m.5763CT":
            hetero = _rand_normal(56, 16, 21, 88)
        elif variant == "m.5801GA":
            hetero = _rand_normal(66, 14, 42, 91)
        elif variant == "m.5793AG":
            hetero = _rand_normal(45, 13, 21, 74)
        else:  # LargeDeletion
            hetero = _rand_normal(38, 11, 18, 62)

        # OXPHOS activities
        if variant == "LargeDel":
            ci  = _rand_normal(28, 9, 14, 47)
            civ = _rand_normal(30, 9, 15, 49)
        elif variant == "m.5801GA":
            ci  = _rand_normal(31, 10, 15, 52)
            civ = _rand_normal(33, 10, 16, 54)
        else:
            ci  = _rand_normal(41, 12, 20, 65)
            civ = _rand_normal(43, 12, 22, 67)

        # Clinical flags
        cpeo     = variant in ("m.5814AG", "m.5763CT", "LargeDel") or RNG.random() < 0.65
        myo      = True
        cardio   = (variant == "m.5763CT") or (RNG.random() < 0.22)
        snhl     = (variant == "m.5793AG") or (RNG.random() < 0.19)
        compound = variant == "LargeDel"

        age_onset = _rand_normal(35, 12, 14, 62) if variant != "LargeDel" else _rand_normal(29, 9, 12, 48)

        patients.append({
            "id":              f"TC{i+1:02d}",
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
        "title":  "MT-TC — tRNA-Cys (GCA Anticodon) Dashboard",
        "gene":   "MT-TC",
        "omim":   "*590020",
        "strand": "L-strand",
        "rcrs":   "5761–5826",
        "nt_length": 66,
        "anticodon": "GCA",
        "tRNA_type": "tRNA-Cys",
        "genome_order": "TENTH tRNA — 10th in human mitochondrial genome",

        "gene_facts": {
            "gene":          "MT-TC (OMIM *590020)",
            "product":       "tRNA-Cys — 66 nt; GCA anticodon; decodes UGC (Watson-Crick) & UGU (wobble G34:U)",
            "strand":        "L-strand (light / complementary strand) — NGS pitfall",
            "rCRS_position": "5761–5826",
            "cluster":       "THIRD of FOUR consecutive L-strand tRNAs (MT-TA/TN/TC/TY — rCRS 5587–5891)",
            "5prime_neighbor": "MT-TN (L-strand 5657–5729) — 32 nt gap",
            "3prime_neighbor": "MT-TY (L-strand 5826–5891) — 0 nt gap (shared boundary at 5826)",
            "inheritance":   "Maternal (heteroplasmic); heteroplasmy threshold determines severity",
            "diseases":      "CPEO · Myopathy · Exercise Intolerance · Cardiomyopathy · SNHL (threshold-dependent)",
            "nuclear_ddx":   "CARS2 — biallelic → neonatal/infantile multisystem OXPHOS, NOT adult CPEO",
        },

        "l_strand_ngs_alert": {
            "alert_class": "L-STRAND NGS PITFALL — MANDATORY Reverse-Complement QC",
            "detail": (
                "MT-TC (rCRS 5761–5826) is L-strand encoded. Standard H-strand NGS pipelines will MISS "
                "MT-TC variants. Mandatory reverse-complement processing required for rCRS 5761–5826. "
                "The entire L-strand cluster MT-TA (5587–5655) + MT-TN (5657–5729) + MT-TC (5761–5826) "
                "+ MT-TY (5826–5891) = rCRS 5587–5891 (304 nt block) requires dedicated L-strand QC. "
                "Same pitfall as MT-TQ, MT-TE, MT-TP, MT-ND6."
            ),
        },

        "cars2_ddx_note": {
            "note_class": "CARS2 (Nuclear DDx) — Neonatal/Infantile Multisystem OXPHOS — NOT Adult CPEO",
            "detail": (
                "CARS2 biallelic mutations (OMIM *612800) cause mt-Cys-tRNA aminoacylation failure — "
                "same biochemical step as MT-TC but presenting in neonatal/infantile period with severe "
                "combined OXPHOS deficiency, epileptic encephalopathy, and lactic acidosis. AR inheritance; "
                "WES-detectable. Completely different phenotype from MT-TC (adult CPEO + myopathy, maternal). "
                "Distinguish by: onset (neonatal vs adult), inheritance (AR vs maternal), severity. "
                "WES will detect CARS2 but MISS MT-TC — dedicated mtDNA panel with L-strand QC required."
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
            {"variant": "m.5814AG", "pct": 27.5, "phenotype": "CPEO + myopathy dominant",       "position": "D-stem"},
            {"variant": "m.5763CT", "pct": 22.5, "phenotype": "CPEO + cardiomyopathy",          "position": "T-stem"},
            {"variant": "m.5801GA", "pct": 20.0, "phenotype": "Multisystem / Leigh-like",        "position": "Variable loop"},
            {"variant": "m.5793AG", "pct": 15.0, "phenotype": "Exercise intolerance + SNHL",    "position": "Anticodon loop"},
            {"variant": "LargeDel", "pct": 15.0, "phenotype": "KSS/CPEO — compound TC+TY loss", "position": "TN–TC / TC–TY spanning deletion"},
        ],

        "heteroplasmy_clinical_map": [
            {"threshold_pct": "<20%",   "expected_phenotype": "Subclinical / carrier — no overt disease"},
            {"threshold_pct": "20–40%", "expected_phenotype": "Exercise intolerance; mild CPEO; SNHL (variable)"},
            {"threshold_pct": "40–60%", "expected_phenotype": "CPEO + myopathy; RRF on biopsy; fatigue"},
            {"threshold_pct": "60–80%", "expected_phenotype": "CPEO + myopathy + cardiomyopathy; elevated lactate"},
            {"threshold_pct": ">80%",   "expected_phenotype": "Multisystem; Leigh-like (basal ganglia); KSS features"},
        ],

        "biochemical_fingerprint": {
            "summary": "Combined CI+CIV deficiency with CII NORMAL — mt-translation fingerprint (tRNA-Cys depletion → global mt-translation impairment at UGU/UGC codons in CI and CIV subunits)",
            "complex_i":  "Reduced 35–65% of normal (NADH:ubiquinone oxidoreductase; 7 mtDNA-encoded subunits contain cysteine Fe-S ligands)",
            "complex_ii": "NORMAL (succinate dehydrogenase — entirely nuclear-encoded; diagnostic discriminator vs mitochondrial-disease mimics)",
            "complex_iv": "Reduced 37–67% of normal (CIV has 3 mtDNA-encoded subunits; COX1/COX2/COX3 contain structural Cys residues)",
            "mechanism":  "tRNA-Cys depletion reduces decoding of UGU/UGC codons → impaired elongation in Cys-rich OXPHOS subunits → CI Fe-S cluster assembly defect and CIV structural instability",
        },

        "key_molecular_features": [
            "GCA anticodon decodes UGC (Watson-Crick) AND UGU (wobble G34:U) — BOTH Cys codons in human mt code exclusively",
            "UGA is NOT a Cys codon — decoded as Trp by MT-TW; MT-TC has no role in UGA decoding",
            "L-strand gene: reverse-complement QC MANDATORY — same pitfall as MT-TA, MT-TN, MT-TQ, MT-TE, MT-TP, MT-ND6",
            "THIRD of FOUR consecutive L-strand tRNAs (MT-TA → MT-TN → MT-TC → MT-TY): cluster rCRS 5587–5891",
            "32 nt gap 5′ (from MT-TN 3′ end at 5729 to MT-TC 5′ end at 5761)",
            "0 nt gap 3′ — MT-TC ends at 5826 = MT-TY begins at 5826 (immediately adjacent, shared boundary)",
            "Large deletions spanning MT-TN/MT-TC junction simultaneously ablate tRNA-Asn AND tRNA-Cys",
            "Large deletions at MT-TC/MT-TY boundary simultaneously ablate tRNA-Cys AND tRNA-Tyr (compound two-tRNA loss)",
            "Cys essential for Fe-S cluster ligation in CI (NDUFS1/2/7/8 Fe-S subunits contain conserved Cys-X-X-Cys motifs)",
            "CARS2 (nuclear, AR) → same aminoacylation failure but neonatal/infantile severe multisystem disease — NOT adult CPEO",
        ],

        "clinical_alerts": [
            {
                "alert":  "L-STRAND NGS PITFALL — MT-TC variants MISSED by H-strand pipelines",
                "detail": "Dedicated mtDNA panel with mandatory reverse-complement QC for rCRS 5761–5826. Entire cluster 5587–5891 requires L-strand processing.",
            },
            {
                "alert":  "COMPOUND DELETION RISK — MT-TN/TC and MT-TC/TY boundaries",
                "detail": "32 nt gap between MT-TN and MT-TC is a common deletion hotspot. 0 nt gap between MT-TC and MT-TY means any deletion spanning 5826 removes both tRNA-Cys AND tRNA-Tyr simultaneously.",
            },
            {
                "alert":  "UGA vs UGY DISTINCTION — CRITICAL for codon interpretation",
                "detail": "UGA → Trp (decoded by MT-TW, not MT-TC). MT-TC decodes only UGY (UGU/UGC = Cys). Do not confuse Cys and Trp codon tables in mt-genetic-code interpretation.",
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
    variants = ["m.5814AG", "m.5763CT", "m.5801GA", "m.5793AG", "LargeDel"]
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
                "entity":       "MT-TC (this gene)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Adult CPEO + myopathy; exercise intolerance; cardiomyopathy (m.5763CT); SNHL (m.5793AG)",
                "biochemistry": "CI+CIV deficiency; CII NORMAL; symmetric CI ≈ CIV",
                "ngs":          "L-strand mtDNA panel — L-strand QC mandatory (rCRS 5761–5826)",
                "distinctive":  "L-strand NGS pitfall; GCA anticodon Cys-only; CARS2 nuclear DDx neonatal",
            },
            {
                "entity":       "CARS2 (nuclear DDx)",
                "inheritance":  "AR nuclear — biallelic",
                "phenotype":    "Neonatal/infantile severe OXPHOS: epileptic encephalopathy, lactic acidosis",
                "biochemistry": "Severe combined OXPHOS deficiency (CI+CIII+CIV); mt-Cys-tRNA aminoacylation failure",
                "ngs":          "WES detects CARS2 — does NOT detect MT-TC",
                "distinctive":  "NEONATAL/INFANTILE onset — NOT adult CPEO; severe multisystem; AR not maternal",
            },
            {
                "entity":       "MT-TN (tRNA-Asn, 5657–5729)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; SNHL (m.5692AG); cardiomyopathy (m.5728AG)",
                "biochemistry": "CI+CIV deficiency; CII NORMAL; symmetric CI ≈ CIV",
                "ngs":          "L-strand mtDNA panel — L-strand QC rCRS 5657–5729",
                "distinctive":  "SECOND L-strand tRNA (5657–5729); GTT anticodon Asn; NARS2 DDx Perrault",
            },
            {
                "entity":       "MT-TY (tRNA-Tyr, 5826–5891)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy — fourth L-strand tRNA",
                "biochemistry": "CI+CIV deficiency; CII NORMAL",
                "ngs":          "L-strand mtDNA panel — L-strand QC rCRS 5826–5891",
                "distinctive":  "FOURTH L-strand tRNA; GTA anticodon Tyr; immediately 3′ of MT-TC at shared 5826",
            },
            {
                "entity":       "MT-TA (tRNA-Ala, 5587–5655)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; SNHL (m.5650AG); cardiomyopathy (m.5655AG)",
                "biochemistry": "CI+CIV deficiency; CII NORMAL",
                "ngs":          "L-strand mtDNA panel — L-strand QC rCRS 5587–5655",
                "distinctive":  "FIRST L-strand tRNA; GGC anticodon full-box Ala; AARS2 DDx ovario-leukodystrophy",
            },
            {
                "entity":       "MT-TW (tRNA-Trp, 5512–5579)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "CPEO + myopathy; UGA-Trp recoding disruption → CIV > CI asymmetry",
                "biochemistry": "CI+CIV deficiency; CIV preferentially worse at anticodon variants",
                "ngs":          "H-strand (MT-TW itself); L-strand cluster begins immediately 3′",
                "distinctive":  "UGA → Trp (NOT Cys/STOP); CIV asymmetry; WARS2 DDx neonatal OXPHOS",
            },
            {
                "entity":       "MT-TL1 (MELAS)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Stroke-like episodes, CPEO, ragged-red fibres",
                "biochemistry": "Pan-OXPHOS CI+CIII+CIV deficiency",
                "ngs":          "H-strand mtDNA panel — m.3243AG most common",
                "distinctive":  "STROKE-LIKE EPISODES and MELAS: NOT seen in MT-TC",
            },
            {
                "entity":       "MT-TK (MERRF)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Myoclonic epilepsy, ataxia, ragged-red fibres",
                "biochemistry": "CI+CIV deficiency; elevated lactate",
                "ngs":          "H-strand mtDNA panel — m.8344AG most common",
                "distinctive":  "MYOCLONIC EPILEPSY: NOT seen in MT-TC adult CPEO",
            },
            {
                "entity":       "MT-TI (m.4300AG HCM)",
                "inheritance":  "Maternal heteroplasmic",
                "phenotype":    "Isolated hypertrophic cardiomyopathy — DISTINCTIVE",
                "biochemistry": "CI+CIV deficiency predominates in cardiac muscle",
                "ngs":          "H-strand mtDNA panel",
                "distinctive":  "ISOLATED HCM: NOT seen in MT-TC (cardiomyopathy secondary to CPEO+myopathy)",
            },
            {
                "entity":       "BTBGD / SLC19A3",
                "inheritance":  "AR nuclear",
                "phenotype":    "Biotin-thiamine-responsive Leigh-like — symmetric BG signals",
                "biochemistry": "Normal OXPHOS on treatment",
                "ngs":          "WES",
                "distinctive":  "TREATABLE — biotin + thiamine trial MANDATORY before attributing Leigh MRI to MT-TC",
            },
        ],

        "management_by_variant": [
            {"variant": "m.5814AG", "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Annual ophthalmology; ptosis surgery PRN; L-strand NGS QC verification; CoQ10 + riboflavin"},
            {"variant": "m.5763CT", "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "Annual echo + Holter; T-stem variant — cardiomyopathy risk at >50% heteroplasmy; beta-blocker if HCM/DCM"},
            {"variant": "m.5801GA", "cpeo_risk": "High", "cardio_risk": "Low",      "key_action": "Full OXPHOS surveillance; BTBGD exclusion; Leigh MRI monitoring; early multidisciplinary referral"},
            {"variant": "m.5793AG", "cpeo_risk": "Mod",  "cardio_risk": "Low",      "key_action": "Audiology + cochlear implant; anticodon-loop variant — SNHL in ~50%; watch OXPHOS progression"},
            {"variant": "LargeDel", "cpeo_risk": "High", "cardio_risk": "Moderate", "key_action": "KSS surveillance (cardiac + retina + endocrine); compound MT-TC+TY loss check; CSF lactate; L-strand QC rCRS 5761–5891"},
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
            {"intervention": "Beta-blocker",        "evidence": "First-line if cardiomyopathy (m.5763CT); replaces amiodarone"},
            {"intervention": "GIR 6–8 mg/kg/min",  "evidence": "MANDATORY perioperative — prevents catabolic crisis; never fast in mt disease"},
        ],

        "cohort_statistics": stats,
    }


def get_definitions():
    return {
        "title": "MT-TC — Clinical & Molecular Definitions",
        "gene_definitions": [
            {"term": "MT-TC",               "definition": "Mitochondrially encoded tRNA-Cys gene; L-strand, rCRS 5761–5826, 66 nt (OMIM *590020)"},
            {"term": "tRNA-Cys",            "definition": "Transfer RNA for Cysteine; GCA anticodon, decoding UGC (Watson-Crick) and UGU (wobble G34:U) — both Cys codons in human mt code"},
            {"term": "GCA anticodon",       "definition": "Anticodon of tRNA-Cys (positions 34–36 of the tRNA); G34 wobble position reads both UGC and UGU; Cys is exclusively decoded by MT-TC in human mitochondria"},
            {"term": "UGY codon box",       "definition": "UGU and UGC both encode Cysteine — the 'UGY' split of the UG codon box (UGA→Trp by MT-TW; UGG→Trp by MT-TW; UGU/UGC→Cys by MT-TC)"},
            {"term": "L-strand cluster",    "definition": "MT-TA (5587–5655), MT-TN (5657–5729), MT-TC (5761–5826), MT-TY (5826–5891) — four consecutive L-strand tRNA genes; entire 304 nt block rCRS 5587–5891 requires L-strand QC"},
            {"term": "rCRS",                "definition": "Revised Cambridge Reference Sequence — standard human mtDNA numbering (GenBank NC_012920.1)"},
            {"term": "Heteroplasmy",        "definition": "Mixture of wild-type and mutant mtDNA molecules within a cell/tissue; severity correlates with mutant load (%)"},
        ],
        "biochemical_definitions": [
            {"term": "CI deficiency",       "definition": "NADH:ubiquinone oxidoreductase (Complex I) deficiency — 7 mtDNA-encoded subunits; Fe-S cluster Cys ligands in NDUFS1/2/7/8 are tRNA-Cys dependent"},
            {"term": "CII (normal)",        "definition": "Succinate dehydrogenase — entirely nuclear-encoded; remains normal in mt-tRNA disease (diagnostic discriminator)"},
            {"term": "CIV deficiency",      "definition": "Cytochrome c oxidase (COX) deficiency — 3 mtDNA-encoded subunits including COX1/COX2/COX3; structural Cys residues in COX1/COX2 copper-binding domains"},
            {"term": "Symmetric CI ≈ CIV",  "definition": "MT-TC produces symmetric CI ≈ CIV deficiency — contrast with MT-TW (CIV preferentially worse due to UGA-Trp recoding disruption in MT-CO1)"},
            {"term": "Fe-S cluster (CI)",   "definition": "Iron-sulfur clusters in CI are ligated by conserved Cys residues in NDUFS1, NDUFS2, NDUFS7, NDUFS8; tRNA-Cys depletion impairs elongation at Cys-containing Fe-S subunits"},
            {"term": "mt-translation fingerprint", "definition": "Combined CI+CIV deficiency with CII NORMAL — characteristic of pathogenic mt-tRNA gene mutations"},
            {"term": "Ragged-Red Fibres",   "definition": "Subsarcolemmal mitochondrial accumulation on modified Gomori trichrome; SDH-positive/COX-negative on sequential histochemistry"},
        ],
        "clinical_definitions": [
            {"term": "CPEO",                "definition": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive ophthalmoplegia; predominant MT-TC phenotype"},
            {"term": "KSS",                 "definition": "Kearns-Sayre Syndrome — CPEO + pigmentary retinopathy + cardiac conduction defect; often large mtDNA deletion spanning MT-TC and adjacent tRNAs"},
            {"term": "CARS2 deficiency",    "definition": "OMIM *612800 — AR biallelic CARS2 mutations → mitochondrial Cys-tRNA aminoacylation failure; neonatal/infantile severe OXPHOS deficiency + epileptic encephalopathy — NOT adult CPEO"},
            {"term": "Compound tRNA loss",  "definition": "Large deletions spanning MT-TC/MT-TY (shared boundary at 5826) or MT-TN/MT-TC (32 nt gap at 5729–5761): simultaneous loss of two mt-tRNA species"},
            {"term": "BTBGD / SLC19A3",    "definition": "Biotin-Thiamine-responsive Basal Ganglia Disease — AR treatable Leigh mimic; mandatory exclusion before attributing Leigh MRI to MT-TC"},
            {"term": "UGA codon note",      "definition": "UGA = TRYPTOPHAN in mt genetic code (decoded by MT-TW, UCA anticodon); UGA is NOT a Cys codon; MT-TC decodes only UGY (UGU/UGC = Cys)"},
        ],
        "ngs_definitions": [
            {"term": "L-strand encoding",   "definition": "MT-TC (rCRS 5761–5826) is L-strand encoded; standard NGS H-strand calls MISS variants; reverse-complement pipeline required"},
            {"term": "L-strand NGS pitfall","definition": "H-strand-only NGS pipelines fail to detect L-strand tRNA variants; clinically identical pitfall to MT-TA (5587–5655), MT-TN (5657–5729), MT-TQ (4329–4400), MT-TE (14674–14742), MT-TP (15956–16023), MT-ND6 (14149–14673)"},
            {"term": "Four-gene L-strand cluster", "definition": "MT-TA, MT-TN, MT-TC, MT-TY occupy rCRS 5587–5891 — all L-strand; QC must cover the entire block; MT-TC/MT-TY shared boundary at 5826 requires careful deletion calling"},
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
            {"ref": "Hallmann-2014-CARS2",            "citation": "Hallmann K et al. (2014) A homozygous splice-site mutation in CARS2 is associated with progressive myoclonic epilepsy. Neurology 83:2183–2187"},
            {"ref": "Dallabona-2014-CARS2",           "citation": "Dallabona C et al. (2014) Novel (ovario)leukodystrophy related to AARS2 mutations (CARS2 context). Brain 137:2193–2203"},
            {"ref": "Sprinzl-2005-NAR-tRNA",          "citation": "Sprinzl M & Vassilenko KS (2005) Compilation of tRNA sequences and sequences of tRNA genes. Nucleic Acids Res 33:D139–D140"},
            {"ref": "Barrell-1979-Science",           "citation": "Barrell BG et al. (1979) A different genetic code in human mitochondria. Nature 282:189–194"},
        ],
    }
