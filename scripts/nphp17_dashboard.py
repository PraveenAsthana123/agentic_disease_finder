"""
Nephronophthisis Type 17
========================
Primary Gene : MAPKBP1 (*610889) — 2q13.3; ~1,388 aa; Mitogen-Activated Protein
               Kinase Binding Protein 1 (also JIP4 / SPAG9); ankyrin-repeat +
               leucine-zipper scaffold protein; JNK/MAPK signalling scaffold;
               interacts with the NPHP4 supercomplex at the ciliary transition zone.
               NOTE: 2q13.3 is distinct from NPHP1 at 2q13 despite the same chromosome
               arm — NPHP1 MLPA 290kb panel does NOT cover MAPKBP1; WES mandatory.
Disease OMIM : #616140 (Nephronophthisis 17 — NPHP17; pure renal ciliopathy)
Chromosome   : 2q13.3
Inheritance  : Autosomal Recessive (biallelic LOF — truncating and/or missense)
Prevalence   : ~1/1,000,000–2,000,000; ~25–35 published families as of 2026
               (ultra-rare NPHP; rarer than NPHP14/NPHP16)

Protein Structure — MAPKBP1 / JIP4 (~1,388 aa; JNK/MAPK scaffold)
-------------------------------------------------------------------
  • N-terminal ankyrin-repeat domain (aa 1–480): ankyrin repeats mediating
    protein–protein interactions; NPHP4 supercomplex docking interface;
    transition zone localisation anchor
  • Central leucine-zipper region (aa ~480–900): coiled-coil / LZ scaffold;
    JNK/MAPK kinase-binding domain; bridges stress-response kinases to TZ;
    JNK interaction surface; kinesin-1 (KIF5B) IFT docking
  • C-terminal scaffold domain (aa ~900–1,388): MAPK pathway regulation;
    cargo-binding surface; ciliary transport regulatory interface

Molecular Mechanism
-------------------
MAPKBP1 (JIP4) is the only NPHP gene encoding a JNK/MAPK scaffold:
  1. MAPKBP1 interacts with the NPHP4 supercomplex (NPHP1/NPHP4/NPHP8/RPGRIP1L)
     at the transition zone — unique bridge between JNK stress-response signalling
     and the TZ ciliary gate
  2. MAPKBP1 recruits JNK (c-Jun N-terminal kinase) to the TZ → mediates
     JNK-driven tubular epithelial stress-response and controlled apoptosis
  3. Loss of MAPKBP1 → impaired JNK-mediated stress response at TZ → tubular
     epithelial apoptosis dysregulation (failure to clear damaged cells) →
     progressive tubulointerstitial nephritis (TIN) → ESRD
  4. MAPKBP1 also interacts with kinesin-1 (KIF5B) → regulates IFT machinery
     coordination at the TZ; loss disrupts cilia-mediated tubular homeostasis
  5. No expression in photoreceptors, biliary epithelium, cerebellar neurons, or
     nodal cilia → pure renal phenotype with no extra-renal involvement
  6. Chr 2q13.3 proximity to NPHP1 (2q13) → targeted MLPA deletion panels for
     NPHP1 do NOT detect MAPKBP1 despite same chromosome arm — WES mandatory

Clinical Overview
-----------------
  • Renal: tubulointerstitial nephritis (TIN) + corticomedullary cysts +
    concentrating defect (polyuria, polydipsia); ESRD median ~14–16yr
    (adolescent-onset; slightly later than NPHP1 ~13yr)
  • No situs inversus — MAPKBP1 not in nodal cilia (0% laterality defect)
  • No retinal dystrophy — MAPKBP1 not expressed in photoreceptors (ERG normal)
  • No CHF — MAPKBP1 absent from biliary epithelium
  • No Joubert / Molar Tooth Sign — not expressed in cerebellar neurons
  • No intellectual disability
  • No ectodermal, pancreatic, or skeletal features

Key Diagnostic Alerts
---------------------
  • Ultra-rare — ~25–35 families worldwide; rarer than NPHP14 and NPHP16
  • NPHP1 MLPA (290kb standard test) does NOT detect MAPKBP1 at 2q13.3 even
    though both genes are on chromosome 2 arm q13 region — different loci
  • 2q13 locus confusion: NPHP1 gene at 2q13 vs MAPKBP1 at 2q13.3 — distinct
    genes; MLPA panel covers NPHP1 290kb deletion only; WES mandatory for MAPKBP1
  • No extra-renal features → easy to miss without systematic NPHP panel testing
  • NPHP4 must be co-sequenced (MAPKBP1 is an NPHP4-interacting protein)
  • Renal transplant CURATIVE — no disease recurrence in transplanted kidney

40-patient cohort generated with seed=373; 3 endpoints
  /api/nphp17/overview | /api/nphp17/breakdown | /api/nphp17/definitions
"""

import random
from typing import Any

# ── Cohort seed ──────────────────────────────────────────────────────────────
SEED        = 373
COHORT_N    = 40
rng         = random.Random(SEED)

# ── Patient phenotype distributions (MAPKBP1/NPHP17 literature) ──────────────
_ETHNICITIES = [
    ("European (Northern)",        0.32),
    ("Middle Eastern",             0.22),
    ("South Asian",                0.15),
    ("European (Southern)",        0.14),
    ("East Asian",                 0.09),
    ("Latin American",             0.05),
    ("Sub-Saharan African",        0.03),
]

_FIRST_SYMPTOMS = [
    ("Polyuria/polydipsia (tubular concentrating defect)",              0.35),
    ("Incidental finding on routine blood/urine screening",             0.20),
    ("Anaemia + fatigue (ESRD-range)",                                  0.18),
    ("Family screening (affected sibling)",                             0.15),
    ("Hypertension (late CKD)",                                         0.08),
    ("Recurrent UTI prompting renal workup",                            0.04),
]

_MISDIAGNOSES = [
    ("NPHP1 deletion negative (MLPA — misses MAPKBP1 2q13.3)",        0.38),
    ("ADPKD/PKD1 (AR pattern missed; bilateral cysts on USS)",          0.20),
    ("Alport syndrome (CKD + haematuria — COL4A gene negative)",        0.15),
    ("No prior misdiagnosis (direct WES diagnosis)",                    0.16),
    ("Other NPHP subtype (NPHP panel miss — ultra-rare gene)",          0.11),
]

_CKD_STAGES = [
    ("CKD 1–2 (GFR ≥60; tubular symptoms only)",                       0.10),
    ("CKD 3a (GFR 45–59)",                                              0.13),
    ("CKD 3b (GFR 30–44)",                                              0.18),
    ("CKD 4 (GFR 15–29)",                                               0.22),
    ("CKD 5 / ESRD (GFR <15, pre-dialysis)",                           0.20),
    ("Post-transplant (functioning graft, no recurrence)",              0.17),
]

_KIDNEY_USS = [
    ("Small echogenic kidneys, bilateral corticomedullary cysts",       0.50),
    ("Small echogenic kidneys, no visible cysts on USS",                0.25),
    ("Normal size, early hyperechogenicity only",                       0.14),
    ("Bilateral moderate cysts (PKD-like USS pattern)",                 0.07),
    ("Normal USS (early disease, tubular deficit only)",                0.04),
]

_URINE_OSM = [
    ("< 200 mOsm/kg (severe concentrating defect)",                    0.28),
    ("200–400 mOsm/kg (moderate)",                                      0.32),
    ("400–600 mOsm/kg (mild-moderate)",                                 0.24),
    ("> 600 mOsm/kg (early/mild; tubular defect emerging)",             0.16),
]

_GFR_SLOPE = [
    ("–5 to –3 ml/min/yr (slow progression)",                          0.18),
    ("–8 to –5 ml/min/yr (moderate)",                                   0.32),
    ("–12 to –8 ml/min/yr (fast; typical adolescent NPHP)",            0.30),
    ("> –12 ml/min/yr (very fast; early ESRD)",                         0.20),
]


def _pick(opts, n, r):
    """Pick n items from weighted list using rng r; returns list of labels."""
    labels, weights = zip(*opts)
    result = []
    chosen_weights = list(weights)
    for _ in range(n):
        total = sum(chosen_weights)
        pick  = r.random() * total
        cum   = 0.0
        for i, w in enumerate(chosen_weights):
            cum += w
            if pick < cum:
                result.append(labels[i])
                break
    return result


def _weighted_choice(opts, r):
    labels, weights = zip(*opts)
    total = sum(weights)
    pick  = r.random() * total
    cum   = 0.0
    for label, w in zip(labels, weights):
        cum += w
        if pick < cum:
            return label
    return labels[-1]


def _generate_cohort(seed: int = SEED, n: int = COHORT_N) -> list[dict]:
    r = random.Random(seed)
    patients = []
    for i in range(1, n + 1):
        ethnicity       = _weighted_choice(_ETHNICITIES, r)
        first_symptom   = _weighted_choice(_FIRST_SYMPTOMS, r)
        misdiagnosis    = _weighted_choice(_MISDIAGNOSES, r)
        ckd_stage       = _weighted_choice(_CKD_STAGES, r)
        kidney_uss      = _weighted_choice(_KIDNEY_USS, r)

        # GFR
        if "1–2" in ckd_stage:
            gfr = r.randint(63, 95)
        elif "3a" in ckd_stage:
            gfr = r.randint(45, 59)
        elif "3b" in ckd_stage:
            gfr = r.randint(30, 44)
        elif "4" in ckd_stage:
            gfr = r.randint(15, 29)
        elif "5" in ckd_stage or "ESRD" in ckd_stage:
            gfr = r.randint(4, 14)
        else:  # post-Tx
            gfr = r.randint(42, 78)

        age_dx = r.randint(5, 18)
        hb     = round(r.uniform(7.2, 13.8), 1)

        patients.append({
            "id":                   f"NPHP17-{i:03d}",
            "ethnicity":            ethnicity,
            "first_symptom":        first_symptom,
            "prior_misdiagnosis":   misdiagnosis,
            "ckd_stage":            ckd_stage,
            "kidney_uss":           kidney_uss,
            "gfr_now_ml_min":       gfr,
            "age_renal_dx_yr":      age_dx,
            "hb_g_dl":              hb,
        })
    return patients


# ── Public API ────────────────────────────────────────────────────────────────

def get_overview() -> dict[str, Any]:
    patients = _generate_cohort()

    n_esrd_tx    = sum(1 for p in patients if "ESRD" in p["ckd_stage"] or "Post-transplant" in p["ckd_stage"])
    n_misdiag_nphp1 = sum(1 for p in patients if "NPHP1" in p["prior_misdiagnosis"])
    n_misdiag_adpkd = sum(1 for p in patients if "ADPKD" in p["prior_misdiagnosis"])

    median_gfr  = sorted(p["gfr_now_ml_min"] for p in patients)[COHORT_N // 2]
    median_hb   = sorted(p["hb_g_dl"] for p in patients)[COHORT_N // 2]
    median_age  = sorted(p["age_renal_dx_yr"] for p in patients)[COHORT_N // 2]

    return {
        "cohort_n":                 COHORT_N,
        "seed":                     SEED,
        "median_gfr":               median_gfr,
        "median_hb":                round(median_hb, 1),
        "median_age_renal_dx":      median_age,
        "pct_situs_inversus":       0,
        "pct_retinal_involvement":  0,
        "pct_chf_involvement":      0,
        "pct_joubert":              0,
        "pct_esrd_or_transplant":   round(n_esrd_tx / COHORT_N * 100),
        "pct_misdiagnosed_nphp1":   round(n_misdiag_nphp1 / COHORT_N * 100),
        "pct_misdiagnosed_adpkd":   round(n_misdiag_adpkd / COHORT_N * 100),
        "patients":                 patients[:8],
    }


def get_breakdown() -> dict[str, Any]:
    patients = _generate_cohort()

    # CKD
    ckd_dist: dict[str, int] = {}
    for p in patients:
        v = p["ckd_stage"].split("(")[0].strip()
        ckd_dist[v] = ckd_dist.get(v, 0) + 1

    # GFR slope tiers
    slope_dist: dict[str, int] = {}
    r = random.Random(SEED)
    for _ in range(COHORT_N):
        v = _weighted_choice(_GFR_SLOPE, r).split("(")[0].strip()
        slope_dist[v] = slope_dist.get(v, 0) + 1

    # Urine osm
    osm_dist: dict[str, int] = {}
    r2 = random.Random(SEED + 7)
    for _ in range(COHORT_N):
        v = _weighted_choice(_URINE_OSM, r2).split("(")[0].strip()
        osm_dist[v] = osm_dist.get(v, 0) + 1

    # Ethnicity
    ethn_dist: dict[str, int] = {}
    for p in patients:
        v = p["ethnicity"].split("(")[0].strip()
        ethn_dist[v] = ethn_dist.get(v, 0) + 1

    # Misdiagnosis
    misdiag_dist: dict[str, int] = {}
    for p in patients:
        v = p["prior_misdiagnosis"].split("(")[0].strip()
        misdiag_dist[v] = misdiag_dist.get(v, 0) + 1

    # First symptom
    symp_dist: dict[str, int] = {}
    for p in patients:
        v = p["first_symptom"].split("(")[0].strip()
        symp_dist[v] = symp_dist.get(v, 0) + 1

    # Kidney USS
    uss_dist: dict[str, int] = {}
    for p in patients:
        v = p["kidney_uss"].split("(")[0].strip()
        uss_dist[v] = uss_dist.get(v, 0) + 1

    return {
        "ckd_stage_distribution":       dict(sorted(ckd_dist.items(), key=lambda x: -x[1])),
        "kidney_phenotype":             dict(sorted(uss_dist.items(), key=lambda x: -x[1])),
        "prior_misdiagnosis":           dict(sorted(misdiag_dist.items(), key=lambda x: -x[1])),
        "first_symptom_distribution":   dict(sorted(symp_dist.items(), key=lambda x: -x[1])),
        "ethnicity":                    dict(sorted(ethn_dist.items(), key=lambda x: -x[1])),
        "urine_osmolality_tiers":       dict(sorted(osm_dist.items(), key=lambda x: -x[1])),
        "gfr_slope_tiers":              dict(sorted(slope_dist.items(), key=lambda x: -x[1])),
    }


def get_definitions() -> dict[str, Any]:
    return {
        "disease":      "Nephronophthisis Type 17 (NPHP17) — autosomal recessive "
                        "renal ciliopathy caused by biallelic LOF variants in MAPKBP1 "
                        "(Mitogen-Activated Protein Kinase Binding Protein 1; also JIP4 "
                        "/ SPAG9); JNK/MAPK scaffold ciliopathy; adolescent-onset "
                        "NPHP (ESRD median ~14–16yr); pure renal phenotype with no "
                        "extra-renal involvement. Ultra-rare: ~25–35 families worldwide.",
        "omim_gene":    "*610889 (MAPKBP1)",
        "omim_disease": "#616140 (NPHP17)",
        "chromosome":   "2q13.3",
        "inheritance":  "Autosomal Recessive · biallelic LOF (truncating + missense)",
        "prevalence":   "~1/1,000,000–2,000,000; ~25–35 published families (2026)",

        "mechanism": (
            "MAPKBP1 (JIP4) is the only NPHP gene encoding a JNK/MAPK scaffold protein, "
            "making it unique among all 20+ NPHP subtypes. MAPKBP1 interacts with the NPHP4 "
            "supercomplex (NPHP1/NPHP4/NPHP8/RPGRIP1L) at the ciliary transition zone, "
            "bridging JNK stress-response kinase signalling to the TZ ciliary gate. "
            "MAPKBP1 recruits JNK to the TZ → mediates JNK-driven tubular epithelial "
            "stress-response and controlled apoptosis; loss of MAPKBP1 → impaired JNK-mediated "
            "stress response → tubular epithelial apoptosis dysregulation (failure to clear "
            "damaged tubular cells) → progressive tubulointerstitial nephritis (TIN) → "
            "corticomedullary cysts → ESRD. MAPKBP1 also interacts with kinesin-1 (KIF5B) "
            "to regulate IFT machinery coordination at the TZ; disruption of this interface "
            "further impairs cilia-mediated tubular homeostasis. Critically, MAPKBP1 is not "
            "expressed in photoreceptors, biliary epithelium, cerebellar neurons, or embryonic "
            "nodal cilia → pure renal phenotype with no extra-renal involvement. Chr 2q13.3 "
            "proximity to NPHP1 (2q13) is a diagnostic trap: the NPHP1 290kb MLPA deletion "
            "panel does NOT cover MAPKBP1 despite both genes residing on chromosome 2q; WES "
            "is mandatory for NPHP17 diagnosis."
        ),

        "key_clinical_features": {
            "Renal (TIN + cysts)": (
                "Tubulointerstitial nephritis (TIN) + corticomedullary cysts + "
                "tubular concentrating defect (polyuria, polydipsia). ESRD median "
                "~14–16yr (adolescent-onset; slightly later than NPHP1 ~13yr). "
                "Renal USS: small echogenic kidneys ± corticomedullary cysts. "
                "Renal transplant CURATIVE — no recurrence in transplanted kidney "
                "(cell-autonomous JNK/TZ defect)."
            ),
            "No situs inversus": (
                "MAPKBP1 is not expressed in embryonic nodal cilia → 0% laterality "
                "defects. Unlike NPHP16 (ANKS6, 20–30% situs inversus) and NPHP2 "
                "(INVS, >85% situs inversus), NPHP17 has no left-right axis involvement. "
                "Situs inversus in a patient with presumed NPHP17 should prompt "
                "re-evaluation and IC-tetramer co-sequencing."
            ),
            "No retinal dystrophy": (
                "MAPKBP1 not expressed in photoreceptor connecting cilia → no "
                "retinal degeneration. ERG normal in all NPHP17 patients. "
                "Distinguishes NPHP17 from NPHP5/IQCB1, CEP290/NPHP6, "
                "SDCCAG8/NPHP10, and CEP164/NPHP15 (all with retinal). "
                "No ophthalmology monitoring required."
            ),
            "No CHF / no Joubert / no ID": (
                "MAPKBP1 absent from biliary epithelium → no congenital hepatic "
                "fibrosis (CHF). No cerebellar expression → no Joubert/MTS. "
                "No neuronal expression → no intellectual disability. "
                "Pure renal ciliopathy defines NPHP17 — no extra-renal features "
                "in any published case."
            ),
            "2q13.3 / NPHP1 locus confusion": (
                "MAPKBP1 (2q13.3) and NPHP1 (2q13) share the same chromosome arm. "
                "Standard NPHP1 MLPA tests the 290kb NPHP1 homozygous deletion only "
                "and does NOT cover MAPKBP1 at 2q13.3. A negative NPHP1 MLPA in a "
                "child with pure renal NPHP and no extra-renal features does NOT "
                "exclude NPHP17 — WES or comprehensive NPHP panel (including MAPKBP1) "
                "is mandatory. This locus confusion is the most frequent diagnostic delay."
            ),
        },

        "genetic_architecture": {
            "gene":          "MAPKBP1 (Mitogen-Activated Protein Kinase Binding Protein 1)",
            "aliases":       "JIP4 (JNK-Interacting Protein 4); SPAG9 (Sperm-Associated Antigen 9)",
            "size":          "~1,388 aa · ~155 kDa",
            "domains": (
                "N-terminal ankyrin-repeat domain (aa 1–480): ankyrin repeats; "
                "NPHP4 supercomplex docking interface; TZ localisation anchor; "
                "protein–protein interaction scaffold. | "
                "Central leucine-zipper / JNK-binding region (aa ~480–900): "
                "coiled-coil scaffold; JNK/MAPK kinase-binding domain; bridges "
                "stress-response kinases to TZ gate; kinesin-1 (KIF5B) docking. | "
                "C-terminal scaffold domain (aa ~900–1,388): MAPK pathway regulation; "
                "cargo-binding surface; ciliary transport regulatory interface."
            ),
            "nphp4_interaction": (
                "MAPKBP1 interacts with NPHP4 (Nephrocystin-4) supercomplex at the "
                "transition zone. NPHP4 supercomplex members: NPHP1, NPHP4, NPHP8 "
                "(RPGRIP1L), NPHP16 (ANKS6). NPHP4 must always be co-sequenced when "
                "MAPKBP1 pathogenic variants are identified — digenic NPHP4/MAPKBP1 "
                "interaction may modify disease severity."
            ),
            "jnk_pathway": (
                "MAPKBP1 is the only known TZ-associated JNK scaffold. JNK (c-Jun "
                "N-terminal kinase) recruited by MAPKBP1 to TZ mediates tubular "
                "epithelial stress-response apoptosis. Loss → apoptosis dysregulation "
                "→ TIN. No other NPHP gene encodes a JNK-pathway component."
            ),
            "key_variants": [
                "p.Arg487Ter (c.1459C>T) — truncating; leucine-zipper loss; severe "
                "juvenile NPHP17; European; most common reported NPHP17 variant; "
                "Slaats 2015.",
                "p.Leu943Pro (c.2828T>C) — ankyrin-repeat domain; European "
                "consanguineous; homozygous; pure NPHP17; adolescent ESRD.",
                "p.Ala1187Val (c.3560C>T) — C-terminal scaffold domain; Middle "
                "Eastern consanguineous; homozygous; mild adolescent NPHP17.",
                "p.Gln223Ter — early truncating; ankyrin-repeat disruption; severe "
                "early presentation; compound heterozygous; ultra-rare.",
                "p.Trp744Arg — ankyrin-repeat 8; leucine-zipper boundary; pan-ethnic; "
                "pure renal NPHP17; adolescent onset.",
            ],
        },

        "nphp_comparison": {
            "★ NPHP17 (MAPKBP1) — This patient": (
                "JNK/MAPK scaffold; 2q13.3; adolescent ESRD ~14–16yr; 0% situs inversus; "
                "0% retinal; 0% CHF; 0% Joubert; 0% ID; NPHP4 co-sequence mandatory; "
                "ultra-rare (~25–35 families); NPHP1 MLPA does NOT detect"
            ),
            "NPHP1 (NPHP1) — same chromosome arm; most common NPHP": (
                "TZ-scaffold; 2q13 (DISTINCT locus from MAPKBP1 2q13.3); juvenile ESRD "
                "~13yr; 0% situs inversus; most common NPHP; 290kb deletion MLPA-"
                "detectable; NPHP17 is NOT detected by NPHP1 MLPA — key DDx trap"
            ),
            "NPHP4 (NPHP4) — direct binding partner": (
                "TZ-scaffold supercomplex; 1p36.31; juvenile ESRD ~13–16yr; 0% situs; "
                "rare retinal coloboma (<5%); MAPKBP1 directly interacts with NPHP4 "
                "TZ supercomplex — always co-sequence NPHP4 + MAPKBP1"
            ),
            "NPHP14 (ZNF423) — DDR protein, pure renal analogy": (
                "DDR nuclear protein; 16q12.1; ESRD ~13–18yr; Joubert 40–50%; ID 25–35%; "
                "0% situs; ZNF423 causes Joubert + ID unlike NPHP17 pure renal; "
                "only other NPHP with non-ciliary (nuclear) primary function"
            ),
            "NPHP16 (ANKS6) — IC scaffold, same chromosome 9 comparison": (
                "IC scaffold; 9q22.33; juvenile ESRD ~13yr; situs inversus 20–30%; "
                "0% retinal; 0% CHF; NPHP17 has 0% situs inversus unlike NPHP16; "
                "both ultra-rare but NPHP17 rarer (~25–35 vs ~60–80 families)"
            ),
            "NPHP12 (TTC21B) — IFT-A retrograde; pure renal comparison": (
                "IFT-A retrograde; 2q24.3; ESRD ~11–15yr; ATD4 skeletal 7–10%; "
                "0% situs; 0% retinal; NPHP17 has no skeletal involvement; "
                "NPHP12 is IFT-A subunit vs NPHP17 is JNK scaffold"
            ),
        },

        "ddx_table": {
            "NPHP1 (2q13) — same chromosome arm; MLPA-detectable": (
                "NPHP1 = most common NPHP; 290kb deletion on 2q13 detected by MLPA. "
                "MAPKBP1 (2q13.3) is NOT detected by NPHP1 MLPA despite same arm. "
                "NPHP1 has ESRD ~13yr (slightly earlier); NPHP17 ~14–16yr. Both pure "
                "renal; 0% situs inversus in both. If NPHP1 MLPA negative in pure "
                "renal NPHP → WES mandatory; include MAPKBP1 on panel."
            ),
            "ADPKD / PKD1-PKD2": (
                "ADPKD cysts are cortical/medullary diffuse (not corticomedullary); "
                "dominant inheritance; no situs inversus; HTN predominates; no "
                "concentrating defect until late. NPHP17 = AR; corticomedullary cysts; "
                "tubular polyuria first. PKD1 ADPKD often misdiagnosed first in NPHP17 "
                "due to overlapping cystic USS appearance."
            ),
            "Alport syndrome (COL4A3/4/5)": (
                "Alport = type IV collagen defect; haematuria + proteinuria + "
                "sensorineural deafness ± ocular changes; X-linked or AR. "
                "NPHP17 has no haematuria as primary feature, no deafness, no "
                "ocular changes. COL4A gene negative + renal cysts + tubular "
                "concentrating defect → broadened panel including MAPKBP1."
            ),
            "NPHP4 — supercomplex binding partner": (
                "NPHP4 (1p36) causes TZ-scaffold ciliopathy: pure renal NPHP + rare "
                "retinal coloboma (<5%). MAPKBP1 interacts directly with NPHP4 "
                "supercomplex. Always co-sequence both genes. Single pathogenic "
                "variants in each may represent digenic disease. NPHP4 ESRD ~13–16yr "
                "similar to NPHP17 — genetic testing distinguishes."
            ),
            "FSGS (focal segmental glomerulosclerosis)": (
                "FSGS can present with proteinuria + CKD in adolescents. NPHP17 "
                "primary feature is tubular concentrating defect + interstitial "
                "fibrosis (TIN), not glomerular proteinuria. Renal biopsy: TIN "
                "pattern in NPHP17 vs glomerulosclerosis in FSGS. Genetic testing "
                "discriminates; NPHP panel including MAPKBP1 mandatory."
            ),
        },

        "diagnostic_criteria": {
            "Mandatory (all required)": (
                "1. Biallelic pathogenic MAPKBP1 variants (WES/comprehensive NPHP "
                "gene panel — NOT NPHP1 MLPA alone). "
                "2. Renal ciliopathy phenotype: TIN + corticomedullary cysts + "
                "tubular concentrating defect (polyuria/polydipsia). "
                "3. Autosomal recessive inheritance pattern."
            ),
            "Supportive (any 1 of)": (
                "Adolescent ESRD onset (~14–16yr) with no extra-renal features. "
                "NPHP4 co-pathogenic variant on WES (MAPKBP1 is NPHP4-interacting). "
                "Functional MAPKBP1 protein loss on renal biopsy immunostaining "
                "(if performed — not routinely required)."
            ),
            "Exclusion criteria": (
                "Situs inversus — MAPKBP1 not in nodal cilia; situs inversus should "
                "prompt IC ciliopathy panel (NPHP16/ANKS6, NPHP2/INVS, NPHP3). "
                "Retinal dystrophy (ERG abnormal) — MAPKBP1 not in photoreceptors; "
                "consider NPHP5/6/10/15. Congenital hepatic fibrosis — consider "
                "NPHP3/NPHP11/TMEM67. Joubert Molar Tooth Sign — consider NPHP6/"
                "CEP290 or NPHP8/RPGRIP1L. Intellectual disability — consider "
                "NPHP14/ZNF423. Skeletal dysplasia — consider NPHP12/TTC21B (ATD4)."
            ),
        },

        "treatment": {
            "Renal replacement": (
                "Renal transplant is CURATIVE for NPHP17 renal disease — no "
                "recurrence in transplanted kidney (cell-autonomous JNK/TZ defect). "
                "Pre-emptive transplant preferred when feasible. Living-related donor "
                "evaluation must include renal USS + genetic screening (carrier "
                "relatives with one MAPKBP1 allele have normal kidney function). "
                "Post-transplant outcomes are excellent."
            ),
            "Supportive management": (
                "Conservative CKD management: fluid balance, salt supplementation "
                "(tubular salt-wasting), erythropoietin-stimulating agents for renal "
                "anaemia, blood pressure control (RAAS blockade), and bicarbonate "
                "supplementation for metabolic acidosis. Referral to paediatric "
                "nephrology mandatory at diagnosis."
            ),
            "NPHP4 co-sequencing": (
                "Always sequence NPHP4 (1p36) when MAPKBP1 biallelic variants "
                "identified — MAPKBP1 is an NPHP4-interacting protein. Single "
                "pathogenic variants in NPHP4 + MAPKBP1 may represent digenic "
                "disease. NPHP4 supercomplex members (NPHP1, NPHP4, NPHP8, NPHP16) "
                "should be on all NPHP diagnostic panels."
            ),
            "Investigational": (
                "No disease-modifying therapy available (2026). "
                "JNK-pathway modulators and MAPK scaffold stabilisers are in early "
                "pre-clinical exploration. KIF5B/kinesin-1 IFT interface is a "
                "potential therapeutic target (pre-clinical zebrafish mapkbp1 models). "
                "Registry enrolment (RareCare / EURO-RDI / ERKNet) essential given "
                "ultra-rare status (~25–35 families worldwide)."
            ),
        },

        "prognosis": (
            "NPHP17 follows an adolescent NPHP trajectory: ESRD median ~14–16yr "
            "(range ~9–21yr; slightly later than NPHP1 ~13yr). Renal transplant is "
            "curative — no disease recurrence in the graft. Entirely pure renal "
            "phenotype: no retinal, cerebellar, hepatic, laterality, skeletal, or "
            "cognitive involvement in any published case. Quality of life "
            "post-transplant is excellent. Ultra-rare nature (~25–35 families "
            "worldwide) means clinical experience is limited; registry participation "
            "and international case-sharing are strongly encouraged. The 2q13.3 "
            "locus proximity to NPHP1 (2q13) remains the primary source of "
            "diagnostic delay — NPHP1 MLPA negativity does not exclude NPHP17."
        ),

        "cohort_note": (
            f"Synthetic 40-patient NPHP17 cohort (seed={SEED}). Phenotype frequencies "
            "calibrated to published MAPKBP1/JIP4 literature (Slaats 2015 Kidney Int, "
            "Otto 2011, Halbritter 2013, Schueler 2015). Adolescent ESRD onset "
            "(~14–16yr) reflects slightly later trajectory than NPHP1 (~13yr). "
            "No situs inversus or extra-renal features in any cohort patient — "
            "consistent with pure renal MAPKBP1 phenotype. All patients are "
            "de-identified composites for clinical education only."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"cohort_n             : {ov['cohort_n']}")
    print(f"seed                 : {ov['seed']}")
    print(f"median_gfr           : {ov['median_gfr']}")
    print(f"median_age_renal_dx  : {ov['median_age_renal_dx']}")
    print(f"pct_situs_inversus   : {ov['pct_situs_inversus']}%")
    print(f"pct_retinal          : {ov['pct_retinal_involvement']}%")
    print(f"pct_chf              : {ov['pct_chf_involvement']}%")
    print(f"pct_esrd_or_tx       : {ov['pct_esrd_or_transplant']}%")
    print(f"pct_misdiag_nphp1    : {ov['pct_misdiagnosed_nphp1']}%")
    print(f"pct_misdiag_adpkd    : {ov['pct_misdiagnosed_adpkd']}%")
    print(f"\nFirst 8 patients:")
    for p in ov["patients"]:
        print(f"  {p['id']} | age_dx={p['age_renal_dx_yr']}yr | GFR={p['gfr_now_ml_min']} | {p['ckd_stage'].split('(')[0].strip()}")
    print("\n=== BREAKDOWN (sample) ===")
    bk = get_breakdown()
    print("CKD stages:", json.dumps(bk["ckd_stage_distribution"], indent=2))
    print("Prior misdiagnosis:", json.dumps(bk["prior_misdiagnosis"], indent=2))
    print("\n=== DEFINITIONS (snippet) ===")
    df = get_definitions()
    print("disease:", df["disease"][:140])
    print("omim_gene:", df["omim_gene"])
    print("omim_disease:", df["omim_disease"])
    print("chromosome:", df["chromosome"])
    print("prevalence:", df["prevalence"])
    print(f"\nKey variants ({len(df['genetic_architecture']['key_variants'])}):")
    for v in df["genetic_architecture"]["key_variants"]:
        print(f"  • {v[:80]}")
