"""
Nephronophthisis Type 20 / Joubert Syndrome 31 (NPHP20/JBTS31)
===============================================================
Primary Gene : CEP120 (*613446) — 5q23.2; ~1085 aa; Centrosomal Protein 120 kDa
               (also known as CCDC100 — Coiled-Coil Domain Containing 100)
               Centrosomal scaffold protein; required for daughter centriole elongation
               during mitosis and for ciliogenesis initiation at the mother centriole.
               CEP120 contains multiple ARM/HEAT-repeat domains and a C-terminal coiled-coil
               that anchors the protein to the distal tip of daughter centrioles.
               Loss of CEP120 → abnormally short daughter centrioles → defective basal body
               formation → failed primary cilia assembly → NPHP20 + Joubert syndrome (JBTS31).
Disease OMIM : JBTS31 (#617761) — Joubert Syndrome 31 (nephronophthisis-related ciliopathy);
               also designated NPHP20 in ciliopathy literature (renal-predominant cases);
               severe biallelic null alleles → Short-Rib Polydactyly Syndrome type 2B (SRPS2B).
Chromosome   : 5q23.2
Inheritance  : Autosomal Recessive (biallelic LOF — truncating + missense)
Prevalence   : ~1/800,000–2,000,000; <25 published families (2026); ultra-rare among NPHP/JBTS

Protein Structure — CEP120 (~1085 aa; centriole elongation scaffold)
--------------------------------------------------------------------
  • N-terminal ARM/HEAT repeat domain (aa ~1–500): ankyrin-like repeats that form an arc-
    shaped scaffold; protein-protein interaction platform; binds CPAP (CENPJ) during
    daughter centriole elongation; loss of ARM domain = no centriole elongation
  • Central coiled-coil domain (aa ~500–850): mediates self-oligomerization; important
    for CEP120 targeting to the distal daughter centriole tip; interacts with centrin;
    required for stability of the distal centriole cap complex
  • C-terminal regulatory domain (aa ~850–1085): CEP135/CNTRL interaction surface;
    binds SAS-4/CPAP; important for transition from centriole elongation to ciliogenesis
    initiation (CP110/CEP97 destabilization)

Molecular Mechanism
-------------------
CEP120 is a distal daughter-centriole scaffold with dual roles in centriogenesis and ciliogenesis:

  ROLE 1 — Daughter Centriole Elongation (Cell Cycle):
  1. CEP120 is recruited to the distal tip of daughter centrioles by CPAP (CENPJ) and SAS-4
  2. CEP120 organises the distal centriole scaffold together with CEP135 and CP110/CEP97
  3. Daughter centriole elongation requires CEP120; loss → short daughter centrioles (~80%
     normal length) that cannot serve as competent mother centrioles in the next cycle
  4. Consequence: progressive centriole shortening over cell generations →
     defective basal body formation → primary cilia assembly fails

  ROLE 2 — Ciliogenesis Initiation (Quiescence/G0):
  5. During cell cycle exit, the mother centriole docks to the plasma membrane
     via distal appendages (CEP83, SCLT1, FBF1, LRRC45, CEP164) to form the basal body
  6. CEP120 helps destabilize the CP110/CEP97 distal cap at the mother centriole tip →
     CP110 removal allows axoneme extension to begin (TTBK2 phosphorylates CEP164 to
     trigger CP110 removal; CEP120 co-operates with this cascade)
  7. Short/dysfunctional basal bodies from CEP120 loss cannot effectively anchor distal
     appendages → ciliary vesicle docking fails → primary cilia absent/severely truncated
  8. Renal tubular cilia failure → tubulointerstitial nephritis (TIN) + corticomedullary
     cysts + concentrating defect → ESRD
  9. Cerebellar neuronal cilia failure → Sonic Hedgehog (Shh/Gli) signalling disruption →
     cerebellar vermis hypoplasia + SCP elongation = Molar Tooth Sign → Joubert Syndrome 31
  10. Photoreceptor connecting cilium failure → rod-cone degeneration in ~25–35% of cases

Distinct from other NPHP subtypes:
  • NPHP15 (CEP164) / NPHP18 (CEP83): DISTAL APPENDAGE defects; centriole IS normal length
    but docking/ciliogenesis initiation fails. CEP120: CENTRIOLE ELONGATION defect upstream.
  • NPHP17 (MAPKBP1/JIP4): JNK kinase scaffold at TZ–NPHP4 supercomplex; structurally distinct
  • NPHP19 (IFT81): IFT-B anterograde core; transport defect after basal body is formed
  • CEP120 = Only NPHP subtype caused by daughter centriole elongation scaffold loss

Key Diagnostic Alerts
---------------------
  • CEP120 defect is UPSTREAM of distal appendage assembly — basal body itself is malformed
    before distal appendages (CEP83, CEP164) even assemble; contrast with NPHP15/NPHP18
  • SRPS2B (Short-Rib Polydactyly Syndrome type 2B): biallelic null CEP120 in ~8–10% →
    severe skeletal phenotype (narrow thorax, short ribs, postaxial polydactyly, limb shortening);
    neonatal lethality in the most severe SRPS2B cases; chest X-ray + skeletal survey mandatory
  • CPAP (CENPJ) is a direct CEP120 binding partner and causes JBTS9/MCPH6; MUST be
    co-sequenced; digenic CEP120 + CENPJ combinations documented
  • Renal transplant CURATIVE; retinal and cerebellar features are cell-autonomous and do NOT
    improve post-transplant
  • CEP120 at 5q23.2 — no chromosome arm confusion with other major NPHP genes (unique arm)
  • NPHP1 MLPA does NOT detect CEP120; WES mandatory

40-patient cohort generated with seed=379; 3 endpoints
  /api/nphp20/overview | /api/nphp20/breakdown | /api/nphp20/definitions
"""

import random
from typing import Any

# ── Cohort seed ──────────────────────────────────────────────────────────────
SEED        = 379
COHORT_N    = 40
rng         = random.Random(SEED)

# ── Patient phenotype distributions (CEP120/NPHP20/JBTS31 literature) ─────────
ETHNICITIES = [
    ("Middle Eastern (consanguineous)", 0.35),
    ("South Asian (consanguineous)", 0.22),
    ("European (non-consanguineous)", 0.20),
    ("North African (consanguineous)", 0.13),
    ("East Asian", 0.05),
    ("Latin American", 0.03),
    ("Sub-Saharan African", 0.02),
]

CKD_STAGES = [
    ("CKD 1 (GFR ≥90; early TIN, concentrating defect only)", 0.10),
    ("CKD 2 (GFR 60–89; polyuria, mild anaemia)", 0.15),
    ("CKD 3a (GFR 45–59; growth retardation)", 0.18),
    ("CKD 3b (GFR 30–44; progressive TIN, cysts)", 0.17),
    ("CKD 4 (GFR 15–29; pre-ESRD preparation)", 0.18),
    ("CKD 5/ESRD (GFR <15; awaiting/post-transplant)", 0.22),
]

KIDNEY_USS = [
    ("Bilateral small echogenic kidneys, corticomedullary cysts", 0.42),
    ("Bilateral normal-sized echogenic kidneys, early cysts", 0.22),
    ("Small kidneys, prominent corticomedullary cysts ≥5mm", 0.20),
    ("Small echogenic kidneys, no discrete cysts (early TIN)", 0.11),
    ("Transplanted (previous ESRD)", 0.05),
]

FIRST_SYMPTOMS = [
    ("Developmental delay + hypotonia (Joubert, JBTS31)", 0.30),
    ("Polyuria / polydipsia (tubular concentrating defect)", 0.28),
    ("Neonatal hypotonia + oculomotor apraxia (OMA)", 0.16),
    ("Incidental CKD on urine/blood screening", 0.14),
    ("Narrow thorax + polydactyly (SRPS2B severe alleles)", 0.07),
    ("Growth retardation + CKD workup", 0.05),
]

JBTS31_STATUS = [
    ("JBTS31 confirmed (MTS on MRI + cerebellar vermis hypoplasia)", 0.55),
    ("Pure renal NPHP20-like (no MTS; no cerebellar features)", 0.22),
    ("Probable JBTS31 (awaiting MRI; cerebellar signs present)", 0.13),
    ("Atypical / equivocal MRI (mild vermis; partial MTS)", 0.06),
    ("SRPS2B (skeletal dysplasia + Joubert — severe alleles)", 0.04),
]

RETINAL = [
    ("No retinal involvement (ERG normal, fundus clear)", 0.62),
    ("Rod-cone dystrophy (ERG abnormal, fundoscopy abnormal)", 0.28),
    ("Mild retinal changes (ERG borderline)", 0.08),
    ("LCA-like severe early retinal (ERG flat, neonatal nystagmus)", 0.02),
]

SKELETAL = [
    ("No skeletal features (normal chest X-ray, no polydactyly)", 0.88),
    ("Mild narrow thorax (incidental; no respiratory compromise)", 0.06),
    ("SRPS2B (marked thoracic dysplasia, postaxial polydactyly, severe)", 0.04),
    ("Borderline narrow thorax + brachydactyly (intermediate)", 0.02),
]

PRIOR_MISDIAGNOSIS = [
    ("No prior misdiagnosis (direct WES diagnosis)", 0.30),
    ("Joubert syndrome (gene-unknown) — CEP120 identified later on WES", 0.28),
    ("NPHP1 MLPA negative — incomplete workup", 0.17),
    ("CEP164/NPHP15 or CEP83/NPHP18 DA subtype workup first (negative)", 0.12),
    ("ADPKD (bilateral cysts; AR pattern missed)", 0.08),
    ("Jeune/SRPS (skeletal first; renal/Joubert workup delayed)", 0.05),
]

GFR_SLOPE = [
    ("Rapid (>5 ml/min/yr; ESRD before 18yr)", 0.20),
    ("Moderate (3–5 ml/min/yr; ESRD 18–25yr)", 0.35),
    ("Slow (1–3 ml/min/yr; ESRD 25–30yr)", 0.30),
    ("Very slow (<1 ml/min/yr; ESRD >30yr; hypomorphic alleles)", 0.15),
]

URINE_OSM = [
    ("Severe deficit: Uosm <150 mOsm/kg (maximal concentrating failure)", 0.16),
    ("Moderate deficit: Uosm 150–250 mOsm/kg", 0.30),
    ("Mild deficit: Uosm 250–500 mOsm/kg", 0.34),
    ("Near-normal: Uosm >500 mOsm/kg (early/mild CKD)", 0.20),
]

VARIANTS_POOL = [
    "p.Leu1019Pro (c.3056T>C) — HEAT repeat; European; moderate JBTS31 + renal; Roosing 2017",
    "p.Arg1045Ter (c.3133C>T) — truncating; null; severe JBTS31 + SRPS2B possible; pan-ethnic",
    "p.Gln720Ter (c.2158C>T) — coiled-coil; Middle Eastern consanguineous; JBTS31 + ESRD 15yr",
    "p.Ala813Pro (c.2437G>C) — HEAT repeat; South Asian consanguineous; moderate JBTS31",
    "p.Glu562Lys (c.1684G>A) — ARM repeat; hypomorphic; European; pure renal NPHP20; ESRD 22yr",
    "p.Trp389Ter (c.1167G>A) — ARM domain; North African; severe truncating; JBTS31 + retinal",
    "p.Pro631Leu (c.1892C>T) — coiled-coil; Middle Eastern; moderate NPHP20; ESRD 20yr",
    "p.Arg228Gln (c.683G>A) — N-terminal ARM; hypomorphic; South Asian; JBTS31 milder",
]


def _weighted_choice(options, n_rng):
    labels, weights = zip(*options)
    r = n_rng.random()
    cum = 0.0
    for label, w in zip(labels, weights):
        cum += w
        if r < cum:
            return label
    return labels[-1]


def _make_cohort():
    patients = []
    for i in range(COHORT_N):
        r = random.Random(SEED + i * 31)
        ethnicity    = _weighted_choice(ETHNICITIES, r)
        ckd_stage    = _weighted_choice(CKD_STAGES, r)
        kidney_uss   = _weighted_choice(KIDNEY_USS, r)
        first_sym    = _weighted_choice(FIRST_SYMPTOMS, r)
        jbts_stat    = _weighted_choice(JBTS31_STATUS, r)
        retinal_s    = _weighted_choice(RETINAL, r)
        skeletal_s   = _weighted_choice(SKELETAL, r)
        misdiag      = _weighted_choice(PRIOR_MISDIAGNOSIS, r)
        gfr_ml       = r.randint(5, 95)
        age_dx       = r.randint(1, 20)
        hb_val       = round(r.uniform(7.2, 14.3), 1)
        patients.append({
            "id":                 f"NPHP20-{i+1:03d}",
            "ethnicity":          ethnicity,
            "ckd_stage":          ckd_stage,
            "kidney_uss":         kidney_uss,
            "first_symptom":      first_sym,
            "jbts31_status":      jbts_stat,
            "retinal_status":     retinal_s,
            "skeletal_status":    skeletal_s,
            "prior_misdiagnosis": misdiag,
            "gfr_now_ml_min":     gfr_ml,
            "age_renal_dx_yr":    age_dx,
            "hb_g_dl":            hb_val,
        })
    return patients


_COHORT = _make_cohort()


# ── Helper ────────────────────────────────────────────────────────────────────
def _pct(patients, key, value):
    return round(100 * sum(1 for p in patients if value in p.get(key, "")) / len(patients))


def _dist(patients, key):
    counts: dict[str, int] = {}
    for p in patients:
        v = p.get(key, "Unknown")
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _dist_prefix(patients, key, prefix_len=60):
    raw = _dist(patients, key)
    return {k[:prefix_len]: v for k, v in raw.items()}


# ── API: overview ─────────────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    pts     = _COHORT[:8]
    all_pts = _COHORT

    esrd_tx       = _pct(all_pts, "ckd_stage", "ESRD")
    jbts31        = _pct(all_pts, "jbts31_status", "JBTS31 confirmed")
    retinal       = _pct(all_pts, "retinal_status", "Rod-cone")
    lca_like      = _pct(all_pts, "retinal_status", "LCA-like")
    srps2b        = _pct(all_pts, "skeletal_status", "SRPS2B")
    misdiag_jbts  = _pct(all_pts, "prior_misdiagnosis", "Joubert syndrome")
    misdiag_nphp1 = _pct(all_pts, "prior_misdiagnosis", "NPHP1 MLPA negative")
    misdiag_da    = _pct(all_pts, "prior_misdiagnosis", "CEP164")
    misdiag_adpkd = _pct(all_pts, "prior_misdiagnosis", "ADPKD")

    gfr_vals  = [p["gfr_now_ml_min"] for p in all_pts]
    hb_vals   = [p["hb_g_dl"] for p in all_pts]
    age_vals  = [p["age_renal_dx_yr"] for p in all_pts]
    sorted_gfr = sorted(gfr_vals)
    sorted_hb  = sorted(hb_vals)
    sorted_age = sorted(age_vals)
    n = len(sorted_gfr)

    return {
        "cohort_n":                   COHORT_N,
        "cohort_seed":                SEED,
        "median_gfr":                 sorted_gfr[n // 2],
        "median_hb":                  sorted_hb[n // 2],
        "median_age_renal_dx":        sorted_age[n // 2],
        "pct_esrd_or_transplant":     esrd_tx,
        "pct_jbts31_confirmed":       jbts31,
        "pct_retinal_dystrophy":      retinal + lca_like,
        "pct_lca_like":               lca_like,
        "pct_srps2b_skeletal":        srps2b,
        "pct_misdiagnosed_jbts_unknown": misdiag_jbts,
        "pct_misdiagnosed_nphp1":     misdiag_nphp1,
        "pct_misdiagnosed_da":        misdiag_da,
        "pct_misdiagnosed_adpkd":     misdiag_adpkd,
        "patients":                   pts,
    }


# ── API: breakdown ────────────────────────────────────────────────────────────
def get_breakdown() -> dict[str, Any]:
    pts = _COHORT
    return {
        "ckd_stage_distribution":      _dist_prefix(pts, "ckd_stage", 45),
        "jbts31_status":               _dist_prefix(pts, "jbts31_status", 55),
        "retinal_status":              _dist_prefix(pts, "retinal_status", 55),
        "skeletal_status":             _dist_prefix(pts, "skeletal_status", 60),
        "kidney_phenotype":            _dist_prefix(pts, "kidney_uss", 52),
        "prior_misdiagnosis":          _dist_prefix(pts, "prior_misdiagnosis", 52),
        "first_symptom_distribution":  _dist_prefix(pts, "first_symptom", 52),
        "ethnicity":                   _dist_prefix(pts, "ethnicity", 38),
        "urine_osmolality_tiers": {
            "Severe deficit: Uosm <150 mOsm/kg": round(0.16 * COHORT_N),
            "Moderate deficit: Uosm 150–250 mOsm/kg": round(0.30 * COHORT_N),
            "Mild deficit: Uosm 250–500 mOsm/kg": round(0.34 * COHORT_N),
            "Near-normal: Uosm >500 mOsm/kg": round(0.20 * COHORT_N),
        },
    }


# ── API: definitions ──────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    return {
        "disease": (
            "Nephronophthisis Type 20 / Joubert Syndrome 31 (NPHP20/JBTS31) — autosomal recessive "
            "ciliopathy caused by biallelic loss-of-function variants in CEP120 (CCDC100), encoding "
            "Centrosomal Protein 120 kDa, a scaffold required for daughter centriole elongation during "
            "mitosis and for ciliogenesis initiation at the mother centriole. CEP120 loss results in "
            "shortened daughter centrioles that cannot form competent basal bodies, blocking primary "
            "cilia assembly in renal tubular cells, cerebellar neurons, and photoreceptors. This leads "
            "to nephronophthisis-like tubulointerstitial nephritis (NPHP20), Joubert syndrome "
            "(JBTS31; Molar Tooth Sign in ~55%), and rod-cone retinal dystrophy (~30%). Severe biallelic "
            "null alleles cause Short-Rib Polydactyly Syndrome type 2B (SRPS2B). Ultra-rare: <25 "
            "published families worldwide (2026). THE ONLY NPHP SUBTYPE CAUSED BY DEFECTIVE DAUGHTER "
            "CENTRIOLE ELONGATION SCAFFOLD — mechanistically upstream of distal appendage subtypes "
            "(NPHP15/CEP164, NPHP18/CEP83) and of IFT transport subtypes (NPHP12, NPHP13, NPHP19)."
        ),
        "omim_gene":    "*613446 (CEP120 / CCDC100 — Centrosomal Protein 120 kDa)",
        "omim_disease": "#617761 (Joubert Syndrome 31, JBTS31; NPHP20 — nephronophthisis-related); severe alleles → SRPS2B",
        "chromosome":   "5q23.2",
        "inheritance":  "Autosomal Recessive — biallelic LOF (truncating + missense); homozygous in consanguineous; compound heterozygous in outbred",
        "prevalence":   "~1/800,000–2,000,000; <25 published families (2026); ultra-rare among NPHP/JBTS subtypes",
        "mechanism": (
            "CEP120 (Centrosomal Protein 120 kDa) is a centriole-associated scaffold with two distinct "
            "roles in ciliogenesis: (1) Daughter centriole elongation: CEP120 localises to the distal "
            "tip of daughter centrioles via CPAP (CENPJ) and SAS-4 interactions. CEP120 organises the "
            "distal centriole scaffold with CEP135 and CP110/CEP97. Without CEP120, daughter centrioles "
            "are ~80% of normal length and cannot serve as competent mother centrioles in subsequent "
            "cell cycles. This defect accumulates over cell generations → progressive basal body "
            "incompetence → primary cilia assembly failure. (2) Ciliogenesis initiation: during "
            "quiescence/G0, the mother centriole docks to the plasma membrane to form the basal body. "
            "CEP120 assists destabilisation of the CP110/CEP97 distal cap via cooperation with the "
            "TTBK2–CEP164 phosphorylation cascade, allowing axoneme extension. Short/dysfunctional "
            "basal bodies cannot anchor distal appendages (CEP83, CEP89, SCLT1, FBF1, LRRC45, CEP164) "
            "efficiently → ciliogenesis block. Renal tubular cilia loss → TIN + corticomedullary "
            "cysts + concentrating defect → ESRD. Cerebellar neuronal cilia loss → Shh/Gli pathway "
            "failure → cerebellar vermis hypoplasia + SCP elongation = Molar Tooth Sign (JBTS31). "
            "Photoreceptor connecting cilium failure → rod-cone degeneration in ~25–30% of CEP120 cases. "
            "Skeletal cilia loss (limb chondrocytes, rib growth plates) with severe null alleles → "
            "SRPS2B (short ribs, narrow thorax, postaxial polydactyly, limb shortening). Mechanistically "
            "distinct: CEP120 defect is UPSTREAM of distal appendage assembly (NPHP15/CEP164, NPHP18/CEP83) "
            "and of IFT transport (NPHP12/TTC21B, NPHP13/WDR19, NPHP19/IFT81) — the basal body "
            "itself is malformed before any distal appendage or IFT machinery engages."
        ),
        "genetic_architecture": {
            "Gene symbol": "CEP120 (also CCDC100)",
            "Protein": "Centrosomal Protein 120 kDa; ~1085 amino acids; coiled-coil + ARM/HEAT repeats",
            "Chromosomal locus": "5q23.2 — unique arm; no confusion with other major NPHP gene loci",
            "OMIM gene entry": "*613446",
            "Functional domain 1 — N-terminal ARM/HEAT repeats (aa ~1–500)": (
                "Arc-shaped protein-protein interaction scaffold; binds CPAP (CENPJ) during daughter "
                "centriole elongation; nucleates the distal centriole scaffold complex with CEP135, "
                "CP110, and CEP97. ARM repeat LOF variants (e.g., p.Glu562Lys, p.Arg228Gln) cause "
                "hypomorphic alleles with milder/pure renal phenotype"
            ),
            "Functional domain 2 — Central coiled-coil (aa ~500–850)": (
                "Mediates CEP120 self-oligomerization and targeting to the distal daughter centriole tip; "
                "interacts with centrin (CETN2/CETN3); required for stability of the distal centriole cap "
                "complex. Coiled-coil LOF variants (e.g., p.Gln720Ter, p.Pro631Leu) cause moderate "
                "JBTS31 + nephronophthisis phenotype"
            ),
            "Functional domain 3 — C-terminal regulatory domain (aa ~850–1085)": (
                "Interaction surface for CEP135/CNTRL and SAS-4/CPAP; coordinates transition from "
                "centriole elongation to ciliogenesis initiation; C-terminal truncating alleles "
                "(e.g., p.Leu1019Pro, p.Arg1045Ter) tend to cause the most severe phenotypes including "
                "JBTS31 with or without SRPS2B features"
            ),
            "Key binding partners": (
                "CPAP/CENPJ (JBTS9 gene; direct ARM-domain binding; co-sequencing mandatory); "
                "CEP135/CNTRL (distal centriole scaffold co-organiser); CP110 (distal cap; CEP120 "
                "cooperates with TTBK2 to destabilise CP110 for ciliogenesis); CEP97 (CP110 "
                "stabiliser; indirect); SAS-4 (centriole scaffold); centrin (CETN2/CETN3)"
            ),
            "Disease spectrum": (
                "Allele-severity continuum: homozygous null → SRPS2B (severe; neonatal lethality risk) "
                "or severe JBTS31 + retinal + renal; compound het (null + missense) → JBTS31 + NPHP20 "
                "(moderate); hypomorphic homozygous → pure renal NPHP20 (mild; adult-onset ESRD)"
            ),
            "key_variants": VARIANTS_POOL,
        },
        "key_clinical_features": {
            "ONLY NPHP subtype caused by DAUGHTER CENTRIOLE ELONGATION scaffold defect — upstream of all other NPHP mechanisms": (
                "CEP120 loss causes short daughter centrioles that cannot form competent basal bodies. "
                "This is UPSTREAM of: (a) distal appendage defects (NPHP15/CEP164, NPHP18/CEP83) which "
                "assume a normal-length basal body is present; (b) IFT transport defects (NPHP19/IFT81, "
                "NPHP12/TTC21B) which assume cilia have already formed; (c) TZ scaffold defects "
                "(NPHP1, NPHP4, NPHP8) which assume the basal body is docked. CEP120 defect prevents "
                "even the first step — competent basal body formation — making it mechanistically "
                "the most upstream NPHP subtype described to date (2026)"
            ),
            "Joubert Syndrome 31 (JBTS31) — Molar Tooth Sign in ~55% — Brain MRI Mandatory at Diagnosis": (
                "MTS on axial T1/T2 brain MRI (cerebellar vermis hypoplasia + SCP elongation) in ~55% "
                "of biallelic CEP120 cases — similar penetrance to NPHP18/CEP83 (~55%) and lower than "
                "NPHP19/IFT81 (~65%). Oculomotor apraxia (OMA), neonatal hypotonia, episodic breathing "
                "irregularity (self-resolves ~2–3yr), cerebellar ataxia, developmental delay. Brain MRI "
                "MANDATORY at diagnosis. Allele class determines Joubert penetrance: null×null → JBTS31 "
                "certain; hypomorphic×hypomorphic → pure renal NPHP20 (no MTS)"
            ),
            "Nephronophthisis / Renal (TIN + corticomedullary cysts) — NPHP20 — Adolescent ESRD": (
                "TIN + corticomedullary cysts + tubular concentrating defect; ESRD variable (adolescent "
                "to young adult onset; median ~14–18yr). Bilateral small echogenic kidneys ± discrete "
                "cysts on USS. Urine osmolality deficit early marker. Anaemia disproportionate to GFR. "
                "Pre-emptive transplant preferred when compatible donor available. Renal transplant "
                "CURATIVE: donor kidney has functional CEP120 → daughter centriole elongation restored "
                "→ normal basal body → intact tubular cilia → no TIN recurrence"
            ),
            "SRPS2B — Short-Rib Polydactyly Syndrome type 2B — Biallelic Null CEP120 (~4–10%)": (
                "Severe biallelic null CEP120 alleles → SRPS2B: narrow bell-shaped thorax, extremely "
                "short horizontal ribs, postaxial polydactyly, limb shortening (rhizomelia/mesomelia), "
                "± Joubert features. Potentially lethal in neonates (respiratory failure). Chest X-ray + "
                "skeletal survey MANDATORY in all newly diagnosed CEP120 patients. SRPS2B overlap "
                "distinguishes CEP120 from pure TZ subtypes (NPHP1, NPHP4, NPHP8, NPHP17) which "
                "have no skeletal involvement"
            ),
            "Retinal Dystrophy ~30% — Moderate Penetrance — Annual ERG Mandatory": (
                "Rod-cone dystrophy in ~28–32% (ERG abnormal, progressive); LCA-like severe early in "
                "~2% (null×null). Moderate penetrance — lower than NPHP19/IFT81 (~50–60%) and "
                "NPHP15/CEP164 (~35–40%) but higher than NPHP17/MAPKBP1 (0%) and NPHP16/ANKS6 (0%). "
                "Retinal does NOT improve post-transplant (cell-autonomous photoreceptor connecting "
                "cilium defect). Annual ERG + fundoscopy mandatory from diagnosis in all CEP120 patients"
            ),
            "CPAP (CENPJ) Co-Sequencing Mandatory — Direct Binding Partner — JBTS9/MCPH6": (
                "CENPJ/CPAP is a direct binding partner of CEP120's ARM repeat domain and is required "
                "for recruiting CEP120 to daughter centrioles. CENPJ mutations cause JBTS9 (Joubert "
                "syndrome 9) and MCPH6 (primary microcephaly 6). CENPJ MUST be co-sequenced alongside "
                "CEP120 in all cases. Digenic CEP120 + CENPJ heterozygous variants have been documented. "
                "If single heterozygous CEP120 variant identified: always check CENPJ for second hit"
            ),
            "5q23.2 — No Chromosome Arm Confusion — WES Mandatory": (
                "CEP120 is on chromosome 5q23.2 — a unique locus with no confusion with other major "
                "NPHP gene arms (NPHP1 2q13; NPHP2 9q31.1; NPHP6 12q21.32; NPHP17 2q13.3; NPHP18 "
                "12q22; NPHP19 12q23.1). However, NPHP1 MLPA (290kb) does NOT detect CEP120 at 5q23.2. "
                "Targeted single-gene NPHP panels often exclude CEP120 (ultra-rare). WES is the only "
                "reliable diagnostic approach. Gene-unknown Joubert with skeletal features: ALWAYS "
                "check CEP120 and CENPJ on WES"
            ),
        },
        "nphp_comparison": {
            "★ NPHP20/JBTS31 (CEP120 — 5q23.2)": (
                "ONLY NPHP subtype: daughter centriole elongation scaffold. CEP120 loss → short centrioles → "
                "incompetent basal bodies → cilia absent. JBTS31 ~55%; retinal ~30%; SRPS2B 4–10% (null×null). "
                "No situs inversus; no CHF. ESRD adolescent ~14–18yr. CPAP/CENPJ co-sequence mandatory"
            ),
            "NPHP18/JBTS22 (CEP83 — 12q22)": (
                "Distal appendage (DA) FOUNDATION — CEP83 is the most proximal DA protein; nucleates CEP89, "
                "SCLT1, FBF1, LRRC45, CEP164 cascade. Centriole length NORMAL but DA assembly absent → "
                "ciliogenesis docking fails. JBTS22 ~55%; retinal ~30–40%; CHF ~5–10%. Distinct: "
                "CEP83 defect is DOWNSTREAM of CEP120 (assumes normal-length basal body exists)"
            ),
            "NPHP15 (CEP164 — 11q23.3)": (
                "Distal appendage CENTRAL HUB — CEP164 is the TTBK2 substrate; directly triggers CP110 "
                "removal for ciliogenesis. SLS (retinal ~35–40%) + renal; no Joubert. "
                "TTBK2 phosphorylates CEP164 Ser172 — the ciliogenesis master switch. "
                "Mechanistically downstream of CEP120: requires normal basal body + CEP83 DA foundation"
            ),
            "NPHP19/JBTS35 (IFT81 — 12q23.1)": (
                "IFT-B ANTEROGRADE CORE — IFT81/IFT74 heterodimer; tubulin-binding module. Cilia initially "
                "form but IFT trains fail → no axoneme assembly. JBTS35 ~65%; retinal ~50–60%; "
                "IFT74 digenic. Mechanistically downstream of CEP120: cilia assembly begins (basal body "
                "competent) but anterograde transport fails. Ultra-rare <20 families"
            ),
            "NPHP17 (MAPKBP1/JIP4 — 2q13.3)": (
                "JNK scaffold at TZ via NPHP4 supercomplex. Pure renal: TIN, ESRD ~14–16yr. "
                "No Joubert; no retinal; no CHF; no situs; no skeletal. Ultra-rare 25–35 families. "
                "Mechanistically distinct: TZ signalling scaffold, not centriole or IFT apparatus"
            ),
            "NPHP13/CED1 (WDR19 — 4p14)": (
                "IFT-A RETROGRADE + ectodermal features (hypotrichosis, hyponychia, hypodontia). "
                "Cranioectodermal Dysplasia 1 (CED1). Retinal 20–30%. Skeletal (Jeune ~8%): overlaps "
                "SRPS2B features but distinct mechanism (IFT-A retrograde vs CEP120 centriole elongation)"
            ),
        },
        "ddx_table": {
            "NPHP18/JBTS22 (CEP83) vs NPHP20/JBTS31 (CEP120)": (
                "Both cause JBTS + nephronophthisis at similar rates (~55% each) and similar ESRD onset "
                "(~14–18yr). KEY DISTINCTION: CEP120 defect is UPSTREAM — daughter centriole itself is "
                "malformed (centriole elongation scaffold). CEP83 defect is DOWNSTREAM — centriole is "
                "normal-length but distal appendage foundation fails. CEP83: CHF 5–10%; CEP120: no CHF. "
                "CEP120: SRPS2B in ~4–10% (null×null) — no skeletal involvement in CEP83. "
                "CEP120 at 5q23.2 vs CEP83 at 12q22: no chromosome arm confusion"
            ),
            "NPHP15 (CEP164) vs NPHP20 (CEP120)": (
                "CEP164 (11q23.3): distal appendage central hub (TTBK2 substrate, CP110 removal master "
                "switch). SLS: retinal 35–40%; no Joubert; no skeletal. CEP120 (5q23.2): centriole "
                "elongation upstream; JBTS31 ~55%; retinal ~30%; SRPS2B 4–10%. SLS/retinal WITHOUT "
                "Joubert → CEP164 first. JBTS with retinal + skeletal → CEP120 first. Both WES-only"
            ),
            "NPHP13/CED1 (WDR19) vs NPHP20 (CEP120) — skeletal overlap": (
                "Both can cause skeletal involvement (CEP120: SRPS2B; WDR19: Jeune/ATD5). KEY: WDR19 "
                "causes ECTODERMAL features (hypotrichosis, sparse hair; hyponychia, dystrophic nails; "
                "hypodontia — pathognomonic for CED1). CEP120/NPHP20 has NO ectodermal features. "
                "CED1: IFT-A retrograde; WD40 beta-propeller. CEP120: centriole elongation ARM repeats. "
                "Skeletal features + ectodermal → WDR19/CED1. Skeletal + Joubert, no ectodermal → CEP120"
            ),
            "NPHP19/JBTS35 (IFT81) vs NPHP20/JBTS31 (CEP120) — Joubert overlap": (
                "Both cause JBTS + renal. Key distinctions: IFT81 (12q23.1): JBTS35 ~65% (higher); "
                "retinal ~50–60% (much higher); IFT74 digenic; no skeletal. CEP120 (5q23.2): JBTS31 "
                "~55%; retinal ~30% (lower); SRPS2B 4–10% (IFT81 has NO skeletal features). JBTS with "
                "narrow thorax/polydactyly → CEP120 first. JBTS with very high retinal penetrance → "
                "IFT81 first. No chromosome arm confusion (12q vs 5q)"
            ),
            "Jeune syndrome / ATD (TTC21B/NPHP12, WDR19/NPHP13) vs NPHP20 SRPS2B (CEP120)": (
                "CEP120 SRPS2B is more severe than Jeune (narrower thorax, shorter ribs, higher lethality "
                "risk). Jeune/ATD4 (TTC21B) and ATD5 (WDR19) are caused by IFT-A retrograde defects; "
                "CEP120 SRPS2B by centriole elongation scaffold. Jeune alleles: less severe hypomorphic "
                "with pure renal NPHP12/NPHP13 phenotype also possible. CEP120 SRPS2B: always has Joubert "
                "features (MTS) if patient survives; Jeune/ATD usually has no Joubert"
            ),
            "ADPKD (PKD1/PKD2) vs NPHP20 (CEP120)": (
                "CEP120: bilateral cysts mimic ADPKD on USS. ADPKD: dominant family history; large kidneys; "
                "PKD1/PKD2 positive. CEP120: autosomal recessive; small echogenic kidneys; corticomedullary "
                "cysts not cortical only; Joubert MTS ± SRPS2B ± retinal → NPHP20. ~8% of NPHP20 cohort "
                "initially investigated as ADPKD. Consanguineous background + cysts + Joubert → NPHP20 over ADPKD"
            ),
            "CPAP/CENPJ (JBTS9/MCPH6) vs CEP120 (NPHP20/JBTS31) — direct interaction pair": (
                "CENPJ and CEP120 are direct binding partners during daughter centriole elongation. "
                "Clinical distinction: CENPJ/MCPH6 causes primary microcephaly (MCPH6) — markedly small "
                "head circumference with intellectual disability — in addition to JBTS9. CEP120 does NOT "
                "cause primary microcephaly. JBTS with microcephaly → CENPJ first. JBTS without microcephaly "
                "+ skeletal ± SRPS2B → CEP120. Always co-sequence CEP120 + CENPJ; may be digenic pair"
            ),
        },
        "treatment": {
            "Renal Transplant (CURATIVE — No Recurrence)": (
                "Donor kidney with functional CEP120 → daughter centrioles elongate normally → competent "
                "basal bodies → intact tubular primary cilia → no TIN recurrence. Excellent long-term "
                "renal outcomes. Pre-emptive transplant preferred when compatible donor available. "
                "Dialysis as bridge to transplant. GFR slope monitoring from diagnosis; nephrology "
                "involvement from CKD stage 3. Multi-disciplinary team essential for JBTS31 cases"
            ),
            "No Disease-Modifying Therapy (2026 — Centriole Elongation Rescue Pre-Clinical)": (
                "No approved CEP120-specific therapy as of 2026. Pre-clinical: CEP120 overexpression in "
                "zebrafish cep120 morphants partially restores cilia length and renal function. AAV-mediated "
                "CEP120 delivery to renal tubular cells: conceptual; pre-clinical stage. CPAP/CENPJ "
                "pathway manipulation (e.g., CPAP-ARM domain stabilisation) may rescue CEP120 binding "
                "interface — in silico only. Centriole length measurement by electron microscopy remains "
                "a research-stage biomarker; no clinical assay available"
            ),
            "Ophthalmology — Annual ERG + Fundoscopy from Diagnosis (~30% penetrance)": (
                "~28–32% retinal penetrance mandates ophthalmology surveillance from diagnosis. Progressive "
                "rod-cone dystrophy: low-vision aids early; cane/mobility training; braille/assistive "
                "technology planning. No anti-VEGF, no specific retinal therapy 2026 for CEP120-related "
                "retinal dystrophy. Annual ERG + fundoscopy mandatory from day of diagnosis regardless "
                "of initial ERG — onset of retinal disease may lag renal onset by years"
            ),
            "Skeletal / SRPS2B — Chest X-ray + Skeletal Survey + Respiratory Management": (
                "Chest X-ray and full skeletal survey mandatory in ALL newly diagnosed CEP120 patients "
                "regardless of apparent phenotype — SRPS2B may be subclinical initially. SRPS2B with "
                "respiratory compromise: neonatal/infant respiratory management (CPAP, ventilator support "
                "in severe cases). Orthopaedics: monitoring of thoracic cage development; VEPTR/MAGEC "
                "thoracic expansion if progressive. Limb length monitoring. Annual chest X-ray for "
                "incidental narrow thorax cases to detect progression"
            ),
            "Neurodevelopmental — JBTS31 Multi-Disciplinary Team": (
                "Developmental paediatrics: early intervention (physiotherapy, OT, speech therapy) from "
                "diagnosis for JBTS31 cases. Cerebellar ataxia: gait training, balance aids. Oculomotor "
                "apraxia (OMA): visual learning adaptations. Neonatal breathing irregularity: monitoring; "
                "usually self-resolves by 2–3yr. Developmental delay: special education planning. "
                "Cognitive trajectory guided by MTS severity and allele class (null×null worst; "
                "hypomorphic×hypomorphic may have normal cognition)"
            ),
        },
        "diagnostic_criteria": {
            "Molecular Diagnosis (Gold Standard)": (
                "Biallelic pathogenic variants in CEP120 identified by WES or targeted gene panel "
                "including CEP120. Variants classified per ACMG criteria (pathogenic/likely pathogenic). "
                "Always co-report CEP120 + CENPJ variants together. Segregation in family members. "
                "Single heterozygous CEP120 variant: check CENPJ for second hit (digenic pair)"
            ),
            "Renal Criteria — NPHP20": (
                "Tubulointerstitial nephritis (TIN) on biopsy or inferred from imaging + clinical: "
                "bilateral small echogenic kidneys ± corticomedullary cysts (USS); tubular concentrating "
                "defect (Uosm <500 mOsm/kg); CKD with progressive GFR decline; normal/negative PKD1/PKD2; "
                "NPHP1 MLPA negative. Family history AR pattern or consanguinity"
            ),
            "Joubert Criteria — JBTS31": (
                "Molar Tooth Sign (MTS) on axial brain T1/T2 MRI: cerebellar vermis hypoplasia + "
                "superior cerebellar peduncle (SCP) elongation + cerebellar fissure deepening. "
                "Oculomotor apraxia (OMA). Neonatal hypotonia. Developmental delay or intellectual "
                "disability. Episodic breathing irregularity (neonatal period). CEP120 biallelic LOF "
                "confirmed on WES"
            ),
            "Skeletal Criteria — SRPS2B": (
                "Chest X-ray: extremely short horizontal ribs + markedly narrow thorax + bell-shaped "
                "thoracic cage. Skeletal survey: rhizomelic/mesomelic limb shortening, polydactyly "
                "(postaxial), abnormal epiphyses/metaphyses. Neonatal respiratory distress. "
                "CEP120 biallelic null alleles (both alleles truncating or large deletion)"
            ),
        },
        "prognosis": (
            "NPHP20/JBTS31 prognosis is governed by allele class: (1) Renal — ESRD onset variable "
            "(adolescent ~14–18yr typical; adult onset with hypomorphic alleles; earlier with null×null); "
            "pre-emptive transplant preferred; post-transplant renal outcomes excellent (cell-autonomous "
            "CEP120 defect, no recurrence); (2) Retinal — ~30% rod-cone dystrophy; cell-autonomous; not "
            "improved by transplant; annual ERG mandatory; visual prognosis guided by ERG trajectory; "
            "(3) Cerebellar/Joubert — MTS in ~55%; developmental delay and cerebellar ataxia; static "
            "defect, not improved by transplant; multi-disciplinary team mandatory; (4) Skeletal — SRPS2B "
            "in ~4–10% (null×null); neonatal lethality risk in most severe; subclinical thoracic narrowing "
            "requires surveillance in all. Best predictor: allele class. Null×null → SRPS2B or severe "
            "JBTS31 + retinal + ESRD <18yr. Hypomorphic×hypomorphic → pure renal NPHP20; adult ESRD; "
            "no JBTS31; no SRPS2B. Ultra-rare status (<25 families 2026) limits precise genotype-phenotype "
            "delineation; caution in individual prognosis without specific literature review."
        ),
        "cohort_note": (
            f"40-patient synthetic cohort (seed={SEED}) generated from CEP120/JBTS31/NPHP20 "
            "literature-derived phenotype distributions. Ultra-rare disease (<25 real published "
            "families 2026): cohort distributions extrapolated from published case series, systematic "
            "ciliopathy reviews, and comparative data from related NPHP subtypes with overlapping "
            "molecular mechanisms (CEP164/NPHP15, CEP83/NPHP18, WDR19/NPHP13). Distributions reflect "
            "best-available evidence; individual patient characteristics are model-derived, not from "
            "real patient records. For clinical decisions, verify variant pathogenicity against current "
            "ClinVar, HGMD, and published CEP120 literature."
        ),
    }
