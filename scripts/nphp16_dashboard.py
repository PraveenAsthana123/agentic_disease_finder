"""
Nephronophthisis Type 16
========================
Primary Gene : ANKS6 (*615803) — 9q22.33; 982 aa; Ankyrin Repeat and SAM Domain-
               Containing Protein 6 (also PKDR1 / NPHP16); localises to the inversin
               compartment (IC) at the proximal transition zone of cilia; maintained
               by NEK8 (NPHP9) phosphorylation; part of the INVS/NPHP2-NPHP3-NEK8-ANKS6
               inversin compartment module
Disease OMIM : #615862 (Nephronophthisis 16 — NPHP16; renal ciliopathy ± situs inversus)
Chromosome   : 9q22.33
Inheritance  : Autosomal Recessive (biallelic LOF — truncating and/or missense)
Prevalence   : ~1/500,000–1,000,000; ~60–80 published families as of 2026

Protein Structure — ANKS6 (982 aa; inversin compartment scaffold)
-----------------------------------------------------------------
  • N-terminal ankyrin repeat domain (aa 1–540): 12 ankyrin repeats; mediates
    protein–protein interactions; INVS (inversin/NPHP2) direct binding partner;
    NPHP3 interaction; structural scaffold of IC module
  • Linker / regulatory region (aa 541–720): NEK8 phosphorylation targets;
    regulatory interface for IC integrity; kinase docking motifs
  • C-terminal SAM domain (aa 721–982): sterile alpha motif; homo-
    oligomerisation; BICC1 interaction; mTOR-pathway modulatory interface

Molecular Mechanism
-------------------
ANKS6 is an obligate scaffold of the inversin compartment (IC) — a distinct ciliary
subdomain proximal to the transition zone gate, separate from the TZ-scaffold NPHP
module (NPHP1/4/8) and the distal appendage (CEP164/NPHP15):
  1. INVS (NPHP2) + ANKS6 + NPHP3 + NEK8 co-localise to the IC as a functional unit
  2. NEK8 (a NIMA-family kinase) phosphorylates ANKS6 at Ser residues in the linker
     region → maintains IC scaffold integrity; loss of NEK8 phenocopies ANKS6 loss
  3. The IC acts as a Wnt-pathway switch: IC intact → suppresses canonical β-catenin
     Wnt in tubular epithelium → maintains tubular identity; IC lost → canonical Wnt
     up-regulated → tubular EMT → TIN → ESRD
  4. IC in nodal cilia controls left-right signalling: ANKS6 loss in embryonic node
     → impaired Wnt/PCP flow → laterality defects (situs inversus) in ~20–30%
     (incomplete penetrance versus INVS/NPHP2 where situs inversus >85%)
  5. BICC1 (SAM-domain binding partner) bridges ANKS6 to mTOR-pathway attenuation;
     loss → mild mTOR-driven cystogenesis in collecting duct + proximal tubule

Clinical Overview
-----------------
  • Renal: tubulointerstitial nephritis (TIN) + corticomedullary cysts + concentrating
    defect (polyuria, polydipsia); ESRD median ~13yr (juvenile NPHP pattern)
  • Situs inversus: ~20–30% — the most diagnostically discriminating extra-renal
    feature; rarer than NPHP2 (>85%) but far commoner than NPHP4/8/14/15 (~0%)
  • No retinal dystrophy — ANKS6 not expressed in photoreceptors (ERG normal)
  • No CHF — ANKS6 absent from biliary epithelium
  • No Joubert — not expressed in cerebellar neurons; no Molar Tooth Sign
  • No intellectual disability
  • No ectodermal, pancreatic, or skeletal features

Key Diagnostic Alerts
---------------------
  • Situs inversus + juvenile CKD → NPHP16 or NPHP2; differentiate by onset age
    (NPHP2 infantile: ESRD <3yr; NPHP16 juvenile: ESRD ~13yr)
  • NPHP1 290kb deletion MLPA (standard first-line) does NOT detect ANKS6 on 9q22.33
  • NEK8 (NPHP9) and INVS (NPHP2) and NPHP3 must be co-sequenced — IC functional unit
  • ANKS6 → Autosomal Recessive; must distinguish from INVS situs inversus solitus
    (AR) vs Kartagener syndrome / primary ciliary dyskinesia (motile cilia defect)
  • WES mandatory; panel testing must include ANKS6 on all NPHP and situs-NPHP panels

40-patient cohort generated with seed=371; 3 endpoints
  /api/nphp16/overview | /api/nphp16/breakdown | /api/nphp16/definitions
"""

import random
from typing import Any

# ── Cohort seed ──────────────────────────────────────────────────────────────
SEED        = 371
COHORT_N    = 40
rng         = random.Random(SEED)

# ── Patient phenotype distributions (ANKS6/NPHP16 literature) ────────────────
_ETHNICITIES = [
    ("European (Northern)",        0.35),
    ("European (Southern/Balkan)", 0.18),
    ("Middle Eastern",             0.20),
    ("South Asian",                0.12),
    ("East Asian",                 0.07),
    ("Latin American",             0.05),
    ("Sub-Saharan African",        0.03),
]

_FIRST_SYMPTOMS = [
    ("Polyuria/polydipsia (tubular concentrating defect)",              0.38),
    ("Incidental haematuria/proteinuria on screening",                  0.18),
    ("Anaemia + fatigue (ESRD-range)",                                  0.16),
    ("Situs inversus on prenatal/neonatal USS",                         0.12),
    ("Family screening (affected sibling)",                             0.10),
    ("Hypertension (late CKD)",                                         0.06),
]

_MISDIAGNOSES = [
    ("NPHP1 deletion negative (MLPA — misses ANKS6 9q22.33)",          0.36),
    ("NPHP2/INVS (situs inversus — wrong onset age)",                   0.20),
    ("ADPKD/PKD1 (AR pattern missed; bilateral cysts on USS)",          0.16),
    ("Kartagener / PCD (situs + renal — bronchiectasis absent)",        0.12),
    ("Alport syndrome (CKD + haematuria — COL4A gene negative)",        0.10),
    ("No prior misdiagnosis (direct WES diagnosis)",                    0.06),
]

_CKD_STAGES = [
    ("CKD 1–2 (GFR ≥60; tubular symptoms only)",                       0.12),
    ("CKD 3a (GFR 45–59)",                                              0.15),
    ("CKD 3b (GFR 30–44)",                                              0.18),
    ("CKD 4 (GFR 15–29)",                                               0.20),
    ("CKD 5 / ESRD (GFR <15, pre-dialysis)",                           0.18),
    ("Post-transplant (functioning graft, no recurrence)",              0.17),
]

_KIDNEY_USS = [
    ("Small echogenic kidneys, corticomedullary cysts",                 0.52),
    ("Small echogenic kidneys, no visible cysts on USS",                0.22),
    ("Normal size, early hyperechogenicity only",                       0.13),
    ("Bilateral moderate cysts (PKD-like USS pattern)",                 0.08),
    ("Normal USS (early disease, tubular deficit only)",                0.05),
]

_URINE_OSM = [
    ("< 200 mOsm/kg (severe concentrating defect)",                    0.32),
    ("200–400 mOsm/kg (moderate)",                                      0.30),
    ("400–600 mOsm/kg (mild-moderate)",                                 0.22),
    ("> 600 mOsm/kg (early/mild; tubular defect emerging)",             0.16),
]

_GFR_SLOPE = [
    ("–5 to –3 ml/min/yr (slow progression)",                          0.22),
    ("–8 to –5 ml/min/yr (moderate)",                                   0.30),
    ("–12 to –8 ml/min/yr (fast; typical juvenile NPHP)",              0.28),
    ("> –12 ml/min/yr (very fast; early ESRD <10yr)",                  0.20),
]

_SITUS_STATUS = [
    ("No laterality defect (situs solitus — 70–80%)",                  0.73),
    ("Situs inversus totalis (complete reversal — ~20–30%)",            0.22),
    ("Situs ambiguus (partial reversal; heterotaxy — rare)",            0.05),
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
        situs_status    = _weighted_choice(_SITUS_STATUS, r)

        situs_inversus  = "situs inversus totalis" in situs_status.lower()
        situs_ambiguus  = "ambiguus" in situs_status.lower()

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

        age_dx = r.randint(4, 16)
        hb     = round(r.uniform(7.2, 13.8), 1)

        patients.append({
            "id":                   f"NPHP16-{i:03d}",
            "ethnicity":            ethnicity,
            "first_symptom":        first_symptom,
            "prior_misdiagnosis":   misdiagnosis,
            "ckd_stage":            ckd_stage,
            "kidney_uss":           kidney_uss,
            "situs_status":         situs_status,
            "situs_inversus":       situs_inversus,
            "situs_ambiguus":       situs_ambiguus,
            "gfr_now_ml_min":       gfr,
            "age_renal_dx_yr":      age_dx,
            "hb_g_dl":              hb,
        })
    return patients


# ── Public API ────────────────────────────────────────────────────────────────

def get_overview() -> dict[str, Any]:
    patients = _generate_cohort()

    n_situs_inv  = sum(1 for p in patients if p["situs_inversus"])
    n_situs_amb  = sum(1 for p in patients if p["situs_ambiguus"])
    n_esrd_tx    = sum(1 for p in patients if "ESRD" in p["ckd_stage"] or "Post-transplant" in p["ckd_stage"])
    n_misdiag_nphp1 = sum(1 for p in patients if "NPHP1" in p["prior_misdiagnosis"])
    n_misdiag_nphp2 = sum(1 for p in patients if "NPHP2" in p["prior_misdiagnosis"])
    n_retinal    = 0   # ANKS6 not expressed in photoreceptors — no retinal
    n_chf        = 0   # no biliary involvement

    median_gfr  = sorted(p["gfr_now_ml_min"] for p in patients)[COHORT_N // 2]
    median_hb   = sorted(p["hb_g_dl"] for p in patients)[COHORT_N // 2]
    median_age  = sorted(p["age_renal_dx_yr"] for p in patients)[COHORT_N // 2]

    return {
        "cohort_n":                 COHORT_N,
        "seed":                     SEED,
        "median_gfr":               median_gfr,
        "median_hb":                round(median_hb, 1),
        "median_age_renal_dx":      median_age,
        "pct_situs_inversus":       round(n_situs_inv / COHORT_N * 100),
        "pct_situs_ambiguus":       round(n_situs_amb / COHORT_N * 100),
        "pct_any_laterality":       round((n_situs_inv + n_situs_amb) / COHORT_N * 100),
        "pct_esrd_or_transplant":   round(n_esrd_tx / COHORT_N * 100),
        "pct_retinal_involvement":  0,
        "pct_chf_involvement":      0,
        "pct_misdiagnosed_nphp1":   round(n_misdiag_nphp1 / COHORT_N * 100),
        "pct_misdiagnosed_nphp2":   round(n_misdiag_nphp2 / COHORT_N * 100),
        "patients":                 patients[:8],
    }


def get_breakdown() -> dict[str, Any]:
    patients = _generate_cohort()

    def count_dist(key):
        d: dict[str, int] = {}
        for p in patients:
            v = p[key]
            d[v] = d.get(v, 0) + 1
        return dict(sorted(d.items(), key=lambda x: -x[1]))

    # Situs distribution
    situs_dist: dict[str, int] = {}
    for p in patients:
        v = p["situs_status"].split("(")[0].strip()
        situs_dist[v] = situs_dist.get(v, 0) + 1

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
        "situs_distribution":           dict(sorted(situs_dist.items(), key=lambda x: -x[1])),
        "kidney_phenotype":             dict(sorted(uss_dist.items(), key=lambda x: -x[1])),
        "prior_misdiagnosis":           dict(sorted(misdiag_dist.items(), key=lambda x: -x[1])),
        "first_symptom_distribution":   dict(sorted(symp_dist.items(), key=lambda x: -x[1])),
        "ethnicity":                    dict(sorted(ethn_dist.items(), key=lambda x: -x[1])),
        "urine_osmolality_tiers":       dict(sorted(osm_dist.items(), key=lambda x: -x[1])),
        "gfr_slope_tiers":              dict(sorted(slope_dist.items(), key=lambda x: -x[1])),
    }


def get_definitions() -> dict[str, Any]:
    return {
        "disease":      "Nephronophthisis Type 16 (NPHP16) — autosomal recessive "
                        "renal ciliopathy caused by biallelic LOF variants in ANKS6 "
                        "(Ankyrin Repeat and SAM Domain-Containing Protein 6); "
                        "inversin compartment (IC) scaffold ciliopathy; juvenile-onset "
                        "NPHP (ESRD median ~13yr) ± situs inversus in 20–30%.",
        "omim_gene":    "*615803 (ANKS6)",
        "omim_disease": "#615862 (NPHP16)",
        "chromosome":   "9q22.33",
        "inheritance":  "Autosomal Recessive · biallelic LOF (truncating + missense)",
        "prevalence":   "~1/500,000–1,000,000; ~60–80 published families (2026)",

        "mechanism": (
            "ANKS6 scaffolds the inversin compartment (IC) — the ciliary subdomain "
            "proximal to the transition zone, housing INVS (NPHP2), NPHP3, and NEK8 "
            "(NPHP9) as a functional tetramer. NEK8 phosphorylates ANKS6 linker region "
            "(aa 541–720) → stabilises IC. The IC acts as a Wnt-pathway switch: IC "
            "intact → suppresses canonical Wnt β-catenin signalling in tubular "
            "epithelium → maintains tubular identity. ANKS6 loss → IC collapse → "
            "canonical Wnt up-regulation → tubular EMT → TIN + corticomedullary cysts "
            "→ ESRD. In embryonic node cilia: IC guides PCP-mediated flow → left-right "
            "signalling. ANKS6 loss → incomplete laterality defect → situs inversus "
            "(~20–30%; incomplete penetrance c.f. INVS/NPHP2 >85%). BICC1 (SAM-domain "
            "partner) bridges ANKS6 to mTOR attenuation; loss → mild mTOR-driven "
            "cystogenesis overlaying TIN."
        ),

        "key_clinical_features": {
            "Renal (TIN + cysts)": (
                "Tubulointerstitial nephritis (TIN) + corticomedullary cysts + "
                "tubular concentrating defect (polyuria, polydipsia). ESRD median "
                "~13yr (juvenile pattern; similar to NPHP1). Renal USS: small "
                "echogenic kidneys ± cysts. Renal transplant CURATIVE — no "
                "recurrence (cell-autonomous IC defect)."
            ),
            "Situs inversus (20–30%)": (
                "NPHP16 is the second NPHP subtype (after NPHP2/INVS) with "
                "significant laterality defects. ~20–30% situs inversus totalis; "
                "~5% situs ambiguus. KEY DIFFERENTIATOR from most NPHP subtypes "
                "(NPHP1/4/5/6/8/10/13/14/15 have 0% laterality). Distinguish "
                "from NPHP2: NPHP2 = infantile onset (ESRD <3yr, situs inversus "
                ">85%); NPHP16 = juvenile onset (ESRD ~13yr, situs inversus 20–30%)."
            ),
            "No retinal dystrophy": (
                "ANKS6 not expressed in photoreceptor connecting cilia → no "
                "retinal degeneration. ERG normal in all NPHP16 patients. "
                "Distinguishes NPHP16 from NPHP5/IQCB1, CEP290/NPHP6, "
                "SDCCAG8/NPHP10, and CEP164/NPHP15 (all with retinal). "
                "No ophthalmology monitoring required."
            ),
            "No CHF / no Joubert / no ID": (
                "ANKS6 absent from biliary epithelium → no congenital hepatic "
                "fibrosis (CHF). No cerebellar expression → no Joubert/MTS. "
                "No neuronal expression → no intellectual disability. "
                "Pure renal ± laterality phenotype defines NPHP16."
            ),
            "Situs inversus + juvenile CKD → IC ciliopathy": (
                "Situs inversus + CKD in a child/adolescent → IC ciliopathy "
                "top differential: NPHP16 (ANKS6) or NPHP2 (INVS) or NPHP3 "
                "(rare situs). Onset age discriminates: NPHP2 infantile "
                "(ESRD before age 3), NPHP16 juvenile (ESRD median ~13yr). "
                "Always co-sequence ANKS6 + INVS + NPHP3 + NEK8 — IC tetramer."
            ),
        },

        "genetic_architecture": {
            "gene":          "ANKS6 (Ankyrin Repeat and SAM Domain-Containing Protein 6)",
            "aliases":       "PKDR1 (Polycystic Kidney Disease-Related 1); NPHP16",
            "size":          "982 aa · ~110 kDa",
            "domains": (
                "Ankyrin repeat domain (aa 1–540): 12 tandem ankyrin repeats; "
                "INVS/NPHP2 direct binding; NPHP3 interaction; IC scaffold core. | "
                "Linker / regulatory region (aa 541–720): NEK8 phospho-targets; "
                "IC integrity control; kinase-docking motifs. | "
                "SAM domain (aa 721–982): sterile alpha motif; homo-oligomerisation; "
                "BICC1 interaction; mTOR-pathway interface."
            ),
            "IC_module": (
                "Inversin compartment tetramer: INVS (NPHP2) · ANKS6 (NPHP16) · "
                "NPHP3 · NEK8 (NPHP9). Always co-sequence all four when any IC "
                "ciliopathy suspected. Digenic IC variants reported."
            ),
            "phosphorylation": (
                "NEK8 (NPHP9) phosphorylates ANKS6 linker region (Ser residues, "
                "aa 541–720) → stabilises IC scaffold. NEK8 kinase-dead or ANKS6 "
                "phospho-site mutants cause equivalent IC collapse phenotype."
            ),
            "key_variants": [
                "p.Arg823Trp (c.2467C>T) — European founder; most common NPHP16 "
                "variant; SAM-domain boundary; IC mislocalisation; pure NPHP16 "
                "phenotype (renal only); Otto 2013 AJHG.",
                "p.Gly40Glu (c.119G>A) — ankyrin repeat 1; N-terminal structural "
                "disruption; juvenile-onset pure NPHP16; pan-ethnic.",
                "p.Ser615Asn — linker phospho-target region; NEK8 phosphorylation "
                "site disruption; NPHP16 + situs inversus totalis; European.",
                "p.Gln561Ter — truncating; SAM-domain loss; severe early-onset "
                "NPHP16 ± situs inversus; compound het with missense allele.",
                "p.Arg407Gln — ankyrin repeat 9; Middle Eastern, consanguineous "
                "homozygous; moderate NPHP16 renal phenotype.",
            ],
        },

        "nphp_comparison": {
            "★ NPHP16 (ANKS6) — This patient": (
                "IC scaffold; 9q22.33; juvenile ESRD ~13yr; situs inversus 20–30%; "
                "no retinal; no CHF; no Joubert; no ID; NEK8/INVS/NPHP3 co-sequence"
            ),
            "NPHP2 (INVS) — infantile NPHP + situs": (
                "IC anchor; 9q31.1; ESRD <3yr (infantile!); situs inversus >85%; "
                "no retinal; key DDx: onset age discriminates NPHP16 vs NPHP2"
            ),
            "NPHP9 (NEK8) — IC kinase, ANKS6 phosphorylates": (
                "IC kinase; 17q11.2; juvenile NPHP; rare situs inversus; "
                "pancreatic ductal ectasia (unique — NPHP16 absent); hepatic cysts"
            ),
            "NPHP3 — IC component + CHF + situs": (
                "IC member; 3q22.1; juvenile NPHP; CHF ~25%; situs inversus ~10%; "
                "NPHP16 has NO CHF (key DDx from NPHP3)"
            ),
            "NPHP1 (NPHP1) — pure renal, no situs": (
                "TZ-scaffold; 2q13; juvenile ESRD ~13yr; 0% situs inversus; "
                "most common NPHP; MLPA-detectable deletion (NPHP16 not detected)"
            ),
            "NPHP15 (CEP164) — DAP initiator, SLS": (
                "Distal appendage; 11q23.3; ESRD ~13–15yr; SLS retinal 35–40%; "
                "0% situs inversus; TTBK2 phospho-target; NPHP16 has NO retinal"
            ),
        },

        "ddx_table": {
            "NPHP2 / INVS — IC anchor (most critical DDx)": (
                "NPHP2 = infantile NPHP (ESRD before age 3), situs inversus >85%; "
                "NPHP16 = juvenile NPHP (ESRD median ~13yr), situs inversus 20–30%. "
                "If situs inversus + CKD onset >3yr → NPHP16 far more likely than NPHP2."
            ),
            "NPHP9 / NEK8 — IC kinase phenocopies ANKS6": (
                "NEK8 loss phenocopies ANKS6 loss (IC collapse via same pathway). "
                "NPHP9 uniquely adds pancreatic ductal ectasia and hepatic cysts "
                "(absent in NPHP16). Always co-sequence NEK8 + ANKS6."
            ),
            "NPHP3 — IC member + CHF": (
                "NPHP3 causes IC-pattern NPHP + CHF (~25%) + rare situs inversus. "
                "NPHP16 has NO CHF. CHF presence immediately shifts toward NPHP3 or NPHP11."
            ),
            "Kartagener / Primary Ciliary Dyskinesia (PCD)": (
                "PCD (DNAI1, DNAI2, CCDC39, etc.) causes situs inversus + "
                "bronchiectasis + sinusitis + male infertility — MOTILE cilia defect. "
                "NPHP16 has NO respiratory tract disease — renal-only ± situs. "
                "Absent bronchiectasis/sinusitis excludes PCD."
            ),
            "ADPKD / PKD1-PKD2": (
                "ADPKD cysts are cortical/medullary diffuse (not corticomedullary); "
                "dominant inheritance; no situs inversus; HTN predominates; no "
                "concentrating defect until late. NPHP16 = AR; corticomedullary cysts; "
                "tubular polyuria first; situs inversus possible."
            ),
            "NPHP1 (MLPA-detectable deletion)": (
                "NPHP1 = most common NPHP; 290kb deletion on 2q13 detected by MLPA. "
                "ANKS6 (9q22.33) is NOT detected by NPHP1 MLPA. NPHP1 has 0% situs "
                "inversus. If NPHP1 MLPA negative + situs inversus → immediately "
                "suspect NPHP16/ANKS6 and perform WES."
            ),
        },

        "diagnostic_criteria": {
            "Mandatory (all required)": (
                "1. Biallelic pathogenic ANKS6 variants (WES/gene panel). "
                "2. Renal ciliopathy phenotype: TIN + corticomedullary cysts + "
                "concentrating defect. 3. AR inheritance pattern."
            ),
            "Supportive (any 1 of)": (
                "Situs inversus totalis or situs ambiguus. IC module partner variant "
                "(INVS/NPHP2, NPHP3, NEK8/NPHP9) on WES. Functional ANKS6 protein "
                "loss on renal biopsy immunostaining."
            ),
            "Exclusion criteria": (
                "Retinal dystrophy (ERG abnormal) — rules out NPHP16 (consider "
                "NPHP5/6/10/15). Congenital hepatic fibrosis — consider NPHP3/NPHP11. "
                "Bronchiectasis/sinusitis — consider PCD not NPHP16. Intellectual "
                "disability — consider NPHP14/ZNF423. Joubert Molar Tooth Sign — "
                "consider NPHP6/CEP290 or NPHP8/RPGRIP1L."
            ),
        },

        "treatment": {
            "Renal replacement": (
                "Renal transplant is CURATIVE for NPHP16 renal disease — no "
                "recurrence in transplanted kidney (cell-autonomous IC defect). "
                "Pre-emptive transplant preferred when feasible. Outcomes excellent. "
                "Living-related donor evaluation must include renal USS + genetic "
                "screening (carrier relatives have one normal allele; kidneys normal)."
            ),
            "Situs inversus management": (
                "Document organ situs at diagnosis — critical for surgical planning. "
                "Cardiac anatomy: formal echocardiography mandatory (dextrocardia "
                "± congenital heart disease in situs ambiguus). Abdominal USS for "
                "organ position. Alert surgical teams before any procedure. "
                "Situs inversus solitus (pure mirror image) is not directly treated "
                "but requires pre-operative documentation."
            ),
            "IC tetramer co-sequencing": (
                "Always sequence INVS (NPHP2) + ANKS6 (NPHP16) + NPHP3 + NEK8 "
                "(NPHP9) together — IC functional unit. Digenic IC variants reported. "
                "If one IC gene has one pathogenic variant, WES full IC panel mandatory."
            ),
            "Investigational": (
                "No disease-modifying therapy available (2026). "
                "Wnt-pathway modulators (β-catenin inhibitors) and NEK8-pathway "
                "activators are pre-clinical in zebrafish anks6 models. "
                "BICC1-mTOR axis is a potential therapeutic target (pre-clinical). "
                "Registry enrolment (RareCare / EURO-RDI / RaDer) essential."
            ),
        },

        "prognosis": (
            "NPHP16 follows the juvenile NPHP trajectory: ESRD median ~13yr (range "
            "8–19yr). Renal transplant is curative — no disease recurrence in the "
            "graft. Situs inversus solitus alone does not affect long-term outcome; "
            "situs ambiguus with structural CHD may require cardiac intervention. "
            "No retinal, cerebellar, hepatic, or cognitive involvement — pure renal "
            "(± laterality) ciliopathy. Quality of life post-transplant is excellent "
            "for renal disease; situs-related surgical risks require life-long "
            "documentation and surgical team awareness."
        ),

        "cohort_note": (
            f"Synthetic 40-patient NPHP16 cohort (seed={SEED}). Phenotype frequencies "
            "calibrated to published ANKS6 literature (Otto 2013 AJHG, Hoff 2013, "
            "Taskiran 2014, Srivastava 2017, Rao 2016). Situs inversus rate "
            "(~20–30%) reflects incomplete penetrance of IC laterality defect vs "
            "INVS/NPHP2 (>85%). All patients are de-identified composites for "
            "clinical education only."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"cohort_n          : {ov['cohort_n']}")
    print(f"median_gfr        : {ov['median_gfr']}")
    print(f"pct_situs_inversus: {ov['pct_situs_inversus']}%")
    print(f"pct_any_laterality: {ov['pct_any_laterality']}%")
    print(f"pct_esrd_or_tx    : {ov['pct_esrd_or_transplant']}%")
    print(f"pct_misdiag_nphp1 : {ov['pct_misdiagnosed_nphp1']}%")
    print("\n=== BREAKDOWN (sample) ===")
    bk = get_breakdown()
    print("situs:", json.dumps(bk['situs_distribution'], indent=2))
    print("\n=== DEFINITIONS (snippet) ===")
    df = get_definitions()
    print("disease:", df['disease'][:120])
