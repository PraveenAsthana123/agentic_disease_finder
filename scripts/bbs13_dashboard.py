"""
BBS13 Bardet-Biedl Syndrome Type 13 — MKS1 (Transition Zone Scaffold) Dashboard
==================================================================================
Primary Gene : MKS1 (*609883) — Meckel Syndrome Type 1 protein; 559 aa; Chr 17q22.
               MKS1 is a TRANSITION ZONE (TZ) SCAFFOLD protein and the FIRST BBS GENE
               THAT IS NOT a BBSome subunit or BCC chaperonin component (BBS1–12).
               MKS1 is part of the NPHP-MKS-JBTS ciliary transition zone module,
               a multi-protein complex that forms the ciliary gate (Y-links,
               membrane-to-axoneme tethers) and the diffusion barrier between the
               periciliary membrane and the ciliary compartment.
               Domain organisation (559 aa):
                 N-terminal domain (aa 1–120): NPHP4-binding scaffold; LRR-like
                   secondary structure; contacts NPHP1-NPHP4 complex at TZ base;
                 MKS1/MKSR central domain (aa 121–400): conserved MKSR fold;
                   CC2D2A and B9D1/B9D2 interaction surface; structural scaffold for
                   the MKS sub-module of the TZ (MKS1 · CC2D2A · TMEM216 ·
                   TMEM67 · B9D1 · B9D2 · MKS3/TMEM67);
                 C-terminal domain (aa 401–559): NPHP4 secondary contact; FBF1
                   interaction; tethers MKS module to the NPHP module.
               MKS1 is required for Y-link formation and TZ diffusion barrier
               integrity. Without MKS1, Y-links are structurally disrupted, the
               TZ gate is leaky, and ciliary membrane protein composition is
               dysregulated — GPCR cargoes mismigrate not because the BBSome
               fails to assemble (BBSome is NORMAL in BBS13) but because the
               TZ gate cannot restrict their movement.
               MKS1 LOF IF fingerprint — UNIQUE among all BBS types:
                 BBSome subunits NORMAL (BBS2 NORMAL · BBS4 NORMAL · BBS8 NORMAL ·
                 BBS9 NORMAL); MKKS/BBS6 NORMAL (BCC intact); MKS1 ABSENT.
                 NPHP4 and CC2D2A deranged at TZ.
               ALLELE-CLASS DETERMINES DISEASE SEVERITY (unique to BBS13):
                 Null/null (truncating/truncating) → Meckel-Gruber Syndrome (#249000)
                   — lethal: occipital encephalocele, polydactyly, cystic kidneys;
                 Hypomorphic/hypomorphic → BBS13 (#613464) — full BBS phenotype;
                 Hypomorphic/truncating → BBS13 or JBTS28 (#614291) — variable;
                 BBS13 therefore REQUIRES at least one hypomorphic allele with
                 residual MKS1 scaffold function.
Disease        : Bardet-Biedl Syndrome Type 13 (#613464); allelic: MKS1 (#249000);
                 JBTS28 (#614291)
Frequency      : ~1% of all BBS (very rare); fewer than 80 families reported 2026
Inheritance    : Autosomal recessive (AR); biallelic LOF (at least 1 hypomorphic)
Tri-allelic    : ~3% of BBS13 families
Cohort         : 40 patients; educational/synthetic; seed 357
Endpoints      : /api/bbs13/overview | /api/bbs13/breakdown | /api/bbs13/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 357
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("European",      0.32),   # France, Netherlands, UK, Germany — most reports
        ("MENA (other)",  0.26),   # Saudi Arabia, Iran, Turkey — consanguineous
        ("South Asian",   0.18),   # India, Pakistan — consanguineous
        ("North African", 0.12),   # Tunisia, Morocco — consanguineous
        ("East Asian",    0.06),
        ("Other",         0.06),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "European":      ["c.577+1G>A (splice site, intron 5)", "p.Leu178Pro (c.532T>C)", "p.Gln404Ter (c.1210C>T)"],
        "MENA (other)":  ["p.Arg470Cys (c.1408C>T)", "p.Gln404Ter (c.1210C>T)", "c.577+1G>A (splice site, intron 5)"],
        "South Asian":   ["p.Ile455Val (c.1363A>G)", "p.Leu178Pro (c.532T>C)", "p.Gln404Ter (c.1210C>T)"],
        "North African": ["p.Arg470Cys (c.1408C>T)", "p.Ile455Val (c.1363A>G)", "p.Gln404Ter (c.1210C>T)"],
        "East Asian":    ["p.Arg72His (c.215G>A)", "p.Gln404Ter (c.1210C>T)", "p.Ile455Val (c.1363A>G)"],
        "Other":         ["p.Gln404Ter (c.1210C>T)", "p.Arg72His (c.215G>A)"],
    }

    allele_classes = [
        "Hypomorphic/Hypomorphic",   # BBS13-only allele class
        "Hypomorphic/Truncating",    # BBS13 or JBTS28
        "Splice/Hypomorphic",
        "Splice/Truncating",
        "Missense/Missense",         # some hypomorphic missense
    ]
    allele_weights = [0.34, 0.30, 0.18, 0.12, 0.06]

    # BBS13-specific retinal stages (same rod-first pattern as BBS1-12)
    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.18, 0.30, 0.29, 0.23]

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.32, 0.11, 0.07, 0.05, 0.45]   # ~55% polydactyly (lower than BBS1-12)

    # BBS13-specific renal: NPHP/tubular pattern dominates (unique vs BBS1-12 cystic)
    renal_types = ["Nephronophthisis (tubular atrophy/fibrosis)", "Structural anomaly (horseshoe/duplex)",
                   "Cystic dysplasia (Meckel overlap)", "ESRD", "None"]
    renal_w = [0.24, 0.08, 0.10, 0.10, 0.48]   # ~52% renal (higher NPHP rate)

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.15, 0.42, 0.28, 0.15]

    misdiag = ["Isolated RP / Leber amaurosis", "Obesity syndrome (Prader-Willi screen)",
               "Joubert Syndrome (JBTS28 overlap)", "Alström Syndrome (cone-rod confusion)",
               "No initial misdiagnosis"]
    mis_w   = [0.25, 0.18, 0.12, 0.08, 0.37]

    presentation_ages = ["Birth (polydactyly noted)", "Infancy (nystagmus/strabismus)",
                         "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"]
    pres_w  = [0.22, 0.16, 0.28, 0.22, 0.12]

    patients = []
    for i in range(n):
        eth = rng.choices(eth_labels, weights=eth_weights, k=1)[0]
        v_pool = variants_by_eth[eth]
        v1 = rng.choice(v_pool)
        v2 = rng.choice(v_pool)
        while v2 == v1 and len(v_pool) > 1:
            v2 = rng.choice(v_pool)

        poly_type  = rng.choices(polydactyly_types, weights=poly_w, k=1)[0]
        renal_type = rng.choices(renal_types, weights=renal_w, k=1)[0]
        ret_stage  = rng.choices(retinal_stages, weights=ret_w, k=1)[0]
        allele_cl  = rng.choices(allele_classes, weights=allele_weights, k=1)[0]
        dx_ag      = rng.choices(dx_ages, weights=dx_w, k=1)[0]
        pres_age   = rng.choices(presentation_ages, weights=pres_w, k=1)[0]
        mis_d      = rng.choices(misdiag, weights=mis_w, k=1)[0]

        # BBS13: ~55% polydactyly, ~68% obesity, ~42% cognitive LD,
        #        ~50% hypogonadism, ~45% anosmia, ~52% renal (NPHP dominant),
        #        ~5% CHD, ~12% liver fibrosis (Meckel overlap), ~8% JBTS overlap
        polydactyly   = poly_type != "None"
        obesity       = rng.random() < 0.68
        cognitive_ld  = rng.random() < 0.42
        hypogonadism  = rng.random() < 0.50
        anosmia       = rng.random() < 0.45
        renal_any     = renal_type != "None"
        chd           = rng.random() < 0.05
        liver_fibrosis = rng.random() < 0.12   # unique to BBS13 among BBS types
        jbts_overlap  = allele_cl in ("Hypomorphic/Truncating", "Splice/Truncating") and rng.random() < 0.22
        ret_end       = ret_stage == "Legal blindness / end-stage"
        tri_bbs       = rng.random() < 0.03    # ~3% tri-allelic
        mis_dx        = mis_d != "No initial misdiagnosis"
        esrd          = renal_type == "ESRD"
        consang       = eth in ("MENA (other)", "South Asian", "North African") and rng.random() < 0.65

        patients.append({
            "id":                   f"BBS13-{i+1:03d}",
            "ethnicity":            eth,
            "variant1":             v1,
            "variant2":             v2,
            "allele_class":         allele_cl,
            "polydactyly":          polydactyly,
            "polydactyly_type":     poly_type,
            "obesity":              obesity,
            "cognitive_ld":         cognitive_ld,
            "hypogonadism":         hypogonadism,
            "anosmia":              anosmia,
            "renal_any":            renal_any,
            "renal_type":           renal_type,
            "chd":                  chd,
            "liver_fibrosis":       liver_fibrosis,
            "jbts_overlap":         jbts_overlap,
            "retinal_endstage":     ret_end,
            "retinal_stage":        ret_stage,
            "dx_age_group":         dx_ag,
            "presentation_age":     pres_age,
            "triallelic_bbs":       tri_bbs,
            "initial_misdiagnosis": mis_dx,
            "misdiagnosis":         mis_d,
            "esrd":                 esrd,
            "consanguinity":        consang,
        })
    return patients


_COHORT: List[Dict[str, Any]] = _make_cohort()


def _pct(n: int, total: int = _N) -> float:
    return round(100 * n / total, 1) if total else 0.0


# ── API handlers ──────────────────────────────────────────────────────────────
def get_overview() -> Dict[str, Any]:
    c = _COHORT
    n = _N

    poly_n         = sum(1 for p in c if p["polydactyly"])
    obesity_n      = sum(1 for p in c if p["obesity"])
    cognitive_n    = sum(1 for p in c if p["cognitive_ld"])
    renal_any_n    = sum(1 for p in c if p["renal_any"])
    hypo_n         = sum(1 for p in c if p["hypogonadism"])
    anosmia_n      = sum(1 for p in c if p["anosmia"])
    chd_n          = sum(1 for p in c if p["chd"])
    liver_n        = sum(1 for p in c if p["liver_fibrosis"])
    jbts_n         = sum(1 for p in c if p["jbts_overlap"])
    ret_end_n      = sum(1 for p in c if p["retinal_endstage"])
    tri_n          = sum(1 for p in c if p["triallelic_bbs"])
    mis_n          = sum(1 for p in c if p["initial_misdiagnosis"])
    esrd_n         = sum(1 for p in c if p["esrd"])
    consang_n      = sum(1 for p in c if p["consanguinity"])

    nphp_n         = sum(1 for p in c if p["renal_type"] == "Nephronophthisis (tubular atrophy/fibrosis)")

    ret_stages: Dict[str, int] = {}
    for p in c:
        s = p["retinal_stage"]
        ret_stages[s] = ret_stages.get(s, 0) + 1

    age_dist: Dict[str, int] = {}
    for p in c:
        g = p["dx_age_group"]
        age_dist[g] = age_dist.get(g, 0) + 1

    return {
        "cohort_n": n,
        "gene":     "MKS1",
        "omim_gene": "*609883",
        "omim_disease": "#613464",
        "locus":    "17q22",
        "protein":  "559 aa — Transition Zone scaffold (NPHP-MKS-JBTS module)",
        "mechanism": "Transition zone (TZ) gate scaffolding — NPHP-MKS module; BBSome assembles normally; TZ gate leaky → GPCR mismigration",
        "bbs_frequency": "~1% of all BBS (rare; <80 families 2026)",
        "inheritance": "Autosomal recessive; biallelic LOF (≥1 hypomorphic allele required for BBS phenotype)",
        "triallelic_pct": 3,
        "allele_class_note": "Null/null → Meckel-Gruber (lethal); hypomorphic/hypomorphic → BBS13; hypomorphic/truncating → BBS13 or JBTS28",
        "if_fingerprint": "BBSome subunits NORMAL (BBS2·BBS4·BBS8·BBS9 all present); MKKS/BCC NORMAL; MKS1 ABSENT — unique among all BBS types",
        "key_counts": {
            "polydactyly_n":   poly_n,   "polydactyly_pct":   _pct(poly_n),
            "obesity_n":       obesity_n, "obesity_pct":       _pct(obesity_n),
            "cognitive_n":     cognitive_n, "cognitive_pct":   _pct(cognitive_n),
            "renal_any_n":     renal_any_n, "renal_any_pct":  _pct(renal_any_n),
            "nphp_n":          nphp_n,    "nphp_pct":          _pct(nphp_n),
            "hypogonadism_n":  hypo_n,    "hypogonadism_pct":  _pct(hypo_n),
            "anosmia_n":       anosmia_n, "anosmia_pct":       _pct(anosmia_n),
            "chd_n":           chd_n,     "chd_pct":           _pct(chd_n),
            "liver_fibrosis_n": liver_n,  "liver_fibrosis_pct": _pct(liver_n),
            "jbts_overlap_n":  jbts_n,    "jbts_overlap_pct":  _pct(jbts_n),
            "retinal_endstage_n": ret_end_n, "retinal_endstage_pct": _pct(ret_end_n),
            "triallelic_n":    tri_n,     "triallelic_pct":     _pct(tri_n),
            "misdiagnosis_n":  mis_n,     "misdiagnosis_pct":   _pct(mis_n),
            "esrd_n":          esrd_n,    "esrd_pct":           _pct(esrd_n),
            "consanguinity_n": consang_n, "consanguinity_pct":  _pct(consang_n),
        },
        "systemic_burden": [
            ("Rod-Cone Dystrophy (rod-first)",   n,        _pct(n)),
            ("Obesity (TZ/LepR pathway)",         obesity_n, _pct(obesity_n)),
            ("Polydactyly (post-axial)",           poly_n,   _pct(poly_n)),
            ("Renal anomaly (NPHP dominant)",      renal_any_n, _pct(renal_any_n)),
            ("Nephronophthisis specifically",      nphp_n,   _pct(nphp_n)),
            ("Hypogonadism",                       hypo_n,   _pct(hypo_n)),
            ("Cognitive/LD",                       cognitive_n, _pct(cognitive_n)),
            ("Anosmia/Hyposmia",                  anosmia_n, _pct(anosmia_n)),
            ("Liver fibrosis (Meckel overlap)",    liver_n,  _pct(liver_n)),
            ("Joubert overlap (JBTS28)",           jbts_n,   _pct(jbts_n)),
            ("CHD",                                chd_n,    _pct(chd_n)),
        ],
        "retinal_stage_distribution": {k: v for k, v in ret_stages.items()},
        "dx_age_distribution":        {k: v for k, v in age_dist.items()},
    }


def get_breakdown() -> Dict[str, Any]:
    c = _COHORT
    n = _N

    systemic = [
        ("Rod-Cone Dystrophy (rod-first)",   n,                                          _pct(n)),
        ("Obesity (TZ/LepR pathway)",         sum(1 for p in c if p["obesity"]),          _pct(sum(1 for p in c if p["obesity"]))),
        ("Polydactyly (post-axial)",           sum(1 for p in c if p["polydactyly"]),     _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Renal anomaly (any)",                sum(1 for p in c if p["renal_any"]),       _pct(sum(1 for p in c if p["renal_any"]))),
        ("Nephronophthisis specifically",      sum(1 for p in c if p["renal_type"] == "Nephronophthisis (tubular atrophy/fibrosis)"), _pct(sum(1 for p in c if p["renal_type"] == "Nephronophthisis (tubular atrophy/fibrosis)"))),
        ("ESRD",                               sum(1 for p in c if p["esrd"]),            _pct(sum(1 for p in c if p["esrd"]))),
        ("Hypogonadism",                       sum(1 for p in c if p["hypogonadism"]),    _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Cognitive/LD",                       sum(1 for p in c if p["cognitive_ld"]),    _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("Anosmia/Hyposmia",                  sum(1 for p in c if p["anosmia"]),          _pct(sum(1 for p in c if p["anosmia"]))),
        ("Liver fibrosis (Meckel overlap)",    sum(1 for p in c if p["liver_fibrosis"]), _pct(sum(1 for p in c if p["liver_fibrosis"]))),
        ("Joubert overlap (JBTS28)",           sum(1 for p in c if p["jbts_overlap"]),   _pct(sum(1 for p in c if p["jbts_overlap"]))),
        ("CHD",                                sum(1 for p in c if p["chd"]),             _pct(sum(1 for p in c if p["chd"]))),
    ]

    eth_counts: Dict[str, int] = {}
    for p in c:
        eth_counts[p["ethnicity"]] = eth_counts.get(p["ethnicity"], 0) + 1

    ac_counts: Dict[str, int] = {}
    for p in c:
        ac_counts[p["allele_class"]] = ac_counts.get(p["allele_class"], 0) + 1

    ret_counts: Dict[str, int] = {}
    for p in c:
        ret_counts[p["retinal_stage"]] = ret_counts.get(p["retinal_stage"], 0) + 1

    renal_counts: Dict[str, int] = {}
    for p in c:
        renal_counts[p["renal_type"]] = renal_counts.get(p["renal_type"], 0) + 1

    poly_counts: Dict[str, int] = {}
    for p in c:
        poly_counts[p["polydactyly_type"]] = poly_counts.get(p["polydactyly_type"], 0) + 1

    pres_counts: Dict[str, int] = {}
    for p in c:
        pres_counts[p["presentation_age"]] = pres_counts.get(p["presentation_age"], 0) + 1

    mis_counts: Dict[str, int] = {}
    for p in c:
        mis_counts[p["misdiagnosis"]] = mis_counts.get(p["misdiagnosis"], 0) + 1

    var_counts: Dict[str, int] = {}
    for p in c:
        for v in [p["variant1"], p["variant2"]]:
            var_counts[v] = var_counts.get(v, 0) + 1
    top_variants = sorted(var_counts.items(), key=lambda x: -x[1])[:6]

    return {
        "cohort_n": n,
        "systemic_burden": [
            {"feature": feat, "n": nn, "pct": pct_val}
            for feat, nn, pct_val in systemic
        ],
        "ethnicity_distribution": [
            {"ethnicity": eth, "n": cnt}
            for eth, cnt in sorted(eth_counts.items(), key=lambda x: -x[1])
        ],
        "allele_class_summary": [
            {"label": ac, "n": cnt}
            for ac, cnt in sorted(ac_counts.items(), key=lambda x: -x[1])
        ],
        "retinal_stage_distribution": [
            {"label": s, "n": ret_counts.get(s, 0)}
            for s in ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
        ],
        "renal_distribution": [
            {"label": r, "n": renal_counts.get(r, 0)}
            for r in ["Nephronophthisis (tubular atrophy/fibrosis)", "Structural anomaly (horseshoe/duplex)",
                      "Cystic dysplasia (Meckel overlap)", "ESRD", "None"]
        ],
        "polydactyly_distribution": [
            {"label": p, "n": poly_counts.get(p, 0)}
            for p in ["Post-axial hands+feet", "Post-axial hands only",
                      "Post-axial feet only", "Pre-axial (rare)", "None"]
        ],
        "presentation_distribution": [
            {"label": p, "n": pres_counts.get(p, 0)}
            for p in ["Birth (polydactyly noted)", "Infancy (nystagmus/strabismus)",
                      "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"]
        ],
        "misdiagnosis_distribution": [
            {"label": m, "n": mis_counts.get(m, 0)}
            for m in ["Isolated RP / Leber amaurosis", "Obesity syndrome (Prader-Willi screen)",
                      "Joubert Syndrome (JBTS28 overlap)", "Alström Syndrome (cone-rod confusion)",
                      "No initial misdiagnosis"]
        ],
        "top_variants": [{"variant": v, "n": cnt} for v, cnt in top_variants],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":              "MKS1 (Meckel Syndrome Type 1 protein); Chr 17q22",
            "OMIM_Gene":         "*609883",
            "Chromosome":        "17q22",
            "Protein":           "559 aa — N-terminal NPHP4-binding domain (aa 1–120; LRR-like; NPHP1-NPHP4 contact) · central MKSR domain (aa 121–400; CC2D2A and B9D1/B9D2 interaction; TZ scaffold) · C-terminal domain (aa 401–559; NPHP4 secondary contact; FBF1 interaction; MKS-NPHP module bridge)",
            "Domains":           "aa 1–120: N-terminal domain (NPHP4-binding scaffold; LRR-like secondary structure) · aa 121–400: central MKSR domain (conserved fold; CC2D2A + B9D1 + B9D2 contact surface; core TZ scaffold subunit) · aa 401–559: C-terminal domain (NPHP4 secondary contact; FBF1 tethering; bridges NPHP and MKS modules at TZ)",
            "Module":            "NPHP-MKS-JBTS transition zone (TZ) module — MKS sub-module: MKS1 · CC2D2A · TMEM216 · TMEM67 · B9D1 · B9D2 · MKS3/TMEM67",
            "Inheritance":       "Autosomal Recessive; biallelic LOF (at least 1 hypomorphic allele required for BBS phenotype; null/null → Meckel-Gruber lethal)",
            "Frequency_BBS":     "~1% of all BBS; fewer than 80 families reported worldwide (2026); very rare BBS subtype",
            "Allele_spectrum":   "Null/null (truncating) → Meckel-Gruber (#249000 lethal); hypomorphic/hypomorphic → BBS13; hypomorphic/truncating → BBS13 or JBTS28; BBS13 requires ≥1 hypomorphic allele with residual MKS1 scaffold function",
            "Triallelic_BBS":    "~3% (lower than BBS1-12; compatible partners not yet systematically defined)",
            "Allelic_diseases":  "BBS13 (#613464) · Joubert Syndrome 28 JBTS28 (#614291) · Meckel-Gruber Syndrome MKS1 (#249000) — same gene, allele class determines severity",
            "CHD_penetrance":    "~5%",
            "Population_note":   "Pan-ethnic distribution; no single dominant founder; European, MENA, South Asian cohorts roughly proportionate to overall BBS reporting bias",
        },
        "disease_card": {
            "Disease":            "Bardet-Biedl Syndrome Type 13 (BBS13) — #613464",
            "Prevalence":         "~1:100,000–160,000 worldwide (BBS total); BBS13 subtype ~1% (~1–1.5 in 10,000,000) — very rare",
            "Cardinal_features":  "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR/TZ pathway) · Hypogonadism · Renal anomalies (NPHP pattern dominant) · Cognitive/LD · Anosmia",
            "BBS13_unique":       "FIRST BBS gene that is NOT a BBSome or BCC subunit — BBSome IF NORMAL (BBS2/BBS4/BBS8/BBS9 present); MKS1 ABSENT; NPHP4/CC2D2A deranged at TZ; allele class controls disease tier (MKS1 → BBS13 → JBTS28); liver fibrosis possible (~12%); NPHP renal pattern dominant",
            "Retinal":            "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; same ERG pattern as BBS1-12)",
            "Obesity_mechanism":  "TZ gate leaky → hypothalamic GPCR (LepR) mismigration via TZ permeability failure (not BBSome LOF); GLP1RA applicable; indirect mechanism (BBSome intact but TZ gate dysfunctional)",
            "Renal":              "Nephronophthisis (tubular atrophy, interstitial fibrosis, concentrating defect) DOMINANT — distinct from BBS1-12 cystic/structural pattern; ESRD risk ~10%; cystic dysplasia ~10% (Meckel allele overlap)",
            "Liver":              "Hepatic ductal plate malformation / congenital hepatic fibrosis (~12%) — Meckel overlap; unique among BBS types; liver function testing recommended at diagnosis",
            "JBTS_overlap":       "~8% of BBS13 cohort develops JBTS28 overlap (molar tooth sign; cerebellar vermis hypoplasia) — allele class (hypomorphic/truncating or splice/truncating) → insufficient MKS1 scaffold → TZ fails more severely → JBTS",
            "CHD":                "~5%",
            "Diagnosis":          "Full BBS gene panel (minimum 20–24 genes including MKS1); IF: BBSome panel NORMAL + MKS1 ABSENT + NPHP4 deranged at TZ — unique pattern distinguishing BBS13 from all BBS1-12; liver USS/fibroscan; MRI brain if JBTS overlap suspected",
            "Treatment":          "Symptomatic; GLP1RA for obesity (TZ mechanism, not direct BBSome LOF, but LepR mismigration still occurs); renal surveillance (NPHP pattern — ACE-i/ARB; avoid nephrotoxins; transplant for ESRD); liver monitoring",
            "Prognosis":          "Progressive visual loss as per BBS1-12; renal ESRD risk higher (~10%) due to NPHP pattern; near-normal survival with management; JBTS overlap may affect neurological prognosis",
        },
        "key_variants": [
            {
                "variant":     "p.Arg470Cys (c.1408C>T)",
                "domain":      "Central MKSR domain — CC2D2A interaction surface (Arg470 forms a key hydrogen-bond contact within the MKS1-CC2D2A interface in the TZ MKS sub-module)",
                "consequence": "CC2D2A docking to MKS1 MKSR domain disrupted; MKS sub-module structurally unstable; TZ Y-link assembly impaired; TZ gate leaky → GPCR mismigration; BBSome itself assembles normally (BCC intact); hypomorphic allele — residual MKS1 scaffold function preserved → BBS13 phenotype, not MKS1",
                "ethnicity":   "MENA (Iran, Turkey, Saudi Arabia); North African enrichment; most common BBS13 allele in MENA consanguineous families"
            },
            {
                "variant":     "c.577+1G>A (splice site, intron 5)",
                "domain":      "Intron 5 splice donor — exon 5 skipping → loss of aa 150–195 within central MKSR domain (B9D1/B9D2 binding core loop)",
                "consequence": "Partial loss of B9D1/B9D2 interaction surface; MKS sub-module weakened at B9 protein contacts; TZ gate partially disrupted; hypomorphic effect (residual splice product from cryptic donor preserves partial MKS1 function) → BBS13 phenotype; European cohorts",
                "ethnicity":   "European enrichment (France, UK, Germany); also pan-ethnic"
            },
            {
                "variant":     "p.Ile455Val (c.1363A>G)",
                "domain":      "Central MKSR domain hydrophobic core (Ile455 contributes to the hydrophobic packing of the MKSR β-sheet against the CC2D2A contact helix)",
                "consequence": "Mild destabilisation of MKSR fold → CC2D2A contact slightly weakened; TZ gate mildly leaky → BBS13 phenotype (not JBTS or MKS1 given mild effect); hypomorphic allele class; South Asian enrichment",
                "ethnicity":   "South Asian (India, Pakistan); also North African"
            },
            {
                "variant":     "p.Arg72His (c.215G>A)",
                "domain":      "N-terminal NPHP4-contact domain (Arg72 in the LRR-like surface contacting NPHP4; bridges MKS1 N-terminus to the NPHP1-NPHP4 TZ base complex)",
                "consequence": "NPHP4 docking to MKS1 N-terminus weakened; NPHP-MKS module bridge partially disrupted; TZ gate defective to greater degree than MKSR-only mutations; variable expressivity → some families: BBS13; others: JBTS28 (molar tooth sign); allele class context critical",
                "ethnicity":   "Pan-ethnic; reported across East Asian, European, and MENA cohorts; JBTS28 families enriched for this allele"
            },
            {
                "variant":     "p.Gln404Ter (c.1210C>T)",
                "domain":      "C-terminal domain — truncating null; stops at aa 404; removes entire C-terminal NPHP4-secondary-contact and FBF1-tethering domain",
                "consequence": "Complete loss of MKS-NPHP module bridge (C-terminus removed); when compound heterozygous with a hypomorphic allele (e.g. Ile455Val or splice) → BBS13 phenotype; when homozygous → Meckel-Gruber syndrome (lethal); pan-ethnic truncating allele",
                "ethnicity":   "Pan-ethnic (most common truncating allele in MKS1; present across all ethnic groups)"
            },
            {
                "variant":     "p.Leu178Pro (c.532T>C)",
                "domain":      "Central MKSR domain hydrophobic core (Leu178 is a structural residue in the MKSR α-helix packing against the B9D1-contact β-sheet)",
                "consequence": "MKSR fold destabilised; B9D1/B9D2 contacts weakened; MKS sub-module partially disrupted; hypomorphic allele → BBS13 phenotype; European enrichment; moderate phenotype severity (more severe than Ile455Val, less severe than Gln404Ter-based compound hets)",
                "ethnicity":   "European enrichment (France, Germany, UK)"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG rod-first) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20–24 genes including MKS1, CC2D2A, TMEM67/MKS3); MLPA/CNV for large deletions; sequencing both alleles to classify hypomorphic vs truncating",
            "3. CRITICAL: If gene panel returns two MKS1 alleles, classify both as hypomorphic vs truncating — null/null → verify patient has BBS phenotype NOT Meckel-Gruber (retrospective); biallelic truncating MKS1 surviving to BBS phenotype is extremely rare and should prompt MKS1 re-classification",
            "4. IF panel: BBSome subunits (BBS2, BBS4, BBS8, BBS9) ALL NORMAL — MKS1 ABSENT is the fingerprint; BCC panel (MKKS, BBS10, BBS12) NORMAL; NPHP4 deranged at TZ (IF shows NPHP4 pattern disrupted at ciliary base)",
            "5. MRI brain: if JBTS28 overlap suspected (allele class: hypomorphic/truncating or splice/truncating) → molar tooth sign on axial T1/T2; cerebellar vermis hypoplasia assessment",
            "6. Renal: NPHP pattern dominant — renal ultrasound (look for echogenic cortex, corticomedullary differentiation loss, small cysts at CMJ); annual GFR; urine concentrating ability; urinary cystatin C; avoid nephrotoxins; ESRD risk ~10%",
            "7. Liver: ultrasound + fibroscan at diagnosis — liver fibrosis/CHF in ~12% (Meckel overlap); liver function tests (ALT, GGT, ALP); hepatology referral if abnormal",
            "8. Ophthalmology: ERG (rod-first, scotopic extinguished) + Goldmann VF + OCT; identical protocol to BBS1-12; annual surveillance",
            "9. Obesity: GLP1RA applicable (TZ gate leaky → LepR mismigration → leptin resistance, same downstream effect as BBSome LOF — indirect but similar endpoint); bariatric surgery in refractory cases",
            "10. Genetic counselling: AR inheritance; sibling recurrence 25%; allele class counselling critical — if both parents carry truncating alleles, subsequent pregnancies at risk for Meckel-Gruber; prenatal/preimplantation genetic testing recommended",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; photosensitising drug avoidance; vitamin A supplementation not indicated for BBS13 (TZ mechanism distinct from retinol metabolism); gene therapy investigational (AAV-MKS1 not in clinical trials 2026)",
            "2. Obesity: GLP-1 receptor agonists (semaglutide) — LepR mismigration via TZ permeability failure (indirect mechanism but clinically similar to BBSome LOF); bariatric surgery in refractory cases after ophthalmology risk assessment",
            "3. Renal (NPHP pattern): ACE inhibitor/ARB for CKD slowing; avoid nephrotoxins (NSAIDs, IV contrast, aminoglycosides); early transplant planning for ESRD (~10% risk); renal transplant curative; note NPHP recurrence risk in transplant negligible (intrinsic cilia defect, extrinsic renal correction)",
            "4. Liver: monitor ALT/GGT/ALP and fibroscan annually; avoid hepatotoxins; portal hypertension management if advanced hepatic fibrosis; hepatology co-management",
            "5. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling",
            "6. Anosmia: safety counselling (gas leaks, food spoilage); smoke detectors mandatory",
            "7. Cognitive/LD: early intervention; IEP; neuropsychological assessment; JBTS overlap patients may have additional cerebellar ataxia management needs",
            "8. CHD (~5%): echocardiogram at diagnosis; cardiology surveillance; surgical/catheterisation as indicated",
            "9. Allele-class family counselling: siblings with biallelic truncating MKS1 risk Meckel-Gruber (lethal prenatal); accurate classification of both parental alleles as hypomorphic vs truncating is essential for family planning",
            "10. Surveillance: annual ophthalmology + renal (GFR, urinary concentrating ability, USS) + liver (USS/fibroscan/LFTs) + metabolic + cardiac; JBTS28 overlap cases: additional neurological/cerebellar surveillance",
        ],
        "ddx_table": [
            {"disease": "BBS1–BBS12 (BBSome/BCC genes)", "key_difference": "BBS1-12 LOF → BBSome IF ABNORMAL (BBS2/BBS4/BBS8 absent or reduced, depending on which subunit). BBS13/MKS1 LOF → BBSome IF NORMAL (BBS2/BBS4/BBS8/BBS9 ALL present) + MKS1 ABSENT. A normal BBSome IF panel in a patient with full BBS phenotype strongly suggests a TZ-module gene (MKS1, CEP290, LZTFL1, etc.) rather than a BBSome/BCC gene."},
            {"disease": "Meckel-Gruber Syndrome type 1 (MKS1 null/null)", "key_difference": "MKS1 null/null → Meckel-Gruber (lethal; occipital encephalocele, cystic renal dysplasia, polydactyly). BBS13 → MKS1 hypomorphic/hypomorphic or hypomorphic/truncating → surviving full BBS phenotype. Allele classification (truncating vs hypomorphic) is the key discriminator between these two MKS1 disease tiers."},
            {"disease": "Joubert Syndrome 28 (JBTS28 — MKS1 hypomorphic/truncating)", "key_difference": "JBTS28 → molar tooth sign on brain MRI; cerebellar vermis hypoplasia; some allele overlap with BBS13 (hypomorphic/truncating MKS1). BBS13 → full BBS cardinal features WITHOUT molar tooth sign (BBSome-mediated GPCR trafficking failure dominates; TZ leakiness not severe enough to cause cerebellar malformation). MRI brain discriminates JBTS28 from BBS13 in ambiguous allele-class families."},
            {"disease": "Nephronophthisis (NPHP1/NPHP4 LOF)", "key_difference": "Pure NPHP → renal nephronophthisis + possible retinal dystrophy (SLS = Senior-Løken) but NO polydactyly, NO obesity, NO BBS multi-system features. BBS13 → full BBS phenotype PLUS NPHP renal pattern (not cystic as in BBS1-12). NPHP4 IF deranged in BBS13 (MKS1 is NPHP4's bridging partner) but NPHP4 LOF itself does not cause obesity or polydactyly."},
            {"disease": "Alström Syndrome (ALMS1)", "key_difference": "Cone-rod (cone first, not rod first); no polydactyly; infantile DCM (60%); SNHL; normal MKS1 and BBSome IF. Obesity and renal overlap but ERG pattern, DCM, and hearing loss distinguish."},
            {"disease": "BBS14 / CEP290 (next TZ gene in BBS series)", "key_difference": "CEP290 (BBS14) is the centrosomal/TZ protein IFT-capping factor. Both BBS13 and BBS14 share BBSome-normal IF. CEP290 mutations more commonly cause Leber congenital amaurosis (LCA) or isolated retinal dystrophy (milder alleles) before multi-system BBS. MKS1 mutations more commonly cause NPHP-predominant BBS or Meckel-Gruber (severe). Gene panel distinguishes."},
        ],
        "mechanism_glossary": [
            {
                "term": "Transition Zone (TZ) and the NPHP-MKS-JBTS Module",
                "definition": "The ciliary transition zone (TZ) is a ~0.5 μm region at the base of the cilium between the basal body and the ciliary axoneme. It forms the ciliary gate — a diffusion barrier that controls which membrane proteins enter and exit the ciliary compartment. The TZ is organised into two modules: the NPHP module (NPHP1, NPHP4, NPHP8, RPGRIP1L; tethered to NPHP-NPHP4 complex) and the MKS module (MKS1, CC2D2A, TMEM216, TMEM67/MKS3, B9D1, B9D2). MKS1 bridges these two modules: N-terminal domain contacts NPHP4; central MKSR domain scaffolds the MKS module; C-terminal domain provides secondary NPHP4 contact and FBF1 tethering. Without MKS1, the MKS-NPHP bridge fails and Y-links (the electron-dense TZ membrane-to-axoneme tethers) are structurally disrupted."
            },
            {
                "term": "BBS13 IF Fingerprint — BBSome NORMAL + MKS1 ABSENT (Unique Among All BBS)",
                "definition": "In BBS13 (MKS1 LOF), the BBSome assembles normally — BCC trimer is intact (MKKS/BBS6, BBS10, BBS12 all present); BBS2-BBS7 core dimer forms normally; the complete BBSome octamer reaches the TZ. This is unique among BBS1-12 (where BBSome or BCC assembly fails). On IF, BBS2, BBS4, BBS8, BBS9, and MKKS are ALL present at normal levels and distribution. Only MKS1 is absent. NPHP4 may show deranged TZ distribution (NPHP4 requires MKS1 bridging to localise correctly at the TZ base). The NORMAL BBSome IF panel in a full BBS phenotype patient is the key clinical flag directing to TZ-module gene sequencing."
            },
            {
                "term": "Allele Class Determines Disease Tier — MKS1 null/null → Meckel; hypomorphic → BBS13; intermediate → JBTS28",
                "definition": "MKS1 LOF disease outcome is uniquely allele-class-controlled: (1) null/null (truncating/truncating) → Meckel-Gruber Syndrome (MKS1; #249000) — lethal in the perinatal period; occipital encephalocele, multicystic kidneys, polydactyly; complete TZ failure. (2) hypomorphic/hypomorphic (both alleles preserve residual MKS1 scaffold function) → BBS13 (#613464) — TZ gate leaky but not absent; surviving multi-system BBS. (3) hypomorphic/truncating → BBS13 or JBTS28 (#614291) — intermediate TZ failure; some families develop molar tooth sign (cerebellar vermis hypoplasia = JBTS28), others maintain BBS phenotype without cerebellar malformation, depending on residual MKS1 scaffold activity from the hypomorphic allele. This allele-class severity gradient is unique to BBS13 among all BBS genes."
            },
            {
                "term": "Nephronophthisis (NPHP) Renal Pattern in BBS13 — Distinct from BBS1-12",
                "definition": "BBS1-12 renal phenotype is dominated by structural anomalies (horseshoe kidney, calyceal clubbing, cystic dysplasia) — reflecting impaired cilia-mediated tubulogenesis during development (BBSome/BCC failure in differentiating tubular epithelium). BBS13 renal phenotype is dominated by nephronophthisis (tubular atrophy, interstitial fibrosis, corticomedullary cysts, concentrating defect) — reflecting ongoing TZ gate failure in mature tubular epithelial cilia (MKS1-scaffolded TZ gate is required for correct fluid-flow mechanosensing in differentiated tubules). This NPHP pattern in BBS13 is mechanistically identical to NPHP1/NPHP4 LOF but occurs because MKS1 bridges to NPHP4 — MKS1 loss secondarily disrupts NPHP4 TZ localisation, producing NPHP-like tubular damage."
            },
            {
                "term": "Liver Fibrosis in BBS13 — Meckel Overlap Feature (~12%)",
                "definition": "Ductal plate malformation (DPM) / congenital hepatic fibrosis occurs in ~12% of BBS13 patients — a feature seen in Meckel-Gruber syndrome (where MKS1 null/null causes full DPM) that bleeds into the BBS13 phenotype when allele severity is intermediate. DPM reflects the role of MKS1-dependent TZ function in cholangiocyte (bile duct) cilia during hepatic morphogenesis — without correct TZ gate function in ductal plate progenitors, bile duct remodelling fails and periportal fibrosis develops. This feature is NOT seen in BBS1-12 (BBSome/BCC subunit LOF does not produce hepatic ductal plate malformation), making liver fibrosis in BBS a specific diagnostic pointer toward MKS1 or another TZ-module gene."
            },
            {
                "term": "LepR Mismigration via TZ Gate Failure in BBS13 — Indirect Mechanism",
                "definition": "In BBS1-12, LepR mis-trafficking occurs because the BBSome (or BCC) fails → no complete BBSome → LepR cannot be retrieved from hypothalamic neuronal cilia by IFT-retrograde BBSome-cargo mechanism → LepR constitutively retained in cilia → leptin resistance → obesity. In BBS13, the BBSome assembles normally and reaches the TZ. However, the leaky TZ gate allows abnormal LepR distribution across the TZ boundary — LepR can migrate out of the ciliary compartment at inappropriate rates OR accumulate abnormally within it (depending on TZ permeability direction for LepR). The net effect is disrupted hypothalamic leptin signalling → hyperphagia and obesity — phenotypically identical to BBS1-12 obesity but mechanistically distinct (TZ gate permeability failure, not retrograde IFT cargo retrieval failure). GLP1RA bypass of LepR applies regardless of mechanism."
            },
        ],
    }
