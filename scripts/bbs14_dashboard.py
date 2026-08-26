"""
BBS14 Bardet-Biedl Syndrome Type 14 — CEP290 (Centrosomal Protein 290) Dashboard
===================================================================================
Primary Gene : CEP290 (*610142) — Centrosomal Protein of 290 kDa; 2479 aa; Chr 12q21.32.
               CEP290 is an extremely large coiled-coil scaffolding protein at the
               ciliary transition zone (TZ). It is the SECOND BBS gene (after MKS1/BBS13)
               that is NOT a BBSome subunit or BCC chaperonin component.
               CEP290 localises to the TZ Y-links and the ciliary base, where it
               directly contacts NPHP5 (IQCB1), NPHP6 (itself a historical alias),
               KIF3A, and the NPHP-MKS-JBTS module scaffold.
               Domain organisation (2479 aa):
                 N-terminal domain (aa 1–300): KIF3A-binding (motor docking);
                 Coiled-coil regions 1 (CC1, aa 300–700): microtubule association;
                 Coiled-coil 2 (CC2, aa 700–1100): NPHP5/IQCB1 interaction;
                 Coiled-coil 3 (CC3, aa 1100–1600): structural scaffold;
                 Coiled-coil 4/5 (CC4/5, aa 1600–2479): TZ Y-link anchoring;
                   CC4-5 contains the major BBS14/LCA10 allelic hotspot.
               CEP290 IS REQUIRED for:
                 1. Y-link formation between outer doublet microtubules and the
                    ciliary membrane (structural function, like MKS1 but distinct
                    position — CEP290 is MORE DISTAL at the TZ than MKS1).
                 2. Ciliary entry gate function — limiting free diffusion of
                    non-ciliary membrane proteins into the ciliary compartment.
                 3. Photoreceptor connecting cilium (CC) integrity — hence the
                    LCA10 phenotype with null alleles.
               CEP290 LOF IF fingerprint — UNIQUE, distinguishes BBS14 from BBS13:
                 BBSome subunits NORMAL (BBS2 NORMAL · BBS4 NORMAL · BBS8 NORMAL ·
                 BBS9 NORMAL) — same as BBS13.
                 MKKS/BBS6 NORMAL — BCC intact — same as BBS13.
                 MKS1 NORMAL — MKS sub-module INTACT — DISTINGUISHES BBS14 from BBS13.
                 CEP290 ABSENT at ciliary base.
                 NPHP5/IQCB1 deranged at TZ (reduced — CEP290 docking partner gone).
               ALLELE-CLASS DETERMINES DISEASE SEVERITY (same concept as BBS13):
                 Null/null (truncating/truncating) → Meckel-Gruber Syndrome Type 4
                   (MKS4, #611134) — usually lethal with encephalocele + ARPKD-like;
                 Hypomorphic/hypomorphic (e.g. two CC4/5 missense) → BBS14 (full BBS
                   phenotype) OR LCA10 (#611755 — if retina-selective enrichment);
                 Deep intronic/hypomorphic (IVS26+1655A>G + missense) → LCA10-BBS14
                   spectrum, variable;
                 Hypomorphic/truncating (one CC missense + one null) → BBS14 or JBTS5;
                 BBS14 therefore REQUIRES at least one hypomorphic allele in CEP290.
               KEY DISTINCTION: CEP290 is the MOST commonly mutated gene across ALL
               ciliopathies — the same gene causes LCA10, JBTS5, NPHP6, MKS4, SLSN6,
               and BBS14 depending on allele class and modifier context. As ISOLATED
               BBS14 it is RARE (<1–2% of all BBS); the gene is far more commonly
               encountered in the LCA10 and JBTS contexts.
Disease        : Bardet-Biedl Syndrome Type 14 (#209900); allelic: LCA10 (#611755);
                 JBTS5 (#610188); MKS4 (#611134); NPHP6 (#610188); SLSN6 (#610188)
Frequency      : <1–2% of all BBS (very rare); CEP290 is far more common in LCA/JBTS
Inheritance    : Autosomal recessive (AR); biallelic LOF (at least 1 hypomorphic)
Tri-allelic    : ~2% of BBS14 families (very rare)
Cohort         : 40 patients; educational/synthetic; seed 359
Endpoints      : /api/bbs14/overview | /api/bbs14/breakdown | /api/bbs14/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 359
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("European",      0.35),   # French, Dutch, British, North American European — most LCA10/BBS14 reports
        ("MENA (other)",  0.25),   # Saudi Arabia, Iran, Turkey, Jordan — consanguineous
        ("South Asian",   0.18),   # India, Pakistan — consanguineous
        ("North African", 0.12),   # Tunisia, Morocco — consanguineous; CEP290 hotspot enriched
        ("East Asian",    0.05),
        ("Other",         0.05),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "European":      ["p.Ile1925Asn (c.5774T>A)", "IVS26+1655A>G (deep intronic, LCA10 overlap)", "p.Trp802Ter (c.2405G>A)"],
        "MENA (other)":  ["p.Ser1521Trp (c.4562C>G)", "p.Arg1933Ter (c.5797C>T)", "p.Ile1925Asn (c.5774T>A)"],
        "South Asian":   ["p.Lys1575Met (c.4724A>T)", "p.Ser1521Trp (c.4562C>G)", "p.Arg1933Ter (c.5797C>T)"],
        "North African": ["p.Ser1521Trp (c.4562C>G)", "p.Lys1575Met (c.4724A>T)", "p.Trp802Ter (c.2405G>A)"],
        "East Asian":    ["p.Arg1933Ter (c.5797C>T)", "p.Ile1925Asn (c.5774T>A)", "p.Lys1575Met (c.4724A>T)"],
        "Other":         ["p.Trp802Ter (c.2405G>A)", "p.Arg1933Ter (c.5797C>T)"],
    }

    allele_classes = [
        "Hypomorphic/Hypomorphic",        # BBS14 — two CC4/5 missense, residual CEP290 function
        "Hypomorphic/Truncating",         # BBS14 or JBTS5 — mixed
        "Deep-Intronic/Hypomorphic",      # IVS26+1655A>G compound — LCA10-BBS14 overlap
        "Splice/Hypomorphic",
        "Missense/Missense (mild)",       # two mild missense — moderate BBS14
    ]
    allele_weights = [0.32, 0.30, 0.20, 0.12, 0.06]

    # BBS14-specific retinal stages — same rod-first BBS pattern
    # Note: IVS26+1655A>G compound-het families may show earlier / more severe onset
    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.18, 0.28, 0.30, 0.24]

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.30, 0.10, 0.07, 0.03, 0.50]   # ~50% polydactyly (BBS14 — lower than BBS1-12)

    # BBS14 renal: NPHP dominant (like BBS13 — TZ module); cystic rare; no cystic-BBS1-12 pattern
    renal_types = ["Nephronophthisis (tubular atrophy/fibrosis)", "Structural anomaly (horseshoe/duplex)",
                   "Cystic dysplasia (MKS4 overlap)", "ESRD", "None"]
    renal_w = [0.26, 0.07, 0.08, 0.10, 0.49]   # ~51% renal (NPHP dominant, ~55% literature rate)

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.18, 0.40, 0.27, 0.15]

    misdiag = ["Isolated Leber Congenital Amaurosis (LCA10 — same gene)", "Joubert Syndrome (JBTS5 overlap)",
               "Isolated Nephronophthisis (NPHP6)", "Alström Syndrome (cone-rod confusion)",
               "No initial misdiagnosis"]
    mis_w   = [0.28, 0.14, 0.08, 0.06, 0.44]

    presentation_ages = ["Birth (polydactyly noted)", "Infancy (nystagmus/photophobia)",
                         "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"]
    pres_w  = [0.20, 0.18, 0.28, 0.22, 0.12]

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

        # BBS14: ~50% polydactyly, ~65% obesity, ~38% cognitive LD,
        #        ~48% hypogonadism, ~42% anosmia, ~51% renal (NPHP dominant),
        #        ~4% CHD, ~10% JBTS5 overlap, ~8% LCA10 overlap spectrum
        polydactyly   = poly_type != "None"
        obesity       = rng.random() < 0.65
        cognitive_ld  = rng.random() < 0.38
        hypogonadism  = rng.random() < 0.48
        anosmia       = rng.random() < 0.42
        renal_any     = renal_type != "None"
        chd           = rng.random() < 0.04
        # JBTS5 overlap: allele-class dependent (hypomorphic/truncating, deep-intronic/hypomorphic)
        jbts_overlap  = allele_cl in ("Hypomorphic/Truncating", "Deep-Intronic/Hypomorphic") and rng.random() < 0.24
        # LCA10 spectrum: some families present as LCA10 (severe early retinal) before BBS features evident
        lca10_spectrum = allele_cl in ("Deep-Intronic/Hypomorphic", "Hypomorphic/Hypomorphic") and rng.random() < 0.18
        ret_end       = ret_stage == "Legal blindness / end-stage"
        tri_bbs       = rng.random() < 0.02    # ~2% tri-allelic (very rare)
        mis_dx        = mis_d != "No initial misdiagnosis"
        esrd          = renal_type == "ESRD"
        consang       = eth in ("MENA (other)", "South Asian", "North African") and rng.random() < 0.68

        patients.append({
            "id":                   f"BBS14-{i+1:03d}",
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
            "jbts_overlap":         jbts_overlap,
            "lca10_spectrum":       lca10_spectrum,
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
    jbts_n         = sum(1 for p in c if p["jbts_overlap"])
    lca10_n        = sum(1 for p in c if p["lca10_spectrum"])
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
        "gene":     "CEP290",
        "omim_gene": "*610142",
        "omim_disease": "#209900 (BBS14)",
        "locus":    "12q21.32",
        "protein":  "2479 aa — Centrosomal protein 290 kDa; large coiled-coil TZ scaffold; CC1–CC5 domain architecture",
        "mechanism": "Transition zone (TZ) Y-link scaffolding — NPHP-MKS-JBTS module (distal TZ position); CEP290 anchors Y-links; TZ gate leaky → GPCR mismigration. MKS1 sub-module INTACT (distinguishes BBS14 from BBS13). CEP290 absent, NPHP5/IQCB1 deranged.",
        "bbs_frequency": "<1–2% of all BBS (very rare; CEP290 far more common in LCA10/JBTS5)",
        "inheritance": "Autosomal recessive; biallelic LOF (≥1 hypomorphic allele required for BBS phenotype; null/null → MKS4 lethal)",
        "triallelic_pct": 2,
        "allele_class_note": "Null/null → MKS4 (lethal); hypomorphic/hypomorphic → BBS14 or LCA10; hypomorphic/truncating → BBS14 or JBTS5; deep-intronic (IVS26+1655A>G)/hypomorphic → LCA10-BBS14 spectrum",
        "if_fingerprint": "BBSome subunits NORMAL (BBS2·BBS4·BBS8·BBS9 all present); MKKS/BCC NORMAL; MKS1 NORMAL (KEY: MKS sub-module INTACT — distinguishes from BBS13); CEP290 ABSENT; NPHP5/IQCB1 deranged at TZ",
        "key_counts": {
            "polydactyly_n":    poly_n,    "polydactyly_pct":    _pct(poly_n),
            "obesity_n":        obesity_n,  "obesity_pct":        _pct(obesity_n),
            "cognitive_n":      cognitive_n, "cognitive_pct":     _pct(cognitive_n),
            "renal_any_n":      renal_any_n, "renal_any_pct":    _pct(renal_any_n),
            "nphp_n":           nphp_n,     "nphp_pct":          _pct(nphp_n),
            "hypogonadism_n":   hypo_n,     "hypogonadism_pct":  _pct(hypo_n),
            "anosmia_n":        anosmia_n,  "anosmia_pct":       _pct(anosmia_n),
            "chd_n":            chd_n,      "chd_pct":           _pct(chd_n),
            "jbts_overlap_n":   jbts_n,     "jbts_overlap_pct":  _pct(jbts_n),
            "lca10_spectrum_n": lca10_n,    "lca10_spectrum_pct": _pct(lca10_n),
            "retinal_endstage_n": ret_end_n, "retinal_endstage_pct": _pct(ret_end_n),
            "triallelic_n":     tri_n,      "triallelic_pct":    _pct(tri_n),
            "misdiagnosis_n":   mis_n,      "misdiagnosis_pct":  _pct(mis_n),
            "esrd_n":           esrd_n,     "esrd_pct":          _pct(esrd_n),
            "consanguinity_n":  consang_n,  "consanguinity_pct": _pct(consang_n),
        },
        "systemic_burden": [
            ("Rod-Cone Dystrophy (rod-first)",       n,          _pct(n)),
            ("Obesity (TZ/LepR pathway)",             obesity_n,  _pct(obesity_n)),
            ("Polydactyly (post-axial)",               poly_n,    _pct(poly_n)),
            ("Renal anomaly (NPHP dominant)",          renal_any_n, _pct(renal_any_n)),
            ("Nephronophthisis specifically",          nphp_n,    _pct(nphp_n)),
            ("Hypogonadism",                           hypo_n,    _pct(hypo_n)),
            ("Cognitive/LD",                           cognitive_n, _pct(cognitive_n)),
            ("Anosmia/Hyposmia",                      anosmia_n, _pct(anosmia_n)),
            ("Joubert overlap (JBTS5)",                jbts_n,    _pct(jbts_n)),
            ("LCA10 overlap spectrum",                 lca10_n,   _pct(lca10_n)),
            ("CHD",                                    chd_n,     _pct(chd_n)),
        ],
        "retinal_stage_distribution": {k: v for k, v in ret_stages.items()},
        "dx_age_distribution":        {k: v for k, v in age_dist.items()},
    }


def get_breakdown() -> Dict[str, Any]:
    c = _COHORT
    n = _N

    systemic = [
        ("Rod-Cone Dystrophy (rod-first)",       n,                                               _pct(n)),
        ("Obesity (TZ/LepR pathway)",             sum(1 for p in c if p["obesity"]),              _pct(sum(1 for p in c if p["obesity"]))),
        ("Polydactyly (post-axial)",               sum(1 for p in c if p["polydactyly"]),         _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Renal anomaly (any)",                    sum(1 for p in c if p["renal_any"]),           _pct(sum(1 for p in c if p["renal_any"]))),
        ("Nephronophthisis specifically",          sum(1 for p in c if p["renal_type"] == "Nephronophthisis (tubular atrophy/fibrosis)"), _pct(sum(1 for p in c if p["renal_type"] == "Nephronophthisis (tubular atrophy/fibrosis)"))),
        ("ESRD",                                   sum(1 for p in c if p["esrd"]),                _pct(sum(1 for p in c if p["esrd"]))),
        ("Hypogonadism",                           sum(1 for p in c if p["hypogonadism"]),        _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Cognitive/LD",                           sum(1 for p in c if p["cognitive_ld"]),        _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("Anosmia/Hyposmia",                      sum(1 for p in c if p["anosmia"]),              _pct(sum(1 for p in c if p["anosmia"]))),
        ("Joubert overlap (JBTS5)",                sum(1 for p in c if p["jbts_overlap"]),        _pct(sum(1 for p in c if p["jbts_overlap"]))),
        ("LCA10 overlap spectrum",                 sum(1 for p in c if p["lca10_spectrum"]),      _pct(sum(1 for p in c if p["lca10_spectrum"]))),
        ("CHD",                                    sum(1 for p in c if p["chd"]),                 _pct(sum(1 for p in c if p["chd"]))),
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
                      "Cystic dysplasia (MKS4 overlap)", "ESRD", "None"]
        ],
        "polydactyly_distribution": [
            {"label": p, "n": poly_counts.get(p, 0)}
            for p in ["Post-axial hands+feet", "Post-axial hands only",
                      "Post-axial feet only", "Pre-axial (rare)", "None"]
        ],
        "presentation_distribution": [
            {"label": p, "n": pres_counts.get(p, 0)}
            for p in ["Birth (polydactyly noted)", "Infancy (nystagmus/photophobia)",
                      "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"]
        ],
        "misdiagnosis_distribution": [
            {"label": m, "n": mis_counts.get(m, 0)}
            for m in ["Isolated Leber Congenital Amaurosis (LCA10 — same gene)", "Joubert Syndrome (JBTS5 overlap)",
                      "Isolated Nephronophthisis (NPHP6)", "Alström Syndrome (cone-rod confusion)",
                      "No initial misdiagnosis"]
        ],
        "top_variants": [{"variant": v, "n": cnt} for v, cnt in top_variants],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":             "CEP290 (Centrosomal Protein of 290 kDa); Chr 12q21.32",
            "OMIM_Gene":        "*610142",
            "Chromosome":       "12q21.32",
            "Protein":          "2479 aa — N-terminal KIF3A-binding (aa 1–300) · CC1 microtubule association (aa 300–700) · CC2 NPHP5/IQCB1 interaction (aa 700–1100) · CC3 structural scaffold (aa 1100–1600) · CC4/CC5 TZ Y-link anchoring (aa 1600–2479; allelic hotspot for BBS14 and LCA10)",
            "Domains":          "aa 1–300: N-terminal domain (KIF3A motor docking; microtubule anchoring at centrosome/TZ base) · aa 300–700: CC1 coiled-coil (microtubule association; structural scaffold) · aa 700–1100: CC2 coiled-coil (NPHP5/IQCB1 direct contact — loss of IQCB1 at TZ is IF fingerprint of CEP290 LOF) · aa 1100–1600: CC3 coiled-coil (structural scaffold; interdomain packing) · aa 1600–2479: CC4/CC5 coiled-coil (TZ Y-link anchoring; ciliary membrane-to-axoneme tether; major BBS14/LCA10 allelic hotspot)",
            "Module":           "NPHP-MKS-JBTS transition zone (TZ) module — CEP290 is at the DISTAL TZ position (Y-link anchoring); MKS1 sub-module remains INTACT in BBS14 (key diagnostic distinction from BBS13)",
            "Inheritance":      "Autosomal Recessive; biallelic LOF (at least 1 hypomorphic allele required for BBS phenotype; null/null → MKS4 lethal)",
            "Frequency_BBS":    "<1–2% of all BBS; very rare as isolated BBS14; CEP290 mutations are the MOST COMMON cause of LCA10 (~25% of LCA) and JBTS5 (~8% of JBTS) — BBS14 is the rarest clinical tier",
            "Allele_spectrum":  "Null/null (truncating) → MKS4 (#611134, lethal); hypomorphic/hypomorphic (CC4/5 missense) → BBS14 or LCA10; deep-intronic (IVS26+1655A>G) + hypomorphic → LCA10-BBS14 spectrum; hypomorphic/truncating → BBS14 or JBTS5; BBS14 requires ≥1 hypomorphic allele with residual CEP290 scaffold function",
            "Triallelic_BBS":   "~2% (very rare; compatible third-allele partners not systematically defined)",
            "Allelic_diseases": "BBS14 (#209900) · LCA10 (#611755) · JBTS5 (#610188) · MKS4 (#611134) · NPHP6 (#610188) · SLSN6 (#610188) — SAME GENE, allele class + tissue context determines severity",
            "CHD_penetrance":   "~4%",
            "Population_note":  "Pan-ethnic; IVS26+1655A>G deep intronic variant especially enriched in European/North American LCA10 families; BBS14 phenotype families broadly distributed across MENA, South Asian, European, North African",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 14 (BBS14) — #209900",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS14 subtype <1–2% (~<2 in 10,000,000) — very rare",
            "Cardinal_features": "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR/TZ pathway) · Hypogonadism · Renal anomalies (NPHP dominant) · Cognitive/LD · Anosmia",
            "BBS14_unique":      "SECOND BBS gene (after MKS1/BBS13) not in BBSome or BCC. CEP290 is the MOST commonly mutated ciliopathy gene overall (LCA10, JBTS5, NPHP6, MKS4) but RARE as isolated BBS14. KEY IF DISTINCTION FROM BBS13: MKS1 NORMAL in BBS14 (MKS sub-module intact); only CEP290 absent + NPHP5/IQCB1 deranged. No liver fibrosis (unlike BBS13). JBTS5 overlap ~10%.",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; same pattern as BBS1-13; LCA10-spectrum onset can be earlier in deep-intronic/hypomorphic allele class)",
            "Obesity_mechanism": "TZ gate leaky → hypothalamic GPCR (LepR) mismigration via TZ permeability failure (CEP290 absent — Y-link disrupted → TZ gate defective); BBSome itself intact; indirect mechanism (same endpoint as BBS13/MKS1, different TZ position)",
            "Renal":             "Nephronophthisis (tubular atrophy, interstitial fibrosis, concentrating defect) DOMINANT — same pattern as BBS13; distinct from BBS1-12 cystic/structural phenotype; ESRD risk ~10%; cystic dysplasia (MKS4 overlap) ~8%",
            "Liver":             "No liver fibrosis — DISTINGUISHES BBS14 from BBS13 (MKS1); CEP290 LOF does not affect hepatic ductal plate; standard monitoring only",
            "JBTS_overlap":      "~10% of BBS14 cohort develops JBTS5 overlap (molar tooth sign; cerebellar vermis hypoplasia; oculomotor apraxia) — allele class dependent (hypomorphic/truncating allele class → greater TZ failure → JBTS cerebellar phenotype)",
            "LCA10_spectrum":    "~8%: BBS14 families with two hypomorphic CC4/5 alleles (or IVS26+1655A>G compound) may present first as LCA10 (severe early-onset rod+cone loss with nystagmus in infancy) — full BBS features emerge later; gene panel critical",
            "CHD":               "~4%",
            "Diagnosis":         "Full BBS gene panel (minimum 20–24 genes including CEP290); IF: BBSome panel NORMAL + MKS1 NORMAL + CEP290 ABSENT + IQCB1/NPHP5 deranged at TZ — unique pattern distinguishing BBS14 from BBS13 (MKS1 normal) and BBS1-12 (TZ normal); MRI brain if JBTS5 overlap suspected",
            "Treatment":         "Symptomatic; GLP1RA for obesity; renal surveillance (NPHP pattern — ACE-i/ARB; nephrotoxin avoidance; transplant planning for ESRD); no specific liver monitoring required (unlike BBS13); LCA10-spectrum patients may respond to antisense oligonucleotide (QR-110/sepofarsen — targets IVS26+1655A>G) if that allele is present",
            "Prognosis":         "Progressive visual loss as per BBS1-13; renal ESRD risk ~10% (NPHP pattern); JBTS5 overlap may add cerebellar/neurological burden; near-normal survival with surveillance; IVS26+1655A>G-compound patients may be eligible for emerging CEP290-targeted therapy",
        },
        "key_variants": [
            {
                "variant":     "p.Ile1925Asn (c.5774T>A)",
                "domain":      "CC4 coiled-coil domain — Y-link anchoring region (Ile1925 stabilises the CC4 leucine-zipper hydrophobic core at the TZ membrane-to-axoneme tether interface; disruption weakens Y-link anchoring without abolishing CEP290 localisation)",
                "consequence": "Hypomorphic allele; partial CEP290 scaffold function retained; TZ Y-link weakened but not abolished; TZ gate leaky → GPCR mismigration → BBS14 phenotype; BBSome intact; MKS1 intact; NPHP5/IQCB1 mildly reduced at TZ; European enrichment — BBS14-specific (not commonly reported in LCA10 families)",
                "ethnicity":   "European enrichment (France, Netherlands, UK); also reported in North American European-descent families"
            },
            {
                "variant":     "p.Ser1521Trp (c.4562C>G)",
                "domain":      "CC3 coiled-coil domain — structural scaffold region (Ser1521 is a coiled-coil register-stabilising residue; Trp substitution introduces a bulky hydrophobic side chain disrupting the coiled-coil register and interdomain packing)",
                "consequence": "Hypomorphic allele; CC3 coiled-coil register disrupted → CEP290 misfolded at CC3-CC4 junction; partial loss of CC4-CC5 Y-link function; TZ gate defective → GPCR mismigration; BBS14 phenotype; when compound-heterozygous with truncating allele → BBS14 or JBTS5; MENA enrichment",
                "ethnicity":   "MENA enrichment (Saudi Arabia, Iran, Jordan); also North African"
            },
            {
                "variant":     "p.Lys1575Met (c.4724A>T)",
                "domain":      "CC3-CC4 junction — hinge region (Lys1575 is a solvent-exposed coiled-coil surface residue; loss of positive charge disrupts electrostatic interaction at the CC3-CC4 inter-domain linker)",
                "consequence": "Hypomorphic allele; moderate destabilisation of CC4 domain positioning relative to CC3; TZ Y-link anchoring partially impaired; BBS14 phenotype with incomplete penetrance for NPHP; South Asian enrichment; milder phenotype than Ser1521Trp-compound families",
                "ethnicity":   "South Asian (India, Pakistan); also North African"
            },
            {
                "variant":     "IVS26+1655A>G (deep intronic, c.5668+1655A>G)",
                "domain":      "Intron 26 — 1655 bp downstream of exon 26 splice donor; activates a cryptic pseudo-exon (PE; 128 bp insertion) → introduces premature stop codon within the CC4 domain (PS p.Cys1988Leufs); pseudo-exon inclusion ~50-60% of transcripts",
                "consequence": "Hypomorphic partial loss-of-function: residual correctly spliced transcript (~40-50%) preserves partial CEP290 function; most commonly causes LCA10 (retina-enriched expression of pseudo-exon-bearing transcript); in compound with a mild missense (e.g. Ile1925Asn) → BBS14-spectrum; in compound with truncating → JBTS5; highly amenable to antisense oligonucleotide (sepofarsen/QR-110) splice correction",
                "ethnicity":   "European and North American (British, Dutch, French origin families); MOST COMMON LCA10 allele worldwide"
            },
            {
                "variant":     "p.Arg1933Ter (c.5797C>T)",
                "domain":      "CC4/5 junction — truncating null; stops at aa 1933; removes entire CC5 domain (TZ Y-link anchoring C-terminal anchor)",
                "consequence": "Null allele; complete loss of CC5 Y-link anchoring; when homozygous → MKS4 (Meckel-Gruber variant, usually lethal); when compound heterozygous with hypomorphic (e.g. Ile1925Asn or Ser1521Trp) → BBS14 or JBTS5; when compound with IVS26+1655A>G → JBTS5/BBS14 overlap; pan-ethnic truncating allele",
                "ethnicity":   "Pan-ethnic; most common truncating CEP290 allele in non-consanguineous BBS14/JBTS5 families"
            },
            {
                "variant":     "p.Trp802Ter (c.2405G>A)",
                "domain":      "CC2 domain — truncating null; stops at aa 802; removes CC2 NPHP5/IQCB1 interaction domain + all downstream CC3/CC4/CC5",
                "consequence": "Null allele; complete loss of NPHP5/IQCB1 docking and all downstream TZ scaffold function; compound heterozygous with hypomorphic → BBS14 (when hypomorphic allele retains sufficient CC2-CC3 function) or MKS4/JBTS5 (more severe); more severe truncating allele than Arg1933Ter; pan-ethnic",
                "ethnicity":   "Pan-ethnic; reported in European, MENA, South Asian families"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG rod-first) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20–24 genes including CEP290); MLPA/CNV for large CEP290 deletions; deep-intronic variant testing (IVS26+1655A>G if LCA10-spectrum presentation) — standard exon sequencing will MISS this variant",
            "3. Allele classification: classify both CEP290 alleles as hypomorphic vs truncating; null/null → review for MKS4 (lethal — would not present with BBS phenotype unless highly unusual); identify IVS26+1655A>G if compound with missense (LCA10-BBS14 spectrum)",
            "4. IF panel: BBSome (BBS2, BBS4, BBS8, BBS9) NORMAL; MKKS/BCC (MKKS, BBS10, BBS12) NORMAL; MKS1 NORMAL (CRITICAL distinction from BBS13: MKS sub-module intact in BBS14); CEP290 ABSENT at ciliary base; NPHP5/IQCB1 REDUCED at TZ (CEP290 docking partner lost)",
            "5. MRI brain: if JBTS5 overlap suspected (allele class: hypomorphic/truncating) → axial T1/T2 for molar tooth sign; cerebellar vermis hypoplasia; oculomotor apraxia assessment by paediatric neurology",
            "6. Renal: NPHP pattern dominant — renal ultrasound (echogenic cortex, corticomedullary differentiation loss); annual GFR; urine concentrating ability; avoid nephrotoxins; ESRD risk ~10%; no hepatorenal fibrosis (unlike BBS13)",
            "7. Ophthalmology: ERG (rod-first, scotopic extinguished) + Goldmann VF + OCT retinal thickness; LCA10-spectrum patients: full-field ERG in infancy (cone-rod or rod-cone pattern); sepofarsen (QR-110) eligibility assessment if IVS26+1655A>G confirmed",
            "8. Liver: standard surveillance only — NO liver fibrosis/CHF expected in BBS14 (unlike BBS13/MKS1); liver USS at diagnosis to confirm no unexpected hepatic anomaly; annual LFTs not required unless another indication",
            "9. Obesity: GLP1RA applicable (TZ gate leaky → LepR mismigration — same downstream endpoint as BBSome LOF, indirect mechanism); bariatric surgery in refractory cases after ophthalmology risk assessment",
            "10. Genetic counselling: AR inheritance; sibling recurrence 25%; allele-class counselling critical — if both parents carry null alleles, prenatal testing for MKS4 is essential; sepofarsen eligibility if IVS26+1655A>G identified",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; photosensitising drug avoidance; ERG surveillance annually; CEP290-specific: sepofarsen (QR-110) antisense oligonucleotide suppresses IVS26+1655A>G pseudo-exon inclusion (intravitreal injection; Phase 2/3 data 2026) — ONLY applicable if IVS26+1655A>G confirmed; AAV-CEP290 (e.g. EDIT-101 via CRISPR or viral delivery) investigational",
            "2. Obesity: GLP-1 receptor agonists (semaglutide) — LepR mismigration via TZ permeability failure (CEP290 absent → Y-link broken → TZ gate leaky → indirect LepR mislocalisation); same clinical result as BBSome LOF but distinct mechanism; bariatric surgery in refractory cases",
            "3. Renal (NPHP pattern): ACE inhibitor/ARB for CKD slowing; strict nephrotoxin avoidance (NSAIDs, IV contrast, aminoglycosides); annual GFR, blood pressure, urine PCR; renal transplant planning early for ESRD (~10% risk); transplant curative — renal disease does not recur in transplanted organ",
            "4. Liver: No special liver monitoring required (unlike BBS13/MKS1) — this is a BBS14-specific note; standard LFTs as part of routine health monitoring only",
            "5. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling (usually infertile)",
            "6. Anosmia: safety counselling (gas leaks, food spoilage); smoke detectors mandatory; support for dietary impact",
            "7. Cognitive/LD: early intervention; IEP; neuropsychological assessment; JBTS5 overlap patients add cerebellar ataxia management and physiotherapy needs",
            "8. CEP290-targeted therapies in development (2026): sepofarsen (IVS26+1655A>G-specific); CRISPR-based CEP290 exon editing; AAV5-CEP290 gene augmentation (preclinical) — eligibility for clinical trials should be assessed at diagnosis",
        ],
    }
