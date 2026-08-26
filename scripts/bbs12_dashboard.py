"""
BBS12 Bardet-Biedl Syndrome Type 12 — BBS12 (BCC γ-Ring Structural Subunit) Dashboard
========================================================================================
Primary Gene : BBS12 (*610188) — Bardet-Biedl Syndrome 12 protein; 729 aa; Chr 4q27.
               BBS12 is the STRUCTURAL γ-RING SUBUNIT of the BBSome Chaperonin
               Complex (BCC). The BCC is a group-II chaperonin-like heterotrimer
               (MKKS/BBS6 · BBS10 · BBS12) that co-folds the WD40 β-propellers
               of BBS2 and BBS7 before BBSome assembly can begin (Step 0).
               Domain organisation (729 aa):
                 N-terminal/apical domain (aa 1–180): contacts BBS10 apical surface;
                   Arg140 bridges BBS10-BBS12 apical interface;
                   BBS12 apical contacts BBS10 β-ring at aa 80–160 of BBS10;
                 Intermediate domain (aa 181–420): BCC inter-subunit contacts
                   (MKKS equatorial surface and BBS10 intermediate helix);
                   Leu287 is the key hydrophobic residue in the BCC inter-subunit
                   helix pack;
                 Equatorial domain (aa 421–620): primary MKKS α-subunit equatorial
                   docking surface; Arg445 contacts MKKS equatorial loop;
                   Thr501 in the ATP-analog binding pocket (structural; no ATPase);
                 C-terminal scaffold (aa 621–729): stabilises MKKS-BBS12 equatorial
                   contact; Glu563Ter truncating null removes scaffold → MKKS
                   docking lost → BCC collapses.
               BBS12 LOF → BCC trimer structurally incomplete from γ-ring side →
               BCC cannot co-fold BBS2/BBS7 WD40 barrels → BBS2 and BBS7 misfold
               → BBSome core module (Step 0) fails before assembly → no complete
               BBSome octamer → GPCR cargoes (LepR, SSTR3, D1R, Smo) mis-trafficked
               in cilia → full BBS.
               BBS12 is the THIRD SUBUNIT OF THE BCC TRIMER.  Its IF fingerprint:
                 BBS12 ABSENT + MKKS REDUCED (not absent) + BBS10 REDUCED + BBS2 ABSENT.
               This is ALMOST IDENTICAL to the BBS10 IF fingerprint (BBS10 ABSENT +
               MKKS REDUCED + BBS12 REDUCED + BBS2 ABSENT).  The ABSENT subunit on
               IF identifies which BCC subunit is mutated; full BCC-panel IF
               (MKKS + BBS10 + BBS12 + BBS2) is essential to discriminate BBS6,
               BBS10, and BBS12 LOF — gene panel is the definitive discriminator.
               BBS12 is the MOST COMMON THIRD ALLELE IN TRI-ALLELIC BBS10 FAMILIES
               (~8% of BBS10 biallelic families carry a BBS12 third allele).
               BBS12 is enriched in North African/Maghrebi and Middle Eastern
               consanguineous populations.
Disease        : Bardet-Biedl Syndrome (#209900) — BBS12 subtype
Frequency      : ~5–8% of all BBS; 4th–5th most common BBS gene
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~8% of BBS12 families (BBS10 most common third allele — BCC β-ring
                 equatorial docking partner; MKKS/BBS6 second — BCC α-partner)
Cohort         : 40 patients; educational/synthetic; seed 353
Endpoints      : /api/bbs12/overview | /api/bbs12/breakdown | /api/bbs12/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 353
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("North African", 0.38),   # Tunisia, Morocco, Algeria — highest; Stoetzel 2007 consanguineous
        ("MENA (other)",  0.22),   # Saudi Arabia, Iran, Lebanon consanguineous
        ("European",      0.22),   # France, Netherlands, UK
        ("South Asian",   0.12),
        ("East Asian",    0.04),
        ("Other",         0.02),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "North African": ["p.Arg140Cys (c.418C>T)", "p.Glu563Ter (c.1687G>T)", "p.Leu287Pro (c.860T>C)"],
        "MENA (other)":  ["p.Arg140Cys (c.418C>T)", "p.Arg445Trp (c.1333C>T)", "p.Glu563Ter (c.1687G>T)"],
        "European":      ["p.Arg445Trp (c.1333C>T)", "c.1063+5G>A (splice)", "p.Glu563Ter (c.1687G>T)"],
        "South Asian":   ["p.Leu287Pro (c.860T>C)", "p.Thr501Met (c.1502C>T)", "p.Glu563Ter (c.1687G>T)"],
        "East Asian":    ["p.Glu563Ter (c.1687G>T)", "p.Thr501Met (c.1502C>T)", "p.Arg140Cys (c.418C>T)"],
        "Other":         ["p.Glu563Ter (c.1687G>T)", "p.Arg445Trp (c.1333C>T)"],
    }

    allele_classes = ["Missense/Missense", "Missense/Truncating", "Truncating/Truncating",
                      "Splice/Missense", "Splice/Truncating"]
    allele_weights = [0.30, 0.36, 0.20, 0.09, 0.05]

    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.15, 0.32, 0.31, 0.22]

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.38, 0.14, 0.09, 0.04, 0.35]   # ~65% polydactyly

    renal_types = ["Structural anomaly (horseshoe/duplex)", "Cystic dysplasia",
                   "Tubulointerstitial nephropathy", "ESRD", "None"]
    renal_w = [0.13, 0.14, 0.08, 0.06, 0.59]

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.20, 0.46, 0.24, 0.10]

    misdiag = ["Isolated RP / Leber amaurosis", "Obesity syndrome (Prader-Willi screen)",
               "Alström Syndrome (cone-rod confusion)", "CHARGE/Joubert", "No initial misdiagnosis"]
    mis_w   = [0.28, 0.20, 0.10, 0.07, 0.35]

    presentation_ages = ["Birth (polydactyly noted)", "Infancy (nystagmus/strabismus)",
                         "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"]
    pres_w  = [0.28, 0.18, 0.26, 0.20, 0.08]

    patients = []
    for i in range(n):
        eth = rng.choices(eth_labels, weights=eth_weights, k=1)[0]
        v_pool = variants_by_eth[eth]
        v1 = rng.choice(v_pool)
        v2 = rng.choice(v_pool)
        while v2 == v1:
            v2 = rng.choice(v_pool)

        poly_type  = rng.choices(polydactyly_types, weights=poly_w, k=1)[0]
        renal_type = rng.choices(renal_types, weights=renal_w, k=1)[0]
        ret_stage  = rng.choices(retinal_stages, weights=ret_w, k=1)[0]
        allele_cl  = rng.choices(allele_classes, weights=allele_weights, k=1)[0]
        dx_ag      = rng.choices(dx_ages, weights=dx_w, k=1)[0]
        pres_age   = rng.choices(presentation_ages, weights=pres_w, k=1)[0]
        mis_d      = rng.choices(misdiag, weights=mis_w, k=1)[0]

        # BBS12: ~65% polydactyly, ~82% obesity, ~47% cognitive LD,
        #        ~63% hypogonadism, ~58% anosmia, ~41% renal, ~6% CHD
        polydactyly  = poly_type != "None"
        obesity      = rng.random() < 0.82
        cognitive_ld = rng.random() < 0.47
        hypogonadism = rng.random() < 0.63
        anosmia      = rng.random() < 0.58
        renal_any    = renal_type != "None"
        chd          = rng.random() < 0.06
        ret_end      = ret_stage == "Legal blindness / end-stage"
        tri_bbs      = rng.random() < 0.08   # ~8% tri-allelic (BBS10 most common third allele)
        mis_dx       = mis_d != "No initial misdiagnosis"
        esrd         = renal_type == "ESRD"
        consang      = eth in ("North African", "MENA (other)") and rng.random() < 0.60

        patients.append({
            "id":                   f"BBS12-{i+1:03d}",
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

    poly_n      = sum(1 for p in c if p["polydactyly"])
    obesity_n   = sum(1 for p in c if p["obesity"])
    cognitive_n = sum(1 for p in c if p["cognitive_ld"])
    renal_any_n = sum(1 for p in c if p["renal_any"])
    hypo_n      = sum(1 for p in c if p["hypogonadism"])
    anosmia_n   = sum(1 for p in c if p["anosmia"])
    chd_n       = sum(1 for p in c if p["chd"])
    ret_end_n   = sum(1 for p in c if p["retinal_endstage"])
    tri_n       = sum(1 for p in c if p["triallelic_bbs"])
    mis_n       = sum(1 for p in c if p["initial_misdiagnosis"])
    esrd_n      = sum(1 for p in c if p["esrd"])
    consang_n   = sum(1 for p in c if p["consanguinity"])

    ret_stages: Dict[str, int] = {}
    for p in c:
        s = p["retinal_stage"]
        ret_stages[s] = ret_stages.get(s, 0) + 1

    age_dist: Dict[str, int] = {}
    for p in c:
        g = p["dx_age_group"]
        age_dist[g] = age_dist.get(g, 0) + 1

    return {
        "cohort_n":      n,
        "seed":          _SEED,
        "gene":          "BBS12",
        "omim_gene":     "*610188",
        "chromosome":    "4q27",
        "protein_aa":    729,
        "disease_omim":  "#209900",
        "bbs_subtype":   "BBS12",
        "inheritance":   "Autosomal Recessive (biallelic LOF)",
        "frequency_pct_bbs": "~5–8%",
        "mechanism_class": "BCC γ-Ring Structural Subunit (MKKS-BBS10-BBS12 BCC trimer; co-folds BBS2/BBS7 WD40 barrels; BBSome Step 0)",
        "kpis": {
            "polydactyly_n":        poly_n,
            "polydactyly_pct":      _pct(poly_n),
            "obesity_n":            obesity_n,
            "obesity_pct":          _pct(obesity_n),
            "cognitive_n":          cognitive_n,
            "cognitive_pct":        _pct(cognitive_n),
            "renal_any_n":          renal_any_n,
            "renal_any_pct":        _pct(renal_any_n),
            "hypogonadism_n":       hypo_n,
            "hypogonadism_pct":     _pct(hypo_n),
            "anosmia_n":            anosmia_n,
            "anosmia_pct":          _pct(anosmia_n),
            "chd_n":                chd_n,
            "chd_pct":              _pct(chd_n),
            "retinal_endstage_n":   ret_end_n,
            "retinal_endstage_pct": _pct(ret_end_n),
            "triallelic_n":         tri_n,
            "triallelic_pct":       _pct(tri_n),
            "misdiagnosis_n":       mis_n,
            "misdiagnosis_pct":     _pct(mis_n),
            "esrd_n":               esrd_n,
            "consanguinity_n":      consang_n,
            "consanguinity_pct":    _pct(consang_n),
        },
        "mechanism": (
            "BBS12 (729 aa; Chr 4q27; OMIM *610188) is the STRUCTURAL γ-RING SUBUNIT of the "
            "BBSome Chaperonin Complex (BCC). The BCC is a group-II chaperonin-like heterotrimer "
            "(MKKS/BBS6 [α] · BBS10 [β] · BBS12 [γ]) that acts at the most upstream step of "
            "BBSome biogenesis (Step 0) by co-folding the WD40 β-propeller barrels of BBS2 and "
            "BBS7. Without proper BCC-mediated folding, BBS2 and BBS7 misfold and are degraded "
            "by the proteasome → BBS2-BBS7 obligate core dimer (Step 0) never forms → no complete "
            "BBSome octamer → GPCR cargoes (LepR, SSTR3, D1R, Smo) mis-trafficked in cilia → "
            "leptin resistance, rod-cone dystrophy, polydactyly, renal anomalies, hypogonadism, "
            "anosmia, cognitive impairment. BBS12 provides the γ-ring of the BCC: its N-terminal/"
            "apical domain contacts BBS10 apical surface (Arg140 bridges BBS10-BBS12 apical "
            "interface); its equatorial domain is the primary MKKS docking surface (Arg445 contacts "
            "MKKS equatorial loop). BBS12 LOF → BCC trimer cannot form from the γ-ring side → "
            "MKKS and BBS10 present but REDUCED (partial degradation without their γ-partner) → "
            "BCC activity lost → BBS2/BBS7 misfold → BBSome Step 0 fails."
        ),
        "key_distinction": (
            "BBS12 LOF IF fingerprint: BBS12 ABSENT + MKKS REDUCED (not absent) + BBS10 REDUCED "
            "+ BBS2 ABSENT. This is ALMOST IDENTICAL to BBS10 LOF (BBS10 ABSENT + MKKS REDUCED + "
            "BBS12 REDUCED + BBS2 ABSENT) — the ABSENT subunit on IF is the discriminator. "
            "BBS6/MKKS LOF: MKKS ABSENT (MKKS is the mutated α-subunit); BBS10 REDUCED; "
            "BBS12 REDUCED; BBS2 ABSENT. Full BCC-panel IF (MKKS + BBS10 + BBS12) is mandatory "
            "to identify which BCC subunit is LOF; gene panel is the definitive discriminator. "
            "BBS12 is also the most common third allele in tri-allelic BBS10 families (~8%) — "
            "BBS12 is the direct equatorial docking partner of BBS10 β-ring, so a heterozygous "
            "BBS12 third allele further destabilises BCC in BBS10 biallelic patients. "
            "BBS12 accounts for ~5–8% of all BBS and is the 4th–5th most common BBS gene globally. "
            "North African (Maghrebi) consanguineous families have the highest BBS12 prevalence."
        ),
        "bbs_pathway_comparison": [
            {"gene": "BBS1",         "role": "BBSome cargo-recognition beta-propeller (M390R founder)",
             "function_class": "Structural BBSome Subunit", "omim": "*209901", "frequency_pct": 25},
            {"gene": "BBS2",         "role": "BBSome core dimer partner of BBS7 — WD40 scaffold",
             "function_class": "Structural BBSome Subunit (Core Dimer)", "omim": "*606151", "frequency_pct": 15},
            {"gene": "BBS3 (ARL6)",  "role": "GTPase recruits BBSome to ciliary membrane",
             "function_class": "External GTPase Recruiter", "omim": "*608845", "frequency_pct": 4},
            {"gene": "BBS4",         "role": "TPR scaffold tethers BBSome at PCM1 satellites",
             "function_class": "PCM1 Satellite Tether", "omim": "*600374", "frequency_pct": 3},
            {"gene": "BBS5",         "role": "PH domain anchors BBSome to ciliary PtdIns lipids",
             "function_class": "Ciliary Lipid Anchor", "omim": "*603650", "frequency_pct": 2},
            {"gene": "BBS6 / MKKS",  "role": "BCC α-subunit; Group-II chaperonin-like; co-folds BBS2/BBS7",
             "function_class": "BCC α-Subunit (Upstream Chaperonin)", "omim": "*604896", "frequency_pct": "5–8"},
            {"gene": "BBS7",         "role": "Obligate WD40 dimer partner of BBS2 — core module scaffold",
             "function_class": "Structural BBSome Subunit (Core Dimer)", "omim": "*607590", "frequency_pct": "2–4"},
            {"gene": "BBS8 / TTC8",  "role": "TPR solenoid — IFT-B docking arm; only BBSome-IFT-B bridge",
             "function_class": "IFT-B Docking Arm (Peripheral Module)", "omim": "*608132", "frequency_pct": "2–4"},
            {"gene": "BBS9 / PTHB1", "role": "Bridge subunit — spans core (BBS2-BBS7) and peripheral modules",
             "function_class": "BBSome Inter-Module Bridge", "omim": "*607968", "frequency_pct": 2},
            {"gene": "BBS10",        "role": "BCC β-ring structural subunit; primary BBS12 docking; apical BBS2 contact",
             "function_class": "BCC β-Ring Subunit (Upstream Chaperonin)", "omim": "*610148", "frequency_pct": 20},
            {"gene": "BBS12 [THIS]", "role": "BCC γ-ring structural subunit; primary MKKS equatorial docking; BBS10 apical contact",
             "function_class": "BCC γ-Ring Subunit (Upstream Chaperonin)", "omim": "*610188", "frequency_pct": "5–8"},
        ],
        "retinal_distribution": [
            {"label": s, "n": ret_stages.get(s, 0)}
            for s in ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
        ],
        "age_distribution": {
            "dx_0_4yr":   age_dist.get("0–4 yr", 0),
            "dx_5_11yr":  age_dist.get("5–11 yr", 0),
            "dx_12_17yr": age_dist.get("12–17 yr", 0),
            "dx_18plus":  age_dist.get("18+ yr", 0),
        },
    }


def get_breakdown() -> Dict[str, Any]:
    c = _COHORT
    n = _N

    systemic = [
        ("Rod-cone dystrophy (rod-first; 100%)",              n,  100.0),
        ("Post-axial polydactyly",                            sum(1 for p in c if p["polydactyly"]),       _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Obesity (BMI ≥ 28)",                                sum(1 for p in c if p["obesity"]),            _pct(sum(1 for p in c if p["obesity"]))),
        ("Hypogonadism",                                      sum(1 for p in c if p["hypogonadism"]),       _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Anosmia/Hyposmia",                                  sum(1 for p in c if p["anosmia"]),            _pct(sum(1 for p in c if p["anosmia"]))),
        ("Cognitive/Learning Disability",                     sum(1 for p in c if p["cognitive_ld"]),       _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("Renal Anomaly (any)",                               sum(1 for p in c if p["renal_any"]),          _pct(sum(1 for p in c if p["renal_any"]))),
        ("Congenital Heart Disease (CHD)",                    sum(1 for p in c if p["chd"]),                _pct(sum(1 for p in c if p["chd"]))),
        ("ESRD",                                              sum(1 for p in c if p["esrd"]),               _pct(sum(1 for p in c if p["esrd"]))),
        ("Consanguinity (North African / MENA families)",     sum(1 for p in c if p["consanguinity"]),      _pct(sum(1 for p in c if p["consanguinity"]))),
        ("Tri-allelic BBS (BBS10 dominant 3rd allele)",       sum(1 for p in c if p["triallelic_bbs"]),    _pct(sum(1 for p in c if p["triallelic_bbs"]))),
        ("Initial misdiagnosis",                              sum(1 for p in c if p["initial_misdiagnosis"]), _pct(sum(1 for p in c if p["initial_misdiagnosis"]))),
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
            for r in ["Structural anomaly (horseshoe/duplex)", "Cystic dysplasia",
                      "Tubulointerstitial nephropathy", "ESRD", "None"]
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
                      "Alström Syndrome (cone-rod confusion)", "CHARGE/Joubert", "No initial misdiagnosis"]
        ],
        "top_variants": [{"variant": v, "n": cnt} for v, cnt in top_variants],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":             "BBS12 (Bardet-Biedl Syndrome 12 protein); Chr 4q27",
            "OMIM_Gene":        "*610188",
            "Chromosome":       "4q27",
            "Protein":          "729 aa — N-terminal/apical domain (BBS10 apical contact; Arg140 BBS10-BBS12 interface) · intermediate domain (MKKS and BBS10 inter-subunit contacts; Leu287 hydrophobic helix pack) · equatorial domain (primary MKKS equatorial docking; Arg445 MKKS contact; Thr501 ATP-analog loop) · C-terminal scaffold (MKKS-BBS12 equatorial stability)",
            "Domains":          "aa 1–180: N-terminal/apical (BBS10 β-ring apical contact; Arg140 BBS10-BBS12 interface bridge) · aa 181–420: intermediate (MKKS equatorial contact and BBS10 intermediate helix pack; Leu287 hydrophobic core) · aa 421–620: equatorial (primary MKKS α-subunit docking surface; Arg445 MKKS contact loop; Thr501 ATP-analog loop) · aa 621–729: C-terminal scaffold (MKKS-BBS12 equatorial stability; Glu563Ter removes scaffold → MKKS lost)",
            "BCC_position":     "BCC γ-ring structural subunit — MKKS (α) · BBS10 (β) · BBS12 (γ) trimer; BBS12 provides the γ-ring that docks to MKKS equatorial and contacts BBS10 apical surface",
            "Inheritance":      "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":    "~5–8% of all BBS; 4th–5th most common BBS gene; most common in consanguineous North African (Maghrebi) and Middle Eastern families",
            "Allele_spectrum":  "Missense/truncating > missense/missense > truncating/truncating; Arg140Cys is MENA/North African enriched; Arg445Trp is European enriched; c.1063+5G>A splice is European",
            "Triallelic_BBS":   "~8% (BBS10 most common third allele — BCC β-ring equatorial docking partner; MKKS/BBS6 second — BCC α-partner)",
            "CHD_penetrance":   "~6% (lower end of BBS spectrum; no cardiac-specific role for BBS12 identified)",
            "Population_note":  "BBS12 is enriched in consanguineous North African (Tunisia, Morocco, Algeria) and Middle Eastern families; identified by Stoetzel et al. 2007 Nature Genetics as a novel BBS gene in these populations",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 12 (BBS12) — #209900",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS12 subtype ~5–8% (~1–1.2 in 500,000)",
            "Cardinal_features": "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR mis-traffic) · Hypogonadism · Renal anomalies · Cognitive/LD · Anosmia",
            "BBS12_unique":      "BCC γ-ring LOF: BBS12 ABSENT + MKKS REDUCED (not absent) + BBS10 REDUCED + BBS2 ABSENT — almost identical to BBS10 IF (BBS10 ABSENT + MKKS REDUCED + BBS12 REDUCED); absent subunit identifies LOF gene; full BCC-panel IF essential; gene panel definitive",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; identical pattern to BBS1–11); severity similar to BBS10",
            "Obesity_mechanism": "LepR mis-trafficking (hypothalamic cilia) → leptin resistance → hyperphagia; GLP1RA applicable (identical to all BBSome subunit LOF)",
            "Renal":             "Structural anomalies + cystic dysplasia + TIN; ESRD ~6%; cilia-dependent tubulogenesis",
            "CHD":               "~6%; echocardiogram indicated at diagnosis",
            "Diagnosis":         "Full BBS gene panel (minimum 20–24 genes including BBS12, BBS6/MKKS, BBS10 — all BCC components); MLPA/CNV for large deletions; BCC-panel IF (MKKS + BBS10 + BBS12 + BBS2); absent subunit on IF identifies which BCC gene is mutated",
            "Treatment":         "Symptomatic (no curative gene therapy approved 2026); GLP1RA for obesity; renal surveillance; identical management to BBS1–11",
            "Prognosis":         "Progressive visual loss → blindness 3rd–4th decade; renal variable; survival near-normal with management",
        },
        "key_variants": [
            {
                "variant":     "p.Arg140Cys (c.418C>T)",
                "domain":      "N-terminal/apical domain — BBS10-BBS12 apical interface (Arg140 bridges BBS10 β-ring apical contact with BBS12 γ-ring N-terminal region)",
                "consequence": "BBS10-BBS12 apical interface disrupted; BCC γ-ring cannot dock to BBS10 β-ring; BCC trimer incomplete from γ-ring side; MKKS and BBS10 partially degraded (REDUCED IF); BBS2/BBS7 misfold; BBSome Step 0 fails; full BBS phenotype",
                "ethnicity":   "North African (Maghrebi) founder — Tunisia, Morocco, Algeria; MENA consanguineous enrichment; most common BBS12 allele in North African cohorts (Stoetzel 2007)"
            },
            {
                "variant":     "p.Leu287Pro (c.860T>C)",
                "domain":      "Intermediate domain — BCC inter-subunit hydrophobic helix pack (Leu287 in the α-helix packing against both MKKS equatorial surface and BBS10 intermediate helix)",
                "consequence": "BCC inter-subunit helix pack destabilised; MKKS-BBS12-BBS10 contacts weakened simultaneously; BCC trimer cannot fully assemble; BBS2/BBS7 folding impaired; full BBS phenotype at homozygosity",
                "ethnicity":   "South Asian enrichment; pan-ethnic distribution"
            },
            {
                "variant":     "p.Arg445Trp (c.1333C>T)",
                "domain":      "Equatorial domain — primary MKKS α-equatorial docking surface (Arg445 salt-bridge to MKKS equatorial contact loop; critical for BCC trimer equatorial closure)",
                "consequence": "MKKS equatorial docking to BBS12 lost; BCC trimer cannot close at equatorial surface; MKKS monomeric (REDUCED IF); BBS10 also partially destabilised; BBS2/BBS7 misfold; full BBS phenotype",
                "ethnicity":   "European enrichment (France, Netherlands, UK); also pan-ethnic"
            },
            {
                "variant":     "p.Thr501Met (c.1502C>T)",
                "domain":      "Equatorial domain ATP-analog loop (Thr501 in the ATP-binding-site homolog; structural role; no ATPase activity but loop geometry required for MKKS secondary equatorial contact)",
                "consequence": "Equatorial loop geometry altered; MKKS secondary contact weakened; partial BCC assembly; BBS2/BBS7 folding impaired; full BBS phenotype at homozygosity",
                "ethnicity":   "Pan-ethnic; slight South Asian enrichment"
            },
            {
                "variant":     "p.Glu563Ter (c.1687G>T)",
                "domain":      "Truncating null — stops at aa 563; equatorial C-terminus and entire C-terminal scaffold lost (MKKS primary equatorial docking surface truncated)",
                "consequence": "Complete loss of equatorial C-terminus and C-terminal scaffold → MKKS cannot dock → BCC trimer incomplete → BCC collapse → BBS2/BBS7 misfold → full BBS null phenotype",
                "ethnicity":   "Pan-ethnic (most common truncating allele in BBS12)"
            },
            {
                "variant":     "c.1063+5G>A (splice site)",
                "domain":      "Intron 9 splice donor — leads to exon 9 skipping or cryptic splice activation; removes aa 336–420 (equatorial domain N-terminal region including MKKS primary docking loop)",
                "consequence": "Exon skipping → loss of equatorial N-terminal region → MKKS primary docking lost → BCC trimer incomplete → BBS2/BBS7 misfold → full BBS phenotype; RNA studies show both exon-skip and cryptic products",
                "ethnicity":   "European enrichment (detected in French, British, Dutch cohorts)"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG rod-first) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20–24 genes including BBS12, BBS6/MKKS, BBS10 — all BCC components); MLPA/CNV for large deletions",
            "3. BBS12-specific IF fingerprint: anti-BBS12 (ABSENT) + anti-MKKS (REDUCED, not absent) + anti-BBS10 (REDUCED) + anti-BBS2 (ABSENT); full BCC-panel IF mandatory",
            "4. Key discriminator: if MKKS IF is completely ABSENT → suspect BBS6/MKKS LOF (not BBS12); BBS12 LOF leaves MKKS REDUCED (monomeric, not absent)",
            "5. Key discriminator: if BBS10 IF is completely ABSENT but MKKS REDUCED → suspect BBS10 LOF (not BBS12); BBS12 LOF leaves BBS10 REDUCED (not absent); gene panel is the only definitive discriminator between BBS10 LOF and BBS12 LOF",
            "6. Tri-allelic BBS: screen BBS10 (most common third allele in BBS12 families — β-ring equatorial docking partner) and BBS6/MKKS (second most common — α-subunit BCC partner)",
            "7. Consider BBS12 early in consanguineous North African (Maghrebi) and Middle Eastern families — BBS12 is enriched in these populations; Arg140Cys is a North African founder allele",
            "8. CHD workup: echocardiogram (~6%); renal ultrasound + GFR; cystatin C annually; ESRD risk ~6%; ERG + Goldmann VF + OCT at diagnosis",
            "9. BCC-panel IF: test MKKS + BBS10 + BBS12 + BBS2 together — the pattern of absent vs. reduced subunits across the BCC panel identifies which gene is mutated before gene panel results return",
            "10. Gene panel is the DEFINITIVE discriminator between BBS10 LOF and BBS12 LOF — IF alone cannot separate the two; both give MKKS REDUCED + BBS2 ABSENT; BCC-panel IF identifies which of BBS10 or BBS12 is ABSENT vs REDUCED",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; photosensitising drug avoidance; vitamin A not indicated for BBS; gene therapy investigational",
            "2. Obesity: GLP-1 receptor agonists (semaglutide) — LepR mis-trafficking mechanism identical to all BBSome LOF; bariatric surgery in refractory cases",
            "3. Renal: ACE inhibitor/ARB for CKD progression; renal transplant for ESRD (curative); avoid nephrotoxins",
            "4. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling",
            "5. Anosmia: safety counselling (gas leaks, food spoilage); smoke detectors mandatory; smell training investigational",
            "6. Cognitive/LD: early intervention; IEP; neuropsychological assessment; occupational therapy",
            "7. CHD (~6%): echocardiogram-guided cardiology; no specific BBS12 cardiac lesion profile; surgical/catheterisation as indicated",
            "8. Genetic counselling: AR inheritance; sibling recurrence 25%; prenatal/preimplantation genetic testing available; consanguinity counselling especially in North African/MENA families",
            "9. Co-testing BCC subunits: always screen BBS10 and BBS6/MKKS in BBS12 families (~8% tri-allelic); BCC panel testing recommended",
            "10. Surveillance: annual ophthalmology + renal + metabolic (HbA1c, lipids) + cardiac; growth/pubertal charting in children",
        ],
        "ddx_table": [
            {"disease": "BBS10 (BBS10 gene)", "key_difference": "BBS10 LOF → BBS10 ABSENT by IF; MKKS REDUCED; BBS12 REDUCED; BBS2 ABSENT. BBS12 LOF → BBS12 ABSENT; MKKS REDUCED; BBS10 REDUCED; BBS2 ABSENT. The absent BCC subunit on IF (BBS10 absent vs BBS12 absent) identifies which gene is mutated — gene panel is the definitive discriminator since clinical and IF patterns are almost identical."},
            {"disease": "BBS6 / MKKS (MKKS gene)", "key_difference": "BBS6 LOF → MKKS protein ABSENT by IF (MKKS is the mutated α-subunit); BBS10 REDUCED; BBS12 REDUCED. BBS12 LOF → BBS12 ABSENT; MKKS REDUCED (MKKS monomeric but present). This MKKS absent vs MKKS reduced distinction is the key cellular discriminator between BBS6 and BBS12 LOF."},
            {"disease": "BBS1 (BBS1 gene)", "key_difference": "BBS1 LOF → BBS1 absent IF; BCC is intact (MKKS, BBS10, BBS12 all normal); BBS2 normal. BBS12 LOF → BCC failed; BBS2 absent. BBS1 is a downstream BBSome cargo subunit; BBS12 is upstream BCC chaperonin — different IF panel signatures."},
            {"disease": "BBS2 (BBS2 gene)", "key_difference": "BBS2 LOF → BBS2 absent IF; BCC (MKKS, BBS10, BBS12) all NORMAL (BCC folded BBS2 successfully but BBS2 itself is mutated). BBS12 LOF → BBS2 absent IF AND BCC disrupted (BBS12 absent, MKKS reduced). BCC normal vs BCC disrupted is the key discriminator."},
            {"disease": "Alström Syndrome (ALMS1)", "key_difference": "Cone-rod (cone first, not rod first); no polydactyly; infantile DCM (60%); sensorineural hearing loss; normal BCC IF (MKKS, BBS10, BBS12 all normal); ALMS1 mutations. Obesity and renal overlap but ERG pattern and DCM distinguish."},
            {"disease": "Joubert Syndrome (JBTS)", "key_difference": "Molar tooth sign on brain MRI; cerebellar vermis hypoplasia; BBS12 mutations not causative of classic JBTS; different ciliary module (IFT/transition-zone vs BCC/BBSome pathway)."},
            {"disease": "McKusick-Kaufman Syndrome (MKKS hypomorph)", "key_difference": "MKS caused by hypomorphic MKKS/BBS6 variants (Ile339Val Amish founder); hydrometrocolpos + polydactyly + CHD but NO retinal dystrophy and NO renal disease — BCC partially functional. BBS12 null gives full BBS including retinal and renal."},
        ],
        "mechanism_glossary": [
            {
                "term": "BCC (BBSome Chaperonin Complex) — MKKS · BBS10 · BBS12 Trimer",
                "definition": "The BCC is a group-II chaperonin-like heterotrimer that acts at the most upstream step of BBSome biogenesis (Step 0). It co-folds the WD40 β-propeller barrels of BBS2 and BBS7 so they can form the obligate BBS2-BBS7 core dimer. The BCC architecture mimics a group-II chaperonin ring: MKKS (BBS6) is the α-subunit providing apical substrate contacts; BBS10 is the β-ring structural subunit; BBS12 is the γ-ring structural subunit providing the equatorial MKKS docking surface. LOF of any BCC subunit (MKKS, BBS10, or BBS12) collapses the BCC trimer, causing BBS2 and BBS7 to misfold, and blocking Step 0 of BBSome assembly."
            },
            {
                "term": "BBS12 LOF IF Fingerprint — BBS12 ABSENT + MKKS REDUCED + BBS10 REDUCED + BBS2 ABSENT",
                "definition": "In BBS12 LOF, BBS12 protein (γ-ring) is absent by IF. Without the BBS12 γ-ring, MKKS (α) and BBS10 (β) can individually fold but cannot form the complete BCC trimer — both become partially unstable as monomers/dimers and show REDUCED IF (not absent). BBS2/BBS7 misfold (no BCC-mediated folding) → BBS2 absent. This is ALMOST IDENTICAL to the BBS10 LOF IF fingerprint (BBS10 ABSENT + MKKS REDUCED + BBS12 REDUCED + BBS2 ABSENT). The only discriminating feature is which BCC subunit is ABSENT vs REDUCED: in BBS12 LOF, BBS12 is ABSENT and BBS10 is REDUCED; in BBS10 LOF, BBS10 is ABSENT and BBS12 is REDUCED. Gene panel is definitive."
            },
            {
                "term": "BBS12 as the Most Common Third Allele in Tri-Allelic BBS10 Families",
                "definition": "In tri-allelic BBS families carrying biallelic BBS10 mutations, the most common third allele is BBS12 (~8% of BBS10 families). This is mechanistically predicted: BBS12 is the direct equatorial docking partner of BBS10 in the BCC trimer — a heterozygous BBS12 third allele reduces available BCC trimers, worsening the phenotype in patients already null for BBS10. The reciprocal is also true: BBS10 is the most common third allele in BBS12 families, as BBS10 is BBS12's β-ring equatorial docking partner."
            },
            {
                "term": "BBS12 γ-Ring Equatorial Domain — Primary MKKS Docking Surface",
                "definition": "The equatorial domain of BBS12 (aa 421–620) is the structural region that docks to MKKS (α-subunit) equatorial surface, completing the BCC trimer at its equatorial level. Arg445 forms a critical salt-bridge to the MKKS equatorial contact loop. The equatorial domain contains an ATP-binding-site homolog (Thr501 ATP-analog loop) — like the full group-II chaperonin ring, BCC has an ATP-analog site but no ATPase activity. The C-terminal scaffold (aa 621–729) stabilises the MKKS-BBS12 equatorial contact; Glu563Ter truncating null removes this scaffold → MKKS docking lost → BCC collapses."
            },
            {
                "term": "BBS12 Enrichment in North African and Middle Eastern Consanguineous Families",
                "definition": "BBS12 was identified by Stoetzel et al. (2007, Nature Genetics) in consanguineous North African (Maghrebi) families, where it is significantly enriched compared to other populations. The Arg140Cys variant (c.418C>T) is a North African/MENA founder allele. Consanguinity rates in BBS12 families from these regions can reach 40–60%, explaining why BBS12 constitutes a higher proportion of BBS in North African cohorts than its global frequency of ~5–8% suggests. This is the same population enrichment pattern seen for BBS10 (also enriched in North African/Maghrebi populations), consistent with BCC trimer subunit genes being under similar founder-effect pressures."
            },
            {
                "term": "LepR Ciliary Mis-Trafficking in BBS12 — Identical to All BBSome LOF",
                "definition": "Leptin receptor (LepR) requires the complete BBSome octamer for retrograde IFT-mediated removal from hypothalamic neuronal cilia. In BBS12 LOF, BCC γ-ring failure → BBS2/BBS7 misfold → BBSome Step 0 fails → no complete octamer → LepR constitutively retained in hypothalamic cilia → leptin resistance → hyperphagia and obesity. The mechanism is identical across all BBSome LOF (BBS1–12) and all BCC LOF (BBS6, BBS10, BBS12). GLP-1 receptor agonists bypass this by activating GLP1R interneurons via a non-ciliary pathway — applicable to all BBS subtypes including BBS12."
            },
        ],
    }
