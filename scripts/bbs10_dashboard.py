"""
BBS10 Bardet-Biedl Syndrome Type 10 — BBS10 (BCC β-Ring Structural Subunit) Dashboard
========================================================================================
Primary Gene : BBS10 (*610148) — Bardet-Biedl Syndrome 10 protein (C12orf58);
               723 aa; Chr 12q21.2.
               BBS10 is the STRUCTURAL β-RING SUBUNIT of the BBSome Chaperonin
               Complex (BCC). The BCC is a group-II chaperonin-like heterotrimer
               (MKKS/BBS6 · BBS10 · BBS12) that co-folds the WD40 β-propellers
               of BBS2 and BBS7 before BBSome assembly can begin (Step 0).
               Domain organisation (723 aa):
                 Apical domain (aa 1–190): contacts BBS2 WD40 propeller top-face
                   and MKKS apical domain; Cys91 bridges MKKS-BBS10 interface;
                   Ile339 (via hinge) stabilises BBS2-folding geometry;
                 Intermediate/hinge domain (aa 191–450): BCC inter-subunit
                   contacts (MKKS and BBS12); Leu394 hydrophobic core residue;
                 Equatorial domain (aa 451–650): primary BBS12 docking surface;
                   contains ATP-analog-binding site (no ATPase activity; structural
                   role only); Arg462 contacts BBS12 equatorial surface;
                   Thr483 in ATP-analog loop;
                 C-terminal scaffold (aa 651–723): stabilises equatorial-BBS12
                   contact; Glu559Ter truncating null — equatorial C-terminus and
                   scaffold lost.
               BBS10 LOF → BCC trimer structurally incomplete → BCC cannot
               co-fold BBS2/BBS7 WD40 barrels → BBS2 and BBS7 misfold → BBSome
               core module (Step 0) fails before assembly → no complete BBSome
               octamer → GPCR cargoes (LepR, SSTR3, D1R, Smo) mis-trafficked
               in cilia → full BBS.
               Key cellular discriminator from BBS6/MKKS LOF:
                 BBS6 LOF: MKKS protein absent IF (MKKS itself is the mutated
                   α-subunit); BBS10 and BBS12 REDUCED (BCC trimer destabilised
                   from the α-subunit side); BBS2 absent.
                 BBS10 LOF: BBS10 protein absent IF; MKKS REDUCED (MKKS α-subunit
                   present as monomer but BCC trimer structurally incomplete);
                   BBS12 REDUCED; BBS2 absent (BBS2-BBS7 misfold identical to BBS6).
                 KEY FINGERPRINT: MKKS REDUCED (not absent) + BBS10 ABSENT + BBS2
                   ABSENT = BBS10 LOF. MKKS ABSENT + BBS2 ABSENT = BBS6 LOF.
               BBS10 is SECOND MOST COMMON BBS subtype globally (~20% of all BBS
               cases), equal in frequency to BBS1, making BBS10 together with BBS1
               responsible for ~40–45% of all BBS.  Enriched in European/French and
               North African/Maghrebi populations.
Disease        : Bardet-Biedl Syndrome (#209900) — BBS10 subtype
Frequency      : ~20% of all BBS — tied with BBS1 as the most common subtype
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~8% of BBS10 families (BBS12 most common third allele — BCC
                 trimer partner; MKKS/BBS6 second — BCC partner)
Cohort         : 40 patients; educational/synthetic; seed 351
Endpoints      : /api/bbs10/overview | /api/bbs10/breakdown | /api/bbs10/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 351
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("European",      0.42),   # highest frequency — France, UK, Netherlands, Germany
        ("North African", 0.20),   # Tunisia, Morocco, Algeria (Maghrebi founder effect)
        ("South Asian",   0.16),
        ("MENA (other)",  0.12),
        ("East Asian",    0.05),
        ("Other",         0.05),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "European":      ["p.Cys91Arg (c.271T>C)", "p.Ile339Thr (c.1016T>C)", "p.Glu559Ter (c.1675G>T)"],
        "North African": ["p.Arg462Gln (c.1385G>A)", "p.Glu559Ter (c.1675G>T)", "p.Cys91Arg (c.271T>C)"],
        "South Asian":   ["p.Leu394Pro (c.1181T>C)", "p.Arg462Gln (c.1385G>A)", "p.Glu559Ter (c.1675G>T)"],
        "MENA (other)":  ["p.Arg462Gln (c.1385G>A)", "p.Thr483Met (c.1448C>T)", "p.Glu559Ter (c.1675G>T)"],
        "East Asian":    ["p.Glu559Ter (c.1675G>T)", "p.Thr483Met (c.1448C>T)", "p.Cys91Arg (c.271T>C)"],
        "Other":         ["p.Glu559Ter (c.1675G>T)", "p.Ile339Thr (c.1016T>C)"],
    }

    allele_classes = ["Missense/Missense", "Missense/Truncating", "Truncating/Truncating",
                      "Splice/Missense", "Splice/Truncating"]
    allele_weights = [0.28, 0.38, 0.22, 0.08, 0.04]

    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.14, 0.34, 0.30, 0.22]   # BBS10 — slightly more severe retinal distribution

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.40, 0.14, 0.09, 0.05, 0.32]   # 68% polydactyly — higher than BBS1-9

    renal_types = ["Structural anomaly (horseshoe/duplex)", "Cystic dysplasia",
                   "Tubulointerstitial nephropathy", "ESRD", "None"]
    renal_w = [0.14, 0.14, 0.08, 0.06, 0.58]

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.20, 0.46, 0.24, 0.10]

    misdiag = ["Isolated RP / Leber amaurosis", "Obesity syndrome (Prader-Willi screen)",
               "Alström Syndrome (cone-rod confusion)", "CHARGE/Joubert", "No initial misdiagnosis"]
    mis_w   = [0.28, 0.22, 0.12, 0.06, 0.32]

    presentation_ages = ["Birth (polydactyly noted)", "Infancy (nystagmus/strabismus)",
                         "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"]
    pres_w  = [0.30, 0.18, 0.24, 0.20, 0.08]

    patients = []
    for i in range(n):
        eth = rng.choices(eth_labels, weights=eth_weights, k=1)[0]
        v_pool = variants_by_eth[eth]
        v1 = rng.choice(v_pool)
        v2 = rng.choice(v_pool)
        while v2 == v1:
            v2 = rng.choice(v_pool)

        poly_type = rng.choices(polydactyly_types, weights=poly_w, k=1)[0]
        renal_type = rng.choices(renal_types, weights=renal_w, k=1)[0]
        ret_stage = rng.choices(retinal_stages, weights=ret_w, k=1)[0]
        allele_cl = rng.choices(allele_classes, weights=allele_weights, k=1)[0]
        dx_ag = rng.choices(dx_ages, weights=dx_w, k=1)[0]
        pres_age = rng.choices(presentation_ages, weights=pres_w, k=1)[0]
        mis_d = rng.choices(misdiag, weights=mis_w, k=1)[0]

        # BBS10: ~68% polydactyly, ~85% obesity, ~48% cognitive LD,
        #        ~65% hypogonadism, ~60% anosmia, ~42% renal, ~7% CHD
        polydactyly  = poly_type != "None"
        obesity      = rng.random() < 0.85
        cognitive_ld = rng.random() < 0.48
        hypogonadism = rng.random() < 0.65
        anosmia      = rng.random() < 0.60
        renal_any    = renal_type != "None"
        chd          = rng.random() < 0.07
        ret_end      = ret_stage == "Legal blindness / end-stage"
        tri_bbs      = rng.random() < 0.08   # ~8% tri-allelic
        mis_dx       = mis_d != "No initial misdiagnosis"
        esrd         = renal_type == "ESRD"

        patients.append({
            "id":                  f"BBS10-{i+1:03d}",
            "ethnicity":           eth,
            "variant1":            v1,
            "variant2":            v2,
            "allele_class":        allele_cl,
            "polydactyly":         polydactyly,
            "polydactyly_type":    poly_type,
            "obesity":             obesity,
            "cognitive_ld":        cognitive_ld,
            "hypogonadism":        hypogonadism,
            "anosmia":             anosmia,
            "renal_any":           renal_any,
            "renal_type":          renal_type,
            "chd":                 chd,
            "retinal_endstage":    ret_end,
            "retinal_stage":       ret_stage,
            "dx_age_group":        dx_ag,
            "presentation_age":    pres_age,
            "triallelic_bbs":      tri_bbs,
            "initial_misdiagnosis":mis_dx,
            "misdiagnosis":        mis_d,
            "esrd":                esrd,
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

    ret_stages: Dict[str, int] = {}
    for p in c:
        s = p["retinal_stage"]
        ret_stages[s] = ret_stages.get(s, 0) + 1

    age_dist: Dict[str, int] = {}
    for p in c:
        g = p["dx_age_group"]
        age_dist[g] = age_dist.get(g, 0) + 1

    return {
        "cohort_n":     n,
        "seed":         _SEED,
        "gene":         "BBS10",
        "gene_alias":   "C12orf58",
        "omim_gene":    "*610148",
        "chromosome":   "12q21.2",
        "protein_aa":   723,
        "disease_omim": "#209900",
        "bbs_subtype":  "BBS10",
        "inheritance":  "Autosomal Recessive (biallelic LOF)",
        "frequency_pct_bbs": "~20%",
        "mechanism_class": "BCC β-Ring Structural Subunit (MKKS-BBS10-BBS12 BCC trimer; co-folds BBS2/BBS7 WD40 barrels; BBSome Step 0)",
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
        },
        "mechanism": (
            "BBS10 (723 aa; Chr 12q21.2; OMIM *610148), also called C12orf58, is the STRUCTURAL "
            "β-RING SUBUNIT of the BBSome Chaperonin Complex (BCC). The BCC is a group-II chaperonin-"
            "like heterotrimer (MKKS/BBS6 · BBS10 · BBS12) that acts upstream of BBSome assembly to "
            "co-fold the WD40 β-propeller barrels of BBS2 and BBS7. Without BCC-mediated folding, "
            "BBS2 and BBS7 misfold and are degraded by the proteasome → the BBS2-BBS7 obligate core "
            "dimer (Step 0 of BBSome biogenesis) never forms → no complete BBSome octamer → GPCR "
            "cargoes (LepR, SSTR3, D1R, Smo) mis-trafficked in cilia → leptin resistance, rod-cone "
            "dystrophy, polydactyly, renal anomalies, hypogonadism, anosmia, cognitive impairment. "
            "BBS10 provides the structural scaffold of the BCC β-ring: its apical domain contacts "
            "BBS2 WD40 top-face and MKKS apical domain; its equatorial domain is the primary BBS12 "
            "docking surface. BBS10 LOF → BCC trimer cannot form → MKKS present as monomer (REDUCED "
            "IF, not absent) → BCC activity lost → BBS2/BBS7 misfold → BBSome Step 0 fails. "
            "Critically, BBS10 is the SECOND MOST COMMON BBS subtype globally (~20% of all BBS "
            "cases), equal in frequency to BBS1, making BBS10+BBS1 responsible for ~40–45% of all BBS "
            "worldwide. BBS10 is especially enriched in European and North African (Maghrebi) populations."
        ),
        "key_distinction": (
            "BBS10 LOF produces the SAME downstream BBSome null phenotype as BBS6/MKKS LOF "
            "(both = BCC failure → BBS2/BBS7 misfold → Step 0 fails) but the cellular IF fingerprint "
            "differs critically: in BBS6/MKKS LOF, MKKS protein itself is ABSENT (the mutated α-subunit "
            "is degraded); in BBS10 LOF, MKKS is PRESENT but REDUCED (MKKS α-subunit is intact but "
            "cannot form the BCC trimer without BBS10 β-ring → BCC is monomeric). "
            "BBS10 IF fingerprint: BBS10 ABSENT + MKKS REDUCED + BBS2 ABSENT + BBS12 REDUCED. "
            "BBS6 IF fingerprint: MKKS ABSENT + BBS10 REDUCED + BBS12 REDUCED + BBS2 ABSENT. "
            "BBS10 is the highest-frequency BBS subtype in European cohorts and Newfoundland "
            "(Newfoundland founder — c.271T>C Cys91Arg); tied with BBS1 globally. "
            "BBS1: cargo propeller; M390R founder; BBSome intact but cargo loading fails. "
            "BBS2/BBS7: core dimer scaffold; Step 0 fails; BCC-dependent folding also fails without BCC. "
            "BBS3 (ARL6): GTPase membrane recruiter; BBSome intact but not recruited to ciliary base. "
            "BBS6 (MKKS): BCC α-subunit; MKKS absent IF — distinguishes from BBS10 (MKKS reduced). "
            "BBS10 [THIS]: BCC β-ring; MKKS reduced (not absent) + BBS10 absent. "
            "BBS12: BCC γ-subunit; MKKS reduced + BBS12 absent — IF almost identical to BBS10; "
            "gene panel distinguishes BBS10 from BBS12."
        ),
        "bbs_pathway_comparison": [
            {"gene": "BBS1",          "role": "BBSome cargo-recognition beta-propeller (M390R founder)",
             "function_class": "Structural BBSome Subunit", "omim": "*209901", "frequency_pct": 25},
            {"gene": "BBS2",          "role": "BBSome core dimer partner of BBS7 — WD40 scaffold",
             "function_class": "Structural BBSome Subunit (Core Dimer)", "omim": "*606151", "frequency_pct": 15},
            {"gene": "BBS3 (ARL6)",   "role": "GTPase recruits BBSome to ciliary membrane",
             "function_class": "External GTPase Recruiter", "omim": "*608845", "frequency_pct": 4},
            {"gene": "BBS4",          "role": "TPR scaffold tethers BBSome at PCM1 satellites",
             "function_class": "PCM1 Satellite Tether", "omim": "*600374", "frequency_pct": 3},
            {"gene": "BBS5",          "role": "PH domain anchors BBSome to ciliary PtdIns lipids",
             "function_class": "Ciliary Lipid Anchor", "omim": "*603650", "frequency_pct": 2},
            {"gene": "BBS6 / MKKS",   "role": "BCC α-subunit; Group-II chaperonin-like; co-folds BBS2/BBS7",
             "function_class": "BCC α-Subunit (Upstream Chaperonin)", "omim": "*604896", "frequency_pct": "5–8"},
            {"gene": "BBS7",          "role": "Obligate WD40 dimer partner of BBS2 — core module scaffold",
             "function_class": "Structural BBSome Subunit (Core Dimer)", "omim": "*607590", "frequency_pct": "2–4"},
            {"gene": "BBS8 / TTC8",   "role": "TPR solenoid — IFT-B docking arm; only BBSome-IFT-B bridge",
             "function_class": "IFT-B Docking Arm (Peripheral Module)", "omim": "*608132", "frequency_pct": "2–4"},
            {"gene": "BBS9 / PTHB1",  "role": "Bridge subunit — spans core (BBS2-BBS7) and peripheral modules",
             "function_class": "BBSome Inter-Module Bridge", "omim": "*607968", "frequency_pct": 2},
            {"gene": "BBS10 [THIS]",  "role": "BCC β-ring structural subunit; primary BBS12 docking; apical BBS2 contact",
             "function_class": "BCC β-Ring Subunit (Upstream Chaperonin)", "omim": "*610148", "frequency_pct": 20},
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
        ("Tri-allelic BBS (BBS12/MKKS 3rd allele)",          sum(1 for p in c if p["triallelic_bbs"]),    _pct(sum(1 for p in c if p["triallelic_bbs"]))),
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
            "Gene":             "BBS10 (Bardet-Biedl Syndrome 10 protein; alias C12orf58); Chr 12q21.2",
            "OMIM_Gene":        "*610148",
            "Chromosome":       "12q21.2",
            "Protein":          "723 aa — apical domain (BBS2 WD40 top-face + MKKS apical contact) · intermediate hinge (BCC inter-subunit) · equatorial domain (primary BBS12 docking surface; ATP-analog site) · C-terminal scaffold",
            "Domains":          "aa 1–190: apical (BBS2 WD40 top-face contact; MKKS α-apical contact; Cys91 MKKS interface) · aa 191–450: intermediate/hinge (BCC inter-subunit; Leu394 hydrophobic core; Ile339 hinge folding) · aa 451–650: equatorial (primary BBS12 docking; ATP-analog loop; Arg462 BBS12-equatorial contact; Thr483 ATP-analog loop) · aa 651–723: C-terminal scaffold (equatorial-BBS12 stability)",
            "BCC_position":     "BCC β-ring structural subunit — MKKS (α) · BBS10 (β) · BBS12 (γ) trimer; BBS10 provides the structural scaffold bridging MKKS apical and BBS12 equatorial contacts",
            "Inheritance":      "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":    "~20% of all BBS — tied with BBS1 as the two most common subtypes; ~40–45% of BBS cases are BBS1 or BBS10",
            "Allele_spectrum":  "Missense/truncating > missense/missense > truncating/truncating; Cys91Arg is European enriched (Newfoundland founder); Arg462Gln is MENA/North African enriched",
            "Triallelic_BBS":   "~8% (BBS12 most common third allele — BCC trimer partner; MKKS/BBS6 second — BCC partner; BBS2/BBS7 third — downstream substrates also modifiable)",
            "CHD_penetrance":   "~7% (intermediate; no specific cardiac ciliary role for BBS10 identified)",
            "Population_note":  "BBS10 is the predominant BBS gene in European (especially French) and North African (Maghrebi) populations, accounting for >30% of BBS in these groups",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 10 (BBS10) — #209900",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS10 subtype ~20% (~2–3 in 500,000)",
            "Cardinal_features": "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR mis-traffic) · Hypogonadism · Renal anomalies · Cognitive/LD · Anosmia",
            "BBS10_unique":      "BCC β-ring LOF: MKKS REDUCED (not absent) + BBS10 ABSENT + BBS2 ABSENT + BBS12 REDUCED — distinguishes from BBS6/MKKS LOF (MKKS ABSENT) and BBS12 LOF (BBS12 ABSENT) by IF panel",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; identical pattern to BBS1–9); BBS10 may have slightly earlier onset than BBS1",
            "Obesity_mechanism": "LepR mis-trafficking (hypothalamic cilia) → leptin resistance → hyperphagia; GLP1RA applicable (identical to all BBSome subunit LOF)",
            "Renal":             "Structural anomalies + cystic dysplasia + TIN; ESRD ~6%; cilia-dependent tubulogenesis",
            "CHD":               "~7%; intermediate; echocardiogram indicated at diagnosis",
            "Diagnosis":         "Full BBS gene panel; fibroblast IF: BBS10 absent + MKKS REDUCED (not absent) + BBS12 REDUCED + BBS2 absent — complete BCC panel IF essential to distinguish BBS6/BBS10/BBS12",
            "Treatment":         "Symptomatic (no curative gene therapy approved 2026); GLP1RA for obesity; renal surveillance; same management as BBS1–9",
            "Prognosis":         "Progressive visual loss → blindness 3rd–4th decade; renal variable; survival near-normal with management",
        },
        "key_variants": [
            {
                "variant":     "p.Cys91Arg (c.271T>C)",
                "domain":      "Apical domain — MKKS-BBS10 interface (Cys91 forms inter-subunit disulfide/hydrophobic contact at BCC α-β junction)",
                "consequence": "BCC α-β (MKKS-BBS10) interface disrupted; BCC trimer cannot form; MKKS monomeric (REDUCED IF); BBS2/BBS7 misfold; BBSome Step 0 fails; full BBS phenotype",
                "ethnicity":   "European founder (Newfoundland, France, UK; ~15–20% of European BBS10 alleles)"
            },
            {
                "variant":     "p.Ile339Thr (c.1016T>C)",
                "domain":      "Intermediate hinge domain — BCC hinge stabilisation (Ile339 hydrophobic packing in hinge α-helix 4; required for BBS2-substrate presentation geometry)",
                "consequence": "Hinge flexibility altered; BCC substrate (BBS2 WD40) folding impaired; partial BCC function at permissive temperature; full loss at physiological; BBS phenotype",
                "ethnicity":   "French-Canadian enrichment; pan-European distribution"
            },
            {
                "variant":     "p.Arg462Gln (c.1385G>A)",
                "domain":      "Equatorial domain — BBS12 γ-subunit docking surface (Arg462 salt-bridge to BBS12 Glu291 on equatorial contact loop)",
                "consequence": "BBS12 docking to BBS10 equatorial domain lost; BCC trimer incomplete; MKKS and BBS10 fail to form functional BCC without BBS12; BBS2/BBS7 misfold; full BBS",
                "ethnicity":   "MENA / North African (Maghrebi) founder (Morocco, Tunisia, Algeria)"
            },
            {
                "variant":     "p.Leu394Pro (c.1181T>C)",
                "domain":      "Intermediate domain hydrophobic core (Leu394 packs against hinge-equatorial transition helices; Pro insertion disrupts fold)",
                "consequence": "Intermediate-equatorial domain junction destabilised; BCC structural scaffold collapses; BBS12 docking secondary surface lost; full BCC disruption; full BBS",
                "ethnicity":   "South Asian (enriched; pan-ethnic distribution)"
            },
            {
                "variant":     "p.Thr483Met (c.1448C>T)",
                "domain":      "Equatorial domain ATP-analog loop (Thr483 in the ATP-binding-site homolog; structural role; no ATPase activity but loop geometry required for BBS12 contact)",
                "consequence": "Equatorial loop geometry altered; BBS12 secondary contact weakened; partial BCC assembly; BBS2/BBS7 folding impaired; full BBS phenotype at homozygosity",
                "ethnicity":   "Pan-ethnic; slight MENA enrichment"
            },
            {
                "variant":     "p.Glu559Ter (c.1675G>T)",
                "domain":      "Truncating null — stops at aa 559; equatorial C-terminus and entire C-terminal scaffold lost (BBS12 primary docking surface truncated)",
                "consequence": "Complete loss of equatorial C-terminus and C-terminal scaffold → BBS12 cannot dock → BCC trimer incomplete → BCC collapse → BBS2/BBS7 misfold → full BBS null phenotype",
                "ethnicity":   "Pan-ethnic (most common truncating allele in BBS10)"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG rod-first) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20–24 genes including BBS10, BBS6/MKKS, BBS12 — all BCC components); MLPA/CNV for large deletions",
            "3. BBS10-specific IF fingerprint: anti-BBS10 (ABSENT) + anti-MKKS (REDUCED, not absent) + anti-BBS12 (REDUCED) + anti-BBS2 (ABSENT); full BCC-panel IF recommended",
            "4. Key discriminator: if MKKS IF is completely ABSENT → suspect BBS6/MKKS LOF; BBS10 LOF leaves MKKS present (as monomer, therefore REDUCED)",
            "5. Key discriminator: if BBS12 IF is completely ABSENT but MKKS is reduced → suspect BBS12 LOF (BBS12 is the primary partner of BBS10 equatorial domain); gene panel confirms",
            "6. Tri-allelic BBS: screen BBS12 (most common third allele ~8% — equatorial docking partner) and BBS6/MKKS (second most common — α-subunit BCC partner)",
            "7. CHD workup: echocardiogram (~7% CHD, ASD/VSD); echocardiogram at diagnosis regardless",
            "8. Renal: ultrasound + GFR; cystatin C annually; ESRD risk ~6%; ophthalmology ERG + Goldmann VF; OCT at diagnosis",
            "9. Population screening: in European and North African cohorts, BBS10 accounts for >20% of BBS — test BBS10 early in these populations before proceeding to rarer genes",
            "10. Frequency counselling: BBS10 is the second most common BBS gene globally; in European cohorts, first-tier alongside BBS1; prenatal testing offered after proband confirmation",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; photosensitising drug avoidance; vitamin A not indicated for BBS; gene therapy investigational",
            "2. Obesity: GLP-1 receptor agonists (semaglutide) — LepR mis-trafficking mechanism identical to all BBSome LOF; bariatric surgery in refractory cases",
            "3. Renal: ACE inhibitor/ARB for CKD progression; renal transplant for ESRD (curative); avoid nephrotoxins",
            "4. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling",
            "5. Anosmia: safety counselling (gas leaks, food spoilage); smoke detectors mandatory; smell training investigational",
            "6. Cognitive/LD: early intervention; IEP; neuropsychological assessment; occupational therapy",
            "7. CHD (~7%): echocardiogram-guided cardiology; no specific BBS10 cardiac lesion profile; surgical/catheterisation as indicated",
            "8. Genetic counselling: AR inheritance; sibling recurrence 25%; prenatal/preimplantation genetic testing available",
            "9. Co-testing BCC subunits: always screen BBS12 and BBS6/MKKS in BBS10 families (~8% tri-allelic); BCC panel testing recommended",
            "10. Surveillance: annual ophthalmology + renal + metabolic (HbA1c, lipids) + cardiac; growth/pubertal charting in children",
        ],
        "ddx_table": [
            {"disease": "BBS6 / MKKS (MKKS gene)",  "key_difference": "BBS6 LOF → MKKS protein ABSENT by IF (MKKS is the mutated α-subunit); BBS10 LOF → MKKS REDUCED (not absent — MKKS is intact but monomeric). This MKKS absent vs MKKS reduced distinction is the key cellular discriminator between BBS6 and BBS10; gene panel confirms."},
            {"disease": "BBS12 (BBS12 gene)",        "key_difference": "BBS12 LOF → BBS12 ABSENT by IF; MKKS REDUCED; BBS10 REDUCED. BBS10 LOF → BBS10 ABSENT; MKKS REDUCED; BBS12 REDUCED. The absent subunit on IF indicates which gene is mutated; full BCC-panel IF (MKKS + BBS10 + BBS12) required; gene panel essential."},
            {"disease": "BBS1 (BBS1 gene)",          "key_difference": "BBS1 LOF → BBS1 IF absent; BCC is intact (MKKS, BBS10, BBS12 all normal); BBS2 normal (BCC-folded). BBS10 LOF → BCC failed; BBS2 absent. Mechanistically: BBS1 is downstream BBSome cargo subunit; BBS10 is upstream BCC chaperonin."},
            {"disease": "BBS2 (BBS2 gene)",          "key_difference": "BBS2 LOF → BBS2 absent IF; BCC (MKKS, BBS10, BBS12) intact but non-functional for its substrate. BBS10 LOF → BBS2 absent IF AND BCC disrupted (MKKS reduced). Both give BBS2-absent phenotype but BCC IF panel distinguishes: BCC intact (BBS2 LOF) vs BCC disrupted (BBS10 LOF)."},
            {"disease": "BBS9 / PTHB1 (PTHB1 gene)", "key_difference": "BBS9 LOF → BCC intact (MKKS NORMAL, BBS10 NORMAL); BBS2 REDUCED (core module forms but bridge-less). BBS10 LOF → BCC disrupted (MKKS REDUCED, BBS12 REDUCED); BBS2 ABSENT. MKKS normal vs MKKS reduced is the key discriminator."},
            {"disease": "Alström Syndrome (ALMS1)",  "key_difference": "Cone-rod (cone first, not rod first); no polydactyly; infantile DCM (60%); sensorineural hearing loss; normal BCC IF (MKKS, BBS10, BBS12 all normal); ALMS1 mutations."},
            {"disease": "Joubert Syndrome (JBTS)",   "key_difference": "Molar tooth sign on brain MRI; cerebellar vermis hypoplasia; BBS10 mutations not causative of classic JBTS; different IFT/transition-zone module."},
            {"disease": "McKusick-Kaufman Syndrome (MKKS hypomorph)", "key_difference": "MKS caused by hypomorphic MKKS/BBS6 variants (Ile339Val Amish founder); hydrometrocolpos + polydactyly + CHD but NO retinal dystrophy and NO renal disease — BCC partially functional. BBS10 null gives full BBS including retinal and renal."},
        ],
        "mechanism_glossary": [
            {
                "term": "BCC (BBSome Chaperonin Complex) — MKKS · BBS10 · BBS12 Trimer",
                "definition": "The BCC is a group-II chaperonin-like heterotrimer that acts at the most upstream step of BBSome biogenesis (Step 0). It co-folds the WD40 β-propeller barrels of BBS2 and BBS7 so they can form the obligate BBS2-BBS7 core dimer. The BCC architecture mimics a group-II chaperonin ring: MKKS (BBS6) is the α-subunit providing apical substrate contacts and the ATP-analog binding site; BBS10 is the β-ring structural subunit bridging MKKS and BBS12; BBS12 is the γ-subunit providing equatorial stability. LOF of any BCC subunit (MKKS, BBS10, or BBS12) collapses the BCC trimer, causing BBS2 and BBS7 to misfold and be proteasomally degraded, and blocking Step 0 of BBSome assembly."
            },
            {
                "term": "MKKS REDUCED (not ABSENT) — BBS10 LOF Cellular Fingerprint",
                "definition": "In BBS10 LOF, MKKS protein is expressed normally (MKKS gene is intact) but cannot form the BCC trimer without its β-ring partner (BBS10). The MKKS monomer is less stable than the BCC trimer and is partially degraded, producing a REDUCED IF signal — not absent. In BBS6/MKKS LOF, MKKS itself is the mutated protein → absent IF. This MKKS ABSENT (BBS6) vs MKKS REDUCED (BBS10) distinction is the critical cellular discriminator between the two most upstream BBS pathway defects. Complement with BBS12 IF: BBS12 is also REDUCED in BBS10 LOF (BCC trimer loss affects all partners) — absent in BBS12 LOF."
            },
            {
                "term": "BBS10 as the Second Most Common BBS Gene (20% of All BBS)",
                "definition": "BBS10 is responsible for approximately 20% of all Bardet-Biedl Syndrome cases worldwide, making it co-equal with BBS1 as the two highest-frequency BBS genes. Together, BBS1 and BBS10 account for ~40–45% of all BBS. This frequency reflects BBS10's position in the BCC, which is required for folding the BBS2-BBS7 core dimer — any BCC defect produces a null BBSome phenotype that is phenotypically indistinguishable from BBS1 or BBS2 LOF. BBS10 is especially enriched in European (particularly French) and North African (Maghrebi) populations, where it may account for >30% of BBS cases."
            },
            {
                "term": "BBS10 Equatorial Domain — Primary BBS12 Docking Surface",
                "definition": "The equatorial domain of BBS10 (aa 451–650) is the structural region that anchors BBS12 (γ-subunit) into the BCC trimer. Arg462 forms a critical salt-bridge to BBS12 Glu291 on BBS12's equatorial contact loop. The equatorial domain also contains an ATP-binding-site homolog (Thr483 ATP-analog loop) — like the full group-II chaperonin ring, BCC has an ATP-analog site but no ATPase activity, suggesting the substrate folding mechanism relies on conformational trapping rather than power-stroke ATP hydrolysis. The C-terminal scaffold (aa 651–723) stabilises the equatorial-BBS12 contact; Glu559Ter truncating null removes this scaffold → BBS12 docking lost → BCC collapses."
            },
            {
                "term": "LepR Ciliary Mis-Trafficking in BBS10 — Identical to All BBSome LOF",
                "definition": "Leptin receptor (LepR) requires the complete BBSome octamer for retrograde IFT-mediated removal from hypothalamic neuronal cilia. In BBS10 LOF, BCC failure → BBS2/BBS7 misfold → BBSome Step 0 fails → no complete octamer → LepR constitutively retained in hypothalamic cilia → leptin resistance → hyperphagia and obesity. The mechanism is identical across all BBSome LOF (BBS1–10) and all BCC LOF (BBS6, BBS10, BBS12). GLP-1 receptor agonists bypass this by activating GLP1R interneurons via a non-ciliary pathway — applicable to all BBS subtypes including BBS10."
            },
            {
                "term": "Tri-Allelic BBS in BBS10 — BBS12 as the Dominant Third Allele",
                "definition": "In tri-allelic BBS families carrying biallelic BBS10 mutations, the most common third allele is BBS12 (~8% of BBS10 families). This is mechanistically predicted: BBS12 is the direct equatorial docking partner of BBS10 in the BCC trimer — a heterozygous BBS12 third allele reduces available BCC trimers, worsening the phenotype in patients already null for BBS10. MKKS/BBS6 is the second most common third allele. Unlike BBS9 (which is the universal third allele across all BBSome subunit LOF), BBS10 third alleles are clustered in BCC partners, reflecting the module-specific nature of BCC assembly."
            },
        ],
    }
