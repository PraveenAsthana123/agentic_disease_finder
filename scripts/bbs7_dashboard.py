"""
BBS7 Bardet-Biedl Syndrome Type 7 — BBS7 (WD40 Beta-Propeller Core Dimer) Dashboard
======================================================================================
Primary Gene : BBS7 (*607590) — also called BBS2L1 (BBS2-like protein 1);
               672 aa; WD40 beta-propeller protein homologous to BBS2;
               Chr 4q27.
               BBS7 is a STRUCTURAL BBSome SUBUNIT — the obligate WD40
               beta-propeller dimer partner of BBS2:
                 N-terminal coiled-coil / BBS2-contact domain (aa 1–120):
                   interfaces directly with BBS2 N-terminal coiled-coil
                   to nucleate the BBS2-BBS7 heterodimer (core module);
                   Arg184 at the N-coiled-coil/WD40 junction contacts BBS2
                   Lys156 via salt-bridge — Arg184Gln → dimer interface lost;
                 WD40 beta-propeller (aa 121–600): 7 WD40 repeats forming
                   a canonical 7-bladed beta-propeller; blade 4-5 interface
                   contacts BBS9 (bridge subunit that anchors the BBS2-BBS7
                   core dimer into the full BBSome octamer);
                   Pro532 at WD40 repeat-6/7 loop contacts BBS9 Glu388
                   — Pro532Arg → BBS9 bridge lost → core dimer orphaned
                   from BBSome; Leu523 is a barrel hydrophobic-core residue
                   — Leu523Pro → beta-propeller misfolding → BCC cannot
                   release folded BBS7 → BBS2 also stranded;
                 C-terminal domain (aa 601–672): contacts BBIP10 (BBIP1),
                   the BBSome adaptor that links core dimer to the
                   peripheral module (BBS4-BBS5-BBS8-BBS9 side); loss
                   reduces BBSome octamer stability without fully blocking
                   core dimer formation.
               BBS7 LOF → BBS2-BBS7 obligate dimer cannot form → BBSome
               core module (Step 0) fails → BBSome octamer cannot assemble
               → same downstream ciliary GPCR mis-trafficking cascade as
               BBS2 LOF; phenotypically indistinguishable from BBS2.
               Key cellular discriminator from BBS2 LOF:
                 In BBS7-null cells: anti-BBS2 IF is ABSENT/reduced
                   (BBS2 protein is unstable without BBS7 partner — rapid
                   proteasomal degradation of orphaned BBS2); anti-BBS7 IF:
                   absent; anti-MKKS IF: NORMAL (BCC chaperonin intact —
                   BBS7 was folded successfully by BCC, failure occurs at
                   assembly step, not folding step — key difference from
                   BBS6/MKKS LOF);
                 In BBS2-null cells: anti-BBS7 IF is absent (same mutual
                   destabilisation); anti-MKKS IF: NORMAL.
                 Both BBS2 and BBS7 LOF → absent BBS2 IF + absent BBS7 IF
                   + NORMAL MKKS/BBS10/BBS12 IF — distinguishes BBS2/BBS7
                   from BBS6 (where MKKS IF is absent and BBS10/12 abnormal).
               BCC folding connection: the BBSome Chaperonin Complex
               (MKKS-BBS10-BBS12) co-folds BBS7 WD40 beta-propeller
               alongside BBS2 during translation — in MKKS-null cells,
               BBS7 protein is also absent/reduced (same as BBS2); but
               in BBS7-null cells, MKKS protein is normal.
Disease        : Bardet-Biedl Syndrome (#209900) — BBS7 subtype
Frequency      : ~2–4% of all BBS (rarer than BBS2 at 10–20%)
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~7% of BBS7 families (BBS2 most common third allele —
                 core dimer partner modifier; also BBS9)
Cohort         : 40 patients; educational/synthetic; seed 345
Endpoints      : /api/bbs7/overview | /api/bbs7/breakdown | /api/bbs7/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 345
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("European",      0.38),
        ("MENA",          0.24),
        ("South Asian",   0.19),
        ("East Asian",    0.09),
        ("African",       0.06),
        ("Other",         0.04),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "European":    ["p.Pro532Arg (c.1595C>G)", "p.Trp375Ter (c.1124G>A)", "p.Leu523Pro (c.1568T>C)"],
        "MENA":        ["p.Arg184Gln (c.551G>A)", "p.Trp375Ter (c.1124G>A)", "p.Pro532Arg (c.1595C>G)"],
        "South Asian": ["p.Leu523Pro (c.1568T>C)", "p.Arg184Gln (c.551G>A)", "p.Trp375Ter (c.1124G>A)"],
        "East Asian":  ["p.Pro532Arg (c.1595C>G)", "p.Trp375Ter (c.1124G>A)", "p.Arg184Gln (c.551G>A)"],
        "African":     ["p.Trp375Ter (c.1124G>A)", "p.Leu523Pro (c.1568T>C)"],
        "Other":       ["p.Trp375Ter (c.1124G>A)", "p.Pro532Arg (c.1595C>G)"],
    }

    allele_classes = ["Missense/Missense", "Missense/Truncating", "Truncating/Truncating",
                      "Splice/Missense", "Splice/Truncating"]
    allele_weights = [0.33, 0.33, 0.20, 0.09, 0.05]

    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.15, 0.35, 0.30, 0.20]

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.36, 0.13, 0.10, 0.04, 0.37]

    renal_types = ["Structural anomaly (horseshoe/duplex)", "Cystic dysplasia",
                   "Tubulointerstitial nephropathy", "ESRD", "None"]
    renal_w = [0.14, 0.11, 0.09, 0.06, 0.60]

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.18, 0.42, 0.27, 0.13]

    misdiag = ["Isolated RP / Leber amaurosis", "Obesity syndrome (Prader-Willi screen)",
               "Alström Syndrome (cone-rod confusion)", "CHARGE/Joubert", "No initial misdiagnosis"]
    mis_w   = [0.30, 0.22, 0.09, 0.07, 0.32]

    cohort = []
    for i in range(n):
        eth   = rng.choices(eth_labels, weights=eth_weights)[0]
        v_pool = variants_by_eth[eth]
        v1    = rng.choice(v_pool)
        v2    = rng.choice(v_pool)
        poly  = rng.choices(polydactyly_types, weights=poly_w)[0]
        renal = rng.choices(renal_types, weights=renal_w)[0]
        ret   = rng.choices(retinal_stages, weights=ret_w)[0]
        dx_a  = rng.choices(dx_ages, weights=dx_w)[0]
        allele = rng.choices(allele_classes, weights=allele_weights)[0]
        mis   = rng.choices(misdiag, weights=mis_w)[0]

        cohort.append({
            "id":             i + 1,
            "sex":            rng.choice(["M", "F"]),
            "ethnicity":      eth,
            "variant1":       v1,
            "variant2":       v2,
            "allele_class":   allele,
            "polydactyly":    poly != "None",
            "polydactyly_type": poly,
            "obesity":        rng.random() < 0.80,
            "bmi":            round(rng.uniform(28.0, 42.5), 1) if rng.random() < 0.80 else round(rng.uniform(21.0, 27.5), 1),
            "cognitive_ld":   rng.random() < 0.45,
            "renal_any":      renal != "None",
            "renal_type":     renal,
            "esrd":           renal == "ESRD",
            "hypogonadism":   rng.random() < 0.60,
            "anosmia":        rng.random() < 0.55,
            "chd":            rng.random() < 0.05,
            "retinal_stage":  ret,
            "retinal_endstage": ret in ["Legal blindness / end-stage"],
            "triallelic_bbs": rng.random() < 0.07,   # BBS2 3rd allele — core dimer partner
            "dx_age_group":   dx_a,
            "misdiagnosis":   mis,
            "initial_misdiagnosis": mis != "No initial misdiagnosis",
            "presentation_age": rng.choices(
                ["Birth (polydactyly noted)", "Infancy (nystagmus/strabismus)",
                 "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"],
                weights=[0.22, 0.18, 0.28, 0.22, 0.10]
            )[0],
        })
    return cohort


_COHORT = _make_cohort()


def _pct(n: int, total: int = _N) -> float:
    return round(100.0 * n / total, 1)


# ── API ENDPOINT FUNCTIONS ────────────────────────────────────────────────────

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
        "cohort_n": n,
        "seed":     _SEED,
        "gene":     "BBS7",
        "omim_gene": "*607590",
        "chromosome": "4q27",
        "protein_aa": 672,
        "disease_omim": "#209900",
        "bbs_subtype": "BBS7",
        "inheritance": "Autosomal Recessive (biallelic LOF)",
        "frequency_pct_bbs": "2–4%",
        "mechanism_class": "BBSome Core Structural Subunit (BBS2-BBS7 obligate heterodimer)",
        "kpis": {
            "polydactyly_n":      poly_n,
            "polydactyly_pct":    _pct(poly_n),
            "obesity_n":          obesity_n,
            "obesity_pct":        _pct(obesity_n),
            "cognitive_n":        cognitive_n,
            "cognitive_pct":      _pct(cognitive_n),
            "renal_any_n":        renal_any_n,
            "renal_any_pct":      _pct(renal_any_n),
            "hypogonadism_n":     hypo_n,
            "hypogonadism_pct":   _pct(hypo_n),
            "anosmia_n":          anosmia_n,
            "anosmia_pct":        _pct(anosmia_n),
            "chd_n":              chd_n,
            "chd_pct":            _pct(chd_n),
            "retinal_endstage_n": ret_end_n,
            "retinal_endstage_pct": _pct(ret_end_n),
            "triallelic_n":       tri_n,
            "triallelic_pct":     _pct(tri_n),
            "misdiagnosis_n":     mis_n,
            "misdiagnosis_pct":   _pct(mis_n),
            "esrd_n":             esrd_n,
        },
        "mechanism": (
            "BBS7 (672 aa; Chr 4q27) is a WD40 beta-propeller protein homologous to BBS2 (also called "
            "BBS2L1 — BBS2-like protein 1). BBS7 is a STRUCTURAL BBSome SUBUNIT and the obligate dimer "
            "partner of BBS2 within the BBSome octamer core module. The BBS2-BBS7 heterodimer forms the "
            "structural spine of the BBSome: BBS2 and BBS7 interlock their N-terminal coiled-coil domains "
            "and WD40 beta-propellers to form a stable heterodimer that serves as the nucleation scaffold "
            "for the full BBSome octamer (Step 0 of BBSome assembly). BBS9 acts as the bridge subunit that "
            "contacts both BBS2 and BBS7 WD40 propellers simultaneously, stabilising the core dimer and "
            "enabling recruitment of the peripheral module (BBS4-BBS5-BBS8-BBIP10). BBS7 LOF → BBS2-BBS7 "
            "core dimer cannot form → BBS9 has no scaffold to bridge → BBSome octamer assembly fails at "
            "Step 0 → ciliary GPCR cargo (LepR, SSTR3, D1R, Smo) mis-traffics → rod-cone dystrophy, "
            "leptin resistance, anosmia, polydactyly, hypogonadism, renal cysts. Mutual destabilisation: "
            "in BBS7-null cells, BBS2 protein is also absent/reduced (rapid proteasomal degradation of "
            "orphaned BBS2 without BBS7 partner); in BBS2-null cells, BBS7 protein is also absent. This "
            "mutual co-dependence is the hallmark of the BBS2-BBS7 obligate heterodimer. The BBSome "
            "Chaperonin Complex (BCC: MKKS-BBS10-BBS12) folds BBS7 alongside BBS2 during translation — "
            "BBS7 is a BCC substrate alongside BBS2."
        ),
        "key_distinction": (
            "BBS7 is the obligate WD40 beta-propeller dimer partner of BBS2 — both are core structural "
            "BBSome subunits sharing the same assembly step (Step 0: core dimer). "
            "BBS1: BBSome cargo-recognition subunit (IFT-A linker; M390R founder allele; 25% of BBS). "
            "BBS2: BBS7's core dimer partner — 10–20% of BBS; R631P Bedouin founder allele. "
            "BBS3 (ARL6): external GTPase recruiter — BBSome intact in BBS3 LOF; different step. "
            "BBS4: PCM1 satellite tethering — later assembly step. "
            "BBS5: ciliary lipid (PtdIns) anchor — membrane docking step. "
            "BBS6 (MKKS): upstream chaperonin co-chaperone — folds BBS2 AND BBS7 before assembly. "
            "BBS7 [THIS]: structural core dimer partner of BBS2 — assembly step, not folding step. "
            "Critical cellular discriminator: anti-MKKS IF is NORMAL in BBS7 LOF (BCC chaperonin "
            "is intact; BBS7 WD40 was folded successfully by BCC; failure is at the dimer assembly "
            "step). This distinguishes BBS7 LOF from BBS6/MKKS LOF (where MKKS IF is absent). "
            "BBS2 vs BBS7: both LOF → absent BBS2 IF + absent BBS7 IF; MKKS/BBS10/BBS12 IF normal "
            "in both — gene panel is the ONLY way to distinguish BBS2 from BBS7. "
            "Tri-allelic BBS: BBS2 is the most common third allele in BBS7 families (~7%) — "
            "reflecting the core dimer partner relationship."
        ),
        "bbs_pathway_comparison": [
            {"gene": "BBS1", "role": "BBSome cargo-recognition beta-propeller (M390R founder)",
             "function_class": "Structural BBSome Subunit", "omim": "*209901", "frequency_pct": 25},
            {"gene": "BBS2", "role": "BBSome core dimer partner of BBS7 — WD40 scaffold",
             "function_class": "Structural BBSome Subunit (Core Dimer)", "omim": "*606151", "frequency_pct": 15},
            {"gene": "BBS3 (ARL6)", "role": "GTPase recruits BBSome to ciliary membrane",
             "function_class": "External GTPase Recruiter", "omim": "*608845", "frequency_pct": 4},
            {"gene": "BBS4", "role": "TPR scaffold tethers BBSome at PCM1 satellites",
             "function_class": "PCM1 Satellite Tether", "omim": "*600374", "frequency_pct": 3},
            {"gene": "BBS5", "role": "PH domain anchors BBSome to ciliary PtdIns lipids",
             "function_class": "Ciliary Lipid Anchor", "omim": "*603650", "frequency_pct": 2},
            {"gene": "BBS6 / MKKS", "role": "Chaperonin-like co-chaperone folds BBS2/BBS7/BBS9",
             "function_class": "Upstream Chaperonin Co-Chaperone", "omim": "*604896", "frequency_pct": "5–8"},
            {"gene": "BBS7 [THIS]", "role": "Obligate WD40 dimer partner of BBS2 — core module scaffold",
             "function_class": "Structural BBSome Subunit (Core Dimer)", "omim": "*607590", "frequency_pct": "2–4"},
            {"gene": "BBS9 (PTHB1)", "role": "Bridge subunit contacts BBS2-BBS7 dimer and BBS1-BBS4-BBS5 module",
             "function_class": "BBSome Bridge Subunit", "omim": "*607968", "frequency_pct": 2},
        ],
        "retinal_distribution": [
            {"label": s, "n": ret_stages.get(s, 0)}
            for s in ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
        ],
        "age_distribution": {
            "dx_0_4yr":    age_dist.get("0–4 yr", 0),
            "dx_5_11yr":   age_dist.get("5–11 yr", 0),
            "dx_12_17yr":  age_dist.get("12–17 yr", 0),
            "dx_18plus":   age_dist.get("18+ yr", 0),
        },
    }


def get_breakdown() -> Dict[str, Any]:
    c = _COHORT
    n = _N

    systemic = [
        ("Rod-cone dystrophy (rod-first; 100%)",        n,   100.0),
        ("Post-axial polydactyly",                       sum(1 for p in c if p["polydactyly"]),       _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Obesity (BMI ≥ 28)",                           sum(1 for p in c if p["obesity"]),            _pct(sum(1 for p in c if p["obesity"]))),
        ("Hypogonadism",                                 sum(1 for p in c if p["hypogonadism"]),       _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Anosmia/Hyposmia",                             sum(1 for p in c if p["anosmia"]),            _pct(sum(1 for p in c if p["anosmia"]))),
        ("Cognitive/Learning Disability",                sum(1 for p in c if p["cognitive_ld"]),       _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("Renal Anomaly (any)",                          sum(1 for p in c if p["renal_any"]),          _pct(sum(1 for p in c if p["renal_any"]))),
        ("Congenital Heart Disease (CHD)",               sum(1 for p in c if p["chd"]),                _pct(sum(1 for p in c if p["chd"]))),
        ("ESRD",                                         sum(1 for p in c if p["esrd"]),               _pct(sum(1 for p in c if p["esrd"]))),
        ("Tri-allelic BBS (BBS2/BBS9 3rd allele)",      sum(1 for p in c if p["triallelic_bbs"]),     _pct(sum(1 for p in c if p["triallelic_bbs"]))),
        ("Initial misdiagnosis",                         sum(1 for p in c if p["initial_misdiagnosis"]), _pct(sum(1 for p in c if p["initial_misdiagnosis"]))),
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
            "Gene":              "BBS7 (BBS2-like protein 1; BBS2L1); Chr 4q27",
            "OMIM_Gene":         "*607590",
            "Chromosome":        "4q27",
            "Protein":           "672 aa — WD40 beta-propeller (7 WD40 repeats); obligate BBS2-dimer partner",
            "Domains":           "N-coiled-coil/BBS2-contact (aa 1–120) · WD40 7-bladed propeller (aa 121–600; BBS9 contact at blade 4-5 loop) · C-terminal BBIP10 contact (aa 601–672)",
            "Dimer_partner":     "BBS2 (obligate heterodimer — BBS2-BBS7 core module; assembly Step 0)",
            "BCC_substrate":     "YES — BBS7 WD40 barrel co-folded by MKKS-BBS10-BBS12 (BCC) alongside BBS2 during translation",
            "BBS9_contact":      "Pro532-loop (WD40 repeat 6-7): BBS9 bridge contact site — loss orphans BBS2-BBS7 dimer from octamer",
            "Inheritance":       "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":     "2–4% of all BBS (rarer than BBS2 at 10–20%)",
            "Allele_spectrum":   "Missense/missense ≈ missense/truncating > truncating/truncating; no dominant ethnic founder",
            "Triallelic_BBS":    "~7% (BBS2 most common third allele — dimer partner modifier; also BBS9 bridge subunit)",
            "CHD_penetrance":    "~5% (similar to BBS2; lower than BBS6/MKKS 8%)",
            "No_McKusick":       "No McKusick-Kaufman overlap — that spectrum is MKKS/BBS6 exclusive (different gene, chaperonin role)",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 7 (BBS7) — #209900",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS7 subtype ~2–4%",
            "Cardinal_features": "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR mis-traffic) · Hypogonadism · Renal anomalies · Cognitive/LD · Anosmia",
            "BBS7_vs_BBS2":      "Clinically indistinguishable from BBS2 — gene panel mandatory; no phenotypic biomarker separates them",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; identical to BBS1/2/3/4/5/6)",
            "Obesity_mechanism": "LepR mis-trafficking (hypothalamic cilia) → leptin resistance → hyperphagia; GLP1RA applicable",
            "Renal":             "Structural anomalies + cystic dysplasia + TIN; ESRD ~6%; cilia-dependent tubulogenesis defect",
            "CHD":               "~5%; similar to BBS2; no specific cardiac morphogenesis role beyond ciliary mechanism",
            "Diagnosis":         "Full BBS gene panel (20–24 genes); fibroblast IF: absent BBS7 + absent BBS2 + NORMAL MKKS/BBS10/BBS12",
            "Treatment":         "Symptomatic (no curative gene therapy approved 2026); low-vision aids; GLP1RA for obesity; renal surveillance",
            "Prognosis":         "Progressive visual loss → blindness 3rd–4th decade; renal trajectory variable; survival near-normal with management",
        },
        "key_variants": [
            {
                "variant": "p.Pro532Arg (c.1595C>G)",
                "domain": "WD40 repeat 6-7 loop — BBS9 bridge contact surface (Pro532 contacts BBS9 Glu388)",
                "consequence": "BBS9 bridge lost → BBS2-BBS7 core dimer assembled but cannot recruit BBS9 → orphaned from BBSome octamer; full BBS phenotype",
                "ethnicity": "European (most common European BBS7 allele)"
            },
            {
                "variant": "p.Arg184Gln (c.551G>A)",
                "domain": "N-terminal coiled-coil / BBS2-dimer interface (Arg184 salt-bridge with BBS2 Lys156)",
                "consequence": "BBS2-BBS7 dimer interface disrupted; orphaned BBS2 degraded by proteasome; BBSome core module fails at Step 0; full severe BBS",
                "ethnicity": "MENA (enriched; pan-ethnic distribution)"
            },
            {
                "variant": "p.Leu523Pro (c.1568T>C)",
                "domain": "WD40 barrel hydrophobic core — blade 6 internal leucine (Leu523 stabilises beta-propeller fold)",
                "consequence": "BBS7 WD40 beta-propeller misfolding; BCC cannot release folded BBS7 → BBS2 also stranded; core module fails; full BBS phenotype",
                "ethnicity": "South Asian (enriched)"
            },
            {
                "variant": "p.Trp375Ter (c.1124G>A)",
                "domain": "WD40 repeat 4 — truncating null (stops at aa 375; BBS9 contact domain and C-terminus lost)",
                "consequence": "Complete null — no BBS7 protein; BBS2 rapidly degraded; BBSome core module completely absent; severe BBS phenotype; pan-ethnic",
                "ethnicity": "Pan-ethnic (most common truncating allele)"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20 genes including BBS2/BBS7 and BCC components BBS6/BBS10/BBS12); MLPA/CNV for large deletions",
            "3. BBS7-specific: Sanger confirmation; classify by domain (N-coiled-coil/dimer interface vs WD40 core vs BBS9 contact vs C-terminal BBIP10)",
            "4. Fibroblast IF pattern (key discriminator): anti-BBS7 (absent) + anti-BBS2 (absent/reduced — mutual destabilisation) + anti-MKKS (NORMAL — BCC chaperonin intact); anti-BBS10/BBS12 (NORMAL)",
            "5. Anti-MKKS IF is NORMAL in BBS7 LOF — if MKKS IF is absent, suspect BBS6 (MKKS) LOF not BBS7",
            "6. Western blot: BBS7 protein absent; BBS2 protein reduced ~65% (orphaned BBS2 degraded); MKKS protein PRESENT (normal level)",
            "7. BBS2 vs BBS7: both give absent BBS7 + absent BBS2 IF with NORMAL MKKS IF — gene panel is the ONLY discriminator",
            "8. Tri-allelic BBS: screen BBS2 (most common third allele, ~7%) and BBS9 (bridge subunit modifier)",
            "9. CHD workup: echocardiogram (CHD ~5%); ASD/VSD workup; lower priority than BBS6 (8%) but still indicated",
            "10. Renal: ultrasound + GFR; cystatin C annually; ESRD risk ~6%; ophthalmology ERG + Goldmann VF; OCT at diagnosis",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; photosensitising drug avoidance; vitamin A not indicated for BBS; gene therapy investigational",
            "2. Obesity: GLP-1 receptor agonists (semaglutide) — LepR mis-trafficking mechanism supports ciliary pathway benefit; bariatric surgery in refractory cases",
            "3. Renal: ACE inhibitor/ARB for CKD progression; renal transplant for ESRD (curative — no allograft recurrence); avoid nephrotoxins",
            "4. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling",
            "5. Anosmia: safety counselling (gas leaks, food spoilage); smell training investigational; smoke detectors mandatory",
            "6. Cognitive/LD: early intervention programmes; IEP; neuropsychological assessment; occupational therapy",
            "7. CHD (~5%): echo-guided cardiology follow-up; surgical/catheterisation as indicated; no specific BBS7 cardiac lesion profile",
            "8. Genetic counselling: AR inheritance; sibling recurrence 25%; prenatal/preimplantation genetic testing available",
            "9. BBS2 co-testing: BBS2 sequencing in all suspected BBS7 families (tri-allelic risk 7%; BBS2 as modifier allele)",
            "10. Surveillance: annual ophthalmology + renal + metabolic (HbA1c, lipids) + cardiac; growth/pubertal charting in children",
        ],
        "ddx_table": [
            {"disease": "BBS2 (BBS2 gene)", "key_difference": "BBS2 LOF → absent BBS2 IF + absent BBS7 IF (mutual destabilisation) + NORMAL MKKS IF — IDENTICAL pattern to BBS7 LOF; gene panel is the ONLY discriminator"},
            {"disease": "BBS6 / MKKS (MKKS gene)", "key_difference": "BBS6 LOF → absent MKKS IF + absent BBS2 IF + abnormal BBS10/BBS12 IF; BBS7 LOF → NORMAL MKKS IF — key distinguishing feature"},
            {"disease": "BBS1 (BBS1 gene)", "key_difference": "BBS1 LOF: anti-BBS1 IF absent; BBS2 IF often reduced (BBS1 role in core); M390R European founder; 25% of BBS; gene panel distinguishes"},
            {"disease": "BBS9 / PTHB1 (BBS9 gene)", "key_difference": "BBS9 is the bridge subunit that contacts BBS2-BBS7 dimer; BBS9 LOF → bridge absent but core dimer intact (anti-BBS2/BBS7 IF partly reduced); gene panel distinguishes"},
            {"disease": "BBS10 (BBS10 gene)", "key_difference": "BBS10 BCC co-chaperonin partner; 25% of BBS; anti-MKKS IF normal but anti-BBS10 absent; BCC destabilised → BBS2/BBS7/BBS9 misfolding (same upstream effect as BBS6)"},
            {"disease": "BBS12 (BBS12 gene)", "key_difference": "BBS12 BCC equatorial partner; anti-MKKS IF normal but anti-BBS12 absent; BCC destabilised; gene panel distinguishes from BBS7"},
            {"disease": "Alström Syndrome (ALMS1)", "key_difference": "Cone-rod (cone first, not rod first); no polydactyly; infantile DCM (60%); sensorineural hearing loss; normal BBS7/BBS2 IF; distinct gene"},
            {"disease": "Joubert Syndrome (JBTS)", "key_difference": "Molar tooth sign on brain MRI; cerebellar vermis hypoplasia; no polydactyly; BBS7 mutations not causative of classic JBTS"},
        ],
        "mechanism_glossary": [
            {
                "term": "BBS2-BBS7 Obligate Heterodimer (Core Module)",
                "definition": "BBS2 and BBS7 form an obligate WD40 beta-propeller heterodimer that constitutes the structural core module (Step 0 scaffold) of the BBSome octamer. Neither protein is stable without the other: in BBS7-null cells, BBS2 is rapidly degraded by the proteasome (orphaned without its partner); in BBS2-null cells, BBS7 is similarly destabilised. This mutual co-dependence means both proteins are absent by immunofluorescence in either BBS2 LOF or BBS7 LOF. The heterodimer is the obligate first step in BBSome assembly — without it, all downstream BBSome biogenesis fails."
            },
            {
                "term": "BBS9 Bridge Subunit",
                "definition": "BBS9 (PTHB1; *607968) is the central bridge subunit of the BBSome octamer. BBS9 contacts both BBS2 and BBS7 WD40 propellers simultaneously via its N-terminal PTHB domain and links the BBS2-BBS7 core dimer to the peripheral module (BBS1, BBS4, BBS5, BBS8, BBIP10). BBS7 Pro532 is a key contact with BBS9 Glu388 — Pro532Arg abolishes this bridge. In BBS7-null cells, the BBS9 bridge contact site is absent (no BBS7 scaffold), so BBS9 is also displaced from normal satellite localisation."
            },
            {
                "term": "BCC Folding of BBS7 (MKKS Connection)",
                "definition": "The BBSome Chaperonin Complex (BCC: MKKS-BBS10-BBS12) co-folds BBS7 WD40 beta-propeller during translation alongside BBS2. In MKKS-null (BBS6 LOF) cells, BBS7 protein is absent/reduced — BBS7 is a BCC substrate. Conversely, in BBS7-null cells, MKKS protein is NORMAL (BCC chaperonin is intact; BBS7 WD40 was successfully folded; the failure occurs at the post-folding dimer-assembly step, not the folding step). This is the key mechanistic discriminator: MKKS IF absent = BCC/folding failure (BBS6 LOF); MKKS IF normal = assembly failure (BBS7 or BBS2 LOF)."
            },
            {
                "term": "BBS7 vs BBS2: Gene Panel Is the Only Discriminator",
                "definition": "BBS2 LOF and BBS7 LOF produce clinically identical phenotypes (rod-cone dystrophy, polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia) with identical cellular IF patterns (absent BBS2 IF + absent BBS7 IF + normal MKKS IF). No clinical biomarker, ophthalmological parameter, or laboratory test distinguishes BBS2 from BBS7. The full BBS gene panel (minimum 20 genes) is mandatory. The distinction matters for accurate genetic counselling, family mutation testing, and future gene-specific therapies."
            },
            {
                "term": "LepR Ciliary Mis-Trafficking (Shared Obesity Mechanism)",
                "definition": "Leptin receptor (LepR) is a BBSome GPCR cargo requiring the full BBSome octamer for retrograde IFT-A-mediated removal from hypothalamic neuronal cilia. In BBS7 LOF, BBSome is absent (core module fails Step 0) → LepR constitutively retained in cilia → signal desensitisation → leptin resistance → hyperphagia and progressive obesity despite elevated plasma leptin. This mechanism is shared across all BBSome subunit LOF (BBS1, BBS2, BBS7, BBS9, etc.). GLP-1 receptor agonists partially bypass this by acting on a non-ciliary satiety pathway (GIPR/GLP1R interneurons)."
            },
            {
                "term": "Tri-allelic BBS in BBS7 Families",
                "definition": "Tri-allelic (oligogenic) BBS occurs when a patient carries biallelic pathogenic variants in one BBS gene plus a heterozygous pathogenic variant in a second BBS gene. In BBS7 families, the most common third allele is BBS2 (~7%), reflecting the functional relationship of the core dimer partners — a heterozygous BBS2 variant acts as a genetic modifier reducing the residual BBS2 protein available to partner any partially functional BBS7 variant. BBS9 is the second most common third allele (bridge subunit dependence). Tri-allelic cases typically have more severe phenotypes."
            },
        ],
    }
