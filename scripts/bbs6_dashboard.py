"""
BBS6 Bardet-Biedl Syndrome Type 6 — MKKS (Chaperonin-Like Co-Chaperone) Dashboard
=====================================================================================
Primary Gene : MKKS (*604896) — McKusick-Kaufman Syndrome gene; BBS6;
               570 aa; chaperonin-like fold — group II chaperonin alpha-subunit
               homolog with apical domain (aa 1–195), intermediate domain
               (aa 196–380), and equatorial domain (aa 381–570); Chr 20p12.
               MKKS is NOT a structural BBSome subunit — it is the founding
               member of the BBSome CHAPERONIN COMPLEX (BBS6-BBS10-BBS12),
               which facilitates co-translational folding and assembly of the
               BBSome core module subunits BBS2, BBS7, and BBS9:
                 Apical domain (aa 1–195): substrate binding — contacts BBS2
                   WD40 barrel and BBS7 WD40 barrel during nascent-chain
                   folding; Ala242 (apical-intermediate hinge) contacts BBS2
                   surface — Ala242Ser → BBS2 folding efficiency reduced 60%;
                 Intermediate domain (aa 196–380): hinge region linking apical
                   to equatorial; Ile339 sits at BBS10-contact surface — the
                   Ile339Val Amish variant retains partial BBS10 interaction
                   (hypomorphic → McKusick-Kaufman rather than full BBS);
                 Equatorial domain (aa 381–570): ATP-binding-like fold BUT
                   catalytic glutamate is substituted (Glu → Ala) → NO ATPase
                   activity; BBS12-contact surface resides here; Arg517 bridges
                   BBS12-equatorial interface — Arg517His → BBS12 docking lost
                   → chaperonin complex destabilised → BBS2/BBS7/BBS9 misfolding.
               BBSome chaperonin assembly mechanism (MKKS-dependent step):
                 MKKS-BBS10-BBS12 trimer assembles in cytoplasm → captures
                 nascent BBS2 and BBS7 WD40 chains → co-folds into native
                 WD40 beta-propeller geometry → releases BBS2-BBS7 heterodimer
                 → BBS9 folds via same trimer → BBS2-BBS7-BBS9 core module
                 competent for BBSome octamer assembly (BBS4-dependent step).
                 MKKS LOF → BBS2 and BBS7 misfold → core module (Step 0 of
                 BBSome assembly) fails → same downstream consequence as BBS2
                 LOF (BBSome catastrophic misassembly) but at an EARLIER
                 upstream folding step — making MKKS the most upstream known
                 BBSome biogenesis factor.
               Cellular discriminator from BBS2:
                 Anti-BBS2 IF in MKKS-null cells: ABSENT/REDUCED BBS2 puncta
                 (BBS2 protein is unstable/degraded when misfolded) — identical
                 to BBS2-null cells; anti-MKKS IF: absent MKKS puncta.
                 Western blot: BBS2 protein reduced ~70% in MKKS LOF vs ~90%
                 in BBS2 LOF — subtle but reproducible; MKKS protein absent.
                 Cannot distinguish MKKS vs BBS2 by IF alone — gene panel
                 mandatory. BBS10/BBS12 IF abnormal only in MKKS LOF (not BBS2).
               McKusick-Kaufman Syndrome overlap (MKKS-exclusive):
                 Severe truncating MKKS biallelic → full BBS6 (rod-cone + poly +
                 obesity + renal + cognitive + hypogonadism);
                 Ile339Val Amish hypomorphic → McKusick-Kaufman (hydrometrocolpos,
                 postaxial polydactyly, CHD — NO retinal degeneration, NO obesity
                 progression, NO renal disease); partial MKKS chaperonin activity
                 preserved → BBS2/BBS7 fold adequately for BBSome → ciliary
                 GPCR cargo trafficking partially maintained → retinal + renal
                 spared, cardiac + genital involvement persists via different
                 MKKS-dependent developmental pathway.
Disease        : Bardet-Biedl Syndrome (#209900) — BBS6 subtype (MKKS gene)
                 McKusick-Kaufman Syndrome (#236700) — allele-specific
Frequency      : ~5–8% of all BBS (3rd–5th most common after BBS1/BBS10/BBS2)
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~5% of BBS6 families (BBS10 or BBS12 third allele as modifier)
Cohort         : 40 patients; educational/synthetic; seed 343
Endpoints      : /api/bbs6/overview | /api/bbs6/breakdown | /api/bbs6/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 343
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("European",      0.38),
        ("MENA",          0.22),
        ("South Asian",   0.18),
        ("East Asian",    0.08),
        ("Amish",         0.07),  # elevated vs other BBS types (Ile339Val)
        ("African",       0.04),
        ("Other",         0.03),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "European":    ["p.Ala242Ser (c.724G>T)", "p.Glu292Ter (c.874G>T)", "p.Arg517His (c.1550G>A)"],
        "MENA":        ["p.Glu292Ter (c.874G>T)", "p.Arg517His (c.1550G>A)", "p.Ala242Ser (c.724G>T)"],
        "South Asian": ["p.Arg517His (c.1550G>A)", "p.Glu292Ter (c.874G>T)", "p.Ala242Ser (c.724G>T)"],
        "East Asian":  ["p.Ala242Ser (c.724G>T)", "p.Glu292Ter (c.874G>T)", "p.Arg517His (c.1550G>A)"],
        "Amish":       ["p.Ile339Val (c.1015A>G)", "p.Glu292Ter (c.874G>T)"],  # Amish founder
        "African":     ["p.Glu292Ter (c.874G>T)", "p.Arg517His (c.1550G>A)"],
        "Other":       ["p.Glu292Ter (c.874G>T)", "p.Ala242Ser (c.724G>T)"],
    }

    allele_classes = ["Missense/Missense", "Missense/Truncating", "Truncating/Truncating",
                      "Splice/Missense", "Splice/Truncating"]
    allele_weights = [0.34, 0.31, 0.20, 0.10, 0.05]

    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.15, 0.35, 0.30, 0.20]

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.38, 0.14, 0.09, 0.04, 0.35]

    renal_types = ["Structural anomaly (horseshoe/duplex)", "Cystic dysplasia",
                   "Tubulointerstitial nephropathy", "ESRD", "None"]
    renal_w = [0.16, 0.10, 0.09, 0.07, 0.58]

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.20, 0.40, 0.28, 0.12]

    misdiag = ["Isolated RP / Leber amaurosis", "Obesity syndrome (Prader-Willi screen)",
               "McKusick-Kaufman syndrome", "CHARGE/Alström", "No initial misdiagnosis"]
    mis_w   = [0.30, 0.22, 0.15, 0.08, 0.25]  # McKusick-Kaufman misdiagnosis unique to BBS6

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
            "chd":            rng.random() < 0.08,         # elevated vs BBS1-5
            "retinal_stage":  ret,
            "retinal_endstage": ret in ["Legal blindness / end-stage"],
            "triallelic_bbs": rng.random() < 0.05,
            "dx_age_group":   dx_a,
            "misdiagnosis":   mis,
            "initial_misdiagnosis": mis != "No initial misdiagnosis",
            "presentation_age": rng.choices(
                ["Birth (polydactyly noted)", "Infancy (nystagmus/strabismus)",
                 "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"],
                weights=[0.25, 0.18, 0.25, 0.22, 0.10]
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

    ret_stages  = {}
    for p in c:
        s = p["retinal_stage"]
        ret_stages[s] = ret_stages.get(s, 0) + 1

    age_dist = {}
    for p in c:
        g = p["dx_age_group"]
        age_dist[g] = age_dist.get(g, 0) + 1

    return {
        "cohort_n": n,
        "seed":     _SEED,
        "gene":     "MKKS",
        "omim_gene": "*604896",
        "chromosome": "20p12",
        "protein_aa": 570,
        "disease_omim": "#209900",
        "bbs_subtype": "BBS6",
        "inheritance": "Autosomal Recessive (biallelic LOF)",
        "frequency_pct_bbs": "5–8%",
        "mechanism_class": "BBSome Chaperonin Co-Chaperone (upstream folding)",
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
            "MKKS (BBS6) encodes a 570 aa chaperonin-like protein homologous to the group II "
            "chaperonin alpha-subunit. Unlike classical chaperonins, MKKS lacks ATPase activity "
            "(catalytic Glu → Ala substitution in equatorial domain) but retains substrate-binding "
            "capacity via its apical domain. MKKS forms the BBSome Chaperonin Complex (BCC) with "
            "BBS10 and BBS12 — three paralogous group-II chaperonin-like proteins that co-operate "
            "to fold the BBSome core subunits BBS2 and BBS7 (WD40 beta-propeller proteins) and "
            "BBS9. MKKS LOF → BCC destabilised → BBS2/BBS7/BBS9 misfold and are degraded by the "
            "proteasome → BBSome core module (BBS2-BBS7 dimer + BBS9 bridge, Step 0) cannot form → "
            "BBSome octamer cannot assemble → ciliary GPCR cargo (LepR, SSTR3, D1R, Smo) mis-traffics "
            "→ rod-cone dystrophy, leptin resistance, anosmia, polydactyly, hypogonadism, renal cysts. "
            "This failure is UPSTREAM of BBS2-itself LOF: same endpoint (BBSome catastrophic "
            "misassembly) but at the folding step rather than the structural-subunit step."
        ),
        "key_distinction": (
            "BBS6 (MKKS) is the most upstream BBSome biogenesis factor known. "
            "BBS1 and BBS2: structural BBSome subunits (cargo recognition + core scaffold). "
            "BBS3 (ARL6): external GTPase recruiter. BBS4: PCM1 satellite tethering TPR scaffold. "
            "BBS5: PH domain ciliary membrane lipid anchor. "
            "BBS6 (MKKS): chaperonin-like co-chaperone — folds BBS2/BBS7/BBS9 BEFORE assembly begins. "
            "Cellular IF: anti-BBS2 signal absent/reduced in MKKS-null cells (BBS2 protein degraded); "
            "anti-BBS10 and anti-BBS12 IF also abnormal (BCC partner instability) — unique to MKKS LOF. "
            "McKusick-Kaufman overlap: hypomorphic MKKS alleles (Ile339Val Amish) → MKKS syndrome "
            "(hydrometrocolpos, polydactyly, CHD) without retinal/renal BBS — partial BCC function "
            "maintains sufficient BBSome for ciliary GPCR traffic but not cardiac/genital development. "
            "CHD penetrance (8%) is highest among BBS1-6 — reflects MKKS developmental overlap."
        ),
        "bbs_pathway_comparison": [
            {"gene": "BBS1", "role": "BBSome cargo-recognition beta-propeller (M390R founder)",
             "function_class": "Structural BBSome Subunit", "omim": "*209901", "frequency_pct": 25},
            {"gene": "BBS2", "role": "BBSome core scaffold WD40 dimer with BBS7",
             "function_class": "Structural BBSome Subunit", "omim": "*606151", "frequency_pct": 15},
            {"gene": "BBS3 (ARL6)", "role": "GTPase recruits BBSome to ciliary membrane",
             "function_class": "External GTPase Recruiter", "omim": "*608845", "frequency_pct": 4},
            {"gene": "BBS4", "role": "TPR scaffold tethers BBSome at PCM1 satellites",
             "function_class": "PCM1 Satellite Tether", "omim": "*600374", "frequency_pct": 3},
            {"gene": "BBS5", "role": "PH domain anchors BBSome to ciliary PtdIns lipids",
             "function_class": "Ciliary Lipid Anchor", "omim": "*603650", "frequency_pct": 2},
            {"gene": "BBS6 / MKKS [THIS]", "role": "Chaperonin-like co-chaperone folds BBS2/BBS7/BBS9",
             "function_class": "Upstream Chaperonin Co-Chaperone", "omim": "*604896", "frequency_pct": "5–8"},
            {"gene": "BBS10", "role": "BCC partner; chaperonin-like; co-folds BBS core subunits",
             "function_class": "BCC Co-Chaperone", "omim": "*610148", "frequency_pct": 25},
            {"gene": "BBS12", "role": "BCC partner; chaperonin-like; equatorial BCC anchor",
             "function_class": "BCC Co-Chaperone", "omim": "*610683", "frequency_pct": 5},
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

    # Systemic burden table
    systemic = [
        ("Rod-cone dystrophy (rod-first; 100%)",          n,   100.0),
        ("Post-axial polydactyly",                         sum(1 for p in c if p["polydactyly"]),       _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Obesity (BMI ≥ 28)",                             sum(1 for p in c if p["obesity"]),            _pct(sum(1 for p in c if p["obesity"]))),
        ("Hypogonadism",                                   sum(1 for p in c if p["hypogonadism"]),       _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Anosmia/Hyposmia",                               sum(1 for p in c if p["anosmia"]),            _pct(sum(1 for p in c if p["anosmia"]))),
        ("Cognitive/Learning Disability",                  sum(1 for p in c if p["cognitive_ld"]),       _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("Renal Anomaly (any)",                            sum(1 for p in c if p["renal_any"]),          _pct(sum(1 for p in c if p["renal_any"]))),
        ("Congenital Heart Disease (CHD; MKKS overlap)",  sum(1 for p in c if p["chd"]),                _pct(sum(1 for p in c if p["chd"]))),
        ("ESRD",                                           sum(1 for p in c if p["esrd"]),               _pct(sum(1 for p in c if p["esrd"]))),
        ("Tri-allelic BBS (BBS10/BBS12 3rd allele)",      sum(1 for p in c if p["triallelic_bbs"]),     _pct(sum(1 for p in c if p["triallelic_bbs"]))),
        ("Initial misdiagnosis",                           sum(1 for p in c if p["initial_misdiagnosis"]), _pct(sum(1 for p in c if p["initial_misdiagnosis"]))),
    ]

    # Ethnicity
    eth_counts: Dict[str, int] = {}
    for p in c:
        eth_counts[p["ethnicity"]] = eth_counts.get(p["ethnicity"], 0) + 1

    # Allele class
    ac_counts: Dict[str, int] = {}
    for p in c:
        ac_counts[p["allele_class"]] = ac_counts.get(p["allele_class"], 0) + 1

    # Retinal stage
    ret_counts: Dict[str, int] = {}
    for p in c:
        ret_counts[p["retinal_stage"]] = ret_counts.get(p["retinal_stage"], 0) + 1

    # Renal
    renal_counts: Dict[str, int] = {}
    for p in c:
        renal_counts[p["renal_type"]] = renal_counts.get(p["renal_type"], 0) + 1

    # Polydactyly
    poly_counts: Dict[str, int] = {}
    for p in c:
        poly_counts[p["polydactyly_type"]] = poly_counts.get(p["polydactyly_type"], 0) + 1

    # Presentation age
    pres_counts: Dict[str, int] = {}
    for p in c:
        pres_counts[p["presentation_age"]] = pres_counts.get(p["presentation_age"], 0) + 1

    # Misdiagnosis
    mis_counts: Dict[str, int] = {}
    for p in c:
        mis_counts[p["misdiagnosis"]] = mis_counts.get(p["misdiagnosis"], 0) + 1

    # Top variants
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
                      "McKusick-Kaufman syndrome", "CHARGE/Alström", "No initial misdiagnosis"]
        ],
        "top_variants": [{"variant": v, "n": cnt} for v, cnt in top_variants],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":              "MKKS (McKusick-Kaufman Syndrome gene); also called BBS6",
            "OMIM_Gene":         "*604896",
            "Chromosome":        "20p12",
            "Protein":           "570 aa — chaperonin-like (group II chaperonin alpha-subunit homolog)",
            "Domains":           "Apical (aa 1–195): substrate/BBS2 binding · Intermediate (aa 196–380): BBS10 contact · Equatorial (aa 381–570): BBS12 contact; no ATPase activity",
            "Complex":           "BBSome Chaperonin Complex (BCC): MKKS + BBS10 + BBS12",
            "BCC_substrates":    "BBS2, BBS7, BBS9 (co-translational WD40 barrel folding)",
            "ATPase":            "ABSENT — catalytic Glu replaced by Ala in equatorial domain (pseudoenzyme)",
            "Inheritance":       "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":     "5–8% of all BBS (3rd–5th most common)",
            "Allele_spectrum":   "Missense > missense/truncating > truncating/truncating; Ile339Val Amish hypomorphic → McKusick-Kaufman",
            "McKusick_overlap":  "Hypomorphic alleles → McKusick-Kaufman Syndrome (#236700): hydrometrocolpos, polydactyly, CHD, NO retinal/renal BBS",
            "CHD_penetrance":    "~8% — highest among BBS1-6; reflects MKKS developmental role in cardiac morphogenesis",
            "Triallelic_BBS":    "~5% (BBS10 or BBS12 as third allele — BCC partner modifier)",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 6 (BBS6) — #209900; allele overlap with McKusick-Kaufman Syndrome #236700",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS6 subtype ~5–8%",
            "Cardinal_features": "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR mis-traffic) · Hypogonadism · Renal anomalies · Cognitive/LD",
            "MKKS_exclusive":    "McKusick-Kaufman phenotype spectrum with hypomorphic alleles; highest CHD rate (8%) in BBS1-6",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; identical pattern to BBS1-5)",
            "Obesity_mechanism": "LepR mis-trafficking (hypothalamic cilia) → leptin resistance → hyperphagia; GLP1RA applicable",
            "Renal":             "Structural anomalies + cystic dysplasia + TIN; ESRD ~7%; same mechanism as NPHP (cilia-dependent tubulogenesis)",
            "CHD":               "~8%; ASD/VSD/pulmonary stenosis — reflects MKKS role in cardiac cushion morphogenesis (non-ciliary pathway)",
            "Diagnosis":         "Full BBS gene panel (20–24 genes); MKKS sequencing + MLPA; anti-BBS2 IF in fibroblasts (absent); anti-BBS10/12 IF abnormal",
            "Treatment":         "Symptomatic (no curative gene therapy approved 2026); low-vision aids; anti-obesity (GLP1RA); renal surveillance; CHD cardiology",
            "Prognosis":         "Progressive visual loss → blindness 3rd–4th decade; renal trajectory variable; survival near-normal with management",
        },
        "key_variants": [
            {
                "variant": "p.Ala242Ser (c.724G>T)",
                "domain": "Apical domain — BBS2 WD40 contact hinge (Ala242 at apical-intermediate interface)",
                "consequence": "BBS2 folding efficiency reduced ~60%; partial BBSome assembly deficit; moderate-severe BBS",
                "ethnicity": "European (most common European BBS6 allele)"
            },
            {
                "variant": "p.Ile339Val (c.1015A>G)",
                "domain": "Intermediate domain — BBS10-contact surface",
                "consequence": "Hypomorphic; partial BCC function retained → McKusick-Kaufman (NOT full BBS); Amish founder variant",
                "ethnicity": "Amish (founder); rare elsewhere"
            },
            {
                "variant": "p.Arg517His (c.1550G>A)",
                "domain": "Equatorial domain — BBS12-docking interface (Arg517 salt-bridge BBS12-Glu)",
                "consequence": "BCC destabilised at BBS12 contact; BBS2/BBS7 folding failed; full BBS phenotype",
                "ethnicity": "South Asian / MENA"
            },
            {
                "variant": "p.Glu292Ter (c.874G>T)",
                "domain": "Intermediate domain — truncating null (stops at aa 292; equatorial domain lost)",
                "consequence": "Complete null — no BCC function; full severe BBS phenotype; pan-ethnic",
                "ethnicity": "Pan-ethnic (most common truncating allele)"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20 genes; include BBS10/BBS12 for BCC co-chaperonins); MLPA/CNV for large deletions",
            "3. MKKS-specific: Sanger confirmation of identified variants; classify by domain (apical vs intermediate vs equatorial)",
            "4. Hypomorphic MKKS (Ile339Val): consider McKusick-Kaufman spectrum — look for hydrometrocolpos (females), CHD; retinal/renal may be absent",
            "5. Fibroblast IF: anti-BBS2 (absent/reduced = BCC defect; same in BBS2 LOF); anti-BBS10 + anti-BBS12 (abnormal = MKKS LOF specifically); MKKS protein absent",
            "6. Western blot: BBS2 protein reduced ~70% (BBS2 unstable when misfolded); MKKS absent",
            "7. CHD workup: echocardiogram mandatory (highest CHD rate 8% among BBS1-6) — ASD/VSD/PS",
            "8. Renal: renal ultrasound + GFR; cystatin C annually; ESRD risk 7%",
            "9. Ophthalmology: full-field ERG + Goldmann VF; OCT at diagnosis; low-vision referral",
            "10. Tri-allelic BBS: screen BBS10 and BBS12 for third modifier allele (~5% of BBS6 families)",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; avoid photosensitising drugs; vitamin A not indicated for BBS (no evidence); gene therapy investigational only",
            "2. Obesity: GLP-1 receptor agonists (semaglutide) — LepR mis-trafficking mechanism supports ciliary pathway benefit; bariatric surgery in refractory cases",
            "3. CHD (8%): cardiac surgery/catheterisation as indicated; lifelong cardiology follow-up",
            "4. Renal: ACE inhibitor/ARB for CKD; renal transplant for ESRD (curative — no allograft recurrence); avoid nephrotoxins",
            "5. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment",
            "6. Anosmia: safety counselling (gas leaks, food spoilage); smell training investigational",
            "7. Cognitive/LD: early intervention programmes; IEP; neuropsychological assessment",
            "8. McKusick-Kaufman spectrum (Ile339Val): hydrometrocolpos surgical repair; polydactyly surgical correction; cardiac management",
            "9. Genetic counselling: AR inheritance; sibling recurrence 25%; prenatal/preimplantation genetic testing available",
            "10. Surveillance: annual ophthalmology + renal + metabolic (HbA1c, lipids) + cardiac (if CHD); growth/pubertal charting in children",
        ],
        "ddx_table": [
            {"disease": "BBS2 (BBS2 gene)", "key_difference": "BBS2 LOF → same absent anti-BBS2 IF; BBS2 protein absent; MKKS protein PRESENT (normal); BBS10/BBS12 IF normal; gene panel distinguishes"},
            {"disease": "BBS10 (BBS10 gene)", "key_difference": "BBS10 most common BBS (25%); same BCC co-chaperonin; anti-MKKS IF normal; anti-BBS10 absent; gene panel"},
            {"disease": "BBS12 (BBS12 gene)", "key_difference": "BBS12 BCC partner; anti-MKKS IF normal; anti-BBS12 absent; equatorial domain contact lost (same as Arg517His MKKS)"},
            {"disease": "McKusick-Kaufman Syndrome (#236700)", "key_difference": "Same gene (MKKS) — hypomorphic alleles (Ile339Val Amish); hydrometrocolpos + polydactyly + CHD; NO retinal degeneration, NO renal progression, NO obesity"},
            {"disease": "BBS4 (BBS4 gene)", "key_difference": "BBS4 LOF: anti-BBS4 IF absent (PCM1 puncta absent); MKKS/BBS2 IF normal; different assembly step (satellite tethering)"},
            {"disease": "BBS5 (BBS5 gene)", "key_difference": "BBS5 LOF: anti-BBS4 IF NORMAL; BBS2 IF normal; MKKS IF normal; PH domain lipid anchor defect — distinguishable by fibroblast IF pattern"},
            {"disease": "Alström Syndrome (ALMS1)", "key_difference": "Cone-rod (cone first, not rod first); no polydactyly; DCM infantile onset (60%); sensorineural hearing loss; normal MKKS/BBS2 IF"},
            {"disease": "Joubert Syndrome (JBTS)", "key_difference": "Molar tooth sign on brain MRI; cerebellar vermis hypoplasia; no polydactyly; MKKS mutations not typically causative of JBTS"},
        ],
        "mechanism_glossary": [
            {
                "term": "BBSome Chaperonin Complex (BCC)",
                "definition": "Cytoplasmic trimeric complex of BBS6 (MKKS), BBS10, and BBS12 — all group II chaperonin-like proteins (GroEL/CCT homologs without ATPase activity). The BCC co-folds BBSome core subunits BBS2, BBS7, and BBS9 before BBSome assembly begins. Loss of any BCC component → misfolding of core subunits → BBSome biogenesis failure at Step 0."
            },
            {
                "term": "Group II Chaperonin-Like Proteins",
                "definition": "BBS6, BBS10, and BBS12 are paralogs of the archaeal/eukaryotic group II chaperonin (CCT/TRiC) alpha-subunit. They retain the apical substrate-binding domain and intermediate hinge but lack ATPase activity (equatorial catalytic Glu substituted). They function as pseudo-chaperonins that capture and stabilise WD40 beta-propeller nascent chains (BBS2, BBS7, BBS9) in a folding-competent state."
            },
            {
                "term": "McKusick-Kaufman Syndrome (MKKS) / BBS Allele Spectrum",
                "definition": "MKKS hypomorphic alleles (especially Ile339Val in Amish) → partial BCC activity → sufficient BBSome assembly for ciliary GPCR traffic (retinal/renal/metabolic spared) but insufficient for cardiac cushion and genitourinary morphogenesis → McKusick-Kaufman phenotype (hydrometrocolpos, polydactyly, CHD). Severe truncating MKKS alleles → complete BCC loss → full BBS6 phenotype. The MKKS locus exhibits the largest allele-phenotype spectrum in the BBS gene family."
            },
            {
                "term": "Anti-BBS2 IF Fibroblast Test",
                "definition": "In patient fibroblasts, anti-BBS2 immunofluorescence is absent or markedly reduced in both BBS2 LOF and MKKS (BBS6) LOF — because BBS2 protein is degraded when misfolded. Anti-MKKS IF is absent only in MKKS LOF (not BBS2). Anti-BBS10 and anti-BBS12 IF are abnormal in MKKS LOF (BCC partner instability). This pattern (absent BBS2 + absent MKKS + abnormal BBS10/12) is the cellular fingerprint of BBS6."
            },
            {
                "term": "LepR Ciliary Mis-Trafficking (Obesity Mechanism)",
                "definition": "Leptin receptor (LepR) is a ciliary GPCR cargo of the BBSome. In BBSome-deficient neurons (all BBS subtypes including BBS6), LepR cannot be removed from the ciliary membrane on demand (failed retrograde IFT-A export), leading to constitutive ciliary LepR signalling that paradoxically causes leptin resistance (signal desensitisation). Result: hyperphagia and progressive obesity despite elevated plasma leptin. GLP-1 receptor agonists partially bypass this by acting on a distinct satiety pathway."
            },
            {
                "term": "CHD in BBS6 vs BBS1-5",
                "definition": "MKKS (BBS6) has the highest CHD penetrance (~8%) among BBS subtypes 1-6. This reflects a non-ciliary developmental role of MKKS in cardiac cushion morphogenesis (atrioventricular canal development) that is distinct from its BCC chaperonin function. BBS1-5 CHD rates are 2-7%. The elevated CHD in BBS6 is a key clinical feature linking BBS6 to McKusick-Kaufman Syndrome CHD phenotype even in full BBS cases."
            },
        ],
    }
