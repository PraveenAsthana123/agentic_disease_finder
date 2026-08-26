"""
BBS9 Bardet-Biedl Syndrome Type 9 — PTHB1 (BBSome Bridge Subunit) Dashboard
==============================================================================
Primary Gene : PTHB1 (*607968) — Parathyroid Hormone-responsive B9 protein
               domain 1; also called BBS9 or BBCP; 495 aa; Chr 7p14.3.
               PTHB1 is the unique BRIDGE SUBUNIT of the BBSome octamer — the
               only subunit that simultaneously contacts both the CORE MODULE
               (BBS2-BBS7 WD40 dimer) and the PERIPHERAL MODULE (BBS1-BBS4-
               BBS5-BBS8/TTC8). Without BBS9, the two independently-assembled
               half-complexes cannot join to form the complete octamer.
               Domain organisation (495 aa):
                 N-terminal arm (aa 1–110): contacts BBS2 WD40 propeller blade
                   3–4 and BBS7 WD40 propeller blade 1–2; Ala217 in this arm
                   hydrogen-bonds BBS7 blade-2 loop — Ala217Pro disrupts BBS7
                   docking geometry; Pro189 contacts BBS7 CC domain at the
                   BBS2-BBS7 dimer surface;
                 Central bridge domain (aa 111–310): the unique bridging scaffold;
                   Arg336 contacts BBS1 beta-propeller blade 5 (cargo-recognition
                   subunit) — Arg336Trp severs BBS1-BBS9 contact; Leu303 is a
                   hydrophobic core residue; Leu303Pro destabilises the bridge
                   fold; this domain has no homology to other BBSome subunits —
                   BBS9 is the only subunit with a dedicated bridge domain;
                 C-terminal adapter (aa 311–495): contacts BBS5 PH domain and
                   BBS8 TPR4-6; Glu412Ter truncating null → entire C-terminal
                   adapter lost; BBS5 and BBS8 peripheral contacts abolished;
                   full BBS phenotype.
               PTHB1 LOF → core module (BBS2-BBS7 dimer) and peripheral module
               (BBS1-BBS4-BBS5-BBS8) each form independently but CANNOT DOCK
               to each other → no complete BBSome octamer → GPCR cargoes
               (LepR, SSTR3, D1R, Smo) mis-trafficked in cilia → full BBS.
               Key cellular discriminator from other BBS types:
                 In BBS9-null cells: anti-BBS9 IF: absent; anti-BBS2 IF:
                   REDUCED (BBS2-BBS7 dimer forms but is less stable without
                   the bridge); anti-BBS8 IF: REDUCED (peripheral module
                   mis-localised); anti-MKKS IF: NORMAL (BCC chaperonin intact;
                   BBS2/BBS7 fold correctly but bridge is missing);
                 Key discriminator from BBS2/BBS7 LOF: BBS6/MKKS IF is NORMAL
                   in BBS9 null — BCC is intact; BBS2 forms but is REDUCED,
                   not absent — sub-complex present without bridge;
                 Key discriminator from BBS8 LOF: In BBS8 LOF, BBS9 IF is
                   NORMAL (bridge is present); in BBS9 LOF, BBS9 is absent.
Disease        : Bardet-Biedl Syndrome (#209900) — BBS9 subtype
Frequency      : ~2% of all BBS (rare; <200 families worldwide 2026)
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~6% of BBS9 families (BBS2 most common third allele —
                 core dimer partner; BBS8 second — peripheral module partner)
Cohort         : 40 patients; educational/synthetic; seed 349
Endpoints      : /api/bbs9/overview | /api/bbs9/breakdown | /api/bbs9/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 349
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("European",      0.36),
        ("MENA",          0.26),
        ("South Asian",   0.20),
        ("East Asian",    0.09),
        ("African",       0.05),
        ("Other",         0.04),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "European":    ["p.Ala217Pro (c.649G>C)", "p.Leu303Pro (c.908T>C)", "p.Glu412Ter (c.1234G>T)"],
        "MENA":        ["p.Arg336Trp (c.1006C>T)", "p.Glu412Ter (c.1234G>T)", "p.Ala217Pro (c.649G>C)"],
        "South Asian": ["p.Pro189Leu (c.566C>T)", "p.Arg336Trp (c.1006C>T)", "p.Glu412Ter (c.1234G>T)"],
        "East Asian":  ["p.Ala217Pro (c.649G>C)", "p.Glu412Ter (c.1234G>T)", "p.Leu303Pro (c.908T>C)"],
        "African":     ["p.Glu412Ter (c.1234G>T)", "p.Arg336Trp (c.1006C>T)"],
        "Other":       ["p.Glu412Ter (c.1234G>T)", "p.Ala217Pro (c.649G>C)"],
    }

    allele_classes = ["Missense/Missense", "Missense/Truncating", "Truncating/Truncating",
                      "Splice/Missense", "Splice/Truncating"]
    allele_weights = [0.32, 0.34, 0.22, 0.08, 0.04]

    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.16, 0.36, 0.28, 0.20]

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.36, 0.12, 0.08, 0.04, 0.40]

    renal_types = ["Structural anomaly (horseshoe/duplex)", "Cystic dysplasia",
                   "Tubulointerstitial nephropathy", "ESRD", "None"]
    renal_w = [0.13, 0.13, 0.07, 0.06, 0.61]

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.17, 0.44, 0.25, 0.14]

    misdiag = ["Isolated RP / Leber amaurosis", "Obesity syndrome (Prader-Willi screen)",
               "Alström Syndrome (cone-rod confusion)", "CHARGE/Joubert", "No initial misdiagnosis"]
    mis_w   = [0.31, 0.21, 0.10, 0.06, 0.32]

    cohort = []
    for i in range(n):
        eth    = rng.choices(eth_labels, weights=eth_weights)[0]
        v_pool = variants_by_eth[eth]
        v1     = rng.choice(v_pool)
        v2     = rng.choice(v_pool)
        poly   = rng.choices(polydactyly_types, weights=poly_w)[0]
        renal  = rng.choices(renal_types, weights=renal_w)[0]
        ret    = rng.choices(retinal_stages, weights=ret_w)[0]
        dx_a   = rng.choices(dx_ages, weights=dx_w)[0]
        allele = rng.choices(allele_classes, weights=allele_weights)[0]
        mis    = rng.choices(misdiag, weights=mis_w)[0]

        cohort.append({
            "id":               i + 1,
            "sex":              rng.choice(["M", "F"]),
            "ethnicity":        eth,
            "variant1":         v1,
            "variant2":         v2,
            "allele_class":     allele,
            "polydactyly":      poly != "None",
            "polydactyly_type": poly,
            "obesity":          rng.random() < 0.80,
            "bmi":              round(rng.uniform(28.0, 43.0), 1) if rng.random() < 0.80 else round(rng.uniform(21.0, 27.5), 1),
            "cognitive_ld":     rng.random() < 0.46,
            "renal_any":        renal != "None",
            "renal_type":       renal,
            "esrd":             renal == "ESRD",
            "hypogonadism":     rng.random() < 0.61,
            "anosmia":          rng.random() < 0.57,
            "chd":              rng.random() < 0.05,
            "retinal_stage":    ret,
            "retinal_endstage": ret in ["Legal blindness / end-stage"],
            "triallelic_bbs":   rng.random() < 0.06,   # BBS2 most common 3rd allele; BBS8 second
            "dx_age_group":     dx_a,
            "misdiagnosis":     mis,
            "initial_misdiagnosis": mis != "No initial misdiagnosis",
            "presentation_age": rng.choices(
                ["Birth (polydactyly noted)", "Infancy (nystagmus/strabismus)",
                 "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"],
                weights=[0.20, 0.19, 0.29, 0.22, 0.10]
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
        "cohort_n":     n,
        "seed":         _SEED,
        "gene":         "PTHB1",
        "gene_alias":   "BBS9",
        "omim_gene":    "*607968",
        "chromosome":   "7p14.3",
        "protein_aa":   495,
        "disease_omim": "#209900",
        "bbs_subtype":  "BBS9",
        "inheritance":  "Autosomal Recessive (biallelic LOF)",
        "frequency_pct_bbs": "~2%",
        "mechanism_class": "BBSome Bridge Subunit (spans core module BBS2-BBS7 AND peripheral module BBS1-BBS4-BBS5-BBS8; sole inter-module linker)",
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
            "PTHB1 (495 aa; Chr 7p14.3; OMIM *607968), also called BBS9, is the unique BRIDGE SUBUNIT "
            "of the BBSome octamer. Unlike all other BBSome subunits that reside exclusively in either "
            "the core module (BBS2, BBS7) or the peripheral module (BBS1, BBS4, BBS5, BBS8), PTHB1 "
            "spans BOTH modules simultaneously: its N-terminal arm contacts the BBS2-BBS7 WD40 core "
            "dimer, while its C-terminal adapter contacts BBS5 (PH domain) and BBS8/TTC8 (TPR4-6) in "
            "the peripheral module. This dual anchoring makes PTHB1 the sole inter-module linker that "
            "completes the BBSome octamer. PTHB1 LOF → the core module (BBS2-BBS7 dimer, pre-assembled "
            "by BCC chaperonin MKKS-BBS10-BBS12) and the peripheral module (BBS1-BBS4-BBS5-BBS8) each "
            "form independently but cannot dock to each other → no complete BBSome octamer → GPCR "
            "cargoes (LepR, SSTR3, D1R, Smo) mis-trafficked in cilia → leptin resistance, rod-cone "
            "dystrophy, polydactyly, renal anomalies, hypogonadism, anosmia, cognitive impairment. "
            "Cellular fingerprint: anti-MKKS IF NORMAL (BCC intact; BBS2/BBS7 fold correctly) but "
            "anti-BBS2 IF REDUCED (not absent; BBS2-BBS7 dimer is present but sub-complex is less "
            "stable without bridge), and anti-BBS8 IF REDUCED (peripheral module dis-localised). "
            "BBS9 is also the most frequent third allele in tri-allelic BBS families overall (~5–9% "
            "of BBS1, BBS2, BBS6, BBS8 families carry a BBS9 heterozygous third allele)."
        ),
        "key_distinction": (
            "PTHB1 is the ONLY BBSome subunit that simultaneously contacts both the core module "
            "(BBS2-BBS7) and the peripheral module (BBS1-BBS4-BBS5-BBS8). Its LOF produces a unique "
            "sub-complex phenotype where both half-complexes form but fail to join. "
            "BBS1: cargo-recognition propeller (IFT-A linker; M390R founder; 25%); cargo loading fails. "
            "BBS2/BBS7: core dimer scaffold (Step 0); BBSome octamer never forms; BBS2 absent by IF. "
            "BBS3 (ARL6): GTPase membrane recruiter; BBSome intact but not recruited to ciliary base. "
            "BBS4: PCM1 satellite tether; BBSome partially assembled, not tethered at satellites. "
            "BBS5: ciliary lipid (PtdIns) anchor; BBSome tethered but not membrane-anchored. "
            "BBS6 (MKKS): upstream BCC chaperonin; BBS2/BBS7/BBS9 misfold before assembly begins. "
            "BBS7: obligate BBS2 dimer partner; BBSome Step 0 fails same as BBS2 LOF. "
            "BBS8 (TTC8): IFT-B docking arm; BBSome assembles at PCM1 satellites but cannot enter axoneme. "
            "BBS9 [THIS]: bridge subunit; core and peripheral modules each assemble but CANNOT DOCK to "
            "each other → incomplete octamer → no ciliary GPCR transport. Cellular key: BBS9 IF absent; "
            "BBS2 IF REDUCED (sub-complex present but bridge-less); MKKS IF NORMAL (BCC intact). "
            "BBS9 is also the most common third allele across the BBS spectrum (tri-allelic BBS)."
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
            {"gene": "BBS6 / MKKS",  "role": "Chaperonin-like co-chaperone folds BBS2/BBS7/BBS9",
             "function_class": "Upstream Chaperonin Co-Chaperone", "omim": "*604896", "frequency_pct": "5–8"},
            {"gene": "BBS7",         "role": "Obligate WD40 dimer partner of BBS2 — core module scaffold",
             "function_class": "Structural BBSome Subunit (Core Dimer)", "omim": "*607590", "frequency_pct": "2–4"},
            {"gene": "BBS8 / TTC8",  "role": "TPR solenoid — IFT-B docking arm; only BBSome-IFT-B bridge",
             "function_class": "IFT-B Docking Arm (Peripheral Module)", "omim": "*608132", "frequency_pct": "2–4"},
            {"gene": "BBS9 / PTHB1 [THIS]", "role": "Bridge subunit — spans core (BBS2-BBS7) and peripheral (BBS1-BBS4-BBS5-BBS8) modules",
             "function_class": "BBSome Inter-Module Bridge", "omim": "*607968", "frequency_pct": 2},
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
        ("Rod-cone dystrophy (rod-first; 100%)",             n,  100.0),
        ("Post-axial polydactyly",                            sum(1 for p in c if p["polydactyly"]),       _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Obesity (BMI ≥ 28)",                                sum(1 for p in c if p["obesity"]),            _pct(sum(1 for p in c if p["obesity"]))),
        ("Hypogonadism",                                      sum(1 for p in c if p["hypogonadism"]),       _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Anosmia/Hyposmia",                                  sum(1 for p in c if p["anosmia"]),            _pct(sum(1 for p in c if p["anosmia"]))),
        ("Cognitive/Learning Disability",                     sum(1 for p in c if p["cognitive_ld"]),       _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("Renal Anomaly (any)",                               sum(1 for p in c if p["renal_any"]),          _pct(sum(1 for p in c if p["renal_any"]))),
        ("Congenital Heart Disease (CHD)",                    sum(1 for p in c if p["chd"]),                _pct(sum(1 for p in c if p["chd"]))),
        ("ESRD",                                              sum(1 for p in c if p["esrd"]),               _pct(sum(1 for p in c if p["esrd"]))),
        ("Tri-allelic BBS (BBS2/BBS8 3rd allele)",           sum(1 for p in c if p["triallelic_bbs"]),    _pct(sum(1 for p in c if p["triallelic_bbs"]))),
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
            "Gene":             "PTHB1 (Parathyroid Hormone-responsive B9 protein domain 1; alias BBS9/BBCP); Chr 7p14.3",
            "OMIM_Gene":        "*607968",
            "Chromosome":       "7p14.3",
            "Protein":          "495 aa — N-terminal arm (BBS2-BBS7 core contact) · central bridge domain (BBS1 contact; unique to BBS9) · C-terminal adapter (BBS5 PH + BBS8 TPR4-6 contacts)",
            "Domains":          "aa 1–110: N-terminal arm (BBS2/BBS7 WD40 blade contacts) · aa 111–310: central bridge domain (BBS1 beta-propeller blade-5 contact; no homology to other BBSome subunits — bridge-specific fold) · aa 311–495: C-terminal adapter (BBS5 PH domain + BBS8 TPR4-6 contacts)",
            "Bridge_function":  "Only BBSome subunit that spans BOTH modules simultaneously: N-terminal arm → core module (BBS2-BBS7 dimer); C-terminal adapter → peripheral module (BBS1-BBS4-BBS5-BBS8); sole inter-module linker completing the octamer",
            "BBSome_position":  "Inter-module bridge — PTHB1 is required for the final assembly step joining the two independently-formed half-complexes into the complete BBSome octamer",
            "Inheritance":      "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":    "~2% of all BBS (<200 families worldwide 2026)",
            "Allele_spectrum":  "Missense/truncating ≈ missense/missense > truncating/truncating; no single dominant ethnic founder; MENA Arg336Trp and European Ala217Pro are enriched but not exclusive",
            "Triallelic_BBS":   "~6% (BBS2 most common third allele — core dimer partner; BBS8 second — peripheral module partner shares C-terminal adapter contacts)",
            "CHD_penetrance":   "~5% (intermediate; no specific cardiac ciliary role for PTHB1 identified)",
            "BBS9_as_modifier": "PTHB1 heterozygous variants are the most common third allele across the BBS spectrum — present in ~5–9% of BBS1, BBS2, BBS6, BBS8 tri-allelic families (bridge position makes heterozygous PTHB1 variants modifiers of any BBSome subunit LOF)",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 9 (BBS9) — #209900",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS9 subtype ~2% (~2–3 in 10,000,000)",
            "Cardinal_features": "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR mis-traffic) · Hypogonadism · Renal anomalies · Cognitive/LD · Anosmia",
            "BBS9_unique":       "Sub-complex phenotype: core module (BBS2-BBS7 dimer) and peripheral module (BBS1-BBS4-BBS5-BBS8) each form independently but cannot join; anti-BBS2 IF REDUCED (not absent); anti-MKKS IF NORMAL (BCC intact)",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; identical pattern to BBS1–8)",
            "Obesity_mechanism": "LepR mis-trafficking (hypothalamic cilia) → leptin resistance → hyperphagia; GLP1RA applicable (identical mechanism to all BBSome subunit LOF)",
            "Renal":             "Structural anomalies + cystic dysplasia + TIN; ESRD ~6%; cilia-dependent tubulogenesis",
            "CHD":               "~5%; intermediate among BBS1–9; echocardiogram indicated",
            "Diagnosis":         "Full BBS gene panel (20–24 genes minimum); fibroblast IF: absent BBS9 + REDUCED BBS2 (not absent) + NORMAL MKKS — distinguishes from BBS2 LOF (BBS2 absent) and BBS6/MKKS LOF (MKKS absent)",
            "Treatment":         "Symptomatic (no curative gene therapy approved 2026); low-vision aids; GLP1RA for obesity; renal surveillance; same management framework as BBS1–8",
            "Prognosis":         "Progressive visual loss → blindness 3rd–4th decade; renal variable; survival near-normal with management",
        },
        "key_variants": [
            {
                "variant":     "p.Ala217Pro (c.649G>C)",
                "domain":      "N-terminal arm — BBS7 WD40 blade-2 loop contact (Ala217 hydrogen-bonds BBS7 blade-2 β-turn)",
                "consequence": "BBS2-BBS7 core dimer docking geometry disrupted; bridge N-terminal anchor lost; core-module half-complex does not engage bridge; BBSome incomplete; full BBS phenotype",
                "ethnicity":   "European (mild enrichment; also pan-ethnic)"
            },
            {
                "variant":     "p.Pro189Leu (c.566C>T)",
                "domain":      "N-terminal arm — BBS7 CC domain surface contact (Pro189 contributes to bridge-arm fold against BBS2-BBS7 dimer face)",
                "consequence": "Bridge N-terminal arm destabilised; BBS7 contact weakened; incomplete octamer; full BBS phenotype",
                "ethnicity":   "South Asian (enriched; pan-ethnic distribution)"
            },
            {
                "variant":     "p.Arg336Trp (c.1006C>T)",
                "domain":      "Central bridge domain — BBS1 beta-propeller blade-5 contact (Arg336 salt-bridge to BBS1 Glu244 on blade-5 inner surface)",
                "consequence": "BBS1 contact to bridge severed; peripheral module does not engage bridge via BBS1; BBSome octamer incomplete; full BBS phenotype; also reduces BBS1 stability",
                "ethnicity":   "MENA (enriched; also African)"
            },
            {
                "variant":     "p.Leu303Pro (c.908T>C)",
                "domain":      "Central bridge domain — hydrophobic core of bridge fold (Leu303 packs against bridge α-helices 6-7; Pro insertion breaks helix)",
                "consequence": "Bridge domain fold destabilised; partial loss of both BBS1 and BBS2-BBS7 contacts; intermediate severity; biallelic with truncating variant → full BBS",
                "ethnicity":   "European (mild enrichment; pan-ethnic)"
            },
            {
                "variant":     "p.Glu412Ter (c.1234G>T)",
                "domain":      "Truncating null — stops at aa 412; entire C-terminal adapter (BBS5 PH + BBS8 TPR4-6 contacts) lost",
                "consequence": "Complete loss of C-terminal adapter → peripheral module cannot engage bridge from C-terminal side; combined with any N-terminal allele → full null phenotype; pan-ethnic",
                "ethnicity":   "Pan-ethnic (most common truncating allele)"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20 genes including PTHB1/BBS9, BBS2, BBS7, BBS8, BCC components); MLPA/CNV for large deletions",
            "3. PTHB1-specific: Sanger confirmation; classify variant by bridge domain (N-terminal arm: BBS2-BBS7 contact / central bridge: BBS1 contact / C-terminal adapter: BBS5-BBS8 contact)",
            "4. Fibroblast IF — BBS9 fingerprint: anti-BBS9 (absent) + anti-BBS2 (REDUCED — sub-complex present; BBS2-BBS7 dimer forms but is bridge-less) + anti-MKKS (NORMAL — BCC intact)",
            "5. Key discriminator: if anti-BBS2 is completely ABSENT → reconsider BBS2 or BBS7 LOF (not BBS9); BBS9 LOF leaves residual BBS2 sub-complex detectable by IF",
            "6. Key discriminator: if anti-MKKS is absent → reconsider BBS6/MKKS LOF; BBS9 LOF does not disrupt BCC chaperonin",
            "7. Tri-allelic BBS: screen BBS2 (most common third allele, ~6% of BBS9 families — core dimer partner) and BBS8 (second most common — shares C-terminal adapter contacts)",
            "8. CHD workup: echocardiogram (CHD ~5%; ASD/VSD); echocardiogram at diagnosis regardless",
            "9. Renal: ultrasound + GFR; cystatin C annually; ESRD risk ~6%; ophthalmology ERG + Goldmann VF; OCT at diagnosis",
            "10. BBS9 as third allele: BBS9 is the most common third allele across the BBS spectrum — when diagnosing BBS1, BBS2, BBS6, or BBS8, always screen PTHB1 for a heterozygous modifier",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; photosensitising drug avoidance; vitamin A not indicated for BBS; gene therapy investigational",
            "2. Obesity: GLP-1 receptor agonists (semaglutide) — LepR mis-trafficking mechanism applies identically to BBS9 LOF; bariatric surgery in refractory cases",
            "3. Renal: ACE inhibitor/ARB for CKD progression; renal transplant for ESRD (curative — no allograft recurrence); avoid nephrotoxins",
            "4. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling",
            "5. Anosmia: safety counselling (gas leaks, food spoilage); smell training investigational; smoke detectors mandatory",
            "6. Cognitive/LD: early intervention; IEP; neuropsychological assessment; occupational therapy",
            "7. CHD (~5%): echocardiogram-guided cardiology; no specific BBS9 cardiac lesion profile; surgical/catheterisation as indicated",
            "8. Genetic counselling: AR inheritance; sibling recurrence 25%; prenatal/preimplantation genetic testing available",
            "9. Co-testing: always screen BBS2 and BBS8 in BBS9 families for tri-allelic BBS (~6%); BBS9 heterozygous co-testing in all BBS1/BBS2/BBS6/BBS8 families",
            "10. Surveillance: annual ophthalmology + renal + metabolic (HbA1c, lipids) + cardiac; growth/pubertal charting in children",
        ],
        "ddx_table": [
            {"disease": "BBS2 (BBS2 gene)", "key_difference": "BBS2 LOF → BBS2 IF completely ABSENT (BBS2-BBS7 dimer fails Step 0); BBS9 LOF → BBS2 IF REDUCED (sub-complex forms but bridge-less) — this is the key IF discriminator; gene panel mandatory"},
            {"disease": "BBS7 (BBS7 gene)", "key_difference": "BBS7 LOF → absent BBS2 IF + absent BBS7 IF (obligate dimer mutual destabilisation); BBS9 LOF → BBS2 REDUCED (not absent); BBS7 IF partial; gene panel distinguishes"},
            {"disease": "BBS6 / MKKS (MKKS gene)", "key_difference": "BBS6 LOF → absent MKKS IF + absent BBS2 IF (BCC fails; BBS2/BBS7 misfold before assembly); BBS9 LOF → NORMAL MKKS IF + REDUCED BBS2 IF; BCC distinction is key"},
            {"disease": "BBS8 / TTC8 (TTC8 gene)", "key_difference": "BBS8 LOF → BBSome assembles at PCM1 satellites normally (BBS9 IF NORMAL in BBS8 LOF); BBS9 LOF → BBS9 IF absent; bridge failure prevents octamer completion (pre-satellite step)"},
            {"disease": "BBS1 (BBS1 gene)", "key_difference": "BBS1 LOF → BBS1 IF absent; cargo loading fails but bridge (BBS9) is intact; BBS9 LOF → BBS1 sub-complex may form but cannot dock to bridge; IF patterns distinct; gene panel"},
            {"disease": "BBS4 (BBS4 gene)", "key_difference": "BBS4 LOF → absent BBS4 IF (satellite tethering fails); BBS9 LOF → BBS4 IF may be reduced (peripheral module dis-localised); gene panel essential — no reliable cellular IF alone"},
            {"disease": "Alström Syndrome (ALMS1)", "key_difference": "Cone-rod (cone first, not rod first); no polydactyly; infantile DCM (60%); sensorineural hearing loss; normal PTHB1/BBS9/BBS2 IF; ALMS1 mutations"},
            {"disease": "Joubert Syndrome (JBTS)", "key_difference": "Molar tooth sign on brain MRI; cerebellar vermis hypoplasia; PTHB1 mutations not causative of classic JBTS; different IFT/transition-zone module affected"},
        ],
        "mechanism_glossary": [
            {
                "term": "PTHB1 as the Unique BBSome Inter-Module Bridge",
                "definition": "The BBSome octamer is assembled in two independently-built halves: the core module (BBS2-BBS7 obligate WD40 heterodimer, pre-folded by BCC chaperonin MKKS-BBS10-BBS12) and the peripheral module (BBS1-BBS4-BBS5-BBS8). These two half-complexes cannot join spontaneously — PTHB1 (BBS9) is the only subunit that physically bridges them. Its N-terminal arm simultaneously grips the BBS2 WD40 propeller blade-3-4 and BBS7 WD40 blade-1-2, while its C-terminal adapter contacts BBS5 PH domain and BBS8 TPR4-6. This dual anchoring completes the octamer. PTHB1 LOF → the two half-complexes form normally but stay apart → no complete BBSome → all downstream ciliary GPCR trafficking fails identically to any other BBSome LOF."
            },
            {
                "term": "BBS9 Sub-Complex Phenotype (REDUCED vs ABSENT BBS2 IF)",
                "definition": "In BBS2 or BBS7 LOF, the BBS2-BBS7 obligate dimer fails completely (Step 0 — MKKS-dependent folding still produces BBS2 protein, but the dimer is immediately destabilised and proteasomally degraded). In BBS9 LOF, the BBS2-BBS7 dimer FORMS (MKKS/BCC is intact) but is stranded as a sub-complex without the bridge. This sub-complex is detectable by anti-BBS2 IF — reduced signal, not absent. This BBS2 REDUCED (not absent) pattern, combined with MKKS NORMAL and BBS9 ABSENT, is the BBS9 cellular fingerprint that distinguishes it from BBS2/BBS7 LOF (BBS2 absent), BBS6 LOF (MKKS absent + BBS2 absent), and BBS8 LOF (BBS2 normal + BBSome at satellites)."
            },
            {
                "term": "BBS9 as the Universal Third Allele (Tri-allelic BBS)",
                "definition": "Tri-allelic BBS occurs when a patient carries biallelic pathogenic variants in one BBS gene plus a heterozygous pathogenic variant in a second BBS gene. PTHB1 (BBS9) is the most frequent third allele across the entire BBS spectrum — it is reported as a third allele in BBS1, BBS2, BBS6, BBS8, BBS10, and BBS12 families. This is mechanistically explained by BBS9's bridge position: a heterozygous BBS9 variant reduces bridge stoichiometry, and when combined with homozygous LOF in any other BBSome subunit, further impairs the residual ciliary trafficking, worsening phenotype severity. The bridge contacts BBS2 (core), BBS1, BBS5, and BBS8 (peripheral) simultaneously — heterozygous BBS9 variants therefore modulate virtually every step of BBSome assembly."
            },
            {
                "term": "PTHB1 Central Bridge Domain — No Structural Homologs in BBSome",
                "definition": "PTHB1's central bridge domain (aa 111–310) has no structural homolog in any other BBSome subunit. All other BBSome subunits are either WD40 beta-propellers (BBS1, BBS2, BBS7, BBS9's N-terminal arm), TPR solenoids (BBS4, BBS8), or specialised domains (BBS3/ARL6 GTPase, BBS5 PH). The bridge domain is a unique fold that appears to have evolved specifically to connect the two independently-assembled BBSome modules. Its central residue Arg336 forms a critical salt-bridge with BBS1 Glu244 on the cargo-recognition propeller — this is the only direct structural contact between the BBS1 cargo-loading subunit and the BBS2-BBS7 core scaffold, mediated exclusively through BBS9."
            },
            {
                "term": "LepR Ciliary Mis-Trafficking — Identical Mechanism Across All BBSome LOF",
                "definition": "Leptin receptor (LepR) requires the complete BBSome octamer for retrograde IFT-mediated removal from hypothalamic neuronal cilia. In BBS9 LOF, the incomplete octamer (two half-complexes, no bridge) cannot perform this retrograde transport → LepR constitutively retained in cilia → signal desensitisation → leptin resistance → hyperphagia and obesity despite elevated plasma leptin. The mechanism is mechanistically identical across all BBSome subunit LOF (BBS1–9) — the assembly step differs but the downstream functional failure is the same. GLP-1 receptor agonists bypass this by acting on GLP1R interneurons using a non-ciliary satiety pathway."
            },
            {
                "term": "BCC Chaperonin Integrity in BBS9 LOF — Why MKKS IF Is Normal",
                "definition": "The BBSome chaperonin complex (BCC) consists of MKKS (BBS6), BBS10, and BBS12 — a group II chaperonin-like trimer that co-folds the WD40 beta-propellers of BBS2 and BBS7 before BBSome assembly begins. BCC also assists in folding PTHB1 (BBS9) itself. In BBS9 LOF, PTHB1 protein is absent (LOF variants), but BCC is intact — MKKS, BBS10, and BBS12 are all expressed and form the trimer normally. The BCC successfully folds BBS2-BBS7 (which then dimerises) but has no PTHB1 to place as bridge. This is why anti-MKKS IF is NORMAL in BBS9 LOF — distinguishing BBS9 from BBS6/MKKS LOF where the BCC itself is disrupted."
            },
        ],
    }
