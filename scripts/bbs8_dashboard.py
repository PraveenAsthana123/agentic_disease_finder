"""
BBS8 Bardet-Biedl Syndrome Type 8 — TTC8 (IFT-B Docking Arm / TPR Solenoid) Dashboard
========================================================================================
Primary Gene : TTC8 (*608132) — Tetratricopeptide Repeat Domain 8; also called BBS8;
               598 aa; 13 TPR (tetratricopeptide repeat) domains forming a right-handed
               solenoid superhelix; Chr 14q32.11.
               TTC8 is a STRUCTURAL BBSome SUBUNIT that occupies the IFT-B docking arm
               position in the BBSome octamer.
               Domain organisation (598 aa):
                 TPR1–3 (aa 1–100): N-terminal arm; contacts BBS1 beta-propeller and
                   BBS7 C-terminal tail — interfaces the core module (BBS2-BBS7 dimer)
                   with the peripheral module; Arg229 in TPR3 contacts BBS7
                   C-terminal BBIP10-dock domain; Arg229Trp → core-peripheral
                   interface weakened;
                 TPR4–6 (aa 101–200): central solenoid; contacts BBS9 bridge subunit
                   (PTHB1); essential for BBS9-TTC8 lateral contact stabilising the
                   full octamer; loss disrupts BBS9 bridge positioning;
                 TPR7–9 (aa 201–310): IFT-B docking platform — the UNIQUE function of
                   TTC8; TPR7 Thr347 directly contacts IFT38/CLUAP1 (IFT-B subunit;
                   also called QiF1) at the IFT38 WD40 propeller face; this is the
                   SOLE BBSome-IFT-B molecular interface; Thr347Met → IFT-B
                   docking completely abolished; no other BBSome subunit contacts
                   IFT-B directly — TTC8 is the unique IFT-B bridge of the BBSome;
                 TPR10–11 (aa 311–420): ARL6 (BBS3) GTPase contact solenoid; ARL6-GTP
                   Switch-II contacts TPR10 concave face; Arg413 at TPR8-9 loop
                   contacts both IFT38 and ARL6 — Arg413Gln → dual docking loss;
                 TPR12–13 (aa 421–598): C-terminal cap; contacts BBIP10 (BBSome
                   adaptor); Val479 is a hydrophobic core residue in TPR10 barrel
                   — Val479Ala destabilises solenoid fold.
               TTC8 LOF → BBSome-IFT-B interface lost → assembled BBSome octamer
               cannot dock onto IFT-B in the periciliary region → BBSome cargo
               (LepR, SSTR3, D1R, Smo) cannot be transported retrograde along axoneme
               → accumulates in cilia → mis-trafficking cascade → BBS phenotype.
               Critical discriminator: BBSome ASSEMBLES NORMALLY in TTC8 LOF;
               BBSome accumulates at PCM1 pericentriolar satellites (BBS4-tethered)
               but fails to dock IFT-B → cannot be exported from satellites into
               the ciliary transition zone → this is the LATEST assembly/trafficking
               step disrupted in the known BBS pathway.
               Key cellular discriminator from other BBS types:
                 In BBS8-null cells: anti-BBS8 IF: absent; anti-BBS2 IF: NORMAL
                   (core dimer intact — BBS2-BBS7 unaffected); anti-MKKS IF: NORMAL
                   (BCC intact); anti-BBS4 IF: NORMAL (PCM1 satellite tethering intact;
                   BBSome reaches satellites normally; failure is at IFT-B docking);
                 BBSome intermediates detectable at PCM1 satellites by
                   anti-BBS2/anti-BBS4 co-IF — this is UNIQUE to BBS8 LOF;
                 PCM1 co-IF with anti-BBS2: BBS2 still colocalises with PCM1
                   puncta in BBS8-null cells — distinguishes from BBS4 LOF
                   (where PCM1 tethering itself is impaired).
Disease        : Bardet-Biedl Syndrome (#209900) — BBS8 subtype
Frequency      : ~2–4% of all BBS
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~5% of BBS8 families (BBS9 most common third allele — bridge
                 subunit, shares peripheral module contacts with TTC8)
Cohort         : 40 patients; educational/synthetic; seed 347
Endpoints      : /api/bbs8/overview | /api/bbs8/breakdown | /api/bbs8/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 347
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
        "European":    ["p.Arg229Trp (c.685C>T)", "p.Val479Ala (c.1436T>C)", "p.Glu391Ter (c.1171G>T)"],
        "MENA":        ["p.Arg413Gln (c.1238G>A)", "p.Glu391Ter (c.1171G>T)", "p.Arg229Trp (c.685C>T)"],
        "South Asian": ["p.Thr347Met (c.1040C>T)", "p.Arg413Gln (c.1238G>A)", "p.Glu391Ter (c.1171G>T)"],
        "East Asian":  ["p.Arg229Trp (c.685C>T)", "p.Glu391Ter (c.1171G>T)", "p.Val479Ala (c.1436T>C)"],
        "African":     ["p.Glu391Ter (c.1171G>T)", "p.Arg413Gln (c.1238G>A)"],
        "Other":       ["p.Glu391Ter (c.1171G>T)", "p.Arg229Trp (c.685C>T)"],
    }

    allele_classes = ["Missense/Missense", "Missense/Truncating", "Truncating/Truncating",
                      "Splice/Missense", "Splice/Truncating"]
    allele_weights = [0.34, 0.32, 0.20, 0.09, 0.05]

    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.15, 0.35, 0.30, 0.20]

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.37, 0.13, 0.09, 0.03, 0.38]

    renal_types = ["Structural anomaly (horseshoe/duplex)", "Cystic dysplasia",
                   "Tubulointerstitial nephropathy", "ESRD", "None"]
    renal_w = [0.14, 0.12, 0.08, 0.06, 0.60]

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.18, 0.42, 0.27, 0.13]

    misdiag = ["Isolated RP / Leber amaurosis", "Obesity syndrome (Prader-Willi screen)",
               "Alström Syndrome (cone-rod confusion)", "CHARGE/Joubert", "No initial misdiagnosis"]
    mis_w   = [0.30, 0.22, 0.09, 0.07, 0.32]

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
            "obesity":          rng.random() < 0.82,
            "bmi":              round(rng.uniform(28.0, 43.0), 1) if rng.random() < 0.82 else round(rng.uniform(21.0, 27.5), 1),
            "cognitive_ld":     rng.random() < 0.43,
            "renal_any":        renal != "None",
            "renal_type":       renal,
            "esrd":             renal == "ESRD",
            "hypogonadism":     rng.random() < 0.62,
            "anosmia":          rng.random() < 0.58,
            "chd":              rng.random() < 0.04,
            "retinal_stage":    ret,
            "retinal_endstage": ret in ["Legal blindness / end-stage"],
            "triallelic_bbs":   rng.random() < 0.05,   # BBS9 most common 3rd allele — peripheral module partner
            "dx_age_group":     dx_a,
            "misdiagnosis":     mis,
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
        "cohort_n":     n,
        "seed":         _SEED,
        "gene":         "TTC8",
        "gene_alias":   "BBS8",
        "omim_gene":    "*608132",
        "chromosome":   "14q32.11",
        "protein_aa":   598,
        "disease_omim": "#209900",
        "bbs_subtype":  "BBS8",
        "inheritance":  "Autosomal Recessive (biallelic LOF)",
        "frequency_pct_bbs": "2–4%",
        "mechanism_class": "BBSome IFT-B Docking Arm (TPR solenoid; only BBSome subunit that directly contacts IFT-B)",
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
            "TTC8 (598 aa; Chr 14q32.11; OMIM *608132), also called BBS8, is a 13-TPR (tetratricopeptide repeat) "
            "solenoid protein and a structural subunit of the BBSome octamer. TTC8 occupies the IFT-B docking arm "
            "position — the interface between the assembled BBSome and the IFT-B particle that carries it along "
            "the ciliary axoneme. Critically, TTC8 is the ONLY BBSome subunit that directly contacts IFT-B, "
            "specifically binding IFT38 (CLUAP1/QiF1) via its TPR7–9 platform. TTC8 LOF → BBSome assembles "
            "normally (BBS2-BBS7 core dimer intact; BBS4 PCM1-tethering intact; full octamer forms at PCM1 "
            "satellites) → but assembled BBSome cannot dock onto IFT-B → BBSome is stranded at pericentriolar "
            "satellites and cannot be exported into the ciliary transition zone → GPCR cargoes (LepR, SSTR3, "
            "D1R, Smo) accumulate in cilia rather than being removed retrograde → leptin resistance, anosmia, "
            "polydactyly, hypogonadism, renal cysts, rod-cone dystrophy. TTC8 also contacts ARL6-GTP (BBS3 "
            "GTPase) via its TPR10 concave face — TTC8 participates in both the ARL6-mediated ciliary "
            "membrane recruitment and the IFT-B axonemal transport steps. This dual role makes TTC8 the "
            "critical link between the ARL6 membrane-docking signal and IFT-B-dependent ciliary trafficking."
        ),
        "key_distinction": (
            "TTC8 is the ONLY BBSome subunit that directly contacts IFT-B (via IFT38/CLUAP1 at TPR7–9). "
            "All prior BBS types disrupt BBSome assembly; TTC8 LOF disrupts BBSome-IFT-B DOCKING — a post-assembly step. "
            "BBS1: cargo-recognition propeller (IFT-A linker; M390R founder; 25%); BBSome fails to load GPCR cargo. "
            "BBS2/BBS7: core dimer scaffold (Step 0); BBSome octamer never forms. "
            "BBS3 (ARL6): GTPase membrane recruiter; BBSome intact but not recruited to ciliary base membrane. "
            "BBS4: PCM1 satellite tether; BBSome partially assembled, not tethered at satellites. "
            "BBS5: ciliary lipid (PtdIns) anchor; BBSome tethered but not membrane-anchored. "
            "BBS6 (MKKS): upstream BCC chaperonin; BBS2/BBS7/BBS9 misfold before assembly begins. "
            "BBS7: obligate BBS2 dimer partner; BBSome Step 0 fails same as BBS2 LOF. "
            "BBS8 [THIS]: IFT-B arm; BBSome ASSEMBLES at PCM1 satellites normally (anti-BBS2 IF: NORMAL, "
            "anti-BBS4 IF: NORMAL) but fails to dock IFT-B → stranded at satellites → cannot enter axoneme. "
            "Key cellular fingerprint: BBSome satellites present (anti-BBS2 + anti-PCM1 co-IF positive) "
            "but anti-BBS8 absent — the ONLY BBS subtype with assembled BBSome at satellites but no axonemal entry. "
            "Tri-allelic BBS: BBS9 most common third allele (~5%) — peripheral module partner that bridges "
            "BBS8 to core dimer; BBS9 heterozygous variant worsens IFT-B docking geometry."
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
            {"gene": "BBS8 / TTC8 [THIS]", "role": "TPR solenoid — IFT-B docking arm; only BBSome-IFT-B bridge",
             "function_class": "IFT-B Docking Arm (Peripheral Module)", "omim": "*608132", "frequency_pct": "2–4"},
            {"gene": "BBS9 (PTHB1)", "role": "Bridge subunit contacts BBS2-BBS7 dimer and BBS1-BBS4-BBS5-BBS8 module",
             "function_class": "BBSome Bridge Subunit", "omim": "*607968", "frequency_pct": 2},
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
        ("Rod-cone dystrophy (rod-first; 100%)",          n,  100.0),
        ("Post-axial polydactyly",                         sum(1 for p in c if p["polydactyly"]),       _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Obesity (BMI ≥ 28)",                             sum(1 for p in c if p["obesity"]),            _pct(sum(1 for p in c if p["obesity"]))),
        ("Hypogonadism",                                   sum(1 for p in c if p["hypogonadism"]),       _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Anosmia/Hyposmia",                               sum(1 for p in c if p["anosmia"]),            _pct(sum(1 for p in c if p["anosmia"]))),
        ("Cognitive/Learning Disability",                  sum(1 for p in c if p["cognitive_ld"]),       _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("Renal Anomaly (any)",                            sum(1 for p in c if p["renal_any"]),          _pct(sum(1 for p in c if p["renal_any"]))),
        ("Congenital Heart Disease (CHD)",                 sum(1 for p in c if p["chd"]),                _pct(sum(1 for p in c if p["chd"]))),
        ("ESRD",                                           sum(1 for p in c if p["esrd"]),               _pct(sum(1 for p in c if p["esrd"]))),
        ("Tri-allelic BBS (BBS9 3rd allele — peripheral)", sum(1 for p in c if p["triallelic_bbs"]),    _pct(sum(1 for p in c if p["triallelic_bbs"]))),
        ("Initial misdiagnosis",                           sum(1 for p in c if p["initial_misdiagnosis"]), _pct(sum(1 for p in c if p["initial_misdiagnosis"]))),
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
            "Gene":             "TTC8 (Tetratricopeptide Repeat Domain 8; alias BBS8); Chr 14q32.11",
            "OMIM_Gene":        "*608132",
            "Chromosome":       "14q32.11",
            "Protein":          "598 aa — 13 TPR (tetratricopeptide repeat) domains forming right-handed solenoid superhelix",
            "Domains":          "TPR1–3 (aa 1–100; BBS1/BBS7 interface) · TPR4–6 (aa 101–200; BBS9 lateral contact) · TPR7–9 (aa 201–310; IFT38/CLUAP1 IFT-B docking platform — UNIQUE) · TPR10–11 (aa 311–420; ARL6-GTP contact) · TPR12–13 (aa 421–598; BBIP10 cap)",
            "IFT_B_contact":    "TPR7–9 (Thr347 contacts IFT38/CLUAP1 WD40 face) — the sole BBSome-IFT-B molecular interface; no other BBSome subunit contacts IFT-B directly",
            "BBSome_position":  "Peripheral module — IFT-B docking arm; assembled LAST (after core dimer and satellite tethering steps)",
            "Inheritance":      "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":    "2–4% of all BBS",
            "Allele_spectrum":  "Missense/missense ≈ missense/truncating > truncating/truncating; no dominant ethnic founder",
            "Triallelic_BBS":   "~5% (BBS9 most common third allele — peripheral module bridge; BBS4 also reported)",
            "CHD_penetrance":   "~4% (lowest among BBS1–8; no specific cardiac ciliary role identified for TTC8)",
            "BBSome_assembles": "YES — BBSome octamer FORMS normally in BBS8 LOF; failure is exclusively at IFT-B docking (post-assembly step)",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 8 (BBS8) — #209900",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS8 subtype ~2–4%",
            "Cardinal_features": "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR mis-traffic) · Hypogonadism · Renal anomalies · Cognitive/LD · Anosmia",
            "BBS8_unique":       "BBSome assembles normally at PCM1 satellites — anti-BBS2 IF NORMAL, anti-BBS4 IF NORMAL; failure is at IFT-B docking (only BBS subtype with assembled BBSome at satellites)",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; identical to BBS1–7)",
            "Obesity_mechanism": "LepR mis-trafficking (hypothalamic cilia) → leptin resistance → hyperphagia; GLP1RA applicable",
            "Renal":             "Structural anomalies + cystic dysplasia + TIN; ESRD ~6%; cilia-dependent tubulogenesis",
            "CHD":               "~4%; lowest penetrance in BBS1–8 series; echocardiogram still indicated",
            "Diagnosis":         "Full BBS gene panel (20–24 genes); fibroblast IF: absent BBS8 + NORMAL BBS2 + NORMAL BBS4 + NORMAL MKKS; PCM1 co-IF shows BBS2 at satellites (BBSome assembled, not exported)",
            "Treatment":         "Symptomatic (no curative gene therapy approved 2026); low-vision aids; GLP1RA for obesity; renal surveillance",
            "Prognosis":         "Progressive visual loss → blindness 3rd–4th decade; renal variable; survival near-normal with management",
        },
        "key_variants": [
            {
                "variant":     "p.Arg229Trp (c.685C>T)",
                "domain":      "TPR3 — BBS1/BBS7 core-peripheral interface (Arg229 contacts BBS7 C-terminal BBIP10-dock domain)",
                "consequence": "Core-peripheral module interface weakened; BBSome octamer destabilised; IFT-B docking impaired downstream; full BBS phenotype",
                "ethnicity":   "European (most common European BBS8 allele; also pan-ethnic)"
            },
            {
                "variant":     "p.Thr347Met (c.1040C>T)",
                "domain":      "TPR7 — direct IFT38/CLUAP1 contact site (Thr347 hydrogen-bonds IFT38 WD40 propeller loop)",
                "consequence": "IFT-B docking completely abolished; assembled BBSome stranded at PCM1 satellites; cannot enter axoneme; full severe BBS phenotype",
                "ethnicity":   "South Asian (enriched; pan-ethnic distribution)"
            },
            {
                "variant":     "p.Arg413Gln (c.1238G>A)",
                "domain":      "TPR8–9 loop — dual docking platform (contacts IFT38 AND ARL6-GTP Switch-II simultaneously)",
                "consequence": "Loss of IFT-B docking AND ARL6 GTPase contact; double defect in ciliary trafficking; severe BBS with early retinal onset",
                "ethnicity":   "MENA (enriched; also African)"
            },
            {
                "variant":     "p.Val479Ala (c.1436T>C)",
                "domain":      "TPR10 hydrophobic core — barrel-stabilising valine (Val479 packs against TPR10 alpha-helices)",
                "consequence": "TPR solenoid destabilised from TPR10 onwards; partial loss of ARL6 contact and C-terminal BBIP10 cap; intermediate severity BBS",
                "ethnicity":   "European (mild enrichment; pan-ethnic)"
            },
            {
                "variant":     "p.Glu391Ter (c.1171G>T)",
                "domain":      "Truncating null — stops at aa 391; TPR10–13, ARL6 contact, and BBIP10 cap all lost",
                "consequence": "Complete null — no TTC8 protein; BBSome assembles (BBS2/BBS7/BBS4 intact) but IFT-B arm absent; severe BBS phenotype; pan-ethnic",
                "ethnicity":   "Pan-ethnic (most common truncating allele)"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20 genes including TTC8/BBS8, BBS2, BBS4, BBS9, BCC components); MLPA/CNV for large deletions",
            "3. TTC8-specific: Sanger confirmation; classify variant by TPR domain (TPR3 core-peripheral interface / TPR7-9 IFT-B docking / TPR10-11 ARL6 contact / TPR12-13 BBIP10 cap)",
            "4. Fibroblast IF — BBS8 fingerprint (UNIQUE): anti-BBS8 (absent) + anti-BBS2 (NORMAL — BBSome core dimer intact) + anti-BBS4 (NORMAL — PCM1 tethering intact) + anti-MKKS (NORMAL — BCC intact)",
            "5. PCM1 co-IF (definitive): anti-BBS2 + anti-PCM1 co-staining shows BBS2 still colocalising with PCM1 puncta (BBSome assembled at satellites but stranded) — this is pathognomonic of BBS8 LOF among BBSome subunit defects",
            "6. If anti-BBS2 is absent: reconsider BBS2 or BBS7 LOF (not BBS8); if anti-BBS4 absent: reconsider BBS4 LOF; if anti-MKKS absent: reconsider BBS6/MKKS LOF",
            "7. Electron microscopy / super-resolution: IFT trains in cilia show reduced BBSome incorporation (IFT-B present, BBSome absent from trains) — research tool",
            "8. Tri-allelic BBS: screen BBS9 (most common third allele, ~5% of BBS8 families) and BBS4 (shared peripheral module position)",
            "9. CHD workup: echocardiogram (CHD ~4%; lowest BBS1-8 penetrance); ASD/VSD; lower priority but still indicated",
            "10. Renal: ultrasound + GFR; cystatin C annually; ESRD risk ~6%; ophthalmology ERG + Goldmann VF; OCT at diagnosis",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; photosensitising drug avoidance; vitamin A not indicated for BBS; gene therapy investigational",
            "2. Obesity: GLP-1 receptor agonists (semaglutide) — LepR mis-trafficking mechanism supports ciliary pathway benefit; bariatric surgery in refractory cases",
            "3. Renal: ACE inhibitor/ARB for CKD progression; renal transplant for ESRD (curative — no allograft recurrence); avoid nephrotoxins",
            "4. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling",
            "5. Anosmia: safety counselling (gas leaks, food spoilage); smell training investigational; smoke detectors mandatory",
            "6. Cognitive/LD: early intervention programmes; IEP; neuropsychological assessment; occupational therapy",
            "7. CHD (~4%): echo-guided cardiology follow-up; no specific BBS8 cardiac lesion profile; surgical/catheterisation as indicated",
            "8. Genetic counselling: AR inheritance; sibling recurrence 25%; prenatal/preimplantation genetic testing available",
            "9. BBS9 co-testing: always screen BBS9 in BBS8 families (tri-allelic risk ~5%; BBS9 peripheral module partner)",
            "10. Surveillance: annual ophthalmology + renal + metabolic (HbA1c, lipids) + cardiac; growth/pubertal charting in children",
        ],
        "ddx_table": [
            {"disease": "BBS2 (BBS2 gene)", "key_difference": "BBS2 LOF → absent BBS2 IF; in BBS8 LOF BBS2 IF is NORMAL (core dimer intact) — this is the key discriminator; gene panel mandatory"},
            {"disease": "BBS4 (BBS4 gene)", "key_difference": "BBS4 LOF → absent BBS4 IF (satellite tethering fails); in BBS8 LOF BBS4 IF is NORMAL (tethering intact); PCM1 co-IF pattern different — BBS4 shows reduced PCM1 colocalisation, BBS8 shows preserved PCM1 colocalisation"},
            {"disease": "BBS6 / MKKS (MKKS gene)", "key_difference": "BBS6 LOF → absent MKKS IF + absent BBS2 IF (BCC fails); BBS8 LOF → NORMAL MKKS IF + NORMAL BBS2 IF; BCC chaperonin distinction"},
            {"disease": "BBS7 (BBS7 gene)", "key_difference": "BBS7 LOF → absent BBS2 IF + absent BBS7 IF (core dimer mutual destabilisation); BBS8 LOF → NORMAL BBS2 IF — no core dimer destabilisation; gene panel distinguishes"},
            {"disease": "BBS9 / PTHB1 (BBS9 gene)", "key_difference": "BBS9 bridge subunit; BBS9 LOF → BBS9 IF absent; core dimer partly destabilised (BBS2 reduced); BBS8 LOF → BBS9 IF normal; PCM1 satellite assembly distinct; gene panel"},
            {"disease": "BBS1 (BBS1 gene)", "key_difference": "BBS1 LOF → absent BBS1 IF; M390R European founder (25% BBS); cargo loading defect (BBS1 is the IFT-A linker, not IFT-B dock); BBS8 is the IFT-B arm (different IFT complex)"},
            {"disease": "Alström Syndrome (ALMS1)", "key_difference": "Cone-rod (cone first, not rod first); no polydactyly; infantile DCM (60%); sensorineural hearing loss; normal TTC8/BBS2/BBS8 IF; ALMS1 mutations"},
            {"disease": "Joubert Syndrome (JBTS)", "key_difference": "Molar tooth sign on brain MRI; cerebellar vermis hypoplasia; TTC8 mutations not causative of classic JBTS; different IFT/transition-zone module"},
        ],
        "mechanism_glossary": [
            {
                "term": "TTC8 as the Sole BBSome-IFT-B Bridge",
                "definition": "Among all 8 BBSome structural subunits (BBS1, BBS2, BBS4, BBS5, BBS7, BBS8/TTC8, BBS9, BBIP10), TTC8 is the only one that directly contacts the IFT-B complex. The interaction is mediated by TTC8 TPR7–9 binding to IFT38 (CLUAP1/QiF1), a WD40-domain subunit of IFT-B. This contact is the molecular bridge that allows the assembled BBSome octamer to be loaded onto IFT-B particles and carried bidirectionally along the ciliary axoneme. In TTC8 LOF, this bridge is absent — BBSome assembles normally at PCM1 pericentriolar satellites (detectable by anti-BBS2 co-IF with anti-PCM1) but cannot be exported from satellites into the ciliary transition zone. This is the most distal step in the BBS biogenesis/trafficking pathway known to be disrupted by a BBSome subunit mutation."
            },
            {
                "term": "BBSome Assembles at Satellites but Fails IFT-B Export (BBS8-Specific)",
                "definition": "In BBS8/TTC8-null patient fibroblasts, the BBSome octamer forms normally: BBS2-BBS7 core dimer assembles (anti-BBS2 IF: normal), BBS4 tethers the forming octamer to PCM1 pericentriolar satellites (anti-BBS4 IF: normal), and the peripheral module assembles. The partially or fully assembled BBSome accumulates at PCM1 satellites because without TTC8's IFT-B arm, it cannot dock onto IFT-B particles and be exported. PCM1 co-IF with anti-BBS2 shows BBS2 puncta colocalised with PCM1 — this is UNIQUE to BBS8 LOF. In all other BBS subtypes with assembly defects (BBS2, BBS4, BBS6 LOF), the satellite pattern is abnormal. This makes BBS8 LOF identifiable by combined anti-BBS8 (absent) + anti-BBS2 (normal, satellite puncta) + anti-PCM1 co-IF staining."
            },
            {
                "term": "TTC8 Dual Role: IFT-B Docking AND ARL6 GTPase Contact",
                "definition": "TTC8 participates in two sequential ciliary trafficking events. First, ARL6-GTP (BBS3) contacts TTC8 TPR10 to recruit the assembled BBSome to the ciliary base membrane (ARL6 acts as the membrane GTPase switch). Second, TTC8 TPR7-9 docks the BBSome onto IFT-B for axonemal transport. Variants at the TPR8-9 loop (e.g., p.Arg413Gln) disrupt BOTH contacts simultaneously — the worst functional class of TTC8 variant, associated with early-onset severe retinal disease and high polydactyly penetrance. Hypomorphic alleles at TPR3 (BBS1 interface) or TPR12-13 (BBIP10 cap) preserve partial IFT-B docking and are associated with milder phenotypes."
            },
            {
                "term": "Tri-allelic BBS in BBS8 Families — BBS9 as Preferred Third Allele",
                "definition": "Tri-allelic BBS occurs when a patient carries biallelic pathogenic TTC8 variants plus a heterozygous pathogenic variant in a second BBS gene. In BBS8 families, BBS9 is the most common third allele (~5% of families), because BBS9 (the bridge subunit) contacts TTC8 directly in the assembled BBSome peripheral module — BBS9 Thr489 contacts TTC8 TPR4-5 in the octamer interface. A heterozygous BBS9 variant reduces the BBS9 available to stabilise BBS8's peripheral module position, worsening the IFT-B docking geometry. BBS4 is the second most common third allele (shared peripheral module spatial proximity). Tri-allelic BBS8+BBS9 families typically show earlier retinal onset and more severe renal phenotype."
            },
            {
                "term": "LepR Ciliary Mis-Trafficking (Shared BBS Obesity Mechanism)",
                "definition": "Leptin receptor (LepR) requires the BBSome octamer (specifically IFT-A-mediated retrograde transport) to be periodically removed from hypothalamic neuronal cilia. In BBS8 LOF, BBSome cannot enter the axoneme (stranded at satellites, no IFT-B docking) → LepR constitutively retained in cilia → signal desensitisation → leptin resistance → hyperphagia and progressive obesity despite elevated plasma leptin. The mechanism is identical in outcome to all BBSome subunit LOF (BBS1, 2, 4, 5, 7, 8) — LepR mis-trafficking is the unified obesity mechanism. GLP-1 receptor agonists (semaglutide) bypass this defect by acting on GLP1R-expressing interneurons that use a non-ciliary satiety pathway."
            },
            {
                "term": "IFT-B vs IFT-A: Why BBS8 and BBS1 LOF Produce the Same Phenotype via Different IFT Arms",
                "definition": "The BBSome uses both IFT complexes for ciliary trafficking. BBS1 (the cargo-recognition subunit) contacts IFT-A via its beta-propeller for retrograde BBSome transport out of the cilium. BBS8/TTC8 (the IFT-B arm) contacts IFT-B (specifically IFT38) for anterograde entry and cycling within the cilium. Both interactions are required for BBSome to transport GPCR cargoes. BBS1 LOF → anterograde cargo loading fails (GPCRs cannot be sorted into BBSome). BBS8 LOF → BBSome cannot enter the axoneme at all (stranded at satellites). Both converge on the same downstream GPCR mis-trafficking phenotype (rod-cone dystrophy, obesity, anosmia, etc.) despite disrupting opposite IFT arms."
            },
        ],
    }
