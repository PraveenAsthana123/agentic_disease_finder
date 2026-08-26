"""
BBS16 Bardet-Biedl Syndrome Type 16 — SDCCAG8 Dashboard
=========================================================
Primary Gene : SDCCAG8 (Serologically Defined Colon Cancer Antigen 8;
               also called CELS, OFD7, NPHP10).
               713 aa centrosomal satellite protein.
               Domain organisation (713 aa):
                 CC1 (aa 1–120): N-terminal coiled-coil;
                   self-oligomerisation; PCM1 interaction;
                 CC2 (aa 145–265): Second coiled-coil;
                   NPHP1 interaction interface (Leu192 hydrophobic core);
                 NPHP-module interaction domain (aa 290–450):
                   NPHP4 electrostatic contact surface (Glu485 residue);
                   NPHP8/RPGRIP1L binding; centrosomal satellite tethering;
                 CC3 (aa 470–600): Third coiled-coil;
                   structural scaffolding; Trp568 hydrophobic core;
                 C-terminal segment (aa 600–713):
                   PCM1/centriolar satellite association;
                   Arg645 terminates PCM1-tethering domain.

               MECHANISM:
               SDCCAG8 localises to pericentriolar material (PCM1-positive
               centriolar satellites) and to the subdistal appendages of the
               mother centriole. It anchors the NPHP ciliopathy module
               (NPHP1, NPHP4, NPHP5/IQCB1, NPHP8/RPGRIP1L) to centriolar
               satellites, which is required for proper cilia formation
               (ciliogenesis initiation). SDCCAG8 LOF disrupts satellite-
               mediated delivery of TZ proteins to the ciliary base:
                 1. Cilia are shortened or absent in key cell types
                    (photoreceptors, renal tubule cells, neuronal cilia,
                    limb bud mesenchyme);
                 2. BBSome cannot localise to basal body (no basal body
                    appendage platform);
                 3. LepR and GPCR cargo accumulate in cell body (cannot
                    enter/signal via cilia) → leptin resistance → obesity;
                 4. Photoreceptor ciliary compartment fails to form →
                    opsin mislocalisation → rod-cone degeneration;
                 5. Renal NPHP-type injury (tubular atrophy, interstitial
                    fibrosis) from loss of tubular ciliary signaling.
               SDCCAG8 LOF is a CILIOGENESIS defect — upstream of BBSome,
               upstream of IFT, upstream of TZ: no cilia → no downstream
               ciliary biology. Mechanistically distinct from BBS1-12
               (BBSome LOF), BBS13-14 (TZ scaffold LOF), and BBS15 (IFT-B
               GTPase retrograde).

               NPHP10 alias: Same gene causes NPHP10 (nephronophthisis 10)
               and SLS7 (Senior-Løken Syndrome 7 — retinal dystrophy +
               NPHP) in more severe allele classes. Allele class governs:
                 Biallelic null → SLS7/NPHP10 (severe; early ESRD);
                 Hypomorphic/null compound → BBS16 (full multi-system);
                 Biallelic hypomorphic → milder BBS16 or isolated retinal.

               IF fingerprint — distinguishes BBS16:
                 SDCCAG8 ABSENT from pericentriolar satellites / centriolar
                   subdistal appendages.
                 PCM1 satellites DISORGANISED (not fully absent; SDCCAG8
                   tethering function lost → satellites scattered).
                 NPHP1/NPHP4 DELOCALIZED from ciliary base (satellite
                   delivery failure).
                 Cilia SHORTENED or ABSENT in affected cells (ciliogenesis
                   defect — key contrast with BBS13/BBS14 where TZ gate is
                   leaky but cilia form).
                 BBSome IF: REDUCED/ABSENT at basal body (no platform);
                   not specifically trapped (contrast BBS15).
                 MKKS/BCC: NORMAL (upstream chaperonin intact — BBSome
                   assembly unaffected; it is delivery/basal body anchoring
                   that fails).
                 MKS1: REDUCED/DELOCALIZED (NPHP module disruption; MKS1
                   is downstream of satellite delivery — in contrast to
                   BBS13 where MKS1 itself is absent).

Disease        : Bardet-Biedl Syndrome Type 16 (#615993)
                 Also allele-class: NPHP10 (#613615); SLS7 (#616629)
Frequency      : <1% of all BBS (very rare); ~40-60 families worldwide 2026
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~3% of BBS16 families
Cohort         : 40 patients; educational/synthetic; seed 363
Endpoints      : /api/bbs16/overview | /api/bbs16/breakdown | /api/bbs16/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 363
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("MENA/Bedouin",  0.35),  # Bedouin Arab — Otto 2010 original discovery cohort
        ("European",      0.25),
        ("South Asian",   0.22),  # India, Pakistan — consanguineous
        ("North African", 0.10),  # Tunisia, Morocco
        ("East Asian",    0.05),
        ("Other",         0.03),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "MENA/Bedouin":  ["c.483+1G>A (splice, intron 4)", "p.Arg279Ter (c.835C>T)", "p.Glu485Lys (c.1453G>A)"],
        "European":      ["p.Leu192Pro (c.575T>C)", "p.Arg279Ter (c.835C>T)", "p.Arg645Ter (c.1933C>T)"],
        "South Asian":   ["p.Trp568Gly (c.1702T>G)", "p.Glu485Lys (c.1453G>A)", "p.Arg645Ter (c.1933C>T)"],
        "North African": ["p.Arg279Ter (c.835C>T)", "p.Leu192Pro (c.575T>C)", "p.Arg645Ter (c.1933C>T)"],
        "East Asian":    ["p.Arg645Ter (c.1933C>T)", "p.Trp568Gly (c.1702T>G)", "p.Glu485Lys (c.1453G>A)"],
        "Other":         ["p.Arg645Ter (c.1933C>T)", "p.Arg279Ter (c.835C>T)"],
    }

    allele_classes = [
        "Compound het missense+missense",        # 40% — two missense, residual SDCCAG8 function
        "Compound het missense+truncating",       # 32% — mixed; moderate-severe
        "Homozygous missense (consanguineous)",   # 20% — founder variants MENA/South Asian
        "Biallelic truncating",                   # 5%  — complete null; SLS7/NPHP10 overlap
        "Tri-allelic",                            # 3%
    ]
    allele_weights = [0.40, 0.32, 0.20, 0.05, 0.03]

    retinal_stages = [
        "Nyctalopia only (early)",
        "Bone-spicule RP pattern (moderate)",
        "Severe field loss (<20°)",
        "Legal blindness / end-stage",
    ]
    ret_w = [0.18, 0.30, 0.30, 0.22]

    polydactyly_types = [
        "Post-axial hands+feet",
        "Post-axial hands only",
        "Post-axial feet only",
        "Pre-axial (rare)",
        "None",
    ]
    # ~48% polydactyly (ciliogenesis defect; limb bud mesenchyme cilia required for Shh)
    poly_w = [0.25, 0.12, 0.07, 0.04, 0.52]

    # BBS16 renal: NPHP-type (tubular atrophy/fibrosis) — like BBS13-14, NOT cystic
    renal_types = [
        "Tubular atrophy + interstitial fibrosis (NPHP pattern)",
        "Concentrating defect only",
        "Corticomedullary scarring",
        "CKD stage 2-3 (ESRD risk)",
        "None",
    ]
    renal_w = [0.22, 0.12, 0.10, 0.09, 0.47]  # ~53% renal

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.18, 0.35, 0.30, 0.17]

    misdiag = [
        "Non-syndromic retinitis pigmentosa (no systemic features initially)",
        "Nephronophthisis only (NPHP10 misclassified; BBS features overlooked)",
        "Senior-Løken Syndrome (retinal + renal; BBS features incomplete at presentation)",
        "BBS without molecular subtype (gene panel incomplete)",
        "No initial misdiagnosis",
    ]
    mis_w = [0.25, 0.15, 0.10, 0.20, 0.30]

    presentation_ages = [
        "Birth (polydactyly noted)",
        "Infancy (nystagmus/photophobia)",
        "Early childhood (RP symptoms)",
        "School-age (obesity + RP)",
        "Adolescent/adult",
    ]
    pres_w = [0.20, 0.18, 0.28, 0.24, 0.10]

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

        # BBS16: ~48% polydactyly, ~62% obesity, ~38% cognitive LD,
        #        ~50% hypogonadism, ~42% anosmia, ~53% renal (NPHP-type),
        #        ~5% CHD, ~10% SLS7 overlap (biallelic null allele class)
        polydactyly  = poly_type != "None"
        obesity      = rng.random() < 0.62
        cognitive_ld = rng.random() < 0.38
        hypogonadism = rng.random() < 0.50
        anosmia      = rng.random() < 0.42
        renal_any    = renal_type != "None"
        chd          = rng.random() < 0.05
        ret_end      = ret_stage == "Legal blindness / end-stage"
        tri_bbs      = allele_cl == "Tri-allelic"
        mis_dx       = mis_d != "No initial misdiagnosis"
        consang      = eth in ("MENA/Bedouin", "South Asian", "North African") and rng.random() < 0.72
        sls7_overlap = allele_cl == "Biallelic truncating" and rng.random() < 0.60
        # PCM1 satellites disorganised in all BBS16 (SDCCAG8 tethering lost)
        pcm1_disorg  = True

        patients.append({
            "id":                   f"BBS16-{i+1:03d}",
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
            "consanguinity":        consang,
            "sls7_overlap":         sls7_overlap,
            "pcm1_disorganised":    pcm1_disorg,
        })
    return patients


_COHORT: List[Dict[str, Any]] = _make_cohort()


def _pct(n: int, total: int = _N) -> float:
    return round(100 * n / total, 1) if total else 0.0


# ── API handlers ──────────────────────────────────────────────────────────────
def get_overview() -> Dict[str, Any]:
    c = _COHORT
    n = _N

    poly_n        = sum(1 for p in c if p["polydactyly"])
    obesity_n     = sum(1 for p in c if p["obesity"])
    cognitive_n   = sum(1 for p in c if p["cognitive_ld"])
    renal_any_n   = sum(1 for p in c if p["renal_any"])
    hypo_n        = sum(1 for p in c if p["hypogonadism"])
    anosmia_n     = sum(1 for p in c if p["anosmia"])
    chd_n         = sum(1 for p in c if p["chd"])
    ret_end_n     = sum(1 for p in c if p["retinal_endstage"])
    tri_n         = sum(1 for p in c if p["triallelic_bbs"])
    mis_n         = sum(1 for p in c if p["initial_misdiagnosis"])
    consang_n     = sum(1 for p in c if p["consanguinity"])
    sls7_n        = sum(1 for p in c if p["sls7_overlap"])

    ret_stages: Dict[str, int] = {}
    for p in c:
        s = p["retinal_stage"]
        ret_stages[s] = ret_stages.get(s, 0) + 1

    age_dist: Dict[str, int] = {}
    for p in c:
        g = p["dx_age_group"]
        age_dist[g] = age_dist.get(g, 0) + 1

    return {
        "cohort_n":       n,
        "gene":           "SDCCAG8 / NPHP10",
        "omim_gene":      "*613524",
        "omim_disease":   "#615993 (BBS16)",
        "locus":          "1q43-44",
        "protein":        "713 aa — centrosomal satellite protein; CC1 (aa 1–120; oligomerisation/PCM1 contact); CC2 (aa 145–265; NPHP1 interaction); NPHP-module interaction domain (aa 290–450; NPHP4/NPHP8 contacts); CC3 (aa 470–600; structural scaffolding); C-terminal segment (aa 600–713; PCM1 satellite tethering)",
        "mechanism":      "SDCCAG8 anchors the NPHP ciliopathy module (NPHP1, NPHP4, NPHP5/IQCB1, NPHP8) to PCM1-positive centriolar satellites, enabling delivery of TZ proteins to the ciliary base. SDCCAG8 LOF → centriolar satellite disorganisation → cilia shortened/absent (ciliogenesis defect) → BBSome/IFT/TZ signaling all fail secondarily. LepR/GPCR cargo cannot enter cilia → leptin resistance (obesity) + photoreceptor opsin mislocalisation (retinal degeneration). UPSTREAM ciliogenesis defect — mechanistically distinct from BBSome LOF (BBS1-12), TZ LOF (BBS13-14), IFT-B GTPase LOF (BBS15).",
        "bbs_frequency":  "<1% of all BBS (very rare); ~40-60 families worldwide 2026",
        "inheritance":    "Autosomal recessive; biallelic LOF",
        "triallelic_pct": 3,
        "allele_class_note": "Compound het missense+missense (40%); compound het missense+truncating (32%); homozygous missense consanguineous (20%); biallelic truncating (5%); tri-allelic (3%)",
        "if_fingerprint": "SDCCAG8 ABSENT from pericentriolar satellites; PCM1 satellites DISORGANISED; NPHP1/NPHP4 DELOCALIZED from ciliary base; Cilia SHORTENED/ABSENT (ciliogenesis defect — contrast BBS13/14 where cilia form but TZ gate leaky); BBSome REDUCED at basal body (no platform); MKKS/BCC NORMAL; MKS1 REDUCED/DELOCALIZED",
        "nphp10_note":    "SDCCAG8 is also NPHP10 — same gene causes NPHP10 (isolated nephronophthisis) and SLS7 (Senior-Løken Syndrome 7) in severe/biallelic null allele classes. Allele class governs disease tier: biallelic null → SLS7/NPHP10 (ESRD + severe retinal); compound hypomorphic/null → BBS16.",
        "key_counts": {
            "polydactyly_n":    poly_n,     "polydactyly_pct":   _pct(poly_n),
            "obesity_n":        obesity_n,   "obesity_pct":       _pct(obesity_n),
            "cognitive_n":      cognitive_n, "cognitive_pct":     _pct(cognitive_n),
            "renal_any_n":      renal_any_n, "renal_any_pct":    _pct(renal_any_n),
            "hypogonadism_n":   hypo_n,      "hypogonadism_pct": _pct(hypo_n),
            "anosmia_n":        anosmia_n,   "anosmia_pct":      _pct(anosmia_n),
            "chd_n":            chd_n,       "chd_pct":          _pct(chd_n),
            "retinal_endstage_n": ret_end_n, "retinal_endstage_pct": _pct(ret_end_n),
            "triallelic_n":     tri_n,       "triallelic_pct":   _pct(tri_n),
            "misdiagnosis_n":   mis_n,       "misdiagnosis_pct": _pct(mis_n),
            "consanguinity_n":  consang_n,   "consanguinity_pct": _pct(consang_n),
            "sls7_overlap_n":   sls7_n,      "sls7_overlap_pct": _pct(sls7_n),
        },
        "systemic_burden": [
            ("Rod-Cone Dystrophy (rod-first)",                 n,           _pct(n)),
            ("Obesity (ciliogenesis defect; LepR absent cilia)",obesity_n,  _pct(obesity_n)),
            ("Renal anomaly (NPHP-type tubular/fibrotic)",     renal_any_n, _pct(renal_any_n)),
            ("Polydactyly (post-axial)",                       poly_n,      _pct(poly_n)),
            ("Hypogonadism",                                   hypo_n,      _pct(hypo_n)),
            ("Anosmia/Hyposmia",                              anosmia_n,   _pct(anosmia_n)),
            ("Cognitive/LD",                                   cognitive_n, _pct(cognitive_n)),
            ("CHD",                                            chd_n,       _pct(chd_n)),
            ("SLS7 overlap (severe retinal + NPHP)",           sls7_n,      _pct(sls7_n)),
        ],
        "retinal_stage_distribution": {k: v for k, v in ret_stages.items()},
        "dx_age_distribution":        {k: v for k, v in age_dist.items()},
    }


def get_breakdown() -> Dict[str, Any]:
    c = _COHORT
    n = _N

    systemic = [
        ("Rod-Cone Dystrophy (rod-first)",                  n,                                                    _pct(n)),
        ("Obesity (ciliogenesis; LepR absent cilia)",        sum(1 for p in c if p["obesity"]),                   _pct(sum(1 for p in c if p["obesity"]))),
        ("Renal anomaly (NPHP-type)",                        sum(1 for p in c if p["renal_any"]),                 _pct(sum(1 for p in c if p["renal_any"]))),
        ("Polydactyly (post-axial)",                         sum(1 for p in c if p["polydactyly"]),               _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Hypogonadism",                                     sum(1 for p in c if p["hypogonadism"]),              _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Anosmia/Hyposmia",                                sum(1 for p in c if p["anosmia"]),                   _pct(sum(1 for p in c if p["anosmia"]))),
        ("Cognitive/LD",                                     sum(1 for p in c if p["cognitive_ld"]),              _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("CHD",                                              sum(1 for p in c if p["chd"]),                       _pct(sum(1 for p in c if p["chd"]))),
        ("SLS7 overlap",                                     sum(1 for p in c if p["sls7_overlap"]),              _pct(sum(1 for p in c if p["sls7_overlap"]))),
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
            for r in ["Tubular atrophy + interstitial fibrosis (NPHP pattern)",
                      "Concentrating defect only", "Corticomedullary scarring",
                      "CKD stage 2-3 (ESRD risk)", "None"]
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
            for m in ["Non-syndromic retinitis pigmentosa (no systemic features initially)",
                      "Nephronophthisis only (NPHP10 misclassified; BBS features overlooked)",
                      "Senior-Løken Syndrome (retinal + renal; BBS features incomplete at presentation)",
                      "BBS without molecular subtype (gene panel incomplete)",
                      "No initial misdiagnosis"]
        ],
        "top_variants": [{"variant": v, "n": cnt} for v, cnt in top_variants],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":             "SDCCAG8 / NPHP10 (Serologically Defined Colon Cancer Antigen 8; also CELS, OFD7); Chr 1q43-44",
            "OMIM_Gene":        "*613524",
            "Chromosome":       "1q43-44",
            "Protein":          "713 aa — centrosomal satellite protein; CC1 (aa 1–120; oligomerisation, PCM1 contact); CC2 (aa 145–265; NPHP1 interaction, Leu192 hydrophobic core); NPHP-module interaction domain (aa 290–450; NPHP4 electrostatic contact Glu485, NPHP8/RPGRIP1L binding, satellite tethering); CC3 (aa 470–600; structural scaffolding, Trp568 hydrophobic core); C-terminal segment (aa 600–713; PCM1 satellite association, Arg645 terminates tethering domain)",
            "Domains":          "CC1 (aa 1–120): self-oligomerisation, PCM1 centriolar satellite interaction; CC2 (aa 145–265): NPHP1 interaction surface, Leu192 hydrophobic core; NPHP-module domain (aa 290–450): NPHP4 electrostatic contact (Glu485), NPHP8/RPGRIP1L binding, centrosomal satellite tethering; CC3 (aa 470–600): structural scaffolding, Trp568 hydrophobic core; C-terminal (aa 600–713): PCM1/satellite association, Arg645 key tethering residue",
            "Module":           "Pericentriolar material (PCM1-positive centriolar satellites) and subdistal appendages of mother centriole. SDCCAG8 anchors NPHP module (NPHP1, NPHP4, NPHP5/IQCB1, NPHP8/RPGRIP1L) to centriolar satellites for TZ protein delivery to ciliary base — required for ciliogenesis initiation. NOT a BBSome subunit, NOT a BCC chaperonin, NOT a TZ Y-link scaffold (distinct from BBS1-15).",
            "Inheritance":      "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":    "<1% of all BBS; very rare; ~40-60 families worldwide 2026",
            "Allele_spectrum":  "Compound het missense+missense (40%); compound het missense+truncating (32%); homozygous missense consanguineous (20%); biallelic truncating (5%); tri-allelic (3%)",
            "Triallelic_BBS":   "~3%",
            "NPHP10_SLS7_note": "SDCCAG8 causes NPHP10 (isolated nephronophthisis; *613524/#613615) and SLS7 (Senior-Løken Syndrome 7; retinal dystrophy + NPHP; #616629) in severe/biallelic null allele classes. Allele class governs tier: biallelic null → SLS7/NPHP10; compound hypomorphic/null → BBS16. Same gene, disease tier determined by residual SDCCAG8 function.",
            "Population_note":  "MENA/Bedouin (Otto 2010 original discovery — homozygous c.483+1G>A splice mutation in Bedouin BBS families); European; South Asian. High consanguinity rate (Bedouin, South Asian).",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 16 (BBS16) — #615993; also NPHP10 (#613615); SLS7 (#616629)",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS16 subtype <1% (~1:10,000,000–25,000,000) — very rare",
            "Cardinal_features": "Rod-cone dystrophy (rod-first, scotopic extinguished) · Post-axial polydactyly (~48%) · Obesity (~62%; ciliogenesis defect → LepR absent from cilia) · Hypogonadism (~50%) · Cognitive/LD (~38%) · Renal anomalies (~53%; NPHP-type — tubular atrophy/fibrosis) · Anosmia (~42%) · SLS7 overlap (~10%; biallelic null allele class)",
            "BBS16_unique":      "SDCCAG8 is a CILIOGENESIS factor — upstream of BBSome, IFT, and TZ. LOF → cilia shortened/absent (not leaky gate, not retrograde trap, not BBSome assembly fail — cilia barely form). NPHP10 allele alias: same gene causes nephronophthisis and SLS7 in null allele classes. MENA/Bedouin enrichment. PCM1 satellites disorganised (SDCCAG8 tethering lost). Renal = NPHP-type (tubular atrophy/fibrosis; like BBS13-14), NOT cystic (contrast BBS1-12, BBS15). No liver fibrosis. No Joubert overlap.",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; same ERG pattern as BBS1-15; 100% penetrance); ciliogenesis defect → photoreceptor connecting cilium fails → opsin mislocalisation to cell body",
            "Obesity_mechanism": "SDCCAG8 LOF → cilia absent/shortened in hypothalamic neurons and adipose stromal cells → LepR cannot localise to/signal from cilia → leptin resistance → obesity; mechanism upstream of BBSome (no retrograde trafficking defect required — cilia don't form in first place); same clinical endpoint as all other BBS types",
            "Renal":             "NPHP-type pattern: tubular atrophy, interstitial fibrosis, concentrating defect, CKD; same as BBS13 (MKS1) and BBS14 (CEP290) — all three share NPHP module connection; ESRD risk in adolescence/young adult especially with biallelic null alleles (SLS7 tier); DISTINCT from cystic pattern of BBS1-12 and BBS15",
            "Liver":             "No liver fibrosis — SDCCAG8 does not affect hepatic ductal plate; standard monitoring only (same as BBS14-15, contrast BBS13/MKS1 where liver fibrosis occurs in 12-15%)",
            "No_JBTS_overlap":   "No Joubert syndrome overlap reported — distinguishes BBS16 from BBS13 (JBTS28 8%) and BBS14 (JBTS5 10%); no molar tooth sign expected",
            "SLS7_NPHP10":       "~10% of BBS16 cohort meet SLS7 criteria (severe retinal + NPHP; earlier ESRD); biallelic truncating allele class; register with NPHP and BBS registries; ESRD planning from adolescence",
            "CHD":               "~5% (mid-range BBS spectrum; less than BBS10 at 7%, more than BBS15 at 3%)",
            "Diagnosis":         "Full BBS gene panel (minimum 20-24 genes including SDCCAG8 at Chr 1q43-44); IF: SDCCAG8 ABSENT from pericentriolar satellites; PCM1 satellites DISORGANISED; NPHP1/NPHP4 delocalized; cilia shortened/absent on acetylated α-tubulin staining; BBSome REDUCED at basal body; MKKS NORMAL",
            "Treatment":         "Symptomatic; GLP-1 receptor agonists (semaglutide) for obesity; renal NPHP surveillance (renal USS, GFR annually, ACE-i/ARB for hypertension/proteinuria); ophthalmology surveillance (ERG + OCT + Goldmann VF); biallelic null → ESRD planning (transplant evaluation from adolescence); no gene therapy in clinical trials 2026",
            "Prognosis":         "Progressive visual loss; renal NPHP involvement ~53% (tubular atrophy/fibrosis; ESRD risk in biallelic null ~10%); near-normal survival with surveillance and renal replacement; no specific targeted therapy 2026",
        },
        "key_variants": [
            {
                "variant":     "c.483+1G>A (splice site, intron 4)",
                "domain":      "Intron 4/exon 5 boundary; loss of canonical splice donor → exon 5 skipping (or cryptic splice site activation) → frameshift or in-frame deletion within CC2 (aa 145–265); disrupts NPHP1 interaction surface",
                "consequence": "Null/severe hypomorphic allele; originally reported in Bedouin Arab BBS families (Otto et al. 2010 Nature Genetics — first BBS16 description); homozygous in Bedouin consanguineous kindreds; loss of CC2/NPHP1 interaction surface; NPHP module satellite delivery abolished; full BBS16 + NPHP10 features; MENA/Bedouin enrichment",
                "ethnicity":   "MENA/Bedouin enrichment (Bedouin Arab founding mutation; first described by Otto 2010)"
            },
            {
                "variant":     "p.Arg279Ter (c.835C>T)",
                "domain":      "CC2/NPHP-module junction (aa 265–290 boundary); truncation removes NPHP-module interaction domain (aa 290–450), CC3 (aa 470–600), and C-terminal PCM1 tethering segment (aa 600–713); most critical SDCCAG8 functional domains lost",
                "consequence": "Null allele; complete loss of NPHP4/NPHP8 interactions and PCM1 satellite tethering; SDCCAG8 absent from satellites; NPHP module completely delocalized from ciliary base; severe ciliogenesis failure; pan-ethnic; Bedouin/North African enrichment; biallelic truncating (two truncating alleles) → SLS7/NPHP10 overlap; most commonly detected truncating allele in non-consanguineous BBS16 families",
                "ethnicity":   "Pan-ethnic; Bedouin/North African enrichment; most common truncating BBS16 allele"
            },
            {
                "variant":     "p.Leu192Pro (c.575T>C)",
                "domain":      "CC2 hydrophobic core (aa 145–265; Leu192 is a conserved leucine zipper-type hydrophobic residue in the NPHP1 interaction coiled-coil; Pro substitution introduces a helix-breaking residue disrupting CC2 alpha-helical packing and NPHP1 contact surface)",
                "consequence": "Hypomorphic allele; SDCCAG8 CC2 helix destabilised; NPHP1 interaction surface disrupted; partial ciliogenesis retained (residual SDCCAG8 satellite function); BBS16 phenotype (not SLS7); milder renal involvement; European enrichment; often compound het with truncating allele",
                "ethnicity":   "European enrichment (British, German, Dutch families); compound het partner in European BBS16 families"
            },
            {
                "variant":     "p.Glu485Lys (c.1453G>A)",
                "domain":      "NPHP-module interaction domain (aa 290–450; Glu485 provides the key acidic contact for the electrostatic interaction with NPHP4's basic surface; Lys substitution introduces charge reversal disrupting NPHP4 binding to SDCCAG8)",
                "consequence": "Hypomorphic allele; SDCCAG8-NPHP4 interaction disrupted (charge reversal at Glu485); NPHP4 fails to bind SDCCAG8 at satellites; partial NPHP module delivery defect; CC1/CC2 intact (NPHP1 binding partially preserved); BBS16 phenotype with prominent renal involvement (NPHP4 pathway disrupted); MENA enrichment; moderate phenotype",
                "ethnicity":   "MENA enrichment (Jordan, Turkey, Iran); often homozygous in consanguineous MENA families"
            },
            {
                "variant":     "p.Trp568Gly (c.1702T>G)",
                "domain":      "CC3 hydrophobic core (aa 470–600; Trp568 is a critical indole-ring residue in the CC3 hydrophobic core; Gly substitution removes the bulky aromatic side chain, destabilising CC3 alpha-helical packing and reducing C-terminal segment stability)",
                "consequence": "Hypomorphic allele; CC3 structural scaffolding disrupted; C-terminal PCM1 tethering partially compromised; residual NPHP module delivery (CC1/CC2 intact); BBS16 phenotype (not SLS7); milder renal and retinal severity; South Asian enrichment; often homozygous in consanguineous South Asian families",
                "ethnicity":   "South Asian enrichment (India, Pakistan consanguineous families); often homozygous"
            },
            {
                "variant":     "p.Arg645Ter (c.1933C>T)",
                "domain":      "C-terminal segment (aa 600–713; truncation at aa 645 removes the PCM1/centriolar satellite association domain; Arg645 is within the region required for satellite particle incorporation)",
                "consequence": "Null allele (C-terminal truncation); PCM1 satellite association lost; SDCCAG8 cannot anchor to centriolar satellites → complete NPHP module delivery failure; CC1/CC2/NPHP-domain/CC3 may produce truncated polypeptide but cannot integrate into satellite scaffold; pan-ethnic; biallelic with missense → BBS16; biallelic truncating → SLS7/NPHP10 risk",
                "ethnicity":   "Pan-ethnic; common truncating allele in non-consanguineous families of any ancestry"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG rod-first, scotopic extinguished) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia); note NPHP-type renal (tubular atrophy/fibrosis) is expected in BBS16 — NOT cystic",
            "2. Full BBS gene panel (minimum 20–24 genes including SDCCAG8 at Chr 1q43-44); SDCCAG8 has 18 exons; ensure full coding coverage including splice sites (c.483+1G>A is intronic — require splice-site aware calling); WES/WGS preferred for deep intronic variants",
            "3. Allele classification: classify both SDCCAG8 alleles; biallelic truncating → SLS7/NPHP10 tier (early ESRD; begin nephrology planning from diagnosis); compound missense+truncating → BBS16 (moderate); biallelic missense → mildest BBS16",
            "4. IF panel (BBS16 pattern): SDCCAG8 ABSENT from pericentriolar satellites (anti-SDCCAG8 IF); PCM1 satellites DISORGANISED (anti-PCM1 IF — not absent but scattered); NPHP1/NPHP4 DELOCALIZED (anti-NPHP1 / anti-NPHP4 IF); acetylated α-tubulin staining: cilia SHORTENED or ABSENT; BBSome (anti-BBS2/BBS9) REDUCED at basal body; MKKS NORMAL; MKS1 REDUCED/DELOCALIZED (NPHP module disruption)",
            "5. Distinguish from BBS13 (MKS1 ABSENT on IF; normal PCM1 satellites; cilia present but TZ gate leaky) and BBS14 (MKS1 NORMAL; CEP290 absent; cilia form with leaky TZ; NPHP5/IQCB1 deranged); BBS16 is the only BBS type where cilia are SHORTENED/ABSENT",
            "6. Renal: NPHP-type surveillance — renal ultrasound at diagnosis; annual GFR, electrolytes, urine concentrating ability (osmolality); urine protein/creatinine; blood pressure; ACE-i/ARB for hypertension/proteinuria; biallelic null alleles → renal transplant evaluation from early adolescence (ESRD risk); NOT cystic (no corticomedullary cysts expected — contrast BBS1-12/BBS15)",
            "7. Ophthalmology: ERG (rod-first, scotopic extinguished) + Goldmann VF + OCT retinal thickness annually; ciliogenesis defect → photoreceptor connecting cilium failure → opsin mislocalisation; low-vision aids when VF <20°; no BBS16-specific gene therapy in clinical trials 2026 (SDCCAG8 at 713 aa may require AAV1 capacity)",
            "8. SLS7 screening: In all BBS16 patients with biallelic truncating alleles, monitor for SLS7 progression (early renal failure + severe retinal); register in SLS/NPHP registry; nephrology co-management from diagnosis",
            "9. Liver: No specific liver monitoring required (no ductal plate malformation in BBS16 — unlike BBS13/MKS1); standard LFTs at routine health check only",
            "10. Genetic counselling: AR inheritance; sibling recurrence 25%; MENA/Bedouin/South Asian families: extended family testing for c.483+1G>A (Bedouin) or consanguinity mapping; SDCCAG8 is also NPHP10 — counsel for allele-class dependent severity (biallelic null = SLS7/NPHP10 risk); IVF/PGT available",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; avoid photosensitising drugs; annual ERG + OCT + Goldmann VF surveillance; photoreceptor cilia fail to form (connecting cilium; ciliogenesis defect) → opsin mislocalises to cell body → rod-cone degeneration; no BBS16-specific gene therapy in clinical trials 2026 (SDCCAG8 713 aa — feasible for AAV delivery but not yet in trials); refer to retinal dystrophy centre; register in NPHP/BBS registries for trial eligibility",
            "2. Obesity: GLP-1 receptor agonists (semaglutide/liraglutide) — ciliogenesis defect → LepR cannot signal from absent cilia → leptin resistance → obesity; same GLP-1RA clinical strategy as all BBS types; bariatric surgery in refractory cases after ophthalmology risk stratification",
            "3. Renal (NPHP-type, NOT cystic): Annual renal ultrasound (NPHP pattern — cortical thinning, increased echogenicity, NO cysts); annual GFR, electrolytes, urine osmolality; urine protein/creatinine; blood pressure; ACE-i/ARB for proteinuria/hypertension; strict nephrotoxin avoidance (NSAIDs, IV contrast, aminoglycosides); biallelic null → ESRD in adolescence/young adult → transplant evaluation from age 10-12; living donor transplant preferred (siblings must be tested first — 25% recurrence risk); post-transplant: renal function restored but retinal/systemic BBS features persist",
            "4. Liver: No specific liver monitoring required (unlike BBS13); standard annual LFTs at routine health check only",
            "5. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling (usually infertile in BBS); sperm/oocyte banking discussion in adolescence",
            "6. Anosmia: safety counselling (gas leaks, food spoilage detection, smoke); smoke detectors mandatory; dietary support; olfactory rehabilitation",
            "7. Cognitive/LD: early neuropsychological assessment; individualised education plan (IEP); multidisciplinary developmental support; no cerebellar/JBTS overlap in BBS16 — cerebellar ataxia not expected",
            "8. SLS7 overlap (biallelic null): nephrology from diagnosis; renal transplant planning; ophthalmology acceleration (earlier end-stage retinal in SLS7); combined NPHP + BBS registry registration",
            "9. Emerging research (2026): SDCCAG8 satellite delivery pathway is a novel therapeutic target; PCM1/satellite biology being explored in multiple ciliopathies; SDCCAG8 AAV gene augmentation technically feasible (713 aa, ~2.1 kb CDS — within AAV5/8 capacity); no clinical trials yet; patients should register with BBS and NPHP registries",
        ],
    }
