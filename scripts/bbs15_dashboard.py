"""
BBS15 Bardet-Biedl Syndrome Type 15 — IFT27 / RABL4 Dashboard
==============================================================
Primary Gene : IFT27 (*615870), also known as RABL4 (RAB-like GTPase 4).
               IFT27 is a 225 aa small Rab-like GTPase residing within the
               IFT-B1 sub-complex. It is the ONLY IFT-B subunit classified
               as a BBS gene.
               Domain organisation (225 aa):
                 N-terminal GTPase domain (aa 1–180):
                   P-loop / Walker A motif (aa 12–19): GTP binding;
                   Switch I (aa 56–75): GTP hydrolysis rate modulator;
                     contacts BBSome indirectly via BBS3/ARL6;
                   Switch II (aa 90–110): IFT25/HSPB11 dimer interface;
                     BBS3/ARL6 interaction surface;
                   Interswitch (aa 78–90): conformational relay;
                 C-terminal IFT-B recruitment tail (aa 180–225):
                   Required for incorporation into the IFT-B1 complex;
                   contacts IFT38 and IFT172 within the IFT-B scaffold.
               IFT27 forms an obligate heterodimer with IFT25 (HSPB11)
               within the IFT-B1 complex. Loss of IFT27 destabilises
               IFT25 (IFT25 becomes monomeric and is cleared by the
               proteasome — IFT25 reduced/absent on IF when IFT27 is LOF).

               MECHANISM — UNIQUE AMONG ALL BBS GENES:
               IFT27 in its GTP-bound state recruits the BBSome for
               retrograde exit from cilia by contacting BBS3/ARL6 at the
               ciliary base. IFT27 LOF → BBSome enters cilia normally
               (assembly intact, IFT-B loading intact for anterograde
               transit) but CANNOT EXIT retrograde. This causes:
                 1. BBSome ACCUMULATION inside cilia (ciliary tip
                    signal ELEVATED on IF — pathognomonic for BBS15);
                 2. BBSome subunits PRESENT at cilia base (assembled
                    normally on IF — unlike BBS2/BBS6 where assembly fails);
                 3. GPCR cargo (including LepR, photoreceptor opsins)
                    accumulates inappropriately within the ciliary
                    compartment → multi-system ciliopathy.
               This is the OPPOSITE failure mode from most BBS genes where
               BBSome fails to assemble or load — in BBS15 assembly is
               normal, anterograde IFT loading is normal, but retrograde
               BBSome retrieval from cilia is impaired.

               IF fingerprint — UNIQUE, distinguishes BBS15:
                 BBSome subunits PRESENT at cilia base (assembled normally —
                   unlike BBS2/BBS6 where BBSome absent from base).
                 BBSome ACCUMULATES inside cilia / ciliary tip (elevated tip
                   signal — pathognomonic, anti-phase to BBS2/BBS6).
                 IFT27 ABSENT from IFT-B complex.
                 IFT25/HSPB11 DESTABILISED (dimer partner monomeric).
                 MKKS/BCC NORMAL (upstream chaperonin intact).
                 MKS1 NORMAL (TZ scaffold intact).

Disease        : Bardet-Biedl Syndrome Type 15 (#209900)
Frequency      : <1% of all BBS (very rare); ~30-50 families worldwide 2026
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~3% of BBS15 families
Cohort         : 40 patients; educational/synthetic; seed 361
Endpoints      : /api/bbs15/overview | /api/bbs15/breakdown | /api/bbs15/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 361
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("European",      0.30),   # British, French, Dutch — earlier BBS15 reports
        ("MENA",          0.28),   # Saudi Arabia, Jordan, Iran, Turkey — consanguineous
        ("South Asian",   0.20),   # India, Pakistan — consanguineous
        ("North African", 0.12),   # Tunisia, Morocco — consanguineous
        ("East Asian",    0.05),
        ("Other",         0.05),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "European":      ["p.Val43Asp (c.128T>A)", "p.Arg95Cys (c.283C>T)", "p.Arg160Ter (c.478C>T)"],
        "MENA":          ["p.Phe68Leu (c.202T>C)", "p.Glu102Gly (c.305A>G)", "p.Arg95Cys (c.283C>T)"],
        "South Asian":   ["p.Glu102Gly (c.305A>G)", "p.Phe68Leu (c.202T>C)", "p.Arg160Ter (c.478C>T)"],
        "North African": ["p.Glu102Gly (c.305A>G)", "p.Val43Asp (c.128T>A)", "p.Arg160Ter (c.478C>T)"],
        "East Asian":    ["p.Arg160Ter (c.478C>T)", "p.Val43Asp (c.128T>A)", "p.Glu102Gly (c.305A>G)"],
        "Other":         ["p.Arg160Ter (c.478C>T)", "p.Arg95Cys (c.283C>T)"],
    }

    allele_classes = [
        "Compound het missense+missense",        # 45% — two missense, residual IFT27 function
        "Compound het missense+truncating",       # 30% — mixed
        "Homozygous missense (consanguineous)",   # 18% — founder variants MENA/South Asian
        "Biallelic truncating",                   # 5%  — complete null, most severe
        "Tri-allelic",                            # 3%
    ]
    allele_weights = [0.45, 0.30, 0.18, 0.05, 0.03]  # corrected to sum 1.01 → minor rounding

    retinal_stages = [
        "Nyctalopia only (early)",
        "Bone-spicule RP pattern (moderate)",
        "Severe field loss (<20°)",
        "Legal blindness / end-stage",
    ]
    ret_w = [0.20, 0.32, 0.28, 0.20]

    polydactyly_types = [
        "Post-axial hands+feet",
        "Post-axial hands only",
        "Post-axial feet only",
        "Pre-axial (rare)",
        "None",
    ]
    # ~52% polydactyly (lower than BBS1 70% — partial IFT-B function preserved)
    poly_w = [0.28, 0.12, 0.08, 0.04, 0.48]

    # BBS15 renal: cystic pattern (like BBS1-12, NOT NPHP pattern unlike BBS13-14)
    renal_types = [
        "Corticomedullary cysts",
        "Calyceal clubbing",
        "Horseshoe kidney",
        "CKD stage 2-3",
        "None",
    ]
    renal_w = [0.15, 0.10, 0.05, 0.05, 0.65]   # ~35% renal

    dx_ages = ["0–4 yr", "5–11 yr", "12–17 yr", "18+ yr"]
    dx_w    = [0.15, 0.38, 0.30, 0.17]

    misdiag = [
        "Non-syndromic retinitis pigmentosa (no systemic features initially)",
        "Alström Syndrome (obesity + retinal; BBS15 has polydactyly — Alström lacks)",
        "BBS without molecular subtype (gene panel incomplete)",
        "Isolated nephronophthisis (renal anomaly first presentation)",
        "No initial misdiagnosis",
    ]
    mis_w = [0.28, 0.12, 0.22, 0.08, 0.30]

    presentation_ages = [
        "Birth (polydactyly noted)",
        "Infancy (nystagmus/photophobia)",
        "Early childhood (RP symptoms)",
        "School-age (obesity + RP)",
        "Adolescent/adult",
    ]
    pres_w = [0.18, 0.20, 0.30, 0.22, 0.10]

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

        # BBS15: ~52% polydactyly, ~72% obesity, ~38% cognitive LD,
        #        ~55% hypogonadism, ~45% anosmia, ~35% renal (cystic),
        #        ~3% CHD (lowest BBS spectrum)
        polydactyly  = poly_type != "None"
        obesity      = rng.random() < 0.72
        cognitive_ld = rng.random() < 0.38
        hypogonadism = rng.random() < 0.55
        anosmia      = rng.random() < 0.45
        renal_any    = renal_type != "None"
        chd          = rng.random() < 0.03
        ret_end      = ret_stage == "Legal blindness / end-stage"
        tri_bbs      = allele_cl == "Tri-allelic"
        mis_dx       = mis_d != "No initial misdiagnosis"
        consang      = eth in ("MENA", "South Asian", "North African") and rng.random() < 0.70
        # IFT25 destabilised in all BBS15 patients (IFT27 LOF → IFT25 monomeric)
        ift25_destab = True

        patients.append({
            "id":                   f"BBS15-{i+1:03d}",
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
            "ift25_destabilised":   ift25_destab,
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
        "gene":           "IFT27 / RABL4",
        "omim_gene":      "*615870",
        "omim_disease":   "#209900 (BBS15)",
        "locus":          "22q12.3",
        "protein":        "225 aa — small Rab-like GTPase (IFT-B1 subunit); GTPase domain (aa 1–180; P-loop, Switch I/II, Interswitch); C-terminal IFT-B recruitment tail (aa 180–225); obligate heterodimer with IFT25/HSPB11",
        "mechanism":      "IFT27 GTP-bound state recruits BBSome for retrograde exit from cilia via BBS3/ARL6 contact. IFT27 LOF → BBSome assembles normally, enters cilia normally (anterograde intact), but CANNOT EXIT retrograde — BBSome TRAPPED in cilia. GPCR cargo accumulates. OPPOSITE failure mode from most BBS genes (where BBSome fails to assemble). MKKS/BCC intact; MKS1 intact; IFT25/HSPB11 destabilised.",
        "bbs_frequency":  "<1% of all BBS (very rare); ~30-50 families worldwide 2026",
        "inheritance":    "Autosomal recessive; biallelic LOF",
        "triallelic_pct": 3,
        "allele_class_note": "Compound het missense+missense (45%); compound het missense+truncating (30%); homozygous missense consanguineous (18%); biallelic truncating (5%); tri-allelic (3%)",
        "if_fingerprint": "BBSome PRESENT at cilia base (assembled normally — unlike BBS2/BBS6); BBSome ACCUMULATES in cilia / elevated ciliary tip signal (pathognomonic — retrograde retrieval failure); IFT27 ABSENT from IFT-B; IFT25/HSPB11 DESTABILISED; MKKS/BCC NORMAL; MKS1 NORMAL",
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
        },
        "systemic_burden": [
            ("Rod-Cone Dystrophy (rod-first)",           n,           _pct(n)),
            ("Obesity (LepR trapped in cilia)",           obesity_n,   _pct(obesity_n)),
            ("Polydactyly (post-axial)",                  poly_n,      _pct(poly_n)),
            ("Hypogonadism",                              hypo_n,      _pct(hypo_n)),
            ("Anosmia/Hyposmia",                         anosmia_n,   _pct(anosmia_n)),
            ("Cognitive/LD",                              cognitive_n, _pct(cognitive_n)),
            ("Renal anomaly (cystic pattern)",            renal_any_n, _pct(renal_any_n)),
            ("CHD",                                       chd_n,       _pct(chd_n)),
        ],
        "retinal_stage_distribution": {k: v for k, v in ret_stages.items()},
        "dx_age_distribution":        {k: v for k, v in age_dist.items()},
    }


def get_breakdown() -> Dict[str, Any]:
    c = _COHORT
    n = _N

    systemic = [
        ("Rod-Cone Dystrophy (rod-first)",        n,                                                _pct(n)),
        ("Obesity (LepR trapped in cilia)",        sum(1 for p in c if p["obesity"]),               _pct(sum(1 for p in c if p["obesity"]))),
        ("Polydactyly (post-axial)",               sum(1 for p in c if p["polydactyly"]),           _pct(sum(1 for p in c if p["polydactyly"]))),
        ("Hypogonadism",                           sum(1 for p in c if p["hypogonadism"]),          _pct(sum(1 for p in c if p["hypogonadism"]))),
        ("Anosmia/Hyposmia",                      sum(1 for p in c if p["anosmia"]),               _pct(sum(1 for p in c if p["anosmia"]))),
        ("Cognitive/LD",                           sum(1 for p in c if p["cognitive_ld"]),          _pct(sum(1 for p in c if p["cognitive_ld"]))),
        ("Renal anomaly (any)",                    sum(1 for p in c if p["renal_any"]),             _pct(sum(1 for p in c if p["renal_any"]))),
        ("CHD",                                    sum(1 for p in c if p["chd"]),                   _pct(sum(1 for p in c if p["chd"]))),
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
            for r in ["Corticomedullary cysts", "Calyceal clubbing",
                      "Horseshoe kidney", "CKD stage 2-3", "None"]
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
                      "Alström Syndrome (obesity + retinal; BBS15 has polydactyly — Alström lacks)",
                      "BBS without molecular subtype (gene panel incomplete)",
                      "Isolated nephronophthisis (renal anomaly first presentation)",
                      "No initial misdiagnosis"]
        ],
        "top_variants": [{"variant": v, "n": cnt} for v, cnt in top_variants],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":             "IFT27 / RABL4 (RAB-like GTPase 4); Chr 22q12.3",
            "OMIM_Gene":        "*615870",
            "Chromosome":       "22q12.3",
            "Protein":          "225 aa — small Rab-like GTPase; N-terminal GTPase domain (aa 1–180): P-loop/Walker A (aa 12–19; GTP binding), Switch I (aa 56–75; GTP hydrolysis, BBSome recruitment), Switch II (aa 90–110; IFT25/HSPB11 dimer interface; BBS3/ARL6 contact), Interswitch (aa 78–90); C-terminal IFT-B recruitment tail (aa 180–225; IFT38/IFT172 contact, complex incorporation)",
            "Domains":          "P-loop (aa 12–19): GTP binding; Switch I (aa 56–75): GTP hydrolysis rate, BBSome recruitment relay via BBS3/ARL6 contact; Switch II (aa 90–110): obligate IFT25/HSPB11 dimer interface, BBS3/ARL6 interaction surface; Interswitch (aa 78–90): conformational relay; IFT-B tail (aa 180–225): IFT-B1 complex incorporation via IFT38 and IFT172",
            "Module":           "IFT-B1 sub-complex — IFT27 is the ONLY IFT-B subunit classified as a BBS gene. Forms obligate heterodimer with IFT25/HSPB11. GTP-bound IFT27 recruits BBSome for retrograde exit from cilia via BBS3/ARL6 contact.",
            "Inheritance":      "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":    "<1% of all BBS; very rare; ~30-50 families worldwide 2026",
            "Allele_spectrum":  "Compound het missense+missense (45%); compound het missense+truncating (30%); homozygous missense consanguineous (18%); biallelic truncating (5%); tri-allelic (3%)",
            "Triallelic_BBS":   "~3%",
            "IFT25_note":       "IFT25/HSPB11 is destabilised (monomeric, proteasome-cleared) when IFT27 is LOF — IFT25 reduced/absent on IF is part of BBS15 fingerprint",
            "CHD_penetrance":   "~3% (lowest across BBS spectrum)",
            "Population_note":  "European, MENA, South Asian; consanguineous in MENA/South Asian families",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 15 (BBS15) — #209900",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS15 subtype <1% (~1:10,000,000–30,000,000) — very rare",
            "Cardinal_features": "Rod-cone dystrophy (rod-first, scotopic extinguished) · Post-axial polydactyly (~52%) · Obesity (~72%; LepR trapped in cilia) · Hypogonadism (~55%) · Cognitive/LD (~38%) · Renal anomalies (~35%; cystic pattern) · Anosmia (~45%)",
            "BBS15_unique":      "ONLY BBS gene that is an IFT-B subunit. BBSome TRAPPED in cilia (retrograde retrieval failure) — OPPOSITE to most BBS genes where BBSome fails to assemble. BBSome tip signal ELEVATED on IF (pathognomonic). IFT25/HSPB11 destabilised. MKKS/BCC intact. MKS1 intact. NO liver fibrosis. NO Joubert overlap.",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; same ERG pattern as BBS1-14; rod-first scotopic extinguished; 100% penetrance)",
            "Obesity_mechanism": "LepR trapped in cilia (retrograde BBSome retrieval failure → LepR accumulates in ciliary compartment → cannot signal at plasma membrane → leptin resistance → obesity); mechanism distinct from BBS1-12 (BBSome LOF/assembly) and BBS13-14 (TZ gate); same clinical endpoint",
            "Renal":             "Cystic pattern (corticomedullary cysts, calyceal clubbing) — same as BBS1-12 cystic pattern; DISTINCT from BBS13-14 NPHP pattern (tubular atrophy/fibrosis); ~35% affected; no ESRD pattern documented",
            "Liver":             "No liver fibrosis — same as BBS14; no hepatic ductal plate involvement; standard monitoring only",
            "No_JBTS_overlap":   "No Joubert syndrome overlap reported — distinguishes BBS15 from BBS13 (JBTS28 8%) and BBS14 (JBTS5 10%)",
            "CHD":               "~3% (lowest across entire BBS spectrum)",
            "Diagnosis":         "Full BBS gene panel (minimum 20-24 genes including IFT27); IF: BBSome PRESENT at cilia base AND ELEVATED at ciliary tip (retrograde trap — anti-phase to BBS2/BBS6 absence); IFT27 ABSENT; IFT25/HSPB11 DESTABILISED; MKKS NORMAL; MKS1 NORMAL",
            "Treatment":         "Symptomatic; GLP1RA for obesity (LepR retrograde retrieval failure mechanism); renal surveillance (cystic — renal USS, GFR); no liver monitoring required; ophthalmology surveillance",
            "Prognosis":         "Progressive visual loss; renal cystic involvement ~35% (milder than NPHP pattern); near-normal survival with surveillance; no specific targeted therapy 2026",
        },
        "key_variants": [
            {
                "variant":     "p.Val43Asp (c.128T>A)",
                "domain":      "P-loop GTP-binding motif (aa 12–19 region; Val43 stabilises the P-loop conformation for GTP gamma-phosphate coordination; Asp substitution introduces steric clash and partial charge disruption in the GTP-binding pocket)",
                "consequence": "Hypomorphic allele; partial GTP binding retained; IFT-B1 complex partially destabilised; BBSome retrograde recruitment rate reduced (partial GTP-bound state); BBS15 phenotype; European enrichment; BBS15-specific (not typically reported in other GTPase ciliopathies)",
                "ethnicity":   "European enrichment (British, Dutch, French families)"
            },
            {
                "variant":     "p.Arg95Cys (c.283C>T)",
                "domain":      "Switch II region (aa 90–110; Arg95 is the key electrostatic anchor of the IFT25/HSPB11 dimer interface; loss of Arg guanidinium group disrupts the salt bridge network holding the IFT27-IFT25 obligate heterodimer)",
                "consequence": "Hypomorphic allele; IFT27-IFT25 heterodimer severely disrupted; IFT25/HSPB11 monomeric and proteasome-cleared; IFT27 itself residually present but non-functional at dimer interface; BBSome retrograde recruitment severely impaired; European; BBS-only phenotype (no neuromuscular phenotype)",
                "ethnicity":   "European enrichment; also reported in MENA families"
            },
            {
                "variant":     "p.Phe68Leu (c.202T>C)",
                "domain":      "Switch I region (aa 56–75; Phe68 is a conserved hydrophobic residue in Switch I; Leu substitution alters the Switch I conformation, reducing the GTP hydrolysis rate and altering the timing of BBSome retrograde recruitment relay through BBS3/ARL6 contact)",
                "consequence": "Hypomorphic allele; GTP hydrolysis rate altered → IFT27 remains in GTP-bound state for extended period paradoxically reducing BBSome retrieval synchrony; BBSome accumulates more severely in cilia; MENA enrichment; often compound het with Arg95Cys in European families; MENA families compound het with Glu102Gly",
                "ethnicity":   "MENA enrichment (Saudi Arabia, Jordan, Iran); compound het partner for Arg95Cys in European families"
            },
            {
                "variant":     "p.Glu102Gly (c.305A>G)",
                "domain":      "Switch II region (aa 90–110; Glu102 is part of the BBS3/ARL6 interaction surface on IFT27 Switch II; Gly substitution eliminates the acidic side chain contact with BBS3/ARL6 ARF GTPase domain)",
                "consequence": "Hypomorphic allele; IFT27-BBS3/ARL6 interaction surface disrupted → retrograde BBSome recruitment signal cannot be transduced from IFT27 to BBS3/ARL6 → BBSome accumulates in cilia; IFT25 dimer interface partially preserved (unlike Arg95Cys); MENA and South Asian enrichment; moderate phenotype",
                "ethnicity":   "MENA enrichment (Iran, Turkey, Jordan); South Asian (India, Pakistan)"
            },
            {
                "variant":     "p.Arg160Ter (c.478C>T)",
                "domain":      "C-terminal IFT-B recruitment tail (aa 180–225; truncation at aa 160 removes the entire IFT-B tail domain required for IFT-B1 complex incorporation — IFT38 and IFT172 contacts lost; GTPase domain truncated within Interswitch-Switch II boundary)",
                "consequence": "Null allele; complete loss of IFT-B complex incorporation; IFT27 protein absent from IFT-B1; IFT25/HSPB11 fully destabilised; BBSome retrograde retrieval completely absent; most severe BBS15 allele; pan-ethnic; biallelic truncating → most severe phenotype (complete BBSome cilia trapping)",
                "ethnicity":   "Pan-ethnic; most common truncating BBS15 allele"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG rod-first, scotopic extinguished) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia)",
            "2. Full BBS gene panel (minimum 20–24 genes including IFT27/RABL4 at Chr 22q12.3); note that IFT27 is small (225 aa) — exon coverage must include all coding exons; NGS panel or WES/WGS",
            "3. Allele classification: classify both IFT27 alleles as missense (hypomorphic) vs truncating (null); biallelic truncating → most severe BBS15; compound het missense+missense → mildest (45% of cohort); identify allele class for prognosis and counselling",
            "4. IF panel (pathognomonic BBS15 pattern): BBSome subunits (BBS2, BBS4, BBS8, BBS9) PRESENT at cilia base AND ELEVATED at ciliary tip (retrograde trap — this is ANTI-PHASE to BBS2/BBS6 where BBSome is absent); IFT27 ABSENT from IFT-B; IFT25/HSPB11 DESTABILISED (reduced on IF, confirms IFT27 LOF); MKKS/BCC NORMAL; MKS1 NORMAL",
            "5. No MRI brain required (no Joubert overlap expected in BBS15 — distinguishes from BBS13/BBS14 where JBTS overlap 8-10%); obtain MRI only if unexplained cerebellar findings",
            "6. Renal: cystic pattern (corticomedullary cysts, calyceal clubbing) — renal ultrasound at diagnosis; annual GFR; blood pressure monitoring; avoid nephrotoxins; NOT NPHP pattern (no tubular atrophy/fibrosis expected — distinguishes BBS15 from BBS13-14)",
            "7. Ophthalmology: ERG (rod-first, scotopic extinguished) + Goldmann VF + OCT retinal thickness annually; low-vision aids when VF <20°; no CEP290/IVS26-specific therapy applicable (unlike BBS14)",
            "8. Liver: standard surveillance only — no liver fibrosis expected in BBS15 (unlike BBS13/MKS1); routine LFTs at baseline only unless other indication",
            "9. Obesity: GLP-1 receptor agonist (semaglutide) applicable; mechanism is LepR trapped in cilia (retrograde BBSome retrieval failure → LepR accumulates in cilia); same clinical strategy as BBS1-14 obesity management",
            "10. Genetic counselling: AR inheritance; sibling recurrence 25%; consanguinity counselling for MENA/South Asian families; IFT27 is an IFT-B subunit — novel mechanism context for families; no BBS15-specific targeted therapy in clinical trials 2026",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; avoid photosensitising drugs; annual ERG + OCT surveillance; no BBS15-specific gene therapy in clinical development 2026 (IFT27 is small — AAV delivery feasible in principle but not yet in trials); refer to retinal dystrophy centre",
            "2. Obesity: GLP-1 receptor agonists (semaglutide/liraglutide) — LepR trapped in cilia via retrograde BBSome retrieval failure; same endpoint as BBS1-14 (leptin resistance); same clinical management pathway; bariatric surgery in refractory cases after ophthalmology risk stratification",
            "3. Renal: annual renal ultrasound (cystic pattern — corticomedullary cysts; calyceal clubbing); annual GFR and blood pressure; ACE-i/ARB if proteinuria/hypertension; strict nephrotoxin avoidance (NSAIDs, IV contrast); renal transplant if ESRD (rare in BBS15)",
            "4. Liver: no specific liver monitoring required (no ductal plate malformation/fibrosis in BBS15); standard LFTs at routine health check only",
            "5. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling (usually infertile in BBS)",
            "6. Anosmia: safety counselling (gas leaks, food spoilage, smoke detection); smoke detectors mandatory; dietary and olfactory rehabilitation support",
            "7. Cognitive/LD: early neuropsychological assessment; individualised education plan (IEP); multidisciplinary developmental support; note no cerebellar/JBTS overlap expected — cerebellar ataxia not a BBS15 feature",
            "8. Emerging research (2026): IFT27 retrograde mechanism is a novel therapeutic target — small molecule GTPase activators (stabilising GTP-bound IFT27) are at preclinical stage; patients should be registered with BBS registries for future trial eligibility; IFT27 AAV gene augmentation feasible (small gene, 225 aa) but not yet in clinical trials",
        ],
    }
