"""
BBS11 Bardet-Biedl Syndrome Type 11 — TRIM32 (E3 Ubiquitin Ligase) Dashboard
==============================================================================
Primary Gene : TRIM32 (*602290) — Tripartite Motif Containing 32; 653 aa; Chr 9q33.1.
               TRIM32 is the SOLE E3 UBIQUITIN LIGASE among all known BBS-causative
               proteins. Domain organisation (653 aa):
                 RING finger domain (aa 1–80): catalytic E3 ubiquitin ligase activity;
                   Pro130 is in the RING/B-box linker immediately downstream of the RING;
                 B-box zinc-finger (aa 81–140): zinc-binding; assists RING-substrate
                   recognition; Pro130Ser (LGMD2H allele) disrupts RING/B-box linker;
                 Coiled-coil domain (aa 141–280): mediates TRIM32 self-dimerisation;
                   Arg394 at coiled-coil/NHL junction is a South Asian enriched BBS allele;
                 NHL repeat 1 (aa 281–330); NHL repeat 2 (aa 331–380);
                 NHL repeat 3 (aa 381–430); NHL repeat 4 (aa 431–480);
                 NHL repeat 5 (aa 481–530): Ala502Val in NHL-5 is a BBS-specific allele;
                   Asp487Asn in NHL-5 is a MENA allele with dual BBS/LGMD2H phenotype;
                 NHL repeat 6 (aa 531–580); C-terminus (aa 581–653).
               TRIM32 primary substrates relevant to cilia:
                 HDAC6 (histone deacetylase 6): ciliary tubulin deacetylase; TRIM32
                   ubiquitinates HDAC6 → HDAC6 degradation → ciliary α-tubulin
                   HYPERACETYLATED (paradox: loss of TRIM32 → HDAC6 stabilised →
                   ciliary tubulin hypoacetylated → impaired IFT); the exact directionality
                   of HDAC6-dependent tubulin acetylation in BBS11 is still under
                   investigation, but ciliary function is clearly impaired.
                 Dishevelled (Dvl1, Dvl2): Wnt pathway transducer;
                 PIMT (protein L-isoaspartate O-methyltransferase): protein repair;
                 Actin: cytoskeletal remodelling;
                 c-Myc: proliferative control.
               BBSome is FULLY INTACT in BBS11 (TRIM32) LOF — the only BBS gene
               where all BBSome structural subunits are present by immunofluorescence:
                 Anti-BBS2 IF: NORMAL; anti-MKKS IF: NORMAL; anti-BBS10 IF: NORMAL;
                 Anti-BBS12 IF: NORMAL; anti-BBS1 IF: NORMAL; anti-BBS4 IF: NORMAL.
               Only anti-TRIM32 IF is ABSENT. BBSome assembly proceeds normally;
               ciliary homeostasis is disrupted downstream via HDAC6/ubiquitin pathway.
               Allelic heterogeneity (BBS vs LGMD2H):
                 NHL repeat mutations (repeats 4–6): predominantly BBS phenotype;
                 RING / B-box linker mutations (Pro130Ser, Asp487Asn partial): dual
                   BBS + LGMD2H phenotype; proximal limb-girdle weakness + elevated CK;
                 BBS11 is the only BBS gene that is allele-specifically also a
                   muscular dystrophy gene (LGMD2H = LGMD R11 TRIM32-related).
Disease        : Bardet-Biedl Syndrome Type 11 (#209900) — BBS11 subtype;
                 also LGMD2H / LGMD R11 TRIM32-related (#254110) — allele-specific
Frequency      : <2% of all BBS (rare); often under-recognised due to intact BBSome on IF
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~4% (MKKS/BBS6 most common third allele; no BCC subunit relationship
                 since TRIM32 is not a BCC member — tri-allelic BBS11 is a coincidental
                 combination, not a direct molecular interaction)
Cohort         : 40 patients; educational/synthetic; seed 355
Endpoints      : /api/bbs11/overview | /api/bbs11/breakdown | /api/bbs11/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 355
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("European",      0.45),   # Northern/Central European; Hutterite/Canadian for LGMD2H alleles
        ("MENA",          0.22),   # Middle East North Africa — Asp487Asn dual BBS/LGMD2H allele
        ("South Asian",   0.18),   # Arg394Trp; broadly South Asian
        ("North African", 0.10),   # Tunisia, Morocco — some overlap with MENA BBS cohorts
        ("East Asian",    0.03),
        ("Other",         0.02),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    variants_by_eth = {
        "European":      ["p.Pro130Ser (c.388C>T)", "p.Ala502Val (c.1505C>T)", "p.Leu619Ter (c.1856_1857del)"],
        "MENA":          ["p.Asp487Asn (c.1459G>A)", "p.Leu619Ter (c.1856_1857del)", "p.Pro130Ser (c.388C>T)"],
        "South Asian":   ["p.Arg394Trp (c.1180C>T)", "p.Leu619Ter (c.1856_1857del)", "p.Ala502Val (c.1505C>T)"],
        "North African": ["p.Asp487Asn (c.1459G>A)", "p.Arg394Trp (c.1180C>T)", "p.Leu619Ter (c.1856_1857del)"],
        "East Asian":    ["p.Leu619Ter (c.1856_1857del)", "p.Ala502Val (c.1505C>T)"],
        "Other":         ["p.Leu619Ter (c.1856_1857del)", "p.Pro130Ser (c.388C>T)"],
    }

    allele_classes = ["Missense/Missense", "Missense/Truncating", "Truncating/Truncating",
                      "LGMD2H-allele/BBS-allele", "Missense/LGMD2H-allele"]
    allele_weights = [0.30, 0.32, 0.18, 0.12, 0.08]

    retinal_stages = ["Nyctalopia only (early)", "Bone-spicule RP pattern (moderate)",
                      "Severe field loss (<20°)", "Legal blindness / end-stage"]
    ret_w = [0.20, 0.35, 0.28, 0.17]   # slightly earlier distribution vs BBS10/12

    polydactyly_types = ["Post-axial hands+feet", "Post-axial hands only",
                         "Post-axial feet only", "Pre-axial (rare)", "None"]
    poly_w = [0.32, 0.11, 0.08, 0.04, 0.45]   # ~55% polydactyly (lower than BBS10/12)

    renal_types = ["Structural anomaly (horseshoe/duplex)", "Cystic dysplasia",
                   "Tubulointerstitial nephropathy", "ESRD", "None"]
    renal_w = [0.10, 0.12, 0.06, 0.04, 0.68]  # ~32% renal (lower than BBS10/12)

    dx_ages = ["Birth (polydactyly noted)", "Infancy (nystagmus/strabismus)",
               "Early childhood (RP symptoms)", "School-age (obesity + RP)", "Adolescent/adult"]
    dx_w    = [0.18, 0.28, 0.26, 0.20, 0.08]

    misdiag = ["Isolated RP / Leber amaurosis", "LGMD (myopathy-first presentation)",
               "Obesity syndrome (Prader-Willi screen)", "Alström Syndrome", "No initial misdiagnosis"]
    mis_w   = [0.22, 0.15, 0.18, 0.07, 0.38]

    myopathy_features = ["None (pure BBS alleles)", "Mild proximal weakness (CK elevated 2–5×)",
                         "Moderate LGMD2H (limb-girdle pattern)", "Severe LGMD2H (wheelchair-dependent)"]
    myo_w = [0.60, 0.22, 0.14, 0.04]

    patients: List[Dict[str, Any]] = []
    for i in range(n):
        eth = rng.choices(eth_labels, weights=eth_weights)[0]
        vs  = variants_by_eth.get(eth, ["p.Leu619Ter (c.1856_1857del)"])
        v1  = rng.choice(vs)
        v2  = rng.choice(vs)
        ac  = rng.choices(allele_classes, weights=allele_weights)[0]
        ret = rng.choices(retinal_stages, weights=ret_w)[0]
        poly = rng.choices(polydactyly_types, weights=poly_w)[0]
        renal = rng.choices(renal_types, weights=renal_w)[0]
        dx_age = rng.choices(dx_ages, weights=dx_w)[0]
        misdiagnosis = rng.choices(misdiag, weights=mis_w)[0]
        myopathy = rng.choices(myopathy_features, weights=myo_w)[0]
        tri_allelic = rng.random() < 0.04

        p: Dict[str, Any] = {
            "id":               f"BBS11-{i+1:03d}",
            "gene":             "TRIM32",
            "ethnicity":        eth,
            "variant1":         v1,
            "variant2":         v2,
            "allele_class":     ac,
            "retinal_stage":    ret,
            "polydactyly_type": poly,
            "renal_type":       renal,
            "presentation_age": dx_age,
            "misdiagnosis":     misdiagnosis,
            "myopathy_feature": myopathy,
            "tri_allelic":      tri_allelic,
            # systemic features
            "obesity":          rng.random() < 0.72,
            "hypogonadism":     rng.random() < 0.52,
            "cognitive_ld":     rng.random() < 0.38,
            "anosmia":          rng.random() < 0.48,
            "chd":              rng.random() < 0.03,
        }
        patients.append(p)

    return patients


_COHORT: List[Dict[str, Any]] | None = None

def _cohort() -> List[Dict[str, Any]]:
    global _COHORT
    if _COHORT is None:
        _COHORT = _make_cohort()
    return _COHORT


# ── API response builders ─────────────────────────────────────────────────────
def get_overview() -> Dict[str, Any]:
    c = _cohort()
    n = len(c)

    # systemic burden
    systemic_flags = {
        "Rod-cone dystrophy (rod-first)":    sum(1 for p in c if True),            # 100%
        "Post-axial polydactyly":            sum(1 for p in c if p["polydactyly_type"] != "None"),
        "Obesity (LepR mis-traffic)":        sum(1 for p in c if p["obesity"]),
        "Hypogonadism":                      sum(1 for p in c if p["hypogonadism"]),
        "Renal anomaly":                     sum(1 for p in c if p["renal_type"] != "None"),
        "Cognitive / LD":                    sum(1 for p in c if p["cognitive_ld"]),
        "Anosmia":                           sum(1 for p in c if p["anosmia"]),
        "Congenital heart defect (CHD)":     sum(1 for p in c if p["chd"]),
        "Myopathic features (LGMD2H allele)": sum(1 for p in c if p["myopathy_feature"] != "None (pure BBS alleles)"),
    }
    systemic = [(feat, nn, round(nn / n * 100, 1)) for feat, nn in systemic_flags.items()]

    eth_counts: Dict[str, int] = {}
    for p in c:
        eth_counts[p["ethnicity"]] = eth_counts.get(p["ethnicity"], 0) + 1

    tri = sum(1 for p in c if p["tri_allelic"])
    myopathy = sum(1 for p in c if p["myopathy_feature"] != "None (pure BBS alleles)")

    return {
        "gene":             "TRIM32",
        "disease":          "Bardet-Biedl Syndrome Type 11 (BBS11)",
        "omim_gene":        "*602290",
        "omim_disease":     "#209900",
        "chromosome":       "9q33.1",
        "protein_size":     "653 aa",
        "domains":          "RING finger (aa 1–80) · B-box zinc-finger (aa 81–140) · Coiled-coil (aa 141–280) · 6× NHL repeats (aa 281–653)",
        "function_class":   "E3 ubiquitin ligase — ubiquitinates HDAC6, Dishevelled, PIMT, actin, c-Myc",
        "bbs_role":         "HDAC6 ubiquitination → ciliary tubulin acetylation homeostasis; downstream of BBSome; BBSome INTACT in BBS11 LOF",
        "if_fingerprint":   "ALL BBSome subunits NORMAL (BBS1/2/4/5/7/8/9 IF all normal; MKKS/BBS10/BBS12 IF all normal); only TRIM32 IF ABSENT — unique among all BBS genes",
        "bbsome_integrity": "INTACT — BBS11 is the only BBS gene where BBSome assembly is unaffected",
        "frequency_bbs":    "<2% of all BBS",
        "inheritance":      "Autosomal recessive (AR); biallelic LOF",
        "allelic_note":     "NHL repeat mutations → BBS phenotype; RING/B-box linker mutations (p.Pro130Ser, p.Asp487Asn) → dual BBS + LGMD2H (limb-girdle muscular dystrophy)",
        "tri_allelic_pct":  "~4%",
        "prevalence":       "~1:100,000–160,000 worldwide (BBS total); BBS11 subtype <2%",
        "cohort_n":         n,
        "systemic_burden":  [{"feature": f, "n": nn, "pct": pct} for f, nn, pct in systemic],
        "ethnicity_distribution": [{"ethnicity": e, "n": cnt} for e, cnt in sorted(eth_counts.items(), key=lambda x: -x[1])],
        "tri_allelic_n":    tri,
        "myopathy_n":       myopathy,
        "myopathy_pct":     round(myopathy / n * 100, 1),
        "top_variants": [
            {"variant": "p.Pro130Ser (c.388C>T)", "domain": "RING/B-box linker", "ethnicity": "European (Hutterite founder)", "allelic_note": "dual BBS + LGMD2H"},
            {"variant": "p.Ala502Val (c.1505C>T)", "domain": "NHL repeat 5",    "ethnicity": "European (BBS-specific)"},
            {"variant": "p.Asp487Asn (c.1459G>A)", "domain": "NHL repeat 5",    "ethnicity": "MENA", "allelic_note": "dual BBS + LGMD2H"},
            {"variant": "p.Arg394Trp (c.1180C>T)", "domain": "Coiled-coil/NHL junction", "ethnicity": "South Asian (BBS-specific)"},
            {"variant": "p.Leu619Ter (c.1856_1857del)", "domain": "C-terminus truncating null", "ethnicity": "Pan-ethnic"},
        ],
    }


def get_breakdown() -> Dict[str, Any]:
    c = _cohort()
    n = len(c)

    systemic_flags = {
        "Rod-cone dystrophy":   sum(1 for p in c if True),
        "Post-axial polydactyly": sum(1 for p in c if p["polydactyly_type"] != "None"),
        "Obesity":              sum(1 for p in c if p["obesity"]),
        "Hypogonadism":         sum(1 for p in c if p["hypogonadism"]),
        "Renal anomaly":        sum(1 for p in c if p["renal_type"] != "None"),
        "Cognitive / LD":       sum(1 for p in c if p["cognitive_ld"]),
        "Anosmia":              sum(1 for p in c if p["anosmia"]),
        "CHD":                  sum(1 for p in c if p["chd"]),
        "Myopathic features":   sum(1 for p in c if p["myopathy_feature"] != "None (pure BBS alleles)"),
    }
    systemic = [(feat, nn, round(nn / n * 100, 1)) for feat, nn in systemic_flags.items()]

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

    myo_counts: Dict[str, int] = {}
    for p in c:
        myo_counts[p["myopathy_feature"]] = myo_counts.get(p["myopathy_feature"], 0) + 1

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
            for m in ["Isolated RP / Leber amaurosis", "LGMD (myopathy-first presentation)",
                      "Obesity syndrome (Prader-Willi screen)", "Alström Syndrome", "No initial misdiagnosis"]
        ],
        "myopathy_distribution": [
            {"label": m, "n": myo_counts.get(m, 0)}
            for m in ["None (pure BBS alleles)", "Mild proximal weakness (CK elevated 2–5×)",
                      "Moderate LGMD2H (limb-girdle pattern)", "Severe LGMD2H (wheelchair-dependent)"]
        ],
        "top_variants": [{"variant": v, "n": cnt} for v, cnt in top_variants],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":             "TRIM32 (Tripartite Motif Containing 32); Chr 9q33.1",
            "OMIM_Gene":        "*602290",
            "Chromosome":       "9q33.1",
            "Protein":          "653 aa — RING finger E3 ubiquitin ligase (aa 1–80) · B-box zinc-finger (aa 81–140) · Coiled-coil self-dimerisation (aa 141–280) · 6 NHL repeats (aa 281–653); BBS vs LGMD2H allele location: RING/B-box linker mutations → dual BBS+LGMD2H; NHL repeat 4–6 mutations → predominantly BBS",
            "Domains":          "aa 1–80: RING finger (catalytic E3 ligase; Cys3-His-Cys3 zinc coordination) · aa 81–140: B-box zinc-finger (substrate recognition assistance) · aa 141–280: coiled-coil (TRIM32 homodimerisation) · aa 281–330: NHL-1 · aa 331–380: NHL-2 · aa 381–430: NHL-3 · aa 431–480: NHL-4 (Asp487 at NHL-4/5 junction) · aa 481–530: NHL-5 (Ala502) · aa 531–580: NHL-6 · aa 581–653: C-terminus",
            "E3_substrates":    "HDAC6 (ciliary tubulin deacetylase → ciliary acetylation homeostasis) · Dishevelled/Dvl1/2 (Wnt pathway) · PIMT (protein repair) · Actin · c-Myc",
            "BBSome_integrity": "INTACT — BBSome assembles normally in BBS11 LOF; the only BBS gene where all 8 BBSome subunits are present by IF; ciliary dysfunction is downstream via HDAC6/ubiquitin pathway, not BBSome absence",
            "Inheritance":      "Autosomal Recessive; biallelic LOF",
            "Frequency_BBS":    "<2% of all BBS; often under-recognised because intact BBSome on IF is not consistent with typical BBS IF patterns",
            "Allele_spectrum":  "NHL repeat mutations (repeats 4–6) → predominantly BBS; RING/B-box linker (Pro130Ser) → dual BBS + LGMD2H; Asp487Asn (NHL-4/5 junction) → dual BBS + LGMD2H in MENA populations",
            "Triallelic_BBS":   "~4% (MKKS/BBS6 most common third allele; no direct BCC/BBSome structural relationship — coincidental)",
            "CHD_penetrance":   "~3% (lowest among all BBS genes; no cardiac-specific role for TRIM32 identified)",
            "LGMD2H_note":      "TRIM32 is also LGMD R11/LGMD2H (*254110); allele-specific overlap: RING/B-box linker mutations cause LGMD2H with or without BBS features; NHL mutations cause BBS without myopathy; always assess CK and proximal strength at BBS11 diagnosis",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 11 (BBS11) — #209900; also LGMD R11/LGMD2H (#254110) allele-specifically",
            "Prevalence":        "~1:100,000–160,000 worldwide (BBS total); BBS11 subtype <2%",
            "Cardinal_features": "Rod-cone dystrophy (rod-first) · Post-axial polydactyly · Obesity (LepR mis-traffic via intact-BBSome-but-impaired-ciliary-homeostasis) · Hypogonadism · Renal anomalies · Cognitive/LD · Anosmia · Myopathic features (LGMD2H alleles only)",
            "BBS11_unique":      "TRIM32 is an E3 ubiquitin ligase — the only non-structural BBS protein; BBSome is FULLY INTACT on IF (all subunits normal); only TRIM32 absent; clinically BBS without BBSome disruption; allele-specific LGMD2H overlap",
            "IF_fingerprint":    "ALL BBSome subunits NORMAL by IF (BBS1, BBS2, BBS4, BBS5, BBS7, BBS8, BBS9, MKKS, BBS10, BBS12 all present); only TRIM32 IF absent — pathognomonic of BBS11; intact BBSome does NOT exclude BBS11",
            "Retinal":           "Rod-first scotopic loss → bone-spicule RP → legal blindness (progressive; identical timeline to BBS1–10, BBS12 but slightly milder in some series)",
            "Obesity_mechanism": "Impaired ciliary HDAC6 regulation → ciliary dysfunction → LepR mis-trafficking secondary to cilia impairment (not direct BBSome absence); GLP1RA applicable",
            "Renal":             "Structural anomalies (horseshoe, duplex) + cystic dysplasia; lower ESRD rate (~4%) vs BBS10/12",
            "CHD":               "~3% (lowest in BBS spectrum); echocardiogram indicated at diagnosis",
            "LGMD_overlap":      "RING/B-box alleles (Pro130Ser, Asp487Asn): proximal limb-girdle weakness + elevated CK (2–5×) + EMG myopathic pattern + muscle biopsy showing dystrophic changes; LGMD2H may present decades before BBS features are recognised",
            "Diagnosis":         "Full BBS gene panel (minimum 20–24 genes including TRIM32); LGMD panel (if myopathic features); CK measurement mandatory; IF: intact BBSome + absent TRIM32 is diagnostic",
            "Treatment":         "Symptomatic; GLP1RA for obesity; physiotherapy for LGMD2H myopathy; no TRIM32-specific gene therapy 2026",
            "Prognosis":         "Progressive visual loss; LGMD2H alleles: variable myopathic progression (slowly progressive); survival near-normal with management",
        },
        "key_variants": [
            {
                "variant":     "p.Pro130Ser (c.388C>T)",
                "domain":      "RING/B-box linker (Pro130 is immediately downstream of RING domain, in the flexible linker connecting RING to B-box; the Pro→Ser substitution disrupts linker rigidity and RING/B-box geometric relationship)",
                "consequence": "RING finger geometry altered relative to B-box → E3 ligase activity reduced but not abolished → HDAC6 ubiquitination impaired → ciliary tubulin homeostasis disrupted → BBS phenotype; additionally disrupts B-box zinc coordination geometry → partial loss of substrate recognition → LGMD2H myopathic features in skeletal muscle (high TRIM32 dependency for muscle protein turnover)",
                "ethnicity":   "European (originally described as LGMD2H founder in Hutterite population, North America/Canada); also found in isolated European families as BBS allele at homozygosity or compound heterozygosity with NHL allele"
            },
            {
                "variant":     "p.Ala502Val (c.1505C>T)",
                "domain":      "NHL repeat 5 (Ala502 in the NHL-5 blade inner surface; Ala→Val bulky substitution disrupts NHL β-propeller blade packing)",
                "consequence": "NHL-5 blade geometry altered → TRIM32 substrate recognition impaired (NHL repeats form the substrate-binding β-propeller platform) → HDAC6 binding and ubiquitination reduced → BBS phenotype; no significant LGMD2H myopathy (NHL mutation — muscle-sparing allele class)",
                "ethnicity":   "European (Northern/Central Europe — predominantly BBS-specific allele without myopathic features)"
            },
            {
                "variant":     "p.Asp487Asn (c.1459G>A)",
                "domain":      "NHL repeat 4/5 junction (Asp487 at the C-terminal end of NHL-4; the Asp carboxyl forms a hydrogen bond with NHL-5 N-terminal strand — Asn disrupts this inter-blade contact)",
                "consequence": "NHL-4/5 inter-blade contact weakened → NHL β-propeller platform partially destabilised → substrate recognition impaired across multiple substrates (HDAC6 + muscle-specific substrates) → dual BBS phenotype + partial LGMD2H myopathic features; the dual phenotype is allele-class specific — Asp487Asn affects both ciliary (HDAC6) and myopathic (muscle protein) substrate pathways",
                "ethnicity":   "MENA (Middle East and North Africa — detected in Saudi, Iranian, Lebanese consanguineous families; LGMD2H allele also identified in MENA populations)"
            },
            {
                "variant":     "p.Arg394Trp (c.1180C>T)",
                "domain":      "Coiled-coil/NHL repeat junction (Arg394 at the C-terminal end of the coiled-coil, immediately preceding NHL-1; the Arg side chain bridges the coiled-coil α-helix to the first NHL blade β-strand)",
                "consequence": "Coiled-coil to NHL-1 transition disrupted → TRIM32 homodimerisation geometry altered + NHL β-propeller register disturbed → reduced substrate ubiquitination efficiency → BBS phenotype; no significant myopathy (junction mutation, not RING/B-box)",
                "ethnicity":   "South Asian (predominantly BBS-specific allele; detected in Indian and Pakistani consanguineous families)"
            },
            {
                "variant":     "p.Leu619Ter (c.1856_1857del)",
                "domain":      "Truncating null — frameshift at aa 619; removes C-terminal portion of NHL repeat 6; complete loss of NHL-6 C-terminal structure and downstream C-terminus; no functional TRIM32 protein from this allele",
                "consequence": "Complete loss of E3 ligase activity and substrate recognition → null allele → BBS phenotype at biallelic homozygosity or compound with missense allele; no myopathy (truncating null removes NHL residues, not RING/B-box which drives myopathy)",
                "ethnicity":   "Pan-ethnic (detected in European, South Asian, MENA, North African cohorts; most common truncating allele in TRIM32 BBS series)"
            },
        ],
        "diagnostic_workup": [
            "1. Confirm BBS clinical criteria: rod-cone dystrophy (ERG rod-first) + ≥2 cardinal features (polydactyly, obesity, hypogonadism, renal, cognitive/LD, anosmia); assess CK and proximal strength — myopathic features flag BBS11",
            "2. Full BBS gene panel (minimum 20–24 genes including TRIM32/BBS11); if myopathic features also present, LGMD gene panel (TRIM32 is included in most modern LGMD panels)",
            "3. BBS11-specific IF fingerprint: all BBSome subunits NORMAL by IF (anti-BBS2, anti-MKKS, anti-BBS10, anti-BBS12 all normal); anti-TRIM32 ABSENT — this is pathognomonic for BBS11",
            "4. CRITICAL: intact BBSome by IF does NOT exclude BBS — BBS11 (TRIM32) is the one BBS gene where all BBSome subunits are present; if routine BBS IF panel (BBS2, MKKS) is normal but clinical BBS, always test TRIM32 IF and send gene panel",
            "5. CK measurement: LGMD2H alleles (Pro130Ser, Asp487Asn) typically show CK 2–5× upper limit of normal; EMG if myopathic features; muscle MRI (thigh/calf) for LGMD2H: selective early involvement of rectus femoris and medial gastrocnemius",
            "6. For LGMD2H alleles: muscle biopsy shows dystrophic changes + TRIM32 absent by IHC; western blot confirms absent TRIM32 protein",
            "7. Allele classification matters clinically: NHL repeat homozygous/compound → BBS-only; RING/B-box linker allele (Pro130Ser, Asp487Asn) → test for dual BBS+LGMD2H; neurologist co-management for LGMD2H alleles",
            "8. Renal ultrasound + GFR (lower ESRD rate ~4% but surveillance mandatory); echocardiogram (CHD ~3%); ERG + Goldmann VF + OCT at diagnosis",
            "9. LGMD2H family members: targeted TRIM32 testing for unaffected siblings; LGMD2H is slowly progressive — early physiotherapy is beneficial",
            "10. Gene panel is essential — TRIM32 is missed on standard ciliopathy IF panels that test only BBSome subunits (BBS2, BBS4, BBS5, BBS7, BBS8) since all are normal in BBS11",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; photosensitising drug avoidance; vitamin A not indicated for BBS; retinal gene therapy investigational (no TRIM32 AAV approved 2026)",
            "2. Obesity: GLP-1 receptor agonists (semaglutide, liraglutide) — LepR pathway impaired via ciliary dysfunction (mechanistically different from BBSome-absent BBS but same downstream outcome); applicable to all BBS subtypes",
            "3. LGMD2H myopathy (RING/B-box alleles): physiotherapy (eccentric resistance training for slowly progressive LGMD); avoid statin myopathy; creatine monohydrate supplementation under investigation; respiratory monitoring for advanced cases",
            "4. Renal: ACE inhibitor/ARB for CKD progression; renal transplant for ESRD (curative); avoid nephrotoxins; annual cystatin C",
            "5. Hypogonadism: testosterone (males) or oestrogen-progesterone (females) HRT after pubertal assessment; fertility counselling",
            "6. Anosmia: safety counselling (gas leaks, food spoilage); smoke detectors mandatory; smell training investigational",
            "7. Cognitive/LD: early intervention; IEP; neuropsychological assessment; occupational therapy",
            "8. CHD (~3%): echocardiogram-guided cardiology; standard cardiac management; no BBS11-specific cardiac lesion profile",
            "9. Genetic counselling: AR inheritance; sibling recurrence 25%; prenatal/preimplantation genetic testing available; LGMD2H and BBS11 both AR — same gene panel can identify both risk types in a family",
            "10. Surveillance: annual ophthalmology + renal + metabolic (HbA1c, lipids) + CK (for LGMD2H allele carriers); growth/pubertal charting in children; neurological review if myopathic features",
        ],
        "ddx_table": [
            {"disease": "LGMD2H / LGMD R11 (TRIM32 — same gene)", "key_difference": "LGMD2H is caused by the same TRIM32 gene but predominantly RING/B-box linker alleles (Pro130Ser, Asp487Asn). LGMD2H presents with limb-girdle muscular dystrophy without (or with mild) BBS features. BBS11 presents with full BBS. The distinction is allele-specific: NHL repeat biallelic → BBS; RING/B-box allele biallelic → LGMD2H; RING/B-box compound NHL → dual BBS+LGMD2H. Always assess both phenotypes when TRIM32 variants are found."},
            {"disease": "BBS1 (BBS1 gene)", "key_difference": "BBS1 LOF → BBS1 absent by IF; BBS11 LOF → BBS1 NORMAL by IF (intact BBSome). BBS1 is a structural BBSome cargo adaptor; BBS11 (TRIM32) is an E3 ligase. IF cleanly separates them: any absent BBSome subunit means BBS1–10, BBS12; intact BBSome means BBS11."},
            {"disease": "BBS10 (BBS10 gene)", "key_difference": "BBS10 LOF → BBS10 ABSENT, MKKS REDUCED, BBS12 REDUCED, BBS2 ABSENT (BCC disruption). BBS11 LOF → ALL subunits NORMAL (BBSome intact). IF clearly separates them: any BCC disruption (MKKS/BBS10/BBS12 reduced) excludes BBS11; intact BCC on IF raises BBS11 as top suspect on IF alone."},
            {"disease": "Alström Syndrome (ALMS1)", "key_difference": "Cone-rod (cone first, not rod first); no polydactyly; infantile DCM (60%); sensorineural hearing loss; ALMS1 mutations; no TRIM32 involvement. Both have intact BBSome on IF (Alström is an ALMS1 centriolar protein, not BBSome). ALMS1 gene panel separates from TRIM32."},
            {"disease": "Joubert Syndrome (JBTS, CEP290/NPHP6 etc.)", "key_difference": "Molar tooth sign on brain MRI; cerebellar vermis hypoplasia; IFT/transition-zone ciliary modules not BBSome; TRIM32 mutations not causative of JBTS; no BBS11 → JBTS phenotypic overlap except anosmia."},
            {"disease": "Neuronal ceroid lipofuscinosis-myopathy (NCL-myopathy)", "key_difference": "TRIM32 has been suggested in rare NCL-type presentations but this is not established as a distinct disease class; distinct from BBS11 (no retinal degeneration of NCL type — ERG in BBS11 shows rod-cone dystrophy, not NCL ERG pattern). Standard BBS gene panel should include TRIM32."},
            {"disease": "Bardet-Biedl Syndrome — other genes (BBS1–10, BBS12–22)", "key_difference": "BBS11 is the only BBS gene with fully intact BBSome by IF. Any other BBS gene LOF produces at least one absent BBSome subunit (BBS1–10, BBS12) or BCC disruption (BBS6/MKKS, BBS10, BBS12). The intact BBSome IF is the single most useful pointer to BBS11 in a clinical setting."},
        ],
        "mechanism_glossary": [
            {
                "term": "TRIM32 as the Only E3 Ubiquitin Ligase in the BBS Spectrum",
                "definition": "All other BBS-causative proteins are either structural BBSome subunits (BBS1–BBS9), BCC chaperonin subunits (BBS6/MKKS, BBS10, BBS12), or ciliary trafficking modulators (ARL6/BBS3, BBIP10/BBS18). TRIM32 is the sole E3 ubiquitin ligase among BBS genes — it ubiquitinates substrates for proteasomal or lysosomal degradation. Its connection to BBS is through HDAC6 ubiquitination, which modulates ciliary α-tubulin acetylation and IFT efficiency. Loss of TRIM32 → HDAC6 stabilised → dysregulated tubulin deacetylase → impaired cilia without absent BBSome."
            },
            {
                "term": "Intact BBSome in BBS11 — Pathognomonic IF Fingerprint",
                "definition": "In all other BBS subtypes, loss of the causative gene results in at least one absent (or reduced) BBSome subunit by immunofluorescence. BBS11 (TRIM32) is uniquely different: since TRIM32 does not contribute to BBSome structure, BBSome assembles completely — BBS1, BBS2, BBS4, BBS5, BBS7, BBS8, BBS9 are all present by IF, as are the BCC subunits (MKKS, BBS10, BBS12). Only TRIM32 itself is absent by IF. This means standard BBSome IF panels that test BBS2 and MKKS (and other subunits) will all show NORMAL results in BBS11, which can lead to misidentification as non-BBS. TRIM32 IF must be explicitly requested."
            },
            {
                "term": "HDAC6 Ubiquitination — TRIM32's Ciliary Role",
                "definition": "HDAC6 (histone deacetylase 6) deacetylates α-tubulin at lysine 40, a key modification for IFT motor processivity within cilia. TRIM32 ubiquitinates HDAC6, targeting it for proteasomal degradation. In TRIM32 LOF, HDAC6 is stabilised → enhanced tubulin deacetylation → reduced ciliary α-tubulin acetylation → impaired IFT motor function → ciliary cargo mis-trafficking (including LepR). The downstream phenotype (obesity, retinal degeneration) emerges from ciliary dysfunction mediated by tubulin acetylation homeostasis, not BBSome absence."
            },
            {
                "term": "Allelic Heterogeneity: BBS11 vs LGMD2H (Same TRIM32 Gene)",
                "definition": "The TRIM32 protein has two functional regions with different tissue dependencies: the RING/B-box/coiled-coil region (N-terminal half) and the NHL repeat β-propeller (C-terminal half). Skeletal muscle depends heavily on RING/B-box-mediated ubiquitination of muscle-specific substrates (actin, myosin light chain) for protein quality control. Ciliary cells (photoreceptors, renal tubules) depend on NHL repeat-mediated HDAC6 ubiquitination. RING/B-box linker mutations (Pro130Ser) disrupt muscle-specific substrate recognition → LGMD2H myopathy ± BBS. NHL repeat mutations (Ala502Val) disrupt HDAC6/ciliary substrate recognition → BBS without myopathy. Asp487Asn (NHL-4/5 junction) disrupts both → dual BBS + LGMD2H."
            },
            {
                "term": "BBS11 and the Importance of TRIM32 in Tri-Allelic BBS",
                "definition": "Tri-allelic BBS11 (~4%) is coincidental rather than biochemically coupled: TRIM32 does not interact directly with any BBSome structural subunit in cis, unlike BCC subunit tri-allelic combinations (e.g. BBS10+BBS12 as third alleles because they are direct BCC equatorial partners). A TRIM32 third allele in a BBS patient with biallelic BBS1 (or other) mutations worsens ciliary dysfunction through additive impairment of both BBSome-absent (BBS1) and HDAC6/acetylation-impaired (TRIM32) pathways — a dual-mechanism ciliopathy."
            },
            {
                "term": "LepR Mis-Trafficking in BBS11 — Indirect Mechanism",
                "definition": "In BBSome-absent BBS (BBS1–10, BBS12), LepR is retained in hypothalamic neuronal cilia because the BBSome complex (which mediates retrograde IFT of activated GPCRs) is absent. In BBS11 (TRIM32), the BBSome is intact, but ciliary IFT motor function is impaired via HDAC6-mediated tubulin acetylation dysregulation — IFT motors are less processive on hypoacetylated ciliary axonemes. This reduces the efficiency of BBSome-mediated LepR retrograde transport even with intact BBSome, causing functional (not structural) LepR mis-trafficking and the same leptin resistance/obesity phenotype. GLP-1 receptor agonists bypass this upstream of cilia and are applicable."
            },
        ],
    }
