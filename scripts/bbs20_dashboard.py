"""
BBS20 Bardet-Biedl Syndrome Type 20 — WDPCP Dashboard
=======================================================
Primary Gene : WDPCP (WD Repeat Containing Planar Cell Polarity Effector).
               Also known as Fritz (Frizzled Interacting Protein).
               809 aa (~92 kDa) — BB/TZ scaffold bridging CPLANE complex
               (INTURNED + FUZZY) to BBSome for ciliary entry coordination;
               also a PCP effector linking Frizzled signalling to ciliogenesis.
               Domain organisation (809 aa):
                 N-terminal IDR + INTU/FUZ binding module (aa 1–90):
                   Intrinsically disordered region; BB-targeting signal;
                   INTURNED (INTU) and FUZZY (FUZ) CPLANE docking interface;
                   Arg17 and Trp42 are key INTU contact residues;
                 WD40 β-propeller (aa 91–500):
                   6–7 WD40 repeats forming a canonical β-propeller scaffold;
                   BBSome interaction surface — BBS1 β-propeller (blade 3–4
                   interface, aa 180–260) and BBS9 PTHB1 (blade 5–6, aa 300–410);
                   Pro267 (blade 4) — South Asian variant hotspot;
                   Arg350 (blade 5, BBS1 primary interface) — MENA hotspot;
                   Cys124 (blade 1 N-terminal edge, INTU secondary contact);
                 C-terminal coiled-coil (aa 501–809):
                   Fritz dimerisation domain (WDPCP homo-dimerises at BB);
                   BB/TZ membrane-anchoring region;
                   Glu542 (CC N-terminus) — European dimer-interface variant;
                   Gln700 truncation site — C-terminal null allele;

               MECHANISM — CPLANE↔BBSome Bridge at Basal Body/TZ:
               WDPCP localises to the subdistal appendages and transition zone
               of the basal body. It forms a scaffold:
                 CPLANE complex (INTU + FUZ + CPLANE1/CFAP126) docks to
                   WDPCP N-terminal IDR → WDPCP recruits INTU to the BB;
                 WDPCP WD40 β-propeller contacts BBS1 and BBS9 (BBSome core)
                   → acts as a docking platform for BBSome at the BB/TZ gate;
                 WDPCP C-terminal CC enables homo-dimerisation → creates a
                   stable scaffold for CPLANE↔BBSome co-localisation;
                 Correct BB docking geometry ensures BBSome can receive
                   IFT trains and enter the ciliary compartment.

               WDPCP LOF consequence:
                 INTU loses BB localisation (WDPCP no longer recruits INTU
                   from cytoplasm to BB) → CPLANE complex partially disrupted;
                 BBSome fails to dock properly at BB/TZ gate → BBSome reduced
                   in cilia (BBS2 IF: REDUCED; distinct from BBS17 where
                   BBSome assembles but cannot enter, and BBS18 where
                   assembly fails entirely);
                 PCP pathway partially defective → polydactyly enriched (~60%)
                   compared to average BBS (~45%) — limb bud BB misdocking
                   disrupts GLI3R/FL ratio similarly to CPLANE1/JBTS33;
                 Cilia: PRESENT, SHORTENED (~70–80% WT length), occasionally
                   MISORIENTED — PCP defect in BB docking geometry;
                 IFT trains: ANTEROGRADE and RETROGRADE generally INTACT
                   (IFT-A and IFT-B subunit IF normal) — distinct from BBS19;
                 ARL13B: REDUCED in cilia (BBSome-dependent ARL13B retrieval
                   impaired, consistent with all BBSome LOF types);
                 Single disease tier — no NPHP/MKS/JBTS allelic equivalent.

               IF fingerprint — distinguishes BBS20:
                 WDPCP: ABSENT from BB/subdistal appendages/TZ.
                 INTURNED (INTU): ABSENT/greatly reduced at BB
                   (WDPCP-dependent recruitment lost) — diagnostic for BBS20.
                 BBSome (anti-BBS2): REDUCED in cilia — TZ docking failure.
                 MKS1 (TZ): NORMAL (TZ structural integrity intact).
                 IFT88 (IFT-B anterograde): NORMAL axonemal distribution
                   (no tip accumulation; distinct from BBS19).
                 ARL13B: REDUCED in cilia (BBSome cargo retrieval impaired).
                 Acetylated α-tubulin: cilia PRESENT, SHORTENED,
                   occasionally misoriented — PCP BB-docking defect.
                 MKKS/BCC chaperonin: NORMAL (upstream assembly intact).

               Allele-class notes:
                 All biallelic LOF combinations → BBS20 (single tier);
                 No null/null → MKS, NPHP, or JBTS allelic disease reported
                   (unlike CEP290/BBS14, IFT172/BBS19) — WDPCP is not a
                   TZ structural component or IFT turnaround gene;
                 Tri-allelic BBS: ~2% of BBS20 families.

               Distinction from related BBS genes:
                 BBS20 (WDPCP): CPLANE↔BBSome bridge at BB → INTU ABSENT at BB,
                   BBSome REDUCED in cilia, IFT NORMAL, PCP-enriched polydactyly.
                 BBS17 (LZTFL1): BBSome ciliary ENTRY gatekeeper → BBSome assembles
                   at BB but CANNOT ENTER cilia; Smo entry failure on SAG stimulation.
                 BBS18 (BBIP1): BBSome OCTAMER ASSEMBLY failure → all 8 subunits
                   ABSENT from cilia (assembly not docking/entry failure).
                 BBS19 (IFT172): IFT-B2 TIP CAP failure → IFT88 BULGING TIP
                   accumulation, cilia SHORTENED, dual IFT-B+BBSome defect.
                 BBS1 (BBS1): BBSome core β-propeller; GAE domain contacts WDPCP
                   WD40 blade 3–4 — BBS20 mechanistically upstream of BBS1 in cilia.

Disease        : Bardet-Biedl Syndrome Type 20 (#617119)
Gene OMIM      : *613580
Frequency      : ~0.5–1% of all BBS; ~20–30 families worldwide 2026
Inheritance    : Autosomal recessive (AR); biallelic LOF; single disease tier
Tri-allelic    : ~2% of BBS20 families
Cohort         : 40 patients; educational/synthetic; seed 371
Endpoints      : /api/bbs20/overview | /api/bbs20/breakdown | /api/bbs20/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 371
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("MENA",          0.40),  # Saudi, Jordanian, Lebanese — consanguineous enrichment
        ("European",      0.30),  # sporadic compound het families
        ("South Asian",   0.18),  # Pakistan, India
        ("North African", 0.08),  # consanguineous enrichment
        ("Other",         0.04),
    ]

    # Allele classes — all hypomorphic or null/hypomorphic (no null/null→NPHP tier)
    allele_classes = [
        ("Biallelic hypomorphic",         0.40),
        ("Compound het (null/hypomorphic)", 0.42),
        ("Biallelic null (severe BBS20)",  0.18),
    ]

    # Key pathogenic variants (WDPCP)
    variants = [
        ("Arg350Gln (c.1049G>A)",      0.26),  # WD40 blade 5, BBS1 primary interface — MENA
        ("Cys124Arg (c.370T>C)",        0.18),  # Blade 1 N-term edge, INTU secondary — MENA
        ("Glu542Lys (c.1624G>A)",       0.16),  # C-term CC dimer interface — European
        ("Pro267Leu (c.800C>T)",        0.14),  # WD40 blade 4 — South Asian
        ("c.1328+1G>A (splice-intron10)",0.13), # Null — pan-ethnic
        ("Gln700Ter (c.2098C>T)",       0.08),  # C-term truncation null — European
        ("Other / novel",               0.05),
    ]

    # Misdiagnosis patterns (pre-WES era)
    misdiagnoses = [
        ("Usher syndrome (RP + cognitive)", 0.30),
        ("Non-syndromic RP",               0.22),
        ("Isolated obesity + RP",          0.18),
        ("Prader-Willi syndrome",          0.12),
        ("BBS1 (incorrect panel)",         0.10),
        ("No misdiagnosis",                0.08),
    ]

    # Presentation (first presenting feature)
    presentations = [
        ("Night blindness / RP (retinal)",  0.38),
        ("Obesity in toddlerhood",          0.28),
        ("Post-axial polydactyly at birth", 0.22),
        ("Developmental delay",             0.08),
        ("Renal anomaly on ultrasound",     0.04),
    ]

    # Retinal stage
    retinal_stages = [
        ("Early (night blindness only)",          0.22),
        ("Moderate (peripheral field loss)",       0.35),
        ("Advanced (tunnel vision, VA 6/60)",      0.28),
        ("End-stage (LP / NLP)",                   0.15),
    ]

    # Renal patterns
    renal_patterns = [
        ("Normal",                       0.72),
        ("Renal cysts (structural)",      0.18),
        ("Cortical thinning (mild)",      0.06),
        ("Horseshoe / anomalous kidney",  0.04),
    ]

    # Dx age buckets
    dx_age_buckets = [
        ("0–2 yr (polydactyly / birth)",  0.22),
        ("3–8 yr (obesity + RP)",         0.34),
        ("9–15 yr (full pentad)",         0.26),
        ("16–25 yr (late diagnosis)",     0.14),
        (">25 yr (very late)",            0.04),
    ]

    def pick(choices):
        labels, weights = zip(*choices)
        return rng.choices(labels, weights=weights, k=1)[0]

    cohort = []
    for i in range(n):
        eth   = pick(ethnicities)
        ac    = pick(allele_classes)
        var1  = pick(variants)
        mis   = pick(misdiagnoses)
        pres  = pick(presentations)
        rs    = pick(retinal_stages)
        rp    = pick(renal_patterns)
        dxage = pick(dx_age_buckets)

        polydactyly  = rng.random() < 0.60   # PCP-enriched
        obesity      = rng.random() < 0.62
        cognitive    = rng.random() < 0.35
        hypogonadism = rng.random() < 0.42
        anosmia      = rng.random() < 0.38
        renal_any    = rp != "Normal"
        chd          = rng.random() < 0.05
        triallelic   = rng.random() < 0.02
        consanguinity= eth in ("MENA", "North African") and rng.random() < 0.72
        misdiagnosed = mis != "No misdiagnosis"
        intu_absent  = True  # WDPCP LOF → INTU always absent at BB

        cohort.append({
            "id":            i + 1,
            "ethnicity":     eth,
            "allele_class":  ac,
            "variant_1":     var1,
            "misdiagnosis":  mis,
            "presentation":  pres,
            "retinal_stage": rs,
            "renal_pattern": rp,
            "dx_age_bucket": dxage,
            "polydactyly":   polydactyly,
            "obesity":       obesity,
            "cognitive":     cognitive,
            "hypogonadism":  hypogonadism,
            "anosmia":       anosmia,
            "renal_any":     renal_any,
            "chd":           chd,
            "triallelic":    triallelic,
            "consanguinity": consanguinity,
            "misdiagnosed":  misdiagnosed,
            "intu_absent":   intu_absent,
        })

    return cohort


_COHORT: List[Dict[str, Any]] = _make_cohort()


def get_overview() -> Dict[str, Any]:
    c = _COHORT
    n = len(c)

    def pct(lst): return round(sum(1 for p in c if p[lst]) / n * 100)
    def renal_pct(): return round(sum(1 for p in c if p["renal_any"]) / n * 100)
    def retinal_endstage():
        return round(sum(1 for p in c if "End-stage" in p["retinal_stage"]) / n * 100)

    systemic_burden = [
        ("Post-axial polydactyly (PCP-enriched)", sum(1 for p in c if p["polydactyly"]), pct("polydactyly")),
        ("Obesity",                               sum(1 for p in c if p["obesity"]),      pct("obesity")),
        ("Hypogonadism",                          sum(1 for p in c if p["hypogonadism"]), pct("hypogonadism")),
        ("Cognitive / LD",                        sum(1 for p in c if p["cognitive"]),    pct("cognitive")),
        ("Anosmia / hyposmia",                    sum(1 for p in c if p["anosmia"]),      pct("anosmia")),
        ("Renal anomaly (any)",                   sum(1 for p in c if p["renal_any"]),    renal_pct()),
        ("Congenital heart defect",               sum(1 for p in c if p["chd"]),          pct("chd")),
        ("Tri-allelic BBS",                       sum(1 for p in c if p["triallelic"]),   pct("triallelic")),
        ("Prior misdiagnosis",                    sum(1 for p in c if p["misdiagnosed"]), pct("misdiagnosed")),
        ("Consanguinity",                         sum(1 for p in c if p["consanguinity"]),pct("consanguinity")),
        ("INTU absent at BB (all BBS20)",         n,                                      100),
    ]

    from collections import Counter
    rs_dist = Counter(p["retinal_stage"] for p in c)
    da_dist = Counter(p["dx_age_bucket"] for p in c)

    return {
        "cohort_n":               n,
        "gene":                   "WDPCP",
        "alias":                  "Fritz",
        "protein":                "WD Repeat Containing Planar Cell Polarity Effector",
        "protein_size_aa":        809,
        "locus":                  "2p15",
        "omim_gene":              "*613580",
        "omim_disease":           "#617119",
        "inheritance":            "Autosomal Recessive",
        "bbs_frequency_pct":      "~0.5–1%",
        "worldwide_families":     "20–30 (2026)",
        "triallelic_pct":         2,
        "disease_tier":           "Single tier (BBS20 only — no NPHP/MKS/JBTS allelic variant)",
        "pcp_enrichment":         "Polydactyly enriched (~60%) vs average BBS (~45%) — PCP effector role",
        "key_counts": {
            "polydactyly_pct":       pct("polydactyly"),
            "obesity_pct":           pct("obesity"),
            "cognitive_pct":         pct("cognitive"),
            "hypogonadism_pct":      pct("hypogonadism"),
            "anosmia_pct":           pct("anosmia"),
            "renal_any_pct":         renal_pct(),
            "chd_pct":               pct("chd"),
            "retinal_endstage_pct":  retinal_endstage(),
            "misdiagnosis_pct":      pct("misdiagnosed"),
            "consanguinity_pct":     pct("consanguinity"),
            "intu_absent_pct":       100,
        },
        "systemic_burden":       systemic_burden,
        "retinal_stage_distribution": dict(rs_dist),
        "dx_age_distribution":   dict(da_dist),
    }


def get_breakdown() -> Dict[str, Any]:
    c = _COHORT
    n = len(c)
    from collections import Counter

    def burden_list():
        pairs = [
            ("Post-axial polydactyly (PCP)", "polydactyly"),
            ("Obesity",                      "obesity"),
            ("Hypogonadism",                 "hypogonadism"),
            ("Cognitive / LD",               "cognitive"),
            ("Anosmia / hyposmia",           "anosmia"),
            ("Renal anomaly (any)",          "renal_any"),
            ("Congenital heart defect",      "chd"),
        ]
        return [
            {"feature": feat, "n": sum(1 for p in c if p[k]), "pct": round(sum(1 for p in c if p[k])/n*100)}
            for feat, k in pairs
        ]

    eth_dist  = Counter(p["ethnicity"] for p in c)
    ac_dist   = Counter(p["allele_class"] for p in c)
    rs_dist   = Counter(p["retinal_stage"] for p in c)
    rp_dist   = Counter(p["renal_pattern"] for p in c)
    poly_dist = Counter("Post-axial (PCP)" if p["polydactyly"] else "Absent" for p in c)
    pres_dist = Counter(p["presentation"] for p in c)
    md_dist   = Counter(p["misdiagnosis"] for p in c)
    var_dist  = Counter(p["variant_1"] for p in c)

    return {
        "systemic_burden":           burden_list(),
        "ethnicity_distribution":    [{"ethnicity": k, "n": v} for k, v in eth_dist.most_common()],
        "allele_class_summary":      [{"label": k, "n": v} for k, v in ac_dist.most_common()],
        "retinal_stage_distribution":[{"label": k, "n": v} for k, v in rs_dist.most_common()],
        "renal_distribution":        [{"label": k, "n": v} for k, v in rp_dist.most_common()],
        "polydactyly_distribution":  [{"label": k, "n": v} for k, v in poly_dist.most_common()],
        "presentation_distribution": [{"label": k, "n": v} for k, v in pres_dist.most_common()],
        "misdiagnosis_distribution": [{"label": k, "n": v} for k, v in md_dist.most_common()],
        "top_variants":              [{"variant": k, "n": v} for k, v in var_dist.most_common(6)],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":              "WDPCP",
            "Alias":             "Fritz (Frizzled Interacting Protein)",
            "Full name":         "WD Repeat Containing Planar Cell Polarity Effector",
            "Protein size":      "809 aa (~92 kDa) — CPLANE↔BBSome scaffold at basal body/TZ",
            "Chromosome":        "2p15",
            "OMIM gene":         "*613580",
            "Domains":           "N-terminal IDR + INTU/FUZ binding (aa 1–90, CPLANE docking, Arg17/Trp42 INTU contacts) · WD40 β-propeller (aa 91–500, 6–7 blades, BBS1 blade3–4 and BBS9 blade5–6 interface, Cys124/Pro267/Arg350 hotspots) · C-terminal coiled-coil (aa 501–809, Fritz dimerisation, BB/TZ anchor, Glu542/Gln700 sites)",
            "Expression":        "Ciliated cells — renal tubule, photoreceptors, olfactory epithelium, limb bud, hypothalamus, node",
            "Function":          "Scaffold bridging CPLANE complex (INTU+FUZ+CPLANE1) to BBSome (BBS1+BBS9) at basal body/TZ; recruits INTU to BB; ensures BBSome TZ-docking geometry; PCP effector linking Frizzled signalling to ciliogenesis",
            "Protein complex":   "WDPCP-INTU-FUZ (CPLANE docking module) + WDPCP WD40-BBS1-BBS9 (BBSome docking interface); homo-dimerises via C-terminal CC",
            "Interactors":       "INTU (N-term IDR contact), FUZ (N-term IDR), CPLANE1/CFAP126 (indirect CPLANE bridge), BBS1 (WD40 blade 3–4), BBS9/PTHB1 (WD40 blade 5–6), Frizzled receptors (indirect PCP axis)",
            "Key alias":         "Fritz (Drosophila homologue: frizzled interacting protein; human: WDPCP)",
            "Unique feature":    "Only BBS gene bridging CPLANE complex to BBSome; LOF → INTU ABSENT at BB (unique IF fingerprint not seen in any other BBS type); PCP-enriched polydactyly (~60%)",
        },
        "disease_card": {
            "Disease":            "Bardet-Biedl Syndrome Type 20",
            "OMIM disease":       "#617119",
            "Also":               "Single tier — no NPHP/MKS/JBTS allelic variant (WDPCP not a TZ structural gene)",
            "Mechanism":          "CPLANE↔BBSome bridge failure at BB/TZ — WDPCP LOF removes scaffold; INTU loses BB localisation; BBSome fails to dock at TZ gate → reduced ciliary BBSome; PCP defect from BB misdocking geometry",
            "Cilia morphology":   "Present, SHORTENED (~70–80% WT), occasionally MISORIENTED (PCP defect) — distinct from BBS17 (full-length, entry failure) and BBS19 (bulging-tip shortened)",
            "WDPCP IF":           "ABSENT from BB/subdistal appendages/TZ (protein lost from basal body)",
            "INTU IF":            "ABSENT/greatly reduced at BB — WDPCP-dependent recruitment lost; diagnostic for BBS20 among all BBS types",
            "BBSome IF":          "REDUCED in cilia (BBS2/BBS8 IF) — TZ docking failure; not absent (unlike BBS18 assembly failure)",
            "MKS1 TZ IF":         "NORMAL — TZ structural integrity intact",
            "IFT88 IF":           "NORMAL axonemal distribution — anterograde IFT intact; no tip accumulation (distinct from BBS19)",
            "ARL13B IF":          "REDUCED — BBSome-dependent ARL13B retrieval impaired",
            "MKKS/BCC IF":        "NORMAL — upstream chaperonin and assembly intact",
            "Renal type":         "Structural/cystic ~28% (mild, predominantly cortical cysts); lower than BBS16 SDCCAG8 or BBS13/14; no NPHP-dominant fibrosis",
            "Polydactyly note":   "Post-axial, PCP-enriched: ~60% vs average BBS ~45%; limb bud BB misdocking disrupts GLI3R/FL ratio via PCP pathway (similar to CPLANE1/JBTS33)",
            "Key distinction":    "BBS20 = CPLANE↔BBSome bridge; INTU ABSENT at BB + BBSome REDUCED. BBS17 = entry gatekeeper; BBSome ASSEMBLES but CANNOT ENTER cilia. BBS18 = assembly failure; all subunits ABSENT. BBS19 = IFT-B tip cap; IFT88 BULGING TIP + cilia SHORTENED.",
            "Disease tier":       "Single tier — BBS20 only; no allelic NPHP/MKS/JBTS reported",
            "Inheritance":        "Autosomal recessive biallelic LOF",
            "Prevalence":         "~1/500,000–1,000,000; 20–30 families worldwide 2026",
            "Tri-allelic":        "~2%",
            "First described":    "Human BBS20/WDPCP: Lim et al. / Kim et al. (first WDPCP human variants in BBS cohorts, ~2011–2016); Fritz (Drosophila homologue) first described by Lawrence group (planar cell polarity, ~2008)",
        },
        "key_variants": [
            {
                "variant":     "Arg350Gln (c.1049G>A)",
                "domain":      "WD40 β-propeller — blade 5, BBS1 primary interface (aa 350)",
                "consequence": "Missense; Arg→Gln abolishes a critical salt-bridge at the BBS1-WDPCP WD40 blade 5 docking surface; BBSome fails to dock at BB/TZ; INTU still partially localises (this interface is BBS1-specific, INTU contact aa 1–90 intact); most recurrent MENA allele (Saudi, Jordanian consanguineous cohorts); hypomorphic to moderate BBS20 — renal usually spared; polydactyly 65%",
                "ethnicity":   "MENA (Saudi / Jordanian)",
            },
            {
                "variant":     "Cys124Arg (c.370T>C)",
                "domain":      "WD40 β-propeller — blade 1 N-terminal edge, INTU secondary contact (aa 124)",
                "consequence": "Missense; Cys→Arg disrupts both the WD40 blade 1 structure and a secondary INTU contact surface (Cys124 is at the IDR/WD40 junction bridging both domains); dual defect: INTU BB localisation AND BBSome docking both impaired; compound het MENA allele (often with Arg350Gln); moderate-severe BBS20; renal cysts 40%",
                "ethnicity":   "MENA",
            },
            {
                "variant":     "Glu542Lys (c.1624G>A)",
                "domain":      "C-terminal coiled-coil — N-terminus of CC, Fritz dimerisation interface (aa 542)",
                "consequence": "Missense; Glu→Lys reverses charge at the CC dimer-interface; WDPCP homodimerisation impaired → scaffold less stable at BB; INTU and BBSome still partially recruited but geometry aberrant; European compound het (often with splice c.1328+1G>A); moderate BBS20; polydactyly 50%; renal spared",
                "ethnicity":   "European",
            },
            {
                "variant":     "Pro267Leu (c.800C>T)",
                "domain":      "WD40 β-propeller — blade 4, core structural proline (aa 267)",
                "consequence": "Missense; Pro→Leu disrupts the strict proline-required turn at WD40 blade 4; blade 4–5 β-hairpin destabilised; overall WD40 scaffold partially unfolded; reduced BBS9 (blade 5–6) and BBS1 (blade 3–4) contacts; South Asian enrichment (Pakistan consanguineous families); moderate-severe BBS20; renal cysts 35%",
                "ethnicity":   "South Asian",
            },
            {
                "variant":     "c.1328+1G>A (splice-donor intron 10)",
                "domain":      "Splice-donor intron 10 (exon 10/11 junction; mid WD40 region, aa ~370–400)",
                "consequence": "Abolishes canonical splice-donor; exon 10 skipping causes frameshift and PTC within WD40 blade 5–6; null allele (NMD target); ablates BBS9 docking surface entirely; compound het with hypomorphic alleles in BBS20 families; pan-ethnic; null allele class → combined with hypomorphic → BBS20 (single tier, no NPHP equiv.)",
                "ethnicity":   "Pan-ethnic",
            },
            {
                "variant":     "Gln700Ter (c.2098C>T)",
                "domain":      "C-terminal coiled-coil — mid-CC truncation (aa 700)",
                "consequence": "Truncating null; ablates the final 109 aa of the CC dimerisation domain; WDPCP cannot homodimerise → scaffold fails to anchor at BB; null allele; compound het with hypomorphic (Glu542Lys or Arg350Gln) in European families; moderate-severe BBS20; renal cysts possible 30%; polydactyly 55%",
                "ethnicity":   "European",
            },
        ],
        "diagnostic_workup": [
            "1. BBS multi-gene panel (Invitae/GeneDx/Blueprint Genetics) — WDPCP must be included; BBS20 underdiagnosed on older BBS panels that lack WDPCP; tri-allelic BBS screening recommended if phenotype unexpectedly severe for allele class.",
            "2. Full ophthalmic evaluation: best-corrected VA, full-field ERG (rod-first scotopic loss — same ERG pattern as BBS1–19), Goldmann perimetry, OCT (photoreceptor OS thinning); nystagmus screen; WDPCP retinal degeneration rate similar to BBS1/BBS10.",
            "3. Renal evaluation: renal ultrasound (cortical cysts, structural anomalies — ~28% BBS20 renal involvement); serum creatinine, eGFR, urinalysis; renal anomalies milder than BBS13/14/16; annual ultrasound recommended.",
            "4. Metabolic panel: BMI centiles, fasting glucose/HbA1c, lipid panel, fasting insulin (HOMA-IR); leptin level (BBSome-dependent LepR ciliary trafficking impaired — obesity early and marked).",
            "5. IF on patient fibroblasts or nasal epithelial cells: anti-WDPCP ABSENT at BB + anti-INTU ABSENT at BB (pathognomonic for BBS20, not seen in any other BBS type) + anti-BBS2 REDUCED in cilia + anti-IFT88 NORMAL (no tip accumulation) + acetylated α-tubulin SHORTENED, occasionally misoriented cilia.",
            "6. PCP / polydactyly assessment: X-ray of hands/feet for post-axial digit remnants; metatarsal polydactyly (orthopaedic/podiatric assessment); PCP pathway evaluation — note enrichment ~60% vs average BBS ~45%.",
            "7. Neuropsychological assessment: IQ, adaptive behaviour, speech/language (LD ~35%); ASD overlap documented in BBS cohort.",
            "8. Cardiac echo: CHD rare (~5%) but screen in paediatric BBS; WDPCP PCP pathway role may predispose to septal defects (low incidence).",
            "9. Olfactory testing (UPSIT or Sniffin Sticks) — anosmia ~38%; baseline and annual surveillance.",
            "10. Endocrine: gonadotrophin axis (LH, FSH, testosterone/oestradiol), bone age; hypogonadism ~42%; GH axis (GH defects documented in BBS).",
            "11. DDx checklist: BBS17 (LZTFL1) — Smo entry assay (FAILED in BBS17, NORMAL in BBS20); BBS18 (BBIP1) — BBSome ABSENT in cilia (vs REDUCED in BBS20); BBS19 (IFT172) — IFT88 tip accumulation (ABSENT in BBS20); confirm INTU absent at BB (unique to BBS20) before finalising genotype-phenotype report.",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; vitamin A supplementation per ophthalmologist (limited BBS-specific evidence); DHA/omega-3 (ongoing trials); no WDPCP-specific gene therapy trials as of 2026 (patient numbers too small); WDPCP zebrafish morphant cilia rescue with WDPCP mRNA injection (pre-clinical proof-of-concept).",
            "2. Obesity: early dietary/behavioural intervention; GLP-1 receptor agonists (semaglutide/liraglutide) off-label in BBS obesity — act downstream of LepR; BBSome-dependent LepR retrograde IFT impaired in BBS20 → marked leptin resistance; monitor for weight regain on cessation.",
            "3. Renal (BBS20): ACE inhibitor/ARB for proteinuria; nephrology referral at eGFR <60; annual ultrasound; renal transplant effective (no disease recurrence — cell-autonomous) if ESRD reached.",
            "4. Polydactyly: orthopaedic/plastic surgery in neonatal period for post-axial digital tags (timing per surgical assessment); PCP-associated polydactyly may involve metatarsal anomalies — podiatry assessment.",
            "5. Cognitive/LD: early intervention, IEP, speech therapy, occupational therapy — evidence-based for BBS; ASD overlap documented.",
            "6. Hypogonadism: sex-hormone replacement at puberty (testosterone/oestrogen-progesterone) per endocrinology; fertility counselling (male — oligospermia common; female — oligomenorrhoea).",
            "7. Research: WDPCP mRNA restoration via ASOs (splice-site c.1328+1G>A target amenable to exon-skipping rescue); CRISPR correction in WDPCP-LOF organoids (pre-clinical); small-molecule stabilisers of WDPCP WD40 scaffold (in silico screens ongoing); no IND filed as of 2026.",
            "8. Genetic counselling: AR recurrence risk 25%; carrier testing for siblings; prenatal/PGT-M available; single disease tier simplifies prognosis counselling (no null/null → MKS severity cliff unlike BBS14/BBS19).",
            "9. Annual surveillance: ERG, OCT, eGFR, BMI, UPSIT, gonadotrophins — track BBS20 multi-system progression; renal anomalies generally milder than SDCCAG8 (BBS16) or CEP290 (BBS14).",
        ],
    }
