"""
BBS19 Bardet-Biedl Syndrome Type 19 — IFT172 Dashboard
========================================================
Primary Gene : IFT172 (Intraflagellar Transport Protein 172).
               1749 aa (~198 kDa) — LARGEST IFT-B subunit; C-terminal
               WD40 repeat array; constitutes the TERMINAL CAP of the
               IFT-B2 subcomplex.
               Domain organisation (1749 aa):
                 N-terminal adaptor segment (aa 1–460): disordered
                   linker connecting IFT-B1 (IFT88) and IFT-B2 modules;
                   Gln455 contact surface for IFT88 (IFT-B1 docking);
                 Central β-propeller (aa 461–850): 8-bladed WD40
                   β-propeller; blades 3–6 (aa 600–800) form IFT-B2
                   intra-subcomplex contacts (IFT80, IFT57); blade 4
                   Arg852 tip-anchoring surface;
                 C-terminal WD40 array (aa 851–1749): 12 WD40 repeats
                   (blades 9–20); ciliary tip turnaround platform;
                   Gln914 (blade 5–6 junction) — IFT-A (IFT144) contact
                   for anterograde-to-retrograde switch; Ser1045 (blade 7)
                   — MENA hotspot; Trp1077 truncation hotspot; Leu1574
                   (blade ~16) — IFT88 C-term contact; hypomorphic.

               MECHANISM — IFT-B2 Tip-Turnaround (IFT172 unique function):
               IFT172 is the TERMINAL subunit of the IFT-B complex —
               it caps the anterograde IFT train at the ciliary tip and
               is essential for the turnaround step: releasing IFT-B from
               IFT-A to initiate retrograde trains. IFT172 also recruits
               BBSome to the ciliary tip via the BBS3/ARL6 GTPase cycle
               (with IFT27/BBS15) for retrograde cargo retrieval.

               IFT172 LOF consequence:
                 IFT-B2 subcomplex DISASSEMBLES at the tip — IFT80, IFT57
                   lose their terminal anchor; anterograde trains truncated;
                   cargo accumulates at bulging ciliary tips (tubulovesicular
                   structures visible by EM);
                 Cilia: SHORTENED / DYSMORPHIC (distinct from BBS15 where
                   cilia are full-length) — tip structure disrupted;
                 BBSome: REDUCED in cilia AND tip-trapped (IFT172 recruits
                   BBSome for exit; LOF = dual IFT-B and BBSome retrieval
                   failure — compound phenotype unique to BBS19);
                 LepR retrograde IFT is impaired — LepR ciliary export
                   requires IFT172-mediated tip turnaround; LOF = severe
                   hypothalamic LepR trapping → marked obesity.

               IF fingerprint — distinguishes BBS19:
                 IFT172 ABSENT (protein lost from cilia).
                 IFT88 (IFT-B1 subunit): BULGING TIP accumulation
                   (anterograde cargo stalls at tip without turnaround cap).
                 BBSome (anti-BBS2 IF): REDUCED/TIP-TRAPPED —
                   both IFT-B and BBSome phenotype co-occur (dual).
                 MKKS/BCC: NORMAL (upstream chaperonin intact).
                 MKS1/TZ: NORMAL (transition zone intact).
                 Cilia: SHORTENED, BULGING TIPS — distinct from BBS15
                   (full-length with tip-trapped BBSome only);
                   distinct from BBS18 (full-length, no IFT-B defect).

               Allele-class governs disease tier:
                 Biallelic null (null/null): NPHP17 (nephronophthisis-17,
                   #616033) — severe ESRD + retinal dystrophy + no BBS
                   features; analogous to CEP290 null→MKS vs
                   hypomorphic→BBS14.
                 Compound hypomorphic or null/hypomorphic: BBS19 (#615995)
                   — full pentad with renal cysts (not NPHP fibrosis).
                 Hypomorphic/hypomorphic: milder BBS19, renal spared.

               Distinction from BBS15 (IFT27) and other IFT-BBS genes:
                 BBS19 (IFT172): IFT-B2 TIP CAP; SHORTENED cilia +
                   IFT88 tip accumulation + BBSome reduced/tip-trapped.
                 BBS15 (IFT27): IFT-B1 GTPase; FULL-LENGTH cilia +
                   BBSome tip-trapped + no IFT88 tip accumulation.
                 BBS19 is the ONLY BBS gene causing both IFT-B AND
                 BBSome tip-trapping simultaneously (dual phenotype).

Disease        : Bardet-Biedl Syndrome Type 19 (#615995)
                 Also: NPHP17 (#616033) with biallelic null alleles
Gene OMIM      : *607386
Frequency      : ~0.5–1% of all BBS; ~30–50 families worldwide 2026
Inheritance    : Autosomal recessive (AR); biallelic LOF; at least one
                 hypomorphic allele required for BBS19 (not null/null)
Tri-allelic    : ~3% of BBS19 families
Cohort         : 40 patients; educational/synthetic; seed 369
Endpoints      : /api/bbs19/overview | /api/bbs19/breakdown | /api/bbs19/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 369
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("MENA",          0.38),  # Saudi, Jordanian, Lebanese — Aldahmesh 2014 first description
        ("European",      0.32),  # sporadic compound het families
        ("South Asian",   0.15),  # Pakistan, India
        ("North African", 0.10),  # consanguineous enrichment
        ("Other",         0.05),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    # BBS19-specific IFT172 variants (educational)
    # Ser1045Asn — MENA/Saudi; Aldahmesh 2014 first-reported BBS19 allele
    variants = [
        ("Ser1045Asn (c.3134G>A)",        0.25),  # WD40 blade 7; MENA/Saudi; Aldahmesh 2014
        ("Trp1077Ter (c.3231G>A)",        0.20),  # truncating null; pan-ethnic
        ("Gln914Arg (c.2741A>G)",         0.18),  # WD40 blade 5-6; IFT-A (IFT144) interface; European
        ("Arg852Cys (c.2554C>T)",         0.17),  # β-propeller blade 4; tip-anchoring; South Asian/MENA
        ("c.2943+1G>A (splice intron 21)",0.12),  # splice-donor null; pan-ethnic
        ("Leu1574Pro (c.4721T>C)",        0.08),  # C-term WD40 blade ~16; IFT88 contact; hypomorphic
    ]
    var_labels  = [v[0] for v in variants]
    var_weights = [v[1] for v in variants]

    allele_classes = [
        ("Compound het null/hypomorphic",       0.42),  # most common BBS19 genotype
        ("Biallelic hypomorphic",               0.35),  # milder BBS19
        ("Biallelic null (LOF/LOF)",            0.15),  # NPHP17 overlap — severe renal
        ("Tri-allelic BBS",                     0.03),
        ("Homozygous hypomorphic (founder)",    0.05),  # MENA consanguineous
    ]
    ac_labels  = [a[0] for a in allele_classes]
    ac_weights = [a[1] for a in allele_classes]

    retinal_stages = [
        ("Early (ERG reduced, vision preserved)", 0.12),
        ("Mid (scotopic extinguished)",            0.28),
        ("Advanced (cone ERG affected)",           0.35),
        ("End-stage (NLP/HM)",                     0.25),
    ]
    rs_labels  = [r[0] for r in retinal_stages]
    rs_weights = [r[1] for r in retinal_stages]

    renal_patterns = [
        ("None",                              0.58),
        ("Structural/cystic (BBS19 tier)",    0.18),
        ("CKD stage 1-2",                     0.12),
        ("CKD stage 3-4",                     0.08),
        ("ESRD — NPHP17 biallelic null",      0.04),
    ]
    rp_labels  = [r[0] for r in renal_patterns]
    rp_weights = [r[1] for r in renal_patterns]

    dx_ages = [
        ("0–2 yr (neonatal polydactyly)",      0.28),
        ("3–6 yr (visual symptoms)",           0.35),
        ("7–12 yr (obesity + LD flagged)",     0.27),
        ("13–18 yr (late / partial phenotype)", 0.10),
    ]
    da_labels  = [d[0] for d in dx_ages]
    da_weights = [d[1] for d in dx_ages]

    misdiagnoses = [
        ("Syndromic obesity (unspecified)",    0.32),
        ("Isolated retinal dystrophy",         0.25),
        ("BBS1 / BBS2 (panel pending)",        0.20),
        ("Alström Syndrome",                   0.12),
        ("NPHP (nephronophthisis)",            0.07),
        ("Other ciliopathy",                   0.04),
    ]
    md_labels  = [m[0] for m in misdiagnoses]
    md_weights = [m[1] for m in misdiagnoses]

    presentation_modes = [
        ("Polydactyly at birth",              0.28),
        ("Rod-cone dystrophy first",          0.35),
        ("Obesity + LD first",                0.22),
        ("Renal anomaly first",               0.08),
        ("Incidental panel find",             0.07),
    ]
    pm_labels  = [p[0] for p in presentation_modes]
    pm_weights = [p[1] for p in presentation_modes]

    cohort = []
    for i in range(n):
        eth  = rng.choices(eth_labels,  eth_weights)[0]
        var1 = rng.choices(var_labels,  var_weights)[0]
        var2 = rng.choices(var_labels,  var_weights)[0]
        ac   = rng.choices(ac_labels,   ac_weights)[0]
        rs   = rng.choices(rs_labels,   rs_weights)[0]
        rp   = rng.choices(rp_labels,   rp_weights)[0]
        da   = rng.choices(da_labels,   da_weights)[0]
        md   = rng.choices(md_labels,   md_weights)[0]
        pm   = rng.choices(pm_labels,   pm_weights)[0]

        polydactyly    = rng.random() < 0.55   # post-axial; ~55%
        obesity        = rng.random() < 0.72   # ~72% — severe LepR retrograde IFT failure
        cognitive      = rng.random() < 0.38   # LD/cognitive ~38%
        hypogonadism   = rng.random() < 0.55   # ~55%
        anosmia        = rng.random() < 0.48   # ~48%
        chd            = rng.random() < 0.04   # ~4%
        consanguinity  = rng.random() < 0.55   # MENA enrichment
        triallelic     = rng.random() < 0.03   # ~3%
        misdiagnosed   = rng.random() < 0.70   # rare — panel delay common

        renal_any = rp not in ("None",)

        cohort.append({
            "id":            i + 1,
            "ethnicity":     eth,
            "allele_class":  ac,
            "variant_1":     var1,
            "variant_2":     var2,
            "retinal_stage": rs,
            "renal_pattern": rp,
            "renal_any":     renal_any,
            "dx_age_bucket": da,
            "misdiagnosis":  md,
            "presentation":  pm,
            "polydactyly":   polydactyly,
            "obesity":       obesity,
            "cognitive":     cognitive,
            "hypogonadism":  hypogonadism,
            "anosmia":       anosmia,
            "chd":           chd,
            "consanguinity": consanguinity,
            "triallelic":    triallelic,
            "misdiagnosed":  misdiagnosed,
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
        ("Post-axial polydactyly",     sum(1 for p in c if p["polydactyly"]),    pct("polydactyly")),
        ("Obesity",                    sum(1 for p in c if p["obesity"]),         pct("obesity")),
        ("Hypogonadism",               sum(1 for p in c if p["hypogonadism"]),    pct("hypogonadism")),
        ("Cognitive / LD",             sum(1 for p in c if p["cognitive"]),       pct("cognitive")),
        ("Anosmia / hyposmia",         sum(1 for p in c if p["anosmia"]),         pct("anosmia")),
        ("Renal anomaly (any)",        sum(1 for p in c if p["renal_any"]),       renal_pct()),
        ("Congenital heart defect",    sum(1 for p in c if p["chd"]),             pct("chd")),
        ("Tri-allelic BBS",            sum(1 for p in c if p["triallelic"]),      pct("triallelic")),
        ("Prior misdiagnosis",         sum(1 for p in c if p["misdiagnosed"]),    pct("misdiagnosed")),
        ("Consanguinity",              sum(1 for p in c if p["consanguinity"]),   pct("consanguinity")),
    ]

    from collections import Counter
    rs_dist = Counter(p["retinal_stage"] for p in c)
    da_dist = Counter(p["dx_age_bucket"] for p in c)

    return {
        "cohort_n":               n,
        "gene":                   "IFT172",
        "protein":                "Intraflagellar Transport Protein 172",
        "protein_size_aa":        1749,
        "locus":                  "2p23.3",
        "omim_gene":              "*607386",
        "omim_disease":           "#615995",
        "omim_nphp17":            "#616033",
        "inheritance":            "Autosomal Recessive",
        "bbs_frequency_pct":      "~0.5–1%",
        "worldwide_families":     "30–50 (2026)",
        "triallelic_pct":         3,
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
            ("Post-axial polydactyly",  "polydactyly"),
            ("Obesity",                 "obesity"),
            ("Hypogonadism",            "hypogonadism"),
            ("Cognitive / LD",          "cognitive"),
            ("Anosmia / hyposmia",      "anosmia"),
            ("Renal anomaly (any)",     "renal_any"),
            ("Congenital heart defect", "chd"),
        ]
        return [
            {"feature": feat, "n": sum(1 for p in c if p[k]), "pct": round(sum(1 for p in c if p[k])/n*100)}
            for feat, k in pairs
        ]

    eth_dist  = Counter(p["ethnicity"] for p in c)
    ac_dist   = Counter(p["allele_class"] for p in c)
    rs_dist   = Counter(p["retinal_stage"] for p in c)
    rp_dist   = Counter(p["renal_pattern"] for p in c)
    poly_dist = Counter("Post-axial" if p["polydactyly"] else "Absent" for p in c)
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
            "Gene":              "IFT172",
            "Full name":         "Intraflagellar Transport Protein 172",
            "Protein size":      "1749 aa (~198 kDa) — largest IFT-B subunit",
            "Chromosome":        "2p23.3",
            "OMIM gene":         "*607386",
            "Domains":           "N-term adaptor/IFT88 contact (aa 1–460) · Central β-propeller WD40 (aa 461–850, 8 blades, tip-anchor surface blade 4) · C-terminal WD40 array (aa 851–1749, 12 blades, IFT-A contact Gln914, turnaround platform)",
            "Expression":        "Ciliated cells — renal tubule, photoreceptors, hypothalamus, olfactory epithelium, limb bud, node",
            "Function":          "Terminal cap of IFT-B2 subcomplex; anchors anterograde IFT train at ciliary tip for turnaround; recruits BBS3/ARL6-BBSome for retrograde cargo retrieval; essential for LepR retrograde IFT",
            "Protein complex":   "IFT-B2 subcomplex (IFT172+IFT80+IFT57+IFT52+IFT46+IFT38+IFT20); caps IFT-B1 module (IFT88+IFT81+IFT74+IFT70+IFT56+IFT52+IFT25+IFT27) at ciliary tip",
            "Interactors":       "IFT88 (IFT-B1, N-term contact Gln455/Leu1574), IFT80 (IFT-B2 partner), IFT144 (IFT-A, Gln914 turnaround contact), BBS3/ARL6 (BBSome recruitment), IFT27/BBS15 (tip GTPase cycle)",
            "Key alias":         "IFT172 (no common alias); BBS19 locus gene",
            "Unique feature":    "Largest IFT-B subunit (1749 aa); ONLY IFT-B gene causing both IFT-B tip accumulation AND BBSome tip-trapping — dual phenotype",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 19",
            "OMIM disease":      "#615995",
            "Also":              "NPHP17 (#616033) with biallelic null alleles",
            "Mechanism":         "IFT-B2 tip-turnaround failure — IFT172 LOF removes terminal cap; anterograde trains stall at tip; cargo (including IFT-B and BBSome) accumulates; retrograde trains cannot initiate",
            "Cilia morphology":  "Shortened / dysmorphic with bulging tips — distinct from BBS15 (full-length), BBS18 (full-length), BBS17 (full-length)",
            "IFT172 IF":         "ABSENT from cilia (protein lost); IFT88 IF: BULGING TIP accumulation (stalled cargo)",
            "BBSome IF":         "REDUCED in cilia AND tip-trapped — dual IFT-B + BBSome phenotype; unique to BBS19 among all BBS types",
            "MKKS/BCC IF":       "NORMAL — upstream chaperonin intact",
            "MKS1/TZ IF":        "NORMAL — transition zone intact",
            "Renal type":        "Structural/cystic ~42% in BBS19 tier (NOT NPHP-dominant); NPHP17 tier if biallelic null (ESRD + severe retinal)",
            "Retinal dystrophy": "Rod-cone; rod-FIRST scotopic extinguished",
            "Key distinction":   "BBS19 = IFT-B tip cap loss → SHORTENED cilia + IFT88 tip accumulation + BBSome reduced. BBS15 = IFT GTPase loss → FULL-LENGTH cilia + BBSome tip-trapped only. BBS18 = BBSome assembly failure → full-length cilia, no IFT-B defect.",
            "Allele-class":      "Null/null → NPHP17 (severe); null/hypomorphic or hypomorphic/hypomorphic → BBS19",
            "Inheritance":       "Autosomal recessive biallelic LOF",
            "Prevalence":        "~1/500,000–1,000,000; 30–50 families worldwide 2026",
            "Tri-allelic":       "~3%",
            "First described":   "Aldahmesh et al. 2014 (Nat Commun); Saudi consanguineous families; Ser1045Asn first BBS19 allele",
        },
        "key_variants": [
            {
                "variant":     "Ser1045Asn (c.3134G>A)",
                "domain":      "C-terminal WD40 array — blade 7 (aa 1045)",
                "consequence": "Missense; Ser→Asn alters hydrogen-bond network within WD40 blade 7; destabilises IFT-B2 tip platform; first BBS19 allele reported (Aldahmesh 2014, Nat Commun); MENA/Saudi enrichment; hypomorphic — BBS19 tier, renal spared in homozygotes",
                "ethnicity":   "MENA / Saudi",
            },
            {
                "variant":     "Trp1077Ter (c.3231G>A)",
                "domain":      "C-terminal WD40 array — blade 8 truncation (aa 1077)",
                "consequence": "Truncating null; ablates blades 8–20 (672 aa lost); complete loss of IFT-A (IFT144) contact surface and turnaround platform; IFT-B2 subcomplex collapses; null allele — NPHP17 in biallelic null; BBS19 when compound with hypomorphic; Pan-ethnic",
                "ethnicity":   "Pan-ethnic",
            },
            {
                "variant":     "Gln914Arg (c.2741A>G)",
                "domain":      "Central β-propeller — blade 5–6 junction (aa 914); IFT-A (IFT144) interface",
                "consequence": "Missense; Gln→Arg at the IFT-A docking surface; disrupts anterograde-to-retrograde switch; IFT trains stall at tip without releasing to retrograde; partial turnaround defect; European compound het with null allele; BBS19 tier",
                "ethnicity":   "European",
            },
            {
                "variant":     "Arg852Cys (c.2554C>T)",
                "domain":      "Central β-propeller — blade 4 tip-anchoring surface (aa 852)",
                "consequence": "Missense; Arg→Cys disrupts charged tip-anchoring surface; IFT172 cannot tether to ciliary tip membrane; tip turnaround platform destabilised; IFT-B2 partially deanchored; South Asian/MENA compound het; BBS19 tier",
                "ethnicity":   "South Asian / MENA",
            },
            {
                "variant":     "c.2943+1G>A (splice-donor intron 21)",
                "domain":      "Splice-donor intron 21 (exon 21/22 junction; central WD40 region)",
                "consequence": "Abolishes canonical splice-donor; intron 21 retention causes frameshift and premature termination within WD40 blade region; null allele; compound het with hypomorphic allele in BBS19 families; Pan-ethnic",
                "ethnicity":   "Pan-ethnic",
            },
            {
                "variant":     "Leu1574Pro (c.4721T>C)",
                "domain":      "C-terminal WD40 array — blade ~16 (aa 1574); IFT88 C-term contact",
                "consequence": "Hypomorphic missense; Leu→Pro in C-terminal WD40; disrupts IFT88 long-range contact; weakens IFT-B1/IFT-B2 coupling; partial tip-turnaround impairment; milder BBS19 phenotype — renal often spared; European compound het or homozygous",
                "ethnicity":   "European",
            },
        ],
        "diagnostic_workup": [
            "1. BBS 24-gene panel (Invitae/GeneDx/Blueprint Genetics) — IFT172 must be included; BBS19 underdiagnosed on older panels that lack IFT172; tri-allelic BBS screening recommended if phenotype unexpectedly severe.",
            "2. Full ophthalmic evaluation: best-corrected VA, full-field ERG (rod-first scotopic loss — same ERG pattern as BBS1–18), Goldmann perimetry, OCT (photoreceptor outer segment thinning); nystagmus screen.",
            "3. Renal evaluation: ultrasound (structural/cystic anomalies; cortical thinning if NPHP17 allele class); serum creatinine, eGFR, urinalysis (proteinuria — tubular in NPHP17 tier, glomerular in BBS19 cystic tier); nephrology referral at eGFR <60.",
            "4. Metabolic panel: BMI centiles, waist circumference, fasting glucose/HbA1c, lipid panel, fasting insulin (HOMA-IR); leptin level (IFT172 LOF → severe LepR retrograde IFT failure → markedly elevated leptin resistance).",
            "5. IF/confocal on fibroblasts or RPE: anti-IFT172 ABSENT + anti-IFT88 BULGING TIP accumulation (stalled anterograde cargo) + anti-BBS2 REDUCED/TIP-TRAPPED + acetylated α-tubulin SHORTENED cilia — BBS19 dual IFT-B+BBSome fingerprint.",
            "6. Electron microscopy (research): ciliary tip tubulovesicular structures (characteristic of IFT-B2 turnaround failure); distinguishes BBS19 from BBS15 (no EM tip anomaly in BBS15).",
            "7. Neuropsychological assessment: IQ, adaptive behaviour, speech/language (LD ~38%); ASD overlap documented in BBS cohort.",
            "8. Cardiac echo: CHD rare (~4%) but screen in paediatric BBS.",
            "9. Olfactory testing (UPSIT or Sniffin Sticks) — anosmia ~48%; baseline and annual surveillance.",
            "10. Endocrine: gonadotrophin axis (LH, FSH, testosterone/oestradiol), bone age; hypogonadism ~55%; growth hormone axis (GH axis defects documented in BBS).",
            "11. NPHP17 allele-class triage: if biallelic null (two truncating alleles), anticipate NPHP17 severity — early nephrology, renal transplant planning, skip BBS obesity/polydactyly workup if features absent.",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; avoid high-oxygen environments; vitamin A supplementation (per ophthalmologist — limited BBS evidence); no IFT172-specific gene therapy trials as of 2026 (patient numbers too small); IFT172 overexpression in zebrafish ift172 morphants rescues cilia length and photoreceptor OS.",
            "2. Obesity: early dietary/behavioural intervention; GLP-1 receptor agonists (semaglutide/liraglutide) used off-label in BBS obesity — act downstream of LepR; IFT172 LOF causes severe LepR retrograde IFT failure so leptin resistance is particularly pronounced in BBS19; monitor for weight regain after discontinuation.",
            "3. Renal (BBS19 tier): ACE inhibitor / ARB for proteinuria; nephrology referral at eGFR <60; renal ultrasound annually; renal transplant effective (no disease recurrence in allograft).",
            "4. Renal (NPHP17 tier — biallelic null): early aggressive nephrology; salt supplementation for concentrating defect; erythropoietin for anaemia; renal transplant planning by CKD stage 3; live-donor transplant preferred.",
            "5. Polydactyly: orthopaedic/plastic surgery in neonatal period (post-axial digital tags — timing per surgical assessment); metatarsal polydactyly assessed by podiatry/orthopaedics.",
            "6. Cognitive/LD: early intervention, IEP, speech therapy, occupational therapy — evidence-based; ASD overlap documented.",
            "7. Hypogonadism: sex-hormone replacement at puberty (testosterone/oestrogen-progesterone) per endocrinology; fertility counselling (male — azoospermia/oligospermia; female — oligomenorrhoea).",
            "8. Research: IFT172 mRNA restoration (ASOs targeting splice variants); small-molecule IFT-B2 assembly chaperones; CRISPR correction in organoids (pre-clinical); no IND filed as of 2026.",
            "9. Genetic counselling: AR recurrence risk 25%; carrier testing for siblings; prenatal/PGT-M available; allele-class counselling critical (null/null → NPHP17 severity must be disclosed pre-conception).",
            "10. Annual surveillance: ERG, OCT, eGFR, BMI, UPSIT, gonadotrophins, leptin — track BBS19 multi-system progression; renal trajectory diverges sharply between BBS19 and NPHP17 allele classes.",
        ],
    }
