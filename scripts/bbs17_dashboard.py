"""
BBS17 Bardet-Biedl Syndrome Type 17 — LZTFL1 Dashboard
=======================================================
Primary Gene : LZTFL1 (Leucine Zipper Transcription Factor Like 1;
               also known as HIPPI, BBS17).
               299 aa BBSome-interacting ciliary gatekeeper protein.
               Domain organisation (299 aa):
                 N-terminal segment (aa 1–60): unstructured; interaction
                   surface for BBS7 WD40 (minor contact);
                 Leucine Zipper domain (aa 65–145): 4.5 heptad-repeat LZ;
                   homodimerisation; TULP3-PHD interaction surface;
                   Arg154 at MENA-associated boundary;
                 Coiled-coil / LZ-junction (aa 150–200): BBS9/PTHB1 B9
                   domain binding interface; Pro192 structural hinge;
                 C-terminal regulatory domain (aa 200–299):
                   phosphorylation sites (Ser228, Thr245) governing
                   BBSome-entry gating; Ile241 within hydrophobic core;
                   Trp203 terminates LZ-junction at truncation hotspot.

               MECHANISM — BBSome Ciliary Entry Gating:
               LZTFL1 tethers the assembled BBSome to the distal end of
               the transition zone/ciliary gate, acting as a physical
               gatekeeper that controls BBSome entry into the ciliary
               axoneme. In the resting state, LZTFL1 retains the BBSome
               at the ciliary base. Upon Hedgehog pathway activation
               (Smo accumulation in cilia), LZTFL1 is phosphorylated
               (Ser228/Thr245) and releases the BBSome, allowing it to
               enter the cilium and traffic GPCRs/Smo for Hh signalling.

               LZTFL1 LOF consequence: BBSome correctly assembles (BCC
               chaperonin intact; BBS1/2/4/5/7/8/9 subunits normal) and
               traffics to the basal body. However, the BBSome CANNOT
               efficiently enter the cilia because the normal gatekeeper-
               release mechanism is absent. Net effect: ciliary cargo
               (GPCRs, LepR, Smo) cannot be properly loaded/unloaded at
               the ciliary gate. Cilia themselves form (full length).

               IF fingerprint — distinguishes BBS17:
                 LZTFL1 ABSENT from ciliary base/TZ (anti-LZTFL1 IF).
                 BBSome (BBS2, BBS4, BBS8, BBS9) REDUCED IN CILIA —
                   below normal ciliary body levels; present at basal body
                   (assembled correctly) but entry blocked.
                 MKKS/BCC: NORMAL (upstream chaperonin intact).
                 MKS1: NORMAL (TZ Y-link scaffold intact).
                 Cilia: FULL LENGTH present (NOT shortened — contrast BBS16
                   where cilia absent; NOT trapped at tip — contrast BBS15).
                 Smo: FAILS to accumulate in cilia upon Hh activation
                   (Smo remains at cell surface; BBSome entry blocked).

               Molecular distinction from BBS15 (IFT27/RABL4):
                 BBS15: BBSome TRAPPED in cilia (retrograde exit failure —
                   IFT27 GTP hydrolysis drives retrograde retrieval).
                 BBS17: BBSome CANNOT ENTER cilia efficiently (anterograde
                   entry failure — LZTFL1 gatekeeper absent).
               Both show reduced ciliary BBSome but the mechanism is
               exactly opposite: BBS15 = exit failure; BBS17 = entry failure.

               Connection to TULP3 / GPR75-GPR161 axis:
               LZTFL1 also forms a complex with TULP3 (tubby-like protein 3),
               which mediates import of GPCRs (GPR75, GPR161) into cilia via
               the LZTFL1-BBSome-TULP3 tri-partite gate. Loss of LZTFL1
               therefore also disrupts TULP3-dependent GPCR ciliary entry,
               amplifying the Hh/cAMP signalling defect beyond pure BBSome
               gating. This dual role (BBSome gate + TULP3 scaffold) makes
               BBS17 mechanistically the most complex ciliary gate LOF.

Disease        : Bardet-Biedl Syndrome Type 17 (#615991)
Gene OMIM      : *606568
Frequency      : <1% of all BBS (rare); ~20–35 families worldwide 2026
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~2% of BBS17 families (less than BBS1/2/10)
Cohort         : 40 patients; educational/synthetic; seed 365
Endpoints      : /api/bbs17/overview | /api/bbs17/breakdown | /api/bbs17/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 365
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("South Asian",   0.38),  # Pakistan/India — biallelic null predominance
        ("European",      0.28),
        ("MENA",          0.18),  # Turkey, Iran, Lebanon
        ("North African", 0.08),
        ("East Asian",    0.04),
        ("Other",         0.04),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    # BBS17-specific LZTFL1 variants (educational)
    # Trp203Ter = most common truncating (null); South Asian enrichment
    variants = [
        ("Trp203Ter (c.609G>A)",    0.28),  # truncating; South Asian / Pakistan most common
        ("c.430_433del (p.Pro144fs)",0.18), # frameshift; South Asian / North African
        ("Arg154Gln (c.461G>A)",    0.16),  # LZ-junction; MENA
        ("Pro192Leu (c.575C>T)",    0.14),  # coiled-coil/LZ hinge; European
        ("Thr85Ala (c.253A>G)",     0.12),  # BBS9/PTHB1 interface; European
        ("Ile241Asn (c.722T>A)",    0.12),  # C-terminal hydrophobic core; South Asian
    ]
    var_labels  = [v[0] for v in variants]
    var_weights = [v[1] for v in variants]

    # Allele classes
    allele_classes = [
        ("Biallelic null (LOF/LOF)",           0.42),
        ("Compound het null/hypomorphic",       0.38),
        ("Biallelic hypomorphic",               0.16),
        ("Tri-allelic BBS",                     0.04),
    ]
    ac_labels  = [a[0] for a in allele_classes]
    ac_weights = [a[1] for a in allele_classes]

    retinal_stages = [
        ("Early (ERG reduced, vision preserved)", 0.18),
        ("Mid (scotopic extinguished)",            0.28),
        ("Advanced (cone ERG affected)",           0.30),
        ("End-stage (NLP/HM)",                     0.24),
    ]
    rs_labels  = [r[0] for r in retinal_stages]
    rs_weights = [r[1] for r in retinal_stages]

    renal_patterns = [
        ("None",                              0.58),
        ("Structural anomaly (minor)",        0.12),
        ("CKD stage 1-2",                     0.14),
        ("CKD stage 3-4",                     0.10),
        ("ESRD (transplant/dialysis)",        0.06),
    ]
    rp_labels  = [r[0] for r in renal_patterns]
    rp_weights = [r[1] for r in renal_patterns]

    dx_ages = [
        ("0–2 yr (neonatal polydactyly)",     0.28),
        ("3–6 yr (visual symptoms)",          0.35),
        ("7–12 yr (obesity + LD flagged)",    0.25),
        ("13–18 yr (late / partial phenotype)",0.12),
    ]
    da_labels  = [d[0] for d in dx_ages]
    da_weights = [d[1] for d in dx_ages]

    misdiagnoses = [
        ("Syndromic obesity (unspecified)",   0.32),
        ("Isolated retinal dystrophy",        0.25),
        ("BBS1 / BBS2 (panel pending)",       0.20),
        ("Alström Syndrome",                  0.14),
        ("Other ciliopathy",                  0.09),
    ]
    md_labels  = [m[0] for m in misdiagnoses]
    md_weights = [m[1] for m in misdiagnoses]

    presentation_modes = [
        ("Polydactyly at birth",              0.28),
        ("Rod-cone dystrophy first",          0.36),
        ("Obesity + LD first",                0.22),
        ("Renal anomaly first",               0.08),
        ("Incidental panel find",             0.06),
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
        obesity        = rng.random() < 0.68   # ~68% — BBSome entry failure, LepR cannot signal
        cognitive      = rng.random() < 0.40   # LD/cognitive ~40%
        hypogonadism   = rng.random() < 0.48   # ~48%
        anosmia        = rng.random() < 0.44   # ~44%
        chd            = rng.random() < 0.06   # ~6%
        consanguinity  = rng.random() < 0.55   # South Asian, MENA
        triallelic     = rng.random() < 0.02   # ~2%
        misdiagnosed   = rng.random() < 0.68   # panel delay

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
        "gene":                   "LZTFL1",
        "protein":                "Leucine Zipper TF-Like 1",
        "protein_size_aa":        299,
        "locus":                  "2p22.1",
        "omim_gene":              "*606568",
        "omim_disease":           "#615991",
        "inheritance":            "Autosomal Recessive",
        "bbs_frequency_pct":      "<1%",
        "worldwide_families":     "20–35 (2026)",
        "triallelic_pct":         2,
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

    eth_dist = Counter(p["ethnicity"] for p in c)
    ac_dist  = Counter(p["allele_class"] for p in c)
    rs_dist  = Counter(p["retinal_stage"] for p in c)
    rp_dist  = Counter(p["renal_pattern"] for p in c)
    poly_dist= Counter("Post-axial" if p["polydactyly"] else "Absent" for p in c)
    pres_dist= Counter(p["presentation"] for p in c)
    md_dist  = Counter(p["misdiagnosis"] for p in c)
    var_dist = Counter(p["variant_1"] for p in c)

    return {
        "systemic_burden":          burden_list(),
        "ethnicity_distribution":   [{"ethnicity": k, "n": v} for k, v in eth_dist.most_common()],
        "allele_class_summary":     [{"label": k, "n": v} for k, v in ac_dist.most_common()],
        "retinal_stage_distribution":[{"label": k, "n": v} for k, v in rs_dist.most_common()],
        "renal_distribution":       [{"label": k, "n": v} for k, v in rp_dist.most_common()],
        "polydactyly_distribution": [{"label": k, "n": v} for k, v in poly_dist.most_common()],
        "presentation_distribution":[{"label": k, "n": v} for k, v in pres_dist.most_common()],
        "misdiagnosis_distribution":[{"label": k, "n": v} for k, v in md_dist.most_common()],
        "top_variants":             [{"variant": k, "n": v} for k, v in var_dist.most_common(6)],
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "gene_card": {
            "Gene":              "LZTFL1 (BBS17 / HIPPI)",
            "Full name":         "Leucine Zipper Transcription Factor Like 1",
            "Protein size":      "299 aa (~33 kDa)",
            "Chromosome":        "2p22.1",
            "OMIM gene":         "*606568",
            "Domains":           "LZ (aa 65–145) · LZ-junction (aa 150–200, BBS9/PTHB1 interface) · C-term regulatory (aa 200–299)",
            "Expression":        "Ciliated cells — renal tubule, photoreceptors, hypothalamus, limb bud mesenchyme",
            "Function":          "BBSome ciliary entry gatekeeper; TULP3 co-scaffold for GPCR import; Hh pathway modulator",
            "Protein complex":   "LZTFL1–BBSome–TULP3 tripartite gate at TZ/ciliary entry",
            "Interactors":       "BBS9/PTHB1 (direct), BBS7 (minor), TULP3-PHD, Smo (indirect via BBSome), LepR (indirect)",
            "Phospho-regulation":"Ser228, Thr245 — phosphorylation releases BBSome from gate during Hh activation",
            "Key alias":         "HIPPI (Hip1 Protein Interactor), LZTFL1",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 17",
            "OMIM disease":      "#615991",
            "Mechanism":         "BBSome ciliary entry failure (anterograde gate LOF — OPPOSITE of BBS15 retrograde trap)",
            "Cilia morphology":  "Full length present — NOT shortened/absent (contrast BBS16); NOT tip-trapped (contrast BBS15)",
            "BBSome IF":         "REDUCED IN CILIA; normal assembly at basal body; LZTFL1 absent from TZ/base",
            "MKKS/BCC IF":       "NORMAL",
            "MKS1 IF":           "NORMAL",
            "Smo ciliary entry": "BLOCKED on Hh activation — key functional read-out of BBS17",
            "Renal type":        "Structural/cystic ~42% (NOT NPHP-dominant — contrast BBS13/14/16)",
            "Retinal dystrophy": "Rod-cone; rod-FIRST scotopic extinguished",
            "Key distinction":   "BBS17 vs BBS15: entry failure vs exit failure; both reduce ciliary BBSome",
            "Allele-class":      "Single disease tier — no NPHP/MKS/JBTS equivalent (all alleles BBS17 only)",
            "Inheritance":       "Autosomal recessive biallelic LOF",
            "Prevalence":        "~1/1,000,000–3,000,000; 20–35 families worldwide 2026",
            "Tri-allelic":       "~2%",
        },
        "key_variants": [
            {
                "variant":     "Trp203Ter (c.609G>A)",
                "domain":      "LZ-junction / C-terminal boundary (aa 203)",
                "consequence": "Truncating null; South Asian (Pakistan); most common BBS17-associated LOF allele; deletes entire C-terminal regulatory domain",
                "ethnicity":   "South Asian / Pan-ethnic (null)",
            },
            {
                "variant":     "c.430_433del (p.Pro144fs)",
                "domain":      "LZ domain (aa 144 frameshift)",
                "consequence": "Frameshift null; disrupts BBS9/PTHB1 contact surface; South Asian and North African enrichment",
                "ethnicity":   "South Asian / North African",
            },
            {
                "variant":     "Arg154Gln (c.461G>A)",
                "domain":      "LZ-junction / coiled-coil boundary (aa 154)",
                "consequence": "Hypomorphic; destabilises BBS9/PTHB1 electrostatic contact; partial BBSome-gate assembly; MENA enrichment (Turkey, Iran)",
                "ethnicity":   "MENA",
            },
            {
                "variant":     "Pro192Leu (c.575C>T)",
                "domain":      "Coiled-coil / LZ-junction structural hinge (aa 192)",
                "consequence": "Hypomorphic; Pro→Leu introduces helix-breaking substitution at coiled-coil hinge; partial BBS9 interface disruption; European",
                "ethnicity":   "European",
            },
            {
                "variant":     "Thr85Ala (c.253A>G)",
                "domain":      "LZ domain — BBS9/PTHB1 binding interface (aa 85)",
                "consequence": "Hypomorphic missense; impairs direct BBS9/PTHB1 interaction; reduced BBSome gate assembly; European compound-het partner",
                "ethnicity":   "European",
            },
            {
                "variant":     "Ile241Asn (c.722T>A)",
                "domain":      "C-terminal regulatory domain hydrophobic core (aa 241)",
                "consequence": "Hypomorphic; Ile→Asn disrupts hydrophobic packing near Ser228 phospho-site; attenuated phosphorylation-dependent gate release; South Asian",
                "ethnicity":   "South Asian",
            },
        ],
        "diagnostic_workup": [
            "1. BBS 24-gene panel (Invitae/GeneDx) — LZTFL1 must be included; single-gene testing not recommended given BBS heterogeneity.",
            "2. Full ophthalmic evaluation: best-corrected VA, fundus photography, full-field ERG (rod-first scotopic loss), OCT (photoreceptor layer thinning).",
            "3. Renal ultrasound: structural anomalies / early cysts; serum creatinine, eGFR, urinalysis (proteinuria, casts — tubular).",
            "4. Metabolic panel: BMI centiles, fasting glucose/HbA1c (BBS obesity risk), leptin level (BBS17-specific LepR ciliary signalling failure).",
            "5. Functional Hh assay (research): Smo ciliary entry on SAG stimulation in patient fibroblasts — pathognomonic BBS17 = Smo BLOCKED.",
            "6. IF/confocal on fibroblasts: anti-LZTFL1 (absent at ciliary base) + anti-BBS9 (reduced in cilia) + acetylated α-tubulin (cilia present, full length).",
            "7. Neuropsychological assessment: IQ, adaptive behaviour, speech/language — LD reported in ~40%.",
            "8. Cardiac echo: CHD rare (~6%) but screen in paediatric BBS.",
            "9. Olfactory testing (UPSIT or Sniffin Sticks) — anosmia ~44%; baseline and annual.",
            "10. Endocrine: gonadotrophin axis (LH, FSH, testosterone/oestradiol), bone age — hypogonadism ~48%.",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; avoid high-oxygen environments; vitamin A supplementation (per ophthalmologist guidance — evidence limited in BBS); avoid drugs toxic to photoreceptors (hydroxychloroquine, vigabatrin); gene therapy trials (AAV2-LZTFL1) exploratory.",
            "2. Obesity: early dietary counselling; structured exercise programme; avoid calorie-dense feeds in infancy; no approved PCSK9i or GLP-1 analogues formally trialled in LZTFL1-BBS17 specifically, but semaglutide/liraglutide used off-label in BBS obesity (multi-BBS experience).",
            "3. Leptin resistance: LZTFL1 LOF impairs LepR ciliary signalling — metreleptin (recombinant leptin) unlikely to help if receptor cannot reach cilia; management is behavioural.",
            "4. Renal: ACE inhibitor / ARB for proteinuria; nephrology referral at eGFR <60; renal transplant effective (no disease recurrence in allograft).",
            "5. Polydactyly: orthopaedic surgery in neonatal period (post-axial digital tags vs metatarsal polydactyly — timing per surgical assessment).",
            "6. Cognitive/LD: early intervention, IEP, speech therapy — evidence-based; ASD overlap documented in BBS cohort.",
            "7. Hypogonadism: sex-hormone replacement at puberty (testosterone/oestrogen-progesterone) per endocrinology.",
            "8. Hh pathway consideration: research interest in ciliary Smo rescue (smoothened agonists, SAG) for BBS17 — rationale from Smo entry failure; no approved clinical use.",
            "9. Genetic counselling: AR recurrence risk 25%; carrier testing for siblings; prenatal/PGT-M available.",
            "10. Annual surveillance: ERG, OCT, eGFR, BMI, UPSIT, gonadotrophins — track BBS17 multi-system progression.",
        ],
    }
