"""
BBS18 Bardet-Biedl Syndrome Type 18 — BBIP1 Dashboard
=======================================================
Primary Gene : BBIP1 (BBSome-Interacting Protein 1;
               also known as BBIP10, BBS18).
               232 aa BBSome core subunit — the 8th integral subunit of
               the BBSome complex (alongside BBS1, BBS2, BBS4, BBS5,
               BBS7, BBS8, BBS9).
               Domain organisation (232 aa):
                 N-terminal segment (aa 1–30): unstructured; BBS1 β-propeller
                   contact surface; Gln36 truncation hotspot;
                 Coiled-coil domain (aa 35–160): major structural scaffold
                   for BBSome core assembly; BBIP1-BBS9 primary interface
                   at aa 50–80 (Pro57 key contact); Leu150 deletion
                   hotspot within coiled-coil hydrophobic core;
                 C-terminal domain (aa 161–232): BBS2/BBS7 interaction
                   surface; Thr203 phospho-site; Arg211 truncation
                   hotspot.

               MECHANISM — BBSome Core Assembly:
               BBIP1 is the 8th integral subunit of the BBSome, a
               highly-conserved ciliary coat complex (MW ~450 kDa) that
               traffics GPCRs, LepR, Smo, and other cargo within the
               ciliary axoneme. The BBSome assembles sequentially at the
               periciliary region (basal body); BBIP1 bridges the BBS9
               scaffold subunit to BBS2/BBS7, stabilising the core ring.

               BBIP1 LOF consequence: BBSome FAILS TO ASSEMBLE correctly
               — core ring is destabilised and the complex cannot form a
               stable octamer. Unassembled BBSome subunits accumulate at
               the basal body/cytoplasm (not in cilia). Ciliary cargo
               (GPCRs, LepR, Smo) are stranded outside the cilium.
               Cilia themselves form and are full length (BBIP1 is not
               required for axonemal growth, only for BBSome assembly).

               IF fingerprint — distinguishes BBS18 (assembly failure):
                 BBIP1 ABSENT from cilia and basal body (protein lost).
                 BBSome subunits BBS2, BBS4, BBS5, BBS7, BBS8, BBS9:
                   ALL ABSENT/severely reduced from cilia — assembly
                   failure means no functional BBSome enters cilia
                   (contrast BBS15 where BBSome enters but is tip-trapped;
                    contrast BBS17 where BBSome assembles but cannot enter).
                 MKKS/BCC chaperonin: NORMAL — upstream BCC intact;
                   individual BBS subunits fold but cannot form octamer.
                 MKS1/TZ Y-link scaffold: NORMAL.
                 Cilia: PRESENT, FULL LENGTH — ciliogenesis intact.
                 IFT: NORMAL — IFT-A/B axonemal transport intact.

               Molecular position within BBSome hierarchy:
                 BBS1-2-4-5-7-8-9: core structural subunits (RING+GAE+PH
                   domain scaffold).
                 BBIP1 (BBS18): 8th subunit, CC-domain bridge between
                   BBS9 (scaffold spine) and BBS2/BBS7 (lid).
                 LZTFL1 (BBS17): NOT a subunit — external gate regulator.
                 IFT27 (BBS15): NOT a subunit — GTPase recruiter.
               BBS18 LOF = assembly failure (same tier as BBS1-12 core).

               Distinction from BBS15 (IFT27) and BBS17 (LZTFL1):
                 BBS18: BBSome DOES NOT ASSEMBLE — no ciliary BBSome at all.
                 BBS17: BBSome ASSEMBLES but cannot enter cilia.
                 BBS15: BBSome ASSEMBLES, ENTERS, but is tip-trapped.
               All three show absent ciliary BBSome but at different steps;
               the upstream molecular lesion (assembly vs entry vs exit)
               distinguishes them.

Disease        : Bardet-Biedl Syndrome Type 18 (#615994)
Gene OMIM      : *613605
Frequency      : <1% of all BBS (very rare); ~10–20 families worldwide 2026
Inheritance    : Autosomal recessive (AR); biallelic LOF
Tri-allelic    : ~2% of BBS18 families
Cohort         : 40 patients; educational/synthetic; seed 367
Endpoints      : /api/bbs18/overview | /api/bbs18/breakdown | /api/bbs18/definitions
"""

import random
from typing import Any, Dict, List

_SEED = 367
_N    = 40

# ── reproducible patient cohort ──────────────────────────────────────────────
def _make_cohort(seed: int = _SEED, n: int = _N) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    ethnicities = [
        ("MENA",          0.40),  # Turkey, Iran, Lebanon, Saudi Arabia — consanguineous enrichment
        ("European",      0.30),  # first null allele reported in European family
        ("South Asian",   0.15),  # Pakistan/India
        ("North African", 0.10),  # Algeria, Morocco — consanguineous
        ("Other",         0.05),
    ]
    eth_labels  = [e[0] for e in ethnicities]
    eth_weights = [e[1] for e in ethnicities]

    # BBS18-specific BBIP1 variants (educational)
    # Leu150del (c.448_450delCTT) — first reported; MENA consanguineous
    variants = [
        ("Leu150del (c.448_450delCTT)",  0.28),  # in-frame del; coiled-coil core; MENA; first reported
        ("Gln36Ter (c.106C>T)",          0.20),  # truncating null; European
        ("c.64+2T>C (splice-donor)",     0.18),  # splice null; Middle Eastern
        ("Pro57Ser (c.169C>T)",          0.16),  # hypomorphic; BBS9 interface; European
        ("Arg211Ter (c.631C>T)",         0.10),  # C-terminal truncating null; Pan-ethnic
        ("Thr203Asn (c.608C>A)",         0.08),  # hypomorphic; phospho-site adjacent; MENA/South Asian
    ]
    var_labels  = [v[0] for v in variants]
    var_weights = [v[1] for v in variants]

    allele_classes = [
        ("Biallelic null (LOF/LOF)",           0.40),
        ("Compound het null/hypomorphic",       0.40),
        ("Biallelic hypomorphic",               0.18),
        ("Tri-allelic BBS",                     0.02),
    ]
    ac_labels  = [a[0] for a in allele_classes]
    ac_weights = [a[1] for a in allele_classes]

    retinal_stages = [
        ("Early (ERG reduced, vision preserved)", 0.15),
        ("Mid (scotopic extinguished)",            0.30),
        ("Advanced (cone ERG affected)",           0.32),
        ("End-stage (NLP/HM)",                     0.23),
    ]
    rs_labels  = [r[0] for r in retinal_stages]
    rs_weights = [r[1] for r in retinal_stages]

    renal_patterns = [
        ("None",                              0.70),
        ("Structural anomaly (minor)",        0.12),
        ("CKD stage 1-2",                     0.10),
        ("CKD stage 3-4",                     0.06),
        ("ESRD (transplant/dialysis)",        0.02),
    ]
    rp_labels  = [r[0] for r in renal_patterns]
    rp_weights = [r[1] for r in renal_patterns]

    dx_ages = [
        ("0–2 yr (neonatal polydactyly)",     0.25),
        ("3–6 yr (visual symptoms)",          0.38),
        ("7–12 yr (obesity + LD flagged)",    0.26),
        ("13–18 yr (late / partial phenotype)",0.11),
    ]
    da_labels  = [d[0] for d in dx_ages]
    da_weights = [d[1] for d in dx_ages]

    misdiagnoses = [
        ("Syndromic obesity (unspecified)",   0.34),
        ("Isolated retinal dystrophy",        0.26),
        ("BBS1 / BBS2 (panel pending)",       0.22),
        ("Alström Syndrome",                  0.10),
        ("Other ciliopathy",                  0.08),
    ]
    md_labels  = [m[0] for m in misdiagnoses]
    md_weights = [m[1] for m in misdiagnoses]

    presentation_modes = [
        ("Polydactyly at birth",              0.25),
        ("Rod-cone dystrophy first",          0.38),
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

        polydactyly    = rng.random() < 0.50   # post-axial; ~50%
        obesity        = rng.random() < 0.65   # ~65% — BBSome assembly failure blocks LepR ciliary trafficking
        cognitive      = rng.random() < 0.35   # LD/cognitive ~35%
        hypogonadism   = rng.random() < 0.45   # ~45%
        anosmia        = rng.random() < 0.40   # ~40%
        chd            = rng.random() < 0.04   # ~4%
        consanguinity  = rng.random() < 0.60   # MENA, North African enrichment
        triallelic     = rng.random() < 0.02   # ~2%
        misdiagnosed   = rng.random() < 0.72   # very rare — panel delay high

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
        "gene":                   "BBIP1",
        "protein":                "BBSome-Interacting Protein 1",
        "protein_size_aa":        232,
        "locus":                  "10q25.2",
        "omim_gene":              "*613605",
        "omim_disease":           "#615994",
        "inheritance":            "Autosomal Recessive",
        "bbs_frequency_pct":      "<1%",
        "worldwide_families":     "10–20 (2026)",
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
            "Gene":              "BBIP1 (BBS18 / BBIP10)",
            "Full name":         "BBSome-Interacting Protein 1",
            "Protein size":      "232 aa (~24.9 kDa)",
            "Chromosome":        "10q25.2",
            "OMIM gene":         "*613605",
            "Domains":           "N-term BBS1 contact (aa 1–30) · Coiled-coil scaffold (aa 35–160, BBS9/BBS2/BBS7 interface) · C-term domain (aa 161–232, BBS2/BBS7 lid)",
            "Expression":        "Ciliated cells — renal tubule, photoreceptors, hypothalamus, olfactory epithelium, limb bud",
            "Function":          "8th integral BBSome subunit; CC-domain bridge between BBS9 scaffold and BBS2/BBS7 lid; required for BBSome core assembly",
            "Protein complex":   "BBSome octamer (BBS1+BBS2+BBS4+BBS5+BBS7+BBS8+BBS9+BBIP1); BBIP1 is the CC-domain bridge subunit",
            "Interactors":       "BBS9 (primary, aa 50–80 CC interface), BBS2, BBS7, BBS1 β-propeller (N-term)",
            "Phospho-site":      "Thr203 — C-terminal domain; functional role in BBSome-cargo handoff (Thr203Asn hypomorphic allele)",
            "Key alias":         "BBIP10 (BBSome-Interacting Protein 10), BBS18",
        },
        "disease_card": {
            "Disease":           "Bardet-Biedl Syndrome Type 18",
            "OMIM disease":      "#615994",
            "Mechanism":         "BBSome assembly failure — BBIP1 bridges BBS9 to BBS2/BBS7; LOF destabilises octamer; no functional BBSome in cilia",
            "Cilia morphology":  "Full length present — ciliogenesis intact (BBIP1 not required for axonemal growth)",
            "BBSome IF":         "ALL BBSome subunits (BBS2/4/5/7/8/9) ABSENT/severely reduced from cilia — assembly failure; no functional octamer enters",
            "MKKS/BCC IF":       "NORMAL — upstream chaperonin intact; individual subunits fold but cannot form octamer",
            "MKS1/TZ IF":        "NORMAL — transition-zone Y-link scaffold intact",
            "IFT":               "NORMAL — IFT-A and IFT-B axonemal transport intact",
            "Renal type":        "Structural/cystic ~30% (NOT NPHP-dominant — contrast BBS13/14/16)",
            "Retinal dystrophy": "Rod-cone; rod-FIRST scotopic extinguished",
            "Key distinction":   "BBS18 vs BBS17 vs BBS15: assembly failure vs entry failure vs exit failure; all show absent ciliary BBSome, different upstream lesion",
            "Allele-class":      "Single disease tier — no NPHP/MKS/JBTS equivalent (all alleles BBS18 only)",
            "Inheritance":       "Autosomal recessive biallelic LOF",
            "Prevalence":        "~1/3,000,000–5,000,000; 10–20 families worldwide 2026 (rarest BBS type)",
            "Tri-allelic":       "~2%",
        },
        "key_variants": [
            {
                "variant":     "Leu150del (c.448_450delCTT)",
                "domain":      "Coiled-coil hydrophobic core (aa 150)",
                "consequence": "In-frame 1-residue deletion; disrupts coiled-coil register between BBIP1 and BBS9 CC interface; BBSome assembly abolished; MENA consanguineous enrichment (first reported BBS18 allele — Scheidecker 2014)",
                "ethnicity":   "MENA",
            },
            {
                "variant":     "Gln36Ter (c.106C>T)",
                "domain":      "N-terminal BBS1 contact segment (aa 36)",
                "consequence": "Truncating null; ablates entire CC domain and C-terminal domain; most common null allele in European families; complete BBSome assembly loss",
                "ethnicity":   "European",
            },
            {
                "variant":     "c.64+2T>C (splice-donor intron 2)",
                "domain":      "Splice-donor intron 2 (near aa 21–22 junction)",
                "consequence": "Abolishes splice-donor; intron 2 retention causes frameshift and premature termination; null allele; Middle Eastern consanguineous",
                "ethnicity":   "Middle Eastern / MENA",
            },
            {
                "variant":     "Pro57Ser (c.169C>T)",
                "domain":      "Coiled-coil — BBS9 primary interface (aa 57)",
                "consequence": "Hypomorphic missense; Pro→Ser disrupts CC register at BBS9 contact surface; partial BBSome assembly; residual LepR ciliary trafficking; European compound-het",
                "ethnicity":   "European",
            },
            {
                "variant":     "Arg211Ter (c.631C>T)",
                "domain":      "C-terminal domain — BBS2/BBS7 lid interface (aa 211)",
                "consequence": "Truncating null; deletes BBS2/BBS7 interaction surface; BBSome lid cannot close; null allele; Pan-ethnic",
                "ethnicity":   "Pan-ethnic",
            },
            {
                "variant":     "Thr203Asn (c.608C>A)",
                "domain":      "C-terminal domain adjacent to Thr203 phospho-site (aa 203)",
                "consequence": "Hypomorphic; Thr→Asn adjacent to phosphorylation site; attenuated cargo handoff; partial phenotype; MENA/South Asian",
                "ethnicity":   "MENA / South Asian",
            },
        ],
        "diagnostic_workup": [
            "1. BBS 24-gene panel (Invitae/GeneDx) — BBIP1 must be included; single-gene testing not recommended given BBS heterogeneity; BBS18 often missed on older smaller panels.",
            "2. Full ophthalmic evaluation: best-corrected VA, fundus photography, full-field ERG (rod-first scotopic loss), OCT (photoreceptor layer thinning).",
            "3. Renal ultrasound: structural anomalies / early cysts; serum creatinine, eGFR, urinalysis (proteinuria — tubular, less frequent than BBS13/14/16).",
            "4. Metabolic panel: BMI centiles, fasting glucose/HbA1c (BBS obesity risk), leptin level (BBS18-specific LepR ciliary trafficking failure).",
            "5. IF/confocal on fibroblasts: anti-BBS2 and anti-BBS9 (all absent/reduced from cilia — assembly failure) + acetylated α-tubulin (cilia present, full length); BBIP1 IF absent.",
            "6. BBSome assembly assay (research): immunoprecipitation of BBS9 — in BBS18 fibroblasts, BBS9 IP fails to co-pull BBS2/BBS7/BBS8 (confirming octamer collapse); distinguishes from BBS17 (where IP is intact).",
            "7. Neuropsychological assessment: IQ, adaptive behaviour, speech/language — LD reported in ~35%.",
            "8. Cardiac echo: CHD rare (~4%) but screen in paediatric BBS.",
            "9. Olfactory testing (UPSIT or Sniffin Sticks) — anosmia ~40%; baseline and annual.",
            "10. Endocrine: gonadotrophin axis (LH, FSH, testosterone/oestradiol), bone age — hypogonadism ~45%.",
        ],
        "treatment_summary": [
            "1. Retinal: low-vision aids; avoid high-oxygen environments; vitamin A supplementation (per ophthalmologist guidance — evidence limited in BBS); no AAV-BBIP1 gene therapy trials reported as of 2026 (too few patients).",
            "2. Obesity: early dietary counselling; structured exercise programme; semaglutide/liraglutide used off-label in BBS obesity (multi-BBS experience); LepR ciliary trafficking failure is the molecular basis — GLP-1 agonists act downstream of LepR, retaining some efficacy.",
            "3. Renal: ACE inhibitor / ARB for proteinuria; nephrology referral at eGFR <60; renal transplant effective (no disease recurrence in allograft).",
            "4. Polydactyly: orthopaedic surgery in neonatal period (post-axial digital tags vs metatarsal polydactyly — timing per surgical assessment).",
            "5. Cognitive/LD: early intervention, IEP, speech therapy — evidence-based; ASD overlap documented in BBS cohort.",
            "6. Hypogonadism: sex-hormone replacement at puberty (testosterone/oestrogen-progesterone) per endocrinology.",
            "7. Research: BBSome reconstitution therapy (delivery of full-length BBIP1 by AAV/nanoparticle) is conceptually attractive but no IND/CTA filed as of 2026; pre-clinical zebrafish and mouse models show correction of cilia defects by BBIP1 overexpression.",
            "8. Genetic counselling: AR recurrence risk 25%; carrier testing for siblings; prenatal/PGT-M available; consanguinity counselling important given MENA/North African enrichment.",
            "9. Tri-allelic BBS (~2%): screen for a third pathogenic allele in a second BBS gene if phenotype is unexpectedly severe.",
            "10. Annual surveillance: ERG, OCT, eGFR, BMI, UPSIT, gonadotrophins — track BBS18 multi-system progression.",
        ],
    }
