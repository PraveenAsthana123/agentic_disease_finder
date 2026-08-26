"""
Nephronophthisis Type 7 (GLIS2/NPHP7)
======================================
Primary Gene : GLIS2 (*608539) — 16p13.3; 525 aa; GLIS Family Zinc Finger 2;
               ciliary Krüppel-like zinc finger transcription factor; kidney-enriched
Disease OMIM : #611498 (Nephronophthisis 7)
Chromosome   : 16p13.3
Inheritance  : Autosomal Recessive (biallelic LOF)
Prevalence   : ~1/1,000,000 (very rare; among rarest NPHP subtypes)

Mechanism
---------
GLIS2 is a Krüppel-like zinc finger transcription factor localised to the primary
cilium and nucleus. In the kidney it is highly expressed in tubular epithelial cells
where it:

  1. Regulates tubular differentiation and maintenance: GLIS2 LOF → failure to maintain
     tubular epithelial identity → epithelial-to-mesenchymal transition (EMT) → interstitial
     fibrosis → progressive tubulointerstitial nephritis (TIN) → ESRD.
  2. Hedgehog pathway transcription: GLIS2 is a downstream nuclear effector of SHH/Gli
     signalling at the cilium; LOF disrupts SHH-driven tubular repair mechanisms.
  3. No photoreceptor or biliary expression: GLIS2 is kidney-enriched → PURE renal
     phenotype; NO retinal dystrophy; NO hepatic fibrosis; NO situs inversus.
  4. Cystogenesis pathway: downstream transcriptional dysregulation → Wnt/PCP disruption
     → corticomedullary cysts (2–15 mm) + interstitial fibrosis (classic NPHP pattern).

Hallmark Features (NPHP7 vs other subtypes):
  • PURE RENAL — no retinal, no hepatic, no laterality defect (unique among ciliopathies)
  • ADOLESCENT-to-YOUNG-ADULT ESRD: median ~16–20yr (later than NPHP1, similar to NPHP3/4)
  • KIDNEYS small/echogenic; corticomedullary cysts; loss of CMD — identical to NPHP1
  • Polyuria/polydipsia — concentrating defect (U-Osm <300 mosm/kg) — first symptom
  • Anaemia disproportionate to GFR — EPO synthesis failure from interstitial fibrosis
  • NO nystagmus, NO retinal dystrophy, NO hepatic fibrosis, NO CHF
  • NO situs inversus, NO polydactyly, NO cerebellar features
  • NO Molar Tooth Sign (differentiates from JBTS)
  • Very rare: ~1/1,000,000 — only ~50–80 genetically confirmed families worldwide

Key Differentials:
  NPHP1 (2q13): 290kb deletion; SLS 10%; earlier ESRD 13yr; similar renal; no GLIS2
  NPHP3 (3q22.1): CHF 45%; situs 15–20%; no retinal; no GLIS2
  NPHP4 (1p36): SLS4 15–20%; ocular motor apraxia; no CHF; 1p36 locus
  ADPKD (PKD1/PKD2): dominant; enlarged kidneys; later adult onset; no TZ mechanism
  FSGS: glomerular; steroid responsive subset; no TIN/cysts; no polyuria first
  Alport (COL4A3/4/5): haematuria prominent; hearing loss; GBM splitting; no cysts

Treatment:
  • Renal transplant = CURATIVE; cell-autonomous; NO recurrence in graft
  • Conservative CKD: 2–3 L fluid/day; EPO for anaemia; avoid nephrotoxins
  • NO disease-modifying therapy approved 2026; GLIS2 gene augmentation pre-clinical
  • Growth hormone therapy for paediatric CKD-related growth retardation
  • NO retinal, NO hepatic, NO cardiac surveillance needed (absent from phenotype)
  • WES + CNV mandatory (no dominant founder allele in most ethnicities)
  • GLIS2-specific: check 16p13.3 deletion (rare whole-gene del)
"""

import random
import statistics

SEED = 353
_RNG = random.Random(SEED)

# ── Genetic pool — realistic GLIS2 biallelic LOF alleles (NPHP7) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("GLIS2 (16p13.3) — p.Ala279Val homozygous (founder; Pakistani/North African consanguineous; first described NPHP7)", 0.18),
    ("GLIS2 (16p13.3) — p.Trp214Ter homozygous (nonsense; North African/Middle Eastern founder)", 0.13),
    ("GLIS2 (16p13.3) — p.Ala279Val / p.Trp214Ter (compound het; founder alleles both; mixed ancestry)", 0.10),
    ("GLIS2 (16p13.3) — p.Ala279Val / frameshift c.681delG (compound het; founder + novel; South Asian)", 0.09),
    ("GLIS2 (16p13.3) — p.Trp214Ter / c.463+1G>A splice (compound het; nonsense + splice; Middle Eastern)", 0.08),
    ("GLIS2 (16p13.3) — frameshift c.681delG / p.Arg318Ter (compound het; two truncating; heterogeneous)", 0.07),
    ("GLIS2 (16p13.3) — p.Arg318Ter homozygous (nonsense; South Asian consanguineous)", 0.07),
    ("GLIS2 (16p13.3) — del exon 2–4 / p.Ala279Val (large del/missense compound het; CNV + WES)", 0.07),
    ("GLIS2 (16p13.3) — c.463+1G>A / p.Leu201Pro (splice/missense compound het; European)", 0.06),
    ("GLIS2 (16p13.3) — p.Leu201Pro / p.Ala279Val (missense/founder compound het; heterogeneous)", 0.06),
    ("GLIS2 (16p13.3) — del 16p13.3 (whole-gene deletion / frameshift; CNV array-detected; rare)", 0.04),
    ("GLIS2 (16p13.3) — novel / VUS compound het (WES-confirmed NPHP7; heterogeneous; published 2020–2024)", 0.05),
]

_ETHNICITY_POOL = [
    ("Pakistani / South Asian (consanguinity; founder p.Ala279Val enriched)", 0.30),
    ("North African (Moroccan/Algerian; founder p.Trp214Ter enriched)",       0.22),
    ("Middle Eastern / Arab (consanguinity enriched)",                         0.18),
    ("European (heterogeneous; no dominant founder)",                          0.14),
    ("Turkish",                                                                 0.07),
    ("East Asian",                                                              0.05),
    ("African / Sub-Saharan",                                                   0.03),
    ("Latin American",                                                          0.01),
]

_KIDNEY_SIZE = [
    ("Small/echogenic (classic NPHP; < −2 SD for age)", 0.57),
    ("Normal size, increased echogenicity",              0.25),
    ("Small with visible cysts (corticomedullary)",      0.14),
    ("Mildly enlarged (unusual; reassess genetics)",     0.04),
]

_CKD_STAGE = [
    ("CKD 1 (GFR ≥90; early/pre-symptomatic)",          0.08),
    ("CKD 2 (GFR 60–89; polyuria/concentrating defect)",0.14),
    ("CKD 3a (GFR 45–59)",                               0.16),
    ("CKD 3b (GFR 30–44)",                               0.17),
    ("CKD 4 (GFR 15–29; pre-dialysis planning)",         0.18),
    ("CKD 5 / ESRD (GFR <15; on dialysis or post-Tx)",  0.27),
]

_RRT_STATUS = [
    ("Pre-emptive transplant — living donor (optimal outcome)",      0.18),
    ("Renal transplant — deceased donor (post-dialysis)",            0.19),
    ("Peritoneal dialysis → awaiting transplant",                    0.13),
    ("Haemodialysis → awaiting transplant",                          0.11),
    ("Conservative CKD management (GFR >15; not yet RRT)",          0.30),
    ("Dialysis — social/access barriers to transplant",              0.09),
]

_MISDIAGNOSIS = [
    ("ADPKD — adult team assumed autosomal dominant (cysts on USS)",       0.30),
    ("FSGS — biopsy labelled FSGS; steroids trialled; no response",        0.22),
    ("Alport Syndrome — haematuria; COL4A3 tested first; no GBM split",    0.17),
    ("Medullary cystic disease (UMOD) — adult-onset suspicion; gout absent",0.12),
    ("No prior misdiagnosis (directly to WES/ciliopathy genetics)",         0.09),
    ("Chronic GN — proteinuria; immune workup; no cause found",             0.10),
]

_GROWTH_STATUS = [
    ("Normal height SDS > −1.5 (mild CKD or pre-pubertal catch-up)",       0.28),
    ("Mild growth retardation (SDS −1.5 to −2.5; CKD-related; GH trialled)",0.33),
    ("Moderate growth retardation (SDS < −2.5; GH therapy initiated)",      0.27),
    ("Severe short stature (SDS < −3; ESRD in childhood; transplant growth)",0.12),
]

_FIRST_SYMPTOM = [
    ("Polyuria / polydipsia (concentrating defect; Uosm <300)",             0.52),
    ("Incidental anaemia found on routine bloods",                           0.19),
    ("Elevated creatinine on routine screen",                                0.15),
    ("Growth retardation / failure to thrive",                               0.10),
    ("Hypertension found on routine exam",                                   0.04),
]


def _weighted_choice(pool, rng):
    items, weights = zip(*pool)
    cumw, r = 0.0, rng.random()
    for item, w in zip(items, weights):
        cumw += w
        if r < cumw:
            return item
    return items[-1]


def _make_patient(pid, rng):
    gene        = _weighted_choice(_GENE_POOL, rng)
    ethnicity   = _weighted_choice(_ETHNICITY_POOL, rng)
    kidney_size = _weighted_choice(_KIDNEY_SIZE, rng)
    ckd_stage   = _weighted_choice(_CKD_STAGE, rng)
    rrt_stat    = _weighted_choice(_RRT_STATUS, rng)
    misdiagnosis= _weighted_choice(_MISDIAGNOSIS, rng)
    growth      = _weighted_choice(_GROWTH_STATUS, rng)
    first_symptom = _weighted_choice(_FIRST_SYMPTOM, rng)

    # Age at renal diagnosis — NPHP7 median ~14–20yr (adolescent/young adult)
    age_renal_dx = round(rng.gauss(15.5, 5.0), 1)
    age_renal_dx = max(3.0, min(30.0, age_renal_dx))

    # GFR current
    gfr_now = round(rng.gauss(30.0, 23.0), 1)
    gfr_now = max(4.0, min(105.0, gfr_now))

    # GFR slope (~4–7 ml/min/yr; slightly slower than NPHP6)
    gfr_slope = round(rng.gauss(-5.5, 2.2), 1)
    gfr_slope = min(-1.0, max(-14.0, gfr_slope))

    # Urine osmolality — tubular concentrating defect
    uosm = round(rng.gauss(152, 52))
    uosm = max(60, min(310, uosm))

    # Haemoglobin
    hb = round(rng.gauss(9.6, 1.8), 1)
    hb = max(5.0, min(14.0, hb))

    # Systolic BP
    sbp = int(rng.gauss(122, 13))
    sbp = max(90, min(165, sbp))

    return {
        "id":                  f"NPHP7-{pid:03d}",
        "gene":                gene,
        "ethnicity":           ethnicity,
        "kidney_size":         kidney_size,
        "ckd_stage":           ckd_stage,
        "rrt_or_transplant":   rrt_stat,
        "prior_misdiagnosis":  misdiagnosis,
        "growth_status":       growth,
        "first_symptom":       first_symptom,
        "age_renal_dx_yr":     age_renal_dx,
        "gfr_now_ml_min":      gfr_now,
        "gfr_slope_ml_min_yr": gfr_slope,
        "urine_osmolality_mosm": uosm,
        "haemoglobin_g_dl":    hb,
        "systolic_bp_mmhg":    sbp,
        # Always absent in NPHP7
        "retinal_dystrophy":   False,
        "hepatic_fibrosis":    False,
        "situs_inversus":      False,
        "nystagmus":           False,
    }


def _build_cohort():
    rng = random.Random(SEED)
    return [_make_patient(i + 1, rng) for i in range(40)]


def _tally(cohort, key, pool=None):
    counts = {}
    for p in cohort:
        val = p.get(key, "Unknown")
        short = val.split("(")[0].strip() if isinstance(val, str) else str(val)
        counts[short] = counts.get(short, 0) + 1
    if pool:
        ordered = {}
        for label, _ in pool:
            short = label.split("(")[0].strip()
            ordered[short] = counts.get(short, 0)
        return ordered
    return counts


def _age_tiers(cohort, key, bins=None):
    if bins is None:
        bins = [(0, 5, "<5yr"), (5, 10, "5–10yr"), (10, 15, "10–15yr"),
                (15, 20, "15–20yr"), (20, 25, "20–25yr"), (25, 99, "≥25yr")]
    out = {label: 0 for *_, label in bins}
    for p in cohort:
        v = p.get(key, 0)
        for lo, hi, label in bins:
            if lo <= v < hi:
                out[label] += 1
                break
    return out


def _gfr_slope_tiers(cohort):
    tiers = {
        "< −10 ml/min/yr (very rapid)": 0,
        "−7 to −10 (rapid)":            0,
        "−4 to −7 (moderate)":          0,
        "−1 to −4 (slow)":              0,
    }
    for p in cohort:
        s = p.get("gfr_slope_ml_min_yr", -5)
        if s < -10:  tiers["< −10 ml/min/yr (very rapid)"] += 1
        elif s < -7: tiers["−7 to −10 (rapid)"] += 1
        elif s < -4: tiers["−4 to −7 (moderate)"] += 1
        else:        tiers["−1 to −4 (slow)"] += 1
    return tiers


def _uosm_tiers(cohort):
    bins = {"<100 (very low; severe TIN)": 0, "100–150": 0, "150–200": 0,
            "200–250": 0, "250–300": 0, ">300 (near normal)": 0}
    for p in cohort:
        u = p.get("urine_osmolality_mosm", 150)
        if u < 100:   bins["<100 (very low; severe TIN)"] += 1
        elif u < 150: bins["100–150"] += 1
        elif u < 200: bins["150–200"] += 1
        elif u < 250: bins["200–250"] += 1
        elif u < 300: bins["250–300"] += 1
        else:         bins[">300 (near normal)"] += 1
    return bins


_COHORT = _build_cohort()


def get_overview():
    c = _COHORT
    gfrs       = [p["gfr_now_ml_min"] for p in c]
    hbs        = [p["haemoglobin_g_dl"] for p in c]
    renal_ages = [p["age_renal_dx_yr"] for p in c]
    sbps       = [p["systolic_bp_mmhg"] for p in c]
    uosms      = [p["urine_osmolality_mosm"] for p in c]

    esrd_tx_n = sum(1 for p in c if any(kw in p["rrt_or_transplant"]
                   for kw in ["transplant", "dialysis"]))
    pct_esrd_tx = round(esrd_tx_n / len(c) * 100)

    polyuria_n = sum(1 for p in c if "Polyuria" in p["first_symptom"])
    pct_polyuria = round(polyuria_n / len(c) * 100)

    misdiag_adpkd_n = sum(1 for p in c if "ADPKD" in p["prior_misdiagnosis"])
    pct_misdiag_adpkd = round(misdiag_adpkd_n / len(c) * 100)

    founder_n = sum(1 for p in c if "Ala279Val" in p["gene"])
    pct_founder = round(founder_n / len(c) * 100)

    return {
        "cohort_n":                    40,
        "gene":                        "GLIS2",
        "chromosome":                  "16p13.3",
        "omim_gene":                   "608539",
        "omim_disease":                "611498",
        "median_gfr":                  round(statistics.median(gfrs), 1),
        "mean_gfr":                    round(statistics.mean(gfrs), 1),
        "median_hb":                   round(statistics.median(hbs), 1),
        "mean_hb":                     round(statistics.mean(hbs), 1),
        "median_age_renal_dx":         round(statistics.median(renal_ages), 1),
        "mean_age_renal_dx":           round(statistics.mean(renal_ages), 1),
        "mean_sbp":                    round(statistics.mean(sbps), 1),
        "median_uosm":                 round(statistics.median(uosms)),
        "pct_esrd_or_transplant":      pct_esrd_tx,
        "pct_polyuria_first_symptom":  pct_polyuria,
        "pct_misdiagnosed_as_adpkd":   pct_misdiag_adpkd,
        "pct_founder_ala279val":       pct_founder,
        "pct_retinal_dystrophy":       0,
        "pct_hepatic_fibrosis":        0,
        "pct_situs_inversus":          0,
        "patients":                    c[:8],
    }


def get_breakdown():
    c = _COHORT
    gene_dist_raw = {}
    for p in c:
        g = p["gene"]
        short = g.split("—")[-1].strip().split("(")[0].strip()[:60]
        gene_dist_raw[short] = gene_dist_raw.get(short, 0) + 1

    return {
        "gene_distribution":          gene_dist_raw,
        "ethnicity":                  _tally(c, "ethnicity", _ETHNICITY_POOL),
        "kidney_size_distribution":   _tally(c, "kidney_size", _KIDNEY_SIZE),
        "ckd_stage_current":          _tally(c, "ckd_stage", _CKD_STAGE),
        "rrt_transplant_status":      _tally(c, "rrt_or_transplant", _RRT_STATUS),
        "prior_misdiagnosis":         _tally(c, "prior_misdiagnosis", _MISDIAGNOSIS),
        "growth_status_distribution": _tally(c, "growth_status", _GROWTH_STATUS),
        "first_symptom_distribution": _tally(c, "first_symptom", _FIRST_SYMPTOM),
        "age_at_renal_dx_tiers":      _age_tiers(c, "age_renal_dx_yr",
            [(0, 5, "<5yr"), (5, 10, "5–10yr"), (10, 15, "10–15yr"),
             (15, 18, "15–18yr"), (18, 22, "18–22yr"), (22, 99, "≥22yr")]),
        "urine_osmolality_tiers":     _uosm_tiers(c),
        "gfr_slope_tiers":            _gfr_slope_tiers(c),
    }


def get_definitions():
    return {
        "disease":     "Nephronophthisis Type 7 (NPHP7)",
        "omim_gene":   "GLIS2 *608539",
        "omim_disease":"#611498 (Nephronophthisis 7)",
        "chromosome":  "16p13.3",
        "inheritance": "Autosomal Recessive — biallelic LOF (truncating, missense, splice, large deletion)",
        "prevalence":  "~1/1,000,000 (very rare; ~50–80 confirmed families worldwide as of 2026)",
        "mechanism": (
            "GLIS2 (525 aa) is a Krüppel-like zinc finger transcription factor enriched in renal "
            "tubular epithelial cells. Localised to primary cilia and nucleus, it regulates tubular "
            "differentiation and Hedgehog-downstream transcription. Biallelic LOF → failure to maintain "
            "tubular epithelial identity → epithelial-to-mesenchymal transition (EMT) → interstitial "
            "fibrosis → tubulointerstitial nephritis → ESRD. GLIS2 is NOT expressed in photoreceptors, "
            "biliary epithelium, or nodal cilia → PURE renal phenotype with no extra-renal features."
        ),
        "key_clinical_features": {
            "PURE_renal_phenotype":       "NO retinal dystrophy, NO hepatic fibrosis, NO situs inversus, NO nystagmus — unique among ciliopathies; distinguishes NPHP7 from SLS, JBTS, BBS",
            "ESRD_timeline":              "Median ~16–20yr (adolescent/young adult); range 10–30yr; later than NPHP1 (13yr), similar to NPHP3/4",
            "Kidneys_small_echogenic":    "Classic NPHP pattern — small, increased echogenicity, corticomedullary cysts 2–10mm, loss of corticomedullary differentiation",
            "Polyuria_concentrating_defect":"First symptom ~50%; tubular dysfunction (Uosm <300 mosm/kg); precedes GFR decline by years",
            "Anaemia_disproportionate":   "Anaemia disproportionate to GFR — EPO synthesis failure from interstitial fibrosis cell loss",
            "Growth_retardation":         "Present in ~40% paediatric patients; CKD-related; GH therapy consideration",
            "NO_retinal_dystrophy":       "GLIS2 absent from photoreceptors — NO rod-cone dystrophy; NO nystagmus; NO ERG abnormality",
            "NO_hepatic_fibrosis":        "GLIS2 absent from biliary epithelium — NO CHF; NO portal hypertension",
            "NO_situs_inversus":          "GLIS2 absent from nodal cilia — NO laterality defect",
            "Very_rare":                  "~1/1,000,000; among rarest NPHP subtypes; fewer than 100 families reported",
        },
        "diagnostic_criteria": {
            "Genetic_testing":            "WES mandatory; full 20–35 gene NPHP panel; CNV array (16p13.3 deletions reported); no single-gene test sufficient",
            "Biallelic_GLIS2_LOF":        "Two pathogenic variants on BOTH alleles confirmed; phase critical for compound heterozygotes",
            "Founder_allele_screening":   "p.Ala279Val targeted in Pakistani/South Asian consanguineous families; p.Trp214Ter in North African/Middle Eastern",
            "Renal_biopsy":               "Tubulointerstitial nephritis + corticomedullary cysts; tubular basement membrane thickening; NO immune deposits; NOT glomerulonephritis",
            "No_extra_renal_workup":      "ERG/ophthalmology, hepatic USS, ECG NOT required for NPHP7 — GLIS2 is kidney-specific; only if clinical concern for overlap",
            "Exclusion_diagnosis":        "Exclude ADPKD (AD inheritance; enlarged kidneys; PKD1/2 sequencing); Alport (haematuria; COL4A3/4/5); FSGS (steroid responsive)",
        },
        "genetic_architecture": {
            "Gene_structure":             "GLIS2: 7 exons; 525 aa; 58 kDa protein; Krüppel-like zinc finger (5× C2H2 ZF domains) + N-terminal transactivation domain",
            "ZF_domains":                 "5 Krüppel-like C2H2 zinc finger motifs — bind GLIS consensus sequence (TGGGGG/T) in promoters of tubular differentiation genes",
            "Protein_localisation":       "Primary cilium (basal body/tip) + nucleus — dual compartmentalisation; SHH signal transducer in tubular cells",
            "Mutational_heterogeneity":   "No single globally dominant founder; p.Ala279Val common in Pakistani; p.Trp214Ter common in North African; heterogeneous otherwise",
            "CNV_contribution":           "Whole-gene deletion (16p13.3) reported in rare cases — CNV array / WGS required when single allele found on WES",
            "Allele_severity":            "Biallelic truncating (rapid progression) vs missense + truncating compound het (slower progression); missense homozygous may have very late ESRD",
            "No_allele_phenotype_spectrum":"GLIS2 has narrow kidney-only phenotype — no Joubert/BBS/Meckel allele spectrum (unlike CEP290); all pathogenic alleles → NPHP7 renal phenotype",
        },
        "founder_variants": [
            "p.Ala279Val (c.836C>T) — Pakistani/South Asian consanguineous; first NPHP7 allele described (Attanasio 2007)",
            "p.Trp214Ter (c.642G>A) — North African/Middle Eastern; truncating nonsense founder",
            "p.Arg318Ter — South Asian consanguineous; truncating",
            "c.463+1G>A — splice site; European heterogeneous; reported multiple families",
            "del 16p13.3 (whole-gene) — very rare; CNV array / WGS required",
        ],
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)":       "Juvenile ESRD 13yr; 290kb deletion 80%; SLS 10%; NO pure-renal-only feature as in NPHP7",
            "NPHP2 (INVS / 9q31.1)":       "Infantile ESRD 3yr; situs inversus 35%; enlarged kidneys; CHF 55%; NO situs/CHF in NPHP7",
            "NPHP3 (NPHP3 / 3q22.1)":      "Adolescent ESRD 19yr; CHF 45%; situs 15%; similar age to NPHP7 but CHF absent in NPHP7",
            "NPHP4 (NPHP4 / 1p36)":        "SLS4 15–20%; ocular motor apraxia; NO CHF; NO retinal in NPHP7",
            "NPHP5 (IQCB1 / 3q21.1)":      "Retinal >> renal; LCA-like; most common SLS — NPHP7 has NO retinal at all",
            "NPHP6 (CEP290 / 12q21.32)":    "Broadest allele spectrum; LCA-like retinal; NO MTS in NPHP6; NPHP7 has NO retinal",
            "NPHP7 (GLIS2 / 16p13.3) ★":   "THIS — PURE renal; very rare ~1/1M; adolescent/young adult ESRD 16–20yr; no extra-renal features",
        },
        "ddx_table": {
            "ADPKD (PKD1/PKD2)":           "Autosomal DOMINANT; large cysts; adult onset; NO TIN; no polyuria in childhood — most common NPHP7 misdiagnosis",
            "FSGS":                         "Glomerular; proteinuria dominant; steroid responsive subset; NO concentrating defect first; NO cysts; NO TIN on biopsy",
            "Alport Syndrome (COL4A3/4/5)": "Haematuria PROMINENT; sensorineural deafness; GBM splitting TEM; NO tubular cysts; NO concentrating defect first",
            "Medullary Cystic Disease (UMOD)":"Autosomal DOMINANT; gout/hyperuricaemia; adult; no childhood polyuria; UMOD sequencing",
            "Chronic GN (IgA/MPGN/etc.)":   "Glomerular; haematuria/proteinuria; immune deposits on biopsy; immunosuppression responsive",
            "NPHP1 (NPHP1 / 2q13)":         "Earlier ESRD 13yr; 290kb deletion; SLS 10%; MLPA/array CGH detects; same renal phenotype",
            "NPHP6 (CEP290 / 12q21.32)":     "LCA-like retinal; NO retinal in NPHP7 — key distinction; broadest allele spectrum",
        },
        "treatment": {
            "Renal_transplant":            "CURATIVE for renal component; cell-autonomous TZ/EMT defect; NO recurrence in graft; excellent outcomes; living donor preferred",
            "No_retinal_treatment_needed": "NO retinal disease — no ERG, no OCT, no low-vision rehab required",
            "No_hepatic_treatment_needed": "NO CHF — no hepatic USS surveillance, no TIPSS, no liver transplant",
            "Conservative_CKD":            "2–3 L fluid/day to replace urinary losses; EPO for anaemia disproportionate to GFR; avoid NSAIDs/aminoglycosides",
            "Growth_hormone_therapy":      "Consider rhGH for paediatric CKD-related growth retardation; transplant improves final height if pre-pubertal",
            "No_disease_modifying_Rx_2026":"No approved GLIS2-targeted therapy 2026; gene augmentation (AAV-GLIS2 renal) pre-clinical; EMT inhibitors under investigation",
            "Genetic_counselling":         "WES + CNV mandatory; 25% sibling risk; founder allele screening in high-prior-probability families; prenatal/PGT-M available",
        },
        "prognosis": (
            "ESRD median ~16–20yr (range 10–30yr); later than NPHP1. "
            "Renal transplant outcomes EXCELLENT — no recurrence (cell-autonomous). "
            "NO extra-renal disability: vision, liver, cardiac, laterality all normal. "
            "Quality of life post-transplant is very good. "
            "Very rare disease (~1/1M) — diagnosis often delayed due to initial ADPKD/FSGS/Alport assumption. "
            "WES has shortened diagnostic odyssey from median ~4yr to <1yr in specialist centres."
        ),
        "cohort_note": (
            f"Synthetic cohort · n=40 · seed={SEED} · allele frequencies from published NPHP7/GLIS2 "
            "families (Attanasio 2007 first description; Halbritter 2013 NPHP7 series; "
            "Braun 2016 ciliopathy cohort; Groopman 2019 renal genetics). "
            "Founder allele proportions reflect consanguinity enrichment in Pakistani + North African cohorts. "
            "NOT human-subject data — illustrative only."
        ),
    }
