"""
Nephronophthisis Type 8 (RPGRIP1L/NPHP8) / Joubert Syndrome Type 7 (JBTS7) / Meckel Type 5 (MKS5)
=====================================================================================================
Primary Gene : RPGRIP1L (*610937) — 16q12.2; 1315 aa; Retinitis Pigmentosa GTPase Regulator
               Interacting Protein-1 Like; TZ scaffold; coiled-coil + C2 + PH-like domains;
               directly binds NPHP4 at the transition zone
Disease OMIM : #613237 (Nephronophthisis 8)
               Also JBTS7 (#611560) and MKS5 (#611561) — broadest allele-phenotype spectrum
Chromosome   : 16q12.2
Inheritance  : Autosomal Recessive (biallelic LOF)
Prevalence   : ~1/200,000–500,000 (NPHP8 subtype); JBTS7 similar; MKS5 very rare

Mechanism
---------
RPGRIP1L (1315 aa) is a TZ scaffold protein with:
  - N-terminal coiled-coil domains (homodimerisation + NPHP4 interaction)
  - Central C2 domain (phospholipid membrane targeting at TZ)
  - C-terminal RPGR-interacting domain (RID; connecting cilium → photoreceptor CC)
  - Pleckstrin homology-like domain (PH; PI-3-kinase interaction)

At the transition zone:
  1. RPGRIP1L binds NPHP4 forming the NPHP1-4-8 ternary TZ module → diffusion barrier
  2. LOF → TZ gate collapse → Hh/Wnt/PDGF signalling failure → tubulointerstitial nephritis
  3. In photoreceptor connecting cilium: RPGRIP1L anchors RPGR → CC integrity → rod/cone survival
  4. In cerebellar neurones: RPGRIP1L required for primary cilia → SHH→ Purkinje axon guidance
  5. In biliary epithelium: expressed in cholangiocytes → ductal plate development → CHF risk

ALLELE-PHENOTYPE SPECTRUM (NPHP8 — broadest NPHP subtype after CEP290):
  Biallelic null (truncating + truncating)  → MKS5 (lethal; occipital encephalocele; Meckel)
  Truncating + strong missense (JBTS)       → JBTS7 (Molar Tooth Sign; renal; retinal ±; hepatic ±)
  Truncating + mild missense (NPHP)         → NPHP8 (pure/semi-renal; no MTS or mild MTS)
  Biallelic mild missense                   → NPHP8 (pure renal; later ESRD)
  RPGRIP1L mono + NPHP4 mono (digenic)      → JBTS (compound heterozygosity across loci — key DDx)

Hallmark Features (NPHP8 vs other subtypes):
  • BROAD ALLELE-PHENOTYPE SPECTRUM — MKS5/JBTS7/NPHP8 from same gene (like CEP290)
  • MOLAR TOOTH SIGN (MTS) in JBTS7 subtype — brain MRI MANDATORY to stratify allele class
  • RETINAL DYSTROPHY 25–35% (JBTS7 alleles) — rod-cone dystrophy; LCA-like in severe alleles
  • HEPATIC FIBROSIS 15–20% (JBTS7 alleles) — CHF; ductal plate; portal HTN risk
  • NO SITUS INVERSUS — RPGRIP1L not expressed nodal cilia (unlike NPHP2/3)
  • NPHP8 pure renal: ESRD median ~15–18yr; kidneys small/echogenic; corticomedullary cysts
  • DIGENIC KEY FEATURE: RPGRIP1L monoallelic + NPHP4 monoallelic → Joubert (not NPHP)
  • Male infertility in subset (RPGRIP1L expressed sperm connecting piece / flagella)

Key Differentials:
  NPHP7 (GLIS2 / 16p13.3): pure renal; NO MTS; NO retinal; GLIS2 kidney-only; older ESRD 16–20yr
  NPHP6 (CEP290 / 12q21.32): broadest spectrum; IVS26 → LCA10 only; NPHP6 truncating → NPHP/JBTS/MKS
  JOUBERT (CEP290/AHI1/INPP5E/etc.): MTS present; no RPGRIP1L → digenic excluded
  NPHP4 (NPHP4 / 1p36): TZ module partner; SLS4; ocular motor apraxia; RPGRIP1L digenic overlap
  MECKEL (MKS1/MKS3/etc.): lethal; occipital encephalocele; RPGRIP1L-MKS5 allele class
  BBS (BBS1/BBS10/etc.): obesity; polydactyly; RP; NO TIN cysts; NO MTS usually

Treatment:
  • Renal transplant = CURATIVE for renal component; cell-autonomous; NO recurrence
  • NO retinal improvement post-transplant — cell-autonomous CC defect
  • Low-vision rehabilitation if JBTS7 subtype with retinal dystrophy
  • MTS/JBTS7: neurodevelopmental surveillance; OT/PT/speech; breathing dysrhythmia self-resolves
  • Hepatic surveillance if CHF subtype: annual USS, APRI; TIPSS/liver transplant if portal HTN
  • WES + CNV mandatory; NPHP4 allele status always checked (digenic Joubert)
  • NO disease-modifying therapy 2026; RPGRIP1L gene therapy pre-clinical
"""

import random
import statistics

SEED = 355
_RNG = random.Random(SEED)

# ── Genetic pool — realistic RPGRIP1L biallelic LOF alleles (NPHP8) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("RPGRIP1L (16q12.2) — p.Arg826Ter / missense compound het (truncating + mild missense; pure NPHP8 renal)", 0.16),
    ("RPGRIP1L (16q12.2) — p.Arg826Ter homozygous (truncating; pure NPHP8; consanguineous)", 0.10),
    ("RPGRIP1L (16q12.2) — p.Leu1048Pro / p.Arg826Ter (JBTS7 allele compound het; MTS present)", 0.12),
    ("RPGRIP1L (16q12.2) — p.Ala229Val / p.Arg826Ter (mild missense/truncating; European pure NPHP8)", 0.09),
    ("RPGRIP1L (16q12.2) — exon 14–17 del / p.Leu1048Pro (CNV + JBTS7 missense; MTS subtype)", 0.08),
    ("RPGRIP1L (16q12.2) — p.Glu1243Lys homozygous (Amish/consanguineous founder; JBTS7)", 0.07),
    ("RPGRIP1L (16q12.2) — c.1239+2T>C / p.Arg826Ter (splice + truncating; pure NPHP8)", 0.07),
    ("RPGRIP1L (16q12.2) — p.Thr1196Pro / p.Arg826Ter (missense/truncating compound het; renal + mild retinal)", 0.07),
    ("RPGRIP1L (16q12.2) — p.Leu1048Pro homozygous (JBTS7; consanguineous Middle Eastern; MTS + renal)", 0.06),
    ("RPGRIP1L (16q12.2) — exon 1–4 del / frameshift c.249delA (biallelic null; MKS5-like; lethal not represented)", 0.03),
    ("RPGRIP1L (16q12.2) — p.Arg826Ter / c.1239+2T>C (two truncating; severe; JBTS borderline)", 0.05),
    ("RPGRIP1L (16q12.2) — novel VUS compound het WES-confirmed NPHP8 (heterogeneous; published 2020–2025)", 0.10),
]

_ETHNICITY_POOL = [
    ("European (heterogeneous; no dominant founder)",                           0.30),
    ("Middle Eastern / Arab (consanguinity; p.Leu1048Pro enriched)",           0.22),
    ("North African (Moroccan/Algerian; consanguineous)",                       0.16),
    ("South Asian / Pakistani (consanguinity)",                                  0.12),
    ("Amish / Old Order (p.Glu1243Lys founder; JBTS7 subtype)",                0.06),
    ("Turkish",                                                                  0.06),
    ("East Asian",                                                               0.05),
    ("African / Sub-Saharan",                                                    0.03),
]

_ALLELE_CLASS = [
    ("NPHP8 pure renal (no MTS; small kidneys; TIN; truncating + mild missense)", 0.50),
    ("JBTS7 semi-renal + MTS (Molar Tooth Sign; ± retinal; ± CHF; truncating + strong missense)", 0.40),
    ("JBTS7 severe + MTS + retinal + CHF (biallelic strong; RPGRIP1L + NPHP4 digenic excluded)", 0.10),
]

_KIDNEY_SIZE = [
    ("Small/echogenic (classic NPHP; < −2 SD for age)", 0.55),
    ("Normal size, increased echogenicity",              0.24),
    ("Small with visible cysts (corticomedullary)",      0.16),
    ("Mildly enlarged (unusual; reassess genetics)",     0.05),
]

_CKD_STAGE = [
    ("CKD 1 (GFR ≥90; early/pre-symptomatic)",             0.07),
    ("CKD 2 (GFR 60–89; polyuria/concentrating defect)",   0.14),
    ("CKD 3a (GFR 45–59)",                                  0.15),
    ("CKD 3b (GFR 30–44)",                                  0.17),
    ("CKD 4 (GFR 15–29; approaching transplant listing)",   0.19),
    ("CKD 5 pre-dialysis (GFR <15; imminent ESRD)",         0.09),
    ("Haemodialysis (ESRD; awaiting transplant)",            0.08),
    ("Peritoneal dialysis (ESRD; home therapy)",             0.05),
    ("Post-renal transplant (functioning graft; best outcome)", 0.06),
]

_RRT_STATUS = [
    ("Pre-ESRD (CKD 1–4; conservative management)",              0.52),
    ("On haemodialysis (centre-based; ESRD)",                     0.13),
    ("On peritoneal dialysis (home; ESRD)",                        0.07),
    ("Living donor renal transplant (functioning graft; CURATIVE)", 0.19),
    ("Deceased donor renal transplant (functioning graft)",         0.09),
]

_MISDIAGNOSIS = [
    ("ADPKD (PKD1/PKD2 — autosomal dominant assumed; most common error)",       0.31),
    ("Joubert syndrome (MTS without genetics; CEP290/AHI1 tested first)",        0.15),
    ("FSGS (glomerular biopsy — TIN labelled FSGS; steroids trialled)",          0.14),
    ("Alport Syndrome (haematuria; COL4A3 sequenced first)",                     0.09),
    ("LCA (Leber congenital amaurosis — retinal-only workup; JBTS7 subtype)",    0.08),
    ("Senior-Løken Syndrome (retinal + renal; IQCB1 tested first)",              0.07),
    ("Medullary cystic disease (UMOD tested in adult patient)",                   0.05),
    ("No prior misdiagnosis (direct genetic referral; specialist centre)",        0.11),
]

_GROWTH_STATUS = [
    ("Normal growth (height WNL for age)",                                       0.38),
    ("Mild growth retardation (−1 to −2 SD; CKD-related; GH not yet started)",  0.32),
    ("Moderate growth retardation (< −2 SD; GH therapy considered)",             0.20),
    ("Severe growth retardation (< −3 SD; GH started; renal transplant planned)",0.10),
]

_FIRST_SYMPTOM = [
    ("Polyuria / polydipsia / nocturia (tubular concentrating defect; first symptom)", 0.42),
    ("Anaemia (incidental CKD pickup; disproportionate to GFR)",                       0.18),
    ("Abnormal renal USS (corticomedullary cysts; echogenic; incidental)",             0.12),
    ("Elevated creatinine (incidental school/insurance/sports screening)",             0.10),
    ("Visual symptoms — nystagmus / reduced acuity (JBTS7 allele; retinal)",           0.09),
    ("Developmental delay / hypotonia (JBTS7 MTS subtype; early neurology referral)", 0.06),
    ("Abnormal breathing pattern neonatal (JBTS7; episodic hyperpnoea; self-resolved)",0.03),
]

_RETINAL_STATUS = [
    ("No retinal disease (NPHP8 pure allele class)", 0.65),
    ("Rod-cone dystrophy — moderate (JBTS7 allele; ERG reduced; fundus normal early)", 0.18),
    ("LCA-like severe retinal dystrophy (JBTS7 allele; flat ERG; nystagmus; visual impairment infancy)", 0.12),
    ("Mild retinal changes only (VEP delay; fundus pigment; JBTS7 mild allele)", 0.05),
]

_HEPATIC_STATUS = [
    ("No hepatic fibrosis (NPHP8 pure allele class)",                          0.78),
    ("Mild CHF (enlarged porta hepatis; APRI <1.0; USS only; JBTS7 allele)",   0.12),
    ("Moderate CHF (portal hypertension; APRI >1.0; varices risk; JBTS7)",     0.07),
    ("Severe CHF (portal HTN; varices; TIPSS considered; JBTS7 severe allele)",0.03),
]

_MTS_STATUS = [
    ("No Molar Tooth Sign (pure NPHP8; brain MRI normal cerebellum)",          0.55),
    ("Molar Tooth Sign present — JBTS7 (SCP elongation; vermis hypo; MRI)",    0.40),
    ("Borderline MTS (subtle SCP; radiologist uncertainty; repeat MRI)",        0.05),
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
    gene          = _weighted_choice(_GENE_POOL, rng)
    ethnicity     = _weighted_choice(_ETHNICITY_POOL, rng)
    allele_class  = _weighted_choice(_ALLELE_CLASS, rng)
    kidney_size   = _weighted_choice(_KIDNEY_SIZE, rng)
    ckd_stage     = _weighted_choice(_CKD_STAGE, rng)
    rrt_stat      = _weighted_choice(_RRT_STATUS, rng)
    misdiagnosis  = _weighted_choice(_MISDIAGNOSIS, rng)
    growth        = _weighted_choice(_GROWTH_STATUS, rng)
    first_symptom = _weighted_choice(_FIRST_SYMPTOM, rng)
    retinal       = _weighted_choice(_RETINAL_STATUS, rng)
    hepatic       = _weighted_choice(_HEPATIC_STATUS, rng)
    mts           = _weighted_choice(_MTS_STATUS, rng)

    # Age at renal diagnosis — NPHP8 median ~13–18yr (adolescent; JBTS7 earlier if neuro)
    age_renal_dx = round(rng.gauss(14.5, 5.5), 1)
    age_renal_dx = max(2.0, min(32.0, age_renal_dx))

    # GFR current
    gfr_now = round(rng.gauss(32.0, 24.0), 1)
    gfr_now = max(4.0, min(108.0, gfr_now))

    # GFR slope (~4–8 ml/min/yr for NPHP8; more variable with JBTS7 allele)
    gfr_slope = round(rng.gauss(-5.8, 2.5), 1)
    gfr_slope = min(-1.0, max(-15.0, gfr_slope))

    # Urine osmolality — tubular concentrating defect
    uosm = round(rng.gauss(148, 54))
    uosm = max(60, min(310, uosm))

    # Haemoglobin
    hb = round(rng.gauss(9.5, 1.9), 1)
    hb = max(5.0, min(14.5, hb))

    # Systolic BP
    sbp = int(rng.gauss(123, 14))
    sbp = max(88, min(170, sbp))

    has_retinal  = "No retinal" not in retinal
    has_hepatic  = "No hepatic" not in hepatic
    has_mts      = "No Molar Tooth" not in mts
    has_nystagmus = has_retinal or rng.random() < 0.05

    return {
        "id":                    f"NPHP8-{pid:03d}",
        "gene":                  gene,
        "ethnicity":             ethnicity,
        "allele_class":          allele_class,
        "kidney_size":           kidney_size,
        "ckd_stage":             ckd_stage,
        "rrt_or_transplant":     rrt_stat,
        "prior_misdiagnosis":    misdiagnosis,
        "growth_status":         growth,
        "first_symptom":         first_symptom,
        "retinal_status":        retinal,
        "hepatic_status":        hepatic,
        "mts_status":            mts,
        "age_renal_dx_yr":       age_renal_dx,
        "gfr_now_ml_min":        gfr_now,
        "gfr_slope_ml_min_yr":   gfr_slope,
        "urine_osmolality_mosm": uosm,
        "haemoglobin_g_dl":      hb,
        "systolic_bp_mmhg":      sbp,
        # Derived booleans
        "retinal_dystrophy":     has_retinal,
        "hepatic_fibrosis":      has_hepatic,
        "molar_tooth_sign":      has_mts,
        "situs_inversus":        False,   # RPGRIP1L not nodal
        "nystagmus":             has_nystagmus,
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
    gfrs        = [p["gfr_now_ml_min"] for p in c]
    hbs         = [p["haemoglobin_g_dl"] for p in c]
    renal_ages  = [p["age_renal_dx_yr"] for p in c]
    sbps        = [p["systolic_bp_mmhg"] for p in c]
    uosms       = [p["urine_osmolality_mosm"] for p in c]

    esrd_tx_n = sum(1 for p in c if any(kw in p["rrt_or_transplant"]
                    for kw in ["transplant", "dialysis"]))
    pct_esrd_tx = round(esrd_tx_n / len(c) * 100)

    polyuria_n   = sum(1 for p in c if "Polyuria" in p["first_symptom"])
    pct_polyuria = round(polyuria_n / len(c) * 100)

    misdiag_adpkd_n    = sum(1 for p in c if "ADPKD" in p["prior_misdiagnosis"])
    pct_misdiag_adpkd  = round(misdiag_adpkd_n / len(c) * 100)

    pct_retinal    = round(sum(1 for p in c if p["retinal_dystrophy"])   / len(c) * 100)
    pct_hepatic    = round(sum(1 for p in c if p["hepatic_fibrosis"])    / len(c) * 100)
    pct_mts        = round(sum(1 for p in c if p["molar_tooth_sign"])    / len(c) * 100)
    pct_nystagmus  = round(sum(1 for p in c if p["nystagmus"])           / len(c) * 100)

    jbts7_n   = sum(1 for p in c if "JBTS7" in p["allele_class"] or "MTS" in p["allele_class"])
    pct_jbts7 = round(jbts7_n / len(c) * 100)

    return {
        "cohort_n":                    40,
        "gene":                        "RPGRIP1L",
        "chromosome":                  "16q12.2",
        "omim_gene":                   "610937",
        "omim_disease":                "613237",
        "also_known_as":               "JBTS7 (#611560) / MKS5 (#611561)",
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
        "pct_retinal_dystrophy":       pct_retinal,
        "pct_hepatic_fibrosis":        pct_hepatic,
        "pct_molar_tooth_sign":        pct_mts,
        "pct_nystagmus":               pct_nystagmus,
        "pct_jbts7_allele_class":      pct_jbts7,
        "pct_situs_inversus":          0,
        "patients":                    c[:8],
    }


def get_breakdown():
    c = _COHORT
    gene_dist_raw = {}
    for p in c:
        g = p["gene"]
        short = g.split("—")[-1].strip().split("(")[0].strip()[:65]
        gene_dist_raw[short] = gene_dist_raw.get(short, 0) + 1

    return {
        "gene_distribution":           gene_dist_raw,
        "allele_class_distribution":   _tally(c, "allele_class", _ALLELE_CLASS),
        "ethnicity":                   _tally(c, "ethnicity", _ETHNICITY_POOL),
        "kidney_size_distribution":    _tally(c, "kidney_size", _KIDNEY_SIZE),
        "ckd_stage_current":           _tally(c, "ckd_stage", _CKD_STAGE),
        "rrt_transplant_status":       _tally(c, "rrt_or_transplant", _RRT_STATUS),
        "prior_misdiagnosis":          _tally(c, "prior_misdiagnosis", _MISDIAGNOSIS),
        "growth_status_distribution":  _tally(c, "growth_status", _GROWTH_STATUS),
        "first_symptom_distribution":  _tally(c, "first_symptom", _FIRST_SYMPTOM),
        "retinal_status_distribution": _tally(c, "retinal_status", _RETINAL_STATUS),
        "hepatic_status_distribution": _tally(c, "hepatic_status", _HEPATIC_STATUS),
        "mts_status_distribution":     _tally(c, "mts_status", _MTS_STATUS),
        "age_at_renal_dx_tiers":       _age_tiers(c, "age_renal_dx_yr",
            [(0, 5, "<5yr"), (5, 10, "5–10yr"), (10, 15, "10–15yr"),
             (15, 18, "15–18yr"), (18, 22, "18–22yr"), (22, 99, "≥22yr")]),
        "urine_osmolality_tiers":      _uosm_tiers(c),
        "gfr_slope_tiers":             _gfr_slope_tiers(c),
    }


def get_definitions():
    return {
        "disease":      "Nephronophthisis Type 8 (NPHP8) / Joubert Syndrome Type 7 (JBTS7) / Meckel Syndrome Type 5 (MKS5)",
        "omim_gene":    "RPGRIP1L *610937",
        "omim_disease": "#613237 (Nephronophthisis 8) · also JBTS7 #611560 · MKS5 #611561",
        "chromosome":   "16q12.2",
        "inheritance":  "Autosomal Recessive — biallelic LOF (truncating, missense, splice, large deletion)",
        "prevalence":   "~1/200,000–500,000 (NPHP8 pure); JBTS7 similar; MKS5 rare; combined spectrum ~1/100,000–200,000",
        "mechanism": (
            "RPGRIP1L (1315 aa) is a TZ scaffold protein with coiled-coil, C2, and RPGR-interacting "
            "domains. It directly binds NPHP4 forming the NPHP1-4-8 ternary TZ module (diffusion barrier). "
            "LOF → TZ gate collapse → Hh/Wnt/PDGF signalling failure → tubulointerstitial nephritis → ESRD. "
            "In photoreceptors: RPGRIP1L anchors RPGR at the connecting cilium → rod/cone degeneration "
            "(JBTS7 alleles). In cerebellum: required for Purkinje/granule cell primary cilia → vermis "
            "hypoplasia → Molar Tooth Sign (JBTS7). In cholangiocytes: ductal plate development → CHF risk. "
            "ALLELE-PHENOTYPE SPECTRUM: biallelic null → MKS5 (lethal); truncating + strong missense → JBTS7; "
            "truncating + mild missense → NPHP8 pure renal."
        ),
        "key_clinical_features": {
            "Allele_phenotype_spectrum":  "NPHP8/JBTS7/MKS5 from same gene — allele class determines phenotype (like CEP290); brain MRI mandatory to stratify",
            "Molar_Tooth_Sign_JBTS7":     "40–45% of cohort; MTS on brain MRI = JBTS7 subtype; SCP elongation, vermis aplasia, interpeduncular fossa deepened",
            "ESRD_timeline":              "Median ~15–18yr (NPHP8 pure); earlier in JBTS7 subtype (~12–15yr); range 8–30yr",
            "Kidneys_small_echogenic":    "Classic NPHP pattern — small, echogenic, corticomedullary cysts; NOT enlarged (distinguishes from ARPKD/NPHP2)",
            "Retinal_dystrophy_JBTS7":    "25–35% cohort (JBTS7 allele class); rod-cone dystrophy; LCA-like in severe; NO retinal in NPHP8 pure",
            "Hepatic_fibrosis_JBTS7":     "15–20% cohort (JBTS7 allele class); CHF + ductal plate malformation; portal HTN risk; NOT in NPHP8 pure",
            "NO_situs_inversus":          "RPGRIP1L not expressed nodal cilia — NO laterality defect (differentiates from NPHP2/3)",
            "Digenic_Joubert_key":        "RPGRIP1L monoallelic + NPHP4 (1p36) monoallelic → Joubert — ALWAYS check NPHP4 status in every RPGRIP1L patient",
            "Polyuria_concentrating":     "First symptom ~42%; tubular dysfunction (Uosm <300); precedes GFR decline",
            "Male_infertility":           "Subset — RPGRIP1L expressed sperm connecting piece / flagella; asthenozoospermia",
        },
        "diagnostic_criteria": {
            "Brain_MRI_MANDATORY":        "First investigation after RPGRIP1L found — rule out MTS (JBTS7) vs pure NPHP8; MTS changes surveillance + counselling",
            "NPHP4_genotype_mandatory":   "ALWAYS sequence NPHP4 — digenic Joubert (RPGRIP1L mono + NPHP4 mono) causes full JBTS; panels must include both",
            "Genetic_testing":            "WES + 35-gene NPHP/JBTS panel + CNV array (16q12.2 deletions); RPGRIP1L + full NPHP panel mandatory",
            "Biallelic_RPGRIP1L_LOF":     "Two pathogenic variants on BOTH alleles confirmed; phase critical; check digenic JBTS (NPHP4) before assuming monoallelic",
            "Renal_biopsy":               "TIN + corticomedullary cysts; tubular BM thickening; NO immune deposits; NOT glomerulonephritis",
            "Ophthalmology_JBTS7":        "ERG + OCT + fundus photography if JBTS7 allele or MTS present; NOT required for pure NPHP8",
            "Hepatic_assessment":         "Liver USS + APRI if JBTS7 allele; NOT required for pure NPHP8",
        },
        "genetic_architecture": {
            "Gene_structure":            "RPGRIP1L: 27 exons; 1315 aa; ~147 kDa; coiled-coil (N-term) + C2 domain + RPGR-interacting domain (RID) + PH-like domain",
            "TZ_module":                 "NPHP1-4-8 ternary module: RPGRIP1L (NPHP8) binds NPHP4 directly; NPHP4 binds NPHP1; together form the Y-link scaffold",
            "Photoreceptor_CC":          "RPGRIP1L anchors RPGR at photoreceptor connecting cilium; RPGRIP1L-NPHP8 alleles → CC destabilisation → rod/cone degeneration",
            "Allele_severity":           "Biallelic null → MKS5 (lethal); truncating + strong missense → JBTS7; truncating + mild missense / biallelic missense → NPHP8",
            "Digenic_interaction":       "RPGRIP1L heterozygous + NPHP4 heterozygous → JBTS (triallelic/digenic model); mechanism: reduced TZ module below threshold",
            "CNV_contribution":          "Large exon deletions (16q12.2) reported; CNV array / WGS required when single allele found on WES",
            "No_single_dominant_founder":"No globally dominant founder allele; p.Glu1243Lys enriched in Amish; p.Leu1048Pro Middle Eastern JBTS7; heterogeneous globally",
        },
        "key_variants": [
            "p.Arg826Ter — truncating; pan-ethnic; most common NPHP8 truncating allele (pure renal when compound with mild missense)",
            "p.Leu1048Pro — missense RID domain; JBTS7 allele (MTS when compound with truncating; Middle Eastern/Amish)",
            "p.Ala229Val — mild missense; European; pure NPHP8 (later ESRD; no retinal)",
            "p.Glu1243Lys — missense; Amish founder; JBTS7 (MTS + renal; first published Joubert8 allele)",
            "c.1239+2T>C — splice donor; pan-ethnic truncating equivalent; pure NPHP8",
            "Exon 14–17 deletion — CNV; JBTS7 when compound with strong missense; 16q12.2 loss",
            "p.Thr1196Pro — PH-like domain missense; intermediate spectrum; renal + mild retinal",
        ],
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)":        "Juvenile ESRD 13yr; 290kb deletion; SLS 10%; NO allele-phenotype spectrum (always NPHP1); no JBTS",
            "NPHP2 (INVS / 9q31.1)":        "Infantile ESRD 3yr; situs inversus 35%; enlarged kidneys; CHF 55% — RPGRIP1L has NO situs inversus",
            "NPHP3 (NPHP3 / 3q22.1)":       "Adolescent ESRD 19yr; CHF 45%; situs 15%; NO MTS/JBTS — RPGRIP1L has broader JBTS overlap",
            "NPHP4 (NPHP4 / 1p36)":         "TZ module PARTNER of NPHP8; SLS4; ocular motor apraxia; RPGRIP1L digenic → JBTS; always check both",
            "NPHP5 (IQCB1 / 3q21.1)":       "Most common SLS; severe LCA-like retinal; ESRD 13yr — RPGRIP1L retinal only in JBTS7 alleles",
            "NPHP6 (CEP290 / 12q21.32)":     "Broadest spectrum NPHP; IVS26→LCA10; JBTS5; MKS4; RPGRIP1L has overlapping but distinct spectrum",
            "NPHP7 (GLIS2 / 16p13.3)":       "PURE renal only; very rare; NO JBTS/MKS spectrum; NO retinal — RPGRIP1L has much broader multi-organ phenotype",
            "NPHP8 (RPGRIP1L / 16q12.2) ★": "THIS — NPHP8/JBTS7/MKS5; broad allele-phenotype spectrum; digenic JBTS with NPHP4; retinal 30%; CHF 18%; MTS 40%; ESRD 15–18yr",
        },
        "ddx_table": {
            "ADPKD (PKD1/PKD2)":            "Autosomal DOMINANT; enlarged kidneys; adult onset; NO TIN; most common NPHP8 misdiagnosis",
            "Joubert Syndrome (other genes)":"MTS present but CEP290/AHI1/INPP5E/TMEM67 rather than RPGRIP1L — gene panel mandatory; same MTS phenotype",
            "BBS (BBS1/BBS10 etc.)":         "Obesity; polydactyly; RP; NO MTS usually; NO TIN cysts; NOT confused with pure NPHP8",
            "FSGS":                          "Glomerular; proteinuria dominant; steroid responsive; NO concentrating defect first; NO corticomedullary cysts",
            "Alport (COL4A3/4/5)":           "Haematuria prominent; sensorineural deafness; GBM splitting; NO tubular cysts",
            "LCA (RPE65/GUCY2D etc.)":       "Retinal only; NO renal disease; NO MTS; NO RPGRIP1L — JBTS7 retinal resembles LCA; must add RPGRIP1L to LCA panels",
            "NPHP4 (1p36) digenic":          "Monoallelic NPHP4 + monoallelic RPGRIP1L → Joubert — diagnosis is digenic JBTS not NPHP8; both genes must be sequenced",
        },
        "treatment": {
            "Renal_transplant":             "CURATIVE for renal component; cell-autonomous TZ defect; NO recurrence; excellent outcomes; living donor preferred",
            "Retinal_treatment_JBTS7":      "NO approved retinal therapy 2026; gene augmentation pre-clinical; low-vision rehab; Braille/AT if severe",
            "Hepatic_management_JBTS7":     "Annual USS + APRI if CHF; propranolol for portal HTN; TIPSS/combined liver-kidney transplant if severe",
            "MTS_neurodevelopmental":       "OT/PT/speech/language therapy for JBTS7 motor and speech delay; neonatal breathing dysrhythmia self-resolves 2–3yr",
            "Conservative_CKD":             "2–3 L fluid/day; EPO for anaemia; ACE inhibitor if proteinuria/HTN; avoid NSAIDs/aminoglycosides; annual renal USS",
            "Growth_hormone_therapy":       "Consider rhGH for paediatric CKD-related growth retardation; transplant improves final height if pre-pubertal",
            "No_disease_modifying_Rx_2026": "No approved RPGRIP1L-targeted therapy 2026; gene augmentation (AAV-RPGRIP1L renal/retinal) pre-clinical",
            "Genetic_counselling":          "WES + CNV + NPHP4 status mandatory; 25% sibling risk; prenatal/PGT-M; digenic risk to extended family",
        },
        "prognosis": (
            "ESRD median ~15–18yr (NPHP8 pure; range 8–30yr). Renal transplant EXCELLENT — no recurrence. "
            "JBTS7 subtype: retinal disability is cell-autonomous and does NOT improve post-transplant. "
            "Neurodevelopmental disability (JBTS7) is variable — mild in missense, moderate-severe in biallelic strong alleles. "
            "Hepatic fibrosis managed conservatively; combined liver-kidney transplant when both organs fail (JBTS7). "
            "Overall: renal prognosis excellent post-transplant; extra-renal burden depends entirely on allele class. "
            "Diagnostic odyssey often prolonged (ADPKD/FSGS/LCA misdiagnosis common); WES shortens to <1yr in specialist centres."
        ),
        "cohort_note": (
            f"Synthetic cohort · n=40 · seed={SEED} · allele frequencies from published NPHP8/RPGRIP1L "
            "families (Arts 2007 first RPGRIP1L nephronophthisis; Delous 2007 JBTS7; Baala 2007 MKS5; "
            "Hoefele 2011 RPGRIP1L NPHP8 cohort; Halbritter 2013 NPHP cohort; Braun 2016 ciliopathy series). "
            "Allele class proportions reflect published NPHP8:JBTS7 ratio (~55:45). "
            "NOT human-subject data — illustrative only."
        ),
    }
