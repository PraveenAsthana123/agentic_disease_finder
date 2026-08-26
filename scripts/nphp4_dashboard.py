"""
Nephronophthisis Type 4 (NPHP4 — Juvenile/Adolescent NPHP; Nephrocystin-4 / Nephroretinin)
============================================================================================
Primary Gene : NPHP4 (*607215) — 1p36.31; 1426 aa; Nephrocystin-4 (Nephroretinin);
               TZ scaffold; interacts with NPHP1, NPHP3, RPGRIP1L, NPHP8
Disease OMIM : #606966 (Nephronophthisis 4, Juvenile)
Chromosome   : 1p36.31
Inheritance  : Autosomal Recessive (biallelic LOF)
Prevalence   : ~1/500,000–1/1,000,000 live births (rarer than NPHP1, NPHP2, NPHP3)

Mechanism
---------
NPHP4 (Nephrocystin-4 / Nephroretinin, 1426 aa) is a scaffolding protein of the ciliary
transition zone (TZ) and basal body. It completes the NPHP1-NPHP3-NPHP4 ternary module:
NPHP1 and NPHP4 interact via NPHP1-SH3 ↔ NPHP4-PxxP motif, creating a bipartite scaffold
that is bridged by NPHP3 (ankyrin repeats) and anchored by RPGRIP1L.

Key molecular roles:
  1. TZ gate integrity: NPHP4 is the C-terminal scaffold of the NPHP1-4 complex →
     LOF → TZ diffusion barrier collapse → disrupted ciliary protein import/export
     → failure of Hedgehog, Wnt/PCP and PDGF-Rα signalling in renal tubular epithelium
     → tubular dilation, interstitial fibrosis, corticomedullary cysts → TIN → ESRD
  2. Photoreceptor ciliary maintenance: NPHP4 expressed in connecting cilia of rod and
     cone photoreceptors → LOF → outer segment disc shedding failure → photoreceptor
     degeneration → rod-cone retinal dystrophy → Senior-Løken syndrome in ~15–20%
     (NPHP4 is the SECOND most common Senior-Løken gene after IQCB1/NPHP5)
  3. Ocular motor pathways: NPHP4 expression in ocular motor neurons → subset of patients
     develop nystagmus / oculomotor apraxia (ocular motor abnormalities distinctive of NPHP4)
  4. Sperm flagella: NPHP4 expressed in flagellar axoneme → male infertility subset
  5. NO nodal cilia expression: situs inversus ABSENT (unlike NPHP2 ~35%; NPHP3 ~15–20%)
  6. NO biliary epithelium expression: CHF ABSENT (unlike NPHP2 ~55%; NPHP3 ~45%)

LOF → Juvenile-to-early-adolescent TIN with small kidneys → progressive ESRD median ~17–20 yr.
Cell-autonomous disease → renal transplant CURATIVE, NO graft recurrence.
Retinal involvement does NOT improve after transplant (retina is cell-autonomous separately).

Hallmark Features (comparison with NPHP1/2/3):
  • JUVENILE–ADOLESCENT onset: ESRD median ~17–20 yr (overlaps NPHP1 at 13 yr, later than NPHP2 at 3 yr)
  • Retinal: YES in ~15–20% → Senior-Løken Syndrome (SLS) with rod-cone dystrophy + NPHP
  • Ocular motor abnormalities: nystagmus / oculomotor apraxia — KEY distinguishing NPHP4 feature
  • NO situs inversus: NPHP4 not expressed in nodal cilia (unlike NPHP2 and NPHP3)
  • NO hepatic fibrosis (CHF): NPHP4 not expressed in biliary epithelium (unlike NPHP2/3)
  • Kidneys SMALL (similar to NPHP1) — NOT enlarged as in NPHP2/ARPKD
  • Genetics: heterogeneous biallelic LOF; NO dominant deletion founder (unlike NPHP1 290kb del)
  • WES + CNV analysis required — NPHP4 included in standard ciliopathy panels

Key Differentials:
  NPHP1 (NPHP1/2q13): Juvenile 13yr; 290kb deletion 80%; NO retinal in most; NO situs; NO CHF
  NPHP2 (INVS/9q31): Infantile 3yr; situs inversus 35%; enlarged kidneys; CHF 55%; NO retinal
  NPHP3 (NPHP3/3q22.1): Adolescent 19yr; situs inversus 15-20%; small kidneys; CHF 45%; NO retinal
  NPHP5/IQCB1 (3q21.1): Retinal >> renal; LCA-like; MOST COMMON Senior-Løken gene
  CEP290/NPHP6 (12q21.32): Joubert MTS; LCA10 IVS26; Meckel spectrum; full ciliopathy
  Leber Congenital Amaurosis (LCA): retinal-only; NO renal component; exclude NPHP4 retinal+renal
  Alport (COL4A3/A4/A5): haematuria; proteinuria; GBM splitting; sensorineural hearing loss
  Joubert (multi-gene): Molar Tooth Sign; cerebellar vermis aplasia; oculomotor apraxia overlap

Treatment:
  • Renal transplant = definitive CURATIVE for the renal component (cell-autonomous; NO recurrence)
  • Retinal disease: does NOT improve post-transplant; ophthalmology/retinal specialist co-management
  • Conservative CKD: 2–3 L fluid/day (concentrating defect); avoid nephrotoxins
  • EPO for disproportionate anaemia
  • Low-vision rehabilitation for Senior-Løken subset (ERG + OCT surveillance annually)
  • Gene therapy for retinal component: pre-clinical for NPHP4-associated rod-cone dystrophy
  • No disease-modifying therapy approved 2026 for renal or retinal component
"""

import random
import statistics

SEED = 347
_RNG = random.Random(SEED)

# ── Genetic pool — realistic NPHP4 alleles (heterogeneous; no dominant founder) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("NPHP4 (1p36.31) — p.Arg436Cys / p.Ser840Ter (missense/nonsense compound het; pan-ethnic)",  0.14),
    ("NPHP4 (1p36.31) — p.Gln802Ter / p.Arg436Cys (nonsense/missense compound het; European)",    0.12),
    ("NPHP4 (1p36.31) — c.2670+1G>T / p.Gln802Ter (splice/nonsense compound het)",               0.10),
    ("NPHP4 (1p36.31) — p.Arg436Cys homozygous (South Asian/Middle Eastern consanguineous)",      0.09),
    ("NPHP4 (1p36.31) — del exon 13–15 / p.Ser840Ter (large del compound het; CNV required)",    0.08),
    ("NPHP4 (1p36.31) — p.Leu1044Pro / p.Gln802Ter (missense/nonsense; slower progression)",     0.08),
    ("NPHP4 (1p36.31) — frameshift c.2987delC / p.Arg436Cys (compound het; European)",           0.07),
    ("NPHP4 (1p36.31) — p.Glu826Ter homozygous (Turkish consanguineous)",                        0.07),
    ("NPHP4 (1p36.31) — c.1183-2A>G / p.Ser840Ter (splice acceptor/nonsense compound het)",     0.06),
    ("NPHP4 (1p36.31) — del exon 1-5 homozygous (North African consanguineous; array CGH)",      0.05),
    ("NPHP4 (1p36.31) — p.Leu1044Pro homozygous (biallelic missense; milder SLS/retinal)",      0.05),
    ("NPHP4 (1p36.31) — p.Arg1207Ter / c.2670+1G>T (nonsense/splice compound het)",             0.04),
    ("NPHP4 (1p36.31) — novel / VUS compound het (heterogeneous background; WES-confirmed)",     0.05),
]

_ETHNICITY_POOL = [
    ("European (pan-European heterogeneous)",             0.30),
    ("Middle Eastern / Arab (consanguinity enriched)",    0.24),
    ("South Asian (Indian subcontinent)",                 0.16),
    ("Turkish",                                           0.10),
    ("North African (consanguinity enriched)",            0.08),
    ("East Asian",                                        0.06),
    ("African / Sub-Saharan",                             0.04),
    ("Latin American",                                    0.02),
]

# Situs inversus ABSENT in NPHP4 (NPHP4 not expressed in nodal cilia)
_SITUS_POOL = [
    ("Situs solitus (normal laterality)",                 0.97),
    ("Situs inversus (incidental; non-NPHP4 cause)",     0.03),
]

# CHF ABSENT in NPHP4 (NPHP4 not expressed in biliary epithelium)
_CHF_POOL = [
    ("Absent (no hepatic fibrosis — NPHP4 not expressed in biliary epithelium)",  0.97),
    ("Incidental mild periportal fibrosis (non-specific; not NPHP4-related)",     0.03),
]

# Retinal involvement — Senior-Løken Syndrome in ~15–20% of NPHP4 patients
_RETINAL_POOL = [
    ("No retinal involvement (renal NPHP4 only; normal ERG)",                                      0.76),
    ("Senior-Løken Syndrome — rod-cone dystrophy + NPHP (NPHP4 expressed in photoreceptors)",     0.16),
    ("Subclinical retinal changes (reduced ERG amplitude; no visual symptoms yet)",                 0.05),
    ("Leber-like severe retinal dystrophy (biallelic null; early severe SLS variant)",              0.03),
]

# Ocular motor — key NPHP4 feature
_OCULAR_MOTOR_POOL = [
    ("No ocular motor abnormality",                                          0.72),
    ("Nystagmus (horizontal; congenital or early onset)",                    0.15),
    ("Oculomotor apraxia (saccadic initiation failure; horizontal gaze)",    0.08),
    ("Both nystagmus and oculomotor apraxia",                                0.05),
]

_RRT_POOL = [
    ("CKD stage 3–4 (juvenile/adolescent; close surveillance)",             0.26),
    ("Renal transplant — living related donor",                              0.26),
    ("Renal transplant — deceased donor",                                    0.18),
    ("Haemodialysis (awaiting transplant)",                                  0.16),
    ("Peritoneal dialysis (young adult, bridge to transplant)",              0.08),
    ("CKD stage 2 (slow progression; early polyuria)",                      0.06),
]

_MISDIAG_POOL = [
    ("ADPKD (incorrect AD assumption by adult renal team)",                        0.26),
    ("Alport syndrome (haematuria + CKD → COL4A3 sequenced first)",               0.20),
    ("Leber congenital amaurosis (retinal findings pursued without renal screen)",  0.16),
    ("Focal segmental glomerulosclerosis (FSGS; biopsy tubulointerstitial labelled FSGS)", 0.14),
    ("Medullary cystic disease (UMOD not found; re-tested)",                        0.10),
    ("Idiopathic CKD (no genetic workup initially)",                                0.10),
    ("No prior misdiagnosis (first genetic Dx correct)",                            0.04),
]


def _weighted_pick(pool):
    labels  = [p[0] for p in pool]
    weights = [p[1] for p in pool]
    return _RNG.choices(labels, weights=weights, k=1)[0]


def _make_patient(idx):
    gene         = _weighted_pick(_GENE_POOL)
    ethnicity    = _weighted_pick(_ETHNICITY_POOL)
    situs        = _weighted_pick(_SITUS_POOL)
    chf          = _weighted_pick(_CHF_POOL)
    retinal      = _weighted_pick(_RETINAL_POOL)
    ocular_motor = _weighted_pick(_OCULAR_MOTOR_POOL)
    rrt          = _weighted_pick(_RRT_POOL)
    misdiag      = _weighted_pick(_MISDIAG_POOL)

    # Juvenile-to-adolescent onset — age at diagnosis 7–24 yr (median ~17 yr)
    age_dx  = round(_RNG.betavariate(4, 3) * 17 + 7, 1)
    # GFR at Dx — CKD 2–5 range
    gfr_dx  = int(_RNG.betavariate(3, 2) * 65 + 18)
    gfr_now = max(5, gfr_dx - int(_RNG.betavariate(2, 3) * 42))
    # Urine osmolality — concentrating defect
    u_osm   = int(_RNG.betavariate(2, 5) * 280 + 50)
    # Hgb — disproportionate anaemia
    hgb     = round(8.0 + _RNG.betavariate(2, 3) * 5.5, 1)
    # Kidney size — all SMALL (no NPHP4 enlarged kidneys)
    kidney_size = _RNG.choices(
        ["Small (echogenic, loss CMD)", "Normal-to-small", "Normal (early stage)", "Shrunken (ESRD)"],
        weights=[0.40, 0.28, 0.20, 0.12]
    )[0]
    # GFR slope
    gfr_slope = round(_RNG.betavariate(2, 4) * 8 + 1, 1)

    return {
        "id":                    f"NPHP4-{idx:03d}",
        "gene":                  gene,
        "ethnicity":             ethnicity,
        "age_at_diagnosis_yr":   age_dx,
        "gfr_at_dx_ml_min":      gfr_dx,
        "gfr_now_ml_min":        gfr_now,
        "urine_osmolality_mosm": u_osm,
        "hemoglobin_g_dl":       hgb,
        "situs":                 situs,
        "chf_grade":             chf,
        "retinal":               retinal,
        "ocular_motor":          ocular_motor,
        "kidney_size":           kidney_size,
        "gfr_slope_ml_min_yr":   gfr_slope,
        "rrt_or_transplant":     rrt,
        "prior_misdiagnosis":    misdiag,
        "consanguineous":        ethnicity in (
            "Middle Eastern / Arab (consanguinity enriched)",
            "North African (consanguinity enriched)",
            "South Asian (Indian subcontinent)",
            "Turkish",
        ) and _RNG.random() < 0.60,
    }


_COHORT = [_make_patient(i + 1) for i in range(40)]


def get_overview():
    cohort = _COHORT
    n = len(cohort)

    ages    = [p["age_at_diagnosis_yr"]    for p in cohort]
    gfr_dx  = [p["gfr_at_dx_ml_min"]       for p in cohort]
    u_osm   = [p["urine_osmolality_mosm"]  for p in cohort]
    hgb     = [p["hemoglobin_g_dl"]        for p in cohort]

    pct_retinal    = round(sum(1 for p in cohort if "No retinal" not in p["retinal"]) / n * 100)
    pct_sls        = round(sum(1 for p in cohort if "Senior-Løken" in p["retinal"]) / n * 100)
    pct_ocular_m   = round(sum(1 for p in cohort if "No ocular" not in p["ocular_motor"]) / n * 100)
    pct_rrt        = round(sum(1 for p in cohort if "transplant" in p["rrt_or_transplant"].lower()
                               or "dialysis" in p["rrt_or_transplant"].lower()) / n * 100)
    pct_tx         = round(sum(1 for p in cohort if "transplant" in p["rrt_or_transplant"].lower()) / n * 100)
    pct_consang    = round(sum(1 for p in cohort if p["consanguineous"]) / n * 100)
    pct_misdiag    = round(sum(1 for p in cohort if "No prior" not in p["prior_misdiagnosis"]) / n * 100)
    pct_small      = round(sum(1 for p in cohort if "mall" in p["kidney_size"]
                               or "hrunken" in p["kidney_size"]) / n * 100)

    kpis = {
        "gene":                      "NPHP4 (Nephrocystin-4 / Nephroretinin)",
        "chromosome":                "1p36.31",
        "inheritance":               "Autosomal Recessive (biallelic LOF)",
        "prevalence":                "~1/500,000–1/1,000,000",
        "cohort_type":               "40-patient juvenile/adolescent NPHP4 cohort (seed-347)",
        "syndrome":                  "NPHP4; ±Senior-Løken (retinal); ±ocular motor; no situs; no CHF",
        "cohort_n":                  n,
        "median_age_dx_yr":          round(statistics.median(ages), 1),
        "median_gfr_at_dx_ml_min":   int(statistics.median(gfr_dx)),
        "median_urine_osmolality":   int(statistics.median(u_osm)),
        "mean_hgb_g_dl":             round(statistics.mean(hgb), 1),
        "pct_esrd_or_rrt":           pct_rrt,
        "pct_transplanted":          pct_tx,
        "pct_senior_loken":          pct_sls,
        "pct_retinal_any":           pct_retinal,
        "pct_ocular_motor":          pct_ocular_m,
        "pct_consanguineous":        pct_consang,
        "pct_prior_misdiagnosis":    pct_misdiag,
        "pct_kidneys_small":         pct_small,
    }

    alerts = {
        "retinal_senior_loken": (
            f"{pct_sls}% of cohort have Senior-Løken Syndrome (rod-cone retinal dystrophy + NPHP4 renal). "
            "NPHP4 is the SECOND most common SLS gene after IQCB1/NPHP5. "
            "ERG + OCT at diagnosis; annual ophthalmology surveillance for all NPHP4 patients. "
            "Retinal disease does NOT improve after renal transplant — separate cell-autonomous process."
        ),
        "ocular_motor_distinguisher": (
            f"{pct_ocular_m}% of cohort have ocular motor abnormalities (nystagmus / oculomotor apraxia). "
            "This is a KEY distinguishing feature of NPHP4 — absent in NPHP1/2/3. "
            "Ocular motor apraxia + CKD + small kidneys → NPHP4 gene panel before Joubert workup "
            "(NPHP4 has no Molar Tooth Sign on MRI — differentiates from true Joubert JBTS)."
        ),
        "lca_misdiagnosis_trap": (
            "LCA (Leber Congenital Amaurosis) misdiagnosis: NPHP4 retinal phenotype can resemble LCA. "
            f"{pct_misdiag}% had prior misdiagnosis. Always screen renal function (creatinine + Uosm) "
            "in any LCA patient — NPHP4 retinal disease may precede detectable CKD by years."
        ),
        "no_situs_no_chf": (
            "SITUS INVERSUS: ABSENT (NPHP4 not expressed in nodal cilia). "
            "CHF (hepatic fibrosis): ABSENT (NPHP4 not expressed in biliary epithelium). "
            "Key distinction from NPHP2 (situs 35% + CHF 55%) and NPHP3 (situs 15% + CHF 45%)."
        ),
        "transplant_curative_renal_only": (
            "Renal transplant is CURATIVE for the renal component — cell-autonomous TZ defect, NO graft recurrence. "
            "HOWEVER: retinal disease in Senior-Løken subset does NOT improve — ophthalmology remains "
            "lifelong. Living related donors (obligate heterozygotes) are SAFE."
        ),
    }

    key_facts = [
        "NPHP4 / Nephrocystin-4 (Nephroretinin): 1426 aa; 1p36.31; completes NPHP1-NPHP3-NPHP4 TZ module; AR biallelic LOF",
        "Juvenile-adolescent onset: ESRD median ~17–20 yr (overlaps NPHP1 at 13 yr; later than NPHP2/infantile)",
        "Senior-Løken Syndrome ~15–20%: rod-cone retinal dystrophy + NPHP; NPHP4 is 2nd most common SLS gene after IQCB1",
        "OCULAR MOTOR abnormalities: nystagmus + oculomotor apraxia — KEY distinguishing NPHP4 feature (absent in NPHP1/2/3)",
        "NO situs inversus: NPHP4 not expressed in nodal cilia (unlike NPHP2 ~35%, NPHP3 ~15–20%)",
        "NO hepatic fibrosis (CHF): NPHP4 not expressed in biliary epithelium (unlike NPHP2 ~55%, NPHP3 ~45%)",
        "Kidneys SMALL on USS (echogenic, loss CMD) — NOT enlarged as in NPHP2/ARPKD",
        "Genetics: heterogeneous biallelic LOF; NO dominant deletion founder (unlike NPHP1 290kb 2q13 del); WES + CNV required",
        "NPHP4 interacts with NPHP1 (SH3-PxxP), NPHP3 (ankyrin), RPGRIP1L, NPHP8 — part of NPHP1-3-4 TZ complex",
        "Concentrating defect: Uosm < 300 mosm/kg; polyuria/polydipsia often first symptom (tubular before glomerular)",
        "Anaemia: disproportionate for CKD degree (EPO-producing interstitial cell loss); starts CKD 3",
        "Renal transplant = CURATIVE for renal component; retinal disease (SLS) does NOT improve post-transplant",
        "LCA misdiagnosis trap: NPHP4 retinal phenotype resembles LCA → always check renal function in LCA patients",
        "RPGRIP1L + NPHP4 digenic variants can cause Joubert syndrome — gene panel must include full NPHP/Joubert panel",
    ]

    return {
        "kpis":      kpis,
        "alerts":    alerts,
        "key_facts": key_facts,
        "patients":  cohort[:8],
    }


def get_breakdown():
    cohort = _COHORT
    n = len(cohort)

    # Kidney size
    sizes = {}
    for p in cohort:
        k = p["kidney_size"]
        sizes[k] = sizes.get(k, 0) + 1

    # Retinal involvement
    retinal_dist = {}
    for p in cohort:
        r = p["retinal"].split("(")[0].strip()[:60]
        retinal_dist[r] = retinal_dist.get(r, 0) + 1

    # Ocular motor
    ocular_dist = {}
    for p in cohort:
        o = p["ocular_motor"].split("(")[0].strip()[:55]
        ocular_dist[o] = ocular_dist.get(o, 0) + 1

    # CKD stage (current)
    ckd = {"CKD 5 (ESRD/RRT)": 0, "CKD 4 (15–29)": 0, "CKD 3 (30–44)": 0, "CKD ≤2 (≥45)": 0}
    for p in cohort:
        g = p["gfr_now_ml_min"]
        if g < 15:   ckd["CKD 5 (ESRD/RRT)"] += 1
        elif g < 30: ckd["CKD 4 (15–29)"] += 1
        elif g < 45: ckd["CKD 3 (30–44)"] += 1
        else:        ckd["CKD ≤2 (≥45)"] += 1

    # Urine osmolality
    u_tiers = {"<100 mosm/kg (severe defect)": 0, "100–200": 0, "200–300": 0, ">300 mosm/kg": 0}
    for p in cohort:
        u = p["urine_osmolality_mosm"]
        if u < 100:   u_tiers["<100 mosm/kg (severe defect)"] += 1
        elif u < 200: u_tiers["100–200"] += 1
        elif u < 300: u_tiers["200–300"] += 1
        else:         u_tiers[">300 mosm/kg"] += 1

    # RRT
    rrt_dist = {}
    for p in cohort:
        r = p["rrt_or_transplant"].split("—")[0].split("(")[0].strip()
        rrt_dist[r] = rrt_dist.get(r, 0) + 1

    # Age at Dx tiers
    age_t = {"<10 yr (early childhood)": 0, "10–14 yr (early juvenile)": 0,
             "14–20 yr (juvenile-adolescent)": 0, ">20 yr (young adult)": 0}
    for p in cohort:
        a = p["age_at_diagnosis_yr"]
        if a < 10:   age_t["<10 yr (early childhood)"] += 1
        elif a < 14: age_t["10–14 yr (early juvenile)"] += 1
        elif a < 20: age_t["14–20 yr (juvenile-adolescent)"] += 1
        else:        age_t[">20 yr (young adult)"] += 1

    # Prior misdiagnosis
    misdiag = {}
    for p in cohort:
        m = p["prior_misdiagnosis"].split("(")[0].strip()[:55]
        misdiag[m] = misdiag.get(m, 0) + 1

    # Gene allele distribution
    gene_dist = {}
    for p in cohort:
        tag = p["gene"].split("—")[-1].strip().split("/")[0].strip()[:65]
        gene_dist[tag] = gene_dist.get(tag, 0) + 1

    # Ethnicity
    eth = {}
    for p in cohort:
        e = p["ethnicity"].split("(")[0].strip()
        eth[e] = eth.get(e, 0) + 1

    # GFR slope tiers
    slope_t = {"1–3 ml/min/yr (slow)": 0, "3–6 ml/min/yr (moderate)": 0, ">6 ml/min/yr (rapid)": 0}
    for p in cohort:
        s = p["gfr_slope_ml_min_yr"]
        if s <= 3:   slope_t["1–3 ml/min/yr (slow)"] += 1
        elif s <= 6: slope_t["3–6 ml/min/yr (moderate)"] += 1
        else:        slope_t[">6 ml/min/yr (rapid)"] += 1

    return {
        "kidney_size_distribution":   dict(sorted(sizes.items(), key=lambda x: -x[1])),
        "retinal_involvement":        dict(sorted(retinal_dist.items(), key=lambda x: -x[1])),
        "ocular_motor_distribution":  dict(sorted(ocular_dist.items(), key=lambda x: -x[1])),
        "ckd_stage_current":          ckd,
        "urine_osmolality_tiers":     u_tiers,
        "rrt_transplant_status":      dict(sorted(rrt_dist.items(), key=lambda x: -x[1])),
        "age_at_diagnosis_tiers":     age_t,
        "gfr_slope_tiers":            slope_t,
        "prior_misdiagnosis":         dict(sorted(misdiag.items(), key=lambda x: -x[1])),
        "gene_distribution":          dict(sorted(gene_dist.items(), key=lambda x: -x[1])[:6]),
        "ethnicity":                  dict(sorted(eth.items(), key=lambda x: -x[1])),
    }


def get_definitions():
    return {
        "disease":      "Nephronophthisis Type 4 (NPHP4 — Juvenile/Adolescent NPHP; Senior-Løken Syndrome 4 when retinal)",
        "omim_gene":    "*607215 (NPHP4 — Nephrocystin-4 / Nephroretinin)",
        "omim_disease": "#606966 (Nephronophthisis 4, Juvenile) | Senior-Løken Syndrome 4 when retinal coexists",
        "chromosome":   "1p36.31",
        "inheritance":  "Autosomal Recessive; biallelic LOF; no dominant deletion founder",
        "prevalence":   "~1/500,000–1/1,000,000 live births (rarer than NPHP1, NPHP2, NPHP3)",
        "mechanism":    (
            "NPHP4 (Nephrocystin-4 / Nephroretinin, 1426 aa) completes the NPHP1-NPHP3-NPHP4 ternary "
            "scaffolding module at the ciliary transition zone (TZ) and basal body. NPHP1 and NPHP4 "
            "interact directly via NPHP1-SH3 ↔ NPHP4-PxxP motif; NPHP3 bridges via ankyrin repeats; "
            "RPGRIP1L anchors the complex to the TZ matrix. LOF → TZ gate collapse → disrupted ciliary "
            "protein import/export → failure of Hedgehog, Wnt/PCP and PDGF-Rα signalling in renal "
            "tubular epithelium → tubular dilation, interstitial fibrosis, corticomedullary cysts → "
            "progressive TIN → ESRD. Separately, NPHP4 is expressed in the connecting cilia of rod "
            "and cone photoreceptors → LOF → disc shedding failure → photoreceptor degeneration → "
            "Senior-Løken Syndrome (rod-cone dystrophy + NPHP) in ~15–20% of patients."
        ),
        "genetic_architecture": {
            "gene":                "NPHP4 (1p36.31); 1426 aa; Nephrocystin-4/Nephroretinin; SH3-binding PxxP; coiled-coil C-term",
            "mutation_spectrum":   "Heterogeneous: nonsense, splice-site, frameshift; missense (biallelic milder); large deletions (CNV)",
            "founder_variants":    "p.Arg436Cys — pan-ethnic recurrent; p.Ser840Ter — European cohorts; "
                                   "del exon 1-5 — North African consanguineous; p.Glu826Ter — Turkish consanguineous",
            "diagnostic_strategy": "WES + CNV analysis; NPHP4 included in standard ciliopathy gene panels "
                                   "(NPHP/Joubert/BBS/MKS 40–200 gene panels); MLPA not routinely available; "
                                   "large deletions need array CGH or CNV-WES",
            "genotype_phenotype":  "Biallelic null (nonsense/fs) → ESRD ~14–19 yr ± SLS; biallelic missense "
                                   "(p.Leu1044Pro homo) → slower ESRD ~20–26 yr; retinal severity allele-dependent",
            "complex_genetics":    "RPGRIP1L + NPHP4 digenic mutations can cause Joubert syndrome — "
                                   "full NPHP/Joubert panel mandatory; second-hit co-mutations possible",
        },
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)":   "Juvenile ESRD 13yr; 290kb del 80%; NO retinal (most); NO situs; NO CHF",
            "NPHP2 (INVS / 9q31.1)":  "INFANTILE ESRD 3yr; situs inversus 35%; enlarged kidneys; CHF 55%; NO retinal",
            "NPHP3 (NPHP3 / 3q22.1)": "ADOLESCENT ESRD ~19yr; situs inversus 15–20%; small kidneys; CHF ~45%; NO retinal",
            "NPHP4 (NPHP4 / 1p36.31)": "JUVENILE/ADOLESCENT ESRD ~17–20yr — THIS DISEASE; Senior-Løken SLS ~15–20%; "
                                         "OCULAR MOTOR (nystagmus/OMA); NO situs; NO CHF",
            "NPHP5/IQCB1 (3q21.1)":   "MOST COMMON Senior-Løken gene; severe LCA-like retinal >> renal; retinal dominant",
            "NPHP6/CEP290 (12q21.32)": "Joubert MTS pathognomonic; LCA10 IVS26; BBS14; full ciliopathy spectrum",
        },
        "key_clinical_features": {
            "onset":              "Juvenile to adolescent: ESRD median ~17–20 yr (range 10–25 yr); overlaps NPHP1",
            "polyuria_first":     "Concentrating defect is often FIRST symptom: Uosm < 300 mosm/kg; polyuria/polydipsia",
            "kidneys":            "SMALL on USS (echogenic, loss CMD); NOT enlarged; similar to NPHP1",
            "retinal_sls":        "Senior-Løken Syndrome ~15–20%: rod-cone retinal dystrophy + NPHP4 renal component",
            "ocular_motor":       "KEY FEATURE: nystagmus / oculomotor apraxia in ~20–25%; absent in NPHP1/2/3",
            "no_situs_inversus":  "ABSENT — NPHP4 not expressed in nodal cilia (unlike NPHP2/NPHP3)",
            "no_chf":             "ABSENT — NPHP4 not expressed in biliary epithelium (unlike NPHP2/NPHP3)",
            "anaemia":            "Disproportionate for CKD (EPO interstitial cell loss); starts at CKD stage 3",
            "growth_retardation": "Short stature — CKD-related GH insensitivity in paediatric patients",
            "gfr_decline":        "~3–8 ml/min/yr (similar to NPHP1); inexorable progression to ESRD",
            "normotension":       "BP normal or low (salt-wasting tubular disease); contrast ADPKD (HTN dominant)",
        },
        "diagnostic_criteria": {
            "genetic_gold_std":  "Biallelic pathogenic NPHP4 variants on WES + CNV; ciliopathy panel required",
            "clinical_triggers": "Juvenile CKD + small echogenic kidneys + concentrating defect + corticomedullary cysts "
                                  "± retinal dystrophy ± nystagmus/OMA → NPHP4 panel mandatory",
            "ophthalmology":     "ERG + OCT at diagnosis for all NPHP4 patients — SLS can precede or follow renal Dx; "
                                  "annual surveillance even if normal at Dx",
            "imaging":           "Renal USS: small kidneys; corticomedullary cysts 1–2 cm; echogenic cortex. "
                                  "Brain MRI: NO Molar Tooth Sign (distinguishes NPHP4 from Joubert JBTS). "
                                  "Retinal OCT: outer nuclear layer thinning in SLS subset",
            "labs":              "Uosm < 300 mosm/kg; tubular proteinuria; disproportionate anaemia; "
                                  "elevated creatinine for age; LFTs NORMAL (no CHF)",
            "avoid":             "Do NOT diagnose as pure LCA and skip renal workup — NPHP4 retinal + renal "
                                  "both require management. Do NOT skip gene panel for adolescent FSGS/CKD.",
        },
        "ddx_table": {
            "NPHP1":                "Juvenile ESRD 13yr; SMALL kidneys; 290kb 2q13 del 80%; NO retinal (most); NO situs; NO CHF",
            "NPHP5/IQCB1":          "MOST COMMON SLS gene; severe LCA-like retinal >> renal; no ocular motor apraxia",
            "Joubert (CEP290/AHI1)": "Molar Tooth Sign on MRI mandatory for Joubert Dx; NPHP4 has NO MTS",
            "LCA":                  "Retinal-only; NO renal involvement; CRX/RPGRIP1/CEP290-IVS26; check renal function always",
            "ADPKD":                "AUTOSOMAL DOMINANT; adult; HTN; macrocysts; PKD1/PKD2; NO retinal; NO small kidneys",
            "Alport":               "COL4A3/A4/A5; haematuria + proteinuria + sensorineural hearing loss; GBM splitting on EM",
            "FSGS":                 "Proteinuria dominant; nephrotic; biopsy glomerular primary — NPHP4 biopsy may show FSGS-like TIN",
            "Medullary cystic (UMOD)": "AD; gout/hyperuricaemia; adult onset; NO retinal; medullary not corticomedullary cysts",
        },
        "treatment": {
            "renal_transplant":      "DEFINITIVE CURATIVE for renal component — cell-autonomous TZ defect; NO graft recurrence",
            "retinal_not_cured":     "Retinal disease (SLS) does NOT improve after renal transplant — separate cell-autonomous process",
            "living_donor":          "Obligate heterozygote parents are SAFE donors — normal renal function; normal ERG in carriers",
            "fluid_replacement":     "2–3 L/day fluid (concentrating defect); dehydration accelerates CKD decline",
            "avoid_nephrotoxins":    "NSAIDs, nephrotoxic contrast, aminoglycosides worsen CKD progression",
            "epo":                   "Erythropoietin for disproportionate anaemia (target Hgb 11–12 g/dL)",
            "low_vision_rehab":      "Low-vision rehabilitation, orientation/mobility training for Senior-Løken subset",
            "retinal_gene_therapy":  "Pre-clinical: NPHP4 AAV subretinal gene therapy research; no approved therapy 2026",
            "no_immunosuppression":  "Steroids/MMF/rituximab have NO role in NPHP4 renal disease — will not slow progression",
            "no_dmt_2026":           "No approved disease-modifying therapy 2026; mTOR inhibitor pre-clinical for renal",
        },
        "founder_variants": [
            "p.Arg436Cys (c.1306C>T) — pan-ethnic recurrent; European/Middle Eastern cohorts",
            "p.Ser840Ter (c.2520C>A) — recurrent in European heterogeneous cohorts",
            "del exon 1-5 homozygous — North African consanguineous families; requires array CGH / CNV-WES",
            "p.Glu826Ter — Turkish consanguineous; regional founder",
            "c.2670+1G>T splice site — recurrent in European cohorts; disrupts exon 22 donor",
            "p.Leu1044Pro homozygous — biallelic missense; slower progression to ESRD and milder retinal if SLS",
            "WES + CNV mandatory — large deletions (del exon 13-15; del exon 1-5) require CNV analysis; standard WES may miss",
        ],
        "prognosis": (
            "ESRD by median ~17–20 yr (range 10–25 yr); renal transplant is curative for the renal component "
            "with excellent graft outcomes. Retinal disease (Senior-Løken Syndrome) in ~15–20%: "
            "progressive rod-cone dystrophy leading to legal blindness in adulthood; does NOT improve "
            "post-transplant. Ocular motor abnormalities (nystagmus/OMA) are non-progressive in most. "
            "No hepatic, cardiac, or CNS involvement (no Joubert MTS — differentiates from NPHP6/CEP290). "
            "Biallelic null alleles → earlier ESRD ~14–19 yr; biallelic missense → slower ESRD ~20–26 yr. "
            "Long-term QoL excellent for renal component with timely diagnosis and transplant; "
            "visual impairment in SLS subset requires lifelong ophthalmological co-management."
        ),
        "cohort_note": (
            "Synthetic cohort · 40 patients · NPHP4 (Nephrocystin-4/Nephroretinin) · seed-347 · generated "
            "for clinical decision-support training · not derived from real patient data."
        ),
    }
