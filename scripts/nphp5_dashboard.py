"""
Nephronophthisis Type 5 / Senior-Løken Syndrome 5 (IQCB1/NPHP5)
================================================================
Primary Gene : IQCB1 (*607526) — 3q21.1; 590 aa; IQ motif-containing B1 (NPHP5);
               calmodulin-binding; photoreceptor connecting cilium & renal TZ scaffold
Disease OMIM : #609254 (Senior-Løken Syndrome 5)
               #611498 (Nephronophthisis 5 — renal-only subset)
Chromosome   : 3q21.1
Inheritance  : Autosomal Recessive (biallelic LOF)
Prevalence   : ~1/100,000–1/500,000 live births (commoner than NPHP2–4)

Mechanism
---------
IQCB1 (NPHP5 / IQ motif-containing B1 protein, 590 aa) contains two IQ calmodulin-binding
motifs and a C-terminal coiled-coil domain. It bridges calmodulin signalling and IFT machinery
at the ciliary transition zone and basal body.

Key molecular roles:
  1. Photoreceptor connecting cilia: IQCB1 is expressed in connecting cilia of rod and cone
     photoreceptors and interacts directly with RPGR (Retinitis Pigmentosa GTPase Regulator)
     → LOF → disruption of outer segment disk formation → severe, early-onset rod-cone
     dystrophy resembling Leber Congenital Amaurosis (LCA-like phenotype)
  2. Renal tubular TZ: IQCB1 also localises to renal tubular primary cilia → LOF →
     TZ dysfunction → tubulointerstitial nephritis → corticomedullary cysts → ESRD
     (renal disease usually milder or later than retinal in NPHP5)
  3. RPGR interaction: NPHP5 bridges RPGR (ciliary motor/transport) to the NPHP-RC
     (nephronophthisis retinitis pigmentosa complex) → photoreceptor disc shedding failure
     is the dominant pathology
  4. Calmodulin link: IQ motifs mediate Ca²⁺-calmodulin regulation of ciliary beat/transport
     cycle at the CC gate

LOF → Severe LCA-like early retinal dystrophy (visual impairment in first decade) THEN
progressive renal disease → ESRD median ~13 yr (RETINAL >> RENAL in severity and timing).

Hallmark Features (comparison with other NPHP subtypes):
  • MOST COMMON Senior-Løken Syndrome gene (commoner than NPHP4 which is 2nd most common)
  • Severe LCA-like retinal: nystagmus, visual impairment, ERG markedly reduced/flat — often
    diagnosed as LCA in infancy; NPHP5 should be sequenced in ALL LCA cases with renal signs
  • Retinal >> Renal: visual impairment precedes CKD in most; renal function often normal at
    ophthalmology diagnosis
  • ESRD median ~13 yr (similar to NPHP1); kidneys SMALL
  • No situs inversus; No hepatic fibrosis; No Molar Tooth Sign
  • RPGR interaction: NPHP5 overlaps genetically/phenotypically with X-linked RP pathway
    (retinal gene panel mandatory alongside renal gene panel)
  • Male infertility: rare subset (connecting cilium homology with sperm flagella)

Key Differentials:
  NPHP1 (NPHP1/2q13): Juvenile 13yr; mild/no retinal (10-15%); 290kb del 80%; NO LCA-like
  NPHP4 (NPHP4/1p36): Juvenile 17-20yr; SLS ~15-20%; ocular motor apraxia; 2nd most common SLS
  NPHP6/CEP290 (12q21.32): LCA10 (IVS26); Joubert MTS; BBS14; full ciliopathy spectrum
  Leber Congenital Amaurosis (CRX/RPGRIP1): retinal-only; no renal; exclude NPHP5 LCA+renal
  X-linked RP (RPGR): X-linked; males affected; NPHP5 interacts with RPGR but is AR
  Alport (COL4A3/A4/A5): haematuria; GBM splitting; hearing loss; no retinal

Treatment:
  • Renal transplant = definitive CURATIVE for renal component (cell-autonomous; NO recurrence)
  • Retinal disease does NOT improve after renal transplant — separate cell-autonomous process
  • Low-vision rehabilitation; orientation/mobility training; braille/AT from childhood
  • Gene therapy for IQCB1 retinal component: pre-clinical; AAV-IQCB1 subretinal approach
  • Conservative CKD: 2–3 L fluid/day; EPO for disproportionate anaemia
  • Annual ophthalmology (ERG + OCT) from diagnosis — retinal is the dominant disability
  • Genetic counselling: carrier sibling screening (25% sibling risk)
  • No disease-modifying therapy approved 2026 for retinal or renal component
"""

import random
import statistics

SEED = 349
_RNG = random.Random(SEED)

# ── Genetic pool — realistic IQCB1 alleles (heterogeneous; no single dominant founder) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("IQCB1 (3q21.1) — p.Arg461Ter / p.Trp448Ter (nonsense/nonsense compound het; severe LCA-like)",    0.14),
    ("IQCB1 (3q21.1) — p.Arg461Ter homozygous (Middle Eastern/South Asian consanguineous; null/null)",   0.13),
    ("IQCB1 (3q21.1) — c.1243+1G>A / p.Arg461Ter (splice/nonsense compound het; European)",             0.11),
    ("IQCB1 (3q21.1) — p.Trp448Ter homozygous (North African consanguineous; severe early retinal)",     0.09),
    ("IQCB1 (3q21.1) — p.Gln446Ter / c.1243+1G>A (nonsense/splice compound het; pan-ethnic)",          0.08),
    ("IQCB1 (3q21.1) — del exon 3–6 / p.Arg461Ter (large del compound het; CNV required)",             0.08),
    ("IQCB1 (3q21.1) — p.Arg461Ter / p.Ile398Thr (null/missense compound het; slower renal)",          0.07),
    ("IQCB1 (3q21.1) — p.Gly270Asp / p.Trp448Ter (missense/nonsense compound het; European)",          0.07),
    ("IQCB1 (3q21.1) — frameshift c.1174delA / p.Arg461Ter (compound het; heterogeneous)",             0.06),
    ("IQCB1 (3q21.1) — p.Ile398Thr homozygous (biallelic missense; milder/later renal disease)",       0.05),
    ("IQCB1 (3q21.1) — del exon 1–5 homozygous (Turkish consanguineous; array CGH required)",          0.05),
    ("IQCB1 (3q21.1) — c.640+2T>C / p.Arg461Ter (splice/nonsense compound het; European)",            0.04),
    ("IQCB1 (3q21.1) — novel / VUS compound het (WES-confirmed; heterogeneous background)",             0.03),
]

_ETHNICITY_POOL = [
    ("European (pan-European heterogeneous)",             0.28),
    ("Middle Eastern / Arab (consanguinity enriched)",    0.26),
    ("South Asian (Indian subcontinent)",                 0.16),
    ("Turkish",                                           0.10),
    ("North African (consanguinity enriched)",            0.08),
    ("East Asian",                                        0.06),
    ("African / Sub-Saharan",                             0.04),
    ("Latin American",                                    0.02),
]

# Situs inversus ABSENT in NPHP5 (IQCB1 not expressed in nodal cilia)
_SITUS_POOL = [
    ("Situs solitus (normal laterality)",               0.98),
    ("Situs inversus (incidental; non-IQCB1 cause)",   0.02),
]

# CHF ABSENT in NPHP5 (IQCB1 not expressed in biliary epithelium)
_CHF_POOL = [
    ("Absent (no hepatic fibrosis — IQCB1 not expressed in biliary epithelium)",  0.98),
    ("Incidental mild periportal fibrosis (non-specific; not NPHP5-related)",     0.02),
]

# Retinal involvement — DOMINANT in NPHP5; LCA-like severe early retinal dystrophy in ALL
_RETINAL_POOL = [
    ("Senior-Løken Syndrome — severe LCA-like rod-cone dystrophy + NPHP (markedly reduced/flat ERG)",   0.72),
    ("Senior-Løken Syndrome — moderate rod-cone dystrophy + NPHP (reduced ERG; residual peripheral)",   0.16),
    ("Severe LCA-like retinal (biallelic null; nystagmus; visual impairment < 1 yr)",                   0.08),
    ("Subclinical/milder retinal (biallelic missense p.Ile398Thr; residual visual acuity)",              0.04),
]

# Ocular motor abnormalities — nystagmus from LCA-like early retinal involvement
_OCULAR_POOL = [
    ("Nystagmus (pendular; from early severe visual impairment; near-universal in SLS5)",  0.74),
    ("No nystagmus (later/milder retinal; preserved fixation at diagnosis)",               0.18),
    ("Nystagmus + photophobia (cone-dominant early involvement; severe LCA variant)",      0.08),
]

_PRIOR_MISDIAG_POOL = [
    ("No prior misdiagnosis (IQCB1 panel early; SLS5 Dx correct)",                          0.24),
    ("LCA misdiagnosis — retinal-only panel; renal workup omitted (most common error)",     0.42),
    ("X-linked RP misdiagnosis — RPGR panel negative; IQCB1 not initially included",       0.14),
    ("ADPKD misdiagnosis — AD assumption by adult nephrology; AR genetics not checked",     0.08),
    ("Alport misdiagnosis — haematuria rare; COL4A3 sequenced first; IQCB1 missed",        0.07),
    ("Joubert misdiagnosis — LCA + renal → CEP290 panel sent first; no MTS on MRI",       0.05),
]

_RRT_POOL = [
    ("Pre-dialysis CKD surveillance (GFR ≥ 20; transplant planning phase)",                                        0.22),
    ("Renal transplant — excellent outcome; no graft recurrence (cell-autonomous CURATIVE)",                       0.28),
    ("Haemodialysis bridge to transplant (ESRD GFR < 15; listed for transplant)",                                 0.20),
    ("Peritoneal dialysis bridge (home-based; compliance preserved; waiting transplant)",                          0.12),
    ("Live-donor transplant received (parental/sibling heterozygote donor — SAFE)",                               0.18),
]

def _weighted(pool, rng):
    labels, weights = zip(*pool)
    return rng.choices(labels, weights=weights)[0]


def _make_patient(idx):
    rng = random.Random(SEED + idx * 97)

    gene      = _weighted(_GENE_POOL, rng)
    ethnicity = _weighted(_ETHNICITY_POOL, rng)
    situs     = _weighted(_SITUS_POOL, rng)
    chf       = _weighted(_CHF_POOL, rng)
    retinal   = _weighted(_RETINAL_POOL, rng)
    ocular    = _weighted(_OCULAR_POOL, rng)
    misdiag   = _weighted(_PRIOR_MISDIAG_POOL, rng)
    rrt       = _weighted(_RRT_POOL, rng)

    # Age at retinal diagnosis (visual impairment) — usually first presentation
    # NPHP5: retinal diagnosed in infancy to early childhood in severe LCA-like cases
    age_retinal_dx = round(rng.betavariate(1.5, 5) * 8 + 0.5, 1)  # 0.5–8 yr
    # Age at renal (CKD) diagnosis — usually later than retinal
    age_renal_dx   = round(age_retinal_dx + rng.betavariate(2, 3) * 10 + 1, 1)
    # GFR at renal diagnosis
    gfr_dx  = int(rng.betavariate(3, 2) * 55 + 20)
    # Current GFR
    gfr_now = max(5, gfr_dx - int(rng.betavariate(2, 3) * 38))
    # Urine osmolality — concentrating defect
    u_osm   = int(rng.betavariate(2, 5) * 270 + 55)
    # Hgb — disproportionate anaemia
    hgb     = round(8.2 + rng.betavariate(2, 3) * 5.0, 1)
    # Kidney size — SMALL (similar to NPHP1)
    kidney_size = rng.choices(
        ["Small (echogenic, loss CMD)", "Normal-to-small", "Normal (early stage)", "Shrunken (ESRD)"],
        weights=[0.42, 0.26, 0.20, 0.12]
    )[0]
    # GFR slope
    gfr_slope = round(rng.betavariate(2, 4) * 7 + 1.5, 1)
    # Visual acuity at last assessment
    va = rng.choices(
        ["Light perception / hand movements (severe LCA-like)",
         "CF at 1–2 m (count fingers; legal blindness)",
         "6/60–6/24 (low vision; central spared)",
         "6/24–6/12 (moderate impairment; peripheral loss)",
         "6/12–6/6 (mild; subclinical ERG change only)"],
        weights=[0.28, 0.25, 0.22, 0.15, 0.10]
    )[0]

    return {
        "id":                       f"NPHP5-{idx:03d}",
        "gene":                     gene,
        "ethnicity":                ethnicity,
        "age_retinal_dx_yr":        age_retinal_dx,
        "age_renal_dx_yr":          age_renal_dx,
        "gfr_at_dx_ml_min":         gfr_dx,
        "gfr_now_ml_min":           gfr_now,
        "urine_osmolality_mosm":    u_osm,
        "hemoglobin_g_dl":          hgb,
        "situs":                    situs,
        "chf_grade":                chf,
        "retinal":                  retinal,
        "ocular_motor":             ocular,
        "visual_acuity":            va,
        "kidney_size":              kidney_size,
        "gfr_slope_ml_min_yr":      gfr_slope,
        "rrt_or_transplant":        rrt,
        "prior_misdiagnosis":       misdiag,
        "consanguineous":           ethnicity in (
            "Middle Eastern / Arab (consanguinity enriched)",
            "North African (consanguinity enriched)",
            "South Asian (Indian subcontinent)",
            "Turkish",
        ) and rng.random() < 0.62,
    }


_COHORT = [_make_patient(i + 1) for i in range(40)]


def get_overview():
    cohort = _COHORT
    n = len(cohort)

    ages_retinal = [p["age_retinal_dx_yr"]   for p in cohort]
    ages_renal   = [p["age_renal_dx_yr"]     for p in cohort]
    gfr_dx       = [p["gfr_at_dx_ml_min"]    for p in cohort]
    u_osm        = [p["urine_osmolality_mosm"] for p in cohort]
    hgb          = [p["hemoglobin_g_dl"]      for p in cohort]

    pct_sls          = round(sum(1 for p in cohort if "Senior-Løken" in p["retinal"]) / n * 100)
    pct_nystagmus    = round(sum(1 for p in cohort if "Nystagmus" in p["ocular_motor"]) / n * 100)
    pct_rrt          = round(sum(1 for p in cohort if "transplant" in p["rrt_or_transplant"].lower()
                                  or "dialysis" in p["rrt_or_transplant"].lower()) / n * 100)
    pct_tx           = round(sum(1 for p in cohort if "transplant" in p["rrt_or_transplant"].lower()) / n * 100)
    pct_consang      = round(sum(1 for p in cohort if p["consanguineous"]) / n * 100)
    pct_misdiag      = round(sum(1 for p in cohort if "No prior" not in p["prior_misdiagnosis"]) / n * 100)
    pct_small        = round(sum(1 for p in cohort if "mall" in p["kidney_size"]
                                  or "hrunken" in p["kidney_size"]) / n * 100)
    pct_blind        = round(sum(1 for p in cohort if "Light perception" in p["visual_acuity"]
                                  or "CF at 1" in p["visual_acuity"]) / n * 100)
    pct_lca_misdiag  = round(sum(1 for p in cohort if "LCA misdiagnosis" in p["prior_misdiagnosis"]) / n * 100)

    kpis = {
        "gene":                         "IQCB1 (NPHP5 — IQ motif-containing B1; calmodulin-binding)",
        "chromosome":                   "3q21.1",
        "inheritance":                  "Autosomal Recessive (biallelic LOF)",
        "prevalence":                   "~1/100,000–1/500,000",
        "cohort_type":                  "40-patient NPHP5/IQCB1 Senior-Løken Syndrome 5 cohort (seed-349)",
        "syndrome":                     "MOST COMMON Senior-Løken Syndrome gene; Severe LCA-like Retinal >> Renal",
        "cohort_n":                     n,
        "median_age_retinal_dx_yr":     round(statistics.median(ages_retinal), 1),
        "median_age_renal_dx_yr":       round(statistics.median(ages_renal), 1),
        "median_gfr_at_dx_ml_min":      int(statistics.median(gfr_dx)),
        "median_urine_osmolality":      int(statistics.median(u_osm)),
        "mean_hgb_g_dl":                round(statistics.mean(hgb), 1),
        "pct_esrd_or_rrt":              pct_rrt,
        "pct_transplanted":             pct_tx,
        "pct_senior_loken":             pct_sls,
        "pct_nystagmus":                pct_nystagmus,
        "pct_consanguineous":           pct_consang,
        "pct_prior_misdiagnosis":       pct_misdiag,
        "pct_lca_misdiagnosis":         pct_lca_misdiag,
        "pct_kidneys_small":            pct_small,
        "pct_legally_blind_or_worse":   pct_blind,
    }

    alerts = {
        "most_common_sls_gene": (
            "IQCB1/NPHP5 is the MOST COMMON Senior-Løken Syndrome gene worldwide (commoner than NPHP4/SLS4). "
            f"{pct_sls}% of cohort have confirmed SLS5 (LCA-like retinal + NPHP renal). "
            "IQCB1 must be included in ALL LCA gene panels, ALL Senior-Løken panels, and ALL NPHP panels."
        ),
        "retinal_dominates_clinically": (
            f"Median age retinal Dx: {round(statistics.median(ages_retinal),1)} yr vs renal Dx: "
            f"{round(statistics.median(ages_renal),1)} yr — RETINAL PRECEDES RENAL in most patients. "
            f"{pct_blind}% of cohort are legally blind or worse at last follow-up. "
            "Ophthalmology team often the first to diagnose; renal function must always be checked."
        ),
        "lca_misdiagnosis_most_common": (
            f"{pct_lca_misdiag}% of cohort were initially misdiagnosed as isolated LCA (retinal-only panel; "
            "renal workup omitted). NPHP5/IQCB1 is the key LCA mimic — ALWAYS check creatinine and "
            "urine osmolality in any child with LCA-phenotype; add IQCB1 to LCA gene panels."
        ),
        "transplant_curative_retinal_not": (
            "Renal transplant is CURATIVE for the renal component — cell-autonomous TZ defect, NO graft recurrence. "
            "HOWEVER: retinal disease does NOT improve post-transplant. Lifelong ophthalmology co-management is MANDATORY. "
            "Living related donors (obligate heterozygotes) are SAFE — carrier renal and retinal function normal."
        ),
        "no_situs_no_chf_no_mts": (
            "SITUS INVERSUS: ABSENT (IQCB1 not expressed in nodal cilia). "
            "CHF (hepatic fibrosis): ABSENT (IQCB1 not expressed in biliary epithelium). "
            "Molar Tooth Sign on MRI: ABSENT — differentiates NPHP5 from Joubert/CEP290 spectrum."
        ),
    }

    key_facts = [
        "IQCB1 / NPHP5 (3q21.1): 590 aa; IQ calmodulin-binding motifs; connects IFT to NPHP-RC; interacts with RPGR",
        "MOST COMMON Senior-Løken Syndrome gene worldwide (NPHP4 is second most common)",
        "Dominant phenotype: SEVERE LCA-like rod-cone retinal dystrophy — visual impairment in first decade",
        "Retinal PRECEDES renal disease in most patients — often diagnosed by ophthalmology first",
        "ESRD median ~13 yr (similar to NPHP1): small kidneys; tubular concentrating defect; TIN",
        "Nystagmus: ~75–80% (early visual impairment; pendular nystagmus from LCA-like onset)",
        "NO situs inversus (IQCB1 not in nodal cilia); NO CHF; NO Molar Tooth Sign",
        "LCA misdiagnosis: MOST COMMON error — retinal panel sent without renal workup; IQCB1 missed",
        "RPGR interaction: NPHP5 overlaps X-linked RP pathway; IQCB1 mandatory in RP panel in males with renal disease",
        "Concentrating defect: Uosm < 300 mosm/kg; polyuria/polydipsia may develop after retinal symptoms",
        "Anaemia: disproportionate for CKD degree (EPO-producing interstitial cell loss)",
        "Renal transplant = CURATIVE for renal; retinal disease does NOT improve post-transplant",
        "Gene therapy: AAV-IQCB1 subretinal approach in pre-clinical research; no approved therapy 2026",
        "WES + CNV analysis: IQCB1 included in standard ciliopathy/NPHP/SLS panels; no single dominant founder",
        "Low-vision rehabilitation from childhood: braille, AT, orientation/mobility; LCA-like severity",
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
        r = p["retinal"].split("(")[0].strip()[:65]
        retinal_dist[r] = retinal_dist.get(r, 0) + 1

    # Ocular motor / nystagmus
    ocular_dist = {}
    for p in cohort:
        o = p["ocular_motor"].split("(")[0].strip()[:60]
        ocular_dist[o] = ocular_dist.get(o, 0) + 1

    # Visual acuity distribution
    va_dist = {}
    for p in cohort:
        v = p["visual_acuity"].split("(")[0].strip()[:50]
        va_dist[v] = va_dist.get(v, 0) + 1

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
        r = p["rrt_or_transplant"].split("—")[0].split("(")[0].strip()[:55]
        rrt_dist[r] = rrt_dist.get(r, 0) + 1

    # Age at retinal Dx tiers
    ret_age_t = {"< 1 yr (neonatal/infantile)": 0, "1–3 yr (early childhood)": 0,
                 "3–6 yr (preschool)": 0, "6–10 yr (school age)": 0, ">10 yr (older child/adolescent)": 0}
    for p in cohort:
        a = p["age_retinal_dx_yr"]
        if a < 1:    ret_age_t["< 1 yr (neonatal/infantile)"] += 1
        elif a < 3:  ret_age_t["1–3 yr (early childhood)"] += 1
        elif a < 6:  ret_age_t["3–6 yr (preschool)"] += 1
        elif a < 10: ret_age_t["6–10 yr (school age)"] += 1
        else:        ret_age_t[">10 yr (older child/adolescent)"] += 1

    # Age at renal Dx tiers
    ren_age_t = {"<8 yr": 0, "8–12 yr": 0, "12–16 yr": 0, ">16 yr": 0}
    for p in cohort:
        a = p["age_renal_dx_yr"]
        if a < 8:    ren_age_t["<8 yr"] += 1
        elif a < 12: ren_age_t["8–12 yr"] += 1
        elif a < 16: ren_age_t["12–16 yr"] += 1
        else:        ren_age_t[">16 yr"] += 1

    # Prior misdiagnosis
    misdiag = {}
    for p in cohort:
        m = p["prior_misdiagnosis"].split("—")[0].strip()[:55]
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
    slope_t = {"1–3 ml/min/yr (slow)": 0, "3–5 ml/min/yr (moderate)": 0, ">5 ml/min/yr (rapid)": 0}
    for p in cohort:
        s = p["gfr_slope_ml_min_yr"]
        if s <= 3:   slope_t["1–3 ml/min/yr (slow)"] += 1
        elif s <= 5: slope_t["3–5 ml/min/yr (moderate)"] += 1
        else:        slope_t[">5 ml/min/yr (rapid)"] += 1

    return {
        "kidney_size_distribution":      dict(sorted(sizes.items(), key=lambda x: -x[1])),
        "retinal_involvement":           dict(sorted(retinal_dist.items(), key=lambda x: -x[1])),
        "ocular_motor_nystagmus":        dict(sorted(ocular_dist.items(), key=lambda x: -x[1])),
        "visual_acuity_distribution":    dict(sorted(va_dist.items(), key=lambda x: -x[1])),
        "ckd_stage_current":             ckd,
        "urine_osmolality_tiers":        u_tiers,
        "rrt_transplant_status":         dict(sorted(rrt_dist.items(), key=lambda x: -x[1])),
        "age_at_retinal_dx_tiers":       ret_age_t,
        "age_at_renal_dx_tiers":         ren_age_t,
        "gfr_slope_tiers":               slope_t,
        "prior_misdiagnosis":            dict(sorted(misdiag.items(), key=lambda x: -x[1])),
        "gene_distribution":             dict(sorted(gene_dist.items(), key=lambda x: -x[1])[:6]),
        "ethnicity":                     dict(sorted(eth.items(), key=lambda x: -x[1])),
    }


def get_definitions():
    return {
        "disease":      "Nephronophthisis Type 5 / Senior-Løken Syndrome 5 (IQCB1/NPHP5 — Most Common SLS Gene; Severe LCA-like Retinal >> Renal)",
        "omim_gene":    "*607526 (IQCB1 — IQ motif-containing B1; NPHP5)",
        "omim_disease": "#609254 (Senior-Løken Syndrome 5) | #611498 (Nephronophthisis 5 — renal-only subset)",
        "chromosome":   "3q21.1",
        "inheritance":  "Autosomal Recessive; biallelic LOF; no single dominant founder",
        "prevalence":   "~1/100,000–1/500,000 live births (commoner than NPHP2–4; rare disease)",
        "mechanism":    (
            "IQCB1 (NPHP5 / IQ motif-containing B1 protein, 590 aa) contains two N-terminal IQ "
            "calmodulin-binding motifs and a C-terminal coiled-coil domain. It bridges calmodulin "
            "Ca²⁺-signalling to IFT machinery at the ciliary transition zone and basal body. "
            "In photoreceptors, NPHP5 localises to the connecting cilium and interacts directly "
            "with RPGR (Retinitis Pigmentosa GTPase Regulator), anchoring IFT particle trafficking "
            "required for outer segment disk renewal. LOF → outer segment disk shedding failure → "
            "photoreceptor degeneration → severe early-onset rod-cone dystrophy (LCA-like phenotype). "
            "In renal tubular epithelium, IQCB1 localises to primary cilia and participates in the "
            "NPHP-RC (nephronophthisis retinitis pigmentosa complex) → TZ dysfunction → "
            "tubulointerstitial nephritis → corticomedullary cysts → progressive ESRD. "
            "Retinal pathology typically dominates clinically (earlier onset, greater disability). "
            "Nystagmus arises secondarily from severe early visual impairment (sensory nystagmus)."
        ),
        "genetic_architecture": {
            "gene":                "IQCB1 (3q21.1); 590 aa; NPHP5; two IQ calmodulin-binding motifs; coiled-coil C-term",
            "mutation_spectrum":   "Heterogeneous: nonsense, splice-site, frameshift dominate (null alleles → severe retinal); "
                                   "missense (p.Ile398Thr) → milder renal disease; large deletions (CNV required)",
            "founder_variants":    "p.Arg461Ter — pan-ethnic recurrent (most common); p.Trp448Ter — pan-ethnic; "
                                   "del exon 3–6 — compound het background; del exon 1–5 — Turkish consanguineous; "
                                   "c.1243+1G>A — recurrent European splice site",
            "diagnostic_strategy": "WES + CNV analysis; IQCB1 in standard SLS/NPHP/ciliopathy gene panels "
                                   "(40–200 gene panels); MLPA not routinely available; large deletions need array CGH or CNV-WES; "
                                   "add to LCA panels for all LCA patients with renal signs",
            "genotype_phenotype":  "Biallelic null (nonsense/fs/splice) → severe LCA-like retinal + renal ESRD ~10–15 yr; "
                                   "biallelic missense (p.Ile398Thr homo) → milder/slower renal disease; "
                                   "retinal severity correlates with degree of RPGR-IQCB1 interaction disruption",
            "rpgr_interaction":    "IQCB1 directly binds RPGR (RP3 gene) at photoreceptor connecting cilium — "
                                   "phenotypic overlap with X-linked RP3; IQCB1 mandatory in males with RP + renal disease",
        },
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)":   "Juvenile ESRD 13yr; mild/absent retinal (10-15%); 290kb del 80%; NO CHF; NO situs",
            "NPHP2 (INVS / 9q31.1)":  "INFANTILE ESRD 3yr; situs inversus 35%; enlarged kidneys; CHF 55%; NO retinal",
            "NPHP3 (NPHP3 / 3q22.1)": "ADOLESCENT ESRD 19yr; situs inversus 15–20%; small kidneys; CHF 45%; NO retinal",
            "NPHP4 (NPHP4 / 1p36.31)": "JUVENILE/ADOLESCENT ESRD 17–20yr; SLS4 ~15–20%; ocular motor apraxia; 2nd most common SLS",
            "NPHP5/IQCB1 (3q21.1)":   "MOST COMMON SLS gene; severe LCA-like retinal >> renal; ESRD ~13yr — THIS DISEASE",
            "NPHP6/CEP290 (12q21.32)": "LCA10 IVS26; Joubert MTS; BBS14; full ciliopathy spectrum; 2nd most common LCA gene",
        },
        "key_clinical_features": {
            "retinal_dominant":       "Severe LCA-like rod-cone dystrophy: visual impairment < 1–5 yr; photophobia; nystagmus; flat/markedly reduced ERG",
            "retinal_before_renal":   "Retinal diagnosis PRECEDES renal diagnosis in most; ophthalmology team often diagnoses first",
            "nystagmus":              "~75–80% of patients: pendular nystagmus from early severe visual impairment (sensory nystagmus)",
            "renal_onset":            "ESRD median ~13 yr (similar to NPHP1); insidious concentrating defect; small kidneys",
            "polyuria_thirst":        "Concentrating defect: Uosm < 300 mosm/kg; polyuria/polydipsia may appear after visual symptoms",
            "kidneys":                "SMALL on USS (echogenic, loss CMD); corticomedullary cysts; NOT enlarged",
            "no_situs_inversus":      "ABSENT — IQCB1 not expressed in nodal cilia",
            "no_chf":                 "ABSENT — IQCB1 not expressed in biliary epithelium",
            "no_molar_tooth_sign":    "MRI: NO Molar Tooth Sign — differentiates from Joubert/CEP290 spectrum",
            "anaemia":                "Disproportionate for CKD (EPO-producing interstitial cell loss); starts at CKD 3",
            "normotension":           "BP normal/low (salt-wasting tubular disease); contrast with ADPKD (HTN dominant)",
            "growth_retardation":     "Short stature — CKD-related GH insensitivity in paediatric patients",
        },
        "diagnostic_criteria": {
            "genetic_gold_std":   "Biallelic pathogenic IQCB1 variants on WES + CNV; SLS/NPHP ciliopathy panel required",
            "clinical_triggers":  "LCA-like retinal dystrophy in child + ANY renal abnormality (creatinine↑, Uosm↓, haematuria) "
                                   "→ IQCB1 sequencing URGENT; add IQCB1 to LCA panels for all cases",
            "ophthalmology":      "ERG (markedly reduced/flat rod and cone responses); OCT (outer nuclear layer thinning/loss); "
                                   "fundus (pale disc; pigmentary retinopathy; narrow vessels); FULL ophthalmology at diagnosis",
            "imaging":            "Renal USS: small kidneys; corticomedullary cysts 1–2 cm; echogenic cortex. "
                                   "Brain MRI: NO Molar Tooth Sign (excludes Joubert). "
                                   "Retinal OCT: outer nuclear layer thinning — early/preclinical in milder alleles",
            "labs":               "Uosm < 300 mosm/kg; low-grade tubular proteinuria; disproportionate anaemia; "
                                   "elevated creatinine for age (may be normal early in disease); LFTs NORMAL (no CHF)",
            "avoid":              "Do NOT diagnose as pure LCA and omit renal workup — NPHP5 retinal + renal BOTH require management. "
                                   "Do NOT mistake for X-linked RP (AR not XL; IQCB1 not RPGR). "
                                   "Do NOT assume Joubert without MTS on MRI.",
        },
        "ddx_table": {
            "NPHP1":                   "Juvenile ESRD 13yr; SMALL kidneys; 290kb del 80%; mild/no retinal; NO LCA-like",
            "NPHP4/SLS4":              "SLS4: rod-cone dystrophy ~15–20%; ocular motor apraxia; NPHP4 is 2nd most common SLS gene",
            "LCA (CRX/RPGRIP1/GUCY2D)": "Retinal-only; NO renal involvement; ERG flat; NO CKD; pure retinal gene panel",
            "CEP290/NPHP6/LCA10":      "IVS26 splice variant → LCA10 (retinal) or Joubert (MTS on MRI); check MRI",
            "RPGR X-linked RP":        "X-linked; males predominantly; NO renal; IQCB1 interacts with RPGR but is AR",
            "Joubert (CEP290/AHI1)":   "Molar Tooth Sign on MRI mandatory; NPHP5 has NO MTS — brain MRI differentiates",
            "ADPKD":                   "AD; adult; HTN; macrocysts; PKD1/PKD2; NO retinal; NO small kidneys; NO LCA-like",
            "Alport":                  "COL4A3/A4/A5; haematuria + proteinuria + sensorineural hearing loss; GBM splitting on EM",
            "BBS":                     "Obesity; polydactyly; cognitive delay; retinal degeneration; BBS genes; NO LCA nystagmus alone",
        },
        "treatment": {
            "renal_transplant":     "DEFINITIVE CURATIVE for renal component — cell-autonomous TZ defect; NO graft recurrence; excellent outcome",
            "retinal_not_cured":    "Retinal disease (LCA-like rod-cone dystrophy) does NOT improve after renal transplant — separate cell-autonomous process",
            "living_donor":         "Obligate heterozygote parents/siblings are SAFE donors — carrier renal and retinal function normal",
            "low_vision_rehab":     "LOW VISION REHABILITATION from diagnosis: braille literacy, AT, orientation/mobility training, visual aids",
            "orientation_mobility": "Early childhood intervention: low-vision specialist, orientation and mobility teacher, sighted guide training",
            "fluid_replacement":    "2–3 L/day fluid (concentrating defect); dehydration accelerates CKD decline",
            "epo":                  "Erythropoietin for disproportionate anaemia (target Hgb 11–12 g/dL in children)",
            "gene_therapy":         "Pre-clinical: AAV-IQCB1 subretinal injection for retinal component; no approved therapy 2026",
            "avoid_nephrotoxins":   "NSAIDs, nephrotoxic contrast, aminoglycosides worsen CKD progression",
            "no_immunosuppression": "Steroids/MMF have NO role in NPHP5 renal disease — will not slow TIN progression",
            "no_dmt_2026":          "No approved disease-modifying therapy 2026 for retinal or renal component; gene therapy pre-clinical",
            "annual_surveillance":  "Annual ERG + OCT + fundus; annual renal USS + Uosm + creatinine; ophthalmology + nephrology co-management lifelong",
        },
        "founder_variants": [
            "p.Arg461Ter (c.1381C>T) — pan-ethnic recurrent; most common IQCB1 allele globally",
            "p.Trp448Ter (c.1344G>A) — pan-ethnic recurrent; nonsense; severe LCA-like retinal",
            "c.1243+1G>A — recurrent European splice site; disrupts exon 10 donor",
            "del exon 3–6 — compound het background; CNV-WES or array CGH required",
            "del exon 1–5 homozygous — Turkish consanguineous families; array CGH required",
            "p.Ile398Thr — biallelic missense; milder renal disease; residual connecting cilium function",
            "c.640+2T>C — recurrent European splice site; exon 5 donor disruption",
            "WES + CNV mandatory — large deletions require CNV analysis; standard WES may miss",
        ],
        "rpgr_interaction_note": (
            "NPHP5 (IQCB1) directly interacts with RPGR at photoreceptor connecting cilia. RPGR is the X-linked "
            "Retinitis Pigmentosa GTPase Regulator gene (RP3; Xp11.23). RPGR anchors IFT particle trafficking "
            "required for disk renewal. NPHP5 bridges RPGR to the NPHP-RC module. Clinical implication: "
            "IQCB1 should be included in gene panels for males with retinal dystrophy (RP phenotype) + any "
            "renal abnormality, even if RPGR testing was non-diagnostic."
        ),
        "prognosis": (
            "Retinal: Progressive rod-cone dystrophy leading to legal blindness in most patients by the end of "
            "the first or second decade. LCA-like severity in biallelic null cases: light perception / count "
            "fingers by early adulthood. Milder alleles (p.Ile398Thr) may preserve residual acuity longer. "
            "Retinal disease does NOT improve after renal transplant. "
            "Renal: ESRD by median ~13 yr (range 8–20 yr); renal transplant is curative for the renal component "
            "with excellent graft outcomes; NO recurrence. "
            "Overall: dual disability (visual + renal); early multidisciplinary care (ophthalmology + nephrology + "
            "low-vision team + genetics) is essential from diagnosis. Long-term QoL determined primarily by "
            "visual impairment and access to low-vision services and timely renal transplant."
        ),
        "cohort_note": (
            "Synthetic cohort · 40 patients · IQCB1/NPHP5 (Senior-Løken Syndrome 5) · seed-349 · generated "
            "for clinical decision-support training · not derived from real patient data."
        ),
    }
