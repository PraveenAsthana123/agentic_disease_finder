"""
Nephronophthisis Type 3 (NPHP3 — Adolescent Nephronophthisis; NPHP3 / Nephrocystin-3)
========================================================================================
Primary Gene : NPHP3 (*608002) — 3q22.1; 1330 aa; Nephrocystin-3; TZ scaffold;
               interacts with NPHP1, NPHP4, RPGRIP1L, MKS1
Disease OMIM : #604387 (Nephronophthisis 3, Adolescent)
Chromosome   : 3q22.1
Inheritance  : Autosomal Recessive (biallelic LOF)
Prevalence   : ~1/200,000–1/500,000 live births (rarer than NPHP1 and NPHP2)

Mechanism
---------
NPHP3 (Nephrocystin-3, 1330 aa) is a scaffold protein at the ciliary transition zone (TZ)
and basal body. It forms a multiprotein complex with Nephrocystin-1 (NPHP1), Nephrocystin-4
(NPHP4), RPGRIP1L, and MKS1, all of which are required for TZ structural integrity.

Key molecular roles:
  1. TZ gate integrity: NPHP3 is a core component of the NPHP1-4 module that maintains
     the diffusion barrier between the cilioplasm and cytoplasm → LOF → ciliary protein
     trafficking failure → loss of Hedgehog and Wnt signal transduction
  2. Planar cell polarity: NPHP3 interacts with PTK7 (Protein Tyrosine Kinase 7) and
     VANGL1 to maintain PCP in renal tubular epithelia → LOF → tubular dilation/cysts
  3. Left-right axis: NPHP3 is expressed in nodal cilia → LOF → situs inversus ~15–20%
     (less penetrant than NPHP2 because nodal expression is lower than INVS)
  4. Hepatic biliary morphogenesis: NPHP3 expressed in biliary epithelium →
     ductal plate malformation → congenital hepatic fibrosis ~45%
  5. Sperm flagella: NPHP3 expressed in flagellar axoneme → male infertility subset

LOF → Adolescent TIN with small kidneys → progressive ESRD median ~19 yr.
Cell-autonomous disease → renal transplant CURATIVE, NO graft recurrence.

Hallmark Features (comparison with NPHP1 and NPHP2):
  • ADOLESCENT onset (ESRD median ~19 yr) — later than NPHP1 (13 yr), much later than NPHP2 (3 yr)
  • Situs inversus ~15–20%: nodal cilia involvement but less penetrant than NPHP2 (~35%)
  • Congenital hepatic fibrosis ~45%: present (contrast NPHP1 where absent)
  • Kidneys SMALL (similar to NPHP1; NOT enlarged as in NPHP2)
  • Male infertility: subset with situs inversus (NPHP3 in flagellar axoneme)
  • No retinal dystrophy (NPHP3 not expressed in photoreceptors)
  • No CNS features (contrast Joubert CEP290)

Key Differentials:
  NPHP1 (NPHP1/2q13): Juvenile 13yr; 290kb deletion 80%; NO situs inversus; ±Senior-Løken
  NPHP2 (INVS/9q31): Infantile 3yr; situs inversus 30–50%; enlarged kidneys; CHF 55%
  NPHP4 (NPHP4/1p36): Juvenile; ocular motor abnormalities; Leber-like retinal in some
  ARPKD (PKHD1): enlarged kidneys; CHF; tubular ectasia; NO situs inversus in majority
  Medullary Cystic Disease type 2 (UMOD): AD; gout; hyperuricaemia; adult onset
  Alport (COL4A3/A4/A5): haematuria; proteinuria; glomerular TBM; hearing loss; X-linked/AR

Treatment:
  • Renal transplant = definitive CURATIVE (cell-autonomous; NO recurrence in graft)
  • Conservative CKD: 2–3 L fluid/day (concentrating defect); avoid nephrotoxins
  • EPO for disproportionate anaemia (interstitial EPO-cell loss)
  • GH therapy for growth retardation in paediatric CKD phase
  • Liver surveillance: CHF → portal HTN (USS + APRI annually)
  • No disease-modifying therapy approved 2026; mTOR inhibitor trials pre-clinical
"""

import random
import statistics

SEED = 345
_RNG = random.Random(SEED)

# ── Genetic pool — realistic NPHP3 alleles (heterogeneous; p.Gln872Ter European enriched) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("NPHP3 (3q22.1) — p.Gln872Ter / p.Arg804Ter (biallelic nonsense; European enriched)",  0.16),
    ("NPHP3 (3q22.1) — c.3253+1G>A / p.Gln872Ter (splice/nonsense compound het)",           0.12),
    ("NPHP3 (3q22.1) — p.Arg804Ter homozygous (South Asian consanguineous)",                 0.10),
    ("NPHP3 (3q22.1) — p.Leu1060Pro / p.Gln872Ter (missense/nonsense compound het)",         0.09),
    ("NPHP3 (3q22.1) — c.2890-2A>G / p.Arg804Ter (splice/nonsense)",                        0.08),
    ("NPHP3 (3q22.1) — del exon 14-17 / p.Gln872Ter (large del compound het)",              0.07),
    ("NPHP3 (3q22.1) — p.Trp1120Ter / p.Arg804Ter (biallelic nonsense)",                    0.07),
    ("NPHP3 (3q22.1) — p.Gln872Ter homozygous (Middle Eastern consanguineous)",              0.06),
    ("NPHP3 (3q22.1) — frameshift c.3987delA / p.Gln872Ter (compound het)",                 0.05),
    ("NPHP3 (3q22.1) — p.Ala1128Val / c.3253+1G>A (missense/splice compound het)",          0.05),
    ("NPHP3 (3q22.1) — del exon 7-9 homozygous (Turkish consanguineous founder region)",     0.05),
    ("NPHP3 (3q22.1) — p.Ser345Phe / p.Leu1060Pro (biallelic missense; milder phenotype)",  0.04),
    ("NPHP3 (3q22.1) — novel / VUS compound het (heterogeneous background)",                 0.06),
]

_ETHNICITY_POOL = [
    ("European (pan-European heterogeneous)",             0.32),
    ("Middle Eastern / Arab (consanguinity enriched)",    0.22),
    ("South Asian (Indian subcontinent)",                 0.16),
    ("Turkish",                                           0.10),
    ("North African (consanguinity enriched)",            0.08),
    ("East Asian",                                        0.06),
    ("African / Sub-Saharan",                             0.04),
    ("Latin American",                                    0.02),
]

_SITUS_POOL = [
    ("Situs solitus (normal laterality)",                 0.76),
    ("Situs inversus totalis",                            0.14),
    ("Situs ambiguus / heterotaxy (partial)",             0.06),
    ("Dextrocardia only (partial situs inversus)",        0.04),
]

_CHF_POOL = [
    ("Absent (no hepatic fibrosis detected)",                    0.38),
    ("Mild (periportal fibrosis on biopsy; normal LFTs)",        0.32),
    ("Moderate (bridging fibrosis; elevated GGT/ALP)",           0.20),
    ("Severe (portal HTN — varices / splenomegaly)",             0.10),
]

_RRT_POOL = [
    ("CKD stage 3–4 (adolescent; under close surveillance)",     0.28),
    ("Renal transplant — living related donor",                  0.24),
    ("Renal transplant — deceased donor",                        0.16),
    ("Haemodialysis (awaiting transplant)",                      0.16),
    ("Peritoneal dialysis (young adult, bridge to transplant)",  0.08),
    ("CKD stage 2 (slow progression; mild polyuria)",            0.08),
]

_MISDIAG_POOL = [
    ("Focal segmental glomerulosclerosis (FSGS) — proteinuria mismatch", 0.28),
    ("ADPKD (incorrect AD assumption; adult team first)",                 0.20),
    ("Alport syndrome (COL4A3/A4 sequenced first)",                       0.16),
    ("Medullary cystic disease (UMOD not found; re-tested)",              0.12),
    ("IgA nephropathy (biopsy-driven; no haematuria rechecked)",          0.10),
    ("Idiopathic CKD (no genetic workup initially)",                      0.10),
    ("No prior misdiagnosis (first genetic Dx correct)",                  0.04),
]

_MALE_INFERTILITY_CHOICES = [
    ("Confirmed male infertility (asthenozoospermia)",         0.18),
    ("Mild sperm motility reduction (borderline)",             0.10),
    ("Not applicable (female) or not yet investigated",        0.72),
]


def _weighted_pick(pool):
    labels  = [p[0] for p in pool]
    weights = [p[1] for p in pool]
    return _RNG.choices(labels, weights=weights, k=1)[0]


def _make_patient(idx):
    gene      = _weighted_pick(_GENE_POOL)
    ethnicity = _weighted_pick(_ETHNICITY_POOL)
    situs     = _weighted_pick(_SITUS_POOL)
    chf       = _weighted_pick(_CHF_POOL)
    rrt       = _weighted_pick(_RRT_POOL)
    misdiag   = _weighted_pick(_MISDIAG_POOL)
    infert    = _weighted_pick(_MALE_INFERTILITY_CHOICES)

    # Adolescent onset — age at diagnosis 8–26 yr (median ~19 yr)
    age_dx   = round(_RNG.betavariate(4, 3) * 18 + 8, 1)  # beta shifted to ~19
    # GFR at Dx — variably reduced (adolescent CKD 2–5)
    gfr_dx   = int(_RNG.betavariate(3, 2) * 70 + 15)      # 15–85
    gfr_now  = max(5, gfr_dx - int(_RNG.betavariate(2, 3) * 45))
    # Urine osmolality — concentrating defect
    u_osm    = int(_RNG.betavariate(2, 5) * 280 + 50)     # 50–330 mosm/kg
    # Hgb — disproportionately anaemic (EPO interstitial loss)
    hgb      = round(8.0 + _RNG.betavariate(2, 3) * 5.5, 1)
    # Kidney size (all SMALL in adolescent NPHP)
    kidney_size = _RNG.choices(
        ["Small (echogenic, loss CMD)", "Normal-to-small", "Normal (early stage)", "Shrunken (ESRD)"],
        weights=[0.38, 0.28, 0.22, 0.12]
    )[0]
    # GFR trend slope (ml/min/yr decline)
    gfr_slope = round(_RNG.betavariate(2, 4) * 8 + 1, 1)  # 1–9 ml/min/yr

    return {
        "id":                    f"NPHP3-{idx:03d}",
        "gene":                  gene,
        "ethnicity":             ethnicity,
        "age_at_diagnosis_yr":   age_dx,
        "gfr_at_dx_ml_min":      gfr_dx,
        "gfr_now_ml_min":        gfr_now,
        "urine_osmolality_mosm": u_osm,
        "hemoglobin_g_dl":       hgb,
        "situs":                 situs,
        "chf_grade":             chf,
        "kidney_size":           kidney_size,
        "gfr_slope_ml_min_yr":   gfr_slope,
        "rrt_or_transplant":     rrt,
        "prior_misdiagnosis":    misdiag,
        "male_infertility":      infert,
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

    ages     = [p["age_at_diagnosis_yr"] for p in cohort]
    gfr_dx   = [p["gfr_at_dx_ml_min"]    for p in cohort]
    u_osm    = [p["urine_osmolality_mosm"] for p in cohort]
    hgb      = [p["hemoglobin_g_dl"]      for p in cohort]

    pct_si      = round(sum(1 for p in cohort if "inversus" in p["situs"].lower()) / n * 100)
    pct_chf     = round(sum(1 for p in cohort if "Absent" not in p["chf_grade"]) / n * 100)
    pct_rrt     = round(sum(1 for p in cohort if "transplant" in p["rrt_or_transplant"].lower()
                            or "dialysis" in p["rrt_or_transplant"].lower()) / n * 100)
    pct_tx      = round(sum(1 for p in cohort if "transplant" in p["rrt_or_transplant"].lower()) / n * 100)
    pct_consang = round(sum(1 for p in cohort if p["consanguineous"]) / n * 100)
    pct_misdiag = round(sum(1 for p in cohort if "No prior" not in p["prior_misdiagnosis"]) / n * 100)
    pct_small   = round(sum(1 for p in cohort if "mall" in p["kidney_size"]
                             or "hrunken" in p["kidney_size"]) / n * 100)

    kpis = {
        "gene":                      "NPHP3 (Nephrocystin-3 / NPHP3)",
        "chromosome":                "3q22.1",
        "inheritance":               "Autosomal Recessive (biallelic LOF)",
        "prevalence":                "~1/200,000–1/500,000",
        "cohort_type":               "40-patient adolescent NPHP3 cohort (seed-345)",
        "syndrome":                  "Adolescent NPHP; ±situs inversus; ±CHF; no retinal",
        "cohort_n":                  n,
        "median_age_dx_yr":          round(statistics.median(ages), 1),
        "median_gfr_at_dx_ml_min":   int(statistics.median(gfr_dx)),
        "median_urine_osmolality":   int(statistics.median(u_osm)),
        "mean_hgb_g_dl":             round(statistics.mean(hgb), 1),
        "pct_esrd_or_rrt":           pct_rrt,
        "pct_transplanted":          pct_tx,
        "pct_situs_inversus":        pct_si,
        "pct_chf":                   pct_chf,
        "pct_consanguineous":        pct_consang,
        "pct_prior_misdiagnosis":    pct_misdiag,
        "pct_kidneys_small":         pct_small,
    }

    alerts = {
        "adolescent_onset_clue": (
            "NPHP3 presents in ADOLESCENCE (ESRD median ~19 yr, range 12–25 yr). "
            "Polyuria + growth retardation + CKD in a teenager with SMALL kidneys → gene panel mandatory. "
            "ADPKD misdiagnosis (incorrect AD assumption) is the most common initial error."
        ),
        "chf_surveillance": (
            f"{pct_chf}% of cohort have congenital hepatic fibrosis — liver USS + APRI annually. "
            "CHF can precede ESRD by years; portal HTN warrants upper GI endoscopy ± prophylactic beta-blocker."
        ),
        "situs_inversus_rarer_than_nphp2": (
            f"Situs inversus present in {pct_si}% of NPHP3 (vs ~35% NPHP2 / 0% NPHP1). "
            "Less penetrant because NPHP3 nodal cilia expression is lower than INVS. "
            "PCD must be excluded (nNO + TEM) — situs inversus without renal disease = PCD first."
        ),
        "fsgs_misdiagnosis": (
            "FSGS is the most common initial biopsy diagnosis — NPHP3 shows tubulointerstitial pattern "
            "but sparse glomerular involvement can be labelled FSGS without genetic testing. "
            "Mandate gene panel for any adolescent FSGS before immunosuppression."
        ),
        "transplant_curative": (
            "Renal transplant is CURATIVE — cell-autonomous TZ defect; NO recurrence in graft. "
            "Living related donors (obligate heterozygotes) are SAFE — normal renal function in carriers."
        ),
    }

    key_facts = [
        "NPHP3 / Nephrocystin-3: 1330 aa TZ scaffold; 3q22.1; interacts with NPHP1, NPHP4, RPGRIP1L, MKS1; AR biallelic LOF",
        "Adolescent onset: ESRD median ~19 yr (vs 13 yr NPHP1 juvenile, 3 yr NPHP2 infantile) — latest onset of NPHP1/2/3",
        "Situs inversus ~15–20%: nodal cilia expression — LESS penetrant than NPHP2 (~35%); absent in NPHP1",
        "Kidneys SMALL on USS (similar to NPHP1) — NOT enlarged as in NPHP2/ARPKD",
        "Congenital hepatic fibrosis ~45%: ductal plate malformation (present in NPHP3, absent in NPHP1)",
        "NO retinal dystrophy — NPHP3 not expressed in photoreceptors; no Senior-Løken phenotype",
        "Genetics: heterogeneous; p.Gln872Ter enriched in Europeans; no dominant deletion founder (cf NPHP1)",
        "Male infertility subset: NPHP3 expressed in sperm flagella → asthenozoospermia in situs inversus males",
        "FSGS misdiagnosis: tubulointerstitial pattern → gene panel mandatory before immunosuppression",
        "GFR decline: ~3–8 ml/min/yr; slower than NPHP2 but inexorable → all progress to ESRD",
        "Concentrating defect: Uosm < 300 mosm/kg; polyuria/polydipsia = first symptom (tubular before glomerular)",
        "mTOR inhibition pre-clinical (targets cystogenic mTORC1 pathway); no approved DMT in 2026",
        "NPHP3 allele severity: null alleles (nonsense/fs) → ESRD ~15–19 yr; missense biallelic → ESRD ~22–28 yr",
        "RPGRIP1L / NPHP3 complex: mutations in this module cause Joubert/MKS/NPHP spectrum — allele decides severity",
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

    # Situs
    situs_dist = {}
    for p in cohort:
        s = p["situs"]
        situs_dist[s] = situs_dist.get(s, 0) + 1

    # CHF
    chf_dist = {}
    for p in cohort:
        c = p["chf_grade"]
        chf_dist[c] = chf_dist.get(c, 0) + 1

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
    age_t = {"<12 yr (early/childhood)": 0, "12–16 yr (early adolescent)": 0,
             "16–22 yr (late adolescent)": 0, ">22 yr (young adult)": 0}
    for p in cohort:
        a = p["age_at_diagnosis_yr"]
        if a < 12:   age_t["<12 yr (early/childhood)"] += 1
        elif a < 16: age_t["12–16 yr (early adolescent)"] += 1
        elif a < 22: age_t["16–22 yr (late adolescent)"] += 1
        else:        age_t[">22 yr (young adult)"] += 1

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
        "situs_distribution":         dict(sorted(situs_dist.items(), key=lambda x: -x[1])),
        "chf_grade_distribution":     dict(sorted(chf_dist.items(), key=lambda x: -x[1])),
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
        "disease":      "Nephronophthisis Type 3 (NPHP3 — Adolescent Nephronophthisis)",
        "omim_gene":    "*608002 (NPHP3 — Nephrocystin-3)",
        "omim_disease": "#604387 (Nephronophthisis 3, Adolescent)",
        "chromosome":   "3q22.1",
        "inheritance":  "Autosomal Recessive; biallelic LOF; no dominant founder",
        "prevalence":   "~1/200,000–1/500,000 live births (rarer than NPHP1 and NPHP2)",
        "mechanism":    (
            "NPHP3 (Nephrocystin-3, 1330 aa) is a scaffolding protein of the ciliary transition zone "
            "(TZ) and basal body. It forms the NPHP1-NPHP3-NPHP4 module (NPHP1:NPHP4 coiled-coil "
            "platform bridged by NPHP3 ankyrin repeats) that maintains the diffusion barrier between "
            "cilioplasm and cytoplasm. LOF → TZ gate collapse → disrupted ciliary protein import/export "
            "→ failure of Hedgehog (Smoothened/Gli), Wnt/PCP and PDGF-Rα signalling in renal tubular "
            "epithelium → tubular dilation, interstitial fibrosis and corticomedullary cysts. NPHP3 "
            "is also required in nodal cilia (left-right axis), biliary epithelium (CHF), and "
            "sperm flagella (male infertility)."
        ),
        "genetic_architecture": {
            "gene":                "NPHP3 (3q22.1); 1330 aa; TZ scaffold; ankyrin/coiled-coil domains",
            "mutation_spectrum":   "Heterogeneous: nonsense, splice-site, frameshift; missense (biallelic milder)",
            "founder_variants":    "p.Gln872Ter — enriched in European cohorts; del exon 7-9 — Turkish regional; "
                                   "p.Arg804Ter — South Asian consanguineous families",
            "diagnostic_strategy": "WES + CNV analysis; NPHP3 included in standard ciliopathy gene panels "
                                   "(NPHP/Joubert/BBS/MKS 40–200 gene panels); MLPA not routinely available",
            "genotype_phenotype":  "Biallelic null (nonsense/fs) → ESRD ~15–19 yr; biallelic missense "
                                   "(p.Ser345Phe/p.Leu1060Pro) → milder; ESRD ~22–28 yr",
            "allele_spectrum":     "NPHP3 alleles can cause NPHP3 (OMIM #604387), Meckel-like lethal "
                                   "(null embryonic — rare), or Joubert-spectrum (hypomorphic) depending "
                                   "on residual function — allele severity determines the ciliopathy tier",
        },
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)":  "Juvenile ESRD 13yr; 290kb 2q13 deletion 80%; NO situs inversus; ±Senior-Løken 12%; NO CHF",
            "NPHP2 (INVS / 9q31.1)": "INFANTILE ESRD 3yr; situs inversus 30–50%; enlarged kidneys; CHF 55%; NO retinal",
            "NPHP3 (NPHP3 / 3q22.1)": "ADOLESCENT ESRD ~19yr — THIS DISEASE; situs inversus 15–20%; small kidneys; CHF ~45%; NO retinal",
            "NPHP4 (NPHP4 / 1p36)":  "Juvenile; ocular motor abnormalities; Leber congenital amaurosis overlap; Senior-Løken rare",
            "NPHP5/IQCB1 (3q21.1)":  "Senior-Løken most common gene; severe LCA-like retinal dystrophy; retinal >> renal",
            "NPHP6/CEP290 (12q21.32)": "Joubert MTS pathognomonic; LCA10 IVS26 allele; BBS14; full ciliopathy spectrum",
        },
        "key_clinical_features": {
            "onset":              "Adolescent: ESRD median ~19 yr (range 12–25 yr); latest of NPHP1/2/3 subtypes",
            "polyuria_first":     "Concentrating defect is the FIRST symptom: Uosm < 300 mosm/kg; polyuria/polydipsia in childhood",
            "growth_retardation": "Short stature — CKD-related GH insensitivity; paediatric patients need GH evaluation",
            "kidneys":            "SMALL on USS (echogenic, loss of corticomedullary differentiation); NOT enlarged",
            "situs_inversus":     "15–20%: nodal cilia expression (less penetrant than NPHP2 ~35%); absent in NPHP1",
            "chf":                "Congenital hepatic fibrosis ~45% (ductal plate malformation); bile duct dilation",
            "portal_htn":         "Portal hypertension in CHF subset → oesophageal varices; USS + APRI annually",
            "anaemia":            "Disproportionate for CKD (EPO-producing interstitial cell loss); starts CKD 3",
            "no_retinal":         "NO retinal dystrophy; no Senior-Løken; NPHP3 not expressed in photoreceptors",
            "male_infertility":   "Asthenozoospermia in subset (NPHP3 expressed in sperm flagellar axoneme)",
            "gfr_decline":        "~3–8 ml/min/yr (slower than NPHP2 but inexorable); CKD 5 in adolescence/early adulthood",
            "normotension":       "BP normal or low (salt-wasting); contrast ADPKD which is hypertensive",
        },
        "diagnostic_criteria": {
            "genetic_gold_std":  "Biallelic pathogenic NPHP3 variants on WES + CNV; standard ciliopathy panel required",
            "clinical_triggers": "Adolescent CKD + small echogenic kidneys + concentrating defect + corticomedullary cysts "
                                  "(±situs inversus ±CHF) → NPHP panel mandatory before any renal biopsy immunosuppression",
            "imaging":           "Renal USS: small kidneys; corticomedullary cysts 1–2 cm (tubular origin); echogenic cortex; "
                                  "liver USS: periportal echogenicity (CHF); bile duct dilation; sinus rhythm/situs on echo",
            "labs":              "Uosm < 300 mosm/kg; tubular proteinuria (beta-2 microglobulin); disproportionate anaemia; "
                                  "elevated creatinine for age; elevated GGT/ALP if CHF",
            "avoid":             "Renal biopsy showing FSGS/tubulointerstitial nephritis → do NOT start immunosuppression "
                                  "before genetic panel; NPHP3 does not respond to steroids/mycophenolate",
        },
        "ddx_table": {
            "NPHP1":          "Juvenile ESRD 13yr; SMALL kidneys; 290kb 2q13 deletion 80%; NO situs inversus; NO CHF; ±SLS",
            "NPHP2":          "INFANTILE ESRD 3yr; ENLARGED kidneys (mimics ARPKD); situs inversus 35%; CHF 55%; NO retinal",
            "ARPKD":          "Enlarged kidneys (tubular ectasia); CHF; PKHD1 gene; typically neonatal/infantile; NO situs",
            "ADPKD":          "AUTOSOMAL DOMINANT; adult onset; HTN; macrocysts (>1 cm); haematuria; PKD1/PKD2",
            "Medullary cystic (UMOD)": "AD; gout/hyperuricaemia; adult onset; UMOD gene; medullary cysts; NOT adolescent",
            "Alport":         "COL4A3/A4/A5 biallelic/XL; haematuria (micro); proteinuria; sensorineural hearing loss; GBM splitting on EM",
            "FSGS":           "Primary FSGS: proteinuria dominant; nephrotic syndrome; biopsy glomerular lesion primary; "
                               "BUT biopsy can be FSGS-like in NPHP3 — genetic panel mandatory before immunosuppression",
            "PCD":            "Situs inversus WITHOUT renal failure in childhood; bronchiectasis; sinusitis; nNO < 77 nL/min; "
                               "DNAH5/DNAI1 dynein defect — respiratory distinguishes from NPHP3",
        },
        "treatment": {
            "renal_transplant":   "DEFINITIVE — cell-autonomous TZ defect; NO recurrence in graft; CURATIVE",
            "living_donor":       "Parental donors (obligate heterozygotes) are SAFE — normal renal function in carriers",
            "fluid_replacement":  "2–3 L/day fluid (concentrating defect); avoid dehydration (accelerates CKD progression)",
            "avoid_nephrotoxins": "NSAIDs, nephrotoxic contrast, aminoglycosides — all worsen CKD in NPHP3",
            "epo":                "Erythropoietin for disproportionate anaemia (target Hgb 11–12 g/dL)",
            "gh_therapy":         "Growth hormone therapy for paediatric patients with CKD-related growth retardation",
            "chf_surveillance":   "Annual liver USS + APRI; upper GI endoscopy + prophylactic propranolol if varices present",
            "no_immunosuppression": "Steroids/MMF/rituximab have NO role — will not slow NPHP3 progression",
            "no_dmt_2026":        "No approved disease-modifying therapy; mTOR inhibitor trials pre-clinical",
        },
        "founder_variants": [
            "p.Gln872Ter (c.2614C>T) — enriched in European cohorts (no single pan-European founder; pan-European heterogeneous)",
            "p.Arg804Ter (c.2410C>T) — South Asian consanguineous families; recurrent across Indian subcontinent",
            "del exon 7-9 homozygous — Turkish founder region; gene panel/CNV required (standard WES may miss)",
            "p.Gln872Ter homozygous — Middle Eastern consanguineous families (Bedouin/Gulf enrichment)",
            "c.3253+1G>A splice site — recurrent in European heterogeneous cohorts",
            "WES + CNV mandatory — NPHP3 MLPA not widely available; large deletions need array CGH or CNV-WES",
        ],
        "prognosis": (
            "ESRD by median ~19 yr (range 12–25 yr); renal transplant is curative with excellent outcomes. "
            "CHF subset (~45%) requires ongoing liver surveillance — portal hypertension can precede ESRD. "
            "Situs inversus does not worsen renal prognosis but complicates surgical procedures. "
            "Neurodevelopment and cognition are NORMAL (no CNS ciliopathy features unlike Joubert). "
            "Biallelic missense patients (e.g., p.Ser345Phe/p.Leu1060Pro) may progress more slowly "
            "(ESRD ~22–28 yr) — genotype helps counsel adolescents on transplant timing. "
            "With timely diagnosis, genetic counselling, and renal transplant, long-term quality of life is excellent."
        ),
        "cohort_note": (
            "Synthetic cohort · 40 patients · NPHP3 (Nephrocystin-3) · seed-345 · generated for clinical "
            "decision-support training · not derived from real patient data."
        ),
    }
