"""
Nephronophthisis Type 2 (NPHP2 — Infantile Nephronophthisis; INVS / Inversin)
===============================================================================
Primary Gene : INVS (*243305) — 9q31.1; 1065 aa; Inversin; IFT-zone boundary scaffold; Wnt switch
Disease OMIM : #602088 (NPHP2 — Nephronophthisis 2, Infantile)
Chromosome   : 9q31.1
Inheritance  : Autosomal Recessive (biallelic LOF); no single dominant founder
Prevalence   : ~1/100,000–1/200,000 live births

Mechanism
---------
INVS (Inversin) localises at the proximal segment of the ciliary axoneme (IFT-zone boundary),
bridging the transition zone scaffold and the IFT-B machinery.

Key molecular roles:
  1. Canonical/Non-canonical Wnt switch: INVS sequesters Dishevelled (DVL) away from the
     canonical Wnt/β-catenin pathway, promoting non-canonical PCP/CE signalling →
     LOF → canonical Wnt UNOPPOSED → cystogenesis
  2. Left-right axis determination: INVS is required in nodal cilia → LOF → random
     laterality determination → situs inversus ~30–50% of NPHP2 patients
  3. Renal tubular morphogenesis: PCP/CE failure → tubular dilation → corticomedullary
     cysts (similar to NPHP1 but earlier onset and more severe)
  4. Sperm flagella: INVS expressed in flagellar axoneme → male infertility

LOF → Infantile TIN with polycystic renal enlargement → early fibrosis → ESRD median 3 yr.
NPHP2 is cell-autonomous → renal transplant CURATIVE, NO graft recurrence.

Hallmark Features (vs NPHP1):
  • INFANTILE onset (ESRD median 3 yr) — NPHP1 JUVENILE (13 yr)
  • Situs inversus 30–50% — absent in NPHP1 (UNIQUE NPHP2 feature)
  • Congenital hepatic fibrosis (mild–moderate, ductal plate malformation) — absent in NPHP1
  • Kidneys ENLARGED in infancy → mimics ARPKD (contrast NPHP1 SMALL kidneys)
  • NO retinal dystrophy (no Senior-Løken; INVS not expressed in photoreceptors)
  • NO neurodevelopmental features (unlike Joubert)

Renal Imaging:
  • Infantile renal enlargement with corticomedullary cysts → may mimic ARPKD on USS
  • Kidneys become small/shrunken at ESRD
  • Loss of corticomedullary differentiation; hyperechogenic cortex
  • Congenital hepatic fibrosis → periportal fibrosis on USS/MRI liver

Genetics:
  • Biallelic LOF (splice-site, nonsense, frameshift) widely distributed across INVS
  • NO dominant founder mutation (contrast NPHP1 290 kb deletion in 80%)
  • Full gene sequencing + del/dup analysis required (WES + CNV; no MLPA shortcut)
  • Gene panel must include INVS; situs inversus + infantile renal failure → NPHP2 top DDx

Key Differentials:
  ARPKD (PKHD1): enlarged kidneys with tubular ectasia, congenital hepatic fibrosis,
    NO situs inversus; NPHP2 kidneys may look identical → use genetics
  NPHP1 (NPHP1 del): JUVENILE onset 13 yr, SMALL kidneys, NO situs inversus
  PCD (DNAH5 etc.): situs inversus + bronchiectasis/sinusitis, NO renal failure young
  Meckel-Gruber (MKS1): uniformly lethal prenatal; massive encephalocele + polydactyly
  Joubert (CEP290): Molar Tooth Sign on MRI, cerebellar features, may have retinal/renal
  ADPKD (PKD1/2): AD, adult-onset, haematuria, HTN — NOT infantile ESRD

Treatment:
  • Renal transplant = definitive and CURATIVE (cell-autonomous; NO recurrence in graft)
  • Conservative CKD: nephrotoxin avoidance (NSAIDs, contrast, aminoglycosides)
  • EPO for disproportionate anaemia
  • Liver surveillance: CHF → portal HTN risk; hepatoportoenterostomy rarely needed
  • Situs inversus: no specific therapy; anaesthesia/surgical positioning awareness
  • No disease-modifying therapy approved 2026; anti-Wnt and anti-cystogenic pre-clinical
"""

import random
import statistics

SEED = 343
_RNG = random.Random(SEED)

# ── Genetic pool — realistic INVS alleles (heterogeneous, no dominant founder) ───────────────
_GENE_POOL = [
    # (allele_label, proportion)
    ("INVS (9q31.1) — c.1442+1G>A / p.Gln481Ter (splice/nonsense compound het)",     0.14),
    ("INVS (9q31.1) — p.Arg826Ter / p.Glu601Ter (biallelic nonsense)",                0.11),
    ("INVS (9q31.1) — c.3019C>T p.Arg1007Ter / c.1442+1G>A (splice compound het)",   0.10),
    ("INVS (9q31.1) — p.Lys868Asn / p.Arg994Ter (missense/nonsense compound het)",    0.09),
    ("INVS (9q31.1) — del exon 8-10 / p.Gln481Ter (large del compound het)",          0.08),
    ("INVS (9q31.1) — p.Glu601Ter homozygous (Bedouin Arab founder region)",           0.07),
    ("INVS (9q31.1) — c.1443-2A>G / p.Arg826Ter (splice/nonsense)",                   0.07),
    ("INVS (9q31.1) — p.Gly519Asp / c.1442+1G>A (missense/splice)",                   0.06),
    ("INVS (9q31.1) — p.Cys784Arg / p.Lys868Asn (biallelic missense)",                0.05),
    ("INVS (9q31.1) — c.2654+2T>C / p.Gln481Ter (splice/nonsense)",                   0.05),
    ("INVS (9q31.1) — del exon 12-14 homozygous (Turkish founder region)",             0.05),
    ("INVS (9q31.1) — p.Arg994Ter homozygous (South Asian consanguineous)",            0.04),
    ("INVS (9q31.1) — frameshift c.2456delC / p.Glu601Ter (compound het)",             0.04),
    ("INVS (9q31.1) — novel / VUS compound het (heterogeneous)",                       0.05),
]

_ETHNICITY_POOL = [
    ("Middle Eastern / Arab (consanguinity enriched)",    0.30),
    ("South Asian (Indian subcontinent)",                 0.18),
    ("European (pan-European heterogeneous)",             0.17),
    ("Turkish",                                           0.10),
    ("North African (consanguinity enriched)",            0.09),
    ("East Asian",                                        0.06),
    ("African / Sub-Saharan",                             0.05),
    ("Latin American",                                    0.03),
    ("Other / Mixed",                                     0.02),
]

_SITUS_POOL = [
    ("Situs solitus (normal)",                         0.58),
    ("Situs inversus totalis (complete reversal)",     0.26),
    ("Situs ambiguus / heterotaxy",                    0.10),
    ("Partial situs inversus (dextrocardia only)",     0.06),
]

_RRT_POOL = [
    ("Renal transplant — living related donor (parental carrier)",       0.36),
    ("Renal transplant — deceased donor",                                0.18),
    ("Haemodialysis (awaiting transplant)",                              0.22),
    ("Peritoneal dialysis (infantile; bridge to transplant)",            0.10),
    ("CKD stage 4–5 (pre-dialysis, conservative management)",           0.09),
    ("CKD stage 3 (progressive, under surveillance)",                    0.05),
]

_MISDIAGNOSIS_POOL = [
    ("ARPKD (enlarged kidneys ± CHF in infancy — most common initial mimic)",  0.40),
    ("ADPKD (wrong inheritance assumed)",                                       0.15),
    ("Meckel-Gruber suspected antenatally (incorrect)",                         0.10),
    ("Unclassified cystic kidney disease",                                      0.14),
    ("Idiopathic infantile renal failure",                                      0.11),
    ("Primary polycystic hepatic disease (CHF-dominant presentation)",          0.07),
    ("No prior misdiagnosis (first genetic diagnosis)",                         0.03),
]


def _weighted_pick(pool):
    labels = [p[0] for p in pool]
    weights = [p[1] for p in pool]
    return _RNG.choices(labels, weights=weights, k=1)[0]


def _make_patient(idx):
    gene      = _weighted_pick(_GENE_POOL)
    ethnicity = _weighted_pick(_ETHNICITY_POOL)
    situs     = _weighted_pick(_SITUS_POOL)
    rrt       = _weighted_pick(_RRT_POOL)
    misdiag   = _weighted_pick(_MISDIAGNOSIS_POOL)

    # INFANTILE onset — age at diagnosis 0.1 – 4.0 yr (median ~1.5 yr)
    age_dx   = round(_RNG.betavariate(2, 4) * 4.0 + 0.1, 1)
    # GFR at Dx — severely reduced (infantile ESRD trajectory)
    gfr_dx   = int(_RNG.betavariate(2, 3) * 60 + 5)    # 5–65
    gfr_now  = max(3, gfr_dx - int(_RNG.betavariate(2, 2) * 30))
    # Urine osmolality — concentrating defect
    u_osm    = int(_RNG.betavariate(2, 4) * 250 + 50)  # 50–300 mosm/kg
    # Hgb — disproportionately anaemic
    hgb      = round(7.0 + _RNG.betavariate(2, 3) * 5.0, 1)
    # CHF grade
    chf_grade = _RNG.choices(
        ["Absent", "Mild (periportal fibrosis)", "Moderate (bridging fibrosis)", "Severe (portal HTN)"],
        weights=[0.18, 0.40, 0.30, 0.12]
    )[0]
    # Kidney size
    kidney_size = _RNG.choices(
        ["Enlarged (infantile)", "Normal-to-enlarged", "Normal", "Small (ESRD)"],
        weights=[0.28, 0.24, 0.22, 0.26]
    )[0]

    return {
        "id":                    f"NPHP2-{idx:03d}",
        "gene":                  gene,
        "ethnicity":             ethnicity,
        "age_at_diagnosis_yr":   age_dx,
        "gfr_at_dx_ml_min":      gfr_dx,
        "gfr_now_ml_min":        gfr_now,
        "urine_osmolality_mosm": u_osm,
        "hemoglobin_g_dl":       hgb,
        "situs":                 situs,
        "chf_grade":             chf_grade,
        "kidney_size":           kidney_size,
        "rrt_or_transplant":     rrt,
        "prior_misdiagnosis":    misdiag,
        "consanguineous":        ethnicity in (
            "Middle Eastern / Arab (consanguinity enriched)",
            "North African (consanguinity enriched)",
            "South Asian (Indian subcontinent)"
        ) and _RNG.random() < 0.65,
    }


_COHORT = [_make_patient(i + 1) for i in range(40)]


def get_overview():
    cohort = _COHORT
    n = len(cohort)

    ages     = [p["age_at_diagnosis_yr"] for p in cohort]
    gfr_dx   = [p["gfr_at_dx_ml_min"]    for p in cohort]
    u_osm    = [p["urine_osmolality_mosm"] for p in cohort]
    hgb      = [p["hemoglobin_g_dl"]      for p in cohort]

    pct_si     = round(sum(1 for p in cohort if "inversus" in p["situs"].lower()) / n * 100)
    pct_chf    = round(sum(1 for p in cohort if p["chf_grade"] != "Absent") / n * 100)
    pct_rrt    = round(sum(1 for p in cohort if "transplant" in p["rrt_or_transplant"].lower()
                           or "dialysis" in p["rrt_or_transplant"].lower()) / n * 100)
    pct_tx     = round(sum(1 for p in cohort if "transplant" in p["rrt_or_transplant"].lower()) / n * 100)
    pct_consang = round(sum(1 for p in cohort if p["consanguineous"]) / n * 100)
    pct_misdiag_arpkd = round(sum(1 for p in cohort
                                  if "ARPKD" in p["prior_misdiagnosis"]) / n * 100)
    pct_enlarged = round(sum(1 for p in cohort if "nfantile" in p["kidney_size"]
                             or "nlarged" in p["kidney_size"]) / n * 100)

    kpis = {
        "gene":               "INVS (Inversin / NPHP2)",
        "chromosome":         "9q31.1",
        "inheritance":        "Autosomal Recessive (biallelic LOF)",
        "prevalence":         "~1/100,000–1/200,000",
        "cohort_type":        "40-patient infantile NPHP2 cohort (seed-343)",
        "syndrome":           "Infantile NPHP; no retinal phenotype; situs inversus subset",
        "cohort_n":           n,
        "median_age_dx_yr":   round(statistics.median(ages), 1),
        "median_gfr_at_dx_ml_min":      int(statistics.median(gfr_dx)),
        "median_urine_osmolality":      int(statistics.median(u_osm)),
        "mean_hgb_g_dl":      round(statistics.mean(hgb), 1),
        "pct_esrd_or_rrt":    pct_rrt,
        "pct_transplanted":   pct_tx,
        "pct_situs_inversus": pct_si,
        "pct_chf":            pct_chf,
        "pct_consanguineous": pct_consang,
        "pct_arpkd_misdiagnosis": pct_misdiag_arpkd,
        "pct_kidneys_enlarged": pct_enlarged,
    }

    alerts = {
        "ARPKD_mimic":    (
            "Infantile enlarged kidneys ± CHF can be indistinguishable from ARPKD on USS — "
            "INVS gene sequencing is mandatory; PKHD1 testing first is insufficient alone."
        ),
        "situs_inversus_clue": (
            "Situs inversus in an infant with renal failure → NPHP2 (INVS) must be excluded; "
            "situs inversus is ABSENT in NPHP1 but present in ~30–50% NPHP2."
        ),
        "no_retinal_phenotype": (
            "NPHP2/INVS does NOT cause retinal dystrophy — no Senior-Løken risk; "
            "routine ERG not required (contrast NPHP1 where ERG is mandatory)."
        ),
        "transplant_curative": (
            "Renal transplant is CURATIVE — cell-autonomous; zero recurrence in graft. "
            "Living related donors (obligate heterozygote parents) are SAFE for donation."
        ),
        "chf_liver_surveillance": (
            f"{pct_chf}% of cohort have congenital hepatic fibrosis — liver USS + APRI annually; "
            "portal HTN (varices, splenomegaly) mandates upper GI endoscopy."
        ),
    }

    key_facts = [
        "NPHP2 / INVS (Inversin): 1065 aa ciliary IFT-zone boundary protein, "
          "Wnt canonical-to-non-canonical switch; 9q31.1; AR biallelic LOF",
        "Infantile onset: ESRD median 3 yr (vs 13 yr NPHP1 juvenile) — earliest ESRD of all NPHP subtypes",
        "Situs inversus 30–50%: nodal cilia dysfunction — UNIQUE feature absent in all other NPHP subtypes",
        "Kidneys enlarged in infancy → mimics ARPKD on USS; becomes small at ESRD",
        "Congenital hepatic fibrosis (ductal plate malformation) in ~50–60% — absent in NPHP1",
        "NO retinal dystrophy (Senior-Løken absent) — INVS not expressed in photoreceptors",
        "Genetics: heterogeneous; no dominant founder (contrast NPHP1 290 kb deletion 80%); "
          "WES + CNV mandatory",
        "Male infertility: INVS expressed in sperm flagella → immotile spermatozoa",
        "Renal transplant: EXCELLENT; cell-autonomous; NO recurrence; parental donors SAFE (obligate hets)",
        "Wnt switch mechanism: INVS sequesters DVL → canonical Wnt UNOPPOSED → cystogenesis on LOF",
        "DDx vs ARPKD: INVS gene sequencing mandatory; situs inversus + CHF present in both; "
          "NPHP2 kidneys larger early but genetics diverge",
        "No disease-modifying therapy 2026; anti-Wnt (CK1ε inhibitors) pre-clinical",
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

    # Kidney size distribution
    sizes = {}
    for p in cohort:
        k = p["kidney_size"]
        sizes[k] = sizes.get(k, 0) + 1

    # Situs distribution
    situs_dist = {}
    for p in cohort:
        s = p["situs"]
        situs_dist[s] = situs_dist.get(s, 0) + 1

    # CHF grade
    chf_dist = {}
    for p in cohort:
        c = p["chf_grade"]
        chf_dist[c] = chf_dist.get(c, 0) + 1

    # CKD stage now (based on gfr_now)
    ckd_stage = {"CKD 5 (ESRD/RRT)": 0, "CKD 4 (15–29)": 0, "CKD 3 (30–44)": 0, "CKD ≤2 (≥45)": 0}
    for p in cohort:
        g = p["gfr_now_ml_min"]
        if g < 15:   ckd_stage["CKD 5 (ESRD/RRT)"] += 1
        elif g < 30: ckd_stage["CKD 4 (15–29)"] += 1
        elif g < 45: ckd_stage["CKD 3 (30–44)"] += 1
        else:        ckd_stage["CKD ≤2 (≥45)"] += 1

    # Urine osmolality tiers
    u_tiers = {"<100 mosm/kg (severe defect)": 0, "100–200": 0, "200–300": 0, ">300 mosm/kg": 0}
    for p in cohort:
        u = p["urine_osmolality_mosm"]
        if u < 100:   u_tiers["<100 mosm/kg (severe defect)"] += 1
        elif u < 200: u_tiers["100–200"] += 1
        elif u < 300: u_tiers["200–300"] += 1
        else:         u_tiers[">300 mosm/kg"] += 1

    # RRT status
    rrt_dist = {}
    for p in cohort:
        r = p["rrt_or_transplant"].split("—")[0].split("(")[0].strip()
        rrt_dist[r] = rrt_dist.get(r, 0) + 1

    # Age at diagnosis tiers
    age_t = {"<1 yr (neonatal/early infantile)": 0, "1–2 yr": 0, "2–3 yr": 0, ">3 yr": 0}
    for p in cohort:
        a = p["age_at_diagnosis_yr"]
        if a < 1:   age_t["<1 yr (neonatal/early infantile)"] += 1
        elif a < 2: age_t["1–2 yr"] += 1
        elif a < 3: age_t["2–3 yr"] += 1
        else:       age_t[">3 yr"] += 1

    # Prior misdiagnosis
    misdiag = {}
    for p in cohort:
        m = p["prior_misdiagnosis"].split("(")[0].strip()[:50]
        misdiag[m] = misdiag.get(m, 0) + 1

    # Gene distribution (first allele tag)
    gene_dist = {}
    for p in cohort:
        tag = p["gene"].split("—")[-1].strip().split("/")[0].strip()[:60]
        gene_dist[tag] = gene_dist.get(tag, 0) + 1

    # Ethnicity
    eth = {}
    for p in cohort:
        e = p["ethnicity"].split("(")[0].strip()
        eth[e] = eth.get(e, 0) + 1

    return {
        "kidney_size_distribution":   dict(sorted(sizes.items(), key=lambda x: -x[1])),
        "situs_distribution":         dict(sorted(situs_dist.items(), key=lambda x: -x[1])),
        "chf_grade_distribution":     dict(sorted(chf_dist.items(), key=lambda x: -x[1])),
        "ckd_stage_at_diagnosis_now": ckd_stage,
        "urine_osmolality_tiers":     u_tiers,
        "rrt_transplant_status":      dict(sorted(rrt_dist.items(), key=lambda x: -x[1])),
        "age_at_diagnosis_tiers":     age_t,
        "prior_misdiagnosis":         dict(sorted(misdiag.items(), key=lambda x: -x[1])),
        "gene_distribution":          dict(sorted(gene_dist.items(), key=lambda x: -x[1])[:6]),
        "ethnicity":                  dict(sorted(eth.items(), key=lambda x: -x[1])),
    }


def get_definitions():
    return {
        "disease":      "Nephronophthisis Type 2 (NPHP2 — Infantile Nephronophthisis)",
        "omim_gene":    "*243305 (INVS — Inversin)",
        "omim_disease": "#602088 (Nephronophthisis 2, Infantile)",
        "chromosome":   "9q31.1",
        "inheritance":  "Autosomal Recessive; biallelic LOF; no dominant founder",
        "prevalence":   "~1/100,000–1/200,000 live births (rarer than NPHP1)",
        "mechanism":    (
            "INVS (Inversin, 1065 aa) localises at the ciliary IFT-zone boundary and proximal axoneme. "
            "It acts as a molecular switch: sequesters Dishevelled (DVL) from the canonical Wnt/β-catenin "
            "pathway → promotes non-canonical PCP/CE signalling → correct planar cell polarity and "
            "tubular elongation. LOF → canonical Wnt unopposed → cystogenesis + TIN. "
            "INVS also required for nodal cilia function (situs determination) and sperm flagella."
        ),
        "genetic_architecture": {
            "gene":                "INVS (9q31.1); 1065 aa; IFT-zone boundary scaffold",
            "mutation_spectrum":   "Heterogeneous: splice-site, nonsense, frameshift; large del rare",
            "founder_mutations":   "No pan-ethnic dominant founder; regional enrichment (Bedouin, Turkish)",
            "diagnostic_strategy": "WES + CNV/del-dup analysis; INVS MLPA not widely available",
            "genotype_phenotype":  "Null alleles → severe infantile ESRD; some missense → slightly later onset",
            "detection":           "Standard WES detects most SNV/indel; CNV calling mandatory for deletions",
        },
        "nphp_comparison": {
            "NPHP1 (NPHP1/2q13)": "Juvenile ESRD 13yr; 290kb deletion 80%; NO situs inversus; ±Senior-Løken 12%",
            "NPHP2 (INVS/9q31)":  "Infantile ESRD 3yr; heterogeneous mutations; situs inversus 30-50%; NO retinal",
            "NPHP3 (NPHP3/3q22)": "Adolescent ESRD ~19yr; hepatic fibrosis ± male infertility",
            "NPHP4 (NPHP4/1p36)": "Juvenile; Leber congenital amaurosis overlap; ocular motility defects",
            "NPHP5/IQCB1":        "Senior-Løken syndrome most common gene; severe retinal (LCA-like)",
            "NPHP6/CEP290":       "Joubert spectrum; Meckel spectrum; LCA10 (IVS26 allele); BBS14",
        },
        "key_clinical_features": {
            "onset":              "Infantile: ESRD by median 3 yr (range 1–5 yr); earliest of all NPHP subtypes",
            "situs_inversus":     "30–50%: nodal cilia dysfunction → random laterality; situs ambiguus 10%",
            "kidneys_infantile":  "Enlarged on USS in infancy with corticomedullary cysts → mimics ARPKD",
            "kidneys_esrd":       "Become small/atrophic as TIN progresses; hyperechogenic cortex",
            "chf":                "Congenital hepatic fibrosis (ductal plate malformation) ~50–60%",
            "portal_htn":         "Portal hypertension → oesophageal varices in CHF subset; USS + APRI surveillance",
            "concentrating_defect": "Urine osmolality < 300 mosm/kg: tubular concentrating failure → polyuria",
            "anaemia":            "Disproportionate for CKD stage (EPO-producing interstitial cell loss)",
            "no_retinal":         "Retina NORMAL; no Senior-Løken; INVS not expressed in photoreceptors",
            "male_infertility":   "Immotile spermatozoa (INVS expressed in sperm flagella axoneme)",
            "normotension":       "Normal BP or slight hypotension (salt wasting) — contrast ADPKD",
        },
        "diagnostic_criteria": {
            "genetic_gold_std":  "Biallelic pathogenic INVS variants confirmed on WES + CNV analysis",
            "clinical_triggers": "Infantile renal failure + situs inversus, OR enlarged cystic kidneys "
                                  "WITHOUT PKHD1 mutation (rule out ARPKD first)",
            "imaging":           "Renal USS: enlarged kidneys with corticomedullary cysts in infancy; "
                                  "liver USS: periportal echogenicity (CHF); cardiac USS: situs",
            "labs":              "Uosm < 300 mosm/kg; tubular proteinuria; disproportionate anaemia; "
                                  "elevated creatinine from infancy",
            "no_ERG_required":   "Electroretinogram NOT routinely indicated (no retinal phenotype)",
        },
        "ddx_table": {
            "ARPKD":       "Enlarged kidneys + CHF — PKHD1 gene; bilateral tubular ectasia on USS; "
                            "NO situs inversus; Dx by PKHD1 sequencing (mandatory before NPHP2 assumed)",
            "NPHP1":       "Juvenile ESRD 13yr; SMALL kidneys (NOT enlarged); NO CHF; NO situs inversus; "
                            "2q13 290kb deletion 80%; ±Senior-Løken",
            "PCD":         "Situs inversus + bronchiectasis/sinusitis; NO renal failure in childhood; "
                            "nNO low; dynein arm defect on TEM; DNAH5/DNAI1 genes",
            "Joubert":     "Molar Tooth Sign MRI pathognomonic; cerebellar vermis aplasia; neonatal "
                            "breathing dysrhythmia; CEP290/AHI1 genes; may have NPHP-like renal",
            "Meckel-Gruber": "Uniformly lethal; encephalocele + polydactyly + cystic kidneys triad",
            "ADPKD":       "Adult onset; AUTOSOMAL DOMINANT; HYPERTENSION; macrocysts; haematuria",
        },
        "treatment": {
            "renal_transplant":  "DEFINITIVE — cell-autonomous; NO recurrence in graft; CURATIVE",
            "living_donor":      "Parental donors (obligate heterozygotes) are SAFE — normal renal function",
            "conservative_ckd":  "Avoid NSAIDs, nephrotoxic contrast, aminoglycosides",
            "epo":               "Erythropoietin for disproportionate anaemia (start early CKD)",
            "chf_surveillance":  "Annual liver USS + APRI; endoscopy if portal HTN signs",
            "situs_planning":    "Surgical/anaesthetic teams must be briefed on situs inversus anatomy",
            "no_dmt_2026":       "No approved disease-modifying therapy; anti-Wnt pre-clinical",
        },
        "founder_variants": [
            "No pan-ethnic dominant founder mutation (contrast NPHP1 290 kb deletion 80%)",
            "Regional enrichment: p.Glu601Ter — Bedouin Arab / Middle Eastern consanguineous families",
            "del exon 12-14 homozygous — Turkish founder region (gene panel required)",
            "c.1442+1G>A splice site — recurrent across European cohorts (no single ethnic cluster)",
            "WES + CNV mandatory — no MLPA shortcut (NPHP1 MLPA does NOT cover INVS at 9q31.1)",
        ],
        "prognosis": (
            "ESRD by median 3 yr (range 1–5 yr) — renal transplant is curative with excellent outcomes. "
            "CHF subset requires ongoing liver surveillance (portal HTN risk). "
            "Situs inversus per se does not worsen prognosis but complicates surgical procedures. "
            "Neurodevelopment and cognition are NORMAL (no CNS involvement unlike Joubert). "
            "With early diagnosis and timely transplant, long-term quality of life is excellent."
        ),
        "cohort_note": (
            "Synthetic cohort · 40 patients · INVS/NPHP2 · seed-343 · generated for clinical "
            "decision-support training · not derived from real patient data."
        ),
    }
