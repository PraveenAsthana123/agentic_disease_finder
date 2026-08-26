"""
Nephronophthisis Type 1 (NPHP1 — Juvenile Nephronophthisis; Senior-Løken Syndrome 1 when + retinal dystrophy)
===============================================================================================================
Primary Gene : NPHP1 (*607100) — 2q13; 732 aa; SH3 + ankyrin repeats + coiled-coil; TZ scaffold
Disease OMIM : #256100 (NPHP1 — Nephronophthisis 1)
               #266900 (SLSN1 — Senior-Løken Syndrome 1, NPHP1 + retinal dystrophy)
Chromosome   : 2q13 (multi-locus: NPHP1 accounts for ~25% of all NPHP subtypes)
Inheritance  : Autosomal Recessive (biallelic LOF); most common: homozygous 290 kb deletion
Prevalence   : ~1/50,000–1/100,000 live births; most common genetic cause of ESRD in children

Mechanism
---------
NPHP1 (Nephrocystin-1) is a TZ-scaffold protein forming the NPHP module (NPHP1–NPHP4–NPHP8/RPGRIP1L)
at the ciliary transition zone and focal adhesion / cell–cell junction complexes.

LOF → TZ gate incompetence in renal tubular epithelial cells → ciliary signalling failure →
tubulointerstitial nephritis (TIN) → tubular atrophy → interstitial fibrosis → corticomedullary
cysts (2–5 mm, NOT macrocysts) → progressive CKD → ESRD by median age 13 years.

NPHP1 is cell-autonomous → renal transplant is CURATIVE, with NO disease recurrence in graft.

Hallmark Biomarkers (early):
  • Urine concentrating defect: max Uosm < 300 mOsm/kg (tubular — earliest manifestation)
  • Salt wasting: renal tubular dysfunction → hyponatremia risk
  • Tubular proteinuria: low-MW (β2-microglobulin, RBP) — NOT glomerular proteinuria
  • Anemia: disproportionately severe for CKD stage (EPO interstitial cell loss)
  • Normotension / slight hypotension (NOT hypertensive — contrast ADPKD early)
  • Kidneys: normal to SMALL size on USS (NOT enlarged — contrast ADPKD/ARPKD)

Renal Imaging (USS / MRI):
  • Corticomedullary cysts (2–5 mm): seen in ~70% at ESRD; may be absent early
  • Loss of corticomedullary differentiation; hyperechogenic cortex
  • Normal to reduced renal size
  • Absent hydronephrosis / collecting system dilatation

Senior-Løken Syndrome 1 (SLSN1):
  • ~10–15% of NPHP1 patients develop tapetoretinal degeneration → Senior-Løken
  • ERG (electroretinogram) mandatory at diagnosis to screen for retinal involvement
  • Retinal dystrophy is early-onset (vs. Leber Congenital Amaurosis which presents at birth)

Genetics:
  • Large homozygous deletion (290 kb) at 2q13 encompassing NPHP1: ~80% of NPHP1 patients
    (may be homozygous or compound het with a point mutation on the other allele)
  • Detected by MLPA, SNP array, del-NPHP1 PCR; Sanger sequencing misses deletion
  • Remainder: biallelic point mutations (~20%)
  • Gene panel must include NPHP1 deletion assay (not covered by WES alone reliably)

Treatment:
  • Renal transplant = CURATIVE (first-line for ESRD; NO recurrence)
  • Conservative CKD: avoid NSAIDs, nephrotoxic contrast, aminoglycosides
  • EPO for disproportionate anemia
  • Sodium chloride supplementation for salt wasting
  • Adequate hydration (polyuria → dehydration risk)
  • No approved disease-modifying therapy 2026; mTOR / cystogenic pathway pre-clinical

Key Differentials:
  ADPKD (PKD1/PKD2): enlarged kidneys, macrocysts, HYPERTENSION, AUTOSOMAL DOMINANT
  ARPKD (PKHD1): neonatal onset, massively enlarged kidneys, congenital hepatic fibrosis
  ADTKD-UMOD/MUC1: adult onset, gout, AD inheritance — NOT childhood
  Alström (ALMS1): CKD + cardiomyopathy + cone-rod dystrophy + insulin resistance
  Bardet-Biedl (BBS genes): CKD + polydactyly + obesity + retinal dystrophy
  Joubert (CEP290, CC2D2A): ESRD + Molar Tooth Sign MRI + cerebellar vermis aplasia
"""

import random
import statistics

SEED = 341
_RNG = random.Random(SEED)

# ── Genetic pool — realistic NPHP1 alleles ────────────────────────────────────
_GENE_POOL = [
    # (allele_label, proportion)
    ("NPHP1 (2q13) — del_2q13_290kb homozygous (European/pan-ethnic founder)",     0.52),
    ("NPHP1 (2q13) — del_2q13_290kb / p.Arg697Cys compound heterozygous",          0.14),
    ("NPHP1 (2q13) — del_2q13_290kb / c.IVS16+1G>A splice compound het",           0.08),
    ("NPHP1 (2q13) — p.Glu802Ter / p.Arg724Ter biallelic point mutations",          0.06),
    ("NPHP1 (2q13) — p.Leu188Pro homozygous (Inuit/Indigenous Canadian founder)",   0.05),
    ("NPHP1 (2q13) — del_2q13_290kb / p.Leu188Pro compound het",                   0.04),
    ("NPHP1 (2q13) — p.Arg697Cys / p.Glu802Ter biallelic missense",                0.04),
    ("NPHP4 (1p36.31) — biallelic LOF (NPHP4 subtype; similar phenotype)",          0.04),
    ("NPHP3 (3q22.1) — biallelic LOF (adolescent NPHP3 subtype)",                  0.03),
]

_ETHNICITIES = [
    ("North European (UK/Scandinavian/German)",    0.38),
    ("South Asian (Indian/Pakistani/Sri Lankan)",  0.18),
    ("Middle Eastern / North African",             0.16),
    ("East Asian",                                 0.07),
    ("South European (Mediterranean)",             0.09),
    ("Inuit / Indigenous Canadian",                0.05),
    ("Sub-Saharan African",                        0.04),
    ("Admixed / Multiethnic",                      0.03),
]

_ESRD_AGE_YR = (6, 8, 9, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 16, 17, 18, 20, 22, 25)

_USS_FINDINGS = [
    ("Normal to small kidneys; echogenic cortex; corticomedullary cysts 2–5 mm", 0.45),
    ("Small echogenic kidneys; cysts visible; loss of CMD",                       0.22),
    ("Normal kidneys early (no cysts yet; concentrating defect only)",            0.18),
    ("Hyperechogenic kidneys; CMD lost; no cysts visible (biopsy-proven TIN)",    0.10),
    ("USS unavailable — MRI T2: cortical thinning + CMcysts",                    0.05),
]

_CKD_STAGE_AT_DX = [
    ("CKD Stage 1–2 (eGFR ≥ 60; tubular symptoms only)",  0.28),
    ("CKD Stage 3 (eGFR 30–59)",                          0.38),
    ("CKD Stage 4 (eGFR 15–29)",                          0.22),
    ("CKD Stage 5 / ESRD (eGFR < 15; on RRT / transplant)", 0.12),
]

_PRESENTING_SYMPTOM = [
    ("Polyuria / polydipsia (urine concentrating defect — earliest feature)", 0.42),
    ("Incidental anaemia on routine bloods (disproportionate for CKD stage)", 0.18),
    ("Nocturia / enuresis (secondary to polyuria)",                           0.14),
    ("Fatigue / growth faltering (CKD symptoms)",                             0.12),
    ("Family screening (affected sibling identified first)",                  0.08),
    ("Hypertension workup (late-stage finding)",                              0.04),
    ("Incidental abnormal renal USS (other indication)",                      0.02),
]

_MISDIAGNOSIS_POOL = [
    "Assumed psychogenic polydipsia (delayed work-up)",
    "UTI / recurrent urinary tract infection",
    "Diabetes insipidus (central — excluded by normal AVP/copeptin)",
    "Type 1 diabetes mellitus (polyuria; excluded by normal glucose/GAD)",
    "IgA nephropathy (biopsy-proven TIN misread)",
    "ADPKD (imaging — no macrocysts, smaller kidneys, NOT AD inheritance)",
    "No misdiagnosis (NBS family / direct gene panel referral)",
]

_TRANSPLANT_OUTCOMES = [
    ("Excellent graft function; no recurrence (NPHP1 cell-autonomous)", 0.55),
    ("Good graft function; donor-related complications (not NPHP1)",    0.20),
    ("Not yet reached ESRD (pre-transplant stage)",                     0.18),
    ("Awaiting transplant (on dialysis)",                               0.05),
    ("Deceased donor transplant; functioning well",                     0.02),
]


def _pick(pool, rng):
    labels, weights = zip(*pool)
    r = rng.random()
    cum = 0.0
    for lbl, w in zip(labels, weights):
        cum += w
        if r < cum:
            return lbl
    return labels[-1]


def _build_cohort(n: int = 40) -> list:
    cohort = []
    for i in range(n):
        rng = random.Random(SEED + i * 7)
        pid = f"NPHP-{SEED}-{i+1:03d}"

        allele   = _pick(_GENE_POOL, rng)
        eth      = _pick(_ETHNICITIES, rng)
        consang  = rng.random() < 0.24

        # Onset / diagnosis ages
        symptom_age  = round(rng.uniform(4.0, 12.0), 1)
        dx_age       = round(symptom_age + rng.uniform(0.5, 4.0), 1)
        esrd_age     = rng.choice(_ESRD_AGE_YR)
        age_now      = round(max(dx_age + rng.uniform(0, 10), dx_age + 0.5), 1)

        # eGFR at diagnosis
        ckd_stage    = _pick(_CKD_STAGE_AT_DX, rng)
        egfr_at_dx   = {
            "CKD Stage 1–2": round(rng.uniform(65, 105), 1),
            "CKD Stage 3":   round(rng.uniform(32, 58),  1),
            "CKD Stage 4":   round(rng.uniform(16, 29),  1),
            "CKD Stage 5":   round(rng.uniform(3, 14),   1),
        }.get(ckd_stage.split("(")[0].strip().replace("CKD Stage 1–2", "CKD Stage 1–2").split(" /")[0].strip(), round(rng.uniform(20, 60), 1))

        # Actually pull eGFR from ckd_stage string
        if "Stage 1–2" in ckd_stage:
            egfr_at_dx = round(rng.uniform(65, 105), 1)
        elif "Stage 3" in ckd_stage:
            egfr_at_dx = round(rng.uniform(32, 58), 1)
        elif "Stage 4" in ckd_stage:
            egfr_at_dx = round(rng.uniform(16, 29), 1)
        else:
            egfr_at_dx = round(rng.uniform(3, 14), 1)

        # Maximum urine osmolality (Uosm) — hallmark tubular defect
        # Normal > 800 mOsm/kg; NPHP hallmark: < 300 mOsm/kg
        uosm_max = round(rng.uniform(80, 290), 0) if rng.random() < 0.85 else round(rng.uniform(290, 420), 0)

        # Hemoglobin (disproportionately low for CKD stage)
        hb_g_dl = round(rng.uniform(6.5, 10.5), 1)

        # Blood pressure (normotensive or mild hypotension early)
        sbp = round(rng.uniform(90, 125), 0)
        dbp = round(rng.uniform(55, 82), 0)

        # Tubular proteinuria (low-MW; NOT heavy proteinuria)
        upcr_mg_mmol = round(rng.uniform(8, 55), 1)  # tubular range (NOT nephrotic)

        # Sodium — may be low (salt wasting)
        serum_na = round(rng.uniform(130, 140), 1) if rng.random() < 0.35 else round(rng.uniform(137, 143), 1)

        # Renal USS
        uss_finding  = _pick(_USS_FINDINGS, rng)

        # Senior-Løken (SLSN1): ~12% NPHP1
        slsn = rng.random() < 0.12

        # Situs inversus: RARE in NPHP1 specifically (associated with INVS/NPHP2 infantile)
        situs = rng.random() < 0.03

        # Transplant
        transplanted   = age_now > esrd_age and rng.random() < 0.85
        tx_outcome     = _pick(_TRANSPLANT_OUTCOMES, rng) if transplanted else "Not yet at ESRD / pre-dialysis"
        years_since_tx = round(age_now - esrd_age, 1) if transplanted and age_now > esrd_age else None

        # Presenting symptom
        presenting     = _pick(_PRESENTING_SYMPTOM, rng)

        # Prior misdiagnosis
        prior_dx       = rng.choice(_MISDIAGNOSIS_POOL)

        # Family history (AR — siblings)
        affected_siblings = rng.randint(0, 2) if consang else (1 if rng.random() < 0.20 else 0)

        cohort.append({
            "id":                    pid,
            "allele":                allele,
            "ethnicity":             eth,
            "consanguineous":        consang,
            "symptom_onset_age_yr":  symptom_age,
            "dx_age_yr":             dx_age,
            "esrd_age_yr_projected": esrd_age,
            "age_now_yr":            age_now,
            "ckd_stage_at_dx":       ckd_stage,
            "egfr_at_dx_ml_min":     egfr_at_dx,
            "uosm_max_mosm_kg":      uosm_max,
            "hb_g_dl":               hb_g_dl,
            "sbp_mmhg":              sbp,
            "dbp_mmhg":              dbp,
            "upcr_mg_mmol":          upcr_mg_mmol,
            "serum_na_mmol_l":       serum_na,
            "uss_finding":           uss_finding,
            "senior_loken_slsn":     slsn,
            "situs_inversus":        situs,
            "transplanted":          transplanted,
            "transplant_outcome":    tx_outcome,
            "years_since_transplant": years_since_tx,
            "presenting_symptom":    presenting,
            "prior_misdiagnosis":    prior_dx,
            "affected_siblings":     affected_siblings,
        })

    return cohort


# ── Public API ─────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    cohort = _build_cohort()
    n = len(cohort)

    egfrs    = [p["egfr_at_dx_ml_min"] for p in cohort]
    uosms    = [p["uosm_max_mosm_kg"] for p in cohort]
    hbs      = [p["hb_g_dl"] for p in cohort]
    dx_ages  = [p["dx_age_yr"] for p in cohort]

    pct_slsn      = round(sum(1 for p in cohort if p["senior_loken_slsn"]) / n * 100, 1)
    pct_tx        = round(sum(1 for p in cohort if p["transplanted"]) / n * 100, 1)
    pct_consang   = round(sum(1 for p in cohort if p["consanguineous"]) / n * 100, 1)
    pct_situs     = round(sum(1 for p in cohort if p["situs_inversus"]) / n * 100, 1)
    pct_polyuria  = round(sum(1 for p in cohort if "polyuria" in p["presenting_symptom"].lower() or "polydipsia" in p["presenting_symptom"].lower()) / n * 100, 1)
    pct_del290    = round(sum(1 for p in cohort if "del_2q13_290kb" in p["allele"]) / n * 100, 1)
    pct_uosm_low  = round(sum(1 for p in cohort if p["uosm_max_mosm_kg"] < 300) / n * 100, 1)
    pct_esrd_teen = round(sum(1 for p in cohort if p["esrd_age_yr_projected"] <= 15) / n * 100, 1)
    pct_anemia_severe = round(sum(1 for p in cohort if p["hb_g_dl"] < 9.0) / n * 100, 1)

    kpis = {
        "cohort_n":              n,
        "cohort_type":           "Paediatric + young adult NPHP registry (retrospective)",
        "gene":                  "NPHP1 (2q13) — 290 kb deletion most common (~80% of NPHP1); 25% of all NPHP",
        "syndrome":              "Nephronophthisis 1 (NPHP1); Senior-Løken Syndrome 1 when + retinal dystrophy",
        "chromosome":            "2q13 (NPHP1); multi-locus ciliopathy (NPHP1–NPHP4–NPHP3 commonest)",
        "inheritance":           "Autosomal Recessive (biallelic LOF; often homozygous 290 kb deletion)",
        "prevalence":            "~1/50,000–1/100,000 (most common genetic cause of ESRD in children)",
        "median_dx_age_yr":      round(statistics.median(dx_ages), 1),
        "mean_egfr_at_dx":       round(statistics.mean(egfrs), 1),
        "median_uosm_max":       round(statistics.median(uosms), 1),
        "mean_hb_g_dl":          round(statistics.mean(hbs), 1),
        "pct_senior_loken":      pct_slsn,
        "pct_transplanted":      pct_tx,
        "pct_consanguineous":    pct_consang,
        "pct_situs_inversus":    pct_situs,
        "pct_polyuria_present":  pct_polyuria,
        "pct_del290kb":          pct_del290,
        "pct_uosm_under_300":    pct_uosm_low,
        "pct_esrd_by_age_15":    pct_esrd_teen,
        "pct_severe_anemia":     pct_anemia_severe,
    }

    key_facts = [
        "NPHP1 is the most common genetic cause of ESRD in children — autosomal recessive tubulointerstitial nephropathy",
        "~80% of NPHP1 patients carry the homozygous 290 kb deletion at 2q13 — NOT detectable by standard Sanger/WES alone; MLPA or SNP array required",
        f"Urine concentrating defect (Uosm < 300 mOsm/kg) is the EARLIEST biomarker; {pct_uosm_low}% of cohort below threshold",
        f"Polyuria/polydipsia is the presenting symptom in {pct_polyuria}% — frequently misattributed to diabetes insipidus or diabetes mellitus",
        f"ESRD by median age 13 years; {pct_esrd_teen}% projected ESRD by age 15 years",
        "Kidneys are NORMAL TO SMALL (NOT enlarged — distinguishes from ADPKD); corticomedullary cysts 2–5 mm in ~70% at ESRD",
        "Normotensive / slightly hypotensive early — NOT hypertensive like ADPKD; salt wasting causes low-normal BP",
        f"Anemia disproportionate to CKD stage (EPO interstitial cell loss); {pct_anemia_severe}% Hb < 9 g/dL at diagnosis",
        f"Senior-Løken Syndrome 1 (SLSN1): {pct_slsn}% of NPHP1 patients develop tapetoretinal dystrophy — ERG mandatory at diagnosis",
        "Renal transplant is CURATIVE: NO disease recurrence in graft (NPHP1 is cell-autonomous); outcomes EXCELLENT",
        f"Situs inversus: only {pct_situs}% in NPHP1 (rare) — contrast NPHP2/INVS (infantile NPHP with situs inversus ~25%)",
    ]

    alerts = {
        "concentrating_defect":  "Max Uosm < 300 mOsm/kg in any child with CKD → NPHP1 deletion assay (MLPA/SNP array) MANDATORY alongside gene panel",
        "deletion_assay":        "Standard WES / Sanger MISSES the 290 kb NPHP1 deletion — always add MLPA or chromosomal microarray to the NPHP1 work-up",
        "retinal_screening":     "ERG at diagnosis mandatory in ALL NPHP1 patients to screen for Senior-Løken Syndrome 1 (SLSN1, 10–15%)",
        "transplant_curative":   "ESRD in NPHP1 → renal transplant is curative with NO disease recurrence; pre-emptive living-donor transplant preferred",
        "salt_hydration":        "Salt wasting + polyuria → hyponatremia and dehydration risk with febrile illness; sodium supplementation and hydration plan mandatory",
    }

    return {
        "kpis":      kpis,
        "key_facts": key_facts,
        "alerts":    alerts,
        "patients":  cohort[:8],
    }


def get_breakdown() -> dict:
    cohort = _build_cohort()
    n = len(cohort)

    # Allele/gene distribution
    allele_dist: dict = {}
    for p in cohort:
        key = p["allele"].split("(")[0].strip()[:65]
        allele_dist[key] = allele_dist.get(key, 0) + 1

    # CKD stage at diagnosis
    ckd_dist: dict = {}
    for p in cohort:
        key = p["ckd_stage_at_dx"].split("(")[0].strip()[:55]
        ckd_dist[key] = ckd_dist.get(key, 0) + 1

    # USS finding
    uss_dist: dict = {}
    for p in cohort:
        key = p["uss_finding"].split(";")[0].strip()[:60]
        uss_dist[key] = uss_dist.get(key, 0) + 1

    # Presenting symptom
    pres_dist: dict = {}
    for p in cohort:
        key = p["presenting_symptom"].split("(")[0].strip()[:60]
        pres_dist[key] = pres_dist.get(key, 0) + 1

    # Prior misdiagnosis
    misdx_dist: dict = {}
    for p in cohort:
        key = p["prior_misdiagnosis"].split("(")[0].strip()[:60]
        misdx_dist[key] = misdx_dist.get(key, 0) + 1

    # Ethnicity
    eth_dist: dict = {}
    for p in cohort:
        eth_dist[p["ethnicity"]] = eth_dist.get(p["ethnicity"], 0) + 1

    # eGFR tiers at diagnosis
    egfr_tiers = {
        "≥ 60 mL/min (CKD 1–2; early)":      sum(1 for p in cohort if p["egfr_at_dx_ml_min"] >= 60),
        "30–59 mL/min (CKD 3)":               sum(1 for p in cohort if 30 <= p["egfr_at_dx_ml_min"] < 60),
        "15–29 mL/min (CKD 4)":               sum(1 for p in cohort if 15 <= p["egfr_at_dx_ml_min"] < 30),
        "< 15 mL/min (CKD 5 / ESRD)":         sum(1 for p in cohort if p["egfr_at_dx_ml_min"] < 15),
    }

    # Uosm tiers
    uosm_tiers = {
        "< 150 mOsm/kg (severe defect — classic NPHP)": sum(1 for p in cohort if p["uosm_max_mosm_kg"] < 150),
        "150–299 mOsm/kg (moderate defect)":            sum(1 for p in cohort if 150 <= p["uosm_max_mosm_kg"] < 300),
        "300–500 mOsm/kg (mild defect)":                sum(1 for p in cohort if 300 <= p["uosm_max_mosm_kg"] < 500),
        "≥ 500 mOsm/kg (near-normal — early/mild)":    sum(1 for p in cohort if p["uosm_max_mosm_kg"] >= 500),
    }

    # Hb tiers
    hb_tiers = {
        "< 8 g/dL (severe anaemia)":         sum(1 for p in cohort if p["hb_g_dl"] < 8.0),
        "8.0–9.9 g/dL (moderate anaemia)":   sum(1 for p in cohort if 8.0 <= p["hb_g_dl"] < 10.0),
        "10.0–11.9 g/dL (mild anaemia)":     sum(1 for p in cohort if 10.0 <= p["hb_g_dl"] < 12.0),
        "≥ 12 g/dL (minimal / none)":        sum(1 for p in cohort if p["hb_g_dl"] >= 12.0),
    }

    # Age at diagnosis tiers
    age_tiers = {
        "< 6 yr (early childhood)":           sum(1 for p in cohort if p["dx_age_yr"] < 6),
        "6–10 yr (school age — modal)":        sum(1 for p in cohort if 6 <= p["dx_age_yr"] < 11),
        "11–15 yr (pre-teen / early teen)":    sum(1 for p in cohort if 11 <= p["dx_age_yr"] < 16),
        "≥ 16 yr (late diagnosis)":            sum(1 for p in cohort if p["dx_age_yr"] >= 16),
    }

    # Transplant distribution
    tx_dist = {
        "Transplanted (no recurrence)":        sum(1 for p in cohort if p["transplanted"]),
        "Not yet at ESRD (pre-transplant)":    sum(1 for p in cohort if not p["transplanted"]),
    }

    return {
        "allele_distribution":    allele_dist,
        "ckd_stage_at_dx":        ckd_dist,
        "uss_finding":            uss_dist,
        "presenting_symptom":     pres_dist,
        "prior_misdiagnosis":     misdx_dist,
        "ethnicity":              eth_dist,
        "egfr_tiers":             egfr_tiers,
        "uosm_tiers":             uosm_tiers,
        "hb_tiers":               hb_tiers,
        "age_at_diagnosis_tiers": age_tiers,
        "transplant_status":      tx_dist,
        "summary": {
            "n":                  n,
            "pct_del290kb":       round(sum(1 for p in cohort if "del_2q13_290kb" in p["allele"]) / n * 100, 1),
            "pct_senior_loken":   round(sum(1 for p in cohort if p["senior_loken_slsn"]) / n * 100, 1),
            "pct_transplanted":   round(sum(1 for p in cohort if p["transplanted"]) / n * 100, 1),
            "pct_uosm_under_300": round(sum(1 for p in cohort if p["uosm_max_mosm_kg"] < 300) / n * 100, 1),
            "mean_egfr_at_dx":    round(statistics.mean(p["egfr_at_dx_ml_min"] for p in cohort), 1),
            "mean_hb_g_dl":       round(statistics.mean(p["hb_g_dl"] for p in cohort), 1),
            "median_uosm":        round(statistics.median(p["uosm_max_mosm_kg"] for p in cohort), 1),
            "pct_consanguineous": round(sum(1 for p in cohort if p["consanguineous"]) / n * 100, 1),
        },
    }


def get_definitions() -> dict:
    return {
        "disease":       "Nephronophthisis Type 1 (NPHP1; Juvenile Nephronophthisis)",
        "omim_gene":     "NPHP1 *607100 (2q13; Nephrocystin-1; 732 aa; SH3 + ankyrin repeats + coiled-coil)",
        "omim_disease":  "#256100 (Nephronophthisis 1) · #266900 (Senior-Løken Syndrome 1, NPHP1 + retinal dystrophy)",
        "chromosome":    "2q13 (NPHP1); NPHP4 at 1p36.31; NPHP3 at 3q22.1",
        "inheritance":   "Autosomal Recessive (biallelic LOF; most common: homozygous 290 kb deletion 2q13)",
        "prevalence":    "~1/50,000–1/100,000; most common genetic cause of ESRD in children worldwide",
        "mechanism": (
            "NPHP1 (Nephrocystin-1) is a TZ-scaffold protein forming the NPHP module (NPHP1–NPHP4–NPHP8/RPGRIP1L) "
            "at the ciliary transition zone of renal tubular epithelial cells and at focal adhesion / cell–cell junction complexes. "
            "LOF → TZ gate incompetence → ciliary signalling failure → progressive tubulointerstitial nephritis (TIN) → "
            "tubular atrophy → interstitial fibrosis → corticomedullary cysts (2–5 mm) → CKD → ESRD median age 13 yr. "
            "Disease is cell-autonomous → renal transplant is CURATIVE with NO graft recurrence."
        ),
        "key_nphp1_deletion": (
            "The hallmark NPHP1 allele is a ~290 kb homozygous deletion at chromosome 2q13 encompassing the "
            "entire NPHP1 gene. Present in ~80% of NPHP1 patients (homozygous or compound het with a point mutation). "
            "Standard Sanger sequencing and WES frequently MISS this deletion — MLPA, SNP array, or del-NPHP1 PCR "
            "assay is mandatory for complete NPHP1 genetic testing. This deletion cannot be detected by WES alone."
        ),
        "hallmark_biomarkers": {
            "urine_concentrating_defect": "Max Uosm < 300 mOsm/kg — EARLIEST and most specific tubular defect; precedes creatinine rise",
            "tubular_proteinuria":        "Low-MW proteinuria (β2-microglobulin, RBP, α1-microglobulin) — NOT heavy (nephrotic) glomerular proteinuria",
            "disproportionate_anaemia":   "Hb disproportionately low for CKD stage — EPO-producing interstitial cells lost in TIN",
            "salt_wasting":               "Renal tubular Na+ wasting → hyponatremia risk; sodium supplementation required",
            "normotension":               "Normotensive / slightly hypotensive early — contrasts with ADPKD hypertension; important DDx clue",
            "small_kidneys_on_USS":       "Normal to small renal size — critical differentiator from ADPKD (enlarged) and ARPKD (massively enlarged)",
            "corticomedullary_cysts":     "Cysts 2–5 mm at CMD junction in ~70% at ESRD; may be absent early — absence does NOT exclude NPHP1",
        },
        "clinical_course": {
            "symptom_onset":   "Polyuria/polydipsia typically age 4–10 yr (tubular concentrating failure — earliest symptom)",
            "esrd_timing":     "ESRD median age 13 yr for NPHP1 (range 6–25 yr); NPHP2 (INVS): infantile; NPHP3: adolescent",
            "ckd_progression": "Progressive CKD; eGFR decline ~4–6 mL/min/1.73m²/year; pre-emptive transplant preferred",
            "normotension":    "BP normal to low until very late — absence of hypertension distinguishes from ADPKD",
            "diagnosis_delay": "Mean diagnostic delay 2–4 yr from symptom onset — frequent misdiagnosis as DI, T1D, or UTI",
        },
        "senior_loken_syndrome": {
            "definition":        "Senior-Løken Syndrome 1 (SLSN1) = NPHP1 + tapetoretinal dystrophy (retinal involvement)",
            "frequency":         "~10–15% of NPHP1 patients develop SLSN1",
            "retinal_features":  "Early-onset rod-cone dystrophy; nyctalopia (night blindness); progressive visual loss",
            "erv_mandatory":     "ERG (electroretinogram) MANDATORY at diagnosis in ALL NPHP1 patients — retinal involvement changes management",
            "vs_lca":            "SLSN1 onset later than Leber Congenital Amaurosis (LCA is birth-onset); CEP290 allele severity determines LCA vs JBTS vs SLSN",
            "vs_bbs":            "BBS has retinal dystrophy + polydactyly + obesity (BBSome); SLSN1 = pure NPHP1 + retina (no polydactyly/obesity)",
        },
        "genetics": {
            "nphp1_deletion":     "290 kb homozygous deletion 2q13 — ~80% of NPHP1; founder in European populations; NOT detectable by WES alone",
            "point_mutations":    "p.Arg697Cys, p.Glu802Ter, p.Arg724Ter, c.IVS16+1G>A, p.Leu188Pro (Inuit founder) — compound het with deletion or biallelic",
            "nphp4":              "NPHP4 (1p36.31): ~5% of all NPHP; biochemically and clinically similar to NPHP1; gene panel mandatory",
            "nphp3":              "NPHP3 (3q22.1): adolescent onset; sometimes with situs inversus / NPHP2-like features",
            "panel_requirement":  "Single-gene NPHP1 deletion testing NOT sufficient; full NPHP multi-gene panel (≥20 genes) recommended",
            "deletion_detection": "MLPA (first-line) or SNP-array or del-NPHP1 PCR assay; always add to WES in suspected NPHP",
        },
        "treatment": {
            "renal_transplant":      "CURATIVE — renal transplant has EXCELLENT outcomes; NO disease recurrence in graft (NPHP1 is cell-autonomous); pre-emptive living-donor preferred",
            "avoid_nephrotoxins":    "NSAIDs ABSOLUTE CI; iodinated contrast — pre-hydrate + N-acetylcysteine; aminoglycosides — avoid; renally-dose adjust ALL drugs",
            "epo_therapy":           "Erythropoietin-stimulating agent (ESA) for anaemia disproportionate to CKD stage; target Hb 10–12 g/dL",
            "salt_supplementation":  "Sodium chloride tablets / solution for salt wasting; especially important with febrile illness",
            "fluid_intake":          "Adequate oral fluid intake (polyuria → dehydration risk); mandatory written hydration plan for febrile illness",
            "BP_management":         "RAAS blockade if hypertension develops; avoid aggressive BP lowering (normotensive baseline)",
            "no_disease_modifying":  "No approved disease-modifying therapy 2026; mTOR pathway and cystogenesis research pre-clinical",
        },
        "key_differentials": {
            "ADPKD (PKD1/PKD2)":     "ENLARGED kidneys + macrocysts; AUTOSOMAL DOMINANT; hypertension; hematuria; progressive macrocysts — OPPOSITE of NPHP1",
            "ARPKD (PKHD1)":         "Neonatal-onset; massively enlarged kidneys; congenital hepatic fibrosis; oligohydramnios — peripartum presentation, NOT childhood",
            "ADTKD-UMOD/MUC1":       "Adult onset; gout (UMOD); AUTOSOMAL DOMINANT; no childhood ESRD; fibrosis pattern similar but genetics/onset distinct",
            "Alström Syndrome":      "CKD + DCM + cone-rod dystrophy + obesity + insulin resistance (ALMS1 ciliopathy); NO corticomedullary cysts",
            "Bardet-Biedl Syndrome": "CKD + polydactyly + obesity + retinal dystrophy (BBSome); NO TIN pattern; NPHP1 no polydactyly",
            "Joubert Syndrome":      "ESRD + Molar Tooth Sign on MRI + cerebellar vermis aplasia (CEP290, CC2D2A); NPHP1 NO MRI brain abnormality",
            "NPHP2 (INVS)":          "Infantile nephronophthisis; NPHP2 (inversin/INVS); situs inversus ~25% (unlike NPHP1 ~3%); earlier ESRD (age 1–3 yr)",
        },
        "prognosis": (
            "NPHP1 is NOT lethal in childhood (contrast Meckel-Gruber). Renal transplant at ESRD is CURATIVE with excellent "
            "graft survival and NO disease recurrence. Pre-emptive living-donor transplant is optimal (avoid dialysis in children). "
            "With transplant, life expectancy is normal. Without transplant, ESRD median age 13 yr is fatal without RRT. "
            "Senior-Løken subtype (SLSN1): progressive visual loss managed with low-vision aids; no retinal treatment available 2026. "
            "Diagnostic delay is the key modifiable factor — earlier diagnosis enables pre-emptive transplant planning."
        ),
        "cohort_note": f"Synthetic cohort (seed={SEED}, n=40) — epidemiological proportions match published literature (Hildebrandt et al. 2009 NEJM; Wolf et al. 2013 Kidney Int; Braun et al. 2016 Nature Reviews Nephrology).",
    }
