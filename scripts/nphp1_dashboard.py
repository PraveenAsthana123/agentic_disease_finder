"""
Nephronophthisis Type 1 (NPHP1 — Juvenile Nephronophthisis; Nephrocystin-1)
=============================================================================
Primary Gene : NPHP1 (*607100) — 2q13; 736 aa; Nephrocystin-1; TZ Y-link scaffold (NPHP1-4-8 supercomplex)
Disease OMIM : #256100 (Nephronophthisis 1, Juvenile) + #266900 (Senior-Løken Syndrome 1 / SLS1)
Chromosome   : 2q13
Inheritance  : Autosomal Recessive (biallelic LOF); 290 kb deletion is dominant founder allele (66-80%)
Prevalence   : ~1/50,000–1/200,000 (most common cause of inherited ESRD in children)

Mechanism
---------
NPHP1 (Nephrocystin-1, 736 aa) localises at the transition zone (TZ) Y-links and at
adherens junctions of renal tubular epithelial cells.

Key molecular roles:
  1. TZ scaffold: Core subunit of the NPHP1-NPHP4-NPHP8/RPGRIP1L supercomplex at the
     ciliary transition zone; bridges the axoneme doublets to the ciliary membrane via Y-links.
     LOF → TZ barrier defects → aberrant signal transduction (Hh, Wnt, PDGF) → cystogenesis
  2. Focal adhesion / cell polarity: NPHP1 interacts with PTK2/FAK, Paxillin, TENSIN at
     cell-matrix junctions → LOF → tubular cell polarity loss → TIN initiation
  3. DNA damage response: NPHP1 interacts with ATM at the centrosome → LOF → impaired
     DDR → accelerated apoptosis of tubular cells
  4. Photoreceptor connecting cilium: NPHP1 expressed in photoreceptor CC →
     LOF → Senior-Løken Syndrome 1 (SLS1) retinal dystrophy in 10-15%

LOF → Juvenile tubulointerstitial nephritis (TIN) → corticomedullary cysts → small fibrotic kidneys
→ ESRD median 13 yr (range 4-20 yr). NPHP1 is cell-autonomous → renal transplant CURATIVE.

Hallmark Features:
  • JUVENILE onset (ESRD median 13 yr) — MOST COMMON NPHP
  • ~80-85% of genetically confirmed NPHP carries NPHP1 alleles
  • 290 kb HOMOZYGOUS DELETION at 2q13: 66-80% of patients (MLPA P369 kit — first line!)
  • Kidneys SMALL and echogenic (NOT enlarged — contrast NPHP2 infantile)
  • NO situs inversus (NPHP1 not expressed in nodal cilia)
  • NO congenital hepatic fibrosis (NPHP1 not expressed in biliary epithelium)
  • Senior-Løken Syndrome 1 (SLS1) = NPHP1 + rod-cone retinal dystrophy: 10-15%
  • Joubert Syndrome 4 (JBTS4) = NPHP1 + MTS: rare ~5%
  • Concentrating defect: earliest clinical sign (Uosm < 300 mOsm/kg)
  • Polyuria / polydipsia / enuresis: first symptom in most
  • Anaemia disproportionate to GFR (EPO-producing interstitial cell loss)
  • Normal blood pressure (salt-wasting tendency, not HTN like ADPKD)
  • Cognition NORMAL in pure NPHP1 (no CNS involvement)

Diagnostic Strategy:
  • MLPA (P369 NPHP1 kit, MRC-Holland): detects 290kb del → FIRST LINE for juvenile NPHP
  • If MLPA negative (20-34% of NPHP1): proceed to WES + CNV analysis
  • Sanger sequencing NPHP1 for compound heterozygous SNVs if one deletion found
  • Never use only NPHP1 MLPA for non-deletion alleles (SNV panel mandatory for het del)
  • Ophthalmology + ERG: ALL NPHP1 patients — 10-15% have subclinical retinal involvement

Key Differentials:
  NPHP2 (INVS/9q31): INFANTILE (ESRD 3yr); enlarged kidneys; situs inversus 30-50%;
    CHF 55%; no retinal
  NPHP5/IQCB1: Senior-Løken most common gene; LCA-like severe retinal from birth
  NPHP6/CEP290: Joubert spectrum; LCA10 allele IVS26; BBS14; mechanistically distinct
  ADPKD (PKD1/2): Autosomal DOMINANT; adult onset; macrocysts; HTN; haematuria
  ARPKD (PKHD1): Enlarged kidneys; ductal plate CHF; ductal ectasia on USS; no retinal
  Alport (COL4A5/A3/A4): Haematuria + SNHL + lenticonus; COL4 sequencing

Treatment:
  • Renal transplant = DEFINITIVE; CURATIVE; NO recurrence (cell-autonomous)
  • Living-related donor: obligate heterozygotes are SAFE (one functional allele sufficient)
  • Conservative CKD: nephrotoxin avoidance (NSAIDs, nephrotoxic contrast, aminoglycosides)
  • EPO: erythropoietin for disproportionate anaemia (start at Hb <10 g/dL)
  • Annual ERG + fundoscopy: ALL NPHP1 — detect subclinical SLS1
  • No retinal improvement post-transplant (retinal cell-autonomous, photoreceptor CC defect)
  • No approved disease-modifying therapy 2026; cystogenesis arrest pre-clinical (HDAC6 inhibitors)
"""

from __future__ import annotations
import random
from typing import Any

SEED        = 341
COHORT_N    = 40
rng         = random.Random(SEED)

# ── Patient phenotype distributions (NPHP1 literature: Hildebrand 2009, Wolf 2004, etc.) ──
ETHNICITIES = [
    ("European (non-consanguineous, heterogeneous)", 0.42),
    ("Middle Eastern (consanguineous, 290kb del enriched)", 0.18),
    ("South Asian (India/Pakistan, consanguineous)", 0.14),
    ("North African (Maghreb, consanguineous)", 0.10),
    ("Latin American", 0.07),
    ("East Asian", 0.05),
    ("Sub-Saharan African", 0.03),
    ("Other / Mixed", 0.01),
]

CKD_STAGES = [
    ("CKD 1 (GFR ≥90; concentrating defect only)", 0.10),
    ("CKD 2 (GFR 60–89; polyuria, mild anaemia)", 0.15),
    ("CKD 3a (GFR 45–59; growth retardation, fatigue)", 0.18),
    ("CKD 3b (GFR 30–44; progressive TIN, cysts visible)", 0.20),
    ("CKD 4 (GFR 15–29; pre-ESRD preparation)", 0.20),
    ("CKD 5/ESRD (GFR <15; awaiting or post-transplant)", 0.17),
]

KIDNEY_USS = [
    ("Bilateral small echogenic kidneys, corticomedullary cysts (classic)", 0.48),
    ("Bilateral small hyperechogenic kidneys, no discrete cysts (early TIN)", 0.22),
    ("Small kidneys with prominent corticomedullary cysts ≥5mm", 0.16),
    ("Unilateral findings asymmetric (one kidney smaller)", 0.06),
    ("Transplanted (previous ESRD, native kidneys atrophic)", 0.08),
]

FIRST_SYMPTOMS = [
    ("Polyuria / polydipsia / enuresis (concentrating defect)", 0.45),
    ("Growth retardation / failure to thrive on school health screen", 0.20),
    ("Anaemia found on routine labs (disproportionate for age)", 0.14),
    ("Incidental proteinuria / haematuria on school urine screen", 0.09),
    ("Oculomotor apraxia / nystagmus / poor vision (SLS1 presentation)", 0.07),
    ("Cerebellar ataxia + developmental delay (JBTS4 presentation)", 0.05),
]

SLS1_STATUS = [
    ("No retinal involvement (ERG normal, fundus clear)", 0.74),
    ("Senior-Løken Syndrome 1: rod-cone dystrophy (ERG abnormal)", 0.12),
    ("Subclinical retinal changes (ERG borderline, asymptomatic)", 0.08),
    ("SLS1 with nystagmus (early onset, severe retinal)", 0.04),
    ("JBTS4 (Joubert Syndrome 4: MTS + NPHP1, cerebellar features)", 0.02),
]

JBTS4_STATUS = [
    ("No Joubert features (pure renal NPHP1)", 0.93),
    ("JBTS4 confirmed (MTS on MRI + NPHP1 biallelic)", 0.04),
    ("Equivocal MRI (minor cerebellar changes, awaiting JBTS4 confirmation)", 0.03),
]

GENETIC_ARCHITECTURE = [
    ("290kb del 2q13 HOMOZYGOUS (MLPA P369 positive, both alleles)", 0.48),
    ("290kb del 2q13 / c.1849C>T p.Arg617* compound het (del + nonsense)", 0.22),
    ("290kb del 2q13 / frameshift compound het (del + frameshift)", 0.08),
    ("c.1849C>T p.Arg617* / c.1000C>T p.Arg334* compound het (biallelic nonsense)", 0.07),
    ("c.1481+1G>A (splice) / p.Arg617* compound het", 0.05),
    ("290kb del 2q13 / missense compound het (del + missense, moderate)", 0.05),
    ("Biallelic missense (homozygous or compound het, hypomorphic, late-onset)", 0.03),
    ("Novel / VUS compound het (heterogeneous, WES only)", 0.02),
]

PRIOR_MISDIAGNOSIS = [
    ("No prior misdiagnosis (NPHP1 MLPA positive first)", 0.35),
    ("'Chronic kidney disease — unclassified' (MLPA not sent)", 0.20),
    ("ADPKD suspected (cysts seen on USS, family history misread)", 0.14),
    ("Alport syndrome (haematuria + renal failure, COL4 sequencing first)", 0.10),
    ("ARPKD (enlarged kidneys in infancy misread; actually NPHP2 or pre-NPHP1)", 0.08),
    ("Focal segmental glomerulosclerosis (FSGS) on biopsy misread (TIN mis-classified)", 0.08),
    ("Idiopathic/genetic tubulointerstitial nephritis NOS", 0.05),
]

GFR_SLOPE = [
    ("Rapid (>5 ml/min/yr; ESRD before 16yr)", 0.18),
    ("Moderate (3–5 ml/min/yr; ESRD 16–20yr)", 0.40),
    ("Slow (1–3 ml/min/yr; ESRD 20–25yr)", 0.30),
    ("Very slow (<1 ml/min/yr; ESRD >25yr; biallelic missense hypomorphic)", 0.12),
]

URINE_OSM = [
    ("Severe deficit: Uosm <150 mOsm/kg (maximal concentrating failure)", 0.14),
    ("Moderate deficit: Uosm 150–250 mOsm/kg", 0.28),
    ("Mild deficit: Uosm 250–500 mOsm/kg (early CKD)", 0.38),
    ("Near-normal: Uosm >500 mOsm/kg (mild CKD, early NPHP1)", 0.20),
]

VARIANTS_POOL = [
    "290kb del 2q13 homozygous — MLPA P369 detects both; most common worldwide",
    "290kb del 2q13 / c.1849C>T p.Arg617* — del + European nonsense founder",
    "c.1849C>T p.Arg617* / c.1000C>T p.Arg334* — biallelic European nonsense compound het",
    "c.1481+1G>A (intron 13 splice donor) / c.1849C>T — splice + nonsense compound het",
    "290kb del / c.1668+1G>T (splice) — del + splice compound het",
    "c.1547_1548delGA (frameshift) / c.1849C>T — frameshift + nonsense compound het",
    "p.Pro679Leu (c.2036C>T) homozygous — missense; hypomorphic; late-onset pure renal",
    "Novel frameshift / 290kb del — WES identifies non-deletion allele + MLPA del",
]


def _weighted_choice(options, n_rng):
    labels, weights = zip(*options)
    r = n_rng.random()
    cum = 0.0
    for label, w in zip(labels, weights):
        cum += w
        if r < cum:
            return label
    return labels[-1]


def _make_cohort():
    patients = []
    for i in range(COHORT_N):
        r = random.Random(SEED + i * 31)
        ethnicity    = _weighted_choice(ETHNICITIES, r)
        ckd_stage    = _weighted_choice(CKD_STAGES, r)
        kidney_uss   = _weighted_choice(KIDNEY_USS, r)
        first_sym    = _weighted_choice(FIRST_SYMPTOMS, r)
        sls1_stat    = _weighted_choice(SLS1_STATUS, r)
        jbts4_stat   = _weighted_choice(JBTS4_STATUS, r)
        genetic_arch = _weighted_choice(GENETIC_ARCHITECTURE, r)
        misdiag      = _weighted_choice(PRIOR_MISDIAGNOSIS, r)
        gfr_slope    = _weighted_choice(GFR_SLOPE, r)
        urine_osm    = _weighted_choice(URINE_OSM, r)
        variant      = VARIANTS_POOL[r.randint(0, len(VARIANTS_POOL) - 1)]
        gfr_ml       = r.randint(5, 100)
        age_dx       = r.randint(4, 20)
        hb_val       = round(r.uniform(7.0, 13.5), 1)
        patients.append({
            "id":                 f"NPHP1-{i+1:03d}",
            "ethnicity":          ethnicity,
            "ckd_stage":          ckd_stage,
            "kidney_uss":         kidney_uss,
            "first_symptom":      first_sym,
            "sls1_status":        sls1_stat,
            "jbts4_status":       jbts4_stat,
            "genetic_architecture": genetic_arch,
            "prior_misdiagnosis": misdiag,
            "gfr_slope":          gfr_slope,
            "urine_osm":          urine_osm,
            "variant_label":      variant,
            "gfr_now_ml_min":     gfr_ml,
            "age_renal_dx_yr":    age_dx,
            "hb_g_dl":            hb_val,
        })
    return patients


_COHORT = _make_cohort()


def _pct(patients, key, value):
    return round(100 * sum(1 for p in patients if value in p.get(key, "")) / len(patients))


def _dist(patients, key):
    counts: dict[str, int] = {}
    for p in patients:
        v = p.get(key, "Unknown")
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _dist_prefix(patients, key, prefix_len=60):
    raw = _dist(patients, key)
    return {k[:prefix_len]: v for k, v in raw.items()}


# ── API: overview ──────────────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    pts = _COHORT
    esrd_pts       = [p for p in pts if "ESRD" in p["ckd_stage"]]
    sls1_pts       = [p for p in pts if "Senior-Løken" in p["sls1_status"]]
    jbts4_pts      = [p for p in pts if "JBTS4 confirmed" in p["jbts4_status"]]
    mlpa_del_pts   = [p for p in pts if "HOMOZYGOUS" in p["genetic_architecture"] or "compound het" in p["genetic_architecture"]]
    hom_del_pts    = [p for p in pts if "HOMOZYGOUS" in p["genetic_architecture"]]
    avg_gfr        = round(sum(p["gfr_now_ml_min"] for p in pts) / len(pts), 1)
    avg_age_dx     = round(sum(p["age_renal_dx_yr"] for p in pts) / len(pts), 1)
    avg_hb         = round(sum(p["hb_g_dl"] for p in pts) / len(pts), 1)
    conc_defect    = _pct(pts, "urine_osm", "deficit")

    return {
        "n_patients":             COHORT_N,
        "avg_gfr_ml_min":         avg_gfr,
        "avg_age_renal_dx_yr":    avg_age_dx,
        "avg_hb_g_dl":            avg_hb,
        "pct_esrd":               round(100 * len(esrd_pts) / len(pts)),
        "pct_sls1_retinal":       round(100 * len(sls1_pts) / len(pts)),
        "pct_jbts4":              round(100 * len(jbts4_pts) / len(pts)),
        "pct_290kb_hom_del":      round(100 * len(hom_del_pts) / len(pts)),
        "pct_mlpa_positive":      round(100 * len(mlpa_del_pts) / len(pts)),
        "pct_concentrating_defect": conc_defect,
        "disease":                "Nephronophthisis Type 1 (NPHP1 — Juvenile Nephronophthisis)",
        "gene":                   "NPHP1 (2q13) — Nephrocystin-1, 736 aa, TZ Y-link scaffold",
        "key_facts": [
            "MOST COMMON inherited ESRD in children (~80-85% of solved NPHP)",
            "290 kb deletion 2q13 — 66-80%; MLPA P369 kit is first-line diagnosis",
            "Juvenile ESRD: median 13 yr (range 4-20 yr); small echogenic kidneys",
            "NO situs inversus · NO CHF · NO intellectual disability (pure renal)",
            "Senior-Løken Syndrome 1 (SLS1): NPHP1 + retinal dystrophy — 10-15%",
            "Joubert Syndrome 4 (JBTS4): NPHP1 + MTS — rare ~5%",
            "MLPA first line; WES + CNV for MLPA-negative; annual ERG mandatory",
            "Renal transplant CURATIVE — no recurrence; retina does NOT improve",
        ],
        "esrd_median_yr":         13,
        "esrd_range_yr":          "4–20 yr",
        "typical_first_symptom":  "Polyuria / polydipsia / enuresis (concentrating defect)",
        "diagnostic_first_line":  "MLPA P369 (NPHP1) — detects 290 kb del",
        "omim_gene":              "*607100",
        "omim_disease":           "#256100 (NPHP1) + #266900 (SLS1)",
    }


# ── API: breakdown ─────────────────────────────────────────────────────────────
def get_breakdown() -> dict[str, Any]:
    pts = _COHORT
    return {
        "cohort_size":          COHORT_N,
        "ckd_stage":            _dist_prefix(pts, "ckd_stage", 55),
        "genetic_architecture": _dist_prefix(pts, "genetic_architecture", 65),
        "ethnicity":            _dist_prefix(pts, "ethnicity", 55),
        "first_symptom":        _dist_prefix(pts, "first_symptom", 60),
        "sls1_status":          _dist_prefix(pts, "sls1_status", 60),
        "jbts4_status":         _dist_prefix(pts, "jbts4_status", 60),
        "prior_misdiagnosis":   _dist_prefix(pts, "prior_misdiagnosis", 60),
        "gfr_slope":            _dist_prefix(pts, "gfr_slope", 60),
        "urine_osm":            _dist_prefix(pts, "urine_osm", 60),
        "kidney_uss":           _dist_prefix(pts, "kidney_uss", 60),
        "per_patient":          [
            {
                "id":               p["id"],
                "age_dx_yr":        p["age_renal_dx_yr"],
                "gfr":              p["gfr_now_ml_min"],
                "hb":               p["hb_g_dl"],
                "ckd_stage":        p["ckd_stage"][:40],
                "genetic":          p["genetic_architecture"][:55],
                "sls1":             p["sls1_status"][:40],
                "misdiag":          p["prior_misdiagnosis"][:40],
            }
            for p in pts
        ],
    }


# ── API: definitions ──────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    return {
        "disease":      "Nephronophthisis Type 1 (NPHP1 — Juvenile Nephronophthisis)",
        "omim_gene":    "*607100 (NPHP1 — Nephrocystin-1)",
        "omim_disease": "#256100 (Nephronophthisis 1) + #266900 (Senior-Løken Syndrome 1)",
        "chromosome":   "2q13",
        "inheritance":  "Autosomal Recessive; biallelic LOF; 290kb del is dominant founder allele",
        "prevalence":   "~1/50,000–1/200,000 (most common cause of inherited renal ESRD in children)",
        "mechanism":    (
            "NPHP1 (Nephrocystin-1, 736 aa) is a core subunit of the NPHP1-NPHP4-NPHP8 supercomplex "
            "at the ciliary transition zone (TZ) Y-links, bridging axoneme doublets to the ciliary "
            "membrane. LOF → TZ barrier loss → aberrant Hh, Wnt, PDGF signalling → cystogenesis + TIN. "
            "NPHP1 also stabilises cell-matrix adhesions (FAK/Paxillin interaction) and participates "
            "in the ATM-dependent DNA damage response at the centrosome. In photoreceptors, NPHP1 "
            "localises to the connecting cilium; LOF → Senior-Løken Syndrome 1 (rod-cone dystrophy)."
        ),
        "genetic_architecture": {
            "gene":                "NPHP1 (2q13); 736 aa; TZ Y-link scaffold; 4 TPR domains + coiled-coil",
            "founder_deletion":    "290 kb homozygous deletion 2q13 — 66-80% of NPHP1 patients worldwide",
            "diagnostic_tool":     "MLPA P369 (MRC-Holland) — detects 290kb del; FIRST LINE for juvenile NPHP",
            "mlpa_negative":       "~20-34% NPHP1 are MLPA-negative → WES + CNV analysis for compound het SNVs",
            "common_snvs":         "c.1849C>T p.Arg617* (European nonsense founder); c.1000C>T p.Arg334*; c.1481+1G>A splice",
            "genotype_phenotype":  "Null (del/truncating): ESRD 13yr; biallelic missense (hypomorphic): ESRD 20-25yr",
            "digenic":             "No digenic interactions documented for NPHP1 (contrast NPHP20/CEP120)",
            "detection":           "MLPA (del) + WES+CNV (SNV/del compound het) — do NOT stop at MLPA negative",
        },
        "nphp_comparison": {
            "NPHP1 (NPHP1/2q13)":  "MOST COMMON; Juvenile ESRD 13yr; 290kb del 66%; ±SLS1 12%; NO situs inversus",
            "NPHP2 (INVS/9q31)":   "Infantile ESRD 3yr; heterogeneous; situs inversus 30-50%; CHF 55%",
            "NPHP3 (NPHP3/3q22)":  "Adolescent ESRD ~19yr; hepatic fibrosis ±; CHF rare; ±retinal",
            "NPHP4 (NPHP4/1p36)":  "Juvenile; NPHP1 supercomplex partner; Leber congenital amaurosis overlap",
            "NPHP5/IQCB1":         "SLS most common gene; severe retinal (LCA-like from birth); renal juvenile",
            "NPHP6/CEP290":        "Joubert spectrum; LCA10 (IVS26 allele); BBS14; mechanistically distinct",
        },
        "key_clinical_features": {
            "onset":               "Juvenile: ESRD by median 13 yr (range 4-20 yr); most common inherited ESRD in children",
            "kidneys":             "Small, hyperechogenic on USS; corticomedullary cysts; NO renal enlargement",
            "concentrating_defect": "Uosm < 300 mOsm/kg; polyuria/polydipsia/enuresis — EARLIEST clinical sign",
            "anaemia":             "Disproportionate for CKD stage (EPO-producing interstitial cells lost early)",
            "blood_pressure":      "Normal or low (salt-wasting tendency; NOT hypertensive unlike ADPKD)",
            "sls1_retinal":        "10-15% Senior-Løken Syndrome 1: rod-cone RP-like dystrophy; annual ERG mandatory",
            "jbts4":               "~5% Joubert Syndrome 4: MTS on brain MRI, cerebellar vermis hypoplasia",
            "no_situs_inversus":   "ABSENT — NPHP1 not expressed in nodal cilia (contrast NPHP2 30-50%)",
            "no_chf":              "ABSENT — NPHP1 not expressed in biliary epithelium (contrast NPHP2/NPHP3)",
            "no_intellectual_disability": "Cognition NORMAL in pure NPHP1; JBTS4 may have mild developmental delay",
            "no_polydactyly":      "ABSENT — no skeletal features (contrast NPHP15/CEP164, CEP290 Meckel)",
        },
        "diagnostic_criteria": {
            "genetic_gold_std":  "Biallelic pathogenic NPHP1 variants (del and/or SNV) confirmed by MLPA + WES",
            "first_line":        "MLPA P369 (NPHP1) — send for ALL juvenile CKD + small cystic kidneys",
            "second_line":       "WES + CNV analysis if MLPA negative or single deletion found (compound het)",
            "clinical_triggers": "Juvenile ESRD + concentrating defect + small echogenic kidneys ± family history",
            "imaging":           "Renal USS: small echogenic kidneys, corticomedullary cysts; normal liver (no CHF); "
                                  "brain MRI ONLY if cerebellar signs (JBTS4 suspected)",
            "labs":              "Uosm < 300 mOsm/kg; tubular proteinuria; elevated creatinine; disproportionate anaemia",
            "ophthalmology":     "ERG + fundoscopy: ALL NPHP1 patients annually; 10-15% SLS1 subclinical",
        },
        "ddx_table": {
            "NPHP2 (INVS)":    "Infantile ESRD 3yr; ENLARGED kidneys (NOT small); situs inversus 30-50%; "
                                "CHF 55%; ARPKD mimic; INVS MLPA not available — WES + CNV",
            "ADPKD (PKD1/2)":  "AUTOSOMAL DOMINANT; adult onset 30-50yr; macro-cysts; HYPERTENSION; haematuria; "
                                "family history AD; kidneys ENLARGED; NOT juvenile ESRD",
            "ARPKD (PKHD1)":   "Infantile enlarged kidneys with ductal ectasia; CHF + portal HTN; sausage-shaped "
                                "kidneys on USS; PKHD1 mutations; bilateral tubular ectasia on IVP",
            "Alport (COL4)":   "Persistent haematuria + SNHL + lenticonus; COL4A3/A4/A5; electron microscopy "
                                "GBM thinning/splitting; NOT concentrating defect first",
            "NPHP5/IQCB1":     "Senior-Løken most common: LCA-like from BIRTH; NPHP1 retinal is later/milder; "
                                "both have renal TIN — distinguish by genotype",
            "TIN (idiopathic)": "Biopsy: TIN with no genetic cause — NPHP1 MLPA mandatory before calling idiopathic",
        },
        "treatment": {
            "renal_transplant":   "DEFINITIVE — cell-autonomous; NO recurrence in graft; CURATIVE for renal NPHP1",
            "living_related":     "Parental donors (obligate heterozygotes) are SAFE — single NPHP1 allele sufficient",
            "conservative_ckd":  "Avoid NSAIDs, nephrotoxic contrast (iodinated + gadolinium), aminoglycosides",
            "epo":               "Erythropoietin for disproportionate anaemia; start early before Hb <10 g/dL",
            "fluid_intake":      "High fluid intake (2-3 L/m²/d) to compensate concentrating defect; prevent dehydration",
            "retinal":           "Annual ERG + fundoscopy; low-vision aids if SLS1; retinal does NOT improve post-transplant",
            "jbts4_mdm":         "Multidisciplinary (neurology + nephrology + ophthalmology) if JBTS4; OT/PT for ataxia",
            "no_dmt_2026":       "No approved disease-modifying therapy; HDAC6 inhibitors pre-clinical (cystogenesis arrest)",
        },
        "key_variants": [
            "290kb del 2q13 HOMOZYGOUS — 48% of cohort; MLPA P369 detects; worldwide distribution",
            "290kb del 2q13 / c.1849C>T p.Arg617* — 22%; compound het; European founder nonsense + del",
            "c.1849C>T p.Arg617* / c.1000C>T p.Arg334* — 7%; biallelic nonsense European compound het",
            "c.1481+1G>A (intron13 splice) / c.1849C>T — 5%; splice + nonsense compound het",
            "p.Pro679Leu (c.2036C>T) homozygous — 3%; missense; hypomorphic; late-onset ESRD 20-25yr",
        ],
        "prognosis": (
            "ESRD by median 13 yr (range 4-20 yr). Renal transplant is curative with excellent "
            "long-term outcomes — NPHP1 is cell-autonomous (no recurrence in graft). "
            "SLS1 retinal dystrophy does NOT improve after transplant (retinal cell-autonomous). "
            "Cognition and quality of life are NORMAL for pure renal NPHP1. "
            "JBTS4 has additional cerebellar and developmental considerations. "
            "Early diagnosis via MLPA enables timely CKD monitoring and transplant planning."
        ),
        "cohort_note": (
            "Synthetic cohort · 40 patients · NPHP1/2q13 · seed-341 · generated for clinical "
            "decision-support training · not derived from real patient data."
        ),
    }
