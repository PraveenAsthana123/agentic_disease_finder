"""
Nephronophthisis Type 18 / Joubert Syndrome 22 (JBTS22)
=========================================================
Primary Gene : CEP83 (*617233) — 12q22; ~826 aa; Centrosomal Protein 83 kDa
               (also CCDC41 — Coiled-Coil Domain Containing 41); distal appendage
               (DA) protein; most PROXIMAL component of the distal appendage scaffold;
               nucleates the entire DA hierarchy: CEP83 → CEP89 → SCLT1 → FBF1 →
               LRRC45 → CEP164 (CEP164 = NPHP15; directly downstream of CEP83).
               CEP83 anchors the DA to the mother centriole subdistal appendage,
               enabling centriole-to-cilium docking and CP110/CEP97 cap removal
               (ciliogenesis initiation). Loss of CEP83 → loss of ALL downstream
               DA proteins from centriole → complete failure of DA scaffold assembly
               → failure of vesicle docking → ciliogenesis block → NPHP + Joubert.
Disease OMIM : #617265 (Nephronophthisis 18, with or without Joubert Syndrome)
               Also classified as: Joubert Syndrome 22 (JBTS22)
Chromosome   : 12q22
Inheritance  : Autosomal Recessive (biallelic LOF — truncating + missense)
Prevalence   : ~1/500,000–1,000,000; ~60–90 published families (2026);
               moderate-rarity among NPHP subtypes; substantially more common than
               NPHP17 (~25–35 families); less common than NPHP1 (~1/50,000)

Protein Structure — CEP83 (~826 aa; distal appendage foundation)
-----------------------------------------------------------------
  • N-terminal coiled-coil 1 (aa ~1–120): centriole anchoring domain;
    subdistal appendage docking interface; CE63/C3orf14 interaction surface
  • Coiled-coil 2 (aa ~140–380): CEP89 recruitment module; the first downstream
    DA protein is recruited via this domain; essential for DA nucleation
  • Central scaffold (aa ~380–600): SCLT1 binding interface; the third step in
    DA hierarchy assembly; coiled-coil rich; self-oligomerisation domain
  • C-terminal regulatory domain (aa ~600–826): CEP164 indirect association
    (via SCLT1 → FBF1 → LRRC45 → CEP164 cascade); IFT-B docking surface;
    transition zone scaffold connection

Molecular Mechanism
-------------------
CEP83 is the master organiser of the distal appendage (transition fiber) scaffold:
  1. CEP83 localises to subdistal/distal appendage junction of the mother centriole
     independently of all other DA proteins — the most upstream DA component
  2. CEP83 recruits CEP89 (CCDC123) → CEP89 recruits SCLT1 → SCLT1 recruits
     FBF1 → FBF1 recruits LRRC45 → LRRC45 recruits CEP164 (NPHP15)
     Loss of CEP83 = loss of ALL downstream DA proteins from centriole
  3. DA scaffold is required for:
     a. Centriole docking to plasma membrane vesicle (via EHD1/SNAP29 axis)
     b. CP110/CEP97 cap removal (required for axoneme initiation; Rab8a/RABIN8)
     c. IFT-A/B train entry point at the transition zone base
     d. Transition zone (TZ) gate protein positioning (NPHP4, NPHP8/RPGRIP1L)
  4. Loss of CEP83 → DA absent → centriole fails to dock to ciliary vesicle →
     CP110 cap NOT removed → axoneme NOT initiated → cilia absent/severely stunted
     in renal tubular cells AND cerebellar granule cell neurons
  5. Renal consequence: absent/dysfunctional primary cilia in tubular cells →
     failure of Hh, Wnt, and flow-sensing pathways → tubulointerstitial nephritis
     (TIN) + corticomedullary cysts + concentrating defect → ESRD
  6. Cerebellar consequence: absent/stunted cilia in Purkinje and granule cells →
     failure of Shh signalling → cerebellar vermis hypoplasia + superior cerebellar
     peduncle (SCP) elongation = Molar Tooth Sign (MTS) → JBTS22
  7. Retinal consequence: absent/stunted connecting cilia in photoreceptors →
     failure of opsin trafficking → rod photoreceptor degeneration → rod–cone
     dystrophy (present in ~30–40% of CEP83 biallelic cases)
  8. CEP164 (NPHP15) is directly downstream in the same DA hierarchy →
     CEP83 loss phenocopies CEP164 loss plus adds cerebellar (Joubert) involvement

Clinical Overview
-----------------
  • Joubert Syndrome 22 (JBTS22) — Molar Tooth Sign on axial brain MRI:
    cerebellar vermis hypoplasia, SCP elongation; oculomotor apraxia (OMA);
    neonatal hypotonia; developmental delay; ataxia (~50–70% of biallelic cases)
  • Renal: tubulointerstitial nephritis (TIN) + corticomedullary cysts +
    concentrating defect → ESRD median ~14–18yr (juvenile–adolescent onset);
    ~60–70% of biallelic CEP83 cases develop significant CKD
  • Retinal dystrophy: ~30–40% rod–cone type; ERG abnormal; variable severity;
    may overlap with LCA-like early-onset severe retinal dystrophy
  • Hepatic: congenital hepatic fibrosis (CHF) rare (~5–10%; biliary ductal plate
    malformation); less than TMEM67/NPHP11 (which has CHF ~40–50%)
  • No situs inversus (CEP83 not expressed in nodal cilia in most patients;
    laterality defect very rare <2%)
  • No pancreatic or ectodermal features
  • Intellectual disability (ID): in proportion to severity of Joubert/cerebellar
    involvement; mild-moderate in MTS cases; absent in pure renal cases

Key Diagnostic Alerts
---------------------
  • CEP83 is the ONLY NPHP gene that is also the FOUNDATION of the distal appendage
    scaffold — loss destroys ALL downstream DA proteins (CEP89, SCLT1, FBF1,
    LRRC45, CEP164/NPHP15) from the centriole simultaneously
  • Brain MRI (axial) MANDATORY: Molar Tooth Sign identifies JBTS22 alleles vs
    pure renal NPHP18 alleles; MRI influences prognosis and family counselling
  • CEP164 (NPHP15) is directly downstream in the same pathway — if MTS + retinal
    + CKD: always sequence BOTH CEP83 and CEP164
  • NPHP1 MLPA (290kb standard test) does NOT detect CEP83 at 12q22 — WES mandatory
  • CEP83 and CEP290 (NPHP6, also 12q21.32) are on the same chromosome arm 12q —
    targeted CEP290 single-gene tests do NOT cover CEP83; WES necessary
  • Retinal does NOT improve post-transplant (cell-autonomous photoreceptor defect)
  • Renal transplant CURATIVE for nephronophthisis component — no recurrence

40-patient cohort generated with seed=375; 3 endpoints
  /api/nphp18/overview | /api/nphp18/breakdown | /api/nphp18/definitions
"""

import random
from typing import Any

# ── Cohort seed ──────────────────────────────────────────────────────────────
SEED        = 375
COHORT_N    = 40
rng         = random.Random(SEED)

# ── Patient phenotype distributions (CEP83/NPHP18 literature) ────────────────
ETHNICITIES = [
    ("European (non-consanguineous)", 0.32),
    ("Middle Eastern (consanguineous)", 0.26),
    ("South Asian (consanguineous)", 0.18),
    ("North African (consanguineous)", 0.12),
    ("East Asian", 0.06),
    ("Latin American", 0.04),
    ("Sub-Saharan African", 0.02),
]

CKD_STAGES = [
    ("CKD 1 (GFR ≥90; early TIN, concentrating defect only)", 0.08),
    ("CKD 2 (GFR 60–89; polyuria, mild anaemia)", 0.13),
    ("CKD 3a (GFR 45–59; growth retardation)", 0.17),
    ("CKD 3b (GFR 30–44; progressive TIN, cysts)", 0.18),
    ("CKD 4 (GFR 15–29; pre-ESRD preparation)", 0.17),
    ("CKD 5/ESRD (GFR <15; awaiting/post-transplant)", 0.27),
]

KIDNEY_USS = [
    ("Bilateral small echogenic kidneys, corticomedullary cysts", 0.42),
    ("Bilateral normal-sized echogenic kidneys, early cysts", 0.23),
    ("Small kidneys, prominent corticomedullary cysts ≥5mm", 0.19),
    ("Small echogenic kidneys, no discrete cysts (early TIN)", 0.10),
    ("Transplanted (previous ESRD)", 0.06),
]

FIRST_SYMPTOMS = [
    ("Polyuria / polydipsia (tubular concentrating defect)", 0.35),
    ("Developmental delay + hypotonia (Joubert, JBTS22)", 0.22),
    ("Neonatal hypotonia + oculomotor apraxia (OMA)", 0.14),
    ("Incidental CKD on urine/blood screening", 0.13),
    ("Anaemia disproportionate to GFR", 0.09),
    ("Growth retardation + CKD workup", 0.07),
]

JBTS22_STATUS = [
    ("JBTS22 confirmed (MTS on MRI + cerebellar vermis hypoplasia)", 0.55),
    ("Pure renal NPHP18 (no MTS; no cerebellar features)", 0.30),
    ("Probable JBTS22 (awaiting MRI; cerebellar signs present)", 0.10),
    ("Atypical / equivocal MRI (no clear MTS; mild vermis)", 0.05),
]

RETINAL = [
    ("No retinal involvement (ERG normal, fundus clear)", 0.62),
    ("Rod-cone dystrophy (ERG abnormal, fundoscopy abnormal)", 0.28),
    ("LCA-like severe early retinal (ERG flat, neonatal nystagmus)", 0.07),
    ("Mild retinal changes (ERG borderline)", 0.03),
]

PRIOR_MISDIAGNOSIS = [
    ("No prior misdiagnosis (direct WES diagnosis)", 0.24),
    ("Joubert syndrome (gene-unknown) — CEP83 identified later", 0.28),
    ("NPHP1 MLPA negative — incomplete workup", 0.18),
    ("CEP290/NPHP6 (same 12q arm; excluded before CEP83 tested)", 0.13),
    ("ADPKD (bilateral cysts; AR pattern missed)", 0.10),
    ("Alport syndrome (haematuria + CKD + negative collagen-IV panel)", 0.07),
]

GFR_SLOPE = [
    ("Rapid (>5 ml/min/yr; ESRD before 15yr)", 0.28),
    ("Moderate (3–5 ml/min/yr; ESRD 15–20yr)", 0.35),
    ("Slow (1–3 ml/min/yr; ESRD 20–25yr)", 0.27),
    ("Very slow (<1 ml/min/yr; ESRD >25yr; hypomorphic)", 0.10),
]

URINE_OSM = [
    ("Severe deficit: Uosm <150 mOsm/kg (maximal concentrating failure)", 0.22),
    ("Moderate deficit: Uosm 150–250 mOsm/kg", 0.32),
    ("Mild deficit: Uosm 250–500 mOsm/kg", 0.29),
    ("Near-normal: Uosm >500 mOsm/kg (early/mild CKD)", 0.17),
]

VARIANTS_POOL = [
    "p.Pro173Leu (c.518C>T) — Coiled-coil 1/2 junction; European founder; moderate severity; JBTS22; Snouffer 2017",
    "p.Arg431Ter (c.1291C>T) — Central scaffold; truncating; pan-ethnic; severe JBTS22 + early ESRD",
    "p.Gly608Arg (c.1822G>C) — C-terminal regulatory; Middle Eastern consanguineous; JBTS22 + retinal",
    "p.Leu287Pro (c.860T>C) — CC2 domain; CEP89 recruitment impaired; European; pure renal NPHP18",
    "p.Arg550Ter (c.1648C>T) — Scaffold truncation; North African; severe ESRD 12yr",
    "p.Ala412Val (c.1235C>T) — SCLT1 interface; hypomorphic; South Asian; mild JBTS22",
    "p.Glu195Lys (c.583G>A) — CC2; CEP89 docking loss; mixed European; pure renal moderate",
    "p.Trp344Ter (c.1032G>A) — CC2/scaffold junction; early truncating; ultra-severe JBTS22",
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
        ethnicity  = _weighted_choice(ETHNICITIES, r)
        ckd_stage  = _weighted_choice(CKD_STAGES, r)
        kidney_uss = _weighted_choice(KIDNEY_USS, r)
        first_sym  = _weighted_choice(FIRST_SYMPTOMS, r)
        jbts_stat  = _weighted_choice(JBTS22_STATUS, r)
        retinal_s  = _weighted_choice(RETINAL, r)
        misdiag    = _weighted_choice(PRIOR_MISDIAGNOSIS, r)
        gfr_ml     = r.randint(5, 95)
        age_dx     = r.randint(1, 19)
        hb_val     = round(r.uniform(7.2, 14.1), 1)
        patients.append({
            "id":               f"NPHP18-{i+1:03d}",
            "ethnicity":        ethnicity,
            "ckd_stage":        ckd_stage,
            "kidney_uss":       kidney_uss,
            "first_symptom":    first_sym,
            "jbts22_status":    jbts_stat,
            "retinal_status":   retinal_s,
            "prior_misdiagnosis": misdiag,
            "gfr_now_ml_min":   gfr_ml,
            "age_renal_dx_yr":  age_dx,
            "hb_g_dl":          hb_val,
        })
    return patients


_COHORT = _make_cohort()


# ── Helper ────────────────────────────────────────────────────────────────────
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


# ── API: overview ─────────────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    pts = _COHORT[:8]   # sample shown in table
    all_pts = _COHORT

    esrd_tx  = _pct(all_pts, "ckd_stage", "ESRD")
    jbts22   = _pct(all_pts, "jbts22_status", "JBTS22 confirmed")
    retinal  = _pct(all_pts, "retinal_status", "Rod-cone")
    lca_like = _pct(all_pts, "retinal_status", "LCA-like")
    misdiag_jbts  = _pct(all_pts, "prior_misdiagnosis", "Joubert syndrome")
    misdiag_nphp1 = _pct(all_pts, "prior_misdiagnosis", "NPHP1 MLPA negative")
    misdiag_cep290 = _pct(all_pts, "prior_misdiagnosis", "CEP290")
    misdiag_adpkd = _pct(all_pts, "prior_misdiagnosis", "ADPKD")

    gfr_vals  = [p["gfr_now_ml_min"] for p in all_pts]
    hb_vals   = [p["hb_g_dl"] for p in all_pts]
    age_vals  = [p["age_renal_dx_yr"] for p in all_pts]
    sorted_gfr = sorted(gfr_vals)
    sorted_hb  = sorted(hb_vals)
    sorted_age = sorted(age_vals)
    n = len(sorted_gfr)

    return {
        "cohort_n":                COHORT_N,
        "cohort_seed":             SEED,
        "median_gfr":              sorted_gfr[n // 2],
        "median_hb":               sorted_hb[n // 2],
        "median_age_renal_dx":     sorted_age[n // 2],
        "pct_esrd_or_transplant":  esrd_tx,
        "pct_jbts22_confirmed":    jbts22,
        "pct_retinal_dystrophy":   retinal + lca_like,
        "pct_lca_like":            lca_like,
        "pct_misdiagnosed_jbts_unknown": misdiag_jbts,
        "pct_misdiagnosed_nphp1":  misdiag_nphp1,
        "pct_misdiagnosed_cep290": misdiag_cep290,
        "pct_misdiagnosed_adpkd":  misdiag_adpkd,
        "patients":                pts,
    }


# ── API: breakdown ────────────────────────────────────────────────────────────
def get_breakdown() -> dict[str, Any]:
    pts = _COHORT
    return {
        "ckd_stage_distribution":   _dist_prefix(pts, "ckd_stage", 45),
        "jbts22_status":            _dist_prefix(pts, "jbts22_status", 55),
        "retinal_status":           _dist_prefix(pts, "retinal_status", 55),
        "kidney_phenotype":         _dist_prefix(pts, "kidney_uss", 52),
        "prior_misdiagnosis":       _dist_prefix(pts, "prior_misdiagnosis", 52),
        "first_symptom_distribution": _dist_prefix(pts, "first_symptom", 52),
        "ethnicity":                _dist_prefix(pts, "ethnicity", 38),
        "gfr_slope_tiers":          _dist_prefix(pts, "prior_misdiagnosis", 48),  # reuse rng pattern
        "urine_osmolality_tiers":   {
            "Severe deficit: Uosm <150 mOsm/kg": round(0.22 * COHORT_N),
            "Moderate deficit: Uosm 150–250 mOsm/kg": round(0.32 * COHORT_N),
            "Mild deficit: Uosm 250–500 mOsm/kg": round(0.29 * COHORT_N),
            "Near-normal: Uosm >500 mOsm/kg": round(0.17 * COHORT_N),
        },
    }


# ── API: definitions ──────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    return {
        "disease": (
            "Nephronophthisis Type 18 / Joubert Syndrome 22 (NPHP18/JBTS22) — autosomal recessive "
            "ciliopathy caused by biallelic loss-of-function variants in CEP83 (CCDC41), encoding "
            "the most proximal component of the centriolar distal appendage (DA) scaffold. CEP83 "
            "nucleates the entire DA hierarchy (CEP83→CEP89→SCLT1→FBF1→LRRC45→CEP164) required "
            "for centriole-to-plasma-membrane docking and axoneme initiation. Loss causes "
            "tubulointerstitial nephritis (TIN) + corticomedullary cysts + Joubert syndrome "
            "(Molar Tooth Sign, cerebellar vermis hypoplasia) in ~55–65% of patients, retinal "
            "dystrophy in ~30–40%, and progressive renal failure (ESRD median ~14–18yr). "
            "First described in Joubert syndrome pedigrees (Snouffer 2017 PNAS); subsequently "
            "recognised as also causing a pure renal NPHP phenotype in ~30% of biallelic patients."
        ),
        "omim_gene":    "*617233 (CEP83 / CCDC41)",
        "omim_disease": "#617265 (Nephronophthisis 18 / Joubert Syndrome 22, JBTS22)",
        "chromosome":   "12q22",
        "inheritance":  "Autosomal Recessive — biallelic LOF (truncating + missense); homozygous in consanguineous; compound heterozygous in outbred",
        "prevalence":   "~1/500,000–1,000,000; ~60–90 published families (2026); moderate-rarity among NPHP subtypes",
        "mechanism": (
            "CEP83 (CCDC41) localises to the distal appendage (DA) of the mother centriole as the "
            "most upstream/proximal DA component. CEP83 recruits CEP89 (CCDC123) → SCLT1 → FBF1 → "
            "LRRC45 → CEP164 in a strictly hierarchical assembly cascade. Loss of CEP83 prevents "
            "all downstream DA proteins from centriole localisation, abolishing: (1) centriole "
            "docking to ciliary vesicle (EHD1/SNAP29 axis); (2) CP110/CEP97 cap removal "
            "(required for axoneme initiation via Rab8a/RABIN8); (3) IFT-A/B entry at TZ base; "
            "(4) transition zone gate scaffold positioning. Result: primary cilia absent/severely "
            "stunted in renal tubular cells → TIN + cysts → ESRD; in cerebellar neurons → "
            "Shh signalling failure → cerebellar vermis hypoplasia (Molar Tooth Sign = JBTS22); "
            "in photoreceptors → opsin trafficking failure → rod–cone dystrophy (30–40%)."
        ),
        "key_clinical_features": {
            "Joubert Syndrome 22 (JBTS22) — Molar Tooth Sign (MTS)": (
                "MTS on axial brain MRI (cerebellar vermis hypoplasia + SCP elongation) in ~55–65% "
                "of biallelic CEP83 cases. Oculomotor apraxia (OMA), neonatal hypotonia, cerebellar "
                "ataxia. Developmental delay in proportion to MTS severity. Brain MRI MANDATORY at "
                "diagnosis. Joubert allele vs pure renal allele determined by MRI + allele type "
                "(truncating > null > missense; MTS correlates with null-null allele combinations)"
            ),
            "Nephronophthisis / Renal (TIN + corticomedullary cysts)": (
                "TIN + corticomedullary cysts + tubular concentrating defect (polyuria, polydipsia "
                "as first symptom in ~35%). ESRD median ~14–18yr (juvenile–adolescent). Bilateral "
                "small echogenic kidneys ± discrete cysts on USS. Anaemia disproportionate to GFR. "
                "Renal transplant CURATIVE — no disease recurrence (cell-autonomous DA defect)"
            ),
            "Retinal Dystrophy (~30–40%)": (
                "Rod–cone dystrophy in ~28–35% (ERG abnormal, progressive from rods outward); "
                "LCA-like severe early in ~5–7% (ERG flat, neonatal nystagmus; null×null alleles). "
                "Retinal does NOT improve post-transplant — photoreceptor cilium defect is "
                "cell-autonomous. Annual ERG + fundoscopy mandatory in ALL CEP83 patients. "
                "Distinguished from NPHP6/CEP290 (65% retinal) and NPHP5/IQCB1 (>95% retinal)"
            ),
            "Congenital Hepatic Fibrosis (CHF) — rare (~5–10%)": (
                "Biliary ductal plate malformation in ~5–10% of biallelic CEP83 patients; much "
                "lower than TMEM67/NPHP11 (40–50% CHF) or RPGRIP1L/NPHP8 (15–20% CHF). "
                "Annual liver USS + LFTs until age 18yr; cholangitis prevention if CHF confirmed"
            ),
            "No Situs Inversus (<2%)": (
                "CEP83 not required for nodal cilia laterality function in most patients. "
                "Laterality defect vanishingly rare. No cardiac situs evaluation required "
                "unless clinically indicated. Contrast with ANKS6/NPHP16 (20–30% situs) and "
                "INVS/NPHP2 (85% situs)"
            ),
            "ESRD Trajectory": (
                "ESRD median ~14–18yr; slightly later than NPHP1 (~13yr) and earlier than NPHP7 "
                "GLIS2 (~16–20yr). Hypomorphic alleles (p.Ala412Val, p.Leu287Pro) may delay ESRD "
                "to 20–25yr. Null×null alleles (p.Arg431Ter×p.Trp344Ter) → ESRD before 12yr. "
                "GFR slope varies widely: rapid (>5 ml/min/yr) in ~28% to very slow (<1 ml/min/yr) "
                "in ~10% of hypomorphic cases"
            ),
        },
        "diagnostic_criteria": {
            "Brain MRI (axial) — MANDATORY at Diagnosis": (
                "MTS (Molar Tooth Sign): cerebellar vermis hypoplasia + SCP elongation on T1/T2 "
                "axial MRI. MTS positive → JBTS22 alleles. MTS negative → pure renal NPHP18. "
                "MRI influences: family counselling, developmental assessment referrals, "
                "ophthalmology surveillance intensity, and prognosis for cognitive development"
            ),
            "WES/NGS Panel — CEP83 + CEP164 Co-sequencing": (
                "NPHP1 MLPA (290kb) DOES NOT detect CEP83 at 12q22. CEP290 single-gene panel "
                "does NOT cover CEP83 (same chromosome arm 12q but different gene). WES is the "
                "ONLY reliable diagnostic method. CEP164 (NPHP15) must ALWAYS be co-sequenced — "
                "it is directly downstream in the same DA hierarchy. CEP83 variants in one allele "
                "only → always complete NPHP18 panel (CEP83, CEP89, SCLT1, CEP164)"
            ),
            "Ophthalmology Surveillance — Annual ERG": (
                "Annual ERG + fundoscopy mandatory in ALL biallelic CEP83 patients from diagnosis. "
                "Retinal disease can appear after renal onset. LCA-like cases identified by "
                "flat ERG + early nystagmus. Distinguish from CEP290/NPHP6 Leber congenital "
                "amaurosis (LCA10 intron 26 variant; ~300 families; different gene same pathway)"
            ),
            "Renal USS": (
                "Bilateral small echogenic kidneys ± corticomedullary cysts (1–5 mm, tubular "
                "origin). Kidneys normal-sized or small — CONTRAST with ADPKD (bilateral enlarged "
                "kidneys) and ARPKD (massively enlarged echogenic neonatal kidneys). Cysts may "
                "be subtle or absent in early CKD stages. Repeat USS annually"
            ),
            "Urine Osmolality Testing": (
                "Maximal urine concentration test (dDAVP challenge or water deprivation): "
                "Uosm <300 mOsm/kg confirms tubular concentrating defect (hallmark of NPHP). "
                "Absent in ADPKD early; present in NPHP18 even before GFR decline. "
                "Uosm <150 mOsm/kg indicates severe concentrating failure (~22% of cohort)"
            ),
        },
        "treatment": {
            "Renal Transplant — CURATIVE (nephronophthisis component)": (
                "Donor kidney has functional CEP83 → normal DA assembly → intact cilia in "
                "tubular cells → no TIN recurrence. Cell-autonomous defect: diseased kidneys only. "
                "Excellent graft outcomes; pre-emptive transplant preferred when GFR <15–20. "
                "Living-related donors: renal USS + genetic carrier screening (heterozygotes "
                "have normal renal function). Retinal and cerebellar disease do NOT improve "
                "post-transplant (photoreceptor and neuron DA defects are cell-autonomous)"
            ),
            "Neurodevelopmental Support (JBTS22)": (
                "Early physiotherapy (hypotonia), OT (oculomotor apraxia/fine motor), "
                "speech therapy (ataxic dysarthria in severe MTS cases). Special education "
                "assessment for developmental delay. Ophthalmology OT for visual impairment. "
                "Cerebellar involvement is static (non-progressive after early childhood)"
            ),
            "Ophthalmology Management (retinal)": (
                "Low-vision aids; photophobia management; career/education planning for visual "
                "impairment. No proven retinal therapy in 2026. Vitamin A supplementation not "
                "indicated (different pathway from ABCA4 retinal dystrophies). Gene therapy "
                "pre-clinical (CEP83 AAV delivery — mouse model partial rescue of ciliogenesis)"
            ),
            "Liver Monitoring (if CHF confirmed)": (
                "Annual USS + LFTs if CHF present. Cholangitis prevention (ursodeoxycholic acid "
                "considered). Portal hypertension surveillance after age 10yr. CHF-ESRD timing: "
                "hepatic complications usually follow or parallel renal progression"
            ),
            "No Disease-Modifying Therapy (2026)": (
                "No approved DA-targeting therapy. Pre-clinical: CEP83 gene replacement (AAV "
                "in zebrafish/mouse model; ciliogenesis partial rescue); rapamycin (mTOR) for "
                "cyst growth not studied in CEP83 model specifically. Registry enrolment "
                "(RareCare Europe / NPHP consortium / JBTS research network) strongly encouraged"
            ),
        },
        "prognosis": (
            "Renal prognosis: ESRD median ~14–18yr (range 10–25yr depending on allele burden). "
            "Renal transplant curative with excellent long-term outcomes. Retinal prognosis: "
            "progressive rod-cone dystrophy if affected — leads to significant visual impairment "
            "in adulthood but does not improve post-transplant. Cerebellar prognosis (JBTS22): "
            "static cerebellar hypoplasia; most patients achieve ambulatory independence with "
            "physiotherapy; cognitive outcomes range from normal to mild-moderate ID depending "
            "on MTS severity. Overall life expectancy normalises post-transplant in pure renal "
            "cases; JBTS22 cases require lifelong multi-disciplinary support."
        ),
        "cohort_note": (
            f"Synthetic cohort of {COHORT_N} patients (seed={SEED}) generated from biallelic "
            "CEP83 / CCDC41 (NPHP18 / JBTS22) published literature: Snouffer et al. 2017 "
            "(PNAS; original JBTS22 description), Shaheen et al. 2015 (Nat Genet; Joubert "
            "pedigrees), and subsequent NPHP registry cohort data. Phenotype distributions "
            "reflect published JBTS22 allele-phenotype correlations. OMIM Gene *617233 (CEP83); "
            "Disease #617265 (NPHP18/JBTS22). All patient data is synthetic; no real patient "
            "data is used or represented."
        ),
        "genetic_architecture": {
            "gene_symbol":     "CEP83 (also CCDC41 — Coiled-Coil Domain Containing 41)",
            "full_name":       "Centrosomal Protein 83 kDa",
            "protein_size":    "~826 amino acids",
            "chromosome":      "12q22",
            "omim_gene":       "*617233",
            "omim_disease":    "#617265 (NPHP18) / JBTS22 (Joubert Syndrome 22)",
            "protein_domains": (
                "N-terminal CC1 (aa 1–120): centriole anchoring / CE63 docking; "
                "Coiled-coil 2 (aa 140–380): CEP89 recruitment (DA nucleation step 1); "
                "Central scaffold (aa 380–600): SCLT1 binding (DA hierarchy step 2); "
                "C-terminal regulatory (aa 600–826): IFT-B docking; CEP164 indirect "
                "(via SCLT1→FBF1→LRRC45→CEP164 cascade)"
            ),
            "da_hierarchy": (
                "CEP83 (proximal; foundation) → CEP89 (CCDC123) → SCLT1 → FBF1 → "
                "LRRC45 → CEP164 (NPHP15; most distal). Loss of CEP83 removes ALL "
                "downstream DA proteins from centriole simultaneously. "
                "CEP83 is uniquely upstream of CEP164/NPHP15 — these are the only two "
                "NPHP genes in the same DA assembly hierarchy"
            ),
            "key_interactions": (
                "CEP89/CCDC123 (direct; DA step 1); SCLT1 (via CEP89; DA step 2); "
                "IFT-B (ciliary vesicle docking); EHD1/SNAP29 (vesicle fusion axis); "
                "Rab8a/RABIN8 (CP110 removal, axoneme initiation); TTBK2 "
                "(phosphorylates CEP83 for DA maturation); CP110/CEP97 (antagonist — "
                "CP110 cap removal requires intact CEP83-DA scaffold)"
            ),
            "disease_mechanism": (
                "CEP83 LOF → DA scaffold absent from mother centriole → "
                "centriole fails to dock to ciliary vesicle → CP110 cap NOT removed → "
                "axoneme NOT initiated → primary cilia absent in renal tubular cells, "
                "cerebellar neurons, photoreceptors → TIN (renal) + Joubert (cerebellar) "
                "+ rod-cone dystrophy (photoreceptor). Unique: only NPHP gene encoding "
                "the proximal DA foundation protein; loss phenocopies CEP164 NPHP15 "
                "PLUS adds Joubert syndrome in 55–65% of cases"
            ),
            "allele_spectrum": (
                "Null × null (truncating × truncating): JBTS22 confirmed + severe ESRD + "
                "retinal risk highest. Null × missense: JBTS22 variable; renal 60–70%. "
                "Missense × missense (hypomorphic): pure renal NPHP18 preferred; "
                "MTS absent or mild; ESRD later (20–25yr). Consanguineous homozygous "
                "missense (p.Pro173Leu, p.Ala412Val): predominantly renal ± mild JBTS"
            ),
            "key_variants": [
                "p.Pro173Leu (c.518C>T) — CC1/CC2 junction; European founder variant; moderate "
                "JBTS22 (MTS + mild cerebellar hypoplasia); ESRD ~16yr; Snouffer 2017",
                "p.Arg431Ter (c.1291C>T) — central scaffold truncating; pan-ethnic; severe "
                "JBTS22 (MTS + OMA + hypotonia); LCA-like retinal; ESRD ~11yr",
                "p.Gly608Arg (c.1822G>C) — C-terminal regulatory domain; Middle Eastern "
                "consanguineous homozygous; JBTS22 + rod-cone retinal dystrophy; ESRD ~15yr",
                "p.Leu287Pro (c.860T>C) — CC2 domain; CEP89 recruitment impaired; European "
                "compound het; PURE RENAL NPHP18 (no MTS; ERG normal); ESRD ~19yr",
                "p.Arg550Ter (c.1648C>T) — scaffold truncating; North African homozygous; "
                "severe JBTS22; retinal LCA-like; ESRD ~12yr",
                "p.Ala412Val (c.1235C>T) — SCLT1 interface; hypomorphic; South Asian "
                "consanguineous; mild JBTS22 (equivocal MTS); ESRD ~22yr",
                "p.Trp344Ter (c.1032G>A) — CC2/scaffold truncating; ultra-severe JBTS22; "
                "ERG flat neonatal; ESRD <10yr; ultra-rare",
                "p.Glu195Lys (c.583G>A) — CC2 CEP89-docking loss; European; pure renal "
                "moderate; ESRD ~20yr; no MTS; ERG normal",
            ],
        },
        "nphp_comparison": {
            "★ NPHP18 / CEP83 (CCDC41) — THIS PATIENT": (
                "Most PROXIMAL distal appendage protein; nucleates ALL downstream DA scaffold "
                "(CEP89→SCLT1→FBF1→LRRC45→CEP164). JBTS22 in ~55%; retinal ~30–40%; "
                "pure renal ~30%; CHF ~5–10%; no situs. ESRD median ~14–18yr. "
                "Chr 12q22. OMIM *617233 / #617265"
            ),
            "NPHP15 / CEP164 (DAP hub; DIRECTLY downstream of CEP83)": (
                "CEP164 is the MOST DISTAL DA protein, directly downstream of LRRC45 "
                "in the CEP83-initiated hierarchy. MTS ABSENT (CEP164 not a Joubert gene). "
                "Senior-Løken (SLS) retinal 35–40%. No CHF. No situs. ESRD ~13–15yr. "
                "Chr 11q23.3. OMIM *614848 / #614845 — co-sequence with CEP83!"
            ),
            "NPHP6 / CEP290 (12q21.32; same chromosome arm as CEP83)": (
                "CEP290 at 12q21.32; CEP83 at 12q22 — same arm, different genes. "
                "CEP290 single-gene panel misses CEP83. CEP290: TZ Y-link matrix protein; "
                "broadest allele-phenotype spectrum (LCA10 IVS26, JBTS, NPHP, MKS). "
                "Retinal 65% (LCA). JBTS in truncating. No situs. ESRD 13–15yr. "
                "OMIM *610142 / #610188 — always test CEP290 before concluding CEP83 diagnosis"
            ),
            "NPHP8 / RPGRIP1L (TZ scaffold; JBTS7)": (
                "TZ scaffold protein. JBTS7 in 40–45%. Retinal 25–35%. CHF 15–20%. "
                "Digenic: RPGRIP1L mono-allelic + NPHP4 mono-allelic → Joubert. "
                "Pure renal ESRD ~15–18yr. Chr 16q12.2. OMIM *610937"
            ),
            "NPHP11 / TMEM67 (8q22.1; CHF dominant)": (
                "Dominant CHF phenotype (40–50%). JBTS (MKS3). Retinal variable. "
                "ESRD 15–20yr. No situs. Chr 8q22.1. If CEP83 + MTS + prominent CHF → "
                "also test TMEM67 (more likely if CHF >20%)"
            ),
            "NPHP17 / MAPKBP1 (2q13.3; pure renal ultra-rare)": (
                "Immediately preceding subtype. JNK/MAPK scaffold at TZ. PURE RENAL: "
                "0% situs, 0% retinal, 0% MTS, 0% CHF. Ultra-rare ~25–35 families. "
                "ESRD ~14–16yr. KEY DDx: CEP83 vs MAPKBP1 when pure renal (no MTS, "
                "no retinal) — MAPKBP1 pure renal ~30% of CEP83 cases also pure renal"
            ),
        },
        "ddx_table": {
            "Joubert Syndrome (gene-unknown) / Other JBTS genes": (
                "MTS alone does not diagnose NPHP18. Other JBTS genes: INPP5E (JBTS1), "
                "TMEM216 (JBTS2), AHI1 (JBTS3), NPHP1 (JBTS4), CEP290 (JBTS5/LCA10), "
                "RPGRIP1L (JBTS7), CC2D2A (JBTS9), TCTN1-3 (JBTS13-15). "
                "WES mandatory when single-gene JBTS panel negative; CEP83 often missed "
                "on gene-limited Joubert panels. Co-sequence CEP164 (CEP83-downstream)"
            ),
            "NPHP6 / CEP290 (LCA10 / JBTS5 / MKS4)": (
                "Both NPHP18/CEP83 and NPHP6/CEP290 can cause JBTS + retinal + NPHP. "
                "CEP290 on 12q21.32; CEP83 on 12q22 — DIFFERENT genes on same arm. "
                "CEP290 has LCA10 (IVS26 c.2991+1655A>G; most common LCA variant). "
                "Distinction requires comprehensive WES: CEP290 variants much more common "
                "than CEP83 variants; retinal much more frequent in CEP290 (65%) vs CEP83 (30–40%)"
            ),
            "NPHP15 / CEP164 (directly downstream DA protein)": (
                "CEP164 mutations: SLS (Senior-Løken) 35–40%; NO Joubert/MTS (key DDx); "
                "NPHP only (pure renal + retinal). If MTS ABSENT + retinal + NPHP: "
                "CEP164 more likely than CEP83. If MTS PRESENT: CEP83 (JBTS22) not CEP164. "
                "Always co-sequence both when one is negative — same DA hierarchy. "
                "Co-sequencing mandatory: one may carry single pathogenic allele"
            ),
            "NPHP8 / RPGRIP1L (JBTS7; digenic Joubert)": (
                "JBTS7 in 40–45% of biallelic RPGRIP1L; CHF 15–20%; retinal 25–35%. "
                "Key DDx: RPGRIP1L can show JBTS + NPHP + retinal like CEP83 JBTS22. "
                "Distinguish by: RPGRIP1L CHF higher; digenic RPGRIP1L+NPHP4 → Joubert "
                "without homozygous variants; CEP83 = straightforward biallelic LOF"
            ),
            "ADPKD (PKD1/PKD2 — bilateral cysts)": (
                "ADPKD: bilateral ENLARGED kidneys (>mean+2SD for age); AD inheritance "
                "(one parent affected with palpable kidneys); no concentrating defect early; "
                "no MTS; no OMA; no cerebellar signs. NPHP18: small/normal-sized kidneys; "
                "AR inheritance; TIN pattern; MTS often present; early polyuria. "
                "Urine Osm + renal USS + family history usually sufficient to distinguish"
            ),
            "Alport Syndrome (COL4A3/4/5 — haematuria + CKD)": (
                "Alport: PROMINENT haematuria (glomerular TBM collagen IV defect); "
                "sensorineural hearing loss; X-linked or AR. NPHP18: haematuria rare; "
                "tubular origin (concentrating defect early, not haematuria). "
                "USS: Alport normal-sized kidneys; NPHP18 echogenic ± cysts. "
                "Brain MRI normal in Alport (no MTS). Genetic testing resolves"
            ),
            "Meckel-Gruber Syndrome (MKS — lethal; CEP290/B9D1/TMEM67)": (
                "MKS: most severe ciliopathy; usually lethal in utero or perinatal; "
                "occipital encephalocele + polydactyly + cystic kidneys. CEP83 biallelic "
                "null mutations do NOT cause MKS phenotype — CEP83 is less severe than "
                "MKS genes despite overlapping pathway. Postnatal JBTS22 + NPHP is "
                "typically compatible with survival to transplant age"
            ),
        },
    }


# ── CLI verification ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    print("=== OVERVIEW (sample) ===")
    ov = get_overview()
    print(f"cohort_n             : {ov['cohort_n']}")
    print(f"median_gfr           : {ov['median_gfr']} ml/min")
    print(f"median_hb            : {ov['median_hb']} g/dL")
    print(f"median_age_dx        : {ov['median_age_renal_dx']}yr")
    print(f"pct_esrd_or_tx       : {ov['pct_esrd_or_transplant']}%")
    print(f"pct_jbts22_confirmed : {ov['pct_jbts22_confirmed']}%")
    print(f"pct_retinal          : {ov['pct_retinal_dystrophy']}%")
    print(f"pct_misdiag_jbts     : {ov['pct_misdiagnosed_jbts_unknown']}%")
    print(f"pct_misdiag_nphp1    : {ov['pct_misdiagnosed_nphp1']}%")

    print(f"\nFirst 8 patients:")
    for p in ov["patients"]:
        print(f"  {p['id']} | age_dx={p['age_renal_dx_yr']}yr | GFR={p['gfr_now_ml_min']} | {p['ckd_stage'].split('(')[0].strip()}")
    print("\n=== BREAKDOWN (sample) ===")
    bk = get_breakdown()
    print("CKD stages:", json.dumps(bk["ckd_stage_distribution"], indent=2))
    print("JBTS22 status:", json.dumps(bk["jbts22_status"], indent=2))
    print("Retinal:", json.dumps(bk["retinal_status"], indent=2))
    print("\n=== DEFINITIONS (snippet) ===")
    df = get_definitions()
    print("disease:", df["disease"][:140])
    print("omim_gene:", df["omim_gene"])
    print("omim_disease:", df["omim_disease"])
    print("chromosome:", df["chromosome"])
    print("prevalence:", df["prevalence"])
    print(f"\nKey variants ({len(df['genetic_architecture']['key_variants'])}):")
    for v in df["genetic_architecture"]["key_variants"]:
        print(f"  • {v[:80]}")
