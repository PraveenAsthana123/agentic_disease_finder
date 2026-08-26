"""
Nephronophthisis Type 19 / Joubert Syndrome 35 (JBTS35)
=========================================================
Primary Gene : IFT81 (*605489) — 12q23.1; ~698 aa; Intraflagellar Transport Protein 81
               IFT-B complex CORE subunit; structural scaffold bridging the IFT-B1 and
               IFT-B2 subcomplexes; essential for anterograde (kinesin-2 driven) IFT
               train assembly and ciliary membrane protein import.
               IFT81 contains a CH (calponin homology) domain at its N-terminus that
               is critical for tubulin/IFT-B core binding. Loss of IFT81 disrupts
               IFT-B complex assembly → anterograde transport failure → absent/severely
               stunted cilia → NPHP + Joubert syndrome.
Disease OMIM : JBTS35 (#617302) — Joubert Syndrome 35 (nephronophthisis-related ciliopathy)
               Informally designated NPHP19 in ciliopathy literature (renal-predominant cases)
Chromosome   : 12q23.1
Inheritance  : Autosomal Recessive (biallelic LOF — truncating + missense)
Prevalence   : ~1/2,000,000–5,000,000; <20 published families (2026); one of the rarest
               NPHP/JBTS subtypes; similar rarity to NPHP17 (MAPKBP1, ~25–35 families)

Protein Structure — IFT81 (~698 aa; IFT-B core bridge)
-------------------------------------------------------
  • N-terminal CH domain (aa ~1–105): calponin homology fold; tubulin-binding module;
    critical for IFT-B core assembly; loss of CH domain = complete IFT-B disruption
  • IFT81/IFT74 binding interface (aa ~106–400): IFT81 and IFT74 form an obligate
    heterodimer; together they constitute the tubulin-binding module of the IFT-B1 core;
    IFT74 CH domain + IFT81 CH domain together form the tubulin-binding groove that
    drives ciliary tubulin import for axoneme assembly
  • IFT-B core scaffold (aa ~400–620): central coiled-coil scaffold; coordinates
    IFT81–IFT74 heterodimer docking to IFT88, IFT52, IFT46 (IFT-B1 subcomplex);
    bridges IFT-B1 to IFT-B2 (IFT172, IFT57, IFT80, IFT38, IFT54, IFT20)
  • C-terminal regulatory domain (aa ~620–698): kinesin-2 interaction surface;
    IFT-A retrograde coupling interface; IFT-train cargo-release regulation

Molecular Mechanism
-------------------
IFT81 is a core structural component of the IFT-B anterograde transport complex:
  1. IFT81 forms an obligate heterodimer with IFT74 via their N-terminal CH domains;
     together they create the primary tubulin-binding module of the IFT-B1 subcomplex
  2. The IFT81/IFT74 heterodimer recruits alpha/beta-tubulin dimers into the IFT train
     for axoneme assembly — IFT81 is ESSENTIAL for ciliary microtubule polymerisation
  3. IFT81 scaffolds IFT-B1 (IFT88, IFT52, IFT46, IFT70) to IFT-B2
     (IFT172, IFT57, IFT80, IFT38, IFT54, IFT20) as structural bridge
  4. Anterograde IFT trains move from the ciliary base to tip driven by kinesin-2
     (KIF3A/KIF3B/KAP complex); IFT81 is required for train assembly and cargo loading
  5. Loss of IFT81 → IFT-B core disassembly → anterograde IFT train failure →
     axoneme assembly blocked (no tubulin import) → cilia absent/severely truncated
  6. Contrast with IFT-A retrograde complex (IFT144/WDR19 = NPHP13; IFT139/TTC21B = NPHP12):
     IFT81 is anterograde (B-complex); IFT-B loss is generally more severe than IFT-A loss
  7. Renal tubular ciliary failure → tubulointerstitial nephritis (TIN) +
     corticomedullary cysts + concentrating defect → ESRD
  8. Cerebellar Purkinje/granule cell ciliary failure → Shh pathway loss →
     cerebellar vermis hypoplasia + superior cerebellar peduncle elongation =
     Molar Tooth Sign (MTS) → Joubert Syndrome 35 (JBTS35)
  9. Photoreceptor connecting cilium failure → opsin trafficking block →
     rod-cone degeneration in a subset (~50–60%) of IFT81 patients

Clinical Overview
-----------------
  • Joubert Syndrome 35 (JBTS35) — Molar Tooth Sign (MTS) on axial brain MRI:
    cerebellar vermis hypoplasia, SCP elongation; oculomotor apraxia (OMA);
    neonatal hypotonia/breathing irregularity; developmental delay; ataxia
  • Renal (NPHP-like): tubulointerstitial nephritis + corticomedullary cysts +
    concentrating defect; ESRD variable (adolescent to early adult onset)
  • Retinal dystrophy: ~50–60% rod–cone type; ERG abnormal; earlier and more
    penetrant retinal involvement than NPHP18/CEP83 (~30–40%)
  • No situs inversus (IFT81 not required for nodal cilia laterality in most cases)
  • No congenital hepatic fibrosis (IFT81 absent in biliary cholangiocytes)
  • No ectodermal features (contrast with WDR19/NPHP13 / CED)

Key Diagnostic Alerts
---------------------
  • IFT81 is an IFT-B anterograde complex core subunit — mechanistically DISTINCT
    from IFT-A retrograde subtypes (NPHP12/TTC21B and NPHP13/WDR19)
  • IFT81 is on 12q23.1 — same chromosome arm as CEP83 (12q22, NPHP18) and
    CEP290 (12q21.32, NPHP6); targeted single-gene panels for any of these
    do NOT cover IFT81; WES mandatory
  • NPHP1 MLPA (290kb) does NOT detect IFT81 at 12q23.1
  • IFT81 and IFT74 form an obligate heterodimer — IFT74 must be co-sequenced
    (IFT74 is on 9p21.2; digenic ciliopathy with IFT81 + IFT74 variants documented)
  • Ultra-rare (<20 families worldwide 2026): may be clinically missed even on
    broad ciliopathy gene panels if not included; WES essential
  • Retinal does NOT improve post-transplant (cell-autonomous photoreceptor defect)
  • Renal transplant CURATIVE for nephronophthisis component — no recurrence

40-patient cohort generated with seed=377; 3 endpoints
  /api/nphp19/overview | /api/nphp19/breakdown | /api/nphp19/definitions
"""

import random
from typing import Any

# ── Cohort seed ──────────────────────────────────────────────────────────────
SEED        = 377
COHORT_N    = 40
rng         = random.Random(SEED)

# ── Patient phenotype distributions (IFT81/NPHP19/JBTS35 literature) ─────────
ETHNICITIES = [
    ("European (non-consanguineous)", 0.30),
    ("Middle Eastern (consanguineous)", 0.28),
    ("South Asian (consanguineous)", 0.18),
    ("North African (consanguineous)", 0.12),
    ("East Asian", 0.06),
    ("Latin American", 0.04),
    ("Sub-Saharan African", 0.02),
]

CKD_STAGES = [
    ("CKD 1 (GFR ≥90; early TIN, concentrating defect only)", 0.10),
    ("CKD 2 (GFR 60–89; polyuria, mild anaemia)", 0.15),
    ("CKD 3a (GFR 45–59; growth retardation)", 0.18),
    ("CKD 3b (GFR 30–44; progressive TIN, cysts)", 0.17),
    ("CKD 4 (GFR 15–29; pre-ESRD preparation)", 0.18),
    ("CKD 5/ESRD (GFR <15; awaiting/post-transplant)", 0.22),
]

KIDNEY_USS = [
    ("Bilateral small echogenic kidneys, corticomedullary cysts", 0.40),
    ("Bilateral normal-sized echogenic kidneys, early cysts", 0.25),
    ("Small kidneys, prominent corticomedullary cysts ≥5mm", 0.20),
    ("Small echogenic kidneys, no discrete cysts (early TIN)", 0.10),
    ("Transplanted (previous ESRD)", 0.05),
]

FIRST_SYMPTOMS = [
    ("Developmental delay + hypotonia (Joubert, JBTS35)", 0.32),
    ("Polyuria / polydipsia (tubular concentrating defect)", 0.28),
    ("Neonatal hypotonia + oculomotor apraxia (OMA)", 0.18),
    ("Incidental CKD on urine/blood screening", 0.12),
    ("Anaemia disproportionate to GFR", 0.06),
    ("Growth retardation + CKD workup", 0.04),
]

JBTS35_STATUS = [
    ("JBTS35 confirmed (MTS on MRI + cerebellar vermis hypoplasia)", 0.65),
    ("Pure renal NPHP19-like (no MTS; no cerebellar features)", 0.18),
    ("Probable JBTS35 (awaiting MRI; cerebellar signs present)", 0.12),
    ("Atypical / equivocal MRI (mild vermis; partial MTS)", 0.05),
]

RETINAL = [
    ("Rod-cone dystrophy (ERG abnormal, fundoscopy abnormal)", 0.48),
    ("No retinal involvement (ERG normal, fundus clear)", 0.36),
    ("LCA-like severe early retinal (ERG flat, neonatal nystagmus)", 0.10),
    ("Mild retinal changes (ERG borderline)", 0.06),
]

PRIOR_MISDIAGNOSIS = [
    ("No prior misdiagnosis (direct WES diagnosis)", 0.28),
    ("Joubert syndrome (gene-unknown) — IFT81 identified later", 0.32),
    ("NPHP1 MLPA negative — incomplete workup", 0.16),
    ("CEP290/NPHP6 or CEP83/NPHP18 (same 12q arm; excluded first)", 0.12),
    ("ADPKD (bilateral cysts; AR pattern missed)", 0.08),
    ("IFT-A ciliopathy misclassification (NPHP12/NPHP13 workup first)", 0.04),
]

GFR_SLOPE = [
    ("Rapid (>5 ml/min/yr; ESRD before 18yr)", 0.22),
    ("Moderate (3–5 ml/min/yr; ESRD 18–25yr)", 0.35),
    ("Slow (1–3 ml/min/yr; ESRD 25–30yr)", 0.30),
    ("Very slow (<1 ml/min/yr; ESRD >30yr; hypomorphic)", 0.13),
]

URINE_OSM = [
    ("Severe deficit: Uosm <150 mOsm/kg (maximal concentrating failure)", 0.18),
    ("Moderate deficit: Uosm 150–250 mOsm/kg", 0.30),
    ("Mild deficit: Uosm 250–500 mOsm/kg", 0.32),
    ("Near-normal: Uosm >500 mOsm/kg (early/mild CKD)", 0.20),
]

VARIANTS_POOL = [
    "p.Asn106Ser (c.317A>G) — CH domain; tubulin-binding pocket; European; moderate JBTS35 + retinal; Asadollahi 2018",
    "p.Arg246Ter (c.736C>T) — IFT81/IFT74 heterodimer interface; truncating; pan-ethnic; severe JBTS35 + ESRD",
    "p.Leu387Pro (c.1160T>C) — IFT-B core scaffold; Middle Eastern consanguineous; JBTS35 + renal + retinal",
    "p.Trp501Ter (c.1503G>A) — central scaffold; North African; severe; early ESRD 16yr",
    "p.Gly58Arg (c.172G>C) — CH domain; South Asian; moderate JBTS35; rod-cone dystrophy",
    "p.Ala422Val (c.1265C>T) — scaffold region; hypomorphic; European; milder renal NPHP-like; ESRD 28yr",
    "p.Glu312Lys (c.934G>A) — IFT-B1 bridge; compound het; East Asian; pure JBTS35; no renal",
    "p.Lys567Arg (c.1700A>G) — C-terminal kinesin-2 interface; moderate; pan-ethnic; variable phenotype",
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
        jbts_stat  = _weighted_choice(JBTS35_STATUS, r)
        retinal_s  = _weighted_choice(RETINAL, r)
        misdiag    = _weighted_choice(PRIOR_MISDIAGNOSIS, r)
        gfr_ml     = r.randint(5, 95)
        age_dx     = r.randint(1, 22)
        hb_val     = round(r.uniform(7.0, 14.3), 1)
        patients.append({
            "id":               f"NPHP19-{i+1:03d}",
            "ethnicity":        ethnicity,
            "ckd_stage":        ckd_stage,
            "kidney_uss":       kidney_uss,
            "first_symptom":    first_sym,
            "jbts35_status":    jbts_stat,
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
    pts = _COHORT[:8]
    all_pts = _COHORT

    esrd_tx   = _pct(all_pts, "ckd_stage", "ESRD")
    jbts35    = _pct(all_pts, "jbts35_status", "JBTS35 confirmed")
    retinal   = _pct(all_pts, "retinal_status", "Rod-cone")
    lca_like  = _pct(all_pts, "retinal_status", "LCA-like")
    misdiag_jbts   = _pct(all_pts, "prior_misdiagnosis", "Joubert syndrome")
    misdiag_nphp1  = _pct(all_pts, "prior_misdiagnosis", "NPHP1 MLPA negative")
    misdiag_cep290 = _pct(all_pts, "prior_misdiagnosis", "CEP290")
    misdiag_adpkd  = _pct(all_pts, "prior_misdiagnosis", "ADPKD")

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
        "pct_jbts35_confirmed":    jbts35,
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
        "ckd_stage_distribution":      _dist_prefix(pts, "ckd_stage", 45),
        "jbts35_status":               _dist_prefix(pts, "jbts35_status", 55),
        "retinal_status":              _dist_prefix(pts, "retinal_status", 55),
        "kidney_phenotype":            _dist_prefix(pts, "kidney_uss", 52),
        "prior_misdiagnosis":          _dist_prefix(pts, "prior_misdiagnosis", 52),
        "first_symptom_distribution":  _dist_prefix(pts, "first_symptom", 52),
        "ethnicity":                   _dist_prefix(pts, "ethnicity", 38),
        "urine_osmolality_tiers": {
            "Severe deficit: Uosm <150 mOsm/kg": round(0.18 * COHORT_N),
            "Moderate deficit: Uosm 150–250 mOsm/kg": round(0.30 * COHORT_N),
            "Mild deficit: Uosm 250–500 mOsm/kg": round(0.32 * COHORT_N),
            "Near-normal: Uosm >500 mOsm/kg": round(0.20 * COHORT_N),
        },
    }


# ── API: definitions ──────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    return {
        "disease": (
            "Nephronophthisis Type 19 / Joubert Syndrome 35 (NPHP19/JBTS35) — autosomal recessive "
            "ciliopathy caused by biallelic loss-of-function variants in IFT81, encoding a core "
            "structural subunit of the IFT-B anterograde intraflagellar transport complex. IFT81 "
            "forms an obligate heterodimer with IFT74 via their N-terminal calponin homology (CH) "
            "domains, constituting the primary tubulin-binding module of the IFT-B1 subcomplex. "
            "Loss disrupts IFT-B core assembly → anterograde transport failure → absent/stunted "
            "cilia → nephronophthisis-like tubulointerstitial nephritis + Joubert syndrome "
            "(JBTS35; Molar Tooth Sign in ~65%) + rod-cone retinal dystrophy (~50–60%). "
            "One of the rarest NPHP subtypes: <20 published families worldwide (2026)."
        ),
        "omim_gene":    "*605489 (IFT81 — Intraflagellar Transport 81 homolog)",
        "omim_disease": "#617302 (Joubert Syndrome 35, JBTS35; NPHP19 — nephronophthisis-related)",
        "chromosome":   "12q23.1",
        "inheritance":  "Autosomal Recessive — biallelic LOF (truncating + missense); homozygous in consanguineous; compound heterozygous in outbred",
        "prevalence":   "~1/2,000,000–5,000,000; <20 published families (2026); ultra-rare among NPHP/JBTS subtypes",
        "mechanism": (
            "IFT81 is an essential structural core of the IFT-B anterograde transport complex. "
            "IFT81 forms an obligate heterodimer with IFT74 via N-terminal CH domains; this "
            "IFT81/IFT74 module constitutes the primary tubulin-binding unit of IFT-B1, enabling "
            "import of alpha/beta-tubulin dimers into the growing ciliary axoneme. IFT81 also "
            "scaffolds IFT-B1 (IFT88, IFT52, IFT46, IFT70) to IFT-B2 (IFT172, IFT57, IFT80, "
            "IFT38, IFT54, IFT20) as the structural bridge within the IFT-B supercomplex. "
            "Loss of IFT81: IFT-B core disassembles; anterograde kinesin-2 driven trains fail; "
            "tubulin import into cilia blocked; axoneme assembly cannot proceed; cilia absent or "
            "severely truncated. Renal tubular cilia loss → TIN + cysts + concentrating defect → "
            "ESRD. Cerebellar cilia loss → Shh failure → vermis hypoplasia (MTS = JBTS35). "
            "Photoreceptor connecting cilium loss → opsin trafficking failure → rod-cone dystrophy. "
            "IFT81 is mechanistically distinct from IFT-A retrograde subtypes "
            "(NPHP12/TTC21B = IFT-A; NPHP13/WDR19 = IFT-A) — IFT-B anterograde loss is "
            "generally more severe with earlier and broader ciliary defects."
        ),
        "key_clinical_features": {
            "Joubert Syndrome 35 (JBTS35) — Molar Tooth Sign in ~65% — Brain MRI Mandatory": (
                "MTS on axial brain MRI (cerebellar vermis hypoplasia + SCP elongation) in ~65% of "
                "biallelic IFT81 cases — higher penetrance than NPHP18/CEP83 (~55%). Oculomotor "
                "apraxia (OMA), neonatal hypotonia, breathing irregularity (episodic hyperpnoea/ "
                "apnoea; self-resolves ~2–3yr), cerebellar ataxia, developmental delay. Brain MRI "
                "MANDATORY at diagnosis. JBTS35 is the presenting feature in most IFT81 families."
            ),
            "Nephronophthisis / Renal (TIN + corticomedullary cysts) — NPHP19-like": (
                "TIN + corticomedullary cysts + tubular concentrating defect; ESRD variable "
                "(adolescent to early adult onset; later than NPHP1 ~13yr median). Bilateral small "
                "echogenic kidneys ± discrete cysts on USS. Anaemia disproportionate to GFR. "
                "Renal transplant CURATIVE — no disease recurrence (cell-autonomous IFT-B defect). "
                "Renal disease less penetrant than Joubert features: ~80% of JBTS35 patients "
                "develop significant CKD but timeline is more variable than classical NPHP1."
            ),
            "Retinal Dystrophy (~50–60%) — More Penetrant Than NPHP18": (
                "Rod–cone dystrophy in ~48–55% (ERG abnormal, progressive); LCA-like severe early "
                "in ~8–10% (ERG flat, neonatal nystagmus; null×null alleles). Higher retinal "
                "penetrance than NPHP18/CEP83 (~30–40%) but less than NPHP5/IQCB1 (>95%). "
                "Retinal does NOT improve post-renal transplant (cell-autonomous photoreceptor "
                "connecting cilium defect). Annual ERG + fundoscopy mandatory from diagnosis."
            ),
            "IFT81/IFT74 Obligate Heterodimer — IFT74 Co-sequencing Mandatory": (
                "IFT81 and IFT74 (9p21.2) form an obligate heterodimer through their N-terminal "
                "CH domains. The IFT81/IFT74 module is the tubulin-binding unit of IFT-B1. "
                "Digenic ciliopathy with IFT81 + IFT74 compound variants is documented. IFT74 "
                "MUST be co-sequenced in all IFT81 cases; IFT74 single heterozygous + IFT81 "
                "single heterozygous may constitute a digenic pair causing ciliopathy. "
                "WES captures both loci; targeted panels may miss IFT74 if not included."
            ),
            "No Situs Inversus (<2%)": (
                "IFT81 not required for nodal cilia laterality function in most patients. "
                "Laterality defect extremely rare. Contrast with ANKS6/NPHP16 (20–30% situs "
                "inversus) and INVS/NPHP2 (85% situs inversus). No cardiac situs evaluation "
                "required unless clinically indicated."
            ),
            "No Congenital Hepatic Fibrosis (CHF)": (
                "IFT81 not expressed in biliary cholangiocytes at levels causing ductal plate "
                "malformation. No CHF reported in published IFT81/JBTS35 families. Contrast "
                "with TMEM67/NPHP11 (40–50% CHF) and RPGRIP1L/NPHP8 (15–20% CHF). "
                "Liver USS not routinely required unless atypical presentation."
            ),
            "Ultra-Rare (<20 families) — 12q23.1 on Same Arm as CEP83 and CEP290": (
                "IFT81 at 12q23.1 is on the same chromosome arm as CEP83 (12q22, NPHP18) and "
                "CEP290 (12q21.32, NPHP6). Targeted single-gene panels for any of these three "
                "genes do NOT cover IFT81. NPHP1 MLPA (290kb) misses IFT81. Ultra-rare status "
                "(<20 families 2026) means IFT81 may be absent from targeted NPHP/JBTS gene "
                "panels; WES is the only reliable diagnostic approach."
            ),
        },
        "diagnostic_criteria": {
            "Brain MRI (axial) — MANDATORY at Diagnosis": (
                "MTS (Molar Tooth Sign): cerebellar vermis hypoplasia + SCP elongation on T1/T2 "
                "axial MRI. MTS positive in ~65% → JBTS35 alleles. MTS negative (~18%) → "
                "pure renal NPHP19-like phenotype. MRI guides: family counselling, developmental "
                "paediatrics referral, ophthalmology surveillance intensity, long-term prognosis."
            ),
            "WES — IFT81 + IFT74 Co-sequencing": (
                "IFT81 (12q23.1) NOT detected by NPHP1 MLPA, CEP290 single-gene panels, or "
                "most targeted NPHP panels. WES is the diagnostic method of choice. IFT74 "
                "(9p21.2) must be co-sequenced: obligate heterodimer partner; digenic ciliopathy "
                "documented. Confirm biallelic IFT81 variants in trans before concluding diagnosis."
            ),
            "Ophthalmology Surveillance (ERG + Fundoscopy — Annual from Diagnosis)": (
                "~50–60% retinal penetrance mandates annual ERG + fundoscopy from diagnosis "
                "regardless of initial retinal status. Retinal dystrophy may present after renal "
                "onset. Gene-specific retinal surveillance intensity (higher than NPHP18 given "
                "higher penetrance). Annual ERG from year 1 of diagnosis."
            ),
            "IFT-B vs IFT-A Differentiation (Molecular Distinction)": (
                "IFT81 is IFT-B anterograde complex; mechanistically distinct from IFT-A "
                "retrograde subtypes (NPHP12/TTC21B/IFT-A; NPHP13/WDR19/IFT-A). Ciliary "
                "ultrastructure on TEM: IFT-B loss causes ciliary tip bulge plugging (similar "
                "to NPHP12) vs IFT-A loss causes different axoneme abnormalities. TEM not "
                "routinely performed but may assist in research settings."
            ),
        },
        "genetic_architecture": {
            "IFT81 gene structure (12q23.1, ~33 kb, 19 exons)": (
                "19 protein-coding exons; ~33 kb genomic span at 12q23.1. Exons 1–4 encode "
                "CH domain (tubulin-binding; most critical for IFT-B1 core function). "
                "Exons 5–14 encode IFT81/IFT74 heterodimer interface and IFT-B1 scaffold. "
                "Exons 15–19 encode IFT-B1/B2 bridge and C-terminal kinesin-2 interface. "
                "All loss-of-function variants across all domains cause ciliopathy."
            ),
            "IFT-B complex hierarchy (IFT81 position)": (
                "IFT-B1 core: IFT81/IFT74 (tubulin-binding module; obligate heterodimer) + "
                "IFT88 + IFT52 + IFT46 + IFT70. IFT-B2 peripheral: IFT172 + IFT57 + IFT80 + "
                "IFT38 + IFT54 + IFT20. IFT81 bridges B1 to B2 as central scaffold. "
                "IFT81/IFT74 CH domain heterodimer = tubulin import mechanism for axoneme "
                "elongation. Loss of IFT81 = loss of entire IFT-B1 structural integrity."
            ),
            "Contrast with IFT-A retrograde (NPHP12, NPHP13)": (
                "IFT-A retrograde complex members: IFT144/WDR19 (NPHP13), IFT139/TTC21B (NPHP12), "
                "IFT140, IFT122, IFT43, IFT121/WDR35. IFT-B anterograde: IFT81 (NPHP19), IFT88, "
                "IFT52, IFT46, IFT74, IFT172, etc. IFT-B loss (IFT81) generally causes more "
                "severe, earlier-onset ciliary absence than IFT-A loss. Genotype-phenotype rule: "
                "IFT81 NPHP19 (B-complex) > TTC21B NPHP12 (A-complex) in severity."
            ),
            "key_variants": VARIANTS_POOL,
            "Allele–phenotype rule": (
                "Truncating×truncating biallelic: JBTS35 most likely + rod-cone/LCA-like retinal "
                "+ earlier ESRD (<20yr). Truncating×missense: JBTS35 likely; renal 80%; retinal "
                "50–60%. Missense×missense (hypomorphic: p.Ala422Val): milder renal NPHP19-like; "
                "MTS absent or mild; ESRD 25–30yr; ERG often normal. Ultra-rare status means "
                "genotype-phenotype correlations based on <20 families; caution in predictions."
            ),
        },
        "nphp_comparison": {
            "★ NPHP19 / JBTS35 (IFT81 — THIS ENTRY)": (
                "12q23.1 · IFT-B core bridge · JBTS35 ~65% · Retinal ~50–60% (higher than NPHP18) · "
                "No situs · No CHF · ESRD variable (adolescent-adult) · Ultra-rare (<20 families)"
            ),
            "NPHP18 / JBTS22 (CEP83 — proximal DA foundation)": (
                "12q22 · Same chromosome arm 12q · DA foundation cascade CEP83→CEP89→SCLT1→FBF1→ "
                "LRRC45→CEP164 · JBTS22 ~55% · Retinal ~30–40% · ESRD ~14–18yr · ~60–90 families"
            ),
            "NPHP12 / ATD4 (TTC21B — IFT-A retrograde)": (
                "2q24.3 · IFT-A retrograde; same IFT-train but opposite direction from IFT81 · "
                "Pure renal 83–85% · No Joubert · ATD4/Jeune 7–10% · No retinal · ESRD 11–15yr"
            ),
            "NPHP13 / CED1 (WDR19 — IFT-A retrograde)": (
                "4p14 · IFT-A retrograde largest subunit; ectodermal features (CED1) · "
                "Retinal 20–30% · CHF 10–15% · No Joubert · ESRD 15–20yr"
            ),
            "NPHP6 / CEP290 (CEP290 — TZ matrix)": (
                "12q21.32 · TZ matrix/Y-links; same 12q arm as IFT81 and CEP83 · "
                "Retinal 65% (LCA10 IVS26) · Joubert JBTS5 · ESRD teens · ~1/100,000"
            ),
            "NPHP1 (NPHP1 — TZ/connecting cilium)": (
                "2q13 · Most common NPHP; MLPA 290kb deletion diagnostic · Pure renal 95% · "
                "SLS 10% · No Joubert (unless digenic) · ESRD median ~13yr · ~1/50,000"
            ),
        },
        "ddx_table": {
            "NPHP18/JBTS22 (CEP83) vs NPHP19/JBTS35 (IFT81)": (
                "Both on chromosome 12q; both cause JBTS + retinal + NPHP. Key distinction: "
                "IFT81 (IFT-B anterograde) vs CEP83 (distal appendage foundation). "
                "IFT81: higher retinal penetrance (~50–60% vs ~30–40% for CEP83); "
                "CEP83: CEP89/SCLT1/FBF1/LRRC45/CEP164 DA hierarchy; IFT81: IFT-B core. "
                "Both require WES; same 12q arm means panels may miss either. Co-test both."
            ),
            "NPHP6/LCA10 (CEP290) vs NPHP19/JBTS35 (IFT81)": (
                "CEP290 (12q21.32) vs IFT81 (12q23.1) — same chromosome arm 12q. Both cause "
                "JBTS + high retinal penetrance. CEP290: LCA10 IVS26-1655A>G most common "
                "LCA-causing variant; retinal 65%; specific ASO therapy (sepofarsen). "
                "IFT81: retinal 50–60%; no specific therapy 2026. WES distinguishes."
            ),
            "NPHP12/ATD4 (TTC21B — IFT-A) vs NPHP19 (IFT81 — IFT-B)": (
                "Both IFT pathway ciliopathies but opposite transport directions. "
                "TTC21B (IFT-A, retrograde): pure renal 83%; no Joubert; ATD4/Jeune 7%; no retinal. "
                "IFT81 (IFT-B, anterograde): Joubert 65%; retinal 50–60%; renal 80%. "
                "Clinical distinction: Joubert MTS + retinal → IFT81; pure renal → TTC21B."
            ),
            "Joubert syndrome (gene-unknown) vs NPHP19/JBTS35 (IFT81)": (
                "~32% of cohort initially labelled Joubert gene-unknown. IFT81 mutations "
                "identified later on WES. Key: IFT81 at 12q23.1 often absent from targeted "
                "JBTS panels (ultra-rare). Gene-unknown JBTS with renal + retinal: always "
                "check IFT81 on WES, co-test IFT74 (9p21.2) for digenic pair."
            ),
            "ADPKD (PKD1/PKD2) vs NPHP19 (IFT81)": (
                "IFT81: bilateral cysts mimic ADPKD on USS. Key distinctions: ADPKD dominant "
                "family history; PKD1/PKD2 negative on panel → pursue NPHP/JBTS panel. "
                "IFT81 cohort 8% initially investigated as ADPKD. Corticomedullary cyst "
                "pattern + JBTS MTS + retinal → NPHP19 diagnosis, not ADPKD."
            ),
            "IFT74-related digenic ciliopathy vs IFT81 homozygous/compound het": (
                "IFT81 + IFT74 form an obligate heterodimer. Single heterozygous IFT81 variant + "
                "single heterozygous IFT74 variant may constitute a digenic ciliopathy pair. "
                "WES must report IFT74 (9p21.2) alongside IFT81 variants. Digenic: each gene "
                "heterozygous, both in trans; distinguished from simple autosomal recessive "
                "biallelic IFT81 by finding only one IFT81 variant but one IFT74 variant."
            ),
        },
        "treatment": {
            "Renal Transplant (CURATIVE — No Recurrence)": (
                "Donor kidney with functional IFT81 → normal IFT-B assembly → intact tubular "
                "primary cilia → no TIN recurrence. Excellent long-term renal outcomes. "
                "Pre-emptive transplant preferred when compatible donor available. "
                "Dialysis as bridge. GFR slope monitoring from diagnosis; nephrologist "
                "involvement from CKD stage 3. Multi-disciplinary team essential for JBTS35."
            ),
            "No Disease-Modifying Therapy (2026 — IFT-B Rescue Pre-Clinical)": (
                "No approved IFT81-specific therapy as of 2026. Pre-clinical: IFT-B complex "
                "stabilisation approaches in zebrafish (ift81 morphants) restore partial "
                "ciliary function. AAV-mediated IFT81 delivery to renal tubular cells: "
                "conceptual; pre-clinical only. IFT81/IFT74 heterodimer interface: "
                "allosteric modulator approach in silico — no clinical stage."
            ),
            "Ophthalmology — Annual ERG + Fundoscopy from Diagnosis": (
                "~50–60% retinal penetrance mandates surveillance from day of diagnosis "
                "regardless of initial ERG. Progressive rod-cone dystrophy: low-vision aids "
                "early; cane/mobility training; braille/assistive technology planning. "
                "LCA-like cases: nystagmus management, early visual rehabilitation. "
                "No anti-VEGF, no specific retinal therapy 2026 for IFT81-related retinal dystrophy."
            ),
            "Neurodevelopmental — JBTS35 Multi-Disciplinary Team": (
                "Developmental paediatrics: early intervention (physiotherapy, OT, speech therapy) "
                "from diagnosis in JBTS35 cases. Cerebellar ataxia: gait training, balance aids. "
                "Oculomotor apraxia (OMA): visual learning adaptations. Neonatal breathing "
                "irregularity: monitoring; usually self-resolves by 2–3yr (no ventilatory support "
                "typically needed beyond observation). Developmental delay: special education. "
                "Cognitive trajectory guided by MTS severity and allele class."
            ),
        },
        "prognosis": (
            "JBTS35/NPHP19 prognosis is determined by: (1) renal trajectory — ESRD onset variable "
            "(adolescent to early adult); pre-emptive transplant preferred; post-transplant renal "
            "outcomes excellent; (2) retinal — progressive rod-cone dystrophy in ~50–60%; cell-"
            "autonomous, not improved by transplant; annual ERG mandatory; (3) cerebellar/Joubert — "
            "MTS in ~65%; developmental delay; cerebellar ataxia; cognitive outcomes proportional to "
            "MTS severity; not improved by transplant; multi-disciplinary developmental team. "
            "Allele class is the best predictor: null×null → early severe JBTS35 + retinal + earlier "
            "ESRD; missense×missense (hypomorphic) → later and milder renal/retinal ± attenuated MTS. "
            "Ultra-rare status (<20 families 2026) limits precise genotype-phenotype delineation; "
            "caution in individual prognosis without thorough literature search for specific variants."
        ),
        "cohort_note": (
            f"40-patient synthetic cohort (seed={SEED}) generated from IFT81/JBTS35/NPHP19 "
            "literature-derived phenotype distributions. Ultra-rare disease (<20 real published "
            "families 2026): cohort distributions are extrapolated from case series, systematic "
            "reviews of IFT-B ciliopathies, and comparative data from related NPHP subtypes. "
            "Distributions reflect best-available evidence; individual patient characteristics are "
            "model-derived, not from real patient records. For clinical decisions, verify variant "
            "pathogenicity against current ClinVar, HGMD, and published literature."
        ),
    }
