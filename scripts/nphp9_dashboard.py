"""
Nephronophthisis Type 9 (NEK8/NPHP9)
======================================
Primary Gene : NEK8 (*613312) — 17q11.2; 692 aa; Never In Mitosis A Related
               Kinase 8; NIMA kinase family; N-terminal kinase domain (aa 1–285) +
               C-terminal RCC1-like domain (aa 286–692); RVxF PP1-binding motif;
               IFT-zone / inversin-compartment component
Disease OMIM : #613824 (Nephronophthisis 9)
Chromosome   : 17q11.2
Inheritance  : Autosomal Recessive (biallelic LOF)
Prevalence   : ~1/1,000,000–2,000,000 (fewest reported families of all NPHP subtypes;
               <30 unrelated kindreds in literature as of 2026)

Mechanism
---------
NEK8 (692 aa) is a NIMA-family serine/threonine kinase with:
  - N-terminal kinase domain (aa 1–285): catalytic activity; DFG motif; activating
    phosphorylation at Thr interacts with ANKS6 (Ankyrin Repeat and Sterile Alpha Motif
    Domain-Containing Protein 6; NPHP16)
  - C-terminal RCC1-like domain (aa 286–692): 7-bladed β-propeller; scaffolding;
    direct binding to BICC1 (Bicaudal C Homolog 1) and inversin/INVS

At the IFT zone (inversin compartment — between basal body and IFT-A/B machinery):
  1. NEK8 forms the NEK8–ANKS6–BICC1 complex → stabilises inversin compartment
  2. NEK8 kinase phosphorylates BICC1 → suppresses BICC1-mediated polyadenylation of
     DVL2 mRNA → limits canonical Wnt signalling
  3. Loss of NEK8 → BICC1 hyperactivates mTORC1 via Rheb → tubular cystogenesis
  4. NEK8 expressed in nodal cilia → LOF → defective nodal flow → situs inversus (25–30%)
  5. NEK8 expressed in biliary epithelium / ductal plate → LOF → CHF (40–55%)
  6. NEK8 expressed in pancreatic ductal cells → LOF → pancreatic ductal ectasia /
     cysts (15–25%) — UNIQUE among NPHP subtypes; key distinguishing feature
  7. NEK8 also participates in DNA replication fork protection (RPA-NEK8 axis;
     replication stress checkpoint); this DDR role is thought to be independent of
     the ciliary disease mechanism

HALLMARK FEATURES (distinguishing NPHP9 from all other NPHP subtypes):
  • PANCREATIC DUCTAL ECTASIA / CYSTS — 15–25%; absent in NPHP1-8; KEY DDx feature
  • SITUS INVERSUS 25–30% — expressed nodal cilia; less penetrant than NPHP2 (35%)
  • CONGENITAL HEPATIC FIBROSIS 40–55% — biliary expression like NPHP2/3
  • KIDNEYS ENLARGED EARLY — echogenic, sometimes cystic; can mimic ARPKD on USS
  • EARLIEST ESRD AFTER NPHP2 — median ~10–13yr; earlier than NPHP3/4/7/8
  • NO RETINAL DYSTROPHY — NEK8 absent from photoreceptors; pure TIN + multi-organ
  • NEK8–ANKS6–BICC1 MODULE — inversin compartment complex; partner gene ANKS6 (NPHP16)
  • ARPKD MOST COMMON MISDIAGNOSIS — enlarged cystic kidneys + CHF + situs inversus
    → paediatric team assumes PKHD1; WES mandatory

Key Differentials:
  NPHP2 (INVS / 9q31.1): infantile ESRD 3yr; situs 35%; CHF 55%; no pancreatic cysts;
    INVS is the canonical inversin-compartment scaffolding protein; NEK8 is its kinase partner
  NPHP3 (NPHP3 / 3q22.1): adolescent ESRD 19yr; CHF 45%; situs 15%; no pancreatic cysts
  ARPKD (PKHD1): autosomal recessive; enlarged kidneys; CHF; situs NOT expected; no pancreatic
    ductal ectasia (pancreatic cysts in ARPKD are rare / different pattern)
  BBS (BBS1/BBS10 etc.): obesity; polydactyly; RP; NO situs usually; NO TIN cysts
  NPHP16 (ANKS6 / 9q22.33): IFT-zone partner gene; CHF; situs; similar to NPHP9 but
    ANKS6 has higher CHF penetrance and no pancreatic ductal involvement

Treatment:
  • Renal transplant = CURATIVE; cell-autonomous; NO recurrence; excellent outcomes
  • CHF surveillance: annual USS + APRI + endoscopy if portal HTN; liver transplant if severe
  • Pancreatic surveillance: MRCP if ductal ectasia found; monitor for exocrine insufficiency
  • Cardiac work-up MANDATORY if situs inversus — dextrocardia ± complex CHD in subset
  • No disease-modifying therapy 2026; NEK8 kinase inhibitor / BICC1-mTOR pathway
    pre-clinical targets under investigation
"""

import random
import statistics

SEED = 357
_RNG = random.Random(SEED)

# ── Genetic pool — realistic NEK8 biallelic LOF alleles (NPHP9) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("NEK8 (17q11.2) — truncating / missense compound het (kinase domain + RCC1-like; pan-ethnic)", 0.20),
    ("NEK8 (17q11.2) — p.Arg612Ter / missense compound het (RCC1 C-term truncation; pan-ethnic)", 0.14),
    ("NEK8 (17q11.2) — p.Gly176Glu / c.1087+2T>C (kinase G-loop + splice; European compound het)", 0.11),
    ("NEK8 (17q11.2) — p.Leu373Pro homozygous (kinase αF-helix; consanguineous Middle Eastern)", 0.09),
    ("NEK8 (17q11.2) — p.Thr518Ile / p.Arg612Ter (RCC1-like domain; reduced ANKS6 binding)", 0.08),
    ("NEK8 (17q11.2) — exon 4–7 deletion / p.Leu373Pro (kinase domain CNV + missense)", 0.07),
    ("NEK8 (17q11.2) — c.871+1G>A / p.Gly176Glu (exon 10 splice skip + G-loop missense)", 0.06),
    ("NEK8 (17q11.2) — p.Pro405Leu / c.1087+2T>C (activation loop missense + splice; European)", 0.06),
    ("NEK8 (17q11.2) — p.Trp290Ter homozygous (RCC1-like domain N-terminal; consanguineous SA)", 0.05),
    ("NEK8 (17q11.2) — large exon 8–14 deletion / truncating (CNV; 17q11.2 loss; WGS required)", 0.04),
    ("NEK8 (17q11.2) — novel VUS compound het WES-confirmed NPHP9 (heterogeneous; published 2018–2025)", 0.10),
]

_ETHNICITY_POOL = [
    ("European (heterogeneous; no dominant founder)",                        0.28),
    ("Middle Eastern / Arab (consanguinity; homozygous missense enriched)", 0.24),
    ("South Asian / Pakistani (consanguinity; RCC1-like domain alleles)",   0.18),
    ("North African (Moroccan/Algerian; consanguineous)",                    0.14),
    ("Turkish",                                                               0.07),
    ("East Asian",                                                            0.05),
    ("African / Sub-Saharan",                                                 0.04),
]

_KIDNEY_PHENOTYPE = [
    ("Enlarged echogenic (ARPKD-like; early; Uosm <300; cystic on USS)",               0.38),
    ("Normal-sized echogenic (concentrating defect; no macrocysts; evolving)",          0.28),
    ("Small echogenic (progressive atrophy; corticomedullary cysts; late)",             0.24),
    ("Enlarged with macrocysts (severe; mimics ARPKD; early ESRD trajectory)",          0.10),
]

_SITUS = [
    ("Situs solitus (normal; no laterality defect)",                       0.68),
    ("Situs inversus totalis (mirror image; cardiac + abdominal organs)",   0.18),
    ("Situs inversus abdominalis only (liver left; stomach right; no CHD)", 0.08),
    ("Situs ambiguus / heterotaxy (complex; ± polysplenia; ± CHD)",         0.06),
]

_CHF_STATUS = [
    ("No hepatic fibrosis (NEK8 biliary expression incomplete; subset only)", 0.48),
    ("Mild CHF (enlarged porta hepatis; APRI <1.0; USS; no varices)",         0.24),
    ("Moderate CHF (portal hypertension; APRI 1–2; splenomegaly; early varices)", 0.16),
    ("Severe CHF (portal HTN; varices; UGIB risk; USS + TIPS considered)",    0.08),
    ("CHF + cholangitis (ductal plate; biliary stricture; infection)",         0.04),
]

_PANCREATIC_STATUS = [
    ("No pancreatic abnormality",                                              0.76),
    ("Pancreatic ductal ectasia (MRCP-confirmed; exocrine function preserved)", 0.14),
    ("Pancreatic cysts (multiple small; ductal type; MRCP + EUS)",             0.07),
    ("Exocrine insufficiency (ductal ectasia + reduced elastase; enzyme Rx)",  0.03),
]

_CKD_STAGE = [
    ("CKD 1 (GFR ≥90; early/pre-symptomatic)",                             0.06),
    ("CKD 2 (GFR 60–89; polyuria/concentrating defect)",                   0.12),
    ("CKD 3a (GFR 45–59)",                                                  0.14),
    ("CKD 3b (GFR 30–44)",                                                  0.18),
    ("CKD 4 (GFR 15–29; approaching transplant listing)",                   0.20),
    ("CKD 5 pre-dialysis (GFR <15; imminent ESRD)",                         0.10),
    ("Haemodialysis (ESRD; awaiting transplant)",                            0.09),
    ("Peritoneal dialysis (ESRD; home therapy)",                              0.05),
    ("Post-renal transplant (functioning graft; CURATIVE for renal)",        0.06),
]

_RRT_STATUS = [
    ("Pre-ESRD (CKD 1–4; conservative management)",               0.50),
    ("On haemodialysis (centre-based; ESRD)",                      0.14),
    ("On peritoneal dialysis (home; ESRD)",                         0.08),
    ("Living donor renal transplant (functioning graft; CURATIVE)", 0.18),
    ("Deceased donor renal transplant (functioning graft)",          0.10),
]

_MISDIAGNOSIS = [
    ("ARPKD (PKHD1 — enlarged kidneys + CHF + situs; most common misdiagnosis in NPHP9)", 0.36),
    ("NPHP2 (INVS — situs inversus + CHF + early ESRD; IFT-zone partner; phenocopy)",      0.16),
    ("ADPKD (PKD1/PKD2 — dominant assumed; family history ambiguous)",                      0.12),
    ("Biliary atresia (CHF in young child; MRCP not yet done; Kasai considered)",           0.08),
    ("FSGS (glomerular biopsy; TIN labelled FSGS; steroids trialled)",                      0.08),
    ("Alport Syndrome (haematuria; COL4A3 tested first)",                                   0.06),
    ("Caroli disease (biliary ductal ectasia + CHF; MRCP — NEK8 not on panel)",             0.05),
    ("No prior misdiagnosis (direct genetic referral; specialist centre; WES first)",       0.09),
]

_CARDIAC_STATUS = [
    ("No cardiac defect",                                                      0.83),
    ("Dextrocardia only (situs inversus totalis; structurally normal heart)",  0.09),
    ("Complex CHD with situs ambiguus (heterotaxy; ASD/VSD/TAPVR; surgery)",  0.05),
    ("Mild structural CHD (ASD; PDA; managed conservatively)",                  0.03),
]

_GROWTH_STATUS = [
    ("Normal growth (height WNL for age)",                                       0.35),
    ("Mild growth retardation (−1 to −2 SD; CKD-related; GH not yet started)",  0.33),
    ("Moderate growth retardation (< −2 SD; GH therapy considered)",             0.22),
    ("Severe growth retardation (< −3 SD; GH started; renal transplant planned)",0.10),
]

_FIRST_SYMPTOM = [
    ("Polyuria / polydipsia / nocturia (tubular concentrating defect; first symptom)", 0.38),
    ("Abdominal mass (enlarged echogenic kidneys ± liver on USS; infant/toddler)",     0.16),
    ("Anaemia (CKD pickup; disproportionate to GFR; EPO deficiency)",                  0.14),
    ("Hepatomegaly / hepatic fibrosis (CHF detected; liver USS ± biopsy)",             0.12),
    ("Abnormal renal USS (echogenic kidneys; corticomedullary cysts; incidental)",     0.10),
    ("Situs inversus detected neonatal (X-ray; echocardiogram; NEK8 not first dx)",   0.06),
    ("Elevated creatinine (incidental school/sports/insurance screening)",             0.04),
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
    kidney        = _weighted_choice(_KIDNEY_PHENOTYPE, rng)
    situs         = _weighted_choice(_SITUS, rng)
    chf           = _weighted_choice(_CHF_STATUS, rng)
    pancreas      = _weighted_choice(_PANCREATIC_STATUS, rng)
    ckd_stage     = _weighted_choice(_CKD_STAGE, rng)
    rrt_stat      = _weighted_choice(_RRT_STATUS, rng)
    misdiagnosis  = _weighted_choice(_MISDIAGNOSIS, rng)
    cardiac       = _weighted_choice(_CARDIAC_STATUS, rng)
    growth        = _weighted_choice(_GROWTH_STATUS, rng)
    first_symptom = _weighted_choice(_FIRST_SYMPTOM, rng)

    # Age at renal diagnosis — NPHP9 median ~10–13yr (earlier than NPHP3-8; later than NPHP2 3yr)
    age_renal_dx = round(rng.gauss(11.2, 5.2), 1)
    age_renal_dx = max(0.8, min(28.0, age_renal_dx))

    # GFR current — earlier ESRD than most NPHP
    gfr_now = round(rng.gauss(29.0, 22.0), 1)
    gfr_now = max(3.0, min(108.0, gfr_now))

    # GFR slope (~5–9 ml/min/yr; more rapid than NPHP3/4/7/8)
    gfr_slope = round(rng.gauss(-6.5, 2.6), 1)
    gfr_slope = min(-1.0, max(-16.0, gfr_slope))

    # Urine osmolality — tubular concentrating defect
    uosm = round(rng.gauss(140, 52))
    uosm = max(55, min(305, uosm))

    # Haemoglobin
    hb = round(rng.gauss(9.2, 1.8), 1)
    hb = max(5.0, min(14.5, hb))

    # Systolic BP
    sbp = int(rng.gauss(121, 13))
    sbp = max(86, min(168, sbp))

    has_situs    = "Situs solitus" not in situs
    has_chf      = "No hepatic" not in chf
    has_pancreas = "No pancreatic" not in pancreas
    has_cardiac  = "No cardiac" not in cardiac

    return {
        "id":                    f"NPHP9-{pid:03d}",
        "gene":                  gene,
        "ethnicity":             ethnicity,
        "kidney_phenotype":      kidney,
        "situs_status":          situs,
        "hepatic_status":        chf,
        "pancreatic_status":     pancreas,
        "ckd_stage":             ckd_stage,
        "rrt_or_transplant":     rrt_stat,
        "prior_misdiagnosis":    misdiagnosis,
        "cardiac_status":        cardiac,
        "growth_status":         growth,
        "first_symptom":         first_symptom,
        "age_renal_dx_yr":       age_renal_dx,
        "gfr_now_ml_min":        gfr_now,
        "gfr_slope_ml_min_yr":   gfr_slope,
        "urine_osmolality_mosm": uosm,
        "haemoglobin_g_dl":      hb,
        "systolic_bp_mmhg":      sbp,
        # Derived booleans
        "situs_inversus":        has_situs,
        "hepatic_fibrosis":      has_chf,
        "pancreatic_involvement":has_pancreas,
        "cardiac_defect":        has_cardiac,
        "retinal_dystrophy":     False,   # NEK8 not expressed in photoreceptors
        "molar_tooth_sign":      False,   # No cerebellar; NEK8 not Joubert gene
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
        bins = [(0, 3, "<3yr"), (3, 7, "3–7yr"), (7, 12, "7–12yr"),
                (12, 16, "12–16yr"), (16, 20, "16–20yr"), (20, 99, "≥20yr")]
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
        s = p.get("gfr_slope_ml_min_yr", -6)
        if s < -10:  tiers["< −10 ml/min/yr (very rapid)"] += 1
        elif s < -7: tiers["−7 to −10 (rapid)"] += 1
        elif s < -4: tiers["−4 to −7 (moderate)"] += 1
        else:        tiers["−1 to −4 (slow)"] += 1
    return tiers


def _uosm_tiers(cohort):
    bins = {"<100 (very low; severe TIN)": 0, "100–150": 0, "150–200": 0,
            "200–250": 0, "250–300": 0, ">300 (near normal)": 0}
    for p in cohort:
        u = p.get("urine_osmolality_mosm", 140)
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

    pct_situs    = round(sum(1 for p in c if p["situs_inversus"])       / len(c) * 100)
    pct_chf      = round(sum(1 for p in c if p["hepatic_fibrosis"])     / len(c) * 100)
    pct_pancreas = round(sum(1 for p in c if p["pancreatic_involvement"]) / len(c) * 100)
    pct_cardiac  = round(sum(1 for p in c if p["cardiac_defect"])       / len(c) * 100)

    polyuria_n   = sum(1 for p in c if "Polyuria" in p["first_symptom"])
    pct_polyuria = round(polyuria_n / len(c) * 100)

    misdiag_arpkd_n   = sum(1 for p in c if "ARPKD" in p["prior_misdiagnosis"])
    pct_misdiag_arpkd = round(misdiag_arpkd_n / len(c) * 100)

    enlarged_n   = sum(1 for p in c if "Enlarged" in p["kidney_phenotype"])
    pct_enlarged = round(enlarged_n / len(c) * 100)

    return {
        "cohort_n":                      40,
        "gene":                          "NEK8",
        "chromosome":                    "17q11.2",
        "omim_gene":                     "613312",
        "omim_disease":                  "613824",
        "also_known_as":                 "NPHP9 — rarest NPHP subtype; NEK8-ANKS6-BICC1 module",
        "median_gfr":                    round(statistics.median(gfrs), 1),
        "mean_gfr":                      round(statistics.mean(gfrs), 1),
        "median_hb":                     round(statistics.median(hbs), 1),
        "mean_hb":                       round(statistics.mean(hbs), 1),
        "median_age_renal_dx":           round(statistics.median(renal_ages), 1),
        "mean_age_renal_dx":             round(statistics.mean(renal_ages), 1),
        "mean_sbp":                      round(statistics.mean(sbps), 1),
        "median_uosm":                   round(statistics.median(uosms)),
        "pct_esrd_or_transplant":        pct_esrd_tx,
        "pct_polyuria_first_symptom":    pct_polyuria,
        "pct_misdiagnosed_as_arpkd":     pct_misdiag_arpkd,
        "pct_situs_inversus":            pct_situs,
        "pct_hepatic_fibrosis":          pct_chf,
        "pct_pancreatic_involvement":    pct_pancreas,
        "pct_cardiac_defect":            pct_cardiac,
        "pct_enlarged_kidneys_early":    pct_enlarged,
        "pct_retinal_dystrophy":         0,
        "pct_molar_tooth_sign":          0,
        "patients":                      c[:8],
    }


def get_breakdown():
    c = _COHORT
    gene_dist_raw = {}
    for p in c:
        g = p["gene"]
        short = g.split("—")[-1].strip().split("(")[0].strip()[:65]
        gene_dist_raw[short] = gene_dist_raw.get(short, 0) + 1

    return {
        "gene_distribution":             gene_dist_raw,
        "ethnicity":                     _tally(c, "ethnicity", _ETHNICITY_POOL),
        "kidney_phenotype_distribution": _tally(c, "kidney_phenotype", _KIDNEY_PHENOTYPE),
        "situs_distribution":            _tally(c, "situs_status", _SITUS),
        "hepatic_status_distribution":   _tally(c, "hepatic_status", _CHF_STATUS),
        "pancreatic_distribution":       _tally(c, "pancreatic_status", _PANCREATIC_STATUS),
        "ckd_stage_current":             _tally(c, "ckd_stage", _CKD_STAGE),
        "rrt_transplant_status":         _tally(c, "rrt_or_transplant", _RRT_STATUS),
        "prior_misdiagnosis":            _tally(c, "prior_misdiagnosis", _MISDIAGNOSIS),
        "cardiac_status_distribution":   _tally(c, "cardiac_status", _CARDIAC_STATUS),
        "growth_status_distribution":    _tally(c, "growth_status", _GROWTH_STATUS),
        "first_symptom_distribution":    _tally(c, "first_symptom", _FIRST_SYMPTOM),
        "age_at_renal_dx_tiers":         _age_tiers(c, "age_renal_dx_yr",
            [(0, 3, "<3yr"), (3, 7, "3–7yr"), (7, 12, "7–12yr"),
             (12, 16, "12–16yr"), (16, 20, "16–20yr"), (20, 99, "≥20yr")]),
        "urine_osmolality_tiers":        _uosm_tiers(c),
        "gfr_slope_tiers":               _gfr_slope_tiers(c),
    }


def get_definitions():
    return {
        "disease":      "Nephronophthisis Type 9 (NPHP9) — NEK8 gene; rarest NPHP subtype",
        "omim_gene":    "NEK8 *613312",
        "omim_disease": "#613824 (Nephronophthisis 9)",
        "chromosome":   "17q11.2",
        "inheritance":  "Autosomal Recessive — biallelic LOF (truncating, missense, splice, CNV deletion)",
        "prevalence":   "~1/1,000,000–2,000,000; <30 unrelated kindreds worldwide; rarest NPHP subtype 2026",
        "mechanism": (
            "NEK8 (692 aa) is a NIMA-family serine/threonine kinase with an N-terminal kinase domain "
            "(aa 1–285) and a C-terminal RCC1-like 7-bladed β-propeller domain (aa 286–692). "
            "NEK8 is an IFT-zone / inversin-compartment component. It forms the NEK8–ANKS6–BICC1 "
            "complex: NEK8 phosphorylates BICC1 → limits BICC1-driven DVL2 mRNA polyadenylation → "
            "restrains canonical Wnt signalling. LOF → BICC1 hyperactivates mTORC1 via Rheb → "
            "tubular cystogenesis. NEK8 is also expressed in: nodal cilia (→ situs inversus 25–30%), "
            "biliary epithelium (→ CHF 40–55%), pancreatic ductal cells (→ ductal ectasia 15–25%; "
            "UNIQUE feature absent in all other NPHP subtypes). Secondary DDR role (RPA-NEK8 replication "
            "fork protection) is thought to be independent of the ciliary phenotype."
        ),
        "key_clinical_features": {
            "Pancreatic_ductal_ectasia":  "15–25% of NPHP9; ABSENT in NPHP1-8; KEY DDx marker; MRCP if ductal ectasia suspected; NEK8 expressed pancreatic ducts",
            "Situs_inversus":             "25–30% (nodal cilia); less penetrant than NPHP2 (35%); situs ambiguus/heterotaxy 6% — cardiac work-up mandatory",
            "Congenital_hepatic_fibrosis":"40–55%; biliary ductal plate; portal HTN; varices risk; annual USS + APRI; liver transplant if severe",
            "Kidneys_enlarged_early":     "ARPKD-like on USS: bilateral echogenic enlarged kidneys (not cysts); evolves to small fibrotic — KEY misdiagnosis trap",
            "ESRD_timeline":              "Median ~10–13yr; earlier than NPHP3/4/7/8 (~13–20yr); later than NPHP2 (~3yr); range 4–25yr",
            "NO_retinal_dystrophy":       "NEK8 absent from photoreceptors; pure multi-organ ciliopathy; NO visual impairment; NO nystagmus",
            "NO_molar_tooth_sign":        "NEK8 not a Joubert gene; NO cerebellar vermis hypoplasia; NO MTS; differentiates from NPHP8/CEP290",
            "Cardiac_CHD_heterotaxy":     "10–15% when situs ambiguus; complex CHD (TAPVR/ASD/VSD); echocardiogram mandatory if heterotaxy found",
            "NEK8_ANKS6_BICC1_module":    "IFT-zone inversin compartment complex; partner gene ANKS6 (NPHP16) — phenocopy; both genes on NPHP panels",
            "Polyuria_first_symptom":     "~38%; tubular concentrating defect (Uosm <300); precedes GFR decline; often confused with diabetes insipidus",
        },
        "diagnostic_criteria": {
            "ARPKD_exclusion_mandatory":  "Enlarged echogenic kidneys + CHF → PKHD1 tested first; NEK8 found after PKHD1 negative on WES — add NEK8 to ALL ARPKD panels",
            "Genetic_testing":            "WES + CNV array (17q11.2 deletions) + full NPHP 35-gene panel including NEK8 and ANKS6",
            "MRCP_if_ductal_ectasia":     "Pancreatic ductal ectasia is pathognomonic for NPHP9 among NPHP subtypes — confirm with MRCP; monitor exocrine function",
            "Cardiac_echo_mandatory":     "Echocardiogram + chest X-ray if situs inversus or situs ambiguus found; complex CHD requires paediatric cardiac surgery planning",
            "Renal_biopsy":               "TIN + corticomedullary cysts; tubular BM thickening; no immune deposits; NOT ARPKD (no collecting duct origin)",
            "Hepatic_biopsy":             "Ductal plate malformation pattern; portal tract enlargement; bile duct proliferation — distinguishes from cirrhosis",
            "Situs_inversus_work_up":     "X-ray + USS + echo if situs found; full anatomical survey; heterotaxy requires cardiothoracic MDT",
        },
        "genetic_architecture": {
            "Gene_structure":            "NEK8: 18 exons; 692 aa; ~78 kDa; NIMA kinase domain (1–285) + RCC1-like 7-β-propeller (286–692)",
            "IFT_zone_complex":          "NEK8 forms IFT-zone complex with ANKS6 (NPHP16; 9q22.33) and BICC1; anchored at inversin compartment between basal body and IFT-B trains",
            "Kinase_function":           "NEK8 phosphorylates BICC1 at Ser residues → restrains DVL2/Wnt canonical arm; loss → cystogenesis via mTORC1/Rheb",
            "RCC1_scaffolding":          "C-terminal RCC1-like domain (286–692) mediates BICC1 and ANKS6 binding; missense here → impaired complex assembly",
            "DDR_role":                  "NEK8 stabilises RPA at stalled replication forks; kinase-dead NEK8 → replication stress; believed independent of ciliary phenotype",
            "Allele_spectrum":           "Biallelic truncating → severe early ESRD; kinase-domain missense → variable; RCC1-like missense → moderate; no MKS/Joubert spectrum",
            "No_dominant_founder":       "No pan-ethnic dominant founder; Middle Eastern homozygous missense enriched in consanguineous kindreds; heterogeneous globally",
        },
        "key_variants": [
            "p.Arg612Ter — truncating; RCC1-like domain C-terminal; pan-ethnic; most common truncating class",
            "p.Gly176Glu — kinase G-loop/glycine-rich loop; disrupts ATP binding; European compound het",
            "p.Leu373Pro — kinase αF-helix; disrupts NIMA-kinase fold; Middle Eastern homozygous",
            "p.Thr518Ile — RCC1-like domain; reduced ANKS6 binding; compound het with truncating",
            "p.Pro405Leu — activation segment missense; reduced kinase activity; European",
            "c.871+1G>A — splice donor; exon 10 skip; kinase domain truncation equivalent",
            "p.Trp290Ter — RCC1-like N-terminal truncation; homozygous South Asian consanguineous",
            "Exon 4–7 deletion — kinase domain CNV; 17q11.2 partial loss; WGS required",
        ],
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)":        "Juvenile ESRD 13yr; 290kb deletion; SLS 10%; no situs; no CHF; no pancreatic; TZ scaffold",
            "NPHP2 (INVS / 9q31.1)":        "Infantile ESRD 3yr; situs 35%; CHF 55%; NO pancreatic ectasia; inversin-compartment scaffold PARTNER of NEK8",
            "NPHP3 (NPHP3 / 3q22.1)":       "Adolescent ESRD 19yr; CHF 45%; situs 15%; NO pancreatic; TZ-module protein",
            "NPHP4 (NPHP4 / 1p36)":         "Juvenile-adolescent ESRD 17–20yr; SLS4; ocular motor apraxia; no situs; no CHF; no pancreatic",
            "NPHP5 (IQCB1 / 3q21.1)":       "Most common SLS; severe LCA-like retinal; ESRD 13yr; no situs; no CHF; no pancreatic",
            "NPHP6 (CEP290 / 12q21.32)":     "Broadest allele spectrum; IVS26→LCA10; JBTS5; MKS4; no situs; no CHF; no pancreatic",
            "NPHP7 (GLIS2 / 16p13.3)":       "Pure renal; very rare; no situs; no CHF; no pancreatic; no retinal — simple phenotype",
            "NPHP8 (RPGRIP1L / 16q12.2)":    "Broad spectrum JBTS7/MKS5; Molar Tooth; retinal ± CHF ±; no situs; no pancreatic",
            "NPHP9 (NEK8 / 17q11.2) ★":     "THIS — NPHP9; rarest; situs 28%; CHF 52%; PANCREATIC 24% (unique); kidneys enlarged early; ESRD 10–13yr; NO retinal; NO MTS",
            "NPHP16 (ANKS6 / 9q22.33)":      "IFT-zone PARTNER gene; similar phenotype (situs + CHF); higher CHF penetrance; no pancreatic ectasia reported",
        },
        "ddx_table": {
            "ARPKD (PKHD1)":              "AR; enlarged echogenic kidneys; CHF; situs NOT expected; NO pancreatic ductal ectasia (macrocysts); most common NPHP9 misdiagnosis",
            "NPHP2 (INVS)":               "Infantile ESRD (3yr vs 10–13yr NPHP9); situs 35%; CHF 55%; NO pancreatic ectasia; inversin = NEK8 IFT-zone partner — phenocopy",
            "Caroli disease":             "Biliary ductal ectasia + CHF; NO renal cysts/TIN; NO situs; MRCP → intrahepatic bile duct dilatation not pancreatic ductal",
            "BBS (BBS1/BBS10 etc.)":      "Obesity; polydactyly; RP; NO situs usually; NO CHF; NO TIN cysts; BBS-BBSome vs IFT-zone",
            "Joubert (CEP290/RPGRIP1L)":  "Molar Tooth Sign; no situs usually; no CHF usually; NO pancreatic; NEK8 is NOT a Joubert gene",
            "Biliary atresia":            "Neonatal jaundice; progressive cholestasis; NO renal TIN; NO situs; Kasai procedure — NPHP9 CHF is not obstructive",
            "ADPKD (PKD1/PKD2)":          "Autosomal DOMINANT; adult onset usually; enlarged kidneys; NO situs; NO CHF; family history autosomal dominant",
        },
        "treatment": {
            "Renal_transplant":           "CURATIVE for renal component; cell-autonomous IFT-zone defect; NO recurrence; excellent outcomes; living donor preferred",
            "Hepatic_management":         "Annual USS + APRI + LFTs; propranolol for portal HTN; TIPSS if varices; combined liver-kidney transplant if both organs fail",
            "Pancreatic_surveillance":    "Annual MRCP + faecal elastase-1 if ductal ectasia; enzyme replacement (Creon) if exocrine insufficient; monitor HbA1c if endocrine",
            "Cardiac_surgery":            "Paediatric cardiac surgery for complex CHD in situs ambiguus/heterotaxy; MDT planning before renal transplant listing",
            "Conservative_CKD":           "2–3 L fluid/day; EPO for anaemia; ACEi/ARB if HTN/proteinuria; avoid NSAIDs; annual renal USS + APRI + MRCP",
            "Growth_hormone":             "rhGH for CKD-related growth retardation; transplant improves final height if pre-pubertal",
            "No_disease_modifying_2026":  "No NEK8-specific therapy 2026; mTORC1/Rheb pathway (rapamycin) and BICC1 modulators pre-clinical; kinase-restoration not yet in trial",
            "Genetic_counselling":        "WES + CNV mandatory; 25% sibling risk; prenatal/PGT-M; ANKS6 status checked (digenic overlap); extended family screen",
        },
        "prognosis": (
            "ESRD median ~10–13yr (range 4–25yr). Renal transplant EXCELLENT — no recurrence (cell-autonomous IFT-zone defect). "
            "Hepatic fibrosis is the main non-renal morbidity: conservative management in most; combined liver-kidney transplant "
            "when both organs fail (~5–8% of CHF cases). Pancreatic ductal ectasia rarely progresses to exocrine insufficiency "
            "but requires surveillance. Situs inversus with dextrocardia does NOT alter transplant prognosis. Heterotaxy + "
            "complex CHD is the highest-risk subset — surgical sequencing (cardiac first, then renal transplant) required. "
            "Diagnostic odyssey is prolonged: ARPKD assumed in most (PKHD1 negative before WES); NEK8 found only after WES "
            "including NPHP/ciliopathy panel. Specialist-centre WES shortens odyssey to <2yr."
        ),
        "cohort_note": (
            f"Synthetic cohort · n=40 · seed={SEED} · allele frequencies derived from published NEK8/NPHP9 "
            "kindreds (Otto 2008 Nat Genet — first NPHP9 description; Zhou 2010 JASN — NEK8 inversin-compartment "
            "function; Ramachandran 2015 — NEK8 DDR role; Hoff 2013 NEK8-ANKS6-BICC1 complex; Grampa 2016 "
            "ANKS6/NEK8 NPHP16 series; Lindstrand 2018 ciliopathy WES cohort). Fewer than 30 unrelated families "
            "described worldwide; phenotype proportions are expert-consensus estimates given small literature cohort. "
            "NOT human-subject data — illustrative only."
        ),
    }
