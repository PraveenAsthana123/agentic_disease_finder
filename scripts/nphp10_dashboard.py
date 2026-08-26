"""
Nephronophthisis Type 10 (SDCCAG8/NPHP10) — Bardet-Biedl Syndrome 16 overlap
===============================================================================
Primary Gene : SDCCAG8 (*613524) — 1q44; 713 aa; Serologically Defined Colon
               Cancer Antigen 8 (also: CCCAP — Centrosomal Colon Cancer-Associated
               Protein); centrosomal / basal-body protein; NOT a transition-zone
               scaffold (distinct from NPHP1-4-8 module)
Disease OMIM : #613615 (Nephronophthisis 10 — with or without retinal dystrophy
               and/or cerebellar ataxia)
               Also: #615993 (Bardet-Biedl Syndrome 16 / BBS16 — SDCCAG8; milder
               alleles → BBS-overlap with obesity + cognitive features)
Chromosome   : 1q44
Inheritance  : Autosomal Recessive (biallelic LOF)
Prevalence   : ~1/200,000–500,000; ~100+ reported cases across literature (2026);
               commoner than NPHP7 and NPHP9 but still ultra-rare

Mechanism
---------
SDCCAG8 (713 aa) is a centrosome- and basal-body-associated protein with:
  - N-terminal coiled-coil domain (aa 1–200): centrosome targeting; required for
    subdistal appendage/distal centriole appendage anchoring
  - Central coiled-coil repeats (aa 200–550): scaffold for OFD1 and AHI1 interaction
  - C-terminal domain (aa 550–713): CEP290/NPHP6 and IQCB1/NPHP5 interaction

Molecular functions:
  1. SDCCAG8 localises to centrosome subdistal appendages → anchors basal body to
     cytoskeleton → correct ciliogenesis initiation in renal tubular epithelial cells
  2. Directly binds CEP290 (NPHP6) and IQCB1 (NPHP5) → bridges NPHP network to
     centrosomal platform → dual-network connectivity (NPHP + BBS)
  3. Interacts with OFD1 (Oral-Facial-Digital type 1 protein) at subdistal appendages
  4. Interacts with AHI1 (Abelson Helper Integration 1; JBTS3) → cerebellum expression
     → cerebellar ataxia in subset (~15–20%)
  5. Loss of SDCCAG8 → basal body anchoring failure → ciliogenesis failure → TIN in
     renal tubular cells → corticomedullary cysts → ESRD
  6. SDCCAG8 expressed in photoreceptor connecting cilium → LOF → rod-cone dystrophy
     (50–60%); retina is the second most affected organ
  7. BBS-network link (via ARL6/BBS3 and BBSome): biallelic severe truncating alleles
     → BBS16 overlap (obesity, cognitive impairment, polydactyly rare)
  8. SDCCAG8 NOT expressed in nodal cilia → NO situs inversus
  9. SDCCAG8 NOT expressed in biliary epithelium → NO congenital hepatic fibrosis
  10. SDCCAG8 NOT expressed in pancreatic ducts → NO pancreatic ductal ectasia

HALLMARK FEATURES (distinguishing NPHP10 from all other NPHP subtypes):
  • RETINAL DYSTROPHY 50–60% — most frequent retinal involvement of all ciliopathies
    with NPHP; rod-cone degeneration; may mimic Leber Congenital Amaurosis (LCA)
  • CEREBELLAR ATAXIA 15–20% — AHI1 interaction → cerebellar vermis hypoplasia
    (NOT Joubert molar tooth sign; NO ATAD3A hypoplasia); variable severity
  • BBS16 OVERLAP 15–20% — obesity, mild cognitive impairment (sometimes polydactyly
    rare); SDCCAG8 on extended BBS panels mandatory; BBS without RP is atypical
  • CENTROSOMAL (NOT TZ scaffold) — distinct from NPHP1-4-8-9; basal body anchoring
    failure (not TZ diffusion barrier collapse)
  • NO SITUS INVERSUS — SDCCAG8 absent from nodal cilia; laterality unaffected
  • NO CHF — SDCCAG8 absent from biliary epithelium; no ductal plate dysfunction
  • NO PANCREATIC DUCTAL ECTASIA — absent from pancreatic ducts (unlike NPHP9)
  • ESRD MEDIAN ~13–16yr — intermediate (NPHP1-like); kidneys small, echogenic, TIN

Key Differentials:
  NPHP5 (IQCB1 / 3q21.1): retinal + renal; severe LCA-like; ESRD 13yr; no cerebellum;
    SDCCAG8 DIRECTLY BINDS IQCB1 — must both be on SLS/NPHP panel
  NPHP6 (CEP290 / 12q21.32): retinal + renal; broader allele spectrum; IVS26→LCA10;
    SDCCAG8 DIRECTLY BINDS CEP290 — interactome partners
  BBS (BBS1/BBS10 etc.): polydactyly; obesity; RP; NO TIN cysts; BBS without
    polydactyly + with TIN → SDCCAG8 mandatory
  Joubert (AHI1/CEP290): molar tooth sign; SDCCAG8 ataxia is NOT Joubert (no MTS)
  ALMS1 (Alström): cardiomyopathy + cone-rod + CKD + deafness; no ataxia; no retinal RP
  LCA (RPGRIP1/CEP290 etc.): retinal only presentation; no renal workup → ESRD missed

Treatment:
  • Renal transplant = CURATIVE; cell-autonomous centrosomal defect; NO recurrence
  • Retinal disease: low-vision rehabilitation; does NOT improve post-transplant
  • Cerebellar ataxia: physiotherapy; occupational therapy; independent of renal Tx
  • BBS-overlap: dietary management; metabolic syndrome monitoring; ophthalmology
  • No disease-modifying therapy 2026; SDCCAG8 centrosome stabiliser pre-clinical
"""

import random
import statistics

SEED = 359
_RNG = random.Random(SEED)

# ── Genetic pool — realistic SDCCAG8 biallelic LOF alleles (NPHP10) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("SDCCAG8 (1q44) — truncating / missense compound het (coiled-coil + C-terminal; pan-ethnic)", 0.22),
    ("SDCCAG8 (1q44) — p.Arg232Ter / missense compound het (exon 6 truncation; pan-ethnic)", 0.16),
    ("SDCCAG8 (1q44) — p.Arg117Ter homozygous (N-terminal coiled-coil truncation; European)", 0.10),
    ("SDCCAG8 (1q44) — p.Leu147Pro / c.1012+1G>A (CC-domain missense + splice; pan-ethnic)", 0.09),
    ("SDCCAG8 (1q44) — p.Ala505Glu homozygous (C-terminal; reduced CEP290 binding; Middle Eastern)", 0.08),
    ("SDCCAG8 (1q44) — p.Phe457Ser / truncating (BBS16-overlap allele; milder; obesity + retinal)", 0.07),
    ("SDCCAG8 (1q44) — exon 1–4 deletion / p.Arg232Ter (N-terminal CNV + truncating; 1q44 partial)", 0.07),
    ("SDCCAG8 (1q44) — c.783+2T>G / p.Leu147Pro (exon 9 splice skip + CC missense; South Asian)", 0.06),
    ("SDCCAG8 (1q44) — p.Pro696Leu / p.Arg117Ter (C-terminal missense + truncating; compound het)", 0.05),
    ("SDCCAG8 (1q44) — large exon deletion (CNV; 1q44 loss; WGS required; pan-ethnic)",             0.04),
    ("SDCCAG8 (1q44) — novel VUS compound het WES-confirmed NPHP10 (heterogeneous; 2018–2025)",      0.06),
]

_ETHNICITY_POOL = [
    ("European (heterogeneous; no dominant founder)",                        0.32),
    ("Middle Eastern / Arab (consanguinity; homozygous C-terminal enriched)", 0.22),
    ("South Asian / Pakistani (consanguinity; splice + missense alleles)",   0.18),
    ("North African (Moroccan/Algerian; consanguineous)",                    0.12),
    ("Turkish",                                                               0.07),
    ("East Asian",                                                            0.05),
    ("African / Sub-Saharan",                                                 0.04),
]

_KIDNEY_PHENOTYPE = [
    ("Small echogenic (corticomedullary cysts; loss CMD; classic TIN; NPHP1-like pattern)", 0.52),
    ("Normal-sized echogenic (concentrating defect; no macrocysts; early disease)",         0.28),
    ("Small echogenic bilaterally (advanced TIN; late-stage; approaching ESRD)",            0.14),
    ("Small with macrocysts (severe; bilaterally shrunken; late presentation)",              0.06),
]

_RETINAL_STATUS = [
    ("No retinal dystrophy (SDCCAG8 expression insufficient for degeneration in this patient)", 0.43),
    ("Rod-cone dystrophy (RP-like; night blindness → visual field loss; ERG abnormal)",        0.32),
    ("LCA-like severe retinal (early-onset; flat ERG; nystagmus; visual impairment infancy)",  0.15),
    ("Mild retinal (photophobia; reduced visual acuity; ERG borderline; low-vision aids)",     0.10),
]

_CEREBELLAR_STATUS = [
    ("No cerebellar abnormality (SDCCAG8 AHI1-interaction insufficient in this patient)", 0.82),
    ("Mild cerebellar ataxia (gait instability; MRI vermis hypoplasia; no MTS)",           0.11),
    ("Moderate cerebellar ataxia (broad-based gait; truncal ataxia; physiotherapy)",       0.05),
    ("Severe cerebellar ataxia (non-ambulatory; MRI abnormal; OT + PT + speech)",         0.02),
]

_BBS_OVERLAP = [
    ("No BBS features (pure NPHP10; no obesity; no cognitive impairment)",                 0.80),
    ("BBS16-overlap: obesity + mild cognitive impairment (BMI >30; IQ borderline)",        0.12),
    ("BBS16-overlap: obesity + retinal + mild polydactyly (toe; post-axial; 1 foot)",      0.05),
    ("BBS16-overlap: obesity + moderate cognitive impairment (supported living; SDCCAG8)", 0.03),
]

_CKD_STAGE = [
    ("CKD 1 (GFR ≥90; early/pre-symptomatic)",                             0.08),
    ("CKD 2 (GFR 60–89; polyuria/concentrating defect)",                   0.14),
    ("CKD 3a (GFR 45–59)",                                                  0.16),
    ("CKD 3b (GFR 30–44)",                                                  0.18),
    ("CKD 4 (GFR 15–29; approaching transplant listing)",                   0.18),
    ("CKD 5 pre-dialysis (GFR <15; imminent ESRD)",                         0.10),
    ("Haemodialysis (ESRD; awaiting transplant)",                            0.08),
    ("Peritoneal dialysis (ESRD; home therapy)",                              0.04),
    ("Post-renal transplant (functioning graft; CURATIVE for renal)",        0.04),
]

_RRT_STATUS = [
    ("Pre-ESRD (CKD 1–4; conservative management)",               0.56),
    ("On haemodialysis (centre-based; ESRD)",                      0.14),
    ("On peritoneal dialysis (home; ESRD)",                         0.07),
    ("Living donor renal transplant (functioning graft; CURATIVE)", 0.16),
    ("Deceased donor renal transplant (functioning graft)",          0.07),
]

_MISDIAGNOSIS = [
    ("LCA (retinal only; no renal workup; Leber Congenital Amaurosis panel → SDCCAG8 not tested)", 0.34),
    ("BBS (obesity + retinal + renal → BBS assumed; SDCCAG8 not on standard BBS panel)",           0.18),
    ("ADPKD (cystic kidneys; AD assumed; PKD1/PKD2 tested first)",                                  0.14),
    ("NPHP5 or NPHP6 (retinal + renal; CEP290/IQCB1 tested first; SDCCAG8 interactor missed)",    0.10),
    ("FSGS (glomerular biopsy; TIN labelled FSGS; steroids trialled; no genetic work-up)",          0.08),
    ("Alport Syndrome (haematuria; COL4A3/4A4 tested first)",                                       0.06),
    ("Retinitis Pigmentosa + ADPKD (dual misdiagnosis; RP gene panel negative; PKD assumed)",      0.04),
    ("No prior misdiagnosis (direct genetic referral; specialist centre; WES first)",               0.06),
]

_GROWTH_STATUS = [
    ("Normal growth (height WNL for age)",                                       0.38),
    ("Mild growth retardation (−1 to −2 SD; CKD-related; GH not yet started)",  0.30),
    ("Moderate growth retardation (< −2 SD; GH therapy considered)",             0.22),
    ("Severe growth retardation (< −3 SD; GH started; renal transplant planned)",0.10),
]

_FIRST_SYMPTOM = [
    ("Polyuria / polydipsia / nocturia (tubular concentrating defect; first symptom)", 0.30),
    ("Visual symptoms / nystagmus (retinal dystrophy detected; ophthalmology referral)", 0.28),
    ("Abnormal renal USS (echogenic small kidneys; corticomedullary cysts; incidental)", 0.16),
    ("Anaemia (CKD pickup; disproportionate to GFR; EPO deficiency)",                   0.12),
    ("Gait disturbance (cerebellar ataxia noticed; paediatric neurology → genetics)",   0.08),
    ("Elevated creatinine (incidental school/sports/insurance screening)",               0.06),
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
    retinal       = _weighted_choice(_RETINAL_STATUS, rng)
    cerebellar    = _weighted_choice(_CEREBELLAR_STATUS, rng)
    bbs_overlap   = _weighted_choice(_BBS_OVERLAP, rng)
    ckd_stage     = _weighted_choice(_CKD_STAGE, rng)
    rrt_stat      = _weighted_choice(_RRT_STATUS, rng)
    misdiagnosis  = _weighted_choice(_MISDIAGNOSIS, rng)
    growth        = _weighted_choice(_GROWTH_STATUS, rng)
    first_symptom = _weighted_choice(_FIRST_SYMPTOM, rng)

    # Age at renal diagnosis — NPHP10 median ~13–16yr (similar to NPHP1; wider range)
    age_renal_dx = round(rng.gauss(14.2, 5.8), 1)
    age_renal_dx = max(1.5, min(32.0, age_renal_dx))

    # GFR current
    gfr_now = round(rng.gauss(32.0, 24.0), 1)
    gfr_now = max(3.0, min(112.0, gfr_now))

    # GFR slope (~4–8 ml/min/yr)
    gfr_slope = round(rng.gauss(-5.8, 2.4), 1)
    gfr_slope = min(-1.0, max(-15.0, gfr_slope))

    # Urine osmolality — tubular concentrating defect
    uosm = round(rng.gauss(148, 54))
    uosm = max(58, min(308, uosm))

    # Haemoglobin
    hb = round(rng.gauss(9.6, 1.7), 1)
    hb = max(5.2, min(14.8, hb))

    # Systolic BP — normotensive (like NPHP1)
    sbp = int(rng.gauss(118, 12))
    sbp = max(84, min(165, sbp))

    has_retinal    = "No retinal" not in retinal
    has_cerebellar = "No cerebellar" not in cerebellar
    has_bbs        = "No BBS" not in bbs_overlap

    return {
        "id":                     f"NPHP10-{pid:03d}",
        "gene":                   gene,
        "ethnicity":              ethnicity,
        "kidney_phenotype":       kidney,
        "retinal_status":         retinal,
        "cerebellar_status":      cerebellar,
        "bbs_overlap_status":     bbs_overlap,
        "ckd_stage":              ckd_stage,
        "rrt_or_transplant":      rrt_stat,
        "prior_misdiagnosis":     misdiagnosis,
        "growth_status":          growth,
        "first_symptom":          first_symptom,
        "age_renal_dx_yr":        age_renal_dx,
        "gfr_now_ml_min":         gfr_now,
        "gfr_slope_ml_min_yr":    gfr_slope,
        "urine_osmolality_mosm":  uosm,
        "haemoglobin_g_dl":       hb,
        "systolic_bp_mmhg":       sbp,
        # Derived booleans
        "retinal_dystrophy":      has_retinal,
        "cerebellar_ataxia":      has_cerebellar,
        "bbs_overlap":            has_bbs,
        "situs_inversus":         False,   # SDCCAG8 not in nodal cilia
        "hepatic_fibrosis":       False,   # SDCCAG8 not in biliary epithelium
        "pancreatic_involvement": False,   # SDCCAG8 not in pancreatic ducts
        "molar_tooth_sign":       False,   # Ataxia ≠ Joubert; no MTS
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
        u = p.get("urine_osmolality_mosm", 148)
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

    pct_retinal    = round(sum(1 for p in c if p["retinal_dystrophy"])      / len(c) * 100)
    pct_cerebellar = round(sum(1 for p in c if p["cerebellar_ataxia"])      / len(c) * 100)
    pct_bbs        = round(sum(1 for p in c if p["bbs_overlap"])            / len(c) * 100)

    polyuria_n   = sum(1 for p in c if "Polyuria" in p["first_symptom"])
    pct_polyuria = round(polyuria_n / len(c) * 100)

    visual_first_n   = sum(1 for p in c if "Visual" in p["first_symptom"] or "nystagmus" in p["first_symptom"])
    pct_visual_first = round(visual_first_n / len(c) * 100)

    misdiag_lca_n   = sum(1 for p in c if "LCA" in p["prior_misdiagnosis"])
    pct_misdiag_lca = round(misdiag_lca_n / len(c) * 100)

    return {
        "cohort_n":                      40,
        "gene":                          "SDCCAG8",
        "chromosome":                    "1q44",
        "omim_gene":                     "613524",
        "omim_disease":                  "613615",
        "also_known_as":                 "NPHP10 / BBS16 — centrosomal NPHP; retinal + renal + cerebellar ± BBS overlap",
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
        "pct_visual_symptoms_first":     pct_visual_first,
        "pct_misdiagnosed_as_lca":       pct_misdiag_lca,
        "pct_retinal_dystrophy":         pct_retinal,
        "pct_cerebellar_ataxia":         pct_cerebellar,
        "pct_bbs_overlap":               pct_bbs,
        "pct_situs_inversus":            0,
        "pct_hepatic_fibrosis":          0,
        "pct_pancreatic_involvement":    0,
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
        "gene_distribution":              gene_dist_raw,
        "ethnicity":                      _tally(c, "ethnicity", _ETHNICITY_POOL),
        "kidney_phenotype_distribution":  _tally(c, "kidney_phenotype", _KIDNEY_PHENOTYPE),
        "retinal_status_distribution":    _tally(c, "retinal_status", _RETINAL_STATUS),
        "cerebellar_status_distribution": _tally(c, "cerebellar_status", _CEREBELLAR_STATUS),
        "bbs_overlap_distribution":       _tally(c, "bbs_overlap_status", _BBS_OVERLAP),
        "ckd_stage_current":              _tally(c, "ckd_stage", _CKD_STAGE),
        "rrt_transplant_status":          _tally(c, "rrt_or_transplant", _RRT_STATUS),
        "prior_misdiagnosis":             _tally(c, "prior_misdiagnosis", _MISDIAGNOSIS),
        "growth_status_distribution":     _tally(c, "growth_status", _GROWTH_STATUS),
        "first_symptom_distribution":     _tally(c, "first_symptom", _FIRST_SYMPTOM),
        "age_at_renal_dx_tiers":          _age_tiers(c, "age_renal_dx_yr",
            [(0, 3, "<3yr"), (3, 7, "3–7yr"), (7, 12, "7–12yr"),
             (12, 16, "12–16yr"), (16, 20, "16–20yr"), (20, 99, "≥20yr")]),
        "urine_osmolality_tiers":         _uosm_tiers(c),
        "gfr_slope_tiers":                _gfr_slope_tiers(c),
    }


def get_definitions():
    return {
        "disease":      "Nephronophthisis Type 10 (NPHP10) — SDCCAG8 gene; centrosomal NPHP; NPHP + retinal + cerebellar ± BBS16 overlap",
        "omim_gene":    "SDCCAG8 *613524",
        "omim_disease": "#613615 (Nephronophthisis 10) / #615993 (Bardet-Biedl Syndrome 16 / BBS16)",
        "chromosome":   "1q44",
        "inheritance":  "Autosomal Recessive — biallelic LOF (truncating, missense, splice, CNV deletion)",
        "prevalence":   "~1/200,000–500,000; ~100+ reported cases (2026); commoner than NPHP7/9; ultra-rare",
        "mechanism": (
            "SDCCAG8 (713 aa) is a centrosomal and basal-body-associated protein (NOT a transition-zone "
            "scaffold like NPHP1-4-8). It localises to centriole subdistal appendages to anchor the basal "
            "body to the cytoskeleton, enabling ciliogenesis initiation. Key interactions: CEP290 (NPHP6) "
            "and IQCB1 (NPHP5) — direct binding bridges the NPHP TZ-network to the centrosomal platform. "
            "AHI1 (Jouberin/JBTS3) interaction → cerebellar expression → ataxia in 15–20% of cases. "
            "LOF → basal body anchoring failure → ciliogenesis failure in renal tubular epithelial cells "
            "→ TIN → corticomedullary cysts → ESRD. SDCCAG8 expressed in photoreceptor connecting cilium "
            "→ rod-cone dystrophy (50–60%). BBS-network link via BBSome (ARL6/BBS3) → biallelic severe "
            "truncating alleles → BBS16 overlap (obesity + cognitive ± polydactyly). NOT expressed in "
            "nodal cilia, biliary epithelium, or pancreatic ducts → NO situs inversus, CHF, or pancreatic "
            "ductal ectasia."
        ),
        "key_clinical_features": {
            "Retinal_dystrophy":          "50–60%; rod-cone degeneration; may mimic LCA (early-onset flat ERG); most frequent retinal phenotype of centrosomal NPHP; DOES NOT improve post-renal transplant",
            "Cerebellar_ataxia":          "15–20%; AHI1 interaction → cerebellar vermis hypoplasia; variable severity; NOT Joubert (no molar tooth sign); gait instability; physiotherapy",
            "BBS16_overlap":              "15–20%; severe truncating alleles → obesity + mild cognitive impairment ± rare post-axial polydactyly; SDCCAG8 on ALL extended BBS panels mandatory",
            "Renal_ESRD_timeline":        "Median ~13–16yr (range 4–30yr); small echogenic kidneys; corticomedullary cysts; TIN — similar to NPHP1/5/6 pattern",
            "NO_situs_inversus":          "SDCCAG8 absent from nodal cilia; laterality always normal; differentiates from NPHP2/3/9",
            "NO_CHF":                     "SDCCAG8 absent from biliary epithelium; NO congenital hepatic fibrosis; NO portal HTN; differentiates from NPHP2/3/9",
            "NO_pancreatic_ectasia":      "SDCCAG8 absent from pancreatic ducts; NO ductal ectasia (unique to NPHP9); differentiates from NEK8/NPHP9",
            "Centrosomal_not_TZ":         "Distinct from NPHP1-4-8 (TZ scaffold); basal body anchoring failure mechanism; subdistal appendage complex",
            "SDCCAG8_interactome":        "Direct binding to CEP290 (NPHP6), IQCB1 (NPHP5), OFD1, AHI1 — must be on ALL NPHP+LCA+SLS+BBS extended panels",
            "Polyuria_first_symptom":     "~30%; tubular concentrating defect (Uosm <300); precedes GFR decline; often missed vs retinal presentation first",
        },
        "diagnostic_criteria": {
            "LCA_exclusion_mandatory":    "Retinal-only workup misses renal disease — SDCCAG8 is on ALL LCA gene panels; renal USS mandatory in all LCA patients",
            "BBS_panel_inclusion":        "BBS without polydactyly + with CKD/TIN → SDCCAG8 mandatory; extended BBS panel (>20 genes) includes SDCCAG8",
            "Genetic_testing":            "WES + CNV array (1q44 deletions) + full NPHP/SLS/BBS/Joubert multi-panel including SDCCAG8",
            "ERG_mandatory":              "Electroretinogram at diagnosis in ALL NPHP patients — SDCCAG8 retinal involvement in 50–60%; identifies before severe loss",
            "Brain_MRI":                  "MRI if ataxia or developmental delay — cerebellar vermis hypoplasia (NOT molar tooth sign); differentiates from Joubert",
            "Renal_biopsy":               "TIN + corticomedullary cysts; tubular BM thickening; no immune deposits; NOT nephrotic; NOT FSGS",
            "CEP290_IQCB1_interaction":   "SDCCAG8 binds CEP290 and IQCB1 directly; if CEP290/IQCB1 panel negative, SDCCAG8 is the next interactor to test",
        },
        "genetic_architecture": {
            "Gene_structure":            "SDCCAG8: 10 exons (original literature); 713 aa; ~80 kDa; centrosomal localisation sequence (aa 1–200) + coiled-coil + CEP290/IQCB1-binding C-terminus",
            "Centrosomal_platform":      "SDCCAG8 at centriole subdistal appendages/distal appendages; anchors basal body; required for ciliary axoneme extension (initiation step)",
            "NPHP_network_bridge":       "SDCCAG8 directly binds CEP290 (NPHP6 / TZ-matrix scaffold) and IQCB1 (NPHP5 / IQ-calmodulin RPGR-link) — centrosomal ↔ TZ network connector",
            "AHI1_interaction":          "SDCCAG8 binds AHI1 (Jouberin; JBTS3) → cerebellar development connection; AHI1 loss → Joubert; SDCCAG8-AHI1 dysfunction → ataxia without MTS",
            "BBS16_pathway":             "SDCCAG8 connects to BBSome via ARL6/BBS3 GTPase → severe truncating biallelic alleles → BBSOME disruption → BBS16 phenotype (obesity + cognitive)",
            "Allele_spectrum":           "Truncating (nonsense, frameshift, splice) → severe; severe missense (CC-domain) → NPHP10; milder missense (C-terminal) → BBS16-overlap; no lethal MKS alleles",
            "No_dominant_founder":       "No pan-ethnic dominant founder; European heterogeneous; Middle Eastern/South Asian consanguineous homozygous alleles enriched",
        },
        "key_variants": [
            "p.Arg232Ter — most common truncating allele; exon 6; pan-ethnic; originally reported Otto et al 2010 Nat Genet",
            "p.Arg117Ter — N-terminal CC truncation; European; homozygous in consanguineous families",
            "p.Leu147Pro — CC-domain missense; centrosome-targeting loop disrupted; pan-ethnic compound het",
            "p.Ala505Glu — C-terminal domain; reduced CEP290 binding; Middle Eastern consanguineous",
            "p.Phe457Ser — moderate missense; BBS16-overlap allele (milder); associated with obesity phenotype",
            "p.Pro696Leu — C-terminal near IQCB1-binding site; compound het; European",
            "c.1012+1G>A — splice donor; exon skip; frameshift equivalent; South Asian compound het",
            "c.783+2T>G — splice donor; exon 9 skip; central CC-domain truncation equivalent",
            "Exon 1–4 deletion — N-terminal CNV; 1q44 partial loss; array CGH/WGS required",
        ],
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)":         "Juvenile ESRD 13yr; 290kb deletion; SLS 10%; no situs; no CHF; no retinal (most); TZ scaffold",
            "NPHP2 (INVS / 9q31.1)":         "Infantile ESRD 3yr; situs 35%; CHF 55%; no retinal; inversin-compartment scaffold",
            "NPHP3 (NPHP3 / 3q22.1)":        "Adolescent ESRD 19yr; CHF 45%; situs 15%; no retinal; TZ-module protein",
            "NPHP4 (NPHP4 / 1p36)":          "Juvenile-adolescent ESRD 17–20yr; SLS4; ocular motor apraxia; no situs; no CHF; TZ scaffold",
            "NPHP5 (IQCB1 / 3q21.1)":        "Most common SLS; severe LCA-like retinal; ESRD 13yr; no situs; no CHF; SDCCAG8 DIRECT PARTNER",
            "NPHP6 (CEP290 / 12q21.32)":      "Broadest allele spectrum; IVS26→LCA10; JBTS5; MKS4; no situs; SDCCAG8 DIRECT PARTNER",
            "NPHP7 (GLIS2 / 16p13.3)":        "Pure renal; very rare; no situs; no CHF; no retinal; simple phenotype",
            "NPHP8 (RPGRIP1L / 16q12.2)":     "JBTS7/MKS5; Molar Tooth; retinal ± CHF ±; no situs; no pancreatic",
            "NPHP9 (NEK8 / 17q11.2)":         "Rarest NPHP; situs 28%; CHF 52%; PANCREATIC 24% (unique); enlarged kidneys; NO retinal; NO MTS",
            "NPHP10 (SDCCAG8 / 1q44) ★":     "THIS — NPHP10; centrosomal; retinal 57%; cerebellar 18%; BBS16-overlap 20%; NO situs; NO CHF; NO pancreatic; ESRD 13–16yr",
        },
        "ddx_table": {
            "LCA (RPGRIP1/CEP290 panels)":  "Retinal only; SDCCAG8 missed → renal monitoring omitted → ESRD at unscheduled presentation; SDCCAG8 on ALL LCA panels mandatory",
            "BBS (BBS1/BBS10/BBS7 etc.)":  "Obesity + polydactyly + RP; standard 18-gene BBS panel often excludes SDCCAG8; BBS without polydactyly + TIN → test SDCCAG8",
            "NPHP5 (IQCB1)":               "Retinal + renal (most common SLS); severe LCA-like; NO cerebellar; NO BBS overlap; SDCCAG8 direct IQCB1 interactor — interactome panels critical",
            "NPHP6 (CEP290)":              "Retinal + renal; IVS26→LCA10; no cerebellar; SDCCAG8 direct CEP290 interactor — always test both if one negative",
            "Joubert (AHI1/CEP290)":       "Molar Tooth Sign on MRI; cerebellar vermis hypoplasia WITH MTS; SDCCAG8 ataxia has NO MTS — MRI differentiates",
            "Alström (ALMS1)":             "Cone-rod + CKD + cardiomyopathy + deafness; no RP; no cerebellar; no BBS features; ALMS1 gene panel",
            "ADPKD (PKD1/PKD2)":           "Autosomal DOMINANT; adult onset usually; enlarged kidneys; family history dominant — AR + small kidneys → NPHP10",
            "FSGS":                        "Glomerular biopsy; nephrotic proteinuria; NO TIN cysts; NO retinal; SDCCAG8 missed without genetic work-up",
        },
        "treatment": {
            "Renal_transplant":           "CURATIVE for renal component; cell-autonomous centrosomal defect; NO recurrence; excellent outcomes; living donor preferred",
            "Retinal_management":         "Low-vision rehabilitation; ERG monitoring; visual aids; Braille/AT if severe; retinal does NOT improve post-renal transplant; ophthalmology annually",
            "Cerebellar_ataxia":          "Physiotherapy (gait, balance); occupational therapy; speech therapy if severe; MRI surveillance; independent of renal transplant outcome",
            "BBS_overlap_management":     "Dietician for obesity; metabolic syndrome monitoring (HbA1c, lipids, BP); cognitive support; educational plan if cognitive impairment",
            "Conservative_CKD":           "Adequate hydration (polyuria → dehydration); EPO for anaemia; ACEi/ARB if proteinuric; avoid NSAIDs; annual USS",
            "Growth_hormone":             "rhGH for CKD-related growth retardation; transplant improves final height if pre-pubertal",
            "No_disease_modifying_2026":  "No SDCCAG8-specific therapy 2026; centrosome-stabilising agents and ciliogenesis restoration strategies pre-clinical; no approved trial",
            "Genetic_counselling":        "WES + CNV mandatory; 25% sibling risk; prenatal/PGT-M; extended NPHP+BBS panel; CEP290+IQCB1 interactor screen; family cascade",
        },
        "prognosis": (
            "ESRD median ~13–16yr (range 4–30yr). Renal transplant EXCELLENT — no recurrence (cell-autonomous centrosomal defect). "
            "Retinal dystrophy is the main non-renal morbidity: low-vision rehabilitation from diagnosis; does not improve post-transplant. "
            "Cerebellar ataxia (15–20%) is independent of renal outcome; physiotherapy improves function; rarely progressive to severe disability. "
            "BBS16-overlap (15–20%) — obesity + metabolic syndrome requires lifelong dietary management. "
            "Diagnostic odyssey is prolonged: LCA or BBS assumed in most (retinal/obesity dominated); renal monitoring omitted until CKD symptomatic. "
            "Specialist-centre WES with full NPHP+SLS+BBS interactome panel identifies SDCCAG8 earliest. "
            "SDCCAG8 must be on ALL LCA, NPHP, SLS, BBS, and extended ciliopathy panels — it bridges multiple disease networks."
        ),
        "cohort_note": (
            f"Synthetic cohort · n=40 · seed={SEED} · allele frequencies derived from published SDCCAG8/NPHP10 "
            "kindreds (Otto et al 2010 Nature Genetics — SDCCAG8 identified as NPHP10; Schäfer et al 2011 — "
            "centrosomal localisation; Lindstrand et al 2018 — ciliopathy WES cohort; Ramachandran 2015 — "
            "SDCCAG8-CEP290 interaction; Sayer et al BBS16 registry). ~100+ unrelated families described "
            "worldwide; phenotype proportions are expert-consensus estimates. "
            "NOT human-subject data — illustrative only."
        ),
    }
