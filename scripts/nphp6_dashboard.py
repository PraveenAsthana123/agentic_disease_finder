"""
Nephronophthisis Type 6 / Senior-Løken Syndrome 6 (CEP290/NPHP6)
=================================================================
Primary Gene : CEP290 (*610142) — 12q21.32; 2480 aa; Centrosomal Protein 290 kDa;
               essential TZ matrix scaffold, Y-link assembly, axoneme microtubule anchoring
Disease OMIM : #610189 (Senior-Løken Syndrome 6 / Nephronophthisis 6)
               Note: OMIM #204000 (Leber Congenital Amaurosis 10) same gene / IVS26 allele
Chromosome   : 12q21.32
Inheritance  : Autosomal Recessive (biallelic LOF)
Prevalence   : ~1/100,000–1/300,000 live births (NPHP6/SLS6 subtype; broader CEP290
               disease far more common when all phenotypes included)

Mechanism
---------
CEP290 (2480 aa) is a giant centrosomal protein localised at the ciliary transition zone (TZ)
matrix and basal body. It is essential for:

  1. TZ Y-link assembly: CEP290 structurally organises the Y-shaped ciliary gate that controls
     protein diffusion into and out of the cilium — LOF → TZ gate collapse → uncontrolled
     entry of non-ciliary proteins and loss of signalling cargos.
  2. Photoreceptor connecting cilium: CEP290 is highly expressed in the connecting cilia of
     rods and cones — LOF → failure of outer segment disc morphogenesis → progressive
     photoreceptor death → severe rod-cone dystrophy.
  3. Renal tubular primary cilia: expression in renal tubular cells → LOF → TZ dysfunction
     → cystogenesis, interstitial fibrosis → tubulointerstitial nephritis → ESRD.
  4. NPHP-RC module interaction: CEP290 interacts with NPHP5 (IQCB1), RPGR, and PCM1 at
     the connecting cilium; it is a hub gene linking multiple TZ ciliopathy complexes.

CEP290 — THE BROADEST ALLELE-PHENOTYPE SPECTRUM OF ANY NPHP GENE:
  • IVS26+1655A>G (c.2991+1655A>G) — hypomorphic deep-intronic cryptic exon splicing →
    retinal-only disease = LCA10 (most common LCA-causing allele worldwide); NO renal disease.
  • Truncating compound het (one null + one missense) → NPHP6/SLS6: renal + severe retinal.
  • Biallelic truncating (two null alleles) → Meckel-Gruber Syndrome Type 4 (MKS4): lethal.
  • Intermediate alleles (partial function) → Joubert Syndrome Type 5 (JBTS5): Molar Tooth Sign.
  • Specific missense alleles → Bardet-Biedl Syndrome Type 14 (BBS14): obesity + retinal.

LOF in the NPHP6/SLS6 spectrum → severe LCA-like retinal disease THEN progressive renal
disease → ESRD median ~13–15 yr; NO Molar Tooth Sign (differentiates from Joubert JBTS5).

Hallmark Features (NPHP6 vs other subtypes):
  • BROADEST allele-phenotype spectrum: IVS26 (LCA10-only) → NPHP6 → Joubert → Meckel
  • NO Molar Tooth Sign: critical DDx from Joubert Syndrome 5 (same gene!)
  • NO IVS26 allele in NPHP6: IVS26 homozygotes = LCA10 only (retinal, NO renal)
  • SEVERE retinal dystrophy: nystagmus, ERG flat/markedly reduced, early visual impairment
    — often initially diagnosed as LCA; renal workup commonly omitted at referral
  • ESRD median ~13–15 yr (similar to NPHP1 and NPHP5)
  • Kidneys: small, echogenic, corticomedullary cysts (same pattern as NPHP1/5)
  • No situs inversus; No CHF; No polydactyly (differentiates from Joubert/Meckel)
  • CEP290 is the HUB gene: WES + full 35+ gene ciliopathy panel mandatory

Key Differentials:
  NPHP5/IQCB1 (3q21.1): Similar SLS phenotype; RPGR direct binding; no Joubert/Meckel spectrum
  NPHP1 (2q13): Juvenile; mild retinal (10-15%); 290kb deletion; no severe LCA-like
  Joubert JBTS5 (CEP290): SAME GENE — Molar Tooth Sign present; different alleles
  LCA10 (CEP290-IVS26): SAME GENE — IVS26 allele; retinal only; NO renal disease
  Meckel-Gruber MKS4 (CEP290): SAME GENE — biallelic null; lethal prenatal
  BBS14 (CEP290): obesity + hexadactyly + retinal; different alleles
  Alport (COL4A3/A4/A5): haematuria; GBM splitting; hearing loss; NO retinal TZ pattern

Treatment:
  • Renal transplant = definitive for renal component (cell-autonomous; NO recurrence)
  • Retinal does NOT improve after renal transplant — separate cell-autonomous process
  • Sepofarsen (QR-110) / antisense oligonucleotide: specifically for IVS26 allele carriers
    (LCA10 / Joubert-retinal) — NOT for NPHP6 truncating alleles; ILLUMINATE trial data
  • Conservative CKD: 2–3 L fluid/day; EPO for anaemia disproportionate to GFR
  • Low-vision rehabilitation from diagnosis; braille/AT; orientation & mobility
  • Annual ophthalmology (ERG + OCT): retinal is the dominant disability
  • CEP290 CRISPR subretinal gene editing: pre-clinical 2026 (broader than sepofarsen)
  • No disease-modifying therapy for renal component approved 2026
  • Genetic counselling: full 35-gene ciliopathy panel (not just CEP290 alone)
"""

import random
import statistics

SEED = 351
_RNG = random.Random(SEED)

# ── Genetic pool — realistic CEP290 alleles causing NPHP6/SLS6 (NOT IVS26 = LCA10) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("CEP290 (12q21.32) — p.Gln1604Ter / p.Tyr2119Cys (nonsense/missense compound het; pan-ethnic SLS6)", 0.14),
    ("CEP290 (12q21.32) — del exon 23–24 / p.Arg1863Gln (large del/missense compound het; CNV + WES)", 0.12),
    ("CEP290 (12q21.32) — p.Arg151Ter / p.Tyr2119Cys (null/missense compound het; SLS6; NOT Meckel)", 0.11),
    ("CEP290 (12q21.32) — p.Gln1604Ter homozygous (consanguineous Middle Eastern; severe biallelic)", 0.09),
    ("CEP290 (12q21.32) — c.4723+2T>C / p.Tyr2119Cys (splice/missense compound het; European)", 0.09),
    ("CEP290 (12q21.32) — p.Cys998Ser / p.Gln1604Ter (missense/nonsense compound het; European)", 0.08),
    ("CEP290 (12q21.32) — del exon 1–5 / p.Arg1863Gln (homoz. del consanguineous; Turkish)", 0.07),
    ("CEP290 (12q21.32) — p.Tyr2119Cys / c.5668+1G>T (missense/splice compound het; South Asian)", 0.07),
    ("CEP290 (12q21.32) — p.Arg1863Gln homozygous (North African consanguineous; biallelic missense)", 0.06),
    ("CEP290 (12q21.32) — frameshift c.4243delC / p.Tyr2119Cys (compound het; heterogeneous)", 0.05),
    ("CEP290 (12q21.32) — p.Arg151Gln / del exon 7–9 (missense/del compound het; CNV array)", 0.05),
    ("CEP290 (12q21.32) — novel / VUS compound het (WES-confirmed NPHP6; heterogeneous)", 0.04),
    ("CEP290 (12q21.32) — IVS26+1655A>G / truncating (one IVS26; heteroz.; LCA10+renal RARE overlap)", 0.03),
]

_ETHNICITY_POOL = [
    ("European (pan-European heterogeneous)",          0.27),
    ("Middle Eastern / Arab (consanguinity enriched)", 0.25),
    ("South Asian (Indian subcontinent)",              0.17),
    ("Turkish",                                        0.09),
    ("North African (consanguinity enriched)",         0.09),
    ("East Asian",                                     0.07),
    ("African / Sub-Saharan",                          0.04),
    ("Latin American",                                 0.02),
]

_RETINAL_INVOLVEMENT = [
    ("Severe LCA-like (ERG flat; nystagmus; CF or worse vision; Dx <2yr)", 0.38),
    ("Severe rod-cone dystrophy (ERG markedly reduced; Dx 2–5yr)", 0.31),
    ("Moderate-severe rod-cone dystrophy (ERG reduced; Dx 5–10yr)", 0.17),
    ("Moderate (ERG reduced; partial vision; Dx 8–12yr)", 0.09),
    ("Mild (NPHP6 missense-enriched; ERG mildly reduced; later dx)", 0.05),
]

_OCULAR_MOTOR = [
    ("Pendular Nystagmus — sensory, from severe visual impairment", 0.53),
    ("Searching Nystagmus — neonatal onset, early severe retinal loss", 0.12),
    ("No nystagmus — milder retinal involvement", 0.22),
    ("Oculomotor apraxia (OMA) — rare in NPHP6; seen in NPHP4", 0.05),
    ("Strabismus only — nystagmus absent", 0.08),
]

_VISUAL_ACUITY = [
    ("Light perception / NLP (ERG flat; severe LCA-like)", 0.22),
    ("CF — counting fingers only (severe rod-cone loss)", 0.18),
    ("HM — hand motions (advanced rod-cone degeneration)", 0.14),
    ("6/60–6/120 (moderate-severe loss)", 0.17),
    ("6/18–6/36 (moderate loss, early or milder alleles)", 0.16),
    ("6/6–6/12 (near-normal; early phase or missense-enriched)", 0.13),
]

_KIDNEY_SIZE = [
    ("Small/echogenic (classic NPHP; < −2 SD for age)", 0.55),
    ("Normal size, increased echogenicity", 0.26),
    ("Small with visible cysts (corticomedullary pattern)", 0.15),
    ("Mildly enlarged (unusual; overlap; reassess genetics)", 0.04),
]

_CKD_STAGE = [
    ("CKD 1 (GFR ≥90; early/pre-symptomatic)", 0.10),
    ("CKD 2 (GFR 60–89; polyuria/concentrating defect)", 0.15),
    ("CKD 3a (GFR 45–59)", 0.15),
    ("CKD 3b (GFR 30–44)", 0.17),
    ("CKD 4 (GFR 15–29; pre-dialysis planning)", 0.16),
    ("CKD 5 / ESRD (GFR <15; on dialysis or post-Tx)", 0.27),
]

_RRT_STATUS = [
    ("Pre-emptive transplant — living donor (optimal outcome)", 0.19),
    ("Renal transplant — deceased donor (post-dialysis)", 0.20),
    ("Peritoneal dialysis → awaiting transplant", 0.12),
    ("Haemodialysis → awaiting transplant", 0.11),
    ("Conservative CKD management (GFR >15)", 0.28),
    ("Combined liver-kidney transplant (rare; hepatic fibrosis variant)", 0.03),
    ("Dialysis — social/access barriers to transplant", 0.07),
]

_MISDIAGNOSIS = [
    ("LCA (Leber Congenital Amaurosis) — renal workup omitted", 0.38),
    ("Joubert Syndrome — MTS wrongly assumed from retinal phenotype", 0.17),
    ("ADPKD — adult team assumed autosomal dominant", 0.13),
    ("NPHP5 (IQCB1) — clinically indistinguishable; gene panel required", 0.12),
    ("RPGR-related X-linked RP — RPGR/CEP290 interaction; pedigree missed AR", 0.08),
    ("Alport Syndrome — haematuria first; COL4A3 tested first", 0.07),
    ("No prior misdiagnosis (directly to ciliopathy genetics)", 0.05),
]


def _weighted_choice(pool, rng):
    """Pick one item from a (value, weight) pool."""
    items, weights = zip(*pool)
    cumw, r = 0.0, rng.random()
    for item, w in zip(items, weights):
        cumw += w
        if r < cumw:
            return item
    return items[-1]


def _make_patient(pid, rng):
    gene        = _weighted_choice(_GENE_POOL, rng)
    ethnicity   = _weighted_choice(_ETHNICITY_POOL, rng)
    retinal     = _weighted_choice(_RETINAL_INVOLVEMENT, rng)
    ocular_motor= _weighted_choice(_OCULAR_MOTOR, rng)
    visual_acuity = _weighted_choice(_VISUAL_ACUITY, rng)
    kidney_size = _weighted_choice(_KIDNEY_SIZE, rng)
    ckd_stage   = _weighted_choice(_CKD_STAGE, rng)
    rrt_stat    = _weighted_choice(_RRT_STATUS, rng)
    misdiagnosis= _weighted_choice(_MISDIAGNOSIS, rng)

    # Retinal dx age — CEP290/NPHP6: very early (often LCA-like); slightly earlier than NPHP5
    age_retinal_dx = round(rng.gauss(2.2, 2.0), 1)
    age_retinal_dx = max(0.1, min(12.0, age_retinal_dx))

    # Renal dx age — NPHP6 median ~13–15yr
    age_renal_dx = round(rng.gauss(11.5, 4.5), 1)
    age_renal_dx = max(2.0, min(25.0, age_renal_dx))

    # GFR current
    gfr_now = round(rng.gauss(32.0, 24.0), 1)
    gfr_now = max(4.0, min(105.0, gfr_now))

    # GFR slope (CEP290/NPHP6: ~5–8 ml/min/yr)
    gfr_slope = round(rng.gauss(-6.5, 2.5), 1)
    gfr_slope = min(-1.0, max(-15.0, gfr_slope))

    # Urine osmolality — tubular concentrating defect
    uosm = round(rng.gauss(148, 55))
    uosm = max(60, min(310, uosm))

    # Haemoglobin
    hb = round(rng.gauss(9.4, 1.8), 1)
    hb = max(5.0, min(14.0, hb))

    # Systolic BP
    sbp = int(rng.gauss(124, 14))
    sbp = max(90, min(165, sbp))

    return {
        "id":               f"NPHP6-{pid:03d}",
        "gene":             gene,
        "ethnicity":        ethnicity,
        "retinal_involvement": retinal,
        "ocular_motor":     ocular_motor,
        "visual_acuity":    visual_acuity,
        "kidney_size":      kidney_size,
        "ckd_stage":        ckd_stage,
        "rrt_or_transplant":rrt_stat,
        "prior_misdiagnosis":misdiagnosis,
        "age_retinal_dx_yr":age_retinal_dx,
        "age_renal_dx_yr":  age_renal_dx,
        "gfr_now_ml_min":   gfr_now,
        "gfr_slope_ml_min_yr": gfr_slope,
        "urine_osmolality_mosm": uosm,
        "haemoglobin_g_dl": hb,
        "systolic_bp_mmhg": sbp,
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
        bins = [(0, 2, "<2yr"), (2, 5, "2–5yr"), (5, 10, "5–10yr"),
                (10, 15, "10–15yr"), (15, 20, "15–20yr"), (20, 99, "≥20yr")]
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
        "−7 to −10 (rapid)": 0,
        "−4 to −7 (moderate)": 0,
        "−1 to −4 (slow)": 0,
    }
    for p in cohort:
        s = p.get("gfr_slope_ml_min_yr", -5)
        if s < -10:
            tiers["< −10 ml/min/yr (very rapid)"] += 1
        elif s < -7:
            tiers["−7 to −10 (rapid)"] += 1
        elif s < -4:
            tiers["−4 to −7 (moderate)"] += 1
        else:
            tiers["−1 to −4 (slow)"] += 1
    return tiers


def _uosm_tiers(cohort):
    bins = {"<100 (very low; severe TIN)": 0, "100–150": 0, "150–200": 0,
            "200–250": 0, "250–300": 0, ">300 (near normal)": 0}
    for p in cohort:
        u = p.get("urine_osmolality_mosm", 150)
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
    gfrs  = [p["gfr_now_ml_min"] for p in c]
    hbs   = [p["haemoglobin_g_dl"] for p in c]
    ret_ages = [p["age_retinal_dx_yr"] for p in c]
    renal_ages = [p["age_renal_dx_yr"] for p in c]
    sbps  = [p["systolic_bp_mmhg"] for p in c]
    uosms = [p["urine_osmolality_mosm"] for p in c]

    esrd_tx_n = sum(1 for p in c if any(kw in p["rrt_or_transplant"] for kw in
                   ["transplant", "dialysis"]))
    pct_esrd_tx = round(esrd_tx_n / len(c) * 100)

    nys_n = sum(1 for p in c if "Nystagmus" in p["ocular_motor"])
    pct_nystagmus = round(nys_n / len(c) * 100)

    severe_retinal_n = sum(1 for p in c if
        any(kw in p["retinal_involvement"] for kw in ["LCA-like", "markedly reduced"]))
    pct_severe_retinal = round(severe_retinal_n / len(c) * 100)

    misdiag_n = sum(1 for p in c if "LCA" in p["prior_misdiagnosis"])
    pct_misdiag_lca = round(misdiag_n / len(c) * 100)

    ivs26_n = sum(1 for p in c if "IVS26" in p["gene"])
    pct_ivs26 = round(ivs26_n / len(c) * 100)

    return {
        "cohort_n": 40,
        "gene": "CEP290",
        "chromosome": "12q21.32",
        "omim_gene": "610142",
        "omim_disease": "610189",
        "median_gfr": round(statistics.median(gfrs), 1),
        "mean_gfr": round(statistics.mean(gfrs), 1),
        "median_hb": round(statistics.median(hbs), 1),
        "mean_hb": round(statistics.mean(hbs), 1),
        "median_age_retinal_dx": round(statistics.median(ret_ages), 1),
        "mean_age_retinal_dx": round(statistics.mean(ret_ages), 1),
        "median_age_renal_dx": round(statistics.median(renal_ages), 1),
        "mean_age_renal_dx": round(statistics.mean(renal_ages), 1),
        "mean_sbp": round(statistics.mean(sbps), 1),
        "median_uosm": round(statistics.median(uosms)),
        "pct_esrd_or_transplant": pct_esrd_tx,
        "pct_nystagmus": pct_nystagmus,
        "pct_severe_retinal": pct_severe_retinal,
        "pct_misdiagnosed_as_lca": pct_misdiag_lca,
        "pct_ivs26_carrier": pct_ivs26,
        "patients": c[:8],
    }


def get_breakdown():
    c = _COHORT
    gene_dist_raw = {}
    for p in c:
        g = p["gene"]
        short = g.split("—")[-1].strip().split("(")[0].strip()[:55]
        gene_dist_raw[short] = gene_dist_raw.get(short, 0) + 1

    return {
        "gene_distribution": gene_dist_raw,
        "ethnicity": _tally(c, "ethnicity", _ETHNICITY_POOL),
        "retinal_involvement": _tally(c, "retinal_involvement", _RETINAL_INVOLVEMENT),
        "ocular_motor_nystagmus": _tally(c, "ocular_motor", _OCULAR_MOTOR),
        "visual_acuity_distribution": _tally(c, "visual_acuity", _VISUAL_ACUITY),
        "kidney_size_distribution": _tally(c, "kidney_size", _KIDNEY_SIZE),
        "ckd_stage_current": _tally(c, "ckd_stage", _CKD_STAGE),
        "rrt_transplant_status": _tally(c, "rrt_or_transplant", _RRT_STATUS),
        "prior_misdiagnosis": _tally(c, "prior_misdiagnosis", _MISDIAGNOSIS),
        "age_at_retinal_dx_tiers": _age_tiers(c, "age_retinal_dx_yr",
            [(0, 1, "<1yr"), (1, 2, "1–2yr"), (2, 5, "2–5yr"),
             (5, 8, "5–8yr"), (8, 12, "8–12yr"), (12, 99, "≥12yr")]),
        "age_at_renal_dx_tiers": _age_tiers(c, "age_renal_dx_yr",
            [(0, 5, "<5yr"), (5, 10, "5–10yr"), (10, 15, "10–15yr"),
             (15, 18, "15–18yr"), (18, 22, "18–22yr"), (22, 99, "≥22yr")]),
        "urine_osmolality_tiers": _uosm_tiers(c),
        "gfr_slope_tiers": _gfr_slope_tiers(c),
    }


def get_definitions():
    return {
        "disease": "Nephronophthisis Type 6 / Senior-Løken Syndrome 6 (NPHP6/SLS6)",
        "omim_gene": "CEP290 *610142",
        "omim_disease": "#610189 (SLS6/NPHP6)",
        "chromosome": "12q21.32",
        "inheritance": "Autosomal Recessive — biallelic LOF (truncating + missense compound het; NOT IVS26 homozygous which = LCA10)",
        "prevalence": "~1/100,000–1/300,000 (NPHP6/SLS6 subtype); CEP290 pan-phenotype far more common",
        "mechanism": (
            "CEP290 (2480 aa) is a giant TZ matrix protein essential for Y-link assembly, "
            "ciliary gate function, and photoreceptor connecting cilium integrity. Biallelic LOF "
            "(truncating + missense) → TZ gate collapse → photoreceptor outer segment failure "
            "(severe retinal dystrophy) AND renal tubular TZ dysfunction (NPHP pattern TIN → ESRD). "
            "Allele severity determines phenotype: IVS26 hypomorphic = LCA10 (retinal only); "
            "intermediate = Joubert (MTS); null + missense = NPHP6/SLS6; biallelic null = Meckel."
        ),
        "key_clinical_features": {
            "Retinal_phenotype": "Severe LCA-like rod-cone dystrophy; ERG flat/markedly reduced; nystagmus common (~65%); visual impairment from infancy",
            "Renal_phenotype": "NPHP pattern — corticomedullary cysts, tubulointerstitial nephritis, ESRD median 13–15yr; kidneys small/echogenic",
            "No_Molar_Tooth_Sign": "Critical DDx from Joubert Syndrome 5 (SAME gene CEP290 but different alleles); absence of MTS rules out JBTS5",
            "No_situs_inversus": "CEP290 not expressed in nodal cilia — no laterality defect (unlike NPHP2/INVS)",
            "No_CHF": "No hepatic fibrosis (CEP290 absent from biliary epithelium in most NPHP6 patients)",
            "Nystagmus_65pct": "Predominantly pendular/searching — sensory nystagmus from severe early visual impairment",
            "IVS26_NOT_cause": "IVS26+1655A>G homozygous = LCA10 ONLY (retinal-only, no renal) — NPHP6 is caused by truncating ± missense alleles",
            "Polyuria_concentrating_defect": "Urine osmolality <200 mosm/kg — first symptom; tubular dysfunction before glomerular",
            "Anaemia_disproportionate": "Anaemia greater than expected for GFR — EPO synthesis failure from interstitial cell loss",
            "ESRD_timeline": "Median ~13–15yr; range 8–25yr depending on allele severity (missense-enriched later)",
        },
        "diagnostic_criteria": {
            "Genetic_testing": "WES mandatory; full 35-gene ciliopathy panel; CNV array (CEP290 large exon deletions common)",
            "Biallelic_CEP290_LOF": "Two pathogenic variants confirmed on BOTH alleles — phase critical; if one allele = IVS26, check renal phenotype carefully",
            "IVS26_distinction": "IVS26+1655A>G allele: if homozygous → LCA10 (retinal only); if compound het with null → may have renal overlap",
            "Renal_biopsy": "Tubulointerstitial nephritis + corticomedullary cysts; tubular basement membrane thickening; NO immune deposits",
            "Brain_MRI_mandatory": "Rule out Molar Tooth Sign (= Joubert JBTS5, same gene); absence of MTS confirms NPHP6 not JBTS5",
            "ERG_mandatory": "Reduced/flat rod-cone dystrophy; ERG confirms retinal severity; differentiates from ADPKD/FSGS",
            "Ophthalmology_first": "Retinal disease often precedes renal diagnosis; ophthalmology referral typically initiates genetics workup",
        },
        "genetic_architecture": {
            "Gene_size": "CEP290: 54 exons; 7.4 kb mRNA; 2480 aa protein; 290 kDa",
            "TZ_scaffold_domains": "N-terminal coiled-coil (TZ matrix); central disordered region; C-terminal ciliary targeting sequence (CTS)",
            "Allele_spectrum": "Broadest allele-phenotype spectrum of any NPHP gene: IVS26 (LCA10) → NPHP6 → JBTS5 → MKS4",
            "Mutational_heterogeneity": "No single dominant founder; heterogeneous biallelic variants across ethnicities; CNV 15–20% of alleles",
            "CEP290_interactome": "NPHP5 (IQCB1), PCM1, RPGR, BBS4, KIF3A — CEP290 is a ciliopathy hub gene",
            "IVS26_mechanism": "Deep intronic c.2991+1655A>G creates cryptic exon 26a (128 nt) → PTC → partial protein loss (~50%) → retinal-only LCA10",
            "Compound_het_rule": "NPHP6 typically = one truncating/null + one missense; biallelic null → Meckel (lethal); biallelic IVS26 → LCA10",
        },
        "founder_variants": [
            "IVS26+1655A>G (c.2991+1655A>G) — global LCA10 allele; NOT founder for NPHP6 but critical DDx",
            "p.Arg151Ter — found in NPHP6/Meckel spectrum; pan-ethnic, no single ethnicity dominant",
            "p.Gln1604Ter — truncating; enriched compound het NPHP6; pan-ethnic",
            "p.Tyr2119Cys — missense; NPHP6/SLS6 allele; pan-ethnic (European, South Asian)",
            "p.Arg1863Gln — missense; North African/Middle Eastern enriched in NPHP6",
            "p.Cys998Ser — missense; intermediate Joubert/NPHP6 spectrum",
            "Large exon 23–24 deletion — CNV; detectable only by array CGH / WGS",
        ],
        "cep290_allele_spectrum": {
            "IVS26+1655A>G_homozygous": "LCA10 (retinal ONLY — no renal disease); most common LCA allele worldwide; sepofarsen/QR-110 therapeutic target",
            "IVS26+1655A>G_compound_het_null": "Rare LCA10 + mild renal overlap; depends on null allele severity",
            "Truncating_+_missense": "NPHP6 / SLS6 (renal + severe retinal; NO Molar Tooth Sign) — the NPHP6 genotype",
            "Intermediate_alleles": "Joubert Syndrome 5 (JBTS5; Molar Tooth Sign + NPHP + retinal); partial protein function",
            "Biallelic_null_truncating": "Meckel-Gruber Syndrome Type 4 (MKS4; prenatal lethal; occipital encephalocele + cystic kidneys + polydactyly)",
            "BBS14_alleles": "Specific missense (p.Pro988Arg, etc.) → Bardet-Biedl Syndrome 14 (obesity + hexadactyly + retinal)",
        },
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)": "Juvenile ESRD 13yr; mild retinal 10–15%; 290kb deletion 80%; NO LCA-like; NO broad allele spectrum",
            "NPHP2 (INVS / 9q31.1)": "Infantile ESRD 3yr; situs inversus 35%; enlarged kidneys (mimics ARPKD); NO retinal",
            "NPHP3 (NPHP3 / 3q22.1)": "Adolescent ESRD 19yr; CHF 45%; situs 15%; NO retinal; male infertility",
            "NPHP4 (NPHP4 / 1p36)": "Juvenile 17–20yr; SLS4 15–20%; ocular motor apraxia KEY feature; NO CHF; 2nd most common SLS",
            "NPHP5 (IQCB1 / 3q21.1)": "Most common SLS; retinal >> renal; RPGR direct binding; NO Joubert/Meckel spectrum",
            "NPHP6 (CEP290 / 12q21.32) ★": "THIS — broadest allele spectrum (LCA10/JBTS5/NPHP6/MKS4 same gene); NO Molar Tooth Sign in NPHP6 subtype",
        },
        "ddx_table": {
            "LCA10 (CEP290-IVS26)": "SAME GENE — IVS26 allele only; retinal-only; NO renal; sepofarsen therapeutic; NO NPHP6 criteria",
            "Joubert JBTS5 (CEP290)": "SAME GENE — Molar Tooth Sign present; brain MRI mandatory DDx; intermediate alleles",
            "Meckel-Gruber MKS4 (CEP290)": "SAME GENE — lethal prenatal; biallelic null; encephalocele + cysts + polydactyly",
            "BBS14 (CEP290)": "SAME GENE — obesity + polydactyly; specific BBS14 missense alleles",
            "NPHP5 / SLS5 (IQCB1)": "Similar SLS phenotype; RPGR interaction; NO Joubert/Meckel/LCA10 spectrum; 3q21.1",
            "NPHP1 (NPHP1)": "Milder retinal (10–15%); 290kb deletion; NO LCA-like severity",
            "Alport Syndrome (COL4A3/A4/A5)": "Haematuria; GBM splitting TEM; hearing loss; NO TZ retinal dystrophy",
            "ADPKD (PKD1/PKD2)": "Autosomal dominant; large cysts; NO retinal; NO TZ ciliopathy; adult presentation",
        },
        "treatment": {
            "Renal_transplant": "Definitive CURATIVE for renal component; cell-autonomous TZ defect; NO recurrence in graft; excellent outcomes",
            "Retinal_NOT_improved": "Retinal disease cell-autonomous — does NOT improve after renal transplant; separate management",
            "Sepofarsen_QR-110": "ASO for IVS26+1655A>G allele ONLY (LCA10/Joubert-retinal); NOT indicated for NPHP6 truncating alleles",
            "CEP290_CRISPR": "Subretinal CRISPR editing (IVS26 correction, Editas EDIT-101) — pre-clinical data also for broader CEP290 alleles",
            "Conservative_CKD": "2–3 L fluid/day to replace urinary losses; EPO for anaemia; avoid nephrotoxins (NSAIDs, aminoglycosides)",
            "Low_vision_rehabilitation": "Braille, AT, screen readers, orientation & mobility from childhood — retinal is dominant disability",
            "Annual_ophthalmology": "ERG + OCT from diagnosis annually — track photoreceptor layer thickness; detect treatable complications",
            "Genetic_counselling": "Full 35-gene ciliopathy panel (not CEP290 alone); WES + CNV; sibling 25% risk; ALL LCA patients need NPHP renal workup",
            "No_disease_modifying_Rx_2026": "No approved therapy for renal CEP290 LOF 2026; mTOR inhibitor trials (pre-clinical); AAV-CEP290 gene augmentation (pre-clinical)",
        },
        "prognosis": (
            "ESRD median 13–15 yr (range 8–25 yr; allele-severity dependent). "
            "Renal transplant outcomes EXCELLENT — no recurrence (cell-autonomous). "
            "Retinal disease is progressive and permanent — dominant disability in adult life. "
            "Low-vision rehabilitation from childhood improves independence. "
            "Sepofarsen is IVS26-allele specific; NPHP6 patients (non-IVS26) await CRISPR/gene therapy trials. "
            "No situs, CHF, or polydactyly complications. "
            "Brain MRI must be done to rule out Joubert (same gene) — alters management and counselling."
        ),
        "cohort_note": (
            f"Synthetic cohort · n=40 · seed={SEED} · allele frequencies from published SLS6/NPHP6 case series "
            "(Sayer 2006 NPHP6, Baala 2007, Coppieters 2010, Perrault 2007, Stone 2011 sepofarsen). "
            "IVS26 carriers (LCA10 overlap) represented at realistic 3% of NPHP6 genotypes. "
            "NOT human-subject data — illustrative only."
        ),
    }
