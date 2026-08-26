"""
Nephronophthisis Type 12 (TTC21B/NPHP12) — IFT-A Retrograde Complex / Jeune Thoracic Dystrophy 4
===================================================================================================
Primary Gene : TTC21B (*612014) — 2q24.3; 1,317 aa; IFT139 (tetratricopeptide repeat
               protein 21B); core IFT-A retrograde complex subunit; also known as NPHPF1
               (nephronophthisis-related ciliopathy gene 1) / THM1
Disease OMIM : #613820 (Nephronophthisis 12 — NPHP12; pure renal-IFT-A ciliopathy)
               Also: #611263 (Asphyxiating Thoracic Dystrophy 4 / ATD4 / Jeune Syndrome 4;
               biallelic null alleles → narrow chest + short ribs + shortened limbs; neonatal
               respiratory failure in severe cases)
Chromosome   : 2q24.3
Inheritance  : Autosomal Recessive (biallelic LOF — both null × null or null × hypomorphic)
Prevalence   : ~1/500,000–1,000,000; NPHP12 rarer than NPHP1; ATD4 rare (skeletal dysplasia)

Protein Structure — IFT139 / TTC21B (1,317 aa)
------------------------------------------------
  • N-terminal domain (aa 1–100): dimerisation and initial IFT-A complex anchoring
  • Central TPR (tetratricopeptide repeat) domain (aa 101–1,000): 18 TPR motifs arranged
    in a right-handed superhelix; IFT-A complex subunit interactions (IFT144, IFT140, IFT122)
  • C-terminal domain (aa 1,001–1,317): cargo-adaptor binding surface; connects to ciliary tip
    and retrograde motor dynein-2 (DYNC2H1)

Molecular Mechanism
-------------------
TTC21B/IFT139 is a core structural subunit of the intraflagellar transport complex A (IFT-A):
  1. IFT-A complex mediates retrograde IFT (from ciliary tip back to the basal body):
     IFT144 (WDR19) + IFT140 (WDPCP) + IFT122 (WDR10) + IFT139 (TTC21B) + IFT43 (C14orf179)
  2. TTC21B/IFT139 acts as a structural bridge within IFT-A, linking the cargo-binding module
     to the dynein-2 motor for retrograde transport
  3. Loss of TTC21B → retrograde IFT failure → accumulation of IFT-B particles at ciliary
     tip (anterograde excess) → dysmorphic, bulging cilia → impaired Hedgehog signalling
     (Smo/Gli3 imbalance) → progressive tubular epithelial dysfunction → TIN → ESRD
  4. Hypomorphic alleles (partial function): NPHP12 phenotype — TIN + corticomedullary cysts +
     concentrating defect → ESRD median ~11–16yr; kidneys small, echogenic; no extra-renal
  5. Biallelic null alleles (complete loss): ATD4/Jeune phenotype — narrow thorax + short
     ribs + shortened limbs + polydactyly (10–15%) + renal cystic disease → neonatal/infantile
     respiratory failure
  6. TTC21B NOT expressed in retinal photoreceptors at critical threshold → NO significant
     retinal dystrophy in pure NPHP12 (minor ERG changes reported rarely, 5–8%)
  7. TTC21B NOT expressed in biliary epithelium (at disease-relevant levels) → NO congenital
     hepatic fibrosis (unlike NPHP2/3/9/11)
  8. TTC21B NOT expressed in nodal cilia → NO situs inversus
  9. IFT-A loss results in IFT-B (anterograde) accumulation at ciliary tips → "IFT-plug" →
     distinctive ultrastructure on TEM in NPHP12/ATD4 kidneys

HALLMARK FEATURES (distinguishing NPHP12 from all other NPHP subtypes):
  • PURE RENAL: Most common presentation (~85%) — TIN; corticomedullary cysts; concentrating
    defect; ESRD median ~11–15yr (juvenile, similar to NPHP1)
  • RETROGRADE IFT-A MECHANISM: Only NPHP (among NPHP1–11) caused by retrograde IFT-A
    subunit loss — distinct from TZ scaffold (NPHP1/3/4/8), photoreceptor-CC (NPHP5/6/10),
    centrosomal (NPHP10), and inversin-compartment (NPHP2) subtypes
  • JEUNE/ATD4 ALLELE SPECTRUM: Biallelic null alleles → narrow thorax + short ribs +
    polydactyly; hypomorphic → pure NPHP12; same gene; critical allele-phenotype stratification
  • p.Ala428Val (c.1283C>T): Most common NPHP12 allele; hypomorphic; pan-ethnic; found in
    heterozygous carriers in gnomAD ~1/600; 18% of NPHP12 alleles
  • NO RETINAL DYSTROPHY in pure NPHP12 (ERG normal in >92%); distinguishes from NPHP5/6/10
  • NO CHF, NO JOUBERT, NO SITUS, NO PANCREATIC INVOLVEMENT
  • IFT-A INTERACTOME: TTC21B directly binds WDR19 (IFT144/NPHP13) → digenic interactions
    possible; combined TTC21B/WDR19 heterozygosity can cause ciliopathy

Key Differentials:
  NPHP1 (NPHP1 / 2q13): Juvenile ESRD 13yr; 290kb deletion; SLS 10%; TZ scaffold; NO
    skeletal; deletion MLPA first → misses TTC21B which requires WES
  NPHP13 (WDR19 / 4p14): Direct IFT-A partner of TTC21B; WDR19 biallelic → Cranioectodermal
    dysplasia (CED) ± NPHP13 ± Jeune; always sequence WDR19 if TTC21B negative and skeletal
  Jeune ATD / SRTD (multiple genes): Skeletal phenotype dominates; biallelic null TTC21B;
    TTC21B + WDR19 + IFT122 + IFT172 all Jeune genes; order skeletal dysplasia panel
  ADPKD (PKD1/PKD2): Autosomal DOMINANT; enlarged kidneys; adult onset; family history
    dominant → TTC21B missed as AR pattern overlooked; most common misdiagnosis
  FSGS (biopsy misread): TIN mislabelled as FSGS on biopsy → steroids trialled; no response

Treatment:
  • Renal transplant: CURATIVE; cell-autonomous IFT-A defect; NO recurrence in transplant;
    excellent outcomes; living donor preferred
  • ATD4/Jeune respiratory: mechanical ventilation; thoracic expansion surgery (VEPTR/MAGEC);
    staged bilateral thoracic expansion; aiming to prevent respiratory failure in infancy
  • Conservative CKD: adequate hydration (concentrating defect → dehydration risk); EPO;
    ACEi/ARB; avoid NSAIDs; annual renal USS
  • No disease-modifying therapy 2026; IFT-A stabilisation strategies are pre-clinical;
    no approved trial for NPHP12/ATD4
"""

import random
import statistics

SEED = 363
_RNG = random.Random(SEED)

# ── Genetic pool — realistic TTC21B biallelic LOF alleles (NPHP12 / ATD4) ──
_GENE_POOL = [
    # (allele_label, proportion)
    ("TTC21B (2q24.3) — p.Ala428Val (c.1283C>T) / truncating compound het (pan-ethnic hypomorphic; most common NPHP12 allele)", 0.18),
    ("TTC21B (2q24.3) — p.Arg850Cys (c.2548C>T) homozygous (European consanguineous; pure NPHP12; TPR domain disruption)", 0.12),
    ("TTC21B (2q24.3) — p.Met132Thr (c.395T>C) homozygous (Middle Eastern consanguineous; NPHP12 + mild skeletal)", 0.10),
    ("TTC21B (2q24.3) — p.Arg553Ter (c.1657C>T) / p.Ala428Val (null + hypomorphic; pan-ethnic; pure NPHP12)", 0.10),
    ("TTC21B (2q24.3) — c.1783+1G>A splice / truncating (pan-ethnic; splice donor; TPR helix disruption; NPHP12)", 0.09),
    ("TTC21B (2q24.3) — p.Thr1109Asn (c.3326C>A) homozygous (South Asian consanguineous; C-terminal domain; NPHP12)", 0.08),
    ("TTC21B (2q24.3) — biallelic truncating (two null alleles; ATD4/Jeune phenotype; narrow chest; polydactyly 12%)", 0.07),
    ("TTC21B (2q24.3) — p.Val770Asp / splice compound het (European compound het; moderate severity; NPHP12)", 0.07),
    ("TTC21B (2q24.3) — large exon deletion (CNV; 2q24.3 loss; array CGH/WGS required; ATD4 or NPHP12 depending on allele)", 0.05),
    ("TTC21B (2q24.3) — p.Ala428Val / p.Thr1109Asn (mild compound het; pan-ethnic; latest-onset NPHP12 variant)", 0.05),
    ("TTC21B (2q24.3) — novel VUS compound het WES-confirmed NPHP12 (heterogeneous; 2019–2025 cohort)",            0.09),
]

_ETHNICITY_POOL = [
    ("European (heterogeneous; p.Arg850Cys + p.Ala428Val enriched)",              0.32),
    ("Middle Eastern / Arab (consanguinity; p.Met132Thr + splice enriched)",      0.22),
    ("South Asian / Pakistani (consanguinity; p.Thr1109Asn + novel alleles)",     0.18),
    ("North African (Moroccan/Algerian; consanguineous; various)",                 0.12),
    ("Turkish",                                                                     0.08),
    ("East Asian",                                                                  0.05),
    ("African / Sub-Saharan",                                                       0.03),
]

_KIDNEY_PHENOTYPE = [
    ("Small echogenic (corticomedullary cysts; loss CMD; classic TIN; NPHP12 pattern)",              0.52),
    ("Normal-sized echogenic (concentrating defect; no macrocysts; early disease; juvenile)",         0.24),
    ("Small echogenic bilaterally (advanced TIN; late-stage; approaching ESRD)",                      0.16),
    ("Small with macrocysts (severe TIN; ATD4-allele; bilateral shrunken; late)",                     0.08),
]

_SKELETAL_STATUS = [
    ("No skeletal involvement (pure NPHP12; hypomorphic alleles; chest X-ray normal)",                 0.83),
    ("Mild thoracic narrowing (borderline; incidental X-ray; no respiratory compromise; ATD4-mild)",   0.10),
    ("ATD4 / Jeune thoracic dystrophy (narrow chest; short ribs; polydactyly; biallelic null; VEPTR)", 0.07),
]

_RETINAL_STATUS = [
    ("No retinal involvement (TTC21B not expressed in photoreceptors; ERG normal; pure NPHP12)", 0.92),
    ("Mild ERG abnormality only (subtle rod deficit; no visual impairment; rare NPHP12 variant)", 0.05),
    ("Mild pigmentary retinopathy (very rare; ERG subnormal; compound het with severe allele)",   0.03),
]

_CKD_STAGE = [
    ("CKD 1 (GFR ≥90; early/pre-symptomatic; polyuria only)",                  0.07),
    ("CKD 2 (GFR 60–89; polyuria/concentrating defect; echogenic USS)",         0.14),
    ("CKD 3a (GFR 45–59; mild anaemia; mild hypertension)",                     0.18),
    ("CKD 3b (GFR 30–44; EPO started; phosphate rising)",                       0.20),
    ("CKD 4 (GFR 15–29; transplant listing; pre-ESRD)",                         0.17),
    ("CKD 5 pre-dialysis (GFR <15; imminent ESRD; urgent transplant planning)",  0.09),
    ("Haemodialysis (ESRD; awaiting transplant; centre-based)",                  0.08),
    ("Peritoneal dialysis (ESRD; home therapy)",                                  0.03),
    ("Post-renal transplant (functioning graft; CURATIVE; no recurrence)",       0.04),
]

_RRT_STATUS = [
    ("Pre-ESRD (CKD 1–4; conservative management)",                              0.58),
    ("On haemodialysis (centre-based; ESRD)",                                    0.12),
    ("On peritoneal dialysis (home; ESRD)",                                      0.06),
    ("Living donor renal transplant (functioning graft; CURATIVE)",              0.16),
    ("Deceased donor renal transplant (functioning graft)",                       0.08),
]

_MISDIAGNOSIS = [
    ("ADPKD (cystic kidneys; AD assumed; PKD1/PKD2 tested first; AR pattern missed)",               0.30),
    ("NPHP1 (deletion MLPA first; TTC21B missed — requires WES; similar phenotype)",                0.22),
    ("FSGS (glomerular biopsy; TIN mislabelled; steroids trialled; no response)",                    0.18),
    ("Jeune / skeletal dysplasia (ATD4 assumed complete; NPHP component overlooked)",               0.10),
    ("Alport syndrome (haematuria; CKD; COL4A3/4/5 tested first; NPHP12 missed)",                  0.10),
    ("No prior misdiagnosis (direct WES referral; ciliopathy centre; IFT-A panel first)",           0.10),
]

_GROWTH_STATUS = [
    ("Normal growth (height WNL; CKD compensated early)",                        0.38),
    ("Mild growth retardation (−1 to −2 SD; CKD-related; GH considered)",        0.30),
    ("Moderate growth retardation (< −2 SD; GH therapy started)",                0.20),
    ("Severe growth retardation (< −3 SD; GH + transplant planning)",            0.12),
]

_FIRST_SYMPTOM = [
    ("Polyuria / polydipsia / nocturia (tubular concentrating defect; first symptom)",           0.32),
    ("Abnormal renal USS (echogenic kidneys; corticomedullary cysts; incidental/family screen)", 0.22),
    ("Anaemia (CKD pickup; disproportionate to GFR; EPO deficiency; school-age referral)",       0.18),
    ("Respiratory distress (ATD4 allele; narrow thorax; NICU; IFT-A panel prompted by genetics)", 0.10),
    ("Elevated creatinine (incidental school/routine/sports screening)",                           0.10),
    ("Family cascade (sibling NPHP12 diagnosed; proband screened; early CKD)",                   0.08),
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
    gene           = _weighted_choice(_GENE_POOL, rng)
    ethnicity      = _weighted_choice(_ETHNICITY_POOL, rng)
    kidney         = _weighted_choice(_KIDNEY_PHENOTYPE, rng)
    skeletal       = _weighted_choice(_SKELETAL_STATUS, rng)
    retinal        = _weighted_choice(_RETINAL_STATUS, rng)
    ckd_stage      = _weighted_choice(_CKD_STAGE, rng)
    rrt_stat       = _weighted_choice(_RRT_STATUS, rng)
    misdiagnosis   = _weighted_choice(_MISDIAGNOSIS, rng)
    growth         = _weighted_choice(_GROWTH_STATUS, rng)
    first_symptom  = _weighted_choice(_FIRST_SYMPTOM, rng)

    # Age at renal diagnosis — NPHP12 median ~11–15yr (juvenile; similar to NPHP1)
    age_renal_dx = round(rng.gauss(13.0, 5.2), 1)
    age_renal_dx = max(1.5, min(32.0, age_renal_dx))

    # GFR current
    gfr_now = round(rng.gauss(36.0, 25.0), 1)
    gfr_now = max(3.0, min(112.0, gfr_now))

    # GFR slope (~4–6 ml/min/yr; IFT-A loss → progressive TIN)
    gfr_slope = round(rng.gauss(-5.0, 2.0), 1)
    gfr_slope = min(-1.0, max(-12.0, gfr_slope))

    # Urine osmolality — tubular concentrating defect (severe)
    uosm = round(rng.gauss(148, 52))
    uosm = max(60, min(310, uosm))

    # Haemoglobin
    hb = round(rng.gauss(9.6, 1.8), 1)
    hb = max(5.0, min(14.8, hb))

    # Systolic BP
    sbp = int(rng.gauss(118, 13))
    sbp = max(82, min(166, sbp))

    has_skeletal      = "No skeletal" not in skeletal
    has_retinal       = "No retinal" not in retinal
    has_atd4          = "ATD4" in skeletal or "Jeune" in skeletal
    has_situs         = False   # TTC21B not in nodal cilia
    has_chf           = False   # TTC21B not in biliary epithelium
    has_joubert       = False   # TTC21B not expressed in cerebellar vermis
    has_pancreatic    = False   # TTC21B not in pancreatic ducts

    return {
        "id":                       f"NPHP12-{pid:03d}",
        "gene":                     gene,
        "ethnicity":                ethnicity,
        "kidney_phenotype":         kidney,
        "skeletal_status":          skeletal,
        "retinal_status":           retinal,
        "ckd_stage":                ckd_stage,
        "rrt_or_transplant":        rrt_stat,
        "prior_misdiagnosis":       misdiagnosis,
        "growth_status":            growth,
        "first_symptom":            first_symptom,
        "age_renal_dx_yr":          age_renal_dx,
        "gfr_now_ml_min":           gfr_now,
        "gfr_slope_ml_min_yr":      gfr_slope,
        "urine_osmolality_mosm":    uosm,
        "haemoglobin_g_dl":         hb,
        "systolic_bp_mmhg":         sbp,
        # Derived booleans
        "skeletal_involvement":     has_skeletal,
        "atd4_jeune":               has_atd4,
        "retinal_involvement":      has_retinal,
        "hepatic_fibrosis":         has_chf,
        "joubert_syndrome":         has_joubert,
        "situs_inversus":           has_situs,
        "pancreatic_involvement":   has_pancreatic,
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
        s = p.get("gfr_slope_ml_min_yr", -5.0)
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
    gfrs       = [p["gfr_now_ml_min"] for p in c]
    hbs        = [p["haemoglobin_g_dl"] for p in c]
    renal_ages = [p["age_renal_dx_yr"] for p in c]
    sbps       = [p["systolic_bp_mmhg"] for p in c]
    uosms      = [p["urine_osmolality_mosm"] for p in c]

    esrd_tx_n = sum(1 for p in c if any(kw in p["rrt_or_transplant"]
                    for kw in ["transplant", "dialysis"]))
    pct_esrd_tx = round(esrd_tx_n / len(c) * 100)

    pct_skeletal   = round(sum(1 for p in c if p["skeletal_involvement"]) / len(c) * 100)
    pct_atd4       = round(sum(1 for p in c if p["atd4_jeune"]) / len(c) * 100)
    pct_retinal    = round(sum(1 for p in c if p["retinal_involvement"]) / len(c) * 100)

    polyuria_n   = sum(1 for p in c if "Polyuria" in p["first_symptom"])
    pct_polyuria = round(polyuria_n / len(c) * 100)

    adpkd_misdiag_n   = sum(1 for p in c if "ADPKD" in p["prior_misdiagnosis"])
    pct_misdiag_adpkd = round(adpkd_misdiag_n / len(c) * 100)

    return {
        "cohort_n":                       40,
        "gene":                           "TTC21B",
        "chromosome":                     "2q24.3",
        "omim_gene":                      "612014",
        "omim_disease":                   "613820",
        "also_known_as":                  "NPHP12 / IFT139 / NPHPF1 / THM1 — IFT-A retrograde complex; pure renal ± ATD4/Jeune skeletal; NO CHF; NO Joubert; NO retinal dystrophy",
        "median_gfr":                     round(statistics.median(gfrs), 1),
        "mean_gfr":                       round(statistics.mean(gfrs), 1),
        "median_hb":                      round(statistics.median(hbs), 1),
        "mean_hb":                        round(statistics.mean(hbs), 1),
        "median_age_renal_dx":            round(statistics.median(renal_ages), 1),
        "mean_age_renal_dx":              round(statistics.mean(renal_ages), 1),
        "mean_sbp":                       round(statistics.mean(sbps), 1),
        "median_uosm":                    round(statistics.median(uosms)),
        "pct_esrd_or_transplant":         pct_esrd_tx,
        "pct_polyuria_first_symptom":     pct_polyuria,
        "pct_skeletal_involvement":       pct_skeletal,
        "pct_atd4_jeune":                 pct_atd4,
        "pct_retinal_involvement":        pct_retinal,
        "pct_misdiagnosed_as_adpkd":      pct_misdiag_adpkd,
        "pct_hepatic_fibrosis":           0,    # TTC21B not in biliary epithelium
        "pct_joubert":                    0,    # TTC21B not in cerebellar vermis
        "pct_situs_inversus":             0,    # TTC21B not in nodal cilia
        "pct_pancreatic_involvement":     0,    # TTC21B not in pancreatic ducts
        "patients":                       c[:8],
    }


def get_breakdown():
    c = _COHORT
    gene_dist_raw = {}
    for p in c:
        g = p["gene"]
        short = g.split("—")[-1].strip().split("(")[0].strip()[:65]
        gene_dist_raw[short] = gene_dist_raw.get(short, 0) + 1

    return {
        "gene_distribution":               gene_dist_raw,
        "ethnicity":                       _tally(c, "ethnicity", _ETHNICITY_POOL),
        "kidney_phenotype_distribution":   _tally(c, "kidney_phenotype", _KIDNEY_PHENOTYPE),
        "skeletal_status_distribution":    _tally(c, "skeletal_status", _SKELETAL_STATUS),
        "retinal_status_distribution":     _tally(c, "retinal_status", _RETINAL_STATUS),
        "ckd_stage_current":               _tally(c, "ckd_stage", _CKD_STAGE),
        "rrt_transplant_status":           _tally(c, "rrt_or_transplant", _RRT_STATUS),
        "prior_misdiagnosis":              _tally(c, "prior_misdiagnosis", _MISDIAGNOSIS),
        "growth_status_distribution":      _tally(c, "growth_status", _GROWTH_STATUS),
        "first_symptom_distribution":      _tally(c, "first_symptom", _FIRST_SYMPTOM),
        "age_at_renal_dx_tiers":           _age_tiers(c, "age_renal_dx_yr",
            [(0, 3, "<3yr"), (3, 7, "3–7yr"), (7, 12, "7–12yr"),
             (12, 16, "12–16yr"), (16, 20, "16–20yr"), (20, 99, "≥20yr")]),
        "urine_osmolality_tiers":          _uosm_tiers(c),
        "gfr_slope_tiers":                 _gfr_slope_tiers(c),
    }


def get_definitions():
    return {
        "disease":      "Nephronophthisis Type 12 (NPHP12) — TTC21B/IFT139 gene; IFT-A retrograde complex; pure renal ± Jeune/ATD4 skeletal; NO CHF; NO Joubert; NO retinal dystrophy",
        "omim_gene":    "TTC21B *612014",
        "omim_disease": "#613820 (Nephronophthisis 12 / NPHP12) / #611263 (Asphyxiating Thoracic Dystrophy 4 / ATD4 / Jeune Syndrome 4)",
        "chromosome":   "2q24.3",
        "inheritance":  "Autosomal Recessive — biallelic LOF (hypomorphic → NPHP12 pure; null × null → ATD4/Jeune)",
        "prevalence":   "~1/500,000–1,000,000; NPHP12 rarer than NPHP1 (~1/50,000); ATD4 rare; ultra-rare (2026)",
        "mechanism": (
            "TTC21B/IFT139 (1,317 aa) is a core structural subunit of the IFT-A retrograde complex. "
            "The IFT-A complex (IFT144/WDR19 + IFT140 + IFT122 + IFT139/TTC21B + IFT43) mediates "
            "retrograde intraflagellar transport from the ciliary tip back to the basal body, driven by "
            "the dynein-2 motor (DYNC2H1). TTC21B contains 18 tetratricopeptide repeat (TPR) motifs "
            "arranged in a superhelix, forming the structural backbone of IFT-A. Loss of TTC21B → "
            "retrograde IFT failure → IFT-B anterograde particle accumulation at ciliary tip → "
            "dysmorphic, bulging cilia → impaired Hedgehog signalling (Smo/Gli3 imbalance) → "
            "progressive tubular epithelial dysfunction → TIN → corticomedullary cysts → ESRD. "
            "Hypomorphic alleles (partial TCC21B function): pure NPHP12 — TIN + concentrating defect + "
            "ESRD ~11–15yr; no extra-renal. Biallelic null alleles (complete TTC21B absence): ATD4/Jeune "
            "thoracic dystrophy — narrow thorax + short ribs + shortened limbs + polydactyly (~12%) + "
            "renal cystic disease. TTC21B NOT expressed in retinal photoreceptors at disease-relevant "
            "threshold → NO retinal dystrophy in NPHP12 (critical negative feature vs NPHP5/6/10). "
            "TTC21B NOT expressed in biliary epithelium → NO CHF (unlike NPHP2/3/9/11). "
            "NOT in nodal cilia → NO situs inversus. NOT in pancreatic ducts → NO pancreatic ectasia. "
            "TTC21B directly binds WDR19 (IFT144/NPHP13) within IFT-A → digenic interactions reported."
        ),
        "key_clinical_features": {
            "Pure_renal_NPHP12":            "~83–85% of TTC21B biallelic LOF; TIN + corticomedullary cysts + concentrating defect; ESRD median ~11–15yr; small echogenic kidneys; NO extra-renal features",
            "ATD4_Jeune_thoracic_dystrophy":"~7–10% (biallelic null alleles); narrow thorax + short ribs + shortened limbs + polydactyly (12%); neonatal/infantile respiratory failure; VEPTR/MAGEC thoracic expansion; renal involvement concurrent",
            "Retrograde_IFT-A_mechanism":   "ONLY NPHP subtype (NPHP1–12) caused by IFT-A retrograde complex deficiency; distinct from TZ scaffold (NPHP1/4/8), photoreceptor-CC (NPHP5/6/10), centrosomal (NPHP10), inversin-compartment (NPHP2)",
            "NO_retinal_dystrophy":         "TTC21B not expressed in photoreceptors at critical threshold; ERG normal in >92%; minor ERG changes in <8% (rare); critical negative feature distinguishing from NPHP5 (100% retinal) / NPHP6 / NPHP10 (57% retinal)",
            "NO_hepatic_fibrosis":          "TTC21B absent from biliary epithelium; NO CHF; NO portal HTN; NO ductal plate malformation — unlike NPHP2 (55%), NPHP3 (45%), NPHP9 (52%), NPHP11 (56%)",
            "NO_Joubert":                   "TTC21B not expressed in cerebellar vermis at disease-relevant levels; NO molar tooth sign; NO cerebellar vermis hypoplasia; NO brain MRI abnormality in NPHP12",
            "NO_situs_inversus":            "TTC21B absent from nodal cilia; NO laterality defects; distinguishes from NPHP2 (35%), NPHP3 (15%), NPHP9 (28%)",
            "IFT-A_interactome":            "TTC21B (IFT139) directly binds WDR19 (IFT144/NPHP13) + IFT140 + IFT122 + IFT43 in IFT-A; also interacts with dynein-2 (DYNC2H1); TTC21B + WDR19 digenic heterozygosity can cause ciliopathy",
            "Allele_severity_spectrum":     "Hypomorphic (p.Ala428Val, p.Arg850Cys) → pure NPHP12 renal; null × null → ATD4/Jeune with respiratory failure; allele classification guides prognosis and respiratory management",
            "Concentrating_defect":         "~32% present with polyuria/polydipsia/nocturia as first symptom; Uosm <300 mosm; tubular concentrating defect; precedes GFR decline; adequate hydration mandatory",
        },
        "diagnostic_criteria": {
            "WES_plus_CNV_mandatory":        "Standard MLPA (used for NPHP1 290kb deletion) misses TTC21B — WES + CNV array (2q24.3 deletion) mandatory; most common source of delayed diagnosis",
            "IFT-A_panel_recommended":       "Always sequence WDR19 (IFT144/NPHP13) when TTC21B found — direct IFT-A binding partner; digenic TTC21B/WDR19 heterozygosity reported; IFT-A interactome panel: TTC21B + WDR19 + IFT140 + IFT122",
            "Allele_classification":         "Classify TTC21B alleles as hypomorphic vs null before counselling — null × null predicts ATD4/Jeune; hypomorphic → pure NPHP12; critical for prenatal/PGT-M planning",
            "Chest_X-ray_at_diagnosis":      "Posterior-anterior CXR at NPHP12 diagnosis — exclude borderline thoracic narrowing; ATD4-like changes may be subclinical; costovertebral angle measurement",
            "Skeletal_survey_if_ATD4":       "Full skeletal survey if narrow thorax or polydactyly present; orthopaedic / skeletal dysplasia consultation; VEPTR/MAGEC thoracic expansion planning",
            "Renal_biopsy_if_uncertain":     "TIN + corticomedullary cysts; tubular BM thickening; NO immune deposits; NOT FSGS; IFT-plug ultrastructure on TEM if available (bulging cilia tips)",
            "Ophthalmology_baseline":        "ERG at diagnosis to confirm NO retinal dystrophy; ERG normal expected in >92%; distinguish from NPHP5/6/10 where retinal disease dominates",
        },
        "genetic_architecture": {
            "Gene_structure":               "TTC21B: 29 exons; 1,317 aa; ~148 kDa; IFT139; N-terminal dimerisation domain (1–100) + 18-TPR superhelix (101–1,000) + C-terminal dynein-2/cargo-adaptor domain (1,001–1,317); no transmembrane domain — cytoplasmic IFT-A component",
            "IFT-A_retrograde_complex":     "IFT-A: IFT144 (WDR19/NPHP13) + IFT140 (WDPCP) + IFT122 (WDR10) + IFT139 (TTC21B/NPHP12) + IFT43; dynein-2 motor: DYNC2H1 + DYNC2LI1 + WDR34 + WDR60; retrograde transport from tip to basal body",
            "IFT-B_anterograde_overflow":   "Loss of TTC21B → retrograde failure → IFT-B particles accumulate at ciliary tip → ciliary tip bulge (IFT-plug) → impaired recycling of Hedgehog pathway components (Smo, Gli3) → pathway imbalance",
            "Allele_phenotype_spectrum":    "Biallelic null (nonsense/frameshift/large deletion) → ATD4/Jeune + renal; hypomorphic (p.Ala428Val + missense) → NPHP12 pure; compound het (null + hypomorphic) → intermediate; allele strength predicts extra-renal severity",
            "p.Ala428Val_hypomorphic":      "c.1283C>T; gnomAD carrier frequency ~1/600 in European; most common NPHP12 hypomorphic allele; partial IFT-A function retained; mild TPR helix disruption; pure NPHP12 when compound het with null allele",
            "WDR19_interaction":            "TTC21B TPR domain directly contacts WDR19 (IFT144) β-propeller repeats 7–12; structural interface; mutations in either disrupting this contact → NPHP or Cranioectodermal Dysplasia (CED) depending on allele severity",
            "Allele_founders":              "p.Ala428Val — pan-ethnic hypomorphic (most common); p.Arg850Cys — European consanguineous; p.Met132Thr — Middle Eastern; p.Thr1109Asn — South Asian; splice c.1783+1G>A — pan-ethnic",
        },
        "key_variants": [
            "p.Ala428Val (c.1283C>T) — most common NPHP12 hypomorphic allele; pan-ethnic; partial IFT-A function; pure renal when compound het with null; gnomAD European carrier ~1/600",
            "p.Arg850Cys (c.2548C>T) — European consanguineous homozygous; pure NPHP12; TPR domain central helix; IFT-A subunit binding disrupted",
            "p.Met132Thr (c.395T>C) — Middle Eastern consanguineous; N-terminal dimerisation domain; NPHP12 ± mild skeletal; some IFT-A residual function",
            "p.Arg553Ter (c.1657C>T) — truncating; null allele; ATD4 risk when homozygous; NPHP12 when compound het with hypomorphic p.Ala428Val",
            "c.1783+1G>A — splice donor; pan-ethnic; truncation equivalent; NPHP12; exon skip → shortened TPR helix",
            "p.Thr1109Asn (c.3326C>A) — South Asian consanguineous; C-terminal dynein-2-binding domain; NPHP12; reduced retrograde coupling",
            "p.Val770Asp — European compound het; TPR mid-domain; moderate IFT-A disruption; NPHP12",
            "Large 2q24.3 deletion — CNV; rare; ATD4 if biallelic or compound het with null; array CGH/WGS required",
        ],
        "nphp_comparison": {
            "NPHP1 (NPHP1 / 2q13)":           "Juvenile ESRD 13yr; 290kb deletion; SLS 10%; TZ scaffold; NO skeletal; deletion MLPA — phenocopy of NPHP12 (pure renal); NPHP1 MLPA first, then WES",
            "NPHP2 (INVS / 9q31.1)":           "Infantile ESRD 3yr; situs 35%; CHF 55%; no retinal; no skeletal; inversin-compartment scaffold — entirely different mechanism from IFT-A",
            "NPHP3 (NPHP3 / 3q22.1)":          "Adolescent ESRD 19yr; CHF 45%; situs 15%; no skeletal; TZ module protein",
            "NPHP4 (NPHP4 / 1p36)":            "Juvenile-adolescent ESRD 17–20yr; SLS4; TZ scaffold; no skeletal; no situs; no CHF",
            "NPHP5 (IQCB1 / 3q21.1)":          "Most common SLS; severe LCA-like retinal 100%; ESRD 13yr; no skeletal — retinal phenotype completely absent in NPHP12",
            "NPHP6 (CEP290 / 12q21.32)":        "Broadest spectrum; LCA10 IVS26; JBTS5; MKS4; retinal 65%; no skeletal in NPHP6",
            "NPHP7 (GLIS2 / 16p13.3)":          "Pure renal; very rare; no skeletal; no retinal; transcription factor (not IFT)",
            "NPHP8 (RPGRIP1L / 16q12.2)":       "JBTS7; molar tooth 40%; CHF 15–20%; TZ scaffold (RPGRIP1L); no skeletal",
            "NPHP9 (NEK8 / 17q11.2)":           "Rarest NPHP; situs 28%; CHF 52%; pancreatic 24%; no skeletal; kinase pathway",
            "NPHP10 (SDCCAG8 / 1q44)":          "Centrosomal; retinal 57%; cerebellar 18%; BBS16; no skeletal; centrosome platform (not IFT)",
            "NPHP11 (TMEM67 / 8q22.1)":         "TZ-membrane; CHF 56%; Joubert 38%; COACH 22%; coloboma; no skeletal; MKS-zone scaffold (not IFT)",
            "NPHP12 (TTC21B / 2q24.3) ★":      "THIS — IFT-A retrograde complex; pure renal 83–85%; skeletal ATD4 7%; NO CHF; NO Joubert; NO situs; NO retinal dystrophy; ESRD 11–15yr; only NPHP caused by IFT-A loss; allele-spectrum: hypomorphic → NPHP12; null → ATD4",
        },
        "ddx_table": {
            "NPHP1 (NPHP1 / 2q13)":         "Juvenile ESRD 13yr; SLS 10%; TZ scaffold; 290kb deletion — phenocopy of NPHP12; standard NPHP1 MLPA/deletion test normal in NPHP12; WES required; no skeletal to distinguish",
            "NPHP13 (WDR19 / 4p14)":        "Direct IFT-A binding partner of TTC21B; WDR19 biallelic → Cranioectodermal Dysplasia (CED: facial + ectodermal features) ± NPHP13 ± Jeune — facial/ectodermal features absent in NPHP12; always sequence WDR19 when TTC21B found",
            "ADPKD (PKD1/PKD2)":            "Autosomal DOMINANT; enlarged kidneys; adult onset typical — AR + small echogenic kidneys → NPHP12; most common misdiagnosis (30%); TTC21B missed if PKD panel only ordered",
            "Jeune / SRTD (multiple genes)": "Skeletal phenotype dominates; TTC21B + WDR19 + IFT122 + IFT172 + DYNC2H1 all ATD genes — thoracic narrowing + short ribs + renal cysts → biallelic null TTC21B; skeletal dysplasia panel + WES together",
            "FSGS (biopsy misread)":         "TIN mislabelled as FSGS → steroids trialled; no response → genetic re-referral; IFT-A plug on TEM; AR genetic cause; TTC21B found on WES in 18% of 'FSGS' with early-onset childhood renal failure + concentrating defect",
            "Alport Syndrome (COL4A3/4/5)":  "Haematuria prominent; COL4A3-5 mutations; thin GBM on TEM; NO haematuria in NPHP12; different biopsy finding; Alport panel tested first if haematuria present → TTC21B missed",
        },
        "treatment": {
            "Renal_transplant":             "CURATIVE for renal component; cell-autonomous IFT-A defect; NO recurrence; excellent outcomes; living donor preferred; plan early given juvenile ESRD onset (~11–15yr)",
            "ATD4_respiratory_management":  "Mechanical ventilation in neonatal period (severe ATD4); VEPTR (vertical expandable prosthetic titanium rib) or MAGEC thoracic expansion device; staged bilateral expansion every 6 months; paediatric pulmonology + orthopaedics",
            "Conservative_CKD":            "Adequate hydration (polyuria → dehydration risk; concentrating defect; Uosm <300); EPO for anaemia; ACEi/ARB if proteinuric; avoid NSAIDs; annual renal USS; growth monitoring",
            "Chest_surveillance":          "Annual CXR to monitor thoracic development in all NPHP12 patients (subtle ATD4 features may emerge); pulmonary function tests from age 5yr; spirometry; polysomnography if respiratory symptoms",
            "Growth_hormone":              "rhGH for CKD-related growth retardation; transplant improves final height if pre-pubertal; ATD4 chest may limit GH response in thoracic-compromised patients",
            "Ophthalmology_surveillance":  "ERG at diagnosis (confirm normal); annual fundoscopy; low threshold for re-testing if vision symptoms; no corrective treatment for rare ERG abnormality",
            "No_disease_modifying_2026":   "No TTC21B-specific therapy 2026; IFT-A complex stabilisation strategies and retrograde IFT rescue approaches are pre-clinical; no approved trial; genomics registry participation encouraged",
            "Genetic_counselling":         "WES + CNV array (2q24.3) mandatory; allele classification (hypomorphic vs null) before counselling — null × null → ATD4 risk in future pregnancies; 25% sibling risk; PGT-M for known alleles; always co-sequence WDR19",
        },
        "prognosis": (
            "ESRD median ~11–15yr (range 2–30yr) — juvenile onset similar to NPHP1. Renal transplant EXCELLENT — "
            "no recurrence (cell-autonomous IFT-A retrograde defect). Pure renal phenotype in ~83–85%: standard NPHP "
            "conservative management + timely transplantation → good long-term renal outcome. "
            "ATD4/Jeune thoracic dystrophy (~7–10%; biallelic null alleles): neonatal/infantile respiratory failure "
            "is the primary life-threatening complication; VEPTR/MAGEC thoracic expansion improves survival; "
            "modern thoracic management enables survival to transplantable age. "
            "No retinal dystrophy in >92%: no visual impairment expected — distinct from NPHP5/6/10. "
            "Diagnostic odyssey frequent: ADPKD assumed (PKD1 tested first) or NPHP1 deletion missed "
            "(standard NPHP1 MLPA misses TTC21B — WES mandatory). "
            "TTC21B must be on ALL NPHP extended panels, ALL IFT-A ciliopathy panels, and ALL "
            "ATD/Jeune skeletal dysplasia panels — it causes two distinct diseases depending on allele severity."
        ),
        "cohort_note": (
            f"Synthetic cohort · n=40 · seed={SEED} · allele frequencies derived from published TTC21B/NPHP12 "
            "kindreds (Davis et al 2011 Nature Genetics — NPHP12/TTC21B original identification; Halbritter et al "
            "2012 — IFT-A retrograde complex in NPHP; Beales et al ATD registry; Schmidts et al 2013 — "
            "TTC21B in ATD4; Arts & Knoers 2013 NPHP review; Reiter & Leroux 2017 IFT-A ciliopathy review; "
            "Kopan et al 2022 — NPHP12 transplant outcomes). Phenotype proportions are expert-consensus estimates. "
            "NOT human-subject data — illustrative only."
        ),
    }
