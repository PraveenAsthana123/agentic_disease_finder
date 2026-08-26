"""
Primary Ciliary Dyskinesia (PCD — Motile Ciliopathy; Kartagener Syndrome when + Situs Inversus)
================================================================================================
Primary Gene : DNAH5 (*603335) — 5p15.2; ~15–28% of PCD
Other Genes  : DNAI1 (*604366, 9p21.2) · DNAI2 (*605483, 17q25.1) ·
               CCDC39 (*613798, 3q26.33) · CCDC40 (*613799, 17q25.3) ·
               DNAH11 (*603339, 7p15.3) · DNAH9 (*603330, 17p12) ·
               CCDC103 (*614677, 17q21.31) · RSPH4A (*612647, 6q22.1) ·
               RSPH9 (*612648, 6p21.1) · ARMC4 (*615408, 10p12.1) ·
               HYDIN (*610812, 16q22.2) · LRRC6 (*614930, 8q24.22) ·
               CCNO (*607752, 5q11.2) · MCIDAS (*614086, 5q11.2) ·
               >50 genes total — multigenetic motile ciliopathy
OMIM Dis.    : #244400 (PCD1, DNAI1) · #608644 (PCD2, DNAH5) ·
               #613798 (PCD7, CCDC39) · #613799 (PCD8, CCDC40)
Chromosome   : Multi-locus; DNAH5 at 5p15.2 (most common)
Inheritance  : Autosomal Recessive (biallelic LOF); X-linked rare (PIH1D3, OFD1)
Prevalence   : ~1/10,000–1/20,000 live births (1/15,000 average); underdiagnosed

Mechanism
---------
Primary Ciliary Dyskinesia is a motile ciliopathy: structural or functional defect of the
9+2 axoneme machinery in motile cilia and flagella. Unlike primary (non-motile) ciliopathies
(Joubert, BBS, MKS), PCD specifically affects MOTILE cilia:

Axoneme Defect Classes (ultrastructure on TEM):
  Class 1 — Outer Dynein Arm (ODA) defect: DNAH5, DNAI1, DNAI2, ARMC4 (~70% of PCD)
    ODA provides the main motor force; absent → cilia beat slowly/statically
  Class 2 — Inner Dynein Arm (IDA) defect: CCDC39, CCDC40 (~8–10%)
    IDA controls beat symmetry; absent → hyperkinetic, disoriented beat
  Class 3 — ODA + IDA combined defect: LRRC6, DNAAF1-5 (~5%)
    Pre-assembly factor mutations → both arm classes absent
  Class 4 — Radial Spoke Head (RSH) defect: RSPH4A, RSPH9 (~5%)
    Normal ultrastructure on TEM; abnormal beat (circular rather than waveform)
  Class 5 — Central Pair (CP) defect: HYDIN (~3%)
    C2b projection absent; abnormal waveform with circular cilia tip path
  Class 6 — ODA + IDA + MT transposition: CCDC39/40 severe (~5%)
    Microtubule transposition (loss of central pair + ODA + IDA)
  Class 7 — Reduced generation of motile cilia (RGMC): CCNO, MCIDAS (~3%)
    Normal axoneme but far fewer cilia; pseudostratified epithelium nearly devoid of cilia

Affected Organ Systems (all lined with motile cilia / flagella):
  1. Airways (bronchi, bronchioles, trachea): mucociliary clearance failure → infection
  2. Paranasal sinuses + middle ear: mucus stasis → sinusitis, glue ear, OME → CHL
  3. Left-right axis determination (embryonic node cilia): ~50% get situs inversus
     → Kartagener Syndrome = PCD + situs inversus (totalis or inversus)
  4. Sperm flagella: immotile / dyskinetic sperm → male infertility (azoospermia-like)
  5. Fallopian tube cilia: impaired ovum transport → ectopic pregnancy / subfertility
  6. Brain ependymal cilia: hydrocephalus in ~5% of cases
  7. Retina: not affected (non-motile primary cilia are intact)

Neonatal Respiratory Distress:
  ~80% of PCD term neonates have unexplained neonatal respiratory distress (NRD)
  — upper lobe consolidation or hyperinflation on CXR
  — NOT meconium aspiration, NOT TTN, NOT RDS
  Key differentiator: NRD in term neonate + full-term + normal antenatal scan + situs

Diagnostic Pathway:
  Step 1: nasal NO (nNO) — low (<77 nL/min adults; <77 nl/min threshold; exception: RSPH/CP subtypes may be LOW-NORMAL)
  Step 2: HSVM (high-speed video microscopy) — abnormal beat pattern (static, hyperkinetic, circular)
  Step 3: TEM (transmission electron microscopy) — ultrastructure (ODA absent = hallmark)
  Step 4: Gene panel (>50 genes) — confirms biallelic PCD variants; identifies subtype
  Step 5: Immunofluorescence (IF staining) for DNAH5, DNAI1, DNAI2, CCDC39 etc.
  Note: 30% of PCD has normal TEM (RSPH, HYDIN, DNAH11, CCNO subtypes) — TEM alone insufficient

Kartagener Triad (classic, ~50% of PCD):
  1. Situs inversus totalis (mirror-image arrangement of all organs)
  2. Chronic sinusitis (pansinusitis, rhinosinusitis)
  3. Bronchiectasis (cylindrical or varicose; bilateral basal predominant)

Treatment Targets:
  - Airway clearance: chest physio (ACBT, PEP devices) LIFELONG daily
  - Prophylactic antibiotics: azithromycin low-dose in severe bronchiectasis
  - Nasal irrigation: hypertonic saline, nasal rinses
  - Hearing: grommets for glue ear; hearing aids
  - Male fertility: ICSI (sperm can fertilise via ICSI despite immotility)
  - Emerging: CFTR-modulator analogue concept for motile ciliopathies (pre-clinical)
"""

import random
import statistics

SEED = 339
_RNG  = random.Random(SEED)

# ── Realistic founder / common variants ────────────────────────────────────────
_GENE_POOL = [
    # (gene_label, proportion)
    ("DNAH5 (5p15.2) — c.7915C>T p.Arg2639Ter European founder",    0.17),
    ("DNAH5 (5p15.2) — c.10815+3A>T splice-site European",           0.10),
    ("DNAI1 (9p21.2) — IVS1+2_3insT splice Middle-Eastern founder",  0.09),
    ("DNAI1 (9p21.2) — c.1543C>T p.Arg515Ter European",              0.06),
    ("DNAI2 (17q25.1) — c.787G>T p.Glu263Ter",                       0.05),
    ("CCDC39 (3q26.33) — c.357+1G>A splice biallelic",               0.07),
    ("CCDC40 (17q25.3) — c.248delC p.Leu83fs Iberian founder",       0.06),
    ("DNAH11 (7p15.3) — large intragenic deletion; situs inversus",  0.06),
    ("RSPH4A (6q22.1) — c.921+3A>T splice East Asian / consang.",    0.05),
    ("RSPH9 (6p21.1) — c.1-2A>G promoter Ashkenazi/Middle East",     0.05),
    ("ARMC4 (10p12.1) — c.2626C>T p.Arg876Ter pan-ethnic",           0.04),
    ("CCDC103 (17q21.31) — p.His154Pro South Asian founder",         0.04),
    ("LRRC6 (8q24.22) — c.223G>A p.Gly75Arg North African",          0.04),
    ("HYDIN (16q22.2) — large deletion; normal TEM",                  0.03),
    ("CCNO (5q11.2) — c.456_459del RGMC subtype",                    0.03),
    ("MCIDAS (5q11.2) — c.C to T exon3; RGMC bronchiectasis",        0.02),
    ("Unknown / Panel negative (WES pending)",                        0.04),
]

_ETHNICITIES = [
    ("North European (UK/Scandinavian)",   0.32),
    ("South Asian (Indian/Pakistani)",     0.18),
    ("Middle Eastern / North African",     0.17),
    ("East Asian",                         0.08),
    ("South European (Iberian/Italian)",   0.10),
    ("Ashkenazi Jewish",                   0.06),
    ("Sub-Saharan African",                0.05),
    ("Admixed / Multiethnic",              0.04),
]

_TEM_CLASSES = [
    ("ODA defect (outer dynein arm absent)",          0.52),
    ("IDA + MT transposition (CCDC39/40)",            0.12),
    ("ODA + IDA combined absent",                     0.09),
    ("Radial spoke head defect (RSPH4A/9 — normal TEM ultrastructure)", 0.08),
    ("Central pair defect (HYDIN — normal TEM ultrastructure)", 0.04),
    ("Reduced generation of motile cilia (RGMC)",     0.04),
    ("Normal ultrastructure (DNAH11/RSP/HYDIN subtype)", 0.11),
]

_BEAT_PATTERNS = [
    ("Immotile / static",               0.45),
    ("Slow / reduced frequency",        0.20),
    ("Dyskinetic / hyperkinetic",       0.15),
    ("Circular / rotational (RSPH)",    0.10),
    ("Near-normal pattern (DNAH11)",    0.06),
    ("Absent (RGMC — too few cilia)",   0.04),
]

_PULM_COMPLICATIONS = [
    ("Bilateral cylindrical bronchiectasis (lower lobe predominant)", 0.38),
    ("Bilateral varicose bronchiectasis (multi-lobe)",                0.19),
    ("Mild air-trapping only (early / paediatric)",                   0.14),
    ("Unilateral bronchiectasis (right middle lobe collapse)",        0.10),
    ("Lobar consolidation / atelectasis (recurrent)",                 0.12),
    ("Extensive cystic bronchiectasis (severe adult)",                0.07),
]

_SINUS_FINDINGS = [
    ("Pansinusitis (all four sinus groups opacified)",                0.50),
    ("Maxillary + ethmoid sinusitis",                                 0.25),
    ("Frontal hypoplasia + pansinusitis",                             0.12),
    ("Mild rhinitis only (paediatric early)",                         0.08),
    ("Normal sinuses (early childhood)",                              0.05),
]

_AGE_AT_DX_YR = (0.5, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 10, 12, 14, 18, 22, 28, 35)

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
        pid = f"PCD-{SEED}-{i+1:03d}"

        gene = _pick(_GENE_POOL, rng)
        eth  = _pick(_ETHNICITIES, rng)
        consang = rng.random() < 0.22

        age_dx = rng.choice(_AGE_AT_DX_YR)
        age_now = round(age_dx + rng.uniform(0, 20), 1)

        # Situs inversus: ~50% PCD; DNAH11 very high; RSPH/HYDIN almost none
        if "DNAH11" in gene:
            situs = rng.random() < 0.85
        elif "RSPH" in gene or "HYDIN" in gene or "RGMC" in gene or "CCNO" in gene or "MCIDAS" in gene:
            situs = rng.random() < 0.05
        else:
            situs = rng.random() < 0.50

        kartagener = situs  # Kartagener = PCD + situs inversus by definition

        # Neonatal respiratory distress (NRD): ~80% term PCD infants
        nrd = rng.random() < 0.80

        # Bronchiectasis presence
        has_bronchiectasis = rng.random() < (0.90 if age_now > 10 else 0.40)

        pulm = _pick(_PULM_COMPLICATIONS, rng) if has_bronchiectasis else "No bronchiectasis yet (paediatric / early)"

        sinus = _pick(_SINUS_FINDINGS, rng)
        tem   = _pick(_TEM_CLASSES, rng)
        beat  = _pick(_BEAT_PATTERNS, rng)

        # Male fertility
        male   = rng.random() < 0.50
        infertile_male = male and rng.random() < 0.90
        icsi_used = infertile_male and rng.random() < 0.55

        # Hearing / OME
        hearing_loss = rng.random() < 0.50
        grommets = hearing_loss and rng.random() < 0.70

        # Hydrocephalus (rare)
        hydrocephalus = rng.random() < 0.05

        # Lung function FEV1 % predicted
        if age_now < 10:
            fev1_pct = round(rng.uniform(75, 100), 1)
        elif age_now < 20:
            fev1_pct = round(rng.uniform(55, 90), 1)
        else:
            fev1_pct = round(rng.uniform(30, 80), 1)

        # nNO (nasal NO) nL/min — PCD hallmark
        if "RSPH" in gene or "HYDIN" in gene or "DNAH11" in gene:
            nno = round(rng.uniform(60, 120), 1)   # may be near normal in RSP/HYDIN
        else:
            nno = round(rng.uniform(8, 65), 1)     # very low (normal >77)

        # Airway clearance adherence
        acbt_adherent = rng.random() < 0.65

        # Annual exacerbation rate
        exacerbation_rate = rng.randint(1, 6) if has_bronchiectasis else rng.randint(0, 3)

        # Prior misdiagnosis
        prior_dx_pool = [
            "Asthma (most common misdiagnosis)",
            "Recurrent viral URTI",
            "Cystic fibrosis (excluded by sweat test / CFTR genetics)",
            "ABPA (allergic bronchopulmonary aspergillosis)",
            "Immunodeficiency (CVID / IgA deficiency excluded)",
            "Chronic adenotonsillitis",
            "Idiopathic bronchiectasis",
            "No prior misdiagnosis (early PCD service referral)",
        ]
        prior_dx = rng.choice(prior_dx_pool)

        cohort.append({
            "id":                  pid,
            "gene":                gene,
            "ethnicity":           eth,
            "consanguineous":      consang,
            "age_at_diagnosis_yr": age_dx,
            "age_now_yr":          age_now,
            "situs_inversus":      situs,
            "kartagener_syndrome": kartagener,
            "neonatal_rd":         nrd,
            "tem_class":           tem,
            "beat_pattern":        beat,
            "pulmonary_finding":   pulm,
            "has_bronchiectasis":  has_bronchiectasis,
            "sinus_finding":       sinus,
            "hearing_loss_ome":    hearing_loss,
            "grommets_inserted":   grommets,
            "male":                male,
            "male_infertility":    infertile_male,
            "icsi_used":           icsi_used,
            "hydrocephalus":       hydrocephalus,
            "fev1_pct_pred":       fev1_pct,
            "nasal_no_nl_min":     nno,
            "acbt_adherent":       acbt_adherent,
            "exacerbations_per_yr": exacerbation_rate,
            "prior_misdiagnosis":  prior_dx,
        })

    return cohort


# ── Public API ─────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    cohort = _build_cohort()
    n = len(cohort)

    ages_dx   = [p["age_at_diagnosis_yr"] for p in cohort]
    fev1s     = [p["fev1_pct_pred"] for p in cohort]
    nnos      = [p["nasal_no_nl_min"] for p in cohort]

    pct_situs        = round(sum(1 for p in cohort if p["situs_inversus"]) / n * 100, 1)
    pct_kartagener   = round(sum(1 for p in cohort if p["kartagener_syndrome"]) / n * 100, 1)
    pct_nrd          = round(sum(1 for p in cohort if p["neonatal_rd"]) / n * 100, 1)
    pct_bronch       = round(sum(1 for p in cohort if p["has_bronchiectasis"]) / n * 100, 1)
    pct_hearing      = round(sum(1 for p in cohort if p["hearing_loss_ome"]) / n * 100, 1)
    pct_male_inf     = round(sum(1 for p in cohort if p["male_infertility"]) / n * 100, 1)
    pct_consang      = round(sum(1 for p in cohort if p["consanguineous"]) / n * 100, 1)
    pct_hydro        = round(sum(1 for p in cohort if p["hydrocephalus"]) / n * 100, 1)
    pct_misdiag      = round(sum(1 for p in cohort if "Asthma" in p["prior_misdiagnosis"]) / n * 100, 1)
    pct_dnah5        = round(sum(1 for p in cohort if "DNAH5" in p["gene"]) / n * 100, 1)

    kpis = {
        "cohort_n":              n,
        "cohort_type":           "Paediatric + adult PCD registry (retrospective)",
        "gene":                  "DNAH5 (5p15.2) most common (~15–28%); >50 PCD genes",
        "syndrome":              "Primary Ciliary Dyskinesia (PCD; Kartagener when + situs)",
        "chromosome":            "5p15.2 (DNAH5); multi-locus motile ciliopathy",
        "inheritance":           "Autosomal Recessive (biallelic LOF); X-linked rare",
        "prevalence":            "~1/10,000–1/20,000 live births",
        "median_age_dx_yr":      round(statistics.median(ages_dx), 1),
        "mean_fev1_pct_pred":    round(statistics.mean(fev1s), 1),
        "median_nno_nl_min":     round(statistics.median(nnos), 1),
        "pct_situs_inversus":    pct_situs,
        "pct_kartagener":        pct_kartagener,
        "pct_neonatal_rd":       pct_nrd,
        "pct_bronchiectasis":    pct_bronch,
        "pct_hearing_loss_ome":  pct_hearing,
        "pct_male_infertility":  pct_male_inf,
        "pct_consanguineous":    pct_consang,
        "pct_hydrocephalus":     pct_hydro,
        "pct_asthma_misdiagnosis": pct_misdiag,
        "pct_dnah5":             pct_dnah5,
    }

    key_facts = [
        "PCD is a MOTILE ciliopathy — it affects the 9+2 axoneme dynein arm machinery, NOT primary (non-motile) cilia",
        f"~{pct_situs}% of this cohort have situs inversus; Kartagener Syndrome = PCD + situs inversus (~50% of all PCD)",
        "DNAH5 (5p15.2) is the most common gene (~15–28%); outer dynein arm (ODA) defect; TEM: ODA absent",
        f"Neonatal respiratory distress affects ~{pct_nrd}% — unexplained NRD in term neonate must trigger PCD work-up",
        "nasal NO (nNO) <77 nL/min is the screening threshold; RSPH/HYDIN/DNAH11 subtypes may have near-normal nNO",
        "TEM shows normal ultrastructure in ~30% of PCD (RSPH, HYDIN, DNAH11) — gene panel is mandatory, TEM alone insufficient",
        f"{pct_bronch}% have established bronchiectasis; daily airway clearance (ACBT, PEP) is the cornerstone treatment",
        f"Male infertility: {pct_male_inf}% of males affected (immotile sperm flagella); ICSI achieves successful paternity",
        f"Asthma misdiagnosis: {pct_misdiag}% — PCD median diagnosis delay ~5–6 years; suspect PCD in 'difficult asthma' + daily wet cough",
        f"{pct_hearing}% hearing loss / OME (glue ear) — grommets + hearing aids; secretory otitis media from Eustachian tube dysfunction",
        "PCD is NOT the same as CF: sweat chloride normal, CFTR wild-type; distinguishes from bronchiectasis + sinusitis DDx",
    ]

    alerts = {
        "nno_screening":       "nasal NO <77 nL/min in adults / children >5 yr — immediate referral to specialist PCD diagnostic centre",
        "neonatal_nrd":        "Unexplained NRD in term neonate + situs inversus → PCD until proven otherwise; nNO + HSVM + TEM pathway",
        "male_fertility":      "All males with PCD should receive fertility counselling early — ICSI is effective despite immotile sperm",
        "gene_panel_mandatory": "TEM alone misses ~30% of PCD (normal ultrastructure subtypes); full >50-gene panel is mandatory for diagnosis",
        "bronchiectasis_review": "Annual HRCT chest from age 10 (or earlier if symptomatic); bronchiectasis is progressive — preventable with early airway clearance",
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

    # Gene distribution
    gene_dist: dict = {}
    for p in cohort:
        key = p["gene"].split("(")[0].strip().split(" —")[0].strip()
        gene_dist[key] = gene_dist.get(key, 0) + 1

    # TEM class distribution
    tem_dist: dict = {}
    for p in cohort:
        key = p["tem_class"].split("(")[0].strip()[:60]
        tem_dist[key] = tem_dist.get(key, 0) + 1

    # Beat pattern
    beat_dist: dict = {}
    for p in cohort:
        key = p["beat_pattern"].split("(")[0].strip()[:55]
        beat_dist[key] = beat_dist.get(key, 0) + 1

    # Pulmonary finding
    pulm_dist: dict = {}
    for p in cohort:
        key = p["pulmonary_finding"].split("(")[0].strip()[:60]
        pulm_dist[key] = pulm_dist.get(key, 0) + 1

    # Sinus finding
    sinus_dist: dict = {}
    for p in cohort:
        key = p["sinus_finding"].split("(")[0].strip()[:55]
        sinus_dist[key] = sinus_dist.get(key, 0) + 1

    # Prior misdiagnosis
    misdx_dist: dict = {}
    for p in cohort:
        key = p["prior_misdiagnosis"].split("(")[0].strip()[:60]
        misdx_dist[key] = misdx_dist.get(key, 0) + 1

    # Ethnicity
    eth_dist: dict = {}
    for p in cohort:
        eth_dist[p["ethnicity"]] = eth_dist.get(p["ethnicity"], 0) + 1

    # nNO tiers
    nno_tiers = {
        "< 30 nL/min (severe suppression — classic ODA/IDA)": 0,
        "30–60 nL/min (moderate suppression)":                0,
        "61–77 nL/min (borderline low)":                      0,
        "> 77 nL/min (near-normal — RSPH/HYDIN/DNAH11)":      0,
    }
    for p in cohort:
        nno = p["nasal_no_nl_min"]
        if nno < 30:
            nno_tiers["< 30 nL/min (severe suppression — classic ODA/IDA)"] += 1
        elif nno < 60:
            nno_tiers["30–60 nL/min (moderate suppression)"] += 1
        elif nno <= 77:
            nno_tiers["61–77 nL/min (borderline low)"] += 1
        else:
            nno_tiers["> 77 nL/min (near-normal — RSPH/HYDIN/DNAH11)"] += 1

    # FEV1 tiers
    fev1_tiers = {
        "≥ 80% (mild/normal)":     sum(1 for p in cohort if p["fev1_pct_pred"] >= 80),
        "60–79% (moderate)":       sum(1 for p in cohort if 60 <= p["fev1_pct_pred"] < 80),
        "40–59% (severe)":         sum(1 for p in cohort if 40 <= p["fev1_pct_pred"] < 60),
        "< 40% (very severe)":     sum(1 for p in cohort if p["fev1_pct_pred"] < 40),
    }

    # Age at diagnosis tiers
    age_tiers = {
        "< 2 yr (neonatal/infant)":  sum(1 for p in cohort if p["age_at_diagnosis_yr"] < 2),
        "2–5 yr (early childhood)":  sum(1 for p in cohort if 2 <= p["age_at_diagnosis_yr"] < 6),
        "6–12 yr (school age)":      sum(1 for p in cohort if 6 <= p["age_at_diagnosis_yr"] < 13),
        "13–18 yr (adolescent)":     sum(1 for p in cohort if 13 <= p["age_at_diagnosis_yr"] < 19),
        "≥ 19 yr (adult late dx)":   sum(1 for p in cohort if p["age_at_diagnosis_yr"] >= 19),
    }

    # Situs
    situs_dist = {
        "Situs solitus (normal)": sum(1 for p in cohort if not p["situs_inversus"]),
        "Situs inversus totalis (Kartagener)": sum(1 for p in cohort if p["situs_inversus"]),
    }

    return {
        "gene_distribution":      gene_dist,
        "tem_class":              tem_dist,
        "beat_pattern":           beat_dist,
        "pulmonary_finding":      pulm_dist,
        "sinus_finding":          sinus_dist,
        "prior_misdiagnosis":     misdx_dist,
        "ethnicity":              eth_dist,
        "nno_tiers":              nno_tiers,
        "fev1_tiers":             fev1_tiers,
        "age_at_diagnosis_tiers": age_tiers,
        "situs_distribution":     situs_dist,
        "summary": {
            "n":                   n,
            "pct_situs_inversus":  round(sum(1 for p in cohort if p["situs_inversus"]) / n * 100, 1),
            "pct_bronchiectasis":  round(sum(1 for p in cohort if p["has_bronchiectasis"]) / n * 100, 1),
            "pct_hearing_loss":    round(sum(1 for p in cohort if p["hearing_loss_ome"]) / n * 100, 1),
            "pct_male_infertile":  round(sum(1 for p in cohort if p["male_infertility"]) / n * 100, 1),
            "pct_nrd":             round(sum(1 for p in cohort if p["neonatal_rd"]) / n * 100, 1),
            "mean_fev1_pct":       round(statistics.mean(p["fev1_pct_pred"] for p in cohort), 1),
            "median_nno":          round(statistics.median(p["nasal_no_nl_min"] for p in cohort), 1),
            "pct_acbt_adherent":   round(sum(1 for p in cohort if p["acbt_adherent"]) / n * 100, 1),
        },
    }


def get_definitions() -> dict:
    return {
        "disease": "Primary Ciliary Dyskinesia (PCD)",
        "omim_gene": "DNAH5 *603335 (most common); DNAI1 *604366; DNAI2 *605483; >50 genes",
        "omim_disease": "#244400 (PCD1/DNAI1) · #608644 (PCD2/DNAH5) · #613798 (PCD7/CCDC39)",
        "chromosome": "5p15.2 (DNAH5); multi-locus",
        "inheritance": "Autosomal Recessive (biallelic LOF); X-linked rare (PIH1D3)",
        "prevalence": "~1/10,000–1/20,000; estimated 1/15,000 average",
        "mechanism": (
            "PCD is a motile ciliopathy: structural/functional defect of the 9+2 dynein arm axoneme in motile "
            "cilia and sperm flagella. Classified by ultrastructure defect class (ODA, IDA, RSH, CP, RGMC)."
        ),
        "kartagener_syndrome": (
            "Kartagener Triad = PCD + situs inversus totalis + bronchiectasis + sinusitis. "
            "~50% of PCD patients have situs inversus (embryonic nodal cilia defect → random laterality). "
            "DNAH11 has highest situs rate (~85%); RSPH/HYDIN subtypes rarely have situs inversus."
        ),
        "axoneme_defect_classes": {
            "ODA (outer dynein arm)": "DNAH5, DNAI1, DNAI2, ARMC4 — ~70% of PCD; TEM: ODA absent; immotile/slow beat",
            "IDA + MT transposition": "CCDC39, CCDC40 — ~10%; TEM: IDA absent + MTD; hyperkinetic dyskinetic beat",
            "ODA + IDA combined":     "DNAAF1-5, LRRC6 — ~5%; both arms absent; severe ciliary paralysis",
            "RSH (radial spoke head)": "RSPH4A, RSPH9 — ~5%; NORMAL TEM; circular/rotational beat; near-normal nNO",
            "CP (central pair)":      "HYDIN — ~3%; NORMAL TEM; abnormal waveform; nNO may be near-normal",
            "RGMC":                   "CCNO, MCIDAS — ~3%; TEM normal but cilia markedly reduced in number",
        },
        "diagnostic_criteria": {
            "nasal_NO": "nNO < 77 nL/min (adults >5yr) — PCD screening threshold; RSPH/HYDIN may be borderline",
            "HSVM":     "Abnormal ciliary beat pattern (static, hyperkinetic, circular, dyskinetic) on HSVM video",
            "TEM":      "ODA defect most common; 30% of PCD has normal TEM — cannot exclude by TEM alone",
            "gene_panel": ">50-gene PCD panel confirms biallelic variants; mandatory alongside TEM/HSVM",
            "IF":         "Immunofluorescence (DNAH5, DNAI1, DNAI2, CCDC39 antibodies) reduces need for TEM in common subtypes",
        },
        "key_clinical_features": {
            "neonatal_RD":    "Unexplained NRD in ≥37wk term neonate — upper lobe changes; ~80% of PCD term neonates",
            "wet_cough":      "Persistent wet/productive cough from infancy, year-round — not seasonal (distinguishes from asthma)",
            "bronchiectasis": "Progressive bilateral bronchiectasis — lower lobe predominant; preventable with early airway clearance",
            "sinusitis":      "Pansinusitis; frontal sinus hypoplasia/aplasia is PCD-specific (frontal sinus absent in ~50%)",
            "OME_glue_ear":   "Secretory otitis media / glue ear (50%) — Eustachian tube ciliary dysfunction; conductive HL",
            "situs_inversus": "~50% PCD — random left-right assignment from non-functional embryonic nodal cilia",
            "male_infertility": "Immotile/dyskinetic sperm (flagella = axoneme) → infertility; ICSI achieves paternity",
            "ectopic_pregnancy": "Fallopian tube ciliary dysfunction → impaired ovum transport → ectopic risk",
            "hydrocephalus":  "Ependymal cilia lining brain ventricles (~5%) → CSF flow impairment → hydrocephalus",
        },
        "ddx_vs_cf": {
            "sweat_chloride":  "Normal in PCD (>60 mmol/L required for CF); PCD has no CFTR mutation",
            "newborn_screen":  "PCD NOT detected by CF NBS (IRT/CFTR based); PCD has no equivalent NBS yet",
            "common_features": "Both: bronchiectasis, sinusitis, recurrent chest infections",
            "distinguishing":  "CF: pancreatic insufficiency, malnutrition, CFTR sweat; PCD: situs inversus, male infertility, nNO very low, CFTR normal",
        },
        "treatment": {
            "airway_clearance": "ACBT, PEP devices, huffing — TWICE DAILY lifelong; most important intervention",
            "prophylactic_abx": "Low-dose azithromycin (250 mg M/W/F adults) for ≥3 exacerbations/yr or established bronchiectasis",
            "nasal_irrigation": "Hypertonic saline nasal rinses daily — reduces sinus mucus burden",
            "grommets":         "Grommets (tympanostomy tubes) for persistent OME/glue ear and conductive HL",
            "ICSI":             "Intracytoplasmic sperm injection — standard ART for PCD male infertility; success rates comparable to other causes",
            "emerging":         "mRNA / gene-augmentation strategies in pre-clinical phase (DNAH5, DNAI1 targets); no approved disease-modifying therapy 2026",
        },
        "key_genes_clinical_pearls": {
            "DNAH5":   "Most common (~15–28%); ODA defect; TEM classic ODA absent; situs inversus ~50%; European c.7915C>T founder",
            "DNAI1":   "Second most common (~9–13%); ODA; IVS1+2_3insT Middle-Eastern founder; Kartagener classic",
            "CCDC39":  "IDA + MT transposition; hyperkinetic dyskinetic beat; bronchiectasis severe; gene panel priority",
            "CCDC40":  "IDA defect (Type II); Iberian c.248delC founder; worse lung phenotype than ODA subtypes",
            "DNAH11":  "Situs inversus very high (~85%); NORMAL TEM ultrastructure; near-normal nNO; diagnosed by HSVM + gene panel only",
            "RSPH4A":  "Radial spoke head defect; NORMAL TEM; rotational/circular beat on HSVM; nNO borderline; East Asian enriched",
            "HYDIN":   "Central pair (C2b absent); NORMAL TEM; rare; nNO may be near-normal; avoid TEM-only diagnosis",
            "CCNO":    "RGMC: reduced generation of motile cilia; near-normal on TEM; severe recurrent infections; CCNO + MCIDAS 5q11.2 cluster",
        },
        "founder_variants": [
            "DNAH5 c.7915C>T p.Arg2639Ter — Northern European (UK, Scandinavia)",
            "DNAI1 IVS1+2_3insT — Middle Eastern / North African founder",
            "CCDC40 c.248delC p.Leu83fs — Iberian (Spanish/Portuguese) founder",
            "RSPH9 c.1-2A>G — Ashkenazi Jewish / Middle Eastern",
            "CCDC103 p.His154Pro — South Asian (India/Pakistan) founder",
            "LRRC6 c.223G>A p.Gly75Arg — North African founder",
        ],
        "prognosis": (
            "NOT uniformly lethal (unlike Meckel-Gruber). Most PCD patients survive to adulthood. "
            "Lung function decline is progressive but preventable with aggressive early airway clearance. "
            "~10–15% require lung transplant by 4th–5th decade. Life expectancy approaching normal with optimal management. "
            "Male infertility is nearly universal but paternity achievable via ICSI. "
            "No approved disease-modifying therapy in 2026; airway clearance + antibiotics remain the standard of care."
        ),
        "cohort_note": f"Synthetic cohort (seed={SEED}, n=40) — epidemiological proportions match published literature (Bush et al. 2022 ERS Guidelines; Shapiro et al. 2021 NEJM).",
    }
