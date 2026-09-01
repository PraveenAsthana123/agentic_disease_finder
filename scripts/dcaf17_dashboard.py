"""
Woodhouse-Sakati Syndrome — DCAF17 (C2orf37)
==============================================
40-patient cohort · DCAF17 (2q31.1) · Autosomal Recessive · ~60-70 families worldwide 2026
First described: Woodhouse & Sakati 1983 (Saudi Arabia) — hypogonadism + alopecia + DM + extrapyramidal + hearing loss
NBIA-adjacent iron accumulation disorder — differs from classic NBIA1-7 (no Eye-of-Tiger, no cortical iron)
Cardinal pentad: Hypogonadism + Alopecia + Diabetes + Extrapyramidal features + Sensorineural hearing loss
Hypergonadotropic hypogonadism (primary gonadal failure) — elevated FSH/LH, low sex steroids
DCAF17 nucleolar protein LOF → CUL4A-DDB1 E3 ubiquitin ligase substrate receptor lost → ribosome biogenesis defect
Saudi/Middle East founder mutation: c.436delC (p.Gln146Lysfs*48) — ~60% of worldwide cases
PHT/CBZ AVOID (worsen extrapyramidal + interact with hormone replacement)
LEV PREFERRED first-line AED; hormone replacement (estrogen/testosterone) MANDATORY
GPi-DBS investigational (Level D); no disease-modifying therapy 2026

DCAF17 BIOLOGY:
DCAF17 (also called C2orf37) encodes a 263-amino-acid nucleolar protein functioning as a substrate-recognition
receptor within the CUL4A-DDB1 E3 ubiquitin ligase complex. It contains 3 WD-repeat propeller units
(aa 50-230) forming a beta-propeller scaffold that positions substrates for ubiquitination.
Primary localisation: nucleolus (ribosomal DNA transcription zones) and nucleus.
Function: regulates ubiquitination and proteasomal degradation of ribosome biogenesis factors;
essential for proper nucleolar organisation and rRNA processing (pre-18S, 5.8S, 28S maturation).
DCAF17 LOF (biallelic) → substrate recognition lost → ubiquitin ligase complex misdirected
→ ribosome biogenesis factors accumulate or fail to be recycled → global protein synthesis defect
→ tissues with highest protein turnover most vulnerable: gonadal cells, neurons, hair follicles, pancreatic β-cells.

Why endocrine + neurological?
Gonadal germ cells and Sertoli/granulosa cells divide rapidly and require high ribosome output.
Neurons, especially striatal and cerebellar, are post-mitotic and depend on continuous ribosome renewal.
Hair follicle cycling (anagen/catagen) is protein-synthesis-intensive.
Pancreatic β-cells: insulin synthesis is a high-throughput ribosomal process.

DCAF17 PROTEIN STRUCTURE (263 aa, 2q31.1):
  Signal/NLS region (aa 1-49): nuclear localisation signals; mitochondrial-like targeting ambiguous.
  WD-repeat propeller (aa 50-230): 3 WD-repeat units forming 7-bladed beta-propeller.
    WD1 (aa 50-90): substrate docking; Saudi founder frameshift breaks WD1-WD2 junction.
    WD2 (aa 91-160): DDB1 binding interface; c.436delC (p.Gln146Lysfs*48) disrupts WD2.
    WD3 (aa 161-230): CUL4A scaffold binding; missense variants cluster here.
  C-terminal tail (aa 231-263): nuclear retention; some truncating variants preserve partial NLS.
  No enzymatic active site — purely an adaptor/scaffold within E3 ligase.

PATHOGENIC VARIANT DISTRIBUTION (biallelic LOF mutations, n=40 patients):
  Frameshift/truncating: ~65% (near-complete LOF; classic severe phenotype)
  Missense (WD-repeat disruption): ~25% (partial LOF; milder/late-onset phenotype)
  Splice site: ~8% (exon skipping → partial domain loss)
  Large deletion: ~2%
  FOUNDER: c.436delC (p.Gln146Lysfs*48) — Saudi Arabian / Middle Eastern — ~60% of severe cases.
  European variants: diverse missense + splice; more variable phenotype.

CLINICAL PHENOTYPE:
  HYPOGONADISM (hypergonadotropic, primary):
    Females: primary amenorrhea or premature ovarian insufficiency; infertility (100% symptomatic females).
      FSH elevated (>30 IU/L), LH elevated (>20 IU/L), oestradiol very low (<50 pmol/L).
    Males: small testes (<6 mL Prader orchidometer), azoospermia, absent/sparse pubic + facial hair.
      FSH markedly elevated (>30 IU/L), testosterone low (<5 nmol/L).
    Onset: puberty (failure to develop) or early adulthood regression.
    PATHOGNOMONIC when combined with alopecia + extrapyramidal in any ethnic background.
  ALOPECIA (diffuse, progressive):
    Onset typically before neurological features (often ages 8-18yr).
    Diffuse non-scarring alopecia of scalp; may extend to eyebrows/lashes.
    Mechanism: ribosome biogenesis defect → hair follicle cycling failure → premature catagen.
    NOT androgenic alopecia (pattern differs; bilateral + diffuse from early age).
  EXTRAPYRAMIDAL FEATURES:
    Choreoathetosis (60-70%): involuntary writhing + jerking movements, limbs > trunk.
    Dystonia (50-60%): limb dystonia + facial grimacing; oromandibular in advanced disease.
    Dysarthria (70-80%): secondary to oromandibular dystonia + choreoathetosis.
    Parkinsonism (15-25%): resting tremor, rigidity, bradykinesia — LATE feature (>35yr).
    Cerebellar ataxia (20-30%): mild-moderate; often overshadowed by choreic component.
  DIABETES MELLITUS:
    Type 2-like initially (insulin resistance); β-cell ribosome defect → insulin synthesis failure later.
    70-80% develop DM; often first biochemical abnormality before neurological features.
    Onset age 15-35yr; progresses to insulin-dependence in ~30% by disease course.
  SENSORINEURAL HEARING LOSS (30-40%):
    Bilateral symmetric high-frequency loss; cochlear hair cells vulnerable to ribosome defect.
    Audiometry mandatory — often subclinical until formal testing.
  MRI FINDINGS:
    T2 white matter signal changes: frontal/parietal periventricular — EARLIEST finding.
    Mild iron: GP bilateral SWI hypointensity (MILD — unlike PKAN Eye-of-Tiger).
    SN iron: mild T2/SWI hypointensity — subcortical > cortical.
    No Eye-of-Tiger sign (DDx PKAN/NBIA1).
    No cortical iron (DDx Aceruloplasminemia/CP).
    No cavitations (DDx FTL/Neuroferritinopathy).
    No leukodystrophy pattern (DDx FAHN/NBIA3) — WM changes mild vs FAHN.
  SEIZURES (20-25%):
    Late-onset in most; myoclonic and focal types most common.
    Secondary to white matter + cortical hyperexcitability from neurodegeneration.
    NOT a primary presenting feature (contrast PKAN 30-40%, MPAN 50-60%).
  COGNITIVE DECLINE (50-60%):
    Frontal executive dysfunction earliest; global dementia late.
    Correlates with white matter T2 changes.
  PERIPHERAL NEUROPATHY (25-35%):
    Mixed motor-sensory axonal; NCS mandatory if suspected.
"""

import random

SEED = 531
DISEASE = "Woodhouse-Sakati Syndrome (DCAF17 — C2orf37 Nucleolar Protein Deficiency)"
GENE = "DCAF17 / C2orf37 (263 aa, 2q31.1) — WD-repeat nucleolar protein; biallelic LOF → CUL4A-DDB1 E3 ligase substrate receptor lost → ribosome biogenesis defect"
CHROMOSOME = "2q31.1"
OMIM_GENE = "612515"
OMIM_DISEASE = "241080"
PREVALENCE = "~60-70 families worldwide (AR, 2026); Saudi/Middle-East founder c.436delC"
FIRST_DESCRIBED = "Woodhouse & Sakati, 1983 (Saudi Arabia)"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF; consanguinity in ~70% of reported families"

RNG = random.Random(SEED)

PHENOTYPES = ["Classic WSS", "Neurodegenerative Predominant", "Endocrine Predominant", "Mild/Late-onset"]
PHENOTYPE_WEIGHTS = [0.50, 0.25, 0.20, 0.05]

VARIANTS = [
    "c.436delC / p.Gln146Lysfs*48 (Saudi founder, WD2 frameshift)",
    "c.1A>G / p.Met1? (start codon loss, no protein)",
    "p.Arg67* (nonsense, NLS-WD1 junction, truncation)",
    "c.IVS4+1G>A (splice-site, exon 4 skip, WD2 partial deletion)",
    "p.Asp180Gly (WD3 missense, CUL4A interface, partial LOF)",
    "p.Trp120Cys (WD2 missense, DDB1 binding, moderate LOF)",
    "p.Val95Ala (WD1 missense, substrate docking, mild LOF)",
    "Large deletion 2q31.1 (exons 1-4, complete LOF)",
]
VARIANT_WEIGHTS = [0.60, 0.06, 0.08, 0.08, 0.08, 0.05, 0.03, 0.02]

TREATMENT_OPTIONS = [
    "HRT (oestrogen/progesterone) — females",
    "HRT (testosterone) — males",
    "Levetiracetam (LEV) — AED first-line",
    "Tetrabenazine — chorea",
    "Clonazepam — myoclonus/anxiety",
    "Supportive only / no active Rx",
]


def _make_patient(i: int, phenotype: str) -> dict:
    rng = RNG

    # Common features
    sex = rng.choice(["Female", "Male"])

    if phenotype == "Classic WSS":
        onset_yr = rng.randint(8, 22)
        hypogonadism = True
        alopecia = True
        diabetes = rng.random() < 0.85
        hearing_loss = rng.random() < 0.40
        chorea = rng.random() < 0.70
        dystonia = rng.random() < 0.60
        dysarthria = rng.random() < 0.80
        parkinsonism = rng.random() < 0.20
        seizures = rng.random() < 0.22
        cognitive_decline = rng.random() < 0.55
        neuropathy = rng.random() < 0.30
        wm_changes = True
        gp_iron = rng.random() < 0.60
        fsh = round(rng.uniform(35, 90), 1)
        sex_steroid_low = True
        insulin_dependent = rng.random() < 0.30 if diabetes else False

    elif phenotype == "Neurodegenerative Predominant":
        onset_yr = rng.randint(15, 35)
        hypogonadism = rng.random() < 0.40
        alopecia = rng.random() < 0.50
        diabetes = rng.random() < 0.50
        hearing_loss = rng.random() < 0.30
        chorea = rng.random() < 0.80
        dystonia = rng.random() < 0.75
        dysarthria = rng.random() < 0.85
        parkinsonism = rng.random() < 0.35
        seizures = rng.random() < 0.35
        cognitive_decline = True
        neuropathy = rng.random() < 0.40
        wm_changes = True
        gp_iron = rng.random() < 0.75
        fsh = round(rng.uniform(15, 50), 1) if hypogonadism else round(rng.uniform(4, 12), 1)
        sex_steroid_low = hypogonadism
        insulin_dependent = rng.random() < 0.25 if diabetes else False

    elif phenotype == "Endocrine Predominant":
        onset_yr = rng.randint(10, 20)
        hypogonadism = True
        alopecia = True
        diabetes = rng.random() < 0.90
        hearing_loss = rng.random() < 0.25
        chorea = rng.random() < 0.35
        dystonia = rng.random() < 0.25
        dysarthria = rng.random() < 0.30
        parkinsonism = rng.random() < 0.08
        seizures = rng.random() < 0.12
        cognitive_decline = rng.random() < 0.30
        neuropathy = rng.random() < 0.20
        wm_changes = rng.random() < 0.50
        gp_iron = rng.random() < 0.30
        fsh = round(rng.uniform(30, 80), 1)
        sex_steroid_low = True
        insulin_dependent = rng.random() < 0.20 if diabetes else False

    else:  # Mild/Late-onset
        onset_yr = rng.randint(25, 45)
        hypogonadism = rng.random() < 0.70
        alopecia = rng.random() < 0.60
        diabetes = rng.random() < 0.60
        hearing_loss = rng.random() < 0.20
        chorea = rng.random() < 0.40
        dystonia = rng.random() < 0.30
        dysarthria = rng.random() < 0.35
        parkinsonism = rng.random() < 0.15
        seizures = rng.random() < 0.10
        cognitive_decline = rng.random() < 0.25
        neuropathy = rng.random() < 0.15
        wm_changes = rng.random() < 0.40
        gp_iron = rng.random() < 0.25
        fsh = round(rng.uniform(20, 55), 1) if hypogonadism else round(rng.uniform(3, 10), 1)
        sex_steroid_low = hypogonadism
        insulin_dependent = rng.random() < 0.15 if diabetes else False

    disease_dur_yr = rng.randint(3, 25)
    variant = rng.choices(VARIANTS, weights=VARIANT_WEIGHTS)[0]
    if sex == "Female":
        treatment = rng.choices(
            ["HRT (oestrogen/progesterone) — females", "Levetiracetam (LEV) — AED first-line",
             "Tetrabenazine — chorea", "Supportive only / no active Rx",
             "HRT (oestrogen/progesterone) — females", "Clonazepam — myoclonus/anxiety"],
            weights=[0.40, 0.25, 0.15, 0.10, 0.00, 0.10]
        )[0]
    else:
        treatment = rng.choices(
            ["HRT (testosterone) — males", "Levetiracetam (LEV) — AED first-line",
             "Tetrabenazine — chorea", "Supportive only / no active Rx",
             "Clonazepam — myoclonus/anxiety"],
            weights=[0.40, 0.28, 0.15, 0.10, 0.07]
        )[0]

    return {
        "id": f"WSS-{i+1:03d}",
        "phenotype": phenotype,
        "sex": sex,
        "onset_yr": onset_yr,
        "disease_dur_yr": disease_dur_yr,
        "hypogonadism": hypogonadism,
        "alopecia": alopecia,
        "diabetes": diabetes,
        "insulin_dependent": insulin_dependent,
        "hearing_loss": hearing_loss,
        "chorea": chorea,
        "dystonia": dystonia,
        "dysarthria": dysarthria,
        "parkinsonism": parkinsonism,
        "seizures": seizures,
        "cognitive_decline": cognitive_decline,
        "neuropathy": neuropathy,
        "wm_changes": wm_changes,
        "gp_iron": gp_iron,
        "fsh_iu_l": fsh,
        "sex_steroid_low": sex_steroid_low,
        "variant": variant,
        "treatment": treatment,
    }


def _build_patients() -> list:
    pts = []
    for i, ph in enumerate(RNG.choices(PHENOTYPES, weights=PHENOTYPE_WEIGHTS, k=40)):
        pts.append(_make_patient(i, ph))
    return pts


def _pct(pts: list, key: str) -> int:
    return round(sum(1 for p in pts if p.get(key)) / len(pts) * 100)


def _avg(pts: list, key: str) -> float:
    vals = [p[key] for p in pts if isinstance(p.get(key), (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def get_overview() -> dict:
    pts = _build_patients()
    by_phenotype = {}
    for ph in PHENOTYPES:
        by_phenotype[ph] = [p for p in pts if p["phenotype"] == ph]

    phenotype_dist = [
        {"phenotype": ph, "n": len(g), "pct": round(len(g) / 40 * 100)}
        for ph, g in by_phenotype.items()
    ]

    kpis = {
        "n_patients": 40,
        "n_classic": len(by_phenotype["Classic WSS"]),
        "n_neurodegen": len(by_phenotype["Neurodegenerative Predominant"]),
        "n_endocrine": len(by_phenotype["Endocrine Predominant"]),
        "n_mild": len(by_phenotype["Mild/Late-onset"]),
        "hypogonadism_pct": _pct(pts, "hypogonadism"),
        "alopecia_pct": _pct(pts, "alopecia"),
        "diabetes_pct": _pct(pts, "diabetes"),
        "chorea_pct": _pct(pts, "chorea"),
        "dystonia_pct": _pct(pts, "dystonia"),
        "dysarthria_pct": _pct(pts, "dysarthria"),
        "hearing_loss_pct": _pct(pts, "hearing_loss"),
        "seizures_pct": _pct(pts, "seizures"),
        "cognitive_decline_pct": _pct(pts, "cognitive_decline"),
        "wm_changes_pct": _pct(pts, "wm_changes"),
        "gp_iron_pct": _pct(pts, "gp_iron"),
        "neuropathy_pct": _pct(pts, "neuropathy"),
        "mean_onset_yr": _avg(pts, "onset_yr"),
        "mean_fsh": _avg(pts, "fsh_iu_l"),
        "insulin_dependent_pct": _pct(pts, "insulin_dependent"),
    }

    clinical_highlights = [
        {"finding": "Hypogonadism (hypergonadotropic)", "pct": kpis["hypogonadism_pct"],
         "note": "Elevated FSH/LH + low sex steroids; primary gonadal failure → PATHOGNOMONIC cardinal feature"},
        {"finding": "Alopecia (diffuse, non-androgenic)", "pct": kpis["alopecia_pct"],
         "note": "Often first presenting feature (childhood–teen); diffuse scalp, eyebrows; ribosome biogenesis defect"},
        {"finding": "Dysarthria", "pct": kpis["dysarthria_pct"],
         "note": "Secondary to oromandibular dystonia + chorea; hallmark of extrapyramidal progression"},
        {"finding": "Chorea / Choreoathetosis", "pct": kpis["chorea_pct"],
         "note": "Involuntary writhing + jerking; limbs > trunk; striatal neurodegeneration"},
        {"finding": "Diabetes mellitus", "pct": kpis["diabetes_pct"],
         "note": "Type 2-like initially; ribosome defect in β-cells → insulin synthesis failure; up to 30% insulin-dependent"},
        {"finding": "Dystonia", "pct": kpis["dystonia_pct"],
         "note": "Limb + oromandibular; may co-occur with chorea; GPi-DBS investigational (Level D)"},
        {"finding": "White matter T2 changes", "pct": kpis["wm_changes_pct"],
         "note": "Frontal > parietal periventricular; earliest MRI finding; non-leukodystrophic (DDx FAHN/NBIA3)"},
        {"finding": "GP iron (SWI mild)", "pct": kpis["gp_iron_pct"],
         "note": "Mild bilateral GP hypointensity on SWI/GRE; UNLIKE PKAN (Eye-of-Tiger absent); milder than NBIA1-7"},
        {"finding": "Cognitive decline", "pct": kpis["cognitive_decline_pct"],
         "note": "Frontal executive dysfunction earliest; correlates with white matter T2 burden"},
        {"finding": "Sensorineural hearing loss", "pct": kpis["hearing_loss_pct"],
         "note": "Bilateral symmetric high-frequency; cochlear ribosome defect; audiometry mandatory"},
        {"finding": "Peripheral neuropathy", "pct": kpis["neuropathy_pct"],
         "note": "Mixed motor-sensory axonal; NCS mandatory; adds to functional disability"},
        {"finding": "Seizures", "pct": kpis["seizures_pct"],
         "note": "Late-onset; myoclonic + focal; NOT primary presenting feature; LEV first-line"},
        {"finding": "Parkinsonism (resting tremor, rigidity)", "pct": _pct(pts, "parkinsonism"),
         "note": "Late feature (>35yr); secondary to basal ganglia iron + dopaminergic loss; L-DOPA response variable"},
    ]

    contraindications = [
        {
            "drug": "PHT (Phenytoin)",
            "severity": "AVOID",
            "reason": "Worsens extrapyramidal features (dystonia/chorea aggravation); CYP2C9/2C19 induction significantly reduces hormone replacement levels (oestrogen/testosterone)",
            "alternative": "LEV first-line; VPA if POLG excluded",
        },
        {
            "drug": "CBZ (Carbamazepine)",
            "severity": "AVOID",
            "reason": "Strong CYP3A4 inducer → markedly reduces sex hormone levels → HRT failure; exacerbates extrapyramidal features",
            "alternative": "LEV first-line; LTG second-line (mild CYP induction)",
        },
        {
            "drug": "OXC (Oxcarbazepine)",
            "severity": "CAUTION",
            "reason": "Moderate CYP3A4 inducer → reduces hormone replacement efficacy; extrapyramidal side-effect profile overlaps",
            "alternative": "LEV preferred; LTG acceptable",
        },
        {
            "drug": "VPA (Valproate)",
            "severity": "CAUTION (POLG screen MANDATORY first)",
            "reason": "POLG1 mutation exclusion mandatory before use (mitochondrial overlap); hepatotoxic risk; interacts with hormone metabolism",
            "alternative": "LEV first-line; confirm POLG negative before VPA trial",
        },
        {
            "drug": "LEV (Levetiracetam)",
            "severity": "PREFERRED FIRST-LINE",
            "reason": "Minimal CYP450 induction → does NOT reduce hormone replacement levels; renal excretion → safe hepatic iron profile; broad-spectrum AED",
            "alternative": None,
        },
        {
            "drug": "Iron supplementation (oral/IV)",
            "severity": "CAUTION",
            "reason": "Not primary iron accumulation disease; GP iron mild; excess iron could theoretically worsen GP/SN burden. Not absolutely contraindicated but monitor carefully",
            "alternative": "Treat iron deficiency anaemia only if biochemically confirmed; use MRI iron monitoring",
        },
    ]

    thresholds = [
        {"metric": "FSH", "threshold": ">30 IU/L (females)", "action": "Diagnose hypergonadotropic hypogonadism; initiate HRT (oestrogen/progesterone) immediately"},
        {"metric": "FSH", "threshold": ">30 IU/L (males) + LH>20", "action": "Diagnose primary hypogonadism; initiate testosterone replacement; spermatogenesis unlikely"},
        {"metric": "Oestradiol", "threshold": "<50 pmol/L (females)", "action": "Primary ovarian failure confirmed; HRT protects bone + cardiovascular health"},
        {"metric": "Testosterone", "threshold": "<5 nmol/L (males)", "action": "Testosterone replacement indicated; monitor haematocrit (secondary polycythaemia risk)"},
        {"metric": "HbA1c", "threshold": "≥6.5% (48 mmol/mol)", "action": "Diabetes confirmed; endocrinology referral; SGLT2/metformin first-line; insulin if β-cell exhaustion"},
        {"metric": "Audiometry", "threshold": ">25 dB HL at 4 kHz", "action": "Sensorineural hearing loss confirmed; hearing aid + audiology follow-up"},
        {"metric": "GP SWI", "threshold": "Bilateral hypointensity", "action": "Iron accumulation; correlate with clinical; no mandatory chelation (mild) — monitor with R2*/QSM"},
        {"metric": "WM T2 load", "threshold": "Frontal periventricular", "action": "Baseline cognitive assessment; repeat MRI annually; neuropsychological evaluation"},
    ]

    return {
        "disease": DISEASE,
        "gene": GENE,
        "chromosome": CHROMOSOME,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "prevalence": PREVALENCE,
        "first_described": FIRST_DESCRIBED,
        "inheritance": INHERITANCE,
        "kpis": kpis,
        "phenotype_distribution": phenotype_dist,
        "clinical_highlights": clinical_highlights,
        "contraindications": contraindications,
        "thresholds": thresholds,
        "seed": SEED,
    }


def get_breakdown() -> dict:
    pts = _build_patients()
    by_phenotype = {}
    for ph in PHENOTYPES:
        g = [p for p in pts if p["phenotype"] == ph]
        by_phenotype[ph] = g

    phenotype_breakdown = []
    for ph, g in by_phenotype.items():
        if not g:
            continue
        phenotype_breakdown.append({
            "phenotype": ph,
            "n": len(g),
            "pct": round(len(g) / 40 * 100),
            "mean_onset_yr": _avg(g, "onset_yr"),
            "hypogonadism_pct": _pct(g, "hypogonadism"),
            "alopecia_pct": _pct(g, "alopecia"),
            "diabetes_pct": _pct(g, "diabetes"),
            "chorea_pct": _pct(g, "chorea"),
            "dystonia_pct": _pct(g, "dystonia"),
            "dysarthria_pct": _pct(g, "dysarthria"),
            "seizures_pct": _pct(g, "seizures"),
            "wm_changes_pct": _pct(g, "wm_changes"),
            "gp_iron_pct": _pct(g, "gp_iron"),
            "cognitive_decline_pct": _pct(g, "cognitive_decline"),
            "hearing_loss_pct": _pct(g, "hearing_loss"),
            "neuropathy_pct": _pct(g, "neuropathy"),
            "mean_fsh": _avg(g, "fsh_iu_l"),
        })

    # Variant breakdown
    vcounts = {}
    for p in pts:
        v = p["variant"]
        vcounts[v] = vcounts.get(v, 0) + 1
    variant_breakdown = sorted(
        [{"variant": v, "n": n, "pct": round(n / 40 * 100)} for v, n in vcounts.items()],
        key=lambda x: -x["n"]
    )

    # Treatment breakdown
    tcounts = {}
    for p in pts:
        t = p["treatment"]
        tcounts[t] = tcounts.get(t, 0) + 1
    treatment_breakdown = sorted(
        [{"treatment": t, "n": n, "pct": round(n / 40 * 100)} for t, n in tcounts.items()],
        key=lambda x: -x["n"]
    )

    # Iron regions
    iron_regions = [
        {"region": "GP (Globus Pallidus) — bilateral SWI hypointensity",
         "pct": _pct(pts, "gp_iron"),
         "note": "Mild; present in ~60% Classic WSS; NOT Eye-of-Tiger (DDx PKAN); less prominent than NBIA1-7"},
        {"region": "Substantia Nigra — mild T2/SWI low signal",
         "pct": round(_pct(pts, "gp_iron") * 0.75),
         "note": "SN pars reticulata > compacta; correlates with parkinsonism component; milder than NBIA1-4"},
        {"region": "White Matter — T2 hyperintensity (NOT iron)",
         "pct": _pct(pts, "wm_changes"),
         "note": "Periventricular frontal > parietal; earliest MRI finding; non-iron T2 signal; DDx FAHN leukodystrophy"},
        {"region": "Caudate — mild iron (late feature)",
         "pct": round(_pct(pts, "gp_iron") * 0.45),
         "note": "Late-disease feature; absent in Endocrine Predominant subtype; mild vs FTL caudate predominance"},
        {"region": "Cortex — NO iron accumulation",
         "pct": 0,
         "note": "KEY DDx from Aceruloplasminemia (CP) — no cortical iron in WSS; GPI-anchored CP intact"},
    ]

    # Patient table (sorted by onset age)
    patient_table = sorted(
        [{
            "id": p["id"],
            "phenotype": p["phenotype"],
            "sex": p["sex"],
            "onset_yr": p["onset_yr"],
            "disease_dur_yr": p["disease_dur_yr"],
            "hypogonadism": p["hypogonadism"],
            "alopecia": p["alopecia"],
            "diabetes": p["diabetes"],
            "chorea": p["chorea"],
            "dystonia": p["dystonia"],
            "seizures": p["seizures"],
            "fsh_iu_l": p["fsh_iu_l"],
            "treatment": p["treatment"],
        } for p in pts],
        key=lambda x: x["onset_yr"]
    )[:25]

    return {
        "phenotype_breakdown": phenotype_breakdown,
        "variant_breakdown": variant_breakdown,
        "treatment_breakdown": treatment_breakdown,
        "iron_regions": iron_regions,
        "patient_table": patient_table,
    }


def get_definitions() -> dict:
    defs = [
        {
            "term": "Woodhouse-Sakati-Syndrome-WSS",
            "full": "Woodhouse-Sakati Syndrome (DCAF17/C2orf37 — AR, 2q31.1) — Ultra-rare NBIA-adjacent neurodegeneration",
            "detail": (
                "Cardinal pentad: Hypergonadotropic hypogonadism + Diffuse alopecia + Diabetes mellitus + "
                "Extrapyramidal features (chorea/dystonia) + Sensorineural hearing loss (30-40%). "
                "First described Woodhouse & Sakati 1983 (Saudi Arabia); ~60-70 families worldwide 2026. "
                "Biallelic DCAF17 LOF → nucleolar dysfunction → ribosome biogenesis defect → "
                "protein synthesis failure in metabolically demanding tissues (gonads, neurons, hair follicles, β-cells). "
                "Brain iron mild (GP/SN) — NOT Eye-of-Tiger; white matter T2 changes earlier than iron."
            ),
        },
        {
            "term": "DCAF17-CUL4A-DDB1-E3-Ligase-Complex",
            "full": "DCAF17 — DDB1 and CUL4A-associated factor 17; WD-repeat nucleolar substrate receptor",
            "detail": (
                "263-amino-acid protein; 3 WD-repeat units (aa 50-230) forming 7-bladed beta-propeller. "
                "Functions as substrate-recognition receptor within CUL4A-DDB1-RBX1 E3 ubiquitin ligase complex. "
                "Localised to nucleolus (ribosomal DNA transcription compartment). "
                "DCAF17 LOF → rRNA processing factors not ubiquitinated on schedule → "
                "pre-rRNA maturation delayed → ribosome output reduced → global protein synthesis impaired "
                "in high-demand tissues (gonadal cells, post-mitotic neurons, hair follicle cycling cells)."
            ),
        },
        {
            "term": "Hypergonadotropic-Hypogonadism-WSS",
            "full": "Primary gonadal failure — elevated FSH/LH, low sex steroids — PATHOGNOMONIC combination in WSS",
            "detail": (
                "Females: primary amenorrhea (failure of menarche) or premature ovarian insufficiency (POI). "
                "FSH >30 IU/L; LH >20 IU/L; oestradiol <50 pmol/L. Infertility invariable. "
                "Males: small testes (<6 mL Prader orchidometer); azoospermia; absent pubic/axillary hair. "
                "FSH markedly elevated (>30 IU/L); testosterone <5 nmol/L. "
                "Mechanism: gonadal germ cells + Sertoli/granulosa cells require high ribosome output for "
                "steroidogenesis + gametogenesis; DCAF17 LOF → ribosome failure → gonadal dysfunction. "
                "HRT mandatory: protects bone mineral density + cardiovascular health regardless of neurological stage."
            ),
        },
        {
            "term": "Saudi-Founder-c436delC-pGln146Lysfs48",
            "full": "c.436delC (p.Gln146Lysfs*48) — Saudi Arabian/Middle Eastern founder mutation, ~60% of WSS cases",
            "detail": (
                "1-bp deletion at nucleotide 436 in exon 4 of DCAF17 → frameshift at codon 146 → "
                "premature stop at +48 codons → truncation within WD-repeat 2 domain → "
                "loss of DDB1-binding interface → complete E3 ligase substrate receptor loss → severe phenotype. "
                "Prevalent in consanguineous Saudi Arabian families; also found in other Middle Eastern + North African populations. "
                "Explains phenotypic homogeneity in Saudi cohort (Classic WSS in >80% of c.436delC homozygotes). "
                "Heterozygous carriers: asymptomatic (AD inheritance not established)."
            ),
        },
        {
            "term": "WSS-vs-PKAN-Eye-of-Tiger-DDx",
            "full": "WSS vs PKAN — NO Eye-of-Tiger sign in WSS; GP iron mild and uniform; alopecia + hypogonadism absent in PKAN",
            "detail": (
                "PKAN (PANK2/NBIA1): Eye-of-Tiger sign PATHOGNOMONIC — central T1 GP hyperintensity surrounded by T2 hypointense rim. "
                "WSS: No Eye-of-Tiger. GP SWI hypointensity mild + uniform (no central T1 bright). "
                "PKAN: onset <6yr (classic), retinopathy (50%), acanthocytes (50%), NO hypogonadism, NO alopecia. "
                "WSS: onset 8-35yr, hypogonadism + alopecia + diabetes — completely absent in PKAN. "
                "Definitive differentiation: DCAF17 vs PANK2 gene sequencing."
            ),
        },
        {
            "term": "WSS-vs-Aceruloplasminemia-CP-DDx",
            "full": "WSS vs CP (Aceruloplasminemia) — NO cortical iron in WSS; NO ceruloplasmin deficiency; different endocrine phenotype",
            "detail": (
                "CP (Aceruloplasminemia): Classic TRIAD — brain iron (cortical UNIQUE) + diabetes + retinal degeneration. "
                "Ceruloplasmin UNDETECTABLE; serum ferritin HIGH (>500 ng/mL); microcytic anemia. "
                "WSS: NO cortical iron (GPI-anchored CP intact); ceruloplasmin NORMAL; ferritin NORMAL. "
                "WSS: hypogonadism + alopecia + hearing loss — NOT seen in CP. "
                "CP: retinal degeneration → VGB ABSOLUTE CI (additive retinal toxicity). "
                "WSS: retina spared → VGB not absolutely contraindicated. "
                "Iron biomarkers (ferritin, ceruloplasmin, serum iron/TIBC) separate WSS from CP immediately."
            ),
        },
        {
            "term": "Hormone-Replacement-Therapy-HRT-WSS",
            "full": "HRT — MANDATORY in WSS; protects bone/cardiovascular; does NOT treat neurodegeneration",
            "detail": (
                "Females: combined oestrogen (17β-oestradiol 2 mg/day or equivalent patch) + progestogen (micronised progesterone 100-200 mg days 1-14). "
                "Treat until average age of menopause (51yr) then reassess. "
                "Males: testosterone undecanoate IM 1000 mg/12 wk or testosterone gel (50 mg/day). "
                "Monitor: LH/FSH/oestradiol or testosterone quarterly until stable, then 6-monthly. "
                "CRITICAL drug interaction: PHT/CBZ/OXC are strong CYP3A4 inducers → markedly reduce sex hormone levels → "
                "HRT failure → worsening osteoporosis + cardiovascular risk → AVOID these AEDs in WSS. "
                "LEV: no CYP450 induction → safe first-line AED with HRT."
            ),
        },
        {
            "term": "LEV-Preferred-AED-WSS-Mechanism",
            "full": "LEV (Levetiracetam) — preferred first-line AED in WSS; no CYP induction; no hormone interaction",
            "detail": (
                "Levetiracetam (SV2A modulator): renal excretion (66% unchanged), minimal hepatic metabolism, "
                "NO significant CYP450 induction → hormone replacement levels unaffected. "
                "Broad-spectrum: effective against myoclonic + focal seizures (both relevant in WSS). "
                "Tolerability: irritability/mood side-effect profile manageable; dose 500-3000 mg/day. "
                "PHT/CBZ: CYP2C9/CYP3A4 inducers → reduce oestrogen + testosterone plasma levels → HRT failure → AVOID. "
                "LTG: second-line (mild CYP induction, oestrogen reduces LTG levels → dose adjustment needed with HRT). "
                "POLG1 screen mandatory before VPA (mitochondrial ribosome overlap risk)."
            ),
        },
        {
            "term": "PHT-CBZ-AVOID-WSS-Reasons",
            "full": "PHT and CBZ AVOID in WSS — worsens extrapyramidal features + CYP-mediated HRT failure",
            "detail": (
                "Dual mechanism for avoidance: "
                "(1) Extrapyramidal worsening: PHT/CBZ act on sodium channels and modulate dopaminergic/serotonergic systems "
                "→ can aggravate chorea and dystonia in basal ganglia disease (same mechanism as in FTL/NBIA7). "
                "(2) HRT pharmacokinetic failure: PHT (CYP2C9/3A4 inducer) + CBZ (strong CYP3A4 inducer) "
                "markedly accelerate hepatic metabolism of oestrogen and testosterone → plasma levels halved → "
                "HRT ineffective → unopposed hypogonadism → severe osteoporosis + cardiovascular risk. "
                "This HRT interaction makes PHT/CBZ particularly harmful in WSS vs other NBIA types where HRT is not needed."
            ),
        },
        {
            "term": "Tetrabenazine-Deutetrabenazine-Chorea-WSS",
            "full": "Tetrabenazine/Deutetrabenazine — VMAT2 inhibitor for choreoathetosis in WSS (Level D)",
            "detail": (
                "Tetrabenazine (VMAT2 inhibitor): depletes presynaptic dopamine → reduces striatal dopamine-mediated "
                "chorea signal. Dose: 12.5 mg/day TID; titrate to 25 mg TID (max 100 mg/day). "
                "Deutetrabenazine: longer half-life, better tolerability (fewer peak-dose side-effects), same mechanism. "
                "Evidence in WSS: Level D (extrapolated from Huntington disease + other choreic disorders). "
                "Monitor: depression (risk ~15%), sedation, QTc prolongation. "
                "NOT useful for dystonia component (consider GPi-DBS for severe dystonia — Level D investigational). "
                "L-DOPA: can worsen chorea (increases striatal dopamine tone) — use with extreme caution; "
                "may benefit parkinsonism component if dominant in late disease."
            ),
        },
        {
            "term": "GPi-DBS-Investigational-WSS",
            "full": "GPi-DBS (Globus Pallidus internus Deep Brain Stimulation) — investigational Level D in WSS",
            "detail": (
                "Rationale: GPi is primary target for choreic-dystonic disorders. "
                "Evidence base: <5 WSS cases reported; extrapolated from GPi-DBS in Huntington (Level B) and NBIA subtypes. "
                "Best candidates: predominantly dystonic phenotype; severe functional impairment; stable cognitive status. "
                "Chorea: GPi-DBS shows variable response (Huntington data suggests 50-70% chorea reduction). "
                "Contraindications: severe cognitive decline; depression; medication-refractory psychiatric symptoms. "
                "Hormone replacement must be optimised before DBS (ensure HRT failure not mistaken for DBS non-response). "
                "Level D — multidisciplinary centre with movement disorder expertise required."
            ),
        },
        {
            "term": "WSS-MRI-WM-Changes-vs-Leukodystrophy-DDx",
            "full": "WSS white matter T2 changes — MILD periventricular; NOT true leukodystrophy (DDx FAHN/NBIA3)",
            "detail": (
                "FAHN (FA2H/NBIA3): leukodystrophy is the EARLIEST and MOST PROMINENT MRI feature — "
                "bilateral symmetric confluent T2 hyperintensity of cerebral white matter + cerebellar WM; "
                "progressive from childhood; distinguishes FAHN from all other NBIA subtypes. "
                "WSS: mild periventricular T2 signal (frontal > parietal), non-confluent; "
                "NOT true leukodystrophy — does not involve arcuate/U-fibres; progresses slowly. "
                "WSS: white matter changes precede iron accumulation (earliest MRI finding in WSS). "
                "DDx: FAHN has spastic paraplegia dominant + leukodystrophy + NO hypogonadism/alopecia. "
                "MRI spectroscopy may show mild NAA reduction (neuronal loss) in WSS without lactate (no mitochondrial disease)."
            ),
        },
        {
            "term": "Ribosome-Biogenesis-Disease-Mechanism",
            "full": "DCAF17 LOF → nucleolar dysfunction → ribosome biogenesis failure — WHY gonadal + neuronal selective vulnerability",
            "detail": (
                "Ribosomes are assembled in the nucleolus: rDNA transcribed → pre-rRNA (47S) → "
                "processed to 18S (small subunit) + 5.8S + 28S (large subunit) → assembled with ribosomal proteins "
                "→ exported as functional 80S ribosomes. "
                "DCAF17 (CUL4A-DDB1 substrate receptor): ubiquitinates ribosome biogenesis factors to control "
                "their turnover and prevent nucleolar congestion. DCAF17 LOF → ribosome assembly factors accumulate "
                "→ nucleolar stress → p53 activation → apoptosis in sensitive cell types. "
                "Selective vulnerability: (1) Gonadal germ/Sertoli/granulosa cells: ~8× average ribosome demand (rapid division). "
                "(2) Striatal + cerebellar neurons: post-mitotic, cannot replace lost ribosomes; high synaptic protein demand. "
                "(3) Hair follicle keratinocytes: anagen-phase cycling requires protein burst. "
                "(4) Pancreatic β-cells: insulin synthesis occupies ~50% of ribosomal capacity."
            ),
        },
        {
            "term": "POLG-Mandatory-Before-VPA-WSS",
            "full": "POLG1 mutation screen MANDATORY before valproate use in WSS",
            "detail": (
                "Standard NBIA-adjacent protocol: all patients must have POLG1 screening before valproate prescription. "
                "POLG mutations cause mitochondrial DNA depletion syndrome (Alpers-Huttenlocher when combined with VPA). "
                "DCAF17 itself does not affect mitochondria, but the ribosome biogenesis pathway intersects with "
                "mitochondrial protein synthesis (mitoribosomes also require rRNA processing factors). "
                "VPA-POLG interaction: fulminant hepatic failure risk → must be excluded in all WSS patients. "
                "If POLG negative AND no hepatic compromise: VPA may be used with careful monitoring. "
                "POLG testing: clinical-grade sequencing of POLG gene (coding + splice regions); typical turnaround 4-6 wk. "
                "Alternative while awaiting POLG result: LEV or LTG as bridge therapy."
            ),
        },
        {
            "term": "OMIM-612515-DCAF17-241080-WSS",
            "full": "OMIM Gene DCAF17/C2orf37: 612515 — Disease: Woodhouse-Sakati Syndrome: 241080",
            "detail": (
                "DCAF17 (MIM 612515): officially designated C2orf37 chromosome-2-open-reading-frame 37. "
                "Renamed DCAF17 when function as DDB1 and CUL4-associated factor 17 was characterised. "
                "WSS (MIM 241080): Woodhouse-Sakati Syndrome — autosomal recessive. "
                "First published: Woodhouse NJY, Sakati NA. 'A syndrome of hypogonadism, alopecia, diabetes mellitus, "
                "mental retardation, deafness and ECG abnormalities.' J Med Genet 1983;20:216-219. "
                "Molecular basis identified: Alazami et al. 2008 — c.436delC founder mutation in Saudi families. "
                "Treatment trials: none registered 2026; international WSS patient registry established 2022."
            ),
        },
    ]
    return {"definitions": defs}
