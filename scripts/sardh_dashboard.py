#!/usr/bin/env python3
"""SARDH (Sarcosine Dehydrogenase Deficiency / Sarcosinemia) Epilepsy Dashboard.

SARDH encodes Sarcosine Dehydrogenase, a FAD- and THF-dependent mitochondrial enzyme
that constitutes the second step in the sarcosine/N-methylglycine degradation pathway.

SARDH ENZYMATIC FUNCTION:

  Sarcosine (N-methylglycine) + THF → Glycine + 5,10-methyleneTHF

The enzyme uses:
  - FAD as the electron acceptor (oxidative demethylation of the N-methyl group)
  - THF (tetrahydrofolate) as the 1-carbon acceptor — captures the methyl group to form
    5,10-methyleneTHF (a critical folate cycle intermediate for serine biosynthesis and
    thymidylate synthesis)

SARDH IS THE METABOLIC OPPOSITE OF GNMT:
  GNMT  (9q22.2): SAM + Glycine → SAH + Sarcosine       (makes sarcosine)
  SARDH (9q34.3): Sarcosine + THF → Glycine + 5,10-mTHF (destroys sarcosine)

  In GNMT deficiency:  Sarcosine is ABSENT (cannot be made — GNMT blocks)
  In SARDH deficiency: Sarcosine is ELEVATED (cannot be cleared — SARDH blocks)
  → These are MIRROR-IMAGE sarcosine findings — the fastest single discriminating test.

SARDH LOF — SARCOSINE ACCUMULATES:
  Sarcosine: MARKEDLY ELEVATED (50–800 µmol/L; normal <10) — pathognomonic
  Urine: sarcosinuria — sarcosine spills into urine at high plasma concentrations
  Glycine: NORMAL or mildly elevated (mild accumulation; compensation modest)
  Methionine: NORMAL (<60 µmol/L) — KEY NEGATIVE vs GNMT/MAT1A/AHCY/CBS
  SAM: NORMAL — KEY NEGATIVE vs GNMT (SAM massively elevated)
  SAH: NORMAL — KEY NEGATIVE vs AHCY (SAH massively elevated)
  SAM/SAH ratio: NORMAL (2–4) — KEY NEGATIVE vs GNMT (>10–25) and AHCY (<0.5)
  tHcy: NORMAL (<15 µmol/L) — KEY NEGATIVE vs CBS/MTHFR/MTR/MTRR/AHCY
  MMA: NORMAL — propionate arm entirely intact
  MeCbl/AdoCbl: NORMAL — cobalamin metabolism unaffected
  Dimethylglycine (DMG): NORMAL — DMGDH intact; SARDH is one step downstream of DMGDH
  5,10-methyleneTHF: may be mildly reduced (SARDH product not made → less folate cycle
    intermediate) — theoretically could impair serine/thymidylate synthesis; clinically
    silent in most patients

FOLATE INTERACTION (SARDH-SPECIFIC PARADOX):
  High folate supplementation → increased folate pool → GNMT runs faster (folate
  de-represses GNMT to some extent when SAM is high) → MORE sarcosine produced by GNMT →
  accumulates when SARDH is absent. AVOID excessive folate supplementation.
  High-methionine diet: methionine → SAM → GNMT → more sarcosine → accumulates.

SARDH IS MOSTLY BENIGN:
  Majority of SARDH-deficient patients (detected by NBS or incidentally) are ASYMPTOMATIC.
  Historical reports of IDD association likely reflect ascertainment bias — only symptomatic
  patients were diagnosed in the pre-NBS era. Modern NBS series confirm predominance of
  asymptomatic phenotype.
  Seizures, when present, are mild-moderate; not drug-resistant in most cases.
  Liver disease: ABSENT (unlike GNMT where SAM toxicity drives prominent liver disease).
  Myopathy: ABSENT (unlike AHCY 85-90%).

SARDH vs GNMT — CRITICAL DISTINCTIONS:
  Feature              SARDH deficiency         GNMT deficiency
  Sarcosine            MARKEDLY ELEVATED        ABSENT
  SAM                  NORMAL                   MARKEDLY ELEVATED (800-2000+)
  Methionine           NORMAL                   HIGH (40-500)
  tHcy                 NORMAL                   NORMAL (shared KEY NEGATIVE vs CBS)
  Liver disease        ABSENT                   60-80% (SAM toxicity)
  Overall prognosis    MOSTLY BENIGN            Moderate-severe liver disease

INHERITANCE AND GENE:
  Gene: SARDH, 9q34.3, 972 amino acids
  Mitochondrial enzyme, homodimeric
  FAD-dependent (oxidative demethylation) + THF-dependent (1C acceptor)
  Autosomal Recessive — biallelic LOF required for classic phenotype
  ~50–100 patients reported worldwide (2026) — ultrarare
  Heterozygotes: mildly elevated sarcosine on NBS; asymptomatic

OMIM:
  Gene: *604455 (SARDH)
  Disease: #268900 (Sarcosinemia)
"""

import random
import math

_SEED = 131
_rng = random.Random(_SEED)

# ---------------------------------------------------------------------------
# Phenotypic classes
# ---------------------------------------------------------------------------
PHENOTYPE_CLASSES = [
    "Asymptomatic-Incidental (NBS detection only / no neurological or systemic disease / MODAL in modern NBS era)",
    "Mild-Neurodevelopmental (mild IDD IQ 65-80 / attention-behavioral / possible mild seizures 20-30% / MODAL pre-NBS era)",
    "Classic-Symptomatic (moderate IDD / seizures 50-60% / behavioral / social-communication deficits)",
]

PHENO_WEIGHTS = [0.50, 0.35, 0.15]  # 50% asymptomatic / 35% mild-neurodev / 15% classic

VARIANTS = [
    "p.Arg262Gln",    # catalytic domain; most common worldwide; ~30%; mild-moderate
    "p.Gly449Ser",    # FAD-binding beta-barrel; ~20%; moderate-severe
    "p.Arg374His",    # active site adjacent; ~15%; moderate
    "p.Val208Ile",    # substrate-binding; ~15%; mild (attenuated)
    "p.Cys338Arg",    # disulfide-equivalent stability; ~12%; moderate
    "c.IVS8+1G>A",   # splice-null; ~8%; severe
]

VARIANT_WEIGHTS = [0.30, 0.20, 0.15, 0.15, 0.12, 0.08]

SEIZURE_TYPES = [
    "Focal cortical (partial — sarcosine-mediated cortical excitability perturbation; 50%)",
    "Absence (mild sarcosine neurotoxicity; 20%)",
    "Febrile-associated (metabolic stress + high sarcosine load; 20%)",
    "Myoclonic (Classic-Symptomatic phenotype only; 10%)",
]


def _choose_pheno():
    r = _rng.random()
    cum = 0.0
    for ph, w in zip(PHENOTYPE_CLASSES, PHENO_WEIGHTS):
        cum += w
        if r < cum:
            return ph
    return PHENOTYPE_CLASSES[-1]


def _choose_variant(pheno):
    is_asymp   = "Asymptomatic" in pheno
    is_classic = "Classic"      in pheno
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            # Steer severe variants away from asymptomatic
            if is_asymp and v in ("p.Gly449Ser", "c.IVS8+1G>A"):
                return _rng.choice(["p.Arg262Gln", "p.Val208Ile"])
            # Steer severe variants toward classic
            if is_classic and v in ("p.Val208Ile",):
                return _rng.choice(["p.Gly449Ser", "p.Arg374His", "c.IVS8+1G>A"])
            return v
    return VARIANTS[0]


def _make_patient(idx):
    pheno       = _choose_pheno()
    is_asymp    = "Asymptomatic" in pheno
    is_mild     = "Mild"         in pheno
    is_classic  = "Classic"      in pheno

    # Sarcosine — MARKEDLY ELEVATED; magnitude scales with phenotype severity
    if is_classic:
        sarcosine_umol_l = round(_rng.uniform(350, 800), 1)
        glycine_umol_l   = round(_rng.uniform(350, 500), 1)   # mildly elevated
    elif is_mild:
        sarcosine_umol_l = round(_rng.uniform(150, 400), 1)
        glycine_umol_l   = round(_rng.uniform(280, 420), 1)
    else:  # asymptomatic
        sarcosine_umol_l = round(_rng.uniform(50,  200), 1)
        glycine_umol_l   = round(_rng.uniform(230, 350), 1)

    # All patients: methionine, tHcy, SAM, SAH — ALL NORMAL
    methionine_umol_l = round(_rng.uniform(25,  55),  1)   # NORMAL (<60)
    thcy_umol_l       = round(_rng.uniform(5,   13),  1)   # NORMAL (<15)
    sam_umol_l        = round(_rng.uniform(50,  95),  1)   # NORMAL (<100)
    sah_umol_l        = round(_rng.uniform(8,   22),  1)   # NORMAL (8-30)
    sam_sah_ratio     = round(sam_umol_l / max(sah_umol_l, 1.0), 1)  # NORMAL (2-4)

    # MMA / cobalamin — always normal
    mma_normal   = True
    mecbl_normal = True
    adocbl_normal = True

    # Clinical features
    idd      = _rng.random() < (0.75 if is_classic else (0.40 if is_mild else 0.0))
    seizures = _rng.random() < (0.55 if is_classic else (0.25 if is_mild else 0.0))
    behavioral = _rng.random() < (0.70 if is_classic else (0.55 if is_mild else 0.05))
    attention  = _rng.random() < (0.60 if is_classic else (0.50 if is_mild else 0.05))
    nbs_detected = _rng.random() < (0.90 if is_asymp else (0.65 if is_mild else 0.45))

    # Absent features (key negatives)
    liver_disease  = False
    myopathy       = False
    cardiomyopathy = False
    ectopia_lentis = False

    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    age_onset_mo = (
        _rng.randint(12, 60) if is_classic else
        _rng.randint(18, 84) if is_mild    else
        0  # asymptomatic: NBS-detected, no clinical onset
    )

    variant = _choose_variant(pheno)

    return {
        "id":                    f"SARDH-{idx:03d}",
        "phenotype":             pheno,
        "variant":               variant,
        "age_onset_months":      age_onset_mo,
        # Key biomarker — ELEVATED (pathognomonic)
        "sarcosine_umol_l":      sarcosine_umol_l,
        # Biomarkers — all NORMAL (key negatives)
        "glycine_umol_l":        glycine_umol_l,
        "methionine_umol_l":     methionine_umol_l,
        "homocysteine_umol_l":   thcy_umol_l,
        "sam_umol_l":            sam_umol_l,
        "sah_umol_l":            sah_umol_l,
        "sam_sah_ratio":         sam_sah_ratio,
        "mma_normal":            mma_normal,
        "mecbl_normal":          mecbl_normal,
        "adocbl_normal":         adocbl_normal,
        # Clinical
        "idd":                   idd,
        "seizures":              seizures,
        "seizure_type":          seizure_type,
        "behavioral":            behavioral,
        "attention":             attention,
        "nbs_detected":          nbs_detected,
        "liver_disease":         liver_disease,
        "myopathy":              myopathy,
        "cardiomyopathy":        cardiomyopathy,
        "ectopia_lentis":        ectopia_lentis,
    }


_PATIENTS = [_make_patient(i + 1) for i in range(40)]


# ---------------------------------------------------------------------------
# API response functions
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    n = len(_PATIENTS)

    def avg(key):
        vals = [p[key] for p in _PATIENTS if isinstance(p.get(key), (int, float))]
        return round(sum(vals) / len(vals), 1) if vals else 0

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    pheno_dist = {}
    for p in _PATIENTS:
        ph = p["phenotype"].split(" (")[0]
        pheno_dist[ph] = pheno_dist.get(ph, 0) + 1

    return {
        "dashboard_id": "sardh",
        "title": "SARDH Epilepsy Dashboard",
        "subtitle": (
            "Sarcosine Dehydrogenase Deficiency — Sarcosinemia / "
            "SARDH-972aa-FAD+THF-Dependent-Mitochondrial-Homodimer-9q34.3-AR / "
            "MOSTLY BENIGN — Sarcosine ELEVATED (METABOLIC OPPOSITE of GNMT where Sarcosine ABSENT)"
        ),
        "gene": "SARDH",
        "disease_name": (
            "Sarcosinemia (Sarcosine Dehydrogenase Deficiency)"
        ),
        "chromosome": "9q34.3",
        "inheritance": "Autosomal Recessive — biallelic LOF; heterozygotes mildly elevated sarcosine; asymptomatic",
        "omim_gene":    "OMIM *604455",
        "omim_disease": "OMIM #268900",
        "protein_size": (
            "972 aa; FAD-dependent (oxidative demethylation) + THF-dependent (1C acceptor); "
            "mitochondrial targeting sequence; homodimeric; "
            "catalyses: Sarcosine + THF → Glycine + 5,10-methyleneTHF; "
            "METABOLIC OPPOSITE of GNMT which makes sarcosine; "
            "SARDH deficiency → sarcosine ELEVATED (GNMT deficiency → sarcosine ABSENT)"
        ),
        "prevalence": "~50–100 patients reported worldwide (2026); ultrarare; majority asymptomatic",
        "cohort_n": n,
        "phenotype_distribution": pheno_dist,
        "function": (
            "SARDH (Sarcosine Dehydrogenase) catalyses: Sarcosine (N-methylglycine) + THF → Glycine + 5,10-methyleneTHF. "
            "The enzyme is FAD-dependent: FAD accepts the electron from the N-methyl oxidative demethylation. "
            "THF acts as the 1-carbon acceptor — capturing the methyl group to form 5,10-methyleneTHF, "
            "a key folate cycle intermediate used for serine biosynthesis and thymidylate synthesis. "
            "SARDH is the second step in the sarcosine degradation pathway: "
            "DMG (dimethylglycine) → Sarcosine [via DMGDH] → Glycine [via SARDH]. "
            "SARDH is the METABOLIC OPPOSITE of GNMT: "
            "GNMT (9q22.2) makes sarcosine (SAM + Glycine → SAH + Sarcosine); "
            "SARDH destroys sarcosine (Sarcosine + THF → Glycine + 5,10-mTHF). "
            "SARDH deficiency: sarcosine CANNOT be cleared → MARKEDLY ELEVATED (50–800 µmol/L). "
            "GNMT deficiency: sarcosine CANNOT be made → ABSENT. "
            "This mirror-image sarcosine finding is the single fastest discriminating test."
        ),
        "mechanism": (
            "SARDH LOF → sarcosine (N-methylglycine) cannot be catabolised → "
            "sarcosine MARKEDLY ACCUMULATES in plasma (50–800 µmol/L; normal <10) and urine (sarcosinuria). "
            "Glycine: NORMAL or mildly elevated — mild compensation from incomplete Glycine → Sarcosine cycling; "
            "not at NKH levels (NKH-GLDC/AMT have CSF glycine markedly elevated with CSF/plasma ratio >0.08). "
            "Methionine: NORMAL — SARDH deficiency does NOT affect the methionine/SAM cycle; "
            "KEY NEGATIVE vs GNMT (methionine HIGH), MAT1A (methionine VERY HIGH), AHCY (methionine HIGH). "
            "SAM: NORMAL — SARDH deficiency is entirely downstream of SAM production; "
            "KEY NEGATIVE vs GNMT (SAM MASSIVELY ELEVATED 800–2000+). "
            "tHcy: NORMAL — homocysteine remethylation (MTR/MTRR) and transsulfuration (CBS) are intact; "
            "KEY NEGATIVE vs CBS (tHcy 100–500), MTHFR (50–300), cblE/cblG (40–200). "
            "5,10-methyleneTHF: may be mildly reduced (SARDH product not made) — "
            "theoretically could impair serine/thymidylate synthesis; clinically silent in most. "
            "Folate paradox: high folate → GNMT runs faster → more sarcosine produced → "
            "accumulates when SARDH absent; AVOID excessive folate supplementation. "
            "MOSTLY BENIGN: majority asymptomatic; IDD association was ascertainment bias. "
            "Liver disease ABSENT (unlike GNMT where SAM toxicity → NASH/cirrhosis 60–80%). "
            "Myopathy ABSENT (unlike AHCY 85–90%). No cardiomyopathy. No ectopia lentis."
        ),
        "key_positive_features": (
            "KEY POSITIVES (unique to SARDH among sarcosine/methylation disorders): "
            "(1) Sarcosine MARKEDLY ELEVATED (50–800 µmol/L) — pathognomonic; primary diagnostic marker. "
            "    Urine sarcosinuria — spills at high plasma concentrations. "
            "(2) Methionine NORMAL (<60 µmol/L) — KEY NEGATIVE vs GNMT/MAT1A/AHCY/CBS. "
            "(3) SAM NORMAL (<100 µmol/L) — KEY NEGATIVE vs GNMT (MARKEDLY ELEVATED). "
            "(4) SAM/SAH ratio NORMAL (2–4) — KEY NEGATIVE vs GNMT (>10–25) and AHCY (<0.5). "
            "(5) tHcy NORMAL (<15 µmol/L) — KEY NEGATIVE vs CBS/MTHFR/MTR/MTRR/AHCY. "
            "(6) MMA NORMAL — propionate arm intact; KEY NEGATIVE vs all MMA disorders. "
            "(7) MOSTLY BENIGN phenotype — majority asymptomatic on modern NBS. "
            "KEY NEGATIVES (absent in SARDH): "
            "(1) Liver disease ABSENT — no SAM toxicity (unlike GNMT 60–80%). "
            "(2) Myopathy ABSENT — unlike AHCY 85–90%. "
            "(3) Cardiomyopathy ABSENT — unlike AHCY 60–70%. "
            "(4) Ectopia lentis ABSENT — unlike CBS 90%. "
            "(5) SAM elevation ABSENT — key distinction from GNMT. "
            "(6) NKH-level glycine elevation ABSENT — glycine mildly elevated at most."
        ),
        "nbs_primary": (
            "Amino acid panel — sarcosine (N-methylglycine) elevation (>10 µmol/L; normal <10). "
            "~70–80% of SARDH cases detectable on expanded NBS with amino acid panel including sarcosine. "
            "Standard NBS methionine: NORMAL — SARDH is INVISIBLE on methionine-based NBS. "
            "This contrasts with GNMT (methionine elevated) and MAT1A (methionine very elevated). "
            "Sarcosine screening requires dedicated amino acid panel — not all NBS programmes include it. "
            "Heterozygotes: sarcosine mildly elevated (10–25 µmol/L) — borderline on NBS; asymptomatic. "
            "KEY NBS NEGATIVE: methionine NORMAL distinguishes from GNMT, MAT1A, AHCY, CBS."
        ),
        "nbs_secondary": (
            "Plasma sarcosine: MARKEDLY ELEVATED (50–800 µmol/L; normal <10) — pathognomonic. "
            "Urine sarcosine (sarcosinuria): ELEVATED — spills when plasma >100 µmol/L. "
            "Plasma methionine: NORMAL (<60 µmol/L) — critical KEY NEGATIVE. "
            "SAM/SAH by HPLC: NORMAL ratio (2–4) — critical KEY NEGATIVE vs GNMT (>10–25) / AHCY (<0.5). "
            "SAM: NORMAL (<100 µmol/L) — distinguishes from GNMT immediately. "
            "SAH: NORMAL — distinguishes from AHCY immediately. "
            "Total homocysteine: NORMAL (<15 µmol/L) — distinguishes from CBS/MTHFR/cblE/cblG. "
            "Glycine: NORMAL or mildly elevated (rarely >500 µmol/L; CSF glycine NORMAL). "
            "MMA: NORMAL — propionate arm intact. "
            "Dimethylglycine (DMG): NORMAL — DMGDH intact; one step upstream of SARDH. "
            "Cobalamin (MeCbl, AdoCbl): NORMAL. "
            "Folate: NORMAL or mildly elevated serum (THF not efficiently consumed by SARDH). "
            "Liver function: NORMAL (no SAM hepatotoxicity). "
            "Brain MRI: typically NORMAL. "
            "SARDH enzyme activity in fibroblasts/liver: absent or severely reduced. "
            "Genetic confirmation: SARDH sequencing (9q34.3)."
        ),
        "kpis": {
            "avg_sarcosine_umol_l":    avg("sarcosine_umol_l"),
            "avg_glycine_umol_l":      avg("glycine_umol_l"),
            "avg_methionine_umol_l":   avg("methionine_umol_l"),
            "avg_homocysteine_umol_l": avg("homocysteine_umol_l"),
            "avg_sam_umol_l":          avg("sam_umol_l"),
            "avg_sah_umol_l":          avg("sah_umol_l"),
            "avg_sam_sah_ratio":       avg("sam_sah_ratio"),
            "pct_seizures":            pct(lambda p: p["seizures"]),
            "pct_idd":                 pct(lambda p: p["idd"]),
            "pct_nbs_detected":        pct(lambda p: p["nbs_detected"]),
            "pct_asymptomatic":        pct(lambda p: "Asymptomatic" in p["phenotype"]),
            "pct_liver_disease":       0,   # ABSENT — SARDH has NO liver disease
            "pct_myopathy":            0,   # ABSENT
            "pct_methionine_normal":   100, # ALL patients — methionine is NORMAL
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Focal cortical (partial — sarcosine-mediated cortical excitability perturbation)", "pct": 50},
        {"type": "Absence (mild sarcosine neurotoxicity)", "pct": 20},
        {"type": "Febrile-associated (metabolic stress + high sarcosine load)", "pct": 20},
        {"type": "Myoclonic (Classic-Symptomatic phenotype only)", "pct": 10},
    ]

    metabolic_triggers = [
        {
            "trigger": "High-methionine / high-protein diet",
            "pct": 55,
            "mechanism": (
                "Dietary methionine → MAT1A → SAM → GNMT converts SAM + Glycine → SAH + Sarcosine. "
                "More substrate for GNMT → more sarcosine produced → CANNOT be cleared when SARDH absent. "
                "Low-methionine diet reduces substrate entering the GNMT reaction → less sarcosine generated. "
                "This is the primary rationale for dietary methionine moderation in SARDH deficiency."
            ),
        },
        {
            "trigger": "Folate supplementation paradox (excessive folate intake)",
            "pct": 35,
            "mechanism": (
                "High folate → high 5-methylTHF → de-represses GNMT in some contexts → "
                "GNMT produces more sarcosine from glycine + SAM → sarcosine accumulates when SARDH absent. "
                "This is a SARDH-specific paradox: folate supplements that seem benign in other disorders "
                "can worsen sarcosine load in SARDH deficiency. Avoid high-dose folate (>400 µg/day) "
                "unless medically indicated. Monitor sarcosine levels if folate dose changes."
            ),
        },
        {
            "trigger": "Fasting / catabolic illness / protein catabolism",
            "pct": 30,
            "mechanism": (
                "Protein catabolism releases amino acids including methionine and glycine → "
                "increased substrate for GNMT → more sarcosine production when body is under stress. "
                "Also a general metabolic stressor that can worsen any IEM. "
                "Emergency: maintain caloric intake, avoid prolonged fasting. "
                "No acute metabolic crisis expected (SARDH is not an acute-crisis disorder)."
            ),
        },
    ]

    treatments = [
        {
            "name": "Observation-Only (Asymptomatic)",
            "level": "Level A — Most Cases",
            "note": (
                "For the majority of SARDH-deficient patients who are asymptomatic (detected by NBS), "
                "observation without dietary intervention is Level A recommendation. "
                "Sarcosine elevation alone does not require treatment when there are no symptoms. "
                "Follow-up NBS + developmental surveillance at 6, 12, 24 months."
            ),
        },
        {
            "name": "Low-Methionine Diet",
            "level": "Level B — Symptomatic",
            "note": (
                "Reduces dietary methionine input → less substrate for GNMT → less sarcosine production. "
                "Use only when symptomatic (IDD, behavioral, seizures) AND sarcosine >400 µmol/L. "
                "Mild clinical benefit reported in case series. Does not cure (SARDH still absent). "
                "Monitor plasma sarcosine and nutritional adequacy (methionine is essential amino acid)."
            ),
        },
        {
            "name": "Folate Monitoring (Avoid Excess)",
            "level": "Level B — All Patients",
            "note": (
                "Avoid high-dose folate supplementation (>400 µg/day) — folate paradox: "
                "excess folate may worsen sarcosine load by driving GNMT activity. "
                "Routine folate levels should be maintained in NORMAL range. "
                "If NTD prophylaxis required: use standard dose (400 µg/day) only; discuss with metabolic team."
            ),
        },
        {
            "name": "LEV (Levetiracetam)",
            "level": "Level B — First-Line AED for Seizures",
            "note": (
                "First-line AED when seizures present. No hepatotoxicity, no interaction with "
                "SAM/folate metabolism, no carnitine depletion. "
                "Seizures in SARDH are typically mild-moderate and respond to AED therapy. "
                "Drug-resistant epilepsy is rare (unlike GAMT 60–80% — SARDH is mostly benign)."
            ),
        },
        {
            "name": "VPA (Valproate)",
            "level": "CAUTION — Not Absolute CI",
            "note": (
                "VPA is not absolutely contraindicated in SARDH (no direct MMA effect; no hepatic SAM toxicity). "
                "However: minor carnitine depletion risk; monitor carnitine levels. "
                "No pre-existing liver disease to worsen (unlike GNMT). "
                "LEV preferred as first-line to avoid unnecessary monitoring burden."
            ),
        },
        {
            "name": "SAM Supplements (SAMe)",
            "level": "ABSOLUTE CONTRAINDICATION ⛔",
            "note": (
                "SAM → GNMT → Sarcosine: SAM supplementation directly increases sarcosine production "
                "via GNMT. In SARDH deficiency, sarcosine cannot be cleared → catastrophic sarcosine surge. "
                "Absolute contraindication. OPPOSITE of MAT1A deficiency where SAM is Level A treatment. "
                "This is the SARDH-specific absolute CI (analogous to SAM CI in GNMT)."
            ),
        },
        {
            "name": "Betaine / TMG",
            "level": "HIGH RISK",
            "note": (
                "BHMT: Hcy + Betaine → Methionine → SAM → GNMT → Sarcosine. "
                "Betaine supplementation increases methionine and SAM → more sarcosine via GNMT → "
                "accumulates when SARDH absent. No benefit in SARDH (tHcy is NORMAL). "
                "Avoid unless compelling indication with close monitoring."
            ),
        },
    ]

    drug_risks = [
        {
            "drug": "SAMe (S-Adenosylmethionine)",
            "risk": "ABSOLUTE CONTRAINDICATION ⛔",
            "mechanism": "SAM → GNMT → Sarcosine: direct sarcosine surge when SARDH absent. SARDH cannot clear it.",
        },
        {
            "drug": "Betaine (TMG)",
            "risk": "HIGH RISK",
            "mechanism": "BHMT → methionine → SAM → GNMT → sarcosine. No tHcy benefit (tHcy already normal). Avoid.",
        },
        {
            "drug": "High-dose Folate (>400 µg/day)",
            "risk": "MODERATE RISK — Folate Paradox",
            "mechanism": "Excess folate may drive GNMT → more sarcosine produced → accumulates. Use standard doses only.",
        },
        {
            "drug": "Valproate (VPA)",
            "risk": "CAUTION — Monitor Carnitine",
            "mechanism": "Minor carnitine depletion risk. No pre-existing liver disease in SARDH. LEV preferred.",
        },
        {
            "drug": "High-protein / methionine-enriched supplements",
            "risk": "MODERATE RISK",
            "mechanism": "Methionine → SAM → GNMT → sarcosine load. Avoid methionine-enriched sports nutrition.",
        },
        {
            "drug": "Metformin",
            "risk": "LOW RISK — Standard Monitoring",
            "mechanism": "No direct SARDH metabolic interaction. Lactic acid risk in mitochondrial disorders — "
                         "monitor if mitochondrial function concerns arise (SARDH is mitochondrial).",
        },
    ]

    # Variant breakdown
    variant_counts: dict[str, int] = {}
    for p in _PATIENTS:
        v = p["variant"]
        variant_counts[v] = variant_counts.get(v, 0) + 1

    variants_list = [
        {
            "variant": v,
            "count":   variant_counts.get(v, 0),
            "pct":     round(variant_counts.get(v, 0) / n * 100),
        }
        for v in VARIANTS
    ]

    # SARDH vs GNMT comparison
    phenotype_comparison = {
        "sardh_vs_gnmt": [
            {"feature": "Sarcosine", "sardh": "MARKEDLY ELEVATED (50–800 µmol/L) — PATHOGNOMONIC", "gnmt": "ABSENT (cannot be made)"},
            {"feature": "Methionine", "sardh": "NORMAL (<60 µmol/L) — KEY NEGATIVE", "gnmt": "HIGH (40–500 µmol/L)"},
            {"feature": "SAM", "sardh": "NORMAL (<100 µmol/L) — KEY NEGATIVE", "gnmt": "MARKEDLY ELEVATED (800–2000+)"},
            {"feature": "SAH", "sardh": "NORMAL (8–22 µmol/L)", "gnmt": "LOW/NORMAL (5–30 µmol/L)"},
            {"feature": "SAM/SAH Ratio", "sardh": "NORMAL (2–4)", "gnmt": "MARKEDLY ELEVATED (>10–25) PATHOGNOMONIC"},
            {"feature": "tHcy", "sardh": "NORMAL (<15) — shared KEY NEGATIVE vs CBS", "gnmt": "NORMAL (<20) — shared KEY NEGATIVE vs CBS"},
            {"feature": "Liver Disease", "sardh": "ABSENT", "gnmt": "PROMINENT 60–80% (SAM toxicity → NASH/cirrhosis)"},
            {"feature": "Overall Prognosis", "sardh": "MOSTLY BENIGN (50% asymptomatic)", "gnmt": "Moderate-severe (liver disease dominant)"},
            {"feature": "SAM Supplement", "sardh": "ABSOLUTE CI (worsens sarcosine)", "gnmt": "ABSOLUTE CI (worsens SAM)"},
            {"feature": "Betaine", "sardh": "HIGH RISK (→ more sarcosine)", "gnmt": "ABSOLUTE CI (→ more SAM)"},
        ]
    }

    # Patient sample (first 12)
    sample = [
        {
            "id":            p["id"],
            "phenotype":     p["phenotype"].split(" (")[0],
            "variant":       p["variant"],
            "sarcosine":     p["sarcosine_umol_l"],
            "glycine":       p["glycine_umol_l"],
            "methionine":    p["methionine_umol_l"],
            "sam":           p["sam_umol_l"],
            "sam_sah_ratio": p["sam_sah_ratio"],
            "homocysteine":  p["homocysteine_umol_l"],
            "idd":           p["idd"],
            "seizures":      p["seizures"],
            "nbs":           p["nbs_detected"],
        }
        for p in _PATIENTS[:12]
    ]

    return {
        "dashboard_id":       "sardh",
        "cohort_n":           n,
        "kpi_pcts": {
            "pct_asymptomatic": pct(lambda p: "Asymptomatic" in p["phenotype"]),
            "pct_mild_neurodev": pct(lambda p: "Mild" in p["phenotype"]),
            "pct_classic":      pct(lambda p: "Classic" in p["phenotype"]),
            "pct_idd":          pct(lambda p: p["idd"]),
            "pct_seizures":     pct(lambda p: p["seizures"]),
            "pct_behavioral":   pct(lambda p: p["behavioral"]),
            "pct_attention":    pct(lambda p: p["attention"]),
            "pct_nbs":          pct(lambda p: p["nbs_detected"]),
            "pct_liver_disease": 0,
            "pct_myopathy":     0,
        },
        "phenotype_dist": [
            {"label": "Asymptomatic-Incidental", "pct": 50},
            {"label": "Mild-Neurodevelopmental",  "pct": 35},
            {"label": "Classic-Symptomatic",      "pct": 15},
        ],
        "seizure_types":       seizure_types,
        "metabolic_triggers":  metabolic_triggers,
        "treatments":          treatments,
        "drug_risks":          drug_risks,
        "variants":            variants_list,
        "patient_sample":      sample,
        "phenotype_comparison": phenotype_comparison,
        "variant_distribution": {
            "most_common":  "p.Arg262Gln (~30%) — catalytic domain — mild-moderate",
            "most_severe":  "c.IVS8+1G>A (~8%) — splice-null — severe phenotype",
            "most_attenuated": "p.Val208Ile (~15%) — substrate-binding — mild/attenuated",
        },
        "treatment_matrix": {
            "asymptomatic": "Observation-only (Level A); no dietary intervention; developmental surveillance",
            "mild_neurodev": "Low-methionine diet Level B if sarcosine >400; LEV if seizures; folate monitoring",
            "classic":      "Low-methionine diet Level B; LEV Level B; folate restrict; behavioral support",
        },
        "biomarker_ranges": {
            "sarcosine_asymp_umol_l":   "50–200 (ELEVATED; normal <10)",
            "sarcosine_mild_umol_l":    "150–400 (ELEVATED)",
            "sarcosine_classic_umol_l": "350–800 (MARKEDLY ELEVATED; pathognomonic)",
            "glycine_umol_l":           "230–500 (NORMAL to mildly elevated)",
            "methionine_umol_l":        "25–55 (NORMAL — KEY NEGATIVE)",
            "homocysteine_umol_l":      "5–13 (NORMAL — KEY NEGATIVE vs CBS/MTHFR)",
            "sam_umol_l":               "50–95 (NORMAL — KEY NEGATIVE vs GNMT)",
            "sah_umol_l":               "8–22 (NORMAL — KEY NEGATIVE vs AHCY)",
            "sam_sah_ratio":            "2–4 (NORMAL — KEY NEGATIVE vs GNMT >10 / AHCY <0.5)",
            "mma":                      "NORMAL (all patients)",
        },
    }


def get_definitions() -> dict:
    return {
        "dashboard_id": "sardh",
        "gene_card": {
            "gene":           "SARDH",
            "full_name":      "Sarcosine Dehydrogenase",
            "chromosome":     "9q34.3",
            "size_aa":        972,
            "structure": (
                "Homodimeric; mitochondrial (mitochondrial targeting sequence at N-terminus). "
                "FAD-dependent (electron acceptor for oxidative N-demethylation). "
                "THF-dependent (tetrahydrofolate as 1-carbon acceptor → 5,10-methyleneTHF). "
                "METABOLIC OPPOSITE of GNMT: GNMT makes sarcosine; SARDH destroys sarcosine. "
                "One step downstream of DMGDH (dimethylglycine dehydrogenase) in the "
                "choline → betaine → DMG → sarcosine → glycine degradation pathway."
            ),
            "omim_gene":      "*604455",
            "omim_disease":   "#268900",
            "inheritance":    "Autosomal Recessive",
            "pathway_position": "Choline → Betaine → DMG [DMGDH] → Sarcosine [SARDH] → Glycine",
        },
        "key_concepts": [
            {
                "term": "SARDH as Metabolic Opposite of GNMT",
                "definition": (
                    "GNMT (Glycine N-methyltransferase, 9q22.2) MAKES sarcosine: "
                    "SAM + Glycine → SAH + Sarcosine. "
                    "SARDH (Sarcosine Dehydrogenase, 9q34.3) DESTROYS sarcosine: "
                    "Sarcosine + THF → Glycine + 5,10-methyleneTHF. "
                    "GNMT deficiency → Sarcosine ABSENT (reaction product not made). "
                    "SARDH deficiency → Sarcosine MARKEDLY ELEVATED (reaction substrate not cleared). "
                    "This mirror-image sarcosine finding is the fastest single metabolomic discriminating test "
                    "between the two disorders — and is completely opposite to intuition at first glance."
                ),
            },
            {
                "term": "FAD + THF Dual Cofactor Dependency",
                "definition": (
                    "SARDH requires TWO cofactors: "
                    "(1) FAD (flavin adenine dinucleotide): accepts electrons during oxidative N-demethylation "
                    "    of the N-methyl group of sarcosine — FAD → FADH2. "
                    "(2) THF (tetrahydrofolate): captures the liberated 1-carbon methyl group → "
                    "    THF + 1C → 5,10-methyleneTHF (a key folate cycle intermediate). "
                    "Consequence of SARDH LOF: 5,10-methyleneTHF is not produced by this reaction → "
                    "theoretically reduced folate cycle flux → potential mild impairment of serine biosynthesis "
                    "and thymidylate synthesis. Clinically silent in most patients."
                ),
            },
            {
                "term": "Mostly Benign Phenotype — Ascertainment Bias in Historical Reports",
                "definition": (
                    "Early case series of SARDH deficiency (1960s–1980s) reported IDD in ~60–70% of cases. "
                    "This reflected severe ascertainment bias: only symptomatic patients reached diagnosis. "
                    "Modern newborn screening (NBS) era reveals that the majority of SARDH-deficient individuals "
                    "are ASYMPTOMATIC — detected incidentally on amino acid panel. "
                    "True prevalence of IDD in SARDH deficiency is now estimated at ~40% in Mild-Neurodevelopmental "
                    "and ~75% in Classic-Symptomatic, but these phenotypes represent only 50% of cases combined. "
                    "Overall IDD risk is ~25–30% of ALL SARDH-deficient individuals. "
                    "SARDH is one of the most benign IEM — comparable to hyperphenylalaninemia (non-PKU)."
                ),
            },
            {
                "term": "Folate Paradox in SARDH Deficiency",
                "definition": (
                    "SARDH uses THF as 1-carbon acceptor. When SARDH is absent, THF is NOT consumed by this reaction → "
                    "THF accumulates slightly → serum folate may be mildly elevated. "
                    "Paradoxically, HIGH FOLATE WORSENS sarcosine load: "
                    "excess 5-methylTHF → methyl group donors → GNMT may run faster (folate de-represses GNMT "
                    "in some contexts) → more sarcosine made → accumulates when SARDH absent. "
                    "Clinical implication: AVOID high-dose folate supplementation (>400 µg/day) in SARDH patients. "
                    "This is SARDH-specific — in most other IEM, folate supplementation is neutral or beneficial."
                ),
            },
            {
                "term": "SAM NORMAL — Critical Distinction from GNMT",
                "definition": (
                    "SARDH deficiency: SAM NORMAL because SARDH is entirely downstream of SAM production. "
                    "The SAM cycle (Methionine → SAM → SAH → Hcy) is COMPLETELY INTACT in SARDH deficiency. "
                    "GNMT deficiency: SAM MASSIVELY ELEVATED (800–2000+ µmol/L) — the safety valve is gone. "
                    "This single biomarker (SAM level) immediately separates the two disorders. "
                    "If SAM is elevated → GNMT deficiency (or AHCY). "
                    "If SAM is normal AND sarcosine is elevated → SARDH deficiency."
                ),
            },
            {
                "term": "Why Liver Disease is ABSENT in SARDH (Unlike GNMT)",
                "definition": (
                    "GNMT liver disease mechanism: SAM MASSIVELY ELEVATED → direct hepatotoxicity → "
                    "protein transmethylation adducts, ROS, steatosis → NASH → cirrhosis (60–80%). "
                    "SARDH has NO effect on SAM levels. SAM remains normal. "
                    "No SAM hepatotoxicity → NO liver disease in SARDH deficiency. "
                    "This is a critical clinical distinction: "
                    "if liver disease is present → GNMT or AHCY (not SARDH). "
                    "Liver disease effectively rules out SARDH and points toward GNMT, AHCY, or MAT1A."
                ),
            },
            {
                "term": "SAM Supplement — Absolute CI in SARDH (Same Reasoning as GNMT, Different Path)",
                "definition": (
                    "In GNMT deficiency: SAM is ALREADY massively elevated → SAM supplement worsens directly. "
                    "In SARDH deficiency: SAM supplement → provides substrate for GNMT → "
                    "GNMT makes MORE sarcosine → sarcosine accumulates when SARDH cannot clear it. "
                    "Both disorders share the same ABSOLUTE CONTRAINDICATION for SAM supplements, "
                    "but for mechanistically different reasons. "
                    "Betaine (TMG) also contraindicated in SARDH: "
                    "BHMT → more methionine → more SAM → more GNMT activity → more sarcosine."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disorder": "GNMT Deficiency (Glycine N-Methyltransferase Deficiency)",
                "key_distinction": (
                    "GNMT: Sarcosine ABSENT (cannot make it) — OPPOSITE of SARDH (elevated). "
                    "GNMT: SAM MARKEDLY ELEVATED (800–2000+); SARDH: SAM NORMAL. "
                    "GNMT: Methionine HIGH (40–500); SARDH: Methionine NORMAL. "
                    "GNMT: Liver disease 60–80% (SAM toxicity); SARDH: Liver disease ABSENT. "
                    "Both share: tHcy NORMAL (CBS/MTR intact) — shared KEY NEGATIVE vs CBS."
                ),
            },
            {
                "disorder": "NKH — Non-Ketotic Hyperglycinemia (GLDC / AMT Deficiency)",
                "key_distinction": (
                    "NKH: Glycine MASSIVELY ELEVATED plasma (>1000 µmol/L) + CSF glycine elevated + "
                    "CSF/plasma glycine ratio >0.08 (pathognomonic). Neonatal encephalopathy + hiccups. "
                    "SARDH: Glycine NORMAL or mildly elevated (<500); CSF glycine NORMAL; "
                    "no neonatal encephalopathy; sarcosine MARKEDLY ELEVATED. "
                    "Sarcosine normal in NKH; elevated sarcosine immediately rules out NKH."
                ),
            },
            {
                "disorder": "AHCY Deficiency (Adenosylhomocysteinase Deficiency)",
                "key_distinction": (
                    "AHCY: SAH MASSIVELY ELEVATED + SAM/SAH ratio VERY LOW (<0.5) + MYOPATHY 85–90%. "
                    "SARDH: SAH NORMAL + SAM/SAH ratio NORMAL (2–4) + NO myopathy. "
                    "AHCY: Methionine HIGH, liver disease 70–75%. "
                    "SARDH: Methionine NORMAL, liver disease ABSENT. "
                    "Sarcosine is NORMAL in AHCY; elevated sarcosine rules out AHCY."
                ),
            },
            {
                "disorder": "CBS Deficiency (Classical Homocystinuria)",
                "key_distinction": (
                    "CBS: tHcy MARKEDLY ELEVATED (100–500) — primary diagnostic marker. "
                    "SARDH: tHcy NORMAL (<15). "
                    "CBS: ectopia lentis 90%, marfanoid habitus, thromboembolism. "
                    "SARDH: none of these features. "
                    "Sarcosine is NORMAL in CBS."
                ),
            },
            {
                "disorder": "Hypersarcosinemia from DMGDH Deficiency",
                "key_distinction": (
                    "DMGDH (dimethylglycine dehydrogenase) is one step UPSTREAM of SARDH. "
                    "DMGDH deficiency → DMG ELEVATED + Sarcosine MILDLY ELEVATED (not as high as SARDH). "
                    "SARDH deficiency → Sarcosine MARKEDLY ELEVATED + DMG NORMAL. "
                    "Differentiate by measuring DMG: ELEVATED in DMGDH; NORMAL in SARDH."
                ),
            },
        ],
        "treatment_summary": {
            "first_line": "Observation-only for asymptomatic (Level A); Low-methionine diet Level B for symptomatic; LEV Level B for seizures",
            "absolute_ci": [
                "SAMe/SAM supplements — drives GNMT → more sarcosine (SARDH cannot clear it)",
            ],
            "high_risk": [
                "Betaine (TMG) — BHMT → methionine → SAM → GNMT → sarcosine",
                "High-dose folate (>400 µg/day) — folate paradox worsens sarcosine load",
                "High-protein / methionine-enriched supplements",
            ],
            "caution": [
                "Valproate — minor carnitine monitoring; no liver disease to worsen; LEV preferred",
            ],
            "prognosis": (
                "MOSTLY BENIGN. 50% asymptomatic. Seizures respond to AED when present. "
                "Drug-resistant epilepsy is rare. No liver disease. No myopathy. "
                "Life expectancy generally normal in asymptomatic patients."
            ),
        },
    }
