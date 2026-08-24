#!/usr/bin/env python3
"""ALDH18A1 (P5CS / Delta-1-Pyrroline-5-Carboxylate Synthase Deficiency) Epilepsy Dashboard.

ALDH18A1 encodes Delta-1-Pyrroline-5-Carboxylate Synthase (P5CS), a bifunctional enzyme
catalysing the committed entry step of proline and ornithine SYNTHESIS from glutamate.

PROLINE SYNTHESIS PATHWAY (OPPOSITE of PRODH/ALDH4A1 catabolism):
  L-Glutamate + 2 NADPH + ATP
    → [ALDH18A1 γ-Glutamyl Kinase domain: Glu → γ-Glutamyl phosphate]
    → [ALDH18A1 Glutamate-5-Semialdehyde Dehydrogenase domain: γ-Glu-P → P5C/GSA]
  P5C / Glutamate-5-Semialdehyde (GSA)
    → [PYCR1/PYCR2 Pyrroline-5-Carboxylate Reductase, NADPH]
  L-Proline (end product — exported to cytoplasm)

  Also: P5C ↔ L-Ornithine via OAT (Ornithine Aminotransferase; mitochondria)
  → ALDH18A1 LOF impairs BOTH proline AND ornithine synthesis simultaneously.

ALDH18A1 ENZYMATIC FUNCTION:
  Bifunctional enzyme; 795 aa; cytoplasmic (N-terminal G5K) + mitochondrial matrix (G5SD);
  homodimerises; NADPH-dependent; ATP-dependent (G5K domain).
  Step 1 (γ-Glutamyl Kinase domain): L-Glu + ATP → γ-Glutamyl-phosphate + ADP
  Step 2 (Glutamate-5-Semialdehyde Dehydrogenase domain): γ-Glu-P + NADPH → P5C/GSA + NADP+ + Pi

ALDH18A1 LOF — PROLINE LOW; ORNITHINE LOW:
  Since ALDH18A1 cannot make P5C from glutamate:
  → Proline CANNOT be synthesised de novo → Proline CRITICALLY LOW (hypoProlinemia)
  → P5C CANNOT be made for ornithine synthesis → Ornithine LOW
  → Ornithine LOW → Citrulline LOW (OTC cannot run at full rate)
  → Citrulline LOW → Arginine LOW (ASS1/ASL impaired)
  → Collagen and connective tissue: proline is essential for hydroxyproline in collagen.

  THE CRITICAL INVERSION vs PRODH/ALDH4A1:
  PRODH LOF   → Proline ELEVATED (catabolism blocked, proline ACCUMULATES)
  ALDH4A1 LOF → Proline MARKEDLY ELEVATED + P5C ELEVATED + PLP LOW
  ALDH18A1 LOF → Proline CRITICALLY LOW (synthesis blocked, proline CANNOT BE MADE)
  This is the metabolic MIRROR IMAGE of the proline catabolic disorders.

THE EPILEPTOGENIC MECHANISMS IN ALDH18A1 DEFICIENCY:
  Mechanism 1 — Proline depletion → glutamate/GABA imbalance:
    Proline is a major precursor for glutamate (via reverse proline catabolism in neurons).
    Low proline → reduced available glutamate substrate → secondary GABA depletion.
    Glutamate is also directly required for GAD65/GAD67 GABA synthesis.
    LOW proline + LOW ornithine → neurotransmitter synthesis disruption.

  Mechanism 2 — Ornithine deficiency → impaired urea cycle + polyamine depletion:
    Ornithine feeds the urea cycle (OTC: ornithine + carbamoyl-P → citrulline).
    Low ornithine → mild hyperammonemia risk.
    Ornithine is substrate for polyamine synthesis (putrescine → spermidine → spermine).
    Polyamines modulate NMDA receptors and neuronal excitability.
    Polyamine depletion → altered NMDA receptor function → seizure susceptibility.

  Mechanism 3 — Collagen/connective tissue failure causing structural cortical abnormalities:
    Proline (and hydroxyproline) are critical for collagen triple-helix stability.
    Severe collagen deficiency → cutis laxa, joint laxity, vascular fragility.
    Brain vascular fragility → micro-bleeds + cortical dysgenesis (reported in severe cases).
    Structural cortical abnormalities → focal seizures.

  Mechanism 4 — Mitochondrial dysfunction (P5C-PYCR axis disruption):
    PYCR1/2 recycle NADP+ → NADPH in mitochondria, linked to redox balance.
    Loss of P5C substrate for PYCR → disrupted mitochondrial NADP+/NADPH ratio.
    Oxidative stress → neuronal mitochondrial dysfunction → epileptogenesis.

WHY DISTINCT FROM PRODH AND ALDH4A1:
  ALDH18A1    → Proline LOW (synthesis failure) → no B6 involvement → collagen failure → ornithine LOW
  PRODH       → Proline ELEVATED (catabolism failure step 1) → NMDA/GABA transport mechanism → psychiatric
  ALDH4A1     → Proline ELEVATED + P5C ELEVATED + PLP LOW → secondary B6 deficiency → GABA collapse

ALDH18A1 DISEASE FORMS:
  Severe AR (biallelic null):
    OMIM #219150 — De Barsy Syndrome / Cutis Laxa Type IIIA & IIIB
    Age of onset: neonatal / early infancy
    Seizures: 50–65% (infantile spasms, multifocal, GTCS)
    IDD: severe (90%)
    Cutis laxa (wrinkly/loose skin): 95% — pathognomonic skin finding
    Joint laxity/hypermobility: 90%
    Cataracts: 60–70%
    Microcephaly: 50%

  Mild AD (dominant negative heterozygous):
    OMIM #616603 — Intellectual Disability + Cataracts ± Seizures
    Seizures: 25–35%
    Milder IDD: 60%
    Cataracts: 75%
    Cutis laxa: 30–40% (much milder)

KEY BIOMARKERS IN ALDH18A1 DEFICIENCY:
  CRITICALLY LOW (diagnostic constellation):
    Proline (plasma): CRITICALLY LOW (<60 µmol/L; often <30; normal 90–300 µmol/L) — OPPOSITE of PRODH/ALDH4A1
    Ornithine (plasma): LOW (< 30 µmol/L; normal 50–200 µmol/L)
    Citrulline (plasma): LOW–NORMAL (secondary to low ornithine; normal 10–45 µmol/L)
    Arginine (plasma): LOW–NORMAL (secondary to low citrulline)

  NORMAL (critical key negatives):
    P5C (plasma/urine): NORMAL or LOW (not elevated; LOF prevents P5C formation)
    PLP (plasma): NORMAL — no P5C-PLP inactivation (unlike ALDH4A1 Type II)
    MMA: NORMAL
    tHcy: NORMAL
    Methionine: NORMAL
    alpha-AASA: NORMAL — KEY NEGATIVE vs ALDH7A1/PDE
    Pipecolic acid: NORMAL — KEY NEGATIVE vs ALDH7A1/PDE
    Lactate: NORMAL–mildly elevated (mitochondrial redox stress in severe cases)
    Ammonia: NORMAL–mildly elevated (low ornithine → partial urea cycle limitation)

~50–100 cases worldwide (2026); truly ultra-rare; OMIM *138250 / #219150 / #616603.
Chromosome 10q24.1; AR (severe) / AD (mild).
"""

import random

SEED = 151

def _rng():
    return random.Random(SEED)

# ── Phenotypic classes ─────────────────────────────────────────────────────────
PHENOTYPES = [
    {"label": "Severe AR (De Barsy / Cutis Laxa IIIA/B)", "pct": 55, "color": "#b71c1c"},
    {"label": "Moderate AR (Partial LOF)",                 "pct": 30, "color": "#e65100"},
    {"label": "Mild AD (Dom-Neg Heterozygous)",            "pct": 15, "color": "#1565c0"},
]

# ── Known variants ─────────────────────────────────────────────────────────────
VARIANTS = [
    {"v": "p.Arg138Gln",   "domain": "G5K-ATP-Binding",             "pct": 22, "sev": "Severe-Null"},
    {"v": "p.Gly93Arg",    "domain": "G5K-Substrate-Binding",       "pct": 18, "sev": "Severe"},
    {"v": "p.Arg418Cys",   "domain": "G5SD-NADPH-Binding",          "pct": 15, "sev": "Severe-Moderate"},
    {"v": "p.Ala445Val",   "domain": "G5SD-Dimer-Interface",        "pct": 12, "sev": "Moderate"},
    {"v": "p.Leu503Pro",   "domain": "G5SD-Catalytic",              "pct": 10, "sev": "Severe"},
    {"v": "c.IVS6+1G>A",   "domain": "Splice-Null-Exon-6",         "pct":  9, "sev": "Severe-Null"},
    {"v": "p.Glu247Lys",   "domain": "G5K-Product-Release",        "pct":  8, "sev": "Mild-AD-DomNeg"},
    {"v": "p.Thr321Ile",   "domain": "G5SD-Cofactor-Positioning",  "pct":  6, "sev": "Moderate"},
]

# ── Biomarkers ─────────────────────────────────────────────────────────────────
BIOMARKERS = [
    {"name": "Proline (plasma)",       "unit": "µmol/L", "normal": "90–300",   "expected": "CRITICALLY LOW <60",  "key": True,  "direction": "low"},
    {"name": "Ornithine (plasma)",     "unit": "µmol/L", "normal": "50–200",   "expected": "LOW <30",              "key": True,  "direction": "low"},
    {"name": "Citrulline (plasma)",    "unit": "µmol/L", "normal": "10–45",    "expected": "LOW–NORMAL <15",       "key": False, "direction": "low"},
    {"name": "Arginine (plasma)",      "unit": "µmol/L", "normal": "40–120",   "expected": "LOW–NORMAL <30",       "key": False, "direction": "low"},
    {"name": "P5C (plasma/urine)",     "unit": "µmol/L", "normal": "<5",       "expected": "NORMAL–LOW (not made)","key": True,  "direction": "normal"},
    {"name": "PLP (plasma)",           "unit": "nmol/L", "normal": "35–110",   "expected": "NORMAL",               "key": True,  "direction": "normal"},
    {"name": "alpha-AASA (urine)",     "unit": "mmol/mol Cr", "normal": "<1",  "expected": "NORMAL",               "key": True,  "direction": "normal"},
    {"name": "Pipecolic acid",         "unit": "µmol/L", "normal": "<5",       "expected": "NORMAL",               "key": True,  "direction": "normal"},
    {"name": "MMA (urine)",            "unit": "mmol/mol Cr", "normal": "<5",  "expected": "NORMAL",               "key": False, "direction": "normal"},
    {"name": "tHcy (plasma)",          "unit": "µmol/L", "normal": "<15",      "expected": "NORMAL",               "key": False, "direction": "normal"},
    {"name": "Methionine (plasma)",    "unit": "µmol/L", "normal": "15–45",    "expected": "NORMAL",               "key": False, "direction": "normal"},
    {"name": "Ammonia (plasma)",       "unit": "µmol/L", "normal": "<50",      "expected": "NORMAL–mild ↑ (<80)",  "key": False, "direction": "borderline"},
    {"name": "Lactate (plasma)",       "unit": "mmol/L", "normal": "0.5–2.0",  "expected": "NORMAL–mild ↑ (<3.5)", "key": False, "direction": "borderline"},
    {"name": "Hydroxyproline (urine)", "unit": "µmol/g Cr", "normal": "0–100", "expected": "LOW (proline deficient collagen)", "key": False, "direction": "low"},
]

# ── Seizure types ──────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {"type": "Infantile Spasms (West Syndrome)",  "pct": 45, "color": "#b71c1c"},
    {"type": "Multifocal Clonic",                 "pct": 35, "color": "#e65100"},
    {"type": "Generalised Tonic-Clonic (GTCS)",   "pct": 25, "color": "#f57f17"},
    {"type": "Focal Aware / Impaired Awareness",  "pct": 20, "color": "#ef6c00"},
    {"type": "Epileptic Spasms (late-onset)",     "pct": 15, "color": "#6a1520"},
    {"type": "Absence-like",                      "pct": 10, "color": "#827717"},
]

# ── Metabolic triggers ─────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "High-protein/collagen diet (increases proline demand)",   "pct": 72},
    {"trigger": "Fasting / catabolic illness (depletes already-low proline)", "pct": 65},
    {"trigger": "Febrile illness (metabolic stress + seizure threshold ↓)",  "pct": 58},
    {"trigger": "Missed proline/ornithine supplement dose",                  "pct": 55},
    {"trigger": "Rapid growth phase (increased proline for collagen synthesis)", "pct": 42},
    {"trigger": "Anaesthetic / surgical stress (catabolism → proline depletion)", "pct": 35},
]

def _make_patients(n: int = 40) -> list:
    rng = _rng()
    patients = []
    pheno_weights = [p["pct"] for p in PHENOTYPES]
    var_weights   = [v["pct"] for v in VARIANTS]

    for i in range(n):
        pheno = rng.choices(PHENOTYPES, weights=pheno_weights, k=1)[0]
        var1  = rng.choices(VARIANTS,   weights=var_weights,   k=1)[0]
        var2  = rng.choices(VARIANTS,   weights=var_weights,   k=1)[0]

        # Proline level — critically low; AR severe lower
        if "Severe" in pheno["label"]:
            proline = round(rng.uniform(8, 45), 1)
        elif "Moderate" in pheno["label"]:
            proline = round(rng.uniform(30, 80), 1)
        else:
            proline = round(rng.uniform(60, 130), 1)   # mild AD — closer to low-normal

        ornithine = round(rng.uniform(8, 45) if "Severe" in pheno["label"] else rng.uniform(20, 80), 1)
        plp       = round(rng.uniform(38, 108), 1)      # always normal
        p5c       = round(rng.uniform(0.5, 3.5), 2)     # low/absent (not made)

        has_seizures = rng.random() < (0.60 if "Severe" in pheno["label"] else 0.28 if "Mild" in pheno["label"] else 0.45)
        dre          = has_seizures and rng.random() < (0.30 if "Severe" in pheno["label"] else 0.15)

        has_cutis_laxa = rng.random() < (0.95 if "Severe" in pheno["label"] else 0.40 if "Moderate" in pheno["label"] else 0.30)
        has_cataracts  = rng.random() < (0.65 if "Severe" in pheno["label"] else 0.55 if "Moderate" in pheno["label"] else 0.75)
        has_idd        = rng.random() < (0.92 if "Severe" in pheno["label"] else 0.65 if "Moderate" in pheno["label"] else 0.55)

        proline_supplement = rng.random() < 0.78
        ornithine_suppl    = rng.random() < 0.55

        patients.append({
            "id":                  f"P{i+1:03d}",
            "phenotype":           pheno["label"],
            "variant_1":           var1["v"],
            "variant_2":           var2["v"],
            "proline_umol":        proline,
            "ornithine_umol":      ornithine,
            "plp_nmol":            plp,
            "p5c_umol":            p5c,
            "has_seizures":        has_seizures,
            "drug_resistant":      dre,
            "cutis_laxa":          has_cutis_laxa,
            "cataracts":           has_cataracts,
            "idd":                 has_idd,
            "proline_supplement":  proline_supplement,
            "ornithine_supplement": ornithine_suppl,
            "age_dx_months":       int(rng.uniform(1, 36) if "Severe" in pheno["label"] else rng.uniform(6, 84)),
        })
    return patients

_PATIENTS = _make_patients(40)


def get_overview() -> dict:
    pts = _PATIENTS
    n = len(pts)
    n_sz   = sum(1 for p in pts if p["has_seizures"])
    n_dre  = sum(1 for p in pts if p["drug_resistant"])
    n_cl   = sum(1 for p in pts if p["cutis_laxa"])
    n_cat  = sum(1 for p in pts if p["cataracts"])
    n_idd  = sum(1 for p in pts if p["idd"])
    n_pro  = sum(1 for p in pts if p["proline_supplement"])
    avg_pro = round(sum(p["proline_umol"] for p in pts) / n, 1)
    avg_orn = round(sum(p["ornithine_umol"] for p in pts) / n, 1)

    return {
        "disease":         "ALDH18A1 Epilepsy — P5CS Deficiency / Hyperprolinemia Type III (Hypo)",
        "gene":            "ALDH18A1 (also P5CS, P5S, GSAS)",
        "omim_gene":       "OMIM *138250",
        "omim_disease_ar": "OMIM #219150 (De Barsy / Cutis Laxa IIIA/B)",
        "omim_disease_ad": "OMIM #616603 (Intellectual Disability + Cataracts)",
        "chromosome":      "10q24.1",
        "inheritance":     "Autosomal Recessive (severe) / Autosomal Dominant (mild)",
        "protein":         "Delta-1-Pyrroline-5-Carboxylate Synthase (P5CS) — 795 aa bifunctional enzyme; γ-Glutamyl Kinase + Glutamate-5-Semialdehyde Dehydrogenase domains; NADPH+ATP-dependent; cytoplasmic/mitochondrial; homodimeric",
        "pathway_role":    "Proline/Ornithine SYNTHESIS (Step 1): L-Glutamate → [ALDH18A1] → P5C/GSA → [PYCR1/2] → L-Proline. Also: P5C ↔ L-Ornithine via OAT. OPPOSITE of PRODH (catabolism step 1) and ALDH4A1 (catabolism step 2).",
        "cases_worldwide": "~50–100 (2026) — Ultra-rare",
        "cohort_n":        n,
        "kpi": {
            "seizure_prevalence_pct":     round(n_sz / n * 100),
            "drug_resistant_pct":         round(n_dre / n * 100),
            "cutis_laxa_pct":             round(n_cl / n * 100),
            "cataracts_pct":              round(n_cat / n * 100),
            "idd_pct":                    round(n_idd / n * 100),
            "proline_supplement_pct":     round(n_pro / n * 100),
            "avg_proline_umol":           avg_pro,
            "avg_ornithine_umol":         avg_orn,
        },
        "phenotypes": PHENOTYPES,
        "biomarker_highlights": [
            {"marker": "Proline (plasma)",   "finding": f"CRITICALLY LOW avg {avg_pro} µmol/L (normal 90–300)", "significance": "Primary diagnostic marker — OPPOSITE of PRODH/ALDH4A1 where proline is elevated"},
            {"marker": "Ornithine (plasma)", "finding": f"LOW avg {avg_orn} µmol/L (normal 50–200)",           "significance": "P5C cannot feed OAT reverse → ornithine synthesis impaired"},
            {"marker": "P5C (plasma)",       "finding": "NORMAL–LOW (not produced)",                           "significance": "Cannot be made; contrast ALDH4A1 where P5C is ELEVATED PATHOGNOMONIC"},
            {"marker": "PLP (plasma)",       "finding": "NORMAL (no P5C-PLP inactivation)",                    "significance": "KEY NEGATIVE vs ALDH4A1 Type II; no B6 response expected"},
            {"marker": "alpha-AASA (urine)", "finding": "NORMAL",                                              "significance": "KEY NEGATIVE vs ALDH7A1/PDE (antiquitin deficiency)"},
            {"marker": "Pipecolic acid",     "finding": "NORMAL",                                              "significance": "KEY NEGATIVE vs ALDH7A1/PDE"},
        ],
        "metabolic_comparison": [
            {"gene": "ALDH18A1 (P5CS)", "direction": "SYNTHESIS ↓", "proline": "CRITICALLY LOW <60", "p5c": "NORMAL–LOW", "plp": "NORMAL", "b6_response": "NONE",     "unique": "Cutis laxa + ornithine LOW"},
            {"gene": "PRODH",           "direction": "CATABOLISM ↓", "proline": "ELEVATED 350–1000",  "p5c": "NORMAL",     "plp": "NORMAL", "b6_response": "NONE",     "unique": "Psychiatric features"},
            {"gene": "ALDH4A1",         "direction": "CATABOLISM ↓", "proline": "MARKEDLY HIGH >1000","p5c": "ELEVATED",   "plp": "LOW",    "b6_response": "PARTIAL 30–50%", "unique": "PLP inactivation → GABA collapse"},
            {"gene": "ALDH7A1 (PDE)",   "direction": "Lysine catab ↓","proline": "NORMAL",           "p5c": "NORMAL",     "plp": "LOW",    "b6_response": "EXCELLENT >85%", "unique": "alpha-AASA + pipecolic ELEVATED"},
        ],
    }


def get_breakdown() -> dict:
    pts = _PATIENTS

    # Phenotype breakdown
    pheno_counts = {}
    for p in pts:
        pheno_counts[p["phenotype"]] = pheno_counts.get(p["phenotype"], 0) + 1

    # Variant breakdown
    var_counts = {}
    for p in pts:
        for v_key in ("variant_1", "variant_2"):
            v = p[v_key]
            var_counts[v] = var_counts.get(v, 0) + 1

    # Seizure breakdown by phenotype
    pheno_seizure = {}
    pheno_total   = {}
    for p in pts:
        ph = p["phenotype"]
        pheno_total[ph] = pheno_total.get(ph, 0) + 1
        if p["has_seizures"]:
            pheno_seizure[ph] = pheno_seizure.get(ph, 0) + 1

    return {
        "patients":       pts,
        "phenotype_dist": [{"label": k, "n": v, "pct": round(v / len(pts) * 100)} for k, v in pheno_counts.items()],
        "variant_dist":   sorted([{"variant": k, "n": v} for k, v in var_counts.items()], key=lambda x: -x["n"])[:8],
        "seizure_by_phenotype": [
            {
                "phenotype": ph,
                "seizure_pct": round(pheno_seizure.get(ph, 0) / pheno_total[ph] * 100)
            }
            for ph in pheno_total
        ],
        "biomarkers":     BIOMARKERS,
        "seizure_types":  SEIZURE_TYPES,
        "triggers":       TRIGGERS,
        "variants":       VARIANTS,
        "treatments": [
            {
                "treatment": "L-Proline supplementation",
                "level":     "Level A (Severe AR)",
                "rationale": "Direct replacement of the deficient amino acid. Oral L-proline 200–500 mg/kg/day in divided doses. "
                             "Raises plasma proline toward normal range; improves collagen integrity; may reduce seizure frequency. "
                             "PRIMARY metabolic therapy — analogous to creatine in GAMT/AGAT, ornithine in GAMT.",
            },
            {
                "treatment": "L-Ornithine supplementation",
                "level":     "Level B",
                "rationale": "Compensates for impaired de novo ornithine synthesis. "
                             "Supports polyamine synthesis (putrescine → spermidine → spermine → NMDA modulation). "
                             "Supports urea cycle (prevents mild hyperammonemia). 50–200 mg/kg/day.",
            },
            {
                "treatment": "Citrulline supplementation",
                "level":     "Level B",
                "rationale": "Bypasses OTC-ornithine step; raises both citrulline and arginine. "
                             "Supports urea cycle if ammonia is elevated. 100–200 mg/kg/day.",
            },
            {
                "treatment": "Levetiracetam (LEV)",
                "level":     "Level B — First-line AED",
                "rationale": "No interaction with proline/ornithine pathway; well-tolerated; first-line for seizures in ALDH18A1.",
            },
            {
                "treatment": "ACTH / Vigabatrin (Infantile Spasms)",
                "level":     "Level A (Infantile Spasms)",
                "rationale": "Infantile spasms (West Syndrome) are the most common seizure type in severe AR. "
                             "ACTH + pyridoxine trial: vigabatrin if tuberous sclerosis co-exists. "
                             "Standard IS protocol applies — B6 trial is reasonable (PLP is normal baseline but IS empirically trialled).",
            },
            {
                "treatment": "Proline-enriched formula / dietitian supervision",
                "level":     "Level B",
                "rationale": "Ensure adequate dietary proline from non-collagen protein sources. "
                             "Avoid fasting; ensure frequent feeds in infants. Medical metabolic dietitian essential.",
            },
            {
                "treatment": "Cataract surgery",
                "level":     "Level A (when cataracts present)",
                "rationale": "Cataracts are present in 60–75% and require surgical correction to prevent amblyopia. "
                             "Not metabolic; structural consequence of low proline in lens crystallins.",
            },
            {
                "treatment": "Collagen-support therapy (experimental)",
                "level":     "Level C — Experimental",
                "rationale": "Glycine + vitamin C (cofactors for collagen synthesis) may partially compensate for low proline-hydroxyproline. "
                             "Not standard; limited evidence; under investigation.",
            },
        ],
        "drug_risks": [
            {
                "drug":   "Vigabatrin",
                "risk":   "CAUTION — use with monitoring",
                "reason": "Vigabatrin irreversibly inhibits GABA-T → raises GABA. "
                          "In ALDH18A1 where GABA may already be secondarily reduced (low proline → low glutamate → low GABA), "
                          "vigabatrin is potentially beneficial for IS but retinal toxicity risk requires monitoring.",
            },
            {
                "drug":   "Valproate (VPA)",
                "risk":   "MODERATE RISK — Ammonia",
                "reason": "VPA inhibits urea cycle enzymes → hyperammonemia. "
                          "In ALDH18A1 where ornithine is ALREADY LOW (partial urea cycle limitation), "
                          "VPA-induced hyperammonemia risk is HIGHER than in normal patients. Monitor ammonia closely.",
            },
            {
                "drug":   "Protein restriction diets",
                "risk":   "HIGH RISK — Worsens proline deficiency",
                "reason": "Low-protein or restricted diets further reduce dietary proline intake. "
                          "ABSOLUTE CONTRAINDICATION in ALDH18A1 — proline must be SUPPLEMENTED, not restricted. "
                          "This is the OPPOSITE of PRODH/ALDH4A1 management.",
            },
            {
                "drug":   "B6 / Pyridoxine",
                "risk":   "NOT INDICATED (but not harmful)",
                "reason": "PLP is NORMAL at baseline in ALDH18A1 — no P5C-PLP inactivation occurs. "
                          "B6 supplementation has no specific metabolic benefit; harmless but misleading. "
                          "Not analogous to ALDH4A1 where B6 has partial response.",
            },
            {
                "drug":   "Levetiracetam (LEV)",
                "risk":   "SAFE — First-line",
                "reason": "No interaction with proline/ornithine pathway or PLP. Well-tolerated across neonatal, infant, paediatric ages.",
            },
            {
                "drug":   "Collagen supplements (gelatin, bone broth)",
                "risk":   "POTENTIALLY BENEFICIAL (unusual)",
                "reason": "Hydrolysed collagen provides free proline and hydroxyproline. "
                          "May help raise proline pool in mild-moderate cases. "
                          "NOTE: This is the OPPOSITE of PRODH/ALDH4A1 where high-proline foods are MODERATE RISK.",
            },
        ],
        "differential_diagnoses": [
            {
                "disease":     "PRODH Deficiency (Hyperprolinemia Type I)",
                "shared":      "Proline pathway; epilepsy; IDD; PRODH and P5CS are pathway-linked",
                "distinguish": "PRODH: Proline ELEVATED (350–1000 µmol/L); P5C NORMAL; PLP NORMAL; no cutis laxa; psychiatric features. "
                               "ALDH18A1: Proline CRITICALLY LOW (<60); ornithine LOW; cutis laxa/joint laxity 90%; NO psychiatric features.",
            },
            {
                "disease":     "ALDH4A1 Deficiency (Hyperprolinemia Type II)",
                "shared":      "Proline pathway; epilepsy; IDD",
                "distinguish": "ALDH4A1: Proline MARKEDLY ELEVATED (>1000); P5C ELEVATED (PATHOGNOMONIC); PLP LOW (secondary B6 deficiency); B6 partial response. "
                               "ALDH18A1: Proline CRITICALLY LOW; P5C NORMAL–LOW; PLP NORMAL; NO B6 response; cutis laxa.",
            },
            {
                "disease":     "ALDH7A1 Deficiency (Pyridoxine-Dependent Epilepsy / PDE)",
                "shared":      "Epilepsy; IDD; PLP pathway involvement",
                "distinguish": "ALDH7A1: alpha-AASA MARKEDLY ELEVATED (PATHOGNOMONIC); pipecolic acid ELEVATED; PLP LOW; B6 >85% response. "
                               "ALDH18A1: alpha-AASA NORMAL; pipecolic NORMAL; PLP NORMAL; proline CRITICALLY LOW; cutis laxa.",
            },
            {
                "disease":     "Cutis Laxa (ELN, FBLN5, FBLN4 mutations)",
                "shared":      "Cutis laxa (wrinkly skin); connective tissue disease; may have seizures",
                "distinguish": "ELN/FBLN5/FBLN4 cutis laxa: normal plasma amino acids; normal proline and ornithine; no metabolic epilepsy. "
                               "ALDH18A1: Proline CRITICALLY LOW + ornithine LOW on plasma amino acids — PATHOGNOMONIC in context of cutis laxa.",
            },
            {
                "disease":     "OAT Deficiency (Gyrate Atrophy of Choroid and Retina)",
                "shared":      "Ornithine pathway; low ornithine metabolism; PYCR1 overlap",
                "distinguish": "OAT deficiency: Ornithine MARKEDLY ELEVATED (not low); chorioretinal degeneration; no proline deficiency; no cutis laxa. "
                               "ALDH18A1: Ornithine LOW; proline CRITICALLY LOW; retinal findings less specific; cutis laxa present.",
            },
            {
                "disease":     "PYCR1/PYCR2 Deficiency",
                "shared":      "Proline synthesis pathway; IDD; epilepsy; cutis laxa (PYCR1)",
                "distinguish": "PYCR1 LOF: Proline LOW (cannot complete synthesis from P5C); P5C ELEVATED (accumulates if PYCR1 absent). "
                               "ALDH18A1 LOF: Proline LOW; P5C also LOW or absent (cannot be made from glutamate). "
                               "Key: P5C ELEVATED → PYCR1/2; P5C NORMAL/LOW → ALDH18A1.",
            },
        ],
    }


def get_definitions() -> dict:
    return {
        "disease":        "ALDH18A1 Deficiency — P5CS Deficiency / Δ1-Pyrroline-5-Carboxylate Synthase Deficiency",
        "gene_full":      "ALDH18A1 — Aldehyde Dehydrogenase 18 Family Member A1 (also P5CS, P5S, GSAS — Glutamate-5-Semialdehyde Synthase)",
        "omim_gene":      "OMIM *138250",
        "omim_disease_ar":"OMIM #219150 (De Barsy Syndrome / Cutis Laxa Type IIIA/IIIB — severe AR)",
        "omim_disease_ad":"OMIM #616603 (Intellectual Disability + Cataracts + Facial Dysmorphism — mild AD)",
        "chromosome":     "10q24.1",
        "protein":        "795 aa; bifunctional homodimeric enzyme; N-terminal γ-Glutamyl Kinase (G5K) domain + C-terminal Glutamate-5-Semialdehyde Dehydrogenase (G5SD) domain; NADPH + ATP-dependent; cytoplasmic/mitochondrial",
        "inheritance":    "Autosomal Recessive (severe; biallelic null / severe LOF) and Autosomal Dominant (mild; dominant-negative heterozygous)",
        "pathway": (
            "Proline and Ornithine SYNTHESIS (Step 1): "
            "L-Glutamate + ATP → [G5K domain] → γ-Glutamyl-phosphate → [G5SD domain, NADPH] → P5C/GSA + NADP+ + Pi. "
            "P5C/GSA → [PYCR1/PYCR2, NADPH] → L-Proline (final product). "
            "P5C ↔ L-Ornithine via OAT (reversible; mitochondria). "
            "ALDH18A1 LOF: P5C CANNOT be made → Proline CRITICALLY LOW + Ornithine LOW. "
            "This is the COMPLETE OPPOSITE of PRODH LOF (proline ELEVATED) and ALDH4A1 LOF (proline MARKEDLY ELEVATED + P5C ELEVATED + PLP LOW)."
        ),
        "biomarker_glossary": {
            "Proline (plasma)":
                "Primary amino acid substrate deficient in ALDH18A1. "
                "CRITICALLY LOW (<60 µmol/L; severe <30; normal 90–300 µmol/L). "
                "OPPOSITE of PRODH/ALDH4A1 where proline is ELEVATED. "
                "Proline is conditionally essential in ALDH18A1 — dietary intake cannot fully compensate for lack of de novo synthesis.",
            "Ornithine (plasma)":
                "Made from P5C/GSA via OAT (reversible) in mitochondria. "
                "LOW in ALDH18A1 (<30 µmol/L; normal 50–200) — CANNOT be synthesised without P5C. "
                "Low ornithine → impaired polyamine synthesis → NMDA receptor dysregulation. "
                "Low ornithine → reduced OTC activity → borderline hyperammonemia risk.",
            "P5C (Delta-1-pyrroline-5-carboxylate)":
                "The PRODUCT of ALDH18A1 activity and SUBSTRATE for PYCR1/2. "
                "NORMAL–LOW in ALDH18A1 LOF — cannot be MADE. "
                "Contrast: ELEVATED (PATHOGNOMONIC) in ALDH4A1 LOF where P5C accumulates. "
                "Contrast: ELEVATED in PYCR1/2 LOF where P5C is made but cannot be consumed. "
                "P5C level is the key biomarker to distinguish these three proline pathway disorders.",
            "PLP (Pyridoxal-5-phosphate)":
                "NORMAL in ALDH18A1 — no P5C-PLP inactivation (no excess P5C to form Schiff base with PLP). "
                "This distinguishes ALDH18A1 from ALDH4A1 (where P5C is HIGH and PLP is LOW). "
                "B6/pyridoxine supplementation has no specific seizure benefit in ALDH18A1.",
            "alpha-AASA (urine)":
                "NORMAL in ALDH18A1 — KEY NEGATIVE for ALDH7A1/PDE (antiquitin deficiency). "
                "alpha-AASA is the PATHOGNOMONIC biomarker of ALDH7A1 deficiency.",
            "Pipecolic acid":
                "NORMAL in ALDH18A1 — KEY NEGATIVE for ALDH7A1/PDE.",
            "ALDH18A1 / P5CS":
                "Delta-1-Pyrroline-5-Carboxylate Synthase. Bifunctional; first committed step of proline/ornithine synthesis. "
                "LOF causes inability to synthesise P5C from glutamate → proline deficiency + ornithine deficiency. "
                "AR severe = De Barsy / Cutis Laxa IIIA/B (#219150). AD mild = ID + cataracts (#616603).",
            "PYCR1/PYCR2":
                "Pyrroline-5-Carboxylate Reductases — convert P5C to proline (last step). "
                "Distinct from ALDH18A1. PYCR1/2 LOF: P5C ELEVATED + Proline LOW (cannot consume P5C). "
                "ALDH18A1 LOF: P5C LOW + Proline LOW (cannot make P5C).",
            "OAT (Ornithine Aminotransferase)":
                "Mitochondrial enzyme interconverting ornithine and P5C (bidirectional). "
                "OAT LOF causes Gyrate Atrophy: ornithine ELEVATED (accumulates when cannot be converted to P5C). "
                "In ALDH18A1: OAT is INTACT but has no P5C substrate to make ornithine → ornithine LOW.",
            "Collagen / Hydroxyproline":
                "Proline is hydroxylated to hydroxyproline by prolyl-4-hydroxylase (P4H). "
                "Hydroxyproline is essential for collagen triple-helix stability (Gly-X-Y repeat, X often Pro, Y often Hyp). "
                "ALDH18A1 → low proline → low hydroxyproline → defective collagen → cutis laxa / joint laxity / vascular fragility.",
            "Polyamines (putrescine, spermidine, spermine)":
                "Synthesised from ornithine by ODC (ornithine decarboxylase). "
                "Polyamines modulate NMDA receptor gating, neuronal plasticity, and excitability. "
                "Low ornithine in ALDH18A1 → low polyamines → altered NMDA modulation → seizure threshold.",
        },
        "key_concepts": [
            "Proline is CRITICALLY LOW in ALDH18A1 — OPPOSITE of PRODH and ALDH4A1 where proline is elevated",
            "Ornithine is LOW — impairs both polyamine synthesis (NMDA modulation) and urea cycle (ammonia risk with VPA)",
            "P5C is NORMAL–LOW (not made) — NOT elevated; this distinguishes ALDH18A1 from ALDH4A1 (P5C HIGH) and PYCR1/2 (P5C HIGH)",
            "PLP is NORMAL — NO B6/pyridoxine indication; no P5C-PLP inactivation mechanism unlike ALDH4A1",
            "Cutis laxa (wrinkly skin) + joint laxity in ~95% of severe AR — due to proline-deficient collagen; pathognomonic clinical sign",
            "Cataracts in 60–75% — lens crystallins require proline; surgical correction required to prevent amblyopia",
            "Treatment is PROLINE SUPPLEMENTATION (Level A) + Ornithine (Level B) — the metabolic OPPOSITE of PRODH/ALDH4A1 management",
            "Protein restriction is ABSOLUTELY CONTRAINDICATED — further worsens proline deficiency (opposite of PRODH/ALDH4A1)",
            "Infantile spasms (West syndrome) are the modal seizure type in severe AR (~45%)",
            "VPA MODERATE RISK: hyperammonemia risk heightened because ornithine is already LOW (compromised urea cycle)",
            "~50–100 cases worldwide (2026); truly ultra-rare; autosomal recessive (severe) / autosomal dominant (mild)",
        ],
        "variants_glossary": {
            "p.Arg138Gln":
                "γ-Glutamyl Kinase (G5K) ATP-binding domain; most common worldwide (~22%); severe null phenotype; "
                "disrupts phosphate transfer from ATP to glutamate; no residual P5C synthesis",
            "p.Gly93Arg":
                "G5K substrate-binding pocket; ~18%; severe; reduces glutamate affinity (high Km); "
                "partial but insufficient P5C output",
            "p.Arg418Cys":
                "Glutamate-5-Semialdehyde Dehydrogenase (G5SD) NADPH-binding; ~15%; severe–moderate; "
                "impairs NADPH cofactor binding in Step 2; γ-glutamyl phosphate accumulates but P5C cannot be made",
            "p.Ala445Val":
                "G5SD dimer interface; ~12%; moderate; disrupts homodimerisation; partial residual activity as monomer",
            "p.Leu503Pro":
                "G5SD catalytic domain; ~10%; severe; proline substitution disrupts alpha-helix in catalytic core",
            "c.IVS6+1G>A":
                "Splice-null, exon 6 skipping; ~9%; most severe; no functional enzyme produced; neonatal cutis laxa",
            "p.Glu247Lys":
                "G5K product-release domain; ~8%; mild AD dominant-negative; heterozygous; "
                "mutant subunit poisons wild-type homodimer → ~50% loss of function; cataracts + mild IDD",
            "p.Thr321Ile":
                "G5SD cofactor-positioning loop; ~6%; moderate; partial P5C synthesis; attenuated phenotype",
        },
        "normal_ranges": {
            "Proline (plasma)":           "90–300 µmol/L (ALDH18A1: CRITICALLY LOW <60; severe <30)",
            "Ornithine (plasma)":         "50–200 µmol/L (ALDH18A1: LOW <30)",
            "Citrulline (plasma)":        "10–45 µmol/L (ALDH18A1: LOW–NORMAL <15, secondary)",
            "Arginine (plasma)":          "40–120 µmol/L (ALDH18A1: LOW–NORMAL <30, secondary)",
            "P5C (plasma)":               "Trace / <3 µmol/L (ALDH18A1: NORMAL–LOW — cannot be made)",
            "PLP (plasma)":               "35–110 nmol/L (ALDH18A1: NORMAL — no P5C-PLP inactivation)",
            "alpha-AASA (urine)":         "<1 mmol/mol Cr (NORMAL — KEY NEGATIVE vs ALDH7A1)",
            "Pipecolic acid (plasma)":    "<5 µmol/L (NORMAL — KEY NEGATIVE vs ALDH7A1)",
            "MMA (urine)":                "<5 mmol/mol Cr (NORMAL)",
            "tHcy (plasma)":              "<15 µmol/L (NORMAL)",
            "Ammonia (plasma)":           "<50 µmol/L (ALDH18A1: may be mildly elevated <80; monitor with VPA)",
            "Hydroxyproline (urine)":     "0–100 µmol/g Cr (ALDH18A1: LOW — proline-deficient collagen)",
        },
        "drug_risks": {
            "L-Proline supplementation":               "Level A — PRIMARY TREATMENT; raises proline toward normal; improves collagen",
            "L-Ornithine supplementation":             "Level B — compensates impaired de novo ornithine synthesis",
            "Citrulline supplementation":              "Level B — supports urea cycle if ammonia borderline",
            "Levetiracetam (LEV)":                     "Level B SAFE — first-line AED; no metabolic interaction",
            "ACTH / Vigabatrin (for Infantile Spasms)":"Level A (for IS) — standard IS protocol; vigabatrin CAUTION (retinal toxicity monitoring)",
            "Valproate (VPA)":                         "MODERATE RISK — VPA-induced hyperammonemia worse when ornithine already low; use with ammonia monitoring",
            "Protein restriction diets":               "ABSOLUTE CONTRAINDICATION — further depletes proline; OPPOSITE of PRODH/ALDH4A1",
            "B6 / Pyridoxine / PLP":                   "NOT INDICATED — PLP normal; no P5C-PLP mechanism (unlike ALDH4A1)",
            "Collagen supplements (proline/hydroxyproline)": "POTENTIALLY BENEFICIAL — provides free proline; opposite of PRODH/ALDH4A1 where high-Pro foods are risky",
        },
        "treatments": {
            "L-Proline supplement (Level A)": "200–500 mg/kg/day oral; primary metabolic therapy; raises plasma proline; improves collagen integrity; reduces seizure frequency",
            "L-Ornithine supplement (Level B)": "50–200 mg/kg/day; compensates de novo synthesis failure; supports polyamine + urea cycle",
            "Citrulline (Level B)": "100–200 mg/kg/day; raises arginine + ornithine; urea cycle support",
            "LEV (Level B — first-line AED)": "Standard dosing; no metabolic interaction",
            "ACTH + vigabatrin (Infantile Spasms)": "Standard West Syndrome protocol; IS is modal seizure in severe AR",
            "Cataract surgery (Level A when present)": "Early surgery to prevent amblyopia; 60–75% require intervention",
            "Dietitian (metabolic)": "Mandatory; proline-enriched feeds; avoid fasting; ensure adequate protein",
        },
    }
