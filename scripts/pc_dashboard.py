#!/usr/bin/env python3
"""PC (Pyruvate Carboxylase) Deficiency Dashboard.

PC catalyses the pivotal anaplerotic reaction:
    Pyruvate  +  HCO₃⁻  +  ATP  →  [PC]  →  Oxaloacetate  +  ADP  +  Pi

PC LOF → Oxaloacetate (OAA) CANNOT be replenished from pyruvate:
  → Gluconeogenesis BLOCKED (PEPCK requires OAA → PEP, first step)
  → TCA cycle STARVED of OAA (anaplerotic depletion → NADH backs up → L:P ratio ↑↑)
  → Pyruvate ACCUMULATES → Lactate ↑↑↑ (PRIMARY marker)
  → Alanine ELEVATED (ALT transamination of excess pyruvate)
  → Urea cycle IMPAIRED (OAA → aspartate via GOT2; aspartate needed for ASS1; OAA depletion → NH3 ↑)
  → Citrulline can be LOW or abnormal (aspartate depleted → ASS1 substrate missing)

KEY FACTS (EXAM HIGHEST-YIELD):
  1. Lactate VERY HIGH (>5 mmol/L, often 10–20) — PRIMARY MARKER (lactic acidosis)
  2. Pyruvate ELEVATED — upstream accumulation (L:P ratio elevated >20:1)
  3. Alanine ELEVATED — transamination product of excess pyruvate (plasma AA)
  4. NH3 MILDLY–MODERATELY ELEVATED — SECONDARY (OAA depletion → ASP depleted → urea cycle impaired)
  5. Ketogenic Diet ABSOLUTE CI — increases acetyl-CoA but WITHOUT OAA, cannot enter TCA → worsens crisis
  6. Biotin PARTIALLY EFFECTIVE (PC is biotin-dependent via HLCS biotinylation; some pt partial response)
  7. Triheptanoin (C7 oil) Level B — anaplerotic substrate → propionyl-CoA → succinyl-CoA → OAA bypass
  8. 3 phenotypes: Type A (North American/Simple, 50%), Type B (French/Complex, neonatal severe, 35%),
     Type C (Benign, 15%)

OMIM Disease: #266150 (Pyruvate Carboxylase Deficiency)
OMIM Gene:    *608786 (PC)
Chromosome:   11q13.2
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Protein:      PC = 1178 aa; mitochondrial matrix; homotetrameric; biotin-dependent (BCCP domain Lys1119);
              4 functional domains: BC (biotin carboxylase), BCCP (biotin carboxyl carrier protein),
              CT (carboxyl transferase), PT (pyruvate transcarboxylase tetramerization); ~130 kDa subunit
Prevalence:   ~1:250,000 (general); ~1:3,000 in Ojibwa Cree/Algonquian First Nations communities (Canada)

BIOTIN DEPENDENCY OF PC:
  PC is ONE of FOUR biotin-dependent carboxylases (PC, PCC, MCC, ACC).
  HLCS biotinylates PC at Lys1119 (BCCP domain) → PC becomes active holoenzyme.
  BTD recycles biotin from biocytin → free biotin → HLCS → holocarboxylases.
  PC deficiency = ISOLATED failure of ONE carboxylase (the enzyme itself is mutant).
  HLCS deficiency = ALL FOUR carboxylases inactive simultaneously (ligation defect).
  BTD deficiency = ALL FOUR carboxylases fail INDIRECTLY (biotin depletion).
  KEY DISTINCTION: PC deficiency → C5-OH NORMAL, C3 NORMAL (PCC/MCC/ACC still active if biotin OK).

BIOMARKER PATTERN — PC DEFICIENCY:
  Lactate              5–25 mmol/L (venous) — PRIMARY MARKER; very high in type B (>15)
  Pyruvate             0.2–3.0 mmol/L (normally <0.1 mmol/L)
  L:P ratio            >20:1 (often >50:1 in type B) — reflects NADH excess / TCA back-up
  Alanine              ELEVATED (300–1500 µmol/L; transamination of excess pyruvate)
  NH3                  mildly–moderately elevated in type A/B (OAA depletion → ASP depleted)
  3-OHB (β-OHB)       ELEVATED (paradoxically; FA oxidation intact, but ratio to AcAc high)
  3-OHB:AcAc ratio    ELEVATED (>3:1, reflects NADH excess — opposite of starvation ketosis)
  Citrulline           LOW–NORMAL (aspartate depleted for ASS1 condensation step)
  Glucose              LOW in type B (gluconeogenesis blocked; type A may be normal interictally)
  Bicarbonate          LOW (metabolic acidosis; anion gap elevated)

KEY NEGATIVE MARKERS (CRITICAL DIFFERENTIALS):
  C5-OH    NORMAL — KEY NEGATIVE vs HLCS/BTD/MCC (MCC is NOT blocked in isolated PC deficiency)
  C3       NORMAL — KEY NEGATIVE vs HLCS/BTD/PA (PCC is NOT blocked in isolated PC deficiency)
  3-MCG    ABSENT — KEY NEGATIVE vs HLCS/BTD/MCC
  MMA      ABSENT — KEY NEGATIVE vs MMA/MMUT (PC does not involve cobalamin pathway)
  Biotinidase NORMAL — KEY NEGATIVE vs BTD
  Plasma biotin NORMAL — KEY NEGATIVE vs BTD
  Alloisoleucine ABSENT — KEY NEGATIVE vs MSUD (BCAA catabolism step 2 intact)
  Methylcitrate ABSENT — KEY NEGATIVE vs PA/MMA (PCC intact → propionyl-CoA not accumulated)

PC vs PDH DEFICIENCY (CRITICAL DIFFERENTIAL — BOTH CAUSE LACTIC ACIDOSIS):
  Both cause severe lactic acidosis; distinguishing features:
  PC:   L:P > 20:1 (NADH EXCESS — OAA depletion, NADH backs up); gluconeogenesis blocked;
        alanine ↑↑; NH3 ↑ (OAA→ASP depleted); KD ABSOLUTE CI (worsens crisis); biotin may help
  PDH:  L:P < 10:1 (NADH NOT excess — pyruvate CANNOT be oxidised; NADH normal);
        pyruvate stuck PRE-oxidation; NO NH3 elevation; KD can HELP (bypasses PDH block);
        thiamine LEVEL A; glucose restriction helpful (contrast PC where glucose is the treatment)

TREATMENT:
  IV Glucose:                 LEVEL A — acute crisis; gluconeogenesis blocked → exogenous glucose essential
  Citrate / Aspartate:        LEVEL B — TCA anaplerosis; aspartate replenishes urea cycle substrate
  Triheptanoin (C7 oil):      LEVEL B — propionyl-CoA → succinyl-CoA (TCA anaplerosis, bypasses OAA block)
  Biotin:                     LEVEL B — PC is biotin-dependent; partial responsiveness in some patients
  Low-carbohydrate diet:      LEVEL C — reduces pyruvate load
  Thiamine (B1):              NOT primary (unlike PDH deficiency where thiamine activates PDH)
  Ketogenic Diet:             ABSOLUTE CI — increases acetyl-CoA but OAA absent → acetyl-CoA CANNOT enter TCA;
                              β-oxidation → more NADH → worsens L:P ratio; KD crises are fatal in type B
  VPA:                        AVOID — mitochondrial toxicity; reduces PC expression; worsens lactic acidosis
  Fasting:                    EXTREME HAZARD — gluconeogenesis blocked + catabolism worsens acidosis

PHENOTYPES:
  Type A (North American/Simple):  Moderate lactic acidosis; psychomotor retardation; 50% MODAL;
                                   seizures 60-70%; some L-P ratio 20-40:1
  Type B (French/Complex):         Severe neonatal lactic acidosis + hyperammonemia + hypoglycemia +
                                   citrullinemia; often fatal within weeks; L:P >50:1; 35%
  Type C (Benign):                 Episodic mild lactic acidosis; near-normal development; 15%;
                                   often identified in adulthood

SEIZURES IN PC:
  ~65% overall (type A and B predominantly)
  Mechanism: cerebral energy failure (gluconeogenesis blocked → brain glucose deprivation);
             NADH excess → impaired brain energy metabolism; secondary lactic acidosis neurotoxicity
  Types: infantile spasms, generalised tonic-clonic, focal; DRE in severe type B
  AED: LEV preferred; VPA AVOID

COMMON PATHOGENIC VARIANTS:
  p.Arg631Gln (c.1892G>A):    ~22% — CT domain adjacent to active site; type A (moderate)
  p.Arg451Cys (c.1351C>T):    ~18% — Ojibwa Cree founder; BCCP domain (near Lys1119 biotin site); type A
  p.Arg274His (c.821G>A):     ~10% — BC domain; type B (severe); poor biotin response
  p.Tyr628Cys (c.1883A>G):    ~8%  — CT domain; type A; variable
  c.2367+1G>A (splice):       ~7%  — null allele; type B; neonatal severe
  p.Ala956Thr (c.2866G>A):    ~5%  — Ojibwa Cree; type A moderate (second Cree variant)
  p.Met743Ile (c.2229G>A):    ~5%  — BC-CT linker; attenuated type C
  p.Glu565Gly (c.1694A>G):    ~4%  — BC domain; type A

PREVALENCE NOTES:
  ~1:250,000 general population (range 1:100,000–1:500,000 by region)
  ~1:3,000 in Ojibwa Cree communities (Manitoba/Ontario, Canada) — founder p.Arg451Cys
  ~1:3,500 in Algonquian communities — same Cree/Algonquian founder allele
"""

import random

SEED       = 265      # PC = 265 (AUH = 263, HMGCL = 261, MCC = 259 — progression)
N_PATIENTS = 40

# ── Phenotype classes ─────────────────────────────────────────────────────────
PHENOTYPE_CLASSES = [
    {
        "class": "Type A (North American / Simple — Moderate)",
        "pct": 50,
        "age_onset_months_range": (0, 6),
        "lactate_range": (5.0, 15.0),        # mmol/L
        "pyruvate_range": (0.25, 1.5),       # mmol/L
        "lp_ratio_range": (22, 45),          # L:P
        "alanine_range": (400, 1200),        # µmol/L
        "nh3_range": (60, 180),              # µmol/L (mild secondary elevation)
        "glucose_range": (2.5, 5.5),         # mmol/L (may be low but not as severe as B)
        "bicarb_range": (10, 18),            # mmol/L (acidosis)
        "seizures_prob": 0.65,
        "id_prob": 0.72,
        "hypotonia_prob": 0.60,
        "hypoglycemia_prob": 0.40,
        "note": "MODAL 50%; moderate lactic acidosis; progressive psychomotor retardation; seizures 65%; some biotin partial response",
    },
    {
        "class": "Type B (French / Complex — Neonatal Severe)",
        "pct": 35,
        "age_onset_months_range": (0, 0.5),   # neonatal
        "lactate_range": (12.0, 25.0),        # mmol/L — severe
        "pyruvate_range": (0.8, 3.0),
        "lp_ratio_range": (40, 80),           # severe NADH excess
        "alanine_range": (800, 2500),
        "nh3_range": (120, 500),              # significant secondary hyperammonemia
        "glucose_range": (1.0, 3.0),          # severe hypoglycemia
        "bicarb_range": (5, 14),              # severe acidosis
        "seizures_prob": 0.82,
        "id_prob": 0.90,
        "hypotonia_prob": 0.85,
        "hypoglycemia_prob": 0.80,
        "note": "Neonatal severe; L:P >40:1; hyperammonemia; hypoglycemia; citrullinemia; often fatal in early infancy without intensive support",
    },
    {
        "class": "Type C (Benign — Episodic)",
        "pct": 15,
        "age_onset_months_range": (6, 36),
        "lactate_range": (2.5, 6.0),
        "pyruvate_range": (0.12, 0.4),
        "lp_ratio_range": (15, 25),
        "alanine_range": (200, 500),
        "nh3_range": (30, 60),
        "glucose_range": (3.5, 6.0),
        "bicarb_range": (16, 22),
        "seizures_prob": 0.25,
        "id_prob": 0.15,
        "hypotonia_prob": 0.20,
        "hypoglycemia_prob": 0.15,
        "note": "Episodic mild lactic acidosis; near-normal development; some identified in adulthood; good prognosis if managed",
    },
]

# Common pathogenic variants
VARIANTS = [
    {"variant": "p.Arg631Gln",   "freq": 22, "domain": "CT domain (active site adjacent)", "phenotype": "Type A (moderate); partial residual CT activity",      "note": "c.1892G>A; most common non-Cree allele; reduces CT catalysis; partial biotin response possible"},
    {"variant": "p.Arg451Cys",   "freq": 18, "domain": "BCCP domain (near Lys1119 biotin)", "phenotype": "Type A; Ojibwa Cree founder; moderate",            "note": "c.1351C>T; Cree/Algonquian founder; BCCP domain; biotin-attachment site vicinity; ~1:3,000 in Cree"},
    {"variant": "p.Arg274His",   "freq": 10, "domain": "BC domain (biotin carboxylase)",    "phenotype": "Type B (severe); poor/no biotin response",         "note": "c.821G>A; BC domain disruption; severe neonatal; minimal residual PC activity"},
    {"variant": "p.Tyr628Cys",   "freq": 8,  "domain": "CT domain",                         "phenotype": "Type A; variable severity; moderate",              "note": "c.1883A>G; CT domain; partial activity; responds variably to biotin trial"},
    {"variant": "c.2367+1G>A",   "freq": 7,  "domain": "Splice site (intron 16)",           "phenotype": "Type B (severe); null allele; neonatal",           "note": "Splice donor disruption → exon skip → frameshift → NMD; complete LOF"},
    {"variant": "p.Ala956Thr",   "freq": 5,  "domain": "CT-PT linker (Ojibwa Cree)",        "phenotype": "Type A moderate; second Cree allele",              "note": "c.2866G>A; second Ojibwa Cree founder variant; moderate type A; partially responsive"},
    {"variant": "p.Met743Ile",   "freq": 5,  "domain": "BC-CT linker",                      "phenotype": "Type C (benign); residual activity ~30–40%",       "note": "c.2229G>A; mild phenotype; episodic; near-normal development"},
    {"variant": "p.Glu565Gly",   "freq": 4,  "domain": "BC domain",                         "phenotype": "Type A; moderate; variable biotin response",       "note": "c.1694A>G; BC domain; destabilises biotin carboxylation; moderate reduction in PC activity"},
]


def _gen_cohort():
    """Generate synthetic 40-patient cohort with PC deficiency-realistic biomarker profiles."""
    random.seed(SEED)
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append(PHENOTYPE_CLASSES[0])
    phenotype_dist = phenotype_dist[:N_PATIENTS]

    patients = []
    for i in range(N_PATIENTS):
        cls = phenotype_dist[i]
        pid = f"PC-{i+1:03d}"

        # Variant selection (weighted by frequency)
        var_pool = []
        for v in VARIANTS:
            var_pool.extend([v["variant"]] * v["freq"])
        variant = random.choice(var_pool)

        onset_mo    = round(random.uniform(*cls["age_onset_months_range"]), 1)
        lactate     = round(random.uniform(*cls["lactate_range"]), 1)
        pyruvate    = round(random.uniform(*cls["pyruvate_range"]), 2)
        lp_ratio    = round(lactate / (pyruvate if pyruvate > 0.05 else 0.1), 1)
        alanine     = round(random.uniform(*cls["alanine_range"]), 0)
        nh3         = round(random.uniform(*cls["nh3_range"]), 0)
        glucose     = round(random.uniform(*cls["glucose_range"]), 2)
        bicarb      = round(random.uniform(*cls["bicarb_range"]), 1)

        # KEY NEGATIVES (all should be normal in isolated PC deficiency)
        c5oh        = round(random.uniform(0.05, 0.35), 2)   # NORMAL — KEY NEGATIVE vs HLCS/MCC/BTD
        c3          = round(random.uniform(0.5, 2.8), 2)     # NORMAL — KEY NEGATIVE vs HLCS/PA
        biotinidase = round(random.uniform(5.8, 9.8), 1)     # NORMAL — KEY NEGATIVE vs BTD (nmol/min/mL)

        seiz        = random.random() < cls["seizures_prob"]
        idd         = random.random() < cls["id_prob"]
        hypotonia   = random.random() < cls["hypotonia_prob"]
        hypoglycemia = random.random() < cls["hypoglycemia_prob"]
        vpa_tried   = seiz and random.random() < 0.12    # occasionally given before diagnosis
        kd_tried    = random.random() < 0.05             # rare — absolute CI but sometimes tried before diagnosis
        biotin_trial = random.random() < 0.55            # often tried as part of workup

        patients.append({
            "id": pid,
            "phenotype": cls["class"],
            "variant": variant,
            "age_onset_months": onset_mo,
            "lactate_mmol_l": lactate,           # PRIMARY — very high
            "pyruvate_mmol_l": pyruvate,
            "lp_ratio": lp_ratio,                # L:P > 20:1 — NADH excess hallmark
            "alanine_umol_l": int(alanine),      # ELEVATED
            "nh3_umol_l": int(nh3),              # secondary elevation
            "glucose_mmol_l": glucose,
            "bicarb_mmol_l": bicarb,
            "c5oh_umol_l": c5oh,                 # NORMAL — KEY NEGATIVE
            "c3_umol_l": c3,                     # NORMAL — KEY NEGATIVE
            "biotinidase_activity": biotinidase, # NORMAL — KEY NEGATIVE
            "seizures": seiz,
            "intellectual_disability": idd,
            "hypotonia": hypotonia,
            "hypoglycemia": hypoglycemia,
            "vpa_given": vpa_tried,
            "kd_tried": kd_tried,
            "biotin_trial": biotin_trial,
        })
    return patients


_COHORT = _gen_cohort()


def get_overview():
    pheno_counts = {}
    for p in _COHORT:
        k = p["phenotype"]
        pheno_counts[k] = pheno_counts.get(k, 0) + 1

    pheno_dist = [
        {"class": k, "n": v, "pct": round(v / N_PATIENTS * 100)}
        for k, v in pheno_counts.items()
    ]

    n_seiz     = sum(1 for p in _COHORT if p["seizures"])
    n_idd      = sum(1 for p in _COHORT if p["intellectual_disability"])
    n_hypo_t   = sum(1 for p in _COHORT if p["hypotonia"])
    n_hypogluc = sum(1 for p in _COHORT if p["hypoglycemia"])
    n_vpa      = sum(1 for p in _COHORT if p["vpa_given"])
    n_kd       = sum(1 for p in _COHORT if p["kd_tried"])
    n_biotin   = sum(1 for p in _COHORT if p["biotin_trial"])
    avg_lac    = round(sum(p["lactate_mmol_l"] for p in _COHORT) / N_PATIENTS, 1)
    avg_lp     = round(sum(p["lp_ratio"] for p in _COHORT) / N_PATIENTS, 1)
    avg_ala    = round(sum(p["alanine_umol_l"] for p in _COHORT) / N_PATIENTS, 0)

    return {
        "disease": "PC Deficiency (Pyruvate Carboxylase Deficiency / PCD)",
        "gene": "PC",
        "omim_gene": "608786",
        "omim_disease": "266150",
        "locus": "11q13.2",
        "inheritance": "Autosomal Recessive",
        "prevalence": "~1:250,000 (general); ~1:3,000 in Ojibwa Cree (Canada)",
        "pathway_context": (
            "PC catalyses the critical anaplerotic step: Pyruvate + HCO₃⁻ + ATP → Oxaloacetate + ADP + Pi. "
            "OAA is required for gluconeogenesis (PEPCK: OAA → PEP) AND TCA cycle entry (OAA + Acetyl-CoA → citrate). "
            "PC LOF → gluconeogenesis BLOCKED + TCA STARVED → Lactate ↑↑↑ + L:P ratio ↑↑↑. "
            "PC is ONE of 4 biotin-dependent carboxylases (PC, PCC, MCC, ACC) — isolated PC deficiency "
            "unlike HLCS/BTD where ALL FOUR fail simultaneously."
        ),
        "n_patients": N_PATIENTS,
        "seed": SEED,
        "kpi": {
            "n_patients":    {"label": "Cohort",                    "value": str(N_PATIENTS),                                      "color": "#b71c1c"},
            "avg_lactate":   {"label": "Avg Lactate (mmol/L)",      "value": str(avg_lac),                                         "color": "#c62828"},
            "avg_lp":        {"label": "Avg L:P Ratio",             "value": str(avg_lp),                                          "color": "#d32f2f"},
            "avg_alanine":   {"label": "Avg Alanine (µmol/L)",      "value": str(int(avg_ala)),                                    "color": "#e64a19"},
            "seizures":      {"label": "Seizures",                  "value": f"{n_seiz} ({round(n_seiz/N_PATIENTS*100)}%)",        "color": "#7b1fa2"},
            "idd":           {"label": "Intellectual Disability",   "value": f"{n_idd} ({round(n_idd/N_PATIENTS*100)}%)",         "color": "#4a148c"},
            "hypoglycemia":  {"label": "Hypoglycemia",              "value": f"{n_hypogluc} ({round(n_hypogluc/N_PATIENTS*100)}%)", "color": "#01579b"},
            "kd_ci_event":   {"label": "KD tried (absolute CI!)",   "value": f"{n_kd} pts — KD is ABSOLUTE CI",                   "color": "#bf360c"},
        },
        "phenotype_dist": pheno_dist,
        "kd_absolute_ci_note": (
            "Ketogenic Diet is an ABSOLUTE CONTRAINDICATION in PC deficiency. "
            "KD dramatically increases β-oxidation → Acetyl-CoA → BUT without OAA (PC deficient), "
            "Acetyl-CoA CANNOT enter TCA cycle (citrate synthase requires OAA). "
            "Result: massive NADH accumulation, worsened L:P ratio, paradoxical worsening of lactic acidosis. "
            "β-OHB also accumulates (ketosis without TCA entry). "
            f"In this cohort, {n_kd} patient(s) had KD trial before diagnosis — a critical treatment error."
        ),
        "pc_vs_pdh_note": (
            "PC vs PDH Deficiency — both cause lactic acidosis, critical differential:\n"
            "PC:  L:P ratio >20:1 (NADH EXCESS — OAA depleted, TCA backs up, NADH not re-oxidised); "
            "gluconeogenesis BLOCKED; NH3 ↑ (secondary OAA→ASP depletion); KD ABSOLUTE CI; "
            "biotin may help (PC is biotin-dependent); glucose is treatment.\n"
            "PDH: L:P ratio <10:1 (pyruvate CANNOT be oxidised but NADH not excess); "
            "gluconeogenesis can work (PC intact); NH3 NORMAL; KD can HELP (bypasses PDH); "
            "thiamine LEVEL A; glucose WORSENS (more pyruvate)."
        ),
        "hallmark_biomarker": (
            "Lactate VERY HIGH (5–25 mmol/L) — PRIMARY MARKER. "
            "Pyruvate ELEVATED (L:P ratio >20:1 reflects NADH excess / TCA starvation). "
            "Alanine ELEVATED (transamination of excess pyruvate). "
            "NH3 mildly–moderately elevated (secondary OAA depletion → ASP depleted → urea cycle impaired). "
            "C5-OH NORMAL — KEY NEGATIVE vs HLCS/MCC/BTD. "
            "C3 NORMAL — KEY NEGATIVE vs HLCS/PA (PCC NOT blocked in isolated PC deficiency). "
            "Biotinidase NORMAL — KEY NEGATIVE vs BTD. "
            "MMA ABSENT — KEY NEGATIVE vs MMA/MMUT."
        ),
        "biotin_note": (
            f"Biotin trial performed in {n_biotin}/{N_PATIENTS} patients — PC is biotin-dependent via HLCS biotinylation at Lys1119. "
            "Some type A patients show partial lactate improvement with high-dose biotin (10–40 mg/day). "
            "Biotin does NOT correct mutations in the BC/CT catalytic domains (type B, severe alleles). "
            "A biotin trial is always worthwhile (Level B) but should NOT delay definitive diagnosis. "
            "CRITICAL: Biotin dramatically corrects HLCS/BTD (all 4 carboxylases) but only partially "
            "helps isolated PC deficiency (single enzyme with structural mutation)."
        ),
        "vpa_risk_note": (
            f"{n_vpa} cohort patients received VPA before the diagnosis was established. "
            "VPA AVOID in PC deficiency: (1) mitochondrial toxicity — reduces PC protein expression; "
            "(2) carnitine depletion — worsens metabolic stress; (3) synergistic lactic acidosis risk; "
            "(4) valproyl-CoA competitively inhibits mitochondrial enzymes. "
            "LEV is the preferred AED for seizures in PC deficiency (no mitochondrial effects)."
        ),
    }


def get_breakdown():
    biomarkers = {
        "lactate": {
            "label": "Lactate",
            "normal": "<2.0 mmol/L (venous)",
            "status": "VERY HIGH — 5–25 mmol/L — PRIMARY MARKER (lactic acidosis)",
            "direction": "↑↑↑ PRIMARY",
            "color": "danger",
            "rationale": (
                "PC LOF → OAA CANNOT be replenished from pyruvate. TCA cycle STARVED of OAA → "
                "NADH cannot be re-oxidised via TCA → lactate dehydrogenase (LDH) converts "
                "excess pyruvate + NADH → lactate + NAD⁺ (compensatory). "
                "Lactate very high in all phenotypes; especially severe in type B (>12 mmol/L). "
                "Detected by plasma lactate (bedside), blood gas (elevated anion gap acidosis), "
                "or urine lactate (GC-MS). Lactic acidosis is the diagnostic key to all PC phenotypes."
            ),
        },
        "pyruvate": {
            "label": "Pyruvate",
            "normal": "<0.1 mmol/L",
            "status": "ELEVATED — 0.2–3.0 mmol/L (upstream accumulation)",
            "direction": "↑↑",
            "color": "danger",
            "rationale": (
                "PC is the only enzyme that converts pyruvate → OAA (anaplerosis). "
                "PDH also consumes pyruvate (→ Acetyl-CoA) — in PC deficiency, PDH is INTACT "
                "but the massive pyruvate overflow exceeds PDH capacity. "
                "Pyruvate also accumulates as LDH equilibrium shifts toward lactate. "
                "Plasma pyruvate measurement requires special handling (ice, immediate deproteinisation); "
                "can be falsely elevated pre-analytically — bedside L:P ratio is the reliable index."
            ),
        },
        "lp_ratio": {
            "label": "Lactate:Pyruvate Ratio (L:P)",
            "normal": "<10:1 (most authors use <20:1 as upper normal)",
            "status": "ELEVATED >20:1 — often 30–80:1 in type B — NADH EXCESS signature",
            "direction": "↑↑↑ (CRITICAL DIFFERENTIAL vs PDH)",
            "color": "danger",
            "rationale": (
                "L:P ratio reflects cytoplasmic [NADH]/[NAD⁺] redox state. "
                "PC deficiency → TCA starved → NADH backs up → [NADH] high → L:P ↑↑. "
                "PDH deficiency → pyruvate stuck PRE-oxidation → [NADH] NOT excess → L:P <10:1. "
                "L:P > 20:1 = 'secondary' lactic acidosis (NADH excess from TCA/respiratory chain defect). "
                "L:P < 10:1 = 'primary' pyruvate defect (PDH, pyruvate transporter). "
                "PC: L:P always >20:1 (type A typically 22–45; type B typically 40–80). "
                "PDH: L:P typically <10:1 (pyruvate cannot be oxidised, but NADH not in excess)."
            ),
        },
        "alanine": {
            "label": "Alanine (plasma amino acids)",
            "normal": "<400 µmol/L",
            "status": "ELEVATED — 400–2500 µmol/L — transamination product of excess pyruvate",
            "direction": "↑↑",
            "color": "warning",
            "rationale": (
                "Excess pyruvate → ALT (alanine transaminase) transamination → Alanine + 2-oxoglutarate. "
                "Alanine elevation is proportional to pyruvate burden. "
                "Detected on plasma amino acid (PAA) chromatography — CRITICAL to order PAA "
                "when investigating lactic acidosis. "
                "Alanine >800 µmol/L suggests severe PDH/PC deficiency or MSUD (different context). "
                "Alanine ELEVATED in BOTH PC and PDH deficiency — NOT a distinguishing marker."
            ),
        },
        "nh3": {
            "label": "Ammonia (NH3)",
            "normal": "<50 µmol/L",
            "status": "MILDLY–MODERATELY ELEVATED — 60–500 µmol/L — SECONDARY (OAA depletion → ASP depleted)",
            "direction": "↑ (secondary)",
            "color": "warning",
            "rationale": (
                "OAA depletion (PC LOF) → GOT2 (mitochondrial aspartate transaminase) cannot regenerate "
                "aspartate (OAA + glutamate → aspartate + 2-OG). Aspartate depleted → ASS1 "
                "(argininosuccinate synthase) has no substrate (citrulline + aspartate → argininosuccinate). "
                "Urea cycle impaired SECONDARILY — NOT a primary UCD. "
                "NH3 elevation in PC is SECONDARY, milder than primary UCDs (CPS1, OTC, NAGS, ARG1). "
                "Citrulline may be low or abnormal (condensation step blocked by aspartate depletion). "
                "KEY NEGATIVE: NH3 dramatically improves with IV glucose + OAA restoration (vs primary UCD "
                "which requires ammonia scavengers + protein restriction long-term)."
            ),
        },
        "glucose": {
            "label": "Blood Glucose",
            "normal": "3.5–6.0 mmol/L",
            "status": "LOW in type B (<2.5 mmol/L); variable in type A; NORMAL in type C",
            "direction": "↓ (type B), variable (type A), → (type C)",
            "color": "warning",
            "rationale": (
                "Gluconeogenesis BLOCKED in PC deficiency (PEPCK cannot convert OAA → PEP without OAA). "
                "During fasting, liver cannot make glucose from lactate/pyruvate/alanine → hypoglycemia. "
                "Type B (severe): hypoglycemia prominent — brain deprived of both glucose AND ketones. "
                "Type A (moderate): gluconeogenesis partially impaired; hypoglycemia mainly during illness. "
                "TREATMENT: IV glucose is LEVEL A first-line (exogenous glucose essential; "
                "do NOT withhold glucose in suspected PC deficiency)."
            ),
        },
        "c5oh": {
            "label": "C5-OH (3-Hydroxy-isovalerylcarnitine — NBS)",
            "normal": "<0.4 µmol/L",
            "status": "NORMAL — KEY NEGATIVE vs HLCS/BTD/MCC (MCC is NOT blocked in isolated PC deficiency)",
            "direction": "→ NORMAL (KEY NEGATIVE)",
            "color": "success",
            "rationale": (
                "PC deficiency = ISOLATED failure of ONE biotin-dependent carboxylase (PC itself). "
                "MCC (3-methylcrotonyl-CoA carboxylase) remains fully active in PC deficiency "
                "because MCC has its own biotin via HLCS; PC and MCC are separate enzymes. "
                "C5-OH elevated only when MCC is defective (isolated MCC deficiency) or when "
                "HLCS/BTD causes ALL carboxylases to fail (including MCC → C5-OH ↑). "
                "CRITICAL EXAM POINT: PC deficiency on NBS → C5-OH NORMAL; elevated lactate on VLMS; "
                "NBS does NOT reliably detect PC deficiency (lactate not always on NBS panels)."
            ),
        },
        "c3": {
            "label": "C3 (Propionylcarnitine)",
            "normal": "<5 µmol/L",
            "status": "NORMAL — KEY NEGATIVE vs HLCS/PA (PCC is NOT blocked in isolated PC deficiency)",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "PCC (propionyl-CoA carboxylase) is a separate biotin-dependent enzyme from PC. "
                "In isolated PC deficiency, PCC remains fully active (biotin ligation by HLCS is intact; "
                "only the PC protein itself is mutant). Propionyl-CoA is carboxylated normally → "
                "methylmalonyl-CoA → succinyl-CoA (via MMUT). "
                "C3 elevated only in HLCS/BTD (all carboxylases fail) or PCCA/PCCB (PCC structural defect) "
                "or propionic acidemia from dietary substrate overload."
            ),
        },
        "biotinidase": {
            "label": "Biotinidase Activity",
            "normal": ">4.5 nmol/min/mL (profound <10% activity; partial 10–30%)",
            "status": "NORMAL — KEY NEGATIVE vs BTD (biotinidase is intact; PC enzyme itself is mutant)",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "BTD deficiency causes biotin recycling failure → free biotin depleted → HLCS cannot "
                "biotinylate any of the 4 carboxylases → all fail indirectly. "
                "PC deficiency: biotinidase enzyme is FULLY FUNCTIONAL. Free biotin is available. "
                "HLCS can biotinylate PC (Lys1119) normally — but the PC protein itself is structurally "
                "defective (catalytic domain mutation). Biotin supplementation may partially help "
                "(if residual apoenzyme) but CANNOT correct structural mutations in BC/CT domains. "
                "BIOTINIDASE NORMAL = NOT BTD (contrast with BTD where activity is <30% of normal)."
            ),
        },
    }

    enzyme_mechanism = {
        "function": (
            "PC (PC gene, 11q13.2, AR) encodes a 1178-aa mitochondrial matrix protein (~130 kDa subunit). "
            "Functional enzyme: HOMOTETRAMERIC (4 identical subunits). "
            "4 functional domains per subunit:\n"
            "  (1) BC domain (Biotin Carboxylase): carboxylates biotin using CO₂ + ATP → carboxybiotin\n"
            "  (2) BCCP domain (Biotin Carboxyl Carrier Protein): Lys1119 carries the biotin; swings between BC and CT\n"
            "  (3) CT domain (Carboxyl Transferase): transfers carboxyl from carboxybiotin → pyruvate → OAA\n"
            "  (4) PT domain (allosteric): tetramerization; acetyl-CoA activator binding (PC requires acetyl-CoA as allosteric activator)\n"
            "Biotin attachment: HLCS biotinylates PC at Lys1119 in the BCCP domain → active holoPC."
        ),
        "reaction": (
            "Pyruvate  +  HCO₃⁻  +  ATP  →  [PC]  →  Oxaloacetate  +  ADP  +  Pi\n"
            "(Biotin-dependent; requires Acetyl-CoA as allosteric activator; mitochondrial matrix)\n"
            "Mechanism:\n"
            "  Step 1 (BC domain): Biotin-Lys1119 + CO₂ + ATP → Carboxybiotin-Lys1119 + ADP + Pi\n"
            "  Step 2 (BCCP swings from BC to CT): Carboxybiotin-Lys1119 moves to CT active site\n"
            "  Step 3 (CT domain): Carboxybiotin + Pyruvate → OAA + Biotin-Lys1119 (regenerated)\n"
            "PC LOF → OAA CANNOT be synthesised from pyruvate → gluconeogenesis blocked + TCA starved"
        ),
        "block": (
            "PC LOF → Pyruvate CANNOT be converted to OAA:\n"
            "  → Gluconeogenesis BLOCKED (PEPCK step: OAA → PEP requires OAA from PC; absent OAA → no glucose)\n"
            "  → TCA cycle STARVED (Citrate synthase: OAA + Acetyl-CoA → Citrate; no OAA → TCA backup)\n"
            "  → NADH NOT re-oxidised (TCA and ETC back up → NADH:NAD⁺ ↑ → L:P ratio ↑↑↑)\n"
            "  → Pyruvate ACCUMULATES → LDH converts excess to lactate → LACTIC ACIDOSIS\n"
            "  → Alanine ↑↑ (ALT transamination of excess pyruvate → alanine)\n"
            "  → OAA→ASP depleted → urea cycle impaired (secondary NH3 ↑)"
        ),
        "why_kd_fails": (
            "Ketogenic Diet (KD) is ABSOLUTE CI in PC deficiency:\n"
            "  KD ↑↑↑ FA β-oxidation → Acetyl-CoA floods mitochondria\n"
            "  BUT: Citrate synthase (TCA entry) requires OAA + Acetyl-CoA → Citrate\n"
            "  PC deficient → OAA ABSENT → Acetyl-CoA CANNOT enter TCA cycle\n"
            "  Result: Acetyl-CoA → HMGCS2 → β-OHB (ketones accumulate; ketosis) \n"
            "          + NADH:NAD⁺ ratio worsens (β-oxidation generates NADH; TCA cannot clear it)\n"
            "          + Lactic acidosis WORSENS (more NADH → more lactate via LDH)\n"
            "KD is ONLY helpful in PDH deficiency (bypasses PDH block via Acetyl-CoA directly from FA)\n"
            "PC deficiency: KD → metabolic CATASTROPHE. Contrast: KD = TREATMENT for PDH."
        ),
    }

    seizure_types = [
        {"type": "Generalised tonic-clonic (energy failure, type A/B)", "pct": 40, "note": "Cerebral energy failure (gluconeogenesis blocked → brain glucose deprived); lactic acidosis neurotoxicity"},
        {"type": "Infantile spasms / West syndrome (type B, neonatal)", "pct": 25, "note": "Severe neonatal; type B; hypsarrhythmia; poor response to ACTH; underlying energy failure"},
        {"type": "Focal seizures (cortical energy depletion)",          "pct": 20, "note": "Type A moderate; focal cortical areas vulnerable to L:P elevation"},
        {"type": "Drug-resistant epilepsy (DRE — type B)",             "pct": 15, "note": "Severe type B; DRE reflects irreversible neuronal injury from sustained lactic acidosis"},
    ]

    treatments = [
        {
            "therapy": "IV Glucose (Dextrose infusion)",
            "level": "A",
            "dose": "GIR 8–12 mg/kg/min; D10W at 1.5× maintenance; target glucose 5–8 mmol/L",
            "rationale": "Gluconeogenesis BLOCKED → exogenous glucose is MANDATORY in acute crisis. Glucose provides brain fuel (bypasses blocked gluconeogenesis) and reduces catabolism. FIRST-LINE treatment. Do NOT withhold glucose pending diagnosis.",
        },
        {
            "therapy": "Citrate / Tri-citrate supplementation",
            "level": "B",
            "dose": "K-citrate or Na-citrate 1–3 mEq/kg/day; adjust by blood gas",
            "rationale": "Citrate → intramitochondrial citrate → TCA anaplerosis. Partially bypasses OAA depletion by providing TCA intermediates (citrate → isocitrate → 2-OG). Also treats metabolic acidosis. Combined with aspartate in some protocols.",
        },
        {
            "therapy": "Triheptanoin (C7 oil) — anaplerotic substrate",
            "level": "B",
            "dose": "1–2 g/kg/day oral C7 oil (propionyl-CoA → succinyl-CoA → TCA)",
            "rationale": "C7 fatty acid → propionyl-CoA + acetyl-CoA (via β-oxidation). Propionyl-CoA → D-methylmalonyl-CoA → succinyl-CoA (via MMUT) → enters TCA cycle DIRECTLY (bypassing OAA requirement). Succinyl-CoA supports TCA anaplerosis. Level B evidence; used in some metabolic centres for type A.",
        },
        {
            "therapy": "Aspartate supplementation",
            "level": "B",
            "dose": "100–300 mg/kg/day in divided doses",
            "rationale": "Aspartate directly replenishes the urea cycle substrate (ASS1 requires citrulline + aspartate). Also provides TCA intermediate (aspartate → OAA via GOT2). Reduces secondary hyperammonemia in type B. Used as adjunct to IV glucose.",
        },
        {
            "therapy": "Biotin supplementation",
            "level": "B",
            "dose": "10–40 mg/day oral; trial for 4–8 weeks",
            "rationale": "PC is biotin-dependent (HLCS biotinylates PC at Lys1119). Some patients with type A (especially p.Arg451Cys, p.Ala956Thr) show partial lactate improvement. Biotin cannot correct structural mutations in BC/CT catalytic domains (type B, null alleles). Always trial biotin (safe) but manage expectations.",
        },
        {
            "therapy": "Levetiracetam (LEV)",
            "level": "B",
            "dose": "20–40 mg/kg/day in 2 divided doses",
            "rationale": "First-line AED for PC-associated seizures. No mitochondrial toxicity. No carnitine depletion. Safe metabolic profile in mitochondrial disorders. LEV preferred over VPA in ALL mitochondrial disease contexts.",
        },
        {
            "therapy": "Valproic Acid (VPA)",
            "level": "AVOID",
            "dose": "N/A — avoid in PC deficiency",
            "rationale": "VPA: (1) mitochondrial complex I/II inhibitor → worsens NADH excess and lactic acidosis; (2) reduces PC protein expression (transcriptional suppression); (3) carnitine depletion → metabolic stress. VPA is contraindicated in all mitochondrial energetics disorders including PC deficiency.",
        },
        {
            "therapy": "Ketogenic Diet (KD)",
            "level": "ABSOLUTE CONTRAINDICATION",
            "dose": "N/A — NEVER use in PC deficiency",
            "rationale": "KD floods mitochondria with Acetyl-CoA from FA β-oxidation. BUT without OAA (PC deficient), Acetyl-CoA CANNOT enter TCA (citrate synthase requires OAA). Result: Acetyl-CoA → ketones (β-OHB ↑) + NADH excess worsens → lactic acidosis CATASTROPHICALLY worsened. Fatal in type B. ABSOLUTE CI (contrast: KD is TREATMENT for PDH deficiency).",
        },
        {
            "therapy": "Fasting avoidance",
            "level": "A (EXTREME HAZARD)",
            "dose": "Max fasting: 3–4 h (infants); glucose polymers/D10 during illness; written sick-day protocol",
            "rationale": "Fasting → catabolism → muscle protein breakdown → more alanine/pyruvate substrate flood. Gluconeogenesis BLOCKED → cannot compensate → glucose falls rapidly → crisis. Fasting is EXTREME HAZARD in all phenotypes. Emergency glucose infusion protocol is mandatory.",
        },
    ]

    systemic_features = [
        {"feature": "Lactic acidosis (elevated anion gap)", "pct": 100, "note": "All patients; severity correlates with phenotype; type B may have pH <7.1"},
        {"feature": "Intellectual disability",              "pct": 68,  "note": "Predominant in type A/B; progressive if untreated; severity correlates with lactate burden"},
        {"feature": "Seizures",                             "pct": 63,  "note": "Type A/B; generalised TC and focal; infantile spasms (type B); AED LEV preferred"},
        {"feature": "Hypotonia",                            "pct": 62,  "note": "Generalised; especially type A/B; cerebral + peripheral component"},
        {"feature": "Hypoglycemia",                         "pct": 55,  "note": "Most severe in type B; gluconeogenesis blocked; brain deprived both glucose and ketones"},
        {"feature": "Hyperammonemia (secondary)",           "pct": 48,  "note": "OAA depletion → ASP depleted → urea cycle impaired; type B most severe; responds to glucose"},
        {"feature": "Hepatomegaly",                         "pct": 30,  "note": "Type B; glycogen depletion + lactic acid storage; hepatocellular dysfunction"},
        {"feature": "Pyramidal tract signs",                "pct": 28,  "note": "Type A/B; spasticity from white matter injury (lactic acidosis myelinopathy)"},
        {"feature": "Near-normal development (type C)",     "pct": 15,  "note": "Benign type C; episodic mild lactic acidosis; identified often in adulthood"},
    ]

    cohort_preview = _COHORT[:10]

    return {
        "biomarkers": biomarkers,
        "enzyme_mechanism": enzyme_mechanism,
        "variants": VARIANTS,
        "seizure_types": seizure_types,
        "treatments": treatments,
        "systemic_features": systemic_features,
        "cohort_preview": cohort_preview,
    }


def get_definitions():
    return {
        "PC (Pyruvate Carboxylase)": (
            "Mitochondrial matrix enzyme; 1178 aa per subunit; homotetrameric active form (~520 kDa). "
            "Gene: PC, chromosome 11q13.2, autosomal recessive. "
            "Reaction: Pyruvate + HCO₃⁻ + ATP → Oxaloacetate + ADP + Pi (biotin-dependent anaplerosis). "
            "Requires acetyl-CoA as allosteric ACTIVATOR (allosteric site on PT domain). "
            "4 domains: BC (biotin carboxylase), BCCP (biotin carrier, Lys1119), CT (carboxyl transferase), PT (tetramerization). "
            "HLCS biotinylates Lys1119 → active holoPC. OMIM Gene *608786."
        ),
        "PC Deficiency (Pyruvate Carboxylase Deficiency, PCD)": (
            "Autosomal recessive IEM (biallelic LOF in PC gene, 11q13.2). "
            "OMIM Disease #266150. Prevalence ~1:250,000 general; ~1:3,000 Ojibwa Cree (Canada). "
            "PC LOF → OAA CANNOT be synthesised from pyruvate → gluconeogenesis blocked + TCA starved → "
            "lactic acidosis (PRIMARY) + alanine ↑ + secondary hyperammonemia. "
            "3 phenotypes: Type A North American (50%, moderate), Type B French (35%, severe neonatal), "
            "Type C Benign (15%, episodic mild)."
        ),
        "Pyruvate Anaplerosis (What PC Does)": (
            "Anaplerosis = replenishment of TCA cycle intermediates. "
            "PC is the MAJOR anaplerotic enzyme providing OAA directly into TCA cycle entry. "
            "Without OAA from PC: citrate synthase cannot form citrate (OAA + Acetyl-CoA → citrate is blocked). "
            "TCA cycle 'starves' → fewer reducing equivalents → NADH:NAD⁺ ↑↑ → L:P ratio ↑↑↑. "
            "Also: without OAA, PEPCK cannot make PEP (first step of gluconeogenesis) → glucose synthesis blocked."
        ),
        "L:P Ratio (Lactate:Pyruvate Ratio) — Critical Differential": (
            "L:P ratio reflects cytoplasmic redox state (NADH:NAD⁺ ratio). "
            "PC deficiency: L:P > 20:1 (often 30–80:1 in type B). NADH EXCESS — TCA starved, NADH backs up. "
            "PDH deficiency: L:P < 10:1. Pyruvate CANNOT be oxidised but NADH is NOT in excess. "
            "L:P > 20:1 = 'secondary' lactate excess (from NADH excess — TCA/RC block) → suspect PC, respiratory chain. "
            "L:P < 10:1 = 'primary' pyruvate increase (from PDH block, PDHX, PDHA1) → suspect PDH complex. "
            "CLINICAL PEARL: bedside plasma L:P is the fastest way to differentiate PC from PDH deficiency."
        ),
        "Ketogenic Diet — ABSOLUTE CI in PC Deficiency (Contrast PDH)": (
            "KD floods mitochondria with Acetyl-CoA (via FA β-oxidation). "
            "Citrate synthase entry: Acetyl-CoA + OAA → Citrate — REQUIRES OAA. "
            "PC deficient → OAA absent → Acetyl-CoA CANNOT enter TCA. "
            "Result: Acetyl-CoA → HMGCS2/HMGCL → ketones; NADH excess from β-oxidation worsens; "
            "lactic acidosis worsens dramatically; brain deprived of both glucose and functional TCA. "
            "PC: KD = ABSOLUTE CONTRAINDICATION (life-threatening). "
            "PDH: KD = TREATMENT OPTION (Acetyl-CoA enters TCA bypassing PDH block; beneficial). "
            "EXAMINER PEARL: PC vs PDH — L:P ratio AND KD indication are the two critical differentials."
        ),
        "Biotin and PC — Partial vs Full Responsiveness": (
            "PC is biotin-dependent (HLCS biotinylates PC at Lys1119 BCCP domain). "
            "Some PC alleles (near BCCP domain: p.Arg451Cys, p.Ala956Thr) show partial lactate "
            "improvement with high-dose biotin (10–40 mg/day) — apoenzyme has some residual function. "
            "HLCS/BTD: biotin supplementation DRAMATICALLY corrects ALL 4 carboxylases (complete response). "
            "Isolated PC deficiency: biotin may PARTIALLY help SOME patients — NOT a diagnostic response. "
            "A full biotin response (all 4 carboxylases restored) = HLCS or BTD, NOT isolated PC deficiency. "
            "Partial response = isolated PC deficiency with biotin-proximal mutation (Level B trial always worthwhile)."
        ),
        "PC vs HLCS vs BTD — Biotin-Dependent Carboxylase Spectrum": (
            "All 3 conditions involve biotin-dependent carboxylases (PC, PCC, MCC, ACC). "
            "PC deficiency: ISOLATED (only PC is structurally defective); C5-OH NORMAL; C3 NORMAL; "
            "biotinidase NORMAL; biotin PARTIALLY helps some alleles. Lactic acidosis dominant. "
            "HLCS deficiency: ALL 4 carboxylases inactive simultaneously (HLCS cannot biotinylate any). "
            "C5-OH ↑ + C3 ↑ + lactic acidosis + skin rash + alopecia. Biotin 10–40 mg/day CORRECTS ALL. "
            "BTD deficiency: ALL 4 carboxylases fail INDIRECTLY (biotin recycling stopped). "
            "Same multi-carboxylase picture; biotinidase activity LOW (<30%); SNHL + optic atrophy (late). "
            "Biotin 5–10 mg/day CORRECTS ALL. KEY: PC=C5-OH NORMAL, C3 NORMAL; HLCS/BTD=C5-OH↑,C3↑."
        ),
        "Secondary Hyperammonemia in PC Deficiency": (
            "NH3 elevated in type A/B via OAA depletion mechanism (NOT primary UCD): "
            "OAA depleted (PC LOF) → mitochondrial GOT2 (Asp transaminase) cannot make aspartate "
            "(OAA + Glu → Asp + 2-OG; no OAA → no Asp). "
            "Aspartate depleted → ASS1 (argininosuccinate synthase) has no substrate (Cit + Asp → AAS). "
            "Urea cycle impaired SECONDARILY → NH3 rises. "
            "NH3 also elevated in type B from citrulline/aspartate mismatch. "
            "CRITICAL: PC hyperammonemia improves with IV glucose (OAA restored) — "
            "primary UCD NH3 does NOT improve with glucose alone (requires ammonia scavengers + protein restriction)."
        ),
        "Inheritance & Epidemiology": (
            "Autosomal recessive (AR); biallelic LOF in PC gene (11q13.2). "
            "~1:250,000 general population. Significant founder effect: ~1:3,000 Ojibwa Cree "
            "(Manitoba/Ontario, Canada) — p.Arg451Cys and p.Ala956Thr founder alleles. "
            "Also ~1:3,500 in Algonquian communities (same Cree founder alleles). "
            "Most Western patients: type A (moderate) or type B (severe neonatal). "
            "Type C (benign) may not be diagnosed until adulthood (episodic mild lactic acidosis). "
            "NBS: PC deficiency NOT reliably detected by standard NBS panels (C5-OH, C3 normal). "
            "Lactic acidosis on extended metabolic panel triggers workup."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN keys ===", list(get_breakdown().keys()))
    print("\n=== DEFINITIONS keys ===", list(get_definitions().keys()))
    print(f"\n✓ PC dashboard: {N_PATIENTS} patients, seed={SEED}")
