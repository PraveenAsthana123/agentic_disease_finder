#!/usr/bin/env python3
"""VLCAD (Very Long Chain Acyl-CoA Dehydrogenase Deficiency) Dashboard.

ACADVL gene encodes VERY LONG CHAIN ACYL-CoA DEHYDROGENASE (VLCAD):
  - 655 aa precursor (70 kDa mature monomer after signal cleavage); mitochondrial inner membrane
  - FAD-dependent; homodimeric (unlike MCAD/SCAD tetramers)
  - Catalyses: Long-chain acyl-CoA (C14–C20) + FAD → trans-2-enoyl-acyl-CoA + FADH2
  - Step 1 of mitochondrial beta-oxidation spiral for C14–C20 very long-chain fatty acids

VLCAD LOF → very long-chain acyl-CoA species CANNOT be dehydrogenated:
  Palmitoyl-CoA (C16)         ACCUMULATES → palmitoylcarnitine (C16) ↑
  Oleoyl-CoA (C18:1)          ACCUMULATES → oleoylcarnitine (C18:1) ↑
  Tetradecanoyl-CoA (C14)     ACCUMULATES → tetradecanoylcarnitine (C14) ↑
  Tetradecenoyl-CoA (C14:1)   ACCUMULATES → tetradecenoylcarnitine (C14:1) ↑↑ [PRIMARY NBS]
  → Beta-oxidation spiral ARRESTED at very long-chain stage
  → Cannot generate acetyl-CoA from C14–C20 fatty acids → ENERGY DEFICIT (cardiac + skeletal muscle)
  → Urine: 3-hydroxydicarboxylic acids (C14–C18 chain length); non-specific dicarboxylic acids

KEY FACTS (HIGHEST YIELD):
  1.  C14:1 (Tetradecenoylcarnitine) — PRIMARY NBS MARKER; >0.3 µmol/L tandem MS/MS
  2.  C14:1/C2 ratio — >0.07 highly specific for VLCAD; best discriminator from false positives
  3.  C14, C14:2, C16, C18:1 — also elevated; characteristic long-chain acylcarnitine profile
  4.  C8 NORMAL — KEY NEGATIVE vs MCAD (MCAD elevates C8; VLCAD does NOT)
  5.  CARDIOMYOPATHY — HALLMARK; hypertrophic or dilated; potentially fatal arrhythmias; absent in MCAD
  6.  RHABDOMYOLYSIS — exercise-induced; elevated CK (10,000–100,000 U/L); myoglobinuria; absent in MCAD
  7.  HYPOketotic HYPOGLYCAEMIA — fasting/illness trigger; same mechanism as MCAD but more severe
  8.  FASTING: ABSOLUTE CI — most preventable trigger; no overnight fasting in infants
  9.  KD: ABSOLUTE CI — long-chain fat worsens crisis; MCT (C8–C12) is actually HELPFUL (bypasses VLCAD)
 10.  MCT oil THERAPEUTIC — bypasses VLCAD block; MCAD metabolizes C8–C12; used as dietary fat source
 11.  Triheptanoin (C7 odd-chain) — Level B; anaplerotic; generates propionyl-CoA + acetyl-CoA; TCA support
 12.  VPA: HIGH RISK — inhibits FAO globally; depletes carnitine; worsens energy deficit
 13.  p.Val283Ala (c.848T>C) — most common mild/adult allele; retains ~25% residual VLCAD activity
 14.  ~1:40,000–1:80,000 — less common than MCAD (1:10,000–15,000)
 15.  NO urine glycine conjugates — no HG/SG/PPG (those are MCAD-specific pathognomonic markers)

OMIM Disease: #201475 (VLCAD deficiency — ACADVL)
OMIM Gene:    *609575 (ACADVL)
Chromosome:   17p13.1
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Protein:      ACADVL = 655 aa precursor; 70 kDa mature monomer; mitochondrial inner membrane; homodimeric
              FAD cofactor; FADH2 products fed into ETF/ETFDH electron transfer chain
Prevalence:   ~1:40,000–1:80,000 (general); second most common FAO disorder after MCAD

FATTY ACID BETA-OXIDATION — WHERE VLCAD FITS:
  Very long-chain fatty acid (C14–C20) + CoA → [Acyl-CoA synthetase/FATP] → Very long-chain acyl-CoA
  Very long-chain acyl-CoA → [VLCAD + FAD  BLOCKED ⚠] → trans-2-enoyl-acyl-CoA
  → [MTP/HADHB] → 3-hydroxyacyl-CoA → [MTP/HADHA] → 3-ketoacyl-CoA → [MTP thiolase] → acetyl-CoA + shorter acyl-CoA
  → Shortened acyl-CoA enters LCHAD (C12) → MCAD (C6–C12) → SCAD (C4–C6) spiral

  VLCAD block → C14–C20 acyl-CoAs CANNOT enter spiral → ACCUMULATE as long-chain acylcarnitines
  VLCAD block → FADH2 not generated from long-chain step → less mitochondrial electron flux
  VLCAD block → cardiac and skeletal muscle energy deficit (rely heavily on long-chain FAO)

VLCAD vs MCAD vs LCHAD:
  VLCAD (ACADVL, C14–C20): C14:1 primary NBS; cardiomyopathy + rhabdomyolysis; MCT helpful; homodimer
  MCAD  (ACADM, C6–C12):   C8 primary NBS; Reye-like; HG/SG/PPG pathognomonic urine; no cardiomyopathy
  LCHAD (HADHA, C12–C18):  3-OH-acylcarnitines elevated; maternal AFLP/HELLP; retinopathy; MTP complex

BIOMARKER PATTERN — VLCAD DEFICIENCY:
  C14:1 (Tetradecenoylcarnitine)      >0.3 µmol/L    — PRIMARY NBS MARKER ↑↑↑
  C14:1/C2 ratio                      >0.07           — highly specific discriminator ↑↑
  C14 (Tetradecanoylcarnitine)        >0.4 µmol/L    — elevated ↑↑
  C14:2 (Tetradecadienoylcarnitine)   >0.05 µmol/L   — elevated ↑
  C16 (Palmitoylcarnitine)            >4.0 µmol/L    — elevated ↑
  C18:1 (Oleoylcarnitine)             >2.0 µmol/L    — elevated ↑
  C0 (Free carnitine)                 LOW (secondary depletion from long-chain acylcarnitine conjugation)
  C8 (Octanoylcarnitine)              NORMAL — KEY NEGATIVE vs MCAD
  C8/C10 ratio                        NORMAL — KEY NEGATIVE vs MCAD (where ratio >2.0)
  3-OH-dicarboxylic acids (C14–C18)   ELEVATED — urine organic acids; non-specific
  Adipic/suberic/sebacic acids        ELEVATED non-specifically during crisis
  Blood glucose                       LOW during crisis (hypoketotic hypoglycaemia)
  Ketones (β-OHB)                     LOW/ABSENT despite hypoglycaemia — HALLMARK HYPOKETOSIS
  CK (creatine kinase)                MARKEDLY ELEVATED during rhabdomyolysis (10,000–100,000 U/L)
  Troponin I/T                        ELEVATED during cardiac crisis (cardiomyopathy / arrhythmia)
  BNP/NT-proBNP                       ELEVATED in cardiomyopathy phenotype
  Liver transaminases (ALT/AST)       ELEVATED during crisis (hepatopathy)

CARDIOMYOPATHY (hallmark distinguishing VLCAD from MCAD):
  Type:     Hypertrophic cardiomyopathy (HCM) most common; dilated (DCM) in severe cases
  Onset:    Usually first weeks-months of life in severe phenotype; may develop later in mild
  Risk:     Cardiac arrhythmias (VT/VF) → sudden cardiac death
  Mechanism: Cardiac myocytes rely HEAVILY on long-chain FAO (60–80% of cardiac energy)
             → VLCAD LOF = severe cardiac energy deficit → myocyte hypertrophy/dysfunction
  Monitoring: ECG, echocardiography every 6–12 months; Holter monitor
  Treatment: MCT oil (bypasses VLCAD) + fasting avoidance → often reverses/stabilises HCM
  Note:     Cardiomyopathy can be reversible with MCT diet — unlike most cardiomyopathies

RHABDOMYOLYSIS:
  Trigger:  Prolonged exercise, fasting, fever, cold, high-fat meal without MCT
  Mechanism: Skeletal muscle relies on long-chain FAO during sustained exercise → VLCAD LOF → energy deficit → cell necrosis
  Biomarkers: CK >10,000 U/L (up to 100,000+ U/L); myoglobinuria (dark urine); creatinine elevation (AKI risk)
  Treatment: IV fluids (prevent AKI from myoglobin) + glucose infusion + rest
  Prevention: Avoid prolonged fasting before exercise; warm-up protocol; MCT supplement pre-exercise

WHY MCT IS THERAPEUTIC (KEY EXAM CONCEPT — OPPOSITE OF MCAD):
  MCT oil provides C8 (caprylic) and C10 (capric) fatty acids
  C8–C10 do NOT need VLCAD — they bypass directly to MCAD (which is intact in VLCAD deficiency)
  → MCT provides usable fat-derived energy to cardiac and skeletal muscle
  → MCT is THERAPEUTIC in VLCAD (reduces long-chain fat burden, provides alternate fuel)
  → In MCAD: MCT would WORSEN crisis (MCAD is the blocked enzyme for C8–C12)
  → KD is ABSOLUTE CI in VLCAD because KD uses long-chain fat as primary fuel (worsens VLCAD block)
  → KD is ABSOLUTE CI in MCAD because KD requires fasting intervals (triggers hypoketotic crisis)

TREATMENT SUMMARY:
  MCT oil (Level A):                 Primary long-term dietary fat; bypasses VLCAD; cardiac rescue
  Avoid fasting (Absolute — Level A): Most preventable trigger; glucose infusion during illness
  Long-chain fat restriction (Level A): Reduce dietary saturated long-chain fat; replace with MCT
  Emergency protocol during illness: IV glucose 10% 8–10 mg/kg/min + MCT supplement
  Triheptanoin (C7 oil — Level B):  Anaplerotic; odd-chain; generates propionyl-CoA + acetyl-CoA for TCA
  Uncooked cornstarch:              Overnight fasting prevention; slow-release glucose
  L-carnitine: CONTROVERSIAL — may worsen if long-chain acylcarnitines accumulate (C16, C18:1)
  VPA: HIGH RISK — inhibits mitochondrial FAO; worsens energy deficit; avoid
  KD: ABSOLUTE CI — long-chain fat is the substrate blocked by VLCAD deficiency; catastrophic crisis

PHENOTYPES:
  Severe infantile (Cardiac):     20% — cardiomyopathy + hypoglycaemia in first weeks of life; fatal if untreated
  Childhood/Adolescent (Hepatic): 45% — episodic hypoketotic hypoglycaemia ± rhabdomyolysis; cardiomyopathy mild/absent
  Mild/Adult (Myopathic):         35% — exercise-induced rhabdomyolysis + myopathy in adulthood; residual VLCAD activity

COMMON VARIANTS (ACADVL):
  c.848T>C (p.Val283Ala):   25% mild/adult alleles; retains ~25% residual activity; adult onset myopathy
  c.1349G>A (p.Arg450His): 15% alleles; severe; near-complete LOF; infantile cardiomyopathy
  c.553G>A (p.Gly185Arg):  10% alleles; severe infantile; substrate-binding domain
  c.1A>G (p.Met1Val):       8% alleles; start codon; null allele; severe neonatal
  c.1097G>T (p.Gly366Val): 6% alleles; moderate; FAD-binding region; intermediate phenotype
  Other / compound het:     36% alleles; >200 private mutations known; variable phenotype
"""

import random

SEED = 271        # deterministic cohort (MCAD=269, ACAT1=267, PC=265; VLCAD=271)
N_PATIENTS = 40

random.seed(SEED)

PHENOTYPES = [
    ("Severe Infantile (Cardiac)",           0.20),
    ("Childhood/Hepatic (Episodic)",         0.45),
    ("Mild/Adult (Myopathic)",               0.35),
]

VARIANTS = [
    {"variant": "c.848T>C (p.Val283Ala)",   "freq": 25, "domain": "Substrate-binding domain",      "phenotype": "Mild/Adult", "note": "Retains ~25% residual VLCAD activity; most common in adult-onset myopathy; temperature-sensitive"},
    {"variant": "c.1349G>A (p.Arg450His)",  "freq": 15, "domain": "Catalytic core",                "phenotype": "Severe Infantile", "note": "Near-complete LOF; infantile cardiomyopathy; common Northern European severe allele"},
    {"variant": "c.553G>A (p.Gly185Arg)",  "freq": 10, "domain": "Substrate-binding β-sheet",     "phenotype": "Severe Infantile", "note": "Disrupts substrate channel geometry; very low residual activity"},
    {"variant": "c.1A>G (p.Met1Val)",       "freq":  8, "domain": "Start codon (null)",            "phenotype": "Severe / null",    "note": "No protein produced; complete LOF; neonatal crisis"},
    {"variant": "c.1097G>T (p.Gly366Val)", "freq":  6, "domain": "FAD-binding region",            "phenotype": "Moderate",         "note": "Impaired FAD binding; intermediate phenotype; hepatic-predominant"},
    {"variant": "Other / compound het",     "freq": 36, "domain": "Various (>200 known)",          "phenotype": "Variable",         "note": ">200 private pathogenic variants described; compound heterozygosity very common"},
]


def _weighted_choice(options):
    r = random.random()
    cum = 0.0
    for val, prob in options:
        cum += prob
        if r < cum:
            return val
    return options[-1][0]


def _make_cohort():
    cohort = []
    for i in range(N_PATIENTS):
        ph = _weighted_choice(PHENOTYPES)
        is_severe   = ph == "Severe Infantile (Cardiac)"
        is_episodic = ph == "Childhood/Hepatic (Episodic)"
        is_mild     = ph == "Mild/Adult (Myopathic)"

        onset_mo = (
            random.randint(0, 3)    if is_severe   else
            random.randint(6, 60)   if is_episodic else
            random.randint(120, 360)  # adult onset: 10–30 years
        )

        # C14:1 (tetradecenoylcarnitine) — PRIMARY NBS MARKER
        c14_1 = round(
            random.uniform(1.5, 8.0)   if is_severe   else
            random.uniform(0.5, 3.0)   if is_episodic else
            random.uniform(0.3, 1.5),
            2
        )

        # C14:1 / C2 ratio — highly specific
        c2 = round(random.uniform(8.0, 18.0), 1)
        c14_1_c2_ratio = round(c14_1 / c2, 3)

        # C14 (tetradecanoylcarnitine) — elevated
        c14 = round(
            random.uniform(0.8, 3.5)   if is_severe   else
            random.uniform(0.3, 1.5)   if is_episodic else
            random.uniform(0.2, 0.8),
            2
        )

        # C14:2 (tetradecadienoylcarnitine)
        c14_2 = round(
            random.uniform(0.10, 0.60) if is_severe   else
            random.uniform(0.05, 0.25) if is_episodic else
            random.uniform(0.03, 0.12),
            2
        )

        # C16 (palmitoylcarnitine)
        c16 = round(
            random.uniform(6.0, 20.0)  if is_severe   else
            random.uniform(3.0, 10.0)  if is_episodic else
            random.uniform(1.5, 5.0),
            1
        )

        # C18:1 (oleoylcarnitine)
        c18_1 = round(
            random.uniform(3.0, 12.0)  if is_severe   else
            random.uniform(1.5, 6.0)   if is_episodic else
            random.uniform(0.8, 3.0),
            1
        )

        # C8 (octanoylcarnitine) — NORMAL (KEY NEGATIVE vs MCAD)
        c8 = round(random.uniform(0.03, 0.14), 2)

        # Free carnitine C0 — depleted from long-chain acylcarnitine conjugation
        c0 = round(
            random.uniform(5, 18)   if is_severe   else
            random.uniform(10, 25)  if is_episodic else
            random.uniform(18, 38),
            1
        )

        # Blood glucose during crisis
        glucose = (
            round(random.uniform(0.8, 2.8), 1)  if is_severe   else
            round(random.uniform(1.5, 3.5), 1)  if is_episodic else
            round(random.uniform(3.5, 5.5), 1)   # mild: usually normal (rhabdo not hypoglycaemia)
        )

        # Beta-OHB — LOW (hypoketosis in cardiac/hepatic; normal in mild/myopathic)
        bohb = round(
            random.uniform(0.05, 0.30)  if is_severe   else
            random.uniform(0.08, 0.40)  if is_episodic else
            random.uniform(0.5, 2.5),   # mild: may have normal ketosis
            2
        )

        # CK (creatine kinase) — rhabdomyolysis marker
        ck = (
            random.randint(500, 5000)     if is_severe   else
            random.randint(1000, 20000)   if is_episodic else
            random.randint(5000, 100000)   # mild/adult: high during rhabdo episodes
        )

        # Troponin — cardiac
        troponin_elevated = (
            random.random() < 0.70  if is_severe   else
            random.random() < 0.20  if is_episodic else
            random.random() < 0.10
        )

        # Cardiomyopathy
        cardiomyopathy = (
            random.random() < 0.80  if is_severe   else
            random.random() < 0.15  if is_episodic else
            random.random() < 0.05
        )

        # Liver transaminases
        alt = (
            random.randint(100, 600)  if is_severe   else
            random.randint(50, 300)   if is_episodic else
            random.randint(20, 80)
        )

        # Variant assignment
        v_roll = random.random()
        if v_roll < 0.25:
            variant = VARIANTS[0]["variant"]
        elif v_roll < 0.40:
            variant = VARIANTS[1]["variant"]
        elif v_roll < 0.50:
            variant = VARIANTS[2]["variant"]
        elif v_roll < 0.58:
            variant = VARIANTS[3]["variant"]
        elif v_roll < 0.64:
            variant = VARIANTS[4]["variant"]
        else:
            variant = VARIANTS[5]["variant"]

        seizures = (
            random.random() < 0.60 if is_severe   else
            random.random() < 0.35 if is_episodic else
            random.random() < 0.10
        )

        # Treatment responses
        tx = ["MCT Oil (long-term dietary)", "Avoid Fasting Protocol", "Long-Chain Fat Restriction"]
        if is_severe or is_episodic:
            tx.append("IV Glucose 10%")
            tx.append("Emergency Protocol Card")
        if is_severe:
            tx.append("Cardiac Monitoring (Echo/ECG/Holter)")
        if is_mild:
            tx.append("Triheptanoin C7 Oil (anaplerosis)")
            tx.append("Exercise Warm-Up Protocol")
        if c0 < 15:
            tx.append("L-Carnitine (cautious — if C0 depleted only)")
        if ck > 10000:
            tx.append("IV Fluids (rhabdomyolysis — prevent AKI)")

        cohort.append({
            "id":               f"VLCAD-{i+1:03d}",
            "phenotype":        ph,
            "onset_mo":         onset_mo,
            "c14_1":            c14_1,
            "c14_1_c2_ratio":   c14_1_c2_ratio,
            "c2":               c2,
            "c14":              c14,
            "c14_2":            c14_2,
            "c16":              c16,
            "c18_1":            c18_1,
            "c8":               c8,
            "c0":               c0,
            "glucose":          glucose,
            "bohb":             bohb,
            "ck":               ck,
            "troponin_elevated": troponin_elevated,
            "cardiomyopathy":   cardiomyopathy,
            "alt":              alt,
            "variant":          variant,
            "seizures":         seizures,
            "treatments":       tx,
        })
    return cohort


_COHORT = None


def _get_cohort():
    global _COHORT
    if _COHORT is None:
        _COHORT = _make_cohort()
    return _COHORT


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_overview():
    cohort = _get_cohort()

    ph_counts = {}
    for p in cohort:
        ph_counts[p["phenotype"]] = ph_counts.get(p["phenotype"], 0) + 1

    variant_counts = {}
    for p in cohort:
        variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1

    seizure_n       = sum(1 for p in cohort if p["seizures"])
    cardiac_n       = sum(1 for p in cohort if p["cardiomyopathy"])
    rhabdo_n        = sum(1 for p in cohort if p["ck"] > 5000)
    troponin_n      = sum(1 for p in cohort if p["troponin_elevated"])
    severe_n        = ph_counts.get("Severe Infantile (Cardiac)", 0)
    episodic_n      = ph_counts.get("Childhood/Hepatic (Episodic)", 0)
    mild_n          = ph_counts.get("Mild/Adult (Myopathic)", 0)

    avg_c14_1       = round(sum(p["c14_1"] for p in cohort) / N_PATIENTS, 2)
    avg_c0          = round(sum(p["c0"] for p in cohort) / N_PATIENTS, 1)
    avg_ck          = round(sum(p["ck"] for p in cohort) / N_PATIENTS)

    top_variant     = max(variant_counts, key=variant_counts.get)
    top_v_pct       = round(100 * variant_counts[top_variant] / N_PATIENTS)

    return {
        "disease": "VLCAD Deficiency (Very Long Chain Acyl-CoA Dehydrogenase Deficiency)",
        "gene":    "ACADVL (VLCAD)",
        "locus":   "17p13.1",
        "omim_gene":    "609575",
        "omim_disease": "201475",
        "inheritance":  "Autosomal Recessive (AR)",
        "protein":      "ACADVL — 655 aa precursor; 70 kDa mature; mitochondrial inner membrane; homodimeric; FAD-dependent",
        "prevalence":   "~1:40,000–1:80,000; second most common FAO disorder after MCAD",
        "n_patients":   N_PATIENTS,
        "seed":         SEED,
        "kpis": {
            "total_patients":       N_PATIENTS,
            "severe_infantile_n":   severe_n,
            "episodic_n":           episodic_n,
            "mild_myopathic_n":     mild_n,
            "seizures_n":           seizure_n,
            "cardiomyopathy_n":     cardiac_n,
            "rhabdomyolysis_n":     rhabdo_n,
            "troponin_elevated_n":  troponin_n,
            "avg_c14_1_umol":       avg_c14_1,
            "avg_c0_umol":          avg_c0,
            "avg_ck_u_l":          avg_ck,
        },
        "phenotype_distribution": ph_counts,
        "top_variant":      top_variant,
        "top_variant_pct":  top_v_pct,
        "primary_nbs_marker":       "C14:1 (Tetradecenoylcarnitine) >0.3 µmol/L",
        "secondary_nbs_markers":    ["C14", "C14:2", "C16", "C18:1"],
        "discriminating_ratio":     "C14:1/C2 >0.07 — highly specific for VLCAD",
        "key_negatives_vs_mcad":    ["C8 NORMAL", "C8/C10 ratio NORMAL or LOW", "No HG/SG/PPG in urine"],
        "absolute_ci":              ["Fasting", "Ketogenic Diet", "Long-chain fat overload"],
        "high_risk_drugs":          ["VPA (Valproic acid)"],
        "therapeutic_fat":          "MCT oil (C8–C10 bypasses VLCAD; metabolized by intact MCAD)",
        "first_line_treatment":     "MCT oil + Avoid fasting + Long-chain fat restriction",
        "hallmark_feature":         "Cardiomyopathy (absent in MCAD) + Rhabdomyolysis",
        "clinical_summary": (
            "VLCAD deficiency (ACADVL) is the second most common FAO disorder (~1:40,000–80,000). "
            "ACADVL encodes very long chain acyl-CoA dehydrogenase (655 aa precursor; FAD-dependent; homodimeric) "
            "— Step 1 of beta-oxidation for C14–C20 very long-chain fatty acids. "
            "VLCAD LOF → C14:1 (tetradecenoylcarnitine) accumulates → PRIMARY NBS marker. "
            "HALLMARK: cardiomyopathy (absent in MCAD) + exercise-induced rhabdomyolysis. "
            "Clinical: HYPOketotic hypoglycaemia (fasting/illness) + cardiac arrhythmias + CK elevation. "
            "KEY DISTINCTION vs MCAD: C8 is NORMAL (MCAD elevates C8); MCT is THERAPEUTIC (bypasses VLCAD); "
            "no pathognomonic urine glycine conjugates (HG/SG/PPG are MCAD-specific). "
            "KD is ABSOLUTE CI in VLCAD (long-chain fat floods the blocked enzyme). "
            "MCT oil is the cornerstone of long-term dietary management — reverses cardiomyopathy in many cases."
        ),
        "key_negatives": {
            "c8_normal":              "NORMAL — KEY NEGATIVE vs MCAD (MCAD elevates C8 as PRIMARY NBS marker; VLCAD does not)",
            "c8_c10_ratio_normal":    "NORMAL or LOW — KEY NEGATIVE vs MCAD (MCAD ratio >2.0)",
            "no_hg_sg_ppg":           "ABSENT — no urine glycine conjugates (HG/SG/PPG are MCAD-specific pathognomonic markers)",
            "c3_normal":              "NORMAL — KEY NEGATIVE vs PA/MMA (propionyl-CoA not involved in VLCAD)",
            "mma_absent":             "ABSENT — KEY NEGATIVE vs MMUT deficiency",
        },
    }


def get_breakdown():
    cohort = _get_cohort()
    patients = cohort[:10]  # representative subset for detailed view

    biomarkers = {
        "c14_1": {
            "label":     "C14:1 (Tetradecenoylcarnitine)",
            "normal":    "<0.15 µmol/L",
            "status":    "ELEVATED >0.3 µmol/L — PRIMARY NBS MARKER",
            "direction": "↑↑↑",
            "color":     "danger",
            "rationale": (
                "C14:1 (tetradecenoylcarnitine) is the PRIMARY tandem MS/MS newborn screening marker for VLCAD deficiency. "
                "Tetradecenoyl-CoA (C14:1) is the preferred substrate of VLCAD; when VLCAD is absent, C14:1-CoA "
                "accumulates → transferred to carnitine → C14:1 detected on NBS filter paper. "
                "NBS cut-off: >0.3–0.5 µmol/L (lab-specific). Crisis values: 1.0–8.0+ µmol/L. "
                "C14:1 is more specific for VLCAD than C14 or C16 alone."
            ),
        },
        "c14_1_c2_ratio": {
            "label":     "C14:1 / C2 Ratio (Tetradecenoylcarnitine / Acetylcarnitine)",
            "normal":    "<0.05",
            "status":    "ELEVATED >0.07 — HIGHLY SPECIFIC discriminator for VLCAD",
            "direction": "↑↑",
            "color":     "danger",
            "rationale": (
                "The C14:1/C2 ratio is the most specific discriminator for true VLCAD deficiency vs false-positive NBS. "
                "C2 (acetylcarnitine) provides a denominator that normalises for carnitine status. "
                "C14:1/C2 >0.07 combined with C14:1 >0.3 µmol/L = strong evidence for VLCAD. "
                "The ratio helps distinguish from maternal carnitine depletion or other causes of mild C14:1 elevation. "
                "Some reference laboratories use C14:1/C2 as the primary VLCAD screen rather than C14:1 alone."
            ),
        },
        "c14": {
            "label":     "C14 (Tetradecanoylcarnitine / Myristoylcarnitine)",
            "normal":    "<0.25 µmol/L",
            "status":    "ELEVATED >0.4 µmol/L — secondary accumulation",
            "direction": "↑↑",
            "color":     "danger",
            "rationale": (
                "Tetradecanoyl-CoA (C14) accumulates in VLCAD deficiency because VLCAD also acts on C14. "
                "C14 elevation is supportive but less specific than C14:1. "
                "The combination C14 + C14:1 + C16 + C18:1 = characteristic VLCAD long-chain acylcarnitine profile. "
                "Note: in MCAD deficiency C14 is NORMAL (MCAD acts on C6–C12 only)."
            ),
        },
        "c16": {
            "label":     "C16 (Palmitoylcarnitine)",
            "normal":    "<2.5 µmol/L",
            "status":    "ELEVATED >4.0 µmol/L — supportive marker",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "Palmitoyl-CoA (C16) is a major VLCAD substrate; when VLCAD is absent, C16-CoA accumulates → palmitoylcarnitine elevated. "
                "C16 elevation is supportive of VLCAD but less specific (also elevated in LCHAD/MTP deficiency). "
                "C16/C14:1 ratio helps distinguish VLCAD from LCHAD (where 3-OH-acylcarnitines are more prominent)."
            ),
        },
        "c18_1": {
            "label":     "C18:1 (Oleoylcarnitine)",
            "normal":    "<1.5 µmol/L",
            "status":    "ELEVATED >2.0 µmol/L — supportive marker",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "Oleoyl-CoA (C18:1) is the dominant long-chain fatty acid in myelin and cell membranes; "
                "VLCAD also processes this substrate. "
                "C18:1 (oleoylcarnitine) elevation combined with C14:1 + C16 = full VLCAD profile. "
                "Helpful for distinguishing from LCHAD (where 3-OH-C18:1 is the discriminating marker)."
            ),
        },
        "c8": {
            "label":     "C8 (Octanoylcarnitine)",
            "normal":    "<0.15 µmol/L",
            "status":    "NORMAL (<0.15 µmol/L) — KEY NEGATIVE vs MCAD",
            "direction": "→ NORMAL",
            "color":     "success",
            "rationale": (
                "C8 (octanoylcarnitine) is the PRIMARY NBS marker for MCAD deficiency, not VLCAD. "
                "In VLCAD deficiency, C8 is NORMAL because MCAD (which processes C6–C12) is intact. "
                "C8 NORMAL is the most important exam KEY NEGATIVE vs MCAD. "
                "If both C8 and C14:1 are elevated: consider COMBINED FAO defect or MADD (ETFA/ETFB/ETFDH deficiency). "
                "VLCAD LOF affects only C14–C20 substrates; the medium-chain (C6–C12) pathway remains functional."
            ),
        },
        "c0": {
            "label":     "C0 (Free Carnitine)",
            "normal":    "25–50 µmol/L",
            "status":    "LOW — secondary depletion from long-chain acylcarnitine conjugation",
            "direction": "↓",
            "color":     "warning",
            "rationale": (
                "Free carnitine is depleted in VLCAD deficiency because very long-chain acyl-CoAs "
                "are conjugated to carnitine (forming C14, C14:1, C16, C18:1 acylcarnitines) → depletes free carnitine pool. "
                "L-carnitine supplementation in VLCAD is CONTROVERSIAL — may worsen if long-chain acylcarnitines accumulate "
                "(C16-carnitine and C18:1-carnitine are potentially cardiotoxic). "
                "Most guidelines: only supplement L-carnitine if C0 <10 µmol/L AND symptomatic."
            ),
        },
        "ck": {
            "label":     "CK (Creatine Kinase) — Rhabdomyolysis Marker",
            "normal":    "<200 U/L",
            "status":    "MARKEDLY ELEVATED during rhabdomyolysis episodes (10,000–100,000+ U/L)",
            "direction": "↑↑↑ (episodic)",
            "color":     "danger",
            "rationale": (
                "CK is markedly elevated during rhabdomyolysis episodes in VLCAD deficiency. "
                "Trigger: sustained exercise, prolonged fasting, fever, cold exposure. "
                "Mechanism: skeletal muscle energy deficit → myocyte necrosis → CK release. "
                "CK >10,000 U/L = severe rhabdomyolysis; >50,000 U/L = acute kidney injury risk. "
                "Treatment: IV fluid hydration (prevent AKI from myoglobin precipitation in tubules) + glucose infusion + rest. "
                "CK elevation distinguishes myopathic VLCAD from MCAD (MCAD: CK normal or mildly elevated)."
            ),
        },
        "glucose": {
            "label":     "Blood Glucose",
            "normal":    "3.5–6.0 mmol/L",
            "status":    "LOW during crisis (<3.0 mmol/L) — hypoketotic hypoglycaemia",
            "direction": "↓ (crisis)",
            "color":     "danger",
            "rationale": (
                "Hypoketotic hypoglycaemia in VLCAD: fasting depletes glucose stores → liver cannot oxidise "
                "long-chain fatty acids (VLCAD blocked) → cannot generate ketones → brain deprived of both fuels. "
                "Treatment: IV glucose 10% at 8–10 mg/kg/min immediately reverses hypoglycaemia. "
                "MCT oil (C8–C10, intact MCAD pathway) can be given orally alongside glucose. "
                "Note: hypoglycaemia is more prominent in severe infantile and episodic phenotypes; "
                "mild/myopathic adults may not experience hypoglycaemia (predominant issue = rhabdomyolysis)."
            ),
        },
    }

    treatments = {
        "mct_oil": {
            "label": "MCT Oil (Medium-Chain Triglycerides — C8/C10)",
            "level": "Level A — Primary long-term dietary fat",
            "rationale": (
                "MCT oil provides C8 (caprylic acid) and C10 (capric acid) — these bypass VLCAD completely. "
                "C8–C10 enter as octanoyl-CoA and decanoyl-CoA, processed by intact MCAD. "
                "MCT is the cornerstone of long-term dietary management: provides usable fat-derived energy "
                "to cardiac and skeletal muscle without needing VLCAD. "
                "MCT has reversed hypertrophic cardiomyopathy in many severe infantile cases. "
                "Dose: 1.5–3 g/kg/day; give with meals; titrate to tolerance."
            ),
        },
        "avoid_fasting": {
            "label": "Avoid Fasting (Absolute — Level A)",
            "level": "Level A — Most preventable trigger",
            "rationale": (
                "Fasting forces long-chain FAO as the primary energy source — this is the blocked pathway in VLCAD. "
                "Fasting → VLCAD blocked → no long-chain acetyl-CoA → no ketones → no alternative fuel → crisis. "
                "Protocol: never fast >4h (infants <6 mo), >6h (infants 6–12 mo), >8h (children), >10h (adults). "
                "During illness/surgery: IV glucose 10% at 8–10 mg/kg/min (prevents catabolism). "
                "Emergency protocol card mandatory for all VLCAD patients."
            ),
        },
        "lc_fat_restriction": {
            "label": "Long-Chain Fat Restriction (Level A)",
            "level": "Level A — Reduces substrate load on blocked enzyme",
            "rationale": (
                "Restricting dietary long-chain saturated fat (C14–C20) directly reduces the substrate load on the "
                "deficient VLCAD enzyme → reduces acylcarnitine accumulation → reduces cardiomyopathy/rhabdomyolysis risk. "
                "Long-chain fat should be <10–15% of total energy (vs normal 30–35%). "
                "Replaced by MCT oil (C8–C10) as the primary fat source. "
                "This is the OPPOSITE of MCAD management (where fat restriction is NOT needed)."
            ),
        },
        "triheptanoin": {
            "label": "Triheptanoin (C7 Anaplerotic Oil — Level B)",
            "level": "Level B — Anaplerotic substrate; emerging therapy",
            "rationale": (
                "Triheptanoin is a triglyceride of heptanoic acid (C7 — odd-chain). "
                "Metabolized to propionyl-CoA (C3) + acetyl-CoA: propionyl-CoA enters TCA as succinyl-CoA (anaplerosis). "
                "Replenishes TCA intermediates depleted by the VLCAD block → supports cardiac energy production. "
                "Clinical trials show reduction in cardiomyopathy events and rhabdomyolysis frequency. "
                "Available commercially as UX007 (Ultragenyx); approved for LCFAO disorders in some jurisdictions."
            ),
        },
        "vpa_avoid": {
            "label": "VPA (Valproic Acid) — HIGH RISK; Avoid",
            "level": "HIGH RISK — Avoid (no absolute CI classification but strong evidence of harm)",
            "rationale": (
                "VPA inhibits mitochondrial FAO globally via multiple mechanisms: "
                "(1) Valproyl-CoA competes with long-chain acyl-CoA at carnitine acyltransferase; "
                "(2) VPA depletes carnitine (forms valproylcarnitine); "
                "(3) VPA inhibits electron transfer chain Complex I and IV. "
                "In VLCAD deficiency: VPA worsens the already-impaired long-chain FAO → acute energy deficit → crisis. "
                "Substitute AEDs: levetiracetam, lamotrigine, oxcarbazepine preferred."
            ),
        },
        "kd_absolute_ci": {
            "label": "Ketogenic Diet — ABSOLUTE CONTRAINDICATION",
            "level": "ABSOLUTE CI — Catastrophic in VLCAD",
            "rationale": (
                "The ketogenic diet is ABSOLUTE CI in VLCAD deficiency — unique exam pearl. "
                "KD uses long-chain fat (butter, cream, oils) as primary fuel → floods VLCAD → "
                "massively increases C14:1 + C16 + C18:1 → cardiomyopathy crisis + rhabdomyolysis. "
                "MCT-based variants are also ABSOLUTE CI (MCT provides C8–C10 which do bypass VLCAD, "
                "but true KD formulations use long-chain fat). "
                "Contrast: KD is ABSOLUTE CI in VLCAD (long-chain fat substrate floods blocked enzyme) "
                "AND in MCAD (KD requires fasting intervals → hypoketotic crisis). "
                "Both are absolute CI but for DIFFERENT mechanistic reasons."
            ),
        },
    }

    return {
        "n_patients":   N_PATIENTS,
        "patients":     [
            {
                "id":            p["id"],
                "phenotype":     p["phenotype"],
                "onset_mo":      p["onset_mo"],
                "c14_1":         p["c14_1"],
                "c14_1_c2":      p["c14_1_c2_ratio"],
                "c14":           p["c14"],
                "c16":           p["c16"],
                "c18_1":         p["c18_1"],
                "c8":            p["c8"],
                "c0":            p["c0"],
                "glucose":       p["glucose"],
                "ck":            p["ck"],
                "cardiomyopathy": p["cardiomyopathy"],
                "variant":       p["variant"],
                "seizures":      p["seizures"],
                "treatments":    p["treatments"],
            }
            for p in patients
        ],
        "biomarkers":   biomarkers,
        "treatments":   treatments,
        "variants":     VARIANTS,
        "phenotype_profiles": {
            "Severe Infantile (Cardiac)": {
                "prevalence": "~20%",
                "onset": "Neonatal to 3 months",
                "features": "Hypertrophic/dilated cardiomyopathy; cardiac arrhythmias; hypoketotic hypoglycaemia; high C14:1 >2 µmol/L",
                "prognosis": "Fatal without MCT oil + fasting avoidance; cardiomyopathy often reversible with treatment",
                "key_point": "Cardiomyopathy can be REVERSED with MCT diet — monitor echocardiogram closely",
            },
            "Childhood/Hepatic (Episodic)": {
                "prevalence": "~45%",
                "onset": "6 months to 5 years",
                "features": "Episodic hypoketotic hypoglycaemia; hepatomegaly; ± mild cardiomyopathy; rhabdomyolysis",
                "prognosis": "Good with dietary management; crisis mortality rare if protocol followed",
                "key_point": "Most common VLCAD phenotype; NBS detects pre-symptomatically",
            },
            "Mild/Adult (Myopathic)": {
                "prevalence": "~35%",
                "onset": "10–30 years (or later)",
                "features": "Exercise-induced rhabdomyolysis; CK >10,000 U/L; myalgia; myoglobinuria; no hypoglycaemia",
                "prognosis": "Generally good; acute kidney injury during rhabdo episodes is main risk",
                "key_point": "p.Val283Ala allele most common; residual VLCAD activity ~25%; often missed without NBS",
            },
        },
    }


def get_definitions():
    return {
        "disease":    "VLCAD Deficiency (Very Long Chain Acyl-CoA Dehydrogenase Deficiency)",
        "gene":       "ACADVL — encodes VLCAD (655 aa precursor; 70 kDa mature monomer; mitochondrial inner membrane; FAD-dependent homodimer)",
        "locus":      "17p13.1",
        "omim_gene":  "*609575 (ACADVL gene)",
        "omim_disease": "#201475 (VLCAD deficiency)",
        "inheritance": "Autosomal Recessive (AR); biallelic LOF required",
        "prevalence": "~1:40,000–1:80,000 general population; second most common FAO disorder after MCAD",
        "enzyme_function": (
            "VLCAD catalyses Step 1 of mitochondrial beta-oxidation for very long-chain (C14–C20) fatty acids: "
            "Long-chain acyl-CoA + FAD → trans-2-enoyl-acyl-CoA + FADH2. "
            "VLCAD acts on C14–C20 substrates; MCAD (C6–C12) and SCAD (C4–C6) handle shorter chains. "
            "VLCAD is located on the mitochondrial INNER MEMBRANE (unlike MCAD/SCAD which are in the matrix). "
            "FADH2 from VLCAD feeds into the ETF/ETFDH electron transfer chain → Complex III."
        ),
        "pathomechanism": (
            "VLCAD LOF → C14–C20 acyl-CoAs CANNOT be dehydrogenated → accumulate → transfer to carnitine → "
            "long-chain acylcarnitines in blood (C14:1, C14, C16, C18:1). "
            "Energy deficit in cardiac and skeletal muscle (rely heavily on long-chain FAO for fuel). "
            "Long-chain acylcarnitines (C16, C18:1) may be directly cardiotoxic → arrhythmias. "
            "Hypoketotic hypoglycaemia: cannot generate acetyl-CoA → no ketones → brain deprived during fasting. "
            "Rhabdomyolysis: skeletal muscle ATP depletion during exercise/fasting → myocyte necrosis."
        ),
        "nbs_markers": {
            "primary":    "C14:1 (Tetradecenoylcarnitine) >0.3 µmol/L — tandem MS/MS newborn screening",
            "secondary":  ["C14", "C14:2", "C16", "C18:1"],
            "best_ratio": "C14:1/C2 >0.07 — most specific discriminator for true VLCAD vs false positive",
            "key_negative": "C8 NORMAL (MCAD elevates C8; VLCAD does not — most important exam differentiator)",
        },
        "clinical_triad": ["Cardiomyopathy (±arrhythmia)", "Hypoketotic hypoglycaemia", "Rhabdomyolysis (CK >10,000 U/L)"],
        "hallmark": "Cardiomyopathy (absent in MCAD) — hypertrophic or dilated; reversible with MCT diet",
        "key_differentials": {
            "vs_MCAD": "C8 NORMAL (MCAD=C8 elevated); C14:1 elevated (MCAD=C14 normal); cardiomyopathy present (MCAD=absent); MCT HELPFUL (MCAD=neutral); no HG/SG/PPG (MCAD=pathognomonic)",
            "vs_LCHAD": "No 3-OH-acylcarnitines (LCHAD=3-OH-C16, 3-OH-C18:1 elevated); no retinopathy; no maternal AFLP/HELLP; MTP complex intact",
            "vs_MADD":  "C8 normal in VLCAD (MADD=C4-C18 all elevated); riboflavin may help MADD; MADD affects ETF/ETFDH not VLCAD",
        },
        "treatment_overview": {
            "first_line":    "MCT oil (C8–C10) + fasting avoidance + long-chain fat restriction",
            "emergency":     "IV glucose 10% 8–10 mg/kg/min; IV fluids if rhabdomyolysis; MCT orally",
            "absolute_ci":   ["Fasting (same as MCAD)", "Ketogenic diet (long-chain fat floods VLCAD — catastrophic)"],
            "high_risk":     ["VPA (inhibits FAO; depletes carnitine)"],
            "mct_rationale": "MCT bypasses VLCAD block — C8/C10 processed by intact MCAD; cornerstone of therapy",
            "monitoring":    "Echocardiogram + ECG every 6–12 months; CK during exercise-related symptoms; acylcarnitine profile quarterly",
        },
        "key_exam_pearls": [
            "C14:1 (not C8) is the primary NBS marker for VLCAD — C8 is for MCAD",
            "Cardiomyopathy is the HALLMARK of VLCAD — absent in MCAD (major clinical differentiator)",
            "MCT oil is THERAPEUTIC in VLCAD (bypasses VLCAD via intact MCAD) — do not confuse with MCAD where MCT is NOT specifically needed",
            "KD is ABSOLUTE CI in VLCAD (long-chain fat is the blocked substrate) AND in MCAD (fasting intervals) — but for different mechanistic reasons",
            "Cardiomyopathy in VLCAD can be REVERSED with MCT diet — unlike most inherited cardiomyopathies",
            "C8 NORMAL in VLCAD is the most important KEY NEGATIVE vs MCAD on exam",
            "Rhabdomyolysis distinguishes myopathic VLCAD from MCAD (MCAD: Reye-like, not rhabdomyolysis)",
            "L-carnitine CONTROVERSIAL in VLCAD (unlike MCAD where it's used if C0 depleted) — may trap long-chain acylcarnitines",
            "p.Val283Ala (c.848T>C) — most common mild allele; adult-onset myopathy; 25% residual activity",
            "VLCAD is HOMODIMERIC (inner membrane); MCAD is HOMOTETRAMERIC (matrix) — structural exam point",
        ],
        "variants": VARIANTS,
        "phenotype_summary": {
            "severe_infantile":    "20% — neonatal cardiomyopathy + hypoglycaemia; high C14:1; reversible with MCT",
            "episodic_hepatic":    "45% — childhood episodic hypoglycaemia ± rhabdomyolysis; most common phenotype",
            "mild_myopathic":      "35% — adult exercise-induced rhabdomyolysis; p.Val283Ala common; often NBS-detected",
        },
        "chromosome": "17p13.1",
        "protein_class": "Acyl-CoA dehydrogenase (ACAD family); FAD-dependent; mitochondrial inner membrane; homodimeric",
        "related_disorders": [
            "MCAD deficiency (ACADM, C6–C12, C8 primary NBS, no cardiomyopathy)",
            "LCHAD/MTP deficiency (HADHA/HADHB, 3-OH-acylcarnitines, retinopathy, maternal AFLP)",
            "MADD/GA2 (ETFA/ETFB/ETFDH, all acylcarnitines elevated C4–C18, riboflavin-responsive)",
            "SCAD deficiency (ACADS, C4, usually benign)",
        ],
    }
