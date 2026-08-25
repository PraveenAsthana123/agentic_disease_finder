#!/usr/bin/env python3
"""MCAD (Medium-Chain Acyl-CoA Dehydrogenase Deficiency) Dashboard.

ACADM gene encodes MEDIUM-CHAIN ACYL-CoA DEHYDROGENASE (MCAD):
  - 421 aa (mature processed form); mitochondrial matrix; FAD-dependent; homotetrameric
  - Catalyses: Medium-chain acyl-CoA (C6–C12) + FAD → trans-2-enoyl-acyl-CoA + FADH2
  - Step 1 of mitochondrial beta-oxidation spiral for C6–C12 fatty acids

MCAD LOF → medium-chain acyl-CoA species CANNOT be dehydrogenated:
  Octanoyl-CoA (C8)       ACCUMULATES → octanoylcarnitine (C8) ↑↑ [PRIMARY NBS]
  Hexanoyl-CoA (C6)       partially accumulates → hexanoylcarnitine (C6) ↑
  Decanoyl-CoA (C10)      partially accumulates → decanoylcarnitine (C10) ↑
  cis-4-Decenoyl-CoA (C10:1) accumulates → C10:1 ↑
  → Beta-oxidation spiral ARRESTED at medium-chain stage
  → Cannot generate acetyl-CoA from C6–C12 fatty acids → ENERGY DEFICIT during fasting
  → Urine: hexanoylglycine (HG) + suberylglycine (SG) + phenylpropionylglycine (PPG) [PATHOGNOMONIC]

KEY FACTS (HIGHEST YIELD):
  1. C8 (octanoylcarnitine) — PRIMARY NBS MARKER; >0.3–0.5 µmol/L (expanded NBS tandem MS/MS)
  2. C8/C10 ratio — elevated; helps distinguish MCAD from VLCAD
  3. Hexanoylglycine (HG) — urine PATHOGNOMONIC; glycine conjugate of hexanoyl-CoA
  4. Suberylglycine (SG) — urine PATHOGNOMONIC; dicarboxylyl glycine conjugate
  5. Phenylpropionylglycine (PPG) — urine PATHOGNOMONIC; from intestinal bacterial fermentation of C6 acids
  6. Dicarboxylic acids (adipic, suberic, sebacic) — NON-SPECIFIC; also seen in fasting; less useful
  7. FREE CARNITINE (C0) LOW — secondary depletion; C8 + carnitine conjugate depletes C0
  8. FASTING: ABSOLUTE CI — primary trigger; illness + fasting → Reye-like crisis
  9. VPA: HIGH RISK — inhibits FAO; depletes carnitine; worsens medium-chain block
 10. c.985A>G (p.Lys329Glu) — FOUNDER MUTATION; 80–90% Northern European alleles
 11. HYPOKETOTIC HYPOGLYCAEMIA — HALLMARK; cannot generate ketones from FAO during fasting
 12. C14/C16/C18 NORMAL — KEY NEGATIVE vs VLCAD (which elevates long-chain acylcarnitines)
 13. MOST COMMON FAO DISORDER — 1:10,000–15,000 Northern European; 1:20,000 worldwide
 14. MAJORITY ASYMPTOMATIC on NBS — crisis only with fasting + illness (usually avoidable)
 15. NO DIETARY FAT RESTRICTION — medium-chain fat restriction is WRONG; avoid fasting instead

OMIM Disease: #201450 (MCAD deficiency)
OMIM Gene:    *607008 (ACADM)
Chromosome:   1p31.1
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Protein:      ACADM = 421 aa (mature); mitochondrial matrix; FAD-dependent; homotetrameric
              Active site: Glu376 (catalytic base); FAD cofactor (FADH2 products enter ETF/ETFDH chain)
Prevalence:   ~1:10,000–15,000 Northern European; ~1:20,000 worldwide; most common FAO disorder

FATTY ACID BETA-OXIDATION — WHERE MCAD FITS:
  Fatty acid (C6–C12) + CoA → [Acyl-CoA synthetase] → Medium-chain acyl-CoA
  Medium-chain acyl-CoA → [MCAD + FAD BLOCKED ⚠] → trans-2-enoyl-acyl-CoA
  → [EHHADH / HADH] → 3-hydroxyacyl-CoA → [HADH] → 3-ketoacyl-CoA → [thiolase] → acetyl-CoA + shorter acyl-CoA

  MCAD block → medium-chain acyl-CoAs CANNOT enter spiral → ACCUMULATE as acylcarnitines
  MCAD block → FADH2 not generated → less mitochondrial electron flux → energy deficit
  MCAD block → NO acetyl-CoA from C6–C12 → cannot generate ketones during fasting

VLCAD vs MCAD vs SCAD:
  VLCAD (ACADVL, C14–C18): C14, C14:1, C16, C18:1 elevated; long-chain predominant; cardiomyopathy
  MCAD  (ACADM, C6–C12):   C8 dominant NBS marker; C8/C10 ratio elevated; hepatopathy + hypoglycaemia
  SCAD  (ACADS, C4–C6):    C4 (butyrylcarnitine) elevated; usually benign/asymptomatic

BIOMARKER PATTERN — MCAD DEFICIENCY:
  C8 (Octanoylcarnitine)          >0.3 µmol/L  — PRIMARY NBS MARKER ↑↑↑
  C8/C10 ratio                    >2.0          — elevated; distinguishes MCAD from VLCAD (where C14 dominant)
  C6 (Hexanoylcarnitine)          >0.10 µmol/L — elevated ↑
  C10:1 (cis-4-Decenoylcarnitine) >0.05 µmol/L — elevated ↑
  C10 (Decanoylcarnitine)         >0.15 µmol/L — elevated ↑
  C0 (Free carnitine)             LOW (secondary depletion)
  C14, C16, C18                   NORMAL — KEY NEGATIVE vs VLCAD
  Hexanoylglycine (HG)            >3 mmol/mol Cr — PATHOGNOMONIC urine marker ↑↑
  Suberylglycine (SG)             >5 mmol/mol Cr — PATHOGNOMONIC ↑↑
  Phenylpropionylglycine (PPG)    >2 mmol/mol Cr — PATHOGNOMONIC ↑
  Dicarboxylic acids              ELEVATED non-specifically (adipic, suberic, sebacic) — NON-SPECIFIC
  Blood glucose                   LOW during crisis (hypoketotic hypoglycaemia)
  Ketones (β-OHB)                 LOW/ABSENT despite hypoglycaemia — HALLMARK HYPOKETOSIS
  Liver transaminases             ELEVATED during crisis (hepatopathy)
  NH3                             MILDLY elevated during crisis (Reye-like)

WHY HYPOKETOTIC HYPOGLYCAEMIA:
  During fasting: liver breaks down fatty acids → C6–C12 CANNOT be oxidised (MCAD blocked)
  → Cannot generate acetyl-CoA from medium-chain FA → cannot generate ketones via HMG-CoA pathway
  → Blood glucose falls (no gluconeogenesis from acetyl-CoA) → HYPOGLYCAEMIA
  → No ketone formation → HYPOKETOSIS (ketones inappropriately low for degree of hypoglycaemia)
  HYPOketotic hypoglycaemia = hallmark of FAO disorders (contrast: HYPERketotic in OA disorders)

WHY FASTING IS ABSOLUTE CI:
  Short fasting → liver switches to FA oxidation → MCAD blocked → energy deficit → hypoglycaemia + hypoketosis
  Triggers: overnight fasting, intercurrent illness (anorexia), vomiting, surgery
  Emergency protocol: IV glucose 10 mg/kg/min (anti-catabolic) during any illness + anorexia
  Regular feeding frequency: never fast >4h (infants), >8h (children), >12h (adults)

TREATMENT SUMMARY:
  Avoid fasting (ABSOLUTE — Level A)
  Emergency protocol during illness: IV glucose 10% 8–10 mg/kg/min (anti-catabolic)
  Regular feeding frequency (no overnight fasting in infants)
  L-carnitine: if C0 depleted; controversial (some guidelines reserve for low C0 + symptomatic)
  No dietary fat restriction (WRONG — medium-chain fat is fine; fasting is the enemy)
  VPA: HIGH RISK — inhibits mitochondrial FAO; worsens energy deficit; depletes carnitine
  Riboflavin (B2): NOT generally indicated (unlike MADD/GA2 where riboflavin helps)
  Ketogenic diet: ABSOLUTE CI — requires fasting cycles; worsens MCAD crisis

PHENOTYPES:
  Asymptomatic NBS-detected:       55% — most patients with NBS; no crisis if feeding protocol followed
  Classic Hypoketotic Crisis:       30% — Reye-like; hypoglycaemia + encephalopathy + hepatopathy
  SIDS/SUDS-like (pre-NBS era):    10% — historically before NBS; sudden unexpected death during febrile illness
  Chronic Hepatopathy:              5%  — rare; persistent liver disease; recurrent crises

COMMON VARIANTS (ACADM):
  c.985A>G (p.Lys329Glu):   80–90% Northern European alleles; FAD-binding domain; founder mutation
  c.199T>C (p.Tyr67His):    5% alleles; signal peptide processing region; partially reduces import
  c.583A>G (p.Asn195Asp):   3% alleles; substrate-binding domain; moderate severity
  c.449_452delCTGA:          2% alleles; frameshift; null allele; severe
  c.157T>G (p.Ser53Ala):    2% alleles; mitochondrial targeting region; variable
"""

import random

SEED = 269        # deterministic cohort
N_PATIENTS = 40

random.seed(SEED)

PHENOTYPES = [
    ("Asymptomatic NBS-detected",            0.55),
    ("Classic Hypoketotic Crisis",           0.30),
    ("SIDS/SUDS-like (historical / severe)", 0.10),
    ("Chronic Hepatopathy",                  0.05),
]

VARIANTS = [
    {"variant": "c.985A>G (p.Lys329Glu)",   "freq": 87, "domain": "FAD-binding domain",           "phenotype": "NBS / Classic", "note": "Northern European founder; ~80–90% of Northern European alleles; partial residual FAD affinity"},
    {"variant": "c.199T>C (p.Tyr67His)",     "freq":  5, "domain": "Signal peptide region",        "phenotype": "Variable",      "note": "Impairs mitochondrial import; reduced mature protein level"},
    {"variant": "c.583A>G (p.Asn195Asp)",   "freq":  3, "domain": "Substrate-binding domain",     "phenotype": "Moderate",      "note": "Reduces medium-chain substrate affinity; moderate FAO impairment"},
    {"variant": "c.449_452delCTGA",          "freq":  2, "domain": "Catalytic core (frameshift)",  "phenotype": "Severe / SUDS", "note": "Null allele; complete LOF; severe energy deficit"},
    {"variant": "c.157T>G (p.Ser53Ala)",    "freq":  2, "domain": "Targeting sequence",           "phenotype": "Mild / NBS",    "note": "Temperature-sensitive; may be benign in isolation"},
    {"variant": "Other / compound het",      "freq":  1, "domain": "Various",                      "phenotype": "Variable",      "note": "Compound heterozygotes; p.Lys329Glu + rare null common pairing"},
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
        is_nbs     = ph == "Asymptomatic NBS-detected"
        is_crisis  = ph == "Classic Hypoketotic Crisis"
        is_suds    = ph == "SIDS/SUDS-like (historical / severe)"
        is_hepatic = ph == "Chronic Hepatopathy"

        onset_mo = (
            0                        if is_nbs    else
            random.randint(3, 24)    if is_crisis else
            random.randint(1, 12)    if is_suds   else
            random.randint(6, 36)
        )

        # C8 (octanoylcarnitine) — PRIMARY NBS MARKER
        c8 = round(
            random.uniform(0.8, 4.5)   if is_crisis else
            random.uniform(0.3, 1.2)   if is_nbs    else
            random.uniform(1.2, 6.0)   if is_suds   else
            random.uniform(0.5, 2.5),
            2
        )

        # C6 (hexanoylcarnitine) — elevated
        c6 = round(
            random.uniform(0.12, 0.55) if is_crisis else
            random.uniform(0.08, 0.20) if is_nbs    else
            random.uniform(0.15, 0.70) if is_suds   else
            random.uniform(0.10, 0.40),
            2
        )

        # C10:1 (cis-4-decenoylcarnitine) — elevated
        c10_1 = round(
            random.uniform(0.08, 0.35) if is_crisis else
            random.uniform(0.04, 0.15) if is_nbs    else
            random.uniform(0.10, 0.45) if is_suds   else
            random.uniform(0.06, 0.25),
            2
        )

        # C10 (decanoylcarnitine)
        c10 = round(
            random.uniform(0.18, 0.80) if is_crisis else
            random.uniform(0.10, 0.35) if is_nbs    else
            random.uniform(0.25, 1.00) if is_suds   else
            random.uniform(0.12, 0.50),
            2
        )

        c8_c10_ratio = round(c8 / c10, 1) if c10 > 0 else 0

        # Free carnitine C0 — low secondary depletion
        c0 = round(
            random.uniform(8, 22)   if is_crisis else
            random.uniform(15, 35)  if is_nbs    else
            random.uniform(5, 18)   if is_suds   else
            random.uniform(10, 28),
            1
        )

        # C14 — NORMAL (KEY NEGATIVE vs VLCAD)
        c14 = round(random.uniform(0.05, 0.20), 2)

        # Blood glucose during crisis
        glucose = (
            round(random.uniform(1.2, 3.0), 1)  if is_crisis else
            round(random.uniform(3.5, 5.5), 1)  if is_nbs    else
            round(random.uniform(0.8, 2.5), 1)  if is_suds   else
            round(random.uniform(2.5, 4.5), 1)
        )

        # Beta-OHB — LOW (hypoketosis)
        bohb = round(
            random.uniform(0.1, 0.4)   if is_crisis else  # inappropriately low
            random.uniform(0.3, 1.0)   if is_nbs    else
            random.uniform(0.05, 0.3)  if is_suds   else
            random.uniform(0.2, 0.8),
            2
        )

        # NH3 — mildly elevated during crisis
        nh3 = random.randint(60, 180) if is_crisis else random.randint(20, 60)

        # ALT (hepatopathy)
        alt = (
            random.randint(80, 500)  if is_crisis else
            random.randint(15, 40)   if is_nbs    else
            random.randint(200, 800) if is_suds   else
            random.randint(50, 300)
        )

        # Hexanoylglycine (urine, pathognomonic)
        hg = (
            round(random.uniform(10, 80), 1)   if is_crisis else
            round(random.uniform(2, 15), 1)    if is_nbs    else
            round(random.uniform(20, 120), 1)  if is_suds   else
            round(random.uniform(5, 40), 1)
        )

        # Suberylglycine (urine, pathognomonic)
        sg = (
            round(random.uniform(15, 100), 1)  if is_crisis else
            round(random.uniform(3, 20), 1)    if is_nbs    else
            round(random.uniform(25, 150), 1)  if is_suds   else
            round(random.uniform(8, 50), 1)
        )

        # Variant assignment (weighted to c.985A>G)
        v_roll = random.random()
        if v_roll < 0.87:
            variant = VARIANTS[0]["variant"]
        elif v_roll < 0.92:
            variant = VARIANTS[1]["variant"]
        elif v_roll < 0.95:
            variant = VARIANTS[2]["variant"]
        elif v_roll < 0.97:
            variant = VARIANTS[3]["variant"]
        elif v_roll < 0.99:
            variant = VARIANTS[4]["variant"]
        else:
            variant = VARIANTS[5]["variant"]

        seizures = (
            random.random() < 0.55 if is_crisis else
            False                   if is_nbs    else
            random.random() < 0.70 if is_suds   else
            random.random() < 0.30
        )

        # Treatment responses
        tx = []
        if is_crisis or is_suds or is_hepatic:
            tx.append("IV Glucose 10%")
        tx.append("Avoid Fasting Protocol")
        if c0 < 20:
            tx.append("L-Carnitine (C0 depleted)")
        if not (is_suds or is_nbs):
            tx.append("Emergency Glucose Protocol")

        cohort.append({
            "id":           f"MCAD-{i+1:03d}",
            "phenotype":    ph,
            "onset_mo":     onset_mo,
            "c8":           c8,
            "c6":           c6,
            "c10_1":        c10_1,
            "c10":          c10,
            "c8_c10_ratio": c8_c10_ratio,
            "c0":           c0,
            "c14":          c14,
            "glucose":      glucose,
            "bohb":         bohb,
            "nh3":          nh3,
            "alt":          alt,
            "hg":           hg,
            "sg":           sg,
            "variant":      variant,
            "seizures":     seizures,
            "treatments":   tx,
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

    seizure_n   = sum(1 for p in cohort if p["seizures"])
    crisis_n    = ph_counts.get("Classic Hypoketotic Crisis", 0)
    nbs_n       = ph_counts.get("Asymptomatic NBS-detected", 0)
    hepatic_n   = ph_counts.get("Chronic Hepatopathy", 0)
    suds_n      = ph_counts.get("SIDS/SUDS-like (historical / severe)", 0)

    avg_c8      = round(sum(p["c8"] for p in cohort) / N_PATIENTS, 2)
    avg_c0      = round(sum(p["c0"] for p in cohort) / N_PATIENTS, 1)

    top_variant = max(variant_counts, key=variant_counts.get)
    top_v_pct   = round(100 * variant_counts[top_variant] / N_PATIENTS)

    return {
        "disease": "MCAD Deficiency (Medium-Chain Acyl-CoA Dehydrogenase Deficiency)",
        "gene":    "ACADM (MCAD)",
        "locus":   "1p31.1",
        "omim_gene":    "607008",
        "omim_disease": "201450",
        "inheritance":  "Autosomal Recessive (AR)",
        "protein":      "ACADM — 421 aa mature; mitochondrial matrix; FAD-dependent; homotetrameric",
        "prevalence":   "~1:10,000–15,000 (Northern European); ~1:20,000 worldwide; most common FAO disorder",
        "n_patients":   N_PATIENTS,
        "seed":         SEED,
        "kpis": {
            "total_patients":   N_PATIENTS,
            "nbs_detected":     nbs_n,
            "crisis_n":         crisis_n,
            "seizures_n":       seizure_n,
            "hepatic_n":        hepatic_n,
            "suds_n":           suds_n,
            "avg_c8_umol":      avg_c8,
            "avg_c0_umol":      avg_c0,
        },
        "phenotype_distribution": ph_counts,
        "top_variant":      top_variant,
        "top_variant_pct":  top_v_pct,
        "primary_nbs_marker":       "C8 (Octanoylcarnitine) >0.3 µmol/L",
        "pathognomonic_urine":      ["Hexanoylglycine (HG)", "Suberylglycine (SG)", "Phenylpropionylglycine (PPG)"],
        "absolute_ci":              ["Fasting", "Ketogenic Diet"],
        "high_risk_drugs":          ["VPA (Valproic acid)"],
        "first_line_treatment":     "Avoid fasting + Emergency IV Glucose 10% during illness",
        "clinical_summary": (
            "MCAD deficiency is the most common fatty acid oxidation (FAO) disorder (1:10,000–15,000 Northern European). "
            "ACADM encodes medium-chain acyl-CoA dehydrogenase (421 aa; FAD-dependent) — Step 1 of beta-oxidation "
            "for C6–C12 fatty acids. MCAD LOF → C8 (octanoylcarnitine) accumulates → PRIMARY NBS marker. "
            "Pathognomonic urine: hexanoylglycine + suberylglycine + phenylpropionylglycine. "
            "Clinical: HYPOketotic hypoglycaemia during fasting/illness → Reye-like crisis / encephalopathy. "
            "Most patients diagnosed by NBS are asymptomatic if feeding protocol is followed. "
            "FASTING is the primary and absolute trigger — avoid at all costs. "
            "KEY NEGATIVE vs VLCAD: C14/C16/C18 NORMAL (VLCAD elevates long-chain acylcarnitines)."
        ),
        "key_negatives": {
            "c14_normal":    "NORMAL — KEY NEGATIVE vs VLCAD (which elevates C14, C14:1, C16, C18:1)",
            "ketones_low":   "HYPOketotic (inappropriately LOW ketones despite hypoglycaemia) — KEY NEGATIVE vs OA disorders (which cause HYPERketosis)",
            "c3_normal":     "NORMAL — KEY NEGATIVE vs PA (propionyl-CoA not involved)",
            "mma_absent":    "ABSENT — KEY NEGATIVE vs MMUT deficiency",
            "c5_normal":     "NORMAL — KEY NEGATIVE vs IVA (isovaleryl-CoA not involved)",
        },
    }


def get_breakdown():
    cohort = _get_cohort()
    patients = cohort[:10]  # representative subset for detailed view

    biomarkers = {
        "c8": {
            "label":     "C8 (Octanoylcarnitine)",
            "normal":    "<0.15 µmol/L",
            "status":    "ELEVATED >0.3–0.5 µmol/L — PRIMARY NBS MARKER",
            "direction": "↑↑↑",
            "color":     "danger",
            "rationale": (
                "C8 (octanoylcarnitine) is the PRIMARY tandem MS/MS newborn screening marker for MCAD deficiency. "
                "Octanoyl-CoA (C8) is the preferred substrate of MCAD; when MCAD is absent, octanoyl-CoA accumulates "
                "→ transferred to carnitine → C8 (octanoylcarnitine) released into blood and detected on NBS filter paper. "
                "NBS cut-off: >0.3–0.5 µmol/L (lab-specific; most programmes use >0.3 µmol/L). "
                "Crisis values: 0.5–6.0 µmol/L. Interactal (NBS-detected): 0.3–1.5 µmol/L. "
                "C8 alone has high positive predictive value for MCAD when confirmed with urine organic acids."
            ),
        },
        "c8_c10_ratio": {
            "label":     "C8/C10 Ratio",
            "normal":    "<1.5",
            "status":    "ELEVATED >2.0 — useful discriminator vs VLCAD",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "C8/C10 ratio is elevated in MCAD because C8 accumulates proportionally more than C10 "
                "(C8 is the preferred substrate). "
                "In VLCAD deficiency, C14 and C14:1 dominate; C8/C10 ratio is typically normal or low. "
                "C8/C10 >2.0 combined with C8 >0.3 µmol/L = strongly suggests MCAD. "
                "The ratio helps when C8 is borderline; also used in differential with VLCAD and LCHAD."
            ),
        },
        "c6": {
            "label":     "C6 (Hexanoylcarnitine)",
            "normal":    "<0.06 µmol/L",
            "status":    "ELEVATED >0.08–0.15 µmol/L — secondary accumulation",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "Hexanoyl-CoA (C6) accumulates secondarily in MCAD deficiency — MCAD also acts on hexanoyl-CoA "
                "though less efficiently than octanoyl-CoA. "
                "C6 elevation is supportive but less specific than C8. "
                "The combination C6 + C8 + C10:1 + C10 elevated = characteristic MCAD acylcarnitine profile."
            ),
        },
        "c10_1": {
            "label":     "C10:1 (cis-4-Decenoylcarnitine)",
            "normal":    "<0.04 µmol/L",
            "status":    "ELEVATED >0.05 µmol/L — supportive marker",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "cis-4-Decenoyl-CoA (an unsaturated C10 species from C10 beta-oxidation) accumulates "
                "as MCAD also handles this substrate. "
                "C10:1 combined with C8 elevation improves specificity of MCAD diagnosis. "
                "Less elevated than C8; useful for confirmation when C8 is borderline."
            ),
        },
        "c0": {
            "label":     "C0 (Free Carnitine)",
            "normal":    "25–50 µmol/L",
            "status":    "LOW (<20 µmol/L in many patients) — secondary depletion",
            "direction": "↓",
            "color":     "warning",
            "rationale": (
                "Free carnitine (C0) is depleted in MCAD deficiency because medium-chain acyl-CoAs "
                "are conjugated to carnitine (forming C6, C8, C10 acylcarnitines) → depletes the free carnitine pool. "
                "C0 depletion is secondary (not the primary defect) but can exacerbate energy deficit. "
                "L-carnitine supplementation is controversial; most guidelines reserve it for symptomatic "
                "C0 depletion (<10 µmol/L) rather than universal supplementation."
            ),
        },
        "c14": {
            "label":     "C14 (Tetradecanoylcarnitine / Myristoylcarnitine)",
            "normal":    "<0.25 µmol/L",
            "status":    "NORMAL — KEY NEGATIVE vs VLCAD (which elevates long-chain acylcarnitines)",
            "direction": "→ NORMAL",
            "color":     "success",
            "rationale": (
                "C14 (and C14:1, C16, C18:1) are the PRIMARY markers for VLCAD deficiency. "
                "In MCAD deficiency, C14 is NORMAL because long-chain FAO (VLCAD, LCHAD, TRIFUNCTIONAL PROTEIN) "
                "is intact — MCAD acts only on C6–C12 substrates. "
                "C14 NORMAL is a KEY NEGATIVE vs VLCAD. "
                "If C14 is elevated alongside C8: consider COMBINED deficiency or VLCAD + secondary C8 elevation."
            ),
        },
        "glucose": {
            "label":     "Blood Glucose",
            "normal":    "3.5–6.0 mmol/L",
            "status":    "LOW during crisis (<3.0 mmol/L) — hypoketotic hypoglycaemia HALLMARK",
            "direction": "↓ (crisis)",
            "color":     "danger",
            "rationale": (
                "Hypoglycaemia in MCAD deficiency results from: "
                "(1) Failure to oxidise medium-chain fatty acids → cannot generate acetyl-CoA → no ketogenesis; "
                "(2) Impaired gluconeogenesis (acetyl-CoA normally allosterically activates pyruvate carboxylase); "
                "(3) Continued glucose consumption without fatty acid-derived energy replacement. "
                "Blood glucose <2.6 mmol/L (hypoglycaemia) + inappropriate hypoketosis = hallmark FAO crisis. "
                "Treatment: IV glucose 10% at 8–10 mg/kg/min immediately (reverses crisis rapidly)."
            ),
        },
        "bohb": {
            "label":     "β-Hydroxybutyrate (β-OHB) / Ketones",
            "normal":    "<0.5 mmol/L (fed); up to 2.0 mmol/L (fasting normal ketosis)",
            "status":    "INAPPROPRIATELY LOW during crisis — HYPOketosis is the HALLMARK",
            "direction": "↓ (crisis) — HYPOketotic",
            "color":     "success",
            "rationale": (
                "MCAD blocks medium-chain FAO → cannot generate acetyl-CoA from C6–C12 fatty acids → "
                "cannot generate ketones (β-OHB + AcAc) via HMG-CoA pathway in liver. "
                "During hypoglycaemia, normal response = ketosis (β-OHB >3 mmol/L). "
                "In MCAD crisis: β-OHB <0.5 mmol/L (absent or very low) despite severe hypoglycaemia. "
                "This inappropriately low ketone level IS THE DIAGNOSTIC CLUE for FAO disorders. "
                "CONTRAST: OA disorders (ACAT1, HMGCL etc.) → HYPERketosis during crisis. "
                "CONTRAST: MCAD/VLCAD/LCHAD → HYPOketotic hypoglycaemia."
            ),
        },
        "hexanoylglycine": {
            "label":     "Hexanoylglycine (HG) — Urine OA",
            "normal":    "<2 mmol/mol Cr",
            "status":    "ELEVATED >3–5 mmol/mol Cr — PATHOGNOMONIC",
            "direction": "↑↑",
            "color":     "danger",
            "rationale": (
                "Hexanoylglycine is formed from hexanoyl-CoA + glycine (mitochondrial glycine N-acyltransferase). "
                "Hexanoyl-CoA accumulates in MCAD deficiency → HG elevated in urine. "
                "HG is one of the most specific markers for MCAD deficiency (pathognomonic when elevated). "
                "Best detected on urine GC-MS as part of full organic acid profile. "
                "Collect urine during or immediately after crisis for highest yield."
            ),
        },
        "suberylglycine": {
            "label":     "Suberylglycine (SG) — Urine OA",
            "normal":    "<3 mmol/mol Cr",
            "status":    "ELEVATED >5–10 mmol/mol Cr — PATHOGNOMONIC",
            "direction": "↑↑",
            "color":     "danger",
            "rationale": (
                "Suberylglycine is the glycine conjugate of suberic acid (octanedioic acid, C8 dicarboxylic acid). "
                "Generated via omega-oxidation of accumulated octanoyl-CoA → suberic acid → suberylglycine. "
                "SG is highly specific for MCAD deficiency — rarely elevated in other FAO disorders. "
                "SG + HG + PPG together = DIAGNOSTIC FINGERPRINT of MCAD deficiency. "
                "Suberylglycine can be detected during crisis and between crises in classic patients."
            ),
        },
        "nh3": {
            "label":     "Ammonia (NH3)",
            "normal":    "<50 µmol/L",
            "status":    "MILDLY elevated during crisis (60–180 µmol/L) — secondary; NOT primary UCD",
            "direction": "↑ (mild, crisis)",
            "color":     "warning",
            "rationale": (
                "NH3 is mildly elevated during MCAD crises (Reye-like syndrome) — secondary to: "
                "(1) Energy deficit impairing urea cycle (requires ATP); "
                "(2) Hepatopathy with reduced ammonia clearance; "
                "(3) Fatty acid metabolites partially inhibiting carbamoyl phosphate synthetase. "
                "NH3 in MCAD crisis: typically 60–180 µmol/L (vs >500 µmol/L in primary UCDs). "
                "NH3 inter-ictally: NORMAL. NH3 elevation with Reye-like crisis + hypoketotic hypoglycaemia = FAO disorder."
            ),
        },
        "alt": {
            "label":     "ALT (Alanine Aminotransferase)",
            "normal":    "<40 U/L",
            "status":    "ELEVATED during crisis (hepatopathy) — Reye-like pattern",
            "direction": "↑ (crisis)",
            "color":     "warning",
            "rationale": (
                "Hepatopathy is a feature of MCAD crisis — medium-chain fatty acids (octanoate, hexanoate) "
                "are directly hepatotoxic when they accumulate. "
                "ALT elevation during crisis: 80–800 U/L (Reye-like pattern). "
                "Liver biopsy (historical): microvesicular steatosis — lipid droplets from FAO arrest. "
                "ALT normalises between crises. Liver biopsy is NOT routinely needed given enzymatic/molecular diagnosis."
            ),
        },
    }

    phenotype_biomarker_patterns = [
        {
            "phenotype":  "Asymptomatic NBS-detected",
            "prevalence": "55%",
            "c8":         "0.3–1.2 µmol/L",
            "glucose":    "Normal (3.5–5.5 mmol/L)",
            "bohb":       "Normal",
            "hg":         "2–15 mmol/mol Cr",
            "prognosis":  "Excellent if feeding protocol followed; crisis avoidable with education",
        },
        {
            "phenotype":  "Classic Hypoketotic Crisis",
            "prevalence": "30%",
            "c8":         "0.8–4.5 µmol/L",
            "glucose":    "LOW 1.2–3.0 mmol/L",
            "bohb":       "LOW 0.1–0.4 mmol/L (hypoketotic)",
            "hg":         "10–80 mmol/mol Cr",
            "prognosis":  "Excellent with emergency glucose; poor if delayed; encephalopathy risk",
        },
        {
            "phenotype":  "SIDS/SUDS-like",
            "prevalence": "10%",
            "c8":         "1.2–6.0 µmol/L",
            "glucose":    "VERY LOW <2.5 mmol/L",
            "bohb":       "VERY LOW <0.3 mmol/L",
            "hg":         "20–120 mmol/mol Cr",
            "prognosis":  "High mortality if unrecognised; post-mortem NBS can confirm",
        },
        {
            "phenotype":  "Chronic Hepatopathy",
            "prevalence": "5%",
            "c8":         "0.5–2.5 µmol/L",
            "glucose":    "LOW 2.5–4.5 mmol/L",
            "bohb":       "Low-normal 0.2–0.8 mmol/L",
            "hg":         "5–40 mmol/mol Cr",
            "prognosis":  "Variable; liver function often stabilises with metabolic management",
        },
    ]

    treatment_table = [
        {
            "intervention":  "Avoid Fasting (ALL PATIENTS)",
            "level":         "Level A — ABSOLUTE RULE",
            "rationale":     "Primary and most critical intervention. No overnight fasting (infants: <4h; children: <8h; adults: <12h)",
            "contraindication": None,
        },
        {
            "intervention":  "IV Glucose 10% (8–10 mg/kg/min) — acute crisis",
            "level":         "Level A — FIRST LINE EMERGENCY",
            "rationale":     "Provides immediate anti-catabolic substrate; reverses hypoketotic hypoglycaemia; stops FA mobilisation",
            "contraindication": None,
        },
        {
            "intervention":  "Emergency feeding protocol (illness plan)",
            "level":         "Level A",
            "rationale":     "Every patient/family receives written sick-day rules: glucose polymer (Maxijul/Dextrose) every 2h during illness; IV access if vomiting",
            "contraindication": None,
        },
        {
            "intervention":  "L-Carnitine supplementation",
            "level":         "Level B (if C0 <10 µmol/L or symptomatic depletion)",
            "rationale":     "Replaces secondary carnitine depletion; NOT universally recommended — controversial; may increase long-chain acylcarnitines",
            "contraindication": None,
        },
        {
            "intervention":  "Fasting (primary trigger)",
            "level":         "ABSOLUTE CONTRAINDICATION",
            "rationale":     "Fasting → mobilises FA → C6–C12 cannot be oxidised → energy collapse → Reye-like crisis",
            "contraindication": "ABSOLUTE CI",
        },
        {
            "intervention":  "Ketogenic Diet",
            "level":         "ABSOLUTE CONTRAINDICATION",
            "rationale":     "KD requires prolonged fasting intervals → triggers FA mobilisation → identical to fasting crisis in MCAD",
            "contraindication": "ABSOLUTE CI",
        },
        {
            "intervention":  "VPA (Valproic acid)",
            "level":         "HIGH RISK — avoid if possible",
            "rationale":     "VPA inhibits mitochondrial FAO; depletes carnitine; worsens medium-chain energy deficit; hepatotoxic in FAO disorders",
            "contraindication": "HIGH RISK",
        },
        {
            "intervention":  "Medium-chain fat restriction",
            "level":         "NOT RECOMMENDED — common error",
            "rationale":     "Dietary medium-chain fat restriction is WRONG; fasting (not fat intake) is the trigger; avoid this misconception",
            "contraindication": "Incorrect intervention",
        },
        {
            "intervention":  "Riboflavin (Vitamin B2)",
            "level":         "NOT INDICATED for MCAD",
            "rationale":     "Riboflavin is beneficial in MADD (GA2/ETFA/ETFB) and some VLCAD variants, NOT MCAD. MCAD requires no cofactor supplementation.",
            "contraindication": None,
        },
    ]

    return {
        "biomarkers":               biomarkers,
        "phenotype_patterns":       phenotype_biomarker_patterns,
        "treatment_table":          treatment_table,
        "patient_sample":           patients,
        "variant_table":            VARIANTS,
        "key_differentials": {
            "MCAD_vs_VLCAD":    "C8 dominant (MCAD) vs C14/C14:1 dominant (VLCAD); VLCAD → cardiomyopathy; MCAD → hepatopathy/encephalopathy",
            "MCAD_vs_LCHAD":    "C8 (MCAD) vs C16-OH/C18:1-OH (LCHAD); LCHAD → pigmentary retinopathy + peripheral neuropathy",
            "MCAD_vs_fasting":  "Dicarboxylic acids (adipic/suberic/sebacic) are NON-SPECIFIC for MCAD — also elevated in fasting; HG + SG + PPG are SPECIFIC",
            "MCAD_vs_SCAD":     "C8 (MCAD) vs C4/C4-OH (SCAD); SCAD usually benign; MCAD causes serious crises",
            "HYPOketosis_clue": "Inappropriate hypoketosis during hypoglycaemia = FAO disorder (MCAD/VLCAD/LCHAD/CPT1/CACT). HYPERketosis = OA disorder (ACAT1/HMGCL)",
        },
        "exam_pearls": [
            "C8 (octanoylcarnitine) = PRIMARY NBS MARKER — most common reason for FAO positive NBS",
            "HYPOKETOTIC hypoglycaemia = hallmark (inappropriately LOW ketones during hypoglycaemia)",
            "Hexanoylglycine + suberylglycine = PATHOGNOMONIC urine markers",
            "c.985A>G (p.Lys329Glu) = FOUNDER MUTATION — >80% Northern European alleles",
            "C14 NORMAL — KEY NEGATIVE vs VLCAD (which dominates C14/C14:1/C16)",
            "FASTING = ABSOLUTE CI — most common and preventable trigger of crisis",
            "VPA = HIGH RISK in all FAO disorders — avoid",
            "KD = ABSOLUTE CI — fasting intervals trigger crisis",
            "Medium-chain fat restriction = WRONG intervention (a classic exam trap)",
            "MOST COMMON FAO disorder worldwide; 1:10,000–15,000 Northern European",
            "Reye-like crisis + hypoketotic hypoglycaemia + elevated transaminases = MCAD crisis signature",
            "PPG (phenylpropionylglycine) = produced by intestinal bacteria fermenting hexanoic acid; pathognomonic",
        ],
    }


def get_definitions():
    return {
        "disease_name":   "MCAD Deficiency (Medium-Chain Acyl-CoA Dehydrogenase Deficiency)",
        "gene":           "ACADM (also called MCAD)",
        "locus":          "1p31.1",
        "omim_gene":      "*607008",
        "omim_disease":   "#201450",
        "inheritance":    "Autosomal Recessive (AR) — biallelic LOF required",
        "protein":        "ACADM — 421 aa mature form; mitochondrial matrix; FAD-dependent acyl-CoA dehydrogenase; homotetrameric",
        "enzymatic_function": (
            "Catalyses Step 1 of mitochondrial beta-oxidation for C6–C12 (medium-chain) acyl-CoA species: "
            "Medium-chain acyl-CoA + FAD → trans-2-enoyl-acyl-CoA + FADH2. "
            "FADH2 is transferred to ETF (electron transfer flavoprotein) → ETFDH (ubiquinone oxidoreductase) → "
            "enters respiratory chain at Complex I/III level."
        ),
        "pathway":       "Mitochondrial fatty acid beta-oxidation (Step 1, C6–C12 chain length specificity)",
        "metabolic_block": (
            "ACADM LOF → Medium-chain acyl-CoAs (C6–C12) CANNOT be dehydrogenated → "
            "accumulate as free acids and acylcarnitine conjugates → "
            "beta-oxidation spiral arrested at medium-chain stage → "
            "cannot generate acetyl-CoA from C6–C12 fatty acids → "
            "energy deficit during fasting (when FA oxidation is critical for gluconeogenesis support and ketogenesis)."
        ),
        "nbs_marker":    "C8 (octanoylcarnitine) >0.3 µmol/L on tandem MS/MS expanded newborn screening",
        "confirmatory_biomarkers": {
            "plasma_acylcarnitines": "C8 ↑↑, C6 ↑, C10:1 ↑, C10 ↑; C8/C10 ratio >2.0; C14 NORMAL",
            "urine_organic_acids":   "Hexanoylglycine (HG) ↑↑, Suberylglycine (SG) ↑↑, Phenylpropionylglycine (PPG) ↑; dicarboxylic acids ↑ (non-specific)",
            "enzyme_activity":       "MCAD enzyme activity in lymphocytes/fibroblasts <10% normal (reference method)",
            "molecular":             "ACADM sequencing; c.985A>G (p.Lys329Glu) founder mutation in 80–90% Northern European alleles",
        },
        "clinical_features": {
            "acute_crisis": "Reye-like syndrome: hypoketotic hypoglycaemia + vomiting + encephalopathy + hepatopathy + elevated transaminases + mildly elevated NH3",
            "hallmark":     "HYPOketotic hypoglycaemia — inappropriately low or absent ketones despite severe hypoglycaemia",
            "triggers":     "Fasting (primary), intercurrent illness + anorexia, vomiting, surgery, prolonged sleep",
            "age_of_onset": "Typically 3–24 months (classic); NBS-detected: asymptomatic from birth",
            "chronic":      "Between crises: normal; no chronic metabolic disease if fasting avoided",
        },
        "treatments": {
            "fasting_avoidance": "Level A — ABSOLUTE RULE; most important intervention; feeding frequency protocol mandatory",
            "iv_glucose":        "Level A — 10% dextrose at 8–10 mg/kg/min IV during illness/crisis (anti-catabolic)",
            "l_carnitine":       "Level B — only if C0 <10 µmol/L or symptomatic; controversial",
            "contraindications": ["Fasting (ABSOLUTE CI)", "Ketogenic Diet (ABSOLUTE CI)", "VPA (HIGH RISK)", "Medium-chain fat restriction (WRONG — not indicated)"],
        },
        "prevalence": "~1:10,000–15,000 Northern European; ~1:20,000 worldwide; most common fatty acid oxidation disorder",
        "founder_variant": "c.985A>G (p.Lys329Glu) — 80–90% Northern European alleles; FAD-binding domain; pathogenic",
        "key_exam_facts": [
            "MOST COMMON FAO disorder worldwide",
            "C8 (octanoylcarnitine) = PRIMARY NBS MARKER",
            "HYPOketotic hypoglycaemia = HALLMARK (vs HYPERketosis in OA disorders like ACAT1)",
            "Hexanoylglycine + Suberylglycine + Phenylpropionylglycine = PATHOGNOMONIC urine profile",
            "c.985A>G (p.Lys329Glu) = FOUNDER MUTATION — >80% Northern European alleles",
            "C14 NORMAL = KEY NEGATIVE vs VLCAD",
            "FASTING + ILLNESS = ABSOLUTE TRIGGER — Reye-like crisis",
            "FASTING = ABSOLUTE CI; KD = ABSOLUTE CI; VPA = HIGH RISK",
            "Medium-chain fat restriction = WRONG intervention (exam trap)",
            "IV glucose 10% = FIRST LINE EMERGENCY treatment",
            "PPG (phenylpropionylglycine) from gut bacteria fermenting hexanoic acid = PATHOGNOMONIC",
            "Between crises: biomarkers may be NORMAL or only mildly elevated",
        ],
        "glossary": {
            "ACADM":   "Acyl-CoA Dehydrogenase Medium-chain gene (1p31.1)",
            "MCAD":    "Medium-Chain Acyl-CoA Dehydrogenase enzyme product of ACADM",
            "FAO":     "Fatty Acid Oxidation (mitochondrial beta-oxidation pathway)",
            "C8":      "Octanoylcarnitine — acylcarnitine of 8-carbon (medium-chain) fatty acid",
            "HG":      "Hexanoylglycine — glycine conjugate of hexanoyl-CoA; PATHOGNOMONIC urine marker",
            "SG":      "Suberylglycine — glycine conjugate of suberic acid (C8 dicarboxylic); PATHOGNOMONIC",
            "PPG":     "Phenylpropionylglycine — intestinal bacterial metabolite; PATHOGNOMONIC in MCAD",
            "NBS":     "Newborn Screening — expanded tandem MS/MS detects C8 elevation",
            "ETF":     "Electron Transfer Flavoprotein — receives FADH2 from ACADM in beta-oxidation",
            "ETFDH":   "ETF:Ubiquinone Oxidoreductase — transfers electrons to respiratory chain",
            "Reye-like": "Reye syndrome phenocopy — hepatopathy + encephalopathy + hypoketotic hypoglycaemia",
            "HYPOketosis": "Inappropriately low ketones during hypoglycaemia — hallmark of FAO disorders",
            "p.Lys329Glu": "c.985A>G variant in ACADM; Northern European founder; most common MCAD allele",
        },
        "references": [
            "Grosse SD et al. (2006). Revised estimates for newborn screening. Genet Med 8:539-544.",
            "Rinaldo P et al. (2002). MCAD deficiency: diagnosis, clinical course. Pediatrics 110:1275-1277.",
            "Wilcken B (2010). MCAD deficiency: natural history. J Inherit Metab Dis 33:501-506.",
            "van Maldegem BT et al. (2010). Clinical, biochemical, and genetic heterogeneity in MCAD deficiency. JAMA 303(21):2163.",
            "Merritt JL 2nd et al. (2018). Biochemical and enzymatic confirmation of MCAD deficiency in NBS era. Mol Genet Metab 124(4):266-273.",
            "OMIM #201450 — MCAD Deficiency (ACADM gene #607008). omim.org.",
        ],
    }
