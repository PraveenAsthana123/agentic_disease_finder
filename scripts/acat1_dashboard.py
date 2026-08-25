#!/usr/bin/env python3
"""ACAT1 (Beta-Ketothiolase Deficiency / T2 Deficiency) Dashboard.

ACAT1 (mitochondrial acetoacetyl-CoA thiolase / T2) catalyses TWO reactions:
  (A) Isoleucine catabolism STEP 4 (final step):
        2-Methylacetoacetyl-CoA + CoA  →  [ACAT1]  →  Propionyl-CoA + Acetyl-CoA
  (B) Ketone body UTILISATION (ketolysis, peripheral tissues):
        Acetoacetyl-CoA  + CoA  →  [ACAT1]  →  2 × Acetyl-CoA

ACAT1 LOF → DUAL BLOCK:
  (A) 2-Methylacetoacetyl-CoA ACCUMULATES → 2M3HBA ↑↑ + Tiglylglycine ↑↑ + 2-MAA transient ↑
  (B) Acetoacetyl-CoA CANNOT be utilised by peripheral tissues → HYPERKETOSIS during any catabolic state

KEY FACTS (HIGHEST YIELD):
  1. 2-Methyl-3-hydroxybutyric acid (2M3HBA) — PRIMARY stable marker (200–3000 mmol/mol Cr)
  2. Tiglylglycine — PATHOGNOMONIC glycine conjugate of tiglyl-CoA (upstream intermediate)
  3. 2-Methylacetoacetic acid — PATHOGNOMONIC but VERY UNSTABLE (decarboxylates at room temp → process urine STAT)
  4. C5:1 (tiglylcarnitine) — PRIMARY NBS trigger; C5-OH mildly elevated
  5. C3 (propionylcarnitine) NORMAL — KEY NEGATIVE vs PA (propionyl-CoA not yet made = substrate, not product!)
  6. Methylcitrate ABSENT — KEY NEGATIVE vs PA
  7. MMA ABSENT — KEY NEGATIVE vs MMA (MMUT)
  8. HYPERKETOSIS during crisis — KEY NEGATIVE vs HMGCL (which causes HYPOketotic hypoglycaemia)
  9. KD: ABSOLUTE CI — floods acetoacetyl-CoA pathway that is blocked → catastrophic ketoacidosis
 10. VPA: HIGH RISK — worsens mitochondrial function + promotes hyperammonemia; not absolute CI but HIGH RISK
 11. EPISODIC not chronic — biomarkers can be NORMAL between crises; crisis triggered by illness/fasting/high-protein
 12. Leucine restriction ALONE is insufficient — isoleucine restriction is the key dietary intervention

OMIM Disease: #203750 (2-Methylacetoacetic Aciduria / Beta-Ketothiolase Deficiency)
OMIM Gene:    *607809 (ACAT1)
Chromosome:   11q22.3
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Protein:      ACAT1 = 427 aa; mitochondrial matrix homotetrameric thiolase; CoA-binding active site Cys89
Prevalence:   ~1:100,000–250,000

ISOLEUCINE CATABOLISM — STEP 4 CONTEXT (where ACAT1 fits — LAST STEP):
  L-Isoleucine → BCAT (Step 1a) → L-2-Methylbutyryl-CoA (via KMV)
  L-2-Methylbutyryl-CoA → BCKDH (Step 1b) → (S)-2-Methylbutyryl-CoA (SBCAD step) → 2-Methylbutenoyl-CoA
  2-Methylbutenoyl-CoA (tiglyl-CoA) → [CROTONASE-2] → 2-Methyl-3-hydroxybutyryl-CoA
  2-Methyl-3-hydroxybutyryl-CoA → [HAD: HADH2/SCHAD] → 2-Methylacetoacetyl-CoA
  2-Methylacetoacetyl-CoA + CoA → [ACAT1 BLOCKED ⚠ — STEP 4, FINAL] → Propionyl-CoA + Acetyl-CoA

KEY POINT: ACAT1 is the TERMINAL step in isoleucine catabolism.
  → Propionyl-CoA is the PRODUCT (not the substrate) → C3 NORMAL (KEY NEGATIVE vs PA)
  → Tiglyl-CoA (upstream intermediate) accumulates → tiglylglycine (urine, pathognomonic)
  → 2-Methylacetoacetyl-CoA accumulates → 2-MAA (unstable) + 2M3HBA (stable, reliable)

BIOMARKER PATTERN — ACAT1 DEFICIENCY:
  2M3HBA (2-Methyl-3-hydroxybutyrate)    200–3000 mmol/mol Cr — PRIMARY, stable
  2-MAA (2-Methylacetoacetate)            50–1000 mmol/mol Cr — PATHOGNOMONIC, UNSTABLE (process STAT)
  Tiglylglycine                           50–500 mmol/mol Cr — PATHOGNOMONIC, glycine conjugate
  C5:1 (tiglylcarnitine)                  >0.1 µmol/L — NBS PRIMARY
  C5-OH                                   MILDLY elevated (2M3HBA-carnitine)
  β-OHB (3-Hydroxybutyrate)              MARKEDLY elevated during crisis (>3–15 mmol/L)
  Acetoacetate                            MARKEDLY elevated during crisis
  C3 (propionylcarnitine)                 NORMAL — KEY NEGATIVE vs PA
  Methylcitrate                           ABSENT — KEY NEGATIVE vs PA
  MMA                                     ABSENT — KEY NEGATIVE vs MMUT
  IVG (isovalerylglycine)                 ABSENT — KEY NEGATIVE vs IVA
  3-MGC                                   ABSENT — KEY NEGATIVE vs MGA
  3-HMG                                   ABSENT — KEY NEGATIVE vs HMGCL
  NH3                                     NORMAL inter-ictally; MILDLY elevated during severe crisis
  Glucose                                 NORMAL or MILDLY low (not the profound hypoglycaemia of HMGCL)
  Lactate/LP ratio                        NORMAL inter-ictally — KEY NEGATIVE vs PDH/PC

CRITICAL DISTINCTION — HYPERKETOSIS vs HYPOKETOSIS:
  ACAT1 deficiency: HYPERKETOSIS (ketone utilisation blocked → extreme β-OHB + AcAc during crisis)
  HMGCL deficiency: HYPOKETOSIS (ketone synthesis blocked → absent ketones + hypoglycaemia)
  → Opposite ketone patterns: HYPERKETOTIC crisis in ACAT1 vs HYPOketotic in HMGCL

WHY C3 IS NORMAL (EXAM PEARL):
  In PA (PCCA/PCCB deficiency): propionyl-CoA CARBOXYLASE is blocked → propionyl-CoA ACCUMULATES → C3 ↑
  In ACAT1 deficiency: isoleucine catabolism LAST step is blocked → propionyl-CoA is the PRODUCT
  → The SUBSTRATE (2-methylacetoacetyl-CoA) accumulates, NOT the product (propionyl-CoA)
  → C3 NORMAL; methylcitrate ABSENT — CRITICAL differentiation from PA

WHY KD IS ABSOLUTELY CONTRAINDICATED:
  KD → massive acetoacetyl-CoA generation (from β-oxidation + hepatic ketogenesis)
  ACAT1 blocked → acetoacetyl-CoA CANNOT be converted → extreme ketoacidosis
  Contrast: KD HELPS PDH (bypasses pyruvate → acetyl-CoA block)
  Contrast: KD HELPS in some mitochondrial chain defects
  In ACAT1: KD does the OPPOSITE → catastrophic crisis

PHENOTYPES (EPISODIC — most pronounced during illness, fasting, high protein intake):
  Episodic Ketoacidotic Crisis:   65% — Reye-like crises; severe vomiting, dehydration, pH <7.2
  Asymptomatic NBS-detected:      20% — C5:1 or 2M3HBA on expanded NBS; no crisis history
  Persistent Mild Disease:        10% — mild baseline elevation; occasional mild crises
  Severe Neurological:             5% — null alleles; Leigh-like; progressive ID

KEY CLINICAL NOTE:
  ACAT1 deficiency is characterised by EPISODIC crises — biomarkers can be COMPLETELY NORMAL between episodes.
  Urine OA (2M3HBA, tiglylglycine) and plasma acylcarnitines (C5:1) must be tested DURING or IMMEDIATELY
  AFTER a crisis for optimal diagnostic sensitivity. Emergency urine during acute ketoacidosis is ideal.
  2-Methylacetoacetic acid is extremely unstable (decarboxylates to acetone at room temperature) — freeze immediately.
"""

import random

SEED = 267  # deterministic
N_PATIENTS = 40

random.seed(SEED)

PHENOTYPES = [
    ("Episodic Ketoacidotic Crisis (Classic)", 0.65),
    ("Asymptomatic NBS-detected",             0.20),
    ("Persistent Mild Metabolic Disease",     0.10),
    ("Severe Neurological / Leigh-like",      0.05),
]

VARIANTS = [
    {"variant": "p.Thr297Met",          "freq": 25, "domain": "CoA-binding loop (near Cys89)", "phenotype": "Classic episodic / NBS", "note": "Most common; Asian populations; partial residual thiolase activity"},
    {"variant": "p.Arg208*",            "freq": 15, "domain": "Catalytic core (null)",          "phenotype": "Severe episodic / Leigh", "note": "Null allele; complete LOF; severe phenotype in homozygotes"},
    {"variant": "p.Ala301Pro",          "freq": 12, "domain": "Adjacent to Cys89 active site",  "phenotype": "Classic episodic",       "note": "Pro disrupts helix; abolishes catalysis; common in European cohorts"},
    {"variant": "p.Gly379Asp",          "freq": 10, "domain": "Tetramer interface",              "phenotype": "Classic / severe",        "note": "Disrupts homotetramer assembly; dominant negative effect in compound het"},
    {"variant": "p.Ser267Phe",          "freq":  9, "domain": "Helix-loop near active cleft",   "phenotype": "Mild / NBS",              "note": "Partial activity retained (~15%); milder phenotype; European founder"},
    {"variant": "c.IVS7+1G>A",          "freq":  8, "domain": "Splice donor (null)",             "phenotype": "Severe / Leigh-like",     "note": "Complete exon 7 skipping → premature stop; null allele"},
    {"variant": "p.Asn93Asp",           "freq":  6, "domain": "Adjacent to Cys89 catalytic Cys","phenotype": "Classic episodic",        "note": "Near active-site Cys89; disrupts substrate binding pocket"},
    {"variant": "p.Val151Ala",          "freq":  5, "domain": "Substrate-binding cleft",         "phenotype": "Mild / NBS-detected",     "note": "Partial activity; attenuated phenotype; crisis only with severe illness"},
    {"variant": "p.Ile199Thr",          "freq":  5, "domain": "CoA-thioester binding",           "phenotype": "Moderate episodic",       "note": "Reduces CoA affinity; temperature-sensitive stability"},
    {"variant": "Other / compound het", "freq":  5, "domain": "Various",                         "phenotype": "Variable",                "note": "Compound heterozygotes; phenotype depends on residual activity"},
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
        is_crisis   = ph == "Episodic Ketoacidotic Crisis (Classic)"
        is_nbs      = ph == "Asymptomatic NBS-detected"
        is_mild     = ph == "Persistent Mild Metabolic Disease"
        is_severe   = ph == "Severe Neurological / Leigh-like"

        onset_mo = (
            random.randint(6, 24)  if is_crisis else
            0                       if is_nbs    else
            random.randint(3, 36)  if is_mild   else
            random.randint(1, 12)
        )

        # 2M3HBA levels (stable primary marker)
        m3hba = (
            round(random.uniform(400, 3000))  if is_crisis else
            round(random.uniform(20,  200))   if is_nbs    else
            round(random.uniform(50,  500))   if is_mild   else
            round(random.uniform(200, 2000))
        )

        # Tiglylglycine (elevated in all, higher in crisis)
        tiglyl = (
            round(random.uniform(80,  500))  if is_crisis else
            round(random.uniform(10,  100))  if is_nbs    else
            round(random.uniform(20,  200))  if is_mild   else
            round(random.uniform(80,  400))
        )

        # C5:1 (tiglylcarnitine)
        c5_1 = round(random.uniform(0.15, 0.8) if is_crisis else
                      random.uniform(0.10, 0.3) if is_nbs    else
                      random.uniform(0.10, 0.5), 2)

        # beta-OHB during crisis (hyperketosis)
        bohb = (
            round(random.uniform(3.0, 15.0), 1)  if is_crisis else
            round(random.uniform(0.3,  1.5), 1)  if is_nbs    else
            round(random.uniform(0.5,  3.0), 1)  if is_mild   else
            round(random.uniform(2.0, 12.0), 1)
        )

        # C3 — NORMAL (KEY NEGATIVE vs PA)
        c3 = round(random.uniform(1.0, 3.5), 1)

        # NH3 — normal to mildly elevated
        nh3 = random.randint(30, 120) if is_crisis else random.randint(20, 50)

        # Variant assignment
        v_idx = min(int(random.betavariate(1.2, 3.5) * len(VARIANTS)), len(VARIANTS) - 1)
        variant = VARIANTS[v_idx]["variant"]

        seizures = (
            random.random() < 0.40 if is_crisis else
            False                   if is_nbs    else
            random.random() < 0.20 if is_mild   else
            random.random() < 0.70
        )

        intellectual_disability = (
            random.random() < 0.30 if is_crisis else
            False                   if is_nbs    else
            random.random() < 0.15 if is_mild   else
            True
        )

        cohort.append({
            "id":                         f"ACAT1-{i+1:03d}",
            "phenotype":                  ph,
            "variant":                    variant,
            "age_onset_months":           onset_mo,
            "m3hba_mmol_mol_cr":          m3hba,
            "tiglylglycine_mmol_mol_cr":  tiglyl,
            "c5_1_umol_l":                c5_1,
            "bohb_mmol_l":                bohb,
            "c3_umol_l":                  c3,      # NORMAL
            "nh3_umol_l":                 nh3,
            "c3_normal":                  True,    # KEY NEGATIVE flag
            "ketoacidotic_crisis":        is_crisis,
            "seizures":                   seizures,
            "intellectual_disability":    intellectual_disability,
        })
    return cohort


_COHORT = _make_cohort()


def get_overview():
    total     = len(_COHORT)
    crisis_n  = sum(1 for p in _COHORT if p["phenotype"] == "Episodic Ketoacidotic Crisis (Classic)")
    nbs_n     = sum(1 for p in _COHORT if p["phenotype"] == "Asymptomatic NBS-detected")
    mild_n    = sum(1 for p in _COHORT if p["phenotype"] == "Persistent Mild Metabolic Disease")
    severe_n  = sum(1 for p in _COHORT if p["phenotype"] == "Severe Neurological / Leigh-like")
    seizure_n = sum(1 for p in _COHORT if p["seizures"])
    id_n      = sum(1 for p in _COHORT if p["intellectual_disability"])

    avg_m3hba = round(sum(p["m3hba_mmol_mol_cr"] for p in _COHORT) / total)
    avg_tigl  = round(sum(p["tiglylglycine_mmol_mol_cr"] for p in _COHORT) / total)

    phenotype_dist = [
        {"class": "Episodic Ketoacidotic Crisis",  "n": crisis_n, "pct": round(crisis_n / total * 100)},
        {"class": "Asymptomatic NBS-detected",     "n": nbs_n,    "pct": round(nbs_n    / total * 100)},
        {"class": "Persistent Mild Disease",       "n": mild_n,   "pct": round(mild_n   / total * 100)},
        {"class": "Severe Neurological",           "n": severe_n, "pct": round(severe_n / total * 100)},
    ]

    kpi = {
        "patients":     {"label": "Patients (n)",          "value": str(total),      "color": "#1b5e20"},
        "crisis_pct":   {"label": "Ketoacidotic Crisis",   "value": f"{round(crisis_n/total*100)}%", "color": "#b71c1c"},
        "nbs_pct":      {"label": "NBS-detected",          "value": f"{round(nbs_n/total*100)}%",    "color": "#0d47a1"},
        "seizure_pct":  {"label": "With Seizures",         "value": f"{round(seizure_n/total*100)}%","color": "#e65100"},
        "id_pct":       {"label": "Intellectual Disability","value": f"{round(id_n/total*100)}%",    "color": "#4a148c"},
        "avg_m3hba":    {"label": "Avg 2M3HBA (mmol/mol Cr)","value": str(avg_m3hba),              "color": "#bf360c"},
    }

    return {
        "gene": "ACAT1",
        "disease": (
            "Beta-Ketothiolase Deficiency (T2 Deficiency / 2-Methylacetoacetic Aciduria / "
            "Alpha-Methylacetoacetic Aciduria / 2-Methyl-3-Hydroxybutyric Aciduria)"
        ),
        "omim_gene": "607809",
        "omim_disease": "203750",
        "locus": "11q22.3",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "prevalence": "~1:100,000–250,000",
        "pathway_step": (
            "STEP 4 (FINAL) of isoleucine catabolism: "
            "2-Methylacetoacetyl-CoA + CoA → [ACAT1] → Propionyl-CoA + Acetyl-CoA. "
            "ALSO: Ketone body utilisation: Acetoacetyl-CoA + CoA → [ACAT1] → 2 × Acetyl-CoA"
        ),
        "phenotype_dist": phenotype_dist,
        "kpi": kpi,
        "total": total,
        "crisis_n": crisis_n,
        "nbs_n": nbs_n,
        "mild_n": mild_n,
        "severe_n": severe_n,
        "avg_m3hba": avg_m3hba,
        "avg_tiglylglycine": avg_tigl,
        "hallmark_biomarker": (
            "PRIMARY: 2M3HBA (2-methyl-3-hydroxybutyrate) ELEVATED 200–3000 mmol/mol Cr — stable, reliable. "
            "PATHOGNOMONIC: Tiglylglycine ELEVATED + 2-methylacetoacetate (UNSTABLE — process STAT). "
            "NBS: C5:1 (tiglylcarnitine) PRIMARY trigger. "
            "KEY NEGATIVES: C3 NORMAL (not PA), Methylcitrate ABSENT, MMA ABSENT, 3-HMG ABSENT. "
            "HYPERKETOSIS (not HYPOketosis) during crisis — opposite of HMGCL."
        ),
        "dual_block_note": (
            "DUAL BLOCK: (A) Isoleucine catabolism — 2-methylacetoacetyl-CoA cannot be cleaved → "
            "2M3HBA + tiglylglycine accumulate. "
            "(B) Ketone body utilisation — acetoacetyl-CoA cannot be converted in peripheral tissues → "
            "EXTREME HYPERKETOSIS during any catabolic state (illness, fasting, high-protein). "
            "C3 NORMAL because propionyl-CoA is the PRODUCT of the blocked reaction (not the substrate)."
        ),
        "kd_ci_note": (
            "KD ABSOLUTE CI: Ketogenic diet floods acetoacetyl-CoA → cannot be utilised (ACAT1 blocked) → "
            "catastrophic ketoacidosis. "
            "Contrast: KD HELPS PDH deficiency (bypasses pyruvate oxidation block). "
            "In ACAT1: KD does the OPPOSITE — ABSOLUTE CONTRAINDICATION."
        ),
        "vpa_risk_note": (
            "VPA HIGH RISK: Valproic acid inhibits mitochondrial β-oxidation → increases acetoacetyl-CoA → "
            "worsens ketosis in ACAT1. Also promotes hyperammonemia. "
            "Not absolute CI (unlike in PA/MMA where valproyl-CoA directly inhibits critical enzymes) but "
            "HIGH RISK — avoid unless seizures uncontrolled with safer AEDs."
        ),
        "episodic_note": (
            "EPISODIC DISEASE: Biomarkers can be COMPLETELY NORMAL between crises. "
            "Diagnose during or immediately after ketoacidotic crisis — test urine OA (2M3HBA, tiglylglycine) "
            "and plasma acylcarnitines (C5:1) during ACUTE episode. "
            "2-Methylacetoacetate is extremely unstable — freeze urine IMMEDIATELY."
        ),
        "c3_normal_note": (
            "C3 NORMAL (KEY EXAM PEARL): In PA, propionyl-CoA CARBOXYLASE is blocked → propionyl-CoA "
            "ACCUMULATES → C3 ↑↑. In ACAT1, the TERMINAL step of isoleucine catabolism is blocked → "
            "propionyl-CoA is the PRODUCT (not formed yet) → C3 NORMAL. "
            "Methylcitrate = ABSENT. MMA = ABSENT. → Not propionic acidemia."
        ),
    }


def get_breakdown():
    biomarkers = {
        "m3hba": {
            "label": "2M3HBA (2-Methyl-3-hydroxybutyrate)",
            "normal": "<5 mmol/mol Cr",
            "status": "ELEVATED 200–3000 mmol/mol Cr — PRIMARY STABLE MARKER",
            "direction": "↑↑↑",
            "color": "danger",
            "rationale": (
                "2M3HBA is formed by reduction of 2-methylacetoacetyl-CoA (the accumulated substrate) "
                "via 2-methyl-3-hydroxybutyryl-CoA dehydrogenase. "
                "Unlike 2-methylacetoacetate (which decarboxylates rapidly), 2M3HBA is stable in urine. "
                "It is the most reliable biomarker for ACAT1 deficiency and can be detected by GC-MS. "
                "Reference range: <5 mmol/mol Cr. Crisis levels: 200–3000 mmol/mol Cr. "
                "Even between crises, mildly elevated 2M3HBA (5–50 mmol/mol Cr) may persist in classic phenotypes."
            ),
        },
        "tiglylglycine": {
            "label": "Tiglylglycine",
            "normal": "<5 mmol/mol Cr",
            "status": "ELEVATED 50–500 mmol/mol Cr — PATHOGNOMONIC",
            "direction": "↑↑",
            "color": "danger",
            "rationale": (
                "Tiglylglycine is the glycine conjugate of tiglyl-CoA (the upstream intermediate, "
                "2-methylbutenoyl-CoA). When ACAT1 is blocked, 2-methylacetoacetyl-CoA accumulates → "
                "retrograde flux to tiglyl-CoA → tiglylglycine (mitochondrial glycine N-acyltransferase). "
                "Tiglylglycine is a pathognomonic marker for ACAT1 deficiency (not seen in other organic acidemias). "
                "Combined elevation of 2M3HBA + tiglylglycine = DIAGNOSTIC FINGERPRINT of T2 deficiency."
            ),
        },
        "maa_2": {
            "label": "2-Methylacetoacetate (2-MAA)",
            "normal": "Absent or trace",
            "status": "PATHOGNOMONIC when detectable — but VERY UNSTABLE (decarboxylates to acetone rapidly)",
            "direction": "↑↑ (if fresh)",
            "color": "warning",
            "rationale": (
                "2-MAA is the direct accumulated substrate (free acid from 2-methylacetoacetyl-CoA). "
                "It is PATHOGNOMONIC for ACAT1 deficiency BUT is extremely unstable: "
                "it decarboxylates spontaneously to acetone + propionaldehyde at room temperature. "
                "Urine must be FROZEN IMMEDIATELY and processed within 24h. "
                "Even in optimal conditions, 2-MAA may be undetectable. "
                "The presence of 2-MAA confirms ACAT1 deficiency; its absence does NOT rule it out. "
                "Always rely on 2M3HBA + tiglylglycine as the more reliable markers."
            ),
        },
        "c5_1": {
            "label": "C5:1 Acylcarnitine (Tiglylcarnitine)",
            "normal": "<0.06 µmol/L",
            "status": "ELEVATED >0.1 µmol/L — PRIMARY NBS TRIGGER",
            "direction": "↑",
            "color": "danger",
            "rationale": (
                "Tiglylcarnitine (C5:1) is the primary NBS trigger for ACAT1 deficiency. "
                "Formed from tiglyl-CoA + carnitine via carnitine acyltransferase. "
                "Expanded NBS (tandem MS/MS) detects C5:1 elevation → triggers follow-up urine OA for 2M3HBA + tiglylglycine. "
                "C5:1 may be only mildly elevated between crises → confirm with urine OA during illness. "
                "Differential for C5:1 elevation: ACAT1 deficiency (most common), SBCAD deficiency (2-methylbutyryl-CoA dehydrogenase)."
            ),
        },
        "c5oh": {
            "label": "C5-OH Acylcarnitine",
            "normal": "<0.4 µmol/L",
            "status": "MILDLY ELEVATED — 2-methyl-3-hydroxybutyrylcarnitine (less than MCC/HMGCL)",
            "direction": "↑ (mild)",
            "color": "warning",
            "rationale": (
                "C5-OH elevation in ACAT1 deficiency represents 2-methyl-3-hydroxybutyrylcarnitine "
                "(acylcarnitine of 2M3HBA). Less elevated than in 3-MCC deficiency (where C5-OH is the primary NBS marker). "
                "Differential for C5-OH: 3-MCC deficiency, HMGCL deficiency, ACAT1 deficiency (mild), "
                "SCHAD deficiency, SBCAD deficiency. "
                "C5-OH + C5:1 together on NBS → strongly suggests ACAT1 deficiency."
            ),
        },
        "bohb": {
            "label": "β-Hydroxybutyrate (β-OHB) / Ketones",
            "normal": "<0.5 mmol/L (fasted); <0.1 mmol/L (fed)",
            "status": "MARKEDLY ELEVATED during crisis — 3–15 mmol/L (HYPERKETOSIS)",
            "direction": "↑↑↑ (crisis)",
            "color": "danger",
            "rationale": (
                "ACAT1 blocks ketone UTILISATION in peripheral tissues (acetoacetyl-CoA → 2 × acetyl-CoA). "
                "During illness/fasting/high-protein: liver generates excess ketones (normal hepatic ketogenesis via HMGCS2/HMGCL) "
                "→ ketones released into blood → CANNOT be utilised in periphery (ACAT1 blocked) → "
                "EXTREME HYPERKETOSIS (β-OHB 3–15 mmol/L during crisis). "
                "CRITICAL DISTINCTION: HMGCL deficiency → HYPOketotic (ketone synthesis blocked). "
                "ACAT1 deficiency → HYPERketotic (ketone utilisation blocked). "
                "Normal or elevated ketones = NOT HMGCL."
            ),
        },
        "c3": {
            "label": "C3 (Propionylcarnitine)",
            "normal": "<5.0 µmol/L",
            "status": "NORMAL — KEY NEGATIVE vs Propionic Acidemia (PA) — CRITICAL EXAM PEARL",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "In PA (PCCA/PCCB deficiency): propionyl-CoA CARBOXYLASE is blocked → propionyl-CoA ACCUMULATES → C3 ↑↑. "
                "In ACAT1 deficiency: the TERMINAL step of isoleucine catabolism is blocked → "
                "propionyl-CoA is the PRODUCT (not the substrate): "
                "2-methylacetoacetyl-CoA + CoA → propionyl-CoA + acetyl-CoA. "
                "When ACAT1 is blocked: 2-methylacetoacetyl-CoA accumulates (the substrate) "
                "→ propionyl-CoA CANNOT be generated → C3 remains NORMAL. "
                "C3 NORMAL + methylcitrate ABSENT = NOT propionic acidemia → points to ACAT1."
            ),
        },
        "methylcitrate": {
            "label": "Methylcitrate (Urine Organic Acids)",
            "normal": "Absent",
            "status": "ABSENT — KEY NEGATIVE vs PA (PCCA/PCCB)",
            "direction": "→ ABSENT",
            "color": "success",
            "rationale": (
                "Methylcitrate is PATHOGNOMONIC for propionic acidemia and MMA: "
                "propionyl-CoA + OAA → methylcitrate (via citrate synthase). "
                "In ACAT1 deficiency: propionyl-CoA CANNOT be generated (ACAT1 is blocked BEFORE propionyl-CoA forms) → "
                "methylcitrate = ABSENT. "
                "Methylcitrate ABSENT rules out PA and confirms ACAT1 is not a propionyl-CoA overflow disorder."
            ),
        },
        "mma": {
            "label": "Methylmalonic Acid (MMA)",
            "normal": "<5 mmol/mol Cr",
            "status": "ABSENT — KEY NEGATIVE vs MMUT (methylmalonyl-CoA mutase deficiency)",
            "direction": "→ ABSENT",
            "color": "success",
            "rationale": (
                "MMA is the PATHOGNOMONIC marker of methylmalonyl-CoA mutase (MMUT) deficiency and related CBL disorders. "
                "In ACAT1 deficiency: propionyl-CoA not generated (ACAT1 is the terminal step) → "
                "no propionyl-CoA → no methylmalonyl-CoA → MMA ABSENT. "
                "MMA ABSENT + C3 NORMAL + 2M3HBA ↑ + tiglylglycine ↑ = ACAT1 deficiency fingerprint."
            ),
        },
        "nh3": {
            "label": "Ammonia (NH3)",
            "normal": "<50 µmol/L",
            "status": "NORMAL inter-ictally; MILDLY elevated (50–120 µmol/L) during severe crisis",
            "direction": "→ NORMAL (inter-ictal) / ↑ mild (crisis)",
            "color": "success",
            "rationale": (
                "NH3 is NOT chronically elevated in ACAT1 deficiency (unlike UCDs or PA/MMA). "
                "Mild transient hyperammonemia can occur during severe ketoacidotic crisis: "
                "severe acidosis impairs urea cycle secondarily (substrate diversion). "
                "NH3 in ACAT1 crisis: typically <150 µmol/L (vs >500 µmol/L in severe UCDs). "
                "NH3 chronically normal = KEY NEGATIVE vs UCDs, PA, MMA."
            ),
        },
        "lactate_lp": {
            "label": "Lactate / LP Ratio",
            "normal": "<2.0 mmol/L; LP <20:1",
            "status": "NORMAL inter-ictally — KEY NEGATIVE vs PDH/PC deficiency",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "Lactate and LP ratio are NORMAL in ACAT1 deficiency inter-ictally. "
                "This is a KEY NEGATIVE vs PDH deficiency (LP elevated due to pyruvate stuck) "
                "and PC deficiency (LP elevated due to OAA deficit → NADH excess). "
                "ACAT1 does not affect pyruvate metabolism. "
                "Mild lactate elevation may occur during severe ketoacidotic crisis (secondary to circulatory compromise) "
                "but LP ratio is not elevated."
            ),
        },
        "glucose": {
            "label": "Blood Glucose",
            "normal": "3.9–6.1 mmol/L",
            "status": "NORMAL or MILDLY low — NOT the profound HYPOglycaemia of HMGCL",
            "direction": "→ NORMAL or ↓ mild",
            "color": "success",
            "rationale": (
                "ACAT1 deficiency does NOT cause profound hypoglycaemia. "
                "Gluconeogenesis is intact (PC, FBPase, G6Pase all normal). "
                "Mild glucose lowering may occur during severe vomiting/poor intake in crisis. "
                "CRITICAL: HMGCL deficiency causes SEVERE hypoketotic HYPOglycaemia (hallmark). "
                "ACAT1 causes HYPERketotic crisis with NORMAL or near-normal glucose → key distinction."
            ),
        },
    }

    enzyme_mechanism = {
        "function": (
            "ACAT1 (ACAT1 gene, 11q22.3) encodes mitochondrial acetoacetyl-CoA thiolase (T2), 427 aa.\n"
            "Homotetrameric mitochondrial matrix enzyme. Active site: Cys89 (catalytic nucleophile).\n"
            "Catalyses TWO distinct reactions:\n"
            "  (A) Isoleucine catabolism (FINAL STEP 4):\n"
            "      2-Methylacetoacetyl-CoA + CoA → Propionyl-CoA + Acetyl-CoA\n"
            "      (thiolytic cleavage; CoA attacks C2-C3 bond of 2-methylacetoacetyl-CoA)\n"
            "  (B) Ketone body utilisation (ketolysis) in peripheral tissues:\n"
            "      Acetoacetyl-CoA + CoA → 2 × Acetyl-CoA\n"
            "      (enables peripheral oxidation of acetoacetate/β-OHB to acetyl-CoA for TCA)"
        ),
        "reaction_a": (
            "REACTION A (Isoleucine catabolism, Step 4 — BLOCKED):\n"
            "2-Methylacetoacetyl-CoA + CoA → [ACAT1 ⚠ BLOCKED] → Propionyl-CoA + Acetyl-CoA\n"
            "ACAT1 LOF → 2-Methylacetoacetyl-CoA ACCUMULATES:\n"
            "  → Hydrolysis → 2-Methylacetoacetate (2-MAA, PATHOGNOMONIC, UNSTABLE)\n"
            "  → 2-MAA reduction → 2M3HBA (stable, PRIMARY diagnostic marker)\n"
            "  → Upstream: Tiglyl-CoA accumulates → Tiglylglycine (PATHOGNOMONIC)"
        ),
        "reaction_b": (
            "REACTION B (Ketolysis — BLOCKED in peripheral tissues):\n"
            "Acetoacetyl-CoA + CoA → [ACAT1 ⚠ BLOCKED] → ↛ 2 × Acetyl-CoA\n"
            "ACAT1 LOF → acetoacetyl-CoA CANNOT be utilised in peripheral tissues:\n"
            "  → Hepatic ketogenesis (via HMGCS2 + HMGCL) still INTACT\n"
            "  → Ketones released normally by liver into blood\n"
            "  → Peripheral tissues CANNOT utilise acetoacetate/β-OHB via ACAT1\n"
            "  → EXTREME HYPERKETOSIS during fasting/illness (β-OHB 3–15 mmol/L)"
        ),
        "isoleucine_path": (
            "ISOLEUCINE CATABOLISM — context (ACAT1 = LAST STEP):\n"
            "L-Isoleucine → [BCAT] → KMV (2-keto-3-methylvalerate)\n"
            "KMV → [BCKDH] → (S)-2-Methylbutyryl-CoA → [SBCAD] → Tiglyl-CoA\n"
            "Tiglyl-CoA → [Enoyl-CoA hydratase] → 2-Methyl-3-hydroxybutyryl-CoA\n"
            "2M3HB-CoA → [HADH2/SCHAD] → 2-Methylacetoacetyl-CoA\n"
            "2-Methylacetoacetyl-CoA + CoA → [ACAT1 ⚠ BLOCKED — STEP 4] → ↛ Propionyl-CoA + Acetyl-CoA"
        ),
        "why_c3_normal": (
            "WHY C3 IS NORMAL — EXAM PEARL:\n"
            "In PA: Propionyl-CoA CARBOXYLASE blocked → propionyl-CoA ACCUMULATES → C3 ↑↑\n"
            "In ACAT1: Propionyl-CoA is the PRODUCT of the blocked step (not the substrate)\n"
            "  → The substrate (2-methylacetoacetyl-CoA) accumulates\n"
            "  → Propionyl-CoA CANNOT be generated yet (blocked before it forms)\n"
            "  → C3 remains NORMAL; methylcitrate ABSENT\n"
            "ACAT1 ≠ PA; C3 NORMAL distinguishes them."
        ),
        "ketone_hyperketosis": (
            "HYPERKETOSIS MECHANISM:\n"
            "Liver: HMGCS2 + HMGCL intact → normal ketone synthesis → β-OHB + acetoacetate released\n"
            "Peripheral tissues: ACAT1 BLOCKED → acetoacetate → acetoacetyl-CoA cannot → 2 × Acetyl-CoA\n"
            "Result: EXTREME KETONE ACCUMULATION during illness/fasting\n"
            "Contrast HMGCL: HYPOketotic (synthesis blocked, no ketones released)\n"
            "ACAT1: HYPERketotic (utilisation blocked, ketones accumulate)"
        ),
    }

    seizure_types = [
        {"type": "Seizures during ketoacidotic crisis (metabolic encephalopathy)", "pct": 40, "note": "Most common; metabolic encephalopathy from extreme acidosis + hyperketonaemia → cortical irritation; resolve with crisis treatment"},
        {"type": "Focal seizures (between crises, severe phenotype)",               "pct": 20, "note": "Seen in severe/null allele patients; chronic metabolite toxicity proposed; LEV first-line AED"},
        {"type": "Generalised tonic-clonic",                                        "pct": 15, "note": "During crisis or in severe neurological phenotype; metabolic origin"},
        {"type": "Drug-resistant epilepsy (DRE)",                                   "pct":  5, "note": "Rare; severe neurological phenotype only; careful AED selection (avoid VPA)"},
    ]

    treatments = [
        {
            "therapy": "Isoleucine restriction",
            "level": "A",
            "dose": "Isoleucine 50–120 mg/kg/day via isoleucine-free/low-isoleucine formula; natural protein 0.5–1.0 g/kg/day",
            "rationale": (
                "Reduces 2-methylacetoacetyl-CoA substrate from isoleucine catabolism. "
                "PRIMARY dietary strategy — isoleucine (not leucine) is the key substrate. "
                "Titrate by plasma isoleucine (target 40–150 µmol/L) + urine 2M3HBA + tiglylglycine. "
                "Note: leucine restriction alone is INSUFFICIENT — isoleucine is the specific culprit."
            ),
        },
        {
            "therapy": "L-Carnitine supplementation",
            "level": "A",
            "dose": "100–200 mg/kg/day oral; IV during crisis (50–100 mg/kg)",
            "rationale": (
                "Secondary carnitine depletion from tiglylcarnitine (C5:1) and 2M3HBA-carnitine excretion. "
                "Maintains free carnitine pool, facilitates excretion of accumulated acyl-CoA species. "
                "Essential during crisis when urinary carnitine losses are maximal."
            ),
        },
        {
            "therapy": "Fasting avoidance (emergency glucose protocol)",
            "level": "A",
            "dose": "Max fast: 4 h (infant), 6 h (child), 8 h (adult). IV glucose + bicarbonate during crisis",
            "rationale": (
                "CRITICAL: Fasting triggers catabolism → isoleucine mobilisation + ketone body surge → "
                "ACAT1-blocked ketolysis → hyperketotic crisis. Written sick-day protocol mandatory. "
                "IV glucose (GIR 6–10 mg/kg/min) + NaHCO₃ is the acute crisis treatment. "
                "Glucose suppresses lipolysis/ketogenesis; bicarbonate corrects acidosis."
            ),
        },
        {
            "therapy": "IV Glucose + Sodium Bicarbonate (acute crisis)",
            "level": "A",
            "dose": "IV 10% glucose; GIR 6–10 mg/kg/min; NaHCO₃ 1–2 mmol/kg slow IV for pH <7.2",
            "rationale": (
                "First-line acute treatment for ketoacidotic crisis (pH <7.2, β-OHB >5 mmol/L). "
                "Glucose suppresses endogenous ketogenesis (inhibits lipolysis) and provides energy "
                "to stop catabolism. Bicarbonate corrects severe acidosis. "
                "Avoid high-protein IV amino acid solutions — isoleucine content worsens crisis."
            ),
        },
        {
            "therapy": "Levetiracetam (LEV)",
            "level": "B",
            "dose": "20–40 mg/kg/day in 2 divided doses",
            "rationale": (
                "First-line AED for ACAT1-associated seizures. No carnitine depletion (unlike VPA). "
                "No direct ACAT1 interaction. Safe metabolic profile in organic acidemias."
            ),
        },
        {
            "therapy": "Valproic Acid (VPA)",
            "level": "HIGH RISK",
            "dose": "AVOID if possible. If used: low dose, close monitoring of carnitine, ammonia, LFTs",
            "rationale": (
                "VPA is HIGH RISK in ACAT1 deficiency via two mechanisms: "
                "(1) VPA inhibits mitochondrial β-oxidation → increases free fatty acids → promotes "
                "endogenous ketogenesis → worsens hyperketosis already present; "
                "(2) VPA promotes hyperammonemia (inhibits CPS1 indirectly). "
                "Not absolute CI (no direct valproyl-CoA:ACAT1 active-site inhibition) but HIGH RISK — "
                "prefer LEV, OXC, or LTG. Use VPA only if seizures uncontrolled."
            ),
        },
        {
            "therapy": "Ketogenic Diet (KD)",
            "level": "ABSOLUTE CI",
            "dose": "NEVER use — absolute contraindication",
            "rationale": (
                "KD floods the acetoacetyl-CoA pathway from β-oxidation and hepatic ketogenesis. "
                "ACAT1 is blocked → acetoacetyl-CoA CANNOT be utilised → "
                "catastrophic hyperketotic acidosis potentially fatal. "
                "KD is ABSOLUTELY CONTRAINDICATED in ACAT1 deficiency. "
                "Contrast: KD HELPS PDH deficiency (bypasses pyruvate block). "
                "KD KILLS ACAT1 — critical exam distinction."
            ),
        },
        {
            "therapy": "High-protein diet / High-isoleucine intake",
            "level": "EXTREME HAZARD",
            "dose": "AVOID — dietary isoleucine drives substrate accumulation",
            "rationale": (
                "High-protein diet provides excess isoleucine → increased isoleucine catabolism → "
                "2-methylacetoacetyl-CoA surge → crisis. "
                "All dietary protein sources contain isoleucine — calculate isoleucine intake carefully. "
                "Sick-day protocol must include protein restriction during illness."
            ),
        },
    ]

    systemic_features = [
        {"feature": "Episodic ketoacidotic crisis",          "pct": 65, "note": "Reye-like; vomiting, dehydration, pH <7.2, β-OHB >5; triggers: fever, fasting, high protein, missed diet"},
        {"feature": "Vomiting (acute crisis presentation)",  "pct": 82, "note": "Most prominent crisis symptom; often mimics surgical abdomen before metabolic diagnosis"},
        {"feature": "Metabolic encephalopathy (crisis)",     "pct": 55, "note": "Lethargy → coma during severe acidosis; normalises with glucose + bicarbonate treatment"},
        {"feature": "Developmental delay / speech delay",   "pct": 38, "note": "Variable; correlates with crisis frequency and residual ACAT1 activity; improves with fewer crises"},
        {"feature": "Intellectual disability",               "pct": 30, "note": "Moderate in severe/null allele patients; minimal in NBS-detected cases treated early"},
        {"feature": "Seizures (non-crisis)",                 "pct": 22, "note": "Focal > generalised; especially in severe phenotype; LEV first-line AED"},
        {"feature": "Asymptomatic (NBS only)",               "pct": 20, "note": "C5:1 elevated on NBS; no crisis; requires dietary monitoring + sick-day protocol"},
        {"feature": "Hypotonia",                             "pct": 28, "note": "Generalised; correlates with phenotypic severity; improves with metabolic control"},
        {"feature": "Cardiomyopathy",                        "pct":  0, "note": "ABSENT — KEY NEGATIVE vs Barth syndrome (TAZ, MGA-II); helps distinguish ACAT1 from TAZ"},
        {"feature": "Neutropenia",                           "pct":  0, "note": "ABSENT — KEY NEGATIVE vs Barth syndrome (TAZ, MGA-II)"},
        {"feature": "Chronic lactic acidosis",               "pct":  0, "note": "ABSENT inter-ictally — KEY NEGATIVE vs PDH/PC deficiency; LP ratio NORMAL"},
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
        "ACAT1 (Mitochondrial Acetoacetyl-CoA Thiolase / T2)": (
            "ACAT1 (gene: ACAT1, 11q22.3, AR) encodes mitochondrial acetoacetyl-CoA thiolase, 427 aa. "
            "Homotetrameric; active site Cys89. DUAL FUNCTION: "
            "(A) Terminal enzyme of isoleucine catabolism (step 4): 2-methylacetoacetyl-CoA → propionyl-CoA + acetyl-CoA. "
            "(B) Ketolysis in peripheral tissues: acetoacetyl-CoA → 2 × acetyl-CoA. "
            "Also called T2 (thiolase 2) or beta-ketothiolase. Distinct from CT (cytoplasmic thiolase, ACAT2). "
            "OMIM Gene *607809."
        ),
        "Beta-Ketothiolase Deficiency (T2 Deficiency / 2-Methylacetoacetic Aciduria)": (
            "Autosomal recessive IEM (biallelic LOF in ACAT1, 11q22.3). OMIM Disease #203750. "
            "Prevalence: ~1:100,000–250,000. Also called alpha-methylacetoacetic aciduria or 2-methyl-3-hydroxybutyric aciduria. "
            "DUAL BLOCK: isoleucine catabolism STEP 4 (accumulates 2M3HBA + tiglylglycine) "
            "AND ketone body utilisation (peripheral hyperketosis). "
            "Episodic disease — biomarkers NORMAL between crises. "
            "4 phenotypes: Episodic Ketoacidotic Crisis (65%), Asymptomatic NBS-detected (20%), "
            "Persistent Mild (10%), Severe Neurological (5%)."
        ),
        "2M3HBA (2-Methyl-3-Hydroxybutyrate) — Primary Marker": (
            "ELEVATED 200–3000 mmol/mol Cr in ACAT1 deficiency — primary stable diagnostic marker. "
            "Formed by reduction of accumulated 2-methylacetoacetyl-CoA via 2-methyl-3-hydroxybutyryl-CoA dehydrogenase. "
            "Detected by GC-MS urine organic acids (OA). Reference: <5 mmol/mol Cr. "
            "Stable in urine (unlike 2-MAA). May be mildly elevated (5–50 mmol/mol Cr) between crises. "
            "Combined with tiglylglycine = diagnostic fingerprint of ACAT1 deficiency."
        ),
        "Tiglylglycine — Pathognomonic Marker": (
            "PATHOGNOMONIC for ACAT1 deficiency (and SBCAD deficiency). "
            "Formed from tiglyl-CoA (upstream intermediate in isoleucine catabolism) conjugated to glycine "
            "by mitochondrial glycine N-acyltransferase. "
            "When ACAT1 is blocked: 2-methylacetoacetyl-CoA accumulates → retrograde flux to tiglyl-CoA → tiglylglycine. "
            "Detected by GC-MS urine OA. Reference: <5 mmol/mol Cr. Crisis levels: 50–500 mmol/mol Cr. "
            "Tiglylglycine + 2M3HBA elevated = ACAT1 deficiency diagnosis confirmed."
        ),
        "2-Methylacetoacetate (2-MAA) — Unstable STAT Biomarker": (
            "PATHOGNOMONIC when detectable but EXTREMELY UNSTABLE: decarboxylates spontaneously to acetone "
            "+ propionaldehyde at room temperature. "
            "Clinical rule: freeze urine IMMEDIATELY if ACAT1 crisis suspected. Process within 24h. "
            "Even with optimal handling, 2-MAA may be absent. Its presence confirms ACAT1 but absence does NOT rule it out. "
            "Always confirm with 2M3HBA + tiglylglycine (stable markers)."
        ),
        "HYPERKETOSIS — KEY Distinction from HMGCL Deficiency": (
            "ACAT1 deficiency: HYPERKETOTIC CRISIS (ketone utilisation blocked). "
            "Hepatic ketogenesis (HMGCS2 + HMGCL) is INTACT → liver produces normal ketones → "
            "ketones released into blood → peripheral tissues CANNOT utilise them (ACAT1 blocked) → "
            "extreme β-OHB + acetoacetate during illness. "
            "HMGCL deficiency: HYPOKETOTIC HYPOGLYCAEMIA (ketone synthesis blocked — HMGCL is the ketogenesis enzyme). "
            "ACAT1: hyperketotic, glucose near-normal; HMGCL: hypoketotic, severe hypoglycaemia. "
            "Opposite ketone patterns — critical diagnostic and treatment distinction."
        ),
        "C3 NORMAL — KEY NEGATIVE vs Propionic Acidemia (Exam Pearl)": (
            "In PA (PCCA or PCCB deficiency): propionyl-CoA CARBOXYLASE blocked → propionyl-CoA ACCUMULATES → C3 ↑↑. "
            "In ACAT1 deficiency: propionyl-CoA is the PRODUCT of the ACAT1 reaction (reaction A). "
            "When ACAT1 is blocked: the SUBSTRATE (2-methylacetoacetyl-CoA) accumulates, NOT propionyl-CoA. "
            "Propionyl-CoA cannot even be generated (the step that makes it is blocked). "
            "Therefore C3 = NORMAL; methylcitrate = ABSENT; MMA = ABSENT. "
            "C3 normal + tiglylglycine ↑ + 2M3HBA ↑ = ACAT1, not PA."
        ),
        "KD ABSOLUTE CI — Critical Treatment Distinction": (
            "Ketogenic Diet is ABSOLUTELY CONTRAINDICATED in ACAT1 deficiency. "
            "KD massively increases β-oxidation → generates acetoacetyl-CoA → "
            "peripheral tissues cannot utilise it (ACAT1 blocked) → catastrophic hyperketotic acidosis. "
            "Contrast: KD HELPS PDH deficiency (bypasses pyruvate → acetyl-CoA block). "
            "Contrast: KD HELPS some mitochondrial chain defects. "
            "KD KILLS ACAT1: floods the utilisation pathway that is already blocked. "
            "ACAT1 vs PDH: KD ABSOLUTE CI in ACAT1; KD FIRST-LINE in PDH. Critical exam distinction."
        ),
        "Episodic Disease — When to Test": (
            "ACAT1 deficiency is EPISODIC: biomarkers can be COMPLETELY NORMAL between crises. "
            "Urine OA (2M3HBA, tiglylglycine) and plasma acylcarnitines (C5:1, C5-OH) normalise inter-ictally. "
            "Optimal diagnostic samples: urine during ACUTE ketoacidotic crisis or immediately after admission. "
            "Emergency urine collection (catheter if necessary) during acute crisis is the most sensitive test. "
            "C5:1 by tandem MS/MS on NBS dried blood spot is the primary population-level screen. "
            "Confirmatory: urine OA + plasma acylcarnitines + ACAT1 enzyme assay in fibroblasts + sequencing."
        ),
        "Isoleucine Restriction — Not Leucine": (
            "The dietary intervention is ISOLEUCINE restriction (not leucine). "
            "Isoleucine is the specific amino acid that feeds into the ACAT1-deficient pathway "
            "(via 2-methylbutyryl-CoA → tiglyl-CoA → 2-methylacetoacetyl-CoA). "
            "Leucine catabolism goes through a completely different pathway (IVD → MCC → AUH → HMGCL). "
            "Prescribing leucine-restricted formula (as in MCC or IVA) would be incorrect for ACAT1 deficiency. "
            "Use ISOLEUCINE-free or isoleucine-reduced formula. Target plasma isoleucine 40–150 µmol/L."
        ),
        "Inheritance & Epidemiology": (
            "Autosomal recessive (AR); biallelic LOF in ACAT1 gene (11q22.3). "
            "Prevalence: ~1:100,000–250,000. "
            "Most common in Asian populations (Japan, Korea, Taiwan): p.Thr297Met (~25% of alleles). "
            "European: p.Ala301Pro and p.Ser267Phe are more frequent. "
            "Both sexes equally affected. No cardiac or haematological features (unlike Barth syndrome). "
            "NBS coverage: C5:1 (tiglylcarnitine) detected by expanded tandem MS/MS panels. "
            "Outcome excellent with early diagnosis, isoleucine restriction, and crisis prevention."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN keys ===", list(get_breakdown().keys()))
    print("\n=== DEFINITIONS keys ===", list(get_definitions().keys()))
    print(f"\n✓ ACAT1 dashboard: {N_PATIENTS} patients, seed={SEED}")
