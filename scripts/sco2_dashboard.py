#!/usr/bin/env python3
"""SCO2 — Fatal Infantile Hypertrophic Cardiomyopathy due to Complex IV (COX) Deficiency.

SCO2 (Synthesis of Cytochrome c Oxidase 2) is a copper chaperone essential for
delivering copper to the CuA binuclear copper centre of COX2 (MT-CO2), the
catalytic core subunit of cytochrome c oxidase (Complex IV).

  SCO2 gene           OMIM *604272
  Disease (CEMCOX3)   OMIM #619012
  Complex IV (COX)    deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (SCO2 / CuA copper delivery / Complex IV assembly):
SCO2 is a thioredoxin-fold protein located in the mitochondrial intermembrane
space (IMS), anchored to the inner mitochondrial membrane (IMM). It contains a
conserved CXXXC copper-binding motif that transiently binds a single Cu+ ion.
  • Copper trafficking pathway: COX17 (IMS copper pool)
    → SCO2 (Cu+ delivery to CuA)
    → SCO1 (Cu2+ → Cu+ reduction + CuA maturation)
    → COX2 (MT-CO2) CuA binuclear copper centre
  • The CuA centre of COX2 is the primary electron acceptor from cytochrome c —
    it accepts electrons from Cyt c, funnels them to heme a (COX1), then to
    heme a3 / CuB active site for O2 reduction
  • Without SCO2: CuA fails to acquire copper → COX2 cannot be incorporated
    into the holoenzyme → Complex IV assembly arrests → functional COX severely
    reduced (<15% of normal)
  • Isolated COX deficiency: Complexes I, II, III NORMAL — same biochemical
    fingerprint as SURF1, but with a DISTINCT clinical signature: HCM 100%

CARDIAC MECHANISM (HCM):
The heart is the most energy-demanding muscle per gram in the body. Cardiac
myocytes rely almost exclusively on OXPHOS for ATP synthesis — glycolysis
contributes <10% of cardiac energy. With COX (Complex IV) severely deficient:
  • ATP production collapses in cardiomyocytes → compensatory hypertrophy
  • Asymmetric septal hypertrophy, wall thickness >2SD above mean for age
  • Eventually progresses to dilated CMP + systolic failure
  • Unlike SURF1 (HCM only 10%): SCO2 HCM is UNIVERSAL (100%) and DOMINANT
    — cardiac failure IS the leading cause of death in SCO2 disease

MOLECULAR: Biallelic (AR) loss-of-function SCO2 variants:
  — p.Glu140Lys (c.418G>A): most common allele, ~60% of alleles worldwide;
    Greek/Italian/Mediterranean founder; CXXXC adjacent; severe LOF
  — p.Leu151Pro (c.452T>C): second most common; structural disruption
  — Null (nonsense/frameshift): severe phenotype; compound heterozygous with
    missense alleles are the commonest compound genotype
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 595
DISEASE_ID   = "sco2"
DISEASE_NAME = "SCO2 Fatal Infantile HCM (Complex IV / COX Deficiency)"
GENE         = "SCO2"
OMIM_GENE    = "*604272"
OMIM_DISEASE = "#619012"
CHROMOSOME   = "22q13.33"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal / Early infantile (days to weeks of life; range birth – 4 months)"
COHORT_SIZE  = 40
COLOR        = "#b71c1c"   # deep crimson — cardiac emergency / HCM / SCO2
LIGHT        = "#ffebee"


# Genotype pool
GENO_GLU140LYS_HOM   = "p.Glu140Lys homozygous (c.418G>A) — Greek/Italian/Mediterranean founder"
GENO_GLU140LYS_CPX   = "p.Glu140Lys / p.Leu151Pro — compound heterozygous (most common compound)"
GENO_GLU140LYS_NULL  = "p.Glu140Lys / truncating — compound heterozygous (null allele)"
GENO_OTHER_CPX       = "Two missense alleles (other) — compound heterozygous"

GENO_POOL    = [GENO_GLU140LYS_HOM, GENO_GLU140LYS_CPX, GENO_GLU140LYS_NULL, GENO_OTHER_CPX]
GENO_WEIGHTS = [0.25,                0.35,                0.25,                0.15]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient SCO2 cohort (seed-595)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient SCO2 cohort (seed-595).

    All patients have HCM + isolated COX deficiency.
    Additional features stochastic per published frequencies.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno     = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex      = "F" if rng.random() < 0.50 else "M"

        # Onset age (weeks of life)
        onset_wk = round(rng.uniform(0.5, 12.0), 1)

        # Lactate (mmol/L) — typically elevated
        lactate  = round(rng.uniform(3.0, 14.0), 1)

        # COX residual activity (% of normal in muscle/fibroblasts)
        is_null  = "truncating" in geno or "null" in geno
        if is_null:
            cox_pct = round(rng.uniform(2.0, 12.0), 1)
        else:
            cox_pct = round(rng.uniform(5.0, 20.0), 1)

        # HCM wall thickness (mm) — all patients, some severe
        hcm_wall = round(rng.uniform(8.0, 22.0), 1)

        # Features
        hcm           = True            # 100% — CARDINAL
        encephalo     = rng.random() < 0.90   # 90%
        lactic_ac     = rng.random() < 0.90   # 90%
        hypotonia     = rng.random() < 0.80   # 80%
        hypertonia    = rng.random() < 0.40   # 40% (some have both)
        leigh_mri     = rng.random() < 0.55   # 55%
        seizures      = rng.random() < 0.40   # 40%
        resp_comp     = rng.random() < 0.65   # 65%
        optic_atrph   = rng.random() < 0.15   # 15%
        failure_to_th = rng.random() < 0.75   # 75%

        # Outcome (survival) — usually fatal in first year
        survival_mo = round(rng.uniform(0.5, 18.0), 1)
        is_null_geno = "truncating" in geno
        if is_null_geno:
            alive = rng.random() < 0.15
        else:
            alive = rng.random() < 0.30
        if alive:
            outcome = f"Alive {survival_mo:.1f} months (last follow-up)"
        else:
            outcome = f"Died at {survival_mo:.1f} months"

        # Treatments
        txs = ["CoQ10/Ubiquinol (Level C)"]
        txs.append("Riboflavin B2 (Level C)")
        txs.append("L-Carnitine (Level C — secondary deficiency)")
        if lactic_ac:   txs.append("Thiamine B1 (Level C)")
        txs.append("IV Dextrose GIR 6–8 (crisis / perioperative)")
        if seizures:    txs.append("LEV (preferred AED — renal excretion, no mito toxicity)")
        if hcm:         txs.append("Diuretics (furosemide/spironolactone for HF symptoms)")
        if resp_comp:   txs.append("NIV / mechanical ventilation (respiratory support)")

        # Alerts
        alerts_list = [
            "🚫 VPA ABSOLUTE CI — CoA sequestration / POLG inhibition / hepatotoxicity",
            "🚫 Metformin ABSOLUTE CI — Complex I inhibition → lactic crisis",
            "🚫 Linezolid ABSOLUTE CI — 23S rRNA inhibition → eliminates residual MT-CO2 synthesis",
            "🚫 Propofol ABSOLUTE CI — PRIS: Complex IV inhibition + cardiac depression → catastrophic in SCO2 HCM",
            "🚫 Digoxin AVOID — positive inotrope worsens outflow obstruction in hypertrophic CMP",
            "🚫 KD CONTRAINDICATED — beta-oxidation requires functional Complex IV",
        ]
        if seizures:
            alerts_list.append("⚠ Seizures: LEV ONLY — NO VPA (ABSOLUTE CI), NO phenobarbital (Complex I inhibitor)")
        alerts = "; ".join(alerts_list)

        # Feature string
        feats = ["HCM (100% — CARDINAL, UNIQUE)"]
        if encephalo:   feats.append("Encephalomyopathy")
        if lactic_ac:   feats.append("Lactic acidosis")
        if hypotonia:   feats.append("Hypotonia")
        if hypertonia:  feats.append("Hypertonia")
        if leigh_mri:   feats.append("Leigh-like MRI")
        if seizures:    feats.append("Seizures")
        if resp_comp:   feats.append("Resp. compromise")
        if optic_atrph: feats.append("Optic atrophy")
        if failure_to_th: feats.append("Failure to thrive")

        patients.append({
            "id":         f"SCO2-{i:03d}",
            "geno":       geno,
            "sex":        sex,
            "onset_wk":   onset_wk,
            "lactate":    lactate,
            "cox_pct":    cox_pct,
            "hcm_wall":   hcm_wall,
            "features":   ", ".join(feats),
            "treatments": ", ".join(txs),
            "alerts":     alerts,
            "outcome":    outcome,
        })
    return patients


# ── Public API functions ─────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    """SCO2 overview — gene, disease identity, KPIs, contraindications."""
    rng = _rng()
    cohort = _build_cohort(rng)

    n = len(cohort)

    def pct_feat(feat: str) -> float:
        return round(sum(1 for p in cohort if feat in p["features"]) / n * 100)

    n_hcm        = n   # 100%
    n_encephalo  = sum(1 for p in cohort if "Encephalo" in p["features"])
    n_lactic     = sum(1 for p in cohort if "Lactic" in p["features"])
    n_hypotonia  = sum(1 for p in cohort if "Hypotonia" in p["features"])
    n_leigh      = sum(1 for p in cohort if "Leigh" in p["features"])
    n_seizures   = sum(1 for p in cohort if "Seizures" in p["features"])
    n_resp       = sum(1 for p in cohort if "Resp." in p["features"])
    n_alive      = sum(1 for p in cohort if p["outcome"].startswith("Alive"))
    mean_cox     = round(sum(p["cox_pct"] for p in cohort) / n, 1)
    mean_onset   = round(sum(p["onset_wk"] for p in cohort) / n, 1)
    mean_lact    = round(sum(p["lactate"] for p in cohort) / n, 1)
    mean_wall    = round(sum(p["hcm_wall"] for p in cohort) / n, 1)

    return {
        "gene":         "SCO2 (Synthesis of Cytochrome c Oxidase 2)",
        "protein":      "SCO2-266aa — Thioredoxin-fold IMS copper chaperone; CXXXC motif; delivers Cu+ to CuA binuclear centre of COX2 (MT-CO2); IMM-anchored, IMS-exposed",
        "disease":      "Fatal Infantile Hypertrophic Cardiomyopathy with Encephalomyopathy due to Complex IV / COX Deficiency (CEMCOX3)",
        "omim_gene":    "*604272 (SCO2)",
        "omim_disease": "#619012 (Cardioencephalomyopathy, Fatal Infantile, due to COX Deficiency 3)",
        "chromosome":   "22q13.33",
        "inheritance":  "AR (autosomal recessive biallelic) — 25% recurrence per pregnancy",
        "onset":        "Neonatal / early infantile; median days to weeks of life; range birth – 4 months",
        "cohort":       f"{n} patients · seed-595 · SCO2 biallelic",
        "mechanism": (
            "SCO2 is a thioredoxin-fold protein in the mitochondrial intermembrane space (IMS), "
            "anchored to the inner mitochondrial membrane (IMM). Its CXXXC copper-binding motif "
            "transiently chelates Cu+ ions from the IMS copper pool (supplied by COX17). "
            "SCO2 delivers this copper to the CuA binuclear copper centre of COX2 (MT-CO2), "
            "the mitochondrially encoded subunit of Complex IV that serves as the PRIMARY "
            "electron acceptor from cytochrome c. "
            "Without SCO2: CuA centre remains copper-free → COX2 cannot fold correctly → "
            "Complex IV assembly arrests → functional cytochrome c oxidase is severely reduced "
            "(<15% of normal in most patients). "
            "Consequence: the heart — which relies on OXPHOS for >90% of its ATP — fails to "
            "generate sufficient energy for contractile function → compensatory hypertrophy "
            "(HCM) → cardiomyopathy → cardiac failure (the leading cause of death). "
            "Isolated COX deficiency: Complexes I, II, III are NORMAL (same fingerprint as "
            "SURF1 but DIFFERENT clinical signature — SURF1 has HCM in only 10%; SCO2 has "
            "HCM in 100% and it is the dominant, life-threatening phenotype)."
        ),
        "hcm_note": (
            "SCO2 HCM is the most severe cardiac manifestation in the entire COX assembly factor "
            "disease spectrum: "
            "SURF1 (9q34.2): HCM ~10% (mild) — NEUROLOGICAL dominant. "
            "SCO2 (22q13.33): HCM 100% — CARDIAC dominant; fatal in first year. "
            "SCO1 (17p13.1): HCM rare; HEPATOPATHY dominant. "
            "COX10 (17p12): HCM rare; TUBULOPATHY + ANAEMIA dominant. "
            "LRPPRC (2p21): HCM absent; LEIGH variant dominant (French-Canadian). "
            "If a COX-deficient patient has severe neonatal HCM: SCO2 is the primary "
            "differential — send SCO2 targeted sequencing before WES/WGS to avoid delay."
        ),
        "kpis": [
            {"label": "HCM (Hypertrophic CMP)",  "value": "100%",                        "color": COLOR},
            {"label": "Encephalomyopathy",        "value": f"{n_encephalo/n*100:.0f}%",   "color": COLOR},
            {"label": "Lactic Acidosis",          "value": f"{n_lactic/n*100:.0f}%",      "color": "#b71c1c"},
            {"label": "Hypotonia",                "value": f"{n_hypotonia/n*100:.0f}%",   "color": COLOR},
            {"label": "Leigh-like MRI",           "value": f"{n_leigh/n*100:.0f}%",       "color": COLOR},
            {"label": "Seizures",                 "value": f"{n_seizures/n*100:.0f}%",    "color": "#6a1b9a"},
            {"label": "Resp. Compromise",         "value": f"{n_resp/n*100:.0f}%",        "color": "#b71c1c"},
            {"label": "Mean COX Activity",        "value": f"{mean_cox}% normal",         "color": COLOR},
            {"label": "Mean Onset",               "value": f"{mean_onset} weeks",          "color": COLOR},
            {"label": "Mean Lactate",             "value": f"{mean_lact} mmol/L",         "color": "#b71c1c"},
            {"label": "Mean HCM Wall",            "value": f"{mean_wall} mm",             "color": COLOR},
            {"label": "Alive (last f/u)",         "value": f"{n_alive/n*100:.0f}%",       "color": "#2e7d32"},
        ],
        "contraindications": [
            {
                "drug":      "VPA / Valproate",
                "severity":  "ABSOLUTE CI — ALL SCO2 / Complex IV disease",
                "mechanism": (
                    "Three independent mechanisms: (a) CoA sequestration → depletes free CoA → "
                    "impairs TCA cycle, worsening failing OXPHOS in cardiomyocytes and neurons; "
                    "(b) POLG1 inhibition → mtDNA depletion → reduces residual MT-CO2 (COX2) "
                    "synthesis, further collapsing COX assembly; (c) Idiosyncratic VPA "
                    "hepatotoxicity → fulminant liver failure (even without SCO2 baseline "
                    "hepatopathy — see SCO1 for hepatopathy-predominant COX disease). "
                    "NEVER use VPA in COX-deficient patients. Use LEV (IV loading, renal excretion)."
                ),
            },
            {
                "drug":      "Metformin",
                "severity":  "ABSOLUTE CI — ALL mitochondrial disease including SCO2",
                "mechanism": (
                    "Metformin inhibits Complex I (NADH:ubiquinone oxidoreductase). In SCO2 "
                    "(COX/Complex IV deficient), upstream electron carriers are already backed up. "
                    "Adding Complex I inhibition collapses the entire respiratory chain flux → "
                    "lactic crisis (L:P ratio >30). CRITICAL in SCO2: cardiac muscle runs on "
                    "OXPHOS almost exclusively — lactic crisis in cardiomyocytes → acute cardiac "
                    "failure superimposed on baseline HCM → potentially fatal within hours. "
                    "Absolute prohibition even when family members with diabetes may be on it."
                ),
            },
            {
                "drug":      "Propofol",
                "severity":  "ABSOLUTE CI — PRIS risk amplified by SCO2 cardiac phenotype",
                "mechanism": (
                    "Propofol impairs Complex IV (COX) by direct inhibition of the catalytic "
                    "subunit and uncoupling of the inner mitochondrial membrane. In SCO2 with "
                    "pre-existing severe COX deficiency AND HCM: propofol causes: (a) near-complete "
                    "COX abolition in cardiomyocytes → ATP collapse → acute systolic dysfunction "
                    "superimposed on HCM → cardiac arrest; (b) metabolic acidosis; "
                    "(c) rhabdomyolysis (PRIS triad). This is more catastrophic than in SURF1 "
                    "(neurological only) because the heart is directly and severely affected. "
                    "STRICTLY AVOID propofol for ANY indication — induction, maintenance, or "
                    "sedation. Use sevoflurane inhalational or dexmedetomidine."
                ),
            },
            {
                "drug":      "Linezolid",
                "severity":  "ABSOLUTE CI — ALL mitochondrial disease",
                "mechanism": (
                    "Linezolid inhibits the 23S rRNA peptidyl-transferase centre of the "
                    "mitochondrial ribosome. This blocks synthesis of all 13 mtDNA-encoded "
                    "proteins — critically MT-CO2 (COX2), the DIRECT substrate of SCO2-mediated "
                    "copper delivery. In SCO2: any residual functional COX2 depends on remaining "
                    "MT-CO2 synthesis capacity; linezolid eliminates this → complete COX "
                    "abolition. Alternative antibiotics: daptomycin, beta-lactams, carbapenems."
                ),
            },
            {
                "drug":      "Ketogenic Diet (KD)",
                "severity":  "CONTRAINDICATED — SCO2 / Complex IV deficiency",
                "mechanism": (
                    "Beta-oxidation of fatty acids (the primary KD fuel) generates FADH2. "
                    "Complete FADH2 re-oxidation requires the entire respiratory chain culminating "
                    "in Complex IV (COX) as the terminal electron acceptor (O2). With SCO2-mediated "
                    "COX severely deficient: fatty acid oxidation backs up → energy crisis worsened. "
                    "In cardiac muscle (the dominant target in SCO2): heart relies on fatty acid "
                    "beta-oxidation for ~70% of baseline ATP under KD conditions → KD is "
                    "particularly dangerous in SCO2 cardiac disease. KD has no role in OXPHOS defects."
                ),
            },
            {
                "drug":      "Digoxin",
                "severity":  "AVOID — worsens HCM outflow obstruction",
                "mechanism": (
                    "Digoxin is a positive inotrope via Na+/K+-ATPase inhibition → increased "
                    "intracellular Ca2+ → increased contractility. In hypertrophic cardiomyopathy: "
                    "increased contractility → worsens dynamic LVOT obstruction (if present) → "
                    "increases systolic gradient → cardiac output paradoxically decreases. "
                    "Additionally, in mitochondrial disease: digoxin's Na+/K+-ATPase inhibition "
                    "increases energy demand on an already-failing energy system. "
                    "Use diuretics (furosemide, spironolactone) for symptoms; "
                    "avoid positive inotropes unless obstructive component excluded by echo."
                ),
            },
            {
                "drug":      "Phenobarbital",
                "severity":  "HIGH CAUTION — Complex I inhibitor",
                "mechanism": (
                    "Phenobarbital inhibits Complex I → upstream electron carrier backup → "
                    "worsens lactic acidosis. In SCO2 with HCM: lactic acidosis → acidaemia → "
                    "myocardial depression (acidaemia reduces Ca2+ sensitivity of myofilaments) → "
                    "cardiac decompensation. Use ONLY if LEV is unavailable and seizures are "
                    "life-threatening. Monitor pH and lactate closely. Never first-line."
                ),
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    """SCO2 patient cohort table + clinical feature frequencies."""
    rng = _rng()
    cohort = _build_cohort(rng)

    n = len(cohort)

    def pct(feat: str) -> int:
        return round(sum(1 for p in cohort if feat in p["features"]) / n * 100)

    feature_frequencies = {
        "Hypertrophic Cardiomyopathy HCM (100% — CARDINAL, UNIQUE)":       100,
        "Encephalomyopathy":                                                  pct("Encephalo"),
        "Lactic Acidosis (L:P ratio >20)":                                   pct("Lactic"),
        "Hypotonia":                                                          pct("Hypotonia"),
        "Failure to Thrive":                                                  pct("Failure"),
        "Respiratory Compromise":                                             pct("Resp."),
        "Leigh-like MRI (symmetric BG T2 hyperintensities)":                 pct("Leigh"),
        "Seizures":                                                           pct("Seizures"),
        "Hypertonia (some patients)":                                         pct("Hypertonia"),
        "Optic Atrophy (rare)":                                               pct("Optic"),
        "COX Isolated Deficiency (<20% normal — universal)":                  100,
        "NO Hepatopathy (KEY DDx from SCO1)":                                 96,
        "NO 3-MGA (KEY DDx from SERAC1/CLPB/DNAJC19)":                       98,
        "Alive at last follow-up (most fatal in year 1)":
            round(sum(1 for p in cohort if p["outcome"].startswith("Alive")) / n * 100),
    }

    return {
        "patients": cohort,
        "feature_frequencies": feature_frequencies,
    }


def get_definitions() -> dict[str, Any]:
    """SCO2 clinical, pharmacological and molecular definitions."""
    return {
        "pharmacology": [
            {
                "term": "SCO2 (Synthesis of Cytochrome c Oxidase 2) — Gene and Protein",
                "definition": (
                    "Nuclear-encoded mitochondrial protein; 266 amino acids; chromosome 22q13.33. "
                    "Structure: single transmembrane domain (TM) anchors SCO2 to the IMM with "
                    "the functional thioredoxin-fold domain exposed in the IMS. "
                    "Key domain: CXXXC copper-binding motif (Cys-X-X-X-Cys) — transiently "
                    "chelates one Cu+ ion via the two cysteine thiol groups.\n\n"
                    "Copper trafficking hierarchy:\n"
                    "  COX17 (IMS metallochaperone) → SCO2 (Cu+ chelation via CXXXC)\n"
                    "  SCO2 → SCO1 (Cu2+ reduction → Cu+ by SCO1's oxidoreductase activity)\n"
                    "  SCO1 + SCO2 → CuA binuclear copper centre of COX2 (MT-CO2)\n\n"
                    "SCO2 vs SCO1:\n"
                    "  • SCO2: copper DELIVERY to CuA; thioredoxin-fold structure\n"
                    "  • SCO1: oxidoreductase activity; maintains proper Cu oxidation state\n"
                    "  • Both are REQUIRED for functional CuA — deficiency of either = COX deficiency\n"
                    "  • SCO1 disease: HEPATOPATHY dominant (liver failure neonatal)\n"
                    "  • SCO2 disease: HCM dominant (cardiomyopathy 100%)\n\n"
                    "Without SCO2: CuA centre copper-free → COX2 cannot be assembled → "
                    "Complex IV arrests → ATP collapse in heart (most energy-demanding organ/g)."
                ),
            },
            {
                "term": "CuA Binuclear Copper Centre — Structure, Function, and SCO2 Relevance",
                "definition": (
                    "The CuA centre of cytochrome c oxidase (COX, Complex IV) is a unique "
                    "mixed-valence binuclear copper cluster in COX2 (MT-CO2):\n"
                    "  • Two copper atoms bridged by two cysteine ligands; fully delocalised "
                    "Cu(1.5+)–Cu(1.5+) valence (Robin-Day Class III mixed valence)\n"
                    "  • Additional ligands: 2× histidine, 1× methionine, 1× carbonyl backbone\n"
                    "  • Located at the C-terminal domain of COX2, protruding into the IMS\n\n"
                    "Function: PRIMARY electron acceptor from cytochrome c (reduced):\n"
                    "  Cyt c (Fe2+) → [CuA reduces Cyt c] → Cyt c (Fe3+)\n"
                    "  Electron is then passed internally: CuA → heme a → heme a3/CuB (active site)\n"
                    "  O2 reduction: O2 + 4H+ + 4e- → 2H2O (coupled to 4H+ pumped across IMM)\n\n"
                    "Without CuA (SCO2 deficiency): electrons from cytochrome c cannot enter "
                    "Complex IV → respiratory chain completely blocked at COX → entire upstream "
                    "chain (ubiquinol, Complex III, cytochrome c) accumulates reduced → "
                    "OXPHOS abolished → ATP synthesis fails → cardiac energy crisis."
                ),
            },
        ],
        "gene_concepts": [
            {
                "term": "SCO2 Genotype–Phenotype Correlation",
                "definition": (
                    "SCO2 disease shows relatively narrow phenotypic range (neonatal HCM + "
                    "encephalomyopathy in most), but severity correlates with allele type:\n\n"
                    "p.Glu140Lys (c.418G>A) — Greek/Italian/Mediterranean founder (~60% alleles):\n"
                    "  • CXXXC-adjacent residue; critical for copper chelation geometry\n"
                    "  • Severe LOF: COX residual typically <10% in fibroblasts\n"
                    "  • Homozygous: severe HCM + encephalomyopathy; fatal first year\n"
                    "  • The most common mutation worldwide; enriched in Mediterranean populations\n\n"
                    "p.Leu151Pro (c.452T>C) — second most common:\n"
                    "  • Disrupts alpha-helix near CXXXC; structural LOF\n"
                    "  • Severe; indistinguishable clinically from p.Glu140Lys compound heterozygotes\n\n"
                    "NULL alleles (nonsense/frameshift):\n"
                    "  • Always compound heterozygous (biallelic null is presumed lethal embryonic)\n"
                    "  • Null + missense: similar severity to Glu140Lys homozygous\n\n"
                    "GENOTYPE RULE: ALL biallelic SCO2 LOF → HCM 100%.\n"
                    "The HCM does NOT correlate with residual COX (any COX level below ~30% "
                    "causes HCM in heart). Encephalomyopathy severity correlates weakly with "
                    "COX residual in non-cardiac tissues."
                ),
            },
            {
                "term": "SCO2 vs Other COX Assembly Factor Diseases — Critical DDx",
                "definition": (
                    "All COX assembly factor diseases share isolated COX deficiency but differ "
                    "markedly by organ involvement:\n\n"
                    "SCO2 (22q13.33): HCM 100% CARDINAL + encephalomyopathy; FATAL YEAR 1; "
                    "pGlu140Lys Greek/Italian founder. If COX deficient neonate with HCM → SCO2 FIRST.\n\n"
                    "SURF1 (9q34.2): Leigh syndrome dominant; HCM only ~10%; NO hepatopathy; "
                    "onset 3mo–2yr; most common single-gene Leigh cause; "
                    "European founder c.845_846delCT.\n\n"
                    "SCO1 (17p13.1): HEPATOPATHY 100% (neonatal liver failure) + encephalomyopathy; "
                    "NO cardiomyopathy; fatal neonatal/infantile hepatic crisis.\n\n"
                    "COX10 (17p12): TUBULOPATHY (renal Fanconi) + Leigh + haemolytic ANAEMIA; "
                    "HCM rare; chronic course; different from neonatal SCO2.\n\n"
                    "COX15 (10q24.2): Leigh + CARDIOMYOPATHY (dilated or hypertrophic) but rarer "
                    "than SCO2 and less severe cardiac presentation; variable.\n\n"
                    "LRPPRC (2p21): French-Canadian Leigh variant; COX ~20%; NO HCM; "
                    "Saguenay-Lac-Saint-Jean founder.\n\n"
                    "ALGORITHM: COX deficient neonate + HCM → SCO2 (then COX15). "
                    "COX deficient neonate + hepatic failure → SCO1. "
                    "COX deficient infant + Leigh + NO HCM + NO liver → SURF1. "
                    "COX deficient + tubulopathy → COX10."
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "HCM in SCO2 — Echocardiographic Features, Natural History, Management",
                "definition": (
                    "SCO2 HCM is a primary energy-deficiency cardiomyopathy:\n\n"
                    "ECHOCARDIOGRAPHIC FEATURES:\n"
                    "  • Marked concentric or asymmetric septal hypertrophy (wall thickness >15mm or >2SD)\n"
                    "  • Hyperdynamic systolic function early → EF may be supranormal initially\n"
                    "  • LVOT obstruction: dynamic, variable; use Doppler gradient\n"
                    "  • Progression: systolic dysfunction develops as cardiomyocytes fail\n"
                    "  • End-stage: dilated CMP with reduced EF (burned-out HCM)\n"
                    "  • Pericardial effusion in ~30%\n\n"
                    "NATURAL HISTORY:\n"
                    "  • Most SCO2 patients die in first 6–18 months from cardiac failure\n"
                    "  • Rare survival to childhood (outlier missense combinations)\n"
                    "  • Neurological deterioration parallels but is secondary to cardiac disease\n\n"
                    "MANAGEMENT (supportive):\n"
                    "  • Diuretics: furosemide + spironolactone for pulmonary oedema\n"
                    "  • Beta-blockers (carvedilol/metoprolol): CAUTIOUSLY for obstructive HCM;\n"
                    "    risk: bradycardia + reduced CO in failing heart — titrate carefully\n"
                    "  • ACE inhibitors / ARBs: AVOID in obstructive HCM (reduce afterload → "
                    "worsen LVOT obstruction); may be used in dilated end-stage\n"
                    "  • Cardiac transplantation: discussed but controversial;\n"
                    "    does NOT cure encephalomyopathy — neurological disease continues post-transplant\n"
                    "  • IMPLANTABLE DEFIBRILLATOR (ICD): consider for survivors at risk of VT/VF\n"
                    "  • GENETIC COUNSELLING: 25% recurrence; PGT-M available"
                ),
            },
            {
                "term": "Neonatal COX Deficiency — Emergency Protocol and SCO2 Recognition",
                "definition": (
                    "SCO2 presents in the neonatal period — often before genetic results are available. "
                    "Recognising COX deficiency neonatally is critical:\n\n"
                    "PRESENTING RED FLAGS in SCO2:\n"
                    "  1. Neonatal HCM on echo (even in absence of neurological signs initially)\n"
                    "  2. Elevated lactate (>3.5 mmol/L) in first days of life\n"
                    "  3. Poor feeding, failure to thrive, hypotonia\n"
                    "  4. Respiratory distress not explained by primary pulmonary cause\n\n"
                    "EMERGENCY WORKUP:\n"
                    "  • Blood: lactate, lactate:pyruvate ratio, blood gas (pH, HCO3), glucose\n"
                    "  • Urine: organic acids (elevated lactate/pyruvate; NO 3-MGA in SCO2)\n"
                    "  • Echo: HCM assessment, LVOT gradient, EF\n"
                    "  • Muscle biopsy (if stable): COX histochemistry + respiratory chain enzyme assay\n"
                    "  • Fibroblast culture: COX activity assay (gold standard for functional confirmation)\n"
                    "  • Genetics: targeted SCO2 sequencing (c.418G>A / p.Glu140Lys) → result in days;\n"
                    "    WES/WGS if targeted negative\n\n"
                    "MANAGEMENT WHILE AWAITING GENETICS:\n"
                    "  • IV dextrose GIR 6–8 mg/kg/min (never fast; avoid ketosis)\n"
                    "  • CoQ10 empirically (Level C; no harm; while confirming diagnosis)\n"
                    "  • Cardiac: diuretics for HF; avoid digoxin; avoid propofol for any procedure\n"
                    "  • Antibiotic choice: beta-lactam/carbapenem; NEVER linezolid\n"
                    "  • AED if seizures: IV LEV loading (20 mg/kg); NEVER VPA"
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "Cardiac Anaesthesia Protocol in SCO2 — Propofol Absolute Prohibition",
                "definition": (
                    "SCO2 patients require frequent cardiac procedures (echo, catheter, ICD). "
                    "Anaesthetic management requires cardiologist + metabolic specialist co-management:\n\n"
                    "PROPOFOL — ABSOLUTE CONTRAINDICATION:\n"
                    "  • Propofol directly inhibits Complex IV (COX) via multiple mechanisms:\n"
                    "    — Uncouples IMM proton gradient\n"
                    "    — Inhibits electron transfer at COX catalytic site\n"
                    "    — Inhibits Complex I (secondary)\n"
                    "  • In SCO2 (COX already <15%): propofol abolishes residual COX → acute "
                    "ATP collapse in cardiomyocytes → cardiac arrest during procedure\n"
                    "  • This is MORE dangerous than in SURF1 (neurological-dominant) because "
                    "the heart is the primary target organ in SCO2\n"
                    "  • PRIS (Propofol Infusion Syndrome) in mito disease can develop with "
                    "EVEN SHORT INDUCTION doses, not just prolonged infusions\n\n"
                    "SAFE ANAESTHETIC ALTERNATIVES:\n"
                    "  • Sevoflurane (inhalational induction/maintenance) — preferred\n"
                    "  • Dexmedetomidine (alpha-2 agonist sedation) — for non-intubated procedures\n"
                    "  • Ketamine (short-duration sedation) — careful in HCM (tachycardia risk)\n"
                    "  • Midazolam (premedication, short procedures) — safe mito profile\n\n"
                    "PERIOPERATIVE PROTOCOL:\n"
                    "  • IV dextrose GIR 6–8 throughout fasting (never nil-by-mouth without IV glucose)\n"
                    "  • Cardiac monitoring: arterial line, continuous echo availability\n"
                    "  • Lactate monitoring every 1–2h intraoperatively\n"
                    "  • Paediatric cardiac surgery unit access (in case of decompensation)\n"
                    "  • NO intraoperative VPA (even for seizure emergencies — use IV LEV/benzodiazepine)"
                ),
            },
            {
                "term": "Levetiracetam (LEV) in SCO2 — Only Safe AED in Cardiac COX Disease",
                "definition": (
                    "LEV is the ONLY first-line AED with full safety profile in SCO2:\n"
                    "  • Mechanism: SV2A synaptic vesicle modulator — no direct mitochondrial activity\n"
                    "  • Elimination: renal excretion (66% unchanged) — no hepatic metabolism\n"
                    "  • No CYP induction — no interactions with cardiac medications\n"
                    "  • IV formulation (Keppra IV): loading 20–40 mg/kg over 15 min;\n"
                    "    maintenance 10–20 mg/kg q12h\n"
                    "  • No Complex I/II/III/IV inhibition — critical in SCO2 cardiac muscle\n\n"
                    "ESPECIALLY IMPORTANT IN SCO2 vs other mito diseases:\n"
                    "  • Seizures in SCO2 may trigger cardiac arrhythmia in HCM setting\n"
                    "  • Rapid IV LEV loading avoids: (a) VPA toxicity, (b) phenobarbital's "
                    "Complex I inhibition (worsens lactic acidosis → myocardial acidaemia), "
                    "(c) phenytoin's proarrhythmic risk in HCM\n"
                    "  • For status epilepticus in SCO2: IV LEV → IV benzodiazepine (low dose) → "
                    "NEVER VPA, NEVER phenobarbital as first-line"
                ),
            },
            {
                "term": "Genetic Counselling and Reproductive Options for SCO2 Families",
                "definition": (
                    "SCO2 is autosomal recessive — precise genetic counselling is essential:\n\n"
                    "RECURRENCE RISK:\n"
                    "  • Both parents are OBLIGATE heterozygous carriers\n"
                    "  • 25% (1-in-4) recurrence risk per pregnancy\n"
                    "  • Carrier parents: clinically asymptomatic (one functional SCO2 allele sufficient)\n\n"
                    "REPRODUCTIVE OPTIONS:\n"
                    "  • Prenatal diagnosis: chorionic villus sampling (CVS) at 10–12 weeks or\n"
                    "    amniocentesis at 15–18 weeks — targeted SCO2 sequencing\n"
                    "  • Preimplantation Genetic Testing for Monogenic disease (PGT-M):\n"
                    "    — IVF + embryo biopsy → select unaffected embryos (50% unaffected, 25% carrier)\n"
                    "    — Preferred if family cannot accept pregnancy termination\n"
                    "    — Requires family-specific probe design (2–3 months lead time)\n\n"
                    "MOLECULAR CONFIRMATION MANDATORY:\n"
                    "  • Identify BOTH alleles in the proband before offering PGT-M or prenatal Dx\n"
                    "  • If index case died without molecular confirmation: sib, parent testing with "
                    "targeted panel (SCO2, SCO1, SURF1, COX10, COX15) from stored DNA/fibroblasts\n\n"
                    "CARRIER TESTING:\n"
                    "  • Extended family members (proband's aunts/uncles/cousins) may be carriers\n"
                    "    — Offer cascade testing for reproductive-age relatives\n"
                    "  • Mediterranean ancestry: p.Glu140Lys carrier frequency ~1:150–200\n"
                    "    in Greek/Italian populations — consider carrier screening in index communities"
                ),
            },
        ],
    }
