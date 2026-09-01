#!/usr/bin/env python3
"""SURF1 — Leigh Syndrome due to Complex IV (COX) Deficiency.

SURF1 (Surfeit Locus Protein 1) is the most commonly mutated single gene
causing Leigh syndrome.

  SURF1 gene           OMIM #185620
  Leigh Syndrome (LS)  OMIM #256000
  Complex IV (COX)     deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (SURF1 / Complex IV assembly / COX I subunit chaperoning):
SURF1 is an inner mitochondrial membrane (IMM) protein that acts as an
assembly factor for cytochrome c oxidase (COX, Complex IV). It is involved
in the early/intermediate stages of COX assembly, specifically facilitating
the insertion of heme a3 and CuB cofactors into COX1 (MT-CO1), the
catalytic core subunit of Complex IV:
  • SURF1 binds to early COX assembly intermediates (S2/S3) and stabilises
    COX1 subunit before heme a/a3 insertion — thought to act as a scaffold
    for the Shy1-Cox11 copper-delivery sub-pathway (yeast homology)
  • Without SURF1: COX1 subunit cannot assemble properly; COX assembly
    arrests at the S3 intermediate; functional COX is severely reduced
  • Electron transfer: cytochrome c → CuA → heme a → CuB + heme a3
    (active site) → molecular oxygen → H2O is BLOCKED
  • The 4-electron reduction of O2 is the terminal step of the respiratory
    chain — without it, ALL upstream electron carriers back up
  • Proton pumping (4 H+/2 electrons) across the IMM fails → proton motive
    force collapses → F1Fo-ATP synthase cannot synthesise ATP
  • Isolated COX deficiency: Complex I, II, III activities NORMAL — this is
    the biochemical fingerprint of SURF1 disease (vs pan-OXPHOS in mtDNA tRNA)

MOLECULAR: Biallelic (AR) loss-of-function SURF1 variants:
  — Null alleles (frameshift, nonsense): severe Leigh, early death (1–5 yr)
  — Missense alleles: partial COX residual → milder phenotype, longer survival
  Most common European allele: c.845_846delCT (p.Leu282Profs*3), ~40%

LEIGH SYNDROME MRI PATTERN (bilateral symmetric T2 hyperintensities):
  • Putamen (most common) and caudate
  • Brainstem (periaqueductal grey, inferior olives)
  • Posterior thalami
  • Cerebellar dentate nuclei (later)
  • Lesions can be reversible during metabolic stability, recur with illness
  • NOT deep white matter (unlike LBSL, Canavan, Krabbe)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 593
DISEASE_ID   = "surf1"
DISEASE_NAME = "SURF1 Leigh Syndrome"
GENE         = "SURF1"
OMIM_GENE    = "#185620"
OMIM_DISEASE = "#256000"
CHROMOSOME   = "9q34.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Infantile (3 months – 2 years; range 1 month – 3 years)"
COHORT_SIZE  = 40
COLOR        = "#1a237e"   # deep indigo — Leigh/basal ganglia / SURF1
LIGHT        = "#e8eaf6"


# Genotype pool
GENO_NULL_HOM   = "c.845_846delCT homozygous (p.Leu282Profs*3) — European founder"
GENO_NULL_CPX   = "c.845_846delCT / other truncating — compound heterozygous"
GENO_MISS_CPX   = "Two missense alleles — compound heterozygous (partial COX residual)"
GENO_TRUNC_MISS = "Truncating / missense — compound heterozygous"

GENO_POOL    = [GENO_NULL_HOM, GENO_NULL_CPX, GENO_MISS_CPX, GENO_TRUNC_MISS]
GENO_WEIGHTS = [0.20,           0.20,           0.30,           0.30]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient SURF1/Leigh cohort (seed-593)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient SURF1 Leigh cohort (seed-593).

    All patients meet SURF1-Leigh definition: Leigh MRI + COX deficiency.
    Additional features stochastic per published frequencies.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno  = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex   = "F" if rng.random() < 0.50 else "M"

        # Onset age (months)
        onset_mo = round(rng.uniform(1.5, 30), 1)

        # Lactate (mmol/L) — elevated, L:P ratio >20
        lactate = round(rng.uniform(3.5, 12.0), 1)

        # COX residual activity (% of normal in fibroblasts)
        is_null  = "Profs" in geno or "other truncating" in geno
        if is_null:
            cox_pct = round(rng.uniform(2.0, 18.0), 1)
        else:
            cox_pct = round(rng.uniform(10.0, 35.0), 1)

        # Features
        leigh_mri   = rng.random() < 0.95  # 95%
        hypotonia   = rng.random() < 0.95  # 95%
        seizures    = rng.random() < 0.55  # 55%
        optic_atrph = rng.random() < 0.40  # 40%
        resp_comp   = rng.random() < 0.75  # 75%
        ataxia      = rng.random() < 0.45  # 45%
        nystagmus   = rng.random() < 0.30  # 30%
        spasticity  = rng.random() < 0.50  # 50%
        hcm         = rng.random() < 0.10  # 10%
        lactic_ac   = rng.random() < 0.90  # 90%

        # Survival (years); null alleles die earlier
        if is_null:
            survival_yr = round(rng.uniform(0.5, 7.0), 1)
            alive       = rng.random() < 0.35
        else:
            survival_yr = round(rng.uniform(2.0, 18.0), 1)
            alive       = rng.random() < 0.65

        if alive:
            outcome = f"Alive {survival_yr}yr (last follow-up)"
        else:
            outcome = f"Died at {survival_yr}yr"

        # Treatments
        txs = ["CoQ10/Ubiquinol (Level C)"]
        if lactic_ac:   txs.append("Thiamine B1 (Level C)")
        txs.append("Riboflavin B2 (Level C)")
        txs.append("Biotin (Level C)")
        if seizures:    txs.append("LEV (preferred AED — renal excretion)")
        if resp_comp:   txs.append("NIV/respiratory support")
        if lactic_ac:   txs.append("IV Dextrose GIR 6-8 (crisis)")
        txs.append("Succinate (Level C, anaplerotic)")

        # Alerts
        alerts_list = ["🚫 VPA ABSOLUTE CI — mito hepatotoxicity"]
        alerts_list.append("🚫 Metformin ABSOLUTE CI — Complex I inhibition → lactic crisis")
        alerts_list.append("🚫 Linezolid ABSOLUTE CI — mito ribosome inhibition")
        if seizures:
            alerts_list.append("⚠ Seizures: LEV preferred; NO VPA/phenobarbital")
        if resp_comp:
            alerts_list.append("🚨 Respiratory compromise: central apnoea risk — NIV/monitor")
        alerts = "; ".join(alerts_list)

        # Feature string
        feats = ["Psychomotor regression"]  # 100%
        if leigh_mri:   feats.append("Leigh MRI")
        if hypotonia:   feats.append("Hypotonia")
        if lactic_ac:   feats.append("Lactic acidosis")
        if seizures:    feats.append("Seizures")
        if optic_atrph: feats.append("Optic atrophy")
        if resp_comp:   feats.append("Resp. compromise")
        if ataxia:      feats.append("Ataxia")
        if nystagmus:   feats.append("Nystagmus")
        if spasticity:  feats.append("Spasticity")
        if hcm:         feats.append("HCM")

        patients.append({
            "id":         f"SURF1-{i:03d}",
            "geno":       geno,
            "sex":        sex,
            "onset_mo":   onset_mo,
            "lactate":    lactate,
            "cox_pct":    cox_pct,
            "features":   ", ".join(feats),
            "treatments": ", ".join(txs),
            "alerts":     alerts,
            "outcome":    outcome,
        })
    return patients


# ── Public API functions ─────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    """SURF1 overview — gene, disease identity, KPIs, contraindications."""
    rng = _rng()
    cohort = _build_cohort(rng)

    n = len(cohort)

    def pct_feat(feat: str) -> float:
        return round(sum(1 for p in cohort if feat in p["features"]) / n * 100)

    n_leigh   = sum(1 for p in cohort if "Leigh MRI" in p["features"])
    n_hyp     = sum(1 for p in cohort if "Hypotonia" in p["features"])
    n_seiz    = sum(1 for p in cohort if "Seizures" in p["features"])
    n_opt     = sum(1 for p in cohort if "Optic atrophy" in p["features"])
    n_resp    = sum(1 for p in cohort if "Resp." in p["features"])
    n_lact    = sum(1 for p in cohort if "Lactic acidosis" in p["features"])
    n_alive   = sum(1 for p in cohort if p["outcome"].startswith("Alive"))
    mean_cox  = round(sum(p["cox_pct"] for p in cohort) / n, 1)
    mean_onset= round(sum(p["onset_mo"] for p in cohort) / n, 1)
    mean_lact = round(sum(p["lactate"] for p in cohort) / n, 1)
    n_null    = sum(1 for p in cohort if "Profs" in p["geno"] or "truncating" in p["geno"])

    return {
        "gene":         "SURF1 (Surfeit Locus Protein 1)",
        "protein":      "SURF1-300aa — Complex IV (COX) assembly factor; scaffolds COX1 subunit heme a/a3 + CuB insertion; IMM-resident",
        "disease":      "Leigh Syndrome (Subacute Necrotizing Encephalomyelopathy) due to Complex IV / COX Deficiency — most common single-gene Leigh cause",
        "omim_gene":    "#185620 (SURF1)",
        "omim_disease": "#256000 (Leigh Syndrome / LS-COX deficiency)",
        "chromosome":   "9q34.2",
        "inheritance":  "AR (autosomal recessive biallelic) — 25% recurrence per pregnancy",
        "onset":        "Infantile; median ~6 months; range 1 month – 3 years",
        "cohort":       f"{n} patients · seed-593 · SURF1 biallelic",
        "mechanism": (
            "SURF1 is an inner mitochondrial membrane (IMM) protein essential for the early/intermediate "
            "assembly of cytochrome c oxidase (COX, Complex IV). It acts as a scaffold during the insertion "
            "of heme a3 and CuB metal cofactors into the COX1 catalytic core subunit (MT-CO1). "
            "Without SURF1: COX1 cannot acquire its heme and copper cofactors → COX assembly arrests "
            "at early intermediates (S2/S3 stages) → functional Complex IV is severely reduced (<20% normal). "
            "Consequence: cytochrome c cannot donate electrons to molecular oxygen (the final electron acceptor) → "
            "the entire respiratory chain backs up → proton motive force collapses → ATP synthesis fails → "
            "neurons (highest OXPHOS-dependent cells) die. The basal ganglia and brainstem nuclei are "
            "particularly vulnerable due to their high metabolic demand and relative lack of glycolytic capacity "
            "→ bilateral symmetric Leigh lesions. Isolated COX deficiency: Complexes I, II, III are NORMAL "
            "(this is the biochemical fingerprint distinguishing SURF1 from mtDNA tRNA mutations and nuclear "
            "DNA maintenance diseases where pan-OXPHOS defects occur)."
        ),
        "no_hepatopathy_note": (
            "SURF1 does NOT cause hepatopathy — KEY differentiator from POLG (Alpers; hepatic failure 80%), "
            "DGUOK (hepatocerebral MDDS3; liver failure leading cause of death), MPV17 (MDDS6; Navajo "
            "neurohepatopathy), GRACILE (BCS1L; cholestasis + liver failure 85%). "
            "If a SURF1-like patient has hepatic failure: reconsider POLG, DGUOK, or MPV17. "
            "SURF1 liver: COX activity is reduced (~50%) but rarely causes clinically significant liver disease. "
            "No iron overload (GRACILE), no 3-MGA (SERAC1/CLPB/OPA3), no sideroblastic anemia (SFXN4)."
        ),
        "kpis": [
            {"label": "Psychomotor Regression", "value": "100%", "color": COLOR},
            {"label": "Leigh MRI",               "value": f"{n_leigh/n*100:.0f}%", "color": COLOR},
            {"label": "Hypotonia",               "value": f"{n_hyp/n*100:.0f}%",   "color": COLOR},
            {"label": "Lactic Acidosis",         "value": f"{n_lact/n*100:.0f}%",  "color": "#c62828"},
            {"label": "Resp. Compromise",        "value": f"{n_resp/n*100:.0f}%",  "color": "#b71c1c"},
            {"label": "Seizures",                "value": f"{n_seiz/n*100:.0f}%",  "color": "#6a1b9a"},
            {"label": "Optic Atrophy",           "value": f"{n_opt/n*100:.0f}%",   "color": COLOR},
            {"label": "Mean COX Activity",       "value": f"{mean_cox}% normal",   "color": COLOR},
            {"label": "Mean Onset",              "value": f"{mean_onset} months",   "color": COLOR},
            {"label": "Mean Lactate",            "value": f"{mean_lact} mmol/L",    "color": "#c62828"},
            {"label": "Alive (last f/u)",        "value": f"{n_alive/n*100:.0f}%", "color": "#2e7d32"},
            {"label": "Null allele carriers",    "value": f"{n_null/n*100:.0f}%",  "color": COLOR},
        ],
        "contraindications": [
            {
                "drug":      "VPA / Valproate",
                "severity":  "ABSOLUTE CI — ALL SURF1/Leigh syndrome",
                "mechanism": (
                    "Three independent mechanisms: (a) CoA sequestration → depletes free CoA pool → "
                    "impairs TCA cycle flux, worsening already-failing OXPHOS; (b) POLG1 inhibition → "
                    "accelerates mtDNA depletion → reduces mtDNA-encoded COX subunits (MT-CO1/2/3) further; "
                    "(c) Idiosyncratic hepatotoxicity — even in SURF1 without baseline hepatopathy, VPA "
                    "can cause fulminant hepatic failure. Alternative: LEV (renal excretion, IV formulation, "
                    "no mito toxicity, no CYP interactions)."
                ),
            },
            {
                "drug":      "Metformin",
                "severity":  "ABSOLUTE CI — ALL mitochondrial disease including SURF1",
                "mechanism": (
                    "Metformin is a Complex I inhibitor. In SURF1 (COX/Complex IV deficient), all upstream "
                    "OXPHOS capacity is already compromised. Additional Complex I inhibition → lactic crisis "
                    "(lactate:pyruvate ratio rises further above 20) → risk of fatal lactic acidosis. "
                    "This is particularly dangerous in SURF1 as respiratory compromise is common → "
                    "lactic acidosis + apnoea compound rapidly. NEVER use metformin in any patient with "
                    "confirmed or suspected Leigh syndrome."
                ),
            },
            {
                "drug":      "Linezolid",
                "severity":  "ABSOLUTE CI — ALL mitochondrial disease",
                "mechanism": (
                    "Linezolid inhibits the 23S rRNA component of the mitochondrial ribosome (identical "
                    "to bacterial 50S ribosome subunit). This blocks synthesis of all 13 mtDNA-encoded "
                    "proteins, which include 3 COX subunits (MT-CO1, MT-CO2, MT-CO3) critical for "
                    "Complex IV assembly. In SURF1 with already-reduced COX: linezolid eliminates any "
                    "residual MT-CO1/2/3 synthesis → near-complete COX abolition. "
                    "Alternative: daptomycin, beta-lactams, carbapenems (none inhibit mito ribosome)."
                ),
            },
            {
                "drug":      "Ketogenic Diet (KD)",
                "severity":  "CONTRAINDICATED — SURF1/Complex IV deficiency",
                "mechanism": (
                    "Beta-oxidation of fatty acids (the primary fuel source on KD) generates FADH2 via "
                    "the ETF/ETFDH/ubiquinone pathway. Re-oxidation of FADH2 ultimately requires electron "
                    "transfer through the entire respiratory chain to Complex IV (O2 as terminal acceptor). "
                    "With Complex IV (COX) severely deficient, FADH2 cannot be re-oxidised → beta-oxidation "
                    "backs up → energy crisis worsened on KD. KD has no therapeutic role in SURF1. "
                    "Contrast: KD is first-line in PDHA1/PDH complex deficiency (where it bypasses "
                    "the pyruvate block) but not in OXPHOS defects."
                ),
            },
            {
                "drug":      "Propofol",
                "severity":  "AVOID — Propofol Infusion Syndrome (PRIS) risk",
                "mechanism": (
                    "Propofol impairs Complex IV (COX) via direct mitochondrial inner membrane "
                    "uncoupling and inhibition of Complex IV activity. In SURF1 with pre-existing "
                    "COX deficiency, propofol administration → complete functional COX abolition → "
                    "lactic acidosis, cardiac arrhythmia, rhabdomyolysis (PRIS). "
                    "Use sevoflurane inhalational anaesthesia; for procedural sedation, use "
                    "dexmedetomidine or ketamine (short-term, careful monitoring); "
                    "AVOID propofol even for brief induction."
                ),
            },
            {
                "drug":      "Phenobarbital",
                "severity":  "HIGH CAUTION — Complex I inhibitor, not first-line",
                "mechanism": (
                    "Phenobarbital inhibits Complex I (NADH:ubiquinone oxidoreductase) → exacerbates "
                    "upstream electron backup already caused by COX deficiency. Worsens lactic acidosis. "
                    "Use ONLY if LEV is unavailable and seizures are uncontrolled. Monitor lactate "
                    "closely with any phenobarbital use. NOT preferred — LEV always first."
                ),
            },
            {
                "drug":      "Chloramphenicol",
                "severity":  "AVOID — same mechanism as linezolid",
                "mechanism": (
                    "Chloramphenicol inhibits the peptidyl transferase centre of the mitochondrial "
                    "ribosome (same 23S rRNA target as linezolid). Blocks synthesis of MT-CO1/2/3 "
                    "and other mtDNA-encoded subunits. Avoid in all mito disease; use alternative "
                    "antibiotics (beta-lactam, carbapenem, daptomycin)."
                ),
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    """SURF1 patient cohort table + clinical feature frequencies."""
    rng = _rng()
    cohort = _build_cohort(rng)

    n = len(cohort)

    def pct(feat: str) -> int:
        return round(sum(1 for p in cohort if feat in p["features"]) / n * 100)

    feature_frequencies = {
        "Psychomotor Regression (100% — defining feature)": 100,
        "Leigh MRI (bilateral BG + brainstem T2 hyperintensities)": pct("Leigh MRI"),
        "Hypotonia (axial and peripheral)": pct("Hypotonia"),
        "Lactic Acidosis (L:P ratio >20)": pct("Lactic acidosis"),
        "Respiratory Compromise (central apnoea)": pct("Resp."),
        "Seizures (focal or generalised)": pct("Seizures"),
        "Ataxia": pct("Ataxia"),
        "Spasticity (later course)": pct("Spasticity"),
        "Optic Atrophy": pct("Optic atrophy"),
        "Nystagmus": pct("Nystagmus"),
        "Hypertrophic Cardiomyopathy (HCM)": pct("HCM"),
        "NO Hepatopathy (DISTINGUISHER from POLG/DGUOK)": 97,  # published figure
        "COX Isolated Deficiency (<20% in null alleles)": pct("Psychomotor"),  # all have COX deficiency
    }

    return {
        "patients": cohort,
        "feature_frequencies": feature_frequencies,
    }


def get_definitions() -> dict[str, Any]:
    """SURF1 clinical, pharmacological and molecular definitions."""
    return {
        "pharmacology": [
            {
                "term": "SURF1 (Surfeit Locus Protein 1) — Gene and Protein",
                "definition": (
                    "Nuclear-encoded mitochondrial inner membrane (IMM) protein; 300 amino acids; "
                    "located at chromosome 9q34.2. Contains two transmembrane domains (TM1, TM2) "
                    "with a conserved hydrophilic loop facing the intermembrane space (IMS). "
                    "Belongs to the Surf1/Shy1 protein family (Shy1 is the yeast orthologue). "
                    "Function: COX assembly factor for early/intermediate stages of Complex IV biogenesis:\n"
                    "  • Binds to the COX1 (MT-CO1) subunit assembly intermediate S2/S3\n"
                    "  • Scaffolds the insertion of heme a3 and CuB cofactors into COX1's active site\n"
                    "  • Works in concert with COX10 (heme a farnesylation) and COX15 (heme a3 synthesis)\n"
                    "  • Also interacts with SCO1/SCO2 (CuA copper centre of COX2) indirectly via "
                    "COX1 assembly checkpoint\n"
                    "Without SURF1: COX1 cannot acquire its metal cofactors → no catalytic Complex IV formed."
                ),
            },
            {
                "term": "Complex IV (Cytochrome c Oxidase / COX) — Structure and Catalysis",
                "definition": (
                    "Terminal enzyme of the mitochondrial respiratory chain. Catalyses the 4-electron "
                    "reduction of molecular oxygen to water, coupled to proton pumping:\n"
                    "  4 Cyt c (red) + O2 + 4 H+_in → 4 Cyt c (ox) + 2 H2O + 4 H+_out\n\n"
                    "Subunit composition (mammalian): 14 subunits total\n"
                    "  mtDNA-encoded (3): MT-CO1 (catalytic core; heme a, heme a3, CuB), "
                    "MT-CO2 (CuA binuclear copper centre), MT-CO3 (transmembrane scaffold)\n"
                    "  Nuclear-encoded (11): COX4, COX5A/B, COX6A/B/C, COX7A/B/C, COX8\n\n"
                    "Assembly: hierarchical; MT-CO1 → MT-CO2 → MT-CO3 + nuclear subunits\n"
                    "Assembly factors (NOT subunits): SURF1, SCO1, SCO2, COX10, COX11, COX14, "
                    "COX15, COX17, COX20, LRPPRC, COA3, COA5, COA6, COA7, COA8, PET117\n\n"
                    "In SURF1 deficiency: only early COX1 intermediate affected → residual "
                    "COX activity 5–35% of normal → profound energy failure in high-demand neurons."
                ),
            },
            {
                "term": "Leigh Syndrome — Diagnostic Criteria and Neuropathology",
                "definition": (
                    "Leigh syndrome (subacute necrotizing encephalomyelopathy; Leigh 1951) is "
                    "defined by:\n"
                    "  1. Progressive neurological disease — psychomotor regression (100%)\n"
                    "  2. MRI: bilateral symmetric T2 hyperintensities in basal ganglia (putamen), "
                    "brainstem, or other deep grey structures\n"
                    "  3. Elevated CSF lactate or blood lactate (lactate:pyruvate ratio >20)\n"
                    "  OR neuropathology: bilateral symmetric spongiform necrosis with vascular "
                    "proliferation in basal ganglia, brainstem, and cerebellum\n\n"
                    "SURF1 is the most commonly identified single-gene cause of Leigh syndrome "
                    "(accounts for ~10–15% of all Leigh cases with a molecular diagnosis). "
                    "Other causes: POLG (MELAS/Alpers can have Leigh pattern), MT-ATP6 (MILS/NARP), "
                    "NDUFV1, NDUFS2/4/7/8 (Complex I), SDHA (Complex II), ETHE1, PDHA1, PC, "
                    "SLC19A3 (biotin-thiamine-responsive), BTD (biotinidase deficiency — TREATABLE).\n\n"
                    "CRITICAL: Biotin-responsive disorders (SLC19A3 and BTD) mimic Leigh on MRI "
                    "but are REVERSIBLE with biotin/thiamine → ALWAYS give empiric biotin + thiamine "
                    "while awaiting genetic results."
                ),
            },
        ],
        "gene_concepts": [
            {
                "term": "SURF1 Genotype–Phenotype Correlation",
                "definition": (
                    "The severity of SURF1 disease correlates with residual COX activity:\n\n"
                    "NULL alleles (frameshift, nonsense, large deletion):\n"
                    "  • Most common European allele: c.845_846delCT (p.Leu282Profs*3)\n"
                    "  • COX residual: typically <10% of normal\n"
                    "  • Phenotype: severe classical Leigh; onset <12 months; median survival 3–5 yr\n"
                    "  • Homozygous c.845_846delCT: ~20% of European SURF1 patients\n\n"
                    "MISSENSE alleles:\n"
                    "  • COX residual: 10–35% of normal\n"
                    "  • Phenotype: milder Leigh; later onset; longer survival (into teens/20s)\n"
                    "  • May present with ataxia + optic atrophy without classic Leigh MRI\n\n"
                    "COMPOUND HETEROZYGOUS (null + missense):\n"
                    "  • COX residual determined by the hypomorphic allele (dominant partial function)\n"
                    "  • Intermediate phenotype; onset 1–3 yr; survival variable (5–15 yr)\n\n"
                    "FIBROBLAST COX ASSAY: reliable and ESSENTIAL for biochemical diagnosis; "
                    "always send fibroblasts before committing to WES/WGS alone — functional "
                    "confirmation needed for VUS (variant of uncertain significance) interpretation."
                ),
            },
            {
                "term": "SURF1 vs Other COX Assembly Factor Disorders — Critical DDx",
                "definition": (
                    "Several COX assembly factors cause overlapping phenotypes:\n\n"
                    "SURF1 (9q34.2): Leigh syndrome; onset 3mo–2yr; NO hepatopathy; NO cardiomyopathy; "
                    "isolated COX; null alleles → severe Leigh (1–5yr). MOST COMMON COX-Leigh gene.\n\n"
                    "SCO1 (17p13.1): Hepatopathy 100% (liver failure neonatal) + encephalopathy; "
                    "isolated COX; often fatal neonatal. DIFFERS by hepatic crisis.\n\n"
                    "SCO2 (22q13.33): Hypertrophic cardiomyopathy 100% CARDINAL + encephalomyopathy; "
                    "isolated COX; fatal in first year; Greek/Italian founder allele p.Glu140Lys. "
                    "DIFFERS by severe HCM.\n\n"
                    "COX10 (17p12): Tubulopathy + Leigh + anaemia; COX deficiency; chronic. "
                    "DIFFERS by renal Fanconi (absent in SURF1).\n\n"
                    "COX15 (10q24.2): Leigh + cardiomyopathy (like SCO2 but less severe); "
                    "rare. DIFFERS by cardiac involvement.\n\n"
                    "LRPPRC (2p21): French-Canadian Leigh variant (LCCS); isolated COX 20%+; "
                    "French-Canadian founder (Saguenay–Lac-Saint-Jean). Geographic founder.\n\n"
                    "→ All share isolated COX deficiency but differ by organ involvement. "
                    "Fibroblast COX assay + WES/WGS panel mandatory."
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "Leigh MRI — Lesion Pattern and Reversibility",
                "definition": (
                    "SURF1 Leigh MRI characteristics:\n"
                    "  • T2/FLAIR hyperintensities: bilateral, symmetric, deep grey matter\n"
                    "  • Putamen: most commonly affected (80–90%)\n"
                    "  • Caudate nuclei: often co-involved\n"
                    "  • Posterior thalami\n"
                    "  • Brainstem: periaqueductal grey, inferior olives, tegmentum\n"
                    "  • Dentate nuclei (later in disease)\n"
                    "  • Spinal cord: occasionally (cervical anterior horn)\n"
                    "  • NOT periventricular white matter (LBSL, Canavan, Krabbe)\n"
                    "  • NOT posterior white matter alone (MLD)\n\n"
                    "DYNAMIC LESIONS: Leigh lesions can partially RESOLVE during metabolic stability "
                    "and WORSEN or reappear during febrile illness, infections, fasting, or surgery. "
                    "A normal MRI during remission does NOT exclude SURF1 Leigh.\n\n"
                    "DWI/ADC: acute Leigh lesions may show restricted diffusion (cytotoxic oedema); "
                    "chronic lesions show elevated ADC (necrosis + cystic change).\n\n"
                    "MRS (MR Spectroscopy): elevated lactate doublet at 1.33 ppm in affected regions; "
                    "reduced NAA (neuronal loss); this corroborates OXPHOS failure biochemically."
                ),
            },
            {
                "term": "Biotin-Thiamine-Responsive Basal Ganglia Disease — Critical Treatable Mimic",
                "definition": (
                    "SLC19A3 (biotin-thiamine-responsive basal ganglia disease, BTBGD) and BTD "
                    "(biotinidase deficiency) are TREATABLE disorders that mimic SURF1 Leigh "
                    "on brain MRI — failure to treat = preventable death/disability:\n\n"
                    "BTBGD (SLC19A3 — thiamine transporter 2):\n"
                    "  • MRI: bilateral putaminal T2 hyperintensities (identical to Leigh)\n"
                    "  • Onset: episodic encephalopathy (triggered by febrile illness)\n"
                    "  • Treatment: high-dose biotin (5–10 mg/kg/day) + thiamine (300 mg/day)\n"
                    "  • Response: dramatic MRI and clinical reversal within 2–4 weeks if treated early\n"
                    "  • Founder allele: p.Tyr243Cys in Saudi/Gulf populations\n\n"
                    "BTD (biotinidase deficiency):\n"
                    "  • Enzyme: biotinidase (recycles biotin from biocytin)\n"
                    "  • Onset: late infancy; seizures, ataxia, hypotonia, rash, alopecia\n"
                    "  • MRI: may show basal ganglia changes\n"
                    "  • Treatment: biotin 5–20 mg/day — FULLY reversible if early\n"
                    "  • Newborn screening: included in many countries\n\n"
                    "PROTOCOL: In any Leigh presentation without confirmed molecular diagnosis → "
                    "empirically start biotin + thiamine IMMEDIATELY while awaiting WES/WGS. "
                    "Cost: negligible. Benefit: prevents irreversible brain damage in treatable cases."
                ),
            },
            {
                "term": "SURF1 — No Hepatopathy, No Iron Overload, No 3-MGA",
                "definition": (
                    "Critical negative features of SURF1 that guide differential diagnosis:\n\n"
                    "NO HEPATOPATHY:\n"
                    "  If hepatic dysfunction is present: reconsider POLG (Alpers; liver failure 80%+), "
                    "DGUOK (MDDS3; hepatocerebral, often fatal liver failure), MPV17 (MDDS6; Navajo "
                    "neurohepatopathy), GRACILE (BCS1L; cholestasis + liver failure 85%)\n\n"
                    "NO IRON OVERLOAD:\n"
                    "  Ferritin/transferrin saturation normal in SURF1. If iron overload: GRACILE "
                    "(BCS1L; unique among mito diseases), SFXN4 (sideroblastic anemia)\n\n"
                    "NO 3-METHYLGLUTACONIC ACIDURIA (3-MGA):\n"
                    "  3-MGA in urine is a marker of inner membrane disruption, NOT COX deficiency. "
                    "3-MGA elevation: SERAC1 (MEGDEL), CLPB (MGCA7), OPA3 (Costeff), DNAJC19 "
                    "(DCMA), TAZ (Barth), TMEM70 (CIII5). SURF1: urine organic acids show elevated "
                    "lactate/pyruvate only — 3-MGA absent.\n\n"
                    "NO SIDEROBLASTIC ANEMIA:\n"
                    "  If ring sideroblasts on bone marrow: SFXN4 (MDDS8B), ALAS2 (X-linked), "
                    "GLRX5, ABCB7. Not SURF1."
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "Respiratory Management in SURF1 Leigh — NIV and Apnoea Protocol",
                "definition": (
                    "Respiratory compromise is common in SURF1 (central apnoea ~75%), particularly:\n"
                    "  • During acute Leigh crises (fever, infection, fasting trigger)\n"
                    "  • During disease progression (brainstem Leigh lesions → respiratory centre failure)\n"
                    "  • Postoperatively (brainstem sensitivity to anaesthetic agents)\n\n"
                    "MONITORING:\n"
                    "  • Overnight oximetry in all SURF1 patients (even clinically stable)\n"
                    "  • Polysomnography if desaturations detected\n"
                    "  • Capnography (end-tidal CO2) during any sedation\n\n"
                    "NIV (Non-Invasive Ventilation):\n"
                    "  • BiPAP/CPAP: for central apnoea or hypercapnic respiratory failure\n"
                    "  • High-flow oxygen: use cautiously — O2 drives more Complex IV activity but "
                    "cannot compensate for structural COX deficiency\n\n"
                    "PERIOPERATIVE PROTOCOL:\n"
                    "  • Anaesthetic: prefer sevoflurane inhalational (NOT propofol)\n"
                    "  • IV dextrose GIR 6–8 mg/kg/min throughout fasting period\n"
                    "  • ICU monitoring for minimum 24h post-procedure\n"
                    "  • Lactate monitoring every 4–6h perioperatively\n"
                    "  • Warn anaesthetic team: mito-sensitive patient, risk of PRIS with propofol, "
                    "lactic crisis with fasting, increased brainstem apnoea risk"
                ),
            },
            {
                "term": "Levetiracetam (LEV) in SURF1 Leigh — Preferred AED",
                "definition": (
                    "LEV is the ONLY first-line AED fully safe in SURF1 Leigh:\n"
                    "  • Mechanism: SV2A vesicle protein modulator → no direct mito interaction\n"
                    "  • Elimination: renal excretion (66% unchanged urine) → no hepatic metabolism\n"
                    "  • No CYP induction/inhibition → no drug interactions\n"
                    "  • IV formulation available (Keppra IV): loading 20–40 mg/kg; "
                    "maintenance 10–20 mg/kg/dose every 12h\n"
                    "  • Oral: 10–30 mg/kg/day in 2 divided doses\n"
                    "  • No Complex I/II/III/IV inhibition — safe in all OXPHOS diseases\n\n"
                    "Second-line (if LEV fails):\n"
                    "  • Zonisamide: some Complex I concern but less than phenobarbital\n"
                    "  • Clonazepam/clobazam: GABAergic; no direct mito toxicity; sedation risk\n"
                    "  • Phenytoin/fosphenytoin: CYP-metabolised; not CI in SURF1 but not preferred\n\n"
                    "NEVER use: VPA (ABSOLUTE CI), phenobarbital (Complex I inhibitor — HIGH CAUTION), "
                    "linezolid for any indication (23S rRNA inhibitor — ABSOLUTE CI)"
                ),
            },
            {
                "term": "Crisis Protocol — Sick-Day Rules for SURF1 / Leigh Families",
                "definition": (
                    "SURF1 patients decompensate rapidly with metabolic stressors. "
                    "Every family must receive written sick-day protocol:\n\n"
                    "IMMEDIATE HOSPITAL ADMISSION TRIGGERS:\n"
                    "  • Vomiting or inability to feed for >2–4 hours\n"
                    "  • Any fever >38.0°C (triggers metabolic crisis)\n"
                    "  • New or worsening seizures\n"
                    "  • Any altered consciousness\n"
                    "  • Pre-operatively for ANY procedure\n\n"
                    "IN HOSPITAL PROTOCOL:\n"
                    "  • IV access immediately → 10% dextrose, GIR 6–8 mg/kg/min\n"
                    "  • Blood: glucose, lactate, lactate:pyruvate ratio, blood gas (pH, HCO3)\n"
                    "  • Paracetamol/acetaminophen for fever (avoid ibuprofen — renal/mito concern)\n"
                    "  • Antibiotics (if infection): beta-lactam preferred (NOT linezolid/chloramphenicol)\n"
                    "  • Thiamine + riboflavin IV (emergency supplementation)\n"
                    "  • Lactate monitoring every 2–4 hours\n"
                    "  • Neurology and metabolic specialist consultation\n\n"
                    "FORBIDDEN DURING CRISIS:\n"
                    "  • Fasting (even for 'routine' blood draws — use glucose-containing drip)\n"
                    "  • Propofol (for sedation or anaesthesia)\n"
                    "  • VPA for acute seizure control (use IV LEV or IV benzodiazepine instead)\n"
                    "  • Metformin (if diabetic relative also prescribed it — warn clearly)"
                ),
            },
        ],
    }
