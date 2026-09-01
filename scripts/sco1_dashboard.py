#!/usr/bin/env python3
"""SCO1 — Neonatal Hepatic Failure + Encephalomyopathy due to Complex IV (COX) Deficiency.

SCO1 (Synthesis of Cytochrome c Oxidase 1) is a copper chaperone essential for
the final step of copper loading at the CuA binuclear copper centre of COX2 (MT-CO2),
the catalytic core subunit of cytochrome c oxidase (Complex IV).

  SCO1 gene           OMIM *603644
  Disease (CEMCOX2)   OMIM #604377
  Complex IV (COX)    deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (SCO1 / CuA copper loading / Complex IV assembly):
SCO1 is a 301 amino acid type II transmembrane protein anchored to the inner
mitochondrial membrane (IMM), with its functional domain facing the IMS.
It contains a conserved CXXXXC copper-binding motif (4 inter-cysteine residues,
vs SCO2's CXXXC with 3) that transiently coordinates a single Cu+ ion.
  • Copper trafficking pathway (COX2 CuA maturation):
    COX17 (IMS labile copper pool)
    → SCO2 (Cu+ delivery to CuA — first copper insertion)
    → SCO1 (Cu2+ → Cu+ reduction + CuA copper loading — second copper)
    → COX2 (MT-CO2) CuA binuclear {Cu•Cu} centre COMPLETE
  • SCO1 acts DOWNSTREAM of SCO2: SCO1 oxidises the CXXXC thiolate of SCO2
    (transferring electrons) while simultaneously delivering the final copper
    atom to form the mature CuA binuclear cluster
  • Without SCO1: CuA copper loading stalls at the SCO2-Cu+→CuA step →
    COX2 cannot complete assembly → Complex IV (COX) severely deficient
    (<10-15% of normal control activity)
  • Isolated COX deficiency: Complexes I, II, III NORMAL — same biochemical
    fingerprint as SCO2, SURF1, COX10, COX15 — different clinical signature

HEPATIC MECHANISM (Neonatal Hepatic Failure 100%):
The neonatal liver is uniquely vulnerable to COX deficiency for two reasons:
  1. HIGH mitochondrial mass + OXPHOS dependence: the developing liver relies
     on OXPHOS for gluconeogenesis, fatty acid beta-oxidation, ureagenesis, and
     bile acid synthesis — all ATP-intensive processes barely supported by glycolysis
  2. Hepatocyte copper homeostasis — SCO1 also participates in cellular copper
     export in hepatocytes (independent of mitochondrial function); loss of SCO1
     disrupts cytoplasmic copper redistribution, causing oxidative hepatocyte damage
  • Result: acute neonatal hepatic failure — coagulopathy, conjugated
    hyperbilirubinemia, elevated transaminases (AST/ALT), ascites
  • Hepatic failure is the UNIVERSAL cardinal feature (100%) — the defining
    clinical hallmark that distinguishes SCO1 from SCO2 (NO hepatopathy in SCO2)

KEY SCO1 vs SCO2 DDx (CRITICAL — BOTH CAUSE ISOLATED COX DEFICIENCY):
  • SCO1  →  HEPATOPATHY 100% dominant feature, HCM only ~20%
  • SCO2  →  HCM 100% dominant feature, NO hepatopathy (hepatopathy is a
             KEY negative finding that points AWAY from SCO2 toward SCO1)
  • When a neonate has isolated COX deficiency + liver failure → SCO1
  • When a neonate has isolated COX deficiency + HCM → SCO2
  • The SCO1/SCO2 DDx is THE most critical in COX assembly factor diseases

MOLECULAR: Biallelic (AR) loss-of-function SCO1 variants — compound heterozygous
predominant (homozygous rare):
  — p.Pro174Leu (c.521C>T): most common allele; North European/Canadian founder;
    disrupts helix packing at CXXXXC motif; severely impairs copper binding
  — p.His221Gln (c.663C>A): second most common; metal-binding residue substitution
  — Null alleles (frameshift/nonsense): severe complete LOF; compound heterozygous
    with missense is most common compound genotype
  — p.Glu215Lys: reported in Italian cohort
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 597
DISEASE_ID   = "sco1"
DISEASE_NAME = "SCO1 Neonatal Hepatic Failure + Encephalomyopathy (Complex IV / COX Deficiency)"
GENE         = "SCO1"
PROTEIN      = "SCO1 — 301 aa, IMM-anchored IMS-facing copper chaperone, CXXXXC motif, CuA Cu2+→Cu+ reduction + final copper loading"
OMIM_GENE    = "*603644"
OMIM_DISEASE = "#604377"
CHROMOSOME   = "17p13.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal (hours to days of life; range birth – 2 weeks)"
COHORT_SIZE  = 40
COLOR        = "#e65100"   # deep amber-orange — liver disease / hepatic failure / SCO1
LIGHT        = "#fff3e0"


# Genotype pool
GENO_P174L_HQ      = "p.Pro174Leu / p.His221Gln (c.521C>T / c.663C>A) — compound heterozygous; North European/Canadian"
GENO_P174L_NULL    = "p.Pro174Leu / truncating — compound heterozygous (null allele); severe"
GENO_NULL_MISSENSE = "Truncating / missense — compound heterozygous; complete LOF allele"
GENO_P215K_CPX     = "p.Glu215Lys / missense — compound heterozygous; Italian cohort"

GENO_POOL    = [GENO_P174L_HQ, GENO_P174L_NULL, GENO_NULL_MISSENSE, GENO_P215K_CPX]
GENO_WEIGHTS = [0.40,           0.30,             0.20,               0.10]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient SCO1 cohort (seed-597)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_FEATURES_POOL = [
    "Neonatal hepatic failure",
    "Lactic acidosis",
    "Hypotonia",
    "Encephalopathy",
    "Failure to thrive",
    "Hepatomegaly",
    "Coagulopathy",
    "Hypoglycaemia",
    "Respiratory compromise",
    "Cardiomyopathy (minority)",
    "Renal tubulopathy",
    "Cholestasis",
    "Elevated transaminases",
    "Neurodegeneration",
]

_TX_POOL = [
    "IV Dextrose GIR 6-8",
    "CoQ10/Ubiquinol",
    "Riboflavin B2",
    "Carnitine",
    "Liver transplant (attempted)",
    "Supportive ICU",
    "Thiamine B1",
    "NaHCO3 (lactic acidosis)",
    "Fresh frozen plasma (coagulopathy)",
    "LEV (seizures, if present)",
    "UDCA (cholestasis)",
]

_OUTCOMES = [
    "Died — hepatic failure (neonatal)",
    "Died — multi-organ failure (NICU)",
    "Died — hepatic failure (week 2-4)",
    "Alive — liver transplant (partial recovery, encephalopathy persists)",
    "Alive — supportive (moderate encephalomyopathy)",
]
_OUT_WEIGHTS = [0.35, 0.20, 0.20, 0.15, 0.10]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno       = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex        = rng.choice(["M", "F"])
        onset_day  = rng.choices([1, 2, 3, 5, 7, 10, 14], weights=[15, 20, 18, 15, 12, 12, 8])[0]
        lactate    = round(rng.uniform(6.0, 22.0), 1)
        cox_pct    = rng.randint(4, 15)
        alt        = rng.randint(180, 1400)   # U/L — elevated transaminases
        inr        = round(rng.uniform(1.8, 6.5), 1)   # coagulopathy marker
        has_hcm    = rng.random() < 0.20      # HCM only 20% in SCO1

        # Feature flags — sampled independently for accurate frequency computation
        has_hypotonia   = rng.random() < 0.88
        has_encephalo   = rng.random() < 0.90
        has_hepatomeg   = rng.random() < 0.85
        has_coagulopathy= rng.random() < 0.78
        has_ftt         = rng.random() < 0.80
        has_hypoglycemia= rng.random() < 0.70
        has_resp        = rng.random() < 0.55
        has_renal       = rng.random() < 0.30
        has_cholestasis = rng.random() < 0.72
        has_neurodegenr = rng.random() < 0.68

        feat_list = ["Neonatal hepatic failure"]
        if has_hypotonia:   feat_list.append("Hypotonia")
        if has_encephalo:   feat_list.append("Encephalopathy")
        if has_hepatomeg:   feat_list.append("Hepatomegaly")
        if has_coagulopathy:feat_list.append("Coagulopathy")
        if has_ftt:         feat_list.append("Failure to thrive")
        if has_hcm:         feat_list.append("Cardiomyopathy")
        if has_cholestasis: feat_list.append("Cholestasis")
        if has_hypoglycemia:feat_list.append("Hypoglycaemia")
        if has_resp:        feat_list.append("Respiratory compromise")
        if has_renal:       feat_list.append("Renal tubulopathy")
        if has_neurodegenr: feat_list.append("Neurodegeneration")

        txs = rng.sample(_TX_POOL, k=rng.randint(3, 6))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":            f"SCO1-{i:03d}",
            "geno":          geno,
            "sex":           sex,
            "onset_day":     onset_day,
            "lactate":       lactate,
            "cox_pct":       cox_pct,
            "alt":           alt,
            "inr":           inr,
            "has_hcm":       has_hcm,
            "has_hypotonia": has_hypotonia,
            "has_encephalo": has_encephalo,
            "has_hepatomeg": has_hepatomeg,
            "has_coagulopathy": has_coagulopathy,
            "has_ftt":       has_ftt,
            "has_hypoglycemia": has_hypoglycemia,
            "has_resp":      has_resp,
            "has_renal":     has_renal,
            "has_cholestasis": has_cholestasis,
            "has_neurodegenr": has_neurodegenr,
            "features":      ", ".join(feat_list[:7]),
            "treatments":    ", ".join(txs[:5]),
            "outcome":       outcome,
        })
    return patients


# ── Overview ──────────────────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    rng      = _rng()
    patients = _generate_cohort(rng)

    died  = sum(1 for p in patients if p["outcome"].startswith("Died"))
    alive = COHORT_SIZE - died

    def _pct(key: str) -> int:
        return round(sum(1 for p in patients if p[key]) / COHORT_SIZE * 100)

    feature_frequencies = {
        "Neonatal Hepatic Failure (CARDINAL)": 100,
        "Lactic Acidosis (≥5 mmol/L)": round(sum(1 for p in patients if p["lactate"] >= 5) / COHORT_SIZE * 100),
        "Hypotonia": _pct("has_hypotonia"),
        "Encephalopathy": _pct("has_encephalo"),
        "Hepatomegaly": _pct("has_hepatomeg"),
        "Coagulopathy (INR >1.8)": _pct("has_coagulopathy"),
        "Failure to Thrive": _pct("has_ftt"),
        "Hypoglycaemia": _pct("has_hypoglycemia"),
        "Cholestasis": _pct("has_cholestasis"),
        "Cardiomyopathy (~20% — KEY DDx SCO2-HCM-100%)": _pct("has_hcm"),
        "Respiratory Compromise": _pct("has_resp"),
        "Renal Tubulopathy": _pct("has_renal"),
        "Neurodegeneration": _pct("has_neurodegenr"),
        "NO Iron Overload (KEY DDx GRACILE)": 100,
        "Isolated COX Deficiency (CI/CII/CIII Normal)": 100,
        "Alive at 1 Year": round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)", "value": COHORT_SIZE, "color": COLOR},
        {"label": "Hepatic Failure", "value": "100%", "color": "#bf360c"},
        {"label": "Lactic Acidosis", "value": f"{feature_frequencies['Lactic Acidosis (≥5 mmol/L)']}%", "color": "#b71c1c"},
        {"label": "Hypotonia", "value": f"{feature_frequencies['Hypotonia']}%", "color": COLOR},
        {"label": "HCM (~20%)", "value": f"{feature_frequencies['Cardiomyopathy (~20% \u2014 KEY DDx SCO2-HCM-100%)']}%", "color": "#4a148c"},
        {"label": "Fatal ≤1yr", "value": f"{round(died/COHORT_SIZE*100)}%", "color": "#c62828"},
        {"label": "Alive", "value": f"{alive}", "color": "#2e7d32"},
        {"label": "Seed", "value": f"#{SEED}", "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — Triple Mechanism",
            "mechanism": (
                "CoA sequestration (depletes mitochondrial acetyl-CoA), POLG inhibition "
                "(depletes mtDNA), and direct hepatotoxicity. With SCO1 hepatopathy already "
                "present, VPA is catastrophically additive — reported to precipitate acute "
                "liver failure and death in mitochondrial hepatopathy. NEVER use. "
                "Alternatives: LEV (renal excretion, no mitochondrial toxicity), topiramate "
                "(monitor ammonia), zonisamide."
            ),
        },
        {
            "drug": "Metformin",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "Inhibits Complex I of the respiratory chain → accumulates reducing equivalents "
                "→ exacerbates already-impaired OXPHOS → lactic acidosis crisis. In SCO1 "
                "patients this can be fatal within hours. Contraindicated permanently. "
                "If diabetes coexists: insulin only."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "Inhibits mitochondrial 23S rRNA-equivalent, blocking translation of all 13 "
                "mtDNA-encoded OXPHOS subunits including MT-CO1, MT-CO2, MT-CO3 — eliminating "
                "any residual Complex IV (COX) assembly substrate. In SCO1 patients with "
                "already <15% COX activity, linezolid collapses residual OXPHOS. Use "
                "tedizolid only if essential with extreme monitoring; prefer non-oxazolidinone "
                "alternatives."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "ABSOLUTE CI — PRIS + Hepatic Failure",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol impairs mitochondrial beta-oxidation "
                "and Complex IV activity directly. In SCO1 patients, COX is already severely "
                "deficient — propofol infusion (especially >4mg/kg/h) causes fatal PRIS. "
                "Additionally, SCO1 hepatic failure impairs propofol clearance (hepatic metabolism), "
                "amplifying toxicity. Use volatile agents (sevoflurane/isoflurane) for anaesthesia; "
                "avoid total IV anaesthesia protocols using propofol."
            ),
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "severity": "CONTRAINDICATED",
            "mechanism": (
                "Beta-oxidation of fatty acids feeds acetyl-CoA into the TCA cycle and "
                "requires OXPHOS (Complex IV) for re-oxidation of reducing equivalents. "
                "In SCO1 COX deficiency, KD substrate overwhelms the impaired OXPHOS → "
                "accumulation of acylcarnitines + lactate surge. Glucose (IV dextrose GIR "
                "6-8 mg/kg/min) is the preferred energy substrate."
            ),
        },
        {
            "drug": "Hepatotoxic Drugs (General)",
            "severity": "HIGH CAUTION / AVOID",
            "mechanism": (
                "SCO1 patients have BASELINE hepatic failure — any hepatotoxic agent "
                "compounds this: paracetamol above 15mg/kg/day, azathioprine, fluconazole, "
                "rifampicin. Even standard doses of acetaminophen require dose reduction. "
                "Always check hepatic metabolism (CYP2E1, CYP1A2, glucuronidation) before "
                "prescribing — these pathways are compromised."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Not First Line",
            "mechanism": (
                "Phenobarbital inhibits Complex I (NADH:ubiquinone oxidoreductase), "
                "worsening OXPHOS in an already-deficient respiratory chain. Additionally, "
                "phenobarbital is hepatically metabolised and a CYP inducer that increases "
                "oxidative stress in a failing liver. Use LEV (levetiracetam) instead — "
                "renally excreted, no mitochondrial toxicity, no CYP induction."
            ),
        },
    ]

    return {
        "gene":              GENE,
        "protein":           PROTEIN,
        "disease":           DISEASE_NAME,
        "omim_gene":         OMIM_GENE,
        "omim_disease":      OMIM_DISEASE,
        "chromosome":        CHROMOSOME,
        "inheritance":       INHERITANCE,
        "onset":             ONSET,
        "cohort":            f"{COHORT_SIZE} patients · seed-{SEED} · SCO1 biallelic (Hepatic Failure + COX Deficiency)",
        "mechanism": (
            "SCO1 is a 301 aa type-II IMM transmembrane protein with its CXXXXC copper-binding "
            "domain facing the IMS. It occupies the LAST step of CuA copper loading at COX2 "
            "(MT-CO2): SCO2 delivers the first Cu+ atom and presents the CuA thiolates; SCO1 "
            "then reduces Cu2+→Cu+ and inserts the second copper to form the mature CuA binuclear "
            "{Cu•Cu} cluster. Loss of SCO1 → CuA incompletely metalated → COX2 cannot assemble "
            "into the COX holoenzyme → isolated Complex IV deficiency (<15% of normal). "
            "Complexes I, II, and III are NORMAL — the same biochemical fingerprint as SCO2, "
            "SURF1, COX10, COX15. The distinct organ vulnerability (liver >> heart) distinguishes "
            "SCO1 from SCO2 despite identical biochemistry. SCO1 also participates in cytoplasmic "
            "copper homeostasis in hepatocytes; SCO1 loss disrupts copper efflux, generating "
            "hepatocellular reactive oxygen species and amplifying liver injury independent of "
            "direct OXPHOS failure."
        ),
        "hepatic_note": (
            "HEPATOPATHY 100% CARDINAL — THE defining feature of SCO1 disease. "
            "Neonates present hours to days after birth with acute hepatic failure: "
            "elevated AST/ALT (10-50× ULN), INR >2.0, conjugated hyperbilirubinemia, "
            "hypoglycaemia, hepatomegaly, and progressive coagulopathy. This is the "
            "CRITICAL DDx from SCO2 (NO hepatopathy in SCO2 — hepatopathy = SCO1). "
            "SCO1 vs GRACILE: SCO1 has NO iron overload (GRACILE iron overload 100%). "
            "SCO1 vs POLG/DGUOK: SCO1 has ISOLATED COX deficiency (I, II, III normal). "
            "Liver transplant has been attempted but does NOT cure the encephalomyopathy "
            "— mitochondrial disease persists in brain and other organs."
        ),
        "kpis":               kpis,
        "feature_frequencies": feature_frequencies,
        "contraindications":  contraindications,
    }


# ── Breakdown (patients + feature frequencies) ─────────────────────────────
def get_breakdown() -> dict[str, Any]:
    rng      = _rng()
    patients = _generate_cohort(rng)

    def _pct2(key: str) -> int:
        return round(sum(1 for p in patients if p[key]) / COHORT_SIZE * 100)

    feat_freq = {
        "Neonatal Hepatic Failure (CARDINAL 100%)": 100,
        "Lactic Acidosis (≥5 mmol/L)": round(sum(1 for p in patients if p["lactate"] >= 5) / COHORT_SIZE * 100),
        "Hepatomegaly": _pct2("has_hepatomeg"),
        "Hypotonia": _pct2("has_hypotonia"),
        "Encephalopathy": _pct2("has_encephalo"),
        "Coagulopathy (INR >1.8)": _pct2("has_coagulopathy"),
        "Failure to Thrive": _pct2("has_ftt"),
        "Cholestasis": _pct2("has_cholestasis"),
        "Hypoglycaemia": _pct2("has_hypoglycemia"),
        "Elevated Transaminases (ALT >10× ULN)": round(sum(1 for p in patients if p["alt"] > 400) / COHORT_SIZE * 100),
        "Neurodegeneration": _pct2("has_neurodegenr"),
        "Respiratory Compromise": _pct2("has_resp"),
        "Renal Tubulopathy": _pct2("has_renal"),
        "Cardiomyopathy (~20% — KEY DDx SCO2-HCM-100%)": _pct2("has_hcm"),
        "NO Iron Overload (KEY DDx GRACILE)": 100,
        "Isolated COX Deficiency (CI/CII/CIII Normal)": 100,
        "Died ≤ 1 Year": round(sum(1 for p in patients if p["outcome"].startswith("Died")) / COHORT_SIZE * 100),
    }

    return {
        "patients":          patients,
        "feature_frequencies": feat_freq,
    }


# ── Definitions ────────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "SCO1 — Synthesis of Cytochrome c Oxidase 1 (CXXXXC Copper Chaperone, IMM, 17p13.2)",
            "definition": (
                "SCO1 encodes a 301 amino acid type-II single-pass transmembrane protein of the inner "
                "mitochondrial membrane (IMM), with its functional globular domain projecting into the "
                "intermembrane space (IMS). Its name derives from its yeast homologue Sco1p.\n\n"
                "Structural hallmark: a CXXXXC motif (4 inter-cysteine residues, unlike SCO2's CXXXC "
                "with 3) that coordinates a single Cu ion with two additional conserved histidine "
                "residues. This metal-binding site is essential for copper acceptance from COX17 "
                "via SCO2 and subsequent transfer to the CuA site of COX2.\n\n"
                "Function in CuA assembly:\n"
                "  1. SCO2 receives Cu+ from COX17 → presents one copper atom to the CuA cysteines of COX2\n"
                "  2. SCO1 docks with the SCO2-COX2 intermediate, reduces Cu2+→Cu+ (electron transfer)\n"
                "     and inserts the SECOND copper atom, forming the mature CuA {Cu•Cu} binuclear cluster\n"
                "  3. Mature CuA is the primary electron acceptor from reduced cytochrome c (Cyt c2+)\n\n"
                "SCO1 is therefore INDISPENSABLE for the very last step of CuA metalation. No other "
                "chaperone can substitute. Loss → COX2 assembly fails → Complex IV severely deficient."
            ),
        },
        {
            "term": "COX17 → SCO2 → SCO1 → COX2 CuA — Complete Copper Trafficking Hierarchy",
            "definition": (
                "The CuA binuclear copper centre of COX2 (MT-CO2) requires sequential action of "
                "three copper chaperones in the IMS:\n\n"
                "  Step 1: COX17 (OMIM *604427, 69 aa, twin CX9C motif)\n"
                "    • Accepts Cu+ from the cytoplasmic copper chaperone ATOX1 via the IMM copper pool\n"
                "    • Delivers Cu+ to SCO2 and SCO1 (bifunctional donor)\n\n"
                "  Step 2: SCO2 (OMIM *604272, 266 aa, CXXXC motif)\n"
                "    • Receives Cu+ from COX17\n"
                "    • Inserts the FIRST copper atom into the CuA thiolate pocket of COX2\n"
                "    • Mutations → HCM 100% (cardiac dominant phenotype)\n\n"
                "  Step 3: SCO1 (OMIM *603644, 301 aa, CXXXXC motif)\n"
                "    • Receives Cu+ from COX17 (second copper)\n"
                "    • Acts on the SCO2-COX2 intermediate: reduces Cu2+→Cu+ and inserts final copper\n"
                "    • Mutations → Neonatal Hepatic Failure 100% (hepatic dominant phenotype)\n\n"
                "Despite sharing the same end-point (CuA deficiency → COX deficiency), SCO1 and SCO2 "
                "mutations produce OPPOSITE organ-dominant phenotypes — this is the defining DDx."
            ),
        },
        {
            "term": "Isolated Complex IV (COX) Deficiency — SCO1 Biochemical Signature",
            "definition": (
                "SCO1 deficiency produces isolated COX (Complex IV) deficiency. Enzyme assays show:\n"
                "  • Complex IV (COX): <10-15% of normal control activity → severely reduced\n"
                "  • Complex I (NADH:ubiquinone oxidoreductase): NORMAL\n"
                "  • Complex II (succinate:ubiquinone oxidoreductase): NORMAL\n"
                "  • Complex III (ubiquinol:cytochrome c oxidoreductase): NORMAL\n"
                "  • Citrate synthase (mitochondrial mass marker): normal to mildly elevated\n\n"
                "This is the same biochemical fingerprint as SCO2, SURF1, COX10, COX15, LRPPRC — "
                "isolated COX deficiency is the entire Complex IV assembly factor disease class.\n\n"
                "Key diagnostic tissue: liver biopsy (COX histochemistry + enzyme activity) OR "
                "fibroblast COX assay (gold standard — fibroblasts cultured from skin biopsy). "
                "Blood and CSF lactate/pyruvate ratio typically >20 (LP ratio — OXPHOS block "
                "diagnosis criterion). Molecular confirmation: SCO1 biallelic variants by NGS gene "
                "panel (mitochondrial disease panel or WES/WGS)."
            ),
        },
        {
            "term": "p.Pro174Leu (c.521C>T) — North European/Canadian SCO1 Founder Allele",
            "definition": (
                "The most common SCO1 pathogenic variant worldwide. Proline 174 lies within the "
                "helix that packs against the CXXXXC copper-binding loop — substitution of the "
                "conformationally rigid proline with the flexible leucine disrupts this helix packing, "
                "causing conformational change at the CXXXXC motif that impairs copper binding "
                "affinity by ~5-fold (Leary et al. 2007, HumMolGenet).\n\n"
                "Population genetics: carrier frequency estimated ~1:150-200 in Northern European "
                "populations; most compound heterozygous with p.His221Gln or null alleles.\n\n"
                "Genotype-phenotype correlation: p.Pro174Leu/p.His221Gln (the original compound "
                "heterozygous genotype reported by Valnot et al. 2000) → severe neonatal hepatic "
                "failure; no clear residual function, consistent with near-complete LOF."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "AR (Autosomal Recessive) Inheritance — SCO1 Biallelic Variants",
            "definition": (
                "SCO1 disease follows strict autosomal recessive Mendelian inheritance. Both "
                "parental alleles must carry pathogenic variants. Carrier parents are clinically "
                "unaffected (one functional SCO1 copy is sufficient for normal CuA assembly).\n\n"
                "Recurrence risk: 25% per pregnancy for biallelic-affected offspring.\n"
                "Carrier testing: offer to all first-degree relatives once proband variant(s) confirmed.\n"
                "Prenatal testing: chorionic villus sampling (CVS) or amniocentesis for known "
                "familial variants; preimplantation genetic testing (PGT-M) is available.\n\n"
                "Most SCO1 patients are compound heterozygous (two different pathogenic alleles) "
                "because the disease is rare and consanguinity is not typical in Northern European "
                "cohorts. Homozygous cases (especially p.Pro174Leu/p.Pro174Leu) have been reported "
                "in families with higher carrier rates."
            ),
        },
        {
            "term": "SCO1 vs SCO2 — Genotype-Phenotype Paradox (Same Pathway, Opposite Organ Dominance)",
            "definition": (
                "SCO1 and SCO2 are both copper chaperones for CuA of COX2; both cause isolated "
                "COX deficiency; yet their clinical phenotypes are mirror images:\n\n"
                "SCO1 mutations → HEPATOPATHY dominant (liver failure 100%), HCM ~20%\n"
                "SCO2 mutations → HCM dominant (100%), NO hepatopathy\n\n"
                "Proposed explanation (Leary et al. 2013, Cell Metab):\n"
                "  • SCO1, unlike SCO2, has an additional non-mitochondrial role in hepatocytes: "
                "    it participates in cytoplasmic copper efflux via the Golgi apparatus. Loss of "
                "    SCO1 → intracellular copper accumulation in hepatocytes → ROS → hepatocyte "
                "    death. This additional copper homeostasis function (absent in SCO2) explains "
                "    liver-dominant damage in SCO1.\n"
                "  • Cardiac myocytes have the highest OXPHOS dependence of any cell type, making "
                "    them exquisitely sensitive to COX deficiency; in SCO2 (where the additional "
                "    hepatocyte copper role is intact), the cardiac phenotype dominates.\n\n"
                "Clinical rule: isolated COX deficiency + liver failure → SCO1.\n"
                "              isolated COX deficiency + HCM 100% → SCO2.\n"
                "              Never use liver transplant alone as definitive treatment for SCO1 "
                "              — encephalomyopathy persists."
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "SCO1 DDx Panel — Neonatal Hepatic Failure + Lactic Acidosis (Mitochondrial)",
            "definition": (
                "Differential diagnosis for neonate presenting with hepatic failure + lactic acidosis "
                "where mitochondrial disease is suspected:\n\n"
                "1. SCO1 (COX deficiency): Hepatic failure 100%, NO iron overload, "
                "   HCM only ~20%, isolated COX deficiency (I/II/III normal)\n"
                "2. POLG (mtDNA polymerase gamma): Alpers-Huttenlocher (hepatocerebral), "
                "   multi-complex OXPHOS defect, mtDNA depletion, seizures prominent\n"
                "3. DGUOK (deoxyguanosine kinase): neonatal hepatocerebral, mtDNA depletion, "
                "   nystagmus, hepatic + neurological\n"
                "4. MPV17: hepatocerebral mtDNA depletion, Navajo neurohepatic (founder)\n"
                "5. GRACILE (BCS1L): iron overload 100% (ferritin >2000, sat >90%), Complex III "
                "   deficiency — IRON OVERLOAD distinguishes from SCO1\n"
                "6. Neonatal haemochromatosis (GALD): iron overload + hepatic failure, "
                "   NO lactic acidosis (NOT mitochondrial), IVIg treatment\n"
                "7. Wilson disease: rare neonatal presentation, copper accumulation, Kayser-Fleischer\n"
                "8. Galactosaemia: reducing substances in urine, E. coli sepsis risk, dietary Rx\n"
                "9. Bile acid synthesis defects: cholestasis + liver failure, urinary bile acids dx\n\n"
                "KEY SCO1 fingerprint: hepatic failure + isolated COX deficiency + NO iron overload "
                "+ lactic acidosis (LP ratio >20) + neonatal onset."
            ),
        },
        {
            "term": "Liver Transplant in SCO1 — Partial + Controversial",
            "definition": (
                "Liver transplantation has been performed in a small number of SCO1 patients with "
                "acute liver failure unresponsive to supportive care. Outcomes:\n\n"
                "  • Liver failure: corrected — the transplanted liver has normal SCO1 and normal "
                "    OXPHOS, normalising hepatic function\n"
                "  • Encephalomyopathy: NOT corrected — the diseased brain, muscle, and other "
                "    organs retain biallelic SCO1 mutations; neurological disease progresses\n"
                "  • Long-term survivors post-transplant typically show neurodevelopmental "
                "    delay, myopathy, and metabolic crises affecting non-liver organs\n\n"
                "Contrast with liver transplant in Crigler-Najjar or Wilson disease (curative) — "
                "SCO1 is a systemic disease, not a liver-confined disease.\n\n"
                "The decision to transplant requires multidisciplinary consensus weighing:\n"
                "  - Severity of hepatic failure (acuity, reversibility)\n"
                "  - Degree of extra-hepatic (neurological) involvement\n"
                "  - Family preferences and palliative care discussion\n"
                "  - Prognosis even post-transplant (moderate survival with disability)"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Levetiracetam (LEV) — Preferred AED in SCO1",
            "definition": (
                "Levetiracetam (LEV) is the antiepileptic drug of choice when seizures occur in "
                "SCO1 patients, for multiple reasons:\n"
                "  1. Renal excretion (65% unchanged in urine): avoids hepatic metabolism — "
                "     critical in SCO1 hepatic failure where hepatic CYP enzymes are compromised\n"
                "  2. No mitochondrial toxicity: LEV does not inhibit any respiratory chain "
                "     complex or interfere with OXPHOS\n"
                "  3. No CYP induction: does not induce cytochrome P450 enzymes that could alter "
                "     pharmacokinetics of other drugs used in ICU management\n"
                "  4. IV formulation available: essential for NICU use\n"
                "  5. Broad-spectrum efficacy including focal and generalised seizures\n\n"
                "Dosing in neonates/infants: initiate 10-20 mg/kg/day divided q12h IV; escalate "
                "to 40-60 mg/kg/day if needed. Dose-reduce for renal impairment (GFR <50).\n\n"
                "Avoid: VPA (hepatotoxic + POLG inhibition), phenobarbital (Complex I + hepatic "
                "CYP inducer), phenytoin (hepatically metabolised)."
            ),
        },
        {
            "term": "IV Dextrose Protocol — Energy Substrate in SCO1 (GIR 6-8 mg/kg/min)",
            "definition": (
                "Continuous IV dextrose is the primary metabolic support in SCO1 disease:\n\n"
                "Goal: maintain glucose infusion rate (GIR) 6-8 mg/kg/min to prevent hypoglycaemia "
                "and provide a readily OXPHOS-compatible energy substrate. Glucose → pyruvate → "
                "acetyl-CoA can still partially enter the TCA cycle even with reduced OXPHOS.\n\n"
                "NEVER FAST: even brief fasting in SCO1 → hypoglycaemia + lactic acidosis surge "
                "(glycogen depleted, gluconeogenesis impaired by hepatic failure, fatty acid "
                "oxidation unable to compensate with impaired OXPHOS).\n\n"
                "Monitor: blood glucose q1-2h during acute phase; lactate q4-6h; adjust GIR to "
                "keep glucose 4-6 mmol/L. Avoid glucose >8 mmol/L (osmotic diuresis risk in "
                "compromised kidney).\n\n"
                "Contrast with GRACILE: GIR 8-10 mg/kg/min (more severe hypoglycaemia).\n"
                "KD (ketogenic diet) is CONTRAINDICATED — beta-oxidation requires Complex IV."
            ),
        },
        {
            "term": "CoQ10 / Ubiquinol — Mitochondrial Cofactor Support (Level C Evidence)",
            "definition": (
                "Coenzyme Q10 (CoQ10 / ubiquinol reduced form) is used as adjunctive mitochondrial "
                "cofactor supplementation in SCO1 disease despite no SCO1-specific randomised trial.\n\n"
                "Rationale:\n"
                "  • CoQ10 is the electron carrier between Complexes I/II and Complex III\n"
                "  • Supraphysiological CoQ10 may improve the small residual Complex IV flux\n"
                "  • Antioxidant properties may reduce hepatocellular ROS amplified by copper "
                "    accumulation in SCO1 hepatocytes\n\n"
                "Dosing (Level C / expert consensus): 10-30 mg/kg/day of ubiquinol (reduced form, "
                "superior bioavailability vs ubiquinone). Administer with fat-containing meal.\n\n"
                "Note: CoQ10 does not bypass the SCO1 CuA copper-loading defect — it cannot "
                "rescue Complex IV assembly. It is supportive, not curative.\n\n"
                "Other cofactors: Riboflavin (B2) 50-100 mg/day — Complex I/II FMN/FAD; "
                "Carnitine — secondary carnitine deficiency common in mitochondrial disease; "
                "Thiamine (B1) — empiric in any lactic acidosis until PDHC deficiency excluded."
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(json.dumps({k: v for k, v in ov.items() if k != "contraindications"}, indent=2, default=str))
    print(f"\nKPIs: {[k['label'] + '=' + str(k['value']) for k in ov['kpis']]}")
    print(f"\nContraindications: {len(ov['contraindications'])} drugs")
    print("\n=== BREAKDOWN ===")
    bk = get_breakdown()
    print(f"Patients: {len(bk['patients'])}")
    print(f"Feature frequencies: {list(bk['feature_frequencies'].items())[:5]}")
    print("\n=== DEFINITIONS ===")
    df = get_definitions()
    print(f"Pharmacology terms: {len(df['pharmacology'])}")
    print(f"Gene concepts: {len(df['gene_concepts'])}")
    print(f"Disease concepts: {len(df['disease_concepts'])}")
    print(f"Prescribing safety: {len(df['prescribing_safety'])}")
    print("\n✅ SCO1 dashboard script OK")
