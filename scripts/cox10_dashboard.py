#!/usr/bin/env python3
"""COX10 — Leigh Syndrome + Renal Tubulopathy due to Complex IV (COX) Deficiency.

COX10 (Cytochrome c Oxidase Assembly Factor Heme A:Farnesyltransferase COX10) encodes
a protoheme IX farnesyltransferase that catalyses the FIRST committed step of heme a
biosynthesis — converting protoheme (heme b) to heme o. COX15 then converts heme o
to heme a. Without heme a, COX1 (MT-CO1) cannot mature, and Complex IV cannot assemble.

  COX10 gene          OMIM *602125
  Disease             OMIM #220111 (Cytochrome c oxidase deficiency, infantile
                      encephalomyopathy + Leigh syndrome ± tubulopathy)
  Complex IV (COX)    deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COX10 / heme a biosynthesis / Complex IV assembly):
COX10 is a 441 amino acid multi-pass transmembrane protein of the inner mitochondrial
membrane (IMM). Its active site faces the matrix.
  Heme a biosynthetic pathway:
    Protoheme IX (heme b)   ←  mitochondrial heme biosynthesis (FECH, PPOX)
    ↓  COX10 (protoheme IX farnesyltransferase)
    Heme o (proto-haem farnesylated product, reduced form)
    ↓  COX15 (heme o oxidase → heme a synthase)
    Heme a (13-formyl-15-farnesyl modification of protoporphyrin IX iron)
  • COX1 (MT-CO1) contains TWO heme a groups: heme a (electron relay) and
    heme a3 (the oxygen-binding/reduction site, with CuB)
  • Both heme a centres are essential for COX1 folding, assembly, and catalysis
  • Without COX10: heme o/a biosynthesis stalls → COX1 apo-protein cannot
    integrate its prosthetic groups → COX1 is degraded → Complex IV (COX)
    severely deficient (<10-20% of normal control activity)
  • Isolated COX deficiency: Complexes I, II, III NORMAL — same biochemical
    fingerprint as SCO1, SCO2, SURF1, COX15 — distinguished by TUBULOPATHY

TUBULOPATHY MECHANISM (Renal Fanconi Syndrome 60-70%):
The renal proximal tubule is among the highest OXPHOS-dependent tissues in the body:
  • Proximal tubular reabsorption (glucose, amino acids, phosphate, bicarbonate,
    sodium) is driven by Na+/K+-ATPase, a high-ATP-consuming electrogenic pump
  • OXPHOS failure → insufficient ATP → Na+/K+-ATPase dysfunction → generalised
    Fanconi syndrome (glucosuria, aminoaciduria, phosphaturia, bicarbonaturia)
  • Nephrocalcinosis and tubulointerstitial nephritis have been reported in
    long-surviving COX10-deficient patients
  • CRITICAL DDx: SURF1 (COX deficiency) has rare tubulopathy (<10%);
    COX10 tubulopathy is present in 60-70% — a DEFINING distinguishing feature
    when both SURF1 and COX10 cause Leigh syndrome with isolated COX deficiency

LEIGH SYNDROME (Subacute Necrotising Encephalomyelopathy):
COX10 is a common molecular cause of Leigh or Leigh-like syndrome:
  • Bilateral symmetric signal abnormalities in putamen, caudate, globus pallidus,
    brainstem tegmentum, and cerebral peduncles on T2/FLAIR MRI
  • Psychomotor regression triggered by intercurrent illness
  • LP CSF lactate elevated (>2.1 mmol/L) with elevated CSF/plasma lactate ratio (>0.2)
  • Pattern indistinguishable from SURF1-Leigh on MRI alone — biochemical + genetic
    work-up required; fibroblast COX assay shows isolated deficiency in both

MOLECULAR: Biallelic (AR) loss-of-function COX10 variants — compound heterozygous
predominant (homozygous in consanguineous families):
  — c.1007T>A (p.Leu336His): Brazilian founder mutation; disrupts a conserved
    transmembrane helix; associated with Leigh syndrome + severe tubulopathy
  — c.346C>T (p.Arg116Trp): European; missense at conserved Arg; enzymatic
    activity severely reduced (<5% of normal protoheme farnesyltransferase activity)
  — c.791A>G (p.His264Arg): reported in Turkish/consanguineous families;
    homozygous; encephalomyopathy without prominent tubulopathy
  — c.791+5G>A: splice variant; exon 7 skipping → truncation; LOF
  — Other rare missense/truncating alleles reported globally (10+ families)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 599
DISEASE_ID   = "cox10"
DISEASE_NAME = "COX10 Leigh Syndrome + Renal Tubulopathy (Complex IV / COX Deficiency)"
GENE         = "COX10"
PROTEIN      = "COX10 — 441 aa, IMM multi-pass, protoheme IX farnesyltransferase (heme a/o synthase Step 1)"
OMIM_GENE    = "*602125"
OMIM_DISEASE = "#220111"
CHROMOSOME   = "17p12"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Infantile (2–18 months; range: neonatal – 3 years)"
COHORT_SIZE  = 40
COLOR        = "#00695c"   # deep teal — renal/tubulopathy + Leigh/mitochondrial
LIGHT        = "#e0f2f1"

# Genotype pool
GENO_L336H_COMP   = "p.Leu336His / missense — compound heterozygous; Brazilian-European"
GENO_R116W_NULL   = "p.Arg116Trp / truncating — compound heterozygous; European"
GENO_H264R_HOM    = "p.His264Arg / p.His264Arg — homozygous; Turkish/consanguineous"
GENO_SPL_MISS     = "c.791+5G>A (splice) / missense — compound heterozygous; exon 7 skipping"
GENO_MISC         = "Novel missense / LOF — compound heterozygous; de novo + inherited"

GENO_POOL    = [GENO_L336H_COMP, GENO_R116W_NULL, GENO_H264R_HOM, GENO_SPL_MISS, GENO_MISC]
GENO_WEIGHTS = [0.35,             0.30,             0.15,            0.12,           0.08]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COX10 cohort (seed-599)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8",
    "CoQ10/Ubiquinol",
    "Riboflavin B2",
    "Thiamine B1",
    "Succinate (anaplerotic bypass)",
    "Biotin (empiric, pre-molecular)",
    "NaHCO3 (lactic acidosis)",
    "LEV (seizures)",
    "Fludrocortisone (salt-wasting tubulopathy)",
    "Oral phosphate supplementation",
    "NIV/BiPAP (respiratory failure)",
    "Supportive ICU",
    "Carnitine (secondary deficiency)",
]

_OUTCOMES = [
    "Died — respiratory failure (age 1-3yr)",
    "Died — Leigh crisis, multi-organ failure",
    "Alive — severe encephalomyopathy + CKD",
    "Alive — moderate encephalomyopathy, ambulant",
    "Alive — mild course, missense alleles",
]
_OUT_WEIGHTS = [0.30, 0.25, 0.22, 0.15, 0.08]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno            = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex             = rng.choice(["M", "F"])
        onset_mo        = rng.choices([1, 3, 6, 9, 12, 18, 24, 36],
                                       weights=[5, 12, 20, 20, 18, 12, 8, 5])[0]
        lactate         = round(rng.uniform(3.5, 18.0), 1)
        cox_pct         = rng.randint(6, 20)
        creatinine      = round(rng.uniform(35, 160), 0)   # µmol/L — may be normal or elevated
        urine_glucose   = rng.random() < 0.62              # glucosuria (Fanconi)

        has_leigh_mri   = rng.random() < 0.88
        has_hypotonia   = rng.random() < 0.90
        has_encephalo   = rng.random() < 0.85
        has_tubulopathy = rng.random() < 0.65
        has_seizures    = rng.random() < 0.45
        has_resp        = rng.random() < 0.60
        has_anemia      = rng.random() < 0.25
        has_ataxia      = rng.random() < 0.38
        has_spasticity  = rng.random() < 0.40
        has_nystagmus   = rng.random() < 0.28
        has_regression  = rng.random() < 0.92
        has_snhl        = rng.random() < 0.18

        feat_list = ["Lactic acidosis"]
        if has_leigh_mri:   feat_list.append("Leigh/Leigh-like MRI")
        if has_regression:  feat_list.append("Psychomotor regression")
        if has_hypotonia:   feat_list.append("Hypotonia")
        if has_encephalo:   feat_list.append("Encephalopathy")
        if has_tubulopathy: feat_list.append("Renal tubulopathy (Fanconi)")
        if has_seizures:    feat_list.append("Seizures")
        if has_resp:        feat_list.append("Respiratory compromise")
        if has_ataxia:      feat_list.append("Ataxia")
        if has_spasticity:  feat_list.append("Spasticity")
        if has_nystagmus:   feat_list.append("Nystagmus")
        if has_anemia:      feat_list.append("Normocytic anaemia")
        if has_snhl:        feat_list.append("SNHL")

        txs     = rng.sample(_TX_POOL, k=rng.randint(3, 6))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":              f"COX10-{i:03d}",
            "geno":            geno,
            "sex":             sex,
            "onset_mo":        onset_mo,
            "lactate":         lactate,
            "cox_pct":         cox_pct,
            "creatinine":      creatinine,
            "urine_glucose":   urine_glucose,
            "has_leigh_mri":   has_leigh_mri,
            "has_hypotonia":   has_hypotonia,
            "has_encephalo":   has_encephalo,
            "has_tubulopathy": has_tubulopathy,
            "has_seizures":    has_seizures,
            "has_resp":        has_resp,
            "has_anemia":      has_anemia,
            "has_ataxia":      has_ataxia,
            "has_spasticity":  has_spasticity,
            "has_nystagmus":   has_nystagmus,
            "has_regression":  has_regression,
            "has_snhl":        has_snhl,
            "features":        ", ".join(feat_list[:7]),
            "treatments":      ", ".join(txs[:5]),
            "outcome":         outcome,
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
        "Lactic Acidosis (≥3.5 mmol/L)":              round(sum(1 for p in patients if p["lactate"] >= 3.5) / COHORT_SIZE * 100),
        "Psychomotor Regression (CARDINAL)":            _pct("has_regression"),
        "Leigh / Leigh-like MRI (CARDINAL)":            _pct("has_leigh_mri"),
        "Hypotonia":                                    _pct("has_hypotonia"),
        "Encephalopathy":                               _pct("has_encephalo"),
        "Renal Tubulopathy / Fanconi (DISTINGUISHING)": _pct("has_tubulopathy"),
        "Respiratory Compromise":                       _pct("has_resp"),
        "Seizures":                                     _pct("has_seizures"),
        "Ataxia":                                       _pct("has_ataxia"),
        "Spasticity":                                   _pct("has_spasticity"),
        "Nystagmus":                                    _pct("has_nystagmus"),
        "Normocytic Anaemia":                           _pct("has_anemia"),
        "SNHL":                                         _pct("has_snhl"),
        "NO Hepatopathy (KEY DDx SCO1/POLG/DGUOK)":    100,
        "NO Iron Overload (KEY DDx GRACILE)":           100,
        "NO HCM Dominant (KEY DDx SCO2)":               100,
        "Isolated COX Deficiency (CI/CII/CIII Normal)": 100,
        "Alive":                                        round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",           "value": COHORT_SIZE,                                      "color": COLOR},
        {"label": "Leigh MRI",            "value": f"{feature_frequencies['Leigh / Leigh-like MRI (CARDINAL)']}%",    "color": "#004d40"},
        {"label": "Lactic Acidosis",      "value": f"{feature_frequencies['Lactic Acidosis (≥3.5 mmol/L)']}%",        "color": "#b71c1c"},
        {"label": "Tubulopathy",          "value": f"{feature_frequencies['Renal Tubulopathy / Fanconi (DISTINGUISHING)']}%", "color": "#1565c0"},
        {"label": "Hypotonia",            "value": f"{feature_frequencies['Hypotonia']}%",            "color": COLOR},
        {"label": "Seized",               "value": f"{feature_frequencies['Seizures']}%",             "color": "#6a1b9a"},
        {"label": "Fatal",                "value": f"{round(died/COHORT_SIZE*100)}%",                 "color": "#c62828"},
        {"label": "Seed",                 "value": f"#{SEED}",                                        "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "CoA sequestration (depletes mitochondrial acetyl-CoA pool), POLG inhibition "
                "(depletes mtDNA → further reduces MT-CO1/CO2/CO3 template), and direct "
                "hepatotoxicity. In COX10 patients, VPA compounds OXPHOS failure and may "
                "precipitate fatal lactic acidosis. No safe dose exists — use LEV (renal "
                "excretion, zero mitochondrial toxicity) as first-line AED. If tubulopathy "
                "is present, VPA also worsens metabolic acidosis via CoA depletion."
            ),
        },
        {
            "drug": "Metformin",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "Inhibits Complex I → forces further dependence on glycolysis → lactic acidosis "
                "crisis. COX10 patients already have impaired OXPHOS; adding a Complex I inhibitor "
                "precipitates fatal lactic acidosis. If diabetes mellitus develops (rare in "
                "paediatric COX10), insulin is the only safe agent."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "Inhibits mitochondrial 23S rRNA-equivalent (mt-rRNA), blocking translation of all "
                "13 mtDNA-encoded OXPHOS subunits — including MT-CO1, MT-CO2, MT-CO3, the "
                "structural core of Complex IV. In COX10 patients, MT-CO1 translation already "
                "yields an apo-protein that cannot integrate heme a (unavailable due to COX10 "
                "deficiency); linezolid eliminates even this residual MT-CO1 mRNA translation, "
                "collapsing any spare-capacity COX. Use tedizolid only with extreme caution if "
                "no alternative; prefer non-oxazolidinone agents (amoxicillin-clavulanate, "
                "meropenem, vancomycin) for infections."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "AVOID — PRIS risk",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex IV directly at the "
                "cytochrome aa3 site and impairs fatty acid beta-oxidation. COX10 patients have "
                "intrinsically low COX activity — propofol infusion (>4 mg/kg/h, >48h) carries "
                "severe PRIS risk including rhabdomyolysis, metabolic acidosis, cardiac failure. "
                "Use volatile anaesthetics (sevoflurane, isoflurane) for intraoperative "
                "maintenance; for sedation use dexmedetomidine. If propofol is unavoidable, "
                "limit to single short inductions and monitor CK, lactate, triglycerides."
            ),
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "severity": "CONTRAINDICATED",
            "mechanism": (
                "High-fat beta-oxidation generates NADH and FADH2 that must be re-oxidised via "
                "the respiratory chain; reoxidation at Complex II → ubiquinone → Complex III → "
                "cytochrome c → Complex IV (COX). With COX severely deficient in COX10 disease, "
                "fat oxidation leads to reducing-equivalent accumulation, elevated lactate, and "
                "acylcarnitine build-up. KD is contraindicated in all isolated COX deficiency "
                "diseases (COX10, SURF1, SCO1, SCO2). IV dextrose at GIR 6-8 mg/kg/min is the "
                "preferred energy substrate."
            ),
        },
        {
            "drug": "Chloramphenicol",
            "severity": "AVOID",
            "mechanism": (
                "Same mechanism as linezolid: inhibits 70S (mitochondrial) ribosome elongation, "
                "blocking translation of all 13 mtDNA-encoded OXPHOS subunits including COX "
                "structural components. In COX10 patients, additional OXPHOS translation "
                "inhibition is catastrophic. If treating life-threatening infection, consult "
                "infectious disease; use non-ribosomal alternatives."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Not First Line",
            "mechanism": (
                "Phenobarbital inhibits Complex I (NADH:ubiquinone oxidoreductase), worsening the "
                "already-impaired OXPHOS cascade. Additionally, phenobarbital induces CYP enzymes "
                "increasing oxidative metabolic demand. Use LEV as first-line (renal excretion, no "
                "mitochondrial toxicity). If phenobarbital must be used, monitor lactate closely "
                "and keep plasma levels at the low end of therapeutic range."
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
        "cohort":            f"{COHORT_SIZE} patients · seed-{SEED} · COX10 biallelic (Leigh Syndrome + Tubulopathy + COX Deficiency)",
        "mechanism": (
            "COX10 encodes a 441 aa multi-pass IMM protoheme IX farnesyltransferase — the FIRST "
            "committed step in heme a biosynthesis. It converts protoheme (heme b) to heme o via "
            "farnesyl pyrophosphate addition at the porphyrin vinyl group. COX15 then oxidises heme o "
            "to heme a (the 13-formyl, 17-hydroxyethyl farnesyl-modified iron porphyrin). COX1 "
            "(MT-CO1) requires TWO heme a groups: the heme a electron relay centre and the heme a3 "
            "binuclear oxygen-reduction site (paired with CuB). Without heme a, COX1 apo-protein "
            "cannot integrate its prosthetic groups, is unstable, and is degraded by mitochondrial "
            "quality-control proteases (m-AAA, i-AAA) → Complex IV fails to assemble → isolated "
            "COX deficiency (<20% of normal). Complexes I, II, III are NORMAL — same biochemical "
            "fingerprint as SURF1, SCO1, SCO2, COX15. The defining clinical feature distinguishing "
            "COX10 from SURF1-Leigh is RENAL TUBULOPATHY (Fanconi syndrome, 60-70% of COX10 "
            "patients vs <10% in SURF1)."
        ),
        "tubulopathy_note": (
            "RENAL TUBULOPATHY (FANCONI SYNDROME) — DISTINGUISHING FEATURE OF COX10 vs SURF1:\n"
            "Both COX10 and SURF1 cause isolated Complex IV deficiency and Leigh syndrome, making "
            "them clinically indistinguishable on MRI alone. The KEY DDx is tubulopathy:\n"
            "  • COX10: Renal Fanconi syndrome in 60-70% — glucosuria, aminoaciduria, phosphaturia, "
            "bicarbonaturia, proximal RTA. Nephrocalcinosis and CKD in long-survivors.\n"
            "  • SURF1: Tubulopathy in <10% — NOT a major feature.\n"
            "Mechanism: proximal tubular reabsorption is among the most OXPHOS-demanding processes "
            "in the body (Na+/K+-ATPase driven). COX deficiency → ATP shortage → Na+/K+-ATPase "
            "failure → generalised proximal tubular transport loss (Fanconi). Manage with oral "
            "phosphate, bicarbonate supplementation, fludrocortisone (if salt-wasting), and close "
            "monitoring of renal function / GFR. Avoid nephrotoxic drugs (aminoglycosides, NSAIDs, "
            "high-dose vancomycin). Note: aminoglycosides are not an ABSOLUTE CI in COX10 unlike "
            "KSS, but renal compromise dramatically increases ototoxicity and nephrotoxicity risk."
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
        "Lactic Acidosis (≥3.5 mmol/L)":               round(sum(1 for p in patients if p["lactate"] >= 3.5) / COHORT_SIZE * 100),
        "Psychomotor Regression (CARDINAL)":             _pct2("has_regression"),
        "Leigh / Leigh-like MRI (CARDINAL)":             _pct2("has_leigh_mri"),
        "Hypotonia":                                     _pct2("has_hypotonia"),
        "Encephalopathy":                                _pct2("has_encephalo"),
        "Renal Tubulopathy / Fanconi (DISTINGUISHING)":  _pct2("has_tubulopathy"),
        "Respiratory Compromise":                        _pct2("has_resp"),
        "Seizures":                                      _pct2("has_seizures"),
        "Ataxia":                                        _pct2("has_ataxia"),
        "Spasticity":                                    _pct2("has_spasticity"),
        "Nystagmus":                                     _pct2("has_nystagmus"),
        "Normocytic Anaemia":                            _pct2("has_anemia"),
        "SNHL":                                          _pct2("has_snhl"),
        "Glucosuria (Fanconi)":                          round(sum(1 for p in patients if p["urine_glucose"]) / COHORT_SIZE * 100),
        "NO Hepatopathy (KEY DDx SCO1/POLG/DGUOK)":     100,
        "NO Iron Overload (KEY DDx GRACILE)":            100,
        "Isolated COX Deficiency (CI/CII/CIII Normal)":  100,
        "Died":                                          round(sum(1 for p in patients if p["outcome"].startswith("Died")) / COHORT_SIZE * 100),
    }

    return {
        "patients":            patients,
        "feature_frequencies": feat_freq,
    }


# ── Definitions ────────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "COX10 — Protoheme IX Farnesyltransferase (Heme a/o Synthase Step 1, IMM, 17p12)",
            "definition": (
                "COX10 encodes a 441 amino acid polytopic inner mitochondrial membrane (IMM) protein "
                "that is the founding member of the COX10/CtaB family of bacterial and mitochondrial "
                "heme o synthases. It belongs to the UbiA superfamily of prenyltransferases.\n\n"
                "Enzymatic reaction (COX10):\n"
                "  Protoheme IX (heme b, Fe2+, from FECH) +\n"
                "  Farnesyl pyrophosphate (FPP, from mevalonate pathway)\n"
                "  → Heme o (2-farnesyl,3-methyl-8-vinyl porphyrin, still Fe2+)\n\n"
                "This is the FIRST committed step of heme a biosynthesis, irreversibly committing "
                "protoheme to the pathway. The farnesyl group is added at position 2 of the porphyrin "
                "vinyl group. COX10 requires magnesium cofactor and is membrane-embedded, with the "
                "active site near the matrix face.\n\n"
                "Subsequent step (COX15):\n"
                "  Heme o → Heme a (via oxidative modification at C8: farnesyl → hydroxyethyl-farnesyl,\n"
                "  and C2 via formyl modification — net: heme a has 13-formyl + 17-hydroxyethyl farnesyl "
                "vs heme b baseline)\n\n"
                "COX1 heme sites:\n"
                "  • Heme a: electron relay centre — accepts electron from cytochrome c via CuA\n"
                "  • Heme a3/CuB: binuclear oxygen-reduction site — reduces O2 to 2H2O\n"
                "  Both sites require heme a; COX10 loss abolishes both."
            ),
        },
        {
            "term": "Heme a Biosynthesis — COX10 → COX15 → COX1 (MT-CO1) Assembly Cascade",
            "definition": (
                "The heme a biosynthetic pathway is the critical supply chain for COX1 metalation:\n\n"
                "  1. PPOX/FECH (mitochondrial matrix) → Protoheme IX (heme b)\n"
                "     Source: mitochondrial heme biosynthesis from succinyl-CoA + glycine\n\n"
                "  2. COX10 (IMM, 441 aa) → Heme o\n"
                "     Farnesylation of protoheme b → heme o; first committed step\n\n"
                "  3. COX15 (IMM, 412 aa) → Heme a\n"
                "     Oxidative modification of heme o → heme a; second committed step\n"
                "     COX15 mutations cause overlapping Leigh syndrome (OMIM #615578)\n\n"
                "  4. COX1 (MT-CO1, 514 aa, mtDNA-encoded) → COX apo-protein assembly\n"
                "     COX1 is synthesised by mitochondrial ribosomes; HIGD1A and HIGD2A\n"
                "     stabilise nascent COX1; COX10/COX15-derived heme a is inserted with help\n"
                "     from COX11 (CuB) and CMC1/CMC2 assembly factors\n\n"
                "  5. Full Complex IV holoenzyme (13 subunits: 3 mtDNA + 10 nDNA)\n"
                "     SURF1, SCO1, SCO2, COX20, COX14, TACO1, LRPPRC complete the assembly\n\n"
                "KEY DDx: COX10 vs COX15 — both cause isolated COX deficiency and Leigh syndrome; "
                "distinguish only by molecular genetics (different loci). COX10 has more "
                "tubulopathy (60-70%) than COX15 (~20%). Clinically identical on MRI."
            ),
        },
        {
            "term": "Leigh Syndrome — Bilateral Symmetric Basal Ganglia/Brainstem Necrosis",
            "definition": (
                "Leigh syndrome (subacute necrotising encephalomyelopathy, OMIM #256000) is a "
                "radiological/pathological diagnosis defined by:\n"
                "  • Bilateral, symmetric T2/FLAIR signal abnormalities\n"
                "  • Affecting putamen, caudate, globus pallidus, thalamus, brainstem tegmentum, "
                "    inferior olives, cerebral peduncles (any combination)\n"
                "  • Corresponding to spongiform degeneration, capillary proliferation, and "
                "    neuronal loss on pathology — 'spongy myelinopathy'\n\n"
                "Pathogenesis: OXPHOS failure → ATP depletion → neuronal depolarisation/excitotoxicity "
                "→ selective vulnerability of high-energy-demand relay nuclei (basal ganglia, brainstem). "
                "Metabolic crises (febrile illness, fasting, anaesthesia) trigger decompensation.\n\n"
                "MRI pattern (COX10 vs SURF1):\n"
                "  Both: bilateral symmetric putaminal + brainstem signal\n"
                "  Cannot be distinguished on MRI alone\n"
                "  COX10: tubulopathy on urinalysis provides critical DDx clue\n\n"
                "CSF findings in COX10:\n"
                "  • Lactate elevated (>2.1 mmol/L) in >90%\n"
                "  • CSF protein mildly elevated in some\n"
                "  • CSF/plasma lactate ratio >0.2 supports OXPHOS aetiology\n\n"
                "Biochemistry: COX assay on muscle/fibroblasts — isolated Complex IV deficiency "
                "(<20% of normal). Other complexes normal."
            ),
        },
        {
            "term": "Renal Fanconi Syndrome — Mechanism in COX10 Tubulopathy",
            "definition": (
                "Fanconi syndrome is generalised proximal renal tubular dysfunction characterised by:\n"
                "  • Glucosuria (with normal plasma glucose)\n"
                "  • Generalised aminoaciduria\n"
                "  • Phosphaturia (hypophosphataemia → rickets/osteomalacia in survivors)\n"
                "  • Bicarbonaturia (proximal renal tubular acidosis, Type 2 RTA)\n"
                "  • Uricosuria, kaliuresis\n"
                "  • In severe cases: low-molecular-weight proteinuria\n\n"
                "Mechanism in COX10:\n"
                "  Proximal tubular epithelial cells (PTECs) are among the body's highest "
                "  OXPHOS-consuming cells. Na+/K+-ATPase at the basolateral membrane maintains "
                "  the electrochemical gradient driving secondary active reabsorption of glucose "
                "  (SGLT2), amino acids (SLC1/SLC7 families), phosphate (NaPi-IIa), and "
                "  bicarbonate (NBC1). COX deficiency → insufficient ATP → Na+/K+-ATPase "
                "  failure → loss of all proximal tubular transport simultaneously.\n\n"
                "Management of COX10 tubulopathy:\n"
                "  1. Oral phosphate supplementation (30-50 mg/kg/day in divided doses)\n"
                "  2. Oral bicarbonate (2-5 mEq/kg/day for Type 2 RTA)\n"
                "  3. Fludrocortisone (if salt-wasting + volume depletion)\n"
                "  4. Vitamin D (active form, 1,25-OH2D3) for phosphopenic rickets\n"
                "  5. Avoid nephrotoxic drugs (aminoglycosides double renal risk)\n"
                "  6. Monitor GFR — nephrocalcinosis and progressive CKD in long-survivors\n\n"
                "COX10 vs SURF1 tubulopathy:\n"
                "  COX10: Fanconi in 60-70%  ← MAJOR DISTINGUISHING FEATURE\n"
                "  SURF1: Tubulopathy in <10% ← RARE, NOT a major feature"
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "COX10 Genotype–Phenotype Correlation",
            "definition": (
                "COX10 genotype–phenotype correlation is partial and incomplete:\n\n"
                "  p.Leu336His (c.1007T>A) — Brazilian founder:\n"
                "  • Present in >60% of Brazilian COX10 families\n"
                "  • Associated with Leigh syndrome + severe tubulopathy\n"
                "  • Haplotype analysis: Chr17p12 microsatellite; founder effect in Brazil\n"
                "  • Residual COX10 enzymatic activity: ~3% of normal → severe\n\n"
                "  p.Arg116Trp (c.346C>T) — European:\n"
                "  • Conserved Arg at matrix-facing loop; disrupts substrate binding\n"
                "  • Leigh syndrome; variable tubulopathy\n\n"
                "  p.His264Arg (c.791A>G) — Turkish/consanguineous:\n"
                "  • Homozygous in consanguineous families\n"
                "  • Encephalomyopathy; tubulopathy less prominent\n\n"
                "  Null alleles (frameshift, splice):\n"
                "  • Complete LOF; typically compound heterozygous with missense\n"
                "  • Most severe phenotype; early death common\n\n"
                "  Residual enzymatic activity is the best predictor of phenotypic severity: "
                "<5% → severe Leigh + tubulopathy + early death; 10-20% → milder course, longer survival."
            ),
        },
        {
            "term": "Isolated COX Deficiency — Biochemical Fingerprint Shared by COX10, SURF1, SCO1, SCO2",
            "definition": (
                "All four COX assembly factor diseases produce biochemically identical isolated "
                "Complex IV deficiency:\n"
                "  Complex I (NADH:ubiquinone oxidoreductase): NORMAL\n"
                "  Complex II (succinate dehydrogenase):        NORMAL\n"
                "  Complex III (cytochrome bc1 complex):        NORMAL\n"
                "  Complex IV (cytochrome c oxidase, COX):      SEVERELY REDUCED (<10-20%)\n"
                "  Complex V (ATP synthase):                    NORMAL\n\n"
                "This biochemical fingerprint CANNOT distinguish COX10 from SURF1 from SCO1 "
                "from SCO2 from COX15 from other COX assembly factor diseases. Distinguishing "
                "features are CLINICAL:\n"
                "  COX10:  Leigh + TUBULOPATHY (60-70%) — kidney dominant alongside neuro\n"
                "  SURF1:  Leigh + NO tubulopathy + NO hepatopathy + NO HCM\n"
                "  SCO1:   HEPATOPATHY 100% (neonatal hepatic failure dominant)\n"
                "  SCO2:   HCM 100% (fatal infantile cardiomyopathy dominant)\n\n"
                "Diagnostic workup:\n"
                "  1. Muscle biopsy: COX histochemistry (SDH+/COX- fibres), enzyme assay\n"
                "  2. Fibroblast COX enzyme assay (gold standard, isolated deficiency)\n"
                "  3. Urine amino acids + glucose (Fanconi screen — COX10 specific)\n"
                "  4. Plasma + CSF lactate\n"
                "  5. Brain MRI (Leigh pattern)\n"
                "  6. Gene panel / WES / mtDNA: COX10, SURF1, SCO1, SCO2, COX15, LRPPRC, TACO1"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "COX10 vs SURF1 — Critical DDx (Both Cause Isolated COX Deficiency + Leigh Syndrome)",
            "definition": (
                "COX10 and SURF1 are the two most common nuclear-encoded causes of Leigh syndrome "
                "with isolated Complex IV deficiency. They are biochemically identical but "
                "clinically distinguishable:\n\n"
                "  Feature                COX10           SURF1\n"
                "  ─────────────────────  ──────────────  ───────────────────\n"
                "  Leigh/Leigh-like MRI   90%             95%\n"
                "  Renal tubulopathy      60-70% ←DDx     <10%\n"
                "  Hepatopathy            NO ←DDx SCO1    NO\n"
                "  HCM                    Rare (<5%)      10%\n"
                "  SNHL                   18%             30%\n"
                "  Nystagmus              28%             30%\n"
                "  Seizures               45%             55%\n"
                "  Onset                  Infantile 2-18m Infantile 2-18m\n"
                "  Chromosome             17p12           9q34.2\n"
                "  Key mutation           p.Leu336His     c.845_846delCT (EU)\n"
                "  Diagnosis clue         Urine Fanconi   No tubular findings\n\n"
                "Practical DDx algorithm when isolated COX deficiency + Leigh:\n"
                "  Step 1: Urine amino acids + urine glucose (Fanconi screen)\n"
                "    → Positive (glucosuria + aminoaciduria): COX10 likely\n"
                "    → Negative: SURF1 or other causes\n"
                "  Step 2: Parallel gene sequencing (COX10 + SURF1 + SCO1 + SCO2 panel)"
            ),
        },
        {
            "term": "COX10 vs SCO1/SCO2 vs GRACILE — Four-Way DDx in COX/Complex III Deficiency",
            "definition": (
                "Four mitochondrial diseases cause isolated respiratory chain deficiency in neonates/"
                "infants with Leigh/encephalomyopathic presentation. Critical DDx features:\n\n"
                "  Disease   Complex  Organ Dominant   Iron Overload  Tubulopathy   Hepatopathy\n"
                "  ────────  ───────  ───────────────  ─────────────  ────────────  ──────────\n"
                "  COX10     IV (COX) Neurological/Renal  NO           60-70% ←KEY   NO\n"
                "  SURF1     IV (COX) Neurological         NO           <10%          NO\n"
                "  SCO1      IV (COX) Hepatic 100% ←KEY   NO           30%           100% ←KEY\n"
                "  SCO2      IV (COX) Cardiac HCM 100%    NO           NO            NO\n"
                "  GRACILE   III (bc1) Multi-system       YES ←KEY     80%           YES 85%\n\n"
                "Quick bedside algorithm:\n"
                "  • Isolated COX deficiency + HEPATIC FAILURE → SCO1 (rule out DGUOK, POLG)\n"
                "  • Isolated COX deficiency + HCM → SCO2 (urgent echocardiogram)\n"
                "  • Isolated COX deficiency + LEIGH + TUBULOPATHY → COX10\n"
                "  • Isolated COX deficiency + LEIGH + NO TUBULOPATHY → SURF1\n"
                "  • Complex III deficiency + IRON OVERLOAD → GRACILE (BCS1L)"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "LEV (Levetiracetam) — Preferred AED in COX10 Disease",
            "definition": (
                "Levetiracetam (LEV) is the AED of choice in COX10 disease for the following reasons:\n\n"
                "  1. RENAL excretion: 66% excreted unchanged in urine; minimal hepatic metabolism\n"
                "     → No CYP induction; no hepatotoxic metabolites; no interference with\n"
                "       mitochondrial metabolism\n"
                "  2. No mitochondrial toxicity: LEV does not inhibit any respiratory chain complex\n"
                "  3. IV formulation available: critical for Leigh crisis when oral route compromised\n"
                "  4. Broad-spectrum: focal, myoclonic, generalised tonic-clonic seizures\n"
                "  5. No teratogenic concern in reproductive-age female survivors\n\n"
                "Dosing: 20-60 mg/kg/day in 2 divided doses (paediatric); adjust for renal "
                "impairment if Fanconi syndrome progresses to CKD (eGFR <30: reduce dose 50%).\n\n"
                "Alternatives (if LEV fails):\n"
                "  • Zonisamide (carbonic anhydrase inhibitor; monitor for acidosis in Fanconi)\n"
                "  • Topiramate (monitor ammonia; hepatic metabolism — use with caution)\n"
                "  • ACTH (if infantile spasms — short course)\n"
                "  AVOID: VPA (ABSOLUTE CI), phenobarbital (Complex I inhibitor), vigabatrin\n"
                "         (GABA-T inhibitor, no specific MI CI but no clear benefit)"
            ),
        },
        {
            "term": "IV Dextrose (GIR 6-8) — Energy Substrate in COX10 Metabolic Crisis",
            "definition": (
                "In COX10 disease, OXPHOS is severely deficient. The preferred energy substrate is "
                "glucose (dextrose), delivered as a continuous IV infusion during crises:\n\n"
                "  Glucose infusion rate (GIR): 6-8 mg/kg/min\n"
                "  (GRACILE: 8-10 mg/kg/min due to additional hypoglycaemia risk)\n\n"
                "Rationale:\n"
                "  • Glucose → pyruvate → partial ATP generation via glycolysis (even without OXPHOS)\n"
                "  • Avoids fasting-induced fatty acid mobilisation (which would require COX for FADH2 reoxidation)\n"
                "  • Prevents catabolism-driven lactate surge\n\n"
                "Monitoring during IV dextrose infusion:\n"
                "  • Blood glucose Q2-4h (target 4-8 mmol/L)\n"
                "  • Plasma lactate Q4-8h (target <5 mmol/L)\n"
                "  • Electrolytes (hyponatraemia from free water in D5W; avoid free water excess)\n"
                "  • Serum phosphate (hypophosphataemia common in COX10 tubulopathy)\n\n"
                "NEVER FAST a COX10 patient:\n"
                "  Fasting → fatty acid mobilisation → beta-oxidation → FADH2/NADH overload → "
                "  lactic crisis. Minimum glucose delivery even during anaesthetic induction."
            ),
        },
        {
            "term": "CoQ10/Ubiquinol, Riboflavin, Thiamine, Succinate — Mitochondrial Cofactor Therapy",
            "definition": (
                "Standard mitochondrial cofactor therapy in COX10 disease (all Level C evidence):\n\n"
                "  CoQ10 / Ubiquinol:\n"
                "  • 300-600 mg/day in adults; 10-30 mg/kg/day in children\n"
                "  • Ubiquinol (reduced form) preferred — better oral bioavailability\n"
                "  • Rationale: CoQ10 is the mobile electron carrier between Complex I/II and III;\n"
                "    supplementation may increase spare electron flux and reduce ROS\n"
                "  • Evidence: observational series suggest subjective benefit; no RCT in COX10\n\n"
                "  Riboflavin (B2): 100-400 mg/day\n"
                "  • Cofactor for Complex I (FMN) and Complex II (FAD); may improve residual OXPHOS\n\n"
                "  Thiamine (B1): 100-300 mg/day\n"
                "  • Cofactor for pyruvate dehydrogenase (PDH) and alpha-ketoglutarate dehydrogenase\n"
                "  • MANDATORY empirically in all Leigh syndrome before molecular confirmation\n"
                "    (SLC19A3 thiamine transporter deficiency and THTR2 deficiency are TREATABLE "
                "    mimics of Leigh syndrome — thiamine can be curative)\n\n"
                "  Biotin: 5-20 mg/day empirically\n"
                "  • Biotinidase deficiency (BTD) and holoacarboxylase synthetase (HLCS) deficiency "
                "    are TREATABLE Leigh mimics; biotin is the specific treatment\n"
                "  • Give empirically until BTD enzyme assay results available\n\n"
                "  Succinate: 6-12 g/day (bypass supplement)\n"
                "  • Enters TCA cycle at succinate dehydrogenase (Complex II), bypassing Complex I\n"
                "  • May generate limited ATP via Complex II → ubiquinol → Complex III → IV\n"
                "  • Useful theoretical bypass; limited clinical evidence in isolated COX deficiency"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
