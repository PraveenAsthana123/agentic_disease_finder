#!/usr/bin/env python3
"""COX15 — Leigh Syndrome + Fatal Infantile Hypertrophic Cardiomyopathy (HCM) due to Complex IV (COX) Deficiency.

COX15 (Cytochrome c Oxidase Assembly Factor Heme A:Farnesyl Transferase COX15) encodes
a heme-o oxidase (heme-a synthase) that catalyses the SECOND committed step of heme a
biosynthesis — converting heme o to heme a. COX10 performs the first step (protoheme →
heme o). Without heme a, COX1 (MT-CO1) cannot mature, and Complex IV cannot assemble.

  COX15 gene          OMIM *603646
  Disease             Leigh syndrome / Infantile encephalomyopathy + cardiomyopathy
                      due to cytochrome c oxidase deficiency (related to OMIM #220111)
  Complex IV (COX)    deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COX15 / heme a biosynthesis Step 2 / Complex IV assembly):
COX15 is a 412 amino acid multi-pass transmembrane protein of the inner mitochondrial
membrane (IMM). It catalyses the conversion of heme o to heme a:
  Heme a biosynthetic pathway:
    Protoheme IX (heme b)  ←  mitochondrial heme biosynthesis (FECH, PPOX)
    ↓  COX10 (protoheme IX farnesyltransferase) — STEP 1
    Heme o (2-farnesyl modified porphyrin)
    ↓  COX15 (heme o oxidase → heme a synthase) — STEP 2
    Heme a (13-formyl-2-hydroxyethyl-farnesyl porphyrin, the mature form)
  • COX1 (MT-CO1) contains TWO heme a groups: heme a (electron relay) and
    heme a3 (the oxygen-binding/reduction site, with CuB)
  • Both heme a centres are essential for COX1 folding, assembly, and catalysis
  • Without COX15: heme a biosynthesis stalls at heme o → COX1 apo-protein
    cannot integrate its prosthetic groups → COX1 is degraded → Complex IV (COX)
    severely deficient (<10-20% of normal control activity)
  • Isolated COX deficiency: Complexes I, II, III NORMAL — same biochemical
    fingerprint as COX10, SCO1, SCO2, SURF1

HYPERTROPHIC CARDIOMYOPATHY (HCM) — DISTINGUISHING FEATURE OF COX15:
COX15-deficient hearts develop severe, rapidly progressive HCM:
  • HCM is present in 75-80% of reported COX15 patients — THE cardinal
    distinguishing feature separating COX15 from COX10 (NO HCM in COX10),
    SURF1 (HCM in only ~10%), and SCO1 (HCM in ~20%)
  • Mechanism: myocardium is the most OXPHOS-demanding organ per gram of tissue
    (cardiac output requires ~7 kg ATP/day turnover); isolated COX deficiency →
    severe ATP deficit in cardiomyocytes → hypertrophic remodelling, sarcomeric
    disorganisation, impaired relaxation (restrictive/hypertrophic pattern)
  • Echocardiography: concentric LVH with LVOT obstruction, diastolic dysfunction,
    wall thickness Z-score >+2 SD (often >+5 SD in severe COX15)
  • LVOT obstruction increases systolic load → further energetic demand in an
    already OXPHOS-compromised cell → progressive decompensation
  • Digoxin is CONTRAINDICATED: positive inotrope worsens LVOT obstruction
  • Beta-blockers (propranolol, atenolol) reduce outflow obstruction and O2 demand;
    ACE inhibitors/ARBs AVOID in obstructive HCM (preload reduction dangerous)

LEIGH SYNDROME (COX15 molecular cause):
  • Same neuroradiological pattern as SURF1, COX10, SCO1, SCO2:
    bilateral symmetric putaminal ± brainstem T2/FLAIR signal abnormalities
  • Psychomotor regression triggered by febrile illness or anaesthesia
  • Elevated plasma and CSF lactate; CSF/plasma ratio >0.2
  • Death often in first year due to cardiac failure (HCM) rather than
    respiratory or neurological failure — distinguishes from SURF1 (respiratory
    dominant) and COX10 (renal/encephalopathy dominant)

MOLECULAR: Biallelic (AR) loss-of-function COX15 variants:
  — p.Arg217Trp (c.649C>T): First human COX15 mutation (Oquendo 2004);
    transmembrane domain; severe — infant died at 2 months; HCM + encephalopathy
  — p.Gly339Ser (c.1015G>A): Reported compound heterozygous with c.435+1G>A
    splice mutation; significant residual activity; milder course
  — p.Ala221Thr: Compound heterozygous with truncation; moderate
  — Splice-site and truncating variants: various; typically compound heterozygous
    in non-consanguineous families; rare homozygous in consanguineous pedigrees
  — Overall: <30 unrelated families reported worldwide (ultra-rare)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 601
DISEASE_ID   = "cox15"
DISEASE_NAME = "COX15 Leigh Syndrome + Infantile HCM (Complex IV / COX Deficiency)"
GENE         = "COX15"
PROTEIN      = "COX15 — 412 aa, IMM multi-pass, heme-o oxidase / heme-a synthase (Step 2)"
OMIM_GENE    = "*603646"
OMIM_DISEASE = "#220111 (COX deficiency / Leigh syndrome)"
CHROMOSOME   = "10q24.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Infantile (1–6 months; range: neonatal – 12 months)"
COHORT_SIZE  = 40
COLOR        = "#880e4f"   # deep magenta — cardiac / HCM + Leigh/mitochondrial
LIGHT        = "#fce4ec"

# Genotype pool
GENO_R217W_COMP   = "p.Arg217Trp / missense — compound heterozygous; first reported (Oquendo 2004)"
GENO_G339S_COMP   = "p.Gly339Ser / splice c.435+1G>A — compound heterozygous; milder"
GENO_A221T_COMP   = "p.Ala221Thr / truncating — compound heterozygous; moderate"
GENO_SPL_NULL     = "Splice / frameshift — compound heterozygous; null allele"
GENO_HOM          = "Novel missense / missense — homozygous; consanguineous"

GENO_POOL    = [GENO_R217W_COMP, GENO_G339S_COMP, GENO_A221T_COMP, GENO_SPL_NULL, GENO_HOM]
GENO_WEIGHTS = [0.30,             0.25,             0.20,             0.15,           0.10]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COX15 cohort (seed-601)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8",
    "CoQ10/Ubiquinol",
    "Riboflavin B2",
    "Thiamine B1",
    "Biotin (empiric, pre-molecular)",
    "NaHCO3 (lactic acidosis)",
    "LEV (seizures)",
    "Propranolol (HCM — LVOT obstruction)",
    "Furosemide (HF symptoms)",
    "Carnitine (secondary deficiency)",
    "NIV/BiPAP (respiratory failure)",
    "Supportive ICU",
    "Atenolol (HCM rate control)",
]

_OUTCOMES = [
    "Died — cardiac failure (HCM, age <1yr)",
    "Died — Leigh crisis + cardiac failure",
    "Died — respiratory failure (age 1-2yr)",
    "Alive — severe HCM + encephalomyopathy",
    "Alive — moderate course, partial residual COX",
]
_OUT_WEIGHTS = [0.35, 0.25, 0.18, 0.14, 0.08]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno            = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex             = rng.choice(["M", "F"])
        onset_mo        = rng.choices([1, 2, 3, 4, 6, 9, 12],
                                       weights=[10, 18, 22, 18, 15, 10, 7])[0]
        lactate         = round(rng.uniform(3.0, 16.0), 1)
        cox_pct         = rng.randint(5, 18)

        has_hcm         = rng.random() < 0.78   # HCM — CARDINAL distinguishing feature
        has_leigh_mri   = rng.random() < 0.82
        has_hypotonia   = rng.random() < 0.88
        has_encephalo   = rng.random() < 0.85
        has_tubulopathy = rng.random() < 0.14   # RARE — KEY DDx from COX10 (65%)
        has_seizures    = rng.random() < 0.48
        has_resp        = rng.random() < 0.68
        has_regression  = rng.random() < 0.90
        has_ftt         = rng.random() < 0.72
        has_ataxia      = rng.random() < 0.30
        has_nystagmus   = rng.random() < 0.22
        has_snhl        = rng.random() < 0.12

        feat_list = ["Lactic acidosis"]
        if has_hcm:         feat_list.append("HCM (hypertrophic cardiomyopathy)")
        if has_leigh_mri:   feat_list.append("Leigh/Leigh-like MRI")
        if has_regression:  feat_list.append("Psychomotor regression")
        if has_hypotonia:   feat_list.append("Hypotonia")
        if has_encephalo:   feat_list.append("Encephalopathy")
        if has_ftt:         feat_list.append("Failure to thrive")
        if has_seizures:    feat_list.append("Seizures")
        if has_resp:        feat_list.append("Respiratory compromise")
        if has_tubulopathy: feat_list.append("Renal tubulopathy (Fanconi)")
        if has_ataxia:      feat_list.append("Ataxia")
        if has_nystagmus:   feat_list.append("Nystagmus")
        if has_snhl:        feat_list.append("SNHL")

        txs     = rng.sample(_TX_POOL, k=rng.randint(3, 6))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":              f"COX15-{i:03d}",
            "geno":            geno,
            "sex":             sex,
            "onset_mo":        onset_mo,
            "lactate":         lactate,
            "cox_pct":         cox_pct,
            "has_hcm":         has_hcm,
            "has_leigh_mri":   has_leigh_mri,
            "has_hypotonia":   has_hypotonia,
            "has_encephalo":   has_encephalo,
            "has_tubulopathy": has_tubulopathy,
            "has_seizures":    has_seizures,
            "has_resp":        has_resp,
            "has_regression":  has_regression,
            "has_ftt":         has_ftt,
            "has_ataxia":      has_ataxia,
            "has_nystagmus":   has_nystagmus,
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
        "Lactic Acidosis (≥3.0 mmol/L)":               round(sum(1 for p in patients if p["lactate"] >= 3.0) / COHORT_SIZE * 100),
        "Psychomotor Regression (CARDINAL)":             _pct("has_regression"),
        "HCM — Hypertrophic Cardiomyopathy (DISTINGUISHING)": _pct("has_hcm"),
        "Leigh / Leigh-like MRI":                        _pct("has_leigh_mri"),
        "Hypotonia":                                     _pct("has_hypotonia"),
        "Encephalopathy":                                _pct("has_encephalo"),
        "Failure to Thrive":                             _pct("has_ftt"),
        "Respiratory Compromise":                        _pct("has_resp"),
        "Seizures":                                      _pct("has_seizures"),
        "Ataxia":                                        _pct("has_ataxia"),
        "Nystagmus":                                     _pct("has_nystagmus"),
        "SNHL":                                          _pct("has_snhl"),
        "Renal Tubulopathy (RARE — KEY DDx COX10 65%)": _pct("has_tubulopathy"),
        "NO Hepatopathy (KEY DDx SCO1/POLG/DGUOK)":     100,
        "NO Iron Overload (KEY DDx GRACILE)":            100,
        "Isolated COX Deficiency (CI/CII/CIII Normal)":  100,
        "Alive":                                         round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",     "value": COHORT_SIZE,                                                  "color": COLOR},
        {"label": "HCM",            "value": f"{feature_frequencies['HCM — Hypertrophic Cardiomyopathy (DISTINGUISHING)']}%", "color": "#ad1457"},
        {"label": "Leigh MRI",      "value": f"{feature_frequencies['Leigh / Leigh-like MRI']}%",          "color": "#6a1b9a"},
        {"label": "Lactic Acidosis","value": f"{feature_frequencies['Lactic Acidosis (≥3.0 mmol/L)']}%",   "color": "#b71c1c"},
        {"label": "Hypotonia",      "value": f"{feature_frequencies['Hypotonia']}%",                       "color": COLOR},
        {"label": "Seized",         "value": f"{feature_frequencies['Seizures']}%",                        "color": "#6a1b9a"},
        {"label": "Fatal",          "value": f"{round(died/COHORT_SIZE*100)}%",                             "color": "#c62828"},
        {"label": "Seed",           "value": f"#{SEED}",                                                    "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "CoA sequestration (depletes mitochondrial acetyl-CoA pool), POLG inhibition "
                "(depletes mtDNA → further reduces MT-CO1/CO2/CO3 template), and direct "
                "hepatotoxicity. In COX15 patients, VPA compounds OXPHOS failure and may "
                "precipitate fatal lactic acidosis. In patients with HCM, the combination of "
                "metabolic decompensation + reduced cardiac reserve is catastrophic. No safe "
                "dose exists — use LEV (renal excretion, zero mitochondrial toxicity, cardiac-safe) "
                "as first-line AED in all COX15 patients."
            ),
        },
        {
            "drug": "Metformin",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "Inhibits Complex I → forces further dependence on glycolysis → lactic acidosis "
                "crisis. COX15 patients already have impaired OXPHOS (isolated COX deficiency); "
                "adding a Complex I inhibitor precipitates fatal lactic acidosis. If diabetes "
                "develops in long survivors, insulin is the only safe agent."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "Inhibits mitochondrial 23S rRNA-equivalent (mt-rRNA), blocking translation of all "
                "13 mtDNA-encoded OXPHOS subunits — including MT-CO1, MT-CO2, MT-CO3, the "
                "structural core of Complex IV. In COX15 patients, MT-CO1 translation yields an "
                "apo-protein that cannot integrate heme a (unavailable due to COX15 deficiency); "
                "linezolid eliminates even residual MT-CO1 mRNA translation. With HCM requiring "
                "cardiac output maintenance, any further OXPHOS suppression is lethal. "
                "Use non-oxazolidinone agents for infections."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "ABSOLUTE CI (PRIS + HCM synergy)",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex IV directly at the "
                "cytochrome aa3 site and impairs fatty acid beta-oxidation. COX15 patients have "
                "intrinsically severe COX deficiency — propofol compounds this directly. "
                "Additionally, propofol causes direct myocardial depression, critically dangerous "
                "in a patient with HCM and compromised cardiac output. The combined mitochondrial "
                "+ cardiac depressant effect makes propofol an ABSOLUTE CI in COX15 (analogous to "
                "SCO2). Use volatile anaesthetics (sevoflurane) for anaesthesia; dexmedetomidine "
                "for sedation. NEVER use propofol infusion."
            ),
        },
        {
            "drug": "Digoxin",
            "severity": "ABSOLUTE CI (HCM + LVOT obstruction)",
            "mechanism": (
                "Digoxin is a positive inotrope (increases contractility via Na+/K+-ATPase inhibition "
                "→ elevated intracellular Ca2+). In HCM with LVOT obstruction, increased "
                "contractility WORSENS outflow obstruction, reduces stroke volume, and can "
                "precipitate dynamic LVOT collapse. In COX15 patients with HCM, digoxin causes "
                "acute haemodynamic deterioration and cardiac arrest. Use beta-blockers "
                "(propranolol, atenolol) instead to reduce LVOT gradient and O2 demand."
            ),
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "severity": "CONTRAINDICATED",
            "mechanism": (
                "High-fat beta-oxidation generates NADH and FADH2 that must be re-oxidised via "
                "the respiratory chain; reoxidation at Complex II → ubiquinone → Complex III → "
                "cytochrome c → Complex IV (COX). With COX severely deficient in COX15 disease, "
                "fat oxidation leads to reducing-equivalent accumulation, elevated lactate, and "
                "acylcarnitine build-up. Additionally, high-fat diet increases cardiac workload "
                "in the hypertrophied myocardium. KD is contraindicated in all isolated COX "
                "deficiency diseases. IV dextrose at GIR 6-8 mg/kg/min is the preferred "
                "energy substrate in metabolic crisis."
            ),
        },
        {
            "drug": "ACE Inhibitors / ARBs (in obstructive HCM)",
            "severity": "AVOID in obstructive HCM",
            "mechanism": (
                "In COX15 HCM with LVOT obstruction, vasodilators (ACE inhibitors, ARBs, "
                "nifedipine, hydralazine) reduce systemic vascular resistance (afterload reduction) "
                "and venous preload. This worsens LVOT obstruction by reducing ventricular cavity "
                "size during systole — the Venturi effect increases with reduced preload, "
                "worsening dynamic outflow obstruction → acute syncope or haemodynamic collapse. "
                "If dilated cardiomyopathy develops (late non-obstructive phase), ACEi may become "
                "appropriate — reassess with serial echocardiography."
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
        "cohort":            f"{COHORT_SIZE} patients · seed-{SEED} · COX15 biallelic (Leigh Syndrome + HCM + COX Deficiency)",
        "mechanism": (
            "COX15 encodes a 412 aa multi-pass IMM heme-o oxidase — the SECOND committed "
            "step in heme a biosynthesis. It converts heme o (produced by COX10) to heme a via "
            "an oxidative modification at position C8 (farnesyl → hydroxyethyl-farnesyl) and "
            "formylation at C13, generating the 13-formyl-2-hydroxyethylfarnesyl iron-porphyrin "
            "that is the mature heme a cofactor. COX1 (MT-CO1) requires TWO heme a groups: the "
            "heme a electron relay centre and the heme a3/CuB binuclear oxygen-reduction site. "
            "Without COX15: heme a is absent → COX1 apo-protein cannot integrate prosthetic "
            "groups → COX1 is degraded by mitochondrial quality-control proteases → Complex IV "
            "(COX) fails to assemble → isolated COX deficiency (<18% of normal). Complexes I, "
            "II, III are NORMAL — same biochemical fingerprint as COX10, SURF1, SCO1, SCO2. "
            "The defining clinical feature distinguishing COX15 from other COX-deficiency "
            "syndromes is HYPERTROPHIC CARDIOMYOPATHY (HCM, 75-80%), with relatively RARE "
            "renal tubulopathy (<15%, vs COX10 65%) and a phenotype more cardiac-dominant than "
            "SURF1 (HCM 10%) but less universal than SCO2 (HCM 100%)."
        ),
        "hcm_note": (
            "HYPERTROPHIC CARDIOMYOPATHY (HCM) — DISTINGUISHING FEATURE OF COX15 vs COX10/SURF1:\n"
            "Both COX15 and COX10 cause isolated Complex IV deficiency and Leigh-like MRI, making "
            "them clinically similar on neuroradiology. The KEY DDx is cardiac:\n"
            "  • COX15: HCM in 75-80% — concentric LVH, LVOT obstruction, diastolic dysfunction; "
            "    echo Z-score wall thickness often >+5 SD; cardiac death most common cause of "
            "    mortality in first year.\n"
            "  • COX10: HCM in <5% — NOT a significant cardiac phenotype; renal tubulopathy "
            "    (Fanconi) 65% is the distinguishing feature instead.\n"
            "  • SURF1: HCM in ~10% — infrequent; Leigh + respiratory failure dominant.\n"
            "  • SCO2: HCM in 100% — universal, most severe cardiac phenotype of all COX-assembly "
            "    factor diseases; COX15 has overlapping but less penetrant cardiac disease.\n"
            "Bedside algorithm:\n"
            "  Isolated COX deficiency + LEIGH + HCM 75-80%  → COX15 (echocardiogram mandatory)\n"
            "  Isolated COX deficiency + LEIGH + HCM 100%    → SCO2 (urgent echo)\n"
            "  Isolated COX deficiency + LEIGH + TUBULOPATHY → COX10 (urine amino acids)\n"
            "  Isolated COX deficiency + LEIGH + NO HCM      → SURF1 (no cardiac DDx needed)\n"
            "Management of COX15 HCM: propranolol 0.5-2 mg/kg/day (beta-blocker first-line); "
            "avoid digoxin (ABSOLUTE CI), ACEi/ARB (worsen obstruction in obstructive phase); "
            "echocardiogram every 3-6 months; annual Holter (risk of arrhythmia)."
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
        "Lactic Acidosis (≥3.0 mmol/L)":               round(sum(1 for p in patients if p["lactate"] >= 3.0) / COHORT_SIZE * 100),
        "Psychomotor Regression (CARDINAL)":             _pct2("has_regression"),
        "HCM — Hypertrophic Cardiomyopathy (DISTINGUISHING)": _pct2("has_hcm"),
        "Leigh / Leigh-like MRI":                        _pct2("has_leigh_mri"),
        "Hypotonia":                                     _pct2("has_hypotonia"),
        "Encephalopathy":                                _pct2("has_encephalo"),
        "Failure to Thrive":                             _pct2("has_ftt"),
        "Respiratory Compromise":                        _pct2("has_resp"),
        "Seizures":                                      _pct2("has_seizures"),
        "Ataxia":                                        _pct2("has_ataxia"),
        "Nystagmus":                                     _pct2("has_nystagmus"),
        "SNHL":                                          _pct2("has_snhl"),
        "Renal Tubulopathy (RARE — KEY DDx COX10 65%)": _pct2("has_tubulopathy"),
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
            "term": "COX15 — Heme-o Oxidase / Heme-a Synthase (IMM, 412 aa, 10q24.2)",
            "definition": (
                "COX15 encodes a 412 amino acid polytopic inner mitochondrial membrane (IMM) protein "
                "that catalyses the second and final committed step of heme a biosynthesis:\n\n"
                "Enzymatic reaction (COX15):\n"
                "  Heme o (from COX10) + O2 + NADH\n"
                "  → Heme a (13-formyl-2-hydroxyethylfarnesyl iron-porphyrin)\n\n"
                "Structural features:\n"
                "  • 6 predicted transmembrane helices; matrix-facing active site\n"
                "  • Belongs to the UbiA superfamily (same as COX10)\n"
                "  • Bacterial homologue: CtaA (Bacillus subtilis), Cox15 (S. cerevisiae)\n"
                "  • Yeast Cox15 crystal structure guidance: active site contains conserved "
                "    His residues for haem-iron coordination and a hydrophobic channel for "
                "    the farnesyl side-chain; Arg217 (most common disease mutation) is in TM3\n\n"
                "Heme a product:\n"
                "  • 13-formyl modification (C13 aldehyde not present in heme o)\n"
                "  • 2-hydroxyethyl-farnesyl at C2 (vs farnesyl in heme o)\n"
                "  • The 13-formyl group is critical for electron transfer kinetics at "
                "    the heme a3/CuB oxygen reduction site; without it, Complex IV "
                "    cannot reduce O2 to H2O at physiological rates\n\n"
                "COX1 heme sites:\n"
                "  • Heme a: electron relay centre — accepts electron from cytochrome c via CuA\n"
                "  • Heme a3/CuB: binuclear oxygen-reduction site — reduces O2 to 2H2O and "
                "    pumps 4 protons across the IMM per O2 reduced\n"
                "  Both sites require heme a; COX15 loss abolishes both."
            ),
        },
        {
            "term": "Heme a Biosynthesis — COX10 → COX15 → COX1 (MT-CO1) Assembly Cascade",
            "definition": (
                "The heme a biosynthetic pathway supplies the prosthetic groups for COX1:\n\n"
                "  1. PPOX/FECH (mitochondrial matrix) → Protoheme IX (heme b)\n"
                "     From succinyl-CoA + glycine via 8-step haem biosynthetic pathway\n\n"
                "  2. COX10 (IMM, 441 aa) → Heme o\n"
                "     Farnesylation of protoheme b at C2-vinyl → heme o; first committed step\n"
                "     Gene: COX10, 17p12, OMIM *602125\n\n"
                "  3. COX15 (IMM, 412 aa) → Heme a\n"
                "     Oxidative modification of heme o: C8 farnesyl → hydroxyethyl-farnesyl + "
                "     C13 formylation; second committed step\n"
                "     Gene: COX15, 10q24.2, OMIM *603646\n\n"
                "  4. COX1 (MT-CO1, 514 aa, mtDNA-encoded) → COX apo-protein\n"
                "     COX1 synthesised by mitochondrial ribosomes; HIGD1A/HIGD2A chaperones\n"
                "     stabilise nascent COX1; COX11 inserts CuB; CMC1/CMC2 stabilise COX1\n"
                "     during heme a insertion\n\n"
                "  5. Full Complex IV holoenzyme (13 subunits: 3 mtDNA + 10 nDNA)\n"
                "     SURF1, SCO1, SCO2, COX20, COX14, TACO1, LRPPRC complete assembly\n\n"
                "KEY DDx — COX10 vs COX15 — both cause isolated COX deficiency + Leigh:\n"
                "  COX10: tubulopathy 65% / HCM <5%  → renal dominant\n"
                "  COX15: HCM 75-80% / tubulopathy <15% → cardiac dominant\n"
                "  Cannot distinguish on MRI alone — echocardiogram + urine amino acids mandatory"
            ),
        },
        {
            "term": "COX15 vs SCO2 vs SURF1 — HCM Penetrance Spectrum in COX-Deficiency Syndromes",
            "definition": (
                "Hypertrophic cardiomyopathy (HCM) penetrance across COX-deficiency diseases:\n\n"
                "  Gene   | Complex | HCM | Tubulopathy | Hepatopathy | Iron Overload\n"
                "  -------|---------|-----|-------------|-------------|-------------\n"
                "  SCO2   | IV(CuA) | 100% CARDINAL | <5%  | NO   | NO\n"
                "  COX15  | IV(Cox) | 75-80% MAJOR | <15% | NO   | NO\n"
                "  SCO1   | IV(CuA) | 20%  | <5%  | 100% CARDINAL | NO\n"
                "  SURF1  | IV(Cox) | 10%  | <10% | NO   | NO\n"
                "  COX10  | IV(Cox) | <5%  | 65% CARDINAL | NO | NO\n"
                "  GRACILE| III(bc1)| NO   | 80%  | YES  | 100% CARDINAL\n\n"
                "Clinical algorithm for COX-deficiency + HCM:\n"
                "  • Isolated COX + HCM 100%  → SCO2 (CuA copper chaperone, 22q13.33)\n"
                "  • Isolated COX + HCM 75-80% → COX15 (heme-a synthase, 10q24.2)\n"
                "  • Isolated COX + HCM 20% + Hepatopathy → SCO1 (17p13.2)\n"
                "  • Isolated COX + HCM 10% + Leigh + No Tubulopathy → SURF1 (9q34.2)\n"
                "  • Isolated COX + HCM <5% + Tubulopathy → COX10 (17p12)\n\n"
                "Cardiac management differs by penetrance:\n"
                "  All: propranolol (HCM); avoid digoxin, ACEi/ARBi in obstruction\n"
                "  SCO2/COX15: urgent cardiac transplant assessment if refractory\n"
                "     (NB: transplant does not cure encephalopathy — multidisciplinary ethics)"
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "COX15 Genotype–Phenotype — p.Arg217Trp (c.649C>T) and Allelic Series",
            "definition": (
                "First human COX15 mutation (Oquendo et al., 2004, Hum Mol Genet):\n\n"
                "  p.Arg217Trp (c.649C>T) — TM3 transmembrane domain:\n"
                "  • Infant died at 2 months of age with severe HCM + encephalomyopathy\n"
                "  • Severe OXPHOS defect: COX activity <10% in fibroblasts + muscle\n"
                "  • Protein structurally destabilised; cannot maintain heme-o in active site\n"
                "  • Compound heterozygous with second missense; severe phenotype\n\n"
                "  p.Gly339Ser (c.1015G>A) — C-terminal cytoplasmic loop:\n"
                "  • Reported with splice mutation c.435+1G>A as compound heterozygous\n"
                "  • Milder course: partial residual COX activity (~25-30%)\n"
                "  • Surviving child with moderate HCM and encephalomyopathy\n\n"
                "  p.Ala221Thr — TM3 adjacent; moderate residual activity\n\n"
                "General genotype–phenotype rule:\n"
                "  • Biallelic null (truncating/frameshift) alleles → severe: HCM + Leigh, "
                "    death in infancy\n"
                "  • At least one missense with partial residual enzymatic activity → milder: "
                "    HCM present but survivable; encephalomyopathy less severe\n"
                "  • All COX15 patients: AVOID VPA, metformin, propofol, linezolid, digoxin\n\n"
                "Reference: Oquendo CE et al. Hum Mol Genet. 2004;13(20):2421-2430."
            ),
        },
        {
            "term": "COX15 in the COX-Assembly Factor Disease Spectrum — Heme vs Copper Pathways",
            "definition": (
                "COX-assembly factors fall into two biochemical categories:\n\n"
                "1. HEME A BIOSYNTHESIS PATHWAY (COX10 → COX15 → COX1 heme sites):\n"
                "   COX10 (17p12) — protoheme IX farnesyltransferase; Step 1\n"
                "   COX15 (10q24.2) — heme-o oxidase/heme-a synthase; Step 2\n"
                "   Both cause isolated COX deficiency with Leigh syndrome\n"
                "   Key DDx: COX10 → tubulopathy; COX15 → HCM\n\n"
                "2. CUA COPPER DELIVERY PATHWAY (COX17 → SCO2 → SCO1 → COX2 CuA):\n"
                "   COX17 (3q13.33) — metallochaperone; Cu delivery to IMS\n"
                "   SCO2 (22q13.33) — CXXXC motif; CuA reduction Cu2+ → Cu+; HCM 100%\n"
                "   SCO1 (17p13.2) — CXXXXC motif; final Cu loading to COX2; hepatopathy 100%\n\n"
                "3. COX SCAFFOLD/ASSEMBLY (SURF1 → COX1 heme a3/CuB insertion platform):\n"
                "   SURF1 (9q34.2) — scaffolding for COX1 heme a3/CuB site; Leigh 95%\n\n"
                "4. mRNA STABILITY (LRPPRC → MT-CO1/CO2 mRNA polyadenylation):\n"
                "   LRPPRC (2p21) — French-Canadian Leigh variant; liver + brain\n\n"
                "All share: isolated COX deficiency, Leigh-like MRI, lactic acidosis, VPA CI.\n"
                "Distinguish by: organ-specific secondary features (cardiac/renal/hepatic)."
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "Leigh Syndrome — Bilateral Symmetric Basal Ganglia/Brainstem Necrosis (COX15 Context)",
            "definition": (
                "Leigh syndrome (subacute necrotising encephalomyelopathy, OMIM #256000) is "
                "defined radiologically by:\n"
                "  • Bilateral, symmetric T2/FLAIR signal abnormalities\n"
                "  • Putamen, caudate, globus pallidus, brainstem tegmentum — any combination\n"
                "  • Spongiform degeneration, capillary proliferation, neuronal loss on pathology\n\n"
                "In COX15:\n"
                "  • Leigh MRI present in ~80% — similar to COX10 (~88%) and SCO2 (~55%)\n"
                "  • Death before advanced Leigh on MRI may occur due to cardiac failure first\n"
                "  • Post-mortem shows classical Leigh neuropathology\n"
                "  • CSF lactate elevated (>2.1 mmol/L) in >85%\n\n"
                "Metabolic crisis triggers in COX15:\n"
                "  • Febrile illness (most common — increases metabolic rate; COX demand ↑)\n"
                "  • Anaesthesia/surgery (especially with propofol — ABSOLUTE CI)\n"
                "  • Fasting (releases fatty acids → FADH2 overload in COX-deficient cell)\n"
                "  • Cardiac decompensation (HCM → reduced systemic perfusion → cerebral "
                "    hypoxia superimposed on OXPHOS failure)\n\n"
                "Emergency management:\n"
                "  1. IV dextrose GIR 6-8 mg/kg/min STAT\n"
                "  2. IV NaHCO3 if pH <7.2 (target >7.25)\n"
                "  3. IV LEV if seizures (AVOID VPA)\n"
                "  4. Cardiac monitoring (echocardiogram if HCM deterioration suspected)\n"
                "  5. NEVER use propofol for sedation or anaesthesia\n"
                "  6. NEVER fast — maintain continuous enteral/parenteral glucose"
            ),
        },
        {
            "term": "HCM Management in COX15 — Beta-Blockers, Echocardiographic Monitoring, Cardiac Transplant",
            "definition": (
                "Cardiac management strategy for COX15 HCM:\n\n"
                "FIRST LINE — Beta-Blockade:\n"
                "  Propranolol: 0.5-2 mg/kg/day in 2-3 divided doses\n"
                "  Atenolol: alternative (once-daily, cardioselective)\n"
                "  Mechanism: reduces HR → increases diastolic filling time → reduces LVOT "
                "  gradient; reduces myocardial O2 demand (critical in COX-deficient heart)\n\n"
                "MONITORING:\n"
                "  Echocardiogram: every 3-6 months (baseline + serial wall thickness Z-score)\n"
                "  ECG/Holter: annually (risk of life-threatening arrhythmia)\n"
                "  BNP/NT-proBNP: cardiac failure surrogate\n\n"
                "AVOID:\n"
                "  • Digoxin (ABSOLUTE CI — worsens LVOT obstruction)\n"
                "  • ACEi/ARB in obstructive phase (reduce preload → worsens obstruction)\n"
                "  • Nifedipine/dihydropyridine CCBs (vasodilation → LVOT worsening)\n"
                "  • Verapamil (negative inotrope + vasodilator — may use cautiously for "
                "    rate control if propranolol fails, but cardiology input required)\n\n"
                "LATE/SEVERE:\n"
                "  • ICD if high risk for sudden cardiac death (SCD) — multidisciplinary decision\n"
                "  • Cardiac transplant: life-extending for cardiac failure; does NOT cure "
                "    encephalopathy. Ethical deliberation with family mandatory before listing.\n"
                "  • Palliative pathway appropriate for severe neurological + cardiac disease"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "LEV (Levetiracetam) — Preferred AED in COX15 Disease",
            "definition": (
                "Levetiracetam (LEV) is the AED of choice in COX15 disease:\n\n"
                "  1. RENAL excretion: 66% excreted unchanged in urine; no hepatic CYP metabolism\n"
                "     → No hepatotoxic metabolites; no CYP induction; no interference with "
                "       mitochondrial metabolism\n"
                "  2. No mitochondrial toxicity: LEV does not inhibit any respiratory complex\n"
                "  3. CARDIAC-SAFE: LEV does not affect cardiac conduction, heart rate, or "
                "     contractility — critical in COX15 patients with HCM\n"
                "  4. IV formulation: essential during Leigh crisis when oral route compromised\n"
                "  5. Broad-spectrum efficacy: focal, myoclonic, generalised tonic-clonic\n\n"
                "Dosing: 20-60 mg/kg/day in 2 divided doses (paediatric); IV loading 20-40 mg/kg "
                "over 15 min in acute seizure. Adjust for renal impairment (rare in COX15 unless "
                "tubulopathy develops).\n\n"
                "AVOID: VPA (ABSOLUTE CI — CoA sequestration, POLG inhibition, hepatotoxicity); "
                "phenobarbital (Complex I inhibitor); carbamazepine (cardiac conduction effects "
                "potentially dangerous in HCM arrhythmia); phenytoin (cardiac Class IB effect)"
            ),
        },
        {
            "term": "Beta-Blockers in COX15 HCM vs VPA Interaction",
            "definition": (
                "Beta-blockers are the cornerstone of HCM management in COX15 disease:\n\n"
                "Propranolol (non-selective beta-blocker):\n"
                "  • Reduces LVOT gradient by decreasing outflow velocity\n"
                "  • Reduces myocardial O2 demand (lower HR × less wall tension)\n"
                "  • No mitochondrial toxicity; hepatically metabolised (CYP2D6/2C18)\n"
                "  • Dose: 0.5-2 mg/kg/day; titrate to resting HR 60-80 bpm\n"
                "  • MONITORING: BG (may mask hypoglycaemia — critical in COX15 fasting\n"
                "    risk; ensure continuous glucose supply)\n\n"
                "Atenolol (cardioselective beta-1):\n"
                "  • Renally excreted — advantage in hepatic impairment\n"
                "  • Once-daily dosing improves adherence\n"
                "  • Less bronchospasm risk (cardioselective)\n\n"
                "IMPORTANT INTERACTION WARNING:\n"
                "  VPA is ABSOLUTELY CONTRAINDICATED in COX15 regardless of HCM indication.\n"
                "  Some practitioners may consider VPA for seizures in cardiomyopathy context — "
                "  this is FORBIDDEN in COX15 (any COX-deficiency disease). VPA's CoA "
                "  sequestration compounds OXPHOS failure; hepatotoxicity risk amplified. "
                "  LEV is the safe AED alternative."
            ),
        },
        {
            "term": "IV Dextrose (GIR 6-8) + NEVER FAST — Energy Management in COX15 Crisis",
            "definition": (
                "In COX15 disease, OXPHOS is severely deficient. Preferred energy management:\n\n"
                "  Glucose infusion rate (GIR): 6-8 mg/kg/min\n"
                "  Rationale:\n"
                "  • Glucose → pyruvate → partial ATP via glycolysis even without OXPHOS\n"
                "  • Avoids fasting-induced lipolysis and fatty acid mobilisation\n"
                "    (beta-oxidation requires FADH2 reoxidation at Complex II → III → IV;\n"
                "    COX deficiency blocks this → acylcarnitine accumulation + lactic surge)\n"
                "  • Prevents catabolic-driven lactate surge during Leigh crisis\n\n"
                "NEVER FAST a COX15 patient:\n"
                "  Even perioperative fasting must be minimised:\n"
                "  • Oral glucose polymer until 2h pre-procedure\n"
                "  • IV dextrose commenced immediately on NBM (GIR 6-8)\n"
                "  • Restart feeds immediately post-operatively\n\n"
                "Monitoring during IV dextrose:\n"
                "  • Blood glucose Q2-4h (target 4-8 mmol/L; avoid hyperglycaemia)\n"
                "  • Plasma lactate Q4-8h (target <5 mmol/L during stabilisation)\n"
                "  • Electrolytes (sodium, potassium, phosphate)\n"
                "  • Cardiac monitoring (ECG, echo if HCM decompensation)\n\n"
                "Emergency dextrose bolus (hypoglycaemia):\n"
                "  2 mL/kg of 10% dextrose IV over 5 min; then GIR 6-8 maintenance"
            ),
        },
        {
            "term": "CoQ10/Ubiquinol, Riboflavin, Thiamine, Biotin — Cofactor Therapy in COX15",
            "definition": (
                "Standard mitochondrial cofactor therapy in COX15 disease (all Level C evidence):\n\n"
                "  CoQ10 / Ubiquinol:\n"
                "  • 300-600 mg/day adults; 10-30 mg/kg/day children\n"
                "  • Mobile electron carrier Complex I/II → III; may increase spare electron flux\n"
                "  • Ubiquinol (reduced form) preferred — better oral bioavailability\n\n"
                "  Riboflavin (B2): 100-400 mg/day\n"
                "  • Cofactor for Complex I (FMN) and Complex II (FAD); residual OXPHOS support\n\n"
                "  Thiamine (B1): 100-300 mg/day — MANDATORY empirically in all Leigh\n"
                "  • Pyruvate dehydrogenase (PDH) and alpha-ketoglutarate dehydrogenase cofactor\n"
                "  • SLC19A3 (THTR2) thiamine transporter deficiency and BTD — TREATABLE Leigh "
                "    mimics; thiamine/biotin empiric until molecular confirmation\n\n"
                "  Biotin: 5-20 mg/day empirically\n"
                "  • Biotinidase deficiency (BTD) is a TREATABLE Leigh mimic — give empirically\n\n"
                "  Carnitine: 50-100 mg/kg/day\n"
                "  • Secondary carnitine deficiency common in OXPHOS diseases\n"
                "  • Supports fatty acid clearance (though FA oxidation limited by COX deficiency)\n\n"
                "  All cofactors discontinued if no clinical benefit after 3-6 month trial; "
                "  never delay treatment of reversible mimics (SLC19A3, BTD) for cofactor trial."
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
