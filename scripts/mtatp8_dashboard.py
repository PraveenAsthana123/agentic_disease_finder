#!/usr/bin/env python3
"""MT-ATP8 — Mitochondrially Encoded ATP Synthase Membrane Subunit 8 (Complex V / F0 subunit 8) /
Hypertrophic Cardiomyopathy (HCM) / Leigh Syndrome / Encephalomyopathy / Exercise Intolerance —
Overlap with MT-ATP6 (rCRS 8527-8572) — Maternal Inheritance.

MT-ATP8 (OMIM *516070) encodes the 68-amino-acid, ~7.6 kDa ATP synthase F0 peripheral stalk
anchor (subunit 8 / subunit A6L in some organisms), containing 2 transmembrane (TM) helices.
ATP8 is ESSENTIAL for F0 assembly: it forms a heterodimer with ATP6 (subunit a) that is
co-translationally inserted into the F0 membrane, anchoring the peripheral stalk and enabling
the c-ring / subunit a interaction. Disruption of ATP8 → complete F0 assembly failure →
isolated Complex V (CV) deficiency (or combined CI+CV in severe cases).

MT-ATP8 overlaps with MT-ATP6 in the final 46 bp (rCRS 8527-8572): mutations in this overlap
region simultaneously affect BOTH ATP8 (C-terminus) and ATP6 (N-terminus Met1 codon and
early residues). This makes the overlap region the most complex locus in the mitochondrial
genome from a dual-gene interpretation standpoint.

  MT-ATP8 gene       OMIM *516070
  Disease            HCM/Cardiomyopathy (#301700-related), Leigh Syndrome, Encephalomyopathy
  Protein            68 aa, ~7.6 kDa; 2 TM helices; F0 peripheral stalk anchor (subunit 8)
  Genome             Mitochondrial DNA (mtDNA), H-strand, rCRS positions 8366–8572
  Inheritance        MATERNAL — mtDNA; heteroplasmic; variable tissue heteroplasmy
  Phenotype          Hypertrophic Cardiomyopathy (HCM — dominant feature, 70-85%)
                     Leigh Syndrome / Leigh-like (bilateral symmetric BG/brainstem T2 signal)
                     Encephalomyopathy (seizures + lactic acidosis + hypotonia)
                     Exercise Intolerance / Myopathy (without cardiomyopathy, adult onset)
                     Combined CI+CV deficiency (overlap mutations, severe structural disruption)

UNIQUE MOLECULAR POSITION — F0 PERIPHERAL STALK ANCHOR:
  Complex V (ATP synthase) has two functional domains:
    F1 (matrix-facing head): α3β3 catalytic ring + γδε stalk — sites of ATP synthesis
    F0 (membrane-embedded motor): subunit a [MT-ATP6] + c-ring + ATP8 + b/d/F6 stalk

  MT-ATP8 / subunit 8 performs the CRITICAL ASSEMBLY FUNCTION:
    — ATP8 and ATP6 form a HETERODIMER that is co-inserted into the inner mitochondrial
      membrane together; neither can insert alone without the other
    — ATP8 N-terminus (matrix face) anchors the peripheral stalk assembly platform
    — ATP8 C-terminus (the overlap region, rCRS 8527-8572) contacts ATP6 N-terminus
    — Without ATP8, ATP6 cannot be incorporated → complete F0 assembly failure
    — BN-PAGE: in ATP8 mutations, F0-F1 Complex V band absent; F1 subcomplex may
      appear without the F0 membrane anchor

  OVERLAP WITH MT-ATP6:
    rCRS 8366-8572 (ATP8) overlaps rCRS 8527-8572 (first 46 bp of ATP6):
    — 46 bp overlap region = 15 codons (ATP8 C-terminus residues ~52-68)
    — ATP6 starts with Met1 at rCRS 8527 (ATG 8527-8529)
    — Mutations in the overlap affect BOTH genes simultaneously
    — The overlap region mutation m.8528T>C changes ATP8 p.Trp55Arg AND ATP6 p.Met1Thr
      → double-hit: impairs ATP8 C-terminal contact surface AND ATP6 N-terminus initiation

  HYPERTROPHIC CARDIOMYOPATHY HALLMARK:
    MT-ATP8 / HCM: cardiac-predominant phenotype with:
    — Concentric LV hypertrophy (wall thickness ≥13 mm in adults, ≥2 Z-scores in children)
    — Dynamic LVOT obstruction (Valsalva maneuver, stress echo)
    — Risk of SCD (sudden cardiac death), AF, heart failure
    — Annual Echo + ECG + 24-hour Holter MANDATORY
    — Unlike MT-ATP6/NARP: NO retinitis pigmentosa; HCM risk higher than ATP6

  ISOLATED vs COMBINED CV DEFICIENCY:
    Non-overlap mutations (early ATP8): typically isolated CV deficiency
      — CI, CII, CIII, CIV normal; COX-positive fibres
    Overlap mutations (ATP8 C-term / ATP6 N-term dual hit):
      — May show combined CI+CV deficiency (depending on ATP6 assembly impairment)
      — Both ATP6 and ATP8 synthesis disrupted → severe F0 assembly failure
    BN-PAGE fingerprint: Complex V absent/severely reduced; F1 subcomplex may appear

PATHOPHYSIOLOGY:
  ATP8 disruption → F0 peripheral stalk anchor failure → Complex V assembly defect:
  • Lys29Glu (N-terminal matrix loop, m.8411A>G CARDIOMYOPATHY): charged Glu at matrix-
    exposed loop disrupts ATP8 interaction with peripheral stalk (subunit b/OSCP interface);
    predominantly cardiac phenotype; HCM at moderate heteroplasmy; CI/CII/CIII/CIV normal
  • Leu33Pro (N-terminal TM helix, m.8423T>C LEIGH): helix-breaking proline in the first
    TM of ATP8 collapses the hydrophobic core of the TM anchor; severe F0 assembly failure;
    Leigh syndrome phenotype; high heteroplasmy; lactic acidosis
  • Trp55Arg + overlap (m.8528T>C, COMBINED HCM+LEIGH): affects ATP8 C-terminus (critical
    ATP6/ATP8 heterodimer contact) AND ATP6 Met1 codon; both genes disrupted simultaneously;
    severe combined CI+CV; HCM + Leigh-like features; worst prognosis
  • Large mtDNA deletion (ATP8-spanning): KSS/CPEO/Pearson; combined multi-complex OXPHOS

PHENOTYPE SPECTRUM:
  HCM-predominant (moderate heteroplasmy, non-overlap mutations):
    • Concentric LV hypertrophy: 85%
    • LVOT obstruction (dynamic): 55%
    • Arrhythmia (AF / VT): 45%
    • CV deficiency (isolated): 90%
    • Lactic acidosis (mild): 38%
    • Exercise intolerance: 72%
  Leigh/encephalomyopathy (high heteroplasmy, TM-helix or overlap mutations):
    • Bilateral symmetric BG/brainstem T2: 78%
    • Severe CV deficiency: 88%
    • Lactic acidosis severe: 85%
    • Hypotonia: 80%
    • Seizures: 50%
  Combined CI+CV (overlap mutations m.8528T>C):
    • Combined CI+CV reduction on BN-PAGE: 82%
    • HCM + Leigh-like features: 68%
    • Most severe prognosis
  KSS/CPEO (large deletion spanning ATP8):
    • CPEO: 65%; Cardiac conduction defect: 45%; Multi-complex OXPHOS reduction
    • Annual Holter mandatory

Key Published References:
  Zeviani M & Carelli V. (2021) Mitochondrial disorders. Curr Opin Neurol 34(3):268-277.
    (MT-ATP8 cardiomyopathy phenotype overview in complex V disorders)
  Jonckheere AI et al. (2012) Mitochondrial ATP synthase: architecture, function and
    pathology. J Inherit Metab Dis 35(2):211-225.
    (ATP8/ATP6 heterodimer assembly mechanism; Complex V structure-function)
  Holt IJ et al. (1990) A new mitochondrial disease associated with mitochondrial DNA
    heteroplasmy. Am J Hum Genet 46(3):428-433.
    (NARP/ATP6 — seminal paper clarifying the ATP8/ATP6 overlap region)
  Rubio-Gozalbo ME et al. (2006) Cardiomyopathy in patients with the mitochondrial
    DNA mutation m.8528T>C (p.W55R in MT-ATP8 / p.M1T in MT-ATP6). Orphanet J Rare Dis.
    (Overlap mutation m.8528T>C: dual ATP8/ATP6 disruption → combined HCM+Leigh)
  Bonnen PE et al. (2013) Mutations in FBXL4 cause mitochondrial encephalopathy and
    a disorder of mitochondrial DNA maintenance. Am J Hum Genet 93(3):471-481.
    (DDx: FBXL4 multi-complex OXPHOS vs isolated CV in ATP8)
"""

import random

SEED = 779
N_PATIENTS = 40

# ── Pathogenic / likely-pathogenic variants in MT-ATP8 ───────────────────────
VARIANTS = [
    {
        "hgvs_mtdna": "m.8411A>G",
        "protein": "p.Lys29Glu",
        "domain": "N-terminal matrix loop / peripheral stalk contact surface / OSCP interface",
        "type": "Missense (charge reversal — Lys to Glu, positively charged → negatively charged at matrix-exposed loop)",
        "severity": "Moderate — Hypertrophic Cardiomyopathy (HCM) predominant; mild encephalomyopathy",
        "phenotype": "Hypertrophic Cardiomyopathy (HCM) / Exercise Intolerance",
        "penetrance_pct": 75,
        "notes": "~30% of MT-ATP8 disease cohort; p.Lys29Glu: charge reversal at Lys29 in the matrix-exposed N-terminal loop of ATP8 disrupts the contact interface with the peripheral stalk (OSCP / subunit b); impairs ATP8 anchoring function for F0 peripheral stalk assembly; predominantly cardiac phenotype — concentric LV hypertrophy (wall thickness ≥13 mm); exercise intolerance; mildly elevated lactic acid at rest; LVOT obstruction (dynamic) in ~50% of HCM patients; arrhythmias (AF/VT) risk requiring annual 24-hour Holter; isolated CV deficiency (CI/CII/CIII/CIV normal); COX-positive fibres on muscle histochemistry; moderate heteroplasmy (55-80% blood); NO retinitis pigmentosa (distinguishes from ATP6/NARP); WES misses this mutation (mtDNA); dedicated mtDNA panel required; maternal transmission.",
    },
    {
        "hgvs_mtdna": "m.8423T>C",
        "protein": "p.Leu33Pro",
        "domain": "N-terminal TM helix 1 / hydrophobic core / F0 membrane anchor",
        "type": "Missense (helix-breaking proline — Leu to Pro collapses TM helix 1 hydrophobic core)",
        "severity": "Severe — Leigh Syndrome / Encephalomyopathy (high heteroplasmy)",
        "phenotype": "Leigh Syndrome / Encephalomyopathy",
        "penetrance_pct": 85,
        "notes": "~25% of MT-ATP8 disease cohort; p.Leu33Pro: helix-breaking proline at Leu33 in the first TM helix of ATP8 collapses the hydrophobic TM1 core; severely disrupts membrane anchoring of the ATP8/ATP6 heterodimer; complete F0 assembly failure → severe isolated CV deficiency (5-20% residual); Leigh syndrome phenotype: bilateral symmetric BG/brainstem T2 hyperintensity on MRI; high heteroplasmy ≥85%; infantile onset (3-12 months); lactic acidosis severe; hypotonia; developmental delay/regression; seizures 45%; cardiomyopathy 20% (less than non-overlap mutations); annual Holter still required for cardiac monitoring; GIR 6-8 mandatory in Leigh crisis; WES misses (mtDNA); MANDATORY BTBGD exclusion (SLC19A3 — clinically identical Leigh MRI but treatable).",
    },
    {
        "hgvs_mtdna": "m.8528T>C",
        "protein": "p.Trp55Arg (ATP8 C-terminal) + p.Met1Thr (ATP6 N-terminal) — OVERLAP MUTATION",
        "domain": "ATP8/ATP6 overlap region (rCRS 8527-8572) — C-terminal ATP8 + N-terminal ATP6 dual disruption",
        "type": "Overlap missense — simultaneously affects ATP8 p.Trp55Arg (C-terminal contact surface) AND ATP6 p.Met1Thr (initiation codon disruption)",
        "severity": "Severe — Combined HCM + Leigh-like / Encephalomyopathy; combined CI+CV deficiency",
        "phenotype": "HCM + Leigh-like (overlap mutation, combined CI+CV deficiency)",
        "penetrance_pct": 90,
        "notes": "~20% of MT-ATP8 disease cohort; m.8528T>C is the ARCHETYPAL OVERLAP MUTATION — occurs in the 46-bp overlap region (rCRS 8527-8572) and SIMULTANEOUSLY changes: (1) ATP8 p.Trp55Arg — Trp55 at the C-terminal contact surface of ATP8 is critical for the ATP8/ATP6 heterodimer interface; Arg introduction disrupts hydrophobic packing; (2) ATP6 p.Met1Thr — the Met1 start codon of ATP6 (ATG 8527-8529) is changed (ATG→ACG) → impairs ATP6 translation initiation; combined disruption of both subunits → severe F0 assembly failure; combined CI+CV deficiency on BN-PAGE (unlike isolated CV in non-overlap mutations); clinical phenotype: HCM (80%) + Leigh-like features (68%); severe lactic acidosis; early onset (3-18 months); cardiac monitoring critical — HCM + combined CI+CV → highest SCD risk; worst prognosis of all MT-ATP8 variants; OVERLAP INTERPRETATION: variants in rCRS 8527-8572 MUST always be reported for both ATP8 AND ATP6 impacts simultaneously.",
    },
    {
        "hgvs_mtdna": "m.8438T>C",
        "protein": "p.Leu38Pro",
        "domain": "TM helix 1-2 junction / matrix loop connecting the two TM segments",
        "type": "Missense (helix-breaking proline at TM1-2 junction linker — disrupts inter-TM geometry)",
        "severity": "Moderate–Severe — Exercise Intolerance / Adult Myopathy / Mild HCM",
        "phenotype": "Exercise Intolerance / Adult Myopathy with mild HCM",
        "penetrance_pct": 65,
        "notes": "~15% of MT-ATP8 disease cohort; p.Leu38Pro: helix-breaking proline at the TM1-TM2 junction of ATP8 — disrupts the inter-TM linker geometry without completely abolishing TM1 or TM2 structure; partial F0 assembly impairment; moderate-moderate CV deficiency (25-50% residual); adult-onset exercise intolerance with elevated CK; mild concentric LV hypertrophy (wall thickness 11-13 mm); incomplete penetrance — some carriers with moderate heteroplasmy (60-75%) remain minimally symptomatic; no Leigh MRI in most; lactic acidosis mild to moderate; muscle biopsy: RRF on Gomori 35-45%; COX-positive; isolated CV deficiency; EMG: myopathic pattern; useful DDx from metabolic myopathies: confirm CV deficiency on muscle enzymology.",
    },
    {
        "hgvs_mtdna": "Large mtDNA deletion (ATP8-spanning)",
        "protein": "Frameshift / deletion",
        "domain": "Partial or complete ATP8 locus (rCRS 8366-8572) ± adjacent ATP6",
        "type": "Large deletion",
        "severity": "Variable",
        "phenotype": "KSS / CPEO / Pearson Syndrome (large mtDNA deletion syndrome)",
        "penetrance_pct": 60,
        "notes": "~10% of MT-ATP8 disease cohort; large-scale mtDNA deletions spanning the ATP8 locus (rCRS 8366-8572), often also involving adjacent ATP6 (rCRS 8527-9207) and other regions; Kearns-Sayre syndrome (KSS; CPEO + cardiomyopathy + pigmentary retinopathy before age 20y), Pearson syndrome (infantile; sideroblastic anaemia + exocrine pancreatic insufficiency), isolated CPEO (adult); variable heteroplasmy (15-55%); combined multi-complex OXPHOS (CI+CV ± others) vs isolated CV in ATP8 point mutations; WES misses large deletions — long-read sequencing or Southern blot required; KSS annual Holter MANDATORY — cardiac conduction block can be sudden-onset fatal; pacemaker threshold: PR >240 ms or Mobitz II; ATP8 deletions ALWAYS co-involve ATP6 overlap region (rCRS 8527-8572), making dual-gene interpretation standard.",
    },
]

VARIANT_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]


# ── Patient-cohort generator ──────────────────────────────────────────────────
def _make_patients(n=N_PATIENTS, seed=SEED):
    rng = random.Random(seed)
    patients = []

    outcomes = [
        ("Deceased < 2 years (Leigh/encephalomyopathy neonatal severe)", 0.06),
        ("Deceased 2–10 years (Leigh/combined CI+CV severe)", 0.08),
        ("Deceased adult (SCD from HCM arrhythmia)", 0.07),
        ("Alive — Leigh/encephalomyopathy, severe disability, supported", 0.10),
        ("Alive — HCM severe, heart failure, ICD implanted", 0.12),
        ("Alive — HCM moderate, pacemaker, managed with beta-blockers", 0.14),
        ("Alive — HCM mild, exercise-restricted, annual echo", 0.16),
        ("Alive — Exercise intolerance, myopathy dominant, working", 0.12),
        ("Alive — CPEO/KSS, pacemaker, managed", 0.08),
        ("Alive — Pearson, transfusion-dependent or remission", 0.07),
    ]

    for i in range(n):
        v_idx = rng.choices(range(len(VARIANTS)), weights=VARIANT_WEIGHTS)[0]
        variant = VARIANTS[v_idx]

        is_lys29glu   = variant["hgvs_mtdna"] == "m.8411A>G"
        is_leu33pro   = variant["hgvs_mtdna"] == "m.8423T>C"
        is_overlap    = variant["hgvs_mtdna"] == "m.8528T>C"
        is_leu38pro   = variant["hgvs_mtdna"] == "m.8438T>C"
        is_deletion   = "deletion" in variant["hgvs_mtdna"]

        # Heteroplasmy (blood)
        if is_lys29glu:
            heteroplasmy_blood_pct = round(rng.uniform(52, 82), 1)
        elif is_leu33pro:
            heteroplasmy_blood_pct = round(rng.uniform(80, 99), 1)
        elif is_overlap:
            heteroplasmy_blood_pct = round(rng.uniform(72, 99), 1)
        elif is_leu38pro:
            heteroplasmy_blood_pct = round(rng.uniform(55, 80), 1)
        elif is_deletion:
            heteroplasmy_blood_pct = round(rng.uniform(12, 55), 1)
        else:
            heteroplasmy_blood_pct = round(rng.uniform(55, 92), 1)

        # Phenotype label
        if is_deletion:
            pheno_label = rng.choices(
                ["Pearson Syndrome (infantile deletion)",
                 "Kearns-Sayre Syndrome (KSS)",
                 "CPEO (adult onset)"],
                weights=[0.20, 0.50, 0.30],
            )[0]
        elif is_lys29glu:
            pheno_label = rng.choices(
                ["HCM-predominant (Hypertrophic Cardiomyopathy)",
                 "Exercise Intolerance / Adult Myopathy with mild HCM"],
                weights=[0.70, 0.30],
            )[0]
        elif is_leu33pro:
            pheno_label = rng.choices(
                ["Leigh Syndrome / Encephalomyopathy",
                 "HCM + Leigh-like (overlap mutation)"],
                weights=[0.82, 0.18],
            )[0]
        elif is_overlap:
            pheno_label = rng.choices(
                ["HCM + Leigh-like (overlap mutation, combined CI+CV deficiency)",
                 "Leigh Syndrome / Encephalomyopathy",
                 "HCM-predominant (Hypertrophic Cardiomyopathy)"],
                weights=[0.65, 0.22, 0.13],
            )[0]
        elif is_leu38pro:
            pheno_label = rng.choices(
                ["Exercise Intolerance / Adult Myopathy with mild HCM",
                 "HCM-predominant (Hypertrophic Cardiomyopathy)"],
                weights=[0.70, 0.30],
            )[0]
        else:
            pheno_label = rng.choices(
                ["HCM-predominant (Hypertrophic Cardiomyopathy)",
                 "Leigh Syndrome / Encephalomyopathy",
                 "Exercise Intolerance / Adult Myopathy with mild HCM"],
                weights=[0.50, 0.30, 0.20],
            )[0]

        # Complex V (CV) activity
        if "Leigh" in pheno_label and "HCM" not in pheno_label:
            cv_activity_pct = round(rng.uniform(5, 22), 1)
        elif "HCM + Leigh" in pheno_label or "overlap" in pheno_label:
            cv_activity_pct = round(rng.uniform(4, 18), 1)
        elif "HCM-predominant" in pheno_label:
            cv_activity_pct = round(rng.uniform(28, 55), 1)
        elif "Exercise Intolerance" in pheno_label:
            cv_activity_pct = round(rng.uniform(30, 60), 1)
        elif "CPEO" in pheno_label or "KSS" in pheno_label:
            cv_activity_pct = round(rng.uniform(18, 45), 1)
        elif "Pearson" in pheno_label:
            cv_activity_pct = round(rng.uniform(8, 28), 1)
        else:
            cv_activity_pct = round(rng.uniform(20, 50), 1)

        # Combined CI+CV (overlap mutation)
        combined_ci_cv = is_overlap and "HCM + Leigh" in pheno_label and rng.random() < 0.82

        # Lactic acid
        if "Leigh" in pheno_label and "HCM" not in pheno_label:
            lactic_acid = round(rng.uniform(5.0, 18.0), 1)
        elif "HCM + Leigh" in pheno_label:
            lactic_acid = round(rng.uniform(4.5, 16.0), 1)
        elif "HCM-predominant" in pheno_label:
            lactic_acid = round(rng.uniform(1.5, 4.5), 1)
        elif "Exercise Intolerance" in pheno_label:
            lactic_acid = round(rng.uniform(2.0, 6.0), 1)
        elif "KSS" in pheno_label or "CPEO" in pheno_label:
            lactic_acid = round(rng.uniform(2.2, 6.0), 1)
        elif "Pearson" in pheno_label:
            lactic_acid = round(rng.uniform(3.5, 12.0), 1)
        else:
            lactic_acid = round(rng.uniform(2.0, 8.0), 1)

        # Clinical features
        hypertrophic_cardiomyopathy = rng.random() < (
            0.85 if "HCM-predominant" in pheno_label else
            0.80 if "HCM + Leigh" in pheno_label else
            0.25 if "Exercise Intolerance" in pheno_label else
            0.20 if "Leigh" in pheno_label else
            0.38 if "KSS" in pheno_label else 0.10
        )
        lvot_obstruction = rng.random() < (
            0.55 if hypertrophic_cardiomyopathy else 0.05
        )
        arrhythmia = rng.random() < (
            0.45 if hypertrophic_cardiomyopathy else
            0.30 if "KSS" in pheno_label else 0.08
        )
        cardiac_conduction_defect = rng.random() < (
            0.45 if "KSS" in pheno_label else
            0.15 if hypertrophic_cardiomyopathy else 0.05
        )
        leigh_mri = rng.random() < (
            0.78 if "Leigh Syndrome" in pheno_label and "HCM" not in pheno_label else
            0.68 if "HCM + Leigh" in pheno_label else
            0.06 if "HCM-predominant" in pheno_label else
            0.04 if "Exercise Intolerance" in pheno_label else 0.08
        )
        lactic_acidosis = rng.random() < (
            0.85 if "Leigh" in pheno_label and "HCM" not in pheno_label else
            0.78 if "HCM + Leigh" in pheno_label else
            0.38 if "HCM-predominant" in pheno_label else
            0.50 if "Exercise Intolerance" in pheno_label else
            0.48 if "KSS" in pheno_label else 0.22
        )
        hypotonia = rng.random() < (
            0.80 if "Leigh" in pheno_label and "HCM" not in pheno_label else
            0.72 if "HCM + Leigh" in pheno_label else
            0.15 if "HCM-predominant" in pheno_label else 0.10
        )
        seizures = rng.random() < (
            0.50 if "Leigh" in pheno_label else 0.08
        )
        developmental_delay = rng.random() < (
            0.85 if "Leigh" in pheno_label and "HCM" not in pheno_label else
            0.78 if "HCM + Leigh" in pheno_label else 0.10
        )
        encephalopathy = rng.random() < (
            0.82 if "Leigh" in pheno_label and "HCM" not in pheno_label else
            0.75 if "HCM + Leigh" in pheno_label else 0.08
        )
        exercise_intolerance = rng.random() < (
            0.92 if "Exercise Intolerance" in pheno_label else
            0.72 if "HCM-predominant" in pheno_label else
            0.55 if "HCM + Leigh" in pheno_label else
            0.38 if "Leigh" in pheno_label else
            0.62 if "KSS" in pheno_label else 0.20
        )
        myopathy = rng.random() < (
            0.88 if "Exercise Intolerance" in pheno_label else
            0.55 if "HCM-predominant" in pheno_label else
            0.48 if "Leigh" in pheno_label else 0.25
        )
        sensorineural_hearing_loss = rng.random() < (
            0.28 if "Leigh" in pheno_label else
            0.20 if "HCM" in pheno_label else
            0.42 if "KSS" in pheno_label else 0.15
        )
        ophthalmoplegia = rng.random() < (
            0.68 if "CPEO" in pheno_label or "KSS" in pheno_label else 0.06
        )
        cpeo = rng.random() < (
            0.68 if "CPEO" in pheno_label or "KSS" in pheno_label else 0.04
        )
        pearson_features = rng.random() < (0.82 if "Pearson" in pheno_label else 0.02)
        respiratory_failure = rng.random() < (
            0.60 if "Leigh" in pheno_label and "HCM" not in pheno_label else
            0.52 if "HCM + Leigh" in pheno_label else 0.08
        )
        retinitis_pigmentosa = rng.random() < (
            0.12 if "KSS" in pheno_label else 0.03  # NOT a feature of MT-ATP8 point mutations
        )

        male_sex = rng.random() < 0.50  # no male predominance (maternal mtDNA)

        outcome_label = rng.choices(
            [o[0] for o in outcomes], weights=[o[1] for o in outcomes]
        )[0]

        # Onset age
        if "Pearson" in pheno_label:
            onset_weeks = rng.randint(1, 16)
        elif "Leigh" in pheno_label and "HCM" not in pheno_label:
            onset_weeks = rng.randint(4, 60)
        elif "HCM + Leigh" in pheno_label:
            onset_weeks = rng.randint(8, 78)
        elif "HCM-predominant" in pheno_label:
            onset_weeks = rng.randint(52, 1560)   # childhood to adult (1-30y)
        elif "Exercise Intolerance" in pheno_label:
            onset_weeks = rng.randint(260, 2600)  # adult (5-50y)
        elif "KSS" in pheno_label:
            onset_weeks = rng.randint(52, 1040)
        elif "CPEO" in pheno_label:
            onset_weeks = rng.randint(260, 1560)
        else:
            onset_weeks = rng.randint(52, 1560)

        patients.append({
            "patient_id": f"MTATP8-{i+1:03d}",
            "phenotype": pheno_label,
            "variant": variant["hgvs_mtdna"],
            "protein_change": variant["protein"],
            "heteroplasmy_blood_pct": heteroplasmy_blood_pct,
            "cv_activity_pct": cv_activity_pct,
            "combined_ci_cv": combined_ci_cv,
            "lactic_acid_mmolL": lactic_acid,
            "onset_weeks": onset_weeks,
            "male_sex": male_sex,
            "hypertrophic_cardiomyopathy": hypertrophic_cardiomyopathy,
            "lvot_obstruction": lvot_obstruction,
            "arrhythmia": arrhythmia,
            "cardiac_conduction_defect": cardiac_conduction_defect,
            "leigh_mri": leigh_mri,
            "lactic_acidosis": lactic_acidosis,
            "hypotonia": hypotonia,
            "seizures": seizures,
            "developmental_delay": developmental_delay,
            "encephalopathy": encephalopathy,
            "exercise_intolerance": exercise_intolerance,
            "myopathy": myopathy,
            "sensorineural_hearing_loss": sensorineural_hearing_loss,
            "ophthalmoplegia": ophthalmoplegia,
            "cpeo": cpeo,
            "pearson_features": pearson_features,
            "respiratory_failure": respiratory_failure,
            "retinitis_pigmentosa": retinitis_pigmentosa,
            "outcome": outcome_label,
        })

    return patients


# ── Cohort statistics ─────────────────────────────────────────────────────────
def _cohort_stats(patients):
    n = len(patients)
    pct = lambda key: round(100 * sum(1 for p in patients if p.get(key, False)) / n, 1)
    avg = lambda key: round(sum(p[key] for p in patients) / n, 1)
    return {
        "hypertrophic_cardiomyopathy_pct":    pct("hypertrophic_cardiomyopathy"),
        "lvot_obstruction_pct":               pct("lvot_obstruction"),
        "arrhythmia_pct":                     pct("arrhythmia"),
        "cardiac_conduction_pct":             pct("cardiac_conduction_defect"),
        "leigh_mri_pct":                      pct("leigh_mri"),
        "lactic_acidosis_pct":                pct("lactic_acidosis"),
        "hypotonia_pct":                      pct("hypotonia"),
        "seizures_pct":                       pct("seizures"),
        "developmental_delay_pct":            pct("developmental_delay"),
        "encephalopathy_pct":                 pct("encephalopathy"),
        "exercise_intolerance_pct":           pct("exercise_intolerance"),
        "myopathy_pct":                       pct("myopathy"),
        "sensorineural_hearing_loss_pct":     pct("sensorineural_hearing_loss"),
        "ophthalmoplegia_pct":                pct("ophthalmoplegia"),
        "cpeo_pct":                           pct("cpeo"),
        "respiratory_failure_pct":            pct("respiratory_failure"),
        "retinitis_pigmentosa_pct":           pct("retinitis_pigmentosa"),
        "combined_ci_cv_pct":                 pct("combined_ci_cv"),
        "avg_cv_activity_pct":                avg("cv_activity_pct"),
        "avg_lactic_acid_mmolL":              avg("lactic_acid_mmolL"),
        "avg_heteroplasmy_blood_pct":         avg("heteroplasmy_blood_pct"),
        "deceased_pct": round(
            100 * sum(1 for p in patients if "Deceased" in p["outcome"]) / n, 1
        ),
    }


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    patients = _make_patients()
    stats = _cohort_stats(patients)
    n = len(patients)

    feat_order = [
        ("Hypertrophic Cardiomyopathy (HCM — dominant feature)", "hypertrophic_cardiomyopathy"),
        ("Exercise intolerance", "exercise_intolerance"),
        ("Myopathy", "myopathy"),
        ("LVOT obstruction (dynamic, on Valsalva)", "lvot_obstruction"),
        ("Arrhythmia (AF / VT)", "arrhythmia"),
        ("Lactic acidosis", "lactic_acidosis"),
        ("Leigh MRI (bilateral BG/brainstem)", "leigh_mri"),
        ("Hypotonia", "hypotonia"),
        ("Encephalopathy", "encephalopathy"),
        ("Developmental delay / regression", "developmental_delay"),
        ("Seizures", "seizures"),
        ("Respiratory failure", "respiratory_failure"),
        ("Cardiac conduction defect (Holter)", "cardiac_conduction_defect"),
        ("Sensorineural hearing loss", "sensorineural_hearing_loss"),
        ("CPEO (progressive external ophthalmoplegia)", "cpeo"),
        ("Ophthalmoplegia", "ophthalmoplegia"),
        ("Combined CI+CV deficiency (overlap mutations)", "combined_ci_cv"),
        ("Retinitis Pigmentosa (ABSENT in point mutations — KSS only)", "retinitis_pigmentosa"),
    ]
    features = [
        {
            "feature": label,
            "pct": round(100 * sum(1 for p in patients if p.get(key, False)) / n, 1),
        }
        for label, key in feat_order
    ]

    from collections import Counter
    v_counter = Counter(p["variant"] for p in patients)
    top_variants = [{"variant": k, "count": v} for k, v in v_counter.most_common(5)]

    alerts = [
        "🔴 HCM ANNUAL ECHO + ECG + HOLTER MANDATORY: MT-ATP8 mutations cause Hypertrophic Cardiomyopathy (HCM) in ~60-85% — annual echocardiography + 12-lead ECG + 24-hour Holter required; LVOT obstruction in 55%; arrhythmia (AF/VT) risk; SCD risk — ICD implant threshold per AHA/ESC HCM guidelines; beta-blockers first-line for symptomatic HCM; cardiology referral MANDATORY at diagnosis",
        "🔴 OVERLAP MUTATION m.8528T>C: DUAL ATP8+ATP6 DISRUPTION — the m.8528T>C overlap mutation occurs in the 46-bp region (rCRS 8527-8572) shared by ATP8 and ATP6; simultaneously changes ATP8 p.Trp55Arg (C-terminal heterodimer contact surface) AND ATP6 p.Met1Thr (translation initiation disruption); combined CI+CV deficiency on BN-PAGE (unlike isolated CV in non-overlap mutations); worst prognosis; ALL variants in rCRS 8527-8572 must be interpreted for BOTH ATP8 and ATP6 simultaneously",
        "⚠️ ISOLATED CV vs COMBINED CI+CV FINGERPRINT: Non-overlap ATP8 mutations (m.8411A>G, m.8423T>C, m.8438T>C) → isolated Complex V deficiency (CI/CII/CIII/CIV normal, COX-positive fibres); overlap mutation m.8528T>C → combined CI+CV on BN-PAGE (combined disruption of both subunits); the biochemical fingerprint guides genotype-phenotype correlation and distinguishes from MELAS MT-TL1 (pan-OXPHOS)",
        "⚠️ NO RETINITIS PIGMENTOSA: MT-ATP8 point mutations do NOT cause retinitis pigmentosa (unlike MT-ATP6/NARP where RP is the hallmark); RP in MT-ATP8 occurs ONLY in deletion syndromes (KSS) — its presence distinguishes KSS/CPEO from point mutations; the absence of RP in a Complex V patient points toward MT-ATP8 rather than MT-ATP6/NARP",
        "⚠️ HCM DDx: MT-ATP8 HCM vs sarcomeric HCM (MYH7/MYBPC3) — key distinction: sarcomeric HCM shows NO OXPHOS deficiency on muscle enzymology; MT-ATP8 shows CV (and possibly CI) deficiency; maternal family history of HCM + lactic acidosis + exercise intolerance → mtDNA panel; WES detects sarcomeric HCM but MISSES MT-ATP8 (mtDNA); dedicated mtDNA sequencing required",
        "🔵 ABSOLUTE CI: Metformin (CI inhibitor — additive with CV/ETC dysfunction) · VPA (CoA sequestration + POLG inhibition + hepatotoxicity) · Propofol (PRIS + direct ETC inhibition) · Linezolid (mt-23S rRNA → impairs ATP8 synthesis directly) · Chloramphenicol (mt-ribosome)",
        "🔵 CARDIAC PHARMACOLOGY CAUTION: Digoxin — CAUTION in HCM (positive inotrope may worsen LVOT obstruction); Disopyramide — negative inotrope useful for LVOT obstruction but monitor QTc; Beta-blockers (metoprolol/atenolol) — FIRST-LINE for symptomatic HCM but use with caution if severe lactic acidosis; Verapamil — alternative if beta-blocker intolerant (NOT in decompensated HF or accessory pathways)",
        "🔵 THIAMINE B1 + BIOTIN + GIR MANDATORY: Empiric thiamine B1 (10-20 mg/kg IV in Leigh/HCM crisis) · Biotin (5-20 mg/day, BTBGD exclusion mandatory) · GIR 6-8 mg/kg/min NEVER fast in Leigh/encephalomyopathy",
        "🔵 WES MISSES MT-ATP8: MT-ATP8 is H-strand mtDNA (rCRS 8366-8572); WES does not reliably cover mtDNA mutations or deletions — dedicated mtDNA panel required; muscle biopsy heteroplasmy preferred over blood (may underestimate by 15-30%); overlap mutation m.8528T>C must be reported with BOTH ATP8 and ATP6 protein consequences",
    ]

    pheno_dist = Counter(p["phenotype"] for p in patients)
    phenotype_distribution = [
        {"phenotype": k, "count": v}
        for k, v in pheno_dist.most_common()
    ]

    onset_dist = {"neonatal_0_3m": 0, "infantile_3_18m": 0,
                  "childhood_1_10y": 0, "adolescent_adult_10y_plus": 0}
    for p in patients:
        w = p["onset_weeks"]
        if w < 13:
            onset_dist["neonatal_0_3m"] += 1
        elif w < 78:
            onset_dist["infantile_3_18m"] += 1
        elif w < 520:
            onset_dist["childhood_1_10y"] += 1
        else:
            onset_dist["adolescent_adult_10y_plus"] += 1

    key_molecular_features = [
        {
            "feature": "2 TM helices — F0 peripheral stalk anchor — 68 aa, 7.6 kDa — smallest mtDNA-encoded subunit",
            "value": "68 aa / 2 TM",
            "significance": "MT-ATP8 is the SMALLEST of all mtDNA-encoded subunits (68 aa vs MT-ATP6 226 aa); its 2 TM helices anchor the F0 peripheral stalk; forms an essential heterodimer with ATP6 (subunit a) for co-translational insertion into the F0 membrane; disruption → complete F0 assembly failure"
        },
        {
            "feature": "ATP8/ATP6 heterodimer — co-translational insertion — neither subunit can insert without the other",
            "value": "Essential heterodimer",
            "significance": "ATP8 and ATP6 form a strict heterodimer that is co-translationally inserted into the inner mitochondrial membrane; ATP8 N-terminus (matrix) anchors peripheral stalk; ATP8 C-terminus contacts ATP6 N-terminus at the 46-bp overlap (rCRS 8527-8572); mutations in ATP8 → ATP6 also cannot be assembled → isolated or combined CV deficiency"
        },
        {
            "feature": "HCM dominant phenotype — NOT Retinitis Pigmentosa (distinguishes from MT-ATP6/NARP)",
            "value": "HCM 60-85%",
            "significance": "MT-ATP8 mutations cause Hypertrophic Cardiomyopathy as the dominant phenotype (60-85%) — in contrast to MT-ATP6/NARP where Retinitis Pigmentosa is the hallmark; RP is ABSENT in MT-ATP8 point mutations; HCM annual echo + Holter mandatory; SCD risk assessment per AHA/ESC HCM guidelines; beta-blockers first-line"
        },
        {
            "feature": "Overlap region rCRS 8527-8572 — overlap mutations affect BOTH ATP8 and ATP6 simultaneously",
            "value": "46-bp overlap",
            "significance": "The 46-bp overlap (rCRS 8527-8572) between ATP8 C-terminus and ATP6 N-terminus is the most complex region in the mitochondrial genome; m.8528T>C changes ATP8 p.Trp55Arg AND ATP6 p.Met1Thr simultaneously; combined CI+CV deficiency results (vs isolated CV in non-overlap mutations); all overlap variants must be interpreted for both genes"
        },
        {
            "feature": "Isolated CV (non-overlap) vs Combined CI+CV (overlap) biochemical fingerprint",
            "value": "BN-PAGE fingerprint",
            "significance": "Non-overlap ATP8 mutations → isolated Complex V deficiency (CI/CII/CIII/CIV normal, COX-positive fibres); overlap mutation m.8528T>C → combined CI+CV deficiency on BN-PAGE (both subunits disrupted); biochemical fingerprint guides genotype-phenotype prediction and distinguishes from pan-OXPHOS (MELAS) and CIV-specific (SURF1) disorders"
        },
    ]

    return {
        "gene": "MT-ATP8",
        "omim_gene": "OMIM *516070",
        "protein": "ATP Synthase F0 Peripheral Stalk Anchor / Subunit 8 (68 aa, ~7.6 kDa, 2 TM helices)",
        "module": "F0 Peripheral Stalk Anchor (Complex V / ATP Synthase subunit 8) — rCRS H-strand 8366-8572 (overlaps ATP6 at 8527-8572)",
        "inheritance": "MATERNAL (mtDNA heteroplasmic — HCM moderate; Leigh/encephalomyopathy high; Deletion variable)",
        "primary_disease": "Hypertrophic Cardiomyopathy (HCM) / Leigh Syndrome / Encephalomyopathy / Exercise Intolerance / KSS-CPEO (large deletion)",
        "overlap_region": {
            "rcrs": "8527-8572",
            "length_bp": 46,
            "description": "46-bp overlap between ATP8 C-terminus and ATP6 N-terminus (Met1 start codon)",
            "key_mutation": "m.8528T>C — changes ATP8 p.Trp55Arg AND ATP6 p.Met1Thr simultaneously",
            "consequence": "Combined CI+CV deficiency (non-overlap ATP8 mutations cause isolated CV only)",
        },
        "tm_helices": 2,
        "aa_length": 68,
        "molecular_weight_kda": 7.6,
        "rcrs_positions": "8366-8572",
        "strand": "H-strand",
        "n_patients": n,
        "seed": SEED,
        "cohort_statistics": stats,
        "cohort_summary_features": features,
        "phenotype_distribution": phenotype_distribution,
        "onset_distribution": onset_dist,
        "key_molecular_features": key_molecular_features,
        "top_variants": top_variants,
        "clinical_alerts": alerts,
        "absolute_contraindications": [
            "Metformin (Complex I inhibitor — additive with CV/ETC dysfunction; increases lactic acidosis risk)",
            "Valproic acid / VPA (CoA sequestration + POLG inhibition + hepatotoxicity — triple danger in Leigh)",
            "Propofol (PRIS + direct ETC inhibition — compounding CV deficiency, fatal in Leigh/HCM crisis)",
            "Linezolid (inhibits mt-23S rRNA → impairs MT-ATP8 synthesis directly; CV collapses)",
            "Chloramphenicol (mt-ribosome inhibitor — ATP8 synthesis impaired; avoid)",
            "Fasting / prolonged NPO (GIR 6-8 mg/kg/min MANDATORY in Leigh crisis — NEVER fast)",
            "Digoxin in HCM (positive inotrope may worsen LVOT obstruction — CAUTION, not absolute CI)",
        ],
        "mandatory_empiric_treatments": [
            "Thiamine B1 — 10-20 mg/kg IV in Leigh/HCM crisis; oral 100-300 mg/day maintenance (PDH cofactor — MANDATORY empiric)",
            "Biotin — 5-20 mg/day empiric (pending BTD/SLC19A3 BTBGD exclusion — Leigh mimic, MANDATORY empiric)",
            "GIR 6-8 mg/kg/min — NEVER fast in Leigh/encephalomyopathy crisis",
            "Cardiology referral MANDATORY at diagnosis (HCM surveillance — annual echo + ECG + Holter)",
        ],
        "level_c_treatments": [
            "CoQ10 ubiquinol (10-20 mg/kg/day) — Level C",
            "Riboflavin B2 (FAD/FMN cofactor — Level C)",
            "L-Carnitine — Level C (mitochondrial myopathy component)",
            "Beta-blockers (metoprolol/atenolol) — first-line for symptomatic HCM (LVOT obstruction)",
            "Verapamil — alternative to beta-blockers for HCM if beta-blocker intolerant (NOT decompensated HF)",
        ],
        "preferred_aed": "Levetiracetam (LEV) — preferred AED for MT-ATP8 Leigh/encephalomyopathy seizures; avoid VPA (absolute CI)",
        "cardiac_monitoring_protocol": "Annual echocardiography (LV wall thickness, LVOT gradient, diastolic function) + 12-lead ECG + 24-hour Holter Monitoring — MANDATORY for ALL MT-ATP8 patients regardless of cardiac symptoms; ICD implant threshold per AHA/ESC HCM guidelines (SCD risk ≥6% at 5 years); AF: anticoagulation as per HCM guidelines",
    }


def get_breakdown():
    patients = _make_patients()
    n = len(patients)

    from collections import Counter

    variant_rows = []
    for v in VARIANTS:
        pts = [p for p in patients if p["variant"] == v["hgvs_mtdna"]]
        if not pts:
            continue
        nv = len(pts)
        pct_fn = lambda key, pts=pts, nv=nv: round(100 * sum(1 for p in pts if p.get(key, False)) / nv, 1)
        variant_rows.append({
            "variant": v["hgvs_mtdna"],
            "protein": v["protein"],
            "domain": v["domain"],
            "type": v["type"],
            "severity": v["severity"],
            "phenotype": v["phenotype"],
            "n_patients": nv,
            "penetrance_pct": v["penetrance_pct"],
            "avg_cv_activity_pct": round(sum(p["cv_activity_pct"] for p in pts) / nv, 1),
            "avg_heteroplasmy_pct": round(sum(p["heteroplasmy_blood_pct"] for p in pts) / nv, 1),
            "avg_lactic_acid": round(sum(p["lactic_acid_mmolL"] for p in pts) / nv, 1),
            "avg_onset_weeks": round(sum(p["onset_weeks"] for p in pts) / nv, 0),
            "hcm_pct": pct_fn("hypertrophic_cardiomyopathy"),
            "leigh_mri_pct": pct_fn("leigh_mri"),
            "lactic_acidosis_pct": pct_fn("lactic_acidosis"),
            "exercise_intolerance_pct": pct_fn("exercise_intolerance"),
            "arrhythmia_pct": pct_fn("arrhythmia"),
            "combined_ci_cv_pct": pct_fn("combined_ci_cv"),
            "phenotype_breakdown": dict(Counter(p["phenotype"] for p in pts)),
            "notes": v["notes"],
        })

    pheno_dist = Counter(p["phenotype"] for p in patients)
    pheno_by_variant = {}
    for v in VARIANTS:
        pts = [p for p in patients if p["variant"] == v["hgvs_mtdna"]]
        if pts:
            pheno_by_variant[v["hgvs_mtdna"]] = dict(Counter(p["phenotype"] for p in pts))

    cv_bands = {"<15%": 0, "15-30%": 0, "30-50%": 0, ">50%": 0}
    for p in patients:
        c = p["cv_activity_pct"]
        if c < 15:
            cv_bands["<15%"] += 1
        elif c < 30:
            cv_bands["15-30%"] += 1
        elif c < 50:
            cv_bands["30-50%"] += 1
        else:
            cv_bands[">50%"] += 1

    heteroplasmy_bands = {"<50%": 0, "50-70%": 0, "70-90%": 0, ">90%": 0}
    for p in patients:
        h = p["heteroplasmy_blood_pct"]
        if h < 50:
            heteroplasmy_bands["<50%"] += 1
        elif h < 70:
            heteroplasmy_bands["50-70%"] += 1
        elif h < 90:
            heteroplasmy_bands["70-90%"] += 1
        else:
            heteroplasmy_bands[">90%"] += 1

    outcome_dist = Counter(p["outcome"] for p in patients)
    outcome_rows = [{"outcome": k, "count": v} for k, v in outcome_dist.most_common()]

    ddx_table = [
        {
            "entity": "MT-ATP6 / NARP (same Complex V, different subunit)",
            "distinguishing_feature": "MT-ATP6/NARP hallmark: Retinitis Pigmentosa (RP, 85% of NARP patients) — ABSENT in MT-ATP8 point mutations; MT-ATP6 = 226 aa F0 proton channel; MT-ATP8 = 68 aa peripheral stalk anchor; NARP heteroplasmy threshold (70-90% NARP, ≥90% Leigh) vs MT-ATP8 HCM-predominant; overlap mutation m.8528T>C affects BOTH",
            "key_test": "ERG (RP in ATP6/NARP — rod-cone dystrophy; ABSENT in ATP8); Respiratory chain enzymology (isolated CV in both; combined CI+CV only in ATP8 overlap); full mtDNA sequencing to distinguish 8366-8526 (ATP8 only) vs 8527-8572 (overlap) vs 8527-9207 (ATP6 only)",
        },
        {
            "entity": "Sarcomeric HCM (MYH7, MYBPC3, TNNT2, TNNI3)",
            "distinguishing_feature": "Sarcomeric HCM: autosomal dominant; WES detectable; NO OXPHOS deficiency on muscle enzymology; NO lactic acidosis at rest; NO maternal family history pattern; MT-ATP8 HCM: CV (and possibly CI) deficiency; maternal pedigree; elevated resting lactate; exercise intolerance disproportionate to HCM severity",
            "key_test": "Cardiac MRI + echo; HCM panel WES (MYH7/MYBPC3/etc.); muscle enzymology (CV normal in sarcomeric HCM); lactate/pyruvate; mtDNA sequencing if sarcomeric HCM panel negative + maternal HCM pattern",
        },
        {
            "entity": "SLC19A3 / BTBGD (biotin-thiamine responsive basal ganglia disease)",
            "distinguishing_feature": "BTBGD: Leigh-identical bilateral BG MRI but TREATABLE with biotin+thiamine; autosomal recessive; NO CV deficiency; WES detectable; MANDATORY exclusion before accepting MT-ATP8 Leigh as cause",
            "key_test": "SLC19A3 sequencing (WES); empiric thiamine+biotin trial; CV enzymology (normal in BTBGD vs deficient in MT-ATP8 Leigh)",
        },
        {
            "entity": "SURF1 (CIV-Leigh, nuclear AR)",
            "distinguishing_feature": "SURF1 Leigh: isolated CIV deficiency (COX-negative fibres); CV normal; MT-ATP8 Leigh: isolated CV or combined CI+CV; COX-positive fibres; SURF1 detectable by WES (nuclear); BN-PAGE CIV vs CV fingerprint critical",
            "key_test": "Respiratory chain enzymology CIV vs CV; COX histochemistry; SURF1 WES",
        },
        {
            "entity": "POLG (Alpers / SANDO / mtDNA depletion)",
            "distinguishing_feature": "POLG: hepatopathy (Alpers) + mtDNA depletion on Southern blot; MT-ATP8: NO hepatopathy; WES detects POLG (nuclear); WES misses MT-ATP8 (mtDNA); VPA ABSOLUTE CI in BOTH (POLG+VPA → fatal hepatotoxicity; ATP8+VPA → CoA depletion)",
            "key_test": "Liver enzymes + ammonia; mtDNA quantification (depletion in POLG); POLG WES vs mtDNA panel",
        },
        {
            "entity": "MT-TL1 m.3243A>G (MELAS tRNA-Leu)",
            "distinguishing_feature": "MELAS: pan-OXPHOS reduction (CI+CIII+CIV all depressed); stroke-like episodes; MT-ATP8: isolated CV (or CI+CV only in overlap); no stroke-like episodes in non-overlap; urinary m.3243A>G load diagnostic for MELAS",
            "key_test": "Urinary epithelial cell m.3243A>G load; respiratory chain enzymology (pan-OXPHOS vs isolated CV); Brain MRI (MELAS cortical lesions vs Leigh BG/brainstem)",
        },
        {
            "entity": "Pompe disease (GAA — acid alpha-glucosidase deficiency)",
            "distinguishing_feature": "Pompe: glycogen storage in lysosomes; HCM (neonatal) + myopathy (late-onset); alpha-glucosidase enzyme deficiency on dried blood spot; ERT (enzyme replacement therapy) available; CV/CI normal in Pompe; WES detectable (nuclear GAA gene)",
            "key_test": "Alpha-glucosidase dried blood spot (Pompe); muscle glycogen PAS stain; GAA sequencing; CV enzymology (normal in Pompe); mtDNA sequencing for MT-ATP8",
        },
        {
            "entity": "Friedreich Ataxia (FRDA, frataxin, GAA repeat)",
            "distinguishing_feature": "FRDA: autosomal recessive; HCM (80%) + ataxia; frataxin iron-sulfur cluster deficiency; cardiomyopathy + cerebellar ataxia (not HCM alone); GAA triplet repeat expansion; CV and CI normal; NO lactic acidosis at rest; autosomal recessive (not maternal)",
            "key_test": "FRDA GAA repeat analysis; frataxin Western blot; ECG/ECHO (HCM in FRDA); CV enzymology (normal); maternal pedigree analysis",
        },
    ]

    hcm_severity_table = [
        {"cv_activity": ">50%", "phenotype": "Exercise intolerance / mild HCM", "hcm_prevalence": "25-35%", "lvot": "15%"},
        {"cv_activity": "30-50%", "phenotype": "HCM-predominant (moderate)", "hcm_prevalence": "70-80%", "lvot": "45-55%"},
        {"cv_activity": "15-30%", "phenotype": "HCM severe + mild Leigh features", "hcm_prevalence": "65-75%", "lvot": "35%"},
        {"cv_activity": "4-18%", "phenotype": "Combined HCM + Leigh (overlap mutations)", "hcm_prevalence": "80%", "lvot": "30%"},
        {"cv_activity": "5-22%", "phenotype": "Leigh/Encephalomyopathy (TM-helix mutations)", "hcm_prevalence": "15-20%", "lvot": "5%"},
    ]

    return {
        "gene": "MT-ATP8",
        "n_patients": n,
        "seed": SEED,
        "variant_breakdown": variant_rows,
        "phenotype_distribution": dict(pheno_dist),
        "phenotype_by_variant": pheno_by_variant,
        "cv_activity_bands": cv_bands,
        "heteroplasmy_bands": heteroplasmy_bands,
        "hcm_severity_table": hcm_severity_table,
        "outcome_distribution": outcome_rows,
        "differential_diagnosis": ddx_table,
        "patient_table": [
            {
                "id": p["patient_id"],
                "phenotype": p["phenotype"],
                "variant": p["variant"],
                "protein": p["protein_change"],
                "heteroplasmy_pct": p["heteroplasmy_blood_pct"],
                "cv_pct": p["cv_activity_pct"],
                "lactate": p["lactic_acid_mmolL"],
                "onset_weeks": p["onset_weeks"],
                "hcm": p["hypertrophic_cardiomyopathy"],
                "arrhythmia": p["arrhythmia"],
                "leigh_mri": p["leigh_mri"],
                "combined_ci_cv": p["combined_ci_cv"],
                "outcome": p["outcome"],
            }
            for p in patients
        ],
    }


def get_definitions():
    return {
        "gene": "MT-ATP8",
        "omim_gene": "OMIM *516070",
        "full_name": "Mitochondrially Encoded ATP Synthase Membrane Subunit 8",
        "protein_name": "ATP synthase F0 peripheral stalk anchor / subunit 8 (68 aa, ~7.6 kDa)",
        "aa_length": 68,
        "molecular_weight_kda": 7.6,
        "tm_helices": 2,
        "rcrs_positions": "8366-8572",
        "strand": "H-strand (overlaps MT-ATP6 at rCRS 8527-8572, 46-bp overlap)",
        "module": "F0 Peripheral Stalk Anchor (Complex V / ATP Synthase subunit 8) — forms heterodimer with ATP6 for co-translational F0 membrane insertion",
        "complex_v_structure": "Complex V (ATP synthase): F1 (matrix; α3β3γδε catalytic) + F0 (membrane; subunit a [MT-ATP6] + c-ring + ATP8 [MT-ATP8] + b/d/F6 stalk); ATP8 (subunit 8, 68 aa, 2 TM) anchors the F0 peripheral stalk and forms an essential heterodimer with ATP6 for co-translational membrane insertion; neither ATP8 nor ATP6 can be assembled independently",
        "overlap_definition": "ATP8/ATP6 overlap region: rCRS 8527-8572 (46 bp) — the final 15.3 codons of ATP8 overlap with the first 15.3 codons of ATP6; mutations in this region affect BOTH genes simultaneously; m.8528T>C changes ATP8 p.Trp55Arg (C-terminal contact surface) AND ATP6 p.Met1Thr (initiation codon disruption); combined CI+CV deficiency results vs isolated CV from non-overlap ATP8 mutations",
        "omim_diseases": {
            "hcm": "Hypertrophic Cardiomyopathy associated with MT-ATP8 mutations — concentric LV hypertrophy, LVOT obstruction, arrhythmia, SCD risk",
            "leigh": "Leigh Syndrome — bilateral symmetric BG/brainstem T2 signal; high heteroplasmy; severe CV deficiency",
            "encephalomyopathy": "Encephalomyopathy — seizures, lactic acidosis, hypotonia, developmental delay",
            "exercise_intolerance": "Exercise Intolerance / Myopathy — adult onset; elevated CK; RRF on Gomori; COX-positive",
            "kss_cpeo": "KSS / CPEO / Pearson Syndrome — large mtDNA deletion spanning ATP8 locus (± ATP6)",
        },
        "key_variants": {
            "m.8411A>G": "p.Lys29Glu — N-terminal matrix loop / OSCP interface; ~30% of cohort; charge reversal disrupts peripheral stalk contact; predominantly HCM + exercise intolerance; isolated CV deficiency; moderate heteroplasmy 52-82%",
            "m.8423T>C": "p.Leu33Pro — N-terminal TM helix 1; ~25% of cohort; helix-breaking proline collapses TM1 hydrophobic core; Leigh syndrome / severe encephalomyopathy; high heteroplasmy ≥85%; isolated CV deficiency; BTBGD exclusion mandatory",
            "m.8528T>C": "p.Trp55Arg (ATP8) + p.Met1Thr (ATP6) — OVERLAP MUTATION; ~20% of cohort; dual disruption of ATP8 C-terminal contact surface AND ATP6 initiation codon; combined CI+CV deficiency; HCM + Leigh-like; worst prognosis",
            "m.8438T>C": "p.Leu38Pro — TM1-TM2 junction linker; ~15% of cohort; partial F0 assembly impairment; adult exercise intolerance + mild HCM; moderate CV deficiency 25-50%; incomplete penetrance",
            "large_deletion": "ATP8-spanning deletion — KSS/CPEO/Pearson; always co-involves ATP6 overlap region (rCRS 8527-8572); multi-complex OXPHOS; annual Holter mandatory",
        },
        "hcm_definition": "Hypertrophic Cardiomyopathy (HCM): concentric LV hypertrophy (wall thickness ≥13 mm in adults, ≥2 Z-scores in children) with preserved LV ejection fraction; dynamic LVOT obstruction (Valsalva maneuver, peak gradient >30 mmHg); risk of SCD, AF, heart failure; in MT-ATP8 the mechanism is metabolic (CV/ATP deficiency → myocardial energy starvation → compensatory hypertrophy); management: beta-blockers (first-line symptomatic), disopyramide (LVOT obstruction), verapamil (beta-blocker intolerant), ICD (SCD risk ≥6% at 5y by AHA/ESC calculator); septal reduction therapy in refractory LVOT obstruction",
        "leigh_definition": "Leigh Syndrome: progressive neurodegeneration with bilateral symmetric T2/FLAIR hyperintensity in the basal ganglia, brainstem, and/or thalamus; caused by OXPHOS deficiency (CV in ATP8 mutations); infantile or childhood onset; lactic acidosis; hypotonia; developmental regression; respiratory failure; outcome often fatal in childhood without metabolic support",
        "assembly_mechanism_definition": "ATP8/ATP6 Co-translational Assembly: mitochondrial ribosomes translate ATP8 and ATP6 from separate mRNAs but the two proteins are incorporated into F0 together as a heterodimer; ATP8 N-terminus (matrix face) docks with the peripheral stalk (b-subunit / OSCP); ATP8 C-terminus contacts ATP6 N-terminus in the overlap region; without this heterodimer, neither protein can stably insert into the lipid bilayer — F0 assembly fails completely; F1 (head) may be present on BN-PAGE as a subcomplex without the F0 anchor",
        "combined_ci_cv_definition": "Combined CI+CV Deficiency (overlap mutations): the m.8528T>C overlap mutation simultaneously disrupts ATP8 C-terminus (contact surface) and ATP6 Met1 initiation codon; both F0 subunits fail to be properly synthesized and assembled; the resulting F0 assembly failure can impair Complex I (CI) biogenesis if the shared inner membrane assembly platform is disrupted; BN-PAGE shows absent/reduced bands for BOTH Complex I and Complex V; this combined fingerprint distinguishes overlap mutations from non-overlap ATP8 mutations (isolated CV only)",
        "m8411ag_definition": "m.8411A>G: nucleotide substitution at rCRS position 8411 A→G in MT-ATP8; causes p.Lys29Glu in the N-terminal matrix-exposed loop of subunit 8; charge reversal (positive Lys → negative Glu) disrupts the peripheral stalk contact surface (OSCP / b-subunit interface); predominantly HCM + exercise intolerance phenotype; isolated CV deficiency; moderate heteroplasmy 52-82% blood; cardiomyopathy dominant feature",
        "m8423tc_definition": "m.8423T>C: nucleotide substitution at rCRS position 8423 T→C in MT-ATP8; causes p.Leu33Pro in TM helix 1; helix-breaking proline at Leu33 collapses the hydrophobic TM1 core of subunit 8; complete F0 membrane anchor failure; isolated CV deficiency (5-20% residual); Leigh syndrome at high heteroplasmy ≥85%; early infantile onset; BTBGD exclusion mandatory",
        "m8528tc_definition": "m.8528T>C (OVERLAP MUTATION): occurs at rCRS position 8528 in the 46-bp ATP8/ATP6 overlap region; simultaneously changes (1) ATP8 p.Trp55Arg — critical C-terminal contact surface for ATP8/ATP6 heterodimer; (2) ATP6 p.Met1Thr — initiation codon of ATP6 (ATG 8527-8529) disrupted (ATG→ACG); double-hit results in combined CI+CV deficiency on BN-PAGE; HCM + Leigh-like phenotype; worst prognosis of all MT-ATP8 variants; ALL variants in rCRS 8527-8572 must be interpreted for BOTH ATP8 AND ATP6 consequences",
        "maternal_inheritance_definition": "Maternal (mtDNA) inheritance: all mtDNA is transmitted through the maternal line; affected mothers transmit MT-ATP8 mutations to all children; variable heteroplasmy in offspring due to mitochondrial genetic bottleneck during oogenesis; maternal relatives with unexplained HCM or exercise intolerance should receive targeted mtDNA sequencing; asymptomatic high-heteroplasmy carriers need cardiac surveillance",
        "wes_miss_definition": "WES misses MT-ATP8: MT-ATP8 is encoded by mitochondrial DNA (H-strand rCRS 8366-8572) — WES does not reliably capture mtDNA mutations or large deletions; dedicated mtDNA sequencing (Sanger for specific mutations; NGS mtDNA panel for full coverage) required; muscle biopsy heteroplasmy preferred over blood (blood may underestimate by 15-30%); overlap region mutations (rCRS 8527-8572) require interpretation by a clinical lab experienced with dual-gene mtDNA reporting",
        "btbgd_mandatory_exclusion": "BTBGD (Biotin-Thiamine responsive Basal Ganglia Disease / SLC19A3): MANDATORY exclusion before accepting MT-ATP8 as Leigh cause; bilateral BG signal clinically identical to Leigh/MILS MRI; autosomal recessive SLC19A3 mutation; NO CV deficiency (CV normal in BTBGD); TREATABLE with biotin + thiamine (dramatic MRI reversal); WES detectable; empiric biotin+thiamine trial clinically mandatory while genetic results pending",
        "biochemical_fingerprint": {
            "non_overlap_point_mutations": "Isolated Complex V deficiency (CV ATP synthesis 5-60% depending on variant/heteroplasmy); CI, CII, CIII, CIV all NORMAL on BN-PAGE; COX-positive fibres on histochemistry; BN-PAGE: absent or severely reduced Complex V band; F1 subcomplex may appear without F0 anchor",
            "overlap_mutation_m8528tc": "Combined CI+CV deficiency on BN-PAGE (both Complex I and Complex V reduced); COX-positive fibres; reflects dual disruption of both ATP8 and ATP6 subunits simultaneously; most severe biochemical phenotype",
            "large_deletions": "Multi-complex OXPHOS reduction (CV+CI ± others) when deletion extends beyond ATP8 into adjacent regions including ATP6; multi-complex BN-PAGE fingerprint",
        },
        "absolute_contraindications": {
            "Metformin": "Complex I inhibitor — additive with CV/ETC dysfunction; combined CI+CV in overlap mutations makes this especially dangerous",
            "VPA / Valproic Acid": "CoA sequestration + POLG inhibition + hepatotoxicity — triple danger in Leigh/encephalomyopathy",
            "Propofol": "PRIS (propofol infusion syndrome) + direct ETC inhibition — compounding CV deficiency in HCM crisis or Leigh",
            "Linezolid": "Inhibits mt-23S rRNA → impairs MT-ATP8 (and ATP6) synthesis directly; CV/F0 assembly collapses",
            "Chloramphenicol": "Mt-ribosome inhibitor — impairs ATP8 synthesis; avoid in all MT-ATP8 patients",
            "Fasting": "GIR 6-8 mg/kg/min MANDATORY in Leigh crisis — NEVER fast; HCM patients also at increased metabolic stress risk",
            "Digoxin (CAUTION not absolute CI)": "Positive inotrope may worsen dynamic LVOT obstruction in HCM; use with extreme caution; not an absolute CI but generally avoided in obstructive HCM",
        },
        "recommended_treatments": {
            "thiamine_B1": "Level B — 10-20 mg/kg IV in crisis; oral 100-300 mg/day; PDH cofactor — MANDATORY empiric",
            "biotin": "5-20 mg/day empiric — pending BTD/SLC19A3 (BTBGD) exclusion — MANDATORY empiric in Leigh phenotype",
            "coq10_ubiquinol": "Level C — 10-20 mg/kg/day; supports OXPHOS electron flow",
            "riboflavin_B2": "Level C — FAD/FMN cofactor for CI/CIII",
            "l_carnitine": "Level C — mitochondrial myopathy component",
            "lev": "Levetiracetam — preferred AED; avoid VPA (absolute CI)",
            "gir": "GIR 6-8 mg/kg/min — continuous glucose infusion; NEVER fast in Leigh crisis",
            "beta_blockers": "Metoprolol/atenolol — first-line symptomatic HCM (LVOT obstruction, arrhythmia)",
            "verapamil": "Alternative for HCM if beta-blocker intolerant — NOT in decompensated HF or accessory pathways",
        },
        "specialist_monitoring": {
            "Cardiology": "MANDATORY at diagnosis — annual echocardiography (LV wall thickness, LVOT gradient, diastolic function) + 12-lead ECG + 24-hour Holter; ICD assessment per AHA/ESC HCM SCD risk calculator; AF anticoagulation; HCM sports restriction counselling",
            "Neurology": "Leigh: brain MRI 6-monthly until stable; encephalomyopathy: seizure management with LEV; developmental assessment",
            "Metabolic": "Lactate/pyruvate ratio quarterly in Leigh; plasma amino acids; urine organic acids; CV enzymology from muscle biopsy",
            "Genetics": "Maternal relatives: targeted MT-ATP8 sequencing (and ATP6 in overlap region); heteroplasmy cascade testing; pre-conceptional counselling",
            "Ophthalmology": "Annual fundus exam for deletion patients (KSS pigmentary retinopathy); NOT required for point mutations (no RP feature)",
            "Audiology": "Annual audiometry for deletion patients (KSS hearing loss); consider for point mutation Leigh phenotype",
        },
        "kss_holter_policy": "KSS patients (large deletion spanning ATP8 locus) require annual ECG + Holter monitoring; pacemaker implantation threshold: PR interval >240 ms or Mobitz II/complete heart block; sudden-onset fatal conduction block risk; ATP8 deletions ALWAYS co-involve ATP6 overlap region (rCRS 8527-8572) — interpret for both genes",
        "wes_coverage": "MT-ATP8 is a mitochondrial DNA (H-strand rCRS 8366-8572) gene — WES does NOT reliably cover mtDNA mutations or large deletions; dedicated mtDNA panel (Sanger for specific mutations; NGS mtDNA panel for full coverage) required; muscle biopsy heteroplasmy preferred over blood (blood may underestimate by 15-30%); overlap mutations (rCRS 8527-8572) must be reported for BOTH MT-ATP8 and MT-ATP6 simultaneously by the interpreting laboratory",
        "key_references": [
            "Jonckheere AI et al. (2012) Mitochondrial ATP synthase: architecture, function and pathology. J Inherit Metab Dis 35(2):211-225. [ATP8/ATP6 heterodimer assembly mechanism; Complex V F0 structure-function; disease phenotypes]",
            "Rubio-Gozalbo ME et al. (2006) Cardiomyopathy in patients with the mitochondrial DNA mutation m.8528T>C in MT-ATP8/MT-ATP6. [Overlap mutation m.8528T>C: dual ATP8+ATP6 disruption → combined HCM+Leigh; overlap region interpretation]",
            "Zeviani M & Carelli V. (2021) Mitochondrial disorders. Curr Opin Neurol 34(3):268-277. [MT-ATP8 cardiomyopathy phenotype in context of complex V disorders; current review]",
            "Holt IJ et al. (1990) A new mitochondrial disease associated with mitochondrial DNA heteroplasmy. Am J Hum Genet 46(3):428-433. [NARP/ATP6 founding paper; defines the ATP8/ATP6 overlap region biology for the first time]",
            "Ott M et al. (2016) Mitoribosomes can serve as platforms for co-translational insertion of mitochondrially encoded membrane proteins. Mol Cell 61(4):529-539. [Co-translational ATP8/ATP6 heterodimer membrane insertion mechanism — ATP8 first, ATP6 follows]",
        ],
        "cohort_seed": SEED,
        "n_patients": N_PATIENTS,
        "generated": "2026-09-03",
    }


if __name__ == "__main__":
    import json
    overview = get_overview()
    print(f"MT-ATP8 overview: {overview['n_patients']} patients, "
          f"avg CV {overview['cohort_statistics']['avg_cv_activity_pct']}%, "
          f"avg lactate {overview['cohort_statistics']['avg_lactic_acid_mmolL']} mmol/L")
    print(f"HCM: {overview['cohort_statistics']['hypertrophic_cardiomyopathy_pct']}%")
    print(f"Leigh MRI: {overview['cohort_statistics']['leigh_mri_pct']}%")
    print(f"Exercise intolerance: {overview['cohort_statistics']['exercise_intolerance_pct']}%")
    bd = get_breakdown()
    print(f"Variants: {len(bd['variant_breakdown'])}")
    defs = get_definitions()
    print(f"Key variants: {list(defs['key_variants'].keys())}")
    print("✅ MT-ATP8 dashboard OK")
