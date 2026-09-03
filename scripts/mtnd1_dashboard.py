#!/usr/bin/env python3
"""MT-ND1 — Mitochondrially Encoded NADH:Ubiquinone Oxidoreductase Core Subunit 1 /
Isolated Complex I Deficiency — Leber's Hereditary Optic Neuropathy (LHON) #2 Worldwide +
Leigh Syndrome — MOST CONSERVED CI Subunit — Proximal Membrane Arm (ND1-Module) — Maternal Inheritance.

MT-ND1 (OMIM *516000) encodes the 318-amino-acid, 36 kDa NADH dehydrogenase subunit 1,
the MOST CONSERVED subunit of Complex I (CI) across all eukaryotes. It contains 8 transmembrane
(TM) helices and forms the core anchor of the proximal membrane arm (P-module / ND1-module),
the FIRST sub-module assembled in the step-wise CI biogenesis pathway.

  MT-ND1 gene        OMIM *516000
  Disease            LHON (Leber's Hereditary Optic Neuropathy) #2 primary worldwide / Leigh Syndrome
  Protein            318 aa, 36 kDa; 8 TM helices; PROXIMAL MEMBRANE ARM — ND1-MODULE CORE ANCHOR
  Genome             Mitochondrial DNA (mtDNA), H-strand, rCRS positions 3307–4262
  Inheritance        MATERNAL — mtDNA; near-homoplasmic (LHON) or high heteroplasmy (Leigh)
  Phenotype          LHON (15% of all LHON worldwide; intermediate 22% spontaneous visual recovery)
                     Leigh syndrome (bilateral symmetric BG/brainstem, high heteroplasmy ≥70%)
                     Leigh/MELAS overlap; KSS/CPEO (large deletion); Exercise Intolerance Myopathy

UNIQUE MOLECULAR POSITION — MOST CONSERVED CI SUBUNIT / ND1-MODULE CORE:
  The 7 mtDNA-encoded CI subunits occupy the membrane arm antiporter module:
    ND1 (318 aa) — PROXIMAL membrane arm (H-strand) — MOST CONSERVED — ND1-MODULE CORE ANCHOR
    ND2 (347 aa) — middle antiporter module (H-strand)
    ND3 (115 aa) — N-module/membrane arm junction (H-strand)
    ND4 (459 aa) — central antiporter module (H-strand)
    ND4L (98 aa) — hairpin connector between ND4 and ND6 (H-strand)
    ND5 (603 aa) — largest, distal antiporter module (H-strand)
    ND6 (174 aa) — ONLY L-strand; distal membrane arm TIP; unique 5-TM inverted topology

  ND1 is the FIRST subunit assembled into the P-module (proximal membrane arm). The ND1-module
  (comprising ND1 + NDUFB7 + NDUFB8 + NDUFB9 + NDUFB10 + NDUFB11 + NDUFB4 + NDUFB2)
  is the proximal membrane arm sub-assembly that nucleates CI membrane arm biogenesis.
  Cryo-EM structures (Agip 2018 Nat Struct Mol Biol) confirm ND1 as the structural core
  of the proximal arm, contacting the N-module (soluble arm) at the matrix surface.

  LHON RANK AMONG 3 PRIMARY MUTATIONS:
    MT-ND4 m.11778G>A — #1 worldwide (~70% of LHON) — WORST prognosis (<4% recovery)
    MT-ND1 m.3460G>A  — #2 worldwide (~15% of LHON) — INTERMEDIATE prognosis (22% recovery)
    MT-ND6 m.14484T>C — #3 worldwide (~14% of LHON) — BEST prognosis (50% recovery)

PATHOPHYSIOLOGY:
  ND1 disruption → proximal membrane arm structural failure → isolated CI deficiency:
  • For LHON variants (near-homoplasmic): subtle CI conformational change at the proximal arm
    impairs supercomplex (I+III₂, I+III₂+IV) formation → ↑ ROS → selective retinal ganglion
    cell (RGC) axon death (papillomacular bundle)
  • For Leigh variants (high heteroplasmy ≥70%): severe CI loss → lactic acidosis + Leigh MRI
  LHON-specific mechanism in MT-ND1:
    m.3460G>A (p.Ala52Thr): Ala52 in TM1 — hydrophobic core — subtle conformational shift
    at the proximal arm → partial CI assembly defect → moderate CI reduction (35–55% residual)
    → intermediate spontaneous recovery (22%) between ND4 (worst <4%) and ND6 (best 50%).

PHENOTYPE SPECTRUM:
  LHON (near-homoplasmic, m.3460G>A — #2 primary worldwide):
    • Subacute painless central visual loss — Days to 6 weeks onset: 98%
    • Sequential bilateral — second eye 6–8 weeks later: 97%
    • Red-green dyschromatopsia: 95%
    • Peripapillary telangiectatic microangiopathy (NO FFA leak — KEY DDx NAION): 88%
    • Male predominance: 80–90% (X-linked modifier; incomplete penetrance)
    • Incomplete penetrance: 50–60% males; 10–15% females
    • Mean onset: 20–30 years (between ND6 teens and ND4 20–35 y)
    • INTERMEDIATE spontaneous visual recovery: 22% (vs ND4 <4%; ND6 50%)
    • Tobacco: ABSOLUTE CI — major environmental LHON conversion trigger
    • Ethambutol: ABSOLUTE CI — ALL MT-ND1 LHON genotypes
  LEIGH SYNDROME (high heteroplasmy ≥70%, non-LHON variants):
    • Bilateral symmetric BG/brainstem T2 signal hyperintensity: 85%
    • Lactic acidosis severe: 95%
    • Hypotonia: 92%; Developmental delay/regression: 92%
    • Isolated CI deficiency: 5–25% residual

Key Published References:
  Wallace DC et al. (1988) Mitochondrial DNA mutation associated with Leber's hereditary optic
    neuropathy. Science 242(4884):1427-30. (Seminal CI-LHON discovery paper; context for ND1)
  Huoponen K et al. (1991) A new mtDNA mutation associated with Leber hereditary optic
    neuroretinopathy. Am J Hum Genet 48(6):1147-53. (First description m.3460G>A ND1 LHON)
  Jun AS et al. (1994) Mitochondrial DNA mutation at nucleotide position 3460 and 11778
    in Leber hereditary optic neuropathy. Biochem Biophys Res Commun 200(2):1062-9.
    (ND1 CI assembly context vs ND4)
  Agip AA et al. (2018) Cryo-EM structures of complex I from mouse heart mitochondria
    in two biochemical states. Nat Struct Mol Biol 25(7):548-556.
    (ND1-module proximal arm cryo-EM; ND1 as core anchor confirmed)
  Klopstock T et al. (2011) A randomized placebo-controlled trial of idebenone in Leber's
    hereditary optic neuropathy. Brain 134(Pt 9):2677-86. (RHODOS trial — covers m.3460G>A)
"""

import random

SEED = 759
N_PATIENTS = 40

# ── Pathogenic / likely-pathogenic variants in MT-ND1 ────────────────────────
VARIANTS = [
    {
        "hgvs_mtdna": "m.3460G>A",
        "protein": "p.Ala52Thr",
        "domain": "TM helix 1 / proximal membrane arm hydrophobic core (ND1-module)",
        "type": "Missense",
        "severity": "LHON-primary #2 worldwide",
        "phenotype": "LHON (subacute optic atrophy, near-homoplasmic)",
        "penetrance_pct": 50,
        "notes": "Most common MT-ND1 pathogenic variant; accounts for ~15% of LHON worldwide (#2 after m.11778G>A ND4 ~70%); Ala52 in TM helix 1, hydrophobic core of ND1-module proximal arm; near-homoplasmic (>90%) in blood; INTERMEDIATE visual prognosis: 22% spontaneous recovery (better than ND4 <4%, worse than ND6 50%); male predominance 80-90%; onset 20-30 years; CI residual 35-55% (intermediate between ND4 and ND6 severity); Huoponen 1991 Am J Hum Genet (discovery m.3460G>A).",
    },
    {
        "hgvs_mtdna": "m.3697G>A",
        "protein": "p.Ser110Asn",
        "domain": "TM helix 4 / proton half-channel ND1-module",
        "type": "Missense",
        "severity": "Severe (Leigh / MELAS overlap)",
        "phenotype": "Leigh Syndrome / MELAS Overlap (high heteroplasmy)",
        "penetrance_pct": 80,
        "notes": "~20% of MT-ND1 disease cohort; Ser110Asn in TM4 of ND1; disrupts the proton half-channel geometry at the proximal membrane arm; high heteroplasmy (≥70%) required for Leigh phenotype; bilateral symmetric basal ganglia + brainstem T2 hyperintensity; CI severely reduced (5-25% residual); MELAS-like stroke episodes in some carriers; lactic acidosis 95%; may mimic MELAS-overlap phenotype with partial CI recovery.",
    },
    {
        "hgvs_mtdna": "m.3890G>A",
        "protein": "p.Arg195Gln",
        "domain": "TM helix 7 loop / ND1-module matrix-facing interface",
        "type": "Missense",
        "severity": "Severe (Leigh infantile)",
        "phenotype": "Leigh Syndrome (isolated CI deficiency, severe infantile)",
        "penetrance_pct": 85,
        "notes": "~15% of MT-ND1 disease cohort; Arg195Gln at TM7 loop disrupts charged residue at the matrix-facing ND1 interface critical for N-module docking; severe CI deficiency (CI 5-20% residual); infantile onset (typically 0-12 months); bilateral putamen + brainstem Leigh MRI 85%; hypotonia 92%; lactic acidosis 95%; early death without intervention; GIR 6-8 mandatory; no LHON component.",
    },
    {
        "hgvs_mtdna": "m.4171C>A",
        "protein": "p.Leu289Met",
        "domain": "TM helix 8 / C-terminal ND1-module",
        "type": "Missense",
        "severity": "Intermediate (Leigh / exercise intolerance)",
        "phenotype": "Leigh Syndrome / Exercise Intolerance Myopathy",
        "penetrance_pct": 70,
        "notes": "~10% of MT-ND1 disease cohort; Leu289Met at TM8 C-terminal ND1-module; variable phenotype depending on heteroplasmy — exercise intolerance myopathy at lower heteroplasmy (50-70%), Leigh syndrome at higher heteroplasmy (>70%); CI residual 10-40%; adult-onset exercise intolerance with normal resting CI in some; muscle biopsy may show ragged-red fibres (RRF) in KSS-spectrum.",
    },
    {
        "hgvs_mtdna": "m.3376G>A",
        "protein": "p.Arg5Gln",
        "domain": "N-terminal matrix loop / pre-TM1 ND1-module",
        "type": "Missense",
        "severity": "Mild (moderate CI / MELAS-like)",
        "phenotype": "Moderate CI Deficiency / MELAS-like (near-homoplasmic)",
        "penetrance_pct": 35,
        "notes": "~8% of MT-ND1 disease cohort; Arg5Gln near N-terminus before TM1; moderate CI reduction (CI 40-65% residual); MELAS-like phenotype (stroke-like episodes, lactic acidosis) without classic ND4/ND1 LHON features; lower penetrance than primary LHON variants; may co-exist on same mtDNA as secondary modifier; incomplete penetrance — some carriers oligosymptomatic.",
    },
    {
        "hgvs_mtdna": "Large mtDNA deletion (ND1-spanning)",
        "protein": "Frameshift / deletion",
        "domain": "Partial or complete ND1 locus (H-strand rCRS 3307–4262)",
        "type": "Large deletion",
        "severity": "Variable",
        "phenotype": "Pearson / KSS / CPEO (large mtDNA deletion syndrome)",
        "penetrance_pct": 65,
        "notes": "~12%; large-scale mtDNA deletions (e.g., 'common deletion' 4977 bp or novel deletions) spanning or including the ND1 locus contribute to Pearson syndrome (infantile), Kearns-Sayre syndrome (CPEO + cardiomyopathy + RRF <20y), or CPEO (adult); variable heteroplasmy; KSS annual Holter monitoring mandatory; H-strand encoded, WES misses — long-read or Southern blot sequencing for confirmation; distinct from point mutations by Southern blot breakpoints.",
    },
]

VARIANT_WEIGHTS = [0.35, 0.20, 0.15, 0.10, 0.08, 0.12]


# ── Patient-cohort generator ──────────────────────────────────────────────────
def _make_patients(n=N_PATIENTS, seed=SEED):
    rng = random.Random(seed)
    patients = []

    phenotype_distribution = [
        ("LHON (subacute optic atrophy, near-homoplasmic)", 0.35),
        ("Leigh Syndrome (isolated CI deficiency, high heteroplasmy)", 0.35),
        ("Leigh/MELAS Overlap (high heteroplasmy, stroke-like episodes)", 0.10),
        ("KSS/CPEO (large deletion syndrome)", 0.12),
        ("Exercise Intolerance Myopathy (moderate CI, adult)", 0.08),
    ]

    outcomes = [
        ("Deceased < 2 years (Leigh neonatal severe)", 0.07),
        ("Deceased 2–10 years (Leigh / Leigh-MELAS severe)", 0.08),
        ("Alive — severe visual loss, light perception only (LHON)", 0.18),
        ("Alive — moderate visual loss, legally blind (LHON)", 0.15),
        ("Alive — partial visual recovery ≥20/200 (idebenone / spontaneous)", 0.14),
        ("Alive — stable vision (LHON, oligosymptomatic)", 0.08),
        ("Alive — Leigh active, severe disability (Leigh/MELAS)", 0.12),
        ("Alive — CPEO only, minimal systemic impact (large deletion)", 0.08),
        ("Alive — exercise intolerance, managed (moderate CI)", 0.10),
    ]

    for i in range(n):
        v_idx = rng.choices(range(len(VARIANTS)), weights=VARIANT_WEIGHTS)[0]
        variant = VARIANTS[v_idx]

        is_lhon = variant["hgvs_mtdna"] == "m.3460G>A"
        is_leigh_melas = "MELAS" in variant["phenotype"] and "LHON" not in variant["phenotype"]
        is_leigh = "Leigh" in variant["phenotype"] and "MELAS" not in variant["phenotype"]
        is_deletion = "deletion" in variant["hgvs_mtdna"]
        is_exercise = "Exercise" in variant["phenotype"]
        is_modifier = "MELAS-like" in variant["phenotype"]

        # Heteroplasmy — LHON near-homoplasmic, Leigh high, deletion variable
        if is_lhon:
            heteroplasmy_blood_pct = round(rng.uniform(88, 100), 1)
        elif is_modifier:
            heteroplasmy_blood_pct = round(rng.uniform(80, 99), 1)
        elif is_leigh_melas:
            heteroplasmy_blood_pct = round(rng.uniform(70, 98), 1)
        elif is_leigh:
            heteroplasmy_blood_pct = round(rng.uniform(70, 98), 1)
        elif is_exercise:
            heteroplasmy_blood_pct = round(rng.uniform(50, 80), 1)
        elif is_deletion:
            heteroplasmy_blood_pct = round(rng.uniform(20, 65), 1)
        else:
            heteroplasmy_blood_pct = round(rng.uniform(50, 95), 1)

        # Phenotype label driven by variant + heteroplasmy
        if is_deletion:
            pheno_label = rng.choices(
                ["Pearson Syndrome", "Kearns-Sayre Syndrome (KSS)", "CPEO (adult)"],
                weights=[0.20, 0.50, 0.30],
            )[0]
        elif is_leigh_melas:
            pheno_label = rng.choices(
                ["Leigh/MELAS Overlap (high heteroplasmy, stroke-like episodes)",
                 "Leigh Syndrome (isolated CI deficiency, high heteroplasmy)"],
                weights=[0.65, 0.35],
            )[0]
        elif is_leigh:
            pheno_label = "Leigh Syndrome (isolated CI deficiency, high heteroplasmy)"
        elif is_exercise and heteroplasmy_blood_pct > 70:
            pheno_label = rng.choices(
                ["Leigh Syndrome (isolated CI deficiency, high heteroplasmy)",
                 "Exercise Intolerance Myopathy (moderate CI, adult)"],
                weights=[0.40, 0.60],
            )[0]
        elif is_exercise:
            pheno_label = "Exercise Intolerance Myopathy (moderate CI, adult)"
        elif is_modifier:
            if heteroplasmy_blood_pct > 90:
                pheno_label = rng.choices(
                    ["LHON (subacute optic atrophy, near-homoplasmic)",
                     "Moderate CI Deficiency / MELAS-like (near-homoplasmic)"],
                    weights=[0.30, 0.70],
                )[0]
            else:
                pheno_label = "Moderate CI Deficiency / MELAS-like (near-homoplasmic)"
        elif is_lhon:
            pheno_label = "LHON (subacute optic atrophy, near-homoplasmic)"
        else:
            pheno_label = rng.choices(
                [p[0] for p in phenotype_distribution],
                weights=[p[1] for p in phenotype_distribution],
            )[0]

        # CI activity — LHON has intermediate CI reduction (ND1 position: 35-55% residual)
        if "LHON" in pheno_label and "Leigh" not in pheno_label:
            ci_activity_pct = round(rng.uniform(35, 58), 1)  # intermediate vs ND6 55-78%, ND4 30-50%
        elif "Leigh/MELAS" in pheno_label:
            ci_activity_pct = round(rng.uniform(8, 25), 1)
        elif "Leigh" in pheno_label:
            ci_activity_pct = round(rng.uniform(5, 25), 1)
        elif "CPEO" in pheno_label or "KSS" in pheno_label:
            ci_activity_pct = round(rng.uniform(20, 45), 1)
        elif "Pearson" in pheno_label:
            ci_activity_pct = round(rng.uniform(8, 28), 1)
        elif "Exercise" in pheno_label:
            ci_activity_pct = round(rng.uniform(30, 55), 1)
        elif "MELAS-like" in pheno_label:
            ci_activity_pct = round(rng.uniform(35, 65), 1)
        else:
            ci_activity_pct = round(rng.uniform(25, 55), 1)

        # Lactic acid
        if "LHON" in pheno_label and "Leigh" not in pheno_label:
            lactic_acid = round(rng.uniform(1.2, 3.8), 1)   # typically mild/normal in pure LHON
        elif "Leigh" in pheno_label or "MELAS" in pheno_label:
            lactic_acid = round(rng.uniform(5.5, 20.0), 1)
        elif "CPEO" in pheno_label or "KSS" in pheno_label:
            lactic_acid = round(rng.uniform(2.5, 7.5), 1)
        elif "Exercise" in pheno_label:
            lactic_acid = round(rng.uniform(3.0, 9.0), 1)
        else:
            lactic_acid = round(rng.uniform(1.5, 5.0), 1)

        # LHON-specific features
        optic_atrophy = rng.random() < (
            0.98 if "LHON" in pheno_label else
            0.12 if "MELAS" in pheno_label else 0.08
        )
        sequential_bilateral = rng.random() < (0.97 if optic_atrophy else 0.05)
        peripapillary_telangiectasia = rng.random() < (0.88 if "LHON" in pheno_label and optic_atrophy else 0.08)
        # m.3460G>A: 22% spontaneous recovery (intermediate)
        lhon_recovery = (
            variant["hgvs_mtdna"] == "m.3460G>A"
            and rng.random() < 0.22
        )
        male_sex = rng.random() < (0.85 if "LHON" in pheno_label else 0.50)
        red_green_color_loss = rng.random() < (0.95 if optic_atrophy else 0.05)

        # Neurological features
        leigh_mri = rng.random() < (
            0.85 if "Leigh" in pheno_label else
            0.55 if "MELAS" in pheno_label else
            0.15 if "KSS" in pheno_label else 0.05
        )
        lactic_acidosis = rng.random() < (
            0.95 if "Leigh" in pheno_label else
            0.80 if "MELAS" in pheno_label else
            0.55 if "CPEO" in pheno_label or "KSS" in pheno_label else
            0.08  # LHON — typically normal lactate
        )
        hypotonia = rng.random() < (
            0.92 if "Leigh" in pheno_label else
            0.45 if "MELAS" in pheno_label else 0.12
        )
        dev_delay = rng.random() < (
            0.92 if "Leigh" in pheno_label else
            0.50 if "MELAS" in pheno_label else 0.08
        )
        seizures = rng.random() < (
            0.45 if "Leigh" in pheno_label else
            0.55 if "MELAS" in pheno_label else 0.05
        )
        cardiomyopathy = rng.random() < (
            0.35 if "KSS" in pheno_label else
            0.15 if "Leigh" in pheno_label else 0.05
        )
        cpeo = rng.random() < (
            0.65 if "CPEO" in pheno_label or "KSS" in pheno_label else
            0.06 if "LHON" in pheno_label else 0.05
        )
        exercise_intolerance = rng.random() < (
            0.90 if "Exercise" in pheno_label else
            0.80 if "CPEO" in pheno_label or "KSS" in pheno_label else
            0.30 if "LHON" in pheno_label else 0.25
        )
        rrfs = rng.random() < (
            0.65 if "CPEO" in pheno_label or "KSS" in pheno_label else
            0.35 if "Exercise" in pheno_label else
            0.12 if "Leigh" in pheno_label else 0.05
        )
        respiratory = rng.random() < (
            0.60 if "Leigh" in pheno_label else
            0.25 if "MELAS" in pheno_label else 0.08
        )
        encephalopathy = rng.random() < (
            0.88 if "Leigh" in pheno_label else
            0.65 if "MELAS" in pheno_label else 0.05
        )
        melas_overlap = rng.random() < (
            0.85 if "MELAS" in pheno_label else
            0.05 if "Leigh" in pheno_label else 0.03
        )
        dystonia = rng.random() < (
            0.25 if "Leigh" in pheno_label else
            0.12 if "MELAS" in pheno_label else 0.04
        )
        kss_cpeo = rng.random() < (
            0.85 if "KSS" in pheno_label or "CPEO" in pheno_label or "Pearson" in pheno_label else 0.05
        )

        outcome_label = rng.choices(
            [o[0] for o in outcomes], weights=[o[1] for o in outcomes]
        )[0]

        # Onset age: LHON 20-30y; Leigh infantile; CPEO adult; Exercise adult
        if "Leigh" in pheno_label or "Pearson" in pheno_label:
            onset_weeks = rng.randint(0, 52)
        elif "MELAS" in pheno_label:
            onset_weeks = rng.randint(52, 780)   # 1-15 years
        elif "LHON" in pheno_label:
            # m.3460G>A: 20-30 years typically
            onset_weeks = rng.randint(780, 1560)  # ~15-30 years
        elif "CPEO" in pheno_label or "KSS" in pheno_label:
            onset_weeks = rng.randint(260, 1560)
        elif "Exercise" in pheno_label:
            onset_weeks = rng.randint(780, 2600)  # adult
        else:
            onset_weeks = rng.randint(260, 2080)

        # Visual acuity for LHON patients (logMAR; 0 = normal, 1 = 20/200, 3 = CF)
        if optic_atrophy:
            if lhon_recovery:
                va_logmar = round(rng.uniform(0.0, 0.5), 2)
            else:
                va_logmar = round(rng.uniform(0.8, 3.0), 2)
        else:
            va_logmar = round(rng.uniform(0.0, 0.2), 2)

        patients.append({
            "patient_id": f"MTND1-{i+1:03d}",
            "phenotype": pheno_label,
            "variant": variant["hgvs_mtdna"],
            "protein_change": variant["protein"],
            "heteroplasmy_blood_pct": heteroplasmy_blood_pct,
            "ci_activity_pct": ci_activity_pct,
            "lactic_acid_mmolL": lactic_acid,
            "onset_weeks": onset_weeks,
            "male_sex": male_sex,
            "optic_atrophy": optic_atrophy,
            "sequential_bilateral": sequential_bilateral,
            "peripapillary_telangiectasia": peripapillary_telangiectasia,
            "red_green_color_loss": red_green_color_loss,
            "lhon_recovery": lhon_recovery,
            "visual_acuity_logMAR": va_logmar,
            "leigh_mri": leigh_mri,
            "lactic_acidosis": lactic_acidosis,
            "hypotonia": hypotonia,
            "developmental_delay": dev_delay,
            "seizures": seizures,
            "cardiomyopathy": cardiomyopathy,
            "cpeo": cpeo,
            "exercise_intolerance": exercise_intolerance,
            "ragged_red_fibres": rrfs,
            "respiratory_failure": respiratory,
            "encephalopathy": encephalopathy,
            "melas_overlap": melas_overlap,
            "dystonia": dystonia,
            "kss_cpeo": kss_cpeo,
            "lhon": optic_atrophy,
            "outcome": outcome_label,
        })

    return patients


# ── Cohort statistics ─────────────────────────────────────────────────────────
def _cohort_stats(patients):
    n = len(patients)
    pct = lambda key: round(100 * sum(1 for p in patients if p.get(key, False)) / n, 1)
    avg = lambda key: round(sum(p[key] for p in patients) / n, 1)
    lhon_patients = [p for p in patients if "LHON" in p["phenotype"]]
    lhon_n = len(lhon_patients)
    return {
        "optic_atrophy_pct":                pct("optic_atrophy"),
        "sequential_bilateral_pct":         pct("sequential_bilateral"),
        "peripapillary_telangiectasia_pct": pct("peripapillary_telangiectasia"),
        "red_green_color_loss_pct":         pct("red_green_color_loss"),
        "lhon_recovery_pct":                pct("lhon_recovery"),
        "lhon_phenotype_pct":               round(100 * lhon_n / n, 1),
        "leigh_mri_pct":                    pct("leigh_mri"),
        "lactic_acidosis_pct":              pct("lactic_acidosis"),
        "hypotonia_pct":                    pct("hypotonia"),
        "developmental_delay_pct":          pct("developmental_delay"),
        "seizures_pct":                     pct("seizures"),
        "cardiomyopathy_pct":               pct("cardiomyopathy"),
        "cpeo_pct":                         pct("cpeo"),
        "exercise_intolerance_pct":         pct("exercise_intolerance"),
        "ragged_red_fibres_pct":            pct("ragged_red_fibres"),
        "respiratory_failure_pct":          pct("respiratory_failure"),
        "encephalopathy_pct":               pct("encephalopathy"),
        "melas_overlap_pct":                pct("melas_overlap"),
        "dystonia_pct":                     pct("dystonia"),
        "kss_cpeo_pct":                     pct("kss_cpeo"),
        "avg_ci_activity_pct":              avg("ci_activity_pct"),
        "avg_lactic_acid_mmolL":            avg("lactic_acid_mmolL"),
        "avg_heteroplasmy_blood_pct":       avg("heteroplasmy_blood_pct"),
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
        ("Optic atrophy (LHON)", "optic_atrophy"),
        ("Sequential bilateral visual loss", "sequential_bilateral"),
        ("Peripapillary telangiectasia (NO FFA leak)", "peripapillary_telangiectasia"),
        ("Red-green dyschromatopsia", "red_green_color_loss"),
        ("LHON spontaneous visual recovery (m.3460G>A 22%)", "lhon_recovery"),
        ("Leigh MRI (bilateral BG/brainstem)", "leigh_mri"),
        ("Lactic acidosis", "lactic_acidosis"),
        ("Hypotonia", "hypotonia"),
        ("Developmental delay / regression", "developmental_delay"),
        ("Seizures", "seizures"),
        ("Cardiomyopathy", "cardiomyopathy"),
        ("CPEO (progressive external ophthalmoplegia)", "cpeo"),
        ("Exercise intolerance", "exercise_intolerance"),
        ("Ragged-red fibres (RRF, Gomori)", "ragged_red_fibres"),
        ("Respiratory failure", "respiratory_failure"),
        ("Encephalopathy", "encephalopathy"),
        ("MELAS-like stroke overlap", "melas_overlap"),
        ("Dystonia", "dystonia"),
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
    top_variants = [{"variant": k, "count": v} for k, v in v_counter.most_common(6)]

    alerts = [
        "🔵 LHON PHENOTYPE: MT-ND1 m.3460G>A — #2 primary LHON worldwide (15%); INTERMEDIATE spontaneous visual recovery (22%): better than ND4 (<4%) but less than ND6 (50%); idebenone 900 mg/day first-line within 5-12 months; do NOT assume permanent blindness",
        "🔵 PROXIMAL MEMBRANE ARM / ND1-MODULE: MT-ND1 is the MOST CONSERVED CI subunit — 8 TM helices — ND1-MODULE CORE ANCHOR — first P-module sub-assembly in CI biogenesis; mutations disrupt entire proximal arm nucleation",
        "🔵 MALE PREDOMINANCE 80–90% (LHON): X-linked penetrance modifier; female carriers often asymptomatic; screen ALL maternal relatives regardless of sex",
        "⚠️ LEIGH SYNDROME: heteroplasmy ≥70% → bilateral symmetric BG/brainstem T2 (85%); high-heteroplasmy non-LHON variants (m.3697G>A, m.3890G>A); CI 5-25% residual; GIR 6-8 mg/kg/min MANDATORY; NEVER fast",
        "⚠️ HETEROPLASMY THRESHOLD: <60% heteroplasmy → typically no phenotype; 60-70% → exercise intolerance / MELAS-like; ≥70% → Leigh syndrome; ≥90% → LHON (m.3460G>A near-homoplasmic)",
        "⚠️ PERIPAPILLARY TELANGIECTASIA — NO FFA LEAK: KEY DDx from NAION (sudden disc oedema + FFA leak present) and OPA1 (insidious blue-yellow dyschromatopsia, no telangiectasia)",
        "⚠️ WES MISSES MT-ND1: dedicated mtDNA H-strand sequencing required (rCRS 3307–4262); confirm adequate coverage in QC report; blood-based heteroplasmy may underestimate tissue heteroplasmy",
        "🚫 Tobacco — ABSOLUTE CONTRAINDICATION: #1 environmental LHON trigger; cyanide + CO in cigarette smoke are CI inhibitors; doubles CI inhibition; mandatory cessation counselling in ALL carriers",
        "🚫 Ethambutol — ABSOLUTE CONTRAINDICATION in ALL MT-ND1 LHON genotypes: direct optic nerve toxin via CI inhibition; synergistic optic neuropathy; use rifampicin-isoniazid-pyrazinamide for TB instead",
        "🚫 Metformin — ABSOLUTE CONTRAINDICATION: Complex I inhibitor; fatal lactic acidosis in Leigh/MELAS-overlap; contraindicated across all MT-ND1 phenotypes",
        "🚫 Valproic acid (VPA) — ABSOLUTE CONTRAINDICATION: CoA sequestration + POLG inhibition; especially toxic in Leigh/MELAS-overlap; avoid in all MT-ND1",
        "🚫 Linezolid — ABSOLUTE CONTRAINDICATION: inhibits mt-23S rRNA → reduces all mt-encoded CI subunit synthesis including MT-ND1",
        "🚫 Propofol — ABSOLUTE CONTRAINDICATION: PRIS — direct CI inhibition; fatal in Leigh/MELAS; use sevoflurane for anaesthesia",
        "🚫 Chloramphenicol — ABSOLUTE CONTRAINDICATION: inhibits mt-70S ribosome; reduces all mt-encoded protein synthesis including ND1",
        "✅ Idebenone (Raxone) 900 mg/day — Level B evidence (RHODOS trial 2011, Klopstock Brain); covers all 3 primary LHON mutations including m.3460G>A; early treatment within 5-12 months onset",
        "✅ Thiamine (B1) MANDATORY (Leigh/MELAS) — PDH cofactor; 10-20 mg/kg IV in acute Leigh decompensation; empiric in all OXPHOS presentations",
        "✅ GIR 6–8 mg/kg/min (NEVER fast) — mandatory in Leigh/MELAS-overlap; fasting triggers CI crisis; continuous dextrose",
        "✅ LEV preferred AED — renal clearance, no mitochondrial toxicity; avoid VPA in ALL MT-ND1 phenotypes",
        "✅ CoQ10 / Ubiquinol + Riboflavin (B2) — Level C; empiric mitochondrial cocktail across all phenotypes",
    ]

    return {
        "gene": "MT-ND1",
        "full_name": "Mitochondrially Encoded NADH:Ubiquinone Oxidoreductase Core Subunit 1",
        "alias": "NADH:Ubiquinone Oxidoreductase Core Subunit 1 / ND1",
        "omim_gene": "516000",
        "omim_disease": "LHON (Leber's Hereditary Optic Neuropathy) #2 Worldwide + Leigh Syndrome (mtDNA CI deficiency) — OMIM *516000",
        "disease_name": "Isolated Complex I Deficiency — LHON #2 Worldwide (intermediate recovery 22%) + Leigh Syndrome + Leigh/MELAS Overlap (MT-ND1 proximal membrane arm ND1-module defect)",
        "chromosome": "mtDNA H-strand rCRS m.3307-4262",
        "inheritance": "MATERNAL — mtDNA; heteroplasmic (LHON) / near-homoplasmic (Leigh)",
        "protein_size": "318 aa / 36 kDa / 8 TM helices",
        "protein": {
            "size_aa": 318,
            "kDa": 36,
            "tm_helices": 8,
            "localization": "Integral IMM — proximal membrane arm (P-module); ND1-MODULE CORE ANCHOR; interacts with ND3 (N-module junction), NDUFB7, NDUFB8, NDUFB9, NDUFB10, NDUFB11",
            "role": "Core anchor of the proximal membrane arm (ND1-module); FIRST P-module sub-assembly nucleated; interfaces N-module (soluble arm) at matrix; most conserved CI subunit across eukaryotes",
            "function": "One of 7 mtDNA-encoded CI subunits; 8 TM helices; MOST CONSERVED CI SUBUNIT; forms proximal arm proton half-channel; LHON variants (near-homoplasmic) subtly alter CI conformation → ↑ ROS → RGC axon death; Leigh variants (high heteroplasmy) → complete proximal arm collapse → isolated CI deficiency",
        },
        "ci_assembly_context": "ND1 incorporated as the FIRST and most central subunit of the P-module (proximal membrane arm); ND1-module (ND1 + NDUFB subunits) assembles before middle (ND2) and distal (ND4-ND4L-ND5-ND6) arms; ND1 loss → entire proximal arm fails to form; BN-PAGE: reduced CI holoenzyme + I+III₂ supercomplex; CI 35-55% in LHON, 5-25% in Leigh; CII/CIII/CIV normal (isolated CI deficiency)",
        "heteroplasmy_thresholds": {
            "Near-homoplasmic (>90% blood)": "LHON optic atrophy (m.3460G>A); incomplete penetrance (50-60% males; 10-15% females)",
            "High (70–95% blood)": "Leigh syndrome (m.3697G>A, m.3890G>A, m.4171C>A); bilateral BG/brainstem T2; CI 5-25%",
            "Moderate (50–70% blood)": "Exercise intolerance myopathy (m.4171C>A) / MELAS-like (m.3376G>A); CI 30-55%",
            "Variable (20–65% blood)": "Large deletion syndromes (KSS/Pearson/CPEO); tissue-specific heteroplasmy",
        },
        "bn_page_pattern": "LHON (m.3460G>A): mildly-moderately reduced CI holoenzyme and I+III₂ supercomplex (CI 35-55% residual; intermediate severity between ND4 30-50% and ND6 55-78%); Leigh variants: severely reduced CI (5-25% residual); CII/CIII/CIV typically normal (isolated CI deficiency)",
        "lhon_specific_features": [
            f"Optic atrophy (LHON): {stats['optic_atrophy_pct']}% of cohort",
            f"Sequential bilateral involvement: {stats['sequential_bilateral_pct']}%",
            f"Peripapillary telangiectatic microangiopathy (NO FFA leak): {stats['peripapillary_telangiectasia_pct']}%",
            f"Red-green dyschromatopsia (colour vision loss): {stats['red_green_color_loss_pct']}%",
            f"LHON spontaneous visual recovery (m.3460G>A cohort): {stats['lhon_recovery_pct']}% (target 22%)",
            "m.3460G>A: LHON #2 worldwide (~15% of all LHON); INTERMEDIATE onset 20-30 years",
            "m.3460G>A: INTERMEDIATE prognosis — 22% spontaneous recovery (vs <4% MT-ND4; 50% MT-ND6)",
            "Male predominance 80-90%: X-linked modifier; female incomplete penetrance 10-15%",
            "Tobacco / Ethambutol: ABSOLUTE CI in ALL MT-ND1 LHON genotypes",
        ],
        "key_molecular_features": [
            "MT-ND1: MOST CONSERVED CI subunit across all eukaryotes — 8 TM helices",
            "ND1-MODULE: proximal membrane arm core anchor — FIRST P-module assembled in CI biogenesis",
            "H-strand encoded (rCRS 3307-4262) — unlike MT-ND6 (L-strand); WES misses all mtDNA variants",
            "CI residual in LHON (35-55%): INTERMEDIATE between ND4 LHON (30-50%) and ND6 LHON (55-78%)",
            "Heteroplasmy threshold: >90% → LHON; ≥70% → Leigh; 50-70% → exercise intolerance/MELAS-like",
            "ND1 contacts N-module (peripheral arm) at matrix — largest CI subunit / N-module interface",
        ],
        "cohort_n": n,
        "seed": SEED,
        "patients": patients[:10],
        "cohort_statistics": stats,
        "cohort_summary_features": features,
        "key_clinical_alerts": alerts,
        "top_variant_counts": top_variants,
        "phenotype_distribution": _phenotype_dist(patients),
        "onset_distribution": {
            "neonatal_pct": round(100 * sum(1 for p in patients if p["onset_weeks"] <= 4) / n, 1),
            "infantile_pct": round(100 * sum(1 for p in patients if 5 <= p["onset_weeks"] <= 52) / n, 1),
            "childhood_juvenile_pct": round(100 * sum(1 for p in patients if 53 <= p["onset_weeks"] <= 520) / n, 1),
            "young_adult_pct": round(100 * sum(1 for p in patients if 521 <= p["onset_weeks"] <= 1560) / n, 1),
            "adult_pct": round(100 * sum(1 for p in patients if p["onset_weeks"] > 1560) / n, 1),
        },
        "lhon_vs_leigh_summary": {
            "lhon_pct": stats["lhon_phenotype_pct"],
            "leigh_mri_pct": stats["leigh_mri_pct"],
            "avg_ci_activity_pct": stats["avg_ci_activity_pct"],
            "lhon_recovery_pct": stats["lhon_recovery_pct"],
            "deceased_pct": stats["deceased_pct"],
        },
    }


def _phenotype_dist(patients):
    from collections import Counter
    n = len(patients)
    c = Counter(p["phenotype"] for p in patients)
    return [{"phenotype": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in c.most_common()]


def _rng_count(rate, n=40):
    return round(rate * n)


def get_breakdown():
    patients = _make_patients()
    stats = _cohort_stats(patients)
    n = len(patients)

    from collections import Counter
    vc = Counter(p["variant"] for p in patients)
    variant_dist = [
        {"variant": k, "count": v, "freq_pct": round(100 * v / n, 1)}
        for k, v in vc.most_common()
    ]

    ci_vals = [p["ci_activity_pct"] for p in patients]
    lac_vals = [p["lactic_acid_mmolL"] for p in patients]
    het_vals = [p["heteroplasmy_blood_pct"] for p in patients]

    biochem = {
        "avg_ci_activity_pct":         round(sum(ci_vals) / n, 1),
        "ci_below_25_pct":             round(100 * sum(1 for v in ci_vals if v < 25) / n, 1),
        "ci_25_to_50_pct":             round(100 * sum(1 for v in ci_vals if 25 <= v < 50) / n, 1),
        "ci_50_to_75_pct":             round(100 * sum(1 for v in ci_vals if 50 <= v < 75) / n, 1),
        "ci_above_75_pct":             round(100 * sum(1 for v in ci_vals if v >= 75) / n, 1),
        "avg_lactic_acid_mmolL":       round(sum(lac_vals) / n, 1),
        "lactic_above_5_pct":          round(100 * sum(1 for v in lac_vals if v > 5) / n, 1),
        "lactic_2_to_5_pct":           round(100 * sum(1 for v in lac_vals if 2 <= v <= 5) / n, 1),
        "lactic_normal_below2_pct":    round(100 * sum(1 for v in lac_vals if v < 2) / n, 1),
        "avg_heteroplasmy_blood_pct":  round(sum(het_vals) / n, 1),
        "het_below_70_pct":            round(100 * sum(1 for v in het_vals if v < 70) / n, 1),
        "het_70_to_90_pct":            round(100 * sum(1 for v in het_vals if 70 <= v < 90) / n, 1),
        "het_above_90_pct":            round(100 * sum(1 for v in het_vals if v >= 90) / n, 1),
    }

    oc = Counter(p["outcome"] for p in patients)
    outcomes = [{"outcome": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in oc.most_common()]

    return {
        "gene": "MT-ND1",
        "all_variants": VARIANTS,
        "variant_distribution": variant_dist,
        "phenotype_distribution": _phenotype_dist(patients),
        "biochemistry_distribution": biochem,
        "outcome_distribution": outcomes,
        "cohort_statistics": stats,
        "lhon_vs_other_comparison": {
            "gene": "MT-ND1 vs primary LHON peers",
            "mt_nd4_m11778": {
                "rank": "#1 worldwide (~70% LHON)", "ci_residual": "30–50%",
                "recovery_pct": "<4% (WORST)", "onset_age": "20–35 years",
                "gene_therapy": "EU-approved Lumevoq (AAV2-MT-ND4 2021, m.11778G>A only)"
            },
            "mt_nd1_m3460": {
                "rank": "#2 worldwide (~15% LHON)", "ci_residual": "35–55% (INTERMEDIATE)",
                "recovery_pct": "22% (INTERMEDIATE)", "onset_age": "20–30 years",
                "gene_therapy": "Phase I/II trials ongoing (no approved therapy yet)"
            },
            "mt_nd6_m14484": {
                "rank": "#3 worldwide (~14% LHON)", "ci_residual": "55–78% (MILDEST CI reduction)",
                "recovery_pct": "50% (BEST of 3 primaries)", "onset_age": "15–25 years (youngest)",
                "gene_therapy": "Earlier-stage development vs ND4"
            },
        },
        "bn_page_pattern": {
            "finding": "LHON (m.3460G>A): mildly-moderately reduced CI and I+III₂ supercomplex (CI 35-55% residual; intermediate between ND4 and ND6); Leigh variants: severely reduced CI (5-25%); CII/CIII/CIV normal in both",
            "interpretation": "ND1 LHON causes moderate CI assembly/stability defect (not complete loss); the 35-55% CI residual activity in m.3460G>A explains intermediate recovery (22%) — better than ND4 (<4%) but worse than ND6 (50%); ND1-module position (proximal arm) more central than ND6 tip but less critical than ND4 central antiporter TM11",
            "ddx_value": "Isolated CI + maternal inheritance + heteroplasmy → mtDNA CI gene; ND1 (H-strand rCRS 3307-4262) explicitly sequenced; DDx from NAION by NO FFA leak + telangiectasia; DDx OPA1 by maternal inheritance + male predominance + red-green (not blue-yellow) dyschromatopsia; DDx ND4/ND6 by recovery prognosis",
        },
        "immunoblot_pattern": {
            "MT-ND1 (ND1)": "Absent or reduced (primary loss; proportional to variant severity + heteroplasmy)",
            "NDUFB7/NDUFB8 (ND1-module partners)": "Secondarily reduced (ND1 loss destabilises entire P-module sub-assembly)",
            "MT-ND3 (ND3)": "Secondarily reduced (N-module/membrane arm junction contacts ND1-module)",
            "MT-ND5 (ND5)": "Variable secondary reduction in severe ND1 loss (distal arm depends on P-module foundation)",
            "CI N-module (NDUFS1/NDUFS2)": "Normal in LHON variants; secondarily reduced in severe Leigh",
            "CII (SDHA)": "Normal — isolated CI deficiency",
            "CIII (UQCRC1)": "Normal — isolated CI deficiency",
            "CIV (COX4I1)": "Normal — isolated CI deficiency",
        },
        "treatment_uptake": {
            "Idebenone (Raxone) 900 mg/day (LHON)":     f"{_rng_count(0.68)} / {n} patients",
            "Tobacco cessation counselling":              f"{_rng_count(0.96)} / {n} patients",
            "Ethambutol avoidance documented":           f"{_rng_count(0.90)} / {n} patients",
            "Thiamine (B1) IV/oral (Leigh/MELAS)":       f"{_rng_count(0.45)} / {n} patients",
            "GIR 6–8 mg/kg/min (Leigh/MELAS)":           f"{_rng_count(0.38)} / {n} patients",
            "CoQ10 / Ubiquinol (all phenotypes)":        f"{_rng_count(0.70)} / {n} patients",
            "Riboflavin (B2)":                            f"{_rng_count(0.58)} / {n} patients",
            "LEV (preferred AED, seizure subset)":       f"{_rng_count(0.22)} / {n} patients",
            "Low-vision rehabilitation (LHON)":          f"{_rng_count(0.55)} / {n} patients",
            "Ophthalmology + OCT monitoring":             f"{_rng_count(0.85)} / {n} patients",
        },
        "ddx_table": [
            {
                "condition": "MT-ND4 LHON (m.11778G>A)",
                "key_distinction": "MT-ND1: INTERMEDIATE recovery (22%); central antiporter TM11. MT-ND4: WORST recovery (<4%); central arm TM11; #1 worldwide 70%; gene therapy available (Lumevoq EU 2021)",
                "shared": "Maternal LHON, near-homoplasmic, male predominance, optic atrophy, peripapillary telangiectasia, idebenone treatment",
            },
            {
                "condition": "MT-ND6 LHON (m.14484T>C)",
                "key_distinction": "MT-ND1: 22% recovery, H-strand, proximal arm, 20-30y onset. MT-ND6: BEST 50% recovery, L-strand (ONLY L-strand CI subunit), distal arm TIP, teens onset, French-Canadian founder",
                "shared": "Maternal LHON, incomplete penetrance, male predominance, idebenone treatment",
            },
            {
                "condition": "NAION (Non-arteritic anterior ischaemic optic neuropathy)",
                "key_distinction": "LHON: NO FFA leak (telangiectasia without leak is PATHOGNOMONIC); sequential bilateral (97%); maternal family history; young onset; NAION: sudden UNILATERAL disc oedema + FFA leak; older age (>50y); disc-at-risk; bilateral <15%",
                "shared": "Acute optic neuropathy, central visual loss",
            },
            {
                "condition": "OPA1 (Autosomal dominant optic atrophy)",
                "key_distinction": "LHON: RED-GREEN dyschromatopsia; MATERNAL inheritance; peripapillary telangiectasia; acute subacute onset. OPA1: BLUE-YELLOW dyschromatopsia; AUTOSOMAL DOMINANT; insidious childhood onset; temporal disc pallor, no telangiectasia",
                "shared": "Optic atrophy, visual loss, young onset",
            },
            {
                "condition": "Nuclear CI deficiency (NDUFS1/NDUFS2/NDUFV1/NDUFAF2 etc.)",
                "key_distinction": "MT-ND1: MATERNAL inheritance; mtDNA H-strand; WES misses. Nuclear CI: AUTOSOMAL RECESSIVE; nuclear DNA; WES-diagnosable; biallelic mutations; no heteroplasmy",
                "shared": "Isolated CI deficiency, Leigh syndrome, lactic acidosis",
            },
        ],
    }


def get_definitions():
    return {
        "gene": "MT-ND1",
        "full_name": "Mitochondrially Encoded NADH:Ubiquinone Oxidoreductase Core Subunit 1",
        "alias": "NADH:Ubiquinone Oxidoreductase Core Subunit 1 / ND1",
        "omim_gene": "516000",
        "disease_name": "LHON #2 Worldwide + Leigh Syndrome + Leigh/MELAS Overlap (MT-ND1 proximal membrane arm ND1-module defect)",
        "chromosome": "mtDNA H-strand rCRS m.3307-4262",
        "protein_aa": 318,
        "protein_kDa": 36,
        "tm_helices": 8,
        "inheritance": "MATERNAL — mtDNA; heteroplasmic (LHON) or high heteroplasmy (Leigh)",
        "key_concepts": {
            "MOST CONSERVED CI subunit": "MT-ND1 is the most evolutionarily conserved CI subunit across all eukaryotes. Its 8 TM helices form the proximal membrane arm core. Conservation reflects critical structural role: ND1 is the anchor point of the entire P-module (proximal arm) from which the rest of the membrane arm is built.",
            "ND1-MODULE / Proximal Membrane Arm": "The ND1-module is the FIRST sub-assembly in CI biogenesis. ND1 anchors NDUFB7, NDUFB8, NDUFB9, NDUFB10, NDUFB11 and additional subunits to form the proximal arm. This P-module nucleates BEFORE the middle (ND2-module) and distal (ND4/ND5/ND6 antiporter) modules. Loss of ND1 → entire proximal arm fails.",
            "LHON #2 primary worldwide": "m.3460G>A (p.Ala52Thr) accounts for ~15% of LHON globally (#2 after m.11778G>A ND4 ~70%). INTERMEDIATE spontaneous visual recovery: 22% (better than ND4 <4%, worse than ND6 50%). CI residual 35-55% — intermediate severity reflects proximal arm central position.",
            "Intermediate spontaneous visual recovery (22%)": "m.3460G>A has 22% spontaneous visual recovery — intermediate among primary LHON. The moderate CI residual (35-55%) in ND1 LHON (more severe than ND6 55-78%, less severe than ND4 30-50%) explains intermediate recovery rate. Recovery most often within 6-12 months of onset.",
            "Heteroplasmy threshold — LHON vs Leigh": "Below 60% heteroplasmy: typically no phenotype. 60-70%: exercise intolerance / MELAS-like. ≥70%: Leigh syndrome (bilateral BG/brainstem T2). ≥90%: LHON (m.3460G>A near-homoplasmic). This threshold differs from ND6 where near-homoplasmic is required for ALL LHON phenotypes.",
            "Peripapillary telangiectasia — NO FFA leak": "Hallmark LHON finding: dilated tortuous peripapillary capillaries WITHOUT fluorescein angiography leak. Present in ~88% of LHON before visual loss. KEY DDx from NAION (FFA leak) and OPA1 (no telangiectasia). Red-green (not blue-yellow) dyschromatopsia: DDx from OPA1.",
            "WES misses MT-ND1": "MT-ND1 is mtDNA-encoded (H-strand rCRS 3307-4262). Standard WES is designed for nuclear DNA (nDNA) and routinely misses mtDNA variants. Dedicated mtDNA sequencing panel required; confirm adequate H-strand coverage in NGS QC report.",
            "Male predominance in LHON": "LHON affects males 80-90% despite maternal inheritance. X-linked modifier increases male susceptibility. Incomplete penetrance: 50-60% of males, 10-15% of females with m.3460G>A develop LHON. All maternal relatives at risk regardless of sex.",
            "Tobacco / Ethambutol — absolute CI": "Tobacco: #1 environmental LHON trigger (cyanide + CO are CI inhibitors); ABSOLUTE CI in ALL carriers including pre-symptomatic. Ethambutol: direct optic nerve toxin via CI inhibition; ABSOLUTE CI in ALL MT-ND1 LHON genotypes; use rifampicin-isoniazid-pyrazinamide for TB if needed.",
        },
        "contraindications": [
            {"drug": "Tobacco", "severity": "ABSOLUTE CI", "reason": "Cyanide + CO in cigarette smoke are CI inhibitors; #1 environmental LHON conversion trigger; ABSOLUTE CI in all MT-ND1 carriers including pre-symptomatic; also toxic in Leigh via CI inhibition"},
            {"drug": "Ethambutol", "severity": "ABSOLUTE CI", "reason": "Direct optic nerve toxin via CI inhibition; synergistic optic neuropathy in ALL MT-ND1 LHON genotypes; ABSOLUTE CI regardless of vision status; use rifampicin-isoniazid-pyrazinamide for TB"},
            {"drug": "Metformin", "severity": "ABSOLUTE CI", "reason": "Complex I inhibitor; fatal lactic acidosis in Leigh/MELAS-overlap; contraindicated across all MT-ND1 phenotypes"},
            {"drug": "Valproic acid (VPA)", "severity": "ABSOLUTE CI", "reason": "CoA sequestration + POLG inhibition; especially toxic in Leigh/MELAS-overlap; avoid in all MT-ND1"},
            {"drug": "Linezolid", "severity": "ABSOLUTE CI", "reason": "Inhibits mt-23S rRNA → reduces all mt-encoded CI subunit synthesis including MT-ND1; worsens CI deficiency"},
            {"drug": "Propofol", "severity": "ABSOLUTE CI", "reason": "PRIS — direct CI inhibition; fatal in Leigh/MELAS-overlap; use sevoflurane for all anaesthesia"},
            {"drug": "Chloramphenicol", "severity": "ABSOLUTE CI", "reason": "Inhibits mt-70S ribosome; reduces all mt-encoded protein synthesis including ND1; avoid in all phenotypes"},
            {"drug": "Alcohol", "severity": "AVOID (strong)", "reason": "Acetaldehyde is a CI inhibitor; documented environmental LHON trigger; accelerates RGC axon loss in carriers"},
            {"drug": "Amiodarone", "severity": "AVOID (strong)", "reason": "DION (drug-induced optic neuropathy) + CI inhibitor synergism; compounding optic nerve toxicity in LHON genotypes"},
        ],
        "treatments": [
            {"drug": "Idebenone (Raxone) 900 mg/day", "level": "Level B (RHODOS trial)", "rationale": "Short-chain CoQ analog; bypasses CI to CII→CIII→CIV; covers all 3 primary LHON mutations including m.3460G>A; early treatment within 5-12 months onset yields best response"},
            {"drug": "Tobacco cessation (mandatory)", "level": "Standard of care", "rationale": "Most modifiable risk factor; smoking is #1 environmental LHON trigger; cessation may prevent visual conversion in pre-symptomatic carriers; cessation counselling at every visit"},
            {"drug": "Low-vision rehabilitation", "level": "Standard of care (LHON)", "rationale": "Eccentric viewing training; magnification aids; social/occupational rehabilitation; registered blind status if VA <6/60"},
            {"drug": "Thiamine (B1) IV/oral", "level": "Mandatory (Leigh/MELAS)", "rationale": "PDH cofactor; 10-20 mg/kg IV in acute Leigh decompensation; empiric in all OXPHOS presentations; biotin added empirically"},
            {"drug": "GIR 6–8 mg/kg/min dextrose", "level": "Mandatory (Leigh/MELAS)", "rationale": "NEVER fast; continuous dextrose prevents CI crisis in Leigh and MELAS-overlap phenotypes"},
            {"drug": "CoQ10 / Ubiquinol + Riboflavin (B2)", "level": "Level C", "rationale": "Mitochondrial cocktail; FAD-dependent CI assembly co-factors; empiric across all phenotypes"},
            {"drug": "LEV (Levetiracetam)", "level": "Preferred AED (seizure subset)", "rationale": "Renal clearance; no mitochondrial toxicity; preferred over VPA/CBZ in all MT-ND1 phenotypes"},
            {"drug": "Sevoflurane (anaesthesia)", "level": "Preferred over propofol", "rationale": "Avoid propofol (PRIS); sevoflurane is safe; plan anaesthesia with mito-anaesthesia protocol"},
        ],
        "diagnostic_workup": [
            "Dedicated mtDNA sequencing panel (blood) — NOT WES; confirm adequate H-strand coverage (rCRS 3307-4262) in QC report; may need muscle biopsy mtDNA if blood heteroplasmy underestimates",
            "Fundoscopy: peripapillary telangiectatic microangiopathy (no FFA leak — KEY DDx from NAION)",
            "Fluorescein angiography (FFA): ABSENT leak = LHON (vs NAION: leak present); perform before visual loss if possible",
            "OCT (optical coherence tomography): RNFL (retinal nerve fibre layer) — initial RNFL thickening (peripapillary oedema), later RNFL thinning (atrophy); temporal sector prominent",
            "VEP (visual evoked potential): prolonged P100 latency; reduced amplitude; useful pre-symptomatic monitoring",
            "Plasma lactate + pyruvate + L:P ratio: typically NORMAL in pure LHON (m.3460G>A); elevated in Leigh/MELAS-overlap",
            "Plasma amino acids (elevated alanine): marker of PDH inhibition; elevated in Leigh/MELAS not pure LHON",
            "Brain MRI: typically NORMAL in pure LHON; bilateral symmetric BG/brainstem T2 hyperintensity in Leigh (85%); stroke-like lesions in MELAS-overlap",
            "CI/CII/CIII/CIV activities (muscle or fibroblasts): isolated CI deficiency; CI 35-55% in LHON, 5-25% in Leigh",
            "BN-PAGE: mildly-moderately reduced CI + I+III₂ supercomplex in LHON; severely reduced in Leigh; CII/CIII/CIV normal",
            "Immunoblot: ND1 reduced/absent (primary loss); secondary NDUFB7/NDUFB8 reduction (P-module partners); CI N-module may be normal in LHON",
            "Annual Holter ECG (KSS/large deletion): heart block risk mandatory monitoring",
            "Maternal family testing: screen maternal relatives; genetic counselling (incomplete penetrance; male risk 50-60%; female risk 10-15%)",
        ],
        "references": [
            {
                "citation": "Huoponen K et al. (1991) A new mtDNA mutation associated with Leber hereditary optic neuroretinopathy. Am J Hum Genet 48(6):1147-53.",
                "relevance": "First description of m.3460G>A (p.Ala52Thr) in MT-ND1 — discovery paper for the LHON #2 primary worldwide mutation; established ND1 as a LHON gene",
            },
            {
                "citation": "Klopstock T et al. (2011) A randomized placebo-controlled trial of idebenone in Leber's hereditary optic neuropathy. Brain 134(Pt 9):2677-86.",
                "relevance": "RHODOS trial — Level B evidence for idebenone across all 3 primary LHON mutations including m.3460G>A (MT-ND1); discordant vision patients benefitted most",
            },
            {
                "citation": "Agip AA et al. (2018) Cryo-EM structures of complex I from mouse heart mitochondria in two biochemical states. Nat Struct Mol Biol 25(7):548-556.",
                "relevance": "High-resolution CI cryo-EM; ND1 proximal membrane arm (P-module) position confirmed; ND1 as the MOST CONSERVED CI subunit and core anchor of the ND1-module shown definitively",
            },
            {
                "citation": "Jun AS et al. (1994) A mitochondrial DNA mutation at nucleotide pair 3460 and a mutation at nucleotide pair 11778 share the mtDNA background of a large French-Canadian family. Biochem Biophys Res Commun 200(2):1062-9.",
                "relevance": "Comparative analysis of m.3460G>A (MT-ND1) and m.11778G>A (MT-ND4) in LHON; established CI assembly context and intermediate prognosis of ND1 vs ND4",
            },
            {
                "citation": "Wallace DC et al. (1988) Mitochondrial DNA mutation associated with Leber's hereditary optic neuropathy. Science 242(4884):1427-30.",
                "relevance": "Seminal CI-LHON discovery paper (m.11778G>A ND4); established mtDNA CI mutations as LHON mechanism; foundational context for ND1 m.3460G>A LHON",
            },
        ],
        "cohort_seed": SEED,
        "n_patients": N_PATIENTS,
    }
