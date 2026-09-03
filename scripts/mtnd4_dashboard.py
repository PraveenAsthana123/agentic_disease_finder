#!/usr/bin/env python3
"""MT-ND4 — Mitochondrially Encoded NADH:Ubiquinone Oxidoreductase Core Subunit 4 /
Isolated Complex I Deficiency + LHON (Leber Hereditary Optic Neuropathy) / Leigh Syndrome
Central Antiporter Module — DUAL PHENOTYPE — Maternal Inheritance.

MT-ND4 (OMIM *516003) encodes the 459-amino-acid, 51.7 kDa NADH dehydrogenase subunit 4
of Complex I (NADH:Ubiquinone Oxidoreductase), the CENTRAL antiporter module subunit among
the 7 mtDNA-encoded CI subunits. ND4 forms the central proton-pumping antiporter repeat
within the CI membrane arm, flanked by ND2 (proximal) and ND5 (distal).

MT-ND4 is UNIQUE among CI subunits in mediating TWO entirely distinct clinical syndromes
determined by tissue-specific heteroplasmy:
  1. NEAR-HOMOPLASMIC / LOW HETEROPLASMY in optic tissue → LHON (Leber Hereditary Optic
     Neuropathy): the m.11778G>A (p.Arg340His) mutation in MT-ND4 is THE most common
     mtDNA pathogenic variant worldwide (~70% of all LHON globally).
  2. HIGH HETEROPLASMY (>80%) → Isolated Complex I Deficiency → Leigh syndrome / Leigh-
     MELAS overlap, indistinguishable from other mtDNA CI subunit defects.

  MT-ND4 gene        OMIM *516003
  Disease            LHON (primary LHON mutation #1 worldwide) + Isolated CI Deficiency
                     (Leigh syndrome / MELAS overlap) — dual phenotype gene
  Protein            459 aa, 51.7 kDa; 13 TM helices; central antiporter module
  Genome             Mitochondrial DNA (mtDNA), positions 10760–12137 (rCRS, H-strand)
  Inheritance        MATERNAL — mtDNA; near-homoplasmic in LHON; heteroplasmic in Leigh
  Phenotype          LHON: subacute bilateral optic atrophy, young males 80–90%, worst
                     prognosis of the 3 primary LHON mutations (<4% spontaneous recovery);
                     Leigh/CI: lactic acidosis, hypotonia, developmental regression,
                     bilateral BG/brainstem MRI — heteroplasmy threshold governs phenotype

UNIQUE MOLECULAR POSITION — CENTRAL ANTIPORTER MODULE / DUAL PHENOTYPE:
  MT-ND4 is the CENTRAL of the 3 antiporter repeats in the CI membrane arm:
    ND2 (347 aa) — proximal antiporter module
    ND4 (459 aa) — CENTRAL antiporter module (13 TM helices); ND4 position between ND2
                   and ND5 makes it the structural keystone of proton translocation
    ND5 (603 aa) — distal antiporter module (largest)
  ND4 interacts with ND4L (98 aa; hairpin connector) on the lumenal face and with ND5 via
  the shared TM13/ND5-lateral-helix interface (Agip 2018 Nat Struct Mol Biol).
  LHON mechanism (near-homoplasmic optic tissue):
    m.11778G>A p.Arg340His (TM11): Arg340 coordinates a conserved water-mediated proton-
    relay chain in TM11; Arg→His disrupts the pKa of the proton relay → selective CI
    failure in optic nerve retinal ganglion cell axons (highest CI demand / lowest
    compensatory capacity) → retinal ganglion cell (RGC) apoptosis → optic atrophy.
    Blood mtDNA is near-homoplasmic (>95%) yet asymptomatic until threshold breach in
    retinal tissue. Male predominance (80–90%) is incompletely explained: X-linked modifier
    loci + mitochondrial network architecture differences are implicated.
  LEIGH mechanism (high heteroplasmy):
    Same ND4 loss-of-function → CI collapse across all high-energy tissues (basal ganglia,
    brainstem, muscle) → identical presentation to other mtDNA CI subunit Leigh variants.

HETEROPLASMY THRESHOLD (LHON vs Leigh):
  LHON phenotype: m.11778G>A near-homoplasmic (>95% blood); asymptomatic carriers exist
                  at same heteroplasmy level (incomplete penetrance: 50% lifetime risk males;
                  10% females); environmental triggers (tobacco, alcohol) precipitate onset
  Leigh phenotype: other MT-ND4 variants with >80% heteroplasmy → isolated CI → Leigh

LHON NATURAL HISTORY (m.11778G>A):
  • Onset 15–35 years; first eye: subacute painless central visual loss over days–6 weeks
  • Second eye involvement: 6–8 weeks after first eye in 97% (virtually always bilateral)
  • Nadir vision: typically hand motions / counting fingers; central scotoma; red-green loss
  • Peripapillary telangiectatic microangiopathy: DOES NOT leak on fluorescein angiography
    (KEY DDx vs NAION which leaks on FFA)
  • Spontaneous recovery: <4% with m.11778G>A (vs 22% ND1 m.3460G>A; 50% ND6 m.14484T>C)
  • m.11778G>A has the WORST prognosis of all 3 primary LHON mutations

PATHOGENIC VARIANTS in MT-ND4:
  1. m.11778G>A (p.Arg340His) — TM11, LHON primary; ~65% of MT-ND4 disease; near-homoplasmic;
     optic atrophy young males; <4% spontaneous recovery (Wallace 1988 Science — discovery paper)
  2. m.11696G>A (p.Ser110Asn) — ND4 TM7; Leigh syndrome infantile; ~15%; high heteroplasmy
     >85%; isolated CI deficiency; severe neonatal/infantile presentation
  3. m.10663T>C (p.Val65Ala) — ND4 N-terminal; LHON moderate; ~8%; near-homoplasmic;
     incomplete penetrance; milder visual prognosis than m.11778G>A
  4. m.11253T>C (p.Phe251Leu) — ND4 TM9; Leigh/MELAS overlap; ~7%; high heteroplasmy
     80–90%; CI deficiency + stroke-like episodes
  5. Large ND4-spanning deletion — Pearson/KSS/CPEO; ~3%; heteroplasmic variable
  6. m.12338T>C (ND4 3′ region/ND5 boundary) — ~2%; variable mild CI deficiency

Key Published References:
  Wallace DC et al. (1988) Mitochondrial DNA mutation associated with Leber's hereditary
    optic neuropathy. Science 242:1427-30. (First MT-ND4 m.11778G>A discovery paper)
  Riordan-Eva P & Bhatt DK (2002) Leber's hereditary optic neuropathy. Curr Opin Neurol.
    (LHON natural history; m.11778G>A prognosis; sex bias)
  Klopstock T et al. (2011) A randomized placebo-controlled trial of idebenone in LHON.
    Brain 134(Pt 9):2677-86. (RHODOS trial; idebenone Level B evidence for LHON)
  Vignal-Clermont C et al. (2021) Lenadogene nolparvovec gene therapy for LHON. NEJM.
    (AAV2-MT-ND4; EU conditional approval 2021 for m.11778G>A; gene replacement)
  Agip AA et al. (2018) Cryo-EM structures of complex I from mouse heart mitochondria
    in two biochemical states. Nat Struct Mol Biol 25(7):548-556. (ND4 central antiporter
    structure; ND4-ND5-ND4L interaction; Arg340 proton-relay role)
"""

import random

SEED = 739
N_PATIENTS = 40

# ── Pathogenic / likely-pathogenic variants in MT-ND4 ────────────────────────
VARIANTS = [
    {
        "hgvs_mtdna": "m.11778G>A",
        "protein": "p.Arg340His",
        "domain": "TM helix 11 / central antiporter proton-relay",
        "type": "Missense",
        "severity": "LHON-primary",
        "phenotype": "LHON (subacute optic atrophy, near-homoplasmic)",
        "penetrance_pct": 50,
        "notes": "Most common MT-ND4 pathogenic variant (~65% of MT-ND4 disease); accounts for ~70% of LHON worldwide; Arg340 coordinates water-mediated proton-relay in TM11; near-homoplasmic (>95%) in blood; male predominance 80–90%; <4% spontaneous visual recovery (worst prognosis of 3 primary LHON mutations); onset 15–35 years; Wallace 1988 Science (discovery paper). LHON gene therapy target: lenadogene nolparvovec (Lumevoq, AAV2-MT-ND4) EU conditional approval 2021.",
    },
    {
        "hgvs_mtdna": "m.11696G>A",
        "protein": "p.Ser110Asn",
        "domain": "TM helix 7 / antiporter module mid-domain",
        "type": "Missense",
        "severity": "Severe",
        "phenotype": "Leigh syndrome infantile (high heteroplasmy, isolated CI)",
        "penetrance_pct": 90,
        "notes": "~15% of MT-ND4 disease; Ser110Asn in TM7 disrupts central antiporter module hydrophobic packing; heteroplasmy >85% in blood; severe neonatal/infantile Leigh syndrome; isolated CI deficiency (CI 8–18% residual); indistinguishable from other mtDNA CI Leigh variants; high lactic acidosis 95%; Leigh MRI bilateral BG/brainstem T2 hyperintensity 90%.",
    },
    {
        "hgvs_mtdna": "m.10663T>C",
        "protein": "p.Val65Ala",
        "domain": "ND4 N-terminal / matrix-facing loop",
        "type": "Missense",
        "severity": "LHON-moderate",
        "phenotype": "LHON moderate (near-homoplasmic, incomplete penetrance)",
        "penetrance_pct": 35,
        "notes": "~8%; Val65Ala in the N-terminal region; near-homoplasmic; incomplete penetrance (lower than m.11778G>A); LHON phenotype but milder visual prognosis; higher spontaneous recovery rate than m.11778G>A; some patients asymptomatic despite near-homoplasmic mutation; second LHON primary mutation in MT-ND4.",
    },
    {
        "hgvs_mtdna": "m.11253T>C",
        "protein": "p.Phe251Leu",
        "domain": "TM helix 9 / antiporter core",
        "type": "Missense",
        "severity": "Moderate–Severe",
        "phenotype": "Leigh/MELAS overlap (stroke-like episodes, high heteroplasmy)",
        "penetrance_pct": 80,
        "notes": "~7%; Phe251Leu in TM9 central antiporter core; high heteroplasmy 80–90%; isolated CI deficiency with Leigh MRI and MELAS-like stroke-like episodes (38% in this genotype); similar phenotype to MT-ND5 m.13514A>G but through ND4 central module. CI residual 8–20%.",
    },
    {
        "hgvs_mtdna": "Large ND4-spanning deletion",
        "protein": "Frameshift / deletion",
        "domain": "Entire ND4 or partial deletion",
        "type": "Large deletion",
        "severity": "Variable",
        "phenotype": "Pearson / KSS / CPEO (adult, large mtDNA deletion)",
        "penetrance_pct": 70,
        "notes": "~3%; large-scale mtDNA deletions spanning ND4 (commonly the 'common deletion' 4977 bp or novel deletions); Pearson syndrome (infantile sideroblastic anaemia + exocrine pancreatic failure), Kearns-Sayre syndrome (CPEO+cardiomyopathy+RRF <20y), or CPEO (adult); variable heteroplasmy; distinguish by long-read or Southern blot.",
    },
    {
        "hgvs_mtdna": "m.12338T>C",
        "protein": "ND4 3′ region / ND5 boundary",
        "domain": "ND4 C-terminal / ND4–ND5 junction",
        "type": "Missense",
        "severity": "Mild",
        "phenotype": "Mild CI deficiency / oligosymptomatic variable",
        "penetrance_pct": 30,
        "notes": "~2%; located at the ND4 3′ boundary region (near ND5 start); variable phenotype ranging from oligosymptomatic maternal carriers to mild exercise intolerance and mild CI deficiency; incomplete penetrance; heteroplasmy variable in blood.",
    },
]

VARIANT_WEIGHTS = [0.65, 0.15, 0.08, 0.07, 0.03, 0.02]


# ── Patient-cohort generator ──────────────────────────────────────────────────
def _make_patients(n=N_PATIENTS, seed=SEED):
    rng = random.Random(seed)
    patients = []

    phenotype_distribution = [
        ("LHON (subacute optic atrophy, near-homoplasmic)", 0.40),
        ("Infantile Leigh Syndrome (isolated CI deficiency)", 0.30),
        ("Leigh/MELAS Overlap (stroke-like episodes)", 0.15),
        ("CPEO + Proximal Myopathy (adult, large deletion)", 0.10),
        ("Oligosymptomatic maternal carrier", 0.05),
    ]

    outcomes = [
        ("Deceased < 2 years (Leigh neonatal)", 0.08),
        ("Deceased 2–5 years (Leigh infantile)", 0.07),
        ("Alive — severe visual loss, light perception only", 0.20),
        ("Alive — moderate visual loss, legally blind", 0.20),
        ("Alive — partial visual recovery (idebenone/gene Tx)", 0.12),
        ("Alive — severe disability, ventilator-dependent (Leigh)", 0.10),
        ("Alive — moderate disability, ambulatory with aids (Leigh)", 0.13),
        ("Alive — CPEO only, minimal systemic impact", 0.10),
    ]

    for i in range(n):
        v_idx = rng.choices(range(len(VARIANTS)), weights=VARIANT_WEIGHTS)[0]
        variant = VARIANTS[v_idx]

        # Determine if LHON phenotype or CI/Leigh phenotype based on variant
        is_lhon_variant = variant["phenotype"].startswith("LHON")
        is_leigh_variant = "Leigh" in variant["phenotype"] and not is_lhon_variant
        is_overlap = "MELAS" in variant["phenotype"]
        is_deletion = "deletion" in variant["hgvs_mtdna"]
        is_mild = variant["severity"] == "Mild"

        # Heteroplasmy — LHON near-homoplasmic, Leigh high, others variable
        if is_lhon_variant and variant["hgvs_mtdna"] == "m.11778G>A":
            heteroplasmy_blood_pct = round(rng.uniform(92, 100), 1)
        elif is_lhon_variant:  # m.10663T>C
            heteroplasmy_blood_pct = round(rng.uniform(85, 100), 1)
        elif is_leigh_variant or is_overlap:
            heteroplasmy_blood_pct = round(rng.uniform(78, 95), 1)
        elif is_deletion:
            heteroplasmy_blood_pct = round(rng.uniform(25, 75), 1)
        elif is_mild:
            heteroplasmy_blood_pct = round(rng.uniform(30, 70), 1)
        else:
            heteroplasmy_blood_pct = round(rng.uniform(60, 90), 1)

        # Sex: LHON is 80–90% male; Leigh is ~50/50
        if is_lhon_variant:
            sex = rng.choices(["M", "F"], weights=[0.85, 0.15])[0]
        else:
            sex = rng.choices(["M", "F"], weights=[0.52, 0.48])[0]

        # Phenotype label driven by variant + heteroplasmy
        if is_lhon_variant and variant["hgvs_mtdna"] == "m.11778G>A":
            pheno_label = "LHON (subacute optic atrophy, near-homoplasmic)"
        elif is_lhon_variant:
            pheno_label = rng.choices(
                ["LHON (subacute optic atrophy, near-homoplasmic)",
                 "Oligosymptomatic maternal carrier"],
                weights=[0.65, 0.35],
            )[0]
        elif is_deletion:
            pheno_label = rng.choices(
                ["Pearson Syndrome", "Kearns-Sayre Syndrome", "CPEO + Proximal Myopathy (adult, large deletion)"],
                weights=[0.30, 0.40, 0.30],
            )[0]
        elif is_overlap:
            pheno_label = "Leigh/MELAS Overlap (stroke-like episodes)"
        elif is_leigh_variant:
            if heteroplasmy_blood_pct > 88:
                pheno_label = rng.choices(
                    ["Infantile Leigh Syndrome (isolated CI deficiency)",
                     "Leigh/MELAS Overlap (stroke-like episodes)"],
                    weights=[0.75, 0.25],
                )[0]
            else:
                pheno_label = "Infantile Leigh Syndrome (isolated CI deficiency)"
        elif is_mild:
            pheno_label = "Oligosymptomatic maternal carrier"
        else:
            pheno_label = rng.choices(
                [p[0] for p in phenotype_distribution],
                weights=[p[1] for p in phenotype_distribution],
            )[0]

        # CI activity — LHON has normal/near-normal systemic CI;
        # Leigh has severely reduced CI
        if is_lhon_variant and "LHON" in pheno_label:
            # Systemic CI near-normal in LHON; optic nerve CI selectively vulnerable
            ci_activity_pct = round(rng.uniform(55, 88), 1)
        elif "CPEO" in pheno_label or "KSS" in pheno_label or "Oligosymptomatic" in pheno_label:
            ci_activity_pct = round(rng.uniform(20, 45), 1)
        elif "Overlap" in pheno_label:
            ci_activity_pct = round(rng.uniform(8, 22), 1)
        elif "Infantile Leigh" in pheno_label:
            ci_activity_pct = round(rng.uniform(6, 20), 1)
        elif "Pearson" in pheno_label:
            ci_activity_pct = round(rng.uniform(10, 30), 1)
        else:
            ci_activity_pct = round(rng.uniform(15, 50), 1)

        # Lactic acid — LHON normal; Leigh elevated
        if "LHON" in pheno_label or "Oligosymptomatic" in pheno_label:
            lactic_acid = round(rng.uniform(0.8, 2.2), 1)  # Normal range
        elif ci_activity_pct < 12:
            lactic_acid = round(rng.uniform(7.0, 22.0), 1)
        else:
            lactic_acid = round(rng.uniform(2.5, 10.0), 1)

        # Age of onset (weeks)
        if "LHON" in pheno_label:
            # LHON onset 15–35 years → 780–1820 weeks
            onset_weeks = rng.randint(780, 1820)
        elif "Oligosymptomatic" in pheno_label:
            onset_weeks = rng.randint(1300, 2600)
        elif "Infantile Leigh" in pheno_label:
            onset_weeks = rng.randint(4, 52)
        elif "Overlap" in pheno_label:
            onset_weeks = rng.randint(52, 260)
        elif "CPEO" in pheno_label or "KSS" in pheno_label:
            onset_weeks = rng.randint(260, 1040)
        elif "Pearson" in pheno_label:
            onset_weeks = rng.randint(0, 26)
        else:
            onset_weeks = rng.randint(4, 260)

        # Clinical features — phenotype-specific
        lhon_active = "LHON" in pheno_label

        # LHON features
        optic_atrophy = rng.random() < (0.98 if lhon_active else 0.03)
        central_scotoma = rng.random() < (0.95 if lhon_active else 0.02)
        dyschromatopsia = rng.random() < (0.90 if lhon_active else 0.02)
        peripapillary_microangiopathy = rng.random() < (0.75 if lhon_active else 0.02)

        # Leigh features
        leigh_mri = rng.random() < (
            0.88 if "Leigh" in pheno_label or "Overlap" in pheno_label
            else 0.03 if lhon_active else 0.08
        )
        lactic_acidosis = rng.random() < (
            0.95 if "Leigh" in pheno_label or "Pearson" in pheno_label
            else 0.05 if lhon_active else 0.40
        )
        hypotonia = rng.random() < (
            0.90 if "Leigh" in pheno_label else
            0.05 if lhon_active else 0.25
        )
        dev_delay = rng.random() < (
            0.92 if "Leigh" in pheno_label or "Overlap" in pheno_label
            else 0.03 if lhon_active else 0.20
        )
        seizures = rng.random() < (
            0.50 if "Overlap" in pheno_label else
            0.42 if "Leigh" in pheno_label else 0.05
        )
        stroke_like = rng.random() < (0.38 if "Overlap" in pheno_label else 0.03)
        encephalopathy = rng.random() < (
            0.85 if "Leigh" in pheno_label or "Overlap" in pheno_label
            else 0.05
        )
        respiratory = rng.random() < (
            0.58 if "Infantile Leigh" in pheno_label else
            0.20 if "Overlap" in pheno_label else 0.05
        )
        cardiomyopathy = rng.random() < (
            0.40 if "KSS" in pheno_label else
            0.18 if "Leigh" in pheno_label else 0.03
        )
        spasticity = rng.random() < (
            0.35 if "Leigh" in pheno_label else 0.04
        )

        # CPEO/myopathy features
        cpeo = rng.random() < (
            0.65 if "CPEO" in pheno_label or "KSS" in pheno_label
            else 0.08 if "Overlap" in pheno_label else 0.03
        )
        ptosis = rng.random() < (0.80 if cpeo else 0.06)
        exercise_intolerance = rng.random() < (
            0.85 if "CPEO" in pheno_label or "KSS" in pheno_label
            else 0.55 if "Overlap" in pheno_label else 0.15
        )
        hearing_loss = rng.random() < (
            0.50 if "KSS" in pheno_label or "Overlap" in pheno_label
            else 0.35 if "CPEO" in pheno_label else 0.08
        )
        rrfs = rng.random() < (
            0.65 if "CPEO" in pheno_label or "KSS" in pheno_label
            else 0.25 if "Overlap" in pheno_label else 0.05
        )

        # Gene therapy / idebenone treatment (LHON specific)
        idebenone_rx = rng.random() < (0.55 if lhon_active else 0.08)
        gene_therapy = rng.random() < (
            0.15 if lhon_active and variant["hgvs_mtdna"] == "m.11778G>A" else 0.01
        )

        # Outcome label — LHON vs Leigh different outcomes
        if lhon_active:
            outcome_label = rng.choices(
                ["Alive — severe visual loss, light perception only",
                 "Alive — moderate visual loss, legally blind",
                 "Alive — partial visual recovery (idebenone/gene Tx)",
                 "Alive — bilateral blindness, independent ADLs"],
                weights=[0.35, 0.30, 0.20, 0.15],
            )[0]
        else:
            outcome_label = rng.choices(
                [o[0] for o in outcomes], weights=[o[1] for o in outcomes]
            )[0]

        patients.append({
            "patient_id": f"MTND4-{i+1:03d}",
            "phenotype": pheno_label,
            "variant": variant["hgvs_mtdna"],
            "protein_change": variant["protein"],
            "sex": sex,
            "heteroplasmy_blood_pct": heteroplasmy_blood_pct,
            "ci_activity_pct": ci_activity_pct,
            "lactic_acid_mmolL": lactic_acid,
            "onset_weeks": onset_weeks,
            # LHON features
            "optic_atrophy": optic_atrophy,
            "central_scotoma": central_scotoma,
            "dyschromatopsia_red_green": dyschromatopsia,
            "peripapillary_microangiopathy": peripapillary_microangiopathy,
            "idebenone_rx": idebenone_rx,
            "gene_therapy": gene_therapy,
            # Leigh/CI features
            "leigh_mri": leigh_mri,
            "stroke_like_episodes": stroke_like,
            "lactic_acidosis": lactic_acidosis,
            "hypotonia": hypotonia,
            "developmental_delay": dev_delay,
            "seizures": seizures,
            "cpeo": cpeo,
            "ptosis": ptosis,
            "exercise_intolerance": exercise_intolerance,
            "sensorineural_hearing_loss": hearing_loss,
            "cardiomyopathy": cardiomyopathy,
            "respiratory_failure": respiratory,
            "ragged_red_fibres": rrfs,
            "encephalopathy": encephalopathy,
            "spasticity": spasticity,
            "outcome": outcome_label,
        })

    return patients


# ── Cohort statistics ─────────────────────────────────────────────────────────
def _cohort_stats(patients):
    n = len(patients)
    pct = lambda key: round(100 * sum(1 for p in patients if p.get(key, False)) / n, 1)
    avg = lambda key: round(sum(p[key] for p in patients) / n, 1)
    return {
        # LHON-specific
        "optic_atrophy_pct":                pct("optic_atrophy"),
        "central_scotoma_pct":              pct("central_scotoma"),
        "dyschromatopsia_pct":              pct("dyschromatopsia_red_green"),
        "peripapillary_microangiopathy_pct": pct("peripapillary_microangiopathy"),
        "idebenone_rx_pct":                 pct("idebenone_rx"),
        "gene_therapy_pct":                 pct("gene_therapy"),
        # Leigh/CI-specific
        "leigh_mri_pct":                    pct("leigh_mri"),
        "stroke_like_pct":                  pct("stroke_like_episodes"),
        "lactic_acidosis_pct":              pct("lactic_acidosis"),
        "hypotonia_pct":                    pct("hypotonia"),
        "developmental_delay_pct":          pct("developmental_delay"),
        "seizures_pct":                     pct("seizures"),
        "cpeo_pct":                         pct("cpeo"),
        "ptosis_pct":                       pct("ptosis"),
        "exercise_intolerance_pct":         pct("exercise_intolerance"),
        "hearing_loss_pct":                 pct("sensorineural_hearing_loss"),
        "cardiomyopathy_pct":               pct("cardiomyopathy"),
        "respiratory_failure_pct":          pct("respiratory_failure"),
        "ragged_red_fibres_pct":            pct("ragged_red_fibres"),
        "encephalopathy_pct":               pct("encephalopathy"),
        "spasticity_pct":                   pct("spasticity"),
        # Biochemistry
        "avg_ci_activity_pct":              avg("ci_activity_pct"),
        "avg_lactic_acid_mmolL":            avg("lactic_acid_mmolL"),
        "avg_heteroplasmy_blood_pct":       avg("heteroplasmy_blood_pct"),
        "male_pct": round(
            100 * sum(1 for p in patients if p["sex"] == "M") / n, 1
        ),
        "lhon_phenotype_pct": round(
            100 * sum(1 for p in patients if "LHON" in p["phenotype"]) / n, 1
        ),
        "leigh_phenotype_pct": round(
            100 * sum(1 for p in patients if "Leigh" in p["phenotype"] or "MELAS" in p["phenotype"]) / n, 1
        ),
        "deceased_pct": round(
            100 * sum(1 for p in patients if "Deceased" in p["outcome"]) / n, 1
        ),
    }


# ── Phenotype distribution helper ─────────────────────────────────────────────
def _phenotype_dist(patients):
    from collections import Counter
    n = len(patients)
    c = Counter(p["phenotype"] for p in patients)
    return [{"phenotype": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in c.most_common()]


def _rng_count(rate, n=N_PATIENTS):
    return round(rate * n)


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    patients = _make_patients()
    stats = _cohort_stats(patients)
    n = len(patients)

    # Features ordered by clinical relevance across DUAL phenotype
    feat_order = [
        # LHON features first
        ("Optic atrophy (LHON, subacute bilateral)", "optic_atrophy"),
        ("Central scotoma (LHON, centrocaecal)", "central_scotoma"),
        ("Red-green dyschromatopsia (LHON)", "dyschromatopsia_red_green"),
        ("Peripapillary microangiopathy (LHON; no FFA leak)", "peripapillary_microangiopathy"),
        # Leigh/CI features
        ("Leigh MRI (bilateral BG/brainstem T2 hyperintensity)", "leigh_mri"),
        ("Lactic acidosis (Leigh/CI phenotype)", "lactic_acidosis"),
        ("Developmental delay / regression (Leigh)", "developmental_delay"),
        ("Hypotonia (Leigh/CI phenotype)", "hypotonia"),
        ("Encephalopathy (Leigh/CI)", "encephalopathy"),
        ("Seizures", "seizures"),
        ("Stroke-like episodes (MELAS feature, Leigh/MELAS overlap)", "stroke_like_episodes"),
        ("Respiratory failure (Leigh/infantile)", "respiratory_failure"),
        ("CPEO (progressive external ophthalmoplegia)", "cpeo"),
        ("Ptosis", "ptosis"),
        ("Exercise intolerance", "exercise_intolerance"),
        ("Cardiomyopathy (KSS/Leigh)", "cardiomyopathy"),
        ("Sensorineural hearing loss", "sensorineural_hearing_loss"),
        ("Ragged-red fibres (RRF, Gomori; CPEO/KSS)", "ragged_red_fibres"),
        ("Spasticity (Leigh)", "spasticity"),
        ("Idebenone treatment received (LHON)", "idebenone_rx"),
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
        "🚫 Tobacco (all forms) — ABSOLUTE CONTRAINDICATION in ALL MT-ND4 carriers AND affected (LHON AND Leigh): cyanide in cigarette smoke is a CI inhibitor; IN LHON specifically, tobacco is the strongest precipitating environmental factor for acute vision loss — even passive smoking is dangerous",
        "🚫 Ethambutol — ABSOLUTE CONTRAINDICATION in all MT-ND4 LHON carriers AND affected: optic nerve toxin (mitochondrial inhibitor in retinal ganglion cells); exacerbates MT-ND4-associated optic atrophy; prescribe alternative TB treatment",
        "🚫 Metformin — ABSOLUTE CONTRAINDICATION (BOTH phenotypes): Complex I inhibitor — additive CI failure; in Leigh: fatal lactic acidosis; in LHON: additional CI burden on already vulnerable optic nerve",
        "🚫 Alcohol — ABSOLUTE CONTRAINDICATION in MT-ND4 LHON (acetaldehyde is a CI inhibitor and RGC mitochondrial toxin); precipitates LHON onset in at-risk near-homoplasmic carriers",
        "🚫 Amiodarone — ABSOLUTE CONTRAINDICATION in MT-ND4 LHON: drug-induced optic neuropathy (DION) is synergistic with LHON optic neuropathy; combined → catastrophic bilateral blindness",
        "🚫 Valproic acid (VPA) — ABSOLUTE CONTRAINDICATION: CoA sequestration + POLG inhibition + OXPHOS impairment; especially dangerous in Leigh phenotype (hepatotoxicity risk)",
        "🚫 Linezolid — ABSOLUTE CONTRAINDICATION: inhibits mt-23S rRNA translation → directly reduces MT-ND4 protein synthesis → worsens CI in both LHON and Leigh phenotypes",
        "🚫 Propofol — ABSOLUTE CONTRAINDICATION (Leigh phenotype): PRIS — directly inhibits CI; compounding MT-ND4 CI deficiency → fatal",
        "🚫 IV tPA (stroke-like episodes) — CONTRAINDICATED: stroke-like episodes in Leigh/MELAS overlap are metabolic NOT thrombotic; tPA causes harm; treat with thiamine + hydration",
        "⚠️ m.11778G>A: WORST LHON PROGNOSIS of 3 primary LHON mutations — <4% spontaneous visual recovery (vs 22% ND1 m.3460G>A, 50% ND6 m.14484T>C) — start idebenone EARLY; refer for gene therapy eligibility (lenadogene nolparvovec, m.11778G>A only)",
        "⚠️ SECOND EYE: 97% develop contralateral involvement within 6–8 weeks of first eye — monitor closely; do NOT withhold idebenone waiting for bilateral presentation",
        "⚠️ MALE PENETRANCE: 50% lifetime risk in hemizygous males; 10% females — all maternal relatives near-homoplasmic for m.11778G>A require lifestyle counselling (tobacco, alcohol, ethambutol, amiodarone avoidance) even if asymptomatic",
        "⚠️ FFA (Fluorescein Angiography) — KEY DDx NAION vs LHON: NAION leaks on FFA (ischemic disc edema); LHON peripapillary microangiopathy does NOT leak on FFA — always perform FFA in acute unilateral optic neuropathy",
        "⚠️ WES MISSES MT-ND4: mtDNA variants require dedicated mtDNA sequencing (WGS with mtDNA enrichment, long-read, or mtDNA panel); WES covers mtDNA unreliably",
        "✅ Idebenone (Raxone) 900 mg/day — Level B (RHODOS trial, Klopstock 2011 Brain): only drug with RCT evidence in LHON; start as early as possible for best outcome; continue ≥24 months",
        "✅ Lenadogene nolparvovec (Lumevoq, AAV2-MT-ND4) — gene therapy for m.11778G>A ONLY: EU conditional approval 2021; intravitreal injection; bilateral injection recommended (RESCUE+REVERSE trials); referral to specialist centre",
        "✅ Low vision rehabilitation — Level A: magnifiers, large-print, screen readers, occupational therapy; mobility training; even if visual prognosis poor, quality of life highly dependent on rehabilitation",
        "✅ Leigh phenotype: GIR 6–8 mg/kg/min (NEVER fast); Thiamine B1 MANDATORY IV 10–20 mg/kg; NaHCO3 for acute lactic acidosis; LEV preferred AED (not VPA)",
        "✅ CoQ10/Ubiquinol + Riboflavin (B2): mitochondrial cocktail (Level C); ubiquinol preferred for LHON (short-chain electron shuttling)",
        "✅ Sevoflurane preferred over Propofol for anaesthesia in ALL MT-ND4 patients (LHON and Leigh)",
    ]

    return {
        "gene": "MT-ND4",
        "full_name": "Mitochondrially Encoded NADH:Ubiquinone Oxidoreductase Core Subunit 4",
        "alias": "ND4 / Complex I subunit 4 / MTND4",
        "omim_gene": "516003",
        "omim_disease": "LHON (Leber Hereditary Optic Neuropathy) + Isolated CI Deficiency / Leigh Syndrome — OMIM *516003",
        "disease_name": "LHON (m.11778G>A #1 worldwide) + Isolated Complex I Deficiency — Leigh Syndrome / MELAS Overlap (MT-ND4 Central Antiporter Module Defect)",
        "chromosome": "mtDNA (rCRS positions 10760–12137)",
        "inheritance": "MATERNAL — mtDNA; near-homoplasmic in LHON; heteroplasmic in Leigh/CI deficiency",
        "protein_size": "459 aa, 51.7 kDa; 13 TM helices; central antiporter module",
        "dual_phenotype": {
            "LHON (near-homoplasmic)": "m.11778G>A near-homoplasmic → selective RGC optic nerve CI failure → subacute bilateral optic atrophy; normal systemic CI; male predominance 80–90%; onset 15–35y; <4% spontaneous recovery",
            "Leigh/CI deficiency (high heteroplasmy)": "MT-ND4 variants with >80% heteroplasmy → systemic CI deficiency → Leigh syndrome, MELAS overlap; lactic acidosis, hypotonia, Leigh MRI; phenotype identical to other mtDNA CI subunit Leigh",
        },
        "protein": {
            "size_aa": 459,
            "kDa": 51.7,
            "tm_helices": 13,
            "localization": "Integral IMM — central membrane arm, antiporter module (ND4 submodule); flanked by ND2 (proximal) and ND5 (distal); interacts with ND4L (hairpin), ND5 (lateral helix contact)",
            "role": "Central proton-pumping antiporter repeat; keystone of CI membrane arm proton translocation; ND4 TM11 Arg340 coordinates conserved water-mediated proton relay (disrupted in m.11778G>A LHON)",
            "function": "One of 7 mtDNA-encoded CI subunits; central antiporter module; ND4 loss → proton-pumping failure; LHON: optic nerve selective (highest CI demand, lowest compensatory capacity); Leigh: systemic CI collapse",
        },
        "lhon_clinical_hallmarks": {
            "Subacute painless central visual loss": "Days to 6 weeks progression; NO pain (vs optic neuritis); central/centrocaecal scotoma",
            "Sequential bilateral": "Second eye 6–8 weeks after first in 97%; virtually always bilateral",
            "Sex predominance": "Males 80–90% (X-linked modifier loci + mitochondrial network differences); female penetrance ~10%",
            "Age of onset": "Peak 15–35 years; range reported 2–87 years",
            "Dyschromatopsia": "Red-green (centrocaecal axis colour discrimination loss); NOT blue-yellow (DDx OPA1 dominant optic atrophy)",
            "Peripapillary microangiopathy": "Telangiectatic vessel tortuosity; does NOT leak on FFA (KEY DDx NAION which leaks)",
            "m.11778G>A prognosis": "<4% spontaneous visual recovery — WORST of 3 primary LHON mutations; early idebenone + gene therapy referral",
        },
        "ci_assembly_context": "ND4 is incorporated into the CI membrane arm central antiporter module; ND4 loss → central membrane arm instability; BN-PAGE: CI severely reduced in Leigh phenotype (10–25% residual); ND4-lacking sub-complex accumulates; LHON phenotype: CI on BN-PAGE typically normal or mildly reduced (tissue-specific: optic nerve CI most vulnerable); CI/CII/CIII/CIV — isolated CI deficiency in Leigh; normal CII/CIII/CIV",
        "heteroplasmy_thresholds": {
            "Near-homoplasmic >95% (blood) — m.11778G>A": "LHON phenotype; asymptomatic until penetrance threshold breached in optic tissue; all maternal relatives carry; 50% male / 10% female lifetime risk",
            "<80% mutant (blood) — other variants": "Typically subclinical / oligosymptomatic maternal carriers; Leigh threshold not yet reached",
            "80–90% (blood) — Leigh variants": "Leigh/MELAS overlap; stroke-like episodes possible (38%)",
            ">90% (blood) — Leigh variants": "Severe infantile Leigh syndrome; isolated CI deficiency; lactic acidosis; Leigh MRI bilateral BG/brainstem",
        },
        "bn_page_pattern": "CI severely reduced (10–25% residual) in Leigh phenotype; ND4-lacking sub-complex accumulates in Leigh; LHON phenotype: CI on BN-PAGE typically normal or mildly reduced (tissue-specific: optic nerve CI most vulnerable)",
        "key_biochemical_features": [
            "LHON: systemic CI near-normal on standard assays (optic nerve CI selectively deficient); plasma lactate NORMAL",
            "Leigh/CI: isolated CI deficiency 10–25% residual; CII/CIII/CIV NORMAL",
            "Lactic acidosis: LHON normal lactate; Leigh elevated lactate 3–22 mM; L:P >20",
            "Elevated alanine: PDH inhibition secondary to high NADH/NAD⁺ (Leigh only)",
            "BN-PAGE: LHON: CI near-normal; Leigh: CI severely reduced; ND4-lacking partial sub-complex accumulates",
            "Muscle biopsy: RRF (Gomori trichrome) in CPEO/KSS (65%); COX-positive RRF; absent in LHON (non-muscular disease)",
            "Immunoblot: ND4 reduced/absent in Leigh; secondary reduction of ND5, ND2 (antiporter neighbours); LHON: ND4 variable (near-normal in non-optic tissue)",
            "mtDNA sequencing: identifies m.11778G>A (or other variant); blood near-homoplasmic in LHON; muscle preferred for Leigh heteroplasmy accuracy",
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
            "neonatal_pct": round(100 * sum(1 for p in patients if p["onset_weeks"] <= 3) / n, 1),
            "infantile_pct": round(100 * sum(1 for p in patients if 4 <= p["onset_weeks"] <= 52) / n, 1),
            "childhood_juvenile_pct": round(100 * sum(1 for p in patients if 53 <= p["onset_weeks"] <= 780) / n, 1),
            "young_adult_lhon_pct": round(100 * sum(1 for p in patients if 781 <= p["onset_weeks"] <= 1820) / n, 1),
            "adult_pct": round(100 * sum(1 for p in patients if p["onset_weeks"] > 1820) / n, 1),
        },
    }


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

    lhon_patients = [p for p in patients if "LHON" in p["phenotype"]]
    leigh_patients = [p for p in patients if "Leigh" in p["phenotype"] or "MELAS" in p["phenotype"]]

    biochem = {
        "avg_ci_activity_pct":           round(sum(ci_vals) / n, 1),
        "ci_below_10_pct":               round(100 * sum(1 for v in ci_vals if v < 10) / n, 1),
        "ci_10_to_25_pct":               round(100 * sum(1 for v in ci_vals if 10 <= v < 25) / n, 1),
        "ci_25_to_50_pct":               round(100 * sum(1 for v in ci_vals if 25 <= v < 50) / n, 1),
        "ci_above_50_pct":               round(100 * sum(1 for v in ci_vals if v >= 50) / n, 1),
        "avg_lactic_acid_mmolL":         round(sum(lac_vals) / n, 1),
        "lactic_normal_lhon_pct":        round(100 * sum(1 for v in lac_vals if v < 2.5) / n, 1),
        "lactic_mild_pct":               round(100 * sum(1 for v in lac_vals if 2.5 <= v < 5) / n, 1),
        "lactic_moderate_pct":           round(100 * sum(1 for v in lac_vals if 5 <= v <= 10) / n, 1),
        "lactic_severe_pct":             round(100 * sum(1 for v in lac_vals if v > 10) / n, 1),
        "avg_heteroplasmy_blood_pct":    round(sum(het_vals) / n, 1),
        "het_near_homoplasmic_pct":      round(100 * sum(1 for v in het_vals if v >= 90) / n, 1),
        "het_80_to_90_pct":              round(100 * sum(1 for v in het_vals if 80 <= v < 90) / n, 1),
        "het_60_to_80_pct":              round(100 * sum(1 for v in het_vals if 60 <= v < 80) / n, 1),
        "het_below_60_pct":              round(100 * sum(1 for v in het_vals if v < 60) / n, 1),
        # Phenotype split
        "lhon_n":                        len(lhon_patients),
        "leigh_n":                       len(leigh_patients),
        "lhon_avg_ci_pct":              round(sum(p["ci_activity_pct"] for p in lhon_patients) / max(len(lhon_patients), 1), 1),
        "leigh_avg_ci_pct":             round(sum(p["ci_activity_pct"] for p in leigh_patients) / max(len(leigh_patients), 1), 1),
        "lhon_avg_lactate":             round(sum(p["lactic_acid_mmolL"] for p in lhon_patients) / max(len(lhon_patients), 1), 1),
        "leigh_avg_lactate":            round(sum(p["lactic_acid_mmolL"] for p in leigh_patients) / max(len(leigh_patients), 1), 1),
    }

    oc = Counter(p["outcome"] for p in patients)
    outcomes = [{"outcome": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in oc.most_common()]

    return {
        "gene": "MT-ND4",
        "all_variants": VARIANTS,
        "variant_distribution": variant_dist,
        "phenotype_distribution": _phenotype_dist(patients),
        "biochemistry_distribution": biochem,
        "outcome_distribution": outcomes,
        "cohort_statistics": stats,
        "lhon_vs_leigh_comparison": {
            "LHON phenotype": {
                "n_patients": len(lhon_patients),
                "pct_cohort": round(100 * len(lhon_patients) / n, 1),
                "driving_variant": "m.11778G>A (p.Arg340His) — 70% of LHON worldwide",
                "heteroplasmy": "Near-homoplasmic >92% blood",
                "sex": "Male predominance 80–90%",
                "age_onset": "15–35 years (peak)",
                "systemic_CI": "Near-normal (tissue-specific optic nerve deficiency)",
                "lactate": "Normal (<2.5 mM)",
                "mri": "Brain MRI normal (no Leigh lesions)",
                "optic_features": "Subacute painless bilateral optic atrophy; centrocaecal scotoma; red-green dyschromatopsia; peripapillary microangiopathy (no FFA leak)",
                "prognosis": "<4% spontaneous visual recovery (worst primary LHON mutation)",
                "treatment": "Idebenone 900 mg/day (Level B, RHODOS); gene therapy (m.11778G>A: lenadogene nolparvovec, EU approval 2021); CoQ10/Ubiquinol (Level C)",
                "key_contraindications": "Tobacco ABSOLUTE, Ethambutol ABSOLUTE, Alcohol ABSOLUTE, Amiodarone ABSOLUTE",
            },
            "Leigh/CI Deficiency phenotype": {
                "n_patients": len(leigh_patients),
                "pct_cohort": round(100 * len(leigh_patients) / n, 1),
                "driving_variant": "m.11696G>A (p.Ser110Asn), m.11253T>C (p.Phe251Leu) — high heteroplasmy",
                "heteroplasmy": "High heteroplasmy >80% blood",
                "sex": "Equal sex distribution (~50/50)",
                "age_onset": "Neonatal–infantile (CI variants)",
                "systemic_CI": "Isolated CI deficiency 10–25% residual",
                "lactate": "Elevated 3–22 mM; L:P >20",
                "mri": "Leigh MRI: bilateral symmetric BG/brainstem T2 hyperintensity 88%",
                "optic_features": "No primary optic neuropathy (DDx LHON)",
                "prognosis": "Variable: infantile Leigh 15% deceased <5y; Leigh/MELAS moderate disability",
                "treatment": "GIR 6–8 mg/kg/min + Thiamine IV + NaHCO3 + LEV (not VPA); CoQ10/Ubiquinol",
                "key_contraindications": "Metformin ABSOLUTE, VPA ABSOLUTE, Linezolid ABSOLUTE, Propofol ABSOLUTE",
            },
        },
        "bn_page_pattern": {
            "finding": "LHON: CI near-normal on BN-PAGE (tissue-specific; optic nerve not tested); Leigh: CI severely reduced (10–25% residual); ND4-lacking partial sub-complex accumulates in Leigh phenotype; CII/CIII/CIV normal",
            "interpretation": "LHON CI near-normal systemically masks optic nerve selective vulnerability; Leigh ND4 loss → central antiporter module instability → CI assembly failure; ND4-lacking intermediate accumulates similarly to ND5-lacking P-intermediate in MT-ND5 Leigh",
            "ddx_value": "LHON: DDx by normal systemic CI activity + maternal inheritance + optic phenotype + near-homoplasmic m.11778G>A; Leigh: isolated CI fingerprint; DDx from MELAS (not pan-OXPHOS) and nuclear CI defects (maternal not AR)",
        },
        "immunoblot_pattern": {
            "MT-ND4 (ND4)": "LHON: near-normal in blood/muscle; Leigh: absent or severely reduced (heteroplasmy-proportional)",
            "MT-ND5 (ND5)": "Secondarily reduced in severe Leigh (ND4-ND5 antiporter module interdependence)",
            "MT-ND2 (ND2)": "Variably secondarily reduced in severe Leigh (central antiporter neighbour)",
            "MT-ND4L (ND4L)": "Secondarily reduced (hairpin connector to ND4)",
            "ND1/ND3/ND6": "Variable secondary reduction in severe Leigh CI collapse",
            "CI NDUFS1/NDUFS2 (N-module)": "Near-normal in mild-moderate; secondarily reduced in severe Leigh",
            "CII (SDHA)": "Normal — isolated CI deficiency",
            "CIV (COX4I1)": "Normal — isolated CI deficiency",
        },
        "treatment_uptake": {
            "Idebenone (LHON — Raxone 900 mg/day)":    f"{_rng_count(0.55)} / {n} patients",
            "Lenadogene nolparvovec (gene therapy, m.11778G>A)": f"{_rng_count(0.12)} / {n} patients",
            "Low vision rehabilitation":                 f"{_rng_count(0.70)} / {n} patients",
            "CoQ10 / Ubiquinol":                         f"{_rng_count(0.72)} / {n} patients",
            "Riboflavin (B2)":                           f"{_rng_count(0.60)} / {n} patients",
            "Thiamine (B1) IV/oral (Leigh)":             f"{_rng_count(0.88)} / {n} patients",
            "GIR 6–8 mg/kg/min (Leigh, never fast)":    f"{_rng_count(0.80)} / {n} patients",
            "NaHCO₃ IV (acute lactic acidosis, Leigh)": f"{_rng_count(0.62)} / {n} patients",
            "LEV (preferred AED, Leigh)":                f"{_rng_count(0.38)} / {n} patients",
            "NIV / BiPAP (Leigh)":                       f"{_rng_count(0.30)} / {n} patients",
        },
        "ddx_table": [
            {
                "condition": "NAION (Non-Arteritic Anterior Ischaemic Optic Neuropathy)",
                "key_distinction": "MT-ND4 LHON: peripapillary microangiopathy does NOT leak on FFA; subacute (not sudden); maternal inheritance; near-homoplasmic. NAION: sudden (vascular event); disc edema with FFA leak (ischemic); no maternal relatives affected; no FFA leakage",
                "shared": "Unilateral onset, visual loss, optic disc changes, middle-age",
            },
            {
                "condition": "OPA1 (Dominant Optic Atrophy, Kjer disease)",
                "key_distinction": "MT-ND4 LHON: subacute onset NOT insidious; red-green loss NOT blue-yellow tritanopia; MATERNAL inheritance NOT AD; near-homoplasmic heteroplasmy vs OPA1 AD mutation. OPA1: insidious juvenile onset; blue-yellow colour axis; autosomal dominant; WES diagnostic",
                "shared": "Optic atrophy, bilateral, central visual loss, colour vision defect",
            },
            {
                "condition": "MT-ND1 (m.3460G>A LHON)",
                "key_distinction": "Same LHON syndrome; 22% spontaneous recovery (better than MT-ND4 m.11778G>A <4%); idebenone identical; no gene therapy for m.3460G>A; distinguish by sequencing. MT-ND4 m.11778G>A: worst prognosis; gene therapy eligible",
                "shared": "LHON, maternal inheritance, bilateral optic atrophy, same clinical presentation",
            },
            {
                "condition": "MT-ND6 (m.14484T>C LHON)",
                "key_distinction": "Same LHON syndrome; BEST spontaneous recovery 50% (vs MT-ND4 <4%); younger average onset; counsel patient about prognosis difference; no gene therapy for m.14484T>C; distinguish by sequencing",
                "shared": "LHON, maternal inheritance, subacute bilateral optic atrophy, peripapillary microangiopathy",
            },
            {
                "condition": "Nuclear CI defects (NDUFAF*/NDUF*) — Leigh phenotype",
                "key_distinction": "MT-ND4 Leigh: MATERNAL inheritance, heteroplasmy, WES misses. Nuclear: AR biallelic, WES diagnostic, no maternal relatives affected, no LHON phenotype. BN-PAGE isolated CI similar; distinguish by maternal family testing",
                "shared": "Isolated CI deficiency, Leigh MRI, lactic acidosis, hypotonia",
            },
            {
                "condition": "MT-ND5 (Leigh/MELAS overlap)",
                "key_distinction": "MT-ND5: larger subunit (603 aa vs 459 aa ND4); MELAS stroke-like 42% (higher than MT-ND4 Leigh/MELAS 38%); NO optic atrophy as primary feature; no LHON phenotype in MT-ND5. MT-ND4: DUAL phenotype (LHON or Leigh); same mtDNA molecule → compound mutations possible; distinguish by sequencing",
                "shared": "Maternal inheritance, CI deficiency, Leigh MRI, lactic acidosis",
            },
        ],
    }


def get_definitions():
    return {
        "gene": "MT-ND4",
        "full_name": "Mitochondrially Encoded NADH:Ubiquinone Oxidoreductase Core Subunit 4",
        "alias": "ND4 / Complex I subunit 4 / MTND4",
        "omim_gene": "516003",
        "disease_name": "LHON (m.11778G>A #1 worldwide) + Isolated Complex I Deficiency — Leigh Syndrome / MELAS Overlap (MT-ND4 Central Antiporter Module Defect)",
        "chromosome": "mtDNA (rCRS 10760–12137)",
        "protein_aa": 459,
        "protein_kDa": 51.7,
        "tm_helices": 13,
        "inheritance": "MATERNAL — mtDNA; near-homoplasmic in LHON; heteroplasmic in Leigh/CI deficiency",
        "key_concepts": {
            "Central antiporter module": "MT-ND4 is the central of 3 antiporter repeats (ND2-ND4-ND5) in the CI membrane arm; each repeat constitutes one proton-pumping unit; ND4 is the keystone — flanked by ND2 (proximal) and ND5 (distal); loss of ND4 → central membrane arm structural failure",
            "Dual phenotype (unique)": "MT-ND4 is the only CI subunit that causes BOTH LHON (optic neuropathy, near-homoplasmic, normal systemic CI) AND Leigh syndrome/CI deficiency (high heteroplasmy, systemic CI collapse) — phenotype determined by heteroplasmy level and tissue distribution",
            "m.11778G>A (p.Arg340His)": "World's most common LHON mutation (~70% of all LHON globally); Arg340 TM11 coordinates water-mediated proton relay chain; Arg→His disrupts pKa of proton relay → selective RGC CI failure; near-homoplasmic (>95%) but incomplete penetrance (50% males, 10% females)",
            "LHON prognosis stratification": "m.11778G>A: <4% spontaneous visual recovery (WORST); m.3460G>A (ND1): 22%; m.14484T>C (ND6): 50% (BEST). Prognosis determines urgency of idebenone and gene therapy referral",
            "Peripapillary microangiopathy (no FFA leak)": "LHON characteristic fundus finding; telangiectatic microangiopathy does NOT leak on fluorescein angiography; KEY DDx vs NAION (which leaks on FFA) — always perform FFA in acute optic neuropathy",
            "Gene therapy (lenadogene nolparvovec)": "AAV2-MT-ND4 intravitreal injection; delivers wild-type MT-ND4 to retinal ganglion cells; EU conditional approval 2021 (m.11778G>A ONLY); RESCUE+REVERSE trials showed benefit; not approved for other variants",
            "Heteroplasmy duality": "Near-homoplasmic (>95%): LHON — normal systemic CI; optic nerve selectively vulnerable (highest CI demand, lowest OXPHOS reserve); HIGH heteroplasmy (>80%): Leigh — systemic CI collapse across all tissues",
            "Tobacco absolute contraindication": "Cyanide in cigarette smoke is a direct CI inhibitor; in LHON near-homoplasmic carriers, tobacco is the single strongest precipitating environmental trigger for acute vision loss — even passive smoking is dangerous; ABSOLUTE contraindication in BOTH phenotypes",
            "Ethambutol ABSOLUTE in LHON": "Ethambutol inhibits mitochondria in optic nerve; synergistic with MT-ND4 CI deficiency in retinal ganglion cells; can precipitate irreversible blindness in asymptomatic carriers — ALWAYS check maternal family mtDNA before prescribing ethambutol for TB",
        },
        "contraindications": [
            {"drug": "Tobacco (all forms)", "severity": "ABSOLUTE CI — BOTH phenotypes", "reason": "Cyanide is a CI inhibitor; in LHON strongest precipitating environmental trigger for vision loss; in Leigh additional CI burden; passive smoking also dangerous"},
            {"drug": "Ethambutol", "severity": "ABSOLUTE CI — LHON (carriers and affected)", "reason": "Optic nerve mitochondrial toxin; synergistic with MT-ND4 CI deficiency in RGCs; irreversible blindness risk; always screen for LHON before TB treatment"},
            {"drug": "Alcohol", "severity": "ABSOLUTE CI — LHON phenotype", "reason": "Acetaldehyde is a CI inhibitor and RGC mitochondrial toxin; precipitates LHON onset in near-homoplasmic carriers"},
            {"drug": "Amiodarone", "severity": "ABSOLUTE CI — LHON phenotype", "reason": "Drug-induced optic neuropathy (DION) synergistic with LHON optic neuropathy → catastrophic bilateral blindness"},
            {"drug": "Metformin", "severity": "ABSOLUTE CI — both phenotypes", "reason": "Complex I inhibitor; additive CI failure in LHON (optic nerve); fatal lactic acidosis in Leigh phenotype"},
            {"drug": "Valproic acid (VPA)", "severity": "ABSOLUTE CI — Leigh phenotype", "reason": "CoA sequestration + POLG inhibition + OXPHOS impairment; hepatotoxicity risk"},
            {"drug": "Linezolid", "severity": "ABSOLUTE CI — both phenotypes", "reason": "Inhibits mt-23S rRNA translation → reduces MT-ND4 synthesis directly; worsens CI in LHON optic nerve and Leigh tissues"},
            {"drug": "Propofol", "severity": "ABSOLUTE CI — Leigh phenotype", "reason": "PRIS — directly inhibits CI; compounding MT-ND4 CI deficiency → fatal in Leigh; use sevoflurane instead"},
            {"drug": "IV tPA (stroke-like episodes)", "severity": "CONTRAINDICATED", "reason": "Stroke-like episodes in Leigh/MELAS overlap are metabolic NOT thrombotic; tPA causes harm; treat with thiamine + hydration"},
            {"drug": "Ketogenic diet (severe Leigh)", "severity": "CONTRAINDICATED", "reason": "High FADH2 from β-oxidation requires intact CI-dependent OXPHOS; CI deficiency → FADH2 backlog → metabolic crisis"},
        ],
        "treatments": [
            {"drug": "Idebenone (Raxone) 900 mg/day", "level": "Level B (LHON)", "rationale": "RHODOS trial (Klopstock 2011 Brain): RCT evidence; short-chain CoQ analog bypasses CI inhibition → CII→CIII→CIV; start early for best outcome; continue ≥24 months; most effective if started within 1 year of onset"},
            {"drug": "Lenadogene nolparvovec (Lumevoq)", "level": "Gene therapy (m.11778G>A, EU approval 2021)", "rationale": "AAV2-MT-ND4 intravitreal injection; delivers WT MT-ND4 to RGCs; m.11778G>A ONLY; bilateral injection; refer to specialist centre (Vignal-Clermont 2021 NEJM)"},
            {"drug": "Low vision rehabilitation", "level": "Level A (LHON)", "rationale": "Magnifiers, large-print, screen readers, mobility training; occupational therapy; independent living skills; irreversible in most m.11778G>A — rehabilitation essential"},
            {"drug": "CoQ10 / Ubiquinol", "level": "Level C", "rationale": "CoQ10 is CI-to-CIII electron carrier; ubiquinol preferred (reduced form); 10–20 mg/kg/day; both LHON and Leigh phenotypes"},
            {"drug": "Riboflavin (B2)", "level": "Level C", "rationale": "FAD-dependent CI assembly co-factors; empiric; both phenotypes"},
            {"drug": "Thiamine (B1) IV/oral", "level": "Mandatory empiric (Leigh)", "rationale": "PDH cofactor; CI-associated; 10–20 mg/kg IV in acute Leigh decompensation; oral maintenance; not required in isolated LHON"},
            {"drug": "GIR 6–8 mg/kg/min dextrose", "level": "Mandatory (Leigh)", "rationale": "NEVER fast in Leigh phenotype; continuous dextrose prevents CI crisis; increase during illness/surgery; not required in isolated LHON"},
            {"drug": "NaHCO₃ IV", "level": "Acute rescue (Leigh)", "rationale": "Target pH >7.2 in lactic acidosis; 1–2 mEq/kg IV; Leigh phenotype only"},
            {"drug": "LEV (Levetiracetam)", "level": "Preferred AED (Leigh)", "rationale": "Renal clearance; no mitochondrial toxicity; preferred over VPA/PHT/CBZ in mtDNA CI disease"},
            {"drug": "Sevoflurane (anaesthesia)", "level": "Preferred (both)", "rationale": "Over Propofol (CI inhibitor); all MT-ND4 patients (LHON and Leigh); inform anaesthesiologist of MT-ND4 status"},
        ],
        "diagnostic_workup": [
            "Detailed ophthalmological examination: visual acuity, Ishihara/Farnsworth D-15 (red-green loss), Goldmann visual field (central scotoma), fundoscopy (disc hyperaemia, peripapillary microangiopathy)",
            "Fluorescein angiography (FFA): KEY DDx — LHON does NOT leak on FFA; NAION leaks (ischemic disc edema); perform in all acute optic neuropathy",
            "OCT (optical coherence tomography): RNFL (retinal nerve fibre layer) thinning in LHON; peripapillary RNFL thickness monitors disease progression and treatment response",
            "Dedicated mtDNA sequencing: targeted m.11778G>A testing first (90% of LHON); if negative, full MT-ND4 + MT-ND1 + MT-ND6 panel; WES does NOT reliably cover mtDNA",
            "Blood heteroplasmy quantification: LHON near-homoplasmic (>95%); heteroplasmy level; muscle biopsy only if Leigh phenotype suspected (blood may underestimate by 10–20 ppts)",
            "Maternal family cascade testing: all maternal relatives; heteroplasmy level; lifestyle counselling for all near-homoplasmic carriers (tobacco, alcohol, ethambutol, amiodarone avoidance)",
            "Brain MRI: Leigh phenotype → bilateral symmetric T2 hyperintensity putamen/brainstem; LHON → typically normal (no Leigh lesions); cortical if MELAS overlap",
            "Plasma lactate + pyruvate + L:P ratio: LHON → normal; Leigh → elevated lactate (>2.5 mM), L:P >20",
            "OXPHOS enzyme activities (muscle or fibroblasts): LHON → systemic CI near-normal; Leigh → isolated CI deficiency (CI 10–25%); CII/CIII/CIV normal",
            "BN-PAGE: LHON → CI near-normal; Leigh → CI severely reduced + ND4-lacking sub-complex accumulation",
            "Immunoblot: ND4 near-normal in LHON; ND4 reduced/absent in Leigh; secondary ND5/ND2 reduction in Leigh",
        ],
        "references": [
            {
                "citation": "Wallace DC et al. (1988) Mitochondrial DNA mutation associated with Leber's hereditary optic neuropathy. Science 242:1427-30.",
                "relevance": "Discovery paper for MT-ND4 m.11778G>A; established the first pathogenic mtDNA point mutation; demonstrated maternal inheritance of LHON; foundational for mtDNA disease field",
            },
            {
                "citation": "Riordan-Eva P & Bhatt DK (2002) Leber's hereditary optic neuropathy. Curr Opin Neurol.",
                "relevance": "Comprehensive LHON natural history; m.11778G>A prognosis (<4% recovery); male predominance; sequential bilateral involvement; clinical diagnostic criteria",
            },
            {
                "citation": "Klopstock T et al. (2011) A randomized placebo-controlled trial of idebenone in Leber's hereditary optic neuropathy. Brain 134(Pt 9):2677-86.",
                "relevance": "RHODOS trial; first RCT in LHON; idebenone 900 mg/day Level B evidence; established idebenone as standard LHON treatment; primary outcome: best-recovery visual acuity",
            },
            {
                "citation": "Vignal-Clermont C et al. (2021) Lenadogene nolparvovec gene therapy for Leber hereditary optic neuropathy. N Engl J Med.",
                "relevance": "AAV2-MT-ND4 (lenadogene nolparvovec/Lumevoq) clinical trial; EU conditional approval 2021 for m.11778G>A LHON; intravitreal delivery to RGCs; bilateral injection paradigm; first approved mtDNA gene therapy",
            },
            {
                "citation": "Agip AA et al. (2018) Cryo-EM structures of complex I from mouse heart mitochondria in two biochemical states. Nat Struct Mol Biol 25(7):548-556.",
                "relevance": "High-resolution CI structure; ND4 central antiporter module topology; ND4 TM11 Arg340 proton-relay role; ND4-ND5-ND4L interaction interface; structural basis of m.11778G>A pathogenicity",
            },
        ],
        "cohort_seed": SEED,
        "n_patients": N_PATIENTS,
    }
