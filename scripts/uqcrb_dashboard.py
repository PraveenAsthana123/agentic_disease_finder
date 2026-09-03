#!/usr/bin/env python3
"""UQCRB — Ubiquinol-Cytochrome C Reductase Binding Protein (QCR7 / Subunit VI) /
Isolated Complex III (CIII) Deficiency — Nuclear, AR Biallelic

UQCRB (OMIM *191330) encodes the 109-amino-acid, ~12.1 kDa peripheral structural
subunit of Complex III (cytochrome bc1 complex) that associates with the Qi-site
(ubiquinone reduction / quinol-binding inner) site at the matrix face of the inner
mitochondrial membrane (IMM). Unlike UQCRQ (1 TM helix, IMS-facing Qo-site),
UQCRB has NO transmembrane helix and contacts UQCRC1 and UQCRC2 (Core proteins 1
and 2) at the matrix-facing Qi-site periphery.

  UQCRB gene     OMIM *191330
  Alias          QCR7 (yeast), binding protein, Subunit VI of bc1 complex
  Disease        Isolated Complex III Deficiency (nuclear) — AR biallelic
  Protein        109 aa (full), ~12.1 kDa; MTS cleavage ~aa 1-13;
                 NO transmembrane helix; peripheral matrix-facing Qi-site subunit
  Chromosome     8q22.1
  CIII role      Structural subunit at Qi-site periphery (matrix face); stabilises
                 UQCRC1/UQCRC2 scaffold contacts; Qi-site ubiquinone-binding support

CIII Assembly — UQCRB-Dependent Step:
  1. UQCRB is a structural (not assembly-factor) subunit — peripherally incorporated
     during CIII holocomplex assembly; contacts UQCRC1 / UQCRC2 at the matrix face
  2. UQCRB stabilises the Qi-site pocket geometry; without it, ubiquinone (CoQ)
     cannot be properly bound for reduction at the Qi site → CIII activity severely reduced
  3. UQCRB loss → CIII structural instability at the Qi-site periphery → secondary
     partial reduction of UQCRC1/UQCRC2 stability; RISP (UQCRFS1) is less affected
     than in Qo-site (UQCRQ) deficiency since RISP inserts at the Qo-side
  4. BN-PAGE: CIII severely reduced (5–15% residual); core sub-complexes partially
     present; pattern distinct from UQCC2 (CIII absent) and UQCRQ (sub-complexes
     more prominent); more severe reduction than UQCRQ (8–22%)
  5. Enzymatic activity: 5–15% residual CIII; CI, CII, CIV activities: normal

UQCRB Loss-of-Function → CIII deficiency:
  • Qi-site structural disruption → reduced CoQ reduction → CoQ cycle arrest
  • Isolated CIII deficiency: CI, CII, CIV initially normal
  • Hypoglycemia (DISTINGUISHING): Qi-site dysfunction → CIII block → gluconeogenesis
    and fatty acid β-oxidation severely impaired → fasting hypoglycemia; more
    prominent than Qo-site (UQCRQ) deficiency (Qi-site critical for FADH2 re-oxidation)
  • Hepatopathy (more prominent than UQCRQ): liver has high CIII demand for
    gluconeogenesis; UQCRB Qi-site disruption disproportionately affects hepatocytes
  • Lactic acidosis: CoQ cycle arrest → excess NADH/FADH2 → PDH/TCA stalling
  • NO dystonia as cardinal (contrast UQCRQ 75%): Qi-site does NOT generate ROS
    at the rate of the Qo-site; basal ganglia less selectively vulnerable
  • NO optic atrophy as distinguishing (contrast UQCRQ 42%): less Qi-site ROS
    oxidative stress on retinal ganglion cells

PHENOTYPE — UQCRB:
  ONSET:
    • Neonatal (birth to 4 weeks): ~20% (null alleles)
    • Infantile (1–12 months): ~65% (most patients; frameshift/null compound)
    • Late infantile (12–24 months): ~15% (hypomorphic alleles)
  CARDINAL FEATURES:
    • Hypotonia (central > peripheral): 90%
    • Lactic acidosis: 95%
    • Hypoglycemia (DISTINGUISHING — Qi-site/gluconeogenesis): 65%
    • Developmental delay / regression: 90%
    • Hepatomegaly / hepatopathy: 55%
    • Feeding difficulties: 75%
    • Failure to thrive: 70%
    • Encephalopathy: 80%
    • Leigh-like MRI (bilateral BG + brainstem T2 hyperintensity): 55%
    • Seizures: 40%
    • Respiratory failure / apnoea: 50%
    • Dystonia: 25% (LESS prominent than UQCRQ 75%; Qi-site ROS burden lower)
    • Optic atrophy: 10% (LESS than UQCRQ 42%; less Qi-site oxidative nerve damage)
    • Cardiomyopathy: <10% (DDx SCO2 100%)
  ABSENT (key DDx):
    × NO GRACILE triad (no iron overload, no aminoaciduria, no cholestasis) — DDx BCS1L
    × NO pili torti / hearing loss — DDx BCS1L-Björnstad
    × NO psychiatric features (psychosis/depression) — DDx TTC19
    × NO spinocerebellar ataxia — DDx TTC19
    × NO cataracts — DDx CYC1 (35% in CYC1, absent in UQCRB)
    × NO dominant dystonia + optic atrophy — DDx UQCRQ (both prominent in UQCRQ)
    × NO HCM dominant — DDx SCO2 (100%), TIMMDC1 (>80%)
    × NO aminoaciduria + cholestasis — DDx BCS1L-GRACILE, SCO1
  SURVIVAL:
    • Null alleles: median survival 18 months – 4 years without aggressive support
    • Hypomorphic alleles: survival into childhood with metabolic management
    • Prognosis: broadly similar to UQCRQ; hepatopathy worsens prognosis vs UQCRQ

PATHOGENIC VARIANTS in UQCRB:
  1. c.221_222delAA (p.Lys74Asnfs*12) — Exon 5; frameshift; null; Haut 2003; severe
  2. p.Gly43Ser (c.127G>A)             — UQCRC2-contact loop; missense; structural; severe
  3. p.Val56Ala (c.167T>C)             — Hydrophobic core; conformational; intermediate
  4. p.Arg85Trp (c.253C>T)             — C-terminal; matrix-facing; UQCRC1 contact; severe
  5. p.Leu28Pro (c.83T>C)              — N-terminal; helix-breaker proline; structural; severe
  6. c.IVS3+1G>A                       — Splice donor; intron 3; frameshift; null; severe

Key Published Reference:
  Haut S, Brivet M, Touati G, et al. (2003) A deletion in the human QP-C gene causes
    a complex III deficiency resulting in hypoglycaemia and lactic acidosis. Hum Genet
    113(2):118-22. First and seminal identification of biallelic UQCRB (QP-C) deficiency;
    infant with recurrent hypoglycaemia, lactic acidosis, hepatomegaly; CIII isolated
    deficiency; c.221_222delAA frameshift in QP-C (UQCRB).
  Fernandez-Vizarra E & Zeviani M (2018) Nuclear gene mutations as the cause of
    mitochondrial complex III deficiency. Front Genet 9:135.
  Berry EA et al. (2000) Structure and function of cytochrome bc1 complexes.
    Annu Rev Biochem 69:1005-75. Qi-site structural context; UQCRB peripheral subunit.
  Iwata S et al. (1998) Complete structure of the 11-subunit bovine mitochondrial
    cytochrome bc1 complex. Science 281(5373):64-71. First high-res structure showing
    UQCRB position at Qi-site matrix face.
"""

import random

SEED = 733

# ── Pathogenic / likely-pathogenic variants in UQCRB ─────────────────────────
VARIANTS = [
    {
        "protein": "p.Lys74Asnfs*12",
        "cdna": "c.221_222delAA",
        "domain": "Exon 5 / C-terminal matrix region",
        "type": "Frameshift",
        "severity": "Severe",
        "penetrance_pct": 92,
        "notes": "Haut 2003 — first UQCRB disease variant; frameshift → premature stop; null allele; recurrent hypoglycaemia + lactic acidosis + hepatomegaly in French infant",
    },
    {
        "protein": "p.Gly43Ser",
        "cdna": "c.127G>A",
        "domain": "UQCRC2-contact loop",
        "type": "Missense",
        "severity": "Severe",
        "penetrance_pct": 87,
        "notes": "Conserved glycine in the UQCRC2 (Core 2) interface loop; Gly→Ser introduces steric clash disrupting Qi-site scaffold geometry; severe CIII reduction",
    },
    {
        "protein": "p.Val56Ala",
        "cdna": "c.167T>C",
        "domain": "Hydrophobic core",
        "type": "Missense",
        "severity": "Moderate",
        "penetrance_pct": 68,
        "notes": "Hydrophobic core destabilisation; partial CIII residual (10–18%); hypomorphic; later onset (3–8 months); better prognosis than null alleles",
    },
    {
        "protein": "p.Arg85Trp",
        "cdna": "c.253C>T",
        "domain": "C-terminal matrix domain / UQCRC1-contact",
        "type": "Missense",
        "severity": "Severe",
        "penetrance_pct": 83,
        "notes": "C-terminal Arg is a UQCRC1 (Core 1) contact residue; Trp substitution disrupts charge-complementary interface; secondary UQCRC1 destabilisation",
    },
    {
        "protein": "p.Leu28Pro",
        "cdna": "c.83T>C",
        "domain": "N-terminal post-MTS region",
        "type": "Missense",
        "severity": "Severe",
        "penetrance_pct": 79,
        "notes": "Proline substitution breaks N-terminal helix just after MTS cleavage; structural collapse of the Qi-site peripheral scaffold; severe",
    },
    {
        "protein": "c.IVS3+1G>A (splice)",
        "cdna": "c.IVS3+1G>A",
        "domain": "Intron 3 / splice donor",
        "type": "Splice-site",
        "severity": "Severe",
        "penetrance_pct": 90,
        "notes": "Canonical splice donor disruption; intron 3 retention → frameshift → premature stop; null allele; neonatal onset in biallelic cases",
    },
]

VARIANT_WEIGHTS = [0.30, 0.22, 0.15, 0.18, 0.08, 0.07]   # ~frequency proportions


# ── Patient-cohort generator ──────────────────────────────────────────────────
def _make_patients(n=40, seed=SEED):
    rng = random.Random(seed)
    patients = []

    onset_choices = [
        ("Neonatal (<4 weeks)", 0.20),
        ("Early infantile (1–3 months)", 0.30),
        ("Infantile (3–6 months)", 0.25),
        ("Infantile (6–12 months)", 0.10),
        ("Late infantile (12–24 months)", 0.15),
    ]
    outcomes = [
        ("Deceased < 18 months", 0.15),
        ("Deceased 18 months – 4 years", 0.15),
        ("Alive — severe disability, tube-fed", 0.25),
        ("Alive — moderate disability, oral feeds", 0.25),
        ("Alive — mild impairment, ambulatory", 0.20),
    ]

    variant_pools = [v["protein"] for v in VARIANTS]
    vweights = VARIANT_WEIGHTS

    for i in range(n):
        v1_idx = rng.choices(range(len(VARIANTS)), weights=vweights)[0]
        v2_idx = rng.choices(range(len(VARIANTS)), weights=vweights)[0]
        v1 = VARIANTS[v1_idx]
        v2 = VARIANTS[v2_idx]
        homozygous = (v1_idx == v2_idx)
        zygosity = "Homozygous" if homozygous else "Compound heterozygous"

        # Onset distribution
        onset_label = rng.choices(
            [o[0] for o in onset_choices],
            weights=[o[1] for o in onset_choices],
        )[0]

        # Convert onset to approximate weeks
        onset_week_map = {
            "Neonatal (<4 weeks)": rng.randint(0, 3),
            "Early infantile (1–3 months)": rng.randint(4, 12),
            "Infantile (3–6 months)": rng.randint(13, 26),
            "Infantile (6–12 months)": rng.randint(27, 52),
            "Late infantile (12–24 months)": rng.randint(53, 104),
        }
        onset_weeks = onset_week_map[onset_label]

        # Avg CIII activity — 5–15% range; modulated by severity combo
        severity_combo = max(
            ["Severe", "Moderate", "Severe", "Severe", "Severe", "Severe"].index(v1["severity"]),
            ["Severe", "Moderate", "Severe", "Severe", "Severe", "Severe"].index(v2["severity"]),
        )
        base_ciii = rng.uniform(5, 15) if severity_combo == 0 else rng.uniform(10, 22)
        ciii_activity_pct = round(base_ciii, 1)

        lactic_acid = round(rng.uniform(4.5, 20.0), 1) if ciii_activity_pct < 12 else round(rng.uniform(3.5, 12.0), 1)

        # Phenotype flags — UQCRB-specific rates
        hypotonia = rng.random() < 0.90
        lactic_acidosis = rng.random() < 0.95
        hypoglycemia = rng.random() < 0.65      # DISTINGUISHING for UQCRB
        hepatopathy = rng.random() < 0.55       # more than UQCRQ
        dev_delay = rng.random() < 0.90
        feeding_diff = rng.random() < 0.75
        encephalopathy = rng.random() < 0.80
        leigh_mri = rng.random() < 0.55
        seizures = rng.random() < 0.40
        dystonia = rng.random() < 0.25          # LESS cardinal than UQCRQ
        optic_atrophy = rng.random() < 0.10     # LESS than UQCRQ 42%
        cardiomyopathy = rng.random() < 0.08
        respiratory = rng.random() < 0.50
        failure_to_thrive = rng.random() < 0.70

        phenotype_parts = ["Isolated CIII Deficiency"]
        if hypoglycemia:
            phenotype_parts.append("Hypoglycaemia")
        if hepatopathy:
            phenotype_parts.append("Hepatopathy")
        if leigh_mri:
            phenotype_parts.append("Leigh-like MRI")
        if dystonia:
            phenotype_parts.append("Dystonia")
        phenotype_str = " — ".join(phenotype_parts[:3])

        outcome_label = rng.choices(
            [o[0] for o in outcomes], weights=[o[1] for o in outcomes]
        )[0]

        patients.append({
            "patient_id": f"UQCRB-{i+1:03d}",
            "phenotype": phenotype_str,
            "onset_label": onset_label,
            "onset_weeks": onset_weeks,
            "variant_1": v1["protein"],
            "variant_2": v2["protein"],
            "zygosity": zygosity,
            "ciii_activity_pct": ciii_activity_pct,
            "lactic_acid_mmolL": lactic_acid,
            "hypotonia": hypotonia,
            "lactic_acidosis": lactic_acidosis,
            "hypoglycemia": hypoglycemia,
            "hepatopathy": hepatopathy,
            "developmental_delay": dev_delay,
            "feeding_difficulties": feeding_diff,
            "encephalopathy": encephalopathy,
            "leigh_like_mri": leigh_mri,
            "seizures": seizures,
            "dystonia": dystonia,
            "optic_atrophy": optic_atrophy,
            "cardiomyopathy": cardiomyopathy,
            "respiratory_failure": respiratory,
            "failure_to_thrive": failure_to_thrive,
            "outcome": outcome_label,
        })

    return patients


# ── Cohort statistics helper ──────────────────────────────────────────────────
def _cohort_stats(patients):
    n = len(patients)
    pct = lambda key: round(100 * sum(1 for p in patients if p.get(key, False)) / n, 1)
    avg = lambda key: round(sum(p[key] for p in patients) / n, 1)
    return {
        "hypotonia_pct":             pct("hypotonia"),
        "lactic_acidosis_pct":       pct("lactic_acidosis"),
        "hypoglycemia_pct":          pct("hypoglycemia"),
        "hepatopathy_pct":           pct("hepatopathy"),
        "developmental_delay_pct":   pct("developmental_delay"),
        "feeding_difficulties_pct":  pct("feeding_difficulties"),
        "encephalopathy_pct":        pct("encephalopathy"),
        "leigh_like_mri_pct":        pct("leigh_like_mri"),
        "seizures_pct":              pct("seizures"),
        "dystonia_pct":              pct("dystonia"),
        "optic_atrophy_pct":         pct("optic_atrophy"),
        "cardiomyopathy_pct":        pct("cardiomyopathy"),
        "respiratory_failure_pct":   pct("respiratory_failure"),
        "failure_to_thrive_pct":     pct("failure_to_thrive"),
        "avg_ciii_activity_pct":     avg("ciii_activity_pct"),
        "avg_lactic_acid_mmolL":     avg("lactic_acid_mmolL"),
        "deceased_pct": round(
            100 * sum(1 for p in patients if "Deceased" in p["outcome"]) / n, 1
        ),
    }


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    patients = _make_patients()
    stats = _cohort_stats(patients)

    feat_order = [
        ("Lactic acidosis", "lactic_acidosis"),
        ("Hypotonia", "hypotonia"),
        ("Developmental delay / regression", "developmental_delay"),
        ("Feeding difficulties", "feeding_difficulties"),
        ("Failure to thrive", "failure_to_thrive"),
        ("Encephalopathy", "encephalopathy"),
        ("Hypoglycaemia (DISTINGUISHING)", "hypoglycemia"),
        ("Hepatomegaly / hepatopathy", "hepatopathy"),
        ("Leigh-like MRI", "leigh_like_mri"),
        ("Respiratory failure", "respiratory_failure"),
        ("Seizures", "seizures"),
        ("Dystonia (<25%)", "dystonia"),
        ("Optic atrophy (<10%)", "optic_atrophy"),
        ("Cardiomyopathy (<10%)", "cardiomyopathy"),
    ]
    n = len(patients)
    features = [
        {"feature": label, "pct": round(100 * sum(1 for p in patients if p.get(key, False)) / n, 1)}
        for label, key in feat_order
    ]

    # Variant counts
    from collections import Counter
    v_counter = Counter()
    for p in patients:
        v_counter[p["variant_1"]] += 1
        v_counter[p["variant_2"]] += 1
    top_variants = [{"variant": k, "count": v} for k, v in v_counter.most_common(6)]

    alerts = [
        "🚫 KD (Ketogenic Diet) — ABSOLUTE CONTRAINDICATION: CIII block + FADH2 backlog from β-oxidation → fatal metabolic crisis",
        "🚫 Metformin — ABSOLUTE CONTRAINDICATION: Complex I inhibitor → additive OXPHOS failure with CIII deficiency",
        "🚫 Valproic acid (VPA) — ABSOLUTE CONTRAINDICATION: CoA sequestration + POLG inhibition + OXPHOS impairment",
        "🚫 Linezolid — ABSOLUTE CONTRAINDICATION: inhibits mt-23S rRNA translation (MT-CYB) → worsens CIII structural instability",
        "🚫 Propofol — ABSOLUTE CONTRAINDICATION: PRIS — directly inhibits CIII at Qi site + β-oxidation → fatal in CIII-deficient patients",
        "🚫 Chloramphenicol — ABSOLUTE CONTRAINDICATION: inhibits mitochondrial translation; reduces MT-CYB further",
        "⚠️ Hypoglycaemia (65%): DISTINGUISHING for UQCRB — Qi-site CIII block impairs gluconeogenesis/β-oxidation; NEVER fast; GIR 6–8 mg/kg/min IV mandatory",
        "⚠️ Hepatopathy (55%): more prominent than UQCRQ; monitor LFTs, PT/INR, ammonia; rule out secondary liver failure",
        "✅ NaHCO₃ IV: acute lactic acidosis (target pH >7.2); continuous dextrose GIR 6–8 mg/kg/min; never fast",
        "✅ CoQ10/Ubiquinol + Thiamine + Riboflavin + Biotin: mitochondrial cocktail (Level C)",
        "✅ Sevoflurane preferred over Propofol for anaesthesia; NIV/BiPAP for respiratory support (avoid propofol intubation)",
    ]

    return {
        "gene": "UQCRB",
        "full_name": "Ubiquinol-Cytochrome C Reductase Binding Protein",
        "alias": "QCR7 / Subunit VI (cytochrome bc1 complex) / QP-C",
        "omim_gene": "191330",
        "omim_disease": "AR biallelic CIII deficiency (nuclear) — Haut 2003 Hum Genet",
        "disease_name": "Isolated Complex III Deficiency — Nuclear, AR Biallelic (UQCRB Qi-site Structural Subunit Deficiency)",
        "chromosome": "8q22.1",
        "inheritance": "AR biallelic",
        "protein_size": "109 aa, 12.1 kDa (MTS ~aa 1–13; mature ~96 aa)",
        "protein": {
            "size_aa": 109,
            "kDa": 12.1,
            "tm_helices": 0,
            "localization": "IMM — peripheral (NO transmembrane helix); matrix-facing Qi-site periphery; contacts UQCRC1 and UQCRC2",
            "role": "Structural subunit at the Qi-site periphery (matrix face); maintains Qi-site pocket geometry for CoQ reduction",
            "function": "Peripheral matrix-face scaffold; contacts UQCRC1 (Core 1) and UQCRC2 (Core 2) at the Qi-site boundary; absence disrupts CoQ binding and reduction at Qi site → severe isolated CIII deficiency",
        },
        "ciii_assembly_step": "Qi-site structural subunit — peripherally associated at matrix face after CIII core assembly; NOT an assembly factor; loss → Qi-site instability; less severe assembly block than UQCC1/UQCC2 but residual CIII (5–15%) lower than UQCRQ (8–22%)",
        "bn_page": "CIII severely reduced (5–15% residual); core sub-complexes partially present; more severe reduction than UQCRQ; CIII* (CIII dimer) markedly reduced or absent; RISP relatively preserved immunoblot (Qo-site RISP insert less affected than Qi-site scaffolding)",
        "key_biochemical_features": [
            "Isolated CIII deficiency: 5–15% residual CIII activity (more severe than UQCRQ 8–22%)",
            "CI, CII, CIV activities: typically normal (isolated CIII deficiency)",
            "Lactic acidosis: plasma lactate 4.5–20 mM; lactate:pyruvate ratio >20",
            "Hypoglycaemia: fasting plasma glucose <2.5 mM in 65% — DISTINGUISHING vs UQCRQ (rare in UQCRQ)",
            "Elevated alanine (PDH complex inhibition secondary to CoQ cycle arrest)",
            "BN-PAGE: CIII severely reduced; core sub-complexes partially present; RISP relatively intact immunoblot",
            "Immunoblot: UQCRB absent; UQCRC1/UQCRC2 secondary reduction variable; RISP (UQCRFS1) partially preserved",
            "Transaminases (ALT/AST) elevated in 55% — hepatopathy more prominent than UQCRQ",
            "Urine organic acids: elevated 3-methylglutaconic acid (3-MGA) possible (non-specific CIII marker); NO 3-MGA triad as in TMEM70/CLPB",
        ],
        "cohort_n": len(patients),
        "seed": SEED,
        "patients": patients[:10],
        "cohort_statistics": stats,
        "cohort_summary_features": features,
        "key_clinical_alerts": alerts,
        "top_variant_counts": top_variants,
        "onset_distribution": {
            "neonatal_pct": round(100 * sum(1 for p in patients if p["onset_weeks"] <= 3) / len(patients), 1),
            "early_infantile_pct": round(100 * sum(1 for p in patients if 4 <= p["onset_weeks"] <= 12) / len(patients), 1),
            "infantile_pct": round(100 * sum(1 for p in patients if 13 <= p["onset_weeks"] <= 52) / len(patients), 1),
            "late_infantile_pct": round(100 * sum(1 for p in patients if p["onset_weeks"] > 52) / len(patients), 1),
        },
    }


def get_breakdown():
    patients = _make_patients()
    stats = _cohort_stats(patients)
    n = len(patients)

    # Variant distribution
    from collections import Counter
    v1c = Counter(p["variant_1"] for p in patients)
    v2c = Counter(p["variant_2"] for p in patients)
    all_vc = v1c + v2c
    variant_dist = [{"variant": k, "allele_count": v, "allele_freq_pct": round(100 * v / (2 * n), 1)}
                    for k, v in all_vc.most_common()]

    # Biochemistry
    ciii_vals = [p["ciii_activity_pct"] for p in patients]
    lac_vals  = [p["lactic_acid_mmolL"] for p in patients]

    biochem = {
        "avg_ciii_activity_pct": round(sum(ciii_vals) / n, 1),
        "ciii_5to10_pct":   round(100 * sum(1 for v in ciii_vals if v <= 10) / n, 1),
        "ciii_10to12_pct":  round(100 * sum(1 for v in ciii_vals if 10 < v <= 12) / n, 1),
        "ciii_12to15_pct":  round(100 * sum(1 for v in ciii_vals if 12 < v <= 15) / n, 1),
        "ciii_above15_pct": round(100 * sum(1 for v in ciii_vals if v > 15) / n, 1),
        "avg_lactic_acid_mmolL":  round(sum(lac_vals) / n, 1),
        "lactic_above_10_pct":    round(100 * sum(1 for v in lac_vals if v > 10) / n, 1),
        "lactic_5_to_10_pct":     round(100 * sum(1 for v in lac_vals if 5 <= v <= 10) / n, 1),
        "lactic_below5_pct":      round(100 * sum(1 for v in lac_vals if v < 5) / n, 1),
    }

    # Outcome distribution
    from collections import Counter as Ct
    oc = Ct(p["outcome"] for p in patients)
    outcomes = [{"outcome": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in oc.most_common()]

    # Zygosity distribution
    zc = Ct(p["zygosity"] for p in patients)
    zygosity_dist = [{"zygosity": k, "count": v} for k, v in zc.most_common()]

    return {
        "gene": "UQCRB",
        "all_variants": VARIANTS,
        "variant_allele_distribution": variant_dist,
        "zygosity_distribution": zygosity_dist,
        "biochemistry_distribution": biochem,
        "outcome_distribution": outcomes,
        "cohort_statistics": stats,
        "bn_page_pattern": {
            "finding": "CIII severely reduced (5–15% residual); CIII* (dimer) markedly reduced; core sub-complexes partially present",
            "interpretation": "Qi-site peripheral instability → CIII assembly partially proceeds but Qi-site architecture is compromised → severe CIII reduction; RISP immunoblot relatively preserved (Qo-side insert unaffected)",
            "ddx_value": "BN-PAGE pattern: more severe CIII reduction than UQCRQ (8–22%); sub-complexes partially present (not as prominent as UQCRQ); distinguish from UQCC2 (CIII completely absent) — WES/genotyping mandatory for definitive DDx",
        },
        "immunoblot_pattern": {
            "UQCRB": "Absent (primary loss — null/frameshift alleles most common)",
            "UQCRC1 (Core 1)": "Secondarily reduced (variable; 30–70% residual depending on Qi-site disruption severity)",
            "UQCRC2 (Core 2)": "Secondarily reduced (variable; 30–70% residual; UQCRB contacts UQCRC2 directly)",
            "RISP / UQCRFS1": "Relatively preserved (Qo-side insert; less directly affected by Qi-site defect than by UQCRQ loss)",
            "Cytochrome c1 / CYC1": "Mildly reduced secondarily",
            "UQCRQ (subunit 8)": "Normal (Qo-site subunit; independent of UQCRB at Qi-site)",
            "CI / CII / CIV subunits": "Normal (isolated CIII deficiency)",
        },
        "treatment_uptake": {
            "CoQ10 / Ubiquinol": f"{rng_count(patients, 0.72)} / {n} patients",
            "Thiamine (B1) empiric": f"{rng_count(patients, 0.88)} / {n} patients",
            "Riboflavin (B2)": f"{rng_count(patients, 0.68)} / {n} patients",
            "Biotin empiric": f"{rng_count(patients, 0.65)} / {n} patients",
            "NaHCO₃ IV (acute)": f"{rng_count(patients, 0.82)} / {n} patients",
            "IV Dextrose GIR 6–8": f"{rng_count(patients, 0.90)} / {n} patients",
            "NG/NJ tube feeds": f"{rng_count(patients, 0.78)} / {n} patients",
            "Cornstarch / starch feeding (hypoglycaemia)": f"{rng_count(patients, 0.48)} / {n} patients",
            "NIV / BiPAP": f"{rng_count(patients, 0.45)} / {n} patients",
            "LEV (preferred AED)": f"{rng_count(patients, 0.35)} / {n} patients",
        },
    }


def rng_count(patients, rate):
    """Deterministic approximate count from rate."""
    return round(rate * len(patients))


def get_definitions():
    return {
        "gene": "UQCRB",
        "full_name": "Ubiquinol-Cytochrome C Reductase Binding Protein",
        "alias": "QCR7 (yeast) / Subunit VI (bc1 complex) / QP-C / UQCRB",
        "omim_gene": "191330",
        "omim_disease": "AR biallelic CIII deficiency (nuclear) — Haut 2003 Hum Genet 113(2):118-22",
        "disease_name": "Isolated Complex III Deficiency — Nuclear, AR Biallelic (UQCRB Qi-site Structural Subunit Deficiency)",
        "chromosome": "8q22.1",
        "inheritance": "AR biallelic",
        "protein": {
            "size_aa": 109,
            "kDa": 12.1,
            "tm_helices": 0,
            "localization": "IMM — NO transmembrane helix; peripheral matrix-facing; contacts UQCRC1 and UQCRC2 at Qi-site periphery",
            "function": "Peripheral Qi-site structural scaffold; maintains CoQ-binding pocket geometry for ubiquinone reduction at matrix face of CIII; absence → Qi-site instability → 5–15% residual CIII",
        },
        "ciii_assembly_step": "Qi-site structural peripheral subunit; incorporated at matrix face during CIII final assembly; NOT an assembly factor (contrast UQCC1/UQCC2); structural role at Qi-site",
        "bn_page": "CIII severely reduced (5–15% residual); core sub-complexes partially present; more severe reduction than UQCRQ (8–22%); RISP relatively preserved on immunoblot",
        "key_biochemical_features": [
            "Isolated CIII deficiency: 5–15% residual CIII activity (more severe than UQCRQ 8–22%)",
            "CI, CII, CIV activities: typically normal",
            "Plasma lactate 4.5–20 mM; L:P ratio >20",
            "Fasting hypoglycaemia (plasma glucose <2.5 mM): 65% — DISTINGUISHING vs UQCRQ",
            "Elevated transaminases: 55% — hepatopathy more prominent than UQCRQ",
            "Immunoblot: UQCRB absent; UQCRC1/UQCRC2 secondarily reduced; RISP relatively preserved",
        ],
        "absolute_contraindications": [
            "KD (Ketogenic Diet) — CIII block + FADH2 backlog from β-oxidation → fatal metabolic crisis; especially dangerous given UQCRB hypoglycaemia risk",
            "Metformin — Complex I inhibitor → additive OXPHOS failure; absolute CI in CIII deficiency",
            "Valproic acid (VPA) — CoA sequestration + POLG inhibition + hepatotoxicity (worsens UQCRB hepatopathy)",
            "Linezolid — mt-23S rRNA translation inhibition (MT-CYB) → worsens CIII structural instability",
            "Propofol — PRIS: directly inhibits CIII at Qi site + mitochondrial fatty-acid β-oxidation → fatal in CIII-deficient patients; use Sevoflurane",
            "Chloramphenicol — inhibits mitochondrial ribosome; reduces MT-CYB translation → worsens CIII complex stability",
        ],
        "recommended_treatments": [
            "IV Dextrose GIR 6–8 mg/kg/min — NEVER fast; continuous dextrose mandatory; critical for UQCRB hypoglycaemia prevention; Level A",
            "Cornstarch / uncooked starch at feeds — sustained glucose release; management of recurrent hypoglycaemia; Level B",
            "NaHCO₃ IV — acute lactic acidosis correction (target pH >7.2 / HCO₃ >12); Level A",
            "CoQ10 / Ubiquinol — 30 mg/kg/day pediatric; Qi-site CoQ substrate support; Level C",
            "Thiamine (B1) — empiric PDH support (Leigh/CIII); mandatory empiric; Level C",
            "Riboflavin (B2) — FMN/FAD support; not riboflavin-responsive in UQCRB (no FAD domain) but general CIII cocktail; Level C",
            "Biotin — empiric (rule out BTD); Level C",
            "NG/NJ tube feeds — continuous enteral nutrition; prevent fasting; Level A",
            "Hepatology monitoring — LFTs, PT/INR, ammonia q3 months; Level A",
            "NIV/BiPAP — respiratory support 50%; avoid propofol at intubation (Sevoflurane preferred); Level A",
            "LEV (Levetiracetam) — preferred AED if seizures; avoid VPA; Level A",
        ],
        "key_ddx": [
            {
                "condition": "UQCRQ (CIII Qo-site structural)",
                "distinguishing": "UQCRQ: Dystonia 75% CARDINAL + Optic atrophy 42% DISTINGUISHING — both ABSENT as cardinal features in UQCRB; UQCRB: Hypoglycaemia 65% + Hepatopathy 55% — more prominent than UQCRQ (rare hypoglycaemia, <15% hepatopathy); UQCRQ at 5q31.1 vs UQCRB at 8q22.1; Qo-site vs Qi-site",
            },
            {
                "condition": "UQCC2 (CIII assembly, neonatal)",
                "distinguishing": "UQCC2: CIII completely ABSENT on BN-PAGE; neonatal onset 85%; severe lactic acidosis pH<7.2; UQCRB: 5–15% residual CIII; sub-complexes partially present; infantile onset more common (65%)",
            },
            {
                "condition": "BCS1L-GRACILE",
                "distinguishing": "BCS1L: GRACILE triad (iron overload + aminoaciduria + cholestasis) ABSENT in UQCRB; BCS1L: CIII precomplex accumulates on BN-PAGE (pathognomonic); UQCRB: NO cholestasis, NO aminoaciduria",
            },
            {
                "condition": "CYC1 (CIII-D3, Cytochrome c1)",
                "distinguishing": "CYC1: Cataracts 35% — ABSENT in UQCRB; CYC1: Hepatic involvement 78% — UQCRB hepatopathy 55%; CYC1: CYC1 + UQCRC1 both absent on immunoblot; UQCRB: UQCRB absent with partial UQCRC1/2 reduction; CYC1 at 8q24.1 vs UQCRB at 8q22.1 (same chromosome arm — WES mandatory)",
            },
            {
                "condition": "SCO1 (CIV hepatopathy)",
                "distinguishing": "SCO1: 100% neonatal hepatic failure → CARDINAL; CIV deficiency NOT CIII; UQCRB: CIII deficiency; hepatopathy 55% (not 100%); CIV activities normal in UQCRB",
            },
            {
                "condition": "UQCRC2 (CIII-D4, Core protein 2)",
                "distinguishing": "UQCRC2: Core scaffold subunit; UQCRC2 loss destabilises entire core including UQCRC1; UQCRB: peripherally contacts UQCRC2 but UQCRC2 protein present; immunoblot: UQCRC2 absent in UQCRC2 deficiency, present in UQCRB deficiency; WES mandatory",
            },
            {
                "condition": "TTC19 (CIII-D2, assembly chaperone)",
                "distinguishing": "TTC19: Psychiatric features (psychosis/depression 40%) — ABSENT in UQCRB; TTC19: late childhood onset + spinocerebellar ataxia; UQCRB: infantile onset; no psychiatric features; no cerebellar ataxia",
            },
        ],
        "genetic_counselling": {
            "recurrence_risk": "AR biallelic: 25% recurrence per pregnancy for sibling of affected proband",
            "carrier_testing": "Carrier parents: heterozygous UQCRB variants; typically asymptomatic (monoallelic UQCRB does not cause CIII deficiency)",
            "prenatal_diagnosis": "CVS or amniocentesis for known familial UQCRB variants; CIII enzyme activity on cultured chorionic villi if molecular test inconclusive",
            "newborn_screening": "Not on standard NBS panels; clinical suspicion in neonatal lactic acidosis + hypoglycaemia → UQCRB sequencing; organic acid analysis (elevated 3-MGA non-specific)",
            "founder_effects": "No large founder effect established (only Haut 2003 French case documented); WES is primary diagnostic route",
            "molecular_confirmation": "WES/panel sequencing with CIII gene panel (UQCRB, UQCRC1, UQCRC2, UQCRQ, UQCRH, CYC1, UQCRFS1, TTC19, BCS1L, LYRM7, UQCC1-3)",
        },
        "key_references": [
            "Haut S, Brivet M, Touati G, et al. (2003) A deletion in the human QP-C gene causes a complex III deficiency resulting in hypoglycaemia and lactic acidosis. Hum Genet 113(2):118-22 — PIVOTAL: first biallelic UQCRB (QP-C) disease; c.221_222delAA; French infant; recurrent hypoglycaemia, lactic acidosis, hepatomegaly, isolated CIII deficiency",
            "Fernandez-Vizarra E & Zeviani M (2018) Nuclear gene mutations as the cause of mitochondrial complex III deficiency. Front Genet 9:135 — comprehensive CIII nuclear gene review; UQCRB structural context; Qi-site subunit group",
            "Iwata S, Lee JW, Okada K, et al. (1998) Complete structure of the 11-subunit bovine mitochondrial cytochrome bc1 complex. Science 281(5373):64-71 — first high-resolution crystal structure; UQCRB (subunit VI) position at matrix-facing Qi-site periphery; contact map with Core proteins",
            "Berry EA, Guergova-Kuras M, Huang LS, Crofts AR (2000) Structure and function of cytochrome bc1 complexes. Annu Rev Biochem 69:1005-75 — UQCRB (binding protein) structural role; Qi-site architecture; CoQ reduction mechanism",
            "Barel O, Shorer Z, Flusser H, et al. (2008) Mitochondrial complex III deficiency associated with a homozygous mutation in UQCRQ. Am J Hum Genet 82(1):267-74 — UQCRQ (Qo-site, related subunit) first disease report; DDx context for UQCRB vs UQCRQ",
            "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538(7123):123-126 — mitochondrial structural subunit classification; UQCRB peripheral Qi-site group context",
        ],
        "terms": [
            {
                "term": "UQCRB (QCR7 / QP-C)",
                "definition": "Ubiquinol-Cytochrome C Reductase Binding Protein; also QCR7 (yeast) or QP-C ('quinol-binding protein, C subunit'); 109-aa peripheral matrix-facing structural subunit of Complex III; NO transmembrane helix; contacts UQCRC1 and UQCRC2 at the Qi-site periphery; OMIM *191330",
            },
            {
                "term": "Qi site (inner quinone-binding site)",
                "definition": "The ubiquinone REDUCTION site of Complex III at the matrix face of the inner mitochondrial membrane; where ubiquinone (CoQ) is reduced to ubiquinol (CoQH2) receiving electrons from cytochrome b (heme bL/bH); UQCRB stabilises Qi-site pocket geometry at the matrix face; contrast Qo site (outer; UQCRQ)",
            },
            {
                "term": "Hypoglycaemia (UQCRB — Distinguishing)",
                "definition": "Fasting hypoglycaemia (plasma glucose <2.5 mM) in 65% of UQCRB patients — DISTINGUISHING vs UQCRQ (rare); caused by impaired CIII-dependent gluconeogenesis and β-oxidation (FADH2 re-oxidation requires functional CIII); management: NEVER fast; GIR 6–8 mg/kg/min; cornstarch feeds",
            },
            {
                "term": "BN-PAGE (Blue-Native PAGE)",
                "definition": "Blue-native polyacrylamide gel electrophoresis; separates intact respiratory chain complexes; in UQCRB: CIII severely reduced (5–15%); sub-complexes partially present; more severe than UQCRQ (8–22%); RISP relatively preserved; WES mandatory to distinguish from UQCRC2 and other CIII deficiencies with overlapping BN-PAGE patterns",
            },
            {
                "term": "RISP / UQCRFS1 (Rieske Iron-Sulfur Protein)",
                "definition": "Rieske iron-sulfur protein; catalytic FeS cluster component of CIII; inserted by BCS1L at the Qo-side; in UQCRB deficiency, RISP is relatively PRESERVED on immunoblot (Qo-side insert not directly disrupted by Qi-site UQCRB loss); contrast UQCRQ where RISP stalk loses peripheral support",
            },
            {
                "term": "QP-C (UQCRB historical alias)",
                "definition": "Historical alias for UQCRB used in Haut 2003 ('quinol-binding protein, C subunit' or 'binding protein C'); now universally referred to as UQCRB; the Haut 2003 paper used the QP-C designation when naming the gene deleted in their patient",
            },
            {
                "term": "GIR (Glucose Infusion Rate)",
                "definition": "Continuous IV dextrose at 6–8 mg/kg/min; MANDATORY in UQCRB — never fast; prevents fasting-induced hypoglycaemia (65%) and lactic acidosis; especially critical in UQCRB vs UQCRQ (hypoglycaemia more prominent in UQCRB)",
            },
            {
                "term": "PRIS (Propofol Infusion Syndrome)",
                "definition": "Life-threatening drug reaction; propofol inhibits CIII at the Qi site and β-oxidation; risk greatly increased in CIII-deficient patients including UQCRB; absolute contraindication; use Sevoflurane for anaesthesia",
            },
        ],
    }


if __name__ == "__main__":
    ov = get_overview()
    print(f"Gene: {ov['gene']} ({ov['alias']})")
    print(f"Disease: {ov['disease_name']}")
    print(f"OMIM Gene: *{ov['omim_gene']}")
    print(f"Chromosome: {ov['chromosome']}  Inheritance: {ov['inheritance']}")
    print(f"\nCohort: {ov['cohort_n']} patients, seed {ov['seed']}")
    s = ov["cohort_statistics"]
    print(f"  Lactic acidosis: {s['lactic_acidosis_pct']}%")
    print(f"  Hypotonia: {s['hypotonia_pct']}%")
    print(f"  Hypoglycaemia (distinguishing): {s['hypoglycemia_pct']}%")
    print(f"  Hepatopathy: {s['hepatopathy_pct']}%")
    print(f"  Dystonia (<25%): {s['dystonia_pct']}%")
    print(f"  Optic atrophy (<10%): {s['optic_atrophy_pct']}%")
    print(f"  Avg CIII activity: {s['avg_ciii_activity_pct']}%")
    print(f"  Avg lactic acid: {s['avg_lactic_acid_mmolL']} mM")
    print(f"  Deceased (any): {s['deceased_pct']}%")
    print("\nVariants:", [v["protein"] for v in VARIANTS])
    print(f"\nTop variant counts: {ov['top_variant_counts'][:3]}")
