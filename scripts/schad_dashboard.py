#!/usr/bin/env python3
"""SCHAD Deficiency (HADH / Short-Chain 3-Hydroxyacyl-CoA Dehydrogenase Deficiency) Dashboard.

HADH gene encodes SCHAD (Short-Chain 3-Hydroxyacyl-CoA Dehydrogenase):
  HADH: 302 aa; mitochondrial matrix; homodimeric; NAD-dependent
  Locus: 4q22.1; Autosomal Recessive (AR)
  OMIM Gene: *601609
  OMIM Disease: #231530 (SCHAD deficiency / Hyperinsulinaemic hypoglycaemia, familial, 4 / HHF4)

  HADH CATALYSES (mitochondrial matrix — short-chain specificity C4-C8):
    3-Hydroxyacyl-CoA + NAD+ → 3-Ketoacyl-CoA + NADH  (step 3 of beta-oxidation, short-chain)

UNIQUE MECHANISM — HADH-GDH PROTEIN-PROTEIN INTERACTION:
  HADH normally PHYSICALLY INTERACTS with GDH (GLUD1, glutamate dehydrogenase) inside
  the mitochondrial matrix of pancreatic beta cells.
  HADH INHIBITS GDH — prevents excess glutamate oxidation when HADH is engaged in FAO.
  HADH LOF → GDH uninhibited (no HADH protein to bind GDH) →
    Excess glutamate → glutamate oxidised → α-ketoglutarate + NH4+ + NADH + GTP →
    Excess ATP generated in beta cell → KATP channels CLOSE → membrane depolarisation →
    Ca2+ influx → EXCESS INSULIN SECRETED = CONGENITAL HYPERINSULINISM (HI)
  This mechanism is UNIQUE: SCHAD deficiency causes HI primarily through GDH dysregulation,
  NOT through an energy-deficit FAO crisis (unlike MCAD, VLCAD, LCHAD, CPT2).

KEY NBS MARKER — SCHAD PROFILE:
  C4-OH (3-hydroxybutyrylcarnitine)  MILDLY ELEVATED — primary NBS marker (nonspecific)
  Insulin                             ELEVATED (>2 mU/L during hypoglycaemia)
  Glucose                             CRITICALLY LOW (<1.5 mmol/L neonatal; <2.6 infantile)
  Ammonia                             NORMAL — KEY NEGATIVE vs GLUD1/HHS (where 100-500 μmol/L)
  3-hydroxybutyrate (BHBA)            Mildly elevated paradoxically (SCHAD step blocked)
  Leucine                             Normal plasma levels (cf. GLUD1/HHS where leucine TRIGGERS)
  Acylcarnitine C4-OH                 Mildly elevated; not as distinctive as MCAD C8 or VLCAD C14:1

THREE PHENOTYPES:
  1. Classic Neonatal HI (50%): day 1-5; profound hypoglycaemia <1.5 mmol/L;
     diazoxide responsive ~80%; continuous glucose infusion required
  2. Late Infantile HI (35%): 3-24 months; protein-triggered hypoglycaemia;
     episodic; triggered by high-protein feeds
  3. Mild Attenuated (15%): older children; fasting hypoglycaemia only; diet managed

SEIZURE TYPES (secondary to hypoglycaemia):
  GTCS: 60% (neonatal severe hypoglycaemia)
  Focal: 35%
  Infantile spasms (IS)/West: 20%
  Status epilepticus: 15%
  Myoclonic: 12%

TREATMENT (10 key treatments):
  1. Diazoxide: Level A (first line; 70-80% response; opens KATP channels → K+ outflow →
     repolarisation → less Ca2+ influx → less insulin)
  2. Frequent feeds/cornstarch: Level A (nocturnal hypoglycaemia prevention)
  3. Continuous glucose monitoring: Level A (all HI disorders)
  4. Protein moderation (NOT strict restriction): Level B (reduces GDH activation via leucine)
  5. Octreotide: Level B (somatostatin analogue; suppresses insulin secretion if diazoxide fails)
  6. Sirolimus (mTOR inhibitor): Level C (refractory cases; inhibits mTOR pathway in beta cells)
  7. Partial pancreatectomy: Level B (diffuse HI, refractory to medical management only)
  8. LEV for seizures: Level B (secondary seizure management)
  9. FASTING: ABSOLUTE CI (all HI disorders; worsens hypoglycaemia crisis)
  10. VPA: HIGH RISK (insulin secretagogue effect; worsens HI; avoid)

KEY DIFFERENTIALS:
  GLUD1/HHS (hyperinsulinism-hyperammonaemia):
    - Same protein-sensitive HI; leucine-triggered
    - GLUD1: ammonia 100-500 μmol/L ELEVATED (HHS hallmark)
    - HADH: ammonia NORMAL — KEY DISCRIMINATING NEGATIVE
  ABCC8/KCNJ11 (KATP-channel defects):
    - Similar HI phenotype; focal vs diffuse on 18F-DOPA PET
    - NOT protein-sensitive; no GDH interaction
  IBD (isobutyryl-CoA dehydrogenase deficiency):
    - C4-OH nonspecific overlap; NBS similar
    - Benign; no HI; metabolic not endocrine
  GLUD1 vs HADH: SAME protein-sensitive HI; DIFFERENT ammonia (HIGH vs NORMAL)

PREVALENCE: ~1:50,000 hyperinsulinism cases worldwide; HADH-HHF4 one of rarer HI causes
INHERITANCE: Autosomal Recessive (AR); biallelic LOF required
"""

import random

SEED = 285
random.seed(SEED)

# ── Variant table ──────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "variant": "p.His170Arg (c.509A>G)",
        "freq": 35,
        "domain": "HADH-GDH interaction interface",
        "phenotype": "Classic Neonatal HI; disrupts HADH-GDH inhibitory contact; GDH fully uninhibited",
        "note": (
            "MOST COMMON SCHAD/HADH variant (35%). Located at the HADH-GDH protein-protein "
            "interaction interface. Loss of this contact → GDH fully uninhibited → maximal HI. "
            "Neonatal onset; diazoxide responsive ~80%."
        ),
    },
    {
        "variant": "p.Leu147Pro (c.440T>C)",
        "freq": 20,
        "domain": "Dimer interface (homodimer stability)",
        "phenotype": "Classic Neonatal HI; disrupts homodimer → misfolded protein → no HADH-GDH interaction",
        "note": (
            "20% frequency. Destabilises HADH homodimer; misfolded protein cannot bind GDH "
            "or perform catalysis. Null-equivalent phenotype through structural instability."
        ),
    },
    {
        "variant": "p.Arg236Gln (c.707G>A)",
        "freq": 15,
        "domain": "Catalytic site (NAD-binding)",
        "phenotype": "Classic Neonatal HI; severe catalytic LOF; neonatal onset",
        "note": (
            "15% frequency. Abolishes NAD-binding; complete enzymatic LOF. "
            "Without enzymatic activity, HADH cannot engage GDH → full GDH hyperactivity."
        ),
    },
    {
        "variant": "p.Glu96Ter (c.286G>T)",
        "freq": 12,
        "domain": "Premature stop — null allele",
        "phenotype": "Null; neonatal severe; no HADH protein produced",
        "note": "12% frequency. Premature stop at codon 96; transcript likely degraded by NMD. Complete absence of HADH protein.",
    },
    {
        "variant": "p.Val96Met (c.286G>A)",
        "freq": 8,
        "domain": "Catalytic core (beta-sheet)",
        "phenotype": "Mild attenuated; partial LOF; some residual HADH enzymatic activity",
        "note": "8% frequency. Partial LOF; some residual HADH activity → partial GDH inhibition → milder HI phenotype. Fasting-only presentation.",
    },
    {
        "variant": "c.636+1G>A",
        "freq": 10,
        "domain": "Splice donor site (intron 5)",
        "phenotype": "Null; aberrant splicing → no functional HADH",
        "note": "10% frequency. Canonical splice site disrupted → exon skipping or intron retention → no functional HADH protein produced.",
    },
]

# ── Phenotype distribution ─────────────────────────────────────────────────────
PHENOTYPE_DIST = {
    "Classic Neonatal HI (day 1-5; profound hypoglycaemia; diazoxide responsive ~80%)": 20,
    "Late Infantile HI (3-24 months; protein-triggered; episodic)": 14,
    "Mild Attenuated (older children; fasting hypoglycaemia only)": 6,
}

# ── Seizure type data ─────────────────────────────────────────────────────────
SEIZURE_TYPES = ["GTCS", "Focal", "IS/West", "Status epilepticus", "Myoclonic"]
SEIZURE_PROBS = [0.60, 0.35, 0.20, 0.15, 0.12]


def _make_patient(i):
    """Synthetic SCHAD/HADH patient record (seed=285, deterministic)."""
    rng = random.Random(SEED + i * 53)

    # Assign phenotype
    if i < 20:
        phenotype = "Classic Neonatal HI"
        onset_months = round(rng.uniform(0.02, 0.5), 3)   # Day 1-15
        glucose_nadir = round(rng.uniform(0.4, 1.5), 2)   # <1.5 mmol/L
        insulin_mU = round(rng.uniform(4.0, 35.0), 1)     # Markedly elevated
        c4oh = round(rng.uniform(0.5, 1.8), 2)            # Mildly elevated
        ammonia = round(rng.uniform(20, 55), 0)            # NORMAL (KEY NEGATIVE vs GLUD1)
        bhba = round(rng.uniform(0.3, 1.8), 2)            # Paradoxically mildly elevated
        protein_sensitive = rng.random() < 0.70
        diazoxide_responsive = rng.random() < 0.80
        variant1 = rng.choice([
            "p.His170Arg", "p.Leu147Pro", "p.Arg236Gln", "p.Glu96Ter", "c.636+1G>A",
        ])
        variant2 = rng.choice([
            "p.His170Arg", "p.Leu147Pro", "p.Arg236Gln", "p.Glu96Ter", "c.636+1G>A",
        ])

    elif i < 34:
        phenotype = "Late Infantile HI"
        onset_months = round(rng.uniform(3, 24), 1)
        glucose_nadir = round(rng.uniform(1.0, 2.6), 2)
        insulin_mU = round(rng.uniform(2.0, 18.0), 1)
        c4oh = round(rng.uniform(0.35, 1.2), 2)
        ammonia = round(rng.uniform(18, 48), 0)            # NORMAL
        bhba = round(rng.uniform(0.2, 1.1), 2)
        protein_sensitive = rng.random() < 0.90
        diazoxide_responsive = rng.random() < 0.75
        variant1 = rng.choice([
            "p.His170Arg", "p.Leu147Pro", "p.Arg236Gln", "p.Val96Met",
        ])
        variant2 = rng.choice([
            "p.His170Arg", "p.Leu147Pro", "p.Val96Met", "c.636+1G>A",
        ])

    else:
        phenotype = "Mild Attenuated"
        onset_months = round(rng.uniform(12, 72), 1)
        glucose_nadir = round(rng.uniform(2.0, 3.2), 2)
        insulin_mU = round(rng.uniform(2.0, 8.0), 1)
        c4oh = round(rng.uniform(0.25, 0.8), 2)
        ammonia = round(rng.uniform(15, 40), 0)            # NORMAL
        bhba = round(rng.uniform(0.1, 0.6), 2)
        protein_sensitive = rng.random() < 0.50
        diazoxide_responsive = rng.random() < 0.85
        variant1 = rng.choice(["p.Val96Met", "p.His170Arg"])
        variant2 = rng.choice(["p.Val96Met", "p.Leu147Pro"])

    # Assign seizure types
    seizures = [s for s, p in zip(SEIZURE_TYPES, SEIZURE_PROBS) if rng.random() < p]
    if not seizures and rng.random() < 0.5:
        seizures = ["GTCS"]

    return {
        "id":                   f"SCHAD-{SEED}-{i + 1:02d}",
        "phenotype":            phenotype,
        "onset_age_months":     onset_months,
        "glucose_nadir_mmol":   glucose_nadir,
        "insulin_mU_L":         insulin_mU,
        "c4oh_umol":            c4oh,
        "ammonia_umol":         ammonia,
        "bhba_mmol":            bhba,
        "protein_sensitive":    protein_sensitive,
        "diazoxide_responsive": diazoxide_responsive,
        "seizure_types":        seizures,
        "variant_allele1":      variant1,
        "variant_allele2":      variant2,
    }


PATIENTS = [_make_patient(i) for i in range(40)]


def get_overview():
    n = len(PATIENTS)
    neonatal_n   = sum(1 for p in PATIENTS if p["phenotype"] == "Classic Neonatal HI")
    infantile_n  = sum(1 for p in PATIENTS if p["phenotype"] == "Late Infantile HI")
    mild_n       = sum(1 for p in PATIENTS if p["phenotype"] == "Mild Attenuated")
    diazox_resp  = sum(1 for p in PATIENTS if p["diazoxide_responsive"])
    prot_sens    = sum(1 for p in PATIENTS if p["protein_sensitive"])
    severe_hypo  = sum(1 for p in PATIENTS if p["glucose_nadir_mmol"] < 1.5)
    gtcs_n       = sum(1 for p in PATIENTS if "GTCS" in p["seizure_types"])
    focal_n      = sum(1 for p in PATIENTS if "Focal" in p["seizure_types"])
    is_n         = sum(1 for p in PATIENTS if "IS/West" in p["seizure_types"])
    se_n         = sum(1 for p in PATIENTS if "Status epilepticus" in p["seizure_types"])
    myo_n        = sum(1 for p in PATIENTS if "Myoclonic" in p["seizure_types"])

    avg_glucose  = round(sum(p["glucose_nadir_mmol"] for p in PATIENTS) / n, 2)
    avg_insulin  = round(sum(p["insulin_mU_L"]       for p in PATIENTS) / n, 1)
    avg_c4oh     = round(sum(p["c4oh_umol"]          for p in PATIENTS) / n, 2)
    avg_ammonia  = round(sum(p["ammonia_umol"]        for p in PATIENTS) / n, 0)
    avg_bhba     = round(sum(p["bhba_mmol"]           for p in PATIENTS) / n, 2)

    return {
        "n_patients":     n,
        "seed":           SEED,
        "disease":        "SCHAD Deficiency (HADH / Short-Chain 3-Hydroxyacyl-CoA Dehydrogenase Deficiency / HHF4)",
        "gene":           "HADH",
        "locus":          "4q22.1",
        "omim_gene":      "*601609",
        "omim_disease":   "#231530 (SCHAD deficiency / Hyperinsulinaemic hypoglycaemia, familial, 4 / HHF4)",
        "prevalence":     "~1:50,000 hyperinsulinism cases worldwide",
        "inheritance":    "Autosomal Recessive (AR), biallelic LOF",
        "phenotype_distribution": {
            "Classic Neonatal HI (day 1-5; profound hypoglycaemia; diazoxide responsive ~80%)": neonatal_n,
            "Late Infantile HI (3-24 months; protein-triggered; episodic)": infantile_n,
            "Mild Attenuated (older children; fasting hypoglycaemia only)": mild_n,
        },
        "clinical_features": {
            "diazoxide_responsive":           diazox_resp,
            "protein_sensitive":              prot_sens,
            "severe_hypoglycaemia_lt1.5":     severe_hypo,
            "gtcs_seizures":                  gtcs_n,
            "focal_seizures":                 focal_n,
            "infantile_spasms_west":          is_n,
            "status_epilepticus":             se_n,
            "myoclonic_seizures":             myo_n,
        },
        "biomarkers": {
            "avg_glucose_nadir_mmol":   avg_glucose,
            "avg_insulin_mU_L":         avg_insulin,
            "avg_c4oh_umol":            avg_c4oh,
            "avg_ammonia_umol":         avg_ammonia,
            "avg_bhba_mmol":            avg_bhba,
        },
        "key_exam_facts": [
            "HADH UNIQUE MECHANISM: HADH normally inhibits GDH (GLUD1) by direct protein-protein interaction in beta-cell mitochondria",
            "HADH LOF → GDH uninhibited → excess glutamate oxidation → excess ATP → KATP channels CLOSE → EXCESS INSULIN",
            "HYPERINSULINISM = primary presentation — NOT a typical FAO energy-crisis disorder unlike MCAD/VLCAD/CPT2",
            "Ammonia NORMAL — KEY NEGATIVE vs GLUD1/HHS where ammonia 100-500 μmol/L (CRITICAL EXAM DISCRIMINATOR)",
            "C4-OH (3-hydroxybutyrylcarnitine) MILDLY ELEVATED — primary NBS marker but NONSPECIFIC",
            "Protein-sensitive hypoglycaemia PRESENT — leucine activates uninhibited GDH (same as GLUD1/HHS)",
            "BHBA mildly elevated PARADOXICALLY — SCHAD beta-oxidation step blocked for short-chain substrates",
            "Diazoxide Level A FIRST LINE — 70-80% response; opens KATP channels to suppress insulin",
            "FASTING ABSOLUTE CI — all hyperinsulinism disorders; worsens hypoglycaemia crisis",
            "VPA HIGH RISK — insulin secretagogue effect; avoid in all HI disorders",
            "p.His170Arg (35%) — disrupts HADH-GDH interaction interface; most common variant",
            "THREE PHENOTYPES: Classic Neonatal HI (50%) / Late Infantile HI (35%) / Mild Attenuated (15%)",
            "PROTEIN MODERATION (NOT strict restriction) Level B — reduces GDH activation via leucine pathway",
            "GLUD1 vs HADH: SAME protein-sensitive HI; DIFFERENT ammonia — GLUD1 HIGH, HADH NORMAL",
            "Sirolimus (mTOR inhibitor) Level C for refractory HI — inhibits mTOR pathway in beta cells",
        ],
    }


def get_breakdown():
    patients_out = []
    for p in PATIENTS:
        patients_out.append({
            "id":                   p["id"],
            "phenotype":            p["phenotype"],
            "onset_age_months":     p["onset_age_months"],
            "glucose_nadir_mmol":   p["glucose_nadir_mmol"],
            "insulin_mU_L":         p["insulin_mU_L"],
            "c4oh_umol":            p["c4oh_umol"],
            "ammonia_umol":         p["ammonia_umol"],
            "bhba_mmol":            p["bhba_mmol"],
            "protein_sensitive":    p["protein_sensitive"],
            "diazoxide_responsive": p["diazoxide_responsive"],
            "seizure_types":        p["seizure_types"],
            "variant_allele1":      p["variant_allele1"],
            "variant_allele2":      p["variant_allele2"],
        })

    # Group by phenotype
    phenotype_groups = {}
    for p in PATIENTS:
        grp = p["phenotype"]
        phenotype_groups.setdefault(grp, []).append(p)

    by_phenotype = {}
    for grp, pts in phenotype_groups.items():
        by_phenotype[grp] = {
            "n":                        len(pts),
            "avg_glucose_nadir":        round(sum(x["glucose_nadir_mmol"] for x in pts) / len(pts), 2),
            "avg_insulin":              round(sum(x["insulin_mU_L"]       for x in pts) / len(pts), 1),
            "avg_c4oh":                 round(sum(x["c4oh_umol"]          for x in pts) / len(pts), 2),
            "avg_ammonia":              round(sum(x["ammonia_umol"]        for x in pts) / len(pts), 0),
            "avg_bhba":                 round(sum(x["bhba_mmol"]           for x in pts) / len(pts), 2),
            "diazoxide_responsive_pct": round(sum(1 for x in pts if x["diazoxide_responsive"]) / len(pts) * 100, 1),
            "protein_sensitive_pct":    round(sum(1 for x in pts if x["protein_sensitive"])    / len(pts) * 100, 1),
            "severe_hypo_lt1.5_pct":    round(sum(1 for x in pts if x["glucose_nadir_mmol"] < 1.5) / len(pts) * 100, 1),
            "gtcs_pct":                 round(sum(1 for x in pts if "GTCS" in x["seizure_types"]) / len(pts) * 100, 1),
        }

    # Variant counts
    variant_counts = {}
    for p in PATIENTS:
        for allele in [p["variant_allele1"], p["variant_allele2"]]:
            variant_counts[allele] = variant_counts.get(allele, 0) + 1

    # Treatment summary
    n = len(PATIENTS)
    treatment_summary = {
        "diazoxide_first_line_n":           n,
        "continuous_glucose_monitor_n":     n,
        "frequent_feeds_cornstarch_n":      n,
        "protein_moderation_level_b_n":     sum(1 for p in PATIENTS if p["protein_sensitive"]),
        "octreotide_level_b_n":             sum(1 for p in PATIENTS if not p["diazoxide_responsive"]),
        "sirolimus_level_c_refractory_n":   sum(1 for p in PATIENTS if not p["diazoxide_responsive"]),
        "fasting_absolutely_avoided_n":     n,
        "vpa_avoided_high_risk_n":          n,
        "lev_for_seizures_n":               sum(1 for p in PATIENTS if p["seizure_types"]),
    }

    nbs_summary = {
        "pct_c4oh_elevated_ge0_35":         round(sum(1 for p in PATIENTS if p["c4oh_umol"] >= 0.35) / n * 100, 1),
        "pct_insulin_elevated_ge2":         round(sum(1 for p in PATIENTS if p["insulin_mU_L"] >= 2.0) / n * 100, 1),
        "pct_ammonia_normal_lt60":          round(sum(1 for p in PATIENTS if p["ammonia_umol"] < 60) / n * 100, 1),
        "pct_glucose_lt1_5_critical":       round(sum(1 for p in PATIENTS if p["glucose_nadir_mmol"] < 1.5) / n * 100, 1),
        "pct_diazoxide_responsive":         round(sum(1 for p in PATIENTS if p["diazoxide_responsive"]) / n * 100, 1),
        "pct_protein_sensitive":            round(sum(1 for p in PATIENTS if p["protein_sensitive"]) / n * 100, 1),
        "pct_any_seizure":                  round(sum(1 for p in PATIENTS if p["seizure_types"]) / n * 100, 1),
    }

    return {
        "patients":          patients_out,
        "by_phenotype":      by_phenotype,
        "variant_counts":    variant_counts,
        "treatment_summary": treatment_summary,
        "nbs_summary":       nbs_summary,
    }


def get_definitions():
    return {
        "disease_name": "SCHAD Deficiency (HADH / Short-Chain 3-Hydroxyacyl-CoA Dehydrogenase Deficiency / HHF4)",
        "gene":         "HADH (4q22.1)",
        "locus":        "4q22.1",
        "omim_gene":    "HADH *601609",
        "omim_disease": "#231530 (Hyperinsulinaemic hypoglycaemia, familial, 4 / SCHAD deficiency / HHF4)",
        "inheritance":  "Autosomal Recessive (AR) — biallelic LOF required",
        "terms": {
            "SCHAD_HADH": (
                "SCHAD = Short-Chain 3-Hydroxyacyl-CoA Dehydrogenase. Gene: HADH. "
                "302 amino acids; mitochondrial matrix; homodimeric; NAD-dependent. "
                "Located at 4q22.1. Catalyses step 3 of beta-oxidation for short-chain "
                "substrates (C4-C8): 3-Hydroxyacyl-CoA + NAD+ → 3-Ketoacyl-CoA + NADH. "
                "HADH is UNIQUE among FAO enzymes: it physically interacts with GDH (GLUD1) "
                "in pancreatic beta-cell mitochondria and inhibits GDH, linking FAO activity "
                "to insulin secretion control."
            ),
            "GDH_GLUD1": (
                "GDH = Glutamate Dehydrogenase (gene GLUD1). Mitochondrial enzyme in beta cells. "
                "Oxidises glutamate → α-ketoglutarate + NH4+ + NADH. This generates ATP in beta "
                "cells and stimulates insulin secretion. HADH normally INHIBITS GDH by direct "
                "protein-protein interaction. HADH LOF → GDH uninhibited → excess insulin."
            ),
            "KATP_channels": (
                "ATP-sensitive potassium channels (KATP) in pancreatic beta-cell plasma membrane. "
                "Subunits: SUR1 (ABCC8) + Kir6.2 (KCNJ11). When ATP is HIGH (e.g. from excess GDH "
                "activity), KATP channels CLOSE → membrane depolarises → Ca2+ influx via VGCC → "
                "insulin granule exocytosis. Diazoxide OPENS KATP (therapeutic in SCHAD HI)."
            ),
            "Diazoxide": (
                "KATP channel opener. First-line treatment for SCHAD/HHF4 (Level A; 70-80% response). "
                "Mechanism: opens SUR1-Kir6.2 (KATP) channels → K+ efflux → membrane "
                "hyperpolarisation → less Ca2+ influx → reduced insulin secretion. "
                "Dose: 5-15 mg/kg/day. Side effects: fluid retention, hypertrichosis. "
                "If diazoxide fails → octreotide (Level B) → sirolimus (Level C) → pancreatectomy."
            ),
            "HI_congenital_hyperinsulinism": (
                "Congenital hyperinsulinism (CHI) = genetically determined excess insulin secretion "
                "from pancreatic beta cells. Causes recurrent severe hypoglycaemia. "
                "SCHAD/HHF4 is one of ~12 genetic causes of CHI. Others include: ABCC8/KCNJ11 "
                "(KATP defects; most common CHI), GLUD1 (HHS; hyperammonaemia), GCK (glucokinase; "
                "activation), HNF4A/HNF1A (hepatocyte nuclear factor), HADH (SCHAD). "
                "Classified as diffuse or focal by 18F-DOPA PET scan."
            ),
            "C4_OH_3_hydroxybutyrylcarnitine": (
                "C4-OH = 3-hydroxybutyrylcarnitine. Primary NBS marker for SCHAD deficiency. "
                "Mildly elevated on newborn screening (typically 0.35-2.0 μmol/L vs normal <0.3). "
                "NONSPECIFIC: also elevated in IBD (isobutyryl-CoA dehydrogenase deficiency), "
                "methylmalonic acidaemia (C4-OH overlap). Does NOT distinguish SCHAD from other "
                "C4-OH disorders; clinical context + insulin measurement essential."
            ),
            "Protein_sensitive_HH": (
                "Protein-sensitive hypoglycaemia = hypoglycaemia triggered by protein ingestion. "
                "Mechanism in SCHAD: protein → amino acids including leucine → leucine activates "
                "GDH (allosteric activator) → excess glutamate oxidised → excess ATP → KATP close "
                "→ excess insulin. Because HADH cannot inhibit GDH, leucine-mediated GDH activation "
                "is amplified. This is SHARED with GLUD1/HHS but ABSENT in KATP-channel defects (ABCC8/KCNJ11)."
            ),
            "p_His170Arg": (
                "p.His170Arg (c.509A>G) — most common HADH variant (35%). His170 lies at the "
                "HADH-GDH protein-protein interaction interface. This arginine substitution disrupts "
                "the contact surface between HADH and GDH → GDH fully uninhibited → maximal HI. "
                "Neonatal onset; severe hypoglycaemia. Diazoxide responsive in ~80%."
            ),
            "GLUD1_HADH_interaction": (
                "The HADH-GDH (GLUD1) protein-protein interaction is the central mechanism of "
                "SCHAD deficiency. HADH binds the allosteric antenna region of the GDH hexamer "
                "inside beta-cell mitochondria. When HADH is engaged in short-chain FAO, it "
                "simultaneously inhibits GDH via this physical contact. HADH LOF → no binding → "
                "GDH constitutively active → excess insulin. This explains why SCHAD is a HI "
                "disorder rather than a pure FAO energy-crisis disorder."
            ),
            "mTOR_sirolimus": (
                "mTOR (mechanistic target of rapamycin) — kinase controlling beta-cell mass, "
                "proliferation, and insulin secretion. Sirolimus (rapamycin) = mTOR inhibitor. "
                "Level C therapy for refractory congenital HI (SCHAD or other causes). "
                "Reduces beta-cell proliferation and insulin output. "
                "Used when diazoxide + octreotide fail and partial pancreatectomy is high-risk. "
                "Significant immunosuppressive side-effect profile; specialist centre only."
            ),
            "Diffuse_vs_focal_HI": (
                "18F-DOPA PET scan distinguishes focal HI from diffuse HI. "
                "Focal HI: localised area of beta-cell hyperplasia (somatic LOH of KATP-region 11p15.5 + "
                "inherited ABCC8/KCNJ11 variant); surgical cure by focal resection. "
                "Diffuse HI: all beta cells affected; genetic (SCHAD/HHF4 = always diffuse); "
                "medical management or 95-98% pancreatectomy if refractory. "
                "SCHAD HI is ALWAYS diffuse (germline biallelic LOF)."
            ),
            "NBS_protocol_SCHAD": (
                "NBS (Newborn Bloodspot Screening) for SCHAD: tandem mass spectrometry detects "
                "elevated C4-OH (3-hydroxybutyrylcarnitine). C4-OH nonspecific — must be followed "
                "by clinical evaluation: plasma glucose, insulin during hypoglycaemia, ammonia, "
                "acylcarnitine profile, HADH gene sequencing. Insulin >2 mU/L with glucose <2.6 "
                "mmol/L = inappropriate hyperinsulinism = proceed to genetic CHI panel."
            ),
            "Octreotide": (
                "Octreotide = synthetic somatostatin analogue. Binds SSTR2/SSTR5 on beta cells → "
                "inhibits cAMP, Ca2+ influx, and insulin exocytosis. Level B treatment for SCHAD "
                "HI when diazoxide fails (~20% of cases). Subcutaneous injection 3-4x/day or "
                "long-acting release (Sandostatin LAR) monthly. Risk of necrotising enterocolitis "
                "in neonates — use with caution in first weeks of life."
            ),
            "VPA_high_risk_HI": (
                "Valproate (VPA/sodium valproate) is HIGH RISK in all HI disorders including SCHAD. "
                "Mechanism: VPA acts as an insulin secretagogue (stimulates beta-cell insulin "
                "release) — exactly the opposite of what is needed in HI. Additionally inhibits "
                "FAO and depletes carnitine. If seizures require anticonvulsant treatment in SCHAD, "
                "use LEV (levetiracetam) Level B rather than VPA."
            ),
            "Leucine_role_SCHAD": (
                "Leucine is a potent allosteric ACTIVATOR of GDH. In SCHAD deficiency, "
                "GDH is already uninhibited (no HADH present). Leucine further stimulates this "
                "already-maximal GDH → protein-containing meals → acute HI episodes. "
                "Protein moderation (NOT strict restriction) Level B — reduces leucine load "
                "without causing nutritional deficiency. Strict protein restriction is not needed "
                "because other amino acids are not toxic (unlike MSUD/PKU)."
            ),
            "Pancreatectomy_SCHAD": (
                "Partial pancreatectomy for SCHAD HI = 95-98% near-total pancreatectomy. "
                "Reserved for diffuse HI refractory to medical management (diazoxide + octreotide). "
                "Level B evidence. Complications: post-surgical diabetes mellitus (in up to 50-90% "
                "by adulthood after 98% resection), exocrine pancreatic insufficiency. "
                "All SCHAD is diffuse — no focal lesion to remove (unlike focal ABCC8 HI)."
            ),
            "BHBA_paradox": (
                "3-hydroxybutyrate (BHBA) is mildly elevated in SCHAD deficiency — paradoxical "
                "because HI typically suppresses ketogenesis. Mechanism: HADH enzyme normally "
                "converts 3-hydroxyacyl-CoA → 3-ketoacyl-CoA in short-chain beta-oxidation. "
                "HADH LOF → 3-hydroxy intermediates accumulate → 3-hydroxybutyryl-CoA → "
                "3-hydroxybutyrate released. Despite hyperinsulinism, the blocked short-chain "
                "FAO step generates a small paradoxical BHBA elevation."
            ),
        },
    }
