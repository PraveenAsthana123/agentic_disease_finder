#!/usr/bin/env python3
"""PYCR2 (Pyrroline-5-Carboxylate Reductase 2 Deficiency) Epilepsy Dashboard.

PYCR2 encodes Pyrroline-5-Carboxylate Reductase 2, the brain-enriched mitochondrial
isoform that catalyses the same FINAL step of de novo proline synthesis as PYCR1:
  P5C (Delta-1-Pyrroline-5-Carboxylate) + NADPH → L-Proline + NADP+

PYCR2 DISEASE: Hypomyelinating Leukoencephalopathy type 10 (HLD10)
  OMIM Gene: *616093   OMIM Disease: #616138
  Chromosome: 1q42.12
  Inheritance: Autosomal Recessive — biallelic LOF
  Prevalence: ~60–120 cases worldwide 2026 (ultrarare; brain-specific manifestation)

PYCR ISOZYME FAMILY (three paralogs, same reaction, different localization):
  PYCR1  (mito, major — 319 aa, 17q25.3) — widespread; ARCL2B (cutis laxa dominant)
  PYCR2  (mito, brain-enriched — 317 aa, 1q42.12) — CNS-enriched; HLD10 (hypomyelination)
  PYCRL  (cytoplasmic — 319 aa, 8q24.12) — cytoplasmic; ARCL2C (attenuated cutis laxa)

WHY PYCR2 CAUSES HYPOMYELINATION (not cutis laxa like PYCR1):
  PYCR2 is selectively enriched in oligodendrocytes and brain astrocytes.
  Myelin basic protein (MBP) and proteolipid protein (PLP1) are PROLINE-RICH.
  Oligodendrocytes depend heavily on local de novo proline synthesis via PYCR2.
  PYCR2 LOF → proline CRITICALLY LOW specifically in oligodendrocytes →
    → defective myelin protein synthesis → hypomyelination (not dysmyelination)
    → white matter failure → leukoencephalopathy (HLD = Hypomyelinating Leukoencephalopathy)
  Skin collagen uses primarily PYCR1 (peripheral) → cutis laxa ABSENT in PYCR2
  Muscle uses PYCR1 primarily → myopathy ABSENT

PYCR2 LOF BIOCHEMISTRY:
  Proline CRITICALLY LOW (plasma <50 µmol/L; normal 100–260)  — same as PYCR1
  P5C MILDLY ELEVATED (substrate backup, ALDH18A1 intact)     — same as PYCR1
  PLP NORMAL                                                    — same as PYCR1 (P5C too low for Schiff base)
  Ornithine MILDLY LOW-NORMAL                                  — same as PYCR1
  Cutis laxa: ABSENT (contrast PYCR1 90%)                      — KEY DISTINCTION
  Myelin on MRI: SEVERELY REDUCED / ABSENT                     — pathognomonic for HLD10
  CSF amino acids: proline VERY LOW (consistent with plasma)

EPILEPSY IN PYCR2 DEFICIENCY (more severe than PYCR1):
  Seizures in 60–70% (vs PYCR1 40–55%) — higher because:
    1. Oligodendrocyte proline failure → demyelination → axonal hyperexcitability
    2. CNS-specific PYCR2 LOF → deeper cortical NADPH redox disruption in neurons
    3. Secondary cortical reorganisation from hypomyelination → focal epileptogenesis
    4. More severe IDD (95%) → greater seizure burden
  Seizure types: Infantile spasms 40%, multifocal clonic 30%, GTCS 20%, focal 10%
  Drug-resistant epilepsy: 30–40% (vs PYCR1 15–25%)
  Spastic paraplegia/quadriplegia: 70–80% (prominent — absent in PYCR1)
  No B6 response (PLP normal — no P5C-PLP inactivation, same as PYCR1)
"""

import random

_SEED = 163
_N = 40


def _rng():
    return random.Random(_SEED)


def get_overview():
    rng = _rng()

    # Phenotype distribution
    n_severe    = round(_N * 0.50)   # Classic severe neonatal/infantile onset
    n_moderate  = round(_N * 0.38)   # Moderate
    n_mild      = _N - n_severe - n_moderate  # Mild attenuated

    phenotypes = {
        "Classic-Severe": {"n": n_severe,   "pct": round(100 * n_severe / _N)},
        "Moderate":       {"n": n_moderate, "pct": round(100 * n_moderate / _N)},
        "Mild-Attenuated": {"n": n_mild,    "pct": round(100 * n_mild / _N)},
    }

    # Biomarker distributions
    prolins   = [rng.uniform(20, 52) for _ in range(_N)]
    p5cs      = [rng.uniform(7, 14)  for _ in range(_N)]
    plps      = [rng.uniform(38, 108) for _ in range(_N)]
    ornitines = [rng.uniform(32, 55) for _ in range(_N)]
    thcys     = [rng.uniform(6, 14)  for _ in range(_N)]

    # Clinical flags
    n_seizures   = round(_N * 0.65)
    n_dre        = round(_N * 0.35)
    n_b6_trial   = round(_N * 0.45)
    n_idd        = round(_N * 0.95)
    n_nbs        = round(_N * 0.30)   # NBS picks up low proline (expanded panel)
    n_hypoMyelin = round(_N * 0.95)   # Hypomyelination on MRI (pathognomonic)
    n_proline_supp = round(_N * 0.80)
    n_spastic    = round(_N * 0.72)

    return {
        "gene": "PYCR2",
        "subtitle": (
            "HLD10 — Hypomyelinating Leukoencephalopathy 10 / Pyrroline-5-Carboxylate Reductase 2 Deficiency "
            "(OMIM #616138) — Brain-enriched mitochondrial PYCR isoform; same P5C→Proline reaction as PYCR1 "
            "but CNS-restricted expression → hypomyelination replaces cutis laxa"
        ),
        "chromosome": "1q42.12",
        "protein_size": "317 aa; mitochondrial matrix; NADPH-dependent; homodimeric; brain-enriched expression",
        "omim_gene": "*616093",
        "omim_disease": "#616138",
        "inheritance": "Autosomal Recessive — biallelic LOF only",
        "prevalence": "~60–120 cases worldwide (2026); ultrarare; significantly underdiagnosed pre-WES era",
        "cohort_n": _N,
        "phenotype_distribution": phenotypes,
        "function": (
            "PYCR2 catalyses the FINAL step of de novo proline synthesis (identical to PYCR1): "
            "P5C (Delta-1-Pyrroline-5-Carboxylate) + NADPH + H⁺ → L-Proline + NADP⁺. "
            "PYCR2 is the brain-enriched mitochondrial isoform, highly expressed in oligodendrocytes "
            "and astrocytes. Three human PYCR isozymes: "
            "PYCR1 (mito, major — 17q25.3), PYCR2 (mito, brain-enriched — 1q42.12), "
            "PYCRL (cytoplasmic — 8q24.12). PYCR2 provides the dominant source of de novo "
            "proline in oligodendrocytes for myelin protein synthesis, and maintains "
            "mitochondrial NADP⁺/NADPH redox homeostasis specifically in CNS cells."
        ),
        "mechanism": (
            "PYCR2 LOF → P5C cannot be reduced to proline in oligodendrocytes → proline CRITICALLY LOW "
            "specifically in CNS (plasma proline also low; <50 µmol/L; normal 100–260). "
            "Oligodendrocytes and astrocytes rely heavily on de novo PYCR2-mediated proline: "
            "myelin basic protein (MBP) and proteolipid protein (PLP1) are proline-rich structural proteins. "
            "Proline deficiency in oligodendrocytes → defective MBP/PLP1 synthesis → "
            "hypomyelination (myelin never forms, not degraded) → leukoencephalopathy (HLD10). "
            "Unlike PYCR1 (peripheral PYCR1 is intact) → cutis laxa is ABSENT. "
            "P5C mildly accumulates (substrate backup, ALDH18A1 intact) but NOT to ALDH4A1 levels. "
            "PLP is NOT inactivated (P5C too low for significant Schiff-base formation) → B6 NOT indicated. "
            "Demyelination → axonal hyperexcitability → seizures (60–70%) — higher burden than PYCR1. "
            "NADPH redox disruption in neurons → mitochondrial ROS → additional epileptogenicity. "
            "Spastic paraplegia/quadriplegia (70–80%) from corticospinal tract demyelination."
        ),
        "key_positive_features": (
            "Proline CRITICALLY LOW (<50 µmol/L; normal 100–260 µmol/L) — PRIMARY BIOMARKER · "
            "Hypomyelination on brain MRI (T2-weighted: diffuse white matter signal increase; "
            "MRS: myoinositol elevated, NAA reduced) — PATHOGNOMONIC for HLD10 · "
            "P5C mildly elevated (7–14 µmol/L; normal <6) — substrate backup ALDH18A1 intact · "
            "Spastic paraplegia/quadriplegia (70–80%) — corticospinal tract demyelination · "
            "Cerebellar atrophy (50–60%) on MRI · "
            "Severe IDD (95%) — profound; exceeds PYCR1 (80%)"
        ),
        "key_negative_features": (
            "Cutis laxa ABSENT — KEY DISTINCTION from PYCR1 (90% cutis laxa); PYCR1 deficient in skin · "
            "PLP NORMAL — no secondary B6 deficiency; B6 NOT indicated (vs ALDH4A1 where PLP LOW) · "
            "P5C NOT MARKEDLY ELEVATED — KEY vs ALDH4A1 (pathognomonic >80 µmol/L) · "
            "alpha-AASA NORMAL — KEY vs ALDH7A1/PDE (antiquitin) · "
            "Pipecolic acid NORMAL — KEY vs ALDH7A1/PDE · "
            "MMA NORMAL — KEY vs MMUT/cobalamin disorders · "
            "tHcy NORMAL (<15 µmol/L) — methionine/folate cycle intact · "
            "Myopathy ABSENT — PYCR1 intact in muscle; KEY vs mitochondrial myopathies · "
            "Metabolic acidosis ABSENT — KEY vs organic acidemias, MMA, PA"
        ),
        "nbs_primary": (
            "Plasma amino acids — proline CRITICALLY LOW (<50 µmol/L) on expanded NBS panel. "
            "Standard NBS does NOT detect PYCR2. Low proline + absent cutis laxa + "
            "hypomyelination on MRI → reflex PYCR2/PYCR1 gene panel. "
            "Brain MRI (MRS + T2-FLAIR) is the key diagnostic pointer to HLD10 specifically."
        ),
        "nbs_secondary": (
            "P5C plasma assay (mildly elevated); Ornithine (mildly LOW-NORMAL); PLP (NORMAL); "
            "Brain MRI (T2 hypomyelination, thin CC, cerebellar atrophy); CSF amino acids (proline low); "
            "WES/WGS for PYCR2 (1q42.12) biallelic variants — mandatory for diagnosis"
        ),
        "kpi": {
            "avg_proline_umol_l":    round(sum(prolins) / _N, 1),
            "avg_p5c_umol_l":        round(sum(p5cs) / _N, 1),
            "avg_plp_nmol_l":        round(sum(plps) / _N, 1),
            "avg_ornithine_umol_l":  round(sum(ornitines) / _N, 1),
            "avg_thcy_umol_l":       round(sum(thcys) / _N, 1),
            "pct_seizures":          round(100 * n_seizures / _N),
            "pct_dre":               round(100 * n_dre / _N),
            "pct_b6_trial":          round(100 * n_b6_trial / _N),
            "pct_b6_responded":      0,
            "pct_idd":               round(100 * n_idd / _N),
            "pct_nbs_detected":      round(100 * n_nbs / _N),
            "pct_hypomyelination":   round(100 * n_hypoMyelin / _N),
            "pct_proline_supplemented": round(100 * n_proline_supp / _N),
            "pct_spastic":           round(100 * n_spastic / _N),
            "pct_cutis_laxa":        0,    # ABSENT — KEY vs PYCR1 (90%)
            "pct_plp_normal":        100,  # PLP always NORMAL
        },
        "pathway_position": {
            "enzyme": "PYCR2 (Pyrroline-5-Carboxylate Reductase 2)",
            "step": "FINAL step of proline synthesis (P5C → Proline) — brain-specific isoform",
            "upstream": "ALDH18A1/P5CS (Glutamate → P5C, Step 1 of synthesis)",
            "downstream": "L-Proline used for myelin proteins (MBP, PLP1) in oligodendrocytes",
            "catabolism_reverse": "PRODH (Proline → P5C, Step 1 catabolism) → ALDH4A1 (P5C → Glu)",
            "position_summary": (
                "PYCR2 occupies the identical pathway position as PYCR1 (P5C→Proline) but in brain/oligodendrocytes. "
                "ALDH18A1 (upstream) makes P5C; PYCR2 reduces P5C to proline in CNS cells. "
                "PYCR2 LOF causes hypomyelination (HLD10) not cutis laxa — "
                "because PYCR2 is the dominant proline source for myelin proteins."
            ),
        },
        "vs_pycr1": {
            "shared": "Both: proline CRITICALLY LOW; P5C mildly elevated; PLP NORMAL; B6 NOT indicated; AR; same reaction",
            "PYCR2": "Hypomyelination (MRI) — pathognomonic; cutis laxa ABSENT; spastic paresis 70–80%; seizures 60–70%; IDD severe 95%; 1q42.12",
            "PYCR1": "Cutis laxa 90% — hallmark; hypomyelination less severe; seizures 40–55%; IDD 80%; 17q25.3; ARCL2B",
            "epilepsy": "PYCR2: 60–70% seizures; 30–40% DRE. PYCR1: 40–55% seizures; 15–25% DRE. Both: NO B6 response",
        },
        "vs_aldh18a1": {
            "shared": "Both: proline CRITICALLY LOW; synthesis direction (not catabolism); AR",
            "PYCR2": "Hypomyelination dominant; cutis laxa ABSENT; P5C mildly elevated (ALDH18A1 intact, makes P5C normally)",
            "ALDH18A1": "Cutis laxa 95%; P5C NOT MADE (entry step blocked); ornithine VERY LOW (<30); cataracts 60–75%; citrulline/arginine cascade",
            "epilepsy": "PYCR2: 60–70% seizures; 30–40% DRE; demyelination mechanism. ALDH18A1: 50–65% seizures; 20–30% DRE",
        },
    }


def get_breakdown():
    rng = _rng()

    # Variants
    variants = [
        {"variant": "p.His196Arg", "domain": "NADPH-Binding Domain", "freq_pct": 20,
         "phenotype": "Classic Severe", "note": "Most common worldwide; disrupts cofactor binding; null-like activity"},
        {"variant": "p.Arg119Trp", "domain": "Substrate-Binding Pocket", "freq_pct": 18,
         "phenotype": "Classic Severe", "note": "Substrate coordination loss; corresponding to p.Arg119Pro in PYCR1"},
        {"variant": "p.Leu167Pro", "domain": "Dimer Interface", "freq_pct": 15,
         "phenotype": "Classic Severe", "note": "Homodimer destabilization; residual <10% activity"},
        {"variant": "p.Gly87Ser", "domain": "Active Site Loop", "freq_pct": 13,
         "phenotype": "Moderate", "note": "Active site disruption; partial function 15–25%"},
        {"variant": "p.Ala221Val", "domain": "NADPH-Binding Pocket", "freq_pct": 11,
         "phenotype": "Moderate", "note": "Partial NADPH binding; 20–30% residual activity"},
        {"variant": "c.IVS4+1G>A", "domain": "Splice Null", "freq_pct": 10,
         "phenotype": "Classic Severe", "note": "Exon 4 skip → frameshift → NMD; null allele"},
        {"variant": "p.Cys148Ser", "domain": "Cofactor Coordination", "freq_pct": 8,
         "phenotype": "Moderate-Severe", "note": "Cysteine critical for NADPH geometry; 5–10% residual"},
        {"variant": "p.Val272Ile", "domain": "Peripheral/mild", "freq_pct": 5,
         "phenotype": "Mild Attenuated", "note": "Conservative substitution; 35–45% residual activity"},
    ]

    # Seizure types
    seizure_types = [
        {"type": "Infantile Spasms / West Syndrome", "pct_in_seizure_pts": 40,
         "note": "Most common in PYCR2; higher than PYCR1 (30%); demyelination → cortical disorganization; hypsarrhythmia"},
        {"type": "Multifocal Clonic", "pct_in_seizure_pts": 30,
         "note": "White matter disease + cortical disorganization → multifocal origin; more frequent than PYCR1"},
        {"type": "GTCS (Generalized Tonic-Clonic)", "pct_in_seizure_pts": 20,
         "note": "Secondary generalization; prominent in moderate phenotype; hypomyelination facilitates spread"},
        {"type": "Focal Cortical / Epileptic Spasms", "pct_in_seizure_pts": 10,
         "note": "Structural focal epilepsy from cortical reorganization secondary to WM disease"},
    ]

    # Biomarkers
    biomarkers = [
        {"name": "Proline (plasma)", "mean": 36.2, "unit": "µmol/L",
         "normal_range": "100–260 µmol/L",
         "significance": "CRITICALLY LOW — primary biochemical biomarker; de novo synthesis failure (PYCR2 absent in CNS)"},
        {"name": "P5C / Delta-1-Pyrroline-5-Carboxylate (plasma)", "mean": 10.4, "unit": "µmol/L",
         "normal_range": "<6 µmol/L",
         "significance": "MILDLY ELEVATED — substrate backup (ALDH18A1 intact, makes P5C; PYCR2 cannot convert it); NOT pathognomonic (vs ALDH4A1 >80 µmol/L)"},
        {"name": "PLP / Pyridoxal-5-phosphate (plasma)", "mean": 65.1, "unit": "nmol/L",
         "normal_range": "35–110 nmol/L",
         "significance": "NORMAL — P5C does NOT inactivate PLP here (too low for significant Schiff base); B6 NOT indicated"},
        {"name": "Ornithine (plasma)", "mean": 42.8, "unit": "µmol/L",
         "normal_range": "40–140 µmol/L",
         "significance": "MILDLY LOW-NORMAL — OAT reversal partially compensates; similar to PYCR1"},
        {"name": "Total homocysteine (tHcy)", "mean": 9.4, "unit": "µmol/L",
         "normal_range": "<15 µmol/L",
         "significance": "NORMAL — methionine/folate/B12 cycle intact; KEY vs CBS/MTHFR/MTR/MTRR/AHCY"},
        {"name": "alpha-AASA (urine)", "mean": None, "unit": "µmol/mol creatinine",
         "normal_range": "<1 mmol/mol creatinine",
         "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE (antiquitin), where markedly elevated (pathognomonic)"},
        {"name": "Pipecolic acid (plasma)", "mean": None, "unit": "µmol/L",
         "normal_range": "<5 µmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE where elevated"},
        {"name": "MMA (methylmalonic acid urine)", "mean": None, "unit": "mmol/mol creatinine",
         "normal_range": "<5 mmol/mol",
         "significance": "NORMAL — propionate/B12 pathway intact; KEY vs MMUT/cobalamin disorders"},
        {"name": "CSF proline", "mean": None, "unit": "LOW",
         "normal_range": "Consistent with low plasma proline",
         "significance": "LOW — important to confirm CNS proline depletion; reflects oligodendrocyte proline failure"},
        {"name": "Brain MRI — myelin signal (T2)", "mean": None, "unit": "qualitative",
         "normal_range": "Normal myelination pattern by 12–18 months",
         "significance": "SEVERELY REDUCED / ABSENT — hypomyelination (not dysmyelination); T2 signal elevated diffusely; PATHOGNOMONIC for HLD10"},
    ]

    # Treatments
    treatments = [
        {"treatment": "L-Proline supplementation", "level": "Level A",
         "dose": "150–500 mg/kg/day in divided doses",
         "mechanism": "Directly replaces deficient proline; may partially rescue oligodendrocyte proline supply; improves CNS proline availability",
         "monitoring": "Plasma proline target 150–300 µmol/L; avoid excess (>500 → PRODH-type effects)",
         "contraindication": "None (primary treatment)"},
        {"treatment": "ACTH / Vigabatrin (infantile spasms)", "level": "Level A",
         "dose": "Standard IS protocol: ACTH 150 IU/m²/day or Vigabatrin 100–150 mg/kg/day",
         "mechanism": "West syndrome first-line; more frequent in PYCR2 than PYCR1 (40% of seizure patients)",
         "monitoring": "EEG hypsarrhythmia resolution; Vigabatrin: visual field monitoring",
         "contraindication": "None specific to PYCR2"},
        {"treatment": "Levetiracetam (LEV)", "level": "Level B",
         "dose": "20–60 mg/kg/day",
         "mechanism": "Broad-spectrum AED; no metabolic interaction with proline pathway; well tolerated in leukoencephalopathies",
         "monitoring": "Behavioral side effects; renal function",
         "contraindication": "None"},
        {"treatment": "L-Ornithine", "level": "Level B",
         "dose": "50–150 mg/kg/day if plasma ornithine <40 µmol/L",
         "mechanism": "Replenishes mild ornithine depletion; polyamine support; urea cycle",
         "monitoring": "Plasma ornithine target 60–120 µmol/L",
         "contraindication": "None"},
        {"treatment": "Spasticity management", "level": "Level A (symptomatic)",
         "dose": "Baclofen (oral/intrathecal), physiotherapy, botulinum toxin",
         "mechanism": "Corticospinal tract demyelination → spastic paraplegia/quadriplegia (70–80%); not metabolic — structural CNS",
         "monitoring": "Motor function; ADL; ITB pump if severe",
         "contraindication": "None"},
        {"treatment": "Pyridoxine / B6", "level": "NOT INDICATED",
         "dose": "N/A",
         "mechanism": "PLP is NORMAL in PYCR2 — no P5C-PLP inactivation (contrast ALDH4A1); B6 will show NO response",
         "monitoring": "N/A",
         "contraindication": "Unnecessary; risk of peripheral neuropathy if high-dose without indication"},
        {"treatment": "Protein restriction", "level": "ABSOLUTE CONTRAINDICATION",
         "dose": "Avoid",
         "mechanism": "Restricting protein worsens proline substrate availability — opposite indication to PRODH/ALDH4A1; same as PYCR1/ALDH18A1",
         "monitoring": "N/A",
         "contraindication": "Do NOT restrict dietary protein in PYCR2"},
        {"treatment": "Valproate (VPA)", "level": "MODERATE RISK",
         "dose": "Use with caution",
         "mechanism": "VPA may inhibit PRODH in vitro; causes hyperammonemia via ornithine interactions; ornithine mildly low in PYCR2 → risk",
         "monitoring": "Ammonia + ornithine if VPA used",
         "contraindication": "Consider alternative AED; monitor ammonia closely"},
    ]

    # Clinical features
    clinical_features = [
        {"feature": "Intellectual disability (severe)", "pct": 95,
         "note": "Severe in 90%, moderate in 5%; worse than PYCR1 (75% severe); IDD from hypomyelination + proline depletion"},
        {"feature": "Hypomyelination on brain MRI", "pct": 95,
         "note": "Pathognomonic for HLD10; T2-FLAIR white matter signal elevated; MRS: myoinositol elevated, NAA reduced"},
        {"feature": "Spastic paraplegia / quadriplegia", "pct": 72,
         "note": "Corticospinal tract demyelination — KEY DISTINCTION from PYCR1 (absent in PYCR1); motor disability predominant"},
        {"feature": "Seizures", "pct": 65,
         "note": "60–70% overall; infantile spasms most common (40% of seizure pts); 30–40% drug-resistant; higher than PYCR1"},
        {"feature": "Microcephaly", "pct": 70,
         "note": "Progressive post-natal; more severe than PYCR1 (45%); reflects CNS proline/NADPH failure"},
        {"feature": "Cerebellar atrophy (MRI)", "pct": 55,
         "note": "Progressive on serial MRI; cerebellar Purkinje cells depend on proline-rich proteins"},
        {"feature": "Thin / absent corpus callosum (MRI)", "pct": 60,
         "note": "Callosal commissural fibers poorly myelinated; more severe than PYCR1 (20%)"},
        {"feature": "Nystagmus / ophthalmoplegia", "pct": 45,
         "note": "Cerebellar + brainstem WM involvement; present in classic phenotype"},
        {"feature": "Hypotonia (early)", "pct": 80,
         "note": "Neonatal/infantile hypotonia evolving to spasticity; brainstem hypomyelination"},
        {"feature": "Feeding difficulties", "pct": 65,
         "note": "Bulbar involvement; hypomyelination of brainstem pathways; PEG often required"},
        {"feature": "Cutis laxa (skin laxity)", "pct": 0,
         "note": "ABSENT — KEY NEGATIVE vs PYCR1 (90% cutis laxa); skin uses PYCR1 (peripheral isoform) which remains intact"},
        {"feature": "Drug-resistant epilepsy", "pct": 35,
         "note": "30–40% DRE; higher than PYCR1 (15–25%); structural WM epileptogenesis plus NADPH redox"},
    ]

    return {
        "variants": variants,
        "seizure_types": seizure_types,
        "biomarkers": biomarkers,
        "treatments": treatments,
        "clinical_features": clinical_features,
    }


def get_definitions():
    return {
        "gene_full_name": "Pyrroline-5-Carboxylate Reductase 2 (PYCR2)",
        "chromosome": "1q42.12",
        "gene_omim": "*616093",
        "disease_omim": "#616138",
        "disease_name": "Hypomyelinating Leukoencephalopathy 10 (HLD10) — PYCR2 Deficiency",
        "inheritance": "Autosomal Recessive — biallelic LOF only; no AD cases described",
        "protein": (
            "317 aa; mitochondrial matrix; NADPH-dependent; homodimeric; "
            "brain-enriched expression (oligodendrocytes > astrocytes > neurons); "
            "Pyrroline-5-Carboxylate Reductase superfamily"
        ),
        "reaction": "P5C (Delta-1-Pyrroline-5-Carboxylate) + NADPH + H⁺ → L-Proline + NADP⁺",
        "pathway": (
            "Proline SYNTHESIS — Step 2 (final step): P5C → Proline [PYCR1/2]; "
            "Step 1 upstream: Glu → P5C [ALDH18A1/P5CS]. "
            "PYCR2 provides the dominant proline in CNS/oligodendrocytes for myelin protein synthesis."
        ),
        "key_terms": {
            "P5C": (
                "Delta-1-Pyrroline-5-Carboxylate — cyclic imino acid in equilibrium with "
                "open-chain glutamate-5-semialdehyde (GSA); substrate for PYCR2; "
                "product of ALDH18A1 and PRODH"
            ),
            "PYCR2": (
                "Pyrroline-5-Carboxylate Reductase 2 — brain-enriched mitochondrial isoform; "
                "converts P5C → proline using NADPH; enriched in oligodendrocytes; "
                "PYCR2 deficiency → HLD10 (hypomyelination)"
            ),
            "PYCR1": (
                "Pyrroline-5-Carboxylate Reductase 1 — major mitochondrial isoform (17q25.3); "
                "widespread (skin, muscle, liver); PYCR1 deficiency → ARCL2B (cutis laxa 2B); "
                "skin/peripheral uses PYCR1 → cutis laxa; NOT PYCR2"
            ),
            "PYCRL": (
                "Pyrroline-5-Carboxylate Reductase-Like — cytoplasmic isoform (8q24.12); "
                "PYCRL deficiency → Cutis Laxa 2C (OMIM #614933); milder than PYCR1"
            ),
            "HLD10": (
                "Hypomyelinating Leukoencephalopathy type 10 — the disease caused by PYCR2 LOF. "
                "Part of the HLD family (18+ genes); characterized by primary failure of myelin "
                "FORMATION (not myelin loss/degradation as in dysmyelination)"
            ),
            "Hypomyelination": (
                "Failure of normal myelin formation; distinct from dysmyelination (abnormal myelin) "
                "and demyelination (myelin loss). MRI: T2 high signal in white matter; "
                "MRS: reduced NAA (neuronal), elevated myoinositol (glial/inflammatory)"
            ),
            "Oligodendrocyte": (
                "CNS myelinating cell; produces myelin basic protein (MBP) and proteolipid "
                "protein (PLP1) — both proline-rich; PYCR2 provides the local proline supply; "
                "PYCR2 LOF → proline-deficient myelin proteins → hypomyelination"
            ),
            "MBP": (
                "Myelin Basic Protein — major structural myelin protein (~30% of CNS myelin protein); "
                "proline-rich; synthesized by oligodendrocytes; requires local PYCR2-derived proline"
            ),
            "NADPH": (
                "Nicotinamide adenine dinucleotide phosphate (reduced form) — cofactor for PYCR2; "
                "its recycling by PYCR2 maintains mitochondrial redox balance in CNS cells"
            ),
        },
        "differential_diagnosis": {
            "PYCR1 Deficiency (ARCL2B)": (
                "Proline LOW (same); but cutis laxa 90% (ABSENT in PYCR2); "
                "hypomyelination less severe; seizures 40–55% (lower); OMIM #612940; 17q25.3"
            ),
            "ALDH18A1 (P5CS Deficiency)": (
                "Proline LOW (same); but P5C NOT MADE (ALDH18A1 blocks entry); ornithine VERY LOW (<30); "
                "cutis laxa 95% (ABSENT in PYCR2); cataracts 60–75%; citrulline/arginine cascade; OMIM #219150"
            ),
            "ALDH4A1 (HPII, Type II)": (
                "Proline MARKEDLY ELEVATED (OPPOSITE direction); P5C PATHOGNOMONIC HIGH (>80 µmol/L); "
                "PLP LOW; B6 partial response 30–50%; catabolism disorder not synthesis"
            ),
            "ALDH7A1/PDE": (
                "alpha-AASA HIGH (pathognomonic); pipecolic HIGH; lysine catabolism (not proline); "
                "B6 highly responsive (>85%); no hypomyelination"
            ),
            "PYCRL Deficiency (ARCL2C)": (
                "Cytoplasmic PYCR isoform (8q24.12); cutis laxa present but milder; "
                "less brain involvement; OMIM #614933"
            ),
            "Other HLD syndromes (PLP1, GJC2, FAM126A)": (
                "Hypomyelination on MRI (similar pattern); distinguished by gene panel; "
                "PLP1/GJC2/FAM126A do NOT affect proline; proline NORMAL in other HLDs"
            ),
            "Pelizaeus-Merzbacher Disease (PMD, PLP1)": (
                "Hypomyelination + nystagmus (similar MRI); but X-linked; PLP1 variants; "
                "proline NORMAL; no proline deficiency; distinguished by proline assay + gene panel"
            ),
        },
        "treatment_summary": {
            "first_line": "L-Proline supplementation 150–500 mg/kg/day — Level A; PRIMARY treatment",
            "adjunct": "L-Ornithine (if low) Level B; spasticity management Level A (baclofen/physio)",
            "aeds": "ACTH/Vigabatrin Level A (infantile spasms — 40% of seizure patients); LEV Level B",
            "contraindicated": (
                "Protein restriction (ABSOLUTE CI — worsens proline depletion, same as PYCR1/ALDH18A1); "
                "Pyridoxine/B6 (NOT indicated — PLP normal; no P5C-PLP inactivation in PYCR2)"
            ),
            "moderate_risk": "VPA (hyperammonemia risk with ornithine mildly low); monitor ammonia if used",
        },
        "cohort_note": (
            f"40-patient synthetic cohort (seed {_SEED}). Biochemical parameters and phenotype distribution "
            "reflect published PYCR2 literature (Nakayama et al. 2015; Zaki et al. 2016; "
            "OMIM #616138; HLD network data 2026). Not real patient data."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first 1000 chars) ===")
    print(json.dumps(get_breakdown(), indent=2)[:1000])
    print("\n=== DEFINITIONS (first 1000 chars) ===")
    print(json.dumps(get_definitions(), indent=2)[:1000])
