#!/usr/bin/env python3
"""TMEM70 Complex V Deficiency / 3-MGA-uria Type VI Dashboard.

Mitochondrial Complex V Deficiency Nuclear Type 2 (OMIM 614052) =
Severe neonatal lactic acidosis + hyperammonemia + DCM + 3-MGA-uria

TMEM70 (Transmembrane Protein 70, 260 aa, 8q11.23) is a nuclear-encoded inner
mitochondrial membrane protein required for the biogenesis of ATP synthase (Complex V /
F1-Fo ATPase). LOF → Complex V assembly failure → severe OXPHOS ATP deficit →
lactic acidosis (pyruvate → lactate, no OXPHOS backup) + hyperammonemia (urea cycle
is ATP-dependent) + DCM (cardiomyocytes most OXPHOS-dependent) + 3-MGA-uria overflow.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. 3-MGA-uria Type VI — TMEM70-specific; lactic acidosis + hyperammonemia PATHOGNOMONIC triad
  2. Neonatal onset: pH <7.1 + lactate >10 mmol/L + NH3 50–500 µmol/L mimics urea cycle disorder
  3. Czech/Slovak Roma founder: c.317-2A>G (IVS2 splice donor) ~50% of worldwide alleles
  4. DCM (80-90%) — onset first week to 3 months; both Complex V AND secondary mitophagy failure
  5. Hyperammonemia: ATP-dependent step in urea cycle (carbamoyl phosphate synthetase I) fails
  6. VPA: ABSOLUTE CI (mito disease + already-elevated ammonia → fatal hyperammonemic crisis)
  7. Normal acylcarnitine (no C4-DC) — KEY DDx from TAZ/Barth (C4-DC pathognomonic there)
  8. Emergency neonatal: IV glucose + bicarbonate + ammonia scavengers; avoid fasting
  9. ACE-I + BB for DCM (Level A) when haemodynamically stable
 10. Autosomal recessive: both sexes affected (unlike TAZ which is X-linked males only)
 11. 8q11.23 — TMEM70 gene; ~100+ patients worldwide (ultra-rare)
 12. KD (ketogenic diet): CONTROVERSIAL / potentially HARMFUL in Complex V deficiency

TMEM70 BIOLOGY:
TMEM70 (260 amino acids, 8q11.23) encodes a transmembrane protein of the inner mitochondrial
membrane (IMM). TMEM70 is an assembly factor specifically required for the biogenesis of
the Fo-subunit c-ring (rotor ring) of ATP synthase (Complex V / F1-Fo ATPase).

ATP synthase structure and TMEM70 function:
  F1 head (matrix): α3β3γδε — catalytic subunits; converts ADP + Pi → ATP
  Fo ring (IMM): c-ring (10 c-subunits) + a-subunit — proton translocation channel
  TMEM70 role: required for c-ring assembly and F1–Fo coupling; binds early c-ring intermediates
  TMEM70 LOF → c-ring assembly fails → F1 head remains unassembled → Complex V severely deficient

OXPHOS cascade failure:
  Complex V LOF → ADP not phosphorylated → ATP/ADP ratio ↓↓ →
    NADH + FADH₂ cannot be reoxidised (electron chain backs up) →
    PDH inhibited → pyruvate accumulates → shunted to lactate → LACTIC ACIDOSIS
    Urea cycle (CPSI) ATP-dependent → fails → HYPERAMMONEMIA
    Cardiomyocyte ATP deficit → OXPHOS-dependent cells most vulnerable → DCM
    3-MGA overflow pathway (shared with other mito diseases) → 3-MGA-uria

Roma founder variant c.317-2A>G:
  Intronic splice donor in intron 2; destroys IVS2 splice donor consensus (GT→TT)
  Result: exon 3 skipping → premature stop codon → truncated TMEM70 → complete null (NMD)
  Homozygous in ~45% of Czech/Slovak Roma patients with TMEM70 deficiency
  Compound heterozygous with another TMEM70 allele in ~5%
  Other ethnic backgrounds: compound heterozygous for private missense + splice/null variants

PROTEIN STRUCTURE (260 aa, 8q11.23):
  N-terminal mitochondrial targeting sequence (aa 1-50): IMM localisation
  Transmembrane domain 1 (aa 51-100): IMM anchor, first helix
  Matrix loop (aa 101-180): c-ring binding interface; most pathogenic missense cluster
  Transmembrane domain 2 (aa 181-220): second IMM helix
  C-terminal matrix domain (aa 221-260): F1 coupling and DAPIT/6.8PL regulatory interactions

PATHOGENIC VARIANT DISTRIBUTION (biallelic, n=40, seed-543):
  Roma founder c.317-2A>G (IVS2 splice donor, exon 3 skip): ~50% of alleles
    Homozygous (both alleles): ~45% of patients
    Compound het with another allele: ~5% of patients
  Missense in matrix loop (aa 101-180): ~25% of alleles
    p.Gln175Pro, p.Ala137Val and other private variants; partial residual function possible
  Splice site — other intronic: ~10% of alleles
    IVS3, IVS5 splice variants; variable exon-skipping severity
  Frameshift / nonsense: ~10% of alleles
    Premature stop; complete null; NMD
  Large deletion (aCGH): ~5% of alleles
    Rare; MLPA/array CGH required for detection

CLINICAL PHENOTYPE — TMEM70 COMPLEX V DEFICIENCY:
  NEONATAL CRISIS (100%) — CARDINAL PRESENTATION:
    Severe lactic acidosis: pH <7.1; lactate >10 mmol/L (often 15–25 mmol/L) at hours of life
    Hyperammonemia: NH3 50–500 µmol/L (mean ~150); mimics urea cycle disorders (OTC, CPS1)
    Hypotonia: severe; global; onset at birth
    Respiratory distress: multifactorial (acidosis + PAH + DCM)
    Prognosis: ~35-40% neonatal mortality without ICU + ammonia management
  DILATED CARDIOMYOPATHY (80-90%) — SECOND CARDINAL:
    Onset: first week to 3 months of life
    Systolic dysfunction (EF <50%); LV dilatation; occasionally biventricular
    ACE-I + BB: Level A when haemodynamically stable; diuretics prn
    Transplant required: ~15% (lower than Barth; DCM can partially recover with metabolic stabilisation)
    KEY DDx: DCM also in TAZ/Barth — but TAZ has C4-DC + neutropenia + X-linked; TMEM70 is AR, no C4-DC
  3-METHYLGLUTACONIC ACIDURIA (100%) — TYPE VI:
    Urinary 3-MGA: 50–300 mmol/mol creatinine (higher than Barth; secondary overflow pathway)
    Acylcarnitine: NORMAL — no C4-DC (critical DDx from Barth/TAZ where C4-DC is pathognomonic)
    Lactate/pyruvate elevated: L/P ratio usually >20 (mitochondrial block, not cytoplasmic)
  HYPERAMMONEMIA (90%) — KEY TO 3-MGA-URIA TYPE VI:
    NH3 elevated in neonatal period; reflects urea cycle ATP failure
    Can be severe enough to require ammonia scavengers (sodium benzoate, sodium phenylbutyrate)
    KEY: 3-MGA + hyperammonemia = TMEM70 until proven otherwise (no other 3-MGA disease has this)
  PULMONARY ARTERIAL HYPERTENSION (40-50%):
    Neonatal PAH: secondary to hypoxia + acidosis + vasoconstriction
    Responds to iNO (inhaled nitric oxide) + sildenafil in acute phase
    May require ECMO in severe cases
  HYPOTONIA (100%):
    Severe; global; proximal > distal; persistent in survivors
    PT/OT from neonatal period; supported feeding (NG/PEG in 60%)
  INTELLECTUAL DISABILITY (60-70%):
    Mild to moderate in survivors; caused by neonatal hypoxic-ischemic insult + chronic ATP deficit
    Cognition is NOT normal (DDx from TAZ/Barth where cognition 100% preserved)
  OPTIC ATROPHY: ABSENT (DDx from MECR/OPA3 where optic atrophy prominent)
  SENSORINEURAL HEARING LOSS: ABSENT (DDx from SERAC1 where SNHL 100%)
  NEUTROPENIA: ABSENT (DDx from TAZ/Barth where neutropenia 95%)

TREATMENT & PHARMACOGENOMICS:
  EMERGENCY NEONATAL PROTOCOL:
    Step 1 — Stop catabolism: IV 10% dextrose (GIR 8-10 mg/kg/min); NPO; avoid any fasting
    Step 2 — Correct acidosis: sodium bicarbonate IV bolus; continuous bicarbonate infusion
    Step 3 — Ammonia: if NH3 >200 µmol/L → sodium benzoate (250 mg/kg IV load) +
      sodium phenylbutyrate (250 mg/kg IV load) → nitrogen scavenging; urgent metabolic consult
    Step 4 — DCM support: dopamine/dobutamine prn; ACE-I when stable; diuretics
    Step 5 — Respiratory: intubation + CPAP; iNO for PAH; ECMO if refractory PAH
  ACE Inhibitor + Beta-Blocker: DCM — Level A (when haemodynamically stable)
  L-Carnitine: secondary depletion — Level C; supplement C0 to 30-60 µmol/L
  Riboflavin (B2): empirical — Level D; sometimes tried as general mito cofactor; no Direct Complex V effect
  CoQ10: empirical — Level D; no controlled evidence; some centres use
  VPA (Valproate): ABSOLUTE CONTRAINDICATION — Level A prohibition
    VPA inhibits Complex I AND sequesters CoA → fatal in already-ATP-depleted state
    CRITICAL: VPA increases ammonia independently (BCAA/glutamine interference) → fatal
    hyperammonemic crisis in TMEM70 (baseline ammonia already elevated)
    NEVER use VPA in any TMEM70 patient under ANY circumstances
  LEV (Levetiracetam): Seizures — PREFERRED — Level B
    Renal excretion; no mito toxicity; no ammonia effect; safe in Complex V deficiency
    Monitor renal function (impaired in neonatal acidosis phase)
  Propofol: ABSOLUTE AVOID (PRIS) — mitochondrial disease; prefer ketamine/inhalational
  Ketogenic Diet: CONTRAINDICATED / AVOID
    Fat-based metabolism requires OXPHOS to generate ATP from ketone bodies via NADH/FADH2
    Complex V deficiency prevents ATP synthesis from FADH2 → KD worsens energy deficit
    In contrast to PDH/PC deficiency (where KD bypasses the block), Complex V = CANNOT use KD
  Sodium Benzoate + Sodium Phenylbutyrate: ammonia crisis — Level A (acute)
    Alternative nitrogen pathway; converts glycine → hippurate (benzoate) or
    phenylacetylglutamine (phenylbutyrate) → excreted in urine → reduces ammonia load
  Sildenafil: PAH — Level B (acute and maintenance if PAH persists)
  PHT/CBZ/OXC: AVOID — Na-channel AEDs worsen cardiac conduction; avoid in DCM setting

MONITORING SCHEDULE:
  Neonatal ICU: lactate/pH 2-hourly; ammonia 4-hourly; glucose 1-hourly; echo daily
  Metabolic: lactate + ammonia weekly (first month); monthly (first year)
  Cardiology: echo + BNP monthly (first 6 months); quarterly once stable
  Neurodevelopment: formal assessment 6-monthly; SLT + PT/OT from discharge
  Acylcarnitine + 3-MGA: 3-6 monthly; confirms disease activity
  TMEM70 enzyme (Complex V activity): fibroblasts at diagnosis; not repeated routinely
"""

import random
from datetime import date

SEED = 543  # 40-patient cohort seed

# ── overview ──────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """TMEM70 Complex V Deficiency / 3-MGA-uria Type VI — overview for /api/tmem70/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Mitochondrial Complex V Deficiency, Nuclear Type 2 / 3-MGA-uria Type VI / TMEM70 Deficiency",
        "gene": "TMEM70; Transmembrane Protein 70; ATP synthase assembly factor; 260 aa; IMM c-ring biogenesis; 8q11.23",
        "chromosome": "8q11.23",
        "omim_gene": "612418",
        "omim_disease": "614052",
        "inheritance": "Autosomal recessive; biallelic TMEM70 mutations; both sexes equally affected",
        "prevalence": "~1:500,000–1:1,000,000; ~100+ patients described worldwide (2026); Czech/Slovak Roma enriched",
        "first_described": "Cizkova et al. 2008 (Nat Genet) — identified TMEM70 in Czech Roma with neonatal Complex V deficiency",
        "protein": "TMEM70 (260 aa) — IMM transmembrane assembly factor; required for Fo c-ring (rotor) biogenesis; TMEM70 LOF → Complex V assembly failure → severe ATP deficit",
        "category": "3-MGA-uria Type VI / Complex V (ATP synthase) deficiency / Neonatal mitochondrial crisis",
        "kpis": {
            "neonatal_lactic_acidosis_pct": 100,
            "hyperammonemia_pct": 90,
            "dcm_pct": 85,
            "hypotonia_pct": 100,
            "id_pct": 65,
            "pah_pct": 45,
            "neonatal_mortality_untreated_pct": 38,
            "vpa_ci": "ABSOLUTE CI",
            "c4dc_elevated": "NO (normal acylcarnitine — DDx TAZ/Barth)",
        },
        "clinical_highlights": [
            "Neonatal crisis 100%: severe lactic acidosis (pH <7.1, lactate >10 mmol/L) + hyperammonemia (NH3 50–500 µmol/L) at hours of life",
            "3-MGA + hyperammonemia = TMEM70 PATHOGNOMONIC triad — no other 3-MGA disease causes hyperammonemia",
            "Czech/Slovak Roma founder c.317-2A>G (IVS2 splice donor) ~50% worldwide alleles; homozygous in ~45% of Roma patients",
            "DCM (80-90%): onset first week to 3 months; BOTH complex V failure AND secondary mitophagy dysfunction",
            "Normal acylcarnitine: NO C4-DC (KEY DDx from TAZ/Barth — C4-DC pathognomonic for Barth, absent here)",
            "VPA: ABSOLUTE CONTRAINDICATION — worsens ammonia AND Complex V failure → fatal hyperammonemic crisis",
            "Autosomal recessive: both sexes affected equally (vs TAZ/Barth which is X-linked males only)",
            "KD (ketogenic diet): CONTRAINDICATED — fat-based ATP via Complex V makes KD harmful (worsens the block)",
            "Intellectual disability 60-70% survivors: neonatal insult + chronic ATP deficit (differs from TAZ — cognition 100% normal)",
            "Riboflavin/CoQ10: empirical only; no controlled evidence; Low-dose thiamine sometimes tried",
        ],
        "contraindications": [
            {
                "drug": "VPA (Valproate / Valproic Acid)",
                "level": "ABSOLUTE CI",
                "reason": "Dual mechanism: (1) Complex I inhibition + CoA sequestration → fatal in Complex V-deficient state; (2) VPA independently raises ammonia (BCAA/glutamine interference) → fatal hyperammonemic crisis on top of baseline elevated NH3. NEVER use in TMEM70.",
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "CONTRAINDICATED",
                "reason": "Complex V deficiency = cannot synthesise ATP from ketone-body-derived NADH/FADH2. KD worsens energy deficit. Unlike PDH deficiency (where KD bypasses) — Complex V is the terminal ATP factory; fat-based fuel still requires it. Do NOT use KD in TMEM70.",
            },
            {
                "drug": "Propofol",
                "level": "ABSOLUTE AVOID (PRIS)",
                "reason": "Propofol inhibits Complex I → PRIS (propofol infusion syndrome) risk in any mitochondrial disease. Prefer ketamine + inhalational agents for anaesthesia. Alert anaesthesia team to TMEM70 diagnosis pre-procedure.",
            },
            {
                "drug": "PHT / CBZ / OXC (Na-channel AEDs)",
                "level": "AVOID in DCM",
                "reason": "Na-channel blockade worsens cardiac conduction; additive to DCM instability. Joint cardiology-neurology decision required if unavoidable; continuous cardiac monitoring mandatory.",
            },
            {
                "drug": "Fasting / Prolonged NPO",
                "level": "AVOID — HIGH RISK",
                "reason": "Any catabolic state triggers lactic acidosis in Complex V deficiency (no OXPHOS backup). Maintain IV dextrose GIR ≥6 mg/kg/min during illness, procedures, or NPO periods. Emergency sick-day protocol mandatory for families.",
            },
        ],
        "thresholds": [
            {"marker": "Serum lactate (neonatal)", "cutoff": ">10 mmol/L", "interpretation": "Severe lactic acidosis — start IV bicarbonate, GIR 8-10 mg/kg/min, metabolic emergency protocol"},
            {"marker": "Blood ammonia (NH3)", "cutoff": ">200 µmol/L", "interpretation": "Severe hyperammonemia — start sodium benzoate + phenylbutyrate IV; consider dialysis if >500 µmol/L"},
            {"marker": "pH (arterial/venous)", "cutoff": "<7.1", "interpretation": "Severe metabolic acidosis — sodium bicarbonate bolus + infusion; urgent ICU care"},
            {"marker": "Lactate:pyruvate ratio", "cutoff": ">20", "interpretation": "Mitochondrial block confirmed (vs cytoplasmic); L/P >20 distinguishes OXPHOS defects from other causes of lactic acidosis"},
            {"marker": "LV ejection fraction (echo)", "cutoff": "EF <40%", "interpretation": "DCM significant — start ACE-I + BB (when stable); lower threshold for transplant listing than Barth"},
            {"marker": "Blood glucose", "cutoff": "<3.5 mmol/L", "interpretation": "Hypoglycaemia — IV glucose bolus (2 ml/kg 10% dextrose); maintain GIR ≥6 mg/kg/min continuously"},
        ],
        "ddx_table": [
            {
                "disease": "TAZ — Barth Syndrome (3-MGA Type II)",
                "shared": "DCM, 3-MGA-uria",
                "distinguishing": "TAZ: C4-DC elevated (pathognomonic) + neutropenia 95% + X-linked (males only) + NO hyperammonemia + normal cognition. TMEM70: C4-DC absent + hyperammonemia + AR + ID 65%.",
            },
            {
                "disease": "DNAJC19 — DCMA (3-MGA Type III variant)",
                "shared": "DCM, 3-MGA-uria",
                "distinguishing": "DNAJC19: cerebellar ataxia 95% + male genital anomalies + NO hyperammonemia + normal acylcarnitine + Hutterite founder. TMEM70: hyperammonemia + lactic acidosis + no ataxia initially.",
            },
            {
                "disease": "Urea Cycle Disorders (OTC, CPS1, ASS1)",
                "shared": "Neonatal hyperammonemia",
                "distinguishing": "UCD: NH3 often >500, NO lactic acidosis (or mild), normal lactate:pyruvate, normal organic acids (no 3-MGA), plasma amino acid pattern (citrulline low in OTC/CPS1). TMEM70: 3-MGA + high L/P ratio + lactic acidosis.",
            },
            {
                "disease": "AUH — 3-MGA Type I",
                "shared": "3-MGA-uria",
                "distinguishing": "AUH: primary 3-MGA enzyme defect; NO DCM; NO hyperammonemia; NO lactic acidosis; leucine-restriction responds; C5-OH borderline; normal lactate.",
            },
            {
                "disease": "SERAC1 — MEGDEL (3-MGA Type V)",
                "shared": "3-MGA-uria",
                "distinguishing": "SERAC1: SNHL 100% + Leigh-like MRI + neonatal liver; NO DCM; NO hyperammonemia; normal Complex V; different mechanism (phosphatidylglycerol remodeling).",
            },
        ],
    }


# ── breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    """TMEM70 Complex V Deficiency — patient breakdown for /api/tmem70/breakdown."""
    rng = random.Random(SEED)
    n = 40

    # Phenotype groups
    phenotype_groups = [
        ("Classic severe neonatal crisis (lactic acidosis + hyperammonemia + DCM)", 26),
        ("Severe neonatal + PAH requiring iNO/ECMO", 8),
        ("Attenuated (late-presenting, moderate lactic acidosis, preserved EF)", 6),
    ]
    assert sum(c for _, c in phenotype_groups) == n

    # Variant distribution (biallelic, AR)
    variant_dist = [
        {"variant": "Roma founder c.317-2A>G (IVS2 splice donor — exon 3 skipping, complete null)", "n_alleles": 40, "pct": 50, "effect": "Complete null (NMD); homozygous in 45% of patients; hallmark of Czech/Slovak Roma population"},
        {"variant": "Missense: matrix loop (aa 101-180) — p.Gln175Pro, p.Ala137Val, others", "n_alleles": 20, "pct": 25, "effect": "Partial residual function possible in some missense; slightly attenuated phenotype in compound hets"},
        {"variant": "Other intronic splice variants (IVS3, IVS5)", "n_alleles": 8, "pct": 10, "effect": "Variable exon skipping; partial or complete LOF; severity depends on residual Complex V activity"},
        {"variant": "Frameshift / nonsense variants", "n_alleles": 8, "pct": 10, "effect": "Complete null; premature stop → NMD; severe neonatal phenotype"},
        {"variant": "Large exon deletion (aCGH/MLPA)", "n_alleles": 4, "pct": 5, "effect": "Rare; complete null; requires array CGH or MLPA for detection; no founder"},
    ]

    # Treatment distribution
    treatment_dist = [
        {"treatment": "IV Glucose (dextrose 10% GIR 8-10 mg/kg/min)", "n": 40, "pct": 100, "indication": "Neonatal lactic acidosis — MANDATORY emergency; prevents catabolism; Level A"},
        {"treatment": "IV Sodium Bicarbonate", "n": 40, "pct": 100, "indication": "Severe metabolic acidosis — Level A; correct pH; continuous infusion if needed"},
        {"treatment": "Ammonia scavengers (sodium benzoate + phenylbutyrate)", "n": 36, "pct": 90, "indication": "Hyperammonemia — Level A; if NH3 >200 µmol/L; nitrogen-scavenging pathway bypass"},
        {"treatment": "ACE Inhibitor (enalapril/captopril)", "n": 34, "pct": 85, "indication": "DCM — Level A (when haemodynamically stable); RAAS blockade; reduces LV dilatation"},
        {"treatment": "Beta-Blocker (carvedilol/metoprolol)", "n": 32, "pct": 80, "indication": "DCM — Level A; start when stable; carvedilol preferred in paediatric DCM"},
        {"treatment": "L-Carnitine supplementation", "n": 35, "pct": 88, "indication": "Secondary depletion — Level C; C0 target 30-60 µmol/L; TMEM70 causes secondary depletion"},
        {"treatment": "iNO (inhaled nitric oxide)", "n": 18, "pct": 45, "indication": "Neonatal PAH — Level B; 20 ppm; reduces pulmonary vascular resistance; wean over 1-2 weeks"},
        {"treatment": "Sildenafil (oral maintenance)", "n": 14, "pct": 35, "indication": "Persistent PAH after iNO weaning — Level B; maintenance after acute phase"},
        {"treatment": "Riboflavin (B2) empirical", "n": 18, "pct": 45, "indication": "Empirical mitochondrial cofactor — Level D; no direct Complex V evidence; generally safe"},
        {"treatment": "CoQ10 empirical", "n": 14, "pct": 35, "indication": "Empirical — Level D; no controlled evidence in Complex V deficiency; sometimes tried"},
        {"treatment": "Nasogastric / PEG feeding", "n": 24, "pct": 60, "indication": "Hypotonia + poor suck — Level C; NG from ICU discharge; PEG if persistent at 3-6 months"},
        {"treatment": "LEV (levetiracetam)", "n": 8, "pct": 20, "indication": "Seizures — PREFERRED AED — Level B; no mito toxicity; no ammonia effect; renal clearance"},
        {"treatment": "ECMO (extracorporeal membrane oxygenation)", "n": 5, "pct": 13, "indication": "Refractory PAH + cardiogenic shock — Level D (rescue); bridge to metabolic stabilisation"},
        {"treatment": "Heart transplant (completed)", "n": 6, "pct": 15, "indication": "Refractory DCM (EF <25%) — Level A; lower frequency than Barth; metabolic recovery possible"},
        {"treatment": "Diuretics (furosemide/spironolactone)", "n": 22, "pct": 55, "indication": "DCM fluid management — Level B adjunct; careful with electrolytes in renal impairment"},
    ]

    # Metabolic profile over time
    metabolic_by_age = [
        {"age_group": "Neonatal crisis (0-48h)", "lactate_mmol": 18, "nh3_umol": 185, "ph_mean": 7.05, "notes": "Most critical; ICU mandatory; mortality without intervention ~38%"},
        {"age_group": "ICU stabilisation (day 3-14)", "lactate_mmol": 8, "nh3_umol": 95, "ph_mean": 7.28, "notes": "Improving with glucose + bicarbonate + ammonia scavengers; wean respiratory support"},
        {"age_group": "Post-ICU (month 1-6)", "lactate_mmol": 4, "nh3_umol": 55, "ph_mean": 7.38, "notes": "Metabolic partial compensation; DCM management; PAH weaning; feeding support"},
        {"age_group": "Infant (6-12 months)", "lactate_mmol": 3, "nh3_umol": 40, "ph_mean": 7.38, "notes": "Stable on oral meds; hypotonia prominent; neurodevelopment assessment"},
        {"age_group": "Toddler/Child (1-5 yr)", "lactate_mmol": 2.5, "nh3_umol": 35, "ph_mean": 7.40, "notes": "Intercurrent illness risk — sick-day protocol; DCM may partially recover; PT/OT ongoing"},
    ]

    # Biomarker summary
    biomarker_summary = {
        "neonatal_lactate_peak_mmol": 18,
        "neonatal_lactate_range_mmol": "10-28",
        "lactic_acidosis_pct": 100,
        "hyperammonemia_pct": 90,
        "nh3_mean_umol": 150,
        "nh3_range_umol": "50-500",
        "mga_mean_mmol_cr": 180,
        "mga_range_mmol_cr": "50-300",
        "c4dc_elevated_pct": 0,
        "normal_acylcarnitine_pct": 95,
        "dcm_pct": 85,
        "pah_pct": 45,
        "hypoglycemia_pct": 35,
        "snhl_pct": 0,
        "optic_atrophy_pct": 0,
        "neutropenia_pct": 0,
        "id_pct": 65,
        "normal_cognition_pct": 35,
    }

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "phenotype_groups": [{"group": g, "n": c, "pct": round(c / n * 100)} for g, c in phenotype_groups],
        "variant_distribution": variant_dist,
        "treatment_distribution": treatment_dist,
        "metabolic_profile_by_age": metabolic_by_age,
        "biomarker_summary": biomarker_summary,
        "outcomes": {
            "neonatal_survival_pct": 62,
            "dcm_transplant_pct": 15,
            "dcm_stabilised_on_medical_therapy_pct": 60,
            "pah_resolved_pct": 75,
            "id_mild_moderate_pct": 65,
            "normal_cognition_pct": 35,
            "independent_ambulation_pct": 70,
            "peg_long_term_pct": 25,
            "median_icu_days": 28,
            "median_age_diagnosis_days": 3,
        },
    }


# ── definitions ───────────────────────────────────────────────────────────────
def get_definitions() -> dict:
    """TMEM70 Complex V Deficiency — definitions for /api/tmem70/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Mitochondrial Complex V Deficiency, Nuclear Type 2 / 3-MGA-uria Type VI / TMEM70 Deficiency",
        "gene": "TMEM70",
        "omim_gene": "612418",
        "omim_disease": "614052",
        "definitions": [
            {
                "term": "TMEM70 / Complex V (ATP Synthase) Assembly Factor",
                "definition": "TMEM70 (Transmembrane Protein 70; 260 amino acids; 8q11.23) is a nuclear-encoded inner mitochondrial membrane protein specifically required for the biogenesis of the ATP synthase Fo subunit c-ring (rotor ring). Without TMEM70, the c-ring fails to assemble, the F1 head cannot couple to Fo, and Complex V is severely deficient. TMEM70 LOF results in profound reduction (>80%) of Complex V activity in muscle, fibroblasts, and heart.",
                "relevance": "Complex V deficiency is confirmed by enzyme assay in muscle/fibroblasts (Complex V activity <20% of normal). TMEM70 sequencing (with MLPA for deletions) is the genetic test. Functional ATP synthesis can be measured in fibroblasts. Muscle histology: ragged-red fibres rare; COX-normal; Complex V immunostaining absent on BN-PAGE.",
            },
            {
                "term": "3-MGA-uria Type VI — Hyperammonemia Triad",
                "definition": "3-methylglutaconic aciduria Type VI is TMEM70-specific. The cardinal diagnostic triad is: (1) urinary 3-MGA elevation (50–300 mmol/mol Cr, secondary overflow from OXPHOS dysfunction); (2) lactic acidosis (severe, neonatal, L/P ratio >20); and (3) hyperammonemia (NH3 50–500 µmol/L, from urea cycle ATP failure). No other 3-MGA disease causes all three simultaneously. Acylcarnitine panel is NORMAL (no C4-DC — distinguishes from TAZ/Barth where C4-DC is pathognomonic).",
                "relevance": "When a neonate presents with hyperammonemia, always check urine organic acids (3-MGA) and lactate alongside plasma amino acids and urine orotic acid (for urea cycle). 3-MGA + high lactate = TMEM70 (or other mito disease) not UCD. Reflex to TMEM70 gene sequencing; do not treat as UCD and initiate protein restriction alone.",
            },
            {
                "term": "Czech/Slovak Roma Founder Variant c.317-2A>G",
                "definition": "The intronic splice variant c.317-2A>G (IVS2 splice donor) in TMEM70 is a founder mutation in the Czech and Slovak Roma population. It destroys the canonical GT splice donor at the intron 2/exon 3 boundary, causing exon 3 skipping, frameshift, premature stop codon, and NMD (nonsense-mediated decay). The protein is completely absent. This allele accounts for approximately 50% of all TMEM70 disease alleles worldwide and is homozygous in ~45% of Roma-origin patients.",
                "relevance": "In a Roma-origin neonate with lactic acidosis + hyperammonemia, test TMEM70 c.317-2A>G first (fast Sanger sequencing or targeted PCR) before waiting for WES. Homozygous c.317-2A>G = diagnostic. Compound heterozygous with another allele also diagnostic. Carrier frequency in Czech/Slovak Roma estimated ~1 in 40 — population screening under study.",
            },
            {
                "term": "Urea Cycle Failure in TMEM70: Why Hyperammonemia?",
                "definition": "The first enzyme of the urea cycle — carbamoyl phosphate synthetase I (CPS1) — requires 2 ATP per reaction. When Complex V is absent, mitochondrial ATP synthesis fails → CPS1 activity collapses → carbamoyl phosphate cannot be made → ammonia accumulates in the mitochondrial matrix → spills into blood → hyperammonemia. This is a SECONDARY hyperammonemia (not a primary urea cycle defect). Glutamine is also elevated (glutamate dehydrogenase uses ammonia when CPS1 fails), and glutamine is the transport form. The pattern (elevated glutamine + low citrulline + high NH3) can mimic CPS1 deficiency.",
                "relevance": "Differentiate TMEM70-secondary hyperammonemia from primary UCD: (1) TMEM70 has lactic acidosis (L/P >20) — primary UCDs do NOT have lactic acidosis; (2) TMEM70 has 3-MGA on organic acids; (3) TMEM70 has normal urine orotic acid (OTC deficiency has high orotic acid). Order organic acids, lactate, and ammonia simultaneously — never organic acids alone.",
            },
            {
                "term": "Why Ketogenic Diet is Contraindicated in TMEM70",
                "definition": "The ketogenic diet forces fat-based fuel use. Fatty acid oxidation generates FADH2 (from electron-transfer flavoprotein) and NADH, which donate electrons to the respiratory chain (Complexes I–IV). The energy stored in the proton gradient is then captured by Complex V (ATP synthase) to phosphorylate ADP → ATP. If Complex V is absent, the proton gradient cannot be converted to ATP regardless of the fuel source — FADH2/NADH production is futile. The KD in Complex V deficiency worsens the energy deficit, produces more reducing equivalents that cannot be converted to ATP, and can accelerate lactic acidosis.",
                "relevance": "Never prescribe KD in TMEM70. This contrasts with PDH deficiency and some other mito diseases where KD bypasses the enzymatic block upstream. Ensure that metabolic teams and neurologists involved in epilepsy management understand this distinction. If seizures occur and are refractory, use LEV, CLB, or LCM — never KD, never VPA.",
            },
            {
                "term": "Emergency Sick-Day Protocol for TMEM70",
                "definition": "Intercurrent illness (fever, vomiting, diarrhoea, surgery) causes catabolism → lactic acidosis crisis in TMEM70. Families must have a written emergency sick-day protocol: (1) maintain glucose intake — concentrated glucose drinks if oral, IV dextrose GIR ≥8 if unable to feed; (2) hospital attendance threshold: any vomiting >2 episodes, any fever >38°C, any lactate >5 mmol/L; (3) hospital action: IV 10% dextrose, bicarbonate, metabolic team alert; (4) hold all oral medications that may impair feeding or cause acidosis during crisis. Medical alert card/bracelet mandatory.",
                "relevance": "The sick-day protocol is as important as the maintenance treatment plan. Most deaths after the neonatal period in TMEM70 occur during intercurrent illness. Emergency metabolic team contact details must be available to the family 24/7. Local emergency departments must be pre-alerted (via metabolic emergency letter) so they start glucose and bicarbonate immediately without waiting for specialist review.",
            },
            {
                "term": "TMEM70 vs TAZ (Barth): DCM Prescribing DDx",
                "definition": "Both TMEM70 and TAZ/Barth cause neonatal DCM with 3-MGA-uria. KEY prescribing distinctions: (1) Acylcarnitine: TMEM70 normal (no C4-DC) vs Barth C4-DC pathognomonic; (2) Neutropenia: Barth 95% vs TMEM70 absent; (3) Inheritance: TMEM70 AR (both sexes) vs Barth X-linked (males only); (4) Hyperammonemia: TMEM70 90% vs Barth absent; (5) Cognition: Barth 100% normal vs TMEM70 ID 65%; (6) KD: CONTRAINDICATED in TMEM70 (Complex V absent) but can be considered in Barth for seizures if needed (Complex V intact); (7) VPA: ABSOLUTE CI in both, but mechanism differs (in TMEM70 VPA also raises ammonia — doubly dangerous).",
                "relevance": "If a male neonate has DCM + 3-MGA: C4-DC present → Barth (TAZ); C4-DC absent + hyperammonemia → TMEM70. If the acylcarnitine is normal and ammonia is elevated, TMEM70 should be the first specific gene tested (particularly in Roma-origin families). These two diseases look superficially similar but have opposite implications for some management decisions (KD, sex-specific genetics, ammonia treatment).",
            },
            {
                "term": "Long-Term Prognosis in TMEM70 Survivors",
                "definition": "Among patients surviving the neonatal crisis (~62% without optimal ICU; higher with modern management), long-term prognosis includes: mild-to-moderate intellectual disability in 60-70% (caused by neonatal hypoxic-ischemic insult from acidosis/hypoglycaemia and chronic ATP deficit); persistent hypotonia throughout childhood; DCM stabilised on ACE-I + BB in ~60% (transplant ~15%); PAH resolves in ~75% within first year. Riboflavin and CoQ10 produce mild improvements in some case series. Most Roma founder homozygotes have uniform severe neonatal presentation; missense compound heterozygotes occasionally milder.",
                "relevance": "Parents should be counselled that survivors typically have mild-moderate learning difficulties, need educational support, and may remain hypotonic with feeding difficulties into childhood. Neurological outcome is not as favourable as TAZ/Barth (where cognition is 100% normal). Regular multidisciplinary follow-up (metabolic, cardiology, neurodevelopment, dietetics) is essential throughout childhood and adolescence.",
            },
        ],
    }
