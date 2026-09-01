#!/usr/bin/env python3
"""TAZ Barth Syndrome Dashboard.

Barth Syndrome = X-linked dilated cardiomyopathy + skeletal myopathy + neutropenia + 3-MGA-uria

TAZ (Tafazzin) is a phospholipid-lysophospholipid transacylase in the inner mitochondrial membrane.
LOF → impaired cardiolipin remodeling → MLCL (monolysocardiolipin) accumulates, mature CL depleted
→ inner mitochondrial membrane instability → respiratory chain supercomplex disassembly
→ DCM (100%) + skeletal myopathy + neutropenia (95%) + 3-MGA-uria (Type II)

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. 3-MGA-uria Type II — CARDINAL with C4-DC (3-methylglutarylcarnitine) elevated on acylcarnitine panel
  2. DCM (dilated cardiomyopathy) 100% — onset birth to 2 yr; leading cause of death/transplant
  3. Neutropenia (95%) — cyclic or chronic; bacterial infection risk; G-CSF life-saving
  4. X-linked recessive — hemizygous males affected; carrier females asymptomatic (usually)
  5. MLCL:CL ratio elevated (>0.5) in blood/fibroblasts — most specific laboratory marker
  6. VPA: ABSOLUTE CI (mito disease — cardiolipin deficiency compounds respiratory chain failure)
  7. ACE inhibitor + beta-blocker: MANDATORY from diagnosis for DCM — Level A evidence
  8. Heart transplant required in ~30% (EF<25% refractory); excellent post-transplant outcomes
  9. NO sensorineural hearing loss (KEY DDx from SERAC1/MEGDEL where SNHL 100%)
 10. NO optic atrophy (KEY DDx from OPA3/Costeff, MECR/MEPAN where optic atrophy present)
 11. L-Carnitine: secondary depletion; supplement level C
 12. Xq28 — TAZ gene; ~200-300 patients known worldwide (ultra-rare X-linked)

TAZ BIOLOGY:
TAZ (Tafazzin, 292 amino acids, Xq28) encodes a phospholipid-lysophospholipid transacylase
(transacylase, acyltransferase) in the inner mitochondrial membrane (IMM) and intermembrane space.

Tafazzin catalyses cardiolipin remodeling:
  Step 1: Nascent CL (4 acyl chains; mixed saturated) made by CRLS1 (cardiolipin synthase)
  Step 2: TAZ cleaves one acyl chain → MLCL (monolysocardiolipin) — intermediate
  Step 3: TAZ reacylates MLCL with unsaturated acyl chain (linoleic acid, 18:2) → mature CL

Mature CL (tetralinoleoyl-CL, TLCL) is required for:
  - Respiratory chain supercomplex (respirasomes I+III₂+IV, I+III₂) assembly and stability
  - ATP synthase (Complex V) oligomerisation and cristae curvature
  - Apoptosis regulation (cytochrome c retention)
  - Mito-membrane potential maintenance and mitophagy (PINK1/Parkin)

LOF mechanism:
  TAZ LOF → MLCL accumulates (cannot be reacylated) → mature CL severely depleted →
    Complex I + III + IV supercomplex destabilisation → OXPHOS efficiency ↓ →
    Cardiomyocyte energy failure (high ATP demand) → DCM onset within first year of life
    Neutrophil energy deficit + mito apoptosis signalling → cyclic neutropenia
    Skeletal muscle OXPHOS insufficiency → myopathy, exercise intolerance
    3-MGA overflow pathway (shared with Type I/III/IV/V diseases) → 3-MGA-uria

Barth syndrome cardiolipin signature:
  MLCL:CL ratio >0.5 (normal <0.1) — detectable in dried blood spot, fibroblasts, or lymphocytes
  TLCL (18:2/18:2/18:2/18:2 CL) reduced to <10% of normal in affected males
  Acylcarnitine: C4-DC (3-methylglutarylcarnitine) elevated — reflects OXPHOS backup overflow

PROTEIN STRUCTURE (292 aa, Xq28):
  N-terminal membrane anchor (aa 1-50): IMM/IMS localisation signal; Tafazzin inserts into IMM
  Central transacylase domain (aa 51-200): Lipid binding groove; acyl chain acceptor site;
    GxSxG motif (serine lipase superfamily); most pathogenic missense cluster here
  C-terminal regulatory region (aa 201-292): IMM membrane-spanning helix; oligomerisation
  Most disease-causing missense: aa 57-262 (central catalytic + membrane anchor region)

PATHOGENIC VARIANT DISTRIBUTION (hemizygous males, n=40, seed-541):
  Missense in transacylase domain (aa 51-200): ~50% of alleles
    Diverse: p.Gly197Val, p.Arg94Cys, p.Trp179Stop, and other private variants
  Splice site variants: ~25% of alleles
    c.517-2A>G (intron 4 splice acceptor): ~10% (exon 5 skipping, LOF)
    Other intronic splice variants: ~15%
  Frameshift/nonsense: ~15%
    c.646delC (frameshift): ~5%; other null variants: ~10%
  Large deletions (exon-level): ~10%
    Detectable by aCGH/MLPA; WGS required for breakpoint characterisation

CLINICAL PHENOTYPE — BARTH SYNDROME:
  DILATED CARDIOMYOPATHY (100%) — CARDINAL FEATURE:
    Onset: in utero (fetal hydrops 15%) or first year of life (70% by age 1).
    Systolic dysfunction (EF <55%); LV dilatation; endocardial fibroelastosis rare.
    Left ventricular non-compaction (LVNC) variant in 20-25% — trabeculated LV.
    Management: ACE-I + BB mandatory (Level A); digoxin adjunct; diuretics prn.
    Heart transplant: required in ~30% with refractory EF <25%; post-Tx 10yr survival ~80%.
    KEY DDx: DCM also in DNAJC19 (DCMA) — but DCMA has cerebellar ataxia, NO neutropenia, no C4-DC.
  NEUTROPENIA (95%) — CYCLIC OR CHRONIC:
    Cyclic neutropenia: 21-day cycles; ANC nadir <0.5×10⁹/L; recurrent bacterial infections.
    Chronic neutropenia: ~30% of patients; sustained ANC <1.0×10⁹/L.
    Risk: recurrent bacterial infection (skin/respiratory), sepsis, oral ulcers, gingivitis.
    G-CSF (filgrastim): First-line for recurrent severe infections — Level B.
    Antibiotic prophylaxis (TMP-SMX or azithromycin): consider in recurrent infections — Level C.
    KEY DDx: Neutropenia absent in DNAJC19, OPA3, SERAC1, AUH — UNIQUE to TAZ among 3-MGA diseases.
  SKELETAL MYOPATHY (100%):
    Proximal > distal; lower > upper extremity; exercise intolerance disproportionate to CK.
    CK: mildly elevated or normal (not rhabdomyolysis).
    Fatigability prominent; orthopnea if diaphragm involved.
    PT/OT/exercise therapy: structured, supervised; reduces deconditioning; improves QoL.
  3-METHYLGLUTACONIC ACIDURIA (100%) — TYPE II:
    Urinary 3-MGA: 20-200 mmol/mol creatinine (moderate; lower than AUH Type I primary form).
    C4-DC (3-methylglutarylcarnitine) elevated on acylcarnitine NBS panel — PATHOGNOMONIC FOR BARTH.
    C4-OH may be borderline elevated; C0 (free carnitine) usually low.
    MLCL:CL ratio >0.5 in DBS/fibroblasts — most specific, used to confirm NBS positive.
  GROWTH RETARDATION (95%):
    Short stature (height <3rd centile in 70%); low weight.
    Catch-up growth with cardiac optimisation and nutritional support.
    GH: not indicated; growth reflects cardiac/nutritional status.
  COGNITIVE FUNCTION: NORMAL (100%):
    Intelligence normal — CRITICAL: cognition preserved in Barth syndrome (DDx from most other
    3-MGA diseases where ID is common: AUH, SERAC1, DNAJC19 all have ID in majority).
    Educational planning: standard; support for fatigue/hospitalisation-related school absence.
  OPTIC ATROPHY: ABSENT (KEY DDx from OPA3 100%, MECR 80-90%)
  SENSORINEURAL HEARING LOSS: ABSENT (KEY DDx from SERAC1/MEGDEL 100%)
  MOVEMENT DISORDER: ABSENT (KEY DDx from OPA3 chorea 85%, FTL chorea, DCAF17 extrapyramidal)
  LIVER DISEASE: ABSENT (KEY DDx from SERAC1 neonatal cholestasis 67%)
  INTELLECTUAL DISABILITY: ABSENT (KEY DDx: all other 3-MGA diseases have at least mild ID)

TREATMENT & PHARMACOGENOMICS:
  ACE Inhibitor (enalapril/captopril/lisinopril): DCM — MANDATORY — Level A
    Start from diagnosis; target-dose; reduces LV dilatation and improves systolic function.
    Monitor: renal function, K+, blood pressure.
  Beta-Blocker (carvedilol/metoprolol): DCM — MANDATORY — Level A
    Start when stable; target HR 55-65 bpm; improves cardiac remodeling.
    Carvedilol preferred (alpha-1 + beta-1/2 block; superior to metoprolol in paediatric DCM).
  Heart Transplant: Refractory DCM (EF<25%) — Level A indication (~30% of patients)
    Post-Tx 10yr survival ~80%; TAZ defect is cardiac-specific; cardiomyopathy not recurrent in transplanted heart.
    List early; TAZ patients good post-Tx candidates (preserved cognition, normal other organs).
  G-CSF (Filgrastim): Neutropenia — Level B
    For recurrent severe bacterial infections or ANC <0.5×10⁹/L sustained.
    Dose: 5-10 mcg/kg/day SC; cycle tracking helps predict nadir timing.
    Monitor CBC 2× weekly during dose tititation.
  L-Carnitine: Secondary depletion — Level C
    Carnitine C0 low in majority; supplement to target C0 30-60 µmol/L.
    50-100 mg/kg/day divided; well tolerated.
  Linoleic Acid Supplementation (investigational): Cardiolipin remodeling substrate
    Linoleic acid (18:2n-6) is the preferred acyl donor for TAZ-mediated CL remodeling.
    Investigational: pilot data suggest improved CL profile; not yet standard of care.
  Elamipretide (SS-31): Investigational — Phase II TAZPOWER trial
    Mitochondria-targeted peptide; stabilises CL-cytochrome c interaction; improves OXPHOS.
    TAZPOWER trial (n=12 crossover): improved 6MWT + fatigue — Phase II results 2021.
    Not yet FDA approved; investigational; compassionate use available via BSF.
  VPA (Valproate): ABSOLUTE CONTRAINDICATION — Level A prohibition
    VPA inhibits Complex I + CoA sequestration → fatal hepatotoxicity + lactic acidosis in mito disease.
    Cardiolipin deficiency in Barth compounds this → profound Complex I/IV collapse.
    NEVER give VPA in Barth syndrome (TAZ LOF) under any circumstances.
  LEV (Levetiracetam): Seizures — PREFERRED — Level B
    Renal excretion; no mito toxicity; no CYP interaction; first-line if seizures occur.
    NOTE: Seizures uncommon in Barth (absent brain disease) — ensure DCM-related cerebral embolic
    events excluded before labelling as primary epilepsy.
  Digoxin: DCM adjunct — Level C
    Use with ACE-I/BB if suboptimal response; avoid toxicity (narrow window).
  Warfarin / Anticoagulation: DCM with severely reduced EF <25% or LVNC — Level B
    Thromboembolic risk with severe systolic dysfunction; anticoagulate if EF <25%.
  Antibiotic Prophylaxis: Neutropenia — Level C (TMP-SMX or azithromycin)
    For recurrent severe bacterial infections; review annually.
  IVIG: Neutropenia (refractory, severe infections) — Level D investigational
    Some benefit in cyclic neutropenia unresponsive to G-CSF; limited evidence.
  Nutritional Support: Growth retardation — Level C
    High-calorie diet; NG/PEG tube if failure to thrive; optimise cardiac output first.
  PHT/CBZ/OXC: AVOID — worsens cardiac conduction (Na-channel block); avoid in DCM.

ANAESTHESIA / PROCEDURAL NOTES:
  Alert anaesthesia team: mitochondrial disease (cardiolipin deficiency, Complex I/IV risk).
  Propofol: use with caution (PRIS risk in mito disease); prefer inhalational agents.
  Pre-procedure: ensure ACE-I/BB continued (no peri-operative hold); glucose-containing IV fluids.
  Fasting: minimise pre-operative fasting duration; dextrose IV during NPO period.
  Cardiac monitoring: invasive monitoring for any major procedure (unstable DCM risk).

MONITORING SCHEDULE:
  Cardiology: echo + ECG 3-6 monthly; BNP/troponin quarterly
  Haematology: CBC weekly until stable; then monthly; ANC tracking for cyclic neutropenia
  Metabolic: acylcarnitine + 3-MGA annually; MLCL:CL ratio 1-2 yearly
  Nutrition: weight/height monthly in children; dietitian review
  Neurodevelopment: cognition is NORMAL — standard school support
"""

import random
from datetime import date

SEED = 541  # 40-patient cohort seed

# ── overview ──────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """TAZ Barth Syndrome — overview for /api/taz/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Barth Syndrome (3-MGA-uria Type II / X-linked Cardiomyopathy-Neutropenia-Myopathy)",
        "gene": "TAZ; Tafazzin; phospholipid-lysophospholipid transacylase; 292 aa; IMM cardiolipin remodeling",
        "chromosome": "Xq28",
        "omim_gene": "300394",
        "omim_disease": "302060",
        "inheritance": "X-linked recessive; hemizygous males affected; carrier females asymptomatic (usually)",
        "prevalence": "~1:300,000 live male births; ~200-300 patients known worldwide (2026)",
        "first_described": "Barth et al. 1983 (fatal X-linked cardiomyopathy-neutropenia, Dutch family)",
        "protein": "Tafazzin (TAZ) — inner mitochondrial membrane transacylase; remodels nascent CL → mature tetralinoleoyl-CL (TLCL); MLCL intermediate",
        "category": "3-MGA-uria Type II / X-linked mitochondrial cardiomyopathy / Barth Syndrome",
        "kpis": {
            "dcm_pct": 100,
            "neutropenia_pct": 95,
            "myopathy_pct": 100,
            "normal_cognition_pct": 100,
            "heart_transplant_pct": 30,
            "c4dc_elevated_pct": 100,
            "mlcl_cl_ratio_cutoff": ">0.5",
            "vpa_ci": "ABSOLUTE CI",
        },
        "clinical_highlights": [
            "DCM (dilated cardiomyopathy) 100% — leading cause of morbidity/mortality; onset birth–2 yr",
            "Neutropenia 95% — cyclic or chronic; bacterial infection/sepsis risk; G-CSF life-saving",
            "Skeletal myopathy 100% — proximal weakness; exercise intolerance; normal CK",
            "Growth retardation 95% — short stature; low weight; catches up with cardiac optimisation",
            "Cognition: NORMAL (100%) — unique among 3-MGA diseases; all other types have ID",
            "C4-DC elevated on NBS acylcarnitine — PATHOGNOMONIC for Barth (TAZ); confirms diagnosis",
            "MLCL:CL ratio >0.5 — most specific lab marker; detectable in DBS/fibroblasts",
            "X-linked: hemizygous males; carrier female screening important (TAZ gene sequencing)",
            "Heart transplant: 30% need it; post-Tx outcomes excellent; TAZ defect not recurrent in donor heart",
        ],
        "contraindications": [
            {
                "drug": "VPA (Valproate / Valproic Acid / Depakote)",
                "level": "ABSOLUTE CI",
                "reason": "Complex I inhibition + CoA sequestration → fatal hepatotoxicity + lactic acidosis in cardiolipin deficiency; NEVER use in Barth syndrome",
            },
            {
                "drug": "PHT / CBZ / OXC (Na-channel AEDs)",
                "level": "AVOID in DCM",
                "reason": "Na-channel blockade → cardiac conduction slowing; additive to DCM cardiac instability; cardiac monitoring mandatory if used",
            },
            {
                "drug": "Propofol (procedural)",
                "level": "CAUTION (PRIS risk)",
                "reason": "Propofol inhibits Complex I → PRIS risk in mitochondrial disease; prefer inhalational agents or ketamine; alert anaesthesia team",
            },
            {
                "drug": "Carnitine depletion (avoidance of L-carnitine)",
                "level": "AVOID under-treating",
                "reason": "Secondary carnitine deficiency universal; C0 low in 90%+ of Barth patients; supplement to maintain C0 30-60 µmol/L",
            },
        ],
        "thresholds": [
            {"marker": "MLCL:CL ratio", "cutoff": ">0.5", "interpretation": "Diagnostic for TAZ LOF (normal <0.1); most specific marker for Barth syndrome"},
            {"marker": "C4-DC (acylcarnitine NBS)", "cutoff": "Elevated (>0.5 µmol/L)", "interpretation": "3-methylglutarylcarnitine — pathognomonic for Barth; triggers reflex MLCL:CL testing"},
            {"marker": "Urinary 3-MGA", "cutoff": "20–200 mmol/mol Cr", "interpretation": "Type II classification; secondary overflow pathway; moderate elevation"},
            {"marker": "LV ejection fraction (echo)", "cutoff": "EF <25%", "interpretation": "Threshold for cardiac transplant listing; urgent cardiology review"},
            {"marker": "ANC (absolute neutrophil count)", "cutoff": "<0.5 × 10⁹/L", "interpretation": "Severe neutropenia; start G-CSF (filgrastim); infection prophylaxis"},
        ],
        "ddx_table": [
            {
                "disease": "DNAJC19 (DCMA Syndrome) — 3-MGA Type IV",
                "shared": "DCM 100%, 3-MGA-uria",
                "distinguishing": "DCMA has cerebellar ataxia (95%), male genital anomalies; NO neutropenia; NO C4-DC; AR not X-linked; Hutterite/Mennonite founder",
            },
            {
                "disease": "SERAC1 (MEGDEL Syndrome) — 3-MGA Type V",
                "shared": "3-MGA-uria",
                "distinguishing": "SERAC1 has SNHL 100%; Leigh-like MRI; neonatal liver; NO DCM; NO neutropenia; normal acylcarnitine (no C4-DC)",
            },
            {
                "disease": "OPA3 (Costeff Syndrome) — 3-MGA Type III",
                "shared": "3-MGA-uria",
                "distinguishing": "OPA3 has optic atrophy 100%; chorea 85%; NO DCM; NO neutropenia; Iraqi Jewish founder",
            },
            {
                "disease": "AUH — 3-MGA Type I",
                "shared": "3-MGA-uria",
                "distinguishing": "AUH is primary 3-MGA (enzyme defect); C5-OH borderline; NO C4-DC; NO DCM; NO neutropenia; leucine-restriction responds",
            },
            {
                "disease": "MECR (MEPAN) — 3-MGA + GP iron",
                "shared": "3-MGA-uria",
                "distinguishing": "MECR has GP iron on SWI; optic atrophy 80%; dystonia; NO DCM; NO neutropenia; VPA absolute CI for different mechanism",
            },
        ],
    }


# ── breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    """TAZ Barth Syndrome — patient breakdown for /api/taz/breakdown."""
    rng = random.Random(SEED)
    n = 40

    # Phenotype groups
    phenotype_groups = [
        ("Classic Barth (DCM + neutropenia + myopathy)", 28),
        ("Barth with LVNC (LV non-compaction variant)", 8),
        ("Mild/attenuated Barth (preserved EF, mild neutropenia)", 4),
    ]
    assert sum(c for _, c in phenotype_groups) == n

    # Variant distribution (hemizygous males, X-linked)
    variant_dist = [
        {"variant": "Missense: transacylase domain (aa 51-200) — diverse private variants", "n_alleles": 20, "pct": 50, "effect": "Reduced transacylase activity; MLCL accumulation; severity varies by residual activity"},
        {"variant": "Splice site: c.517-2A>G (intron 4 splice acceptor)", "n_alleles": 4, "pct": 10, "effect": "Exon 5 skipping; frameshift; complete LOF; neonatal DCM onset"},
        {"variant": "Other intronic splice variants", "n_alleles": 6, "pct": 15, "effect": "Variable exon skipping; LOF or partial LOF depending on cryptic splice use"},
        {"variant": "Frameshift: c.646delC and others", "n_alleles": 6, "pct": 15, "effect": "Premature stop; complete null; severe neonatal presentation"},
        {"variant": "Large exon deletion (aCGH/MLPA)", "n_alleles": 4, "pct": 10, "effect": "Complete null; requires array CGH or MLPA for detection; WGS breakpoint characterisation"},
    ]

    # Treatment distribution
    treatment_dist = [
        {"treatment": "ACE Inhibitor (enalapril/captopril)", "n": 40, "pct": 100, "indication": "DCM — MANDATORY Level A; all patients from diagnosis"},
        {"treatment": "Beta-Blocker (carvedilol/metoprolol)", "n": 38, "pct": 95, "indication": "DCM — MANDATORY Level A; start when haemodynamically stable"},
        {"treatment": "L-Carnitine supplementation", "n": 38, "pct": 95, "indication": "Secondary carnitine depletion — Level C; C0 target 30-60 µmol/L"},
        {"treatment": "G-CSF (filgrastim)", "n": 30, "pct": 75, "indication": "Neutropenia — Level B; for ANC <0.5×10⁹/L or recurrent infections"},
        {"treatment": "Antibiotic prophylaxis (TMP-SMX/azithromycin)", "n": 18, "pct": 45, "indication": "Neutropenia-related infection — Level C"},
        {"treatment": "Diuretics (furosemide/spironolactone)", "n": 25, "pct": 63, "indication": "DCM fluid management — Level B adjunct"},
        {"treatment": "Anticoagulation (warfarin/LMWH)", "n": 12, "pct": 30, "indication": "Severe DCM EF<25% or LVNC — Level B thromboembolism prophylaxis"},
        {"treatment": "Heart transplant (completed)", "n": 12, "pct": 30, "indication": "Refractory DCM EF<25% — Level A indication; excellent post-Tx outcomes"},
        {"treatment": "Elamipretide (SS-31) investigational", "n": 3, "pct": 8, "indication": "Compassionate use / trial — stabilises CL-cytochrome c; TAZPOWER trial Phase II"},
        {"treatment": "Nutritional support / PEG tube", "n": 10, "pct": 25, "indication": "Failure to thrive — Level C; high-calorie diet; PEG if oral intake inadequate"},
        {"treatment": "LEV (levetiracetam)", "n": 4, "pct": 10, "indication": "Seizures (uncommon in Barth) — preferred over VPA (ABSOLUTE CI)"},
        {"treatment": "ICD / cardiac device", "n": 6, "pct": 15, "indication": "Arrhythmia / ventricular tachycardia — Level B in refractory DCM with arrhythmia"},
    ]

    # Cardiac outcomes by age
    cardiac_by_age = [
        {"age_group": "Neonatal (<1 month)", "dcm_present_pct": 40, "mean_ef": 38, "notes": "In utero hydrops 15%; neonatal presentation with severe DCM"},
        {"age_group": "Infant (1-12 months)", "dcm_present_pct": 85, "mean_ef": 42, "notes": "Majority diagnosed in first year; echo screening at diagnosis"},
        {"age_group": "Toddler (1-3 yr)", "dcm_present_pct": 100, "mean_ef": 45, "notes": "100% have DCM by age 3; stabilised with ACE-I/BB in 70%"},
        {"age_group": "Child (3-12 yr)", "dcm_present_pct": 100, "mean_ef": 48, "notes": "Improved EF with optimised therapy; transplant ~15% this age"},
        {"age_group": "Adolescent/Adult (>12 yr)", "dcm_present_pct": 100, "mean_ef": 50, "notes": "Survivors: improved long-term; cumulative transplant 30%"},
    ]

    # MLCL:CL ratio and biomarkers
    biomarker_summary = {
        "mlcl_cl_ratio_mean": 0.72,
        "mlcl_cl_ratio_range": "0.45-1.20",
        "mlcl_cl_above_05_pct": 100,
        "c4dc_elevated_pct": 100,
        "c0_carnitine_low_pct": 90,
        "mga_mean_mmol_cr": 85,
        "mga_range_mmol_cr": "20-200",
        "lvnc_pct": 20,
        "neutropenia_cyclic_pct": 60,
        "neutropenia_chronic_pct": 35,
        "neutropenia_absent_pct": 5,
        "normal_cognition_pct": 100,
        "snhl_pct": 0,
        "optic_atrophy_pct": 0,
        "movement_disorder_pct": 0,
    }

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "phenotype_groups": [{"group": g, "n": c, "pct": round(c / n * 100)} for g, c in phenotype_groups],
        "variant_distribution": variant_dist,
        "treatment_distribution": treatment_dist,
        "cardiac_outcomes_by_age": cardiac_by_age,
        "biomarker_summary": biomarker_summary,
        "outcomes": {
            "heart_transplant_pct": 30,
            "post_tx_10yr_survival_pct": 80,
            "ef_normalised_on_therapy_pct": 35,
            "sepsis_mortality_pct": 8,
            "median_age_at_diagnosis_months": 6,
            "median_age_at_transplant_years": 4,
            "school_attendance_normal_pct": 75,
            "independent_ambulation_pct": 90,
        },
    }


# ── definitions ───────────────────────────────────────────────────────────────
def get_definitions() -> dict:
    """TAZ Barth Syndrome — definitions for /api/taz/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Barth Syndrome (3-MGA-uria Type II / X-linked Cardiomyopathy-Neutropenia-Myopathy)",
        "gene": "TAZ",
        "omim_gene": "300394",
        "omim_disease": "302060",
        "definitions": [
            {
                "term": "TAZ / Tafazzin",
                "definition": "292-amino acid phospholipid-lysophospholipid transacylase localised to the inner mitochondrial membrane (IMM) and intermembrane space. TAZ remodels nascent cardiolipin (CL) by removing a saturated acyl chain to form MLCL (monolysocardiolipin), then reacylating MLCL with linoleic acid (18:2) to produce mature tetralinoleoyl-CL (TLCL). TAZ is encoded at Xq28.",
                "relevance": "TAZ LOF → impaired CL remodeling → MLCL accumulates, mature TLCL depleted → respiratory chain supercomplex destabilisation → OXPHOS failure → DCM + skeletal myopathy + neutropenia + 3-MGA-uria.",
            },
            {
                "term": "Cardiolipin (CL) and MLCL:CL Ratio",
                "definition": "Cardiolipin is a unique dimeric phospholipid found almost exclusively in the IMM and required for respiratory chain supercomplex (I+III₂+IV) assembly. Mature CL has four linoleic acid (18:2) chains (TLCL). MLCL (monolysocardiolipin) is the TAZ reaction intermediate. MLCL:CL ratio >0.5 (normal <0.1) indicates TAZ LOF — the most specific laboratory marker for Barth syndrome, detectable in dried blood spot (DBS) or fibroblasts.",
                "relevance": "MLCL:CL ratio is the confirmatory test after C4-DC elevation on NBS. Request from specialist laboratory (Barth Syndrome Foundation has referral lab list). Ratio >0.5 in a male with DCM + neutropenia = Barth until proven otherwise.",
            },
            {
                "term": "3-MGA-uria Type II (Barth Syndrome)",
                "definition": "3-methylglutaconic aciduria Type II is the designation for Barth syndrome — elevated urinary 3-methylglutaconic acid (3-MGA, 20-200 mmol/mol Cr) secondary to OXPHOS dysfunction and HMG-CoA pathway overflow. Barth is the only X-linked 3-MGA disease. C4-DC (3-methylglutarylcarnitine) is uniquely elevated in Barth on acylcarnitine panels — the other 3-MGA types (I, III, IV, V) do NOT have C4-DC elevation.",
                "relevance": "C4-DC elevation on standard NBS acylcarnitine panel is pathognomonic for Barth. Any male with C4-DC elevation should have MLCL:CL ratio and TAZ gene sequencing expedited. Normal acylcarnitine panel (no C4-DC) virtually excludes Barth.",
            },
            {
                "term": "Dilated Cardiomyopathy (DCM) in Barth Syndrome",
                "definition": "100% of Barth syndrome patients develop DCM. Mechanism: TLCL deficiency → Complex I + III + IV supercomplex destabilisation → ATP synthesis failure in cardiomyocytes (highest OXPHOS demand per cell type) → LV systolic dysfunction → LV dilatation. EF typically 20-50% at diagnosis. LVNC (left ventricular non-compaction) variant in 20-25% — trabeculated LV endocardium on echo/MRI.",
                "relevance": "DCM is the main determinant of prognosis. ACE inhibitor + beta-blocker from diagnosis (Level A) stabilises EF in ~70%. Heart transplant required in ~30% with refractory EF <25%; TAZ defect is cardiac-specific and does not recur in donor heart. Post-transplant 10yr survival ~80%.",
            },
            {
                "term": "Neutropenia in Barth Syndrome",
                "definition": "95% of TAZ-LOF patients have cyclic (60%) or chronic (35%) neutropenia. Mechanism: neutrophil precursors have high mitochondrial OXPHOS demand; TLCL deficiency → apoptosis of marrow granulocyte precursors → reduced ANC. Cyclic neutropenia: 21-day cycles with ANC nadir <0.5×10⁹/L. Risk: recurrent bacterial infections, sepsis, oral ulcers, gingivitis. Neutropenia absent in all other 3-MGA diseases — unique to TAZ.",
                "relevance": "G-CSF (filgrastim) 5-10 mcg/kg/day SC is first-line for neutropenia-related infections (Level B). Requires CBC monitoring twice weekly during dose titration. Infection early warning education for families essential. Sepsis mortality 8% — prompt antibiotic initiation for fever with neutropenia.",
            },
            {
                "term": "X-linked Inheritance and Carrier Females",
                "definition": "Barth syndrome is X-linked recessive — hemizygous males (one X) are affected; females with one mutant TAZ allele are obligate carriers. Carrier females are typically asymptomatic because of preferential skewed X-inactivation (normal allele expressed preferentially in most tissues). Rare: highly skewed X-inactivation in some carrier females → mild DCM or borderline MLCL:CL ratio. All sons of a carrier female have 50% chance of Barth syndrome; all daughters are at risk of being carriers.",
                "relevance": "Family screening: offer TAZ gene sequencing to all female first-degree relatives of affected males. Prenatal/preimplantation diagnosis available. Carrier female cardiac screening (echo) recommended 5-yearly. Unlike autosomal diseases in the 3-MGA series (AUH, OPA3, SERAC1, DNAJC19 are AR), Barth is X-linked — no affected females unless extreme skewing.",
            },
            {
                "term": "VPA Absolute Contraindication in Barth Syndrome",
                "definition": "Valproate (VPA) is absolutely contraindicated in Barth syndrome. Mechanism: VPA inhibits Complex I (direct; additive to pre-existing Complex I/IV supercomplex deficiency from TLCL depletion) AND sequesters CoA (CoA trapping) → acyl-CoA build-up → mitochondrial toxicity. VPA also causes hepatotoxicity via POLG pathway independent of this mechanism. In Barth: VPA has caused fatal hepatotoxic-lactic acidotic crises even at low doses.",
                "relevance": "Barth patients presenting with seizures (uncommon — always exclude DCM-related cerebral embolism first) should receive LEV (levetiracetam) as first-line AED. If escalation needed: CLB, LCM, or KD (ketogenesis intact in Barth) — never VPA. Ensure hospital neurology and cardiology teams are jointly aware of the VPA prohibition.",
            },
            {
                "term": "Elamipretide (SS-31) — Investigational Therapy",
                "definition": "Elamipretide (SS-31, MTP-131) is a mitochondria-targeted tetrapeptide that selectively binds to cardiolipin at the IMM. Mechanism: stabilises CL-cytochrome c interaction at Complex III → reduces electron leak → improves OXPHOS efficiency. TAZPOWER trial (n=12, crossover, 2021): elamipretide improved 6-minute walk test distance and fatigue scores vs placebo in Barth syndrome adults.",
                "relevance": "Not yet FDA approved; investigational use available via Barth Syndrome Foundation compassionate use programme. Phase II data promising but sample size small. Discuss with Barth specialist before initiating. SteadyMed/Stealth BioTherapeutics manufactures; TAZPOWER 2 long-term extension ongoing.",
            },
            {
                "term": "Barth Syndrome Foundation (BSF) Registry",
                "definition": "The Barth Syndrome Foundation maintains the international Barth Syndrome Registry — the primary source for natural history data, clinical trial recruitment, MLCL:CL reference lab referrals, and specialist clinic directory (>15 expert centres worldwide). BSF also maintains a family support network and coordinates the Barth Syndrome Special Interest Group (BSSIG) at international cardiology/genetics meetings.",
                "relevance": "Newly diagnosed patients and their families should be referred to the Barth Syndrome Foundation for registry enrolment, specialist clinic connection, and trial eligibility assessment. Clinical teams should register with BSF to access updated diagnostic and management protocols and to report novel variants.",
            },
            {
                "term": "TAZ vs DNAJC19 DCM: Prescribing DDx",
                "definition": "Both TAZ (Barth) and DNAJC19 (DCMA) cause DCM with 3-MGA-uria. KEY distinctions for prescribing: (1) TAZ = X-linked, only males; DNAJC19 = AR, any sex; (2) Barth has neutropenia 95% — DNAJC19 has NO neutropenia; (3) Barth has C4-DC on NBS — DNAJC19 has normal acylcarnitine; (4) DNAJC19 has cerebellar ataxia 95% — Barth has normal neurology; (5) DNAJC19 has male genital anomalies (cryptorchidism 75%) — Barth does not; (6) VPA moderate caution in DNAJC19 vs ABSOLUTE CI in Barth; (7) Barth cognition normal; DNAJC19 mild ID in 30-40%.",
                "relevance": "If a male has DCM + 3-MGA-uria: C4-DC elevated → Barth (TAZ). C4-DC normal + cerebellar ataxia → DNAJC19. C4-DC normal + SNHL → SERAC1. These three genes cover the major DCM-3MGA overlap; acylcarnitine panel + clinical features separate them without needing WES in most cases.",
            },
        ],
    }
