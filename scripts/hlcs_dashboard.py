#!/usr/bin/env python3
"""HLCS (Holocarboxylase Synthetase / Multiple Carboxylase Deficiency — Neonatal/Early-Onset) Epilepsy Dashboard.

HLCS encodes holocarboxylase synthetase (also called biotin-[propionyl-CoA-carboxylase
(ATP-hydrolysing)] ligase), the enzyme that covalently attaches biotin to all four
biotin-dependent carboxylases in mitochondria and cytoplasm:

  FOUR BIOTIN-DEPENDENT CARBOXYLASES THAT HLCS MUST BIOTINYLATE:
    1. Pyruvate carboxylase (PC)          — mitochondrial; gluconeogenesis + TCA anaplerosis
    2. Propionyl-CoA carboxylase (PCC)    — mitochondrial; odd-chain FA / Ile/Val/Met catabolism
       (α-subunit PCCA + β-subunit PCCB — separate genes; HLCS biotinylates PCCA biotin domain)
    3. 3-Methylcrotonyl-CoA carboxylase (MCC) — mitochondrial; leucine catabolism
       (α-subunit MCCC1 + β-subunit MCCC2 — separate genes; HLCS biotinylates MCCC1 biotin domain)
    4. Acetyl-CoA carboxylase (ACC)       — cytoplasmic; fatty acid synthesis
       (ACACA/ACC1 ubiquitous + ACACB/ACC2 — HLCS biotinylates ACACA biotin domain)

  HLCS CATALYTIC MECHANISM (biotin ligase / biotinyl-AMP intermediate):
    Step 1: Biotin + ATP → biotinyl-AMP (adenylyl intermediate) + PPi  [HLCS active site]
    Step 2: Biotinyl-AMP + ε-amino group of specific lysine (biotin acceptor site on apoapocarboxylase)
            → holocarboxylase (biotinylated, active) + AMP
    HLCS LOF → biotin cannot be attached to any of the four apocarboxylases →
    all four carboxylases remain in their INACTIVE apoenzyme forms → MULTIPLE simultaneous
    metabolic blocks → Multiple Carboxylase Deficiency (MCD) — neonatal onset.

  METABOLIC CONSEQUENCES of HLCS LOF (four simultaneous blocks):
    PC block (pyruvate carboxylase absent):
      Pyruvate → [BLOCKED] → oxaloacetate → gluconeogenesis BLOCKED;
      TCA anaplerosis via OAA impaired; LACTIC ACIDOSIS (pyruvate/lactate accumulate);
      HYPERAMMONEMIA (OAA deficiency → reduced N-acetylglutamate → carbamoyl-phosphate
      synthetase 1 activity drops → urea cycle flux reduced → NH4+ rises)
    PCC block (propionyl-CoA carboxylase absent):
      Propionyl-CoA (from Ile/Val/Met/odd-chain FA) → [BLOCKED] → methylmalonyl-CoA ABSENT;
      → propionylglycine, 3-hydroxypropionate, methylcitrate accumulate in urine;
      → propionylcarnitine (C3) elevated in NBS acylcarnitines
    MCC block (3-methylcrotonyl-CoA carboxylase absent):
      3-Methylcrotonyl-CoA (from leucine catabolism) → [BLOCKED] → 3-methylglutaconyl-CoA ABSENT;
      → 3-hydroxyisovalerate, 3-methylcrotonylglycine, 3-methylcrotonylcarnitine (C5-OH) accumulate;
      → C5-OH elevated in NBS (MOST SENSITIVE NBS MARKER for MCD)
    ACC block (acetyl-CoA carboxylase absent):
      Acetyl-CoA → [BLOCKED] → malonyl-CoA (fatty acid synthesis precursor) ABSENT;
      → impaired de novo lipogenesis; skin barrier dysfunction (fatty acid deficiency)

  PATHOGNOMONIC COMBINED BIOMARKER PATTERN in HLCS deficiency:
    Urine organic acids: 3-hydroxyisovaleric acid (HIGH — MCC block) +
                        3-methylcrotonylglycine (HIGH — MCC block) +
                        methylcitric acid (HIGH — PCC block) +
                        3-hydroxypropionate (HIGH — PCC block) +
                        propionylglycine (HIGH — PCC block) +
                        lactic acid (HIGH — PC block)
    Plasma acylcarnitines NBS: C5-OH (3-methylcrotonylcarnitine, HIGH — MCC block)
                              + C3 (propionylcarnitine, elevated — PCC block)
    Blood gases: metabolic acidosis (anion-gap) + ketonuria
    Ammonia: elevated (from PC block → OAA depletion → urea cycle impairment)
    BIOTIN LEVEL: often NORMAL (not a biotin transport defect; HLCS = biotin LIGATION failure)
    BIOTINIDASE ACTIVITY: NORMAL (key differential from BTD deficiency)

AUTOSOMAL RECESSIVE:
  HLCS gene: 21q22.13; Autosomal Recessive (AR)
  Protein: ~726 aa full-length isoform (~517 aa short isoform; long isoform active)
  OMIM gene: *609018; disease: #253270 (MULTIPLE CARBOXYLASE DEFICIENCY, NEONATAL /
             EARLY-ONSET MULTIPLE CARBOXYLASE DEFICIENCY)
  Prevalence: ~1:87,000 (Japan) to ~1:200,000+ (worldwide); most common in Japan

HLCS vs BTD (Biotinidase Deficiency) — CRITICAL DIFFERENTIAL:
  Both HLCS and BTD cause Multiple Carboxylase Deficiency;
  Both respond dramatically to biotin supplementation.
  HLCS: neonatal/early-infantile onset (day 1–10); biotinidase activity NORMAL;
        biotin plasma level often NORMAL; enzyme defect is LIGATION of biotin.
  BTD: onset usually >6 months (late-onset form) or early-infantile;
       biotinidase activity <30% of normal (DIAGNOSTIC); biotin plasma level LOW;
       enzyme defect is RECYCLING of protein-bound biotin → biotin depletion.
  THE SINGLE MOST IMPORTANT DIAGNOSTIC TEST: serum biotinidase activity (NORMAL in HLCS; LOW in BTD).

BIOTIN TREATMENT — DRAMATIC RESPONSE (if started early):
  Biotin 10–20 mg/day PO (standard); 20–40 mg/day for some severe/null HLCS variants;
  Corrects all four carboxylase deficiencies simultaneously;
  Clinical improvement (seizures, rash, acidosis) typically within 24–48 hours of biotin initiation;
  If started at birth (NBS detection via C5-OH): prevents neurological damage;
  If started late: neurological damage may be irreversible.
  Biotin is LEVEL A, highest priority, first treatment in ANY suspected MCD.

KEY VARIANTS (clinically established):
  p.Arg508Trp: most common European/North American; mild-moderate; biotin 10–20 mg/day responsive
  p.Leu216Arg: Japanese founder variant (~40% of Japanese HLCS alleles); classic neonatal severe;
               higher biotin dose (20–40 mg/day) required for full correction
  p.Val550Met: intermediate severity; biotin 10–20 mg/day responsive; European
  p.Cys129Ser: near biotransfer active site; severe; higher biotin dose needed
  p.Asn543Tyr: catalytic domain; severe neonatal; null-like for ligation activity
  c.1519C>T: splice donor site mutation; in-frame skip or null; classic neonatal

CLINICAL PRESENTATIONS:
  1. Classic Neonatal (55%): onset day 1–10; metabolic acidosis, ketosis, hyperammonemia;
     skin rash (periorificial dermatitis), alopecia, seizures (myoclonic/tonic), coma; lethal if untreated
  2. Early Infantile (30%): onset 1–3 months; skin + hair findings + developmental delay;
     seizures (infantile spasms / myoclonic); lactic acidosis
  3. Juvenile/Atypical (10%): later onset 3–24 months; variable; may present with skin only or seizures only
  4. Mild/Partial (5%): biochemically mild; biotin-responsive; sometimes detected only on NBS

EPILEPSY SPECIFICS:
  Seizure types: neonatal myoclonic (most common), tonic, multifocal clonic, infantile spasms (West syndrome),
                burst-suppression pattern on neonatal EEG
  Mechanism: metabolic acidosis + hyperammonemia + energy failure (PC block → impaired glucose/TCA) +
             biotin-dependent GABA metabolism disruption → neuronal hyperexcitability
  EEG: burst-suppression (neonatal), hypsarrhythmia (infantile spasms), multifocal spikes
  Seizures respond dramatically to biotin (not just AEDs alone)

TREATMENT EVIDENCE:
  Biotin 10–20 mg/day PO: LEVEL A (corrects all 4 carboxylases; dramatic clinical response)
  Acute crisis: IV glucose + correct acidosis + biotin IV 10–20 mg (Level A; urgent)
  Carnitine supplementation: Level B (secondary depletion from acylcarnitine accumulation)
  Protein restriction: mild leucine + isoleucine/valine/methionine restriction during acute (Level B)
  LEV first-line AED (Level B; biotin correction reduces seizure burden but AED needed short-term)
  VPA: CAUTION (not absolute CI; carnitine depletion risk + can worsen organic acidaemia)
  Biotin lifelong: yes — HLCS enzyme absent permanently; biotin supplementation required lifelong

CONTRAINDICATIONS / HAZARDS:
  Delayed biotin initiation: EXTREME HAZARD — every day without biotin = more brain injury
  Fasting: HAZARD — PC block → severe hypoglycemia + lactic crisis
  Avidin (raw egg white): ABSOLUTE CONTRAINDICATION — binds dietary biotin → blocks absorption
  VPA: CAUTION (not CI like MSUD)
"""

import random

random.seed(53)


# ─────────────────────────────────────────────────────────────────────────────
#  PATIENT COHORT — 40 synthetic HLCS-deficient patients
# ─────────────────────────────────────────────────────────────────────────────

PHENOTYPES = (
    ["Classic Neonatal MCD"] * 22 +
    ["Early Infantile MCD"] * 12 +
    ["Juvenile/Atypical MCD"] * 4 +
    ["Mild/Partial MCD"] * 2
)

SEXES = ["M", "F"]
VARIANTS = [
    "p.Leu216Arg/p.Leu216Arg",
    "p.Leu216Arg/p.Arg508Trp",
    "p.Arg508Trp/p.Arg508Trp",
    "p.Arg508Trp/p.Val550Met",
    "p.Cys129Ser/p.Arg508Trp",
    "p.Asn543Tyr/p.Leu216Arg",
    "p.Val550Met/p.Val550Met",
    "p.Arg508Trp/c.1519C>T",
    "p.Cys129Ser/c.1519C>T",
    "p.Asn543Tyr/p.Arg508Trp",
]


def _make_patient(idx, phenotype):
    sex = random.choice(SEXES)
    neonatal = phenotype == "Classic Neonatal MCD"
    early_inf = phenotype == "Early Infantile MCD"
    juvenile = phenotype == "Juvenile/Atypical MCD"
    mild = phenotype == "Mild/Partial MCD"

    if neonatal:
        onset = random.randint(0, 1)
        lactate = round(random.uniform(4.5, 14.0), 1)
        ammonia = random.randint(90, 380)
        c5oh_nm = random.randint(3200, 9800)
        c3_nm = random.randint(8, 28)
        rash = random.random() < 0.78
        alopecia = random.random() < 0.65
        seizures = True
        biotin_dose_mg = random.choice([10, 20, 40])
        biotin_response = random.random() < 0.92
        delayed_dx = random.random() < 0.30
        nbs_detected = random.random() < 0.55
        neuro_sequelae = random.random() < 0.45 if not nbs_detected else random.random() < 0.12
    elif early_inf:
        onset = random.randint(1, 4)
        lactate = round(random.uniform(2.8, 7.5), 1)
        ammonia = random.randint(55, 180)
        c5oh_nm = random.randint(1800, 6200)
        c3_nm = random.randint(5, 18)
        rash = random.random() < 0.72
        alopecia = random.random() < 0.60
        seizures = random.random() < 0.82
        biotin_dose_mg = random.choice([10, 20])
        biotin_response = random.random() < 0.95
        delayed_dx = random.random() < 0.20
        nbs_detected = random.random() < 0.60
        neuro_sequelae = random.random() < 0.28 if not nbs_detected else random.random() < 0.08
    elif juvenile:
        onset = random.randint(3, 24)
        lactate = round(random.uniform(1.8, 4.5), 1)
        ammonia = random.randint(30, 90)
        c5oh_nm = random.randint(800, 3500)
        c3_nm = random.randint(3, 12)
        rash = random.random() < 0.60
        alopecia = random.random() < 0.50
        seizures = random.random() < 0.55
        biotin_dose_mg = random.choice([10, 20])
        biotin_response = random.random() < 0.96
        delayed_dx = random.random() < 0.35
        nbs_detected = random.random() < 0.50
        neuro_sequelae = random.random() < 0.15
    else:  # mild
        onset = random.randint(6, 60)
        lactate = round(random.uniform(1.2, 2.8), 1)
        ammonia = random.randint(20, 55)
        c5oh_nm = random.randint(300, 1200)
        c3_nm = random.randint(2, 7)
        rash = random.random() < 0.40
        alopecia = random.random() < 0.30
        seizures = random.random() < 0.25
        biotin_dose_mg = 10
        biotin_response = random.random() < 0.98
        delayed_dx = random.random() < 0.50
        nbs_detected = random.random() < 0.45
        neuro_sequelae = random.random() < 0.05

    carnitine_def = random.random() < 0.48
    variant = random.choice(VARIANTS)
    biotinidase_normal = True  # always normal in HLCS (vs LOW in BTD)

    return {
        "id": f"HLCS-{idx+1:03d}",
        "sex": sex,
        "phenotype": phenotype,
        "onset_age_months": onset,
        "lactate_mmol": lactate,
        "ammonia_umol": ammonia,
        "c5oh_acylcarnitine_nmol": c5oh_nm,
        "c3_propionylcarnitine_nmol": c3_nm,
        "skin_rash_periorificial": rash,
        "alopecia": alopecia,
        "seizures_present": seizures,
        "biotin_dose_mg": biotin_dose_mg,
        "biotin_responsive": biotin_response,
        "nbs_detected": nbs_detected,
        "delayed_diagnosis": delayed_dx,
        "neuro_sequelae": neuro_sequelae,
        "carnitine_deficient": carnitine_def,
        "biotinidase_activity_normal": biotinidase_normal,
        "variant": variant,
    }


_PATIENTS = [_make_patient(i, PHENOTYPES[i]) for i in range(40)]


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_overview():
    """Return gene metadata, KPIs, phenotype distribution, four-carboxylase pathway,
    biotin ligation mechanism, and high-risk situations for the 40-patient HLCS cohort.

    Emphasises HLCS as the biotin ligase that activates ALL FOUR biotin-dependent
    carboxylases simultaneously; dramatic biotin responsiveness; neonatal onset;
    biotinidase activity NORMAL (key differential from BTD deficiency).
    """
    pts = _PATIENTS
    n = len(pts)
    rash_pct = round(sum(1 for p in pts if p["skin_rash_periorificial"]) / n * 100, 1)
    alopecia_pct = round(sum(1 for p in pts if p["alopecia"]) / n * 100, 1)
    seizure_pct = round(sum(1 for p in pts if p["seizures_present"]) / n * 100, 1)
    biotin_resp_pct = round(sum(1 for p in pts if p["biotin_responsive"]) / n * 100, 1)
    nbs_pct = round(sum(1 for p in pts if p["nbs_detected"]) / n * 100, 1)
    neuro_pct = round(sum(1 for p in pts if p["neuro_sequelae"]) / n * 100, 1)
    carnitine_pct = round(sum(1 for p in pts if p["carnitine_deficient"]) / n * 100, 1)
    male_pct = round(sum(1 for p in pts if p["sex"] == "M") / n * 100, 1)
    avg_lactate = round(sum(p["lactate_mmol"] for p in pts) / n, 1)
    avg_ammonia = round(sum(p["ammonia_umol"] for p in pts) / n, 0)
    delayed_pct = round(sum(1 for p in pts if p["delayed_diagnosis"]) / n * 100, 1)

    return {
        "gene": "HLCS",
        "protein": "Holocarboxylase Synthetase (Biotin-[propionyl-CoA-carboxylase] Ligase)",
        "aliases": "HCS; MCD; MCCD1; biotin-[acetyl-CoA-carboxylase] ligase; holocarboxylase synthetase",
        "locus": "21q22.13",
        "aa_length": "726 (full-length isoform); 517 (short isoform; long isoform biologically active)",
        "cofactor": "Biotin (vitamin H / B7) — HLCS uses ATP to form biotinyl-AMP intermediate, then transfers biotin to specific lysine residues on all 4 apocarboxylases; HLCS does NOT carry biotin itself",
        "mechanism": "HLCS is the master biotin ligase that biotinylates ALL FOUR biotin-dependent carboxylases (PC, PCC, MCC, ACC). HLCS LOF → all four apocarboxylases remain inactive → simultaneous PC+PCC+MCC+ACC blocks → lactic acidosis (PC) + organic aciduria (PCC+MCC) + hyperammonemia (PC→OAA depletion→urea cycle) + seizures → Multiple Carboxylase Deficiency (MCD), neonatal onset",
        "omim_gene": "*609018",
        "omim_disease": "#253270 (MULTIPLE CARBOXYLASE DEFICIENCY, NEONATAL / EARLY-ONSET MCD)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "pathognomonic_pattern": "C5-OH (3-methylcrotonylcarnitine) elevated in NBS acylcarnitines + urine 3-hydroxyisovaleric acid + methylcitric acid + skin rash (periorificial) + alopecia + biotinidase activity NORMAL",
        "key_distinguishing_feature": "Biotinidase activity NORMAL in HLCS (biotin ligation defect) vs LOW in BTD (biotin recycling defect); HLCS neonatal onset vs BTD usually >6 months; biotin treatment dramatically effective in BOTH",
        "structural_brain_hallmark": "Diffuse cortical atrophy and white matter changes in untreated/late-diagnosed; basal ganglia signal changes; MRI normalises with biotin treatment if started early",
        "kpis": {
            "cohort_n": n,
            "avg_lactate_mmol": avg_lactate,
            "avg_ammonia_umol": int(avg_ammonia),
            "skin_rash_periorificial_pct": rash_pct,
            "alopecia_pct": alopecia_pct,
            "seizures_pct": seizure_pct,
            "biotin_responsive_pct": biotin_resp_pct,
            "nbs_detected_pct": nbs_pct,
            "neuro_sequelae_pct": neuro_pct,
            "carnitine_deficient_pct": carnitine_pct,
            "delayed_diagnosis_pct": delayed_pct,
            "male_pct": male_pct,
        },
        "phenotype_distribution": [
            {"label": "Classic Neonatal MCD", "pct": 55, "color": "#b71c1c"},
            {"label": "Early Infantile MCD", "pct": 30, "color": "#e65100"},
            {"label": "Juvenile/Atypical MCD", "pct": 10, "color": "#f9a825"},
            {"label": "Mild/Partial MCD", "pct": 5, "color": "#0277bd"},
        ],
        "four_carboxylases_blocked": [
            {
                "enzyme": "Pyruvate Carboxylase (PC — gene PC)",
                "reaction": "Pyruvate + CO₂ + ATP → Oxaloacetate + ADP + Pi  [gluconeogenesis entry + TCA anaplerosis]",
                "block_consequence": "BLOCKED IN HLCS DEFICIENCY: OAA not formed → gluconeogenesis BLOCKED → lactic acidosis + hypoglycemia; TCA anaplerosis impaired → energy failure; OAA depletion → N-acetylglutamate synthesis drops → carbamoyl-phosphate synthetase 1 (CPS1) activity reduced → HYPERAMMONEMIA",
                "biomarker": "Lactic acidosis (L:P ratio elevated); hypoglycemia; hyperammonemia"
            },
            {
                "enzyme": "Propionyl-CoA Carboxylase (PCC — PCCA + PCCB subunits)",
                "reaction": "Propionyl-CoA + CO₂ + ATP → (S)-Methylmalonyl-CoA + ADP + Pi  [odd-chain FA / Ile/Val/Met catabolism]",
                "block_consequence": "BLOCKED IN HLCS DEFICIENCY: propionyl-CoA accumulates → propionylglycine + 3-hydroxypropionate + methylcitrate accumulate in urine; propionylcarnitine (C3) elevated in NBS acylcarnitines; carnitine depleted by propionylcarnitine esterification",
                "biomarker": "Urine: propionylglycine, 3-hydroxypropionate, methylcitric acid; NBS: C3 (propionylcarnitine) elevated"
            },
            {
                "enzyme": "3-Methylcrotonyl-CoA Carboxylase (MCC — MCCC1 + MCCC2 subunits)",
                "reaction": "3-Methylcrotonyl-CoA + CO₂ + ATP → 3-Methylglutaconyl-CoA + ADP + Pi  [leucine catabolism step 4]",
                "block_consequence": "BLOCKED IN HLCS DEFICIENCY: 3-methylcrotonyl-CoA accumulates → 3-hydroxyisovaleric acid + 3-methylcrotonylglycine accumulate in urine; 3-methylcrotonylcarnitine (C5-OH) elevated in NBS; THIS IS THE MOST SENSITIVE NBS MARKER for MCD",
                "biomarker": "Urine: 3-hydroxyisovaleric acid (HIGHEST CONCENTRATION), 3-methylcrotonylglycine; NBS: C5-OH (3-methylcrotonylcarnitine) — most sensitive NBS screen hit for HLCS"
            },
            {
                "enzyme": "Acetyl-CoA Carboxylase (ACC — ACACA cytoplasmic + ACACB mitochondrial)",
                "reaction": "Acetyl-CoA + CO₂ + ATP → Malonyl-CoA + ADP + Pi  [fatty acid synthesis rate-limiting step]",
                "block_consequence": "BLOCKED IN HLCS DEFICIENCY: malonyl-CoA not formed → de novo fatty acid synthesis impaired → skin barrier lipid deficiency → periorificial skin rash (erythematous, scaly, eczema-like) + seborrheic-like dermatitis; hair follicle lipid deficiency → ALOPECIA; these cutaneous features are highly characteristic of MCD",
                "biomarker": "Clinical: periorificial dermatitis (rash around eyes, nose, mouth, buttocks) + alopecia (hair loss including eyebrows/lashes)"
            },
        ],
        "hlcs_vs_btd_comparison": {
            "shared": "Both HLCS and BTD deficiency cause Multiple Carboxylase Deficiency (MCD); both respond dramatically to biotin supplementation; both show elevated C5-OH in NBS + organic acids in urine",
            "hlcs": {
                "onset": "Neonatal (day 1–10 classic) or early infantile (1–4 months)",
                "defect": "BIOTIN LIGATION: HLCS cannot attach biotin to any apocarboxylase",
                "biotinidase_activity": "NORMAL (biotinidase enzyme is intact — HLCS is the ligase, not the recycler)",
                "biotin_level": "Often NORMAL plasma biotin (ligation defect, not biotin deficiency per se)",
                "biotin_dose": "10–40 mg/day (higher dose for severe variants)",
            },
            "btd": {
                "onset": "Usually >6 months (late-onset) or early infantile; rarely neonatal",
                "defect": "BIOTIN RECYCLING: BTD cannot cleave biotin from biocytin/biotinyl-peptides → biotin depletion",
                "biotinidase_activity": "LOW (<30% of normal) — DIAGNOSTIC; <10% = profound deficiency",
                "biotin_level": "LOW plasma biotin (true biotin deficiency from recycling failure)",
                "biotin_dose": "5–20 mg/day (lower dose usually sufficient as recycling failure compensated by supplement)",
            },
            "diagnostic_test": "Serum biotinidase activity: NORMAL in HLCS → suspect HLCS; LOW in BTD → BTD confirmed. Gene panel (HLCS vs BTD) for definitive diagnosis.",
        },
        "high_risk_situations": [
            {"situation": "Delayed biotin initiation", "risk": "EXTREME HAZARD",
             "detail": "Every day without biotin in neonatal MCD → progressive brain injury from metabolic acidosis + hyperammonemia + energy failure. Start biotin immediately on clinical suspicion — do NOT wait for confirmatory gene panel. NBS C5-OH positive → biotin same day."},
            {"situation": "Raw egg white ingestion (avidin)", "risk": "ABSOLUTE CI in patients on biotin",
             "detail": "Raw egg white contains avidin — a glycoprotein that binds biotin with extremely high affinity (Kd ~10⁻¹⁵ M) and completely blocks dietary biotin absorption; can precipitate acute MCD crisis even in treated patients. Cooked egg white denatures avidin → safe."},
            {"situation": "Fasting / prolonged NPO", "risk": "HAZARD",
             "detail": "PC block → gluconeogenesis impaired → rapid hypoglycemia + lactic acidosis during fasting; IV glucose mandatory during any fasting period (surgery, illness). Unlike MSUD, fasting hazard is primarily from hypoglycemia/acidosis, not BCAA surge."},
            {"situation": "VPA (valproate)", "risk": "CAUTION (not absolute CI)",
             "detail": "VPA depletes carnitine (secondary) and can worsen organic acidemia acidosis; NOT the triple contraindication of MSUD/VPA, but CAUTION warranted; monitor carnitine and organic acids if VPA used for refractory seizures; LEV preferred."},
        ],
    }


def get_breakdown():
    """Return biomarkers, key variants, patient cohort, seizure types, metabolic triggers,
    and treatments for HLCS deficiency (Multiple Carboxylase Deficiency — neonatal form).

    Biomarkers emphasise the FOUR-CARBOXYLASE block pattern (C5-OH, methylcitrate,
    3-OH-isovalerate, 3-methylcrotonylglycine, propionylglycine) plus NORMAL biotinidase
    activity (key differential from BTD). Biotin treatment response is dramatic.
    """
    biomarkers = [
        {"name": "C5-OH Acylcarnitine (3-methylcrotonylcarnitine, nmol/L)", "normal": "<0.5 µmol/L", "hlcs_range": "1.8–9.8 µmol/L (high in classic neonatal)",
         "significance": "MOST SENSITIVE NBS marker for HLCS/MCD; from MCC block (leucine catabolism); C5-OH elevated = MCC impaired; triggers NBS recall + metabolic work-up; also elevated in isolated MCCC1/MCCC2 deficiency, 3-methylglutaconic aciduria, 3-OH-3-MGA — urine OA confirms MCD pattern"},
        {"name": "Urine 3-Hydroxyisovaleric Acid (µmol/mmolCr)", "normal": "<30", "hlcs_range": "400–3500 (high in classic neonatal)",
         "significance": "Highest-concentration urine metabolite in HLCS; from MCC block (3-methylcrotonyl-CoA → 3-hydroxyisovaleric acid via alternative thioesterase); along with 3-methylcrotonylglycine confirms MCC impairment; responds rapidly to biotin"},
        {"name": "Urine 3-Methylcrotonylglycine (µmol/mmolCr)", "normal": "ABSENT", "hlcs_range": "50–800 (MCC block marker)",
         "significance": "Formed by glycine conjugation of 3-methylcrotonyl-CoA (alternative pathway when MCC blocked); specific MCC impairment marker; also elevated in isolated MCCC1/MCCC2 deficiency; combined with C5-OH and 3-OH-isovalerate confirms MCC block"},
        {"name": "Urine Methylcitric Acid (µmol/mmolCr)", "normal": "<5", "hlcs_range": "50–600 (PCC block)",
         "significance": "Citrate synthase condenses propionyl-CoA (accumulated from PCC block) with OAA → methylcitric acid; SPECIFIC marker of propionyl-CoA accumulation (PCC block); elevated in HLCS, propionic acidemia (PA), methylmalonic acidemia (MMA) — confirms PCC impairment in MCD"},
        {"name": "Urine 3-Hydroxypropionate (µmol/mmolCr)", "normal": "<15", "hlcs_range": "30–400 (PCC block)",
         "significance": "From propionyl-CoA oxidation when PCC blocked; confirms propionyl-CoA pathway impairment; elevated in HLCS + PA/MMA (but in HLCS always co-elevated with MCC metabolites — key combined pattern)"},
        {"name": "Lactate (mmol/L)", "normal": "<2.2", "hlcs_range": "3.5–14.0 (PC block; severe in neonatal)",
         "significance": "From PC block (pyruvate cannot enter TCA via OAA → lactate rises); also secondary energy failure; L:P ratio elevated (>25 in severe cases) but not as dramatically as Complex I deficiency; responds to biotin + glucose"},
        {"name": "Ammonia (µmol/L)", "normal": "<50 (neonatal <110)", "hlcs_range": "55–380 (PC block → OAA depletion → urea cycle impairment)",
         "significance": "PC block → OAA depletion → N-acetylglutamate (NAG) synthesis impaired → CPS1 activity reduced → hyperammonemia; usually moderate (50–200 µmol/L); distinguishes MCD from pure organic acidemia; responds to biotin + IV glucose"},
        {"name": "Biotinidase Activity (nmol/min/mL)", "normal": ">10 (>70% of normal)", "hlcs_range": "NORMAL — typically 10–20 nmol/min/mL",
         "significance": "KEY DIAGNOSTIC MARKER: biotinidase activity is NORMAL in HLCS deficiency (BTD enzyme is intact; only HLCS/biotin ligase is defective). If biotinidase activity is LOW (<30% of normal) → BTD deficiency confirmed; NORMAL biotinidase in context of MCD biochemistry → HLCS deficiency. ALWAYS check biotinidase when MCD suspected."},
        {"name": "Plasma Biotin (nmol/L)", "normal": "0.5–2.0", "hlcs_range": "Often normal (0.4–1.8 nmol/L); occasionally mildly low",
         "significance": "Plasma biotin often NORMAL in HLCS (biotin is present but cannot be ligated); contrast with BTD where biotin is LOW (recycling failure → depletion). However, biotin supplementation is still essential — the defective HLCS is overcome by providing high-dose biotin (substrate excess forces partial ligation)"},
        {"name": "Blood Glucose (mmol/L)", "normal": "3.5–5.5", "hlcs_range": "Hypoglycemia 1.5–3.0 in acute neonatal crisis",
         "significance": "PC block → gluconeogenesis impaired → hypoglycemia; worsened by fasting; IV glucose mandatory in acute crisis; NBS detection prevents hypoglycemia by enabling pre-symptomatic treatment"},
    ]

    key_variants = [
        {"variant": "p.Leu216Arg", "effect": "Leucine-to-arginine in N-terminal domain; Japanese founder variant (~40% of Japanese HLCS alleles); severely impairs biotin binding and ATP utilisation; reduced K_m for biotin AND reduced V_max; higher biotin dose required (20–40 mg/day)", "phenotype": "Classic Neonatal MCD (Japanese) — most common severe variant in East Asia"},
        {"variant": "p.Arg508Trp", "effect": "Arginine-to-tryptophan in C-terminal catalytic domain; most common European/North American variant; reduces biotin affinity (K_m shift) but retains some ligation activity; responsive to 10–20 mg/day biotin", "phenotype": "Mild-to-moderate; often detected on NBS; best outcomes with early treatment"},
        {"variant": "p.Val550Met", "effect": "Valine-to-methionine in catalytic domain; reduces enzyme stability and biotin-AMP intermediate formation; intermediate severity; responsive to 10–20 mg/day biotin", "phenotype": "Intermediate; European; some residual ligation activity"},
        {"variant": "p.Cys129Ser", "effect": "Cysteine-to-serine at biotransfer active site; Cys129 may participate in biotin intermediate stabilisation; severely impairs ligation; null-like activity for most substrates; higher biotin dose (20–40 mg/day) required", "phenotype": "Severe neonatal; higher biotin requirement; good response if dose adequate"},
        {"variant": "p.Asn543Tyr", "effect": "Asparagine-to-tyrosine in catalytic domain; disrupts biotin-AMP intermediate formation; near-null for catalytic activity; steric clash with biotin binding pocket; early lethal if untreated", "phenotype": "Severe classic neonatal; responds to biotin but requires higher dose + early initiation"},
        {"variant": "c.1519C>T", "effect": "Splice donor site mutation; causes in-frame exon skip or premature stop; null allele; no functional HLCS protein; depends on trans allele severity for phenotype", "phenotype": "Classic neonatal if compound with severe allele; NBS detection critical for outcome"},
    ]

    patient_sample = _PATIENTS[:10]

    seizure_types = [
        {"type": "Neonatal myoclonic seizures (metabolic)", "pct": 72},
        {"type": "Tonic seizures (metabolic encephalopathy)", "pct": 58},
        {"type": "Multifocal clonic seizures", "pct": 48},
        {"type": "Infantile spasms / West syndrome (early infantile)", "pct": 38},
        {"type": "Focal seizures", "pct": 28},
        {"type": "Status epilepticus (acute crisis)", "pct": 22},
        {"type": "Burst-suppression pattern (neonatal EEG)", "pct": 65},
    ]

    trigger_types = [
        {"trigger": "Delayed biotin initiation / missed NBS diagnosis", "pct": 88},
        {"trigger": "Febrile illness / infection (metabolic stress)", "pct": 75},
        {"trigger": "Fasting / prolonged NPO (PC block → hypoglycemia)", "pct": 70},
        {"trigger": "Raw egg white ingestion (avidin → biotin blockade)", "pct": 45},
        {"trigger": "Surgical stress / catabolism without IV glucose", "pct": 40},
        {"trigger": "Anticonvulsant use depleting carnitine (VPA)", "pct": 25},
        {"trigger": "Growth spurts / puberty (increased metabolic demand)", "pct": 20},
    ]

    treatments = [
        {"drug": "Biotin supplementation 10–20 mg/day PO (Level A — FIRST LINE, URGENT)", "level": "A", "response_pct": 93,
         "color": "#004d40",
         "note": "MOST IMPORTANT TREATMENT — corrects all four carboxylase deficiencies simultaneously by providing substrate excess that partially overcomes HLCS kinetic defect. Dramatic response: metabolic acidosis corrects within 24–48 h; seizures reduce within 48–72 h; skin rash and alopecia improve within 1–2 weeks; MRI lesions can partially reverse if started early. Standard dose 10–20 mg/day; severe variants (p.Leu216Arg, p.Cys129Ser) may need 40 mg/day. Lifelong; never stop. NEONATAL PRESENTATION: start biotin immediately on clinical suspicion — do NOT wait for gene panel."},
        {"drug": "IV Biotin + IV Glucose (Level A — acute neonatal crisis)", "level": "A", "response_pct": 90,
         "color": "#0277bd",
         "note": "Acute neonatal crisis (metabolic acidosis + hyperammonemia + hypoglycemia + seizures): (1) IV glucose GIR 6–10 mg/kg/min (correct hypoglycemia + suppress catabolism); (2) biotin IV 10–20 mg immediately; (3) bicarbonate for severe acidosis (pH <7.1); (4) correct hyperammonemia (protein restriction + glucose anabolism); (5) carnitine IV if levels low. Most patients improve dramatically within 24 hours of IV biotin + glucose."},
        {"drug": "Carnitine supplementation (Level B)", "level": "B", "response_pct": 65,
         "color": "#4a148c",
         "note": "Secondary carnitine depletion from propionylcarnitine (C3) + 3-methylcrotonylcarnitine (C5-OH) esterification; free carnitine often low; supplement 50–100 mg/kg/day; monitor free carnitine and acylcarnitine profile; essential in acute crisis to prevent carnitine-depleted secondary crisis; biotin treatment reduces acylcarnitine accumulation over time"},
        {"drug": "Protein restriction — mild leucine/odd-chain AA (Level B — acute phase)", "level": "B", "response_pct": 70,
         "color": "#1b5e20",
         "note": "Moderate restriction during acute crisis: reduce leucine (MCC block → 3-methylcrotonyl-CoA accumulation) and isoleucine/valine/methionine/odd-chain FA (PCC block → propionyl-CoA accumulation); NOT as strict as MSUD (BCAA catabolism not as severely toxic); usually temporary during acute crisis; normal protein diet tolerated once metabolic control achieved with biotin; synthetic formula not usually required long-term"},
        {"drug": "LEV (Levetiracetam) — first-line AED (Level B)", "level": "B", "response_pct": 62,
         "color": "#1565c0",
         "note": "First-line AED for seizure control while biotin takes effect; no carnitine depletion; no organic acid pathway interactions; safe metabolic profile; seizures resolve with biotin correction of metabolic crisis — AED may be weaned once metabolic control established; needed short-term in most patients"},
    ]

    high_risk_drugs = [
        {"drug": "Raw egg white (avidin)", "risk": "ABSOLUTE CI in treated MCD patients",
         "mechanism": "Avidin (raw egg white glycoprotein) binds biotin with Kd ~10⁻¹⁵ M — effectively irreversible; completely prevents dietary biotin absorption; can precipitate acute MCD crisis even in well-controlled HLCS patients; cooked egg white denatures avidin — SAFE. All MCD patients must avoid raw egg white permanently."},
        {"drug": "Fasting / prolonged NPO without glucose cover", "risk": "HAZARD",
         "mechanism": "PC block → gluconeogenesis impaired → rapid hypoglycemia + lactic acidosis; PC block also impairs TCA anaplerosis → energy failure accelerates; IV glucose at GIR 6–10 mg/kg/min mandatory during any fasting (pre-surgical, illness, vomiting). Unlike MSUD, hypoglycemia is the primary acute danger (not BCAA surge)."},
        {"drug": "VPA (valproate)", "risk": "CAUTION (not absolute CI unlike MSUD)",
         "mechanism": "VPA depletes carnitine → secondary carnitine deficiency can worsen acylcarnitine-mediated organic acidemia burden; VPA may also worsen acidosis in metabolic decompensation; not the triple contraindication of MSUD/VPA; use with monitoring of carnitine + organic acids if needed; LEV preferred first-line AED."},
        {"drug": "Delayed biotin initiation", "risk": "EXTREME HAZARD",
         "mechanism": "Every day without biotin in symptomatic HLCS = continued metabolic acidosis + energy failure + hyperammonemia + neuronal injury; irreversible brain damage can occur within days in untreated classic neonatal MCD. Start biotin on CLINICAL SUSPICION — NBS C5-OH elevation = biotin same day; do not wait for full metabolic workup or gene panel before initiating biotin."},
    ]

    return {
        "biomarkers": biomarkers,
        "key_variants": key_variants,
        "patients_sample": patient_sample,
        "seizure_types": seizure_types,
        "trigger_types": trigger_types,
        "treatments": treatments,
        "high_risk_drugs": high_risk_drugs,
    }


def get_definitions():
    """Return gene card, key concepts, diagnostic thresholds, and differential diagnosis
    for HLCS deficiency (Multiple Carboxylase Deficiency — neonatal/early-onset form).

    Emphasises: HLCS as biotin ligase for all four carboxylases; biotinidase NORMAL;
    dramatic biotin responsiveness; HLCS vs BTD differential; NBS via C5-OH;
    neonatal onset; four-carboxylase simultaneous block pattern.
    """
    return {
        "gene_card": {
            "Gene": "HLCS (Holocarboxylase Synthetase)",
            "Also known as": "HCS; MCD; MCCD1; biotin-[propionyl-CoA-carboxylase (ATP-hydrolysing)] ligase; biotin-[acetyl-CoA-carboxylase] ligase",
            "Subunit": "Monomeric enzyme; ~726 aa full-length isoform; localised in both mitochondria and cytoplasm; bifunctional — biotinylates mitochondrial (PC, PCC, MCC) and cytoplasmic (ACC) carboxylases",
            "Chromosome": "21q22.13",
            "Protein length": "~726 aa full-length (biologically active); ~517 aa short isoform",
            "Cofactor": "Uses biotin (vitamin H/B7) as substrate + ATP (forms biotinyl-AMP intermediate) → transfers biotin covalently to specific lysine residues on all four apocarboxylases (PC, PCC PCCA subunit, MCC MCCC1 subunit, ACC ACACA subunit)",
            "Function": "Master biotin ligase: biotinylates ALL FOUR biotin-dependent carboxylases. HLCS LOF → none of the four carboxylases can be activated → simultaneous PC+PCC+MCC+ACC deficiency → multiple metabolic blocks → Multiple Carboxylase Deficiency (MCD)",
            "Complex partners": "Acts on: PC (pyruvate carboxylase) + PCCA (propionyl-CoA carboxylase α) + MCCC1 (3-methylcrotonyl-CoA carboxylase α) + ACACA/ACACB (acetyl-CoA carboxylase)",
            "Inheritance": "Autosomal Recessive (AR) — biallelic LOF",
            "OMIM gene": "*609018",
            "OMIM disease": "#253270 (MULTIPLE CARBOXYLASE DEFICIENCY, NEONATAL / EARLY-ONSET MCD)",
            "Prevalence": "~1:87,000 Japan (highest worldwide); ~1:200,000+ globally; most common in Japan due to p.Leu216Arg founder",
            "Biochemical block": "Simultaneous block: PC (lactic acidosis + hyperammonemia) + PCC (methylcitrate + propionylglycine + C3-carnitine) + MCC (3-OH-isovalerate + 3-methylcrotonylglycine + C5-OH-carnitine) + ACC (skin/hair lipid deficiency → rash + alopecia)",
            "Pathognomonic pattern": "C5-OH on NBS + 3-OH-isovalerate + methylcitrate + lactic acidosis + skin rash/alopecia + NORMAL biotinidase activity",
            "Primary toxic species": "Organic acid accumulation from PCC+MCC blocks; lactic acidosis from PC block; hyperammonemia from PC/OAA depletion",
            "First-line treatment": "Biotin supplementation 10–20 mg/day PO (Level A) — IMMEDIATE, before confirmatory testing",
            "Absolute CI": "Raw egg white (avidin — blocks biotin absorption)",
        },
        "key_concepts": [
            {"term": "HLCS as the master biotin ligase — why ALL FOUR carboxylases fail simultaneously",
             "definition": "Biotin is a vitamin cofactor that must be covalently attached to the ε-amino group of a specific lysine residue in each biotin-dependent carboxylase. This biotinylation reaction is catalysed exclusively by HLCS. HLCS mechanism: (1) HLCS + biotin + ATP → HLCS-biotinyl-AMP complex (biotinyl-adenylate intermediate) + PPi; (2) HLCS-biotinyl-AMP + apocarboxylase-Lys → holocarboxylase (biotinylated, active) + AMP. HLCS LOF = all four apocarboxylases remain as inactive apo-forms. Since PC, PCC, MCC, and ACC each perform distinct, non-redundant metabolic steps, their simultaneous failure creates four independent metabolic crises at once — lactic acidosis (PC), organic aciduria (PCC+MCC), fatty acid synthesis failure (ACC). No other enzyme can substitute for HLCS."},
            {"term": "HLCS vs BTD — two causes of Multiple Carboxylase Deficiency",
             "definition": "Both HLCS and Biotinidase (BTD) deficiency cause MCD, but through opposite mechanisms. HLCS: biotin is present and recycled normally, but cannot be LIGATED (attached) to apocarboxylases — biotinylation step blocked. BTD: biotin is ligated normally initially, but RECYCLED biotin (from protein degradation releasing biocytin/biotinyl-peptides) cannot be cleaved by biotinidase → free biotin not regenerated → progressive biotin depletion → all four carboxylases gradually lose their biotin cofactor. KEY DISTINCTION: (1) HLCS = neonatal/early-infantile onset; BTD = usually >6 months (late-onset) or early infantile; (2) HLCS = biotinidase activity NORMAL; BTD = biotinidase activity <30% of normal — DIAGNOSTIC; (3) HLCS = plasma biotin often NORMAL; BTD = plasma biotin LOW. Both treated with biotin, but HLCS often requires higher doses (10–40 mg/day) vs BTD (5–20 mg/day)."},
            {"term": "Why biotin treatment works in HLCS deficiency despite the enzyme defect",
             "definition": "HLCS with LOF pathogenic variants typically shows a kinetic defect: either severely reduced affinity for biotin (elevated K_m) or reduced catalytic efficiency (reduced V_max), or both. The K_m for biotin in wild-type HLCS is ~0.3–1 µmol/L. In p.Arg508Trp (common mild variant), K_m rises to ~3–10 µmol/L; in p.Leu216Arg (severe Japanese founder), K_m rises to ~20–50 µmol/L. Plasma biotin in humans is ~0.5–2 nmol/L (i.e., 0.5–2 nmol/L = far below normal K_m and far further below mutant K_m). Providing 10–40 mg/day oral biotin raises plasma biotin to ~50–500 nmol/L — up to 100× the normal level — which partially overcomes the kinetic defect by mass action, enabling residual HLCS enzyme to biotinylate the apocarboxylases at a sufficient rate. Null alleles with no enzyme activity cannot be rescued by biotin alone at any dose. Complete null genotypes benefit from the combined treatment but may have residual metabolic impairment."},
            {"term": "NBS detection via C5-OH (3-methylcrotonylcarnitine) — importance of early biotin",
             "definition": "3-Methylcrotonylcarnitine (C5-OH) elevation in newborn blood spot MS/MS is the MOST SENSITIVE newborn screening marker for HLCS/MCD. C5-OH is elevated because MCC block causes 3-methylcrotonyl-CoA accumulation → esterification with carnitine. C5-OH is also elevated in isolated MCCC1/MCCC2 deficiency and 3-methylcrotonyl-CoA carboxylase deficiency (usually benign maternal) — differentiation requires urine organic acids (MCD shows methylcitrate + 3-OH-isovalerate + 3-methylcrotonylglycine + lactic acid together, vs isolated MCC showing only 3-OH-isovalerate + 3-methylcrotonylglycine). IMPORTANCE: NBS-detected HLCS infants who receive biotin within the first week of life have dramatically better outcomes — seizures, metabolic acidosis, and neurological damage are entirely preventable with pre-symptomatic treatment. Late-detected HLCS (classic neonatal) already in encephalopathy on day 3–7 often has permanent neurological injury even with prompt biotin initiation."},
            {"term": "Four biomarker blocks — how to recognise HLCS on urine organic acids",
             "definition": "HLCS deficiency creates a FOUR-STREAM organic acid pattern that is pathognomonic in combination: (1) MCC block markers — 3-hydroxyisovaleric acid (highest, often >500 µmol/mmolCr), 3-methylcrotonylglycine; (2) PCC block markers — methylcitric acid (>50 µmol/mmolCr), 3-hydroxypropionate, propionylglycine; (3) PC block marker — lactic acid (elevated); (4) ACC block marker — not detectable by OA (clinical: skin rash + alopecia). No other organic acidaemia produces ALL of these simultaneously. Isolated propionic acidemia (PA) shows PCC-block metabolites ONLY; isolated MCCC1/MCCC2 shows MCC-block metabolites ONLY; isolated PC deficiency shows lactic acidosis ONLY. The combination of ≥2 blocks on OA = MCD = check biotinidase (HLCS vs BTD) + start biotin."},
            {"term": "Seizures in HLCS deficiency — mechanism and biotin response",
             "definition": "Seizures in HLCS deficiency arise from multiple mechanisms: (1) Metabolic acidosis (lactic acid + organic acids) impairs neuronal membrane stability; (2) Hypoglycemia (PC block → gluconeogenesis impaired) → neuronal energy failure → excitotoxicity; (3) Hyperammonemia (PC→OAA depletion→CPS1 impairment) → glutamate accumulation → NMDAR activation; (4) Biotin deficiency affects GABA synthesis — GABA transaminase and glutamate decarboxylase (GAD) are biotin-dependent or biotin-responsive enzymes in some pathways. Neonatal EEG shows burst-suppression pattern — similar to other neonatal metabolic encephalopathies. Biotin treatment corrects all these mechanisms simultaneously → seizures usually cease within 48–72 hours. Short-term LEV useful while biotin takes effect. Key teaching: in any neonatal seizure with metabolic acidosis + hyperammonemia + skin rash, start BIOTIN immediately."},
        ],
        "diagnostic_thresholds": {
            "c5oh_nbs_recall_threshold": ">0.5 µmol/L (C5-OH/3-methylcrotonylcarnitine in DBS MS/MS) → recall for metabolic work-up",
            "c5oh_hlcs_range": "1.8–9.8 µmol/L in classic neonatal HLCS; 0.8–6.2 µmol/L early infantile",
            "urine_3oh_isovalerate_high": ">100 µmol/mmolCr (classic); typically 400–3500 in neonatal crisis; most sensitive urine OA marker",
            "urine_methylcitrate_high": ">20 µmol/mmolCr (confirms PCC block component of MCD)",
            "lactate_hlcs_range": "3.5–14.0 mmol/L in classic neonatal; 1.8–4.5 in early infantile; responds rapidly to biotin",
            "ammonia_hlcs_range": "55–380 µmol/L (neonatal); 30–90 µmol/L (early infantile); responds to biotin + glucose",
            "biotinidase_activity_hlcs": "NORMAL (>10 nmol/min/mL; >70% of normal); LOW activity (<30%) → BTD deficiency",
            "biotin_dose_standard": "10–20 mg/day PO (standard for p.Arg508Trp, p.Val550Met); 20–40 mg/day for severe/null variants",
            "biotin_response_timeline": "Metabolic acidosis: 24–48 h; Seizures: 48–72 h; Skin rash: 7–14 days; Alopecia: 2–8 weeks; MRI: months (partial reversal if started early)",
            "biotinidase_btd_diagnostic": "<30% of normal biotinidase activity = BTD deficiency; <10% = profound BTD; contrast with HLCS (NORMAL biotinidase)",
            "hypoglycemia_threshold": "Blood glucose <2.6 mmol/L = emergency IV glucose; PC block makes prolonged fasting extremely hazardous",
        },
        "differential_diagnosis": [
            {"disease": "BTD deficiency (Biotinidase Deficiency)",
             "distinguishing_features": "SAME urine OA pattern (MCC+PCC+PC+ACC blocks) and same C5-OH elevation as HLCS. KEY DIFFERENCE: biotinidase activity <30% of normal in BTD (DIAGNOSTIC); onset usually >6 months (late-onset) vs neonatal in HLCS; plasma biotin LOW in BTD vs often NORMAL in HLCS. Both respond to biotin — BTD usually 5–20 mg/day, HLCS 10–40 mg/day."},
            {"disease": "Isolated MCCC1 or MCCC2 deficiency (3-Methylcrotonyl-CoA Carboxylase Deficiency)",
             "distinguishing_features": "C5-OH elevated in NBS + urine 3-OH-isovalerate + 3-methylcrotonylglycine; ONLY the MCC-block metabolites — NO methylcitrate/propionylglycine (PCC block absent) + NO lactic acidosis (PC intact) + NO skin rash (ACC intact). Biotinidase NORMAL (same as HLCS). Often BENIGN — many are asymptomatic maternal MCCC1/MCCC2 deficiency detected via infant NBS. Gene panel differentiates MCCC1 vs MCCC2 vs HLCS."},
            {"disease": "Propionic Acidemia (PA — PCCA/PCCB deficiency)",
             "distinguishing_features": "PA: methylcitrate + 3-hydroxypropionate + propionylglycine (PCC block) + C3-carnitine elevated; NO MCC-block metabolites (3-OH-isovalerate normal); NO lactic acidosis from PC block (different mechanism); NO skin rash. C5-OH NOT elevated in PA. Biotinidase NORMAL. PCCA/PCCB gene panel distinguishes from HLCS. Does NOT respond to biotin."},
            {"disease": "Methylmalonic Acidemia (MMA — MUT, MMAA, MMAB deficiency)",
             "distinguishing_features": "MMA: methylmalonic acid elevated + C3-carnitine elevated; methylcitrate elevated (same as PA/HLCS PCC block); NO MCC metabolites; NO lactic acidosis from PC block; may have cobalamin-responsive forms. C5-OH NOT a primary MMA marker. Biotinidase NORMAL. Does NOT respond to biotin."},
            {"disease": "Pyruvate Carboxylase Deficiency (PC — PC gene deficiency)",
             "distinguishing_features": "PC deficiency: lactic acidosis + hyperammonemia (same as HLCS PC-block consequences); HOWEVER: NO MCC metabolites (3-OH-isovalerate normal), NO PCC metabolites (methylcitrate normal), NO skin rash (ACC intact). Urine OA shows lactate only. Biotinidase NORMAL. Does NOT respond to biotin (enzyme itself absent, not just un-biotinylated). Plasma amino acids may show citrulline elevation (urea cycle)."},
            {"disease": "3-Methylglutaconic Aciduria (MLCCD1, SERAC1, TAZ)",
             "distinguishing_features": "3-methylglutaconic acid elevated (downstream of MCC block in some forms); C5-OH occasionally mildly elevated. However: methylcitrate NORMAL (PCC intact); lactate NORMAL (PC intact); biotinidase NORMAL; skin rash/alopecia absent. Does NOT respond to biotin."},
        ],
    }
