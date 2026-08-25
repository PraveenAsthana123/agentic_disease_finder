#!/usr/bin/env python3
"""NAGS (N-Acetylglutamate Synthase) Deficiency Epilepsy Dashboard.

NAGS encodes N-Acetylglutamate Synthase, a mitochondrial matrix enzyme:
  L-Glutamate + Acetyl-CoA  →  N-Acetylglutamate (NAG) + CoA-SH
  NAG is the OBLIGATE allosteric activator of CPS1 (urea cycle step 1)

NAGS DISEASE: N-Acetylglutamate Synthase Deficiency (NAGSD)
  OMIM Disease: #237310   Gene: NAGS, OMIM *608300
  Chromosome: 17q21.31
  Inheritance: Autosomal Recessive (AR)
  Protein: 534 aa; mitochondrial matrix; GNAT (GCN5-related N-acetyltransferase) superfamily
  Prevalence: ~1:2,000,000 (ultra-rare; rarest urea cycle disorder)

MECHANISM — LOSS-OF-FUNCTION (upstream cofactor block → CPS1 INACTIVE → urea cycle arrested):
  Normal NAGS: Glutamate + Acetyl-CoA → NAG + CoA-SH  [NAG is obligate CPS1 activator]
  NAGS LOF: NAG CANNOT be produced → CPS1 COMPLETELY INACTIVE despite intact CPS1 enzyme
  Consequence: Same as CPS1 deficiency — urea cycle blocked at step 1 (most proximal)
  Citrulline: NOT synthesised → CRITICALLY LOW (same as CPS1/OTC)
  NO carbamoyl-P overflow: NAGS LOF → CPS1 inactive → no carbamoyl-P made → no orotic acid
    → Urine orotic acid NORMAL (SAME as CPS1; KEY DISTINCTION from OTC where orotic HIGH)
  Glutamate: ELEVATED (substrate of NAGS cannot be converted → accumulates)
  Arginine: allosteric activator of NAGS; LOW downstream (not synthesised)

POSITION IN UREA CYCLE — NAGS AS UPSTREAM COFACTOR REGULATOR:
  NAGS: Glutamate + Acetyl-CoA → NAG + CoA  [COFACTOR GENERATOR — upstream of CPS1]
  Step 1: NH₃ + CO₂ + 2ATP → [CPS1, requires NAG] → Carbamoyl-P      (entry)
  Step 2: Carbamoyl-P + Ornithine → [OTC] → Citrulline + Pi
  Step 3: Citrulline + Aspartate → [ASS1] → Argininosuccinate
  Step 4: Argininosuccinate → [ASL] → Arginine + Fumarate
  Step 5: Arginine → [ARG1] → Ornithine + Urea

  ARGININE POSITIVE FEEDBACK ON NAGS:
    Arginine allosterically activates NAGS (binds NAGS active site) → more NAG → CPS1 more active
    Arginine supplementation → increases NAGS activity (if residual NAGS protein present)

NCG (N-CARBAMYLGLUTAMATE) — THE SPECIFIC NAGS THERAPY:
  N-carbamylglutamate (NCG / Carbaglu) is a structural analogue of NAG
  NCG activates CPS1 DIRECTLY without requiring NAG from NAGS
  In NAGS deficiency: NCG COMPLETELY bypasses the NAGS block → full CPS1 activation
  NAGS is the ONLY UCD where a single oral drug can COMPLETELY normalise ammonia LONG-TERM
  NCG response rate: ~65-70% of NAGS patients — complete normalisation
  Complete NCG response DISTINGUISHES NAGS from CPS1 (partial/no response in CPS1)
  Dose: 100-250 mg/kg/day; oral; safe for chronic use; may avoid liver transplant

NAGS BIOCHEMISTRY (LOF → NAG absent → CPS1 inactive → same as CPS1):
  Plasma ammonia:        CRITICALLY ELEVATED (>500 µmol/L neonatal; >200 late-onset crisis) — PATHOGNOMONIC
  Plasma citrulline:     CRITICALLY LOW (<5 µmol/L; normal 15-35 µmol/L) — same as CPS1 and OTC
  Urine orotic acid:     NORMAL (<6 µmol/mol Cr) — KEY NEGATIVE (SAME as CPS1, UNLIKE OTC)
  Plasma arginine:       LOW (<25 µmol/L; normal 60-120 µmol/L) — downstream product not synthesised
  Plasma glutamate:      ELEVATED (>150 µmol/L; normal 50-100 µmol/L) — NAGS substrate accumulates
  NAG (mitochondrial):   ABSENT / VERY LOW — DIRECT MARKER of NAGS deficiency
  Plasma glutamine:      ELEVATED (600-900 µmol/L) — GS detoxifies ammonia
  PLP (plasma):          NORMAL — KEY NEGATIVE vs PNPO/ALDH4A1
  alpha-AASA (urine):    NORMAL — KEY NEGATIVE vs ALDH7A1/PDE
  Pipecolic acid:        NORMAL — KEY NEGATIVE vs PDE and peroxisomal disorders
  tHcy:                  NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR
  MMA:                   NORMAL — KEY NEGATIVE vs methylmalonic acidemia
  GABA (CSF):            NORMAL — KEY NEGATIVE vs ABAT (HIGH) and GAD1 (LOW)
  GHB (urine):           NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1)
  Glycine:               NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH)
  GAA:                   NORMAL — KEY NEGATIVE vs GAMT

VPA AND NAGS — MOST DIRECT MECHANISM IN ALL UCD:
  VPA (valproate) is a COMPETITIVE INHIBITOR of NAGS
  VPA directly occupies the NAGS active site → NAGS cannot bind glutamate/acetyl-CoA
  Result: NO NAG produced → CPS1 COMPLETELY INACTIVE → catastrophic hyperammonemia
  This is the MOST DIRECT VPA mechanism in any UCD:
    In OTC: VPA → CPS1 inhibited secondarily (via NAGS block)
    In CPS1: VPA → NAGS blocked → no NAG → CPS1 OFF
    In NAGS: VPA → NAGS directly blocked → no NAG → CPS1 OFF
  All UCD have ABSOLUTE CI for VPA; NAGS has the most direct mechanism of all

EPILEPSY IN NAGS DEFICIENCY (secondary to hyperammonemia):
  Overall seizure rate: ~55-65% (ammonia impairs GABA-A receptor + NMDA overactivation)
  Seizure types:
    GTCS: ~40% (modal in acute hyperammonemic crisis)
    Myoclonic encephalopathy: ~20% (high-amplitude, multi-focal; ammonia-driven)
    Focal seizures: ~18% (focal cortical dysfunction; frontal predominant)
    Absence-like (metabolic): ~12% (metabolic encephalopathy correlate)
    Status epilepticus: ~25% (neurological emergency; neonatal presentation)
  EEG: burst-suppression (neonatal severe), triphasic waves (ammonia encephalopathy),
       diffuse slowing (interictal), rhythmic bifrontal delta
  Drug-resistant epilepsy: 20-30% (correlates with hyperammonemic crisis severity)
  MRI acute: cytotoxic/vasogenic oedema; diffuse cortical/subcortical change
  MRI chronic: cortical atrophy, gliosis (after repeated crises)

NON-SEIZURE NEUROLOGIC FEATURES:
  Intellectual disability: 55-70% (proportional to peak ammonia and crisis duration)
  Cerebral oedema (acute): 40-55% neonatal-onset
  Behavioural/psychiatric: 15-25% (ADHD, executive dysfunction — late-onset)
  Protein aversion: 50-60% late-onset patients

NON-NEUROLOGIC FEATURES:
  Hepatomegaly: 20-30% (metabolic stress; milder than CPS1)
  Elevated transaminases: 15-25% (metabolic crisis-related)
  Growth retardation: 25-40% (protein restriction + metabolic instability)

PHENOTYPE CLASSES:
  NCG-Responsive (65%): catalytic/binding domain variants with residual NAGS; NCG → complete NH₃ normalisation
    Key distinguishing feature vs CPS1: NAGS = COMPLETE NCG response; CPS1 = partial or none
    Many avoid liver transplant; may present in adulthood; protein aversion key clue
  Neonatal-Onset (20%): null/near-null NAGS; no residual enzyme; ammonia >500 day 1-3
    Similar severity to CPS1 neonatal; CRRT/HD mandatory; early liver transplant if not NCG-responsive
  Late-Onset Episodic (15%): residual 5-20% NAGS; triggered by illness/protein/fasting
    Variable onset late infancy to adulthood; episodic crises; NCG trial mandatory

TREATMENTS (ACUTE CRISIS):
  IV dextrose 10%: Level A — nitrogen-free energy; suppresses catabolism; 10-12 mg/kg/min GIR
  Sodium benzoate + Sodium phenylacetate (BUPHENYL/RAVICTI): Level A — nitrogen scavengers
  N-carbamylglutamate (NCG/Carbaglu): Level A PRIMARY — TRIAL IN ALL; SPECIFIC for NAGS
    Dose 100-250 mg/kg/day; complete NH₃ normalisation in ~65-70% patients
    The DISTINGUISHING therapeutic response that identifies NAGS vs CPS1
  Arginine (IV then oral): Level A — becomes conditionally essential; 150-250 mg/kg/day
    Also allosterically activates NAGS (if residual protein) — NAGS-specific dual benefit
  Citrulline (oral/IV): Level A — replenishes downstream urea cycle products
  Haemodialysis / CRRT: Level A — for ammonia >500 µmol/L; neonates CRRT preferred

TREATMENTS (CHRONIC):
  N-carbamylglutamate (NCG/Carbaglu): Level A PRIMARY — SPECIFIC THERAPY FOR NAGS
    First-line chronic therapy if NCG-responsive (65-70%); avoids liver transplant in responders
    NAGS is the ONLY UCD where a single drug can provide complete long-term ammonia control
  Glycerol phenylbutyrate (RAVICTI): Level A — nitrogen scavenger; oral liquid
  Protein restriction: Level A — 0.5-1.5 g/kg/day natural protein + essential AA supplement
  Arginine supplementation (oral, chronic): Level B — allosteric NAGS activator; replenishes downstream
  Liver transplantation: Level A CURATIVE (if not NCG-responsive) — restores hepatic NAGS
    Note: many NCG-responsive patients avoid transplant with NCG therapy
  Levetiracetam (LEV): Level B — first-line AED; SV2A mechanism; no hepatotoxicity

ABSOLUTE CONTRAINDICATIONS:
  Valproate/VPA: ABSOLUTE CI — DIRECTLY inhibits NAGS (competitive inhibitor of NAGS active site).
    This is the MOST DIRECT VPA mechanism in ANY UCD (VPA is a substrate analogue for NAGS).
    NAGS = direct target of VPA; CPS1 and OTC are secondary targets.
    Even single dose → catastrophic hyperammonemia. Multiple fatalities documented.
    CI extends to ALL urea cycle disorders.
  High protein diet: ABSOLUTE CI — nitrogen overload → ammonia surge; NAGS cannot produce NAG
  L-Asparaginase: ABSOLUTE CI — hyperammonemia + UCD block = catastrophic
  Prolonged fasting: HIGH RISK — protein catabolism → nitrogen load → ammonia surge
  Systemic glucocorticoids (high-dose): HIGH RISK — catabolic; protein breakdown → NH₃ surge
  Valproyl-CoA-producing prodrugs (valpromide, valproyl-glycine): ABSOLUTE CI — same mechanism

DIFFERENTIAL DIAGNOSIS:
  CPS1 deficiency: Biochemically IDENTICAL (both: ammonia HIGH, citrulline LOW, orotic NORMAL)
    NCG trial DISTINGUISHES: COMPLETE response = NAGS; partial/no response = CPS1
    Gene panel mandatory (NAGS + CPS1 sequenced together)
  OTC deficiency: Orotic acid HIGH in OTC (KEY POSITIVE) vs NORMAL in NAGS/CPS1 — easy distinction
    OTC is X-linked; NAGS is AR; OTC 100× more common than NAGS
  ASS1 deficiency (Citrullinemia I): Citrulline VERY HIGH (>100) vs LOW in NAGS — OPPOSITE direction
  GLUD1 GoF (HHS): ALSO hypoglycemia + hyperinsulinism; citrulline NORMAL; NH₃ 100-500 not >1000
  HHH syndrome (SLC25A15): Homocitrullinuria pathognomonic; ornithine VERY HIGH; orotic HIGH (like OTC)
  OAT deficiency: Ammonia NORMAL — KEY NEGATIVE vs NAGS (CRITICALLY HIGH); ornithine VERY HIGH

VARIANTS (NAGS gene — catalytic domain, glutamate-binding, acetyl-CoA binding, structural):
  p.Arg518His: ~20%; catalytic domain; moderate-severe; NCG-responsive (binds acetyl-CoA)
  p.Arg101Cys: ~17%; N-terminal mitochondrial region; severe neonatal; not NCG-responsive
  p.Gly518Ser: ~12%; catalytic domain; moderate-severe; NCG-responsive
  p.Arg226Gln: ~10%; glutamate-binding; moderate; NCG-responsive (substrate affinity reduced)
  c.IVS3+1G>A: ~9%; splice null; exon skip → NMD; severe; not NCG-responsive; pan-ethnic
  p.Lys405Glu: ~8%; acetyl-CoA binding; moderate; NCG-responsive
  p.Tyr521Cys: ~7%; GNAT domain core; moderate-severe; variably NCG-responsive
  p.Val321Gly: ~5%; structural stability; mild; NCG-responsive; late-onset

NEONATAL SCREENING:
  Standard NBS: citrulline LOW flag (same as CPS1/OTC; all <5 µmol/L) — CANNOT distinguish NAGS from CPS1/OTC by NBS
  Urine orotic acid: NORMAL in NAGS (like CPS1; unlike OTC where HIGH) — post-NBS differentiator for OTC vs NAGS/CPS1
  Confirmatory: plasma amino acids + urine orotic acid + gene panel (NAGS + CPS1 + OTC sequenced together)
  NCG trial: mandatory in all newly diagnosed; complete dramatic NH₃ response = NAGS; partial = CPS1
"""
import random

_N    = 40    # cohort size (consistent with all expert dashboards)
_SEED = 205   # deterministic seed (CPS1=199, next +6=205)


def _rng():
    return random.Random(_SEED)


# ──────────────────────────────────────────────────────────────────────────────
def get_overview():
    rng = _rng()

    # Phenotype distribution
    n_ncg      = round(_N * 0.65)   # NCG-Responsive: hallmark of NAGS; complete NH₃ normalisation
    n_neonatal = round(_N * 0.20)   # Neonatal-Onset: null/near-null; severe; day 1-3
    n_late     = _N - n_ncg - n_neonatal  # Late-Onset Episodic: residual NAGS; triggered crises

    phenotypes = {
        "NCG-Responsive":        {"n": n_ncg,      "pct": round(100 * n_ncg / _N)},
        "Neonatal-Onset (Null)": {"n": n_neonatal, "pct": round(100 * n_neonatal / _N)},
        "Late-Onset Episodic":   {"n": n_late,     "pct": round(100 * n_late / _N)},
    }

    # Biomarker distributions (NAGS LOF → No NAG → CPS1 inactive → same as CPS1)
    amm_ncg      = [rng.uniform(100, 500) for _ in range(n_ncg)]      # NCG-responsive: variable onset
    amm_neonatal = [rng.uniform(500, 2000) for _ in range(n_neonatal)] # Neonatal: critically high
    amm_late     = [rng.uniform(150, 600) for _ in range(n_late)]      # Late-onset: episodic
    all_amm = amm_ncg + amm_neonatal + amm_late
    avg_amm = round(sum(all_amm) / _N)

    # Citrulline (critically low — no carbamoyl-P to feed OTC step 2)
    cit_all = [rng.uniform(0.3, 4.8) for _ in range(_N)]
    avg_cit = round(sum(cit_all) / _N, 1)

    # Urine orotic acid — NORMAL in NAGS (same as CPS1; no carbamoyl-P overflow)
    orot_all = [rng.uniform(0.4, 5.2) for _ in range(_N)]  # normal <6 µmol/mol Cr
    avg_orot = round(sum(orot_all) / _N, 1)

    # Arginine (low — downstream product absent + allosteric activator of NAGS)
    arg_all = [rng.uniform(4, 20) for _ in range(_N)]
    avg_arg = round(sum(arg_all) / _N, 1)

    # Glutamate (elevated — NAGS substrate accumulates; NAGS-specific biomarker)
    glu_all = [rng.uniform(120, 280) for _ in range(_N)]   # normal 50-100 µmol/L
    avg_glu = round(sum(glu_all) / _N)

    # Clinical outcome flags
    n_seizures         = round(_N * 0.60)
    n_dre              = round(_N * 0.25)
    n_idd              = round(_N * 0.63)
    n_se               = round(_N * 0.25)
    n_cerebral_oedema  = round(_N * 0.48)
    n_protein_aversion = round(_N * 0.55)
    n_ncg_responders   = n_ncg  # 65% — the defining feature

    return {
        "subtitle": (
            "NAGS Deficiency — Autosomal recessive urea cycle cofactor disorder (rarest UCD, ~1:2,000,000). "
            "NAGS catalyses Glutamate + Acetyl-CoA → NAG (N-acetylglutamate), the OBLIGATE allosteric activator of CPS1. "
            "LOF: No NAG → CPS1 COMPLETELY INACTIVE despite intact CPS1 enzyme → total urea cycle arrest. "
            "Biochemically IDENTICAL to CPS1: ammonia HIGH, citrulline LOW, orotic acid NORMAL. "
            "HALLMARK: NCG (N-carbamylglutamate) COMPLETELY normalises ammonia in ~65% patients — the ONLY UCD treatable by a single oral drug. "
            "VPA ABSOLUTE CI — directly inhibits NAGS at its active site (most direct VPA mechanism in any UCD)."
        ),
        "gene": "NAGS",
        "chromosome": "17q21.31",
        "protein_size": "534 aa; mitochondrial matrix; GNAT superfamily acetyltransferase",
        "omim_gene": "608300",
        "omim_disease": "237310",
        "inheritance": "Autosomal Recessive (AR) — both alleles LOF required",
        "cohort_n": _N,
        "seed": _SEED,
        "phenotype_distribution": phenotypes,
        "kpi": {
            "avg_plasma_ammonia_umol_l":       avg_amm,
            "avg_plasma_citrulline_umol_l":    avg_cit,
            "avg_urine_orotic_acid_umol_mol":  avg_orot,
            "avg_plasma_arginine_umol_l":      avg_arg,
            "avg_plasma_glutamate_umol_l":     avg_glu,
            "pct_seizures":                    round(100 * n_seizures / _N),
            "pct_dre":                         round(100 * n_dre / _N),
            "pct_idd":                         round(100 * n_idd / _N),
            "pct_status_epilepticus":          round(100 * n_se / _N),
            "pct_cerebral_oedema":             round(100 * n_cerebral_oedema / _N),
            "pct_protein_aversion":            round(100 * n_protein_aversion / _N),
            "pct_ncg_responsive":              round(100 * n_ncg_responders / _N),
        },
        "biomarker_normals": {
            "description": "KEY NEGATIVE biomarkers — NORMAL in NAGS (distinguishes from other diseases)",
            "orotic_acid_normal":   "NORMAL (<6 µmol/mol Cr) — NO carbamoyl-P made → no overflow → same as CPS1 (UNLIKE OTC)",
            "plp_normal":           "NORMAL — NAGS NOT PLP-dependent (KEY NEG vs PNPO/ALDH4A1)",
            "alpha_aasa_normal":    "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE",
            "pipecolic_normal":     "NORMAL — KEY NEGATIVE vs PDE/peroxisomal disorders",
            "thcy_normal":          "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR",
            "mma_normal":           "NORMAL — KEY NEGATIVE vs methylmalonic acidemia",
            "gaba_normal":          "NORMAL — KEY NEGATIVE vs ABAT (HIGH) / GAD1 (LOW)",
            "ghb_normal":           "NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB HIGH",
            "glycine_normal":       "NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH)",
            "gaa_normal":           "NORMAL — KEY NEGATIVE vs GAMT",
            "creatine_normal":      "NORMAL — KEY NEGATIVE vs GAMT/AGAT/SLC6A8",
        },
        "key_positives": {
            "ammonia_critically_high":         "Plasma ammonia >200 µmol/L crisis (>500 neonatal; normal <50) — PATHOGNOMONIC",
            "citrulline_critically_low":       "Plasma citrulline <5 µmol/L (normal 15-35) — same as CPS1 and OTC; no carbamoyl-P fed to OTC",
            "orotic_acid_NORMAL_same_as_CPS1": (
                "Urine orotic acid NORMAL (<6 µmol/mol Cr) — CRITICAL. "
                "No NAG → CPS1 inactive → no carbamoyl-P made → no pyrimidine overflow → no orotic acid. "
                "Distinguishes NAGS+CPS1 from OTC (where orotic acid is HIGH)."
            ),
            "glutamate_elevated_NAGS_specific": (
                "Plasma glutamate ELEVATED (>150 µmol/L; normal 50-100) — NAGS substrate cannot be converted → accumulates. "
                "NAGS-specific finding that helps distinguish from CPS1 (where glutamate elevation is secondary/indirect)."
            ),
            "ncg_COMPLETE_response_DIAGNOSTIC": (
                "NCG (N-carbamylglutamate) trial: COMPLETE NH₃ normalisation in 65% of NAGS patients. "
                "Complete response = NAGS deficiency. Partial/no response = CPS1 deficiency. "
                "This is the SINGLE therapeutic test that distinguishes NAGS from CPS1. "
                "Gene panel mandatory to confirm (NAGS + CPS1 sequenced together)."
            ),
        },
        "pathway_context": {
            "role": "NAGS is the UPSTREAM COFACTOR GENERATOR — produces NAG, the obligate allosteric activator of CPS1",
            "reaction": "Glutamate + Acetyl-CoA → N-Acetylglutamate (NAG) + CoA-SH  [GNAT superfamily]",
            "block_consequence": "No NAG → CPS1 CANNOT be activated → carbamoyl-P not made → citrulline not made → urea cycle arrested",
            "arginine_feedback": "Arginine allosterically activates NAGS (binds NAGS) → more NAG → more CPS1 activity (positive feedback in urea cycle)",
            "ncg_bypass": "NCG (N-carbamylglutamate) directly activates CPS1 → bypasses NAGS block completely → restores urea cycle",
            "vpa_mechanism": "VPA competitively inhibits NAGS active site → no NAG → CPS1 OFF → most direct VPA mechanism in any UCD",
            "vs_cps1": "NAGS LOF = same phenotype as CPS1 LOF but NCG COMPLETELY normalises (vs partial in CPS1); gene panel required",
            "vs_otc": "Both have NH₃ HIGH, citrulline LOW; OTC has orotic acid HIGH (NAGS/CPS1 have orotic NORMAL) — easy distinction",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_breakdown():
    rng = _rng()

    n_ncg      = round(_N * 0.65)
    n_neonatal = round(_N * 0.20)
    n_late     = _N - n_ncg - n_neonatal

    # Seizure type breakdown
    n_seizures = round(_N * 0.60)
    seizure_types = {
        "GTCS (acute hyperammonemic crisis)": {"pct": 40, "n": round(n_seizures * 0.40)},
        "Myoclonic encephalopathy":           {"pct": 20, "n": round(n_seizures * 0.20)},
        "Focal seizures":                     {"pct": 18, "n": round(n_seizures * 0.18)},
        "Absence-like (metabolic)":           {"pct": 12, "n": round(n_seizures * 0.12)},
        "Status epilepticus":                 {"pct": 25, "n": round(n_seizures * 0.25)},
    }

    # Crisis triggers
    triggers = {
        "Febrile illness / infection":       {"pct": 75, "detail": "Protein catabolism → endogenous nitrogen surge; most common precipitant"},
        "High protein meal":                 {"pct": 68, "detail": "Exogenous nitrogen exceeds urea cycle capacity (zero in NAGS LOF)"},
        "Prolonged fasting":                 {"pct": 62, "detail": "Muscle catabolism → endogenous nitrogen; common neonatal and late-onset trigger"},
        "Valproate exposure":                {"pct": 48, "detail": "DIRECTLY inhibits NAGS active site → no NAG → CPS1 completely off → crisis"},
        "Post-surgical catabolism":          {"pct": 45, "detail": "Catabolic stress → protein breakdown → ammonia surge; high risk period"},
        "Corticosteroid therapy":            {"pct": 35, "detail": "Catabolic hormone → protein breakdown → NH₃ surge in NAGS patients"},
        "Post-partum (females, AR)":         {"pct": 18, "detail": "Catabolic state; AR inheritance → females equally at risk (unlike OTC X-linked)"},
        "Arginine depletion (illness/diet)": {"pct": 30, "detail": "Arginine activates NAGS; depletion reduces NAGS activity further → crisis risk in partial NAGS"},
    }

    # Treatment efficacy
    treatments = {
        "N-carbamylglutamate (NCG / Carbaglu)": {
            "level": "A", "category": "SPECIFIC NAGS therapy (NAG analogue → activates CPS1 directly)", "efficacy_pct": 97,
            "mechanism": (
                "NCG directly activates CPS1 without requiring NAG from NAGS. "
                "COMPLETE NH₃ normalisation in ~65-70% NAGS patients (vs partial/no response in CPS1). "
                "NAGS is the ONLY UCD where a single oral drug can fully normalise ammonia long-term. "
                "100-250 mg/kg/day. Mandatory trial in ALL newly diagnosed patients — safe even in non-responders."
            ),
        },
        "Sodium benzoate + Sodium phenylacetate (RAVICTI)": {
            "level": "A", "category": "Nitrogen scavenger", "efficacy_pct": 83,
            "mechanism": "Benzoate → hippurate (removes glycine-N); phenylacetate → PAG (removes glutamine-N). 2N per cycle.",
        },
        "Arginine IV (acute) + oral (chronic)": {
            "level": "A", "category": "Conditionally essential AA + allosteric NAGS activator", "efficacy_pct": 82,
            "mechanism": (
                "Becomes conditionally essential in NAGS (downstream product absent). "
                "NAGS-specific dual benefit: arginine also allosterically activates NAGS (if residual protein). "
                "150-250 mg/kg/day oral chronic; IV in acute crisis."
            ),
        },
        "Haemodialysis / CRRT (acute crisis)": {
            "level": "A", "category": "Ammonia removal (physical)", "efficacy_pct": 92,
            "mechanism": "For ammonia >500 µmol/L; CRRT preferred in neonates; HD more efficient per hour in older patients.",
        },
        "Liver transplantation (if not NCG-responsive)": {
            "level": "A", "category": "Curative (restores hepatic NAGS)", "efficacy_pct": 97,
            "mechanism": (
                "Restores hepatic NAGS → NAG produced → CPS1 activated → full urea cycle function. "
                "Note: ~65% of NAGS patients may avoid transplant if NCG-responsive. "
                "AR inheritance: living related donor possible. Does NOT reverse pre-existing neurological damage."
            ),
        },
        "IV dextrose 10% (GIR 10-12 mg/kg/min)": {
            "level": "A", "category": "Anti-catabolic (acute)", "efficacy_pct": 75,
            "mechanism": "Nitrogen-free energy; suppresses endogenous protein catabolism; first step in acute protocol.",
        },
        "Citrulline supplementation": {
            "level": "A", "category": "Urea cycle replenishment", "efficacy_pct": 76,
            "mechanism": "Metabolised downstream of NAGS/CPS1 block; replenishes arginine and citrulline; 150-250 mg/kg/day.",
        },
        "Arginine supplementation (chronic, low-dose)": {
            "level": "B", "category": "Allosteric NAGS activator (NAGS-specific)", "efficacy_pct": 60,
            "mechanism": "Arginine allosterically activates NAGS enzyme; increases NAG production in partial NAGS; NAGS-specific benefit.",
        },
        "Levetiracetam (LEV)": {
            "level": "B", "category": "AED — first-line in NAGS", "efficacy_pct": 68,
            "mechanism": "SV2A mechanism; no hepatotoxicity; no effect on ammonia; safe in all UCD. Seizures secondary to ammonia.",
        },
        "Protein restriction (natural)": {
            "level": "A", "category": "Chronic dietary management", "efficacy_pct": 78,
            "mechanism": "0.5-1.5 g/kg/day natural protein + essential AA supplement; limits exogenous nitrogen load.",
        },
    }

    # Drug risks
    drug_risks = {
        "Valproate / VPA": {
            "risk": "ABSOLUTE CI",
            "detail": (
                "DIRECTLY inhibits NAGS active site (VPA is a competitive inhibitor of NAGS). "
                "This is the MOST DIRECT VPA mechanism in ANY UCD — VPA directly targets NAGS. "
                "(In OTC/CPS1, VPA affects NAGS secondarily; in NAGS, VPA IS the primary target.) "
                "Result: NO NAG → CPS1 COMPLETELY INACTIVE → catastrophic hyperammonemia. "
                "Even single therapeutic dose → fatal hyperammonemia documented. "
                "CI extends to ALL urea cycle disorders."
            ),
        },
        "High protein diet": {
            "risk": "ABSOLUTE CI",
            "detail": "Nitrogen overload; impossible to metabolise without any NAG/CPS1/urea cycle function.",
        },
        "L-Asparaginase (chemotherapy)": {
            "risk": "ABSOLUTE CI",
            "detail": "Depletes asparagine → hyperammonemia; combined with NAGS = catastrophic urea cycle arrest.",
        },
        "Valpromide / valproyl-glycine (VPA prodrugs)": {
            "risk": "ABSOLUTE CI",
            "detail": "Converted to VPA in vivo → same NAGS direct inhibition mechanism → catastrophic hyperammonemia.",
        },
        "Prolonged fasting": {
            "risk": "HIGH RISK",
            "detail": "Endogenous protein catabolism → nitrogen load → ammonia surge. Emergency protocol on any fast >4-6h.",
        },
        "Systemic glucocorticoids (high-dose)": {
            "risk": "HIGH RISK",
            "detail": "Catabolic → protein breakdown → nitrogen load → ammonia surge in NAGS. Use only if unavoidable with monitoring.",
        },
        "Topiramate": {
            "risk": "MODERATE RISK",
            "detail": "Carbonic anhydrase inhibition affects CO₂ availability for CPS1 reaction. "
                      "Case reports of hyperammonemia. Monitor ammonia; never combine with VPA.",
        },
        "Acetaminophen / paracetamol (chronic high dose)": {
            "risk": "MODERATE RISK",
            "detail": "Hepatotoxicity risk; glutathione depletion. Use acetaminophen sparingly; monitor liver function.",
        },
    }

    # Variants
    variants = [
        {"name": "p.Arg518His",  "domain": "Catalytic domain (acetyl-CoA binding)",   "pct": 20, "severity": "Moderate-Severe; NCG-responsive",      "responsive": True},
        {"name": "p.Arg101Cys",  "domain": "N-terminal mitochondrial region",          "pct": 17, "severity": "Severe neonatal; null; not NCG-responsive", "responsive": False},
        {"name": "p.Gly518Ser",  "domain": "Catalytic domain",                         "pct": 12, "severity": "Moderate-Severe; NCG-responsive",      "responsive": True},
        {"name": "p.Arg226Gln",  "domain": "Glutamate-binding (substrate)",             "pct": 10, "severity": "Moderate; NCG-responsive",             "responsive": True},
        {"name": "c.IVS3+1G>A", "domain": "Splice null (exon skip → NMD)",            "pct": 9,  "severity": "Severe; pan-ethnic; not NCG-responsive","responsive": False},
        {"name": "p.Lys405Glu",  "domain": "Acetyl-CoA binding",                       "pct": 8,  "severity": "Moderate; NCG-responsive",             "responsive": True},
        {"name": "p.Tyr521Cys",  "domain": "GNAT domain core",                         "pct": 7,  "severity": "Moderate-Severe; variably responsive", "responsive": True},
        {"name": "p.Val321Gly",  "domain": "Structural stability",                     "pct": 5,  "severity": "Mild; NCG-responsive; late-onset",      "responsive": True},
    ]

    # Differentials
    differentials = [
        {
            "disease": "CPS1 deficiency",
            "key_diff": (
                "Biochemically IDENTICAL to NAGS (both: ammonia HIGH, citrulline LOW, orotic NORMAL). "
                "NCG trial distinguishes: COMPLETE NH₃ normalisation = NAGS; partial/no response = CPS1. "
                "Gene panel mandatory (NAGS + CPS1 sequenced together)."
            ),
            "distinguishing": "NCG therapeutic trial response + gene panel",
        },
        {
            "disease": "OTC deficiency",
            "key_diff": (
                "Urine orotic acid HIGH in OTC (carbamoyl-P overflows from intact CPS1) "
                "vs NORMAL in NAGS (no carbamoyl-P made since CPS1 inactive). "
                "OTC is X-linked (NAGS is AR); OTC 100× more common than NAGS."
            ),
            "distinguishing": "Urine orotic acid + inheritance pattern",
        },
        {
            "disease": "ASS1 deficiency (Citrullinemia type I)",
            "key_diff": "Citrulline VERY HIGH (>100 µmol/L in ASS1 vs <5 in NAGS) — OPPOSITE direction; no confusion.",
            "distinguishing": "Plasma citrulline direction",
        },
        {
            "disease": "GLUD1 GoF (HHS syndrome)",
            "key_diff": "GLUD1: ALSO hypoglycemia + hyperinsulinism; citrulline NORMAL; orotic NORMAL; ammonia 100-500 not >1000.",
            "distinguishing": "Glucose, insulin, citrulline (NORMAL in GLUD1)",
        },
        {
            "disease": "OAT deficiency",
            "key_diff": "OAT: ammonia NORMAL (KEY NEGATIVE — OAT ammonia is NORMAL vs NAGS CRITICALLY HIGH); ornithine VERY HIGH (400-1500).",
            "distinguishing": "Ammonia level (OAT = normal; NAGS = critically HIGH)",
        },
        {
            "disease": "HHH syndrome (SLC25A15)",
            "key_diff": "HHH: homocitrullinuria pathognomonic; ornithine VERY HIGH; orotic HIGH (like OTC, not NAGS).",
            "distinguishing": "Homocitrullinuria + orotic acid + ornithine",
        },
    ]

    return {
        "seizure_types": seizure_types,
        "triggers": triggers,
        "treatments": treatments,
        "drug_risks": drug_risks,
        "variants": variants,
        "differentials": differentials,
        "phenotype_detail": {
            "NCG-Responsive": {
                "n": n_ncg,
                "pct": round(100 * n_ncg / _N),
                "description": (
                    "Catalytic/binding domain variants with residual NAGS protein; NCG activates CPS1 directly. "
                    "COMPLETE NH₃ normalisation with NCG — the hallmark distinguishing NAGS from CPS1. "
                    "Many avoid liver transplant; some present in adulthood. Protein aversion key clinical clue. "
                    "NCG-responsive = the most important phenotype to identify; changes management entirely."
                ),
                "avg_peak_ammonia_umol_l": 260,
                "seizure_rate_pct": 42,
                "ncg_complete_response_pct": 100,
            },
            "Neonatal-Onset (Null)": {
                "n": n_neonatal,
                "pct": round(100 * n_neonatal / _N),
                "description": (
                    "Null/near-null NAGS; no residual enzyme. Day 1-3 presentation. "
                    "Ammonia often >800 µmol/L at presentation; cerebral oedema common; "
                    "CRRT mandatory; liver transplant if not NCG-responsive."
                ),
                "avg_peak_ammonia_umol_l": 850,
                "seizure_rate_pct": 78,
                "ncg_complete_response_pct": 0,
            },
            "Late-Onset Episodic": {
                "n": n_late,
                "pct": round(100 * n_late / _N),
                "description": (
                    "Residual 5-20% NAGS activity; episodic crises triggered by illness/protein/fasting. "
                    "Range late infancy to adulthood. Protein aversion strong clue. "
                    "NCG trial mandatory; may be responsive."
                ),
                "avg_peak_ammonia_umol_l": 310,
                "seizure_rate_pct": 50,
                "ncg_complete_response_pct": 60,
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "disease": "NAGS Deficiency (N-Acetylglutamate Synthase Deficiency, NAGSD)",
        "omim_disease": "237310",
        "omim_gene": "608300",
        "gene": "NAGS",
        "chromosome": "17q21.31",
        "inheritance": "Autosomal Recessive (AR); biallelic LOF required; males and females equally affected",
        "prevalence": "~1:2,000,000; ultra-rare; rarest urea cycle disorder (OTC 100× more common; CPS1 ~2× more common)",
        "enzyme": "N-Acetylglutamate Synthase (NAGS); 534 aa; GNAT superfamily acetyltransferase; mitochondrial matrix",
        "reaction": "L-Glutamate + Acetyl-CoA → N-Acetylglutamate (NAG) + CoA-SH  [obligate cofactor production for CPS1]",
        "pathway_role": (
            "NAGS is the UPSTREAM COFACTOR GENERATOR — not a urea cycle enzyme per se but its obligate regulator. "
            "NAG (N-acetylglutamate) is the MANDATORY allosteric activator of CPS1 (urea cycle step 1). "
            "No NAGS → no NAG → CPS1 inactive → urea cycle arrested at step 1 (same as CPS1 LOF)."
        ),
        "mechanism_of_disease": (
            "NAGS LOF → N-acetylglutamate (NAG) CANNOT be produced. "
            "CPS1 (urea cycle step 1) requires NAG as obligate allosteric activator — cannot function without it. "
            "Result: CPS1 COMPLETELY INACTIVE despite intact CPS1 enzyme → carbamoyl-P not made → urea cycle arrested. "
            "Identical biochemical phenotype to CPS1 LOF: ammonia HIGH, citrulline LOW, orotic NORMAL. "
            "Glutamate accumulates (NAGS substrate; uniquely elevated in NAGS vs CPS1). "
            "Arginine depletion (downstream) reduces NAGS activation further (positive feedback break)."
        ),
        "ncg_mechanism": (
            "N-carbamylglutamate (NCG / Carbaglu) is a structural analogue of NAG. "
            "NCG activates CPS1 DIRECTLY (binds the NAG-binding site on CPS1) without requiring NAGS. "
            "In NAGS deficiency: NCG COMPLETELY bypasses the NAGS block → full CPS1 activation → urea cycle restored. "
            "NAGS is the ONLY UCD where a single oral drug (NCG) can provide COMPLETE long-term ammonia control. "
            "NCG complete response (~65-70%) DISTINGUISHES NAGS from CPS1 (partial/no response in CPS1). "
            "NCG dose: 100-250 mg/kg/day; trial mandatory in ALL newly diagnosed; safe even if non-responsive."
        ),
        "arginine_nags_axis": (
            "Arginine allosterically activates NAGS (binds to NAGS regulatory site → increases NAG production). "
            "In normal physiology: arginine → NAGS more active → more NAG → CPS1 more active → more urea cycle flux (positive feedback). "
            "In NAGS LOF: arginine depletion (no arginine made downstream) removes the NAGS activator → compounded NAGS inactivity. "
            "Arginine supplementation in NAGS (if residual protein): dual benefit — replenishes essential AA + partially activates NAGS."
        ),
        "vpa_nags_mechanism": (
            "VPA (valproate) is a COMPETITIVE INHIBITOR of NAGS active site. "
            "VPA directly occupies the NAGS substrate-binding site → prevents glutamate + acetyl-CoA from binding. "
            "This is the MOST DIRECT VPA mechanism in ANY UCD: "
            "  - In NAGS: VPA directly targets NAGS (the primary enzyme affected) "
            "  - In CPS1: VPA targets NAGS secondarily (NAGS → no NAG → CPS1 off) "
            "  - In OTC: VPA targets NAGS secondarily (same chain) "
            "NAGS patients are at EXTREME risk from VPA — even the lowest dose. ABSOLUTE CI."
        ),
        "key_biomarker_positives": {
            "plasma_ammonia":    "CRITICALLY HIGH (>500 µmol/L neonatal; >200 late-onset crisis; normal <50) — PATHOGNOMONIC",
            "plasma_citrulline": "CRITICALLY LOW (<5 µmol/L; normal 15-35) — no carbamoyl-P to feed OTC; same as CPS1",
            "plasma_arginine":   "LOW (<25 µmol/L; normal 60-120) — downstream product not synthesised",
            "plasma_glutamate":  "ELEVATED (>150 µmol/L; normal 50-100) — NAGS substrate accumulates; NAGS-specific",
            "plasma_glutamine":  "ELEVATED (600-900 µmol/L) — GS detoxifies ammonia → glutamine",
        },
        "key_biomarker_negatives": {
            "urine_orotic_acid": "NORMAL (<6 µmol/mol Cr) — CRITICAL. No carbamoyl-P made (CPS1 inactive) → no pyrimidine overflow (same as CPS1, UNLIKE OTC)",
            "plp":               "NORMAL — NAGS not PLP-dependent (KEY NEG vs PNPO/ALDH4A1)",
            "alpha_aasa":        "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE",
            "pipecolic":         "NORMAL — KEY NEGATIVE vs PDE/peroxisomal disorders",
            "thcy":              "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR",
            "mma":               "NORMAL — KEY NEGATIVE vs methylmalonic acidemia",
            "gaba_csf":          "NORMAL — KEY NEGATIVE vs ABAT (DRAMATICALLY HIGH) and GAD1 (LOW)",
            "ghb_urine":         "NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB dramatically HIGH",
            "glycine":           "NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH)",
            "gaa":               "NORMAL — KEY NEGATIVE vs GAMT",
        },
        "critical_nags_vs_cps1_distinction": (
            "Biochemically IDENTICAL (both: ammonia HIGH, citrulline LOW, orotic NORMAL, arginine LOW). "
            "NCG (N-carbamylglutamate) trial is the SINGLE distinguishing test: "
            "  COMPLETE NH₃ normalisation → NAGS deficiency "
            "  Partial or no response → CPS1 deficiency "
            "Gene panel mandatory in all cases (NAGS + CPS1 sequenced simultaneously). "
            "Glutamate elevation is more pronounced in NAGS (substrate accumulation) but not pathognomonic."
        ),
        "critical_nags_vs_otc_distinction": (
            "Urine orotic acid: NORMAL in NAGS (no carbamoyl-P made → no overflow → same as CPS1) "
            "but HIGH in OTC (carbamoyl-P made at step 1 but blocked at step 2 → overflows to pyrimidines). "
            "This single test easily distinguishes NAGS+CPS1 group from OTC. "
            "Both NAGS and OTC have ammonia HIGH, citrulline LOW — orotic acid is the key differentiator."
        ),
        "absolute_contraindications": {
            "valproate_vpa": (
                "ABSOLUTE CI — directly inhibits NAGS active site (VPA is a competitive substrate-site inhibitor). "
                "This is the MOST DIRECT VPA mechanism in any UCD — NAGS is the PRIMARY VPA target. "
                "Even single dose → catastrophic hyperammonemia. Multiple fatalities. "
                "CI extends to ALL urea cycle disorders."
            ),
            "vpa_prodrugs": (
                "Valpromide, valproyl-glycine — ABSOLUTE CI. Converted to VPA in vivo → same NAGS direct inhibition."
            ),
            "high_protein_diet": "ABSOLUTE CI — nitrogen overload; CPS1 inactive due to no NAG; no urea cycle function",
            "l_asparaginase": "ABSOLUTE CI — hyperammonemia + total urea cycle arrest = catastrophic",
        },
        "first_line_treatments": {
            "ncg_carbaglu": "NCG (N-carbamylglutamate) 100-250 mg/kg/day — Level A PRIMARY; SPECIFIC NAGS therapy; complete response in 65%",
            "nitrogen_scavengers": "Sodium benzoate + phenylacetate (RAVICTI) — Level A chronic + acute",
            "liver_transplant": "Curative — Level A if not NCG-responsive; restores hepatic NAGS; AR = living donor possible",
            "acute_ammonia_removal": "CRRT/HD for ammonia >500 µmol/L — Level A; neonates: CRRT preferred",
            "aed": "Levetiracetam (LEV) — Level B first-line AED (no hepatotoxicity; seizures secondary to ammonia)",
        },
        "seizure_mechanism": (
            "Ammonia impairs GABA-A receptor gating (reduces inhibitory tone) and over-activates NMDA receptors. "
            "Net: reduced inhibition + increased excitation → seizure threshold lowered. "
            "Cerebral oedema (cytotoxic + vasogenic) → intracranial hypertension → secondary seizures. "
            "EEG: burst-suppression (severe neonatal), triphasic waves (ammonia encephalopathy), diffuse slowing."
        ),
        "ar_inheritance_note": (
            "AR inheritance: BOTH parents are obligate heterozygous carriers; recurrence risk 25% per pregnancy. "
            "Males and females equally affected — contrast with OTC (X-linked; males predominantly). "
            "Heterozygous NAGS carriers: typically asymptomatic. "
            "Cascade carrier testing essential after index diagnosis. Living related liver donor possible for transplant."
        ),
        "unique_features_vs_other_ucd": (
            "1. ONLY UCD where a single oral drug (NCG/Carbaglu) can COMPLETELY normalise ammonia long-term. "
            "2. Upstream cofactor enzyme (not a direct urea cycle enzyme) — blocks CPS1 indirectly. "
            "3. Arginine has a unique allosteric activating effect on NAGS (positive feedback). "
            "4. VPA's most direct mechanism is NAGS inhibition — NAGS is the PRIMARY VPA target in UCD. "
            "5. Glutamate elevation is NAGS-specific (substrate accumulation) — not seen in CPS1 LOF."
        ),
    }
