#!/usr/bin/env python3
"""CPS1 (Carbamoyl Phosphate Synthetase 1) Deficiency Epilepsy Dashboard.

CPS1 encodes Carbamoyl Phosphate Synthetase 1, a mitochondrial matrix enzyme:
  NH₃ + CO₂ + 2ATP  →  Carbamoyl phosphate + 2ADP + Pi  (urea cycle, step 1)
  Allosteric activator: N-acetylglutamate (NAG) produced by NAGS — MANDATORY COFACTOR

CPS1 DISEASE: Carbamoyl Phosphate Synthetase 1 Deficiency
  OMIM Disease: #237300   Gene: CPS1, OMIM *608307
  Chromosome: 2q34
  Inheritance: Autosomal Recessive (AR) — CONTRAST with OTC (X-linked)
  Protein: 1500 aa; mitochondrial matrix; largest urea cycle enzyme; single polypeptide
  Prevalence: ~1:800,000–1,300,000 (rarer than OTC at 1:14,000–50,000)

MECHANISM — LOSS-OF-FUNCTION (urea cycle block → ammonia cannot be detoxified):
  Normal CPS1: NH₃ + CO₂ + 2ATP → Carbamoyl-P  [requires NAG as obligate allosteric activator]
  CPS1 LOF: Carbamoyl phosphate CANNOT be made → urea cycle BLOCKED at step 1 (most proximal)
  Consequence: ammonia cannot be incorporated into any urea cycle intermediate
  Citrulline: NOT synthesised (OTC has no substrate) → CRITICALLY LOW
  NO carbamoyl-P overflow: pyrimidine pathway receives NO excess substrate
    → Urine orotic acid NORMAL (KEY DISTINCTION from OTC where orotic HIGH)
  Arginine: not synthesised → becomes conditionally essential → LOW
  Glutamine/Alanine: elevated as alternative NH₃ disposal routes

POSITION IN UREA CYCLE (CPS1 = step 1 of 5 steps):
  NH₃ + CO₂ + 2ATP → [CPS1, BLOCKED] → Carbamoyl-P                 (step 1 — ENTRY)
  Carbamoyl-P + Ornithine → [OTC] → Citrulline + Pi                 (step 2)
  Citrulline + Aspartate → [ASS1] → Argininosuccinate               (step 3)
  Argininosuccinate → [ASL] → Arginine + Fumarate                   (step 4)
  Arginine → [ARG1] → Ornithine + Urea                              (step 5)

  CRITICAL KEY DISTINCTION CPS1 vs OTC:
    CPS1 LOF: Carbamoyl-P NOT MADE → no overflow → orotic acid NORMAL
    OTC LOF:  Carbamoyl-P MADE but NOT USED → CP overflows → orotic acid HIGH
    BOTH have: Ammonia HIGH, Citrulline LOW — orotic acid is the SINGLE differentiating test

NAG / NAGS ALLOSTERIC AXIS (CPS1-specific clinically important):
  N-acetylglutamate synthase (NAGS) produces NAG from glutamate + acetyl-CoA
  NAG is the OBLIGATE allosteric activator of CPS1
  NAGS LOF → NAG absent → CPS1 INACTIVE despite intact CPS1 enzyme → same phenotype as CPS1 LOF
  VPA inhibits NAGS → less NAG → CPS1 less active → ABSOLUTE CI in all UCD including CPS1
  N-carbamylglutamate (NCG/Carbaglu): structural analogue of NAG → activates CPS1 directly
    NCG-responsive CPS1 variants (~15–20%): usually NAG-binding domain mutations
    NCG trial (100–250 mg/kg/day) should be PERFORMED IN ALL newly diagnosed patients

CPS1 BIOCHEMISTRY (LOF → hyperammonemia + citrulline deficiency + NORMAL orotic acid):
  Plasma ammonia:        CRITICALLY ELEVATED (>500 µmol/L neonatal; >200 late-onset crisis) — PATHOGNOMONIC
  Plasma citrulline:     CRITICALLY LOW (<5 µmol/L; normal 15–35 µmol/L) — same as OTC
  Urine orotic acid:     NORMAL (<6 µmol/mol Cr) — KEY NEGATIVE that DISTINGUISHES CPS1 FROM OTC
  Plasma arginine:       LOW (<25 µmol/L; normal 60–120 µmol/L) — downstream product absent
  Plasma ornithine:      MILDLY ELEVATED or NORMAL — substrate for OTC backs up; less than OAT (400–1500)
  Plasma glutamine:      ELEVATED (600–900 µmol/L) — GS detoxifies ammonia
  Plasma alanine:        ELEVATED (500–800 µmol/L) — ALT-mediated NH₃ disposal
  PLP (plasma):          NORMAL — KEY NEGATIVE vs PNPO/ALDH4A1 (CPS1 NOT PLP-dependent)
  alpha-AASA (urine):    NORMAL — KEY NEGATIVE vs ALDH7A1/PDE
  Pipecolic acid:        NORMAL — KEY NEGATIVE vs PDE and peroxisomal disorders
  tHcy:                  NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR homocysteine disorders
  MMA:                   NORMAL — KEY NEGATIVE vs methylmalonic acidemia
  GABA (CSF):            NORMAL — KEY NEGATIVE vs ABAT (HIGH) and GAD1 (LOW)
  GHB (urine):           NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB dramatically HIGH
  Glycine:               NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH)
  GAA:                   NORMAL — KEY NEGATIVE vs GAMT (50–300 µmol/L)
  Creatine:              NORMAL — KEY NEGATIVE vs GAMT/AGAT/SLC6A8

EPILEPSY IN CPS1 DEFICIENCY (secondary to hyperammonemia):
  Overall seizure rate: ~60–70% (ammonia directly impairs GABA-A receptor + NMDA overactivation)
  Seizure types:
    GTCS: ~45% (modal in acute hyperammonemic crisis)
    Myoclonic encephalopathy: ~25% (high-amplitude, multi-focal; ammonia-driven)
    Focal seizures: ~20% (focal cortical dysfunction; often frontal)
    Absence-like (metabolic): ~15% (metabolic encephalopathy; triphasic EEG correlate)
    Status epilepticus: ~30% (higher than OTC — step 1 block more complete; neonatal mortality high)
  EEG: burst-suppression (neonatal severe), triphasic waves (ammonia encephalopathy),
       diffuse slowing (interictal), rhythmic bifrontal delta
  Drug-resistant epilepsy: 25–35% (correlates with severity/duration of hyperammonemic crises)
  MRI acute: cytotoxic/vasogenic oedema; diffusion restriction (parieto-occipital/diffuse)
  MRI chronic: cortical atrophy, gliosis, periventricular white matter changes

NON-SEIZURE NEUROLOGIC FEATURES:
  Intellectual disability: 70–85% (proportional to peak ammonia and crisis duration)
  Cerebral oedema (acute): 55–65% of neonatal-onset; major cause of death
  Cortical atrophy (MRI): 50–65% of neonatal crisis survivors
  Behavioural/psychiatric: 20–30% (ADHD, anxiety, executive dysfunction late-onset)
  Protein aversion: 55–65% late-onset patients (intuitive dietary adaptation)

NON-NEUROLOGIC FEATURES:
  Hepatomegaly: 35–45% (liver is primary site of CPS1 expression)
  Elevated transaminases: 25–40% (metabolic stress; not primary liver disease)
  Coagulopathy (acute): 20–30% (liver synthetic dysfunction in severe crisis)
  Growth retardation: 35–45% (protein restriction + metabolic instability)
  Hyperammonemic coma: 40–50% neonatal-onset (without treatment; major mortality cause)

PHENOTYPE CLASSES:
  Neonatal-Onset (55%): null/near-null CPS1; no NAG binding; ammonia >500–2000 µmol/L day 1–3
    Often more severe than OTC neonatal (step 1 block = TOTAL urea cycle arrest)
    Emergency presentation; seizures common; coma; ~40–50% neonatal mortality without CRRT
  Late-Onset (30%): residual 5–20% CPS1 activity; triggered by illness/protein load/fasting/surgery
    Range: late infancy to adulthood; protein aversion key clinical clue; episodic crisis
  NCG-Responsive (15%): NAG-binding domain variants; partial CPS1 activation with N-carbamylglutamate
    NCG dramatically reduces ammonia (often >50% reduction); may avoid liver transplant
    Milder phenotype overall; some present in adulthood

TREATMENTS (ACUTE CRISIS):
  IV dextrose 10%: Level A — suppress catabolism; 10–12 mg/kg/min GIR; provide nitrogen-free energy
  Sodium benzoate + Sodium phenylacetate (BUPHENYL/RAVICTI): Level A — nitrogen scavengers
    Benzoate → hippurate (+ glycine N waste); phenylacetate → phenylacetylglutamine (+ glutamine N)
    Removes 2 N per cycle; critical in acute hyperammonemic management
  Arginine (IV then oral): Level A — becomes conditionally essential in CPS1; 150–250 mg/kg/day
  Citrulline (oral/IV): Level A — alternative; metabolised downstream of CPS1 block
  Haemodialysis / CRRT: Level A — for ammonia >500 µmol/L or unresponsive to medical therapy
  N-carbamylglutamate (NCG/Carbaglu): Level A — TRIAL IN ALL patients; activates residual CPS1
    100–250 mg/kg/day; dramatic response in NCG-responsive cases (15–20%); safe even in non-responders
  Stop protein intake: Level A — 24–48 h protein-free; then restart with restricted natural protein

TREATMENTS (CHRONIC):
  N-carbamylglutamate (NCG/Carbaglu): Level A PRIMARY (if responsive) — long-term ammonia control
    First-line specific therapy if NCG-responsive; avoids/delays liver transplant
  Glycerol phenylbutyrate (RAVICTI): Level A — nitrogen scavenger; oral liquid; chronic management
  Protein restriction: Level A — 0.5–1.5 g/kg/day natural protein + essential AA supplement
  Liver transplantation: Level A CURATIVE — restores hepatic CPS1; normalises urea cycle
    Optimal timing before second/third crisis; does NOT reverse pre-existing neurological damage
  Levetiracetam (LEV): Level B — first-line AED in CPS1; SV2A mechanism; no hepatotoxicity
  Arginine/Citrulline (oral, chronic): Level B — replenish downstream urea cycle products

ABSOLUTE CONTRAINDICATIONS:
  Valproate/VPA: ABSOLUTE CI — inhibits NAGS (N-acetylglutamate synthase → NAG production) +
    mitochondrial beta-oxidation + complex I. NAG is the obligate CPS1 activator.
    VPA reduces NAG → CPS1 cannot be activated → CATASTROPHIC hyperammonemia.
    Mechanism in CPS1 is MORE DIRECT than in OTC: NAGS inhibition → no NAG → CPS1 COMPLETELY INACTIVATED
    Multiple fatalities; CI extends to ALL urea cycle disorders
  High protein diet: ABSOLUTE CI — nitrogen overload → ammonia surge
  L-Asparaginase: ABSOLUTE CI — depletes asparagine → hyperammonemia; combined with CPS1 = catastrophic
  Prolonged fasting: HIGH RISK — endogenous nitrogen from protein catabolism → ammonia surge
  Systemic glucocorticoids (high-dose): HIGH RISK — catabolic → muscle protein → ammonia
  Topiramate: MODERATE RISK — case reports of hyperammonemia; mechanism: carbonic anhydrase + VPA-like
  Chloramphenicol: HIGH RISK — mitochondrial toxin; directly inhibits CPS1 synthesis (protein synthesis block)
  Acetaminophen/paracetamol (chronic): HIGH RISK — hepatotoxicity in CPS1 (liver site of expression)

DIFFERENTIAL DIAGNOSIS:
  OTC deficiency: Ammonia HIGH (same); citrulline LOW (same); DIFFERENT: orotic acid HIGH (OTC) vs NORMAL (CPS1)
    OTC is X-linked (CPS1 is AR); OTC is 50x more common than CPS1
  NAGS deficiency: Biochemically IDENTICAL to CPS1 (no carbamoyl-P in both); orotic NORMAL in both
    CRITICAL DISTINCTION: NAGS responds dramatically to NCG (therapeutic); CPS1 responds only if NCG-responsive
    NCG trial distinguishes: complete response = NAGS; partial/none = CPS1
  ASS1 deficiency (citrullinemia I): Citrulline VERY HIGH (>100 µmol/L vs <5 in CPS1) — OPPOSITE
    Orotic acid elevated (carbamoyl-P made in CPS1 block, backed up before OTC — ASS1 block upstream of that)
  GLUD1 GoF: Ammonia 100–500 (lower); ALSO hypoglycemia + hyperinsulinism; citrulline NORMAL; orotic NORMAL
  HHH syndrome (SLC25A15): Ornithine VERY HIGH; homocitrullinuria; orotic acid HIGH (like OTC not CPS1)
  OAT deficiency: Ammonia NORMAL (KEY NEGATIVE — OAT ammonia NORMAL vs CPS1 CRITICALLY HIGH)

VARIANTS (CPS1 — ATP-binding site, NAG-binding domain, substrate binding, catalytic):
  p.Glu1024Gly: ~12%; allosteric/NAG-binding domain; moderate-severe; partially NCG-responsive
  p.Arg1459His: ~10%; phosphate binding / active site; severe neonatal; null function
  p.Thr1443Ile: ~9%; ATP-binding domain active site; severe neonatal
  p.Ser1450Pro: ~8%; C-terminal catalytic domain; severe; no residual activity
  c.IVS22+1G>A: ~7%; splice null; exon skip → NMD; severe; pan-ethnic
  p.Ala1475Val: ~6%; BCT (bicarbonate and carbamoyl-P transfer) domain; moderate
  p.Lys1201Asn: ~6%; NAG-binding pocket; NCG-RESPONSIVE; mild-moderate; late-onset in some
  p.Arg1453Cys: ~5%; substrate-binding active site; moderate-severe

NEONATAL SCREENING:
  Standard NBS: citrulline LOW flag (same as OTC; both <5 µmol/L) — CANNOT distinguish CPS1 from OTC by NBS alone
  Urine orotic acid (NBS supplement): NORMAL in CPS1 (unlike OTC where HIGH) — post-NBS differentiator
  Confirmatory: plasma amino acids + urine orotic acid + gene panel (CPS1 + OTC + NAGS)
"""
import random

_N    = 40    # cohort size (consistent with all expert dashboards)
_SEED = 199   # deterministic seed (OTC=193, next +6=199)


def _rng():
    return random.Random(_SEED)


# ──────────────────────────────────────────────────────────────────────────────
def get_overview():
    rng = _rng()

    # Phenotype distribution
    n_neonatal   = round(_N * 0.55)   # Neonatal-Onset: null/near-null; day 1-3; most severe UCD
    n_late       = round(_N * 0.30)   # Late-Onset: residual 5-20% activity; episodic
    n_ncg        = _N - n_neonatal - n_late  # NCG-Responsive: NAG-binding domain; partial activation with NCG

    phenotypes = {
        "Neonatal-Onset (Null)": {"n": n_neonatal, "pct": round(100 * n_neonatal / _N)},
        "Late-Onset (Partial)":  {"n": n_late,     "pct": round(100 * n_late / _N)},
        "NCG-Responsive":        {"n": n_ncg,       "pct": round(100 * n_ncg / _N)},
    }

    # Biomarker distributions (CPS1 LOF → ammonia HIGH, citrulline LOW, orotic NORMAL)
    amm_neonatal = [rng.uniform(600, 2000) for _ in range(n_neonatal)]
    amm_late     = [rng.uniform(150, 600)  for _ in range(n_late)]
    amm_ncg      = [rng.uniform(80, 300)   for _ in range(n_ncg)]
    all_amm = amm_neonatal + amm_late + amm_ncg
    avg_amm = round(sum(all_amm) / _N)

    # Citrulline (critically low — same as OTC; no carbamoyl-P to feed OTC step 2)
    cit_neonatal = [rng.uniform(0.3, 3.0) for _ in range(n_neonatal)]
    cit_late     = [rng.uniform(1.0, 5.0) for _ in range(n_late)]
    cit_ncg      = [rng.uniform(2.0, 8.0) for _ in range(n_ncg)]
    all_cit = cit_neonatal + cit_late + cit_ncg
    avg_cit = round(sum(all_cit) / _N, 1)

    # Urine orotic acid — NORMAL in CPS1 (no carbamoyl-P overflow; none produced)
    orot_all = [rng.uniform(0.5, 5.5) for _ in range(_N)]   # normal range <6 µmol/mol Cr
    avg_orot = round(sum(orot_all) / _N, 1)

    # Arginine (low, downstream deficient)
    arg_all  = [rng.uniform(5, 22)    for _ in range(_N)]
    avg_arg  = round(sum(arg_all) / _N, 1)

    # Ornithine (mildly elevated; substrate for OTC backs up; less than OAT)
    orn_all  = [rng.uniform(40, 130)  for _ in range(_N)]
    avg_orn  = round(sum(orn_all) / _N)

    # Clinical outcome flags
    n_seizures       = round(_N * 0.65)
    n_dre            = round(_N * 0.30)
    n_idd            = round(_N * 0.78)
    n_se             = round(_N * 0.30)
    n_cerebral_oedema = round(_N * 0.60)
    n_protein_aversion = round(_N * 0.60)
    n_hepatomegaly    = round(_N * 0.40)

    return {
        "subtitle": (
            "CPS1 Deficiency — Autosomal recessive urea cycle disorder (rarest proximal UCD). "
            "CPS1 catalyses NH₃ + CO₂ + 2ATP → Carbamoyl-P (step 1 of 5; most proximal urea cycle enzyme). "
            "LOF: no carbamoyl-P produced → total urea cycle arrest → ammonia CRITICALLY HIGH. "
            "KEY DISTINCTION from OTC: orotic acid NORMAL in CPS1 (no carbamoyl-P overflow). "
            "NCG (N-carbamylglutamate) trial mandatory — 15–20% respond (NAG-binding domain variants). "
            "VPA ABSOLUTE CI — inhibits NAGS → no NAG → CPS1 completely inactivated."
        ),
        "gene": "CPS1",
        "chromosome": "2q34",
        "protein_size": "1500 aa; mitochondrial matrix; NAG-allosteric enzyme; largest UCD enzyme",
        "omim_gene": "608307",
        "omim_disease": "237300",
        "inheritance": "Autosomal Recessive (AR) — both alleles LOF required",
        "cohort_n": _N,
        "seed": _SEED,
        "phenotype_distribution": phenotypes,
        "kpi": {
            "avg_plasma_ammonia_umol_l":      avg_amm,
            "avg_plasma_citrulline_umol_l":   avg_cit,
            "avg_urine_orotic_acid_umol_mol": avg_orot,
            "avg_plasma_arginine_umol_l":     avg_arg,
            "avg_plasma_ornithine_umol_l":    avg_orn,
            "pct_seizures":                   round(100 * n_seizures / _N),
            "pct_dre":                        round(100 * n_dre / _N),
            "pct_idd":                        round(100 * n_idd / _N),
            "pct_status_epilepticus":         round(100 * n_se / _N),
            "pct_cerebral_oedema":            round(100 * n_cerebral_oedema / _N),
            "pct_protein_aversion":           round(100 * n_protein_aversion / _N),
            "pct_hepatomegaly":               round(100 * n_hepatomegaly / _N),
        },
        "biomarker_normals": {
            "description": "KEY NEGATIVE biomarkers — NORMAL in CPS1 (distinguishes from other diseases)",
            "orotic_acid_normal":    "NORMAL (<6 µmol/mol Cr) — CRITICAL KEY POSITIVE NEGATIVE vs OTC (HIGH)",
            "plp_normal":            "NORMAL — CPS1 NOT PLP-dependent (KEY NEG vs PNPO/ALDH4A1)",
            "alpha_aasa_normal":     "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE",
            "pipecolic_normal":      "NORMAL — KEY NEGATIVE vs PDE/peroxisomal disorders",
            "thcy_normal":           "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR",
            "mma_normal":            "NORMAL — KEY NEGATIVE vs MMUT/cblA/cblB",
            "gaba_normal":           "NORMAL — KEY NEGATIVE vs ABAT (HIGH) / GAD1 (LOW)",
            "ghb_normal":            "NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB HIGH",
            "glycine_normal":        "NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH)",
            "gaa_normal":            "NORMAL — KEY NEGATIVE vs GAMT (50–300 µmol/L GAA)",
            "creatine_normal":       "NORMAL — KEY NEGATIVE vs GAMT/AGAT/SLC6A8",
        },
        "key_positives": {
            "ammonia_critically_high":    "Plasma ammonia >200 µmol/L crisis (>500 neonatal) — PATHOGNOMONIC",
            "citrulline_critically_low":  "Plasma citrulline <5 µmol/L (normal 15–35) — PATHOGNOMONIC (same as OTC)",
            "orotic_acid_NORMAL_key_negative": (
                "Urine orotic acid NORMAL (<6 µmol/mol Cr) — CRITICAL. "
                "No carbamoyl-P produced → no overflow into pyrimidine pathway → no orotic acid. "
                "This is the SINGLE biochemical test distinguishing CPS1 from OTC."
            ),
            "ncg_response_diagnostic": (
                "N-carbamylglutamate (NCG) trial: 15–20% of CPS1 patients respond with >50% NH₃ reduction. "
                "NCG-responsive = NAG-binding domain variant. NAGS deficiency shows COMPLETE response."
            ),
        },
        "pathway_context": {
            "step": "Step 1 of 5 in urea cycle (mitochondrial) — most proximal; entry point",
            "reaction": "NH₃ + CO₂ + 2ATP → Carbamoyl phosphate + 2ADP + Pi  [NAG required]",
            "block_consequence": "Carbamoyl-P NOT produced → no substrate for OTC (step 2) → citrulline not made",
            "nag_axis": "NAG (from NAGS) is obligate allosteric activator of CPS1 — NAGS LOF = identical phenotype",
            "vs_otc": "CPS1 blocks step 1 → orotic acid NORMAL; OTC blocks step 2 → carbamoyl-P overflows → orotic HIGH",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_breakdown():
    rng = _rng()

    n_neonatal = round(_N * 0.55)
    n_late     = round(_N * 0.30)
    n_ncg      = _N - n_neonatal - n_late

    # Seizure type breakdown
    n_seizures = round(_N * 0.65)
    seizure_types = {
        "GTCS (acute hyperammonemic crisis)": {"pct": 45, "n": round(n_seizures * 0.45)},
        "Myoclonic encephalopathy":           {"pct": 25, "n": round(n_seizures * 0.25)},
        "Focal seizures":                     {"pct": 20, "n": round(n_seizures * 0.20)},
        "Absence-like (metabolic)":           {"pct": 15, "n": round(n_seizures * 0.15)},
        "Status epilepticus":                 {"pct": 30, "n": round(n_seizures * 0.30)},
    }

    # Crisis triggers
    triggers = {
        "Febrile illness / infection":   {"pct": 78, "detail": "Protein catabolism → endogenous nitrogen surge"},
        "High protein meal":             {"pct": 70, "detail": "Exogenous nitrogen load exceeds urea cycle capacity (= zero)"},
        "Prolonged fasting":             {"pct": 65, "detail": "Muscle catabolism → endogenous nitrogen; common neonatal trigger"},
        "Post-surgical catabolism":      {"pct": 50, "detail": "Catabolic stress → protein breakdown → ammonia surge"},
        "Valproate exposure":            {"pct": 45, "detail": "NAGS inhibition → no NAG → CPS1 COMPLETELY inactivated → crisis"},
        "Corticosteroid therapy":        {"pct": 38, "detail": "Catabolic hormone → protein breakdown → NH₃ surge"},
        "Chloramphenicol exposure":      {"pct": 20, "detail": "Mitochondrial toxin → CPS1 synthesis inhibited → crisis"},
        "Post-partum (females, AR)":     {"pct": 15, "detail": "Catabolic state; CPS1 AR so females equally at risk (unlike OTC)"},
    }

    # Treatment efficacy
    treatments = {
        "N-carbamylglutamate (NCG / Carbaglu)": {
            "level": "A", "category": "CPS1 activator (NAG analogue)", "efficacy_pct": 90,
            "mechanism": "Structural analogue of NAG → directly activates residual CPS1. "
                         "15–20% NCG-responsive (NAG-binding domain variants). Full response = NAGS. "
                         "Trial mandatory in ALL patients — safe even if non-responsive.",
        },
        "Sodium benzoate + Sodium phenylacetate (RAVICTI)": {
            "level": "A", "category": "Nitrogen scavenger", "efficacy_pct": 85,
            "mechanism": "Benzoate → hippurate (removes glycine-N); phenylacetate → PAG (removes glutamine-N). 2N per cycle.",
        },
        "Arginine IV (acute) + oral (chronic)": {
            "level": "A", "category": "Conditionally essential AA", "efficacy_pct": 80,
            "mechanism": "Becomes essential in CPS1 (not synthesised); 150–250 mg/kg/day oral chronic; IV in acute crisis.",
        },
        "Haemodialysis / CRRT (acute crisis)": {
            "level": "A", "category": "Ammonia removal (physical)", "efficacy_pct": 92,
            "mechanism": "First-line for ammonia >500 µmol/L; clears ammonia faster than any pharmacotherapy. "
                         "CRRT preferred in neonates; HD more efficient per hour in older patients.",
        },
        "Liver transplantation": {
            "level": "A", "category": "Curative (CPS1 hepatic enzyme)", "efficacy_pct": 97,
            "mechanism": "Restores hepatic CPS1; normalises urea cycle; does NOT reverse prior neurological damage. "
                         "Optimal pre-second/third crisis. AR inheritance: living related donor possible.",
        },
        "IV dextrose 10% (GIR 10–12 mg/kg/min)": {
            "level": "A", "category": "Anti-catabolic (acute)", "efficacy_pct": 75,
            "mechanism": "Provides nitrogen-free energy; suppresses endogenous protein catabolism; first step in acute protocol.",
        },
        "Citrulline supplementation": {
            "level": "A", "category": "Urea cycle replenishment", "efficacy_pct": 78,
            "mechanism": "Metabolised downstream of CPS1 block; replenishes arginine and citrulline; 150–250 mg/kg/day.",
        },
        "Levetiracetam (LEV)": {
            "level": "B", "category": "AED — first-line in CPS1", "efficacy_pct": 70,
            "mechanism": "SV2A mechanism; no hepatotoxicity; no effect on ammonia; safe in all UCD. Seizures secondary to ammonia.",
        },
        "Protein restriction (natural)": {
            "level": "A", "category": "Chronic dietary management", "efficacy_pct": 80,
            "mechanism": "0.5–1.5 g/kg/day natural protein + essential AA supplement; limits exogenous nitrogen load.",
        },
    }

    # Drug risks
    drug_risks = {
        "Valproate / VPA": {
            "risk": "ABSOLUTE CI",
            "detail": (
                "Inhibits NAGS → no NAG production → CPS1 CANNOT be activated. "
                "In CPS1 deficiency, VPA mechanism is MORE DIRECT than in OTC: targets the CPS1 activator. "
                "Even single therapeutic dose → catastrophic hyperammonemia. Multiple fatalities. "
                "CI in ALL urea cycle disorders."
            ),
        },
        "High protein diet": {
            "risk": "ABSOLUTE CI",
            "detail": "Nitrogen overload → ammonia surge; impossible to metabolise without functional CPS1.",
        },
        "L-Asparaginase (chemotherapy)": {
            "risk": "ABSOLUTE CI",
            "detail": "Depletes asparagine → hyperammonemia; combined with CPS1 = catastrophic urea cycle arrest.",
        },
        "Chloramphenicol": {
            "risk": "HIGH RISK",
            "detail": "Mitochondrial protein synthesis inhibitor → CPS1 (mitochondrial) synthesis directly impaired. "
                      "Documented hyperammonemia in CPS1 patients. Use alternative antibiotics.",
        },
        "Prolonged fasting": {
            "risk": "HIGH RISK",
            "detail": "Endogenous protein catabolism → nitrogen load → ammonia surge. Emergency protocol on any fast >4–6h.",
        },
        "Systemic glucocorticoids (high-dose)": {
            "risk": "HIGH RISK",
            "detail": "Catabolic hormone → protein breakdown → internal nitrogen load → ammonia surge in CPS1.",
        },
        "Topiramate": {
            "risk": "MODERATE RISK",
            "detail": "Carbonic anhydrase inhibition affects CO₂ availability for CPS1 reaction (NH₃ + CO₂ + 2ATP). "
                      "Case reports of hyperammonemia; monitor ammonia if used; avoid combining with VPA.",
        },
        "Acetaminophen / paracetamol (chronic high dose)": {
            "risk": "MODERATE RISK",
            "detail": "Hepatotoxicity risk in CPS1 (liver main CPS1 expression site); glutathione depletion.",
        },
    }

    # Variants
    variants = [
        {"name": "p.Glu1024Gly",  "domain": "NAG-binding allosteric domain",     "pct": 12, "severity": "Moderate-Severe; partially NCG-responsive", "responsive": True},
        {"name": "p.Arg1459His",  "domain": "Phosphate binding / active site",    "pct": 10, "severity": "Severe neonatal; null function",             "responsive": False},
        {"name": "p.Thr1443Ile",  "domain": "ATP-binding active site",            "pct": 9,  "severity": "Severe neonatal",                           "responsive": False},
        {"name": "p.Ser1450Pro",  "domain": "C-terminal catalytic domain",        "pct": 8,  "severity": "Severe; no residual activity",              "responsive": False},
        {"name": "c.IVS22+1G>A", "domain": "Splice null (exon skip→NMD)",        "pct": 7,  "severity": "Severe; pan-ethnic",                        "responsive": False},
        {"name": "p.Ala1475Val",  "domain": "BCT domain (carbamoyl-P transfer)",  "pct": 6,  "severity": "Moderate",                                  "responsive": False},
        {"name": "p.Lys1201Asn",  "domain": "NAG-binding pocket",                "pct": 6,  "severity": "Mild-Moderate; NCG-RESPONSIVE; late-onset",  "responsive": True},
        {"name": "p.Arg1453Cys",  "domain": "Substrate-binding active site",      "pct": 5,  "severity": "Moderate-Severe",                           "responsive": False},
    ]

    # Differentials
    differentials = [
        {
            "disease": "OTC deficiency",
            "key_diff": "Urine orotic acid HIGH in OTC (carbamoyl-P overflows) vs NORMAL in CPS1 (no carbamoyl-P made). "
                        "OTC is X-linked (CPS1 is AR); OTC is 50× more common than CPS1.",
            "distinguishing": "Urine orotic acid + inheritance",
        },
        {
            "disease": "NAGS deficiency",
            "key_diff": "Biochemically IDENTICAL to CPS1 (both: ammonia HIGH, citrulline LOW, orotic NORMAL). "
                        "NCG trial DISTINGUISHES: complete dramatic response = NAGS; partial/no response = CPS1.",
            "distinguishing": "NCG (N-carbamylglutamate) trial response",
        },
        {
            "disease": "ASS1 deficiency (citrullinemia type I)",
            "key_diff": "Citrulline VERY HIGH (>100 µmol/L in ASS1 vs <5 µmol/L in CPS1) — OPPOSITE direction.",
            "distinguishing": "Plasma citrulline direction",
        },
        {
            "disease": "GLUD1 GoF (HHS syndrome)",
            "key_diff": "GLUD1: ALSO hypoglycemia + hyperinsulinism; citrulline NORMAL; orotic NORMAL; ammonia 100–500 not >1000.",
            "distinguishing": "Glucose, insulin, citrulline",
        },
        {
            "disease": "OAT deficiency",
            "key_diff": "OAT: ammonia NORMAL (KEY NEGATIVE — critical distinction); ornithine VERY HIGH (400–1500).",
            "distinguishing": "Ammonia level (OAT = normal vs CPS1 = critically HIGH)",
        },
        {
            "disease": "HHH syndrome (SLC25A15)",
            "key_diff": "HHH: homocitrullinuria pathognomonic; ornithine HIGH; orotic HIGH (like OTC, not CPS1).",
            "distinguishing": "Homocitrullinuria + orotic acid",
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
            "Neonatal-Onset (Null)": {
                "n": n_neonatal,
                "pct": round(100 * n_neonatal / _N),
                "description": (
                    "Null/near-null CPS1; no residual enzyme. Day 1–3 presentation (earlier than OTC). "
                    "Ammonia often >1000 µmol/L at presentation; status epilepticus common; "
                    "40–50% neonatal mortality without emergent CRRT. Most severe proximal UCD."
                ),
                "avg_peak_ammonia_umol_l": 1050,
                "seizure_rate_pct": 82,
            },
            "Late-Onset (Partial)": {
                "n": n_late,
                "pct": round(100 * n_late / _N),
                "description": (
                    "Residual 5–20% CPS1 activity; episodic crises triggered by illness/protein/fasting. "
                    "Protein aversion key clue; range late infancy to adulthood; IDD correlates with crises."
                ),
                "avg_peak_ammonia_umol_l": 320,
                "seizure_rate_pct": 52,
            },
            "NCG-Responsive": {
                "n": n_ncg,
                "pct": round(100 * n_ncg / _N),
                "description": (
                    "NAG-binding domain variants; NCG (N-carbamylglutamate) activates residual CPS1. "
                    "Often milder; some present in adulthood; NCG may avoid liver transplant. "
                    "Complete NCG response = NAGS deficiency (gene panel mandatory to distinguish)."
                ),
                "avg_peak_ammonia_umol_l": 180,
                "seizure_rate_pct": 35,
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "disease": "CPS1 Deficiency (Carbamoyl Phosphate Synthetase 1 Deficiency)",
        "omim_disease": "237300",
        "omim_gene": "608307",
        "gene": "CPS1",
        "chromosome": "2q34",
        "inheritance": "Autosomal Recessive (AR); biallelic LOF required; males and females equally affected",
        "prevalence": "~1:800,000–1,300,000; rarest proximal urea cycle disorder (OTC is 50× more common)",
        "enzyme": "Carbamoyl Phosphate Synthetase 1 (CPS1); 1500 aa; largest urea cycle enzyme; mitochondrial matrix",
        "reaction": "NH₃ + CO₂ + 2ATP → Carbamoyl phosphate + 2ADP + Pi  [NAG obligate allosteric activator]",
        "pathway_position": "Mitochondrial matrix; step 1 of 5 in urea cycle; most proximal enzyme; entry point for NH₃",
        "mechanism_of_disease": (
            "CPS1 LOF → carbamoyl phosphate CANNOT be produced → total urea cycle arrest at step 1. "
            "No substrate for OTC (step 2) → citrulline cannot be made (LOW). "
            "No carbamoyl-P overflow (unlike OTC) → orotic acid NORMAL. "
            "Ammonia accumulates, detoxified via glutamine and alanine until overwhelmed. "
            "NAG (N-acetylglutamate from NAGS) is the obligate allosteric activator — VPA abolishes NAG production."
        ),
        "nag_axis": (
            "N-acetylglutamate (NAG) is the OBLIGATE allosteric activator of CPS1. "
            "NAGS (N-acetylglutamate synthase) produces NAG from glutamate + acetyl-CoA. "
            "VPA → inhibits NAGS → no NAG → CPS1 COMPLETELY INACTIVE. "
            "N-carbamylglutamate (NCG) = structural NAG analogue → activates residual CPS1 directly. "
            "NCG-responsive variants have intact CPS1 protein but impaired NAG-binding → NCG bypasses this."
        ),
        "key_biomarker_positives": {
            "plasma_ammonia": "CRITICALLY HIGH (>500 µmol/L neonatal; >200 late-onset crisis; normal <50) — PATHOGNOMONIC",
            "plasma_citrulline": "CRITICALLY LOW (<5 µmol/L; normal 15–35) — same as OTC; no carbamoyl-P to feed OTC",
            "plasma_arginine": "LOW (<25 µmol/L; normal 60–120) — downstream product not synthesised",
            "plasma_glutamine": "ELEVATED (600–900 µmol/L) — GS detoxifies ammonia → glutamine",
            "plasma_alanine": "ELEVATED (500–800 µmol/L) — ALT-mediated NH₃ disposal alternative route",
        },
        "key_biomarker_negatives": {
            "urine_orotic_acid": "NORMAL (<6 µmol/mol Cr) — CRITICAL KEY NEGATIVE distinguishing CPS1 from OTC",
            "plp": "NORMAL — CPS1 not PLP-dependent (KEY NEG vs PNPO/ALDH4A1 where PLP LOW)",
            "alpha_aasa": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE",
            "pipecolic": "NORMAL — KEY NEGATIVE vs PDE/peroxisomal disorders",
            "thcy": "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR",
            "mma": "NORMAL — KEY NEGATIVE vs methylmalonic acidemia",
            "gaba_csf": "NORMAL — KEY NEGATIVE vs ABAT (DRAMATICALLY HIGH) and GAD1 (LOW)",
            "ghb_urine": "NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB dramatically HIGH",
            "glycine": "NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH) where glycine HIGH",
            "gaa": "NORMAL — KEY NEGATIVE vs GAMT (GAA 50–300 µmol/L PATHOGNOMONIC)",
            "creatine": "NORMAL — KEY NEGATIVE vs GAMT/AGAT/SLC6A8",
        },
        "critical_cps1_vs_otc_distinction": (
            "Urine orotic acid: NORMAL in CPS1 (no carbamoyl-P produced → no overflow) "
            "but HIGH in OTC (carbamoyl-P produced at step 1 but CANNOT be used at step 2 → overflows to pyrimidines). "
            "This is the SINGLE biochemical test that distinguishes CPS1 from OTC. "
            "Both have: ammonia HIGH, citrulline LOW, arginine LOW."
        ),
        "critical_cps1_vs_nags_distinction": (
            "NAGS deficiency is biochemically IDENTICAL to CPS1 (same: ammonia HIGH, citrulline LOW, orotic NORMAL). "
            "N-carbamylglutamate (NCG) trial: COMPLETE dramatic NH₃ normalisation = NAGS deficiency. "
            "PARTIAL or NO response = CPS1 deficiency. Gene panel mandatory to confirm both genes."
        ),
        "absolute_contraindications": {
            "valproate_vpa": (
                "ABSOLUTE CI — inhibits NAGS → no NAG production → CPS1 CANNOT be activated. "
                "In CPS1 deficiency: VPA mechanism is MORE DIRECT than in OTC: abolishes the CPS1 activator. "
                "Even single dose → catastrophic hyperammonemia. Multiple fatalities documented. "
                "CI extends to ALL urea cycle disorders (CPS1, OTC, ASS1, ASL, ARG1, NAGS)."
            ),
            "high_protein_diet": "ABSOLUTE CI — nitrogen load → ammonia surge; CPS1 cannot process any NH₃",
            "l_asparaginase": "ABSOLUTE CI — hyperammonemia + combined UCD block = catastrophic",
            "chloramphenicol": (
                "HIGH RISK — mitochondrial protein synthesis inhibitor; CPS1 is a mitochondrial enzyme; "
                "inhibits CPS1 synthesis directly. Documented hyperammonemia in CPS1 patients. "
                "Use alternative antibiotics."
            ),
        },
        "first_line_treatments": {
            "ncg_carbaglu": "N-carbamylglutamate (NCG) 100–250 mg/kg/day — Level A TRIAL IN ALL; activates residual CPS1",
            "nitrogen_scavengers": "Sodium benzoate + phenylacetate (RAVICTI) — Level A chronic + acute",
            "liver_transplant": "Curative — Level A; restores hepatic CPS1; AR = living related donor possible",
            "acute_ammonia_removal": "CRRT/HD for ammonia >500 µmol/L — Level A; neonates: CRRT preferred",
            "aed": "Levetiracetam (LEV) — Level B first-line AED (no hepatotoxicity; no ammonia effect)",
        },
        "seizure_mechanism": (
            "Ammonia impairs GABA-A receptor gating (reduces inhibitory tone) and over-activates NMDA receptors. "
            "Net: reduced inhibition + increased excitation → seizure threshold lowered. "
            "Cerebral oedema (cytotoxic + vasogenic) → intracranial hypertension → secondary seizures. "
            "EEG: burst-suppression (severe neonatal), triphasic waves (ammonia encephalopathy), diffuse slowing."
        ),
        "ar_inheritance_note": (
            "AR inheritance: BOTH parents are obligate carriers; recurrence risk 25% per pregnancy. "
            "Males and females equally affected — contrast with OTC (X-linked, males predominantly). "
            "Heterozygous carriers: typically asymptomatic; may have subtle amino acid changes. "
            "Cascade carrier testing of family members essential after index diagnosis."
        ),
    }
