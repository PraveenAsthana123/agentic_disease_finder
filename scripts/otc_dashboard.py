#!/usr/bin/env python3
"""OTC (Ornithine Transcarbamylase) Deficiency Epilepsy Dashboard.

OTC encodes Ornithine Transcarbamylase, a mitochondrial matrix enzyme:
  L-Ornithine + Carbamoyl phosphate  →  L-Citrulline + Phosphate  (urea cycle, step 2)

OTC DISEASE: Ornithine Transcarbamylase Deficiency (OTCD)
  OMIM Disease: #311250   Gene: OTC, OMIM *300461
  Chromosome: Xp21.1
  Inheritance: X-linked — hemizygous males severely affected; carrier females 25% symptomatic
  Protein: 354 aa; mitochondrial matrix homotrimer; ~40 kDa monomer; expressed primarily in liver
  Prevalence: ~1:14,000–50,000; most common urea cycle disorder; ~1:14,000 in some registries

MECHANISM — LOSS-OF-FUNCTION (urea cycle block → ammonia cannot be detoxified):
  Normal OTC: Ornithine + Carbamoyl-P → Citrulline + Pi  (mitochondrial matrix)
  OTC LOF: Citrulline CANNOT be made → urea cycle BLOCKED at step 2
  Upstream accumulation: Carbamoyl phosphate overflows into cytoplasm →
     enters pyrimidine synthesis → OROTIC ACID accumulates (urine orotic acid HIGH)
  Ammonia accumulates: cannot enter urea cycle via OTC → AMMONIA CRITICALLY ELEVATED
  Ornithine: substrate for OTC → mildly accumulates (50–150 µmol/L; not dramatic like OAT)
  Arginine: downstream of block → LOW (not synthesised; essential amino acid in OTCD)
  Glutamine: elevated (GS detoxifies ammonia → glutamine; normal 400–700, OTC 600–900)

POSITION IN UREA CYCLE (OTCD = step 2 of 5 steps):
  NH3 + CO2 + 2ATP → [CPS1] → Carbamoyl-P  (step 1)
  Carbamoyl-P + Ornithine → [OTC, BLOCKED] → Citrulline  (step 2)
  Citrulline + Asp → [ASS1] → Argininosuccinate  (step 3)
  Argininosuccinate → [ASL] → Arginine + Fumarate  (step 4)
  Arginine → [ARG1] → Ornithine + Urea  (step 5 — regenerates ornithine)

  KEY PATHWAY CONTRASTS:
    CPS1 LOF: Carbamoyl-P NOT made → orotic acid NORMAL (no CP overflow); ammonia HIGH
    OTC LOF:  Carbamoyl-P made but CANNOT be used → CP overflows → orotic acid HIGH; ammonia HIGH
    ASS1 LOF: Citrulline cannot be converted → citrulline VERY HIGH (citrullinemia type I)
    ASL LOF:  Argininosuccinate accumulates; citrulline moderately elevated; argininosuccinic aciduria
    ARG1 LOF: Arginine accumulates; hyperargininemia; mild ammonia elevation

OTC BIOCHEMISTRY (LOF → hyperammonemia + citrulline deficiency + orotic aciduria):
  Plasma ammonia:        CRITICALLY ELEVATED (200–>1000 µmol/L in crisis; normal <50 µmol/L) — PATHOGNOMONIC
  Plasma citrulline:     CRITICALLY LOW (<5 µmol/L; normal 15–35 µmol/L) — PATHOGNOMONIC KEY POSITIVE
  Urine orotic acid:     HIGH (>10–20 µmol/mol Cr; normal <6 µmol/mol Cr) — PATHOGNOMONIC KEY POSITIVE
  Plasma arginine:       LOW (<25 µmol/L; normal 60–120 µmol/L) — downstream product absent
  Plasma ornithine:      MILDLY ELEVATED or NORMAL (50–150 µmol/L; normal <100) — substrate backs up
  Plasma glutamine:      ELEVATED (600–900 µmol/L; normal 400–700) — ammonia detox by GS
  Plasma alanine:        ELEVATED (500–800 µmol/L; normal 200–450) — alternative ammonia disposal
  PLP (plasma):          NORMAL — KEY NEGATIVE vs PNPO/ALDH4A1 (OTC is NOT PLP-dependent)
  alpha-AASA (urine):    NORMAL — KEY NEGATIVE vs ALDH7A1/PDE (>30 mmol/mol Cr in PDE)
  Pipecolic acid:        NORMAL — KEY NEGATIVE vs PDE and peroxisomal disorders
  tHcy:                  NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR homocysteine disorders
  MMA:                   NORMAL — KEY NEGATIVE vs methylmalonic acidemia (MMUT/cblA/cblB)
  GABA (CSF):            NORMAL — KEY NEGATIVE vs ABAT (GABA dramatically high) and GAD1 (GABA low)
  GHB (urine):           NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB dramatically high
  Glycine:               NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH) where glycine elevated
  GAA:                   NORMAL — KEY NEGATIVE vs GAMT (50–300 µmol/L GAA)
  Creatine:              NORMAL — KEY NEGATIVE vs GAMT/AGAT/SLC6A8

  CRITICAL KEY DISTINCTION OTC vs CPS1:
    Urine orotic acid HIGH in OTC (carbamoyl phosphate overflows to pyrimidine pathway)
    Urine orotic acid NORMAL in CPS1 (carbamoyl phosphate NOT made, no overflow)
    Both have: ammonia HIGH, citrulline LOW — orotic acid is the SINGLE differentiating test

EPILEPSY IN OTC DEFICIENCY (secondary to hyperammonemia and cerebral oedema):
  Overall seizure rate: ~60–70% (high; ammonia directly impairs GABA-A and NMDA receptors)
  Mechanism: ammonia → altered GABA-A receptor gating + NMDA receptor overactivation
  Seizure types:
    GTCS (generalised tonic-clonic): ~45% of those with seizures — MODAL in acute hyperammonemic crisis
    Focal seizures: ~30% (focal cortical dysfunction secondary to cerebral oedema/infarction)
    Myoclonic: ~20% (ammonia encephalopathy pattern; multifocal)
    Absence-like: ~15% (metabolic encephalopathy)
    Status epilepticus: ~25% (acute hyperammonemic crisis; constitutes a neurological emergency)
  EEG:
    Burst-suppression: severe neonatal crisis pattern (ammonia >500 µmol/L)
    Diffuse slowing: most common interictal finding
    Triphasic waves: characteristic of ammonia encephalopathy (metabolic brain dysfunction)
    Rhythmic delta: bifrontal; often with high-amplitude runs
  Drug-resistant epilepsy: 20–30% (correlates with severity and duration of hyperammonemic crises)
  MRI acute: cytotoxic/vasogenic oedema (diffusion restriction parieto-occipital/diffuse)
  MRI chronic: cortical atrophy, gliosis, white matter changes (watershed/border-zone)
  Delayed intellectual disability: correlates with degree/duration of hyperammonemia episodes

NON-SEIZURE NEUROLOGIC FEATURES:
  Intellectual disability: 70–80% (directly proportional to peak ammonia and crisis duration)
  Cerebral oedema (acute): 50–60% of neonatal-onset cases; major cause of death/disability
  Cortical atrophy (MRI): 40–60% of survivors of neonatal crisis
  Behavioural/psychiatric: 25–35% (ADHD, anxiety, depression in late-onset patients)
  Protein aversion (behavioural): 60–70% in late-onset — patients intuitively restrict protein

NON-NEUROLOGIC FEATURES:
  Hepatomegaly: 40–50% (liver site of OTC expression; steatosis common)
  Elevated transaminases: 30–40% (secondary to metabolic stress; not primary liver disease)
  Coagulopathy (acute): 20–30% (liver synthetic dysfunction in severe crisis)
  Growth retardation: 30–40% (chronic protein restriction + metabolic instability)
  Hyperactivity / ADHD-like: 35–45% (inter-ictal; especially in partially treated cases)

PHENOTYPE CLASSES:
  Neonatal-Onset Males (null) (40%): hemizygous null; no residual OTC activity; present day 1–5
    Peak ammonia often >500–2000 µmol/L; often lethal without emergent treatment; seizures common
  Late-Onset Males (partial) (35%): residual OTC activity 5–25%; first crisis triggered by
    illness/protein load/catabolic stress; present from infancy to adulthood; milder
  Symptomatic Females (Lyon) (25%): skewed X-inactivation favours mutant allele; late-onset usual;
    protein aversion as clue; serum ammonia mildly-moderately elevated; occasionally severe

TREATMENTS (ACUTE CRISIS):
  IV dextrose 10%: Level A — suppress catabolism, provide energy without nitrogen
  Sodium benzoate + Sodium phenylacetate (BUPHENYL/RAVICTI): Level A — nitrogen scavengers
    Benzoate → hippurate (+ glycine N waste); phenylacetate → phenylacetylglutamine (+ glutamine N waste)
    Removes 2 nitrogen molecules per molecule of waste nitrogen excreted in urine
  Arginine (IV then oral): Level A — replenish downstream urea cycle; becomes essential amino acid in OTC
  Citrulline (oral/IV): Level A — alternative to arginine; metabolised downstream of OTC block
  Haemodialysis / CRRT: Level A — for ammonia >500 µmol/L unresponsive to medical therapy
  Stop protein intake: Level A — 24–48 h protein-free; then restart with restricted natural protein

TREATMENTS (CHRONIC):
  Glycerol phenylbutyrate (RAVICTI): Level A PRIMARY long-term — nitrogen scavenger; oral liquid
    More palatable than sodium phenylbutyrate; reduces ammonia 40–60% in chronic management
  Protein restriction (natural protein): Level A — limit natural protein 0.5–1.5 g/kg/day
    + essential amino acid supplement to maintain nitrogen balance without excess ammonia
  Citrulline supplementation (150–200 mg/kg/day): Level A — replenishes citrulline/arginine
  Liver transplantation: Level A CURATIVE — restores full hepatic OTC expression; normalises urea cycle
    Does NOT reverse pre-existing neurological damage; optimal timing before first/second crisis
  Levetiracetam (LEV): Level B — first-line AED in OTC; SV2A mechanism; no hepatotoxicity; safe
  Arginine (oral, 100–250 mg/kg/day): Level B — alternative to citrulline; replenishes arginine
  Benzodiazepines (acute SE): Level B — lorazepam/diazepam for acute seizure management
  Carnitine (L-carnitine): Level C — some evidence for concurrent carnitine depletion

ABSOLUTE CONTRAINDICATIONS:
  Valproate/VPA: ABSOLUTE CI — inhibits CPS1 (upstream of OTC), inhibits mitochondrial beta-oxidation
    → dramatically worsens hyperammonemia; multiple case fatalities in urea cycle disorder + VPA
    This CI extends to ALL urea cycle disorders (OTC, CPS1, ASS1, ASL, ARG1, NAGS)
    Even therapeutic VPA doses can precipitate hyperammonemic crisis in OTC carriers
    Mechanism: VPA-CoA derivatives inhibit N-acetylglutamate synthase (NAGS) → less CPS1 activator
  High protein diet: ABSOLUTE CI — nitrogen overload → ammonia surge
  L-Asparaginase (chemotherapy): ABSOLUTE CI — depletes asparagine → hyperammonemia via
    aspartate depletion; combined with OTC block = catastrophic ammonia elevation
  Prolonged fasting: HIGH RISK — protein catabolism → internal nitrogen load → ammonia surge
  Systemic glucocorticoids (high-dose): HIGH RISK — catabolic → muscle protein breakdown → ammonia
  Topiramate: MODERATE RISK — case reports of hyperammonemia; mechanism: carbonic anhydrase + VPA-like
  Acetaminophen/paracetamol (chronic high dose): HIGH RISK — hepatotoxicity in pre-existing liver disease
  N2O (nitrous oxide anaesthesia): MODERATE RISK — inactivates methionine synthase; exacerbates
    nitrogen handling; use alternative anaesthetic agents; ensure arginine supplement perioperatively

DIFFERENTIAL DIAGNOSIS (diseases with similar findings):
  CPS1 deficiency: Ammonia HIGH (same); citrulline LOW (same); DIFFERENT: orotic acid NORMAL (CPS1)
    vs HIGH (OTC) — KEY DISTINGUISHING TEST; CPS1 AR (not X-linked); rarer than OTC
  ASS1 deficiency (citrullinemia type I): Citrulline VERY HIGH (vs LOW in OTC); ammonia elevated;
    orotic acid elevated; AR; plasma citrulline >100 µmol/L (OTC <5 µmol/L) — OPPOSITE citrulline
  ASL deficiency: Argininosuccinic acid detectable; citrulline mildly elevated; AR
  NAGS deficiency: Similar to CPS1; orotic acid NORMAL; responds to N-carbamylglutamate
  OAT deficiency: Ornithine VERY HIGH (400–1500; vs mildly elevated <150 in OTC); ammonia NORMAL
    (KEY NEGATIVE in OAT; KEY POSITIVE in OTC); citrulline NORMAL in OAT (vs LOW in OTC)
  GLUD1 GoF: Ammonia elevated (100–500 µmol/L); BUT also hypoglycemia + hyperinsulinism;
    citrulline NORMAL in GLUD1 (vs LOW in OTC); orotic acid NORMAL; no HPA in OTC
  HHH syndrome (SLC25A15): Ornithine HIGH (like OAT); ammonia elevated; but citrulline NORMAL/LOW
    and orotic acid HIGH (similar to OTC) — distinguished by homocitrullinuria in HHH
  ABAT/GAD1: GABA pathway disorders; ammonia NORMAL; completely different biochemistry

VARIANTS (OTC — ornithine binding site, carbamoyl phosphate binding site, trimer interface):
  p.Arg320His: ~15%; ornithine binding domain; males severe neonatal-onset; females variable
  p.Glu228Lys: ~12%; carbamoyl phosphate binding; neonatal-onset; severe
  p.Arg129His: ~11%; active site; neonatal-onset; males lethal without treatment
  p.Ile172Thr: ~10%; MTS processing region; severe; compound heterozygous in females
  p.Asp263Gly: ~9%; active site; severe neonatal
  c.IVS6+1G>A: splice null; ~8%; exon 6 skip → frameshift → NMD; severe
  p.Ala23Thr: MTS/signal region; ~7%; mild (partial function retained); late-onset males
  p.Thr329Lys: ~6%; moderate; occasional late-onset males; residual 5–15% activity

NEONATAL SCREENING:
  Standard NBS (tandem MS): citrulline LOW → can flag; but OTC citrulline is very low and
    the cut-off is <5 µmol/L making it difficult to distinguish from normal low values in some labs
  Urine orotic acid (NBS supplement): flag for OTC; some expanded NBS panels include
  Plasma amino acids confirm: citrulline <5 + ammonia elevated = OTC until proven otherwise
"""
import random

_N    = 40    # cohort size (consistent with all expert dashboards)
_SEED = 193   # deterministic seed (OAT=187, next +6=193)


def _rng():
    return random.Random(_SEED)


# ──────────────────────────────────────────────────────────────────────────────
def get_overview():
    rng = _rng()

    # Phenotype distribution
    n_neonatal   = round(_N * 0.40)   # Neonatal-Onset Males (null): biallelic null; no residual OTC
    n_late_onset = round(_N * 0.35)   # Late-Onset Males (partial): residual 5-25%
    n_females    = _N - n_neonatal - n_late_onset  # Symptomatic Females (Lyon): X-inactivation skewed

    phenotypes = {
        "Neonatal-Onset Males": {"n": n_neonatal,   "pct": round(100 * n_neonatal   / _N)},
        "Late-Onset Males":     {"n": n_late_onset, "pct": round(100 * n_late_onset / _N)},
        "Symptomatic Females":  {"n": n_females,    "pct": round(100 * n_females    / _N)},
    }

    # Biomarker distributions (OTC LOF → ammonia HIGH, citrulline LOW, orotic acid HIGH)
    amm_neonatal   = [rng.uniform(500, 1500) for _ in range(n_neonatal)]
    amm_late       = [rng.uniform(150, 500)  for _ in range(n_late_onset)]
    amm_females    = [rng.uniform(80, 300)   for _ in range(n_females)]
    all_amm = amm_neonatal + amm_late + amm_females
    avg_amm = round(sum(all_amm) / _N)

    # Citrulline (critically low in OTC)
    cit_neonatal = [rng.uniform(0.5, 3.5) for _ in range(n_neonatal)]
    cit_late     = [rng.uniform(1.0, 5.0) for _ in range(n_late_onset)]
    cit_females  = [rng.uniform(2.0, 8.0) for _ in range(n_females)]
    all_cit = cit_neonatal + cit_late + cit_females
    avg_cit = round(sum(all_cit) / _N, 1)

    # Urine orotic acid (elevated in OTC — carbamoyl phosphate overflow to pyrimidine pathway)
    orot_neonatal = [rng.uniform(50, 200)  for _ in range(n_neonatal)]
    orot_late     = [rng.uniform(20, 100)  for _ in range(n_late_onset)]
    orot_females  = [rng.uniform(10, 60)   for _ in range(n_females)]
    all_orot = orot_neonatal + orot_late + orot_females
    avg_orot = round(sum(all_orot) / _N)

    # Arginine (low, downstream of block)
    arg_all  = [rng.uniform(8, 25)   for _ in range(_N)]
    avg_arg  = round(sum(arg_all) / _N, 1)

    # Ornithine (mildly elevated, substrate backs up)
    orn_all  = [rng.uniform(50, 150) for _ in range(_N)]
    avg_orn  = round(sum(orn_all) / _N)

    # Clinical outcome flags
    n_seizures = round(_N * 0.65)
    n_dre      = round(_N * 0.25)
    n_idd      = round(_N * 0.75)
    n_se       = round(_N * 0.25)
    n_cerebral_oedema = round(_N * 0.55)
    n_protein_aversion = round(_N * 0.65)
    n_hepatomegaly     = round(_N * 0.45)

    return {
        "subtitle": (
            "OTC Deficiency (OTCD) — X-linked urea cycle disorder. "
            "OTC catalyses ornithine + carbamoyl phosphate → citrulline (step 2 of 5 in urea cycle). "
            "LOF: citrulline CANNOT be made; ammonia accumulates; carbamoyl phosphate overflows → "
            "orotic acid (urine). Most common urea cycle disorder (~1:14,000–50,000). "
            "Neonatal-onset males: hemizygous null, crisis day 1–5; ammonia >500 µmol/L. "
            "VPA ABSOLUTE CI — inhibits CPS1/NAGS, catastrophic hyperammonemia."
        ),
        "gene": "OTC",
        "chromosome": "Xp21.1",
        "protein_size": "354 aa; mitochondrial matrix homotrimer; ~40 kDa monomer",
        "omim_gene": "300461",
        "omim_disease": "311250",
        "inheritance": "X-linked (XL) — hemizygous males; carrier females 25% symptomatic",
        "cohort_n": _N,
        "seed": _SEED,
        "phenotype_distribution": phenotypes,
        "kpi": {
            "avg_plasma_ammonia_umol_l":   avg_amm,
            "avg_plasma_citrulline_umol_l": avg_cit,
            "avg_urine_orotic_acid_umol_mol": avg_orot,
            "avg_plasma_arginine_umol_l":  avg_arg,
            "avg_plasma_ornithine_umol_l": avg_orn,
            "pct_seizures":            round(100 * n_seizures / _N),
            "pct_dre":                 round(100 * n_dre / _N),
            "pct_idd":                 round(100 * n_idd / _N),
            "pct_status_epilepticus":  round(100 * n_se / _N),
            "pct_cerebral_oedema":     round(100 * n_cerebral_oedema / _N),
            "pct_protein_aversion":    round(100 * n_protein_aversion / _N),
            "pct_hepatomegaly":        round(100 * n_hepatomegaly / _N),
        },
        "biomarker_normals": {
            "description": "KEY NEGATIVE biomarkers — NORMAL in OTC (distinguishes from other diseases)",
            "plp_normal":       "NORMAL — OTC NOT PLP-dependent (KEY NEG vs PNPO/ALDH4A1)",
            "alpha_aasa_normal": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE",
            "pipecolic_normal":  "NORMAL — KEY NEGATIVE vs PDE/peroxisomal",
            "thcy_normal":       "NORMAL — KEY NEGATIVE vs CBS/MTHFR",
            "mma_normal":        "NORMAL — KEY NEGATIVE vs MMUT/cblA/cblB",
            "gaba_normal":       "NORMAL — KEY NEGATIVE vs ABAT (HIGH) / GAD1 (LOW)",
            "ghb_normal":        "NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) GHB HIGH",
            "glycine_normal":    "NORMAL — KEY NEGATIVE vs NKH/GLDC/AMT/GCSH",
            "gaa_normal":        "NORMAL — KEY NEGATIVE vs GAMT (50–300 µmol/L)",
            "creatine_normal":   "NORMAL — KEY NEGATIVE vs GAMT/AGAT/SLC6A8",
        },
        "key_positives": {
            "ammonia_critically_high": "Plasma ammonia >200 µmol/L in crisis (>500 neonatal) — PATHOGNOMONIC",
            "citrulline_critically_low": "Plasma citrulline <5 µmol/L (normal 15–35) — PATHOGNOMONIC",
            "urine_orotic_acid_high": "Urine orotic acid >10–20 µmol/mol Cr — PATHOGNOMONIC",
            "ctc_vs_cps1_key_distinction": (
                "Orotic acid HIGH in OTC (carbamoyl-P overflows to pyrimidines) "
                "vs NORMAL in CPS1 (carbamoyl-P not made — no overflow) — SINGLE KEY DISTINGUISHING TEST"
            ),
        },
        "pathway_context": {
            "step": "Step 2 of 5 in urea cycle (mitochondrial)",
            "reaction": "Ornithine + Carbamoyl-phosphate → Citrulline + Pi",
            "block_consequence": "Citrulline cannot be made; carbamoyl-P overflows → orotic acid",
            "upstream_block_vs_CPS1": "CPS1 blocks step 1 (carbamoyl-P synthesis); no overflow → orotic normal",
            "downstream_comparison": "ASS1 blocks step 3 → citrulline VERY HIGH (opposite of OTC)",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_breakdown():
    rng = _rng()

    n_neonatal   = round(_N * 0.40)
    n_late_onset = round(_N * 0.35)
    n_females    = _N - n_neonatal - n_late_onset

    # Seizure type breakdown
    n_seizures = round(_N * 0.65)
    seizure_types = {
        "GTCS (acute hyperammonemic crisis)": {"pct": 45, "n": round(n_seizures * 0.45)},
        "Focal seizures":                     {"pct": 30, "n": round(n_seizures * 0.30)},
        "Myoclonic (encephalopathy)":         {"pct": 20, "n": round(n_seizures * 0.20)},
        "Absence-like (metabolic)":           {"pct": 15, "n": round(n_seizures * 0.15)},
        "Status epilepticus":                 {"pct": 25, "n": round(n_seizures * 0.25)},
    }

    # Trigger distribution
    triggers = {
        "Febrile illness / infection": {"pct": 75, "detail": "Protein catabolism → ammonia surge"},
        "High protein meal":           {"pct": 68, "detail": "Nitrogen load exceeds OTC capacity"},
        "Prolonged fasting":           {"pct": 60, "detail": "Muscle catabolism → endogenous nitrogen"},
        "Post-surgical catabolism":    {"pct": 48, "detail": "Systemic stress → protein breakdown"},
        "Valproate exposure":          {"pct": 40, "detail": "CPS1/NAGS inhibition → crisis"},
        "Corticosteroid therapy":      {"pct": 35, "detail": "Catabolic hormone → protein breakdown"},
        "Pregnancy / post-partum":     {"pct": 28, "detail": "Symptomatic females; post-partum crisis"},
        "Growth spurt":                {"pct": 22, "detail": "Increased protein turnover"},
    }

    # Treatment efficacy
    treatments = {
        "Sodium benzoate + phenylacetate (RAVICTI/BUPHENYL)": {
            "level": "A", "category": "Nitrogen scavenger", "efficacy_pct": 85,
            "mechanism": "Conjugate glycine (benzoate → hippurate) + glutamine (phenylacetate → PAG); excrete N",
        },
        "Citrulline supplementation (150–200 mg/kg/day)": {
            "level": "A", "category": "Urea cycle replenishment", "efficacy_pct": 82,
            "mechanism": "Replenishes citrulline downstream of OTC block; metabolised to arginine",
        },
        "Arginine IV (acute) + oral (chronic)": {
            "level": "A", "category": "Essential amino acid", "efficacy_pct": 80,
            "mechanism": "Becomes conditionally essential in OTC; replenishes arginine cycle",
        },
        "Liver transplantation": {
            "level": "A", "category": "Curative (OTC hepatic enzyme)", "efficacy_pct": 98,
            "mechanism": "Restores hepatic OTC; normalises urea cycle; does NOT reverse neurological damage",
        },
        "Haemodialysis / CRRT (acute crisis)": {
            "level": "A", "category": "Ammonia removal", "efficacy_pct": 90,
            "mechanism": "For ammonia >500 µmol/L unresponsive to pharmacotherapy; emergent",
        },
        "IV Dextrose 10% (acute)": {
            "level": "A", "category": "Anti-catabolic", "efficacy_pct": 70,
            "mechanism": "Suppress catabolism; provide calories without nitrogen",
        },
        "Levetiracetam (LEV)": {
            "level": "B", "category": "AED — seizure control", "efficacy_pct": 65,
            "mechanism": "SV2A; no hepatotoxicity; safe in OTC; first-line AED choice",
        },
        "Benzodiazepines (acute SE)": {
            "level": "B", "category": "Acute SE", "efficacy_pct": 60,
            "mechanism": "GABA-A agonism; rescue treatment for acute status epilepticus",
        },
        "Carnitine (L-carnitine)": {
            "level": "C", "category": "Secondary support", "efficacy_pct": 35,
            "mechanism": "Some carnitine depletion in metabolic crisis; replace if low",
        },
    }

    # Drug risks
    drug_risks = {
        "Valproate (VPA)": {
            "risk": "ABSOLUTE CI",
            "detail": (
                "Inhibits CPS1 and NAGS (N-acetylglutamate synthase — CPS1 activator) "
                "+ impairs mitochondrial beta-oxidation. Even therapeutic VPA → hyperammonemic crisis. "
                "Multiple fatalities reported. CI in ALL urea cycle disorders. "
                "Mechanism: VPA-CoA derivatives inhibit NAGS → less N-acetylglutamate → less CPS1 activation "
                "→ carbamoyl-P cannot be made → worsens the upstream bottleneck in OTC deficiency."
            ),
        },
        "High protein diet": {
            "risk": "ABSOLUTE CI",
            "detail": "Nitrogen overload → ammonia exceeds OTC residual capacity → crisis",
        },
        "L-Asparaginase (chemotherapy)": {
            "risk": "ABSOLUTE CI",
            "detail": (
                "Depletes plasma asparagine → hyperammonemia via aspartate/asparagine depletion; "
                "combined with OTC block = catastrophic ammonia elevation; use alternative chemo if possible"
            ),
        },
        "Prolonged fasting / catabolism": {
            "risk": "HIGH RISK",
            "detail": "Internal protein catabolism → endogenous nitrogen load; pre-emptive glucose + nitrogen scavenger",
        },
        "Systemic glucocorticoids (high dose)": {
            "risk": "HIGH RISK",
            "detail": "Catabolic hormones → muscle protein breakdown → nitrogen load; avoid or use lowest dose",
        },
        "Topiramate": {
            "risk": "MODERATE RISK",
            "detail": "Case reports of hyperammonemia; carbonic anhydrase inhibition; avoid combining with VPA",
        },
        "Acetaminophen / paracetamol (chronic)": {
            "risk": "HIGH RISK",
            "detail": "Hepatotoxicity risk in pre-existing liver disease (OTC patients with steatosis/elevated LFTs)",
        },
        "N2O / nitrous oxide (anaesthesia)": {
            "risk": "MODERATE RISK",
            "detail": "Inactivates methionine synthase; exacerbates nitrogen handling; use alternative agents; ensure arginine perioperatively",
        },
    }

    # Variant distribution
    variants = [
        {"name": "p.Arg320His",   "domain": "Ornithine binding",             "pct": 15, "severity": "Severe (neonatal)", "responsive": False},
        {"name": "p.Glu228Lys",   "domain": "Carbamoyl-P binding",           "pct": 12, "severity": "Severe (neonatal)", "responsive": False},
        {"name": "p.Arg129His",   "domain": "Active site",                   "pct": 11, "severity": "Severe (neonatal)", "responsive": False},
        {"name": "p.Ile172Thr",   "domain": "MTS processing region",         "pct": 10, "severity": "Severe",            "responsive": False},
        {"name": "p.Asp263Gly",   "domain": "Active site",                   "pct": 9,  "severity": "Severe (neonatal)", "responsive": False},
        {"name": "c.IVS6+1G>A",   "domain": "Splice null (exon 6 skip→NMD)", "pct": 8,  "severity": "Severe",            "responsive": False},
        {"name": "p.Ala23Thr",    "domain": "MTS/signal region",             "pct": 7,  "severity": "Mild (late-onset)", "responsive": True},
        {"name": "p.Thr329Lys",   "domain": "Trimer interface",              "pct": 6,  "severity": "Moderate",          "responsive": True},
    ]

    # Differentials
    differentials = [
        {
            "disease": "CPS1 deficiency",
            "key_diff": "Urine orotic acid NORMAL (vs HIGH in OTC) — SINGLE key test; otherwise identical",
            "distinguishing": "Orotic acid",
        },
        {
            "disease": "ASS1 deficiency (citrullinemia I)",
            "key_diff": "Citrulline VERY HIGH (>100 µmol/L vs <5 in OTC) — OPPOSITE direction",
            "distinguishing": "Plasma citrulline direction",
        },
        {
            "disease": "OAT deficiency (gyrate atrophy)",
            "key_diff": "Ammonia NORMAL in OAT (KEY NEGATIVE); ornithine VERY HIGH (400–1500 vs mildly elevated in OTC)",
            "distinguishing": "Ammonia + ornithine level",
        },
        {
            "disease": "GLUD1 GoF (hyperinsulinism-hyperammonemia)",
            "key_diff": "GLUD1: hypoglycemia + hyperinsulinism; citrulline NORMAL; orotic NORMAL; ammonia 100–500 not >1000",
            "distinguishing": "Glucose, insulin, citrulline",
        },
        {
            "disease": "NAGS deficiency",
            "key_diff": "Orotic acid NORMAL (like CPS1); responds to N-carbamylglutamate (diagnostic/therapeutic trial)",
            "distinguishing": "Orotic acid + NCG response",
        },
        {
            "disease": "HHH syndrome (SLC25A15)",
            "key_diff": "Homocitrullinuria in HHH; ornithine also HIGH; OTC has no homocitrullinuria",
            "distinguishing": "Homocitrullinuria",
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
            "Neonatal-Onset Males": {
                "n": n_neonatal,
                "pct": round(100 * n_neonatal / _N),
                "description": "Hemizygous null; no OTC residual; ammonia >500 µmol/L day 1–5; seizures + coma; ~50% survive to discharge with aggressive treatment",
                "avg_peak_ammonia_umol_l": 850,
                "seizure_rate_pct": 80,
            },
            "Late-Onset Males": {
                "n": n_late_onset,
                "pct": round(100 * n_late_onset / _N),
                "description": "Residual 5–25% OTC activity; protein aversion; first crisis triggered by illness/protein; range neonatal to adult presentation",
                "avg_peak_ammonia_umol_l": 280,
                "seizure_rate_pct": 55,
            },
            "Symptomatic Females": {
                "n": n_females,
                "pct": round(100 * n_females / _N),
                "description": "Skewed X-inactivation (Lyon effect); late-onset typical; protein aversion; occasional severe crisis in post-partum period",
                "avg_peak_ammonia_umol_l": 160,
                "seizure_rate_pct": 40,
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "disease": "OTC Deficiency (Ornithine Transcarbamylase Deficiency, OTCD)",
        "omim_disease": "311250",
        "omim_gene": "300461",
        "gene": "OTC",
        "chromosome": "Xp21.1",
        "inheritance": "X-linked (XL); hemizygous males; carrier females ~25% symptomatic (Lyon/skewed X-inactivation)",
        "prevalence": "~1:14,000–50,000; most common urea cycle disorder worldwide",
        "enzyme": "Ornithine Transcarbamylase (OTC); mitochondrial matrix homotrimer; 354 aa; ~40 kDa monomer",
        "reaction": "Ornithine + Carbamoyl phosphate → Citrulline + Pi  [urea cycle step 2 of 5]",
        "pathway_position": "Mitochondrial matrix; step 2 of 5 in urea cycle; downstream of CPS1 (step 1), upstream of ASS1 (step 3)",
        "mechanism_of_disease": (
            "OTC LOF → citrulline cannot be made → urea cycle blocked → ammonia accumulates → "
            "carbamoyl phosphate overflows into cytoplasm → enters pyrimidine synthesis pathway → "
            "orotic acid produced and excreted in urine (PATHOGNOMONIC). "
            "Glutamine and alanine elevated as alternative ammonia disposal routes."
        ),
        "key_biomarker_positives": {
            "plasma_ammonia": "CRITICALLY HIGH (>200 µmol/L crisis; >500 neonatal; normal <50) — PATHOGNOMONIC",
            "plasma_citrulline": "CRITICALLY LOW (<5 µmol/L; normal 15–35) — PATHOGNOMONIC; opposite of ASS1",
            "urine_orotic_acid": "HIGH (>10–20 µmol/mol Cr; normal <6) — PATHOGNOMONIC; KEY vs CPS1 (NORMAL in CPS1)",
            "plasma_arginine": "LOW (<25 µmol/L; normal 60–120) — downstream product absent",
            "plasma_glutamine": "ELEVATED (600–900 µmol/L) — ammonia detox via glutamine synthetase",
            "plasma_alanine": "ELEVATED (500–800 µmol/L) — alternative ammonia disposal via ALT",
        },
        "key_biomarker_negatives": {
            "plp": "NORMAL — OTC not PLP-dependent (KEY NEG vs PNPO/ALDH4A1 where PLP LOW)",
            "alpha_aasa": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE",
            "pipecolic": "NORMAL — KEY NEGATIVE vs PDE/peroxisomal disorders",
            "thcy": "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR",
            "mma": "NORMAL — KEY NEGATIVE vs methylmalonic acidemia",
            "gaba_csf": "NORMAL — KEY NEGATIVE vs ABAT (HIGH) and GAD1 (LOW)",
            "ghb_urine": "NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB dramatically HIGH",
            "glycine": "NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH) where glycine HIGH",
            "gaa": "NORMAL — KEY NEGATIVE vs GAMT (GAA 50–300 µmol/L PATHOGNOMONIC)",
            "creatine": "NORMAL — KEY NEGATIVE vs GAMT/AGAT/SLC6A8",
        },
        "critical_otc_vs_cps1_distinction": (
            "Urine orotic acid: HIGH in OTC (carbamoyl-P overflows to pyrimidines) "
            "but NORMAL in CPS1 (no carbamoyl-P produced, so no overflow). "
            "This is the SINGLE biochemical test that distinguishes OTC from CPS1 deficiency."
        ),
        "absolute_contraindications": {
            "valproate_vpa": (
                "ABSOLUTE CI — inhibits CPS1 + NAGS (N-acetylglutamate synthase) + mitochondrial beta-oxidation. "
                "Even low-dose VPA → hyperammonemic crisis in OTC. Multiple fatalities. "
                "VPA-CoA → inhibits NAGS → less N-acetylglutamate → less CPS1 activation → worsens OTC block. "
                "CI extends to ALL urea cycle disorders (OTC, CPS1, ASS1, ASL, ARG1, NAGS)."
            ),
            "high_protein_diet": "ABSOLUTE CI — nitrogen overload → ammonia surge",
            "l_asparaginase": "ABSOLUTE CI — hyperammonemia via asparagine depletion; combined with OTC block = catastrophic",
        },
        "first_line_treatments": {
            "nitrogen_scavengers": "Sodium benzoate + phenylacetate (RAVICTI/BUPHENYL) — Level A chronic",
            "citrulline": "Citrulline supplementation 150–200 mg/kg/day — Level A",
            "liver_transplant": "Curative — Level A; restores hepatic OTC; normalise urea cycle",
            "acute_ammonia_removal": "Haemodialysis/CRRT for ammonia >500 µmol/L — Level A",
            "aed": "Levetiracetam (LEV) — Level B first-line AED (no hepatotoxicity, no ammonia effect)",
        },
        "seizure_mechanism": (
            "Ammonia disrupts GABA-A receptor gating (inhibitory) and over-activates NMDA receptors (excitatory). "
            "Net result: reduced inhibition + increased excitation = seizure threshold lowered. "
            "High ammonia also causes cerebral oedema → secondary intracranial hypertension → seizures. "
            "EEG: burst-suppression (severe), triphasic waves (ammonia encephalopathy), diffuse slowing."
        ),
        "x_linkage_note": (
            "Males: hemizygous; null mutations → neonatal crisis; hypomorphic → late-onset. "
            "Females: carrier heterozygotes; 75% asymptomatic; 25% symptomatic via skewed X-inactivation (Lyon effect). "
            "Female symptomatic risk highest in: pregnancy, post-partum period, illness, high protein intake. "
            "Female carriers: protein aversion is the key clinical clue (often undiagnosed for years)."
        ),
    }
