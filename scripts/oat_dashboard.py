#!/usr/bin/env python3
"""OAT (Ornithine Aminotransferase) Epilepsy Dashboard.

OAT encodes Ornithine Aminotransferase, a PLP-dependent mitochondrial enzyme:
  L-Ornithine + α-Ketoglutarate  →  L-Glutamate-γ-semialdehyde (GSA = P5C) + L-Glutamate

OAT DISEASE: Gyrate Atrophy of the Choroid and Retina
  OMIM Disease: #258870   Gene: OAT, OMIM *258870
  Chromosome: 10q26.13
  Inheritance: Autosomal Recessive — Loss-of-Function (biallelic)
  Protein: 439 aa; mitochondrial matrix homotetramer; PLP-dependent; ~46 kDa monomer
  Prevalence: ~300–500 cases worldwide 2026; Finnish founder enrichment (~1:50,000 Finland)

MECHANISM — LOSS-OF-FUNCTION (ornithine catabolism block → ornithine accumulates):
  Normal OAT reaction: L-Ornithine + α-KG → GSA/P5C + Glutamate  (mitochondrial)
  OAT LOF: Ornithine CANNOT be catabolised → ornithine accumulates → 400–1,500 µmol/L
  P5C (from ornithine route) NOT produced via OAT  →  mild proline reduction possible
  BUT: ALDH18A1 still makes P5C from glutamate → proline synthesis only mildly impaired
  Ornithine is retinotoxic at very high plasma levels → progressive choroidal/retinal atrophy
  B6-responsive subset (~30–40%): pyridoxine restores partial OAT activity via PLP augmentation

POSITION IN ORNITHINE/PROLINE CYCLE:
  Glutamate → [ALDH18A1/P5CS] → P5C ← [OAT, blocked] ← Ornithine
  P5C → [PYCR1/PYCR2] → L-Proline
  Ornithine → [AGAT] → Guanidinoacetate → [GAMT] → Creatine (separate route)
  Arginine → [Arginase] → Ornithine + Urea  ← ARGININE RESTRICTION targets this step

  Key pathway comparison in proline/ornithine group:
    ALDH18A1 LOF: Glutamate → P5C BLOCKED → proline CRITICALLY LOW + ornithine LOW
    OAT LOF:      Ornithine → P5C BLOCKED → ornithine DRAMATICALLY HIGH (not ALDH18A1's problem)
    PRODH LOF:    Proline → P5C BLOCKED → proline ELEVATED
    ALDH4A1 LOF:  P5C → glutamate BLOCKED → P5C ELEVATED + PLP inactivated
    PYCR1/2 LOF:  P5C → proline BLOCKED → proline LOW (similar to ALDH18A1 downstream)

OAT BIOCHEMISTRY (LOF → ornithine accumulates):
  Plasma ornithine: VERY HIGH (400–1,500 µmol/L; normal <200 µmol/L) — PATHOGNOMONIC
  Urine ornithine: HIGH (aminoaciduria; ornithine overflow)
  Plasma proline: LOW-NORMAL (30–150 µmol/L; mildly reduced; less severe than ALDH18A1)
  Plasma glutamate: NORMAL-LOW (OAT makes glutamate from ornithine; blocked in OAT LOF)
  PLP (plasma): NORMAL (OAT is PLP-dependent but LOF does NOT inactivate PLP itself;
                enzyme absent, not PLP; KEY NEGATIVE vs ALDH4A1 where P5C inactivates PLP)
  Pyridoxal (plasma): NORMAL — KEY NEGATIVE vs PNPO (where PLP synthesis enzyme absent)
  alpha-AASA (urine): NORMAL — KEY NEGATIVE vs ALDH7A1/PDE (>30 mmol/mol Cr in PDE)
  Pipecolic acid: NORMAL — KEY NEGATIVE vs ALDH7A1/PDE and peroxisomal disorders
  tHcy: NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR/MTRR homocysteine disorders
  MMA: NORMAL — KEY NEGATIVE vs methylmalonic acidemia (MMUT/cblB)
  GABA: NORMAL — KEY NEGATIVE vs ABAT (GABA dramatically high) and GAD1 (GABA critically low)
  GHB (urine): NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB dramatically high
  CSF glucose: NORMAL — KEY NEGATIVE vs GLUT1 deficiency
  Creatine (plasma): NORMAL-LOW (AGAT uses ornithine as substrate; elevated ornithine
                     may partially impair creatine synthesis; supplemental creatine may help)
  GAA (guanidinoacetate): NORMAL-LOW in some (see creatine note; usually within range)

RETINAL/OPHTHALMOLOGIC FEATURES (dominant non-seizure manifestation):
  Gyrate atrophy of choroid and retina: 100% — PATHOGNOMONIC for OAT deficiency
    Circular/scalloped patches of chorioretinal atrophy starting in periphery
    Progressive centripetal spread → central visual field loss over decades
    Night blindness (nyctalopia): onset typically 5–15 years; often first symptom
    Visual field constriction: tunnel vision → legal blindness by 40–55 years (untreated)
    ERG: rod-dominated dysfunction (scotopic loss > photopic)
    OCT: thinning of outer nuclear layer and photoreceptor layer in affected zones
  Posterior subcapsular cataracts: 50–70% (by adulthood)
  High myopia (>6 D): 70–80%
  Vitreous haze / vitreous floaters: 20–30%

EPILEPSY IN OAT DEFICIENCY (secondary to hyperornithinemia):
  Overall seizure rate: ~30–40% (lower than most metabolic epilepsies;
    ornithine has NMDA agonist-like effects at high levels → cortical hyperexcitability)
  Focal seizures: ~40% of those with epilepsy (MODAL seizure type in OAT)
  Absence seizures: ~35% (generalised; ornithine disrupts GABA/glutamate balance)
  GTCS: ~30%
  Myoclonic: ~15%
  Infantile spasms: ~5% (rare; severe neonatal phenotype or very high ornithine)
  Drug-resistant epilepsy: ~15–20% (lower than most metabolic epilepsies)
  EEG: focal or multifocal epileptiform; photosensitivity occasional (retinal dysfunction)
  MRI: usually normal; mild white matter changes in severe cases; NO gyrate pattern on MRI
       (gyrate atrophy is a retinal/choroidal finding, NOT a brain MRI pattern)

NON-SEIZURE NEUROLOGIC FEATURES:
  Intellectual disability: ~25–35% (mild-moderate; NOT the dominant feature in OAT)
  Proximal myopathy: 30–40% (type 2 muscle fibre atrophy; tubular aggregates on biopsy;
                     ornithine toxic to muscle mitochondria)
  Fine/sparse/dry hair: 20–30%

B6-RESPONSIVE SUBSET (~30–40% of OAT patients):
  Large-dose pyridoxine (300–1,200 mg/day) → PLP augmentation → partial OAT activity restored
  Plasma ornithine may fall 40–80% in good responders
  Retinal progression significantly slower; prognosis markedly better
  Mandatory trial in ALL new OAT patients before finalising management plan

TREATMENTS:
  Arginine restriction (dietary): Level A PRIMARY
    - Arginine → ornithine (via arginase); restricting dietary arginine reduces ornithine input
    - Arginine-free or very-low-arginine amino acid formula + restricted protein
    - Plasma ornithine falls 30–60% with strict adherence; retinal progression slows
    - Cannot fully normalise ornithine; combined approach with B6 (if responsive) is optimal
  Pyridoxine (B6): Level A (for B6-responsive subset ~30–40%)
    - PLP trial MANDATORY at diagnosis: 300–1,200 mg/day × 6 weeks minimum
    - Responders: ornithine falls 40–80%; slower retinal progression; reduce to maintenance dose
    - Non-responders: continue arginine restriction ± creatine ± lysine; B6 not continued
    - Toxicity: sensory neuropathy (EMG monitoring >300 mg/day chronic use)
  Creatine supplementation: Level B
    - Ornithine competes with AGAT substrate; supplemental creatine may bypass impaired synthesis
    - Benefit for myopathy documented; some seizure improvement reported
  Lysine supplementation: Level B (controversial)
    - Ornithine and lysine share renal tubular transporter (SLC7A6/SLC7A9)
    - Lysine loading → increased ornithine urinary excretion; may lower plasma ornithine 10–20%
    - Evidence limited; used adjunctively to arginine restriction
  Proline supplementation: Level C
    - Plasma proline mildly low; supplementation may support collagen synthesis
    - Low-level evidence; not standard of care
  Levetiracetam (LEV): Level B (for seizures)
    - SV2A mechanism; GABA-independent; safe; first-line AED in OAT
  Valproate (VPA): Level B / MODERATE RISK
    - Effective AED; but inhibits OTC (ornithine transcarbamylase) → hyperammonemia risk
    - Use with monitoring of ammonia; avoid in patients with already high ornithine + ammonia
  Annual ophthalmology review: MANDATORY
    - Annual fundoscopy + ERG + OCT + visual field perimetry
    - Document and track gyrate atrophy progression; adjust treatment if rapid worsening
    - ERG detects subclinical progression before patient-reported vision loss

ABSOLUTE CONTRAINDICATIONS:
  High-arginine diet/supplements: ABSOLUTE CI — arginine is the immediate ornithine precursor
    (arginase: arginine → ornithine + urea); high arginine → dramatic ornithine surge → retinal
    toxicity acceleration; high-protein animal foods (beef, fish, eggs, nuts) = arginine-rich loads
  Isoniazid (INH): ABSOLUTE CI (B6-responsive patients)
    PLP antagonist → abolishes any residual OAT activity → ornithine cannot be cleared by partial
    OAT activity; for B6-NON-responsive patients: HIGH RISK (no residual activity to abolish, but
    PLP depletion still harmful to other PLP-dependent reactions)
  Arginine supplementation: ABSOLUTE CI — marketed for immune / wound-healing / sports use;
    dramatically worsens ornithine accumulation; widely available as supplement (danger)

HIGH-RISK (contextual):
  High-protein diet (general): HIGH RISK — arginine content in animal proteins; even moderate
    protein restriction benefit; arginine-free formula provides essential AAs without arginine load
  Valproate (high dose): HIGH RISK — OTC inhibition → hyperammonemia + hepatotoxicity; safer AEDs
    (LEV, LTG, TPM) preferred; if VPA used, ammonia monitoring mandatory
  N2O (nitrous oxide): MODERATE RISK — inactivates methionine synthase (MTR); no direct OAT
    interaction; general anaesthetic caution in metabolic disorders

VARIANTS (OAT — PLP-binding, ornithine channel, tetramer interface):
  p.Leu402Pro: PLP-binding β-barrel, Finnish founder, ~30% of Finnish patients, severe, NOT B6-responsive
  p.Arg180Thr: PLP Schiff-base lysine adjacent, ~20%, severe, NOT B6-responsive
  p.Glu318Lys: Active site, ~15%, moderate-severe
  p.Arg154Cys: Tetramer interface, ~12%, moderate, sometimes B6-responsive
  p.Gly237Val: Ornithine-binding substrate channel, ~10%, moderate
  c.IVS6+1G>A: Splice null (intron 6 → exon 6 skip → NMD), ~9%, severe, NOT B6-responsive
  p.Val332Met: B6-responsive; PLP binding partially preserved, ~8%, mild-moderate
  p.Thr181Ile: Hypomorphic; B6-responsive; ~6%, mild; Finnish compound heterozygous with Leu402Pro

PHENOTYPE CLASSES:
  Classic-Severe (60%): biallelic null/severe; no B6 response; ornithine >800 µmol/L;
                         legal blindness by 45–55 years untreated; myopathy in 40%; seizures 35%
  B6-Responsive (30%): at least one hypomorphic allele; ornithine partially controlled;
                        slower retinal progression; later legal blindness (>60y if compliant)
  Mild-Attenuated (10%): partial OAT activity (>10% residual); late-onset gyrate atrophy;
                          mild hyperornithinemia (200–400 µmol/L); seizures rare

DIFFERENTIAL DIAGNOSES (diseases with similar findings):
  ALDH18A1 deficiency (P5CS deficiency): Proline CRITICALLY LOW + ornithine LOW (both LOW in P5C
    pathway; OAT = ornithine HIGH); cutis laxa; cataracts common
  PYCR1/2 deficiency: Proline LOW; ornithine normal-low; no retinal atrophy; cutis laxa (PYCR1)
  AGAT deficiency: Guanidinoacetate (GAA) LOW + creatine LOW + ornithine NORMAL; no retinal atrophy
  OTC deficiency: Ornithine HIGH + ammonia HIGH + citrulline LOW (urea cycle block; completely
    different to OAT where ammonia NORMAL); citrulline and ammonia clarify
  Choroideremia (CHM): X-linked; fundus similar to gyrate atrophy; ornithine NORMAL; CHM mutation
  Stargardt disease (ABCA4): macular atrophy; ornithine NORMAL; different fundus pattern
  Bardet-Biedl syndrome: rod-cone dystrophy; ornithine NORMAL; obesity, polydactyly present
"""
import random

_N    = 40    # cohort size (consistent with all expert dashboards)
_SEED = 187   # deterministic seed (GAD1=181, next +6=187)


def _rng():
    return random.Random(_SEED)


# ──────────────────────────────────────────────────────────────────────────────
def get_overview():
    rng = _rng()

    # Phenotype distribution
    n_severe   = round(_N * 0.60)   # Classic-Severe: biallelic null; no B6 response
    n_b6resp   = round(_N * 0.30)   # B6-Responsive: partial PLP augmentation
    n_mild     = _N - n_severe - n_b6resp  # Mild-Attenuated: >10% residual OAT activity

    phenotypes = {
        "Classic-Severe":  {"n": n_severe,  "pct": round(100 * n_severe  / _N)},
        "B6-Responsive":   {"n": n_b6resp,  "pct": round(100 * n_b6resp  / _N)},
        "Mild-Attenuated": {"n": n_mild,    "pct": round(100 * n_mild    / _N)},
    }

    # Biomarker distributions (OAT LOF → ornithine accumulates)
    orn_severe = [rng.uniform(600, 1500) for _ in range(n_severe)]
    orn_b6resp = [rng.uniform(300, 700)  for _ in range(n_b6resp)]   # partially controlled
    orn_mild   = [rng.uniform(200, 400)  for _ in range(n_mild)]
    all_orn = orn_severe + orn_b6resp + orn_mild
    avg_orn = round(sum(all_orn) / _N)

    plp_vals   = [rng.uniform(24, 58) for _ in range(_N)]   # NORMAL — PLP not inactivated
    proline_vals = [rng.uniform(30, 150) for _ in range(_N)]  # LOW-NORMAL
    avg_plp    = round(sum(plp_vals) / _N, 1)
    avg_pro    = round(sum(proline_vals) / _N)

    pct_seizures   = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.35) / _N)
    pct_focal      = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.40) / _N)
    pct_absence    = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.35) / _N)
    pct_dre        = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.18) / _N)
    pct_idd        = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.30) / _N)
    pct_myopathy   = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.35) / _N)
    pct_cataract   = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.60) / _N)
    pct_myopia     = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.75) / _N)

    return {
        "gene": "OAT",
        "subtitle": (
            "OAT Deficiency — Ornithine Catabolism Blocked (Gyrate Atrophy) — "
            "Plasma ornithine dramatically elevated → progressive chorioretinal atrophy + epilepsy"
        ),
        "chromosome": "10q26.13",
        "protein_size": "439 aa; mitochondrial matrix homotetramer; PLP-dependent; ~46 kDa monomer",
        "omim_gene": "*258870",
        "omim_disease": "#258870",
        "prevalence": "~300–500 cases worldwide 2026; Finnish founder enrichment (~1:50,000 Finland)",
        "inheritance": "Autosomal Recessive — Loss-of-Function (biallelic); B6-responsive subset ~30–40%",
        "cohort_n": _N,
        "function": (
            "OAT catalyses: L-Ornithine + α-Ketoglutarate → Glutamate-γ-semialdehyde (P5C/GSA) + L-Glutamate. "
            "OAT LOF → ornithine CANNOT be catabolised → ornithine accumulates 400–1,500 µmol/L. "
            "P5C (from ornithine route) not produced via OAT; ALDH18A1 still makes P5C from glutamate "
            "so proline synthesis only mildly impaired. "
            "High ornithine is retinotoxic → progressive gyrate atrophy of choroid and retina."
        ),
        "mechanism": (
            "PATHWAY: Arginine → [Arginase] → Ornithine + Urea → [OAT, BLOCKED] → P5C → [PYCR1/2] → Proline. "
            "OAT block: ornithine cannot enter P5C/proline synthesis (via this route). "
            "ALDH18A1 still synthesises P5C from glutamate → proline only mildly reduced (unlike ALDH18A1 LOF). "
            "Arginine restriction targets Arginase step to reduce ornithine input. "
            "B6-responsive subset: PLP augmentation partially restores hypomorphic OAT → ornithine falls 40–80%."
        ),
        "key_positive_features": (
            "Plasma ornithine VERY HIGH ({orn} µmol/L avg; normal <200 µmol/L) — PATHOGNOMONIC. "
            "Gyrate atrophy of choroid/retina on fundoscopy — 100% — PATHOGNOMONIC clinical sign. "
            "Urine ornithine HIGH (aminoaciduria). "
            "Plasma proline LOW-NORMAL ({pro} µmol/L avg; mildly reduced)."
        ).format(orn=avg_orn, pro=avg_pro),
        "key_negative_features": (
            "PLP NORMAL (enzyme absent, not PLP — KEY NEGATIVE vs ALDH4A1 where P5C inactivates PLP). "
            "alpha-AASA NORMAL (vs PDE/ALDH7A1 where >30 mmol/mol Cr). "
            "Pipecolic NORMAL. tHcy NORMAL. MMA NORMAL. GABA NORMAL (vs ABAT high, GAD1 low). "
            "GHB NORMAL (vs SSADH/ALDH5A1 where GHB dramatically high). "
            "Ammonia NORMAL (vs OTC/CPS1 urea cycle defects where ammonia HIGH). "
            "Citrulline NORMAL (vs urea cycle disorders). "
            "CSF glucose NORMAL (vs GLUT1 deficiency). "
            "MRI brain usually NORMAL (gyrate = retinal/choroidal, NOT brain MRI pattern)."
        ),
        "phenotype_distribution": phenotypes,
        "kpi": {
            "avg_plasma_ornithine_umol_l": avg_orn,
            "avg_proline_umol_l": avg_pro,
            "avg_plp_nmol_l": avg_plp,
            "pct_seizures": pct_seizures,
            "pct_focal_seizures": pct_focal,
            "pct_absence_seizures": pct_absence,
            "pct_dre": pct_dre,
            "pct_idd": pct_idd,
            "pct_proximal_myopathy": pct_myopathy,
            "pct_cataracts": pct_cataract,
            "pct_high_myopia": pct_myopia,
        },
        "nbs_primary": (
            "NOT on standard newborn screening panels; ornithine aminoacidogram (plasma amino acids) "
            "required — plasma ornithine >400 µmol/L is diagnostic flag. "
            "NBS extended panels may detect hyperornithinemia in some programmes."
        ),
        "nbs_secondary": (
            "Urine amino acids (ornithine overflow); ophthalmology referral for fundoscopy + ERG; "
            "OAT sequencing (molecular confirmation); pyridoxine trial (B6-responsiveness assessment); "
            "OAT enzyme activity in fibroblasts (research/specialty)."
        ),
        "pathway_position": (
            "Ornithine/proline cycle: ALDH18A1 makes P5C from Glutamate; OAT makes P5C from Ornithine (this step blocked); "
            "PYCR1/2 convert P5C → Proline. PRODH + ALDH4A1 catabolise Proline → P5C → Glutamate. "
            "OAT LOF: ornithine piles up; P5C from ornithine route absent; proline only mildly reduced "
            "because ALDH18A1 (P5CS) still provides P5C from glutamate."
        ),
        "b6_response_note": (
            "B6-RESPONSIVE SUBSET (~30–40%): pyridoxine 300–1,200 mg/day × ≥6 weeks → plasma ornithine "
            "falls 40–80% in good responders (variants with partial PLP binding preserved). "
            "Trial MANDATORY at diagnosis. Monitor with EMG if >300 mg/day chronic use (sensory neuropathy)."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_breakdown():
    rng = _rng()

    biomarkers = [
        {"name": "Plasma Ornithine",
         "value": "400–1,500 µmol/L (normal <200)",
         "direction": "VERY HIGH",
         "significance": "PATHOGNOMONIC — primary accumulating metabolite; retinotoxic at >300 µmol/L"},
        {"name": "Urine Ornithine",
         "value": "Elevated (aminoaciduria)",
         "direction": "HIGH",
         "significance": "Overflow aminoaciduria; ornithine exceeds tubular reabsorption capacity"},
        {"name": "Plasma Proline",
         "value": "30–150 µmol/L (normal 100–450)",
         "direction": "LOW-NORMAL",
         "significance": "Mildly reduced; P5C from ornithine route blocked; ALDH18A1 still provides P5C from glutamate"},
        {"name": "Plasma PLP",
         "value": "24–58 nmol/L (NORMAL range)",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs ALDH4A1 (where P5C inactivates PLP); OAT LOF does not inactivate PLP"},
        {"name": "Plasma Glutamate",
         "value": "Slightly low-normal",
         "direction": "NORMAL-LOW",
         "significance": "OAT makes glutamate from ornithine; OAT LOF reduces this glutamate source (minor effect)"},
        {"name": "Creatine (plasma)",
         "value": "NORMAL or low-normal",
         "direction": "NORMAL",
         "significance": "AGAT uses ornithine as substrate; elevated ornithine may partially impair AGAT; creatine supplementation may help"},
        {"name": "Guanidinoacetate (GAA)",
         "value": "NORMAL-LOW in some",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs GAMT deficiency (GAA HIGH in GAMT); AGAT competes with elevated ornithine"},
        {"name": "alpha-AASA",
         "value": "NORMAL",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs ALDH7A1/PDE (>30 mmol/mol Cr in pyridoxine-dependent epilepsy)"},
        {"name": "Pipecolic acid",
         "value": "NORMAL",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs ALDH7A1/PDE and peroxisomal disorders (elevated in both)"},
        {"name": "tHcy (total homocysteine)",
         "value": "NORMAL",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs CBS (100–500 µmol/L), MTHFR, MTR, MTRR remethylation disorders"},
        {"name": "MMA (methylmalonic acid)",
         "value": "NORMAL",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs MMUT/methylmalonic acidemia (MMA dramatically elevated there)"},
        {"name": "Plasma GABA",
         "value": "NORMAL",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs ABAT (GABA dramatically HIGH) and GAD1 (GABA critically LOW)"},
        {"name": "GHB (4-hydroxybutyrate)",
         "value": "NORMAL",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs SSADH/ALDH5A1 deficiency (GHB >1,000 mmol/mol Cr there)"},
        {"name": "Ammonia (plasma)",
         "value": "NORMAL",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs urea cycle disorders (OTC/CPS1/ASS1/ASL where ammonia HIGH)"},
        {"name": "Citrulline",
         "value": "NORMAL",
         "direction": "NORMAL",
         "significance": "KEY NEGATIVE vs urea cycle disorders (citrulline LOW in CPS1/OTC; HIGH in ASS1)"},
    ]

    clinical_features = [
        {"feature": "Gyrate atrophy of choroid and retina (fundoscopy)",
         "pct": 100, "note": "PATHOGNOMONIC; circular/scalloped chorioretinal atrophy patches; peripheral → central progression"},
        {"feature": "Night blindness (nyctalopia)",
         "pct": 95, "note": "Often first symptom; onset 5–15 years; rod-dominant ERG dysfunction; progresses over decades"},
        {"feature": "High myopia (>6 D)",
         "pct": 75, "note": "Present in most patients; often detected in childhood; predates gyrate atrophy fundus signs"},
        {"feature": "Posterior subcapsular cataracts",
         "pct": 60, "note": "Develop by adulthood; ornithine-toxic lens epithelium; may require surgery"},
        {"feature": "Progressive visual field constriction",
         "pct": 90, "note": "Tunnel vision; worsens with gyrate atrophy expansion; legal blindness by 40–55y untreated"},
        {"feature": "Proximal myopathy",
         "pct": 35, "note": "Type 2 muscle fibre atrophy; tubular aggregates on biopsy; ornithine toxic to muscle mitochondria"},
        {"feature": "Intellectual disability (mild-moderate)",
         "pct": 30, "note": "NOT the dominant feature; epilepsy, if present, contributes; less severe than ALDH18A1/PYCR"},
        {"feature": "Fine/sparse/dry hair",
         "pct": 25, "note": "Ornithine toxicity to hair follicles; mild cosmetic finding; underrecognised"},
        {"feature": "Epilepsy",
         "pct": 35, "note": "Focal or absence predominant; secondary to high ornithine CNS effects; less refractory than most metabolic epilepsies"},
        {"feature": "Drug-resistant epilepsy (DRE)",
         "pct": 18, "note": "Lower DRE rate than most metabolic epilepsies; seizures may improve with ornithine-lowering treatment"},
    ]

    seizure_types = [
        {"type": "Focal (partial) seizures", "pct": 40, "note": "MODAL type; cortical ornithine effects; frontotemporal often"},
        {"type": "Absence seizures", "pct": 35, "note": "Ornithine disrupts GABA/glutamate cortical balance; 3Hz-ish spike-wave"},
        {"type": "GTCS (generalised tonic-clonic)", "pct": 30, "note": "Secondary generalisation from focal or de novo generalised"},
        {"type": "Myoclonic", "pct": 15, "note": "Less common; if present, consider ALDH4A1 differential"},
        {"type": "Infantile spasms", "pct": 5, "note": "Rare; only in severe neonatal ornithine accumulation or very early onset"},
    ]

    treatments = [
        {"tx": "Arginine restriction (dietary)", "level": "A",
         "mechanism": "Arginine → ornithine via arginase; restricting arginine reduces ornithine input; arginine-free amino acid formula + restricted animal protein; plasma ornithine falls 30–60%"},
        {"tx": "Pyridoxine (B6) — B6-responsive patients", "level": "A",
         "mechanism": "PLP augmentation → partial restoration of hypomorphic OAT activity; 300–1,200 mg/day × ≥6 weeks trial; ornithine falls 40–80% in responders; MANDATORY trial at diagnosis"},
        {"tx": "Creatine supplementation", "level": "B",
         "mechanism": "AGAT uses ornithine; high ornithine may impair creatine synthesis via AGAT; creatine bypass improves myopathy and may modestly help seizures"},
        {"tx": "Lysine supplementation", "level": "B",
         "mechanism": "Ornithine and lysine share renal tubular transporter (SLC7A6); lysine loading → increased ornithine urinary excretion; may lower plasma ornithine 10–20%; adjunctive use"},
        {"tx": "Proline supplementation", "level": "C",
         "mechanism": "Plasma proline mildly low; supplementation may support collagen/muscle synthesis; limited evidence; not standard of care"},
        {"tx": "Levetiracetam (LEV)", "level": "B",
         "mechanism": "SV2A mechanism; GABA-independent; first-line AED in OAT; safe; good tolerability; preferred over VPA"},
        {"tx": "Lamotrigine (LTG)", "level": "B",
         "mechanism": "Sodium channel + AMPA; effective for focal and absence seizures; no hyperammonemia risk unlike VPA"},
        {"tx": "Valproate (VPA)", "level": "B / MODERATE RISK",
         "mechanism": "Broad-spectrum AED; effective for absence + GTCS; but inhibits OTC → hyperammonemia risk in ornithine-disordered patients; monitor ammonia; use with caution"},
        {"tx": "Annual ophthalmology monitoring", "level": "MANDATORY",
         "mechanism": "Annual fundoscopy + ERG + OCT + visual field perimetry; tracks gyrate atrophy progression; document and adjust treatment if rapid worsening; cataract surveillance"},
        {"tx": "Cataract surgery", "level": "A (when indicated)",
         "mechanism": "Posterior subcapsular cataracts → significant vision loss when dense; early surgery recommended; standard phacoemulsification; no OAT-specific surgical risk"},
    ]

    drug_risks = [
        {"drug": "High-arginine diet / supplements", "risk": "ABSOLUTE CI",
         "reason": "Arginine is the immediate ornithine precursor (arginase: arginine → ornithine + urea); high arginine → dramatic ornithine surge → retinal toxicity acceleration; high-protein animal foods (beef, fish, eggs, dairy, peanuts) are arginine-rich — restrict carefully"},
        {"drug": "Arginine supplements (marketed)", "risk": "ABSOLUTE CI",
         "reason": "Arginine marketed for sports performance, wound healing, immune function; any supplement form dramatically worsens ornithine accumulation in OAT deficiency"},
        {"drug": "Isoniazid (INH)", "risk": "ABSOLUTE CI (B6-responsive) / HIGH RISK (others)",
         "reason": "PLP antagonist → abolishes residual OAT activity in B6-responsive patients; all residual ornithine-lowering capacity lost; catastrophic ornithine surge; for B6-non-responsive: still harmful to other PLP-dependent reactions (GAD1/2, ABAT, etc.)"},
        {"drug": "Valproate (high dose)", "risk": "HIGH RISK",
         "reason": "OTC inhibition → hyperammonemia, especially with high baseline ornithine competing in the urea-ornithine exchange; use low doses if needed, always monitor ammonia"},
        {"drug": "High-protein diet (general)", "risk": "HIGH RISK",
         "reason": "Arginine content in animal proteins provides ornithine substrate; even moderate dietary protein restriction helpful; replace with arginine-free amino acid formula"},
        {"drug": "Cycloserine", "risk": "MODERATE RISK",
         "reason": "PLP antagonist (anti-tuberculosis agent); less immediately catastrophic than INH in B6-non-responsive but still harmful to PLP-dependent enzyme cascade; avoid"},
        {"drug": "N2O (nitrous oxide)", "risk": "MODERATE RISK",
         "reason": "Inactivates methionine synthase (MTR); no direct OAT interaction but general metabolic caution in PLP-dependent disease; short anaesthetic exposure probably acceptable with warning to anaesthesiologist"},
    ]

    differentials = [
        {"disease": "ALDH18A1 deficiency (P5CS deficiency)",
         "shared": "Ornithine/proline pathway; cataracts; AR inheritance",
         "distinguishing": "ALDH18A1: proline CRITICALLY LOW + ornithine LOW (synthesis entry blocked). OAT: ornithine VERY HIGH + proline LOW-NORMAL (catabolism blocked). Ornithine goes in OPPOSITE directions."},
        {"disease": "PYCR1/2 deficiency",
         "shared": "Proline pathway; AR; IDD; cutis laxa (PYCR1)",
         "distinguishing": "PYCR1: proline LOW + cutis laxa 90%; ornithine LOW-NORMAL; NO retinal atrophy. OAT: ornithine HIGH; NO cutis laxa; gyrate atrophy is OAT-specific."},
        {"disease": "OTC deficiency (Ornithine Transcarbamylase)",
         "shared": "Ornithine elevated; metabolic episodic decompensation",
         "distinguishing": "OTC: ammonia DRAMATICALLY HIGH + citrulline LOW; X-linked; urea cycle block. OAT: ammonia NORMAL; citrulline NORMAL; autosomal recessive; retinal atrophy present."},
        {"disease": "Choroideremia (CHM — CHM gene)",
         "shared": "Gyrate-like fundus appearance; nyctalopia; visual field loss",
         "distinguishing": "CHM: X-linked; ornithine COMPLETELY NORMAL; plasma amino acids normal; CHM (Rab escort protein-1) gene mutation. OAT: ornithine VERY HIGH on plasma amino acids."},
        {"disease": "Stargardt disease (ABCA4)",
         "shared": "Progressive retinal dystrophy; visual loss",
         "distinguishing": "Stargardt: macular pattern (not gyrate peripheral); ornithine NORMAL; ABCA4 mutation; flecked retina fundus; no epilepsy or myopathy"},
        {"disease": "AGAT deficiency (Arginine:glycine amidinotransferase)",
         "shared": "Ornithine pathway involvement; AR; mild IDD",
         "distinguishing": "AGAT: ornithine NORMAL; GAA (guanidinoacetate) CRITICALLY LOW + creatine LOW; NO retinal atrophy; creatine supplementation is curative. OAT: ornithine VERY HIGH; GAA normal."},
    ]

    variants = [
        {"variant": "p.Leu402Pro", "domain": "PLP-binding β-barrel", "freq_pct": 30, "severity": "Severe",
         "note": "Finnish founder variant; most common in Finnish patients; complete LOF; NOT B6-responsive"},
        {"variant": "p.Arg180Thr", "domain": "PLP Schiff-base adjacent", "freq_pct": 20, "severity": "Severe",
         "note": "PLP anchoring arginine replaced; complete LOF; NOT B6-responsive"},
        {"variant": "p.Glu318Lys", "domain": "Active site", "freq_pct": 15, "severity": "Moderate-Severe",
         "note": "Active site disruption; catalytic efficiency markedly reduced; partial activity"},
        {"variant": "p.Arg154Cys", "domain": "Tetramer interface", "freq_pct": 12, "severity": "Moderate",
         "note": "Homotetramer destabilisation; sometimes B6-responsive (partial PLP binding preserved)"},
        {"variant": "p.Gly237Val", "domain": "Ornithine-binding channel", "freq_pct": 10, "severity": "Moderate",
         "note": "Ornithine substrate channel disruption; reduced Km for ornithine; partial activity"},
        {"variant": "c.IVS6+1G>A", "domain": "Splice site (intron 6)", "freq_pct": 9, "severity": "Severe",
         "note": "Exon 6 skipping → frameshift → NMD → null allele; NOT B6-responsive"},
        {"variant": "p.Val332Met", "domain": "PLP-binding (partial)", "freq_pct": 8, "severity": "Mild-Moderate",
         "note": "B6-RESPONSIVE; partial PLP binding preserved; ornithine falls 50–70% on high-dose B6; slower retinal progression"},
        {"variant": "p.Thr181Ile", "domain": "Cofactor-binding minor", "freq_pct": 6, "severity": "Mild",
         "note": "Hypomorphic; B6-RESPONSIVE; Finnish compound heterozygous with p.Leu402Pro; attenuated phenotype; late onset"},
    ]

    n_severe  = round(_N * 0.60)
    n_b6resp  = round(_N * 0.30)

    patients = []
    for i in range(_N):
        pheno = (
            "Classic-Severe"  if i < n_severe else
            "B6-Responsive"   if i < n_severe + n_b6resp else
            "Mild-Attenuated"
        )
        if pheno == "Classic-Severe":
            orn = rng.uniform(600, 1500)
        elif pheno == "B6-Responsive":
            orn = rng.uniform(300, 700)
        else:
            orn = rng.uniform(200, 400)

        patients.append({
            "id": f"OAT-{i+1:03d}",
            "phenotype": pheno,
            "plasma_ornithine_umol_l": round(orn),
            "proline_umol_l": round(rng.uniform(30, 150)),
            "plp_nmol_l": round(rng.uniform(24, 58), 1),
            "age_diagnosis_years": round(rng.uniform(
                2, 15 if pheno == "Classic-Severe" else 5 if pheno == "B6-Responsive" else 10),
                1),
            "gyrate_atrophy": True,  # 100%
            "cataracts": rng.random() < (0.70 if pheno == "Classic-Severe" else 0.50 if pheno == "B6-Responsive" else 0.30),
            "myopia": rng.random() < (0.80 if pheno != "Mild-Attenuated" else 0.60),
            "epilepsy": rng.random() < (0.40 if pheno == "Classic-Severe" else 0.25 if pheno == "B6-Responsive" else 0.10),
            "dre": rng.random() < (0.22 if pheno == "Classic-Severe" else 0.10 if pheno == "B6-Responsive" else 0.05),
            "myopathy": rng.random() < (0.40 if pheno == "Classic-Severe" else 0.25 if pheno == "B6-Responsive" else 0.10),
            "b6_response": pheno == "B6-Responsive",
            "seizure_type": rng.choice(["Focal", "Absence", "GTCS", "Myoclonic", "None"])
                if rng.random() < 0.35 else "None",
        })

    return {
        "biomarkers": biomarkers,
        "clinical_features": clinical_features,
        "seizure_types": seizure_types,
        "treatments": treatments,
        "drug_risks": drug_risks,
        "differentials": differentials,
        "variants": variants,
        "patients": patients,
        "n": _N,
        "n_severe": n_severe,
        "n_b6resp": n_b6resp,
        "n_mild": _N - n_severe - n_b6resp,
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "gene": "OAT",
        "full_name": "Ornithine Aminotransferase",
        "disease_name": "Gyrate Atrophy of the Choroid and Retina",
        "omim_gene": "*258870",
        "omim_disease": "#258870",
        "chromosome": "10q26.13",
        "inheritance": "Autosomal Recessive (biallelic LOF); B6-responsive subset 30–40%",
        "protein": "439 aa; mitochondrial matrix homotetramer; PLP-dependent; ~46 kDa monomer",
        "enzyme_function": (
            "Catalyses: L-Ornithine + α-Ketoglutarate → Glutamate-γ-semialdehyde (P5C/GSA) + L-Glutamate. "
            "OAT is the mitochondrial enzyme linking ornithine catabolism to the proline/P5C pool. "
            "OAT LOF → ornithine accumulates (400–1,500 µmol/L) → retinotoxic → progressive gyrate atrophy."
        ),
        "pathway": (
            "Arginine → [Arginase] → Ornithine → [OAT, BLOCKED] → P5C/GSA → [PYCR1/PYCR2] → L-Proline. "
            "Parallel input: Glutamate → [ALDH18A1/P5CS] → P5C → [PYCR1/2] → L-Proline (unaffected in OAT LOF). "
            "Catabolism: L-Proline → [PRODH] → P5C → [ALDH4A1] → L-Glutamate."
        ),
        "key_terms": [
            {"term": "Gyrate atrophy", "definition": "Circular/scalloped chorioretinal atrophy patches starting peripherally; pathognomonic of OAT deficiency; ornithine is retinotoxic at >300 µmol/L"},
            {"term": "Ornithine (L-ornithine)", "definition": "Non-protein amino acid; intermediate in urea cycle (OTC uses ornithine + carbamoyl-phosphate → citrulline); accumulates in OAT deficiency"},
            {"term": "P5C (Δ1-pyrroline-5-carboxylate)", "definition": "Glutamate-γ-semialdehyde/P5C; central intermediate connecting ornithine, proline, and glutamate; made by OAT from ornithine and by ALDH18A1 from glutamate"},
            {"term": "PLP (pyridoxal-5'-phosphate)", "definition": "OAT cofactor; NORMAL in OAT deficiency (enzyme absent, not PLP); KEY NEGATIVE vs ALDH4A1 where P5C inactivates PLP via Schiff base"},
            {"term": "B6-responsive OAT", "definition": "Subset (~30–40%) with hypomorphic OAT alleles retaining partial PLP-binding capacity; high-dose B6 → PLP augmentation → ornithine falls 40–80%; MANDATORY trial at diagnosis"},
            {"term": "Arginase (ARG1)", "definition": "Urea cycle enzyme: arginine + H2O → ornithine + urea; dietary arginine restriction reduces ornithine synthesis by limiting arginase substrate"},
            {"term": "AGAT (GATM)", "definition": "Arginine:glycine amidinotransferase; uses ornithine + glycine → guanidinoacetate (GAA) + citrulline (first step creatine synthesis); elevated ornithine may partially impair AGAT"},
            {"term": "Nyctalopia", "definition": "Night blindness; earliest symptom of gyrate atrophy (rod-dominant ERG dysfunction); onset typically 5–15 years"},
            {"term": "ERG (electroretinogram)", "definition": "Mandatory monitoring tool in OAT; detects pre-symptomatic retinal dysfunction; scotopic (rod) amplitude loss precedes photopic (cone)"},
            {"term": "Finnish founder (p.Leu402Pro)", "definition": "Most common OAT variant in Finland (~1:50,000 prevalence in Finns); enriched in Finnish genetic isolate; complete LOF; not B6-responsive"},
            {"term": "Tubular aggregates", "definition": "Pathological muscle biopsy finding in OAT myopathy; abnormal SR membrane aggregates on electron microscopy; ornithine-related mitochondrial dysfunction"},
            {"term": "Aminoaciduria (ornithine)", "definition": "Ornithine overflow into urine; tubular reabsorption saturated; urine amino acid chromatography shows ornithine peak"},
            {"term": "SLC7A6/SLC7A7", "definition": "Cationic amino acid transporters; shared renal tubular transport for ornithine, lysine, arginine; lysine supplementation → competitive inhibition → increases urinary ornithine excretion (adjunct therapy)"},
        ],
        "pathway_summary": (
            "OAT bridges ornithine catabolism to the P5C/proline pool. "
            "When OAT is absent: ornithine cannot be funnelled to P5C; ornithine accumulates dramatically. "
            "High ornithine is directly retinotoxic (affects choroidal capillaries and RPE). "
            "ALDH18A1 still synthesises P5C from glutamate, so proline is only mildly reduced — "
            "unlike ALDH18A1 LOF where proline is CRITICALLY LOW. "
            "Arginine restriction (reducing ornithine input via arginase) + B6 trial (restoring "
            "residual OAT in responsive subset) are the therapeutic pillars."
        ),
        "key_pathway_comparisons": [
            {
                "pair": "OAT vs ALDH18A1",
                "description": "Both involve ornithine/P5C. ALDH18A1 LOF: ornithine LOW + proline CRITICALLY LOW (synthesis entry blocked upstream). OAT LOF: ornithine VERY HIGH + proline LOW-NORMAL (catabolism entry blocked; ALDH18A1 intact). Ornithine direction is OPPOSITE.",
            },
            {
                "pair": "OAT vs PRODH/ALDH4A1",
                "description": "PRODH/ALDH4A1 are proline CATABOLISM enzymes (proline → P5C → glutamate). OAT is an ornithine → P5C enzyme (separate route). PRODH LOF: proline HIGH. ALDH4A1 LOF: P5C HIGH + PLP inactivated. OAT LOF: ornithine HIGH; proline/P5C from ornithine route absent.",
            },
            {
                "pair": "OAT vs OTC deficiency",
                "description": "Both: ornithine elevated. OTC: ammonia DRAMATICALLY HIGH + citrulline LOW (urea cycle block) — life-threatening hyperammonemia. OAT: ammonia NORMAL; citrulline NORMAL — no urea cycle block; retinal not urea-cycle disease. Ammonia level is the decisive lab.",
            },
        ],
        "registered": "2026-08-24",
        "cohort_n": _N,
        "seed": _SEED,
    }
