#!/usr/bin/env python3
"""DBT (Dihydrolipoamide Branched-Chain Transacylase / BCKDH-E2 / MSUD Type 2) Epilepsy Dashboard.

DBT encodes the E2 subunit (dihydrolipoamide branched-chain transacylase) of the
branched-chain alpha-ketoacid dehydrogenase complex (BCKDH), the inner-core scaffold
that carries the lipoamide swinging arms and catalyses the transacylation step:

  BCKDH complex architecture:
    E1 heterotetrameric decarboxylase: BCKDHA (E1α, TPP-binding, catalytic)
                                     + BCKDHB (E1β, structural, stabilises E1α)
    E2 cubic core: DBT (E2, dihydrolipoamide branched-chain transacylase) — THIS GENE
    E3 homodimer:  DLD (E3, dihydrolipoamide dehydrogenase — SHARED with PDH, αKGDH, GCS)
    Kinase/phosphatase:  BCKDK (inactivates E1α) / PPM2C (reactivates E1α)

  BCKDH catalytic cycle (Step 2 of BCAA catabolism — IRREVERSIBLE):
    Step 2a (E1, TPP-dependent): branched-chain α-ketoacid + TPP-E1 →
             hydroxyacyl-TPP intermediate → CO₂ released → acyl-TPP formed
    Step 2b (E1→E2 transfer, DBT catalyses): acyl group transferred from
             hydroxyacyl-TPP on E1α to the lipoamide arm on E2 (DBT, Lys82/Lys99)
             → acyl-dihydrolipoamide-E2 intermediate
    Step 2c (E2 transacylase): acyl group transferred from E2-lipoamide to CoA →
             branched-chain acyl-CoA product (isovaleryl-CoA / 2-methylbutyryl-CoA / isobutyryl-CoA)
             + dihydrolipoamide-E2 (reduced, needs re-oxidation by E3/DLD)
    Step 2d (E3/DLD): dihydrolipoamide-E2 re-oxidised → lipoamide-E2 (ready for next cycle)
             using FAD/NAD+ (DLD is shared E3 — same as PDH, αKGDH, GCS)

  DBT (E2) structural role:
    DBT forms the INNER CUBIC CORE of the BCKDH megacomplex (~5 MDa);
    DBT monomers self-assemble into a 24-mer cubic core (same architecture as PDH E2/DLAT);
    E1 heterotetramers (BCKDHA α₂ + BCKDHB β₂) dock to the surface of the DBT core;
    DLD homodimers associate with DBT core via a specialised E3-binding domain (E3BD);
    DBT LOF → E2 core cannot form or is functionally compromised → E1 cannot dock →
    DLD cannot anchor → entire BCKDH complex disassembles → transacylation step blocked →
    BCAA accumulate → MSUD Type 2 (DBT deficiency).

  DBT role in BCAA catabolism — THREE substrates, identical block as BCKDHA/BCKDHB:
    Leucine → KIC (alpha-ketoisocaproate) → [E1 decarboxylates] → isovaleryl-TPP →
              [DBT/E2 transacylates] → isovaleryl-CoA BLOCKED in DBT deficiency
    Isoleucine → KMV (alpha-keto-beta-methylvalerate) → [E1] → 2-methylbutyryl-TPP →
                [DBT/E2] → 2-methylbutyryl-CoA BLOCKED
    Valine → KIV (alpha-ketoisovalerate) → [E1] → isobutyryl-TPP →
             [DBT/E2] → isobutyryl-CoA BLOCKED

  DBT LOF — same clinical outcome as BCKDHA and BCKDHB:
    E2 transacylation step cannot proceed → branched-chain acyl-CoAs not formed →
    upstream α-ketoacids (KIC, KMV, KIV) accumulate → BCAA accumulate (Leu>Ile,Val) →
    alloisoleucine appears (PATHOGNOMONIC) → leucine neurotoxicity → MSUD phenotype.

  DBT AND LIPOAMIDE (not TPP):
    DBT E2 contains the lipoamide cofactor (lipoyl domains at Lys82 and Lys99);
    Lipoamide is the "swinging arm" — covalently attached lipoic acid carries the acyl group
    between E1 (decarboxylation active site) and E2 (transacylase active site) and E3 (oxidation);
    Thiamine (vitamin B1) → TPP binds to E1α (BCKDHA), NOT to DBT E2;
    DBT E2 is NOT directly rescued by thiamine — however, improving E1α stability via TPP
    saturation may improve E1-E2 docking in some partially-misfolded DBT variants;
    expected thiamine response in DBT deficiency: ~5-10% (higher than BCKDHB ~5% because
    some DBT variants impair E1-E2 interface and may benefit indirectly from better E1α folding);
    ALL MSUD patients receive mandatory thiamine trial ≥3 months (Level A) regardless of subtype.

AUTOSOMAL RECESSIVE:
  DBT gene: 1p21.2; Autosomal Recessive (AR)
  ~421 aa precursor (~400 aa mature after mitochondrial targeting sequence cleavage)
  MSUD prevalence: ~1:185,000 globally (combined all MSUD types);
  DBT-type MSUD: less common than BCKDHA/B; exact frequency unclear (~15-20% of all MSUD cases)
  OMIM gene: *248610; disease: #248600 (MAPLE SYRUP URINE DISEASE, TYPE II)
  Note: MSUD types 1A (BCKDHA), 1B (BCKDHB), 2 (DBT) are biochemically IDENTICAL at presentation;
  MSUD type 3 (DLD deficiency) has additional PDH/αKGDH/GCS blocks (unique multi-system pattern).

KEY VARIANTS (clinically established):
  p.Gly245Ser: Japanese founder variant; most common DBT variant in East Asian patients;
               glycine-to-serine in the E2 transacylase domain; intermediate-to-classic phenotype;
               some thiamine responsiveness reported in a subset of homozygous carriers
  p.Arg268Cys: European; cysteine substitution at E1-E2 docking interface; severe classic MSUD
  p.Ile369Thr: European; transacylase catalytic domain; intermediate phenotype; some E2 residual
  p.Tyr391Asn: lipoyl domain region; disrupts lipoyl arm swing; severe classic/neonatal
  c.1228-2A>C: splice acceptor site; exon-skipping/null allele; classic neonatal MSUD
  del.exon5-6: deletion; null allele; classic neonatal; elevated leucine from day 3

CLINICAL PRESENTATIONS (identical biochemistry to BCKDHA / BCKDHB):
  1. Classic MSUD (65%): neonatal onset day 3-7; maple syrup odour; progressive encephalopathy;
     hypotonia → hypertonia; burst suppression on neonatal EEG; cerebral oedema; lethal untreated
  2. Intermediate MSUD (15%): onset 5 months-7 years; chronic protein intolerance;
     progressive neurodevelopmental delay; acute decompensation with metabolic stress
  3. Intermittent MSUD (12%): episodic crisis only during metabolic stress (fever/fasting/surgery);
     normal between episodes; developmental outcomes better if leucine controlled
  4. Thiamine-Responsive MSUD (8%): significant response to high-dose B1 (10-200 mg/kg/day);
     thought to improve E1-E2 docking stability indirectly via TPP-saturated E1α;
     slightly higher rate than BCKDHB (~5%) due to interface-affecting DBT variants

PRIMARY ACCUMULATING TOXIC SPECIES (identical to BCKDHA and BCKDHB):
  1. Leucine: most neurotoxic; LAT1 competitive inhibition; mTOR hyperactivation; myelinopathy
  2. Alpha-ketoisocaproate (KIC): myelinopathic; direct mitochondrial toxin
  3. Alloisoleucine: pathognomonic for any MSUD — formed by KMV stereoisomerisation;
     >5 µmol/L plasma = DIAGNOSTIC; ABSENT in all non-MSUD conditions
  4. Alpha-ketoisovalerate (KIV) and alpha-keto-β-methylvalerate (KMV)
  5. Branched-chain hydroxy acids (DNPH-positive)

TREATMENT (identical to BCKDHA/BCKDHB):
  BCAA-restricted diet + BCAA-free synthetic formula (Level A, lifelong)
  Thiamine B1 high-dose trial ≥3 months ALL patients (Level A; ~5-10% response in DBT deficiency)
  Acute crisis: IV glucose GIR 8-12 + insulin 0.05-0.1 U/kg/h + BCAA-free AA infusion (Level A)
  HD/CRRT if leucine still rising >1500 µmol/L (Level A)
  Carnitine supplementation (Level B — secondary depletion)
  Liver transplant (Level B — corrects ~80% BCKDH activity; consider before age 10)
  LEV (levetiracetam) first-line AED (Level B)
  VPA: ABSOLUTE CONTRAINDICATION (triple mechanism: carnitine depletion + BCKDH inhibition +
       mitochondrial hepatotoxicity) — NEVER use in MSUD of any type (BCKDHA, BCKDHB, DBT, DLD)
"""

import random

random.seed(47)


# ─────────────────────────────────────────────────────────────────────────────
#  PATIENT COHORT — 40 synthetic DBT-deficient patients
# ─────────────────────────────────────────────────────────────────────────────

# Phenotype weights: Classic 65%, Intermediate 15%, Intermittent 12%, Thiamine-Responsive 8%
PHENOTYPES = (
    ["Classic MSUD (neonatal)"] * 26 +
    ["Intermediate MSUD"] * 6 +
    ["Intermittent MSUD"] * 5 +
    ["Thiamine-Responsive MSUD"] * 3
)

SEXES = ["M", "F"]
VARIANTS = [
    "p.Gly245Ser/p.Gly245Ser",
    "p.Gly245Ser/p.Ile369Thr",
    "p.Arg268Cys/p.Arg268Cys",
    "p.Arg268Cys/c.1228-2A>C",
    "p.Tyr391Asn/p.Tyr391Asn",
    "p.Ile369Thr/c.1228-2A>C",
    "p.Gly245Ser/c.1228-2A>C",
    "del.exon5-6/c.1228-2A>C",
    "p.Arg268Cys/del.exon5-6",
    "p.Ile369Thr/p.Ile369Thr",
]


def _make_patient(idx, phenotype):
    sex = random.choice(SEXES)
    classic = phenotype == "Classic MSUD (neonatal)"
    intermediate = phenotype == "Intermediate MSUD"
    intermittent = phenotype == "Intermittent MSUD"
    thiamine_r = phenotype == "Thiamine-Responsive MSUD"

    if classic:
        onset = random.randint(0, 1)
        leu = random.randint(1800, 4200)
        ile = random.randint(350, 900)
        val = random.randint(500, 1400)
        allo = random.randint(28, 95)
        maple = True
        oedema = random.random() < 0.72
        mri_dwi = random.random() < 0.82
        thiamine_resp = random.random() < 0.03
        diet_ok = random.random() < 0.55
        transplant = random.random() < 0.20
        vpa_avoided = random.random() < 0.98
    elif intermediate:
        onset = random.randint(4, 30)
        leu = random.randint(350, 950)
        ile = random.randint(120, 380)
        val = random.randint(180, 560)
        allo = random.randint(8, 35)
        maple = random.random() < 0.45
        oedema = random.random() < 0.18
        mri_dwi = random.random() < 0.35
        thiamine_resp = random.random() < 0.10
        diet_ok = random.random() < 0.72
        transplant = random.random() < 0.12
        vpa_avoided = random.random() < 0.96
    elif intermittent:
        onset = random.randint(8, 72)
        leu = random.randint(140, 400)
        ile = random.randint(60, 200)
        val = random.randint(80, 300)
        allo = random.randint(5, 18)
        maple = random.random() < 0.25
        oedema = random.random() < 0.08
        mri_dwi = random.random() < 0.18
        thiamine_resp = random.random() < 0.12
        diet_ok = random.random() < 0.85
        transplant = random.random() < 0.05
        vpa_avoided = random.random() < 0.94
    else:  # thiamine_r
        onset = random.randint(1, 12)
        leu = random.randint(280, 700)
        ile = random.randint(90, 280)
        val = random.randint(140, 420)
        allo = random.randint(6, 25)
        maple = random.random() < 0.55
        oedema = random.random() < 0.12
        mri_dwi = random.random() < 0.22
        thiamine_resp = True
        diet_ok = random.random() < 0.88
        transplant = random.random() < 0.05
        vpa_avoided = random.random() < 0.97

    carnitine_def = random.random() < 0.40
    variant = random.choice(VARIANTS)

    return {
        "id": f"DBT-{idx+1:03d}",
        "sex": sex,
        "phenotype": phenotype,
        "onset_age_months": onset,
        "plasma_leucine_umol": leu,
        "plasma_isoleucine_umol": ile,
        "plasma_valine_umol": val,
        "alloisoleucine_umol": allo,
        "maple_odour": maple,
        "cerebral_oedema": oedema,
        "mri_diffusion_restriction": mri_dwi,
        "thiamine_responsive": thiamine_resp,
        "diet_well_controlled": diet_ok,
        "liver_transplant": transplant,
        "carnitine_deficient": carnitine_def,
        "vpa_avoided": vpa_avoided,
        "variant": variant,
    }


_PATIENTS = [_make_patient(i, PHENOTYPES[i]) for i in range(40)]


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_overview():
    """Return gene metadata, KPIs, phenotype distribution, BCKDH complex components,
    BCAA catabolism pathway, and high-risk drugs for the 40-patient DBT cohort.

    Emphasises E2 transacylase catalytic role (lipoamide-arm swinging), the 24-mer
    cubic core scaffold, biochemical identity with BCKDHA/BCKDHB, and the rationale
    for slightly higher thiamine response (~5-10%) vs BCKDHB (~5%).
    """
    pts = _PATIENTS
    n = len(pts)
    maple_pct = round(sum(1 for p in pts if p["maple_odour"]) / n * 100, 1)
    oedema_pct = round(sum(1 for p in pts if p["cerebral_oedema"]) / n * 100, 1)
    mri_pct = round(sum(1 for p in pts if p["mri_diffusion_restriction"]) / n * 100, 1)
    thr_pct = round(sum(1 for p in pts if p["thiamine_responsive"]) / n * 100, 1)
    transplant_pct = round(sum(1 for p in pts if p["liver_transplant"]) / n * 100, 1)
    carnitine_pct = round(sum(1 for p in pts if p["carnitine_deficient"]) / n * 100, 1)
    vpa_ok_pct = round(sum(1 for p in pts if p["vpa_avoided"]) / n * 100, 1)
    male_pct = round(sum(1 for p in pts if p["sex"] == "M") / n * 100, 1)
    avg_leu = round(sum(p["plasma_leucine_umol"] for p in pts) / n, 0)
    avg_allo = round(sum(p["alloisoleucine_umol"] for p in pts) / n, 1)

    # DRE rate — metabolic epilepsies have high DRE when uncontrolled
    dre_pts = sum(1 for p in pts if p["plasma_leucine_umol"] > 600 and not p["diet_well_controlled"])
    dre_pct = round(dre_pts / n * 100, 1)

    return {
        "gene": "DBT",
        "protein": "Dihydrolipoamide Branched-Chain Transacylase (E2 subunit of BCKDH complex)",
        "aliases": "BCKADE2; BCATE2; BCKDE2; BCKAD-E2",
        "locus": "1p21.2",
        "aa_length": "421",
        "cofactor": "Lipoamide (covalently attached lipoic acid on Lys82 and Lys99 lipoyl domains — NOT TPP; TPP is on E1α/BCKDHA)",
        "mechanism": "DBT forms the inner 24-mer cubic core of the BCKDH complex; carries the lipoamide 'swinging arm' that shuttles acyl groups from E1α (decarboxylation) → E2 (transacylation to CoA); DBT LOF → E2 core collapses → E1 cannot dock → transacylation BLOCKED → branched-chain acyl-CoAs not formed → KIC/KMV/KIV + Leu/Ile/Val accumulate → MSUD Type 2",
        "omim_gene": "*248610",
        "omim_disease": "#248600 (MAPLE SYRUP URINE DISEASE, TYPE II / MSUD TYPE 2)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "pathognomonic_marker": "Alloisoleucine >5 µmol/L (plasma) — IDENTICAL to BCKDHA and BCKDHB deficiency; absent in all non-MSUD conditions",
        "key_distinguishing_feature": "E2 CATALYTIC transacylase role (lipoamide-arm); 24-mer cubic core scaffold; DBT LOF blocks E1-E2 transfer step; thiamine response ~5-10% (indirect via E1α stabilisation) — between BCKDHA ~10-20% and BCKDHB ~5%",
        "structural_brain_hallmark": "White matter diffusion restriction (DWI) 80%+ — MSUD myelinopathy pattern; cerebral oedema in acute classic crisis; basal ganglia in severe neonatal",
        "kpis": {
            "cohort_n": n,
            "avg_plasma_leucine_umol": int(avg_leu),
            "avg_alloisoleucine_umol": avg_allo,
            "maple_odour_pct": maple_pct,
            "cerebral_oedema_acute_pct": oedema_pct,
            "mri_diffusion_restriction_pct": mri_pct,
            "thiamine_responsive_pct": thr_pct,
            "liver_transplant_pct": transplant_pct,
            "carnitine_deficient_pct": carnitine_pct,
            "dre_pct": dre_pct,
            "vpa_avoided_pct": vpa_ok_pct,
            "male_pct": male_pct,
        },
        "phenotype_distribution": [
            {"label": "Classic MSUD (neonatal)", "pct": 65, "color": "#b71c1c"},
            {"label": "Intermediate MSUD", "pct": 15, "color": "#e65100"},
            {"label": "Intermittent MSUD", "pct": 12, "color": "#f9a825"},
            {"label": "Thiamine-Responsive MSUD", "pct": 8, "color": "#0277bd"},
        ],
        "bckdh_complex_components": [
            {
                "component": "E1α heterotetrameric decarboxylase (BCKDHA — TPP-binding, catalytic)",
                "function": "Contains the TPP active site; oxidatively decarboxylates branched-chain α-ketoacids (KIC, KMV, KIV) → branched-chain acyl-TPP intermediates; transfers acyl group to DBT/E2 lipoamide arm; NORMAL in isolated DBT deficiency"
            },
            {
                "component": "E1β structural subunit (BCKDHB — no TPP site, stabilises E1α)",
                "function": "Structural partner of E1α in the α₂β₂ heterotetramer; required for E1α stability and assembly; NORMAL in isolated DBT deficiency"
            },
            {
                "component": "E2 core — DBT (THIS GENE — dihydrolipoamide transacylase, 24-mer cubic core)",
                "function": "BLOCKED IN DBT DEFICIENCY: forms the inner cubic core scaffold (~24-mer); contains lipoyl domains (Lys82, Lys99) whose lipoamide 'swinging arms' accept acyl group from E1α and transfer to CoA (transacylation); also provides docking sites for E1 heterotetramers and E3/DLD homodimers"
            },
            {
                "component": "E3 DLD homodimer (dihydrolipoamide dehydrogenase — SHARED)",
                "function": "FAD-dependent re-oxidation of reduced dihydrolipoamide-E2 → lipoamide-E2 (regenerates lipoamide for next cycle); DLD is SHARED with PDH, αKGDH, and GCS; NORMAL in isolated DBT deficiency (DLD free enzyme NORMAL — key negative)"
            },
            {
                "component": "BCKDK / PPM2C (kinase / phosphatase regulatory pair)",
                "function": "BCKDK phosphorylates E1α-Ser293 → inactivates BCKDH (limits BCAA catabolism); PPM2C dephosphorylates → reactivates; regulatory in response to nutritional state; NORMAL in DBT deficiency"
            },
        ],
        "bcaa_catabolism_pathway": {
            "step_1_reversible": "BCAA + α-ketoglutarate → branched-chain α-ketoacid + glutamate [branched-chain aminotransferase, BCAT2 in mitochondria]; REVERSIBLE — BCAAs can be regenerated from ketoacids",
            "step_2_irreversible_blocked": "Branched-chain α-ketoacid → branched-chain acyl-CoA + CO₂ [BCKDH complex: E1(BCKDHA/B) decarboxylates → E2(DBT) transacylates to CoA → E3(DLD) re-oxidises lipoamide]; BLOCKED IN DBT DEFICIENCY (E2 transacylation step missing)",
            "downstream_leu": "KIC (α-ketoisocaproate) → [BLOCKED at DBT/E2 step] → isovaleryl-CoA (cannot be formed) → isovaleryl-CoA → 3-methylcrotonyl-CoA → downstream Leu catabolism (ALL DOWNSTREAM BLOCKED)",
            "downstream_val": "KIV (α-ketoisovalerate) → [BLOCKED at DBT/E2 step] → isobutyryl-CoA (cannot be formed) → methylmalonate/succinate pathway (ALL DOWNSTREAM BLOCKED)",
            "downstream_ile": "KMV (α-keto-β-methylvalerate) → [BLOCKED at DBT/E2 step] → 2-methylbutyryl-CoA (cannot be formed) → 2-methylbutyrylglycine pathway (ALL DOWNSTREAM BLOCKED)",
            "alloisoleucine_formation": "KMV (isoleucine-derived ketoacid) accumulates → spontaneous transamination to D-alloisoleucine configuration → alloisoleucine appears in plasma >5 µmol/L → PATHOGNOMONIC FOR MSUD (BCKDHA/BCKDHB/DBT deficiency)"
        },
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
             "mechanism": "Triple CI in MSUD (any type including DBT): (1) carnitine depletion → secondary carnitine-deficient crisis; (2) direct BCKDH complex inhibition → worsens BCAA accumulation; (3) mitochondrial hepatotoxicity (POLG1-equivalent risk). NEVER use in DBT deficiency."},
            {"drug": "Fasting / prolonged NPO", "risk": "EXTREME HAZARD",
             "mechanism": "Muscle protein catabolism → BCAA surge → leucine crisis within 2-6 hours. IV glucose + BCAA-free AAs mandatory during any fasting. Pre-plan sick-day protocol."},
            {"drug": "Unrestricted natural protein", "risk": "EXTREME HAZARD",
             "mechanism": "All natural proteins contain leucine, isoleucine, valine; any unrestricted high-protein meal → rapid BCAA surge → encephalopathy. BCAA-free synthetic formula is the metabolic cornerstone."},
            {"drug": "Surgery / catabolic stress without metabolic protocol", "risk": "HAZARD",
             "mechanism": "Surgical stress + NPO + catabolism → BCAA surge → leucine crisis. All surgery requires perioperative IV glucose + BCAA monitoring + metabolic team involvement."},
        ],
    }


def get_breakdown():
    """Return biomarkers, key variants, patient cohort (first 10 of 40), seizure types,
    metabolic triggers, and treatments for DBT deficiency.

    Biomarkers emphasise the shared MSUD pattern (Leu/Ile/Val, alloisoleucine, KIC/KMV/KIV,
    DNPH positive) plus the KEY NEGATIVES that distinguish isolated DBT deficiency from
    DLD deficiency (PDH NORMAL, αKGDH NORMAL, DLD free enzyme NORMAL, GCS NORMAL).
    """
    biomarkers = [
        {"name": "Plasma Leucine (µmol/L)", "normal": "<200", "dbt_range": ">2000 µmol/L (classic acute); 400–1200 µmol/L (intermediate); 200–600 µmol/L (intermittent/thiamine-responsive)",
         "significance": "Most neurotoxic BCAA; LAT1 competitive inhibition; mTOR hyperactivation; myelinopathy. Crisis threshold: >400 (encephalopathy); >1000 (cerebral oedema); >1500 → HD/CRRT"},
        {"name": "Alloisoleucine (µmol/L)", "normal": "ABSENT", "dbt_range": ">5 µmol/L (any MSUD subtype) — PATHOGNOMONIC",
         "significance": "D-alloisoleucine: formed by KMV stereoisomerisation; present ONLY in MSUD (BCKDHA/BCKDHB/DBT); confirms MSUD; gene panel required to identify DBT vs BCKDHA/B"},
        {"name": "Plasma Isoleucine (µmol/L)", "normal": "40–100", "dbt_range": "300–900 µmol/L (classic); 150–500 µmol/L (intermediate)",
         "significance": "Accumulates with leucine; generates alloisoleucine (pathognomonic via KMV-stereoisomerisation)"},
        {"name": "Plasma Valine (µmol/L)", "normal": "150–350", "dbt_range": "500–1800 µmol/L (classic); 200–800 µmol/L (intermediate)",
         "significance": "Third BCAA accumulating; neurotoxic (less so than leucine); elevated proportionally"},
        {"name": "Urine Branched-Chain Ketoacids (KIC/KMV/KIV, µmol/mmolCr)", "normal": "<10", "dbt_range": "KIC 800–4000; KIV, KMV elevated proportionally",
         "significance": "Direct upstream substrate accumulation from E2/DBT block; confirms BCKDH impairment"},
        {"name": "DNPH test (2,4-dinitrophenylhydrazine)", "normal": "NEGATIVE", "dbt_range": "POSITIVE — yellow precipitate (branched-chain α-ketoacids react with DNPH)",
         "significance": "Bedside screening; non-specific for ketoacids; positive in MSUD, IVA, MMA, PA — confirms α-ketoaciduria; follow with plasma amino acids + urine organic acids"},
        {"name": "Branched-Chain Acylcarnitines (C5/C4 in MS/MS NBS)", "normal": "<0.5 µmol/L", "dbt_range": "C5-carnitine (isovalerylcarnitine) mildly elevated; C4 (isobutyrylcarnitine) elevated",
         "significance": "Newborn screening marker; less specific for MSUD vs IVA (IVA has higher C5); confirms BCAA pathway impairment; combined with Leu/Ile/Val elevation on NBS amino acids"},
        {"name": "BCKDH complex activity (fibroblasts/lymphocytes)", "normal": ">20 nmol/min/mg protein", "dbt_range": "<5% of normal (classic); 5–25% (intermediate)",
         "significance": "Definitive enzymatic diagnosis; DBT deficiency measured as reduced E2 transacylase or total complex activity; specific DBT E2 sub-assay possible but gene panel preferred"},
        {"name": "PDH activity", "normal": "Normal range", "dbt_range": "NORMAL — isolated DBT deficiency does NOT affect PDH (different E2 subunit: DLAT not DBT)",
         "significance": "KEY NEGATIVE: if PDH also reduced → suggests DLD deficiency (E3/DLD is shared) or DLAT deficiency; NORMAL PDH confirms isolated BCKDH block in DBT deficiency"},
        {"name": "αKGDH complex activity", "normal": "Normal range", "dbt_range": "NORMAL — isolated DBT deficiency does NOT affect αKGDH (different E2 subunit: DLST not DBT)",
         "significance": "KEY NEGATIVE: if αKGDH also reduced → DLD deficiency likely; NORMAL αKGDH in DBT deficiency"},
        {"name": "DLD free enzyme activity", "normal": ">10 nmol/min/mg", "dbt_range": "NORMAL — DLD (E3) protein is intact in isolated DBT deficiency",
         "significance": "KEY NEGATIVE — distinguishes DBT deficiency from DLD/E3 deficiency, where all four complexes are reduced. NORMAL DLD in DBT deficiency."},
        {"name": "Lactate / Pyruvate (mmol/L)", "normal": "L<2.2; P<0.1; L:P<20", "dbt_range": "NORMAL or mildly elevated; L:P ratio NORMAL (10–20) — PDH intact",
         "significance": "KEY NEGATIVE: normal L:P ratio distinguishes DBT from PDH deficiency (LP ratio normal 10-20 in both) and from Complex I (LP ratio >25); but L may rise secondary to metabolic stress"},
    ]

    key_variants = [
        {"variant": "p.Gly245Ser", "effect": "Glycine-to-serine in E2 transacylase catalytic domain; glycine at 245 is conserved in the transacylase fold; Ser substitution mildly disrupts folding; most common DBT variant in East Asian (Japanese) patients", "phenotype": "Intermediate/Classic MSUD — Japanese founder; some thiamine responsiveness in homozygotes via indirect E1α stabilisation"},
        {"variant": "p.Arg268Cys", "effect": "Arginine-to-cysteine at E1-E2 docking interface; disrupts E1-heterotetmer-to-E2-core interaction; thiol group of Cys may form incorrect disulphide; E2 core structurally compromised", "phenotype": "Severe Classic MSUD (European)"},
        {"variant": "p.Ile369Thr", "effect": "Isoleucine-to-threonine in transacylase catalytic domain; hydrophobic→hydrophilic substitution in core of domain; partial residual E2 activity in some assays", "phenotype": "Intermediate MSUD (European); some diet-manageable patients"},
        {"variant": "p.Tyr391Asn", "effect": "Tyrosine-to-asparagine near lipoyl domain tethering region; disrupts the lipoamide swinging arm geometry; lipoyl-Lys99 region destabilised; reduced acyl-transfer efficiency", "phenotype": "Severe Classic MSUD; early neonatal encephalopathy"},
        {"variant": "c.1228-2A>C", "effect": "Splice acceptor site mutation; exon skipping → in-frame deletion or null allele; complete loss of E2 transacylase domain; no functional DBT produced", "phenotype": "Classic Neonatal MSUD; severe; leucine >3000 µmol/L on day 3–5"},
        {"variant": "del.exon5-6", "effect": "Genomic deletion encompassing exons 5 and 6; removes N-terminal lipoyl domains (Lys82, Lys99); null for E2 transacylase; BCKDH complex cannot form inner core", "phenotype": "Classic Neonatal MSUD; non-responsive to thiamine"},
    ]

    # Patient sample (first 10 of 40)
    patient_sample = _PATIENTS[:10]

    seizure_types = [
        {"type": "Tonic seizures (metabolic encephalopathy)", "pct": 68},
        {"type": "Myoclonic seizures", "pct": 58},
        {"type": "Infantile spasms / West syndrome", "pct": 45},
        {"type": "Focal seizures", "pct": 38},
        {"type": "Absence-like spells", "pct": 30},
        {"type": "Status epilepticus (acute crisis)", "pct": 48},
        {"type": "Burst suppression pattern (neonatal EEG)", "pct": 62},
    ]

    trigger_types = [
        {"trigger": "Febrile illness / infection", "pct": 92},
        {"trigger": "Fasting / prolonged NPO", "pct": 88},
        {"trigger": "Unrestricted dietary protein", "pct": 85},
        {"trigger": "Surgical stress / catabolism", "pct": 70},
        {"trigger": "Exercise-induced catabolism", "pct": 55},
        {"trigger": "Psychological stress", "pct": 40},
        {"trigger": "Growth spurts / puberty", "pct": 35},
    ]

    treatments = [
        {"drug": "BCAA-restricted diet + BCAA-free synthetic formula (Level A — lifelong)", "level": "A", "response_pct": 95,
         "color": "#004d40",
         "note": "Metabolic cornerstone: provide ALL amino acids EXCEPT leucine, isoleucine, valine via synthetic formula; natural protein only at minimum to prevent catabolism; target plasma Leu <200 µmol/L maintenance; identical management to BCKDHA and BCKDHB; lifelong, non-negotiable"},
        {"drug": "Thiamine B1 high-dose trial (Level A — ALL patients)", "level": "A", "response_pct": 8,
         "color": "#0277bd",
         "note": "10–20 mg/kg/day (up to 100–200 mg/day); ALL DBT/MSUD patients receive mandatory ≥3-month trial regardless of genotype. IMPORTANT: DBT/E2 does NOT contain the TPP active site (that is on E1α/BCKDHA); thiamine works INDIRECTLY — TPP saturation of E1α may improve E1-E2 docking interface in some partially-misfolded DBT variants; response rate ~5-10% in DBT deficiency (slightly higher than BCKDHB ~5% because some DBT variants impair E1-E2 interface and benefit from better E1α folding; lower than BCKDHA ~10-20% because the direct TPP-binding rescue mechanism does not apply to DBT). Non-responders do not worsen. Lifelong in responders."},
        {"drug": "IV glucose + insulin (Level A — acute crisis)", "level": "A", "response_pct": 90,
         "color": "#0277bd",
         "note": "GIR 8–12 mg/kg/min IV glucose + insulin 0.05–0.1 U/kg/h; suppresses muscle BCAA catabolism; BCAA-free amino acid infusion in parallel; HD/CRRT if leucine still rising >1500 µmol/L despite above; identical protocol to BCKDHA and BCKDHB acute crisis"},
        {"drug": "Carnitine supplementation (Level B)", "level": "B", "response_pct": 60,
         "color": "#4a148c",
         "note": "Secondary carnitine depletion from branched-chain acylcarnitine esterification; monitor free/total carnitine ratio; supplement if free carnitine <25 µmol/L; 50–100 mg/kg/day; also supports renal carnitine excretion monitoring"},
        {"drug": "Liver transplantation (Level B)", "level": "B", "response_pct": 80,
         "color": "#4a148c",
         "note": "Liver provides ~80% of whole-body BCKDH complex (including DBT/E2) activity; transplant corrects metabolic defect; allows liberalised diet (some restriction still needed); best outcomes before 10 years of age; carnitine monitoring post-transplant; identical benefit to BCKDHA/BCKDHB transplant"},
        {"drug": "LEV — Levetiracetam (Level B)", "level": "B", "response_pct": 60,
         "color": "#1b5e20",
         "note": "First-line AED for seizure control in MSUD (DBT deficiency); no BCAA pathway interaction; no hepatotoxicity; safe metabolic profile; seizure control secondary to metabolic normalisation; VPA ABSOLUTELY CONTRAINDICATED"},
    ]

    high_risk_drugs = [
        {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
         "mechanism": "Triple CI in DBT/MSUD: (1) secondary carnitine depletion → carnitine-depleted crisis; (2) direct BCKDH complex inhibition → acute BCAA rise; (3) mitochondrial hepatotoxicity (POLG1-equivalent). NEVER use in DBT deficiency."},
        {"drug": "Fasting / NPO without IV cover", "risk": "EXTREME HAZARD",
         "mechanism": "Muscle protein catabolism → BCAA surge → leucine encephalopathy within hours. IV glucose mandatory for any fasting period."},
        {"drug": "Unrestricted dietary protein", "risk": "EXTREME HAZARD",
         "mechanism": "All natural proteins contain BCAA; any unrestricted meal → leucine crisis. BCAA-free formula is non-negotiable."},
        {"drug": "Surgery without metabolic protocol", "risk": "HAZARD",
         "mechanism": "Catabolic stress + NPO → BCAA crisis. Perioperative IV glucose + BCAA monitoring + metabolic team required."},
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
    """Return gene card, key concepts, diagnostic thresholds, and differential diagnosis.

    Emphasises E2 catalytic/scaffold role, lipoamide arm mechanism, 24-mer cubic core,
    thiamine response rationale (~5-10% in DBT, intermediate between BCKDHA and BCKDHB),
    and the biochemical indistinguishability of MSUD Types 1A/1B/2 requiring gene panel.
    """
    return {
        "gene_card": {
            "Gene": "DBT (Dihydrolipoamide Branched-Chain Transacylase)",
            "Also known as": "BCKADE2; BCATE2; BCKDE2; BCKAD-E2 (branched-chain α-ketoacid dehydrogenase complex E2 subunit)",
            "Subunit": "E2 CATALYTIC transacylase subunit + inner cubic core scaffold of BCKDH complex (24-mer); distinct from E1α (BCKDHA, TPP-binding) and E1β (BCKDHB, structural E1α partner); lipoamide arm on Lys82/Lys99",
            "Chromosome": "1p21.2",
            "Protein length": "~421 aa precursor (~400 aa mature after MTS cleavage)",
            "Cofactor": "Lipoamide (covalently attached lipoic acid at Lys82 and Lys99 lipoyl domains) — NOT TPP; TPP binds E1α/BCKDHA; thiamine trial mandatory (Level A) but response ~5-10% in DBT (indirect, via E1α stabilisation)",
            "Function": "E2 inner cubic core scaffold; lipoamide swinging arms accept acyl group from E1α (after decarboxylation) and transfer to CoA (transacylation); also provides docking platform for E1 heterotetramers and E3/DLD homodimers; DBT LOF → cubic core absent → E1 cannot dock → transacylation blocked → BCAA accumulate",
            "Complex partners": "BCKDHA (E1α, catalytic/TPP) + BCKDHB (E1β, structural/E1α-stabiliser) + DBT (E2, this gene) + DLD (E3, FAD homodimer, SHARED with PDH/αKGDH/GCS)",
            "Inheritance": "Autosomal Recessive (AR) — biallelic LOF",
            "OMIM gene": "*248610",
            "OMIM disease": "#248600 (MAPLE SYRUP URINE DISEASE, TYPE II / DBT-MSUD)",
            "Prevalence": "~1:185,000 globally (combined MSUD); DBT-type ~15-20% of all MSUD; more common in East Asian populations (p.Gly245Ser founder)",
            "Biochemical block": "BCKDH complex Step 2b/2c (E2 transacylation) BLOCKED — acyl-TPP intermediate from E1α cannot be transferred to CoA; KIC/KMV/KIV + Leu/Ile/Val accumulate",
            "Pathognomonic marker": "Alloisoleucine >5 µmol/L (plasma) — IDENTICAL to BCKDHA and BCKDHB; present ONLY in MSUD",
            "Primary toxic species": "Leucine (most neurotoxic); KIC (alpha-ketoisocaproate, myelinopathic)",
            "First-line treatment": "BCAA-restricted diet + BCAA-free synthetic formula (Level A) + thiamine trial (Level A; ~5-10% response in DBT deficiency)",
            "ABSOLUTE CI": "Valproate (VPA) — triple contraindication (carnitine depletion + BCKDH inhibition + hepatotoxicity)",
        },
        "key_concepts": [
            {"term": "DBT E2 catalytic role — transacylation and lipoamide swinging arm",
             "definition": "DBT (E2) is the catalytic transacylase of the BCKDH complex AND the structural inner core scaffold. The catalytic mechanism: (1) E1α (BCKDHA) decarboxylates the branched-chain α-ketoacid, generating an acyl-hydroxyethyl-TPP intermediate; (2) this acyl group is transferred to the lipoamide arm of DBT/E2 (covalently attached lipoic acid at Lys82 or Lys99 — the 'swinging arm') → acyl-dihydrolipoamide-E2; (3) the acyl group is then transferred from E2-lipoamide to CoA (transacylation), generating the branched-chain acyl-CoA product; (4) E3/DLD re-oxidises the reduced dihydrolipoamide-E2 → lipoamide-E2 for the next cycle. DBT LOF blocks step (2)/(3) — acyl-TPP cannot be transferred → upstream substrates (KIC, KMV, KIV) and BCAA accumulate."},
            {"term": "Why DBT thiamine response (~5-10%) is between BCKDHA (~10-20%) and BCKDHB (~5%)",
             "definition": "Thiamine (B1) → TPP (thiamine pyrophosphate), which binds directly to E1α (BCKDHA) active site. In BCKDHA deficiency: direct TPP saturation can stabilise a partially-misfolded E1α active site → direct enzyme rescue (~10-20% respond). In BCKDHB deficiency: E1β has no TPP site; even saturated E1α cannot fully compensate for absent E1β → only ~5% respond. In DBT deficiency: TPP binds E1α and may improve E1α stability and E1α-E1β assembly → the better-folded E1 heterotetramer may dock more effectively to a partially-functional DBT/E2 core in variants that mildly impair E1-E2 interaction (e.g. p.Gly245Ser, p.Ile369Thr) rather than null alleles → ~5-10% response rate. This is an indirect rescue (via improved E1α → better E1-E2 docking) not a direct rescue (no TPP on E2). Null DBT alleles are non-responsive."},
            {"term": "24-mer cubic core architecture — why DBT LOF is catastrophic for the whole complex",
             "definition": "The BCKDH E2 core (DBT) self-assembles into a 24-subunit cubic scaffold (~5 MDa megacomplex) analogous to pyruvate dehydrogenase E2 (DLAT) and α-ketoglutarate dehydrogenase E2 (DLST). This cubic core: (1) provides the structural platform for E1 heterotetramer docking (via E1-binding domain on E2); (2) provides the anchoring site for E3/DLD homodimers (via E3-binding domain/E3BD on E2); (3) carries the lipoamide swinging arms that physically shuttle acyl groups between E1, E2 (transacylase site), and E3. If DBT is absent or severely misfolded: no cubic core forms → E1 heterotetramers cannot dock → DLD/E3 cannot anchor → the entire BCKDH megacomplex disassembles → step 2 BCAA catabolism BLOCKED (identical clinical outcome to BCKDHA or BCKDHB LOF)."},
            {"term": "Alloisoleucine — pathognomonic MSUD marker (identical in DBT deficiency)",
             "definition": "Alloisoleucine (D-allo-isoleucine) is formed when KMV (alpha-keto-beta-methylvalerate, the isoleucine-derived α-ketoacid) accumulates in MSUD: spontaneous transamination back to the D-stereoisomer configuration of isoleucine generates alloisoleucine. Alloisoleucine is absent from all normal plasma and appears ONLY in MSUD (any type: BCKDHA/BCKDHB/DBT deficiency). A single plasma amino acids test showing alloisoleucine >5 µmol/L confirms MSUD and is pathognomonic. Alloisoleucine levels are IDENTICAL across all MSUD subtypes — gene panel (BCKDHA + BCKDHB + DBT ± DLD) is required for genotype-specific diagnosis."},
            {"term": "BCKDHA vs BCKDHB vs DBT — biochemically identical at presentation",
             "definition": "MSUD caused by BCKDHA (Type 1A), BCKDHB (Type 1B), or DBT (Type 2) deficiency is biochemically IDENTICAL: same BCAA elevation pattern (Leu>>Ile,Val), same alloisoleucine (pathognomonic), same urine branched-chain ketoacids, same DNPH-positive urine, same BCKDH complex assay showing <5–25% of normal activity. Clinical phenotype (Classic/Intermediate/Intermittent/Thiamine-Responsive) is determined by the specific variant severity, not the gene. Only DNA sequencing (gene panel: BCKDHA + BCKDHB + DBT + DLD) definitively identifies the causative gene. PDH NORMAL + αKGDH NORMAL + DLD free enzyme NORMAL in all three — this combination distinguishes from DLD/E3 deficiency."},
            {"term": "DLD deficiency vs DBT deficiency — how to distinguish",
             "definition": "DLD (E3 subunit) deficiency causes MSUD-3, which superficially resembles DBT deficiency (same BCAA elevation, same alloisoleucine). KEY DISTINGUISHING FEATURES: (1) DLD deficiency also has lactic acidosis (PDH block: PDH reduced); (2) 2-hydroxyglutarate elevated (αKGDH block); (3) glycine mildly elevated (GCS partial block); (4) DLD free enzyme activity <10% of normal. In DBT deficiency: PDH NORMAL, αKGDH NORMAL, GCS NORMAL, DLD free enzyme NORMAL — ONLY the BCKDH complex is reduced. If any of PDH, αKGDH, or DLD free enzyme are reduced → suspect DLD deficiency, not isolated DBT or BCKDHA/B deficiency."},
            {"term": "Acute crisis management — leucine removal urgency",
             "definition": "Acute metabolic crisis (classic neonatal presentation or decompensation in known patient): (1) STOP all natural protein immediately; (2) HIGH-RATE IV glucose GIR 8–12 mg/kg/min to suppress muscle protein catabolism (most important anabolic intervention); (3) insulin 0.05–0.1 U/kg/h to drive leucine into anabolism and suppress muscle BCAA release; (4) IV BCAA-free amino acid solution to maintain anabolism and prevent further protein breakdown; (5) if leucine >1500 µmol/L or rising despite above: haemodialysis (HD) or continuous renal replacement therapy (CRRT) — most effective for rapid leucine clearance (leucine distributes mainly in plasma). Target: plasma leucine <300 µmol/L within 24–48 hours. Identical protocol for DBT, BCKDHA, and BCKDHB."},
        ],
        "diagnostic_thresholds": {
            "plasma_leucine_normal": "<200 µmol/L",
            "plasma_leucine_classic_msud": ">1000 µmol/L (acute crisis); typically 2000–4500 µmol/L in classic neonatal DBT",
            "plasma_leucine_target_maintenance": "<200 µmol/L (chronic diet therapy); <100 µmol/L ideal for optimal neurodevelopment",
            "alloisoleucine_diagnostic": ">5 µmol/L in plasma — PATHOGNOMONIC for MSUD (DBT/BCKDHA/BCKDHB)",
            "alloisoleucine_normal": "ABSENT (undetectable by standard plasma amino acids)",
            "bckdh_complex_activity_classic": "<5% of normal (fibroblasts or lymphocytes — total BCKDH complex assay)",
            "bckdh_complex_activity_intermediate": "5–25% of normal residual activity",
            "thiamine_trial_duration": "≥3 months at 10–20 mg/kg/day; repeat amino acids to assess leucine/allo-Ile response; ~5-10% of DBT patients respond (indirect mechanism via E1α stabilisation)",
            "leucine_crisis_threshold": ">400 µmol/L → acute encephalopathy risk; >1000 µmol/L → cerebral oedema; >2000 µmol/L → life-threatening",
            "hd_crrt_indication": "Plasma leucine rising rapidly OR >1500 µmol/L despite IV glucose + insulin — HD most effective for leucine removal",
            "pdh_dbt_deficiency": "NORMAL — PDH uses DLAT (its own E2), not DBT; isolated DBT deficiency does NOT affect PDH",
            "akgdh_dbt_deficiency": "NORMAL — αKGDH uses DLST (its own E2), not DBT; isolated DBT deficiency does NOT affect αKGDH",
            "dld_free_activity_dbt": "NORMAL — DLD protein is intact in isolated DBT deficiency; if reduced → suspect DLD/E3 deficiency",
        },
        "differential_diagnosis": [
            {"disease": "BCKDHA deficiency (MSUD Type 1A)",
             "distinguishing_features": "BIOCHEMICALLY IDENTICAL to DBT: same BCAA elevation, same alloisoleucine, same urine ketoacids, same BCKDH complex activity <5%. KEY DIFFERENCE: BCKDHA (E1α) contains the TPP-binding domain → thiamine response ~10-20% in BCKDHA vs ~5-10% in DBT. ONLY gene sequencing (BCKDHA vs DBT panel) definitively distinguishes. Gene panel mandatory."},
            {"disease": "BCKDHB deficiency (MSUD Type 1B)",
             "distinguishing_features": "BIOCHEMICALLY IDENTICAL to DBT: same BCAA elevation, same alloisoleucine, same ketoaciduria, same BCKDH complex activity <5%. BCKDHB (E1β) has NO TPP site → lowest thiamine response rate (~5%) vs DBT (~5-10%). Gene panel (BCKDHA + BCKDHB + DBT) required."},
            {"disease": "DLD deficiency (MSUD Type 3 / E3 deficiency)",
             "distinguishing_features": "DLD deficiency: BCAA elevated (BCKDH block) + alloisoleucine POSITIVE (resembles DBT). ADDITIONALLY: lactic acidosis (PDH block), 2-hydroxyglutarate elevated (αKGDH block), glycine elevated (GCS partial block), DLD free enzyme activity <10%. In DBT deficiency: PDH NORMAL, αKGDH NORMAL, GCS NORMAL, DLD free enzyme NORMAL — ONLY BCKDH complex is reduced."},
            {"disease": "Isovaleric acidaemia (IVA — IVDH/IVD deficiency)",
             "distinguishing_features": "IVA: isovaleryl-CoA dehydrogenase block (step 3 of Leu catabolism, DOWNSTREAM of BCKDH E2/DBT); C5-isovalerylcarnitine dramatically elevated in NBS (vs mildly elevated in MSUD); isovaleric acid ('sweaty feet' odour) in urine; alloisoleucine ABSENT (BCKDH/DBT intact); plasma Leu/Val/Ile NORMAL. Distinguish from DBT: no alloisoleucine in IVA vs pathognomonic in DBT."},
            {"disease": "Propionic acidaemia (PA) / Methylmalonic acidaemia (MMA)",
             "distinguishing_features": "PA/MMA: involve Val/Ile catabolism downstream (propionyl-CoA → methylmalonyl-CoA pathway); C3-carnitine elevated in NBS; methylmalonic acid in MMA urine; neutropaenia + thrombocytopaenia more prominent; alloisoleucine ABSENT; BCAA levels often normal (BCKDH/DBT intact upstream)."},
            {"disease": "Glutaric aciduria Type 2 (MADD / ETFDH/ETFA/ETFB deficiency)",
             "distinguishing_features": "MADD: multiple acyl-CoA dehydrogenase deficiency; broad acylcarnitine profile (C4, C5, C6, C8, C10 + glutaric, ethylmalonic, isovaleric acids in urine); riboflavin-responsive (FAD cofactor); alloisoleucine ABSENT; BCAA not selectively elevated; 'sweaty feet' odour from isovaleric acid."},
        ],
    }
