#!/usr/bin/env python3
"""BCKDHB (Branched-Chain Keto Acid Dehydrogenase E1β Subunit / MSUD Type 1B) Epilepsy Dashboard.

BCKDHB encodes the E1β subunit of the branched-chain alpha-ketoacid dehydrogenase complex (BCKDH),
which catalyses the second, irreversible step in branched-chain amino acid (BCAA) catabolism:

  BCKDH complex structure: BCKDHA+BCKDHB (E1, thiamine-dependent decarboxylase, α₂β₂ heterotetramer)
                            + DBT (E2, dihydrolipoamide branched-chain transacylase)
                            + DLD (E3, dihydrolipoamide dehydrogenase — SHARED with PDH, αKGDH, GCS)
  Step 1 (reversible): BCAA + α-KG → branched-chain α-ketoacid + glutamate (aminotransferase; reversible)
  Step 2 (irreversible, BLOCKED): branched-chain α-ketoacid → branched-chain acyl-CoA
    Leucine → α-ketoisocaproate (KIC) → isovaleryl-CoA + CO₂  [BCKDH uses TPP; E1 decarboxylates]
    Isoleucine → α-keto-β-methylvalerate (KMV) → 2-methylbutyryl-CoA + CO₂
    Valine → α-ketoisovalerate (KIV) → isobutyryl-CoA + CO₂

  BCKDHB (E1β) role: E1β is the STRUCTURAL subunit of the E1 heterotetrameric decarboxylase (α₂β₂);
    E1β does NOT contain the TPP (thiamine pyrophosphate) active site — that resides on E1α (BCKDHA);
    E1β is essential for E1α STABILITY and for correct assembly of the α₂β₂ E1 heterotetramer;
    without functional E1β, the E1 heterotetramer cannot assemble → E1α is unstable and degraded →
    the entire BCKDH complex loses decarboxylase activity → BCAA accumulate → MSUD Type 1B.
  BCKDHB LOF: E1 heterotetrameric decarboxylase cannot assemble or is structurally compromised;
    α-ketoacids accumulate; BCAA accumulate (Leu >> Ile, Val); alloisoleucine appears (PATHOGNOMONIC).

KEY STRUCTURAL DISTINCTION BETWEEN E1α (BCKDHA) AND E1β (BCKDHB):
  E1α (BCKDHA, 445 aa): catalytic — contains the TPP active site; decarboxylates branched-chain
    α-ketoacids; high-dose thiamine directly targets E1α → TPP saturation → stabilises mutant E1α
    → significant thiamine response rate (~10–20% in BCKDHA variants).
  E1β (BCKDHB, ~368 aa precursor, ~342 aa mature): structural — does NOT bind TPP directly;
    wraps around the E1α dimer to stabilise it; required for correct E1α folding and α₂β₂ assembly;
    thiamine CANNOT directly rescue E1β loss of function; ONLY a minority of BCKDHB variants
    (those causing mild misfolding that improves complex stability when TPP-E1α is saturated) may
    show any benefit → thiamine response rate is LOWER (~5%) in BCKDHB vs BCKDHA (~10–20%).
    Thiamine trial is STILL MANDATORY (Level A) for all MSUD patients — it does not harm and
    may help the rare assembly-stabilised variant.

PRIMARY ACCUMULATING TOXIC SPECIES (IDENTICAL TO BCKDHA):
  1. Leucine (most neurotoxic): plasma levels typically >2000 µmol/L in classic MSUD (normal <200)
  2. Alpha-ketoisocaproate (KIC, Leu α-keto acid): elevated; myelinopathic; directly toxic
  3. Alloisoleucine: pathognomonic — formed by stereoisomerisation of isoleucine-derived keto acids;
     alloisoleucine >5 µmol/L is DIAGNOSTIC; present ONLY in MSUD / BCKDH deficiency
  4. Alpha-ketoisovalerate (KIV, Val α-keto acid) and alpha-keto-β-methylvalerate (KMV, Ile keto acid)
  5. Branched-chain hydroxy acids: 2-hydroxyisocaproate, 2-hydroxyisovalerate (DNPH-positive)

LEUCINE NEUROTOXICITY MECHANISM (identical to BCKDHA):
  Leucine: enters brain via LAT1 (SLC7A5) large neutral amino acid transporter; competitively inhibits
  phenylalanine / tyrosine / tryptophan transport → depletes aromatic amino acid pool → inhibits
  myelination (tyrosine → DOPA pathway, phenylalanine → Phe pathway); mTOR hyperactivation;
  direct mitochondrial dysfunction; KIC and KICA (α-ketoisocaproate) myelinopathic at high concentrations.
  TREATMENT URGENCY: plasma leucine >400 µmol/L → acute encephalopathy; >1000 µmol/L → cerebral oedema;
  >2000 µmol/L → life-threatening brainstem herniation (cerebral oedema from osmotic effects + myelinopathy).

THIAMINE RESPONSE IN MSUD TYPE 1B (E1β SUBTYPE):
  ~5% of MSUD Type 1B (BCKDHB mutations) show any meaningful thiamine response — LOWER than BCKDHA
  because E1β does NOT contain the TPP-binding domain. Thiamine works by increasing intracellular TPP
  concentration → saturating residual E1α → improving E1α stability or partial decarboxylase activity.
  When E1β is structurally defective, even saturated TPP-E1α cannot fully assemble into a functional
  heterotetramer → limited thiamine benefit. Only BCKDHB variants that mildly impair assembly (not
  null alleles) might benefit indirectly. ALL patients must receive a 3-month trial (Level A evidence)
  regardless of BCKDHB vs BCKDHA genotype — it is safe and the rare responder benefits.

AUTOSOMAL RECESSIVE:
  BCKDHB gene: 6q14.1; Autosomal Recessive (AR)
  ~368 aa precursor (~342 aa mature after mitochondrial targeting sequence cleavage)
  MSUD prevalence: ~1:185,000 globally; no strong founder effect in BCKDHB (unlike BCKDHA Mennonite)
  OMIM gene: *248611; disease: #248600 (MAPLE SYRUP URINE DISEASE, TYPE IB)
  Note: MSUD types 1A (BCKDHA), 1B (BCKDHB), 2 (DBT) are biochemically identical at presentation;
  MSUD type 3 (DLD deficiency) has additional PDH/αKGDH/GCS blocks.

KEY VARIANTS (clinically established):
  p.Arg183Trp: E1α-E1β interface domain; disrupts heterotetramer assembly; severe classic MSUD
  p.Gly245Arg: hydrophobic core disruption; E1β misfolding; severe classic MSUD
  p.Leu164Phe: hydrophobic core; severe misfolding; classic MSUD neonatal
  p.Tyr301Cys: near BCKDHA docking surface; disrupts E1α-E1β interaction; severe
  p.Ala311Val: structural domain; milder; some residual E1β; intermediate MSUD
  c.866-1G>A: splice acceptor null allele; neonatal classic; loss of E1β expression

CLINICAL PRESENTATIONS (identical to BCKDHA):
  1. Classic MSUD (70%): neonatal presentation day 3–7; maple syrup odour; encephalopathy;
     hypotonia progressing to hypertonia; seizures; cerebral oedema; death if untreated within days
  2. Intermediate MSUD (10%): presentation 5 months–7 years; chronic protein intolerance;
     episodic neurological regression; milder plasma BCAA elevation
  3. Intermittent MSUD (10%): episodic crises during catabolic stress (illness, surgery, fasting);
     normal development between crises; baseline BCAA often borderline
  4. Thiamine-responsive MSUD (10% phenotype label, but ACTUAL response rate ~5% in BCKDHB):
     E1β does not bind TPP; rare assembly-stabilising variants may show minor benefit;
     trial mandatory but expected response rate lower than BCKDHA

TREATMENT PRIORITIES (identical to BCKDHA):
  1. BCAA-restricted diet with synthetic formula (Level A): acute AND chronic management;
     Leu restriction is the most critical component (Leu most toxic); use BCAA-free synthetic formula
     to provide all other amino acids; natural protein provided only to meet minimum biological needs;
     leucine intake target: <100 mg/kg/day (maintenance); adjust to target plasma Leu <200 µmol/L
  2. Thiamine (B1, Level A): 10–20 mg/kg/day (max 200 mg/day) for ALL patients as initial trial;
     ~5% of BCKDHB patients may show response (lower than BCKDHA ~10–20% because E1β lacks TPP domain);
     trial for ≥3 months before declaring non-responsive; safe and mandatory regardless
  3. Acute crisis management (Level A): IV glucose GIR 8–12 mg/kg/min + insulin 0.05–0.1 U/kg/h;
     BCAA-free amino acid infusion; HD/continuous CRRT if leucine rising rapidly or >1500 µmol/L
  4. Liver transplantation (Level B): corrects metabolic defect (liver provides ~80% BCKDH activity);
     post-transplant leucine tolerance markedly improved; BCAA diet still needed but less restrictive;
     carnitine supplementation post-transplant; best results when transplanted before 10 years
  5. Carnitine (Level B): secondary carnitine depletion occurs due to esterification of branched-chain
     acyl-CoA species; monitor free/total carnitine; supplement if low
  6. LEV — Levetiracetam (Level B): first-line AED for seizures in MSUD

ABSOLUTE CONTRAINDICATIONS:
  VPA (Valproate): ABSOLUTE CI — carnitine depletion (worsens secondary carnitine deficiency);
    inhibits BCKDH complex (worsens metabolic block); hepatotoxicity in mitochondrial disorders
  Fasting: EXTREME HAZARD — BCAA mobilisation from muscle protein breakdown → leucine surge
  High-protein/unconstrained diet: EXTREME HAZARD — BCAA ingestion → leucine crisis
  Surgery/anaesthesia without metabolic coverage: HAZARD — catabolic stress → BCAA surge
  Infection/febrile illness without BCAA monitoring: HAZARD — catabolic state → crisis
"""

import random
from datetime import datetime, timezone

random.seed(47)  # reproducible cohort (BCKDHB seed 47; BCKDHA: 46; DLST: 45; OGDH: 44; PDHX: 43)


PHENOTYPES = [
    ("Classic MSUD (Neonatal)", 0.70),
    ("Intermediate MSUD", 0.10),
    ("Intermittent MSUD", 0.10),
    ("Thiamine-Responsive MSUD", 0.10),
]

VARIANTS = [
    "p.Arg183Trp",    # E1α-E1β interface; severe classic
    "p.Gly245Arg",    # hydrophobic core misfolding; severe classic
    "p.Leu164Phe",    # hydrophobic core; neonatal classic
    "p.Tyr301Cys",    # near BCKDHA docking surface; severe
    "p.Ala311Val",    # structural; milder; intermediate
]


def _patients():
    pts = []
    pid = 1
    for phenotype_label, fraction in PHENOTYPES:
        n_pheno = max(1, round(40 * fraction))
        is_classic = "Classic" in phenotype_label
        is_intermediate = "Intermediate" in phenotype_label
        is_intermittent = "Intermittent" in phenotype_label
        is_thiamine = "Thiamine" in phenotype_label

        for _ in range(n_pheno):
            sex = random.choice(["M", "F"])

            # Onset age (months)
            if is_classic:
                onset = round(random.uniform(0.1, 0.5), 2)   # day 3–14 neonatal
            elif is_intermediate:
                onset = round(random.uniform(6, 36), 1)
            elif is_intermittent:
                onset = round(random.uniform(12, 84), 1)
            else:  # thiamine-responsive phenotype label (actual response rate ~5% in BCKDHB)
                onset = round(random.uniform(1, 24), 1)

            # Plasma leucine (µmol/L) — normal <200
            if is_classic:
                leu = round(random.uniform(1800, 3800))
            elif is_intermediate:
                leu = round(random.uniform(400, 1200))
            elif is_intermittent:
                leu = round(random.uniform(200, 800))
            else:  # thiamine-responsive label
                leu = round(random.uniform(300, 1000))

            # Plasma isoleucine (µmol/L) — normal <100
            ile = round(random.uniform(0.35 * leu, 0.55 * leu))

            # Plasma valine (µmol/L) — normal <300
            val = round(random.uniform(0.40 * leu, 0.60 * leu))

            # Alloisoleucine (µmol/L) — pathognomonic >5; normally absent
            if is_classic:
                allo_ile = round(random.uniform(60, 300))
            elif is_intermediate:
                allo_ile = round(random.uniform(20, 100))
            elif is_intermittent:
                allo_ile = round(random.uniform(5, 50))
            else:
                allo_ile = round(random.uniform(10, 80))

            # Urine alpha-ketoacids (DNPH test): elevated in all; GC-MS urine organic acids
            urine_kic = round(random.uniform(800 if is_classic else 100, 4000 if is_classic else 1200))

            # Plasma lactate — secondary to mitochondrial dysfunction
            lactate = round(random.uniform(3.5 if is_classic else 1.5, 14 if is_classic else 5.0), 1)

            # Maple syrup odour
            maple_odour = is_classic or (not is_intermittent and random.random() < 0.6)

            # Cerebral oedema (acute crisis)
            cerebral_oedema = is_classic and random.random() < 0.55

            # MRI findings
            mri_diffusion_restriction = is_classic and random.random() < 0.80  # white matter diffusion restriction
            cerebellar_myelinopathy = (is_classic or is_intermediate) and random.random() < 0.45
            basal_ganglia_involvement = is_classic and random.random() < 0.35

            # Thiamine trial — mandatory for ALL patients; actual response rate ~5% for BCKDHB
            on_thiamine = True
            # BCKDHB thiamine response rate ~5% (lower than BCKDHA ~10-20%; E1β lacks TPP domain)
            # Only rare assembly-stabilising variants (e.g. p.Ala311Val) may benefit indirectly
            thiamine_responsive = (is_thiamine and random.random() < 0.50) or (
                is_intermediate and random.random() < 0.08
            )

            # Diet adherence
            on_bcaa_restricted_diet = True
            diet_well_controlled = thiamine_responsive or (not is_classic and random.random() < 0.65)

            # Seizures
            seizures = is_classic or random.random() < 0.40
            dre = seizures and random.random() < (0.35 if is_classic else 0.15)

            # Liver transplant
            liver_transplant = random.random() < 0.20

            # Carnitine supplementation
            on_carnitine = True  # almost all need
            carnitine_deficient = random.random() < 0.60

            # VPA avoided
            vpa_avoided = random.random() < 0.92

            # Variant assignment
            if is_classic:
                variant = random.choice(VARIANTS[:4])  # severe variants predominate in classic
            elif is_thiamine:
                variant = "p.Ala311Val"   # mildest structural variant; may partially benefit
            elif is_intermediate:
                variant = random.choice(["p.Ala311Val", "p.Tyr301Cys"])
            else:
                variant = random.choice(VARIANTS)

            pts.append({
                "id": f"BCKDHB-{pid:03d}",
                "sex": sex,
                "phenotype": phenotype_label,
                "onset_age_months": onset,
                "plasma_leucine_umol": leu,
                "plasma_isoleucine_umol": ile,
                "plasma_valine_umol": val,
                "alloisoleucine_umol": allo_ile,
                "urine_kic_umol_mmolCr": urine_kic,
                "plasma_lactate_mmol": lactate,
                "maple_odour": maple_odour,
                "cerebral_oedema": cerebral_oedema,
                "mri_diffusion_restriction": mri_diffusion_restriction,
                "cerebellar_myelinopathy": cerebellar_myelinopathy,
                "basal_ganglia_involvement": basal_ganglia_involvement,
                "on_thiamine": on_thiamine,
                "thiamine_responsive": thiamine_responsive,
                "on_bcaa_restricted_diet": on_bcaa_restricted_diet,
                "diet_well_controlled": diet_well_controlled,
                "seizures": seizures,
                "dre": dre,
                "liver_transplant": liver_transplant,
                "on_carnitine": on_carnitine,
                "carnitine_deficient": carnitine_deficient,
                "vpa_avoided": vpa_avoided,
                "variant": variant,
            })
            pid += 1
    return pts


_COHORT = _patients()


def get_overview():
    """Return top-level dashboard summary, KPIs, complex diagram, pathway, and drug risk table.

    Covers gene identity, BCKDH complex role of E1β, phenotype distribution, and cohort KPIs
    derived from the 40-patient reproducible cohort (seed 47).
    """
    c = _COHORT
    n = len(c)
    male_n = sum(1 for p in c if p["sex"] == "M")
    avg_leu = round(sum(p["plasma_leucine_umol"] for p in c) / n)
    avg_ile = round(sum(p["plasma_isoleucine_umol"] for p in c) / n)
    avg_val = round(sum(p["plasma_valine_umol"] for p in c) / n)
    avg_allo = round(sum(p["alloisoleucine_umol"] for p in c) / n)
    maple_n = sum(1 for p in c if p["maple_odour"])
    cerebral_oedema_n = sum(1 for p in c if p["cerebral_oedema"])
    diffusion_n = sum(1 for p in c if p["mri_diffusion_restriction"])
    thiamine_resp_n = sum(1 for p in c if p["thiamine_responsive"])
    transplant_n = sum(1 for p in c if p["liver_transplant"])
    dre_n = sum(1 for p in c if p["dre"])
    vpa_ok = sum(1 for p in c if p["vpa_avoided"])
    carnitine_def_n = sum(1 for p in c if p["carnitine_deficient"])

    return {
        "dashboard": "BCKDHB Epilepsy (Maple Syrup Urine Disease Type 1B — BCKDH E1β Subunit Deficiency)",
        "gene": "BCKDHB",
        "protein": "Branched-Chain Alpha-Keto Acid Dehydrogenase E1β Subunit (BCKDH complex E1 structural subunit)",
        "locus": "6q14.1",
        "aa_length": 368,
        "cofactor": (
            "No direct cofactor — E1β is the STRUCTURAL (not catalytic) subunit of the BCKDH E1 "
            "heterotetrameric decarboxylase; TPP (thiamine pyrophosphate) binds E1α (BCKDHA), NOT E1β; "
            "thiamine trial is STILL MANDATORY (Level A) for all MSUD patients including BCKDHB, but "
            "the expected response rate is LOWER (~5%) than BCKDHA (~10–20%) precisely because E1β "
            "does not contain the TPP-binding domain — indirect benefit via improved complex stability "
            "is only possible in rare assembly-competent variants (e.g. p.Ala311Val)"
        ),
        "mechanism": (
            "BCKDHB encodes the E1β structural subunit of the BCKDH complex. E1β does NOT catalyse "
            "any reaction directly but is REQUIRED for: (1) E1α stability — without E1β, E1α is "
            "unstable and degraded; (2) correct assembly of the α₂β₂ E1 heterotetramer — the two E1β "
            "subunits wrap around the two E1α subunits to form the functional decarboxylase. "
            "BCKDHB LOF → E1 heterotetramer cannot assemble → entire BCKDH complex non-functional → "
            "branched-chain α-keto acids (KIC/KMV/KIV) accumulate → BCAA (Leu/Ile/Val) accumulate → "
            "alloisoleucine appears (PATHOGNOMONIC). Clinical presentation IDENTICAL to BCKDHA (Type 1A). "
            "Leucine (most neurotoxic): inhibits LAT1 aromatic AA transport → myelination failure + mTOR hyperactivation."
        ),
        "omim_gene": "*248611",
        "omim_disease": "#248600 (MAPLE SYRUP URINE DISEASE, TYPE IB)",
        "inheritance": "Autosomal Recessive (AR, 6q14.1) — both sexes equally affected; no sex bias; no strong founder effect in BCKDHB (unlike BCKDHA Mennonite p.Tyr393Asn)",
        "pathognomonic_marker": (
            "Alloisoleucine >5 µmol/L (plasma, simultaneously measured with Leu/Ile/Val) — "
            "PATHOGNOMONIC for MSUD (any BCKDH deficiency type 1A/1B/2). "
            "Alloisoleucine is formed by spontaneous stereoisomerisation of isoleucine-derived alpha-keto "
            "acid (KMV) intermediates; absent from normal plasma; its presence confirms BCKDH block. "
            "Confirmed by: NBS tandem MS (elevated C5:1 / leucine/isoleucine peaks), plasma amino acids "
            "(Leu/Ile/Val elevation + alloisoleucine), urine organic acids (KIC, KMV, KIV by GC-MS), DNPH test."
        ),
        "key_distinguishing_feature": (
            "MSUD Type 1B (BCKDHB) vs DLD deficiency: BCKDHB — isolated BCKDH block → BCAA↑ + alloisoleucine "
            "PATHOGNOMONIC + PDH NORMAL + αKGDH NORMAL + GCS NORMAL + DLD free enzyme NORMAL. "
            "DLD deficiency: ALL FOUR complexes blocked (PDH + αKGDH + BCKDH + GCS) → BCAA↑ + lactate↑↑ + "
            "2-HG↑ + glycine↑ + DLD free enzyme <10%. "
            "BCKDHB vs BCKDHA vs DBT: biochemically IDENTICAL — gene panel mandatory to distinguish. "
            "BCKDHB-specific: E1β is the structural (not catalytic) subunit — thiamine response rate ~5% "
            "(lower than BCKDHA ~10–20%) because E1β does NOT contain the TPP-binding domain."
        ),
        "structural_brain_hallmark": (
            "White matter diffusion restriction — myelinopathy: symmetric T2 FLAIR hyperintensity and "
            "DWI restriction in posterior limbs of internal capsules, cerebral peduncles, posterior "
            "white matter (classic neonatal — 'MSUD myelinopathy pattern'; 80% in classic BCKDHB); "
            "Cerebellar white matter involvement (45%); basal ganglia involvement (35% in classic/severe); "
            "cerebral oedema (55% acute neonatal crisis) — resolves with BCAA normalisation."
        ),
        "kpis": {
            "cohort_n": n,
            "male_pct": round(100 * male_n / n),
            "avg_plasma_leucine_umol": avg_leu,
            "avg_plasma_isoleucine_umol": avg_ile,
            "avg_plasma_valine_umol": avg_val,
            "avg_alloisoleucine_umol": avg_allo,
            "maple_odour_pct": round(100 * maple_n / n),
            "cerebral_oedema_acute_pct": round(100 * cerebral_oedema_n / n),
            "mri_diffusion_restriction_pct": round(100 * diffusion_n / n),
            "thiamine_responsive_pct": round(100 * thiamine_resp_n / n),
            "liver_transplant_pct": round(100 * transplant_n / n),
            "carnitine_deficient_pct": round(100 * carnitine_def_n / n),
            "dre_pct": round(100 * dre_n / n),
            "vpa_avoided_pct": round(100 * vpa_ok / n),
        },
        "bckdh_complex_components": [
            {"component": "E1α — BCKDHA",
             "function": "Catalytic subunit of E1 decarboxylase (α₂β₂ heterotetramer); contains the TPP active site; decarboxylates branched-chain α-ketoacids (KIC/KMV/KIV) via TPP-hydroxyalkyl-TPP intermediate; transfers acyl group to DBT (E2) lipoamide arm. BCKDHA LOF → E1 non-functional → BCAA accumulate (MSUD Type 1A). Thiamine response rate ~10–20% in BCKDHA (TPP directly targets E1α active site)."},
            {"component": "E1β — BCKDHB ← THIS GENE",
             "function": "STRUCTURAL subunit of E1 decarboxylase (α₂β₂ heterotetramer); does NOT contain the TPP active site (that is on E1α/BCKDHA); required for E1α stability and correct assembly of the α₂β₂ E1 heterotetramer; without E1β, E1α is unstable and degraded → BCKDH complex cannot form. BCKDHB LOF causes MSUD Type 1B (biochemically identical to Type 1A). Thiamine response rate LOWER (~5%) than BCKDHA because E1β does NOT bind TPP directly — only rare assembly-stabilising variants benefit indirectly."},
            {"component": "E2 — DBT (dihydrolipoamide branched-chain transacylase)",
             "function": "24-mer cubic core with lipoyl domains; receives acyl group from E1 via lipoamide arm; transfers to CoA → branched-chain acyl-CoA; DBT LOF causes MSUD Type 2."},
            {"component": "E3 — DLD (dihydrolipoamide dehydrogenase, FAD homodimer)",
             "function": "Regenerates DBT lipoamide arm; SHARED with PDH, αKGDH, GCS complexes; DLD is INTACT in BCKDHB deficiency."},
            {"component": "BCKDK (kinase) / PPM2C (phosphatase)",
             "function": "BCKDK phosphorylates E1α (Ser337, Ser341) → inactivates BCKDH; BT2 inhibits BCKDK → activates BCKDH (therapeutic target in research). Not relevant to loss-of-function MSUD."},
        ],
        "bcaa_catabolism_pathway": {
            "step_1_reversible": "BCAA + α-KG → branched-chain α-ketoacid + Glu (aminotransferase BCAT1/BCAT2; reversible; explains why diet restriction can normalise plasma levels)",
            "step_2_irreversible_blocked": "Branched-chain α-ketoacid → acyl-CoA + CO₂ (BCKDH complex; BLOCKED in BCKDHB deficiency because E1β loss → E1 heterotetramer cannot assemble; irreversible step makes accumulation rapid)",
            "downstream_leu": "KIC (Leu) → isovaleryl-CoA → 3-methylcrotonyl-CoA → HMG-CoA → acetyl-CoA + acetoacetate",
            "downstream_val": "KIV (Val) → isobutyryl-CoA → propionyl-CoA → methylmalonyl-CoA → succinyl-CoA (joins TCA)",
            "downstream_ile": "KMV (Ile) → 2-methylbutyryl-CoA → propionyl-CoA + acetyl-CoA",
            "alloisoleucine_formation": "KMV (Ile keto acid) ⇌ 2-hydroxy-3-methylvalerate intermediates → spontaneous transamination back → alloisoleucine (D-allo-isoleucine); PATHOGNOMONIC",
        },
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
             "mechanism": "Triple CI: (1) carnitine depletion → worsens secondary carnitine deficiency; (2) direct BCKDH complex inhibition → worsens BCAA accumulation; (3) mitochondrial hepatotoxicity. Equivalent to POLG1 contraindication. NEVER use in MSUD (any type including BCKDHB)."},
            {"drug": "Fasting / prolonged NPO", "risk": "EXTREME HAZARD",
             "mechanism": "Fasting → muscle protein catabolism → massive BCAA release → leucine surge → acute encephalopathy. IV glucose (GIR ≥8 mg/kg/min) + BCAA-free formula MANDATORY during any fasting period."},
            {"drug": "High natural protein diet", "risk": "EXTREME HAZARD",
             "mechanism": "All natural proteins contain BCAA; unrestricted diet → leucine crisis within hours. Synthetic BCAA-free formula mandatory as protein source."},
            {"drug": "Surgery/anaesthesia without metabolic protocol", "risk": "HAZARD",
             "mechanism": "Catabolic stress + fasting → BCAA surge; requires perioperative IV glucose + amino acid infusion + frequent leucine monitoring."},
            {"drug": "Corticosteroids (systemic)", "risk": "CAUTION",
             "mechanism": "Catabolic effect → muscle protein breakdown → BCAA release; use only if essential; increase BCAA monitoring frequency."},
            {"drug": "Phenobarbital", "risk": "CAUTION",
             "mechanism": "CYP induction may alter thiamine/carnitine metabolism; sedation can mask metabolic crisis; monitor closely."},
        ],
        "phenotype_distribution": [
            {"label": "Classic MSUD (Neonatal Crisis)", "pct": 70, "color": "#b71c1c"},
            {"label": "Intermediate MSUD", "pct": 10, "color": "#e65100"},
            {"label": "Intermittent MSUD (Episodic)", "pct": 10, "color": "#f9a825"},
            {"label": "Thiamine-Responsive MSUD", "pct": 10, "color": "#558b2f"},
        ],
        "generated": datetime.now(timezone.utc).isoformat(),
    }


def get_breakdown():
    """Return detailed biomarkers, key variants, patient sample, seizure/trigger types, treatments.

    First 10 of 40 patients shown in patients_sample. Biomarker column uses 'bckdhb_range'.
    Treatments reflect BCKDHB-specific thiamine response rate (~5%, lower than BCKDHA ~10–20%).
    """
    c = _COHORT
    sample = c[:10]

    return {
        "biomarkers": [
            {"name": "Plasma Leucine", "normal": "<200 µmol/L",
             "bckdhb_range": ">2000 µmol/L (classic acute); 400–1200 µmol/L (intermediate); 200–800 µmol/L (intermittent/thiamine-label)",
             "significance": "PRIMARY toxic species; most neurotoxic BCAA; LAT1 transporter competition → myelination failure; mTOR hyperactivation; leucine >1000 µmol/L → cerebral oedema risk; target <200 µmol/L maintenance. Identical elevation pattern to BCKDHA — gene panel required to distinguish."},
            {"name": "Alloisoleucine (D-allo-Ile)", "normal": "ABSENT (undetectable in normal plasma)",
             "bckdhb_range": ">5 µmol/L (any MSUD subtype) — PATHOGNOMONIC",
             "significance": "PATHOGNOMONIC MARKER for MSUD: formed by spontaneous stereoisomerisation of KMV (Ile keto acid) intermediates; absent from all non-MSUD conditions; present in BCKDHA/BCKDHB/DBT deficiency equally; persists even during diet control if not fully normalised"},
            {"name": "Plasma Isoleucine", "normal": "<100 µmol/L",
             "bckdhb_range": "300–1800 µmol/L (classic); 150–600 µmol/L (intermediate)",
             "significance": "Second most elevated BCAA; source of alloisoleucine (via KMV); neurotoxic at high levels; less toxic than leucine but monitored in parallel"},
            {"name": "Plasma Valine", "normal": "<300 µmol/L",
             "bckdhb_range": "500–2000 µmol/L (classic); 200–900 µmol/L (intermediate)",
             "significance": "Most elevated in plasma (by mass) but least neurotoxic; Val catabolism enters propionyl-CoA pathway → succinyl-CoA (joins TCA)"},
            {"name": "Urine alpha-ketoacids (GC-MS)", "normal": "Trace or absent",
             "bckdhb_range": "KIC 800–4000 µmol/mmol Cr; KIV, KMV elevated proportionally",
             "significance": "Urine organic acids (GC-MS): alpha-ketoisocaproate (KIC, Leu), alpha-ketoisovalerate (KIV, Val), alpha-keto-β-methylvalerate (KMV, Ile) elevated; KIC is directly myelinopathic; DNPH test positive (ketoacids)"},
            {"name": "DNPH (dinitrophenylhydrazine) urine test", "normal": "Negative",
             "bckdhb_range": "POSITIVE — yellow precipitate (alpha-ketoacids react with DNPH)",
             "significance": "Bedside screening test for branched-chain ketoaciduria; positive in classic/intermediate MSUD; may be negative in intermittent forms between crises; not specific (also positive in PKU, tyrosinaemia) — confirm with GC-MS"},
            {"name": "Plasma Lactate", "normal": "<2.2 mmol/L",
             "bckdhb_range": "3.5–14 mmol/L (classic acute); 1.5–5 mmol/L (stable)",
             "significance": "Secondary to mitochondrial dysfunction from leucine toxicity and KIC accumulation; NOT as primary a feature as in PDH/αKGDH deficiency; lactic acidosis worst in acute crisis"},
            {"name": "Urine BCAA and branched-chain hydroxy acids", "normal": "Absent/trace",
             "bckdhb_range": "2-hydroxyisocaproate, 2-hydroxyisovalerate elevated",
             "significance": "Branched-chain hydroxy acids from reduction of keto acids; further confirms BCKDH block; detectable by GC-MS urine organic acids"},
            {"name": "BCKDH enzyme activity (fibroblasts/lymphocytes)", "normal": ">60% normal",
             "bckdhb_range": "<5% of normal (classic); 5–25% (intermediate/thiamine-label)",
             "significance": "Definitive enzymatic diagnosis; measured as 14CO₂ release from [1-14C]-leucine; post-thiamine trial activity increase confirms thiamine-responsiveness (expected in <5% of BCKDHB patients vs ~10–20% in BCKDHA)"},
            {"name": "PDH enzyme activity", "normal": ">60% normal",
             "bckdhb_range": "NORMAL — PDH intact in isolated BCKDHB deficiency",
             "significance": "KEY NEGATIVE: PDH normal confirms only BCKDH is deficient; DLD/E3 is shared but DLD protein is intact; PDH reduced would suggest DLD deficiency"},
            {"name": "αKGDH complex activity", "normal": ">60% normal",
             "bckdhb_range": "NORMAL — αKGDH intact in isolated BCKDHB deficiency",
             "significance": "KEY NEGATIVE: αKGDH normal confirms only BCKDH is deficient; 2-oxoglutarate NORMAL (no OGDH/DLST/DLD block)"},
            {"name": "DLD free enzyme activity", "normal": ">60% normal",
             "bckdhb_range": "NORMAL — DLD protein intact in BCKDHB deficiency",
             "significance": "KEY NEGATIVE: DLD free activity normal excludes DLD deficiency (where all four complexes are blocked); in DLD deficiency, DLD free activity <10%"},
        ],
        "key_variants": [
            {"variant": "p.Arg183Trp", "effect": "E1α-E1β interface domain; Arg183 is a critical contact residue at the E1α-E1β heterodimerisation interface; Trp substitution disrupts electrostatic and hydrogen-bond contacts → heterotetramer assembly fails → E1α destabilised and degraded; null-equivalent complex activity", "phenotype": "Classic MSUD — Severe"},
            {"variant": "p.Gly245Arg", "effect": "Hydrophobic core of E1β; Gly245 is a conserved glycine in a tight structural turn; Arg introduces a large charged side chain → severe local misfolding of E1β hydrophobic core → E1β cannot fold correctly → E1α-E1β assembly fails; classic null phenotype", "phenotype": "Classic MSUD — Severe"},
            {"variant": "p.Leu164Phe", "effect": "Hydrophobic core of E1β; Leu164 participates in conserved hydrophobic packing cluster; Phe substitution creates steric clash → severe misfolding → E1β unstable → heterotetramer cannot form; neonatal crisis", "phenotype": "Classic MSUD — Neonatal"},
            {"variant": "p.Tyr301Cys", "effect": "Near the BCKDHA (E1α) docking surface of E1β; Tyr301 hydrogen-bonds with the E1α interface; Cys substitution ablates this contact and may form spurious disulphide → E1α-E1β interaction disrupted → heterotetramer destabilised; severe but may retain very minor residual activity", "phenotype": "Classic MSUD — Severe"},
            {"variant": "p.Ala311Val", "effect": "Structural domain of E1β; Ala311 lies in a moderately packed region; Val substitution causes partial steric perturbation but does not abolish E1β folding; some residual E1β structure and partial E1α-E1β assembly preserved → intermediate phenotype; this variant (minor structural impact) is the type most likely to show any indirect thiamine benefit in BCKDHB (~5% response rate)", "phenotype": "Intermediate MSUD — milder; some residual E1β"},
            {"variant": "c.866-1G>A (splice acceptor)", "effect": "Splice acceptor mutation at intron 8–exon 9 boundary; abolishes canonical splice site → exon 9 skipping or cryptic splice activation → frameshift → premature stop → NMD → complete loss of E1β expression; null allele", "phenotype": "Classic MSUD — Neonatal (null allele)"},
        ],
        "patients_sample": [
            {
                "id": p["id"],
                "sex": p["sex"],
                "phenotype": p["phenotype"],
                "onset_age_months": p["onset_age_months"],
                "plasma_leucine_umol": p["plasma_leucine_umol"],
                "plasma_isoleucine_umol": p["plasma_isoleucine_umol"],
                "plasma_valine_umol": p["plasma_valine_umol"],
                "alloisoleucine_umol": p["alloisoleucine_umol"],
                "maple_odour": p["maple_odour"],
                "cerebral_oedema": p["cerebral_oedema"],
                "mri_diffusion_restriction": p["mri_diffusion_restriction"],
                "thiamine_responsive": p["thiamine_responsive"],
                "diet_well_controlled": p["diet_well_controlled"],
                "liver_transplant": p["liver_transplant"],
                "dre": p["dre"],
                "vpa_avoided": p["vpa_avoided"],
                "variant": p["variant"],
            }
            for p in sample
        ],
        "seizure_types": [
            {"type": "Tonic seizures (neonatal)", "pct": 65},
            {"type": "Myoclonic seizures", "pct": 55},
            {"type": "Infantile spasms (IS) / West syndrome", "pct": 45},
            {"type": "Focal seizures (EEG-confirmed)", "pct": 40},
            {"type": "Absence-like with metabolic encephalopathy", "pct": 30},
            {"type": "Status epilepticus (metabolic crisis)", "pct": 50},
            {"type": "Burst suppression (neonatal EEG)", "pct": 60},
        ],
        "trigger_types": [
            {"trigger": "Fasting / catabolism", "pct": 90},
            {"trigger": "Febrile illness (any infection)", "pct": 82},
            {"trigger": "High natural protein intake", "pct": 75},
            {"trigger": "Surgery / perioperative stress", "pct": 65},
            {"trigger": "Psychological stress (older patients)", "pct": 40},
            {"trigger": "VPA administration (iatrogenic crisis)", "pct": 15},
        ],
        "treatments": [
            {"drug": "BCAA-restricted diet + BCAA-free synthetic formula (Level A)", "level": "A", "response_pct": 85,
             "color": "#1b5e20",
             "note": "First-line chronic management; Leu restriction critical (target Leu <200 µmol/L); BCAA-free formula provides other amino acids, vitamins, minerals; natural protein only to meet minimum biological need; must be lifelong. Identical to BCKDHA management."},
            {"drug": "Thiamine B1 — high dose (Level A)", "level": "A", "response_pct": 5,
             "color": "#0277bd",
             "note": "10–20 mg/kg/day (max 200 mg/day); ALL BCKDHB patients receive mandatory trial ≥3 months. IMPORTANT: thiamine response rate is LOWER in BCKDHB (~5%) than BCKDHA (~10–20%) because E1β does NOT contain the TPP-binding domain — E1α (BCKDHA) is the TPP-binding subunit, not E1β. Only rare assembly-stabilising variants (e.g. p.Ala311Val) may benefit indirectly when TPP-saturated E1α forms a more stable partial complex with misfolded E1β. Non-responders (majority) do not worsen. Lifelong in the rare responder."},
            {"drug": "IV glucose + insulin (Level A — acute crisis)", "level": "A", "response_pct": 90,
             "color": "#0277bd",
             "note": "GIR 8–12 mg/kg/min IV glucose + insulin 0.05–0.1 U/kg/h; anabolic insulin suppresses muscle BCAA release; BCAA-free amino acid infusion in parallel; HD/CRRT if Leu still rising >1500 µmol/L"},
            {"drug": "Carnitine supplementation (Level B)", "level": "B", "response_pct": 60,
             "color": "#4a148c",
             "note": "Secondary carnitine depletion from branched-chain acylcarnitine esterification; monitor free/total carnitine ratio; supplement if free carnitine <25 µmol/L; 50–100 mg/kg/day"},
            {"drug": "Liver transplantation (Level B)", "level": "B", "response_pct": 80,
             "color": "#4a148c",
             "note": "Liver provides ~80% of whole-body BCKDH activity; transplant corrects metabolic defect in majority; allows liberalised diet (still some restriction needed); best outcome if transplanted before 10 years; carnitine supplement post-transplant"},
            {"drug": "LEV — Levetiracetam (Level B)", "level": "B", "response_pct": 60,
             "color": "#1b5e20",
             "note": "First-line AED for seizures in MSUD; no BCAA pathway interaction; no hepatotoxicity; safe metabolic profile"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
             "mechanism": "Triple CI: carnitine depletion + BCKDH direct inhibition + mitochondrial hepatotoxicity. NEVER use in MSUD (any type including BCKDHB)."},
            {"drug": "Fasting", "risk": "EXTREME HAZARD",
             "mechanism": "Muscle catabolism → BCAA surge → leucine crisis within hours. IV glucose mandatory."},
            {"drug": "Unrestricted natural protein", "risk": "EXTREME HAZARD",
             "mechanism": "All natural proteins contain BCAA; any unrestricted meal → leucine surge."},
            {"drug": "Surgery without metabolic protocol", "risk": "HAZARD",
             "mechanism": "Catabolic stress + NPO → BCAA crisis. Requires perioperative IV glucose + BCAA monitoring."},
        ],
    }


def get_definitions():
    """Return gene card, key concepts, diagnostic thresholds, and differential diagnosis.

    Emphasises E1β structural role, lower thiamine response rate (~5%) vs E1α/BCKDHA (~10–20%),
    and the biochemical indistinguishability of MSUD Types 1A/1B/2 requiring gene panel.
    """
    return {
        "gene_card": {
            "Gene": "BCKDHB (Branched-Chain Keto Acid Dehydrogenase E1β Subunit)",
            "Also known as": "BCKDE1B; MSUD1B",
            "Subunit": "E1β STRUCTURAL subunit of BCKDH complex (with BCKDHA forming E1α₂β₂ heterotetramer); does NOT contain the TPP active site",
            "Chromosome": "6q14.1",
            "Protein length": "~368 aa precursor (~342 aa mature after MTS cleavage)",
            "Cofactor": "None directly — E1β is structural, NOT catalytic; TPP binds E1α (BCKDHA), not E1β; thiamine trial mandatory (Level A) but response rate LOWER (~5%) than BCKDHA (~10–20%)",
            "Function": "Required for E1α stability and correct assembly of the α₂β₂ E1 heterotetrameric decarboxylase; BCKDHB LOF → E1α unstable → heterotetramer cannot form → BCKDH complex non-functional → BCAA accumulate",
            "Complex partners": "BCKDHA (E1α catalytic/TPP subunit) + DBT (E2, lipoamide transacylase) + DLD (E3, FAD homodimer, SHARED with PDH/αKGDH/GCS)",
            "Inheritance": "Autosomal Recessive (AR) — biallelic LOF",
            "OMIM gene": "*248611",
            "OMIM disease": "#248600 (MAPLE SYRUP URINE DISEASE, TYPE IB)",
            "Prevalence": "~1:185,000 globally (combined MSUD); no strong founder effect in BCKDHB unlike BCKDHA Mennonite (p.Tyr393Asn)",
            "Biochemical block": "Step 2 of BCAA catabolism — irreversible oxidative decarboxylation BLOCKED (E1 heterotetramer cannot assemble without E1β)",
            "Pathognomonic marker": "Alloisoleucine >5 µmol/L (plasma) — present ONLY in MSUD; identical to BCKDHA and DBT",
            "Primary toxic species": "Leucine (most neurotoxic); KIC (alpha-ketoisocaproate, myelinopathic)",
            "First-line treatment": "BCAA-restricted diet + BCAA-free synthetic formula (Level A) + thiamine trial (Level A; ~5% response in BCKDHB)",
            "ABSOLUTE CI": "Valproate (VPA) — triple contraindication (carnitine depletion + BCKDH inhibition + hepatotoxicity)",
        },
        "key_concepts": [
            {"term": "BCKDHB E1β — structural role vs E1α catalytic role",
             "definition": "E1β (BCKDHB) and E1α (BCKDHA) together form the E1 heterotetrameric decarboxylase (α₂β₂) of the BCKDH complex. E1α contains the TPP (thiamine pyrophosphate) active site and performs oxidative decarboxylation. E1β is the structural partner: it does not perform any catalytic step but is required for E1α stability (prevents E1α proteolytic degradation) and for correct assembly of the two E1α and two E1β subunits into the functional heterotetramer. BCKDHB LOF → E1β absent or misfolded → E1α cannot be stabilised → heterotetramer cannot form → BCKDH non-functional → identical clinical presentation to BCKDHA deficiency."},
            {"term": "Why thiamine response is LOWER in BCKDHB (~5%) than BCKDHA (~10–20%)",
             "definition": "High-dose thiamine works by increasing intracellular TPP (thiamine pyrophosphate) concentration. TPP binds to the E1α (BCKDHA) subunit active site — NOT to E1β (BCKDHB). In BCKDHA deficiency: TPP saturation can stabilise a partially misfolded E1α active site → partial enzyme rescue (10–20% respond). In BCKDHB deficiency: even with saturated TPP-E1α, if E1β is absent or severely misfolded, the heterotetramer still cannot assemble → no catalytic benefit. Only BCKDHB variants that cause mild structural perturbation of E1β (e.g. p.Ala311Val) — where E1β is partially folded and capable of some E1α interaction — might show marginal indirect benefit when E1α is fully TPP-saturated and more conformationally stable. Expected response: <5% of BCKDHB patients. Thiamine trial remains MANDATORY (Level A) for all MSUD patients regardless of genotype — safe, and the rare responder benefits substantially."},
            {"term": "Alloisoleucine — pathognomonic MSUD marker",
             "definition": "Alloisoleucine (D-allo-isoleucine) is formed by a specific stereoisomerisation pathway from KMV (alpha-keto-beta-methylvalerate, the isoleucine-derived alpha-ketoacid). When BCKDH is blocked (by BCKDHB or any other MSUD gene), KMV accumulates and spontaneous transamination back to the D-alloisoleucine configuration occurs. Alloisoleucine is absent from normal plasma and appears ONLY in MSUD — it is pathognomonic for BCKDH deficiency of any type (1A/1B/2). A single plasma amino acids test showing alloisoleucine >5 µmol/L confirms MSUD. Alloisoleucine levels are identical across BCKDHA, BCKDHB, and DBT deficiency — gene panel required for gene-level diagnosis."},
            {"term": "BCAA-free synthetic formula — the metabolic cornerstone",
             "definition": "Patients with BCKDHB deficiency require a synthetic amino acid formula that provides ALL amino acids EXCEPT leucine, isoleucine, and valine. This formula serves as the primary protein source, providing essential AAs, vitamins, and minerals while preventing BCAA accumulation. Natural protein is provided only at the minimum level to support anabolism (minimise muscle catabolism). Leucine target: <200 µmol/L maintenance; <100 µmol/L ideal. Management is identical to BCKDHA."},
            {"term": "BCKDHB vs BCKDHA vs DBT — indistinguishable biochemically",
             "definition": "MSUD caused by BCKDHB (Type 1B), BCKDHA (Type 1A), or DBT (Type 2) deficiency is biochemically IDENTICAL: same BCAA elevation pattern, same alloisoleucine, same urine ketoacids, same enzyme assay showing <5% BCKDH complex activity. The only way to distinguish is DNA sequencing or gene panel. Enzyme typing (testing specific subunit activity in isolation) is technically difficult. Gene panel (BCKDHA + BCKDHB + DBT + DLD) is mandatory."},
            {"term": "DLD deficiency — why it is NOT BCKDHB deficiency",
             "definition": "DLD (shared E3 subunit) deficiency causes MSUD-3, which resembles BCKDHB biochemically (BCAA elevated, alloisoleucine positive) but additionally has: lactic acidosis (PDH block), 2-hydroxyglutarate elevated (αKGDH block), glycine elevated (GCS partial block), and DLD free enzyme activity <10%. In BCKDHB deficiency: PDH normal, αKGDH normal, GCS normal, DLD free enzyme NORMAL — only BCKDH complex activity is reduced."},
            {"term": "Acute crisis management — leucine removal",
             "definition": "Acute metabolic crisis (classic neonatal or decompensation): (1) STOP all natural protein; (2) HIGH-RATE IV glucose GIR 8–12 mg/kg/min to suppress muscle protein catabolism; (3) insulin 0.05–0.1 U/kg/h to drive leucine into anabolism; (4) IV BCAA-free amino acid solution to maintain anabolism and prevent protein breakdown; (5) if leucine >1500 µmol/L or rising despite above: haemodialysis (HD) or continuous CRRT — most effective for rapid leucine removal. Target: leucine <300 µmol/L within 24–48 hours. Identical protocol to BCKDHA crisis management."},
        ],
        "diagnostic_thresholds": {
            "plasma_leucine_normal": "<200 µmol/L",
            "plasma_leucine_classic_msud": ">1000 µmol/L (acute); typically 2000–4000 µmol/L in classic neonatal",
            "plasma_leucine_target_maintenance": "<200 µmol/L (chronic diet therapy)",
            "alloisoleucine_diagnostic": ">5 µmol/L in plasma — PATHOGNOMONIC for MSUD (any type including BCKDHB)",
            "alloisoleucine_normal": "ABSENT (undetectable by standard plasma amino acids)",
            "bckdh_complex_activity_classic": "<5% of normal (fibroblasts or lymphocytes)",
            "bckdh_complex_activity_intermediate": "5–25% of normal",
            "thiamine_trial_duration": "≥3 months at 10–20 mg/kg/day to assess response; expected responders <5% in BCKDHB vs ~10–20% in BCKDHA",
            "leucine_crisis_threshold": ">400 µmol/L → acute encephalopathy risk; >1000 µmol/L → cerebral oedema; >2000 µmol/L → life-threatening",
            "hd_indication": "Plasma leucine rising rapidly OR >1500 µmol/L despite IV glucose + insulin",
            "pdh_bckdhb_deficiency": "NORMAL (key negative to exclude DLD deficiency)",
            "akgdh_bckdhb_deficiency": "NORMAL (key negative to exclude DLD deficiency)",
            "dld_free_activity_bckdhb": "NORMAL (key negative — only BCKDH complex reduced in isolated BCKDHB deficiency)",
        },
        "differential_diagnosis": [
            {"disease": "BCKDHA deficiency (MSUD Type 1A)",
             "distinguishing_features": "BIOCHEMICALLY IDENTICAL to BCKDHB: same BCAA elevation, same alloisoleucine, same urine ketoacids, same BCKDH complex activity <5%. KEY DIFFERENCE: thiamine response rate HIGHER in BCKDHA (~10–20%) vs BCKDHB (~5%) because BCKDHA (E1α) contains the TPP-binding domain. ONLY gene sequencing (BCKDHA vs BCKDHB) definitively distinguishes. Gene panel mandatory."},
            {"disease": "DBT deficiency (MSUD Type 2)",
             "distinguishing_features": "BIOCHEMICALLY IDENTICAL to BCKDHB: same BCAA elevation, same alloisoleucine, same ketoaciduria, same BCKDH complex activity <5%. DBT deficiency (E2 transacylase subunit): isolated E2 enzyme activity test may show specific E2 deficiency, but gene panel (BCKDHA + BCKDHB + DBT) is standard."},
            {"disease": "DLD deficiency (MSUD Type 3 / DLD / E3 deficiency)",
             "distinguishing_features": "DLD deficiency: BCAA elevated (BCKDH block) + alloisoleucine POSITIVE (resembles BCKDHB). BUT additionally: lactic acidosis (PDH block), 2-hydroxyglutarate elevated (αKGDH block), glycine elevated (GCS partial block), DLD free enzyme activity <10%. In BCKDHB: PDH NORMAL, αKGDH NORMAL, GCS NORMAL, DLD free enzyme NORMAL."},
            {"disease": "Isovaleric acidaemia (IVA — IVDH deficiency)",
             "distinguishing_features": "IVA: isovaleryl-CoA dehydrogenase block (step 3 of Leu catabolism, downstream of BCKDH); isovalerylcarnitine (C5) elevated in NBS; isovaleric acid (sweaty feet odour) elevated in urine; alloisoleucine ABSENT; plasma Leu/Val/Ile NORMAL (BCKDH intact)."},
            {"disease": "Propionic acidaemia / Methylmalonic acidaemia",
             "distinguishing_features": "MMA/PA: involve Val catabolism downstream (propionyl-CoA pathway); elevated C3-carnitine, methylmalonic acid; alloisoleucine ABSENT; BCAA levels often normal (BCKDH intact upstream); thrombocytopaenia + neutropaenia more prominent."},
            {"disease": "Glutaric aciduria Type 2 (MADD/ETFDH)",
             "distinguishing_features": "MADD: multiple acyl-CoA dehydrogenase deficiency; many organic acids elevated (ethylmalonic, glutaric, isovaleric, etc.); riboflavin-responsive; alloisoleucine ABSENT; BCAA not selectively elevated."},
        ],
    }
