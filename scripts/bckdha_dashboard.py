#!/usr/bin/env python3
"""BCKDHA (Branched-Chain Keto Acid Dehydrogenase E1α Subunit / MSUD Type 1A) Epilepsy Dashboard.

BCKDHA encodes the E1α subunit of the branched-chain alpha-ketoacid dehydrogenase complex (BCKDH),
which catalyses the second, irreversible step in branched-chain amino acid (BCAA) catabolism:

  BCKDH complex structure: BCKDHA+BCKDHB (E1, thiamine-dependent decarboxylase, α₂β₂ heterotetramer)
                            + DBT (E2, dihydrolipoamide branched-chain transacylase)
                            + DLD (E3, dihydrolipoamide dehydrogenase — SHARED with PDH, αKGDH, GCS)
  Step 1 (reversible): BCAA + α-KG → branched-chain α-ketoacid + glutamate (aminotransferase; reversible)
  Step 2 (irreversible, BLOCKED): branched-chain α-ketoacid → branched-chain acyl-CoA
    Leucine → α-ketoisocaproate (KIC) → isovaleryl-CoA + CO₂  [BCKDH uses TPP; E1 decarboxylates]
    Isoleucine → α-keto-β-methylvalerate (KMV) → 2-methylbutyryl-CoA + CO₂
    Valine → α-ketoisovalerate (KIV) → isobutyryl-CoA + CO₂

  BCKDHA (E1α) role: E1α and E1β together form the E1 heterotetrameric decarboxylase (α₂β₂);
    E1α contains the active site with TPP (thiamine pyrophosphate) cofactor binding; catalyses
    oxidative decarboxylation of the branched-chain α-ketoacid via TPP-hydroxyalkyl-TPP intermediate;
    transfers the acyl group to DBT (E2) lipoamide arm.
  BCKDHA LOF: E1 heterotetrameric decarboxylase cannot form or is non-functional; α-ketoacids
    accumulate; BCAA accumulate (Leu >> Ile, Val); alloisoleucine appears (PATHOGNOMONIC).

PRIMARY ACCUMULATING TOXIC SPECIES:
  1. Leucine (most neurotoxic): plasma levels typically >2000 µmol/L in classic MSUD (normal <200)
  2. Alpha-ketoisocaproate (KIC, Leu α-keto acid): elevated; myelinopathic; directly toxic
  3. Alloisoleucine: pathognomonic — formed by stereoisomerisation of isoleucine-derived keto acids;
     alloisoleucine >5 µmol/L is DIAGNOSTIC; present ONLY in MSUD / BCKDH deficiency
  4. Alpha-ketoisovalerate (KIV, Val α-keto acid) and alpha-keto-β-methylvalerate (KMV, Ile keto acid)
  5. Branched-chain hydroxy acids: 2-hydroxyisocaproate, 2-hydroxyisovalerate (DNPH-positive)

LEUCINE NEUROTOXICITY MECHANISM:
  Leucine: enters brain via LAT1 (SLC7A5) large neutral amino acid transporter; competitively inhibits
  phenylalanine / tyrosine / tryptophan transport → depletes aromatic amino acid pool → inhibits
  myelination (tyrosine → DOPA pathway, phenylalanine → Phe pathway); mTOR hyperactivation;
  direct mitochondrial dysfunction; KIC and KICA (α-ketoisocaproate) myelinopathic at high concentrations.
  TREATMENT URGENCY: plasma leucine >400 µmol/L → acute encephalopathy; >1000 µmol/L → cerebral oedema;
  >2000 µmol/L → life-threatening brainstem herniation (cerebral oedema from osmotic effects + myelinopathy).

THIAMINE-RESPONSIVE MSUD (E1α subtype):
  ~10–20% of MSUD Type 1A (BCKDHA mutations) are thiamine-responsive (Level A):
  High-dose thiamine (10–20 mg/kg/day, max 200 mg/day) → TPP saturation → stabilises residual
  E1α mutant protein via chaperone-like effect; reduces plasma BCAA by 30–70% in responders;
  p.Glu422Lys and similar variants near TPP-binding domain most likely to respond.
  ALL BCKDHA/BCKDHB/DBT patients should receive a 3-month thiamine trial (Level A evidence).

AUTOSOMAL RECESSIVE:
  BCKDHA gene: 19q13.2; Autosomal Recessive (AR)
  ~445 aa precursor (mature ~418 aa after MTS cleavage)
  MSUD prevalence: ~1:185,000 globally; ~1:176 in Old Order Mennonite (founder effect)
  Mennonite founder mutation: p.Tyr393Asn — accounts for ~80–90% of Mennonite MSUD alleles
  OMIM gene: *608348; disease: #248600 (MAPLE SYRUP URINE DISEASE, TYPE IA)
  Note: MSUD types 1A (BCKDHA), 1B (BCKDHB), 2 (DBT) are biochemically identical at presentation;
  MSUD type 3 (DLD deficiency) has additional PDH/αKGDH/GCS blocks.

KEY VARIANTS:
  p.Tyr393Asn: Mennonite founder; TPP-binding domain; classic severe MSUD; most common worldwide allele
  p.Arg252Trp: common European; E1β docking interface (BCKDHA-BCKDHB assembly); classic MSUD
  p.Glu422Lys: TPP-binding domain; thiamine-responsive; milder phenotype; ~20–30% residual activity
  p.Ile281Phe: TPP-binding domain; classic severe; no thiamine response; neonatal crisis
  p.Phe364Cys: active site; severe; loss of oxidative decarboxylation capacity
  c.1312A>T (p.Ile438Phe): C-terminal structural domain; compound heterozygote, severe
  Large exonic deletions (exons 2–4 or 3–6): null alleles; neonatal presentation, severe

CLINICAL PRESENTATIONS:
  1. Classic MSUD (70%): neonatal presentation day 3–7; maple syrup odour; encephalopathy;
     hypotonia progressing to hypertonia; seizures; cerebral oedema; death if untreated within days
  2. Intermediate MSUD (10%): presentation 5 months–7 years; chronic protein intolerance;
     episodic neurological regression; milder plasma BCAA elevation
  3. Intermittent MSUD (10%): episodic crises during catabolic stress (illness, surgery, fasting);
     normal development between crises; baseline BCAA often borderline
  4. Thiamine-responsive MSUD (10%): responds to high-dose thiamine (10–20 mg/kg/day);
     most common in p.Glu422Lys and similar TPP-binding variants; partial enzyme activity preserved

TREATMENT PRIORITIES:
  1. BCAA-restricted diet with synthetic formula (Level A): acute AND chronic management;
     Leu restriction is the most critical component (Leu most toxic); use BCAA-free synthetic formula
     to provide all other amino acids; natural protein provided only to meet minimum biological needs;
     leucine intake target: <100 mg/kg/day (maintenance); adjust to target plasma Leu <200 µmol/L
  2. Thiamine (B1, Level A): 10–20 mg/kg/day (max 200 mg/day) for ALL patients as initial trial;
     ~10–20% of BCKDHA patients are thiamine-responsive; trial for ≥3 months before declaring non-responsive
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

random.seed(46)  # reproducible cohort (BCKDHA seed 46; DLST: 45; OGDH: 44; PDHX: 43)


PHENOTYPES = [
    ("Classic MSUD (Neonatal)", 0.70),
    ("Intermediate MSUD", 0.10),
    ("Intermittent MSUD", 0.10),
    ("Thiamine-Responsive MSUD", 0.10),
]

VARIANTS = [
    "p.Tyr393Asn",       # Mennonite founder
    "p.Arg252Trp",       # European
    "p.Glu422Lys",       # thiamine-responsive
    "p.Ile281Phe",       # severe
    "p.Phe364Cys",       # active site severe
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
            else:  # thiamine-responsive
                onset = round(random.uniform(1, 24), 1)

            # Plasma leucine (µmol/L) — normal <200
            if is_classic:
                leu = round(random.uniform(1800, 3800))
            elif is_intermediate:
                leu = round(random.uniform(400, 1200))
            elif is_intermittent:
                leu = round(random.uniform(200, 800))
            else:  # thiamine-responsive
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

            # Thiamine trial
            on_thiamine = True  # all should receive trial
            thiamine_responsive = is_thiamine or (is_intermediate and random.random() < 0.20)

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

            # Variant
            if "Mennonite" in phenotype_label or is_classic:
                variant = random.choice(VARIANTS[:2])  # Tyr393Asn or Arg252Trp most common
            elif is_thiamine:
                variant = "p.Glu422Lys"  # thiamine-responsive variant
            else:
                variant = random.choice(VARIANTS)

            pts.append({
                "id": f"BCKDHA-{pid:03d}",
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
        "dashboard": "BCKDHA Epilepsy (Maple Syrup Urine Disease Type 1A — BCKDH E1α Subunit Deficiency)",
        "gene": "BCKDHA",
        "protein": "Branched-Chain Alpha-Keto Acid Dehydrogenase E1α Subunit (BCKDH complex E1 catalytic subunit)",
        "locus": "19q13.2",
        "aa_length": 445,
        "cofactor": "TPP (thiamine pyrophosphate) — covalently binds E1α active site; thiamine-responsive variants respond to high-dose B1 supplementation (Level A trial mandatory for ALL MSUD patients)",
        "mechanism": (
            "BCKDHA encodes the E1α subunit of the BCKDH complex, which irreversibly decarboxylates "
            "branched-chain alpha-keto acids (from Leu, Ile, Val catabolism). "
            "The E1 heterotetrameric decarboxylase (BCKDHA₂ × BCKDHB₂) uses TPP to form a "
            "hydroxyalkyl-TPP intermediate from KIC/KMV/KIV, then transfers the acyl group to "
            "the DBT (E2) lipoamide arm → branched-chain acyl-CoA is formed. "
            "BCKDHA LOF → E1 heterotetrameric decarboxylase non-functional → BCAA (Leu/Ile/Val) "
            "and branched-chain α-ketoacids accumulate → alloisoleucine appears (PATHOGNOMONIC). "
            "Leucine (most neurotoxic): inhibits LAT1 aromatic AA transport → myelination failure + mTOR hyperactivation."
        ),
        "omim_gene": "*608348",
        "omim_disease": "#248600 (MAPLE SYRUP URINE DISEASE, TYPE IA)",
        "inheritance": "Autosomal Recessive (AR, 19q13.2) — both sexes equally affected; no sex bias; Old Order Mennonite enrichment (1:176)",
        "pathognomonic_marker": (
            "Alloisoleucine >5 µmol/L (plasma, simultaneously measured with Leu/Ile/Val) — "
            "PATHOGNOMONIC for MSUD (any BCKDH deficiency type 1A/1B/2). "
            "Alloisoleucine is formed by spontaneous stereoisomerisation of isoleucine-derived alpha-keto "
            "acid (KMV) intermediates; absent from normal plasma; its presence confirms BCKDH block. "
            "Confirmed by: NBS tandem MS (elevated C5:1 / leucine/isoleucine peaks), plasma amino acids "
            "(Leu/Ile/Val elevation + alloisoleucine), urine organic acids (KIC, KMV, KIV by GC-MS), DNPH test."
        ),
        "key_distinguishing_feature": (
            "MSUD Type 1A (BCKDHA) vs DLD deficiency: BCKDHA — isolated BCKDH block → BCAA↑ + alloisoleucine "
            "PATHOGNOMONIC + PDH NORMAL + αKGDH NORMAL + GCS NORMAL + DLD free enzyme NORMAL. "
            "DLD deficiency: ALL FOUR complexes blocked (PDH + αKGDH + BCKDH + GCS) → BCAA↑ + lactate↑↑ + "
            "2-HG↑ + glycine↑ + DLD free enzyme <10%. "
            "BCKDHA vs BCKDHB vs DBT: biochemically IDENTICAL — gene panel mandatory to distinguish."
        ),
        "structural_brain_hallmark": (
            "White matter diffusion restriction — myelinopathy: symmetric T2 FLAIR hyperintensity and "
            "DWI restriction in posterior limbs of internal capsules, cerebral peduncles, posterior "
            "white matter (classic neonatal — 'MSUD myelinopathy pattern'; 80% in classic BCKDHA); "
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
            {"component": "E1α — BCKDHA ← THIS GENE",
             "function": "Catalytic subunit of E1 decarboxylase (α₂β₂ heterotetramer); TPP active site; decarboxylates branched-chain α-ketoacids (KIC/KMV/KIV) via TPP-hydroxyalkyl-TPP intermediate; BCKDHA LOF → E1 non-functional → BCAA accumulate."},
            {"component": "E1β — BCKDHB",
             "function": "Structural subunit of E1 decarboxylase (α₂β₂ heterotetramer); required for E1α stability and assembly; BCKDHB LOF causes MSUD Type 1B (biochemically identical to Type 1A)."},
            {"component": "E2 — DBT (dihydrolipoamide branched-chain transacylase)",
             "function": "24-mer cubic core with lipoyl domains; receives acyl group from E1 via lipoamide arm; transfers to CoA → branched-chain acyl-CoA; DBT LOF causes MSUD Type 2."},
            {"component": "E3 — DLD (dihydrolipoamide dehydrogenase, FAD homodimer)",
             "function": "Regenerates DBT lipoamide arm; SHARED with PDH, αKGDH, GCS complexes; DLD is INTACT in BCKDHA deficiency."},
            {"component": "BCKDK (kinase) / PPM2C (phosphatase)",
             "function": "BCKDK phosphorylates E1α (Ser337, Ser341) → inactivates BCKDH; BT2 inhibits BCKDK → activates BCKDH (therapeutic target in research). Not relevant to loss-of-function MSUD."},
        ],
        "bcaa_catabolism_pathway": {
            "step_1_reversible": "BCAA + α-KG → branched-chain α-ketoacid + Glu (aminotransferase BCAT1/BCAT2; reversible; explains why diet restriction can normalise plasma levels)",
            "step_2_irreversible_blocked": "Branched-chain α-ketoacid → acyl-CoA + CO₂ (BCKDH complex; BLOCKED in BCKDHA deficiency; irreversible step makes accumulation rapid)",
            "downstream_leu": "KIC (Leu) → isovaleryl-CoA → 3-methylcrotonyl-CoA → HMG-CoA → acetyl-CoA + acetoacetate",
            "downstream_val": "KIV (Val) → isobutyryl-CoA → propionyl-CoA → methylmalonyl-CoA → succinyl-CoA (joins TCA)",
            "downstream_ile": "KMV (Ile) → 2-methylbutyryl-CoA → propionyl-CoA + acetyl-CoA",
            "alloisoleucine_formation": "KMV (Ile keto acid) ⇌ 2-hydroxy-3-methylvalerate intermediates → spontaneous transamination back → alloisoleucine (D-allo-isoleucine); PATHOGNOMONIC",
        },
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
             "mechanism": "Triple CI: (1) carnitine depletion → worsens secondary carnitine deficiency; (2) direct BCKDH complex inhibition → worsens BCAA accumulation; (3) mitochondrial hepatotoxicity. Equivalent to POLG1 contraindication. NEVER use in MSUD."},
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
    c = _COHORT
    sample = c[:10]

    return {
        "biomarkers": [
            {"name": "Plasma Leucine", "normal": "<200 µmol/L",
             "bckdha_range": ">2000 µmol/L (classic acute); 400–1200 µmol/L (intermediate); 200–800 µmol/L (intermittent/thiamine-resp)",
             "significance": "PRIMARY toxic species; most neurotoxic BCAA; LAT1 transporter competition → myelination failure; mTOR hyperactivation; leucine >1000 µmol/L → cerebral oedema risk; target <200 µmol/L maintenance"},
            {"name": "Alloisoleucine (D-allo-Ile)", "normal": "ABSENT (undetectable in normal plasma)",
             "bckdha_range": ">5 µmol/L (any MSUD subtype) — PATHOGNOMONIC",
             "significance": "PATHOGNOMONIC MARKER for MSUD: formed by spontaneous stereoisomerisation of KMV (Ile keto acid) intermediates; absent from all non-MSUD conditions; present in BCKDHA/BCKDHB/DBT deficiency equally; persists even during diet control if not fully normalised"},
            {"name": "Plasma Isoleucine", "normal": "<100 µmol/L",
             "bckdha_range": "300–1800 µmol/L (classic); 150–600 µmol/L (intermediate)",
             "significance": "Second most elevated BCAA; source of alloisoleucine (via KMV); neurotoxic at high levels; less toxic than leucine but monitored in parallel"},
            {"name": "Plasma Valine", "normal": "<300 µmol/L",
             "bckdha_range": "500–2000 µmol/L (classic); 200–900 µmol/L (intermediate)",
             "significance": "Most elevated in plasma (by mass) but least neurotoxic; Val catabolism enters propionyl-CoA pathway → succinyl-CoA (joins TCA)"},
            {"name": "Urine alpha-ketoacids (GC-MS)", "normal": "Trace or absent",
             "bckdha_range": "KIC 800–4000 µmol/mmol Cr; KIV, KMV elevated proportionally",
             "significance": "Urine organic acids (GC-MS): alpha-ketoisocaproate (KIC, Leu), alpha-ketoisovalerate (KIV, Val), alpha-keto-β-methylvalerate (KMV, Ile) elevated; KIC is directly myelinopathic; DNPH test positive (ketoacids)"},
            {"name": "DNPH (dinitrophenylhydrazine) urine test", "normal": "Negative",
             "bckdha_range": "POSITIVE — yellow precipitate (alpha-ketoacids react with DNPH)",
             "significance": "Bedside screening test for branched-chain ketoaciduria; positive in classic/intermediate MSUD; may be negative in intermittent forms between crises; not specific (also positive in PKU, tyrosinaemia) — confirm with GC-MS"},
            {"name": "Plasma Lactate", "normal": "<2.2 mmol/L",
             "bckdha_range": "3.5–14 mmol/L (classic acute); 1.5–5 mmol/L (stable)",
             "significance": "Secondary to mitochondrial dysfunction from leucine toxicity and KIC accumulation; NOT as primary a feature as in PDH/αKGDH deficiency; lactic acidosis worst in acute crisis"},
            {"name": "Urine BCAA and branched-chain hydroxy acids", "normal": "Absent/trace",
             "bckdha_range": "2-hydroxyisocaproate, 2-hydroxyisovalerate elevated",
             "significance": "Branched-chain hydroxy acids from reduction of keto acids; further confirms BCKDH block; detectable by GC-MS urine organic acids"},
            {"name": "BCKDH enzyme activity (fibroblasts/lymphocytes)", "normal": ">60% normal",
             "bckdha_range": "<5% of normal (classic); 5–25% (intermediate/thiamine-resp)",
             "significance": "Definitive enzymatic diagnosis; measured as 14CO₂ release from [1-14C]-leucine; post-thiamine trial activity increase confirms thiamine-responsiveness"},
            {"name": "PDH enzyme activity", "normal": ">60% normal",
             "bckdha_range": "NORMAL — PDH intact in isolated BCKDHA deficiency",
             "significance": "KEY NEGATIVE: PDH normal confirms only BCKDH is deficient; DLD/E3 is shared but DLD protein is intact; PDH reduced would suggest DLD deficiency"},
            {"name": "αKGDH complex activity", "normal": ">60% normal",
             "bckdha_range": "NORMAL — αKGDH intact in isolated BCKDHA deficiency",
             "significance": "KEY NEGATIVE: αKGDH normal confirms only BCKDH is deficient; 2-oxoglutarate NORMAL (no OGDH/DLST/DLD block)"},
            {"name": "DLD free enzyme activity", "normal": ">60% normal",
             "bckdha_range": "NORMAL — DLD protein intact in BCKDHA deficiency",
             "significance": "KEY NEGATIVE: DLD free activity normal excludes DLD deficiency (where all four complexes are blocked); in DLD deficiency, DLD free activity <10%"},
        ],
        "key_variants": [
            {"variant": "p.Tyr393Asn (c.1178A>C)", "effect": "TPP-binding domain; disrupts thiamine pyrophosphate coordination in E1α active site; complete loss of decarboxylase activity in homozygotes; most common BCKDHA variant worldwide", "phenotype": "Classic MSUD — Mennonite founder (80–90% of Mennonite alleles)"},
            {"variant": "p.Arg252Trp", "effect": "E1β-docking interface of E1α; disrupts E1α-E1β heterotetrameric assembly; E1 complex cannot form; null-equivalent", "phenotype": "Classic MSUD — European"},
            {"variant": "p.Glu422Lys", "effect": "TPP-binding domain; reduces TPP affinity; high-dose thiamine increases intracellular TPP → stabilises residual E1α → partial activity restoration", "phenotype": "Thiamine-Responsive MSUD (intermediate severity)"},
            {"variant": "p.Ile281Phe", "effect": "TPP-binding domain (near active site); severe loss of function; no thiamine response (affinity too impaired for mass-action rescue)", "phenotype": "Classic MSUD"},
            {"variant": "p.Phe364Cys", "effect": "Catalytic active site; abolishes oxidative decarboxylase activity; no TPP hydroxyalkyl-TPP intermediate formed; complete null", "phenotype": "Classic MSUD — Severe"},
            {"variant": "c.1312A>T (p.Ile438Phe)", "effect": "C-terminal structural domain; destabilises E1α protein; partial loss; often compound heterozygote", "phenotype": "Classic to Intermediate MSUD"},
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
             "note": "First-line chronic management; Leu restriction critical (target Leu <200 µmol/L); BCAA-free formula provides other amino acids, vitamins, minerals; natural protein only to meet minimum biological need; must be lifelong"},
            {"drug": "Thiamine B1 — high dose (Level A)", "level": "A", "response_pct": 15,
             "color": "#0277bd",
             "note": "10–20 mg/kg/day (max 200 mg/day); ALL BCKDHA patients receive trial ≥3 months; ~10–20% responders (p.Glu422Lys and similar TPP-binding domain variants); responders show 30–70% BCAA reduction; lifelong in responders"},
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
             "mechanism": "Triple CI: carnitine depletion + BCKDH direct inhibition + mitochondrial hepatotoxicity. NEVER use in MSUD."},
            {"drug": "Fasting", "risk": "EXTREME HAZARD",
             "mechanism": "Muscle catabolism → BCAA surge → leucine crisis within hours. IV glucose mandatory."},
            {"drug": "Unrestricted natural protein", "risk": "EXTREME HAZARD",
             "mechanism": "All natural proteins contain BCAA; any unrestricted meal → leucine surge."},
            {"drug": "Surgery without metabolic protocol", "risk": "HAZARD",
             "mechanism": "Catabolic stress + NPO → BCAA crisis. Requires perioperative IV glucose + BCAA monitoring."},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "Gene": "BCKDHA (Branched-Chain Keto Acid Dehydrogenase E1α Subunit)",
            "Also known as": "BCKDE1A; MSUD1A",
            "Subunit": "E1α catalytic subunit of BCKDH complex (with BCKDHB forming E1α₂β₂ heterotetramer)",
            "Chromosome": "19q13.2",
            "Protein length": "~445 aa precursor (~418 aa mature after MTS cleavage)",
            "Cofactor": "TPP (thiamine pyrophosphate) — covalently bound to E1α active site",
            "Function": "Oxidative decarboxylation of branched-chain alpha-ketoacids (KIC from Leu, KMV from Ile, KIV from Val) via TPP-hydroxyalkyl-TPP mechanism; BCKDHA LOF → BCAA accumulate",
            "Complex partners": "BCKDHB (E1β structural subunit) + DBT (E2, lipoamide transacylase) + DLD (E3, FAD homodimer, SHARED with PDH/αKGDH/GCS)",
            "Inheritance": "Autosomal Recessive (AR) — biallelic LOF",
            "OMIM gene": "*608348",
            "OMIM disease": "#248600 (MAPLE SYRUP URINE DISEASE, TYPE IA)",
            "Prevalence": "~1:185,000 globally; ~1:176 Old Order Mennonite (p.Tyr393Asn founder)",
            "Biochemical block": "Step 2 of BCAA catabolism — irreversible oxidative decarboxylation BLOCKED",
            "Pathognomonic marker": "Alloisoleucine >5 µmol/L (plasma) — present ONLY in MSUD",
            "Primary toxic species": "Leucine (most neurotoxic); KIC (alpha-ketoisocaproate, myelinopathic)",
            "First-line treatment": "BCAA-restricted diet + BCAA-free synthetic formula (Level A) + thiamine trial (Level A)",
            "ABSOLUTE CI": "Valproate (VPA) — triple contraindication (carnitine depletion + BCKDH inhibition + hepatotoxicity)",
        },
        "key_concepts": [
            {"term": "Alloisoleucine — pathognomonic MSUD marker",
             "definition": "Alloisoleucine (D-allo-isoleucine, also called L-alloisoleucine or 4-methyl isoleucine diastereomer) is formed by a specific stereoisomerisation pathway from KMV (alpha-keto-beta-methylvalerate, the isoleucine-derived alpha-ketoacid). When BCKDH is blocked, KMV accumulates and spontaneous transamination back to the D-alloisoleucine configuration occurs. Alloisoleucine is absent from normal plasma and appears ONLY in MSUD — it is pathognomonic for BCKDH deficiency of any type (1A/1B/2). A single plasma amino acids test showing alloisoleucine >5 µmol/L confirms MSUD."},
            {"term": "BCAA-free synthetic formula — the metabolic cornerstone",
             "definition": "Patients with BCKDHA deficiency require a synthetic amino acid formula that provides ALL amino acids EXCEPT leucine, isoleucine, and valine. This formula serves as the primary protein source, providing essential AAs, vitamins, and minerals while preventing BCAA accumulation. Natural protein is provided only at the minimum level to support anabolism (minimise muscle catabolism). Leucine target: <200 µmol/L maintenance; <100 µmol/L ideal."},
            {"term": "Thiamine responsiveness in BCKDHA deficiency",
             "definition": "~10–20% of BCKDHA patients carry variants that reduce TPP affinity without abolishing binding (e.g., p.Glu422Lys). High-dose thiamine (10–20 mg/kg/day, max 200 mg/day) increases intracellular TPP concentration → saturates residual E1α → partial restoration of BCKDH activity → 30–70% reduction in plasma BCAA. ALL MSUD patients (Type 1A/1B/2) must receive a 3-month high-dose thiamine trial. Non-responders do not worsen with thiamine. Level A evidence."},
            {"term": "LAT1 competitive inhibition — leucine neurotoxicity mechanism",
             "definition": "Leucine (and other BCAA) enter the brain via the LAT1 transporter (SLC7A5 / 4F2-LAT1 heterodimer). LAT1 is a sodium-independent large neutral amino acid transporter. At high plasma leucine levels, leucine competitively displaces phenylalanine, tyrosine, and tryptophan from LAT1 → depletes brain pool of aromatic amino acids → impairs neurotransmitter synthesis (catecholamines, serotonin) and myelin protein synthesis → myelinopathy. Additionally, high leucine → mTORC1 hyperactivation → disrupts autophagy and proteostasis."},
            {"term": "BCKDHA vs BCKDHB vs DBT — indistinguishable biochemically",
             "definition": "MSUD caused by BCKDHA (Type 1A), BCKDHB (Type 1B), or DBT (Type 2) deficiency is biochemically IDENTICAL: same BCAA elevation pattern, same alloisoleucine, same urine ketoacids, same enzyme assay showing <5% BCKDH complex activity. The only way to distinguish is DNA sequencing or gene panel. Enzyme typing (testing specific subunit activity in isolation) is technically difficult. Gene panel (BCKDHA + BCKDHB + DBT + DLD) is mandatory."},
            {"term": "DLD deficiency — why it is NOT BCKDHA deficiency",
             "definition": "DLD (shared E3 subunit) deficiency causes MSUD-3, which resembles BCKDHA biochemically (BCAA elevated, alloisoleucine positive) but additionally has: lactic acidosis (PDH block), 2-hydroxyglutarate elevated (αKGDH block), glycine elevated (GCS partial block), and DLD free enzyme activity <10%. In BCKDHA deficiency: PDH normal, αKGDH normal, GCS normal, DLD free enzyme NORMAL — only BCKDH complex activity is reduced."},
            {"term": "Acute crisis management — leucine removal",
             "definition": "Acute metabolic crisis (classic neonatal or decompensation): (1) STOP all natural protein; (2) HIGH-RATE IV glucose GIR 8–12 mg/kg/min to suppress muscle protein catabolism; (3) insulin 0.05–0.1 U/kg/h to drive leucine into anabolism; (4) IV BCAA-free amino acid solution to maintain anabolism and prevent protein breakdown; (5) if leucine >1500 µmol/L or rising despite above: haemodialysis (HD) or continuous CRRT — most effective for rapid leucine removal. Target: leucine <300 µmol/L within 24–48 hours."},
        ],
        "diagnostic_thresholds": {
            "plasma_leucine_normal": "<200 µmol/L",
            "plasma_leucine_classic_msud": ">1000 µmol/L (acute); typically 2000–4000 µmol/L in classic neonatal",
            "plasma_leucine_target_maintenance": "<200 µmol/L (chronic diet therapy)",
            "alloisoleucine_diagnostic": ">5 µmol/L in plasma — PATHOGNOMONIC for MSUD (any type)",
            "alloisoleucine_normal": "ABSENT (undetectable by standard plasma amino acids)",
            "bckdh_complex_activity_classic": "<5% of normal (fibroblasts or lymphocytes)",
            "bckdh_complex_activity_intermediate": "5–25% of normal",
            "thiamine_trial_duration": "≥3 months at 10–20 mg/kg/day to assess response",
            "leucine_crisis_threshold": ">400 µmol/L → acute encephalopathy risk; >1000 µmol/L → cerebral oedema; >2000 µmol/L → life-threatening",
            "hd_indication": "Plasma leucine rising rapidly OR >1500 µmol/L despite IV glucose + insulin",
            "pdh_bckdha_deficiency": "NORMAL (key negative to exclude DLD deficiency)",
            "akgdh_bckdha_deficiency": "NORMAL (key negative to exclude DLD deficiency)",
            "dld_free_activity_bckdha": "NORMAL (key negative — only BCKDH complex reduced in isolated BCKDHA deficiency)",
        },
        "differential_diagnosis": [
            {"disease": "BCKDHB deficiency (MSUD Type 1B)",
             "distinguishing_features": "BIOCHEMICALLY IDENTICAL to BCKDHA: same BCAA elevation, same alloisoleucine, same urine ketoacids, same BCKDH complex activity <5%. ONLY gene sequencing (BCKDHB vs BCKDHA) distinguishes. Gene panel mandatory."},
            {"disease": "DBT deficiency (MSUD Type 2)",
             "distinguishing_features": "BIOCHEMICALLY IDENTICAL to BCKDHA: same BCAA elevation, same alloisoleucine, same ketoaciduria, same BCKDH complex activity <5%. DBT deficiency (E2 subunit): isolated E2 enzyme activity test may show specific E2 deficiency, but gene panel (BCKDHA + BCKDHB + DBT) is standard."},
            {"disease": "DLD deficiency (MSUD Type 3 / DLD / E3 deficiency)",
             "distinguishing_features": "DLD deficiency: BCAA elevated (BCKDH block) + alloisoleucine POSITIVE (resembles BCKDHA). BUT additionally: lactic acidosis (PDH block), 2-hydroxyglutarate elevated (αKGDH block), glycine elevated (GCS partial block), DLD free enzyme activity <10%. In BCKDHA: PDH NORMAL, αKGDH NORMAL, GCS NORMAL, DLD free enzyme NORMAL."},
            {"disease": "Isovaleric acidaemia (IVA — IVDH deficiency)",
             "distinguishing_features": "IVA: isovaleryl-CoA dehydrogenase block (step 3 of Leu catabolism, downstream of BCKDH); isovalerylcarnitine (C5) elevated in NBS; isovaleric acid (sweaty feet odour) elevated in urine; alloisoleucine ABSENT; plasma Leu/Val/Ile NORMAL (BCKDH intact)."},
            {"disease": "Propionic acidaemia / Methylmalonic acidaemia",
             "distinguishing_features": "MMA/PA: involve Val catabolism downstream (propionyl-CoA pathway); elevated C3-carnitine, methylmalonic acid; alloisoleucine ABSENT; BCAA levels often normal (BCKDH intact upstream); thrombocytopaenia + neutropaenia more prominent."},
            {"disease": "Glutaric aciduria Type 2 (MADD/ETFDH)",
             "distinguishing_features": "MADD: multiple acyl-CoA dehydrogenase deficiency; many organic acids elevated (ethylmalonic, glutaric, isovaleric, etc.); riboflavin-responsive; alloisoleucine ABSENT; BCAA not selectively elevated."},
        ],
    }
