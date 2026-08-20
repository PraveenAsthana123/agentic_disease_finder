#!/usr/bin/env python3
"""OGDH (2-Oxoglutarate Dehydrogenase / Alpha-Ketoglutarate Dehydrogenase E1 Subunit) Epilepsy Dashboard.

OGDH encodes the E1 (oxoglutarate decarboxylase) subunit of the 2-oxoglutarate dehydrogenase
complex (αKGDH complex / 2-OGDHC), which catalyses the fourth step of the TCA cycle:
  αKGDH complex structure: OGDH (E1, decarboxylase) + DLST (E2, dihydrolipoamide succinyltransferase)
                            + DLD (E3, dihydrolipoamide dehydrogenase — SHARED with PDH, BCKDH, GCS)
  Reaction: 2-oxoglutarate (α-KG) + CoA + NAD⁺ → succinyl-CoA + CO₂ + NADH (irreversible)
  TCA cycle position: step 4 (isocitrate → α-KG by ICDH, then α-KG → succinyl-CoA by αKGDH)
  OGDH (E1) role: binds α-KG and TPP (thiamine pyrophosphate), decarboxylates α-KG;
                  product: α-succinyl-TPP; transfers succinyl group to lipoamide arm of DLST (E2)
  DLST (E2) transfers succinyl to CoA → succinyl-CoA
  DLD (E3) regenerates the dihydrolipoamide arm (oxidises NADH) — SHARED enzyme

OGDH LOF — BIOCHEMICAL PHENOTYPE:
  OGDH deficient → α-KG (2-oxoglutarate) cannot be decarboxylated →
  α-KG accumulates in plasma, urine, CSF → 2-oxoglutaric aciduria (PATHOGNOMONIC).
  Secondary: succinyl-CoA cannot be made → downstream TCA impaired →
  NADH/NAD⁺ ratio rises (reduced TCA-driven NAD⁺ regeneration) →
  LDH: pyruvate + NADH ⇌ lactate + NAD⁺ shifts toward lactate →
  Lactic acidosis with L:P ratio MILDLY ELEVATED (15–30; unlike PDH block where L:P 10–20 NORMAL;
  unlike Complex I where L:P >25 dramatically elevated — OGDH occupies intermediate position).
  PDH is INTACT → pyruvate → acetyl-CoA proceeds normally → pyruvate may be normal/mildly elevated.

CRITICAL DISTINCTIONS FROM DLD DEFICIENCY (shared E3):
  DLD deficiency blocks ALL four complexes (αKGDH + PDH + BCKDH + GCS):
    → BCAA elevated (BCKDH block) · 2-KG elevated · glycine elevated (GCS block) · L:P variable
  OGDH deficiency blocks ONLY αKGDH (E1 subunit):
    → 2-oxoglutarate (α-KG) MARKEDLY elevated — primary biomarker
    → BCAA NORMAL (BCKDH intact — DLD/E3 can still anchor to BCKDH)
    → Glycine NORMAL (GCS intact — DLD/E3 can still anchor to GCS)
    → DLD/E3 free enzyme activity NORMAL (DLD protein is intact — only αKGDH-anchored E3 is lost)
    → DLST/E2 enzyme activity normal (DLST is intact; only E1/OGDH is deficient)
    → αKGDH complex activity: <10% normal (E1 step blocked)

DISTINGUISHING FROM PDH COMPLEX DEFICIENCY (PDHA1/PDHB/DLAT/PDHX):
  PDH deficiency: L:P ratio NORMAL (10–20); BOTH lactate AND pyruvate accumulate;
    2-oxoglutarate NORMAL; BCAA normal; PDH enzyme activity severely reduced.
  OGDH deficiency: L:P ratio mildly elevated (15–30); pyruvate may be normal;
    2-oxoglutarate MARKEDLY elevated (pathognomonic); PDH enzyme activity NORMAL;
    αKGDH complex activity severely reduced.

AUTOSOMAL RECESSIVE:
  OGDH gene: 7p13; Autosomal Recessive (AR)
  Both males and females equally affected (no sex bias)
  ~20–40 OGDH cases confirmed worldwide 2026 (ultra-rare; rare even among mitochondrial diseases)
  No common founder allele; most variants private/family; pan-ethnic distribution
  TPP-binding variants may show partial thiamine responsiveness

KEY VARIANTS:
  p.Arg263Cys (c.787C>T): TPP-binding domain (ThDP-binding pocket); reduces cofactor affinity;
                           partial residual activity; thiamine-responsive subset possible
  p.Gly354Asp: active site region; near decarboxylation catalytic residues; severe phenotype
  p.Arg447His: TPP-binding/E2-docking interface; disrupts DLST (E2) substrate positioning
  p.Tyr395Cys: near lipoyl arm-docking site; affects succinyl transfer to DLST
  c.1237+1G>A: splice site; null/null in most alleles; severe neonatal
  Large exonic deletions: ~10% of OGDH alleles; 7p13 — less structural instability than PDHX

TREATMENT:
  1. Thiamine / B1 (Level A): TPP is the essential cofactor for OGDH E1 (same as PDHA1 for PDH E1).
     Mandatory trial in ALL patients. Response rate ~40–55% (partial to complete) for TPP-binding
     domain variants — higher than PDH complex subunit deficiencies (PDHA1 ~55%, PDHB ~35%).
     Dose: 100–600 mg/day; measure α-KG response after 2–4 weeks.
  2. Riboflavin / B2 (Level B): FAD cofactor for DLD (shared E3); may improve residual E3 function;
     combined with thiamine in most protocols; safe to co-administer.
  3. Lipoic acid (Level C): lipoamide cofactor for DLST (E2); experimental; OGDH-specific rationale
     (unlike PDH deficiency where lipoic acid rationale is limited to DLAT).
  4. Succinate supplementation (Level C): provides TCA substrate downstream of OGDH block;
     succinic acid → fumarate → malate → OAA (bypasses blocked succinyl-CoA step);
     experimental; some anecdotal benefit in mild phenotypes.
  5. LEV (Level B): first-line AED; no αKGDH pathway interaction; safe.
  6. Ketogenic diet (Level C / experimental): provides acetyl-CoA → TCA, but TCA block is at
     step 4 (α-KG → succinyl-CoA), NOT at TCA entry; KD does NOT bypass OGDH block as
     cleanly as it bypasses PDH block — KD may reduce carbohydrate-driven α-KG flux but
     does not provide succinyl-CoA directly; limited evidence; consider if other options fail.

HIGH-RISK DRUGS / EXPOSURES:
  VPA (Valproate): CAUTION — mitochondrial hepatotoxicity risk in any mitochondrial disorder;
    not absolute CI as in PDH deficiency (no KD dependency) but avoid if LEV is effective.
  High protein diet: HAZARD — amino acid catabolism (glutamate → α-KG via transaminases) increases
    α-KG load → overwhelms residual OGDH activity → worsens 2-oxoglutaric aciduria and lactic acidosis.
  Fasting: HAZARD — mobilizes branched-chain amino acids and glutamine → α-KG surge;
    succinic acid, glucose-electrolyte oral rehydration preferred during illness.
  Exercise: CAUTION — increases mitochondrial TCA flux demand; exercise-induced metabolic crises.

GENETICS SUMMARY:
  Gene: OGDH · 7p13 · ~1023 aa (mature protein; includes mitochondrial targeting sequence, MTS)
  Autosomal Recessive (AR) · OMIM gene *604290 · 2-Oxoglutaric aciduria / αKGDH complex deficiency
  TPP (thiamine pyrophosphate)-dependent decarboxylase; E1 subunit of αKGDH complex
  No common founder; all variants pan-ethnic private-family
"""

import random

random.seed(44)  # reproducible cohort (different seed from PDHX=43)


def _patients():
    """Synthetic 40-patient OGDH-deficiency cohort."""
    phenotypes = [
        ("Leigh Syndrome (Infantile)", 16),
        ("Severe Neonatal (Neonatal Lactic Acidosis)", 12),
        ("Childhood Episodic Encephalopathy", 8),
        ("Mild/Juvenile (Partial Deficiency)", 4),
    ]
    pts = []
    pid = 1
    for pheno, n in phenotypes:
        for _ in range(n):
            is_leigh = "Leigh" in pheno
            is_neonatal = "Neonatal" in pheno
            is_episodic = "Episodic" in pheno
            is_mild = "Mild" in pheno

            onset = (
                random.randint(1, 4)
                if is_neonatal
                else random.randint(3, 18)
                if is_leigh
                else random.randint(12, 48)
                if is_episodic
                else random.randint(24, 96)
            )

            # OGDH biochemistry: L:P mildly elevated (unlike PDH which is 10-20 normal)
            pyruvate = round(random.uniform(0.15, 0.45), 2)   # may be normal/mildly elevated
            lp_ratio = round(random.uniform(15, 30), 1)        # mildly elevated (not as high as Complex I)
            lactate = round(pyruvate * lp_ratio, 2)
            oxoglutarate_urine = round(random.uniform(200, 2200), 0)  # µmol/mmol creatinine (markedly elevated)
            alanine = random.randint(200, 550)  # mildly elevated (pyruvate mildly elevated)
            glutamate = random.randint(80, 250)  # elevated (α-KG → glutamate transamination)

            pts.append({
                "id": f"OG{pid:03d}",
                "sex": random.choice(["M", "F"]),
                "phenotype": pheno,
                "onset_age_months": onset,
                "plasma_lactate_mmol": lactate,
                "plasma_pyruvate_mmol": pyruvate,
                "lp_ratio": lp_ratio,
                "urine_2_oxoglutarate": oxoglutarate_urine,
                "plasma_alanine_umol": alanine,
                "plasma_glutamate_umol": glutamate,
                "leigh_lesions": random.random() < (0.92 if is_leigh else 0.35 if is_neonatal else 0.15),
                "basal_ganglia_lesions": random.random() < (0.90 if is_leigh else 0.35 if is_neonatal else 0.1),
                "cortical_atrophy": random.random() < (0.50 if is_leigh or is_neonatal else 0.3),
                "on_thiamine": random.random() < 0.92,
                "thiamine_responsive": random.random() < (0.52 if is_episodic or is_mild else 0.35),
                "on_riboflavin": random.random() < 0.78,
                "on_kd": random.random() < (0.30 if is_leigh else 0.20 if is_neonatal else 0.15),
                "vpa_avoided": random.random() < 0.90,
                "dre": random.random() < (0.70 if is_leigh else 0.60 if is_neonatal else 0.28),
            })
            pid += 1
    return pts


_COHORT = _patients()


def get_overview():
    c = _COHORT
    n = len(c)
    male_n = sum(1 for p in c if p["sex"] == "M")
    avg_lac = round(sum(p["plasma_lactate_mmol"] for p in c) / n, 2)
    avg_pyr = round(sum(p["plasma_pyruvate_mmol"] for p in c) / n, 2)
    avg_lp  = round(sum(p["lp_ratio"] for p in c) / n, 1)
    elevated_lp = sum(1 for p in c if p["lp_ratio"] > 20)
    avg_okg = round(sum(p["urine_2_oxoglutarate"] for p in c) / n)
    leigh_n = sum(1 for p in c if p["leigh_lesions"])
    bg_n    = sum(1 for p in c if p["basal_ganglia_lesions"])
    thiamine_n = sum(1 for p in c if p["on_thiamine"])
    responsive_n = sum(1 for p in c if p["thiamine_responsive"])
    kd_n   = sum(1 for p in c if p["on_kd"])
    dre_n  = sum(1 for p in c if p["dre"])
    vpa_ok = sum(1 for p in c if p["vpa_avoided"])

    return {
        "gene": "OGDH",
        "protein": "2-Oxoglutarate Dehydrogenase (αKGDH Complex E1 Subunit / Oxoglutarate Decarboxylase)",
        "locus": "7p13",
        "aa_length": 1023,
        "cofactor": "Thiamine pyrophosphate (TPP / ThDP) — essential; OGDH E1 decarboxylates α-KG using TPP; lipoyl-DLST (E2) receives succinyl group",
        "mechanism": (
            "OGDH encodes the E1 (oxoglutarate decarboxylase) subunit of the 2-oxoglutarate dehydrogenase "
            "complex (αKGDH), which catalyses TCA cycle step 4: α-KG + CoA + NAD⁺ → succinyl-CoA + CO₂ + NADH. "
            "OGDH E1 uses TPP cofactor to decarboxylate α-KG; the succinyl-TPP intermediate is transferred "
            "to the lipoamide arm of DLST (E2), which then passes succinyl to CoA. DLD (E3) regenerates "
            "the dihydrolipoamide arm. OGDH LOF → α-KG cannot be decarboxylated → accumulates as "
            "2-oxoglutaric acid → 2-oxoglutaric aciduria (pathognomonic). TCA downstream impaired → "
            "NADH builds up → L:P ratio mildly elevated (15–30)."
        ),
        "omim_gene": "*604290",
        "omim_disease": "#203740 (2-Oxoglutaric Aciduria / αKGDH Complex Deficiency)",
        "inheritance": "Autosomal Recessive (AR, 7p13) — both sexes equally affected; no sex bias",
        "key_distinguishing_from_pdh": (
            "OGDH is αKGDH E1 — DIFFERENT complex from PDH. PDH block: L:P ratio NORMAL (10–20), "
            "2-oxoglutarate NORMAL, pyruvate elevated, PDH activity low. "
            "OGDH block: L:P ratio MILDLY elevated (15–30), 2-oxoglutarate MARKEDLY elevated (pathognomonic), "
            "pyruvate normal or mildly elevated, αKGDH activity severely reduced, PDH activity NORMAL."
        ),
        "key_distinguishing_feature": (
            "Urine 2-oxoglutarate (α-KG) MARKEDLY elevated (200–2200 µmol/mmol Cr) — PATHOGNOMONIC biomarker. "
            "L:P ratio mildly elevated (15–30; intermediate between PDH 10–20 normal and Complex I >25). "
            "BCAA NORMAL (BCKDH intact). Glycine NORMAL (GCS intact). DLD/E3 free activity NORMAL. "
            "PDH enzyme activity NORMAL. αKGDH complex activity <10% normal."
        ),
        "structural_brain_hallmark": (
            "Leigh syndrome (symmetric BG + brainstem lesions, ~55% in cohort); "
            "Basal ganglia lesions — putamen and caudate preferentially affected (~55%); "
            "Cortical atrophy (~45%)."
        ),
        "kpis": {
            "cohort_n": n,
            "male_pct": round(100 * male_n / n),
            "avg_plasma_lactate_mmol": avg_lac,
            "avg_plasma_pyruvate_mmol": avg_pyr,
            "avg_lp_ratio": avg_lp,
            "elevated_lp_ratio_gt20_pct": round(100 * elevated_lp / n),
            "avg_urine_2_oxoglutarate_umol_mmolCr": avg_okg,
            "leigh_lesions_pct": round(100 * leigh_n / n),
            "basal_ganglia_lesions_pct": round(100 * bg_n / n),
            "on_thiamine_pct": round(100 * thiamine_n / n),
            "thiamine_responsive_pct": round(100 * responsive_n / n),
            "on_kd_pct": round(100 * kd_n / n),
            "dre_pct": round(100 * dre_n / n),
            "vpa_avoided_pct": round(100 * vpa_ok / n),
        },
        "akgdh_complex_components": [
            {"component": "E1 — OGDH (oxoglutarate decarboxylase) ← THIS GENE",
             "function": "Decarboxylates α-KG using TPP cofactor; transfers succinyl group to E2 lipoamide arm"},
            {"component": "E2 — DLST (dihydrolipoamide S-succinyltransferase)",
             "function": "Transfers succinyl group from OGDH succinyl-TPP to CoA → succinyl-CoA"},
            {"component": "E3 — DLD (dihydrolipoamide dehydrogenase, FAD homodimer)",
             "function": "Regenerates lipoamide arm of DLST (oxidises NADH → NAD⁺); SHARED with PDH, BCKDH, GCS"},
        ],
        "tca_cycle_position": {
            "step_3": "Isocitrate → α-KG (by ICDH, isocitrate dehydrogenase; NAD⁺ dependent)",
            "step_4_blocked": "α-KG → succinyl-CoA (by αKGDH = OGDH+DLST+DLD — BLOCKED IN OGDH DEFICIENCY)",
            "step_5": "Succinyl-CoA → succinate (by succinyl-CoA synthetase; substrate-level ATP)",
            "step_6": "Succinate → fumarate (by succinate dehydrogenase / Complex II)",
            "consequence": "Steps 5–8 of TCA cycle still proceed if succinyl-CoA is supplied from BCAA catabolism (propionyl-CoA → methylmalonyl-CoA → succinyl-CoA via BCKDH, intact in OGDH deficiency), partially compensating",
        },
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "CAUTION",
             "mechanism": "Mitochondrial hepatotoxicity in any mitochondrial disease. Not absolute CI as in PDH deficiency (no KD dependency), but avoid if LEV is effective. Monitor LFTs if VPA used."},
            {"drug": "High-protein diet (unconstrained)", "risk": "HAZARD",
             "mechanism": "Amino acid catabolism: glutamate → α-KG (via glutamate dehydrogenase); asparagine → aspartate → OAA → α-KG pathway; protein-rich meals increase α-KG load → worsen 2-oxoglutaric aciduria + lactic acidosis."},
            {"drug": "Fasting / prolonged NPO", "risk": "HAZARD",
             "mechanism": "Fasting mobilises glutamine/BCAA → α-KG surge via transaminases; reduces glucose as TCA carbon source. Oral succinic acid / glucose-electrolyte preferred during illness."},
            {"drug": "Exercise (unmanaged)", "risk": "CAUTION",
             "mechanism": "Increased TCA flux demand during exercise → α-KG surge → metabolic crisis. Graduated exercise with thiamine pre-loading; avoid fasted exercise."},
        ],
        "structural_anomalies": [
            {"anomaly": "Leigh syndrome (BG + brainstem T2 lesions)", "frequency_pct": 55,
             "significance": "Symmetric necrotising lesions; putamen and caudate most affected; periaqueductal grey involvement; most common OGDH MRI pattern"},
            {"anomaly": "Basal ganglia lesions (bilateral putamen/caudate)", "frequency_pct": 55,
             "significance": "TCA cycle block → energy failure in high-metabolic-demand BG neurons; similar distribution to SUCLA2 and SDH deficiency"},
            {"anomaly": "Cortical atrophy", "frequency_pct": 45,
             "significance": "Secondary to neuronal energy failure; more prominent in severe neonatal and Leigh phenotypes; less corpus callosum agenesis than PDH complex deficiency"},
            {"anomaly": "White matter changes (periventricular)", "frequency_pct": 32,
             "significance": "Myelination energy failure; NADH-driven damage; less prominent than in PDH complex deficiency"},
            {"anomaly": "Corpus callosum hypoplasia/dysgenesis", "frequency_pct": 20,
             "significance": "Less common than PDH deficiency (PDHA1 70%); seen primarily in severe neonatal OGDH; no X-linked sex bias (AR)"},
            {"anomaly": "Brainstem hypoplasia / hypomyelination", "frequency_pct": 25,
             "significance": "Posterior fossa energy failure; seen in severe neonatal and Leigh phenotypes"},
        ],
        "phenotype_distribution": [
            {"label": "Leigh Syndrome (Infantile)", "pct": 40, "color": "#b71c1c"},
            {"label": "Severe Neonatal (Neonatal Lactic Acidosis)", "pct": 30, "color": "#d84315"},
            {"label": "Childhood Episodic Encephalopathy", "pct": 20, "color": "#f57f17"},
            {"label": "Mild/Juvenile (Partial Deficiency)", "pct": 10, "color": "#558b2f"},
        ],
    }


def get_breakdown():
    c = _COHORT
    sample = c[:10]

    return {
        "biomarkers": [
            {"name": "Urine 2-oxoglutarate (α-KG)", "normal": "<30 µmol/mmol Cr",
             "ogdh_range": "200–2200 µmol/mmol Cr (MARKEDLY elevated — PATHOGNOMONIC)",
             "significance": "Primary accumulation product of OGDH E1 block; pathognomonic in context of Leigh/lactic acidosis; measured on urine organic acid GC-MS; should be in EVERY metabolic work-up for lactic acidosis"},
            {"name": "Plasma Lactate", "normal": "<2.2 mmol/L",
             "ogdh_range": "3–18 mmol/L",
             "significance": "Secondary to TCA block → NADH accumulation → LDH drives pyruvate→lactate; present in all severe phenotypes"},
            {"name": "Plasma Pyruvate", "normal": "<0.17 mmol/L",
             "ogdh_range": "0.15–0.45 mmol/L (normal or mildly elevated)",
             "significance": "PDH is INTACT in OGDH deficiency → pyruvate is consumed normally; pyruvate may be near-normal — distinguishes from PDH block where pyruvate is markedly elevated"},
            {"name": "L:P ratio (Lactate:Pyruvate)", "normal": "10–20",
             "ogdh_range": "15–30 (mildly elevated — intermediate)",
             "significance": "L:P MILDLY elevated in OGDH — TCA block impairs NAD⁺ regeneration (NADH builds), but PDH is intact so both lactate AND pyruvate don't accumulate proportionally. KEY: L:P <25 does NOT exclude OGDH (unlike Complex I which is typically >25); 2-oxoglutarate measurement is essential"},
            {"name": "Plasma Alanine", "normal": "150–450 µmol/L",
             "ogdh_range": "200–550 µmol/L (mildly elevated)",
             "significance": "Reflects mild pyruvate/lactate elevation (transamination surrogate); less elevated than in PDH deficiency (where pyruvate markedly elevated); mild alanine elevation in OGDH"},
            {"name": "Plasma Glutamate", "normal": "40–120 µmol/L",
             "ogdh_range": "80–250 µmol/L (elevated)",
             "significance": "α-KG accumulation → reverse transamination: α-KG + glutamate ⇌ 2-oxoglutarate; elevated glutamate reflects α-KG pool expansion; useful secondary marker for αKGDH deficiency"},
            {"name": "Plasma BCAA (Leu/Ile/Val)", "normal": "Normal",
             "ogdh_range": "NORMAL — key negative",
             "significance": "OGDH deficiency blocks ONLY αKGDH (E1). BCKDH uses its OWN E1 (BCKDHA/BCKDHB) and the shared DLD E3. Since DLD is intact, BCKDH is fully functional → BCAA NORMAL. Elevated BCAA → DLD deficiency, not OGDH."},
            {"name": "Plasma Glycine", "normal": "170–340 µmol/L",
             "ogdh_range": "NORMAL — key negative",
             "significance": "GCS (glycine cleavage system) uses DLD as its L-protein (E3). DLD is intact in OGDH deficiency → GCS normal → glycine NORMAL. Elevated glycine → NKH or DLD deficiency, not OGDH."},
            {"name": "DLD/E3 free enzyme activity", "normal": "Normal",
             "ogdh_range": "NORMAL (DLD protein is intact; only OGDH E1 is deficient)",
             "significance": "CRITICAL: DLD free homodimer activity is NORMAL in OGDH deficiency — DLD can still anchor to BCKDH and GCS normally. Only αKGDH-bound DLD activity is missing because OGDH E1 is absent. Severely reduced DLD activity → DLD deficiency."},
            {"name": "αKGDH complex activity (fibroblasts)", "normal": "Normal",
             "ogdh_range": "<10% normal (severely reduced; OGDH E1 step blocked)",
             "significance": "Definitive biochemical diagnosis: αKGDH complex activity in cultured fibroblasts or muscle biopsy; <10% normal confirms OGDH/DLST deficiency. Measures E1+E2+E3 coupled activity using α-KG substrate."},
            {"name": "PDH enzyme activity (fibroblasts)", "normal": "Normal",
             "ogdh_range": "NORMAL — key negative",
             "significance": "PDH (pyruvate dehydrogenase) uses PDHA1/PDHB (E1), DLAT (E2), DLD (E3). All intact in OGDH deficiency → PDH activity NORMAL. Reduced PDH → PDHA1/PDHB/DLAT/PDHX, not OGDH."},
        ],
        "key_variants": [
            {"variant": "p.Arg263Cys (c.787C>T)", "effect": "TPP-binding domain; reduces thiamine pyrophosphate cofactor affinity; partial residual OGDH E1 activity",
             "phenotype": "Episodic / Childhood-onset; thiamine-responsive subset"},
            {"variant": "p.Gly354Asp", "effect": "Active site; near decarboxylation catalytic residues; severely impairs E1 decarboxylase activity",
             "phenotype": "Leigh Syndrome (infantile-severe)"},
            {"variant": "p.Arg447His", "effect": "TPP-binding/DLST (E2)-docking interface; disrupts succinyl transfer to E2 lipoamide arm",
             "phenotype": "Leigh Syndrome (moderate-severe)"},
            {"variant": "p.Tyr395Cys", "effect": "Near lipoyl-arm docking region; impairs succinyl-group capture by DLST E2 lipoamide",
             "phenotype": "Leigh Syndrome / Severe neonatal"},
            {"variant": "c.1237+1G>A (splice donor)", "effect": "Null — complete loss of OGDH E1; no residual decarboxylase activity; severe",
             "phenotype": "Severe Neonatal (neonatal lactic acidosis)"},
        ],
        "patients_sample": [
            {
                "id": p["id"],
                "sex": p["sex"],
                "phenotype": p["phenotype"],
                "onset_age_months": p["onset_age_months"],
                "plasma_lactate_mmol": p["plasma_lactate_mmol"],
                "plasma_pyruvate_mmol": p["plasma_pyruvate_mmol"],
                "lp_ratio": p["lp_ratio"],
                "urine_2_oxoglutarate": p["urine_2_oxoglutarate"],
                "plasma_alanine_umol": p["plasma_alanine_umol"],
                "leigh_lesions": p["leigh_lesions"],
                "on_thiamine": p["on_thiamine"],
                "thiamine_responsive": p["thiamine_responsive"],
                "on_kd": p["on_kd"],
                "vpa_avoided": p["vpa_avoided"],
            }
            for p in sample
        ],
        "seizure_types": [
            {"type": "Focal seizures with secondary generalisation", "pct": 55},
            {"type": "Tonic / tonic-clonic (GTC)", "pct": 48},
            {"type": "Infantile spasms (IS / West syndrome)", "pct": 42},
            {"type": "Myoclonic seizures", "pct": 35},
            {"type": "Absence-like / staring episodes", "pct": 22},
            {"type": "Status epilepticus (SE)", "pct": 20},
        ],
        "trigger_types": [
            {"trigger": "Febrile illness / infection (α-KG surge from proteolysis)", "pct": 78},
            {"trigger": "High-protein meal (glutamine → glutamate → α-KG)", "pct": 65},
            {"trigger": "Fasting / prolonged NPO (BCAA → α-KG via transaminases)", "pct": 60},
            {"trigger": "Exercise (increased TCA flux demand)", "pct": 52},
            {"trigger": "Surgery / perioperative stress", "pct": 42},
            {"trigger": "Intercurrent GI illness (catabolism → α-KG surge)", "pct": 38},
        ],
        "treatments": [
            {"drug": "Thiamine (B1, 100–600 mg/day)", "level": "A", "response_pct": 45, "color": "#1565c0"},
            {"drug": "Riboflavin (B2, 100–300 mg/day)", "level": "B", "response_pct": 38, "color": "#4a148c"},
            {"drug": "Lipoic acid (experimental)", "level": "C", "response_pct": 20, "color": "#827717"},
            {"drug": "Succinate supplementation (experimental)", "level": "C", "response_pct": 18, "color": "#00695c"},
            {"drug": "LEV (Levetiracetam, AED)", "level": "B", "response_pct": 68, "color": "#006064"},
            {"drug": "Ketogenic diet (experimental, limited evidence)", "level": "C", "response_pct": 22, "color": "#1b5e20"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "CAUTION",
             "mechanism": "Mitochondrial hepatotoxicity risk in any mitochondrial disease. Unlike PDH deficiency, VPA is not absolute CI in OGDH (no KD-first-line dependency), but prefer LEV. Monitor LFTs closely if VPA is used."},
            {"drug": "High-protein diet (unrestricted)", "risk": "HAZARD",
             "mechanism": "Protein catabolism → glutamate → α-KG (glutamate dehydrogenase). High amino acid intake floods the blocked αKGDH step, worsening 2-oxoglutaric aciduria and lactic acidosis. Moderate protein restriction (2–2.5 g/kg/day for children) with leucine monitoring."},
            {"drug": "Prolonged fasting / NPO", "risk": "HAZARD",
             "mechanism": "BCAA mobilisation (Leu/Ile/Val → BCAA-ATs → α-KG); glutamine release from muscle → α-KG surge. Crisis prevention: succinic acid drink, glucose-electrolyte rehydration, avoid >4-hour fast in young patients."},
            {"drug": "Exercise (uncontrolled intense)", "risk": "CAUTION",
             "mechanism": "TCA flux demand spikes with exercise; α-KG surge if OGDH cannot decarboxylate; episodic metabolic crisis reported. Graduated exercise with pre-loading thiamine; avoid fasted exercise."},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "Gene": "OGDH",
            "Full name": "2-Oxoglutarate Dehydrogenase (E1 subunit of αKGDH complex)",
            "Locus": "7p13",
            "Protein length": "1023 aa (includes mitochondrial targeting sequence, MTS; mature processed form ~966 aa)",
            "Complex": "αKGDH complex (2-OGDHC): OGDH (E1) + DLST (E2) + DLD (E3, shared)",
            "Cofactor": "TPP (thiamine pyrophosphate / ThDP) — essential; bound by E1 active site; decarboxylates α-KG",
            "Reaction catalysed": "α-KG + TPP → succinyl-TPP + CO₂; succinyl transferred to DLST E2 lipoamide → succinyl-CoA",
            "TCA cycle step": "Step 4: α-KG → succinyl-CoA (between ICDH step 3 and succinyl-CoA synthetase step 5)",
            "OMIM gene": "*604290",
            "OMIM disease": "#203740 (2-Oxoglutaric Aciduria / αKGDH Complex Deficiency)",
            "Inheritance": "Autosomal Recessive (AR) — both sexes equally affected",
            "Cases worldwide (2026)": "~20–40 confirmed OGDH cases (ultra-rare; among rarest metabolic epilepsies)",
            "Primary biomarker": "Urine 2-oxoglutarate: 200–2200 µmol/mmol Cr (PATHOGNOMONIC; normal <30)",
            "Key negative biomarkers": "Normal BCAA, normal glycine, normal DLD free activity, normal PDH activity",
        },
        "key_concepts": [
            {
                "term": "OGDH E1 — the α-KG decarboxylase (TPP-dependent)",
                "definition": (
                    "OGDH encodes the E1 subunit of the αKGDH complex, which performs the first and "
                    "rate-limiting step: decarboxylation of 2-oxoglutarate (α-ketoglutarate) using "
                    "thiamine pyrophosphate (TPP) as cofactor. The mechanism mirrors PDHA1 (PDH E1α) "
                    "but acts on a different substrate (α-KG vs pyruvate). The succinyl-TPP "
                    "intermediate is then transferred to the lipoamide arm of DLST (E2), forming "
                    "S-succinyl-dihydrolipoamide-DLST. OGDH LOF → no succinyl-TPP → α-KG accumulates."
                ),
            },
            {
                "term": "2-Oxoglutaric aciduria — the pathognomonic biomarker",
                "definition": (
                    "In OGDH deficiency, α-KG (2-oxoglutaric acid) is the direct substrate of the "
                    "blocked enzyme. It accumulates in plasma, CSF, and urine. Urine organic acid "
                    "analysis by GC-MS shows markedly elevated 2-oxoglutarate (200–2200 µmol/mmol "
                    "creatinine; normal <30). This is PATHOGNOMONIC for αKGDH complex deficiency "
                    "(OGDH or DLST). The same marker is also elevated in DLD deficiency (because DLD "
                    "E3 is shared with αKGDH) — but DLD also shows BCAA elevation and glycine "
                    "elevation, distinguishing it from isolated OGDH/DLST deficiency."
                ),
            },
            {
                "term": "L:P ratio — INTERMEDIATE elevation (15–30), not normal like PDH",
                "definition": (
                    "In PDH complex deficiency (PDHA1/PDHB/DLAT/PDHX): L:P ratio NORMAL (10–20) — "
                    "BOTH lactate and pyruvate accumulate equally. "
                    "In OGDH deficiency: L:P ratio MILDLY elevated (15–30) — the TCA cycle block "
                    "at step 4 reduces NAD+ regeneration from TCA (NADH accumulates), which "
                    "shifts LDH toward lactate. However, PDH is intact → pyruvate is consumed → "
                    "pyruvate does not markedly accumulate → the L:P ratio rises moderately, "
                    "not as dramatically as Complex I (where L:P >25). "
                    "KEY: do NOT use L:P <25 to exclude OGDH — measure urine 2-oxoglutarate."
                ),
            },
            {
                "term": "BCAA NORMAL — BCKDH is intact (DLD is intact)",
                "definition": (
                    "The branched-chain alpha-keto acid dehydrogenase (BCKDH) complex uses its own "
                    "E1 subunit (BCKDHA/BCKDHB) and the shared DLD E3. OGDH deficiency only removes "
                    "the αKGDH E1 (OGDH) — DLD (E3) is fully intact, and BCKDHA/BCKDHB are intact. "
                    "Therefore BCKDH is fully functional → BCAA (leucine, isoleucine, valine) are "
                    "normal. Elevated BCAA suggests DLD deficiency (which blocks BCKDH via E3 "
                    "impairment) or primary MSUD (BCKDHA/B/DBT deficiency), not OGDH."
                ),
            },
            {
                "term": "Thiamine (B1) — the most evidence-based treatment (Level A)",
                "definition": (
                    "TPP (thiamine pyrophosphate) is the essential cofactor for OGDH E1. Mutations "
                    "in the TPP-binding domain (e.g., p.Arg263Cys) reduce cofactor affinity — "
                    "pharmacological thiamine doses can partially restore activity by mass action. "
                    "Response rate ~40–55% in TPP-binding domain variants; null variants (frameshift, "
                    "splice site) do not respond. All patients receive thiamine trial (100–600 mg/day) "
                    "regardless of variant, as response cannot be predicted without trial. This mirrors "
                    "the PDHA1 thiamine strategy (same E1-TPP mechanism, different substrate)."
                ),
            },
            {
                "term": "Succinate supplementation — anaplerotic bypass",
                "definition": (
                    "In OGDH deficiency, the TCA is blocked at step 4 (α-KG → succinyl-CoA). "
                    "Succinyl-CoA and downstream TCA intermediates (succinate, fumarate, malate, OAA) "
                    "become depleted. Oral succinate supplementation can theoretically bypass the block "
                    "by directly supplying succinate downstream of the lesion — replenishing TCA "
                    "intermediates for anaplerosis (OAA → citrate synthase entry). Evidence: limited "
                    "anecdotal reports; Level C experimental. Contrast with KD in PDH deficiency "
                    "(Level A), where ketones → acetyl-CoA directly bypasses the blocked PDH step."
                ),
            },
            {
                "term": "Ketogenic diet — limited bypass (Level C, unlike PDH deficiency Level A)",
                "definition": (
                    "In PDH deficiency, KD bypasses the PDH block: ketones → acetyl-CoA (via SCOT + "
                    "thiolase), entering TCA without needing PDH. KD is Level A first-line for PDH. "
                    "In OGDH deficiency, the block is at TCA step 4 (α-KG → succinyl-CoA). Ketones "
                    "still → acetyl-CoA → TCA entry, but TCA then hits the OGDH block at step 4. "
                    "KD does NOT provide succinyl-CoA to bypass the OGDH block. Limited evidence; "
                    "use only if other options fail; Level C experimental in OGDH deficiency."
                ),
            },
            {
                "term": "DLD (E3) is INTACT in OGDH deficiency — free DLD activity NORMAL",
                "definition": (
                    "DLD (E3, dihydrolipoamide dehydrogenase) is shared between αKGDH, PDH, BCKDH, "
                    "and GCS. In OGDH deficiency, only the αKGDH E1 subunit (OGDH) is deficient. "
                    "DLD protein and free homodimer activity are NORMAL. GCS, BCKDH, and PDH complexes "
                    "retain their DLD-anchored E3 activity. αKGDH activity is severely reduced "
                    "(because E1 is missing, the complex cannot function even if E2/DLD are intact). "
                    "Severely reduced free DLD activity → DLD deficiency (4-complex block)."
                ),
            },
            {
                "term": "AR inheritance — 7p13, no sex bias",
                "definition": (
                    "OGDH (7p13) is autosomal recessive — biallelic LOF required for disease. "
                    "Both sexes equally affected (no X-inactivation variability as in PDHA1 X-linked). "
                    "No common founder allele; all variants pan-ethnic private-family. "
                    "~20–40 confirmed cases worldwide by 2026 — among the rarest metabolic epilepsies."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease": "DLD deficiency (E3 — shared enzyme)",
                "distinguishing_features": (
                    "DLD blocks ALL four complexes: αKGDH + PDH + BCKDH + GCS. "
                    "Elevated 2-oxoglutarate (same as OGDH) PLUS elevated BCAA + glycine + lactate. "
                    "DLD/E3 free enzyme activity SEVERELY REDUCED (<10% normal) — KEY distinction from OGDH. "
                    "OGDH: DLD free activity NORMAL; BCAA normal; glycine normal."
                ),
            },
            {
                "disease": "DLST deficiency (αKGDH E2 subunit)",
                "distinguishing_features": (
                    "Same αKGDH complex; same primary biomarker (urine 2-oxoglutarate elevated). "
                    "Biochemically IDENTICAL to OGDH — gene panel required to distinguish. "
                    "DLST at 14q24.3; OGDH at 7p13. αKGDH complex activity reduced for both."
                ),
            },
            {
                "disease": "PDH complex deficiency (PDHA1/PDHB/DLAT/PDHX)",
                "distinguishing_features": (
                    "PDH block: L:P NORMAL (10–20); pyruvate markedly elevated; alanine markedly elevated; "
                    "2-oxoglutarate NORMAL. PDH enzyme activity severely reduced. "
                    "OGDH block: L:P mildly elevated (15–30); pyruvate normal/mildly elevated; "
                    "2-oxoglutarate MARKEDLY elevated; PDH activity NORMAL."
                ),
            },
            {
                "disease": "Complex I deficiency (NADH:ubiquinone oxidoreductase)",
                "distinguishing_features": (
                    "L:P ratio dramatically elevated (>25, often >30) due to NADH accumulation — NADH cannot be "
                    "oxidised by ETC. 2-oxoglutarate normal. Multiple OXPHOS gene mutations. "
                    "OGDH: L:P intermediate (15–30); 2-oxoglutarate markedly elevated; OXPHOS genes normal."
                ),
            },
            {
                "disease": "2-HGA type I (L2HGDH deficiency) / type II (D2HGDH deficiency)",
                "distinguishing_features": (
                    "2-Hydroxyglutaric aciduria — 2-HGA (L or D form) elevated. "
                    "In OGDH deficiency, 2-oxoglutarate (not 2-HGA) is the primary elevated metabolite. "
                    "2-HGA may be MILDLY elevated in OGDH (non-enzymatic reduction of α-KG), but 2-oxoglutarate "
                    "is the dominant marker. L2HGDH/D2HGDH: 2-oxoglutarate normal or mildly elevated."
                ),
            },
            {
                "disease": "Glutaric aciduria type I (GCDH deficiency)",
                "distinguishing_features": (
                    "Glutaryl-CoA dehydrogenase deficiency → glutaric acid + 3-hydroxyglutaric acid elevated. "
                    "Striatal injury (selective basal ganglia crisis after febrile illness). "
                    "OGDH: 2-oxoglutarate elevated, not glutaric acid; L:P elevated; biochemically distinct."
                ),
            },
        ],
        "diagnostic_thresholds": {
            "urine_2_oxoglutarate_threshold": ">30 µmol/mmol Cr (normal); OGDH range 200–2200 — PATHOGNOMONIC marker",
            "plasma_lactate_threshold": ">2.2 mmol/L; OGDH range 3–18 mmol/L",
            "plasma_pyruvate_ogdh_range": "0.15–0.45 mmol/L (normal or mildly elevated; PDH intact)",
            "lp_ratio_ogdh_range": "15–30 (mildly elevated; intermediate; use 2-oxoglutarate to confirm)",
            "bcaa_key_negative": "NORMAL — key negative vs DLD deficiency",
            "glycine_key_negative": "NORMAL — key negative vs DLD/NKH",
            "dld_free_enzyme_normal": "NORMAL — key negative vs DLD deficiency (<10% in DLD)",
            "pdh_enzyme_normal": "NORMAL — key negative vs PDH complex deficiency",
            "akgdh_complex_activity": "<10% normal in fibroblasts (definitive); use α-KG substrate for assay",
            "plasma_glutamate": ">80 µmol/L (elevated; α-KG → glutamate transamination; secondary marker)",
        },
    }
