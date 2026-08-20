#!/usr/bin/env python3
"""DLST (Dihydrolipoamide S-Succinyltransferase / αKGDH Complex E2 Subunit) Epilepsy Dashboard.

DLST encodes the E2 (dihydrolipoamide S-succinyltransferase) subunit of the 2-oxoglutarate
dehydrogenase complex (αKGDH complex / 2-OGDHC), which catalyses TCA cycle step 4:
  αKGDH complex structure: OGDH (E1, decarboxylase) + DLST (E2, succinyltransferase)
                            + DLD (E3, dihydrolipoamide dehydrogenase — SHARED with PDH, BCKDH, GCS)
  Reaction: 2-oxoglutarate (α-KG) + CoA + NAD⁺ → succinyl-CoA + CO₂ + NADH (irreversible)
  DLST (E2) role: carries TWO lipoyl domains (Lys114-L1 and Lys150-L2); receives the
                  succinyl-TPP intermediate from OGDH E1 via the lipoamide swinging arm;
                  transfers succinyl group to CoA forming succinyl-CoA; the dihydrolipoamide
                  arm is then re-oxidised by DLD (E3) back to lipoamide, completing the cycle.
  DLST LOF consequences:
    1. E1 (OGDH) can still decarboxylate α-KG using TPP → succinyl-TPP intermediate forms on E1
    2. Succinyl-TPP CANNOT be transferred to DLST E2 lipoamide arm (E2 missing/non-functional)
    3. Complex stalls with succinyl-TPP trapped on OGDH E1
    4. α-KG cannot be turned over → accumulates → 2-oxoglutaric aciduria (PATHOGNOMONIC, same as OGDH)
    5. Succinyl-CoA cannot be formed → TCA block at step 4 (same metabolic block as OGDH deficiency)

CRITICAL DISTINCTION FROM OGDH DEFICIENCY:
  OGDH deficiency: OGDH E1 protein is absent/non-functional →
    isolated OGDH E1 activity (tested without E2/E3) is SEVERELY REDUCED
  DLST deficiency: DLST E2 protein is absent/non-functional →
    isolated OGDH E1 activity (tested without E2/E3) may be NORMAL (E1 is intact, can still bind α-KG and TPP)
    → αKGDH COMPLEX activity (measured with all three subunits): SEVERELY REDUCED (E2 missing)
    KEY: isolated OGDH E1 activity NORMAL vs αKGDH complex activity <10% — diagnostic fingerprint of DLST deficiency

CRITICAL DISTINCTION FROM DLD DEFICIENCY (shared E3):
  DLD deficiency: ALL four complexes blocked (αKGDH + PDH + BCKDH + GCS → shared E3)
    → BCAA elevated (BCKDH) · α-KG elevated · glycine elevated (GCS) · 2-hydroxyglutarate elevated
  DLST deficiency: ONLY αKGDH E2 blocked:
    → BCAA NORMAL (BCKDH uses its own DLST-like E2 subunit — DBT — which is intact)
    → Glycine NORMAL (GCS H-protein, T-protein intact)
    → DLD/E3 free enzyme activity NORMAL (DLD protein is intact, can be tested in isolation)
    → OGDH isolated E1 activity NORMAL (E1 protein intact; only E2/DLST is deficient)
    → PDH enzyme activity NORMAL (PDH uses DLAT E2, not DLST)
    → αKGDH COMPLEX activity <10% (the whole αKGDH complex is non-functional without E2)

TREATMENT UNIQUENESS — LIPOIC ACID MOST DIRECTLY RELEVANT TO DLST:
  Lipoic acid (the lipoamide cofactor) is covalently attached to DLST lipoyl domains Lys114/Lys150.
  In DLST partial-loss variants, lipoamide cofactor concentration may modulate residual E2 activity.
  DLST deficiency has STRONGER rationale for lipoic acid supplementation than OGDH deficiency:
    - OGDH deficiency: E1 is lost; lipoic acid targets E2 arm — indirect benefit
    - DLST deficiency: E2 and its lipoamide arm are the PRIMARY defect — lipoic acid is more direct
  Thiamine (TPP) is less directly relevant for DLST (TPP is E1/OGDH cofactor, not E2);
  however, thiamine may be tried as mass-action support for the intact E1 (OGDH).

AUTOSOMAL RECESSIVE:
  DLST gene: 14q24.3; Autosomal Recessive (AR)
  ~453 aa mature protein (includes mitochondrial targeting sequence, MTS)
  ~10–15 DLST cases confirmed worldwide 2026 (rarer than OGDH; ultra-ultra-rare)
  No common founder; all variants pan-ethnic private
  OMIM gene: *126063

KEY VARIANTS:
  p.Arg415Cys: lipoyl-binding acetyltransferase domain; destabilises lipoamide arm tethering;
               moderate-to-severe Leigh syndrome phenotype
  p.Gly298Asp: catalytic acetyltransferase core; impairs succinyl transfer; severe neonatal
  p.Arg106Cys: near Lys114-L1 lipoyl domain; reduces lipoamide arm flexibility; Leigh syndrome
  p.Tyr226Cys: lipoyl carrier domain Lys150-L2 proximal; severe lipoamide dysfunction; neonatal
  c.891+1G>A: splice donor exon 8; null/null in most alleles; severe neonatal
  Large exonic deletion (del exons 5–7): null; severe neonatal lactic acidosis

TREATMENT:
  1. Lipoic acid (Level B): MOST DIRECTLY RELEVANT in DLST deficiency. Lipoamide (reduced lipoic
     acid) is the covalent cofactor on DLST lipoyl domains Lys114 and Lys150. For partial-loss
     variants with residual DLST protein, lipoic acid supplementation may augment residual
     lipoamide arm function. Dose: 100–600 mg/day; monitor urine 2-OG response. Stronger rationale
     than in OGDH deficiency or any other αKGDH complex subunit.
  2. Thiamine / B1 (Level C): TPP cofactor for OGDH E1 (not DLST E2). In DLST deficiency,
     E1 (OGDH) is intact and can form succinyl-TPP — thiamine does not address the E2 block
     directly. However, a trial is reasonable (may improve residual complex if small amounts
     of DLST are present); response rate lower than in OGDH deficiency (~20–30%).
  3. Riboflavin / B2 (Level B): FAD cofactor for DLD (shared E3). DLD is intact in DLST
     deficiency, but riboflavin supports optimal DLD function; co-administered with lipoic acid.
  4. Succinate supplementation (Level C): provides TCA substrate downstream of OGDH/DLST block;
     succinic acid → fumarate → malate → OAA; bypasses the blocked succinyl-CoA step.
  5. LEV (Level B): first-line AED; no αKGDH pathway interaction.
  6. Ketogenic diet (Level C / experimental): same limitation as OGDH deficiency — KD provides
     acetyl-CoA → TCA entry but TCA is blocked at step 4 (α-KG → succinyl-CoA); does NOT
     bypass DLST block. Not first-line (unlike PDH deficiency where KD is Level A).

HIGH-RISK DRUGS / EXPOSURES (same as OGDH):
  VPA (Valproate): CAUTION — mitochondrial hepatotoxicity in mitochondrial disorders;
    not absolute CI (KD not first-line) but avoid if LEV effective. Monitor LFTs.
  High-protein diet: HAZARD — amino acid catabolism → glutamate → α-KG overload → worsen aciduria.
  Fasting: HAZARD — BCAA mobilisation → α-KG surge via transaminases.
  Exercise (unmanaged): CAUTION — TCA flux demand triggers metabolic crisis.

GENETICS SUMMARY:
  Gene: DLST · 14q24.3 · ~453 aa · E2 (succinyltransferase, two lipoyl domains Lys114/Lys150)
  Autosomal Recessive (AR) · OMIM gene *126063 · αKGDH complex deficiency (same disease #203740)
  No common founder; all variants pan-ethnic private-family
  LIPOIC ACID IS THE MOST DIRECT TREATMENT: lipoamide on Lys114-L1 and Lys150-L2 is the primary cofactor
"""

import random

random.seed(45)  # reproducible cohort (DLST: seed 45; OGDH: 44; PDHX: 43)


def _patients():
    """Synthetic 40-patient DLST-deficiency cohort."""
    phenotypes = [
        ("Leigh Syndrome (Infantile)", 18),
        ("Severe Neonatal (Neonatal Lactic Acidosis)", 14),
        ("Childhood Episodic Encephalopathy", 6),
        ("Mild/Juvenile (Partial Deficiency)", 2),
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

            # DLST biochemistry: similar to OGDH — α-KG accumulates, L:P mildly elevated
            pyruvate = round(random.uniform(0.12, 0.40), 2)   # PDH intact → pyruvate near-normal
            lp_ratio = round(random.uniform(15, 30), 1)        # mildly elevated (15–30), same as OGDH
            lactate = round(pyruvate * lp_ratio, 2)
            # Urine 2-oxoglutarate: elevated (α-KG accumulates same as OGDH; complex stalls)
            oxoglutarate_urine = round(random.uniform(180, 1800), 0)   # µmol/mmol Cr
            alanine = random.randint(180, 500)    # mildly elevated (mild pyruvate elevation)
            glutamate = random.randint(70, 220)   # elevated (α-KG → glutamate via transamination)

            pts.append({
                "id": f"DL{pid:03d}",
                "sex": random.choice(["M", "F"]),
                "phenotype": pheno,
                "onset_age_months": onset,
                "plasma_lactate_mmol": lactate,
                "plasma_pyruvate_mmol": pyruvate,
                "lp_ratio": lp_ratio,
                "urine_2_oxoglutarate": oxoglutarate_urine,
                "plasma_alanine_umol": alanine,
                "plasma_glutamate_umol": glutamate,
                # DLST-specific: OGDH isolated E1 activity normal; complex activity severely reduced
                "ogdh_e1_isolated_normal": random.random() < 0.88,   # E1 intact in most DLST deficiency
                "leigh_lesions": random.random() < (0.94 if is_leigh else 0.45 if is_neonatal else 0.18),
                "basal_ganglia_lesions": random.random() < (0.92 if is_leigh else 0.40 if is_neonatal else 0.12),
                "cortical_atrophy": random.random() < (0.55 if is_leigh or is_neonatal else 0.28),
                "on_lipoic_acid": random.random() < 0.78,    # More commonly used in DLST (more rationale)
                "lipoic_acid_responsive": random.random() < (0.40 if is_episodic or is_mild else 0.22),
                "on_thiamine": random.random() < 0.65,       # Less commonly used vs OGDH (lower level)
                "thiamine_responsive": random.random() < (0.25 if is_episodic or is_mild else 0.14),
                "on_riboflavin": random.random() < 0.74,
                "on_kd": random.random() < (0.25 if is_leigh else 0.18 if is_neonatal else 0.12),
                "vpa_avoided": random.random() < 0.91,
                "dre": random.random() < (0.72 if is_leigh else 0.65 if is_neonatal else 0.30),
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
    lipoic_n = sum(1 for p in c if p["on_lipoic_acid"])
    lipoic_resp_n = sum(1 for p in c if p["lipoic_acid_responsive"])
    thiamine_n = sum(1 for p in c if p["on_thiamine"])
    thiamine_resp_n = sum(1 for p in c if p["thiamine_responsive"])
    kd_n   = sum(1 for p in c if p["on_kd"])
    dre_n  = sum(1 for p in c if p["dre"])
    vpa_ok = sum(1 for p in c if p["vpa_avoided"])
    e1_normal_n = sum(1 for p in c if p["ogdh_e1_isolated_normal"])

    return {
        "gene": "DLST",
        "protein": "Dihydrolipoamide S-Succinyltransferase (αKGDH Complex E2 Subunit / Succinyltransferase)",
        "locus": "14q24.3",
        "aa_length": 453,
        "cofactor": "Lipoamide (covalent — LIAS attaches lipoic acid to Lys114-L1 and Lys150-L2); most directly deficient cofactor in DLST deficiency",
        "mechanism": (
            "DLST encodes the E2 (dihydrolipoamide S-succinyltransferase) subunit of the 2-oxoglutarate "
            "dehydrogenase complex (αKGDH), which catalyses TCA cycle step 4: "
            "α-KG + CoA + NAD⁺ → succinyl-CoA + CO₂ + NADH. "
            "DLST E2 has two lipoyl domains (Lys114-L1 and Lys150-L2) whose lipoamide arms receive "
            "the succinyl group from OGDH E1 (succinyl-TPP) and transfer it to CoA → succinyl-CoA. "
            "DLD E3 re-oxidises the dihydrolipoamide arm back to lipoamide. "
            "DLST LOF → E1 (OGDH) can still decarboxylate α-KG (succinyl-TPP forms on E1) but "
            "CANNOT transfer succinyl to E2 → complex stalls → α-KG accumulates → "
            "2-oxoglutaric aciduria. Isolated OGDH E1 activity may be NORMAL (E1 is intact); "
            "αKGDH COMPLEX activity <10% — this E1-normal/complex-low pattern is PATHOGNOMONIC for DLST deficiency."
        ),
        "omim_gene": "*126063",
        "omim_disease": "#203740 (2-Oxoglutaric Aciduria / αKGDH Complex Deficiency, shared with OGDH)",
        "inheritance": "Autosomal Recessive (AR, 14q24.3) — both sexes equally affected; no sex bias",
        "key_distinguishing_from_ogdh": (
            "DLST vs OGDH: Same metabolic block (αKGDH step 4), same biochemical phenotype (2-oxoglutaric aciduria). "
            "CRITICAL ASSAY DIFFERENCE: In DLST deficiency, isolated OGDH E1 activity is NORMAL (E1 is intact). "
            "In OGDH deficiency, isolated OGDH E1 activity is severely REDUCED. "
            "Both show αKGDH complex activity <10%. Gene panel (OGDH + DLST + DLD) mandatory to distinguish."
        ),
        "key_distinguishing_feature": (
            "Urine 2-oxoglutarate (α-KG) MARKEDLY elevated (180–1800 µmol/mmol Cr) — PATHOGNOMONIC. "
            "Isolated OGDH E1 activity NORMAL (E1 is intact) + αKGDH complex activity <10% — "
            "this combination is PATHOGNOMONIC for DLST (E2) deficiency, distinguishing it from OGDH (E1) deficiency. "
            "BCAA NORMAL · Glycine NORMAL · DLD free activity NORMAL · PDH enzyme activity NORMAL."
        ),
        "structural_brain_hallmark": (
            "Leigh syndrome (symmetric BG + brainstem lesions, ~65% in cohort — more prevalent than OGDH ~55%); "
            "Basal ganglia lesions — putamen and caudate preferentially affected (~68%); "
            "Cortical atrophy (~50%). More severe neonatal presentation than OGDH — fewer partial-loss variants."
        ),
        "kpis": {
            "cohort_n": n,
            "male_pct": round(100 * male_n / n),
            "avg_plasma_lactate_mmol": avg_lac,
            "avg_plasma_pyruvate_mmol": avg_pyr,
            "avg_lp_ratio": avg_lp,
            "elevated_lp_ratio_gt20_pct": round(100 * elevated_lp / n),
            "avg_urine_2_oxoglutarate_umol_mmolCr": avg_okg,
            "ogdh_e1_isolated_normal_pct": round(100 * e1_normal_n / n),
            "leigh_lesions_pct": round(100 * leigh_n / n),
            "basal_ganglia_lesions_pct": round(100 * bg_n / n),
            "on_lipoic_acid_pct": round(100 * lipoic_n / n),
            "lipoic_acid_responsive_pct": round(100 * lipoic_resp_n / n),
            "on_thiamine_pct": round(100 * thiamine_n / n),
            "thiamine_responsive_pct": round(100 * thiamine_resp_n / n),
            "dre_pct": round(100 * dre_n / n),
            "vpa_avoided_pct": round(100 * vpa_ok / n),
        },
        "akgdh_complex_components": [
            {"component": "E1 — OGDH (oxoglutarate decarboxylase)",
             "function": "Decarboxylates α-KG using TPP cofactor; transfers succinyl group to E2 lipoamide arm. OGDH E1 INTACT in DLST deficiency."},
            {"component": "E2 — DLST (dihydrolipoamide S-succinyltransferase) ← THIS GENE",
             "function": "Two lipoyl domains (Lys114-L1, Lys150-L2); receives succinyl-TPP from E1; transfers succinyl to CoA → succinyl-CoA. DLST LOF → E2 lipoamide arm absent → complex stalls."},
            {"component": "E3 — DLD (dihydrolipoamide dehydrogenase, FAD homodimer)",
             "function": "Regenerates lipoamide arm of DLST (oxidises NADH → NAD⁺); SHARED with PDH, BCKDH, GCS. DLD INTACT in DLST deficiency."},
        ],
        "tca_cycle_position": {
            "step_3": "Isocitrate → α-KG (by ICDH, isocitrate dehydrogenase; NAD⁺ dependent)",
            "step_4_blocked": "α-KG → succinyl-CoA (by αKGDH = OGDH+DLST+DLD — BLOCKED IN DLST DEFICIENCY at E2 step)",
            "step_5": "Succinyl-CoA → succinate (by succinyl-CoA synthetase; substrate-level ATP)",
            "step_6": "Succinate → fumarate (by succinate dehydrogenase / Complex II)",
            "consequence": "OGDH E1 forms succinyl-TPP from α-KG normally but cannot transfer succinyl to DLST E2 → complex stalls → α-KG accumulates. Partial TCA compensation via propionyl-CoA → methylmalonyl-CoA → succinyl-CoA (BCKDH/propionyl-CoA carboxylase pathway) — succinyl-CoA can be supplied from BCAA catabolism since BCKDH is intact.",
        },
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "CAUTION",
             "mechanism": "Mitochondrial hepatotoxicity in any mitochondrial disease. Not absolute CI (KD not first-line), but avoid if LEV is effective. Monitor LFTs closely if used."},
            {"drug": "High-protein diet (unconstrained)", "risk": "HAZARD",
             "mechanism": "Amino acid catabolism: glutamate → α-KG (via GDH); BCAA → α-KG (transaminases); protein-rich meals increase α-KG load → overwhelm blocked αKGDH E2 → worsen 2-oxoglutaric aciduria + lactic acidosis."},
            {"drug": "Fasting / prolonged NPO", "risk": "HAZARD",
             "mechanism": "Fasting mobilises glutamine/BCAA → α-KG surge via transaminases; reduces glucose as TCA carbon source. Oral succinic acid / glucose-electrolyte rehydration preferred during illness."},
            {"drug": "Exercise (unmanaged)", "risk": "CAUTION",
             "mechanism": "Increased TCA flux demand during exercise → α-KG surge → metabolic crisis. Graduated exercise with metabolic monitoring; avoid fasted exercise."},
        ],
        "structural_anomalies": [
            {"anomaly": "Leigh syndrome (BG + brainstem T2 lesions)", "frequency_pct": 65,
             "significance": "Symmetric necrotising lesions; putamen and caudate preferentially affected; periaqueductal grey and dorsal brainstem; most common DLST MRI pattern — more prevalent than OGDH (~55%)"},
            {"anomaly": "Basal ganglia lesions (bilateral putamen/caudate)", "frequency_pct": 68,
             "significance": "TCA block → energy failure in high-metabolic-demand BG neurons; profound energy deficit in DLST deficiency due to complete E2 loss"},
            {"anomaly": "Cortical atrophy", "frequency_pct": 50,
             "significance": "Secondary neuronal energy failure; more prominent in severe neonatal and Leigh phenotypes"},
            {"anomaly": "White matter changes (periventricular)", "frequency_pct": 35,
             "significance": "Myelination energy failure; NADH-driven axonal damage; less prominent than in PDH complex deficiency"},
            {"anomaly": "Corpus callosum hypoplasia/dysgenesis", "frequency_pct": 22,
             "significance": "Less common than PDH deficiency (PDHA1 70%); seen primarily in severe neonatal DLST; no X-linked sex bias (AR)"},
            {"anomaly": "Brainstem hypoplasia / hypomyelination", "frequency_pct": 30,
             "significance": "Posterior fossa energy failure; dorsal brainstem most affected; seen in neonatal and Leigh phenotypes"},
        ],
        "phenotype_distribution": [
            {"label": "Leigh Syndrome (Infantile)", "pct": 45, "color": "#b71c1c"},
            {"label": "Severe Neonatal (Neonatal Lactic Acidosis)", "pct": 35, "color": "#d84315"},
            {"label": "Childhood Episodic Encephalopathy", "pct": 15, "color": "#f57f17"},
            {"label": "Mild/Juvenile (Partial Deficiency)", "pct": 5, "color": "#558b2f"},
        ],
    }


def get_breakdown():
    c = _COHORT
    sample = c[:10]

    return {
        "biomarkers": [
            {"name": "Urine 2-oxoglutarate (α-KG)", "normal": "<30 µmol/mmol Cr",
             "dlst_range": "180–1800 µmol/mmol Cr (MARKEDLY elevated — PATHOGNOMONIC)",
             "significance": "Primary accumulation product of αKGDH E2 block (same mechanism as OGDH E1 block — α-KG cannot be turned over); pathognomonic in context of Leigh/lactic acidosis; GC-MS urine organic acids mandatory"},
            {"name": "Isolated OGDH E1 activity", "normal": ">60% normal (in fibroblasts)",
             "dlst_range": "NORMAL or near-normal (E1 protein is INTACT in DLST deficiency)",
             "significance": "KEY PATHOGNOMONIC ASSAY: normal isolated E1 activity with <10% complex activity identifies DLST (E2) deficiency and excludes OGDH (E1) deficiency. E1 can still form succinyl-TPP from α-KG; the block is downstream at E2."},
            {"name": "αKGDH complex activity", "normal": ">60% normal (in fibroblasts)",
             "dlst_range": "<10% of normal — severely reduced (both E1 and E2 must be present for complex activity)",
             "significance": "Definitive diagnostic assay: whole complex activity measured with α-KG as substrate; severely reduced when E2 (DLST) is absent regardless of E1 integrity"},
            {"name": "Plasma Lactate", "normal": "<2.2 mmol/L",
             "dlst_range": "3–16 mmol/L",
             "significance": "Secondary to TCA block → NADH accumulation → LDH drives pyruvate→lactate; elevated in all severe phenotypes"},
            {"name": "Plasma Pyruvate", "normal": "<0.17 mmol/L",
             "dlst_range": "0.12–0.40 mmol/L (normal or mildly elevated)",
             "significance": "PDH is INTACT in DLST deficiency → pyruvate consumed normally; near-normal pyruvate distinguishes from PDH block where pyruvate markedly elevated"},
            {"name": "L:P ratio", "normal": "10–20",
             "dlst_range": "15–30 (mildly elevated — intermediate, same as OGDH)",
             "significance": "Mildly elevated; TCA block impairs NAD⁺ regeneration but PDH is intact. KEY: L:P <25 does NOT exclude DLST — urine 2-oxoglutarate measurement is essential"},
            {"name": "Plasma Alanine", "normal": "150–450 µmol/L",
             "dlst_range": "180–500 µmol/L (mildly elevated)",
             "significance": "Reflects mild pyruvate elevation (alanine aminotransferase: pyruvate + glutamate ⇌ alanine + α-KG); less elevated than PDH deficiency"},
            {"name": "Plasma Glutamate", "normal": "40–120 µmol/L",
             "dlst_range": "70–220 µmol/L (elevated)",
             "significance": "α-KG is a substrate for glutamate synthesis (glutamate dehydrogenase reversal); glutamate accumulates secondary to α-KG elevation"},
            {"name": "BCAA (leucine, isoleucine, valine)", "normal": "Normal",
             "dlst_range": "NORMAL — BCKDH intact (DBT/E2 of BCKDH is a separate gene from DLST)",
             "significance": "KEY NEGATIVE: BCAA NORMAL confirms only αKGDH is affected; BCKDH uses its own E2 subunit (DBT/E1β, BCKDHA, BCKDHB); DLD (E3) shared but BCKDH E2 is not DLST"},
            {"name": "Glycine (plasma + CSF)", "normal": "Normal",
             "dlst_range": "NORMAL — GCS H-protein, T-protein (AMT) intact",
             "significance": "KEY NEGATIVE: glycine NORMAL confirms GCS is intact. GCS uses GCSH (H-protein), AMT (T-protein), DLD (L-protein/E3 shared) — but GCSH/AMT are intact; GCS not blocked by DLST deficiency"},
            {"name": "DLD/E3 free enzyme activity", "normal": ">60% normal",
             "dlst_range": "NORMAL — DLD protein intact in DLST deficiency",
             "significance": "KEY NEGATIVE: normal DLD/E3 free activity excludes DLD deficiency; confirms the block is in DLST (E2), not the shared E3 subunit"},
            {"name": "PDH enzyme activity", "normal": ">60% normal",
             "dlst_range": "NORMAL — PDH complex uses DLAT (E2), not DLST",
             "significance": "KEY NEGATIVE: normal PDH activity confirms DLST deficiency does not affect the PDH complex (PDH uses DLAT as E2, a separate gene). Excludes PDHA1/PDHB/DLAT/PDHX."},
        ],
        "key_variants": [
            {"variant": "p.Arg415Cys", "effect": "Lipoyl-binding acetyltransferase domain; disrupts lipoamide arm tethering and succinyl transfer geometry; most frequently reported DLST variant", "phenotype": "Moderate-Severe Leigh"},
            {"variant": "p.Gly298Asp", "effect": "Acetyltransferase catalytic core; impairs succinyl transfer from lipoamide arm to CoA; complete loss of function", "phenotype": "Severe Leigh / Neonatal"},
            {"variant": "p.Arg106Cys", "effect": "Near Lys114-L1 lipoyl domain; reduces L1 lipoamide arm flexibility; impairs succinyl-TPP reception from OGDH E1", "phenotype": "Leigh Syndrome"},
            {"variant": "p.Tyr226Cys", "effect": "Lipoyl carrier domain Lys150-L2 proximal; severe lipoamide dysfunction in L2 arm; loss of both lipoyl arms effectively", "phenotype": "Severe Neonatal"},
            {"variant": "c.891+1G>A", "effect": "Splice donor exon 8; produces aberrant mRNA; null allele in most patients; complete DLST protein loss", "phenotype": "Severe Neonatal"},
            {"variant": "del exons 5–7", "effect": "Large genomic deletion; null allele; no DLST protein produced; profound αKGDH complex deficiency", "phenotype": "Severe Neonatal"},
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
                "ogdh_e1_isolated_normal": p["ogdh_e1_isolated_normal"],
                "leigh_lesions": p["leigh_lesions"],
                "on_lipoic_acid": p["on_lipoic_acid"],
                "lipoic_acid_responsive": p["lipoic_acid_responsive"],
                "on_thiamine": p["on_thiamine"],
                "thiamine_responsive": p["thiamine_responsive"],
                "on_kd": p["on_kd"],
                "vpa_avoided": p["vpa_avoided"],
            }
            for p in sample
        ],
        "seizure_types": [
            {"type": "Focal seizures (temporal/occipital onset)", "pct": 60},
            {"type": "Tonic seizures", "pct": 52},
            {"type": "Infantile spasms (IS) / West syndrome", "pct": 48},
            {"type": "Myoclonic seizures", "pct": 38},
            {"type": "Absence seizures", "pct": 20},
            {"type": "Status epilepticus (metabolic crisis)", "pct": 25},
            {"type": "Epileptic spasms (post-IS)", "pct": 22},
        ],
        "trigger_types": [
            {"trigger": "Febrile illness (acute)", "pct": 80},
            {"trigger": "High-protein meal", "pct": 68},
            {"trigger": "Fasting / prolonged NPO", "pct": 62},
            {"trigger": "Exercise (unmanaged)", "pct": 45},
            {"trigger": "Surgical stress / anaesthesia", "pct": 38},
            {"trigger": "GI illness with reduced feeding", "pct": 35},
        ],
        "treatments": [
            {"drug": "Lipoic acid (Level B)", "level": "B", "response_pct": 35,
             "color": "#1565c0",
             "note": "Most directly relevant treatment in DLST deficiency; lipoamide on Lys114-L1 and Lys150-L2 is the primary deficient cofactor; partial-loss variants may respond with improved residual E2 function"},
            {"drug": "Riboflavin B2 (Level B)", "level": "B", "response_pct": 52,
             "color": "#1565c0",
             "note": "FAD cofactor for DLD (shared E3); DLD is intact in DLST deficiency; riboflavin optimises DLD function; safe co-administration with lipoic acid"},
            {"drug": "Thiamine B1 (Level C)", "level": "C", "response_pct": 22,
             "color": "#827717",
             "note": "TPP cofactor for OGDH E1 (not E2/DLST); trial reasonable (may support mass-action re-engagement of partially present DLST variants); lower response rate than in OGDH deficiency"},
            {"drug": "Succinate supplementation (Level C)", "level": "C", "response_pct": 18,
             "color": "#4a148c",
             "note": "Anaplerotic bypass: oral succinate → fumarate → malate → OAA; replenishes TCA intermediates downstream of DLST block; experimental"},
            {"drug": "LEV — Levetiracetam (Level B)", "level": "B", "response_pct": 65,
             "color": "#1b5e20",
             "note": "First-line AED; no αKGDH pathway interaction; safe in metabolic epilepsies"},
            {"drug": "Ketogenic diet (Level C)", "level": "C", "response_pct": 20,
             "color": "#827717",
             "note": "Experimental only; same limitation as OGDH deficiency — KD does NOT bypass αKGDH E2 block; not first-line unlike PDH deficiency"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "CAUTION",
             "mechanism": "Mitochondrial hepatotoxicity in any mitochondrial disease. CAUTION (not absolute CI unlike PDH deficiency where KD is first-line). Monitor LFTs."},
            {"drug": "High-protein diet", "risk": "HAZARD",
             "mechanism": "Amino acid catabolism → glutamate → α-KG surge → worsens αKGDH E2 block → 2-oxoglutaric aciduria aggravated."},
            {"drug": "Fasting / prolonged NPO", "risk": "HAZARD",
             "mechanism": "BCAA mobilisation → α-KG surge via transaminases; prefer oral succinic acid or glucose-electrolyte during illness."},
            {"drug": "Exercise (unmanaged)", "risk": "CAUTION",
             "mechanism": "TCA flux demand during exercise → α-KG surge → metabolic crisis in DLST deficiency."},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "Gene": "DLST (Dihydrolipoamide S-Succinyltransferase)",
            "Subunit": "E2 subunit of αKGDH complex (2-oxoglutarate dehydrogenase complex)",
            "Chromosome": "14q24.3",
            "Protein length": "~453 aa (mature; includes mitochondrial targeting sequence)",
            "Lipoyl domains": "Two: Lys114-L1 and Lys150-L2 (lipoamide covalently attached by LIAS)",
            "Catalytic domain": "Acetyltransferase / succinyltransferase domain — transfers succinyl from lipoamide to CoA",
            "Function": "Receives succinyl group from OGDH E1 (succinyl-TPP) via lipoamide swinging arm; transfers succinyl to CoA forming succinyl-CoA",
            "Complex partners": "OGDH (E1, decarboxylase, TPP-dependent) + DLD (E3, FAD homodimer, SHARED with PDH/BCKDH/GCS)",
            "Inheritance": "Autosomal Recessive (AR) — biallelic LOF",
            "OMIM gene": "*126063",
            "OMIM disease": "#203740 (2-Oxoglutaric Aciduria / αKGDH Complex Deficiency — same disease as OGDH deficiency)",
            "Prevalence": "~10–15 cases confirmed worldwide (2026) — rarer than OGDH (~20–40 cases)",
            "TCA step": "Step 4: α-KG → succinyl-CoA (E2 transfer step — BLOCKED in DLST deficiency)",
            "Primary biomarker": "Urine 2-oxoglutarate (GC-MS): 180–1800 µmol/mmol Cr (PATHOGNOMONIC)",
            "Pathognomonic assay pattern": "Isolated OGDH E1 activity NORMAL + αKGDH complex activity <10% → DLST (E2) deficiency",
            "Key negative biomarkers": ["BCAA NORMAL", "Glycine NORMAL", "DLD free activity NORMAL", "PDH activity NORMAL"],
            "Most directly relevant treatment": "Lipoic acid (Level B) — lipoamide on Lys114/Lys150 is the primary deficient cofactor",
            "Key distinguishing assay from OGDH": "Isolated OGDH E1 activity: NORMAL in DLST, SEVERELY REDUCED in OGDH",
        },
        "key_concepts": [
            {"term": "αKGDH complex E2 deficiency",
             "definition": "Loss of DLST (E2 succinyltransferase) abolishes the succinyl transfer step: OGDH E1 can form succinyl-TPP from α-KG, but cannot donate succinyl to a missing E2 lipoamide arm. The whole complex stalls; α-KG accumulates as 2-oxoglutaric acid."},
            {"term": "Isolated E1 activity vs complex activity — DLST fingerprint",
             "definition": "In DLST deficiency, isolated OGDH E1 activity (assayed without E2/E3) is NORMAL — E1 protein is intact and can decarboxylate α-KG using TPP. But αKGDH complex activity (assayed with full substrate α-KG and all subunits) is <10% — because E2 is absent and the transfer step fails. This E1-normal/complex-low pattern is the PATHOGNOMONIC diagnostic signature of DLST deficiency."},
            {"term": "Lipoamide swinging arm — Lys114-L1 and Lys150-L2",
             "definition": "DLST has two lipoyl domains: L1 (Lys114) and L2 (Lys150). Lipoamide (reduced lipoic acid) is covalently attached to these lysines by LIAS (lipoic acid synthase). The swinging arm receives the succinyl group from OGDH E1 succinyl-TPP and carries it to the catalytic acetyltransferase domain, which transfers succinyl to CoA. Loss of DLST = loss of this arm = stalled complex."},
            {"term": "DLST vs DLAT — structural analogy",
             "definition": "DLST (E2 of αKGDH) and DLAT (E2 of PDH complex) are structural analogues: both are acetyltransferases with lipoyl swinging arms, both form the inner cubic scaffold of their respective complex. However, DLST has SUCCINYL-transferase activity (succinyl-CoA) while DLAT has ACETYL-transferase activity (acetyl-CoA). They are separate genes (DLST: 14q24.3; DLAT: 11q23.1) and do NOT substitute for each other."},
            {"term": "2-oxoglutaric aciduria — identical in OGDH and DLST deficiency",
             "definition": "Both OGDH (E1 block) and DLST (E2 block) produce the same primary biomarker: urine 2-oxoglutarate elevation. The mechanism differs: OGDH E1 block prevents α-KG decarboxylation; DLST E2 block stalls the complex after succinyl-TPP formation, but net result is that α-KG cannot be turned over and accumulates. Gene panel is required to distinguish."},
            {"term": "DBT — BCKDH E2 subunit (NOT DLST)",
             "definition": "The E2 subunit of the branched-chain ketoacid dehydrogenase complex (BCKDH) is encoded by DBT (dihydrolipoamide branched-chain transacylase), chromosome 1p21.2. DBT is a separate gene from DLST (αKGDH E2). DLST deficiency does NOT affect BCKDH function → BCAA NORMAL."},
            {"term": "Lipoic acid treatment rationale in DLST",
             "definition": "Lipoic acid (converted to lipoamide by LIAS) is covalently attached to DLST at Lys114 (L1) and Lys150 (L2). This lipoamide is the primary catalytic cofactor of E2 — it carries the succinyl group. In DLST partial-loss variants, supplemental lipoic acid may increase lipoamide availability to stabilise residual DLST protein or augment residual activity. This is a STRONGER rationale than in OGDH deficiency (where E2/DLST is intact) and DLAT deficiency."},
            {"term": "Succinate supplementation",
             "definition": "Succinate (succinic acid) is the TCA intermediate downstream of succinyl-CoA (step 5: succinyl-CoA → succinate by succinyl-CoA synthetase). Oral succinate bypasses the αKGDH E2 block by providing succinic acid directly → fumarate → malate → OAA. Anaplerotic strategy; experimental in αKGDH deficiency; more logical in DLST/OGDH than in PDH deficiency where TCA entry (not internal step) is blocked."},
        ],
        "diagnostic_thresholds": {
            "urine_2_oxoglutarate_pathological": ">60 µmol/mmol Cr (suspicious) / >180 µmol/mmol Cr (diagnostic in DLST context)",
            "urine_2_oxoglutarate_dlst_range": "180–1800 µmol/mmol Cr (markedly elevated; GC-MS urine organic acids)",
            "isolated_ogdh_e1_activity_dlst": "NORMAL (>60% of normal in fibroblasts) — KEY PATHOGNOMONIC FINDING distinguishing DLST from OGDH",
            "akgdh_complex_activity_cutoff": "<10% of normal in fibroblasts → definitive αKGDH complex deficiency (OGDH or DLST or DLD)",
            "plasma_lactate_pathological": ">2.2 mmol/L; DLST range: 3–16 mmol/L",
            "plasma_pyruvate_dlst": "0.12–0.40 mmol/L (normal or mildly elevated; PDH intact → pyruvate near-normal)",
            "lp_ratio_dlst": "15–30 (mildly elevated; intermediate — same as OGDH; neither PDH-normal 10–20 nor Complex I >25)",
            "plasma_alanine_dlst": "180–500 µmol/L (mildly elevated; less than PDH deficiency)",
            "bcaa_dlst": "NORMAL — confirms BCKDH intact (DBT/E2 of BCKDH separate gene; not DLST)",
            "glycine_dlst": "NORMAL — confirms GCS intact (GCSH/AMT intact; DLD/E3 intact in DLST)",
            "dld_free_activity_dlst": "NORMAL — DLD protein is intact; only αKGDH complex is non-functional",
            "pdh_enzyme_activity_dlst": "NORMAL — PDH uses DLAT (E2), not DLST",
        },
        "differential_diagnosis": [
            {"disease": "OGDH deficiency (αKGDH E1)",
             "distinguishing_features": "OGDH deficiency: isolated OGDH E1 activity SEVERELY REDUCED (<10%) — E1 is the deficient protein. DLST deficiency: isolated OGDH E1 activity NORMAL. Both show urine 2-oxoglutarate elevated and αKGDH complex activity <10%. Gene panel (OGDH + DLST) mandatory."},
            {"disease": "DLD deficiency (αKGDH/PDH/BCKDH/GCS E3)",
             "distinguishing_features": "DLD: ALL four complexes blocked → BCAA elevated (BCKDH) + glycine elevated (GCS) + 2-hydroxyglutarate elevated + DLD free enzyme activity <10%. DLST: αKGDH only; BCAA normal, glycine normal, DLD free activity NORMAL."},
            {"disease": "DLAT deficiency (PDH E2)",
             "distinguishing_features": "DLAT: L:P ratio NORMAL (10–20); both lactate and pyruvate elevated; urine 2-oxoglutarate NORMAL; PDH activity <10%; DLST complex activity normal. DLST: urine 2-OG MARKEDLY elevated; PDH activity NORMAL; DLST complex activity <10%."},
            {"disease": "PDHA1 / PDHB / PDHX deficiency (PDH complex E1α, E1β, E3BP)",
             "distinguishing_features": "PDH complex deficiency: L:P ratio NORMAL (10–20); urine 2-oxoglutarate NORMAL; PDH activity severely reduced; αKGDH complex activity NORMAL. DLST: urine 2-OG markedly elevated; PDH activity NORMAL; αKGDH complex <10%."},
            {"disease": "Complex I (NADH:ubiquinone oxidoreductase) deficiency",
             "distinguishing_features": "Complex I: L:P ratio >25 (dramatically elevated, not just mildly); urine 2-OG may be mildly elevated but not pathognomonic; BG lesions similar but respiratory chain assay showing Complex I deficiency. DLST: L:P 15–30 (mildly elevated); 2-OG markedly elevated; RC enzymology normal."},
            {"disease": "SUCLA2 / SUCLG1 deficiency (succinyl-CoA ligase)",
             "distinguishing_features": "SUCLA2/SUCLG1: methylmalonic aciduria + mildly elevated plasma MMA; elevated C4-carnitine (methylmalonyl-CoA accumulation); urine 2-OG NORMAL. DLST: urine 2-OG markedly elevated; MMA normal; C4-carnitine normal."},
        ],
    }
