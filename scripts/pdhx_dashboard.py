#!/usr/bin/env python3
"""PDHX (PDH Complex Component X / E3-Binding Protein / E3BP) Epilepsy Dashboard.

PDHX encodes the E3-binding protein (E3BP), also called Protein X or Component X,
of the Pyruvate Dehydrogenase Complex (PDH complex / PDC):
  PDC structure: E1 (E1α₂E1β₂ tetramer, PDHA1+PDHB) + E2 (DLAT, 24-mer cubic core)
                 + E3BP/Protein X (PDHX) + E3 (DLD)
  PDH complex catalyses: Pyruvate + CoA + NAD⁺ → Acetyl-CoA + CO₂ + NADH (irreversible)
  PDHX (E3BP) role: LINKER PROTEIN — bridges the E2 (DLAT) cubic core and E3 (DLD);
                    E3BP is structurally similar to E2 but CANNOT perform acetyltransferase
                    activity; its sole function is to anchor E3 (DLD) to the E2 core;
                    E3BP contains its own lipoyl domain (Lys173) that is the substrate for
                    E3 (DLD) — E3 regenerates the E3BP lipoyl domain (oxidises NADH);
                    without functional PDHX, E3 (DLD) dissociates from the complex →
                    E3 cannot regenerate the lipoamide arms → PDH complex stalls.

PDHX LOF — BIOCHEMICAL PHENOTYPE IDENTICAL TO PDHA1/PDHB/DLAT:
  No E3BP → E3 (DLD) cannot anchor to PDH complex → lipoamide arms cannot be regenerated →
  PDH complex inactive → pyruvate accumulates → LDH converts pyruvate → lactate.
  CRITICAL: L:P ratio NORMAL (10–20) — both lactate AND pyruvate accumulate equally.
  NO BCAA elevation (unlike DLD which also blocks BCKDH).
  NO 2-hydroxyglutarate elevation (unlike DLD which also blocks αKGDH).
  The ONLY biochemical way to distinguish PDHX from PDHA1/PDHB/DLAT: gene panel.

PDHX UNIQUE DISTINGUISHING FEATURE:
  • DLD/E3 enzyme activity (free homodimer assay): NORMAL or slightly reduced.
    In DLD deficiency: severely reduced (<10% normal) — key negative for PDHX.
  • PDHX-E3BP has its OWN lipoyl domain (Lys173) — distinct from DLAT's two lipoyl
    domains (Lys173-L1, Lys259-L2). E3 works on ALL three lipoyl domains:
    DLAT-L1, DLAT-L2, and E3BP-lipoyl.
  • Unlike DLAT (structural E2 core collapse), PDHX loss is more targeted:
    E2 (DLAT) core remains structurally intact — only E3 anchorage is lost.
  • Larger-than-average gene deletions reported (~20% of PDHX cases) — structural
    genomic instability at 11p13 makes copy number variants more frequent.

AUTOSOMAL RECESSIVE — SAME AS PDHB AND DLAT (NO SEX BIAS):
  PDHX = 11p13; Autosomal Recessive (AR)
  Both males and females equally affected (no sex bias, unlike PDHA1 X-linked dominant)
  ~30–50 PDHX cases reported worldwide 2026 (ultra-rare; rarer than PDHA1, commoner than DLAT)
  No consanguinity-specific founder allele; all variants pan-ethnic, private-family
  Large deletions (~20% of alleles) due to 11p13 structural instability

CRITICAL BIOCHEMICAL FINGERPRINT (identical to PDHA1/PDHB/DLAT — PDH block fingerprint):
  • Lactate:Pyruvate (L:P) ratio NORMAL (~10–20) — BOTH rise together
  • CSF lactate typically > plasma lactate (brain preferentially affected)
  • CSF pyruvate elevated (>0.17 mmol/L)
  • Plasma alanine elevated (pyruvate transamination surrogate marker, 450–1400 µmol/L)
  • NO BCAA elevation (unlike DLD)
  • NO elevated 2-hydroxyglutarate (unlike DLD)
  • DLD/E3 free enzyme activity: NORMAL or slightly reduced (unlike DLD <10% normal)
  • PDHX enzyme activity (E3BP within complex): severely reduced or absent
  Gene panel (PDHA1 + PDHB + DLAT + PDHX) MANDATORY to assign correct component

STRUCTURAL BRAIN ANOMALIES (similar to PDHA1/PDHB/DLAT):
  1. Leigh syndrome — symmetric necrotising lesions basal ganglia + brainstem (~50%)
  2. Corpus callosum agenesis/dysgenesis (~40%) — less than PDHA1 (70%, sex bias absent)
  3. Ventriculomegaly + cortical atrophy (energy failure during neuronal migration)
  4. Periventricular white matter changes (energy failure during myelination)
  5. Cerebral cortical dysplasia (~15%, more common in severe neonatal)
  6. Brainstem hypoplasia / hypomyelination (posterior fossa energy failure)

PHENOTYPE CLASSES (PDHX, 2026):
  Leigh Syndrome (infantile BG/brainstem lesions): most common (~40%)
  Severe Neonatal (neonatal lactic acidosis + CC agenesis): ~25%
  Childhood Episodic (illness/exercise-triggered lactic acidosis): ~25%
  Mild Subacute/Juvenile (chronic partial enzyme deficiency): ~10%
  ~30–50 PDHX cases worldwide 2026; OMIM gene *608769; disease #245349

KEY VARIANTS:
  p.Arg445Cys (c.1333C>T): affects E3-binding C-terminal domain; partial E3BP function;
                            most commonly reported PDHX missense; moderate-severe phenotype
  p.Glu262Lys: lipoyl domain region; disrupts E3-lipoyl domain interaction; Leigh syndrome
  p.Arg302His: near lipoyl domain Lys173; reduces E3 regeneration of E3BP lipoyl arm; severe
  Large deletion (exons 7–9): gene-level deletion causing null E3BP; severe neonatal phenotype
  c.1150C>T (p.Arg384Cys): C-terminal E3-docking domain; moderate; episodic childhood
  c.547dupA (p.Ile183Asnfs*6): frameshift; null; neonatal/infantile Leigh

TREATMENT (identical indications to PDHA1/PDHB/DLAT — same PDH complex block):
  1. Ketogenic diet (Level A — FIRST LINE): same mechanism as PDHA1/PDHB/DLAT; ketones bypass PDH block;
     ketones → Acetyl-CoA via thiolase; direct substitute for blocked E3BP-dependent complex step
  2. Thiamine / B1 (Level A): TPP cofactor for E1α; trial all patients (100–600 mg/day);
     PDHX does not directly bind TPP, but thiamine may enhance residual E1 activity if
     partial E3BP scaffolding remains; response similar to PDHB/DLAT (~30–35%)
  3. L-Carnitine (Level B): secondary depletion; supports β-oxidation for ketone production on KD
  4. DCA (Dichloroacetate, Level B): inhibits PDK1/3 → prevents E1α Ser293 phosphorylation;
     only if any residual E3BP-mediated E3 anchoring remains (partial-loss variants); neuropathy risk
  5. LEV (Level B): first-line AED; no metabolic interaction with PDH pathway

HIGH-RISK DRUGS (identical to PDHA1/PDHB/DLAT):
  VPA: ABSOLUTE CI — mitochondrial hepatotoxicity + carnitine depletion (same as PDHA1/PDHB/DLAT)
  High-carbohydrate diet: EXTREME HAZARD (floods pyruvate; same as PDHA1/PDHB/DLAT)
  Glucose-only IV drip: HIGH RISK (pyruvate load without ketone buffer)
  Phenobarbital: CAUTION (enzyme inducer; may worsen hepatic mitochondrial function)

GENETICS SUMMARY:
  Gene: PDHX · 11p13 · 501 aa (including mitochondrial targeting sequence) · E3BP / Component X
  Autosomal Recessive (AR) · OMIM gene *608769 · disease #245349 (PDH complex deficiency)
  No common founder allele; all variants pan-ethnic; large deletions in ~20% of alleles
"""

import random

random.seed(43)  # reproducible cohort (different seed from DLAT=42)


def _patients():
    """Synthetic 40-patient PDHX-deficiency cohort."""
    phenotypes = [
        ("Leigh Syndrome (Infantile)", 16),
        ("Severe Neonatal", 10),
        ("Childhood Episodic", 10),
        ("Mild Subacute/Juvenile", 4),
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
                random.randint(1, 6)
                if is_neonatal
                else random.randint(3, 18)
                if is_leigh
                else random.randint(6, 36)
                if is_episodic
                else random.randint(12, 60)
            )

            # L:P ratio normal (key fingerprint — identical to PDHA1/PDHB/DLAT)
            pyruvate = round(random.uniform(0.3, 1.5), 2)
            lp_ratio = round(random.uniform(10, 20), 1)
            lactate = round(pyruvate * lp_ratio, 2)
            alanine = random.randint(450, 1400)

            pts.append({
                "id": f"PX{pid:03d}",
                "sex": random.choice(["M", "F"]),  # AR — equal sex distribution
                "phenotype": pheno,
                "onset_age_months": onset,
                "plasma_lactate_mmol": lactate,
                "plasma_pyruvate_mmol": pyruvate,
                "lp_ratio": lp_ratio,
                "plasma_alanine_umol": alanine,
                "cc_agenesis": random.random() < (0.55 if is_neonatal else 0.35 if is_leigh else 0.1),
                "leigh_lesions": random.random() < (0.90 if is_leigh else 0.30 if is_neonatal else 0.1),
                "on_kd": random.random() < (0.95 if is_leigh or is_neonatal else 0.80),
                "on_thiamine": random.random() < 0.95,
                "vpa_avoided": random.random() < 0.97,
                "dre": random.random() < (0.68 if is_leigh else 0.55 if is_neonatal else 0.22),
                "large_deletion": random.random() < 0.20,  # PDHX-specific: large deletions ~20%
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
    normal_lp = sum(1 for p in c if 10 <= p["lp_ratio"] <= 20)
    avg_ala = round(sum(p["plasma_alanine_umol"] for p in c) / n)
    cc_n   = sum(1 for p in c if p["cc_agenesis"])
    leigh_n = sum(1 for p in c if p["leigh_lesions"])
    kd_n   = sum(1 for p in c if p["on_kd"])
    dre_n  = sum(1 for p in c if p["dre"])
    vpa_ok = sum(1 for p in c if p["vpa_avoided"])
    del_n  = sum(1 for p in c if p["large_deletion"])

    return {
        "gene": "PDHX",
        "protein": "PDH Complex Component X / E3-Binding Protein (E3BP)",
        "locus": "11p13",
        "aa_length": 501,
        "cofactor": "Lipoic acid (covalently attached to Lys173 of E3BP lipoyl domain); E3BP has NO acetyltransferase activity",
        "mechanism": (
            "PDHX encodes the E3-binding protein (E3BP/Component X) — a structural adaptor in the PDH complex. "
            "E3BP is anchored within the E2 (DLAT) cubic core via its C-terminal domain, and recruits E3 (DLD) "
            "to the complex via its E3-docking domain. E3BP contains its own lipoyl domain (Lys173) that serves "
            "as a substrate for E3's lipoamide-regeneration reaction. PDHX LOF: E3 (DLD) cannot anchor to the "
            "PDH complex → lipoamide arms cannot be reoxidised → PDH complex stalls → pyruvate cannot enter "
            "TCA cycle → lactic acidosis. E2 (DLAT) core itself remains physically intact."
        ),
        "omim_gene": "*608769",
        "omim_disease": "#245349",
        "inheritance": "Autosomal Recessive (AR, 11p13) — both sexes equally affected; no sex bias",
        "key_distinguishing_from_pdha1": (
            "PDHX is AUTOSOMAL RECESSIVE (AR, 11p13) — no sex bias; males and females equally affected. "
            "PDHA1 is X-linked dominant (Xp22.12) — males severely affected, females variable. "
            "Biochemistry IDENTICAL: L:P ratio normal (10–20); gene panel PDHA1/PDHB/DLAT/PDHX MANDATORY."
        ),
        "key_distinguishing_feature": (
            "L:P ratio NORMAL (10–20): BOTH lactate AND pyruvate rise equally (unlike Complex I where L:P >25). "
            "NO BCAA elevation (unlike DLD/E3 deficiency). NO 2-hydroxyglutarate (unlike DLD). "
            "DLD/E3 free enzyme activity NORMAL or slightly reduced (unlike DLD deficiency where <10%). "
            "PDHX E3BP activity absent — E3 cannot anchor to PDH complex. Large genomic deletions in ~20%."
        ),
        "structural_brain_hallmark": (
            "Leigh syndrome (symmetric BG + brainstem lesions, ~50% in cohort); "
            "Corpus callosum agenesis/dysgenesis (~40%); Periventricular WM changes (~35%)."
        ),
        "kpis": {
            "cohort_n": n,
            "male_pct": round(100 * male_n / n),
            "avg_plasma_lactate_mmol": avg_lac,
            "avg_plasma_pyruvate_mmol": avg_pyr,
            "avg_lp_ratio": avg_lp,
            "normal_lp_ratio_10_20_pct": round(100 * normal_lp / n),
            "avg_plasma_alanine_umol": avg_ala,
            "cc_agenesis_pct": round(100 * cc_n / n),
            "leigh_lesions_pct": round(100 * leigh_n / n),
            "on_kd_pct": round(100 * kd_n / n),
            "dre_pct": round(100 * dre_n / n),
            "vpa_avoided_pct": round(100 * vpa_ok / n),
            "large_deletion_pct": round(100 * del_n / n),
        },
        "pdh_complex_components": [
            {"component": "E1 (PDHA1 + PDHB) — E1α₂β₂ tetramer",
             "function": "Pyruvate decarboxylation + reductive acetylation of E2 lipoyl arm (TPP cofactor)"},
            {"component": "E2 (DLAT) — 24-mer cubic core",
             "function": "Acetyl transfer from dihydrolipoamide to CoA; structural scaffold anchoring E1, E3BP, E3"},
            {"component": "E3BP (PDHX) — bridges E3 to E2 core ← THIS GENE",
             "function": "Anchors E3 (DLD) to E2 core; contains lipoyl domain Lys173 that E3 regenerates"},
            {"component": "E3 (DLD) — FAD homodimer",
             "function": "Regenerates dihydrolipoamide → lipoamide (oxidises NADH cofactor); shared with αKGDH, BCKDH, GCS"},
            {"component": "PDK1/2/3/4 — kinases",
             "function": "Phosphorylates E1α Ser293/300 → inactivates PDH complex (glucose-fed state)"},
            {"component": "PDP1/2 — phosphatases",
             "function": "Dephosphorylates E1α → activates PDH complex; Ca²⁺-activated"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
             "mechanism": "Mitochondrial hepatotoxicity + carnitine depletion → destroys KD therapy (same as PDHA1/PDHB/DLAT). NEVER in PDH complex deficiency."},
            {"drug": "High-carbohydrate diet / enteral feeds", "risk": "EXTREME HAZARD",
             "mechanism": "Floods pyruvate → overwhelms blocked PDH complex → acute lactic acidosis crisis. KD is treatment; high CHO is the opposite."},
            {"drug": "Glucose-only IV (without ketone source)", "risk": "HIGH RISK",
             "mechanism": "Pyruvate load without ketone buffer. Use balanced IV with lipid emulsion or early KD formula."},
            {"drug": "Phenobarbital (PB)", "risk": "CAUTION",
             "mechanism": "CYP enzyme inducer — may worsen hepatic mitochondrial function; not absolute CI but avoid if LEV available."},
        ],
        "structural_anomalies": [
            {"anomaly": "Leigh syndrome (BG + brainstem T2 lesions)", "frequency_pct": 50,
             "significance": "Symmetric necrotising lesions; most common PDHX MRI finding; putamen + periaqueductal grey most affected"},
            {"anomaly": "Corpus callosum agenesis/dysgenesis", "frequency_pct": 40,
             "significance": "Structural anomaly from energy failure during commissural axon development; no X-linked sex bias (AR)"},
            {"anomaly": "Periventricular white matter changes", "frequency_pct": 35,
             "significance": "Myelination energy failure; early MRI change in some cases"},
            {"anomaly": "Ventriculomegaly", "frequency_pct": 38,
             "significance": "Secondary cortical atrophy with neuronal energy failure; correlates with disease severity"},
            {"anomaly": "Cerebral cortical dysplasia", "frequency_pct": 15,
             "significance": "More common in severe neonatal PDHX; energy failure disrupts neuronal migration during embryogenesis"},
            {"anomaly": "Brainstem hypoplasia / hypomyelination", "frequency_pct": 22,
             "significance": "Posterior fossa energy failure; seen in neonatal and severe Leigh phenotype"},
        ],
        "phenotype_distribution": [
            {"label": "Leigh Syndrome (Infantile)", "pct": 40, "color": "#b71c1c"},
            {"label": "Severe Neonatal", "pct": 25, "color": "#d84315"},
            {"label": "Childhood Episodic", "pct": 25, "color": "#f57f17"},
            {"label": "Mild Subacute/Juvenile", "pct": 10, "color": "#558b2f"},
        ],
    }


def get_breakdown():
    c = _COHORT
    sample = c[:10]

    return {
        "biomarkers": [
            {"name": "Plasma Lactate", "normal": "<2.2 mmol/L",
             "pdhx_range": "3–18 mmol/L",
             "significance": "PDH block → pyruvate→lactate via LDH; primary biochemical indicator; identical to PDHA1/PDHB/DLAT"},
            {"name": "Plasma Pyruvate", "normal": "<0.17 mmol/L",
             "pdhx_range": "0.3–1.5 mmol/L",
             "significance": "Accumulates (E3 cannot regenerate lipoamide → E2 stalls → E1 product trapped); KEY: L:P ratio normal (both rise equally)"},
            {"name": "L:P ratio (Lactate:Pyruvate)", "normal": "10–20",
             "pdhx_range": "10–20 (NORMAL — KEY FINGERPRINT)",
             "significance": "Normal L:P distinguishes PDH deficiency (PDHX or PDHA1/PDHB/DLAT) from Complex I (L:P >25). GENE PANEL needed to distinguish"},
            {"name": "Plasma Alanine", "normal": "150–450 µmol/L",
             "pdhx_range": "450–1400 µmol/L",
             "significance": "Pyruvate transamination surrogate; sensitive chronic marker; identical in PDHX/PDHA1/PDHB/DLAT"},
            {"name": "CSF Lactate", "normal": "<2.1 mmol/L",
             "pdhx_range": "3–9 mmol/L (often > plasma)",
             "significance": "Brain preferentially affected by PDH block; CSF > plasma lactate typical"},
            {"name": "CSF Pyruvate", "normal": "<0.17 mmol/L",
             "pdhx_range": "0.3–1.0 mmol/L",
             "significance": "Elevated CSF pyruvate confirms PDH block at brain level; drawn simultaneously with plasma"},
            {"name": "Plasma BCAA (Leu/Ile/Val)", "normal": "Normal",
             "pdhx_range": "NORMAL — key negative",
             "significance": "PDHX does NOT block BCKDH (unlike DLD/E3). Normal BCAA distinguishes PDHX from DLD deficiency"},
            {"name": "2-Hydroxyglutarate", "normal": "Normal",
             "pdhx_range": "NORMAL — key negative",
             "significance": "PDHX does NOT block αKGDH (unlike DLD/E3). Normal 2-HG distinguishes PDHX from DLD deficiency"},
            {"name": "DLD/E3 free enzyme activity", "normal": "Normal",
             "pdhx_range": "NORMAL or slightly reduced (free homodimer intact)",
             "significance": "CRITICAL PDHX fingerprint: unlike DLD deficiency (<10% normal), E3 as a free homodimer is intact in PDHX deficiency — only complex-anchored E3 is missing"},
            {"name": "PDHX/E3BP complex activity", "normal": "Normal",
             "pdhx_range": "<10% normal (E3 cannot anchor to PDH complex)",
             "significance": "PDH complex activity severely reduced; E3BP-anchored E3 regeneration is absent; measured in cultured fibroblasts"},
        ],
        "key_variants": [
            {"variant": "p.Arg445Cys (c.1333C>T)", "effect": "E3-docking C-terminal domain; partial E3BP function; most common PDHX missense",
             "phenotype": "Moderate-severe / Childhood episodic"},
            {"variant": "p.Glu262Lys", "effect": "Lipoyl domain region; disrupts E3-mediated regeneration of E3BP lipoyl arm",
             "phenotype": "Leigh Syndrome (infantile)"},
            {"variant": "p.Arg302His", "effect": "Near lipoyl domain Lys173; reduces E3 reoxidation of E3BP lipoyl arm; partial function loss",
             "phenotype": "Leigh Syndrome (severe)"},
            {"variant": "c.547dupA (p.Ile183Asnfs*6)", "effect": "Frameshift; null E3BP; complete loss of E3 anchorage to PDH complex",
             "phenotype": "Severe Neonatal / infantile Leigh"},
            {"variant": "c.1150C>T (p.Arg384Cys)", "effect": "C-terminal E3-docking domain; reduces E3 binding affinity; moderate residual function",
             "phenotype": "Moderate / Childhood episodic"},
            {"variant": "Large deletion (exons 7–9)", "effect": "Genomic deletion; null E3BP; ~20% of PDHX alleles; 11p13 structural instability",
             "phenotype": "Severe Neonatal"},
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
                "plasma_alanine_umol": p["plasma_alanine_umol"],
                "cc_agenesis": p["cc_agenesis"],
                "leigh_lesions": p["leigh_lesions"],
                "on_kd": p["on_kd"],
                "on_thiamine": p["on_thiamine"],
                "vpa_avoided": p["vpa_avoided"],
                "large_deletion": p["large_deletion"],
            }
            for p in sample
        ],
        "seizure_types": [
            {"type": "Focal seizures with secondary generalisation", "pct": 58},
            {"type": "Tonic / tonic-clonic (GTC)", "pct": 50},
            {"type": "Infantile spasms (IS / West syndrome)", "pct": 38},
            {"type": "Myoclonic seizures", "pct": 32},
            {"type": "Absence-like / staring", "pct": 25},
            {"type": "Status epilepticus (SE)", "pct": 18},
        ],
        "trigger_types": [
            {"trigger": "Febrile illness / infection", "pct": 80},
            {"trigger": "Glucose/carbohydrate load", "pct": 72},
            {"trigger": "Fasting / prolonged NPO", "pct": 52},
            {"trigger": "Surgery / anaesthesia (glucose-heavy IV)", "pct": 55},
            {"trigger": "Exercise-induced lactic acidosis", "pct": 50},
            {"trigger": "Intercurrent GI illness (poor intake)", "pct": 38},
        ],
        "treatments": [
            {"drug": "Ketogenic Diet (KD)", "level": "A", "response_pct": 80, "color": "#1b5e20"},
            {"drug": "Thiamine (B1, 100–600 mg/day)", "level": "A", "response_pct": 32, "color": "#1565c0"},
            {"drug": "L-Carnitine (secondary depletion support)", "level": "B", "response_pct": 70, "color": "#4a148c"},
            {"drug": "DCA (Dichloroacetate) — partial-loss only", "level": "B", "response_pct": 25, "color": "#827717"},
            {"drug": "LEV (Levetiracetam, AED)", "level": "B", "response_pct": 70, "color": "#006064"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
             "mechanism": "Mitochondrial hepatotoxicity + carnitine depletion → destroys KD therapy. NEVER in PDH complex deficiency."},
            {"drug": "High-carbohydrate diet / enteral feeds", "risk": "EXTREME HAZARD",
             "mechanism": "Floods pyruvate → overwhelms blocked PDHX E3BP step → acute lactic acidosis. KD is the treatment; high CHO is the hazard."},
            {"drug": "Glucose-only IV drip", "risk": "HIGH RISK",
             "mechanism": "Pyruvate load without ketone buffer; causes acute metabolic decompensation in PDHX deficiency."},
            {"drug": "Phenobarbital", "risk": "CAUTION",
             "mechanism": "CYP enzyme inducer; may worsen hepatic mitochondrial function; avoid if LEV available."},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "Gene": "PDHX",
            "Full name": "PDH Complex Component X / E3-Binding Protein (E3BP)",
            "Locus": "11p13",
            "Protein length": "501 aa (including mitochondrial targeting sequence, MTS)",
            "PDH complex role": "E3BP — structural linker anchoring E3 (DLD) to E2 (DLAT) core; contains lipoyl domain Lys173; NO acetyltransferase activity",
            "Lipoyl attachment site": "Lys173 (single lipoyl domain on E3BP) — covalent attachment by LIAS; different from DLAT's Lys173-L1 and Lys259-L2",
            "Reaction catalysed": "None — E3BP is NOT an enzyme; it is a structural adaptor/linker protein",
            "OMIM gene": "*608769",
            "OMIM disease": "#245349 (PDH complex deficiency, E3BP/PDHX type)",
            "Inheritance": "Autosomal Recessive (AR) — both sexes equally affected",
            "Cases worldwide (2026)": "~30–50 confirmed PDHX cases (ultra-rare; rarer than PDHA1/PDHB, commoner than DLAT)",
            "Key negative biomarkers": "Normal BCAA, normal 2-HG; DLD free enzyme activity NORMAL or slightly reduced (unlike DLD deficiency)",
            "Unique feature": "Large genomic deletions in ~20% of PDHX alleles (11p13 structural instability); gene panel + CNV analysis recommended",
        },
        "key_concepts": [
            {
                "term": "E3BP — structural linker, not an enzyme",
                "definition": (
                    "PDHX encodes the E3-binding protein (E3BP), which has NO catalytic activity. "
                    "Unlike E1α (PDHA1), E1β (PDHB), E2 (DLAT), or E3 (DLD), E3BP does not catalyse "
                    "any chemical reaction. Its sole function is structural: it anchors within the E2 "
                    "(DLAT) cubic core via its C-terminal swinging arm domain, and presents its E3-docking "
                    "domain to recruit E3 (DLD) to the PDH complex. Without E3BP, E3 is free in the "
                    "mitochondrial matrix but cannot access the PDH complex → lipoamide arms remain "
                    "reduced → entire PDH complex stalls."
                ),
            },
            {
                "term": "E3BP lipoyl domain — E3's substrate within PDH complex",
                "definition": (
                    "E3BP (PDHX) has its own single lipoyl domain at Lys173 (distinct from DLAT's "
                    "L1-Lys173 and L2-Lys259). This E3BP lipoyl domain is specifically the substrate "
                    "for E3 (DLD) within the PDH complex — E3 oxidises the reduced E3BP lipoamide arm "
                    "(producing NADH) to regenerate it for the next catalytic cycle. PDHX mutations that "
                    "disrupt this lipoyl domain impair E3's ability to complete its reoxidation reaction "
                    "within the PDH complex, even if E3 (DLD) as a free protein is intact."
                ),
            },
            {
                "term": "CRITICAL DLD/E3 enzyme activity distinction: PDHX vs DLD",
                "definition": (
                    "In DLD deficiency: DLD/E3 free homodimer enzyme activity is severely reduced (<10% "
                    "normal) — the DLD protein itself is non-functional. "
                    "In PDHX deficiency: DLD/E3 free homodimer enzyme activity is NORMAL or only slightly "
                    "reduced — DLD protein is intact but cannot anchor to the PDH complex without E3BP. "
                    "This distinction is CRITICAL for biochemical diagnosis. If free DLD activity is "
                    "severely reduced → DLD deficiency. If free DLD activity is normal → PDHX or other "
                    "PDH complex component (requires gene panel PDHA1/PDHB/DLAT/PDHX)."
                ),
            },
            {
                "term": "Biochemically IDENTICAL to PDHA1, PDHB, and DLAT",
                "definition": (
                    "PDHX deficiency produces the same PDH block fingerprint: "
                    "L:P ratio NORMAL (10–20); both lactate AND pyruvate accumulate; elevated plasma "
                    "alanine; CSF lactate > plasma lactate. The ONLY biochemical distinction from DLD: "
                    "normal BCAA, normal 2-HG, normal/near-normal free DLD enzyme activity. "
                    "Gene panel PDHA1/PDHB/DLAT/PDHX is MANDATORY — biochemistry alone cannot "
                    "distinguish the four PDH complex component deficiencies."
                ),
            },
            {
                "term": "L:P ratio NORMAL — PDH block fingerprint",
                "definition": (
                    "In PDH complex deficiency (PDHA1, PDHB, DLAT, PDHX), BOTH lactate AND pyruvate "
                    "accumulate proportionally → L:P ratio remains NORMAL (10–20). This is the key "
                    "distinguisher from Complex I deficiency, where primarily NADH accumulates → "
                    "LDH pushes pyruvate→lactate disproportionately → L:P ratio >25."
                ),
            },
            {
                "term": "Ketogenic diet mechanism (E3BP bypass)",
                "definition": (
                    "KD provides ketone bodies (β-hydroxybutyrate, acetoacetate) from hepatic fatty acid "
                    "β-oxidation. Ketones cross the BBB and enter neurons, where succinyl-CoA:3-oxoacid-CoA "
                    "transferase (SCOT) and thiolase convert them to Acetyl-CoA — BYPASSING the blocked "
                    "PDH complex entirely. This is why KD is FIRST-LINE Level A therapy for ALL PDH complex "
                    "subunit deficiencies (PDHA1, PDHB, DLAT, PDHX)."
                ),
            },
            {
                "term": "Large genomic deletions — PDHX-specific risk",
                "definition": (
                    "Unlike PDHA1, PDHB, or DLAT, PDHX at chromosome 11p13 has higher susceptibility to "
                    "genomic structural variants. Approximately 20% of pathogenic PDHX alleles carry large "
                    "deletions (partial or complete gene deletions, exon 7–9 being most common) rather than "
                    "point mutations. This means standard sequencing may miss ~20% of alleles — "
                    "CNV (copy number variant) analysis or MLPA (multiplex ligation-dependent probe "
                    "amplification) is RECOMMENDED alongside gene sequencing for PDHX diagnosis."
                ),
            },
            {
                "term": "AR inheritance — no sex bias",
                "definition": (
                    "PDHX (11p13) is autosomal recessive — biallelic LOF required. Both sexes equally "
                    "affected (no X-inactivation variability as in PDHA1). No haploinsufficiency in "
                    "obligate carrier parents. No consanguinity-specific founder allele — all variants "
                    "pan-ethnic and private-family."
                ),
            },
            {
                "term": "Thiamine (B1) — less responsive than PDHA1",
                "definition": (
                    "Thiamine provides TPP cofactor for E1α (PDHA1) — directly enhances E1 activity. "
                    "In PDHX deficiency, the E3BP block is at the E3-anchoring step (downstream of E1 "
                    "and E2). Thiamine trial is still MANDATORY (Level A) in all PDH complex deficiencies "
                    "as response cannot be predicted without trial, but expect ~30–35% partial response "
                    "(less than PDHA1 ~55%; similar to PDHB ~35% and DLAT ~30–40%)."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease": "PDHA1 deficiency (E1α)",
                "distinguishing_features": (
                    "X-linked dominant (males severe, females variable); CC agenesis 70%; L:P normal; "
                    "BCAA normal; 2-HG normal; DLD normal. Gene panel: PDHA1 mutation."
                ),
            },
            {
                "disease": "PDHB deficiency (E1β)",
                "distinguishing_features": (
                    "AR (11p13 — same chromosome arm!); Leigh 65%; L:P normal; BCAA normal; 2-HG normal; "
                    "DLD normal. Gene panel: PDHB mutation. Note: PDHB at 3p14.3 — different chromosome from PDHX."
                ),
            },
            {
                "disease": "DLAT deficiency (E2)",
                "distinguishing_features": (
                    "AR (11q23.1); L:P normal; BCAA normal; 2-HG normal; DLD/E3 activity NORMAL. "
                    "DLAT E2 enzyme activity <10% normal. Gene panel: DLAT mutation."
                ),
            },
            {
                "disease": "DLD deficiency (E3)",
                "distinguishing_features": (
                    "AR (7q31.1); L:P normal OR mildly elevated; BCAA elevated (BCKDH block); "
                    "2-HG elevated (αKGDH block); glycine mildly elevated (GCS partial block); "
                    "DLD/E3 free enzyme activity severely reduced (<10% normal) — KEY DISTINCTION from PDHX."
                ),
            },
            {
                "disease": "Complex I deficiency (NADH:ubiquinone oxidoreductase)",
                "distinguishing_features": (
                    "L:P ratio >25 (NADH accumulation → disproportionate lactate rise — KEY DIFFERENTIATOR). "
                    "Multiple OXPHOS gene mutations (MT-ND1-6, NDUFS1-8, NDUFV1-2 etc.). "
                    "No pyruvate accumulation proportional to lactate."
                ),
            },
            {
                "disease": "Pyruvate carboxylase (PC) deficiency",
                "distinguishing_features": (
                    "L:P ratio >25 (NADH elevated); elevated citruline, lysine, proline; "
                    "CSF lactate similar to plasma; no CC agenesis; neonatal form: fatal."
                ),
            },
        ],
        "diagnostic_thresholds": {
            "plasma_lactate_mmol_threshold": ">2.2 mmol/L (normal <2.2)",
            "plasma_pyruvate_mmol_threshold": ">0.17 mmol/L (normal <0.17)",
            "lp_ratio_pdh_range": "10–20 (NORMAL — PDH block fingerprint; Complex I: >25)",
            "plasma_alanine_umol_threshold": ">450 µmol/L (normal 150–450; PDHX range 450–1400)",
            "csf_lactate_threshold": ">2.1 mmol/L; typically greater than plasma in PDHX",
            "bcaa_key_negative": "NORMAL — key negative vs DLD deficiency",
            "2hg_key_negative": "NORMAL — key negative vs DLD deficiency",
            "dld_free_enzyme_pdhx": "NORMAL or slightly reduced (unlike DLD deficiency <10% normal)",
            "pdhx_e3bp_complex_activity": "<10% normal (definitive — within PDH complex, fibroblasts)",
            "cnv_analysis": "RECOMMENDED alongside sequencing — large deletions in ~20% of PDHX alleles",
        },
    }
