#!/usr/bin/env python3
"""DLAT (Dihydrolipoamide S-Acetyltransferase / E2 Subunit) Epilepsy Dashboard.

DLAT encodes the E2 subunit (dihydrolipoamide acetyltransferase) of the Pyruvate
Dehydrogenase Complex (PDH complex / PDC):
  PDC structure: E1 (E1α₂E1β₂ tetramer, PDHA1+PDHB) + E2 (DLAT, 24-mer cubic core)
                 + E3BP/E3-binding protein (PDHX) + E3 (DLD)
  PDH complex catalyses: Pyruvate + CoA + NAD⁺ → Acetyl-CoA + CO₂ + NADH (irreversible)
  DLAT (E2) role: STRUCTURAL CORE — forms the 24-subunit cubic inner scaffold of the PDH complex;
                   carries two N-terminal lipoyl domains (L1 at Lys173, L2 at Lys259) that serve as
                   "swinging arms" shuttling the acetyl group from E1 to CoA;
                   E2 also anchors E1, E3, and E3BP (PDHX) — without functional DLAT, the entire
                   PDH complex falls apart structurally.

DLAT LOF — BIOCHEMICAL PHENOTYPE IDENTICAL TO PDHA1 AND PDHB:
  No E2 → E1 cannot transfer acetyl group to CoA → PDH block → pyruvate accumulates →
  LDH converts pyruvate → lactate (lactic acidosis).
  CRITICAL: L:P ratio NORMAL (10–20) — both lactate AND pyruvate accumulate equally.
  NO BCAA elevation (unlike DLD which also blocks BCKDH).
  NO 2-hydroxyglutarate elevation (unlike DLD which also blocks αKGDH).
  The ONLY biochemical way to distinguish DLAT from PDHA1/PDHB/PDHX: gene panel.

DLAT STRUCTURAL UNIQUENESS:
  • 24-mer cubic E2 core: 24 DLAT monomers self-assemble into cubic scaffold (~5 MDa with E3BP)
  • 2 lipoyl domains per DLAT monomer (Lys173, Lys259) — lipoic acid covalently attached by LIAS
  • DLAT LOF destroys the core: E1 cannot dock → E3 cannot anchor via E3BP → complete complex collapse
  • Unlike PDHA1 (E1α) or PDHB (E1β), DLAT loss is more structurally catastrophic — destroys the
    entire scaffolding architecture, not just one enzymatic step
  • E3BP (PDHX) binds directly to DLAT's C-terminal domain — PDHX cannot assemble if DLAT absent

AUTOSOMAL RECESSIVE — SAME AS PDHB (NO SEX BIAS):
  DLAT = 11q23.1; Autosomal Recessive (AR)
  Both males and females equally affected (no sex bias, unlike PDHA1 X-linked dominant)
  ~10–25 DLAT cases reported worldwide 2026 (ultra-rare; rarer than PDHA1 and PDHB)
  No consanguinity-specific founder allele; all variants pan-ethnic, private-family

CRITICAL BIOCHEMICAL FINGERPRINT (identical to PDHA1/PDHB — PDH block fingerprint):
  • Lactate:Pyruvate (L:P) ratio NORMAL (~10–20) — BOTH rise together
  • CSF lactate typically > plasma lactate (brain preferentially affected)
  • CSF pyruvate elevated (>0.17 mmol/L)
  • Plasma alanine elevated (pyruvate transamination surrogate marker, 450–1300 µmol/L)
  • NO BCAA elevation (unlike DLD)
  • NO elevated 2-hydroxyglutarate (unlike DLD)
  • Dihydrolipoamide dehydrogenase assay (DLD/E3 activity) NORMAL — key negative
  • DLAT enzyme activity in fibroblasts: severely reduced (<10% normal)
  Gene panel (PDHA1 + PDHB + DLAT + PDHX) MANDATORY to assign correct component

STRUCTURAL BRAIN ANOMALIES (similar to PDHA1/PDHB):
  1. Leigh syndrome — symmetric necrotising lesions basal ganglia + brainstem (~55%)
  2. Corpus callosum agenesis/dysgenesis (~35%)
  3. Ventriculomegaly + cortical atrophy (energy failure during neuronal migration)
  4. Periventricular white matter changes (energy failure during myelination)
  5. Cerebral cortical dysplasia (~15%, more common in severe neonatal)
  Note: DLAT has similar spectrum to PDHB but potentially more structurally severe
  due to complete core collapse (E1 AND E3 lose anchorage simultaneously)

PHENOTYPE CLASSES (DLAT, 2026):
  Leigh Syndrome (infantile BG/brainstem lesions): most common (~45%)
  Severe Neonatal (neonatal lactic acidosis + CC agenesis): ~25%
  Childhood Episodic (illness/exercise-triggered lactic acidosis): ~20%
  Mild Subacute/Juvenile (chronic partial enzyme deficiency): ~10%
  ~10–25 DLAT cases worldwide 2026 (ultra-rare); OMIM gene *608770; disease #245348 (PDH deficiency)

KEY VARIANTS:
  p.Arg272* (c.814C>T): null variant; complete E2 loss; severe neonatal phenotype; early death
  p.His382Arg: affects lipoyl domain L2 region; reduces lipoic acid attachment at Lys259;
               partial E2 function; infantile Leigh syndrome; moderate severity
  p.Lys173Arg: direct mutation of Lys173 lipoyl attachment site;
               lipoic acid cannot attach to L1 domain; severely reduced E2 activity; Leigh syndrome
  p.Gly365Asp: acetyltransferase catalytic domain; reduces acetyl transfer from dihydrolipoamide to CoA;
               moderate infantile phenotype; episodic decompensation
  c.1124-2A>G: canonical splice acceptor; exon 10 skipping; frameshift in E2 catalytic domain;
               neonatal/infantile Leigh syndrome
  Large deletions (partial/complete DLAT): complete loss of E2; severe neonatal phenotype

TREATMENT (identical indications to PDHA1/PDHB — same PDH complex block):
  1. Ketogenic diet (Level A — FIRST LINE): same mechanism as PDHA1/PDHB; ketones bypass PDH block;
     ketones → Acetyl-CoA via thiolase; direct substitute for blocked E2 product
  2. Thiamine / B1 (Level A): TPP cofactor for E1α; trial all patients (100–600 mg/day);
     DLAT does not directly bind TPP, but thiamine may enhance residual E1 activity
     if any residual E2 scaffolding remains; response similar to PDHB (~30–40%)
  3. Lipoic acid supplementation (Level C, experimental): substrate for lipoylation of E2 domains;
     may partially restore lipoyl arm function in hypomorphic variants; no proven clinical benefit
  4. L-Carnitine (Level B): secondary depletion; supports β-oxidation for ketone production on KD
  5. DCA (Dichloroacetate, Level B): inhibits PDK1/3 → prevents E1α Ser293 phosphorylation;
     only if any residual E2 activity remains (partial-loss variants); neuropathy risk
  6. LEV (Level B): first-line AED; no metabolic interaction with PDH pathway

HIGH-RISK DRUGS (identical to PDHA1/PDHB):
  VPA: ABSOLUTE CI — mitochondrial hepatotoxicity + carnitine depletion (same as PDHA1/PDHB)
  High-carbohydrate diet: EXTREME HAZARD (floods pyruvate; same as PDHA1/PDHB)
  Glucose-only IV drip: HIGH RISK (pyruvate load without ketone buffer)
  Phenobarbital: CAUTION (enzyme inducer; may worsen hepatic mitochondrial function)

GENETICS SUMMARY:
  Gene: DLAT · 11q23.1 · 729 aa (including mitochondrial targeting sequence) · E2 subunit
  Autosomal Recessive (AR) · OMIM gene *608770 · disease #245348 (PDH complex deficiency)
  No common founder allele; all variants pan-ethnic
"""

import random

random.seed(42)  # reproducible cohort


def _patients():
    """Synthetic 40-patient DLAT-deficiency cohort."""
    phenotypes = [
        ("Leigh Syndrome (Infantile)", 18),
        ("Severe Neonatal", 10),
        ("Childhood Episodic", 8),
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

            # L:P ratio normal (key fingerprint — identical to PDHA1/PDHB)
            pyruvate = round(random.uniform(0.3, 1.5), 2)
            lp_ratio = round(random.uniform(10, 20), 1)
            lactate = round(pyruvate * lp_ratio, 2)
            alanine = random.randint(450, 1300)

            pts.append({
                "id": f"DL{pid:03d}",
                "sex": random.choice(["M", "F"]),  # AR — equal sex distribution
                "phenotype": pheno,
                "onset_age_months": onset,
                "plasma_lactate_mmol": lactate,
                "plasma_pyruvate_mmol": pyruvate,
                "lp_ratio": lp_ratio,
                "plasma_alanine_umol": alanine,
                "cc_agenesis": random.random() < (0.55 if is_neonatal else 0.30 if is_leigh else 0.1),
                "leigh_lesions": random.random() < (0.90 if is_leigh else 0.35 if is_neonatal else 0.1),
                "on_kd": random.random() < (0.95 if is_leigh or is_neonatal else 0.80),
                "on_thiamine": random.random() < 0.95,
                "vpa_avoided": random.random() < 0.97,
                "dre": random.random() < (0.72 if is_leigh else 0.55 if is_neonatal else 0.25),
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

    return {
        "gene": "DLAT",
        "protein": "Dihydrolipoamide S-Acetyltransferase (E2 subunit of PDH complex)",
        "locus": "11q23.1",
        "aa_length": 729,
        "cofactor": "Lipoic acid (covalently attached to Lys173 [L1] and Lys259 [L2]); CoA (substrate for acetyl transfer)",
        "mechanism": (
            "DLAT forms the 24-subunit cubic inner scaffold (E2 core) of the PDH complex. "
            "Two lipoyl domains (Lys173, Lys259) serve as 'swinging arms' that pick up the acetyl "
            "group from E1 and transfer it to CoA, producing Acetyl-CoA. DLAT also anchors E3 via "
            "E3BP (PDHX). DLAT LOF: E2 core collapses → E1 cannot dock → E3 cannot anchor → entire "
            "PDH complex inactive → pyruvate cannot enter TCA → lactic acidosis."
        ),
        "omim_gene": "*608770",
        "omim_disease": "#245348",
        "inheritance": "Autosomal Recessive (AR, 11q23.1) — both sexes equally affected; no sex bias",
        "key_distinguishing_from_pdha1": (
            "DLAT is AUTOSOMAL RECESSIVE (AR, 11q23.1) — no sex bias; males and females equally affected. "
            "PDHA1 is X-linked dominant (Xp22.12) — males severely affected, females variable. "
            "Biochemistry IDENTICAL: L:P ratio normal (10–20); gene panel PDHA1/PDHB/DLAT/PDHX MANDATORY."
        ),
        "key_distinguishing_feature": (
            "L:P ratio NORMAL (10–20): BOTH lactate AND pyruvate rise equally (unlike Complex I where L:P >25). "
            "NO BCAA elevation (unlike DLD/E3 deficiency). NO 2-hydroxyglutarate (unlike DLD). "
            "DLD/E3 enzyme activity NORMAL in fibroblasts (unlike DLD deficiency). "
            "DLAT enzyme activity severely reduced (<10% normal) — E2 assay in fibroblasts is diagnostic."
        ),
        "structural_brain_hallmark": (
            "Leigh syndrome (symmetric BG + brainstem lesions, ~55% in cohort); "
            "Corpus callosum agenesis/dysgenesis (~35%); Periventricular WM changes (~38%)."
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
        },
        "pdh_complex_components": [
            {"component": "E1 (PDHA1 + PDHB) — E1α₂β₂ tetramer",
             "function": "Pyruvate decarboxylation + reductive acetylation of E2 lipoyl arm (TPP cofactor)"},
            {"component": "E2 (DLAT) — 24-mer cubic core ← THIS GENE",
             "function": "Acetyl transfer from dihydrolipoamide to CoA; structural scaffold anchoring E1, E3BP, E3"},
            {"component": "E3BP (PDHX) — bridges E3 to E2 core",
             "function": "Anchors E3 (DLD) to E2 core; contains lipoyl domain that E3 regenerates"},
            {"component": "E3 (DLD) — FAD homodimer",
             "function": "Regenerates dihydrolipoamide → lipoamide (oxidises NADH cofactor); shared with αKGDH, BCKDH, GCS"},
            {"component": "PDK1/2/3/4 — kinases",
             "function": "Phosphorylates E1α Ser293/300 → inactivates PDH complex (glucose-fed state)"},
            {"component": "PDP1/2 — phosphatases",
             "function": "Dephosphorylates E1α → activates PDH complex; Ca²⁺-activated"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
             "mechanism": "Mitochondrial hepatotoxicity + carnitine depletion → destroys KD therapy (same as PDHA1/PDHB). NEVER in PDH complex deficiency."},
            {"drug": "High-carbohydrate diet / enteral feeds", "risk": "EXTREME HAZARD",
             "mechanism": "Floods pyruvate → overwhelms blocked PDH complex → acute lactic acidosis crisis. KD is treatment; high CHO is the opposite."},
            {"drug": "Glucose-only IV (without ketone source)", "risk": "HIGH RISK",
             "mechanism": "Pyruvate load without ketone buffer. Use balanced IV with lipid emulsion or early KD formula."},
            {"drug": "Phenobarbital (PB)", "risk": "CAUTION",
             "mechanism": "CYP enzyme inducer — may worsen hepatic mitochondrial function; not absolute CI but avoid if LEV available."},
        ],
        "structural_anomalies": [
            {"anomaly": "Leigh syndrome (BG + brainstem T2 lesions)", "frequency_pct": 55,
             "significance": "Symmetric necrotising lesions; most common DLAT MRI finding; putamen + periaqueductal grey most affected"},
            {"anomaly": "Corpus callosum agenesis/dysgenesis", "frequency_pct": 35,
             "significance": "Structural anomaly from energy failure during commissural axon development; less common than PDHA1 (70%) — no X-linked sex bias"},
            {"anomaly": "Periventricular white matter changes", "frequency_pct": 38,
             "significance": "Myelination energy failure; earliest MRI change in some cases"},
            {"anomaly": "Ventriculomegaly", "frequency_pct": 40,
             "significance": "Secondary cortical atrophy with neuronal energy failure; correlates with disease severity"},
            {"anomaly": "Cerebral cortical dysplasia", "frequency_pct": 15,
             "significance": "More common in severe neonatal DLAT; energy failure disrupts neuronal migration during embryogenesis"},
            {"anomaly": "Brainstem hypoplasia", "frequency_pct": 20,
             "significance": "Energy failure in posterior fossa development; seen in neonatal severe phenotype"},
        ],
        "phenotype_distribution": [
            {"label": "Leigh Syndrome (Infantile)", "pct": 45, "color": "#b71c1c"},
            {"label": "Severe Neonatal", "pct": 25, "color": "#d84315"},
            {"label": "Childhood Episodic", "pct": 20, "color": "#f57f17"},
            {"label": "Mild Subacute/Juvenile", "pct": 10, "color": "#558b2f"},
        ],
    }


def get_breakdown():
    c = _COHORT
    sample = c[:10]

    return {
        "biomarkers": [
            {"name": "Plasma Lactate", "normal": "<2.2 mmol/L",
             "dlat_range": "3–18 mmol/L",
             "significance": "PDH block → pyruvate→lactate via LDH; primary biochemical indicator; identical to PDHA1/PDHB"},
            {"name": "Plasma Pyruvate", "normal": "<0.17 mmol/L",
             "dlat_range": "0.3–1.5 mmol/L",
             "significance": "Accumulates (E2 block → E1 product cannot be transferred to CoA); KEY: L:P ratio normal (both rise equally)"},
            {"name": "L:P ratio (Lactate:Pyruvate)", "normal": "10–20",
             "dlat_range": "10–20 (NORMAL — KEY FINGERPRINT)",
             "significance": "Normal L:P distinguishes PDH deficiency (DLAT or PDHA1/PDHB) from Complex I (L:P >25). GENE PANEL needed to distinguish DLAT vs PDHA1 vs PDHB vs PDHX"},
            {"name": "Plasma Alanine", "normal": "150–450 µmol/L",
             "dlat_range": "450–1300 µmol/L",
             "significance": "Pyruvate transamination surrogate; sensitive chronic marker; identical in DLAT/PDHA1/PDHB"},
            {"name": "CSF Lactate", "normal": "<2.1 mmol/L",
             "dlat_range": "3–9 mmol/L (often > plasma)",
             "significance": "Brain preferentially affected by PDH block; CSF > plasma lactate typical"},
            {"name": "CSF Pyruvate", "normal": "<0.17 mmol/L",
             "dlat_range": "0.3–1.0 mmol/L",
             "significance": "Elevated CSF pyruvate confirms PDH block at brain level; drawn simultaneously with plasma"},
            {"name": "Plasma BCAA (Leu/Ile/Val)", "normal": "Normal",
             "dlat_range": "NORMAL — key negative",
             "significance": "DLAT does NOT block BCKDH (unlike DLD/E3). Normal BCAA distinguishes DLAT from DLD deficiency"},
            {"name": "2-Hydroxyglutarate", "normal": "Normal",
             "dlat_range": "NORMAL — key negative",
             "significance": "DLAT does NOT block αKGDH (unlike DLD/E3). Normal 2-HG distinguishes DLAT from DLD deficiency"},
            {"name": "DLD/E3 enzyme activity", "normal": "Normal",
             "dlat_range": "NORMAL — key negative (E3 activity intact)",
             "significance": "E3 (DLD) is separate from E2 (DLAT). Normal E3 activity distinguishes DLAT from DLD deficiency"},
            {"name": "DLAT enzyme activity (fibroblasts)", "normal": "Normal",
             "dlat_range": "<10% normal (severely reduced)",
             "significance": "Definitive biochemical diagnosis; E2 acetyltransferase activity severely reduced or absent"},
        ],
        "key_variants": [
            {"variant": "p.Arg272* (c.814C>T)", "effect": "Null; complete E2 core loss; E1 cannot dock; E3 cannot anchor via E3BP",
             "phenotype": "Severe Neonatal / early Leigh"},
            {"variant": "p.His382Arg", "effect": "Lipoyl domain L2 region; reduces lipoic acid attachment at Lys259; partial E2 function",
             "phenotype": "Infantile Leigh Syndrome"},
            {"variant": "p.Lys173Arg", "effect": "Direct mutation of L1 lipoyl attachment site Lys173; lipoic acid cannot attach to L1 domain",
             "phenotype": "Leigh Syndrome (severe)"},
            {"variant": "p.Gly365Asp", "effect": "Acetyltransferase catalytic domain; reduces acetyl transfer from dihydrolipoamide to CoA",
             "phenotype": "Moderate infantile / episodic"},
            {"variant": "c.1124-2A>G", "effect": "Canonical splice acceptor; exon 10 skipping → frameshift in E2 catalytic domain; null-equivalent",
             "phenotype": "Neonatal / infantile Leigh"},
            {"variant": "Large deletion (partial/full DLAT)", "effect": "Complete E2 absence; E2 24-mer core collapses entirely",
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
            }
            for p in sample
        ],
        "seizure_types": [
            {"type": "Focal seizures with secondary generalisation", "pct": 60},
            {"type": "Tonic / tonic-clonic (GTC)", "pct": 52},
            {"type": "Infantile spasms (IS / West syndrome)", "pct": 42},
            {"type": "Myoclonic seizures", "pct": 35},
            {"type": "Absence-like / staring", "pct": 22},
            {"type": "Status epilepticus (SE)", "pct": 20},
        ],
        "trigger_types": [
            {"trigger": "Febrile illness / infection", "pct": 82},
            {"trigger": "Glucose/carbohydrate load", "pct": 70},
            {"trigger": "Fasting / prolonged NPO", "pct": 55},
            {"trigger": "Surgery / anaesthesia (glucose-heavy IV)", "pct": 58},
            {"trigger": "Exercise-induced lactic acidosis", "pct": 48},
            {"trigger": "Intercurrent GI illness (poor intake)", "pct": 40},
        ],
        "treatments": [
            {"drug": "Ketogenic Diet (KD)", "level": "A", "response_pct": 82, "color": "#1b5e20"},
            {"drug": "Thiamine (B1, 100–600 mg/day)", "level": "A", "response_pct": 35, "color": "#1565c0"},
            {"drug": "L-Carnitine (secondary depletion support)", "level": "B", "response_pct": 70, "color": "#4a148c"},
            {"drug": "DCA (Dichloroacetate) — partial-loss only", "level": "B", "response_pct": 28, "color": "#827717"},
            {"drug": "Lipoic acid supplementation (experimental)", "level": "C", "response_pct": 15, "color": "#37474f"},
            {"drug": "LEV (Levetiracetam, AED)", "level": "B", "response_pct": 72, "color": "#006064"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
             "mechanism": "Mitochondrial hepatotoxicity + carnitine depletion → destroys KD therapy. NEVER in PDH complex deficiency."},
            {"drug": "High-carbohydrate diet / enteral feeds", "risk": "EXTREME HAZARD",
             "mechanism": "Floods pyruvate → overwhelms blocked DLAT E2 step → acute lactic acidosis. KD is the treatment; high CHO is the hazard."},
            {"drug": "Glucose-only IV drip", "risk": "HIGH RISK",
             "mechanism": "Pyruvate load without ketone buffer; causes acute metabolic decompensation in DLAT deficiency."},
            {"drug": "Phenobarbital", "risk": "CAUTION",
             "mechanism": "CYP enzyme inducer; may worsen hepatic mitochondrial function; avoid if LEV available."},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "Gene": "DLAT",
            "Full name": "Dihydrolipoamide S-Acetyltransferase (E2 subunit of PDH complex)",
            "Locus": "11q23.1",
            "Protein length": "729 aa (including mitochondrial targeting sequence, MTS)",
            "PDH complex role": "E2 subunit — 24-mer cubic inner scaffold; lipoyl swinging arms; acetyl transfer E1→CoA; anchors E1, E3BP, E3",
            "Lipoyl attachment sites": "Lys173 (L1 domain) and Lys259 (L2 domain) — covalent attachment by LIAS",
            "Reaction catalysed": "Dihydrolipoamide (acetyl-dihydrolipoamide) + CoA → Acetyl-CoA + dihydrolipoamide",
            "OMIM gene": "*608770",
            "OMIM disease": "#245348 (PDH complex deficiency)",
            "Inheritance": "Autosomal Recessive (AR) — both sexes equally affected",
            "Cases worldwide (2026)": "~10–25 confirmed DLAT cases (ultra-rare; rarer than PDHA1/PDHB)",
            "Key negative biomarkers": "Normal BCAA, normal 2-HG, NORMAL DLD/E3 enzyme activity",
        },
        "key_concepts": [
            {
                "term": "E2 24-mer cubic core",
                "definition": (
                    "DLAT (E2) self-assembles into a 24-subunit cubic scaffold (~7 nm edge cube) that forms "
                    "the structural centre of the PDH complex. 12 DLAT dimers → 4 trimers → 8 trimers = 24-mer cube. "
                    "This core is the docking platform for E1 (via E1β) and E3BP/PDHX (which anchors E3). "
                    "Without functional DLAT, the entire complex architecture collapses — E1 AND E3 both lose "
                    "their anchorage simultaneously (more catastrophic than isolated E1α or E1β loss)."
                ),
            },
            {
                "term": "Lipoyl swinging arm mechanism",
                "definition": (
                    "Each DLAT monomer contains two lipoyl domains (L1 at Lys173, L2 at Lys259). "
                    "Lipoic acid is covalently attached to these lysines by LIAS (lipoyl synthetase). "
                    "The lipoamide arm 'swings' between the E1 active site (picks up acetyl group) and "
                    "the E2 active site (transfers acetyl to CoA, producing Acetyl-CoA). DLAT mutations at "
                    "Lys173 or in the lipoyl domain regions directly block this swinging arm mechanism."
                ),
            },
            {
                "term": "Biochemically IDENTICAL to PDHA1 and PDHB",
                "definition": (
                    "DLAT deficiency produces the same PDH block fingerprint as PDHA1 and PDHB: "
                    "L:P ratio NORMAL (10–20); both lactate AND pyruvate accumulate; elevated plasma alanine; "
                    "CSF lactate > plasma lactate. The ONLY biochemical distinction from DLD: normal BCAA, "
                    "normal 2-HG, NORMAL DLD/E3 enzyme activity. Gene panel PDHA1/PDHB/DLAT/PDHX is MANDATORY "
                    "— biochemistry alone cannot distinguish the four PDH complex component deficiencies."
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
                "term": "Ketogenic diet mechanism (E2 bypass)",
                "definition": (
                    "KD provides ketone bodies (β-hydroxybutyrate, acetoacetate) from hepatic fatty acid "
                    "β-oxidation. Ketones cross the BBB and enter neurons, where succinyl-CoA:3-oxoacid-CoA "
                    "transferase (SCOT) and thiolase convert them to Acetyl-CoA — BYPASSING the blocked "
                    "PDH complex entirely. This is why KD is FIRST-LINE Level A therapy for ALL PDH complex "
                    "subunit deficiencies (PDHA1, PDHB, DLAT, PDHX)."
                ),
            },
            {
                "term": "DLAT LOF — structural catastrophe",
                "definition": (
                    "Unlike isolated PDHA1 (E1α) or PDHB (E1β) loss, DLAT LOF is uniquely destructive: "
                    "the E2 core is the physical scaffold that holds E1 AND E3 (via E3BP/PDHX) in place. "
                    "Without DLAT, E1 loses its docking site and E3BP/PDHX cannot anchor E3 → both the "
                    "pyruvate decarboxylation step (E1) AND the lipoamide regeneration step (E3) fail "
                    "simultaneously due to structural disassembly, even if E1α, E1β, PDHX, and DLD "
                    "proteins are individually intact."
                ),
            },
            {
                "term": "AR inheritance — no sex bias",
                "definition": (
                    "DLAT (11q23.1) is autosomal recessive — biallelic LOF required. Both sexes equally "
                    "affected (no X-inactivation variability as in PDHA1). No haploinsufficiency in "
                    "obligate carrier parents. No consanguinity-specific founder allele — all variants "
                    "pan-ethnic and private-family."
                ),
            },
            {
                "term": "Thiamine (B1) — less responsive than PDHA1",
                "definition": (
                    "Thiamine provides TPP cofactor for E1α (PDHA1) — directly enhances E1 activity. "
                    "In DLAT deficiency, the E2 block is downstream of E1. Thiamine may help only if "
                    "residual E2 scaffolding persists (partial-loss variants). Thiamine trial is still "
                    "MANDATORY (Level A) in all PDH complex deficiencies as response cannot be predicted "
                    "without trial, but expect ~30–40% partial response (less than PDHA1 ~55%)."
                ),
            },
            {
                "term": "Lipoic acid supplementation (experimental)",
                "definition": (
                    "Lipoic acid is the cofactor covalently attached to DLAT at Lys173/Lys259 by LIAS. "
                    "Exogenous lipoic acid supplementation is theoretically attractive for hypomorphic "
                    "DLAT variants with partially reduced lipoylation. However, clinical evidence is "
                    "very limited (Level C) — free lipoic acid must first be activated by LIAS before "
                    "attachment, and severely truncated DLAT cannot be rescued."
                ),
            },
            {
                "term": "VPA absolute contraindication",
                "definition": (
                    "Valproate (VPA) is ABSOLUTELY CONTRAINDICATED in ALL PDH complex subunit deficiencies "
                    "(PDHA1, PDHB, DLAT, PDHX). Dual mechanism: (1) mitochondrial hepatotoxicity — "
                    "direct mitochondrial membrane damage with energy failure; (2) carnitine depletion — "
                    "VPA depletes free carnitine, impairing β-oxidation and therefore ketone production "
                    "for KD therapy (the primary treatment). NEVER prescribe VPA in PDH complex deficiency."
                ),
            },
        ],
        "thresholds": [
            {"parameter": "Plasma Lactate", "threshold": "<2.2 mmol/L",
             "dlat_range": "3–18 mmol/L (elevated)"},
            {"parameter": "Plasma Pyruvate", "threshold": "<0.17 mmol/L",
             "dlat_range": "0.3–1.5 mmol/L (elevated)"},
            {"parameter": "L:P ratio (KEY FINGERPRINT)", "threshold": "10–20",
             "dlat_range": "10–20 (NORMAL — PDH block fingerprint; Complex I: L:P >25)"},
            {"parameter": "Plasma Alanine", "threshold": "150–450 µmol/L",
             "dlat_range": "450–1300 µmol/L (elevated — pyruvate transamination surrogate)"},
            {"parameter": "CSF Lactate", "threshold": "<2.1 mmol/L",
             "dlat_range": "3–9 mmol/L (often > plasma lactate)"},
            {"parameter": "Plasma BCAA (Leu/Ile/Val)", "threshold": "Normal",
             "dlat_range": "NORMAL (key negative — distinguishes DLAT from DLD/E3)"},
            {"parameter": "2-Hydroxyglutarate", "threshold": "Normal",
             "dlat_range": "NORMAL (key negative — distinguishes DLAT from DLD/E3)"},
            {"parameter": "DLD/E3 enzyme activity (fibroblasts)", "threshold": "Normal",
             "dlat_range": "NORMAL — E3 is intact in DLAT deficiency"},
            {"parameter": "DLAT/E2 enzyme activity (fibroblasts)", "threshold": "Normal",
             "dlat_range": "<10% normal (severely reduced — definitive biochemical diagnosis)"},
        ],
        "differential_diagnosis": [
            {"condition": "PDHA1 (E1α deficiency)", "distinguishing": "X-linked dominant (Xp22.12): males severely affected, females variable severity. Biochemically IDENTICAL — L:P normal. Gene panel MANDATORY."},
            {"condition": "PDHB (E1β deficiency)", "distinguishing": "AR (3p14.3): same sex distribution as DLAT. Biochemically IDENTICAL — L:P normal. Gene panel MANDATORY."},
            {"condition": "PDHX (E3BP deficiency)", "distinguishing": "AR (11p13): biochemically IDENTICAL to DLAT. Gene panel MANDATORY."},
            {"condition": "DLD (E3 deficiency)", "distinguishing": "AR (7q31.1): BCAA elevated (BCKDH block) + 2-HG elevated (αKGDH block) + glycine mildly elevated (GCS block). KEY biochemical distinction from DLAT."},
            {"condition": "Complex I deficiency (NDUFS/NDUFV genes)", "distinguishing": "L:P ratio >25 (primarily NADH accumulation → LDH pushes lactate disproportionately). KEY distinguisher from all PDH complex deficiencies."},
            {"condition": "Pyruvate Carboxylase (PC) deficiency", "distinguishing": "L:P ratio >25 (same direction as Complex I); citrulline, lysine, ammonia elevated; very low acetyl-CoA markers. No structural brain anomalies."},
            {"condition": "MELAS (mt DNA, MT-TL1/MT-ND5)", "distinguishing": "L:P ratio >25 typically; maternal inheritance; stroke-like episodes; elevated blood lactate with CSF discordance; muscle biopsy RRF."},
            {"condition": "Fumarase deficiency (FH)", "distinguishing": "Fumaric aciduria on urine organic acids; no lactic acidosis fingerprint; severe intellectual disability; brain malformation similar but biochemistry distinct."},
        ],
    }
