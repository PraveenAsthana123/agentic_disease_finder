"""FASTKD2 — Combined Oxidative Phosphorylation Deficiency (CI + CIV)
FASTKD2-916aa-99.4kDa-mt-Matrix-FAST1-FAST2-RAP-HOOK-Domains-2q33.1
Ghezzi-2008-AmJHumGenet-First-FASTKD2-Combined-OXPHOS-Deficiency
40-patient cohort seed-649 | 3 endpoints: get_overview / get_breakdown / get_definitions
"""
import random

SEED = 649
N_PATIENTS = 40

random.seed(SEED)

# ── Pathogenic variants (FASTKD2) ─────────────────────────────────────────────
VARIANTS = [
    {
        "genotype":  "p.Glu323Lys (c.967G>A) — FAST_1 domain core — Italian/European founder",
        "n": 12, "pct": 30,
        "domain":    "FAST_1 domain core (mt-RNA binding interface)",
        "severity":  "Severe",
        "note":      "Most common variant; Italian and pan-European founder; homozygous or compound "
                     "heterozygous; disrupts FAST_1 domain core conformation → loss of 16S mt-rRNA "
                     "binding/processing activity → aberrant rRNA accumulation → impaired large mitoribosome "
                     "assembly → combined CI + CIV deficiency; early childhood Leigh-like MRI >70%; "
                     "Ghezzi 2008 original cohort Italian siblings (Ghezzi et al. AJHG 2008).",
    },
    {
        "genotype":  "p.Arg611Ter (c.1831C>T) — RAP domain truncation",
        "n": 9, "pct": 22,
        "domain":    "RAP domain (RNA-associated protein)",
        "severity":  "Severe",
        "note":      "Nonsense variant; truncates RAP domain required for mt-RNA substrate recognition; "
                     "complete LOF; null allele — no FASTKD2 protein detected; biallelic null results "
                     "in most severe biochemical phenotype; combined CI + CIV <20% residual; various "
                     "ethnic backgrounds; Leigh-like MRI >80% in null allele carriers.",
    },
    {
        "genotype":  "p.Gly508Arg (c.1522G>C) — FAST_2 domain structural",
        "n": 7, "pct": 18,
        "domain":    "FAST_2 domain (structural core)",
        "severity":  "Severe",
        "note":      "Glycine-to-arginine substitution disrupts the structural core of the FAST_2 domain; "
                     "bulky Arg side chain destabilises the compact FAST_2 fold required for mt-rRNA "
                     "processing; near-complete LOF; CI residual 20-30%, CIV 25-35%; progressive "
                     "encephalopathy with early childhood onset; cerebellar predominant MRI in 60%.",
    },
    {
        "genotype":  "p.Leu199Pro (c.596T>C) — MTS-proximal linker, helix-breaking proline",
        "n": 6, "pct": 15,
        "domain":    "MTS-proximal linker region (aa 35-220)",
        "severity":  "Moderate-Severe",
        "note":      "Proline substitution disrupts an alpha-helix in the MTS-proximal linker connecting "
                     "the import signal to the FAST_1 domain; helix-breaking Pro destabilises the overall "
                     "FASTKD2 mt-matrix architecture; partial LOF; CI residual 30-45%, CIV 35-50%; "
                     "attenuated phenotype relative to null alleles; longer survival possible.",
    },
    {
        "genotype":  "Compound biallelic splice/null (compound heterozygous)",
        "n": 6, "pct": 15,
        "domain":    "Various (splice affects multiple exons)",
        "severity":  "Severe",
        "note":      "Compound heterozygous splice-site plus truncating/null combinations; complete LOF "
                     "in both alleles; combined CI + CIV <20% residual; biochemically indistinguishable "
                     "from homozygous null — WES/WGS mandatory for allele characterisation; various "
                     "ethnic backgrounds; Leigh-like MRI near-universal.",
    },
]

# ── Clinical features ─────────────────────────────────────────────────────────
FEATURES = [
    ("Progressive encephalopathy / psychomotor regression", 90, "Near-universal; regression after apparent early development; hallmark of FASTKD2"),
    ("Lactic acidosis",               88, "Baseline elevated; crisis episodes; combined CI+CIV → more pronounced than isolated CIV"),
    ("Hypotonia",                     80, "Generalised; axial > appendicular; early sign"),
    ("Feeding difficulty / NG tube",  65, "Oro-pharyngeal dysmotility; NG support in majority"),
    ("Growth failure",                60, "Failure to thrive; weight below 3rd centile"),
    ("Cerebellar ataxia / dysmetria", 48, "Cerebellar involvement PROMINENT — key DDx vs SURF1 (NO ataxia cardinal); dysmetria, gait ataxia"),
    ("Respiratory compromise",        45, "Progressive hypoventilation; NIV support; less frequent than neonatal CIV diseases"),
    ("Leigh-like MRI (brainstem/cerebellar)", 65, "Brainstem + cerebellar predominant (NOT classic BG Leigh of SURF1 95%); key MRI DDx"),
    ("Seizures",                      55, "Moderate prevalence; NOT cardinal (contrast PET100/COX8A); focal > generalised"),
    ("Hepatopathy (elevated transaminases)", 35, "Mild-moderate hepatic involvement; KEY DDx vs POLG (Alpers hepatopathy severe)"),
    ("NO HCM (key DDx — SCO2/COA5/COA6)", 0, "Cardiomyopathy ABSENT — key negative vs SCO2 (100%), COA5 (88%), COA6 (90%)"),
    ("NO Fanconi tubulopathy (key DDx — COX10)", 0, "Renal tubulopathy ABSENT — key negative vs COX10 (65%)"),
    ("NO PINDAC triad (key DDx — COX4I1)", 0, "PINDAC (pancreatic + anaemia + calvarial) ABSENT — key negative vs COX4I1"),
    ("Early childhood onset (3-18 months)", 100, "NOT neonatal — key DDx vs COX8A, PET100 (neonatal/infantile); 3-18 month onset distinguishes FASTKD2"),
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug":     "Valproic acid (VPA / Depakote)",
        "severity": "ABSOLUTE CI",
        "reason":   "VPA sequesters CoA → impairs mitochondrial β-oxidation → worsens lactic acidosis. "
                    "VPA also inhibits POLG → accelerates mtDNA depletion. FASTKD2 patients already have "
                    "CI deficiency — VPA creates a compounded energetic crisis. Use LEV, LTG, or TPM "
                    "for seizure control; VPA is potentially fatal in combined CI+CIV deficiency.",
    },
    {
        "drug":     "Metformin",
        "severity": "ABSOLUTE CI",
        "reason":   "Metformin is a direct Complex I inhibitor. FASTKD2 patients already have CI residual "
                    "activity of 20-50% — metformin inhibits the remaining CI function → catastrophic "
                    "lactic acidosis and energy failure. This is a double-hit: CI deficiency (intrinsic) "
                    "plus CI inhibition (drug). Absolutely forbidden even if diabetes/hyperglycaemia develops.",
    },
    {
        "drug":     "Linezolid",
        "severity": "ABSOLUTE CI",
        "reason":   "Linezolid inhibits the mitochondrial 23S rRNA large ribosomal subunit. FASTKD2 is "
                    "an mt-16S rRNA processing factor — FASTKD2-deficient patients already have impaired "
                    "large mitoribosome assembly. Linezolid is doubly toxic: it inhibits mt-translation "
                    "(MT-CO1/CO2/CO3 for CIV; MT-ND subunits for CI) AND exacerbates the existing "
                    "mt-RNA processing defect. Absolutely forbidden — use alternative antibiotics.",
    },
    {
        "drug":     "Chloramphenicol",
        "severity": "ABSOLUTE CI",
        "reason":   "Chloramphenicol inhibits the mitochondrial 70S ribosome → blocks translation of "
                    "all 13 mt-encoded OXPHOS subunits (including MT-CO1/CO2/CO3 for CIV and MT-ND "
                    "subunits for CI). In FASTKD2 patients with already impaired mitoribosome assembly, "
                    "chloramphenicol eliminates all remaining mt-translation → complete OXPHOS failure.",
    },
    {
        "drug":     "Propofol",
        "severity": "ABSOLUTE CI",
        "reason":   "Propofol Infusion Syndrome (PRIS): propofol uncouples the mitochondrial membrane "
                    "and directly inhibits the respiratory chain (including CIV). FASTKD2 patients with "
                    "20-50% residual CI and 25-55% residual CIV activity are at high risk for PRIS at "
                    "standard anaesthetic doses. Use sevoflurane for all anaesthesia — never propofol.",
    },
    {
        "drug":     "Ketogenic diet (KD)",
        "severity": "CONTRAINDICATED",
        "reason":   "Beta-oxidation of fatty acids (the metabolic basis of KD) requires functional Complex I "
                    "and Complex IV. FASTKD2 patients have combined CI + CIV deficiency — KD markedly "
                    "worsens metabolic acidosis and energy deficit. KD is used for seizures in some "
                    "mitochondrial diseases (e.g., POLG) but is specifically contraindicated when CI or "
                    "CIV is deficient. Never use in FASTKD2.",
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {"tx": "CoQ10 (Ubiquinol)",        "level": "C",            "note": "10-30 mg/kg/day; electron carrier bypass; may partially compensate complex deficiency"},
    {"tx": "Thiamine (B1)",            "level": "C-MANDATORY",  "note": "Empiric 10-30 mg/kg/day; rule out SLC19A3/BTD before attributing to FASTKD2"},
    {"tx": "Biotin",                   "level": "C-MANDATORY",  "note": "Empiric 10 mg/day; BTD exclusion mandatory — biotinidase deficiency can mimic OXPHOS disease"},
    {"tx": "Riboflavin (B2)",          "level": "C",            "note": "100-200 mg/day; FAD cofactor critical for Complex I (NADH dehydrogenase)"},
    {"tx": "L-Carnitine",              "level": "C",            "note": "100-200 mg/kg/day; secondary carnitine deficiency common in OXPHOS disease"},
    {"tx": "GIR 6-8 mg/kg/min",       "level": "A",            "note": "Continuous IV glucose — anti-catabolic; NEVER fast; fasting triggers CI+CIV crisis"},
    {"tx": "LEV (Levetiracetam)",      "level": "B",            "note": "Preferred AED — renal excretion, no CYP interaction, no mitochondrial toxicity; avoid VPA"},
    {"tx": "NIV/BiPAP",               "level": "A",            "note": "Respiratory compromise 45% — early non-invasive ventilation preferred"},
    {"tx": "Sevoflurane",              "level": "A",            "note": "Only acceptable anaesthetic — propofol ABSOLUTELY CONTRAINDICATED (PRIS)"},
    {"tx": "Nasogastric tube",         "level": "A",            "note": "Feeding difficulty 65% — NG support early to maintain caloric intake and GIR"},
    {"tx": "URSODEOXYCHOLIC ACID",     "level": "C",            "note": "For hepatopathy (35%) — hepatoprotective; monitor liver function 3-monthly"},
]

# ── DDx matrix ────────────────────────────────────────────────────────────────
DDX_MATRIX = [
    {
        "disease": "SURF1 (COXPD1, 9q34.2) — MOST IMPORTANT DDx",
        "shared":  "Leigh MRI, Lactic acidosis, Hypotonia, AR, Childhood onset",
        "distinguishing": "SURF1 = ISOLATED CIV deficiency (CI NORMAL); FASTKD2 = COMBINED CI+CIV. "
                          "SURF1 Leigh = classic basal ganglia (95% BG dominant); FASTKD2 = brainstem + "
                          "cerebellar predominant (65%). SURF1 has NO cerebellar ataxia; FASTKD2 has "
                          "cerebellar ataxia in 48% — CRITICAL MRI + biochemical DDx. "
                          "If combined CI+CIV → exclude SURF1 biochemically; confirm FASTKD2 by WES.",
    },
    {
        "disease": "m.3243A>G (MTTL1) — Mitochondrial MELAS",
        "shared":  "Combined CI+CIV deficiency, Lactic acidosis, Encephalopathy, Seizures",
        "distinguishing": "MTTL1 m.3243A>G = MATERNAL INHERITANCE — never AR. Heteroplasmy detectable "
                          "in blood/urine. MELAS stroke-like episodes, deafness, diabetes (not features "
                          "of FASTKD2). Deletion of FASTKD2 alleles = AR biallelic. Blood heteroplasmy "
                          "analysis separates mt-DNA mutations from nuclear AR causes of combined deficiency.",
    },
    {
        "disease": "POLG (Alpers-Huttenlocher, 15q26.1)",
        "shared":  "Combined OXPHOS deficiency, Lactic acidosis, Encephalopathy, AR, Hepatopathy",
        "distinguishing": "POLG = mtDNA DEPLETION (reduced mtDNA copy number on Southern/NGS) — "
                          "FASTKD2 = mt-RNA processing defect, NO mtDNA depletion. POLG hepatopathy "
                          "is severe/Alpers pattern (100% fatal hepatic failure); FASTKD2 hepatopathy "
                          "is mild (35%, transaminase elevation only). POLG: VPA triggers Alpers crisis "
                          "(same mechanism, same ABSOLUTE CI). mtDNA quantitation is the key separator.",
    },
    {
        "disease": "SCO2 (COXPD2, 22q13.33)",
        "shared":  "CIV involvement, Lactic acidosis, AR, Neonatal/infantile",
        "distinguishing": "SCO2 = HCM 100% CARDINAL (hypertrophic cardiomyopathy); FASTKD2 = NO HCM. "
                          "SCO2 = ISOLATED CIV (not combined CI+CIV); FASTKD2 = COMBINED CI+CIV. "
                          "HCM at bedside = SCO2/COA5/COA6, NOT FASTKD2. CK elevated in SCO2 (myopathy).",
    },
    {
        "disease": "COX10 (COXPD3, 17p12)",
        "shared":  "CIV deficiency, Leigh, Lactic acidosis, AR",
        "distinguishing": "COX10 = Fanconi tubulopathy 65% + normochromic anaemia 80%; FASTKD2 = "
                          "NO tubulopathy + NO anaemia. COX10 = isolated CIV; FASTKD2 = combined CI+CIV. "
                          "Renal tubulopathy + anaemia = COX10, not FASTKD2.",
    },
    {
        "disease": "COX8A (COXPD15, 11q13.1) — Neonatal onset contrast",
        "shared":  "Leigh-like MRI, Lactic acidosis, AR, Seizures",
        "distinguishing": "COX8A onset = NEONATAL to 6 months (FASTKD2 = early childhood 3-18 months). "
                          "COX8A = isolated CIV (not combined CI+CIV); FASTKD2 = combined CI+CIV. "
                          "COX8A seizures CARDINAL (85%); FASTKD2 seizures moderate 55%, NOT cardinal. "
                          "Onset timing is the first clinical separator.",
    },
    {
        "disease": "PET100 (COXPD13, 8q21.12) — Neonatal onset contrast",
        "shared":  "Leigh-like, Lactic acidosis, AR, NO HCM",
        "distinguishing": "PET100 onset = NEONATAL/infantile (birth to 4 months); FASTKD2 = 3-18 months. "
                          "PET100 = ISOLATED CIV <3-8% residual; FASTKD2 = COMBINED CI+CIV 20-50% CI. "
                          "PET100 is Lebanese founder; FASTKD2 is Italian/pan-European founder. "
                          "Combined deficiency (CI+CIV) = FASTKD2 biochemical fingerprint vs isolated CIV = PET100.",
    },
]

# ── Cohort generator ──────────────────────────────────────────────────────────
PHENOTYPE_WEIGHTS = [
    ("Homozygous pGlu323Lys (Italian/European founder)", 0.30),
    ("Homozygous pArg611Ter (null)",                     0.22),
    ("pGly508Arg compound heterozygous",                 0.18),
    ("pLeu199Pro compound heterozygous",                 0.15),
    ("Biallelic splice/null compound",                   0.15),
]


def _generate_cohort():
    rng = random.Random(SEED)
    patients = []
    for i in range(N_PATIENTS):
        geno_roll = rng.random()
        cumulative = 0.0
        geno = PHENOTYPE_WEIGHTS[-1][0]
        for g, w in PHENOTYPE_WEIGHTS:
            cumulative += w
            if geno_roll < cumulative:
                geno = g
                break

        severe = ("null" in geno.lower() or "arg611" in geno.lower()
                  or "splice" in geno.lower() or "gly508" in geno.lower())
        onset_mo  = rng.randint(3, 8)   if severe else rng.randint(8, 18)
        lactate   = round(rng.uniform(9.0, 18.0)  if severe else rng.uniform(5.5, 11.0), 1)
        ci_pct    = rng.randint(20, 35) if severe else rng.randint(36, 50)
        civ_pct   = rng.randint(25, 40) if severe else rng.randint(41, 55)
        leigh     = rng.random() < (0.75 if severe else 0.55)
        hcm       = False  # FASTKD2: NO HCM
        seizures  = rng.random() < (0.62 if severe else 0.45)
        ataxia    = rng.random() < (0.55 if severe else 0.40)
        hepato    = rng.random() < 0.35
        survived  = rng.random() < (0.52 if severe else 0.72)
        sex       = rng.choice(["M", "F"])

        patients.append({
            "id":           f"FKD2-{i+1:03d}",
            "sex":          sex,
            "genotype":     geno,
            "onset_mo":     onset_mo,
            "lactate_mM":   lactate,
            "ci_pct":       ci_pct,
            "civ_pct":      civ_pct,
            "leigh_mri":    "Yes" if leigh  else "No",
            "hcm":          "No",
            "seizures":     "Yes" if seizures else "No",
            "ataxia":       "Yes" if ataxia  else "No",
            "hepato":       "Yes" if hepato  else "No",
            "survived_2yr": "Yes" if survived else "No",
        })
    return patients


_COHORT = _generate_cohort()


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    avg_lactate  = round(sum(p["lactate_mM"] for p in _COHORT) / N_PATIENTS, 1)
    avg_ci       = round(sum(p["ci_pct"]     for p in _COHORT) / N_PATIENTS, 1)
    avg_civ      = round(sum(p["civ_pct"]    for p in _COHORT) / N_PATIENTS, 1)
    leigh_pct    = round(100 * sum(1 for p in _COHORT if p["leigh_mri"] == "Yes") / N_PATIENTS)
    seiz_pct     = round(100 * sum(1 for p in _COHORT if p["seizures"]  == "Yes") / N_PATIENTS)
    ataxia_pct   = round(100 * sum(1 for p in _COHORT if p["ataxia"]    == "Yes") / N_PATIENTS)
    hepato_pct   = round(100 * sum(1 for p in _COHORT if p["hepato"]    == "Yes") / N_PATIENTS)
    surv_pct     = round(100 * sum(1 for p in _COHORT if p["survived_2yr"] == "Yes") / N_PATIENTS)

    return {
        "disease":              "Combined Oxidative Phosphorylation Deficiency due to FASTKD2 Deficiency (Ghezzi-Syndrome)",
        "omim_gene":            "612322",
        "omim_disease":         "N/A — no single OMIM disease number; phenotype described in Ghezzi 2008 AJHG",
        "gene":                 "FASTKD2",
        "alias":                "FASTKD2 (Fas-Activated Serine/Threonine Kinase Domain-Containing Protein 2) — "
                                "catalytically inactive pseudo-kinase; mt-matrix RNA processing factor",
        "chromosome":           "2q33.1",
        "inheritance":          "Autosomal Recessive (AR) — biallelic LOF",
        "protein":              "916 aa; ~99.4 kDa; mitochondrial matrix (no TM helix); MTS aa 1-35; "
                                "contains FAST_1 + FAST_2 + RAP (RNA-associated protein) + HOOK domains; "
                                "NOT a true kinase (catalytically inactive); functions in 16S mt-rRNA processing "
                                "and mitoribosome large subunit (mt-LSU) quality control",
        "onset":                "Early childhood — 3 to 18 months (NOT neonatal; key DDx vs COX8A, PET100)",
        "mechanism":            "FASTKD2 LOF → aberrant 16S mt-rRNA accumulation (failure of mt-rRNA turnover/processing) "
                                "→ impaired large mitoribosomal subunit (mt-LSU) assembly → reduced OXPHOS complex "
                                "biogenesis → combined CI + CIV deficiency (20-50% CI residual; 25-55% CIV residual; "
                                "CIII mildly reduced in ~40%). NOT an isolated CIV disease — combined deficiency is "
                                "the biochemical fingerprint distinguishing FASTKD2 from PET100/COX14/COA3.",
        "assembly_pathway":     "Mitochondrial ribosome large subunit (mt-LSU) biogenesis — FASTKD2 processes "
                                "16S mt-rRNA and controls mt-LSU assembly quality; LOF impairs translation of "
                                "all 13 mt-encoded OXPHOS subunits (7 CI: MT-ND1/2/3/4/4L/5/6; 3 CIV: MT-CO1/2/3; "
                                "2 ATP synthase: MT-ATP6/8; 1 CIII: MT-CYB); CI + CIV most severely affected",
        "biochemical_fingerprint": "Combined CI + CIV deficiency (CI residual 20-50%; CIV residual 25-55%); "
                                   "CIII mildly reduced in ~40%; DISTINCT from isolated CIV diseases (SURF1, PET100, COX14). "
                                   "The combined CI+CIV pattern is the key biochemical clue — if both CI and CIV are "
                                   "deficient with normal CII, consider FASTKD2 and mt-DNA mutations (distinguish by inheritance).",
        "cardinal_feature":     "Combined CI + CIV deficiency + early childhood onset (3-18 months) + cerebellar-predominant "
                                "Leigh-like MRI + progressive encephalopathy + NO HCM — distinguishes FASTKD2 from "
                                "isolated CIV diseases; WES confirms FASTKD2 biallelic mutations",
        "cohort_size":          N_PATIENTS,
        "avg_lactate_mM":       avg_lactate,
        "avg_ci_residual_pct":  avg_ci,
        "avg_civ_residual_pct": avg_civ,
        "kpis": {
            "leigh_mri_pct":     leigh_pct,
            "seizures_pct":      seiz_pct,
            "ataxia_pct":        ataxia_pct,
            "hepato_pct":        hepato_pct,
            "survived_2yr_pct":  surv_pct,
            "hcm_pct":           0,
            "tubulopathy_pct":   0,
            "pindac_pct":        0,
        },
        "key_ddx_negatives": [
            "NO HCM — key DDx vs SCO2 (100%), COA5 (88%), COA6 (90%) — absence rules these out",
            "NO Fanconi tubulopathy — key DDx vs COX10 (65%) — absence rules out COX10",
            "NO PINDAC triad — key DDx vs COX4I1 (PINDAC pathognomonic) — absence rules out COX4I1",
            "NO neonatal onset — onset 3-18 months; key DDx vs COX8A (neonatal) and PET100 (neonatal/infantile)",
            "Combined CI+CIV (NOT isolated CIV) — key DDx vs SURF1/PET100/COX14/COA3 (all isolated CIV)",
            "AR inheritance (NOT maternal) — key DDx vs m.3243A>G MTTL1 (maternal); confirm with mtDNA heteroplasmy",
            "NO mtDNA depletion — key DDx vs POLG (Alpers mtDNA depletion) — FASTKD2 = mt-RNA processing, not replication",
        ],
        "key_contrasts": {
            "FASTKD2_vs_SURF1":   "SURF1 = isolated CIV (CI NORMAL); FASTKD2 = combined CI+CIV. SURF1 Leigh = classic BG "
                                  "(95%); FASTKD2 Leigh = brainstem/cerebellar (65%). SURF1 NO ataxia; FASTKD2 ataxia 48%. "
                                  "Biochemical combined deficiency is the primary FASTKD2 vs SURF1 separator.",
            "FASTKD2_vs_MTTL1":   "Both combined CI+CIV, but MTTL1 m.3243A>G = MATERNAL inheritance + heteroplasmy + "
                                  "MELAS features (stroke-like, deafness, diabetes). FASTKD2 = AR biallelic. "
                                  "Inheritance pattern is the critical separator.",
            "FASTKD2_vs_POLG":    "Both AR, both combined deficiency, both can have hepatopathy. POLG = mtDNA DEPLETION "
                                  "(low copy number); FASTKD2 = mt-RNA processing (normal mtDNA copy number). "
                                  "POLG hepatopathy severe/fatal (Alpers); FASTKD2 hepatopathy mild (35%).",
            "FASTKD2_vs_PET100":  "PET100 = isolated CIV <3-8% residual + NEONATAL onset; FASTKD2 = combined CI+CIV "
                                  "20-50% CI + EARLY CHILDHOOD 3-18 months. Biochemistry separates these clinically.",
            "FASTKD2_vs_SCO2":    "SCO2 = HCM 100% CARDINAL + isolated CIV; FASTKD2 = NO HCM + combined CI+CIV. "
                                  "HCM absence rules out SCO2/COA5/COA6 at bedside.",
        },
    }


def get_breakdown():
    avg_lactate = round(sum(p["lactate_mM"] for p in _COHORT) / N_PATIENTS, 1)
    avg_ci      = round(sum(p["ci_pct"]     for p in _COHORT) / N_PATIENTS, 1)
    avg_civ     = round(sum(p["civ_pct"]    for p in _COHORT) / N_PATIENTS, 1)

    # Genotype distribution
    geno_counts = {}
    for p in _COHORT:
        geno_counts[p["genotype"]] = geno_counts.get(p["genotype"], 0) + 1
    genotype_dist = [
        {"genotype": g, "n": n, "pct": round(100 * n / N_PATIENTS)}
        for g, n in sorted(geno_counts.items(), key=lambda x: -x[1])
    ]

    # Feature prevalence
    feature_prev = [
        {
            "feature": feat,
            "pct":     pct,
            "n":       round(N_PATIENTS * pct / 100),
            "note":    note,
        }
        for feat, pct, note in FEATURES
    ]

    # Outcome by genotype class
    severe_pts = [p for p in _COHORT if ("null" in p["genotype"].lower()
                                          or "arg611" in p["genotype"].lower()
                                          or "splice" in p["genotype"].lower()
                                          or "gly508" in p["genotype"].lower())]
    mild_pts   = [p for p in _COHORT if p not in severe_pts]
    sev_surv   = round(100 * sum(1 for p in severe_pts if p["survived_2yr"] == "Yes") / max(len(severe_pts), 1))
    mild_surv  = round(100 * sum(1 for p in mild_pts   if p["survived_2yr"] == "Yes") / max(len(mild_pts),   1))

    return {
        "genotype_dist":  genotype_dist,
        "feature_prev":   feature_prev,
        "patient_table":  _COHORT,
        "variants":       VARIANTS,
        "contraindications": CONTRAINDICATIONS,
        "treatments":     TREATMENTS,
        "ddx_matrix":     DDX_MATRIX,
        "outcome": {
            "null_allele_2yr_survival_pct":    sev_surv,
            "missense_allele_2yr_survival_pct": mild_surv,
            "note": "2-year survival stratified by allele class (null/truncating vs missense). "
                    "Null alleles: severe combined deficiency, Leigh-like, high early mortality. "
                    "Missense alleles (pLeu199Pro): 30-45% residual CI, 35-50% residual CIV; better prognosis.",
        },
        "avg_lactate_mM":      avg_lactate,
        "avg_ci_residual_pct": avg_ci,
        "avg_civ_residual_pct": avg_civ,
        "biomarker_summary": {
            "lactate":   f"{avg_lactate} mM (mean; crisis >15 mM; combined deficiency drives higher lactate than isolated CIV)",
            "ci_pct":    f"{avg_ci}% residual CI (mean; 20-50% range; lowest in null alleles)",
            "civ_pct":   f"{avg_civ}% residual CIV (mean; 25-55% range)",
            "ciii":      "Normal to mildly reduced (~40% of patients have mild CIII reduction); CII NORMAL",
            "combined":  "Combined CI+CIV deficiency = FASTKD2 biochemical fingerprint; NOT isolated CIV",
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "FASTKD2 (Fas-Activated Serine/Threonine Kinase Domain-Containing Protein 2, OMIM *612322) "
            "encodes a 916-amino-acid mitochondrial matrix protein (~99.4 kDa) that contains FAST_1, FAST_2, "
            "RAP (RNA-associated protein), and HOOK domains. Despite its name, FASTKD2 is catalytically "
            "inactive (pseudo-kinase) — it has no true kinase activity. FASTKD2 functions as a mitochondrial "
            "RNA processing factor, specifically involved in 16S mt-rRNA turnover and large mitoribosomal "
            "subunit (mt-LSU) biogenesis. It is targeted to the mitochondrial matrix by a 35-aa N-terminal "
            "MTS that is cleaved after import. No transmembrane helix — FASTKD2 is a soluble matrix protein."
        ),
        "pathomechanism": (
            "FASTKD2 LOF → failure of 16S mt-rRNA processing/turnover → aberrant 16S rRNA accumulates → "
            "impaired large mitoribosomal subunit (mt-LSU) assembly → reduced mt-translation capacity for "
            "all 13 mt-encoded OXPHOS subunits. The 7 mt-encoded CI subunits (MT-ND1/2/3/4/4L/5/6) and "
            "3 mt-encoded CIV subunits (MT-CO1/2/3) are most severely affected, producing the characteristic "
            "combined CI + CIV deficiency (20-50% residual CI; 25-55% residual CIV). CIII (MT-CYB) is "
            "mildly reduced in ~40% of patients. Complex II (CII) is nuclear-encoded and NORMAL. "
            "The combined deficiency pattern distinguishes FASTKD2 from isolated CIV diseases (SURF1, PET100, "
            "COX14, COA3) that affect only CIV-specific assembly factors."
        ),
        "fastkd2_vs_isolated_civ_critical": (
            "CRITICAL BIOCHEMICAL SEPARATOR: FASTKD2 produces COMBINED CI + CIV deficiency. "
            "Isolated CIV diseases (SURF1, PET100, COX14, COA3, COX8A, SCO1, SCO2, COX10, COX15, COX20, "
            "COA5, COA6) have NORMAL CI. When the enzyme analysis shows BOTH CI and CIV deficient "
            "(with normal CII, normal/near-normal CIII), the differential narrows to: "
            "(1) FASTKD2 (AR, nuclear, 2q33.1) — mt-RNA processing; "
            "(2) mt-DNA mutations affecting mt-LSU (MTTL1 m.3243A>G = maternal, heteroplasmy detectable); "
            "(3) POLG (AR, mtDNA depletion — distinguish by mtDNA copy number). "
            "The combined pattern is the most important clue to reach FASTKD2 diagnosis."
        ),
        "cerebellar_mri_ddx": (
            "FASTKD2 MRI pattern: brainstem + cerebellar signal changes (Leigh-like) in 65% — "
            "cerebellar involvement is PROMINENT and distinguishes FASTKD2 from classic Leigh syndrome. "
            "Classic SURF1/COXPD1 Leigh = symmetric basal ganglia (putamen, globus pallidus) + periaqueductal "
            "grey matter (95% BG-dominant). FASTKD2 Leigh-like = brainstem (inferior olive, dorsal medulla) + "
            "cerebellar white matter and dentate nuclei. The cerebellar-predominant MRI pattern + cerebellar "
            "ataxia (48%) together suggest FASTKD2 (or COX20) rather than classic Leigh. MRI distribution "
            "should always be correlated with enzyme analysis (isolated vs combined deficiency)."
        ),
        "metformin_double_hit_critical": (
            "METFORMIN IS ABSOLUTELY CONTRAINDICATED in FASTKD2 — double-hit mechanism: "
            "(1) FASTKD2 patients have INTRINSIC Complex I deficiency (20-50% residual CI). "
            "(2) Metformin DIRECTLY INHIBITS Complex I (its primary mechanism of action for blood glucose lowering). "
            "Administering metformin to a FASTKD2 patient = blocking 20-50% residual CI activity → near-total "
            "CI failure → catastrophic lactic acidosis + energy crisis. This is analogous to administering "
            "metformin to a patient with MELAS (m.3243A>G) — absolutely forbidden. Never use metformin "
            "in any patient with confirmed or suspected CI deficiency."
        ),
        "linezolid_double_toxicity": (
            "LINEZOLID IS DOUBLY TOXIC IN FASTKD2 — dual mechanism: "
            "(1) Linezolid inhibits the mt-23S rRNA large ribosomal subunit (same target as FASTKD2's "
            "substrate — the 16S mt-rRNA of the large mitoribosome). FASTKD2 patients already have impaired "
            "mt-LSU assembly; linezolid compounds this by further inhibiting the already-compromised "
            "mitoribosome. "
            "(2) Linezolid blocks mt-translation of all 13 mt-encoded OXPHOS subunits → eliminates "
            "remaining CI (MT-ND subunits) and CIV (MT-CO1/2/3) capacity. "
            "This double-hit makes linezolid more dangerous in FASTKD2 than in isolated CIV diseases. "
            "Use alternative antibiotics (tetracyclines, macrolides, quinolones are relatively safer)."
        ),
        "vpa_danger_combined_deficiency": (
            "VPA (VALPROATE) IS ABSOLUTELY CONTRAINDICATED in FASTKD2: "
            "VPA sequesters CoA → impairs mitochondrial β-oxidation → worsens lactic acidosis. "
            "VPA also inhibits POLG → accelerates mtDNA depletion. In FASTKD2, CI is already deficient — "
            "the combined energetic burden of VPA-induced β-oxidation failure + CI deficiency creates "
            "a catastrophic metabolic crisis. Use LEV (levetiracetam) as first-line AED — renal "
            "excretion, no CYP interaction, no mitochondrial toxicity. LTG or TPM are acceptable "
            "alternatives. VPA is the most common avoidable cause of iatrogenic deterioration in "
            "mitochondrial disease patients."
        ),
        "never_fast_gir_protocol": (
            "FASTKD2 patients must NEVER be fasted. Fasting triggers lipolysis → β-oxidation of fatty acids "
            "requires functional CI AND CIV → combined CI+CIV deficiency → acute metabolic decompensation. "
            "Protocol: Continuous IV glucose infusion rate (GIR) 6-8 mg/kg/min; maintain via NG tube "
            "enterally when possible; IV dextrose (10%) perioperatively; emergency glucose protocol during "
            "intercurrent illness. Sick-day rules: double carbohydrate intake, immediate glucose if "
            "vomiting/not tolerating oral feeds. GIR is the single most impactful acute intervention."
        ),
        "references": [
            {
                "citation": "Ghezzi D et al. Am J Hum Genet. 2008;83(4):468-478.",
                "note":     "First description of FASTKD2 mutations causing combined OXPHOS deficiency in "
                            "2 Italian siblings (compound heterozygous). Identified FASTKD2 as a mitochondrial "
                            "RNA-processing factor. Combined CI + CIV deficiency biochemical profile established. "
                            "Leigh-like MRI with cerebellar involvement noted.",
            },
            {
                "citation": "Popow J et al. Science. 2011;331(6024):1534-1537.",
                "note":     "Characterised FASTKD2 as part of the mt-RNA processing complex (FASTK family). "
                            "Established FASTKD2's role in 16S mt-rRNA maturation and mt-large ribosomal "
                            "subunit quality control — mechanistic basis for combined OXPHOS deficiency.",
            },
            {
                "citation": "Jourdain AA et al. EMBO Mol Med. 2015;7(10):1388-1407.",
                "note":     "Comprehensive FASTK family functional characterisation. FASTKD2 specifically "
                            "processes 16S mt-rRNA and controls mitoribosome large subunit (mt-LSU) assembly. "
                            "Confirmed catalytic inactivity of FASTKD2 kinase domain — pseudo-kinase.",
            },
            {
                "citation": "Rahman S. Dev Med Child Neurol. 2015;57(11):986-998.",
                "note":     "Clinical review of combined OXPHOS deficiencies including FASTKD2. "
                            "DDx approach for combined vs isolated respiratory chain defects. "
                            "Management principles for combined CI+CIV deficiency.",
            },
        ],
        "management_summary": (
            "FASTKD2 management follows combined OXPHOS deficiency principles: "
            "(1) NEVER fast — continuous GIR 6-8 mg/kg/min; NG tube early; sick-day protocol. "
            "(2) Thiamine + Biotin empirical supplementation MANDATORY (BTD/SLC19A3 exclusion before diagnosing FASTKD2). "
            "(3) Riboflavin (B2) especially important — FAD cofactor for Complex I. "
            "(4) CoQ10 (ubiquinol), L-carnitine as cofactor support. "
            "(5) Seizures: LEV first-line — NO VPA (ABSOLUTE CI — CI already deficient). "
            "(6) Respiratory: early BiPAP/NIV for progressive hypoventilation (45% of patients). "
            "(7) Hepatopathy (35%): monitor LFTs quarterly; UDCA hepatoprotection; avoid hepatotoxic drugs. "
            "(8) Cerebellar ataxia (48%): physiotherapy, occupational therapy, adaptive equipment. "
            "(9) Anaesthesia: sevoflurane ONLY — propofol ABSOLUTELY FORBIDDEN (PRIS). "
            "(10) Linezolid ABSOLUTELY FORBIDDEN (doubly toxic — mt-LSU inhibitor in a mt-RNA processing defect). "
            "(11) WES/WGS: mandatory for molecular diagnosis — biallelic FASTKD2 mutations; mtDNA quantitation "
            "to exclude POLG/mtDNA depletion. Maternal inheritance history to exclude m.3243A>G."
        ),
    }
