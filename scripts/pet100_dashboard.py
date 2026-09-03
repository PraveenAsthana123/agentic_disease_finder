"""PET100 — COXPD13 · Complex IV Deficiency Nuclear Type 13
PET100-77aa-8.6kDa-Single-TM-Helix-IMM-Anchored-IMS-Facing-C-Terminus-8q21.12
Lim-2013-AmJHumGenet-Lebanese-Founder-pLeu8Ser-First-PET100-COXPD13
40-patient cohort seed-645 | 3 endpoints: get_overview / get_breakdown / get_definitions
"""
import random

SEED = 645
N_PATIENTS = 40

random.seed(SEED)

# ── Pathogenic variants (PET100 / COXPD13) ───────────────────────────────────
VARIANTS = [
    {
        "genotype":  "p.Leu8Ser (c.23T>C) — Lebanese founder",
        "n": 14, "pct": 35,
        "domain":    "TM helix (N-terminal hydrophobic core)",
        "severity":  "Severe",
        "note":      "Most common worldwide; homozygous in Lebanese/Middle-Eastern patients; "
                     "disrupts TM helix anchoring → PET100 fails to integrate into IMM → "
                     "early CIV assembly block; isolated COX deficiency <3% residual; "
                     "Leigh MRI >80%; death <1yr without supportive care (Lim 2013).",
    },
    {
        "genotype":  "p.Arg67* (c.199C>T) — truncating",
        "n": 9, "pct": 23,
        "domain":    "IMS-facing C-terminus",
        "severity":  "Severe",
        "note":      "Nonsense variant; eliminates IMS-facing region required for "
                     "CIV assembly factor interaction; complete LOF; biallelic null "
                     "phenotype identical to pLeu8Ser; severe Leigh.",
    },
    {
        "genotype":  "p.Ala44Val (c.131C>T) — missense",
        "n": 7, "pct": 18,
        "domain":    "Linker region (TM–IMS junction)",
        "severity":  "Severe",
        "note":      "Disrupts flexible linker between TM helix and IMS domain; "
                     "reduces interaction with MT-CO1 early assembly intermediate; "
                     "COX residual ~5-8%; Leigh MRI in ~70%; clinical course milder "
                     "than null alleles but still severe neonatal/infantile.",
    },
    {
        "genotype":  "p.Pro62Leu (c.185C>T) — missense",
        "n": 5, "pct": 13,
        "domain":    "IMS C-terminal (conserved Pro-helix breaker)",
        "severity":  "Moderate-Severe",
        "note":      "Proline-to-leucine — disrupts IMS loop conformation critical "
                     "for CIV subunit docking; partial LOF; 8-12% residual COX; "
                     "attenuated Leigh; longer survival vs null alleles.",
    },
    {
        "genotype":  "Biallelic splice/null (compound)",
        "n": 5, "pct": 13,
        "domain":    "Various",
        "severity":  "Severe",
        "note":      "Compound heterozygous splice + truncating combinations; "
                     "complete LOF; COX <3% residual; indistinguishable from "
                     "homozygous null by biochemistry — WES mandatory.",
    },
]

# ── Clinical features ─────────────────────────────────────────────────────────
FEATURES = [
    ("Lactic acidosis",               90, "Baseline elevated; severe crisis episodes"),
    ("Encephalopathy / global delay", 88, "Progressive — regression after apparent normal early development"),
    ("Hypotonia",                     82, "Generalised; axial > appendicular"),
    ("Feeding difficulty / NG tube",  80, "Oro-pharyngeal dysmotility; nasogastric support in majority"),
    ("Seizures",                      68, "NOT cardinal (contrast COX8A 85% cardinal); focal/multifocal > generalised"),
    ("Leigh MRI (basal ganglia/BS)",  72, "Symmetric signal abnormalities BG + brainstem; NOT cerebellar (COX20 DDx)"),
    ("Respiratory compromise",        60, "Progressive — NIV/BiPAP support required in majority"),
    ("Growth failure",                75, "Failure to thrive; IUGR in 30%"),
    ("Psychomotor regression",        85, "Near-universal after 3–6 months of apparent normal development"),
    ("NO HCM (key DDx — SCO2/COA5)", 0,  "Cardiomyopathy ABSENT — key negative distinguisher vs SCO2 (100%), COA5 (88%)"),
    ("NO hepatopathy (key DDx — SCO1)", 0, "Hepatic failure ABSENT — key negative distinguisher vs SCO1 (100% neonatal)"),
    ("NO tubulopathy (key DDx — COX10)", 0, "Fanconi tubulopathy ABSENT — key negative vs COX10 (65%)"),
    ("NO ataxia (key DDx — COX20)",   0,  "Cerebellar ataxia ABSENT — key negative vs COX20 (100% cardinal)"),
    ("IUGR / prenatal",               30, "Intrauterine growth restriction in subset"),
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug":     "Valproic acid (VPA / Depakote)",
        "severity": "ABSOLUTE CI",
        "reason":   "VPA sequesters CoA → impairs mitochondrial β-oxidation → worsens lactic "
                    "acidosis in CIV deficiency; inhibits POLG → accelerates mtDNA depletion. "
                    "In PET100 (<3% residual COX) VPA administration is potentially fatal. "
                    "Use LEV, LTG, or TPM for seizures.",
    },
    {
        "drug":     "Metformin",
        "severity": "ABSOLUTE CI",
        "reason":   "Metformin inhibits mitochondrial Complex I → combined CI+CIV block in "
                    "patients with severe CIV deficiency → catastrophic lactic acidosis. "
                    "Forbidden even if diabetes/hyperglycaemia develops.",
    },
    {
        "drug":     "Propofol",
        "severity": "ABSOLUTE CI",
        "reason":   "Propofol Infusion Syndrome (PRIS): propofol inhibits mitochondrial "
                    "respiratory chain (CIV + uncoupling). In PET100 (<3-8% residual COX) "
                    "propofol triggers fatal PRIS at lower doses and shorter exposure. "
                    "Use sevoflurane for all anaesthesia.",
    },
    {
        "drug":     "Linezolid",
        "severity": "ABSOLUTE CI",
        "reason":   "Linezolid inhibits mt-23S rRNA → blocks translation of MT-CO1/CO2/CO3 "
                    "(the 3 mt-encoded CIV subunits). Eliminates all residual CIV activity "
                    "in PET100 patients → irreversible mitochondrial crisis.",
    },
    {
        "drug":     "Chloramphenicol",
        "severity": "ABSOLUTE CI",
        "reason":   "Inhibits mt-ribosome (70S) → blocks all 13 mt-encoded OXPHOS subunit "
                    "translations including MT-CO1/CO2/CO3 → total CIV failure.",
    },
    {
        "drug":     "Ketogenic diet (KD)",
        "severity": "CONTRAINDICATED",
        "reason":   "β-oxidation of fatty acids requires functional Complex IV. In PET100 "
                    "(<3-8% residual CIV) KD markedly worsens lactic acidosis and energy "
                    "deficit. KD may trigger fatal metabolic decompensation.",
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {"tx": "CoQ10 (Ubiquinol)",    "level": "C", "note": "10-30 mg/kg/day; electron carrier bypass"},
    {"tx": "Thiamine (B1)",        "level": "C-MANDATORY", "note": "Empiric 10-30 mg/kg/day; rule out SLC19A3/BTD"},
    {"tx": "Biotin",               "level": "C-MANDATORY", "note": "Empiric 10 mg/day; BTD exclusion"},
    {"tx": "Riboflavin (B2)",      "level": "C",   "note": "100-200 mg/day; FAD cofactor for CI/CIII"},
    {"tx": "L-Carnitine",          "level": "C",   "note": "100-200 mg/kg/day; secondary depletion risk"},
    {"tx": "GIR 6-8 mg/kg/min",   "level": "A",   "note": "Continuous IV glucose — anti-catabolic; NEVER fast"},
    {"tx": "LEV (Levetiracetam)",  "level": "B",   "note": "Preferred AED — renal excretion, no CYP interaction, no mtDNA toxicity"},
    {"tx": "NIV/BiPAP",           "level": "A",   "note": "Respiratory compromise 60%+ — early BiPAP preferred over intubation"},
    {"tx": "Sevoflurane",          "level": "A",   "note": "Preferred anaesthetic — NOT propofol (ABSOLUTE CI — PRIS)"},
    {"tx": "Nasogastric tube",     "level": "A",   "note": "Feeding difficulty 80%+ — NG support early"},
]

# ── DDx matrix ────────────────────────────────────────────────────────────────
DDX_MATRIX = [
    {
        "disease": "COX14 (COXPD9, 17q21.33)",
        "shared":  "Leigh MRI, Lactic acidosis, Hypotonia, Isolated COX deficiency, NO HCM, NO hepatopathy",
        "distinguishing": "COX14 and PET100 are BIOCHEMICALLY IDENTICAL (both <5% residual CIV, "
                          "both NO HCM/hepato/tubulo/ataxia). WES/WGS is THE ONLY separator. "
                          "Neither PET100 nor COX14 has a cardinal clinical distinguisher. "
                          "This pair is the most challenging DDx in the COXPD series.",
    },
    {
        "disease": "COA3 (COXPD10, 17q24.2)",
        "shared":  "Leigh MRI, Lactic acidosis, Hypotonia, Isolated COX deficiency, NO HCM",
        "distinguishing": "COA3 and PET100 are also biochemically near-identical. "
                          "WES mandatory. The MITRAC pathway (COA3/COX14) vs PET100 pathway "
                          "distinction is molecular, not clinical.",
    },
    {
        "disease": "SCO2 (COXPD2, 22q13.33)",
        "shared":  "Isolated COX deficiency, Leigh, Lactic acidosis, Neonatal/infantile",
        "distinguishing": "HCM in SCO2 = 100% CARDINAL; HCM in PET100 = 0%. "
                          "HCM at bedside = SCO2/COA5/COA6, NOT PET100.",
    },
    {
        "disease": "COX10 (COXPD3, 17p12)",
        "shared":  "Isolated COX, Leigh, Lactic acidosis",
        "distinguishing": "Fanconi tubulopathy 65% in COX10, 0% in PET100. "
                          "Anaemia 80% in COX10, 0% in PET100.",
    },
    {
        "disease": "SCO1 (COXPD4, 17p13.1)",
        "shared":  "Isolated COX, Neonatal, Lactic acidosis",
        "distinguishing": "Neonatal hepatic failure 100% in SCO1, 0% in PET100. "
                          "Hepatic failure at bedside = SCO1, NOT PET100.",
    },
    {
        "disease": "COX20 (COXPD8, 2q11.2)",
        "shared":  "Isolated COX, Neurological",
        "distinguishing": "Ataxia 100% CARDINAL in COX20; childhood onset (NOT neonatal). "
                          "PET100 = neonatal/infantile, NO ataxia.",
    },
    {
        "disease": "SURF1 (COXPD1, 9q34.2)",
        "shared":  "Leigh MRI (both ~75%), Isolated COX, AR",
        "distinguishing": "SURF1 Leigh is the most common; PET100 is Lebanese/rare founder. "
                          "SURF1 seizures are NOT cardinal; PET100 seizures 68%. Molecular.",
    },
]

# ── Cohort generator ──────────────────────────────────────────────────────────
PHENOTYPE_WEIGHTS = [
    ("Homozygous pLeu8Ser (Lebanese)",       0.35),
    ("Homozygous pArg67* (null)",            0.23),
    ("pAla44Val compound heterozygous",      0.18),
    ("pPro62Leu compound heterozygous",      0.13),
    ("Biallelic splice/null compound",       0.11),
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

        severe = "null" in geno.lower() or "leu8ser" in geno.lower() or "arg67" in geno.lower()
        onset_mo  = rng.randint(0, 2) if severe else rng.randint(1, 5)
        lactate   = round(rng.uniform(8.0, 18.0) if severe else rng.uniform(5.0, 12.0), 1)
        cox_pct   = rng.randint(1, 4) if severe else rng.randint(5, 12)
        leigh     = rng.random() < (0.82 if severe else 0.62)
        hcm       = False  # PET100: NO HCM
        seizures  = rng.random() < (0.72 if severe else 0.62)
        hepato    = False  # PET100: NO hepatopathy
        survived  = rng.random() < (0.38 if severe else 0.68)
        sex       = rng.choice(["M", "F"])

        patients.append({
            "id":          f"PET-{i+1:03d}",
            "sex":         sex,
            "genotype":    geno,
            "onset_mo":    onset_mo,
            "lactate_mM":  lactate,
            "cox_pct":     cox_pct,
            "leigh_mri":   "Yes" if leigh else "No",
            "hcm":         "No",
            "seizures":    "Yes" if seizures else "No",
            "hepato":      "No",
            "survived_1yr": "Yes" if survived else "No",
        })
    return patients


_COHORT = _generate_cohort()

# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    avg_lactate = round(sum(p["lactate_mM"] for p in _COHORT) / N_PATIENTS, 1)
    avg_cox     = round(sum(p["cox_pct"]    for p in _COHORT) / N_PATIENTS, 1)
    leigh_pct   = round(100 * sum(1 for p in _COHORT if p["leigh_mri"] == "Yes") / N_PATIENTS)
    seiz_pct    = round(100 * sum(1 for p in _COHORT if p["seizures"]  == "Yes") / N_PATIENTS)
    surv_pct    = round(100 * sum(1 for p in _COHORT if p["survived_1yr"] == "Yes") / N_PATIENTS)

    return {
        "disease":              "Complex IV Deficiency, Nuclear Type 13 (COXPD13 / PET100 Assembly Factor Deficiency)",
        "omim_gene":            "614848",
        "omim_disease":         "614855",
        "gene":                 "PET100",
        "alias":                "PET100 (Protein Expression Technology 100) — CIV Early Assembly Factor",
        "chromosome":           "8q21.12",
        "inheritance":          "Autosomal Recessive (AR) — biallelic LOF",
        "protein":              "77 aa; 8.6 kDa; single TM helix (aa 2-24); IMS-facing C-terminus; "
                                "inner mitochondrial membrane; required for early MT-CO1 co-translational "
                                "CIV assembly (PET100-MITRAC-early branch); interacts with newly synthesized "
                                "MT-CO1 before subunit integration into maturing CIV complex",
        "onset":                "Neonatal to early infantile (birth to 4 months)",
        "mechanism":            "PET100 LOF → failure of early MT-CO1 co-translational stabilisation → "
                                "MT-CO1 degraded by IMM proteases (YME1L/AFG3L2) → isolated Complex IV "
                                "deficiency (<3-8% residual CIV; CI, CII, CIII NORMAL = biochemical fingerprint). "
                                "Functionally analogous to COX14 and COA3 in the MT-CO1 branch of MITRAC.",
        "assembly_pathway":     "MT-CO1 early co-translational branch (MITRAC) — PET100 joins after MT-CO1 "
                                "synthesis, before COX14/COA3 complex formation; distinct from MT-CO2 branch (COX20)",
        "biochemical_fingerprint": "Isolated COX deficiency <3-8% residual; Complex I, II, III = NORMAL. "
                                   "This BIOCHEMICAL FINGERPRINT is shared with COX14, COA3, COX4I1 — "
                                   "WES/WGS is the ONLY separator for these four diseases.",
        "cardinal_feature":     "Leigh syndrome + severe lactic acidosis + isolated CIV deficiency + "
                                "NO HCM + NO hepatopathy + NO tubulopathy — clinically nearly identical "
                                "to COX14 (COXPD9) and COA3 (COXPD10); WES mandatory for diagnosis",
        "cohort_size":          N_PATIENTS,
        "avg_lactate_mM":       avg_lactate,
        "avg_cox_residual_pct": avg_cox,
        "kpis": {
            "leigh_mri_pct":     leigh_pct,
            "seizures_pct":      seiz_pct,
            "survived_1yr_pct":  surv_pct,
            "hcm_pct":           0,
            "hepato_pct":        0,
            "tubulopathy_pct":   0,
        },
        "key_ddx_negatives": [
            "NO HCM — key DDx vs SCO2 (100%), COA5 (88%), COA6 (90%), COX15 (78%) — absence rules these out",
            "NO hepatopathy — key DDx vs SCO1 (100% neonatal hepatic failure) — absence rules out SCO1",
            "NO tubulopathy/Fanconi — key DDx vs COX10 (65%) — absence rules out COX10",
            "NO anaemia — key DDx vs COX10 (80%), COX4I1 dyserythropoietic (88%) — absence rules these out",
            "NO cerebellar ataxia — key DDx vs COX20 (100% cardinal) — neonatal onset rules out COX20",
            "WES/WGS MANDATORY — PET100 vs COX14 vs COA3 CANNOT be distinguished at the bedside",
        ],
        "key_contrasts": {
            "PET100_vs_COX14": "BIOCHEMICALLY IDENTICAL — both have isolated COX <5%, Leigh, NO HCM/hepato/tubulo. "
                               "ONLY WES/WGS separates. PET100 = Lebanese/rare founder; COX14 = various ethnic groups. "
                               "Clinically indistinguishable — the most challenging DDx pair in COXPD.",
            "PET100_vs_COA3":  "ALSO biochemically identical. Both MT-CO1 MITRAC branch. Both <5% residual CIV. "
                               "PET100 is slightly earlier in the assembly cascade than COA3. WES only.",
            "PET100_vs_SCO2":  "HCM 100% in SCO2 vs 0% in PET100. HCM = SCO2, NOT PET100.",
            "PET100_vs_SCO1":  "Neonatal hepatic failure 100% in SCO1 vs 0% in PET100. Hepatic failure = SCO1.",
            "PET100_vs_COX8A": "COX8A: SEIZURES CARDINAL (85%), brain-dominant, NO multi-system. "
                               "PET100: Leigh + multi-system, seizures 68% (not cardinal).",
        },
    }


def get_breakdown():
    avg_lactate = round(sum(p["lactate_mM"] for p in _COHORT) / N_PATIENTS, 1)
    avg_cox     = round(sum(p["cox_pct"]    for p in _COHORT) / N_PATIENTS, 1)

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
    severe_pts  = [p for p in _COHORT if "leu8ser" in p["genotype"].lower() or "arg67" in p["genotype"].lower() or "splice" in p["genotype"].lower()]
    mild_pts    = [p for p in _COHORT if p not in severe_pts]
    sev_surv    = round(100 * sum(1 for p in severe_pts if p["survived_1yr"] == "Yes") / max(len(severe_pts), 1))
    mild_surv   = round(100 * sum(1 for p in mild_pts   if p["survived_1yr"] == "Yes") / max(len(mild_pts),   1))

    return {
        "genotype_dist":  genotype_dist,
        "feature_prev":   feature_prev,
        "patient_table":  _COHORT,
        "variants":       VARIANTS,
        "contraindications": CONTRAINDICATIONS,
        "treatments":     TREATMENTS,
        "ddx_matrix":     DDX_MATRIX,
        "outcome": {
            "null_allele_1yr_survival_pct":    sev_surv,
            "missense_allele_1yr_survival_pct": mild_surv,
            "note": "1-year survival stratified by allele class (null vs missense). "
                    "Null alleles: severe Leigh, death in majority without aggressive supportive care. "
                    "Missense alleles: some residual CIV activity (5-12%); longer survival possible.",
        },
        "avg_lactate_mM":       avg_lactate,
        "avg_cox_residual_pct": avg_cox,
        "biomarker_summary": {
            "lactate":   f"{avg_lactate} mM (mean; severe crisis >15 mM)",
            "cox_pct":   f"{avg_cox}% residual CIV (mean; <5% in null alleles)",
            "ci_cii_ciii": "NORMAL — biochemical fingerprint of isolated CIV",
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "PET100 (Protein Expression Technology 100, OMIM *614848) encodes a 77-amino-acid "
            "mitochondrial inner membrane protein (8.6 kDa) with a single N-terminal transmembrane "
            "helix (aa 2-24) that anchors it in the IMM with its C-terminus facing the "
            "intermembrane space. PET100 is one of the earliest-acting assembly factors in the "
            "MT-CO1 co-translational branch of Complex IV biogenesis, interacting with newly "
            "synthesized MT-CO1 polypeptide before it is incorporated into larger CIV assembly "
            "intermediates. PET100 functions in the same early MITRAC cascade as COX14 and COA3, "
            "though it joins the assembly sequence at a distinct step."
        ),
        "pathomechanism": (
            "PET100 LOF → failure to stabilise newly synthesised MT-CO1 polypeptide → MT-CO1 is "
            "degraded by inner membrane proteases (YME1L, AFG3L2) before it can be incorporated into "
            "CIV assembly intermediates → no mature CIV complex is formed → isolated Complex IV "
            "deficiency (<3-8% residual activity; Complexes I, II, III remain NORMAL, providing "
            "the biochemical fingerprint). The consequence is near-total loss of mitochondrial "
            "electron transport chain capacity at the terminal step, producing severe lactic "
            "acidosis and energy failure in high-demand tissues (brain, heart, muscle)."
        ),
        "coxpd13_vs_coxpd9_coxpd10_critical": (
            "MOST CHALLENGING DDx IN COXPD SERIES: PET100 (COXPD13), COX14 (COXPD9), and "
            "COA3 (COXPD10) are all early MT-CO1 MITRAC branch assembly factors. All three "
            "produce BIOCHEMICALLY IDENTICAL disease: isolated CIV <5%, Leigh MRI, lactic "
            "acidosis, NO HCM, NO hepatopathy, NO tubulopathy, NO ataxia. The ONLY clinical "
            "distinguisher is ethnic background (PET100 = Lebanese/Middle-Eastern founder; "
            "COX14/COA3 = pan-ethnic). WES/WGS is MANDATORY to distinguish these three diseases. "
            "This is the most important teaching point for PET100/COXPD13 — never diagnose "
            "COXPD13 by biochemistry alone."
        ),
        "assembly_pathway_detail": (
            "CIV (cytochrome c oxidase) assembly follows two main MITRAC branches: "
            "(1) MT-CO1 branch: MT-CO1 translation → COX14/COA3/PET100 co-translational "
            "stabilisation → MITRAC72/MITRAC12 complex → haem a/a3 insertion via COX10/COX15 → "
            "integration with nuclear-encoded subunits → CIV homodimer. "
            "(2) MT-CO2 branch: MT-CO2 → COX20 stabilisation → SCO1/SCO2/COA6 CuA metalation → "
            "COX2 module integration. PET100 acts exclusively in branch (1), distinct from "
            "the COX20/SCO pathway (branch 2)."
        ),
        "vpa_linezolid_propofol_danger": (
            "THREE ABSOLUTE CI DRUGS IN PET100 — all commonly used in ICU/neurology: "
            "(1) VPA (Valproate/Depakote): CoA sequestration → β-oxidation failure + POLG inhibition → "
            "accelerated mtDNA depletion → potentially fatal in CIV deficiency. Use LEV instead. "
            "(2) Propofol: PRIS (Propofol Infusion Syndrome) = uncoupling + CIV inhibition → "
            "at <3-8% residual CIV, propofol elimination of residual activity is immediately life-threatening. "
            "Use sevoflurane. (3) Linezolid: mt-23S rRNA inhibition → blocks MT-CO1/CO2/CO3 "
            "translation → eliminates all residual CIV → irreversible. Never use in PET100."
        ),
        "never_fast_gir_protocol": (
            "PET100 patients must NEVER be fasted. Fasting triggers lipolysis → β-oxidation requires "
            "CIV → CIV deficiency → acute metabolic decompensation → lactic crisis. Protocol: "
            "Continuous IV glucose infusion rate (GIR) 6-8 mg/kg/min; maintain via NG tube enterally "
            "when possible; IV dextrose (10%) perioperatively; emergency glucose protocol during "
            "any intercurrent illness. Sick-day rules: double carbohydrate intake, immediate glucose "
            "if vomiting/not tolerating oral feeds."
        ),
        "references": [
            {
                "citation": "Lim SC et al. Am J Hum Genet. 2013;92(4):553-562.",
                "note":     "First description of PET100 mutations causing COXPD13 in Lebanese patients. "
                            "Identified the pLeu8Ser (c.23T>C) Lebanese founder mutation. Characterised "
                            "PET100 as an early CIV assembly factor required for MT-CO1 stability.",
            },
            {
                "citation": "Mick DU et al. Cell Metab. 2012;15(4):544-554.",
                "note":     "MITRAC complex characterisation — established the MT-CO1 and MT-CO2 branches "
                            "of CIV assembly and the role of early assembly factors (COX14, COA3, MITRAC12/72). "
                            "Contextualises PET100 in the MITRAC cascade.",
            },
            {
                "citation": "Stroud DA et al. Cell Metab. 2015;22(2):302-315.",
                "note":     "Comprehensive CIV assembly factor survey using SILAC-based proteomics. "
                            "Confirmed PET100 as an early MT-CO1 co-translational assembly factor.",
            },
            {
                "citation": "Rahman S. Dev Med Child Neurol. 2015;57(11):986-998.",
                "note":     "Clinical review of mitochondrial Complex IV deficiency — covers COXPD series "
                            "including COXPD13/PET100, differential diagnosis approach, management principles.",
            },
        ],
        "management_summary": (
            "PET100/COXPD13 management follows standard mitochondrial Leigh syndrome principles: "
            "(1) NEVER fast — continuous GIR 6-8 mg/kg/min; NG tube early; sick-day protocol. "
            "(2) Thiamine + Biotin empirical supplementation MANDATORY (BTD/SLC19A3 exclusion). "
            "(3) CoQ10 (ubiquinol), riboflavin, L-carnitine as cofactor support. "
            "(4) Seizures: LEV first-line — renal excretion, no mitochondrial toxicity; AVOID VPA. "
            "(5) Respiratory: early BiPAP/NIV for progressive hypoventilation. "
            "(6) Anaesthesia: sevoflurane ONLY — propofol ABSOLUTELY FORBIDDEN. "
            "(7) WES/WGS: mandatory for molecular diagnosis — cannot distinguish from COX14/COA3 biochemically. "
            "(8) Enrol in natural history registry / gene therapy trial if available."
        ),
    }
