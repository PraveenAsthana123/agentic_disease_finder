"""MTFMT — Mitochondrial Methionyl-tRNA Formyltransferase
mt-tRNA Formylation Defect → Leigh Syndrome / Combined OXPHOS Deficiency (CI + CIII + CIV)
MTFMT-376aa-41.6kDa-mt-Matrix-No-TM-Helix-N10-Formyl-THF-Dependent-15q22.31
Tucker-2011-AmJHumGenet-First-MTFMT-Leigh-Combined-OXPHOS-Deficiency
40-patient cohort seed-651 | 3 endpoints: get_overview / get_breakdown / get_definitions
"""
import random

SEED = 651
N_PATIENTS = 40

random.seed(SEED)

# ── Pathogenic variants (MTFMT) ───────────────────────────────────────────────
VARIANTS = [
    {
        "genotype":  "c.626C>T — splice-altering hypomorphic — Western European founder",
        "n": 14, "pct": 35,
        "domain":    "Exon 4 / IVS4 splice-donor region (formylation domain proximal)",
        "severity":  "Moderate-Severe",
        "note":      "Most common MTFMT variant; Western European founder allele (Tucker 2011); disrupts "
                     "the exon 4 splice-donor site → partial exon skipping → ~25% residual mt-tRNA "
                     "formylation when homozygous; compound heterozygous with null allele produces "
                     "intermediate-severe combined OXPHOS deficiency (CI 25-40%, CIII 30-45%, CIV 40-55% "
                     "residual); Leigh-like MRI with basal ganglia and brainstem involvement; onset "
                     "6-18 months; survival better than null/null combinations.",
    },
    {
        "genotype":  "c.626C>T / p.Trp326Ter (c.978G>A) — compound heterozygous hypomorphic+null",
        "n": 10, "pct": 25,
        "domain":    "Splice-donor + catalytic domain truncation (dual-allele LOF)",
        "severity":  "Severe",
        "note":      "Compound heterozygous: hypomorphic splice allele + null truncating allele; "
                     "net residual formylation ~10-15%; pan-OXPHOS deficiency most severe (CI <25%, "
                     "CIII <30%, CIV <40% residual); Leigh syndrome with bilateral basal ganglia + "
                     "brainstem + periventricular lesions; onset 3-12 months; high neonatal/infantile "
                     "morbidity; Tucker 2011 original cohort contained several such compound patients.",
    },
    {
        "genotype":  "p.Arg237Trp (c.709C>T) — catalytic fold disruption",
        "n": 7, "pct": 17,
        "domain":    "Catalytic core (N10-formyl-THF binding fold; Arg237 contacts formyl group)",
        "severity":  "Severe",
        "note":      "Arginine-to-tryptophan substitution at the conserved Arg237 position within the "
                     "N10-formyl-tetrahydrofolate (N10-formyl-THF) binding cavity; bulky Trp residue "
                     "sterically clashes with the formyl-donor substrate → near-complete loss of "
                     "formyl-transfer activity; residual CI <20%, CIII <25%, CIV <35%; severe Leigh "
                     "syndrome; onset 3-8 months; typically biallelic with second severe allele.",
    },
    {
        "genotype":  "p.Leu261Pro (c.782T>C) — helix-breaking proline in dimerisation helix",
        "n": 5, "pct": 12,
        "domain":    "α-helix 7 (MTFMT homodimer interface; Leu261 in hydrophobic core contact)",
        "severity":  "Moderate-Severe",
        "note":      "Proline substitution disrupts an alpha-helix critical for MTFMT homodimerisation "
                     "and substrate binding orientation; helix-breaking Pro destabilises the dimer "
                     "interface required for optimal mt-tRNA formylation; partial LOF (~30-40% residual "
                     "formylation); CI 30-45%, CIII 35-50%, CIV 45-60% residual; attenuated Leigh-like "
                     "phenotype; onset 12-24 months; longer survival vs null alleles.",
    },
    {
        "genotype":  "Biallelic null / truncating compound (homozygous null or compound null+null)",
        "n": 4, "pct": 11,
        "domain":    "Various (frameshift, nonsense, large deletion — complete LOF)",
        "severity":  "Very Severe",
        "note":      "Biallelic null alleles (homozygous truncating or two different null alleles); "
                     "complete absence of MTFMT protein → zero formylation → catastrophic pan-OXPHOS "
                     "failure; CI <10%, CIII <15%, CIV <20% residual; earliest onset (birth to 3 "
                     "months); most severe Leigh syndrome with extensive brainstem + BG + periventricular "
                     "lesions; high neonatal mortality; only described in consanguineous families.",
    },
]

# ── Phenotype distribution weights for cohort generation ─────────────────────
PHENOTYPE_WEIGHTS = [
    ("c.626C>T — splice-altering hypomorphic — Western European founder",                 0.35),
    ("c.626C>T / p.Trp326Ter (c.978G>A) — compound heterozygous hypomorphic+null",       0.25),
    ("p.Arg237Trp (c.709C>T) — catalytic fold disruption",                                0.17),
    ("p.Leu261Pro (c.782T>C) — helix-breaking proline in dimerisation helix",             0.12),
    ("Biallelic null / truncating compound (homozygous null or compound null+null)",       0.11),
]

# ── Clinical features ─────────────────────────────────────────────────────────
FEATURES = [
    ("Lactic acidosis (baseline elevated)",             95, "Near-universal; pan-OXPHOS deficiency → severe lactate elevation; crisis episodes pH <7.1"),
    ("Hypotonia (generalised)",                         92, "Axial > appendicular; early and prominent sign across all genotypes"),
    ("Psychomotor regression / encephalopathy",         88, "Hallmark feature; regression after apparent early milestones in moderate alleles"),
    ("Leigh-like MRI (basal ganglia + brainstem)",      80, "BG predominant (putamen, caudate, globus pallidus) + dorsal brainstem; pan-OXPHOS pattern"),
    ("Feeding difficulty / NG tube dependence",         72, "Oro-pharyngeal dysmotility; NG support common; aspiration risk"),
    ("Respiratory compromise",                          60, "Progressive hypoventilation; NIV support; intercurrent respiratory infections dangerous"),
    ("Seizures (focal and/or generalised)",             45, "Moderate prevalence; NOT cardinal contrast to COX8A/FASTKD2 patterns; LEV preferred"),
    ("Growth failure / failure to thrive",              65, "Weight <3rd centile; caloric support via NG essential"),
    ("Developmental delay (global)",                    85, "All developmental domains affected; motor > language in early phase"),
    ("Periventricular white matter change",             35, "Additional to classic BG Leigh changes; reflects severe pan-OXPHOS impact on white matter"),
    ("NO HCM (key DDx — SCO2/COA5/COA6)",               0, "Cardiomyopathy ABSENT — key negative vs SCO2 (HCM 100%), COA5 (88%), COA6 (90%)"),
    ("NO Fanconi tubulopathy (key DDx — COX10)",         0, "Renal Fanconi ABSENT — key negative vs COX10 (65%)"),
    ("NO GRACILE triad (key DDx — BCS1L)",               0, "No iron overload/aminoaciduria/cholestasis — key negative vs BCS1L"),
    ("NO neonatal onset (6-24 months typical)",        100, "NOT neonatal in most (exception: biallelic null) — key DDx vs COX8A, PET100, UQCC2 (neonatal)"),
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug":     "Valproic acid (VPA / Depakote)",
        "severity": "ABSOLUTE CI",
        "reason":   "VPA sequesters CoA → blocks mitochondrial β-oxidation → worsens lactic acidosis. "
                    "MTFMT patients already have pan-OXPHOS deficiency (CI + CIII + CIV all impaired) — "
                    "VPA adds an additional metabolic burden that can precipitate fatal metabolic crisis. "
                    "VPA also inhibits POLG → risk of further mtDNA damage. Use LEV (levetiracetam) "
                    "first-line for all seizures; LTG or TPM acceptable alternatives. "
                    "VPA is the most dangerous and most commonly prescribed avoidable drug in mito disease.",
    },
    {
        "drug":     "Metformin",
        "severity": "ABSOLUTE CI",
        "reason":   "Metformin directly inhibits Complex I (primary mechanism for glucose-lowering). "
                    "MTFMT patients have CI residual of 15-45% depending on genotype — metformin eliminates "
                    "the remaining CI function → catastrophic lactic acidosis + energy failure. "
                    "Analogous to administering metformin in MELAS (m.3243A>G) — universally forbidden "
                    "in any confirmed or suspected CI deficiency.",
    },
    {
        "drug":     "Linezolid",
        "severity": "ABSOLUTE CI",
        "reason":   "Linezolid inhibits the mt-large ribosomal subunit (mt-23S rRNA target). "
                    "MTFMT patients have INTRINSICALLY impaired mt-translation (due to absent N-formyl-Met-tRNA); "
                    "linezolid further blocks the already-compromised mitoribosome → eliminates all remaining "
                    "mt-encoded OXPHOS subunit translation → total collapse of CI + CIII + CIV + CV. "
                    "Use alternative antibiotics: tetracyclines, macrolides (short course), quinolones "
                    "(with ECG monitoring), glycopeptides (vancomycin for gram-positive coverage).",
    },
    {
        "drug":     "Chloramphenicol",
        "severity": "ABSOLUTE CI",
        "reason":   "Chloramphenicol binds the mt-50S ribosomal subunit peptidyl transferase site → "
                    "blocks mt-translation of all 13 mt-encoded subunits. Doubly toxic in MTFMT: "
                    "(1) directly blocks the same mt-ribosome MTFMT is already failing to prime correctly, "
                    "(2) eliminates all residual OXPHOS capacity. Never use in any mitochondrial translation defect.",
    },
    {
        "drug":     "Propofol",
        "severity": "ABSOLUTE CI",
        "reason":   "Propofol Infusion Syndrome (PRIS) risk is maximal in pan-OXPHOS deficiency. "
                    "Propofol inhibits mitochondrial respiration at multiple points (uncouples OXPHOS, "
                    "impairs β-oxidation). With CI + CIII + CIV all deficient, any additional OXPHOS "
                    "inhibition causes immediate cardiac failure. Use sevoflurane-based anaesthesia only. "
                    "Even bolus propofol for procedural sedation is dangerous — avoid completely.",
    },
    {
        "drug":     "Ketogenic diet (KD)",
        "severity": "CONTRAINDICATED",
        "reason":   "KD shifts energy metabolism to β-oxidation of fatty acids (ketone body production). "
                    "Fatty acid β-oxidation requires functional Complex I (NADH → NAD+ regeneration) "
                    "AND functional Complex III + IV (electron transport). MTFMT patients have all three "
                    "deficient → KD forces cells into a metabolic pathway that requires the exact enzymes "
                    "that are dysfunctional → acute decompensation and worsening lactic acidosis. "
                    "Maintain high-carbohydrate, low-fat diet; glucose is the safer fuel.",
    },
]

# ── Treatments / management ───────────────────────────────────────────────────
TREATMENTS = [
    {
        "intervention": "Continuous glucose infusion (GIR 6-8 mg/kg/min)",
        "level":        "Level A — Mandatory",
        "reason":       "Prevents lipolysis and ketone-dependent metabolism; glucose bypasses defective "
                        "OXPHOS during crises; NG tube early for continuous enteral glucose delivery; "
                        "IV dextrose (10%) perioperatively and during intercurrent illness; sick-day "
                        "protocol: increase carbohydrate 50-100%, avoid fasting >4 hours.",
    },
    {
        "intervention": "Thiamine (B1) — empirical",
        "level":        "Level C — Mandatory empiric",
        "reason":       "Leigh syndrome DDx exclusion: SLC19A3-THTR2 (biotin-thiamine-responsive BDE) "
                        "is treatable with thiamine + biotin. Before genetic diagnosis is confirmed, "
                        "empirical thiamine MUST be given — response within 3-5 days rules in SLC19A3. "
                        "Thiamine is also a CI cofactor (pyruvate dehydrogenase, α-ketoglutarate DH). "
                        "Dose: 10-20 mg/kg/day IV/NG in acute phase.",
    },
    {
        "intervention": "Biotin — empirical",
        "level":        "Level C — Mandatory empiric",
        "reason":       "Biotinidase deficiency (BTD) and holocarboxylase synthetase deficiency (HLCS) "
                        "cause Leigh-like presentations treatable with biotin. Empirical biotin is "
                        "MANDATORY before molecular diagnosis; if BTD/HLCS — dramatic response within days. "
                        "Also supports pyruvate carboxylase (PC) activity. Dose: 10-20 mg/day.",
    },
    {
        "intervention": "Riboflavin (B2) — FAD cofactor for CI and CIII",
        "level":        "Level C — Recommended",
        "reason":       "Riboflavin (FAD precursor) is an essential cofactor for Complex I (FMN-containing "
                        "NDUFV1/NDUFS1 subunits) and Complex III (FAD in BCS1L/RISP assembly). "
                        "In pan-OXPHOS deficiency, supplementing FAD cofactor may partially support residual "
                        "activity of the most severely affected complexes. Dose: 100-400 mg/day.",
    },
    {
        "intervention": "CoQ10 (ubiquinol form)",
        "level":        "Level C — Recommended",
        "reason":       "Coenzyme Q10 is the mobile electron carrier between CI/CII and CIII. "
                        "Supplementing CoQ10 may support residual electron transfer between deficient CI "
                        "and CIII. Ubiquinol (reduced form) preferred over ubiquinone for bioavailability. "
                        "Dose: 10-30 mg/kg/day; monitor for diarrhoea at high doses.",
    },
    {
        "intervention": "Levetiracetam (LEV) — first-line AED",
        "level":        "Level B — Preferred seizure management",
        "reason":       "LEV: renal excretion (no hepatic CYP450 metabolism), no mitochondrial toxicity, "
                        "no effect on OXPHOS or CoA pathways. Safe in pan-OXPHOS deficiency. "
                        "Dose: 10-60 mg/kg/day; excellent tolerability in infants. "
                        "Alternatives: lamotrigine (LTG), topiramate (TPM). NEVER VPA.",
    },
    {
        "intervention": "Respiratory support (NIV / BiPAP)",
        "level":        "Level B — 60% of patients",
        "reason":       "Progressive respiratory compromise in 60% of MTFMT patients due to pan-OXPHOS "
                        "deficiency in respiratory muscles + brainstem respiratory centres (Leigh lesions). "
                        "Early NIV/BiPAP reduces respiratory muscle work and improves gas exchange. "
                        "Continuous SpO2 + end-tidal CO2 monitoring; low threshold for intubation during "
                        "acute respiratory infections. Sevoflurane ONLY for any anaesthesia — NOT propofol.",
    },
    {
        "intervention": "Folate supplementation (folinic acid)",
        "level":        "Level C — Disease-specific",
        "reason":       "MTFMT uses N10-formyl-tetrahydrofolate (N10-formyl-THF) as the formyl-group donor "
                        "for mt-tRNA formylation. Supplementing with folinic acid (5-formyl-THF, leucovorin) "
                        "may increase the substrate availability for residual MTFMT activity. "
                        "This is a disease-specific rationale not present in other mito diseases; "
                        "dose: 2-5 mg/day folinic acid in selected cases.",
    },
]

# ── DDx matrix ────────────────────────────────────────────────────────────────
DDX_MATRIX = [
    {
        "vs":          "FASTKD2",
        "key":         "CI+CIV deficient (CIII spared) vs MTFMT CI+CIII+CIV all deficient",
        "detail":      "FASTKD2 = combined CI + CIV deficiency; CIII mildly affected in only 40%; "
                       "pattern = CI+CIV, brainstem/cerebellar MRI, early childhood onset. "
                       "MTFMT = pan-OXPHOS CI + CIII + CIV all significantly reduced; basal ganglia "
                       "predominant Leigh MRI. Biochemical enzyme analysis is the critical separator: "
                       "if CIII is significantly deficient → favours MTFMT over FASTKD2.",
    },
    {
        "vs":          "SURF1 (Isolated CIV, Leigh)",
        "key":         "SURF1 = isolated CIV; MTFMT = combined CI+CIII+CIV (NO isolated CIV in MTFMT)",
        "detail":      "SURF1 produces isolated CIV deficiency (CI, CII, CIII all NORMAL). "
                       "MTFMT produces pan-OXPHOS (CI + CIII + CIV all reduced). "
                       "When enzyme analysis shows only CIV deficient with normal CI → SURF1 / isolated CIV family; "
                       "when CI + CIII + CIV all deficient → MTFMT / tRNA/ribosome/mt-translation defect.",
    },
    {
        "vs":          "MTTL1 (m.3243A>G MELAS)",
        "key":         "MATERNAL inheritance + heteroplasmy vs MTFMT AR + biallelic nuclear",
        "detail":      "m.3243A>G (MTTL1, tRNALeu) causes combined OXPHOS deficiency similar to MTFMT. "
                       "Critical separator: MTTL1 = MATERNAL inheritance; heteroplasmy detectable in "
                       "blood/urine/muscle; MELAS features (stroke-like episodes, deafness, diabetes). "
                       "MTFMT = AR nuclear mutation; no heteroplasmy; no maternal pedigree. "
                       "Family history and mtDNA heteroplasmy testing are mandatory before diagnosing MTFMT.",
    },
    {
        "vs":          "POLG (Alpers-Huttenlocher)",
        "key":         "POLG = mtDNA depletion + severe hepatopathy; MTFMT = no depletion, mild hepatopathy",
        "detail":      "POLG mutations cause mtDNA depletion (low copy number in muscle/liver), severe "
                       "progressive hepatopathy (Alpers syndrome), and VPA-triggered hepatic failure. "
                       "MTFMT = normal mtDNA copy number; hepatopathy absent or mild. "
                       "mtDNA quantitation (qPCR) is the key biochemical separator. VPA danger applies to both.",
    },
    {
        "vs":          "TACO1 (COX assembly factor, Isolated CIV)",
        "key":         "TACO1 = isolated CIV (COXPD4); MTFMT = pan-OXPHOS including CI and CIII",
        "detail":      "TACO1 (Translational Activator of COX I) specifically promotes MT-CO1 translation "
                       "→ isolated CIV deficiency. MTFMT affects all 13 mt-encoded proteins → pan-OXPHOS. "
                       "Both are AR nuclear mt-translation regulatory genes. Enzyme analysis separates: "
                       "isolated CIV → TACO1/COX14/COA3/PET100 family; pan-deficiency → MTFMT/tRNA genes.",
    },
    {
        "vs":          "TK2 (mtDNA depletion, myopathic)",
        "key":         "TK2 = myopathic mtDNA depletion, CK very high; MTFMT = encephalopathic, CK normal",
        "detail":      "TK2 mutations cause myopathic mtDNA depletion with very high CK, proximal myopathy, "
                       "respiratory failure, minimal encephalopathy; Leigh syndrome ABSENT. "
                       "MTFMT causes encephalopathic Leigh syndrome with mild CK; no depletion. "
                       "Distribution of involvement (brain vs muscle) and CK level separate these.",
    },
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

        null_null   = "biallelic null" in geno.lower()
        compound    = "compound" in geno.lower()
        missense    = "arg237" in geno.lower() or "leu261" in geno.lower()
        hypomorphic = "c.626" in geno.lower() and "compound" not in geno.lower()

        # Severity tiers
        if null_null:
            onset_mo  = rng.randint(0, 3)
            lactate   = round(rng.uniform(14.0, 22.0), 1)
            ci_pct    = rng.randint(5, 15)
            ciii_pct  = rng.randint(8, 18)
            civ_pct   = rng.randint(12, 22)
            leigh     = True
            survived  = rng.random() < 0.25
        elif compound:
            onset_mo  = rng.randint(3, 12)
            lactate   = round(rng.uniform(10.0, 18.0), 1)
            ci_pct    = rng.randint(15, 30)
            ciii_pct  = rng.randint(18, 35)
            civ_pct   = rng.randint(25, 42)
            leigh     = rng.random() < 0.85
            survived  = rng.random() < 0.48
        elif missense:
            onset_mo  = rng.randint(3, 10)
            lactate   = round(rng.uniform(9.0, 16.0), 1)
            ci_pct    = rng.randint(18, 32)
            ciii_pct  = rng.randint(20, 38)
            civ_pct   = rng.randint(30, 48)
            leigh     = rng.random() < 0.80
            survived  = rng.random() < 0.52
        else:   # hypomorphic splice only
            onset_mo  = rng.randint(6, 24)
            lactate   = round(rng.uniform(5.5, 11.0), 1)
            ci_pct    = rng.randint(28, 48)
            ciii_pct  = rng.randint(32, 52)
            civ_pct   = rng.randint(42, 60)
            leigh     = rng.random() < 0.65
            survived  = rng.random() < 0.70

        seizures = rng.random() < (0.52 if (null_null or compound) else 0.38)
        feeding  = rng.random() < (0.82 if (null_null or compound) else 0.60)
        resp     = rng.random() < (0.72 if (null_null or compound) else 0.48)
        sex      = rng.choice(["M", "F"])

        patients.append({
            "id":            f"MTFMT-{i+1:03d}",
            "sex":           sex,
            "genotype":      geno,
            "onset_mo":      onset_mo,
            "lactate_mM":    lactate,
            "ci_pct":        ci_pct,
            "ciii_pct":      ciii_pct,
            "civ_pct":       civ_pct,
            "leigh_mri":     "Yes" if leigh    else "No",
            "hcm":           "No",
            "seizures":      "Yes" if seizures  else "No",
            "feeding_ng":    "Yes" if feeding   else "No",
            "resp_support":  "Yes" if resp      else "No",
            "survived_2yr":  "Yes" if survived  else "No",
        })
    return patients


_COHORT = _generate_cohort()


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    avg_lactate = round(sum(p["lactate_mM"] for p in _COHORT) / N_PATIENTS, 1)
    avg_ci      = round(sum(p["ci_pct"]     for p in _COHORT) / N_PATIENTS, 1)
    avg_ciii    = round(sum(p["ciii_pct"]   for p in _COHORT) / N_PATIENTS, 1)
    avg_civ     = round(sum(p["civ_pct"]    for p in _COHORT) / N_PATIENTS, 1)
    leigh_pct   = round(100 * sum(1 for p in _COHORT if p["leigh_mri"]    == "Yes") / N_PATIENTS)
    seiz_pct    = round(100 * sum(1 for p in _COHORT if p["seizures"]     == "Yes") / N_PATIENTS)
    resp_pct    = round(100 * sum(1 for p in _COHORT if p["resp_support"] == "Yes") / N_PATIENTS)
    surv_pct    = round(100 * sum(1 for p in _COHORT if p["survived_2yr"] == "Yes") / N_PATIENTS)

    return {
        "disease":      "Leigh Syndrome / Combined OXPHOS Deficiency due to MTFMT Deficiency "
                        "(mt-tRNA Formylation Defect)",
        "omim_gene":    "611766",
        "omim_disease": "614947",
        "gene":         "MTFMT",
        "alias":        "MTFMT (Mitochondrial Methionyl-tRNA Formyltransferase; also HSMTFMT, FMT) — "
                        "formylates the mitochondrial initiator met-tRNA (mt-tRNAiMet) using "
                        "N10-formyl-tetrahydrofolate (N10-formyl-THF) as formyl-group donor; "
                        "N-formyl-Met-tRNAiMet is MANDATORY for the translation initiation of all "
                        "13 mt-encoded OXPHOS subunits; enzyme class: N10-formyltetrahydrofolate-dependent "
                        "methionyl-tRNA formyltransferase (EC 2.1.2.9)",
        "chromosome":   "15q22.31",
        "inheritance":  "Autosomal Recessive (AR) — biallelic LOF",
        "protein":      "376 aa; ~41.6 kDa; mitochondrial matrix (NO transmembrane helix); "
                        "MTS ~29 aa (cleaved after import); catalytic domain contains N10-formyl-THF "
                        "binding pocket and mt-tRNAiMet anticodon stem-loop contact site; "
                        "functions as a HOMODIMER; strictly mitochondria-specific formyltransferase "
                        "(distinct from cytoplasmic initiator tRNA formylation pathway)",
        "onset":        "Late infancy to early childhood — 6 to 24 months typical; biallelic null alleles "
                        "can present neonatally (birth to 3 months); splice hypomorphic alleles may delay "
                        "onset to 12-24 months. NOT uniformly neonatal — key DDx vs COX8A, PET100, UQCC2.",
        "mechanism":    "MTFMT LOF → loss of N-formylation of mt-Met-tRNAiMet (initiator tRNA) → "
                        "N-formyl-Met-tRNAiMet absent → mitoribosomal translation initiation for ALL "
                        "13 mt-encoded proteins severely impaired → pan-OXPHOS deficiency: "
                        "CI (7 mt-encoded subunits: MT-ND1/2/3/4/4L/5/6) most severely affected; "
                        "CIII (1 subunit: MT-CYB) significantly reduced; CIV (3 subunits: MT-CO1/2/3) "
                        "moderately reduced; CV (2 subunits: MT-ATP6/8) variably reduced. "
                        "CII is nuclear-encoded → NORMAL. Pan-OXPHOS = key diagnostic clue.",
        "assembly_pathway": "Mitochondrial translation initiation — MTFMT formylates mt-tRNAiMet using "
                            "N10-formyl-THF → N-formyl-Met-tRNAiMet binds to mt-small ribosomal subunit "
                            "(mt-SSU) P-site with IF2mt (MTIF2) → forms 55S mt-initiation complex at "
                            "mRNA AUG start codon → translation of all 13 mt-encoded OXPHOS subunits "
                            "depends on this formylated initiator. MTFMT is the gatekeeper for "
                            "mt-translation initiation — deficiency propagates to ALL mt-encoded proteins.",
        "biochemical_fingerprint": "Pan-OXPHOS deficiency: CI 15-45% residual (most severely affected, "
                                   "7 mt-encoded subunits); CIII 20-50% residual (1 mt-encoded subunit, "
                                   "MT-CYB); CIV 25-55% residual (3 mt-encoded subunits, MT-CO1-3); "
                                   "CII NORMAL (entirely nuclear-encoded). "
                                   "KEY DISTINCTION: CI + CIII + CIV all significantly reduced = "
                                   "mt-translation defect (MTFMT, mt-tRNA genes, mitoribosome diseases). "
                                   "CI+CIV only (CIII preserved) favours FASTKD2 or mt-LSU diseases. "
                                   "Isolated CI = NDUF* family. Isolated CIV = SURF1/COX14/COA3 family.",
        "cardinal_feature": "Pan-OXPHOS deficiency (CI + CIII + CIV all significantly reduced, CII NORMAL) "
                            "+ Leigh-like MRI (basal ganglia + brainstem predominant) + AR inheritance + "
                            "late infancy onset (6-24 months) + NO HCM — distinguishes MTFMT from "
                            "isolated-complex diseases; WES/WGS confirms biallelic MTFMT variants; "
                            "mtDNA copy number NORMAL (distinguishes from POLG/mtDNA depletion); "
                            "maternal inheritance ABSENT (distinguishes from m.3243A>G MTTL1/MELAS).",
        "cohort_size":          N_PATIENTS,
        "avg_lactate_mM":       avg_lactate,
        "avg_ci_residual_pct":  avg_ci,
        "avg_ciii_residual_pct": avg_ciii,
        "avg_civ_residual_pct": avg_civ,
        "kpis": {
            "leigh_mri_pct":     leigh_pct,
            "seizures_pct":      seiz_pct,
            "resp_support_pct":  resp_pct,
            "survived_2yr_pct":  surv_pct,
            "hcm_pct":           0,
            "tubulopathy_pct":   0,
            "mtdna_depletion":   "No",
            "cii_status":        "Normal",
        },
        "key_ddx_negatives": [
            "NO HCM — key DDx vs SCO2 (100%), COA5 (88%), COA6 (90%) — cardiomyopathy absence rules these out",
            "NO Fanconi tubulopathy — key DDx vs COX10 (65%) — renal tubular involvement absent in MTFMT",
            "NO maternal inheritance — key DDx vs m.3243A>G MTTL1 / MELAS — confirm: check maternal pedigree + blood heteroplasmy",
            "NO mtDNA depletion — key DDx vs POLG (Alpers depletion) — mtDNA copy number NORMAL in MTFMT (nuclear formylation defect)",
            "NO isolated CIV — key DDx vs SURF1/PET100/COX14/COA3 — MTFMT has CI + CIII + CIV all deficient (NOT isolated CIV)",
            "NO isolated CI — key DDx vs NDUF* family — MTFMT has CIII and CIV also deficient (NOT isolated CI)",
            "NO GRACILE triad — key DDx vs BCS1L — MTFMT has no iron overload, aminoaciduria, or cholestasis",
            "CII NORMAL — diagnostic clue: if CII also deficient → consider SDH-related or coenzyme Q disorders",
        ],
        "key_contrasts": {
            "MTFMT_vs_FASTKD2":  "FASTKD2 = CI + CIV deficient (CIII largely spared, mt-rRNA processing); "
                                  "MTFMT = CI + CIII + CIV all deficient (mt-tRNA formylation, affects all 13 mt-encoded proteins). "
                                  "MRI: FASTKD2 = cerebellar/brainstem; MTFMT = BG + brainstem (more classic Leigh). "
                                  "Biochemical enzyme panel is the primary separator.",
            "MTFMT_vs_MTTL1":    "Both cause pan-OXPHOS deficiency. Separator: MTTL1 m.3243A>G = MATERNAL inheritance, "
                                  "heteroplasmy, MELAS phenotype (stroke-like episodes, deafness, diabetes). "
                                  "MTFMT = AR nuclear, no heteroplasmy, no MELAS features.",
            "MTFMT_vs_SURF1":    "SURF1 = strictly isolated CIV (CI and CIII NORMAL) + classic Leigh (BG + PAG 95%). "
                                  "MTFMT = pan-OXPHOS (CI + CIII + CIV all reduced). "
                                  "CI normality is the critical clue pointing to isolated CIV disease family.",
            "MTFMT_vs_POLG":     "Both AR, both Leigh-like, both pan-OXPHOS. Separator: POLG = mtDNA DEPLETION "
                                  "(low copy number on qPCR), severe hepatopathy (Alpers). "
                                  "MTFMT = normal mtDNA copy number, absent/mild hepatopathy.",
        },
    }


def get_breakdown():
    avg_lactate = round(sum(p["lactate_mM"] for p in _COHORT) / N_PATIENTS, 1)
    avg_ci      = round(sum(p["ci_pct"]     for p in _COHORT) / N_PATIENTS, 1)
    avg_ciii    = round(sum(p["ciii_pct"]   for p in _COHORT) / N_PATIENTS, 1)
    avg_civ     = round(sum(p["civ_pct"]    for p in _COHORT) / N_PATIENTS, 1)

    geno_counts = {}
    for p in _COHORT:
        geno_counts[p["genotype"]] = geno_counts.get(p["genotype"], 0) + 1
    genotype_dist = [
        {"genotype": g, "n": n, "pct": round(100 * n / N_PATIENTS)}
        for g, n in sorted(geno_counts.items(), key=lambda x: -x[1])
    ]

    feature_prev = [
        {"feature": feat, "pct": pct, "n": round(N_PATIENTS * pct / 100), "note": note}
        for feat, pct, note in FEATURES
    ]

    severe_pts  = [p for p in _COHORT if "null" in p["genotype"].lower() or "compound" in p["genotype"].lower() or "arg237" in p["genotype"].lower()]
    mild_pts    = [p for p in _COHORT if p not in severe_pts]
    sev_surv    = round(100 * sum(1 for p in severe_pts if p["survived_2yr"] == "Yes") / max(len(severe_pts), 1))
    mild_surv   = round(100 * sum(1 for p in mild_pts   if p["survived_2yr"] == "Yes") / max(len(mild_pts),   1))

    return {
        "genotype_dist":     genotype_dist,
        "feature_prev":      feature_prev,
        "patient_table":     _COHORT,
        "variants":          VARIANTS,
        "contraindications": CONTRAINDICATIONS,
        "treatments":        TREATMENTS,
        "ddx_matrix":        DDX_MATRIX,
        "outcome": {
            "null_allele_2yr_survival_pct":     sev_surv,
            "hypomorphic_allele_2yr_survival_pct": mild_surv,
            "note": "2-year survival stratified by allele severity. Biallelic null/severe missense alleles "
                    "have worst prognosis (neonatal-early infantile onset, pan-OXPHOS collapse). "
                    "Hypomorphic splice alleles (c.626C>T homozygous) retain 25% residual formylation "
                    "and have substantially better 2-year survival.",
        },
        "avg_lactate_mM":       avg_lactate,
        "avg_ci_residual_pct":  avg_ci,
        "avg_ciii_residual_pct": avg_ciii,
        "avg_civ_residual_pct": avg_civ,
        "biomarker_summary": {
            "lactate":  f"{avg_lactate} mM (mean; crisis >18 mM in null alleles; pan-OXPHOS drives higher baseline than isolated-complex diseases)",
            "ci_pct":   f"{avg_ci}% residual CI (mean; 5-48% range; most severely affected complex in MTFMT)",
            "ciii_pct": f"{avg_ciii}% residual CIII (mean; 8-52% range; MT-CYB translation also impaired)",
            "civ_pct":  f"{avg_civ}% residual CIV (mean; 12-60% range; relatively better than CI but still significantly deficient)",
            "cii":      "NORMAL — CII is entirely nuclear-encoded (no mt-translation required); CII normality confirms the defect is mt-translational",
            "mtdna":    "NORMAL copy number — mtDNA qPCR is NORMAL in MTFMT (formylation defect, NOT replication defect). Rules out POLG/TK2/DGUOK.",
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "MTFMT (Mitochondrial Methionyl-tRNA Formyltransferase, OMIM *611766) encodes a "
            "376-amino-acid mitochondrial matrix enzyme (~41.6 kDa) responsible for N-formylation "
            "of the mitochondrial initiator methionyl-tRNA (mt-tRNAiMet). The reaction: "
            "Met-tRNAiMet + N10-formyl-THF → N-formyl-Met-tRNAiMet + THF. "
            "The product (fMet-tRNAiMet) is essential for initiating the translation of ALL 13 "
            "mitochondrially-encoded OXPHOS subunits by the mitoribosome. "
            "MTFMT contains a 29-aa N-terminal mitochondrial targeting sequence (MTS) cleaved "
            "after import; it functions as a homodimer in the matrix with no TM helix. "
            "EC class: 2.1.2.9 (N10-formyltetrahydrofolate-dependent methionyl-tRNA formyltransferase). "
            "Human MTFMT is strictly mitochondria-specific and evolutionarily conserved (bacterial homologs "
            "perform the analogous fMet-tRNA formylation for prokaryotic translation initiation)."
        ),
        "pathomechanism": (
            "MTFMT LOF → absent N-formyl-Met-tRNAiMet → mitochondrial translation initiation fails for "
            "ALL 13 mt-encoded proteins: "
            "7 CI subunits (MT-ND1, MT-ND2, MT-ND3, MT-ND4, MT-ND4L, MT-ND5, MT-ND6); "
            "1 CIII subunit (MT-CYB); "
            "3 CIV subunits (MT-CO1, MT-CO2, MT-CO3); "
            "2 CV subunits (MT-ATP6, MT-ATP8). "
            "Result: pan-OXPHOS deficiency where CI is most severely affected (7 mt-encoded subunits; "
            "most vulnerable to mt-translation failure), followed by CIII (1 subunit), then CIV (3 subunits). "
            "Complex II (CII) is entirely nuclear-encoded → NORMAL — this CII normality is a critical "
            "diagnostic clue distinguishing mt-translation defects from primary CII mutations (SDH genes). "
            "The severity of residual OXPHOS activity correlates with the degree of residual formylation "
            "activity (hypomorphic splice alleles retain ~25% → attenuated pan-deficiency; null alleles "
            "→ near-zero formylation → catastrophic collapse of all mt-encoded OXPHOS subunits)."
        ),
        "pan_oxphos_diagnostic_significance": (
            "CRITICAL DIAGNOSTIC CLUE — Pan-OXPHOS deficiency (CI + CIII + CIV all significantly reduced, "
            "CII NORMAL) narrows the differential to mt-translation defects: "
            "(1) Nuclear-encoded mt-translation genes: MTFMT (formylation), MTIF2 (initiation factor), "
            "TACO1 (COX-I specific), mt-tRNA aminoacyl-tRNA synthetase genes (DARS2, RARS2, YARS2, etc.), "
            "mt-ribosome structural subunits (MRPS*, MRPL* genes). "
            "(2) Mitochondrial DNA mutations: mt-tRNA genes (MTTL1 m.3243A>G, MTTx mutations), "
            "mt-rRNA, large-scale deletions. "
            "(3) mtDNA depletion: POLG, TWNK, TK2, DGUOK (reduced mtDNA template → reduced mt-translation). "
            "KEY SEPARATOR: "
            "- Maternal inheritance → mt-tRNA or mt-rRNA mutation (MTTL1, other mt-tRNA genes). "
            "- AR nuclear + normal mtDNA copy number → MTFMT or nuclear mt-translation gene. "
            "- AR nuclear + low mtDNA copy number → mtDNA depletion syndrome (POLG, TK2, DGUOK family)."
        ),
        "folinic_acid_rationale_unique_to_mtfmt": (
            "MTFMT is UNIQUE among mitochondrial diseases in having a specific substrate supplementation "
            "rationale: MTFMT uses N10-formyl-tetrahydrofolate (N10-formyl-THF) as the formyl-group donor. "
            "Supplementing folinic acid (5-formyl-THF, leucovorin — a reduced folate that is converted "
            "to N10-formyl-THF intracellularly) may increase substrate availability for residual MTFMT "
            "activity in hypomorphic alleles. This is analogous to cofactor supplementation in other "
            "inborn errors where the enzyme retains partial activity: riboflavin for riboflavin-responsive "
            "CI deficiency; biotin for biotinidase deficiency. The evidence for folinic acid in MTFMT "
            "is currently Level C (case reports/rationale), but the biochemical logic is sound. "
            "No other mitochondrial OXPHOS deficiency has a direct folate-supplementation rationale."
        ),
        "mtdna_copy_normal_key_ddx": (
            "MTFMT-related disease shows NORMAL mtDNA copy number on quantitative PCR of muscle/liver. "
            "This is the most important separator from mtDNA depletion syndromes (POLG, TWNK, DGUOK, "
            "TK2, SUCLA2, SUCLG1, RRM2B, MPV17, FBXL4 — all cause mtDNA depletion). "
            "MTFMT is a formylation defect (enzymatic LOF in the matrix) — it does NOT affect mtDNA "
            "replication, maintenance, or copy number control. "
            "Protocol: When pan-OXPHOS deficiency is found in muscle biopsy, always: "
            "(1) Quantitate mtDNA copy number (qPCR muscle + liver) → if LOW: depletion syndrome → POLG/TK2/DGUOK family. "
            "(2) Test blood/urine for mtDNA heteroplasmy → if positive: mt-tRNA mutation (m.3243A>G etc.). "
            "(3) If mtDNA copy number NORMAL + no heteroplasmy + AR pedigree → prioritise MTFMT on WES/WGS panel. "
            "Getting this algorithm right avoids misdiagnosis and reaches MTFMT sooner."
        ),
        "vpa_triple_danger_mtfmt": (
            "VPA (VALPROIC ACID / VALPROATE) is triply dangerous in MTFMT: "
            "(1) VPA sequesters free CoA → depletes mitochondrial CoA pool → blocks β-oxidation "
            "and pyruvate utilisation → worsens lactic acidosis on top of pan-OXPHOS deficiency. "
            "(2) VPA inhibits POLG → mtDNA strand-break repair impaired → POLG failure in MTFMT patients "
            "compounds mt-genome instability (even though MTFMT itself is a formylation defect, not "
            "a replication defect, POLG inhibition adds further OXPHOS impairment). "
            "(3) VPA is hepatotoxic; combined with pan-OXPHOS hepatic impairment in MTFMT "
            "→ risk of acute hepatic failure (particularly dangerous in patients with subclinical "
            "mitochondrial hepatopathy). "
            "Use LEVETIRACETAM exclusively as first-line AED. If LEV insufficient: lamotrigine or "
            "topiramate. NEVER valproate in any confirmed or suspected mitochondrial disease."
        ),
        "references": [
            {
                "citation": "Tucker EJ et al. Am J Hum Genet. 2011;89(2):167-173.",
                "note":     "First description of biallelic MTFMT mutations causing Leigh syndrome with "
                            "combined OXPHOS deficiency. Identified the c.626C>T splice-altering Western "
                            "European founder variant and its hypomorphic nature (~25% residual formylation). "
                            "Established MTFMT as the human mitochondrial initiator tRNA formyltransferase "
                            "and confirmed pan-OXPHOS deficiency as the biochemical fingerprint.",
            },
            {
                "citation": "Neeve VC et al. J Med Genet. 2013;50(6):399-406.",
                "note":     "Clinical expansion of MTFMT-related disease phenotype in an independent "
                            "European cohort. Confirmed c.626C>T as the predominant Western European "
                            "founder allele. Documented variable expressivity with c.626C>T homozygous "
                            "patients showing milder presentations (residual formylation in splice "
                            "hypomorphic allele). Detailed MRI characterisation (BG + brainstem Leigh).",
            },
            {
                "citation": "Haack TB et al. Am J Hum Genet. 2012;90(2):314-320.",
                "note":     "Broader mitochondrial translation defect discovery panel including MTFMT. "
                            "Confirmed pan-OXPHOS deficiency pattern (CI + CIII + CIV) as the biochemical "
                            "hallmark of mt-translation factor deficiencies. Differential diagnosis "
                            "framework for nuclear vs mtDNA-encoded translation defects.",
            },
            {
                "citation": "Koc EC, Spremulli LL. J Biol Chem. 2002;277(38):35541-35549.",
                "note":     "Biochemical characterisation of human mitochondrial translation initiation. "
                            "Demonstrated requirement for N-formyl-Met-tRNAiMet in human mt-ribosome "
                            "initiation complex formation (IF2mt/MTIF2 + fMet-tRNA + mt-SSU). "
                            "Provided mechanistic basis for understanding MTFMT deficiency consequences.",
            },
        ],
        "management_summary": (
            "MTFMT management — pan-OXPHOS deficiency protocol: "
            "(1) NEVER fast — continuous GIR 6-8 mg/kg/min; NG tube early; sick-day protocol essential. "
            "(2) Thiamine (B1) 10-20 mg/kg/day IV MANDATORY empirically — exclude SLC19A3 (BTBGD). "
            "(3) Biotin 10-20 mg/day MANDATORY empirically — exclude BTD/HLCS. "
            "(4) Riboflavin (B2) 100-400 mg/day — FAD cofactor for CI and CIII. "
            "(5) Folinic acid 2-5 mg/day — disease-specific: supplies N10-formyl-THF substrate for "
            "    residual MTFMT activity in hypomorphic alleles. "
            "(6) CoQ10 ubiquinol 10-30 mg/kg/day — mobile electron carrier support. "
            "(7) Seizures: LEV ONLY — NO VPA (ABSOLUTE CI — triple danger in pan-OXPHOS deficiency). "
            "(8) Respiratory: NIV/BiPAP in 60% — early respiratory monitoring essential. "
            "(9) Anaesthesia: SEVOFLURANE ONLY — propofol ABSOLUTELY FORBIDDEN (PRIS). "
            "(10) Linezolid ABSOLUTELY FORBIDDEN — mt-ribosome block doubly toxic in mt-translation defect. "
            "(11) Metformin ABSOLUTELY FORBIDDEN — direct CI inhibitor in CI-deficient patient. "
            "(12) WES/WGS: biallelic MTFMT variants confirm diagnosis; mtDNA quantitation NORMAL "
            "     (distinguishes from POLG/TK2/DGUOK depletion syndromes); blood/urine heteroplasmy "
            "     testing to exclude m.3243A>G MTTL1 (if maternal pedigree or MELAS features). "
            "(13) Muscle biopsy: pan-OXPHOS deficiency + COX/SDH staining (mosaic COX-negative fibres); "
            "     electron microscopy: mitochondrial proliferation; no structural mt abnormality. "
            "(14) Trial of N-acetyl-L-cysteine (NAC) as antioxidant — emerging evidence in some OXPHOS defects."
        ),
    }
