#!/usr/bin/env python3
"""MT-CYB — Mitochondrially Encoded Cytochrome b / Isolated Complex III (CIII) Deficiency
Most Common mtDNA Cause of Isolated CIII Deficiency — Exercise Intolerance / Myopathy (Mild)
to Leigh Syndrome (Severe) — Maternal Inheritance — Heteroplasmy.

MT-CYB (OMIM *516020) encodes the 380-amino-acid, 42.7 kDa cytochrome b subunit of Complex
III (cytochrome bc1 complex), the ONLY mitochondrially-encoded structural subunit of CIII.
Cytochrome b contains both the Qo-site (quinol oxidation / outer site, where ubiquinol is
oxidised and electrons transferred to RISP/UQCRFS1) and the Qi-site (quinone reduction /
inner site, where ubiquinone is reduced) — making it the CATALYTIC CORE of CIII.

  MT-CYB gene       OMIM *516020
  Disease           Isolated CIII Deficiency (mitochondrial / maternal inheritance)
  Protein           380 aa, 42.7 kDa; 8 transmembrane helices; heme bL (Qi-site) + heme bH (Qo-site)
  Genome            Mitochondrial DNA (mtDNA), positions 14747–15887 (rCRS)
  Inheritance       MATERNAL — mtDNA; heteroplasmic (rarely homoplasmic)
  Phenotype         Exercise intolerance/myopathy (adult-onset, mild, ~60%) to Leigh syndrome
                    (infantile, severe, ~40%) — wide clinical spectrum from heteroplasmy level

UNIQUE MOLECULAR POSITION — ONLY mtDNA-ENCODED CIII STRUCTURAL SUBUNIT:
  All 10 other CIII structural subunits (UQCRC1, UQCRC2, UQCRB, UQCRQ, UQCRH, UQCR10,
  UQCR11, UQCRFS1/RISP, CYC1, UQCRB) are NUCLEAR-encoded, AR biallelic.
  MT-CYB is the sole MITOCHONDRIALLY-encoded structural subunit — the catalytic heart of CIII
  containing BOTH the Qo-site and Qi-site within a single polypeptide chain.
  This explains:
  (1) MATERNAL inheritance (vs AR biallelic for all nuclear CIII defects)
  (2) HETEROPLASMY — variable mutant mtDNA load → clinical spectrum
  (3) TISSUE HETEROPLASMY — heteroplasmy level differs in blood vs muscle;
      muscle biopsy (>80% mutant) required for diagnosis in mild phenotypes
  (4) EXERCISE-SPECIFIC phenotype — high mutant mtDNA shifts in exercise;
      attacks of myoglobinuria with exercise (DDx from nuclear CIII defects)

PATHOPHYSIOLOGY:
  Cytochrome b contains the proton channel for the Q-cycle:
  • Qo-site (outer; heme bH, 2Fe-2S RISP interface): ubiquinol (CoQH2) is oxidised;
    one electron to RISP (Fe-S cluster N2) → CYC1 → cytochrome c → CIV;
    one electron to heme bL → heme bH (transmembrane Q-cycle)
  • Qi-site (inner; heme bL): ubiquinone (CoQ) is reduced using the heme bH electron
    and a proton from the matrix → CoQH2; second half of Q-cycle
  Loss-of-function → Q-cycle arrest → CIII activity 0–25% (varies with heteroplasmy) →
  NADH/FADH2 backlog → lactic acidosis + reduced ATP synthesis.
  In mild heteroplasmy (muscle 50–80% mutant): CIII sufficient at rest, fails under
  exercise demand → exercise intolerance, myalgia, CK elevation, myoglobinuria.
  In high heteroplasmy (>90% mutant): severe neonatal/infantile CIII deficiency → Leigh.

PHENOTYPE SPECTRUM:
  MILD (adult-onset exercise intolerance — ~60% of MT-CYB disease):
    • Exercise intolerance (100% of mild phenotype): exertional fatigue after moderate activity
    • Myalgia + cramps: 90%
    • CK elevation (during/after attacks): 85% (up to 10–50× ULN)
    • Myoglobinuria/rhabdomyolysis: 45% (recurrent dark urine with exercise)
    • Normal between episodes: 80%
    • Serum lactate mildly elevated at rest; dramatically elevated post-exercise
    • Maternal family members variably affected (heteroplasmy variable per relative)
  SEVERE (infantile/childhood — ~40% of MT-CYB disease):
    • Leigh syndrome or Leigh-like: 65% (severe infantile)
    • Lactic acidosis (severe, neonatal): 92%
    • Hypotonia: 88%
    • Encephalopathy: 80%
    • Developmental delay: 85%
    • Cardiomyopathy: 28% (less than SCO2/COA6 but present in severe cases)
    • Seizures: 38%

DISTINGUISHING FEATURES vs OTHER CIII DEFICIENCIES (nuclear-encoded):
  vs ALL nuclear CIII defects (UQCRQ/UQCRB/UQCRC1/UQCRC2/CYC1/BCS1L/UQCC1/UQCC2/TTC19):
    • MATERNAL inheritance (not AR) — family history of maternal relatives affected
    • HETEROPLASMY — sequencing shows mutant load typically 50–95%; WES misses mtDNA variants;
      need dedicated mtDNA sequencing / long-read sequencing / MLPA
    • Exercise intolerance in ~60% — rare in nuclear CIII defects (usually severe neonatal)
    • Muscle biopsy: ragged-red fibres (RRF) on Gomori trichrome (COX-negative/CIII-negative) —
      most nuclear CIII defects lack RRF in infants
    • Myoglobinuria — ABSENT in all nuclear-encoded CIII defects (pathognomonic for mtDNA myopathy)
  vs UQCRQ (nuclear, AR):
    • MT-CYB: NO dystonia as cardinal feature (UQCRQ: 75% dystonia cardinal)
    • MT-CYB: NO optic atrophy in mild form (UQCRQ: 42% optic atrophy)
    • MT-CYB: Exercise intolerance / myopathy dominant; UQCRQ: neurodegenerative
  vs UQCRB (nuclear, AR):
    • MT-CYB: NO hypoglycaemia (UQCRB: 65% — DISTINGUISHING); NO hepatopathy prominent
    • MT-CYB: Maternal inheritance; UQCRB: AR biallelic
  vs BCS1L (nuclear, AR, GRACILE/Bjornstad):
    • MT-CYB: NO GRACILE triad (iron overload / aminoaciduria / cholestasis)
    • MT-CYB: NO Bjornstad pili torti; NO SNHL as dominant feature
    • BCS1L: CIII precomplex on BN-PAGE (pathognomonic); MT-CYB: CIII absent/severely reduced
  vs TTC19 (nuclear, AR):
    • MT-CYB: NO psychiatric features (TTC19: 38% psychosis/depression)
    • MT-CYB: Maternal inheritance; TTC19: AR

THERAPY:
  No targeted MT-CYB rescue therapy clinically available.
  Mitochondrial disease supportive care:
  ABSOLUTE contraindications (direct CIII inhibitors / mito toxins):
    Valproate (VPA) — CoA sequestration + POLG inhibition; increased mtDNA depletion risk
    Metformin      — directly inhibits Complex I → additive OXPHOS failure
    Linezolid      — inhibits mt-23S rRNA → blocks MT-CYB translation itself (double-hit)
    Chloramphenicol — same mt-ribosome inhibition as linezolid; reduces residual cytochrome b
    Propofol       — PRIS: direct CIII inhibitor (Qi-site) + β-oxidation blockade;
                     contraindicated especially in severe infant forms
  Ketogenic diet: CONTRAINDICATED in severe infantile (forces NADH/FADH2 via β-oxidation
    into CIII-deficient chain); use with CAUTION in mild adult exercise intolerance only
    under specialist guidance (some reports of exacerbation)
  AVOID triggers: prolonged fasting, strenuous exercise, febrile illness (raise mutant
    mtDNA burden in muscle); bed rest + IV fluids + IV dextrose during crises
  Level C cofactors:
    CoQ10/Ubiquinol — CIII electron carrier substrate; Level C
    Riboflavin (B2) — CI/CIII FAD-dependent cofactor; Level C
    Thiamine (B1)   — empiric PDH/TCA support; Level C mandatory empiric
    Biotin          — empiric (rule out BTD); Level C
  CRISIS management:
    IV dextrose (GIR 6–8 mg/kg/min) for severe lactic acidosis + inability to eat
    IV NaHCO₃ — target pH >7.2
    NIV/BiPAP — respiratory failure in severe cases
    Creatine monohydrate (5–10 g/day adults) — modest CK/exercise tolerance benefit
    Avoid prolonged immobility; early physiotherapy in severe forms
"""

import random

SEED = 735

# Pathogenic variants in MT-CYB causing isolated CIII deficiency
# Sources: Andreu 1999 NEJM, Hanna 2002 Neurology, Sarzi 2008 Hum Mol Genet,
# Moran 2010 JIMD, Blakely 2005 Brain, Lamantea 2002 Brain, Keightley 2000
VARIANTS = [
    {
        "change": "p.Gly34Glu",
        "cdna": "c.101G>A",
        "domain": "CD helix / heme bL-proximal (Qi-site vicinity)",
        "type": "Missense",
        "severity": "Mild",
        "phenotype": "Exercise intolerance / myopathy",
        "heteroplasmy_range": "55–80% in muscle",
        "notes": "Recurrent; Gly-34 is at the heme bL-proximal CD helix; severe reduction of Qi-site function; exercise intolerance + CK elevation; Andreu 1999 NEJM-related spectrum",
        "allele_freq_pct": 25,
    },
    {
        "change": "p.Val356Met",
        "cdna": "c.1066G>A",
        "domain": "H helix / Qo-site peripheral",
        "type": "Missense",
        "severity": "Mild–Moderate",
        "phenotype": "Exercise intolerance ± myoglobinuria",
        "heteroplasmy_range": "60–88% in muscle",
        "notes": "H-helix Val-356 contacts the Qo-site loop; partial RISP docking impairment; exercise-triggered myoglobinuria; adult onset typical",
        "allele_freq_pct": 18,
    },
    {
        "change": "p.Arg368His",
        "cdna": "c.1103G>A",
        "domain": "Qo-site (PEWY loop region)",
        "type": "Missense",
        "severity": "Moderate–Severe",
        "phenotype": "Myopathy / infantile multisystem",
        "heteroplasmy_range": "72–95% in muscle",
        "notes": "Arg-368 at the conserved PEWY loop of the Qo-site; impairs ubiquinol oxidation and RISP docking; intermediate phenotype; Sarzi 2008 Hum Mol Genet spectrum",
        "allele_freq_pct": 15,
    },
    {
        "change": "p.Trp156Arg",
        "cdna": "c.466T>C",
        "domain": "DE helix / heme bH environment",
        "type": "Missense",
        "severity": "Severe",
        "phenotype": "Leigh syndrome / infantile CIII deficiency",
        "heteroplasmy_range": "88–100% (near homoplasmic)",
        "notes": "Trp-156 is in the DE helix adjacent to heme bH; disrupts heme bH binding geometry; severe neonatal/infantile CIII deficiency; Leigh-like MRI; Lamantea 2002 Brain",
        "allele_freq_pct": 20,
    },
    {
        "change": "Nonsense / frameshift / large deletion",
        "cdna": "Multiple — mtDNA deletion or truncating variant",
        "domain": "Variable — mtDNA segment",
        "type": "Null / large deletion",
        "severity": "Severe",
        "phenotype": "Severe infantile / Leigh / neonatal",
        "heteroplasmy_range": "90–100% (near homoplasmic) or large mtDNA deletion",
        "notes": "Truncating point mutations + partial/complete MT-CYB deletions; near-complete loss of cytochrome b → CIII absent on BN-PAGE; Leigh syndrome; severe neonatal lactic acidosis; Hanna 2002 Neurology",
        "allele_freq_pct": 22,
    },
]

_VARIANT_CHOICES = [v["change"] for v in VARIANTS]
_VARIANT_WEIGHTS = [v["allele_freq_pct"] for v in VARIANTS]


def _make_patients():
    rng = random.Random(SEED)

    # Phenotype distribution: ~60% mild (exercise intolerance), ~40% severe (infantile/Leigh)
    patients = []
    for i in range(40):
        pid = f"MTCYB-{i+1:03d}"

        # Draw variant (for mtDNA, patients have a single mtDNA variant at various heteroplasmy)
        variant = rng.choices(_VARIANT_CHOICES, weights=_VARIANT_WEIGHTS, k=1)[0]
        var_obj = next(v for v in VARIANTS if v["change"] == variant)

        # Determine phenotype based on variant severity + heteroplasmy
        is_severe = var_obj["severity"] in ("Severe",) or (var_obj["severity"] == "Moderate–Severe" and rng.random() < 0.6)
        is_moderate = var_obj["severity"] == "Moderate–Severe" and not is_severe

        # Heteroplasmy level (muscle tissue %)
        if is_severe:
            heteroplasmy_pct = rng.uniform(85, 100)
        elif is_moderate:
            heteroplasmy_pct = rng.uniform(70, 90)
        else:
            heteroplasmy_pct = rng.uniform(55, 82)

        # Onset
        if is_severe:
            onset_weeks = rng.randint(0, 12)     # neonatal to 3 months
            phenotype_label = "Leigh syndrome / severe infantile CIII deficiency"
        elif is_moderate:
            onset_weeks = rng.randint(52, 260)   # 1–5 years
            phenotype_label = "Infantile multisystem CIII deficiency"
        else:
            onset_weeks = rng.randint(780, 2600) # 15–50 years (adult)
            phenotype_label = "Adult exercise intolerance / myopathy"

        # CIII activity (varies with heteroplasmy; higher heteroplasmy → lower residual)
        ciii_base = max(2, 28 - 0.25 * heteroplasmy_pct + rng.uniform(-4, 4))
        ciii_activity_pct = round(max(2, min(25, ciii_base)), 1)

        # Lactic acid
        if is_severe:
            lactic_acid = round(rng.uniform(6, 18), 1)
        elif is_moderate:
            lactic_acid = round(rng.uniform(3, 10), 1)
        else:
            lactic_acid_rest = round(rng.uniform(1.5, 4), 1)
            lactic_acid = lactic_acid_rest  # at rest; post-exercise >10 mM

        # Clinical features by phenotype
        exercise_intolerance = not is_severe  # virtually 100% of mild/moderate
        myalgia = not is_severe and rng.random() < 0.90
        ck_elevated = not is_severe and rng.random() < 0.85
        myoglobinuria = not is_severe and rng.random() < 0.45
        lactic_acidosis = is_severe and rng.random() < 0.92 or is_moderate and rng.random() < 0.72
        hypotonia = is_severe and rng.random() < 0.88 or is_moderate and rng.random() < 0.60
        encephalopathy = is_severe and rng.random() < 0.80
        developmental_delay = is_severe and rng.random() < 0.85 or is_moderate and rng.random() < 0.50
        leigh_like_mri = is_severe and rng.random() < 0.65 or is_moderate and rng.random() < 0.20
        seizures = is_severe and rng.random() < 0.38 or is_moderate and rng.random() < 0.15
        cardiomyopathy = is_severe and rng.random() < 0.28
        ragged_red_fibers = not is_severe and rng.random() < 0.48 or is_severe and rng.random() < 0.20
        cox_neg_fibers = not is_severe and rng.random() < 0.55 or is_severe and rng.random() < 0.30
        respiratory_failure = is_severe and rng.random() < 0.45

        # Maternal relatives affected
        maternal_family_affected = rng.random() < 0.68  # heteroplasmy variable

        # Outcome
        if is_severe and ciii_activity_pct < 5:
            outcome_choices = ["Deceased (neonatal)", "Deceased (infantile)", "Alive on ventilator"]
            outcome_wts = [0.45, 0.35, 0.20]
        elif is_severe:
            outcome_choices = ["Deceased (infantile)", "Alive — severe disability", "Alive — moderate disability"]
            outcome_wts = [0.40, 0.35, 0.25]
        elif is_moderate:
            outcome_choices = ["Alive — moderate disability", "Alive — mild disability", "Deceased (childhood)"]
            outcome_wts = [0.55, 0.35, 0.10]
        else:
            outcome_choices = ["Alive — exercise restriction", "Alive — episodic myoglobinuria managed", "Alive — mild disability"]
            outcome_wts = [0.60, 0.28, 0.12]
        outcome = rng.choices(outcome_choices, weights=outcome_wts, k=1)[0]

        sex = rng.choice(["M", "F"])  # equal in mtDNA (unlike X-linked nuclear)

        patients.append({
            "patient_id": pid,
            "sex": sex,
            "onset_weeks": onset_weeks,
            "phenotype": phenotype_label,
            "variant": variant,
            "heteroplasmy_pct_muscle": round(heteroplasmy_pct, 1),
            "ciii_activity_pct": ciii_activity_pct,
            "lactic_acid_mmolL": lactic_acid,
            "exercise_intolerance": exercise_intolerance,
            "myalgia": myalgia,
            "ck_elevated": ck_elevated,
            "myoglobinuria": myoglobinuria,
            "lactic_acidosis": lactic_acidosis,
            "hypotonia": hypotonia,
            "encephalopathy": encephalopathy,
            "developmental_delay": developmental_delay,
            "leigh_like_mri": leigh_like_mri,
            "seizures": seizures,
            "cardiomyopathy": cardiomyopathy,
            "ragged_red_fibers": ragged_red_fibers,
            "cox_neg_fibers": cox_neg_fibers,
            "respiratory_failure": respiratory_failure,
            "maternal_family_affected": maternal_family_affected,
            "outcome": outcome,
            "inheritance": "Maternal (mtDNA — heteroplasmic)",
        })
    return patients


def _cohort_stats(patients):
    n = len(patients)

    def pct(field):
        return round(100 * sum(1 for p in patients if p.get(field)) / n, 1)

    return {
        "n": n,
        "exercise_intolerance_pct": pct("exercise_intolerance"),
        "myalgia_pct": pct("myalgia"),
        "ck_elevated_pct": pct("ck_elevated"),
        "myoglobinuria_pct": pct("myoglobinuria"),
        "lactic_acidosis_pct": pct("lactic_acidosis"),
        "hypotonia_pct": pct("hypotonia"),
        "encephalopathy_pct": pct("encephalopathy"),
        "developmental_delay_pct": pct("developmental_delay"),
        "leigh_like_mri_pct": pct("leigh_like_mri"),
        "seizures_pct": pct("seizures"),
        "cardiomyopathy_pct": pct("cardiomyopathy"),
        "ragged_red_fibers_pct": pct("ragged_red_fibers"),
        "cox_neg_fibers_pct": pct("cox_neg_fibers"),
        "maternal_family_affected_pct": pct("maternal_family_affected"),
        "avg_ciii_activity_pct": round(sum(p["ciii_activity_pct"] for p in patients) / n, 1),
        "avg_lactic_acid_mmolL": round(sum(p["lactic_acid_mmolL"] for p in patients) / n, 1),
        "deceased_pct": round(100 * sum(1 for p in patients if "Deceased" in p["outcome"]) / n, 1),
        "severe_pct": round(100 * sum(1 for p in patients if "Leigh" in p["phenotype"] or "infantile" in p["phenotype"]) / n, 1),
        "mild_adult_pct": round(100 * sum(1 for p in patients if "exercise" in p["phenotype"].lower()) / n, 1),
    }


def get_overview():
    patients = _make_patients()
    stats = _cohort_stats(patients)

    features = [
        {"feature": "Exercise intolerance (mild)", "pct": stats["exercise_intolerance_pct"]},
        {"feature": "Myalgia / cramps (mild)", "pct": stats["myalgia_pct"]},
        {"feature": "CK elevation (during attacks)", "pct": stats["ck_elevated_pct"]},
        {"feature": "Lactic acidosis (severe form)", "pct": stats["lactic_acidosis_pct"]},
        {"feature": "Hypotonia (severe form)", "pct": stats["hypotonia_pct"]},
        {"feature": "Developmental delay (severe)", "pct": stats["developmental_delay_pct"]},
        {"feature": "Encephalopathy (severe)", "pct": stats["encephalopathy_pct"]},
        {"feature": "Leigh-like MRI (severe)", "pct": stats["leigh_like_mri_pct"]},
        {"feature": "Myoglobinuria (mild form)", "pct": stats["myoglobinuria_pct"]},
        {"feature": "Ragged-red fibres (muscle Bx)", "pct": stats["ragged_red_fibers_pct"]},
        {"feature": "COX-negative fibres (muscle Bx)", "pct": stats["cox_neg_fibers_pct"]},
        {"feature": "Seizures (severe form)", "pct": stats["seizures_pct"]},
        {"feature": "Cardiomyopathy (severe)", "pct": stats["cardiomyopathy_pct"]},
        {"feature": "Maternal family history", "pct": stats["maternal_family_affected_pct"]},
    ]

    from collections import Counter
    v_counter = Counter(p["variant"] for p in patients)
    top_variants = [{"variant": k, "count": v} for k, v in v_counter.most_common(5)]

    alerts = [
        "🚫 Valproic acid (VPA) — ABSOLUTE CONTRAINDICATION: CoA sequestration + POLG inhibition; increases mtDNA depletion risk (doubly toxic in MT-CYB patients)",
        "🚫 Metformin — ABSOLUTE CONTRAINDICATION: Complex I inhibitor → additive OXPHOS failure; absolute CI in all CIII deficiency including MT-CYB",
        "🚫 Linezolid — ABSOLUTE CONTRAINDICATION: inhibits mt-23S rRNA translation → directly blocks MT-CYB synthesis itself (double-hit mechanism)",
        "🚫 Chloramphenicol — ABSOLUTE CONTRAINDICATION: mt-ribosome inhibitor; reduces residual cytochrome b translation",
        "🚫 Propofol — ABSOLUTE CONTRAINDICATION: PRIS — directly inhibits CIII Qi-site + β-oxidation blockade; fatal in CIII-deficient patients",
        "⚠️ Exercise intolerance (60% of patients): DISTINGUISHING mtDNA pattern — exercise precipitates lactic crisis + myoglobinuria; avoid strenuous exercise; cool-down protocols mandatory",
        "⚠️ Myoglobinuria/rhabdomyolysis (45%): PATHOGNOMONIC for mtDNA myopathy — dark urine post-exercise; vigorous IV hydration mandatory; avoid NSAIDs; kidney monitoring",
        "⚠️ MATERNAL INHERITANCE: test maternal relatives (variable heteroplasmy); mtDNA sequencing NOT standard WES — need dedicated mtDNA long-read or MLPA",
        "⚠️ Muscle biopsy required in mild phenotype: blood heteroplasmy unreliable (<10% in blood, >70% in muscle common); muscle is gold standard",
        "✅ Creatine monohydrate (5–10 g/day adults): modest improvement in exercise tolerance and CK normalisation (Level C evidence)",
        "✅ IV Dextrose + NaHCO₃: acute lactic crisis management; never fast during intercurrent illness (GIR 6–8 mg/kg/min IV)",
        "✅ CoQ10/Ubiquinol + Riboflavin + Thiamine: mitochondrial cocktail; Level C (no randomised trial)",
        "✅ Sevoflurane preferred over Propofol for anaesthesia; avoid prolonged fasting pre-operatively",
    ]

    return {
        "gene": "MT-CYB",
        "full_name": "Mitochondrially Encoded Cytochrome b",
        "alias": "Cytochrome b / Cytb / MT-CYB (HGNC approved) / Apocytochrome b",
        "omim_gene": "516020",
        "omim_disease": "Isolated CIII deficiency (mitochondrial) — Andreu 1999 NEJM, Hanna 2002 Neurology",
        "disease_name": "Isolated Complex III Deficiency — Mitochondrial (Maternal Inheritance) — Exercise Intolerance to Leigh Syndrome Spectrum",
        "chromosome": "Mitochondrial DNA (mtDNA) — rCRS positions 14747–15887",
        "inheritance": "Maternal (mtDNA) — heteroplasmic (rarely homoplasmic)",
        "protein_size": "380 aa, 42.7 kDa — 8 TM helices — heme bL (Qi) + heme bH (Qo)",
        "protein": {
            "size_aa": 380,
            "kDa": 42.7,
            "tm_helices": 8,
            "localization": "Inner mitochondrial membrane — ONLY mtDNA-encoded CIII structural subunit; spans IMM with 8 TM helices; heme bL at Qi-site (matrix side), heme bH at Qo-site (IMS side)",
            "role": "CATALYTIC CORE of CIII — contains BOTH Qo-site (ubiquinol oxidation / RISP docking / heme bH) and Qi-site (ubiquinone reduction / heme bL); forms the CIII dimer at the bc1 complex core",
            "function": "Cytochrome b provides the entire Q-cycle scaffold: Qo-site oxidises CoQH2 → electrons to RISP (Fe-S cluster) → CYC1 → cytochrome c; Qi-site reduces CoQ → CoQH2 using heme bH electron + matrix proton",
        },
        "ciii_assembly_step": "CATALYTIC structural core of CIII — stabilised earliest by UQCC1/UQCC2 heterodimer; assembly proceeds: MT-CYB + UQCC1/UQCC2 → CYC1/UQCRC1/UQCRC2 core → RISP (BCS1L-inserted) → peripheral subunits (UQCRQ/UQCRB/UQCRH/UQCR10/UQCR11). In MT-CYB deficiency: CIII absent or severely reduced; sub-complexes may partially accumulate depending on mutation severity",
        "bn_page": "CIII absent or severely reduced (<5–25% residual, varies with heteroplasmy level); no CIII precomplex accumulation (DDx BCS1L where precomplex accumulates); CI, CII, CIV typically normal (isolated CIII deficiency)",
        "key_biochemical_features": [
            "Isolated CIII deficiency: residual CIII 2–25% (varies with heteroplasmy; lower heteroplasmy → higher residual)",
            "CI, CII, CIV activities: typically NORMAL (isolated CIII deficiency); ddx from MTFMT/FASTKD2 (combined CI+CIV/CIII+CIV deficiency)",
            "Serum CK: massively elevated during attacks (10–50× ULN) in exercise intolerance phenotype; useful diagnostic trigger",
            "Plasma lactate: mildly elevated at rest in mild form (2–4 mM); severely elevated post-exercise (>10 mM); always severe (>6 mM) in Leigh/infantile form",
            "Lactate:pyruvate (L:P) ratio >20 (mitochondrial block pattern)",
            "Blood heteroplasmy often LOW (<10–30%) in mild exercise intolerance — blood unreliable; MUSCLE tissue heteroplasmy >70% in most symptomatic patients",
            "Muscle biopsy: ragged-red fibres (RRF) ~48% of mild adult phenotype; COX-negative / CIII-negative fibres on histochemistry ~55%; absent in nuclear CIII defects",
            "Myoglobin elevated / myoglobinuria in 45% of mild phenotype during exercise attacks",
            "BN-PAGE: CIII absent or severely reduced; no CIII precomplex (DDx BCS1L); CIII sub-complexes variable",
            "mtDNA sequencing (not WES): required for diagnosis — WES does not reliably detect mtDNA point mutations; long-read sequencing / Sanger confirmation on muscle DNA",
        ],
        "cohort_n": len(patients),
        "seed": SEED,
        "patients": patients[:10],
        "cohort_statistics": stats,
        "cohort_summary_features": features,
        "key_clinical_alerts": alerts,
        "top_variant_counts": top_variants,
        "phenotype_distribution": {
            "adult_exercise_intolerance_pct": stats["mild_adult_pct"],
            "infantile_severe_pct": stats["severe_pct"],
        },
        "onset_distribution": {
            "neonatal_0_to_4wk_pct": round(100 * sum(1 for p in patients if p["onset_weeks"] <= 4) / len(patients), 1),
            "infantile_5_to_52wk_pct": round(100 * sum(1 for p in patients if 5 <= p["onset_weeks"] <= 52) / len(patients), 1),
            "childhood_1_to_15yr_pct": round(100 * sum(1 for p in patients if 52 < p["onset_weeks"] <= 780) / len(patients), 1),
            "adult_over_15yr_pct": round(100 * sum(1 for p in patients if p["onset_weeks"] > 780) / len(patients), 1),
        },
    }


def get_breakdown():
    patients = _make_patients()
    stats = _cohort_stats(patients)
    n = len(patients)

    from collections import Counter
    v_counter = Counter(p["variant"] for p in patients)
    variant_dist = [
        {"variant": k, "count": v, "allele_freq_pct": round(100 * v / n, 1)}
        for k, v in v_counter.most_common()
    ]

    ciii_vals = [p["ciii_activity_pct"] for p in patients]
    lac_vals  = [p["lactic_acid_mmolL"] for p in patients]
    hetero_vals = [p["heteroplasmy_pct_muscle"] for p in patients]

    biochem = {
        "avg_ciii_activity_pct": round(sum(ciii_vals) / n, 1),
        "ciii_below5_pct":   round(100 * sum(1 for v in ciii_vals if v < 5) / n, 1),
        "ciii_5to10_pct":   round(100 * sum(1 for v in ciii_vals if 5 <= v < 10) / n, 1),
        "ciii_10to15_pct":  round(100 * sum(1 for v in ciii_vals if 10 <= v < 15) / n, 1),
        "ciii_15to25_pct":  round(100 * sum(1 for v in ciii_vals if 15 <= v <= 25) / n, 1),
        "avg_lactic_acid_mmolL": round(sum(lac_vals) / n, 1),
        "lactic_above_10_pct": round(100 * sum(1 for v in lac_vals if v > 10) / n, 1),
        "lactic_5_to_10_pct": round(100 * sum(1 for v in lac_vals if 5 <= v <= 10) / n, 1),
        "lactic_below5_pct": round(100 * sum(1 for v in lac_vals if v < 5) / n, 1),
        "avg_heteroplasmy_muscle_pct": round(sum(hetero_vals) / n, 1),
        "heteroplasmy_above85_pct": round(100 * sum(1 for v in hetero_vals if v > 85) / n, 1),
        "heteroplasmy_70to85_pct": round(100 * sum(1 for v in hetero_vals if 70 <= v <= 85) / n, 1),
        "heteroplasmy_below70_pct": round(100 * sum(1 for v in hetero_vals if v < 70) / n, 1),
    }

    from collections import Counter as Ct
    oc = Ct(p["outcome"] for p in patients)
    outcomes = [{"outcome": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in oc.most_common()]

    pc = Ct(p["phenotype"] for p in patients)
    phenotype_dist = [{"phenotype": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in pc.most_common()]

    return {
        "gene": "MT-CYB",
        "all_variants": VARIANTS,
        "variant_distribution": variant_dist,
        "phenotype_distribution": phenotype_dist,
        "biochemistry_distribution": biochem,
        "outcome_distribution": outcomes,
        "cohort_statistics": stats,
        "bn_page_pattern": {
            "finding": "CIII absent or severely reduced (2–25% residual; correlates with muscle heteroplasmy level); CI, CII, CIV activities normal",
            "interpretation": "MT-CYB loss → Q-cycle arrest → CIII activity proportional to residual wild-type MT-CYB (heteroplasmy level); NO CIII precomplex accumulation (DDx BCS1L); isolated CIII deficiency pattern",
            "ddx_value": "BN-PAGE: CIII absent/severely reduced WITHOUT precomplex — DDx from BCS1L (precomplex pathognomonic for BCS1L); RISP absent on immunoblot (BCS1L: RISP absent + precomplex; MT-CYB: CIII absent + NO precomplex); mtDNA sequencing on muscle DNA required (not WES)",
        },
        "immunoblot_pattern": {
            "Cytochrome b (MT-CYB)": "Absent or severely reduced (proportional to heteroplasmy)",
            "UQCRC1 (Core 1)": "Secondarily reduced (scaffold destabilised without cytochrome b core)",
            "UQCRC2 (Core 2)": "Secondarily reduced (parallel to UQCRC1)",
            "RISP / UQCRFS1": "Absent or severely reduced (RISP inserts after cytochrome b core — no scaffold, no RISP)",
            "CYC1 (Cytochrome c1)": "Secondarily reduced",
            "UQCRQ / UQCRB (peripheral)": "Absent or reduced (no core scaffold to anchor peripheral subunits)",
            "CI / CII / CIV subunits": "Normal (isolated CIII deficiency — hallmark of MT-CYB vs MTFMT/FASTKD2)",
        },
        "treatment_uptake": {
            "CoQ10 / Ubiquinol": f"{rng_count(patients, 0.78)} / {n} patients",
            "Thiamine (B1) empiric": f"{rng_count(patients, 0.82)} / {n} patients",
            "Riboflavin (B2)": f"{rng_count(patients, 0.72)} / {n} patients",
            "Creatine monohydrate (adult mild)": f"{rng_count(patients, 0.48)} / {n} patients",
            "IV Dextrose GIR 6–8 (acute crisis)": f"{rng_count(patients, 0.68)} / {n} patients",
            "IV NaHCO₃ (acute lactic acidosis)": f"{rng_count(patients, 0.62)} / {n} patients",
            "NIV / BiPAP (severe form)": f"{rng_count(patients, 0.35)} / {n} patients",
            "LEV (preferred AED if seizures)": f"{rng_count(patients, 0.30)} / {n} patients",
            "Exercise restriction + cooling": f"{rng_count(patients, 0.55)} / {n} patients",
        },
    }


def rng_count(patients, rate):
    return round(rate * len(patients))


def get_definitions():
    return {
        "gene": "MT-CYB",
        "full_name": "Mitochondrially Encoded Cytochrome b",
        "alias": "Cytb / Apocytochrome b / MT-CYB (HGNC) / MTCYB",
        "omim_gene": "516020",
        "omim_disease": "Isolated CIII deficiency (mitochondrial) — maternal inheritance",
        "disease_name": "Isolated Complex III Deficiency — Mitochondrial (Maternal) — Exercise Intolerance / Myopathy to Leigh Syndrome Spectrum",
        "chromosome": "Mitochondrial DNA (mtDNA), rCRS 14747–15887",
        "inheritance": "Maternal (mtDNA) — heteroplasmic (variable mutant load per tissue)",
        "protein": {
            "size_aa": 380,
            "kDa": 42.7,
            "tm_helices": 8,
            "localization": "IMM — 8 TM helices; ONLY mtDNA-encoded structural subunit of CIII; heme bL at matrix (Qi-site); heme bH at IMS (Qo-site)",
            "function": "Catalytic core of CIII — provides both the Qo-site (ubiquinol oxidation / RISP docking) and Qi-site (ubiquinone reduction) of the Q-cycle. Loss → complete Q-cycle arrest → CIII activity proportional to remaining wild-type MT-CYB (heteroplasmy)",
        },
        "ciii_assembly_step": "Catalytic structural core of CIII — UQCC1/UQCC2 heterodimer stabilises nascent MT-CYB immediately after translation; then UQCRC1/UQCRC2 core assembly around MT-CYB; RISP inserted by BCS1L; peripheral subunits (UQCRQ/UQCRB) added last",
        "bn_page": "CIII absent or severely reduced (2–25%, proportional to heteroplasmy); no precomplex accumulation; CI, CII, CIV normal; isolated CIII pattern",
        "key_biochemical_features": [
            "Isolated CIII deficiency: residual CIII 2–25% (heteroplasmy-dependent)",
            "CI, CII, CIV: NORMAL — DDx from MTFMT (combined CI+CIII+CIV) and FASTKD2 (combined CI+CIV)",
            "CK massively elevated (10–50× ULN) in exercise intolerance phenotype during attacks",
            "Plasma lactate: mildly elevated at rest (mild); severely elevated (>6 mM) in severe form; >10 mM post-exercise in mild form",
            "Muscle heteroplasmy >70% in symptomatic patients (blood heteroplasmy unreliable, often <10–30%)",
            "RRF on Gomori trichrome + COX-negative fibres on muscle histochemistry (mild adult phenotype)",
            "mtDNA sequencing on muscle DNA required — WES does not reliably detect mtDNA point mutations",
        ],
        "absolute_contraindications": [
            "Valproic acid (VPA) — CoA sequestration + POLG inhibition; increases mtDNA depletion risk; doubly toxic in MT-CYB patients",
            "Metformin — Complex I inhibitor; absolute CI in all mitochondrial CIII deficiency including MT-CYB",
            "Linezolid — inhibits mt-23S rRNA translation; DIRECTLY blocks MT-CYB synthesis (double-hit: disease gene is the translation target)",
            "Chloramphenicol — mt-ribosome inhibitor; same mechanism as linezolid; blocks residual cytochrome b translation",
            "Propofol — PRIS: direct CIII Qi-site inhibitor + β-oxidation blockade; fatal in CIII-deficient patients; use Sevoflurane",
        ],
        "recommended_treatments": [
            "Avoid strenuous exercise + prolonged fasting — key triggers for lactic crisis and myoglobinuria",
            "IV Dextrose GIR 6–8 mg/kg/min — during intercurrent illness/crisis; never fast; Level A",
            "IV NaHCO₃ — acute lactic acidosis correction (target pH >7.2 / HCO₃ >12); Level A",
            "IV hydration (0.9% NaCl) — acute myoglobinuria/rhabdomyolysis; target urine output >1 mL/kg/h; Level A",
            "Creatine monohydrate (5–10 g/day adults) — exercise tolerance improvement + CK normalisation; Level C",
            "CoQ10 / Ubiquinol — 15–30 mg/kg/day; CIII electron carrier substrate support; Level C",
            "Thiamine (B1) — empiric PDH/TCA support; mandatory empiric; Level C",
            "Riboflavin (B2) — FAD/FMN support; CI/CIII cofactor; Level C",
            "LEV (Levetiracetam) — preferred AED if seizures; avoid VPA; Level A",
            "Sevoflurane — preferred over Propofol for anaesthesia; Level A",
            "Avoid NSAIDs during myoglobinuria episode (nephrotoxicity risk); Level A",
        ],
        "key_ddx": [
            {
                "condition": "UQCRQ (nuclear AR, CIII Qo-site structural)",
                "distinguishing": "UQCRQ: AR biallelic (not maternal); Dystonia 75% CARDINAL + Optic atrophy 42% DISTINGUISHING; NO exercise intolerance as primary; NO myoglobinuria; NO ragged-red fibres; blood heteroplasmy N/A; both isolated CIII deficiency",
            },
            {
                "condition": "UQCRB (nuclear AR, CIII Qi-site structural)",
                "distinguishing": "UQCRB: AR biallelic (not maternal); Hypoglycaemia 65% DISTINGUISHING (ABSENT in MT-CYB exercise intolerance form); Hepatopathy 55%; NO exercise intolerance as primary feature",
            },
            {
                "condition": "BCS1L (nuclear AR, CIII assembly, GRACILE/Bjornstad)",
                "distinguishing": "BCS1L: CIII precomplex ACCUMULATES on BN-PAGE (pathognomonic) — MT-CYB: NO precomplex; BCS1L: GRACILE triad (iron overload, aminoaciduria, cholestasis) ABSENT in MT-CYB; BCS1L: AR inheritance (not maternal)",
            },
            {
                "condition": "MTFMT (combined CI+CIII+CIV deficiency)",
                "distinguishing": "MTFMT: pan-OXPHOS deficiency (CI+CIII+CIV all reduced) — MT-CYB: ISOLATED CIII (CI and CIV normal); MTFMT: AR biallelic nuclear gene; MT-CYB: maternal mtDNA; both can cause Leigh syndrome — enzyme assay essential",
            },
            {
                "condition": "FASTKD2 (combined CI+CIV deficiency)",
                "distinguishing": "FASTKD2: combined CI+CIV (not isolated CIII); AR biallelic nuclear; MT-CYB: isolated CIII; maternal inheritance — enzyme pattern is the key separator",
            },
            {
                "condition": "TTC19 (nuclear AR, CIII assembly, neurological)",
                "distinguishing": "TTC19: Psychiatric features (38%) + spinocerebellar ataxia + spasticity ABSENT in MT-CYB; TTC19: AR inheritance; MT-CYB: maternal; TTC19: no exercise intolerance as primary; MT-CYB: exercise intolerance/myoglobinuria dominant in mild form",
            },
            {
                "condition": "Other mtDNA myopathies (MTTL1 m.3243A>G, MTTI, MTTW)",
                "distinguishing": "MTTL1 (m.3243A>G MELAS): combined OXPHOS deficiency (not isolated CIII); stroke-like episodes; hearing loss; diabetes — ABSENT in MT-CYB; MT-CYB: isolated CIII on enzyme assay separates from MELAS; mtDNA sequencing is definitive",
            },
        ],
        "genetic_counselling": {
            "recurrence_risk": "MATERNAL inheritance: all children of an affected mother are at risk; risk proportional to maternal heteroplasmy level; paternal transmission DOES NOT OCCUR (mtDNA is maternally inherited)",
            "heteroplasmy_testing": "Maternal relatives (mother, maternal siblings, maternal aunts/uncles) should have mtDNA testing on muscle (blood unreliable); variable heteroplasmy per relative is expected",
            "prenatal_diagnosis": "Prenatal mtDNA heteroplasmy testing via CVS or amniocentesis available but heteroplasmy level in amniocytes may NOT predict severity (tissue heteroplasmy differs); genetic counselling mandatory",
            "blood_vs_muscle": "Blood heteroplasmy often very low (<10–30%) in exercise intolerance phenotype — MUSCLE tissue is required for diagnostic mtDNA sequencing; blood-based testing is unreliable for mild MT-CYB mutations",
            "molecular_confirmation": "mtDNA sequencing (Sanger or long-read) on muscle DNA; WES does NOT reliably detect mtDNA point mutations; MLPA for deletions; heteroplasmy quantification required (not just variant detection)",
            "de_novo_variants": "Some severe infantile MT-CYB variants arise de novo in the egg or early embryo; maternal testing may be negative even in severe cases",
        },
        "key_references": [
            "Andreu AL, Hanna MG, Reichmann H, et al. (1999) Exercise intolerance due to mutations in the cytochrome b gene of mitochondrial DNA. N Engl J Med 341(14):1037-44 — PIVOTAL: first major MT-CYB series; exercise intolerance + myoglobinuria; Gly-34 and other Qo/Qi-site variants",
            "Hanna MG, Nelson IP, Morgan-Hughes JA, Wood NW (2002) Mitochondrial encephalomyopathy with cytochrome b mutations: features of the exercise-induced phenotype. Neurology 58:1047 — exercise-induced phenotype characterisation; heteroplasmy correlates with severity",
            "Sarzi E, Brown S, Lebon S, et al. (2008) A novel recurrent mitochondrial DNA mutation in ND3 gene is associated with isolated Complex I deficiency causing Leigh syndrome and dystonia. Am J Med Genet — context for MT-encoded subunit disease; MT-CYB comparison",
            "Lamantea E, Carrara F, Mariotti C, et al. (2002) A novel nonsense mutation (Q352X) in the mitochondrial cytochrome b gene associated with a combined deficiency of complexes I and III. Neuromuscul Disord 12(1):49-52 — severe pediatric MT-CYB; combined OXPHOS context",
            "Fernandez-Vizarra E & Zeviani M (2015) Nuclear gene mutations as the cause of mitochondrial complex III deficiency. Front Genet 9:135 — comprehensive CIII nuclear vs mtDNA gene review; MT-CYB as the mtDNA-encoded core",
            "Iwata S, Lee JW, Okada K, et al. (1998) Complete structure of the 11-subunit bovine mitochondrial cytochrome bc1 complex. Science 281(5373):64-71 — crystal structure revealing cytochrome b Qo and Qi sites, heme bL/bH positions, 8-TM architecture",
            "Berry EA, Guergova-Kuras M, Huang LS, Crofts AR (2000) Structure and function of cytochrome bc1 complexes. Annu Rev Biochem 69:1005-75 — Q-cycle mechanism; cytochrome b role; Qo/Qi site chemistry",
        ],
        "terms": [
            {
                "term": "MT-CYB (Cytochrome b)",
                "definition": "The only mitochondrially-encoded structural subunit of Complex III; 380 aa, 42.7 kDa, 8 TM helices; contains BOTH the Qo-site (heme bH, IMS face) and Qi-site (heme bL, matrix face); forms the catalytic core of the CIII dimer; mutations → isolated CIII deficiency via maternal inheritance; OMIM *516020",
            },
            {
                "term": "Heteroplasmy",
                "definition": "The coexistence of wild-type and mutant mtDNA molecules in the same cell or tissue; key concept in MT-CYB disease — cells have ~1000s of mtDNA copies; disease manifests when mutant load exceeds a tissue-specific threshold (~70–85% in muscle). Blood heteroplasmy is often very low (<30%) in mild MT-CYB cases; muscle biopsy is required",
            },
            {
                "term": "Exercise intolerance (MT-CYB — Distinguishing)",
                "definition": "Inability to sustain moderate physical activity due to CIII-dependent ATP failure during exercise; at rest, residual CIII (~20%) is sufficient; during exercise, energy demand exceeds CIII capacity → lactic acidosis + myalgia + CK elevation. DISTINGUISHING for mtDNA CIII deficiency — rarely seen in nuclear CIII defects (UQCRQ/UQCRB/BCS1L)",
            },
            {
                "term": "Myoglobinuria / Rhabdomyolysis",
                "definition": "Release of myoglobin from damaged muscle into urine (brown/dark urine) after exercise-induced CIII failure and cell necrosis; occurs in ~45% of MT-CYB exercise intolerance patients. Requires vigorous IV hydration to prevent acute kidney injury. PATHOGNOMONIC for mtDNA myopathy in CIII deficiency context",
            },
            {
                "term": "Ragged-Red Fibres (RRF)",
                "definition": "Subsarcolemmal mitochondrial accumulation seen on Gomori trichrome stain as a red-stained irregular fibre border; occurs in ~48% of adult MT-CYB patients; reflects proliferation of abnormal mitochondria as a compensatory response to CIII failure; ABSENT in nuclear CIII defects (infants lack this compensatory response)",
            },
            {
                "term": "Q-cycle (Mitchell Q-cycle)",
                "definition": "The CIII electron transfer mechanism coupling CoQ oxidation/reduction to proton translocation across the IMM; cytochrome b (MT-CYB) provides the entire scaffold: Qo-site oxidises CoQH2 (2 electrons: one to RISP/CYC1/cytochrome c, one to heme bH); heme bH reduces ubiquinone at Qi-site → CoQH2; net: 2 protons pumped per electron to cytochrome c",
            },
            {
                "term": "Maternal Inheritance",
                "definition": "mtDNA is inherited exclusively through the maternal line — sperm mitochondria are degraded after fertilisation. All children of an affected mother can inherit the mutant MT-CYB allele; heteroplasmy level transmitted is variable (bottleneck effect). Father-to-child transmission NEVER occurs. This distinguishes MT-CYB from ALL nuclear CIII defects (which are AR biallelic)",
            },
            {
                "term": "Creatine Monohydrate",
                "definition": "Dietary supplement (5–10 g/day in adults); phosphocreatine buffer provides ATP during rapid exercise bursts bypassing CIII; modest improvement in exercise tolerance and CK normalisation documented in mitochondrial myopathy including MT-CYB; Level C evidence; not curative but symptomatic benefit",
            },
        ],
    }


if __name__ == "__main__":
    ov = get_overview()
    print(f"Gene: {ov['gene']} ({ov['alias']})")
    print(f"Disease: {ov['disease_name']}")
    print(f"OMIM Gene: *{ov['omim_gene']}")
    print(f"Chromosome: {ov['chromosome']}  Inheritance: {ov['inheritance']}")
    print(f"\nCohort: {ov['cohort_n']} patients, seed {ov['seed']}")
    s = ov["cohort_statistics"]
    print(f"  Exercise intolerance (mild): {s['exercise_intolerance_pct']}%")
    print(f"  Myalgia: {s['myalgia_pct']}%")
    print(f"  CK elevation: {s['ck_elevated_pct']}%")
    print(f"  Myoglobinuria: {s['myoglobinuria_pct']}%")
    print(f"  Lactic acidosis (severe): {s['lactic_acidosis_pct']}%")
    print(f"  Hypotonia (severe): {s['hypotonia_pct']}%")
    print(f"  Leigh-like MRI (severe): {s['leigh_like_mri_pct']}%")
    print(f"  Avg CIII activity: {s['avg_ciii_activity_pct']}%")
    print(f"  Avg lactic acid: {s['avg_lactic_acid_mmolL']} mM")
    print(f"  Deceased (any): {s['deceased_pct']}%")
    print(f"  Maternal family history: {s['maternal_family_affected_pct']}%")
    print(f"\nPhenotype distribution:")
    print(f"  Adult exercise intolerance: {s['mild_adult_pct']}%")
    print(f"  Infantile/severe: {s['severe_pct']}%")
    print("\nVariants:", [v["change"] for v in VARIANTS])
