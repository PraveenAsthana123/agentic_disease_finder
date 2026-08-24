#!/usr/bin/env python3
"""BTD (Biotinidase / Multiple Carboxylase Deficiency — Late-onset/Infantile) Epilepsy Dashboard.

BTD encodes biotinidase (EC 3.5.1.12), the enzyme that cleaves the amide bond in biocytin
(biotinyl-lysine) and short biotinyl-peptides released during protein turnover and intestinal
digestion of dietary protein, RECYCLING free biotin for re-use.

  BIOTIN RECYCLING PATHWAY (BTD is the KEY RECYCLING enzyme):
    Holocarboxylases (biotinylated carboxylases) → proteolytic degradation
    → biocytin (biotinyl-L-lysine) + short biotinyl-peptides
    → BTD cleaves the biotinyl-amide bond → FREE BIOTIN released
    → Free biotin re-conjugated to new apocarboxylases by HLCS
    BTD also releases biotin from dietary protein (intestinal BTD)
    and from biocytin in blood.
    BTD LOF → biocytin accumulates (inhibits HLCS competitively);
    free biotin cannot be regenerated → FUNCTIONAL BIOTIN DEFICIENCY →
    HLCS substrate deplete → ALL FOUR biotin-dependent carboxylases
    become insufficient → Multiple Carboxylase Deficiency (MCD) — LATE-onset.

  SAME FOUR CARBOXYLASES AFFECTED (as in HLCS deficiency):
    1. Pyruvate carboxylase (PC)              — gluconeogenesis + TCA anaplerosis
    2. Propionyl-CoA carboxylase (PCC)        — odd-chain FA / Ile/Val/Met catabolism
    3. 3-Methylcrotonyl-CoA carboxylase (MCC) — leucine catabolism
    4. Acetyl-CoA carboxylase (ACC)           — fatty acid synthesis

  METABOLIC CONSEQUENCES (same four-block pattern as HLCS, but later onset):
    PC block:  lactic acidosis, hyperammonemia (OAA depletion → urea cycle impairment)
    PCC block: methylcitrate, 3-hydroxypropionate, propionylglycine, C3 acylcarnitine elevated
    MCC block: 3-hydroxyisovalerate, 3-methylcrotonylglycine, C5-OH elevated (NBS SENSITIVE)
    ACC block: impaired fatty acid synthesis → skin barrier defect → perioral rash, alopecia

  PATHOGNOMONIC DIAGNOSTIC CLUES in BTD:
    Serum biotinidase activity: PROFOUND deficiency (<10% normal) or PARTIAL (10-30%)
    Urine organic acids: 3-hydroxyisovalerate (MCC) + methylcitrate (PCC) + lactate (PC) elevated
    Plasma acylcarnitines: C5-OH elevated (NBS), C3 (propionylcarnitine) elevated
    Plasma biotin: LOW (primary biotin depletion)
    Biocytin: elevated in blood/CSF (substrate accumulation)
    HLCS activity: NORMAL (HLCS enzyme itself is intact — only substrate depletion)
    KEY: BTD activity <10% = PROFOUND; 10-30% = PARTIAL; >30% = normal/heterozygote

  NBS (Newborn Screening — PRIMARY fluorometric BTD enzyme assay):
    Primary screen: serum biotinidase enzyme activity (colorimetric/fluorometric)
    C5-OH + C3 acylcarnitines: elevated (same as HLCS) — NBS may trigger workup
    Profound BTD (<10% activity): treat immediately with biotin 5-10 mg/day
    Partial BTD (10-30% activity): treat with biotin 2-5 mg/day (prevent under physiologic stress)
    BTD DNA confirmation: sequencing BTD gene (3p25.1)

  BTD vs HLCS — CRITICAL DIFFERENTIAL:
    BTD:  onset 2-12 months (infantile) or rarely up to 5 years; biotinidase LOW; biotin LOW
          Unique features: SENSORINEURAL HEARING LOSS + OPTIC ATROPHY (neurological hallmarks)
          Mechanism: biotin RECYCLING failure → gradual functional depletion
    HLCS: onset neonatal (day 1-10); biotinidase NORMAL; biotin often NORMAL
          Unique features: earlier onset, more severe acidosis; no hearing/vision loss
          Mechanism: biotin LIGATION failure → carboxylases can't be activated

  DISTINGUISHING FEATURE: SENSORINEURAL HEARING LOSS — present in 75% of BTD patients
    (when diagnosis is delayed); mechanism = auditory nerve biotin depletion;
    REVERSIBLE if biotin started early; IRREVERSIBLE if delayed >3-6 months.
    Optic atrophy in ~30% of late-diagnosed patients.
    Spastic paraparesis, developmental delay: also reversible with early biotin.

  BIOTIN TREATMENT:
    Profound BTD: biotin 5-10 mg/day PO lifelong (LEVEL A)
    Partial BTD: biotin 2-5 mg/day (LEVEL A; some guidelines 5 mg/day standardized)
    Raw egg white (avidin) — ABSOLUTE CI: blocks intestinal biotin absorption
    Response: seizures 24-72h, acidosis 24-48h, rash 1-2 weeks, alopecia 2-8 weeks,
    hearing loss + optic atrophy: may NOT reverse if structural damage established

AUTOSOMAL RECESSIVE:
  BTD gene: 3p25.1; Autosomal Recessive (AR)
  Protein: ~543 aa (60-kDa glycoprotein); active as monomer; plasma + tissue forms
  OMIM gene: *609019; disease: #253260 (BIOTINIDASE DEFICIENCY)
  Prevalence: ~1:61,000 combined (profound + partial); 1:112,000 profound alone
  One of most common NBS-detected inborn errors of metabolism worldwide

KEY VARIANTS IN BTD:
  p.Asp444His (c.1330G>C): most common profound allele globally (~30-40% profound alleles)
    — catalytic residue in active site; <10% residual activity; classic neonatal/infantile
  p.Arg538Cys (c.1612C>T): common European profound allele; disulfide bond disruption
  p.Gln456His (c.1368G>C): European profound; active-site adjacent
  p.Leu237_Lys238HisinsPhe (c.755_756delinsTCC): Jewish founder; partial-to-profound
  p.Ala171Thr (c.511G>A): common partial; 30-50% residual activity; milder phenotype
  c.98_104del7ins3 (del. exon 1 region): partial; common in Middle East
  p.Cys35Arg (c.103T>C): early-onset severe; near catalytic site

CLINICAL PRESENTATION SPECTRUM:
  Profound BTD (<10% activity) — 70% of affected:
    Seizures (infantile spasms, myoclonic, tonic-clonic): onset 2-8 months
    Hypotonia + developmental delay: 2-12 months
    Alopecia (total + eyebrows + lashes): ACC block + direct biotin depletion
    Perioral skin rash: ACC block (identical pattern to HLCS)
    SENSORINEURAL HEARING LOSS: 75% (auditory nerve biotin depletion)
    Optic atrophy: 30% (if diagnosis delayed)
    Lactic acidosis + ketoacidosis: episodic, stress-triggered
    Breathing abnormalities (Kussmaul, laryngospasm in crisis): PC block
    Fungal infections (Candida): impaired cell-mediated immunity
  Partial BTD (10-30% activity) — 30% of affected:
    Milder/asymptomatic under normal conditions
    Symptomatic under physiologic stress (illness, fasting, pregnancy)
    Hypotonia, skin rash under stress
    May NOT develop hearing loss if treated early
"""

import random
import math

_RNG = random.Random(61)   # seed=61 for BTD cohort (consistent)

# ------------------------------------------------------------------
# CLINICAL CONSTANTS
# ------------------------------------------------------------------
_PHENOTYPES = [
    ("Profound BTD — Classic Infantile",          28),  # 70% (28/40)
    ("Profound BTD — Neonatal-Onset",              4),  # 10%  (4/40)
    ("Partial BTD — Symptomatic Childhood",        5),  # 12.5% (5/40)
    ("Partial BTD — Stress-Only/Asymptomatic",     3),  # 7.5% (3/40)
]

_VARIANTS = [
    ("p.Asp444His",          "c.1330G>C",  "active site — catalytic Asp", "Profound",  "Global founder ~35% profound alleles"),
    ("p.Arg538Cys",          "c.1612C>T",  "disulfide bond disruption",   "Profound",  "Common European profound"),
    ("p.Gln456His",          "c.1368G>C",  "active site adjacent",        "Profound",  "European profound; <5% activity"),
    ("p.Ala171Thr",          "c.511G>A",   "substrate channel",           "Partial",   "Common partial; ~30-50% activity"),
    ("p.Cys35Arg",           "c.103T>C",   "near catalytic site",         "Profound",  "Early-onset severe"),
    ("c.98_104del7ins3",     "exon 1 del", "frameshift/null",             "Partial",   "Common Middle East; partial"),
    ("p.Leu237_Lys238ins",   "c.755_756",  "insertion; 3D misfolding",    "Profound",  "Jewish founder; complex allele"),
]

_SEIZURE_TYPES = [
    ("Infantile Spasms (West Syndrome)",   72, "Hypsarrhythmia; ACTH second-line after biotin"),
    ("Myoclonic Seizures",                 65, "Generalized; biotin responsive within 24-72h"),
    ("Generalized Tonic-Clonic (GTC)",     55, "Post-infantile spasms evolution"),
    ("Atonic / Drop Attacks",             38, "Fall risk; helmet indicated until biotin response"),
    ("Focal with impaired awareness",      30, "Temporal lobe origin; auditory cortex biotin depletion"),
    ("Absence-like",                       22, "Staring spells; EEG: generalized spike-wave"),
    ("Status Epilepticus (crisis)",        18, "During metabolic decompensation; IV biotin URGENT"),
]

_TREATMENTS = [
    ("Biotin (profound)",    "Level A",  95, "5-10 mg/day PO lifelong; start SAME DAY as diagnosis; if NBS positive do not wait for confirmation"),
    ("Biotin (partial)",     "Level A",  90, "2-5 mg/day PO lifelong; many programs use 5 mg/day standard dose for all BTD"),
    ("IV Biotin (crisis)",   "Level A",  93, "1 mg/kg IV or IM (max 10 mg) for acute decompensation; oral restart when stable"),
    ("IV Glucose + NaHCO3", "Level A",  85, "GIR 8-12 mg/kg/min; correct metabolic acidosis; stop catabolism"),
    ("Carnitine",            "Level B",  70, "Secondary carnitine depletion (C5-OH + C3 sequester carnitine); L-carnitine 50-100 mg/kg/day"),
    ("LEV (Levetiracetam)",  "Level B",  75, "First-line AED bridge until biotin works; well-tolerated; seizures should resolve in 24-72h"),
    ("ACTH / Vigabatrin",    "Level B",  60, "For infantile spasms NOT responding to biotin alone; vigabatrin: visual field monitoring"),
    ("VPA",                  "CAUTION",   0, "Not absolute CI unlike MSUD/DLD/POLG1; but may worsen carnitine depletion — use only if no alternative"),
    ("Raw egg white (avidin)","ABSOLUTE CI", 0, "Avidin blocks intestinal biotin absorption; equivalent to removing all biotin treatment"),
    ("Biotinidase enzyme",   "Not available", 0, "No ERT available; biotin supplementation bypasses enzyme deficiency completely"),
]

_TRIGGERS = [
    ("Intercurrent illness / fever",      88, "Increased metabolic turnover → acute biotin depletion crisis"),
    ("Fasting / prolonged NPO",           75, "PC block → hypoglycemia; MCC/PCC blocks → ketoacidosis"),
    ("Surgery / anesthesia",              60, "Catabolic state; ensure IV glucose + continue biotin perioperatively"),
    ("High-protein intake (excess)",      45, "PCC block overload from Ile/Val/Met; maintain normal protein intake"),
    ("Raw egg white consumption",         40, "Avidin antagonizes free biotin; absolute CI once diagnosis established"),
    ("Anticonvulsant polypharmacy",       30, "Some AEDs reduce biotin absorption; monitor biotin levels"),
    ("Missed biotin doses (≥2 days)",     82, "Rapid metabolic decompensation; emphasize adherence + refill supplies"),
]

_HIGH_RISK_DRUGS = [
    ("Raw egg white / avidin", "ABSOLUTE CI", "Avidin irreversibly binds biotin in gut; blocks absorption; equivalent to no biotin"),
    ("VPA (valproate)",        "CAUTION",     "Carnitine depletion synergy with C5-OH + C3; avoid if possible; monitor carnitine"),
    ("Carbamazepine",          "CAUTION",     "May reduce biotin absorption at gut level; monitor biotin levels annually"),
]

# ------------------------------------------------------------------
# PATIENT COHORT GENERATOR (40 patients)
# ------------------------------------------------------------------
def _make_cohort():
    patients = []
    pid = 1
    phenotype_list = []
    for phenotype, count in _PHENOTYPES:
        phenotype_list.extend([phenotype] * count)

    for i, phenotype in enumerate(phenotype_list):
        rng = _RNG
        sex = "M" if rng.random() < 0.50 else "F"   # BTD 50:50
        is_profound = "Profound" in phenotype
        is_neonatal = "Neonatal" in phenotype

        onset_age = (
            rng.uniform(0.5, 3.0) if is_neonatal else
            rng.uniform(2.0, 10.0) if is_profound else
            rng.uniform(8.0, 36.0)
        )  # months

        # Biomarkers
        btd_activity = (
            rng.uniform(0.5, 8.0)   if is_profound else
            rng.uniform(12.0, 28.0)
        )  # % of normal
        c5oh = rng.uniform(0.8, 4.5) if is_profound else rng.uniform(0.3, 1.0)   # µmol/L (NBS)
        c3   = rng.uniform(3.5, 8.5) if is_profound else rng.uniform(1.5, 3.5)   # µmol/L (NBS)
        three_oh_isoval = rng.uniform(150, 450) if is_profound else rng.uniform(40, 120)  # µmol/mmolCr
        methylcitrate   = rng.uniform(10, 40)   if is_profound else rng.uniform(3, 12)    # µmol/mmolCr
        lactate         = rng.uniform(3.2, 8.5) if is_profound else rng.uniform(1.5, 3.0) # mmol/L

        # Biotin level in plasma (LOW in BTD vs NORMAL in HLCS)
        biotin_plasma = rng.uniform(80, 250) if is_profound else rng.uniform(250, 650)  # pmol/L (normal >450)

        # Clinical features
        alopecia      = True if (is_profound and rng.random() < 0.80) else (rng.random() < 0.25)
        skin_rash     = True if (is_profound and rng.random() < 0.75) else (rng.random() < 0.20)
        hearing_loss  = True if (is_profound and onset_age > 4 and rng.random() < 0.75) else False
        optic_atr     = True if (is_profound and onset_age > 6 and rng.random() < 0.30) else False
        seizures      = True if (is_profound and rng.random() < 0.85) else (rng.random() < 0.40)
        hypotonia     = True if (is_profound and rng.random() < 0.75) else (rng.random() < 0.20)
        candida       = True if (is_profound and rng.random() < 0.35) else False
        nbs_detected  = rng.random() < 0.88     # 88% NBS detection rate
        biotin_dose   = rng.uniform(5, 10) if is_profound else rng.uniform(2, 5)
        biotin_resp   = rng.random() < 0.93     # 93% respond to biotin
        neuro_seq     = True if (not nbs_detected and onset_age > 6 and rng.random() < 0.45) else False
        dev_delay     = True if (neuro_seq and rng.random() < 0.70) else False

        # Variant
        v = _VARIANTS[i % len(_VARIANTS)]
        variant = f"{v[0]} / {v[0]}"  # simplify: homozygous for display

        patients.append({
            "id":             f"BTD-{pid:03d}",
            "sex":            sex,
            "phenotype":      phenotype,
            "onset_age_months": round(onset_age, 1),
            "btd_activity_pct": round(btd_activity, 1),
            "c5oh_umol_l":    round(c5oh, 2),
            "c3_umol_l":      round(c3, 2),
            "three_oh_isovalerate_umol_mmolCr": round(three_oh_isoval, 0),
            "methylcitrate_umol_mmolCr":        round(methylcitrate, 1),
            "lactate_mmol_l": round(lactate, 2),
            "biotin_plasma_pmol_l": round(biotin_plasma, 0),
            "alopecia":       alopecia,
            "skin_rash":      skin_rash,
            "hearing_loss_snhl": hearing_loss,
            "optic_atrophy":  optic_atr,
            "seizures":       seizures,
            "hypotonia":      hypotonia,
            "candida":        candida,
            "nbs_detected":   nbs_detected,
            "biotin_dose_mg_day": round(biotin_dose, 1),
            "biotin_responsive": biotin_resp,
            "neuro_sequelae": neuro_seq,
            "dev_delay":      dev_delay,
            "variant_genotype": variant,
        })
        pid += 1
    return patients


_COHORT = _make_cohort()


# ------------------------------------------------------------------
# PUBLIC API FUNCTIONS
# ------------------------------------------------------------------
def get_overview():
    """Cohort KPIs, phenotype distribution, BTD pathway, BTD vs HLCS differential, high-risk situations."""
    n = len(_COHORT)
    avg_btd_act   = round(sum(p["btd_activity_pct"] for p in _COHORT) / n, 1)
    avg_c5oh      = round(sum(p["c5oh_umol_l"] for p in _COHORT) / n, 2)
    avg_lactate   = round(sum(p["lactate_mmol_l"] for p in _COHORT) / n, 2)
    alopecia_pct  = round(sum(1 for p in _COHORT if p["alopecia"]) / n * 100, 1)
    hearing_pct   = round(sum(1 for p in _COHORT if p["hearing_loss_snhl"]) / n * 100, 1)
    optic_pct     = round(sum(1 for p in _COHORT if p["optic_atrophy"]) / n * 100, 1)
    seizure_pct   = round(sum(1 for p in _COHORT if p["seizures"]) / n * 100, 1)
    skin_pct      = round(sum(1 for p in _COHORT if p["skin_rash"]) / n * 100, 1)
    nbs_pct       = round(sum(1 for p in _COHORT if p["nbs_detected"]) / n * 100, 1)
    resp_pct      = round(sum(1 for p in _COHORT if p["biotin_responsive"]) / n * 100, 1)
    neuro_pct     = round(sum(1 for p in _COHORT if p["neuro_sequelae"]) / n * 100, 1)
    candida_pct   = round(sum(1 for p in _COHORT if p["candida"]) / n * 100, 1)
    hypotonia_pct = round(sum(1 for p in _COHORT if p["hypotonia"]) / n * 100, 1)

    phenotype_dist = []
    for phenotype, count in _PHENOTYPES:
        pct = round(count / n * 100, 1)
        phenotype_dist.append({"phenotype": phenotype, "count": count, "pct": pct})

    four_carboxylases = [
        {
            "enzyme": "Pyruvate Carboxylase (PC)",
            "gene": "PC", "chromosome": "11q13.2",
            "role": "Gluconeogenesis + TCA anaplerosis (pyruvate → OAA)",
            "btd_block_consequence": "Lactic acidosis (pyruvate accumulates), hyperammonemia (OAA depletion → urea cycle impairment), hypoglycemia",
            "biomarker": "Elevated lactate/pyruvate ratio; elevated ammonia",
        },
        {
            "enzyme": "Propionyl-CoA Carboxylase (PCC)",
            "gene": "PCCA / PCCB", "chromosome": "13q32.3 / 3q22.3",
            "role": "Ile/Val/Met/odd-chain FA catabolism (propionyl-CoA → methylmalonyl-CoA)",
            "btd_block_consequence": "Methylcitrate, 3-hydroxypropionate, propionylglycine accumulate; propionylcarnitine (C3) elevated in NBS",
            "biomarker": "Urine methylcitrate + 3-OH-propionate; NBS C3 elevated",
        },
        {
            "enzyme": "3-Methylcrotonyl-CoA Carboxylase (MCC)",
            "gene": "MCCC1 / MCCC2", "chromosome": "3q27.1 / 5q13.2",
            "role": "Leucine catabolism (3-methylcrotonyl-CoA → 3-methylglutaconyl-CoA)",
            "btd_block_consequence": "3-Hydroxyisovalerate, 3-methylcrotonylglycine, C5-OH acylcarnitine accumulate; C5-OH is NBS MOST SENSITIVE marker",
            "biomarker": "Urine 3-OH-isovalerate (HIGH); NBS C5-OH elevated — MOST SENSITIVE",
        },
        {
            "enzyme": "Acetyl-CoA Carboxylase (ACC)",
            "gene": "ACACA / ACACB", "chromosome": "17q12 / 12q24.11",
            "role": "Fatty acid synthesis (acetyl-CoA → malonyl-CoA)",
            "btd_block_consequence": "Impaired de novo fatty acid synthesis → skin barrier dysfunction → perioral dermatitis, alopecia (secondary to ACC block + direct biotin depletion)",
            "biomarker": "Clinical: alopecia + perioral skin rash (no specific biochemical NBS marker for ACC block)",
        },
    ]

    btd_vs_hlcs = {
        "title": "BTD vs HLCS — Two Routes to MCD (CRITICAL DIFFERENTIAL)",
        "note": "Both cause Multiple Carboxylase Deficiency; both respond to biotin; distinguished by biotinidase activity + onset age",
        "comparison": [
            {"feature": "Gene",                 "BTD": "BTD (biotinidase), 3p25.1",          "HLCS": "HLCS (holocarboxylase synthetase), 21q22.13"},
            {"feature": "Mechanism",            "BTD": "Biotin RECYCLING failure (biocytin cleavage impaired)", "HLCS": "Biotin LIGATION failure (can't attach biotin to apocarboxylases)"},
            {"feature": "Biotinidase activity", "BTD": "DEFICIENT (<10% profound; 10-30% partial) — DIAGNOSTIC KEY", "HLCS": "NORMAL — KEY differential from BTD"},
            {"feature": "Plasma biotin level",  "BTD": "LOW (actual biotin depletion)",       "HLCS": "NORMAL (biotin available but can't be used)"},
            {"feature": "Biocytin level",       "BTD": "ELEVATED (substrate accumulates + HLCS inhibitor)", "HLCS": "Normal"},
            {"feature": "Onset",                "BTD": "Infantile: 2-12 months (profound) / childhood (partial)", "HLCS": "Neonatal: Day 1-10 (classic) / early infantile"},
            {"feature": "NBS primary screen",   "BTD": "Biotinidase enzyme assay (fluorometric) — PRIMARY", "HLCS": "C5-OH + C3 acylcarnitines (no direct HLCS enzyme NBS)"},
            {"feature": "SNHL / optic atrophy", "BTD": "YES — sensorineural hearing loss 75%; optic atrophy 30%", "HLCS": "Rare — onset too rapid for neuronal depletion pattern"},
            {"feature": "Fungal infections",    "BTD": "Candida infections (~35%) — T-cell immunity impaired", "HLCS": "Rare"},
            {"feature": "Biotin dose",          "BTD": "5-10 mg/day (profound); 2-5 mg/day (partial)", "HLCS": "10-40 mg/day (higher — ligation kinetics require mass action)"},
            {"feature": "OMIM",                 "BTD": "Gene *609019 / Disease #253260",      "HLCS": "Gene *609018 / Disease #253270"},
            {"feature": "Prevalence",           "BTD": "~1:61,000 combined; 1:112,000 profound", "HLCS": "~1:87,000 (Japan) to 1:200,000+ worldwide"},
        ]
    }

    high_risk = [
        {"situation": "Raw egg white / avidin consumption", "risk": "ABSOLUTE CI", "detail": "Avidin protein binds biotin with extreme affinity (Kd ~10⁻¹⁵ M) in intestine; completely blocks biotin absorption; equivalent to no treatment; raw eggs are absolutely forbidden"},
        {"situation": "Missed biotin doses (≥2 consecutive days)", "risk": "EXTREME HAZARD", "detail": "Rapid metabolic decompensation; biocytin cannot be cleaved; free biotin depleted; seizures may restart within 48-72h; emphasize to family: NEVER skip; carry emergency supply"},
        {"situation": "Intercurrent illness / fever / surgery", "risk": "HIGH HAZARD", "detail": "Catabolic stress increases biotin turnover and carboxylase demand; IV glucose + IV biotin during NPO/surgery; do NOT hold biotin perioperatively"},
        {"situation": "Delayed diagnosis (>3-6 months symptomatic)", "risk": "BRAIN INJURY HAZARD", "detail": "Sensorineural hearing loss and optic atrophy may become IRREVERSIBLE after auditory/optic nerve biotin depletion establishes structural damage; NBS detection is critical"},
        {"situation": "Fasting / prolonged NPO",     "risk": "HIGH HAZARD", "detail": "PC block → hypoglycemia + lactic acidosis; maintain IV dextrose if NPO; never fast BTD patients >4h without IV glucose cover"},
        {"situation": "Carbamazepine / enzyme inducers", "risk": "CAUTION", "detail": "Chronic CBZ may reduce biotin absorption at intestinal level; monitor serum biotin + BTD activity annually; increase biotin dose if clinically indicated"},
    ]

    return {
        "gene": "BTD",
        "full_name": "Biotinidase",
        "disease": "Biotinidase Deficiency (Multiple Carboxylase Deficiency — Late-onset/Infantile)",
        "chromosome": "3p25.1",
        "inheritance": "Autosomal Recessive (AR)",
        "protein_size": "543 aa (60-kDa glycoprotein; plasma + intestinal forms)",
        "omim_gene": "*609019",
        "omim_disease": "#253260",
        "function": "Biotin recycling enzyme — cleaves biocytin (biotinyl-lysine) and biotinyl-peptides to regenerate free biotin; also releases biotin from dietary protein (intestinal BTD)",
        "mechanism": "BTD LOF → biocytin accumulates + free biotin cannot be regenerated → functional biotin depletion → all four biotin-dependent carboxylases (PC/PCC/MCC/ACC) become insufficient → MCD",
        "prevalence": "~1:61,000 combined (profound + partial); 1:112,000 profound; one of most common NBS-detected IEM worldwide",
        "key_negative": "HLCS activity NORMAL (enzyme itself intact; only substrate depletion); distinguish from HLCS by biotinidase assay",
        "nbs_primary": "Serum biotinidase enzyme activity (fluorometric assay) — DIRECT; profound <10%, partial 10-30%",
        "nbs_secondary": "C5-OH (3-methylcrotonylcarnitine) + C3 (propionylcarnitine) acylcarnitines elevated (same as HLCS)",
        "cohort_n": n,
        "kpis": {
            "avg_btd_activity_pct": avg_btd_act,
            "avg_c5oh_umol_l":      avg_c5oh,
            "avg_lactate_mmol_l":   avg_lactate,
            "alopecia_pct":         alopecia_pct,
            "snhl_pct":             hearing_pct,
            "optic_atrophy_pct":    optic_pct,
            "seizure_pct":          seizure_pct,
            "skin_rash_pct":        skin_pct,
            "nbs_detected_pct":     nbs_pct,
            "biotin_responsive_pct":resp_pct,
            "neuro_sequelae_pct":   neuro_pct,
            "candida_infections_pct":candida_pct,
            "hypotonia_pct":        hypotonia_pct,
        },
        "phenotype_distribution":   phenotype_dist,
        "four_carboxylases":        four_carboxylases,
        "btd_vs_hlcs":              btd_vs_hlcs,
        "high_risk_situations":     high_risk,
    }


def get_breakdown():
    """Biomarkers, key variants, patient cohort sample, seizure types, metabolic triggers, treatments."""
    biomarkers = [
        {
            "name": "Serum biotinidase activity",
            "normal": ">30% of mean normal (>4.0 nmol/min/mL)",
            "btd_range": "Profound: <10% (<1.3 nmol/min/mL); Partial: 10-30%",
            "significance": "PRIMARY DIAGNOSTIC TEST — direct enzymatic evidence of BTD deficiency; determines treatment urgency",
            "method": "Colorimetric/fluorometric; biotinyl-p-aminobenzoate substrate; NBS standard",
        },
        {
            "name": "Plasma biotin level",
            "normal": ">450 pmol/L",
            "btd_range": "Profound: 80-250 pmol/L (LOW); Partial: 250-400 pmol/L",
            "significance": "LOW in BTD (actual biotin depletion) vs NORMAL in HLCS — KEY differential; confirms functional biotin deficiency",
            "method": "Competitive binding assay / HPLC-MS; confirmatory test",
        },
        {
            "name": "C5-OH acylcarnitine (NBS)",
            "normal": "<0.4 µmol/L",
            "btd_range": "Profound: 0.8-4.5 µmol/L (elevated); Partial: 0.3-1.0 µmol/L (may be borderline)",
            "significance": "MOST SENSITIVE NBS marker for MCD (same as HLCS); triggers MCD workup; from MCC block (3-methylcrotonylcarnitine)",
            "method": "Tandem mass spectrometry (MS/MS) NBS bloodspot",
        },
        {
            "name": "C3 acylcarnitine (NBS)",
            "normal": "<3.5 µmol/L",
            "btd_range": "Profound: 3.5-8.5 µmol/L (elevated); Partial: 1.5-3.5 µmol/L",
            "significance": "PCC block (propionylcarnitine); co-elevated with C5-OH; pattern = MCD (not pure PA which lacks C5-OH)",
            "method": "Tandem mass spectrometry (MS/MS) NBS bloodspot",
        },
        {
            "name": "Urine 3-hydroxyisovalerate",
            "normal": "<50 µmol/mmolCr",
            "btd_range": "Profound: 150-450 µmol/mmolCr; Partial: 40-120 µmol/mmolCr",
            "significance": "MCC block (most abundant urine organic acid in BTD); closely mirrors severity",
            "method": "Urine organic acids (GC-MS)",
        },
        {
            "name": "Urine methylcitrate",
            "normal": "<5 µmol/mmolCr",
            "btd_range": "Profound: 10-40 µmol/mmolCr; Partial: 3-12 µmol/mmolCr",
            "significance": "PCC block (same as PA/MMA but lower range); in combination with C5-OH = MCD signature",
            "method": "Urine organic acids (GC-MS)",
        },
        {
            "name": "Plasma lactate",
            "normal": "0.5-2.0 mmol/L",
            "btd_range": "Profound: 3.2-8.5 mmol/L; Partial: 1.5-3.0 mmol/L",
            "significance": "PC block (pyruvate → OAA blocked); lactic acidosis marker; elevated L/P ratio",
            "method": "Blood gas / plasma lactate; always check L:P ratio",
        },
        {
            "name": "Biocytin (biotinyl-lysine)",
            "normal": "<25 nmol/L plasma",
            "btd_range": "Elevated: 50-500 nmol/L (substrate accumulation in BTD)",
            "significance": "Substrate for BTD; accumulates when BTD is deficient; also competitively inhibits HLCS; confirms BTD mechanism",
            "method": "LC-MS/MS plasma; research/specialist labs",
        },
        {
            "name": "HLCS enzyme activity",
            "normal": "Full activity (biotinylation of apocarboxylases)",
            "btd_range": "NORMAL in BTD (HLCS enzyme itself intact; only substrate depletion)",
            "significance": "KEY NEGATIVE: HLCS NORMAL confirms mechanism is recycling failure (BTD) not ligation failure (HLCS); distinguishes two MCD types",
            "method": "Cell-based HLCS assay; usually inferred from biotinidase assay rather than directly measured",
        },
    ]

    variants = []
    for v in _VARIANTS:
        variants.append({
            "variant": v[0],
            "cdna": v[1],
            "domain": v[2],
            "severity": v[3],
            "note": v[4],
        })

    sample = _COHORT[:15]  # first 15 patients for display

    seizure_types = [
        {"type": s[0], "pct": s[1], "note": s[2]}
        for s in _SEIZURE_TYPES
    ]

    metabolic_triggers = [
        {"trigger": t[0], "pct": t[1], "mechanism": t[2]}
        for t in _TRIGGERS
    ]

    high_risk_drugs = [
        {"drug": d[0], "risk": d[1], "mechanism": d[2]}
        for d in _HIGH_RISK_DRUGS
    ]

    treatments = [
        {
            "treatment": t[0],
            "evidence": t[1],
            "response_pct": t[2],
            "note": t[3],
        }
        for t in _TREATMENTS
    ]

    return {
        "biomarkers":         biomarkers,
        "key_variants":       variants,
        "patient_sample":     sample,
        "seizure_types":      seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "high_risk_drugs":    high_risk_drugs,
        "treatments":         treatments,
    }


def get_definitions():
    """Gene card, key concepts, diagnostic thresholds, differential diagnosis."""
    gene_card = {
        "Gene symbol":      "BTD",
        "Full name":        "Biotinidase",
        "Alternative names":"BTD; formerly pCBP (plasmatic chromatin-binding protein — historical misnomer)",
        "Chromosome":       "3p25.1",
        "Protein length":   "543 amino acids (60-kDa glycoprotein; N-glycosylated; active as monomer)",
        "Protein forms":    "Plasma BTD (serum; cleaves biocytin from circulation); Intestinal BTD (releases biotin from dietary biotinyl-peptides)",
        "EC number":        "EC 3.5.1.12 (biocytin hydrolase / biotinyl-amide hydrolase)",
        "Catalytic mechanism": "Hydrolyses the amide bond between the carboxyl group of biotin and the ε-amino group of lysine in biocytin (biotinyl-L-Lys) and biotinyl-peptides → releases free biotin",
        "Active site":      "Catalytic triad Cys508-His508-Asp (cysteine nucleophile); Asp444 is essential catalytic residue (most common pathogenic variant)",
        "Inheritance":      "Autosomal Recessive (AR); biallelic LOF pathogenic variants",
        "OMIM Gene":        "*609019",
        "OMIM Disease":     "#253260 (BIOTINIDASE DEFICIENCY)",
        "Prevalence":       "~1:61,000 combined; 1:112,000 profound; 1:129,000 partial alone",
        "NBS inclusion":    "Universal NBS in USA, Canada, UK, EU, Australia since 1980s-2000s; most common NBS-detected IEM in some regions",
    }

    key_concepts = [
        {
            "concept": "Why BTD LOF causes FOUR carboxylase deficiencies simultaneously",
            "explanation": "Biotinidase cleaves biocytin (biotinyl-lysine), the end-product of holocarboxylase degradation, regenerating FREE biotin. Without BTD: (1) free biotin cannot be recycled from endogenous protein turnover; (2) dietary biotinyl-peptides are not digested; (3) biocytin accumulates in blood and CSF, competitively inhibiting HLCS; (4) free biotin becomes depleted → HLCS has insufficient substrate → all four apocarboxylases (PC, PCC, MCC, ACC) cannot be biotinylated → all four carboxylases fail simultaneously → MCD. The mechanism is INDIRECT (biotin depletion) vs HLCS which is DIRECT (LIGATION failure). End result is the same: four-carboxylase MCD."
        },
        {
            "concept": "BTD vs HLCS — why onset is later in BTD than HLCS",
            "explanation": "At birth, the infant has MATERNAL biotin stores (biotin crosses placenta; maternal HLCS biotinylates fetal carboxylases). In BTD deficiency: carboxylases work at birth (maternal biotin adequate); as infant metabolizes protein, biocytin accumulates and free biotin cannot be regenerated → GRADUAL biotin depletion over weeks to months → onset usually 2-12 months. In HLCS deficiency: even maternal biotin cannot be attached to carboxylases (ligation enzyme absent) → carboxylases NEVER function effectively → onset day 1-10 neonatal. This explains the later onset of BTD vs HLCS."
        },
        {
            "concept": "Sensorineural Hearing Loss in BTD — mechanism and reversibility",
            "explanation": "Hearing loss occurs in 75% of profoundly BTD-deficient patients diagnosed late. Mechanism: auditory nerve (cochlear neurons, spiral ganglion) has high biotin turnover requirement; biotin depletion → dysfunction of mitochondrial carboxylases in cochlear neurons → energy failure → progressive auditory nerve degeneration. Critical point: hearing loss IS REVERSIBLE with early biotin (if started before structural degeneration); becomes IRREVERSIBLE once spiral ganglion neurons degenerate (typically >3-6 months of symptomatic depletion). NBS-detected patients started on biotin at birth DO NOT develop hearing loss. Optic atrophy (30%) follows the same principle. This is why BTD is in every NBS panel."
        },
        {
            "concept": "Why biotin works in BTD — pharmacological bypass of enzyme deficiency",
            "explanation": "In BTD deficiency, the PRIMARY problem is FREE BIOTIN DEPLETION (BTD cannot regenerate it). Treatment with oral pharmacological biotin (5-10 mg/day) provides a massive excess of exogenous free biotin that: (1) completely bypasses the BTD recycling pathway; (2) provides sufficient substrate for HLCS to biotinylate all four apocarboxylases; (3) exceeds daily biotin requirement (~30-100 µg/day normal) by 50-100× — this ensures no depletion even without recycling. The biotin dose is LOWER than HLCS (5-10 mg vs 10-40 mg) because HLCS enzyme is INTACT in BTD — normal ligation activity means modest biotin excess suffices; HLCS deficiency requires mass action (very high biotin) to overcome the kinetic Km defect."
        },
        {
            "concept": "NBS screening — why biotinidase enzyme assay is the primary test (not C5-OH)",
            "explanation": "BTD NBS uses a DIRECT ENZYMATIC ASSAY (colorimetric/fluorometric, biotinyl-p-aminobenzoate substrate → p-aminobenzoate detected colorimetrically). This is DIRECT measurement of BTD enzyme activity — far more sensitive and specific than C5-OH (which detects MCC block indirectly). C5-OH can be normal in partial BTD or early profound BTD before metabolic crisis. In contrast, the enzyme assay detects BTD deficiency from day 1 of birth (enzyme is absent/reduced from birth), regardless of metabolic status. This is why BTD NBS identifies patients BEFORE any symptoms develop — enabling pre-symptomatic biotin treatment and complete prevention of hearing loss."
        },
    ]

    diagnostic_thresholds = [
        {"parameter": "Serum biotinidase activity — profound",    "threshold": "<10% of mean normal (<1.3 nmol/min/mL)", "action": "Start biotin 5-10 mg/day SAME DAY; confirm with DNA; audiology"},
        {"parameter": "Serum biotinidase activity — partial",     "threshold": "10-30% of mean normal",               "action": "Start biotin 2-5 mg/day; confirm with DNA; monitor under stress"},
        {"parameter": "Plasma biotin level — LOW",                "threshold": "<450 pmol/L (LOW in BTD vs normal in HLCS)", "action": "Confirms functional biotin depletion; differentiates from HLCS"},
        {"parameter": "NBS C5-OH acylcarnitine recall",           "threshold": ">0.4-0.5 µmol/L (varies by lab)",    "action": "Triggers MCD workup: urine OA + plasma acylcarnitines + BTD activity"},
        {"parameter": "Urine 3-OH-isovalerate",                   "threshold": ">100 µmol/mmolCr (significant elevation)", "action": "MCC block confirmed; MCD panel; start biotin empirically if BTD pending"},
        {"parameter": "Audiology (SNHL monitoring)",              "threshold": "Annual audiogram in profound BTD",    "action": "Baseline at diagnosis; recheck 6-monthly under age 3; SNHL → hearing aids"},
        {"parameter": "Ophthalmology (optic atrophy monitoring)", "threshold": "Annual VEP + fundoscopy in late-diagnosed", "action": "Optic atrophy may partially reverse with biotin; low-vision support if established"},
        {"parameter": "Biotin dose response (seizures)",          "threshold": "Seizure cessation 24-72h after biotin", "action": "If no response by 72h → EEG + neurology review; ensure dose adequacy"},
    ]

    differential = [
        {
            "disease": "HLCS Deficiency",
            "distinguishing": "Biotinidase NORMAL; biotin plasma NORMAL; onset neonatal day 1-10; HLCS LOF; 21q22.13; OMIM #253270; NO SNHL/optic atrophy; biotin dose 10-40 mg (higher)",
        },
        {
            "disease": "Isolated MCC Deficiency (MCCC1/MCCC2)",
            "distinguishing": "C5-OH elevated ONLY (no methylcitrate/lactate); biotinidase NORMAL; biotin NORMAL; usually benign; no MCD pattern; single-enzyme defect not multi-enzyme",
        },
        {
            "disease": "Propionic Acidemia (PA — PCCA/PCCB)",
            "distinguishing": "C3 elevated WITHOUT C5-OH; methylcitrate elevated; biotinidase NORMAL; no response to biotin; PCC isolated defect (not MCD); hyperammonemia more severe; ketoacidosis acute neonatal",
        },
        {
            "disease": "Methylmalonic Acidemia (MMA — MUT/MMAB/etc.)",
            "distinguishing": "C3 elevated; methylmalonate markedly elevated in urine (NOT methylcitrate); biotinidase NORMAL; no C5-OH; no MCC block; no biotin response; cobalamin responsiveness in some subtypes",
        },
        {
            "disease": "Pyruvate Carboxylase Deficiency (PC)",
            "distinguishing": "Lactic acidosis + hyperammonemia ONLY; no C5-OH/C3 elevation; biotinidase NORMAL; no MCC/PCC blocks; citrullinemia + ketouria pattern distinct; no biotin response",
        },
        {
            "disease": "Non-Ketotic Hyperglycinemia (NKH — GLDC/AMT/GCSH)",
            "distinguishing": "Markedly elevated CSF/plasma glycine; CSF:plasma glycine ratio >0.08 DIAGNOSTIC; no C5-OH/C3; no lactate pattern; no biotinidase deficiency; sodium benzoate treatment; no biotin response",
        },
        {
            "disease": "DLD Deficiency (E3 subunit — four-complex block)",
            "distinguishing": "FOUR-complex block (PDH + αKGDH + BCKDH + GCS); elevated BCAA + lactate + 2-HG + glycine SIMULTANEOUSLY; biotinidase NORMAL; biotin NORMAL; DLD LOF; 7q31.1; no biotin response (not a biotin disorder); VPA ABSOLUTE CI",
        },
    ]

    return {
        "gene_card":             gene_card,
        "key_concepts":          key_concepts,
        "diagnostic_thresholds": diagnostic_thresholds,
        "differential_diagnosis": differential,
    }


# ------------------------------------------------------------------
# STANDALONE TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    import json
    ov = get_overview()
    bk = get_breakdown()
    df = get_definitions()
    print(f"Overview KPIs: {json.dumps(ov['kpis'], indent=2)}")
    print(f"Patients: {len(bk['patient_sample'])} (of {ov['cohort_n']})")
    print(f"Biomarkers: {len(bk['biomarkers'])}")
    print(f"Concepts: {len(df['key_concepts'])}")
    print(f"Differential: {len(df['differential_diagnosis'])}")
    print("BTD dashboard OK")
