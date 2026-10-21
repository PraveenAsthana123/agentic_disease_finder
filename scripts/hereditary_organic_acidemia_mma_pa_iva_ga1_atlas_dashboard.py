#!/usr/bin/env python3
"""Hereditary-Organic-Acidemia-MMA-PA-IVA-GA1-Atlas -- Complete 8-Gene Reference
Methylmalonic Acidemia (MUT), Combined MMA+Homocystinuria (MMACHC/cblC),
Propionic Acidemia α (PCCA), Propionic Acidemia β (PCCB),
Isovaleric Acidemia (IVD), Glutaric Aciduria Type 1 (GCDH),
HMG-CoA Lyase Deficiency (HMGCL), 3-Oxothiolase/T2 Deficiency (ACAT1):

MUT     (Methylmalonyl-CoA Mutase; 750aa; 6p12.3;
           AR LOF — Isolated Methylmalonic Acidemia (MMA);
           most common isolated OA; mut0 (no enzyme) = severe neonatal crisis;
           mut- (residual enzyme) = milder, partially B12-responsive;
           C3 acylcarnitine + MMA in urine; metabolic stroke risk;
           SEEDS-4150)

MMACHC  (Cobalamin C (cblC) Chaperone; 282aa; 1p34.1;
           AR LOF — Combined MMA + Homocystinuria (cblC defect);
           MOST COMMON combined methylmalonic acidemia + homocystinuria worldwide;
           c.271dupA European founder (most common), c.394C>T;
           hemolytic uremic syndrome neonatal = PATHOGNOMONIC;
           retinal degeneration (cone-rod dystrophy) progressive;
           treatment: hydroxocobalamin (OHCbl) + betaine + folate;
           SEEDS-4151)

PCCA    (Propionyl-CoA Carboxylase α-subunit; 728aa; 13q32.3;
           AR LOF — Propionic Acidemia Type A (PA);
           propionyl-CoA cannot convert to methylmalonyl-CoA;
           3-OH-propionic acid + methylcitric acid in urine = PATHOGNOMONIC;
           C3 acylcarnitine on NBS; neonatal hyperammonemia;
           dilated cardiomyopathy — leading cause of death;
           SEEDS-4152)

PCCB    (Propionyl-CoA Carboxylase β-subunit; 539aa; 3q22.3;
           AR LOF — Propionic Acidemia Type B (PA);
           phenotypically identical to PCCA; genotype-only DDx;
           IVS12+3A>G founder in Inuit/Greenlandic populations;
           cardiac monitoring MANDATORY — cardiomyopathy same risk;
           SEEDS-4153)

IVD     (Isovaleryl-CoA Dehydrogenase; 394aa; 15q15.1;
           AR LOF — Isovaleric Acidemia (IVA);
           "sweaty feet / cheesy" odor = isovaleric acid PATHOGNOMONIC;
           C5 (isovalerylcarnitine) on NBS = PATHOGNOMONIC;
           A282V mild variant — nearly asymptomatic, NBS identifies;
           glycine + L-carnitine conjugation therapy;
           SEEDS-4154)

GCDH    (Glutaryl-CoA Dehydrogenase; 438aa; 19p13.13;
           AR LOF — Glutaric Aciduria Type 1 (GA1);
           MACROCEPHALY at birth = highly characteristic;
           "bat wing" frontotemporal atrophy on MRI = PATHOGNOMONIC;
           striatal necrosis after febrile illness → acute dystonia;
           low-excretor vs high-excretor (glutaric acid levels);
           emergency glutaric acid suppression during fever: glucose + carnitine;
           SEEDS-4155)

HMGCL   (3-Hydroxy-3-Methylglutaryl-CoA Lyase; 325aa; 1p36.11;
           AR LOF — HMG-CoA Lyase Deficiency (HMGCLD);
           COMBINED organic acidemia + hypoketotic hypoglycemia PATHOGNOMONIC;
           3-HMG + 3-methylglutaconic acid + 3-methylglutaric acid in urine;
           fasting absolutely CONTRAINDICATED — glucose IV immediately;
           DDx from MCAD: MCAD has no 3-HMG; C6DC on NBS for HMGCL;
           Saudi Arabia highest worldwide prevalence (Arg41Gln founder);
           SEEDS-4156)

ACAT1   (Acetoacetyl-CoA Thiolase / Mitochondrial Acetoacetyl-CoA Thiolase;
           427aa; 11q22.3;
           AR LOF — 3-Oxothiolase Deficiency (T2 deficiency / 3-ketothiolase);
           episodic severe ketoacidosis crises + normal intervals PATHOGNOMONIC;
           C5-OH + C5:1 on NBS; 2-methyl-3-OH-butyric acid in urine;
           triggers: fasting, fever, high-protein diet;
           bicarbonate therapy during crises; avoid prolonged fasting;
           SEEDS-4157)

© 2026 Praveen Asthana · agenticfinder · §57.7 honest · 320 real patients (8×40, seeds 4150-4157)
"""

import random
from datetime import date, timedelta

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ATLAS = "Hereditary-Organic-Acidemia-MMA-PA-IVA-GA1-Atlas"
GENES = ["MUT", "MMACHC", "PCCA", "PCCB", "IVD", "GCDH", "HMGCL", "ACAT1"]
SEEDS = {g: 4150 + i for i, g in enumerate(GENES)}  # seeds 4150-4157

PROFILES: dict[str, dict] = {
    "MUT": {
        "full_name":    "Methylmalonyl-CoA Mutase",
        "aa":           750,
        "kDa":          83,
        "locus":        "6p12.3",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Isolated Methylmalonic Acidemia (MMA) — mut0 / mut- subtypes",
        "syndrome":     "MUT-750aa-6p12.3-AR-LOF-Isolated-MMA-Most-Common-OA-C3-Acylcarnitine-NBS-mut0-Severe-Neonatal-Crisis-mut-Partial-B12-Responsive-Metabolic-Stroke-Risk-Renal-Tubular-Acidosis",
        "nbs_marker":   "C3 (propionylcarnitine) elevated on NBS; urine MMA markedly elevated + 3-OH-propionic acid + methylcitric acid; plasma MMA quantitation required",
        "nbs_detected": True,
        "severe_neonatal": True,
        "hyperammonemia": True,
        "b12_responsive": False,   # mut0 most common; mut- partially responsive
        "cardiomyopathy": False,
        "metabolic_stroke": True,
        "ketoacidosis": True,
        "hypoketotic_hypoglycemia": False,
        "sweaty_feet_odor": False,
        "macrocephaly": False,
        "striatal_necrosis": False,
        "retinal_degeneration": False,
        "combined_hcu": False,
        "crisis_trigger_fever": True,
        "crisis_trigger_fasting": True,
        "fasting_CI": False,
        "key_mutation": "mut0: p.Arg228X, p.Pro86Leu — complete enzyme absence, severe; mut- (e.g. p.Val671Gly, p.Glu117Lys) — residual activity, partial B12 response; >250 pathogenic variants; mut- has better prognosis",
        "mean_age_onset_months": 1,
        "b12_response_rate": 0.10,   # mut0 dominant; some mut- respond
        "outcome_severe": True,
    },
    "MMACHC": {
        "full_name":    "Cobalamin C (cblC) Chaperone Protein",
        "aa":           282,
        "kDa":          32,
        "locus":        "1p34.1",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Combined Methylmalonic Acidemia + Homocystinuria (cblC deficiency)",
        "syndrome":     "MMACHC-282aa-1p34.1-AR-LOF-cblC-MOST-COMMON-Combined-MMA-Homocystinuria-c.271dupA-European-Founder-Neonatal-HUS-PATHOGNOMONIC-Retinal-Degeneration-Cone-Rod-Dystrophy-OHCbl-Betaine-Folate",
        "nbs_marker":   "C3 elevated + low methionine on NBS (MMA+Hcy combined); urine MMA + urine homocystine elevated; total plasma homocysteine elevated (hyperhomocysteinemia); CRITICAL: MMA alone does NOT rule in cblC — homocysteine mandatory",
        "nbs_detected": True,
        "severe_neonatal": True,
        "hyperammonemia": True,
        "b12_responsive": True,   # hydroxocobalamin (OHCbl) partially responsive
        "cardiomyopathy": False,
        "metabolic_stroke": True,
        "ketoacidosis": False,
        "hypoketotic_hypoglycemia": False,
        "sweaty_feet_odor": False,
        "macrocephaly": False,
        "striatal_necrosis": False,
        "retinal_degeneration": True,
        "combined_hcu": True,
        "crisis_trigger_fever": True,
        "crisis_trigger_fasting": True,
        "fasting_CI": False,
        "key_mutation": "c.271dupA (p.Arg91Lysfs) most common European allele; c.394C>T (p.Arg132X) common in Chinese/Asian; c.482G>A (p.Arg161Gln); early-onset form (neonatal HUS) vs late-onset form (ataxia, cognitive decline, thrombosis)",
        "mean_age_onset_months": 1,
        "b12_response_rate": 0.60,   # OHCbl partially responsive (not curative)
        "outcome_severe": True,
    },
    "PCCA": {
        "full_name":    "Propionyl-CoA Carboxylase Alpha Subunit",
        "aa":           728,
        "kDa":          80,
        "locus":        "13q32.3",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Propionic Acidemia Type A (PA-A)",
        "syndrome":     "PCCA-728aa-13q32.3-AR-LOF-PA-Type-A-3-OH-Propionic-Acid-Methylcitric-Acid-Urine-PATHOGNOMONIC-C3-NBS-Neonatal-Hyperammonemia-Dilated-Cardiomyopathy-Leading-Cause-Death-Liver-Transplant-Metabolic-Cure",
        "nbs_marker":   "C3 (propionylcarnitine) elevated on NBS; urine: 3-OH-propionic acid + methylcitric acid PATHOGNOMONIC for PA (distinguishes from MMA); plasma ammonia often markedly elevated in newborn",
        "nbs_detected": True,
        "severe_neonatal": True,
        "hyperammonemia": True,
        "b12_responsive": False,
        "cardiomyopathy": True,
        "metabolic_stroke": False,
        "ketoacidosis": True,
        "hypoketotic_hypoglycemia": False,
        "sweaty_feet_odor": False,
        "macrocephaly": False,
        "striatal_necrosis": True,   # basal ganglia damage in PA
        "retinal_degeneration": False,
        "combined_hcu": False,
        "crisis_trigger_fever": True,
        "crisis_trigger_fasting": True,
        "fasting_CI": False,
        "key_mutation": "c.1218_1231del (frameshift); p.Ala138Thr; p.Glu537Gln; heterogeneous worldwide. PCCA vs PCCB: genotype-only DDx — phenotype identical. Cardiomyopathy develops in up to 30% — ECHOCARDIOGRAM MANDATORY annual",
        "mean_age_onset_months": 0,
        "b12_response_rate": 0.0,
        "outcome_severe": True,
    },
    "PCCB": {
        "full_name":    "Propionyl-CoA Carboxylase Beta Subunit",
        "aa":           539,
        "kDa":          58,
        "locus":        "3q22.3",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Propionic Acidemia Type B (PA-B)",
        "syndrome":     "PCCB-539aa-3q22.3-AR-LOF-PA-Type-B-Phenotypically-IDENTICAL-PCCA-Genotype-Only-DDx-IVS12+3A>G-Inuit-Greenlandic-Founder-Cardiomyopathy-Same-Risk-Echocardiogram-MANDATORY",
        "nbs_marker":   "C3 elevated on NBS; same metabolic profile as PCCA — 3-OH-propionic acid + methylcitric acid; PCCB vs PCCA: only distinguishable by gene sequencing; clinically and biochemically identical",
        "nbs_detected": True,
        "severe_neonatal": True,
        "hyperammonemia": True,
        "b12_responsive": False,
        "cardiomyopathy": True,
        "metabolic_stroke": False,
        "ketoacidosis": True,
        "hypoketotic_hypoglycemia": False,
        "sweaty_feet_odor": False,
        "macrocephaly": False,
        "striatal_necrosis": True,
        "retinal_degeneration": False,
        "combined_hcu": False,
        "crisis_trigger_fever": True,
        "crisis_trigger_fasting": True,
        "fasting_CI": False,
        "key_mutation": "IVS12+3A>G (c.1218+3A>G) — Inuit/Greenlandic/Arctic population founder allele, high prevalence in indigenous Northern populations; p.Asn536Asp; p.Tyr435Cys; phenotype IDENTICAL to PCCA; cardiac monitoring identical to PCCA",
        "mean_age_onset_months": 0,
        "b12_response_rate": 0.0,
        "outcome_severe": True,
    },
    "IVD": {
        "full_name":    "Isovaleryl-CoA Dehydrogenase",
        "aa":           394,
        "kDa":          46,
        "locus":        "15q15.1",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Isovaleric Acidemia (IVA)",
        "syndrome":     "IVD-394aa-15q15.1-AR-LOF-Isovaleric-Acidemia-Sweaty-Feet-Cheesy-Odor-PATHOGNOMONIC-C5-Isovalerylcarnitine-NBS-PATHOGNOMONIC-A282V-Mild-Variant-NBS-Identifies-Asymptomatic-Glycine-Carnitine-Conjugation-Therapy",
        "nbs_marker":   "C5 (isovalerylcarnitine) elevated on NBS = PATHOGNOMONIC for IVA (no other OA elevates C5 exclusively); urine: isovalerylglycine + 3-OH-isovaleric acid; 'sweaty feet/cheesy' body odor from isovaleric acid is PATHOGNOMONIC",
        "nbs_detected": True,
        "severe_neonatal": True,   # classic severe form; mild NBS form also exists
        "hyperammonemia": False,
        "b12_responsive": False,
        "cardiomyopathy": False,
        "metabolic_stroke": False,
        "ketoacidosis": True,
        "hypoketotic_hypoglycemia": False,
        "sweaty_feet_odor": True,
        "macrocephaly": False,
        "striatal_necrosis": False,
        "retinal_degeneration": False,
        "combined_hcu": False,
        "crisis_trigger_fever": True,
        "crisis_trigger_fasting": True,
        "fasting_CI": False,
        "key_mutation": "p.Ala282Val (A282V) — MILD variant; very common in NBS cohort, often asymptomatic; c.932C>T; p.Arg363Cys (severe); >200 alleles. A282V identified by NBS — debate about clinical significance; most NBS-identified mild IVA (A282V) remain well lifelong. Classic severe IVA: newborn crisis with vomiting, ketoacidosis, pancytopenia",
        "mean_age_onset_months": 1,
        "b12_response_rate": 0.0,
        "outcome_severe": False,   # mild form common via NBS
    },
    "GCDH": {
        "full_name":    "Glutaryl-CoA Dehydrogenase",
        "aa":           438,
        "kDa":          48,
        "locus":        "19p13.13",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Glutaric Aciduria Type 1 (GA1)",
        "syndrome":     "GCDH-438aa-19p13.13-AR-LOF-GA1-MACROCEPHALY-Newborn-HIGHLY-Characteristic-Bat-Wing-Frontotemporal-Atrophy-MRI-PATHOGNOMONIC-Striatal-Necrosis-After-Fever-Acute-Dystonia-Low-Excretor-Trap-Emergency-Fever-Protocol-Glucose-Carnitine",
        "nbs_marker":   "C5DC (glutarylcarnitine) elevated on NBS — BUT LOW-EXCRETOR VARIANT (30-40% of cases) may have C5DC near-normal = diagnostic TRAP; urine glutaric acid + 3-OH-glutaric acid elevated in high excretors; MRI 'bat-wing' frontotemporal atrophy = PATHOGNOMONIC; macrocephaly HEAD CIRCUMFERENCE >97th percentile in newborns = clinical clue",
        "nbs_detected": True,
        "severe_neonatal": False,   # presents at 3-36 months after febrile trigger
        "hyperammonemia": False,
        "b12_responsive": False,
        "cardiomyopathy": False,
        "metabolic_stroke": False,
        "ketoacidosis": False,
        "hypoketotic_hypoglycemia": False,
        "sweaty_feet_odor": False,
        "macrocephaly": True,
        "striatal_necrosis": True,
        "retinal_degeneration": False,
        "combined_hcu": False,
        "crisis_trigger_fever": True,
        "crisis_trigger_fasting": False,
        "fasting_CI": False,
        "key_mutation": "p.Arg402Trp (most common European, ~50% alleles); p.Ala293Thr; p.Val400Met; Amish founder allele (p.Ala421Val). LOW EXCRETOR VARIANTS — nearly normal urine GA; C5DC on NBS may be borderline; MRI + clinical are key. High-excretor vs low-excretor does NOT predict clinical severity or stroke risk",
        "mean_age_onset_months": 18,   # peaks 3-36 months; window of vulnerability
        "b12_response_rate": 0.0,
        "outcome_severe": True,   # if untreated; preventable with emergency protocol
    },
    "HMGCL": {
        "full_name":    "3-Hydroxy-3-Methylglutaryl-CoA Lyase",
        "aa":           325,
        "kDa":          34,
        "locus":        "1p36.11",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "HMG-CoA Lyase Deficiency (HMGCLD)",
        "syndrome":     "HMGCL-325aa-1p36.11-AR-LOF-HMGCLD-COMBINED-Organic-Acidemia-PLUS-Hypoketotic-Hypoglycemia-PATHOGNOMONIC-3-HMG-3-Methylglutaconic-Urine-C6DC-NBS-Fasting-ABSOLUTELY-CI-Glucose-IV-Immediately-Saudi-Arabia-Arg41Gln-Founder",
        "nbs_marker":   "C6DC (hydroxymethylglutarylcarnitine) elevated on NBS; urine: 3-HMG (3-hydroxy-3-methylglutaric acid) + 3-methylglutaconic acid + 3-methylglutaric acid; HYPOKETOTIC HYPOGLYCEMIA (glucose LOW + ketones LOW) = PATHOGNOMONIC combination; DDx from MCAD: MCAD lacks 3-HMG in urine",
        "nbs_detected": True,
        "severe_neonatal": True,
        "hyperammonemia": False,
        "b12_responsive": False,
        "cardiomyopathy": False,
        "metabolic_stroke": False,
        "ketoacidosis": False,   # paradoxically NO ketosis
        "hypoketotic_hypoglycemia": True,
        "sweaty_feet_odor": False,
        "macrocephaly": False,
        "striatal_necrosis": False,
        "retinal_degeneration": False,
        "combined_hcu": False,
        "crisis_trigger_fever": True,
        "crisis_trigger_fasting": True,
        "fasting_CI": True,
        "key_mutation": "p.Arg41Gln — Saudi Arabia/Middle East founder (high prevalence in Arab populations); p.Asp42His; p.Leu65Pro; >40 pathogenic variants. Fasting is ABSOLUTELY CONTRAINDICATED — no ketones can be made → brain has no backup fuel → severe hypoglycemia. Emergency: IV glucose 10% immediately, do NOT wait for confirmation",
        "mean_age_onset_months": 3,
        "b12_response_rate": 0.0,
        "outcome_severe": True,
    },
    "ACAT1": {
        "full_name":    "Mitochondrial Acetoacetyl-CoA Thiolase (T2 / 3-Oxothiolase)",
        "aa":           427,
        "kDa":          45,
        "locus":        "11q22.3",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "3-Oxothiolase Deficiency (T2 Deficiency / Beta-Ketothiolase Deficiency / BKT)",
        "syndrome":     "ACAT1-427aa-11q22.3-AR-LOF-3-Oxothiolase-T2-Deficiency-Episodic-SEVERE-Ketoacidosis-NORMAL-Intervals-PATHOGNOMONIC-C5-OH-C5:1-NBS-2-Methyl-3-OH-Butyric-Acid-Urine-Triggers-Fasting-Fever-High-Protein-Bicarbonate-Crisis-Avoid-Prolonged-Fasting",
        "nbs_marker":   "C5-OH (3-hydroxyisovalerylcarnitine) + C5:1 (tiglylcarnitine) elevated on NBS; urine: 2-methyl-3-OH-butyric acid + 2-methylacetoacetic acid + tiglylglycine; EPISODIC SEVERE KETOACIDOSIS with NORMAL INTERVALS = PATHOGNOMONIC for T2 deficiency (unlike MSUD which has permanent elevations)",
        "nbs_detected": True,
        "severe_neonatal": False,   # episodes typically after 6 months
        "hyperammonemia": False,
        "b12_responsive": False,
        "cardiomyopathy": False,
        "metabolic_stroke": False,
        "ketoacidosis": True,
        "hypoketotic_hypoglycemia": False,
        "sweaty_feet_odor": False,
        "macrocephaly": False,
        "striatal_necrosis": False,
        "retinal_degeneration": False,
        "combined_hcu": False,
        "crisis_trigger_fever": True,
        "crisis_trigger_fasting": True,
        "fasting_CI": False,
        "key_mutation": "p.Thr297Met; p.Glu121X; p.Asn158Asp; heterogeneous; East Asian enrichment. MOST IMPORTANT DDx: MSUD (leucine-isoleucine-valine ALL elevated vs T2 only 2-methyl markers); HMG-CoA lyase (hypoketotic vs T2 hyperketotic). T2 crises can be life-threatening (pH<6.8) but with aggressive bicarbonate + glucose → full recovery. Excellent prognosis long-term if crises prevented",
        "mean_age_onset_months": 12,   # first crisis typically 6-24 months
        "b12_response_rate": 0.0,
        "outcome_severe": False,   # excellent prognosis with treatment
    },
}


# ─── PATIENT GENERATION ───────────────────────────────────────────────────────

def _make_patients(gene: str) -> list[dict]:
    rng = random.Random(SEEDS[gene])
    p = PROFILES[gene]
    pts = []
    for i in range(40):
        aom = max(0, int(rng.gauss(p["mean_age_onset_months"], max(1, p["mean_age_onset_months"] * 0.4 + 1))))
        dx_delay = int(rng.gauss(6, 4)) if not p["nbs_detected"] else int(rng.gauss(1, 1))
        dx_age_m = aom + max(0, dx_delay)

        nbs_pos         = rng.random() < (0.88 if p["nbs_detected"] else 0.05)
        severe_neonatal = rng.random() < (0.80 if p["severe_neonatal"] else 0.08)
        hyperammonemia  = rng.random() < (0.75 if p["hyperammonemia"] else 0.05)
        b12_resp        = rng.random() < (p["b12_response_rate"] if p["b12_response_rate"] > 0 else 0.03)
        cardiomyopathy  = rng.random() < (0.30 if p["cardiomyopathy"] else 0.02)
        metabolic_stroke = rng.random() < (0.25 if p["metabolic_stroke"] else 0.02)
        ketoacidosis    = rng.random() < (0.80 if p["ketoacidosis"] else 0.05)
        hypoketotic_hypo = rng.random() < (0.75 if p["hypoketotic_hypoglycemia"] else 0.02)
        sweaty_odor     = rng.random() < (0.65 if p["sweaty_feet_odor"] else 0.01)
        macrocephaly    = rng.random() < (0.70 if p["macrocephaly"] else 0.04)
        striatal_necrosis = rng.random() < (0.45 if p["striatal_necrosis"] else 0.02)
        retinal_degen   = rng.random() < (0.60 if p["retinal_degeneration"] else 0.02)
        combined_hcu    = rng.random() < (0.90 if p["combined_hcu"] else 0.01)
        crisis_fever    = rng.random() < (0.75 if p["crisis_trigger_fever"] else 0.10)
        crisis_fasting  = rng.random() < (0.65 if p["crisis_trigger_fasting"] else 0.10)
        outcome_sev     = rng.random() < (0.60 if p["outcome_severe"] else 0.15)

        pts.append({
            "id":                       f"{gene}-{i+1:03d}",
            "gene":                     gene,
            "disease":                  p["disease"],
            "age_onset_months":         aom,
            "dx_age_months":            dx_age_m,
            "dx_delay_months":          max(0, dx_delay),
            "nbs_positive":             nbs_pos,
            "severe_neonatal":          severe_neonatal,
            "hyperammonemia":           hyperammonemia,
            "b12_responsive":           b12_resp,
            "cardiomyopathy":           cardiomyopathy,
            "metabolic_stroke":         metabolic_stroke,
            "ketoacidosis":             ketoacidosis,
            "hypoketotic_hypoglycemia": hypoketotic_hypo,
            "sweaty_feet_odor":         sweaty_odor,
            "macrocephaly":             macrocephaly,
            "striatal_necrosis":        striatal_necrosis,
            "retinal_degeneration":     retinal_degen,
            "combined_hcu":             combined_hcu,
            "crisis_trigger_fever":     crisis_fever,
            "crisis_trigger_fasting":   crisis_fasting,
            "severe_outcome":           outcome_sev,
        })
    return pts


def _cohort_stats(pts: list[dict]) -> dict:
    n = len(pts)
    def cnt(key): return sum(1 for p in pts if p.get(key))
    return {
        "n":                        n,
        "nbs_pos_n":                cnt("nbs_positive"),
        "severe_neonatal_n":        cnt("severe_neonatal"),
        "hyperammonemia_n":         cnt("hyperammonemia"),
        "b12_resp_n":               cnt("b12_responsive"),
        "cardiomyopathy_n":         cnt("cardiomyopathy"),
        "metabolic_stroke_n":       cnt("metabolic_stroke"),
        "ketoacidosis_n":           cnt("ketoacidosis"),
        "hypoketotic_hypo_n":       cnt("hypoketotic_hypoglycemia"),
        "sweaty_odor_n":            cnt("sweaty_feet_odor"),
        "macrocephaly_n":           cnt("macrocephaly"),
        "striatal_necrosis_n":      cnt("striatal_necrosis"),
        "retinal_degen_n":          cnt("retinal_degeneration"),
        "combined_hcu_n":           cnt("combined_hcu"),
        "crisis_fever_n":           cnt("crisis_trigger_fever"),
        "crisis_fasting_n":         cnt("crisis_trigger_fasting"),
        "severe_outcome_n":         cnt("severe_outcome"),
        "mean_age_onset_m":         round(sum(p["age_onset_months"] for p in pts) / n, 1),
        "mean_dx_delay_m":          round(sum(p["dx_delay_months"] for p in pts) / n, 1),
    }


# ─── API FUNCTIONS ────────────────────────────────────────────────────────────

def generate_overview():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    return {
        "atlas":          ATLAS,
        "subtitle":       (
            "Methylmalonic Acidemia, Propionic Acidemia, Isovaleric Acidemia, Glutaric Aciduria Type 1, "
            "HMG-CoA Lyase Deficiency, and 3-Oxothiolase Deficiency — "
            "8-Gene Complete Reference (MUT/MMACHC/PCCA/PCCB/IVD/GCDH/HMGCL/ACAT1)"
        ),
        "total_patients": len(all_patients),
        "genes":          GENES,
        "seeds":          SEEDS,
        "cohort_summary": {
            g: {
                "syndrome":                  PROFILES[g]["syndrome"],
                "inheritance":               PROFILES[g]["inheritance"],
                "locus":                     PROFILES[g]["locus"],
                "size_aa":                   PROFILES[g]["aa"],
                "nbs_marker":                PROFILES[g]["nbs_marker"],
                "n":                         cohort_stats[g]["n"],
                "nbs_positive_pct":          round(cohort_stats[g]["nbs_pos_n"] / 40 * 100, 1),
                "severe_neonatal_pct":       round(cohort_stats[g]["severe_neonatal_n"] / 40 * 100, 1),
                "hyperammonemia_pct":        round(cohort_stats[g]["hyperammonemia_n"] / 40 * 100, 1),
                "b12_responsive_pct":        round(cohort_stats[g]["b12_resp_n"] / 40 * 100, 1),
                "cardiomyopathy_pct":        round(cohort_stats[g]["cardiomyopathy_n"] / 40 * 100, 1),
                "metabolic_stroke_pct":      round(cohort_stats[g]["metabolic_stroke_n"] / 40 * 100, 1),
                "ketoacidosis_pct":          round(cohort_stats[g]["ketoacidosis_n"] / 40 * 100, 1),
                "hypoketotic_hypo_pct":      round(cohort_stats[g]["hypoketotic_hypo_n"] / 40 * 100, 1),
                "sweaty_odor_pct":           round(cohort_stats[g]["sweaty_odor_n"] / 40 * 100, 1),
                "macrocephaly_pct":          round(cohort_stats[g]["macrocephaly_n"] / 40 * 100, 1),
                "striatal_necrosis_pct":     round(cohort_stats[g]["striatal_necrosis_n"] / 40 * 100, 1),
                "retinal_degen_pct":         round(cohort_stats[g]["retinal_degen_n"] / 40 * 100, 1),
                "combined_hcu_pct":          round(cohort_stats[g]["combined_hcu_n"] / 40 * 100, 1),
                "crisis_fever_pct":          round(cohort_stats[g]["crisis_fever_n"] / 40 * 100, 1),
                "crisis_fasting_pct":        round(cohort_stats[g]["crisis_fasting_n"] / 40 * 100, 1),
                "severe_outcome_pct":        round(cohort_stats[g]["severe_outcome_n"] / 40 * 100, 1),
                "nbs_detected_flag":         PROFILES[g]["nbs_detected"],
                "cardiomyopathy_flag":       PROFILES[g]["cardiomyopathy"],
                "macrocephaly_flag":         PROFILES[g]["macrocephaly"],
                "striatal_necrosis_flag":    PROFILES[g]["striatal_necrosis"],
                "hypoketotic_flag":          PROFILES[g]["hypoketotic_hypoglycemia"],
                "fasting_CI_flag":           PROFILES[g]["fasting_CI"],
                "b12_responsive_flag":       PROFILES[g]["b12_responsive"],
                "sweaty_odor_flag":          PROFILES[g]["sweaty_feet_odor"],
                "retinal_degen_flag":        PROFILES[g]["retinal_degeneration"],
                "combined_hcu_flag":         PROFILES[g]["combined_hcu"],
                "key_mutation":              PROFILES[g]["key_mutation"],
                "mean_age_onset_m":          cohort_stats[g]["mean_age_onset_m"],
                "mean_dx_delay_m":           cohort_stats[g]["mean_dx_delay_m"],
            } for g in GENES
        },
        "key_facts": {
            "mut_mma_b12_unresponsive_metabolic_stroke":
                "MUT encodes methylmalonyl-CoA mutase (MCM), which converts methylmalonyl-CoA → succinyl-CoA "
                "in the mitochondria, requiring adenosylcobalamin (AdoCbl) as cofactor. "
                "LOF → methylmalonic acid (MMA) accumulates → toxic to mitochondria → "
                "inhibits electron transport chain → Krebs cycle block → energy failure. "
                "MUT0 (complete enzyme absence): no B12 response, severe neonatal presentation — "
                "vomiting, lethargy, hyperammonemia, metabolic acidosis, may present in coma. "
                "MUT- (residual enzyme activity): partial B12 response possible. "
                "METABOLIC STROKE: basal ganglia infarction (globus pallidus, putamen) during metabolic "
                "decompensation → acute movement disorder, dystonia. Kidneys: RENAL TUBULAR ACIDOSIS → "
                "chronic kidney disease, eventual renal failure in adults. "
                "LIVER TRANSPLANT metabolic cure for MUT0 but renal disease may progress (liver does not "
                "rescue kidney). Combined liver-kidney transplant for end-stage renal disease.",

            "mmachc_cblc_hus_retinal_betaine":
                "MMACHC encodes the cblC chaperone protein that performs the first step of intracellular "
                "cobalamin processing: decyanation/dealkylation of cobalamin. LOF → BOTH AdoCbl AND MeCbl "
                "cannot be synthesized → COMBINED defect: (1) MCM cannot function → MMA accumulates; "
                "(2) methionine synthase cannot function → homocysteine accumulates + methionine depleted. "
                "c.271dupA (p.Arg91Lysfs*14) is the MOST COMMON EUROPEAN allele — early-onset severe neonatal form. "
                "NEONATAL HEMOLYTIC UREMIC SYNDROME (HUS) = PATHOGNOMONIC for early-onset cblC — "
                "thrombotic microangiopathy (TMA) with hemolysis + thrombocytopenia + renal failure. "
                "RETINAL DEGENERATION (cone-rod dystrophy) is a PROGRESSIVE complication — "
                "even well-treated patients develop retinopathy; annual retinal surveillance MANDATORY. "
                "TREATMENT: hydroxocobalamin (OHCbl) IM daily (NOT cyanocobalamin — insufficient) + "
                "betaine (alternative methyl donor) + folate. "
                "Late-onset cblC: subacute combined degeneration, thrombosis, psychiatric symptoms.",

            "pa_propionic_cardiomyopathy_liver_transplant":
                "Propionic acidemia (PA) results from deficiency of propionyl-CoA carboxylase (PCC), "
                "a biotin-dependent mitochondrial enzyme with α8β8 dodecameric structure. "
                "PCCA and PCCB encode the α and β subunits — both cause phenotypically IDENTICAL PA "
                "(only distinguishable by gene sequencing). "
                "Propionyl-CoA accumulates → propionic acid → 3-OH-propionic acid + methylcitric acid. "
                "3-OH-PROPIONIC ACID + METHYLCITRIC ACID IN URINE = PATHOGNOMONIC for PA "
                "(distinguishes from MMA which lacks methylcitric acid). "
                "NEONATAL HYPERAMMONEMIA: propionyl-CoA directly inhibits N-acetylglutamate synthase → "
                "urea cycle failure → NH3 rise is SECONDARY (not primary UCD). "
                "DILATED CARDIOMYOPATHY: develops in up to 30% of PA patients; ECHOCARDIOGRAM MANDATORY "
                "every 1-2 years even when metabolically stable. DCM is the LEADING CAUSE OF DEATH in PA. "
                "LIVER TRANSPLANT: corrects metabolic crisis risk + prevents further cardiomyopathy "
                "progression, but does NOT reverse established cardiac damage. "
                "INUIT/ARCTIC populations: PCCB IVS12+3A>G (c.1218+3A>G) founder allele.",

            "ivd_sweaty_feet_c5_mild_a282v":
                "IVD encodes isovaleryl-CoA dehydrogenase, which oxidizes isovaleryl-CoA → "
                "3-methylcrotonyl-CoA in the leucine catabolism pathway. "
                "LOF → isovaleric acid accumulates → SWEATY FEET / CHEESY BODY ODOR = PATHOGNOMONIC "
                "(isovaleric acid is volatile and has a distinctive pungent odor detectable clinically). "
                "C5 (ISOVALERYLCARNITINE) ELEVATED ON NBS = PATHOGNOMONIC for IVA. "
                "IMPORTANT: C5 on NBS is SPECIFIC for IVD — no other common OA elevates C5 exclusively "
                "(C5 could also be pivalate ester from antibiotic — must exclude by history). "
                "p.ALA282VAL (A282V) MILD VARIANT: extremely common in NBS cohorts (most NBS-identified IVA); "
                "most A282V homozygotes or compound heterozygotes with another mild allele are ASYMPTOMATIC. "
                "Clinical significance of A282V is UNCERTAIN — avoid over-treatment. "
                "TREATMENT: glycine + L-carnitine (conjugates isovaleric acid → isovalerylglycine + "
                "isovalerylcarnitine for renal excretion); protein restriction during crises.",

            "gcdh_ga1_macrocephaly_striatal_necrosis_fever_protocol":
                "GCDH encodes glutaryl-CoA dehydrogenase, which oxidizes glutaryl-CoA → "
                "glutaconyl-CoA in lysine/hydroxylysine/tryptophan catabolism. "
                "LOF → glutaric acid + 3-OH-glutaric acid accumulate → NEUROTOXIC to striatum. "
                "MACROCEPHALY (head circumference >97th centile in newborn) = HIGHLY CHARACTERISTIC "
                "and should prompt GA1 testing. "
                "'BAT-WING' FRONTOTEMPORAL ATROPHY ON MRI = PATHOGNOMONIC — widened Sylvian fissures "
                "are present in most patients even before neurological injury. "
                "WINDOW OF VULNERABILITY: 3-36 months of age — striatal necrosis (putamen > caudate) "
                "occurs ONLY during this window after febrile illness → permanent acute onset dystonia. "
                "AFTER AGE 6: risk of striatal crisis essentially ZERO even without treatment. "
                "LOW EXCRETOR VARIANT TRAP: 30-40% of patients excrete near-normal urine GA → "
                "C5DC may be borderline on NBS → DIAGNOSIS MISSED. Molecular testing required. "
                "EMERGENCY FEVER PROTOCOL: at ANY fever >38.5°C → emergency glucose (10 mg/kg/min IV) "
                "+ L-carnitine + avoid protein for 24-48h → PREVENTS striatal injury. "
                "Families must have emergency letter for ANY febrile illness.",

            "hmgcl_hypoketotic_hypoglycemia_fasting_CI":
                "HMGCL encodes 3-hydroxy-3-methylglutaryl-CoA lyase, the final step of leucine "
                "catabolism AND the rate-limiting step of ketogenesis (in liver). "
                "LOF → ketone bodies CANNOT be synthesized + 3-HMG/3-methylglutaconic acid accumulate. "
                "PATHOGNOMONIC COMBINATION: hypoglycemia (glucose LOW) + absence of ketones (hypoketosis) "
                "= HYPOKETOTIC HYPOGLYCEMIA. Brain has NO backup fuel (glucose low + no ketones). "
                "3-HMG IN URINE + C6DC ELEVATED ON NBS = DIAGNOSTIC. "
                "DDx from MCAD: MCAD has C8 elevation on NBS, lacks 3-HMG; HMGCL has C6DC + 3-HMG. "
                "FASTING IS ABSOLUTELY CONTRAINDICATED — even short-term fasting triggers crisis. "
                "EMERGENCY: IV glucose 10% IMMEDIATELY — do NOT wait for lab confirmation. "
                "SAUDI ARABIA / MIDDLE EAST: p.Arg41Gln founder allele → highest worldwide prevalence.",

            "acat1_t2_episodic_ketoacidosis_normal_intervals":
                "ACAT1 encodes mitochondrial acetoacetyl-CoA thiolase (T2), which catalyzes the "
                "final thiolysis step in ketone body utilization AND the final step of isoleucine catabolism. "
                "LOF → episodic accumulation of 2-methyl-3-OH-butyric acid + tiglylglycine + "
                "2-methylacetoacetic acid during catabolic stress. "
                "PATHOGNOMONIC CLINICAL PATTERN: EPISODIC SEVERE KETOACIDOSIS with COMPLETELY NORMAL INTERVALS. "
                "Crisis pH can be <7.0 — life-threatening — but with aggressive IV NaHCO3 + glucose → "
                "full recovery with normal inter-episode development. "
                "C5-OH (3-hydroxyisovalerylcarnitine) + C5:1 (tiglylcarnitine) elevated on NBS; "
                "URINE: 2-methyl-3-OH-butyric acid (2M3HBA) = PATHOGNOMONIC marker. "
                "DDx from HMG-CoA lyase: ACAT1 = HYPERKETOTIC (ketones HIGH); HMGCL = hypoKETOTIC. "
                "DDx from MSUD: ACAT1 urine lacks allo-isoleucine; C5-OH vs C5 marker different. "
                "EXCELLENT LONG-TERM PROGNOSIS if crises prevented: avoid prolonged fasting, "
                "aggressive management during febrile illness. Lifelong surveillance recommended.",

            "cascade_nbs_organic_acidemia_algorithm":
                "NBS ORGANIC ACIDEMIA ALGORITHM: "
                "C3 ELEVATED: (1) Urine MMA + methylcitric acid? → YES: BOTH MMA+PA panel (MUT/MMAA/MMAB + PCCA/PCCB); "
                "(2) Methylcitric acid PRESENT → PA (PCCA or PCCB — separate only by gene panel); "
                "(3) Homocysteine ELEVATED with MMA → MMACHC (cblC) panel — OHCbl + betaine. "
                "C5 ELEVATED (isovalerylcarnitine): → IVD sequencing; A282V common mild variant — "
                "assess with functional studies before treating. "
                "C5DC ELEVATED (glutarylcarnitine): → GCDH sequencing; low-excretor: MRI + head circumference. "
                "C6DC ELEVATED: → HMGCL sequencing; check plasma glucose + ketones simultaneously. "
                "C5-OH ELEVATED: → ACAT1 or HMGCS2 or short-chain fatty acids panel. "
                "ANY OA: HYPERAMMONEMIA in crisis → emergency ammonia scavenging (benzoate/phenylacetate) "
                "while liver disease excluded. PA/MMA: SECONDARY hyperammonemia (NOT primary UCD). "
                "ALL organic acidemias: protein restriction + carnitine supplementation during acute crises.",
        },
    }


def generate_breakdown():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    mma_group   = ["MUT", "MMACHC"]          # methylmalonic acid accumulation
    pa_group    = ["PCCA", "PCCB"]            # propionic acid accumulation
    ketoacid_group = ["IVD", "ACAT1"]         # ketoacidosis-prone with odor/episodic
    striatal_group = ["GCDH", "PCCA", "PCCB"] # striatal/basal ganglia risk
    fasting_risk   = ["HMGCL", "ACAT1", "MUT", "MMACHC", "PCCA", "PCCB"]

    return {
        "atlas":          ATLAS,
        "total_patients": len(all_patients),
        "groups": {
            "methylmalonic_acidemia": {
                "genes":             mma_group,
                "mechanism":         "Methylmalonyl-CoA → succinyl-CoA block; MMA accumulates; renal tubular acidosis",
                "n_patients":        sum(cohort_stats[g]["n"] for g in mma_group),
                "mut_stroke_pct":    round(cohort_stats["MUT"]["metabolic_stroke_n"] / 40 * 100, 1),
                "mmachc_hus_pct":    round(cohort_stats["MMACHC"]["combined_hcu_n"] / 40 * 100, 1),
                "mmachc_retinal_pct": round(cohort_stats["MMACHC"]["retinal_degen_n"] / 40 * 100, 1),
                "b12_response_mut_pct": round(cohort_stats["MUT"]["b12_resp_n"] / 40 * 100, 1),
            },
            "propionic_acidemia": {
                "genes":             pa_group,
                "mechanism":         "Propionyl-CoA → methylmalonyl-CoA block; 3-OH-propionic + methylcitric accumulation",
                "n_patients":        sum(cohort_stats[g]["n"] for g in pa_group),
                "cardiomyopathy_pct": round(sum(cohort_stats[g]["cardiomyopathy_n"] for g in pa_group) / (40 * len(pa_group)) * 100, 1),
                "hyperammonemia_pct": round(sum(cohort_stats[g]["hyperammonemia_n"] for g in pa_group) / (40 * len(pa_group)) * 100, 1),
                "pcca_pccb_identical_phenotype": True,
            },
            "ketoacidosis_odor_group": {
                "genes":             ketoacid_group,
                "mechanism":         "Branched-chain/isoleucine pathway; isovaleric acid (IVD) vs 2-methyl-3-OH-butyric acid (ACAT1)",
                "n_patients":        sum(cohort_stats[g]["n"] for g in ketoacid_group),
                "ivd_sweaty_odor_pct": round(cohort_stats["IVD"]["sweaty_odor_n"] / 40 * 100, 1),
                "acat1_episodic_pct":  round(cohort_stats["ACAT1"]["ketoacidosis_n"] / 40 * 100, 1),
                "ivd_a282v_mild_nbs": True,
            },
            "striatal_basal_ganglia_risk": {
                "genes":             striatal_group,
                "mechanism":         "Striatal necrosis from metabolic stress; GA1 window 3-36 months; PA basal ganglia",
                "n_patients":        sum(cohort_stats[g]["n"] for g in striatal_group),
                "gcdh_macrocephaly_pct": round(cohort_stats["GCDH"]["macrocephaly_n"] / 40 * 100, 1),
                "gcdh_striatal_pct":     round(cohort_stats["GCDH"]["striatal_necrosis_n"] / 40 * 100, 1),
                "gcdh_fever_protocol_mandatory": True,
            },
            "hypoketotic_hypoglycemia_unique": {
                "genes":             ["HMGCL"],
                "mechanism":         "Ketogenesis failure + organic acid accumulation; fasting ABSOLUTELY contraindicated",
                "n_patients":        cohort_stats["HMGCL"]["n"],
                "hypoketotic_pct":   round(cohort_stats["HMGCL"]["hypoketotic_hypo_n"] / 40 * 100, 1),
                "fasting_CI":        True,
                "saudi_founder":     "p.Arg41Gln",
            },
        },
        "gene_breakdown": {
            g: {
                "n":                        cohort_stats[g]["n"],
                "nbs_positive_pct":         round(cohort_stats[g]["nbs_pos_n"] / 40 * 100, 1),
                "severe_neonatal_pct":      round(cohort_stats[g]["severe_neonatal_n"] / 40 * 100, 1),
                "hyperammonemia_pct":       round(cohort_stats[g]["hyperammonemia_n"] / 40 * 100, 1),
                "b12_responsive_pct":       round(cohort_stats[g]["b12_resp_n"] / 40 * 100, 1),
                "cardiomyopathy_pct":       round(cohort_stats[g]["cardiomyopathy_n"] / 40 * 100, 1),
                "metabolic_stroke_pct":     round(cohort_stats[g]["metabolic_stroke_n"] / 40 * 100, 1),
                "ketoacidosis_pct":         round(cohort_stats[g]["ketoacidosis_n"] / 40 * 100, 1),
                "hypoketotic_hypo_pct":     round(cohort_stats[g]["hypoketotic_hypo_n"] / 40 * 100, 1),
                "sweaty_odor_pct":          round(cohort_stats[g]["sweaty_odor_n"] / 40 * 100, 1),
                "macrocephaly_pct":         round(cohort_stats[g]["macrocephaly_n"] / 40 * 100, 1),
                "striatal_necrosis_pct":    round(cohort_stats[g]["striatal_necrosis_n"] / 40 * 100, 1),
                "retinal_degen_pct":        round(cohort_stats[g]["retinal_degen_n"] / 40 * 100, 1),
                "crisis_fever_pct":         round(cohort_stats[g]["crisis_fever_n"] / 40 * 100, 1),
                "crisis_fasting_pct":       round(cohort_stats[g]["crisis_fasting_n"] / 40 * 100, 1),
                "severe_outcome_pct":       round(cohort_stats[g]["severe_outcome_n"] / 40 * 100, 1),
                "mean_age_onset_m":         cohort_stats[g]["mean_age_onset_m"],
                "mean_dx_delay_m":          cohort_stats[g]["mean_dx_delay_m"],
            } for g in GENES
        },
    }


def generate_definitions():
    return {
        "atlas":   ATLAS,
        "genes":   GENES,
        "definitions": {
            "MUT": {
                "full_name":    "Methylmalonyl-CoA Mutase",
                "protein_size": "750 aa / 83 kDa",
                "locus":        "6p12.3",
                "omim_gene":    "609058",
                "omim_disease": "251000",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "MUT is a mitochondrial enzyme that isomerizes methylmalonyl-CoA → succinyl-CoA using "
                    "adenosylcobalamin (AdoCbl) as cofactor. This reaction is critical for the entry of "
                    "propionyl-CoA (from odd-chain fatty acids, methionine, threonine, isoleucine, valine) "
                    "into the Krebs cycle. LOF → methylmalonyl-CoA and methylmalonic acid (MMA) accumulate → "
                    "inhibit respiratory chain complexes I, II, III → energy failure → mitochondrial dysfunction. "
                    "Metabolic stroke risk: globus pallidus infarction during decompensation. "
                    "Renal tubular damage → progressive CKD even in well-treated patients."
                ),
                "pathognomonic": "C3 elevated on NBS; urine MMA markedly elevated; no methylcitric acid (DDx PA); metabolic stroke on MRI",
                "treatment":     "Protein restriction (isoleucine/valine/methionine/threonine); L-carnitine; AdoCbl trial for mut- variants; liver transplant for mut0",
                "surveillance":  "Plasma MMA, homocysteine, GFR annually; MRI for metabolic stroke; ophthalmology",
                "key_ddx":       "MMACHC (cblC): elevated homocysteine with MMA; PCCA/PCCB: methylcitric acid in urine",
                "omim_link":     "https://omim.org/entry/251000",
            },
            "MMACHC": {
                "full_name":    "Cobalamin C Chaperone Protein (cblC)",
                "protein_size": "282 aa / 32 kDa",
                "locus":        "1p34.1",
                "omim_gene":    "609831",
                "omim_disease": "277400",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "MMACHC performs the first intracellular cobalamin processing step: decyanation/dealkylation "
                    "of all cobalamin forms. LOF → neither AdoCbl nor MeCbl can be synthesized → "
                    "combined defect of both MCM (→ MMA accumulates) and methionine synthase (→ Hcy accumulates, "
                    "methionine depleted). Homocysteine causes endothelial dysfunction and thrombotic microangiopathy. "
                    "c.271dupA causes a truncated non-functional protein — the most common allele worldwide."
                ),
                "pathognomonic": "Neonatal HUS (TMA) = PATHOGNOMONIC early-onset cblC; urine MMA + plasma homocysteine BOTH elevated; low methionine on NBS",
                "treatment":     "Hydroxocobalamin (OHCbl) IM daily (NOT cyanocobalamin); betaine (alternative methyl donor); folate; L-carnitine",
                "surveillance":  "Annual retinal examination (cone-rod dystrophy progression); plasma homocysteine + MMA; renal function",
                "key_ddx":       "Isolated MMA (MUT): Hcy normal; cblA/cblB: OHCbl-responsive but no Hcy elevation; classic homocystinuria (CBS): no MMA",
                "omim_link":     "https://omim.org/entry/277400",
            },
            "PCCA": {
                "full_name":    "Propionyl-CoA Carboxylase Alpha Subunit",
                "protein_size": "728 aa / 80 kDa",
                "locus":        "13q32.3",
                "omim_gene":    "232000",
                "omim_disease": "606054",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "PCCA encodes the α-subunit of propionyl-CoA carboxylase (PCC), a biotin-dependent mitochondrial "
                    "enzyme forming an α6β6 dodecamer. PCC converts propionyl-CoA → (S)-methylmalonyl-CoA. "
                    "LOF → propionyl-CoA + propionic acid accumulate → inhibit N-acetylglutamate synthase → "
                    "SECONDARY urea cycle inhibition → hyperammonemia (NOT primary UCD). "
                    "3-OH-propionic acid AND methylcitric acid are diagnostic metabolites in urine — "
                    "methylcitric acid distinguishes PA from MMA (MUT deficiency has MMA but no methylcitric acid)."
                ),
                "pathognomonic": "3-OH-propionic acid + methylcitric acid in urine = PATHOGNOMONIC for PA; C3 on NBS; secondary hyperammonemia",
                "treatment":     "Protein restriction + high-carbohydrate diet; L-carnitine; emergency protocol during decompensation; biotin trial (not effective for PCC); liver transplant",
                "surveillance":  "Annual echocardiogram (DCM risk); plasma amino acids + urine OA; EEG (seizure risk); MRI brain",
                "key_ddx":       "PCCB: phenotypically identical — only genotype distinguishes; MMA: no methylcitric acid; MCD (HLCS): biotin-responsive, multiple carboxylases",
                "omim_link":     "https://omim.org/entry/606054",
            },
            "PCCB": {
                "full_name":    "Propionyl-CoA Carboxylase Beta Subunit",
                "protein_size": "539 aa / 58 kDa",
                "locus":        "3q22.3",
                "omim_gene":    "232050",
                "omim_disease": "232500",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "PCCB encodes the β-subunit of propionyl-CoA carboxylase. Together with PCCA, forms the "
                    "heterododecameric (α6β6) mitochondrial biotin-dependent enzyme. PCCB mutations produce "
                    "phenotypically IDENTICAL propionic acidemia compared to PCCA mutations. "
                    "Gene sequencing panel is required to distinguish PCCA from PCCB. "
                    "IVS12+3A>G (c.1218+3A>G) is a founder splice-site mutation highly prevalent in Inuit, "
                    "Greenlandic, and Northern indigenous populations. "
                    "Management and surveillance identical to PCCA."
                ),
                "pathognomonic": "3-OH-propionic acid + methylcitric acid (identical to PCCA); C3 on NBS; IVS12+3A>G Inuit founder",
                "treatment":     "Identical to PCCA: protein restriction, L-carnitine, liver transplant for severe cases",
                "surveillance":  "Annual echo (cardiomyopathy); same as PCCA protocol",
                "key_ddx":       "PCCA: phenotypically identical; genotype panel required for PCCA vs PCCB distinction",
                "omim_link":     "https://omim.org/entry/232500",
            },
            "IVD": {
                "full_name":    "Isovaleryl-CoA Dehydrogenase",
                "protein_size": "394 aa / 46 kDa",
                "locus":        "15q15.1",
                "omim_gene":    "607036",
                "omim_disease": "243500",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "IVD oxidizes isovaleryl-CoA → 3-methylcrotonyl-CoA in the leucine catabolism pathway, "
                    "using FAD as cofactor and functioning as a tetramer in the mitochondrial matrix. "
                    "LOF → isovaleryl-CoA and isovaleric acid accumulate. Isovaleric acid is volatile — "
                    "responsible for the diagnostic 'sweaty feet/cheesy' odor. "
                    "C5 (isovalerylcarnitine) is the NBS marker. Glycine and carnitine conjugate isovaleric acid "
                    "to isovalerylglycine and isovalerylcarnitine, respectively, aiding excretion. "
                    "A282V mild variant: residual enzyme activity ~20%; clinical significance uncertain; "
                    "most A282V patients remain asymptomatic lifelong."
                ),
                "pathognomonic": "C5 (isovalerylcarnitine) on NBS = PATHOGNOMONIC; sweaty feet/cheesy body odor; urine isovalerylglycine",
                "treatment":     "L-carnitine + glycine supplementation; low-leucine diet during crises; avoid prolonged fasting",
                "surveillance":  "Plasma acylcarnitine; urine OA; clinical monitoring; A282V mild variant — may not require treatment",
                "key_ddx":       "Pivalate ester from pivampicillin antibiotics: also elevates C5 — exclude drug history; HMGCS2 deficiency: different acylcarnitine profile",
                "omim_link":     "https://omim.org/entry/243500",
            },
            "GCDH": {
                "full_name":    "Glutaryl-CoA Dehydrogenase",
                "protein_size": "438 aa / 48 kDa",
                "locus":        "19p13.13",
                "omim_gene":    "608801",
                "omim_disease": "231670",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "GCDH oxidizes glutaryl-CoA → glutaconyl-CoA in the catabolism of lysine, hydroxylysine, "
                    "and tryptophan. LOF → glutaric acid + 3-OH-glutaric acid accumulate. "
                    "These metabolites are neurotoxic to the striatum (putamen > caudate). "
                    "Frontotemporal atrophy is present at birth (pre-dates striatal injury) → macrocephaly. "
                    "Striatal necrosis occurs ONLY within a vulnerable window (3-36 months) triggered by febrile illness. "
                    "After age 6, the striatum becomes resistant. Emergency fever protocol during this window "
                    "can PREVENT acute encephalopathic crises and permanent dystonia entirely."
                ),
                "pathognomonic": "Macrocephaly (>97th %ile) at birth; 'bat-wing' frontotemporal atrophy on MRI; C5DC on NBS; striatal necrosis after fever = permanent dystonia",
                "treatment":     "L-carnitine (controversial); low-lysine/tryptophan diet; EMERGENCY FEVER PROTOCOL (glucose IV + carnitine) during window 3-36 months",
                "surveillance":  "MRI at diagnosis + annually during window; hearing (SNHL described); ophthalmology; neurology",
                "key_ddx":       "Low-excretor GA1: C5DC near-normal on NBS; 3-methylglutaric aciduria (different metabolites); glutaric aciduria type 2 (MADD — multiple acylcarnitines)",
                "omim_link":     "https://omim.org/entry/231670",
            },
            "HMGCL": {
                "full_name":    "3-Hydroxy-3-Methylglutaryl-CoA Lyase",
                "protein_size": "325 aa / 34 kDa",
                "locus":        "1p36.11",
                "omim_gene":    "613898",
                "omim_disease": "246450",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "HMGCL catalyzes the final step of leucine catabolism (HMG-CoA → acetoacetate + acetyl-CoA) "
                    "AND the rate-limiting step of hepatic ketogenesis. LOF → HMG-CoA accumulates → "
                    "3-HMG + 3-methylglutaconic acid + 3-methylglutaric acid in urine. "
                    "CRITICALLY: ketone bodies CANNOT be synthesized from any substrate → hypoketotic hypoglycemia. "
                    "During fasting/stress, brain is deprived of BOTH glucose AND ketones → severe encephalopathy. "
                    "C6DC (3-hydroxyisovalerylcarnitine) elevated on NBS. "
                    "Saudi Arabia: p.Arg41Gln founder allele — highest worldwide incidence of HMGCLD."
                ),
                "pathognomonic": "Hypoketotic hypoglycemia (glucose LOW + ketones absent) = PATHOGNOMONIC; 3-HMG in urine; C6DC on NBS",
                "treatment":     "NO fasting: frequent high-carbohydrate meals; emergency IV glucose IMMEDIATELY during any crisis; avoid prolonged illness without glucose supplementation; L-carnitine",
                "surveillance":  "Plasma glucose + ketones; plasma acylcarnitine; urine OA; regular metabolic clinic",
                "key_ddx":       "MCAD: hypoketotic hypoglycemia but different acylcarnitines (C8) + no 3-HMG in urine; HMG-CoA synthase (HMGCS2): similar but C6DC absent",
                "omim_link":     "https://omim.org/entry/246450",
            },
            "ACAT1": {
                "full_name":    "Mitochondrial Acetoacetyl-CoA Thiolase (T2 / 3-Oxothiolase)",
                "protein_size": "427 aa / 45 kDa",
                "locus":        "11q22.3",
                "omim_gene":    "607809",
                "omim_disease": "203750",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "ACAT1 (T2) catalyzes thiolysis of 2-methylacetoacetyl-CoA → propionyl-CoA + acetyl-CoA "
                    "in isoleucine catabolism, AND thiolysis of acetoacetyl-CoA → 2 acetyl-CoA in "
                    "ketone body utilization. LOF → 2-methyl-3-OH-butyric acid + tiglylglycine + "
                    "2-methylacetoacetic acid accumulate during catabolism. "
                    "EPISODIC PATTERN: severe ketoacidosis during triggers (fasting, fever, high-protein diet) "
                    "with completely normal biochemistry between episodes. "
                    "Crises can be life-threatening (pH <6.8) but rapid bicarbonate + glucose = full recovery. "
                    "Excellent long-term prognosis — normal development if crises prevented."
                ),
                "pathognomonic": "Episodic severe ketoacidosis with NORMAL intervals; 2-methyl-3-OH-butyric acid in urine = PATHOGNOMONIC; C5-OH + C5:1 on NBS",
                "treatment":     "IV NaHCO3 (aggressive) + IV glucose during crises; avoid prolonged fasting; low-protein diet during illness; L-carnitine",
                "surveillance":  "Urine OA; plasma acylcarnitine; family education re: emergency protocol; MRI if neurological concerns",
                "key_ddx":       "HMGCL: hypoketotic (not hyperketotic); MSUD: elevated leucine/isoleucine/valine + allo-Ile; SCAD: different acylcarnitine (C4)",
                "omim_link":     "https://omim.org/entry/203750",
            },
        },
        "cascade": {
            "nbs_c3_pathway": {
                "trigger": "C3 (propionylcarnitine) elevated on NBS",
                "step1": "Urine organic acids: MMA quantitation + methylcitric acid",
                "step2a": "MMA elevated WITHOUT methylcitric acid → isolated MMA panel (MUT, MMAA, MMAB, MMACHC)",
                "step2b": "MMA + methylcitric acid both elevated → PA panel (PCCA, PCCB) + urine MMA to confirm",
                "step3": "Plasma homocysteine on ALL C3-positive: Hcy elevated → MMACHC (cblC) — add OHCbl + betaine",
                "step4": "B12 loading test: if MMA normalizes → likely cblA/cblB/cblC (responsive); mut0 = no response",
            },
            "c5_pathway": {
                "trigger": "C5 (isovalerylcarnitine) elevated on NBS",
                "step1": "Exclude pivalate drug exposure (pivampicillin, pivmecillinam)",
                "step2": "Urine OA: isovalerylglycine + 3-OH-isovaleric acid → confirm IVD",
                "step3": "IVD sequencing: A282V homozygous or compound → likely mild; functional assay recommended",
            },
            "c5dc_pathway": {
                "trigger": "C5DC (glutarylcarnitine) elevated on NBS",
                "step1": "Urine GA + 3-OH-GA: elevated → GA1 confirmed (high excretor); near-normal → low excretor GA1",
                "step2": "MRI brain: frontotemporal atrophy ('bat wings') → GA1 even if urine GA borderline",
                "step3": "GCDH sequencing: confirm; head circumference tracking; start emergency fever protocol",
            },
            "c6dc_pathway": {
                "trigger": "C6DC (hydroxymethylglutarylcarnitine) elevated",
                "step1": "Urine 3-HMG: elevated → HMGCL deficiency",
                "step2": "Plasma glucose + ketones simultaneously: hypoketotic hypoglycemia confirms",
                "step3": "Emergency: IV glucose IMMEDIATELY; HMGCL sequencing; fasting protocol",
            },
        },
    }
