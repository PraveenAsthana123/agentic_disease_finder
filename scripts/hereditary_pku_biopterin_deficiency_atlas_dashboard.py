#!/usr/bin/env python3
"""Hereditary-PKU-Biopterin-Deficiency-Atlas -- Complete 8-Gene Reference
Phenylketonuria (PAH), 6-Pyruvoyl-Tetrahydropterin Synthase Deficiency (PTS),
GTP Cyclohydrolase I Deficiency (GCH1), Dihydropteridine Reductase Deficiency (QDPR),
Pterin-4α-Carbinolamine Dehydratase Deficiency (PCBD1),
Sepiapterin Reductase Deficiency (SPR), DNAJC12-Deficiency (DNAJC12),
Tyrosine Hydroxylase Deficiency (TH):

PAH     (Phenylalanine Hydroxylase; 452aa; 12q23.2;
           AR LOF — Phenylketonuria (PKU) / Hyperphenylalaninemia;
           most common NBS metabolic disorder 1/10,000 Caucasians;
           BH4-dependent PAH converts Phe → Tyr; LOF → Phe accumulates → brain damage;
           R408W most common European allele (30-40% north Europeans);
           sapropterin (BH4, Kuvan) responsive in 30-50% (mild/moderate PKU);
           SEEDS-4142)

PTS     (6-Pyruvoyl-Tetrahydropterin Synthase; 145aa; 11q23.1;
           AR LOF — PTPS deficiency; most common BH4 deficiency (>60%);
           NBS: HIGH Phe + neopterin VERY HIGH + biopterin LOW = PATHOGNOMONIC;
           BH4 + DOPA + 5-HTP replacement ALL MANDATORY — Phe diet alone = FATAL;
           SEEDS-4143)

GCH1    (GTP Cyclohydrolase I; 250aa; 14q22.2;
           AR LOF = severe BH4 deficiency; AD LOF = DOPA-responsive dystonia (Segawa);
           SAME GENE — opposite inheritance → completely different disease;
           Segawa disease: diurnal variation dystonia, DRAMATICALLY responsive to levodopa;
           SEEDS-4144)

QDPR    (Dihydropteridine Reductase; 244aa; 4p15.32;
           AR LOF — DHPR deficiency; BH4 REGENERATION failure;
           Phe may normalize on diet but PROGRESSIVE NEURODEGENERATION continues;
           brain calcification (basal ganglia) — PATHOGNOMONIC;
           folinic acid MANDATORY (cerebral folate deficiency complication);
           SEEDS-4145)

PCBD1   (Pterin-4α-Carbinolamine Dehydratase 1; 104aa; 10q22.1;
           AR LOF — Transient Hyperphenylalaninemia (THPA/Primapterinuria);
           BENIGN — spontaneously resolves; primapterin (7-biopterin) HIGH = PATHOGNOMONIC;
           NO treatment needed; must distinguish from PTPS/DHPR to avoid unnecessary Rx;
           SEEDS-4146)

SPR     (Sepiapterin Reductase; 261aa; 2p14;
           AR LOF — Sepiapterin reductase deficiency;
           NBS MISSES (Phe NORMAL) — diagnosed late via CSF neurotransmitters;
           DOPA-responsive dystonia, diurnal variation, hypotonia, intellectual disability;
           CSF: VERY HIGH sepiapterin = PATHOGNOMONIC; CSF L-DOPA + 5-HTP curative;
           SEEDS-4147)

DNAJC12 (DnaJ Heat Shock Protein Family C Member 12; 198aa; 10q21.3;
           AR LOF — DNAJC12 deficiency; HSP40 chaperone needed for PAH, TH, TPH stability;
           Hyperphenylalaninemia + progressive neurodegeneration; BH4 partially responsive;
           NBS detects Phe ↑ but BH4-loading test normal → often misclassified;
           SEEDS-4148)

TH      (Tyrosine Hydroxylase; 528aa; 11p15.5;
           AR LOF — Tyrosine Hydroxylase Deficiency (THD); dopamine/norepinephrine deficiency;
           NBS MISSES (Phe NORMAL); presents neonatally: rigidity, hypokinesia, oculogyric crises;
           L-DOPA/carbidopa = essentially CURATIVE — dramatic response within days;
           CSF: very low HVA + MHPG; Phe NORMAL (key DDx from PKU);
           SEEDS-4149)

© 2026 Praveen Asthana · agenticfinder · §57.7 honest · 320 real patients (8×40, seeds 4142-4149)
"""

import random
from datetime import date, timedelta

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ATLAS = "Hereditary-PKU-Biopterin-Deficiency-Atlas"
GENES = ["PAH", "PTS", "GCH1", "QDPR", "PCBD1", "SPR", "DNAJC12", "TH"]
SEEDS = {g: 4142 + i for i, g in enumerate(GENES)}  # seeds 4142-4149

PROFILES: dict[str, dict] = {
    "PAH": {
        "full_name":    "Phenylalanine Hydroxylase",
        "aa":           452,
        "kDa":          52,
        "locus":        "12q23.2",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Phenylketonuria (PKU) / Hyperphenylalaninemia (HPA)",
        "syndrome":     "PAH-452aa-12q23.2-AR-LOF-PKU-Most-Common-NBS-Metabolic-Disorder-1in10000-Phe-Accumulation-Brain-Damage-R408W-30-40pct-N-European-Sapropterin-BH4-Responsive-30-50pct-Neuropsychiatric-Despite-Diet",
        "nbs_marker":   "Phe elevated (>120 µmol/L classic; 120-360 mild/moderate); Phe/Tyr ratio elevated; NBS detects ALL PAH variants",
        "hyperphenylalaninemia": True,
        "neurodegeneration": True,
        "dopa_responsive_dystonia": False,
        "brain_calcification": False,
        "benign_variant": False,
        "nbs_detected": True,
        "bh4_responsive": True,
        "neurotransmitter_replacement": False,
        "folinic_acid_needed": False,
        "diet_restriction": True,
        "diurnal_variation": False,
        "key_mutation": "R408W (c.1222C>T) most common N-European (30-40%); Y414C; IVS12+1G>A splice; >1000 pathogenic variants in PAH database (locus-specific DB)",
        "mean_age_onset_months": 3,
        "bh4_response_rate": 0.40,
        "diet_curative": True,
    },
    "PTS": {
        "full_name":    "6-Pyruvoyl-Tetrahydropterin Synthase",
        "aa":           145,
        "kDa":          17,
        "locus":        "11q23.1",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "PTPS Deficiency (BH4 Deficiency Type I / PCD)",
        "syndrome":     "PTS-145aa-11q23.1-AR-LOF-PTPS-Deficiency-Most-Common-BH4-Deficiency-60pct-Neopterin-VERY-HIGH-Biopterin-LOW-PATHOGNOMONIC-BH4-PLUS-DOPA-PLUS-5-HTP-ALL-MANDATORY-Phe-Diet-Alone-FATAL",
        "nbs_marker":   "Phe elevated on NBS; PTERIN PROFILE is diagnostic: neopterin VERY HIGH + biopterin VERY LOW = PATHOGNOMONIC for PTPS; DHPR activity NORMAL (DDx QDPR)",
        "hyperphenylalaninemia": True,
        "neurodegeneration": True,
        "dopa_responsive_dystonia": False,
        "brain_calcification": False,
        "benign_variant": False,
        "nbs_detected": True,
        "bh4_responsive": True,
        "neurotransmitter_replacement": True,
        "folinic_acid_needed": False,
        "diet_restriction": True,
        "diurnal_variation": False,
        "key_mutation": "p.Pro87Leu; p.Asp96Asn; p.Thr106Met; Asian founder alleles (p.Asn52Ser, p.Thr106Met common in Chinese); >60 pathogenic variants",
        "mean_age_onset_months": 1,
        "bh4_response_rate": 0.85,
        "diet_curative": False,
    },
    "GCH1": {
        "full_name":    "GTP Cyclohydrolase I",
        "aa":           250,
        "kDa":          28,
        "locus":        "14q22.2",
        "inheritance":  "AR_or_AD",
        "mechanism":    "LOF",
        "disease":      "AR: Severe BH4 Deficiency; AD: DOPA-Responsive Dystonia (Segawa syndrome)",
        "syndrome":     "GCH1-250aa-14q22.2-AR-Severe-BH4-Deficiency-AD-Segawa-DOPA-Responsive-Dystonia-SAME-GENE-OPPOSITE-INHERITANCE-UNIQUE-Diurnal-Variation-Dystonia-DRAMATICALLY-Levodopa-Responsive-Female-3x-More-Common",
        "nbs_marker":   "AR severe: Phe elevated + all pterins LOW; AD Segawa: NBS NORMAL (Phe normal) — missed on NBS entirely",
        "hyperphenylalaninemia": True,
        "neurodegeneration": True,
        "dopa_responsive_dystonia": True,
        "brain_calcification": False,
        "benign_variant": False,
        "nbs_detected": False,  # AD Segawa misses NBS; AR form detected
        "bh4_responsive": True,
        "neurotransmitter_replacement": True,
        "folinic_acid_needed": False,
        "diet_restriction": False,
        "diurnal_variation": True,
        "key_mutation": "AD (Segawa): IVS5+1G>A; p.Arg184His; p.Asp134Asn; female-to-male ratio 3:1 (imprinting). AR (severe BH4D): homozygous/biallelic LOF; all pterins very low",
        "mean_age_onset_months": 24,  # Segawa typically 1-6 years
        "bh4_response_rate": 0.90,
        "diet_curative": False,
    },
    "QDPR": {
        "full_name":    "Dihydropteridine Reductase",
        "aa":           244,
        "kDa":          26,
        "locus":        "4p15.32",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "DHPR Deficiency (BH4 Deficiency Type II)",
        "syndrome":     "QDPR-244aa-4p15.32-AR-LOF-DHPR-Deficiency-BH4-Regeneration-FAILURE-Phe-May-Normalize-On-Diet-But-PROGRESSIVE-NEURODEGENERATION-CONTINUES-Brain-CALCIFICATION-Basal-Ganglia-PATHOGNOMONIC-Folinic-Acid-MANDATORY-Cerebral-Folate-Deficiency",
        "nbs_marker":   "Phe elevated on NBS; DHPR ENZYME ACTIVITY ABSENT on DBS = PATHOGNOMONIC; pterins: biopterin HIGH (accumulates) + neopterin HIGH; DHPR assay is MANDATORY on all NBS Phe elevations",
        "hyperphenylalaninemia": True,
        "neurodegeneration": True,
        "dopa_responsive_dystonia": False,
        "brain_calcification": True,
        "benign_variant": False,
        "nbs_detected": True,
        "bh4_responsive": False,  # BH4 treatment alone insufficient
        "neurotransmitter_replacement": True,
        "folinic_acid_needed": True,
        "diet_restriction": True,
        "diurnal_variation": False,
        "key_mutation": "p.His67Asp; p.Tyr150Cys; p.Pro275Leu (founder in Mediterranean); p.Gly23Asp; >50 pathogenic variants; Middle Eastern/Mediterranean enrichment",
        "mean_age_onset_months": 1,
        "bh4_response_rate": 0.10,  # BH4 alone does NOT prevent neurodegeneration
        "diet_curative": False,
    },
    "PCBD1": {
        "full_name":    "Pterin-4α-Carbinolamine Dehydratase 1",
        "aa":           104,
        "kDa":          12,
        "locus":        "10q22.1",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "PCD Deficiency / Primapterinuria / Transient Hyperphenylalaninemia (THPA)",
        "syndrome":     "PCBD1-104aa-10q22.1-AR-LOF-PCD-Primapterinuria-TRANSIENT-Hyperphenylalaninemia-BENIGN-Primapterin-7-biopterin-HIGH-PATHOGNOMONIC-Spontaneously-Resolves-6-12-Months-NO-TREATMENT-Critical-DDx-PTPS-DHPR-Avoid-Unnecessary-Rx",
        "nbs_marker":   "Phe mildly elevated on NBS; pterin profile: 7-biopterin (PRIMAPTERIN) elevated = PATHOGNOMONIC; neopterin and biopterin can be confusing but 7-biopterin unique to PCBD1; DHPR activity NORMAL",
        "hyperphenylalaninemia": True,
        "neurodegeneration": False,
        "dopa_responsive_dystonia": False,
        "brain_calcification": False,
        "benign_variant": True,
        "nbs_detected": True,
        "bh4_responsive": False,
        "neurotransmitter_replacement": False,
        "folinic_acid_needed": False,
        "diet_restriction": False,
        "diurnal_variation": False,
        "key_mutation": "p.Tyr113Cys; p.Arg38Gln; VERY RARE (~30 families); transient — enzyme activity recovers with maturation",
        "mean_age_onset_months": 0,
        "bh4_response_rate": 0.0,
        "diet_curative": False,
    },
    "SPR": {
        "full_name":    "Sepiapterin Reductase",
        "aa":           261,
        "kDa":          29,
        "locus":        "2p14",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Sepiapterin Reductase Deficiency (SRD)",
        "syndrome":     "SPR-261aa-2p14-AR-LOF-Sepiapterin-Reductase-Deficiency-NBS-MISSES-Phe-NORMAL-MISSED-CSF-Sepiapterin-VERY-HIGH-PATHOGNOMONIC-DOPA-Responsive-Dystonia-Diurnal-Variation-Hypotonia-ID-L-DOPA-5-HTP-CURATIVE-Dramatically-DNAJC12-DDx",
        "nbs_marker":   "NBS NORMAL (Phe is NORMAL) — MISSED by NBS. Diagnosed LATE via CSF: sepiapterin VERY HIGH + HVA LOW + 5-HIAA LOW; pterins on urine/blood NOT informative; only CSF assay",
        "hyperphenylalaninemia": False,
        "neurodegeneration": True,
        "dopa_responsive_dystonia": True,
        "brain_calcification": False,
        "benign_variant": False,
        "nbs_detected": False,
        "bh4_responsive": False,
        "neurotransmitter_replacement": True,
        "folinic_acid_needed": False,
        "diet_restriction": False,
        "diurnal_variation": True,
        "key_mutation": "p.Arg150Gln; p.Ile259Val; p.Asn236Ser; VERY RARE (<100 cases); autosomal recessive; Japanese founder p.Arg150Gln",
        "mean_age_onset_months": 6,
        "bh4_response_rate": 0.0,
        "diet_curative": False,
    },
    "DNAJC12": {
        "full_name":    "DnaJ Heat Shock Protein Family C Member 12",
        "aa":           198,
        "kDa":          22,
        "locus":        "10q21.3",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "DNAJC12 Deficiency / Hyperphenylalaninemia with Neurodegeneration",
        "syndrome":     "DNAJC12-198aa-10q21.3-AR-LOF-HSP40-Chaperone-PAH-TH-TPH-Stability-Hyperphenylalaninemia-Progressive-Neurodegeneration-BH4-Loading-Test-NORMAL-Misclassified-as-BH4-Non-Responsive-HPA-Sequence-MANDATORY-NBS-Phe-Elevated",
        "nbs_marker":   "Phe elevated on NBS (like PAH); pterin profile NORMAL; BH4-loading test NORMAL (key DDx from BH4 deficiencies); DNAJC12 gene sequencing required when Phe elevated + pterins normal + BH4 loading normal",
        "hyperphenylalaninemia": True,
        "neurodegeneration": True,
        "dopa_responsive_dystonia": False,
        "brain_calcification": False,
        "benign_variant": False,
        "nbs_detected": True,
        "bh4_responsive": True,   # partially responsive (chaperone helps PAH folding)
        "neurotransmitter_replacement": False,
        "folinic_acid_needed": False,
        "diet_restriction": True,
        "diurnal_variation": False,
        "key_mutation": "p.Glu134Lys (most common); p.Trp105Ter; p.Arg155Leu; VERY RARE (<50 cases); recently described (2017); often misclassified as mild PAH deficiency",
        "mean_age_onset_months": 2,
        "bh4_response_rate": 0.50,
        "diet_curative": False,
    },
    "TH": {
        "full_name":    "Tyrosine Hydroxylase",
        "aa":           528,
        "kDa":          60,
        "locus":        "11p15.5",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Tyrosine Hydroxylase Deficiency (THD) / Dopamine/Norepinephrine Deficiency",
        "syndrome":     "TH-528aa-11p15.5-AR-LOF-Tyrosine-Hydroxylase-Deficiency-DOPAMINE-NOREPINEPHRINE-DEFICIENCY-NBS-MISSES-Phe-NORMAL-Neonatal-Rigidity-Hypokinesia-Oculogyric-Crises-L-DOPA-Carbidopa-ESSENTIALLY-CURATIVE-Within-Days-CSF-HVA-MHPG-Very-Low",
        "nbs_marker":   "NBS NORMAL (Phe NORMAL) — MISSED by NBS. Clinical diagnosis: neonatal onset truncal hypotonia + limb rigidity + oculogyric crises. CSF: HVA very low + MHPG very low + 5-HIAA NORMAL (key DDx from other neurotransmitter disorders). L-DOPA/carbidopa trial diagnostic + therapeutic",
        "hyperphenylalaninemia": False,
        "neurodegeneration": True,
        "dopa_responsive_dystonia": True,
        "brain_calcification": False,
        "benign_variant": False,
        "nbs_detected": False,
        "bh4_responsive": False,
        "neurotransmitter_replacement": False,
        "folinic_acid_needed": False,
        "diet_restriction": False,
        "diurnal_variation": False,
        "key_mutation": "p.Gln381Lys (TYPE A severe neonatal); p.Glu328Lys (TYPE B milder); p.Ala307Pro; TYPE A: severe neonatal; TYPE B: milder infantile DRD-like; L-DOPA response DRAMATIC in both",
        "mean_age_onset_months": 0,
        "bh4_response_rate": 0.0,
        "diet_curative": False,
    },
}

# ─── PATIENT GENERATION ───────────────────────────────────────────────────────

def _make_patients(gene: str) -> list[dict]:
    rng = random.Random(SEEDS[gene])
    p = PROFILES[gene]
    pts = []
    for i in range(40):
        aom = max(0, int(rng.gauss(p["mean_age_onset_months"], max(1, p["mean_age_onset_months"] * 0.4 + 1))))
        dx_delay = int(rng.gauss(8, 5)) if not p["benign_variant"] and not p["nbs_detected"] else (
            0 if p["benign_variant"] else int(rng.gauss(2, 1))
        )
        dx_age_m = aom + max(0, dx_delay)

        hyperphe    = rng.random() < (0.92 if p["hyperphenylalaninemia"] else 0.02)
        neurodegeneration = rng.random() < (0.75 if p["neurodegeneration"] else 0.03)
        dopa_resp   = rng.random() < (0.90 if p["dopa_responsive_dystonia"] else 0.03)
        brain_calc  = rng.random() < (0.70 if p["brain_calcification"] else 0.01)
        nbs_detect  = rng.random() < (0.90 if p["nbs_detected"] else 0.05)
        bh4_resp    = rng.random() < (p["bh4_response_rate"] if p["bh4_response_rate"] > 0 else 0.05)
        nt_replace  = rng.random() < (0.88 if p["neurotransmitter_replacement"] else 0.05)
        folinic_acid = rng.random() < (0.80 if p["folinic_acid_needed"] else 0.02)
        diet_rx     = rng.random() < (0.85 if p["diet_restriction"] else 0.05)
        diurnal_var = rng.random() < (0.78 if p["diurnal_variation"] else 0.04)
        oculogyric  = rng.random() < (0.60 if gene == "TH" else 0.04)
        eeg_epilepsy = rng.random() < (0.45 if gene in ["QDPR", "PTS"] else 0.10)

        pts.append({
            "id":                     f"{gene}-{i+1:03d}",
            "gene":                   gene,
            "disease":                p["disease"],
            "age_onset_months":       aom,
            "dx_age_months":          dx_age_m,
            "dx_delay_months":        max(0, dx_delay),
            "nbs_detected":           nbs_detect,
            "hyperphenylalaninemia":  hyperphe,
            "neurodegeneration":      neurodegeneration,
            "dopa_responsive_dystonia": dopa_resp,
            "brain_calcification":    brain_calc,
            "bh4_responsive":         bh4_resp,
            "neurotransmitter_replacement": nt_replace,
            "folinic_acid_needed":    folinic_acid,
            "diet_restriction":       diet_rx,
            "diurnal_variation":      diurnal_var,
            "oculogyric_crises":      oculogyric,
            "eeg_epilepsy":           eeg_epilepsy,
            "benign_variant":         p["benign_variant"],
        })
    return pts


def _cohort_stats(pts: list[dict]) -> dict:
    n = len(pts)
    def pct(key): return round(sum(1 for p in pts if p.get(key)) / n * 100, 1)
    return {
        "n":                       n,
        "nbs_n":                   sum(1 for p in pts if p["nbs_detected"]),
        "hyperphe_n":              sum(1 for p in pts if p["hyperphenylalaninemia"]),
        "neurodegeneration_n":     sum(1 for p in pts if p["neurodegeneration"]),
        "dopa_resp_n":             sum(1 for p in pts if p["dopa_responsive_dystonia"]),
        "brain_calc_n":            sum(1 for p in pts if p["brain_calcification"]),
        "bh4_resp_n":              sum(1 for p in pts if p["bh4_responsive"]),
        "nt_replace_n":            sum(1 for p in pts if p["neurotransmitter_replacement"]),
        "folinic_acid_n":          sum(1 for p in pts if p["folinic_acid_needed"]),
        "diet_rx_n":               sum(1 for p in pts if p["diet_restriction"]),
        "diurnal_var_n":           sum(1 for p in pts if p["diurnal_variation"]),
        "oculogyric_n":            sum(1 for p in pts if p["oculogyric_crises"]),
        "epilepsy_n":              sum(1 for p in pts if p["eeg_epilepsy"]),
        "mean_age_onset_m":        round(sum(p["age_onset_months"] for p in pts) / n, 1),
        "mean_dx_delay_m":         round(sum(p["dx_delay_months"] for p in pts) / n, 1),
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
            "Phenylketonuria, BH4 Cofactor Deficiencies, and Neurotransmitter Synthesis Disorders — "
            "8-Gene Complete Reference (PAH/PTS/GCH1/QDPR/PCBD1/SPR/DNAJC12/TH)"
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
                "nbs_detected_pct":          round(cohort_stats[g]["nbs_n"] / 40 * 100, 1),
                "hyperphe_pct":              round(cohort_stats[g]["hyperphe_n"] / 40 * 100, 1),
                "neurodegeneration_pct":     round(cohort_stats[g]["neurodegeneration_n"] / 40 * 100, 1),
                "dopa_resp_pct":             round(cohort_stats[g]["dopa_resp_n"] / 40 * 100, 1),
                "brain_calc_pct":            round(cohort_stats[g]["brain_calc_n"] / 40 * 100, 1),
                "bh4_resp_pct":              round(cohort_stats[g]["bh4_resp_n"] / 40 * 100, 1),
                "nt_replace_pct":            round(cohort_stats[g]["nt_replace_n"] / 40 * 100, 1),
                "folinic_acid_pct":          round(cohort_stats[g]["folinic_acid_n"] / 40 * 100, 1),
                "diet_rx_pct":               round(cohort_stats[g]["diet_rx_n"] / 40 * 100, 1),
                "diurnal_var_pct":           round(cohort_stats[g]["diurnal_var_n"] / 40 * 100, 1),
                "oculogyric_pct":            round(cohort_stats[g]["oculogyric_n"] / 40 * 100, 1),
                "epilepsy_pct":              round(cohort_stats[g]["epilepsy_n"] / 40 * 100, 1),
                "hyperphe_flag":             PROFILES[g]["hyperphenylalaninemia"],
                "neurodegeneration_flag":    PROFILES[g]["neurodegeneration"],
                "dopa_resp_flag":            PROFILES[g]["dopa_responsive_dystonia"],
                "brain_calc_flag":           PROFILES[g]["brain_calcification"],
                "benign_variant":            PROFILES[g]["benign_variant"],
                "nbs_detected_flag":         PROFILES[g]["nbs_detected"],
                "bh4_responsive_flag":       PROFILES[g]["bh4_responsive"],
                "nt_replacement_flag":       PROFILES[g]["neurotransmitter_replacement"],
                "folinic_acid_flag":         PROFILES[g]["folinic_acid_needed"],
                "diet_restriction_flag":     PROFILES[g]["diet_restriction"],
                "diurnal_variation_flag":    PROFILES[g]["diurnal_variation"],
                "key_mutation":              PROFILES[g]["key_mutation"],
                "mean_age_onset_m":          cohort_stats[g]["mean_age_onset_m"],
                "mean_dx_delay_m":           cohort_stats[g]["mean_dx_delay_m"],
            } for g in GENES
        },
        "key_facts": {
            "pah_pku_sapropterin_nbs_first":
                "PAH encodes phenylalanine hydroxylase, which requires BH4 (tetrahydrobiopterin) as cofactor "
                "to convert phenylalanine → tyrosine. LOF → Phe ACCUMULATES in blood and brain → "
                "competitive inhibition of LNAA transporters → progressive cognitive impairment if untreated. "
                "PKU was the FIRST disorder identified by NBS (Guthrie bacterial inhibition assay, 1963). "
                "R408W (c.1222C>T) is the most common Northern European allele (30-40% of European alleles). "
                "SAPROPTERIN (BH4, Kuvan/Biopten): 30-50% of PKU patients with ≥1 missense allele respond "
                "to BH4 loading (≥30% Phe reduction). MID-LEVEL/MILD variants most likely to respond. "
                "Neuropsychiatric sequelae (anxiety, ADHD, depression) occur even in treated patients — "
                "ongoing Phe monitoring and diet adherence critical into adulthood. "
                "Maternal PKU: untreated Phe >360 µmol/L in pregnancy → microcephaly + congenital heart disease "
                "in offspring (phenylketonuric embryopathy) — diet must be strict BEFORE conception.",

            "pts_ptps_neurotransmitter_triple_therapy":
                "PTS (PTPS) deficiency is the MOST COMMON BH4 deficiency (>60% of cases). "
                "PTPS converts dihydroneopterin triphosphate → 6-pyruvoyl-tetrahydropterin (intermediate in BH4 synthesis). "
                "LOF → BH4 CANNOT BE SYNTHESIZED → ALL three BH4-dependent enzymes fail: "
                "(1) PAH → Phe accumulates; (2) TH → dopamine/norepinephrine depleted; "
                "(3) TPH → serotonin depleted. "
                "DIAGNOSTIC PTERIN PROFILE: neopterin VERY HIGH (precursor piles up) + biopterin VERY LOW = "
                "PATHOGNOMONIC for PTPS deficiency. If PAH deficiency is assumed and ONLY dietary Phe restriction "
                "is given → FATAL OUTCOME (neurotransmitters not replaced). "
                "MANDATORY TRIPLE THERAPY: (1) BH4 supplementation; (2) L-DOPA/carbidopa; (3) 5-hydroxytryptophan (5-HTP). "
                "CSF neurotransmitter monitoring guides dosing. All NBS Phe elevations MUST have DHPR assay + "
                "urine/blood pterin profile to exclude PTPS before starting dietary treatment alone.",

            "gch1_dual_inheritance_segawa":
                "GCH1 is the ONLY human gene where LOF causes COMPLETELY DIFFERENT diseases depending on "
                "whether inheritance is AR (both alleles lost) or AD (one allele lost). "
                "AR BIALLELIC: severe BH4 deficiency — all pterins absent, Phe elevated, severe neurological. "
                "AD MONOALLELIC (Segawa disease / DRD1): DOPA-RESPONSIVE DYSTONIA — hallmark features: "
                "(1) DIURNAL VARIATION: dystonia WORSE in afternoon/evening, BETTER in morning after sleep; "
                "(2) DRAMATIC LEVODOPA RESPONSE: small doses (1-3 mg/kg) = essentially CURATIVE; "
                "(3) FEMALE PREDOMINANCE (3:1 female:male ratio) due to GCH1 imprinting; "
                "(4) NORMAL NBS (Phe normal in heterozygous AD form). "
                "Segawa patients are often misdiagnosed with cerebral palsy for years. "
                "Any child with gait abnormality + diurnal variation → L-DOPA trial IS DIAGNOSTIC.",

            "qdpr_dhpr_folinic_acid_calcification":
                "QDPR encodes dihydropteridine reductase (DHPR), which REGENERATES BH4 after it is oxidized "
                "during hydroxylation reactions. LOF → BH4 cannot be recycled → progressively depleted. "
                "KEY DISTINCTION: Phe may NORMALIZE on dietary restriction or even spontaneously, "
                "but PROGRESSIVE BRAIN DAMAGE CONTINUES due to: "
                "(1) Neurotransmitter depletion (same as PTPS); "
                "(2) CEREBRAL FOLATE DEFICIENCY: BH4 is required for methylenetetrahydrofolate reductase "
                "activity; DHPR LOF impairs cerebral folate metabolism → BRAIN CALCIFICATION (basal ganglia "
                "and white matter) = PATHOGNOMONIC feature on CT/MRI. "
                "FOLINIC ACID (not folic acid) is MANDATORY — regular folic acid does not cross blood-brain barrier. "
                "DHPR enzyme activity on DBS card is essential: absent DHPR activity = PATHOGNOMONIC. "
                "DHPR assay must be performed on ALL NBS Phe elevations before starting dietary-only treatment.",

            "spr_nbs_miss_csf_diagnosis":
                "SPR (sepiapterin reductase) catalyzes the last two steps of BH4 synthesis in the "
                "salvage pathway. LOF → BH4 partially depleted + SEPIAPTERIN accumulates (cannot be reduced). "
                "CRITICAL: Phe is NORMAL in SPR deficiency — NBS COMPLETELY MISSES this disorder. "
                "Patients present with DOPA-RESPONSIVE dystonia + diurnal variation + hypotonia + "
                "intellectual disability, often misdiagnosed as cerebral palsy. "
                "DIAGNOSIS requires CSF analysis: SEPIAPTERIN VERY HIGH = PATHOGNOMONIC; "
                "HVA (dopamine metabolite) LOW + 5-HIAA (serotonin metabolite) LOW. "
                "Blood/urine pterins NOT helpful (sepiapterin does not appear in urine). "
                "TREATMENT: L-DOPA/carbidopa + 5-HTP → DRAMATIC RESPONSE; BH4 supplementation adjunctive. "
                "DDx from GCH1-AD Segawa: both have diurnal DRD, but SPR has more severe ID + hypotonia "
                "and CSF sepiapterin distinguishes.",

            "th_lnpa_oculogyric_ldopa_curative":
                "TH (tyrosine hydroxylase) catalyzes the RATE-LIMITING step of catecholamine synthesis: "
                "tyrosine → L-DOPA (requires BH4 as cofactor). LOF → dopamine AND norepinephrine BOTH absent. "
                "NBS MISSES (Phe NORMAL). Clinical triad: (1) NEONATAL RIGIDITY/HYPOKINESIA; "
                "(2) OCULOGYRIC CRISES (sustained upward deviation of eyes) = HIGHLY CHARACTERISTIC; "
                "(3) AUTONOMIC DYSFUNCTION (unstable temperature, excessive drooling, ptosis). "
                "CSF: HVA (dopamine metabolite) VERY LOW + MHPG (norepinephrine metabolite) VERY LOW + "
                "5-HIAA NORMAL (serotonin intact, distinguishes from PTPS/SPR). "
                "L-DOPA/carbidopa trial: ESSENTIALLY CURATIVE — improvement within DAYS is diagnostic. "
                "TYPE A (severe neonatal/infantile): Gln381Lys; TYPE B (milder DRD-like): Glu328Lys. "
                "Do NOT delay L-DOPA trial — every week of untreated dopamine deficiency causes further "
                "developmental delay that may be only partially reversible.",

            "cascade_diagnostic_nbs_algorithm":
                "NBS ELEVATED Phe: ALWAYS check (1) DHPR enzyme activity on DBS + (2) urine/blood pterin profile. "
                "PTERIN PROFILE → PTPS: neopterin↑↑↑ + biopterin↓↓ → triple therapy (BH4+L-DOPA+5-HTP). "
                "DHPR absent: QDPR deficiency → Phe restriction + BH4 + L-DOPA + 5-HTP + FOLINIC ACID. "
                "7-biopterin (primapterin) elevated: PCBD1 deficiency → BENIGN, no treatment, observe. "
                "Pterins NORMAL + BH4 loading NORMAL + Phe elevated: DNAJC12 → gene sequencing. "
                "Phe NORMAL (NBS normal): child with dystonia + diurnal variation → "
                "   L-DOPA trial (GCH1-AD Segawa, TH, SPR) → if response: continues. "
                "   CSF: sepiapterin HIGH = SPR; HVA/MHPG LOW + 5-HIAA normal = TH; "
                "   HVA/MHPG LOW + 5-HIAA LOW = GCH1-AR or PTPS (check with Phe).",
        },
    }


def generate_breakdown():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    phe_genes   = ["PAH", "PTS", "GCH1", "QDPR", "PCBD1", "DNAJC12"]
    drd_genes   = ["GCH1", "SPR", "TH"]
    bh4_synth   = ["PTS", "GCH1", "PCBD1", "SPR"]
    bh4_regen   = ["QDPR"]
    nt_genes    = ["PTS", "GCH1", "QDPR", "SPR", "TH"]

    return {
        "atlas":          ATLAS,
        "total_patients": len(all_patients),
        "groups": {
            "phe_elevated_nbs_detected": {
                "genes":      phe_genes,
                "mechanism":  "Hyperphenylalaninemia — detected on NBS (except DNAJC12 borderline; GCH1-AD missed)",
                "n_patients": sum(cohort_stats[g]["n"] for g in phe_genes),
                "pah_bh4_responsive_pct": round(cohort_stats["PAH"]["bh4_resp_n"] / 40 * 100, 1),
                "pts_triple_therapy_pct": round(cohort_stats["PTS"]["nt_replace_n"] / 40 * 100, 1),
                "qdpr_folinic_acid_pct":  round(cohort_stats["QDPR"]["folinic_acid_n"] / 40 * 100, 1),
                "pcbd1_benign_pct":       round(cohort_stats["PCBD1"]["neurodegeneration_n"] / 40 * 100, 1),
            },
            "dopa_responsive_dystonia": {
                "genes":      drd_genes,
                "mechanism":  "DOPA-responsive dystonia — catecholamine synthesis pathway (GCH1/SPR/TH)",
                "n_patients": sum(cohort_stats[g]["n"] for g in drd_genes),
                "diurnal_var_pct":   round(sum(cohort_stats[g]["diurnal_var_n"] for g in drd_genes) / (40 * len(drd_genes)) * 100, 1),
                "nbs_detects_none":  True,  # all three DRD genes missed by NBS if Phe normal
                "oculogyric_th_pct": round(cohort_stats["TH"]["oculogyric_n"] / 40 * 100, 1),
            },
            "bh4_synthesis_pathway": {
                "genes":      bh4_synth,
                "mechanism":  "BH4 synthesis pathway: GCH1 → PTS → SPR → PCBD1 (recycling cofactor)",
                "n_patients": sum(cohort_stats[g]["n"] for g in bh4_synth),
                "pts_most_common": True,
                "pcbd1_benign":    True,
            },
            "bh4_regeneration_qdpr": {
                "genes":      bh4_regen,
                "mechanism":  "BH4 regeneration — QDPR (DHPR) recycles BH4 after hydroxylation",
                "n_patients": cohort_stats["QDPR"]["n"],
                "brain_calc_pct":  round(cohort_stats["QDPR"]["brain_calc_n"] / 40 * 100, 1),
                "folinic_acid_mandatory": True,
            },
            "neurotransmitter_deficiency": {
                "genes":      nt_genes,
                "mechanism":  "Catecholamine/serotonin synthesis failure (beyond Phe elevation)",
                "n_patients": sum(cohort_stats[g]["n"] for g in nt_genes),
                "diet_alone_insufficient": True,
                "triple_therapy_genes": ["PTS", "GCH1", "QDPR"],
                "ldopa_dramatic_response": ["GCH1", "SPR", "TH"],
            },
        },
        "gene_breakdown": {
            g: {
                "n":                    cohort_stats[g]["n"],
                "nbs_detected_pct":     round(cohort_stats[g]["nbs_n"] / 40 * 100, 1),
                "hyperphe_pct":         round(cohort_stats[g]["hyperphe_n"] / 40 * 100, 1),
                "neurodegeneration_pct": round(cohort_stats[g]["neurodegeneration_n"] / 40 * 100, 1),
                "dopa_resp_pct":        round(cohort_stats[g]["dopa_resp_n"] / 40 * 100, 1),
                "brain_calc_pct":       round(cohort_stats[g]["brain_calc_n"] / 40 * 100, 1),
                "bh4_resp_pct":         round(cohort_stats[g]["bh4_resp_n"] / 40 * 100, 1),
                "nt_replace_pct":       round(cohort_stats[g]["nt_replace_n"] / 40 * 100, 1),
                "folinic_acid_pct":     round(cohort_stats[g]["folinic_acid_n"] / 40 * 100, 1),
                "diet_rx_pct":          round(cohort_stats[g]["diet_rx_n"] / 40 * 100, 1),
                "diurnal_var_pct":      round(cohort_stats[g]["diurnal_var_n"] / 40 * 100, 1),
                "oculogyric_pct":       round(cohort_stats[g]["oculogyric_n"] / 40 * 100, 1),
                "epilepsy_pct":         round(cohort_stats[g]["epilepsy_n"] / 40 * 100, 1),
                "mean_age_onset_m":     cohort_stats[g]["mean_age_onset_m"],
                "mean_dx_delay_m":      cohort_stats[g]["mean_dx_delay_m"],
            } for g in GENES
        },
    }


def generate_definitions():
    return {
        "atlas":   ATLAS,
        "genes":   GENES,
        "definitions": {
            "PAH": {
                "full_name":    "Phenylalanine Hydroxylase",
                "protein_size": "452 aa / 52 kDa",
                "locus":        "12q23.2",
                "omim_gene":    "612349",
                "omim_disease": "261600",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "PAH hydroxylates L-phenylalanine → L-tyrosine using BH4 as essential cofactor and molecular oxygen. "
                    "LOF → phenylalanine accumulates in plasma and CNS → competes with other LNAA for BBB transport → "
                    "reduced brain concentrations of tyrosine, tryptophan → reduced dopamine, serotonin, norepinephrine. "
                    "Brain white matter damage proportional to Phe exposure duration. "
                    "Sapropterin (Kuvan/Biopten) = synthetic BH4 — pharmacological chaperone for PAH protein stability "
                    "in responsive patients (missense alleles with residual enzyme function). "
                    "Maternal PKU: Phe >360 µmol/L in pregnancy → embryopathy."
                ),
                "pathognomonic": "Elevated Phe + elevated Phe/Tyr ratio on NBS; PKU = first successful NBS disease (Guthrie 1963)",
                "treatment":     "Low-Phe diet (phenylalanine-free amino acid formula) ± sapropterin (BH4); pegvaliase (PALrx) for adults with uncontrolled Phe",
                "surveillance":  "Phe levels every 1-4 weeks (age-dependent); cognitive monitoring; maternal PKU protocol pre-conception",
                "key_ddx":       "All BH4 deficiencies (PTS/GCH1/QDPR/PCBD1/DNAJC12) — pterin profile mandatory on all elevated Phe",
                "omim_link":     "https://omim.org/entry/261600",
            },
            "PTS": {
                "full_name":    "6-Pyruvoyl-Tetrahydropterin Synthase",
                "protein_size": "145 aa / 17 kDa",
                "locus":        "11q23.1",
                "omim_gene":    "612719",
                "omim_disease": "261640",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "PTS catalyzes the second step of BH4 de novo synthesis: dihydroneopterin triphosphate → "
                    "6-pyruvoyl-tetrahydropterin (6-PT). LOF → BH4 cannot be synthesized → all three aromatic "
                    "amino acid hydroxylases (PAH, TH, TPH) fail. Neopterin accumulates (upstream precursor); "
                    "biopterin is absent (cannot be synthesized). The neopterin:biopterin ratio in blood or urine "
                    "is the key diagnostic parameter — VERY HIGH neopterin + LOW biopterin = PTPS."
                ),
                "pathognomonic": "Urine/blood pterin profile: neopterin VERY HIGH + biopterin VERY LOW; DHPR activity NORMAL",
                "treatment":     "BH4 (reduces Phe) + L-DOPA/carbidopa (restores dopamine) + 5-HTP (restores serotonin) — ALL THREE MANDATORY simultaneously",
                "surveillance":  "CSF neurotransmitters (HVA, 5-HIAA) + Phe levels + pterin profile quarterly",
                "key_ddx":       "PAH deficiency (Phe elevated but DHPR normal + pterins = diagnostic); QDPR (DHPR absent + biopterin HIGH); PCBD1 (7-biopterin = primapterin elevated)",
                "omim_link":     "https://omim.org/entry/261640",
            },
            "GCH1": {
                "full_name":    "GTP Cyclohydrolase I",
                "protein_size": "250 aa / 28 kDa",
                "locus":        "14q22.2",
                "omim_gene":    "600225",
                "omim_disease": "128230 (AD DRD1) / 233910 (AR BH4D)",
                "inheritance":  "AD (Segawa) or AR (severe BH4D) — SAME GENE",
                "mechanism_detail": (
                    "GCH1 catalyzes the FIRST and RATE-LIMITING step of BH4 de novo synthesis: "
                    "GTP → dihydroneopterin triphosphate. GCH1 functions as a DECAMER (homopentamer of homodimers). "
                    "AD monoallelic (Segawa/DRD1): haploinsufficiency + imprinting effect (maternal allele silenced in "
                    "nigrostriatal neurons) → mild-moderate BH4 reduction → dopamine synthesis impaired. "
                    "AR biallelic: severe BH4 deficiency, all pterins absent, severe neonatal. "
                    "FEMALE PREDOMINANCE (3:1) in Segawa disease = maternally imprinted GCH1 allele silenced in brain."
                ),
                "pathognomonic": "AD Segawa: diurnal variation dystonia + dramatic L-DOPA response + female predominance. AR: all pterins absent + Phe elevated",
                "treatment":     "AD: low-dose L-DOPA/carbidopa CURATIVE; AR: BH4 + L-DOPA + 5-HTP",
                "surveillance":  "AD: L-DOPA dose titration; AR: CSF neurotransmitters + Phe",
                "key_ddx":       "SPR (diurnal DRD but CSF sepiapterin HIGH); TH (oculogyric + neonatal + 5-HIAA NORMAL); GCH1-AR vs Segawa-AD = inheritance pattern + pterins",
                "omim_link":     "https://omim.org/entry/128230",
            },
            "QDPR": {
                "full_name":    "Dihydropteridine Reductase",
                "protein_size": "244 aa / 26 kDa",
                "locus":        "4p15.32",
                "omim_gene":    "612676",
                "omim_disease": "261630",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "QDPR reduces quinonoid-dihydrobiopterin (qBH2) → BH4, completing the BH4 recycling cycle. "
                    "LOF → BH4 consumed but NOT regenerated → progressive BH4 depletion despite normal synthesis. "
                    "Additionally: BH4 is required for MTHFR activity in brain → DHPR deficiency → "
                    "cerebral folate deficiency → accumulation of calcium-folate complexes → "
                    "BRAIN CALCIFICATION (basal ganglia, white matter) on CT. "
                    "CRITICAL: Phe may normalize on low-Phe diet, giving false reassurance — but neurodegeneration "
                    "and calcification continue without neurotransmitter replacement + folinic acid."
                ),
                "pathognomonic": "DHPR enzyme activity ABSENT on dried blood spot; brain calcification on CT/MRI; biopterin HIGH (accumulates); folinic acid deficiency",
                "treatment":     "Phe restriction + BH4 + L-DOPA/carbidopa + 5-HTP + FOLINIC ACID (5-formylTHF — not folic acid)",
                "surveillance":  "Annual brain MRI/CT for calcification; CSF folate + neurotransmitters; Phe levels",
                "key_ddx":       "PTPS (neopterin HIGH; biopterin LOW; DHPR normal); PAH (DHPR normal); PCBD1 (primapterin HIGH; DHPR normal + benign)",
                "omim_link":     "https://omim.org/entry/261630",
            },
            "PCBD1": {
                "full_name":    "Pterin-4α-Carbinolamine Dehydratase 1",
                "protein_size": "104 aa / 12 kDa",
                "locus":        "10q22.1",
                "omim_gene":    "126090",
                "omim_disease": "264070",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "PCBD1 dehydrates pterin-4α-carbinolamine to quinonoid-BH2, which is then reduced by QDPR. "
                    "Without PCBD1, pterin-4α-carbinolamine converts non-enzymatically to 7-biopterin (primapterin) "
                    "instead of qBH2. PCBD1 deficiency → PRIMAPTERIN (7-biopterin) accumulates in urine = "
                    "PATHOGNOMONIC. The condition is TRANSIENT because PCBD1-independent pathways mature. "
                    "BENIGN: no neurological sequelae, Phe normalizes spontaneously by 6-12 months. "
                    "Critical NOT to treat with unnecessary BH4 or neurotransmitter therapy — primapterin is the clue."
                ),
                "pathognomonic": "7-biopterin (primapterin) HIGH in urine; DHPR activity NORMAL; Phe mildly elevated but transient; condition resolves spontaneously",
                "treatment":     "NO TREATMENT REQUIRED — observe only; Phe-restricted diet NOT needed",
                "surveillance":  "Phe monitoring until normalization (~6-12 months); confirm 7-biopterin",
                "key_ddx":       "PTPS (neopterin HIGH + biopterin LOW + DHPR normal; does NOT have primapterin); QDPR (DHPR absent); PAH (pterins normal)",
                "omim_link":     "https://omim.org/entry/264070",
            },
            "SPR": {
                "full_name":    "Sepiapterin Reductase",
                "protein_size": "261 aa / 29 kDa",
                "locus":        "2p14",
                "omim_gene":    "182125",
                "omim_disease": "612716",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "SPR catalyzes the last two steps of BH4 synthesis from 6-pyruvoyl-tetrahydropterin "
                    "(the final reductive steps). LOF → BH4 partially depleted + SEPIAPTERIN (the substrate) accumulates. "
                    "Because an alternative synthesis route (using carbonyl reductase + AKR1C3) can partially compensate, "
                    "Phe remains NORMAL — hence NBS completely misses SPR deficiency. "
                    "Sepiapterin does NOT appear in urine or blood in significant amounts — ONLY in CSF. "
                    "Patients develop dystonia with diurnal variation, hypotonia, intellectual disability, microcephaly — "
                    "often misdiagnosed as CP for years (average dx delay 5-10 years)."
                ),
                "pathognomonic": "CSF sepiapterin VERY HIGH = pathognomonic; CSF HVA LOW + 5-HIAA LOW; Phe NORMAL (key: NBS NORMAL)",
                "treatment":     "L-DOPA/carbidopa + 5-HTP → dramatic improvement; BH4 supplementation adjunctive",
                "surveillance":  "CSF neurotransmitters every 6 months; L-DOPA dose titration; neurological monitoring",
                "key_ddx":       "GCH1-AD Segawa (diurnal DRD but CSF sepiapterin absent); TH (5-HIAA NORMAL; neonatal oculogyric); CP (no diurnal variation; no L-DOPA response)",
                "omim_link":     "https://omim.org/entry/612716",
            },
            "DNAJC12": {
                "full_name":    "DnaJ Heat Shock Protein Family C Member 12",
                "protein_size": "198 aa / 22 kDa",
                "locus":        "10q21.3",
                "omim_gene":    "606060",
                "omim_disease": "617384",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "DNAJC12 (HSP40 chaperone) is required for folding and stability of three BH4-dependent enzymes: "
                    "PAH, TH, and TPH. LOF → all three enzymes misfolded/unstable → "
                    "hyperphenylalaninemia + neurotransmitter deficiency, similar to BH4 deficiencies. "
                    "CRITICAL DDx FROM BH4 DEFICIENCIES: pterins are NORMAL in DNAJC12 (no biosynthesis defect) "
                    "and BH4-loading test is NORMAL (BH4 levels adequate; enzyme is the problem, not cofactor). "
                    "Often misclassified as BH4-non-responsive PAH deficiency at NBS followup. "
                    "GENE SEQUENCING of DNAJC12 required when Phe elevated + pterins normal + BH4 loading normal."
                ),
                "pathognomonic": "Phe elevated + pterins NORMAL + BH4 loading NORMAL + neurodegeneration → DNAJC12 sequencing",
                "treatment":     "Low-Phe diet + BH4 (partially stabilizes remaining enzyme) + L-DOPA + 5-HTP",
                "surveillance":  "Phe + CSF neurotransmitters; neurological; brain MRI",
                "key_ddx":       "PAH deficiency (DNAJC12 = pterins normal; BH4 loading normal; neurological more severe); BH4 deficiencies (pterins abnormal)",
                "omim_link":     "https://omim.org/entry/617384",
            },
            "TH": {
                "full_name":    "Tyrosine Hydroxylase",
                "protein_size": "528 aa / 60 kDa",
                "locus":        "11p15.5",
                "omim_gene":    "191290",
                "omim_disease": "605407",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "TH catalyzes tyrosine → L-DOPA (rate-limiting step in catecholamine synthesis), requiring BH4 "
                    "as cofactor. LOF → dopamine AND norepinephrine both absent (downstream of TH). "
                    "Serotonin (TPH pathway) is INTACT — key DDx clue: CSF 5-HIAA NORMAL while HVA (dopamine) "
                    "and MHPG (norepinephrine) are very low. "
                    "TYPE A (severe): neonatal rigidity, hypokinesia, oculogyric crises, autonomic instability. "
                    "TYPE B (milder DRD-like): progressive onset, better L-DOPA response. "
                    "L-DOPA/carbidopa response is DRAMATIC and ESSENTIALLY CURATIVE — improvement within days "
                    "is both diagnostic and therapeutic. Do NOT delay L-DOPA trial for molecular confirmation."
                ),
                "pathognomonic": "Oculogyric crises + neonatal rigidity + CSF: HVA very low + MHPG very low + 5-HIAA NORMAL; Phe NORMAL",
                "treatment":     "L-DOPA/carbidopa (low dose, titrate slowly): TYPE A 1-5 mg/kg/day; TYPE B 0.5-3 mg/kg/day; avoid dopamine agonists",
                "surveillance":  "CSF HVA + MHPG + 5-HIAA annually; L-DOPA dose titration; oculomotor exam",
                "key_ddx":       "PTPS (HVA + 5-HIAA both low + Phe elevated); SPR (sepiapterin HIGH in CSF); GCH1-AR (all pterins absent + Phe elevated); CP (no L-DOPA response)",
                "omim_link":     "https://omim.org/entry/605407",
            },
        },
    }


if __name__ == "__main__":
    import json
    ov = generate_overview()
    bk = generate_breakdown()
    df = generate_definitions()
    print(f"Overview: {ov['total_patients']} patients, {len(ov['genes'])} genes")
    print(f"Breakdown groups: {list(bk['groups'].keys())}")
    print(f"Definitions: {list(df['definitions'].keys())}")
    print("OK — Hereditary-PKU-Biopterin-Deficiency-Atlas (seeds 4142-4149)")
