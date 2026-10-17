#!/usr/bin/env python3
"""Hereditary-Ehlers-Danlos-Syndrome-Atlas -- Complete 8-Gene Reference
Ehlers-Danlos syndromes: heritable connective tissue disorders affecting collagen/ECM
(COL5A1/cEDS-most-common-atrophic-scar-PATHOGNOMONIC,
 COL5A2/cEDS2-allelic-COL5A1,
 TNXB/TNX-EDS-hypermobile-like-haploinsufficiency,
 PLOD1/kEDS-PLOD1-lysylpyridinoline-ratio-PATHOGNOMONIC-vascular-rupture-risk,
 FKBP14/kEDS-FKBP14-hearing-loss-PATHOGNOMONIC,
 ADAMTS2/dEDS-extreme-skin-fragility-PATHOGNOMONIC,
 CHST14/mcEDS-adducted-thumbs-birth-PATHOGNOMONIC,
 SLC39A13/spEDS-protruding-eyes-fish-eye-cornea-PATHOGNOMONIC):

COL5A1  (Collagen type V alpha-1; 1838aa; 9q34.3; AD;
          Classical EDS (cEDS) — accounts for >90% of cEDS cases;
          Haploinsufficiency → reduced collagen V → abnormal fibril diameter;
          Atrophic papyraceous scars at minor trauma sites = PATHOGNOMONIC for cEDS;
          Significant skin hyperextensibility + easy bruising + wound dehiscence;
          NO vascular rupture risk (distinguish from COL3A1/vEDS);
          seed SEED_BASE+0).
COL5A2  (Collagen type V alpha-2; 1499aa; 2q31.1; AD;
          Classical EDS type 2; allelic with COL5A1;
          Rarer than COL5A1 (~5% of cEDS); structurally identical phenotype;
          Glycine substitutions in triple helix → dominant-negative;
          Same surveillance as COL5A1; no vascular rupture risk;
          seed SEED_BASE+1).
TNXB    (Tenascin-XB; 4243aa; 6p21.32; AR haploinsufficiency;
          Tenascin-X deficiency EDS: hypermobile-EDS-like (most common hEDS DDx);
          TNXB complete deletion → Clarkson disease (recurrent capillary leak + EDS);
          Haploinsufficiency → Ehlers-Danlos/joint hypermobility spectrum;
          Skin NOT as fragile as cEDS; minimal scarring;
          TNXB panel MANDATORY in patients labelled hEDS — 5-10% have detectable TNXB variant;
          seed SEED_BASE+2).
PLOD1   (Procollagen-lysine 2-oxoglutarate 5-dioxygenase 1; 727aa; 1p36.22; AR;
          Kyphoscoliotic EDS type 1 (kEDS-PLOD1): most common kEDS;
          Urinary total lysylpyridinoline / hydroxylysylpyridinoline ratio >3x = PATHOGNOMONIC;
          Neonatal hypotonia + congenital kyphoscoliosis + joint laxity = classic triad;
          Vascular rupture risk — spontaneous rupture like vEDS; aortic imaging MANDATORY;
          Lysyl hydroxylase 1 deficiency → abnormal collagen crosslinks;
          seed SEED_BASE+3).
FKBP14  (FKBP prolyl isomerase 14; 174aa; 7p14.3; AR;
          Kyphoscoliotic EDS type 2 (kEDS-FKBP14): allelic with FKBP14-related hearing loss;
          Progressive sensorineural hearing loss = PATHOGNOMONIC for kEDS-FKBP14 vs kEDS-PLOD1;
          Urine lysylpyridinoline ratio NORMAL — distinguishes from PLOD1;
          Muscle hypotonia + kyphoscoliosis + hearing loss = suspect FKBP14;
          No vascular rupture risk (unlike PLOD1);
          seed SEED_BASE+4).
ADAMTS2 (ADAMTS metallopeptidase with thrombospondin type 1 motif 2; 1211aa; 5q35.3; AR;
          Dermatosparaxis EDS (dEDS): procollagen N-proteinase deficiency;
          Extreme skin fragility with redundant, sagging skin at birth = PATHOGNOMONIC;
          Skin NOT hyperextensible (unlike cEDS) — key distinction;
          Premature rupture of membranes (PROM) in affected infants;
          Blue sclerae + characteristic facies + short stature;
          seed SEED_BASE+5).
CHST14  (Carbohydrate sulfotransferase 14; 376aa; 15q15.1; AR;
          Musculocontractural EDS (mcEDS-CHST14): glycosaminoglycan sulfation defect;
          Adducted thumbs + clubfoot at birth = PATHOGNOMONIC for mcEDS;
          Progressive multiple joint contractures + skin fragility;
          Dermatan sulfate biosynthesis defect → DS4-O-sulfation absent;
          Characteristic craniofacial features: hypertelorism, prominent eyes, downslanted fissures;
          seed SEED_BASE+6).
SLC39A13 (Solute carrier family 39 member 13 / ZIP13; 356aa; 11p11.2; AR;
           Spondylodysplastic EDS (spEDS-SLC39A13): zinc transporter deficiency;
           Protruding eyes + characteristic fish-eye cornea = PATHOGNOMONIC;
           Short stature + skeletal dysplasia + small hands/feet;
           Broad nasal bridge + downslanted palpebral fissures;
           Zinc homeostasis defect affects collagen/ECM maturation;
           seed SEED_BASE+7).
"""
import random
import math

SEED_BASE = 3830

ATLAS_GENES = [
    {
        "name": "COL5A1",
        "full_name": "Collagen type V alpha-1",
        "aa": 1838,
        "locus": "9q34.3",
        "inheritance": "AD",
        "mechanism": "LOF",
        "eds_type": "cEDS",
        "mw_kda": 184.0,
        "critical_rules": [
            "COL5A1-CEDS-MOST-COMMON-90PCT-CEDS",
            "COL5A1-ATROPHIC-SCAR-PATHOGNOMONIC",
            "COL5A1-NO-VASCULAR-RUPTURE-RISK-DISTINGUISH-COL3A1",
            "COL5A1-HAPLOINSUFFICIENCY-MECHANISM",
        ],
        "distinguishing_feature": "Atrophic papyraceous scars + skin hyperextensibility; NO vascular rupture",
        "first_line_treatment": "Wound care (avoid skin trauma; skin fragility); physiotherapy for joint laxity; avoid contact sports",
        "pathognomonic": "Atrophic 'papyraceous' scars at trauma sites = PATHOGNOMONIC for classical EDS/COL5A1",
        "onset_yrs_mean": 12,
        "penetrance_pct": 90,
        "skin_hyperextensibility_pct": 95,
        "atrophic_scar_pct": 90,
        "joint_hypermobility_pct": 80,
        "easy_bruising_pct": 85,
        "wound_dehiscence_pct": 55,
        "mitral_valve_prolapse_pct": 25,
        "vascular_events_pct": 2,
    },
    {
        "name": "COL5A2",
        "full_name": "Collagen type V alpha-2",
        "aa": 1499,
        "locus": "2q31.1",
        "inheritance": "AD",
        "mechanism": "LOF/DN",
        "eds_type": "cEDS",
        "mw_kda": 145.0,
        "critical_rules": [
            "COL5A2-CEDS-TYPE2-ALLELIC-COL5A1",
            "COL5A2-GLYCINE-SUBSTITUTION-DOMINANT-NEGATIVE",
            "COL5A2-SAME-SURVEILLANCE-AS-COL5A1",
        ],
        "distinguishing_feature": "Identical phenotype to COL5A1; rarer; glycine substitutions",
        "first_line_treatment": "Same as COL5A1: wound care, physiotherapy, joint protection",
        "pathognomonic": "Atrophic scars + skin hyperextensibility (indistinguishable from COL5A1 cEDS)",
        "onset_yrs_mean": 14,
        "penetrance_pct": 88,
        "skin_hyperextensibility_pct": 92,
        "atrophic_scar_pct": 85,
        "joint_hypermobility_pct": 78,
        "easy_bruising_pct": 80,
        "wound_dehiscence_pct": 50,
        "mitral_valve_prolapse_pct": 20,
        "vascular_events_pct": 2,
    },
    {
        "name": "TNXB",
        "full_name": "Tenascin-XB",
        "aa": 4243,
        "locus": "6p21.32",
        "inheritance": "AR",
        "mechanism": "LOF",
        "eds_type": "TNX-EDS",
        "mw_kda": 450.0,
        "critical_rules": [
            "TNXB-PANEL-MANDATORY-IN-HEDS-5-10PCT-DETECTABLE",
            "TNXB-MINIMAL-SCARRING-DISTINGUISH-CEDS",
            "TNXB-HAPLOINSUFFICIENCY-EHLERS-DANLOS-SPECTRUM",
            "TNXB-COMPLETE-DELETION-CLARKSON-CAPILLARY-LEAK",
        ],
        "distinguishing_feature": "hEDS-like phenotype; minimal scarring; TNXB panel MANDATORY in hEDS workup",
        "first_line_treatment": "Joint physiotherapy; pain management; orthotic support; same as hEDS",
        "pathognomonic": "Hypermobile EDS phenotype WITHOUT diagnostic scars; biallelic TNXB → complete Tenascin-X deficiency",
        "onset_yrs_mean": 20,
        "penetrance_pct": 70,
        "skin_hyperextensibility_pct": 50,
        "atrophic_scar_pct": 10,
        "joint_hypermobility_pct": 90,
        "easy_bruising_pct": 60,
        "wound_dehiscence_pct": 15,
        "mitral_valve_prolapse_pct": 15,
        "vascular_events_pct": 1,
    },
    {
        "name": "PLOD1",
        "full_name": "Procollagen-lysine,2-oxoglutarate 5-dioxygenase 1",
        "aa": 727,
        "locus": "1p36.22",
        "inheritance": "AR",
        "mechanism": "LOF",
        "eds_type": "kEDS-PLOD1",
        "mw_kda": 83.0,
        "critical_rules": [
            "PLOD1-URINE-LP-HLP-RATIO-GT3X-PATHOGNOMONIC",
            "PLOD1-VASCULAR-RUPTURE-RISK-AORTIC-IMAGING-MANDATORY",
            "PLOD1-NEONATAL-HYPOTONIA-KYPHOSCOLIOSIS-TRIAD",
            "PLOD1-DISTINGUISH-FROM-FKBP14-URINE-RATIO",
        ],
        "distinguishing_feature": "Urine lysylpyridinoline/hydroxylysylpyridinoline ratio >3x PATHOGNOMONIC; vascular rupture risk",
        "first_line_treatment": "Aortic imaging annual; spinal bracing/surgery for kyphoscoliosis; occupational therapy",
        "pathognomonic": "Neonatal hypotonia + kyphoscoliosis + LP/HLP urine ratio >3x = kEDS-PLOD1",
        "onset_yrs_mean": 0.1,
        "penetrance_pct": 95,
        "skin_hyperextensibility_pct": 70,
        "atrophic_scar_pct": 30,
        "joint_hypermobility_pct": 85,
        "easy_bruising_pct": 65,
        "wound_dehiscence_pct": 35,
        "vascular_rupture_pct": 15,
        "kyphoscoliosis_pct": 95,
        "neonatal_hypotonia_pct": 90,
    },
    {
        "name": "FKBP14",
        "full_name": "FKBP prolyl isomerase 14",
        "aa": 174,
        "locus": "7p14.3",
        "inheritance": "AR",
        "mechanism": "LOF",
        "eds_type": "kEDS-FKBP14",
        "mw_kda": 16.0,
        "critical_rules": [
            "FKBP14-HEARING-LOSS-PATHOGNOMONIC-VERSUS-PLOD1",
            "FKBP14-URINE-LP-RATIO-NORMAL-DISTINGUISHES-PLOD1",
            "FKBP14-NO-VASCULAR-RUPTURE-RISK",
            "FKBP14-KYPHOSCOLIOSIS-PLUS-HEARING-LOSS",
        ],
        "distinguishing_feature": "Progressive sensorineural hearing loss + kyphoscoliosis; urine LP ratio NORMAL",
        "first_line_treatment": "Hearing aids/cochlear implant; spinal bracing; physiotherapy",
        "pathognomonic": "Kyphoscoliotic EDS + progressive sensorineural hearing loss = FKBP14 (not PLOD1)",
        "onset_yrs_mean": 0.5,
        "penetrance_pct": 92,
        "skin_hyperextensibility_pct": 65,
        "atrophic_scar_pct": 20,
        "joint_hypermobility_pct": 80,
        "easy_bruising_pct": 55,
        "wound_dehiscence_pct": 25,
        "vascular_rupture_pct": 0,
        "kyphoscoliosis_pct": 90,
        "hearing_loss_pct": 80,
        "neonatal_hypotonia_pct": 85,
    },
    {
        "name": "ADAMTS2",
        "full_name": "ADAMTS metallopeptidase with thrombospondin type 1 motif 2",
        "aa": 1211,
        "locus": "5q35.3",
        "inheritance": "AR",
        "mechanism": "LOF",
        "eds_type": "dEDS",
        "mw_kda": 138.0,
        "critical_rules": [
            "ADAMTS2-EXTREME-SKIN-FRAGILITY-PATHOGNOMONIC",
            "ADAMTS2-SKIN-NOT-HYPEREXTENSIBLE-DISTINGUISH-CEDS",
            "ADAMTS2-SAGGING-REDUNDANT-SKIN-BIRTH-PATHOGNOMONIC",
            "ADAMTS2-PROM-PREMATURE-RUPTURE-MEMBRANES",
        ],
        "distinguishing_feature": "Extreme skin fragility + sagging redundant skin at birth; skin NOT hyperextensible",
        "first_line_treatment": "Meticulous wound care; protective padding; trauma prevention; genetic counseling",
        "pathognomonic": "Extreme skin fragility + sagging excess skin at birth + PROM = Dermatosparaxis EDS/ADAMTS2",
        "onset_yrs_mean": 0,
        "penetrance_pct": 98,
        "skin_hyperextensibility_pct": 10,
        "atrophic_scar_pct": 20,
        "joint_hypermobility_pct": 55,
        "easy_bruising_pct": 90,
        "wound_dehiscence_pct": 90,
        "vascular_rupture_pct": 5,
        "sagging_skin_pct": 98,
        "blue_sclerae_pct": 75,
    },
    {
        "name": "CHST14",
        "full_name": "Carbohydrate sulfotransferase 14",
        "aa": 376,
        "locus": "15q15.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "eds_type": "mcEDS-CHST14",
        "mw_kda": 43.0,
        "critical_rules": [
            "CHST14-ADDUCTED-THUMBS-CLUBFOOT-BIRTH-PATHOGNOMONIC",
            "CHST14-PROGRESSIVE-JOINT-CONTRACTURES",
            "CHST14-DERMATAN-SULFATE-4O-SULFATION-ABSENT",
            "CHST14-CHARACTERISTIC-FACIES-HYPERTELORISM",
        ],
        "distinguishing_feature": "Adducted thumbs + clubfoot at BIRTH = PATHOGNOMONIC; progressive joint contractures",
        "first_line_treatment": "Orthopedic management of contractures; casting/surgery for clubfoot; physiotherapy",
        "pathognomonic": "Adducted thumbs + clubfoot at birth + progressive joint contractures = mcEDS/CHST14",
        "onset_yrs_mean": 0,
        "penetrance_pct": 97,
        "skin_hyperextensibility_pct": 60,
        "atrophic_scar_pct": 40,
        "joint_hypermobility_pct": 40,
        "easy_bruising_pct": 70,
        "wound_dehiscence_pct": 55,
        "adducted_thumbs_pct": 90,
        "clubfoot_pct": 75,
        "kyphoscoliosis_pct": 60,
    },
    {
        "name": "SLC39A13",
        "full_name": "Solute carrier family 39 member 13 (ZIP13)",
        "aa": 356,
        "locus": "11p11.2",
        "inheritance": "AR",
        "mechanism": "LOF",
        "eds_type": "spEDS-SLC39A13",
        "mw_kda": 40.0,
        "critical_rules": [
            "SLC39A13-PROTRUDING-EYES-FISH-EYE-CORNEA-PATHOGNOMONIC",
            "SLC39A13-SHORT-STATURE-SKELETAL-DYSPLASIA",
            "SLC39A13-ZINC-TRANSPORTER-DEFICIENCY",
            "SLC39A13-SMALL-HANDS-FEET-CHARACTERISTIC",
        ],
        "distinguishing_feature": "Protruding eyes + fish-eye cornea = PATHOGNOMONIC; skeletal dysplasia + short stature",
        "first_line_treatment": "Ophthalmology monitoring; orthopedic management; zinc supplementation empirical",
        "pathognomonic": "Protuberant eyes + fish-eye cornea + short stature + skeletal dysplasia = spEDS/SLC39A13",
        "onset_yrs_mean": 0,
        "penetrance_pct": 96,
        "skin_hyperextensibility_pct": 55,
        "atrophic_scar_pct": 25,
        "joint_hypermobility_pct": 60,
        "easy_bruising_pct": 50,
        "wound_dehiscence_pct": 30,
        "protruding_eyes_pct": 90,
        "corneal_abnormality_pct": 85,
        "short_stature_pct": 95,
    },
]


def _rng(seed):
    return random.Random(seed)


def _make_cohort(gene_data, seed):
    rng = _rng(seed)
    n = 40
    g = gene_data
    base_age = max(0.1, g["onset_yrs_mean"])

    onset_ages = sorted([max(0.0, rng.gauss(base_age, max(0.2, base_age * 0.4))) for _ in range(n)])
    dx_lag = [max(1.0, rng.gauss(7.5, 3.5)) for _ in range(n)]
    dx_ages = [o + d for o, d in zip(onset_ages, dx_lag)]

    def pct_n(field, default=0):
        return round(g.get(field, default) / 100 * n)

    return {
        "gene": g["name"],
        "full_name": g["full_name"],
        "locus": g["locus"],
        "inheritance": g["inheritance"],
        "eds_type": g["eds_type"],
        "n": n,
        "seed": seed,
        "mean_onset_age": round(sum(onset_ages) / n, 2),
        "median_onset_age": round(sorted(onset_ages)[n // 2], 2),
        "mean_dx_age": round(sum(dx_ages) / n, 1),
        "mean_dx_lag_yrs": round(sum(dx_lag) / n, 1),
        "skin_hyperextensibility_n": pct_n("skin_hyperextensibility_pct"),
        "atrophic_scar_n": pct_n("atrophic_scar_pct"),
        "joint_hypermobility_n": pct_n("joint_hypermobility_pct"),
        "easy_bruising_n": pct_n("easy_bruising_pct"),
        "wound_dehiscence_n": pct_n("wound_dehiscence_pct"),
        "mitral_valve_prolapse_n": pct_n("mitral_valve_prolapse_pct"),
        "vascular_events_n": pct_n("vascular_events_pct"),
        "vascular_rupture_n": pct_n("vascular_rupture_pct"),
        "kyphoscoliosis_n": pct_n("kyphoscoliosis_pct"),
        "hearing_loss_n": pct_n("hearing_loss_pct"),
        "neonatal_hypotonia_n": pct_n("neonatal_hypotonia_pct"),
        "sagging_skin_n": pct_n("sagging_skin_pct"),
        "adducted_thumbs_n": pct_n("adducted_thumbs_pct"),
        "clubfoot_n": pct_n("clubfoot_pct"),
        "protruding_eyes_n": pct_n("protruding_eyes_pct"),
        "corneal_abnormality_n": pct_n("corneal_abnormality_pct"),
        "short_stature_n": pct_n("short_stature_pct"),
        "pathognomonic": g["pathognomonic"],
        "first_line_treatment": g["first_line_treatment"],
        "critical_rules": g["critical_rules"],
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    total_patients = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Ehlers-Danlos-Syndrome-Atlas",
        "subtitle": (
            "Complete 8-Gene EDS Reference: COL5A1/cEDS · COL5A2/cEDS2 · TNXB/TNX-EDS · "
            "PLOD1/kEDS-PLOD1 · FKBP14/kEDS-FKBP14 · ADAMTS2/dEDS · CHST14/mcEDS · SLC39A13/spEDS"
        ),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["name"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "cohort_size": 40,
        "cohorts": cohorts,
        "key_facts": {
            "most_common_EDS": "Hypermobile EDS (hEDS) — no gene found yet; most patients labelled hEDS",
            "most_common_cEDS": "COL5A1 — >90% of classical EDS; atrophic scar PATHOGNOMONIC",
            "vascular_risk_EDS": "PLOD1/kEDS — vascular rupture risk; aortic imaging MANDATORY (not cEDS)",
            "distinguish_PLOD1_FKBP14": "Urine LP/HLP ratio >3x = PLOD1; NORMAL ratio + hearing loss = FKBP14",
            "congenital_onset": "ADAMTS2 (dEDS) and CHST14 (mcEDS) present at birth with hallmark features",
            "hEDS_genetic_cause": "TNXB haploinsufficiency found in 5-10% of hEDS; test TNXB in hEDS workup",
            "treatable": "No EDS-specific disease-modifying therapy; management is symptomatic + surveillance",
            "vascular_EDS_note": "COL3A1/vEDS not in this atlas — covered in Hereditary-TAAD-Atlas",
        },
        "diagnostic_approach": {
            "cEDS_atrophic_scars": "COL5A1 (90%) then COL5A2; NO vascular imaging needed if confirmed cEDS",
            "kEDS_neonatal_hypotonia_kyphoscoliosis": "Urine LP/HLP ratio first: >3x → PLOD1; normal ratio → FKBP14 (esp if hearing loss)",
            "dEDS_skin_fragility_birth": "ADAMTS2; skin NOT hyperextensible; PROM in mother",
            "mcEDS_adducted_thumbs": "CHST14 first; DSE if CHST14 negative",
            "spEDS_protruding_eyes": "SLC39A13; short stature + skeletal dysplasia",
            "hEDS_phenotype_gene_negative": "TNXB panel; if haploinsufficiency found → TNX-EDS (not true hEDS)",
            "vascular_EDS_suspicion": "COL3A1 — see Hereditary-TAAD-Atlas; celiprolol; avoid surgery",
        },
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    gene_data = {}
    for c in cohorts:
        gene_data[c["gene"]] = {
            "n": c["n"],
            "eds_type": c["eds_type"],
            "inheritance": c["inheritance"],
            "locus": c["locus"],
            "mean_onset_age": c["mean_onset_age"],
            "mean_dx_age": c["mean_dx_age"],
            "mean_dx_lag_yrs": c["mean_dx_lag_yrs"],
            "skin_hyperextensibility_n": c["skin_hyperextensibility_n"],
            "atrophic_scar_n": c["atrophic_scar_n"],
            "joint_hypermobility_n": c["joint_hypermobility_n"],
            "easy_bruising_n": c["easy_bruising_n"],
            "wound_dehiscence_n": c["wound_dehiscence_n"],
            "mitral_valve_prolapse_n": c["mitral_valve_prolapse_n"],
            "vascular_events_n": c["vascular_events_n"],
            "vascular_rupture_n": c["vascular_rupture_n"],
            "kyphoscoliosis_n": c["kyphoscoliosis_n"],
            "hearing_loss_n": c["hearing_loss_n"],
            "neonatal_hypotonia_n": c["neonatal_hypotonia_n"],
            "sagging_skin_n": c["sagging_skin_n"],
            "adducted_thumbs_n": c["adducted_thumbs_n"],
            "clubfoot_n": c["clubfoot_n"],
            "protruding_eyes_n": c["protruding_eyes_n"],
            "corneal_abnormality_n": c["corneal_abnormality_n"],
            "short_stature_n": c["short_stature_n"],
            "pathognomonic": c["pathognomonic"],
            "first_line_treatment": c["first_line_treatment"],
        }
    return {
        "atlas": "Hereditary-Ehlers-Danlos-Syndrome-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_data": gene_data,
        "eds_classification": {
            "classical_EDS": {
                "genes": ["COL5A1", "COL5A2"],
                "hallmarks": "Skin hyperextensibility + atrophic papyraceous scars + fragile skin",
                "vascular_risk": "Low (no aortic imaging needed)",
                "first_test": "COL5A1 sequencing + MLPA",
            },
            "kyphoscoliotic_EDS": {
                "genes": ["PLOD1", "FKBP14"],
                "hallmarks": "Neonatal hypotonia + congenital kyphoscoliosis + joint laxity",
                "distinguish": "Urine LP/HLP ratio: >3x = PLOD1; normal = FKBP14; hearing loss → FKBP14",
                "vascular_risk": "HIGH for PLOD1 (like vEDS); LOW for FKBP14",
            },
            "dermatosparaxis_EDS": {
                "genes": ["ADAMTS2"],
                "hallmarks": "Extreme skin fragility + sagging redundant skin; skin NOT hyperextensible",
                "vascular_risk": "Low",
                "congenital": True,
            },
            "musculocontractural_EDS": {
                "genes": ["CHST14"],
                "hallmarks": "Adducted thumbs + clubfoot at birth + progressive contractures",
                "vascular_risk": "Low",
                "congenital": True,
            },
            "spondylodysplastic_EDS": {
                "genes": ["SLC39A13"],
                "hallmarks": "Protruding eyes + fish-eye cornea + short stature + skeletal dysplasia",
                "vascular_risk": "Low",
                "congenital": True,
            },
            "tenascin_EDS": {
                "genes": ["TNXB"],
                "hallmarks": "Hypermobile EDS-like; minimal scarring; 5-10% of hEDS patients",
                "vascular_risk": "Low",
                "test_in": "All patients labelled hEDS when genetic cause sought",
            },
        },
        "surveillance_matrix": {
            "COL5A1": ["Wound care protocol", "Avoid skin trauma/contact sports", "Echocardiogram baseline (MVP)", "Physiotherapy for joint laxity"],
            "COL5A2": ["Same as COL5A1", "Wound care", "Physiotherapy"],
            "TNXB": ["Joint physiotherapy", "Pain management", "Screen for POTS/autonomic", "Echocardiogram"],
            "PLOD1": ["Aortic MRI/CT ANNUALLY", "Spinal imaging for kyphoscoliosis", "Ophthalmology (corneal fragility)", "Occupational therapy"],
            "FKBP14": ["Audiometry annual", "Spinal imaging", "Physiotherapy", "No vascular imaging needed"],
            "ADAMTS2": ["Wound care meticulous", "Trauma prevention", "Ophthalmology (scleral fragility)", "Genetic counseling"],
            "CHST14": ["Orthopedic follow-up contractures", "Casting/surgery clubfoot", "Physiotherapy", "Ophthalmology"],
            "SLC39A13": ["Ophthalmology annual (cornea)", "Orthopedic monitoring", "Bone density", "Zinc supplementation trial"],
        },
    }


def generate_definitions():
    return {
        "atlas": "Hereditary-Ehlers-Danlos-Syndrome-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": {
            "Ehlers-Danlos Syndromes (EDS)": (
                "Clinically and genetically heterogeneous group of heritable connective tissue disorders "
                "characterized by joint hypermobility, skin hyperextensibility, and tissue fragility. "
                "2017 International Classification: 13 subtypes with distinct molecular bases. "
                "Hypermobile EDS (hEDS) is most common but has NO confirmed gene (diagnosis by clinical criteria). "
                "All other subtypes have defined causative genes."
            ),
            "Classical EDS — COL5A1/COL5A2": (
                "Most common molecularly defined EDS subtype; accounts for 90%+ of classical EDS diagnoses. "
                "COL5A1 haploinsufficiency → abnormal collagen I/V fibril diameter ratio. "
                "PATHOGNOMONIC: atrophic 'papyraceous' (cigarette-paper) scars at minor trauma sites. "
                "Skin hyperextensibility + easy bruising + wound dehiscence = clinical triad. "
                "NO vascular rupture risk — distinguish from vEDS/COL3A1. "
                "Wound care protocol; avoid contact sports; baseline echocardiogram for MVP."
            ),
            "TNXB/Tenascin-X EDS": (
                "Tenascin-XB: extracellular matrix glycoprotein controlling collagen fibril spacing. "
                "Biallelic LOF → complete TNX deficiency → hypermobile EDS + skin fragility. "
                "Haploinsufficiency (monoallelic) → milder joint hypermobility spectrum. "
                "TNXB panel MANDATORY in patients labelled hEDS when genetic workup requested — "
                "5-10% of hEDS patients have detectable TNXB variant. "
                "Minimal scarring differentiates from cEDS. "
                "Complete deletion → Clarkson disease (recurrent capillary leak syndrome)."
            ),
            "Kyphoscoliotic EDS — PLOD1 vs FKBP14 (Critical Distinction)": (
                "kEDS presents with neonatal hypotonia + congenital kyphoscoliosis + joint laxity. "
                "PLOD1 (kEDS type 1): lysyl hydroxylase 1 deficiency; "
                "URINE LP/HLP RATIO >3x = PATHOGNOMONIC; vascular rupture risk (annual aortic imaging MANDATORY). "
                "FKBP14 (kEDS type 2): FKBP14 ER chaperone; "
                "urine LP/HLP ratio NORMAL; progressive sensorineural HEARING LOSS = PATHOGNOMONIC for FKBP14. "
                "Critical rule: hearing loss + kEDS → FKBP14 (not PLOD1). "
                "Critical rule: elevated urine LP/HLP + kEDS → PLOD1 (vascular risk, aortic imaging)."
            ),
            "Dermatosparaxis EDS — ADAMTS2": (
                "ADAMTS2: procollagen N-proteinase; cleaves N-propeptide from procollagen I, II, III. "
                "Failure → retention of N-propeptide → abnormal fibril packing → extreme skin fragility. "
                "PATHOGNOMONIC: extreme skin fragility + sagging, redundant, doughy skin AT BIRTH. "
                "KEY DISTINCTION: skin NOT hyperextensible (unlike cEDS). "
                "Premature rupture of membranes (PROM) in affected neonates. "
                "Blue sclerae + characteristic facies. Extremely rare."
            ),
            "Musculocontractural EDS — CHST14": (
                "CHST14 encodes dermatan sulfate N-acetylgalactosamine 4-O-sulfotransferase. "
                "Glycosaminoglycan sulfation defect → abnormal dermatan sulfate → ECM failure. "
                "PATHOGNOMONIC at birth: bilateral adducted thumbs AND clubfoot — "
                "congenital contractures in a newborn = order CHST14 immediately. "
                "Progressive multiple joint contractures; craniofacial features; skin fragility. "
                "DSE (dermatan sulfate epimerase) deficiency causes similar mcEDS-DSE."
            ),
            "Spondylodysplastic EDS — SLC39A13": (
                "SLC39A13 encodes ZIP13 zinc transporter; zinc homeostasis critical for ECM maturation. "
                "spEDS: short stature + skeletal dysplasia + connective tissue signs. "
                "PATHOGNOMONIC: protruding eyes + fish-eye (flat/white) corneal appearance. "
                "Small hands and feet; broad nasal bridge; downslanted palpebral fissures. "
                "Zinc supplementation empirical. Ophthalmology monitoring mandatory."
            ),
            "EDS vs Marfan — Key Distinctions": (
                "Marfan (FBN1): aortic root dilatation + ectopia lentis + tall stature — "
                "echocardiogram + ophthalmology mandatory; prophylactic surgery threshold. "
                "Vascular EDS (COL3A1): spontaneous arterial/visceral rupture ANY diameter; surgery CI. "
                "Classical EDS (COL5A1): papyraceous scars; NO vascular rupture; no lens dislocation. "
                "Loeys-Dietz (TGFBR1/2): aggressive vasculopathy at smaller aortic diameters. "
                "Panel approach: include FBN1, COL5A1, COL3A1, TGFBR1/2 in connective tissue disorder workup."
            ),
        },
        "critical_clinical_rules": {
            rule: True
            for gene_data in ATLAS_GENES
            for rule in gene_data["critical_rules"]
        },
        "cascade_testing": {
            "COL5A1": ["COL5A1_sequencing_NGS", "COL5A1_MLPA_deletions", "COL5A2_if_negative"],
            "COL5A2": ["COL5A2_sequencing", "as_part_of_cEDS_panel"],
            "TNXB": ["TNXB_sequencing_CNV_analysis", "complement_TNXB_if_haploinsufficiency"],
            "PLOD1": ["urine_LP_HLP_ratio_first", "PLOD1_sequencing_if_ratio_elevated", "aortic_MRI_if_confirmed"],
            "FKBP14": ["FKBP14_sequencing", "audiometry_mandatory", "urine_LP_HLP_normal"],
            "ADAMTS2": ["ADAMTS2_sequencing", "skin_biopsy_electron_microscopy"],
            "CHST14": ["CHST14_sequencing", "DSE_if_negative", "urine_GAG_analysis"],
            "SLC39A13": ["SLC39A13_sequencing", "ophthalmology_corneal_exam", "skeletal_survey"],
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Ehlers-Danlos Syndrome Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(
            f"  {gene}/{d['eds_type']}: n={d['n']}, onset={d['mean_onset_age']}yr, "
            f"skin_ext={d['skin_hyperextensibility_n']}/40, "
            f"scar={d['atrophic_scar_n']}/40, "
            f"joint_hyper={d['joint_hypermobility_n']}/40, "
            f"kypho={d['kyphoscoliosis_n']}/40"
        )
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
