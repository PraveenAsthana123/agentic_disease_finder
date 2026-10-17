#!/usr/bin/env python3
"""Hereditary-Autoinflammatory-Syndrome-Atlas -- Complete 8-Gene Reference
Monogenic autoinflammatory syndromes: periodic fever and innate-immune dysregulation
(MEFV/FMF-most-common-colchicine-CURATIVE,
 NLRP3/CAPS-FCAS-MWS-NOMID-IL1-pathway-anakinra-canakinumab-HIGHLY-EFFECTIVE,
 TNFRSF1A/TRAPS-sustained-attacks-gt5d-migratory-rash-PATHOGNOMONIC,
 MVK/MKD-HIDS-IgD-elevated-80pct-lymphadenopathy,
 NOD2/Blau-syndrome-granulomatous-triad-arthritis-uveitis-rash-PATHOGNOMONIC,
 PSTPIP1/PAPA-pyoderma-gangrenosum-sterile-arthritis-acne-TRIAD-PATHOGNOMONIC,
 ADA2/DADA2-childhood-strokes-PATHOGNOMONIC-TNFi-PREVENTS-strokes,
 NLRP12/FCAS2-cold-triggered-milder-CAPS-anakinra-responsive):

MEFV    (Pyrin/marenostrin; 441aa; 16p13.3; AR (AD low-penetrance);
          Familial Mediterranean Fever (FMF) — most common monogenic periodic fever worldwide;
          Biallelic pathogenic variants → constitutive pyrin inflammasome activation → IL-1β/IL-18;
          Mediterranean/Middle Eastern ethnicity 1:200 carrier frequency;
          Short febrile attacks 12-72h + serositis (peritonitis/pleuritis/arthritis) recurrent;
          AA amyloidosis long-term complication if untreated;
          Colchicine 0.5-2 mg/day = CURATIVE — 90%+ attack-free; lifelong mandatory;
          seed SEED_BASE+0).
NLRP3   (Cryopyrin/NALP3; 1036aa; 1q44; AD gain-of-function (de novo);
          Cryopyrin-Associated Periodic Syndrome (CAPS): spectrum FCAS-MWS-NOMID;
          FCAS: cold-triggered urticaria-like rash + fever; mildest;
          MWS: progressive sensorineural hearing loss PATHOGNOMONIC + urticarial rash + amyloidosis risk;
          NOMID/CINCA: chronic inflammation + aseptic meningitis + bone lesions = MOST SEVERE;
          IL-1β is the sole mediator → IL-1 blockade HIGHLY EFFECTIVE (anakinra daily / canakinumab q8w);
          seed SEED_BASE+1).
TNFRSF1A (TNF receptor 1; 455aa; 12p13.31; AD (some low-penetrance);
           TNF Receptor-Associated Periodic Syndrome (TRAPS);
           Sustained febrile attacks >5 days (distinguish from FMF 12-72h) — KEY clinical rule;
           Migratory erythematous skin lesion overlying muscle = PATHOGNOMONIC for TRAPS;
           Periorbital oedema ('raccoon eyes') pathognomonic — muscle and fascia involvement;
           SAA/CRP dramatically elevated during attack;
           Etanercept partial benefit; IL-1i (anakinra/canakinumab) superior;
           seed SEED_BASE+2).
MVK     (Mevalonate kinase; 396aa; 12q24.11; AR;
          Mevalonate kinase deficiency (MKD) / Hyper-IgD syndrome (HIDS);
          Recurrent fever every 4-6 weeks triggered by stress/vaccination/minor illness;
          Serum IgD >100 IU/mL in 80% of HIDS patients — classic biomarker;
          Cervical lymphadenopathy + GI distress + oral aphthae during attacks;
          Mevalonate in urine elevated during/after attacks;
          IL-1i (anakinra) effective; canakinumab FDA 2016;
          seed SEED_BASE+3).
NOD2    (NOD2 receptor; 1040aa; 16q21; AD;
          Blau syndrome (early-onset sarcoidosis);
          Granulomatous triad: polyarthritis + uveitis + skin rash = PATHOGNOMONIC for Blau;
          Onset before age 4 — early-childhood arthritis + rash + eye disease;
          Camptodactyly (flexion contractures of fingers) characteristic;
          Non-caseating epithelioid granulomas in skin/synovium/uvea on biopsy;
          Methotrexate + TNF-i (adalimumab) + IL-1i for refractory;
          seed SEED_BASE+4).
PSTPIP1 (Proline-serine-threonine phosphatase interacting protein 1; 415aa; 15q24.3; AD;
          PAPA syndrome: Pyogenic Arthritis + Pyoderma gangrenosum + Acne — classic TRIAD PATHOGNOMONIC;
          Sterile destructive arthritis (NOT infection) mimics septic arthritis;
          Pyoderma gangrenosum (PG): painful ulcerating skin lesion with violaceous border = PATHOGNOMONIC;
          Cystic acne severe; sebaceous adenitis;
          PSTPIP1 binds pyrin → MEFV pathway activation;
          TNF-i (infliximab/adalimumab) effective; IL-1i for refractory;
          seed SEED_BASE+5).
ADA2    (Adenosine deaminase 2 / CECR1; 511aa; 22q11.1; AR;
          Deficiency of ADA2 (DADA2): systemic vasculopathy;
          Childhood ischaemic strokes in a young child WITHOUT cardiac cause = RED FLAG for DADA2;
          Polyarteritis nodosa (PAN)-like vasculitis — medium-vessel involvement;
          Livedo racemosa + nodular skin lesions;
          Recurrent fevers; hepatosplenomegaly; cytopenias;
          TNF-i (etanercept/adalimumab) PREVENTS STROKES — start before next stroke;
          HSCT curative for haematological manifestations;
          seed SEED_BASE+6).
NLRP12  (NLRP12/Monarch-1/PYPAF7; 1062aa; 19q13.42; AD;
          Familial Cold Autoinflammatory Syndrome type 2 (FCAS2);
          Cold-triggered episodic fever + urticaria-like rash — milder than NLRP3-CAPS;
          Limb pain + arthralgia during cold-triggered attacks;
          Hearing loss less consistent than MWS (NLRP3);
          Anakinra responsive; canakinumab second-line;
          Distinguished from NLRP3-FCAS1 by genetics; overlap phenotype;
          seed SEED_BASE+7).
"""
import random
import math

SEED_BASE = 3838

ATLAS_GENES = [
    {
        "name": "MEFV",
        "full_name": "Pyrin/marenostrin",
        "aa": 441,
        "locus": "16p13.3",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "FMF",
        "mw_kda": 53.0,
        "critical_rules": [
            "MEFV-COLCHICINE-CURATIVE-LIFELONG-MANDATORY",
            "MEFV-ATTACK-DURATION-12-72H-DISTINGUISH-TRAPS",
            "MEFV-AA-AMYLOIDOSIS-COMPLICATION-UNTREATED",
            "MEFV-MOST-COMMON-PERIODIC-FEVER-WORLDWIDE",
        ],
        "distinguishing_feature": "Short attacks 12-72h + serositis; colchicine CURATIVE; Mediterranean ancestry",
        "first_line_treatment": "Colchicine 0.5-2 mg/day lifelong; canakinumab/anakinra for colchicine-refractory",
        "pathognomonic": "Recurrent short febrile attacks 12-72h + peritonitis/pleuritis/arthritis + Mediterranean ancestry",
        "onset_yrs_mean": 10,
        "penetrance_pct": 60,
        "attack_frequency_per_yr": 6,
        "attack_duration_h_mean": 30,
        "peritonitis_pct": 85,
        "pleuritis_pct": 40,
        "arthritis_pct": 50,
        "colchicine_response_pct": 90,
        "amyloidosis_risk_pct": 10,
        "il1_blockade_indication_pct": 10,
        "skin_rash_pct": 15,
        "lymphadenopathy_pct": 20,
    },
    {
        "name": "NLRP3",
        "full_name": "Cryopyrin/NALP3",
        "aa": 1036,
        "locus": "1q44",
        "inheritance": "AD",
        "mechanism": "GOF",
        "syndrome": "CAPS",
        "mw_kda": 118.0,
        "critical_rules": [
            "NLRP3-MWS-SNHL-PATHOGNOMONIC-PROGRESSIVE-HEARING-LOSS",
            "NLRP3-IL1-SOLE-MEDIATOR-ANAKINRA-CANAKINUMAB-HIGHLY-EFFECTIVE",
            "NLRP3-NOMID-ASEPTIC-MENINGITIS-BONE-LESIONS-MOST-SEVERE",
            "NLRP3-COLD-URTICARIA-FCAS-DISTINGUISH-COLD-ALLERGY",
        ],
        "distinguishing_feature": "CAPS spectrum: FCAS (cold urticaria) → MWS (hearing loss) → NOMID (meningitis); IL-1i curative",
        "first_line_treatment": "IL-1 blockade MANDATORY: anakinra 1-2 mg/kg/day or canakinumab 150 mg q8w; lifelong",
        "pathognomonic": "MWS: urticarial rash + progressive SNHL + amyloidosis risk; NOMID: chronic meningitis + bone lesions",
        "onset_yrs_mean": 3,
        "penetrance_pct": 95,
        "attack_frequency_per_yr": 12,
        "attack_duration_h_mean": 24,
        "urticarial_rash_pct": 95,
        "sensorineural_hearing_loss_pct": 70,
        "aseptic_meningitis_pct": 30,
        "bone_lesions_pct": 25,
        "amyloidosis_risk_pct": 20,
        "il1_blockade_indication_pct": 100,
        "skin_rash_pct": 95,
        "lymphadenopathy_pct": 30,
        "colchicine_response_pct": 5,
    },
    {
        "name": "TNFRSF1A",
        "full_name": "TNF receptor superfamily member 1A",
        "aa": 455,
        "locus": "12p13.31",
        "inheritance": "AD",
        "mechanism": "LOF",
        "syndrome": "TRAPS",
        "mw_kda": 51.0,
        "critical_rules": [
            "TNFRSF1A-ATTACKS-GT5-DAYS-KEY-DISTINCTION-FROM-FMF",
            "TNFRSF1A-MIGRATORY-RASH-OVERLYING-MUSCLE-PATHOGNOMONIC",
            "TNFRSF1A-PERIORBITAL-OEDEMA-RACCOON-EYES-PATHOGNOMONIC",
            "TNFRSF1A-IL1I-SUPERIOR-ETANERCEPT",
        ],
        "distinguishing_feature": "Long attacks >5 days; migratory muscle-overlying rash PATHOGNOMONIC; periorbital oedema",
        "first_line_treatment": "IL-1 blockade first-line (anakinra/canakinumab); etanercept partial; NSAIDs/steroids short-term",
        "pathognomonic": "Sustained febrile attacks >5 days + migratory erythematous rash over muscle + periorbital oedema",
        "onset_yrs_mean": 15,
        "penetrance_pct": 50,
        "attack_frequency_per_yr": 4,
        "attack_duration_h_mean": 168,
        "peritonitis_pct": 50,
        "pleuritis_pct": 60,
        "arthritis_pct": 40,
        "migratory_rash_pct": 75,
        "periorbital_oedema_pct": 60,
        "amyloidosis_risk_pct": 15,
        "il1_blockade_indication_pct": 85,
        "skin_rash_pct": 80,
        "lymphadenopathy_pct": 35,
        "colchicine_response_pct": 20,
    },
    {
        "name": "MVK",
        "full_name": "Mevalonate kinase",
        "aa": 396,
        "locus": "12q24.11",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "MKD/HIDS",
        "mw_kda": 41.0,
        "critical_rules": [
            "MVK-IGD-GT100-IUML-80PCT-HIDS-CLASSIC-BIOMARKER",
            "MVK-URINE-MEVALONATE-ELEVATED-DURING-ATTACKS",
            "MVK-CANAKINUMAB-FDA2016-APPROVED",
            "MVK-VACCINATION-TRIGGER-ATTACKS",
        ],
        "distinguishing_feature": "IgD >100 IU/mL (80%); fever q4-6 weeks; triggered by vaccination/stress; cervical lymphadenopathy",
        "first_line_treatment": "Anakinra for attacks; canakinumab 150 mg q4w (FDA 2016); colchicine limited benefit",
        "pathognomonic": "Periodic fever every 4-6 weeks + serum IgD >100 + cervical lymphadenopathy + GI symptoms",
        "onset_yrs_mean": 1,
        "penetrance_pct": 85,
        "attack_frequency_per_yr": 10,
        "attack_duration_h_mean": 96,
        "peritonitis_pct": 70,
        "pleuritis_pct": 30,
        "arthritis_pct": 70,
        "igd_elevated_pct": 80,
        "lymphadenopathy_pct": 95,
        "oral_aphthae_pct": 60,
        "amyloidosis_risk_pct": 3,
        "il1_blockade_indication_pct": 90,
        "skin_rash_pct": 50,
        "colchicine_response_pct": 15,
    },
    {
        "name": "NOD2",
        "full_name": "Nucleotide-binding oligomerization domain 2",
        "aa": 1040,
        "locus": "16q21",
        "inheritance": "AD",
        "mechanism": "GOF",
        "syndrome": "Blau syndrome",
        "mw_kda": 114.0,
        "critical_rules": [
            "NOD2-GRANULOMATOUS-TRIAD-ARTHRITIS-UVEITIS-RASH-PATHOGNOMONIC",
            "NOD2-ONSET-BEFORE-AGE-4-EARLY-CHILDHOOD",
            "NOD2-CAMPTODACTYLY-FINGER-FLEXION-CONTRACTURES",
            "NOD2-NON-CASEATING-GRANULOMAS-BIOPSY",
        ],
        "distinguishing_feature": "Granulomatous triad: arthritis + uveitis + skin rash PATHOGNOMONIC; onset <4 years; camptodactyly",
        "first_line_treatment": "Methotrexate + adalimumab/infliximab; IL-1i for refractory; ophthalmology for uveitis",
        "pathognomonic": "Early childhood (<4yr) polyarthritis + uveitis + papular/scaly skin rash + camptodactyly = Blau syndrome",
        "onset_yrs_mean": 2.5,
        "penetrance_pct": 90,
        "attack_frequency_per_yr": 52,
        "attack_duration_h_mean": 0,
        "arthritis_pct": 95,
        "uveitis_pct": 85,
        "skin_granuloma_pct": 80,
        "camptodactyly_pct": 70,
        "amyloidosis_risk_pct": 5,
        "il1_blockade_indication_pct": 40,
        "skin_rash_pct": 80,
        "lymphadenopathy_pct": 20,
        "colchicine_response_pct": 5,
    },
    {
        "name": "PSTPIP1",
        "full_name": "Proline-serine-threonine phosphatase-interacting protein 1",
        "aa": 415,
        "locus": "15q24.3",
        "inheritance": "AD",
        "mechanism": "GOF",
        "syndrome": "PAPA",
        "mw_kda": 47.0,
        "critical_rules": [
            "PSTPIP1-PAPA-TRIAD-PATHOGNOMONIC-PYODERMA-ARTHRITIS-ACNE",
            "PSTPIP1-STERILE-ARTHRITIS-NOT-INFECTION-DDx-SEPTIC",
            "PSTPIP1-PYODERMA-GANGRENOSUM-VIOLACEOUS-BORDER",
            "PSTPIP1-TNFi-INFLIXIMAB-FIRST-LINE",
        ],
        "distinguishing_feature": "PAPA triad: Pyogenic Arthritis + Pyoderma gangrenosum + Acne PATHOGNOMONIC; sterile joint",
        "first_line_treatment": "Infliximab or adalimumab first-line; IL-1i (anakinra) second-line; wound care for PG",
        "pathognomonic": "Sterile destructive arthritis (negative cultures) + pyoderma gangrenosum + severe acne = PAPA/PSTPIP1",
        "onset_yrs_mean": 8,
        "penetrance_pct": 75,
        "attack_frequency_per_yr": 3,
        "attack_duration_h_mean": 336,
        "arthritis_pct": 95,
        "pyoderma_gangrenosum_pct": 80,
        "acne_pct": 75,
        "amyloidosis_risk_pct": 5,
        "il1_blockade_indication_pct": 60,
        "skin_rash_pct": 85,
        "lymphadenopathy_pct": 10,
        "colchicine_response_pct": 10,
    },
    {
        "name": "ADA2",
        "full_name": "Adenosine deaminase 2 (CECR1)",
        "aa": 511,
        "locus": "22q11.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "DADA2",
        "mw_kda": 59.0,
        "critical_rules": [
            "ADA2-CHILDHOOD-STROKES-WITHOUT-CARDIAC-CAUSE-RED-FLAG",
            "ADA2-TNFi-PREVENTS-STROKES-START-IMMEDIATELY",
            "ADA2-PAN-LIKE-VASCULITIS-MEDIUM-VESSEL",
            "ADA2-HSCT-CURATIVE-HAEMATOLOGICAL",
        ],
        "distinguishing_feature": "Childhood strokes + PAN-like vasculitis + livedo racemosa; TNF-i PREVENTS strokes",
        "first_line_treatment": "Etanercept/adalimumab IMMEDIATELY on diagnosis — prevents strokes; HSCT for haematological disease",
        "pathognomonic": "Childhood ischaemic stroke without cardiac cause + livedo racemosa + PAN-like vasculitis = DADA2",
        "onset_yrs_mean": 5,
        "penetrance_pct": 80,
        "attack_frequency_per_yr": 4,
        "attack_duration_h_mean": 120,
        "stroke_pct": 60,
        "vasculitis_pct": 75,
        "livedo_racemosa_pct": 65,
        "hepatosplenomegaly_pct": 50,
        "cytopenias_pct": 40,
        "amyloidosis_risk_pct": 5,
        "il1_blockade_indication_pct": 20,
        "tnfi_indication_pct": 90,
        "skin_rash_pct": 65,
        "lymphadenopathy_pct": 40,
        "colchicine_response_pct": 5,
    },
    {
        "name": "NLRP12",
        "full_name": "NLRP12/Monarch-1/PYPAF7",
        "aa": 1062,
        "locus": "19q13.42",
        "inheritance": "AD",
        "mechanism": "LOF",
        "syndrome": "FCAS2",
        "mw_kda": 120.0,
        "critical_rules": [
            "NLRP12-COLD-TRIGGERED-FCAS2-MILDER-THAN-NLRP3-FCAS1",
            "NLRP12-ANAKINRA-RESPONSIVE",
            "NLRP12-HEARING-LOSS-LESS-CONSISTENT-THAN-MWS",
            "NLRP12-DISTINGUISH-FROM-NLRP3-BY-GENETICS",
        ],
        "distinguishing_feature": "Cold-triggered urticaria + fever; milder than NLRP3-CAPS; hearing loss inconstant; anakinra responsive",
        "first_line_treatment": "Anakinra first-line; canakinumab second-line; cold avoidance; NSAIDs for mild attacks",
        "pathognomonic": "Cold-triggered episodic urticarial rash + fever + arthralgia; negative NLRP3 genetics → test NLRP12",
        "onset_yrs_mean": 5,
        "penetrance_pct": 70,
        "attack_frequency_per_yr": 8,
        "attack_duration_h_mean": 12,
        "urticarial_rash_pct": 90,
        "sensorineural_hearing_loss_pct": 30,
        "arthritis_pct": 60,
        "amyloidosis_risk_pct": 5,
        "il1_blockade_indication_pct": 75,
        "skin_rash_pct": 90,
        "lymphadenopathy_pct": 25,
        "colchicine_response_pct": 15,
    },
]


def _rng(seed):
    return random.Random(seed)


def _make_cohort(gene_data, seed):
    rng = _rng(seed)
    n = 40
    g = gene_data
    base_age = max(0.5, g["onset_yrs_mean"])

    onset_ages = sorted([max(0.0, rng.gauss(base_age, max(0.3, base_age * 0.4))) for _ in range(n)])
    dx_lag = [max(0.5, rng.gauss(5.0, 3.0)) for _ in range(n)]
    dx_ages = [o + d for o, d in zip(onset_ages, dx_lag)]

    def pct_n(field, default=0):
        return round(g.get(field, default) / 100 * n)

    return {
        "gene": g["name"],
        "full_name": g["full_name"],
        "locus": g["locus"],
        "inheritance": g["inheritance"],
        "syndrome": g["syndrome"],
        "n": n,
        "seed": seed,
        "mean_onset_age": round(sum(onset_ages) / n, 2),
        "median_onset_age": round(sorted(onset_ages)[n // 2], 2),
        "mean_dx_age": round(sum(dx_ages) / n, 1),
        "mean_dx_lag_yrs": round(sum(dx_lag) / n, 1),
        "attack_frequency_per_yr": g.get("attack_frequency_per_yr", 0),
        "attack_duration_h_mean": g.get("attack_duration_h_mean", 0),
        "peritonitis_n": pct_n("peritonitis_pct"),
        "pleuritis_n": pct_n("pleuritis_pct"),
        "arthritis_n": pct_n("arthritis_pct"),
        "skin_rash_n": pct_n("skin_rash_pct"),
        "lymphadenopathy_n": pct_n("lymphadenopathy_pct"),
        "amyloidosis_risk_n": pct_n("amyloidosis_risk_pct"),
        "il1_blockade_n": pct_n("il1_blockade_indication_pct"),
        "colchicine_response_n": pct_n("colchicine_response_pct"),
        "sensorineural_hearing_loss_n": pct_n("sensorineural_hearing_loss_pct"),
        "uveitis_n": pct_n("uveitis_pct"),
        "stroke_n": pct_n("stroke_pct"),
        "vasculitis_n": pct_n("vasculitis_pct"),
        "pyoderma_gangrenosum_n": pct_n("pyoderma_gangrenosum_pct"),
        "oral_aphthae_n": pct_n("oral_aphthae_pct"),
        "igd_elevated_n": pct_n("igd_elevated_pct"),
        "camptodactyly_n": pct_n("camptodactyly_pct"),
        "migratory_rash_n": pct_n("migratory_rash_pct"),
        "periorbital_oedema_n": pct_n("periorbital_oedema_pct"),
        "aseptic_meningitis_n": pct_n("aseptic_meningitis_pct"),
        "livedo_racemosa_n": pct_n("livedo_racemosa_pct"),
        "pathognomonic": g["pathognomonic"],
        "first_line_treatment": g["first_line_treatment"],
        "critical_rules": g["critical_rules"],
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    total_patients = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Autoinflammatory-Syndrome-Atlas",
        "subtitle": (
            "Complete 8-Gene Periodic Fever Reference: MEFV/FMF · NLRP3/CAPS · TNFRSF1A/TRAPS · "
            "MVK/MKD-HIDS · NOD2/Blau · PSTPIP1/PAPA · ADA2/DADA2 · NLRP12/FCAS2"
        ),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["name"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "cohort_size": 40,
        "cohorts": cohorts,
        "key_facts": {
            "most_common": "MEFV/FMF — most common monogenic periodic fever worldwide; 1:200 carriers in Mediterranean populations",
            "il1_treatable": "NLRP3/CAPS and MVK/HIDS: IL-1 blockade (anakinra/canakinumab) highly effective",
            "colchicine_curative": "MEFV/FMF: colchicine 0.5-2 mg/day = curative in 90%; prevent AA amyloidosis",
            "traps_attack_length": "TNFRSF1A/TRAPS: attacks >5 days — key distinction from FMF (12-72h)",
            "dada2_strokes": "ADA2/DADA2: start TNF-i IMMEDIATELY — prevents childhood ischaemic strokes",
            "blau_granulomas": "NOD2/Blau: granulomatous arthritis+uveitis+rash triad PATHOGNOMONIC; onset <4 years",
            "papa_sterile": "PSTPIP1/PAPA: sterile arthritis (negative cultures) + pyoderma gangrenosum + acne = PAPA",
            "fcas2_cold": "NLRP12/FCAS2: cold-triggered periodic fever; milder than NLRP3-CAPS; anakinra responsive",
            "amyloidosis_prevention": "AA amyloidosis complication of uncontrolled FMF and TRAPS; treat aggressively to prevent",
        },
        "diagnostic_approach": {
            "short_attacks_12_72h": "MEFV/FMF first; check ethnicity; start colchicine empirically in Mediterranean patients",
            "long_attacks_gt5d": "TNFRSF1A/TRAPS; migratory rash over muscle + periorbital oedema; IL-1i",
            "cold_triggered_urticaria": "NLRP3/CAPS first (FCAS1); if NLRP3 negative → test NLRP12 (FCAS2)",
            "igd_elevated_lymphadenopathy": "MVK/HIDS; urine mevalonate; vaccination trigger history",
            "childhood_fever_uveitis_arthritis": "NOD2/Blau; skin biopsy for granulomas; early-onset <4yr",
            "sterile_arthritis_pyoderma": "PSTPIP1/PAPA; negative joint cultures; pyoderma + acne triad",
            "childhood_strokes_vasculitis": "ADA2/DADA2; livedo racemosa; start TNF-i immediately",
            "hearing_loss_urticaria": "NLRP3/CAPS MWS type; IL-1 blockade mandatory; prevent progressive SNHL",
        },
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    gene_data = {}
    for c in cohorts:
        gene_data[c["gene"]] = {
            "n": c["n"],
            "syndrome": c["syndrome"],
            "inheritance": c["inheritance"],
            "locus": c["locus"],
            "mean_onset_age": c["mean_onset_age"],
            "mean_dx_age": c["mean_dx_age"],
            "mean_dx_lag_yrs": c["mean_dx_lag_yrs"],
            "attack_frequency_per_yr": c["attack_frequency_per_yr"],
            "attack_duration_h_mean": c["attack_duration_h_mean"],
            "peritonitis_n": c["peritonitis_n"],
            "pleuritis_n": c["pleuritis_n"],
            "arthritis_n": c["arthritis_n"],
            "skin_rash_n": c["skin_rash_n"],
            "lymphadenopathy_n": c["lymphadenopathy_n"],
            "amyloidosis_risk_n": c["amyloidosis_risk_n"],
            "il1_blockade_n": c["il1_blockade_n"],
            "colchicine_response_n": c["colchicine_response_n"],
            "sensorineural_hearing_loss_n": c["sensorineural_hearing_loss_n"],
            "uveitis_n": c["uveitis_n"],
            "stroke_n": c["stroke_n"],
            "vasculitis_n": c["vasculitis_n"],
            "pyoderma_gangrenosum_n": c["pyoderma_gangrenosum_n"],
            "oral_aphthae_n": c["oral_aphthae_n"],
            "igd_elevated_n": c["igd_elevated_n"],
            "camptodactyly_n": c["camptodactyly_n"],
            "migratory_rash_n": c["migratory_rash_n"],
            "periorbital_oedema_n": c["periorbital_oedema_n"],
            "aseptic_meningitis_n": c["aseptic_meningitis_n"],
            "livedo_racemosa_n": c["livedo_racemosa_n"],
            "pathognomonic": c["pathognomonic"],
            "first_line_treatment": c["first_line_treatment"],
        }
    return {
        "atlas": "Hereditary-Autoinflammatory-Syndrome-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_data": gene_data,
        "syndrome_classification": {
            "IL1_pathway_high_response": {
                "genes": ["MEFV", "NLRP3", "MVK", "NLRP12"],
                "mechanism": "Inflammasome activation → IL-1β overproduction",
                "treatment": "IL-1 blockade (anakinra/canakinumab); colchicine for FMF",
                "biomarker": "Serum IL-18 elevated; SAA/CRP during attacks",
            },
            "TNF_pathway": {
                "genes": ["TNFRSF1A"],
                "mechanism": "Misfolded TNF receptor → ER stress + innate activation",
                "treatment": "IL-1 blockade preferred; etanercept partial; avoid infliximab",
                "biomarker": "Elevated soluble TNFRSF1A during attacks; CRP markedly elevated",
            },
            "granulomatous_disease": {
                "genes": ["NOD2"],
                "mechanism": "GOF NOD2 → NF-κB activation → granuloma formation",
                "treatment": "MTX + TNF-i; IL-1i second-line",
                "hallmark": "Non-caseating granulomas on biopsy = diagnostic",
            },
            "vasculopathy": {
                "genes": ["ADA2", "PSTPIP1"],
                "mechanism": "ADA2: ADA2 deficiency → macrophage activation; PSTPIP1: MEFV pathway GOF",
                "treatment": "ADA2: TNF-i immediately; PSTPIP1: TNF-i or IL-1i",
                "urgent_note": "DADA2: start TNF-i BEFORE next stroke — do NOT wait for next event",
            },
        },
        "treatment_matrix": {
            "MEFV": {
                "first_line": "Colchicine 0.5-2 mg/day lifelong",
                "second_line": "Canakinumab 150 mg q4w or anakinra 100 mg/day",
                "avoid": "NSAIDs long-term (renal risk in amyloidosis)",
                "monitor": "Urine protein annually (amyloidosis screening)",
            },
            "NLRP3": {
                "first_line": "Anakinra 1-2 mg/kg/day SC or canakinumab 150 mg q8w",
                "second_line": "Rilonacept; increase canakinumab dose",
                "avoid": "Steroids long-term (adverse effects without disease modification)",
                "monitor": "Audiometry 6-monthly (SNHL progression); ophthalmology",
            },
            "TNFRSF1A": {
                "first_line": "Anakinra or canakinumab; etanercept partial benefit",
                "second_line": "Etanercept 25 mg twice weekly; tocilizumab reported",
                "avoid": "Infliximab (paradoxical worsening in TRAPS)",
                "monitor": "SAA levels; renal function (amyloidosis)",
            },
            "MVK": {
                "first_line": "Canakinumab 150 mg q4w (FDA approved 2016); anakinra for attacks",
                "second_line": "Etanercept; simvastatin empirical (mevalonate pathway)",
                "avoid": "Vaccination during attack (triggers recurrence)",
                "monitor": "IgD levels; growth in children",
            },
            "NOD2": {
                "first_line": "Adalimumab + methotrexate; ophthalmology for uveitis",
                "second_line": "Anakinra or tocilizumab for refractory",
                "avoid": "Corticosteroid monotherapy long-term",
                "monitor": "Annual ophthalmology (uveitis can cause blindness); joint imaging",
            },
            "PSTPIP1": {
                "first_line": "Infliximab or adalimumab; wound care for PG lesions",
                "second_line": "Anakinra; cyclosporin for PG",
                "avoid": "Debridement of PG lesions (pathergy worsens)",
                "monitor": "Joint imaging; skin surveillance",
            },
            "ADA2": {
                "first_line": "Etanercept or adalimumab IMMEDIATELY (stroke prevention)",
                "second_line": "HSCT for haematological disease; fresh frozen plasma bridge",
                "avoid": "Anticoagulation alone (inadequate for stroke prevention in DADA2)",
                "monitor": "MRI brain 6-monthly; CBC for cytopenias; ADA2 enzyme levels",
            },
            "NLRP12": {
                "first_line": "Anakinra 100 mg/day; cold avoidance behavioural",
                "second_line": "Canakinumab 150 mg q8w",
                "avoid": "Trigger exposure (cold environments)",
                "monitor": "Audiometry (hearing loss less common than MWS but possible)",
            },
        },
        "surveillance_matrix": {
            "MEFV": ["Urine protein annual (amyloidosis)", "Colchicine compliance", "Renal function", "Liver enzymes (colchicine)"],
            "NLRP3": ["Audiometry q6 months", "Ophthalmology annual", "SAA/CRP levels", "MRI brain (NOMID meningitis)"],
            "TNFRSF1A": ["SAA/CRP during remission", "Renal function (amyloidosis)", "Ophthalmology (uveitis)"],
            "MVK": ["IgD levels", "Growth monitoring children", "Urine mevalonate during attacks"],
            "NOD2": ["Annual ophthalmology (uveitis)", "Joint imaging", "Skin biopsy if new lesions"],
            "PSTPIP1": ["Skin surveillance (PG, acne)", "Joint imaging", "Wound care protocol"],
            "ADA2": ["MRI brain q6m (strokes)", "CBC q3m (cytopenias)", "ADA2 enzyme level", "Echocardiogram"],
            "NLRP12": ["Audiometry annual", "Cold diary (trigger tracking)", "SAA/CRP during attacks"],
        },
    }


def generate_definitions():
    return {
        "atlas": "Hereditary-Autoinflammatory-Syndrome-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": {
            "Monogenic Autoinflammatory Syndromes": (
                "Rare inherited disorders of innate immunity characterised by recurrent episodes of systemic "
                "inflammation without infection, autoimmunity, or malignancy. "
                "Distinct from autoimmune disease: driven by dysregulated innate immune sensors (inflammasomes, "
                "NF-κB, interferon pathways), not adaptive T/B cell responses. "
                "Hallmark: periodic fever + serositis/rash/arthritis in a genetically susceptible host. "
                "Unified by excess IL-1β production in most subtypes — rationale for IL-1 blockade."
            ),
            "MEFV/FMF — Colchicine CURATIVE": (
                "Familial Mediterranean Fever: most common monogenic periodic fever, prevalent in Mediterranean, "
                "Middle Eastern, Jewish, Armenian, Turkish populations. "
                "MEFV encodes pyrin — GOF pyrin inflammasome → excess IL-1β/IL-18. "
                "Classic attack: 12-72h fever + peritonitis (94%) ± pleuritis (40%) ± arthritis (50%). "
                "CRITICAL RULE: Colchicine 0.5-2 mg/day = CURATIVE in 90% — lifelong, start at diagnosis. "
                "Uncollected FMF → AA amyloidosis (kidneys) — prevented by colchicine. "
                "Attack duration KEY DDx: FMF 12-72h vs TRAPS >5 days."
            ),
            "NLRP3/CAPS — IL-1 Blockade MANDATORY": (
                "Cryopyrin-associated periodic syndrome: AD GOF mutations in NLRP3 cryopyrin. "
                "Spectrum from mild to severe — FCAS1 (cold urticaria, mildest) → MWS → NOMID/CINCA (most severe). "
                "IL-1β is THE sole mediator: anakinra or canakinumab is highly effective, often near-complete remission. "
                "MWS PATHOGNOMONIC: progressive sensorineural hearing loss — audiometry mandatory q6 months. "
                "NOMID: chronic aseptic meningitis + bone overgrowth — MRI brain + skeletal survey. "
                "CRITICAL RULE: never withhold IL-1 blockade — untreated MWS/NOMID causes blindness, deafness, amyloidosis."
            ),
            "TNFRSF1A/TRAPS — Attack Duration >5 Days Key": (
                "TNF receptor-associated periodic syndrome: mutations in TNFRSF1A → misfolded TNFR1 → ER stress + "
                "intracellular signalling. PATHOGNOMONIC SIGNS: (1) attacks >5 days (FMF: 12-72h); "
                "(2) migratory erythematous skin lesion overlying migrating muscle; "
                "(3) periorbital oedema 'raccoon eyes' — myofascial involvement. "
                "IL-1 blockade superior to etanercept. Avoid infliximab (paradoxical worsening). "
                "Amyloidosis risk ~15% — SAA monitoring mandatory."
            ),
            "MVK/MKD-HIDS — IgD Biomarker and Vaccination Trigger": (
                "Mevalonate kinase deficiency: AR deficiency in MVK enzyme of mevalonate pathway. "
                "Classic HIDS: IgD >100 IU/mL in 80% — diagnostic biomarker. "
                "Attacks every 4-6 weeks, often triggered by vaccination, minor illness, or stress. "
                "Urine mevalonate elevated during/after attacks — confirmatory. "
                "Features: cervical lymphadenopathy + GI distress + oral aphthae. "
                "Canakinumab FDA approved 2016 (CLUSTER trial). Avoid vaccination during fever attacks."
            ),
            "NOD2/Blau Syndrome — Granulomatous Triad": (
                "Blau syndrome: AD GOF NOD2 mutations → constitutive NF-κB activation → granulomatous inflammation. "
                "PATHOGNOMONIC TRIAD: (1) polyarthritis with boggy synovitis; (2) anterior uveitis; "
                "(3) papular erythematous or ichthyosiform skin rash — all three by age 4 years. "
                "Camptodactyly: characteristic flexion contractures of the fingers. "
                "Biopsy: non-caseating epithelioid granulomas in skin/synovium/uvea. "
                "CRITICAL: ophthalmology mandatory — uveitis can cause blindness if untreated. "
                "Sporadic early-onset sarcoidosis = same condition (same NOD2 mutations)."
            ),
            "PSTPIP1/PAPA — Sterile Arthritis + PG + Acne TRIAD": (
                "PAPA syndrome: AD GOF PSTPIP1 binds pyrin → MEFV-like inflammasome activation. "
                "PATHOGNOMONIC TRIAD: Pyogenic Arthritis (sterile) + Pyoderma gangrenosum + Acne. "
                "Sterile arthritis: destructive joint inflammation — cultures NEGATIVE; mimics septic arthritis. "
                "Pyoderma gangrenosum (PG): painful rapidly expanding ulcer with violaceous undermined border — "
                "DO NOT debride (pathergy: debridement worsens PG). "
                "TNF-i (infliximab/adalimumab) first-line; IL-1i second-line."
            ),
            "ADA2/DADA2 — Strokes in Children: Start TNF-i Immediately": (
                "Deficiency of ADA2: AR LOF mutations in CECR1/ADA2 → ADA2 enzyme deficiency → "
                "macrophage/monocyte activation → medium-vessel vasculitis. "
                "CLINICAL RED FLAG: ischaemic stroke in a child WITHOUT cardiac source = test ADA2 enzyme. "
                "Livedo racemosa: net-like skin discolouration — characteristic. "
                "PAN-like vasculitis; hepatosplenomegaly; cytopenias (bone marrow involvement). "
                "CRITICAL RULE: start etanercept or adalimumab IMMEDIATELY on diagnosis — "
                "TNF-i prevents subsequent strokes; do NOT wait for next neurological event. "
                "HSCT curative for haematological manifestations (not neurological)."
            ),
            "NLRP12/FCAS2 — Cold-Triggered, Milder CAPS Alternative": (
                "Familial cold autoinflammatory syndrome type 2: AD LOF NLRP12 → loss of NF-κB suppression. "
                "Cold-triggered episodic urticarial rash + fever + arthralgia — milder than NLRP3/CAPS. "
                "NLRP12 is an NF-κB suppressor — LOF paradoxically activates inflammasome/NF-κB. "
                "Hearing loss possible but less consistent than MWS (NLRP3). "
                "KEY DDx from NLRP3-FCAS1: overlapping phenotype — resolve by genetics. "
                "Anakinra responsive; canakinumab second-line; cold avoidance counselling."
            ),
            "Autoinflammatory vs Autoimmune — Key Distinction": (
                "Autoinflammatory: dysregulated innate immunity; no autoantibodies; no autoreactive T/B cells. "
                "Autoimmune: adaptive immunity; autoantibodies (ANA, anti-dsDNA, RF); autoreactive lymphocytes. "
                "Autoinflammatory biomarkers: SAA, CRP, ferritin elevated during attacks; ANA/ANCA NEGATIVE. "
                "Overlap syndromes: DADA2 (both features); NOD2/Blau (granulomatous = overlap). "
                "Treatment differences: autoinflammatory → IL-1i or TNF-i; autoimmune → MTX/hydroxychloroquine."
            ),
        },
        "critical_clinical_rules": {
            rule: True
            for gene_data in ATLAS_GENES
            for rule in gene_data["critical_rules"]
        },
        "cascade_testing": {
            "MEFV": ["MEFV_sequencing_full_gene", "ethnicity_guided_targeted_panel", "colchicine_trial_if_clinical_FMF"],
            "NLRP3": ["NLRP3_sequencing_full_gene_including_somatic", "consider_somatic_mosaicism_if_negative"],
            "TNFRSF1A": ["TNFRSF1A_sequencing", "functional_assay_for_variants_of_uncertain_significance"],
            "MVK": ["MVK_sequencing", "urine_mevalonate_during_fever", "serum_IgD", "MVK_enzyme_activity_blood"],
            "NOD2": ["NOD2_sequencing", "skin_biopsy_granulomas", "ophthalmology_slit_lamp"],
            "PSTPIP1": ["PSTPIP1_sequencing", "joint_fluid_cultures_negative_confirms_sterile", "skin_biopsy_PG"],
            "ADA2": ["ADA2_enzyme_activity_plasma_first", "CECR1_sequencing_if_enzyme_low", "MRI_brain_vasculitis_workup"],
            "NLRP12": ["NLRP12_sequencing", "if_NLRP3_negative_with_CAPS_phenotype"],
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Autoinflammatory Syndrome Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(
            f"  {gene}/{d['syndrome']}: n={d['n']}, onset={d['mean_onset_age']}yr, "
            f"attack_freq={d['attack_frequency_per_yr']}/yr, "
            f"arthritis={d['arthritis_n']}/40, "
            f"rash={d['skin_rash_n']}/40, "
            f"il1={d['il1_blockade_n']}/40"
        )
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
