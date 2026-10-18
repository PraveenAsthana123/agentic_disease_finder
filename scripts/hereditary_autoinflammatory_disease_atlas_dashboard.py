#!/usr/bin/env python3
"""Hereditary-Autoinflammatory-Disease-Atlas -- Complete 8-Gene Reference
Monogenic Autoinflammatory Diseases (MAIDs) — Periodic Fever Syndromes & Inflammasomopathies
(MEFV/FMF-Most-Common-AR-AD-Colchicine-First-Line-Amyloidosis-PATHOGNOMONIC,
 NLRP3/CAPS-FCAS-MWS-NOMID-Spectrum-AD-GOF-Canakinumab-FDA2016-Urticarial-Rash-Cold-Trigger,
 TNFRSF1A/TRAPS-AD-GOF-Longer-Episodes-Periorbital-Edema-Migratory-Rash-Etanercept-NOT-Effective,
 MVK/MKD-HIDS-AR-Mevalonate-Kinase-Deficiency-IgD-Elevated-Statins-NOT-HELPFUL-Canakinumab,
 PSTPIP1/PAPA-Syndrome-AD-GOF-Pyoderma-Gangrenosum-Acne-Pyogenic-Arthritis-Triad-PATHOGNOMONIC,
 NOD2/Blau-Syndrome-EOS-AD-GOF-Uveitis-Arthritis-Dermatitis-GRANULOMATOUS-TRIAD-PATHOGNOMONIC,
 IL1RN/DIRA-AR-Neonatal-Onset-Sterile-Osteomyelitis-Pustulosis-Periostitis-Anakinra-CURATIVE,
 LPIN2/Majeed-Syndrome-AR-CRMO-Congenital-Dyserythropoietic-Anemia-Neutrophilic-Dermatosis):

MEFV   (Pyrin/Marenostrin; 781aa; 16p13.3; AR (classical) / AD (attenuated);
         Pyrin (MEFV) is a member of the TRIM superfamily expressed in myeloid cells;
         Pyrin forms an inflammasome via SPRY/B-box/coiled-coil; guards against Rho-GTPase inactivation;
         MEFV LOF/gain → dysregulated IL-1β/IL-18 release via caspase-1 activation;
         Most common monogenic AID: Mediterranean populations (Turks/Armenians/Arabs/Ashkenazi Jews);
         Classic FMF: recurrent febrile attacks 12-72h; PERITONITIS/PLEURITIS/PERICARDITIS/SYNOVITIS;
         AMYLOIDOSIS (AA-type, renal) is the devastating complication in untreated FMF — PATHOGNOMONIC;
         Colchicine: FIRST-LINE, highly effective (prevents attacks AND amyloidosis — taken lifelong);
         IL-1 inhibitors (anakinra/canakinumab) for colchicine-resistant FMF;
         Common mutations: M694V (most severe, highest amyloid risk), E148Q (mild, AR+AD), M680I, V726A;
         Homozygous M694V = highest amyloidosis risk; heterozygous M694V = intermediate;
         seed SEED_BASE+0).
NLRP3  (Cryopyrin/CIAS1; 1036aa; 1q44; AD-GOF;
         NLRP3 is the inflammasome scaffold — forms the NLRP3-ASC-caspase-1 complex;
         AD-GOF mutations → constitutively active NLRP3 inflammasome → excessive IL-1β/IL-18;
         CAPS spectrum (Cryopyrin-Associated Periodic Syndrome):
           FCAS (Familial Cold Autoinflammatory Syndrome): mildest; cold-triggered urticarial rash + fever;
           MWS (Muckle-Wells Syndrome): intermediate; progressive SNHL PATHOGNOMONIC; amyloidosis risk;
           NOMID/CINCA (Neonatal-Onset Multisystem Inflammatory Disease): most severe; CNS (papilledema/meningitis), arthropathy, facial dysmorphism;
         URTICARIAL RASH triggered by cold = PATHOGNOMONIC for CAPS/FCAS-MWS;
         SNHL progressive in MWS = must check audiogram at diagnosis and annually;
         Canakinumab (anti-IL-1β, FDA-approved 2016 for CAPS) — highly effective, prevents SNHL progression;
         seed SEED_BASE+1).
TNFRSF1A (TNF receptor superfamily member 1A; 234aa; 12p13.31; AD-GOF;
         TNFRSF1A encodes the 55-kDa TNF receptor (TNFR1) — primary mediator of TNF-α inflammatory signalling;
         AD-GOF mutations → misfolding of TNFR1 → endoplasmic reticulum (ER) retention → NF-κB activation;
         TRAPS (TNF Receptor-Associated Periodic Syndrome): longest episodes of all periodic fevers (days-weeks);
         Characteristic features: MIGRATORY ERYTHEMATOUS RASH, PERIORBITAL EDEMA PATHOGNOMONIC, myalgia;
         Episodes longer than FMF (3 days → weeks vs 12-72h); triggered by stress, infection, menses;
         CRITICAL: etanercept (soluble TNFR2) is LESS EFFECTIVE than IL-1 blockers for TRAPS;
         Anakinra/canakinumab preferred over anti-TNF biologics in TRAPS;
         Amyloidosis risk (AA-type) in untreated/inadequately treated TRAPS;
         Low-penetrance variants (R92Q, P46L) — common in population; unclear pathogenicity;
         seed SEED_BASE+2).
MVK    (Mevalonate kinase; 396aa; 12q24.11; AR;
         MVK catalyses the second step of the mevalonate pathway (mevalonic acid → phosphomevalonic acid);
         Partial MVK deficiency → mevalonate accumulates → exact mechanism unclear (geranylgeranylation defect);
         Complete MVK absence → mevalonic aciduria (most severe, acidosis crises);
         MKD (Mevalonate Kinase Deficiency) = HIDS (Hyperimmunoglobulinemia D and Periodic Fever Syndrome);
         IgD elevated >100 IU/mL in majority of patients — original diagnostic test (now: MVK sequencing);
         KEY NEGATIVE FINDING: IgD can be normal in ~30% — do NOT exclude by normal IgD alone;
         Fever episodes 4-7 days; lymphadenopathy (CERVICAL ADENOPATHY PATHOGNOMONIC in children);
         Statins NOT helpful (inhibit HMG-CoA reductase, upstream of MVK) — may worsen by reducing mevalonate;
         Canakinumab FDA-approved for MKD (2016); anakinra effective;
         URINE MEVALONIC ACID during fever episode — elevated = diagnostic confirmation;
         seed SEED_BASE+3).
PSTPIP1 (Proline-serine-threonine phosphatase-interacting protein 1 / CD2BP1; 415aa; 15q24.3; AD-GOF;
         PSTPIP1 is an adaptor protein linking F-actin regulation to pyrin inflammasome activation;
         AD-GOF mutations → hyperphosphorylated PSTPIP1 → enhanced pyrin binding → IL-1β overproduction;
         PAPA syndrome triad: PYOGENIC STERILE ARTHRITIS + PYODERMA GANGRENOSUM + ACNE CONGLOBATA — PATHOGNOMONIC;
         Pathergy positive (trauma-induced skin lesions) in >90% of PAPA;
         PAMI (PSTPIP1-Associated Myeloid-Related Proteinemia Inflammatory) syndrome: rare severe variant;
         Treatment: IL-1 blockers (anakinra first-line); TNF blockers used for articular disease;
         Skin lesions (pyoderma gangrenosum) respond to systemic corticosteroids + IL-1 inhibition;
         BIOLOGIC SWITCH: if anakinra fails for arthritis, switch to IL-1β (canakinumab); pyoderma often needs steroids;
         Differential: PASH (PG+Acne+Suppurative Hidradenitis) — PSTPIP1 panel in severe acne+PG combo;
         seed SEED_BASE+4).
NOD2   (Nucleotide-binding oligomerization domain-containing protein 2 / CARD15; 1040aa; 16q12.1; AD-GOF;
         NOD2 is a cytosolic innate immune receptor sensing muramyl dipeptide (MDP) from bacterial peptidoglycan;
         AD-GOF mutations (Blau/EOS mutations) → constitutive NF-κB activation → granuloma formation;
         Note: LOF NOD2 mutations → Crohn's disease susceptibility (opposite effect — different mutation class);
         BLAU SYNDROME: sporadic = Early-Onset Sarcoidosis (EOS); familial = Blau syndrome (same mutations);
         GRANULOMATOUS TRIAD PATHOGNOMONIC: (1) UVEITIS (panuveitis; band keratopathy; cataracts), (2) ARTHRITIS (boggy polyarticular synovitis, camptodactyly), (3) SKIN (ichthyosiform dermatitis, macular rash);
         Onset typically in first decade; no fever as prominent as other MAIDs;
         Blindness from uveitis is the most feared complication — ophthalmology follow-up MANDATORY;
         Treatment: corticosteroids + methotrexate; TNF inhibitors (infliximab/adalimumab) for refractory;
         DISTINGUISH from sarcoidosis: in Blau/EOS no hilar adenopathy, ACE often normal, onset <5yr;
         seed SEED_BASE+5).
IL1RN  (Interleukin-1 receptor antagonist; 177aa; 2q14.1; AR;
         IL-1RN encodes IL-1Ra — the endogenous competitive antagonist of IL-1α and IL-1β;
         IL1RN LOF → unopposed IL-1 signalling → severe systemic inflammation from NEONATAL ONSET;
         DIRA (Deficiency of IL-1 Receptor Antagonist): neonatal onset sterile multifocal osteomyelitis;
         TRIAD PATHOGNOMONIC: (1) STERILE MULTIFOCAL OSTEOMYELITIS (periostitis), (2) PUSTULOSIS (generalized), (3) PERIOSTITIS (widened periosteal reaction on X-ray);
         Neonatal onset: day 1-7 of life; severe distress; failure to thrive; can be fatal without treatment;
         ANAKINRA IS CURATIVE: anakinra replaces missing IL-1Ra; dramatic response within 24-48h; lifelong therapy;
         Canadian founder (Puerto Rico also reported): deletion in 2q14.1 enriched in specific populations;
         DISTINGUISH from septic arthritis: sterile (culture negative) + pustulosis + periostitis → DIRA;
         X-ray: periosteal widening + metaphyseal irregularity (NOT joint space narrowing of septic arthritis);
         seed SEED_BASE+6).
LPIN2  (Phosphatidic acid phosphatase Lipin-2; 890aa; 18p11.31; AR;
         LPIN2 encodes Lipin-2, a phospholipid phosphatase that dephosphorylates phosphatidic acid (PA) → DAG;
         Lipin-2 expressed in bone marrow, liver, skin; regulates lipid metabolism and NF-κB signalling via PA;
         LPIN2 LOF → MAJEED SYNDROME: chronic recurrent multifocal osteomyelitis (CRMO) + hematologic + skin;
         MAJEED TRIAD: (1) CHRONIC RECURRENT MULTIFOCAL OSTEOMYELITIS (CRMO) — sterile bone inflammation; (2) CONGENITAL DYSERYTHROPOIETIC ANEMIA (CDA) — dysplastic erythropoiesis, transfusion-dependent; (3) NEUTROPHILIC DERMATOSIS (Sweet syndrome-like) — PATHOGNOMONIC TRIAD;
         Unlike other MAIDs: CRMO starts in infancy; CDA = bone marrow failure; both distinguish from other CRMO;
         Treatment: NSAIDs/corticosteroids for bone/skin; G-CSF for severe neutropenia; IL-1 blockers (anakinra);
         Amyloidosis risk if untreated chronically;
         First described in Jordanian Arab families (AR founder mutations in consanguineous pedigrees);
         Bisphosphonates for CRMO bone disease if IL-1 inhibition insufficient;
         seed SEED_BASE+7).
"""

import random
import math

ATLAS_NAME   = "Hereditary-Autoinflammatory-Disease-Atlas"
SEED_BASE    = 3966   # seeds 3966-3973 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8

GENES = ["MEFV", "NLRP3", "TNFRSF1A", "MVK", "PSTPIP1", "NOD2", "IL1RN", "LPIN2"]

GENE_DATA = {
    "MEFV": {
        "full_name":     "Pyrin/Marenostrin",
        "aa":            781,
        "locus":         "16p13.3",
        "kda":           "86",
        "inheritance":   "AR (classical) / AD-LOF (attenuated)",
        "syndrome":      "Familial Mediterranean Fever (FMF)",
        "pathway":       "Pyrin inflammasome / IL-1β",
        "onset_range":   (3, 120),    # months
        "episode_dur_h": (12, 72),
        "trigger":       "stress/infection/menses/fatty-meals",
        "colchicine_resp_pct": 0.90,
        "il1_blocker_pct":     0.15,
        "amyloid_risk_pct":    0.25,
        "snhl_pct":            0.04,
        "rash_pct":            0.30,
        "uveitis_pct":         0.05,
        "crmo_pct":            0.00,
        "anaemia_pct":         0.10,
        "pathognomonic":       "AA AMYLOIDOSIS (renal) in untreated FMF + peritonitis/pleuritis attacks",
        "key_drug":            "Colchicine (lifelong; prevents attacks AND amyloidosis)",
    },
    "NLRP3": {
        "full_name":     "Cryopyrin/CIAS1",
        "aa":            1036,
        "locus":         "1q44",
        "kda":           "118",
        "inheritance":   "AD-GOF",
        "syndrome":      "CAPS (FCAS / MWS / NOMID-CINCA)",
        "pathway":       "NLRP3 inflammasome / IL-1β constitutive",
        "onset_range":   (0, 6),
        "episode_dur_h": (12, 48),
        "trigger":       "cold-exposure/stress",
        "colchicine_resp_pct": 0.10,
        "il1_blocker_pct":     0.95,
        "amyloid_risk_pct":    0.20,
        "snhl_pct":            0.55,
        "rash_pct":            0.98,
        "uveitis_pct":         0.25,
        "crmo_pct":            0.00,
        "anaemia_pct":         0.15,
        "pathognomonic":       "COLD-TRIGGERED URTICARIAL RASH + progressive SNHL (MWS) + neonatal onset (NOMID)",
        "key_drug":            "Canakinumab (anti-IL-1β; FDA 2016 for CAPS); anakinra alternative",
    },
    "TNFRSF1A": {
        "full_name":     "Tumor necrosis factor receptor 1 (TNFR1/p55)",
        "aa":            234,
        "locus":         "12p13.31",
        "kda":           "55",
        "inheritance":   "AD-GOF (ER retention of misfolded receptor)",
        "syndrome":      "TRAPS (TNF Receptor-Associated Periodic Syndrome)",
        "pathway":       "ER retention / NF-κB / IL-1β",
        "onset_range":   (3, 180),
        "episode_dur_h": (72, 336),   # 3d to 2 weeks
        "trigger":       "stress/infection/vaccination/menses",
        "colchicine_resp_pct": 0.20,
        "il1_blocker_pct":     0.80,
        "amyloid_risk_pct":    0.20,
        "snhl_pct":            0.05,
        "rash_pct":            0.88,
        "uveitis_pct":         0.15,
        "crmo_pct":            0.00,
        "anaemia_pct":         0.18,
        "pathognomonic":       "PERIORBITAL EDEMA + migratory erythematous rash + LONG episodes (days-weeks)",
        "key_drug":            "Anakinra/canakinumab (IL-1 blockers preferred over etanercept in TRAPS)",
    },
    "MVK": {
        "full_name":     "Mevalonate kinase",
        "aa":            396,
        "locus":         "12q24.11",
        "kda":           "43",
        "inheritance":   "AR",
        "syndrome":      "MKD / HIDS (Hyperimmunoglobulinemia D and Periodic Fever Syndrome)",
        "pathway":       "Mevalonate pathway / geranylgeranylation defect / IL-1β",
        "onset_range":   (0, 12),
        "episode_dur_h": (96, 168),   # 4-7 days
        "trigger":       "vaccination/infection/physical-trauma/stress",
        "colchicine_resp_pct": 0.25,
        "il1_blocker_pct":     0.78,
        "amyloid_risk_pct":    0.10,
        "snhl_pct":            0.03,
        "rash_pct":            0.60,
        "uveitis_pct":         0.10,
        "crmo_pct":            0.00,
        "anaemia_pct":         0.12,
        "pathognomonic":       "CERVICAL LYMPHADENOPATHY + elevated IgD + urine mevalonic acid elevated during fever",
        "key_drug":            "Canakinumab FDA-approved for MKD; anakinra; NOT statins (worsen pathway block)",
    },
    "PSTPIP1": {
        "full_name":     "CD2BP1 / PSTPIP1",
        "aa":            415,
        "locus":         "15q24.3",
        "kda":           "47",
        "inheritance":   "AD-GOF",
        "syndrome":      "PAPA Syndrome (Pyogenic Arthritis + Pyoderma Gangrenosum + Acne conglobata)",
        "pathway":       "Pyrin-PSTPIP1 interaction / IL-1β hyperactivation",
        "onset_range":   (6, 36),
        "episode_dur_h": (120, 720),  # days to weeks
        "trigger":       "trauma/pathergy",
        "colchicine_resp_pct": 0.15,
        "il1_blocker_pct":     0.82,
        "amyloid_risk_pct":    0.08,
        "snhl_pct":            0.02,
        "rash_pct":            0.95,
        "uveitis_pct":         0.05,
        "crmo_pct":            0.00,
        "anaemia_pct":         0.08,
        "pathognomonic":       "PAPA TRIAD: sterile PYOGENIC ARTHRITIS + PYODERMA GANGRENOSUM + ACNE CONGLOBATA",
        "key_drug":            "Anakinra (IL-1 blocker) first-line; steroids for pyoderma gangrenosum skin lesions",
    },
    "NOD2": {
        "full_name":     "CARD15 (NOD2)",
        "aa":            1040,
        "locus":         "16q12.1",
        "kda":           "114",
        "inheritance":   "AD-GOF (Blau/EOS); AD-LOF → Crohn's susceptibility (different!)",
        "syndrome":      "Blau Syndrome / Early-Onset Sarcoidosis (EOS)",
        "pathway":       "NOD2 / NF-κB / granuloma formation",
        "onset_range":   (0, 60),
        "episode_dur_h": (0, 0),      # not episodic — continuous granulomatous disease
        "trigger":       "non-episodic (continuous granulomatous disease)",
        "colchicine_resp_pct": 0.10,
        "il1_blocker_pct":     0.30,
        "amyloid_risk_pct":    0.05,
        "snhl_pct":            0.10,
        "rash_pct":            0.80,
        "uveitis_pct":         0.95,
        "crmo_pct":            0.00,
        "anaemia_pct":         0.15,
        "pathognomonic":       "GRANULOMATOUS TRIAD: UVEITIS + ARTHRITIS + SKIN (ichthyosiform dermatitis) — no fever!",
        "key_drug":            "Corticosteroids + methotrexate; TNF inhibitors (infliximab) for refractory uveitis",
    },
    "IL1RN": {
        "full_name":     "Interleukin-1 receptor antagonist (IL-1Ra)",
        "aa":            177,
        "locus":         "2q14.1",
        "kda":           "20",
        "inheritance":   "AR",
        "syndrome":      "DIRA (Deficiency of IL-1 Receptor Antagonist)",
        "pathway":       "Unopposed IL-1α/IL-1β signalling (IL-1Ra absent)",
        "onset_range":   (0, 1),      # neonatal: day 1-7 of life
        "episode_dur_h": (0, 0),      # continuous (not episodic)
        "trigger":       "neonatal onset — continuous from birth",
        "colchicine_resp_pct": 0.05,
        "il1_blocker_pct":     0.99,  # anakinra essentially curative
        "amyloid_risk_pct":    0.12,
        "snhl_pct":            0.03,
        "rash_pct":            0.95,
        "uveitis_pct":         0.05,
        "crmo_pct":            0.85,
        "anaemia_pct":         0.20,
        "pathognomonic":       "NEONATAL sterile multifocal OSTEOMYELITIS + PUSTULOSIS + PERIOSTITIS — dramatic anakinra response",
        "key_drug":            "Anakinra (replaces missing IL-1Ra; CURATIVE — lifelong daily injection mandatory)",
    },
    "LPIN2": {
        "full_name":     "Lipin-2 phosphatidic acid phosphatase",
        "aa":            890,
        "locus":         "18p11.31",
        "kda":           "101",
        "inheritance":   "AR",
        "syndrome":      "Majeed Syndrome (CRMO + CDA + Neutrophilic Dermatosis)",
        "pathway":       "Lipid metabolism dysregulation / NF-κB via PA accumulation",
        "onset_range":   (0, 6),      # infancy
        "episode_dur_h": (0, 0),      # chronic — not episodic
        "trigger":       "chronic disease — CRMO continuous bone inflammation",
        "colchicine_resp_pct": 0.20,
        "il1_blocker_pct":     0.70,
        "amyloid_risk_pct":    0.15,
        "snhl_pct":            0.05,
        "rash_pct":            0.85,
        "uveitis_pct":         0.10,
        "crmo_pct":            0.98,
        "anaemia_pct":         0.98,   # CDA = congenital dyserythropoietic anemia
        "pathognomonic":       "MAJEED TRIAD: CRMO + CONGENITAL DYSERYTHROPOIETIC ANEMIA + neutrophilic dermatosis",
        "key_drug":            "IL-1 blockers (anakinra) + G-CSF (neutropenia) + bisphosphonates (bone disease)",
    },
}


def _rng(seed):
    return random.Random(seed)


def _simulate_patient(gene, seed):
    rng      = _rng(seed)
    gd       = GENE_DATA[gene]
    onset_lo, onset_hi = gd["onset_range"]
    onset_mo = rng.randint(onset_lo, max(onset_lo + 1, onset_hi))

    colchicine   = rng.random() < gd["colchicine_resp_pct"]
    il1_blocker  = rng.random() < gd["il1_blocker_pct"]
    amyloid      = rng.random() < gd["amyloid_risk_pct"]
    snhl         = rng.random() < gd["snhl_pct"]
    rash         = rng.random() < gd["rash_pct"]
    uveitis      = rng.random() < gd["uveitis_pct"]
    crmo         = rng.random() < gd["crmo_pct"]
    anaemia      = rng.random() < gd["anaemia_pct"]

    return {
        "onset_months":  onset_mo,
        "colchicine":    colchicine,
        "il1_blocker":   il1_blocker,
        "amyloid":       amyloid,
        "snhl":          snhl,
        "rash":          rash,
        "uveitis":       uveitis,
        "crmo":          crmo,
        "anaemia":       anaemia,
    }


def _aggregate_gene(gene, gene_idx):
    patients = [_simulate_patient(gene, SEED_BASE + gene_idx * N_PER_GENE + i) for i in range(N_PER_GENE)]
    n        = len(patients)

    def pct(k):
        return round(sum(1 for p in patients if p[k]) / n * 100, 1)

    mean_onset = round(sum(p["onset_months"] for p in patients) / n, 1)

    return {
        "gene":              gene,
        "n":                 n,
        "syndrome":          GENE_DATA[gene]["syndrome"],
        "inheritance":       GENE_DATA[gene]["inheritance"],
        "locus":             GENE_DATA[gene]["locus"],
        "kda":               GENE_DATA[gene]["kda"],
        "aa":                GENE_DATA[gene]["aa"],
        "colchicine_pct":    pct("colchicine"),
        "il1_blocker_pct":   pct("il1_blocker"),
        "amyloid_pct":       pct("amyloid"),
        "snhl_pct":          pct("snhl"),
        "rash_pct":          pct("rash"),
        "uveitis_pct":       pct("uveitis"),
        "crmo_pct":          pct("crmo"),
        "anaemia_pct":       pct("anaemia"),
        "mean_onset_months": mean_onset,
        "pathognomonic":     GENE_DATA[gene]["pathognomonic"],
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    total_amyloid   = round(sum(c["amyloid_pct"]     for c in cohorts) / len(cohorts), 1)
    total_snhl      = round(sum(c["snhl_pct"]        for c in cohorts) / len(cohorts), 1)
    total_uveitis   = round(sum(c["uveitis_pct"]     for c in cohorts) / len(cohorts), 1)
    total_crmo      = round(sum(c["crmo_pct"]        for c in cohorts) / len(cohorts), 1)
    total_rash      = round(sum(c["rash_pct"]        for c in cohorts) / len(cohorts), 1)
    total_il1       = round(sum(c["il1_blocker_pct"] for c in cohorts) / len(cohorts), 1)
    total_colch     = round(sum(c["colchicine_pct"]  for c in cohorts) / len(cohorts), 1)

    return {
        "atlas":    ATLAS_NAME,
        "genes":    GENES,
        "total_n":  TOTAL_N,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "amyloid_pct":          total_amyloid,
            "snhl_pct":             total_snhl,
            "uveitis_pct":          total_uveitis,
            "crmo_pct":             total_crmo,
            "rash_pct":             total_rash,
            "il1_blocker_pct":      total_il1,
            "colchicine_pct":       total_colch,
        },
        "key_facts": [
            "MEFV (FMF): most common monogenic AID (Mediterranean); AMYLOIDOSIS in untreated = devastatingly preventable — colchicine lifelong mandatory",
            "NLRP3 (CAPS): cold-triggered urticarial rash PATHOGNOMONIC; MWS progressive SNHL; NOMID neonatal onset + CNS; canakinumab FDA-approved 2016",
            "TNFRSF1A (TRAPS): LONGEST EPISODES (days-weeks); PERIORBITAL EDEMA PATHOGNOMONIC; MIGRATORY rash; IL-1 blockers preferred over etanercept",
            "MVK (MKD/HIDS): CERVICAL LYMPHADENOPATHY in children; elevated IgD (NOT always elevated — do not exclude); statins NOT helpful (may worsen)",
            "PSTPIP1 (PAPA): PATHOGNOMONIC TRIAD pyogenic arthritis + pyoderma gangrenosum + acne conglobata; pathergy positive; anakinra first-line",
            "NOD2 (Blau/EOS): GRANULOMATOUS disease NOT episodic fever; UVEITIS most feared complication (blindness); ophthalmology MANDATORY; GOF ≠ Crohn's LOF",
            "IL1RN (DIRA): NEONATAL onset day 1-7; sterile osteomyelitis + pustulosis + periostitis; ANAKINRA CURATIVE within 24-48h — life-saving diagnosis",
            "LPIN2 (Majeed): CRMO + CONGENITAL DYSERYTHROPOIETIC ANEMIA + neutrophilic dermatosis TRIAD PATHOGNOMONIC; anemia = transfusion-dependent; first Jordanian Arab description",
        ],
        "il1_pathway_genes":  ["MEFV", "NLRP3", "TNFRSF1A", "MVK", "PSTPIP1", "IL1RN"],
        "nfkb_granuloma_genes": ["NOD2"],
        "lipid_pathway_genes":  ["LPIN2"],
        "cohort_summaries":   cohorts,
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    diagnostic_groups = {
        "pyrin_inflammasome": {
            "genes":          ["MEFV", "PSTPIP1"],
            "mechanism":      "Pyrin inflammasome hyperactivation → IL-1β overproduction",
            "shared_feature": "Pyrin-dependent IL-1β processing; respond to IL-1 inhibitors",
            "key_ddx":        "MEFV: periodic peritonitis/pleuritis. PSTPIP1: PAPA triad (skin/joint/no peritonitis)",
        },
        "nlrp3_inflammasome": {
            "genes":          ["NLRP3"],
            "mechanism":      "Constitutively active NLRP3 inflammasome → IL-1β/IL-18 excess",
            "shared_feature": "AD-GOF; cold triggers; urticarial rash; spectrum from FCAS → MWS → NOMID",
            "key_ddx":        "CAPS spectrum: FCAS (cold/mild) vs MWS (SNHL) vs NOMID (CNS+arthropathy+neonatal)",
        },
        "tnf_er_stress": {
            "genes":          ["TNFRSF1A"],
            "mechanism":      "ER retention of misfolded TNFR1 → NF-κB + secondary IL-1β",
            "shared_feature": "Long episodes; AD-GOF; periorbital edema; migratory rash",
            "key_ddx":        "TRAPS vs FMF: episode duration (TRAPS days-weeks; FMF 12-72h); TRAPS periorbital edema",
        },
        "mevalonate_pathway": {
            "genes":          ["MVK"],
            "mechanism":      "Mevalonate accumulation → geranylgeranylation defect → caspase-1 activation",
            "shared_feature": "AR; infantile onset; lymphadenopathy; IgD elevated; urine mevalonate elevated",
            "key_ddx":        "MKD vs FMF: cervical lymphadenopathy prominent; statins CONTRAPRODUCTIVE; IgD may be normal",
        },
        "il1_receptor_il1ra": {
            "genes":          ["IL1RN"],
            "mechanism":      "Absent IL-1Ra → unopposed IL-1α/IL-1β → sterile osteomyelitis + pustulosis",
            "shared_feature": "AR; neonatal onset; anakinra CURATIVE; non-episodic continuous inflammation",
            "key_ddx":        "DIRA vs neonatal sepsis: culture negative; periostitis on X-ray; dramatic anakinra response",
        },
        "nod2_granuloma": {
            "genes":          ["NOD2"],
            "mechanism":      "GOF NOD2 → constitutive NF-κB → granuloma formation in eye/joint/skin",
            "shared_feature": "AD-GOF; granulomatous (not neutrophilic); onset childhood; no prominent fever",
            "key_ddx":        "Blau/EOS vs sarcoidosis: no hilar adenopathy; ACE often normal; onset <5yr; NOD2 GOF (not LOF-Crohn's)",
        },
        "lipid_inflammasome": {
            "genes":          ["LPIN2"],
            "mechanism":      "PA accumulation (Lipin-2 absent) → NF-κB → CRMO + CDA + dermatosis",
            "shared_feature": "AR; CRMO continuous; CDA transfusion-dependent; consanguineous Arab families",
            "key_ddx":        "Majeed vs other CRMO: CDA distinguishes (anemia not seen in other CRMO); Jordanian/Arab founder",
        },
    }

    pathognomonic_features = {
        "aa_amyloidosis":         "AA amyloidosis (renal) in untreated FMF = PATHOGNOMONIC LONG-TERM COMPLICATION; prevented by colchicine lifelong",
        "cold_urticaria_caps":    "Cold-triggered urticarial rash = PATHOGNOMONIC for NLRP3/CAPS; distinguish from cold urticaria (no fever/systemic)",
        "periorbital_edema":      "Periorbital edema = PATHOGNOMONIC for TRAPS; combined with migratory rash and long episodes",
        "cervical_adenopathy":    "Cervical lymphadenopathy = PATHOGNOMONIC in MKD/HIDS children; trigger: vaccination",
        "papa_triad":             "Pyogenic arthritis + pyoderma gangrenosum + acne conglobata = PATHOGNOMONIC PAPA TRIAD; pathergy positive",
        "blau_granuloma_triad":   "Uveitis + boggy arthritis + ichthyosiform dermatitis = PATHOGNOMONIC Blau/EOS triad; NO prominent fever",
        "dira_neonatal_osteom":   "Neonatal-onset sterile multifocal osteomyelitis + pustulosis + periostitis = PATHOGNOMONIC DIRA",
        "majeed_cda":             "CRMO + congenital dyserythropoietic anemia + neutrophilic dermatosis = PATHOGNOMONIC Majeed triad",
        "colchicine_prevents_amyloid": "FMF CRITICAL: colchicine prevents BOTH attacks AND AA amyloidosis — never stop; lifelong therapy",
        "il1_blocker_coverage":   "ALL 8 genes respond to IL-1 pathway inhibition (anakinra/canakinumab); primary or adjunctive treatment",
        "etanercept_traps_caution": "TNFRSF1A (TRAPS): etanercept (soluble TNFR2) less effective than IL-1 blockers — do NOT use etanercept first-line",
        "statins_mvk_ci":         "MVK (MKD/HIDS): statins CONTRAINDICATED — inhibit HMG-CoA reductase (upstream) → further reduce mevalonate flux",
    }

    return {
        "gene_cohorts":         cohorts,
        "diagnostic_groups":    diagnostic_groups,
        "pathognomonic_features": pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: Recurrent fever with systemic features → ethnicity + episode duration + trigger assessment",
            "Step 2: Episode 12-72h + peritonitis/pleuritis + Mediterranean origin → MEFV first (panel: M694V, E148Q, M680I, V726A)",
            "Step 3: Cold-triggered urticarial rash (any age, any ethnicity) → NLRP3 panel; check SNHL + CNS involvement",
            "Step 4: Long episodes (>3 days) + periorbital edema + migratory rash → TNFRSF1A sequencing; IL-1 blocker preferred over etanercept",
            "Step 5: Infantile fever + cervical lymphadenopathy + post-vaccination flare → MVK enzyme + urine mevalonate acid during fever",
            "Step 6: Pyogenic arthritis + pyoderma gangrenosum + acne → PSTPIP1 panel; pathergy test",
            "Step 7: Childhood granulomatous triad (uveitis + arthritis + skin) → NOD2 panel; ophthalmology referral URGENT",
            "Step 8: Neonatal sterile osteomyelitis + pustulosis + periostitis → IL1RN sequencing; start anakinra empirically IMMEDIATELY",
            "Step 9: CRMO + congenital anemia + neutrophilic dermatosis (infancy) → LPIN2 sequencing; transfusion support + IL-1 blocker",
            "Step 10: All diagnoses: check for AA amyloidosis (SAA levels annually); ophthalmology if any ocular symptoms",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "FMF (Familial Mediterranean Fever)", "definition": "Most common monogenic autoinflammatory disease. MEFV mutation → pyrin inflammasome hyperactivation → IL-1β. Recurrent febrile attacks 12-72h: serositis (peritonitis/pleuritis/pericarditis/synovitis). Mediterranean origin (Turks/Armenians/Arabs/Ashkenazi). AA amyloidosis in untreated = preventable tragedy. Colchicine lifelong mandatory — prevents attacks AND amyloidosis."},
            {"term": "CAPS (Cryopyrin-Associated Periodic Syndrome)", "definition": "AD-GOF NLRP3 spectrum. FCAS (cold-triggered urticarial rash + fever — mildest), MWS (progressive SNHL — audiogram annually), NOMID/CINCA (neonatal onset + CNS involvement: papilledema, aseptic meningitis, sensorineural deafness, overgrowth arthropathy — most severe). Cold trigger + urticarial rash (not true urticaria) = PATHOGNOMONIC for CAPS. Canakinumab (anti-IL-1β) FDA-approved 2016."},
            {"term": "TRAPS (TNF Receptor-Associated Periodic Syndrome)", "definition": "AD-GOF TNFRSF1A. ER retention of misfolded TNFR1 → NF-κB + IL-1β. Longest episodes of all periodic fevers (3 days to weeks; FMF = 12-72h). PERIORBITAL EDEMA PATHOGNOMONIC. Migratory erythematous rash. Myalgia. Etanercept (TNFR2 soluble) less effective than IL-1 blockers. Anakinra/canakinumab preferred. Amyloidosis risk."},
            {"term": "MKD / HIDS (Mevalonate Kinase Deficiency)", "definition": "AR MVK. Partial MVK → HIDS (periodic fever + lymphadenopathy + IgD elevation); complete LOF → mevalonic aciduria (metabolic crisis). CERVICAL LYMPHADENOPATHY cardinal in children. IgD elevated >100 IU/mL in ~70% — but may be normal (do NOT exclude). Urine mevalonic acid during fever = diagnostic. Statins CONTRAINDICATED. Canakinumab FDA-approved 2016."},
            {"term": "PAPA Syndrome", "definition": "AD-GOF PSTPIP1. PATHOGNOMONIC TRIAD: (1) Pyogenic sterile arthritis (destructive, especially knee/ankle/wrist), (2) Pyoderma gangrenosum (ulcerating skin lesions; pathergy), (3) Acne conglobata (severe cystic). PAMI (PSTPIP1-associated myeloid-related proteinemia) = severe rare variant with hepatosplenomegaly. Anakinra first-line (IL-1 blocker). Steroids for pyoderma lesions."},
            {"term": "Blau Syndrome / Early-Onset Sarcoidosis (EOS)", "definition": "AD-GOF NOD2 (CARD15). Blau = familial; EOS = sporadic (same mutation, same disease). GRANULOMATOUS TRIAD: (1) Panuveitis (blindness risk — ophthalmology MANDATORY), (2) Boggy polyarticular synovitis with camptodactyly, (3) Ichthyosiform/macular dermatitis. No prominent fever. No hilar adenopathy (unlike sarcoidosis). GOF NOD2 mutations ≠ LOF NOD2 (Crohn's susceptibility). TNF inhibitors (infliximab) for refractory uveitis."},
            {"term": "DIRA (Deficiency of IL-1 Receptor Antagonist)", "definition": "AR IL1RN. Absent IL-1Ra → unopposed IL-1α/IL-1β signalling. NEONATAL ONSET (day 1-7): sterile multifocal osteomyelitis + generalized pustulosis + periostitis (widened periosteal reaction on X-ray). ANAKINRA IS CURATIVE — dramatic response within 24-48h. Never stop (lifelong daily anakinra). Life-threatening if undiagnosed. Canadian/Puerto Rican founders. Distinguish from neonatal sepsis: culture-negative, periostitis present."},
            {"term": "Majeed Syndrome", "definition": "AR LPIN2. MAJEED TRIAD: (1) CRMO (chronic recurrent multifocal osteomyelitis — sterile bone inflammation from infancy), (2) Congenital dyserythropoietic anemia (CDA — transfusion-dependent), (3) Neutrophilic dermatosis (Sweet syndrome-like or psoriasiform). CDA distinguishes Majeed from all other CRMO-associated MAIDs. Jordanian Arab consanguineous families. IL-1 blockers + G-CSF + bisphosphonates. Amyloidosis risk."},
            {"term": "AA Amyloidosis", "definition": "Secondary (reactive) amyloidosis — serum amyloid A (SAA) protein deposits in organs (kidney primary). Complication of chronic untreated/poorly-treated autoinflammatory diseases: FMF (highest risk), TRAPS, MKD, Majeed. Renal failure develops. PREVENTABLE by: colchicine in FMF, IL-1 blockers in other MAIDs. Monitor SAA annually. Renal biopsy for diagnosis. Treatment of underlying AID + anti-IL-6 (tocilizumab) or anti-IL-1 to reduce SAA."},
            {"term": "Colchicine in FMF", "definition": "Microtubule polymerisation inhibitor that disrupts pyrin inflammasome assembly. FIRST-LINE for FMF. Highly effective: prevents attacks in 90% + prevents AA amyloidosis. Must be taken LIFELONG (including during remission — amyloid risk if stopped). Safe in pregnancy. M694V homozygous: highest amyloid risk → strict adherence essential. Colchicine-resistant FMF (persistent attacks or SAA elevation despite 3mg/day): add anakinra or canakinumab."},
            {"term": "Anakinra (IL-1 Receptor Antagonist)", "definition": "Recombinant human IL-1Ra. Competitive antagonist of both IL-1α and IL-1β at IL-1R1. Daily subcutaneous injection. Approved for DIRA (CURATIVE), NOMID, Still's disease. Off-label: FMF-resistant, TRAPS, MKD, PAPA, Majeed. Rapid onset (24-48h). Short half-life (6h): ideal for as-needed use in FMF attacks; must be daily for continuous MAIDs (DIRA, Blau). Less convenient than canakinumab (monthly)."},
            {"term": "Canakinumab (Anti-IL-1β)", "definition": "Human monoclonal antibody against IL-1β. Monthly subcutaneous injection. FDA-approved: CAPS (2009), SJIA (2013), TRAPS/MKD/FMF (2016). Longer dosing interval than anakinra (monthly vs daily). Does not block IL-1α (anakinra blocks both). Preferred for CAPS (NLRP3) and MKD. NLRP3 CAPS: prevents SNHL progression — start early in MWS. Cost-effectiveness vs anakinra varies by disease."},
            {"term": "Pathergy Test (PAPA/PSTPIP1)", "definition": "Skin hyper-reactivity to trauma: minor needle puncture or trauma → exaggerated inflammatory papule/pustule at site. >90% positive in PAPA syndrome. Reflects underlying IL-1 pathway hyperactivation via PSTPIP1-pyrin axis. Also seen in Behçet's disease (different gene). Positive pathergy = important clinical pointer to PAPA/PSTPIP1 in pyoderma gangrenosum workup."},
            {"term": "Mevalonate Pathway & MAIDs", "definition": "Mevalonate pathway: HMG-CoA → mevalonate (MVK) → phosphomevalonate → ... → isoprenoids (cholesterol, geranylgeranyl-PP). MVK LOF → mevalonate accumulates; reduced geranylgeranylation → Rho-GTPase/Rac1 dysregulation → NLRP3/caspase-1 activation → IL-1β. WHY STATINS ARE CONTRAINDICATED: statins inhibit HMG-CoA reductase (upstream of MVK block) → further reduces mevalonate production → worsens isoprenoid depletion."},
        ],
        "key_distinguishers": {
            "episode_duration":         "TRAPS: days-weeks (longest). MKD: 4-7 days. FMF: 12-72h (shortest of classical). NLRP3/CAPS: hours-days. DIRA/Majeed/NOD2-Blau: continuous (not episodic).",
            "il1_vs_tnf_vs_colchicine": "FMF: colchicine first-line. ALL MAIDs: IL-1 blockers (anakinra/canakinumab) effective. TRAPS: IL-1 blockers >> etanercept. MVK: statins ABSOLUTELY NOT. NOD2-Blau: steroids + MTX ± TNF inhibitors.",
            "ethnicity_clues":          "FMF: Turkish/Armenian/Arab/Ashkenazi. MVK/HIDS: NW European/Dutch/French. DIRA: Canadian/Puerto Rican founders. Majeed: Jordanian/Arab consanguineous. TRAPS: Irish/Scottish (but pan-ethnic).",
            "neonatal_onset":           "DIRA (IL1RN): day 1-7 osteomyelitis + pustulosis + periostitis → anakinra immediately. Majeed (LPIN2): CRMO + CDA from infancy. NOMID (NLRP3): neonatal systemic inflammation + CNS.",
            "ocular_involvement":       "NOD2 (Blau): panuveitis PATHOGNOMONIC — blindness risk; ophthalmology MANDATORY. NLRP3 (NOMID): papilledema/cataracts. Others: uveitis uncommon.",
            "amyloid_risk":             "Highest: MEFV (M694V homozygous) prevented by colchicine. Moderate: TNFRSF1A/TRAPS. Lower: MVK/LPIN2. MONITOR: SAA levels annually ALL untreated MAIDs.",
            "bone_disease":             "CRMO present: IL1RN (DIRA) + LPIN2 (Majeed). Absent in: MEFV/NLRP3/TNFRSF1A/MVK/PSTPIP1/NOD2. CDA (anemia): ONLY Majeed/LPIN2.",
        },
        "treatment_matrix": {
            "colchicine":       "FMF (MEFV): first-line, prevent attacks + amyloidosis; others: limited benefit",
            "anakinra":         "DIRA (IL1RN): CURATIVE; NOMID (NLRP3): first-line; all others: effective",
            "canakinumab":      "CAPS (NLRP3): FDA-approved; MKD (MVK): FDA-approved; TRAPS/FMF: FDA-approved 2016",
            "corticosteroids":  "NOD2 (Blau): first-line; PAPA pyoderma: adjunct; all others: burst only for flares",
            "methotrexate":     "NOD2 (Blau): steroid-sparing; PAPA: second-line for arthritis",
            "infliximab":       "NOD2 (Blau refractory uveitis): preferred TNF inhibitor; TRAPS: less effective than IL-1",
            "etanercept_avoid": "TNFRSF1A (TRAPS): etanercept LESS EFFECTIVE — prefer IL-1 blocker; may mask amyloid",
            "statins_avoid":    "MVK (MKD): STATINS CONTRAINDICATED — worsen mevalonate pathway defect",
            "gcf_bisphosphonate": "LPIN2 (Majeed): G-CSF for neutropenia; bisphosphonates for CRMO bone disease",
        },
    }


if __name__ == "__main__":
    import json
    ov = generate_overview()
    bk = generate_breakdown()
    df = generate_definitions()
    print(f"Overview genes: {ov['genes']}")
    print(f"Total N: {ov['total_n']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Gene cohorts: {len(bk['gene_cohorts'])}")
    print(f"Definitions: {len(df['definitions'])}")
    print("OK")
