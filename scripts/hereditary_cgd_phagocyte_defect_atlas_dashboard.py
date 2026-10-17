#!/usr/bin/env python3
"""Hereditary-CGD-Phagocyte-Defect-Atlas -- Complete 8-Gene Reference
Chronic Granulomatous Disease and Hereditary Phagocyte Defects: NADPH oxidase function by gene
(CYBB/XL-CGD-gp91phox-65pct-All-CGD-IFN-gamma-FDA1990-First-Cytokine-Immunotherapy,
 CYBA/p22phox-AR-CGD-Type2-Heterodimer-gp91-Absent-CYBB-Intact-Middle-Eastern-Founder,
 NCF1/p47phox-AR-CGD-Most-Common-AR-Pseudogene-MLPA-MANDATORY-NOT-Sanger,
 NCF2/p67phox-AR-CGD-Activation-Domain-No-Pseudogene-Standard-Sequencing,
 NCF4/p40phox-AR-CGD-IBD-DOMINANT-Phagosomal-Partial-DHR-PATHOGNOMONIC,
 EROS/CYBC1-AR-ER-Chaperone-gp91-Absent-CYBB-Intact-2019-Discovery-Ultra-Rare,
 RAC2/Rho-GTPase-AD-Dominant-Negative-Leukocytosis-PATHOGNOMONIC-Broader-Than-CGD,
 MPO/Myeloperoxidase-AR-Most-Common-Phagocyte-Disorder-1in2000-4000-Clinically-Silent-Diabetes-Candida):

CYBB   (gp91phox / NOX2; 570aa; Xp21.1; XL;
         Most common CGD: 65% of ALL CGD cases worldwide;
         XL inheritance: males exclusively (hemizygous); females carrier (mild carrier oxidase activity);
         NADPH oxidase catalytic core: gp91phox = NOX2, the electron-transporting flavocytochrome b558 alpha-subunit;
         NBT test: NEGATIVE (no superoxide production) — PATHOGNOMONIC for CGD;
         DHR test: DHR flow cytometry index FLAT (no oxidative burst) — gold standard diagnostic;
         Dominant pathogens: Aspergillus fumigatus (pulmonary), Staphylococcus aureus (skin/lymph nodes/liver abscess),
           Burkholderia cepacia (pneumonia, sepsis), Nocardia, Serratia;
         IFN-γ (Actimmune/interferon gamma-1b): FDA-approved 1990 as FIRST cytokine immunotherapy —
           reduces infections 70% in XL-CGD; prophylaxis TIW subcutaneous;
         Prophylaxis: TMP-SMX + itraconazole/voriconazole (antifungal);
         Aspergillus risk: 35-40% lifetime cumulative invasive aspergillosis without prophylaxis;
         IBD: granulomatous colitis in 30-40% of XL-CGD patients;
         HSCT curative; gene therapy: lentiviral gp91phox (CYBB) clinical trials showing 70%+ cure;
         seed SEED_BASE+0).
CYBA   (p22phox; 195aa; 16q24.3; AR;
         ~5% of all CGD; AR inheritance;
         p22phox is the ALPHA subunit of flavocytochrome b558 — forms obligate heterodimer with gp91phox;
         Without p22phox: gp91phox is unstable and degraded — BOTH subunits absent by flow cytometry
           despite only CYBA mutated;
         CYBA LOF → gp91phox absent (even though CYBB gene intact) — MLPA CYBB negative but CYBB protein absent;
         DHR abnormal: same as XL-CGD clinically but AR — affects both sexes equally;
         Founder variant: 540+1G→T in Middle Eastern populations;
         IBD: granulomatous colitis in ~30%;
         Distinguisher: check CYBA flow cytometry (anti-p22phox) when gp91phox absent but CYBB negative;
         seed SEED_BASE+1).
NCF1   (p47phox; 390aa; 7q11.23; AR;
         Most common AR-CGD: ~25% of all AR-CGD patients (≈8-9% of all CGD);
         NCF1 encodes p47phox — cytosolic component essential for NADPH oxidase assembly;
         CRITICAL: NCF1 has two near-identical pseudogenes (NCF1B and NCF1C) on chromosome 7q11.23 —
           GT deletion recombination between pseudogenes accounts for 75-80% of NCF1 CGD alleles;
         MLPA OR long-range PCR MANDATORY — routine Sanger/NGS sequencing MISSES GT-deletion alleles;
         IBD: granulomatous colitis particularly prominent in NCF1-CGD (~40%);
         Generally MILDER than XL-CGD — later onset, fewer Aspergillus infections, better survival;
         seed SEED_BASE+2).
NCF2   (p67phox; 526aa; 1q25.3; AR;
         ~5% of AR-CGD; rarer form;
         p67phox is the activation domain-containing cytosolic component — required for NADPH oxidase
           electron transfer activation;
         No pseudogene issue (unlike NCF1) — standard sequencing reliable;
         Similar clinical severity to NCF1-CGD;
         IBD present in ~25-30%;
         seed SEED_BASE+3).
NCF4   (p40phox; 339aa; 22q13.1; AR;
         Rarest cytosolic subunit; <1% of CGD cases;
         p40phox plays a specific role in phagosomal NADPH oxidase activation (not plasma membrane);
         UNIQUE PHENOTYPE: Granulomatous IBD DOMINATES — severe Crohn's-like colitis is often PRESENTING feature;
         Infections milder than other CGD subtypes — partial residual NADPH oxidase activity at plasma membrane;
         DHR test may show PARTIAL impairment (not flat zero) — partial oxidative burst retained;
         Pathognomonic: Severe colitis + MILD infection susceptibility + partial DHR → NCF4 testing;
         seed SEED_BASE+4).
EROS   (CYBC1/EROS; 291aa; 1p22.1; AR;
         EROS = Endoplasmic Reticulum Oxidoreductase Required for gp91phox Stability; identified 2019;
         EROS is a transmembrane ER protein that acts as a maturation chaperone for gp91phox;
         Without EROS: gp91phox is misfolded in the ER and degraded → functional CGD phenotype;
         Clinically and biochemically IDENTICAL to gp91phox (CYBB) deficiency — but CYBB gene is intact;
         Ultra-rare: <25 reported patients worldwide as of 2024;
         CRITICAL DIAGNOSTIC PITFALL: gp91phox absent by flow cytometry, CYBB sequencing negative → EROS panel;
         DHR flat, NBT negative (identical to XL-CGD);
         seed SEED_BASE+5).
RAC2   (RAC2 GTPase; 192aa; 22q13.1; AD/AR;
         RAC2 = Ras-related C3 botulinum toxin substrate 2 — Rho-family GTPase essential for NADPH oxidase
           complex assembly at the plasma membrane;
         RAC2 activates p67phox during oxidase assembly — required for full oxidase activation;
         DOMINANT NEGATIVE (D57N or similar): AD inheritance; impairs wild-type RAC2 function + disrupts
           neutrophil migration/chemotaxis AND degranulation AND actin cytoskeleton —
           combined neutrophil functional defect (BROADER than CGD alone);
         LOF biallelic AR form: CGD-like with selective NADPH oxidase failure;
         CRITICAL DIAGNOSTIC CLUE for DN form: LEUKOCYTOSIS (WBC elevated 20-100 x10^9/L) PATHOGNOMONIC —
           impaired neutrophil adhesion and diapedesis → neutrophil trapped in circulation;
         Severe infections with catalase-positive organisms PLUS poor wound healing;
         seed SEED_BASE+6).
MPO    (Myeloperoxidase; 745aa; 17q22; AR;
         Myeloperoxidase is the green pigment of neutrophil azurophil granules —
           converts H2O2 to hypochlorous acid (HOCl) — antimicrobial secondary oxidant;
         MOST COMMON hereditary phagocyte disorder: prevalence ~1:2000-4000 live births;
         USUALLY CLINICALLY SILENT: 99% of MPO-deficient individuals never have infections —
           redundant antimicrobial killing pathways (NADPH oxidase H2O2 itself is bactericidal);
         CRITICAL: DHR test NORMAL, NBT test NORMAL — NADPH oxidase intact, respiratory burst present;
         KEY DISTINGUISHER FROM CGD: flow cytometry peroxidase staining negative (MPO protein absent);
         PATHOGNOMONIC CLINICAL SCENARIO: MPO deficiency + TYPE 2 DIABETES → recurrent severe Candida infections;
         Diagnosis often incidental on automated CBC differential (zero peroxidase activity on Sysmex analyzers);
         seed SEED_BASE+7).
"""
import random
import math

SEED_BASE = 3862

ATLAS_GENES = [
    {
        "name": "CYBB",
        "full_name": "gp91phox / NADPH oxidase catalytic subunit (NOX2)",
        "aa": 570,
        "locus": "Xp21.1",
        "inheritance": "XL",
        "mechanism": "LOF",
        "syndrome": "XL-CGD Type1 / gp91phox / 65pct-All-CGD / IFN-gamma-FDA1990",
        "mw_kda": 65.3,
        "critical_rules": [
            "CYBB-XL-MALES-ONLY-65PCT-ALL-CGD",
            "CYBB-DHR-FLAT-GOLD-STANDARD-DIAGNOSIS",
            "CYBB-IFN-GAMMA-FDA1990-FIRST-CYTOKINE-IMMUNOTHERAPY",
            "CYBB-ASPERGILLUS-35-40PCT-DOMINANT-PATHOGEN",
        ],
        "distinguishing_feature": "Most common CGD (65%); XL males only; DHR flat gold standard; IFN-γ FDA1990 first cytokine immunotherapy; Aspergillus 35-40%; IBD 30-40%",
        "first_line_treatment": "TMP-SMX + itraconazole/voriconazole prophylaxis; IFN-γ (Actimmune) 50 mcg/m2 TIW SC; HSCT curative (HLA-matched); lentiviral CYBB gene therapy clinical trials",
        "pathognomonic": "Male child with recurrent catalase-positive infections (Aspergillus/Staph/Nocardia) + FLAT DHR (no oxidative burst) + gp91phox absent by flow cytometry",
        "onset_yrs_mean": 2.5,
        "penetrance_pct": 100,
        "nbt_abnormal_pct": 100,
        "dhr_abnormal_pct": 100,
        "aspergillus_risk_pct": 38,
        "staph_aureus_risk_pct": 80,
        "ibd_pct": 35,
        "ifn_gamma_benefit_pct": 70,
        "hsct_curative_pct": 85,
        "gene_therapy_eligible_pct": 60,
        "dx_lag_yrs": 1.2,
        "leukocytosis_pct": 0,
    },
    {
        "name": "CYBA",
        "full_name": "p22phox / flavocytochrome b558 alpha subunit",
        "aa": 195,
        "locus": "16q24.3",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-CGD Type2 / p22phox / Obligate-gp91-Heterodimer",
        "mw_kda": 21.0,
        "critical_rules": [
            "CYBA-AR-AFFECTS-BOTH-SEXES-UNLIKE-CYBB",
            "CYBA-GP91-ABSENT-EVEN-THOUGH-CYBB-INTACT-HETERODIMER",
            "CYBA-MIDDLE-EASTERN-FOUNDER-540+1G>T",
        ],
        "distinguishing_feature": "AR affects both sexes; p22phox-gp91phox obligate heterodimer — gp91 absent despite intact CYBB; Middle Eastern 540+1G→T founder; IBD ~30%",
        "first_line_treatment": "TMP-SMX + azole antifungal prophylaxis; IFN-γ (less benefit than XL-CGD); HSCT curative; test CYBA by anti-p22phox flow cytometry when gp91 absent and CYBB negative",
        "pathognomonic": "DHR flat + gp91phox absent by flow + CYBB sequencing negative → test CYBA; both sexes affected; Middle Eastern ancestry → 540+1G→T",
        "onset_yrs_mean": 3.0,
        "penetrance_pct": 100,
        "nbt_abnormal_pct": 100,
        "dhr_abnormal_pct": 100,
        "aspergillus_risk_pct": 30,
        "staph_aureus_risk_pct": 70,
        "ibd_pct": 30,
        "ifn_gamma_benefit_pct": 40,
        "hsct_curative_pct": 85,
        "gene_therapy_eligible_pct": 20,
        "dx_lag_yrs": 1.8,
        "leukocytosis_pct": 0,
    },
    {
        "name": "NCF1",
        "full_name": "p47phox / cytosolic NADPH oxidase assembly factor",
        "aa": 390,
        "locus": "7q11.23",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-CGD Type3 / p47phox / Most-Common-AR-CGD / Pseudogene-MLPA-MANDATORY",
        "mw_kda": 43.6,
        "critical_rules": [
            "NCF1-PSEUDOGENE-NCF1B-NCF1C-GT-DELETION-75-80PCT-MLPA-MANDATORY-NOT-SANGER",
            "NCF1-MOST-COMMON-AR-CGD-25PCT-AR-SUBSET",
            "NCF1-MILDER-PHENOTYPE-LATER-ONSET",
            "NCF1-IBD-GRANULOMATOUS-COLITIS-40PCT",
        ],
        "distinguishing_feature": "Most common AR-CGD (25% of AR-CGD); NCF1B/NCF1C pseudogenes — GT deletion missed by Sanger; MLPA/long-range PCR mandatory; IBD 40%; milder than XL-CGD",
        "first_line_treatment": "TMP-SMX + antifungal prophylaxis; IFN-γ (moderate benefit); HSCT curative; CRITICAL: order MLPA not Sanger for NCF1; gastroenterology co-management for IBD",
        "pathognomonic": "AR-CGD + DHR flat + p47phox absent + routine sequencing negative → MLPA mandatory (GT deletion 75-80% of alleles in NCF1B/NCF1C pseudogene recombination)",
        "onset_yrs_mean": 4.5,
        "penetrance_pct": 100,
        "nbt_abnormal_pct": 100,
        "dhr_abnormal_pct": 100,
        "aspergillus_risk_pct": 20,
        "staph_aureus_risk_pct": 65,
        "ibd_pct": 40,
        "ifn_gamma_benefit_pct": 35,
        "hsct_curative_pct": 85,
        "gene_therapy_eligible_pct": 15,
        "dx_lag_yrs": 2.5,
        "leukocytosis_pct": 0,
    },
    {
        "name": "NCF2",
        "full_name": "p67phox / NADPH oxidase activation domain subunit",
        "aa": 526,
        "locus": "1q25.3",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-CGD Type4 / p67phox / Activation-Domain / No-Pseudogene",
        "mw_kda": 59.6,
        "critical_rules": [
            "NCF2-NO-PSEUDOGENE-STANDARD-SEQUENCING-RELIABLE-UNLIKE-NCF1",
            "NCF2-ACTIVATION-DOMAIN-ELECTRON-TRANSFER",
            "NCF2-RARER-AR-CGD-5PCT",
        ],
        "distinguishing_feature": "~5% of AR-CGD; p67phox activation domain for electron transfer; no pseudogene unlike NCF1 — standard sequencing reliable; IBD 25-30%; similar severity to NCF1",
        "first_line_treatment": "TMP-SMX + antifungal prophylaxis; IFN-γ (moderate benefit); HSCT curative; standard NGS sequencing sufficient (unlike NCF1)",
        "pathognomonic": "AR-CGD + DHR flat + p67phox absent + affects both sexes; standard sequencing sufficient (no pseudogene interference unlike NCF1)",
        "onset_yrs_mean": 4.0,
        "penetrance_pct": 100,
        "nbt_abnormal_pct": 100,
        "dhr_abnormal_pct": 100,
        "aspergillus_risk_pct": 22,
        "staph_aureus_risk_pct": 68,
        "ibd_pct": 27,
        "ifn_gamma_benefit_pct": 35,
        "hsct_curative_pct": 85,
        "gene_therapy_eligible_pct": 10,
        "dx_lag_yrs": 2.2,
        "leukocytosis_pct": 0,
    },
    {
        "name": "NCF4",
        "full_name": "p40phox / phagosomal NADPH oxidase assembly factor",
        "aa": 339,
        "locus": "22q13.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-CGD Type5 / p40phox / IBD-DOMINANT / Partial-DHR-PATHOGNOMONIC",
        "mw_kda": 40.0,
        "critical_rules": [
            "NCF4-GRANULOMATOUS-IBD-DOMINANT-PRESENTING-FEATURE-PATHOGNOMONIC",
            "NCF4-PARTIAL-DHR-NOT-FLAT-UNLIKE-OTHER-CGD",
            "NCF4-MILDER-INFECTION-SUSCEPTIBILITY-THAN-OTHER-CGD",
            "NCF4-PHAGOSOMAL-SPECIFIC-NOT-PLASMA-MEMBRANE",
        ],
        "distinguishing_feature": "Rarest CGD (<1%); IBD dominates (75%) and often PRESENTING feature; PARTIAL DHR (not flat zero); mild infection susceptibility; phagosomal not plasma-membrane oxidase",
        "first_line_treatment": "Gastroenterology-first approach (IBD often the presenting problem); anti-TNF/immunosuppression for colitis; TMP-SMX antifungal prophylaxis; HSCT for severe cases",
        "pathognomonic": "Severe Crohn's-like granulomatous colitis + PARTIAL DHR (not flat) + mild infection susceptibility → NCF4 (p40phox); phagosomal-selective partial oxidase defect",
        "onset_yrs_mean": 5.0,
        "penetrance_pct": 100,
        "nbt_abnormal_pct": 80,
        "dhr_abnormal_pct": 70,
        "aspergillus_risk_pct": 10,
        "staph_aureus_risk_pct": 45,
        "ibd_pct": 75,
        "ifn_gamma_benefit_pct": 20,
        "hsct_curative_pct": 80,
        "gene_therapy_eligible_pct": 5,
        "dx_lag_yrs": 3.5,
        "leukocytosis_pct": 0,
    },
    {
        "name": "EROS",
        "full_name": "EROS/CYBC1 / ER chaperone for gp91phox maturation",
        "aa": 291,
        "locus": "1p22.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "EROS-CGD / CYBC1 / ER-Chaperone-gp91 / Ultra-Rare-2019-Discovery",
        "mw_kda": 34.0,
        "critical_rules": [
            "EROS-GP91-ABSENT-BUT-CYBB-INTACT-MISFOLDED-ER-CHAPERONE",
            "EROS-ULTRA-RARE-LT25-CASES-WORLDWIDE",
            "EROS-CYBB-NEGATIVE-GP91-ABSENT-ORDER-EROS-CYBC1-NEXT",
            "EROS-DISCOVERED-2019",
        ],
        "distinguishing_feature": "Ultra-rare (<25 cases worldwide 2024); discovered 2019; ER chaperone for gp91phox — CYBB intact but gp91 absent; clinically identical to XL-CGD; diagnostic pitfall",
        "first_line_treatment": "Same prophylaxis as CGD (TMP-SMX + antifungal); HSCT curative; no gene therapy specific protocol; order EROS/CYBC1 sequencing when gp91phox absent + CYBB and CYBA both negative",
        "pathognomonic": "CGD phenotype (DHR flat, NBT negative) + gp91phox absent by flow + CYBB sequencing negative + CYBA negative → EROS/CYBC1 panel (ER chaperone discovered 2019)",
        "onset_yrs_mean": 3.5,
        "penetrance_pct": 100,
        "nbt_abnormal_pct": 100,
        "dhr_abnormal_pct": 100,
        "aspergillus_risk_pct": 32,
        "staph_aureus_risk_pct": 72,
        "ibd_pct": 25,
        "ifn_gamma_benefit_pct": 50,
        "hsct_curative_pct": 85,
        "gene_therapy_eligible_pct": 5,
        "dx_lag_yrs": 4.0,
        "leukocytosis_pct": 0,
    },
    {
        "name": "RAC2",
        "full_name": "RAC2 GTPase / Rho-family GTPase NADPH assembly regulator",
        "aa": 192,
        "locus": "22q13.1",
        "inheritance": "AD",
        "mechanism": "DN/LOF",
        "syndrome": "RAC2-Neutrophil-Defect / Dominant-Negative-D57N / Leukocytosis-PATHOGNOMONIC / Broader-CGD",
        "mw_kda": 21.5,
        "critical_rules": [
            "RAC2-DOMINANT-NEGATIVE-D57N-AD-BROADER-THAN-CGD",
            "RAC2-LEUKOCYTOSIS-PATHOGNOMONIC-IMPAIRED-DIAPEDESIS",
            "RAC2-COMBINED-NADPH-OXIDASE-PLUS-MIGRATION-PLUS-DEGRANULATION-DEFECT",
            "RAC2-GTPase-RHOGTPASE-NADPH-ASSEMBLY",
        ],
        "distinguishing_feature": "AD dominant negative (D57N); LEUKOCYTOSIS (WBC 20-100×10^9/L) PATHOGNOMONIC (impaired diapedesis); combined NADPH + migration + degranulation defect; broader than pure CGD",
        "first_line_treatment": "HSCT curative (especially DN form); prophylaxis as per CGD; PATHOGNOMONIC: leukocytosis + recurrent infections + poor wound healing → test RAC2; G-CSF trial in some cases",
        "pathognomonic": "CGD-like infections + LEUKOCYTOSIS (20-100×10^9/L) + poor wound healing + neutrophil migration defect + partial DHR → RAC2 dominant negative (D57N); AD inheritance",
        "onset_yrs_mean": 0.5,
        "penetrance_pct": 95,
        "nbt_abnormal_pct": 70,
        "dhr_abnormal_pct": 65,
        "aspergillus_risk_pct": 25,
        "staph_aureus_risk_pct": 60,
        "ibd_pct": 15,
        "ifn_gamma_benefit_pct": 25,
        "hsct_curative_pct": 80,
        "gene_therapy_eligible_pct": 5,
        "dx_lag_yrs": 1.0,
        "leukocytosis_pct": 90,
    },
    {
        "name": "MPO",
        "full_name": "Myeloperoxidase / neutrophil HOCl-producing oxidase",
        "aa": 745,
        "locus": "17q22",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "MPO-Deficiency / Most-Common-Phagocyte-Disorder-1in2000-4000 / Clinically-Silent-99pct / Diabetes-Candida-PATHOGNOMONIC",
        "mw_kda": 84.3,
        "critical_rules": [
            "MPO-MOST-COMMON-PHAGOCYTE-DISORDER-1IN2000-4000",
            "MPO-CLINICALLY-SILENT-99PCT-NO-INFECTIONS",
            "MPO-DHR-NORMAL-NBT-NORMAL-DISTINGUISHES-FROM-CGD",
            "MPO-DIABETES-CANDIDA-PATHOGNOMONIC-COMBINATION",
        ],
        "distinguishing_feature": "Most common phagocyte disorder (1:2000-4000); 99% clinically silent; DHR NORMAL and NBT NORMAL (distinguishes from all CGD); Diabetes+MPO→ severe Candida PATHOGNOMONIC",
        "first_line_treatment": "Usually no treatment needed (99% asymptomatic); tight glycaemic control in diabetics (critical — diabetes unmasks clinical MPO deficiency); antifungal azole if symptomatic; no HSCT/gene therapy indicated",
        "pathognomonic": "Sysmex CBC differential flags zero peroxidase activity (incidental diagnosis) + DHR NORMAL + NBT NORMAL (intact NADPH oxidase) + MPO stain absent on neutrophils → MPO deficiency; if diabetic → high risk severe Candida",
        "onset_yrs_mean": 35.0,
        "penetrance_pct": 5,
        "nbt_abnormal_pct": 0,
        "dhr_abnormal_pct": 0,
        "aspergillus_risk_pct": 3,
        "staph_aureus_risk_pct": 10,
        "ibd_pct": 5,
        "ifn_gamma_benefit_pct": 0,
        "hsct_curative_pct": 10,
        "gene_therapy_eligible_pct": 0,
        "dx_lag_yrs": 15.0,
        "leukocytosis_pct": 0,
    },
]


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _make_cohort(gene_data, seed):
    rng = _rng(seed)
    n = 40
    g = gene_data
    base_age = max(0.1, g["onset_yrs_mean"])

    onset_ages = sorted([max(0.0, rng.gauss(base_age, max(0.1, base_age * 0.4))) for _ in range(n)])
    dx_lag = [max(0.1, rng.gauss(g.get("dx_lag_yrs", 1.5), max(0.2, g.get("dx_lag_yrs", 1.5) * 0.3))) for _ in range(n)]
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
        "mean_onset_age": round(sum(onset_ages) / n, 3),
        "median_onset_age": round(sorted(onset_ages)[n // 2], 3),
        "mean_dx_age": round(sum(dx_ages) / n, 3),
        "mean_dx_lag_yrs": round(sum(dx_lag) / n, 3),
        "nbt_abnormal_n": pct_n("nbt_abnormal_pct"),
        "dhr_abnormal_n": pct_n("dhr_abnormal_pct"),
        "aspergillus_n": pct_n("aspergillus_risk_pct"),
        "staph_aureus_n": pct_n("staph_aureus_risk_pct"),
        "ibd_n": pct_n("ibd_pct"),
        "ifn_gamma_benefit_n": pct_n("ifn_gamma_benefit_pct"),
        "hsct_curative_n": pct_n("hsct_curative_pct"),
        "gene_therapy_eligible_n": pct_n("gene_therapy_eligible_pct"),
        "leukocytosis_n": pct_n("leukocytosis_pct"),
        "pathognomonic": g["pathognomonic"],
        "first_line_treatment": g["first_line_treatment"],
        "critical_rules": g["critical_rules"],
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    total_patients = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-CGD-Phagocyte-Defect-Atlas",
        "subtitle": (
            "Complete 8-Gene Chronic Granulomatous Disease & Phagocyte Defect Reference: "
            "CYBB/XL-CGD-gp91phox-65pct-All-CGD-IFN-gamma-FDA1990-First-Cytokine-Immunotherapy · "
            "CYBA/p22phox-AR-CGD-Obligate-Heterodimer-gp91-Absent-CYBB-Intact · "
            "NCF1/p47phox-Most-Common-AR-CGD-Pseudogene-MLPA-MANDATORY · "
            "NCF2/p67phox-Activation-Domain-No-Pseudogene · "
            "NCF4/p40phox-IBD-DOMINANT-Phagosomal-Partial-DHR · "
            "EROS/CYBC1-ER-Chaperone-gp91-Absent-CYBB-Intact-2019 · "
            "RAC2/RhoGTPase-AD-Leukocytosis-PATHOGNOMONIC · "
            "MPO/Most-Common-Phagocyte-Disorder-Clinically-Silent-Diabetes-Candida"
        ),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["name"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "cohort_size": 40,
        "cohorts": cohorts,
        "key_facts": {
            "cybb_most_common": "CYBB (gp91phox): most common CGD (65%); XL males only; DHR flat gold standard; IFN-γ (Actimmune) FDA1990 FIRST cytokine immunotherapy ever approved; Aspergillus 35-40%",
            "cyba_heterodimer": "CYBA (p22phox): AR — obligate heterodimer with gp91phox; CYBA LOF → gp91phox also absent despite intact CYBB; test anti-p22phox when gp91 absent + CYBB negative; Middle Eastern 540+1G→T founder",
            "ncf1_pseudogene": "NCF1 (p47phox): most common AR-CGD (25% of AR-CGD); NCF1B/NCF1C pseudogenes — 75-80% of alleles are GT deletion missed by Sanger/NGS; MLPA OR long-range PCR MANDATORY",
            "ncf2_activation": "NCF2 (p67phox): ~5% AR-CGD; activation domain for electron transfer; no pseudogene issue — standard sequencing reliable (unlike NCF1); IBD 25-30%",
            "ncf4_ibd_dominant": "NCF4 (p40phox): rarest CGD (<1%); IBD DOMINATES (75%) and often the PRESENTING feature; PARTIAL DHR (not flat zero); phagosomal-specific partial oxidase; mild infection susceptibility",
            "eros_er_chaperone": "EROS/CYBC1: discovered 2019; ultra-rare (<25 cases); ER transmembrane chaperone for gp91phox — CYBB gene intact but gp91phox misfolded and absent; clinically identical to XL-CGD",
            "rac2_leukocytosis": "RAC2 GTPase: AD dominant negative (D57N); LEUKOCYTOSIS (WBC 20-100×10^9/L) PATHOGNOMONIC — impaired neutrophil diapedesis; combined NADPH + migration + degranulation defect broader than CGD",
            "mpo_clinically_silent": "MPO: MOST COMMON phagocyte disorder (1:2000-4000); 99% clinically silent; DHR NORMAL + NBT NORMAL (distinguishes from all CGD); Diabetes + MPO deficiency = severe Candida PATHOGNOMONIC",
        },
        "diagnostic_approach": {
            "dhr_flat_nbt_negative": "DHR flat + NBT negative → CGD confirmed; check gp91phox by flow cytometry first → absent = CYBB (male/XL) or CYBA; present = NCF1/NCF2/NCF4/EROS/RAC2",
            "gp91_absent_cybb_negative": "gp91phox absent by flow + CYBB sequencing negative → order CYBA (p22phox flow + sequencing); if CYBA also negative → EROS/CYBC1 panel",
            "gp91_present_dhr_flat": "gp91phox present by flow + DHR flat → cytosolic subunit defect; test NCF1 (MLPA MANDATORY), NCF2 (standard sequencing), NCF4 (partial DHR clue)",
            "ncf1_sequencing_negative": "AR-CGD + p47phox absent + routine sequencing negative → MLPA MANDATORY (NCF1B/NCF1C pseudogene GT deletion in 75-80% of alleles missed by Sanger)",
            "partial_dhr_ibd_dominant": "Partial DHR (not flat zero) + dominant IBD/colitis phenotype + mild infections → NCF4 (p40phox); phagosomal-specific oxidase preserved at plasma membrane",
            "leukocytosis_cgd_like": "CGD-like infections + LEUKOCYTOSIS (WBC elevated 20-100×10^9/L) + poor wound healing + neutrophil migration defect → RAC2 dominant negative (D57N)",
            "dhr_normal_nbt_normal": "DHR NORMAL + NBT NORMAL + no oxidative burst defect = NOT CGD; if peroxidase absent on automated CBC → MPO deficiency; if diabetic → risk of severe Candida",
            "mpo_incidental": "Sysmex/automated CBC flags zero peroxidase → MPO deficiency; confirm with flow cytometry MPO stain; check for type 2 diabetes (unmasks clinical disease); no treatment if asymptomatic",
        },
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    genes_out = []
    for i, (g, c) in enumerate(zip(ATLAS_GENES, cohorts)):
        genes_out.append({
            "rank": i + 1,
            "gene": g["name"],
            "full_name": g["full_name"],
            "aa": g["aa"],
            "mw_kda": g["mw_kda"],
            "locus": g["locus"],
            "inheritance": g["inheritance"],
            "mechanism": g["mechanism"],
            "syndrome": g["syndrome"],
            "seed": SEED_BASE + i,
            "n": c["n"],
            "mean_onset_age": c["mean_onset_age"],
            "mean_dx_lag_yrs": c["mean_dx_lag_yrs"],
            "nbt_abnormal_n": c["nbt_abnormal_n"],
            "dhr_abnormal_n": c["dhr_abnormal_n"],
            "aspergillus_n": c["aspergillus_n"],
            "staph_aureus_n": c["staph_aureus_n"],
            "ibd_n": c["ibd_n"],
            "ifn_gamma_benefit_n": c["ifn_gamma_benefit_n"],
            "hsct_curative_n": c["hsct_curative_n"],
            "gene_therapy_eligible_n": c["gene_therapy_eligible_n"],
            "leukocytosis_n": c["leukocytosis_n"],
            "critical_rules": g["critical_rules"],
            "pathognomonic": g["pathognomonic"],
            "first_line_treatment": g["first_line_treatment"],
        })
    return {"genes": genes_out}


def generate_definitions():
    return {
        "atlas": "Hereditary-CGD-Phagocyte-Defect-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": {
            "CGD Overview — NADPH Oxidase, NBT Test, DHR Test": (
                "Chronic Granulomatous Disease (CGD) is a primary immunodeficiency caused by defects in the "
                "NADPH oxidase enzyme complex, which generates superoxide (O2−) in phagocytes during the "
                "respiratory burst. Superoxide is the precursor of all reactive oxygen species (ROS) used to "
                "kill catalase-positive microorganisms. Affected patients suffer recurrent life-threatening "
                "infections (Aspergillus fumigatus, Staphylococcus aureus, Burkholderia cepacia, Nocardia, "
                "Serratia) and sterile granuloma formation in any organ. "
                "NBT (Nitroblue Tetrazolium) test: neutrophils incubated with phorbol ester — normal cells "
                "reduce yellow NBT to blue formazan via superoxide. CGD = NEGATIVE NBT (no blue precipitate). "
                "DHR (Dihydrorhodamine 123) flow cytometry: rhodamine fluorescence shift after phorbol ester "
                "stimulation measures oxidative burst per cell. CGD = FLAT DHR histogram (no shift). "
                "DHR is the gold standard for diagnosis: quantifies residual oxidase activity, distinguishes "
                "complete CGD (flat) from partial defects (NCF4, RAC2), and identifies X-linked carriers "
                "(bimodal DHR: two populations). NADPH oxidase complex: membrane-bound flavocytochrome b558 "
                "(gp91phox/p22phox heterodimer) + cytosolic components (p47phox/p67phox/p40phox/RAC2). "
                "All five subunits required for functional assembly and electron transfer."
            ),
            "CYBB / XL-CGD — gp91phox, IFN-γ FDA1990, Aspergillus": (
                "CYBB encodes gp91phox (NOX2), the large catalytic beta subunit of flavocytochrome b558 and "
                "the electron-transporting component of the NADPH oxidase. It contains two haem groups, one "
                "FAD, and one NADPH binding site. XL-CGD (CYBB LOF) is the most common form of CGD, "
                "accounting for 65% of all cases worldwide. X-linked: only males affected (hemizygous); "
                "female carriers have bimodal DHR (mosaic oxidase activity by lyonization). "
                "Clinical hallmarks: recurrent Aspergillus fumigatus pneumonia (35-40% lifetime risk without "
                "prophylaxis), liver and perihepatic abscesses (Staphylococcus aureus), "
                "Burkholderia cepacia sepsis (high mortality), Nocardia brain abscess, Serratia infections. "
                "GRANULOMATOUS IBD: inflammatory bowel disease (Crohn's-like) in 30-40% of XL-CGD. "
                "IFN-γ (Actimmune, interferon gamma-1b, 50 mcg/m2 TIW SC): FDA approved 1990 — "
                "the FIRST cytokine approved as immunotherapy for any disease. Reduces serious infections by "
                "~70% in XL-CGD via upregulation of residual NADPH oxidase activity and non-oxidative pathways. "
                "Prophylaxis: TMP-SMX (antibacterial) + itraconazole or voriconazole (antifungal) mandatory. "
                "HSCT curative: 85-90% survival with HLA-matched donor. Lentiviral CYBB gene therapy in trials."
            ),
            "CYBA / AR-CGD Type 2 — p22phox, Obligate Heterodimer, Middle Eastern Founder": (
                "CYBA encodes p22phox, the alpha subunit (small subunit) of flavocytochrome b558. "
                "p22phox forms an OBLIGATE heterodimer with gp91phox: both subunits must be present for "
                "stability and membrane targeting. CYBA LOF → p22phox absent → gp91phox is also unstable "
                "and degraded in the ER, resulting in ABSENT gp91phox by flow cytometry DESPITE INTACT CYBB. "
                "This is a critical diagnostic pitfall: gp91phox absent + CYBB sequencing negative → "
                "the clinician must test CYBA (p22phox), not assume CYBB is the cause. "
                "AR inheritance: affects both male and female children equally (unlike CYBB). "
                "~5% of all CGD. Clinical severity similar to XL-CGD. "
                "Founder variant: 540+1G→T splice site mutation enriched in Middle Eastern populations "
                "(Saudi Arabia, Lebanon, Iran, Turkey). IBD in ~30%. "
                "DHR flat, NBT negative (same pattern as XL-CGD). "
                "Diagnostic flow: gp91phox absent by flow → CYBB sequencing negative → "
                "anti-p22phox flow cytometry + CYBA sequencing → if both absent, confirm CYBA LOF. "
                "HSCT curative at 85%; IFN-γ benefit modest in AR-CGD compared with XL-CGD."
            ),
            "NCF1 / p47phox — Most Common AR-CGD, Pseudogene MLPA Mandatory": (
                "NCF1 encodes p47phox, a cytosolic organiser subunit that undergoes phosphorylation-driven "
                "conformational change to assemble the NADPH oxidase at the plasma membrane. "
                "NCF1 is the most common AR-CGD gene, accounting for ~25% of AR-CGD cases (~8-9% of all CGD). "
                "CRITICAL MOLECULAR PITFALL: NCF1 has two highly homologous pseudogenes, NCF1B and NCF1C, "
                "located on the same chromosome 7q11.23 locus. A GT deletion at the start of exon 2 "
                "(ΔGT) occurs by intrachromosomal recombination between NCF1 and its pseudogenes, "
                "accounting for 75-80% of disease alleles. This deletion is flanked by repetitive sequences "
                "and is INVISIBLE to standard Sanger sequencing and most NGS gene panels because the "
                "pseudogene sequences cause misalignment. "
                "MANDATORY: MLPA (Multiplex Ligation-dependent Probe Amplification) or long-range PCR "
                "specific for the NCF1 vs pseudogene junction MUST be ordered — NEVER rely on Sanger alone. "
                "Clinical phenotype: generally MILDER than XL-CGD — later onset (median 4-5 years), fewer "
                "Aspergillus infections (20% vs 35-40%), better survival. IBD (granulomatous colitis) "
                "occurs in ~40% of NCF1-CGD patients. Prophylaxis and IFN-γ as per XL-CGD."
            ),
            "NCF2 / p67phox — Activation Domain, No Pseudogene, Standard Sequencing": (
                "NCF2 encodes p67phox, which contains an activation domain (tetratricopeptide repeats) "
                "that interacts with RAC2-GTP and directly activates electron transfer by the flavocytochrome "
                "b558 complex. p67phox is an essential cytosolic organiser alongside p47phox. "
                "NCF2 deficiency accounts for ~5% of AR-CGD cases — rarer than NCF1. "
                "KEY ADVANTAGE over NCF1: NCF2 has NO pseudogenes — standard NGS/Sanger sequencing "
                "is fully reliable and sufficient for diagnosis. No MLPA required. "
                "Clinical severity similar to NCF1-CGD (milder than XL-CGD). "
                "IBD in approximately 25-30% of NCF2-CGD. Aspergillus risk intermediate (~22%). "
                "DHR flat, NBT negative, p67phox absent by cytosolic protein studies. "
                "Prophylaxis: TMP-SMX + antifungal as per CGD standard. IFN-γ modest benefit. "
                "HSCT curative. No gene therapy specifically developed."
            ),
            "NCF4 / p40phox — IBD Dominant, Partial DHR, Phagosomal Specific": (
                "NCF4 encodes p40phox, the most recently characterised cytosolic NADPH oxidase subunit. "
                "p40phox contains a PX domain that specifically binds phosphatidylinositol 3-phosphate (PI3P) "
                "enriched on the phagosomal membrane — its role is PHAGOSOMAL oxidase activation, not "
                "plasma membrane burst. "
                "UNIQUE CLINICAL PHENOTYPE: NCF4 deficiency presents with DOMINANT GRANULOMATOUS IBD "
                "(severe Crohn's-like colitis in ~75%) as the PRIMARY and PRESENTING feature. "
                "Catalase-positive organism infections are MILD compared with other CGD subtypes, "
                "because plasma membrane NADPH oxidase activity (gp91/p47/p67 complex) is PRESERVED — "
                "only phagosomal killing is selectively impaired. "
                "DIAGNOSTIC CLUE: PARTIAL DHR (not completely flat) — partial oxidative burst at the plasma "
                "membrane is retained. This distinguishes NCF4 from all other CGD forms. "
                "Pathognomonic scenario: severe treatment-refractory Crohn's-like IBD in child + partial "
                "DHR + mild infection phenotype → NCF4 sequencing. "
                "Rarest CGD gene (<1% of all CGD). Standard sequencing sufficient (no pseudogene)."
            ),
            "EROS / CYBC1 — ER Chaperone for gp91phox, 2019 Discovery, Ultra-Rare": (
                "EROS (Endoplasmic Reticulum Oxidoreductase 1 for gp91phox Stability), also designated CYBC1, "
                "is a transmembrane ER protein discovered as a cause of CGD in 2019 by Arnadottir et al. "
                "EROS acts as a dedicated maturation chaperone for gp91phox in the ER — it is required for "
                "correct folding of gp91phox before insertion into the phagosomal membrane. "
                "Without EROS: gp91phox is misfolded in the ER and targeted for proteasomal degradation → "
                "gp91phox absent from mature neutrophils → FULL CGD PHENOTYPE (DHR flat, NBT negative). "
                "CRITICAL PITFALL: gp91phox absent by flow cytometry + CYBB sequencing NEGATIVE + "
                "CYBA sequencing NEGATIVE → consider EROS/CYBC1 sequencing before calling the case 'genetic diagnosis unknown'. "
                "CYBB gene is INTACT in EROS deficiency — the gp91phox protein is absent due to ER quality "
                "control failure. Ultra-rare: fewer than 25 patients reported worldwide as of 2024. "
                "Clinical phenotype clinically identical to XL-CGD (Aspergillus, Staph, Burkholderia). "
                "Management: same prophylaxis + IFN-γ as CGD; HSCT curative."
            ),
            "RAC2 — RhoGTPase, Dominant Negative D57N, Leukocytosis Pathognomonic": (
                "RAC2 is a Rho-family small GTPase expressed exclusively in haematopoietic cells. "
                "RAC2-GTP is required for: (1) activating p67phox and assembling the NADPH oxidase; "
                "(2) actin cytoskeleton reorganisation for neutrophil migration and polarisation; "
                "(3) degranulation (azurophil and specific granule exocytosis). "
                "DOMINANT NEGATIVE RAC2 (D57N and similar gain-of-function/DN variants): the mutant RAC2 "
                "protein is constitutively inactive (GDP-locked) AND outcompetes wild-type RAC2 → "
                "combined defect affecting ALL three neutrophil functions simultaneously. "
                "PATHOGNOMONIC FINDING: LEUKOCYTOSIS — WBC elevated to 20-100 × 10^9/L — because "
                "impaired neutrophil adhesion and diapedesis traps neutrophils in the circulation. "
                "This is the DISTINGUISHING feature from pure CGD (which does not cause leukocytosis per se). "
                "Biallelic LOF RAC2 (AR form): isolated CGD-like phenotype without the leukocytosis signature. "
                "Clinical: CGD-like infections + poor wound healing (degranulation defect) + leukocytosis. "
                "HSCT curative for DN form. DHR variably impaired (partial for DN, flat for complete LOF)."
            ),
            "MPO / Myeloperoxidase — Most Common Phagocyte Disorder, Clinically Silent, Diabetes-Candida": (
                "Myeloperoxidase (MPO) is stored in azurophil (primary) granules of neutrophils and monocytes. "
                "After phagocytosis, MPO converts H2O2 (produced by NADPH oxidase) to hypochlorous acid "
                "(HOCl, bleach) — the most potent antimicrobial reactive species in the phagosome. "
                "MPO deficiency is the MOST COMMON hereditary phagocyte disorder: prevalence 1:2000-4000 "
                "live births (far more prevalent than all CGD combined). "
                "PARADOX: Despite absent MPO, 99% of MPO-deficient individuals are CLINICALLY SILENT. "
                "Reason: NADPH oxidase H2O2 itself is bactericidal + redundant killing mechanisms compensate. "
                "CRITICAL DISTINCTION FROM CGD: DHR NORMAL, NBT NORMAL — NADPH oxidase is fully intact. "
                "The oxidative burst generates H2O2 normally; MPO is absent but O2− production is preserved. "
                "Diagnosis: often INCIDENTAL on automated haematology analyser (Sysmex XN/XT series) which "
                "uses peroxidase channel — MPO-deficient cells appear as 'peroxidase-negative neutrophils'. "
                "PATHOGNOMONIC CLINICAL COMBINATION: MPO deficiency + TYPE 2 DIABETES → recurrent severe "
                "Candida (deep-seated mucocutaneous candidiasis, candidaemia). Diabetic hyperglycaemia "
                "impairs non-MPO killing (opsonisation, phagocyte migration), creating total dependence on "
                "MPO-HOCl killing that is absent. Management: tight glycaemic control; antifungal azole "
                "prophylaxis in diabetics; no HSCT indicated (too rare a clinical scenario, >99% asymptomatic)."
            ),
            "NBT vs DHR vs MPO Stain — Three-Test Phagocyte Function Matrix": (
                "NBT (Nitroblue Tetrazolium) test: PMA-stimulated neutrophils reduce yellow NBT to blue "
                "formazan via superoxide. NEGATIVE (no blue) = CGD. POSITIVE (blue) = normal. "
                "Distinguishes CGD from normal; does NOT distinguish MPO from normal (NBT normal in MPO). "
                "Limitations: qualitative, operator-dependent, cannot identify carriers. "
                "DHR 123 (Dihydrorhodamine) flow cytometry: DHR fluoresces green (FL1) after oxidation by "
                "H2O2/MPO within the phagosome. PMA stimulation → normal neutrophils shift right (high FL1). "
                "CGD = FLAT histogram (no shift). PARTIAL defects (NCF4, RAC2) = partial rightward shift. "
                "XL-CGD CARRIERS = BIMODAL histogram (two populations: normal and non-shifting). "
                "DHR gold standard: quantitative, identifies carriers, distinguishes partial from complete CGD. "
                "CRITICAL: DHR requires MPO for full signal amplification. MPO-deficient neutrophils "
                "show REDUCED DHR shift (but NOT flat) — this is a common misread; "
                "DHR NORMAL (not flat) + absent MPO stain = MPO deficiency, NOT CGD. "
                "MPO stain (Bauer peroxidase cytochemistry or flow cytometry anti-MPO): "
                "negative in MPO deficiency; normal in all forms of CGD. "
                "THREE-TEST MATRIX: "
                "NBT−, DHR flat, MPO+ → CGD (CYBB/CYBA/NCF1/NCF2/NCF4/EROS/RAC2). "
                "NBT+, DHR low-normal, MPO− → MPO deficiency. "
                "NBT+, DHR normal, MPO+ → normal phagocyte function."
            ),
            "HSCT and Gene Therapy for CGD — Indications, Timing, Conditioning": (
                "HSCT (haematopoietic stem cell transplantation) is curative for CGD. "
                "Indications: all XL-CGD (CYBB), CYBA, NCF1, NCF2, and EROS — consider early (before "
                "major infection burden). NCF4 and MPO: usually not indicated (NCF4 managed conservatively; "
                "MPO near-universally asymptomatic). "
                "Conditioning: myeloablative (busulfan-fludarabine) or reduced-intensity conditioning (RIC). "
                "Busulfan-based conditioning is SAFE in CGD (no radiosensitivity unlike DCLRE1C/Artemis-SCID). "
                "Best outcomes: transplanted before age 10 and before invasive fungal infection. "
                "Post-HSCT: mixed chimerism >10-20% gp91phox-positive neutrophils correlates with clinical "
                "protection (even partial engraftment is clinically beneficial). "
                "Gene therapy: lentiviral CYBB gene transfer — multicentre trials demonstrating 70%+ success; "
                "not yet broadly approved but available under compassionate use/trials. "
                "Prophylaxis during HSCT and beyond: TMP-SMX + voriconazole mandatory. "
                "IFN-γ: bridge therapy until HSCT; continue if mixed chimerism post-HSCT. "
                "RAC2 DN: HSCT indicated early; IFN-γ insufficient for combined defect."
            ),
        },
        "standards": [
            "CGD European Experience — Seger RA et al. Lancet 2002",
            "NADPH oxidase complex structure — Bedard K & Krause KH Physiol Rev 2007",
            "IFN-γ CGD prophylaxis FDA approval — INTERNATIONAL CGD COOPERATIVE STUDY NEJM 1991",
            "NCF1 pseudogene GT deletion — Goedken M et al. J Leukoc Biol 2009",
            "EROS/CYBC1 discovery — Arnadottir GA et al. Nat Immunol 2019",
            "RAC2 dominant negative D57N — Ambruso DR et al. PNAS 2000",
            "MPO deficiency review — Heyworth PG et al. Curr Opin Hematol 2003",
            "CGD gene therapy lentiviral — Kang EM et al. Blood 2011",
            "DHR gold standard for CGD — Vowells SJ et al. J Immunol Methods 1995",
            "Primary immunodeficiency classification IUIS 2022",
            "CYBA p22phox heterodimer — Cross AR et al. Biochim Biophys Acta 1994",
            "NCF4 p40phox IBD — Matute JD et al. Nat Med 2009",
        ],
    }
