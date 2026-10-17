#!/usr/bin/env python3
"""Hereditary-Complement-System-Deficiency-Atlas -- Complete 8-Gene Reference
Complement pathway deficiencies: classical / alternative / terminal / regulation
(C1QA/C1q-deficiency-SLE-PATHOGNOMONIC-most-severe-autoimmunity,
 C2/most-common-complement-deficiency-SLE-like-encapsulated-bacteria,
 C3/central-pivot-recurrent-encapsulated-bacteria-MPGN-type-II,
 CFB/alternative-pathway-AR-rare-bacterial-infections,
 CFP/properdin-XL-meningococcal-7000x-risk-HIGHEST,
 C7/terminal-MAC-neisserial-meningococcal-recurrent,
 CFH/aHUS-MPGN2-AMD-eculizumab-FDA2007-FIRST-complement-therapy,
 CD46/MCP-aHUS-incomplete-penetrance-eculizumab-responsive):

C1QA    (Complement C1q subunit A; 245aa; 1p36.11; AR;
          C1q deficiency — most severe autoimmune phenotype of all complement deficiencies;
          C1q is the recognition subunit of C1 (C1qr2s2 complex), opsonises apoptotic debris;
          Without C1q: apoptotic cell clearance defective → nuclear self-antigen presentation → SLE;
          93% of C1q-deficient patients develop SLE or SLE-like disease — HIGHEST autoimmunity;
          Photosensitive rash + renal disease + anti-dsDNA positive — clinically indistinguishable from SLE;
          Recurrent bacterial infections (encapsulated organisms);
          C1q protein assay diagnostic; no specific complement replacement therapy; hydroxychloroquine/immunosuppression;
          seed SEED_BASE+0).
C2      (Complement C2; 898aa; 6p21.32; AR;
          Most common complement deficiency in European populations (1:10,000 heterozygotes);
          Biallelic LOF → absent classical pathway activation (C4 intact via lectin pathway);
          28-bp deletion (Type I) accounts for 95% of C2 deficiency alleles in Europeans;
          50% of C2-deficient patients develop SLE-like disease — autoimmunity predominates;
          Recurrent Streptococcus pneumoniae + Haemophilus infections;
          Classical pathway absent → CH50 = 0; AH50 normal (alternative pathway intact);
          Pneumococcal + meningococcal vaccines MANDATORY; penicillin prophylaxis in childhood;
          seed SEED_BASE+1).
C3      (Complement C3; 1663aa; 19p13.3; AR;
          C3 is the central pivot of complement — all three pathways converge on C3;
          C3 deficiency = most severe susceptibility to encapsulated bacterial infections;
          Recurrent severe pneumococcal/meningococcal/Haemophilus bacteraemia starting in infancy;
          MPGN type II (dense deposit disease, DDD): C3 nephritic factor → unregulated C3 activation in GBM;
          CH50 = 0 AND AH50 = 0 — both classical and alternative pathways absent;
          Low C3 + normal C4 during non-acute phase = C3 breakdown not classical pathway;
          Plasma infusions (fresh frozen plasma — contains C3) bridge before specific therapy;
          seed SEED_BASE+2).
CFB     (Complement Factor B; 764aa; 6p21.32; AR;
          Factor B = serine protease of the alternative pathway C3 convertase (C3bBb);
          AR LOF → absent alternative pathway amplification;
          Rare: <50 families reported worldwide; severe susceptibility to encapsulated bacteria;
          AH50 = 0; CH50 normal or mildly reduced (classical pathway uses factor B minimally);
          Also: rare GOF CFB variants cause C3 glomerulopathy/MPGN via hyperactivated AP;
          LOF patients: Streptococcus pneumoniae + Neisseria meningitidis infections predominant;
          Vaccines MANDATORY; prophylactic antibiotics;
          seed SEED_BASE+3).
CFP     (Complement Factor P / Properdin; 442aa; Xp11.23; XL;
          Properdin is the ONLY positive regulator of complement — stabilises alternative pathway C3 convertase;
          X-linked inheritance → males almost exclusively affected (hemizygous LOF);
          Type I: null allele (most severe); Type II: low properdin (<50%); Type III: dysfunctional;
          Meningococcal disease risk 7000x elevated — HIGHEST meningococcal susceptibility of any complement deficiency;
          Recurrent fulminant Neisseria meningitidis PATHOGNOMONIC for properdin deficiency;
          AH50 reduced or absent; CH50 normal (classical pathway intact);
          Meningococcal conjugate ACWY + serogroup B vaccines MANDATORY; lifelong penicillin prophylaxis;
          seed SEED_BASE+4).
C7      (Complement C7; 843aa; 5p13.1; AR;
          Terminal complement component: C5b-6-7-8-9 = membrane attack complex (MAC);
          C7 LOF → MAC cannot assemble → gram-negative lysis defect;
          Recurrent Neisseria meningitidis and gonorrhoea bacteraemia — neisserial tropism;
          Most common terminal complement deficiency in Japan and South Africa (founder effects);
          CH50 = 0 (total haemolytic complement absent); AH50 = 0; C3 normal; C4 normal;
          All terminal complement deficiencies (C5-C9): neisserial tropism is SHARED feature;
          Meningococcal ACWY + B vaccines MANDATORY; penicillin/ciprofloxacin prophylaxis; avoid eculizumab (blocks MAC — worsens infections);
          seed SEED_BASE+5).
CFH     (Complement Factor H; 1231aa; 1q31.3; AD/AR;
          Factor H is the primary regulator of the alternative pathway — protects host cells from self-attack;
          Heterozygous LOF → aHUS (atypical haemolytic uraemic syndrome) — thrombotic microangiopathy;
          aHUS TRIAD: microangiopathic haemolytic anaemia + thrombocytopenia + acute kidney injury;
          Biallelic null CFH → C3 glomerulopathy + MPGN type II (Dense Deposit Disease, DDD);
          AMD (age-related macular degeneration): CFH Y402H polymorphism — 2-3x increased AMD risk;
          Eculizumab (anti-C5 monoclonal, FDA 2007 — FIRST complement-directed therapy ever approved) TRANSFORMS aHUS prognosis;
          Ravulizumab (long-acting anti-C5, FDA 2019) — 8-weekly dosing preferred;
          Prophylactic meningococcal vaccination MANDATORY before eculizumab (anti-C5 blocks MAC);
          seed SEED_BASE+6).
CD46    (Membrane Cofactor Protein / MCP; 347aa; 1q32.2; AD;
          MCP is a membrane-bound complement regulator on all nucleated cells except red blood cells;
          AD LOF → aHUS susceptibility (incomplete penetrance ~50%);
          Milder aHUS phenotype than CFH: often paediatric onset, shorter TMA episodes, better renal prognosis;
          Incomplete penetrance: many carriers have recurrent aHUS only after triggers (pregnancy/infection);
          MCP loss → endothelium and glomerular cells cannot regulate complement → local TMA;
          eculizumab/ravulizumab responsive: some patients can discontinue after initial TMA resolution (unlike CFH-aHUS);
          ADAMTS13 activity > 10% + negative Shiga toxin stool PCR + complement genetic panel = diagnostic workup;
          seed SEED_BASE+7).
"""
import random
import math

SEED_BASE = 3846

ATLAS_GENES = [
    {
        "name": "C1QA",
        "full_name": "Complement C1q subunit A",
        "aa": 245,
        "locus": "1p36.11",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "C1q Deficiency / SLE-like",
        "mw_kda": 26.7,
        "critical_rules": [
            "C1QA-SLE-93PCT-HIGHEST-AUTOIMMUNITY-ALL-COMPLEMENT-DEFICIENCIES",
            "C1QA-APOPTOTIC-CELL-CLEARANCE-DEFECTIVE-SELF-ANTIGEN-PRESENTATION",
            "C1QA-CH50-ABSENT-AH50-NORMAL-CLASSICAL-PATHWAY-ONLY",
            "C1QA-NO-COMPLEMENT-REPLACEMENT-HYDROXYCHLOROQUINE-IMMUNOSUPPRESSION",
        ],
        "distinguishing_feature": "93% SLE-like disease — highest autoimmunity; C1q assay diagnostic; no specific replacement therapy",
        "first_line_treatment": "Hydroxychloroquine + immunosuppression (AZA/MMF) for SLE phenotype; infection prophylaxis",
        "pathognomonic": "SLE-like disease (photosensitive rash + nephritis + anti-dsDNA) + C1q protein undetectable + very early onset",
        "onset_yrs_mean": 5,
        "penetrance_pct": 93,
        "sle_pct": 93,
        "bacterial_infection_pct": 65,
        "renal_involvement_pct": 60,
        "photosensitive_rash_pct": 70,
        "ch50_absent_pct": 100,
        "ah50_normal_pct": 100,
        "complement_replacement_pct": 0,
        "meningococcal_pct": 25,
        "ahus_pct": 0,
        "mpgn_pct": 10,
        "stroke_pct": 5,
        "dx_lag_yrs": 3.0,
    },
    {
        "name": "C2",
        "full_name": "Complement C2",
        "aa": 898,
        "locus": "6p21.32",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "C2 Deficiency / SLE-like + Bacterial Infections",
        "mw_kda": 102.6,
        "critical_rules": [
            "C2-MOST-COMMON-COMPLEMENT-DEFICIENCY-EUROPEANS-1-10000",
            "C2-28BP-DELETION-TYPE1-95PCT-ALLELES",
            "C2-CH50-ABSENT-AH50-NORMAL-CLASSICAL-PATHWAY-DEFECT",
            "C2-PNEUMOCOCCAL-MENINGOCOCCAL-VACCINES-MANDATORY",
        ],
        "distinguishing_feature": "Most common complement deficiency; 28-bp deletion 95% of European alleles; SLE-like 50%; CH50=0 AH50=normal",
        "first_line_treatment": "Pneumococcal + meningococcal conjugate vaccines; penicillin prophylaxis childhood; immunosuppression for SLE",
        "pathognomonic": "SLE-like disease + recurrent pneumococcal bacteraemia + CH50=0 with normal AH50 in European patient",
        "onset_yrs_mean": 8,
        "penetrance_pct": 80,
        "sle_pct": 50,
        "bacterial_infection_pct": 75,
        "renal_involvement_pct": 30,
        "photosensitive_rash_pct": 45,
        "ch50_absent_pct": 100,
        "ah50_normal_pct": 100,
        "complement_replacement_pct": 0,
        "meningococcal_pct": 35,
        "ahus_pct": 0,
        "mpgn_pct": 5,
        "stroke_pct": 2,
        "dx_lag_yrs": 4.5,
    },
    {
        "name": "C3",
        "full_name": "Complement C3",
        "aa": 1663,
        "locus": "19p13.3",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "C3 Deficiency / Recurrent Encapsulated Bacteria + MPGN-II",
        "mw_kda": 187.1,
        "critical_rules": [
            "C3-CENTRAL-PIVOT-ALL-THREE-PATHWAYS-CONVERGE",
            "C3-CH50-ABSENT-AH50-ABSENT-BOTH-PATHWAYS-LOST",
            "C3-MPGN-TYPE-II-DDD-C3-NEPHRITIC-FACTOR",
            "C3-ENCAPSULATED-BACTERIA-MOST-SEVERE-SUSCEPTIBILITY",
        ],
        "distinguishing_feature": "CH50=0 AND AH50=0 (both pathways absent); most severe bacterial susceptibility; MPGN-II/DDD",
        "first_line_treatment": "All vaccines (pneumococcal, meningococcal ACWY+B, Hib); lifelong penicillin prophylaxis; FFP for acute infections; no specific replacement",
        "pathognomonic": "Recurrent invasive encapsulated bacterial infections from infancy + CH50=0 AND AH50=0 + low C3 with normal C4",
        "onset_yrs_mean": 2,
        "penetrance_pct": 100,
        "sle_pct": 20,
        "bacterial_infection_pct": 100,
        "renal_involvement_pct": 40,
        "photosensitive_rash_pct": 15,
        "ch50_absent_pct": 100,
        "ah50_normal_pct": 0,
        "complement_replacement_pct": 10,
        "meningococcal_pct": 70,
        "ahus_pct": 5,
        "mpgn_pct": 40,
        "stroke_pct": 3,
        "dx_lag_yrs": 1.5,
    },
    {
        "name": "CFB",
        "full_name": "Complement Factor B",
        "aa": 764,
        "locus": "6p21.32",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "Factor B Deficiency / Alternative Pathway AP Absent",
        "mw_kda": 86.9,
        "critical_rules": [
            "CFB-AH50-ABSENT-CH50-NORMAL-ALTERNATIVE-PATHWAY-DEFECT",
            "CFB-LOF-AR-RARE-LT50-FAMILIES-WORLDWIDE",
            "CFB-GOF-CAUSES-C3-GLOMERULOPATHY-MPGN-OPPOSITE-PHENOTYPE",
            "CFB-ENCAPSULATED-BACTERIA-PNEUMOCOCCAL-MENINGOCOCCAL-PREDOMINANT",
        ],
        "distinguishing_feature": "AH50=0, CH50=normal (alternative pathway specifically absent); GOF variants cause MPGN (opposite of LOF); <50 families worldwide",
        "first_line_treatment": "Vaccines mandatory (pneumococcal, meningococcal); prophylactic antibiotics; no specific complement therapy",
        "pathognomonic": "AH50=0 with normal CH50 in patient with recurrent bacterial infections → test CFB gene",
        "onset_yrs_mean": 3,
        "penetrance_pct": 90,
        "sle_pct": 10,
        "bacterial_infection_pct": 90,
        "renal_involvement_pct": 15,
        "photosensitive_rash_pct": 8,
        "ch50_absent_pct": 0,
        "ah50_normal_pct": 0,
        "complement_replacement_pct": 0,
        "meningococcal_pct": 55,
        "ahus_pct": 0,
        "mpgn_pct": 5,
        "stroke_pct": 1,
        "dx_lag_yrs": 5.0,
    },
    {
        "name": "CFP",
        "full_name": "Complement Factor P / Properdin",
        "aa": 442,
        "locus": "Xp11.23",
        "inheritance": "XL",
        "mechanism": "LOF",
        "syndrome": "Properdin Deficiency / Meningococcal 7000x Risk",
        "mw_kda": 53.1,
        "critical_rules": [
            "CFP-MENINGOCOCCAL-7000X-RISK-HIGHEST-OF-ALL-COMPLEMENT-DEFICIENCIES",
            "CFP-X-LINKED-MALES-HEMIZYGOUS-AFFECTED-FEMALES-CARRIERS",
            "CFP-AH50-REDUCED-ABSENT-CH50-NORMAL-ALTERNATIVE-PATHWAY",
            "CFP-MENINGOCOCCAL-ACWY-PLUS-B-VACCINES-LIFELONG-PENICILLIN-MANDATORY",
        ],
        "distinguishing_feature": "7000× meningococcal risk; X-linked; AH50 reduced/absent, CH50 normal; properdin ONLY positive regulator of complement",
        "first_line_treatment": "Meningococcal conjugate ACWY + serogroup B vaccines + lifelong penicillin V prophylaxis (ALL males with properdin deficiency)",
        "pathognomonic": "Recurrent fulminant Neisseria meningitidis in male + family history of meningococcal deaths + XL inheritance pattern",
        "onset_yrs_mean": 12,
        "penetrance_pct": 95,
        "sle_pct": 5,
        "bacterial_infection_pct": 95,
        "renal_involvement_pct": 5,
        "photosensitive_rash_pct": 3,
        "ch50_absent_pct": 0,
        "ah50_normal_pct": 0,
        "complement_replacement_pct": 0,
        "meningococcal_pct": 95,
        "ahus_pct": 0,
        "mpgn_pct": 0,
        "stroke_pct": 2,
        "dx_lag_yrs": 6.0,
    },
    {
        "name": "C7",
        "full_name": "Complement C7",
        "aa": 843,
        "locus": "5p13.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "C7 Deficiency / Terminal MAC Absent / Neisserial Tropism",
        "mw_kda": 92.0,
        "critical_rules": [
            "C7-CH50-ABSENT-AH50-ABSENT-BOTH-ZERO-C3-NORMAL-C4-NORMAL-TERMINAL-PATHWAY",
            "C7-NEISSERIAL-TROPISM-MENINGOCOCCAL-GONOCOCCAL-RECURRENT",
            "C7-JAPAN-SOUTH-AFRICA-FOUNDER-EFFECT-MOST-COMMON-TERMINAL-DEFICIENCY",
            "C7-ECULIZUMAB-CI-BLOCKS-MAC-WORSENS-INFECTIONS",
        ],
        "distinguishing_feature": "CH50=0, AH50=0, C3 normal, C4 normal (terminal MAC specifically absent); Japan/South Africa founder alleles; eculizumab CONTRAINDICATED",
        "first_line_treatment": "Meningococcal vaccines (ACWY + B); penicillin/ciprofloxacin prophylaxis; NO eculizumab (would worsen infections by blocking MAC)",
        "pathognomonic": "Recurrent Neisseria meningitidis ± Neisseria gonorrhoea + CH50=0, AH50=0 with normal C3 and C4",
        "onset_yrs_mean": 10,
        "penetrance_pct": 85,
        "sle_pct": 8,
        "bacterial_infection_pct": 90,
        "renal_involvement_pct": 5,
        "photosensitive_rash_pct": 5,
        "ch50_absent_pct": 100,
        "ah50_normal_pct": 0,
        "complement_replacement_pct": 0,
        "meningococcal_pct": 90,
        "ahus_pct": 0,
        "mpgn_pct": 2,
        "stroke_pct": 2,
        "dx_lag_yrs": 7.0,
    },
    {
        "name": "CFH",
        "full_name": "Complement Factor H",
        "aa": 1231,
        "locus": "1q31.3",
        "inheritance": "AD",
        "mechanism": "LOF",
        "syndrome": "aHUS / MPGN-II / AMD",
        "mw_kda": 139.1,
        "critical_rules": [
            "CFH-AHUS-TRIAD-MAHA-THROMBOCYTOPENIA-AKI-PATHOGNOMONIC",
            "CFH-ECULIZUMAB-FDA2007-FIRST-COMPLEMENT-THERAPY-EVER",
            "CFH-MENINGOCOCCAL-VACCINE-MANDATORY-BEFORE-ECULIZUMAB",
            "CFH-Y402H-SNP-AMD-2-3X-RISK-COMMON-POPULATION-VARIANT",
        ],
        "distinguishing_feature": "aHUS TMA + first complement therapy approved (eculizumab 2007); Y402H AMD risk SNP; meningococcal vaccine before eculizumab MANDATORY",
        "first_line_treatment": "Eculizumab 900mg IV weekly ×4 then 1200mg q2w (or ravulizumab 2400-3000mg q8w) — transforms aHUS prognosis; Meningococcal ACWY+B vaccines ≥2wk before eculizumab",
        "pathognomonic": "Microangiopathic haemolytic anaemia + thrombocytopenia + AKI + normal ADAMTS13 + STEC-negative + CFH variant",
        "onset_yrs_mean": 6,
        "penetrance_pct": 50,
        "sle_pct": 5,
        "bacterial_infection_pct": 20,
        "renal_involvement_pct": 95,
        "photosensitive_rash_pct": 5,
        "ch50_absent_pct": 20,
        "ah50_normal_pct": 0,
        "complement_replacement_pct": 5,
        "meningococcal_pct": 15,
        "ahus_pct": 95,
        "mpgn_pct": 30,
        "stroke_pct": 25,
        "dx_lag_yrs": 2.0,
    },
    {
        "name": "CD46",
        "full_name": "Complement Regulatory Protein MCP / CD46",
        "aa": 347,
        "locus": "1q32.2",
        "inheritance": "AD",
        "mechanism": "LOF",
        "syndrome": "aHUS / Incomplete Penetrance / Trigger-Dependent",
        "mw_kda": 35.5,
        "critical_rules": [
            "CD46-AHUS-INCOMPLETE-PENETRANCE-50PCT-TRIGGER-DEPENDENT",
            "CD46-ECULIZUMAB-RESPONSIVE-SOME-DISCONTINUE-UNLIKE-CFH",
            "CD46-MCP-ABSENT-ON-LEUKOCYTES-DIAGNOSTIC-BY-FLOW-CYTOMETRY",
            "CD46-ADAMTS13-GT10PCT-STEC-NEGATIVE-COMPLEMENT-PANEL-WORKUP",
        ],
        "distinguishing_feature": "Incomplete penetrance ~50%; triggers (pregnancy/infection) precipitate aHUS; MCP absent on leukocytes (flow cytometry diagnostic); some can discontinue eculizumab",
        "first_line_treatment": "Eculizumab/ravulizumab for acute TMA; assess MCP expression by flow cytometry; some patients discontinue after TMA resolution (unlike CFH-aHUS)",
        "pathognomonic": "Recurrent aHUS episodes with triggers + normal periods + absent MCP on leukocytes by flow cytometry + AD family history",
        "onset_yrs_mean": 18,
        "penetrance_pct": 50,
        "sle_pct": 3,
        "bacterial_infection_pct": 15,
        "renal_involvement_pct": 90,
        "photosensitive_rash_pct": 3,
        "ch50_absent_pct": 10,
        "ah50_normal_pct": 50,
        "complement_replacement_pct": 0,
        "meningococcal_pct": 5,
        "ahus_pct": 90,
        "mpgn_pct": 5,
        "stroke_pct": 20,
        "dx_lag_yrs": 3.0,
    },
]


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _make_cohort(gene_data, seed):
    rng = _rng(seed)
    n = 40
    g = gene_data
    base_age = max(0.5, g["onset_yrs_mean"])

    onset_ages = sorted([max(0.0, rng.gauss(base_age, max(0.3, base_age * 0.4))) for _ in range(n)])
    dx_lag = [max(0.5, rng.gauss(g.get("dx_lag_yrs", 4.0), 2.0)) for _ in range(n)]
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
        "sle_n": pct_n("sle_pct"),
        "bacterial_infection_n": pct_n("bacterial_infection_pct"),
        "renal_involvement_n": pct_n("renal_involvement_pct"),
        "photosensitive_rash_n": pct_n("photosensitive_rash_pct"),
        "ch50_absent_n": pct_n("ch50_absent_pct"),
        "meningococcal_n": pct_n("meningococcal_pct"),
        "ahus_n": pct_n("ahus_pct"),
        "mpgn_n": pct_n("mpgn_pct"),
        "stroke_n": pct_n("stroke_pct"),
        "complement_replacement_n": pct_n("complement_replacement_pct"),
        "pathognomonic": g["pathognomonic"],
        "first_line_treatment": g["first_line_treatment"],
        "critical_rules": g["critical_rules"],
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    total_patients = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Complement-System-Deficiency-Atlas",
        "subtitle": (
            "Complete 8-Gene Complement Pathway Reference: C1QA/C1q-SLE-93pct · C2/Most-Common-SLE-European · "
            "C3/Central-Pivot-Encapsulated-Bacteria · CFB/AP-Absent · CFP/Properdin-Meningococcal-7000x · "
            "C7/Terminal-MAC-Neisserial · CFH/aHUS-Eculizumab-FDA2007 · CD46/MCP-aHUS-Incomplete-Penetrance"
        ),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["name"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "cohort_size": 40,
        "cohorts": cohorts,
        "key_facts": {
            "c1q_autoimmunity": "C1QA/C1q deficiency: 93% develop SLE-like disease — HIGHEST autoimmunity of all complement deficiencies",
            "c2_most_common": "C2 deficiency: most common complement deficiency in Europeans (1:10,000); 28-bp deletion = 95% of alleles",
            "c3_central": "C3 deficiency: CH50=0 AND AH50=0; most severe bacterial susceptibility; all pathways use C3",
            "cfp_meningococcal": "Properdin/CFP deficiency: 7000x meningococcal risk; X-linked (males affected); only positive complement regulator",
            "c7_neisserial": "C7 deficiency: CH50=0, AH50=0, C3 normal, C4 normal (terminal pathway); neisserial tropism; eculizumab CONTRAINDICATED",
            "cfh_ahus": "CFH/Factor H: aHUS TMA; eculizumab FDA 2007 = FIRST complement-directed therapy; meningococcal vaccines before eculizumab MANDATORY",
            "cd46_triggers": "CD46/MCP: aHUS with incomplete penetrance (50%); triggers include pregnancy/infection; flow cytometry diagnostic",
            "vaccines_all": "ALL complement-deficient patients: pneumococcal + meningococcal ACWY + serogroup B + Hib vaccines MANDATORY",
        },
        "diagnostic_approach": {
            "sle_like_early_onset": "Test C1q protein → C1QA/C1R/C1S gene panel; if C1q normal → C4/C2 (classical pathway); CH50=0, AH50=normal",
            "ch50_zero_ah50_normal": "Classical pathway defect: C1q/C1r/C1s → C4 → C2 pathway; autoimmune phenotype predominates",
            "ch50_zero_ah50_zero": "All pathways absent: C3 deficiency most likely (central pivot); confirm low C3 + normal C4",
            "ah50_zero_ch50_normal": "Alternative pathway defect: CFP (properdin, XL) or CFB; male patient with meningococcal disease → properdin first",
            "ch50_zero_ah50_zero_c3_normal": "Terminal pathway defect: C5-C9; neisserial tropism; Japan/South Africa ancestry → C7/C9 founders",
            "ahus_triad": "MAHA + thrombocytopenia + AKI + ADAMTS13 >10% + STEC negative → complement genetic panel (CFH, CD46, CFI, C3, CFB, THBD)",
            "recurrent_meningococcal_male": "Properdin deficiency (XL); measure AH50; family history; meningococcal vaccine + penicillin prophylaxis",
            "amd_cfi_cfh": "CFH Y402H SNP (common); no aHUS treatment indicated for AMD alone; anti-VEGF ± complement inhibitor trials",
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
            "sle_n": c["sle_n"],
            "bacterial_infection_n": c["bacterial_infection_n"],
            "renal_involvement_n": c["renal_involvement_n"],
            "ahus_n": c["ahus_n"],
            "meningococcal_n": c["meningococcal_n"],
            "mpgn_n": c["mpgn_n"],
            "stroke_n": c["stroke_n"],
            "critical_rules": g["critical_rules"],
            "pathognomonic": g["pathognomonic"],
            "first_line_treatment": g["first_line_treatment"],
        })
    return {"genes": genes_out}


def generate_definitions():
    return {
        "atlas": "Hereditary-Complement-System-Deficiency-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": {
            "Complement System Overview": (
                "The complement system is a cascade of ~30 plasma and membrane proteins that opsonise pathogens, "
                "recruit phagocytes, and lyse gram-negative bacteria via the membrane attack complex (MAC, C5b-9). "
                "Three activation pathways converge on C3: (1) Classical — activated by antibody-antigen complexes "
                "(C1q recognition); (2) Lectin — activated by mannose/GlcNAc on pathogens (MBL); "
                "(3) Alternative — spontaneous C3 hydrolysis amplified by properdin (positive regulator). "
                "All three generate C3 convertase → cleave C3 → C3b (opsonin) + C3a (anaphylatoxin) → "
                "C5 convertase → C5b → MAC assembly (C5b-6-7-8-9). "
                "Dysregulation causes either infection susceptibility (LOF) or autoimmunity/TMA (regulation failure)."
            ),
            "C1QA/C1q Deficiency — SLE 93% (Most Severe Autoimmunity)": (
                "C1q is the first pattern-recognition protein of the classical complement pathway. "
                "C1q binds: (1) antibody Fc regions (IgG, IgM); (2) apoptotic cell surfaces (phosphatidylserine, DNA). "
                "Without C1q: apoptotic cells not cleared → nuclear antigens (dsDNA, histones, ribonucleoproteins) "
                "present persistently to dendritic cells → autoimmunity → SLE. "
                "93% of C1q-deficient patients develop SLE or SLE-like disease — HIGHEST autoimmunity of all complement deficiencies. "
                "CH50=0 (classical pathway absent); AH50=normal (alternative intact). "
                "C1q protein assay: absent or severely reduced. C1 inhibitor normal. "
                "Treatment: hydroxychloroquine + mycophenolate/azathioprine; no complement replacement available commercially. "
                "CRITICAL: diagnose in any child with severe early-onset SLE and low C4/C2 — measure C1q protein."
            ),
            "C2 Deficiency — Most Common, 28-bp Deletion, European Founder": (
                "Most common complement deficiency in European populations (carrier frequency 1:100; homozygous 1:10,000). "
                "Type I (null, 95%): 28-bp deletion in exon 6 → null allele; no C2 protein produced. "
                "Classical pathway activation fails at C2 cleavage step → C4 accumulates (intact), no C3 convertase via classical route. "
                "CH50=0; AH50=normal (alternative pathway C2-independent). "
                "50% develop SLE-like disease; 75% have recurrent pneumococcal or Haemophilus infections. "
                "Pneumococcal vaccination (PCV13 + PPSV23) + meningococcal conjugate ACWY + serogroup B MANDATORY. "
                "Penicillin prophylaxis in childhood. "
                "DIAGNOSIS: CH50=0, AH50=normal, normal C1q, low C4 consumption (but C4 normal or mildly reduced), absent C2 haemolytic assay."
            ),
            "C3 Deficiency — Central Pivot, Both Pathways Absent, MPGN-II": (
                "C3 is the convergence point of all three complement pathways and the most abundant complement protein. "
                "C3 LOF → NO opsonisation, NO C5 cleavage, NO MAC formation from any pathway. "
                "CH50=0 AND AH50=0 (pathognomonic for C3 deficiency or late complement component deficiency). "
                "Distinguish: C3 low, C4 normal (AP or direct C3 consumption) vs C3 low, C4 low (classical pathway). "
                "Infections: most severe susceptibility to Streptococcus pneumoniae, Neisseria, Haemophilus — bacteraemia from infancy. "
                "MPGN type II (Dense Deposit Disease, DDD): C3 nephritic factor (C3NeF) → stabilises AP C3 convertase → "
                "uncontrolled C3 deposition in glomerular basement membrane. "
                "Fresh frozen plasma (contains C3) for acute infections. Complement genetic panel mandatory. "
                "CRITICAL: any child with recurrent bacteraemia starting in infancy + CH50=0 + AH50=0 → test C3."
            ),
            "CFB / Factor B Deficiency — Alternative Pathway Specific": (
                "Factor B forms the alternative pathway C3 convertase (C3bBb) — catalytic component. "
                "LOF → absent AP amplification; classical and lectin pathways intact. "
                "AH50=0; CH50=normal (classical pathway uses factor D/properdin/factor B for AP only). "
                "Rare: <50 families reported; severe bacterial susceptibility (encapsulated organisms). "
                "IMPORTANT DISTINCTION: GOF (gain-of-function) CFB variants cause the OPPOSITE phenotype — "
                "C3 glomerulopathy and MPGN via hyperactivated AP C3 convertase. "
                "LOF = infection susceptibility; GOF = kidney disease. Interpret variant type carefully. "
                "Diagnosis: AH50 assay absent; CH50 normal; confirm by functional factor B assay + gene sequencing."
            ),
            "CFP / Properdin Deficiency — 7000x Meningococcal Risk (X-Linked)": (
                "Properdin is the ONLY POSITIVE regulator of complement — stabilises C3bBb convertase on pathogen surfaces. "
                "X-linked inheritance: males hemizygous (affected); females typically carriers. "
                "Type I (null): most severe; Type II (reduced, 1-10%); Type III (dysfunctional): milder. "
                "Meningococcal disease risk 7000x baseline — HIGHEST meningococcal susceptibility of all complement deficiencies. "
                "Neisseria meningitidis: fulminant disease, often fatal without prophylaxis. "
                "AH50 reduced/absent; CH50 normal. Serum properdin protein assay: absent or reduced. "
                "MANDATORY for ALL affected males: Meningococcal ACWY conjugate + serogroup B vaccine + "
                "lifelong oral penicillin V 250mg twice daily. "
                "CRITICAL: In any male with first or recurrent meningococcal disease → measure AH50 + properdin level."
            ),
            "C7 Deficiency — Terminal MAC Absent, Neisserial Tropism, Eculizumab CONTRAINDICATED": (
                "C7 is required for MAC (C5b-6-7-8-9) assembly on bacterial membranes. "
                "Terminal complement deficiency: C5/C6/C7/C8/C9 deficiencies share the neisserial tropism. "
                "C7 deficiency: most common terminal deficiency in Japan (p.Ser180Asn founder) and South Africa. "
                "CH50=0, AH50=0 (both zero — MAC cannot assemble); C3 NORMAL; C4 NORMAL — this pattern is PATHOGNOMONIC for terminal complement deficiency. "
                "Recurrent Neisseria meningitidis AND Neisseria gonorrhoea bacteraemia — gramnegative lysis defect. "
                "CRITICAL: ECULIZUMAB IS CONTRAINDICATED in C5-C9 deficiency — eculizumab blocks MAC; "
                "these patients have no MAC and derive zero benefit; anti-C5 would worsen infection risk. "
                "Treatment: meningococcal vaccines + penicillin/ciprofloxacin prophylaxis — NO complement inhibitors."
            ),
            "CFH / Factor H Deficiency — aHUS, MPGN-II, AMD, Eculizumab FDA 2007": (
                "Factor H is the primary fluid-phase negative regulator of the alternative pathway — "
                "binds C3b and accelerates decay of C3bBb; acts as cofactor for factor I-mediated C3b cleavage. "
                "Heterozygous LOF CFH → aHUS (atypical haemolytic uraemic syndrome): "
                "TRIAD: microangiopathic haemolytic anaemia (MAHA) + thrombocytopenia + AKI. "
                "Distinguish from TTP (ADAMTS13 <10%) and STEC-HUS (stool PCR positive). "
                "aHUS workup: ADAMTS13 activity >10%, STEC PCR negative, C3 low, C4 normal/low, CFH variant. "
                "ECULIZUMAB (anti-C5 monoclonal antibody): FDA approved 2011 (first for aHUS) — "
                "originally FDA 2007 for PNH — FIRST complement-directed therapy ever approved. "
                "RAVULIZUMAB (long-acting, 8-weekly): preferred over eculizumab. "
                "PRE-TREATMENT MANDATE: Meningococcal ACWY + serogroup B vaccines ≥2 weeks before eculizumab "
                "(eculizumab blocks MAC → neisserial susceptibility). "
                "Biallelic CFH null → C3 nephritic factor → MPGN-II/DDD (renal biopsy confirms). "
                "CFH Y402H SNP: common variant (30-40% population), 2-3x AMD risk — NOT aHUS-causing alone."
            ),
            "CD46 / MCP Deficiency — aHUS Incomplete Penetrance, Trigger-Dependent": (
                "Membrane cofactor protein (MCP/CD46) is expressed on all nucleated cells; "
                "acts as cofactor for factor I-mediated inactivation of C3b/C4b on self-surfaces. "
                "AD LOF → aHUS susceptibility — INCOMPLETE PENETRANCE (~50%): "
                "many carriers only develop aHUS after triggers (pregnancy, viral infection, diarrhoeal illness). "
                "Milder phenotype than CFH-aHUS: more episodic TMA, better renal recovery, lower ESRD risk. "
                "DIAGNOSIS: MCP expression on leukocytes absent or severely reduced by flow cytometry (DIAGNOSTIC). "
                "Eculizumab/ravulizumab for acute TMA episodes — highly effective. "
                "DISTINCTION FROM CFH-aHUS: some CD46-aHUS patients successfully discontinue eculizumab "
                "after TMA resolution (unlike CFH-aHUS where indefinite treatment usually required). "
                "aHUS WORKUP PROTOCOL: ADAMTS13 >10% + STEC stool PCR negative + C3/C4/CFH/CFI/CD46/CFB/C3/THBD panel."
            ),
            "Complement Screening Tests — CH50 and AH50": (
                "CH50 (total haemolytic complement): measures functional activity of the CLASSICAL pathway (C1-C9). "
                "AH50 (alternative haemolytic complement): measures functional activity of the ALTERNATIVE pathway. "
                "Interpretation matrix: "
                "(1) CH50=0, AH50=normal → Classical pathway defect: C1q/C1r/C1s → C4 → C2. "
                "(2) CH50=0, AH50=0, C3 LOW → C3 deficiency (central pivot). "
                "(3) CH50=0, AH50=0, C3 normal, C4 normal → Terminal pathway defect (C5-C9). "
                "(4) AH50=0, CH50=normal → Alternative pathway defect: CFB or CFP (properdin). "
                "(5) CH50 borderline, AH50 normal → Heterozygous complement deficiency; repeat after acute phase. "
                "IMPORTANT: both CH50 and AH50 are consumed during active infection/autoimmunity — "
                "draw samples during quiescent disease for accurate pathway assessment."
            ),
            "Eculizumab / Ravulizumab — Complement Inhibitors and Meningococcal Risk": (
                "Eculizumab (Soliris®): anti-C5 humanised monoclonal antibody; FDA 2007 PNH, 2011 aHUS, 2019 NMO-SD. "
                "Ravulizumab (Ultomiris®): long-acting anti-C5 (FcRn-mediated recycling); FDA 2019; 8-weekly dosing. "
                "MECHANISM: block C5 cleavage → no C5a (anaphylatoxin) + no C5b (MAC initiation). "
                "CRITICAL NEISSERIAL RISK: MAC is the primary defence against gram-negative encapsulated bacteria. "
                "Eculizumab increases meningococcal disease risk 1000-2000x above baseline. "
                "MANDATORY PRE-TREATMENT: Meningococcal ACWY conjugate + serogroup B (Bexsero/Trumenba) vaccines, "
                "ideally ≥2 weeks before eculizumab; if urgent treatment needed → prophylactic antibiotics cover gap. "
                "Meningococcal prophylaxis: ciprofloxacin 500mg ×1 dose or penicillin V 250mg bid during treatment. "
                "CONTRAINDICATED in: C5/C6/C7/C8/C9 deficiency (no benefit; removes only available defence mechanism)."
            ),
            "aHUS Diagnostic Workup — TMA Differential": (
                "Thrombotic microangiopathy (TMA) diagnosis: haemolytic anaemia (blood film: schistocytes) + "
                "thrombocytopenia + organ injury (AKI, CNS, cardiac). "
                "Step 1 — Exclude TTP: ADAMTS13 activity <10% = TTP → plasma exchange. "
                "Step 2 — Exclude STEC-HUS: stool culture + Shiga toxin PCR positive = STEC-HUS (supportive only; NO eculizumab). "
                "Step 3 — aHUS: ADAMTS13 >10% + STEC negative → complement genetic panel + C3/C4/CFH/CFB levels + "
                "MCP flow cytometry + anti-factor H antibody. "
                "aHUS triggers: pregnancy (especially post-partum), diarrhoeal infections, drugs. "
                "Genes: CFH (25%), MCP/CD46 (15%), CFI (10%), C3 (10%), CFB (5%), THBD (5%); 35% no variant found. "
                "Empiric eculizumab early in severe/deteriorating aHUS while awaiting genetics."
            ),
        },
        "standards": [
            "KDIGO Clinical Practice Guideline for Glomerular Diseases 2021 — C3 glomerulopathy + aHUS",
            "European aHUS working party — Loirat C et al. Orphanet J Rare Dis 2016",
            "Eculizumab for aHUS — Legendre CM et al. NEJM 2013",
            "Ravulizumab phase 3 aHUS — Rondeau E et al. Kidney Int Rep 2020",
            "Complement deficiency and SLE — Walport MJ, NEJM 2001",
            "Properdin deficiency and meningococcal disease — Fijen CA et al. Clin Exp Immunol 1999",
            "Primary complement deficiencies — Grumach AS, Kirschfink M. Eur J Immunol 2014",
            "C2 deficiency 28-bp deletion — Jonsson G et al. J Clin Immunol 2011",
            "ACIP meningococcal vaccine recommendations for complement deficiency — CDC 2020",
        ],
    }
