#!/usr/bin/env python3
"""Hereditary-Complement-Deficiency-Atlas -- Complete 8-Gene Reference
Complement System Deficiency Diseases
(C1QA/SLE-Highest-Risk-90pct-C1q-Def-Lupus-Like-Rash-PATHOGNOMONIC-Anti-dsDNA-Negative-DDx,
 C2/Most-Common-Complement-Def-Western-1:20000-SLE-30pct-Encapsulated-Bacteria-Pneumococcal-Vaccines,
 C4A/Classical-Pathway-Amplifier-SLE-Partial-Def-LOF-Alleles-C4A-NULL-Most-Common-European,
 C3/Central-Complement-Component-Encapsulated-Bacteria-HIGHEST-Severity-MPGN-PATHOGNOMONIC,
 CFB/Alternative-Pathway-Factor-B-Recurrent-Neisseria-Meningitidis-MPGN-Type2-Dense-Deposit,
 CFD/Alternative-Pathway-Factor-D-RAREST-Recurrent-Neisseria-North-Africa-Founder-2x-Vaccination,
 C5/Terminal-Pathway-Entry-Neisseria-Meningitidis-PATHOGNOMONIC-Eculizumab-FDA2007-CURATIVE,
 CFP/Properdin-Only-XLR-Complement-Deficiency-Neisseria-Boys-Fulminant-Meningococcemia-30pct-Mortality):

C1QA  (Complement component 1 Q subcomponent alpha chain; 245aa; 1p36.12; AR;
         C1QA encodes the alpha chain of C1q, the recognition subunit of the C1 complex (C1q+C1r+C1s);
         C1q binds antibody-antigen complexes to activate the classical complement pathway;
         C1q also plays critical roles in apoptotic cell clearance and immune tolerance;
         C1q DEFICIENCY: HIGHEST autoimmune risk of all complement deficiencies — 90% of C1q-deficient patients develop SLE;
         LUPUS-LIKE RASH (photosensitive butterfly rash) IS PATHOGNOMONIC of C1q deficiency;
         KEY DDx: ANTI-dsDNA ANTIBODIES NEGATIVE in C1q deficiency SLE — distinguishes from idiopathic SLE;
         Other features: glomerulonephritis, vasculitis, recurrent skin infections, Streptococcal infections;
         Mechanism: impaired clearance of apoptotic cells → nuclear antigens exposed → anti-nuclear antibody formation;
         Treatment: hydroxychloroquine (antimalarial; reduces photosensitivity and lupus activity);
         Pneumococcal + meningococcal vaccination MANDATORY;
         Very rare — fewer than 40 patients reported worldwide;
         seed SEED_BASE+0).
C2     (Complement component 2; 752aa; 6p21.33; AR;
         C2 is the serine protease of the classical and lectin pathways;
         C1s cleaves C2 into C2a (large catalytic fragment) and C2b → C2a joins C4b to form C3 convertase C4b2a;
         C2 DEFICIENCY: MOST COMMON complement deficiency in Western populations (1 in 20,000 North Europeans);
         SLE occurs in approximately 30% of C2-deficient individuals (lower risk than C1q but more prevalent overall);
         PRIMARY RISK: recurrent infections with ENCAPSULATED BACTERIA (Streptococcus pneumoniae, Haemophilus influenzae);
         PNEUMOCOCCAL VACCINES (PCV13 + PPSV23) MANDATORY — reduces invasive pneumococcal disease dramatically;
         PENICILLIN PROPHYLAXIS: lifelong daily penicillin V for all C2-deficient patients;
         Common haplotype: C2 null allele (28bp deletion in exon 6) on MHC haplotype HLA-A25-B18-DR2 (most European C2 null);
         Other features: Streptococcal bacteraemia, meningitis, discoid lupus, anti-Ro/anti-La antibodies;
         complement CH50 haemolytic assay = ZERO in homozygous C2 deficiency (classical pathway absent);
         seed SEED_BASE+1).
C4A    (Complement component 4A; 1744aa; 6p21.33; AR+AD (partial deficiency from LOF alleles);
         C4A is a thioester-containing complement protein of the classical and lectin pathways;
         C4 is cleaved by activated C1s into C4a (anaphylatoxin) and C4b (opsonin) → C4b binds to activating surfaces;
         C4A preferentially reacts with amino groups (protein antigens); C4B with hydroxyl groups (polysaccharide surfaces);
         C4A NULL allele: most common MHC null allele in European populations (~15% carrier frequency);
         PARTIAL DEFICIENCY (heterozygous C4A*Q0): 2-3× increased SLE risk; partial complement activity preserved;
         COMPLETE C4A DEFICIENCY (homozygous): higher SLE risk; recurrent infections with encapsulated bacteria;
         Classical pathway amplifier — reduced C4A → impaired immune complex clearance → inflammatory deposit;
         C4A and C4B gene copy number varies (CNV) due to genomic duplications/deletions in MHC region;
         C4A deficiency SLE: similar phenotype to idiopathic SLE; anti-dsDNA may be positive (unlike C1q);
         Complement CH50 may be near-normal if C4B present; C4 functional assay (haem-specific) needed;
         seed SEED_BASE+2).
C3     (Complement component 3; 1663aa; 19p13.3; AR;
         C3 is the CENTRAL molecule of the complement system — all three pathways converge on C3;
         C3 is the most abundant complement protein in plasma (~1.2 mg/mL);
         C3 DEFICIENCY: most severe susceptibility to ENCAPSULATED BACTERIA of all complement deficiencies;
         SEVERE AND RECURRENT: Streptococcus pneumoniae meningitis/bacteraemia, Neisseria meningitidis, Haemophilus influenzae;
         MEMBRANOPROLIFERATIVE GLOMERULONEPHRITIS (MPGN) IS PATHOGNOMONIC of C3 deficiency;
         Mechanism: unregulated alternative pathway (C3 absent → C3bBb formation absent → AP turnover uncontrolled) → MPGN;
         Additional features: autoimmunity (SLE-like), partial lipodystrophy, recurrent skin infections;
         C3 deficiency also causes failure of opsonisation → impaired phagocytosis of encapsulated bacteria;
         ALL COMPLEMENT PATHWAYS BLOCKED (C3 is downstream of all three) → most immunocompromised complement phenotype;
         Eculizumab (anti-C5) is emerging as treatment for MPGN in C3 deficiency (blocks terminal pathway activation);
         Pneumococcal + meningococcal + Hib vaccines MANDATORY; antibiotic prophylaxis;
         seed SEED_BASE+3).
CFB    (Complement factor B; 764aa; 6p21.33; AR;
         CFB encodes Factor B, the serine protease of the alternative pathway;
         Factor D cleaves Factor B bound to C3b → generates Bb fragment → C3bBb = C3 convertase of alternative pathway;
         Factor B deficiency: SELECTIVE alternative pathway deficiency (classical/lectin pathways intact);
         Primary susceptibility: RECURRENT NEISSERIA MENINGITIDIS (alternative pathway critical for Neisseria opsonisation);
         MPGN TYPE 2 (DENSE DEPOSIT DISEASE): characteristic renal complication of alternative pathway factor deficiencies;
         Mechanism of Dense Deposit Disease: unregulated alternative pathway → C3 deposits in GBM dense deposits;
         Rare disease — very few families reported; alternative pathway function can be measured by AP50 assay;
         AP50 (alternative pathway haemolytic assay) = ZERO; CH50 (classical) = NORMAL or near-normal in CFB deficiency;
         Eculizumab used for Dense Deposit Disease in CFB deficiency;
         Neisseria meningitidis 4-valent vaccine MANDATORY; consider 2× schedule;
         seed SEED_BASE+4).
CFD    (Complement factor D; 259aa; 19p13.3; AR;
         CFD encodes Factor D, the rate-limiting enzyme of the alternative pathway — present in lowest concentration of all complement proteins;
         Factor D is the serine protease that cleaves Factor B when bound to C3b to generate the alternative pathway C3 convertase;
         Factor D DEFICIENCY: RAREST complement deficiency — fewer than 15 families reported worldwide;
         NORTH AFRICA FOUNDER EFFECT: cluster of Factor D deficiency reported in Moroccan/North African populations;
         PRIMARY SUSCEPTIBILITY: RECURRENT NEISSERIA MENINGITIDIS — identical to other alternative pathway deficiencies;
         2× VACCINATION SCHEDULE MANDATORY: meningococcal ACWY (MenACWY) + MenB, both given in 2-dose primary series;
         Factor D deficiency: AP50 = ZERO; CH50 = NORMAL;
         Very rare — because Factor D is present in such low plasma concentration, complete loss is profoundly immunocompromising for Neisseria;
         Treatment: 2-dose meningococcal vaccination schedule; antibiotic prophylaxis with penicillin V;
         No specific immunotherapy beyond vaccination and prophylaxis currently available;
         seed SEED_BASE+5).
C5     (Complement component 5; 1676aa; 9q21.2; AR;
         C5 is the ENTRY POINT of the terminal complement pathway (MAC formation);
         C5 convertases (C4b2a3b or C3bBbC3b) cleave C5 → C5a (potent anaphylatoxin + chemotactic) + C5b (MAC nucleation);
         C5b initiates assembly of C5b-6-7-8-9 = MEMBRANE ATTACK COMPLEX (MAC);
         MAC is essential for lysis of Gram-negative bacteria, especially NEISSERIA species;
         C5 DEFICIENCY: PATHOGNOMONIC association with RECURRENT NEISSERIA MENINGITIDIS (meningococcal disease);
         All terminal complement deficiencies (C5-C9) share this Neisseria susceptibility — C5 is the gatekeeper;
         ECULIZUMAB (anti-C5 monoclonal antibody; FDA-approved 2007 for PNH) = CURATIVE for terminal complement activation;
         RAVULIZUMAB (long-acting anti-C5; monthly dosing) reduces injection burden vs eculizumab (every 2 weeks);
         NOTE: Eculizumab treatment creates FUNCTIONAL C5 deficiency → increases meningococcal risk → vaccination MANDATORY before eculizumab;
         Additional features: lupus-like manifestations, anti-nuclear antibodies;
         All terminal pathway deficiencies (C5-C9, Factor P) → recurrent meningococcal disease is the unifying clinical feature;
         seed SEED_BASE+6).
CFP    (Complement factor P / Properdin; 442aa; Xp11.23; XLR;
         CFP encodes Properdin, the ONLY POSITIVE REGULATOR of the complement system;
         Properdin stabilises the alternative pathway C3 convertase (C3bBb) and C5 convertase (C3bBbC3b) → prolongs their half-life 10×;
         Without Properdin, alternative pathway C3 convertase decays rapidly → AP cannot amplify efficiently;
         Properdin deficiency = ONLY X-LINKED RECESSIVE COMPLEMENT DEFICIENCY;
         MALES ONLY (XLR) — boys and young men at risk; female carriers protected;
         PATHOGNOMONIC NEISSERIA SUSCEPTIBILITY: young boys presenting with Neisseria meningitidis in first two decades of life;
         FULMINANT MENINGOCOCCEMIA: 30% MORTALITY — most severe and rapidly fatal of all complement-deficiency presentations;
         Type I Properdin deficiency: absent Properdin protein; Type II: low but present; Type III: dysfunctional;
         2× VACCINATION MANDATORY: MenACWY + MenB vaccines both in 2-dose series; revaccinate every 5 years;
         Penicillin V prophylaxis for high-risk boys; family screening MANDATORY (XLR → maternal uncles/brothers at risk);
         Three Properdin deficiency types (I/II/III): all share same clinical phenotype — Type I most common;
         seed SEED_BASE+7).
"""

import random
import math

ATLAS_NAME   = "Hereditary-Complement-Deficiency-Atlas"
SEED_BASE    = 3982   # seeds 3982-3989 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8

GENES = ["C1QA", "C2", "C4A", "C3", "CFB", "CFD", "C5", "CFP"]

GENE_DATA = {
    "C1QA": {
        "full_name":             "Complement component 1 Q subcomponent alpha chain (C1q α-chain)",
        "aa":                    245,
        "locus":                 "1p36.12",
        "kda":                   "27",
        "inheritance":           "AR (autosomal recessive; very rare <40 families worldwide)",
        "syndrome":              "C1q Deficiency (SLE-HIGHEST risk complement deficiency; 90% develop SLE)",
        "pathway":               "Classical pathway absent → impaired apoptotic cell clearance → nuclear antigen exposure → autoimmunity",
        "onset_range":           (12, 120),   # childhood onset lupus
        "sle_pct":               0.90,        # 90% C1q-deficient patients develop SLE
        "bacterial_pct":         0.55,        # encapsulated bacteria susceptibility
        "mpgn_pct":              0.35,        # glomerulonephritis
        "neisseria_pct":         0.25,        # Neisseria meningitidis
        "eculizumab_pct":        0.05,
        "vaccine_2dose_pct":     0.90,        # vaccination mandatory
        "lupus_rash_pct":        0.80,        # PATHOGNOMONIC lupus-like rash
        "anti_dsdna_neg_pct":    0.75,        # anti-dsDNA negative DDx
        "hcq_pct":               0.85,        # hydroxychloroquine
        "pathognomonic":         "LUPUS-LIKE RASH (photosensitive butterfly rash) PATHOGNOMONIC; ANTI-dsDNA NEGATIVE = DDx from idiopathic SLE; 90% C1q-deficient patients develop SLE — HIGHEST autoimmune risk; impaired apoptotic clearance",
        "key_drug":              "Hydroxychloroquine (antimalarial; reduces photosensitivity and lupus activity); pneumococcal + meningococcal vaccination MANDATORY",
    },
    "C2": {
        "full_name":             "Complement component 2 (C2; classical/lectin pathway serine protease)",
        "aa":                    752,
        "locus":                 "6p21.33",
        "kda":                   "102",
        "inheritance":           "AR (most common: HLA-A25-B18-DR2 haplotype 28bp deletion in exon 6)",
        "syndrome":              "C2 Deficiency (most common complement deficiency in Western populations; 1:20,000 North Europeans)",
        "pathway":               "Classical pathway C3 convertase formation absent → impaired opsonisation → encapsulated bacteria + SLE-30pct",
        "onset_range":           (6, 240),    # variable; infections in childhood, lupus adulthood
        "sle_pct":               0.30,        # 30% develop SLE
        "bacterial_pct":         0.85,        # encapsulated bacteria primary risk
        "mpgn_pct":              0.20,
        "neisseria_pct":         0.35,
        "eculizumab_pct":        0.05,
        "vaccine_2dose_pct":     0.95,        # pneumococcal vaccination MANDATORY
        "lupus_rash_pct":        0.25,
        "anti_dsdna_neg_pct":    0.40,        # some anti-dsDNA positive unlike C1q
        "hcq_pct":               0.30,        # hydroxychloroquine for SLE subset
        "pathognomonic":         "MOST COMMON complement deficiency in Western populations (1:20,000 North Europeans); ENCAPSULATED BACTERIA primary risk (Streptococcus pneumoniae, Haemophilus influenzae); SLE ~30%; PNEUMOCOCCAL VACCINES (PCV13+PPSV23) + PENICILLIN PROPHYLAXIS MANDATORY",
        "key_drug":              "Pneumococcal vaccines (PCV13 + PPSV23) MANDATORY; lifelong penicillin V prophylaxis; meningococcal vaccination",
    },
    "C4A": {
        "full_name":             "Complement component 4A (C4A; classical/lectin pathway amplifier; thioester protein)",
        "aa":                    1744,
        "locus":                 "6p21.33",
        "kda":                   "193",
        "inheritance":           "AR+AD (complete deficiency AR; partial from heterozygous C4A*Q0 null allele)",
        "syndrome":              "C4A Deficiency (partial: 2-3× SLE risk; C4A NULL most common MHC null allele in Europeans)",
        "pathway":               "Classical pathway C4b opsonisation reduced → impaired immune complex clearance → inflammatory tissue deposits → SLE",
        "onset_range":           (18, 360),   # variable; often discovered in adulthood during SLE workup
        "sle_pct":               0.65,        # heterozygous partial + complete
        "bacterial_pct":         0.55,
        "mpgn_pct":              0.30,
        "neisseria_pct":         0.25,
        "eculizumab_pct":        0.05,
        "vaccine_2dose_pct":     0.85,
        "lupus_rash_pct":        0.50,
        "anti_dsdna_neg_pct":    0.30,        # anti-dsDNA may be positive (differs from C1q)
        "hcq_pct":               0.65,
        "pathognomonic":         "C4A NULL ALLELE most common MHC null allele in Europeans (~15% carrier); PARTIAL DEFICIENCY increases SLE risk 2-3×; COMPLETE DEFICIENCY: SLE + encapsulated bacteria; anti-dsDNA may be POSITIVE (unlike C1q DDx); MHC CNV analysis required",
        "key_drug":              "Hydroxychloroquine for SLE manifestations; pneumococcal + meningococcal vaccination; C4 functional assay for diagnosis",
    },
    "C3": {
        "full_name":             "Complement component 3 (C3; central complement protein; all pathways converge)",
        "aa":                    1663,
        "locus":                 "19p13.3",
        "kda":                   "187",
        "inheritance":           "AR (autosomal recessive; rare; most severe infection susceptibility)",
        "syndrome":              "C3 Deficiency (most severe encapsulated bacteria susceptibility; MPGN pathognomonic; all complement pathways blocked)",
        "pathway":               "All three pathways (classical/lectin/alternative) absent → most severe opsonisation defect → MPGN + encapsulated bacteria",
        "onset_range":           (0, 36),     # often presents in early childhood with severe infection
        "sle_pct":               0.50,
        "bacterial_pct":         0.98,        # HIGHEST encapsulated bacteria severity
        "mpgn_pct":              0.80,        # MPGN PATHOGNOMONIC
        "neisseria_pct":         0.65,
        "eculizumab_pct":        0.40,        # eculizumab emerging for MPGN
        "vaccine_2dose_pct":     0.98,
        "lupus_rash_pct":        0.35,
        "anti_dsdna_neg_pct":    0.30,
        "hcq_pct":               0.40,
        "pathognomonic":         "MEMBRANOPROLIFERATIVE GLOMERULONEPHRITIS (MPGN) PATHOGNOMONIC; MOST SEVERE encapsulated bacteria susceptibility (ALL THREE PATHWAYS BLOCKED — C3 central to all); Streptococcus pneumoniae meningitis recurrent; Eculizumab emerging for MPGN; Pneumococcal + MenACWY + MenB + Hib vaccines MANDATORY",
        "key_drug":              "Eculizumab (emerging for C3 deficiency MPGN); pneumococcal + meningococcal + Hib vaccination MANDATORY; antibiotic prophylaxis lifelong",
    },
    "CFB": {
        "full_name":             "Complement factor B (CFB; alternative pathway serine protease; C3bBb convertase component)",
        "aa":                    764,
        "locus":                 "6p21.33",
        "kda":                   "93",
        "inheritance":           "AR (very rare; alternative pathway selectively absent; classical intact)",
        "syndrome":              "Factor B Deficiency (selective alternative pathway deficiency; MPGN Type 2/Dense Deposit Disease; Neisseria susceptibility)",
        "pathway":               "Alternative pathway C3 convertase C3bBb absent → unregulated AP activation → Dense Deposit Disease + Neisseria susceptibility",
        "onset_range":           (12, 180),
        "sle_pct":               0.25,
        "bacterial_pct":         0.65,
        "mpgn_pct":              0.70,        # Dense Deposit Disease characteristic
        "neisseria_pct":         0.80,        # recurrent Neisseria meningitidis
        "eculizumab_pct":        0.60,        # eculizumab for Dense Deposit Disease
        "vaccine_2dose_pct":     0.95,
        "lupus_rash_pct":        0.15,
        "anti_dsdna_neg_pct":    0.50,
        "hcq_pct":               0.20,
        "pathognomonic":         "MPGN TYPE 2 DENSE DEPOSIT DISEASE characteristic of Factor B deficiency; RECURRENT NEISSERIA MENINGITIDIS; AP50 = ZERO (alternative pathway absent); CH50 = NORMAL (classical pathway intact) — AP50/CH50 ratio distinguishes; Eculizumab for Dense Deposit Disease",
        "key_drug":              "Eculizumab (for Dense Deposit Disease / MPGN Type 2); meningococcal vaccination MANDATORY (2-dose MenACWY + MenB); AP50 assay for diagnosis",
    },
    "CFD": {
        "full_name":             "Complement factor D (CFD; alternative pathway rate-limiting serine protease; lowest plasma concentration)",
        "aa":                    259,
        "locus":                 "19p13.3",
        "kda":                   "24",
        "inheritance":           "AR (autosomal recessive; RAREST complement deficiency <15 families; North Africa founder)",
        "syndrome":              "Factor D Deficiency (rarest complement deficiency; alternative pathway absent; Neisseria susceptibility; North Africa founder)",
        "pathway":               "Factor D absent → cannot cleave Factor B bound to C3b → alternative pathway C3 convertase C3bBb cannot form → AP absent",
        "onset_range":           (6, 120),
        "sle_pct":               0.15,
        "bacterial_pct":         0.75,
        "mpgn_pct":              0.45,
        "neisseria_pct":         0.88,        # primary susceptibility Neisseria
        "eculizumab_pct":        0.30,
        "vaccine_2dose_pct":     0.98,        # 2× vaccination schedule MANDATORY
        "lupus_rash_pct":        0.10,
        "anti_dsdna_neg_pct":    0.60,
        "hcq_pct":               0.10,
        "pathognomonic":         "RAREST complement deficiency (<15 families worldwide); NORTH AFRICA FOUNDER EFFECT (Moroccan/North African cluster); RECURRENT NEISSERIA MENINGITIDIS primary presentation; 2× VACCINATION SCHEDULE MANDATORY (MenACWY + MenB dual-dose primary series); AP50 = ZERO; penicillin V prophylaxis",
        "key_drug":              "2-dose MenACWY + MenB vaccination (2× series MANDATORY); penicillin V prophylaxis lifelong; AP50 assay for diagnosis; family screening in North African families",
    },
    "C5": {
        "full_name":             "Complement component 5 (C5; terminal pathway entry; MAC nucleation; anaphylatoxin C5a)",
        "aa":                    1676,
        "locus":                 "9q21.2",
        "kda":                   "190",
        "inheritance":           "AR (autosomal recessive; terminal complement pathway absent)",
        "syndrome":              "C5 Deficiency (terminal pathway entry absent; recurrent Neisseria meningitidis PATHOGNOMONIC; eculizumab FDA-approved 2007)",
        "pathway":               "C5 absent → MAC (C5b-9) cannot form → Gram-negative bacteria not lysed → Neisseria meningitidis recurrent",
        "onset_range":           (12, 240),
        "sle_pct":               0.30,
        "bacterial_pct":         0.60,
        "mpgn_pct":              0.20,
        "neisseria_pct":         0.92,        # PATHOGNOMONIC Neisseria susceptibility
        "eculizumab_pct":        0.75,        # ECULIZUMAB FDA 2007 CURATIVE
        "vaccine_2dose_pct":     0.98,
        "lupus_rash_pct":        0.20,
        "anti_dsdna_neg_pct":    0.40,
        "hcq_pct":               0.25,
        "pathognomonic":         "NEISSERIA MENINGITIDIS RECURRENT — PATHOGNOMONIC for terminal complement deficiency; ECULIZUMAB (anti-C5 monoclonal; FDA-approved 2007) CURATIVE for terminal complement activation; RAVULIZUMAB (monthly dosing) alternative; NOTE: eculizumab creates functional C5 deficiency → vaccination MANDATORY BEFORE starting eculizumab",
        "key_drug":              "Eculizumab (anti-C5; FDA 2007; curative for terminal pathway activation); Ravulizumab (monthly dosing); vaccination with MenACWY + MenB MANDATORY before starting eculizumab",
    },
    "CFP": {
        "full_name":             "Complement factor P / Properdin (CFP; only XLR complement deficiency; AP convertase stabiliser)",
        "aa":                    442,
        "locus":                 "Xp11.23",
        "kda":                   "53",
        "inheritance":           "XLR (X-linked recessive — ONLY XLR complement deficiency; males only classically)",
        "syndrome":              "Properdin Deficiency (only XLR complement deficiency; fulminant meningococcemia 30% mortality; boys/young men)",
        "pathway":               "Properdin absent → AP C3 convertase C3bBb decays rapidly (not stabilised) → AP amplification severely reduced → Neisseria uncontrolled",
        "onset_range":           (12, 240),   # boys/young men
        "sle_pct":               0.10,
        "bacterial_pct":         0.60,
        "mpgn_pct":              0.15,
        "neisseria_pct":         0.95,        # PATHOGNOMONIC Neisseria boys
        "eculizumab_pct":        0.20,
        "vaccine_2dose_pct":     0.98,        # 2× vaccination MANDATORY
        "lupus_rash_pct":        0.08,
        "anti_dsdna_neg_pct":    0.50,
        "hcq_pct":               0.08,
        "pathognomonic":         "ONLY X-LINKED RECESSIVE COMPLEMENT DEFICIENCY; MALES ONLY; FULMINANT MENINGOCOCCEMIA 30% MORTALITY — most rapidly fatal complement deficiency presentation; NEISSERIA MENINGITIDIS PATHOGNOMONIC in boys/young men; 2× VACCINATION MANDATORY (MenACWY + MenB dual-dose primary); family screening MANDATORY (maternal uncles/brothers at risk)",
        "key_drug":              "MenACWY + MenB vaccination (2-dose primary series MANDATORY); revaccinate every 5 years; penicillin V prophylaxis; family screening (XLR pattern); emergency protocols for suspected meningococcemia",
    },
}


def _rng(seed):
    return random.Random(seed)


def _simulate_patient(gene, seed):
    rng = _rng(seed)
    gd  = GENE_DATA[gene]
    onset_lo, onset_hi = gd["onset_range"]
    onset_mo = rng.randint(onset_lo, max(onset_lo + 1, onset_hi))

    sle              = rng.random() < gd["sle_pct"]
    bacterial        = rng.random() < gd["bacterial_pct"]
    mpgn             = rng.random() < gd["mpgn_pct"]
    neisseria        = rng.random() < gd["neisseria_pct"]
    eculizumab       = rng.random() < gd["eculizumab_pct"]
    vaccine_2dose    = rng.random() < gd["vaccine_2dose_pct"]
    lupus_rash       = rng.random() < gd["lupus_rash_pct"]
    anti_dsdna_neg   = rng.random() < gd["anti_dsdna_neg_pct"]
    hcq              = rng.random() < gd["hcq_pct"]

    return {
        "onset_months":    onset_mo,
        "sle":             sle,
        "bacterial":       bacterial,
        "mpgn":            mpgn,
        "neisseria":       neisseria,
        "eculizumab":      eculizumab,
        "vaccine_2dose":   vaccine_2dose,
        "lupus_rash":      lupus_rash,
        "anti_dsdna_neg":  anti_dsdna_neg,
        "hcq":             hcq,
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
        "sle_pct":           pct("sle"),
        "bacterial_pct":     pct("bacterial"),
        "mpgn_pct":          pct("mpgn"),
        "neisseria_pct":     pct("neisseria"),
        "eculizumab_pct":    pct("eculizumab"),
        "vaccine_2dose_pct": pct("vaccine_2dose"),
        "lupus_rash_pct":    pct("lupus_rash"),
        "anti_dsdna_neg_pct": pct("anti_dsdna_neg"),
        "hcq_pct":           pct("hcq"),
        "mean_onset_months": mean_onset,
        "pathognomonic":     GENE_DATA[gene]["pathognomonic"],
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    total_sle           = round(sum(c["sle_pct"]           for c in cohorts) / len(cohorts), 1)
    total_bacterial     = round(sum(c["bacterial_pct"]     for c in cohorts) / len(cohorts), 1)
    total_mpgn          = round(sum(c["mpgn_pct"]          for c in cohorts) / len(cohorts), 1)
    total_neisseria     = round(sum(c["neisseria_pct"]      for c in cohorts) / len(cohorts), 1)
    total_eculizumab    = round(sum(c["eculizumab_pct"]    for c in cohorts) / len(cohorts), 1)
    total_vaccine_2dose = round(sum(c["vaccine_2dose_pct"] for c in cohorts) / len(cohorts), 1)
    total_lupus_rash    = round(sum(c["lupus_rash_pct"]    for c in cohorts) / len(cohorts), 1)
    total_hcq           = round(sum(c["hcq_pct"]           for c in cohorts) / len(cohorts), 1)

    return {
        "atlas":    ATLAS_NAME,
        "genes":    GENES,
        "total_n":  TOTAL_N,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "sle_pct":           total_sle,
            "bacterial_pct":     total_bacterial,
            "mpgn_pct":          total_mpgn,
            "neisseria_pct":     total_neisseria,
            "eculizumab_pct":    total_eculizumab,
            "vaccine_2dose_pct": total_vaccine_2dose,
            "lupus_rash_pct":    total_lupus_rash,
            "hcq_pct":           total_hcq,
        },
        "key_facts": [
            "C1QA (C1q Deficiency): LUPUS-LIKE RASH PATHOGNOMONIC; ANTI-dsDNA NEGATIVE distinguishes from idiopathic SLE; 90% develop SLE = HIGHEST autoimmune risk of all complement deficiencies; impaired apoptotic clearance; hydroxychloroquine + vaccination MANDATORY",
            "C2 (Most Common Complement Def Western 1:20,000): PNEUMOCOCCAL VACCINES (PCV13+PPSV23) + PENICILLIN PROPHYLAXIS MANDATORY; SLE ~30%; ENCAPSULATED BACTERIA primary risk (S. pneumoniae/H. influenzae); HLA-A25-B18-DR2 null haplotype",
            "C4A (Classical Pathway Amplifier): C4A NULL most common MHC null allele in Europeans (~15% carrier); PARTIAL DEFICIENCY 2-3× SLE risk; anti-dsDNA may be POSITIVE (unlike C1q — important DDx); CNV analysis required",
            "C3 (Central Complement Component): MOST SEVERE encapsulated bacteria susceptibility (ALL THREE PATHWAYS BLOCKED); MPGN PATHOGNOMONIC; Eculizumab emerging for MPGN; ALL vaccines MANDATORY (pneumococcal + MenACWY + MenB + Hib)",
            "CFB (Alternative Pathway Factor B): MPGN TYPE 2 DENSE DEPOSIT DISEASE characteristic; RECURRENT NEISSERIA MENINGITIDIS; AP50 = ZERO + CH50 NORMAL distinguishes from classical pathway deficiencies; Eculizumab for Dense Deposit Disease",
            "CFD (Factor D Deficiency — RAREST): <15 families worldwide; NORTH AFRICA FOUNDER EFFECT; RECURRENT NEISSERIA primary; 2× VACCINATION SCHEDULE MANDATORY (MenACWY + MenB dual-dose primary); AP50 = ZERO",
            "C5 (Terminal Pathway Entry): NEISSERIA MENINGITIDIS RECURRENT PATHOGNOMONIC for terminal complement deficiency; ECULIZUMAB (anti-C5; FDA 2007) CURATIVE; RAVULIZUMAB monthly alternative; vaccination MANDATORY BEFORE starting eculizumab",
            "CFP (Properdin — ONLY XLR Complement Deficiency): MALES ONLY; FULMINANT MENINGOCOCCEMIA 30% MORTALITY — most rapidly fatal complement deficiency; PATHOGNOMONIC Neisseria in boys/young men; 2× VACCINATION + family screening MANDATORY",
        ],
        "classical_pathway_group":    ["C1QA", "C2", "C4A"],
        "central_complement_group":   ["C3"],
        "alternative_pathway_group":  ["CFB", "CFD"],
        "terminal_pathway_group":     ["C5"],
        "properdin_xlr_group":        ["CFP"],
        "cohort_summaries":           cohorts,
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    diagnostic_groups = {
        "classical_pathway_autoimmune": {
            "genes":          ["C1QA", "C2", "C4A"],
            "mechanism":      "Classical pathway early components absent → impaired immune complex clearance + impaired apoptotic cell clearance → SLE-spectrum autoimmunity",
            "shared_feature": "SLE predominant phenotype; CH50 = ZERO (classical); AP50 = NORMAL; anti-nuclear antibodies; photosensitivity",
            "key_ddx":        "C1QA: anti-dsDNA NEGATIVE; lupus rash PATHOGNOMONIC; 90% SLE risk. C2: anti-dsDNA may be positive; most common; penicillin prophylaxis. C4A: partial deficiency from null allele; 2-3× SLE risk heterozygous.",
        },
        "central_complement_severe": {
            "genes":          ["C3"],
            "mechanism":      "C3 absent → ALL three pathways non-functional → most severe opsonisation defect; MPGN from unregulated AP + immune complex deposition",
            "shared_feature": "Most severe encapsulated bacteria susceptibility; MPGN pathognomonic; CH50 = ZERO + AP50 = ZERO (both absent)",
            "key_ddx":        "C3 deficiency vs C1q/C2/C4A: C3 blocks ALL pathways (CH50 + AP50 both zero vs classical-only deficiency); MPGN more prominent in C3; severity highest.",
        },
        "alternative_pathway_neisseria": {
            "genes":          ["CFB", "CFD"],
            "mechanism":      "Alternative pathway C3 convertase (C3bBb) absent or unstable → impaired AP amplification → Neisseria susceptibility + Dense Deposit Disease",
            "shared_feature": "AP50 = ZERO; CH50 = NORMAL; recurrent Neisseria meningitidis; MPGN/Dense Deposit Disease; rare",
            "key_ddx":        "CFB: Factor B absent → C3bBb cannot form; Dense Deposit Disease prominent. CFD: Factor D absent → B cleavage impossible; RAREST; North Africa founder. Both: AP50/CH50 ratio diagnostic.",
        },
        "terminal_pathway_neisseria": {
            "genes":          ["C5"],
            "mechanism":      "MAC (C5b-9) cannot be initiated → Gram-negative bacteria (especially Neisseria) not lysed → recurrent meningococcal disease",
            "shared_feature": "CH50 = ZERO; AP50 = ZERO; recurrent Neisseria meningitidis PATHOGNOMONIC; eculizumab (anti-C5) curative",
            "key_ddx":        "C5 vs C6-C9 deficiency: all share Neisseria susceptibility; C5 is gatekeeper — eculizumab targets C5 and therefore TREATS C5-C9 activation but CREATES functional C5 deficiency → revaccinate. C5 deficiency: same eculizumab mechanism.",
        },
        "properdin_xlr": {
            "genes":          ["CFP"],
            "mechanism":      "Properdin absent → AP C3 convertase (C3bBb) half-life shortened 10× → AP amplification severely impaired → Neisseria uncontrolled → fulminant meningococcemia",
            "shared_feature": "ONLY XLR complement deficiency; males only; fulminant meningococcemia 30% mortality; AP50 = ZERO; family screening mandatory",
            "key_ddx":        "CFP vs all other complement deficiencies: ONLY XLR (all others AR or AD). CFP males only. Fulminant meningococcemia = most rapidly fatal presentation. Types I/II/III: all same clinical; Type I most common (absent protein).",
        },
    }

    pathognomonic_features = {
        "c1qa_lupus_rash_anti_dsdna_neg":   "C1QA/C1q deficiency: LUPUS-LIKE PHOTOSENSITIVE RASH PATHOGNOMONIC + ANTI-dsDNA ANTIBODIES NEGATIVE = key DDx from idiopathic SLE where anti-dsDNA positive; 90% risk SLE = HIGHEST in atlas; impaired apoptotic clearance mechanism",
        "c2_most_common_western":            "C2 deficiency: MOST COMMON complement deficiency in Western populations (1:20,000 North Europeans); HLA-A25-B18-DR2 haplotype null allele; PNEUMOCOCCAL VACCINES (PCV13+PPSV23) + PENICILLIN PROPHYLAXIS MANDATORY",
        "c4a_null_european":                 "C4A NULL: most common MHC null allele in Europeans (~15% carrier frequency); PARTIAL heterozygous deficiency confers 2-3× SLE risk; COMPLETE deficiency rare but high SLE risk + bacterial infections; CNV analysis required",
        "c3_mpgn_pathognomonic":             "C3 deficiency: MEMBRANOPROLIFERATIVE GLOMERULONEPHRITIS (MPGN) PATHOGNOMONIC; ALL THREE COMPLEMENT PATHWAYS BLOCKED (C3 central) = most severe opsonisation defect; Eculizumab emerging for renal protection; ALL vaccines MANDATORY",
        "cfb_dense_deposit_disease":         "CFB/Factor B deficiency: MPGN TYPE 2 DENSE DEPOSIT DISEASE characteristic; AP50 = ZERO but CH50 = NORMAL (classical intact) — this ratio diagnostic for alternative pathway-specific deficiency; Eculizumab for Dense Deposit Disease",
        "cfd_rarest_north_africa":           "CFD/Factor D deficiency: RAREST complement deficiency (<15 families worldwide); NORTH AFRICA FOUNDER EFFECT (Moroccan/North African cluster); 2× VACCINATION SCHEDULE MANDATORY; AP50 = ZERO; penicillin prophylaxis",
        "c5_neisseria_eculizumab_fda2007":   "C5 deficiency: NEISSERIA MENINGITIDIS RECURRENT = PATHOGNOMONIC for terminal complement deficiency; ECULIZUMAB (anti-C5 monoclonal; FDA-approved 2007) targets C5 = CURATIVE for terminal pathway activation; RAVULIZUMAB monthly alternative; vaccination BEFORE eculizumab MANDATORY",
        "cfp_only_xlr_fulminant":            "CFP/Properdin deficiency: ONLY X-LINKED RECESSIVE COMPLEMENT DEFICIENCY; MALES ONLY; FULMINANT MENINGOCOCCEMIA 30% MORTALITY = most rapidly fatal presentation in atlas; 2× VACCINATION (MenACWY + MenB dual-dose primary) + FAMILY SCREENING MANDATORY",
        "ch50_ap50_diagnostic_algorithm":    "CH50 + AP50 assay algorithm: CH50 = ZERO + AP50 = ZERO → classical and alternative both absent (C3, C5-C9). CH50 = ZERO + AP50 = NORMAL → classical pathway early component (C1q/C2/C4). CH50 = NORMAL + AP50 = ZERO → alternative pathway (CFB, CFD, Properdin). CH50 reduced → C3 deficiency most likely.",
        "vaccination_before_eculizumab":     "ALL complement-deficient patients: VACCINATION MANDATORY; for eculizumab-treated patients: MenACWY + MenB MUST be administered AT LEAST 2 WEEKS BEFORE starting eculizumab — eculizumab creates functional C5 deficiency and dramatically increases meningococcal risk if unvaccinated",
        "penicillin_prophylaxis_mandatory":  "C2 deficiency: lifelong daily PENICILLIN V PROPHYLAXIS MANDATORY — dramatically reduces invasive encapsulated bacterial disease; similar prophylaxis for C1q, C4A complete, C3 deficiency; amoxicillin alternative if penicillin not tolerated",
        "properdin_family_screening":        "CFP/Properdin XLR: FAMILY SCREENING MANDATORY — X-linked means maternal uncles, brothers, and male cousins at risk; screen all maternal male relatives with AP50; female carriers have 50% complement activity (usually asymptomatic but screen their sons)",
    }

    return {
        "gene_cohorts":           cohorts,
        "diagnostic_groups":      diagnostic_groups,
        "pathognomonic_features": pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: Screen with CH50 (classical pathway haemolytic) + AP50 (alternative pathway haemolytic) — zero values indicate complement pathway deficiency; confirm with serum C3, C4 levels",
            "Step 2: CH50 = ZERO + AP50 = NORMAL → early classical pathway deficiency (C1q/C2/C4) → send C1q level, C2 level, C4A/C4B levels; C1q absent → C1QA sequencing; C2/C4A gene panel",
            "Step 3: C1q absent + lupus-like rash + anti-dsDNA NEGATIVE → C1QA sequencing URGENT; hydroxychloroquine + vaccination; <40 families worldwide = very rare",
            "Step 4: C2 absent + encapsulated bacteria (S. pneumoniae/H. influenzae) + European ancestry → C2 sequencing; start PCV13 + PPSV23 + penicillin V IMMEDIATELY",
            "Step 5: CH50 = ZERO + AP50 = ZERO + C3 absent → C3 deficiency; most severe phenotype; nephrology referral (MPGN); Eculizumab consideration; ALL vaccination types MANDATORY",
            "Step 6: CH50 = NORMAL + AP50 = ZERO → alternative pathway deficiency → Factor B level + Factor D level + Properdin level",
            "Step 7: Factor B absent → CFB sequencing; AP50 = ZERO; Dense Deposit Disease workup (renal biopsy); Eculizumab for renal protection; MenACWY + MenB vaccination",
            "Step 8: Factor D absent → CFD sequencing; AP50 = ZERO; RAREST; North Africa ancestry increases prior probability; 2-dose MenACWY + MenB primary series MANDATORY",
            "Step 9: Properdin absent + MALE patient → CFP sequencing; XLR confirmation; FAMILY SCREENING immediately (maternal male relatives); 2-dose MenACWY + MenB primary series; emergency meningococcemia protocol",
            "Step 10: All terminal complement deficiencies (CH50 = ZERO from C5 level absent) → C5 deficiency → Eculizumab evaluation; VACCINATE with MenACWY + MenB at least 2 weeks BEFORE starting Eculizumab; infectious disease co-management",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "C1q Deficiency (C1QA)", "definition": "AR C1QA (1p36.12; 245aa). C1q is the recognition subunit of the C1 complex that binds antibody-antigen complexes and apoptotic cells. C1q deficiency: HIGHEST autoimmune risk of all complement deficiencies (90% develop SLE). LUPUS-LIKE PHOTOSENSITIVE RASH = PATHOGNOMONIC. KEY DDx: ANTI-dsDNA ANTIBODIES NEGATIVE in C1q deficiency SLE (unlike idiopathic SLE where anti-dsDNA positive). Mechanism: impaired apoptotic cell clearance → nuclear antigens exposed → anti-nuclear antibody formation. CH50 = ZERO; AP50 = NORMAL. Very rare (<40 families). Treatment: hydroxychloroquine + pneumococcal + meningococcal vaccination MANDATORY."},
            {"term": "C2 Deficiency", "definition": "AR C2 (6p21.33; 752aa). Most common complement deficiency in Western populations (1 in 20,000 North Europeans). C2 is the serine protease of the classical and lectin pathways that forms C3 convertase C4b2a. Common null allele on HLA-A25-B18-DR2 haplotype (28bp deletion exon 6). Primary risk: ENCAPSULATED BACTERIA (Streptococcus pneumoniae, Haemophilus influenzae) — recurrent meningitis/bacteraemia. SLE in ~30%. PNEUMOCOCCAL VACCINES (PCV13 + PPSV23) MANDATORY. PENICILLIN V PROPHYLAXIS lifelong. CH50 = ZERO; AP50 = NORMAL."},
            {"term": "C4A Deficiency", "definition": "C4A (6p21.33; 1744aa; AR for complete deficiency; heterozygous null allele for partial). C4A encodes the thioester complement protein that preferentially reacts with protein-surface antigens. C4A NULL allele: most common MHC null allele in European populations (~15% carrier). PARTIAL DEFICIENCY (heterozygous C4A*Q0): 2-3× SLE risk. COMPLETE C4A DEFICIENCY: higher SLE risk + encapsulated bacteria susceptibility. KEY DDx from C1q deficiency: anti-dsDNA may be POSITIVE in C4A deficiency. MHC copy number variation (CNV) analysis required for accurate C4A/C4B gene copy number determination."},
            {"term": "C3 Deficiency", "definition": "AR C3 (19p13.3; 1663aa). C3 is the central complement protein — ALL THREE PATHWAYS (classical, lectin, alternative) converge on C3. C3 deficiency = most severe complement deficiency for encapsulated bacteria susceptibility (ALL pathways blocked; no opsonisation at all). MEMBRANOPROLIFERATIVE GLOMERULONEPHRITIS (MPGN) PATHOGNOMONIC — mechanism: unregulated alternative pathway C3 deposition in glomerular basement membrane. CH50 = ZERO + AP50 = ZERO (both pathways absent). Eculizumab emerging for MPGN treatment. ALL vaccines MANDATORY: PCV13 + PPSV23 + MenACWY + MenB + Hib."},
            {"term": "Factor B Deficiency (CFB)", "definition": "AR CFB (6p21.33; 764aa). Factor B is the serine protease of the alternative pathway that forms C3 convertase C3bBb when cleaved by Factor D. Factor B deficiency: SELECTIVE alternative pathway absence (classical pathway intact). AP50 = ZERO; CH50 = NORMAL — this ratio distinguishes alternative pathway deficiency from classical. RECURRENT NEISSERIA MENINGITIDIS primary susceptibility. MPGN TYPE 2 (DENSE DEPOSIT DISEASE) characteristic renal complication. Eculizumab for Dense Deposit Disease. Very rare; alternative pathway function measured by AP50 assay."},
            {"term": "Factor D Deficiency (CFD)", "definition": "AR CFD (19p13.3; 259aa). Factor D is the rate-limiting enzyme of the alternative pathway present at lowest plasma concentration (~2 μg/mL). RAREST complement deficiency — fewer than 15 families worldwide. NORTH AFRICA FOUNDER EFFECT: cluster in Moroccan/North African populations. Factor D cleaves Factor B bound to C3b → generates Bb → C3bBb = AP C3 convertase. Factor D absent → AP completely non-functional. AP50 = ZERO; CH50 = NORMAL. PRIMARY: RECURRENT NEISSERIA MENINGITIDIS. 2× VACCINATION SCHEDULE MANDATORY (MenACWY + MenB both 2-dose primary series). Penicillin V prophylaxis lifelong."},
            {"term": "C5 Deficiency", "definition": "AR C5 (9q21.2; 1676aa). C5 = entry point of terminal complement pathway. C5 convertases cleave C5 → C5a (potent anaphylatoxin + chemotactic) + C5b (nucleates MAC assembly C5b-6-7-8-9). MAC essential for lysis of Gram-negative bacteria, especially Neisseria. C5 deficiency: RECURRENT NEISSERIA MENINGITIDIS PATHOGNOMONIC for terminal complement deficiency. CH50 = ZERO + AP50 = ZERO. ECULIZUMAB (anti-C5 monoclonal antibody; FDA-approved 2007 for PNH) = CURATIVE for terminal complement activation. RAVULIZUMAB (monthly dosing) alternative. CRITICAL: vaccination with MenACWY + MenB MANDATORY AT LEAST 2 WEEKS BEFORE starting eculizumab (eculizumab creates functional C5 deficiency)."},
            {"term": "Properdin Deficiency (CFP)", "definition": "XLR CFP (Xp11.23; 442aa). Properdin = ONLY POSITIVE REGULATOR of complement + ONLY XLR complement deficiency. Properdin stabilises AP C3 convertase (C3bBb) and C5 convertase 10× → without Properdin, convertases decay rapidly → AP amplification severely reduced. MALES ONLY (XLR). PATHOGNOMONIC: young boys/men with Neisseria meningitidis. FULMINANT MENINGOCOCCEMIA: 30% MORTALITY — most rapidly fatal complement deficiency. Types: Type I (absent protein; most common), Type II (low protein), Type III (dysfunctional). 2× VACCINATION MANDATORY (MenACWY + MenB dual-dose primary series; revaccinate every 5 years). FAMILY SCREENING MANDATORY (maternal male relatives at risk)."},
            {"term": "Eculizumab (Anti-C5 Mechanism)", "definition": "Eculizumab = humanised monoclonal antibody against complement C5 (blocks C5 cleavage). FDA-approved 2007 for paroxysmal nocturnal haemoglobinuria (PNH); 2011 for atypical HUS. COMPLEMENT DEFICIENCY USE: terminal pathway activation in C3 deficiency (MPGN), CFB deficiency (Dense Deposit Disease), C5 deficiency. CRITICAL CAVEAT: Eculizumab creates FUNCTIONAL C5 DEFICIENCY → dramatically increases meningococcal disease risk if unvaccinated. VACCINATION RULE: MenACWY + MenB MUST be given at least 2 WEEKS BEFORE starting eculizumab. Ravulizumab (long-acting anti-C5; monthly dosing) reduces injection frequency. Both require infectious disease co-management and meningococcal awareness."},
            {"term": "CH50 / AP50 Diagnostic Algorithm", "definition": "CH50 (classical pathway haemolytic assay): measures lysis of antibody-sensitised sheep erythrocytes by serum complement. AP50 (alternative pathway haemolytic assay): measures lysis of rabbit erythrocytes via alternative pathway. ALGORITHM: CH50 = ZERO + AP50 = ZERO → C3 or terminal pathway (C5-C9) deficiency. CH50 = ZERO + AP50 = NORMAL → classical pathway early component (C1q/C2/C4) deficiency. CH50 = NORMAL + AP50 = ZERO → alternative pathway (Factor B, Factor D, Properdin) deficiency. CH50 reduced (not zero) → C3 partial; check C3/C4 levels. Frozen/shipped samples lose complement activity → PROCESS FRESH within 2 hours of collection."},
            {"term": "Meningococcal Vaccination in Complement Deficiency", "definition": "ALL complement-deficient patients require meningococcal vaccination regardless of specific deficiency. STANDARD PROTOCOL: MenACWY (quadrivalent conjugate vaccine: serogroups A/C/W/Y) + MenB (serogroup B vaccine: Bexsero or Trumenba) — both vaccines required because MenACWY does not cover serogroup B. 2-DOSE PRIMARY SERIES: for Alternative Pathway and Properdin deficiencies (higher Neisseria risk) — 2 doses of each vaccine 2 months apart, then REVACCINATE EVERY 5 YEARS. ECULIZUMAB RULE: vaccinate at least 2 weeks before starting. PROPHYLAXIS: add penicillin V prophylaxis for highest-risk (C2, C3, Properdin deficiency). Children: follow immunisation schedule + complement-deficiency supplemental doses."},
            {"term": "Complement and Autoimmunity Mechanism", "definition": "Classical pathway early components (C1q, C2, C4) play critical roles in immune tolerance beyond infection defence: (1) APOPTOTIC CELL CLEARANCE: C1q binds apoptotic cells → facilitates phagocytosis → prevents nuclear antigen release → prevents anti-nuclear antibody formation; (2) IMMUNE COMPLEX CLEARANCE: classical pathway opsonises immune complexes → facilitates clearance by phagocytes → prevents inflammatory deposition. DEFICIENCY → impaired clearance → nuclear antigens accumulate → autoimmunity (SLE spectrum). C1q deficiency: 90% SLE (highest). C2: 30% SLE. C4A null heterozygous: 2-3× SLE risk. Mechanism explains why COMPLEMENT DEFICIENCY SLE may have NEGATIVE anti-dsDNA (C1q) vs POSITIVE (C2/C4A)."},
            {"term": "Properdin Types I/II/III", "definition": "Properdin deficiency classified into three types by protein quantity/function: TYPE I (most common): complete absence of Properdin protein on Western blot/ELISA; AP50 = ZERO; caused by null mutations (nonsense/frameshift/large deletion). TYPE II: reduced but detectable Properdin protein (10-50% of normal); AP50 severely reduced; caused by missense mutations affecting secretion/stability. TYPE III: normal or near-normal Properdin level but dysfunctional (fails to stabilise C3bBb convertase); AP50 = ZERO despite measurable protein; functional assay required for diagnosis (not just protein level). ALL TYPES: same clinical phenotype (Neisseria meningitidis susceptibility; fulminant meningococcemia risk); same management (2× vaccination + prophylaxis)."},
        ],
        "key_distinguishers": {
            "ch50_ap50_algorithm":        "CH50=ZERO + AP50=ZERO: C3 deficiency (central) or terminal C5-C9. CH50=ZERO + AP50=NORMAL: classical early (C1q/C2/C4A). CH50=NORMAL + AP50=ZERO: alternative pathway (CFB/CFD/CFP). CH50 reduced but not zero: check C3/C4 levels for partial deficiency.",
            "autoimmune_vs_infection_risk": "Highest autoimmune risk: C1QA (90% SLE). Highest infection risk: C3 (most severe, all pathways). Highest Neisseria risk: CFP/Properdin (30% fatal meningococcemia). Most common deficiency: C2 (1:20,000 Western). Rarest: CFD (<15 families).",
            "sex_pattern":                "CFP/Properdin: XLR — MALES ONLY. All others (C1QA/C2/C4A/C3/CFB/CFD/C5): AR — both sexes equally affected.",
            "anti_dsdna_distinguisher":   "C1QA deficiency SLE: anti-dsDNA NEGATIVE (distinguishes from idiopathic SLE). C2/C4A deficiency SLE: anti-dsDNA may be POSITIVE (similar to idiopathic SLE). C3 deficiency: anti-dsDNA variable.",
            "mpgn_clue":                  "C3 deficiency: MPGN pathognomonic (both classical and alternative absent → C3 deposits). CFB deficiency: Dense Deposit Disease (MPGN Type 2) characteristic (alternative pathway-specific). C3 nephropathy in complement deficiency → measure C3, Factor H, Factor I levels + AP50/CH50.",
            "xlr_complement_only":        "CFP/Properdin is the ONLY X-linked complement deficiency. Males presenting with Neisseria + family history of male meningococcal deaths in maternal lineage → CFP sequencing. AP50 assay in maternal male relatives.",
            "eculizumab_timing":          "Eculizumab requires vaccination BEFORE starting (not after). MenACWY + MenB at least 2 weeks prior. C5 deficiency treated with eculizumab = paradox: the drug creates functional C5 deficiency — must vaccinate to compensate. Same rule for Dense Deposit Disease (CFB deficiency).",
        },
        "treatment_matrix": {
            "hydroxychloroquine":          "C1QA deficiency: hydroxychloroquine for lupus-like manifestations (reduces photosensitivity, joint disease, rash). C4A deficiency SLE: hydroxychloroquine standard SLE management. C2 deficiency SLE subset: hydroxychloroquine.",
            "pneumococcal_vaccines":       "C2 deficiency: PCV13 + PPSV23 MANDATORY (both conjugate + polysaccharide). C3 deficiency: PCV13 + PPSV23 + annual revaccination. All classical pathway deficiencies: pneumococcal vaccination required. Repeat PPSV23 every 5 years.",
            "penicillin_prophylaxis":      "C2 deficiency: lifelong daily penicillin V MANDATORY. C3 deficiency: penicillin V prophylaxis recommended. C1QA: prophylaxis for Streptococcal infections. Amoxicillin if penicillin not tolerated.",
            "eculizumab":                  "C5 deficiency: Eculizumab (anti-C5; FDA 2007) curative for terminal pathway. C3 deficiency MPGN: Eculizumab emerging. CFB deficiency Dense Deposit Disease: Eculizumab. ALWAYS vaccinate MenACWY + MenB ≥2 weeks before starting.",
            "ravulizumab":                 "C5 deficiency: Ravulizumab (long-acting anti-C5; monthly dosing) reduces injection burden vs eculizumab (every 2 weeks). Same efficacy; same vaccination requirement.",
            "meningococcal_vaccination":   "ALL complement deficiencies: MenACWY + MenB MANDATORY. Alternative pathway + Properdin: 2-dose primary series each vaccine + revaccinate every 5 years. Before eculizumab: both vaccines ≥2 weeks prior. Children: follow immunisation schedule + additional doses.",
            "family_screening":            "CFP/Properdin (XLR): maternal male relatives MUST be screened (AP50 assay). C2 (HLA-A25-B18-DR2): test first-degree relatives. C3/CFB/CFD (AR): siblings 25% risk of homozygosity — test.",
            "renal_management":            "C3 deficiency MPGN: ACE inhibitor/ARB for proteinuria; Eculizumab for progressive MPGN; nephrology referral MANDATORY. CFB Dense Deposit Disease: Eculizumab; complement factor analysis (Factor H, I, B levels).",
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
