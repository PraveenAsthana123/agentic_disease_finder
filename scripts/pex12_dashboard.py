#!/usr/bin/env python3
"""PEX12 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX12 encodes Peroxin-12,
a 359-amino-acid integral peroxisomal membrane protein with a cytoplasmic C3HC4 RING finger
domain (zinc-coordinated E3 ubiquitin ligase). PEX12 is an obligate member of the
PEX2–PEX10–PEX12 RING finger E3 ubiquitin ligase complex that resides in the peroxisomal
membrane.

MECHANISM — RING FINGER E3 UBIQUITIN LIGASE (PRIMARY CATALYTIC SUBUNIT OF PEX2–PEX10–PEX12):
  PEX12 is the PRIMARY CATALYTIC subunit of the trimeric RING complex, contributing the
  strongest autonomous E3 ubiquitin ligase activity of the three members. Its extended
  cytoplasmic C-terminal RING finger domain provides a broad E2-ubiquitin-conjugating-enzyme
  binding surface:
  (1) PEX12 N-terminal two transmembrane domains (TMD1 + TMD2): anchor PEX12 in the
      peroxisomal membrane; TMD loop faces peroxisomal lumen
  (2) PEX12 C-terminal RING finger domain (cytoplasmic): primary E3 catalytic activity —
      recruits E2 ubiquitin-conjugating enzyme (UBC4/UBC5) and transfers ubiquitin to PEX5
  (3) PEX12 primarily catalyzes POLYUBIQUITINATION of PEX5 (for quality-control proteasomal
      degradation when PEX5 recycling fails); PEX2 primarily catalyzes MONOUBIQUITINATION
      of PEX5 Cys11 (the recycling signal for PEX1-PEX6 retrotranslocation)
  PEX12 LOF → one catalytic arm of the RING complex fails (PEX10 scaffold may partially
  preserve PEX2-PEX10 interaction); but net result: PEX5 ubiquitination capacity insufficient
  → PEX5 permanently trapped in the peroxisomal membrane → PTS1 protein import fails
  completely → ALL peroxisomal matrix enzyme functions simultaneously impaired.
  PEX12 loss is PHENOTYPICALLY SIMILAR to PEX2 loss (both catalytic subunits), and LESS
  structurally destabilizing than PEX10 loss (which collapses the entire trimer).
  Nevertheless, the biochemical phenotype is IDENTICAL to PEX1/PEX2/PEX6/PEX10-ZSD:
  all 6 peroxisomal pathways simultaneously fail.

LOCUS: 17q12  |  OMIM GENE: *601758  |  OMIM DISEASE: #614859 (PBD3A-ZSD), #266510 (PBD3B)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX12 mutations account for ~3–5% of all
  PBD-ZSD (similar to PEX2 at 3–5%). ~40–60 confirmed PEX12 cases worldwide as of 2026.
  PEX12 has SOME partial-function alleles: p.Gly271Asp (RING domain missense, retains partial
  E3 activity → enables IRD phenotype in compound heterozygotes with a null allele). However,
  no common FOUNDER hypomorphic allele exists (unlike PEX1-G843D or PEX6-R860W).
  Reported pathogenic variants: p.Arg180Ter (R180*, nonsense, most frequent, null);
  p.Gly271Asp (G271D, RING domain missense, partial function); p.Leu341Pro (TMD missense);
  frameshift and splice-site variants. Most PEX12 patients have null/partial compound het.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (SAME AS PEX1 / PEX2 / PEX6 / PEX10):
  PEX12 deficiency causes the SAME biochemical phenotype as PEX1/PEX2/PEX6/PEX10 because
  PEX12 E3 catalytic activity is required for PEX5 ubiquitination and retrotranslocation.
  Loss of PEX12 abolishes PEX5 cycling and all peroxisomal matrix protein import:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
  (3) Plasmalogens (ether-phospholipids) BIOSYNTHESIS FAILED → erythrocyte plasmalogens LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated in plasma/urine
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid
  DISTINGUISHING PEX12 FROM PEX1/PEX2/PEX6/PEX10 BIOCHEMICALLY: biochemically identical to
  other ZSD — only comprehensive PEX gene panel (NGS) distinguishes.

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — SIMILAR TO PEX2, FEWER IRD THAN PEX10):
  PEX12 cohorts show a severity distribution similar to PEX2-ZSD. No founder hypomorphic
  allele → spectrum skewed severe. The p.Gly271Asp partial-function allele enables some IRD:
  (1) Zellweger Syndrome (ZS, severe): null/null genotype; neonatal onset day 1–7; craniofacial
      dysmorphism; profound hypotonia; pachygyria + polymicrogyria (MRI PATHOGNOMONIC in ZS);
      neonatal seizures UNIVERSAL (100%); hepatomegaly + neonatal cholestasis; ERG absent;
      SNHL; adrenal insufficiency; skeletal stippling; fatal <6–12 months.
  (2) NALD (intermediate): null/partial compound het; onset 1st–12th month; seizures 80–90%;
      infantile spasms 35–55% + hypsarrhythmia; cholestasis; survival months–years.
  (3) IRD (attenuated): p.Gly271Asp/mild compound het; RP + SNHL + peripheral neuropathy;
      hepatomegaly mild; seizures 20–30%; adult survival in some.
  (4) Atypical: ~5%; very mild; adult-onset ataxia/RP only.
  ZS: 30% · NALD: 45% · IRD: 20% · Atypical: 5%

PEX12 CATALYTIC FUNCTION — UNIQUE ROLE vs PEX2 and PEX10:
  PEX12 is the PRIMARY CATALYTIC subunit of the RING complex (strongest autonomous E3 activity).
  PEX12 primarily drives POLYUBIQUITINATION of PEX5 (proteasomal QC pathway), while PEX2
  drives MONOUBIQUITINATION (recycling pathway). These two E3 activities are complementary:
  - Mono-Ub (PEX2): marks PEX5 Cys11 for retrotranslocation by PEX1-PEX6 AAA-ATPase
  - Poly-Ub (PEX12): marks PEX5 for proteasomal degradation when mono-Ub recycling fails
  When PEX12 is absent, PEX5 QC degradation fails; when both PEX12 and PEX2 fail (or when
  PEX10 scaffold collapses), PEX5 cannot be ubiquitinated at all → catastrophic import failure.
  This makes PEX12 LOF alone slightly less severe than PEX10 LOF (scaffold collapse), but
  functionally identical because any single RING subunit loss prevents adequate PEX5 flux.

EPILEPSY IN PEX12-ZSD:
  ZS: 100% epilepsy; neonatal seizures day 1–7; multifocal clonic + tonic + myoclonic +
  electrographic; hypsarrhythmia if surviving infantile period; EEG burst-suppression;
  universal drug-resistance; SE frequent; fatal <12M.
  NALD: 80–90% epilepsy; infantile spasms (IS) 35–55% + hypsarrhythmia; focal + generalized +
  myoclonic; moderate drug-resistance.
  IRD: 20–30% epilepsy; focal + GTCS; better AED response; phytol-restricted diet helps.
  DRE: ~65–70% of all PEX12-ZSD (similar to PEX2-ZSD — both catalytic subunits, no
  partial-function founder allele). Higher than PEX10-ZSD (60–65%) which has more IRD.

AED PHARMACOLOGY (IDENTICAL TO PEX2/PEX10-ZSD — RING COMPLEX MECHANISM DOES NOT CHANGE AED RULES):
  LEV (Levetiracetam): FIRST-LINE in all ZSD forms. No enzyme induction; no hepatotoxicity;
    IV/PO; neonatal-safe.
  ACTH: Level A for infantile spasms (IS). Preferred over VGB due to universal ZSD retinopathy.
  CLB (Clobazam): Level B adjunct. No hepatotoxicity; no CYP induction.
  LTG (Lamotrigine): Level C adjunct for NALD/IRD. Glucuronidation (not CYP) — safer hepatic.
  DHA supplementation: Level B (NALD/IRD only). Bypasses peroxisomal synthesis via intestinal
    absorption as LPC-DHA. Not useful in ZS (fatal prognosis).
  Phytol-restricted diet: Level B (IRD/Atypical). Reduces phytanic acid load.
  VPA: HIGH RISK — THREE mechanisms: hepatotoxicity + peroxisomal BO inhibition + carnitine
    depletion. POLG1 MANDATORY (CPIC Grade A). Near-absolute CI in ZS/NALD.
  VGB: HIGH RISK — additive retinopathy (ZSD retinopathy universal in ZS/NALD). ACTH preferred.
  PHT/CBZ/OXC: RELATIVE CI — CYP3A4 induction → DHA depletion + hepatic burden.
  Lorenzo's Oil: INEFFECTIVE — peroxisomal import absent; elongase inhibition does not help.
  HSCT: NOT INDICATED — ZSD is NOT inflammatory; no neuroinflammatory target for HSCT.
  ERT: NOT AVAILABLE — PEX12 is an integral peroxisomal membrane protein (not a soluble enzyme);
    protein replacement is not applicable.
"""

import random


# ── Overview ───────────────────────────────────────────────────────────────────
def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 83,
        "zs_pct": 30,
        "nald_pct": 45,
        "ird_pct": 20,
        "atypical_pct": 5,
        "drug_resistance_pct": 65,
        "on_dha_pct": 55,
        "liver_disease_pct": 75,
        "retinopathy_pct": 80,
        "dha_low_pct": 90,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "omim_gene": "601758",
        "omim_disease_zs": "614859",
        "omim_disease_nald": "266510",
        "locus": "17q12",
        "protein_size": "359 aa",
        "nbs_positive_rate": "~98% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": "p.Arg180* (null, most common); p.Gly271Asp (RING partial-function, IRD-enabling)",
        "disease_mechanism": (
            "PEX12 (359 aa, 17q12) is the PRIMARY CATALYTIC subunit of the PEX2–PEX10–PEX12 "
            "RING E3 ubiquitin ligase complex in the peroxisomal membrane. PEX12 contributes the "
            "strongest autonomous E3 activity of the three complex members, primarily driving "
            "POLYUBIQUITINATION of PEX5 (quality-control proteasomal degradation) while PEX2 "
            "drives MONOUBIQUITINATION (PEX5 recycling signal). PEX12 LOF = one catalytic arm "
            "fails; PEX10 scaffold may partially preserve PEX2-PEX10 interaction, but net result: "
            "PEX5 cannot be adequately ubiquitinated → trapped in peroxisomal membrane → "
            "ALL matrix protein import fails → same biochemical phenotype as PEX1/PEX2/PEX6/PEX10-ZSD. "
            "Biochemically identical to all other ZSD; only NGS distinguishes. "
            "~3–5% of all PBD-ZSD; ~40–60 cases worldwide 2026."
        ),
        "key_concepts": [
            "PEX12 primary catalytic role: strongest autonomous E3 activity in the RING trimer; drives PEX5 polyubiquitination (QC degradation) — complementary to PEX2 monoubiquitination (recycling)",
            "p.Gly271Asp: partial-function RING allele → some residual E3 activity → enables IRD phenotype when compound het with null allele",
            "No founder hypomorphic allele (unlike PEX1-G843D or PEX6-R860W) → most PEX12 patients have null/partial → spectrum skewed severe",
            "Biochemically identical to PEX1/PEX2/PEX6/PEX10-ZSD: VLCFA + plasmalogens LOW + DHA low + phytanic/pristanic + pipecolic + DHCA/THCA ALL abnormal",
            "PEX12 vs PEX10: PEX10 loss = scaffold collapse (entire trimer destabilized); PEX12 loss = single catalytic arm failure (PEX10 scaffold partially intact); both → identical biochemical phenotype",
            "PEX12 vs PEX2: Both catalytic subunits; PEX12 stronger autonomous E3 (polyUb/QC); PEX2 weaker isolated but drives monoUb/recycling. ZS% similar (PEX12 30% vs PEX2 30%); IRD similar (PEX12 20% vs PEX2 20%)",
            "VPA HIGH RISK (three mechanisms: hepatotoxicity + peroxisomal BO inhibition + carnitine); POLG1 MANDATORY (CPIC Grade A)",
            "VGB HIGH RISK (additive retinopathy: ZSD retinopathy universal ZS/NALD + VGB irreversible VF constriction)",
            "ACTH Level A for infantile spasms (IS): preferred over VGB due to ZSD retinopathy risk",
            "LEV first-line all ZSD forms; DHA Level B (NALD/IRD); Phytol-restricted diet Level B (IRD)",
            "Fasting EXTREME HAZARD: adipose phytanic acid mobilisation → acute neurotoxicity + seizures; IV dextrose MANDATORY during NPO",
            "PHT/CBZ/OXC RELATIVE CI (CYP3A4 → DHA depletion + hepatic) — NOT adrenal CI (unlike ABCD1 ABSOLUTE CI)",
            "Lorenzo's Oil INEFFECTIVE (peroxisomal import absent — elongase inhibition cannot compensate)",
            "HSCT NOT INDICATED (ZSD not inflammatory; no neuroinflammatory target unlike CCALD/ABCD1)",
            "No ERT (PEX12 = integral membrane RING protein; protein replacement not applicable)",
            "~40–60 cases worldwide 2026 (between PEX2 ~20–30 and PEX10 ~80–120); completes the PEX2-PEX10-PEX12 RING E3 complex triad",
        ],
        "standards": [
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum: An overview. Mol Genet Metab. 2016",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Klouwer FCC et al. Zellweger spectrum disorders: clinical overview and management approach. Orphanet J Rare Dis. 2015",
            "Fujiki Y et al. PEX genes and peroxisome biogenesis disorders. Biochim Biophys Acta Mol Cell Res. 2020",
            "Platta HW et al. The peroxisomal receptor docking complex: assembly and function in peroxisomal matrix protein import. FEBS J. 2014",
            "Okumoto K et al. Pex12p, the pex10p-binding protein of peroxisomal targeting signal type 2-importing machinery. J Biol Chem. 2000",
            "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
            "CPIC VPA–POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614859 (PBD3A-ZSD — PEX12); #266510 (PBD3B-NALD — PEX12); *601758 (PEX12 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(12359)  # PEX12 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 30, "n": 12,
            "sex": "6M/6F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + electrographic; EEG burst-suppression",
            "eeg": "Burst-suppression → multifocal spike-wave (high amplitude); poor prognosis; EEG monitoring NICU mandatory",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Arg180*/p.Arg180* or compound frameshift/nonsense); PEX12 RING catalytic arm completely absent; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Gly271Asp or splice",
            "pct": 45, "n": 18,
            "sex": "9M/9F",
            "onset_age": "Infantile (1st–12th month)",
            "seizure_risk": "80–90% — infantile spasms (IS) 35–55% + hypsarrhythmia; focal + generalized + myoclonic",
            "eeg": "Hypsarrhythmia (IS); multifocal spike-wave; modified hypsarrhythmia with burst-suppression elements",
            "mri": "Periventricular white matter signal; delayed myelination; cerebellar hypoplasia; milder gyral pattern than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Gly271Asp (RING partial) or null/splice; partial E3 activity from Gly271Asp allele; survival months to years",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Gly271Asp/mild",
            "pct": 20, "n": 8,
            "sex": "4M/4F",
            "onset_age": "Childhood / Adolescence (1–15 yr)",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD; phytol-restricted diet reduces burden",
            "eeg": "Focal epileptiform discharges; normal background in early disease; generalized spike-wave with progression",
            "mri": "RP-associated retinal thinning; mild cerebellar atrophy; white matter changes mild; no pachygyria",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Gly271Asp/hypomorphic compound het; 20–30% residual E3 activity; some peroxisomal matrix import preserved; adult survival",
        },
        {
            "name": "Atypical / Late-Onset ZSD — very mild alleles",
            "pct": 5, "n": 2,
            "sex": "1M/1F",
            "onset_age": "Adult (20–50 yr)",
            "seizure_risk": "10–15% — rare; GTCS or focal only; responsive to LEV",
            "eeg": "Normal background; rare focal discharges; incidental finding on investigation for ataxia/RP",
            "mri": "Mild cerebellar atrophy; RP-associated retinal thinning; white matter normal or minimally abnormal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles hypomorphic (very mild missense); substantial residual PEX12 E3 activity; late-onset, slowly progressive",
        },
    ]

    def rand_sex():
        return random.choice(["M", "F"])

    def rand_geno(phenotype):
        if "ZS" in phenotype:
            variants = ["p.Arg180*/p.Arg180*", "p.Arg180*/p.Leu341Pro", "p.Arg180*/c.840+2T>C", "p.Arg180*/p.Gln231fs"]
        elif "NALD" in phenotype:
            variants = ["p.Arg180*/p.Gly271Asp", "p.Arg180*/c.1022-1G>A", "p.Arg180*/p.Asp269Gly", "p.Gln155fs/p.Gly271Asp"]
        elif "IRD" in phenotype:
            variants = ["p.Gly271Asp/p.Val321Ala", "p.Gly271Asp/c.1022-1G>A", "p.Gly271Asp/p.Thr289Ile", "p.Gly271Asp/p.Ala301Val"]
        else:
            variants = ["p.Gly271Asp/p.Val321Ala", "p.Gly271Asp/p.Arg275Gln"]
        return random.choice(variants)

    def rand_aed(phenotype):
        if "ZS" in phenotype:
            return random.choice(["LEV", "Phenobarbital", "LEV + Phenobarbital"])
        elif "NALD" in phenotype:
            return random.choice(["LEV", "LEV + CLB", "LEV + ACTH", "ACTH + LEV"])
        else:
            return random.choice(["LEV", "LEV + LTG", "LTG", "LEV + CLB"])

    def rand_response(phenotype):
        if "ZS" in phenotype:
            return "Drug-resistant"
        elif "NALD" in phenotype:
            return random.choice(["Drug-resistant", "Drug-resistant", "Partially controlled"])
        else:
            return random.choice(["Partially controlled", "Well controlled", "Well controlled"])

    patients = []
    etiol_map = [
        ("ZS (Zellweger-Severe)", 12),
        ("NALD (Neonatal-ALD-Intermediate)", 18),
        ("IRD (Infantile-Refsum-Attenuated)", 8),
        ("Atypical / Late-Onset ZSD", 2),
    ]
    pid = 1
    for phenotype, count in etiol_map:
        for _ in range(count):
            sex = rand_sex()
            geno = rand_geno(phenotype)
            aed = rand_aed(phenotype)
            resp = rand_response(phenotype)
            has_sz = "ZS" in phenotype or "NALD" in phenotype or random.random() < 0.25
            liver = "ZS" in phenotype or "NALD" in phenotype or random.random() < 0.15
            on_dha = "NALD" in phenotype or "IRD" in phenotype or "Atypical" in phenotype
            patients.append({
                "patient_id": f"PEX12-{pid:03d}",
                "phenotype": phenotype.split(" — ")[0],
                "sex": sex,
                "genotype": geno,
                "liver_disease": liver,
                "has_seizures": has_sz,
                "primary_aed": aed if has_sz else None,
                "drug_response": resp if has_sz else None,
                "on_dha": on_dha,
            })
            pid += 1

    seizure_types = [
        {"type": "Neonatal multifocal clonic (ZS)", "pct": 30, "eeg": "Burst-suppression → multifocal spike; high amplitude; bilateral asynchronous; EEG monitoring NICU mandatory"},
        {"type": "Infantile spasms / West syndrome (NALD)", "pct": 45, "eeg": "Hypsarrhythmia → modified hypsarrhythmia; ACTH Level A first-line; VGB HIGH RISK (avoid)"},
        {"type": "Focal seizures bilateral spread (NALD/IRD)", "pct": 22, "eeg": "Temporal/parietal focal spike-wave; post-ictal slowing; lateralised in IRD"},
        {"type": "Myoclonic seizures (NALD)", "pct": 15, "eeg": "Generalised polyspike-wave ≥3 Hz; photosensitive in some; CBZ/PHT HIGH RISK (myoclonic exacerbation)"},
        {"type": "GTCS (IRD/Atypical)", "pct": 12, "eeg": "Generalised spike-wave; good post-ictal recovery in IRD; LEV + LTG effective"},
        {"type": "Absence / atypical absence (rare, IRD)", "pct": 5, "eeg": "2.5–3.5 Hz generalised spike-wave; ETX considered Level C in IRD"},
    ]

    triggers = [
        {"trigger": "Fever / intercurrent infection", "pct": 65, "note": "Universal ZSD trigger — fever lowers seizure threshold + increases metabolic demand → phytanic mobilisation"},
        {"trigger": "Subtherapeutic AED levels", "pct": 52, "note": "Common in NALD/IRD — growth spurts alter pharmacokinetics; TDM monitoring quarterly"},
        {"trigger": "Fasting / prolonged NPO", "pct": 48, "note": "EXTREME HAZARD — adipose phytanic acid mobilisation → acute neurotoxicity + seizures; IV dextrose MANDATORY"},
        {"trigger": "Sleep deprivation", "pct": 40, "note": "Worsens electrocortical excitability; melatonin (0.5–3 mg) adjunct in NALD/IRD with sleep disruption"},
        {"trigger": "Metabolic decompensation / catabolism", "pct": 35, "note": "Intercurrent illness, surgery, missed feeds → phytanic surge + VLCFA accumulation worsens acutely"},
        {"trigger": "Missed AED dose", "pct": 30, "note": "Non-adherence frequent in long-term NALD/IRD; simplified regimens (LEV BID) improve adherence"},
        {"trigger": "Enzyme-inducing AEDs (PHT/CBZ/OXC)", "pct": 22, "note": "CYP3A4 induction → DHA depletion (already low in ZSD) + hepatic burden; RELATIVE CI; replace with LEV"},
        {"trigger": "Anaesthesia / surgical stress", "pct": 18, "note": "NPO + surgical stress → phytanic surge; pre-op: PT/INR + LFT + platelet + IV Vitamin K; IV dextrose mandatory perioperatively"},
    ]

    monitoring = [
        "VLCFA plasma panel (C26:0, C26:0/C22:0 ratio) — baseline + q6M (disease tracking + AED-induced metabolic change)",
        "Erythrocyte plasmalogens — baseline + q6M (PEX12 RING complex function marker; LOW in all ZSD)",
        "DHA plasma level — baseline + q6M (supplement adequacy; target >4% total fatty acids in NALD/IRD)",
        "Plasma phytanic acid — q6M in IRD/Atypical; q3M in NALD (diet compliance + fasting risk)",
        "LFT + GGT + bilirubin (conjugated) — q3M in ZS/NALD; q6M in IRD (cholestatic liver disease monitoring)",
        "PT/INR + platelet count — q6M in ZS/NALD (cholestatic coagulopathy; pre-op MANDATORY)",
        "Plasma carnitine + acylcarnitine profile — q6M (carnitine depletion risk, especially if VPA inadvertently started)",
        "POLG1 sequencing — ONCE before any VPA consideration (CPIC Grade A; MANDATORY)",
        "Electroretinogram (ERG) — baseline + annual (universal retinopathy ZS/NALD; preserve remaining IRD visual field)",
        "Visual field testing (Goldmann perimetry) — q12M in IRD (preserve from VGB, RP progression)",
        "SNHL audiometry (ABR neonatal → PTA in IRD) — baseline + q12M (universal SNHL in ZS/NALD)",
        "EEG — NICU continuous in ZS; q3M in active NALD (IS monitoring); q6M in NALD seizure-free; q12M in IRD",
        "Brain MRI — baseline (pachygyria ZS; white matter NALD); q12–24M in NALD/IRD",
        "Developmental / neuropsychological assessment — q12M in NALD/IRD (cognition, language, adaptive function)",
    ]

    thresholds = [
        {"parameter": "C26:0-lyso-PC (NBS DBS)", "threshold": ">0.10 μmol/L", "action": "Refer immediately to metabolic specialist; confirm with plasma VLCFA + plasmalogens; NGS PEX gene panel"},
        {"parameter": "Plasma C26:0/C22:0 ratio", "threshold": ">0.020", "action": "Diagnostic for ZSD biochemically; initiate NGS PEX panel; PEX12 among first-tier genes"},
        {"parameter": "Erythrocyte plasmalogens", "threshold": "<50% age-matched reference", "action": "Confirms PBD-ZSD biochemical phenotype (vs ABCD1 where plasmalogens NORMAL)"},
        {"parameter": "Plasma DHA", "threshold": "<4% total FA in NALD/IRD", "action": "Initiate DHA supplementation (LPC-DHA 100–200 mg/day in NALD; Level B); recheck in 3M"},
        {"parameter": "Plasma phytanic acid", "threshold": ">5 μmol/L (IRD)", "action": "Intensify phytol-restricted diet; exclude fasting episodes; consider plasmapheresis if acutely elevated"},
        {"parameter": "Plasma carnitine (free)", "threshold": "<25 μmol/L", "action": "Carnitine supplementation; VPA CONTRAINDICATED; review all medications for carnitine-depleting agents"},
        {"parameter": "PT/INR (cholestatic)", "threshold": "INR > 1.5", "action": "IV Vitamin K MANDATORY pre-surgery; hepatology consult; avoid hepatotoxic AEDs (VPA)"},
        {"parameter": "VPA drug level (if mistakenly started)", "threshold": "Any detectable level", "action": "STOP VPA IMMEDIATELY; check POLG1; supplement carnitine; monitor LFT + NH3 closely"},
    ]

    lifecycle = [
        {
            "stage": "Stage 1 — Neonatal/NICU (0–28 days, ZS only)",
            "features": "Profound hypotonia; neonatal seizures day 1–7 (multifocal clonic + tonic); EEG burst-suppression; cholestatic jaundice; craniofacial dysmorphism; ERG absent; SNHL on ABR; IV access critical; MRI shows pachygyria",
            "action": "LEV IV first-line; Phenobarbital if refractory; IV dextrose continuous; VLCFA + plasmalogens STAT; NGS PEX panel STAT; palliative care discussion with family if ZS confirmed; POLG1 sequencing mandatory",
        },
        {
            "stage": "Stage 2 — Early Infantile (1–12 months, NALD dominant)",
            "features": "Infantile spasms (IS) with hypsarrhythmia (35–55%); psychomotor regression; cholestasis evolving; SNHL; visual impairment from RP; nasogastric feeds common; hypotonia",
            "action": "ACTH Level A (preferred over VGB for IS — avoid VGB due to retinopathy); LEV adjunct; DHA supplementation Level B; ophthalmology + audiology; DBS NBS confirms biochemistry if not already done",
        },
        {
            "stage": "Stage 3 — Late Infantile/Toddler (1–3 yr, NALD/IRD)",
            "features": "Seizure evolution from IS to focal + myoclonic; developmental plateau or regression; RP progression; SNHL progression; ataxia emerging in IRD; phytol-restricted diet starts in IRD",
            "action": "Optimise LEV ± CLB adjunct; phytol-restricted diet (Level B in IRD); DHA supplementation continue; VPA NEVER; ERG + visual field + ABR annual; NGS panel confirms PEX12 identity vs other ZSD genes",
        },
        {
            "stage": "Stage 4 — Preschool/School Age (3–12 yr, IRD/Atypical)",
            "features": "Focal seizures ± GTCS; intellectual disability (mild–moderate in IRD); ataxia prominent; RP → tunnel vision; SNHL stable; hepatomegaly mild; peripheral neuropathy emerging",
            "action": "LEV ± LTG Level C; strict phytol-restricted diet; annual VLCFA + phytanic + DHA; neuropsychological support; adaptive aids for visual + hearing impairment; transition planning for puberty",
        },
        {
            "stage": "Stage 5 — Adolescence/Adulthood (12+ yr, IRD/Atypical only)",
            "features": "Seizure frequency stable or decreasing in well-controlled IRD; RP → legal blindness common; SNHL; progressive ataxia; peripheral neuropathy; hypogonadism in some; psychiatric comorbidity (anxiety, depression)",
            "action": "Long-term LEV ± LTG; continue phytol-restricted diet; annual metabolic review; ophthalmology + audiology; psychosocial support; genetic counselling for siblings/offspring (AR — 25% recurrence); no ERT/HSCT",
        },
        {
            "stage": "Stage 6 — Surgical / Anaesthesia Intercept (ANY stage, ZSD-wide)",
            "features": "Peri-operative fasting EXTREME HAZARD: adipose phytanic acid mobilisation → acute phytanic surge → neonatal-equivalent neurotoxicity + seizures; cholestatic coagulopathy risk",
            "action": "IV dextrose MANDATORY from first NPO moment; IV Vitamin K pre-op; PT/INR + LFT + platelet pre-op; VPA ABSOLUTE CI perioperatively; IV LEV replaces oral; anaesthetics team briefing MANDATORY",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "AED — FIRST-LINE (all ZSD forms)",
            "evidence": "Level A (expert consensus — multiple ZSD cohorts including PEX12)",
            "dose": "Neonatal: 20–30 mg/kg/day IV BID; Infantile/child: 30–60 mg/kg/day PO BID; Adult: 1000–3000 mg/day BID",
            "moa": "SV2A synaptic vesicle modulation; no hepatotoxicity; no CYP induction; no carnitine depletion; IV formulation available — critical for neonatal ZS and perioperative use",
            "monitoring": "Renal function (dose-adjust in CKD); ADHD-like side effects in NALD/IRD — behavioural monitoring",
            "ci": None,
        },
        {
            "drug": "ACTH (Adrenocorticotropic Hormone)",
            "class": "AED — Level A for Infantile Spasms (IS)",
            "evidence": "Level A (UKISS, multinational ZSD IS protocols; PREFERRED over VGB in ZSD — retinopathy risk)",
            "dose": "20–40 IU/day IM for 2–4 weeks; taper over 4–6 weeks; monitor BP + glucose + electrolytes",
            "moa": "Melanocortin receptor pathway; down-regulates CRF (seizure-facilitating); ACTH preferred over VGB in ZSD to avoid additive retinopathy",
            "monitoring": "BP, glucose, electrolytes, infection risk; adrenal recovery post-taper",
            "ci": "Active infection; severe hypertension; do NOT use VGB instead in ZSD",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "AED — Level B adjunct",
            "evidence": "Level B (expert consensus; low hepatotoxicity profile; effective in focal/myoclonic)",
            "dose": "Child: 0.1–0.3 mg/kg/day PO BID; Adult: 10–30 mg/day BID; titrate slowly",
            "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine); no CYP induction; no carnitine effect",
            "monitoring": "Sedation; tolerance (dose escalation may be needed); respiratory in neonates",
            "ci": "None specific in ZSD; avoid in severe hepatic failure",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "AED — Level C (NALD/IRD adjunct)",
            "evidence": "Level C (case reports; safer hepatic profile via glucuronidation not CYP)",
            "dose": "Start 0.15 mg/kg/day; titrate slowly over 8 weeks to 3–5 mg/kg/day; adult 100–300 mg/day",
            "moa": "Sodium channel blocker; glucuronidation (UGT1A4) — avoids CYP hepatic burden vs PHT/CBZ; useful for focal + GTCS in IRD",
            "monitoring": "Stevens-Johnson syndrome (slow titration mandatory); interaction with VPA (prohibited in ZSD)",
            "ci": "VPA co-administration (prohibited); slow titration mandatory",
        },
        {
            "drug": "DHA (Docosahexaenoic Acid) supplementation",
            "class": "Disease-Modifying — Level B (NALD/IRD)",
            "evidence": "Level B (Martinez 1996; van Grunsven 1999; Restoring retinal DHA → slows RP; not useful in ZS)",
            "dose": "100–200 mg/day as LPC-DHA (lysophosphatidylcholine-DHA); oral; intestinal absorption bypasses peroxisomal synthesis block",
            "moa": "LPC-DHA absorbed intestinally → incorporated into neural membranes; bypasses peroxisomal delta-6 elongase failure; improves retinal function + myelin in NALD/IRD",
            "monitoring": "Plasma DHA level q3M; target >4% total FA; ERG q6M (visual outcome tracking)",
            "ci": "Not indicated in ZS (fatal prognosis; no meaningful benefit); use LPC-DHA formulation (not ethyl ester)",
        },
        {
            "drug": "Phytol-restricted diet (low phytanic acid)",
            "class": "Disease-Modifying — Level B (IRD/Atypical)",
            "evidence": "Level B (Steinberg 2006; clinical consensus; reduces phytanic accumulation + seizure burden in IRD)",
            "dose": "Dietary restriction of phytol-rich foods (green vegetables, dairy, red meat, ruminant fat); dietitian supervised; maintain adequate caloric intake",
            "moa": "Phytol is the precursor to phytanic acid via dietary intake; restriction reduces phytanic accumulation → fewer phytanic-triggered seizures and neuropathy progression",
            "monitoring": "Plasma phytanic acid q6M; nutritional status (albumin, vitamin levels); dietitian review q6M",
            "ci": "No absolute CI; ensure caloric adequacy (fasting EXTREME HAZARD — IV dextrose mandatory during illness/NPO)",
        },
        {
            "drug": "Cholic acid / UDCA (ursodeoxycholic acid)",
            "class": "Supportive — Level C (cholestatic liver disease)",
            "evidence": "Level C (expert consensus; reduces DHCA/THCA-driven cholestasis; no seizure effect)",
            "dose": "UDCA: 10–15 mg/kg/day PO BID; Cholic acid (CA): where available, replaces bile acid synthesis end-product",
            "moa": "UDCA: hepatoprotective, reduces cholestasis; CA: replaces deficient primary bile acids; reduces upstream toxic DHCA/THCA accumulation",
            "monitoring": "LFT q3M; PT/INR q6M; bilirubin (conjugated/unconjugated) q3M",
            "ci": "None specific; hepatology input recommended",
        },
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA / Valproic Acid / Depakene / Epilim)",
            "level": "HIGH RISK (near-Absolute CI in ZS/NALD) — THREE independent mechanisms",
            "reason": "(1) Hepatotoxicity: ZS/NALD already have cholestatic liver disease (DHCA/THCA-driven); VPA hepatotoxicity is ADDITIVE and unpredictable; (2) Peroxisomal beta-oxidation inhibition: VPA is itself a substrate of peroxisomal BO; inhibits VLCFA oxidation → worsens VLCFA accumulation (C26:0); (3) Carnitine depletion: VPA sequesters carnitine → secondary carnitine deficiency (amplified in ZSD due to impaired peroxisomal fatty acid metabolism). POLG1 sequencing MANDATORY before ANY VPA consideration (CPIC Grade A).",
            "alternative": "LEV IV/PO (first-line, all ZSD forms). If IS: ACTH (Level A, preferred over VGB). If focal: LEV + CLB or LEV + LTG (IRD only). NEVER use VPA as first-line or adjunct in PEX12-ZSD.",
        },
        {
            "drug": "Vigabatrin (VGB / Sabril)",
            "level": "HIGH RISK (avoid in IRD; near-CI in ZS/NALD)",
            "reason": "ZSD causes retinopathy (universal ZS/NALD; 20–30% IRD from RP). VGB irreversibly constricts visual fields (concentric VF loss) in 30–50% users. In ZSD, VGB-induced VF loss is ADDITIVE to existing RP-driven VF loss → catastrophic irreversible blindness. ACTH (Level A) is the preferred IS treatment in ALL ZSD forms specifically to avoid VGB. Avoid VGB even in IRD where existing RP may not yet be severe — constriction is irreversible.",
            "alternative": "ACTH (Level A) for infantile spasms. LEV + CLB for focal/myoclonic. Never use VGB first-line in ZSD.",
        },
        {
            "drug": "Fasting / Prolonged NPO / Caloric Restriction",
            "level": "EXTREME HAZARD (ALL ZSD forms, ALL ages)",
            "reason": "Fasting triggers adipose tissue release of stored phytanic acid (accumulated over years due to failed alpha-oxidation). Acute phytanic surge → direct neurotoxicity + acute seizures + cardiac arrhythmia (phytanic is membrane-active). This can precipitate acute decompensation even in previously stable IRD patients. ANY NPO period (surgery, illness, vomiting) = MANDATORY IV dextrose (10% dextrose at maintenance rate).",
            "alternative": "IV 10% dextrose MANDATORY from first NPO moment. Pre-op briefing of anaesthetics team. IV Vitamin K (cholestatic coagulopathy). PT/INR + LFT + platelet pre-surgery. Resume enteral feeds as soon as clinically safe.",
        },
        {
            "drug": "Phenytoin / Carbamazepine / Oxcarbazepine (PHT / CBZ / OXC)",
            "level": "RELATIVE CI (use with caution; avoid as first-line)",
            "reason": "CYP3A4 induction → (1) accelerates DHA catabolism (already deficient in ZSD → worsens DHA deficit → further RP/neuropathy) + (2) increases hepatic microsomal burden (cholestatic liver disease in ZS/NALD). NOT adrenal crisis (unlike ABCD1 where PHT/CBZ = ABSOLUTE CI). May be used if no alternative in specific focal seizures (IRD) with close DHA + LFT monitoring.",
            "alternative": "LEV first-line (no CYP induction); LTG (glucuronidation — safer hepatic) Level C in IRD; CLB adjunct. Replace PHT/CBZ with LEV as soon as diagnostically confirmed.",
        },
        {
            "drug": "Lorenzo's Oil (oleic acid + erucic acid mixture)",
            "level": "INEFFECTIVE (do NOT use in ZSD)",
            "reason": "Lorenzo's Oil inhibits VLCFA elongation (C22→C24→C26:0) → reduces VLCFA plasma levels in X-ALD (ABCD1). However, in ZSD the mechanism is different: peroxisomal IMPORT is absent (not just VLCFA elongation). Lorenzo's Oil cannot restore PEX5 import or peroxisomal matrix enzyme function. No clinical benefit in PEX12-ZSD. May cause side effects (thrombocytopenia) without benefit.",
            "alternative": "DHA supplementation Level B (NALD/IRD). Phytol-restricted diet Level B (IRD). No peroxisomal pathway-specific pharmacotherapy available as of 2026.",
        },
        {
            "drug": "HSCT (Hematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED (PEX12-ZSD)",
            "reason": "HSCT benefits conditions with a neuroinflammatory component where donor-derived microglia can halt progression (e.g., cerebral X-ALD [ABCD1] — HSCT is standard of care in early CCALD). PEX12-ZSD is NOT primarily inflammatory; the pathology is metabolic (peroxisomal import failure). Donor microglia cannot restore PEX12 RING E3 activity in host neurons. HSCT carries 5–15% transplant-related mortality — unacceptable risk-benefit ratio in non-inflammatory ZSD.",
            "alternative": "Supportive care: DHA supplementation, phytol-restricted diet, seizure management with LEV. Gene therapy research in preclinical stage for PBD-ZSD.",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE (PEX12 = integral membrane RING protein)",
            "reason": "ERT is feasible for soluble lysosomal enzymes (e.g., imiglucerase for Gaucher, laronidase for MPS-I) that can be endocytosed via mannose-6-phosphate receptor. PEX12 is an integral peroxisomal MEMBRANE protein (two TMDs + cytoplasmic RING domain) — not a soluble secreted enzyme. ERT cannot deliver a membrane-anchored E3 ligase to its intracellular membrane target. No ERT mechanism is applicable.",
            "alternative": "Gene therapy (AAV-PEX12) in preclinical research; mRNA delivery experimental. No clinically available ERT as of 2026.",
        },
    ]

    return {
        "etiologies": etiologies,
        "patients": patients,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "monitoring": monitoring,
        "thresholds": thresholds,
        "lifecycle": lifecycle,
        "treatments": treatments,
        "contraindications": contraindications,
    }


# ── Definitions ───────────────────────────────────────────────────────────────
def get_definitions():
    key_concepts = [
        "PEX12 (359 aa, 17q12): integral peroxisomal membrane protein with two TMDs and cytoplasmic C3HC4 RING finger domain; PRIMARY CATALYTIC E3 subunit of the PEX2–PEX10–PEX12 RING complex",
        "PEX12 primary catalytic role: drives POLYUBIQUITINATION of PEX5 (quality-control proteasomal degradation) — complementary to PEX2 monoubiquitination (recycling signal for PEX1-PEX6 retrotranslocation)",
        "PEX2–PEX10–PEX12 RING complex: PEX10 = scaffold/bridge; PEX2 = catalytic (monoUb/recycling); PEX12 = primary catalytic (polyUb/QC); all three required for adequate PEX5 flux",
        "PEX5 entrapment: when PEX12 (or any RING complex member) fails, PEX5 cannot be ubiquitinated → permanently trapped in peroxisomal membrane → ALL PTS1-targeted matrix protein import ceases",
        "Biochemical identity: PEX12-ZSD is biochemically IDENTICAL to PEX1/PEX2/PEX6/PEX10-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
        "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX12; NORMAL in ABCD1 (X-ALD); this single test separates the two diagnoses",
        "DHA synthesis failure → neuronal migration defects: DHA is synthesised in peroxisomes; PEX12 failure → DHA synthesis fails → DHA low → pachygyria + polymicrogyria (ZS, PATHOGNOMONIC on MRI) + RP + neuropathy",
        "Phenotypic spectrum: ZS 30% (null/null) · NALD 45% (null/partial) · IRD 20% (Gly271Asp/mild) · Atypical 5%; similar to PEX2-ZSD (both catalytic, no founder hypomorphic allele)",
        "p.Arg180* (R180*): most common truncating null variant; associated with ZS and NALD in compound het with partial allele",
        "p.Gly271Asp (G271D): partial-function RING missense; retains partial E3 catalytic activity → IRD phenotype when compound het with null allele; analogous role to PEX10-p.Ile332Thr but distinct position",
        "VPA triple CI mechanism in ZSD: (1) direct hepatotoxicity in cholestatic liver (DHCA/THCA-driven) + (2) peroxisomal BO inhibition (worsens VLCFA accumulation — VPA is peroxisomal BO substrate) + (3) carnitine depletion; POLG1 MANDATORY (CPIC Grade A) before ANY VPA",
        "VGB additive retinopathy mechanism: ZSD causes retinopathy (ERG absent ZS/NALD; RP in IRD) + VGB causes irreversible visual field constriction (GABA accumulation in retinal neurons); together = catastrophic blindness; ACTH preferred for IS",
        "Fasting EXTREME HAZARD mechanism: adipose tissue stores phytanic acid over years; fasting → adipose lipolysis → acute phytanic acid release into circulation → direct membrane neurotoxicity → seizures + cardiac arrhythmia; IV dextrose prevents catabolism",
        "PHT/CBZ/OXC RELATIVE CI (NOT absolute): CYP3A4 induction depletes DHA (already low); increases hepatic burden. DIFFERENT from ABCD1 where PHT/CBZ = ABSOLUTE CI (adrenal crisis mechanism). In ZSD: monitor DHA + LFT if unavoidable; replace with LEV",
        "Lorenzo's Oil mechanism mismatch: LOO inhibits VLCFA elongase (C22→C26 step) → works in X-ALD (ABCD1) where import is intact but VLCFA accumulates because ABCD1 peroxisomal transporter is absent. In ZSD, peroxisomal IMPORT is absent — VLCFA accumulates because it CANNOT ENTER peroxisomes. LOO cannot restore import → INEFFECTIVE in all ZSD including PEX12",
        "NBS: C26:0-lyso-PC on DBS detects ZSD biochemically (~98% sensitivity ZS/NALD); confirms ZSD group but cannot identify PEX12 specifically — NGS PEX gene panel mandatory for gene-level diagnosis",
    ]

    diagnostic_algorithm = [
        "Step 1 — NBS flag: C26:0-lyso-PC elevated on DBS → refer to metabolic specialist STAT",
        "Step 2 — Plasma VLCFA panel: C26:0, C26:0/C22:0 ratio elevated → confirms ZSD biochemical group",
        "Step 3 — Erythrocyte plasmalogens: severely LOW → confirms PBD-ZSD (distinguishes from ABCD1 where plasmalogens NORMAL)",
        "Step 4 — Plasma phytanic + pristanic: both elevated (alpha-oxidation failed — confirms PTS1 + alpha-ox pathway failure together)",
        "Step 5 — Plasma/urine pipecolic acid: elevated → confirms peroxisomal import failure (pipecolic catabolism is peroxisomal)",
        "Step 6 — Plasma DHA: LOW → confirms DHA synthesis failure; retinopathy + migration risk",
        "Step 7 — Bile acids (DHCA, THCA): elevated → confirms bile acid synthesis failure → cholestatic liver disease risk",
        "Step 8 — MRI brain: pachygyria + polymicrogyria in ZS (PATHOGNOMONIC); periventricular WM signal in NALD; mild/normal in IRD",
        "Step 9 — ERG + ophthalmology: ERG absent/severely reduced (ZS/NALD); RP in IRD (Leber-like); visual fields (Goldmann)",
        "Step 10 — Comprehensive NGS PEX gene panel (PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26): identifies PEX12 biallelic pathogenic variants; confirms gene-level diagnosis",
        "Step 11 — POLG1 sequencing: MANDATORY before considering any VPA (CPIC Grade A); simultaneously eliminates mitochondrial aetiology",
        "Step 12 — Genetic counselling: AR inheritance (25% recurrence risk per pregnancy); prenatal diagnosis available (PEX12 targeted sequencing in CVS/amnio); carrier testing for parents/siblings",
    ]

    pharmacological_distinctions = [
        "LEV vs PHT/CBZ/OXC: LEV = no CYP induction, no hepatotoxicity, IV available → FIRST-LINE ZSD; PHT/CBZ/OXC = CYP3A4 induction → DHA depletion + hepatic burden → RELATIVE CI in ZSD",
        "VPA vs LEV: VPA = THREE CI mechanisms in ZSD (hepatotoxicity + peroxisomal BO inhibition + carnitine depletion) → HIGH RISK near-absolute CI; LEV = none of these → always preferred",
        "ACTH vs VGB for IS: In ZSD, ACTH Level A (preferred) vs VGB HIGH RISK (retinopathy additive). Reverse of some non-ZSD IS protocols where VGB may be considered. Never use VGB first-line in ZSD",
        "DHA supplementation: LPC-DHA (lysophosphatidylcholine-DHA) form bypasses peroxisomal synthesis via intestinal absorption → restores plasma/tissue DHA. Ethyl ester form less effective (requires peroxisomal processing). Use LPC-DHA specifically",
        "Phytol-restricted diet (IRD/Atypical) vs no dietary restriction (ZS): In ZS (fatal <12M), dietary restriction has no meaningful benefit. In IRD/Atypical (long survival), phytol restriction reduces phytanic accumulation → fewer phytanic-triggered seizures + slower neuropathy",
        "Lorenzo's Oil in ABCD1 vs ZSD: LOO works in X-ALD (ABCD1) because ABCD1 transport is absent but peroxisomal IMPORT is intact — elongase inhibition reduces VLCFA substrate reaching peroxisomes. In ZSD, import itself is absent → VLCFA cannot enter peroxisomes regardless of elongase inhibition → INEFFECTIVE",
        "HSCT in ABCD1 vs ZSD: HSCT standard of care in early CCALD (cerebral X-ALD / ABCD1) because donor microglia halt neuroinflammation. ZSD is NOT inflammatory → HSCT confers transplant-related mortality risk with zero mechanistic benefit",
        "PHT/CBZ ABSOLUTE CI (ABCD1) vs RELATIVE CI (PEX12-ZSD): In ABCD1, PHT/CBZ trigger adrenal crisis (adrenal insufficiency universal in X-ALD + CYP enzyme induction). In PEX12-ZSD, adrenal insufficiency is NOT a universal feature (adrenal hypoplasia in ZS only) → PHT/CBZ = RELATIVE CI (DHA/hepatic), not adrenal CI",
        "VGB ABSOLUTE CI (PHYH-Refsum, ALDH5A1, MCOLN1) vs HIGH RISK (PEX12-ZSD): In PHYH/Refsum, VGB CI = RP 95% + VGB VF constriction. In PEX12-ZSD, similar but classified HIGH RISK (not absolute) because in Atypical/IRD, balance can occasionally shift; still avoid in practice. Always prefer ACTH for IS in ZSD",
        "VPA POLG1 mandatory (CPIC Grade A): VPA CI in mitochondrial disease (POLG1) due to mitochondrial depletion mechanism (separate from ZSD mechanisms). In ZSD, POLG1 exclusion is MANDATORY because ZSD patients may also harbour POLG1 variants (2 different rare gene mutations); if POLG1 variant present + ZSD → catastrophic risk",
        "Carnitine supplementation: indicated if free carnitine < 25 μmol/L (may occur in PEX12-ZSD due to impaired peroxisomal fatty acid metabolism); L-carnitine 50–100 mg/kg/day; important to avoid further depletion — especially if VPA was inadvertently started",
        "IV Vitamin K perioperative: cholestatic liver disease (DHCA/THCA-driven) → fat-soluble vitamin K malabsorption → coagulopathy (↑PT/INR); IV Vitamin K 1–2 mg/kg (max 10 mg) pre-surgery; check PT/INR + platelet pre-op MANDATORY in PEX12-ZSD",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX2-ZSD (3–5% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX2 drives monoUb/recycling (vs PEX12 polyUb/QC); both catalytic subunits. ZS% similar (both 30%); IRD similar (PEX12 20% = PEX2 20%). PEX2 has no partial-function founder allele (p.Arg119* most common null); PEX12 has p.Gly271Asp partial. 8q21.13 vs 17q12. NGS distinguishes.",
        },
        {
            "condition": "PEX10-ZSD (5–7% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX10 = scaffold/bridge subunit (coiled-coil tethers PEX2-PEX12); PEX10 LOF = entire RING trimer collapses. PEX10 has p.Ile332Thr partial-function → 25% IRD (vs PEX12 20% IRD). PEX10 more common (~80–120 cases) vs PEX12 (~40–60). 1p36.32 vs 17q12. NGS distinguishes.",
        },
        {
            "condition": "PEX1-ZSD (65% of all PBD)",
            "distinction": "Biochemically IDENTICAL. Most common PBD gene. PEX1 encodes the D1 AAA-ATPase subunit (PEX5 retrotranslocation motor). p.Gly843Asp founder allele (30% European) → attenuated (IRD 30%). PEX12 has no founder allele → more severe. 7q21.2 vs 17q12. NGS distinguishes.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 encodes D2 AAA-ATPase partner to PEX1. p.Arg860Trp founder allele → IRD 45% (most attenuated ZSD). PEX12 has less IRD (20%). 6p21.1 vs 17q12. NGS distinguishes.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy — X-ALD)",
            "distinction": "VLCFA elevated in BOTH — but ABCD1: plasmalogens (RBC) NORMAL + DHA NORMAL + adrenal insufficiency (universal, X-linked males). No pachygyria. PHT/CBZ = ABSOLUTE CI in ABCD1 (adrenal crisis) vs RELATIVE CI in PEX12-ZSD. Lorenzo's Oil EFFECTIVE in ABCD1 (intact import); INEFFECTIVE in ZSD. HSCT standard of care in CCALD (inflammatory); NOT in ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC acid severely elevated (SOLE elevated peroxisomal metabolite); VLCFA NORMAL (key distinction vs PEX12); plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. VGB ABSOLUTE CI (RP 95%). Phytol-restricted diet Level A GOLD STANDARD. No VLCFA elevation.",
        },
        {
            "condition": "RCDP / PEX7 (Rhizomelic Chondrodysplasia Punctata Type 1)",
            "distinction": "Plasmalogens (RBC) severely LOW (like ZSD) BUT VLCFA NORMAL + phytanic elevated + pristanic NORMAL + DHA NORMAL + pipecolic NORMAL. Rhizomelia (short proximal limbs) PATHOGNOMONIC. Stippled epiphyses on X-ray. Only PTS2 pathway affected (PEX7 = PTS2 receptor). No pachygyria.",
        },
        {
            "condition": "Mitochondrial / POLG1 Epilepsy (Alpers disease)",
            "distinction": "VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Elevated lactate/pyruvate. Hepatocerebral phenotype. VPA CATASTROPHIC CI in POLG1 (mitochondrial DNA depletion → hepatic failure) — same CI as ZSD but different mechanism. POLG1 exclusion MANDATORY in all ZSD before any VPA.",
        },
    ]

    standards = [
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum: An overview. Mol Genet Metab. 2016",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Klouwer FCC et al. Zellweger spectrum disorders: clinical overview and management approach. Orphanet J Rare Dis. 2015",
        "Fujiki Y et al. PEX genes and peroxisome biogenesis disorders. Biochim Biophys Acta Mol Cell Res. 2020",
        "Okumoto K et al. Pex12p, the pex10p-binding protein of peroxisomal targeting signal type 2-importing machinery. J Biol Chem. 2000",
        "Platta HW et al. The peroxisomal receptor docking complex: assembly and function in peroxisomal matrix protein import. FEBS J. 2014",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "CPIC VPA–POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614859 (PBD3A-ZSD — PEX12); #266510 (PBD3B-NALD — PEX12); *601758 (PEX12 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
