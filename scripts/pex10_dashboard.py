#!/usr/bin/env python3
"""PEX10 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX10 encodes Peroxin-10,
a 337-amino-acid integral peroxisomal membrane protein with a cytoplasmic C3HC4 RING finger
domain (zinc-coordinated E3 ubiquitin ligase). PEX10 is an obligate member of the
PEX2–PEX10–PEX12 RING finger E3 ubiquitin ligase complex that resides in the peroxisomal
membrane.

MECHANISM — RING FINGER E3 UBIQUITIN LIGASE (SCAFFOLD/BRIDGE SUBUNIT OF PEX2–PEX10–PEX12):
  PEX10 is the SCAFFOLD/BRIDGE member of the trimeric RING complex. Unlike PEX2 and PEX12
  which each contribute primarily catalytic E3 activity, PEX10's coiled-coil domain physically
  tethers PEX2 and PEX12 together:
  (1) PEX10 N-terminal transmembrane domain (TMD): anchors PEX10 in the peroxisomal membrane
  (2) PEX10 coiled-coil domain: bridges PEX2 (on one side) and PEX12 (on the other side),
      forming the structural core of the trimeric RING complex
  (3) PEX10 C-terminal RING finger domain: contributes E3 ubiquitin ligase catalytic activity
      (E2 ubiquitin-conjugating enzyme recruitment)
  PEX10 LOF → RING complex STRUCTURALLY DESTABILIZED (not merely one catalytic subunit lost)
  → PEX5 cannot be ubiquitinated by the complex → PEX5 permanently trapped in peroxisomal
  membrane → PTS1 protein import fails completely → ALL peroxisomal matrix enzyme functions
  simultaneously impaired. This is STRUCTURALLY MORE DESTABILIZING than single-subunit PEX2 or
  PEX12 loss, because PEX10 is the bridging scaffold.
  Nevertheless, the biochemical phenotype is IDENTICAL to PEX1/PEX2/PEX6-ZSD: all 6
  peroxisomal pathways simultaneously fail.

LOCUS: 1p36.32  |  OMIM GENE: *602859  |  OMIM DISEASE: #614870 (ZSD-PBD6A), #614871 (ZSD-PBD6B)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX10 mutations account for ~5–7% of all
  PBD-ZSD (more common than PEX2 at 3–5%; third most common ZSD gene after PEX1 ~65% and
  PEX6 ~10%). ~80–120 confirmed PEX10 cases worldwide as of 2026.
  PEX10 has SOME partial-function alleles: p.Ile332Thr (RING domain missense, retains ~10–15%
  E3 activity → enables mild/IRD phenotype) — unlike PEX2 which has NO partial-function allele.
  However, no common FOUNDER hypomorphic allele exists (unlike PEX1-G843D or PEX6-R860W).
  Reported pathogenic variants: p.Arg234* (nonsense, most frequent, null);
  p.Ile332Thr (RING domain, partial function); c.350T>C p.Leu117Pro (TMD/coiled-coil);
  splice-site and frameshift variants. Most PEX10 patients have null/partial compound het.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (SAME AS PEX1 / PEX2 / PEX6):
  PEX10 deficiency causes the SAME biochemical phenotype as PEX1/PEX2/PEX6 because PEX10
  scaffolding is required for PEX5 retrotranslocation. Complete scaffold failure abolishes
  PEX5 cycling and all peroxisomal matrix protein import:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
  (3) Plasmalogens (ether-phospholipids) BIOSYNTHESIS FAILED → erythrocyte plasmalogens LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated in plasma/urine
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid
  DISTINGUISHING PEX10 FROM PEX1/PEX2/PEX6 BIOCHEMICALLY: biochemically identical to other
  ZSD — only comprehensive PEX gene panel (NGS) distinguishes.

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — SLIGHTLY MORE IRD THAN PEX2 DUE TO p.Ile332Thr):
  PEX10 cohorts show SLIGHTLY MORE IRD (25%) than PEX2-ZSD (20%), because the p.Ile332Thr
  partial-function RING allele allows some residual complex activity when compound het with a
  null allele. However, no founder hypomorphic allele exists → overall spectrum still skewed
  severe compared to PEX1/PEX6:
  (1) Zellweger Syndrome (ZS, severe): null/null genotype; neonatal onset day 1–7; craniofacial
      dysmorphism; profound hypotonia; pachygyria + polymicrogyria (MRI PATHOGNOMONIC in ZS);
      neonatal seizures UNIVERSAL (100%); hepatomegaly + neonatal cholestasis; ERG absent;
      SNHL; adrenal insufficiency; skeletal stippling; fatal <6–12 months.
  (2) NALD (intermediate): null/partial (p.Ile332Thr compound het or splice-site); onset
      1st–12th month; seizures 80–90%; infantile spasms 35–55% + hypsarrhythmia; cholestasis;
      survival months–years.
  (3) IRD (attenuated): p.Ile332Thr/mild variant compound het; RP + SNHL + peripheral
      neuropathy; hepatomegaly mild; seizures 20–30%; adult survival in some.
      IRD 25% PEX10 > 20% IRD PEX2 — reflects partial-function allele p.Ile332Thr.
  (4) Atypical: ~5%; very mild; adult-onset ataxia/RP only.

PEX10 SCAFFOLD FUNCTION — UNIQUE FEATURE vs PEX2 and PEX12:
  Unlike PEX2 (RING catalytic subunit) or PEX12 (RING catalytic subunit), PEX10 serves
  as the BRIDGE that physically holds the trimer together. PEX10 coiled-coil domain contacts
  both PEX2 and PEX12 simultaneously. Loss of PEX10 → ENTIRE COMPLEX DISSOCIATES:
  PEX2 and PEX12 lose their membrane localization and mutual interaction, even if both PEX2
  and PEX12 are individually intact. This structural role means PEX10 missense alleles that
  disrupt the coiled-coil produce ZS/NALD (complex falls apart), while missense in the RING
  finger domain only (p.Ile332Thr) partially spares complex assembly → allows some IRD.

EPILEPSY IN PEX10-ZSD:
  ZS: 100% epilepsy; neonatal seizures day 1–7; multifocal clonic + tonic + myoclonic +
  electrographic; hypsarrhythmia if surviving infantile period; EEG burst-suppression;
  universal drug-resistance; SE frequent; fatal <12M.
  NALD: 80–90% epilepsy; infantile spasms (IS) 35–55% + hypsarrhythmia; focal + generalized +
  myoclonic; moderate drug-resistance.
  IRD: 20–30% epilepsy; focal + GTCS; better AED response; phytol-restricted diet helps.
  DRE: ~60–65% of all PEX10-ZSD (slightly less than PEX2-ZSD 65–70% due to more IRD patients).

AED PHARMACOLOGY (IDENTICAL TO PEX2-ZSD — RING COMPLEX MECHANISM DOES NOT CHANGE AED RULES):
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
  ERT: NOT AVAILABLE — PEX10 is an integral peroxisomal membrane protein (not a soluble enzyme);
    protein replacement is not applicable.
"""

import random


# ── Overview ───────────────────────────────────────────────────────────────────
def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 82,
        "zs_pct": 25,
        "nald_pct": 45,
        "ird_pct": 25,
        "atypical_pct": 5,
        "drug_resistance_pct": 62,
        "on_dha_pct": 58,
        "liver_disease_pct": 72,
        "retinopathy_pct": 78,
        "dha_low_pct": 88,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "omim_gene": "602859",
        "omim_disease_zs": "614870",
        "omim_disease_nald": "614871",
        "locus": "1p36.32",
        "protein_size": "337 aa",
        "nbs_positive_rate": "~98% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": "p.Arg234* (null, most common); p.Ile332Thr (RING partial-function, IRD-enabling)",
        "disease_mechanism": (
            "PEX10 (337 aa, 1p36.32) is the SCAFFOLD/BRIDGE subunit of the PEX2–PEX10–PEX12 "
            "RING E3 ubiquitin ligase complex in the peroxisomal membrane. PEX10's coiled-coil "
            "domain physically tethers PEX2 and PEX12 together; loss of PEX10 DESTABILIZES THE "
            "ENTIRE COMPLEX (both PEX2 and PEX12 lose their membrane association and mutual "
            "interaction). This makes PEX10 LOF structurally more disruptive than single PEX2 or "
            "PEX12 loss. Result: PEX5 (PTS1 cargo receptor) cannot be ubiquitinated → trapped in "
            "peroxisomal membrane → ALL matrix protein import fails → same biochemical phenotype "
            "as PEX1/PEX2/PEX6-ZSD. Biochemically identical; only NGS distinguishes. "
            "~5–7% of all PBD-ZSD; ~80–120 cases worldwide 2026."
        ),
        "key_concepts": [
            "PEX10 scaffold role: bridges PEX2 and PEX12 via coiled-coil — loss of PEX10 dissociates entire RING trimer, not just one catalytic subunit",
            "p.Ile332Thr: partial-function RING allele (~10–15% residual E3 activity) → enables IRD phenotype (25% IRD in PEX10 vs 20% in PEX2)",
            "No founder hypomorphic allele (unlike PEX1-G843D or PEX6-R860W) → most PEX10 patients have null/partial → spectrum skewed severe",
            "Biochemically identical to PEX1/PEX2/PEX6-ZSD: VLCFA + plasmalogens LOW + DHA low + phytanic/pristanic + pipecolic + DHCA/THCA ALL abnormal",
            "PEX10 vs PEX2: PEX10 scaffold failure is structurally more complete (whole complex collapses); PEX2 loss = single catalytic arm failure (PEX12 and complex partially retain structure)",
            "VPA HIGH RISK (three mechanisms: hepatotoxicity + peroxisomal BO inhibition + carnitine); POLG1 MANDATORY (CPIC Grade A)",
            "VGB HIGH RISK (additive retinopathy: ZSD retinopathy universal ZS/NALD + VGB irreversible VF constriction)",
            "ACTH Level A for infantile spasms (IS): preferred over VGB due to ZSD retinopathy risk",
            "LEV first-line all ZSD forms; DHA Level B (NALD/IRD); Phytol-restricted diet Level B (IRD)",
            "Fasting EXTREME HAZARD: adipose phytanic acid mobilisation → acute neurotoxicity + seizures; IV dextrose MANDATORY during NPO",
            "PHT/CBZ/OXC RELATIVE CI (CYP3A4 → DHA depletion + hepatic) — NOT adrenal CI (unlike ABCD1 ABSOLUTE CI)",
            "Lorenzo's Oil INEFFECTIVE (peroxisomal import absent — elongase inhibition cannot compensate)",
            "HSCT NOT INDICATED (ZSD not inflammatory; no neuroinflammatory target unlike CCALD/ABCD1)",
            "No ERT (PEX10 = integral membrane RING protein; protein replacement not applicable)",
            "C26:0-lyso-PC NBS biomarker + erythrocyte plasmalogens LOW = biochemical ZSD diagnosis; NGS panel required to distinguish PEX10 from other ZSD genes",
            "~80–120 cases worldwide 2026 (more common than PEX2 ~20–30 and PEX12 ~40–60, less than PEX1 and PEX6)",
        ],
        "standards": [
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum: An overview. Mol Genet Metab. 2016",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Klouwer FCC et al. Zellweger spectrum disorders: clinical overview and management approach. Orphanet J Rare Dis. 2015",
            "Fujiki Y et al. PEX genes and peroxisome biogenesis disorders. Biochim Biophys Acta Mol Cell Res. 2020",
            "Platta HW et al. The peroxisomal receptor docking complex: assembly and function in peroxisomal matrix protein import. FEBS J. 2014",
            "Okumoto K et al. The peroxin Pex10p is a ubiquitin ligase required for import of PTS2-targeted proteins. J Cell Biol. 2011",
            "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
            "CPIC VPA–POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614870 (PBD6A, Zellweger — PEX10); #614871 (PBD6B, NALD — PEX10); *602859 (PEX10 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(10428)  # PEX10 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 25, "n": 10,
            "sex": "5M/5F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + electrographic; EEG burst-suppression",
            "eeg": "Burst-suppression → multifocal spike-wave (high amplitude); poor prognosis; EEG monitoring NICU mandatory",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Arg234*/p.Arg234* or compound frameshift/nonsense); PEX10 RING complex completely absent from peroxisomal membrane; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Ile332Thr or splice",
            "pct": 45, "n": 18,
            "sex": "9M/9F",
            "onset_age": "Infancy (1–12 months)",
            "seizure_risk": "80–90% — infantile spasms (35–55%) + hypsarrhythmia; focal + myoclonic",
            "eeg": "Hypsarrhythmia (35–55% NALD) + modified hypsarrhythmia; ACTH Level A; multi-focal spikes",
            "mri": "WM dysmyelination (periventricular + deep WM); cerebellar hypoplasia; less severe than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Arg234*/p.Ile332Thr (most common NALD compound het in PEX10); or null/splice-site; partial complex stability with p.Ile332Thr",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Ile332Thr/mild or mild/mild",
            "pct": 25, "n": 10,
            "sex": "5M/5F",
            "onset_age": "Early childhood (1–10 years) to adolescence",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD",
            "eeg": "Focal spikes (temporal/occipital); generalized polyspike-wave (rare); usually AED-responsive",
            "mri": "Cerebellar atrophy (late); WM signal mild; no pachygyria (cortical migration intact in IRD)",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Ile332Thr/mild missense compound het (RING partially functional: ~10–15% residual activity); slowest progression; adult survival possible",
        },
        {
            "name": "Atypical / Late-Onset ZSD",
            "pct": 5, "n": 2,
            "sex": "1M/1F",
            "onset_age": "Adolescence or adulthood",
            "seizure_risk": "10–20% — focal seizures; often years after RP/SNHL diagnosis",
            "eeg": "Focal spikes; rare secondary generalization; AED-responsive",
            "mri": "Minimal WM changes; cerebellar atrophy possible; no migration anomalies",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Ultra-mild alleles (deep intronic, exon-skipping allowing partial transcript); lowest peroxisomal dysfunction; RP + SNHL dominate; seizures minor feature",
        },
    ]

    phenotype_weights = [
        ("ZS (Zellweger-Severe)", 25),
        ("NALD (Neonatal-ALD-Intermediate)", 45),
        ("IRD (Infantile-Refsum-Attenuated)", 25),
        ("Atypical / Late-Onset ZSD", 5),
    ]
    phenotypes_pool = []
    for (ph, w) in phenotype_weights:
        phenotypes_pool.extend([ph] * w)

    genotype_map = {
        "ZS (Zellweger-Severe)": ["p.Arg234*/p.Arg234*", "p.Arg234*/c.del_fs", "p.Leu117Pro/p.Arg234*", "frameshift/frameshift"],
        "NALD (Neonatal-ALD-Intermediate)": ["p.Arg234*/p.Ile332Thr", "splice/p.Ile332Thr", "p.Arg234*/splice", "p.Leu117Pro/p.Ile332Thr"],
        "IRD (Infantile-Refsum-Attenuated)": ["p.Ile332Thr/p.Ile332Thr", "p.Ile332Thr/mild_mis", "mild_mis/mild_mis"],
        "Atypical / Late-Onset ZSD": ["deep_intron/p.Ile332Thr", "mild_splice/mild_mis"],
    }

    aed_map = {
        "ZS (Zellweger-Severe)": ["LEV (IV)", "PHB (IV)", "LEV (IV) + CLB"],
        "NALD (Neonatal-ALD-Intermediate)": ["LEV", "LEV + CLB", "LEV + ACTH", "LEV + CLB + LTG"],
        "IRD (Infantile-Refsum-Attenuated)": ["LEV", "LEV + LTG", "LTG", "CLB + LEV"],
        "Atypical / Late-Onset ZSD": ["LEV", "LTG", "CLB"],
    }

    resp_map = {
        "ZS (Zellweger-Severe)": ["Drug-resistant", "Drug-resistant", "Drug-resistant"],
        "NALD (Neonatal-ALD-Intermediate)": ["Drug-resistant", "Partially controlled", "Partially controlled", "Drug-resistant"],
        "IRD (Infantile-Refsum-Attenuated)": ["Controlled", "Partially controlled", "Controlled"],
        "Atypical / Late-Onset ZSD": ["Controlled"],
    }

    liver_risk = {
        "ZS (Zellweger-Severe)": 1.0,
        "NALD (Neonatal-ALD-Intermediate)": 0.85,
        "IRD (Infantile-Refsum-Attenuated)": 0.35,
        "Atypical / Late-Onset ZSD": 0.15,
    }

    sz_risk = {
        "ZS (Zellweger-Severe)": 1.0,
        "NALD (Neonatal-ALD-Intermediate)": 0.87,
        "IRD (Infantile-Refsum-Attenuated)": 0.25,
        "Atypical / Late-Onset ZSD": 0.15,
    }

    patients = []
    for i in range(40):
        ph = random.choice(phenotypes_pool)
        has_liver = random.random() < liver_risk[ph]
        has_sz = random.random() < sz_risk[ph]
        geno_pool = genotype_map[ph]
        aed_pool = aed_map[ph]
        resp_list = resp_map[ph]
        patients.append({
            "patient_id": f"PEX10-{str(i+1).zfill(3)}",
            "phenotype": ph,
            "sex": random.choice(["M", "F"]),
            "genotype": random.choice(geno_pool),
            "liver_disease": has_liver,
            "has_seizures": has_sz,
            "primary_aed": random.choice(aed_pool) if has_sz else None,
            "drug_response": random.choice(resp_list) if has_sz else None,
            "on_dha": ph != "ZS (Zellweger-Severe)" and random.random() < 0.68,
        })

    seizure_types = [
        {"type": "Neonatal multifocal clonic (ZS — neonatal day 1–7)", "pct": 25,
         "eeg": "Burst-suppression → multifocal spike-wave; high amplitude; background suppression; NICU EEG mandatory"},
        {"type": "Infantile spasms / West syndrome (NALD — 1–6 months)", "pct": 45,
         "eeg": "Hypsarrhythmia (35–55% NALD) + modified hypsarrhythmia; ACTH Level A preferred over VGB (ZSD retinopathy)"},
        {"type": "Tonic seizures (ZS + NALD)", "pct": 35,
         "eeg": "Generalized voltage attenuation → fast recruiting EEG; NICU monitoring required in ZS/NALD"},
        {"type": "Myoclonic seizures (NALD + IRD)", "pct": 30,
         "eeg": "Generalized polyspike-wave 3–5 Hz; photosensitivity variable; CLB Level B adjunct"},
        {"type": "Focal seizures (IRD + Atypical)", "pct": 25,
         "eeg": "Focal spikes temporal/occipital; better AED response in IRD; IRD 25% PEX10 > 20% PEX2"},
        {"type": "Generalized tonic-clonic (IRD)", "pct": 15,
         "eeg": "Generalized polyspike-wave; LEV/LTG usually effective in IRD"},
        {"type": "Status epilepticus (ZS + NALD)", "pct": 22,
         "eeg": "Non-convulsive SE common in ZS; IV LEV loading mandatory; fosphenytoin AVOIDED (DHA depletion + hepatic)"},
    ]

    triggers = [
        {"trigger": "Febrile illness (ZS + NALD)", "pct": 78,
         "note": "Most common trigger across all ZSD; aggressive antipyretic management; IV LEV prophylaxis during febrile illness in NALD; ZS: universal trigger but rarely actionable (fatal prognosis)"},
        {"trigger": "Fasting / NPO / Anaesthesia", "pct": 72,
         "note": "EXTREME HAZARD: starvation → adipose phytanic acid mobilisation → acute metabolic neurotoxicity + seizure precipitation; IV dextrose MANDATORY during any pre-surgical fast; IV Vitamin K (cholestatic coagulopathy ZS/NALD)"},
        {"trigger": "Phytanic acid surge (dietary — IRD/Atypical)", "pct": 48,
         "note": "IRD/Atypical: dietary phytol (meat fats, dairy, fish oils) → phytanic acid accumulates → proconvulsant; strict phytol-restricted diet Level B in IRD/Atypical"},
        {"trigger": "Metabolic decompensation (hepatic)", "pct": 55,
         "note": "Cholestatic liver dysfunction (ZS/NALD) → hyperammonaemia + coagulopathy + electrolyte disturbance → seizure precipitant; monitor LFT + ammonia + INR q2–4 weeks"},
        {"trigger": "Sleep deprivation (IRD)", "pct": 30,
         "note": "IRD patients report sleep-deprived focal seizures; normal sleep hygiene counselling; less severe trigger than in ZS/NALD"},
        {"trigger": "Anaesthesia peri-operative period", "pct": 28,
         "note": "Phytanic acid mobilisation from adipose during surgical fasting; pre-op IV glucose loading + IV Vitamin K + LFT + PT/INR mandatory; brief anaesthesia protocols preferred"},
        {"trigger": "VPA initiation (iatrogenic)", "pct": 20,
         "note": "VPA HIGH RISK: acute hepatic failure (cholestatic baseline) and peroxisomal BO inhibition exacerbating VLCFA accumulation; POLG1 MANDATORY before any VPA attempt"},
    ]

    monitoring = [
        "VLCFA panel (C26:0, C26:0/C22:0, C24:0/C22:0 ratios) — at diagnosis + q6M (NALD/IRD) / q3M (ZS if surviving)",
        "Erythrocyte plasmalogens (DMA-16:0, DMA-18:0) — at diagnosis + q12M; LOW confirms ZSD (key ABCD1 distinction)",
        "DHA plasma/erythrocyte — at diagnosis + q6M; guides DHA supplementation dosing (NALD/IRD)",
        "LFT + GGT + bilirubin (direct) + PT/INR — q4W (ZS), q8W (NALD), q12W (IRD); cholestatic liver disease monitoring",
        "Urinary pipecolic acid — at diagnosis; elevated in ALL ZSD (metabolic confirmation)",
        "DHCA + THCA (plasma) — at diagnosis; bile acid intermediates confirm peroxisomal beta-oxidation failure",
        "EEG (awake + sleep) — at seizure onset + q6M (ZS/NALD during active management); video-EEG for spell characterisation",
        "Brain MRI (1.5T or 3T) — at diagnosis; pachygyria/PMG (ZS PATHOGNOMONIC); WM signal + cerebellar volume q12M (NALD/IRD)",
        "ERG (electroretinogram) — at diagnosis + q6M; retinal degeneration monitoring; VGB absolutely contraindicated if ANY ERG abnormality",
        "Audiological evaluation (BERA/ABR + pure tone) — at diagnosis + q12M; SNHL universal in ZS/NALD",
        "Adrenal function (ACTH stimulation / basal cortisol) — at diagnosis; adrenal insufficiency possible (hydrocortisone perioperatively)",
        "Skeletal survey (radiograph) — at diagnosis (ZS): chondrodysplasia punctata (stippled patella/acetabulum)",
        "Comprehensive PEX gene panel (NGS) — mandatory at diagnosis to confirm PEX10 vs PEX1/PEX2/PEX6/PEX12; biochemically identical",
        "POLG1 sequencing — MANDATORY before any VPA trial (CPIC Grade A); POLG1 LOF + VPA = catastrophic hepatotoxicity",
    ]

    thresholds = [
        {"parameter": "C26:0-lyso-PC (DBS, NBS)", "threshold": ">0.60 μmol/L",
         "action": "Reflexive full VLCFA panel + PEX gene panel; initiate DHA immediately (NALD/IRD); contact metabolic genetics urgently"},
        {"parameter": "C26:0 plasma (VLCFA)", "threshold": ">1.30 μg/mL",
         "action": "Confirm ZSD: add plasmalogens (RBC) + pipecolic acid + DHA; if plasmalogens LOW → ZSD (not ABCD1); gene panel"},
        {"parameter": "Plasmalogen ratio (RBC)", "threshold": "<0.070 (C16-DMA/C16:0)",
         "action": "Confirms ZSD vs ABCD1; combined with elevated C26:0 → all PBD-ZSD; NGS panel to identify which PEX gene"},
        {"parameter": "DHA (plasma)", "threshold": "<100 μmol/L",
         "action": "Start DHA supplementation (NALD/IRD): 20–50 mg/kg/day; recheck q3M until stable"},
        {"parameter": "ALT / AST", "threshold": ">3× ULN",
         "action": "Hold VPA (if any); increase LFT monitoring to weekly; GI/hepatology consultation; consider UDCA (Level C)"},
        {"parameter": "INR (coagulopathy)", "threshold": ">1.5 pre-procedure",
         "action": "IV Vitamin K 1 mg/kg (max 10 mg) × 3 doses; postpone elective surgery; FFP if INR >2.5 + active bleeding"},
        {"parameter": "Phytanic acid (plasma — IRD)", "threshold": ">200 μmol/L",
         "action": "Strict phytol-restricted diet counselling; dietitian consultation; IV glucose if acute surge during illness"},
        {"parameter": "Ammonia", "threshold": ">80 μmol/L",
         "action": "Withhold VPA (if prescribed); protein restriction; lactulose; N-acetylcysteine; hepatology consultation"},
    ]

    lifecycle = [
        {
            "stage": "Stage 1 — Neonatal (Birth to 1 Month) — ZS DOMINATED",
            "features": "Dysmorphic features (large fontanelle, flat face, epicanthal folds); profound hypotonia; neonatal seizures day 1–5 (100% ZS); hepatomegaly; cholestasis; ERG absent; adrenal insufficiency; NICU level care",
            "action": "NBS C26:0-lyso-PC + emergency VLCFA panel + PEX gene panel (confirm PEX10); IV LEV loading (20–30 mg/kg); IV glucose (avoid fasting); IV Vit K; adrenal testing + hydrocortisone if crisis; palliative care discussion (ZS: fatal prognosis)",
        },
        {
            "stage": "Stage 2 — Infantile (1–12 Months) — NALD DOMINATED",
            "features": "Infantile spasms (35–55%) + hypsarrhythmia; cholestasis; hepatomegaly; retinopathy developing; SNHL confirmed; DHA low; psychomotor delay; moderate-severe disease",
            "action": "ACTH (Level A, 20–40 U/day IM × 2 weeks → taper) for IS; DHA supplementation (20–50 mg/kg/day); phytol-restricted diet begins; ERG + audiology; VPA AVOID; VGB AVOID; LEV maintenance",
        },
        {
            "stage": "Stage 3 — Early Childhood (1–5 Years) — NALD/IRD TRANSITION",
            "features": "Ongoing seizures (focal + myoclonic); progressive demyelination (WM MRI); cerebellar atrophy; SNHL + RP evolving; intellectual disability (variable); hepatomegaly stabilising in IRD",
            "action": "Optimise AED (LEV ± CLB ± LTG); DHA maintenance; quarterly VLCFA + LFT; physiotherapy + OT; hearing aids; vision aids; special education planning; genetic counselling (AR 25% recurrence risk)",
        },
        {
            "stage": "Stage 4 — Later Childhood (5–12 Years) — IRD DOMINANT",
            "features": "RP progression (tunnel vision); SNHL severe in many; peripheral neuropathy (gait disturbance); seizures often well-controlled in IRD (25% of PEX10 cohort — more than PEX2 20%); liver disease quiescent in IRD",
            "action": "Annual ophthalmology (VF + OCT + ERG); audiological rehabilitation; gait physiotherapy; school integration; DHA + phytol-restricted diet; AVOID CYP-INDUCING AEDs (PHT/CBZ/OXC deplete DHA); LEV/LTG/CLB preferred",
        },
        {
            "stage": "Stage 5 — Adolescent/Adult (>12 Years) — IRD/ATYPICAL ONLY",
            "features": "Advanced RP (often legally blind); severe SNHL; peripheral polyneuropathy; seizures 20–30% (IRD); hepatomegaly resolved; intellectual disability moderate; azoospermia possible (males)",
            "action": "Cochlear implant evaluation; low-vision rehabilitation; anti-epileptic therapy (LEV/LTG preferred, no CYP inducers); genetic counselling for family planning (AR inheritance); phytol-restricted diet lifelong",
        },
        {
            "stage": "Stage 6 — End-of-Life / Palliative (ZS/Severe NALD)",
            "features": "ZS: fatal <6–12 months; NALD (severe): deterioration over months–years; respiratory failure; hepatic failure; intractable seizures; comfort-focused care priorities",
            "action": "Palliative care team from diagnosis (ZS); goals-of-care discussion; comfort AED dosing (IV LEV + oral CLB); feeding support (NGT/PEG in NALD); home hospice if appropriate; bereavement support",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "Level A — First-line (all ZSD forms)",
            "evidence": "Universal first-line AED in PBD-ZSD regardless of form; no enzyme induction (no DHA depletion); no hepatotoxicity; IV/PO; neonatal-safe (weight-based loading 20–30 mg/kg IV); SV2A mechanism",
            "dose": "Neonatal: 10–20 mg/kg/day IV → titrate; Infant/Child: 20–40 mg/kg/day (max 60 mg/kg/day); IV for SE loading",
            "moa": "SV2A (synaptic vesicle glycoprotein 2A) — inhibits presynaptic neurotransmitter release; no CYP induction",
            "monitoring": "Serum level 20–40 μg/mL (optional); CBC (rare haematological effects); behavioural side effects in NALD/ZS — monitor",
            "ci": "None in ZSD — safest AED for this population",
        },
        {
            "drug": "ACTH (Adrenocorticotropic Hormone)",
            "class": "Level A — Infantile spasms (NALD; ZS if surviving infantile period)",
            "evidence": "First-line for infantile spasms/West syndrome in PBD-ZSD; preferred over VGB due to universal ZSD retinopathy; UKISS, WEST, INFANT trials support ACTH in IS",
            "dose": "20–40 U/day IM × 14 days → 20 U every other day × 14 days → taper; monitor BP, glucose, infection",
            "moa": "MC2R agonist → glucocorticoid → anti-inflammatory/anti-epileptic; direct melanocortin CNS effects on IS",
            "monitoring": "BP, glucose, potassium, infection signs daily during high-dose phase; growth monitoring during taper; adrenal axis recovery post-taper",
            "ci": "Active untreated infection; uncontrolled diabetes; severe osteoporosis",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "Level B — Adjunct (all ZSD forms)",
            "evidence": "Adjunct for focal + myoclonic + tonic seizures; 1,5-benzodiazepine (less sedating than clonazepam for chronic use); no hepatotoxicity; no enzyme induction; safe in ZSD liver disease",
            "dose": "0.1–0.3 mg/kg/day in 2 divided doses (max 1.0 mg/kg/day); start low, titrate",
            "moa": "GABA-A allosteric modulator (benzodiazepine site); 1,5-BZD isomer preferred for chronic use",
            "monitoring": "Sedation; tolerance development over months; gradual taper required for discontinuation",
            "ci": "None specific in ZSD; avoid abrupt discontinuation (withdrawal seizures)",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "Level C — Adjunct (NALD/IRD focal + GTCS)",
            "evidence": "Adjunct for focal and GTCS in NALD/IRD; hepatic glucuronidation (NOT CYP2D6/3A4) → safer hepatic profile vs CBZ/PHT; no DHA depletion; slow titration mandatory (SJS risk)",
            "dose": "0.15 mg/kg/day × 2 weeks → 0.3 mg/kg/day × 2 weeks → titrate to 2–10 mg/kg/day",
            "moa": "Na+ channel blocker (voltage-gated) + Ca2+ channel modulation; broad-spectrum AED",
            "monitoring": "Rash monitoring — slow titration critical; LFT (minimal hepatic risk via glucuronidation); levels optional",
            "ci": "Avoid rapid titration; avoid combination with VPA (inhibits LTG glucuronidation — and VPA HIGH RISK in ZSD)",
        },
        {
            "drug": "DHA (Docosahexaenoic Acid) Supplementation",
            "class": "Level B — Disease-modifying (NALD/IRD)",
            "evidence": "Restores DHA-depleted neuronal and retinal membranes; Level B for NALD/IRD (stabilises retinal function, may slow CNS progression); not proven in ZS (fatal prognosis dominates)",
            "dose": "20–50 mg/kg/day (ethyl ester or lysophospholipid form); with meals (fat-soluble); recheck plasma DHA q3M; target 100–150 μmol/L",
            "moa": "Restores n-3 PUFA in neuronal/retinal membranes; improves membrane fluidity; not a cure",
            "monitoring": "Plasma DHA q3M; ERG (track retinal DHA response); no known toxicity; use purified algal DHA (avoid fish-oil phytanic acid contamination)",
            "ci": "None in ZSD; avoid fish-oil phytanic acid contamination",
        },
        {
            "drug": "Phytol-Restricted Diet",
            "class": "Level B — Dietary (IRD/Atypical/NALD surviving)",
            "evidence": "Reduces dietary phytanic acid load; essential in IRD (phytanic acid proconvulsant + neurotoxic at high levels); restrict red meat/dairy fat/fish; dietitian-led",
            "dose": "Dietary: restrict phytol-containing foods to <10 mg phytol/day; vegetable oils + lean proteins allowed; caloric adequacy ESSENTIAL",
            "moa": "Reduces phytol → phytanic acid conversion; lowers plasma phytanic acid → reduces proconvulsant effects",
            "monitoring": "Plasma phytanic acid q6M (target <200 μmol/L IRD, <50 μmol/L stable); nutritional status; dietitian review q6–12M",
            "ci": "None; MANDATORY to maintain caloric adequacy — fasting/caloric restriction = phytanic acid mobilisation from adipose = EXTREME HAZARD",
        },
        {
            "drug": "Cholic Acid / UDCA",
            "class": "Level C — Supportive (ZS/NALD cholestatic liver disease)",
            "evidence": "Level C; may reduce cholestatic burden from DHCA/THCA bile acid intermediates; improves bile flow; supportive rather than curative in ZSD",
            "dose": "Cholic acid: 5–10 mg/kg/day PO; UDCA: 10–20 mg/kg/day PO in 2–3 divided doses",
            "moa": "Cholic acid: exogenous primary bile acid competitively reduces DHCA/THCA. UDCA: hydrophilic protective bile acid; cholestasis palliation",
            "monitoring": "LFT q4–8W; bilirubin; GGT; urine bile acids; hepatology co-management",
            "ci": "Complete biliary obstruction (rare in ZSD); adjust dose for hepatic impairment",
        },
    ]

    contraindications = [
        {
            "drug": "VPA (Valproate / Valproic Acid)",
            "level": "HIGH RISK — Near-Absolute CI in ZS/NALD (RELATIVE CI in IRD with extreme caution)",
            "reason": (
                "THREE independent mechanisms: "
                "(1) HEPATOTOXICITY — ZS/NALD have pre-existing cholestatic liver disease; VPA dramatically "
                "increases hepatic failure risk (Reye-like syndrome, hyperammonaemia, fulminant liver failure); "
                "(2) PEROXISOMAL BETA-OXIDATION INHIBITION — VPA inhibits peroxisomal (and mitochondrial) "
                "fatty acid oxidation → worsens VLCFA accumulation in ZSD; "
                "(3) CARNITINE DEPLETION — VPA sequesters acylcarnitines → depletes free carnitine; "
                "worsens any pre-existing carnitine deficiency in ZSD. "
                "POLG1 sequencing MANDATORY (CPIC Grade A) before ANY VPA use: POLG1 LOF + VPA = "
                "catastrophic hepatotoxicity (Alpers-Huttenlocher syndrome)."
            ),
            "alternative": "LEV (first-line) + CLB (Level B) + LTG (Level C); ACTH for infantile spasms (Level A)",
        },
        {
            "drug": "VGB (Vigabatrin)",
            "level": "HIGH RISK — Additive Irreversible Blindness",
            "reason": (
                "ZSD retinopathy is UNIVERSAL in ZS and NALD (ERG absent); 20–30% in IRD. "
                "VGB causes irreversible, cumulative, concentric visual field constriction in "
                "~30–50% of patients on long-term therapy (SABRIL REMS). "
                "VGB + pre-existing ZSD retinopathy = ADDITIVE IRREVERSIBLE BLINDNESS. "
                "ACTH (Level A) is the preferred treatment for infantile spasms in ZSD. "
                "AVOID VGB in IRD (where visual field is still partially preservable). "
                "Last resort in NALD only if ACTH fails AND patient is already severely visually impaired."
            ),
            "alternative": "ACTH Level A (infantile spasms); LEV + CLB for other seizure types",
        },
        {
            "drug": "PHT / CBZ / OXC (Hepatic Enzyme Inducers)",
            "level": "RELATIVE CI — DHA Depletion + Hepatic Burden",
            "reason": (
                "CYP3A4 induction by PHT/CBZ/OXC: (1) accelerates DHA metabolism → further reduces "
                "already-low plasma DHA → worsens retinopathy and neuronal membrane stability; "
                "(2) additional hepatic CYP burden compounds cholestatic liver disease (ZS/NALD). "
                "NOTE: unlike ABCD1 (X-ALD), there is NO adrenal crisis mechanism in ZSD — "
                "PHT/CBZ are NOT absolutely contraindicated by adrenal mechanism in ZSD. "
                "Relative CI only: use if LEV/LTG/CLB fail; monitor DHA + LFT."
            ),
            "alternative": "LEV IV (replaces fosphenytoin in SE); LTG (glucuronidation, not CYP); CLB adjunct",
        },
        {
            "drug": "Fasting / Anaesthesia / Surgical NPO",
            "level": "EXTREME HAZARD — Phytanic Acid Mobilisation Crisis",
            "reason": (
                "Any period of fasting → adipose tissue lipolysis → phytanic acid release into "
                "circulation (phytanic acid stored in adipose in ZSD, especially IRD/NALD survivors). "
                "Acute phytanic acid surge → acute neurological crisis + seizure precipitation. "
                "ZS/NALD: severe cholestatic coagulopathy (INR elevated) → surgical haemorrhage risk. "
                "MANDATORY pre-operative protocol: IV glucose 10% minimum 4 hours before NPO; "
                "IV Vitamin K pre-op; PT/INR + LFT pre-op; minimal fasting duration; IV glucose throughout."
            ),
            "alternative": "IV glucose 10% during NPO; minimal fasting; IV Vitamin K; anaesthesia liaison",
        },
        {
            "drug": "Lorenzo's Oil",
            "level": "INEFFECTIVE — Mechanism Mismatch",
            "reason": (
                "Lorenzo's Oil reduces VLCFA elongase substrate and lowers plasma C26:0 in ABCD1 "
                "(where peroxisomal beta-oxidation is partially functional). In ZSD including PEX10-ZSD, "
                "ALL peroxisomal import is absent — VLCFA cannot enter the peroxisome even if elongase "
                "activity is reduced at the ER. No meaningful effect on VLCFA in any PBD-ZSD."
            ),
            "alternative": "DHA supplementation (Level B, NALD/IRD); phytol-restricted diet (IRD); palliative AED (ZS)",
        },
        {
            "drug": "HSCT (Haematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED — No Neuroinflammatory Window",
            "reason": (
                "HSCT corrects ABCD1 (CCALD) by replacing CNS microglia that arrest inflammatory "
                "demyelination. ZSD (PEX10) is NOT an inflammatory disease: brain damage is from "
                "failed neuronal migration (pachygyria in ZS — fetal DHA deficiency) and metabolic "
                "failure — NOT microglial inflammation. No neuroinflammatory target for HSCT in ZSD."
            ),
            "alternative": "Supportive: DHA (Level B, NALD/IRD); phytol-restricted diet; AED; multidisciplinary care",
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


# ── Definitions ────────────────────────────────────────────────────────────────
def get_definitions():
    key_concepts = [
        "PEX10 — Peroxin-10 (337 aa, 1p36.32): integral peroxisomal membrane protein; N-terminal TMD + coiled-coil domain + C-terminal C3HC4 RING finger E3 ubiquitin ligase domain; SCAFFOLD/BRIDGE subunit of PEX2–PEX10–PEX12 RING complex",
        "PEX10 as scaffold/bridge: PEX10's coiled-coil domain physically tethers PEX2 (one side) and PEX12 (other side) within the trimeric RING complex; loss of PEX10 → entire complex DISSOCIATES (both PEX2 and PEX12 lose membrane association) — more structurally disruptive than loss of single PEX2 or PEX12 catalytic subunit",
        "PEX10 vs PEX2 structural role: PEX2 LOF = loss of one catalytic arm (E3 activity impaired, PEX10–PEX12 scaffold partially preserved); PEX10 LOF = scaffold collapse (both catalytic arms — PEX2 and PEX12 — lose their structural support simultaneously). Both → same biochemical outcome.",
        "p.Ile332Thr — partial-function RING missense: Ile332 is in the RING finger domain but is NOT a zinc-coordinating Cys/His; the Thr substitution partially retains RING structure (~10–15% E3 activity); enables ZSD IRD phenotype (25% IRD in PEX10 vs 20% in PEX2) when compound het with null allele",
        "p.Arg234* — most common PEX10 null allele: nonsense at Arg234 (coiled-coil/RING junction); complete loss of C-terminal RING + disruption of PEX12 interaction; null phenotype → ZS or NALD depending on second allele",
        "Biochemical identity with PEX1/PEX2/PEX6-ZSD: all 6 peroxisomal pathways simultaneously fail — VLCFA elevated + plasmalogens LOW + DHA low + phytanic/pristanic elevated + pipecolic elevated + DHCA/THCA elevated; cannot distinguish PEX10-ZSD from PEX1/PEX2/PEX6/PEX12-ZSD on biochemical panel alone; NGS mandatory",
        "C26:0-lyso-PC (DBS, NBS): newborn screening biomarker; sensitivity ~98% for ZS/NALD; enables pre-symptomatic PEX10-ZSD diagnosis; reflexes to full VLCFA panel + PEX gene panel",
        "Erythrocyte plasmalogens (DMA-16:0 / DMA-18:0): LOW in ALL PBD-ZSD including PEX10; NORMAL in ABCD1 → essential single test to distinguish ZSD from X-ALD at biochemical level",
        "DHA deficiency: peroxisomal import of ACSL enzymes fails → DHA synthesis impaired → DHA low → (i) retinopathy (ERG absent ZS/NALD) + (ii) impaired neuroblast migration (pachygyria/PMG in ZS-PATHOGNOMONIC)",
        "VPA–ZSD: THREE independent CI mechanisms — hepatotoxicity (cholestatic liver baseline), peroxisomal beta-oxidation inhibition (worsens VLCFA), carnitine depletion; POLG1 + VPA = fatal Alpers-Huttenlocher (CPIC Grade A mandatory)",
        "VGB–ZSD retinopathy: vigabatrin → irreversible concentric VF constriction (30–50% long-term); ZSD retinopathy universal in ZS/NALD → additive permanent blindness; ACTH Level A preferred for IS",
        "Fasting hazard: phytanic acid stored in adipose (IRD/NALD survivors); fasting → acute phytanic surge → neurotoxicity + seizures; IV glucose MANDATORY during NPO",
        "~80–120 PEX10 cases worldwide 2026: more common than PEX2 (~20–30) and PEX12 (~40–60); third most common ZSD gene after PEX1 (~65%) and PEX6 (~10%)",
        "PHT/CBZ RELATIVE CI (not absolute): unlike ABCD1 where PHT/CBZ cause adrenal crisis (VLCFA-dependent adrenal cortex steroidogenesis) — ZSD has no adrenal steroidogenesis pathway dependency on PHT/CBZ; relative CI only via DHA depletion",
        "Fasting/adipose phytanic hazard: phytanic acid accumulates in adipose in surviving ZSD patients (IRD/NALD); fasting → adipose lipolysis → phytanic acid surge → acute neurotoxicity + seizure crisis",
        "No ERT for PEX10: PEX10 is an integral peroxisomal membrane RING protein, not a soluble enzyme; protein replacement therapy is not technically feasible for membrane-embedded proteins",
    ]

    diagnostic_algorithm = [
        "Step 1 — Clinical suspicion: neonatal hypotonia + facial dysmorphism + seizures day 1–7 (ZS) OR infantile spasms + hepatomegaly (NALD) OR adult RP + SNHL + cerebellar ataxia (IRD-like) → immediate VLCFA panel",
        "Step 2 — VLCFA panel (plasma): C26:0 elevated? C26:0/C22:0 ratio elevated? → confirms peroxisomal VLCFA oxidation failure",
        "Step 3 — Erythrocyte plasmalogens (DMA-16:0, DMA-18:0): LOW → ZSD (PBD-ZSD); NORMAL → ABCD1 (XLD) or single-enzyme deficiency; this single test separates ZSD from ABCD1",
        "Step 4 — Additional metabolites: pipecolic acid (elevated ALL ZSD); phytanic + pristanic acid (elevated ZSD); DHA (low ZSD); DHCA+THCA bile acids (elevated ZSD)",
        "Step 5 — NBS result review: C26:0-lyso-PC DBS elevated? → ZSD panel reflexed; confirm with full biochemical panel",
        "Step 6 — Brain MRI (1.5T/3T): ZS = pachygyria + PMG + periventricular germinolytic cysts + hypomyelination (PATHOGNOMONIC); NALD = WM dysmyelination + cerebellar hypoplasia; IRD = cerebellar atrophy (late)",
        "Step 7 — Comprehensive PEX gene panel (NGS): test PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26 simultaneously; biochemically identical — only gene panel distinguishes",
        "Step 8 — Ophthalmological assessment: ERG (flash ERG in neonates) — ERG absent/severely reduced = ZSD retinopathy (universal in ZS/NALD); VF + OCT for cooperative children in NALD/IRD",
        "Step 9 — POLG1 sequencing: MANDATORY before any VPA consideration (CPIC Grade A); POLG1 LOF = absolute VPA contraindication (fatal Alpers-Huttenlocher syndrome risk)",
        "Step 10 — Cardiac evaluation: ECG + echocardiogram (congenital heart defects in ZS — 20–30%: ASD, VSD, PDA)",
        "Step 11 — Adrenal function testing: ACTH stimulation test + basal cortisol; adrenal insufficiency in ZS → hydrocortisone replacement perioperatively",
        "Step 12 — Multidisciplinary team: neurometabolic specialist + neurologist + hepatologist + ophthalmologist + audiologist + geneticist + dietitian + palliative care (ZS) + physiotherapy/OT; refer to ZSD expert centre",
    ]

    pharmacological_distinctions = [
        "PEX10-ZSD vs ABCD1 (X-ALD): LEV preferred in both. PHT/CBZ are ABSOLUTE CI in ABCD1 (adrenal crisis — adrenal cortex steroidogenesis requires VLCFA-free environment, disrupted by PHT/CBZ enzyme induction), but only RELATIVE CI in PEX10-ZSD (DHA depletion + hepatic — NOT adrenal mechanism). Critical distinction.",
        "VPA in PEX10-ZSD vs VPA in RCDP/PEX7: In RCDP (PEX7) the VLCFA pathway is NORMAL (only PTS2 import affected); VPA RELATIVE CI in PEX7 (hepatotoxicity only) but does NOT inhibit peroxisomal VLCFA oxidation. In ZSD ALL pathways impaired — VPA adds peroxisomal BO inhibition = higher combined risk.",
        "VGB in ZSD vs VGB in SSADH (ALDH5A1): VGB ABSOLUTE CI in SSADH (metabolic mechanism: inhibits GABA-T → more SSA → more GHB → metabolic worsening). In PEX10-ZSD, VGB is HIGH RISK (retinopathy mechanism — NOT metabolic; GABA-T pathway intact in ZSD). Different CI mechanisms for same drug.",
        "ACTH in PEX10-ZSD: LEVEL A for infantile spasms — preferred over VGB because of ZSD retinopathy. ZSD cohort: ACTH for IS in any peroxisomal disorder with ZS/NALD phenotype where VGB is contraindicated by retinopathy.",
        "DHA supplementation: Level B in ZSD (NALD/IRD); not applicable to ABCD1 (DHA pathway functional in ABCD1 — only VLCFA transport affected). DHA bypasses peroxisomal synthesis by intestinal absorption as LPC-DHA.",
        "PHT/CBZ/OXC in ZSD vs PEX7 (RCDP): Both RELATIVE CI in ZSD (CYP3A4 → DHA depletion + hepatic). In PEX7/RCDP: VLCFA/DHA NORMAL — PHT/CBZ cause less DHA harm (already normal in RCDP); slightly less restrictive in RCDP than ZSD.",
        "PEX10 vs PEX2 spectrum: PEX10 has p.Ile332Thr partial-function allele → 25% IRD vs 20% IRD in PEX2. More IRD in PEX10 = slightly lower DRE rate (62% vs 65–70% PEX2) = slightly more patients benefit from AED optimisation.",
        "Fasting hazard: HIGHEST in ZSD (phytanic + VLCFA mobilisation) and Adult Refsum/PHYH (pure phytanic). Moderate in RCDP/PEX7. Minimal in ABCD1 (short fasting not significantly mobilising adrenal/CNS VLCFA in most patients).",
        "LEV universal first-line: ALL peroxisomal epilepsies (ZSD, RCDP, ACOX1, ABCD1, SSADH, Adult Refsum). No CYP induction, no hepatotoxicity, no DHA depletion, neonatal IV formulation, SV2A mechanism.",
        "Retinopathy-directed VGB avoidance: RP present in ZSD (ZS/NALD universal) + PHYH (Adult Refsum, RP 95%) + ACOX1 (ERG 85%) + RCDP/PEX7 (cataracts/RP 70%) → VGB ABSOLUTE or HIGH RISK in ALL. LEV/CLB replace VGB across the peroxisomal epilepsy spectrum.",
        "Scaffold vs catalytic subunit loss: PEX10 LOF = scaffold collapse of RING trimer (PEX2 and PEX12 both lose support simultaneously) vs PEX2 LOF = single catalytic arm failure. Both → same biochemical outcome. No clinical or pharmacological difference in AED management.",
        "Antenatal counselling: AR (25% sibling recurrence); prenatal diagnosis by CVS/amniocentesis + PEX gene panel; NBS C26:0-lyso-PC enables pre-symptomatic diagnosis in siblings; parental carrier testing mandatory after PEX10 proband confirmed.",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX1-ZSD (most common ZSD — 65% of all PBD)",
            "distinction": "Biochemically IDENTICAL to PEX10-ZSD. PEX1 has common hypomorphic allele p.Gly843Asp-G843D (~30% of all PEX1 alleles) → more IRD (30–35%). PEX10-ZSD has 25% IRD (more than PEX2-ZSD 20%, less than PEX1-ZSD 30%). ONLY NGS gene panel distinguishes.",
        },
        {
            "condition": "PEX2-ZSD (3–5% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX2 has NO partial-function allele → less IRD (20%) vs PEX10 (25%). PEX10 is scaffold/bridge; PEX2 is catalytic. Both → same biochemical phenotype, similar but slightly different phenotypic distribution. 8q21.13 vs 1p36.32. NGS distinguishes.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 has p.Arg860Trp (R860W) founder allele → most attenuated ZSD (IRD 45% PEX6 vs 25% PEX10 vs 20% PEX2). 6p21.1 vs 1p36.32. NGS panel distinguishes.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy — XLD)",
            "distinction": "VLCFA elevated in BOTH — but ABCD1 has NORMAL plasmalogens (RBC) and NORMAL DHA; ZSD has LOW plasmalogens + LOW DHA. No cortical migration defects (no pachygyria) in ABCD1. PHT/CBZ ABSOLUTE CI in ABCD1 (adrenal); only RELATIVE CI in PEX10-ZSD (DHA/hepatic). Lorenzo's Oil works in ABCD1, not ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC acid severely elevated (sole elevated peroxisomal metabolite); VLCFA NORMAL (key distinction); plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. VGB ABSOLUTE CI (RP 95% — strongest retinopathy-VGB CI among peroxisomal disorders). Phytol-restricted diet Level A GOLD STANDARD.",
        },
        {
            "condition": "RCDP / PEX7 (Rhizomelic Chondrodysplasia Punctata)",
            "distinction": "Plasmalogens (RBC) severely LOW (like ZSD) BUT VLCFA NORMAL + phytanic elevated + pristanic NORMAL + DHA NORMAL + pipecolic NORMAL. Rhizomelia (short proximal limbs) PATHOGNOMONIC. No pachygyria. Single PTS2 pathway affected only.",
        },
        {
            "condition": "PEX12-ZSD (5% of all PBD)",
            "distinction": "Biochemically IDENTICAL to PEX10-ZSD. PEX12 is the third RING complex member (catalytic subunit like PEX2). PEX12 LOF = single catalytic arm failure (PEX10 scaffold may partially preserve PEX2 interaction). ~40–60 cases worldwide. 17q12 vs PEX10 1p36.32. NGS distinguishes.",
        },
        {
            "condition": "Mitochondrial / POLG1 Epilepsy",
            "distinction": "VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Elevated lactate/pyruvate. POLG1 LOF: Alpers disease (hepatocerebral). VPA CATASTROPHIC CI (POLG1) — same CI as in ZSD but for different mechanism (mitochondrial depletion vs peroxisomal BO inhibition + hepatotoxicity). Ragged-red fibres + COX-negative fibres on muscle biopsy.",
        },
    ]

    standards = [
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum: An overview. Mol Genet Metab. 2016",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Klouwer FCC et al. Zellweger spectrum disorders: clinical overview and management approach. Orphanet J Rare Dis. 2015",
        "Fujiki Y et al. PEX genes and peroxisome biogenesis disorders. Biochim Biophys Acta Mol Cell Res. 2020",
        "Okumoto K et al. The peroxin Pex10p is a ubiquitin ligase required for import of PTS2-targeted proteins. J Cell Biol. 2011",
        "Platta HW et al. The peroxisomal receptor docking complex: assembly and function in peroxisomal matrix protein import. FEBS J. 2014",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "CPIC VPA–POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614870 (PBD6A, Zellweger — PEX10); #614871 (PBD6B, NALD — PEX10); *602859 (PEX10 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
