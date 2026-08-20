#!/usr/bin/env python3
"""PEX2 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX2 encodes Peroxin-2
(PXMP3 / PAF-1), a 305-amino-acid integral peroxisomal membrane protein with a cytoplasmic
C3HC4 RING finger domain (zinc-coordinated E3 ubiquitin ligase). PEX2 is an obligate member
of the PEX2–PEX10–PEX12 RING finger E3 ubiquitin ligase complex that resides in the
peroxisomal membrane.

MECHANISM — RING FINGER E3 UBIQUITIN LIGASE (DISTINCT FROM PEX1/PEX6 AAA-ATPASE ARM):
  After PEX5 (cytosolic PTS1 receptor, 69 kDa) delivers PTS1-tagged matrix proteins to
  the peroxisomal lumen and becomes inserted in the peroxisomal membrane (docking/translocation
  intermediate), the PEX2–PEX10–PEX12 RING complex catalyzes two fates:
  (1) Monoubiquitination of PEX5-Cys11 → triggers retrotranslocation by PEX1–PEX6 AAA-ATPase
      complex → PEX5 recycled to cytosol for next import cycle (productive recycling).
  (2) Polyubiquitination of PEX5-Lys (when PEX5 recycling is impaired or PEX5 is damaged) →
      proteasomal degradation of PEX5 (quality control / last-resort pathway).
  PEX2 LOF → RING finger ablated → PEX5 cannot be ubiquitinated → PEX5 permanently trapped in
  peroxisomal membrane → PTS1 protein import fails completely → ALL peroxisomal matrix enzyme
  functions simultaneously impaired. This is functionally IDENTICAL to PEX1/PEX6 AAA-ATPase
  failure: the two arms (E3 ligase + AAA-ATPase) are co-dependent, and failure of either
  abolishes PEX5 retrotranslocation and hence all matrix protein import.

LOCUS: 8q21.13  |  OMIM GENE: *170993  |  OMIM DISEASE: #614862 (ZSD-PBD5A), #614863 (ZSD-PBD5B)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX2 mutations account for ~3–5% of all
  PBD-ZSD (rare ZSD gene; PEX1 ~65%, PEX6 ~10%, PEX12 ~5%, PEX2 ~3–5%, others rare).
  ~20–30 confirmed PEX2 cases worldwide as of 2026 (ultra-rare). No common hypomorphic
  founder allele (unlike PEX1-G843D or PEX6-R860W) → most PEX2 patients have null/null or
  compound null genotypes → PEX2 cohorts are SKEWED TOWARD SEVERE (ZS + NALD; IRD rare).
  Reported pathogenic variants: p.Arg119* (nonsense, most frequent); p.Cys295Ser (RING domain,
  ablates E3 catalytic activity); splice-site and frameshift variants. Cys295 is a RING
  finger zinc-coordinating residue: any missense here = null-equivalent.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (SAME AS PEX1 / PEX6):
  PEX2 deficiency causes the SAME biochemical phenotype as PEX1/PEX6 because all three
  proteins are required for PEX5 retrotranslocation. Loss of any one prevents PEX5
  cycling and abolishes peroxisomal matrix protein import:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated (PHYH is a PTS2 enzyme, but
      the PTS2 import receptor PEX7 is also a PEX5L-cargo in mammals — indirect impairment)
      + pristanic acid elevated (AMACR, a PTS1 enzyme)
  (3) Plasmalogens (ether-phospholipids) BIOSYNTHESIS FAILED → erythrocyte plasmalogens LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated in plasma/urine
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid
  DISTINGUISHING PEX2 FROM PEX1/PEX6 BIOCHEMICALLY: biochemically identical to PEX1-ZSD and
  PEX6-ZSD — only comprehensive PEX gene panel (NGS) distinguishes.

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — MORE SEVERE THAN PEX6 DUE TO ABSENT HYPOMORPHIC ALLELE):
  PEX2 cohorts show MORE severe disease (more ZS + NALD, less IRD) vs PEX6 because no common
  hypomorphic allele exists:
  (1) Zellweger Syndrome (ZS, severe): null/null genotype; neonatal onset day 1–7; craniofacial
      dysmorphism (large anterior fontanelle, high forehead, flat face, epicanthal folds);
      profound hypotonia; pachygyria + polymicrogyria (cortical neuronal migration defects —
      PATHOGNOMONIC on MRI, caused by DHA deficiency in fetal neuronal membranes + peroxisomal
      lipid biosynthesis failure); neonatal seizures UNIVERSAL (100%); hepatomegaly + neonatal
      cholestasis; retinopathy (ERG absent); SNHL; adrenal insufficiency; skeletal stippling;
      fatal <6–12 months; supportive/palliative only.
  (2) NALD (intermediate): mild/hypomorphic mutation + null; onset 1st–12th month; seizures
      80–90%; infantile spasms 40–60% + hypsarrhythmia; cholestasis; survival months–years.
  (3) IRD (attenuated): rare in PEX2 (no common hypomorphic allele); mild variant/variant;
      RP + SNHL + peripheral neuropathy; hepatomegaly mild; seizures 20–30%; adult survival.
      Less common than in PEX6 (IRD 20% PEX2 vs 45% PEX6) — reflects allele spectrum.

EPILEPSY IN PEX2-ZSD:
  ZS: 100% epilepsy; neonatal seizures day 1–7; multifocal clonic + tonic + myoclonic +
  electrographic; hypsarrhythmia if surviving infantile period; EEG burst-suppression →
  multifocal spikes; universal drug-resistance; SE frequent; fatal <12M.
  NALD: 80–90% epilepsy; infantile spasms (IS) 40–60% + hypsarrhythmia; focal + generalized +
  myoclonic; moderate drug-resistance.
  IRD (rare): 20–30% epilepsy; focal + GTCS; better AED response; phytol-restricted diet helps.
  DRE: ~65–70% of all PEX2-ZSD (higher than PEX6 due to more severe allele spectrum).

AED PHARMACOLOGY (IDENTICAL TO PEX1/PEX6-ZSD — RING MECHANISM DOES NOT CHANGE AED RULES):
  LEV (Levetiracetam): FIRST-LINE in all ZSD forms; no enzyme induction; no hepatotoxicity;
    IV/PO; neonatal-safe (weight-based dosing); no interaction with DHA or bile acids; SV2A.
  VPA (Valproate): HIGH RISK (THREE independent mechanisms — IDENTICAL to PEX1/PEX6):
    (a) Hepatotoxicity — ZS/NALD have baseline cholestatic liver disease → VPA dramatically
        increases hepatic failure risk; POLG1 exclusion MANDATORY (CPIC Grade A) before VPA;
    (b) VPA inhibits peroxisomal beta-oxidation → worsens VLCFA accumulation in ZSD;
    (c) VPA → carnitine depletion; pre-existing carnitine deficiency in some ZSD.
    → RELATIVE-TO-ABSOLUTE CI depending on liver function; avoid in ZS/NALD.
  VGB (Vigabatrin): HIGH RISK:
    Pre-existing ZSD retinopathy (universal ZS/NALD; 20–30% IRD) + VGB irreversible VF loss =
    additive irreversible blindness. ACTH Level A for IS preferred over VGB.
    AVOID VGB entirely in ZSD unless last resort and patient already has severe visual loss.
  PHT / CBZ / OXC: RELATIVE CI (CYP3A4 enzyme induction → DHA depletion + hepatic burden).
  ACTH: Level A for infantile spasms/hypsarrhythmia (preferred over VGB).
  CLB (Clobazam): Level B; adjunct; no hepatotoxicity; no enzyme induction; safe in ZSD.
  LTG (Lamotrigine): Level C adjunct; glucuronidation (NOT CYP); safer hepatic profile.
  DHA supplementation: Level B (NALD/IRD). No proven disease-modifying therapy.
  Fasting / Anaesthesia: EXTREME HAZARD — adipose phytanic acid release → acute neurotoxicity;
    IV dextrose MANDATORY during any pre-surgical fast. Pre-op: PT/INR + LFT + IV Vit K.
  No ERT (PEX2 is a membrane-embedded E3 ligase — cannot be delivered as enzyme).
  No HSCT (unlike ABCD1/CCALD — no neuroinflammatory window in ZSD to arrest disease).
  Lorenzo's Oil: INEFFECTIVE (peroxisomal matrix import absent — VLCFA cannot enter peroxisome).
"""

import random

random.seed(314)  # reproducible — PEX2 seed


# ── Overview ──────────────────────────────────────────────────────────────────
def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 87,
        "zs_pct": 30,
        "nald_pct": 45,
        "ird_pct": 20,
        "atypical_pct": 5,
        "drug_resistance_pct": 67,
        "on_dha_pct": 52,
        "liver_disease_pct": 52,
        "retinopathy_pct": 83,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "dha_low_pct": 100,
        "omim_gene": "170993",
        "omim_disease": "#614862 / #614863",
        "locus": "8q21.13",
        "inheritance": "Autosomal Recessive (AR), biallelic LOF",
        "common_variant": "p.Arg119* (most frequent nonsense); p.Cys295Ser (RING domain — ablates E3 activity). No common hypomorphic allele (unlike PEX1-G843D / PEX6-R860W) → spectrum skewed severe.",
        "nbs_positive_rate": "~1/80,000 births (C26:0-lyso-PC DBS panel)",
        "disease_mechanism": (
            "PEX2 encodes Peroxin-2 (305 aa), an integral peroxisomal membrane protein with a "
            "cytoplasmic C3HC4 RING finger E3 ubiquitin ligase domain. PEX2 forms the "
            "PEX2–PEX10–PEX12 RING complex in the peroxisomal membrane. After PEX5 (PTS1 "
            "receptor) delivers matrix cargo and becomes lodged in the membrane, the PEX2–PEX10–PEX12 "
            "complex monoubiquitinates PEX5-Cys11 — the essential signal that triggers the "
            "PEX1–PEX6 AAA-ATPase to retrotranslocate PEX5 back to the cytosol for recycling. "
            "PEX2 LOF abolishes this ubiquitination step → PEX5 trapped permanently → ALL "
            "peroxisomal matrix protein import fails. Biochemically identical to PEX1-ZSD and "
            "PEX6-ZSD. Only comprehensive PEX gene panel (NGS) distinguishes PEX2 from PEX1/PEX6. "
            "PEX2 cohorts are skewed toward severe disease (ZS+NALD dominant) because no common "
            "hypomorphic allele exists to produce attenuated IRD phenotype."
        ),
        "key_concepts": [
            "PEX2 = RING finger E3 ubiquitin ligase (PEX2–PEX10–PEX12 complex); DIFFERENT mechanism from PEX1/PEX6 AAA-ATPase — but SAME biochemical outcome (PEX5 trapped → ALL import fails)",
            "PEX2 monoubiquitinates PEX5-Cys11 → obligatory signal for PEX1–PEX6 ATPase to retrotranslocate PEX5; the two arms (E3 ligase + ATPase) are co-dependent",
            "No common hypomorphic allele (vs PEX1-G843D ~30% or PEX6-R860W ~18%) → PEX2 spectrum SKEWED SEVERE (ZS 30% + NALD 45%; IRD only 20%)",
            "p.Arg119* = most frequent PEX2 variant; p.Cys295Ser (zinc-coordinating RING residue) = null-equivalent missense — ablates E3 catalytic activity",
            "~3–5% of all PBD-ZSD; ~20–30 confirmed cases worldwide 2026 (ultra-rare even within ZSD)",
            "Biochemically IDENTICAL to PEX1-ZSD and PEX6-ZSD: VLCFA up + plasmalogens LOW + DHA low + phytanic/pristanic up + pipecolic up. ONLY NGS panel distinguishes.",
            "Plasmalogens (RBC) LOW — KEY DISTINCTION from ABCD1 (plasmalogens NORMAL in ABCD1); ESSENTIAL ZSD screening test",
            "DHA low → impaired neuronal membrane fluidity → failed neuroblast migration from germinal matrix → pachygyria/PMG (ZS PATHOGNOMONIC on MRI)",
            "LEV first-line all ZSD forms (no enzyme induction, no hepatotoxicity, IV/PO, neonatal-safe)",
            "VPA HIGH RISK — THREE mechanisms: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A)",
            "VGB HIGH RISK — retinopathy (universal ZS/NALD) + VGB irreversible VF constriction = additive blindness. ACTH Level A for IS (preferred). AVOID VGB in IRD.",
            "ACTH Level A for infantile spasms/hypsarrhythmia (NALD); DHA supplementation Level B (NALD/IRD)",
            "Fasting = EXTREME HAZARD: adipose phytanic acid mobilization → acute neurotoxicity + seizures; IV dextrose MANDATORY pre-surgical; IV Vit K for cholestatic coagulopathy (ZS/NALD)",
            "No ERT (PEX2 is membrane-embedded E3 ligase — enzyme replacement not feasible); No HSCT (not inflammatory disease unlike ABCD1/CCALD)",
            "Lorenzo's Oil INEFFECTIVE: targets VLCFA elongase but VLCFA cannot enter peroxisome (import absent) — mechanism mismatch",
            "PEX2 locus 8q21.13 is distinct from PEX1 (7q21.2) and PEX6 (6p21.1) — multi-gene ZSD panel essential; single-gene testing misses 95% of ZSD cases",
        ],
        "standards": [
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum: An overview of current diagnosis, clinical manifestations and treatment guidelines. Mol Genet Metab. 2016",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Klouwer FCC et al. Zellweger spectrum disorders: clinical overview and management approach. Orphanet J Rare Dis. 2015",
            "Steinberg SJ et al. Peroxisome biogenesis disorders. Neurology. 2006",
            "Fujiki Y et al. PEX genes and peroxisome biogenesis disorders. Biochim Biophys Acta Mol Cell Res. 2020",
            "Wanders RJ, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006",
            "Platta HW et al. The peroxisomal receptor docking complex: assembly and function. FEBS J. 2014",
            "CPIC VPA–POLG1 guideline. cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/ (CPIC Level A)",
        ],
    }


# ── Breakdown (patients + etiologies + seizures + triggers + treatments + CIs) ─
def get_breakdown():
    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null",
            "pct": 30,
            "n": 12,
            "sex": "M/F 6/6",
            "onset_age": "Neonatal (day 1–5)",
            "seizure_risk": "100% (multifocal clonic + tonic + myoclonic + electrographic)",
            "eeg": "Burst-suppression → multifocal spikes + hypsarrhythmia",
            "mri": "Pachygyria + PMG + periventricular germinolytic cysts + hypomyelination (PATHOGNOMONIC)",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic null/null (two LOF alleles — nonsense, frameshift, splice): p.Arg119*/null "
                "most common. Zero residual PEX2 E3 ligase activity → PEX5 permanently trapped → "
                "complete peroxisomal import failure. Fatal <6–12 months; palliative care only. "
                "Higher ZS% (30%) vs PEX6-ZS (18%) reflects absent hypomorphic allele in PEX2."
            ),
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — mild/null",
            "pct": 45,
            "n": 18,
            "sex": "M/F 9/9",
            "onset_age": "1st–9th month",
            "seizure_risk": "80–90% (infantile spasms 40–60%; focal + myoclonic)",
            "eeg": "Hypsarrhythmia 40–60% + multifocal spikes + background slowing",
            "mri": "Dysmyelination + periventricular WM abnormalities + cerebellar hypoplasia",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Mild/hypomorphic variant + null LOF: partial residual PEX2 E3 activity; some PEX5 "
                "recycling possible. Includes p.Cys295Ser (null-equivalent RING missense) + milder "
                "splice/missense combinations. Survival months–years; DHA supplementation Level B; "
                "ACTH Level A for infantile spasms; no disease-modifying therapy proven."
            ),
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — hypomorphic/hypomorphic",
            "pct": 20,
            "n": 8,
            "sex": "M/F 4/4",
            "onset_age": "Toddler–early school age (2–7 years)",
            "seizure_risk": "20–30% (focal + GTCS; better AED response than ZS/NALD)",
            "eeg": "Focal spikes + generalized slowing; no hypsarrhythmia",
            "mri": "Mild WM abnormalities; cerebellar atrophy (later); cortical migration near-normal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Rare in PEX2: requires two partial-function alleles — but PEX2 has no common "
                "hypomorphic allele like PEX6-R860W; IRD cases represent novel mild missense "
                "combinations. RP + SNHL + peripheral neuropathy; hepatomegaly mild; intellectual "
                "disability variable; survival adolescence–adulthood; phytol-restricted diet + DHA."
            ),
        },
        {
            "name": "Atypical / Late-Onset ZSD (PEX2)",
            "pct": 5,
            "n": 2,
            "sex": "M/F 1/1",
            "onset_age": "Late childhood–early adulthood",
            "seizure_risk": "15–25% (focal only)",
            "eeg": "Focal spikes temporal/occipital",
            "mri": "Isolated cerebellar atrophy or normal for age",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Ultra-rare: novel mild missense outside RING domain retaining partial E3 activity. "
                "Adult presentation; biochemical diagnosis on targeted panel after RP/SNHL workup."
            ),
        },
    ]

    # 40 patients
    phenotype_pool = (
        ["ZS (Zellweger-Severe)"] * 12
        + ["NALD (Neonatal-ALD-Intermediate)"] * 18
        + ["IRD (Infantile-Refsum-Attenuated)"] * 8
        + ["Atypical / Late-Onset ZSD"] * 2
    )
    genotype_map = {
        "ZS (Zellweger-Severe)": [
            "p.Arg119*/p.Arg119*", "p.Arg119*/c.355+1G>T", "p.Arg119*/p.Gln104*",
            "p.Arg119*/c.402delA", "c.354G>A/p.Arg119*",
        ],
        "NALD (Neonatal-ALD-Intermediate)": [
            "p.Cys295Ser/p.Arg119*", "p.Leu221Pro/p.Arg119*", "p.Arg119*/c.503-2A>G",
            "p.Gly266Arg/p.Arg119*", "p.Val271Ala/c.402delA",
        ],
        "IRD (Infantile-Refsum-Attenuated)": [
            "p.Leu221Pro/p.Gly266Arg", "p.Val271Ala/p.Leu221Pro",
            "p.Gly266Arg/p.Gly266Arg",
        ],
        "Atypical / Late-Onset ZSD": [
            "p.Ala197Val/p.Leu221Pro",
        ],
    }
    aed_pool = ["LEV", "LEV+CLB", "LEV+LTG", "ACTH→LEV", "LEV+PHB"]
    response_map = {
        "ZS (Zellweger-Severe)": ["Drug-resistant", "Drug-resistant", "Drug-resistant"],
        "NALD (Neonatal-ALD-Intermediate)": ["Drug-resistant", "Partially controlled", "Partially controlled"],
        "IRD (Infantile-Refsum-Attenuated)": ["Controlled", "Partially controlled", "Drug-resistant"],
        "Atypical / Late-Onset ZSD": ["Controlled"],
    }

    patients = []
    for i, ph in enumerate(phenotype_pool):
        geno_list = genotype_map.get(ph, ["Unknown"])
        resp_list = response_map.get(ph, ["Partially controlled"])
        is_zs_nald = ph in ("ZS (Zellweger-Severe)", "NALD (Neonatal-ALD-Intermediate)")
        has_sz = random.random() < (1.0 if ph == "ZS (Zellweger-Severe)"
                                    else 0.85 if ph == "NALD (Neonatal-ALD-Intermediate)"
                                    else 0.25 if ph == "IRD (Infantile-Refsum-Attenuated)"
                                    else 0.2)
        patients.append({
            "patient_id": f"PEX2-{i+1:03d}",
            "phenotype": ph,
            "sex": random.choice(["M", "F"]),
            "genotype": random.choice(geno_list),
            "liver_disease": is_zs_nald and random.random() < 0.80,
            "has_seizures": has_sz,
            "primary_aed": random.choice(aed_pool) if has_sz else None,
            "drug_response": random.choice(resp_list) if has_sz else None,
            "on_dha": ph != "ZS (Zellweger-Severe)" and random.random() < 0.65,
        })

    seizure_types = [
        {"type": "Neonatal multifocal clonic (ZS — neonatal day 1–7)", "pct": 30,
         "eeg": "Burst-suppression → multifocal spike-wave; high amplitude; background suppression; poor prognosis"},
        {"type": "Infantile spasms / West syndrome (NALD — 1–6 months)", "pct": 45,
         "eeg": "Hypsarrhythmia (40–60% NALD) + modified hypsarrhythmia; ACTH Level A response"},
        {"type": "Tonic seizures (ZS + NALD)", "pct": 38,
         "eeg": "Generalized voltage attenuation → fast recruiting EEG; NICU monitoring required"},
        {"type": "Myoclonic seizures (NALD + IRD)", "pct": 32,
         "eeg": "Generalized polyspike-wave 3–5 Hz; photosensitivity variable"},
        {"type": "Focal seizures (IRD + Atypical)", "pct": 22,
         "eeg": "Focal spikes temporal/occipital/parietal; better AED response in IRD"},
        {"type": "Generalized tonic-clonic (IRD)", "pct": 18,
         "eeg": "Generalized polyspike-wave; rare in IRD; controlled with LEV/LTG"},
        {"type": "Status epilepticus (ZS + NALD)", "pct": 26,
         "eeg": "Non-convulsive SE common in ZS; requires IV LEV/PHB; refractory to benzodiazepines"},
    ]

    triggers = [
        {"trigger": "Febrile illness (ZS + NALD)", "pct": 75,
         "note": "Most common trigger; any fever dramatically lowers seizure threshold in ZSD; aggressive antipyretic management; IV LEV prophylaxis during febrile illness"},
        {"trigger": "Fasting / NPO / Anaesthesia", "pct": 70,
         "note": "EXTREME HAZARD: starvation → adipose phytanic acid mobilisation → acute metabolic neurotoxicity + seizure precipitation; IV dextrose MANDATORY during any pre-surgical fast; cancel elective procedures if IV access not achievable"},
        {"trigger": "Phytanic acid surge (dietary non-compliance — IRD)", "pct": 52,
         "note": "IRD: dietary phytol intake (meat fats, dairy, fish oils) → phytanic acid accumulates → proconvulsant effect; strict phytol-restricted diet Level B in IRD/Atypical"},
        {"trigger": "Metabolic decompensation (hepatic)", "pct": 50,
         "note": "Cholestatic liver dysfunction (ZS/NALD) → hyperammonaemia + coagulopathy + electrolyte disturbance → seizure precipitant; monitor LFT + ammonia + INR q2–4 weeks"},
        {"trigger": "Sleep deprivation (IRD)", "pct": 28,
         "note": "Less severe trigger than in ZS/NALD; IRD patients report sleep-deprived focal seizures; normal sleep hygiene counselling"},
        {"trigger": "Anaesthesia peri-operative period", "pct": 25,
         "note": "Phytanic acid mobilisation from adipose during surgical fasting; pre-op IV glucose loading + IV Vitamin K (cholestatic coagulopathy) + LFT + PT/INR mandatory"},
        {"trigger": "VPA initiation (iatrogenic)", "pct": 18,
         "note": "VPA HIGH RISK: acute hepatic failure (cholestatic baseline) and/or peroxisomal BO inhibition exacerbating VLCFA accumulation; POLG1 MANDATORY before any VPA attempt"},
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
        "ERG (electroretinogram) — at diagnosis + q6M; retinal degeneration monitoring; guides VGB decision (contraindicated if ANY ERG abnormality)",
        "Audiological evaluation (BERA/ABR + pure tone) — at diagnosis + q12M; SNHL universal in ZS/NALD",
        "Adrenal function (ACTH stimulation / basal cortisol) — at diagnosis; adrenal insufficiency (may require hydrocortisone perioperatively)",
        "Skeletal survey (radiograph) — at diagnosis (ZS): chondrodysplasia punctata (stippled patella/acetabulum)",
        "Comprehensive PEX gene panel (NGS) — mandatory at diagnosis to confirm PEX2 vs PEX1/PEX6/PEX10/PEX12; biochemically identical",
        "POLG1 sequencing — MANDATORY before any VPA trial (CPIC Grade A); POLG1 LOF + VPA = catastrophic hepatotoxicity",
    ]

    thresholds = [
        {"parameter": "C26:0-lyso-PC (DBS, NBS)", "threshold": ">0.60 μmol/L", "action": "Reflexive full VLCFA panel + PEX gene panel; initiate DHA immediately (NALD); contact metabolic genetics urgently"},
        {"parameter": "C26:0 plasma (VLCFA)", "threshold": ">1.30 μg/mL", "action": "Confirm ZSD: add plasmalogens (RBC) + pipecolic acid + DHA; if plasmalogens LOW → ZSD (not ABCD1); gene panel"},
        {"parameter": "Plasmalogen ratio (RBC)", "threshold": "<0.070 (C16-DMA/C16:0)", "action": "Confirms ZSD vs ABCD1; combined with elevated C26:0 → all PBD-ZSD included; gene panel to distinguish PEX gene"},
        {"parameter": "DHA (plasma)", "threshold": "<100 μmol/L", "action": "Start DHA supplementation (NALD/IRD): 20–50 mg/kg/day; recheck q3M until stable"},
        {"parameter": "ALT / AST", "threshold": ">3× ULN", "action": "Hold VPA (if any); increase LFT monitoring to weekly; GI/hepatology consultation; consider ursodeoxycholic acid (Level C)"},
        {"parameter": "INR (coagulopathy)", "threshold": ">1.5 pre-procedure", "action": "IV Vitamin K 1 mg/kg (max 10 mg) × 3 doses; postpone elective surgery; FFP if INR >2.5 + active bleeding"},
        {"parameter": "Phytanic acid (plasma — IRD)", "threshold": ">200 μmol/L", "action": "Strict phytol-restricted diet counselling; dietitian consultation; IV glucose if acute surge during illness"},
        {"parameter": "Ammonia", "threshold": ">80 μmol/L", "action": "Withhold VPA (if prescribed); protein restriction; lactulose; N-acetylcysteine; hepatology consultation"},
    ]

    lifecycle = [
        {
            "stage": "Stage 1 — Neonatal (Birth to 1 Month) — ZS DOMINATED",
            "features": "Dysmorphic features (large fontanelle, flat face, epicanthal folds); profound hypotonia; neonatal seizures day 1–5 (100% ZS); hepatomegaly; cholestasis; ERG absent; adrenal insufficiency; NICU level care",
            "action": "NBS C26:0-lyso-PC + emergency VLCFA panel + PEX gene panel; IV LEV loading (20–30 mg/kg); IV glucose (avoid fasting); IV Vit K; adrenal testing + hydrocortisone if crisis; palliative care discussion (ZS: fatal prognosis)",
        },
        {
            "stage": "Stage 2 — Infantile (1–12 Months) — NALD DOMINATED",
            "features": "Infantile spasms (40–60%) + hypsarrhythmia; cholestasis; hepatomegaly; retinopathy developing; SNHL confirmed; DHA low; psychomotor delay; moderate-severe disease",
            "action": "ACTH (Level A, 20–40 U/day IM × 2 weeks → taper) for infantile spasms; DHA supplementation (20–50 mg/kg/day); phytol-restricted diet begins; ERG + audiology; VPA AVOID; VGB AVOID; LEV maintenance",
        },
        {
            "stage": "Stage 3 — Early Childhood (1–5 Years) — NALD/IRD TRANSITION",
            "features": "Ongoing seizures (focal + myoclonic in NALD/IRD); progressive demyelination (WM MRI); cerebellar atrophy; SNHL + RP evolving; intellectual disability (variable severity); hepatomegaly stabilising in IRD",
            "action": "Optimise AED (LEV ± CLB ± LTG); DHA maintenance; quarterly VLCFA + LFT; physiotherapy + occupational therapy; hearing aids; vision aids; special education planning; genetic counselling (AR 25% recurrence risk for siblings)",
        },
        {
            "stage": "Stage 4 — Later Childhood (5–12 Years) — IRD DOMINANT",
            "features": "RP progression (tunnel vision); SNHL severe in many; peripheral neuropathy (gait disturbance); seizures often well-controlled in IRD; liver disease quiescent; intellectual disability stable",
            "action": "Annual ophthalmology (VF + OCT + ERG); audiological rehabilitation; gait physiotherapy; school integration; DHA + phytol-restricted diet; AVOID ENZYME INDUCING AEDs (CYP3A4 → DHA depletion); LEV/LTG/CLB preferred",
        },
        {
            "stage": "Stage 5 — Adolescent/Adult (>12 Years) — IRD/ATYPICAL ONLY",
            "features": "Advanced RP (often legally blind); severe SNHL; peripheral polyneuropathy; seizures 20–30%; hepatomegaly resolved; intellectual disability moderate; azoospermia possible (males, peroxisomal lipid defect)",
            "action": "Cochlear implant evaluation; low-vision rehabilitation; anti-epileptic therapy (LEV/LTG preferred, no CYP inducers); genetic counselling for family planning (AR inheritance); phytol-restricted diet lifelong",
        },
        {
            "stage": "Stage 6 — End-of-Life / Palliative (ZS/Severe NALD)",
            "features": "ZS: fatal <6–12 months; NALD (severe): deterioration over months–years; respiratory failure; hepatic failure; intractable seizures; comfort-focused care priorities",
            "action": "Palliative care team involvement from diagnosis (ZS); goals-of-care discussion; comfort AED dosing (IV LEV + oral CLB); feeding support (NGT/PEG in NALD); home hospice if appropriate; bereavement support",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "Level A — First-line (all ZSD forms)",
            "evidence": "Universal first-line AED in PBD-ZSD regardless of form; no enzyme induction (no DHA depletion); no hepatotoxicity; IV/PO; neonatal-safe (weight-based loading 20–30 mg/kg IV); SV2A mechanism",
            "dose": "Neonatal: 10–20 mg/kg/day IV → titrate; Infant/Child: 20–40 mg/kg/day (max 60 mg/kg/day); IV for SE loading",
            "moa": "SV2A (synaptic vesicle glycoprotein 2A) — inhibits presynaptic neurotransmitter release; no CYP induction",
            "monitoring": "Serum level 20–40 μg/mL (optional, clinical titration primary); CBC (rare haematological effects); behavioural side effects (irritability in NALD/ZS — use oral ZipLEV formulation if tolerated)",
            "ci": "None in ZSD — safest AED for this population",
        },
        {
            "drug": "ACTH (Adrenocorticotropic Hormone)",
            "class": "Level A — Infantile spasms (NALD; ZS if surviving infantile period)",
            "evidence": "First-line for infantile spasms/West syndrome in PBD-ZSD; preferred over VGB due to universal ZSD retinopathy; UKISS, WEST, INFANT trials support ACTH in IS",
            "dose": "20–40 U/day IM × 14 days → 20 U every other day × 14 days → taper; monitor BP, glucose, infection; cortisol suppression expected (taper protocol mandatory)",
            "moa": "MC2R agonist (adrenal cortex) → glucocorticoid → anti-inflammatory/anti-epileptic; direct melanocortin CNS effects on IS",
            "monitoring": "BP, glucose, potassium, infection signs daily during high-dose phase; growth monitoring during taper; adrenal axis recovery post-taper",
            "ci": "Active untreated infection; uncontrolled diabetes; severe osteoporosis",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "Level B — Adjunct (all ZSD forms)",
            "evidence": "Adjunct for focal + myoclonic + tonic seizures; 1,5-benzodiazepine (less sedating than clonazepam for chronic use); no hepatotoxicity; no enzyme induction; safe in ZSD liver disease",
            "dose": "0.1–0.3 mg/kg/day in 2 divided doses (max 1.0 mg/kg/day); start low, titrate",
            "moa": "GABA-A allosteric modulator (benzodiazepine site); 1,5-BZD isomer preferred for chronic use vs 1,4-BZD clonazepam",
            "monitoring": "Sedation; tolerance development over months; gradual taper required for discontinuation",
            "ci": "None specific in ZSD; avoid abrupt discontinuation (withdrawal seizures)",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "Level C — Adjunct (NALD/IRD focal + GTCS)",
            "evidence": "Adjunct for focal and GTCS in NALD/IRD; hepatic glucuronidation (NOT CYP2D6/3A4) → safer hepatic profile vs CBZ/PHT; no DHA depletion; slow titration mandatory (Steven-Johnson risk)",
            "dose": "0.15 mg/kg/day × 2 weeks → 0.3 mg/kg/day × 2 weeks → titrate to 2–10 mg/kg/day (slow titration to avoid SJS); faster titration if on valproate interaction (but VPA HIGH RISK in ZSD — avoid combination)",
            "moa": "Na+ channel blocker (voltage-gated) + Ca2+ channel modulation; broad-spectrum AED",
            "monitoring": "Rash monitoring — slow titration critical; LFT (glucuronidation — minimal hepatic risk vs CYP-based AEDs); levels optional",
            "ci": "Avoid rapid titration; avoid in combination with VPA (VPA inhibits LTG glucuronidation → LTG toxicity — and VPA HIGH RISK in ZSD anyway)",
        },
        {
            "drug": "DHA (Docosahexaenoic Acid) Supplementation",
            "class": "Level B — Disease-modifying (NALD/IRD)",
            "evidence": "Restores DHA-depleted neuronal and retinal membranes; Level B evidence for NALD/IRD (stabilises retinal function, may slow CNS progression); not proven disease-modifying in ZS (fatal prognosis dominates); peroxisome-bypassed supplementation (dietary DHA absorbed intestinally as LPC)",
            "dose": "20–50 mg/kg/day (ethyl ester or lysophospholipid form); administer with meals (fat-soluble absorption); recheck plasma DHA q3M and titrate to target 100–150 μmol/L",
            "moa": "Restores n-3 PUFA in neuronal/retinal membranes; improves membrane fluidity; may partially rescue neuronal migration in utero (prenatal supplementation in diagnosed pregnancies — experimental); not a cure",
            "monitoring": "Plasma DHA q3M; plasma total PUFA; ERG (can track retinal DHA response); no known toxicity at therapeutic doses",
            "ci": "None in ZSD; ensure formulation avoids fish-oil phytanic acid contamination (some fish oils contain phytanic acid — use purified algal DHA)",
        },
        {
            "drug": "Phytol-Restricted Diet",
            "class": "Level B — Dietary (IRD/Atypical/NALD where surviving)",
            "evidence": "Reduces dietary phytanic acid load (phytol → phytanic acid via intestinal metabolism); essential in IRD (phytanic acid proconvulsant + neurotoxic at high levels); restrict red meat/dairy fat/fish — principal phytol sources; dietitian-led",
            "dose": "Dietary: restrict phytol-containing foods to <10 mg phytol/day; consult specialist dietitian; vegetable oils (phytol-free) + lean proteins allowed; caloric adequacy essential (fasting = EXTREME HAZARD)",
            "moa": "Reduces phytol substrate available for intestinal conversion → phytanic acid; lowers plasma phytanic acid → reduces neuromodulatory/proconvulsant effects",
            "monitoring": "Plasma phytanic acid q6M (target <200 μmol/L IRD, <50 μmol/L if stable); nutritional status (protein, calories, vitamin K); dietitian review q6–12M",
            "ci": "None; MANDATORY to maintain caloric adequacy — fasting/caloric restriction = phytanic acid mobilisation from adipose = EXTREME HAZARD; IV glucose available if caloric intake uncertain",
        },
        {
            "drug": "Cholic Acid (Ursodeoxycholic Acid — UDCA)",
            "class": "Level C — Experimental supportive (ZS/NALD cholestatic liver disease)",
            "evidence": "Level C; may reduce cholestatic burden from DHCA/THCA bile acid intermediates; improves bile flow; used in bile acid synthesis disorders as disease-modifying therapy — in ZSD, supportive rather than curative",
            "dose": "Cholic acid: 5–10 mg/kg/day PO; UDCA: 10–20 mg/kg/day PO in 2–3 divided doses",
            "moa": "Cholic acid: exogenous primary bile acid replaces toxic DHCA/THCA; competitively reduces cholestatic intermediates. UDCA: hydrophilic protective bile acid; cholestasis palliation",
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
                "fatty acid oxidation → worsens VLCFA accumulation in ZSD (additional burden on already-failed "
                "peroxisomal system); "
                "(3) CARNITINE DEPLETION — VPA sequesters acylcarnitines → depletes free carnitine; "
                "pre-existing carnitine deficiency in some ZSD patients worsens. "
                "POLG1 sequencing MANDATORY (CPIC Grade A) before ANY VPA use: POLG1 LOF + VPA = "
                "catastrophic hepatotoxicity (Alpers-Huttenlocher syndrome). "
                "Avoid in ZS (uniformly fatal prognosis, hepatotoxicity risk unconscionable). "
                "Extreme caution in NALD (near-absolute CI). IRD: use only if LEV+CLB+LTG all fail, "
                "LFT baseline normal, POLG1 negative; LFT weekly if used."
            ),
            "alternative": "LEV (first-line) + CLB (Level B) + LTG (Level C); ACTH for infantile spasms (Level A)",
        },
        {
            "drug": "VGB (Vigabatrin)",
            "level": "HIGH RISK — Additive Irreversible Blindness",
            "reason": (
                "ZSD retinopathy is UNIVERSAL in ZS and NALD (ERG absent); 20–30% in IRD. "
                "VGB causes irreversible, cumulative, concentric visual field (VF) constriction in "
                "~30–50% of patients on long-term therapy (SABRIL REMS). "
                "VGB + pre-existing ZSD retinopathy = ADDITIVE IRREVERSIBLE BLINDNESS. "
                "ACTH (Level A) is the preferred treatment for infantile spasms in ZSD — avoids this risk entirely. "
                "AVOID VGB in IRD (where visual field is still partially preservable). "
                "Last resort in NALD only if ACTH fails AND patient is already severely visually impaired: "
                "ophthalmological baseline (VF + OCT + ERG) mandatory before use; VF monitoring q3M (OCT-based in non-verbal patients)."
            ),
            "alternative": "ACTH Level A (infantile spasms — IS); LEV + CLB for other seizure types",
        },
        {
            "drug": "PHT / CBZ / OXC (Hepatic Enzyme Inducers)",
            "level": "RELATIVE CI — DHA Depletion + Hepatic Burden",
            "reason": (
                "CYP3A4 induction by PHT/CBZ/OXC: (1) accelerates DHA metabolism → further reduces "
                "already-low plasma DHA → worsens retinopathy and neuronal membrane stability; "
                "(2) additional hepatic CYP burden compounds cholestatic liver disease (ZS/NALD). "
                "NOTE: unlike ABCD1 (X-ALD), there is NO adrenal crisis mechanism in ZSD — "
                "PHT/CBZ are NOT absolutely contraindicated by adrenal mechanism in ZSD (distinction from ABCD1). "
                "Relative CI only: use if LEV/LTG/CLB fail; monitor DHA + LFT; IV LEV should replace "
                "fosphenytoin in ZSD status epilepticus protocols."
            ),
            "alternative": "LEV IV (replaces fosphenytoin in SE); LTG (glucuronidation, not CYP); CLB adjunct",
        },
        {
            "drug": "Fasting / Anaesthesia / Surgical NPO",
            "level": "EXTREME HAZARD — Phytanic Acid Mobilisation Crisis",
            "reason": (
                "Any period of fasting (including surgical NPO) → adipose tissue lipolysis → "
                "phytanic acid release into circulation (phytanic acid is stored in adipose in ZSD, "
                "especially in IRD/NALD where patients survive long enough to accumulate it). "
                "Acute phytanic acid surge → acute neurological crisis + seizure precipitation. "
                "ZS/NALD: severe cholestatic coagulopathy (INR elevated) → surgical haemorrhage risk. "
                "MANDATORY pre-operative protocol: IV glucose 10% minimum 4 hours before NPO; "
                "IV Vitamin K 1 mg/kg pre-op (cholestatic coagulopathy); PT/INR + LFT pre-op; "
                "anaesthesia team briefed; minimal fasting duration; IV glucose throughout procedure."
            ),
            "alternative": "IV glucose 10% during NPO; minimal fasting; IV Vitamin K; anaesthesia liaison",
        },
        {
            "drug": "Lorenzo's Oil (glyceryl trioleate + glyceryl trierucate)",
            "level": "INEFFECTIVE — Mechanism Mismatch",
            "reason": (
                "Lorenzo's Oil was designed for ABCD1 (X-ALD) where peroxisomal beta-oxidation is "
                "partially functional — it reduces VLCFA elongase substrate and lowers plasma C26:0. "
                "In ZSD (including PEX2-ZSD), ALL peroxisomal import is absent — VLCFA cannot enter "
                "the peroxisome even if elongase activity is reduced at the ER. Lorenzo's Oil therefore "
                "has no meaningful effect on VLCFA in ZSD. Do not use in PBD-ZSD."
            ),
            "alternative": "DHA supplementation (Level B, NALD/IRD); phytol-restricted diet (IRD); palliative AED (ZS)",
        },
        {
            "drug": "HSCT (Haematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED — No Neuroinflammatory Window",
            "reason": (
                "HSCT corrects ABCD1 (X-ALD) / CCALD by replacing CNS microglia with donor cells "
                "that arrest inflammatory demyelination — but ONLY in an early pre-symptomatic or "
                "minimally symptomatic window. ZSD (PEX2) is NOT an inflammatory disease: the brain "
                "damage is due to failed neuronal migration (pachygyria) during fetal development and "
                "progressive demyelination from metabolic failure — NOT from microglial inflammation. "
                "There is no neuroinflammatory target for HSCT to correct in ZSD. Do not refer for HSCT."
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
        "PEX2 — Peroxin-2 (305 aa, 8q21.13): integral peroxisomal membrane protein with cytoplasmic C3HC4 RING finger E3 ubiquitin ligase domain; obligate member of PEX2–PEX10–PEX12 RING complex",
        "PEX2–PEX10–PEX12 RING complex: three-membered E3 ubiquitin ligase complex embedded in the peroxisomal membrane; catalyzes both monoubiquitination (PEX5 recycling) and polyubiquitination (PEX5 degradation) of PEX5",
        "PEX5 monoubiquitination (Cys11): PRODUCTIVE recycling signal — PEX2-PEX10-PEX12 monoubiquitinates PEX5-Cys11 → PEX1–PEX6 AAA-ATPase recognizes this tag → retrotranslocates PEX5 to cytosol for next import cycle",
        "PEX5 polyubiquitination (Lys): QUALITY CONTROL — PEX2-PEX10-PEX12 polyubiquitinates damaged/stalled PEX5-Lys → proteasomal degradation of irreversibly trapped PEX5; prevents membrane clogging",
        "E3 Ligase vs AAA-ATPase co-dependence: the RING finger (E3 ubiquitin, PEX2 arm) and AAA-ATPase (PEX1/PEX6 arm) are co-required — failure of EITHER arm (E3 cannot tag PEX5, OR ATPase cannot extract tagged PEX5) → complete PEX5 trapping → identical biochemical outcome",
        "No hypomorphic founder allele in PEX2: unlike PEX1-G843D (~30%) and PEX6-R860W (~18%), PEX2 has no common partial-function allele → most PEX2 patients carry two severe alleles → phenotypic distribution skewed severe (ZS 30% + NALD 45% + IRD only 20%)",
        "p.Cys295Ser — RING finger missense, null-equivalent: Cys295 is a zinc-coordinating residue of the C3HC4 RING domain; any missense at a zinc-coordination position ablates E3 catalytic activity = functionally null (not hypomorphic); critical for genotype interpretation",
        "Biochemical identity with PEX1/PEX6-ZSD: VLCFA up + plasmalogens (RBC) LOW + DHA low + phytanic/pristanic up + pipecolic up + DHCA/THCA up — cannot distinguish from PEX1-ZSD or PEX6-ZSD on biochemical panel alone; NGS gene panel mandatory",
        "C26:0-lyso-PC (DBS, NBS): newborn screening biomarker; C26:0 very long-chain fatty acid esterified to lysophosphatidylcholine; elevated in ALL PBD-ZSD (including PEX2); sensitivity ~98% for ZS/NALD; enables pre-symptomatic diagnosis in NBS programs",
        "Erythrocyte plasmalogens (DMA-16:0 / DMA-18:0): ether-phospholipids synthesised only in peroxisomes (GNPAT + AGPS + FAR1 pathway — PTS1 enzymes); LOW in ALL PBD-ZSD but NORMAL in ABCD1 → essential test to distinguish ZSD from XLD at biochemical level",
        "DHA deficiency mechanism: peroxisomal import of ACSL (long-chain acyl-CoA synthetases) fails → DHA synthesis (retroconversion from C22:6 in peroxisomes) impaired → DHA low in neuronal/retinal membranes → (i) retinopathy (ERG absent in ZS/NALD) + (ii) impaired neuroblast migration (pachygyria/PMG in ZS — fetal brain DHA-dependent)",
        "Pachygyria / PMG in ZS (PATHOGNOMONIC): neuronal migration from germinal matrix requires DHA in neuronal membranes AND peroxisomal lipid biosynthesis for correct timing; PEX2-ZS → fetal DHA deficiency → failed migration → broad simplified gyri (pachygyria) + polymicrogyria; PATHOGNOMONIC on MRI for ZS",
        "VPA–ZSD: THREE independent contraindication mechanisms — hepatotoxicity (cholestatic liver baseline), peroxisomal beta-oxidation inhibition (worsens VLCFA), carnitine depletion; POLG1 LOF + VPA = fatal Alpers-Huttenlocher syndrome (CPIC Grade A mandatory screening)",
        "VGB–ZSD retinopathy: vigabatrin causes irreversible concentric VF constriction in 30–50% long-term; ZSD retinopathy is universal in ZS/NALD → additive permanent blindness; ACTH (Level A) preferred for IS; VGB never first/second-line in ZSD",
        "Fasting/adipose phytanic hazard: phytanic acid accumulates in adipose in surviving ZSD patients (IRD/NALD); fasting → adipose lipolysis → phytanic acid surge → acute neurotoxicity + seizure crisis; IV glucose MANDATORY during any NPO period; IV Vitamin K for cholestatic coagulopathy; cannot be overemphasised",
        "PEX2 ultra-rarity: ~3–5% of all PBD-ZSD; ~20–30 confirmed cases worldwide 2026; institutional experience limited; refer to expert ZSD center (Kennedy Krieger, University of Amsterdam, Radboud, UCL) for management",
    ]

    diagnostic_algorithm = [
        "Step 1 — Clinical suspicion: neonatal hypotonia + facial dysmorphism + seizures day 1–7 (ZS) OR infantile spasms + hepatomegaly (NALD) OR adult RP + SNHL + cerebellar ataxia (IRD-like) → immediate VLCFA panel",
        "Step 2 — VLCFA panel (plasma): C26:0 elevated? C26:0/C22:0 ratio elevated? → confirms peroxisomal VLCFA oxidation failure → proceed to Step 3",
        "Step 3 — Erythrocyte plasmalogens (DMA-16:0, DMA-18:0): LOW → ZSD (PBD-ZSD); NORMAL → ABCD1 (XLD) or single-enzyme deficiency (ACOX1/HSD17B4); this single test separates ZSD from ABCD1",
        "Step 4 — Additional metabolites: pipecolic acid (elevated in ALL ZSD); phytanic + pristanic acid (elevated in ZSD; phytanic-only elevation = PHYH adult Refsum); DHA (low in ZSD); DHCA+THCA bile acids (elevated in ZSD)",
        "Step 5 — NBS result review: C26:0-lyso-PC DBS elevated? → ZSD panel reflexed; confirm with full biochemical panel; NBS false-negative rare but possible in mild alleles",
        "Step 6 — Brain MRI (1.5T/3T): ZS = pachygyria + PMG + periventricular germinolytic cysts + hypomyelination (PATHOGNOMONIC); NALD = WM dysmyelination + cerebellar hypoplasia; IRD = cerebellar atrophy (late)",
        "Step 7 — Comprehensive PEX gene panel (NGS): test PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26 simultaneously; biochemically identical — only gene panel distinguishes; ACMG-classified variants",
        "Step 8 — Ophthalmological assessment: ERG (flash ERG in neonates) — ERG absent/severely reduced = ZSD retinopathy (universal in ZS/NALD); VF testing in cooperative children; OCT for retinal layer assessment in NALD/IRD",
        "Step 9 — POLG1 sequencing: MANDATORY before any VPA consideration (CPIC Grade A); POLG1 LOF = absolute VPA contraindication (fatal Alpers-Huttenlocher syndrome risk)",
        "Step 10 — Cardiac evaluation: ECG + echocardiogram (congenital heart defects in ZS — 20–30%; ASD, VSD, PDA; may be incidental or clinically significant)",
        "Step 11 — Adrenal function testing: ACTH stimulation test + basal cortisol; adrenal insufficiency in ZS (from peroxisomal steroidogenesis failure) → hydrocortisone replacement perioperatively",
        "Step 12 — Multidisciplinary team assembly: neurometabolic specialist + neurologist + hepatologist + ophthalmologist + audiologist + geneticist + dietitian + palliative care (ZS) + physiotherapy/OT; refer to ZSD expert centre",
    ]

    pharmacological_distinctions = [
        "PEX2-ZSD vs ABCD1 (X-ALD): LEV preferred in both; PHT/CBZ are ABSOLUTE CI in ABCD1 (adrenal crisis — glucocorticoid synthesis requires VLCFA oxidation in adrenal cortex), but only RELATIVE CI in ZSD (DHA depletion + hepatic — NOT adrenal mechanism in ZSD). Critical distinction.",
        "VPA in ZSD vs VPA in RCDP/PEX7: In RCDP (PEX7) the VLCFA pathway is NORMAL (only PTS2 import affected); VPA still RELATIVE CI in PEX7 (hepatotoxicity) but does NOT inhibit peroxisomal VLCFA oxidation (VLCFA pathway intact in RCDP). In ZSD ALL pathways impaired — VPA adds peroxisomal BO inhibition on top of hepatotoxicity = higher risk.",
        "VGB in ZSD vs VGB in SSADH (ALDH5A1): VGB ABSOLUTE CI in SSADH (metabolic mechanism — inhibits GABA-T → more SSA → more GHB → worsens 4-hydroxybutyricaciduria). In ZSD, VGB is HIGH RISK (retinopathy mechanism — NOT metabolic; GABA-T pathway not disrupted in ZSD). Different CI mechanisms for same drug.",
        "ACTH in ZSD: LEVEL A for infantile spasms (IS) — preferred over VGB because of ZSD retinopathy. In ABCD1, ACTH is NOT a specific treatment. Key: ACTH for IS in ANY peroxisomal disorder with ZS/NALD phenotype where VGB is contraindicated by retinopathy.",
        "DHA supplementation: LEVEL B in ZSD (NALD/IRD); no proven benefit in ZS (fatal prognosis). Not applicable to ABCD1 (DHA pathway functional in ABCD1 — only VLCFA transport affected). DHA bypasses peroxisomal synthesis by intestinal absorption as LPC-DHA.",
        "Lorenzo's Oil: EFFECTIVE in ABCD1 (partial VLCFA elongase inhibition reduces C26:0 synthesis); INEFFECTIVE in PEX2-ZSD (peroxisomal import absent — VLCFA cannot be oxidised regardless of elongase activity reduction). Mechanism-specific: targets elongase, not import.",
        "PHT/CBZ/OXC in ZSD vs PEX7 (RCDP): Both are RELATIVE CI in ZSD (CYP3A4 → DHA depletion + hepatic). In PEX7/RCDP: plasmalogens LOW but VLCFA/DHA NORMAL — PHT/CBZ cause less harm to DHA (already normal in RCDP) but still hepatic concern; slightly less restrictive in RCDP than ZSD.",
        "Fasting hazard: HIGHEST risk in ZSD (phytanic + VLCFA mobilisation from adipose) and Adult Refsum/PHYH (pure phytanic). Moderate risk in RCDP/PEX7 (phytanic + plasmalogens disrupted). Minimal risk in ABCD1 (VLCFA in adrenal/CNS not significantly mobilised by short fasting in most patients).",
        "LEV in ALL peroxisomal epilepsies: universal first-line (ZSD, RCDP, ACOX1, ABCD1, SSADH, Adult Refsum). No CYP induction, no hepatotoxicity, no DHA depletion, neonatal IV formulation, SV2A mechanism. The only AED with universal safety across all peroxisomal/metabolic epilepsies.",
        "Retinopathy-directed VGB avoidance: RP present in ZSD (ZS/NALD universal) + PHYH (Adult Refsum, RP 95%) + ACOX1 (ERG 85%) + RCDP/PEX7 (cataracts/RP 70%) → VGB ABSOLUTE or HIGH RISK in ALL these conditions. LEV/CLB replace VGB in seizure management across the entire peroxisomal spectrum.",
        "Carnitine supplementation: May be indicated if VPA is used (despite HIGH RISK status) in IRD — L-carnitine 50–100 mg/kg/day; also beneficial for acylcarnitine sequestration from VLCFA-CoA esters in ZSD. Baseline carnitine measurement recommended in all ZSD patients.",
        "Antenatal counselling: AR (25% sibling recurrence); prenatal diagnosis by CVS/amniocentesis + PEX gene panel; NBS C26:0-lyso-PC enables pre-symptomatic diagnosis in siblings; parental carrier testing mandatory after PEX2 proband confirmed; genetic counsellor essential for family planning.",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX1-ZSD (most common ZSD — 65% of all PBD)",
            "distinction": "Biochemically IDENTICAL to PEX2-ZSD. PEX1 has common hypomorphic allele p.Gly843Asp-G843D (~30% of all PEX1 alleles) → more IRD (30–35%) vs PEX2-ZSD (only 20% IRD). PEX1-ZSD has more IRD, PEX2-ZSD has more ZS/NALD. ONLY NGS gene panel distinguishes.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 has p.Arg860Trp (R860W) hypomorphic allele (~18% of PEX6 alleles) → most attenuated ZSD subtype (IRD 45% PEX6 vs 20% PEX2). PEX6 locus 6p21.1 vs PEX2 8q21.13. NGS panel distinguishes.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy — XLD)",
            "distinction": "VLCFA elevated in BOTH — but ABCD1 has NORMAL plasmalogens (RBC) and NORMAL DHA; ZSD has LOW plasmalogens + LOW DHA. No cortical migration defects (no pachygyria) in ABCD1. Adrenal insufficiency mechanism different (direct VLCFA adrenal toxicity in ABCD1). PHT/CBZ ABSOLUTE CI in ABCD1 (adrenal); only RELATIVE CI in ZSD (DHA/hepatic). Lorenzo's Oil works in ABCD1, not ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC acid severely elevated (sole elevated peroxisomal metabolite); VLCFA NORMAL (key distinction); plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. VGB ABSOLUTE CI (RP 95% — strongest retinopathy-VGB CI among peroxisomal disorders). Phytol-restricted diet Level A GOLD STANDARD. Distinct from ZSD.",
        },
        {
            "condition": "RCDP / PEX7 (Rhizomelic Chondrodysplasia Punctata)",
            "distinction": "Plasmalogens (RBC) severely LOW (like ZSD) BUT VLCFA NORMAL + phytanic elevated + pristanic NORMAL + DHA NORMAL + pipecolic NORMAL. Rhizomelia (short proximal limbs) PATHOGNOMONIC. Stippled epiphyses (88%). Cataracts (72%). No pachygyria (cortical migration intact). Single PTS2 pathway affected only (not all 6 ZSD pathways).",
        },
        {
            "condition": "Krabbe Disease (GALC deficiency)",
            "distinction": "Lysosomal (NOT peroxisomal) leukodystrophy; VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Psychosine accumulation → galactosylceramide deficiency → demyelination. HSCT effective in pre-symptomatic Krabbe (unlike ZSD). NBS (psychosine DBS) differentiates.",
        },
        {
            "condition": "Mitochondrial / POLG1 Epilepsy",
            "distinction": "VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Elevated lactate/pyruvate. Mitochondrial variants on NGS. POLG1 LOF: ALPERS disease (hepatocerebral syndrome); VPA CATASTROPHIC CI (same as in ZSD for different reason). Muscle biopsy: ragged-red fibres + COX-negative fibres. Mitochondrial DNA depletion.",
        },
    ]

    standards = [
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum: An overview of current diagnosis, clinical manifestations and treatment guidelines. Mol Genet Metab. 2016",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Klouwer FCC et al. Zellweger spectrum disorders: clinical overview and management approach. Orphanet J Rare Dis. 2015",
        "Fujiki Y et al. PEX genes and peroxisome biogenesis disorders. Biochim Biophys Acta Mol Cell Res. 2020",
        "Platta HW et al. The peroxisomal receptor docking complex: assembly and function in peroxisomal matrix protein import. FEBS J. 2014",
        "Waterham HR et al. Human disorders in peroxisome development and function. Biochim Biophys Acta. 2016",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "CPIC VPA–POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614862 (PBD5A, Zellweger — PEX2); #614863 (PBD5B, NALD — PEX2); *170993 (PEX2 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
