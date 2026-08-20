#!/usr/bin/env python3
"""PEX26 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX26 encodes Peroxin-26
(formerly KIAA0713), a 305-amino-acid tail-anchored peroxisomal membrane protein. The
cytoplasmic N-terminal domain of PEX26 acts as the MEMBRANE DOCKING PLATFORM that recruits
the PEX1-PEX6 AAA-ATPase heterodimer to the peroxisomal membrane.

MECHANISM — MEMBRANE ANCHOR / DOCKING PLATFORM FOR PEX1-PEX6 AAA-ATPASE:
  PEX26 is the MEMBRANE ANCHOR that positions the PEX1-PEX6 AAA-ATPase motor on the
  peroxisomal membrane, enabling PEX5 retrotranslocation:
  (1) PEX26 C-terminal single transmembrane domain (TMD): anchors PEX26 as a tail-anchored
      protein in the peroxisomal membrane (post-translational insertion via GET pathway)
  (2) PEX26 N-terminal cytoplasmic domain (~250 aa): interacts directly with PEX6 C-terminal
      D2 AAA domain — recruits the PEX1-PEX6 heterodimer to the peroxisomal membrane surface
  (3) PEX26 LOF → PEX1-PEX6 heterodimer remains soluble in the cytoplasm, cannot localise
      to the peroxisome → PEX5 retrotranslocation fails → PEX5 permanently trapped in
      peroxisomal membrane → ALL peroxisomal matrix protein import ceases.
  CONCEPTUAL POSITION IN THE RETROTRANSLOCATION MACHINERY:
    PEX26 (membrane anchor/dock) → recruits PEX1-PEX6 (AAA-ATPase motor) → PEX2-PEX10-PEX12
    RING complex ubiquitinates PEX5 → PEX1-PEX6 pulls ubiquitinated PEX5 out of membrane →
    PEX5 released into cytoplasm → deubiquitinated → recycled → another import cycle.
  PEX26 LOF is PHENOTYPICALLY AND BIOCHEMICALLY IDENTICAL to PEX1 LOF and PEX6 LOF: in all
  three cases the PEX5 retrotranslocation step fails; the biochemical consequence is the same.
  PEX26 is UPSTREAM of the AAA-ATPase motor: PEX26 brings the motor to the membrane, then the
  motor acts. Without the motor's membrane docking, all downstream steps (including RING
  complex ubiquitination events that depend on PEX5 cycling) cannot occur productively.

LOCUS: 22q11.21  |  OMIM GENE: *608666  |  OMIM DISEASE: #614864 (PBD7A-ZSD), #614865 (PBD7B-NALD)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX26 mutations account for ~2–4% of all
  PBD-ZSD. ~15–25 confirmed PEX26 cases worldwide as of 2026 (among the rarest PBD-ZSD genes,
  comparable to PEX2 at 3–5% and fewer than PEX10 at 5–7%).
  p.Arg98Trp (R98W): most studied missense — cytoplasmic domain; reduces PEX1-PEX6 recruitment
  efficiency (~30–40% residual docking activity); enables IRD phenotype in compound heterozygotes
  with a null allele. No common FOUNDER allele with high prevalence (unlike PEX1-G843D ~30%
  European, PEX6-R860W ~18% European). Other known variants: p.Arg275* (nonsense, null);
  p.Leu188Pro (TMD-adjacent missense, null-equivalent); splice-site variants; frameshift.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (SAME AS PEX1 / PEX6 / PEX2 / PEX10 / PEX12):
  PEX26 deficiency causes the SAME biochemical phenotype as PEX1/PEX6/PEX2/PEX10/PEX12
  because PEX26 membrane docking is required for PEX5 retrotranslocation. Loss of PEX26
  abolishes PEX5 cycling and all peroxisomal matrix protein import:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
  (3) Plasmalogens (ether-phospholipids) BIOSYNTHESIS FAILED → erythrocyte plasmalogens LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated in plasma/urine
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid
  DISTINGUISHING PEX26 FROM PEX1/PEX6: biochemically identical to both — only NGS distinguishes.

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — COMPARABLE TO PEX2/PEX12 — NO FOUNDER HYPOMORPHIC):
  PEX26 cohorts show a severity distribution comparable to PEX2 and PEX12 ZSD. No common
  founder hypomorphic allele → spectrum skewed severe. p.Arg98Trp enables some IRD:
  (1) Zellweger Syndrome (ZS, severe): null/null; neonatal day 1–7; craniofacial dysmorphism;
      profound hypotonia; pachygyria + polymicrogyria (MRI PATHOGNOMONIC in ZS);
      neonatal seizures UNIVERSAL (100%); hepatomegaly + neonatal cholestasis; ERG absent;
      SNHL; fatal <6–12 months.
  (2) NALD (intermediate): null/R98W or null/partial splice; onset 1st–12th month;
      seizures 80–90%; infantile spasms (IS) 35–55% + hypsarrhythmia; cholestasis; survival months–years.
  (3) IRD (attenuated): R98W/mild or hypomorphic/hypomorphic; RP + SNHL + peripheral
      neuropathy; mild hepatomegaly; seizures 20–30%; adult survival in some cases.
  (4) Atypical: ~5%; very mild; adult-onset ataxia/RP only.
  ZS: 25% · NALD: 50% · IRD: 20% · Atypical: 5%

PEX26 UNIQUE MOLECULAR ROLE — UPSTREAM ANCHOR vs AAA-ATPASE MOTOR vs RING E3 COMPLEX:
  PEX26, PEX1, and PEX6 form a hierarchical retrotranslocation sub-machinery:
    PEX26 (anchor) → PEX1-PEX6 motor (retrotranslocation) → needs PEX2-PEX10-PEX12 RING E3 (ubiquitination)
  PEX26 LOF phenocopies PEX1 LOF and PEX6 LOF because all three lead to PEX5 being trapped in
  the peroxisomal membrane. However, the molecular position differs:
  - PEX26: membrane positioning of the motor (upstream anchor)
  - PEX1/PEX6: mechanical retrotranslocation of ubiquitinated PEX5 (the motor itself)
  - PEX2/PEX10/PEX12: ubiquitination of PEX5 (signals motor to act)
  Together these constitute the complete PEX5 retrotranslocation cycle. PEX26 deficiency thus
  represents a failure of the POSITIONING STEP — mechanistically upstream of PEX1/PEX6 LOF
  but phenotypically indistinguishable.

EPILEPSY IN PEX26-ZSD:
  ZS: 100% epilepsy; neonatal seizures day 1–7; multifocal clonic + tonic + myoclonic;
  hypsarrhythmia if surviving infantile period; EEG burst-suppression; drug-resistance universal.
  NALD: 80–90% epilepsy; infantile spasms (IS) 35–55% + hypsarrhythmia; focal + myoclonic.
  IRD: 20–30% epilepsy; focal + GTCS; better AED response.
  DRE: ~60–65% of all PEX26-ZSD with epilepsy (comparable to PEX10-ZSD 60–65% due to similar
  distribution of IRD 20%).

AED PHARMACOLOGY (IDENTICAL TO PEX1/PEX6/PEX2/PEX10/PEX12-ZSD):
  LEV: FIRST-LINE (all ZSD forms). No CYP induction; no hepatotoxicity; IV available.
  ACTH: Level A for IS. Preferred over VGB due to ZSD retinopathy.
  CLB: Level B adjunct. No hepatotoxicity; no CYP induction.
  LTG: Level C (IRD/NALD). Glucuronidation — safer hepatic profile.
  DHA: Level B (NALD/IRD). Bypasses peroxisomal synthesis via LPC-DHA intestinal absorption.
  Phytol-restricted diet: Level B (IRD/Atypical). Reduces phytanic accumulation.
  VPA: HIGH RISK — THREE mechanisms: hepatotoxicity + peroxisomal BO inhibition + carnitine.
    POLG1 MANDATORY (CPIC Grade A). Near-absolute CI in ZS/NALD.
  VGB: HIGH RISK — additive retinopathy. ACTH preferred for IS.
  PHT/CBZ/OXC: RELATIVE CI — CYP3A4 induction → DHA depletion + hepatic burden.
  Lorenzo's Oil: INEFFECTIVE — peroxisomal import absent.
  HSCT: NOT INDICATED — ZSD is NOT inflammatory.
  ERT: NOT AVAILABLE — PEX26 is an integral membrane protein.
"""

import random


# ── Overview ───────────────────────────────────────────────────────────────────
def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 82,
        "zs_pct": 25,
        "nald_pct": 50,
        "ird_pct": 20,
        "atypical_pct": 5,
        "drug_resistance_pct": 62,
        "on_dha_pct": 57,
        "liver_disease_pct": 73,
        "retinopathy_pct": 78,
        "dha_low_pct": 90,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "omim_gene": "608666",
        "omim_disease_zs": "614864",
        "omim_disease_nald": "614865",
        "locus": "22q11.21",
        "protein_size": "305 aa",
        "nbs_positive_rate": "~98% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": (
            "p.Arg98Trp (R98W, cytoplasmic domain missense, partial PEX1-PEX6 recruitment, IRD-enabling); "
            "p.Arg275* (null, truncating); p.Leu188Pro (TMD-adjacent, null-equivalent)"
        ),
        "disease_mechanism": (
            "PEX26 (305 aa, 22q11.21) is the MEMBRANE ANCHOR / DOCKING PLATFORM for the PEX1-PEX6 "
            "AAA-ATPase heterodimer. Its single C-terminal transmembrane domain (tail-anchored) inserts "
            "post-translationally into the peroxisomal membrane; its N-terminal cytoplasmic domain "
            "directly recruits PEX6 (D2 AAA domain) — docking the PEX1-PEX6 motor onto the peroxisome. "
            "PEX26 LOF → PEX1-PEX6 remains cytoplasmic, cannot retrotranslocate PEX5 → PEX5 trapped "
            "in peroxisomal membrane → ALL peroxisomal matrix import fails. "
            "Biochemically identical to PEX1/PEX6/PEX2/PEX10/PEX12-ZSD; only NGS distinguishes. "
            "~2–4% of all PBD-ZSD; ~15–25 cases worldwide 2026."
        ),
        "key_concepts": [
            "PEX26 membrane anchor role: recruits PEX1-PEX6 AAA-ATPase to peroxisomal membrane — the DOCKING PLATFORM upstream of the retrotranslocation motor",
            "Mechanistic hierarchy: PEX26 (anchor) → PEX1-PEX6 (motor) → PEX2-PEX10-PEX12 RING E3 (ubiquitination) — PEX26 LOF is upstream of PEX1/PEX6 LOF but phenotypically identical",
            "p.Arg98Trp (R98W): cytoplasmic domain missense; reduces PEX1-PEX6 recruitment efficiency (~30-40% residual docking); enables IRD phenotype as compound het with null allele",
            "No founder hypomorphic allele (unlike PEX1-G843D ~30% European, PEX6-R860W ~18% European) → most PEX26 patients null/partial → spectrum skewed severe",
            "Biochemically identical to PEX1/PEX6/PEX2/PEX10/PEX12-ZSD: VLCFA + plasmalogens LOW + DHA low + phytanic/pristanic + pipecolic + DHCA/THCA ALL abnormal",
            "PEX26 tail-anchored topology: C-terminal TMD in peroxisomal membrane (GET pathway insertion); N-terminal cytoplasmic domain recruits PEX6 — no lumenal domain, unlike type I/II transmembrane PMPs",
            "PEX26 vs PEX1/PEX6: all three LOF → PEX5 retrotranslocation failure. PEX26 = docking position (upstream); PEX1/PEX6 = mechanical pull (motor). Phenotype and biochemistry IDENTICAL",
            "VPA HIGH RISK (three mechanisms: hepatotoxicity + peroxisomal BO inhibition + carnitine); POLG1 MANDATORY (CPIC Grade A)",
            "VGB HIGH RISK (additive retinopathy: ZSD retinopathy universal ZS/NALD + VGB irreversible VF constriction); ACTH Level A for IS",
            "LEV first-line all ZSD forms; DHA Level B (NALD/IRD); Phytol-restricted diet Level B (IRD)",
            "Fasting EXTREME HAZARD: adipose phytanic acid mobilisation → acute neurotoxicity + seizures; IV dextrose MANDATORY during NPO",
            "PHT/CBZ/OXC RELATIVE CI (CYP3A4 → DHA depletion + hepatic) — NOT adrenal CI (unlike ABCD1 ABSOLUTE CI)",
            "Lorenzo's Oil INEFFECTIVE (peroxisomal import absent — mechanism mismatch with X-ALD)",
            "HSCT NOT INDICATED (ZSD not inflammatory; donor microglia cannot restore PEX26 membrane docking)",
            "No ERT (PEX26 = tail-anchored membrane protein; protein replacement not applicable)",
            "~15–25 cases worldwide 2026 (rarest of the ZSD genes in this series); completes the PEX26-PEX1-PEX6 retrotranslocation docking machinery",
        ],
        "standards": [
            "Fujiki Y et al. New Insights into Peroxisomal Protein Targeting Pathways. Annu Rev Cell Dev Biol. 2020",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
            "Tamura S et al. Peroxisome biogenesis factors — their roles in peroxisomal membrane and matrix protein import. FEBS J. 2014",
            "Matsumoto N et al. The peroxin Pex26p functions as a molecular tether of the AAA-type ATPase Pex1p-Pex6p complex. Biochem Biophys Res Commun. 2003",
            "Furuki S et al. Mutations in the peroxin gene PEX26 that cause peroxisome biogenesis disorder of complementation group 8 provide a genotype-phenotype correlation. Am J Hum Genet. 2006",
            "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614864 (PBD7A-ZSD — PEX26); #614865 (PBD7B-NALD — PEX26); *608666 (PEX26 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(26305)  # PEX26 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 25, "n": 10,
            "sex": "5M/5F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + electrographic; EEG burst-suppression",
            "eeg": "Burst-suppression → multifocal spike-wave; high amplitude; EEG monitoring NICU mandatory; poor prognosis",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination; enlarged ventricles",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Arg275*/p.Arg275* or p.Arg275*/frameshift/splice); PEX26 TMD absent or severely truncated; PEX1-PEX6 cannot dock; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Arg98Trp or splice",
            "pct": 50, "n": 20,
            "sex": "10M/10F",
            "onset_age": "Infantile (1st–12th month)",
            "seizure_risk": "80–90% — infantile spasms (IS) 35–55% + hypsarrhythmia; focal + generalized + myoclonic",
            "eeg": "Hypsarrhythmia (IS period); multifocal spike-wave; modified hypsarrhythmia with burst-suppression elements",
            "mri": "Periventricular white matter signal; delayed myelination; cerebellar hypoplasia; milder gyral pattern than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Arg98Trp (partial docking) or null/splice; partial PEX1-PEX6 recruitment from R98W allele; survival months to years",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Arg98Trp/mild",
            "pct": 20, "n": 8,
            "sex": "4M/4F",
            "onset_age": "Childhood / Adolescence (1–15 yr)",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD; phytol-restricted diet reduces burden",
            "eeg": "Focal epileptiform discharges; normal background early; generalized spike-wave with progression",
            "mri": "RP-associated retinal thinning; mild cerebellar atrophy; white matter changes mild; no pachygyria",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Arg98Trp/hypomorphic compound het; ~30–40% residual PEX1-PEX6 docking activity; partial peroxisomal matrix import; adult survival",
        },
        {
            "name": "Atypical / Late-Onset ZSD — very mild alleles",
            "pct": 5, "n": 2,
            "sex": "1M/1F",
            "onset_age": "Adult (20–50 yr)",
            "seizure_risk": "10–15% — rare GTCS or focal; responsive to LEV",
            "eeg": "Normal background; rare focal discharges; incidental finding on investigation for ataxia/RP",
            "mri": "Mild cerebellar atrophy; RP-associated retinal thinning; white matter normal or minimally abnormal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles hypomorphic (very mild missense); substantial residual PEX26 docking; late-onset slowly progressive",
        },
    ]

    def rand_sex():
        return random.choice(["M", "F"])

    def rand_geno(phenotype):
        if "ZS" in phenotype:
            variants = ["p.Arg275*/p.Arg275*", "p.Arg275*/p.Leu188Pro", "p.Arg275*/c.691+1G>A", "p.Arg275*/p.Trp200*"]
        elif "NALD" in phenotype:
            variants = ["p.Arg275*/p.Arg98Trp", "p.Arg275*/c.562-2A>G", "p.Trp200*/p.Arg98Trp", "p.Leu188Pro/p.Arg98Trp"]
        elif "IRD" in phenotype:
            variants = ["p.Arg98Trp/p.Val231Ala", "p.Arg98Trp/c.562-2A>G", "p.Arg98Trp/p.Ile255Thr", "p.Arg98Trp/p.Ala278Val"]
        else:
            variants = ["p.Arg98Trp/p.Gly124Ser", "p.Arg98Trp/p.Ile90Val"]
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
        ("ZS (Zellweger-Severe)", 10),
        ("NALD (Neonatal-ALD-Intermediate)", 20),
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
                "patient_id": f"PEX26-{pid:03d}",
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
        {"type": "Neonatal multifocal clonic (ZS)", "pct": 25, "eeg": "Burst-suppression → multifocal spike; bilateral asynchronous; EEG NICU mandatory; universal in ZS"},
        {"type": "Infantile spasms / West syndrome (NALD)", "pct": 50, "eeg": "Hypsarrhythmia → modified; ACTH Level A first-line; VGB HIGH RISK (avoid due to retinopathy)"},
        {"type": "Focal seizures bilateral spread (NALD/IRD)", "pct": 22, "eeg": "Temporal/parietal focal spike-wave; post-ictal slowing; lateralised in IRD; LEV first-line"},
        {"type": "Myoclonic seizures (NALD)", "pct": 15, "eeg": "Generalised polyspike-wave ≥3 Hz; photosensitive in some; CBZ/PHT HIGH RISK (myoclonic exacerbation)"},
        {"type": "GTCS (IRD/Atypical)", "pct": 10, "eeg": "Generalised spike-wave; good post-ictal recovery in IRD; LEV + LTG effective"},
        {"type": "Absence / atypical absence (rare, IRD)", "pct": 4, "eeg": "2.5–3.5 Hz generalised spike-wave; ETX considered Level C in IRD with absence predominance"},
    ]

    triggers = [
        {"trigger": "Fever / intercurrent infection", "pct": 65, "note": "Universal ZSD trigger — fever lowers seizure threshold + increases metabolic demand → phytanic mobilisation"},
        {"trigger": "Subtherapeutic AED levels", "pct": 52, "note": "Common in NALD/IRD — growth spurts alter pharmacokinetics; TDM quarterly"},
        {"trigger": "Fasting / prolonged NPO", "pct": 48, "note": "EXTREME HAZARD — adipose phytanic mobilisation → acute neurotoxicity; IV dextrose MANDATORY"},
        {"trigger": "Sleep deprivation", "pct": 40, "note": "Worsens electrocortical excitability; melatonin 0.5–3 mg adjunct in NALD/IRD with sleep disruption"},
        {"trigger": "Metabolic decompensation / catabolism", "pct": 35, "note": "Intercurrent illness, surgery, missed feeds → phytanic surge + VLCFA accumulation worsens acutely"},
        {"trigger": "Missed AED dose", "pct": 30, "note": "Non-adherence frequent in long-term NALD/IRD; simplified LEV BID regimen improves adherence"},
        {"trigger": "Enzyme-inducing AEDs (PHT/CBZ/OXC)", "pct": 22, "note": "CYP3A4 induction → DHA depletion (already low) + hepatic burden; RELATIVE CI; replace with LEV"},
        {"trigger": "Anaesthesia / surgical stress", "pct": 18, "note": "NPO + surgical stress → phytanic surge; pre-op: PT/INR + LFT + platelet + IV VitK; IV dextrose mandatory"},
    ]

    monitoring = [
        "VLCFA plasma panel (C26:0, C26:0/C22:0 ratio) — baseline + q6M (disease tracking + AED-induced metabolic change)",
        "Erythrocyte plasmalogens — baseline + q6M (PEX26 membrane docking function marker; LOW in all ZSD)",
        "DHA plasma level — baseline + q6M (supplement adequacy; target >4% total fatty acids in NALD/IRD)",
        "Plasma phytanic acid — q6M in IRD/Atypical; q3M in NALD (diet compliance + fasting risk)",
        "LFT + GGT + bilirubin (conjugated) — q3M in ZS/NALD; q6M in IRD (cholestatic liver disease monitoring)",
        "PT/INR + platelet count — q6M in ZS/NALD (cholestatic coagulopathy; pre-op MANDATORY)",
        "Plasma carnitine + acylcarnitine profile — q6M (carnitine depletion risk, especially if VPA inadvertently started)",
        "POLG1 sequencing — ONCE before any VPA consideration (CPIC Grade A; MANDATORY)",
        "Electroretinogram (ERG) — baseline + annual (universal retinopathy ZS/NALD; preserve remaining IRD visual field)",
        "Visual field testing (Goldmann perimetry) — q12M in IRD (preserve from VGB, RP progression)",
        "SNHL audiometry (ABR neonatal → PTA in IRD) — baseline + q12M (universal SNHL in ZS/NALD)",
        "EEG — NICU continuous in ZS; q3M in active NALD (IS monitoring); q6M in seizure-free NALD; q12M in IRD",
        "Brain MRI — baseline (pachygyria ZS; white matter NALD); q12–24M in NALD/IRD",
        "Developmental / neuropsychological assessment — q12M in NALD/IRD (cognition, language, adaptive function)",
    ]

    thresholds = [
        {"parameter": "C26:0-lyso-PC (NBS DBS)", "threshold": ">0.10 μmol/L", "action": "Refer to metabolic specialist STAT; confirm with plasma VLCFA + plasmalogens; NGS PEX gene panel (include PEX26)"},
        {"parameter": "Plasma C26:0/C22:0 ratio", "threshold": ">0.020", "action": "Diagnostic for ZSD biochemically; initiate NGS PEX panel; PEX26 among first-tier genes after PEX1/PEX6"},
        {"parameter": "Erythrocyte plasmalogens", "threshold": "<50% age-matched reference", "action": "Confirms PBD-ZSD biochemical phenotype (vs ABCD1 where plasmalogens NORMAL)"},
        {"parameter": "Plasma DHA", "threshold": "<4% total FA in NALD/IRD", "action": "Initiate DHA supplementation (LPC-DHA 100–200 mg/day NALD; Level B); recheck in 3M"},
        {"parameter": "Plasma phytanic acid", "threshold": ">5 μmol/L (IRD)", "action": "Intensify phytol-restricted diet; exclude fasting; consider plasmapheresis if acutely elevated"},
        {"parameter": "Plasma carnitine (free)", "threshold": "<25 μmol/L", "action": "Carnitine supplementation; VPA CONTRAINDICATED; review all medications for carnitine-depleting agents"},
        {"parameter": "PT/INR (cholestatic)", "threshold": "INR > 1.5", "action": "IV Vitamin K MANDATORY pre-surgery; hepatology consult; avoid hepatotoxic AEDs (VPA)"},
        {"parameter": "VPA drug level (if mistakenly started)", "threshold": "Any detectable level", "action": "STOP VPA IMMEDIATELY; check POLG1; supplement carnitine; monitor LFT + NH3 closely"},
    ]

    lifecycle = [
        {
            "stage": "Stage 1 — Neonatal/NICU (0–28 days, ZS only)",
            "features": "Profound hypotonia; neonatal seizures day 1–7 (multifocal clonic + tonic); EEG burst-suppression; cholestatic jaundice; craniofacial dysmorphism; ERG absent; SNHL on ABR; MRI shows pachygyria",
            "action": "LEV IV first-line; Phenobarbital if refractory; IV dextrose continuous; VLCFA + plasmalogens STAT; NGS PEX panel STAT (include PEX26); palliative care discussion; POLG1 mandatory",
        },
        {
            "stage": "Stage 2 — Early Infantile (1–12 months, NALD dominant)",
            "features": "Infantile spasms (IS) with hypsarrhythmia (35–55%); psychomotor regression; cholestasis evolving; SNHL; visual impairment from RP; nasogastric feeds",
            "action": "ACTH Level A (preferred over VGB for IS — ZSD retinopathy risk); LEV adjunct; DHA Level B; ophthalmology + audiology baseline; confirm PEX26 biallelic variants by NGS",
        },
        {
            "stage": "Stage 3 — Late Infantile/Toddler (1–3 yr, NALD/IRD)",
            "features": "Seizure evolution from IS to focal + myoclonic; developmental plateau; RP progression; SNHL; ataxia emerging in IRD; phytol-restricted diet starts in IRD",
            "action": "Optimise LEV ± CLB adjunct; phytol-restricted diet Level B (IRD); DHA continue; VPA NEVER; ERG + visual field + ABR annual; NGS confirms PEX26 identity vs PEX1/PEX6",
        },
        {
            "stage": "Stage 4 — Preschool/School Age (3–12 yr, IRD/Atypical)",
            "features": "Focal ± GTCS; intellectual disability mild–moderate; ataxia prominent; RP → tunnel vision; SNHL stable; hepatomegaly mild; peripheral neuropathy emerging",
            "action": "LEV ± LTG Level C; strict phytol-restricted diet; annual VLCFA + phytanic + DHA; neuropsychological support; adaptive aids; transition planning",
        },
        {
            "stage": "Stage 5 — Adolescence/Adulthood (12+ yr, IRD/Atypical only)",
            "features": "Seizure frequency stable in well-controlled IRD; RP → legal blindness; SNHL; progressive ataxia; peripheral neuropathy; psychiatric comorbidity",
            "action": "Long-term LEV ± LTG; phytol-restricted diet; annual metabolic review; genetic counselling (AR — 25% recurrence); no ERT/HSCT; gene therapy preclinical",
        },
        {
            "stage": "Stage 6 — Surgical / Anaesthesia Intercept (ANY stage)",
            "features": "Peri-operative fasting EXTREME HAZARD: phytanic surge → neonatal-equivalent neurotoxicity; cholestatic coagulopathy risk (ZS/NALD)",
            "action": "IV dextrose MANDATORY from first NPO moment; IV Vitamin K pre-op; PT/INR + LFT + platelet pre-op; VPA ABSOLUTE CI perioperatively; IV LEV replaces oral; anaesthetics team briefing MANDATORY",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "AED — FIRST-LINE (all ZSD forms)",
            "evidence": "Level A (expert consensus — multiple ZSD cohorts including PEX26)",
            "dose": "Neonatal: 20–30 mg/kg/day IV BID; Infantile/child: 30–60 mg/kg/day PO BID; Adult: 1000–3000 mg/day BID",
            "moa": "SV2A synaptic vesicle modulation; no hepatotoxicity; no CYP induction; no carnitine depletion; IV available",
            "monitoring": "Renal function (dose-adjust in CKD); ADHD-like side effects in NALD/IRD",
            "ci": None,
        },
        {
            "drug": "ACTH (Adrenocorticotropic Hormone)",
            "class": "AED — Level A for Infantile Spasms (IS)",
            "evidence": "Level A (UKISS; multinational ZSD protocols; PREFERRED over VGB in ZSD — retinopathy risk)",
            "dose": "20–40 IU/day IM 2–4 weeks; taper over 4–6 weeks; monitor BP + glucose + electrolytes",
            "moa": "Melanocortin receptor pathway; down-regulates CRF; preferred over VGB in ZSD to avoid additive retinopathy",
            "monitoring": "BP, glucose, electrolytes, infection risk; adrenal recovery post-taper",
            "ci": "Active infection; severe hypertension; do NOT substitute VGB in ZSD",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "AED — Level B adjunct",
            "evidence": "Level B (expert consensus; low hepatotoxicity; effective in focal/myoclonic)",
            "dose": "Child: 0.1–0.3 mg/kg/day PO BID; Adult: 10–30 mg/day BID",
            "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine); no CYP induction; no carnitine effect",
            "monitoring": "Sedation; tolerance; respiratory in neonates",
            "ci": "Severe hepatic failure; avoid in neonatal ZS respiratory compromise",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "AED — Level C (NALD/IRD adjunct)",
            "evidence": "Level C (case reports; glucuronidation — safer hepatic vs PHT/CBZ)",
            "dose": "Start 0.15 mg/kg/day; titrate slowly over 8 weeks to 3–5 mg/kg/day; adult 100–300 mg/day",
            "moa": "Sodium channel blocker; glucuronidation (UGT1A4) — avoids CYP hepatic burden; useful for focal + GTCS in IRD",
            "monitoring": "Stevens-Johnson syndrome (slow titration mandatory); no VPA co-administration",
            "ci": "VPA co-administration (prohibited in ZSD); slow titration mandatory",
        },
        {
            "drug": "DHA (Docosahexaenoic Acid) supplementation",
            "class": "Disease-Modifying — Level B (NALD/IRD)",
            "evidence": "Level B (Martinez 1996; van Grunsven 1999; LPC-DHA bypasses peroxisomal synthesis block)",
            "dose": "100–200 mg/day as LPC-DHA; oral; intestinal absorption bypasses peroxisomal DHA synthesis failure",
            "moa": "LPC-DHA absorbed intestinally → incorporated into neural membranes; bypasses peroxisomal synthesis; improves retinal + myelin in NALD/IRD",
            "monitoring": "Plasma DHA q3M; target >4% total FA; ERG q6M",
            "ci": "Not indicated in ZS (fatal prognosis); use LPC-DHA formulation specifically",
        },
        {
            "drug": "Phytol-restricted diet (low phytanic acid)",
            "class": "Disease-Modifying — Level B (IRD/Atypical)",
            "evidence": "Level B (Steinberg 2006; clinical consensus; reduces phytanic accumulation in IRD)",
            "dose": "Restrict phytol-rich foods (green vegetables, dairy, ruminant fat); dietitian supervised; maintain caloric intake",
            "moa": "Phytol → phytanic acid via dietary intake; restriction reduces accumulation → fewer phytanic-triggered seizures + neuropathy",
            "monitoring": "Plasma phytanic q6M; nutritional status; dietitian review q6M",
            "ci": "No absolute CI; ensure caloric adequacy (fasting EXTREME HAZARD — IV dextrose mandatory during NPO/illness)",
        },
        {
            "drug": "Cholic acid / UDCA",
            "class": "Supportive — Level C (cholestatic liver disease)",
            "evidence": "Level C (expert consensus; reduces DHCA/THCA-driven cholestasis)",
            "dose": "UDCA: 10–15 mg/kg/day PO BID; Cholic acid where available as primary bile acid replacement",
            "moa": "UDCA hepatoprotective; CA replaces deficient primary bile acids; reduces upstream DHCA/THCA accumulation",
            "monitoring": "LFT q3M; PT/INR q6M; bilirubin q3M; hepatology input",
            "ci": "None specific; hepatology recommended",
        },
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA / Valproic Acid / Depakene / Epilim)",
            "level": "HIGH RISK (near-Absolute CI in ZS/NALD) — THREE independent mechanisms",
            "reason": "(1) Hepatotoxicity: ZS/NALD have cholestatic liver disease (DHCA/THCA-driven); VPA adds direct hepatotoxicity; (2) Peroxisomal beta-oxidation inhibition: VPA is a peroxisomal BO substrate → inhibits VLCFA oxidation → worsens C26:0 accumulation; (3) Carnitine depletion: VPA sequesters carnitine → secondary deficiency (amplified in ZSD). POLG1 MANDATORY (CPIC Grade A).",
            "alternative": "LEV IV/PO (first-line). ACTH Level A (IS). LEV + CLB or LEV + LTG (IRD focal). NEVER VPA in PEX26-ZSD.",
        },
        {
            "drug": "Vigabatrin (VGB / Sabril)",
            "level": "HIGH RISK (avoid in IRD; near-CI in ZS/NALD)",
            "reason": "ZSD causes retinopathy (universal ZS/NALD; 20–30% IRD as RP). VGB irreversibly constricts visual fields (concentric VF loss, 30–50% users). In ZSD, VGB-induced VF loss is ADDITIVE to existing RP → catastrophic irreversible blindness. ACTH (Level A) is always preferred for IS in all ZSD forms. Avoid VGB even in IRD.",
            "alternative": "ACTH (Level A) for IS. LEV + CLB for focal/myoclonic. Never VGB first-line in ZSD.",
        },
        {
            "drug": "Fasting / Prolonged NPO / Caloric Restriction",
            "level": "EXTREME HAZARD (ALL ZSD forms, ALL ages)",
            "reason": "Fasting triggers adipose release of stored phytanic acid (accumulated due to failed alpha-oxidation). Acute phytanic surge → direct neurotoxicity + acute seizures + cardiac arrhythmia. Can precipitate acute decompensation even in stable IRD. ANY NPO period = MANDATORY IV dextrose (10% at maintenance).",
            "alternative": "IV 10% dextrose MANDATORY from first NPO moment. Pre-op briefing. IV Vitamin K (coagulopathy). PT/INR + LFT + platelet pre-surgery. Resume enteral feeds ASAP.",
        },
        {
            "drug": "Phenytoin / Carbamazepine / Oxcarbazepine (PHT / CBZ / OXC)",
            "level": "RELATIVE CI (avoid first-line; caution if used)",
            "reason": "CYP3A4 induction → (1) DHA catabolism accelerated (already deficient in ZSD → worsens RP/neuropathy) + (2) hepatic microsomal burden (cholestatic liver ZS/NALD). NOT adrenal crisis (unlike ABCD1 ABSOLUTE CI — no adrenal insufficiency in PEX26-ZSD as a universal feature). May be used in specific focal seizures (IRD) if no alternative with close DHA + LFT monitoring.",
            "alternative": "LEV first-line; LTG glucuronidation Level C (IRD); CLB adjunct. Replace with LEV on diagnosis.",
        },
        {
            "drug": "Lorenzo's Oil (oleic acid + erucic acid)",
            "level": "INEFFECTIVE (do NOT use in ZSD)",
            "reason": "Lorenzo's Oil inhibits VLCFA elongation → works in X-ALD (ABCD1) because peroxisomal import is intact but ABCD1 transporter absent. In ZSD, peroxisomal IMPORT is absent (PEX26 membrane anchor failure → PEX1-PEX6 cannot dock → PEX5 cannot cycle → matrix import fails). LOO cannot restore PEX26 docking or PEX5 cycling. No benefit; possible side effects (thrombocytopenia).",
            "alternative": "DHA Level B (NALD/IRD). Phytol-restricted diet Level B (IRD). No VLCFA-specific therapy available 2026.",
        },
        {
            "drug": "HSCT (Hematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED (PEX26-ZSD)",
            "reason": "HSCT benefits conditions with a neuroinflammatory component (cerebral X-ALD / ABCD1 — donor microglia halt demyelination). PEX26-ZSD is NOT primarily inflammatory; pathology is metabolic (peroxisomal import failure from absent membrane anchor). Donor microglia cannot restore PEX26 docking or PEX1-PEX6 localisation. HSCT carries 5–15% transplant-related mortality — unacceptable risk-benefit.",
            "alternative": "Supportive care: DHA, phytol-restricted diet, LEV. Gene therapy (AAV9-PEX26) in preclinical research.",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE (PEX26 = tail-anchored membrane protein)",
            "reason": "ERT is feasible for soluble lysosomal enzymes that can be endocytosed via mannose-6-phosphate receptor. PEX26 is a tail-anchored peroxisomal MEMBRANE protein (C-terminal TMD inserted via GET pathway into peroxisomal membrane, cytoplasmic N-terminal domain) — not a soluble secreted enzyme. ERT cannot deliver a membrane-anchored docking protein to its intracellular target. No applicable ERT mechanism.",
            "alternative": "Gene therapy (AAV-PEX26) in preclinical research; mRNA delivery experimental. No clinically available ERT 2026.",
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
        "PEX26 (305 aa, 22q11.21): tail-anchored peroxisomal membrane protein (C-terminal TMD, GET-pathway insertion); cytoplasmic N-terminal domain is the MEMBRANE DOCKING PLATFORM for PEX1-PEX6 AAA-ATPase",
        "PEX26 membrane anchor mechanism: PEX26 N-terminal cytoplasmic domain binds PEX6 D2 AAA domain → recruits PEX1-PEX6 heterodimer to peroxisomal membrane surface → enables PEX5 retrotranslocation",
        "Mechanistic hierarchy: PEX26 (anchor) → PEX1-PEX6 (motor) → PEX2-PEX10-PEX12 RING E3 (ubiquitination) → PEX5 cycling. PEX26 is UPSTREAM of PEX1/PEX6; PEX26 LOF phenocopies PEX1/PEX6 LOF",
        "PEX5 retrotranslocation failure: PEX26 LOF → PEX1-PEX6 stays cytoplasmic → cannot retrotranslocate PEX5 → PEX5 trapped in peroxisomal membrane → ALL PTS1-targeted matrix protein import fails",
        "Biochemical identity: PEX26-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX2/PEX10/PEX12-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
        "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX26; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
        "DHA synthesis failure: DHA synthesised peroxisomally; PEX26 failure → DHA synthesis fails → DHA low → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + RP + neuropathy",
        "Phenotypic spectrum: ZS 25% (null/null) · NALD 50% (null/R98W) · IRD 20% (R98W/mild) · Atypical 5%; comparable to PEX2/PEX12-ZSD (no founder hypomorphic allele, spectrum skewed severe)",
        "p.Arg275* (R275*): most common null truncating variant; associated with ZS and NALD in compound het with partial allele",
        "p.Arg98Trp (R98W): partial-function cytoplasmic domain missense; reduces PEX1-PEX6 docking efficiency by ~60–70%; enables IRD phenotype when compound het with null allele",
        "VPA triple CI mechanism in ZSD: (1) direct hepatotoxicity in cholestatic liver + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A) before any VPA",
        "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD; IRD in 20%) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH preferred for IS in all ZSD forms",
        "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → direct neurotoxicity + seizures + arrhythmia; IV dextrose prevents catabolism; MANDATORY during any NPO",
        "PHT/CBZ/OXC RELATIVE CI (NOT absolute, unlike ABCD1): CYP3A4 induction depletes DHA + increases hepatic burden. No adrenal crisis in PEX26-ZSD (adrenal insufficiency not universal). Different from ABCD1 ABSOLUTE CI.",
        "Lorenzo's Oil mechanism mismatch: LOO inhibits VLCFA elongase → effective in ABCD1 (import intact, transporter absent). In ZSD, peroxisomal IMPORT is absent (PEX26 anchor failure → no PEX5 cycling). LOO cannot restore membrane docking → INEFFECTIVE",
        "NBS: C26:0-lyso-PC on DBS detects ZSD biochemically (~98% sensitivity ZS/NALD); confirms ZSD group but cannot identify PEX26 specifically — comprehensive NGS PEX gene panel required for gene-level diagnosis",
    ]

    diagnostic_algorithm = [
        "Step 1 — NBS flag: C26:0-lyso-PC elevated on DBS → refer to metabolic specialist STAT",
        "Step 2 — Plasma VLCFA panel: C26:0, C26:0/C22:0 ratio elevated → confirms ZSD biochemical group",
        "Step 3 — Erythrocyte plasmalogens: severely LOW → confirms PBD-ZSD (distinguishes from ABCD1 where plasmalogens NORMAL)",
        "Step 4 — Plasma phytanic + pristanic: both elevated (alpha-oxidation failed — confirms PTS1 + alpha-ox pathway failure)",
        "Step 5 — Plasma/urine pipecolic acid: elevated → confirms peroxisomal import failure",
        "Step 6 — Plasma DHA: LOW → confirms DHA synthesis failure; retinopathy + migration defect risk",
        "Step 7 — Bile acids (DHCA, THCA): elevated → confirms bile acid synthesis failure → cholestatic liver disease",
        "Step 8 — MRI brain: pachygyria + polymicrogyria in ZS (PATHOGNOMONIC); periventricular WM signal in NALD; mild/normal in IRD",
        "Step 9 — ERG + ophthalmology: ERG absent/severely reduced (ZS/NALD); RP in IRD; visual fields (Goldmann)",
        "Step 10 — Comprehensive NGS PEX gene panel (PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26): identifies PEX26 biallelic pathogenic variants; confirms gene-level diagnosis",
        "Step 11 — POLG1 sequencing: MANDATORY before considering any VPA (CPIC Grade A); eliminates mitochondrial aetiology",
        "Step 12 — Genetic counselling: AR inheritance (25% recurrence risk); prenatal diagnosis available (PEX26 targeted sequencing in CVS/amnio); carrier testing for parents/siblings",
    ]

    pharmacological_distinctions = [
        "LEV vs PHT/CBZ/OXC in ZSD: LEV = no CYP induction, no hepatotoxicity, IV available → FIRST-LINE; PHT/CBZ/OXC = CYP3A4 → DHA depletion + hepatic burden → RELATIVE CI",
        "VPA vs LEV: VPA = THREE CI mechanisms (hepatotoxicity + peroxisomal BO inhibition + carnitine depletion) → HIGH RISK near-absolute CI; LEV = none → always preferred",
        "ACTH vs VGB for IS: In ZSD, ACTH Level A (preferred) vs VGB HIGH RISK (retinopathy additive). Never use VGB first-line in any ZSD form",
        "DHA supplementation: LPC-DHA (lysophosphatidylcholine-DHA) form bypasses peroxisomal synthesis via intestinal absorption. Ethyl ester less effective (requires peroxisomal processing). Specifically use LPC-DHA",
        "Phytol-restricted diet in IRD vs no benefit in ZS: In ZS (fatal <12M), restriction has no benefit. In IRD/Atypical (long survival), phytol restriction reduces phytanic accumulation → fewer seizures + slower neuropathy",
        "Lorenzo's Oil in ABCD1 vs ZSD: LOO effective in ABCD1 (import intact, elongase inhibition reduces VLCFA). In PEX26-ZSD, import itself absent (PEX26 docking failure) → LOO cannot help → INEFFECTIVE",
        "HSCT in ABCD1 vs ZSD: HSCT standard of care in early CCALD (neuroinflammatory). ZSD is NOT inflammatory → HSCT = transplant mortality risk with zero mechanistic benefit",
        "PHT/CBZ ABSOLUTE CI (ABCD1) vs RELATIVE CI (PEX26-ZSD): In ABCD1, PHT/CBZ → adrenal crisis (universal adrenal insufficiency + CYP induction). In PEX26-ZSD, no universal adrenal insufficiency → PHT/CBZ = RELATIVE CI (DHA/hepatic only, not adrenal)",
        "VPA POLG1 mandatory (CPIC Grade A): POLG1 exclusion mandatory in ZSD because: (1) VPA CI in POLG1 (separate mitochondrial mechanism) + (2) ZSD patients may also carry POLG1 variants → catastrophic combined risk",
        "Carnitine supplementation: indicated if free carnitine <25 μmol/L (may occur in PEX26-ZSD due to impaired peroxisomal fatty acid metabolism); L-carnitine 50–100 mg/kg/day; critical if VPA inadvertently started",
        "IV Vitamin K perioperative: cholestatic liver (DHCA/THCA-driven) → fat-soluble vitamin K malabsorption → coagulopathy (↑PT/INR); IV Vitamin K 1–2 mg/kg (max 10 mg) pre-surgery; check PT/INR + platelet pre-op MANDATORY",
        "PEX26 vs PEX1/PEX6 therapeutic implications: All three LOF → same biochemical phenotype → same AED rules. PEX26 is upstream (docking anchor), not the motor. No different treatment targets; gene therapy approach would differ (AAV-PEX26 vs AAV-PEX1/PEX6)",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX1-ZSD (65% of all PBD — most common)",
            "distinction": "Biochemically IDENTICAL. PEX1 = D1 AAA-ATPase subunit (motor), anchored by PEX26. p.Gly843Asp founder allele (30% European) → IRD 30% (more attenuated than PEX26). PEX26 far rarer (~15–25 cases vs PEX1 hundreds). PEX1 at 7q21.2 vs PEX26 at 22q11.21. NGS distinguishes. Both responsive to same ZSD AED protocol.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 = D2 AAA-ATPase subunit (partner to PEX1; docked by PEX26). p.Arg860Trp (R860W) founder allele → IRD 45% (most attenuated ZSD). PEX26 has less IRD (20%). 6p21.1 vs 22q11.21. NGS distinguishes. PEX6 LOF = motor fails; PEX26 LOF = motor cannot dock — phenotype identical.",
        },
        {
            "condition": "PEX12-ZSD (3–5% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX12 = PRIMARY CATALYTIC RING E3 subunit (polyUb PEX5). PEX12 is DOWNSTREAM of PEX26-PEX1-PEX6 (ubiquitination step). p.Gly271Asp partial function → IRD 20% (similar to PEX26 20%). Both ~same rarity. 17q12 vs 22q11.21. NGS distinguishes.",
        },
        {
            "condition": "PEX2-ZSD (3–5% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX2 = secondary catalytic RING E3 subunit (monoUb PEX5 recycling). No founder allele → skewed severe (same as PEX26). ZS 30% (slightly more ZS than PEX26 25%). 8q21.13 vs 22q11.21. NGS distinguishes.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy)",
            "distinction": "VLCFA elevated in BOTH. ABCD1: plasmalogens (RBC) NORMAL + DHA NORMAL + adrenal insufficiency (universal, X-linked males). PHT/CBZ = ABSOLUTE CI in ABCD1 (adrenal crisis) vs RELATIVE CI in PEX26-ZSD. Lorenzo's Oil EFFECTIVE in ABCD1 (intact import); INEFFECTIVE in ZSD. HSCT standard CCALD; NOT in ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC severely elevated (SOLE elevated peroxisomal metabolite); VLCFA NORMAL (key vs PEX26); plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. VGB ABSOLUTE CI (RP 95%). Phytol-restricted diet Level A GOLD STANDARD.",
        },
        {
            "condition": "RCDP / PEX7 (Rhizomelic Chondrodysplasia Punctata Type 1)",
            "distinction": "Plasmalogens (RBC) LOW (like ZSD) BUT VLCFA NORMAL + phytanic elevated + pristanic NORMAL + DHA NORMAL + pipecolic NORMAL. Only PTS2 pathway affected. Rhizomelia PATHOGNOMONIC. Stippled epiphyses on X-ray. No pachygyria.",
        },
        {
            "condition": "Mitochondrial / POLG1 Epilepsy (Alpers)",
            "distinction": "VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Lactate/pyruvate elevated. Hepatocerebral phenotype. VPA CATASTROPHIC CI in POLG1 (mitochondrial depletion) — same practical CI as ZSD but different mechanism. POLG1 MANDATORY in all ZSD before VPA.",
        },
    ]

    standards = [
        "Fujiki Y et al. New Insights into Peroxisomal Protein Targeting Pathways. Annu Rev Cell Dev Biol. 2020",
        "Matsumoto N et al. The peroxin Pex26p functions as a molecular tether of the AAA-type ATPase Pex1p-Pex6p complex. Biochem Biophys Res Commun. 2003",
        "Furuki S et al. Mutations in PEX26 that cause peroxisome biogenesis disorder of complementation group 8. Am J Hum Genet. 2006",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
        "Tamura S et al. Peroxisome biogenesis factors — their roles in peroxisomal membrane and matrix protein import. FEBS J. 2014",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614864 (PBD7A-ZSD — PEX26); #614865 (PBD7B-NALD — PEX26); *608666 (PEX26 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
