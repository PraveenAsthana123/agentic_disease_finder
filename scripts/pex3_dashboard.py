#!/usr/bin/env python3
"""PEX3 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX3 encodes Peroxin-3,
a 373-amino-acid TYPE I INTEGRAL PEROXISOMAL MEMBRANE PROTEIN. PEX3 is the DOCKING
RECEPTOR on the peroxisomal membrane for PEX19 (the cytosolic PMP chaperone), and is
part of the PEX3-PEX16-PEX19 early membrane biogenesis axis.

MECHANISM — DOCKING RECEPTOR FOR PEX19 / EARLY MEMBRANE BIOGENESIS:
  PEX3 is the EARLIEST-ACTING peroxin in the peroxisomal membrane biogenesis cascade:
  (1) PEX3 topology: single N-terminal transmembrane segment anchors PEX3 in the
      peroxisomal membrane; the large C-terminal cytoplasmic domain exposes the
      PEX19-BINDING SITE to the cytoplasm. PEX3 is a type I PMP.
  (2) PEX3-PEX19 interaction: PEX19 (a farnesylated, predominantly cytosolic protein)
      carries newly synthesised PMPs as a chaperone; PEX19 docks onto PEX3's cytoplasmic
      domain → PEX19 releases its PMP cargo into the peroxisomal membrane (class I PMP
      insertion). Without PEX3, PEX19 cannot stably engage the peroxisomal membrane,
      and PMPs are degraded or mislocalise.
  (3) PEX16 role: PEX16 is itself a PMP delivered by the PEX19 pathway; PEX16 acts as
      an additional membrane cue that stabilises nascent peroxisomal membrane and
      promotes PEX3 delivery from the ER. PEX3-PEX16 co-operate in establishing the
      peroxisomal membrane identity.
  (4) PEX3 LOF → PEX19 cannot dock → ALL PMPs fail to insert (ALDP/ABCD1, ABCD2/3,
      PMP70, PEX11, PEX14-anchor, PMP34, ALDH3A2, etc.) → peroxisomal membrane vesicles
      either absent or "ghost" (lipid only, no protein content) → matrix protein import
      secondarily fails entirely (PEX14 translocon cannot form; PEX1-PEX6 docking via
      PEX26 cannot occur without PEX26 in the membrane) → ALL peroxisomal metabolic
      pathways fail simultaneously.
  CONCEPTUAL POSITION — MOST UPSTREAM MEMBRANE BIOGENESIS DEFECT:
    PEX3 acts BEFORE PEX26 (PEX3 establishes the membrane; PEX26 is then inserted by
    PEX19 into that membrane). PEX3 LOF is UPSTREAM of PEX26, PEX1-PEX6, and
    PEX2-PEX10-PEX12 — it prevents the membrane from forming at all, not just
    preventing cycling of an import receptor.
  PEX3 LOF = biochemically IDENTICAL to PEX1/PEX6/PEX2/PEX10/PEX12/PEX26-ZSD.

LOCUS: 6q24.2  |  OMIM GENE: *603164  |  OMIM DISEASE: #614867 (PBD10A-ZSD), #614868 (PBD10B-NALD)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX3 mutations account for ~1–2% of all
  PBD-ZSD. ~10–20 confirmed PEX3 cases worldwide as of 2026 (among the rarest PBD-ZSD
  genes). No common founder allele (unlike PEX1-G843D ~30% European, PEX6-R860W ~18%
  European). p.Leu61Pro (L61Pro, near the C-terminal end of the TMD) is the best-studied
  partial-function allele; it reduces PEX3 membrane targeting efficiency and enables a
  mildly attenuated phenotype when compound heterozygous with a null allele.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (IDENTICAL TO PEX1-ZSD):
  PEX3 deficiency causes the SAME biochemical phenotype as all other PBD-ZSD because
  PEX3 is required to establish the peroxisomal membrane itself. Without the PEX3-PEX19
  docking axis, no peroxisomal membrane proteins (PMPs) are inserted:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
  (3) Plasmalogens BIOSYNTHESIS FAILED → erythrocyte plasmalogens LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD including PEX3]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — NO FOUNDER HYPOMORPHIC ALLELE):
  PEX3 cohorts show a severity distribution skewed toward severe forms because there is
  no common founder hypomorphic allele. p.Leu61Pro enables some IRD:
  (1) Zellweger Syndrome (ZS, severe): null/null; neonatal day 1–7; craniofacial dysmorphism;
      profound hypotonia; pachygyria + polymicrogyria (MRI PATHOGNOMONIC in ZS);
      neonatal seizures UNIVERSAL (100%); hepatomegaly + neonatal cholestasis; ERG absent;
      SNHL; fatal <6–12 months.
  (2) NALD (intermediate): null/L61Pro or null/hypomorphic; onset 1st–12th month;
      seizures 80–90%; infantile spasms (IS) 35–55% + hypsarrhythmia; cholestasis; survival months–years.
  (3) IRD (attenuated): L61Pro/mild or hypomorphic/hypomorphic; RP + SNHL + peripheral
      neuropathy; mild hepatomegaly; seizures 20–30%; adult survival in some cases.
  (4) Atypical: ~5%; very mild; adult-onset ataxia/RP only.
  ZS: 35% · NALD: 45% · IRD: 15% · Atypical: 5%

PEX3 UNIQUE MOLECULAR ROLE — EARLIEST MEMBRANE BIOGENESIS STEP:
  PEX3 is mechanistically UPSTREAM of every other peroxin in the ZSD series so far:
    PEX3 (membrane receptor — establishes membrane)
    → PEX19 (PMP chaperone, docks on PEX3, inserts all PMPs)
    → PEX16 (PMP, stabilises membrane)
    → PEX26 (tail-anchored PMP, inserted by PEX19, anchors PEX1-PEX6)
    → PEX1/PEX6 AAA-ATPase (retrotranslocates PEX5)
    → PEX2/PEX10/PEX12 RING E3 (ubiquitinates PEX5)
    → PEX5 cycling (imports PTS1 cargo)
  PEX3 LOF collapses ALL downstream steps simultaneously because it prevents the
  peroxisomal membrane from forming — a qualitatively different (and more upstream)
  defect than PEX26, PEX1/PEX6, or PEX2/PEX10/PEX12 LOF.

EPILEPSY IN PEX3-ZSD:
  ZS: 100% epilepsy; neonatal seizures day 1–7; multifocal clonic + tonic + myoclonic;
  hypsarrhythmia if surviving infantile period; EEG burst-suppression; drug-resistance universal.
  NALD: 80–90% epilepsy; infantile spasms (IS) 35–55% + hypsarrhythmia; focal + myoclonic.
  IRD: 20–30% epilepsy; focal + GTCS; better AED response.
  DRE: ~65–70% of all PEX3-ZSD with epilepsy (slightly higher than PEX26 due to more ZS 35%).

AED PHARMACOLOGY (IDENTICAL TO ALL PBD-ZSD):
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
  Lorenzo's Oil: INEFFECTIVE — peroxisomal membrane absent.
  HSCT: NOT INDICATED — ZSD is NOT inflammatory.
  ERT: NOT AVAILABLE — PEX3 is an integral membrane protein; no PMP delivery route.
"""

import random


# ── Overview ───────────────────────────────────────────────────────────────────
def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 84,
        "zs_pct": 35,
        "nald_pct": 45,
        "ird_pct": 15,
        "atypical_pct": 5,
        "drug_resistance_pct": 67,
        "on_dha_pct": 52,
        "liver_disease_pct": 77,
        "retinopathy_pct": 82,
        "dha_low_pct": 100,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "omim_gene": "603164",
        "omim_disease_zs": "614867",
        "omim_disease_nald": "614868",
        "locus": "6q24.2",
        "protein_size": "373 aa",
        "nbs_positive_rate": "~98% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": (
            "p.Leu61Pro (L61Pro, near C-terminal end of TMD; reduces membrane targeting efficiency; "
            "partial-function allele enabling NALD/IRD in compound het with null); "
            "p.Gln85* (null, truncating — ZS/NALD); frameshift/splice-site variants (null)"
        ),
        "disease_mechanism": (
            "PEX3 (373 aa, 6q24.2) is a TYPE I INTEGRAL PEROXISOMAL MEMBRANE PROTEIN — "
            "the DOCKING RECEPTOR for PEX19 (cytosolic PMP chaperone) on the peroxisomal membrane. "
            "PEX3's single N-terminal transmembrane segment anchors it in the membrane; its large "
            "cytoplasmic C-terminal domain provides the PEX19-binding site. "
            "PEX3 LOF → PEX19 cannot engage the peroxisomal membrane → ALL peroxisomal membrane "
            "proteins (PMPs) fail to insert (ALDP/ABCD1, PMP70, PEX14, PEX26, ALDH3A2, etc.) → "
            "peroxisomal membranes absent or 'ghost' (lipid only) → matrix import secondarily fails → "
            "ALL peroxisomal pathways impaired simultaneously. "
            "PEX3 acts UPSTREAM of PEX26, PEX1/PEX6, and PEX2/PEX10/PEX12 — it establishes the "
            "membrane that all downstream peroxins require. "
            "Biochemically identical to PEX1/PEX6/PEX2/PEX10/PEX12/PEX26-ZSD; only NGS distinguishes. "
            "~1–2% of all PBD-ZSD; ~10–20 cases worldwide 2026."
        ),
        "key_concepts": [
            "PEX3 (373 aa, 6q24.2): type I integral peroxisomal membrane protein; N-terminal TMD anchors PEX3; large cytoplasmic C-terminal domain = DOCKING RECEPTOR for PEX19",
            "PEX3-PEX19 docking: PEX19 (farnesylated cytosolic PMP chaperone) carries newly synthesised PMPs; PEX19 docks onto PEX3's cytoplasmic domain → releases PMP cargo into peroxisomal membrane (class I PMP insertion pathway)",
            "PEX3-PEX16-PEX19 early biogenesis axis: PEX16 (another PMP) co-operates with PEX3 in establishing peroxisomal membrane identity; PEX3 is the receptor that anchors the entire PEX19-mediated PMP insertion system",
            "PEX3 LOF → PEX19 cannot dock → ALL PMPs fail to insert → peroxisomal membranes absent or 'ghost' (lipid bilayer only, no proteins) → ALL peroxisomal matrix import secondarily fails",
            "Most upstream defect in the PEX series: PEX3 establishes the membrane before any other biogenesis steps. PEX3 LOF is upstream of PEX26 (also a PMP requiring PEX19 for insertion), PEX1/PEX6 motor, and PEX2/PEX10/PEX12 RING complex",
            "Biochemical identity: PEX3-ZSD is biochemically IDENTICAL to all other PBD-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
            "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX3; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
            "DHA synthesis failure: DHA synthesised peroxisomally; PEX3 deficiency → membrane absent → DHA synthesis fails → DHA low → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + RP + neuropathy",
            "Phenotypic spectrum: ZS 35% (null/null, more ZS than PEX26 25%) · NALD 45% · IRD 15% · Atypical 5%; no founder hypomorphic allele → spectrum skewed severe",
            "p.Leu61Pro (L61Pro): near C-terminal end of TMD; reduces membrane insertion efficiency of PEX3 (~20–40% residual membrane targeting); enables NALD/IRD phenotype as compound het with null allele",
            "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A)",
            "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD; IRD in 15%) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH Level A preferred for IS",
            "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → acute neurotoxicity + seizures; IV dextrose MANDATORY during any NPO",
            "Lorenzo's Oil INEFFECTIVE (peroxisomal membrane absent — PEX3 LOF prevents PMP insertion; not just import; LOO cannot restore the membrane receptor)",
            "HSCT NOT INDICATED (ZSD not inflammatory; donor microglia cannot restore PEX3 membrane receptor or PEX19 docking)",
            "No ERT (PEX3 = integral type I membrane protein; protein replacement not applicable for a membrane receptor)",
            "~10–20 cases worldwide 2026 (rarest of the ZSD series built); COMPLETES the PEX3-PEX16-PEX19 early membrane biogenesis axis",
        ],
        "standards": [
            "Fang Y et al. PEX3 functions as a PEX19 docking factor in the import of class I peroxisomal membrane proteins. J Cell Biol. 2004",
            "Muntau AC et al. Defective peroxisome membrane synthesis due to mutations in human PEX3 causes Zellweger syndrome, infantile Refsum disease and neonatal adrenoleukodystrophy. Am J Hum Genet. 2000",
            "Ghaedi K et al. The peroxisome assembly-defective Chinese hamster ovary cell mutant ZP92 and Zellweger syndrome patient fibroblasts complementation groups 12 and 13. J Biol Chem. 2000",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
            "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
            "Fransen M et al. Peroxisomal dynamics and disease. Curr Opin Cell Biol. 2006",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614867 (PBD10A-ZSD — PEX3); #614868 (PBD10B-NALD — PEX3); *603164 (PEX3 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(36241)  # PEX3 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 35, "n": 14,
            "sex": "7M/7F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + electrographic; EEG burst-suppression",
            "eeg": "Burst-suppression → multifocal spike-wave; bilateral asynchronous; EEG monitoring NICU mandatory; poor prognosis",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination; enlarged ventricles",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Gln85*/p.Gln85* or p.Gln85*/frameshift/splice); PEX3 absent from membrane; PEX19 cannot dock; ghost peroxisomes; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Leu61Pro or splice",
            "pct": 45, "n": 18,
            "sex": "9M/9F",
            "onset_age": "Infantile (1st–12th month)",
            "seizure_risk": "80–90% — infantile spasms (IS) 35–55% + hypsarrhythmia; focal + generalized + myoclonic",
            "eeg": "Hypsarrhythmia (IS period); multifocal spike-wave; modified hypsarrhythmia with burst-suppression elements",
            "mri": "Periventricular white matter signal; delayed myelination; cerebellar hypoplasia; milder gyral pattern than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Leu61Pro (partial membrane insertion) or null/splice; p.Leu61Pro near TMD-cytoplasmic junction reduces PEX3 membrane targeting ~60%; partial PEX19 docking; survival months to years",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Leu61Pro/mild",
            "pct": 15, "n": 6,
            "sex": "3M/3F",
            "onset_age": "Childhood / Adolescence (2–18 yr)",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD; phytol-restricted diet reduces burden",
            "eeg": "Focal epileptiform discharges; normal background early; generalised spike-wave with progression",
            "mri": "RP-associated retinal thinning; mild cerebellar atrophy; white matter changes mild; no pachygyria",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Leu61Pro/hypomorphic compound het; ~20–40% residual PEX3 membrane insertion → partial PEX19 docking → partial PMP insertion → partial peroxisomal function; adult survival possible",
        },
        {
            "name": "Atypical / Late-Onset ZSD — very mild alleles",
            "pct": 5, "n": 2,
            "sex": "1M/1F",
            "onset_age": "Adult (18–50 yr)",
            "seizure_risk": "10–15% — rare GTCS or focal; responsive to LEV",
            "eeg": "Normal background; rare focal discharges; incidental finding during ataxia/RP workup",
            "mri": "Mild cerebellar atrophy; RP retinal thinning; white matter normal or minimally abnormal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles hypomorphic (very mild missense near TMD); substantial residual PEX3 membrane insertion; late-onset slowly progressive ataxia + RP",
        },
    ]

    def rand_sex():
        return random.choice(["M", "F"])

    def rand_geno(phenotype):
        if "ZS" in phenotype:
            variants = ["p.Gln85*/p.Gln85*", "p.Gln85*/c.122del", "p.Gln85*/p.Arg117*", "p.Gln85*/c.1+2T>C"]
        elif "NALD" in phenotype:
            variants = ["p.Gln85*/p.Leu61Pro", "p.Arg117*/p.Leu61Pro", "p.Leu61Pro/c.122del", "p.Gln85*/c.572-1G>T"]
        elif "IRD" in phenotype:
            variants = ["p.Leu61Pro/p.Ala314Val", "p.Leu61Pro/c.572-1G>T", "p.Leu61Pro/p.Gly324Ser", "p.Leu61Pro/p.Pro332Leu"]
        else:
            variants = ["p.Leu61Pro/p.Ile345Met", "p.Leu61Pro/p.Ala314Val"]
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
        ("ZS (Zellweger-Severe)", 14),
        ("NALD (Neonatal-ALD-Intermediate)", 18),
        ("IRD (Infantile-Refsum-Attenuated)", 6),
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
                "patient_id": f"PEX3-{pid:03d}",
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
        {"type": "Neonatal multifocal clonic (ZS)", "pct": 35, "eeg": "Burst-suppression → multifocal spike; bilateral asynchronous; EEG NICU mandatory; universal in ZS"},
        {"type": "Infantile spasms / West syndrome (NALD)", "pct": 45, "eeg": "Hypsarrhythmia → modified; ACTH Level A first-line; VGB HIGH RISK (avoid — ZSD retinopathy additive)"},
        {"type": "Focal seizures bilateral spread (NALD/IRD)", "pct": 22, "eeg": "Temporal/parietal focal spike-wave; post-ictal slowing; LEV first-line"},
        {"type": "Myoclonic seizures (NALD)", "pct": 18, "eeg": "Generalised polyspike-wave ≥3 Hz; CBZ/PHT HIGH RISK (myoclonic exacerbation)"},
        {"type": "GTCS (IRD/Atypical)", "pct": 12, "eeg": "Generalised spike-wave; good post-ictal recovery in IRD; LEV + LTG effective"},
        {"type": "Absence / atypical absence (rare, IRD)", "pct": 3, "eeg": "2.5–3.5 Hz generalised spike-wave; ETX Level C consideration in IRD"},
    ]

    triggers = [
        {"trigger": "Fever / intercurrent infection", "pct": 68, "note": "Universal ZSD trigger — fever lowers seizure threshold + increases metabolic demand → phytanic mobilisation"},
        {"trigger": "Subtherapeutic AED levels", "pct": 55, "note": "Common in NALD/IRD — growth spurts alter pharmacokinetics; TDM quarterly"},
        {"trigger": "Fasting / prolonged NPO", "pct": 50, "note": "EXTREME HAZARD — adipose phytanic mobilisation → acute neurotoxicity; IV dextrose MANDATORY"},
        {"trigger": "Sleep deprivation", "pct": 42, "note": "Worsens electrocortical excitability; melatonin 0.5–3 mg adjunct in NALD/IRD with sleep disruption"},
        {"trigger": "Metabolic decompensation / catabolism", "pct": 38, "note": "Intercurrent illness, surgery, missed feeds → phytanic surge + VLCFA accumulation worsens acutely"},
        {"trigger": "Missed AED dose", "pct": 32, "note": "Non-adherence frequent in long-term NALD/IRD; simplified LEV BID regimen improves adherence"},
        {"trigger": "Enzyme-inducing AEDs (PHT/CBZ/OXC)", "pct": 20, "note": "CYP3A4 induction → DHA depletion (already low) + hepatic burden; RELATIVE CI; replace with LEV"},
        {"trigger": "Anaesthesia / surgical stress", "pct": 17, "note": "NPO + surgical stress → phytanic surge; pre-op: PT/INR + LFT + platelet + IV VitK; IV dextrose mandatory"},
    ]

    monitoring = [
        "VLCFA plasma panel (C26:0, C26:0/C22:0 ratio) — baseline + q6M (disease tracking + AED-induced metabolic change)",
        "Erythrocyte plasmalogens — baseline + q6M (PEX3-PEX19 PMP insertion function marker; LOW in all ZSD)",
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
        {"parameter": "C26:0-lyso-PC (NBS DBS)", "threshold": ">0.10 μmol/L", "action": "Refer to metabolic specialist STAT; confirm with plasma VLCFA + plasmalogens; NGS PEX gene panel (include PEX3)"},
        {"parameter": "Plasma C26:0/C22:0 ratio", "threshold": ">0.020", "action": "Diagnostic for ZSD biochemically; initiate NGS PEX panel; PEX3 among first-tier genes after PEX1/PEX6"},
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
            "features": "Profound hypotonia; neonatal seizures day 1–7 (multifocal clonic + tonic); EEG burst-suppression; cholestatic jaundice; craniofacial dysmorphism; ERG absent; SNHL on ABR; MRI shows pachygyria; ghost peroxisomes on fibroblast catalase staining",
            "action": "LEV IV first-line; Phenobarbital if refractory; IV dextrose continuous; VLCFA + plasmalogens STAT; NGS PEX panel STAT (include PEX3); palliative care discussion; POLG1 mandatory",
        },
        {
            "stage": "Stage 2 — Early Infantile (1–12 months, NALD dominant)",
            "features": "Infantile spasms (IS) with hypsarrhythmia (35–55%); psychomotor regression; cholestasis evolving; SNHL; visual impairment from RP; nasogastric feeds; ghost peroxisomes on fibroblast EM",
            "action": "ACTH Level A (preferred over VGB for IS — ZSD retinopathy risk); LEV adjunct; DHA Level B; ophthalmology + audiology baseline; confirm PEX3 biallelic variants by NGS",
        },
        {
            "stage": "Stage 3 — Late Infantile/Toddler (1–3 yr, NALD/IRD)",
            "features": "Seizure evolution from IS to focal + myoclonic; developmental plateau; RP progression; SNHL; ataxia emerging in IRD; phytol-restricted diet starts in IRD",
            "action": "Optimise LEV ± CLB adjunct; phytol-restricted diet Level B (IRD); DHA continue; VPA NEVER; ERG + visual field + ABR annual; NGS confirms PEX3 identity vs PEX1/PEX6",
        },
        {
            "stage": "Stage 4 — Preschool/School Age (3–12 yr, IRD/Atypical)",
            "features": "Focal ± GTCS; intellectual disability mild–moderate; ataxia prominent; RP → tunnel vision; SNHL stable; hepatomegaly mild; peripheral neuropathy emerging",
            "action": "LEV ± LTG Level C; strict phytol-restricted diet; annual VLCFA + phytanic + DHA; neuropsychological support; adaptive aids; transition planning",
        },
        {
            "stage": "Stage 5 — Adolescence/Adulthood (12+ yr, IRD/Atypical only)",
            "features": "Seizure frequency stable in well-controlled IRD; RP → legal blindness; SNHL; progressive ataxia; peripheral neuropathy; psychiatric comorbidity",
            "action": "Long-term LEV ± LTG; phytol-restricted diet; annual metabolic review; genetic counselling (AR — 25% recurrence); no ERT/HSCT; gene therapy (AAV-PEX3) preclinical",
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
            "evidence": "Level A (expert consensus — multiple ZSD cohorts including PEX3)",
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
            "alternative": "LEV IV/PO (first-line). ACTH Level A (IS). LEV + CLB or LEV + LTG (IRD focal). NEVER VPA in PEX3-ZSD.",
        },
        {
            "drug": "Vigabatrin (VGB / Sabril)",
            "level": "HIGH RISK (avoid in IRD; near-CI in ZS/NALD)",
            "reason": "ZSD causes retinopathy (universal ZS/NALD; 15–20% IRD as RP). VGB irreversibly constricts visual fields (concentric VF loss, 30–50% users). In ZSD, VGB-induced VF loss is ADDITIVE to existing RP → catastrophic irreversible blindness. ACTH (Level A) is always preferred for IS in all ZSD forms.",
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
            "reason": "CYP3A4 induction → (1) DHA catabolism accelerated (already deficient in ZSD → worsens RP/neuropathy) + (2) hepatic microsomal burden (cholestatic liver ZS/NALD). NOT adrenal crisis (unlike ABCD1 ABSOLUTE CI — no universal adrenal insufficiency in PEX3-ZSD). May be used in specific focal seizures (IRD) if no alternative with close DHA + LFT monitoring.",
            "alternative": "LEV first-line; LTG glucuronidation Level C (IRD); CLB adjunct. Replace with LEV on diagnosis.",
        },
        {
            "drug": "Lorenzo's Oil (oleic acid + erucic acid)",
            "level": "INEFFECTIVE (do NOT use in ZSD)",
            "reason": "Lorenzo's Oil inhibits VLCFA elongation → works in X-ALD (ABCD1) because peroxisomal import is intact but ABCD1 transporter absent. In PEX3-ZSD, the PEROXISOMAL MEMBRANE ITSELF is absent (PEX3 LOF → PEX19 cannot dock → all PMPs fail → peroxisomes are ghost vesicles). LOO cannot restore the membrane receptor PEX3 or reconstitute peroxisomal membrane biogenesis. No benefit; possible side effects (thrombocytopenia).",
            "alternative": "DHA Level B (NALD/IRD). Phytol-restricted diet Level B (IRD). No VLCFA-specific therapy available 2026.",
        },
        {
            "drug": "HSCT (Hematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED (PEX3-ZSD)",
            "reason": "HSCT benefits conditions with a neuroinflammatory component (cerebral X-ALD / ABCD1 — donor microglia halt demyelination). PEX3-ZSD is NOT primarily inflammatory; pathology is metabolic (peroxisomal membrane biogenesis failure). Donor microglia cannot restore PEX3 membrane receptor function or reconstitute peroxisomal membrane biogenesis. HSCT carries 5–15% transplant-related mortality — unacceptable risk-benefit.",
            "alternative": "Supportive care: DHA, phytol-restricted diet, LEV. Gene therapy (AAV9-PEX3) in preclinical research.",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE (PEX3 = type I integral membrane protein)",
            "reason": "ERT is feasible for soluble lysosomal enzymes that can be endocytosed via mannose-6-phosphate receptor. PEX3 is a TYPE I INTEGRAL PEROXISOMAL MEMBRANE PROTEIN (N-terminal TMD in membrane, C-terminal cytoplasmic domain = PEX19 receptor) — not a soluble secreted enzyme. ERT cannot deliver an integral membrane receptor protein to its intracellular target membrane. No applicable ERT mechanism.",
            "alternative": "Gene therapy (AAV-PEX3) in preclinical research; mRNA delivery experimental. No clinically available ERT 2026.",
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
        "PEX3 (373 aa, 6q24.2): type I integral peroxisomal membrane protein; single N-terminal transmembrane domain anchors PEX3 in the peroxisomal membrane; large C-terminal cytoplasmic domain = DOCKING RECEPTOR for PEX19",
        "PEX3 function: PEX19 (farnesylated cytosolic PMP chaperone) docks onto PEX3's cytoplasmic domain → releases PMP cargo (ALDP, PMP70, PEX14, PEX26, etc.) into the peroxisomal membrane via class I PMP insertion pathway",
        "PEX3-PEX16-PEX19 early biogenesis axis: PEX16 (a PMP) co-operates with PEX3 to establish membrane identity; together they form the LANDING PLATFORM for PEX19-mediated PMP insertion — the founding event of peroxisome biogenesis",
        "PEX3 LOF outcome: PEX19 cannot dock → ALL peroxisomal membrane proteins fail to insert → peroxisomes absent or 'ghost vesicles' (lipid bilayer only, no PMPs) → ALL peroxisomal matrix import secondarily fails",
        "Most upstream defect in ZSD series built: PEX3 establishes the peroxisomal membrane BEFORE any other biogenesis step. PEX26, PEX1/PEX6, PEX2/PEX10/PEX12 all depend on a functional peroxisomal membrane that PEX3-PEX19 creates",
        "Biochemical identity: PEX3-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX2/PEX10/PEX12/PEX26-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
        "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX3; NORMAL in ABCD1 (X-ALD); single test separates the diagnoses",
        "DHA synthesis failure: DHA synthesised peroxisomally (requires PTS1-targeted enzymes which need the peroxisomal membrane to exist); PEX3 failure → DHA synthesis fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + RP + neuropathy",
        "Phenotypic spectrum: ZS 35% (null/null) · NALD 45% · IRD 15% · Atypical 5%; more ZS than PEX26 (25%) — no founder hypomorphic allele, spectrum skewed severe toward null genotypes",
        "p.Leu61Pro (L61Pro): near C-terminal end of TMD; partial reduction of PEX3 membrane insertion efficiency (~20–40% residual); enables NALD/IRD when compound het with null allele — best-studied hypomorphic PEX3 variant",
        "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A) before any VPA consideration",
        "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD; IRD in 15%) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH preferred for IS in all ZSD forms",
        "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → direct neurotoxicity + seizures + arrhythmia; IV dextrose prevents catabolism; MANDATORY during any NPO",
        "PHT/CBZ/OXC RELATIVE CI (NOT absolute, unlike ABCD1): CYP3A4 induction depletes DHA + increases hepatic burden. No adrenal crisis in PEX3-ZSD (adrenal insufficiency not universal). Different from ABCD1 ABSOLUTE CI",
        "Lorenzo's Oil mechanism mismatch: LOO inhibits VLCFA elongase → effective in ABCD1 (import intact, transporter absent). In PEX3-ZSD, the peroxisomal MEMBRANE IS ABSENT (PEX3 receptor failure → PEX19 cannot dock → no PMPs → ghost vesicles). LOO cannot restore membrane biogenesis → INEFFECTIVE",
        "Ghost peroxisomes in PEX3 LOF: unlike some peroxin deficiencies that produce functional ghosts with intact membranes, PEX3 LOF is so upstream that peroxisomal membrane vesicles may be absent entirely or reduced to lipid bilayers without any PMP content — detectable by catalase immunofluorescence + EM in fibroblasts",
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
        "Step 10 — Fibroblast catalase staining + EM: absent or ghost peroxisomes (confirms PBD mechanistically; seen in PEX3 and other upstream biogenesis defects)",
        "Step 11 — Comprehensive NGS PEX gene panel (PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26): identifies PEX3 biallelic pathogenic variants; confirms gene-level diagnosis",
        "Step 12 — POLG1 sequencing: MANDATORY before considering any VPA (CPIC Grade A); eliminates mitochondrial aetiology",
        "Step 13 — Genetic counselling: AR inheritance (25% recurrence risk); prenatal diagnosis available (PEX3 targeted sequencing in CVS/amnio); carrier testing for parents/siblings",
    ]

    pharmacological_distinctions = [
        "LEV vs PHT/CBZ/OXC in ZSD: LEV = no CYP induction, no hepatotoxicity, IV available → FIRST-LINE; PHT/CBZ/OXC = CYP3A4 → DHA depletion + hepatic burden → RELATIVE CI",
        "VPA vs LEV: VPA = THREE CI mechanisms (hepatotoxicity + peroxisomal BO inhibition + carnitine depletion) → HIGH RISK near-absolute CI; LEV = none → always preferred",
        "ACTH vs VGB for IS: In ZSD, ACTH Level A (preferred) vs VGB HIGH RISK (retinopathy additive). Never use VGB first-line in any ZSD form",
        "DHA supplementation: LPC-DHA (lysophosphatidylcholine-DHA) form bypasses peroxisomal synthesis via intestinal absorption. Ethyl ester less effective (requires peroxisomal processing). Specifically use LPC-DHA",
        "Phytol-restricted diet in IRD vs no benefit in ZS: In ZS (fatal <12M), restriction has no benefit. In IRD/Atypical (long survival), phytol restriction reduces phytanic accumulation → fewer seizures + slower neuropathy",
        "Lorenzo's Oil in ABCD1 vs PEX3-ZSD: LOO effective in ABCD1 (import intact, elongase inhibition reduces VLCFA). In PEX3-ZSD, peroxisomal MEMBRANE absent (PEX3 receptor failure) → LOO cannot help → INEFFECTIVE",
        "HSCT in ABCD1 vs ZSD: HSCT standard of care in early CCALD (neuroinflammatory). PEX3-ZSD is NOT inflammatory → HSCT = transplant mortality risk with zero mechanistic benefit",
        "PHT/CBZ ABSOLUTE CI (ABCD1) vs RELATIVE CI (PEX3-ZSD): In ABCD1, PHT/CBZ → adrenal crisis (universal adrenal insufficiency + CYP induction). In PEX3-ZSD, no universal adrenal insufficiency → PHT/CBZ = RELATIVE CI (DHA/hepatic only, not adrenal)",
        "VPA POLG1 mandatory (CPIC Grade A): POLG1 exclusion mandatory in ZSD because: (1) VPA CI in POLG1 (separate mitochondrial mechanism) + (2) ZSD patients may also carry POLG1 variants → catastrophic combined risk",
        "Carnitine supplementation: indicated if free carnitine <25 μmol/L; L-carnitine 50–100 mg/kg/day; critical if VPA inadvertently started",
        "IV Vitamin K perioperative: cholestatic liver (DHCA/THCA-driven) → fat-soluble vitamin K malabsorption → coagulopathy (↑PT/INR); IV Vitamin K 1–2 mg/kg (max 10 mg) pre-surgery; check PT/INR + platelet pre-op MANDATORY",
        "PEX3 vs PEX26 therapeutic implications: Both LOF → same biochemical phenotype → same AED rules. PEX3 is upstream (membrane receptor — enables PEX19 docking and ALL PMP insertion including PEX26 itself). No different treatment targets; gene therapy approach would differ (AAV-PEX3 must restore the membrane receptor upstream of PEX26)",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX1-ZSD (65% of all PBD — most common)",
            "distinction": "Biochemically IDENTICAL. PEX1 = D1 AAA-ATPase subunit (motor); PEX26 (a PMP) anchors PEX1-PEX6 to the membrane that PEX3 built. p.Gly843Asp founder allele (30% European) → IRD 30%. PEX3 far rarer (~10–20 cases). PEX3 at 6q24.2 vs PEX1 at 7q21.2. NGS distinguishes. Both responsive to same ZSD AED protocol.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 = D2 AAA-ATPase subunit (partner to PEX1; docked on peroxisomal membrane via PEX26, which requires PEX3 for membrane insertion). p.Arg860Trp (R860W) founder allele → IRD 45% (most attenuated ZSD). PEX3 has less IRD (15%). 6q24.2 vs 6p21.1. NGS distinguishes.",
        },
        {
            "condition": "PEX26-ZSD (~2–4% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX26 = tail-anchored PMP (membrane docking platform for PEX1-PEX6 motor); PEX26 is ITSELF a PMP that requires PEX3-PEX19 for insertion into the peroxisomal membrane. PEX3 LOF is UPSTREAM of PEX26 — PEX26 cannot be inserted without PEX3. Both very rare; phenotype identical. 6q24.2 (PEX3) vs 22q11.21 (PEX26). NGS distinguishes.",
        },
        {
            "condition": "PEX12-ZSD / PEX10-ZSD / PEX2-ZSD (RING E3 triad, 3–7% combined)",
            "distinction": "Biochemically IDENTICAL. PEX2/PEX10/PEX12 = RING E3 ubiquitin ligase triad (ubiquitinate PEX5); all three are PMPs requiring PEX3-PEX19 for membrane insertion. PEX3 LOF upstream of all three. No founder alleles in PEX3; PEX10 and PEX12 have partial-function alleles enabling more IRD. NGS distinguishes.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy)",
            "distinction": "VLCFA elevated in BOTH. ABCD1: plasmalogens (RBC) NORMAL + DHA NORMAL + adrenal insufficiency (universal, X-linked males). PHT/CBZ = ABSOLUTE CI in ABCD1 (adrenal crisis) vs RELATIVE CI in PEX3-ZSD. Lorenzo's Oil EFFECTIVE in ABCD1 (intact peroxisome, transporter absent); INEFFECTIVE in PEX3-ZSD (membrane itself absent). HSCT standard CCALD; NOT in PEX3-ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC severely elevated (SOLE elevated peroxisomal metabolite); VLCFA NORMAL (key vs PEX3); plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. Phytol-restricted diet Level A GOLD STANDARD. PHYH is a PTS2-cargo enzyme requiring PEX7 for import; its deficiency is downstream of the membrane biogenesis PEX3 step.",
        },
        {
            "condition": "RCDP / PEX7 (Rhizomelic Chondrodysplasia Punctata Type 1)",
            "distinction": "Plasmalogens (RBC) LOW (like ZSD) BUT VLCFA NORMAL + phytanic elevated + pristanic NORMAL + DHA NORMAL + pipecolic NORMAL. Only PTS2 pathway affected (PEX7 receptor deficient). Rhizomelia PATHOGNOMONIC. Stippled epiphyses on X-ray. No pachygyria. PEX7 protein also requires PEX3-PEX19 for membrane insertion — PEX3 LOF upstream of PEX7.",
        },
        {
            "condition": "Mitochondrial / POLG1 Epilepsy (Alpers)",
            "distinction": "VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Lactate/pyruvate elevated. Hepatocerebral phenotype. VPA CATASTROPHIC CI in POLG1 (mitochondrial depletion) — same practical CI as ZSD but different mechanism. POLG1 MANDATORY in all ZSD before VPA.",
        },
    ]

    standards = [
        "Fang Y et al. PEX3 functions as a PEX19 docking factor in the import of class I peroxisomal membrane proteins. J Cell Biol. 2004",
        "Muntau AC et al. Defective peroxisome membrane synthesis due to mutations in human PEX3 causes Zellweger syndrome. Am J Hum Genet. 2000",
        "Ghaedi K et al. The peroxisome membrane-targeting signal of PEX3. J Biol Chem. 2000",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
        "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
        "Fransen M et al. Peroxisomal dynamics and disease. Curr Opin Cell Biol. 2006",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614867 (PBD10A-ZSD — PEX3); #614868 (PBD10B-NALD — PEX3); *603164 (PEX3 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
