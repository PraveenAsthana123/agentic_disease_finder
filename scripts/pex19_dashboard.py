#!/usr/bin/env python3
"""PEX19 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX19 encodes Peroxin-19,
a 299-amino-acid FARNESYLATED CYTOPLASMIC PMP CHAPERONE — the molecular shuttler that
carries newly synthesised class I Peroxisomal Membrane Proteins (PMPs) from the cytoplasm
to the peroxisomal membrane, docking onto PEX3. PEX19 COMPLETES the
PEX3-PEX16-PEX19 early membrane biogenesis axis.

MECHANISM — CYTOPLASMIC PMP CHAPERONE / PEX3-DOCKING ARM:
  PEX19 is the EFFECTOR ARM of the PEX3-PEX16-PEX19 membrane biogenesis axis:
  (1) PEX19 structure: 299 aa; predominantly cytoplasmic; C-terminal CAAX motif
      (Cys296-Ser-Ile-Met) undergoes farnesylation by farnesyltransferase → farnesyl
      group is essential for PEX19 membrane affinity and PMP binding efficiency.
      N-terminal domain (aa 1–44): nuclear export signal + dimerisation motif.
      C-terminal domain (aa 240–299): PMP-binding / mPTS-recognition domain.
      Central flexible linker: PEX3-binding site (aa 14–30: PEX3 interaction helix).
  (2) Class I PMP insertion pathway (PEX19-mediated):
      Step 1 — Synthesis: ribosome synthesises PMP in cytoplasm; PEX19 binds the mPTS
               (membrane-targeting signal) in the PMP immediately post-translationally,
               preventing PMP aggregation and premature membrane targeting.
      Step 2 — Chaperoning: PEX19·PMP complex travels through cytoplasm.
      Step 3 — Docking: PEX19 N-terminal PEX3-binding helix docks onto PEX3's
               cytoplasmic C-terminal domain on the peroxisomal membrane.
      Step 4 — Insertion: PMP is released from PEX19 into the lipid bilayer of the
               peroxisomal membrane; mechanism may involve lipid transfer / conformational
               change driven by PEX3 engagement.
      Step 5 — Recycling: PEX19 is released back to cytoplasm for another round.
  (3) PEX16 context: PEX16 (type II integral PMP delivered from ER) recruits PEX3 from
      ER-derived vesicles to the peroxisomal membrane. PEX3 (type I integral PMP), once
      correctly localised on the peroxisomal membrane (requiring PEX16), becomes the
      receptor for PEX19. PEX19 CANNOT dock if PEX3 is absent or mislocalized (as occurs
      in PEX16 LOF). PEX19 LOF prevents all PMPs from inserting even if PEX3/PEX16 are
      intact.
  (4) PEX19 LOF → all class I PMPs (ALDP/ABCD1, PMP70, PEX14, PEX26, PEX11β, ALDH3A2,
      PMP34, etc.) fail to insert → peroxisomal membranes absent or 'ghost' (lipid only,
      no protein content) → matrix protein import secondarily fails entirely (PEX14
      translocon cannot form; PEX5 import receptor cycling fails; PEX1-PEX6 motor via
      PEX26 cannot function) → ALL peroxisomal metabolic pathways fail simultaneously.
  (5) Farnesylation of PEX19 is required for function: p.Cys296Ser abolishes farnesylation
      → complete LOF. p.Ala318Val (or equivalent near-CAAX missense) reduces farnesylation
      efficiency by ~50–70% → partial PMP docking → enables NALD/IRD phenotype.
  AXIS SUMMARY:
    PEX16 (ER→peroxisome delivery, recruits PEX3) →
    PEX3 (docking receptor on peroxisomal membrane) →
    PEX19 (cytoplasmic chaperone, docks to PEX3, inserts all class I PMPs)
  PEX19 LOF = biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX16/PEX2/PEX10/PEX12/PEX26-ZSD.

LOCUS: 1q22  |  OMIM GENE: *600279  |  OMIM DISEASE: #614882 (PBD11A-ZSD), #614883 (PBD11B-NALD)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX19 mutations account for ~1–3% of all
  PBD-ZSD. ~15–30 confirmed PEX19 cases worldwide as of 2026. Rarer than PEX16 (~30–60
  cases) and PEX3 (~10–20 cases). No common founder allele (unlike PEX1-G843D ~30% European,
  PEX6-R860W ~18% European). p.Ala318Val (near CAAX farnesylation site) is the best-studied
  partial-function allele; reduces farnesylation efficiency and PMP docking → NALD/IRD
  phenotype when compound heterozygous with null. p.Cys296Ser abolishes farnesylation
  completely → null phenotype.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (IDENTICAL TO PEX1-ZSD):
  PEX19 deficiency causes the SAME biochemical phenotype as all other PBD-ZSD because PEX19
  is required to insert ALL class I PMPs into the peroxisomal membrane:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
  (3) Plasmalogens BIOSYNTHESIS FAILED → erythrocyte plasmalogens LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD including PEX19]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — NO FOUNDER HYPOMORPHIC ALLELE):
  PEX19 cohorts show a severity distribution similar to PEX3:
  ZS ~35%  (null/null → neonatal seizures, fatal <12M)
  NALD ~45% (null/partial → infantile spasms, survival months–years)
  IRD ~15%  (partial/partial → childhood-onset, adult survival)
  Atypical ~5% (mild alleles → adult-onset RP + ataxia + neuropathy)
  ZS fraction (35%) similar to PEX3 (35%); slightly higher than PEX16 (30%).

EPILEPSY IN PEX19-ZSD:
  ZS: 100% epilepsy; neonatal seizures day 1–7; multifocal clonic + tonic + myoclonic;
  hypsarrhythmia if surviving infantile period; EEG burst-suppression; DRE universal.
  NALD: 80–90% epilepsy; infantile spasms (IS) 35–55% + hypsarrhythmia; focal + myoclonic.
  IRD: 20–30% epilepsy; focal + GTCS; better AED response.
  DRE: ~65–70% of all PEX19-ZSD with epilepsy (same as PEX3 — same mechanistic tier).

AED PHARMACOLOGY (IDENTICAL TO ALL PBD-ZSD):
  LEV: FIRST-LINE (all ZSD forms). No CYP induction; no hepatotoxicity; IV available.
  ACTH: Level A for IS. Preferred over VGB due to ZSD retinopathy.
  CLB: Level B adjunct. No hepatotoxicity; no CYP induction.
  LTG: Level C (IRD/NALD). Glucuronidation — safer hepatic profile.
  DHA: Level B (NALD/IRD). Bypasses peroxisomal synthesis via LPC-DHA intestinal absorption.
  Phytol-restricted diet: Level B (IRD/Atypical). Reduces phytanic accumulation.
  VPA: HIGH RISK — THREE mechanisms: hepatotoxicity + peroxisomal BO inhibition + carnitine.
    POLG1 MANDATORY (CPIC Grade A). Near-absolute CI in ZS/NALD.
  VGB: HIGH RISK — additive retinopathy (ZSD retinopathy + VGB VF constriction). ACTH preferred for IS.
  PHT/CBZ/OXC: RELATIVE CI — CYP3A4 induction → DHA depletion + hepatic burden.
  Lorenzo's Oil: INEFFECTIVE — peroxisomal membrane absent (PEX19 LOF → no PMPs insert → no BO possible).
  HSCT: NOT INDICATED — ZSD is NOT inflammatory (no microglia rescue here).
  ERT: NOT AVAILABLE — PEX19 is a farnesylated cytoplasmic chaperone; protein replacement
    therapy not applicable for a soluble chaperone without a classical lysosomal delivery route.
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
        "omim_gene": "600279",
        "omim_disease_zs": "614882",
        "omim_disease_nald": "614883",
        "locus": "1q22",
        "protein_size": "299 aa",
        "nbs_positive_rate": "~98% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": (
            "p.Ala318Val (near C-terminal CAAX farnesylation motif; reduces farnesylation efficiency "
            "~50–70%; partial PEX19 docking activity → NALD/IRD when compound het with null); "
            "p.Cys296Ser (abolishes CAAX farnesylation completely → null phenotype → ZS/NALD); "
            "p.Leu78fs (frameshift null, truncating — ZS/NALD); splice-site variants (null)"
        ),
        "disease_mechanism": (
            "PEX19 (299 aa, 1q22) is a FARNESYLATED CYTOPLASMIC PMP CHAPERONE — "
            "the effector arm of the PEX3-PEX16-PEX19 early membrane biogenesis axis. "
            "PEX19 binds newly synthesised class I PMPs (including ALDP/ABCD1, PMP70, PEX14, PEX26, "
            "PEX11β, ALDH3A2) via its C-terminal mPTS-recognition domain immediately post-translationally, "
            "preventing aggregation. The PEX19·PMP complex travels to the peroxisomal membrane, where "
            "PEX19's N-terminal PEX3-binding helix docks onto PEX3 (which is positioned by PEX16). "
            "PMP cargo is released into the lipid bilayer; PEX19 recycles to cytoplasm. "
            "FARNESYLATION of PEX19's C-terminal CAAX (Cys296-Ser-Ile-Met) is REQUIRED for normal "
            "membrane affinity and PMP docking efficiency. "
            "PEX19 LOF → ALL class I PMPs fail to insert → ghost peroxisomes (lipid only, no proteins) → "
            "matrix import secondarily fails → ALL peroxisomal metabolic pathways fail simultaneously. "
            "Biochemically identical to PEX1/PEX6/PEX3/PEX16/PEX2/PEX10/PEX12/PEX26-ZSD; "
            "only NGS distinguishes. ~1–3% of all PBD-ZSD; ~15–30 cases worldwide 2026. "
            "COMPLETES the PEX3-PEX16-PEX19 early membrane biogenesis axis."
        ),
        "key_concepts": [
            "PEX19 (299 aa, 1q22): farnesylated cytoplasmic PMP chaperone; C-terminal CAAX (Cys296-Ser-Ile-Met) farnesylated by farnesyltransferase; N-terminal PEX3-binding helix (aa 14–30); C-terminal mPTS-recognition / PMP-binding domain (aa 240–299)",
            "Class I PMP insertion pathway: PEX19 binds PMP mPTS post-translationally → PEX19·PMP complex docks to PEX3 on peroxisomal membrane → PMP inserts into lipid bilayer → PEX19 recycles. PEX19 is the effector arm; PEX3 is the docking receptor",
            "COMPLETES PEX3-PEX16-PEX19 axis: PEX16 (ER→peroxisome, recruits PEX3) → PEX3 (peroxisomal membrane receptor for PEX19) → PEX19 (cytoplasmic chaperone, docks to PEX3, inserts ALL class I PMPs). All three must be functional for PMP insertion to occur",
            "PEX19 LOF → ALL class I PMPs fail to insert: includes ALDP/ABCD1, PMP70, PEX14 (translocon docking), PEX26 (PEX1/PEX6 anchor), PEX11β, ALDH3A2, PMP34 — ALL absent from membrane → ghost peroxisomes (lipid bilayer only) → matrix import fails → ALL peroxisomal metabolic pathways simultaneously impaired",
            "Farnesylation is critical for PEX19 function: CAAX motif Cys296-Ser-Ile-Met undergoes farnesylation by farnesyltransferase; farnesyl group required for membrane affinity and PMP binding efficiency; p.Cys296Ser abolishes farnesylation → complete null phenotype; p.Ala318Val reduces efficiency → partial function → NALD/IRD",
            "Biochemical identity: PEX19-ZSD is biochemically IDENTICAL to all other PBD-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes PEX19 from PEX1/PEX6/PEX3/PEX16/PEX2/PEX10/PEX12/PEX26-ZSD",
            "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX19; NORMAL in ABCD1 (X-ALD); single test separates diagnoses; PHT/CBZ can be used in ABCD1 (adrenal crisis ABSOLUTE CI) but RELATIVE CI in ZSD",
            "DHA synthesis failure: DHA synthesised peroxisomally; PEX19 deficiency → membrane absent → DHA synthesis fails → DHA low → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + RP + peripheral neuropathy",
            "Phenotypic spectrum: ZS 35% (null/null — fatal <12M) · NALD 45% · IRD 15% · Atypical 5%; no founder hypomorphic allele → spectrum more severe than PEX16 (NALD modal 50%); ZS fraction (35%) equals PEX3",
            "p.Ala318Val (near CAAX site): reduces farnesylation efficiency ~50–70% → partial PEX19 membrane affinity → partial PMP insertion → partial peroxisomal function; enables NALD/IRD phenotype as compound het with null allele",
            "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A). Near-absolute CI in ZS/NALD",
            "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD; progresses IRD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH Level A preferred for IS; AVOID VGB in all ZSD forms",
            "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → acute neurotoxicity + seizures; IV dextrose MANDATORY during any NPO or pre-surgical fast; PT/INR + IV Vitamin K (cholestatic coagulopathy ZS/NALD)",
            "Lorenzo's Oil INEFFECTIVE: peroxisomal membrane absent — PEX19 LOF prevents ALL PMP insertion; peroxisomal β-oxidation cannot be restored by dietary substrate manipulation when the organelle itself has no functional membrane",
            "HSCT NOT INDICATED: ZSD not inflammatory (no adaptive immune component); donor microglia cannot restore PEX19 chaperone function or PMP insertion; contrast with ABCD1 where microglial replacement partially helpful",
            "No ERT: PEX19 is a farnesylated cytoplasmic chaperone — not a soluble lysosomal enzyme; protein replacement therapy not applicable; farnesylation in cytoplasm requires intracellular delivery which is not yet achievable",
            "~15–30 cases worldwide 2026; COMPLETES the PEX3-PEX16-PEX19 early membrane biogenesis axis — the foundational triad that establishes the peroxisomal membrane identity before all other biogenesis steps",
        ],
        "standards": [
            "Matsuzono Y et al. Human PEX19: cDNA cloning by functional complementation, mutation analysis in a patient with Zellweger syndrome, and potential role in peroxisomal membrane assembly. Proc Natl Acad Sci USA. 1999;96:2116–2121",
            "Fransen M et al. Analysis of human Pex19p's domain structure by pentapeptide scanning mutagenesis. J Mol Biol. 2001;305:1193–1208",
            "Fang Y et al. PEX3 functions as a PEX19 docking factor in the import of class I peroxisomal membrane proteins. J Cell Biol. 2004;164:863–875",
            "Jones JM et al. Multiple distinct targeting signals in integral peroxisomal membrane proteins. J Cell Biol. 2004;153:1141–1149",
            "Sacksteder KA et al. PEX19 binds multiple peroxisomal membrane proteins, is predominantly cytoplasmic, and is required for peroxisome membrane synthesis. J Cell Biol. 2000;149:373–384",
            "Fujiki Y et al. New insights into peroxisomal protein targeting pathways — farnesylation of PEX19 and PMP docking. Annu Rev Cell Dev Biol. 2020",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012;1822:1430–1441",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016;117:313–321",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614882 (PBD11A-ZSD — PEX19); #614883 (PBD11B-NALD — PEX19); *600279 (PEX19 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(19192)  # PEX19 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 35, "n": 14,
            "sex": "7M/7F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + myoclonic; EEG burst-suppression; DRE universal",
            "eeg": "Burst-suppression → multifocal spike-wave; bilateral asynchronous; EEG NICU monitoring mandatory; poor prognosis",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination; enlarged ventricles",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Leu78fs/p.Leu78fs or p.Leu78fs/p.Cys296Ser or frameshift/splice); PEX19 absent; no PMP docking possible; ghost peroxisomes; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Ala318Val or splice",
            "pct": 45, "n": 18,
            "sex": "9M/9F",
            "onset_age": "Infantile (1st–12th month)",
            "seizure_risk": "80–90% — infantile spasms (IS) 35–55% + hypsarrhythmia; focal + generalised + myoclonic",
            "eeg": "Hypsarrhythmia (IS period); multifocal spike-wave; modified hypsarrhythmia with burst-suppression elements",
            "mri": "Periventricular white matter signal; delayed myelination; cerebellar hypoplasia; milder gyral pattern than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Ala318Val (partial farnesylation ~30–50% efficiency) or null/splice hypomorphic; partial PEX19 docking → partial PMP insertion → partial peroxisomal function; survival months to years",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Ala318Val/mild",
            "pct": 15, "n": 6,
            "sex": "3M/3F",
            "onset_age": "Childhood / Adolescence (2–18 yr)",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD; phytol-restricted diet reduces phytanic burden",
            "eeg": "Focal epileptiform discharges; normal background early; generalised spike-wave with progression",
            "mri": "RP-associated retinal thinning; mild cerebellar atrophy; white matter changes mild; no pachygyria",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Ala318Val/hypomorphic compound het; ~30–50% residual PEX19 farnesylation + docking → partial PMP insertion → partial peroxisomal function; progressive ataxia + RP; adult survival possible",
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
            "variant_detail": "Both alleles hypomorphic (near-CAAX missense with substantial residual farnesylation); late-onset slowly progressive ataxia + RP + neuropathy",
        },
    ]

    def rand_sex():
        return random.choice(["M", "F"])

    def rand_geno(phenotype):
        if "ZS" in phenotype:
            variants = [
                "p.Leu78fs/p.Leu78fs",
                "p.Leu78fs/p.Cys296Ser",
                "p.Leu78fs/c.234del",
                "p.Leu78fs/p.Arg139*",
            ]
        elif "NALD" in phenotype:
            variants = [
                "p.Leu78fs/p.Ala318Val",
                "p.Cys296Ser/p.Ala318Val",
                "p.Leu78fs/c.741-1G>T",
                "p.Arg139*/p.Ala318Val",
            ]
        elif "IRD" in phenotype:
            variants = [
                "p.Ala318Val/p.Thr263Met",
                "p.Ala318Val/c.741-1G>T",
                "p.Ala318Val/p.Leu295Pro",
                "p.Ala318Val/p.Pro287Ser",
            ]
        else:
            variants = ["p.Ala318Val/p.Ile291Thr", "p.Ala318Val/p.Val283Leu"]
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
                "patient_id": f"PEX19-{pid:03d}",
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
        {
            "type": "Neonatal multifocal clonic (ZS)",
            "pct": 35,
            "eeg": "Burst-suppression → multifocal spike-wave; bilateral asynchronous; EEG NICU mandatory; universal in ZS; DRE 100%",
        },
        {
            "type": "Infantile spasms / West syndrome (NALD)",
            "pct": 45,
            "eeg": "Hypsarrhythmia → modified hypsarrhythmia; ACTH Level A first-line; VGB HIGH RISK (avoid — ZSD retinopathy additive blindness)",
        },
        {
            "type": "Focal seizures bilateral spread (NALD/IRD)",
            "pct": 22,
            "eeg": "Temporal/parietal focal spike-wave; post-ictal slowing; LEV first-line; LTG second-line IRD",
        },
        {
            "type": "Myoclonic seizures (NALD/IRD)",
            "pct": 18,
            "eeg": "Generalised polyspike-wave ≥3 Hz; CBZ/PHT RELATIVE CI (myoclonic exacerbation risk + CYP3A4 → DHA depletion)",
        },
        {
            "type": "GTCS (IRD/Atypical)",
            "pct": 12,
            "eeg": "Generalised spike-wave; good post-ictal recovery in IRD; LEV + LTG effective combination",
        },
        {
            "type": "Absence seizures (Atypical/late IRD)",
            "pct": 5,
            "eeg": "3 Hz generalised spike-wave; CLB adjunct; LTG Level C effective",
        },
    ]

    triggers = [
        {
            "trigger": "Febrile illness / intercurrent infection",
            "pct": 62,
            "note": "Most common trigger across all ZSD forms; fever reduces seizure threshold; aggressive antipyretics; IV LEV available for SE; ZS patients require ICU-level management",
        },
        {
            "trigger": "FASTING / NPO / Anaesthesia — EXTREME HAZARD",
            "pct": 55,
            "note": "EXTREME HAZARD: starvation → adipose lipolysis → free phytanic acid release → acute neurotoxicity + seizure cascade. IV dextrose MANDATORY pre-op. Pre-op bloods: PT/INR + LFT + platelet count + IV Vitamin K (cholestatic coagulopathy ZS/NALD)",
        },
        {
            "trigger": "Sleep deprivation",
            "pct": 38,
            "note": "Lowers seizure threshold universally; ZS/NALD infants require night-time monitoring; apnoea monitoring mandatory ZS (brainstem involvement)",
        },
        {
            "trigger": "Missed AED dose (LEV/CLB)",
            "pct": 30,
            "note": "LEV missed dose → rebound seizure risk within 24–48h especially NALD; caregivers require written protocol for NPO/vomiting scenarios",
        },
        {
            "trigger": "Metabolic decompensation (phytanic surge)",
            "pct": 28,
            "note": "Intercurrent illness + catabolism → acute phytanic acid surge from adipose → acute encephalopathy + SE; IV dextrose + carnitine supplementation; phytol-restricted diet reduces baseline load",
        },
        {
            "trigger": "Physiological stress (surgery, trauma)",
            "pct": 22,
            "note": "Surgery → catabolism → phytanic surge; IV dextrose protocol mandatory peri-operatively; avoid long NPO fasting; check PT/INR (cholestatic coagulopathy) + IV Vitamin K before any invasive procedure",
        },
        {
            "trigger": "Phytol-rich dietary indiscretion (IRD)",
            "pct": 15,
            "note": "IRD/Atypical: dairy fat (butter, cheese, full-fat milk), ruminant fats, certain fish → phytanic load increase → triggering within 48–72h; phytol-restricted diet Level B IRD",
        },
        {
            "trigger": "Hypoglycaemia",
            "pct": 12,
            "note": "Liver disease (ZS/NALD) impairs gluconeogenesis → hypoglycaemia → seizures; glucose monitoring mandatory especially post-prandial in ZS/NALD; IV dextrose protocol",
        },
    ]

    monitoring = [
        "VLCFA panel (C26:0, C24:0, C22:0, C26:0/C22:0 ratio) — baseline + every 6 months (ZS/NALD) or 12 months (IRD)",
        "Erythrocyte plasmalogens (RBC) — baseline; repeat if clinical change (remains low in ZSD, confirms diagnosis vs ABCD1)",
        "DHA level (plasma/RBC) — baseline + every 6 months; supplement if low (NALD/IRD — Level B evidence)",
        "Phytanic acid — baseline + every 3–6 months IRD (dietary monitoring); every 6 months NALD",
        "Pristanic acid — baseline; confirms alpha-oxidation defect",
        "Pipecolic acid (plasma + urine) — baseline; confirms ZSD diagnosis",
        "LFTs (ALT, AST, GGT, bilirubin, bile acids) — monthly ZS; every 3 months NALD; every 6 months IRD",
        "PT/INR + vitamin K — every 3 months ZS/NALD (cholestatic coagulopathy); before any procedure",
        "Ophthalmology (ERG + fundus + VF) — every 6 months all forms; annual IRD/Atypical; VGB absolutely contraindicated",
        "Audiology (BAER/ABR) — baseline + every 12 months (sensorineural hearing loss ZS/NALD)",
        "Brain MRI (T1/T2/DWI/spectroscopy) — baseline; repeat at 6M if ZS/NALD; annual MRS for GHB/DHA signal",
        "EEG (interictal + ictal) — monthly NICU ZS; every 3 months NALD; as indicated IRD",
        "Adrenal function (AM cortisol + ACTH stimulation) — baseline all forms; note: adrenal insufficiency LESS common than ABCD1 but reported in ZS/NALD",
        "Developmental assessment (Bayley scales / Griffiths) — every 6 months NALD/IRD; Vineland adaptive behaviour",
    ]

    thresholds = [
        {
            "parameter": "C26:0 plasma",
            "threshold": ">1.3 μg/mL",
            "action": "Confirmed VLCFA elevation — initiate ZSD diagnostic cascade (RBC plasmalogens + DHA + phytanic + pipecolic + molecular)",
        },
        {
            "parameter": "Erythrocyte plasmalogens (C16:0-DMA)",
            "threshold": "<20% of normal",
            "action": "Confirms ZSD (not ABCD1 — ABCD1 plasmalogens NORMAL); trigger NGS PBD panel; start dietary counselling",
        },
        {
            "parameter": "DHA (plasma)",
            "threshold": "<2.5% of total fatty acids",
            "action": "Initiate DHA supplementation (Level B — NALD/IRD); LPC-DHA oral preferred (bypasses peroxisomal synthesis)",
        },
        {
            "parameter": "Phytanic acid",
            "threshold": ">10 μg/mL",
            "action": "Initiate phytol-restricted diet (IRD — Level B); avoid fasting — IV dextrose protocol; review dietary diary",
        },
        {
            "parameter": "ALT / AST",
            "threshold": ">3× ULN",
            "action": "VPA absolute CI; reinforce phytol diet + fasting avoidance; consider UDCA (Level C) for cholestasis; GI / hepatology referral",
        },
        {
            "parameter": "PT/INR",
            "threshold": ">1.5",
            "action": "IV Vitamin K before any procedure; FFP if active bleed; avoid aspirin/NSAIDs; hepatology input",
        },
        {
            "parameter": "ERG (photopic/scotopic amplitude)",
            "threshold": "<50% baseline reduction",
            "action": "VGB contraindicated (additive retinopathy); ophthalmology urgent; DHA supplementation review; VF mapping",
        },
        {
            "parameter": "Seizure frequency",
            "threshold": "≥1 per week (IS) or ≥3 per month (focal)",
            "action": "ACTH (IS — Level A, preferred over VGB); LEV dose optimisation; add CLB (Level B); neurogenetics review for updated NGS",
        },
    ]

    lifecycle = [
        {
            "stage": "Stage 1 — Neonatal / Fetal (0–4 weeks, ZS/severe NALD)",
            "features": "Prenatal: VLCFA elevated on amniocentesis / CVS; dysmorphic facies at birth; hypotonia; neonatal seizures day 1–7; EEG burst-suppression; hepatomegaly; coagulopathy; NBS positive (C26:0-lyso-PC DBS)",
            "action": "NICU admission; IV LEV seizure control; avoid VPA/VGB; IV dextrose protocol; Vitamin K IV; ophthalmology within 48h; confirm with plasma VLCFA + plasmalogens + molecular; palliative care pathway ZS",
        },
        {
            "stage": "Stage 2 — Infantile (1–12 months, NALD/ZS survivors)",
            "features": "Infantile spasms onset; hypsarrhythmia; hypotonia progression; feeding difficulties; failure to thrive; liver disease progresses; auditory brainstem response abnormal; early DHA deficiency retinopathy",
            "action": "ACTH (IS — Level A); LEV adjunct; avoid VGB; DHA supplementation (NALD — Level B); phytol-restricted formula; monthly LFTs; BAER; ophthalmology ERG",
        },
        {
            "stage": "Stage 3 — Childhood (1–10 yr, NALD/IRD)",
            "features": "Seizure type evolution: IS → focal ± bilateral spread; developmental plateau; neurosensory deterioration (hearing + vision); phytol-restriction becomes critical; ataxia emergence IRD",
            "action": "LEV ± CLB ± LTG; optimise DHA; phytol-restricted diet; annual ophthalmology + audiology; 6-monthly MRI; neuropsychology; physiotherapy",
        },
        {
            "stage": "Stage 4 — Adolescence (10–18 yr, IRD/Atypical)",
            "features": "Stable epilepsy (focal/GTCS, well-controlled); progressive ataxia + RP; sensorineural hearing loss; peripheral neuropathy; reproductive counselling (AR — 25% sibling risk)",
            "action": "LEV ± LTG maintenance; phytol-restricted diet; DHA ongoing; audiological aids; orthopedics for ataxia; genetic counselling; employment/education planning",
        },
        {
            "stage": "Stage 5 — Adult (18–50 yr, IRD/Atypical)",
            "features": "Chronic stable epilepsy; slowly progressive ataxia + retinitis pigmentosa; peripheral neuropathy; osteoporosis (bile acid malabsorption); career + independence affected",
            "action": "Long-term AED continuation; annual biochemical panel; ophthalmology + audiology; bone density DEXA; UDCA Level C for cholestasis; independent living assessment",
        },
        {
            "stage": "Stage 6 — End-of-Life / Palliative (ZS: <12M; NALD: 1–5 yr)",
            "features": "Progressive neurological deterioration; apnoea; refractory seizures; hepatic failure; comfort care; family support; genetic counselling for future pregnancies",
            "action": "Palliative care team; symptom control (LEV + midazolam PR for SE); family conference; prenatal diagnosis options for future pregnancies (amniocentesis / CVS + molecular)",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "First-line AED — ALL ZSD forms (ZS, NALD, IRD, Atypical)",
            "evidence": "Level A (expert consensus + observational) — no hepatotoxicity, no CYP induction, IV available for SE/NPO",
            "dose": "20–40 mg/kg/day PO or IV (neonates: 30–50 mg/kg/day; NICU: IV infusion); titrate to seizure freedom",
            "moa": "SV2A vesicular synaptic protein modulator — reduces neurotransmitter release; no CYP interaction; renal excretion",
            "monitoring": "Renal function (eGFR); behaviour (irritability in neonates — ZS high-dose); therapeutic drug monitoring optional",
            "ci": "None relevant in ZSD. No hepatotoxicity. IV form preferred peri-operatively / NPO.",
        },
        {
            "drug": "ACTH (Adrenocorticotropic Hormone)",
            "class": "Level A — Infantile Spasms (IS) in NALD/ZS survivors",
            "evidence": "Level A for IS — more evidence than VGB; preferred in ZSD due to ZSD retinopathy (VGB additive blindness risk eliminated)",
            "dose": "Synthetic ACTH (Synacthen): 0.5–1 mg/day IM × 2 weeks → taper; or natural ACTH 40–80 IU/day × 4 weeks → taper per local protocol",
            "moa": "MC2R activation → adrenal cortisol + neuroactive steroids; central anticonvulsant effect on GABA/glutamate; reduces hypsarrhythmia",
            "monitoring": "BP (hypertension common); electrolytes (hypokalaemia); blood glucose; infection screen before/during; ophthalmology (separate from VGB CI); growth",
            "ci": "Active infection (must screen and treat first); uncontrolled hypertension. Adrenal crisis risk lower in ZSD than ABCD1 (use corticosteroid replacement during taper).",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "Level B adjunct — NALD/IRD (focal/myoclonic/IS adjunct)",
            "evidence": "Level B — low hepatotoxicity, no CYP3A4 induction (CYP2C19 only), useful for focal + myoclonic adjunct",
            "dose": "0.1–0.3 mg/kg/day (pediatric); titrate to max 1 mg/kg/day; twice-daily dosing",
            "moa": "GABA-A positive allosteric modulator (benzodiazepine class); 1,5-benzodiazepine with lower tolerance risk than classical 1,4-BZDs",
            "monitoring": "Sedation; tolerance development (dose escalation); CYP2C19 phenotyping (PM: higher CLB levels); liver enzymes (minimal effect)",
            "ci": "Avoid in severe hepatic impairment (ZS/NALD with active liver failure). Monitor sedation in combination with LEV in neonates.",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "Level C — IRD/Atypical (focal + GTCS)",
            "evidence": "Level C — favourable hepatic profile (glucuronidation, not oxidation); effective focal + GTCS in IRD/Atypical forms",
            "dose": "0.15 mg/kg/day × 2 weeks → 0.3 mg/kg/day × 2 weeks → target 1–5 mg/kg/day; SLOW titration to minimise rash",
            "moa": "Voltage-gated sodium channel blockade; glutamate release reduction; also HCN channel modulation",
            "monitoring": "Rash (SJS risk — 1:1000 pediatric with rapid titration); FBC; LFTs (minimal hepatic); LTG levels if comedication with valproate (prohibited in ZSD — do not co-prescribe)",
            "ci": "VPA co-administration PROHIBITED (VPA doubles LTG levels — toxic AND VPA is HIGH RISK in ZSD). Rapid titration → SJS. Use caution in severe hepatic impairment.",
        },
        {
            "drug": "DHA (Docosahexaenoic Acid) supplementation",
            "class": "Level B disease-modifying — NALD/IRD (bypasses peroxisomal DHA synthesis)",
            "evidence": "Level B — improves retinal and neurological function in NALD/IRD; LPC-DHA preferred (intestinal absorption bypasses peroxisomal synthesis defect)",
            "dose": "LPC-DHA (sn-2 DHA-lysophosphatidylcholine): 100–200 mg/day; or conventional DHA (ethyl ester): 100–500 mg/day; adjust by plasma DHA level",
            "moa": "Bypasses peroxisomal DHA biosynthesis deficit (peroxisomally synthesised DHA absent in PEX19 LOF); LPC-DHA absorbed directly via intestine → systemic delivery without peroxisomal step",
            "monitoring": "Plasma DHA (target >4% of total FA); RBC DHA; visual function (ERG) — may show improvement in IRD; liver enzymes (safe)",
            "ci": "None relevant. Fish-derived DHA only (algal OK). Avoid excess omega-6 competition (arachidonic acid balance).",
        },
        {
            "drug": "Phytol-restricted diet",
            "class": "Level B dietary disease-modification — IRD/Atypical (reduces phytanic acid load)",
            "evidence": "Level B — reduces phytanic acid accumulation in IRD/Atypical; slows neurological deterioration; reduces seizure trigger burden",
            "dose": "Restrict phytol-rich foods: ruminant dairy (cheese, butter, full-fat milk), ruminant fat, certain fish (tuna), spinach, kale; dietitian referral essential; maintain calorie/protein adequacy",
            "moa": "Phytol → phytanic acid (via alpha-oxidation defect in ZSD → phytanic accumulates); restricting dietary phytol reduces substrate for phytanic accumulation → lowers neurotoxic burden",
            "monitoring": "Phytanic acid (plasma) — target <10 μg/mL; dietary diary review monthly; nutritional assessment (Vitamin A, D, K, E monitoring — fat-soluble vitamins); weight and growth",
            "ci": "Phytol-restriction insufficient in ZS/severe NALD (too severe — liver disease + poor tolerance); primarily IRD/Atypical. Fasting EXTREME HAZARD (adipose phytanic release).",
        },
        {
            "drug": "Ursodeoxycholic acid (UDCA) + Vitamin K",
            "class": "Level C supportive — cholestasis management (ZS/NALD)",
            "evidence": "Level C observational — UDCA improves bile flow and reduces hepatic bile acid toxicity in cholestatic ZSD; IV Vitamin K mandatory for coagulopathy",
            "dose": "UDCA: 10–20 mg/kg/day in 2–3 divided doses; IV Vitamin K 1–2 mg/day prophylactic; Vitamin K1 5–10 mg IV before procedures",
            "moa": "UDCA replaces toxic bile acids; choleretic + hepatoprotective; Vitamin K provides cofactor for coagulation factors II, VII, IX, X (impaired in cholestasis)",
            "monitoring": "LFTs; bilirubin; bile acids; PT/INR (target <1.5); platelet count; nutritional assessment",
            "ci": "UDCA generally well tolerated in ZSD; avoid large dose in acute hepatic decompensation. Vitamin K: rapid IV bolus may cause flushing — infuse slowly.",
        },
    ]

    contraindications = [
        {
            "drug": "Valproic Acid (VPA / Valproate)",
            "level": "HIGH RISK — THREE mechanisms (near-Absolute CI in ZS/NALD)",
            "reason": (
                "THREE independent toxicity mechanisms in ZSD: "
                "(1) HEPATOTOXICITY — baseline cholestatic liver disease in ZS/NALD makes VPA-induced hepatotoxicity risk catastrophic; "
                "(2) PEROXISOMAL BETA-OXIDATION INHIBITION — VPA metabolites directly inhibit peroxisomal beta-oxidation → VLCFA accumulation worsens; "
                "(3) CARNITINE DEPLETION — VPA sequesters CoA → secondary carnitine deficiency in already metabolically fragile patients. "
                "POLG1 MANDATORY screening (CPIC Grade A) before any VPA use."
            ),
            "alternative": "LEV first-line (all forms). ACTH Level A (IS). CLB Level B adjunct. LTG Level C (IRD). NEVER use VPA in ZS/NALD.",
        },
        {
            "drug": "Vigabatrin (VGB)",
            "level": "HIGH RISK — Additive irreversible retinopathy",
            "reason": (
                "ZSD causes retinopathy (RP) universally in ZS/NALD (100%) and in IRD (progressive); "
                "VGB irreversible visual field (VF) constriction in >50% of patients on long-term therapy; "
                "combined ZSD retinopathy + VGB VF constriction = catastrophic additive irreversible blindness. "
                "AVOID in ALL ZSD forms including IRD. ACTH (Level A) is the standard for IS in ZSD."
            ),
            "alternative": "ACTH Level A (IS — preferred over VGB due to ZSD retinopathy). LEV + CLB for non-IS seizures. NEVER use VGB when ZSD retinopathy present.",
        },
        {
            "drug": "Fasting / NPO / Pre-operative starvation",
            "level": "EXTREME HAZARD — Phytanic acid surge",
            "reason": (
                "Starvation → adipose lipolysis → free phytanic acid release into plasma → acute phytanic neurotoxicity → "
                "seizure cascade, cardiac arrhythmia, cranial nerve palsies. "
                "IV dextrose (10% dextrose) MANDATORY during any pre-surgical fast, inter-current illness with poor intake, "
                "or NPO period. Pre-op assessment: PT/INR + LFT + platelets + IV Vitamin K (cholestatic coagulopathy). "
                "Anaesthesia team MUST be briefed on PEX19-ZSD protocol."
            ),
            "alternative": "IV 10% dextrose (minimum 4–6 mg/kg/min glucose infusion rate) during NPO; nasogastric feeds if poor oral intake; emergency card with dextrose protocol",
        },
        {
            "drug": "Phenytoin (PHT) / Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
            "level": "RELATIVE CI — CYP3A4 induction + hepatic burden",
            "reason": (
                "CYP3A4 enzyme induction → accelerated DHA catabolism → DHA depletion (worsens neurological and retinal outcomes); "
                "hepatic microsomal burden in already-compromised cholestatic liver (ZS/NALD); "
                "CBZ/OXC myoclonic exacerbation risk. "
                "CRITICAL DISTINCTION from ABCD1: PHT/CBZ ABSOLUTE CI in ABCD1 (adrenal crisis — adrenal suppression); "
                "in ZSD PHT/CBZ are RELATIVE CI (no adrenal suppression mechanism — ZSD adrenal insufficiency uncommon). "
                "Use IV LEV for ZSD status epilepticus instead."
            ),
            "alternative": "IV LEV (SE/status — no hepatic burden, no CYP induction, IV available). LTG (Level C, IRD). CLB (Level B).",
        },
        {
            "drug": "Lorenzo's Oil (erucic acid + oleic acid 4:1)",
            "level": "INEFFECTIVE — Membrane absent, mechanism mismatch",
            "reason": (
                "Lorenzo's Oil works by competitive inhibition of fatty acid elongase (ELOVL1) to reduce VLCFA synthesis — "
                "requires intact peroxisomal membrane to have any metabolic benefit. "
                "PEX19 LOF → NO PMPs insert → NO functional peroxisomal membrane → NO peroxisomal β-oxidation possible → "
                "reducing VLCFA synthesis does not restore absent β-oxidation pathway. "
                "Multiple clinical trials: no benefit in ZSD (mechanism mismatch — developed for ABCD1/AMN where peroxisomal membrane intact). "
                "Do not prescribe in PEX19-ZSD."
            ),
            "alternative": "DHA supplementation (Level B — NALD/IRD). Phytol-restricted diet (Level B — IRD). Symptom management. No disease-modifying equivalent to Lorenzo's Oil exists for ZSD.",
        },
        {
            "drug": "HSCT (Haematopoietic Stem Cell Transplant)",
            "level": "NOT INDICATED — Not inflammatory (mechanism mismatch)",
            "reason": (
                "HSCT partially benefits ABCD1/AMN (X-ALD) because X-ALD has a cerebral inflammatory component — "
                "donor-derived microglia replace pro-inflammatory host microglia and slow demyelination. "
                "ZSD (PEX19-ZSD) is NOT an inflammatory demyelinating disease — it is a peroxisomal biogenesis failure. "
                "Donor microglia CANNOT restore PEX19 chaperone function, PEX3/PEX16/PEX19 axis, or PMP insertion. "
                "HSCT in ZSD provides no known benefit and carries transplant-related mortality. "
                "Do not refer for HSCT in any form of PBD-ZSD."
            ),
            "alternative": "Supportive care. DHA Level B. Phytol-restricted diet Level B. Palliative pathway (ZS/severe NALD).",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE — Cytoplasmic chaperone, not lysosomal enzyme",
            "reason": (
                "ERT is applicable to lysosomal storage disorders (e.g., Gaucher: imiglucerase; Fabry: agalsidase; Pompe: alglucosidase) — "
                "recombinant enzyme delivered intravenously is endocytosed via mannose-6-phosphate receptor to lysosomes. "
                "PEX19 is a FARNESYLATED CYTOPLASMIC CHAPERONE — no lysosomal targeting, no M6P receptor pathway, "
                "no secretory route. PEX19 recombinant protein cannot be delivered to the cytoplasm via IV infusion. "
                "Gene therapy (AAV-PEX19) is under research but not clinically available 2026."
            ),
            "alternative": "No ERT equivalent. Research: AAV gene therapy (preclinical). Supportive: DHA, diet, AEDs, palliative.",
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
    return {
        "key_concepts": [
            "PEX19 (299 aa, 1q22): farnesylated cytoplasmic PMP chaperone; CAAX motif at C-terminus (Cys296-Ser-Ile-Met) farnesylated by farnesyltransferase; N-terminal PEX3-binding helix (aa 14–30); C-terminal mPTS-recognition domain (aa 240–299)",
            "Farnesylation of PEX19 is ESSENTIAL for function: CAAX farnesylation provides membrane affinity and optimal PMP-binding geometry; p.Cys296Ser = complete LOF (null phenotype); p.Ala318Val near CAAX reduces farnesylation efficiency ~50–70% → partial function → NALD/IRD",
            "Class I PMP insertion pathway (PEX19-mediated): (1) PEX19 binds PMP mPTS post-translationally in cytoplasm → (2) PEX19·PMP complex travels to peroxisomal membrane → (3) PEX19 N-terminal helix docks onto PEX3 cytoplasmic domain → (4) PMP inserts into lipid bilayer → (5) PEX19 recycled to cytoplasm",
            "COMPLETES PEX3-PEX16-PEX19 axis: PEX16 (ER-derived vesicle delivery, recruits PEX3 to peroxisomal membrane) → PEX3 (type I integral PMP, docking receptor for PEX19) → PEX19 (cytoplasmic chaperone, effector arm, inserts ALL class I PMPs). All three required; each LOF produces ghost peroxisomes.",
            "Ghost peroxisomes in PEX19 LOF: lipid bilayer intact (derived from ER), but NO peroxisomal membrane proteins (PMPs) — no ALDP, no PMP70, no PEX14, no PEX26, no PEX11β, no PEX2/10/12 RING complex. IF: PEX14 (matrix translocon marker) absent from peroxisomal ghosts; PEX3 normal localization (PEX3 is delivered by PEX16 from ER independently of PEX19 — this is the key IF distinction from PEX3/PEX16 LOF)",
            "IF clue distinguishing PEX19 LOF from PEX3/PEX16 LOF: in PEX19 LOF, PEX3 is correctly localized to peroxisomal ghosts (PEX3 delivery via PEX16 is PEX19-independent — PEX3/PEX16 form their own axis); but PEX14 (matrix translocon docking protein — a class I PMP requiring PEX19 for insertion) is absent. In PEX3 LOF or PEX16 LOF, PEX3 itself is absent/mislocalized.",
            "Biochemical identity — ALL ZSD identical on plasma panel: C26:0 ↑, C24:0 ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑. ONLY NGS distinguishes PEX19-ZSD from PEX1/PEX6/PEX3/PEX16/PEX2/PEX10/PEX12/PEX26-ZSD",
            "Plasmalogens LOW — distinguishes ALL ZSD (including PEX19) from ABCD1: RBC plasmalogens severely reduced in PBD-ZSD; NORMAL in ABCD1 (X-ALD). Critical test: ABCD1 adrenal crisis requires ABSOLUTE CI for PHT/CBZ (adrenal suppression); ZSD PHT/CBZ only RELATIVE CI (CYP3A4/DHA depletion). Plasmalogens (RBC) single test separates these diagnoses.",
            "DHA low → migration defects (ZS MRI PATHOGNOMONIC): peroxisomally synthesised DHA absent in PEX19 LOF → pachygyria + polymicrogyria on MRI (ZS) + RP + peripheral neuropathy. DHA bypassed by LPC-DHA oral supplementation (intestinal route, no peroxisomal step required).",
            "Phenotypic spectrum — no founder allele: ZS 35% (null/null) · NALD 45% · IRD 15% · Atypical 5%. No common founder allele unlike PEX1 (G843D ~30% European) or PEX6 (R860W ~18% European). ~15–30 cases worldwide 2026 — among rarest PBD-ZSD. ZS fraction 35% equals PEX3; more severe than PEX16 NALD-modal (50% NALD).",
            "VPA triple mechanism in ZSD: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (metabolite inhibits peroxisomal enzyme; worsens VLCFA) + (3) carnitine depletion (CoA sequestration). POLG1 MANDATORY exclusion (CPIC Grade A). Near-absolute CI in ZS/NALD. NEVER use.",
            "VGB additive retinopathy — specific ZSD hazard: VGB irreversible VF constriction >50% patients + ZSD retinopathy (universal ZS/NALD) = catastrophic additive permanent blindness. ACTH Level A is the standard for IS in ZSD. AVOID VGB in ALL ZSD forms including mild IRD.",
            "Fasting EXTREME HAZARD: adipose lipolysis during starvation releases phytanic acid (stored from dietary phytol) → acute phytanic surge → neurotoxicity + cardiac arrhythmia + seizure cascade. IV 10% dextrose MANDATORY during any NPO, pre-operative fast, or illness with poor intake. Anaesthesia team briefing mandatory.",
            "Lorenzo's Oil INEFFECTIVE in ZSD: mechanism requires intact peroxisomal membrane for beta-oxidation; PEX19 LOF removes entire PMP-containing membrane → beta-oxidation impossible regardless of substrate. Developed for ABCD1/AMN (intact peroxisomal membrane, single transporter defect). Multiple trials confirm no ZSD benefit.",
            "No ERT in PEX19-ZSD: PEX19 is a farnesylated cytoplasmic chaperone — not a lysosomal enzyme; no M6P receptor pathway; cannot be delivered IV to cytoplasm. Contrast with lysosomal enzymes (Gaucher, Fabry, Pompe) where M6P-receptor endocytosis delivers recombinant enzyme to lysosomes.",
            "HSCT NOT indicated: ZSD not inflammatory; no microglial rescue mechanism (contrast ABCD1 where donor microglia slow neuroinflammation); transplant mortality unjustified. PEX19 is an ubiquitous cytoplasmic gene — haematopoietic chimerism does not rescue peroxisomal biogenesis in non-haematopoietic tissues.",
        ],
        "diagnostic_algorithm": [
            "Step 1 — Clinical suspicion: neonatal dysmorphic facies + hypotonia + seizures → ZSD top differential; OR infantile spasms + hypsarrhythmia + liver disease → ZSD NALD; OR childhood progressive RP + ataxia + sensorineural HL → ZSD IRD/Atypical",
            "Step 2 — VLCFA panel (plasma): C26:0, C24:0, C22:0, C26:0/C22:0 ratio. ELEVATED in ALL PBD-ZSD including PEX19-ZSD. Also elevated in ABCD1 (X-ALD); proceed to plasmalogens to distinguish.",
            "Step 3 — Erythrocyte plasmalogens (RBC): SEVERELY LOW in ALL PBD-ZSD including PEX19. NORMAL in ABCD1. LOW → confirms ZSD; NORMAL → ABCD1 workup (ABCD1 gene sequencing).",
            "Step 4 — Biochemical confirmation: DHA (low), phytanic acid (elevated), pristanic acid (elevated), pipecolic acid (elevated), DHCA+THCA (elevated bile acid intermediates). All abnormal in ZSD including PEX19-ZSD.",
            "Step 5 — Brain MRI (T1/T2/DWI): ZS — pachygyria + polymicrogyria (PATHOGNOMONIC) + periventricular germinolytic cysts + hypomyelination. NALD — white matter signal + delayed myelination + mild migration defects. IRD — mild cerebellar atrophy + RP signal.",
            "Step 6 — IF / electron microscopy of fibroblasts: GHOST PEROXISOMES (small, round, lipid-bilayer only vesicles by EM); PEX3 NORMALLY LOCALIZED (to ghost membrane — distinguishes PEX19 LOF from PEX3 or PEX16 LOF where PEX3 is absent/cytoplasmic); PEX14 ABSENT from peroxisomal membrane (PEX14 is a class I PMP requiring PEX19 for insertion).",
            "Step 7 — NGS panel (PBD gene panel): sequence all known PBD-ZSD genes (PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX11β, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26). Biallelic PEX19 pathogenic variants confirm PEX19-ZSD. If negative: extended peroxisomal gene panel.",
            "Step 8 — Confirm PEX19 pathogenic variants: null variants (frameshift, nonsense, splice-site) → ZS/NALD; near-CAAX missense (p.Ala318Val type) → NALD/IRD. p.Cys296Ser (abolishes farnesylation) = null.",
            "Step 9 — Ophthalmology (ERG + fundus): ZS — absent/severely reduced ERG; NALD — ERG reduction + RP onset; IRD — progressive RP well-documented. Confirms and grades retinopathy. VGB contraindication established.",
            "Step 10 — Initiate LEV + avoid VPA/VGB: start LEV; avoid VPA (triple mechanism); avoid VGB (additive retinopathy); ACTH if IS; DHA supplementation (NALD/IRD); IV dextrose protocol; phytol-restricted diet (IRD).",
            "Step 11 — Multi-disciplinary team: neurology + genetics + hepatology + ophthalmology + audiology + dietetics + neonatology (ZS) + palliative care (ZS/severe NALD).",
            "Step 12 — Family testing: siblings (25% risk, AR); molecular prenatal diagnosis offer (CVS at 11–13 wk or amniocentesis 15–16 wk); NBS programme enrolment if C26:0-lyso-PC DBS available locally.",
            "Step 13 — Research registry: enrol in PBD-ZSD international registry (EuroPBD / PBD-ZSD Consortium); consider AAV gene therapy trial eligibility (IRD/Atypical only — investigational 2026).",
        ],
        "pharmacological_distinctions": [
            "LEV vs VPA in ZSD: LEV first-line — no CYP induction, no hepatotoxicity, IV available. VPA HIGH RISK: THREE mechanisms (hepatotoxicity + peroxisomal BO inhibition + carnitine depletion). POLG1 MANDATORY (CPIC Grade A). Never use VPA in ZS/NALD.",
            "ACTH vs VGB for IS in ZSD: ACTH Level A PREFERRED — VGB produces irreversible VF constriction which is additive with ZSD retinopathy → catastrophic blindness. ACTH preferred for IS in ALL ZSD forms including less severe IRD.",
            "PHT/CBZ in ZSD vs ABCD1: PHT/CBZ ABSOLUTE CI in ABCD1 (adrenal insufficiency risk — adrenal gland infiltrated by cholesterol esters, adrenal stimulation during PHT/CBZ stress triggers crisis). PHT/CBZ RELATIVE CI in ZSD — CYP3A4 induction depletes DHA and adds hepatic burden, but NO adrenal crisis mechanism. IV LEV replaces in ZSD SE.",
            "Lorenzo's Oil: INEFFECTIVE in ZSD (no PMP membrane, no beta-oxidation possible) vs MODERATELY HELPFUL in ABCD1 early (reduces VLCFA synthesis; intact peroxisomal membrane allows some metabolic benefit). Do not prescribe Lorenzo's Oil in PEX19-ZSD.",
            "HSCT: BENEFICIAL (partial) in ABCD1 cerebral ALD (inflammatory demyelination — donor microglia replacement). NOT INDICATED in ZSD (not inflammatory; no donor-cell rescue mechanism for peroxisomal biogenesis).",
            "DHA supplementation in ZSD: Level B (NALD/IRD) — bypasses peroxisomal DHA synthesis via LPC-DHA oral route (intestinal absorption, no peroxisomal step). Standard DHA (fish oil ethyl ester) also effective. Not useful in ZS (too severe, absorptive capacity limited).",
            "Phytol-restricted diet: Level B (IRD/Atypical) — reduces dietary phytol → reduces phytanic acid substrate. NOT sufficient for ZS/NALD (peroxisomal alpha-oxidation completely absent; dietary restriction + fasting avoidance + IV dextrose more critical).",
            "Cholic acid / UDCA: Level C supportive (cholestasis ZS/NALD) — reduces bile acid synthesis burden; hepatoprotective. Vitamin K (IV) mandatory peri-procedurally (cholestatic coagulopathy).",
            "Fasting protocol — ZSD-specific: adipose phytanic release during any catabolic state. IV 10% dextrose mandatory. Pre-op: PT/INR + platelets + IV Vitamin K + LEV IV peri-op. Contrast with ABCD1: PHT/CBZ stress → adrenal crisis; in ZSD fasting → phytanic surge (different mechanism, same severity).",
            "PEX19 IF vs PEX3/PEX16 IF: PEX19 LOF → PEX3 NORMALLY LOCALIZED on ghost peroxisome membrane (PEX3 delivered by PEX16 independently of PEX19) but PEX14 (class I PMP) ABSENT. PEX3 LOF → PEX3 ABSENT from peroxisomal ghost + PEX19 cannot dock. PEX16 LOF → PEX3 CYTOPLASMIC/ER (mislocalized) + PEX19 cannot dock (because PEX3 not on membrane). IF differentiates the three upstream LOFs.",
            "Farnesylation of PEX19 — drug relevance: statins inhibit HMG-CoA reductase → reduce isoprenoid pool → reduce farnesyl pyrophosphate → could theoretically further impair PEX19 farnesylation in PEX19 hypomorphic patients. Pharmacological concern in near-CAAX missense carriers; statin use requires monitoring in PEX19-ZSD patients with IRD/Atypical phenotype.",
            "No ERT in PEX19-ZSD vs lysosomal disorders: ERT relies on M6P-receptor endocytosis to deliver recombinant lysosomal enzymes. PEX19 = cytoplasmic farnesylated chaperone; no lysosomal targeting, no M6P route. Cannot be administered IV. Gene therapy (AAV-PEX19) investigational — not clinical 2026.",
        ],
        "differential_diagnosis": [
            {
                "condition": "PEX1-ZSD / PEX6-ZSD (most common ZSD ~60% combined)",
                "distinction": "Biochemically IDENTICAL to PEX19-ZSD. PEX1 founder allele G843D (~30% European); PEX6 founder R860W (~18% European). Both are AAA-ATPase import motor subunits (PEX5 retrotranslocation). Only NGS distinguishes. PEX19 LOF affects PMP insertion (upstream), not import motor.",
            },
            {
                "condition": "PEX3-ZSD (type I integral PMP, PEX19 docking receptor)",
                "distinction": "Biochemically identical. IF: PEX3 ABSENT from peroxisomal ghost (vs PEX19 LOF where PEX3 is NORMALLY LOCALIZED on ghost membrane). PEX3 LOF prevents PEX19 docking; PEX19 LOF prevents PMP insertion downstream of PEX3. Key IF diagnostic clue: PEX3 localization on fibroblast IF.",
            },
            {
                "condition": "PEX16-ZSD (type II integral PMP, PEX3 recruitment factor, ER-to-peroxisome)",
                "distinction": "Biochemically identical. IF: PEX3 CYTOPLASMIC / ER (mislocalized) in PEX16 LOF (PEX16 recruits PEX3 from ER vesicles; without PEX16 PEX3 cannot reach membrane). In PEX19 LOF: PEX3 NORMALLY LOCALIZED on ghost. NALD is MODAL in PEX16 (50%) vs ZS modal in PEX19 (35%). IF PEX3 localization distinguishes PEX16 vs PEX19 LOF.",
            },
            {
                "condition": "PEX2 / PEX10 / PEX12-ZSD (RING E3 ubiquitin ligase triad)",
                "distinction": "Biochemically identical. RING triad ubiquitinates PEX5 for retrotranslocation (import receptor cycling). Ghost peroxisomes but with PMPs present (membrane intact) — matrix import defective. Contrast with PEX19-ZSD where ghost peroxisomes have NO PMPs (membrane biogenesis defect, not import cycling defect). IF: PMP70 PRESENT on peroxisomal membrane in PEX2/10/12-ZSD.",
            },
            {
                "condition": "ABCD1 / X-ALD (X-linked adrenoleukodystrophy)",
                "distinction": "VLCFA elevated (identical). Plasmalogens NORMAL (KEY — ZSD plasmalogens LOW). No liver disease. Adrenal insufficiency (absent in most ZSD). Inflammatory cerebral demyelination in cALD. PHT/CBZ ABSOLUTE CI (adrenal crisis) in ABCD1 vs RELATIVE CI in ZSD. HSCT beneficial in cALD vs NOT INDICATED in ZSD. Lorenzo's Oil moderately helpful in ABCD1 vs INEFFECTIVE in ZSD.",
            },
            {
                "condition": "PHYH (Refsum disease — phytanic alpha-oxidase deficiency)",
                "distinction": "Phytanic elevated (isolated — pristanic NORMAL, VLCFA NORMAL, plasmalogens NORMAL). AR. No VLCFA elevation, no liver disease in infancy. Onset adult or childhood. Phytol-restricted diet highly effective (intact peroxisomal membrane). Plasma exchange in acute crises. Distinct from ZSD by single-pathway defect.",
            },
            {
                "condition": "RCDP (Rhizomelic Chondrodysplasia Punctata — PEX7/GNPAT/AGPS)",
                "distinction": "Plasmalogens LOW (like ZSD). VLCFA NORMAL (unlike ZSD — VLCFA elevated in ZSD). Phytanic elevated only in PEX7-RCDP (alpha-oxidation also impaired via PTS2 targeting defect). Rhizomelic limb shortening + stippled epiphyses on X-ray — pathognomonic of RCDP. Distinct clinical presentation from ZSD.",
            },
            {
                "condition": "Mitochondrial disease / respiratory chain defect",
                "distinction": "Neonatal hypotonia + seizures + multi-organ failure can mimic ZS. Lactate elevated (mitochondrial); VLCFA NORMAL; plasmalogens NORMAL. Brain MRI: basal ganglia signal (Leigh syndrome) vs migration defects (ZS). CSF lactate/pyruvate ratio elevated. POLG1 mutation: POLG1 testing mandatory in ZSD before any VPA — overlap in phenotype.",
            },
        ],
        "standards": [
            "Matsuzono Y et al. Human PEX19: cDNA cloning by functional complementation, mutation analysis in a patient with Zellweger syndrome, and potential role in peroxisomal membrane assembly. Proc Natl Acad Sci USA. 1999;96:2116–2121",
            "Fransen M et al. Analysis of human Pex19p's domain structure by pentapeptide scanning mutagenesis. J Mol Biol. 2001;305:1193–1208",
            "Fang Y et al. PEX3 functions as a PEX19 docking factor in the import of class I peroxisomal membrane proteins. J Cell Biol. 2004;164:863–875",
            "Sacksteder KA et al. PEX19 binds multiple peroxisomal membrane proteins, is predominantly cytoplasmic, and is required for peroxisome membrane synthesis. J Cell Biol. 2000;149:373–384",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012;1822:1430–1441",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016;117:313–321",
            "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614882 (PBD11A-ZSD — PEX19); #614883 (PBD11B-NALD — PEX19); *600279 (PEX19 gene, 1q22)",
        ],
    }
