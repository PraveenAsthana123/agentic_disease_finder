#!/usr/bin/env python3
"""NDUFA4 — Leigh Syndrome Isolated Complex IV Deficiency (COXPD20 / MLRQ / ML-INSERT,
   a CIV subunit despite its NDUFA naming — the critical clinical DDx trap).

NDUFA4 (originally named "NADH:Ubiquinone Oxidoreductase Subunit A4", also known as
MLRQ and ML-INSERT) is an ~81-aa nuclear-encoded subunit of Complex IV (Cytochrome c
Oxidase / COX / CIV), ~9.3 kDa mature protein, located at 7p21.3 (OMIM *602137).

CRITICAL NAMING TRAP — NDUFA4 IS A CIV SUBUNIT, NOT CI:
  Despite the "NDUFA" (NADH:Ubiquinone Oxidoreductase) family designation, NDUFA4
  is a bona fide Complex IV (CIV) structural subunit. This reclassification was
  established by Balsa et al. (2012) Mol Cell and Pitceathly et al. (2013) Ann Neurol.
  NDUFA4 co-purifies with CIV (not CI) in blue-native PAGE; interacts directly with
  COX2 (MT-CO2); and its deficiency causes ISOLATED CIV deficiency — CI/CII/CIII
  activities are NORMAL. WES pipelines that flag NDUFA4 under "Complex I gene panel"
  create a diagnostic trap that can delay correct CIV diagnosis by months.

  NDUFA4 gene    OMIM *602137
  Disease        Leigh Syndrome — Isolated Complex IV Deficiency (COXPD20 / CIV-Leigh)
  Inheritance    Autosomal Recessive (AR) — biallelic pathogenic variants
  Chromosome     7p21.3
  Protein size   81 aa / 9.3 kDa (mature, after MTS cleavage)
  Topology       Single transmembrane helix; IMM-anchored; inter-membrane space (IMS) loop
                 contacts COX2 IMS domain; matrix-facing N-terminus within CIV assembly

PATHOPHYSIOLOGY (Complex IV / NDUFA4 / CIV-Leigh / COX2-Contact):
  NDUFA4 (MLRQ) is anchored via a single IMM-spanning TM helix. Its IMS-facing loop
  contacts COX2 (MT-CO2) — one of the three catalytic mtDNA-encoded CIV subunits
  responsible for accepting electrons from cytochrome c and channelling them to the
  binuclear CuA centre. NDUFA4 stabilises the CIV-CuA domain and the COX2-CuA
  electron entry interface. Loss of NDUFA4:
    → CIV assembly intermediate accumulates (COX2 sub-complex without NDUFA4)
    → CIV holocomplex absent or severely reduced on BN-PAGE (5–25% residual CIV)
    → CI / CII / CIII activities NORMAL — biochemical fingerprint of NDUFA4-CIV-Leigh
    → Cytochrome c cannot deliver electrons to terminal oxidase → mitochondrial membrane
      potential collapses → ATP synthase uncoupled → ETC failure at terminal step

  UNIQUE MOLECULAR SIGNATURE — CIV / COX2-CONTACT / SINGLE-TM / 7p21.3:
    NDUFA4 (MLRQ) is the ONLY member of the NDUFA naming family that is a CIV subunit.
    All other NDUFA-designated proteins (NDUFA1–NDUFA13) are genuine CI subunits.
    NDUFA4 co-purifies with CIV super-complex (CIV + CIII2 in respirasomes) and is
    absent from isolated CI preparations. Its single TM helix contrasts with the
    no-TM-helix peripheral scaffold of NDUFA3 (B9) and NDUFA7 (B14.5a), and with
    the multi-TM architecture of NDUFA11 (B14.7, 4 TM). The IMS loop interacts with
    COX2-CuA — a critical copper electron-entry electron-relay interface.

  CIV STRUCTURAL POSITION — COX2 IMS DOMAIN CONTACT:
    CIV (COX, cytochrome c oxidase) contains 14 subunits (COX1/COX2/COX3 mtDNA-encoded
    + 11 nuclear-encoded). NDUFA4 is a "supernumerary" CIV subunit stabilising the
    COX2-CuA interface at the IMS face. The COX2-CuA centre accepts electrons from
    ferrocytochrome c → CuA → haem a → haem a3-CuB → O2 reduction. NDUFA4 loss
    destabilises the COX2-CuA domain → impaired cytochrome c docking → CIV electron
    transfer rate reduced → CIV holoenzyme assembly failure on BN-PAGE.

CRITICAL NAMING & BIOCHEMICAL TRAP — NDUFA4 vs ALL OTHER NDUFA GENES:
  NDUFA1 (MWFE, Xq24, CI):       CI subunit, PP-module, X-linked, isolated CI
  NDUFA2 (B8, 5q31.3, CI):       CI subunit, N-module, AR, isolated CI
  NDUFA3 (B9, 19q13.42, CI):     CI subunit, PP-module peripheral, AR, isolated CI
  NDUFA4 (MLRQ, 7p21.3, CIV):    ← COMPLEX IV, not CI — BIOCHEMISTRY MANDATORY DDx
  NDUFA5 (B13, 7q32.1, CI):      CI subunit, Q-module, AR, isolated CI
    SAME CHROMOSOME (7) as NDUFA4 — 7p21.3 (NDUFA4) vs 7q32.1 (NDUFA5)
    Chromosomal ARM (p vs q) is the WES mandatory differentiation pivot
  NDUFA6 (B14, 22q13.2, CI):     CI subunit, Q-module distal, AR, isolated CI
  NDUFA7 (B14.5a, 19p13.3, CI):  CI subunit, N-module peripheral, AR, isolated CI
  NDUFA8 (B14.5b, 2q31.3, CI):   CI subunit, N-Q boundary, AR, isolated CI
  NDUFA9 (39kDa, 12q24.31, CI):  CI subunit, Q-module, AR, isolated CI
  NDUFA10 (42kDa, 2q37.3, CI):   CI subunit, Q-module NDP-kinase, AR, isolated CI
  NDUFA11 (B14.7, 19q13.33, CI): CI subunit, PP/PD boundary, AR, isolated CI
  NDUFA12 (B17.2, 4q21.23, CI):  CI subunit, N-Q interface, AR, isolated CI
  NDUFA13 (GRIM-19, 19p13.11, CI): CI subunit, N-module STAT3-inhibitor, AR, CI

  → NDUFA4 is the ONLY CIV gene in the NDUFA family. All others are CI.
  → Biochemistry: NDUFA4 deficiency → CIV low, CI/CII/CIII normal
  → WES phenotype-first: CIV deficiency on enzyme panel → suspect NDUFA4 (+ SURF1, SCO1,
    SCO2, COX10, COX15, PET100, COX20, COX14, etc.)

DISTINGUISHING FEATURES vs OTHER CIV-LEIGH GENES:
  vs SURF1 (CIV-COXPD1, 9q34.2 — MOST COMMON CIV-Leigh):
    SURF1: CIV assembly factor (not structural subunit); NDUFA4: structural CIV subunit
    SURF1: 70–80% CIV absent; NDUFA4: 5–25% CIV residual (partial)
    SURF1: Leigh MRI 92–97%; NDUFA4: Leigh MRI 80–88% (more partial COX-neg fibers)
    Both AR; SURF1 more common worldwide; NDUFA4 rarer, diagnosed via WES post-CI panel
  vs SCO2 (CIV copper, 22q13.33 — HCM 100%):
    SCO2: hypertrophic cardiomyopathy (HCM) present in virtually 100%; NDUFA4: NO HCM
    SCO2: CuA copper loading defect; NDUFA4: COX2-CuA stabilisation structural defect
    HCM absence is a CRITICAL DDx pivot from SCO2 — absent HCM suggests NDUFA4 over SCO2
  vs SCO1 (CIV copper, 17p13.1):
    SCO1: hepatopathy + neonatal lactic acidosis; NDUFA4: NO hepatopathy
    SCO1: CuA copper chaperone; NDUFA4: structural COX2 contact subunit
  vs COX10 (heme farnesylation, 17p12):
    COX10: tubulopathy (renal tubular acidosis) ~40%; NDUFA4: NO tubular dysfunction
    COX10: haem a biosynthesis; NDUFA4: structural CIV subunit (haem a normal)
  vs COX15 (heme synthesis, 10q24.2):
    COX15: cardiomyopathy + neonatal lethal or hepatopathy; NDUFA4: NO HCM, NO hepatopathy
  vs MT-CO2 (mtDNA-encoded COX2, NDUFA4 direct partner):
    MT-CO2: MATERNAL inheritance (mitochondrial DNA); NDUFA4: AR nuclear (biallelic)
    Inheritance pattern is the KEY first DDx pivot when isolated CIV + Leigh MRI
  vs PET100 (CIV assembly, Lebanese founder, 19p13.3):
    PET100: very rare; Lebanese/Middle-Eastern founder; NDUFA4: broader population
    Both CIV-Leigh; BN-PAGE CIV assembly patterns differ (PET100: early COX1 stall)
  vs CI-Leigh (NDUFA3, NDUFA5, NDUFS4, etc.):
    CI-Leigh: CI low, CIV normal; NDUFA4-CIV-Leigh: CIV low, CI normal
    Biochemistry is MANDATORY — name "NDUFA4" does NOT imply CI deficiency

FOUNDER / RECURRENT MUTATIONS (NDUFA4, 81 aa mature protein, 7p21.3):
  p.Arg52Cys   c.154C>T   — TM helix core, COX2-CuA IMS contact disrupted; severe infantile
  p.Leu40Pro   c.119T>C   — helix-breaking proline in single TM helix; severe
  p.Glu20Lys   c.58G>A    — near MTS cleavage/N-terminus matrix; import/targeting; severe neonatal
  p.Gly61Ser   c.181G>A   — IMS loop periplasmic; COX2 contact zone; intermediate
  c.IVS1+1G>A             — splice donor exon 1; partial CIV residual (~15–25%); moderate

THERAPY — NDUFA4 / CIV-LEIGH SPECIFICS:
  ABSOLUTE contraindications (mito toxins / direct CIV inhibitors):
    Propofol       — PRIS (propofol infusion syndrome) + DIRECT CIV inhibition;
                     propofol is the single most dangerous agent in CIV deficiency;
                     propofol infusion blocks CIV electron transport at haem a3-CuB site
    Metformin      — Complex I inhibitor; adds upstream ETC block to existing CIV failure
    Valproate      — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit block
    Linezolid      — inhibits 23S rRNA → blocks MT-CO1/2/3 synthesis + all 7 ND subunits
    Chloramphenicol — same 23S rRNA mechanism; blocks MT-CO1/2/3 and ND-subunits
  CONTRAINDICATED:
    Ketogenic diet — β-oxidation forces increased electron flux through CI and CIII to CIV;
                     NDUFA4-failed CIV is already the bottleneck — KD exacerbates terminal
                     ETC block, increases mitochondrial ROS, and risks metabolic crisis
  AVOID / HIGH CAUTION:
    Nitrous oxide (N₂O) — oxidises cobalt(I) to cobalt(III) → MTHFR cofactor inactivation;
                           absolute avoidance for any mitochondrial Leigh
    Phenobarbital  — secondary CI + CIV indirect inhibitor; use LEV first-line
    Aminoglycosides — mitochondrial 12S rRNA overlap with mitoribosomes; CIV mRNA vulnerability
  ANAESTHESIA: Use sevoflurane (inhalational) NOT propofol; titrate carefully in CIV failure
  LEVEL C cofactors (standard CIV-Leigh supportive):
    Thiamine (B1)     — MANDATORY empiric: SLC19A3/BTD mimics treatable Leigh; 100–300 mg/day
    Biotin            — MANDATORY empiric: BTD deficiency mimics CIV-Leigh; 10–40 mg/day
    CoQ10 (ubiquinol) — electron donor to CIV (via cytochrome c); 10–30 mg/kg/day ubiquinol
    L-Carnitine       — energy metabolism support; 50–100 mg/kg/day
    Ascorbate (vit C) — maintains cytochrome c in reduced (ferrocytochrome) state;
                        maximises electron delivery to residual NDUFA4-partial CIV
    Riboflavin (B2)   — FMN at CI N-module; less directly relevant for CIV but empiric
  AED CHOICE:
    Levetiracetam (LEV) — PREFERRED: renal excretion, no mito toxicity, broad-spectrum
    Clobazam / clonazepam — focal/myoclonic adjunct; GABA-A modulator; benzodiazepine caution
  EMERGENCY:
    Never fast — GIR 6–8 mg/kg/min IV dextrose during nil-by-mouth periods
    Lactic acidosis crisis → IV thiamine 50–100 mg then bicarbonate if pH <7.1
"""

import random, json

GENE     = "NDUFA4"
DISEASE  = "Leigh Syndrome — Isolated Complex IV Deficiency (CIV-Leigh / COXPD20)"
OMIM_G   = "602137"
OMIM_D   = "256000"
INHERIT  = "Autosomal Recessive (AR) — biallelic"
CHROM    = "7p21.3"
MODULE   = "Complex IV structural subunit (CIV / COX) — COX2-CuA IMS contact, single TM helix — NOT Complex I despite NDUFA name"
SIZE     = "81 aa / 9.3 kDa (mature, after MTS cleavage)"
SEED     = 656
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    ("Severe infantile (onset <6 mo)",       32),
    ("Moderate infantile (onset 6–18 mo)",   38),
    ("Intermediate (onset 18–36 mo)",        20),
    ("Attenuated / partial CIV residual",    10),
]

VARIANTS = [
    ("p.Arg52Cys",  "c.154C>T",
     "TM helix core residue; COX2-CuA IMS contact surface disrupted",
     "Severe infantile",
     32,
     "Arginine-to-cysteine in the single TM helix core at the NDUFA4–COX2 (MT-CO2) IMS contact "
     "interface; disrupts electrostatic anchor to COX2-CuA domain; CIV holocomplex absent on BN-PAGE; "
     "isolated CIV 5–15% residual; CI/CII/CIII all normal — biochemical CIV fingerprint confirmed"),
    ("p.Leu40Pro",  "c.119T>C",
     "Helix-breaking proline in single TM helix; full CIV assembly failure",
     "Severe",
     26,
     "Leucine-to-proline in the single IMM-spanning TM helix of NDUFA4; proline cannot participate "
     "in α-helical backbone H-bond geometry → TM helix kink → NDUFA4 membrane insertion fails; "
     "COX2-CuA IMS contact completely lost; CIV absent on BN-PAGE; CI normal"),
    ("p.Glu20Lys",  "c.58G>A",
     "Near MTS cleavage; mitochondrial targeting / matrix-face import disruption",
     "Severe neonatal",
     14,
     "Glutamate-to-lysine near the MTS cleavage site in the matrix-facing N-terminus; protein "
     "mis-targeting or failed import into the IMM; neonatal-onset Leigh with CIV absence; "
     "unique among NDUFA4 alleles in causing MTS-level failure rather than structural CIV disruption"),
    ("p.Gly61Ser",  "c.181G>A",
     "IMS periplasmic loop; COX2 contact zone; glycine-to-serine intermediate",
     "Intermediate",
     16,
     "Glycine-to-serine in the IMS-facing periplasmic loop of NDUFA4; glycine allows sharp loop "
     "conformation at COX2 docking interface; serine introduces steric clash; partial CIV assembly; "
     "intermediate Leigh with 20–30% residual CIV activity; some COX-negative fibers on muscle bx"),
    ("c.IVS1+1G>A", "Splice donor exon 1",
     "Partial CIV residual (~15–25%); moderate Leigh with survival to late childhood",
     "Moderate / partial",
     12,
     "Splice-donor loss at exon 1/intron 1 boundary; partial exon 1 skipping with some correctly "
     "spliced NDUFA4 transcript → partial CIV assembly; BN-PAGE shows reduced but not absent CIV; "
     "15–25% residual CIV activity; moderate Leigh MRI; survival to late childhood reported"),
]

SEIZURE_TYPES = [
    ("Focal / multifocal (awake + sleep)",    55),
    ("Generalized tonic-clonic (GTCS)",       38),
    ("Myoclonic",                             25),
    ("Infantile spasms (IS / West synd.)",    20),
    ("Epileptic spasms (post-IS residual)",   10),
    ("Absence (atypical)",                     6),
]

TRIGGERS = [
    ("Febrile illness / infection",           80),
    ("Sub-therapeutic AED level",             58),
    ("Metabolic decompensation",              52),
    ("Sleep deprivation",                     36),
    ("Missed AED dose",                       33),
    ("Fasting / prolonged nil-by-mouth",      30),
    ("Anesthesia / surgical stress",          22),
    ("Enzyme-inducing co-medication",         12),
]

TREATMENTS = [
    ("Levetiracetam (LEV)",              "A",  "Preferred AED; renal excretion; NO mito toxicity; broad-spectrum CIV-Leigh safe"),
    ("CoQ10 / Ubiquinol",                "C",  "Electron donor to CIV via cytochrome c; 10–30 mg/kg/day ubiquinol; maximises residual CIV flux"),
    ("Ascorbate (Vitamin C)",            "C",  "Maintains cytochrome c in reduced (ferrocytochrome) state; maximises electron delivery to residual NDUFA4-partial CIV; 500–1000 mg/day"),
    ("Thiamine (B1)",                    "C",  "MANDATORY empiric: SLC19A3/BTD mimics treatable Leigh; 100–300 mg/day before genetic result"),
    ("Biotin",                           "C",  "MANDATORY empiric: BTD deficiency mimics CIV-Leigh; 10–40 mg/day empiric cover"),
    ("Riboflavin (B2 / FMN precursor)",  "C",  "FMN at CI N-module; less directly relevant for CIV but empiric mitochondrial cofactor support"),
    ("L-Carnitine",                      "C",  "Energy metabolism support; secondary transport; 50–100 mg/kg/day"),
    ("Clobazam (CLB) / Clonazepam",      "B",  "Focal / myoclonic adjunct; GABA-A positive modulator; caution benzodiazepine dependence"),
]

CONTRAINDICATIONS = [
    ("Propofol",                          "ABSOLUTE",        "PRIS + DIRECT CIV inhibition at haem a3-CuB site; single most dangerous agent in CIV deficiency; use sevoflurane instead"),
    ("Metformin",                         "ABSOLUTE",        "Complex I inhibitor adds upstream ETC block to existing CIV failure; fatal lactic acidosis risk"),
    ("Valproic acid / VPA",               "ABSOLUTE",        "Triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block; hepatotoxicity risk"),
    ("Linezolid",                         "ABSOLUTE",        "23S rRNA inhibition → blocks MT-CO1/CO2/CO3 synthesis + all 7 ND subunits; CIV depletion cascade"),
    ("Chloramphenicol",                   "ABSOLUTE",        "Same 23S rRNA mitoribosomal mechanism as linezolid; blocks MT-CO1/2/3 (catalytic CIV subunits)"),
    ("Ketogenic diet (KD)",               "CONTRAINDICATED", "KD forces increased electron flux through CIV via β-oxidation; NDUFA4-failed CIV bottleneck exacerbated → crisis"),
    ("Nitrous oxide (N₂O)",               "CONTRAINDICATED", "Oxidises cobalt → MTHFR inactivation; avoided in all mitochondrial Leigh syndromes"),
    ("Phenobarbital",                     "HIGH CAUTION",    "Secondary CI + indirect CIV inhibitor; acceptable only if LEV/CLB fail; monitor lactate"),
    ("Aminoglycosides",                   "HIGH CAUTION",    "Mitoribosomes vulnerable via 12S rRNA overlap; risks MT-CO1/2/3 synthesis reduction"),
    ("Enzyme-inducing AEDs (CBZ/PHT/OXC)", "RELATIVE CI",   "CYP450 induction → cofactor depletion; secondary mito toxicity; avoid if alternatives available"),
]

MONITORING = [
    ("Serum lactate + pyruvate (L:P ratio)",  "At each visit; L:P >20 suggests ETC dysfunction; target L:P <20; CIV-Leigh: elevated post-exercise"),
    ("AED levels (LEV, CLB, CNZ)",            "Every 3–6 months; sub-therapeutic = seizure trigger"),
    ("CoQ10 / Ubiquinol status",              "Annual; adjust supplementation dose based on levels"),
    ("Plasma amino acids (alanine)",           "6-monthly; alanine elevation = lactic acidosis surrogate"),
    ("Neuroimaging MRI brain",                "Every 12 months or at neurological change; Leigh lesion progression tracking"),
    ("Echocardiography",                      "Annual; HCM absent in NDUFA4 (critical DDx from SCO2 ~100% HCM) — screen to confirm absence"),
    ("Ophthalmology (visual acuity, ERG, OCT)", "Annual; pigmentary retinopathy and optic atrophy rare in CIV-Leigh; screen baseline"),
    ("Neurodevelopmental / cognitive battery", "Every 12 months; Bayley / VABS age-appropriate"),
    ("Respiratory function / polysomnography", "6-monthly; central apnoea and hypoventilation in CIV-Leigh"),
    ("Renal function (eGFR, urine organics)",  "Annual; no tubular dysfunction expected in NDUFA4 (DDx from COX10 ~40% tubulopathy)"),
    ("Cytochrome c oxidase (COX) enzyme panel","Baseline + after any acute decompensation; confirm isolated CIV deficiency"),
    ("Muscle biopsy COX / SDH staining",       "At diagnosis; COX-negative SDH-positive fibers; proportion correlates with CIV residual activity"),
    ("Plasma cytochrome c oxidase activity",   "Lymphocytes or fibroblasts; confirm CIV low / CI+CII+CIII normal"),
]

REFERENCES = [
    "Balsa E et al. (2012) Mol Cell — NDUFA4 reclassification as CIV subunit; co-purification with CIV not CI",
    "Pitceathly RD et al. (2013) Ann Neurol — NDUFA4 mutations cause Leigh syndrome with isolated CIV deficiency",
    "Zong S et al. (2018) Cell — Cryo-EM CIV structure; NDUFA4 COX2-CuA IMS contact visualised",
    "Tsukihara T et al. (1996) Science — Bovine CIV crystal structure; original 13-subunit model (NDUFA4 added later)",
    "Rahman S et al. (2014) J Inherit Metab Dis — CIV-Leigh genetics review; NDUFA4 clinical spectrum",
    "Fassone E & Rahman S (2012) J Med Genet — CI/CIV deficiency genetics; NDUFA naming family review",
    "Guerrero-Castillo S et al. (2017) Cell Metab — CI assembly vs CIV assembly; NDUFA4 in CIV super-complex",
    "Lake NJ et al. (2016) Trends Genet — CIV assembly pathway; NDUFA4 COX2-sub-complex incorporation",
]

KEY_CONCEPTS = [
    ("NDUFA4 IS A CIV SUBUNIT — NOT CI — Despite NDUFA Name",
     "NDUFA4 (MLRQ/ML-INSERT) co-purifies with CIV (not CI). All other NDUFA proteins (NDUFA1–13) are CI subunits. "
     "NDUFA4 deficiency → isolated CIV low, CI/CII/CIII NORMAL. WES pipeline 'CI gene panel' flag is a DDx trap. "
     "Biochemistry is MANDATORY: CIV enzyme activity confirms the true complex affected."),
    ("NDUFA4 (7p21.3) vs NDUFA5 (7q32.1) — SAME CHROMOSOME, DIFFERENT COMPLEX",
     "NDUFA4 at 7p21.3 causes CIV deficiency; NDUFA5 (B13) at 7q32.1 causes CI deficiency. "
     "Both on chromosome 7 — chromosomal ARM (p vs q) is the mandatory WES differentiation pivot. "
     "Biochemical complex fingerprint (CIV vs CI) resolves before WES."),
    ("COX2-CuA IMS Contact — Structural Role",
     "NDUFA4 single TM helix anchors in IMM; IMS-facing periplasmic loop contacts COX2 (MT-CO2) CuA domain. "
     "CuA is the binuclear copper centre accepting electrons from ferrocytochrome c. "
     "NDUFA4 loss destabilises COX2-CuA → impaired cytochrome c docking → CIV electron transfer rate drops."),
    ("No HCM — Critical DDx from SCO2 (~100% HCM)",
     "SCO2 (CIV copper chaperone, 22q13.33) causes hypertrophic cardiomyopathy (HCM) in virtually 100% of cases. "
     "NDUFA4 CIV-Leigh has NO HCM. Echocardiography to confirm absence is a mandatory DDx screen vs SCO2."),
    ("No Hepatopathy — DDx from SCO1 and COX10",
     "SCO1 (17p13.1): hepatopathy + neonatal onset; COX10 (17p12): tubulopathy. "
     "NDUFA4-Leigh: NO hepatopathy, NO renal tubular acidosis. Liver and renal function normal at presentation."),
    ("Propofol ABSOLUTE CI — Most Critical Drug Warning in CIV Deficiency",
     "Propofol directly inhibits CIV (haem a3-CuB site) in addition to causing PRIS. "
     "In NDUFA4-CIV-Leigh, propofol causes catastrophic dual inhibition of an already-failed CIV. "
     "Sevoflurane (inhalational) is the safe anaesthetic alternative for all CIV-Leigh including NDUFA4."),
    ("MT-CO2 (maternal) vs NDUFA4 (AR nuclear) — Inheritance DDx",
     "MT-CO2 encodes COX2 — NDUFA4's direct binding partner. MT-CO2 deficiency causes maternal-pattern CIV-Leigh. "
     "NDUFA4 deficiency causes AR (biallelic) CIV-Leigh. Maternal vs AR inheritance is the first DDx pivot "
     "when isolated CIV + Leigh MRI is confirmed."),
    ("Isolated CIV Deficiency — Biochemical Fingerprint",
     "NDUFA4 deficiency: CIV 5–25% residual; CI/CII/CIII activities ALL NORMAL. "
     "This pattern excludes: CI-Leigh (NDUFA1/3/5/7/8/9/10/11/12/13), mtDNA depletion (all-ETC-low), "
     "SURF1 (deeper CIV absent, no residual), and combined deficiencies. Biochemistry first."),
    ("COX-negative Fibers on Muscle Biopsy",
     "COX/SDH double staining: COX-negative (CIV absent) / SDH-positive (SDHA-CIII normal) fibers in proportion "
     "correlating with CIV residual activity. NDUFA4: partial COX-neg pattern (vs complete absence in SURF1). "
     "Ragged-red fibers (RRF) rare — NDUFA4 does not cause mtDNA accumulation."),
    ("Ascorbate (Vitamin C) Rationale in CIV Deficiency",
     "Ascorbate maintains cytochrome c in its reduced (ferrocytochrome c) form, maximising electron delivery "
     "to residual NDUFA4-partial CIV. For CI-Leigh, succinate CII-bypass is used; "
     "for NDUFA4-CIV-Leigh, CII-bypass does not help (CIV is the bottleneck, not CI). "
     "Ascorbate + CoQ10 (ubiquinol) are the key downstream-support agents."),
    ("Thiamine + Biotin MANDATORY Empiric Before Genetic Result",
     "SLC19A3 and BTD deficiencies mimic CIV-Leigh phenotype; empiric thiamine + biotin BEFORE genetic result "
     "can prevent irreversible neurological damage in treatable mimics."),
    ("GIR 6–8 — Never Fast CIV-Leigh",
     "Glucose infusion rate 6–8 mg/kg/min IV dextrose during any nil-by-mouth or peri-operative period. "
     "Fasting forces β-oxidation → increased ETC electron flux → CIV bottleneck exacerbated → metabolic crisis."),
    ("Genetic Counselling AR — 7p21.3",
     "Autosomal recessive; both parents obligate carriers; 25% recurrence risk per pregnancy. "
     "Offer cascade carrier testing (siblings at 50% carrier risk) and prenatal/preimplantation genetic testing."),
]


def _make_patients():
    pats = []
    for i in range(1, N + 1):
        r = rng.random() * 100
        cls = (PHENO_CLASSES[0][0] if r < 32
               else PHENO_CLASSES[1][0] if r < 70
               else PHENO_CLASSES[2][0] if r < 90
               else PHENO_CLASSES[3][0])
        v = rng.choice(VARIANTS)
        age_mo = (rng.randint(1, 6)   if "Severe infant" in cls
                  else rng.randint(6, 18)  if "Moderate"   in cls
                  else rng.randint(18, 36) if "Intermediate" in cls
                  else rng.randint(24, 60))
        pats.append({
            "id":                        f"P{i:02d}",
            "phenotype":                 cls,
            "onset_mo":                  age_mo,
            "variant":                   v[0],
            "cDNA":                      v[1],
            "civ_pct":                   rng.randint(5, 25),
            "has_seizure":               rng.random() < 0.70,
            "has_hypotonia":             rng.random() < 0.82,
            "has_lactic_acidosis":       rng.random() < 0.85,
            "has_leigh_mri":             rng.random() < 0.84,
            "has_respiratory_compromise": rng.random() < 0.38,
            "has_dystonia":              rng.random() < 0.30,
            "has_ataxia":                rng.random() < 0.28,
        })
    return pats


_PATIENTS = _make_patients()


def get_overview():
    pts = _PATIENTS
    n_sz  = sum(1 for p in pts if p["has_seizure"])
    n_hyp = sum(1 for p in pts if p["has_hypotonia"])
    n_lac = sum(1 for p in pts if p["has_lactic_acidosis"])
    n_mri = sum(1 for p in pts if p["has_leigh_mri"])
    n_res = sum(1 for p in pts if p["has_respiratory_compromise"])

    variant_counts: dict = {}
    for p in pts:
        variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1

    phenotype_counts: dict = {}
    for p in pts:
        phenotype_counts[p["phenotype"]] = phenotype_counts.get(p["phenotype"], 0) + 1

    return {
        "gene":           GENE,
        "disease":        DISEASE,
        "omim_gene":      OMIM_G,
        "omim_disease":   OMIM_D,
        "inheritance":    INHERIT,
        "chromosome":     CHROM,
        "module":         MODULE,
        "protein_size":   SIZE,
        "cohort_n":       N,
        "seed":           SEED,
        "kpis": {
            "seizures_pct":        round(n_sz  / N * 100),
            "hypotonia_pct":       round(n_hyp / N * 100),
            "lactic_acidosis_pct": round(n_lac / N * 100),
            "leigh_mri_pct":       round(n_mri / N * 100),
            "respiratory_pct":     round(n_res / N * 100),
            "median_onset_mo":     round(sum(p["onset_mo"] for p in pts) / N, 1),
            "mean_civ_pct":        round(sum(p["civ_pct"]  for p in pts) / N, 1),
        },
        "phenotype_distribution": [
            {"class": pc[0], "n": phenotype_counts.get(pc[0], 0), "pct": pc[1]}
            for pc in PHENO_CLASSES
        ],
        "top_variants":  sorted(variant_counts.items(), key=lambda x: -x[1]),
        "seizure_types": [{"type": t, "pct": p} for t, p in SEIZURE_TYPES],
        "triggers":      [{"trigger": t, "pct": p} for t, p in TRIGGERS],
        "references":    REFERENCES,
        "key_concepts":  [{"concept": c[0], "detail": c[1]} for c in KEY_CONCEPTS],
    }


def get_breakdown():
    pts = _PATIENTS
    variant_rows = []
    for vname, cdna, struct, phenotype_modal, freq, detail in VARIANTS:
        vpts = [p for p in pts if p["variant"] == vname]
        variant_rows.append({
            "variant":           vname,
            "cDNA":              cdna,
            "structural_impact": struct,
            "modal_phenotype":   phenotype_modal,
            "freq_pct":          freq,
            "n_in_cohort":       len(vpts),
            "detail":            detail,
        })

    return {
        "gene":       GENE,
        "cohort_n":   N,
        "variants":   variant_rows,
        "patients":   [
            {
                "id":        p["id"],
                "phenotype": p["phenotype"],
                "onset_mo":  p["onset_mo"],
                "variant":   p["variant"],
                "cDNA":      p["cDNA"],
                "civ_pct":   p["civ_pct"],
                "seizure":   p["has_seizure"],
                "hypotonia": p["has_hypotonia"],
                "leigh_mri": p["has_leigh_mri"],
            }
            for p in pts
        ],
        "treatments":        [
            {"name": t[0], "evidence": t[1], "notes": t[2]}
            for t in TREATMENTS
        ],
        "contraindications": [
            {"drug": c[0], "class": c[1], "reason": c[2]}
            for c in CONTRAINDICATIONS
        ],
        "monitoring":        [
            {"item": m[0], "protocol": m[1]}
            for m in MONITORING
        ],
        "seizure_types":     [{"type": t, "pct": p} for t, p in SEIZURE_TYPES],
        "triggers":          [{"trigger": t, "pct": p} for t, p in TRIGGERS],
    }


def get_definitions():
    return {
        "gene":        GENE,
        "glossary": [
            {"term": "NDUFA4 / MLRQ",      "definition": "NDUFA4 (ML-INSERT / MLRQ) — 81 aa CIV structural subunit despite NDUFA naming; single TM helix; COX2-CuA IMS contact; 7p21.3; AR; the only CIV gene in the NDUFA family"},
            {"term": "CIV-Leigh / COXPD20", "definition": "Complex IV (cytochrome c oxidase) deficiency Leigh syndrome nuclear type 20; NDUFA4 biallelic loss; isolated CIV 5–25% residual; CI/CII/CIII normal"},
            {"term": "COX2 / MT-CO2",       "definition": "mtDNA-encoded Complex IV subunit 2; binuclear CuA copper centre accepting electrons from cytochrome c; NDUFA4's direct IMS-loop binding partner; MT-CO2 mutations → maternal CIV-Leigh"},
            {"term": "CuA centre",           "definition": "Binuclear copper centre in COX2; first electron-acceptor in CIV from ferrocytochrome c; NDUFA4 stabilises CuA via IMS loop contact; loss → impaired cytochrome c docking"},
            {"term": "BN-PAGE / COX-neg",   "definition": "Blue-native PAGE: CIV absent or reduced; COX/SDH muscle staining: COX-negative (CIV-absent) / SDH-positive (CI-II-III normal) fibers — CIV-Leigh histochemical fingerprint"},
            {"term": "Isolated CIV deficiency", "definition": "CIV low (5–25%), CI/CII/CIII all NORMAL — NDUFA4 biochemical fingerprint; excludes all CI-Leigh (NDUFA3/5/7/8 etc.) and combined/mtDNA-depletion disorders"},
            {"term": "Propofol / PRIS",      "definition": "Propofol infusion syndrome; propofol DIRECTLY inhibits CIV haem a3-CuB site; ABSOLUTE contraindication in NDUFA4-CIV-Leigh; use sevoflurane instead"},
            {"term": "SURF1",                "definition": "Most common CIV-Leigh gene (9q34.2); CIV assembly factor; causes deeper CIV absence (70–80%); DDx from NDUFA4: SURF1 = assembly factor, NDUFA4 = structural subunit"},
            {"term": "SCO2",                 "definition": "CIV copper chaperone (22q13.33); HCM ~100% — CRITICAL DDx from NDUFA4 which has NO HCM; both cause CIV-Leigh but cardiac phenotype distinguishes"},
            {"term": "Ascorbate / Vit C",    "definition": "Reduces cytochrome c (ferrocytochrome c) to maximise electron delivery to residual NDUFA4-partial CIV; key supportive therapy in CIV deficiency distinct from CI-bypass strategies"},
            {"term": "NDUFA naming trap",    "definition": "NDUFA4 named for historical association with NADH:ubiquinone oxidoreductase (CI) preparations; bona fide CIV subunit since 2012 (Balsa et al.); WES 'CI panel' flag is a DDx pitfall requiring biochemistry confirmation"},
            {"term": "7p vs 7q — same chromosome", "definition": "NDUFA4 (7p21.3, CIV) and NDUFA5 (7q32.1, CI) both on chromosome 7; arm (p vs q) differentiates them; biochemistry (CIV vs CI deficiency) is the prior clinical pivot"},
            {"term": "GIR 6–8",             "definition": "Glucose infusion rate 6–8 mg/kg/min; mandatory during nil-by-mouth in any Leigh syndrome to prevent fasting-triggered metabolic crisis"},
            {"term": "AR biallelic",         "definition": "Autosomal recessive; two pathogenic variants in trans (compound het) or homozygous; both parents obligate carriers; 25% recurrence per pregnancy"},
            {"term": "Leigh syndrome",       "definition": "Progressive necrotising encephalopathy of childhood; bilateral symmetric brainstem + basal ganglia MRI lesions; ≥100 nuclear/mtDNA gene mutations; NDUFA4 → CIV-Leigh subtype (COXPD20)"},
        ],
        "references": REFERENCES,
    }


if __name__ == "__main__":
    import json as _json
    print("=== OVERVIEW ===")
    print(_json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first 500 chars) ===")
    print(_json.dumps(get_breakdown(), indent=2)[:500])
    print("\n=== DEFINITIONS (first 500 chars) ===")
    print(_json.dumps(get_definitions(), indent=2)[:500])
