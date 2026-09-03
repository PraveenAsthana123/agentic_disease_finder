#!/usr/bin/env python3
"""MT-CO1 — Leigh Syndrome / Bilateral Striatal Necrosis / CIV-Leigh / LHON-plus
   Cytochrome c Oxidase Subunit I — CIV ASSEMBLY NUCLEUS — 514 aa / 57 kDa.

MT-CO1 (also COX1 / COI / MTCO1) is the PRIMARY mtDNA-encoded subunit of Complex IV
(Cytochrome c Oxidase / COX / CIV), located at mtDNA rCRS m.5904–7445 (H-strand).
514 amino acids / 57 kDa — the LARGEST mtDNA-encoded CIV subunit; 12 transmembrane helices.
OMIM gene: *516030.

MT-CO1 IS THE CIV ASSEMBLY NUCLEUS:
  All CIV assembly factors — COA3 (MITRAC12), COX14, COA6, PET100, SURF1, COX18 —
  load onto nascent MT-CO1 co-translationally, at the mitoribosome exit tunnel,
  BEFORE any other CIV subunit joins. MT-CO1 is translated first; COX2 and COX3
  (the other mtDNA-encoded CIV subunits) join subsequent sub-assembly intermediates.
  The COX1-early sub-complex (≈COX1 + nuclear subunits COX4/COX5a/COX6a/COX6c)
  precedes the addition of COX2-containing MITRAC sub-complex.

CATALYTIC FUNCTION — HEME a + HEME a3-CuB (OXYGEN-REDUCTION SITE):
  MT-CO1 contains TWO haem groups unique to CIV:
    • Heme a:    Upstream electron relay — receives electrons from CuA (in MT-CO2)
    • Heme a3-CuB: The TERMINAL oxygen-reduction site — reduces O₂ → 2H₂O.
  This is the FINAL step of the entire mitochondrial electron transport chain.
  Propofol DIRECTLY inhibits the heme a3-CuB site — the mechanistic basis for its
  ABSOLUTE contraindication in all CIV deficiency disorders.

12 TRANSMEMBRANE HELICES — LARGEST mtDNA CIV TOPOLOGY:
  MT-CO1 contains 12 TM helices spanning the IMM, forming the structural scaffold
  on which ALL 11 nuclear-encoded CIV accessory subunits assemble.
  MT-CO2 (228 aa, 2 TM) and MT-CO3 (261 aa, 7 TM) are added in subsequent steps.

DISEASE PHENOTYPES:
  1. Bilateral Striatal Necrosis (BSN) — m.6930G>A — MOST DISTINCTIVE MT-CO1 PHENOTYPE
     Bilateral putamen + caudate T2 hyperintensity, WITHOUT classic brainstem Leigh;
     often misdiagnosed as ADEM or Huntington-mimetic; COX-negative muscle fibers.
  2. Leigh Syndrome / Severe CIV-Leigh — m.6708T>A — infantile, BG+brainstem MRI
  3. LHON-plus / Sensorineural Hearing Loss — m.7444G>A — stop-codon read-through
  4. MELAS-Leigh Overlap — m.6812T>A — stroke-like episodes + Leigh MRI
  5. KSS / CPEO / Pearson — Large deletion spanning CO1 — multi-systemic

Maternal inheritance (mtDNA). WES misses MT-CO1 — dedicated mtDNA sequencing required.
H-strand encoded (unlike MT-ND6 which is L-strand — coverage check must confirm H-strand).
"""

import random

GENE     = "MT-CO1"
ALIAS    = "COX1 / COI / MTCO1"
DISEASE  = "Bilateral Striatal Necrosis / CIV-Leigh / LHON-plus / MELAS-Leigh / KSS-CPEO"
OMIM_G   = "516030"
OMIM_D   = "256000"   # Leigh syndrome (BSN + CIV-Leigh phenotypes)
INHERIT  = "Maternal (mtDNA, H-strand, m.5904–7445 rCRS)"
CHROM    = "mtDNA rCRS m.5904–7445 (H-strand)"
MODULE   = ("CIV Assembly Nucleus — 12 TM helices — Heme a + Heme a3-CuB "
            "(oxygen-reduction terminal site); LARGEST mtDNA-encoded CIV subunit")
SIZE     = "514 aa / 57 kDa (12 TM helices)"
SEED     = 743
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    ("Bilateral striatal necrosis (BSN / m.6930G>A)",           35),
    ("Leigh syndrome / severe infantile CIV (m.6708T>A)",       25),
    ("LHON-plus / SNHL (m.7444G>A stop read-through)",          18),
    ("MELAS-Leigh overlap (m.6812T>A)",                         12),
    ("KSS / CPEO / Pearson (large deletion spanning CO1)",       10),
]

VARIANTS = [
    ("m.6930G>A",
     "p.Lys259Glu",
     "TM7 extracellular loop — heme a3-CuB access channel disrupted; BSN pattern",
     "Bilateral striatal necrosis (BSN)",
     35,
     "Lysine-to-glutamate in the MT-CO1 TM7 extracellular loop region adjacent to the "
     "heme a3-CuB access channel. Disrupts CIV holoenzyme stability preferentially in "
     "striatal neurons (high ETC demand + lowest antioxidant reserve). BN-PAGE: CIV "
     "absent or severely reduced; CI/CII/CIII normal. MRI: bilateral T2 hyperintensity "
     "in putamen + caudate — NOT brainstem-first (DDx from classic Leigh). Often "
     "misdiagnosed as ADEM, metabolic encephalopathy, or Huntington-mimetic in early course."),
    ("m.6708T>A",
     "p.Ile230Thr",
     "TM5 hydrophobic core — heme a pocket destabilised; severe CIV-Leigh",
     "Leigh syndrome / severe infantile CIV",
     25,
     "Isoleucine-to-threonine in TM5 of MT-CO1 — the hydrophobic core region that "
     "encloses the heme a iron-porphyrin electron relay. Polar threonine introduces "
     "steric clash with heme a coordination; CIV holocomplex absent or <10% residual. "
     "Classical Leigh MRI: bilateral BG + brainstem lesions. Infantile onset <6 months. "
     "Severe lactic acidosis (12–20 mM); seizures; respiratory failure."),
    ("m.7444G>A",
     "Stop codon read-through (p.*514Arg ext*?)",
     "3' end — MT-CO1 stop codon → arginine → extended C-terminus disrupts COX5a docking",
     "LHON-plus / Sensorineural hearing loss",
     18,
     "Guanine-to-adenine at the MT-CO1 stop codon (m.7444) converts UGA→AGA → arginine "
     "read-through, extending MT-CO1 by several residues before a downstream stop. "
     "The extended C-terminus disrupts the COX5a nuclear subunit docking interface. "
     "Phenotype: LHON (optic atrophy) + sensorineural hearing loss (SNHL) co-occurring — "
     "a LHON-plus syndrome distinct from the classic LHON triad (ND4/ND1/ND6). "
     "CI is mildly reduced (55–75%); CIV is the primary deficiency. Maternal inheritance. "
     "NOTE: m.7444G>A disrupts the overlapping tRNA-Ser(UCN) promoter region — "
     "SNHL partly due to tRNA-Ser biogenesis impairment in cochlear hair cells."),
    ("m.6812T>A",
     "p.Asp291Asn",
     "TM7 heme a3-CuB coordinating His285/His290 neighbourhood; MELAS-Leigh overlap",
     "MELAS-Leigh overlap",
     12,
     "Aspartate-to-asparagine at position 291 in TM7, near the histidine residues "
     "(His285, His290) that coordinate the CuB copper ion in the heme a3-CuB site. "
     "Asparagine disrupts local CuB coordination geometry → partial CIV loss (20–40% "
     "residual). Produces MELAS-Leigh phenotype: stroke-like episodes (NOT thrombotic — "
     "avoid tPA), bilateral BG lesions, lactic acidosis 8–15 mM. Heteroplasmy: "
     "60–80% mutant → MELAS/Leigh overlap; >85% → severe CIV-Leigh. "
     "Note: IV tPA CONTRAINDICATED — stroke-like episodes are metabolic, not embolic."),
    ("Large deletion (5.9–7.4 kb, CO1-spanning)",
     "Multi-gene deletion",
     "Spans MT-CO1 + flanking tRNAs → KSS / CPEO / Pearson syndrome",
     "KSS / CPEO / Pearson",
     10,
     "Large-scale single deletion spanning m.5904–7445 (MT-CO1) plus flanking tRNAs "
     "(tRNA-Ile, tRNA-Gln, tRNA-Met) and portions of ND2. Combined CI + CIV deficiency; "
     "multiple OXPHOS complexes reduced proportional to deletion heteroplasmy. "
     "Tissue-specific segregation: KSS in muscle + ophthalmoplegia + retinopathy; "
     "CPEO (chronic progressive external ophthalmoplegia) in adult-onset; "
     "Pearson in childhood (sideroblastic anemia + exocrine pancreatic failure). "
     "Long-read mtDNA sequencing or Southern blot required — Illumina short-read "
     "often fails to detect large deletions across repetitive MT-CO1 region."),
]

SEIZURE_TYPES = [
    ("Focal / multifocal",                          52),
    ("Generalized tonic-clonic (GTCS)",             38),
    ("Myoclonic",                                   28),
    ("Infantile spasms (IS)",                       18),
    ("Epileptic spasms (post-IS residual)",         10),
    ("Absence (atypical)",                           6),
]

TRIGGERS = [
    ("Febrile illness / infection",                 82),
    ("Metabolic decompensation",                    64),
    ("Sub-therapeutic AED level",                   55),
    ("Fasting / prolonged nil-by-mouth",            42),
    ("Sleep deprivation",                           35),
    ("Missed AED dose",                             30),
    ("Anaesthesia / surgical stress",               24),
    ("High-carbohydrate load (rebound OXPHOS flux)", 15),
]

TREATMENTS = [
    ("Levetiracetam (LEV)",               "A",
     "Preferred AED; renal excretion; NO mitochondrial toxicity; broad-spectrum; "
     "safe in all MT-CO1 phenotypes including BSN and Leigh"),
    ("CoQ10 / Ubiquinol",                 "C",
     "Electron donor to CIV via cytochrome c; maximises residual CIV flux at heme a entry; "
     "10–30 mg/kg/day ubiquinol preferred (better bioavailability); start early"),
    ("Ascorbate (Vitamin C)",             "C",
     "Reduces cytochrome c (ferrocytochrome c) → maximises electron delivery to "
     "residual MT-CO1-partial CIV at heme a entry; 500–1000 mg/day; rationale same as NDUFA4"),
    ("Thiamine (B1)",                     "C",
     "MANDATORY empiric before genetic result: SLC19A3 / BTD mimic CIV-Leigh and BSN; "
     "100–300 mg/day IV/oral; prevents irreversible neurological damage in treatable mimics"),
    ("Biotin",                            "C",
     "MANDATORY empiric: BTD deficiency mimics CIV-Leigh/BSN; 10–40 mg/day; "
     "give with thiamine as empiric cover before molecular diagnosis"),
    ("Riboflavin (B2 / FMN-FAD)",         "C",
     "FAD cofactor for SDHA (CII) and other OXPHOS enzymes; mitochondrial cofactor "
     "support; 50–400 mg/day; less directly critical for CIV vs CI disorders"),
    ("L-Carnitine",                       "C",
     "Energy metabolism support via acylcarnitine transport; 50–100 mg/kg/day; "
     "secondary but part of standard mitochondrial disease cocktail"),
    ("Baclofen / Botulinum toxin-A",      "C",
     "For dystonia in BSN phenotype (m.6930G>A); baclofen PO 5–40 mg/day titrated; "
     "BTX-A focal injection for focal dystonia refractory to oral agents"),
    ("Clobazam (CLB)",                    "B",
     "Focal / myoclonic adjunct; GABA-A positive modulator; "
     "bedtime dosing for nocturnal seizures; caution benzodiazepine dependence"),
]

CONTRAINDICATIONS = [
    ("Propofol",                        "ABSOLUTE",
     "DIRECT INHIBITOR of heme a3-CuB site in MT-CO1 — the oxygen-reduction catalytic "
     "centre. In MT-CO1 deficiency, propofol causes DUAL catastrophic inhibition: "
     "(1) PRIS (propofol infusion syndrome — mitochondrial toxicity) AND "
     "(2) direct heme a3-CuB active-site blockade of the already-compromised CIV. "
     "Use SEVOFLURANE (inhalational) for ALL anaesthesia in MT-CO1 patients."),
    ("Metformin",                       "ABSOLUTE",
     "Complex I (NADH:ubiquinone oxidoreductase) inhibitor; adds upstream ETC block "
     "to existing CIV failure. CI inhibition → NADH backlog → lactic acidosis crisis. "
     "Fatal combined CI + CIV failure in MT-CO1 Leigh / BSN."),
    ("Valproic acid / VPA",             "ABSOLUTE",
     "Triple mechanism: CoA sequestration (depletes mitochondrial CoA pool) + "
     "POLG inhibition (mtDNA replication failure) + suppression of mtDNA-encoded "
     "ND and CO subunit expression. Hepatotoxicity risk via POLG interaction."),
    ("Linezolid",                       "ABSOLUTE",
     "23S rRNA peptidyl transferase inhibitor → directly blocks MT-CO1 protein "
     "synthesis at mitoribosome. MT-CO1 loss collapses the entire CIV assembly "
     "nucleus — all 14 CIV subunits fail to assemble. Also blocks ND1–ND6 + ND4L "
     "(all 7 mtDNA CI subunits) simultaneously. ABSOLUTE CI in any MT-CO1 disorder."),
    ("Chloramphenicol",                 "ABSOLUTE",
     "Same 23S rRNA mitoribosomal mechanism as linezolid; inhibits peptidyl "
     "transferase → blocks MT-CO1 / CO2 / CO3 synthesis. Fatal in CIV deficiency."),
    ("Nitrous oxide (N₂O)",             "CONTRAINDICATED",
     "Irreversibly oxidises cobalt in methylcobalamin → MTHFR inactivation; "
     "folate cycle collapse. Avoided in all mitochondrial Leigh syndromes."),
    ("Ketogenic diet (KD)",             "CONTRAINDICATED",
     "KD forces β-oxidation → excess FADH2 flux via CII → increased electron pressure "
     "at CoQ → CIV bottleneck exacerbated → metabolic crisis. "
     "NOT safe in MT-CO1 CIV-Leigh or BSN (unlike some CI-Leigh where KD may help)."),
    ("IV t-PA (thrombolysis)",          "CONTRAINDICATED — MELAS-Leigh phenotype",
     "MELAS-Leigh stroke-like episodes are METABOLIC (cytotoxic oedema), NOT thrombotic. "
     "tPA in MELAS-Leigh → haemorrhagic transformation. Treat with thiamine + "
     "IV dextrose GIR 6–8 + CoQ10. MRI DWI may mimic ischaemia — confirm metabolic aetiology."),
    ("Phenobarbital",                   "HIGH CAUTION",
     "Secondary CI inhibitor; also induces hepatic CYP enzymes → cofactor depletion. "
     "Acceptable only if LEV / CLB fail; monitor serum lactate closely."),
    ("Aminoglycosides",                 "HIGH CAUTION",
     "12S rRNA mitoribosomal overlap vulnerability → reduced MT-CO1/CO2/CO3 synthesis "
     "possible at high doses; risk amplified in m.7444G>A (SNHL phenotype — cochlear "
     "damage further compounded by aminoglycoside ototoxicity)."),
    ("Enzyme-inducing AEDs (CBZ/PHT/OXC)", "RELATIVE CI",
     "CYP3A4/2C induction → cofactor depletion (CoQ10, carnitine, B vitamins); "
     "secondary mitochondrial toxicity; avoid if alternatives available."),
]

MONITORING = [
    ("Serum lactate + pyruvate (L:P ratio)",
     "At every clinic visit; L:P >20 = ETC dysfunction; target <20; "
     "CIV-Leigh: elevated post-exercise and post-prandially; "
     "BSN: check at decompensation (viral illness / fever)"),
    ("AED levels (LEV, CLB)",
     "Every 3–6 months; sub-therapeutic levels = major seizure trigger in MT-CO1 Leigh"),
    ("CoQ10 / Ubiquinol plasma levels",
     "Annual; adjust dose; target upper-normal range; ubiquinol preferred (better absorption)"),
    ("Plasma amino acids (alanine)",
     "6-monthly; elevated alanine = lactic acidosis surrogate; "
     "alanine >450 μmol/L warrants urgent metabolic review"),
    ("MRI brain + spectroscopy",
     "Every 12 months or at neurological change; "
     "BSN: serial putamen/caudate T2/FLAIR; Leigh: BG + brainstem lesion tracking; "
     "MRS: elevated lactate doublet at 1.33 ppm in lesions"),
    ("Echocardiography",
     "Annual; NO HCM expected in MT-CO1 (critical DDx from SCO2 ~100% HCM); "
     "confirm cardiomyopathy absence; if HCM present → reconsider SCO2 / COX15"),
    ("Ophthalmology (VA, ERG, OCT)",
     "Annual; LHON-plus phenotype (m.7444G>A): optic nerve head OCT RNFL; "
     "ERG for retinal involvement in KSS / large deletion; "
     "pigmentary retinopathy in KSS — annual fundus photography"),
    ("Audiometry",
     "Annual; mandatory in m.7444G>A (LHON-plus / SNHL phenotype); "
     "baseline before aminoglycoside use even in other phenotypes"),
    ("Renal function (eGFR, urine organics, protein)",
     "Annual; NO renal tubular acidosis in MT-CO1 (DDx from COX10 ~40% tubulopathy); "
     "confirm absence; KSS large deletion may have secondary tubulopathy"),
    ("Haematology (CBC, reticulocytes, bone marrow if Pearson)",
     "At diagnosis for KSS / large deletion phenotype; "
     "Pearson syndrome: sideroblastic anemia + vacuolated marrow precursors — "
     "transfusion dependence in severe cases"),
    ("Cytochrome c oxidase (CIV) enzyme activity",
     "Baseline in fibroblasts / lymphocytes / muscle: CIV activity vs CI/CII/CIII; "
     "MT-CO1 mutations: isolated CIV OR combined CIV+CI (large deletion); "
     "confirm CIV is primary complex deficient"),
    ("COX / SDH dual staining (muscle biopsy)",
     "At diagnosis; COX-negative / SDH-positive fibers = CIV histochemical signature; "
     "RRF in large deletion (KSS/CPEO) but rare in point mutations; "
     "proportion of COX-negative fibers correlates with severity"),
    ("mtDNA heteroplasmy (muscle)",
     "At diagnosis + 2-yearly; blood underestimates by 15–30% vs muscle; "
     "heteroplasmy threshold: <60% mild/subclinical, 60–80% Leigh/BSN, >85% severe; "
     "threshold shifts with age (heteroplasmy may increase in post-mitotic tissues)"),
    ("Long-read mtDNA sequencing (large deletion screen)",
     "For KSS / CPEO / Pearson phenotype; Illumina short-read often misses "
     "the large CO1-spanning deletion — Oxford Nanopore or PacBio preferred; "
     "Southern blot also acceptable for deletion detection"),
]

REFERENCES = [
    "Tsukihara T et al. (1996) Science 272:1136 — Bovine CIV crystal structure; "
    "MT-CO1 12-TM topology; heme a + heme a3-CuB positions resolved",
    "Mick DU et al. (2012) Cell Metab 16:441 — MITRAC complex; COA3/COX14 co-translational "
    "MT-CO1 assembly; MT-CO1 as CIV assembly nucleus confirmed by pulse-chase",
    "Stroud DA et al. (2015) Cell Metab 21:765 — Comprehensive CIV assembly survey; "
    "MT-CO1-early sub-complex defined; COX4/COX5a join first",
    "Karadimas CL et al. (2000) Ann Neurol 48:424 — MT-CO1 m.6930G>A bilateral striatal "
    "necrosis: pure CIV deficiency + striatal T2 hyperintensity",
    "Petruzzella V et al. (1994) Hum Mol Genet 3:1105 — MT-CO1 stop-codon mutation "
    "(m.7444G>A) with LHON + sensorineural deafness",
    "Clark KM et al. (1999) Nat Genet 23:347 — MT-CO1 heteroplasmy; BSN clinical series",
    "Zong S et al. (2018) Cell 173:1051 — Cryo-EM CIV + respirasomes; MT-CO1 "
    "central scaffold position in CIV super-complex confirmed",
    "Dröse S & Brandt U (2012) Adv Exp Med Biol — Propofol mechanism; direct "
    "heme a3-CuB site inhibition in CIV confirmed by FTIR spectroscopy",
]

KEY_CONCEPTS = [
    ("MT-CO1 = CIV ASSEMBLY NUCLEUS — All COA/Assembly Factors Load Here First",
     "MT-CO1 is translated first among the 3 mtDNA-encoded CIV subunits. "
     "Co-translational loading: COA3 (MITRAC12), COX14, COX18, COA6, PET100 all bind "
     "nascent MT-CO1 at the mitoribosome exit tunnel before MT-CO1 reaches its final IMM topology. "
     "The COX1-early sub-complex (MT-CO1 + COX4/5a/6a/6c nuclear subunits) is the "
     "first stable intermediate. MT-CO2 sub-complex (MITRAC branch) adds later; MT-CO3 last. "
     "MT-CO1 deficiency → entire assembly cascade collapses before it begins."),
    ("HEME a + HEME a3-CuB = TERMINAL OXYGEN-REDUCTION SITE",
     "MT-CO1 is unique: it contains BOTH haem groups in CIV. "
     "Heme a (upstream): receives electrons from CuA (in MT-CO2/COX2). "
     "Heme a3-CuB (downstream): the ACTIVE SITE where O₂ → 2H₂O (4e⁻ transfer). "
     "This is the TERMINAL step of the entire ETC. "
     "Propofol directly binds and inhibits heme a3-CuB — this is the molecular basis "
     "for propofol's ABSOLUTE contraindication in ALL CIV deficiency disorders."),
    ("BILATERAL STRIATAL NECROSIS (BSN) — m.6930G>A MOST DISTINCTIVE MT-CO1 PHENOTYPE",
     "m.6930G>A (p.Lys259Glu) causes bilateral putamen + caudate T2/FLAIR hyperintensity "
     "WITHOUT classic diffuse Leigh brainstem involvement. "
     "Striata are uniquely vulnerable: highest basal ETC demand + lowest antioxidant reserve. "
     "Initially misdiagnosed as: ADEM (no enhancement, metabolic context), "
     "Wilson disease (no KF rings, no hepatic copper), Huntington mimetic (AD family history absent). "
     "COX-negative muscle fibers confirm CIV deficiency. Treat: GIR 6–8 + thiamine + CoQ10 + LEV."),
    ("LINEZOLID DIRECTLY BLOCKS MT-CO1 SYNTHESIS — The Assembly Nucleus Collapse Mechanism",
     "Linezolid inhibits 23S rRNA peptidyl transferase → blocks ALL mitoribosomal translation. "
     "MT-CO1 (the assembly nucleus) is the MOST CRITICAL target: its loss collapses "
     "the entire CIV assembly cascade. Also blocked simultaneously: MT-CO2, MT-CO3, "
     "ND1–ND6, ND4L (all 10 mtDNA-encoded ETC subunits). "
     "Even short courses (7–10 days) cause measurable CIV activity reduction. "
     "ABSOLUTE CI in ALL CIV deficiency. Chloramphenicol: same mechanism."),
    ("H-STRAND ENCODED — Unlike MT-ND6 (L-strand)",
     "MT-CO1 is encoded on the H-strand of mtDNA (m.5904–7445, rCRS). "
     "ALL 7 mtDNA CI subunits (ND1–ND5, ND4L, ND6 is L-strand exception) and "
     "MT-CO1/CO2/CO3 (all CIV) are H-strand encoded EXCEPT MT-ND6 (L-strand). "
     "NGS: H-strand coverage check is standard for MT-CO1; L-strand coverage is NOT needed. "
     "m.7444G>A is at the H-strand CO1/tRNA-Ser(UCN) boundary — requires careful "
     "H-strand orientation confirmation in the NGS pipeline."),
    ("PROPOFOL — ABSOLUTE CI — MOST DANGEROUS AGENT IN CIV DEFICIENCY",
     "Propofol inhibits CIV (heme a3-CuB) by two mechanisms: "
     "(1) DIRECT site-directed inhibition of heme a3-CuB oxygen-reduction active site (FTIR confirmed) "
     "(2) PRIS (Propofol Infusion Syndrome) = mitochondrial ETC uncoupling at CI + CIV. "
     "In MT-CO1 deficiency, propofol causes COMBINED direct heme a3-CuB block + mitochondrial collapse "
     "in an already-failed CIV — uniformly catastrophic. "
     "SAFE ALTERNATIVE: Sevoflurane (inhalational). Ketamine second-line if inhalational unavailable."),
    ("12 TRANSMEMBRANE HELICES — LARGEST mtDNA-encoded CIV TOPOLOGY",
     "MT-CO1 (514 aa) has 12 TM helices spanning the IMM — more than MT-CO3 (7 TM) "
     "or MT-CO2 (2 TM). Forms the central scaffold on which all 11 nuclear-encoded "
     "accessory CIV subunits assemble (COX4, COX5a, COX5b, COX6a, COX6b, COX6c, "
     "COX7a, COX7b, COX7c, COX8a, NDUFA4/MLRQ). "
     "12 TM also explains why MT-CO1 mutations produce a wide spectrum — different mutations "
     "disrupt heme a pocket, heme a3-CuB site, proton channels, or assembly interfaces."),
    ("ISOLATED CIV vs COMBINED DEFICIENCY — Genotype Correlates",
     "MT-CO1 point mutations (m.6930G>A, m.6708T>A, m.6812T>A) → ISOLATED CIV deficiency "
     "(CI/CII/CIII normal) — BN-PAGE CIV absent or severely reduced. "
     "MT-CO1 large deletion (KSS/CPEO/Pearson) → COMBINED CI + CIV + variable CIII deficiency "
     "proportional to deletion size and heteroplasmy. "
     "m.7444G>A (stop read-through) → CIV low + mild CI reduction (55–75%) from secondary OXPHOS coupling effects. "
     "Biochemistry first: isolated vs combined guides which complex to prioritise."),
    ("NO HCM — Critical DDx from SCO2 (100% HCM)",
     "SCO2 (copper chaperone, 22q13.33) causes hypertrophic cardiomyopathy in "
     "~100% of affected patients. MT-CO1-CIV-Leigh has NO HCM. "
     "Echocardiography to confirm HCM absence is MANDATORY in all CIV-Leigh — "
     "if HCM present, reconsider SCO2 / COX15 / COA6 (which can cause HCM). "
     "Also: NO hepatopathy (DDx SCO1 — hepatopathy 100%); NO tubulopathy (DDx COX10 ~40%)."),
    ("GIR 6–8 — NEVER FAST in CIV-Leigh / BSN",
     "Glucose infusion rate 6–8 mg/kg/min during ANY nil-by-mouth period. "
     "Fasting → β-oxidation activation → excess FADH2 via CII → CoQ pool reduced → "
     "electron pressure at CIV bottleneck → metabolic crisis. "
     "BSN patients decompensate acutely with febrile illness — GIR + antipyretics are the "
     "FIRST intervention. Never delay GIR while investigating fever aetiology."),
    ("WES MISSES MT-CO1 — Dedicated mtDNA Sequencing Required",
     "WES panels typically exclude mtDNA or have poor coverage of the highly repetitive "
     "MT-CO1 region (m.5904–7445). MT-CO1 variants (especially large deletions and heteroplasmic "
     "point mutations) are NOT reliably detected by standard WES. "
     "Dedicated mtDNA sequencing (MitoSeq / high-coverage NGS) in MUSCLE (not blood — "
     "blood underestimates heteroplasmy by 15–30%) is required. "
     "Large deletions need long-read sequencing or Southern blot."),
    ("Thiamine + Biotin MANDATORY Empiric — Treatable BSN Mimics",
     "SLC19A3 (biotin-thiamine-responsive basal ganglia disease / BTBGD) causes "
     "bilateral striatal necrosis clinically and radiologically INDISTINGUISHABLE from "
     "MT-CO1 m.6930G>A BSN. BTBGD is RAPIDLY FATAL without thiamine but FULLY REVERSIBLE "
     "with early thiamine + biotin. Give BEFORE mtDNA result. "
     "Also: BTD deficiency (biotinidase) mimics CIV-Leigh phenotype. "
     "Empiric: Thiamine 100–300 mg/day + Biotin 10–40 mg/day — while awaiting results."),
    ("GIR 6–8 / Ascorbate + CoQ10 — Cytochrome c Reduction Strategy",
     "In CIV deficiency (MT-CO1 as the catalytic core): "
     "Ascorbate maintains cytochrome c in the REDUCED (ferrocytochrome c) state → "
     "maximises electron donation rate to the heme a entry point of residual CIV. "
     "CoQ10/Ubiquinol provides the reduced CoQH2 pool → cytochrome c → heme a → heme a3-CuB. "
     "For CI-Leigh (NDUFA subunits): succinate CII-bypass is used instead. "
     "For CIV-Leigh (MT-CO1): succinate CII-bypass does NOT help (CIV is the bottleneck); "
     "use ascorbate + CoQ10 to maximise flux through residual MT-CO1 CIV."),
]


def _make_patients():
    pats = []
    for i in range(1, N + 1):
        r = rng.random() * 100
        if   r < 35:  cls = PHENO_CLASSES[0][0]
        elif r < 60:  cls = PHENO_CLASSES[1][0]
        elif r < 78:  cls = PHENO_CLASSES[2][0]
        elif r < 90:  cls = PHENO_CLASSES[3][0]
        else:         cls = PHENO_CLASSES[4][0]

        v = rng.choice(VARIANTS[:4])   # pick from point mutations (not deletion) for per-patient
        if "BSN" in cls:    v = VARIANTS[0]
        elif "Leigh" in cls and "MELAS" not in cls:  v = VARIANTS[1]
        elif "LHON" in cls: v = VARIANTS[2]
        elif "MELAS" in cls: v = VARIANTS[3]
        else:               v = VARIANTS[4]

        is_bsn = "BSN" in cls
        is_kss = "KSS" in cls
        is_lhon = "LHON" in cls

        pats.append({
            "id":                          f"P{i:02d}",
            "phenotype":                   cls,
            "variant":                     v[0],
            "cDNA":                        v[1],
            "civ_pct":                     rng.randint(3, 22) if not is_kss else rng.randint(8, 35),
            "onset_mo":                    (rng.randint(6, 36) if is_bsn
                                            else rng.randint(1, 6) if "Leigh" in cls and not is_kss
                                            else rng.randint(24, 240) if is_lhon or is_kss
                                            else rng.randint(3, 18)),
            "has_seizure":                 rng.random() < (0.62 if not is_kss else 0.40),
            "has_hypotonia":               rng.random() < (0.80 if not is_lhon else 0.25),
            "has_lactic_acidosis":         rng.random() < (0.88 if not is_lhon else 0.42),
            "has_leigh_mri":               rng.random() < (0.22 if is_bsn else 0.80 if not is_lhon else 0.15),
            "has_bsn_mri":                 is_bsn,
            "has_optic_atrophy":           rng.random() < (0.92 if is_lhon else 0.12),
            "has_snhl":                    rng.random() < (0.78 if is_lhon else 0.10),
            "has_respiratory":             rng.random() < (0.35 if not is_kss and not is_lhon else 0.18),
            "has_dystonia":                rng.random() < (0.62 if is_bsn else 0.18),
            "has_ophthalmoplegia":         rng.random() < (0.85 if is_kss else 0.08),
        })
    return pats


_PATIENTS = _make_patients()


def get_overview():
    pts = _PATIENTS
    n_sz   = sum(1 for p in pts if p["has_seizure"])
    n_hyp  = sum(1 for p in pts if p["has_hypotonia"])
    n_lac  = sum(1 for p in pts if p["has_lactic_acidosis"])
    n_mri  = sum(1 for p in pts if p["has_leigh_mri"])
    n_bsn  = sum(1 for p in pts if p["has_bsn_mri"])
    n_oa   = sum(1 for p in pts if p["has_optic_atrophy"])
    n_resp = sum(1 for p in pts if p["has_respiratory"])

    variant_counts: dict = {}
    for p in pts:
        variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1

    phenotype_counts: dict = {}
    for p in pts:
        phenotype_counts[p["phenotype"]] = phenotype_counts.get(p["phenotype"], 0) + 1

    return {
        "gene":           GENE,
        "alias":          ALIAS,
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
            "seizures_pct":          round(n_sz  / N * 100),
            "hypotonia_pct":         round(n_hyp / N * 100),
            "lactic_acidosis_pct":   round(n_lac / N * 100),
            "leigh_mri_pct":         round(n_mri / N * 100),
            "bsn_mri_pct":           round(n_bsn / N * 100),
            "optic_atrophy_pct":     round(n_oa  / N * 100),
            "respiratory_pct":       round(n_resp / N * 100),
            "mean_civ_pct":          round(sum(p["civ_pct"] for p in pts) / N, 1),
            "median_onset_mo":       round(
                sorted(p["onset_mo"] for p in pts)[N // 2], 1),
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
    for vname, aa, struct, phenotype_modal, freq, detail in VARIANTS:
        vpts = [p for p in pts if p["variant"] == vname]
        variant_rows.append({
            "variant":           vname,
            "amino_acid":        aa,
            "structural_impact": struct,
            "modal_phenotype":   phenotype_modal,
            "freq_pct":          freq,
            "n_in_cohort":       len(vpts),
            "detail":            detail,
        })

    return {
        "gene":      GENE,
        "cohort_n":  N,
        "variants":  variant_rows,
        "patients": [
            {
                "id":           p["id"],
                "phenotype":    p["phenotype"],
                "onset_mo":     p["onset_mo"],
                "variant":      p["variant"],
                "amino_acid":   p["cDNA"],
                "civ_pct":      p["civ_pct"],
                "seizure":      p["has_seizure"],
                "hypotonia":    p["has_hypotonia"],
                "lactic_ac":    p["has_lactic_acidosis"],
                "leigh_mri":    p["has_leigh_mri"],
                "bsn_mri":      p["has_bsn_mri"],
                "optic_atrophy": p["has_optic_atrophy"],
                "snhl":         p["has_snhl"],
                "dystonia":     p["has_dystonia"],
                "ophthalmo":    p["has_ophthalmoplegia"],
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
        "gene":     GENE,
        "glossary": [
            {"term": "MT-CO1 / COX1 / COI",
             "definition": ("514 aa / 57 kDa mtDNA H-strand encoded Complex IV subunit 1; "
                            "CIV assembly nucleus; 12 TM helices; Heme a + Heme a3-CuB; "
                            "OMIM *516030; m.5904–7445 rCRS")},
            {"term": "CIV Assembly Nucleus",
             "definition": ("MT-CO1 is the first CIV subunit synthesised; all assembly factors "
                            "(COA3/MITRAC, COX14, COA6, PET100, SURF1, COX18) load onto nascent "
                            "MT-CO1 co-translationally at the mitoribosome. MT-CO2/CO3 join later.")},
            {"term": "Heme a + Heme a3-CuB",
             "definition": ("Two iron-porphyrin groups unique to MT-CO1 in CIV. "
                            "Heme a = upstream electron relay (from CuA/COX2); "
                            "Heme a3-CuB = TERMINAL oxygen-reduction site (O₂ → 2H₂O). "
                            "Propofol directly inhibits heme a3-CuB — absolute CI.")},
            {"term": "Bilateral Striatal Necrosis (BSN)",
             "definition": ("Bilateral putamen + caudate T2/FLAIR hyperintensity caused by "
                            "MT-CO1 m.6930G>A; NOT classic Leigh (brainstem-sparing); "
                            "misdiagnosed as ADEM, Wilson, or Huntington-mimetic; CIV deficiency")},
            {"term": "m.6930G>A (BSN)",
             "definition": ("p.Lys259Glu in MT-CO1 TM7 loop; heme a3-CuB access channel disrupted; "
                            "bilateral striatal necrosis; isolated CIV deficiency; "
                            "striatal neurons most vulnerable — highest ETC demand, lowest antioxidant reserve")},
            {"term": "m.7444G>A (LHON-plus)",
             "definition": ("Stop codon read-through p.*514Arg; disrupts COX5a docking + tRNA-Ser(UCN); "
                            "phenotype: LHON (optic atrophy) + SNHL; CIV reduced; "
                            "rare among LHON-primary variants — MT-ND4/ND1/ND6 are the dominant three")},
            {"term": "MITRAC / COA3-COX14 Complex",
             "definition": ("MT-CO1 co-translational assembly complex; COA3 (MITRAC12) + COX14 "
                            "stabilise nascent MT-CO1 at mitoribosome exit; first assembly intermediate; "
                            "shields MT-CO1 from YME1L/AFG3L2 IMM proteases until folded correctly")},
            {"term": "Isolated CIV Deficiency",
             "definition": ("CIV 3–22% residual; CI/CII/CIII ALL NORMAL — MT-CO1 point-mutation fingerprint; "
                            "excludes CI-Leigh (NDUFA subunits), CIII-Leigh (UQCC2/BCS1L), "
                            "combined deficiency (large deletion), and mtDNA depletion (all-ETC-low)")},
            {"term": "Propofol / PRIS",
             "definition": ("Propofol Infusion Syndrome; propofol DIRECTLY inhibits heme a3-CuB in MT-CO1 "
                            "PLUS causes PRIS mitochondrial uncoupling; ABSOLUTE CI in all CIV disorders; "
                            "use sevoflurane for any anaesthesia in MT-CO1 patients")},
            {"term": "COX/SDH Dual Stain",
             "definition": ("Muscle histochemistry: COX-negative (CIV absent, brown oxidase activity lost) / "
                            "SDH-positive (SDHA / CII normal, blue staining retained) fibers = "
                            "CIV histochemical signature; proportion correlates with MT-CO1 heteroplasmy")},
            {"term": "KSS / CPEO / Pearson",
             "definition": ("Large mtDNA deletion phenotypes: KSS = ophthalmoplegia + retinopathy + heart block; "
                            "CPEO = progressive external ophthalmoplegia (adult onset); "
                            "Pearson = sideroblastic anemia + pancreatic failure (childhood). "
                            "Large deletion spanning MT-CO1 → combined CI + CIV deficiency")},
            {"term": "SLC19A3 / BTBGD",
             "definition": ("Biotin-Thiamine-Responsive Basal Ganglia Disease; bilateral striatal necrosis "
                            "INDISTINGUISHABLE from MT-CO1 m.6930G>A on MRI; rapidly fatal without thiamine; "
                            "fully reversible with early thiamine + biotin — MANDATORY empiric treatment")},
            {"term": "H-strand (MT-CO1)",
             "definition": ("MT-CO1 encoded on mtDNA Heavy strand (m.5904–7445 rCRS); "
                            "ALL 3 mtDNA CIV subunits (CO1/CO2/CO3) are H-strand encoded; "
                            "unlike MT-ND6 (L-strand) — H-strand NGS coverage check is standard for MT-CO1")},
            {"term": "GIR 6–8",
             "definition": ("Glucose infusion rate 6–8 mg/kg/min IV dextrose during nil-by-mouth; "
                            "MANDATORY in CIV-Leigh + BSN; fasting → β-oxidation → CIV bottleneck crisis; "
                            "BSN patients decompensate acutely with fever — GIR is first-line management")},
            {"term": "Leigh Syndrome",
             "definition": ("Progressive necrotising encephalopathy of childhood; bilateral symmetric "
                            "brainstem + basal ganglia MRI lesions; MT-CO1 mutations → CIV-Leigh subtype; "
                            ">100 nuclear/mtDNA genes cause Leigh; MT-CO1 is one of the mtDNA-encoded causes")},
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
