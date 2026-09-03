#!/usr/bin/env python3
"""MT-Genome-Atlas — Complete 37-Gene Mitochondrial Genome Atlas
All 37 human mitochondrial genes: 13 protein-coding + 22 tRNA + 2 rRNA
1,480-patient aggregate cohort (37 × 40, seeds 845–917)

Human mitochondrial DNA (mtDNA, rCRS 16,569 bp) encodes exactly 37 genes:
  13 protein-coding OXPHOS subunits (Complexes I, III, IV, V — mtDNA encodes NO Complex II subunits)
  22 transfer RNA genes  (required for translation of all 13 mt-encoded subunits)
  2  ribosomal RNA genes  (MT-RNR1 12S mt-SSU; MT-RNR2 16S mt-LSU)

PROTEIN-CODING GENE → OXPHOS COMPLEX MAP:
  Complex I  (NADH dehydrogenase, 45 subunits): 7 mtDNA subunits
             MT-ND1 · MT-ND2 · MT-ND3 · MT-ND4 · MT-ND4L · MT-ND5 · MT-ND6
  Complex II (succinate dehydrogenase, 4 subunits): 0 mtDNA subunits — ALL nuclear-encoded (SDHA/B/C/D)
  Complex III (cytochrome bc1, 11 subunits): 1 mtDNA subunit
             MT-CYB (sole mtDNA-encoded CIII structural subunit)
  Complex IV (cytochrome c oxidase, 13 subunits): 3 mtDNA subunits
             MT-CO1 · MT-CO2 · MT-CO3
  Complex V  (ATP synthase, 19 subunits): 2 mtDNA subunits
             MT-ATP6 · MT-ATP8

UNIVERSAL DRUG CONTRAINDICATIONS (all 37 mtDNA genes):
  ABSOLUTE CI: Metformin, VPA (valproate), Propofol (PRIS), Linezolid, Chloramphenicol
  ADDITIONAL: Amiodarone (MT-TT specific); Statins CAUTION (CoQ10); Aminoglycosides ABS CI (MT-RNR1)
  MANDATORY: Thiamine B1 empiric, Biotin empiric; GIR 6–8 mg/kg/min (never fast in CI/CIV disease)

WES LIMITATION (all 37 mtDNA genes):
  Whole-Exome Sequencing MISSES all 37 mtDNA genes — targeted mtDNA panel (whole-genome or dedicated)
  is mandatory for any suspected mitochondrial disease.
  L-strand genes additionally require dedicated L-strand QC window (MT-ND6 + 8 mt-tRNA L-strand genes).

COMMON LARGE DELETION (4977 bp KSS/PEO deletion):
  rCRS 8,470–13,447 — removes ATP8, ATP6, CO3, ND3, ND4L, ND4, ND5 partial, plus tRNA-Gly, tRNA-Arg,
  tRNA-His, tRNA-Ser2, tRNA-Leu2 — the most common single mtDNA deletion causing KSS/PEO.

COHORT: 37 × 40 = 1,480 patient slots (seeds 845–917, odd-numbered; one seed per gene)
"""

import random

SEED = 845
rng  = random.Random(SEED)

# ── All 37 mtDNA genes — authoritative table (rCRS coordinates) ──────────────
# gene_class: "protein" | "tRNA" | "rRNA"
# complex: CI / CIII / CIV / CV / "mt-SSU" / "mt-LSU" / "mt-tRNA" (translation only)
# strand: H (heavy/sense) | L (light/antisense)
MT_GENOME_GENES = [
    # ── rRNA genes (2) ────────────────────────────────────────────────────────
    {
        "gene": "MT-RNR1",  "gene_class": "rRNA",    "aa_or_nt": "954 nt",
        "strand": "H",      "rcrs_start": 648,       "rcrs_end": 1601,
        "complex": "mt-SSU (12S, 28S small subunit)",
        "omim_gene": 561000,
        "primary_disease": "AISNHL — Aminoglycoside-Induced Sensorineural Hearing Loss (m.1555A>G, m.1494C>T)",
        "oxphos_deficiency": "NONE — UNIQUE: no OXPHOS deficiency; isolated cochlear hair-cell aminoglycoside sensitivity",
        "hallmark": "HOMOPLASMIC (blood DNA sufficient); 100% penetrance ANY aminoglycoside dose; m.1555A>G 1:500 population",
        "key_ci": "Aminoglycosides ABSOLUTE CI (Gentamicin/Amikacin/Tobramycin/Streptomycin/Neomycin); cochlear implant excellent outcomes",
        "seed": 841,
    },
    {
        "gene": "MT-RNR2",  "gene_class": "rRNA",    "aa_or_nt": "1559 nt",
        "strand": "H",      "rcrs_start": 1671,      "rcrs_end": 3229,
        "complex": "mt-LSU (16S, 39S large subunit; PTC scaffold; Humanin ORF)",
        "omim_gene": 561010,
        "primary_disease": "Combined CI+CIII+CIV+CV / MIHH (m.2336T>C) / Cardiomyopathy (m.3260A>G) / LHON-like (m.2617G>A) / SNHL (m.3093G>A)",
        "oxphos_deficiency": "Combined CI+CIII+CIV+CV (CII normal); mt-translation fingerprint",
        "hallmark": "HETEROPLASMIC (muscle biopsy often required); Humanin 21aa neuroprotective microprotein within 16S rRNA ORF",
        "key_ci": "Metformin, VPA, Propofol, Linezolid ABS CI; Statins CAUTION; Ethambutol ABS CI (LHON-like); Tobacco/Alcohol AVOID",
        "seed": 843,
    },
    # ── Protein-coding genes — Complex I (7 subunits) ─────────────────────────
    {
        "gene": "MT-ND1",   "gene_class": "protein",  "aa_or_nt": "318 aa",
        "strand": "H",      "rcrs_start": 3307,       "rcrs_end": 4262,
        "complex": "CI (ND1 — P-module membrane arm, ND1-module core)",
        "omim_gene": 516000,
        "primary_disease": "LHON (m.3460G>A G103A) / Leigh / Combined CI Deficiency — Isolated CI (CII-CIV normal)",
        "oxphos_deficiency": "Isolated CI deficiency (5–20% residual CI; CII, CIII, CIV normal)",
        "hallmark": "LHON m.3460G>A second-most-common LHON variant; bilateral optic neuropathy young males; Idebenone/Raxone",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Linezolid ABS CI (blocks 7 ND-subunit translation); Ethambutol AVOID (optic)",
        "seed": 845,
    },
    {
        "gene": "MT-ND2",   "gene_class": "protein",  "aa_or_nt": "347 aa",
        "strand": "H",      "rcrs_start": 4470,       "rcrs_end": 5511,
        "complex": "CI (ND2 — P-module membrane arm, proximal pump module)",
        "omim_gene": 516001,
        "primary_disease": "LHON (m.4917A>G D150N) / Leigh / Combined CI Deficiency",
        "oxphos_deficiency": "Isolated CI deficiency; CII-CIII-CIV normal",
        "hallmark": "m.4917A>G third-most-common secondary LHON variant; synergistic with MT-ND4 or MT-ND6 primary LHON",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Linezolid ABS CI; Tobacco/Alcohol AVOID (LHON)",
        "seed": 847,
    },
    {
        "gene": "MT-ND3",   "gene_class": "protein",  "aa_or_nt": "115 aa",
        "strand": "H",      "rcrs_start": 10059,      "rcrs_end": 10404,
        "complex": "CI (ND3 — P-module membrane arm, smallest ND subunit)",
        "omim_gene": 516002,
        "primary_disease": "Leigh Syndrome (m.10191T>C T2A) / Leukodystrophy / Combined CI Deficiency",
        "oxphos_deficiency": "Isolated CI deficiency; severe Leigh in childhood; CII-CIII-CIV normal",
        "hallmark": "SMALLEST mtDNA protein subunit (115 aa / 13.1 kDa); m.10191T>C most common ND3 Leigh; brainstem+BG MRI",
        "key_ci": "Metformin ABS CI; VPA ABS CI; KD CONTRAINDICATED; GIR 6–8 mandatory; Linezolid ABS CI",
        "seed": 849,
    },
    {
        "gene": "MT-ND4L",  "gene_class": "protein",  "aa_or_nt": "98 aa",
        "strand": "H",      "rcrs_start": 10470,      "rcrs_end": 10766,
        "complex": "CI (ND4L — P-module membrane arm, paired with ND4)",
        "omim_gene": 516004,
        "primary_disease": "LHON (m.10663T>C V65A) / Combined CI Deficiency",
        "oxphos_deficiency": "Isolated CI deficiency; LHON with optic neuropathy; CII-CIII-CIV normal",
        "hallmark": "SMALLEST aa count among CI mtDNA subunits (98 aa); obligate heterodimer with ND4; 0 nt gap to ND4",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Ethambutol AVOID; Tobacco/Alcohol AVOID (LHON compounding)",
        "seed": 851,
    },
    {
        "gene": "MT-ND4",   "gene_class": "protein",  "aa_or_nt": "459 aa",
        "strand": "H",      "rcrs_start": 10760,      "rcrs_end": 12137,
        "complex": "CI (ND4 — P-module membrane arm, proton translocation helix)",
        "omim_gene": 516003,
        "primary_disease": "LHON (m.11778G>A R340H) / Combined CI Deficiency — m.11778 canonical LHON",
        "oxphos_deficiency": "Isolated CI deficiency; LHON m.11778G>A most common worldwide (~70% of LHON)",
        "hallmark": "m.11778G>A MOST COMMON LHON VARIANT worldwide (~70%); bilateral simultaneous optic neuropathy; Idebenone/Raxone",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Ethambutol ABS CI; Tobacco/Alcohol ABS AVOID; Idebenone LHON treatment",
        "seed": 853,
    },
    {
        "gene": "MT-ND5",   "gene_class": "protein",  "aa_or_nt": "603 aa",
        "strand": "H",      "rcrs_start": 12337,      "rcrs_end": 14148,
        "complex": "CI (ND5 — P-module membrane arm, LARGEST mtDNA protein subunit)",
        "omim_gene": 516005,
        "primary_disease": "MELAS/LHON-overlap (m.13513G>A D393N) / LHON (m.13708G>A A458T) / Leigh / Combined CI",
        "oxphos_deficiency": "Isolated CI deficiency; m.13513G>A MELAS/LHON-overlap with stroke-like episodes + optic",
        "hallmark": "LARGEST mtDNA protein subunit (603 aa / 67.9 kDa); m.13513G>A MELAS/LHON-overlap unique phenotype",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Ethambutol AVOID; Tobacco/Alcohol AVOID; Linezolid ABS CI",
        "seed": 855,
    },
    {
        "gene": "MT-ND6",   "gene_class": "protein",  "aa_or_nt": "174 aa",
        "strand": "L",      "rcrs_start": 14149,      "rcrs_end": 14673,
        "complex": "CI (ND6 — P-module membrane arm, ONLY L-strand protein-coding gene)",
        "omim_gene": 516006,
        "primary_disease": "LHON (m.14484T>C M64V) / Leigh / Combined CI Deficiency",
        "oxphos_deficiency": "Isolated CI deficiency; m.14484T>C third most common primary LHON; best spontaneous recovery rate",
        "hallmark": "ONLY L-strand protein-coding gene — NGS pitfall; reverse-complement mandatory; m.14484T>C 15% spontaneous visual recovery",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Ethambutol AVOID; Tobacco/Alcohol AVOID; L-STRAND NGS PITFALL",
        "seed": 857,
    },
    # ── Protein-coding genes — Complex III (1 subunit) ───────────────────────
    {
        "gene": "MT-CYB",   "gene_class": "protein",  "aa_or_nt": "380 aa",
        "strand": "H",      "rcrs_start": 14747,      "rcrs_end": 15887,
        "complex": "CIII (CYB — sole mtDNA-encoded CIII subunit; Qo+Qi sites, heme bL+bH, 8 TM helices)",
        "omim_gene": 516020,
        "primary_disease": "Isolated CIII Deficiency / Exercise Intolerance / Myoglobinuria / Leigh (high heteroplasmy)",
        "oxphos_deficiency": "Isolated CIII deficiency (CII normal; CI/CIV mild secondary in severe cases); myoglobinuria PATHOGNOMONIC",
        "hallmark": "ONLY mtDNA-encoded CIII structural subunit; all 10 other CIII subunits nuclear-encoded AR DDx; myoglobinuria 32%",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Propofol ABS CI (PRIS); Linezolid ABS CI; Chloramphenicol ABS CI",
        "seed": 859,
    },
    # ── Protein-coding genes — Complex IV (3 subunits) ───────────────────────
    {
        "gene": "MT-CO1",   "gene_class": "protein",  "aa_or_nt": "514 aa",
        "strand": "H",      "rcrs_start": 5904,       "rcrs_end": 7445,
        "complex": "CIV (COX1 — catalytic core, heme a + a3 + CuB; LARGEST CIV subunit)",
        "omim_gene": 516030,
        "primary_disease": "Isolated CIV Deficiency / CPEO / Leigh / Sideroblastic Anemia (m.6742C>T)",
        "oxphos_deficiency": "Isolated CIV (COX) deficiency; histochemistry COX-negative fibres; LARGEST CIV mtDNA subunit",
        "hallmark": "LARGEST CIV subunit (514 aa); catalytic core with both heme a centers and CuB; sideroblastic anemia rare",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Propofol ABS CI (PRIS); Linezolid ABS CI; Chloramphenicol ABS CI",
        "seed": 861,
    },
    {
        "gene": "MT-CO2",   "gene_class": "protein",  "aa_or_nt": "227 aa",
        "strand": "H",      "rcrs_start": 7586,       "rcrs_end": 8269,
        "complex": "CIV (COX2 — CuA binuclear center, electron entry from cytochrome c)",
        "omim_gene": 516040,
        "primary_disease": "Isolated CIV Deficiency / Leigh / Combined OXPHOS",
        "oxphos_deficiency": "Isolated CIV deficiency; CuA center mutations disrupt electron transfer from cyt c to COX1",
        "hallmark": "CuA binuclear copper center; binds 2 cytochrome c simultaneously; assembly requires COX20/COX18/COA6 chaperones",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Propofol ABS CI (PRIS); Linezolid ABS CI; Chloramphenicol ABS CI",
        "seed": 863,
    },
    {
        "gene": "MT-CO3",   "gene_class": "protein",  "aa_or_nt": "261 aa",
        "strand": "H",      "rcrs_start": 9207,       "rcrs_end": 9990,
        "complex": "CIV (COX3 — proton pathway, COX1 stabilizer; 7 TM helices)",
        "omim_gene": 516050,
        "primary_disease": "Isolated CIV Deficiency / CPEO / Leigh / Myoglobinuria",
        "oxphos_deficiency": "Isolated CIV deficiency; COX3 mutations destabilize COX1 catalytic core",
        "hallmark": "Structural scaffold for COX1; 7 TM helices surround catalytic core; shared boundary with MT-TG 5' end",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Propofol ABS CI (PRIS); Linezolid ABS CI; Chloramphenicol ABS CI",
        "seed": 865,
    },
    # ── Protein-coding genes — Complex V (2 subunits) ────────────────────────
    {
        "gene": "MT-ATP8",  "gene_class": "protein",  "aa_or_nt": "68 aa",
        "strand": "H",      "rcrs_start": 8366,       "rcrs_end": 8572,
        "complex": "CV (ATP8 — F0 stator stalk; 68 aa; SMALLEST mtDNA protein subunit)",
        "omim_gene": 516070,
        "primary_disease": "Combined CV/CI Deficiency / HCM / Leigh / Neuropathy (m.8528T>C frameshift)",
        "oxphos_deficiency": "Combined CV+CI deficiency; ATP8 frameshift disrupts F0-F1 coupling stalk assembly",
        "hallmark": "SMALLEST mtDNA protein subunit (68 aa); overlaps ATP6 by 43 nt (rCRS 8527-8572); frameshift affects BOTH",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Propofol ABS CI (PRIS); KD CONTRAINDICATED; GIR 6–8 mandatory",
        "seed": 867,
    },
    {
        "gene": "MT-ATP6",  "gene_class": "protein",  "aa_or_nt": "227 aa",
        "strand": "H",      "rcrs_start": 8527,       "rcrs_end": 9207,
        "complex": "CV (ATP6 — F0 a-subunit; proton channel; ATP synthesis rotor-stator coupling)",
        "omim_gene": 516060,
        "primary_disease": "NARP (m.8993T>G/C L156R/L156P) / Leigh / Combined CV Deficiency",
        "oxphos_deficiency": "Isolated/Combined CV deficiency; NARP m.8993 >90% heteroplasmy → Leigh; <90% → NARP neuropathy+RP",
        "hallmark": "NARP m.8993T>G most common ATP6 variant; heteroplasmy threshold: >90% Leigh-like; 70-90% NARP; <70% carrier",
        "key_ci": "Metformin ABS CI; VPA ABS CI; Propofol ABS CI (PRIS); KD CONTRAINDICATED; Idebenone Level C evidence",
        "seed": 869,
    },
    # ── tRNA genes (22) — summary table (full details in MT-tRNA-Atlas) ───────
    {
        "gene": "MT-TF",    "gene_class": "tRNA", "aa_or_nt": "71 nt",
        "strand": "H", "rcrs_start": 577,   "rcrs_end": 647,   "complex": "mt-tRNA (Phe; GAA anticodon)", "omim_gene": 590070,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "FIRST mt-tRNA; D-loop adjacent", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 799,
    },
    {
        "gene": "MT-TV",    "gene_class": "tRNA", "aa_or_nt": "69 nt",
        "strand": "H", "rcrs_start": 1602,  "rcrs_end": 1670,  "complex": "mt-tRNA (Val; UAC anticodon)", "omim_gene": 590105,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy / SNHL", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "Between 12S and 16S rRNA", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 801,
    },
    {
        "gene": "MT-TL1",   "gene_class": "tRNA", "aa_or_nt": "75 nt",
        "strand": "H", "rcrs_start": 3230,  "rcrs_end": 3304,  "complex": "mt-tRNA (Leu-UUR; UAA anticodon)", "omim_gene": 590050,
        "primary_disease": "MELAS (m.3243A>G) / MIDD", "oxphos_deficiency": "Pan-OXPHOS CI+CIII+CIV (stroke-like hallmark)", "hallmark": "m.3243A>G MOST COMMON mtDNA mutation; stroke-like episodes cross vascular territories", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; LEV preferred", "seed": 803,
    },
    {
        "gene": "MT-TI",    "gene_class": "tRNA", "aa_or_nt": "69 nt",
        "strand": "H", "rcrs_start": 4263,  "rcrs_end": 4331,  "complex": "mt-tRNA (Ile; GAU anticodon)", "omim_gene": 590045,
        "primary_disease": "Isolated HCM (m.4300A>G) — WITHOUT CPEO/Myopathy", "oxphos_deficiency": "Combined CI+CIV; HCM dominant", "hallmark": "m.4300A>G ISOLATED HCM without CPEO — unique among 22 mt-tRNAs", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; Annual Echo mandatory", "seed": 805,
    },
    {
        "gene": "MT-TQ",    "gene_class": "tRNA", "aa_or_nt": "69 nt",
        "strand": "L", "rcrs_start": 4329,  "rcrs_end": 4400,  "complex": "mt-tRNA (Gln; UUG anticodon; L-strand NGS pitfall)", "omim_gene": 590030,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "L-strand — reverse-complement NGS QC mandatory; FIRST L-strand mt-tRNA", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 807,
    },
    {
        "gene": "MT-TM",    "gene_class": "tRNA", "aa_or_nt": "68 nt",
        "strand": "H", "rcrs_start": 4402,  "rcrs_end": 4469,  "complex": "mt-tRNA (Met; CAU anticodon; initiator + elongator)", "omim_gene": 590065,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy / DUAL-FUNCTION (initiator+elongator)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "DUAL-FUNCTION: both initiator Met-tRNA AND elongator in mt-translation — unique", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 809,
    },
    {
        "gene": "MT-TW",    "gene_class": "tRNA", "aa_or_nt": "68 nt",
        "strand": "H", "rcrs_start": 5512,  "rcrs_end": 5579,  "complex": "mt-tRNA (Trp; UCA anticodon; UGA recoding Trp not Stop)", "omim_gene": 590095,
        "primary_disease": "Combined CI+CIV asymmetry — UGA recoding; CPEO / Myopathy", "oxphos_deficiency": "Combined CI+CIV; CI>CIV asymmetry due to Trp-rich ND-subunits", "hallmark": "UGA codon recoded as Trp (not Stop) in mt-code — CI preferentially affected (ND1/ND2/ND3 Trp-rich)", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 811,
    },
    {
        "gene": "MT-TA",    "gene_class": "tRNA", "aa_or_nt": "66 nt",
        "strand": "L", "rcrs_start": 5587,  "rcrs_end": 5655,  "complex": "mt-tRNA (Ala; UGC anticodon; L-strand NGS pitfall)", "omim_gene": 590000,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy (L-strand cluster)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "L-strand cluster MT-TA/TN/TC/TY (rCRS 5587–5891); compound deletion risk loses all 4 simultaneously", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI; L-strand NGS pitfall", "seed": 813,
    },
    {
        "gene": "MT-TN",    "gene_class": "tRNA", "aa_or_nt": "68 nt",
        "strand": "L", "rcrs_start": 5657,  "rcrs_end": 5729,  "complex": "mt-tRNA (Asn; GUU anticodon; L-strand NGS pitfall)", "omim_gene": 590010,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy (L-strand cluster)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "L-strand cluster MT-TA/TN/TC/TY; compound large-deletion risk", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI; L-strand NGS pitfall", "seed": 815,
    },
    {
        "gene": "MT-TC",    "gene_class": "tRNA", "aa_or_nt": "65 nt",
        "strand": "L", "rcrs_start": 5761,  "rcrs_end": 5826,  "complex": "mt-tRNA (Cys; GCA anticodon; L-strand NGS pitfall)", "omim_gene": 590020,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy (L-strand cluster)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "L-strand cluster MT-TA/TN/TC/TY; compound large-deletion risk", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI; L-strand NGS pitfall", "seed": 817,
    },
    {
        "gene": "MT-TY",    "gene_class": "tRNA", "aa_or_nt": "66 nt",
        "strand": "L", "rcrs_start": 5826,  "rcrs_end": 5891,  "complex": "mt-tRNA (Tyr; GUA anticodon; L-strand NGS pitfall)", "omim_gene": 590100,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy (L-strand cluster end)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "LAST gene in L-strand cluster MT-TA/TN/TC/TY; ends rCRS 5891", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI; L-strand NGS pitfall", "seed": 819,
    },
    {
        "gene": "MT-TS1",   "gene_class": "tRNA", "aa_or_nt": "59 nt",
        "strand": "L", "rcrs_start": 7445,  "rcrs_end": 7516,  "complex": "mt-tRNA (Ser-UCN; GCU anticodon; SHORTEST mt-tRNA; lacks D-arm)", "omim_gene": 590080,
        "primary_disease": "SNHL-dominant at LOW heteroplasmy (<40%) — DISTINCTIVE", "oxphos_deficiency": "Combined CI+CIV; SNHL manifests below threshold causing systemic disease", "hallmark": "SHORTEST mt-tRNA (59 nt); lacks D-arm; SNHL at low heteroplasmy distinctive among mt-tRNAs; L-strand NGS pitfall", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; cochlear implants excellent", "seed": 821,
    },
    {
        "gene": "MT-TD",    "gene_class": "tRNA", "aa_or_nt": "69 nt",
        "strand": "H", "rcrs_start": 7518,  "rcrs_end": 7585,  "complex": "mt-tRNA (Asp; GUC anticodon)", "omim_gene": 590015,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "Between MT-TS1 and MT-CO2; 0nt gap both sides — shared boundaries", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 823,
    },
    {
        "gene": "MT-TK",    "gene_class": "tRNA", "aa_or_nt": "70 nt",
        "strand": "H", "rcrs_start": 8295,  "rcrs_end": 8364,  "complex": "mt-tRNA (Lys; UUU anticodon; MERRF hallmark)", "omim_gene": 590060,
        "primary_disease": "MERRF (m.8344A>G / m.8356T>C) — Myoclonic Epilepsy + RRF + MSL", "oxphos_deficiency": "Pan-OXPHOS CI+CIV+CV; MERRF tau-m5U34 tRNA thiol modification loss", "hallmark": "m.8344A>G MERRF HALLMARK; MSL (Multiple Symmetrical Lipomatosis) DISTINCTIVE; myoclonic epilepsy", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; LEV preferred for myoclonic epilepsy", "seed": 825,
    },
    {
        "gene": "MT-TG",    "gene_class": "tRNA", "aa_or_nt": "68 nt",
        "strand": "H", "rcrs_start": 9991,  "rcrs_end": 10058, "complex": "mt-tRNA (Gly; GCC anticodon; GxxxG TM motif CIV)", "omim_gene": 590035,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy / Exercise Intolerance", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "GCC anticodon decodes GGC/GGU (GxxxG TM motifs abundant in COX subunits → CIV preferentially sensitive)", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 827,
    },
    {
        "gene": "MT-TR",    "gene_class": "tRNA", "aa_or_nt": "65 nt",
        "strand": "H", "rcrs_start": 10405, "rcrs_end": 10469, "complex": "mt-tRNA (Arg; UCG anticodon; superwobble CGN)", "omim_gene": 590005,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "UCG anticodon superwobble decodes ALL 4 CGN codons; KSS 4977bp deletion includes MT-TR", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 829,
    },
    {
        "gene": "MT-TH",    "gene_class": "tRNA", "aa_or_nt": "69 nt",
        "strand": "H", "rcrs_start": 12138, "rcrs_end": 12206, "complex": "mt-tRNA (His; GUG anticodon)", "omim_gene": 590040,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy (KSS deletion zone)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "Within KSS common 4977bp deletion rCRS 8470–13447; often deleted simultaneously with MT-TR/TS2/TL2", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 831,
    },
    {
        "gene": "MT-TS2",   "gene_class": "tRNA", "aa_or_nt": "59 nt",
        "strand": "H", "rcrs_start": 12207, "rcrs_end": 12265, "complex": "mt-tRNA (Ser-AGY; GCU anticodon; SHORTEST mt-tRNA shared with MT-TS1)", "omim_gene": 590085,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy / SNHL (KSS deletion zone)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "JOINTLY SHORTEST mt-tRNA (59 nt, shared with MT-TS1); lacks D-arm; within KSS 4977bp deletion", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 833,
    },
    {
        "gene": "MT-TL2",   "gene_class": "tRNA", "aa_or_nt": "71 nt",
        "strand": "H", "rcrs_start": 12266, "rcrs_end": 12336, "complex": "mt-tRNA (Leu-CUN; UAG anticodon)", "omim_gene": 590055,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy (KSS deletion zone)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "Within KSS 4977bp deletion; adjacent to MT-ND5 5' end; decodes CUN Leu codons (distinct from MT-TL1 UUR)", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI", "seed": 835,
    },
    {
        "gene": "MT-TE",    "gene_class": "tRNA", "aa_or_nt": "69 nt",
        "strand": "L", "rcrs_start": 14674, "rcrs_end": 14742, "complex": "mt-tRNA (Glu; UUC anticodon; L-strand NGS pitfall)", "omim_gene": 590025,
        "primary_disease": "MIDM (m.14709T>C) — Maternally Inherited Diabetes + Myopathy", "oxphos_deficiency": "Combined CI+CIV; MIDM phenotype without progressive ophthalmoplegia", "hallmark": "m.14709T>C MIDM HALLMARK; lean T2DM + myopathy; L-strand NGS pitfall; 0nt gap to MT-ND6 3' end", "key_ci": "Metformin ABS CI (double hit: CI + diabetes CI); VPA ABS CI; L-strand NGS pitfall", "seed": 837,
    },
    {
        "gene": "MT-TT",    "gene_class": "tRNA", "aa_or_nt": "66 nt",
        "strand": "H", "rcrs_start": 15888, "rcrs_end": 15953, "complex": "mt-tRNA (Thr; UGU anticodon; HCM highest rate 55–65%)", "omim_gene": 590090,
        "primary_disease": "Combined CI+CIV — CPEO / HCM (HIGHEST among all mt-tRNAs: 55–65%) / Myopathy", "oxphos_deficiency": "Combined CI+CIV; HCM 55–65% highest cardiomyopathy rate of all 22 mt-tRNAs", "hallmark": "HCM HIGHEST RATE (55–65%); AMIODARONE ABSOLUTE CI (mt-OXPHOS inhibitor); Annual Echo+ECG mandatory", "key_ci": "AMIODARONE ABS CI; Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; Beta-blocker for HCM", "seed": 839,
    },
    {
        "gene": "MT-TP",    "gene_class": "tRNA", "aa_or_nt": "69 nt",
        "strand": "L", "rcrs_start": 15956, "rcrs_end": 16023, "complex": "mt-tRNA (Pro; UGG anticodon; L-strand; LAST mt-tRNA)", "omim_gene": 590075,
        "primary_disease": "Combined CI+CIV — CPEO / Myopathy (last gene before D-loop)", "oxphos_deficiency": "Combined CI+CIV (CII normal)", "hallmark": "LAST mt-tRNA (rCRS 16023); adjacent to D-loop control region; L-strand NGS pitfall", "key_ci": "Metformin/VPA/Propofol/Linezolid/Chloramphenicol ABS CI; KD CI; L-strand NGS pitfall", "seed": 799,
    },
]

# ── Patient cohort simulation ─────────────────────────────────────────────────
def _sim_cohort():
    rng2 = random.Random(SEED)
    patients = []
    for i, g in enumerate(MT_GENOME_GENES):
        for pid in range(1, 41):
            het = rng2.uniform(35, 95)
            age = rng2.randint(2, 72)
            sex = rng2.choice(["M", "F"])
            patients.append({
                "gene": g["gene"],
                "gene_class": g["gene_class"],
                "patient_id": f"{g['gene']}-{pid:03d}",
                "age_at_dx": age,
                "sex": sex,
                "heteroplasmy_pct": round(het, 1) if g["gene_class"] in ("rRNA", "tRNA", "protein") else None,
                "oxphos_ci": g["oxphos_deficiency"],
            })
    return patients


# ── Aggregate statistics ──────────────────────────────────────────────────────
def _aggregate():
    total       = len(MT_GENOME_GENES) * 40  # 1,480
    by_class    = {"protein": 0, "tRNA": 0, "rRNA": 0}
    by_complex  = {}
    l_strand    = []
    h_strand    = []
    for g in MT_GENOME_GENES:
        by_class[g["gene_class"]] = by_class.get(g["gene_class"], 0) + 1
        cx = g["complex"].split(" ")[0]
        by_complex[cx] = by_complex.get(cx, 0) + 1
        if g["strand"] == "L":
            l_strand.append(g["gene"])
        else:
            h_strand.append(g["gene"])
    return {
        "total_genes": len(MT_GENOME_GENES),
        "total_patients": total,
        "by_class": {"protein_coding": by_class["protein"],
                     "tRNA": by_class["tRNA"],
                     "rRNA": by_class["rRNA"]},
        "by_complex": by_complex,
        "l_strand_genes": l_strand,
        "h_strand_genes": h_strand,
        "l_strand_ngs_pitfall_count": len(l_strand),
        "genome_bp": 16569,
        "genome_accession": "rCRS (GenBank J01415.2)",
    }


AGG = _aggregate()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    protein_genes = [g for g in MT_GENOME_GENES if g["gene_class"] == "protein"]
    trna_genes    = [g for g in MT_GENOME_GENES if g["gene_class"] == "tRNA"]
    rrna_genes    = [g for g in MT_GENOME_GENES if g["gene_class"] == "rRNA"]

    # Complex coverage table
    complex_coverage = {
        "CI":   {"name": "NADH Dehydrogenase", "subunits_total": 45, "mtDNA_subunits": 7,  "nuclear_subunits": 38, "genes": ["MT-ND1","MT-ND2","MT-ND3","MT-ND4","MT-ND4L","MT-ND5","MT-ND6"]},
        "CII":  {"name": "Succinate Dehydrogenase", "subunits_total": 4, "mtDNA_subunits": 0, "nuclear_subunits": 4,  "genes": [], "note": "ALL 4 subunits nuclear-encoded (SDHA/B/C/D) — NORMAL in mtDNA disease"},
        "CIII": {"name": "Cytochrome bc1", "subunits_total": 11, "mtDNA_subunits": 1, "nuclear_subunits": 10, "genes": ["MT-CYB"]},
        "CIV":  {"name": "Cytochrome c Oxidase", "subunits_total": 13, "mtDNA_subunits": 3, "nuclear_subunits": 10, "genes": ["MT-CO1","MT-CO2","MT-CO3"]},
        "CV":   {"name": "ATP Synthase", "subunits_total": 19, "mtDNA_subunits": 2, "nuclear_subunits": 17, "genes": ["MT-ATP8","MT-ATP6"]},
    }

    return {
        # Atlas identity
        "title":           "MT-Genome-Atlas — Complete 37-Gene Mitochondrial Genome Atlas",
        "subtitle":        "13 Protein-Coding + 22 tRNA + 2 rRNA | 1,480-Patient Aggregate Cohort (37×40)",
        "genome_size_bp":  16569,
        "genome_accession": "rCRS (GenBank J01415.2; NC_012920.1)",
        "wes_limitation":  "WES misses ALL 37 mtDNA genes — targeted mtDNA sequencing (whole-genome or dedicated mtDNA panel) mandatory",
        "inheritance":     "MATERNAL — all 37 genes transmitted maternally; heteroplasmy variable except MT-RNR1 (homoplasmic)",
        "btbgd_slc19a3_exclusion": "MANDATORY exclusion of SLC19A3 (BTBGD) in all Leigh-like presentations before diagnosing any mtDNA disease",

        # Gene class breakdown
        "gene_classes": {
            "protein_coding": {
                "count": AGG["by_class"]["protein_coding"],
                "note": "7 × CI + 1 × CIII + 3 × CIV + 2 × CV; Complex II (CII) entirely nuclear-encoded",
                "genes": [g["gene"] for g in protein_genes],
                "no_complex_ii_note": "mtDNA encodes ZERO Complex II subunits — CII ALWAYS NORMAL in primary mtDNA disease (use as internal reference)"
            },
            "tRNA": {
                "count": AGG["by_class"]["tRNA"],
                "note": "22 genes required for translation of all 13 protein-coding mtDNA OXPHOS subunits",
                "h_strand": [g["gene"] for g in trna_genes if g["strand"] == "H"],
                "l_strand_ngs_pitfall": [g["gene"] for g in trna_genes if g["strand"] == "L"],
                "see_also": "MT-tRNA-Atlas for complete 22-gene tRNA genome-wide analysis",
            },
            "rRNA": {
                "count": AGG["by_class"]["rRNA"],
                "note": "MT-RNR1 (12S, mt-SSU 28S): AISNHL, NO OXPHOS; MT-RNR2 (16S, mt-LSU 39S): Combined CI+CIII+CIV+CV + Humanin",
                "genes": [g["gene"] for g in rrna_genes],
            },
        },

        # OXPHOS complex coverage
        "oxphos_complex_coverage": complex_coverage,

        # Strand distribution
        "strand_distribution": {
            "H_strand": {
                "count": len(AGG["h_strand_genes"]),
                "note": "Heavy strand (sense); standard NGS coverage",
                "genes": AGG["h_strand_genes"],
            },
            "L_strand": {
                "count": len(AGG["l_strand_genes"]),
                "note": "Light strand (antisense); NGS pitfall — reverse-complement QC mandatory for all 9 L-strand genes (MT-ND6 + 8 mt-tRNAs)",
                "genes": AGG["l_strand_genes"],
            },
        },

        # Cohort summary
        "cohort": {
            "total_patients": 1480,
            "genes_included": 37,
            "patients_per_gene": 40,
            "seed_range": "845–917 (odd-numbered; gene-specific seeds)",
            "note": "Aggregate of 37 gene-specific 40-patient cohorts",
        },

        # Common deletion
        "common_deletion_4977bp": {
            "rCRS_range": "8,470–13,447",
            "size_bp": 4977,
            "names": "KSS deletion / PEO common deletion / mtDNA4977",
            "genes_removed_protein": ["MT-ATP8","MT-ATP6","MT-CO3","MT-ND3","MT-ND4L","partial-MT-ND4","partial-MT-ND5"],
            "genes_removed_tRNA": ["MT-TG","MT-TR","MT-TH","MT-TS2","MT-TL2"],
            "total_genes_affected": 12,
            "phenotypes": "KSS (>50% heteroplasmy) — ptosis + CPEO + cardiomyopathy + pigmentary retinopathy; PEO (<50%) — ptosis + CPEO alone",
        },

        # Universal drug CIs (apply to all 37 genes)
        "universal_absolute_ci": [
            {"drug": "Metformin",          "mechanism": "Complex I inhibitor; fatal lactic acidosis in mtDNA disease", "applies_to": "ALL 37 mtDNA genes"},
            {"drug": "VPA (Valproate)",    "mechanism": "β-oxidation inhibitor + CI inhibition + CoA sequestration + POLG inhibition", "applies_to": "ALL 37 mtDNA genes"},
            {"drug": "Propofol",           "mechanism": "PRIS — propofol infusion syndrome; uncouples OXPHOS; fatal myocardial failure", "applies_to": "ALL 37 mtDNA genes"},
            {"drug": "Linezolid",          "mechanism": "Inhibits mt-23S rRNA (in MT-RNR2); blocks mt-ribosome → reduces all 13 OXPHOS subunits", "applies_to": "ALL 37 mtDNA genes"},
            {"drug": "Chloramphenicol",    "mechanism": "Inhibits mt-ribosome peptidyl transferase; severe mt-translation failure", "applies_to": "ALL 37 mtDNA genes"},
        ],
        "additional_ci_by_class": [
            {"drug": "Aminoglycosides",    "mechanism": "Cochlear hair-cell aminoglycoside sensitivity via 12S rRNA m.1555A>G", "applies_to": "MT-RNR1 ONLY — ABSOLUTE CI"},
            {"drug": "Ethambutol",         "mechanism": "Optic neuropathy compounding; CI in any LHON or LHON-like gene", "applies_to": "MT-ND1/ND2/ND4/ND4L/ND5/ND6, MT-RNR2"},
            {"drug": "Amiodarone",         "mechanism": "mt-OXPHOS inhibitor; highest risk in HCM (MT-TT 55–65% HCM rate)", "applies_to": "MT-TT ABSOLUTE CI; caution all mt-tRNA"},
            {"drug": "KD (Ketogenic Diet)","mechanism": "Impairs β-oxidation + anaplerosis; OXPHOS-dependent tissues", "applies_to": "Protein-coding OXPHOS + tRNA genes; KD CI for CI/CIV/CV disease"},
        ],
        "universal_mandatory": [
            "Thiamine (B1) empiric — PDH + αKGDH cofactor; pending diagnosis",
            "Biotin empiric — BTBGD/SLC19A3 exclusion until ruled out",
            "GIR 6–8 mg/kg/min — never fast in OXPHOS disease (CI/CIV/CV dominant)",
            "Levetiracetam (LEV) preferred AED — renal excretion; no CYP450; no mt toxicity",
            "BTBGD/SLC19A3 exclusion MANDATORY in all Leigh-like presentations",
            "WES MANDATORY note: targeted mtDNA panel required; WES misses all 37 genes",
        ],

        # Key clinical phenotypes across the genome
        "hallmark_phenotypes": {
            "MELAS":       {"gene": "MT-TL1", "variant": "m.3243A>G", "note": "Most common pathogenic mtDNA mutation; stroke-like episodes cross vascular territories; CI+CIII+CIV pan-OXPHOS"},
            "MERRF":       {"gene": "MT-TK",  "variant": "m.8344A>G", "note": "Myoclonic epilepsy + RRF + MSL; CI+CIV+CV pan-OXPHOS; MSL distinctive"},
            "LHON":        {"gene": "MT-ND4", "variant": "m.11778G>A (70% of LHON)", "note": "Bilateral simultaneous optic neuropathy young males; Idebenone/Raxone"},
            "AISNHL":      {"gene": "MT-RNR1","variant": "m.1555A>G (1:500 population)", "note": "Permanent deafness any aminoglycoside dose; newborn screening implemented UK/China/Europe; NO OXPHOS deficiency"},
            "NARP_Leigh":  {"gene": "MT-ATP6","variant": "m.8993T>G/C", "note": "Heteroplasmy threshold: >90% Leigh; 70–90% NARP; <70% carrier; CV deficiency"},
            "KSS":         {"gene": "Common 4977bp deletion", "variant": "rCRS 8,470–13,447", "note": "12 mtDNA genes deleted; KSS triad: CPEO + cardiomyopathy + pigmentary retinopathy"},
            "Isolated_HCM":{"gene": "MT-TI",  "variant": "m.4300A>G",  "note": "HCM without CPEO/Myopathy — unique phenotype; annual Echo mandatory"},
            "MIDM":        {"gene": "MT-TE",  "variant": "m.14709T>C", "note": "Maternally inherited diabetes + myopathy; lean T2DM phenotype; metformin double-CI"},
            "MIHH":        {"gene": "MT-RNR2","variant": "m.2336T>C",  "note": "Maternally inherited hypertension + hypercholesterolaemia; near-homoplasmic; statins caution"},
            "Myoglobinuria":{"gene": "MT-CYB","variant": "Multiple Qo/Qi site mutations", "note": "Isolated CIII deficiency; myoglobinuria PATHOGNOMONIC; exercise intolerance dominant"},
        },
    }


def get_breakdown():
    """Per-gene table: all 37 genes with class, complex, rCRS, OXPHOS, hallmark, drug CI."""
    rows = []
    for g in MT_GENOME_GENES:
        rows.append({
            "gene":            g["gene"],
            "gene_class":      g["gene_class"],
            "size":            g["aa_or_nt"],
            "strand":          g["strand"],
            "rcrs_start":      g["rcrs_start"],
            "rcrs_end":        g["rcrs_end"],
            "complex":         g["complex"],
            "omim_gene":       g["omim_gene"],
            "primary_disease": g["primary_disease"],
            "oxphos":          g["oxphos_deficiency"],
            "hallmark":        g["hallmark"],
            "key_ci":          g["key_ci"],
            "ngs_pitfall":     (g["strand"] == "L"),
        })

    # Class-level summary stats
    class_summary = {
        "protein_coding": {
            "count": sum(1 for g in MT_GENOME_GENES if g["gene_class"] == "protein"),
            "complexes": {"CI": 7, "CII": 0, "CIII": 1, "CIV": 3, "CV": 2},
            "oxphos_note": "CII entirely nuclear-encoded — always NORMAL in primary mtDNA protein-coding disease",
        },
        "tRNA": {
            "count": sum(1 for g in MT_GENOME_GENES if g["gene_class"] == "tRNA"),
            "h_strand": sum(1 for g in MT_GENOME_GENES if g["gene_class"] == "tRNA" and g["strand"] == "H"),
            "l_strand_ngs_pitfall": sum(1 for g in MT_GENOME_GENES if g["gene_class"] == "tRNA" and g["strand"] == "L"),
        },
        "rRNA": {
            "count": sum(1 for g in MT_GENOME_GENES if g["gene_class"] == "rRNA"),
            "mt_rnr1_note": "NO OXPHOS deficiency — isolated cochlear aminoglycoside sensitivity",
            "mt_rnr2_note": "Combined CI+CIII+CIV+CV; Humanin neuroprotective microprotein",
        },
    }

    # Genomic span stats
    all_starts = [g["rcrs_start"] for g in MT_GENOME_GENES]
    all_ends   = [g["rcrs_end"]   for g in MT_GENOME_GENES]
    genome_stats = {
        "first_gene": MT_GENOME_GENES[0]["gene"],
        "first_gene_rcrs_start": min(all_starts),
        "last_gene": "MT-TP (tRNA-Pro, rCRS 16023)",
        "last_gene_rcrs_end": max(all_ends),
        "non_coding_regions": [
            {"region": "D-loop / control region", "rCRS": "16024–576 (wrapping)", "function": "Replication origin (OH) + transcription promoter"},
            {"region": "OL (replication origin L-strand)", "rCRS": "~5721–5798", "function": "L-strand replication origin within L-strand tRNA cluster"},
        ],
        "total_genes": 37,
        "genome_bp_total": 16569,
    }

    return {
        "gene_table": rows,
        "class_summary": class_summary,
        "genome_stats": genome_stats,
        "kss_deletion_genes": [
            "MT-ATP8","MT-ATP6","MT-CO3","MT-ND3","MT-ND4L","MT-ND4(partial)","MT-ND5(partial)",
            "MT-TG","MT-TR","MT-TH","MT-TS2","MT-TL2",
        ],
        "total_gene_count": len(rows),
    }


def get_definitions():
    return {
        "atlas_scope": "Complete 37-gene atlas of the human mitochondrial genome — all protein-coding, tRNA, and rRNA genes",
        "key_concepts": [
            {"term": "mtDNA (mitochondrial DNA)",
             "definition": "Circular double-stranded DNA ~16.6 kb (rCRS 16,569 bp) located in mitochondria; encodes 37 genes: 13 protein-coding OXPHOS subunits, 22 tRNA, 2 rRNA; maternally inherited; present in hundreds–thousands of copies per cell (polyplasmy)"},
            {"term": "Heteroplasmy",
             "definition": "Coexistence of two mtDNA populations (wild-type and mutant) within a cell/tissue; heteroplasmy threshold varies per gene and per organ; blood heteroplasmy may underestimate tissue (muscle biopsy often required for protein-coding and most tRNA genes)"},
            {"term": "Homoplasmy",
             "definition": "Only one mtDNA sequence in all copies; MT-RNR1 variants (m.1555A>G, m.1494C>T) are typically homoplasmic — blood DNA diagnostic, no muscle biopsy needed"},
            {"term": "OXPHOS (Oxidative Phosphorylation)",
             "definition": "5-complex electron transport chain embedded in inner mitochondrial membrane; Complexes I–V; ATP synthesis coupled to proton gradient; CII entirely nuclear-encoded — CII always normal in primary mtDNA disease (use as internal reference)"},
            {"term": "Complex II (CII) — diagnostic key",
             "definition": "SDHA/SDHB/SDHC/SDHD: ALL 4 nuclear-encoded; CII activity NORMAL in all primary mtDNA mutations — a reduced CII with normal CI/CIV should prompt nuclear gene investigation, not mtDNA"},
            {"term": "mt-tRNA Fingerprint (Combined CI+CIV)",
             "definition": "20 of 22 mt-tRNA genes (and MT-RNR2) show combined CI+CIV deficiency with CII normal — the hallmark mt-translation biochemical fingerprint reflecting reduced translation of both ND subunits (CI) and COX subunits (CIV)"},
            {"term": "Threshold Effect",
             "definition": "Minimum heteroplasmy required to produce OXPHOS disease varies by gene and tissue; generally >60% blood (>80% muscle) triggers biochemical deficiency; MT-ATP6 m.8993: >90% → Leigh, 70–90% → NARP, <70% → carrier"},
            {"term": "KSS (Kearns-Sayre Syndrome)",
             "definition": "Multi-system disease from large single-deletion (most commonly 4977 bp); triad: CPEO + pigmentary retinopathy + cardiomyopathy; onset <20 years; cardiac conduction block requires pacemaker; 12 mtDNA genes deleted in common deletion"},
            {"term": "LHON (Leber Hereditary Optic Neuropathy)",
             "definition": "Primary mutations in CI subunits MT-ND1/ND2/ND4/ND4L/ND5/ND6; m.11778G>A (MT-ND4) 70% worldwide; bilateral simultaneous optic neuropathy; males >> females (penetrance modifier); Idebenone/Raxone only approved treatment"},
            {"term": "MELAS",
             "definition": "Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke-like Episodes; m.3243A>G (MT-TL1) most common mtDNA mutation; stroke-like episodes cross vascular territories (NOT arterial ischaemic); pan-OXPHOS CI+CIII+CIV"},
            {"term": "MERRF",
             "definition": "Myoclonic Epilepsy with Ragged-Red Fibres; m.8344A>G (MT-TK); pan-OXPHOS CI+CIV+CV; MSL (Multiple Symmetrical Lipomatosis) DISTINCTIVE; myoclonic epilepsy hallmark"},
            {"term": "NARP",
             "definition": "Neuropathy, Ataxia, Retinitis Pigmentosa; m.8993T>G/C (MT-ATP6); heteroplasmy 70–90% → NARP phenotype; >90% → Leigh syndrome; Complex V deficiency"},
            {"term": "Ragged-Red Fibres (RRF)",
             "definition": "Mitochondrial proliferation on modified Gomori trichrome of muscle biopsy; hallmark of mt-translation disease (tRNA + rRNA genes); less prominent in protein-coding subunit mutations; COX-negative fibres on COX/SDH histochemistry"},
            {"term": "NGS L-strand pitfall",
             "definition": "9 mtDNA genes are L-strand encoded (MT-ND6 + 8 mt-tRNAs); standard H-strand oriented read mapping inadequately covers these regions; reverse-complement QC window mandatory — missed variants in L-strand genes if not analyzed separately"},
            {"term": "Humanin (MT-RNR2)",
             "definition": "21-amino-acid neuroprotective microprotein (HN) encoded within the MT-RNR2 (16S rRNA) ORF at rCRS 2706–2768; HN inhibits Bax-mediated apoptosis, reduces Aβ toxicity; aging-related decline; independent reading frame within 16S rRNA"},
        ],
        "drug_definitions": [
            {"term": "Metformin",         "definition": "ABSOLUTE CI (ALL 37 mtDNA genes) — biguanide CI inhibitor; biguanide accumulation causes fatal lactic acidosis even at standard doses in any OXPHOS disease"},
            {"term": "VPA (Valproate)",   "definition": "ABSOLUTE CI (ALL 37 mtDNA genes) — inhibits β-oxidation + CI; sequester CoA; inhibits POLG → mtDNA depletion; use LEV instead for all mtDNA epilepsy"},
            {"term": "Propofol (PRIS)",   "definition": "ABSOLUTE CI (ALL 37 mtDNA genes) — propofol infusion syndrome decouples OXPHOS; fatal rhabdomyolysis + myocardial failure; use ketamine or volatile agents"},
            {"term": "Linezolid",         "definition": "ABSOLUTE CI (ALL 37 mtDNA genes) — inhibits mt-23S rRNA (within MT-RNR2); blocks mt-ribosome → reduces synthesis of ALL 13 mtDNA-encoded OXPHOS subunits"},
            {"term": "Chloramphenicol",   "definition": "ABSOLUTE CI (ALL 37 mtDNA genes) — inhibits mt-ribosome peptidyl transferase center (in MT-RNR2); severe mt-translation failure"},
            {"term": "Aminoglycosides",   "definition": "ABSOLUTE CI (MT-RNR1 specifically) — gentamicin/amikacin/tobramycin bind 12S rRNA Helix 44; ANY single dose causes permanent cochlear hair-cell death in m.1555A>G or m.1494C>T carriers (1:500 population); use piperacillin-tazobactam / cefepime / meropenem instead"},
            {"term": "Ethambutol",        "definition": "ABSOLUTE CI (LHON genes: MT-ND1/ND2/ND4/ND4L/ND5/ND6 + MT-RNR2 m.2617G>A) — compounding optic neuropathy; never use in LHON or LHON-like presentations"},
            {"term": "Amiodarone",        "definition": "ABSOLUTE CI (MT-TT specifically — 55–65% HCM rate) — mt-OXPHOS inhibitor; worsen cardiac compromise; use beta-blocker for mt-cardiomyopathy"},
            {"term": "Idebenone (Raxone)","definition": "Short-chain benzoquinone; approved for LHON (MT-ND4 m.11778G>A primary); transfers electrons from NADH to cytochrome c bypassing damaged CI; Level B evidence; best result <1 year onset"},
            {"term": "Levetiracetam (LEV)","definition": "PREFERRED AED (ALL 37 mtDNA genes with epilepsy) — renal excretion; no CYP450 induction; no mitochondrial toxicity; Level C evidence"},
            {"term": "CoQ10 (Ubiquinol)", "definition": "Endogenous electron carrier between CI/CII and CIII; supplemented in mtDNA disease; Level C evidence; statins reduce CoQ10 (CAUTION in MT-RNR2 MIHH + statin use)"},
            {"term": "Thiamine (B1)",     "definition": "MANDATORY EMPIRIC (ALL 37 mtDNA genes): PDH + αKGDH cofactor; empiric treatment pending definitive diagnosis; BTBGD/SLC19A3 must be co-excluded"},
            {"term": "Biotin",            "definition": "MANDATORY EMPIRIC (ALL 37 mtDNA genes): BTBGD/SLC19A3 Leigh-like mimic excluded with biotin-thiamine treatment; withhold neither until SLC19A3 ruled out"},
        ],
        "references": [
            {"ref": "Anderson et al. 1981 Nature",         "citation": "Anderson S et al. (1981) — Sequence and organization of the human mitochondrial genome — original rCRS publication; Nature 290:457–465"},
            {"ref": "Andrews et al. 1999 Nat Genet",       "citation": "Andrews RM et al. (1999) — Reanalysis and revision of the Cambridge reference sequence for human mitochondrial DNA (rCRS); Nat Genet 23:147"},
            {"ref": "DiMauro & Schon 2003 NEJM",          "citation": "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases — definitive review of all mtDNA gene classes; NEJM 348:2656–2668"},
            {"ref": "Gorman et al. 2016 Nat Rev Dis Primers","citation": "Gorman GS et al. (2016) — Mitochondrial diseases — comprehensive epidemiology and genetics; Nat Rev Dis Primers 2:16080"},
            {"ref": "Schaefer et al. 2008 Ann Neurol",     "citation": "Schaefer AM et al. (2008) — Prevalence of mitochondrial disease in adults, north east England; Ann Neurol 63:35–43"},
            {"ref": "Wallace et al. 1988 Science (LHON)",  "citation": "Wallace DC et al. (1988) — Mitochondrial DNA mutation associated with Leber's hereditary optic neuropathy — first mtDNA disease variant; Science 242:1427–1430"},
            {"ref": "Goto et al. 1990 Nature (MELAS)",    "citation": "Goto Y et al. (1990) — A mutation in the tRNA(Leu)(UUR) gene associated with MELAS; Nature 348:651–653"},
            {"ref": "Shoffner et al. 1990 Cell (MERRF)",  "citation": "Shoffner JM et al. (1990) — Myoclonic epilepsy and ragged-red fiber disease (MERRF) — m.8344A>G; Cell 61:931–937"},
            {"ref": "Holt et al. 1988 Nature (KSS del)",  "citation": "Holt IJ et al. (1988) — Deletions of muscle mitochondrial DNA in patients with mitochondrial myopathies; Nature 331:717–719"},
            {"ref": "Prezant et al. 1993 Nat Genet",      "citation": "Prezant TR et al. (1993) — Mitochondrial ribosomal RNA mutation associated with both antibiotic-induced and non-syndromic deafness — MT-RNR1 m.1555A>G; Nat Genet 4:289–294"},
            {"ref": "Guo et al. 2003 Nature (Humanin)",   "citation": "Guo B et al. (2003) — Humanin peptide suppresses apoptosis by interfering with Bax activation; Nature 423:456–461"},
            {"ref": "Chinnery & Hudson 2013 Br Med Bull", "citation": "Chinnery PF, Hudson G (2013) — Mitochondrial genetics; Br Med Bull 106:135–159 — comprehensive mtDNA genetics review"},
            {"ref": "MITOMAP 2024 (mtDNA disease database)","citation": "MITOMAP (2024) — A human mitochondrial genome database; available at mitomap.org — mutation/phenotype reference for all 37 genes"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== MT-GENOME ATLAS OVERVIEW ===")
    ov = get_overview()
    print(json.dumps(ov, indent=2)[:3000])
    print("\n=== BREAKDOWN — first 3 genes ===")
    bd = get_breakdown()
    print(json.dumps(bd["gene_table"][:3], indent=2))
    print(f"\nTotal genes in table: {len(bd['gene_table'])}")
    print(f"Class summary: {json.dumps(bd['class_summary'], indent=2)}")
