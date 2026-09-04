#!/usr/bin/env python3
"""CoQ10-Biosynthesis-Atlas — Complete 10-Gene Primary Coenzyme Q10 (Ubiquinone) Deficiency Atlas
PDSS1 · PDSS2 · COQ2 · COQ4 · COQ5 · COQ6 · COQ7 · COQ8A (ADCK3) · COQ8B (ADCK4) · COQ9
400-patient aggregate cohort (10 × 40, seeds 760–769)

Coenzyme Q10 (Ubiquinone-10, CoQ10) facts:
  - Lipophilic electron/proton carrier in the IMM — sole mobile ETC carrier between CI/CII → CIII
  - Reduced form: QH2 (ubiquinol, antioxidant); Oxidised form: Q (ubiquinone, electron acceptor)
  - Biosynthesis: de novo from para-hydroxybenzoate (pHBA) + decaprenyl-PP (10-isoprene units)
  - Step 1: PDSS1/PDSS2 heterotetramers produce decaprenyl-PP (the isoprenoid 'tail')
  - Step 2: COQ2 attaches pHBA to the tail (4-hydroxybenzoate:polyprenyl transferase)
  - Steps 3-8: COQ4/5/6/7/9 (+ mtDNA-encoded components) complete ring modifications
  - COQ8A/COQ8B are atypical kinases (ADCK family) — activate the COQ enzyme complex
  - CIII substrate: QH2 donates 2e⁻ to CIII (bc1 complex) in the Q cycle
  - 'Q-pool': ~40% ETC electron flux routed through CoQ10
  - CoQ10 ALWAYS NORMAL in isolated CI, CIII, CIV, or CV mutations → powerful internal reference
  - Primary CoQ10 deficiency: AR mutations in CoQ10 biosynthesis genes → ALL 5 ETC complexes reduced
  - Secondary CoQ10 deficiency: seen in ETFDH (MADD), ACAD9, statin therapy, ageing

ATLAS SCOPE (10 nuclear genes — all steps of CoQ10 biosynthesis):
  Decaprenyl-PP Synthesis (isoprene tail):
    PDSS1 (subunit 1), PDSS2 (subunit 2)
  Benzoate Ring Attachment:
    COQ2
  Ring Modification / COQ Complex (scaffold + oxygenation + methylation + hydroxylation):
    COQ4, COQ5, COQ6, COQ7, COQ9
  COQ Complex Activation (atypical kinases):
    COQ8A (ADCK3), COQ8B (ADCK4)

PHENOTYPIC SPECTRUM (Primary CoQ10 Deficiency):
  - Nephrotic syndrome / SRNS (COQ2, COQ6, COQ8B)
  - Cerebellar ataxia (COQ8A/ADCK3 — ARCA2; PDSS1; COQ4)
  - Leigh syndrome / encephalopathy (COQ4, COQ7, COQ9)
  - SNHL (COQ6 70%) + nephrotic syndrome
  - Multisystem mitochondrial disease (PDSS2, COQ2)
  - Exercise intolerance + myopathy (COQ8A)
  - Infantile multisystem (COQ2 most common for infantile form)

KEY CLINICAL RULE: CoQ10 supplementation (10-30 mg/kg/day) may RESCUE primary CoQ10 deficiency
  — treatment window exists before irreversible organopathy, especially nephrotic syndrome

BIOCHEMICAL FINGERPRINT (Primary CoQ10 Deficiency):
  - CoQ10 level markedly reduced in muscle and/or fibroblasts (confirm in MUSCLE — plasma unreliable)
  - Multiple ETC complex deficiencies when CoQ10 critically low (CI, CIII, CIV activity reduced)
  - BUT: mild-moderate CoQ10 deficiency → isolated complex deficiency pattern may persist
  - CoQ10 biosynthesis flux assay (¹³C-pHBA pulse) most specific
  - Urinary HMGC / organic acids: methylglutaconate NOT typically elevated (unlike CV deficiency)
  - SRNS: podocyte CoQ10 synthesis critical → COQ2/COQ6/COQ8B most renal-dominant

COHORT: 10 × 40 = 400 patient slots (seeds 760–769; gene-specific seeds)
"""

import random

SEED = 770
rng  = random.Random(SEED)

# ── All 10 nuclear-encoded CoQ10 biosynthesis genes — authoritative table ─────
# gene_class: "decaprenyl_pp" | "ring_attachment" | "ring_modification" | "coq_kinase"
COQ_GENES = [
    # ── Decaprenyl-PP Synthesis ───────────────────────────────────────────────
    {
        "gene": "PDSS1", "alias": "DPS1 / hDPS1", "aa": "359 aa", "kDa": "40.0 kDa",
        "gene_class": "decaprenyl_pp", "pathway_step": "Step 1a — decaprenyl-PP tail synthesis (subunit 1)",
        "coq_function": "Solanesyl diphosphate synthase subunit 1; PDSS1/PDSS2 form obligate heterotetramer (α2β2); catalyses sequential condensation of isopentenyl-PP units → decaprenyl-PP (10-isoprene tail); PDSS1 provides chain-length specificity for CoQ10 (vs CoQ9 in rat); first step of de novo CoQ10 biosynthesis pathway in IMM",
        "omim_gene": 607429, "chromosome": "10p12.1", "seed": 760,
        "phenotype": "CoQ10 Deficiency Type 1 — Leigh syndrome + Nephrotic Syndrome + Deafness (PDSS1-type; Salviati 2005)",
        "disease": "CoQ10 deficiency type 1 (OMIM #607426) — PDSS1; heterotetramer disruption → decaprenyl-PP absent → all CoQ10 biosynthesis arrested; Leigh syndrome 65%; nephrotic syndrome 45%; SNHL 35%; multisystem; first PDSS1 report: Salviati 2005 AmJHumGenet",
        "disease_omim": 607426, "inheritance": "AR",
        "hallmark": "PDSS1 = FIRST STEP IN CoQ10 TAIL SYNTHESIS; PDSS1/PDSS2 heterotetramer obligate — LOF of either → complete decaprenyl-PP failure → no CoQ10 synthesis possible; most upstream gene in pathway; Leigh + nephrotic combination highly characteristic; CoQ10 supplementation MUST START EARLY (before irreversible podocyte injury); PDSS2 mutations more common (Salviati 2005 landmark)",
        "key_ddx": "vs PDSS2: PDSS2 is the β subunit of same heterotetramer; vs COQ2: COQ2 downstream (ring attachment); vs Alport syndrome: Alport is COL4A3/4/5 — no ETC defect, no Leigh; vs FSGS: FSGS is histologic pattern — check CoQ10 in ALL SRNS under age 10; vs mtDNA depletion (DGUOK): multi-complex but CoQ10 NORMAL in DGUOK",
        "founder_variant": "p.Arg321Cys (Saudi); p.Gln381Stop; p.Ile199Thr",
        "coq10_muscle_pct": 12, "srns_rate": 0.45, "snhl_rate": 0.35,
        "ataxia_rate": 0.20, "leigh_rate": 0.65, "myopathy_rate": 0.30,
        "exercise_intol_rate": 0.35, "lactic_ac_rate": 0.75,
        "hcm_rate": 0.20, "retinopathy_rate": 0.15,
        "coq10_response": "Yes — CoQ10 10-30 mg/kg/day; renal partial response if started early",
        "median_onset_months": 6,
    },
    {
        "gene": "PDSS2", "alias": "DLP1 / hDPS2", "aa": "399 aa", "kDa": "45.0 kDa",
        "gene_class": "decaprenyl_pp", "pathway_step": "Step 1b — decaprenyl-PP tail synthesis (subunit 2)",
        "coq_function": "Solanesyl diphosphate synthase subunit 2 (β chain of PDSS1/PDSS2 heterotetramer); PDSS2 is the catalytic subunit (contains the FARM and SARM motifs for isoprene condensation); Kagan 2006 first human PDSS2 disease report; kd/kd mouse model of CoQ10 deficiency with nephropathy (podocyte degeneration) and encephalopathy",
        "omim_gene": 610564, "chromosome": "6q21", "seed": 761,
        "phenotype": "CoQ10 Deficiency Type 2 — Leigh syndrome + Focal Segmental Glomerulosclerosis (PDSS2-type; Kagan 2006)",
        "disease": "CoQ10 deficiency type 2 (OMIM #610564) — PDSS2; Kagan 2006 NEJM landmark — child with Leigh + FSGS + CoQ10 ↓ in muscle; PDSS2 catalytic β subunit of PDSS1/PDSS2 heterotetramer; full pathway block → severe multi-complex deficiency; kd/kd mouse validates PDSS2→CoQ10→podocyte→FSGS pathway",
        "disease_omim": 614651, "inheritance": "AR",
        "hallmark": "PDSS2 = MOST COMMON PDSS GENE MUTATED IN HUMANS; Kagan 2006 NEJM first human case + kd/kd mouse first animal model; FSGS + Leigh combination pathognomonic for PDSS1/PDSS2; CoQ10 supplementation arrested renal progression in original Kagan case; PDSS2 mutations more frequent than PDSS1 (larger gene, more splice sites); FARM/SARM motifs in PDSS2 = catalytic prenyl condensation sites",
        "key_ddx": "vs PDSS1: PDSS1 = α subunit (chain-length specificity) vs PDSS2 = β (catalytic); vs COQ2: COQ2 downstream — ring not tail; vs NPHS2/podocin FSGS: podocin-FSGS = no ETC defect, no CoQ10 reduction; vs COQ8B (ADCK4 SRNS): COQ8B more pure renal (kinase domain); vs NPC (Niemann-Pick C1): storage disorder vs ETC",
        "founder_variant": "p.Arg329Gln (Pakistani); p.Lys77Met (Kagan 2006 original); p.Tyr288Stop",
        "coq10_muscle_pct": 10, "srns_rate": 0.55, "snhl_rate": 0.20,
        "ataxia_rate": 0.15, "leigh_rate": 0.70, "myopathy_rate": 0.35,
        "exercise_intol_rate": 0.30, "lactic_ac_rate": 0.80,
        "hcm_rate": 0.15, "retinopathy_rate": 0.10,
        "coq10_response": "Yes — CoQ10 10-30 mg/kg/day; renal response documented (Kagan 2006 NEJM)",
        "median_onset_months": 4,
    },
    # ── Ring Attachment ────────────────────────────────────────────────────────
    {
        "gene": "COQ2", "alias": "Para-hydroxybenzoate-polyprenyl transferase", "aa": "422 aa", "kDa": "47.7 kDa",
        "gene_class": "ring_attachment", "pathway_step": "Step 2 — 4-hydroxybenzoate:polyprenyl transferase (ring attachment to tail)",
        "coq_function": "Attaches 4-hydroxybenzoate (pHBA — the aromatic ring precursor, derived from tyrosine) to decaprenyl-PP tail via O-alkylation; creates the first complete CoQ intermediate (decaprenyl-4-hydroxybenzoate); IMM-anchored enzyme; COQ2 mutations most common cause of infantile primary CoQ10 deficiency; COQ2 also found in sporadic multiple system atrophy (MSA) association",
        "omim_gene": 609825, "chromosome": "4q21.23", "seed": 762,
        "phenotype": "CoQ10 Deficiency Type 1 — Infantile Encephalomyopathy + Nephrotic Syndrome + Retinopathy (COQ2-type; Quinzii 2006)",
        "disease": "CoQ10 deficiency type 1 (OMIM #607426) — COQ2; Quinzii 2006 Nature Genetics; MOST COMMON infantile primary CoQ10 deficiency gene; infantile encephalomyopathy (80%) + nephrotic syndrome (50%) + retinopathy/SNHL; MSA (sporadic neurodegenerative): COQ2 p.Val393Ala + Met128Val in Japanese MSA cohort (Mitsui 2013)",
        "disease_omim": 607426, "inheritance": "AR (Mendelian) / dominant MSA risk (population variant)",
        "hallmark": "COQ2 = MOST COMMONLY MUTATED IN INFANTILE PRIMARY CoQ10 DEFICIENCY; ring-tail junction enzyme — upstream failure completely abolishes CoQ10; nephrotic syndrome + infantile encephalopathy combination strongly suggests COQ2; COQ2 MSA link: sporadic MSA patients carry heterozygous COQ2 risk variants (Mitsui 2013 NEJM) — gain-of-MSA-risk not classic AR deficiency; CoQ10 supplementation (high-dose) may reverse nephrotic syndrome if started within 3 months",
        "key_ddx": "vs PDSS1/2: PDSS tail vs COQ2 ring attachment — both abolish CoQ10 but COQ2 more common infantile; vs COQ6 (SRNS + SNHL): COQ6 more prominent SNHL 70%, COQ2 more retinopathy; vs NPHS1/nephrin SRNS: no ETC defect; vs MSA (sporadic): COQ2 risk variant vs AR CoQ10 deficiency — different dosage; vs MECR (neonatal SRNS): MECR = mitochondrial enoyl-ACP reductase, not CoQ pathway",
        "founder_variant": "p.Arg387Stop (Italian, Quinzii 2006); p.Ser146Asn; p.Val393Ala + p.Met128Val (MSA, Japanese)",
        "coq10_muscle_pct": 8, "srns_rate": 0.50, "snhl_rate": 0.30,
        "ataxia_rate": 0.20, "leigh_rate": 0.55, "myopathy_rate": 0.40,
        "exercise_intol_rate": 0.35, "lactic_ac_rate": 0.80,
        "hcm_rate": 0.15, "retinopathy_rate": 0.40,
        "coq10_response": "Yes — CoQ10 10-30 mg/kg/day; renal response best if < 3 months steroid-resistant",
        "median_onset_months": 3,
    },
    # ── Ring Modification / COQ Complex ───────────────────────────────────────
    {
        "gene": "COQ4", "alias": "COQ4 / Uncharacterised LOC51181", "aa": "291 aa", "kDa": "33.2 kDa",
        "gene_class": "ring_modification", "pathway_step": "Step 3 — COQ complex scaffold / stabiliser (Coq4p group)",
        "coq_function": "COQ4 is the scaffold/anchor of the multi-enzyme COQ complex (COQ3-COQ9) within the IMM; does not perform direct catalysis but maintains the complex architecture; stabilises COQ5/COQ6/COQ7/COQ9 interactions; COQ4 LOF → COQ complex disassembly → multiple downstream modification steps fail; Lohman 2019 structural characterisation of yeast Coq4",
        "omim_gene": 612898, "chromosome": "9q34.11", "seed": 763,
        "phenotype": "CoQ10 Deficiency Type 7 — Neonatal Leigh Syndrome + HCM + Lactic Acidosis (COQ4-type; Salviati 2012 / Brea-Calvo 2015)",
        "disease": "CoQ10 deficiency type 7 (OMIM #616276) — COQ4; scaffold disruption → COQ complex failure → very low CoQ10; neonatal Leigh-like 75%; HCM 55%; severe neonatal lactic acidosis; Salviati 2012 first COQ4 human mutations; pure CoQ10 deficiency without structural CI/CII/CIII/CIV mutations",
        "disease_omim": 616276, "inheritance": "AR",
        "hallmark": "COQ4 = COQ COMPLEX MASTER SCAFFOLD — destruction of COQ4 abolishes the entire multi-enzyme complex; neonatal presentation with Leigh + HCM + severe acidosis; HCM 55% highest among ring-modification genes (similar to CV-deficiency HCM but CoQ10 path not CV); neonatal Leigh 75% is very high — strongly consider CoQ10 in neonatal Leigh with HCM before ordering WES; CoQ10 supplementation: response variable (scaffold assembly may not fully rescue)",
        "key_ddx": "vs TMEM70 (CV deficiency + 3-MGA): TMEM70 has 3-MGA + Roma founder; COQ4 no 3-MGA, HCM via different mechanism; vs SURF1 (CIV-Leigh): CIV markedly reduced in SURF1, CoQ10 NORMAL; vs PDSS1/2 upstream: PDSS → no tail; COQ4 → tail+ring present but complex scaffold absent; vs COQ7: COQ7 hydroxylase step; vs Pompe (GAA): GSD, no ETC defect",
        "founder_variant": "p.Arg104Cys (Brea-Calvo 2015); p.Gly249Val (Salviati 2012); p.Tyr246Cys",
        "coq10_muscle_pct": 15, "srns_rate": 0.15, "snhl_rate": 0.20,
        "ataxia_rate": 0.30, "leigh_rate": 0.75, "myopathy_rate": 0.40,
        "exercise_intol_rate": 0.30, "lactic_ac_rate": 0.85,
        "hcm_rate": 0.55, "retinopathy_rate": 0.15,
        "coq10_response": "Variable — CoQ10 supplementation; response depends on residual assembly; renal minimal",
        "median_onset_months": 1,
    },
    {
        "gene": "COQ5", "alias": "COQ5 / 2-methoxy-6-polyprenyl-1,4-benzoquinol methylase", "aa": "375 aa", "kDa": "41.3 kDa",
        "gene_class": "ring_modification", "pathway_step": "Step 5 — C-methylation of CoQ ring (SAM-dependent methyltransferase)",
        "coq_function": "SAM-dependent methyltransferase in the COQ complex; methylates position C-5 of the CoQ ring; conserved from yeast Coq5p; part of the multi-enzyme complex with COQ3/COQ4/COQ6/COQ7/COQ9; two methylation steps in CoQ10 ring (C-5 by COQ5; O-methylation by COQ3 at C-3 and C-4); COQ5 disruption → C-methylation step blocked → downstream ring modifications impaired",
        "omim_gene": 616359, "chromosome": "12q24.31", "seed": 764,
        "phenotype": "CoQ10 Deficiency Type 5 — Ataxia + Cognitive Decline + Myopathy (COQ5-type; Malicdan 2018)",
        "disease": "CoQ10 deficiency type 5 (OMIM #616359) — COQ5; Malicdan 2018 first human COQ5 disease; 4-month-old with cerebellar atrophy + myopathy + lactic acidosis; CoQ10 10-20% in muscle; exercise intolerance + progressive ataxia in adolescent/adult forms; rare — <20 patients reported",
        "disease_omim": 616359, "inheritance": "AR",
        "hallmark": "COQ5 = SAM-DEPENDENT C-5 METHYLTRANSFERASE — critical ring modification step; methyltransferase class (shared domain structure with LYRM7 in CIII); COQ5 disease favours neurologic (cerebellar atrophy, ataxia, myopathy) over renal; exercise intolerance in adolescent form; COQ10 supplementation may stabilise neurologic decline but not reverse established cerebellar atrophy; Malicdan 2018 first human disease",
        "key_ddx": "vs COQ8A (ADCK3/ARCA2): ARCA2 is the most common ataxia-CoQ10 gene; COQ5 rarer, younger onset; vs FRDA (frataxin): iron-sulfur cluster disorder, no CoQ10 reduction; vs COQ9: COQ9 has early infantile Leigh; COQ5 more adolescent ataxia; vs LYRM7 (CIII assembly): LYRM7 methyltransferase domain for CIII not CoQ; vs SCA types: SCA1/2/3 — no CoQ10 reduction",
        "founder_variant": "p.Arg97Cys (Malicdan 2018); p.Gly154Arg (methyltransferase motif); p.Ala303Val",
        "coq10_muscle_pct": 18, "srns_rate": 0.10, "snhl_rate": 0.15,
        "ataxia_rate": 0.75, "leigh_rate": 0.20, "myopathy_rate": 0.65,
        "exercise_intol_rate": 0.70, "lactic_ac_rate": 0.55,
        "hcm_rate": 0.10, "retinopathy_rate": 0.10,
        "coq10_response": "Partial — CoQ10 10-20 mg/kg/day; stabilises neurologic; established ataxia irreversible",
        "median_onset_months": 48,
    },
    {
        "gene": "COQ6", "alias": "COQ6 / UbiH-like flavoprotein monooxygenase", "aa": "447 aa", "kDa": "49.8 kDa",
        "gene_class": "ring_modification", "pathway_step": "Step 4 — C-5 hydroxylation of CoQ ring (FAD-dependent monooxygenase)",
        "coq_function": "FAD-dependent monooxygenase in the COQ complex; hydroxylates C-5 position of the CoQ ring intermediate; requires FAD as cofactor; COQ6 is the renal isoform-critical step (podocytes express high COQ6 for CoQ10-dependent antioxidant protection); COQ6 LOF → CoQ10 ↓ in podocytes → podocyte oxidative damage → SRNS; SNHL from stria vascularis CoQ10 depletion",
        "omim_gene": 614647, "chromosome": "14q24.3", "seed": 765,
        "phenotype": "CoQ10 Deficiency Type 6 — SRNS + Sensorineural Hearing Loss + Retinopathy (COQ6-type; Heeringa 2011)",
        "disease": "CoQ10 deficiency type 6 (OMIM #614651) — COQ6; Heeringa 2011 JCI — SRNS (60%) + SNHL (70%) combined hallmark; FAD-dependent ring hydroxylation failure → CoQ10 ↓ in podocytes + stria vascularis; SNHL 70% is highest among all CoQ10 genes; CoQ10 supplementation improves renal and hearing outcomes if started within months of SRNS onset",
        "disease_omim": 614651, "inheritance": "AR",
        "hallmark": "COQ6 = 'RENAL + HEARING' CoQ10 GENE; SRNS + SNHL 70% combination is near-pathognomonic for COQ6; FAD-dependent enzyme (unlike SAM-dependent COQ5); SNHL from stria vascularis CoQ10 depletion (cochlear — not DPOAE); SRNS from podocyte CoQ10 antioxidant failure — podocytes are uniquely vulnerable (high oxidative stress, mitochondria-rich); CoQ10 10-30 mg/kg/day within 3 months may prevent dialysis",
        "key_ddx": "vs COQ2 (SRNS + retinopathy): COQ2 retinopathy 40% > SNHL; COQ6 SNHL 70% > retinopathy; vs COQ8B (ADCK4 SRNS): COQ8B rare SNHL; vs Alport (SNHL + SRNS): COL4A3/4/5 — no ETC defect, no CoQ10 reduction; vs NPHS2/podocin FSGS: no CoQ10, no hearing; vs branchio-oto-renal syndrome (BOR): EYA1/SIX1/SIX5 — no ETC defect",
        "founder_variant": "p.Arg196Stop (Heeringa 2011 Dutch); p.Ser546Phe (FAD-binding); p.Gly255Arg",
        "coq10_muscle_pct": 14, "srns_rate": 0.60, "snhl_rate": 0.70,
        "ataxia_rate": 0.15, "leigh_rate": 0.25, "myopathy_rate": 0.20,
        "exercise_intol_rate": 0.20, "lactic_ac_rate": 0.60,
        "hcm_rate": 0.10, "retinopathy_rate": 0.25,
        "coq10_response": "Yes — CoQ10 10-30 mg/kg/day; renal + hearing response if started early (< 3 months SRNS)",
        "median_onset_months": 24,
    },
    {
        "gene": "COQ7", "alias": "COQ7 / MCLK1 / CLK-1 / CAT5", "aa": "217 aa", "kDa": "24.5 kDa",
        "gene_class": "ring_modification", "pathway_step": "Step 6 — C-5 hydroxylation (diiron monooxygenase) — penultimate step",
        "coq_function": "Diiron monooxygenase (not FAD-dependent unlike COQ6); catalyses C-5 hydroxylation of demethoxy-ubiquinone → 5-demethylubiquinol (penultimate biosynthesis step); COQ7 = CLK-1 in C. elegans (Ewbank 1997 — clk-1 mutation extends lifespan via CoQ8 accumulation of partially-hydroxylated intermediate); uses CoQ intermediate DMQ10 as substrate; small 217aa di-iron hydroxylase",
        "omim_gene": 601683, "chromosome": "16p12.3", "seed": 766,
        "phenotype": "CoQ10 Deficiency Type 8 — Infantile Leigh-like + Lactic Acidosis + Hypotonia + Spasticity (COQ7-type; Freyer 2015)",
        "disease": "CoQ10 deficiency type 8 (OMIM #616733) — COQ7; Freyer 2015 Brain; infantile Leigh-like with lactic acidosis; penultimate step → DMQ10 (partially hydroxylated) accumulates; COQ7 disease → DMQ10 in urine (specific biomarker); CoQ10 supplementation response: partial (DMQ10 accumulation can partially substitute); hypotonia + spasticity combination; < 30 patients reported",
        "disease_omim": 616733, "inheritance": "AR",
        "hallmark": "COQ7 = PENULTIMATE STEP ENZYME (diiron hydroxylase); DMQ10 ACCUMULATION is a specific biomarker of COQ7 deficiency (only gene where upstream intermediate accumulates visibly in urine/fibroblasts); C. elegans clk-1/COQ7 mutants → extended lifespan (DMQ8 partially substitutes for CoQ8); infantile severe presentation; hypotonia + spasticity unusual combination for Leigh; CoQ10 may work via DMQ10 partial bypass",
        "key_ddx": "vs COQ4 (neonatal Leigh + HCM): COQ4 has HCM 55%; COQ7 Leigh without HCM, with DMQ10 accumulation; vs SURF1 (CIV-Leigh): CIV enzyme markedly reduced, CoQ10 NORMAL; vs COQ9 (infantile Leigh): COQ9 accumulates RHB (a different intermediate); vs BTBGD (neonatal encephalopathy): SLC19A3 — biotin-thiamine responsive, no CoQ10 reduction",
        "founder_variant": "p.Lys122Asn (Freyer 2015 Swedish); p.Arg38Stop; p.His147Gln (di-iron site)",
        "coq10_muscle_pct": 20, "srns_rate": 0.10, "snhl_rate": 0.15,
        "ataxia_rate": 0.25, "leigh_rate": 0.70, "myopathy_rate": 0.50,
        "exercise_intol_rate": 0.40, "lactic_ac_rate": 0.80,
        "hcm_rate": 0.20, "retinopathy_rate": 0.10,
        "coq10_response": "Partial — CoQ10 10-20 mg/kg/day; DMQ10 partially functional; some metabolic improvement",
        "median_onset_months": 3,
    },
    {
        "gene": "COQ9", "alias": "COQ9 / FLJ22761", "aa": "315 aa", "kDa": "35.8 kDa",
        "gene_class": "ring_modification", "pathway_step": "Step 7 — COQ complex regulatory subunit + ring-hydroxylation coordination",
        "coq_function": "Regulatory/structural subunit of the COQ complex; COQ9 binds DMQ10 intermediate and COQ7; presents DMQ10 substrate to COQ7 for hydroxylation; CoQ9 crystal structure (Floyd 2016 PNAS) reveals lipid-binding groove that captures CoQ intermediates; COQ9 acts as a lipid-binding chaperone within the complex; RHB (rhodoquinone precursor) accumulates in COQ9 deficiency (specific biomarker)",
        "omim_gene": 612837, "chromosome": "16q13", "seed": 767,
        "phenotype": "CoQ10 Deficiency Type 9 — Neonatal Leigh Syndrome + Fatal Neonatal Lactic Acidosis (COQ9-type; Duncan 2009)",
        "disease": "CoQ10 deficiency type 9 (OMIM #614874) — COQ9; Duncan 2009 first human case — fatal neonatal Leigh; most severe CoQ10 deficiency presentation; RHB accumulation diagnostic; CoQ10 supplementation: limited response (neonatal fatality in most severe); COQ9 regulates CoQ7 activity via substrate channelling; LOF → substrate channelling failure → CoQ10 <5% normal",
        "disease_omim": 614874, "inheritance": "AR",
        "hallmark": "COQ9 = SUBSTRATE CHAPERONE FOR COQ7 (lipid-binding groove captures DMQ10 → hands to COQ7); RHB (rhodoquinone precursor) ACCUMULATION is COQ9-specific biomarker (DMQ10 shunted to RHB in absence of COQ9 delivery to COQ7); fatal neonatal Leigh is the most severe primary CoQ10 deficiency phenotype; Floyd 2016 PNAS crystal structure (tunnel architecture); COQ9 on 16q13 — same chromosome as COQ7 (16p12.3) — different arms",
        "key_ddx": "vs COQ7: COQ7 → DMQ10 accumulates; COQ9 → RHB accumulates (different intermediate); vs PDSS1/2 (upstream): PDSS → no tail, no CoQ10 intermediate; COQ9 → intermediate accumulates; vs fatal neonatal Leigh (NDUFS4-CI): CII/CIII/CIV normal in CI-Leigh; multiple complex reduction in COQ9; vs COQ4: COQ4 Leigh + HCM; COQ9 no HCM, more severe neonatal",
        "founder_variant": "p.Arg244Stop (Duncan 2009 UK); p.Ile25Asn (lipid-binding groove); p.Leu270Pro",
        "coq10_muscle_pct": 5, "srns_rate": 0.15, "snhl_rate": 0.10,
        "ataxia_rate": 0.20, "leigh_rate": 0.90, "myopathy_rate": 0.45,
        "exercise_intol_rate": 0.25, "lactic_ac_rate": 0.90,
        "hcm_rate": 0.25, "retinopathy_rate": 0.10,
        "coq10_response": "Poor — CoQ10 10-30 mg/kg; very limited in fatal neonatal; substrate channelling cannot be corrected",
        "median_onset_months": 0.5,
    },
    # ── COQ Complex Activation — Atypical Kinases ────────────────────────────
    {
        "gene": "COQ8A", "alias": "ADCK3 / aarF domain containing kinase 3", "aa": "647 aa", "kDa": "73.0 kDa",
        "gene_class": "coq_kinase", "pathway_step": "Step 8a — COQ complex activating kinase (ADCK family, atypical kinase)",
        "coq_function": "Atypical kinase (ADCK family — aarF domain) that phosphorylates and activates the COQ enzyme complex; COQ8A (ADCK3) = mitochondrial matrix kinase; activates COQ5 and COQ7 within the complex; ATPase activity demonstrated (Stefely 2015 Mol Cell); LOF → COQ complex activity suppressed → CoQ10 ↓ 30-60%; not as severe as upstream gene mutations; Heeringa 2011 first COQ8A disease (ARCA2); ARCA2 = Autosomal Recessive Cerebellar Ataxia type 2",
        "omim_gene": 612016, "chromosome": "1q42.13", "seed": 768,
        "phenotype": "Autosomal Recessive Cerebellar Ataxia Type 2 (ARCA2) — CoQ10 Deficiency + Exercise Intolerance + Myopathy + Cognitive Decline",
        "disease": "ARCA2 (OMIM #612016) — COQ8A/ADCK3; cerebellar ataxia (95%) + exercise intolerance + myopathy + CoQ10 30-60% reduction in muscle; onset childhood-adolescence; slowly progressive; CoQ10 supplementation consistently improves exercise tolerance and may stabilise cerebellar progression (Lagier-Tourenne 2008 AJHG); Mediterranean/North African founder variants",
        "disease_omim": 612016, "inheritance": "AR",
        "hallmark": "COQ8A (ADCK3) = MOST COMMON CoQ10 GENE IN CEREBELLAR ATAXIA; ARCA2 is the clinical syndrome (pure cerebellar, slowly progressive, CoQ10 responsive); atypical kinase — ADCK family related to UbiB in E. coli (activates UbiH/COQ6-equivalent); Lagier-Tourenne 2008 AJHG first ARCA2 + COQ8A link; COQ10 supplementation: Level B evidence for exercise tolerance improvement in ARCA2 (Montero 2019 systematic review); Mediterranean North African enrichment",
        "key_ddx": "vs COQ8B (ADCK4): COQ8B → SRNS; COQ8A → ARCA2 cerebellar ataxia; both are ADCK kinases on different chromosomes; vs FRDA (frataxin): Friedreich ataxia — iron overload on cardiac MRI, no CoQ10 reduction; vs ARSACS (SACS): Quebec founder, spasticity + ataxia + retinopathy; vs ATCAY (caytaxin): CoQ10 NORMAL; vs SCA28 (AFG3L2): AFG3L2 = mtAA protease, not CoQ10",
        "founder_variant": "p.Ala326Val (North African founder, Lagier-Tourenne 2008); p.Tyr514Cys; p.Arg634Stop",
        "coq10_muscle_pct": 35, "srns_rate": 0.05, "snhl_rate": 0.15,
        "ataxia_rate": 0.95, "leigh_rate": 0.10, "myopathy_rate": 0.70,
        "exercise_intol_rate": 0.90, "lactic_ac_rate": 0.40,
        "hcm_rate": 0.05, "retinopathy_rate": 0.10,
        "coq10_response": "Yes — CoQ10 10-30 mg/kg/day; Level B evidence for exercise tolerance; stabilises cerebellar progression",
        "median_onset_months": 84,
    },
    {
        "gene": "COQ8B", "alias": "ADCK4 / aarF domain containing kinase 4", "aa": "669 aa", "kDa": "75.0 kDa",
        "gene_class": "coq_kinase", "pathway_step": "Step 8b — COQ complex activating kinase (ADCK family, podocyte isoform)",
        "coq_function": "Atypical kinase (ADCK family) with preferential high expression in kidney podocytes and cardiac tissue; activates COQ enzyme complex in podocytes; COQ8B LOF → CoQ10 ↓ in podocytes → podocyte oxidative stress + autophagic failure → SRNS; Ashraf 2013 JASN first COQ8B human disease; zebrafish coq8b morphant develops proteinuria (podocyte-specific validation); ADCK4 = COQ8B in current gene nomenclature",
        "omim_gene": 615573, "chromosome": "19q13.2", "seed": 769,
        "phenotype": "CoQ10 Deficiency — SRNS + Podocyte CoQ10 Deficiency (COQ8B/ADCK4-type; Ashraf 2013)",
        "disease": "Primary CoQ10 deficiency — COQ8B; Ashraf 2013 JASN; SRNS (90%) dominant presentation; focal segmental glomerulosclerosis (FSGS) on biopsy; CoQ10 reduction 30-50% in podocytes (plasma CoQ10 may be normal — PLASMA IS UNRELIABLE); COQ8B = podocyte kinase isoform; CoQ10 supplementation may reduce proteinuria (Ashraf 2013, Ebarasi 2015); usually limited neurologic/systemic involvement",
        "disease_omim": 615573, "inheritance": "AR",
        "hallmark": "COQ8B (ADCK4) = MOST RENAL-SPECIFIC CoQ10 GENE; SRNS/FSGS dominant with minimal systemic involvement (distinguishes from COQ2/COQ6 multi-system); podocyte isoform — test CoQ10 in MUSCLE or FIBROBLASTS not plasma (plasma CoQ10 NORMAL in podocyte-specific deficiency); Ashraf 2013 JASN zebrafish model validates pathway; CoQ10 supplementation 10 mg/kg/day — reduces proteinuria in ~60% (Ashraf 2013); young adult FSGS without family history → screen COQ8B",
        "key_ddx": "vs COQ8A (ADCK3): ADCK3 → cerebellar ataxia; ADCK4 → pure SRNS; both ADCK but different tissue tropism; vs COQ6 (SRNS + SNHL 70%): COQ6 has prominent SNHL; COQ8B SNHL minimal; vs COQ2 (infantile SRNS): COQ2 younger + systemic + retinopathy; COQ8B young adult, isolated renal; vs WT1 (Wilms + SRNS): WT1 in DDS/Frasier (ambiguous genitalia); vs INF2 dominant FSGS: INF2 = actin dynamics, dominant inheritance",
        "founder_variant": "p.Arg399Stop (Ashraf 2013 Pakistani consanguineous); p.Gly516Val (kinase domain); p.Leu671Pro",
        "coq10_muscle_pct": 45, "srns_rate": 0.90, "snhl_rate": 0.05,
        "ataxia_rate": 0.05, "leigh_rate": 0.05, "myopathy_rate": 0.15,
        "exercise_intol_rate": 0.15, "lactic_ac_rate": 0.25,
        "hcm_rate": 0.05, "retinopathy_rate": 0.05,
        "coq10_response": "Yes — CoQ10 10 mg/kg/day; reduces proteinuria in ~60%; prevents ESRD progression if early",
        "median_onset_months": 120,
    },
]

# ── Drug contraindications across all 10 CoQ10 deficiency genes ───────────────
DRUG_CIS = {
    "statins_absolute_ci": {
        "drug": "Statins (HMG-CoA Reductase Inhibitors)",
        "risk": "ABSOLUTE CI in confirmed primary CoQ10 deficiency",
        "mechanism": "Statins block mevalonate pathway (HMG-CoA → mevalonate) — inhibiting isoprenyl-PP synthesis (same precursor as decaprenyl-PP tail for CoQ10); statins reduce plasma CoQ10 30-50%; in primary CoQ10 deficiency any further reduction can be fatal (muscle necrosis, cardiac arrest); even 'low-dose' statins contraindicated",
        "action": "Stop statin immediately; substitute with PCSK9 inhibitor (evolocumab/alirocumab) if lipid-lowering needed",
        "applies_to": "All 10 CoQ10 deficiency genes",
    },
    "vpa_avoid": {
        "drug": "Valproic Acid (VPA)",
        "risk": "AVOID — secondary CoQ10 depletion",
        "mechanism": "VPA inhibits CoQ10 biosynthesis at the PDSS1/PDSS2 step (reduces decaprenyl-PP synthesis) AND depletes CoA pool (succinyl-CoA required for pHBA ring formation); secondary CoQ10 reduction 20-40% on VPA; in primary CoQ10 deficiency this additional depletion can precipitate crisis",
        "action": "Use LEV or LCM if seizure control needed; if VPA essential add CoQ10 100 mg/kg/day prophylaxis",
        "applies_to": "All 10 CoQ10 deficiency genes (especially severe: PDSS1, PDSS2, COQ2, COQ9)",
    },
    "rapamycin_caution": {
        "drug": "Rapamycin / Sirolimus (mTOR inhibitor)",
        "risk": "CAUTION in SRNS (COQ2, COQ6, COQ8B)",
        "mechanism": "Rapamycin used empirically in FSGS/SRNS can worsen proteinuria in CoQ10 deficiency (podocyte mTORC1 required for CoQ10 synthesis regulation); before using rapamycin in SRNS, screen for CoQ10 deficiency genes",
        "action": "Test CoQ10 in muscle before rapamycin; if CoQ10 deficiency: use CoQ10 supplementation, NOT rapamycin",
        "applies_to": "COQ2, COQ6, COQ8B (renal subtype genes)",
    },
    "metformin_avoid": {
        "drug": "Metformin",
        "risk": "AVOID — CI inhibitor, worsens multi-complex ETC deficiency",
        "mechanism": "Metformin inhibits CI (NADH dehydrogenase); in severe primary CoQ10 deficiency all complexes (CI, CIII, CIV) are reduced; CI inhibition → lactic acidosis risk; contraindicated when ETC multi-complex deficiency present",
        "action": "Do not use; substitute with insulin or GLP-1 agonist for glycaemic control",
        "applies_to": "PDSS1, PDSS2, COQ2, COQ4, COQ9 (severe multi-complex deficiency genes)",
    },
    "coq10_supplementation": {
        "drug": "CoQ10 Supplementation (Ubiquinol or Ubiquinone)",
        "risk": "TREATMENT — high-dose (10-30 mg/kg/day)",
        "mechanism": "Exogenous CoQ10 bypasses biosynthesis defect and replenishes the CoQ pool; ubiquinol (reduced form) better absorbed; treatment window critical especially for nephropathy (COQ2, COQ6, COQ8B) — irreversible podocyte loss occurs within 3-6 months of SRNS onset without treatment",
        "action": "Start 10-30 mg/kg/day CoQ10 (ubiquinol preferred) immediately on biochemical diagnosis; do not wait for gene confirmation; monitor CoQ10 in muscle (not plasma) for adequacy",
        "applies_to": "All 10 CoQ10 deficiency genes — PRIMARY TREATMENT",
    },
}

# ── WES utility across all 10 genes ─────────────────────────────────────────
WES_UTILITY = {
    "wes_detects": "All 10 nuclear CoQ10 biosynthesis genes are detected by WES (all nuclear-encoded)",
    "plasma_unreliable": "CRITICAL: Plasma CoQ10 levels are unreliable for diagnosis of primary CoQ10 deficiency — must measure in MUSCLE BIOPSY or skin FIBROBLASTS",
    "muscle_biopsy": "Muscle CoQ10 by HPLC or LC-MS/MS is the gold standard; levels <30% of normal are diagnostic",
    "biosynthesis_flux_assay": "¹³C-labelled pHBA flux assay in fibroblasts is the most specific functional test (measures de novo CoQ10 synthesis rate)",
    "mtdna_subunits": "CoQ10 is non-protein — not encoded by mtDNA; ALL 10 biosynthesis genes are nuclear; WES detects all",
    "secondary_exclusion": "Exclude secondary CoQ10 deficiency (statins, ETFDH, ACAD9, riboflavin deficiency) before diagnosing primary",
    "btbgd_exclusion": "BTBGD (SLC19A3) must be excluded in all suspected metabolic encephalopathies — biotin-thiamine responsive, potentially fatal if missed",
}

# ── Hallmark clinical phenotypes ─────────────────────────────────────────────
HALLMARK_PHENOTYPES = {
    "srns_coq10_triad": "SRNS (FSGS) + CoQ10 ↓ in muscle → consider COQ2, COQ6, COQ8B; add SNHL → strongly COQ6; add retinopathy → COQ2",
    "ataxia_coq10": "Cerebellar ataxia + exercise intolerance + CoQ10 ↓ → COQ8A (ARCA2) most common; also PDSS1, COQ4, COQ5",
    "leigh_coq10": "Leigh/neonatal Leigh + multi-complex ETC ↓ + CoQ10 ↓ → COQ4 (with HCM), COQ9 (most severe, RHB biomarker), PDSS1/2, COQ7 (DMQ10 biomarker)",
    "coq7_dmq10": "DMQ10 accumulation in urine/fibroblasts = PATHOGNOMONIC for COQ7 deficiency",
    "coq9_rhb": "RHB (rhodoquinone precursor) accumulation = COQ9-specific biomarker",
    "coq10_plasma_trap": "NEVER rely on plasma CoQ10 for diagnosis — must use muscle or fibroblast CoQ10 by LC-MS",
    "supplementation_window": "CoQ10 supplementation window: SRNS (COQ2/COQ6/COQ8B) — 3-6 months from onset; ataxia (COQ8A) — best within 2 years; neonatal Leigh — minimal response (too late at clinical presentation)",
}

# ── Aggregate clinical table across all 10 genes ─────────────────────────────
def _patient_cohort(gene_data, n=40):
    """Generate 40 synthetic patients per gene for aggregate statistics."""
    rng_g = random.Random(gene_data["seed"])
    patients = []
    for i in range(n):
        coq10_pct = max(2, rng_g.gauss(gene_data["coq10_muscle_pct"], gene_data["coq10_muscle_pct"] * 0.25))
        patients.append({
            "gene": gene_data["gene"],
            "coq10_muscle_pct": round(coq10_pct, 1),
            "srns": rng_g.random() < gene_data["srns_rate"],
            "snhl": rng_g.random() < gene_data["snhl_rate"],
            "ataxia": rng_g.random() < gene_data["ataxia_rate"],
            "leigh": rng_g.random() < gene_data["leigh_rate"],
            "myopathy": rng_g.random() < gene_data["myopathy_rate"],
            "exercise_intol": rng_g.random() < gene_data["exercise_intol_rate"],
            "lactic_ac": rng_g.random() < gene_data["lactic_ac_rate"],
            "hcm": rng_g.random() < gene_data["hcm_rate"],
            "retinopathy": rng_g.random() < gene_data["retinopathy_rate"],
            "onset_months": max(0.5, rng_g.gauss(gene_data["median_onset_months"],
                                                  gene_data["median_onset_months"] * 0.4 + 1)),
        })
    return patients


def get_overview():
    """Return high-level CoQ10-Biosynthesis-Atlas overview for the /api/coq10-biosynthesis-atlas/overview endpoint."""
    all_patients = []
    for g in COQ_GENES:
        all_patients.extend(_patient_cohort(g))

    n = len(all_patients)
    avg_coq10 = round(sum(p["coq10_muscle_pct"] for p in all_patients) / n, 1)
    srns_n     = sum(1 for p in all_patients if p["srns"])
    snhl_n     = sum(1 for p in all_patients if p["snhl"])
    ataxia_n   = sum(1 for p in all_patients if p["ataxia"])
    leigh_n    = sum(1 for p in all_patients if p["leigh"])
    myopathy_n = sum(1 for p in all_patients if p["myopathy"])
    lactic_n   = sum(1 for p in all_patients if p["lactic_ac"])
    hcm_n      = sum(1 for p in all_patients if p["hcm"])

    gene_counts = {"decaprenyl_pp": 0, "ring_attachment": 0, "ring_modification": 0, "coq_kinase": 0}
    for g in COQ_GENES:
        gene_counts[g["gene_class"]] += 1

    return {
        "atlas": "CoQ10-Biosynthesis-Atlas",
        "subtitle": "Complete 10-Gene Primary Coenzyme Q10 (Ubiquinone) Deficiency Reference",
        "n_genes": len(COQ_GENES),
        "n_decaprenyl_pp": gene_counts["decaprenyl_pp"],
        "n_ring_attachment": gene_counts["ring_attachment"],
        "n_ring_modification": gene_counts["ring_modification"],
        "n_coq_kinase": gene_counts["coq_kinase"],
        "n_patients": n,
        "cohort_formula": f"{len(COQ_GENES)} genes × 40 patients = {n} patient aggregate (seeds 760–769)",
        "function": "CoQ10 (Ubiquinone-10) — sole mobile IMM electron/proton carrier bridging CI/CII → CIII",
        "pathway": "pHBA + Decaprenyl-PP → PDSS1/2 (tail) → COQ2 (attachment) → COQ4/5/6/7/9 (ring) → COQ8A/B (activation)",
        "all_nuclear": True,
        "wes_detects_all": True,
        "plasma_unreliable": "CRITICAL: Plasma CoQ10 unreliable — diagnose via MUSCLE or FIBROBLAST CoQ10 (LC-MS/MS)",
        "gene_list": {
            "decaprenyl_pp_2": ["PDSS1", "PDSS2"],
            "ring_attachment_1": ["COQ2"],
            "ring_modification_4": ["COQ4", "COQ5", "COQ6", "COQ7", "COQ9"],
            "coq_kinase_2": ["COQ8A (ADCK3)", "COQ8B (ADCK4)"],
        },
        "aggregate_clinical": {
            "n_patients": n,
            "avg_coq10_muscle_pct": avg_coq10,
            "srns_pct": round(srns_n / n * 100, 1),
            "snhl_pct": round(snhl_n / n * 100, 1),
            "ataxia_pct": round(ataxia_n / n * 100, 1),
            "leigh_pct": round(leigh_n / n * 100, 1),
            "myopathy_pct": round(myopathy_n / n * 100, 1),
            "lactic_acidosis_pct": round(lactic_n / n * 100, 1),
            "hcm_pct": round(hcm_n / n * 100, 1),
        },
        "drug_contraindications": DRUG_CIS,
        "wes_utility": WES_UTILITY,
        "hallmark_phenotypes": HALLMARK_PHENOTYPES,
        "key_rules": {
            "statins_absolute_ci": "Statins ABSOLUTE CI in ALL primary CoQ10 deficiency (block decaprenyl-PP → further CoQ10 depletion)",
            "coq10_treatment": "CoQ10 10-30 mg/kg/day (ubiquinol preferred) — PRIMARY TREATMENT for all 10 genes",
            "plasma_trap": "NEVER diagnose primary CoQ10 deficiency by plasma CoQ10 alone — measure in MUSCLE BIOPSY or FIBROBLASTS",
            "srns_screen": "All SRNS under age 15 with no nephrin/podocin: SCREEN CoQ10 in muscle + COQ gene panel before any other workup",
            "vpa_avoid": "VPA depletes CoQ10 30-40% via mevalonate block — avoid in all confirmed CoQ10 deficiency genes",
        },
        "cii_always_normal": "CII is ALL nuclear — CII activity is NORMAL in primary CoQ10 deficiency (CoQ10 feeds CIII not CII directly; CII normal = not multi-complex mtDNA disease)",
        "btbgd_exclusion": "BTBGD (SLC19A3) MANDATORY EXCLUSION in all metabolic encephalopathy before CoQ10 workup — biotin-thiamine responsive, potentially fatal if missed",
    }


def get_breakdown():
    """Return per-gene breakdown for the /api/coq10-biosynthesis-atlas/breakdown endpoint."""
    result = []
    for g in COQ_GENES:
        pts = _patient_cohort(g)
        n = len(pts)
        result.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "gene_class": g["gene_class"],
            "pathway_step": g["pathway_step"],
            "chromosome": g["chromosome"],
            "omim_gene": g["omim_gene"],
            "disease_omim": g["disease_omim"],
            "inheritance": g["inheritance"],
            "phenotype_summary": g["phenotype"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "founder_variant": g["founder_variant"],
            "coq10_response": g["coq10_response"],
            "cohort_n": n,
            "avg_coq10_muscle_pct": round(sum(p["coq10_muscle_pct"] for p in pts) / n, 1),
            "srns_pct": round(sum(1 for p in pts if p["srns"]) / n * 100, 1),
            "snhl_pct": round(sum(1 for p in pts if p["snhl"]) / n * 100, 1),
            "ataxia_pct": round(sum(1 for p in pts if p["ataxia"]) / n * 100, 1),
            "leigh_pct": round(sum(1 for p in pts if p["leigh"]) / n * 100, 1),
            "myopathy_pct": round(sum(1 for p in pts if p["myopathy"]) / n * 100, 1),
            "exercise_intol_pct": round(sum(1 for p in pts if p["exercise_intol"]) / n * 100, 1),
            "lactic_ac_pct": round(sum(1 for p in pts if p["lactic_ac"]) / n * 100, 1),
            "hcm_pct": round(sum(1 for p in pts if p["hcm"]) / n * 100, 1),
            "retinopathy_pct": round(sum(1 for p in pts if p["retinopathy"]) / n * 100, 1),
            "median_onset_months": g["median_onset_months"],
        })
    return {"genes": result, "n_genes": len(result), "n_patients_total": len(result) * 40}


def get_definitions():
    """Return clinical definitions for the /api/coq10-biosynthesis-atlas/definitions endpoint."""
    return {
        "terms": {
            "CoQ10_ubiquinone": "Coenzyme Q10 (Ubiquinone-10) — lipophilic electron/proton carrier in the IMM; reduced form ubiquinol (QH2) acts as antioxidant; sole mobile ETC carrier between CI/CII → CIII (Q cycle); 10 isoprene units in the tail (hence CoQ10 in humans vs CoQ9 in rodents)",
            "primary_coq10_deficiency": "AR mutations in CoQ10 biosynthesis genes (PDSS1/2, COQ2, COQ4-9, COQ8A/B) → reduced CoQ10 in tissues; distinct from secondary CoQ10 deficiency (statins, ETFDH, riboflavin deficiency)",
            "secondary_coq10_deficiency": "Reduced CoQ10 as secondary consequence of another disease or drug: statins (most common cause, reduce CoQ10 30-50%), ETFDH mutations, ACAD9, ageing — must exclude before diagnosing primary",
            "PDSS1_PDSS2_heterotetramer": "PDSS1 (α, chain-length specificity) + PDSS2 (β, catalytic) form obligate α2β2 heterotetramer; catalyses sequential head-to-tail condensation of isopentenyl-PP units to form decaprenyl-PP (the 10-isoprene lipid tail); first dedicated step in CoQ10 biosynthesis",
            "COQ_complex": "Multi-enzyme complex in the IMM matrix face: COQ3-COQ9 (+ COQ8A/B kinases) form a ~1 MDa biosynthesis cluster; substrate channelling within the complex completes all ring modification steps; COQ4 is the structural scaffold; disruption of ANY component can disassemble the whole complex",
            "SRNS": "Steroid-Resistant Nephrotic Syndrome — proteinuria that fails to remit with full-dose prednisolone (2 mg/kg/day for ≥ 4 weeks); CoQ10 genes (COQ2, COQ6, COQ8B) cause SRNS via podocyte CoQ10 depletion → oxidative damage → glomerular barrier failure",
            "FSGS": "Focal Segmental Glomerulosclerosis — histologic pattern (not a diagnosis) on renal biopsy; seen in SRNS from COQ2/COQ6/COQ8B; segmental sclerosis of glomerular tufts; also in NPHS1/2/WT1/INF2 — CoQ10 must be in MUSCLE to diagnose CoQ FSGS",
            "ARCA2": "Autosomal Recessive Cerebellar Ataxia Type 2 — the clinical syndrome of COQ8A (ADCK3) deficiency; slowly progressive cerebellar ataxia onset childhood-adolescence; exercise intolerance + myopathy; CoQ10 responsive (Level B evidence); Mediterranean/North African enrichment (p.Ala326Val founder)",
            "DMQ10": "Demethoxy-ubiquinone-10 — the partially-hydroxylated CoQ10 biosynthesis intermediate that accumulates in COQ7 deficiency; DMQ10 detected in urine or fibroblasts is PATHOGNOMONIC for COQ7; COQ7 should be the first gene checked when DMQ10 is found",
            "RHB": "Rhodoquinone precursor (sulphide form of CoQ intermediate) — accumulates specifically in COQ9 deficiency; biomarker distinguishing COQ9 from COQ7 (which accumulates DMQ10 not RHB)",
            "Q_cycle": "Peter Mitchell's Q cycle (Nobel 1978) — at CIII: QH2 donates 2e⁻ sequentially; one to cytochrome c (via RISP) and one recycled to Q at Qi site; net: 2QH2 + Q + 2Cyt-cox → 2Q + QH2 + 2Cyt-cred + 4H⁺(IMS); CoQ10 is the essential substrate",
            "muscle_coq10_lc_ms": "Gold-standard CoQ10 measurement by LC-MS/MS in frozen muscle biopsy; levels < 30% of normal (age-matched) = diagnostic of primary CoQ10 deficiency; plasma CoQ10 correlates poorly with tissue CoQ10 and must NOT be used for diagnosis",
            "ubiquinol": "Reduced form of CoQ10 (QH2 / Ubiquinol-10); the electron-donor form; also acts as lipid-soluble antioxidant in IMM; commercially available as CoQ10 supplement (ubiquinol form better absorbed than ubiquinone form); half-life ~48h in plasma",
            "statin_coq10_depletion": "HMG-CoA reductase inhibitors (statins) block mevalonate → farnesyl-PP → geranylgeranyl-PP (isoprene precursors needed for decaprenyl-PP CoQ10 tail synthesis); statins deplete plasma CoQ10 by 30-50%; rhabdomyolysis risk in primary CoQ10 deficiency + statin = absolute CI",
            "coq_biosynthesis_flux_assay": "¹³C6-labelled para-hydroxybenzoate (pHBA) pulse assay in cultured fibroblasts — measures rate of de novo CoQ10 synthesis by tracking ¹³C label incorporation via LC-MS; most sensitive functional test to confirm primary CoQ10 deficiency and identify which step is blocked",
            "adck_family": "aarF domain-containing kinase family (ADCK1-5); COQ8A (ADCK3) and COQ8B (ADCK4) are mitochondrial atypical kinases that phosphorylate and activate the COQ enzyme complex; related to UbiB in E. coli; ATPase activity demonstrated; not classical protein kinases (lack activation loop typical kinase features)",
        },
        "gene_classes": {
            "decaprenyl_pp": "PDSS1, PDSS2 — synthesise the 10-isoprene lipid tail (decaprenyl-PP) of CoQ10; most upstream step; LOF = complete pathway arrest",
            "ring_attachment": "COQ2 — attaches 4-hydroxybenzoate ring to decaprenyl-PP tail; second step; LOF = complete pathway arrest; most common infantile primary CoQ10 gene",
            "ring_modification": "COQ4 (scaffold), COQ5 (C-methylation), COQ6 (C-5 hydroxylation, FAD), COQ7 (C-5 hydroxylation, diiron), COQ9 (substrate chaperone for COQ7) — complete ring decoration; LOF = partial-to-complete CoQ10 reduction with unique biomarker intermediates",
            "coq_kinase": "COQ8A (ADCK3), COQ8B (ADCK4) — atypical kinases that activate the COQ enzyme complex; LOF = partial CoQ10 reduction (30-60%) with tissue-specific tropism (ADCK3: cerebellar; ADCK4: podocyte)",
        },
        "supplementation_protocols": {
            "severe_infantile": "COQ2, PDSS1/2, COQ9, COQ4 (severe): CoQ10 (ubiquinol) 20-30 mg/kg/day divided BID; start immediately on suspicion; do not wait for WES",
            "arca2_coa8a": "COQ8A (ARCA2): CoQ10 (ubiquinol) 15-20 mg/kg/day; exercise tolerance improves within 3 months; cerebellar stabilisation goal",
            "srns_renal": "COQ2, COQ6, COQ8B (renal): CoQ10 (ubiquinol) 10-20 mg/kg/day; target muscle CoQ10 > 50% normal; monitor proteinuria monthly",
            "monitoring": "Muscle CoQ10 by LC-MS/MS at 3 and 12 months; plasma CoQ10 NOT used for monitoring; urine protein/creatinine ratio monthly in renal subtype",
        },
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}")
    print(f"Avg CoQ10 muscle: {ov['aggregate_clinical']['avg_coq10_muscle_pct']}%")
    print(f"SRNS %: {ov['aggregate_clinical']['srns_pct']}")
    print(f"Ataxia %: {ov['aggregate_clinical']['ataxia_pct']}")
    print(json.dumps(ov['gene_list'], indent=2))
