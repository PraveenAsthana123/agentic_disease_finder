#!/usr/bin/env python3
"""Vitamin-Atlas — Complete 8-Gene Vitamin-Responsive Epileptic Encephalopathies Atlas
ALDH7A1 (Pyridoxine-Dependent Epilepsy, PDE — B6/pyridoxine responsive, AASA biomarker) ·
PNPO   (PLP-Responsive Neonatal Epilepsy — PLP NOT pyridoxine, fatal if missed) ·
PLPBP  (PLP Homeostasis Protein Epilepsy — B6 responsive, AASA NORMAL distinguishes from PDE) ·
TPK1   (Thiamine Pyrophosphokinase 1 — B1/TPP responsive, serum thiamine can be NORMAL) ·
SLC19A3 (Biotin-Thiamine-Responsive Basal Ganglia Disease — EMERGENCY B1+B7, BG necrosis) ·
BTD    (Biotinidase Deficiency — B7/biotin, NBS-detected, completely preventable 4D) ·
HLCS   (Holocarboxylase Synthetase / Multiple Carboxylase Deficiency — biotin neonatal, BTD NORMAL) ·
SLC52A2 (BVVL Syndrome / Riboflavin Transporter 2 — B2/riboflavin, MCAD-like profile, dramatic response)
320-patient aggregate cohort (8 × 40, seeds 966–973)

Vitamin-Responsive Epileptic Encephalopathies — Key Principles:
  - TREAT BEFORE DIAGNOSIS IS CONFIRMED: all 8 disorders are emergencies where vitamin
    empiric therapy must precede confirmatory testing. Delay = irreversible neuronal death.
  - B6 FORMS MATTER CRITICALLY:
      PDE (ALDH7A1): pyridoxine (oral/IV) → WORKS; PLP also works.
      PNPO: pyridoxal-5'-phosphate (PLP) ONLY; pyridoxine FAILS → misdiagnosis = death.
      PLPBP: PLP preferred; some respond to pyridoxine.
      → IN ANY NEONATAL REFRACTORY SEIZURE: try pyridoxine first; if fails → PLP next.
  - THIAMINE TRAPS:
      TPK1: serum thiamine HIGH or normal (non-phosphorylated accumulates); measure TPP.
      SLC19A3: MRI is key (bilateral BG necrosis); thiamine + BIOTIN both mandatory.
  - BIOTIN FORMS:
      BTD: biotinidase ABSENT → recycled biotin LOST; free biotin supplementation bypasses recycling.
      HLCS: biotinidase NORMAL; HLCS cannot attach biotin to carboxylases; same OA profile → key DDx.
  - RIBOFLAVIN (BVVL):
      SLC52A2: acylcarnitine profile mimics MCAD (C8, C10 elevated); MCAD enzyme normal.
      High-dose riboflavin (10-40 mg/kg/day) can reverse even advanced respiratory failure.

COHORT: 8 × 40 = 320 patient slots (seeds 966–973; gene-specific seeds)
"""

import random

SEED_BASE = 966

VIT_GENES = [
    # ── ALDH7A1 — Pyridoxine-Dependent Epilepsy (PDE) ────────────────────────────────
    {
        "gene": "ALDH7A1", "protein": "Antiquitin (Aldehyde Dehydrogenase 7A1)",
        "alias": "ALDH7A1 — Pyridoxine-Dependent Epilepsy (PDE, OMIM #266100); B6 responsive",
        "aa": "539 aa", "kDa": "62 kDa",
        "gene_class": (
            "Aldehyde dehydrogenase 7A1 (antiquitin); mitochondrial matrix enzyme; "
            "catabolises alpha-aminoadipic semialdehyde (AASA) and its cyclic form "
            "1-piperideine-6-carboxylate (P6C) in the L-lysine degradation pathway; "
            "deficiency → AASA/P6C accumulate → P6C reacts with and inactivates "
            "pyridoxal-5'-phosphate (PLP, the active B6 form) via Knoevenagel condensation "
            "→ functional PLP deficiency → PLP-dependent enzyme failure → neonatal/infantile seizures"
        ),
        "vit_subgroup": "Vitamin B6 (Pyridoxine) — Pyridoxine-Dependent Epilepsy",
        "vitamin": "B6 / Pyridoxine + Lysine restriction",
        "locus": "5q31.2", "omim_gene": 107323, "omim_disease": 266100,
        "inheritance": "AR. 5q31.2. Both sexes equally. Incidence ~1/100,000-700,000.",
        "seed_offset": 0,
        "onset_range_y": (0.0, 3.0),
        "phenotype": (
            "Classic PDE: neonatal refractory seizures (onset often in utero or first hours of life); "
            "multifocal clonic, tonic, and myoclonic seizures; burst-suppression on EEG; "
            "fails multiple AEDs; RESPONSE TO PYRIDOXINE 100mg IV within minutes is diagnostic. "
            "Atypical PDE: onset 1-3 years; broader seizure types; responds to oral pyridoxine. "
            "Common in PDE: developmental delay, autistic features, intellectual disability — "
            "even with seizure control; lysine restriction + folinic acid improve neurodevelopment. "
            "Antenatal: fetal distress, abnormal CTG, neonatal asphyxia-like presentation."
        ),
        "disease": (
            "ALDH7A1 encodes antiquitin (539 aa, 62 kDa), a mitochondrial aldehyde dehydrogenase "
            "that catabolises alpha-aminoadipic semialdehyde (AASA) in the L-lysine degradation "
            "pathway (saccharopine pathway). Deficiency → AASA + its cyclic tautomer P6C (1-piperideine-"
            "6-carboxylate) accumulate → P6C inactivates PLP (active B6) by Knoevenagel condensation → "
            "functional PLP deficiency despite normal dietary B6 intake → PLP-dependent enzymes fail "
            "(GABA synthesis via GAD, aromatic AA decarboxylase, etc.) → seizures.\n\n"
            "BIOMARKERS: (1) Urine alpha-AASA markedly elevated (the PRIMARY biomarker); "
            "(2) Plasma/CSF pipecolic acid elevated; (3) Plasma/CSF P6C elevated; "
            "(4) CSF PLP normal or low (CONTRAST TO PNPO where CSF PLP very low). "
            "AASA elevation is PATHOGNOMONIC: also elevated in folinic-acid responsive seizures.\n\n"
            "BIOCHEMISTRY OF TREATMENT: Pyridoxine (oral/IV) → converted to PLP → excess PLP "
            "overrides P6C inactivation by mass action; PLP directly supplements. "
            "LYSINE RESTRICTION reduces substrate load → less AASA generated; independent of pyridoxine. "
            "FOLINIC ACID (leucovorin): corrects secondary folate cycle disruption by P6C.\n\n"
            "TRIPLE THERAPY: pyridoxine + lysine restriction + folinic acid → best neurodevelopmental outcomes.\n\n"
            "COMMON MUTATIONS: p.Glu399Gln — most common European pathogenic allele. "
            "p.Arg266Gln — second most common. c.1364+1G>A — splice. "
            "No clear genotype-phenotype correlation in seizure severity. "
            "Large deletions: 5q31 contiguous deletions cause PDE + additional features."
        ),
        "hallmark": (
            "ALDH7A1/PDE HALLMARKS: "
            "(1) AASA URINE ELEVATED — PATHOGNOMONIC biomarker; measured by LC-MS/MS. "
            "(2) PYRIDOXINE 100mg IV → SEIZURE CESSATION WITHIN MINUTES (may take up to 30 min): "
            "   this empiric trial is FIRST AID in any neonatal refractory seizure. "
            "(3) TRIPLE THERAPY: pyridoxine (15-30 mg/kg/day) + lysine restriction + folinic acid. "
            "(4) NBS: AASA in dried blood spot — not yet standard; research NBS programs. "
            "(5) ANTENATAL ONSET: fetal seizures in utero reported; pyridoxine to mother. "
            "(6) NEURODEVELOPMENTAL OUTCOME IMPROVED by lysine restriction even if seizure-free. "
            "(7) EEG BURST-SUPPRESSION → pyridoxine → normalisation is the bedside diagnosis. "
            "(8) CSF PLP NORMAL or LOW (vs PNPO where CSF PLP very low — key DDx)."
        ),
        "nbs_marker": "Not standard NBS; research NBS: urine/DBS AASA by LC-MS/MS",
        "key_biomarker": "Urine AASA (alpha-aminoadipic semialdehyde) — PATHOGNOMONIC; plasma/CSF pipecolic acid",
        "severity_spectrum": "Neonatal most severe; atypical variants (late infantile) milder; NDD common despite seizure control",
        "treatments": ["Pyridoxine 15-30mg/kg/day (lifelong)", "Lysine restriction", "Folinic acid (leucovorin)", "Pyridoxal-5'-phosphate (PLP) alternative"],
        "emergency": "Pyridoxine 100mg IV — try in ANY neonatal refractory seizure",
        "ci_drugs": ["Standard AEDs (ineffective — use only for acute bridge)", "High-dose leucovorin without pyridoxine (incomplete)"],
    },
    # ── PNPO — PLP-Responsive Neonatal Epilepsy ──────────────────────────────────────
    {
        "gene": "PNPO", "protein": "Pyridox(am)ine Phosphate Oxidase",
        "alias": "PNPO — PLP-Responsive Neonatal Epilepsy (PLPNEE, OMIM #610090); PLP NOT pyridoxine",
        "aa": "261 aa", "kDa": "30 kDa",
        "gene_class": (
            "Pyridox(am)ine phosphate oxidase; FMN-containing enzyme; catalyses conversion of "
            "pyridoxine-5'-phosphate (PNP) and pyridoxamine-5'-phosphate (PMP) → pyridoxal-5'-phosphate "
            "(PLP, the active cofactor); PNPO deficiency → PNP/PMP cannot be converted to PLP → "
            "primary PLP deficiency → all PLP-dependent enzymes fail (>160 enzymes): GAD (GABA), "
            "aromatic L-amino acid decarboxylase (DA/5HT), glycine decarboxylase, cystathionine beta-synthase, etc."
        ),
        "vit_subgroup": "Vitamin B6 (PLP form) — PLP-Responsive Neonatal Epilepsy",
        "vitamin": "B6 / Pyridoxal-5'-Phosphate (PLP) — NOT pyridoxine",
        "locus": "17q21.32", "omim_gene": 610090, "omim_disease": 610090,
        "inheritance": "AR. 17q21.32. Both sexes equally. Incidence ~1/500,000 estimated.",
        "seed_offset": 1,
        "onset_range_y": (0.0, 0.5),
        "phenotype": (
            "Neonatal epilepsy: onset often within first 24-48 hours of life (up to 50% within 24h); "
            "severe burst-suppression EEG; multifocal seizures; premature infants especially vulnerable. "
            "Fails pyridoxine trial (CRITICAL differentiator from PDE/ALDH7A1). "
            "Metabolic crisis: lactic acidosis, hypoglycaemia, thrombocytopenia. "
            "CSF: elevated aromatic amino acids (phenylalanine, tyrosine, tryptophan elevated because "
            "aromatic L-amino acid decarboxylase/AADC fails without PLP); "
            "CSF PLP markedly low (PRIMARY finding); urine vanillactic acid elevated. "
            "Neurodevelopmental outcome: variable — better with early PLP treatment; many have "
            "intellectual disability and seizures requiring ongoing PLP."
        ),
        "disease": (
            "PNPO encodes pyridox(am)ine phosphate oxidase (261 aa, 30 kDa), a flavin-mononucleotide "
            "(FMN)-containing enzyme that catalyses the final step in PLP biosynthesis from dietary "
            "pyridoxine: PNP + O2 → PLP + H2O2; also PMP + O2 → PLP + NH3. "
            "Deficiency → primary CSF/brain PLP deficiency → failure of >160 PLP-dependent enzymes.\n\n"
            "WHY PYRIDOXINE FAILS IN PNPO: pyridoxine is absorbed → phosphorylated to PNP by pyridoxal "
            "kinase — BUT PNP cannot be converted to PLP without PNPO → PNP accumulates → PNP is "
            "actually a competitive inhibitor of residual PNPO → pyridoxine supplementation makes things "
            "WORSE in severe PNPO deficiency. PLP supplementation bypasses PNPO entirely.\n\n"
            "BIOMARKERS: (1) CSF PLP markedly low (PRIMARY marker); (2) CSF aromatic AAs elevated "
            "(AADC uses PLP → AADC failure → Phe, Tyr, Trp accumulate); (3) CSF threonine elevated "
            "(threonine aldolase uses PLP); (4) Urine vanillactic acid elevated (dopamine catabolite); "
            "(5) Urine AASA NORMAL (key DDx from PDE/ALDH7A1); (6) Plasma/CSF glycine may be elevated "
            "(glycine cleavage system uses PLP); (7) AADC activity in plasma markedly reduced.\n\n"
            "TREATMENT: pyridoxal-5'-phosphate (PLP, pyridoxamine-di-HCl or pyridoxal-HCl) "
            "30-50 mg/kg/day in 3-4 divided doses; dose titrated by response; EEG normalisation "
            "confirms adequacy. Monitor liver function (PLP hepatotoxicity reported at high doses). "
            "Folinic acid may be added (PLP-folate interaction).\n\n"
            "COMMON MUTATIONS: p.Arg229His — most prevalent (Middle Eastern founder); "
            "p.Asp33Val — severe neonatal lethal if untreated; c.246-1G>A — splice variant. "
            "p.Arg116Gln — partial residual activity; atypical onset."
        ),
        "hallmark": (
            "PNPO/PLP-NEONATAL EPILEPSY HALLMARKS: "
            "(1) PLP NOT PYRIDOXINE — pyridoxine FAILS (may worsen by competitive inhibition); "
            "   if pyridoxine trial fails → PLP IMMEDIATELY. "
            "(2) CSF PLP MARKEDLY LOW — primary diagnostic marker (LP mandatory in neonatal epilepsy). "
            "(3) CSF AROMATIC AAs ELEVATED — AADC failure marker (Phe, Tyr, Trp high). "
            "(4) URINE AASA NORMAL — KEY DDx from ALDH7A1/PDE. "
            "(5) NEONATAL ONSET DAY 1-2 — earlier than most PDE (though PDE also neonatal). "
            "(6) PREMATURE INFANTS ESPECIALLY VULNERABLE — low PLP reserve. "
            "(7) PYRIDOXINE COMPETITIVE INHIBITION — dose-dependent worsening of PNPO deficiency. "
            "(8) HIGH-DOSE PLP MONITORING — liver enzymes q6-8 weeks; reduce dose if ALT rising."
        ),
        "nbs_marker": "No standard NBS; AADC biomarkers (VLA, 3-OMD) on DBS — research only",
        "key_biomarker": "CSF PLP markedly low; CSF aromatic AAs elevated; urine vanillactic acid; AASA NORMAL",
        "severity_spectrum": "Severe neonatal (onset day 1, fatal untreated); milder atypical (later onset, partial response to pyridoxine)",
        "treatments": ["Pyridoxal-5'-phosphate (PLP) 30-50mg/kg/day", "Folinic acid adjunct", "LFT monitoring mandatory"],
        "emergency": "PLP 30mg/kg IV/oral — if pyridoxine fails in neonatal seizures → PLP IMMEDIATELY",
        "ci_drugs": ["Pyridoxine (may worsen PNPO by competitive inhibition)", "Isoniazid (consumes PLP)", "D-penicillamine (inactivates PLP)"],
    },
    # ── PLPBP — PLP Homeostasis Protein Epilepsy ─────────────────────────────────────
    {
        "gene": "PLPBP", "protein": "Pyridoxal Phosphate Binding Protein (formerly PROSC)",
        "alias": "PLPBP — PLP Homeostasis Protein Epilepsy (OMIM #617290); B6 responsive, AASA NORMAL",
        "aa": "312 aa", "kDa": "35 kDa",
        "gene_class": (
            "Pyridoxal phosphate binding protein (PLPBP, formerly PROSC — proline synthetase co-transcribed "
            "homologue, bacterial); maintains PLP homeostasis by acting as an intracellular PLP buffer/chaperone; "
            "belongs to COG0325 family; widespread expression; PLPBP binds PLP and prevents "
            "PLP-mediated carbonyl stress by sequestering reactive PLP aldehyde; deficiency → "
            "PLP misbalance → elevated free PLP toxicity in some compartments AND PLP deficiency "
            "in others → seizures by mechanism distinct from ALDH7A1 and PNPO"
        ),
        "vit_subgroup": "Vitamin B6 (PLP Homeostasis) — PLPBP Epilepsy",
        "vitamin": "B6 / Pyridoxal-5'-Phosphate (PLP) preferred; pyridoxine in some",
        "locus": "8p11.23", "omim_gene": 604436, "omim_disease": 617290,
        "inheritance": "AR. 8p11.23. Both sexes equally. Rare; ~50 cases reported globally.",
        "seed_offset": 2,
        "onset_range_y": (0.0, 1.0),
        "phenotype": (
            "Neonatal to infantile onset epilepsy (typically first weeks to months); "
            "focal or generalised seizures; burst-suppression or hypsarrhythmia on EEG; "
            "less uniform than PNPO (more variable: neonatal through infantile). "
            "Clinical response: most respond to PLP (some to pyridoxine); doses often higher. "
            "Neurodevelopmental outcomes: variable — intellectual disability + autism spectrum common. "
            "CSF PLP: low (but often less dramatically than PNPO); "
            "Urine AASA: NORMAL (KEY DDx from ALDH7A1/PDE — this is the critical distinguishing feature). "
            "Some patients present with non-epileptic movement disorders (myoclonus) ± spasticity."
        ),
        "disease": (
            "PLPBP (312 aa, 35 kDa) acts as an intracellular PLP reservoir protein, maintaining "
            "the equilibrium of free vs enzyme-bound PLP and preventing PLP-mediated carbonyl stress. "
            "Free PLP aldehyde is highly reactive (forms Schiff bases with protein lysine residues); "
            "PLPBP buffers this reactivity. Deficiency → PLP misbalance → excess free PLP in some "
            "compartments (creating carbonyl stress) AND inadequate PLP delivery to target enzymes.\n\n"
            "BIOMARKERS: (1) CSF PLP low (less dramatically than PNPO); (2) Urine AASA NORMAL "
            "(CRITICAL DDx from ALDH7A1/PDE); (3) Urine/plasma amino acids may show subtle changes; "
            "(4) CSF aromatic AAs: less dramatically elevated than PNPO; (5) Pipecolic acid NORMAL "
            "(unlike ALDH7A1). No single pathognomonic biomarker — diagnosis by gene panel + response.\n\n"
            "TREATMENT: PLP 30-50 mg/kg/day preferred (some respond to pyridoxine); "
            "higher doses than PDE often required; response is often slower than ALDH7A1. "
            "Folinic acid adjunct in some. Monitoring: liver function, developmental milestones.\n\n"
            "MUTATIONS: p.Arg241Gln — reported in multiple families; affects PLP-binding pocket. "
            "p.Leu175Pro — helix-breaking mutation. p.Pro113Leu — N-terminal. "
            "No clear genotype-phenotype: same mutation → variable severity.\n\n"
            "DIAGNOSTIC PATHWAY: neonatal/infantile epilepsy → pyridoxine trial → if partial/no response "
            "→ PLP trial → gene panel including PLPBP; AASA measurement to exclude ALDH7A1."
        ),
        "hallmark": (
            "PLPBP HALLMARKS: "
            "(1) AASA NORMAL — KEY DDx FROM ALDH7A1/PDE (most important distinguishing test). "
            "(2) CSF PLP LOW — less dramatically than PNPO; variable severity. "
            "(3) PYRIDOXINE MAY PARTIALLY WORK — unlike PNPO; mixed response pattern. "
            "(4) PLP PREFERRED OVER PYRIDOXINE — use PLP first if PLPBP suspected. "
            "(5) VARIABLE PHENOTYPE — neonatal through infantile; broader clinical spectrum. "
            "(6) RARE (~50 cases) — requires gene panel diagnosis; B6-responsive epilepsy panels. "
            "(7) HIGHER PLP DOSES OFTEN NEEDED vs ALDH7A1. "
            "(8) CARBONYL STRESS MECHANISM — distinct from both ALDH7A1 (substrate inactivation) and PNPO (synthesis block)."
        ),
        "nbs_marker": "No standard NBS; research NBS not yet established",
        "key_biomarker": "CSF PLP low; urine AASA NORMAL (key DDx); pipecolic acid NORMAL; diagnosis by gene panel",
        "severity_spectrum": "Variable: severe neonatal to milder infantile; intellectual disability common",
        "treatments": ["Pyridoxal-5'-phosphate (PLP) 30-50mg/kg/day", "Pyridoxine (partial responders)", "Folinic acid adjunct"],
        "emergency": "PLP empiric trial in any B6-non-responsive neonatal epilepsy",
        "ci_drugs": ["Isoniazid (PLP depletor)", "D-penicillamine", "Pyrazinamide"],
    },
    # ── TPK1 — Thiamine Pyrophosphokinase 1 Deficiency ───────────────────────────────
    {
        "gene": "TPK1", "protein": "Thiamine Pyrophosphokinase 1",
        "alias": "TPK1 — Thiamine Metabolism Dysfunction Syndrome 5 (THMD5, OMIM #614389); B1/TPP responsive",
        "aa": "243 aa", "kDa": "27 kDa",
        "gene_class": (
            "Thiamine pyrophosphokinase 1; ATP-dependent enzyme; phosphorylates thiamine (vitamin B1) → "
            "thiamine pyrophosphate (TPP, the active diphosphorylated coenzyme form); TPP is the essential "
            "cofactor for: pyruvate dehydrogenase complex (PDC), branched-chain alpha-ketoacid "
            "dehydrogenase (BCKDH), 2-oxoglutarate dehydrogenase (2OGDC), and transketolase (TKT); "
            "TPK1 deficiency → TPP cannot be synthesised from thiamine → all TPP-dependent pathways fail "
            "→ lactic acidosis + encephalopathy"
        ),
        "vit_subgroup": "Vitamin B1 (Thiamine) — Thiamine Pyrophosphokinase Deficiency",
        "vitamin": "B1 / Thiamine (pharmacological) or Thiamine Pyrophosphate (direct)",
        "locus": "7q35", "omim_gene": 606370, "omim_disease": 614389,
        "inheritance": "AR. 7q35. Both sexes equally. Very rare; <50 cases reported.",
        "seed_offset": 3,
        "onset_range_y": (0.0, 5.0),
        "phenotype": (
            "Episodic encephalopathy with ataxia; triggered by fever, illness, fasting, high carbohydrate load; "
            "progressive neurological deterioration between episodes; lactic acidosis during crises; "
            "MRI: bilateral symmetric T2 hyperintensity in basal ganglia and brainstem (Leigh-like) "
            "acutely, with restricted diffusion; subacute/chronic: cerebellar atrophy. "
            "Infantile to early childhood onset (typically 1-5 years). "
            "EEG: non-specific slowing; seizures variably present (myoclonic, focal). "
            "Peripheral neuropathy in some (axonal motor > sensory)."
        ),
        "disease": (
            "TPK1 (243 aa, 27 kDa) phosphorylates thiamine using ATP: thiamine + ATP → TPP + AMP. "
            "TPP (thiamine pyrophosphate) is the cofactor for: (1) Pyruvate dehydrogenase complex "
            "(PDC: converts pyruvate → acetyl-CoA; failure → lactate accumulation, inability to use "
            "glucose for energy); (2) 2-Oxoglutarate dehydrogenase (2OGDC in Krebs cycle: "
            "alpha-ketoglutarate → succinyl-CoA); (3) BCKDH (branched-chain AA catabolism); "
            "(4) Transketolase (pentose phosphate pathway). TPK1 deficiency → TPP cannot be "
            "synthesised from absorbed dietary thiamine → functional TPP deficiency → "
            "all four pathways fail in parallel → metabolic crisis with lactic acidosis + "
            "Leigh-like encephalopathy.\n\n"
            "CRITICAL DIAGNOSTIC TRAP: plasma/blood thiamine can be NORMAL or even ELEVATED "
            "(non-phosphorylated thiamine accumulates because it cannot be converted to TPP); "
            "MUST MEASURE THIAMINE PYROPHOSPHATE (TPP) SPECIFICALLY — TPP is critically low "
            "even when total thiamine appears normal. Many patients misdiagnosed as PDC deficiency "
            "or 2OGDC deficiency without measuring TPP or sequencing TPK1.\n\n"
            "BIOMARKERS: (1) Blood/CSF TPP LOW (PRIMARY — must request specifically); "
            "(2) Plasma thiamine elevated or normal (TRAP — does not reflect function); "
            "(3) Lactate elevated (especially during crises); pyruvate elevated; L:P ratio; "
            "(4) CSF lactate/pyruvate; (5) Organic acids: elevated 2-oxoglutarate, pyruvate, lactate; "
            "(6) Brain MRI: bilateral BG + brainstem T2 high (Leigh-like); DWI restriction acutely.\n\n"
            "TREATMENT: high-dose thiamine (100-600 mg/day in adults; 5-20 mg/kg/day in children) → "
            "mass action drives residual TPK1 activity; OR direct TPP supplementation (not widely "
            "commercially available). Carbohydrate restriction during crises. Ketogenic diet may help. "
            "Trigger avoidance: fever management (antipyretics), avoid fasting, avoid high carbohydrate load.\n\n"
            "MUTATIONS: p.Arg157Cys — most reported; p.Gly177Glu; splice variants affecting exon-intron boundaries."
        ),
        "hallmark": (
            "TPK1 HALLMARKS: "
            "(1) SERUM THIAMINE NORMAL OR HIGH — DIAGNOSTIC TRAP; must measure THIAMINE PYROPHOSPHATE (TPP). "
            "(2) EPISODIC CRISES: fever/illness/fasting triggers → lactic acidosis + encephalopathy. "
            "(3) MRI LEIGH-LIKE: bilateral BG + brainstem T2 high + DWI restriction during crises. "
            "(4) ALL FOUR TPP-DEPENDENT ENZYMES FAIL: PDC + 2OGDC + BCKDH + TKT in parallel. "
            "(5) HIGH-DOSE THIAMINE TREATMENT: 100-600mg/day drives residual TPK1 by mass action. "
            "(6) PYRUVATE ELEVATED (PDC failure) + 2-OXOGLUTARATE ELEVATED (2OGDC failure) together. "
            "(7) DDx WITH PDC DEFICIENCY: PDC enzyme assay may show secondary reduction; "
            "   TPK1 = all 4 pathways fail vs PDC = isolated PDC. "
            "(8) CARBOHYDRATE LOAD WORSENS: pyruvate load overwhelms failing PDC → crisis."
        ),
        "nbs_marker": "Not standard NBS; research: TPP in DBS by HPLC",
        "key_biomarker": "Blood/CSF TPP LOW (primary); serum thiamine normal/high (TRAP); lactate elevated; Leigh-like MRI",
        "severity_spectrum": "Variable: episodic encephalopathy to progressive Leigh syndrome; outcome better with early high-dose thiamine",
        "treatments": ["High-dose thiamine 5-20mg/kg/day", "Thiamine pyrophosphate (research)", "Ketogenic diet", "Crisis: carbohydrate restriction + thiamine IV"],
        "emergency": "Thiamine IV immediately in any Leigh-like crisis + measure TPP before and after",
        "ci_drugs": ["High carbohydrate loads (worsens crisis)", "Prolonged fasting", "Furosemide (thiamine depletor)", "Alcohol (B1 depletor)"],
    },
    # ── SLC19A3 — Biotin-Thiamine-Responsive Basal Ganglia Disease ───────────────────
    {
        "gene": "SLC19A3", "protein": "Thiamine Transporter 2 (THTR2)",
        "alias": "SLC19A3 — Biotin-Thiamine-Responsive Basal Ganglia Disease (BTBGD, OMIM #607483); EMERGENCY B1+B7",
        "aa": "496 aa", "kDa": "54 kDa",
        "gene_class": (
            "Solute carrier 19A3 (SLC19A3); thiamine transporter 2 (THTR2); 12-transmembrane domain "
            "transporter; primary transporter of thiamine across intestinal epithelium and blood-brain "
            "barrier; also has biotin transport capacity in certain tissues; deficiency → inadequate "
            "CNS thiamine uptake despite normal serum thiamine → cerebral thiamine deficiency → "
            "basal ganglia vulnerability (highest metabolic demand, most thiamine-dependent)"
        ),
        "vit_subgroup": "Vitamin B1 (Thiamine) + B7 (Biotin) — BTBGD / Biotin-Thiamine-Responsive BG Disease",
        "vitamin": "B1 (Thiamine) + B7 (Biotin) — BOTH REQUIRED; either alone is insufficient",
        "locus": "2q36.3", "omim_gene": 606152, "omim_disease": 607483,
        "inheritance": "AR. 2q36.3. Both sexes equally. Enriched in Middle East (Bahrain, Saudi Arabia founder mutations).",
        "seed_offset": 4,
        "onset_range_y": (0.3, 15.0),
        "phenotype": (
            "Episodic subacute encephalopathy triggered by fever, febrile illness, surgery, or trauma; "
            "typical onset 3 months to 10 years (broad range). "
            "During episode: confusion, ataxia, dystonia, dysarthria, dysphagia, ophthalmoplegia, "
            "seizures, respiratory compromise — LIFE-THREATENING. "
            "MRI SIGNATURE: bilateral symmetric T2 hyperintensity and DWI restriction in caudate + "
            "putamen + globus pallidus; brainstem tegmentum; thalami; cortex in severe cases. "
            "CRITICAL: WITHOUT TREATMENT — permanent basal ganglia necrosis → permanent dystonia, "
            "vegetative state, death. WITH PROMPT THIAMINE + BIOTIN → near-complete reversal of MRI "
            "changes AND clinical recovery even in severe episodes. "
            "Biotin alone or thiamine alone is INSUFFICIENT — BOTH vitamins required."
        ),
        "disease": (
            "SLC19A3 (496 aa, 54 kDa) is the primary thiamine transporter at the intestinal brush-border "
            "and blood-brain barrier. Unlike SLC19A2 (THTR1, ubiquitous), SLC19A3 is especially critical "
            "for CNS thiamine import. Deficiency → cerebral thiamine deficiency despite normal plasma "
            "thiamine → basal ganglia (highest metabolic rate per gram in brain) most vulnerable → "
            "bilateral basal ganglia necrosis during metabolic stress.\n\n"
            "WHY BIOTIN WORKS: biotin upregulates expression of SLC5A6 (SMVT, sodium multivitamin "
            "transporter), which also transports thiamine with lower affinity → partial compensation "
            "for lost SLC19A3 function. Neither vitamin alone provides complete correction.\n\n"
            "BIOMARKERS: (1) Plasma/CSF thiamine may be NORMAL or low-normal (transport defect, "
            "not supply defect — serum is not informative); (2) MRI is the PRIMARY diagnostic tool "
            "(bilateral BG DWI restriction in crisis); (3) CSF thiamine low (if measured); "
            "(4) Biotin levels may be mildly low in some patients; (5) Lactate may be elevated.\n\n"
            "TREATMENT: thiamine 5-10 mg/kg/day + biotin 5-10 mg/kg/day BOTH (oral or IV). "
            "EMERGENCY: IV thiamine 100-200mg + biotin 5-10mg/kg immediately for any febrile episode "
            "in a known patient or suspected new diagnosis → can reverse MRI changes within 48-72h "
            "if given promptly. Delay of even 12-24 hours can result in permanent damage. "
            "LONG-TERM: maintenance thiamine + biotin; fever protocol (increase doses at first fever).\n\n"
            "FOUNDER MUTATIONS: p.Ser194del — Bahraini founder; p.Gly172del — Saudi founder; "
            "p.Asn255Asp; p.Leu298Arg. Common in Middle East, North Africa, South Asian populations."
        ),
        "hallmark": (
            "SLC19A3/BTBGD HALLMARKS: "
            "(1) BILATERAL BG NECROSIS ON MRI — caudate + putamen + GP DWI restriction = EMERGENCY. "
            "(2) THIAMINE + BIOTIN BOTH REQUIRED — either alone INSUFFICIENT; give BOTH immediately. "
            "(3) IV THIAMINE + BIOTIN IS EMERGENCY REVERSAL TREATMENT — can reverse even severe episodes. "
            "(4) SERUM THIAMINE MAY BE NORMAL — transport defect; MRI > lab in diagnosis. "
            "(5) FEVER/ILLNESS TRIGGER — every febrile episode = EMERGENCY in known BTBGD. "
            "(6) FEVER PROTOCOL: increase thiamine+biotin dose immediately at any temperature >38°C. "
            "(7) MIDDLE EAST FOUNDER MUTATIONS — p.Ser194del (Bahrain); p.Gly172del (Saudi Arabia). "
            "(8) PROPHYLACTIC MAINTENANCE MANDATORY — cannot stop vitamins; relapse = permanent damage."
        ),
        "nbs_marker": "Not standard NBS; genetic screening in high-prevalence populations (Middle East)",
        "key_biomarker": "MRI bilateral BG DWI restriction (primary); serum thiamine unreliable; CSF thiamine low",
        "severity_spectrum": "Life-threatening acute episodes; with prompt treatment → near-complete recovery; untreated → permanent BG necrosis",
        "treatments": ["Thiamine 5-10mg/kg/day + Biotin 5-10mg/kg/day (BOTH)", "IV thiamine + biotin emergency protocol", "Fever protocol (dose increase at first fever)"],
        "emergency": "IV thiamine 100-300mg + biotin 5-10mg/kg immediately for ANY febrile crisis in BTBGD",
        "ci_drugs": ["Thiamine alone (insufficient)", "Biotin alone (insufficient)", "Corticosteroids (may impair thiamine transport)", "Furosemide (thiamine depletor)"],
    },
    # ── BTD — Biotinidase Deficiency ─────────────────────────────────────────────────
    {
        "gene": "BTD", "protein": "Biotinidase",
        "alias": "BTD — Biotinidase Deficiency (OMIM #253260); B7/biotin, NBS-detected, 4D completely preventable",
        "aa": "543 aa", "kDa": "61 kDa",
        "gene_class": (
            "Biotinidase; serum glycoprotein; cleaves biotin from biocytin (biotinyl-lysine) and "
            "biotinylated peptides released during normal protein turnover; recycles endogenous biotin; "
            "also cleaves dietary biotinyl-peptides in intestine; deficiency → biotin recycling fails → "
            "secondary biotin deficiency → all four biotin-dependent carboxylases lose activity: "
            "pyruvate carboxylase (PC), propionyl-CoA carboxylase (PCC), 3-methylcrotonyl-CoA carboxylase "
            "(MCC), and acetyl-CoA carboxylase (ACC)"
        ),
        "vit_subgroup": "Vitamin B7 (Biotin) — Biotinidase Deficiency",
        "vitamin": "B7 / Biotin (free biotin bypasses need for biotinidase recycling)",
        "locus": "3p25.1", "omim_gene": 609019, "omim_disease": 253260,
        "inheritance": "AR. 3p25.1. Both sexes equally. Incidence ~1/60,000-70,000 (profound); 1/40,000 (all forms). NBS standard.",
        "seed_offset": 5,
        "onset_range_y": (0.0, 10.0),
        "phenotype": (
            "UNTREATED — Profound deficiency (<1% activity): 4D signs: "
            "(1) Dermatitis (eczematoid rash, especially periorificial + scalp); "
            "(2) alopecia (diffuse hair loss); "
            "(3) Developmental delay (intellectual disability, regression); "
            "(4) sensorineural Deafness (hearing loss, irreversible once established). "
            "PLUS: seizures (mixed types, infantile spasms), ataxia, optic atrophy, "
            "spastic paraparesis, metabolic acidosis, organic aciduria, immunodeficiency. "
            "Partial deficiency (10-30% activity): milder; may present only under metabolic stress. "
            "NBS-DETECTED AND TREATED: completely ASYMPTOMATIC with normal development — "
            "most important point; biotin supplementation prevents ALL manifestations."
        ),
        "disease": (
            "BTD (543 aa, 61 kDa) is a serum glycoprotein that catalyses the hydrolysis of biocytin "
            "(biotin-epsilon-L-lysine) released during normal protein catabolism → free biotin recycled. "
            "In the intestine, BTD also releases biotin from dietary biotinyl-peptides. "
            "Deficiency → secondary biotin deficiency → all four biotin-dependent carboxylases fail: "
            "(1) Pyruvate carboxylase (PC): pyruvate → oxaloacetate; failure → lactic acidosis; "
            "(2) Propionyl-CoA carboxylase (PCC): propionyl-CoA → methylmalonyl-CoA; failure → propionic acid; "
            "(3) 3-Methylcrotonyl-CoA carboxylase (MCC): leucine catabolism; failure → 3-OH-isovalerate; "
            "(4) Acetyl-CoA carboxylase (ACC): fatty acid synthesis/regulation.\n\n"
            "BIOMARKERS: (1) Biotinidase enzyme activity (serum/plasma) — PRIMARY screening test; "
            "<10% = profound; 10-30% = partial; NBS uses fluorometric assay on DBS. "
            "(2) Organic acids: 3-OH-isovalerate, methylcitrate, 3-OH-propionate, lactate; "
            "(3) Plasma acylcarnitines: C5-OH elevated (3-methylcrotonyl-glycine); "
            "(4) Urine 3-OH-isovalerate; (5) Molecular testing BTD gene for confirmation and counselling.\n\n"
            "TREATMENT: free biotin 5-20 mg/day (not biotinyl compounds — BTD cannot cleave them); "
            "free biotin bypasses the recycling defect entirely. LIFELONG supplementation. "
            "CRITICAL: biotin must be CONTINUED THROUGH ILLNESS (absorptive demand increases). "
            "Sensorineural hearing loss: may be irreversible even with biotin treatment if already established "
            "— EARLY TREATMENT ESSENTIAL (NBS allows presymptomatic identification).\n\n"
            "KEY MUTATIONS: p.Asp444His (c.1330G>C) — most common pathogenic allele (~50% of mutations); "
            "profound deficiency. c.98_104del — 7-bp deletion; severe. p.Arg538Cys — common in Southern Europe; "
            "partial deficiency. Double gene mutations: often compound heterozygotes."
        ),
        "hallmark": (
            "BTD HALLMARKS: "
            "(1) 4D TRIAD: Dermatitis + alopecia + Developmental delay + sensorineural Deafness. "
            "(2) NBS STANDARD IN MOST COUNTRIES — biotinidase enzyme on DBS; "
            "   NBS-detected + treated → COMPLETELY NORMAL DEVELOPMENT. "
            "(3) FREE BIOTIN 5-20mg/day COMPLETELY PREVENTIVE — simple, cheap, effective. "
            "(4) BIOTINIDASE ENZYME NOT AFFECTED (contrast HLCS where biotinidase is NORMAL). "
            "(5) SENSORINEURAL HEARING LOSS IS IRREVERSIBLE once established — biotin reverses most but not hearing. "
            "(6) p.Asp444His MOST COMMON ALLELE (~50% of pathogenic variants in BTD). "
            "(7) ORGANIC ACID PROFILE: 3-OH-isovalerate + methylcitrate + 3-OH-propionate + lactate. "
            "(8) PARTIAL DEFICIENCY (10-30%): treat with biotin regardless; stressed state may trigger symptoms."
        ),
        "nbs_marker": "STANDARD NBS: biotinidase enzyme activity on DBS (fluorometric/colorimetric); first-tier screen",
        "key_biomarker": "Biotinidase enzyme activity <10% (profound) or <30% (partial); organic acids (3-OH-IVA, MCA); urine biotin metabolites",
        "severity_spectrum": "Profound (<1%): classic 4D + seizures; Partial (10-30%): mild/asymptomatic; NBS-detected: fully preventable",
        "treatments": ["Free biotin 5-20mg/day (lifelong)", "Increased dose during illness", "Hearing monitoring (audiometry 6-monthly)"],
        "emergency": "Biotin 10-20mg/day immediately in any metabolic crisis with organic aciduria + skin/hair changes",
        "ci_drugs": ["Biotin-containing compounds without free biotin (BTD cannot cleave them)", "Prolonged raw egg white diet (avidin blocks biotin absorption)", "Valproate (impairs biotin metabolism)"],
    },
    # ── HLCS — Holocarboxylase Synthetase / Multiple Carboxylase Deficiency ──────────
    {
        "gene": "HLCS", "protein": "Holocarboxylase Synthetase",
        "alias": "HLCS — Multiple Carboxylase Deficiency (MCD, OMIM #253270); biotin neonatal; BTD enzyme NORMAL",
        "aa": "726 aa", "kDa": "83 kDa",
        "gene_class": (
            "Holocarboxylase synthetase; mitochondrial and cytoplasmic isoforms; catalyses attachment "
            "of biotin to all four apocarboxylases (PC, PCC, MCC, ACC) using biotin + ATP → "
            "holocarboxylase + AMP + PPi; also biotinylates histones (epigenetic role); "
            "deficiency → biotin cannot be attached to any carboxylase despite normal biotin supply → "
            "all four carboxylases inactive → same metabolic profile as BTD but biotinidase is NORMAL"
        ),
        "vit_subgroup": "Vitamin B7 (Biotin) — Holocarboxylase Synthetase Deficiency / Multiple Carboxylase Deficiency",
        "vitamin": "B7 / Biotin (pharmacological dose; high-affinity mutants need high dose)",
        "locus": "21q22.13", "omim_gene": 609018, "omim_disease": 253270,
        "inheritance": "AR. 21q22.13. Both sexes equally. Rare; ~80 cases reported. Japanese founder effect.",
        "seed_offset": 6,
        "onset_range_y": (0.0, 1.0),
        "phenotype": (
            "Predominantly NEONATAL or early infantile onset (contrast BTD: more infantile/childhood). "
            "Neonatal form: severe metabolic acidosis within hours to days of birth; "
            "ketosis, vomiting, poor feeding, hypotonia, coma; often fatal if unrecognised. "
            "Infantile form: seizures, hypotonia, developmental delay, perioral/periorificial rash, alopecia. "
            "HIGH-AFFINITY HLCS MUTANTS: some mutations reduce biotin affinity (Km) → respond to "
            "high-dose biotin (10-40 mg/day); genetic subtype determines dose requirement. "
            "Same 4D signs as BTD possible but more commonly presents as neonatal metabolic crisis. "
            "Organic aciduria identical to BTD (see biomarkers). "
            "Biotinidase enzyme NORMAL (KEY DDx from BTD — same OA profile, different enzyme defect)."
        ),
        "disease": (
            "HLCS (726 aa, 83 kDa) attaches biotin covalently to apocarboxylases using biotin + ATP. "
            "Two-step reaction: (1) HLCS + biotin + ATP → HLCS•biotinyl-AMP + PPi; "
            "(2) HLCS•biotinyl-AMP + apocarboxylase-Lys → holocarboxylase + HLCS + AMP. "
            "Deficiency → all four carboxylases (PC, PCC, MCC, ACC) remain as inactive apoenzymes → "
            "same metabolic disruption as BTD but by a different mechanism: "
            "HLCS deficiency = biotin cannot be attached; BTD deficiency = biotin cannot be recycled. "
            "The organic acid profile and clinical features are IDENTICAL between HLCS and BTD.\n\n"
            "KEY DISTINCTION FROM BTD: Biotinidase enzyme activity is COMPLETELY NORMAL in HLCS deficiency. "
            "This is the most important laboratory differentiation.\n\n"
            "BIOMARKERS: IDENTICAL to BTD: (1) 3-OH-isovalerate; (2) methylcitric acid; "
            "(3) 3-OH-propionate; (4) lactic acid; (5) C5-OH acylcarnitine elevated; "
            "(6) biotinidase enzyme NORMAL (KEY DDx); (7) HLCS enzyme activity in fibroblasts — "
            "reduced with elevated Km (biotin requirement).\n\n"
            "BIOTIN TREATMENT: 10-40 mg/day (higher doses than BTD because mutations reduce HLCS "
            "affinity for biotin → pharmacological doses compensate by mass action). "
            "NEONATAL EMERGENCY: biotin 10-20 mg immediately in any neonate with metabolic acidosis "
            "+ organic aciduria + hair/skin changes. Response usually dramatic (hours to days). "
            "JAPANESE FOUNDER MUTATIONS: p.Leu237Pro and p.Val550Met — common in Japan, "
            "cause classical HLCS deficiency with high biotin Km.\n\n"
            "NBS: Not standard. Some countries with pilot MCD-NBS programs use organic acid markers."
        ),
        "hallmark": (
            "HLCS HALLMARKS: "
            "(1) BIOTINIDASE ENZYME NORMAL — KEY DDx FROM BTD (SAME ORGANIC ACID PROFILE, DIFFERENT ENZYME). "
            "(2) NEONATAL ONSET — more commonly neonatal than BTD (earlier, more acute metabolic crisis). "
            "(3) SAME ORGANIC ACID PROFILE AS BTD: 3-OH-IVA + methylcitrate + 3-OH-propionate + lactate. "
            "(4) HIGH-DOSE BIOTIN REQUIRED: 10-40mg/day (reduce Km of HLCS by mass action). "
            "(5) DRAMATIC BIOTIN RESPONSE — neonatal metabolic crisis reverses within 24-48h of biotin. "
            "(6) NOT ON STANDARD NBS — biotinidase assay NORMAL so NBS misses HLCS deficiency. "
            "(7) HISTONE BIOTINYLATION AFFECTED — epigenetic role of HLCS; neuronal impact. "
            "(8) JAPANESE FOUNDER: p.Leu237Pro + p.Val550Met — high prevalence in Japan."
        ),
        "nbs_marker": "NOT detected by standard biotinidase NBS (enzyme normal); organic acid NBS may detect (C5-OH, methylcitrate)",
        "key_biomarker": "Organic acids (3-OH-IVA, methylcitrate — identical to BTD); biotinidase NORMAL (key DDx); HLCS enzyme in fibroblasts",
        "severity_spectrum": "Neonatal severe (metabolic crisis, fatal untreated); infantile milder; high-dose biotin responsive",
        "treatments": ["Biotin 10-40mg/day (higher than BTD)", "Emergency: biotin 20mg immediately in neonatal crisis", "Dose guided by organic acid normalisation"],
        "emergency": "Biotin 10-20mg immediately in any neonatal metabolic acidosis with organic aciduria",
        "ci_drugs": ["Raw egg white (avidin; blocks biotin absorption)", "Valproate (biotin depletion)", "Long-term omeprazole (biotin malabsorption)"],
    },
    # ── SLC52A2 — BVVL Syndrome / Riboflavin Transporter 2 ───────────────────────────
    {
        "gene": "SLC52A2", "protein": "Riboflavin Transporter 2 (RFVT2)",
        "alias": "SLC52A2 — Brown-Vialetto-Van Laere Syndrome 2 (BVVL2, OMIM #614707); B2/riboflavin, MCAD-like acylcarnitines",
        "aa": "460 aa", "kDa": "51 kDa",
        "gene_class": (
            "Solute carrier family 52, member 2 (SLC52A2); riboflavin transporter 2 (RFVT2); "
            "11-transmembrane domain proton-coupled transporter; primary transporter of riboflavin "
            "(vitamin B2) across intestinal epithelium and blood-brain barrier; "
            "deficiency → riboflavin deficiency in CNS and peripheral nerves despite normal dietary intake → "
            "FAD/FMN cofactor deficiency → failure of all flavoenzymes including complex I (NADH dehydrogenase), "
            "ETF dehydrogenase (fatty acid oxidation), and multiple other FAD-dependent dehydrogenases"
        ),
        "vit_subgroup": "Vitamin B2 (Riboflavin) — BVVL Syndrome / Riboflavin Transporter Deficiency",
        "vitamin": "B2 / Riboflavin (pharmacological 10-40mg/kg/day); dramatic response possible",
        "locus": "8q24.13", "omim_gene": 607882, "omim_disease": 614707,
        "inheritance": "AR. 8q24.13. Both sexes equally. Rare; ~100 cases reported globally (BVVL1+2 combined).",
        "seed_offset": 7,
        "onset_range_y": (0.5, 25.0),
        "phenotype": (
            "BVVL = Brown-Vialetto-Van Laere syndrome: progressive bulbar palsy + sensorineural hearing loss. "
            "Onset: infantile to adult (broad spectrum; SLC52A2 tends to have earlier onset than SLC52A3). "
            "CARDINAL FEATURES: "
            "(1) Sensorineural hearing loss (often FIRST presenting sign; bilateral, progressive); "
            "(2) Bulbar palsy: dysarthria, dysphagia, facial weakness, tongue wasting; "
            "(3) Cranial nerve palsies: VII, IX, X, XII most common; VI (ophthalmoplegia); "
            "(4) Respiratory failure (bilateral phrenic nerve palsy → diaphragm weakness → "
            "   LIFE-THREATENING; often precipitates diagnosis); "
            "(5) Optic atrophy; "
            "(6) Limb weakness (posterior column + UMN); ataxia. "
            "ACYLCARNITINE PROFILE: C8, C10, C10:1, C12 elevated (MCAD-like pattern) — "
            "because ETF-QO (uses FAD) is impaired → medium-chain fatty acid oxidation blocked; "
            "CRITICAL: must not confuse with MCAD deficiency (MCAD enzyme normal in BVVL)."
        ),
        "disease": (
            "SLC52A2 (460 aa, 51 kDa) is the primary riboflavin transporter at the intestinal brush-border "
            "and blood-brain barrier. Deficiency → CNS/peripheral nerve riboflavin deficiency → "
            "failure of all FAD/FMN-dependent enzymes: (1) Electron transport chain complex I (FMN) → "
            "mitochondrial dysfunction; (2) Complex II (FAD) → Krebs cycle/ETC failure; "
            "(3) ETF-QO (electron transfer flavoprotein dehydrogenase, FAD) → fatty acid oxidation block → "
            "acylcarnitine profile mimics MCAD/MADD; (4) DHODH (pyrimidine synthesis); "
            "(5) Multiple other flavoenzymes.\n\n"
            "MCAD-LIKE ACYLCARNITINE PITFALL: C8, C10, C10:1 elevated in both MCAD and BVVL; "
            "in BVVL, MCAD enzyme assay is NORMAL (MCAD has FAD cofactor but is structurally intact; "
            "FAD cofactor is depleted → MCAD cannot function → same acylcarnitine pattern). "
            "NBS: BVVL can be picked up on NBS by elevated C8 → must investigate further if NBS-positive "
            "for MCAD but MCAD enzyme activity is normal.\n\n"
            "BRAINSTEM MRI: T2 hyperintensity in dorsal brainstem (nuclei of CN VII, IX, X, XII); "
            "auditory brainstem response (ABR) typically abnormal (prolonged or absent wave V). "
            "NCS/EMG: motor axonal neuropathy (cranial > peripheral).\n\n"
            "BIOMARKERS: (1) Acylcarnitines: C8, C10, C10:1 elevated (MCAD-like — CRITICAL pitfall); "
            "(2) Plasma riboflavin LOW; (3) MCAD enzyme assay NORMAL (KEY DDx from MCAD); "
            "(4) Urine organic acids: ethylmalonic acid, glutaric acid, 2-OH-glutaric acid "
            "(MADD-like pattern in severe deficiency); (5) Flavin profiling: FMN, FAD low.\n\n"
            "RIBOFLAVIN TREATMENT: 10-40 mg/kg/day (pharmacological dose far above dietary RDA of 1-2 mg/day); "
            "even patients with severe bulbar palsy + respiratory failure can show DRAMATIC REVERSAL of "
            "neurological deficits within weeks of riboflavin at pharmacological doses. "
            "Hearing loss: may be partially reversible with early treatment; late irreversible.\n\n"
            "SLC52A3 vs SLC52A2: SLC52A3 (BVVL1, OMIM #211530) causes identical syndrome; "
            "SLC52A2 (BVVL2) has earlier onset; both respond to riboflavin."
        ),
        "hallmark": (
            "SLC52A2/BVVL HALLMARKS: "
            "(1) MCAD-LIKE ACYLCARNITINE PROFILE (C8/C10 elevated) + NEUROLOGICAL DISEASE → think BVVL. "
            "(2) MCAD ENZYME NORMAL — KEY DDx from MCAD deficiency (enzyme intact but FAD-depleted). "
            "(3) SENSORINEURAL HEARING LOSS OFTEN FIRST — bilateral progressive; may precede bulbar by years. "
            "(4) RESPIRATORY FAILURE FROM DIAPHRAGM WEAKNESS — life-threatening; monitor spirometry. "
            "(5) HIGH-DOSE RIBOFLAVIN 10-40mg/kg/day → DRAMATIC REVERSAL even in advanced disease. "
            "(6) BRAINSTEM MRI: dorsal brainstem T2 high (CN nuclei); auditory brainstem responses absent. "
            "(7) NBS PICK-UP POSSIBLE via C8 elevation → confirm by MCAD enzyme assay + SLC52A2 sequencing. "
            "(8) ETF-QO (FAD-dependent) impairment → MADD-like urine OA (ethylmalonic, glutaric acids)."
        ),
        "nbs_marker": "May trigger MCAD NBS flag (C8 elevated); MCAD enzyme assay then distinguishes; SLC52A2 sequencing confirms",
        "key_biomarker": "C8/C10 acylcarnitines elevated (MCAD-like); MCAD enzyme NORMAL (key DDx); plasma riboflavin low; urine ethylmalonic/glutaric acid",
        "severity_spectrum": "Progressive bulbar + respiratory failure (severe); hearing loss often irreversible; riboflavin can reverse bulbar even if late",
        "treatments": ["Riboflavin 10-40mg/kg/day (pharmacological)", "Hearing aids/cochlear implants", "Ventilatory support (BiPAP/invasive)", "Swallowing therapy + PEG"],
        "emergency": "Riboflavin 10-20mg/kg/day immediately in any suspected BVVL + respiratory compromise",
        "ci_drugs": ["Probenecid (impairs SLC52A2 at low concentrations)", "Phenothiazines (riboflavin antagonist at high doses)", "Prolonged riboflavin-deficient diet"],
    },
]


def _make_patients(gene_data):
    """Generate deterministic 40-patient cohort for a vitamin-responsive epilepsy gene."""
    gene = gene_data["gene"]
    rng = random.Random(SEED_BASE + gene_data["seed_offset"])
    patients = []
    onset_lo, onset_hi = gene_data["onset_range_y"]

    for i in range(40):
        sex = rng.choice(["M", "F"])
        age_onset = round(rng.uniform(onset_lo, onset_hi), 2)
        dx_delay = round(rng.uniform(0.1, 5.0), 2)
        age_dx = round(age_onset + dx_delay, 2)
        age_now = round(age_dx + rng.uniform(0.5, 15.0), 2)
        severity = rng.choices(
            ["Severe", "Moderate", "Mild"],
            weights=[40, 40, 20], k=1
        )[0]
        vitamin = gene_data["vitamin"].split("/")[0].strip()
        tx_map = {
            "ALDH7A1": rng.choice(["Pyridoxine + lysine restriction + folinic acid", "Pyridoxine only", "Pyridoxine + lysine restriction", "PLP + lysine restriction"]),
            "PNPO":    rng.choice(["PLP monotherapy", "PLP + folinic acid", "PLP high-dose", "Supportive (died before Dx)"]),
            "PLPBP":   rng.choice(["PLP", "Pyridoxine", "PLP + pyridoxine", "PLP + folinic acid"]),
            "TPK1":    rng.choice(["High-dose thiamine 5-10mg/kg/day", "Thiamine + KD", "Thiamine + riboflavin", "Thiamine IV acute"]),
            "SLC19A3": rng.choice(["Thiamine + biotin", "Thiamine + biotin + fever protocol", "Thiamine alone (suboptimal)", "Biotin + thiamine IV"]),
            "BTD":     rng.choice(["Free biotin 10mg/day", "Free biotin 5mg/day", "Free biotin 20mg/day", "NBS-detected: prophylactic biotin"]),
            "HLCS":    rng.choice(["Biotin 20-40mg/day", "Biotin 10mg/day", "Biotin 40mg/day + organic acid monitoring", "Biotin neonatal emergency"]),
            "SLC52A2": rng.choice(["Riboflavin 10-40mg/kg/day", "Riboflavin + ventilatory support", "Riboflavin monotherapy", "Riboflavin + hearing aids"]),
        }[gene]
        outcome = rng.choice(["Seizure-free", "Seizure reduction >50%", "Partial response", "Treatment-refractory", "NDD despite control"])
        patients.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age_onset_y": age_onset,
            "age_dx_y": age_dx,
            "age_now_y": age_now,
            "dx_delay_y": round(dx_delay, 2),
            "sex": sex,
            "severity": severity,
            "vitamin": vitamin,
            "treatment": tx_map,
            "outcome": outcome,
        })
    return patients


def _all_patients():
    all_pts = []
    for g in VIT_GENES:
        all_pts.extend(_make_patients(g))
    return all_pts


# ─── Public API ───────────────────────────────────────────────────────────────

def get_overview():
    pts = _all_patients()
    n = len(pts)
    sev_counts = {"Severe": 0, "Moderate": 0, "Mild": 0}
    for p in pts:
        sev_counts[p["severity"]] += 1
    avg_delay = round(sum(p["dx_delay_y"] for p in pts) / n, 2)
    avg_onset = round(sum(p["age_onset_y"] for p in pts) / n, 2)
    gene_counts = {}
    for p in pts:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
    return {
        "atlas": "Vitamin-Responsive Epileptic Encephalopathies Atlas",
        "subtitle": "Complete 8-Gene Vitamin-Responsive Epileptic Encephalopathies Reference",
        "description": (
            "The Vitamin-Responsive Epileptic Encephalopathies Atlas covers 8 treatable genetic epilepsies "
            "where specific vitamin cofactors are both the primary treatment and the window to normal "
            "development. B6-responsive (ALDH7A1/PDE, PNPO/PLP-neonatal, PLPBP), B1-responsive "
            "(TPK1/thiamine pyrophosphokinase, SLC19A3/BTBGD requiring B1+B7), B7-responsive "
            "(BTD/biotinidase, HLCS/multiple carboxylase), and B2-responsive (SLC52A2/BVVL). "
            "The critical principle: TREAT BEFORE DIAGNOSIS IS CONFIRMED."
        ),
        "genes": [g["gene"] for g in VIT_GENES],
        "n_genes": len(VIT_GENES),
        "total_patients": n,
        "avg_onset_y": avg_onset,
        "avg_dx_delay_y": avg_delay,
        "severity_distribution": sev_counts,
        "gene_counts": gene_counts,
        "vitamin_groups": {
            "B6 (Pyridoxine/PLP)": ["ALDH7A1", "PNPO", "PLPBP"],
            "B1 (Thiamine)": ["TPK1", "SLC19A3"],
            "B7 (Biotin)": ["BTD", "HLCS"],
            "B2 (Riboflavin)": ["SLC52A2"],
        },
        "key_teaching": [
            "TREAT BEFORE DIAGNOSIS: empiric vitamin trial while awaiting confirmatory testing",
            "B6 FORMS MATTER: PDE (ALDH7A1) → pyridoxine; PNPO → PLP ONLY (pyridoxine FAILS/WORSENS)",
            "AASA PATHOGNOMONIC for ALDH7A1/PDE; AASA NORMAL in PNPO and PLPBP",
            "TPK1: serum thiamine NORMAL or HIGH — TRAP; measure TPP specifically (thiamine pyrophosphate)",
            "SLC19A3/BTBGD: thiamine + biotin BOTH required; either alone insufficient; MRI BG necrosis = emergency",
            "BTD vs HLCS: same organic acid profile; biotinidase enzyme NORMAL in HLCS (KEY DDx)",
            "SLC52A2/BVVL: C8/C10 acylcarnitines mimic MCAD; MCAD enzyme NORMAL in BVVL",
            "NBS catches BTD reliably; HLCS MISSED by biotinidase NBS; TPK1/SLC19A3/BVVL not on standard NBS",
            "HEARING LOSS in BTD and BVVL may be IRREVERSIBLE — early treatment essential",
            "RIBOFLAVIN can reverse advanced bulbar palsy + respiratory failure in BVVL",
        ],
        "emergency_summary": {
            "ALDH7A1": "Pyridoxine 100mg IV — ANY neonatal refractory seizure",
            "PNPO":    "PLP 30mg/kg oral/IV — if pyridoxine trial fails → PLP immediately",
            "PLPBP":   "PLP 30mg/kg — B6-responsive epilepsy not responding fully to pyridoxine",
            "TPK1":    "Thiamine IV + measure TPP — Leigh-like crisis with fever trigger",
            "SLC19A3": "Thiamine IV + biotin immediately — BG necrosis on MRI at ANY fever",
            "BTD":     "Biotin 10-20mg/day — metabolic acidosis + organic aciduria + alopecia/rash",
            "HLCS":    "Biotin 20mg neonatal — metabolic acidosis in first days + normal biotinidase",
            "SLC52A2": "Riboflavin 20mg/kg/day — progressive bulbar + hearing loss + C8/C10 elevated",
        },
        "seeds_used": f"{SEED_BASE}–{SEED_BASE + len(VIT_GENES) - 1}",
    }


def get_breakdown():
    result = []
    for gd in VIT_GENES:
        pts = _make_patients(gd)
        sev = {"Severe": 0, "Moderate": 0, "Mild": 0}
        for p in pts:
            sev[p["severity"]] += 1
        avg_delay = round(sum(p["dx_delay_y"] for p in pts) / len(pts), 1)
        avg_onset = round(sum(p["age_onset_y"] for p in pts) / len(pts), 1)
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        top_tx = sorted(treatments.items(), key=lambda x: -x[1])[:3]
        outcomes = {}
        for p in pts:
            outcomes[p["outcome"]] = outcomes.get(p["outcome"], 0) + 1
        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "vit_subgroup": gd["vit_subgroup"],
            "vitamin": gd["vitamin"],
            "n_patients": len(pts),
            "severity_distribution": sev,
            "avg_onset_y": avg_onset,
            "avg_dx_delay_y": avg_delay,
            "top_treatments": [{"treatment": t, "n": n} for t, n in top_tx],
            "outcome_distribution": outcomes,
            "hallmark": gd["hallmark"],
            "nbs_marker": gd["nbs_marker"],
            "key_biomarker": gd["key_biomarker"],
            "severity_spectrum": gd["severity_spectrum"],
            "emergency": gd["emergency"],
            "ci_drugs": gd["ci_drugs"],
            "disease_summary": gd["disease"][:600] + "…",
            "phenotype_summary": gd["phenotype"][:400] + "…",
            "gene_class": gd["gene_class"][:400] + "…",
        })
    return {"breakdown": result, "total_genes": len(result)}


def get_definitions():
    return {
        "definitions": [
            {"term": "Pyridoxal-5'-Phosphate (PLP)", "definition": "The active, phosphorylated coenzyme form of vitamin B6 (pyridoxine); cofactor for >160 enzymes including GABA synthesis (GAD), aromatic amino acid decarboxylase (AADC/DOPA decarboxylase), glycine cleavage system, transamination reactions; dietary pyridoxine must be converted to PLP by pyridoxal kinase + PNPO."},
            {"term": "Alpha-Aminoadipic Semialdehyde (AASA)", "definition": "Intermediate in L-lysine catabolism via the saccharopine pathway; catabolised by ALDH7A1 (antiquitin); accumulates in PDE (ALDH7A1 deficiency) and inactivates PLP via Knoevenagel condensation with P6C; PATHOGNOMONIC biomarker for PDE/ALDH7A1 (measured in urine by LC-MS/MS); NORMAL in PNPO and PLPBP — critical diagnostic distinction."},
            {"term": "P6C (1-Piperideine-6-Carboxylate)", "definition": "Cyclic tautomer of AASA; the reactive species that covalently inactivates PLP in PDE (ALDH7A1 deficiency) via Knoevenagel condensation; explains why dietary pyridoxine is insufficient — PLP is continuously inactivated by P6C as fast as it is supplied; pharmacological doses of pyridoxine overwhelm P6C by mass action."},
            {"term": "Pyridoxine-Dependent Epilepsy (PDE)", "definition": "Refractory neonatal/infantile epilepsy caused by ALDH7A1 mutations; AASA/P6C accumulate → PLP inactivated; RESPONDS TO PYRIDOXINE 100mg IV (diagnosis + treatment); urine AASA pathognomonic; triple therapy: pyridoxine + lysine restriction + folinic acid; seizures cease, but intellectual disability persists without lysine restriction."},
            {"term": "Pyridox(am)ine Phosphate Oxidase (PNPO)", "definition": "FMN-containing enzyme that converts PNP and PMP → PLP (final step in PLP synthesis); deficiency causes primary PLP deficiency; CRITICAL: responds to PLP but NOT pyridoxine (pyridoxine → PNP via pyridoxal kinase → PNPO needed to make PLP → if PNPO absent, PNP accumulates and competitively inhibits residual PNPO); CSF PLP markedly low; AASA normal."},
            {"term": "Thiamine Pyrophosphate (TPP)", "definition": "The active diphosphorylated coenzyme form of vitamin B1 (thiamine); synthesised from thiamine + ATP by thiamine pyrophosphokinase 1 (TPK1); essential cofactor for PDC (pyruvate dehydrogenase complex), 2OGDC, BCKDH, and transketolase; TPK1 deficiency: serum thiamine NORMAL or HIGH (cannot be phosphorylated) but TPP critically low — MUST MEASURE TPP SPECIFICALLY."},
            {"term": "Biotin-Thiamine-Responsive Basal Ganglia Disease (BTBGD)", "definition": "Life-threatening episodic encephalopathy caused by SLC19A3 (THTR2/thiamine transporter 2) mutations; MRI bilateral BG necrosis (DWI restriction) during fever-triggered crises; EMERGENCY TREATMENT: IV thiamine + biotin BOTH immediately; biotin works by upregulating SLC5A6 (SMVT) which partially compensates for lost SLC19A3 function; Middle East founder mutations (Bahrain, Saudi Arabia)."},
            {"term": "Biocytin", "definition": "Biotin-epsilon-L-lysine; released by proteolytic digestion of biotinylated carboxylases during protein turnover; recycled by biotinidase (BTD) back to free biotin; BTD deficiency → biocytin accumulates → secondary biotin deficiency → all four carboxylases (PC, PCC, MCC, ACC) fail; free dietary biotin supplementation bypasses BTD defect entirely."},
            {"term": "Biotinidase vs HLCS: Key DDx", "definition": "Both cause identical organic acid profiles (3-OH-isovalerate + methylcitrate + 3-OH-propionate + lactate) and similar clinical features (seizures, alopecia, rash). CRITICAL distinction: Biotinidase enzyme activity — LOW in BTD deficiency (enzyme absent); NORMAL in HLCS deficiency (biotinidase intact, HLCS cannot attach biotin to carboxylases). Age of onset: HLCS typically neonatal; BTD typically infantile-childhood. NBS: BTD detected (biotinidase assay); HLCS missed (biotinidase is normal)."},
            {"term": "Holocarboxylase Synthetase (HLCS)", "definition": "Enzyme that attaches biotin covalently to all four apocarboxylases (PC, PCC, MCC, ACC) and to histones; deficiency → all carboxylases remain as inactive apoenzymes regardless of biotin supply; HLCS-deficiency responds to higher biotin doses (pharmacological) because biotin Km of mutant HLCS is elevated; biotinidase activity is NORMAL (BTD enzyme is intact)."},
            {"term": "Brown-Vialetto-Van Laere (BVVL) Syndrome", "definition": "Progressive pontobulbar palsy + sensorineural hearing loss caused by SLC52A2 (BVVL2) or SLC52A3 (BVVL1) mutations; riboflavin transporter deficiency → CNS/PNS FAD/FMN depletion → ETF-QO failure → acylcarnitine profile mimics MCAD (C8, C10 elevated); high-dose riboflavin 10-40mg/kg/day causes dramatic reversal even of advanced bulbar palsy; MCAD enzyme is NORMAL in BVVL."},
            {"term": "Riboflavin Transporter Deficiency (RTD)", "definition": "Collective term for SLC52A1 (RTD1), SLC52A2 (RTD2/BVVL2), and SLC52A3 (RTD3/BVVL1) deficiencies; all cause riboflavin import failure into brain/PNS despite normal dietary intake; treated with pharmacological riboflavin 10-40mg/kg/day bypassing the transporter by passive diffusion at high concentrations; FAD/FMN supplementation investigational."},
            {"term": "MCAD-like Acylcarnitine Pitfall in BVVL", "definition": "SLC52A2/BVVL presents with C8, C10, C10:1 acylcarnitines elevated — identical to medium-chain acyl-CoA dehydrogenase (MCAD) deficiency. Mechanism: MCAD uses FAD cofactor; FAD depleted in BVVL → MCAD structurally intact but functionally unable to catalyse → medium-chain accumulation. DDx: MCAD enzyme activity NORMAL in BVVL; neurological features (hearing loss, bulbar palsy) absent in MCAD; riboflavin trial therapeutic in BVVL."},
            {"term": "Empiric Vitamin Trial", "definition": "Clinical strategy of administering specific vitamins BEFORE diagnostic confirmation in suspected vitamin-responsive epileptic encephalopathies; justified because: (1) treatment delay = irreversible neuronal damage; (2) vitamins are safe at therapeutic doses; (3) response is diagnostic. Protocol: pyridoxine 100mg IV → if fails → PLP 30mg/kg → CSF for PLP/AASA; thiamine 100mg IV → biotin 10mg in any BG necrosis; riboflavin 20mg/kg in BVVL-like presentation."},
            {"term": "Lysine Restriction in PDE", "definition": "Dietary restriction of lysine (the amino acid upstream of AASA in L-lysine catabolism via saccharopine pathway) in ALDH7A1/PDE; reduces AASA generation → less P6C formed → less PLP inactivation; improves neurodevelopmental outcomes independently of pyridoxine; part of 'triple therapy' (pyridoxine + lysine restriction + folinic acid); strict lysine restriction targets plasma lysine 50-100 nmol/mL."},
            {"term": "Fever Protocol in Vitamin Transportopathies", "definition": "Emergency plan for patients with SLC19A3/BTBGD and SLC52A2/BVVL: at first temperature >38°C → immediately increase vitamin dose (thiamine + biotin for SLC19A3; riboflavin for SLC52A2) to 2-3× maintenance; if not responsive → hospital immediately for IV administration; in BTBGD, failure to act within hours → irreversible BG necrosis; all families must have fever protocol in writing with emergency contacts."},
            {"term": "4D Signs of Biotinidase Deficiency", "definition": "Classic untreated BTD deficiency features: (1) Dermatitis — eczematoid periorificial and scalp rash; (2) alopecia — diffuse or patchy hair loss; (3) Developmental delay — intellectual disability and regression; (4) sensorineural Deafness — bilateral progressive hearing loss (may be irreversible). PLUS seizures, ataxia, optic atrophy. With NBS and early biotin supplementation — ALL manifestations are completely preventable."},
        ]
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("=== VITAMIN ATLAS OVERVIEW ===")
    print(f"Genes: {ov['genes']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Avg onset: {ov['avg_onset_y']} y")
    print(f"Avg dx delay: {ov['avg_dx_delay_y']} y")
    print(f"Vitamin groups: {ov['vitamin_groups']}")
    bd = get_breakdown()
    print(f"\n=== BREAKDOWN ({bd['total_genes']} genes) ===")
    for g in bd["breakdown"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, onset {g['avg_onset_y']}y, delay {g['avg_dx_delay_y']}y")
    defs = get_definitions()
    print(f"\n=== DEFINITIONS: {len(defs['definitions'])} terms ===")
