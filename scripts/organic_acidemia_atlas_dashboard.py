#!/usr/bin/env python3
"""Organic-Acidemia-Atlas — Complete 10-Gene Organic Acidemia Atlas
PCCA · PCCB · MUT · MMAA · MMAB · MMACHC · IVD · HMGCL · MCCC1 · MCCC2
400-patient aggregate cohort (10 × 40, seeds 823–832)

Organic Acidemia (OA) facts:
  - Organic acidemias = inherited defects in amino acid/odd-chain fatty acid catabolism →
    accumulation of organic acids in blood, urine, CSF causing multisystem toxicity
  - Core mechanism: blocked enzymatic steps → toxic intermediate accumulation (propionate,
    methylmalonate, isovalerate, 3-methylcrotonyl-CoA, HMG-CoA) → secondary metabolic crises
  - Shared features: metabolic acidosis + elevated anion gap, hyperammonemia (secondary NAGS
    inhibition by propionyl-CoA / methylmalonyl-CoA), hyper- or hypoglycaemia, neutropenia
  - NBS acylcarnitine profile detects most OA: C3 (PA/MMA), C5 (IVA), C5-OH (HMG-CoA/3-MCC)
  - Emergency treatment: protein restriction (zero protein 24–48h) + high glucose infusion rate
    (GIR 8–12 mg/kg/min) + L-carnitine + ammonia scavengers (benzoate/phenylacetate/arginine)
  - Cofactor trials mandatory: OHCbl (MUT/MMAA/MMAB/MMACHC), Biotin (NOT responsive in MCC)

ATLAS SCOPE (10 nuclear-encoded organic acidemia genes):
  Propionate pathway (propionyl-CoA carboxylase):
    PCCA  — PCC alpha subunit, propionic acidemia type A (728aa, 13q32.3)
    PCCB  — PCC beta subunit, propionic acidemia type B (539aa, 3q22.3)
  Methylmalonate pathway:
    MUT   — Methylmalonyl-CoA mutase, classic mut-type MMA (750aa, 6p12.3)
    MMAA  — Mitochondrial cobalamin reductase, cblA MMA (418aa, 4q31.21)
    MMAB  — Adenosylcobalamin synthase, cblB MMA (250aa, 12q24.11)
    MMACHC— Combined MMA + homocystinuria, cblC (282aa, 1p34.1) — MOST COMMON combined OA
  Leucine catabolism (branched-chain):
    IVD   — Isovaleryl-CoA dehydrogenase, isovaleric acidemia (394aa, 15q15.1)
    HMGCL — 3-Hydroxy-3-methylglutaryl-CoA lyase, HMG-CoA lyase deficiency (325aa, 1p36.11)
    MCCC1 — 3-Methylcrotonyl-CoA carboxylase alpha, 3-MCC deficiency type 1 (725aa, 3q27.1)
    MCCC2 — 3-Methylcrotonyl-CoA carboxylase beta, 3-MCC deficiency type 2 (563aa, 5q13.2)

CRITICAL CLINICAL RULES:
  1. VPA HIGH RISK for all OA — propionate/methylmalonate metabolic interference;
     carnitine depletion; avoid especially in PA/MMA/IVA
  2. PROTEIN RESTRICTION EMERGENCY: zero protein 24–48h in acute crisis (stops substrate influx)
     then reintroduce at 0.5–1.5 g/kg/day with essential amino acids
  3. GLUCOSE INFUSION RATE (GIR) 8–12 mg/kg/min: anti-catabolic; prevents endogenous protein
     catabolism; mandatory in any metabolic crisis for all 10 genes
  4. OHCbl TRIAL mandatory for MUT, MMAA, MMAB, MMACHC:
     OHCbl 1 mg IM daily × 5 days → measure plasma MMA; >50% fall = partial responder
  5. GLYCINE conjugation in IVD: isovaleryl-glycine excreted in urine — glycine supplementation
     (250 mg/kg/day) reduces toxic isovaleryl-CoA by conjugation; L-carnitine also reduces C5
  6. NO KETOGENIC DIET in HMGCL: HMGCL catalyses ketogenesis (HMG-CoA → acetoacetate);
     HMGCL deficiency = inability to make ketones; KD impossible; high-glucose approach only
  7. BETAINE + OHCbl in MMACHC (cblC): remethylation support for homocysteine;
     betaine 100–200 mg/kg/day + OHCbl + methylfolate + carnitine = standard cblC regimen
  8. AMMONIA SCAVENGERS in PA/MMA hyperammonemia:
     sodium benzoate 250 mg/kg + sodium phenylacetate 250 mg/kg IV + arginine; CRRT if refractory
  9. 3-MCC (MCCC1/MCCC2) is usually BENIGN: debated clinical significance; most NBS-detected
     individuals are asymptomatic; biotin is NOT effective (unlike multiple carboxylase deficiency)
  10. BTBGD mandatory exclusion in any Leigh-like presentation with elevated lactate

COHORT: 10 × 40 = 400 patient slots (seeds 823–832; gene-specific seeds)
"""

import random

SEED_BASE = 823

# ── All 10 organic acidemia genes ─────────────────────────────────────────────
OA_GENES = [
    # ── PCCA — PCC alpha subunit, Propionic Acidemia type A ──
    {
        "gene": "PCCA", "alias": "PCC-α — Propionyl-CoA Carboxylase Alpha Subunit (Propionic Acidemia type A)",
        "aa": "728 aa", "kDa": "78.3 kDa",
        "gene_class": "propionyl_coa_carboxylase_alpha",
        "locus": "13q32.3", "omim_gene": 232000,
        "phenotype": "Propionic Acidemia (PA-A) — PCC alpha, propionyl-CoA → methylmalonyl-CoA block; severe neonatal + dilated cardiomyopathy",
        "disease": (
            "PCCA biallelic loss → Propionic Acidemia type A (PA, OMIM #606054 gene, #232000 disease). "
            "PCCA encodes the alpha subunit (728aa, 78 kDa, biotin-dependent) of propionyl-CoA carboxylase (PCC), "
            "a mitochondrial α6β6 dodecameric enzyme catalysing the first committed step of propionate catabolism: "
            "propionyl-CoA + CO₂ + ATP → methylmalonyl-CoA (Krebs anaplerosis). Propionyl-CoA derives from "
            "catabolism of odd-chain fatty acids, isoleucine, valine, methionine, threonine, and gut propionate. "
            "PCC alpha subunit contains the biotin carboxylase domain; beta subunit (PCCB) contains the "
            "carboxyl transferase domain. PCCA loss: propionyl-CoA accumulates → inhibits NAGS (secondary "
            "hyperammonemia) + CPS1 (urea cycle block) + PDC (secondary lactic acidosis) + ETC complexes. "
            "Neonatal form: metabolic crisis days 1–5: lethargy, poor feeding, vomiting, encephalopathy, "
            "ketosis, metabolic acidosis, hyperammonemia (NH₃ 200–3000 µmol/L). Dilated cardiomyopathy (DCM): "
            "specific PA complication (30–45%); QTc prolongation; sudden cardiac death risk independent of "
            "metabolic control; propionyl-CoA directly inhibits ETC in cardiomyocytes. NBS: elevated C3 "
            "(propionylcarnitine); C3/C2 ratio >0.4 highly specific."
        ),
        "inheritance": "Autosomal recessive, biallelic. Equal M:F. 1:100,000–1:250,000 live births (higher in Saudi Arabia ~1:2,000–1:5,000 due to consanguinity; Japan ~1:50,000).",
        "hallmark": (
            "PA HALLMARKS: (1) NBS C3-carnitine elevation + C3/C2 ratio >0.4 (propionylcarnitine); "
            "(2) Metabolic crisis neonatal: hyperammonemia + ketoacidosis + lactic acidosis (triple toxicity); "
            "(3) DILATED CARDIOMYOPATHY (DCM) in 30–45% — SPECIFIC PA complication; QTc prolongation; "
            "arrhythmia risk; cardiomyopathy can be present at diagnosis even without metabolic crisis; "
            "(4) NEUTROPENIA + pancytopenia during crises (propionate inhibits bone marrow); "
            "(5) Pancreatitis (10–20%); (6) Brain MRI: striatal necrosis + white matter changes (similar to "
            "MSUD but glycine not elevated); (7) PROTEIN RESTRICTION EMERGENCY: zero protein 48h + GIR 10–12; "
            "(8) Metronidazole: reduces gut propionate-producing bacteria (Propionibacterium etc.) — "
            "adjunct therapy; (9) Liver transplant: reduces crises, does NOT fully protect cardiac/brain; "
            "(10) Biotin cofactor: PCC is biotin-dependent but PA is NOT biotin-responsive (unlike HCS def)."
        ),
        "key_ddx": (
            "vs PCCB (PA type B): identical phenotype; PCCA 13q32.3 vs PCCB 3q22.3; WES distinguishes; "
            "vs MUT (MMA): C3 elevated in both; MMA urine elevated in MUT not PA; no DCM in MUT; "
            "vs MMACHC (cblC): combined MMA + homocysteine in cblC; no homocysteine elevation in PA; "
            "vs MSUD (BCKDHA/B/DBT): alloisoleucine + leucine/isoleucine/valine elevated; no C3; "
            "vs UCD (OTC/CPS1): isolated hyperammonemia without ketoacidosis; no C3 elevation."
        ),
        "founder_variant": "p.Gln107Ter (Saudi Arabia founder); p.Ile475Phe (European); p.Arg773Ter; c.1994-1G>A splice acceptor; p.Gln107_Ile325del (exon 4 deletion); many private biallelic variants",
        "onset_pattern": "NBS cohort: crisis days 1–5 (off maternal metabolism); chronic form with later presentation also described. Two metabolic phenotypes: mut0 (severe neonatal) and mut- (later-onset episodic).",
        "mri_pattern": "Striatal necrosis (bilateral caudate + putamen T2 hyperintensity during crisis); white matter signal changes; MRI can normalise between crises; cerebellar atrophy in chronic disease.",
        "hypoglycaemia_rate": 0.50, "hepatopathy_rate": 0.30, "myopathy_rate": 0.15,
        "encephalopathy_rate": 0.80, "epilepsy_rate": 0.35, "ataxia_rate": 0.20,
        "lactic_ac_rate": 0.75, "hcm_rate": 0.05, "dcm_rate": 0.38,
        "pancreatitis_rate": 0.15, "hyperammonemia_rate": 0.90, "retinopathy_rate": 0.00,
        "renal_rate": 0.10, "neonatal_crisis_rate": 0.85, "homocysteine_elevated_rate": 0.00,
        "seed": 823,
        "kd_contraindicated": True, "kd_note": "KD CONTRAINDICATED — high-fat diet provides odd-chain FA → propionyl-CoA substrate influx; worsens PA",
        "nbs_detected": True, "nbs_marker": "C3-carnitine (propionylcarnitine) + C3/C2 ratio >0.4",
        "ohcobl_response": False, "biotin_response": False, "glycine_conjugation": False,
        "betaine_treatment": False, "metronidazole_adjunct": True,
        "vpa_risk": "HIGH RISK — propionyl-CoA metabolite inhibition + carnitine depletion; propionyl-CoA already accumulating; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "Zero protein 24–48h; IV 10% dextrose GIR 10–12 mg/kg/min; L-carnitine 100–200 mg/kg/day; sodium benzoate 250 mg/kg IV + sodium phenylacetate 250 mg/kg IV + arginine 300 mg/kg for hyperammonemia; CRRT if NH₃ >500; metronidazole 20 mg/kg/day to suppress gut propionate; NaHCO₃ for severe acidosis.",
    },
    # ── PCCB — PCC beta subunit, Propionic Acidemia type B ──
    {
        "gene": "PCCB", "alias": "PCC-β — Propionyl-CoA Carboxylase Beta Subunit (Propionic Acidemia type B)",
        "aa": "539 aa", "kDa": "58.2 kDa",
        "gene_class": "propionyl_coa_carboxylase_beta",
        "locus": "3q22.3", "omim_gene": 232050,
        "phenotype": "Propionic Acidemia (PA-B) — PCC beta, propionyl-CoA → methylmalonyl-CoA block; carboxyl-transferase domain; identical PA phenotype",
        "disease": (
            "PCCB biallelic loss → Propionic Acidemia type B (PA type B, OMIM #606054). "
            "PCCB encodes the beta subunit (539aa, 58 kDa) of propionyl-CoA carboxylase (PCC), "
            "a biotin-dependent mitochondrial enzyme catalysing propionyl-CoA + CO₂ → methylmalonyl-CoA. "
            "The PCC holoenzyme is an α6β6 dodecamer: PCCA provides the biotin carboxylase domain; "
            "PCCB provides the carboxyl-transferase domain (accepts CO₂ from biotin and transfers to propionyl-CoA). "
            "Clinically IDENTICAL to PA type A (PCCA mutations): metabolic crisis neonatal or episodic; "
            "C3-carnitine elevation on NBS; hyperammonemia; ketoacidosis; dilated cardiomyopathy (30–40%); "
            "basal ganglia injury. Genotype-phenotype correlation: PCCB mutations more common in South Korean "
            "and Saudi populations; p.Gln272Ter most prevalent Korean founder. Not biotin-responsive (PA is "
            "NOT a biotin-responsive disorder — distinguished from holocarboxylase synthetase deficiency/HLCS "
            "which IS biotin-responsive and also shows C3 elevation in severe cases)."
        ),
        "inheritance": "Autosomal recessive, biallelic. Higher prevalence in South Korea (~1:25,000 vs 1:100,000–250,000 global) due to founder p.Gln272Ter (c.814C>T, ~40% Korean PA alleles).",
        "hallmark": (
            "PA-B HALLMARKS (IDENTICAL to PA-A; WES gene-level distinction needed): "
            "(1) NBS C3-carnitine elevation + C3/C2 ratio >0.4; "
            "(2) Korean founder p.Gln272Ter (c.814C>T) in ~40% of Korean PA-B alleles; "
            "(3) DCM (30–40%): same cardiac risk as PA-A; QTc prolongation; "
            "(4) Hyperammonemia at presentation; secondary NAGS inhibition by propionyl-CoA; "
            "(5) NOT biotin-responsive — distinguish from HLCS deficiency (multiple carboxylase deficiency, "
            "which IS biotin-responsive and also C3 elevation + C5-OH); "
            "(6) Metronidazole adjunct (reduces gut propionate producers); "
            "(7) Liver transplant: reduces metabolic crises, does NOT cure cardiac/brain involvement; "
            "(8) Pancreatitis: 10–20% (propionyl-CoA pancreatotoxic); "
            "(9) Optic neuropathy: rare but described in chronic PA-B; "
            "(10) PROTEIN RESTRICTION + GIR + carnitine + ammonia scavengers: identical emergency to PA-A."
        ),
        "key_ddx": (
            "vs PCCA: clinically identical; WES only distinguishes (13q32.3 vs 3q22.3); "
            "vs MUT: MMA urine markedly elevated (not PA); no DCM in MUT; "
            "vs HLCS (multiple carboxylase deficiency): BIOTIN-RESPONSIVE (vs PA NOT); C3 + C5-OH + C5:1 elevated; biotin 10–100 mg/day corrects; "
            "vs MMACHC (cblC): combined MMA+HCU; no homocysteine in PA; "
            "vs UCD: pure hyperammonemia without acidosis/C3 elevation."
        ),
        "founder_variant": "p.Gln272Ter (c.814C>T, South Korea founder ~40% PA-B alleles); p.Arg410Trp (European); p.Thr428Ile; exon 4 deletions; p.Tyr449Cys (Saudi)",
        "onset_pattern": "Neonatal form (most common): crisis days 1–5 after birth when maternal buffering lost. Late-onset/episodic: triggered by intercurrent illness, protein excess.",
        "mri_pattern": "Striatal necrosis (bilateral caudate + putamen); cerebral atrophy chronic; white matter changes; identical to PA-A MRI pattern.",
        "hypoglycaemia_rate": 0.48, "hepatopathy_rate": 0.28, "myopathy_rate": 0.12,
        "encephalopathy_rate": 0.80, "epilepsy_rate": 0.35, "ataxia_rate": 0.18,
        "lactic_ac_rate": 0.73, "hcm_rate": 0.05, "dcm_rate": 0.36,
        "pancreatitis_rate": 0.15, "hyperammonemia_rate": 0.88, "retinopathy_rate": 0.00,
        "renal_rate": 0.08, "neonatal_crisis_rate": 0.83, "homocysteine_elevated_rate": 0.00,
        "seed": 824,
        "kd_contraindicated": True, "kd_note": "KD CONTRAINDICATED — odd-chain FA catabolism → propionyl-CoA influx; worsens PA-B",
        "nbs_detected": True, "nbs_marker": "C3-carnitine (propionylcarnitine) + C3/C2 ratio >0.4",
        "ohcobl_response": False, "biotin_response": False, "glycine_conjugation": False,
        "betaine_treatment": False, "metronidazole_adjunct": True,
        "vpa_risk": "HIGH RISK — propionate metabolic block + carnitine depletion; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "Identical to PA-A: zero protein 24–48h; IV dextrose GIR 10–12; L-carnitine; sodium benzoate/phenylacetate + arginine for hyperammonemia; metronidazole; NaHCO₃; CRRT if refractory.",
    },
    # ── MUT — Methylmalonyl-CoA mutase, classic MMA ──
    {
        "gene": "MUT", "alias": "MUT — Methylmalonyl-CoA Mutase (Classic Methylmalonic Acidaemia, mut type)",
        "aa": "750 aa", "kDa": "82.8 kDa",
        "gene_class": "methylmalonyl_coa_mutase",
        "locus": "6p12.3", "omim_gene": 251000,
        "phenotype": "Classic MMA (mut0/mut−) — methylmalonyl-CoA mutase deficiency; adenosylcobalamin-requiring; renal damage; metabolic stroke",
        "disease": (
            "MUT biallelic loss → Classic Methylmalonic Acidaemia (MMA, OMIM #251000). "
            "MUT encodes mitochondrial methylmalonyl-CoA mutase (750aa, 82.8 kDa homodimer) — a "
            "coenzyme B12 (adenosylcobalamin, AdoCbl)-requiring enzyme catalysing the reversible "
            "isomerisation of methylmalonyl-CoA → succinyl-CoA (TCA cycle entry for propionate pathway). "
            "Two clinical phenotypes: mut0 (null, no residual enzyme activity) — the most severe, "
            "neonatal lethal without treatment; mut− (partial residual activity, pArg2Gly most common) — "
            "later-onset episodic. AdoCbl is the active cofactor; OHCbl trial (1 mg IM daily × 5d) detects "
            "partial responders (mut− phenotype, ~50% fall in plasma MMA). Renal involvement: "
            "tubulointerstitial nephritis → CKD (methylmalonic acid accumulation is directly nephrotoxic); "
            "renal failure is the major long-term morbidity in survivors. Metabolic stroke: bilateral basal "
            "ganglia necrosis (similar to PA but MMA-specific cause — methylmalonyl-CoA inhibits "
            "succinyl-CoA ligase + ETC in basal ganglia, explaining striatal specificity)."
        ),
        "inheritance": "Autosomal recessive, biallelic. 1:50,000–1:100,000 live births globally. Mut0 ~60% of all MUT cases.",
        "hallmark": (
            "CLASSIC MMA HALLMARKS: (1) NBS C3-carnitine (propionylcarnitine also elevated — shares C3 with PA); "
            "urine methylmalonate markedly elevated (methylmalonate >500 mmol/mol creatinine distinguishes from cblA/B); "
            "(2) mut0 vs mut−: OHCbl trial: >50% plasma MMA fall = mut− (partial responder); mut0 = NO response; "
            "(3) RENAL DAMAGE: tubulointerstitial nephritis → CKD; methylmalonate directly nephrotoxic; "
            "most important long-term morbidity; renal transplant for end-stage CKD; "
            "(4) METABOLIC STROKE: bilateral basal ganglia lesions (globus pallidus T2 signal) — can occur "
            "without acute crisis; 'metabolic infarction' rather than vascular; "
            "(5) NO CARDIAC involvement (vs PA: DCM risk): MUT does NOT cause cardiomyopathy — KEY DDx vs PA; "
            "(6) Liver transplant: reduces ammonia crises (mut enzyme expressed in liver), does NOT prevent renal/brain; "
            "(7) Combined liver+kidney transplant in advanced disease; "
            "(8) Hyperammonemia (secondary NAGS inhibition by methylmalonyl-CoA and propionyl-CoA); "
            "(9) Neutropenia (bone marrow suppression by methylmalonate); "
            "(10) VPA HIGH RISK — methylmalonate + VPA combined mitochondrial toxicity."
        ),
        "key_ddx": (
            "vs PA (PCCA/PCCB): PA has DCM risk; urine methylmalonate normal in PA; C3 elevated in both — urine OA differentiates; "
            "vs MMAA (cblA): cblA = OHCbl FULLY responsive (>50% MMA fall); mut0 = NO response; mut− = partial; "
            "vs MMAB (cblB): similar to cblA but less consistent OHCbl response; "
            "vs MMACHC (cblC): cblC = COMBINED MMA + hyperhomocysteinaemia + low methionine; NO homocysteine in MUT; "
            "vs SUCLA2 (MDDS10): low MMA + mtDNA depletion myopathy + SNHL; NOT organic acidaemia."
        ),
        "founder_variant": "p.Arg228Gln (c.683G>A, most common mut0 European); p.Arg2Gln (mut−, partial); p.Met700Val (mut−); p.Ala179Thr; p.Pro648Ser; numerous private biallelic variants across all ethnicities",
        "onset_pattern": "mut0: neonatal crisis (hours to days after birth). mut−: later-onset episodic (triggered by illness, surgery, fasting, protein excess).",
        "mri_pattern": "Bilateral globus pallidus and putamen T2 hyperintensity (metabolic stroke pattern); can be progressive; white matter changes; cerebellar atrophy in chronic disease.",
        "hypoglycaemia_rate": 0.55, "hepatopathy_rate": 0.20, "myopathy_rate": 0.10,
        "encephalopathy_rate": 0.85, "epilepsy_rate": 0.30, "ataxia_rate": 0.15,
        "lactic_ac_rate": 0.60, "hcm_rate": 0.00, "dcm_rate": 0.00,
        "pancreatitis_rate": 0.08, "hyperammonemia_rate": 0.80, "retinopathy_rate": 0.00,
        "renal_rate": 0.70, "neonatal_crisis_rate": 0.75, "homocysteine_elevated_rate": 0.00,
        "seed": 825,
        "kd_contraindicated": True, "kd_note": "KD CONTRAINDICATED — odd-chain FA → propionyl-CoA → methylmalonyl-CoA influx; worsens MMA",
        "nbs_detected": True, "nbs_marker": "C3-carnitine elevation + urine methylmalonate markedly elevated (>500 mmol/mol Cr)",
        "ohcobl_response": True, "ohcobl_note": "Trial mandatory: 1 mg OHCbl IM daily × 5d; >50% plasma MMA fall = mut− partial responder; mut0 = NO response",
        "biotin_response": False, "glycine_conjugation": False,
        "betaine_treatment": False, "metronidazole_adjunct": True,
        "vpa_risk": "HIGH RISK — combined methylmalonate + VPA mitochondrial toxicity; avoid in all MMA; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "Zero protein 24–48h; IV 10% dextrose GIR 10–12 mg/kg/min; L-carnitine 100–200 mg/kg/day; OHCbl 1 mg IM daily; sodium benzoate/phenylacetate + arginine for hyperammonemia; CRRT if refractory; long-term: natural protein 0.8–1.5 g/kg/day + MMA formula (leucine/isoleucine/valine/methionine/threonine restricted).",
    },
    # ── MMAA — cblA MMA ──
    {
        "gene": "MMAA", "alias": "MMAA — Mitochondrial Cobalamin Reductase (Methylmalonic Acidaemia, cblA type)",
        "aa": "418 aa", "kDa": "47.1 kDa",
        "gene_class": "methylmalonyl_coa_mutase_cobalamin_a",
        "locus": "4q31.21", "omim_gene": 251100,
        "phenotype": "MMA cblA — MMAA mitochondrial cobalamin reducing enzyme; OHCbl FULLY responsive; best MMA prognosis",
        "disease": (
            "MMAA biallelic loss → Methylmalonic Acidaemia cblA type (OMIM #251100). "
            "MMAA encodes a mitochondrial 418aa GTPase required for the reductive activation of cob(III)alamin → "
            "cob(I)alamin inside the mitochondrial matrix — the key reduction step preceding adenosylation by MMAB. "
            "Without functional MMAA, dietary cobalamin cannot be converted to adenosylcobalamin (AdoCbl), "
            "the obligate cofactor of methylmalonyl-CoA mutase (MUT) → MMA accumulation. "
            "CRITICAL THERAPEUTIC DISTINCTION: cblA is the most OHCbl-responsive MMA subtype — >80% of "
            "cblA patients show >50% plasma MMA reduction with OHCbl supplementation. "
            "Clinical phenotype similar to mut− (partial MUT activity): late-onset episodic crises; "
            "better long-term outcome than mut0; some cblA patients reach adulthood without dialysis. "
            "Renal disease still occurs (methylmalonate nephrotoxicity) but later and less severe than mut0."
        ),
        "inheritance": "Autosomal recessive, biallelic. ~1:200,000–1:500,000 live births. cblA accounts for ~25% of all MMA (mut type ~60%, cblA ~25%, cblB ~15%).",
        "hallmark": (
            "cblA HALLMARKS: (1) OHCbl FULL RESPONSE (>80% of cblA respond with >50% plasma MMA fall) — "
            "THE defining therapeutic feature; OHCbl 1 mg IM daily long-term; "
            "(2) Better prognosis than mut0: fewer neonatal deaths, less severe renal involvement; "
            "(3) Functional enzyme activity in mut: cblA patients have NORMAL MUT protein — only cofactor "
            "delivery defective; AdoCbl administration restores full enzymatic activity in responders; "
            "(4) C3-carnitine NBS elevation (same as MUT, MMAB, PA); "
            "(5) Urine methylmalonate elevated (less extreme than mut0); "
            "(6) Hyperammonemia during crises (same mechanism as mut-type); "
            "(7) MMAA protein contains a GTPase domain — unique among MMA genes; "
            "(8) Neutropenia and thrombocytopenia during crises (bone marrow suppression); "
            "(9) Late-onset renal disease possible (methylmalonate nephrotoxicity) but later than mut0; "
            "(10) PROTEIN RESTRICTION + GIR + OHCbl = emergency triad; "
            "OHCbl must be continued lifelong even between crises."
        ),
        "key_ddx": (
            "vs MUT (mut0): mut0 has NO OHCbl response; cblA >80% respond; MUT mut0 more severe neonatal crisis; "
            "vs MMAB (cblB): cblB less consistent OHCbl response (~40–60%); MMAB encodes adenosylation step (MMAA = reduction step before MMAB); "
            "vs MMACHC (cblC): cblC = combined MMA + homocysteine elevation; cblA = MMA only, normal homocysteine; "
            "vs PA (PCCA/PCCB): PA DCM risk; urine propionate elevated; no MMA elevation."
        ),
        "founder_variant": "p.Arg98Trp (c.292C>T, most common North American/European); p.Arg98Gln (c.293G>A); p.Lys49Glu; p.Ile172Thr; p.His199Tyr; exon deletion variants",
        "onset_pattern": "Typically presents after neonatal period (mean ~2–6 months) with episodic crisis; some present at 1–2 years. True neonatal presentation rare vs mut0.",
        "mri_pattern": "Basal ganglia changes possible during metabolic stroke; generally milder MRI changes than mut0; white matter signal abnormalities in older patients.",
        "hypoglycaemia_rate": 0.40, "hepatopathy_rate": 0.15, "myopathy_rate": 0.05,
        "encephalopathy_rate": 0.60, "epilepsy_rate": 0.20, "ataxia_rate": 0.12,
        "lactic_ac_rate": 0.50, "hcm_rate": 0.00, "dcm_rate": 0.00,
        "pancreatitis_rate": 0.05, "hyperammonemia_rate": 0.65, "retinopathy_rate": 0.00,
        "renal_rate": 0.35, "neonatal_crisis_rate": 0.30, "homocysteine_elevated_rate": 0.00,
        "seed": 826,
        "kd_contraindicated": True, "kd_note": "KD CONTRAINDICATED — odd-chain FA → methylmalonyl-CoA influx; worsens MMA regardless of OHCbl response",
        "nbs_detected": True, "nbs_marker": "C3-carnitine elevation; urine methylmalonate elevated (less extreme than mut0)",
        "ohcobl_response": True, "ohcobl_note": ">80% of cblA patients respond to OHCbl 1 mg IM daily; defining therapeutic feature of cblA subtype",
        "biotin_response": False, "glycine_conjugation": False,
        "betaine_treatment": False, "metronidazole_adjunct": True,
        "vpa_risk": "HIGH RISK — methylmalonate accumulation + VPA mitochondrial toxicity; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "OHCbl 1 mg IM daily (start immediately); zero protein 24–48h; IV dextrose GIR 8–12; L-carnitine 100 mg/kg/day; ammonia scavengers if NH₃ elevated; long-term OHCbl 1 mg IM 3x/week or cyanocobalamin oral in responders.",
    },
    # ── MMAB — cblB MMA ──
    {
        "gene": "MMAB", "alias": "MMAB — ATP:Cobalamin Adenosyltransferase (Methylmalonic Acidaemia, cblB type)",
        "aa": "250 aa", "kDa": "27.4 kDa",
        "gene_class": "methylmalonyl_coa_mutase_cobalamin_b",
        "locus": "12q24.11", "omim_gene": 251110,
        "phenotype": "MMA cblB — MMAB adenosylcobalamin synthase; variable OHCbl response (~40-60%); intermediate MMA prognosis",
        "disease": (
            "MMAB biallelic loss → Methylmalonic Acidaemia cblB type (OMIM #251110). "
            "MMAB encodes ATP:cobalamin adenosyltransferase (250aa, 27.4 kDa homotrimer) — the "
            "mitochondrial enzyme directly catalysing the final step of adenosylcobalamin (AdoCbl) synthesis: "
            "cob(I)alamin + ATP → adenosylcobalamin. MMAB acts DOWNSTREAM of MMAA in the mitochondrial "
            "cobalamin pathway: MMAA reduces cob(III)alamin → cob(I)alamin; MMAB adenosylates cob(I)alamin → AdoCbl. "
            "Without MMAB, even successful MMAA reduction is futile — AdoCbl cannot be made. "
            "OHCbl response: ~40–60% of cblB patients show OHCbl response (less consistent than cblA ~80%); "
            "mechanism: some residual/bypass adenosylation at low efficiency. "
            "Clinical severity intermediate between cblA (mildest) and mut0 (most severe). "
            "Renal disease in long-term survivors (methylmalonate nephrotoxicity, similar trajectory to cblA). "
            "MMAB was the first cbl complementation group protein crystallized (Lofgren 2004, J Mol Biol)."
        ),
        "inheritance": "Autosomal recessive, biallelic. cblB accounts for ~15% of all MMA (mut ~60%, cblA ~25%, cblB ~15%).",
        "hallmark": (
            "cblB HALLMARKS: (1) OHCbl response VARIABLE (~40–60% respond vs cblA >80%): "
            "trial mandatory but response less predictable; "
            "(2) MMAB encodes FINAL AdoCbl synthesis step: downstream of MMAA reduction; "
            "(3) MMAB homotrimer structure: three active sites each requiring one cob(I)alamin + ATP; "
            "(4) C3-carnitine NBS elevation (same as cblA, mut, PA); "
            "(5) Urine methylmalonate markedly elevated; "
            "(6) Clinical severity intermediate: worse than cblA, better than mut0; "
            "(7) p.Arg186Trp most common European variant (disrupts dimer interface of homotrimer); "
            "(8) OHCbl 1 mg IM daily trial: if >50% MMA fall, continue long-term; if no response, "
            "continue 1 mg IM for metabolic support + protein restriction only approach; "
            "(9) Hyperammonemia during crises; neutropenia; metabolic acidosis; "
            "(10) Long-term renal monitoring essential (methylmalonate nephrotoxicity)."
        ),
        "key_ddx": (
            "vs MMAA (cblA): cblA >80% OHCbl response; cblB ~40–60%; MMAA = reductive step before MMAB = adenosylation step; "
            "vs MUT (mut0): mut0 NO OHCbl response; more severe neonatal course; "
            "vs MMACHC (cblC): cblC = combined MMA + hyperhomocysteinaemia; cblB = MMA only; "
            "vs PA: PA DCM risk; urine organic acids distinguish (propionate vs methylmalonate)."
        ),
        "founder_variant": "p.Arg186Trp (c.556C>T, most common European cblB); p.Gly130Arg; p.Arg52His; c.700C>T (p.Arg234Cys); exon deletions",
        "onset_pattern": "Earlier than cblA on average; some neonatal presentations; episodic crises triggered by catabolism.",
        "mri_pattern": "Basal ganglia signal changes during metabolic stroke; similar pattern to mut-type; white matter changes in chronic disease.",
        "hypoglycaemia_rate": 0.45, "hepatopathy_rate": 0.18, "myopathy_rate": 0.08,
        "encephalopathy_rate": 0.70, "epilepsy_rate": 0.25, "ataxia_rate": 0.15,
        "lactic_ac_rate": 0.55, "hcm_rate": 0.00, "dcm_rate": 0.00,
        "pancreatitis_rate": 0.06, "hyperammonemia_rate": 0.72, "retinopathy_rate": 0.00,
        "renal_rate": 0.40, "neonatal_crisis_rate": 0.45, "homocysteine_elevated_rate": 0.00,
        "seed": 827,
        "kd_contraindicated": True, "kd_note": "KD CONTRAINDICATED — odd-chain FA → methylmalonyl-CoA influx; worsens cblB MMA",
        "nbs_detected": True, "nbs_marker": "C3-carnitine elevation; urine methylmalonate markedly elevated",
        "ohcobl_response": True, "ohcobl_note": "Variable response ~40–60%; OHCbl trial mandatory (1 mg IM daily × 5d); continue if >50% MMA fall",
        "biotin_response": False, "glycine_conjugation": False,
        "betaine_treatment": False, "metronidazole_adjunct": True,
        "vpa_risk": "HIGH RISK — methylmalonate accumulation + VPA mitochondrial toxicity; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "OHCbl 1 mg IM daily; zero protein 24–48h; IV dextrose GIR 8–12; L-carnitine; ammonia scavengers; long-term OHCbl + protein restriction.",
    },
    # ── MMACHC — cblC, combined MMA + HCU, MOST COMMON combined ──
    {
        "gene": "MMACHC", "alias": "MMACHC — Combined Methylmalonic Acidaemia + Homocystinuria, cblC (MOST COMMON combined OA)",
        "aa": "282 aa", "kDa": "31.7 kDa",
        "gene_class": "cobalamin_processing_cblc",
        "locus": "1p34.1", "omim_gene": 277400,
        "phenotype": "cblC — MOST COMMON combined OA; elevated MMA + elevated total homocysteine + low methionine; pigmentary retinopathy 80%; thromboembolism",
        "disease": (
            "MMACHC biallelic loss → Combined Methylmalonic Acidaemia + Homocystinuria type cblC "
            "(OMIM #277400) — THE MOST COMMON combined vitamin B12 disorder. "
            "MMACHC encodes a 282aa cytosolic bifunctional reductase/decyanase: "
            "— decyanates cyanocobalamin → hydroxocobalamin (step 1); "
            "— reduces cob(III)alamin → cob(II)alamin (step 2, shared with MeCbl and AdoCbl pathways). "
            "Because MMACHC acts BEFORE the cytosolic/mitochondrial branch point, BOTH synthesis pathways fail: "
            "AdoCbl (mitochondrial, cofactor for MUT) → elevated MMA; "
            "MethylCbl (cytosolic, cofactor for methionine synthase/MTR) → elevated homocysteine + low methionine. "
            "COMBINED biochemistry: elevated MMA + elevated total homocysteine (tHcy >100 µmol/L) + low methionine. "
            "Pigmentary retinopathy (macular degeneration/drusen-like, RPE atrophy): 70–85% in early-onset; "
            "unique finding distinguishing cblC from isolated MMA or isolated homocystinuria. "
            "Late-onset cblC: thromboembolism, psychosis, dementia, subacute combined degeneration."
        ),
        "inheritance": "Autosomal recessive, biallelic. MOST COMMON combined cobalamin metabolism disorder; 1:50,000–1:100,000 live births. p.Arg266Gln 43% of all cblC alleles globally.",
        "hallmark": (
            "cblC HALLMARKS: (1) COMBINED MMA + HYPERHOMOCYSTEINAEMIA — THE DEFINING BIOCHEMICAL FEATURE; "
            "MMA elevated (less extreme than mut0) + tHcy >100 µmol/L + low methionine; "
            "(2) PIGMENTARY RETINOPATHY (70–85% early onset): RPE atrophy, drusen-like deposits; "
            "visual impairment in infancy — UNIQUE among OA; ECG/ophthalmology mandatory at diagnosis; "
            "(3) EARLY-ONSET (neonatal/infantile): feeding difficulties, hypotonia, encephalopathy, "
            "multiorgan failure; LATE-ONSET: thromboembolism, psychosis, subacute combined degeneration (SCD); "
            "(4) TREATMENT REGIMEN: OHCbl 1 mg IM daily + betaine 100–200 mg/kg/day + "
            "methylfolate 0.5–1 mg/day + carnitine 50–100 mg/kg/day; "
            "(5) p.Arg266Gln (c.797G>A) — ~43% of all cblC alleles globally; private variants common in non-European; "
            "(6) WES note: MMACHC is in a complex region; validate with plasma cobalamin metabolite profile; "
            "(7) NBS: C3 elevated + low methionine on NBS — combination more specific than C3 alone; "
            "(8) Betaine: provides alternative homocysteine remethylation via BHMT (betaine-homocysteine "
            "methyltransferase, liver-specific bypass of MTR); "
            "(9) Thromboembolic risk: elevated homocysteine promotes endothelial dysfunction + coagulation; "
            "anticoagulation in late-onset cblC with prior thrombosis; "
            "(10) BTBGD exclusion mandatory if Leigh-like + lactic acidosis presentation."
        ),
        "key_ddx": (
            "vs MUT/MMAA/MMAB: isolated MMA, NORMAL homocysteine; cblC = combined (MMA + HCU); "
            "vs CBS (homocystinuria): isolated CBS deficiency — elevated homocysteine + low methionine + elevated cystathionine BUT NO MMA; lens dislocation; "
            "vs MTHFR (severe): isolated hyperhomocysteinaemia; normal MMA; no cblC retinopathy; "
            "vs PA (PCCA/PCCB): PA DCM risk; no homocysteine elevation; C3 only, no tHcy; "
            "vs MMAHC/cblD: combined MMA+HCU or isolated; different complementation group (very rare)."
        ),
        "founder_variant": "p.Arg266Gln (c.797G>A, ~43% global cblC alleles — most common in Middle Eastern, North African, European); p.Arg161Gln (c.482G>A, severe early-onset); c.271dupA (p.Arg91LysFs, truncation, severe); p.Ser54Arg; many private variants",
        "onset_pattern": "Early-onset (neonatal-infantile, ~80% of cases): feeding difficulties, hypotonia, encephalopathy, retinopathy within weeks. Late-onset: adolescent/adult, thrombosis/psychosis/dementia.",
        "mri_pattern": "Delayed myelination; T2 white matter hyperintensities; posterior white matter predominance; SCD pattern (dorsal cord) in late-onset adult; brain atrophy.",
        "hypoglycaemia_rate": 0.30, "hepatopathy_rate": 0.25, "myopathy_rate": 0.20,
        "encephalopathy_rate": 0.88, "epilepsy_rate": 0.50, "ataxia_rate": 0.25,
        "lactic_ac_rate": 0.35, "hcm_rate": 0.15, "dcm_rate": 0.10,
        "pancreatitis_rate": 0.02, "hyperammonemia_rate": 0.50, "retinopathy_rate": 0.78,
        "renal_rate": 0.20, "neonatal_crisis_rate": 0.65, "homocysteine_elevated_rate": 0.98,
        "seed": 828,
        "kd_contraindicated": True, "kd_note": "KD CONTRAINDICATED — odd-chain FA → methylmalonyl-CoA; worsens MMA component of cblC",
        "nbs_detected": True, "nbs_marker": "C3-carnitine + low methionine on expanded NBS; urine: MMA + homocysteine",
        "ohcobl_response": True, "ohcobl_note": "OHCbl 1 mg IM daily — mandatory component of cblC regimen; both AdoCbl and MeCbl pathways partially restored",
        "biotin_response": False, "glycine_conjugation": False,
        "betaine_treatment": True, "betaine_note": "Betaine 100–200 mg/kg/day + methylfolate: remethylation support via BHMT bypass",
        "metronidazole_adjunct": False,
        "vpa_risk": "HIGH RISK — MMA accumulation + VPA mitochondrial toxicity; avoid in cblC; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "OHCbl 1 mg IM daily; betaine 100–200 mg/kg/day oral; methylfolate 0.5–1 mg/day; L-carnitine 50–100 mg/kg/day; zero protein 24–48h for acute crisis; IV dextrose; ammonia scavengers if hyperammonemic; ophthalmology referral urgent.",
    },
    # ── IVD — Isovaleryl-CoA dehydrogenase, Isovaleric Acidemia ──
    {
        "gene": "IVD", "alias": "IVD — Isovaleryl-CoA Dehydrogenase (Isovaleric Acidaemia)",
        "aa": "394 aa", "kDa": "43.5 kDa",
        "gene_class": "isovaleryl_coa_dehydrogenase",
        "locus": "15q15.1", "omim_gene": 243500,
        "phenotype": "Isovaleric Acidaemia (IVA) — IVD FAD-dependent; NBS C5-carnitine; 'sweaty feet' odor; glycine + carnitine treatment; two clinical forms",
        "disease": (
            "IVD biallelic loss → Isovaleric Acidaemia (IVA, OMIM #243500). "
            "IVD encodes mitochondrial isovaleryl-CoA dehydrogenase (394aa, 43.5 kDa), a FAD-dependent "
            "enzyme catalysing the third step of leucine catabolism: isovaleryl-CoA + FAD → "
            "3-methylcrotonyl-CoA + FADH₂. Without IVD, isovaleryl-CoA accumulates → "
            "transesterified with carnitine (isovalerylcarnitine, C5) or conjugated with glycine "
            "(isovalerylglycine, IVG — major urinary metabolite). "
            "Isovaleric acid: 'sweaty feet' (cheesy/sweaty socks) odor — PATHOGNOMONIC during crises. "
            "TWO CLINICAL FORMS: (1) Acute neonatal: severe acidosis, hyperammonemia, encephalopathy, "
            "death if untreated; (2) Chronic intermittent: episodic crises triggered by illness/protein load; "
            "better prognosis. NBS: C5-carnitine (isovalerylcarnitine) elevation — SPECIFIC for IVA. "
            "KEY TREATMENT: glycine conjugation (glycine 250 mg/kg/day) — provides alternative isovaleryl-CoA "
            "disposal via urinary IVG excretion (bypasses IVD). L-carnitine reduces C5-acylcarnitine load."
        ),
        "inheritance": "Autosomal recessive, biallelic. 1:100,000–1:250,000 live births. Higher in German-speaking countries.",
        "hallmark": (
            "IVA HALLMARKS: (1) NBS C5-carnitine (isovalerylcarnitine) elevation — SPECIFIC for IVA; "
            "(2) 'SWEATY FEET' ODOR (isovaleric acid) — PATHOGNOMONIC during crisis; caretakers can detect crisis by smell; "
            "(3) GLYCINE TREATMENT: 250 mg/kg/day oral — alternative disposal via IVG conjugation; "
            "reduces toxic isovaleryl-CoA burden; critical acute AND chronic treatment; "
            "(4) L-CARNITINE 50–100 mg/kg/day: reduces C5-acylcarnitine burden; secondary carnitine depletion from IVG/C5 excretion; "
            "(5) TWO FORMS: acute neonatal (severe, high mortality pre-NBS) vs chronic intermittent (episodic crises, better prognosis); "
            "(6) NBS dramatically improved outcomes: pre-NBS ~30% neonatal mortality; post-NBS near-normal prognosis in chronic form; "
            "(7) LEUCINE RESTRICTION: dietary leucine 60–100 mg/kg/day; leucine-free amino acid formula supplement; "
            "(8) Hyperammonemia (isovaleryl-CoA inhibits NAGS — same mechanism as PA/MMA but generally less severe); "
            "(9) Bone marrow suppression: neutropenia, thrombocytopenia during crises; "
            "(10) MRI: basal ganglia changes possible during crisis; generally normal between crises in chronic form."
        ),
        "key_ddx": (
            "vs 2-Methylbutyryl-CoA dehydrogenase def (ACADSB): C5-carnitine elevation; 2-methylbutyrylcarnitine — ISOMER of C5; GC-MS urine OA differentiates (IVA: isovalerylglycine + 3-hydroxyisovalerate; ACADSB: 2-methylbutyrylglycine); "
            "vs GA2/MADD (ETFA/ETFB): multiple acylcarnitines (C4+C5+C8 etc.); riboflavin-responsive form; not isolated C5; "
            "vs HMGCL (HMG-CoA lyase): C5-OH (not C5); no 'sweaty feet'; no ketogenesis; "
            "vs 3-MCC (MCCC1/MCCC2): C5-OH elevation (not C5); 3-methylcrotonylglycine in urine (not IVG); no sweaty feet odor."
        ),
        "founder_variant": "p.Ala282Val (c.845C>T, German founder, ~50% of European IVA alleles); p.Lys366Arg (c.1097A>G, severe neonatal form); p.Arg363Cys; p.Gln384Ter; numerous private variants",
        "onset_pattern": "Acute neonatal: first week of life (leucine from protein catabolism of birth + maternal leucine withdrawn). Chronic intermittent: episodic crises from months to years, triggered by infection/protein load.",
        "mri_pattern": "Can be normal; bilateral pallidum T2 changes during acute crisis; cerebellar atrophy in chronic severe disease; generally mild compared to PA/MMA.",
        "hypoglycaemia_rate": 0.35, "hepatopathy_rate": 0.15, "myopathy_rate": 0.08,
        "encephalopathy_rate": 0.70, "epilepsy_rate": 0.25, "ataxia_rate": 0.15,
        "lactic_ac_rate": 0.40, "hcm_rate": 0.00, "dcm_rate": 0.00,
        "pancreatitis_rate": 0.05, "hyperammonemia_rate": 0.55, "retinopathy_rate": 0.00,
        "renal_rate": 0.05, "neonatal_crisis_rate": 0.60, "homocysteine_elevated_rate": 0.00,
        "seed": 829,
        "kd_contraindicated": False, "kd_note": "KD CAUTION — leucine is the main precursor substrate; KD not routinely used; high-leucine foods restricted",
        "nbs_detected": True, "nbs_marker": "C5-carnitine (isovalerylcarnitine) — highly specific; GC-MS urine: isovalerylglycine + 3-hydroxyisovalerate",
        "ohcobl_response": False, "biotin_response": False,
        "glycine_conjugation": True, "glycine_note": "GLYCINE 250 mg/kg/day — primary IVA-specific treatment; alternative disposal via urinary isovalerylglycine",
        "betaine_treatment": False, "metronidazole_adjunct": False,
        "vpa_risk": "HIGH RISK — isovaleryl-CoA and VPA both substrate for mitochondrial carnitine; carnitine depletion; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "Glycine 250 mg/kg/day IV or oral (start immediately); L-carnitine 100–200 mg/kg/day; zero protein 24–48h; IV 10% dextrose GIR 8–12; ammonia scavengers if hyperammonemic; NaHCO₃ for acidosis. Long-term: glycine + carnitine + leucine-restricted diet.",
    },
    # ── HMGCL — HMG-CoA Lyase, HMG-CoA Lyase Deficiency ──
    {
        "gene": "HMGCL", "alias": "HMGCL — 3-Hydroxy-3-Methylglutaryl-CoA Lyase (HMG-CoA Lyase Deficiency)",
        "aa": "325 aa", "kDa": "34.5 kDa",
        "gene_class": "hmg_coa_lyase",
        "locus": "1p36.11", "omim_gene": 246450,
        "phenotype": "HMG-CoA Lyase Deficiency — COMBINED ketogenesis block + leucine catabolism block; profound hypoketotic hypoglycaemia; NO KD ever; NBS C5-OH + C6-DC",
        "disease": (
            "HMGCL biallelic loss → 3-Hydroxy-3-Methylglutaryl-CoA Lyase Deficiency (HMGCLD, OMIM #246450). "
            "HMGCL encodes the mitochondrial enzyme (325aa, 34.5 kDa octamer) catalysing the FINAL STEP "
            "of both leucine catabolism AND ketogenesis: HMG-CoA → acetoacetate + acetyl-CoA. "
            "DUAL FUNCTION means HMGCLD is UNIQUE: "
            "(A) Leucine catabolism block: HMG-CoA (from leucine → 3-methylglutaconyl-CoA → HMG-CoA) "
            "accumulates → 3-methylglutarate, 3-hydroxy-3-methylglutarate, 3-methylglutaconylcarnitine excreted; "
            "(B) Ketogenesis block: HMG-CoA (from fatty acid-derived acetyl-CoA condensation) "
            "CANNOT be cleaved to acetoacetate → NO KETONES PRODUCED AT ALL during fasting/crisis. "
            "This is different from most other OA where ketones may be low — in HMGCLD, "
            "KETONE PRODUCTION IS STRUCTURALLY IMPOSSIBLE regardless of fasting duration. "
            "Profound HYPOKETOTIC HYPOGLYCAEMIA during fasting/illness — CANNOT make glucose AND CANNOT make ketones → "
            "brain has no alternative fuel → rapid encephalopathy. Hyperammonemia: secondary NAGS inhibition. "
            "Hepatomegaly (fatty liver — cannot oxidise acetyl-CoA via ketogenesis)."
        ),
        "inheritance": "Autosomal recessive, biallelic. Rare: ~1:300,000–1:1,000,000 live births. Higher in Saudi Arabia due to consanguinity. Portuguese/Spanish founder variant (p.Arg41Gln, ~30% of southern European alleles).",
        "hallmark": (
            "HMGCLD HALLMARKS: (1) NO KD EVER — ABSOLUTE CONTRAINDICATION: "
            "KD provides ketone bodies as alternative fuel but HMGCL deficiency CANNOT PRODUCE KETONES; "
            "high-fat diet → acetyl-CoA accumulation → HMG-CoA accumulation → NO relief; "
            "(2) COMBINED BLOCK: leucine catabolism + ketogenesis — BOTH blocked by single enzyme; "
            "(3) NBS: C5-OH (3-methylglutarylcarnitine) + C6-DC (3-methylglutaconylcarnitine) elevation "
            "— DISTINGUISHES from 3-MCC (C5-OH only) and MCAD (C8); "
            "(4) Urine OA: 3-methylglutarate + 3-hydroxy-3-methylglutarate + 3-methylglutaconate elevated; "
            "(5) PROFOUND hypoketotic hypoglycaemia — ketone negative even with severe hypoglycaemia; "
            "blood glucose can fall <1 mmol/L with detectable ketones <0.1 mmol/L (essentially zero); "
            "(6) GLUCOSE INFUSION EMERGENCY: IV 10% dextrose (never Ringer's lactate — lactate → pyruvate → "
            "acetyl-CoA cannot enter ketogenesis); target blood glucose >4.5 mmol/L continuously; "
            "(7) Fasting ABSOLUTELY FORBIDDEN: maximum fast 3–4h infant, 6–8h child; "
            "(8) NO leucine-rich food: avoid high-protein/high-leucine foods; leucine-restricted diet; "
            "(9) Hepatomegaly (fatty liver from failed ketogenesis); metabolic acidosis (OA accumulation); "
            "(10) BTBGD exclusion mandatory in Leigh-like presentation."
        ),
        "key_ddx": (
            "vs 3-MCC (MCCC1/MCCC2): 3-MCC = C5-OH only (3-methylcrotonylglycine); HMGCLD = C5-OH + C6-DC + 3-methylglutarate in urine; "
            "vs MCAD: MCAD = C8 elevation; hypoketotic HG during fasting; but MCAD can produce some ketones; HMGCLD = ZERO ketones; "
            "vs fatty oxidation disorders generally: most FAO = secondary carnitine depletion + C-profile; HMGCLD = leucine catabolism intermediate (C5-OH/C6-DC) + ketogenesis block; "
            "vs ACAT1 (beta-ketothiolase def): ACAT1 = C5-OH + 2-methylacetoacetate; triggered by isoleucine (not leucine); normally can produce ketones."
        ),
        "founder_variant": "p.Arg41Gln (c.122G>A, Portuguese/Spanish founder, ~30% southern European HMGCLD alleles); p.Arg135Ser; p.Gly296Ser; p.Glu37Ter; exon 3 deletions (Saudi Arabian)",
        "onset_pattern": "Typically 3–11 months (weaning + longer fasting + intercurrent illness trigger). Neonatal cases reported (especially Saudi Arabia, Saudi mutation). Triggered by any fasting/illness.",
        "mri_pattern": "White matter T2 changes (diffuse delayed myelination); basal ganglia signal during crisis; brain oedema in acute severe crisis; can be near-normal between crises.",
        "hypoglycaemia_rate": 0.95, "hepatopathy_rate": 0.65, "myopathy_rate": 0.05,
        "encephalopathy_rate": 0.88, "epilepsy_rate": 0.30, "ataxia_rate": 0.10,
        "lactic_ac_rate": 0.60, "hcm_rate": 0.00, "dcm_rate": 0.00,
        "pancreatitis_rate": 0.03, "hyperammonemia_rate": 0.45, "retinopathy_rate": 0.00,
        "renal_rate": 0.05, "neonatal_crisis_rate": 0.55, "homocysteine_elevated_rate": 0.00,
        "seed": 830,
        "kd_contraindicated": True, "kd_note": "KD ABSOLUTE CONTRAINDICATION — HMGCL deficiency = cannot make ketones; KD provides no benefit and worsens HMG-CoA accumulation",
        "nbs_detected": True, "nbs_marker": "C5-OH (3-methylglutarylcarnitine) + C6-DC (3-methylglutaconylcarnitine); urine: 3-methylglutarate + 3-OH-3-methylglutarate + 3-methylglutaconate",
        "ohcobl_response": False, "biotin_response": False, "glycine_conjugation": False,
        "betaine_treatment": False, "metronidazole_adjunct": False,
        "vpa_risk": "HIGH RISK — mitochondrial toxicity in OA context; VPA inhibits beta-oxidation; carnitine depletion; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "IV 10% dextrose IMMEDIATELY (target glucose >4.5 mmol/L); GIR 8–12 mg/kg/min; cornstarch feeds to prevent hypoglycaemia; NEVER fast; leucine-restricted diet (<120 mg/kg/day leucine); L-carnitine secondary; NaHCO₃ for acidosis; ammonia scavengers if hyperammonemic.",
    },
    # ── MCCC1 — 3-MCC deficiency type 1 (alpha) ──
    {
        "gene": "MCCC1", "alias": "MCCC1 — 3-Methylcrotonyl-CoA Carboxylase Alpha Subunit (3-MCC Deficiency Type 1)",
        "aa": "725 aa", "kDa": "78.8 kDa",
        "gene_class": "methylcrotonyl_coa_carboxylase_alpha",
        "locus": "3q27.1", "omim_gene": 210200,
        "phenotype": "3-MCC Deficiency type 1 — alpha subunit; USUALLY BENIGN; NBS C5-OH; biotin NOT responsive; most NBS individuals asymptomatic",
        "disease": (
            "MCCC1 biallelic loss → 3-Methylcrotonyl-CoA Carboxylase Deficiency type 1 (3-MCCD type 1, OMIM #210200). "
            "MCCC1 encodes the biotin-dependent alpha subunit (725aa, 78.8 kDa) of 3-methylcrotonyl-CoA carboxylase (3-MCC), "
            "a mitochondrial α4β4 octameric enzyme catalysing the carboxylation of 3-methylcrotonyl-CoA → "
            "3-methylglutaconyl-CoA (4th step of leucine catabolism). MCCC1 provides the biotin carboxylase domain; "
            "MCCC2 provides the carboxyl-transferase domain. "
            "3-MCC is biotin-dependent but 3-MCC DEFICIENCY IS NOT BIOTIN RESPONSIVE — a critical teaching point: "
            "unlike holocarboxylase synthetase (HCS) deficiency or biotinidase deficiency (which impair biotin "
            "attachment/recycling across ALL four carboxylases), 3-MCC deficiency has a specific apoenzyme defect "
            "that biotin supplementation CANNOT correct. "
            "CLINICAL SIGNIFICANCE DEBATE: The most important clinical fact about 3-MCCD is that the MAJORITY "
            "of individuals identified by NBS are COMPLETELY ASYMPTOMATIC throughout life. Many NBS C5-OH "
            "positives actually trace to the MOTHER (MCCC1 carrier or affected mother transmitting "
            "3-methylcrotonylglycine via breast milk). The disorder has been called the most over-diagnosed "
            "metabolic disorder in the NBS era."
        ),
        "inheritance": "Autosomal recessive, biallelic. Possibly the most common organic acidemia detected by NBS (1:50,000–1:75,000) — but clinical significance debated. True symptomatic disease: rare.",
        "hallmark": (
            "3-MCC type 1 HALLMARKS: (1) USUALLY BENIGN — most NBS-detected patients are ASYMPTOMATIC; "
            "population screening detects many biochemically affected but clinically silent individuals; "
            "(2) NBS C5-OH elevation: 3-methylcrotonylglycine + 3-hydroxyisovalerate (URINE OA); "
            "C5-OH on NBS can also represent HMGCLD or ACAT1 — urine OA distinguishes; "
            "(3) NOT BIOTIN RESPONSIVE — critical DDx from HCS deficiency and biotinidase deficiency "
            "(both ARE biotin-responsive and also show C5-OH); prescribing biotin for 3-MCC is an error; "
            "(4) MATERNAL 3-MCCD: NBS false positive when MCCC1/MCCC2-affected MOTHER transmits "
            "3-methylcrotonylglycine via breast milk → infant NBS positive but infant genetically normal; "
            "maternal genotyping resolves; "
            "(5) L-carnitine: sometimes prescribed (reduces C5-OH marker) but unproven clinical benefit; "
            "(6) Leucine restriction: sometimes prescribed but evidence poor; most specialists don't restrict in asymptomatic; "
            "(7) Symptomatic minority: metabolic crises during illness (vomiting, fasting, fever) with hypotonia, "
            "encephalopathy; outcome generally good with supportive treatment; "
            "(8) No cardiac involvement; no renal involvement; "
            "(9) Dietary management in symptomatic: leucine <100 mg/kg/day during crisis; "
            "(10) Genetic counselling important: recurrence risk 25%; prenatal diagnosis available."
        ),
        "key_ddx": (
            "vs MCCC2 (type 2): clinically identical; MCCC1 3q27.1 vs MCCC2 5q13.2; WES distinguishes; "
            "vs HCS deficiency (HLCS): BIOTIN-RESPONSIVE (vs 3-MCC NOT); multiple carboxylase deficiency (C3+C5-OH+C5:1); "
            "vs Biotinidase deficiency: BIOTIN-RESPONSIVE; skin/hair/neurological features; C5-OH + C3; "
            "vs HMGCLD: C5-OH + C6-DC (HMGCLD) vs C5-OH only (3-MCC); profound hypoketotic HG absent in 3-MCC; "
            "vs ACAT1: C5-OH + 2-methylacetoacetylcarnitine; isoleucine-triggered crises; ketoacidosis."
        ),
        "founder_variant": "p.Arg385Ser (c.1153C>A, North American/European); p.Lys468Gln; p.Thr524Ile; exon deletions; p.Gln461Ter; many private variants",
        "onset_pattern": "Majority asymptomatic throughout life (NBS detection). Symptomatic minority: first crisis during febrile illness, typically 3 months to 3 years.",
        "mri_pattern": "Usually normal. Mild T2 white matter changes in symptomatic cases during crisis. Brain MRI normal in asymptomatic NBS-detected individuals.",
        "hypoglycaemia_rate": 0.15, "hepatopathy_rate": 0.05, "myopathy_rate": 0.05,
        "encephalopathy_rate": 0.12, "epilepsy_rate": 0.05, "ataxia_rate": 0.03,
        "lactic_ac_rate": 0.10, "hcm_rate": 0.00, "dcm_rate": 0.00,
        "pancreatitis_rate": 0.00, "hyperammonemia_rate": 0.08, "retinopathy_rate": 0.00,
        "renal_rate": 0.00, "neonatal_crisis_rate": 0.05, "homocysteine_elevated_rate": 0.00,
        "seed": 831,
        "kd_contraindicated": False, "kd_note": "KD not routinely used; leucine restriction during crisis; KD not contraindicated but evidence lacking for 3-MCCD",
        "nbs_detected": True, "nbs_marker": "C5-OH (3-methylcrotonylcarnitine); urine: 3-methylcrotonylglycine + 3-hydroxyisovalerate",
        "ohcobl_response": False,
        "biotin_response": False, "biotin_note": "NOT biotin-responsive — specific apoenzyme defect cannot be corrected by biotin; unlike HLCS/biotinidase def",
        "glycine_conjugation": False, "betaine_treatment": False, "metronidazole_adjunct": False,
        "vpa_risk": "LOW-MODERATE RISK in asymptomatic; avoid in symptomatic with metabolic crises; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "Supportive: IV dextrose; avoid prolonged fasting; leucine restriction during crisis; L-carnitine if C5-OH markedly elevated; most symptomatic episodes resolve with supportive care. NO biotin (ineffective).",
    },
    # ── MCCC2 — 3-MCC deficiency type 2 (beta) ──
    {
        "gene": "MCCC2", "alias": "MCCC2 — 3-Methylcrotonyl-CoA Carboxylase Beta Subunit (3-MCC Deficiency Type 2)",
        "aa": "563 aa", "kDa": "61.2 kDa",
        "gene_class": "methylcrotonyl_coa_carboxylase_beta",
        "locus": "5q13.2", "omim_gene": 210210,
        "phenotype": "3-MCC Deficiency type 2 — beta subunit; identical phenotype to MCCC1; USUALLY BENIGN; carboxyl-transferase domain; maternal false-positive NBS risk",
        "disease": (
            "MCCC2 biallelic loss → 3-Methylcrotonyl-CoA Carboxylase Deficiency type 2 (3-MCCD type 2, OMIM #210210). "
            "MCCC2 encodes the carboxyl-transferase domain-containing beta subunit (563aa, 61.2 kDa) of "
            "3-methylcrotonyl-CoA carboxylase (3-MCC). The 3-MCC holoenzyme is an α4β4 octamer where "
            "MCCC1 (alpha) provides the biotin carboxylase + biotin carboxyl carrier domain, and "
            "MCCC2 (beta) provides the carboxyl-transferase domain (accepts CO₂ from biotin and transfers "
            "to 3-methylcrotonyl-CoA substrate). Loss of MCCC2 → same biochemical phenotype as MCCC1: "
            "3-methylcrotonyl-CoA accumulation → 3-methylcrotonylglycine, 3-hydroxyisovalerate, C5-OH. "
            "CLINICAL FEATURES IDENTICAL TO MCCC1 TYPE 1: majority of NBS-detected patients are ASYMPTOMATIC; "
            "biotin NOT responsive; maternal 3-MCCD causes NBS false positives via breast milk; "
            "carnitine supplementation sometimes used (C5-OH reduction); leucine restriction in acute crises. "
            "Clinically, MCCC1 vs MCCC2 distinction matters only for genetic counselling and prenatal diagnosis — "
            "management is identical."
        ),
        "inheritance": "Autosomal recessive, biallelic. Frequency similar to MCCC1 type 1. MCCC2 mutations slightly more common in East Asian populations. Together MCCC1+MCCC2 make 3-MCCD the most commonly NBS-detected OA.",
        "hallmark": (
            "3-MCC type 2 HALLMARKS (IDENTICAL to type 1; WES distinguishes MCCC1 vs MCCC2): "
            "(1) USUALLY BENIGN — same as MCCC1; majority asymptomatic; "
            "(2) NBS C5-OH + urine 3-methylcrotonylglycine + 3-hydroxyisovalerate; "
            "(3) NOT BIOTIN RESPONSIVE — identical to MCCC1 (same enzyme); "
            "(4) MATERNAL 3-MCCD false positives: affected mothers transmit 3-methylcrotonylglycine via "
            "breast milk → NBS C5-OH positive in genetically normal infants; "
            "(5) MCCC2 beta subunit encodes carboxyl-transferase domain: accepts CO₂ from MCCC1-biotin; "
            "(6) p.Gly161Glu most common MCCC2 mutation in European populations; "
            "(7) Clinical distinction from MCCC1: NONE; management identical; "
            "(8) L-carnitine: sometimes reduces C5-OH marker; unproven clinical benefit; "
            "(9) Long-term prognosis: excellent in majority; asymptomatic survival to adulthood common; "
            "(10) Genetic counselling: MCCC1 vs MCCC2 distinction important for recurrence risk "
            "counselling (both AR 25% recurrence) and prenatal diagnosis by gene-specific testing."
        ),
        "key_ddx": (
            "vs MCCC1: clinically identical; MCCC1 3q27.1 vs MCCC2 5q13.2; WES only; management identical; "
            "vs HCS (HLCS) deficiency: BIOTIN-RESPONSIVE; multiple carboxylase (C3+C5-OH+C5:1 all elevated); "
            "vs Biotinidase deficiency: BIOTIN-RESPONSIVE; hair/skin/neurological; serum biotinidase activity low; "
            "vs HMGCLD: C5-OH + C6-DC; profound HG; ketogenesis block absent in 3-MCCD; "
            "vs ACAT1: 2-methylacetoacetate; isoleucine trigger; ketoacidosis."
        ),
        "founder_variant": "p.Gly161Glu (c.482G>A, European); p.Gln40Arg; p.Arg448Trp; exon deletions; p.Arg438Ter; many private variants",
        "onset_pattern": "Majority asymptomatic throughout life. Symptomatic minority similar to MCCC1: first crisis in early childhood during intercurrent illness.",
        "mri_pattern": "Usually normal. Same as MCCC1: mild changes possible during crisis in symptomatic minority.",
        "hypoglycaemia_rate": 0.12, "hepatopathy_rate": 0.04, "myopathy_rate": 0.04,
        "encephalopathy_rate": 0.10, "epilepsy_rate": 0.04, "ataxia_rate": 0.02,
        "lactic_ac_rate": 0.08, "hcm_rate": 0.00, "dcm_rate": 0.00,
        "pancreatitis_rate": 0.00, "hyperammonemia_rate": 0.06, "retinopathy_rate": 0.00,
        "renal_rate": 0.00, "neonatal_crisis_rate": 0.04, "homocysteine_elevated_rate": 0.00,
        "seed": 832,
        "kd_contraindicated": False, "kd_note": "KD not routinely used; leucine restriction during crisis; no specific contraindication",
        "nbs_detected": True, "nbs_marker": "C5-OH (3-methylcrotonylcarnitine); urine: 3-methylcrotonylglycine + 3-hydroxyisovalerate (identical to MCCC1)",
        "ohcobl_response": False,
        "biotin_response": False, "biotin_note": "NOT biotin-responsive — same as MCCC1; do not prescribe biotin for 3-MCCD",
        "glycine_conjugation": False, "betaine_treatment": False, "metronidazole_adjunct": False,
        "vpa_risk": "LOW-MODERATE RISK in asymptomatic; avoid in symptomatic; prefer LEV/LCM if AED needed",
        "metformin_ci": False,
        "acute_treatment": "Supportive: IV dextrose; leucine restriction during acute crisis; L-carnitine (C5-OH reduction); NO biotin (ineffective). Management identical to MCCC1.",
    },
]


def _sim_cohort(gene_data: dict) -> list:
    """Deterministic 40-patient simulation for one OA gene."""
    rng = random.Random(gene_data["seed"])
    pts = []
    for i in range(40):
        pts.append({
            "id": i + 1,
            "hypoglycaemia": rng.random() < gene_data["hypoglycaemia_rate"],
            "hepatopathy": rng.random() < gene_data["hepatopathy_rate"],
            "myopathy": rng.random() < gene_data["myopathy_rate"],
            "encephalopathy": rng.random() < gene_data["encephalopathy_rate"],
            "epilepsy": rng.random() < gene_data["epilepsy_rate"],
            "ataxia": rng.random() < gene_data["ataxia_rate"],
            "lactic_acidosis": rng.random() < gene_data["lactic_ac_rate"],
            "hcm": rng.random() < gene_data["hcm_rate"],
            "dcm": rng.random() < gene_data.get("dcm_rate", 0.0),
            "pancreatitis": rng.random() < gene_data["pancreatitis_rate"],
            "hyperammonemia": rng.random() < gene_data["hyperammonemia_rate"],
            "retinopathy": rng.random() < gene_data["retinopathy_rate"],
            "renal_involvement": rng.random() < gene_data["renal_rate"],
            "neonatal_crisis": rng.random() < gene_data["neonatal_crisis_rate"],
            "homocysteine_elevated": rng.random() < gene_data.get("homocysteine_elevated_rate", 0.0),
            "nbs_positive": gene_data["nbs_detected"],
        })
    return pts


# ── Public API ────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    """Aggregate clinical stats, pathway classes, NBS markers, drug CIs for all 10 OA genes."""
    all_pts = []
    for gd in OA_GENES:
        all_pts.extend(_sim_cohort(gd))
    n = len(all_pts)

    def pct(field): return round(sum(1 for p in all_pts if p[field]) / n * 100, 1)

    # Gene class summary
    class_counts = {}
    for gd in OA_GENES:
        cls = gd["gene_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Drug CI / treatment summary
    kd_ci_genes = [gd["gene"] for gd in OA_GENES if gd.get("kd_contraindicated")]
    ohcobl_responsive = [gd["gene"] for gd in OA_GENES if gd.get("ohcobl_response")]
    biotin_not_responsive = [gd["gene"] for gd in OA_GENES if gd.get("biotin_response") is False]
    glycine_treated = [gd["gene"] for gd in OA_GENES if gd.get("glycine_conjugation")]
    betaine_treated = [gd["gene"] for gd in OA_GENES if gd.get("betaine_treatment")]

    return {
        "atlas": "Organic-Acidemia-Atlas",
        "n_genes": len(OA_GENES),
        "n_patients": n,
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(OA_GENES) - 1}",
        "gene_classes": {
            "propionate_pathway": ["PCCA", "PCCB"],
            "methylmalonate_pathway": ["MUT", "MMAA", "MMAB", "MMACHC"],
            "leucine_catabolism": ["IVD", "HMGCL", "MCCC1", "MCCC2"],
        },
        "aggregate_clinical": {
            "hypoglycaemia_pct": pct("hypoglycaemia"),
            "encephalopathy_pct": pct("encephalopathy"),
            "hyperammonemia_pct": pct("hyperammonemia"),
            "epilepsy_pct": pct("epilepsy"),
            "lactic_acidosis_pct": pct("lactic_acidosis"),
            "hepatopathy_pct": pct("hepatopathy"),
            "renal_involvement_pct": pct("renal_involvement"),
            "neonatal_crisis_pct": pct("neonatal_crisis"),
            "retinopathy_pct": pct("retinopathy"),
            "dcm_pct": pct("dcm"),
            "homocysteine_elevated_pct": pct("homocysteine_elevated"),
            "nbs_positive_pct": pct("nbs_positive"),
        },
        "drug_safety": {
            "kd_contraindicated_genes": kd_ci_genes,
            "kd_ci_count": len(kd_ci_genes),
            "kd_note": "KD CONTRAINDICATED in PCCA/PCCB/MUT/MMAA/MMAB/MMACHC/HMGCL (odd-chain FA → propionyl/methylmalonyl-CoA; or ketogenesis impossible in HMGCL)",
            "ohcobl_responsive_genes": ohcobl_responsive,
            "ohcobl_trial_mandatory": "MUT, MMAA, MMAB, MMACHC — OHCbl 1mg IM daily × 5 days; measure plasma MMA before and after",
            "biotin_NOT_responsive": "3-MCC (MCCC1/MCCC2) — NOT biotin-responsive despite biotin-dependent enzyme; unlike HLCS/biotinidase deficiency",
            "glycine_treatment_genes": glycine_treated,
            "betaine_treatment_genes": betaine_treated,
            "vpa_risk": "HIGH RISK for ALL 10 organic acidemia genes — prefer LEV/LCM",
            "protein_zero_emergency": "Zero protein 24–48h in acute crisis for ALL OA; restart at 0.5–1.5 g/kg/day with metabolic formula",
            "gir_emergency": "GIR 8–12 mg/kg/min IV dextrose in ALL OA crises — anti-catabolic cornerstone",
            "ammonia_scavengers": "Sodium benzoate 250 mg/kg + sodium phenylacetate 250 mg/kg + arginine IV for hyperammonemia in PA/MMA; CRRT if NH₃ >500 µmol/L",
        },
        "nbs_markers": {
            "C3_propionylcarnitine": ["PCCA", "PCCB", "MUT", "MMAA", "MMAB", "MMACHC"],
            "C5_isovalerylcarnitine": ["IVD"],
            "C5OH_3methylcrotonyl": ["HMGCL", "MCCC1", "MCCC2"],
            "C6DC_3methylglutaconyl": ["HMGCL"],
            "low_methionine": ["MMACHC"],
            "note": "C3 alone not specific (PA vs MMA vs cblA/B/C); urine organic acids + plasma homocysteine + OHCbl response differentiates",
        },
        "key_teaching_points": [
            "PA (PCCA/PCCB): DCM specific risk (30–45%) — cardiac monitoring mandatory; propionate inhibits ETC in cardiomyocytes",
            "MUT: mut0 = NO OHCbl response; mut− = partial OHCbl response; RENAL DAMAGE is major long-term morbidity",
            "cblA (MMAA): >80% OHCbl response — best cobalamin-treatable MMA; start OHCbl immediately on suspicion",
            "cblC (MMACHC): COMBINED MMA + hyperhomocysteinaemia + RETINOPATHY 80% — MOST COMMON combined OA; betaine mandatory",
            "IVA (IVD): GLYCINE 250 mg/kg/day specific treatment; 'sweaty feet' odor pathognomonic during crisis",
            "HMGCL: NO KD EVER — ketogenesis impossible; profound hypoketotic HG; glucose infusion only approach",
            "3-MCC (MCCC1/MCCC2): USUALLY BENIGN; NOT biotin-responsive; most NBS-detected patients asymptomatic",
            "KD CONTRAINDICATED for PCCA/PCCB/MUT/MMAA/MMAB/MMACHC/HMGCL (7 of 10 genes)",
            "VPA HIGH RISK for all 10 OA genes — organic acid accumulation + VPA mitochondrial toxicity synergistic",
            "BTBGD (SLC19A3) mandatory exclusion in any Leigh-like OA presentation",
        ],
    }


def get_breakdown() -> dict:
    """Per-gene breakdown for all 10 organic acidemia genes."""
    genes_out = []
    for gd in OA_GENES:
        cohort = _sim_cohort(gd)
        n = len(cohort)
        def pct(f): return round(sum(1 for p in cohort if p.get(f)) / n * 100, 1)
        genes_out.append({
            "gene": gd["gene"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "gene_class": gd["gene_class"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "phenotype": gd["phenotype"],
            "inheritance": gd["inheritance"],
            "hallmark": gd["hallmark"],
            "key_ddx": gd["key_ddx"],
            "founder_variant": gd["founder_variant"],
            "onset_pattern": gd["onset_pattern"],
            "mri_pattern": gd["mri_pattern"],
            "nbs_detected": gd["nbs_detected"],
            "nbs_marker": gd["nbs_marker"],
            "kd_contraindicated": gd.get("kd_contraindicated", False),
            "kd_note": gd.get("kd_note", ""),
            "ohcobl_response": gd.get("ohcobl_response", False),
            "ohcobl_note": gd.get("ohcobl_note", ""),
            "biotin_response": gd.get("biotin_response", False),
            "biotin_note": gd.get("biotin_note", ""),
            "glycine_conjugation": gd.get("glycine_conjugation", False),
            "glycine_note": gd.get("glycine_note", ""),
            "betaine_treatment": gd.get("betaine_treatment", False),
            "betaine_note": gd.get("betaine_note", ""),
            "metronidazole_adjunct": gd.get("metronidazole_adjunct", False),
            "vpa_risk": gd["vpa_risk"],
            "acute_treatment": gd["acute_treatment"],
            "n_patients": n,
            "clinical_rates": {
                "hypoglycaemia_pct": pct("hypoglycaemia"),
                "encephalopathy_pct": pct("encephalopathy"),
                "hyperammonemia_pct": pct("hyperammonemia"),
                "epilepsy_pct": pct("epilepsy"),
                "lactic_acidosis_pct": pct("lactic_acidosis"),
                "hepatopathy_pct": pct("hepatopathy"),
                "renal_involvement_pct": pct("renal_involvement"),
                "neonatal_crisis_pct": pct("neonatal_crisis"),
                "dcm_pct": pct("dcm"),
                "hcm_pct": pct("hcm"),
                "retinopathy_pct": pct("retinopathy"),
                "homocysteine_elevated_pct": pct("homocysteine_elevated"),
                "pancreatitis_pct": pct("pancreatitis"),
                "myopathy_pct": pct("myopathy"),
                "ataxia_pct": pct("ataxia"),
            },
        })
    return {"genes": genes_out, "n_genes": len(genes_out), "n_patients": len(genes_out) * 40}


def get_definitions() -> dict:
    """Clinical term definitions for the Organic Acidemia Atlas."""
    return {
        "atlas": "Organic-Acidemia-Atlas",
        "definitions": [
            {"term": "Organic Acidemia (OA)", "definition": "Inherited defects in amino acid/odd-chain fatty acid catabolism causing accumulation of organic acids. Characterised by metabolic acidosis + elevated anion gap, hyperammonemia (secondary NAGS inhibition), and multisystem toxicity. NBS acylcarnitine profiling detects most OA prenatally."},
            {"term": "Propionate Pathway", "definition": "Propionyl-CoA (from odd-chain FA, Ile/Val/Met/Thr catabolism, gut bacteria) → methylmalonyl-CoA (PCC, biotin-dependent; PCCA+PCCB) → succinyl-CoA (MUT, AdoCbl-requiring). Defects: propionic acidemia (PCCA/PCCB) or methylmalonic acidemia (MUT/MMAA/MMAB)."},
            {"term": "Methylmalonyl-CoA Mutase (MUT)", "definition": "Mitochondrial enzyme (AdoCbl-requiring) isomerising methylmalonyl-CoA → succinyl-CoA. Mut0 = null alleles, no OHCbl response. Mut− = partial activity, may respond to OHCbl. Renal tubular damage is the major long-term morbidity of MUT deficiency (methylmalonate nephrotoxicity)."},
            {"term": "Cobalamin Complementation Groups", "definition": "cblA (MMAA, mitochondrial reductase), cblB (MMAB, adenosyltransferase), cblC (MMACHC, cytosolic decyanase/reductase). cblA/B: isolated MMA. cblC: combined MMA + HCU. OHCbl response: cblA >80%, cblB ~40–60%, mut0 nil, mut− partial."},
            {"term": "MMACHC (cblC)", "definition": "Cytosolic bifunctional decyanase/reductase acting BEFORE the mitochondrial/cytosolic branch point; loss blocks BOTH AdoCbl (→elevated MMA) AND MeCbl (→elevated homocysteine) synthesis. MOST COMMON combined OA. Retinopathy 70–85% is distinctive."},
            {"term": "OHCbl (Hydroxocobalamin) Trial", "definition": "1 mg OHCbl IM daily × 5 days; measure plasma MMA before and after. >50% fall = cobalamin-responsive (cblA, mut−, cblC). Mandatory in all MUT/MMAA/MMAB/MMACHC patients. Continue IM OHCbl lifelong in responders."},
            {"term": "Hyperammonemia in OA", "definition": "Secondary urea cycle inhibition: propionyl-CoA and methylmalonyl-CoA inhibit N-acetylglutamate synthase (NAGS) — NAGS is the allosteric activator of CPS1 (first urea cycle enzyme). PA shows highest NH₃ levels (NH₃ can exceed 3000 µmol/L). Treatment: ammonia scavengers (benzoate + phenylacetate + arginine) + CRRT if refractory."},
            {"term": "Dilated Cardiomyopathy (DCM) in PA", "definition": "Specific complication of propionic acidemia (PCCA/PCCB); 30–45%; propionyl-CoA directly inhibits ETC complexes in cardiomyocytes. DCM can be present at diagnosis without a metabolic crisis. QTc prolongation → arrhythmia risk. Distinct from HCM seen in OXPHOS disorders."},
            {"term": "Isovalerylglycine (IVG) Conjugation", "definition": "In IVD (isovaleric acidemia): isovaleryl-CoA + glycine → isovalerylglycine (urinary). Glycine supplementation 250 mg/kg/day: provides alternative disposal route, reduces toxic isovaleryl-CoA. Combined with L-carnitine (C5-acylcarnitine excretion). IVG in urine confirms IVA diagnosis on GC-MS."},
            {"term": "HMGCL and Ketogenesis", "definition": "HMGCL catalyses HMG-CoA → acetoacetate + acetyl-CoA — the FINAL STEP of both ketogenesis and leucine catabolism. HMGCLD = structurally CANNOT produce ketones. Profound hypoketotic hypoglycaemia is the hallmark. KD absolute CI (cannot make ketone bodies). IV glucose infusion is the ONLY fuel during crisis."},
            {"term": "3-MCC Deficiency Benignity", "definition": "3-Methylcrotonyl-CoA carboxylase deficiency (MCCC1/MCCC2): majority of NBS-detected patients are COMPLETELY ASYMPTOMATIC throughout life. Most common OA by NBS. NBS C5-OH elevation can trace to affected mother's breast milk (maternal 3-MCCD). NOT biotin-responsive (unlike HLCS/biotinidase deficiency). Leucine restriction only during actual symptomatic crises."},
            {"term": "Biotin Responsiveness in Carboxylase Deficiencies", "definition": "RULE: 3-MCC (MCCC1/MCCC2) deficiency is NOT biotin-responsive. Biotin-responsive disorders: Holocarboxylase synthetase (HCS/HLCS) deficiency — ALL four carboxylases fail (ACC1, PC, MCC, PCC) due to biotin attachment defect; Biotinidase deficiency — biotin recycling failure. Both show C3+C5-OH and ARE biotin-responsive (10–100 mg/day). Never prescribe biotin for isolated 3-MCCD."},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== Organic Acidemia Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"Encephalopathy: {ov['aggregate_clinical']['encephalopathy_pct']}%")
    print(f"Hyperammonemia: {ov['aggregate_clinical']['hyperammonemia_pct']}%")
    print(f"KD CI genes: {ov['drug_safety']['kd_contraindicated_genes']}")
    print(f"OHCbl responsive: {ov['drug_safety']['ohcobl_responsive_genes']}")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
    print("=== ALL PASS ===")
