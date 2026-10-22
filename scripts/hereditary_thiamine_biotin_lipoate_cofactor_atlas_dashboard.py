#!/usr/bin/env python3
"""Hereditary-Thiamine-Biotin-Lipoate-Cofactor-Transport-Metabolism-Atlas -- Complete 8-Gene Reference
Water-Soluble Vitamin Cofactor Disorders: Thiamine Transport/Metabolism + Biotin Recycling + Lipoate Synthesis
(SLC19A3/BTBGD-Biotin-Thiamine-Responsive-Basal-Ganglia-Disease-Episodic-Encephalopathy-Bilateral-BG-PATHOGNOMONIC-Biotin+Thiamine-COMPLETE-REVERSAL,
 SLC19A2/TRMA-Rogers-Syndrome-Megaloblastic-Anemia+Diabetes+SNHL-Triad-PATHOGNOMONIC-Thiamine-100-300mg-Anemia+DM-Reversible-Hearing-IRREVERSIBLE,
 TPK1/THMD1-Thiamine-Pyrophosphokinase-Encephalopathy-Lactic-Acidosis-Thiamine-Responsive,
 SLC25A19/THMD4-Amish-Microcephaly-2-Oxoglutarate-Aciduria-Founder-pG177A-Thiamine-Partially-Responsive,
 BTD/Biotinidase-Deficiency-NBS-DETECTABLE-1in60000-Biotin-5mg-CURATIVE-Dermatitis-Alopecia-Optic-Atrophy,
 HLCS/Multiple-Carboxylase-Deficiency-Neonatal-Ketoacidosis-Lactic-Acidosis-Rash-Alopecia-Biotin-Responsive-NBS-Misses,
 LIAS/Lipoic-Acid-Synthase-Deficiency-Combined-Complex-I+II+PDH+alphaKGDH-Leigh-Like-Lipoate-Partially-Effective,
 LIPT1/Lipoyl-Transferase-Deficiency-Combined-PDH+alphaKGDH-Neonatal-Encephalopathy-Lactic-Acidosis-Lipoate):

SLC19A3 (Solute carrier family 19 member 3 / Thiamine transporter 2 / THTR2; 496aa; 2q37.3; AR-LOF;
     SLC19A3 encodes THTR2, the primary thiamine transporter in the brain (CNS-dominant expression);
     THTR2 mediates cellular uptake of thiamine (vitamin B1) in the central nervous system;
     also transports biotin into cells (dual thiamine/biotin transport capacity);
     SLC19A3 LOF → thiamine AND biotin deficiency selectively in brain (blood levels may be NORMAL);
     BIOTIN-THIAMINE-RESPONSIVE BASAL GANGLIA DISEASE (BTBGD) — OMIM 607483;
     PATHOGNOMONIC TRIAD ON MRI: bilateral symmetrical signal abnormalities in:
       (1) caudate head and putamen (basal ganglia — MOST DISTINCTIVE),
       (2) globus pallidus,
       (3) cortical ribboning (cortical restricted diffusion on DWI);
     CLINICAL: episodic sub-acute encephalopathy precipitated by fever/illness;
       episodes: confusion, seizures, dystonia, external ophthalmoplegia, dysphagia, dysarthria;
       if UNTREATED: progressive neurological deterioration, coma, death;
     PATHOGNOMONIC FEATURE: bilateral caudate+putamen necrosis on MRI during crisis = BTBGD until proven otherwise;
     MRI MIMICS: Wernicke encephalopathy (different distribution, no caudate), Leigh syndrome (less caudate), Wilson disease (older onset);
     TREATMENT: biotin 5-10 mg/day PLUS thiamine 100-300 mg/day — COMPLETE REVERSAL if started before irreversible damage;
       MRI lesions CAN reverse completely with treatment;
       initiate empirically when BTBGD suspected (do NOT wait for genetics);
     CRITICAL: biotin alone is INSUFFICIENT (both biotin AND thiamine required);
     FOUNDER: multiple Middle Eastern founders; specific variant p.E320Q (Moroccan), p.G549E (Saudi/Gulf);
     NBS: NOT detected by standard acylcarnitine or amino acid screen;
     seed SEED_BASE+0).

SLC19A2 (Solute carrier family 19 member 2 / Thiamine transporter 1 / THTR1; 497aa; 1q24.2; AR-LOF;
     SLC19A2 encodes THTR1, the high-affinity thiamine transporter expressed in intestinal epithelium and
       erythroid progenitors; mediates GI absorption of thiamine and thiamine uptake by bone marrow precursors;
     SLC19A2 LOF → impaired intestinal thiamine absorption → systemic thiamine deficiency affecting:
       (1) erythroid progenitors (megaloblastic change), (2) beta cells, (3) cochlea;
     THIAMINE-RESPONSIVE MEGALOBLASTIC ANEMIA (TRMA) / ROGERS SYNDROME — OMIM 249270;
     PATHOGNOMONIC TRIAD:
       (1) Megaloblastic anemia (B12/folate NORMAL — distinguishes from nutritional megaloblastic);
       (2) Non-autoimmune diabetes mellitus (insulin secretion failure — beta cell thiamine-dependent);
       (3) Sensorineural hearing loss (SNHL);
     CRITICAL DISTINCTION: all three components present in ~50% patients at diagnosis; partial presentations common;
     THIAMINE RESPONSE: anemia responds to thiamine 25-100 mg/day (reticulocyte crisis within 1-2 weeks);
       diabetes: partially responsive (may reduce insulin dose); SNHL: IRREVERSIBLE regardless of treatment;
     CRITICAL: early thiamine reduces diabetes and anemia severity; SNHL remains permanent — hearing aids;
     AUTOANTIBODIES NEGATIVE (distinguishes from autoimmune diabetes);
     RING SIDEROBLASTS may be present on bone marrow (iron-laden mitochondria);
     NBS: NOT detected by standard screens (not an acylcarnitine or amino acid disorder);
     FOUNDER: specific variants in various populations; no single dominant founder worldwide;
     seed SEED_BASE+1).

TPK1  (Thiamine pyrophosphokinase 1; 262aa; 7q35; AR-LOF;
     TPK1 encodes thiamine pyrophosphokinase, the enzyme that phosphorylates thiamine → thiamine pyrophosphate (TPP);
     TPP is the ACTIVE cofactor form of thiamine, required by:
       (1) Pyruvate dehydrogenase complex (PDH) — pyruvate → acetyl-CoA;
       (2) Alpha-ketoglutarate dehydrogenase (alphaKGDH/OGDH) — TCA cycle step;
       (3) Transketolase (TKT) — pentose phosphate pathway;
       (4) BCKDH (branched-chain keto-acid dehydrogenase) — BCAA catabolism;
     TPK1 LOF → TPP deficiency → ALL TPP-dependent enzymes fail simultaneously;
     THIAMINE METABOLISM DYSFUNCTION SYNDROME TYPE 1 (THMD1) — OMIM 614458;
     CLINICAL: severe early-onset encephalopathy (usually neonatal or infantile);
       lactic acidosis (PDH failure → pyruvate accumulates);
       brain atrophy on MRI; Leigh-like basal ganglia lesions; seizures; hypotonia;
     BIOCHEMISTRY: lactate/pyruvate elevated; 2-oxoglutarate elevated (alphaKGDH);
       alloisoleucine may be elevated (BCKDH); combined organic acid abnormalities;
     TREATMENT: high-dose thiamine 200-600 mg/day (bypasses TPK1 by providing excess substrate
       to overwhelm residual TPK1 activity or alternative phosphorylation);
       some patients show dramatic improvement; response rate ~40-60%;
     EMPIRICAL THIAMINE TRIAL mandatory in any infant/child with lactic acidosis + basal ganglia disease
       while awaiting genetic diagnosis;
     MOLECULAR DIAGNOSIS: whole exome/genome sequencing recommended (small gene, few pathogenic variants known);
     seed SEED_BASE+2).

SLC25A19 (Solute carrier family 25 member 19 / Mitochondrial deoxynucleotide carrier / DNC; 320aa; 17q25.3; AR-LOF;
     SLC25A19 encodes DNC (deoxynucleotide carrier) — mitochondrial inner membrane transporter that
       imports dTTP/dATP (for mtDNA maintenance) AND thiamine pyrophosphate (TPP);
     SLC25A19 LOF → DUAL pathology: impaired mitochondrial TPP import + impaired dNTP supply for mtDNA;
     TWO DISTINCT PHENOTYPES:
       (1) AMISH LETHAL MICROCEPHALY (OMIM 607196):
           severe congenital microcephaly (HC -7 to -10 SD at birth); 2-oxoglutarate aciduria;
           PATHOGNOMONIC: Amish Old Order founder mutation p.G177A (c.530G>A) — very high allele frequency
             in Amish (1 in 10 births carrier);
           early death (most <3 years); no effective treatment;
           distinct from primary microcephaly (no MPH1-ASPM pathway);
       (2) THIAMINE METABOLISM DYSFUNCTION SYNDROME TYPE 4 (THMD4) (OMIM 613710):
           milder mutations; Leigh syndrome-like; 2-oxoglutarate aciduria; partial thiamine responsive;
     BIOCHEMISTRY: 2-oxoglutarate MARKEDLY ELEVATED in urine (alphaKGDH uses TPP; TPP shortage in
       mitochondria → 2-OG accumulates) — KEY BIOMARKER;
     TREATMENT: thiamine 100-300 mg/day — partial improvement in THMD4; INEFFECTIVE for severe Amish microcephaly;
     PRENATAL DIAGNOSIS: mandatory in Amish couples (carrier frequency ~1/10 births);
     NBS: NOT detected by standard NBS (organic acid pattern may show 2-OG elevation);
     seed SEED_BASE+3).

BTD   (Biotinidase; 543aa; 3p25.1; AR-LOF;
     Biotinidase recycles biotin (vitamin H/B7) from biocytin (biotinyl-lysine, product of biotin-enzyme
       degradation); BTD cleaves the amide bond between biotin and lysine, releasing free biotin for reuse;
     BTD LOF → biotin cannot be recycled → secondary biotin deficiency (biotin intake normal but cannot be recycled);
     BIOTINIDASE DEFICIENCY (BTD deficiency, late-onset MCD) — OMIM 253260;
     PREVALENCE: 1:60,000-80,000 births — one of the MOST COMMON metabolic NBS disorders;
     NEWBORN SCREENING DETECTABLE: fluorometric BTD enzyme activity assay; detected before symptoms in high-income countries;
     PROFOUND DEFICIENCY (<10% BTD activity): without treatment → dermatitis + alopecia + neurological crisis;
       seizures (epileptic spasms, myoclonic); hearing loss; optic atrophy; progressive neurological deterioration;
     PARTIAL DEFICIENCY (10-30% BTD activity): milder; may be symptomatic only with biotin-poor diet;
     PATHOGNOMONIC CLINICAL FEATURES (if untreated):
       (1) Perioral dermatitis + erythematous rash PATHOGNOMONIC;
       (2) ALOPECIA (patchy or total — ALL THREE if biotin not given);
       (3) Optic atrophy (partial or complete — may be IRREVERSIBLE if treatment delayed);
       (4) Hearing loss (also potentially IRREVERSIBLE);
     BIOTIN 5-10 mg/day: COMPLETELY CURATIVE if started early (before optic/hearing damage);
       seizures resolve within days to weeks; dermatitis clears; hair regrows;
     CRITICAL: DO NOT MISTAKE for autoimmune alopecia or seborrhoeic eczema in infancy;
     BIOTIN INTERFERENCE: biotin supplementation interferes with immunoassay-based lab tests
       (TSH, troponin, hCG falsely LOW or HIGH depending on assay platform) — caution with doses >5 mg/day;
     FOUNDER: specific variants in different populations; p.D444H common in European populations;
     seed SEED_BASE+4).

HLCS  (Holocarboxylase synthetase; 726aa; 21q22.13; AR-LOF;
     HLCS (holocarboxylase synthetase) attaches biotin covalently to all four biotin-dependent carboxylases:
       (1) Propionyl-CoA carboxylase (PCC; PCCA/PCCB) — propionic acid catabolism;
       (2) 3-methylcrotonyl-CoA carboxylase (MCC) — leucine catabolism;
       (3) Pyruvate carboxylase (PC) — gluconeogenesis;
       (4) Acetyl-CoA carboxylase (ACC1/2) — fatty acid synthesis;
     HLCS LOF → NONE of the four carboxylases can be biotinylated → ALL FOUR fail simultaneously;
     MULTIPLE CARBOXYLASE DEFICIENCY NEONATAL-TYPE (MCD1, HLCS deficiency) — OMIM 253270;
     CLINICAL: neonatal presentation (vs BTD which is late-onset); severe ketoacidosis (propionic/lactic);
       organic acid storm: 3-hydroxyisovaleric acid + 3-methylcrotonylglycine + propionate + lactate ALL ELEVATED;
       rash (erythematous, seborrhoeic-like); alopecia; hypotonia; vomiting; lethargy; coma if untreated;
     NBS: FREQUENTLY MISSES (unlike BTD which is detected by enzyme assay; HLCS is diagnosed by organic acids);
       acylcarnitine: C5OH elevated (3-methylcrotonyl-carnitine); C3 propionylcarnitine elevated; non-specific;
       may mimic PA/MMA/3-MCC initially until comprehensive organic acids done;
     CRITICAL DDx from BTD: NEONATAL onset (BTD is usually later infantile/early childhood);
       BTD activity NORMAL in HLCS deficiency (biotinidase enzyme not deficient; the attachment enzyme HLCS is deficient);
     BIOTIN RESPONSIVE: high-dose biotin 10-20 mg/day → rapid reversal; excess free biotin overwhelms reduced
       HLCS affinity (HLCS has reduced but not absent activity in most patients);
     TREATMENT: biotin 10-20 mg/day; continued lifelong; dose may need increase during illness;
     FOUNDER: p.L237P (European founder variant — Japan clusters); p.R508W; p.V550M;
     seed SEED_BASE+5).

LIAS  (Lipoic acid synthase; 370aa; 4q32.3; AR-LOF;
     LIAS encodes lipoic acid synthase, a mitochondrial enzyme that catalyses the final step of de novo
       lipoic acid synthesis: inserts two sulfur atoms into octanoyl-ACP using S-adenosylmethionine (SAM)
       and an iron-sulfur cluster (radical SAM enzyme);
     Lipoic acid is an ESSENTIAL cofactor covalently bound to multiple mitochondrial enzyme complexes:
       H-protein (GCSH, GCS), E2 subunits of PDH, alphaKGDH (OGDH), BCKDH, KGDH;
     LIAS LOF → de novo lipoic acid synthesis fails → ALL lipoate-dependent complexes deficient simultaneously:
       PDH (pyruvate → acetyl-CoA), alphaKGDH (TCA cycle), BCKDH (BCAA catabolism), GCS (glycine cleavage);
     LIPOIC ACID SYNTHESIS DEFICIENCY — OMIM 614462;
     CLINICAL: neonatal or early infantile onset; severe lactic acidosis (PDH failure);
       combined OXPHOS complex I+II+III deficiency pattern on enzyme assays (lipoate required for CI assembly);
       Leigh syndrome-like brain MRI (bilateral basal ganglia + brainstem lesions);
       encephalopathy; seizures; hypotonia; cardiomyopathy in some;
     BIOCHEMISTRY: lactate VERY HIGH; pyruvate elevated; 2-oxoglutaric acid elevated; glycine elevated
       (GCS failure); combined organic acid profile mimics multiple simultaneous enzyme defects;
     TREATMENT: lipoic acid supplementation (racemic alpha-lipoic acid 60-100 mg/kg/day) partially effective
       (dietary lipoic acid can partially bypass de novo synthesis via salvage/scavenging pathways);
       thiamine high-dose (for PDH component); riboflavin (for CI component); no curative treatment;
     CLINICAL NOTE: all four lipoate-dependent complexes deficient = PATHOGNOMONIC for LIAS or LIPT1 deficiency;
       distinguish from LIPT1 which also causes combined deficiency but via attachment failure (not synthesis);
     NBS: NOT detected (no specific acylcarnitine marker);
     seed SEED_BASE+6).

LIPT1 (Lipoyl transferase 1; 376aa; 2q11.2; AR-LOF;
     LIPT1 encodes lipoyl transferase 1, which attaches pre-formed lipoate (octanoate-derived or salvaged)
       to the lipoyl domains of target proteins: E2 subunits of PDH and alphaKGDH, and H-protein of GCS;
     LIPT1 acts DOWNSTREAM of LIPT2 (which transfers lipoate from ACP to LIPT1 substrate domains);
     LIPT1 LOF → lipoate CANNOT be attached → PDH + alphaKGDH + GCS all fail (BCKDH less affected vs LIAS);
     LIPOIC ACID ATTACHMENT DEFECT TYPE 1 — OMIM 614963;
     CLINICAL: severe neonatal encephalopathy; lactic acidosis; hypoglycemia;
       Leigh-like MRI; seizures; multi-organ failure; early death if untreated;
     BIOCHEMISTRY: lactate elevated; 2-oxoglutarate elevated (alphaKGDH); glycine elevated (GCS failure);
       slightly DIFFERENT from LIAS: BCKDH typically LESS affected in LIPT1 (LIPT1 substrate specificity);
       alloisoleucine less elevated compared to LIAS (helps differentiate LIAS vs LIPT1);
     TREATMENT: lipoic acid supplementation (same as LIAS); partially effective; thiamine + riboflavin;
       no curative treatment;
     LIPT1 vs LIAS DDx:
       LIAS: all four complexes (PDH+alphaKGDH+BCKDH+GCS) severely deficient; BCKDH involved;
       LIPT1: PDH+alphaKGDH+GCS deficient; BCKDH less affected (LIPT1 substrate specificity);
       Molecular diagnosis required for definitive differentiation;
     CLINICAL NOTE: if combined lipoate-dependent complex deficiency confirmed → test both LIAS and LIPT1;
       whole exome sequencing preferred;
     seed SEED_BASE+7).

SEED_BASE = 4206
"""

import json
import random
from typing import Any


ATLAS_ID   = "hereditary-thiamine-biotin-lipoate-cofactor-atlas"
ATLAS_NAME = "Hereditary-Thiamine-Biotin-Lipoate-Cofactor-Transport-Metabolism-Atlas"
SEED_BASE  = 4206
N_PER_GENE = 40

GENES = ["SLC19A3", "SLC19A2", "TPK1", "SLC25A19", "BTD", "HLCS", "LIAS", "LIPT1"]

GENE_META = {
    "SLC19A3":  {"aa": 496,  "locus": "2q37.3",  "kda": 55,  "inh": "AR",  "disease": "Biotin-Thiamine-Responsive Basal Ganglia Disease (BTBGD)",           "omim_gene": 606152, "omim_disease": 607483},
    "SLC19A2":  {"aa": 497,  "locus": "1q24.2",  "kda": 55,  "inh": "AR",  "disease": "Thiamine-Responsive Megaloblastic Anemia (TRMA / Rogers syndrome)",  "omim_gene": 603941, "omim_disease": 249270},
    "TPK1":     {"aa": 262,  "locus": "7q35",    "kda": 30,  "inh": "AR",  "disease": "Thiamine Metabolism Dysfunction Syndrome Type 1 (THMD1)",             "omim_gene": 606712, "omim_disease": 614458},
    "SLC25A19": {"aa": 320,  "locus": "17q25.3", "kda": 35,  "inh": "AR",  "disease": "Amish Microcephaly / THMD4 (SLC25A19 deficiency)",                   "omim_gene": 606521, "omim_disease": 607196},
    "BTD":      {"aa": 543,  "locus": "3p25.1",  "kda": 60,  "inh": "AR",  "disease": "Biotinidase Deficiency (late-onset multiple carboxylase deficiency)",  "omim_gene": 609019, "omim_disease": 253260},
    "HLCS":     {"aa": 726,  "locus": "21q22.13","kda": 80,  "inh": "AR",  "disease": "Multiple Carboxylase Deficiency neonatal-type (MCD1, HLCS deficiency)","omim_gene": 609018, "omim_disease": 253270},
    "LIAS":     {"aa": 370,  "locus": "4q32.3",  "kda": 42,  "inh": "AR",  "disease": "Lipoic Acid Synthesis Deficiency (LIAS deficiency)",                  "omim_gene": 605205, "omim_disease": 614462},
    "LIPT1":    {"aa": 376,  "locus": "2q11.2",  "kda": 42,  "inh": "AR",  "disease": "Lipoic Acid Attachment Defect Type 1 (LIPT1 deficiency)",              "omim_gene": 610284, "omim_disease": 614963},
}

GENE_PATHOGNOMONIC = {
    "SLC19A3":  "Bilateral symmetrical caudate+putamen necrosis on MRI during febrile crisis PATHOGNOMONIC; COMPLETE REVERSAL with biotin 5-10 mg/day + thiamine 100-300 mg/day — DO NOT WAIT for genetics, treat empirically",
    "SLC19A2":  "Megaloblastic anemia + non-autoimmune diabetes + SNHL triad PATHOGNOMONIC; autoantibodies NEGATIVE; anemia+DM reverse with thiamine; SNHL IRREVERSIBLE — start hearing aids early; thiamine 25-100 mg/day",
    "TPK1":     "Neonatal/infantile encephalopathy + lactic acidosis + combined TPP-dependent enzyme failure (PDH+alphaKGDH); high-dose thiamine 200-600 mg/day empirically in ANY infant with lactic acidosis + basal ganglia disease",
    "SLC25A19": "2-oxoglutarate MARKEDLY ELEVATED in urine PATHOGNOMONIC for THMD4; Amish microcephaly: HC -7 to -10 SD at birth + p.G177A founder in Old Order Amish; prenatal testing mandatory in Amish couples",
    "BTD":      "Perioral dermatitis + alopecia + optic atrophy triad in infancy PATHOGNOMONIC; NBS-DETECTABLE (fluorometric BTD enzyme assay); biotin 5-10 mg/day COMPLETELY CURATIVE if before optic/hearing damage; 1:60,000 births",
    "HLCS":     "Neonatal ketoacidosis + lactic acidosis + 3-OHisovalerate + 3-MCG + propionate ALL ELEVATED; BTD activity NORMAL (distinguishes from BTD deficiency); biotin 10-20 mg/day → rapid reversal; NBS often misses",
    "LIAS":     "ALL four lipoate-dependent complexes (PDH+alphaKGDH+BCKDH+GCS) simultaneously deficient PATHOGNOMONIC; lactic acidosis + 2-oxoglutarate + glycine ALL elevated; Leigh-like MRI; lipoate supplementation partially effective",
    "LIPT1":    "PDH+alphaKGDH+GCS deficient (BCKDH LESS affected vs LIAS — key DDx); neonatal encephalopathy + lactic acidosis; lipoate supplementation; LIPT1 vs LIAS: molecular diagnosis + BCKDH activity required",
}

GENE_KEY_DRUG = {
    "SLC19A3":  "Biotin 5-10 mg/day + Thiamine 100-300 mg/day — BOTH required simultaneously, empirically BEFORE genetic confirmation; COMPLETE MRI reversal possible if early",
    "SLC19A2":  "Thiamine 25-100 mg/day lifelong; monitor anemia (reticulocyte response in 1-2 weeks); monitor HbA1c/insulin requirements; hearing aids for SNHL (irreversible); insulin for DM if needed",
    "TPK1":     "High-dose thiamine 200-600 mg/day; empirical trial in any infant with lactic acidosis+BG disease; thiamine pyrophosphate directly if available (bypasses TPK1); riboflavin + CoQ10 supportive",
    "SLC25A19": "Thiamine 100-300 mg/day — partial response in THMD4; insufficient for severe Amish microcephaly; prenatal genetic diagnosis + counselling for Amish families; supportive care",
    "BTD":      "Biotin 5-10 mg/day oral lifelong (100 mcg/day is INSUFFICIENT — pharmacological doses required); NBS-detected patients start at diagnosis; CRITICAL: do NOT use high-dose biotin (>5 mg) without clinical justification — lab test interference",
    "HLCS":     "High-dose biotin 10-20 mg/day oral lifelong; may need 20-40 mg/day during metabolic crises; carnitine supplementation (carnitine depleted by toxic acylcarnitines); protein restriction may help transiently",
    "LIAS":     "Alpha-lipoic acid (ALA) 60-100 mg/kg/day + thiamine 200 mg/day + riboflavin 200 mg/day; no curative therapy; KETOGENIC DIET CONTRAINDICATED (worsens PDH/lactic acidosis); supportive care",
    "LIPT1":    "Alpha-lipoic acid (ALA) 60-100 mg/kg/day + thiamine + riboflavin; same as LIAS (overlapping mechanism); KETOGENIC DIET CONTRAINDICATED; supportive; liver transplant NOT curative (not a hepatic disease exclusively)",
}

THIAMINE_GROUP = ["SLC19A3", "SLC19A2", "TPK1", "SLC25A19"]
BIOTIN_GROUP   = ["BTD", "HLCS"]
LIPOATE_GROUP  = ["LIAS", "LIPT1"]

NBS_DETECTS = {
    "SLC19A3":  False,
    "SLC19A2":  False,
    "TPK1":     False,
    "SLC25A19": False,
    "BTD":      True,   # fluorometric BTD enzyme assay — universally in NBS panels
    "HLCS":     False,  # acylcarnitine often non-specific; organic acid profile required
    "LIAS":     False,
    "LIPT1":    False,
}

VITAMIN_TREATMENT = {
    "SLC19A3":  "Biotin + Thiamine (dual vitamin therapy — brain-specific transport failure)",
    "SLC19A2":  "Thiamine (intestinal absorption failure — systemic deficiency)",
    "TPK1":     "Thiamine (phosphorylation failure — TPP synthesis blocked)",
    "SLC25A19": "Thiamine (mitochondrial import failure — TPP transport blocked)",
    "BTD":      "Biotin (recycling failure — free biotin depleted)",
    "HLCS":     "Biotin (attachment failure — excess free biotin overwhelms reduced HLCS affinity)",
    "LIAS":     "Lipoic acid + Thiamine (synthesis failure — de novo lipoate absent)",
    "LIPT1":    "Lipoic acid + Thiamine (attachment failure — pre-formed lipoate not transferred)",
}

REVERSIBLE_WITH_TREATMENT = {
    "SLC19A3":  "COMPLETELY reversible (MRI lesions can resolve) if treated before irreversible necrosis",
    "SLC19A2":  "Anemia + DM reversible; SNHL IRREVERSIBLE regardless of treatment timing",
    "TPK1":     "Partial reversal in ~40-60%; early treatment improves outcomes; some permanent deficit",
    "SLC25A19": "Partial (THMD4 mild form); Amish microcephaly — largely irreversible",
    "BTD":      "COMPLETELY reversible if NBS-detected and treated before optic/hearing damage; delayed diagnosis → permanent deficits",
    "HLCS":     "Reversible with biotin if started promptly; persistent ID if significant metabolic crises before diagnosis",
    "LIAS":     "Partially reversible; Leigh-like lesions may stabilise but not fully reverse; progressive otherwise",
    "LIPT1":    "Partially reversible with lipoate; similar to LIAS; severe neonatal onset often fatal without treatment",
}

FOUNDER_MUTATIONS = {
    "SLC19A3":  "p.E320Q (Moroccan/North African), p.G549E (Saudi/Gulf Arab), p.R175Q (pan-ethnic); multiple ethnic-specific founders",
    "SLC19A2":  "Various ethnic-specific variants; no single dominant worldwide founder; p.G172D (Western Europe); consanguineous families common",
    "TPK1":     "Rare; mostly private family variants; limited population data due to rarity",
    "SLC25A19": "p.G177A (Old Order Amish — carrier frequency ~1 in 10 Amish births, most common in this population); c.530G>A",
    "BTD":      "p.D444H (Europe — partial deficiency, ~50% of profound alleles in Europe); p.C33* + Q456H compound (Middle East); p.R538C; multiple population-specific founders",
    "HLCS":     "p.L237P (European/Japanese); p.R508W (pan-ethnic); p.V550M; multiple ethnic clusters",
    "LIAS":     "Private family variants; consanguineous families; no dominant founder established (rare, <100 cases worldwide)",
    "LIPT1":    "Private family variants; consanguineous families; no dominant founder (very rare)",
}

CASCADE_BIOMARKERS = {
    "SLC19A3":  "Normal plasma thiamine + biotin (blood transport intact); MRI bilateral BG lesions; CSF thiamine LOW (brain-specific deficiency)",
    "SLC19A2":  "Blood thiamine LOW; megaloblastic anemia (high MCV, normal B12/folate); HbA1c elevated; audiogram SNHL",
    "TPK1":     "Plasma thiamine normal or elevated (cannot be phosphorylated); elevated TPP:thiamine ratio reversed; plasma lactate elevated",
    "SLC25A19": "Plasma lactate elevated; urine 2-oxoglutarate MARKEDLY ELEVATED (alphaKGDH failure); fibroblast TPP uptake assay",
    "BTD":      "Serum BTD enzyme activity <10% (profound) or 10-30% (partial) by fluorometric assay; urine 3-OHisovalerate elevated (MCC failure)",
    "HLCS":     "BTD activity NORMAL; urine organic acids: 3-OHisovalerate + 3-MCG + propionate + lactate ALL elevated; C5OH + C3 acylcarnitine elevated",
    "LIAS":     "Combined lipoate-dependent enzyme assays (PDH+alphaKGDH+GCS+BCKDH ALL low); fibroblast/muscle; elevated lactate + 2-OG + glycine + alloisoleucine",
    "LIPT1":    "PDH+alphaKGDH+GCS low (BCKDH relatively spared); lactate + 2-OG + glycine elevated; alloisoleucine less elevated vs LIAS; fibroblast enzyme assay",
}


def _make_cohort(gene: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    meta = GENE_META[gene]
    n = N_PER_GENE

    male_n = rng.randint(n // 3, 2 * n // 3)
    female_n = n - male_n

    # Onset in months (0=neonatal, 1-6=early infantile, 6-24=late infantile, >24=childhood)
    if gene == "SLC19A3":
        median_onset = rng.uniform(12, 36)    # episodic, often early childhood at first crisis
    elif gene == "SLC19A2":
        median_onset = rng.uniform(6, 24)     # neonatal anemia or childhood presentation
    elif gene == "TPK1":
        median_onset = rng.uniform(0, 3)      # neonatal/early infantile
    elif gene == "SLC25A19":
        median_onset = rng.uniform(0, 0.5)    # congenital microcephaly (birth)
    elif gene == "BTD":
        median_onset = rng.uniform(3, 18)     # late-onset (2 months to 5 years if NBS missed)
    elif gene == "HLCS":
        median_onset = rng.uniform(0, 1)      # neonatal
    elif gene in ("LIAS", "LIPT1"):
        median_onset = rng.uniform(0, 2)      # neonatal/early infantile

    # Lactic acidosis rate
    lactic_acidosis_pct = (
        rng.uniform(0.80, 0.95) if gene in ("TPK1", "LIAS", "LIPT1", "HLCS") else
        rng.uniform(0.30, 0.55) if gene in ("SLC19A3", "SLC25A19") else
        rng.uniform(0.05, 0.15)  # SLC19A2, BTD
    )

    # MRI brain lesions
    mri_bg_lesions_pct = (
        rng.uniform(0.85, 0.97) if gene == "SLC19A3" else   # bilateral BG PATHOGNOMONIC
        rng.uniform(0.65, 0.85) if gene in ("TPK1", "LIAS", "LIPT1") else   # Leigh-like
        rng.uniform(0.70, 0.90) if gene == "SLC25A19" else  # severe microcephaly + BG
        rng.uniform(0.10, 0.25)  # BTD/HLCS/SLC19A2: less specific MRI
    )

    # NBS detection
    nbs_detected_pct = rng.uniform(0.88, 0.97) if NBS_DETECTS[gene] else rng.uniform(0.01, 0.05)

    # Treatment response (vitamin supplementation)
    tx_response_pct = (
        rng.uniform(0.82, 0.97) if gene in ("SLC19A3", "BTD", "HLCS") else  # near-complete if early
        rng.uniform(0.55, 0.80) if gene in ("SLC19A2", "TPK1") else           # partial
        rng.uniform(0.25, 0.55)  # SLC25A19, LIAS, LIPT1 — partial/poor
    )

    # Seizures rate
    seizures_pct = (
        rng.uniform(0.75, 0.92) if gene in ("SLC19A3", "TPK1", "LIAS", "LIPT1") else
        rng.uniform(0.30, 0.55) if gene in ("HLCS", "BTD") else
        rng.uniform(0.05, 0.20)  # SLC19A2, SLC25A19
    )

    # Hearing loss rate
    hearing_loss_pct = (
        rng.uniform(0.55, 0.80) if gene == "SLC19A2" else  # SNHL IRREVERSIBLE in TRMA
        rng.uniform(0.30, 0.55) if gene == "BTD" else       # hearing loss if treatment delayed
        rng.uniform(0.03, 0.12)
    )

    # Dermatitis/rash (biotin-deficiency signs)
    dermatitis_pct = (
        rng.uniform(0.65, 0.85) if gene in ("BTD", "HLCS") else
        rng.uniform(0.02, 0.08)
    )

    # Alopecia
    alopecia_pct = (
        rng.uniform(0.55, 0.80) if gene in ("BTD", "HLCS") else
        rng.uniform(0.01, 0.06)
    )

    # Megaloblastic anemia (TRMA specific)
    megaloblastic_pct = (
        rng.uniform(0.80, 0.95) if gene == "SLC19A2" else
        rng.uniform(0.01, 0.05)
    )

    # Diabetes (TRMA specific)
    diabetes_pct = (
        rng.uniform(0.60, 0.85) if gene == "SLC19A2" else
        rng.uniform(0.01, 0.05)
    )

    return {
        "gene":                     gene,
        "n":                        n,
        "seed":                     seed,
        "male_n":                   male_n,
        "female_n":                 female_n,
        "disease":                  meta["disease"],
        "locus":                    meta["locus"],
        "aa":                       meta["aa"],
        "kda":                      meta["kda"],
        "inh":                      meta["inh"],
        "cofactor_group":           (
            "thiamine" if gene in THIAMINE_GROUP else
            "biotin" if gene in BIOTIN_GROUP else
            "lipoate"
        ),
        "vitamin_treatment":        VITAMIN_TREATMENT[gene],
        "nbs_detected_pct":         round(nbs_detected_pct, 2),
        "treatment_response_pct":   round(tx_response_pct, 2),
        "reversible_with_treatment":REVERSIBLE_WITH_TREATMENT[gene],
        "median_onset_months":      round(median_onset, 2),
        "lactic_acidosis_pct":      round(lactic_acidosis_pct, 2),
        "mri_bg_lesions_pct":       round(mri_bg_lesions_pct, 2),
        "seizures_pct":             round(seizures_pct, 2),
        "hearing_loss_pct":         round(hearing_loss_pct, 2),
        "dermatitis_pct":           round(dermatitis_pct, 2),
        "alopecia_pct":             round(alopecia_pct, 2),
        "megaloblastic_pct":        round(megaloblastic_pct, 2),
        "diabetes_pct":             round(diabetes_pct, 2),
        "pathognomonic":            GENE_PATHOGNOMONIC[gene],
        "key_drug":                 GENE_KEY_DRUG[gene],
        "founder_mutation":         FOUNDER_MUTATIONS[gene],
        "cascade_biomarker":        CASCADE_BIOMARKERS[gene],
    }


def generate_overview() -> dict[str, Any]:
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total = sum(c["n"] for c in cohorts)

    return {
        "atlas_id":       ATLAS_ID,
        "atlas_name":     ATLAS_NAME,
        "total_patients": total,
        "genes_count":    len(GENES),
        "seeds":          f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "gene_list":      GENES,
        "gene_cohorts":   cohorts,
        "atlas_summary": {
            "pathway_covered":         "Thiamine Transport/Metabolism (SLC19A3/SLC19A2/TPK1/SLC25A19) + Biotin Recycling/Attachment (BTD/HLCS) + Lipoate Synthesis/Attachment (LIAS/LIPT1)",
            "thiamine_group_genes":    THIAMINE_GROUP,
            "biotin_group_genes":      BIOTIN_GROUP,
            "lipoate_group_genes":     LIPOATE_GROUP,
            "nbs_detected_genes":      [g for g in GENES if NBS_DETECTS[g]],
            "completely_reversible":   ["SLC19A3", "BTD"],
            "partially_irreversible":  ["SLC19A2"],
            "critical_empirical_tx":   "SLC19A3 (BTBGD): start biotin+thiamine EMPIRICALLY before genetics — MRI bilateral BG = treat NOW",
            "all_ar_inheritance":      "All 8 genes are autosomal recessive (AR)",
            "critical_ddx_rule":       "PRIMARY SPLIT: Thiamine-type (brain/systemic) vs Biotin-type (carboxylase symptoms: dermatitis/alopecia) vs Lipoate-type (combined multi-complex failure); within Thiamine: BTBGD (SLC19A3) has bilateral BG on MRI; TRMA (SLC19A2) has megaloblastic anemia triad; TPK1 has neonatal lactic acidosis; SLC25A19 has 2-OG aciduria + Amish microcephaly",
        },
    }


def generate_breakdown() -> dict[str, Any]:
    pathway_groups = {
        "thiamine_transport_metabolism_SLC19A3_SLC19A2_TPK1_SLC25A19": {
            "genes":         THIAMINE_GROUP,
            "mechanism":     "SLC19A3 (THTR2): brain-specific thiamine + biotin transporter — CNS-selective deficiency with normal blood levels; SLC19A2 (THTR1): intestinal + erythroid thiamine transporter — systemic thiamine deficiency affecting bone marrow and beta cells; TPK1: cytoplasmic enzyme that phosphorylates thiamine → TPP (active cofactor) — TPP deficiency affects all TPP-dependent enzymes; SLC25A19: mitochondrial inner membrane transporter for TPP AND dNTPs — selective mitochondrial TPP deficiency.",
            "shared_feature": "All four genes cause thiamine-related dysfunction; lactic acidosis from PDH failure is common; Leigh-like or BG lesions on MRI; high-dose thiamine supplementation is the cornerstone of treatment.",
            "key_ddx":        "SLC19A3 (BTBGD): bilateral caudate+putamen on MRI; fever-triggered episodic encephalopathy; plasma thiamine NORMAL (brain-specific). SLC19A2 (TRMA): megaloblastic anemia + DM + SNHL triad; plasma thiamine LOW. TPK1: neonatal lactic acidosis + combined TPP-dependent enzyme failure; early severe onset. SLC25A19: urine 2-oxoglutarate MARKEDLY elevated; Amish microcephaly or THMD4.",
        },
        "biotin_recycling_attachment_BTD_HLCS": {
            "genes":         BIOTIN_GROUP,
            "mechanism":     "BTD (biotinidase): recycles biotin from biocytin after carboxylase protein turnover — secondary biotin deficiency from failed recycling; presents later (late-onset MCD). HLCS (holocarboxylase synthetase): attaches biotin to all four carboxylases (PCC, MCC, PC, ACC) — primary biotin attachment failure; neonatal onset, ALL four carboxylases fail simultaneously.",
            "shared_feature": "Both: biotin-responsive (pharmacological biotin reverses carboxylase deficiency); dermatitis + alopecia (biotin-deficiency signs); seizures; multiple carboxylase biochemistry (3-OHisovalerate + 3-MCG + propionate elevated in urine).",
            "key_ddx":        "BTD: NBS-DETECTED (fluorometric BTD enzyme assay); BTD activity <10% (profound) or 10-30% (partial); late-onset (infantile-early childhood); BTD enzyme can be measured in dried blood spot. HLCS: NBS MISSES; BTD activity NORMAL in HLCS deficiency (critical DDx); NEONATAL onset; severe ketoacidosis + lactic acidosis; requires urine organic acids + molecular diagnosis.",
        },
        "lipoate_synthesis_attachment_LIAS_LIPT1": {
            "genes":         LIPOATE_GROUP,
            "mechanism":     "LIAS: synthesises lipoic acid de novo in mitochondria (radical SAM enzyme, inserts S atoms into octanoyl-ACP). LIPT1: transfers pre-formed lipoate to target protein lipoyl domains (PDH E2, alphaKGDH E2, GCS H-protein). Both cause combined deficiency of ALL lipoate-dependent complexes when severely deficient.",
            "shared_feature": "Both: neonatal/early infantile onset; severe lactic acidosis; combined multi-enzyme deficiency (PDH+alphaKGDH+GCS); Leigh-like MRI; glycine elevated (GCS failure, mimics NKH); partial response to alpha-lipoic acid + thiamine.",
            "key_ddx":        "LIAS: ALL four lipoate-dependent complexes severely deficient INCLUDING BCKDH; alloisoleucine elevated (BCKDH). LIPT1: PDH+alphaKGDH+GCS deficient but BCKDH relatively SPARED; alloisoleucine less elevated. Both require fibroblast/muscle enzyme assays + molecular diagnosis to differentiate.",
        },
    }

    diagnostic_cascade = [
        "STEP 1 — PLASMA LACTATE + PYRUVATE: elevated L:P ratio >20 in TPK1/LIAS/LIPT1/HLCS (PDH failure from cofactor deficiency); normal or mildly elevated in BTD/SLC19A2/SLC19A3 at baseline",
        "STEP 2 — URINE ORGANIC ACIDS (critical for BTD/HLCS and lipoate groups): 3-hydroxyisovalerate + 3-methylcrotonylglycine ELEVATED → biotin-dependent MCC failure (BTD or HLCS); 2-oxoglutarate MARKEDLY ELEVATED → alphaKGDH failure (SLC25A19/LIAS/LIPT1/TPK1); propionate elevated → biotin-dependent PCC failure (BTD/HLCS)",
        "STEP 3 — SERUM BTD ENZYME ACTIVITY (DBS or serum): <10% profound deficiency, 10-30% partial → BTD deficiency; NORMAL BTD activity → BTD excluded; HLCS deficiency has NORMAL BTD activity — critical DDx point",
        "STEP 4 — BLOOD THIAMINE LEVELS (plasma and RBC thiamine): LOW in SLC19A2 (systemic transport failure); NORMAL or HIGH in SLC19A3 (blood transport intact — brain-specific); NORMAL in TPK1 (phosphorylation failure, not transport)",
        "STEP 5 — BRAIN MRI (emergency): bilateral symmetrical caudate+putamen ± DWI restriction → BTBGD (SLC19A3) — treat empirically NOW with biotin+thiamine; Leigh-like bilateral BG + brainstem → TPK1/LIAS/LIPT1; severe microcephaly → SLC25A19; no BG changes → consider BTD/HLCS/SLC19A2",
        "STEP 6 — COMPLETE BLOOD COUNT + BLOOD FILM: megaloblastic anemia (hypersegmented neutrophils + macrocytes, HIGH MCV) with NORMAL B12/folate → SLC19A2 TRMA; ring sideroblasts on marrow aspirate in TRMA; BTD/HLCS: normochromic anemia possible from metabolic crises",
        "STEP 7 — BLOOD GLUCOSE + HbA1c: non-autoimmune DM (autoantibodies NEGATIVE, insulin-requiring) with megaloblastic anemia → SLC19A2 TRMA; BTD/HLCS: hypoglycemia possible from carboxylase failure; LIAS/LIPT1: hypoglycemia from PDH/alphaKGDH failure",
        "STEP 8 — FIBROBLAST/MUSCLE ENZYME ASSAYS (if lipoate group suspected): PDH activity, alphaKGDH activity, BCKDH activity, GCS activity; ALL FOUR severely low → LIAS; PDH+alphaKGDH+GCS low but BCKDH relatively spared → LIPT1",
        "STEP 9 — EMPIRICAL VITAMIN TRIAL (do NOT wait for genetics): suspected BTBGD → start biotin 10 mg + thiamine 200 mg immediately; suspected BTD/HLCS → start biotin 20 mg immediately; suspected TPK1/THMD → start thiamine 200 mg immediately; response confirms diagnosis in SLC19A3/BTD/HLCS",
        "STEP 10 — MOLECULAR GENE PANEL / WES: BTBGD gene panel (SLC19A3); TRMA gene panel (SLC19A2); lipoate panel (LIAS+LIPT1+LIPT2); biotin panel (BTD+HLCS); TPK1 Sanger sequencing; SLC25A19 Sanger (especially Amish populations)",
    ]

    treatment_pathways = [
        {
            "gene":             "SLC19A3",
            "disease":          "BTBGD",
            "first_line":       "Biotin 5-10 mg/day + Thiamine 100-300 mg/day SIMULTANEOUSLY — both vitamins required, not one alone; treat empirically on MRI diagnosis, do not wait for genetic confirmation",
            "precision_therapy":"Higher doses (biotin 20 mg + thiamine 600 mg) during metabolic crises; biotin+thiamine brain penetrance maintained even with dietary variation; some patients reduce dose out of crisis",
            "avoid":            "Using biotin alone (insufficient — thiamine also required); vigabatrin/valproate (may worsen seizure control); prolonged fever without early escalation of vitamin doses",
            "monitoring":       "MRI every 6-12 months (monitor lesion regression); CBC (megaloblastic signs from thiamine deficiency); seizure diary; developmental assessments; plasma thiamine (should be normal on therapy)",
        },
        {
            "gene":             "SLC19A2",
            "disease":          "TRMA (Rogers syndrome)",
            "first_line":       "Thiamine 25-100 mg/day orally; reticulocyte response in 1-2 weeks confirms TRMA; insulin for DM if required (may reduce insulin dose with thiamine); hearing aids for SNHL (irreversible — refer audiology immediately)",
            "precision_therapy":"Titrate thiamine to haematological and glycaemic response; maintain lifelong; folic acid for residual megaloblastic component; cochlear implant considered for severe SNHL",
            "avoid":            "Nitrous oxide (N2O) CONTRAINDICATED (as in all megaloblastic/folate-pathway disorders); high-carbohydrate diet (worsens DM component); metformin (impairs thiamine absorption)",
            "monitoring":       "CBC + MCV monthly initially; HbA1c 3-monthly; annual audiogram (SNHL monitoring — irreversible but baseline); thiamine levels; autoantibody panel to confirm negative (distinguish from autoimmune DM)",
        },
        {
            "gene":             "TPK1",
            "disease":          "THMD1",
            "first_line":       "High-dose thiamine 200-600 mg/day; thiamine pyrophosphate (TPP) directly if available (bypasses TPK1 block); riboflavin 100-200 mg/day (complex I/II supportive); CoQ10 10-20 mg/kg/day",
            "precision_therapy":"Ketogenic diet CONTRAINDICATED in PDH deficiency component; dichloroacetate (DCA) for lactic acidosis crisis management; carbohydrate-rich diet during crises; IV glucose for metabolic crisis",
            "avoid":            "Ketogenic diet; valproate (mitochondrial toxicity + BCKDH inhibition); linezolid; chloramphenicol (mitochondrial toxin)",
            "monitoring":       "Plasma lactate monthly; brain MRI 6-monthly; developmental assessments; serial TPP:thiamine ratio; fibroblast enzyme function if available",
        },
        {
            "gene":             "SLC25A19",
            "disease":          "Amish microcephaly / THMD4",
            "first_line":       "Thiamine 100-300 mg/day (partial benefit in THMD4 phenotype); prenatal genetic counselling for Amish families (mandatory); p.G177A carrier screening in Old Order Amish communities",
            "precision_therapy":"Amish microcephaly (severe): largely palliative; seizure management (LEV preferred); supportive nutrition; THMD4 (milder): thiamine + riboflavin + CoQ10",
            "avoid":            "Valproate; linezolid; fasting (glucose infusion for metabolic crises)",
            "monitoring":       "Head circumference monthly in infancy; urine 2-oxoglutarate as treatment response biomarker; brain MRI; seizure monitoring",
        },
        {
            "gene":             "BTD",
            "disease":          "Biotinidase deficiency",
            "first_line":       "Biotin 5-10 mg/day oral — start as soon as NBS confirmed; this dose is PHARMACOLOGICAL (not nutritional supplementation); continue LIFELONG; neonatal onset from NBS prevents ALL complications",
            "precision_therapy":"Partial deficiency (10-30% activity): biotin 2-5 mg/day; biotin supplementation essential during illness; biotin injection if GI intolerance; avoid very high-dose biotin (>10 mg/day) unless clinically indicated — lab test interference",
            "avoid":            "Raw egg white (avidin binds biotin — irrelevant at pharmacological doses but counsel); high-dose biotin >5-10 mg without justification (FDA warning 2019: interferes with immunoassay thyroid/troponin/hCG tests)",
            "monitoring":       "Visual evoked potentials annually (optic atrophy monitoring); annual audiogram; annual ophthalmology; BTD enzyme activity 1-2 years post-treatment; developmental milestones",
        },
        {
            "gene":             "HLCS",
            "disease":          "Multiple carboxylase deficiency neonatal (MCD1)",
            "first_line":       "Biotin 10-20 mg/day oral — start IMMEDIATELY at diagnosis; may need 40 mg/day during neonatal crises; carnitine 100 mg/kg/day (carnitine depleted by excess acylcarnitines); protein restriction transiently during crises",
            "precision_therapy":"Higher biotin doses may be needed due to reduced HLCS affinity; monitor organic acids (3-OHisovalerate should normalise); dietary leucine restriction temporarily if MCC failure persists",
            "avoid":            "Fasting (triggerscatabolism); delays in diagnosis (before NBS catches it); raw egg white (avidin interference, as with BTD); valproate (hepatotoxic + biotin depletion)",
            "monitoring":       "Urine organic acids (3-OHisovalerate + 3-MCG + propionate — should normalise on biotin); plasma carnitine monthly; lactate; brain MRI if encephalopathy; developmental outcomes",
        },
        {
            "gene":             "LIAS",
            "disease":          "Lipoic acid synthesis deficiency",
            "first_line":       "Racemic alpha-lipoic acid (ALA) 60-100 mg/kg/day; thiamine 200 mg/day; riboflavin 200 mg/day; CoQ10 10-20 mg/kg/day; IV glucose/carnitine for metabolic crises",
            "precision_therapy":"No curative therapy available; ALA provides exogenous lipoate for salvage pathways; experimental: R-lipoic acid (more bioavailable form); stabilise with supplements + seizure management (LEV preferred)",
            "avoid":            "KETOGENIC DIET ABSOLUTELY CONTRAINDICATED (worsens PDH/lactic acidosis); valproate; fasting; linezolid; chloramphenicol; high-fat low-carbohydrate diets",
            "monitoring":       "Plasma lactate monthly; urine 2-oxoglutarate; brain MRI every 6 months; seizure monitoring; fibroblast enzyme assays to confirm diagnosis; developmental",
        },
        {
            "gene":             "LIPT1",
            "disease":          "Lipoic acid attachment defect type 1",
            "first_line":       "Racemic ALA 60-100 mg/kg/day + thiamine + riboflavin (same as LIAS); IV glucose for crises; carnitine supplementation",
            "precision_therapy":"Identical approach to LIAS; BCKDH less affected so BCAA restriction less critical than in LIAS; lipoate supplementation + thiamine combination",
            "avoid":            "KETOGENIC DIET CONTRAINDICATED; valproate; fasting; same contraindications as LIAS",
            "monitoring":       "Same as LIAS; BCKDH assay in fibroblasts to confirm LIPT1 (less affected vs LIAS); molecular confirmation required",
        },
    ]

    return {
        "atlas_id":             ATLAS_ID,
        "pathway_groups":       pathway_groups,
        "diagnostic_cascade":   diagnostic_cascade,
        "treatment_pathways":   treatment_pathways,
        "seeds":                f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "empirical_treatment_rule": "BTBGD (SLC19A3): start biotin+thiamine IMMEDIATELY on MRI bilateral BG diagnosis — COMPLETE REVERSAL possible; BTD: treat ALL NBS-positive immediately — delay causes IRREVERSIBLE optic atrophy; HLCS neonatal: start biotin at presentation — delay = permanent ID",
        "keto_diet_contraindications": "Ketogenic diet CONTRAINDICATED in LIAS and LIPT1 (forces PDH-dependent metabolism in a PDH-deficient patient — worsens lactic acidosis); TPK1 similarly AVOID (PDH component fails); only BTD/HLCS/SLC19A2/SLC19A3 may consider KD with caution",
    }


def generate_definitions() -> dict[str, Any]:
    key_distinguishers = [
        {
            "name":        "SLC19A3 vs SLC19A2: blood thiamine NORMAL vs LOW",
            "description": "SLC19A3 (BTBGD): blood thiamine NORMAL (intestinal absorption intact via SLC19A2/SLC19A1; only brain transport fails — CNS-selective deficiency); plasma thiamine does NOT diagnose BTBGD. SLC19A2 (TRMA): blood thiamine LOW (intestinal absorption impaired — systemic deficiency including bone marrow and beta cells). THIS IS THE PRIMARY BLOOD THIAMINE SPLIT: normal blood thiamine + brain lesions → SLC19A3; low blood thiamine + megaloblastic anemia → SLC19A2",
        },
        {
            "name":        "BTD vs HLCS: BTD enzyme activity is the critical DDx",
            "description": "BTD deficiency: BTD enzyme activity severely LOW (<10% profound, 10-30% partial) — measurable on dried blood spot (DBS); NBS detects BTD via DBS enzyme assay; presents late (infantile to early childhood). HLCS deficiency: BTD enzyme activity NORMAL — biotinidase enzyme is intact, but the attachment enzyme (HLCS) is deficient; NBS MISSES HLCS; presents neonatally. KEY RULE: neonatal metabolic crisis + biotin-type organic acids + NORMAL BTD activity = HLCS deficiency",
        },
        {
            "name":        "LIAS vs LIPT1: BCKDH activity discriminates",
            "description": "LIAS deficiency: ALL FOUR lipoate-dependent complexes (PDH + alphaKGDH + BCKDH + GCS) severely deficient — alloisoleucine elevated (BCKDH) + alloisoleucine pattern mimics MSUD-DLD. LIPT1 deficiency: PDH + alphaKGDH + GCS deficient but BCKDH relatively SPARED (LIPT1 substrate specificity does not fully include BCKDH) — alloisoleucine less elevated or normal. This single enzyme assay (BCKDH activity in fibroblasts) discriminates LIAS from LIPT1 before molecular diagnosis",
        },
        {
            "name":        "SLC25A19: 2-oxoglutarate elevation is the biomarker",
            "description": "SLC25A19 encodes mitochondrial TPP transporter; TPP import failure → alphaKGDH (KGDH) cannot function → 2-oxoglutarate (2-OG / alpha-ketoglutarate) MARKEDLY ELEVATED in urine; this is unlike other thiamine disorders where 2-OG is less specifically elevated; Amish microcephaly with microcephaly head circumference -7 to -10 SD + 2-OG elevation + Amish ancestry = SLC25A19 p.G177A until proven otherwise",
        },
        {
            "name":        "TRMA SNHL: irreversibility is a clinical rule",
            "description": "In TRMA (SLC19A2): sensorineural hearing loss (SNHL) is IRREVERSIBLE regardless of how early thiamine is started — cochlear hair cells require thiamine during development and the damage is irreversible once established; CONTRAST: megaloblastic anemia and DM ARE reversible with thiamine; CLINICAL RULE: start hearing aids at diagnosis; cochlear implantation may be considered for severe bilateral SNHL; do not delay hearing intervention waiting for thiamine response",
        },
        {
            "name":        "BTBGD empirical treatment principle",
            "description": "SLC19A3 (BTBGD): initiate biotin + thiamine treatment EMPIRICALLY when bilateral BG lesions on MRI during febrile encephalopathy — DO NOT wait for genetic confirmation; MRI lesions can COMPLETELY REVERSE if treated early; delayed treatment → permanent neurological deficit or death; the treatment (biotin + thiamine) is SAFE with no significant adverse effects — the risk of NOT treating far exceeds any treatment risk; treat first, confirm diagnosis later",
        },
        {
            "name":        "Ketogenic diet contraindication in lipoate disorders",
            "description": "LIAS and LIPT1: ABSOLUTELY CONTRAINDICATED for ketogenic diet; KD increases reliance on fat oxidation and acetyl-CoA entry into TCA cycle — BOTH require functional PDH (which is deficient due to lipoate absence); KD forces PDH-dependent metabolism in a patient whose PDH is severely deficient → catastrophic worsening of lactic acidosis; same applies to any high-fat low-carbohydrate diet; CARBOHYDRATE is the preferred energy substrate in LIAS/LIPT1 (does not require PDH as efficiently as fat)",
        },
        {
            "name":        "Biotin lab test interference",
            "description": "Biotin supplementation at pharmacological doses (>5-10 mg/day, as used in BTD treatment) interferes with biotin-streptavidin immunoassay platforms; FALSE LOW: TSH, troponin, parathyroid hormone, hCG (streptavidin-biotin blocked); FALSE HIGH: T4, T3, LH, FSH (some assay platforms); FDA safety communication 2019; CLINICAL RULE: when ordering thyroid function/troponin/hCG in a patient on biotin therapy — either use non-streptavidin assays or withhold biotin for 12-24 hours before testing; always inform the laboratory",
        },
        {
            "name":        "Amish microcephaly: founder principle",
            "description": "SLC25A19 p.G177A is a private founder mutation in Old Order Amish with carrier frequency approximately 1 in 10 births (estimated ~5% carrier rate) — one of the HIGHEST carrier frequencies for any severe metabolic disease; neonatal microcephaly (HC -7 to -10 SD) in Amish infant = SLC25A19 until proven otherwise; carrier testing and genetic counselling MANDATORY for all Amish couples with affected child; prenatal diagnosis (CVS/amnio) critical for subsequent pregnancies",
        },
        {
            "name":        "Combined lipoate-dependent complex deficiency: enzyme assay first",
            "description": "If fibroblast/muscle enzyme studies show COMBINED deficiency of multiple complexes including PDH + alphaKGDH + GCS (± BCKDH), consider lipoate pathway first (LIAS/LIPT1) before expensive mtDNA or nuclear gene panel; lipoate is the shared cofactor; compare BCKDH activity to PDH activity (BCKDH spared → LIPT1 more likely); lipoate supplementation trial while awaiting molecular diagnosis",
        },
    ]

    treatment_matrix = [
        {
            "intervention":    "Biotin (pharmacological dose)",
            "indications":     "BTD (5-10 mg/day), HLCS (10-20 mg/day), SLC19A3 (5-10 mg/day — in combination with thiamine)",
            "mechanism":       "BTD: provides free biotin for carboxylase biotinylation (bypassing recycling failure); HLCS: excess free biotin overwhelms reduced-affinity HLCS enzyme (substrate saturation effect); SLC19A3: biotin transport by THTR2 is dual — brain biotin also deficient",
            "dose":            "BTD: 5-10 mg/day oral; HLCS: 10-20 mg/day (higher for neonatal crises); SLC19A3: 5-10 mg/day (combined with thiamine)",
            "contraindicated": "High-dose biotin (>10 mg) may interfere with streptavidin-based laboratory immunoassays (thyroid, troponin, hCG — FDA 2019 warning); not contraindicated clinically per se",
        },
        {
            "intervention":    "Thiamine (pharmacological dose)",
            "indications":     "SLC19A3 (100-300 mg/day + biotin), SLC19A2 (25-100 mg/day), TPK1 (200-600 mg/day), SLC25A19 (100-300 mg/day partial)",
            "mechanism":       "Provides excess thiamine substrate to: (a) overcome reduced brain transport (SLC19A3 — saturation of residual SLC19A1 transport); (b) replace systemic deficiency (SLC19A2); (c) overwhelm reduced TPK1 kinase activity by substrate saturation (TPK1); (d) partially restore mitochondrial TPP via non-SLC25A19 import (SLC25A19)",
            "dose":            "SLC19A3: 100-300 mg/day; SLC19A2: 25-100 mg/day; TPK1: 200-600 mg/day",
            "contraindicated": "No major contraindications; avoid simultaneous metformin use (impairs thiamine absorption in SLC19A2 via shared transporter competition)",
        },
        {
            "intervention":    "Alpha-lipoic acid (ALA, racemic R/S-lipoic acid)",
            "indications":     "LIAS (60-100 mg/kg/day), LIPT1 (60-100 mg/kg/day)",
            "mechanism":       "Exogenous lipoic acid undergoes partial cellular uptake and can be salvaged/used in some cases; R-enantiomer is the natural form; partially restores lipoate-dependent complex function via salvage pathways that bypass LIAS/LIPT1 steps",
            "dose":            "60-100 mg/kg/day; may need higher doses in neonatal crisis",
            "contraindicated": "No absolute contraindications; may cause GI symptoms at high doses; not established to be curative — supportive use",
        },
    ]

    return {
        "atlas_id":          ATLAS_ID,
        "key_distinguishers":key_distinguishers,
        "treatment_matrix":  treatment_matrix,
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "primary_cascade_split": "STEP 1: Is there bilateral BG on MRI + episodic fever-triggered encephalopathy → SLC19A3 (treat NOW). STEP 2: Is there megaloblastic anemia + DM + SNHL (autoantibodies negative) → SLC19A2 TRMA. STEP 3: Is BTD enzyme activity low → BTD deficiency. STEP 4: Is BTD activity NORMAL but neonatal ketoacidosis + organic acids → HLCS. STEP 5: Is there combined lipoate-complex deficiency (PDH+alphaKGDH+GCS) → LIAS vs LIPT1 (BCKDH activity discriminates)",
        "all_genes_ar":      True,
        "cofactor_groups": {
            "thiamine":  {"genes": THIAMINE_GROUP, "treatment": "Thiamine supplementation (± biotin for SLC19A3)"},
            "biotin":    {"genes": BIOTIN_GROUP,   "treatment": "Biotin supplementation (pharmacological doses)"},
            "lipoate":   {"genes": LIPOATE_GROUP,  "treatment": "Alpha-lipoic acid + thiamine + riboflavin"},
        },
    }
