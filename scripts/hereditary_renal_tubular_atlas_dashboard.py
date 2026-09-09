#!/usr/bin/env python3
"""Hereditary-Renal-Tubular-Atlas — Complete 8-Gene Hereditary Renal Tubular Disorders Atlas
SLC12A1 (NKCC2 / Bartter type 1; 1099 aa; 15q21.1; AR;
         Na-K-2Cl co-transporter 2 — thick ascending limb (TAL);
         neonatal Bartter — severe polyhydramnios, premature birth, life-threatening salt wasting;
         indomethacin + KCl + NaCl supplementation;
         seed SEED_BASE+0) .
KCNJ1   (ROMK / Bartter type 2; 391 aa; 11q24.3; AR;
         renal outer medullary K channel — recycles K in TAL + collecting duct;
         transient HYPERKALEMIA in neonates PATHOGNOMONIC before HypOkalemia;
         severe neonatal polyhydramnios;
         seed SEED_BASE+1) .
CLCNKB  (ClC-Kb / Bartter type 3; 687 aa; 1p36.13; AR;
         basolateral chloride channel in TAL and DCT;
         MOST COMMON Bartter subtype; classic childhood/milder presentation;
         complete gene deletion most common mutation;
         seed SEED_BASE+2) .
BSND    (Barttin / Bartter type 4; 320 aa; 1p32.3; AR;
         beta-subunit of ClC-Ka AND ClC-Kb chloride channels;
         Bartter type 4 = Bartter + SENSORINEURAL HEARING LOSS PATHOGNOMONIC;
         barttin expressed in ear + kidney;
         seed SEED_BASE+3) .
SLC12A3 (NCCT / Gitelman; 1021 aa; 16q13; AR;
         Na-Cl co-transporter in distal convoluted tubule (DCT);
         HYPOMAGNESAEMIA + LOW URINE CALCIUM = PATHOGNOMONIC — distinguishes from Bartter;
         most common hereditary salt-wasting nephropathy 1:40,000;
         seed SEED_BASE+4) .
CLCN5   (ClC-5 / Dent disease 1; 746 aa; Xp11.23; XLR;
         endosomal chloride/H+ exchanger in proximal tubule;
         LOW-MOLECULAR-WEIGHT PROTEINURIA PATHOGNOMONIC + nephrocalcinosis + nephrolithiasis;
         tubular proteinuria (beta-2-microglobulin) on dipstick negative for albumin;
         seed SEED_BASE+5) .
OCRL    (Lowe syndrome / OCRL1; 901 aa; Xq26.1; XLR;
         phosphatidylinositol 4,5-bisphosphate 5-phosphatase — endosomal trafficking;
         CONGENITAL CATARACTS + INTELLECTUAL DISABILITY + RENAL TUBULAR ACIDOSIS = TRIAD PATHOGNOMONIC;
         dense cataracts present at birth;
         seed SEED_BASE+6) .
AGXT    (Primary Hyperoxaluria type 1 / PH1; 392 aa; 2q37.3; AR;
         alanine-glyoxylate aminotransferase — peroxisomal; converts glyoxylate to glycine;
         SYSTEMIC OXALOSIS — nephrocalcinosis → ESRD → oxalate deposition in bones/heart/retina;
         pyridoxine (B6) reduces oxalate in 20-30% (responsive genotypes);
         combined liver-kidney transplant curative — ONLY disease cured by OLT;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1910-1917)
"""

import random

SEED_BASE = 1910

RENAL_TUBULAR_GENES = [
    # -- SLC12A1 / NKCC2 — Bartter type 1 — Neonatal most severe ----------------------
    {
        "gene": "SLC12A1",
        "alt_name": "NKCC2 / Bartter-1",
        "protein": (
            "SLC12A1/NKCC2 -- 15q21.1 AR -- NKCC2-1099aa -- "
            "Bartter-Type-1-Neonatal-Most-Severe -- "
            "Polyhydramnios-Premature-Life-Threatening-Salt-Wasting -- "
            "Hypokalemia-Metabolic-Alkalosis-Hypercalciuria-Nephrocalcinosis -- "
            "Indomethacin-Plus-KCl-Plus-NaCl-Supplementation"
        ),
        "locus": "15q21.1",
        "protein_size": "1099 aa",
        "inheritance": "AR",
        "age_of_onset": "Prenatal (polyhydramnios 3rd trimester) / neonatal",
        "key_biomarker": (
            "Serum: hypokalemia, metabolic alkalosis (high bicarbonate), hypercalcemia, "
            "elevated renin + aldosterone, low-normal BP; "
            "urine: hypercalciuria (spot urine Ca:Cr > 0.7), elevated prostaglandin E2; "
            "fetal: polyhydramnios from fetal polyuria (TAL dysfunction); "
            "SLC12A1 gene sequencing; "
            "renal ultrasound: nephrocalcinosis"
        ),
        "pathognomonic": (
            "Neonatal severe Bartter: polyhydramnios → premature birth → life-threatening salt + water wasting; "
            "hypokalemic metabolic alkalosis + hypercalciuria + elevated renin/aldosterone + normal-to-low BP; "
            "nephrocalcinosis from hypercalciuria; "
            "similar to KCNJ1 (Bartter type 2) — gene sequencing required to distinguish"
        ),
        "treatment": (
            "INDOMETHACIN — prostaglandin synthesis inhibitor; reduces urine prostaglandin E2; "
            "primary pharmacological treatment; 1-2 mg/kg/day; monitor renal function; "
            "ELECTROLYTE SUPPLEMENTATION: oral KCl + NaCl supplementation lifelong; "
            "adjust to normal serum K; high-dose supplementation needed; "
            "POTASSIUM-SPARING DIURETICS: spironolactone or amiloride as adjunct; "
            "RENAL PROTECTION: monitor for CKD progression from nephrocalcinosis; "
            "GROWTH HORMONE: some use for growth retardation; "
            "MONITORING: serum K, bicarbonate, renin, renal function 3-monthly"
        ),
        "critical_flags": [
            "POLYHYDRAMNIOS-NEONATAL-BARTTER — TAL NKCC2 failure → fetal polyuria → polyhydramnios; premature birth; suspect Bartter 1/2 in any polyhydramnios + prematurity + hypokalemia",
            "HYPOKALEMIA-METABOLIC-ALKALOSIS-HALLMARK — low K + high bicarb + normal-to-low BP = tubular Bartter; CONTRAST HYPERtension + HYPOkalemia = aldosteronism/Liddle; always check BP",
            "HYPERCALCIURIA-NEPHROCALCINOSIS — daily urine Ca elevated; nephrocalcinosis by ultrasound in infancy; monitor CKD progression; indomethacin + citrate reduce Ca excretion",
            "INDOMETHACIN-FIRST-LINE — COX inhibitor reduces prostaglandin E2 which amplifies tubular dysfunction; correct electrolytes BEFORE indomethacin if severely hypokalemic",
            "NKCC2-TAL-LOOP-DIURETIC-MIMIC — NKCC2 is the target of furosemide/bumetanide; Bartter 1 = 'congenital furosemide effect'; hypokalemia + alkalosis = furosemide mimic",
            "DISTINGUISH-TYPE-1-2-3 — Bartter 1 (SLC12A1) and 2 (KCNJ1) = neonatal severe; Bartter 3 (CLCNKB) = classic childhood; Bartter 4 (BSND) + SNHL; gene sequencing required",
            "RENAL-CONCENTRATING-DEFECT — TAL is critical for urine concentration via countercurrent mechanism; Bartter 1/2: polyuria, polydipsia even after correction",
            "GROWTH-FAILURE-CHRONIC — chronic electrolyte loss + metabolic alkalosis → growth retardation; early diagnosis + treatment improves growth outcomes"
        ],
        "alias": (
            "SLC12A1 (NKCC2 — Na-K-2Cl cotransporter 2); OMIM gene 600839; "
            "Bartter syndrome type 1 OMIM 601678. "
            "15q21.1; 1099 aa; ~120 kDa; thick ascending limb apical membrane; autosomal recessive. "
            "FUNCTION: SLC12A1 encodes NKCC2, the apical Na-K-2Cl cotransporter in the thick ascending limb (TAL) "
            "of the loop of Henle. NKCC2 reabsorbs Na, K, and 2Cl simultaneously using the Na gradient "
            "established by Na/K-ATPase. This drives 25-30% of renal NaCl reabsorption. "
            "TAL is electrically impermeable to water — its salt reabsorption creates the "
            "hyperosmolar medullary interstitium required for urinary concentration. "
            "NKCC2 is also the molecular target of loop diuretics (furosemide, bumetanide). "
            "Loss of NKCC2 = functional loop diuretic in utero and postnatally. "
            "CLINICAL PRESENTATION: "
            "Prenatal: polyhydramnios (3rd trimester) from fetal polyuria; premature birth; "
            "Neonatal: life-threatening Na, K, Cl wasting; hypokalemic metabolic alkalosis; "
            "hypercalciuria → nephrocalcinosis; fever from dehydration; "
            "Low-to-normal blood pressure (no Liddle); elevated renin + aldosterone; "
            "chronic: growth failure, polyuria, polydipsia, nephrocalcinosis → CKD. "
            "DIAGNOSIS: "
            "Serum: hypokalemia, metabolic alkalosis, hyponatremia. "
            "Urine: high Ca:Cr ratio, elevated prostaglandin E2, high Cl. "
            "Genetics: SLC12A1 sequencing (>150 mutations). "
            "MANAGEMENT: "
            "Indomethacin (1-2 mg/kg/day divided). "
            "Oral KCl + NaCl supplementation. "
            "Add spironolactone or amiloride for additional K sparing. "
            "Monitor renal function, growth, nephrocalcinosis. "
            "Renal transplant if ESRD."
        ),
    },
    # -- KCNJ1 / ROMK — Bartter type 2 — Transient hyperK then hypoK ------------------
    {
        "gene": "KCNJ1",
        "alt_name": "ROMK / Bartter-2",
        "protein": (
            "KCNJ1/ROMK -- 11q24.3 AR -- ROMK-391aa -- "
            "Bartter-Type-2-Transient-Neonatal-Hyperkalemia-PATHOGNOMONIC -- "
            "Then-Hypokalemia-Metabolic-Alkalosis-Evolves-Later -- "
            "ROMK-Dual-Role-TAL-K-Recycling-Plus-Collecting-Duct-K-Secretion -- "
            "Severe-Neonatal-Polyhydramnios-Salt-Wasting"
        ),
        "locus": "11q24.3",
        "protein_size": "391 aa",
        "inheritance": "AR",
        "age_of_onset": "Prenatal (polyhydramnios) / neonatal",
        "key_biomarker": (
            "Serum: TRANSIENT HYPERKALEMIA in first days → evolves to hypokalemia + metabolic alkalosis; "
            "elevated renin + aldosterone; hypercalciuria; "
            "urine: high Ca:Cr, elevated PGE2; "
            "KCNJ1 gene sequencing; "
            "fetal: polyhydramnios; premature birth; "
            "renal ultrasound: nephrocalcinosis"
        ),
        "pathognomonic": (
            "TRANSIENT NEONATAL HYPERKALEMIA then switch to hypokalemia = PATHOGNOMONIC Bartter type 2 pattern; "
            "ROMK has dual role: (1) K recycling in TAL (NKCC2 function) AND (2) K secretion in collecting duct; "
            "loss of collecting duct K secretion → initial hyperkalemia; "
            "loss of TAL K recycling → eventual hypokalemia as aldosterone system activates; "
            "distinct from Bartter 1 by transient hyperK phase"
        ),
        "treatment": (
            "INDOMETHACIN — as for Bartter type 1; COX inhibitor; 1-2 mg/kg/day; "
            "ELECTROLYTE SUPPLEMENTATION: KCl + NaCl; "
            "CAUTION DURING HYPERKALEMIC PHASE: avoid K supplementation until hypoK ensues; "
            "POTASSIUM-SPARING DIURETICS: amiloride preferred over spironolactone (since aldosterone already elevated); "
            "monitor serum K carefully — can oscillate; "
            "renal function monitoring; nephrocalcinosis management; "
            "TRANSITION: initial hyperK management → then switch to K supplementation as hypoK develops; "
            "careful monitoring during transition critical"
        ),
        "critical_flags": [
            "TRANSIENT-HYPERKALEMIA-NEONATAL-PATHOGNOMONIC — initial hyperK (collecting duct K secretion lost) → converts to hypoK (TAL recycling defect dominates); unique Bartter 2 hallmark",
            "ROMK-DUAL-ROLE — ROMK channels in both TAL (K recycling for NKCC2) and cortical collecting duct (K secretion = principal K excretory pathway); loss → biphasic K disturbance",
            "DO-NOT-SUPPLEMENT-K-IN-HYPERKALEMIC-PHASE — neonatal Bartter 2: initial hyperK is dangerous; K supplementation in this phase → fatal hyperkalemia; wait until hypoK develops",
            "KCNJ1-PSEUDO-HYPOALDOSTERONISM — initial hyperK + aldosterone ELEVATED mimics pseudohypoaldosteronism type 1; but aldosterone IS high (secondary); gene sequencing distinguishes",
            "POLYHYDRAMNIOS-PREMATURE-SAME-AS-TYPE1 — prenatal presentation indistinguishable from Bartter 1; only neonatal biphasic K course distinguishes clinically before genetics",
            "AMILORIDE-PREFERRED-ADJUNCT — in Bartter 2, collecting duct K secretion already lost; amiloride blocks collecting duct Na channel (not K channel); preferred over spironolactone",
            "LONG-TERM-HYPOKALEMIA-DOMINANT — after first weeks: hypoK becomes the chronic problem; same management as Bartter 1: indomethacin + K + NaCl supplementation",
            "CKD-NEPHROCALCINOSIS — chronic hypercalciuria → nephrocalcinosis → CKD; monitor eGFR; citrate supplementation may reduce Ca oxalate crystallization"
        ],
        "alias": (
            "KCNJ1 (ROMK — renal outer medullary K channel); OMIM gene 600359; "
            "Bartter syndrome type 2 OMIM 241200. "
            "11q24.3; 391 aa; ~45 kDa; inward rectifier K channel; autosomal recessive. "
            "FUNCTION: KCNJ1 encodes ROMK (Kir1.1), an ATP-regulated inward rectifier potassium channel. "
            "ROMK has two critical renal roles: "
            "(1) TAL: apical K recycling — NKCC2 imports K but luminal K is low; ROMK recycles K back to lumen "
            "to maintain NKCC2 activity; without ROMK recycling, NKCC2 function collapses; "
            "(2) Cortical collecting duct (CCD): principal cell apical K secretion — the primary mechanism of "
            "urinary K excretion regulated by aldosterone and flow. "
            "Loss of ROMK → dual defect: TAL fails (salt wasting → hypokalemia, ultimately) + "
            "CCD K secretion fails (initial hyperkalemia). "
            "CLINICAL PRESENTATION: "
            "Prenatal: polyhydramnios from fetal polyuria; premature birth. "
            "Neonatal: transient HYPERKALEMIA (CCD K secretion lost) — unique to Bartter 2; "
            "evolves over days-weeks to hypokalemia as TAL salt-wasting activates RAAS; "
            "metabolic alkalosis, hypercalciuria, nephrocalcinosis. "
            "DIAGNOSIS: "
            "Serum: biphasic K (hyperK → hypoK) + metabolic alkalosis. "
            "KCNJ1 sequencing: many loss-of-function mutations. "
            "MANAGEMENT: "
            "Indomethacin + electrolyte supplementation (after hyperK phase resolves). "
            "Careful monitoring of K during transition. "
            "Nephrocalcinosis surveillance."
        ),
    },
    # -- CLCNKB / ClC-Kb — Bartter type 3 — Most common; classic childhood ------------
    {
        "gene": "CLCNKB",
        "alt_name": "ClC-Kb / Bartter-3",
        "protein": (
            "CLCNKB/ClC-Kb -- 1p36.13 AR -- ClC-Kb-687aa -- "
            "Bartter-Type-3-MOST-COMMON-Bartter-Classic-Childhood -- "
            "Basolateral-Chloride-Channel-TAL-DCT -- "
            "Complete-Gene-Deletion-MOST-COMMON-Mutation -- "
            "Milder-Presentation-Later-Onset-Variable-Phenotype"
        ),
        "locus": "1p36.13",
        "protein_size": "687 aa",
        "inheritance": "AR",
        "age_of_onset": "Infancy to childhood (milder than types 1 and 2)",
        "key_biomarker": (
            "Serum: hypokalemia + metabolic alkalosis + normal-to-low BP + elevated renin/aldosterone; "
            "urine: high Cl, variable hypercalciuria (less than types 1/2); "
            "CLCNKB gene sequencing: complete deletion of CLCNKB most common (>50% alleles); "
            "no polyhydramnios in most cases; "
            "phenotype can overlap with Gitelman (same basolateral DCT Cl channel)"
        ),
        "pathognomonic": (
            "MOST COMMON hereditary Bartter subtype; "
            "classic Bartter: hypokalemia + metabolic alkalosis + elevated renin/aldosterone + normal-to-low BP; "
            "later onset (infancy-childhood) and milder than types 1/2; "
            "complete CLCNKB gene deletion detectable by MLPA; "
            "phenotypic overlap with Gitelman (variable DCT involvement) — genotyping essential"
        ),
        "treatment": (
            "INDOMETHACIN — COX inhibitor, first line; less dramatic response than types 1/2 in some; "
            "ELECTROLYTE SUPPLEMENTATION: KCl chronically; "
            "POTASSIUM-SPARING DIURETICS: amiloride or spironolactone; "
            "ACE INHIBITORS: used in some centres for additional RAAS blockade; "
            "less aggressive supplementation needed than neonatal types; "
            "MONITORING: serum K, bicarb, renin, renal function; "
            "DIET: high-salt diet; avoid diuretics (furosemide worsens); "
            "prognosis generally better than types 1/2 for renal function"
        ),
        "critical_flags": [
            "MOST-COMMON-BARTTER-CLCNKB — Bartter type 3 is the most prevalent Bartter subtype worldwide; consider in any child with hypokalemic metabolic alkalosis + normal BP",
            "COMPLETE-DELETION-MLPA — >50% of CLCNKB mutations = complete gene deletion; standard Sanger sequencing misses deletions; MLPA or gene dosage analysis mandatory",
            "PHENOTYPIC-OVERLAP-GITELMAN — ClC-Kb expressed in DCT as well as TAL; CLCNKB mutations can produce Gitelman-like phenotype (hypoMg, low urine Ca); genotyping mandatory",
            "BASOLATERAL-CHANNEL — ClC-Kb exports Cl across basolateral membrane of TAL and DCT; apical channels are NKCC2 (TAL) and NCCT (DCT); both depend on basolateral Cl exit via ClC-Kb",
            "MILDER-NEONATAL-COURSE — unlike types 1/2: usually no severe polyhydramnios; not premature; presentation in infancy-childhood with growth failure + electrolyte disturbance",
            "BARTTIN-CO-DEPENDENCY — ClC-Kb requires barttin (BSND) as beta-subunit; BSND mutations affect both ClC-Ka and ClC-Kb simultaneously (Bartter type 4 + SNHL)",
            "DIAGNOSE-BEFORE-RENAL-FAILURE — chronic untreated hypokalemia → vacuolar nephropathy → progressive CKD; early diagnosis + treatment preserves renal function",
            "VARIABLE-PHENOTYPE — Bartter 3 phenotypic spectrum: mild (asymptomatic electrolyte anomaly) to severe (growth retardation, polyuria, nephrocalcinosis); genotype-phenotype poor"
        ],
        "alias": (
            "CLCNKB (ClC-Kb — kidney-specific basolateral chloride channel b); OMIM gene 602023; "
            "Bartter syndrome type 3 OMIM 607364. "
            "1p36.13; 687 aa; ~74 kDa; basolateral membrane of TAL and DCT; autosomal recessive. "
            "FUNCTION: CLCNKB encodes ClC-Kb, a voltage-gated chloride channel of the CLC family. "
            "ClC-Kb localises to the basolateral membrane of the TAL and distal convoluted tubule (DCT). "
            "It functions as a heterodimer with barttin (BSND as β-subunit). "
            "In TAL: ClC-Kb exports Cl across the basolateral membrane after NKCC2 imports Na/K/2Cl apically. "
            "In DCT: ClC-Kb similarly exports Cl basolaterally after NCCT imports Na/Cl apically. "
            "Loss of ClC-Kb → impaired Cl exit → intracellular Cl accumulation → NKCC2 + NCCT activity falls "
            "→ salt wasting → hypokalemic metabolic alkalosis. "
            "CLCNKB shares 96% sequence identity with CLCNKA — gene panel must resolve both. "
            "CLINICAL PRESENTATION: "
            "Infancy to childhood: hypokalemia, metabolic alkalosis, elevated renin/aldosterone, normal-to-low BP. "
            "Milder than neonatal Bartter types 1/2: no polyhydramnios typically; failure to thrive, polyuria. "
            "Variable: some patients have Gitelman-like features (hypoMg, hypocalciuria). "
            "DIAGNOSIS: "
            "CLCNKB sequencing + MLPA (deletion analysis). "
            "Serum and urine electrolyte profile. "
            "MANAGEMENT: "
            "Indomethacin + KCl supplementation + amiloride/spironolactone. "
            "Monitoring for CKD. Better prognosis than types 1/2."
        ),
    },
    # -- BSND / Barttin — Bartter type 4 — SNHL distinguishes -------------------------
    {
        "gene": "BSND",
        "alt_name": "Barttin / Bartter-4",
        "protein": (
            "BSND/Barttin -- 1p32.3 AR -- Barttin-320aa -- "
            "Bartter-Type-4-PLUS-SENSORINEURAL-HEARING-LOSS-PATHOGNOMONIC -- "
            "Beta-Subunit-ClC-Ka-AND-ClC-Kb-Both-Channels -- "
            "Expressed-Inner-Ear-Marginal-Cells-Stria-Vascularis -- "
            "Severe-Neonatal-Bartter-Plus-SNHL-Distinct-From-Other-Types"
        ),
        "locus": "1p32.3",
        "protein_size": "320 aa",
        "inheritance": "AR",
        "age_of_onset": "Neonatal (Bartter) + congenital SNHL",
        "key_biomarker": (
            "Serum: hypokalemia + metabolic alkalosis + elevated renin/aldosterone (same as other Bartter types); "
            "SENSORINEURAL HEARING LOSS present from birth PATHOGNOMONIC; "
            "ABR (auditory brainstem response) confirms SNHL; "
            "renal: polyhydramnios, premature birth (neonatal severity similar to types 1/2); "
            "BSND gene sequencing; "
            "urine: hypercalciuria, elevated PGE2"
        ),
        "pathognomonic": (
            "Bartter syndrome TYPE 4 = ONLY Bartter type with SENSORINEURAL HEARING LOSS — PATHOGNOMONIC; "
            "barttin is the beta-subunit for BOTH ClC-Ka AND ClC-Kb; "
            "stria vascularis marginal cells require barttin+ClC-Ka for endolymph K+ secretion; "
            "loss of barttin → simultaneous renal Bartter + inner ear dysfunction; "
            "SNHL is congenital, bilateral, severe-profound; cochlear implants required"
        ),
        "treatment": (
            "RENAL TREATMENT (same as other Bartter types): "
            "INDOMETHACIN + KCl + NaCl supplementation; "
            "HEARING LOSS TREATMENT (separate, parallel): "
            "HEARING AIDS early — for partial hearing loss; "
            "COCHLEAR IMPLANTS — for severe-profound SNHL; early implantation (before age 2) for language development; "
            "JOINT MANAGEMENT: nephrology + ENT/audiology from birth; "
            "GENETIC COUNSELLING: AR 25% recurrence; prenatal diagnosis available; "
            "MONITORING: renal function + nephrocalcinosis + audiological assessment annually"
        ),
        "critical_flags": [
            "BARTTER-PLUS-SNHL-TYPE-4-ONLY — SNHL is absent in types 1/2/3/5; present in type 4 (BSND) ONLY; any Bartter + SNHL = type 4 until proven otherwise; ABR at diagnosis mandatory",
            "BARTTIN-BETA-SUBUNIT-BOTH-ClC — barttin stabilises BOTH ClC-Ka and ClC-Kb; loss = both channels non-functional; ClC-Ka also expressed in thin ascending limb + stria vascularis",
            "COCHLEAR-IMPLANT-EARLY — SNHL severe-profound from birth; cochlear implant before age 2 critical for language development; do NOT delay for metabolic stabilisation",
            "STRIA-VASCULARIS-ENDOLYMPH — ClC-Ka + barttin in marginal cells secrete Cl into endolymph (low Cl) to drive K recycling → high endolymph K; loss → abnormal endolymph → hair cell death",
            "NEONATAL-SEVERITY-TYPE1-LIKE — Bartter 4 severity similar to types 1/2: polyhydramnios, premature, life-threatening neonatal salt wasting; more severe than type 3",
            "HYPOTHYROIDISM-OVERLAP — some BSND patients reported with associated thyroid dysfunction; screen thyroid function at diagnosis and annually",
            "ABR-NEONATAL-HEARING-SCREEN — ABR (not OAE alone) required; OAE may be normal in some SNHL forms; ABR is the definitive test; perform before discharge from NICU",
            "RENAL-FUNCTION-LONG-TERM — similar CKD risk from nephrocalcinosis as other Bartter types; monitor eGFR annually; renal transplant preserves only renal function, not hearing"
        ],
        "alias": (
            "BSND (Barttin); OMIM gene 606412; "
            "Bartter syndrome type 4 OMIM 602522. "
            "1p32.3; 320 aa; ~35 kDa; beta-subunit protein; autosomal recessive. "
            "FUNCTION: BSND encodes barttin, a small transmembrane protein that acts as the obligatory beta-subunit "
            "for BOTH ClC-Ka (CLCNKA) and ClC-Kb (CLCNKB) chloride channels. "
            "Barttin is required for trafficking of ClC-Ka and ClC-Kb to the plasma membrane and for their proper gating. "
            "Without barttin, both ClC-Ka and ClC-Kb are retained intracellularly and are non-functional. "
            "Tissue expression: "
            "(1) Kidney: basolateral membranes of TAL (ClC-Ka + ClC-Kb) and DCT (ClC-Kb) — salt reabsorption. "
            "(2) Inner ear: marginal cells of the stria vascularis — ClC-Ka + barttin secretes Cl into endolymph; "
            "required for high-K endolymph (endocochlear potential); loss → absent endocochlear potential → SNHL. "
            "CLINICAL PRESENTATION: "
            "Renal: severe neonatal Bartter (similar to types 1/2): polyhydramnios, premature, "
            "hypokalemic metabolic alkalosis, salt wasting, hypercalciuria, nephrocalcinosis. "
            "Hearing: congenital bilateral severe-profound SNHL from absent endocochlear potential. "
            "DIAGNOSIS: "
            "BSND sequencing; ABR confirms SNHL; renal electrolytes. "
            "MANAGEMENT: "
            "Indomethacin + electrolyte replacement; cochlear implant (early); joint nephrology-audiology care."
        ),
    },
    # -- SLC12A3 / NCCT — Gitelman — Most common hereditary salt wasting nephropathy ----
    {
        "gene": "SLC12A3",
        "alt_name": "NCCT / Gitelman",
        "protein": (
            "SLC12A3/NCCT -- 16q13 AR -- NCCT-1021aa -- "
            "Gitelman-Syndrome-MOST-COMMON-Hereditary-Salt-Wasting-1:40000 -- "
            "HYPOMAGNESAEMIA-LOW-URINE-CALCIUM-PATHOGNOMONIC-Distinguishes-from-Bartter -- "
            "Distal-Convoluted-Tubule-Na-Cl-Cotransporter-Thiazide-Target -- "
            "Mild-Presentation-Craving-Salt-Tetany-Chondrocalcinosis"
        ),
        "locus": "16q13",
        "protein_size": "1021 aa",
        "inheritance": "AR",
        "age_of_onset": "Childhood to adult (often mild, late presentation)",
        "key_biomarker": (
            "Serum: hypokalemia + metabolic alkalosis + HYPOMAGNESAEMIA (low Mg) PATHOGNOMONIC; "
            "urine: HYPOCALCIURIA (low Ca:Cr) — PATHOGNOMONIC; elevated urinary Mg: inappropriate; "
            "elevated renin + aldosterone; normal-low BP; "
            "SLC12A3 gene sequencing: >400 mutations; p.Arg913Gln + p.Thr60Met most common European; "
            "ECG: QTc prolongation from hypoMg + hypoK"
        ),
        "pathognomonic": (
            "HYPOMAGNESAEMIA + LOW URINE CALCIUM = PATHOGNOMONIC GITELMAN — absent in all Bartter subtypes; "
            "thiazide diuretic effect: DCT NCCT inhibition → NaCl wasting but Ca reabsorption INCREASES; "
            "Gitelman = 'congenital thiazide effect'; "
            "most common hereditary tubular disorder 1:40,000; "
            "ECG QTc prolongation from combined hypoMg + hypoK"
        ),
        "treatment": (
            "MAGNESIUM SUPPLEMENTATION — ORAL: high-dose Mg oxide/chloride/citrate chronically; "
            "target serum Mg > 0.6 mmol/L; divided doses; GI side effects common; "
            "IV MAGNESIUM for symptomatic hypoMg (tetany, arrhythmia); "
            "POTASSIUM SUPPLEMENTATION: oral KCl chronically; "
            "POTASSIUM-SPARING DIURETICS: amiloride or spironolactone; "
            "INDOMETHACIN: less effective in Gitelman than Bartter; some centres use; "
            "ECG MONITORING: QTc prolongation; avoid QT-prolonging drugs; "
            "SALT INTAKE: encourage liberal salt intake; "
            "PREGNANCY: careful monitoring of Mg + K; may worsen during pregnancy"
        ),
        "critical_flags": [
            "HYPOMAGNESAEMIA-PATHOGNOMONIC-GITELMAN — low serum Mg ALWAYS present; absent in all Bartter types; if hypoK alkalosis WITHOUT hypoMg → Bartter, not Gitelman",
            "LOW-URINE-CALCIUM-DISTINGUISHES — DCT Ca reabsorption INCREASES when Na-Cl transport fails (NaCl depletion → Na/Ca exchanger activated); low urine Ca = Gitelman; hypercalciuria = Bartter",
            "CONGENITAL-THIAZIDE-EFFECT — NCCT is the target of thiazide diuretics; Gitelman = permanent thiazide effect; hypoK + hypoMg + hypocalciuria is thiazide side-effect pattern",
            "CHONDROCALCINOSIS-CHRONIC — chronic hypoMg → Mg pyrophosphate crystal deposition in cartilage; chondrocalcinosis on X-ray in adult Gitelman; joints, menisci, symphysis pubis",
            "QTC-PROLONGATION — combined hypoMg + hypoK = QTc prolongation + torsades risk; ECG at diagnosis; avoid macrolides, fluoroquinolones, antipsychotics (QT-prolonging drugs)",
            "ORAL-MG-ABSORPTION-POOR — GI intolerance limits oral Mg dose; split into multiple small doses; IV Mg for crises; some use amiloride to increase distal Mg reabsorption",
            "MOST-COMMON-TUBULAR-DISORDER — prevalence 1:40,000 (possibly higher; many undiagnosed); common cause of unexplained hypoK + hypoMg in adults; consider before primary aldosteronism workup",
            "PREGNANCY-RISK — Gitelman can worsen in pregnancy (increased renal Mg loss); monitor Mg + K monthly; IV Mg if tetany; liaise with obstetrics"
        ],
        "alias": (
            "SLC12A3 (NCCT — Na-Cl cotransporter); OMIM gene 600968; "
            "Gitelman syndrome OMIM 263800. "
            "16q13; 1021 aa; ~112 kDa; apical membrane of distal convoluted tubule (DCT); autosomal recessive. "
            "FUNCTION: SLC12A3 encodes NCCT (NCC), the apical Na-Cl cotransporter of the DCT. "
            "NCCT electroneutrally co-imports Na and Cl from the tubular lumen. "
            "It is the primary target of thiazide diuretics (hydrochlorothiazide, chlorthalidone). "
            "DCT reabsorbs ~5-8% of filtered NaCl. "
            "When NCCT fails: Na and Cl wasting → secondary hyperaldosteronism → K wasting (hypokalemia). "
            "Additionally: DCT Na depletion activates apical Na/Ca exchange → increased Ca reabsorption → "
            "hypocalciuria (opposite of hypercalciuria in Bartter). "
            "DCT Mg reabsorption via TRPM6 is also impaired (mechanism: NCCT dysfunction disrupts Mg entry) → hypoMg. "
            "CLINICAL PRESENTATION: "
            "Typically mild; often diagnosed in adulthood or incidentally. "
            "Symptoms: fatigue, cramps, tetany, salt craving, polyuria. "
            "Chronic: chondrocalcinosis (Mg pyrophosphate), QTc prolongation, female fertility issues. "
            "DIAGNOSIS: "
            "Serum: hypoK, metabolic alkalosis, hypoMg. Urine: hypocalciuria, elevated K/Mg. "
            "SLC12A3 sequencing: >400 mutations; biallelic pathogenic variants = diagnosis. "
            "MANAGEMENT: "
            "Oral Mg (lifelong); oral KCl; amiloride adjunct. ECG monitoring. "
            "Avoid QT-prolonging drugs. Monitor in pregnancy."
        ),
    },
    # -- CLCN5 / ClC-5 — Dent disease 1 — LMW proteinuria + nephrolithiasis ----------
    {
        "gene": "CLCN5",
        "alt_name": "ClC-5 / Dent-1",
        "protein": (
            "CLCN5/ClC-5 -- Xp11.23 XLR -- ClC-5-746aa -- "
            "Dent-Disease-1-X-Linked-Low-Molecular-Weight-Proteinuria-PATHOGNOMONIC -- "
            "Nephrocalcinosis-Nephrolithiasis-Progressive-CKD -- "
            "Proximal-Tubule-Endosomal-Cl-H-Exchanger -- "
            "Tubular-Proteinuria-Beta-2-Microglobulin-NOT-Albumin-Hallmark"
        ),
        "locus": "Xp11.23",
        "protein_size": "746 aa",
        "inheritance": "XLR",
        "age_of_onset": "Childhood (males); carrier females often asymptomatic",
        "key_biomarker": (
            "LOW-MOLECULAR-WEIGHT (LMW) PROTEINURIA PATHOGNOMONIC: "
            "beta-2-microglobulin, alpha-1-microglobulin, retinol-binding protein elevated in urine; "
            "urine dipstick may be NEGATIVE (albumin not the primary protein lost); "
            "hypercalciuria (>0.25 mg/mg urine Ca:Cr); "
            "nephrocalcinosis + nephrolithiasis on renal ultrasound; "
            "CLCN5 gene sequencing; "
            "Fanconi syndrome features in some: glucosuria, aminoaciduria, phosphaturia"
        ),
        "pathognomonic": (
            "LMW PROTEINURIA + nephrocalcinosis + nephrolithiasis in a male = PATHOGNOMONIC Dent disease; "
            "urine beta-2-microglobulin elevated >1000x normal; "
            "dipstick proteinuria NEGATIVE (not albumin) — misleading; "
            "X-linked: females are carriers (usually asymptomatic or mild LMW proteinuria); "
            "Fanconi syndrome in 30-50% of cases"
        ),
        "treatment": (
            "NO SPECIFIC DISEASE-MODIFYING TREATMENT — supportive; "
            "CALCIUM OXALATE STONE PREVENTION: "
            "high fluid intake; avoid dehydration; "
            "thiazide diuretics: reduce hypercalciuria (paradoxically — thiazide reduces urine Ca); "
            "potassium citrate: alkalinise urine, reduce stone formation; "
            "FANCONI SYNDROME: phosphate + vitamin D supplementation if hypophosphataemic rickets; "
            "AVOID VITAMIN C SUPPLEMENTATION — vitamin C metabolised to oxalate; "
            "CKD MANAGEMENT: standard nephrology; ACE inhibitor for proteinuria control; "
            "RENAL REPLACEMENT: many males reach ESRD 30-50 yr; plan for transplant; "
            "GENETIC COUNSELLING: X-linked; carrier females: 50% sons affected; screen daughters"
        ),
        "critical_flags": [
            "LMW-PROTEINURIA-NEGATIVE-DIPSTICK — beta-2-microglobulin elevated but dipstick NEGATIVE; never dismiss proteinuria as absent based on dipstick alone in suspected Dent; send urine beta-2-MG",
            "X-LINKED-MALES-AFFECTED — females are carriers; usually asymptomatic or mild LMW proteinuria only; full Dent phenotype in males; examine family history for maternal ESRD",
            "HYPERCALCIURIA-STONES-NEPHROCALCINOSIS — urine Ca:Cr > 0.25; renal ultrasound shows medullary nephrocalcinosis; stones from childhood; recurrent stones in young male → Dent",
            "FANCONI-SUBSET — 30-50% of Dent 1: proximal tubular Fanconi: glucosuria + aminoaciduria + phosphaturia + bicarbonaturia + LMW proteinuria; hypophosphataemic rickets in severe cases",
            "CLCN5-ENDOSOMAL-Cl-H-EXCHANGER — ClC-5 in apical early endosomes of proximal tubule; required for megalin/cubilin-mediated endocytosis of LMW proteins; loss → LMW proteins spill into urine",
            "DENT-1-vs-DENT-2 — Dent 1: CLCN5 (XLR); Dent 2: OCRL (XLR; same gene as Lowe syndrome, milder alleles); Dent 2 may have mild intellectual disability absent in Dent 1; gene sequencing distinguishes",
            "ESRD-30-50yr-MALES — most males reach ESRD by age 30-50; plan transplant early; transplanted kidney does not recur Dent disease",
            "THIAZIDE-REDUCES-HYPERCALCIURIA — paradoxical: thiazide reduces urine Ca (DCT Ca reabsorption); reduces stone formation and nephrocalcinosis progression in Dent; use low-dose hydrochlorothiazide"
        ],
        "alias": (
            "CLCN5 (ClC-5 — chloride channel 5); OMIM gene 300008; "
            "Dent disease 1 OMIM 300009. "
            "Xp11.23; 746 aa; ~83 kDa; voltage-gated Cl/H+ exchanger; X-linked recessive. "
            "FUNCTION: CLCN5 encodes ClC-5, a member of the CLC family. "
            "Unlike ClC-Kb/Ka (plasma membrane Cl channels), ClC-5 is a Cl/H+ exchanger in apical endosomes "
            "of the proximal tubule S1/S2 segments. "
            "ClC-5 acidifies endosomes by exchanging luminal Cl for cytoplasmic H+, creating the low pH required "
            "for efficient receptor-mediated endocytosis by megalin and cubilin. "
            "Megalin/cubilin bind LMW proteins, albumin, and lipid carriers in the glomerular filtrate "
            "and endocytose them into proximal tubule cells for degradation. "
            "Loss of ClC-5 → defective endosomal acidification → megalin/cubilin endocytosis fails → "
            "LMW proteins (beta-2-MG, RBP, alpha-1-MG) not reabsorbed → LMW proteinuria. "
            "Additionally: disrupted endocytosis → impaired 25-OH-D3 reabsorption → hypercalciuria. "
            "CLINICAL PRESENTATION (males): "
            "LMW proteinuria (beta-2-MG, alpha-1-MG, RBP); negative urine dipstick; "
            "hypercalciuria; nephrocalcinosis + nephrolithiasis from childhood; "
            "variable Fanconi syndrome; progressive CKD → ESRD by 30-50 years. "
            "DIAGNOSIS: "
            "Urine beta-2-MG + Ca:Cr ratio; renal ultrasound. CLCN5 sequencing. "
            "MANAGEMENT: "
            "Thiazide + potassium citrate for stones. Phosphate/VitD if Fanconi. CKD care. Transplant planning."
        ),
    },
    # -- OCRL / Lowe syndrome — Cataracts + ID + renal tubular acidosis ----------------
    {
        "gene": "OCRL",
        "alt_name": "OCRL1 / Lowe",
        "protein": (
            "OCRL/OCRL1 -- Xq26.1 XLR -- OCRL1-901aa -- "
            "Lowe-Syndrome-Oculocerebrorenal-Syndrome-TRIAD-PATHOGNOMONIC -- "
            "CONGENITAL-CATARACTS-Plus-INTELLECTUAL-DISABILITY-Plus-RENAL-TUBULAR-ACIDOSIS -- "
            "PI45P2-5-Phosphatase-Endosomal-Golgi-Trafficking -- "
            "Dent-Disease-2-Milder-OCRL-Alleles"
        ),
        "locus": "Xq26.1",
        "protein_size": "901 aa",
        "inheritance": "XLR",
        "age_of_onset": "Congenital (cataracts) / neonatal-infancy (renal + neurological)",
        "key_biomarker": (
            "CLINICAL TRIAD: congenital cataracts + intellectual disability + renal tubular dysfunction PATHOGNOMONIC; "
            "dense posterior lenticular cataracts visible at birth on slit lamp; "
            "renal: Fanconi syndrome (LMW proteinuria + aminoaciduria + glucosuria + phosphaturia + RTA); "
            "CSF: aminoaciduria; "
            "serum: hyperchloraemic metabolic acidosis (proximal RTA); hypophosphataemia; "
            "OCRL gene sequencing; "
            "elevated OCRL enzyme substrate in urine (PI(4,5)P2 pathway)"
        ),
        "pathognomonic": (
            "LOWE SYNDROME TRIAD: congenital cataracts + intellectual disability (hypotonia, developmental delay) "
            "+ Fanconi syndrome (proximal renal tubular acidosis) = PATHOGNOMONIC; "
            "dense bilateral posterior cataracts present at birth; "
            "lens extraction in first weeks of life required; "
            "X-linked: females are carriers (carrier females have punctate posterior lens opacities — carrier sign)"
        ),
        "treatment": (
            "CATARACT SURGERY: lens extraction in first weeks of life; optical rehabilitation (contacts/glasses); "
            "RENAL FANCONI: "
            "phosphate supplementation + active vitamin D (1-alpha-hydroxycholecalciferol or calcitriol); "
            "potassium citrate + sodium citrate for RTA + stone prevention; "
            "bicarbonate supplementation for metabolic acidosis; "
            "NEUROLOGICAL: early intervention (physiotherapy, OT, speech); "
            "antiepileptics if seizures; behavioural management; "
            "MONITORING: renal function, serum PO4, urine Ca:Cr, ECG (QTc), ophthalmology annually; "
            "GENETIC COUNSELLING: X-linked; carrier females: posterior lens opacities (50% carrier screen); "
            "prenatal diagnosis available"
        ),
        "critical_flags": [
            "CONGENITAL-CATARACTS-FIRST-SIGN — dense bilateral posterior cataracts present at birth; lens extraction in first weeks mandatory to prevent amblyopia; ophthalmology emergency",
            "LOWE-TRIAD-PATHOGNOMONIC — cataracts + ID + Fanconi RTA = diagnostic; all three must be present for Lowe diagnosis; Dent 2 = milder OCRL alleles without ID/cataracts",
            "CARRIER-FEMALES-LENS-OPACITIES — heterozygous females: punctate posterior lens opacities on slit lamp; carrier screening test for at-risk females; does NOT impair vision",
            "PROXIMAL-RTA-FANCONI — proximal tubular dysfunction: bicarbonaturia + phosphaturia + aminoaciduria + glucosuria + LMW proteinuria; hyperchloraemic metabolic acidosis; hypophosphataemic rickets",
            "OCRL-DENT-2-ALLELIC — milder OCRL mutations = Dent disease 2 (LMW proteinuria + nephrocalcinosis only; no cataracts or ID); Lowe and Dent 2 are allelic disorders; severity reflects residual OCRL activity",
            "PHOSPHATE-RICKETS — hypophosphataemia from renal phosphate wasting → rickets/osteomalacia; phosphate + calcitriol supplementation essential; monitor alkaline phosphatase",
            "BEHAVIOURAL-FEATURES — intellectual disability range mild-severe; stereotypies, self-injurious behaviour, autism-like features common in Lowe; behavioural psychology input important",
            "RENAL-PROGNOSIS — Fanconi syndrome → CKD progression; median ESRD age 30-40 yr in Lowe (faster than Dent 1); renal transplant improves renal function but not neurological/ocular features"
        ],
        "alias": (
            "OCRL (OCRL1 — oculocerebrorenal syndrome of Lowe protein 1); OMIM gene 300535; "
            "Lowe syndrome OMIM 309000; Dent disease 2 OMIM 300555. "
            "Xq26.1; 901 aa; ~105 kDa; phosphatidylinositol 4,5-bisphosphate 5-phosphatase; X-linked recessive. "
            "FUNCTION: OCRL encodes a phosphatidylinositol 4,5-bisphosphate [PI(4,5)P2] 5-phosphatase. "
            "OCRL1 localises to the Golgi apparatus and early/recycling endosomes. "
            "It converts PI(4,5)P2 → PI(4)P, regulating endosomal trafficking and actin cytoskeleton dynamics. "
            "In the proximal tubule: OCRL1 is required for megalin/cubilin-mediated endocytosis (same as ClC-5). "
            "Loss of OCRL1 → PI(4,5)P2 accumulation → disrupted endosomal sorting → Fanconi syndrome. "
            "In the eye: OCRL1 required for lens development → congenital cataracts. "
            "In the nervous system: OCRL1 required for neuronal function → intellectual disability. "
            "MILDER ALLELES: Dent disease 2 — proximal tubular dysfunction without eye/brain involvement "
            "(residual OCRL1 activity sufficient for lens/neuronal development). "
            "CLINICAL PRESENTATION: "
            "Congenital bilateral posterior cataracts (present at birth). "
            "Intellectual disability (hypotonia, delayed milestones, behavioural features). "
            "Fanconi syndrome: LMW proteinuria, RTA, hypophosphataemia, aminoaciduria, glucosuria. "
            "Carrier females: punctate posterior lens opacities. "
            "DIAGNOSIS: "
            "Slit lamp cataracts at birth + Fanconi on urine + OCRL sequencing. "
            "MANAGEMENT: "
            "Urgent cataract extraction; phosphate + VitD; citrate for RTA; early developmental intervention. "
            "Renal transplant for ESRD."
        ),
    },
    # -- AGXT / PH1 — Primary Hyperoxaluria type 1 — Systemic oxalosis ---------------
    {
        "gene": "AGXT",
        "alt_name": "PH1 / Alanine-Glyoxylate Aminotransferase",
        "protein": (
            "AGXT/PH1 -- 2q37.3 AR -- AGXT-392aa -- "
            "Primary-Hyperoxaluria-Type-1-Peroxisomal-Enzyme -- "
            "Systemic-Oxalosis-Nephrocalcinosis-ESRD-Oxalate-Deposits-Bone-Heart-Retina -- "
            "Combined-Liver-Kidney-Transplant-CURATIVE-Only-Disease-Cured-by-OLT -- "
            "Pyridoxine-Responsive-20-30pct-B6-REDUCES-Oxalate"
        ),
        "locus": "2q37.3",
        "protein_size": "392 aa",
        "inheritance": "AR",
        "age_of_onset": "Infancy to adulthood (bimodal: infantile severe + adult recurrent stones)",
        "key_biomarker": (
            "Urine oxalate: markedly elevated (>0.7 mmol/1.73m2/day; normal <0.45); "
            "plasma oxalate elevated (especially with CKD); "
            "urine glycolate elevated (PATHOGNOMONIC for PH1 vs PH2/PH3); "
            "AGXT gene sequencing: p.Gly170Arg most common allele (25-40% PH1); "
            "renal ultrasound + CT: nephrocalcinosis + bilateral calcium oxalate stones; "
            "liver biopsy: AGXT enzyme activity absent or reduced; "
            "pyridoxine trial: urine oxalate reduction >30% = B6-responsive"
        ),
        "pathognomonic": (
            "SYSTEMIC OXALOSIS = PATHOGNOMONIC PH1: oxalate deposition in kidneys → bones → heart → retina → "
            "peripheral vessels (once ESRD and oxalate cannot be excreted); "
            "urine glycolate elevated = PH1 specific (not PH2 which has L-glycerate; not PH3); "
            "nephrocalcinosis in infancy (infantile oxalosis) = most severe; "
            "combined liver-kidney transplant is the ONLY cure — liver provides new AGXT"
        ),
        "treatment": (
            "PYRIDOXINE (VITAMIN B6) — 5-20 mg/kg/day; trial for 3 months; "
            "responsive alleles (p.Gly170Arg, p.Phe152Ile): >30% oxalate reduction; continue lifelong; "
            "NON-RESPONSIVE: proceed to lumasiran or transplant; "
            "LUMASIRAN (ALNYLAM, Oxlumo FDA2020) — RNAi therapy; siRNA targets LDHA (reduces oxalate production); "
            "subcutaneous; FIRST MEDICAL TREATMENT reducing oxalate in non-responsive PH1; "
            "HYDRATION: maintain urine volume >3 L/m2/day; prevents oxalate crystallisation; "
            "CALCIUM-FREE DIET; "
            "DIALYSIS: standard HD inadequate for oxalate clearance; intensive (8 hr/day or PD) if ESRD; "
            "COMBINED LIVER-KIDNEY TRANSPLANT: liver provides new AGXT; kidney replaces failed organ; "
            "LTx alone (not combined): used in non-dialysis-dependent patients with severe PH1; "
            "OXALATE-REMOVAL POST-ESRD: plasma oxalate >100 µmol/L → systemic oxalosis accelerates"
        ),
        "critical_flags": [
            "SYSTEMIC-OXALOSIS-AFTER-ESRD — oxalate deposits in bone (sclerotic lesions), heart (CMP), retina (calcium oxalate crystals), peripheral nerves, blood vessels after ESRD; preventable with early transplant",
            "LUMASIRAN-FDA2020 — Oxlumo (givosiran for AGXT); RNAi-based siRNA targets HAO1 (glycolate oxidase) → reduces glyoxylate → reduces oxalate; SC injection monthly; first non-vitamin drug for PH1",
            "PYRIDOXINE-TRIAL-MANDATORY — B6 responsive alleles: p.Gly170Arg (25-40% of PH1 alleles), p.Phe152Ile, p.Ile244Thr; all PH1 patients trialled regardless of genotype before advancing therapy",
            "COMBINED-LKT-CURATIVE — liver transplant provides functional AGXT; kidney transplant replaces failed organ; LKT is definitive cure; isolated kidney transplant alone = recurrence (oxalosis in new kidney)",
            "URINE-GLYCOLATE-PH1-SPECIFIC — elevated urine glycolate distinguishes PH1 from PH2 (L-glycerate) and PH3 (HOGA); urine organic acid panel distinguishes all three types",
            "INFANTILE-OXALOSIS-FATAL — infantile presentation: ESRD in infancy; systemic oxalosis; fatal without urgent intensive dialysis + LKT; highest mortality PH subtype",
            "STANDARD-HD-INADEQUATE — oxalate clearance on standard 4hr HD insufficient; intensive HD (8 hr/day + high flux membrane) or peritoneal dialysis needed; plasma oxalate <30 µmol/L target before LKT",
            "p.Gly170Arg-MISTARGETING — most common mutation: AGXT mislocalises from peroxisome to mitochondria; pyridoxine corrects mistargeting in this allele; mechanism of B6 responsiveness"
        ],
        "alias": (
            "AGXT (alanine-glyoxylate aminotransferase / AGT); OMIM gene 604285; "
            "Primary hyperoxaluria type 1 (PH1) OMIM 259900. "
            "2q37.3; 392 aa; ~43 kDa; peroxisomal matrix enzyme; PLP-dependent; autosomal recessive. "
            "FUNCTION: AGXT encodes alanine-glyoxylate aminotransferase (AGT), a liver-specific peroxisomal enzyme "
            "that transamines glyoxylate to glycine (using alanine as amino donor). "
            "This reaction is the primary route for glyoxylate detoxification. "
            "Loss of AGT → glyoxylate accumulates → oxidised to oxalate by lactate dehydrogenase (LDHA) → "
            "massive urinary oxalate excretion → calcium oxalate crystallisation. "
            "p.Gly170Arg (most common): AGT mislocalises to mitochondria instead of peroxisomes; "
            "pyridoxine corrects mitochondrial localisation → restored peroxisomal function. "
            "CLINICAL PRESENTATION: "
            "Infantile oxalosis (severe): ESRD before age 5 + systemic oxalosis (neonatal form possible). "
            "Classic: recurrent bilateral calcium oxalate nephrolithiasis from childhood; "
            "nephrocalcinosis → progressive CKD → ESRD (typically 20-40 yr in classic form). "
            "Late-onset: recurrent adult stones; may not diagnose until ESRD. "
            "Post-ESRD systemic oxalosis: bone pain/fractures, cardiac CMP, retinal deposits, peripheral neuropathy. "
            "DIAGNOSIS: "
            "Urine oxalate + glycolate; plasma oxalate; renal imaging. "
            "AGXT sequencing; liver biopsy enzyme activity (gold standard). "
            "Pyridoxine trial. "
            "MANAGEMENT: "
            "B6 (responsive) + lumasiran + aggressive hydration. "
            "Combined liver-kidney transplant for definitive cure. "
            "Intensive dialysis if ESRD while awaiting transplant."
        ),
    },
]


def _make_cohort(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    mean_age = {
        "SLC12A1": 0.05,   # neonatal
        "KCNJ1":   0.05,   # neonatal
        "CLCNKB":  2.5,    # childhood
        "BSND":    0.1,    # neonatal
        "SLC12A3": 14.0,   # adolescent/adult
        "CLCN5":   8.0,    # childhood-adolescent
        "OCRL":    0.2,    # infancy (cataracts at birth)
        "AGXT":    6.0,    # childhood (bimodal)
    }.get(gene_entry["gene"], 5.0)
    sd_age = {
        "SLC12A1": 0.1,
        "KCNJ1":   0.1,
        "CLCNKB":  3.0,
        "BSND":    0.2,
        "SLC12A3": 10.0,
        "CLCN5":   5.0,
        "OCRL":    0.5,
        "AGXT":    8.0,
    }.get(gene_entry["gene"], 3.0)
    ages = [round(rng.gauss(mean_age, sd_age), 2) for _ in range(n)]
    ages = [max(0.0, min(60.0, a)) for a in ages]
    sexes = [rng.choice(["M", "F"]) for _ in range(n)]
    # X-linked: predominantly males affected
    if gene_entry["inheritance"] == "XLR":
        sexes = [rng.choices(["M", "F"], weights=[9, 1])[0] for _ in range(n)]
    if gene_entry["gene"] in ("SLC12A1", "KCNJ1", "BSND"):
        severities = [rng.choices(["moderate", "severe"], weights=[2, 4])[0] for _ in range(n)]
    elif gene_entry["gene"] in ("SLC12A3",):
        severities = [rng.choices(["mild", "moderate", "severe"], weights=[5, 3, 1])[0] for _ in range(n)]
    elif gene_entry["gene"] == "AGXT":
        severities = [rng.choices(["mild", "moderate", "severe"], weights=[2, 3, 3])[0] for _ in range(n)]
    else:
        severities = [rng.choice(["mild", "moderate", "severe"]) for _ in range(n)]
    return [
        {
            "patient_id": f"{gene_entry['gene']}-{i+1:03d}",
            "gene": gene_entry["gene"],
            "age_at_diagnosis_yr": ages[i],
            "sex": sexes[i],
            "severity": severities[i],
            "inheritance": gene_entry["inheritance"],
            "locus": gene_entry["locus"],
        }
        for i in range(n)
    ]


def overview() -> dict:
    all_patients = []
    for idx, g in enumerate(RENAL_TUBULAR_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        all_patients.extend(cohort)

    total = len(all_patients)
    gene_counts = {}
    for p in all_patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1

    age_vals = [p["age_at_diagnosis_yr"] for p in all_patients]
    avg_age = round(sum(age_vals) / len(age_vals), 1)
    severe_count = sum(1 for p in all_patients if p["severity"] == "severe")

    gene_summary = []
    for g in RENAL_TUBULAR_GENES:
        gene_summary.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "n_patients": gene_counts.get(g["gene"], 0),
            "critical_flags": g["critical_flags"],
        })

    return {
        "atlas": "Hereditary-Renal-Tubular-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Renal Tubular Disorders Atlas — "
            "SLC12A1/NKCC2-1099aa-15q21.1-AR-Bartter-Type-1-Neonatal-Polyhydramnios-Salt-Wasting-Indomethacin | "
            "KCNJ1/ROMK-391aa-11q24.3-AR-Bartter-Type-2-Transient-Neonatal-HyperK-Then-HypoK-PATHOGNOMONIC | "
            "CLCNKB/ClC-Kb-687aa-1p36.13-AR-Bartter-Type-3-MOST-COMMON-Classic-Childhood-Complete-Deletion | "
            "BSND/Barttin-320aa-1p32.3-AR-Bartter-Type-4-Plus-SNHL-PATHOGNOMONIC-Cochlear-Implant | "
            "SLC12A3/NCCT-1021aa-16q13-AR-Gitelman-HypoMg-Low-Urine-Ca-PATHOGNOMONIC-Thiazide-Mimic | "
            "CLCN5/ClC-5-746aa-Xp11.23-XLR-Dent-Disease-1-LMW-Proteinuria-PATHOGNOMONIC-Nephrocalcinosis | "
            "OCRL/OCRL1-901aa-Xq26.1-XLR-Lowe-Cataracts-ID-Fanconi-RTA-TRIAD-Cataract-Surgery-Neonatal | "
            "AGXT/PH1-392aa-2q37.3-AR-Primary-Hyperoxaluria-1-Systemic-Oxalosis-Lumasiran-FDA2020-OLT-CURATIVE | "
            "320-Patient-Aggregate-8x40-seeds-1910-1917"
        ),
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(RENAL_TUBULAR_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(RENAL_TUBULAR_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "BARTTER-4-SNHL-ONLY-TYPE-WITH-HEARING-LOSS: BSND (Barttin) = ONLY Bartter type with SNHL; ABR mandatory in all Bartter; cochlear implant early if profound SNHL; types 1/2/3/5 have normal hearing",
            "GITELMAN-HYPOMAGNESAEMIA-DISTINGUISHES: hypoMg + hypocalciuria = Gitelman (SLC12A3); ALL Bartter types have normal-high urine Ca + NO hypoMg; checking Mg + urine Ca separates Gitelman from Bartter",
            "BARTTER-2-TRANSIENT-HYPERK: KCNJ1 (Bartter 2) = transient neonatal hyperkalemia THEN hypokalemia; do NOT supplement K in hyperK phase; unique biphasic K pattern = Bartter 2 until proven otherwise",
            "AGXT-COMBINED-LKT-CURATIVE: PH1 is the ONLY disease CURED by isolated liver transplant (LTx provides AGXT); combined liver-kidney for ESRD patients; isolated kidney transplant recurs — do NOT perform alone",
            "LUMASIRAN-PH1-RNAi: Oxlumo (lumasiran) FDA 2020 — first RNAi drug for inherited metabolic disease for a renal indication; reduces urine oxalate in non-B6-responsive PH1; given before ESRD to slow progression",
            "DENT-LMW-PROTEINURIA-DIPSTICK-NEGATIVE: CLCN5 (Dent 1) + OCRL (Dent 2/Lowe) — LMW proteinuria is beta-2-MG not albumin; urine dipstick may be NEGATIVE; always measure urine beta-2-microglobulin separately",
            "LOWE-CONGENITAL-CATARACTS-URGENT: OCRL — cataracts at birth; lens extraction in first weeks of life prevents visual deprivation amblyopia; ophthalmology emergency; do not delay for metabolic stabilisation",
            "CLCNKB-DELETION-MLPA-MANDATORY: Bartter 3 (CLCNKB) most common mutation is complete gene deletion; Sanger sequencing misses it; always add MLPA or gene dosage analysis to Bartter workup",
            "ALL-BARTTER-INDOMETHACIN: all four Bartter types respond to indomethacin (prostaglandin inhibition); types 1/2 most dramatic response; types 3/4 variable; start early in confirmed Bartter",
            "GITELMAN-MOST-COMMON-TUBULAR: SLC12A3 Gitelman 1:40,000 = most common hereditary tubular disorder; often diagnosed late; young adult with unexplained hypoK + hypoMg + low urine Ca = Gitelman until proven otherwise",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(RENAL_TUBULAR_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(RENAL_TUBULAR_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Renal-Tubular-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "alt_name": g.get("alt_name", ""),
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in RENAL_TUBULAR_GENES
        ],
        "glossary": {
            "Bartter syndrome": "Group of autosomal recessive renal tubular disorders causing salt wasting via loss of thick ascending limb (TAL) or distal tubule transport; types 1-5; all share hypokalemic metabolic alkalosis + elevated renin/aldosterone + normal-low BP",
            "Gitelman syndrome": "Most common hereditary salt-wasting nephropathy (1:40,000); SLC12A3/NCCT deficiency; distal convoluted tubule NaCl transport; hypoK + metabolic alkalosis + hypoMg + hypocalciuria; thiazide diuretic-like effect",
            "Thick ascending limb (TAL)": "Segment of loop of Henle — impermeable to water; NaCl reabsorption by NKCC2 (apical) and ClC-Kb + ClC-Ka (basolateral); creates medullary hyperosmolarity; target of loop diuretics (furosemide)",
            "Distal convoluted tubule (DCT)": "Renal tubule segment beyond loop of Henle; NaCl reabsorption by NCCT (apical — thiazide target); Ca reabsorption by TRPV5; Mg reabsorption by TRPM6; target of thiazide diuretics",
            "Hypoketotic hypoglycemia": "Not relevant here — this is a renal atlas; see FAO atlas",
            "LMW proteinuria": "Low-molecular-weight proteinuria — loss of small proteins (beta-2-microglobulin <12 kDa, retinol-binding protein 21 kDa, alpha-1-microglobulin 31 kDa) filtered by glomerulus but not reabsorbed by damaged proximal tubule; hallmark of Dent disease and Lowe syndrome",
            "Fanconi syndrome": "Generalised proximal tubular dysfunction — combination of LMW proteinuria + aminoaciduria + glucosuria + phosphaturia + bicarbonaturia + uricosuria; causes include Lowe (OCRL), Dent (CLCN5), Wilson, Fanconi anaemia, Lowe; leads to hypophosphataemic rickets if severe",
            "Nephrocalcinosis": "Calcium deposits within renal parenchyma (not collecting system); medullary nephrocalcinosis = most common; causes: hypercalciuria (Bartter, Dent), hyperoxaluria (PH1); detected by ultrasound or CT; risk factor for CKD",
            "Primary hyperoxaluria (PH)": "Inherited defects in glyoxylate metabolism leading to oxalate overproduction; three types: PH1 (AGXT, most common/severe), PH2 (GRHPR), PH3 (HOGA1); oxalate crystallises in kidney then systemically; PH1 treated with lumasiran + pyridoxine + OLT",
            "Systemic oxalosis": "Oxalate deposition in extra-renal tissues after ESRD in PH1; sites: bone (sclerotic lesions, fractures), heart (conduction defects, cardiomyopathy), retina (crystal deposits), peripheral nerves, vessels; prevented by early effective treatment before ESRD",
            "Lumasiran (Oxlumo)": "RNA interference (RNAi) drug; siRNA targeting LDHA mRNA (glycolate oxidase/HAO1 in liver); reduces hepatic oxalate production; subcutaneous monthly; FDA approved 2020 for PH1; first RNAi drug for a metabolic renal disorder",
            "NKCC2 (SLC12A1)": "Na-K-2Cl cotransporter 2; apical membrane of TAL; reabsorbs 1 Na + 1 K + 2 Cl per cycle; responsible for ~25-30% of total renal NaCl reabsorption; molecular target of furosemide/bumetanide; Bartter type 1 gene",
            "NCCT (SLC12A3)": "Na-Cl cotransporter; apical membrane of DCT; reabsorbs 1 Na + 1 Cl; ~5-8% of filtered NaCl; target of thiazide diuretics; Gitelman syndrome gene; also modulates Ca and Mg handling in DCT",
            "Barttin (BSND)": "Small accessory beta-subunit required by BOTH ClC-Ka and ClC-Kb kidney chloride channels; also expressed in stria vascularis of inner ear (marginal cells); loss = Bartter type 4 (renal) + sensorineural hearing loss (cochlear)",
            "OCRL1 / OCRL": "Phosphatidylinositol 4,5-bisphosphate 5-phosphatase; expressed in Golgi + endosomes + proximal tubule; required for megalin/cubilin endocytosis; mutations cause Lowe syndrome (severe) or Dent disease 2 (mild); PI(4,5)P2 → PI(4)P",
            "Megalin (LRP2) / Cubilin": "Multiligand endocytic receptors on proximal tubule apical surface; bind and internalise albumin, LMW proteins, 25-OH-D3, vitamin-B12-IF complex, apoA-I; require ClC-5 and OCRL1 for endosomal acidification; loss = LMW proteinuria",
            "Pyridoxine (B6) responsive PH1": "Approximately 20-30% of PH1 patients; most common responsive allele: p.Gly170Arg (mistargeting mutation); B6 corrects subcellular localisation of AGXT from mitochondria to peroxisomes; urine oxalate drops >30% = responsive; trial ALL PH1 regardless of genotype",
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-RENAL-TUBULAR-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (Gitelman — most common) ===")
    bd = breakdown()
    git = next(g for g in bd["genes"] if g["gene"] == "SLC12A3")
    print(json.dumps(git, indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:5], indent=2))
