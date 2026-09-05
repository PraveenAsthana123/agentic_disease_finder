#!/usr/bin/env python3
"""Renal-Tubular-Disorders-Atlas — Complete 8-Gene Hereditary Renal Tubular Transport Disorders Atlas
SLC12A1 (NKCC2; 1099 aa; 15q21.1; OMIM gene 600839;
          Bartter Syndrome Type 1 (AR) — antenatal polyhydramnios, nephrocalcinosis, HYPERcalciuria;
          loop diuretic molecular target; indomethacin partially effective;
          NKCC2 = Na-K-2Cl cotransporter; thick ascending limb) ·
KCNJ1   (ROMK/Kir1.1; 391 aa; 11q24.3; OMIM gene 600359;
          Bartter Syndrome Type 2 (AR) — neonatal HYPERKALEMIA before aldosterone effect;
          ROMK = apical K⁺ recycling channel; NKCC2 requires ROMK to recycle K⁺;
          Antenatal polyhydramnios; HYPERcalciuria; self-correcting hyperkalemia in infancy) ·
CLCNKB  (ClC-Kb; 687 aa; 1p36.13; OMIM gene 602023;
          Bartter Syndrome Type 3 (AR) — MOST COMMON Bartter; later onset, milder;
          Phenotypic overlap with Gitelman possible (normo- or hypocalciuria some);
          Del/dup with CLCNKA: MLPA MANDATORY — complete gene deletions frequent;
          ClC-Kb = basolateral Cl⁻ channel in TAL and DCT) ·
BSND    (Barttin; 320 aa; 1p32.3; OMIM gene 606412;
          Bartter Syndrome Type 4a (AR) — SENSORINEURAL DEAFNESS PATHOGNOMONIC;
          Barttin = essential β-subunit of both ClC-Ka AND ClC-Kb; cochlear stria vascularis;
          Most severe Bartter; renal failure in 2nd-3rd decade;
          GJB2/GJB6 MANDATORY exclusion FIRST in any deafness + tubulopathy) ·
SLC12A3 (NCC/TSC; 1021 aa; 16q13; OMIM gene 600968;
          Gitelman Syndrome (AR) — HYPOmagnesaemia + HYPOcalciuria (DDx HALLMARK vs Bartter);
          Tetany, chondrocalcinosis, QTc prolongation; late onset vs Bartter;
          NCC = Na-Cl cotransporter; distal convoluted tubule;
          Gitelman = HYPO-calciuria; Bartter = HYPER-calciuria: SINGLE MOST IMPORTANT DDx) ·
SCNN1B  (ENaC-β; 640 aa; 16p12.2; OMIM gene 600760;
          Liddle Syndrome (AD GOF) — early severe HYPERTENSION, LOW aldosterone, LOW renin;
          PHA1B (AR LOF) — pseudohypoaldosteronism type 1B; neonatal salt wasting;
          Liddle: AMILORIDE FIRST (NOT spironolactone — ENaC not aldosterone-dependent);
          ENaC β-subunit PY motif mutation → ENaC stays open → Na retention → volume expansion) ·
CLCN5   (ClC-5; 746 aa; Xp11.23; OMIM gene 300008;
          Dent Disease Type 1 (XLR) — LOW-MOLECULAR-WEIGHT proteinuria MANDATORY feature;
          HYPERcalciuria + nephrocalcinosis + nephrolithiasis + progressive CKD;
          MLPA mandatory — large deletions; X-linked → males severely affected, females mosaic;
          Dent-2 = OCRL gene; Tubular endosomal Cl⁻/H⁺ exchanger; receptor recycling impaired) ·
SLC34A1 (NaPi-IIa; 639 aa; 5q35.3; OMIM gene 182309;
          Hereditary Hypophosphataemic Rickets with Hypercalciuria (HHRH; AR) — FGF23 LOW;
          FGF23 LOW is CRITICAL DDx from X-linked hypophosphataemia (XLH) where FGF23 HIGH;
          Calcitriol first; NO isolated phosphate supplementation (worsens calciuria);
          NaPi-IIa = proximal tubule apical Na-phosphate cotransporter)
320-patient aggregate cohort (8 x 40, seeds 1350-1357)
"""

import random

SEED_BASE = 1350

RENAL_TUBULAR_GENES = [
    # ── SLC12A1 — Bartter Syndrome Type 1 ──
    {
        "gene": "SLC12A1",
        "protein": "NKCC2 (Na-K-2Cl Cotransporter 2)",
        "alias": (
            "SLC12A1; OMIM gene 600839; Bartter Syndrome Type 1 #601678 (AR); "
            "15q21.1; 1099 aa; ~120 kDa; SLC12 family cation-chloride cotransporter; "
            "expressed on apical membrane of thick ascending limb (TAL) of Henle's loop; "
            "NKCC2 transports 1Na⁺:1K⁺:2Cl⁻ into tubular cells (electroneutral); "
            "furosemide/bumetanide act by blocking NKCC2 — Bartter1 patients mimic loop diuretic intoxication; "
            "LOF → failure of NaCl reabsorption in TAL → renal salt wasting + secondary hyperaldosteronism"
        ),
        "aa": "1099 aa",
        "kDa": "~120 kDa",
        "locus": "15q21.1",
        "omim_gene": 600839,
        "omim_disease": 601678,
        "inheritance": (
            "AR biallelic LOF — homozygous or compound heterozygous; "
            "de novo mutations rare; antenatal diagnosis if family history; "
            "consanguinity increases risk; no AD form described"
        ),
        "gene_class": (
            "SLC12A1 encodes NKCC2, the Na-K-2Cl cotransporter expressed exclusively on the apical "
            "membrane of thick ascending limb (TAL) cells and macula densa. "
            "FUNCTION: NKCC2 drives secondary active reabsorption of Na⁺, K⁺, and Cl⁻ (1:1:2) powered "
            "by the basolateral Na⁺/K⁺-ATPase gradient. NKCC2 activity requires KCNJ1 (ROMK) to recycle K⁺ "
            "back to the lumen (K⁺ is rate-limiting substrate). Ca²⁺ reabsorption in TAL is paracellular "
            "and voltage-driven — LOF of NKCC2 abolishes the lumen-positive potential → HYPERcalciuria → nephrocalcinosis. "
            "PATHOPHYSIOLOGY: Biallelic LOF → failure of TAL NaCl reabsorption → renal salt wasting → "
            "volume depletion → RAAS activation (high renin, high aldosterone) → hypokalemia, metabolic alkalosis. "
            "Prostaglandins (PGE2) amplify the tubular effect; indomethacin reduces PGE2 → partial improvement. "
            "ANTENATAL: Polyuria begins in utero (18-24 weeks) → polyhydramnios → prematurity. "
            "POSTNATAL: Failure to thrive, hypercalciuria, nephrocalcinosis, growth retardation."
        ),
        "hallmarks": [
            "Antenatal polyhydramnios (onset 18–24 wk gestation) — universal",
            "Neonatal/infantile salt-wasting crisis — hypokalaemic metabolic alkalosis",
            "HYPERcalciuria → nephrocalcinosis (key DDx from Gitelman HYPO-calciuria)",
            "Renal salt wasting: low-normal BP, high renin, high aldosterone",
            "Normal serum Ca²⁺ (unlike BSND/Bartter4 where Ca can vary)",
            "Prostaglandin-mediated amplification: PGE2 elevated; indomethacin partially effective",
            "Loop diuretic phenocopy: furosemide/bumetanide block NKCC2 (same transporter)",
            "Polyuria and polydipsia — persistent despite treatment",
        ],
        "treatment_alerts": [
            "INDOMETHACIN: First-line adjunct — reduces PGE2 amplification; start after neonatal period",
            "AVOID FRUSEMIDE/BUMETANIDE: Block NKCC2 — worsens Bartter 1 (act on same transporter)",
            "KCl supplementation MANDATORY: hypokalaemia life-threatening (arrhythmia, paralysis)",
            "SPIRONOLACTONE: Adjunct for persistent hyperaldosteronism; not first-line alone",
            "RENIN/ALDOSTERONE monitoring: RAAS activation persists; titrate accordingly",
            "NEPHROCALCINOSIS surveillance: Annual renal USS; avoid hypervitaminosis D",
            "ACE INHIBITOR: NOT first-line in Bartter (RAAS activated but BP usually low-normal)",
            "AMILORIDE: Limited role in Bartter 1 (ENaC downstream; aldosterone-driven component)",
        ],
        "key_ddx": [
            "Gitelman (SLC12A3): HYPOcalciuria + HYPOmagnesaemia — OPPOSITE calciuria",
            "Bartter 2 (KCNJ1): Neonatal HYPERKALEMIA initially — unique to Bartter 2",
            "Bartter 3 (CLCNKB): Later onset, milder, variable calciuria — MLPA required",
            "Bartter 4 (BSND): SENSORINEURAL DEAFNESS + most severe — Barttin beta subunit",
            "Furosemide excess: Check medication history — loop diuretics cause identical biochemistry",
            "Antenatal Bartter: Polyhydramnios DDx includes Bartter 1/2 (NOT Gitelman — mild, later)",
            "ROMK hyperkalemia phase: Neonatal hyperK is Bartter 2 NOT 1 or 3",
        ],
        "clinical_pearls": [
            "NKCC2 = furosemide target → Bartter 1 is the genetic equivalent of chronic furosemide excess",
            "Nephrocalcinosis distinguishes from Gitelman (Gitelman: hypocalciuria → NO nephrocalcinosis)",
            "Antenatal polyhydramnios in otherwise normal fetus: order SLC12A1 + KCNJ1 first",
            "Indomethacin effective in Bartter 1/2 but less so in 3; avoid <6 months (renal immaturity)",
            "Urine Ca:Cr ratio >0.7 (mg/mg) confirms hypercalciuria; spot urine sufficient for monitoring",
        ],
        "seed": 1350,
        "n_patients": 40,
        "etiologies": [
            ("Nonsense/frameshift null (homozygous)", 0.30),
            ("Missense transmembrane domain (compound het)", 0.28),
            ("Splice-site biallelic", 0.20),
            ("Large deletion (MLPA detected)", 0.12),
            ("Missense + null compound het", 0.10),
        ],
        "age_onset_months_range": (0, 3),
        "sex_ratio_M": 0.50,
    },
    # ── KCNJ1 — Bartter Syndrome Type 2 ──
    {
        "gene": "KCNJ1",
        "protein": "ROMK (Kir1.1 / Renal Outer Medullary K⁺ channel)",
        "alias": (
            "KCNJ1; OMIM gene 600359; Bartter Syndrome Type 2 #241200 (AR); "
            "11q24.3; 391 aa; ~45 kDa; Kir1.1 inward-rectifier K⁺ channel; "
            "expressed on apical membrane of TAL and cortical collecting duct; "
            "ROMK recycles K⁺ to lumen to maintain NKCC2 function; "
            "also mediates K⁺ secretion in collecting duct; "
            "LOF → NKCC2 stalls (K⁺ depleted) → antenatal Bartter + paradoxical neonatal HYPERKALEMIA"
        ),
        "aa": "391 aa",
        "kDa": "~45 kDa",
        "locus": "11q24.3",
        "omim_gene": 600359,
        "omim_disease": 241200,
        "inheritance": (
            "AR biallelic LOF — homozygous or compound heterozygous; "
            "heterozygous carriers generally unaffected; "
            "neonatal presentation; consanguinity increases prevalence"
        ),
        "gene_class": (
            "KCNJ1 encodes ROMK (Kir1.1), an inward-rectifier K⁺ channel critical for two distinct functions: "
            "(1) K⁺ RECYCLING IN TAL: NKCC2 imports K⁺ but TAL lumen has little K⁺; ROMK secretes K⁺ back → "
            "maintains K⁺ availability for NKCC2 → LOF stalls NKCC2 → Bartter phenotype. "
            "(2) K⁺ SECRETION IN COLLECTING DUCT: ROMK is the principal K⁺ secretion channel; "
            "LOF → inability to secrete K⁺ in CCD → paradoxical neonatal HYPERKALEMIA despite high aldosterone. "
            "PARADOX: Hyperaldosteronism (from salt wasting) should drive K⁺ secretion, but without ROMK channel, "
            "K⁺ cannot exit → transient neonatal hyperkalemia (resolves over weeks-months as BIG-K channels mature). "
            "POSTNATAL: As maxi-K (BKCa) channels mature, K⁺ secretion recovers → hyperkalemia resolves → "
            "classic hypokalemic alkalosis pattern emerges in infancy/childhood."
        ),
        "hallmarks": [
            "Antenatal polyhydramnios — universal (same as Bartter 1, more severe)",
            "NEONATAL HYPERKALEMIA — PATHOGNOMONIC: only Bartter type 2 (ROMK LOF in CCD)",
            "Hyperkalemia resolves spontaneously over weeks-months (BIG-K channel maturation)",
            "Subsequently develops hypokalemia + metabolic alkalosis (classic Bartter pattern)",
            "HYPERcalciuria + nephrocalcinosis (same as Bartter 1)",
            "High renin + high aldosterone (despite transient hyperkalemia — ALDOSTERONE CANNOT ACT without ROMK)",
            "Salt wasting: volume depletion, poor feeding, FTT",
            "Urine K⁺ paradoxically LOW despite hyperaldosteronism during neonatal period",
        ],
        "treatment_alerts": [
            "NEONATAL HYPERKALEMIA: Do NOT treat aggressively — resolves spontaneously; avoid K⁺-wasting Rx",
            "INDOMETHACIN: Effective adjunct (same PGE2 mechanism as Bartter 1); wait >6 months",
            "KCl SUPPLEMENTATION: Required once hyperkalemia resolves and hypokalemia develops",
            "MONITOR K⁺ TRAJECTORY: Serial K⁺ tracking — transitions from high to low over weeks",
            "AVOID SPIRONOLACTONE NEONATALLY: Worsens hyperkalemia in neonatal period (blocks aldosterone K⁺ action)",
            "NEPHROCALCINOSIS surveillance: Annual renal USS; restrict dietary Ca load is NOT needed",
            "GENETIC COUNSELLING: AR; 25% recurrence risk; antenatal diagnosis available",
        ],
        "key_ddx": [
            "Bartter 1 (SLC12A1): NO neonatal hyperkalemia; HYPERcalciuria present",
            "Pseudohypoaldosteronism type 1 (PHA1): Also neonatal hyperkalemia — but Na wasting is URINARY+SWEAT+STOOL; no alkalosis",
            "Adrenal insufficiency: Neonatal hyperkalemia — check cortisol + 17OHP; no polyhydramnios antenatal",
            "Gitelman: NEVER presents neonatally; HYPOcalciuria; older children/adults",
            "Bartter 4 (BSND): Deafness + most severe; no hyperkalemia phase",
            "Hyperkalemia phase DDx: Bartter 2 is ONLY Bartter type with neonatal hyperkalemia",
        ],
        "clinical_pearls": [
            "ROMK LOF = paradox: high aldosterone + high K⁺ neonatally — ROMK is the K⁺ exit channel",
            "Hyperkalemia resolves spontaneously as maxi-K (BKCa) channels mature — do not over-treat",
            "Distinguish from PHA1: Bartter 2 has ALKALOSIS + polyhydramnios; PHA1 has ACIDOSIS",
            "Antenatal polyhydramnios + neonatal hyperK → order KCNJ1 first, then SLC12A1",
            "Urine K:Cr ratio helps: paradoxically low during hyperkalemia phase (CCD K secretion absent)",
        ],
        "seed": 1351,
        "n_patients": 40,
        "etiologies": [
            ("Missense pore-region (compound het)", 0.32),
            ("Nonsense/frameshift (homozygous)", 0.28),
            ("Splice-site biallelic", 0.22),
            ("Missense + null compound het", 0.12),
            ("Large deletion", 0.06),
        ],
        "age_onset_months_range": (0, 1),
        "sex_ratio_M": 0.50,
    },
    # ── CLCNKB — Bartter Syndrome Type 3 ──
    {
        "gene": "CLCNKB",
        "protein": "ClC-Kb (Voltage-gated chloride channel Kb)",
        "alias": (
            "CLCNKB; OMIM gene 602023; Bartter Syndrome Type 3 #607364 (AR); "
            "1p36.13; 687 aa; ~83 kDa; CLC family voltage-gated Cl⁻ channel; "
            "expressed on basolateral membrane of TAL, DCT, CCD; paired with CLCNKA on chromosome 1p; "
            "requires BSND (Barttin) as β-subunit for trafficking and function; "
            "LOF → impaired Cl⁻ exit → depolarises tubular cell → NKCC2/NCC stall; "
            "MLPA MANDATORY: entire CLCNKB gene deletions are frequent (homozygous del common in African ancestry)"
        ),
        "aa": "687 aa",
        "kDa": "~83 kDa",
        "locus": "1p36.13",
        "omim_gene": 602023,
        "omim_disease": 607364,
        "inheritance": (
            "AR biallelic LOF — homozygous deletion is MOST COMMON variant (especially African ancestry); "
            "compound heterozygous missense variants; "
            "MLPA detects whole-gene deletions (sequencing alone misses these); "
            "heterozygous carriers clinically unaffected"
        ),
        "gene_class": (
            "CLCNKB encodes ClC-Kb, a basolateral voltage-gated Cl⁻ channel expressed in TAL, DCT, and cortical "
            "collecting duct. ClC-Kb functions as a homodimer and requires BSND (Barttin) as β-subunit. "
            "FUNCTION: Cl⁻ imported by NKCC2 (TAL) or NCC (DCT) must exit basolaterally → ClC-Kb provides this "
            "exit route → LOF causes Cl⁻ accumulation → cell depolarisation → NKCC2/NCC stall. "
            "PHENOTYPIC SPECTRUM: Most common Bartter type. Later onset (infancy to childhood vs neonatal in B1/B2). "
            "Milder overall. Some patients have NORMOCALCIURIA or even HYPOCALCIURIA (Gitelman-like DCT phenotype) "
            "due to ClC-Kb's role in DCT as well as TAL. "
            "GENOMICS: CLCNKB and CLCNKA are tandemly arranged 1p36.13 (~11 kb apart); "
            "nonallelic homologous recombination causes whole-gene deletions → MLPA essential. "
            "Biallelic CLCNKA+CLCNKB deletion → severe phenotype + deafness (like BSND). "
            "TREATMENT SIMILAR TO B1: Indomethacin + KCl ± spironolactone ± amiloride."
        ),
        "hallmarks": [
            "MOST COMMON Bartter syndrome type (30-50% of all Bartter)",
            "Later onset than Bartter 1/2 — infancy to childhood; milder",
            "Hypokalaemic metabolic alkalosis + salt wasting",
            "Variable calciuria: HYPERcalciuria (TAL phenotype) OR normocalciuria/HYPOcalciuria (DCT phenotype)",
            "High renin + high aldosterone",
            "Growth retardation, polyuria, polydipsia",
            "WHOLE-GENE DELETIONS COMMON — sequencing alone MISSES these",
            "MLPA MANDATORY for CLCNKB (especially African ancestry where del rates >50%)",
        ],
        "treatment_alerts": [
            "MLPA MANDATORY: Sequencing alone misses whole-gene deletions; report must include MLPA",
            "INDOMETHACIN: Effective but response less predictable than Bartter 1/2",
            "KCl SUPPLEMENTATION: Essential; titrate to maintain K⁺ >3.0 mEq/L",
            "SPIRONOLACTONE: Useful adjunct for persistent hyperaldosteronism",
            "CALCIURIA MONITORING: Clarify if hyper- or hypocalciuria (determines nephrocalcinosis risk)",
            "GITELMAN OVERLAP: If HYPOcalciuria + HYPOmagnesaemia → check SLC12A3 (Gitelman) also",
            "PHENOTYPE MIMICRY: Bartter 3 can phenocopy Gitelman — genotype essential",
        ],
        "key_ddx": [
            "Gitelman (SLC12A3): HYPOcalciuria + HYPOmagnesaemia; later onset; DCT phenotype — CLCNKB can mimic",
            "Bartter 1 (SLC12A1): More severe, antenatal, HYPERcalciuria obligate",
            "Bartter 4 (BSND): DEAFNESS + more severe; no deafness in Bartter 3",
            "Barttin subunit (BSND): Barttin is shared beta-subunit of ClC-Ka and ClC-Kb; loss → both fail → worse",
            "CLCNKA+CLCNKB dual deletion: Deafness phenotype like Bartter 4 (barttin-like); check both",
            "Laxative abuse/surreptitious diuretic: Check urine diuretic screen; no mutation",
        ],
        "clinical_pearls": [
            "Most common Bartter type but most phenotypically variable — Gitelman overlap possible",
            "MLPA is MANDATORY: a negative panel sequencing result does NOT exclude CLCNKB Bartter",
            "African ancestry → whole-gene deletion most common variant (MLPA especially critical)",
            "Calciuria directs management: HYPERcalciuria → annual USS (nephrocalcinosis); HYPOcalciuria → Mg supplementation",
            "Spironolactone ± amiloride combination often more effective than indomethacin alone in Bartter 3",
        ],
        "seed": 1352,
        "n_patients": 40,
        "etiologies": [
            ("Homozygous whole-gene deletion (MLPA)", 0.35),
            ("Compound het missense", 0.28),
            ("Compound het del + missense", 0.18),
            ("Nonsense/frameshift biallelic", 0.12),
            ("Splice-site biallelic", 0.07),
        ],
        "age_onset_months_range": (2, 24),
        "sex_ratio_M": 0.50,
    },
    # ── BSND — Bartter Syndrome Type 4a ──
    {
        "gene": "BSND",
        "protein": "Barttin (β-subunit of ClC-Ka and ClC-Kb)",
        "alias": (
            "BSND; OMIM gene 606412; Bartter Syndrome Type 4a #602522 (AR); "
            "1p32.3; 320 aa; ~36 kDa; two-transmembrane regulatory protein; "
            "β-subunit essential for surface trafficking AND gating of BOTH ClC-Ka AND ClC-Kb; "
            "expressed in kidney TAL/DCT and COCHLEAR stria vascularis (endolymph K⁺ homeostasis); "
            "LOF → failure of BOTH ClC-Ka and ClC-Kb → most severe Bartter + SENSORINEURAL DEAFNESS; "
            "GJB2/GJB6 exclusion MANDATORY FIRST in any SNHL + tubulopathy presentation"
        ),
        "aa": "320 aa",
        "kDa": "~36 kDa",
        "locus": "1p32.3",
        "omim_gene": 606412,
        "omim_disease": 602522,
        "inheritance": (
            "AR biallelic LOF — homozygous or compound heterozygous; "
            "Barttin essential for both kidney and cochlear ClC channel function; "
            "phenotype = Bartter + SNHL (both mandatory); "
            "heterozygous carriers: usually unaffected (may have mild audiometric changes)"
        ),
        "gene_class": (
            "BSND encodes Barttin, a two-transmembrane regulatory protein that is the obligate β-subunit "
            "of both ClC-Ka and ClC-Kb. DUAL ROLE: "
            "(1) KIDNEY: Barttin enables surface expression and gating of ClC-Ka and ClC-Kb on basolateral "
            "membranes of TAL and DCT → LOF abolishes Cl⁻ exit → BOTH TAL and DCT fail → most severe Bartter. "
            "(2) COCHLEA: Stria vascularis K⁺ secretion into endolymph requires ClC-Ka/Kb channels with Barttin; "
            "LOF → failure of endolymph K⁺ maintenance → sensorineural deafness. "
            "PHENOTYPE: Most severe Bartter subtype. Antenatal polyhydramnios. Severe neonatal salt wasting. "
            "Renal failure develops in 2nd-3rd decade. BILATERAL SENSORINEURAL HEARING LOSS (universal). "
            "GENOMIC NOTE: Biallelic deletion of CLCNKA+CLCNKB (adjacent genes at 1p36.13) → Bartter Type 4b "
            "(functionally equivalent to Type 4a — BSND LOF = same as losing both ClC channels)."
        ),
        "hallmarks": [
            "BILATERAL SENSORINEURAL DEAFNESS — PATHOGNOMONIC for Bartter Type 4a",
            "Most severe Bartter: Antenatal polyhydramnios + severe neonatal salt wasting",
            "HYPERcalciuria + nephrocalcinosis universal",
            "Renal failure progression: 2nd-3rd decade (most severe Bartter subtype)",
            "High renin + high aldosterone (extreme)",
            "Failure to thrive — severe from neonatal period",
            "GJB2/GJB6 exclusion MANDATORY FIRST: commonest SNHL genes must be excluded",
            "Audiology + annual renal USS + yearly K⁺/Mg²⁺/Ca²⁺ monitoring mandatory",
        ],
        "treatment_alerts": [
            "GJB2/GJB6 EXCLUSION MANDATORY: Most common causes of SNHL must be excluded before diagnosing BSND",
            "COCHLEAR IMPLANT: May be appropriate as SNHL is sensorineural and progressive; ENT referral mandatory",
            "INDOMETHACIN: Adjunct but less effective in Bartter 4 than in 1/2 (more severe baseline)",
            "KCl SUPPLEMENTATION: Essential; often requires higher doses due to severity",
            "SPIRONOLACTONE ± AMILORIDE: Adjunct; titrate to persistent hyperaldosteronism",
            "RENAL FAILURE SURVEILLANCE: eGFR annually from age 10; prepare renal replacement planning",
            "AVOID NEPHROTOXINS: NSAIDs (including indomethacin — balance benefit/nephrotoxicity at low eGFR)",
            "BLOOD PRESSURE: Often low-normal; DO NOT over-aggressively treat BP (RAAS-driven low baseline)",
        ],
        "key_ddx": [
            "GJB2/GJB6 SNHL: Most common SNHL — must exclude before BSND (no tubulopathy)",
            "CLCNKA+CLCNKB dual deletion (Bartter 4b): Clinically identical to 4a; same management",
            "Bartter 1 (SLC12A1): No deafness; less severe; antenatal",
            "Bartter 3 (CLCNKB): No deafness; milder; later onset",
            "Pendred syndrome (SLC26A4): EVA + SNHL but NO tubulopathy; thyroid goiter",
            "Usher syndrome: SNHL + retinal dystrophy; NO tubulopathy",
        ],
        "clinical_pearls": [
            "Barttin is the BETA-SUBUNIT for BOTH ClC-Ka AND ClC-Kb — loss = double-knockout in kidney and cochlea",
            "SNHL is bilateral, sensorineural, present from birth — audiology in first week of life mandatory",
            "Cochlear implant outcomes can be good — refer ENT early (hearing aid first, CI if needed)",
            "Renal failure trajectory: 50% reach ESRD by age 30 — transplant evaluation early (adolescence)",
            "Antenatal BSND: Polyhydramnios + known family history → fetal genotyping",
        ],
        "seed": 1353,
        "n_patients": 40,
        "etiologies": [
            ("Missense Gly47Arg (founder) compound het", 0.30),
            ("Nonsense/frameshift homozygous", 0.25),
            ("Splice-site biallelic", 0.22),
            ("Missense basolateral targeting domain", 0.13),
            ("Large deletion BSND", 0.10),
        ],
        "age_onset_months_range": (0, 2),
        "sex_ratio_M": 0.50,
    },
    # ── SLC12A3 — Gitelman Syndrome ──
    {
        "gene": "SLC12A3",
        "protein": "NCC (Na-Cl Cotransporter / Thiazide-sensitive cotransporter, TSC)",
        "alias": (
            "SLC12A3; OMIM gene 600968; Gitelman Syndrome #263800 (AR); "
            "16q13; 1021 aa; ~110 kDa; SLC12 family cation-chloride cotransporter; "
            "expressed on apical membrane of distal convoluted tubule (DCT); "
            "NCC transports 1Na⁺:1Cl⁻ (electroneutral); thiazide diuretics block NCC; "
            "LOF → NaCl wasting in DCT → HYPOcalciuria (Ca²⁺ reabsorption increases) + HYPOmagnesaemia; "
            "GITELMAN = HYPOcalciuria + HYPOmag; BARTTER = HYPERcalciuria"
        ),
        "aa": "1021 aa",
        "kDa": "~110 kDa",
        "locus": "16q13",
        "omim_gene": 600968,
        "omim_disease": 263800,
        "inheritance": (
            "AR biallelic LOF — homozygous or compound heterozygous; "
            "MOST COMMON hereditary tubulopathy (1:40,000 general population; carrier rate ~1%); "
            "heterozygous carriers: often asymptomatic but may have borderline low K⁺/Mg²⁺; "
            "compound heterozygous genotype very common — full gene sequencing + MLPA needed"
        ),
        "gene_class": (
            "SLC12A3 encodes NCC (Na-Cl cotransporter, also called TSC for thiazide-sensitive cotransporter), "
            "expressed exclusively on the apical membrane of DCT1 cells. "
            "FUNCTION: NCC mediates electroneutral Na⁺ and Cl⁻ reabsorption (1:1 stoichiometry), driven by the "
            "basolateral Na⁺/K⁺-ATPase. Thiazide diuretics (HCTZ, chlorthalidone) block NCC → Gitelman is "
            "the genetic equivalent of chronic thiazide therapy. "
            "CALCIURIA PARADOX: NCC LOF → DCT Na⁺ depletion → upregulates basolateral Na⁺/Ca²⁺ exchanger NCX1 → "
            "increased Ca²⁺ reabsorption → HYPOcalciuria (no nephrocalcinosis). "
            "MAGNESIUM: DCT is primary site for active Mg²⁺ reabsorption (TRPM6); "
            "NCC LOF disrupts DCT driving force → secondary TRPM6 impairment → HYPOmagnesaemia → tetany. "
            "PHENOTYPE: Mildest hereditary tubulopathy. Onset in childhood/adolescence/young adult. "
            "Cramps, tetany, fatigue, palpitations (QTc prolongation from HYPOmag)."
        ),
        "hallmarks": [
            "HYPOcalciuria — PATHOGNOMONIC DDx from all Bartter subtypes (Bartter = HYPERcalciuria)",
            "HYPOmagnesaemia — universal in Gitelman; cardinal feature",
            "NO nephrocalcinosis (HYPOcalciuria protective)",
            "Late onset: childhood, adolescence, young adult (NOT antenatal/neonatal)",
            "Hypokalaemic metabolic alkalosis (milder than Bartter)",
            "Tetany, muscle cramps, weakness (HYPOmag + HYPOkalemia)",
            "QTc prolongation — HYPOmag + HYPOkalemia — arrhythmia risk",
            "Chondrocalcinosis (Mg²⁺ pyrophosphate deposition) — in adults with chronic HYPOmag",
        ],
        "treatment_alerts": [
            "Mg SUPPLEMENTATION MANDATORY: Oral MgCl₂ or MgSO₄; IV in tetany/paralysis",
            "MAGNESIUM SUPPLEMENTATION DOSE: 10–20 mmol/day elemental Mg oral; goal Mg²⁺ >0.6 mmol/L",
            "KCl SUPPLEMENTATION: Co-administer; HYPOkalemia often responds poorly without Mg correction",
            "SPIRONOLACTONE: Potassium-sparing adjunct for persistent HYPOkalemia",
            "AMILORIDE: Alternative potassium-sparing; often more effective than spironolactone in Gitelman",
            "QTc MONITORING: Baseline ECG; annual ECG; avoid QTc-prolonging drugs",
            "AVOID THIAZIDES ABSOLUTELY: Block NCC — identical to disease mechanism; catastrophic worsening",
            "CHONDROCALCINOSIS: X-ray knees/wrists for Mg²⁺ pyrophosphate if chronic; no specific Rx",
            "INDOMETHACIN: LESS effective in Gitelman than Bartter (PGE2 role less prominent in DCT)",
        ],
        "key_ddx": [
            "Bartter types 1-4: HYPERcalciuria (Bartter) vs HYPOcalciuria (Gitelman) — SINGLE KEY DDx",
            "Bartter 3 (CLCNKB): Can have DCT phenotype mimicking Gitelman — genotype essential",
            "Primary hyperaldosteronism: HYPOkalemia but BP HIGH (Gitelman BP low-normal); no HYPOmag",
            "Laxative/diuretic abuse: Urine diuretic screen; no SLC12A3 mutation",
            "EAST/SeSAME syndrome (KCNJ13): HYPOmag + HYPOkalemia + epilepsy + ataxia + SNHL",
            "Hypomagnesaemia-hypercalciuria (CLDN16/CLDN19): HYPERcalciuria — opposite calciuria",
        ],
        "clinical_pearls": [
            "MOST COMMON hereditary tubulopathy — carrier rate ~1%; suspect in any unexplained HYPOmag+HYPOkalemia",
            "HYPOcalciuria is the diagnostic anchor: spot urine Ca:Cr <0.1 mg/mg distinguishes from Bartter",
            "Treat Mg FIRST before K⁺ — HYPOkalemia often doesn't correct until Mg normalised",
            "Avoid thiazides absolutely — they block NCC (the mutant transporter); worsens all parameters",
            "QTc monitoring: Risk of torsades de pointes with concurrent QTc-prolonging medications",
            "Amiloride often outperforms spironolactone for K⁺ retention in Gitelman",
        ],
        "seed": 1354,
        "n_patients": 40,
        "etiologies": [
            ("Compound het missense (most common)", 0.45),
            ("Homozygous missense founder (Mediterranean)", 0.20),
            ("Nonsense/frameshift compound het", 0.18),
            ("Splice-site biallelic", 0.12),
            ("Large deletion (MLPA)", 0.05),
        ],
        "age_onset_months_range": (24, 240),
        "sex_ratio_M": 0.45,
    },
    # ── SCNN1B — Liddle Syndrome / PHA1B ──
    {
        "gene": "SCNN1B",
        "protein": "ENaC-β (Epithelial Sodium Channel β-subunit)",
        "alias": (
            "SCNN1B; OMIM gene 600760; Liddle Syndrome #177200 (AD GOF) / PHA1B #264350 (AR LOF); "
            "16p12.2; 640 aa; ~72 kDa; degenerin/ENaC superfamily; "
            "ENaC = heterotrimeric channel (α-β-γ-δ) expressed in collecting duct (kidney), lung, colon; "
            "β-subunit PY motif (PPXY) controls channel ubiquitination and retrieval; "
            "GOF (PY motif del/frameshift) → ENaC stays open → Na retention → volume hypertension; "
            "LOF (biallelic) → PHA1B — neonatal salt wasting without adrenal failure"
        ),
        "aa": "640 aa",
        "kDa": "~72 kDa",
        "locus": "16p12.2",
        "omim_gene": 600760,
        "omim_disease_liddle": 177200,
        "omim_disease_pha1b": 264350,
        "omim_disease": 177200,
        "inheritance": (
            "Liddle Syndrome — AD GOF (PY motif truncation/frameshift/missense at C-terminus β-subunit); "
            "virtually 100% penetrance; PY motif del → Nedd4-2 cannot ubiquitinate ENaC → ENaC stays at surface; "
            "PHA1B — AR biallelic LOF of SCNN1A/B/G or NEDD4L; "
            "heterozygous ENaC β carriers: usually subclinical (aldosterone can compensate partially)"
        ),
        "gene_class": (
            "SCNN1B encodes the β-subunit of the Epithelial Sodium Channel (ENaC), a constitutively open "
            "Na⁺ channel expressed on the apical membrane of collecting duct principal cells, distal airway, "
            "and colon. ENaC is the final effector of aldosterone action: aldosterone → SGK1 → phosphorylates "
            "Nedd4-2 → prevents Nedd4-2 from ubiquitinating ENaC → more ENaC at surface → more Na⁺ reabsorption. "
            "LIDDLE MECHANISM: PY motif truncation/mutation in SCNN1B or SCNN1G β/γ C-terminus → "
            "Nedd4-2 binding site destroyed → ENaC constitutively at surface regardless of aldosterone → "
            "chronic Na⁺ retention → hypertension → SUPPRESSED aldosterone + SUPPRESSED renin (volume-expanded). "
            "DIAGNOSIS: LOW ALDOSTERONE + LOW RENIN + HYPOkalemia + HYPERTENSION. "
            "AMILORIDE BLOCKS ENaC directly → treats Liddle. SPIRONOLACTONE FAILS (works via aldosterone). "
            "PHA1B: Opposite — biallelic LOF → ENaC cannot function → neonatal salt wasting + hyperkalemia + acidosis "
            "in kidney + lung + colon simultaneously."
        ),
        "hallmarks": [
            "LIDDLE: Early severe HYPERTENSION (childhood/young adult)",
            "LIDDLE: LOW RENIN + LOW ALDOSTERONE — suppressed RAAS (volume-expanded state)",
            "LIDDLE: HYPOkalemia (ENaC Na⁺ retention → K⁺ secretion via ROMK)",
            "LIDDLE: Metabolic alkalosis",
            "LIDDLE: AMILORIDE CURES (ENaC blocker) — SPIRONOLACTONE FAILS (aldosterone is already LOW)",
            "PHA1B: Neonatal salt wasting + HYPERkalemia + metabolic acidosis",
            "PHA1B: Multi-organ: kidney + lung (respiratory distress) + colon",
            "PHA1B: HIGH renin + HIGH aldosterone (aldosterone cannot act without ENaC)",
        ],
        "treatment_alerts": [
            "LIDDLE: AMILORIDE FIRST — blocks ENaC directly; start at 5-10 mg/day; titrate",
            "LIDDLE: TRIAMTERENE: Alternative ENaC blocker if amiloride unavailable",
            "LIDDLE: SPIRONOLACTONE DOES NOT WORK — ENaC activity is aldosterone-INDEPENDENT in Liddle",
            "LIDDLE: LOW-SODIUM DIET: Adjunct; reduces substrate for ENaC Na⁺ retention",
            "LIDDLE: BP TARGET <130/80 with amiloride; may need adjunct antihypertensive if refractory",
            "LIDDLE: AVOID ACE INHIBITORS/ARBs alone: RAAS already suppressed; little additive benefit",
            "PHA1B: HIGH-DOSE FLUDROCORTISONE: NOT effective (ENaC absent — cannot respond); Na supplementation only",
            "PHA1B: GRADUAL IMPROVEMENT: ENaC expression improves with age in some organs; avoid steroid excess",
        ],
        "key_ddx": [
            "Primary hyperaldosteronism (Conn): HYPOkalemia + hypertension but HIGH aldosterone (opposite RAAS)",
            "Apparent mineralocorticoid excess (HSD11B2): Similar low-aldosterone HTN — 11β-HSD2 deficiency; liquorice",
            "Glucocorticoid-remediable aldosteronism (KCNJ5): HIGH aldosterone + suppressed renin (not low aldosterone)",
            "Renovascular hypertension: HIGH renin (Liddle = LOW renin)",
            "Cushing syndrome: HIGH cortisol + HTN; RAAS varies; no ENaC genotype",
            "Gordon syndrome (WNK4/WNK1/KLHL3/CUL3): HYPERkalemia + HTN + acidosis (Liddle: HYPOkalemia)",
            "PHA1B vs PHA1A: PHA1A = MC2R mutation (adrenal only); PHA1B = multi-organ (kidney + lung + colon)",
        ],
        "clinical_pearls": [
            "LIDDLE = LOW renin + LOW aldosterone + HYPOkalemia + HTN: do NOT give spironolactone first",
            "Amiloride is the diagnostic-therapeutic test: if BP normalises and K⁺ rises → Liddle confirmed",
            "PY motif C-terminal truncations in SCNN1B: exon 13 frameshift hotspot — look here first",
            "PHA1B: Lung involvement (respiratory distress) distinguishes from PHA1A (renal only); CXR for pulmonary oedema",
            "Young onset severe HTN with low-renin low-aldosterone → family history → genetic panel essential",
        ],
        "seed": 1355,
        "n_patients": 40,
        "etiologies": [
            ("Liddle PY-motif frameshift/truncation SCNN1B C-terminus", 0.50),
            ("Liddle missense at PY motif Tyr residue", 0.20),
            ("PHA1B biallelic null SCNN1B", 0.15),
            ("Liddle SCNN1G PY-motif (sister subunit)", 0.10),
            ("Liddle missense outside PY motif", 0.05),
        ],
        "age_onset_months_range": (0, 360),
        "sex_ratio_M": 0.50,
    },
    # ── CLCN5 — Dent Disease Type 1 ──
    {
        "gene": "CLCN5",
        "protein": "ClC-5 (Voltage-gated chloride/proton exchanger 5)",
        "alias": (
            "CLCN5; OMIM gene 300008; Dent Disease Type 1 #300009 (XLR); "
            "Xp11.23; 746 aa; ~83 kDa; CLC family 2Cl⁻/H⁺ exchanger (antiporter); "
            "expressed on early endosomal membranes of proximal tubular cells; "
            "ClC-5 acidifies endosomes → receptor-ligand dissociation → receptor recycling; "
            "LOF → megalin/cubilin receptor recycling impaired → low-molecular-weight proteinuria + absorptive hypercalciuria; "
            "MLPA MANDATORY — large deletions in ~15% of Dent 1; "
            "Dent 2 = OCRL gene (Lowe syndrome variant, AR/XLR)"
        ),
        "aa": "746 aa",
        "kDa": "~83 kDa",
        "locus": "Xp11.23",
        "omim_gene": 300008,
        "omim_disease": 300009,
        "inheritance": (
            "X-linked recessive (XLR) — males severely affected; "
            "female carriers: usually mosaic phenotype (mild LMW proteinuria, rarely nephrolithiasis); "
            "de novo mutations ~20% (no family history); "
            "MLPA mandatory for large deletions; full coding sequence sequencing required for point mutations"
        ),
        "gene_class": (
            "CLCN5 encodes ClC-5, a 2Cl⁻/H⁺ antiporter expressed on early endosomal membranes in proximal "
            "tubular (PT) cells, intercalated cells of collecting duct, and intestinal cells. "
            "FUNCTION IN PROXIMAL TUBULE: PT cells reabsorb filtered proteins (albumin, retinol-binding protein, "
            "β2-microglobulin, α1-microglobulin) via receptor-mediated endocytosis using megalin and cubilin. "
            "ClC-5 acidifies early endosomes (2Cl⁻ in, 1H⁺ out) → acidification enables receptor-ligand "
            "dissociation → ligand delivered to lysosome, receptor recycled to surface. "
            "ClC-5 LOF → endosomal acidification fails → megalin/cubilin cannot recycle → "
            "LMW proteinuria (diagnostic hallmark) + absorptive hypercalciuria (impaired Ca²⁺ 1α-hydroxylase recycling). "
            "PROGRESSIVE CKD: PT cell vacuolation, lysosomal storage → tubulointerstitial nephritis → CKD stage 3+ by 4th-5th decade. "
            "DENT 1 vs DENT 2: Dent 2 (OCRL) additionally has intellectual disability, cataracts (Lowe syndrome overlap)."
        ),
        "hallmarks": [
            "LOW-MOLECULAR-WEIGHT (LMW) PROTEINURIA — MANDATORY FEATURE (β2-microglobulin, RBP elevated)",
            "HYPERcalciuria — universal in males",
            "Nephrocalcinosis and/or nephrolithiasis (calcium oxalate/phosphate stones)",
            "Progressive CKD — 30-80% of males reach CKD3+ by 5th decade",
            "X-linked: Males severely affected; female carriers mosaic (mild LMW proteinuria)",
            "Phosphaturia ± glucosuria ± aminoaciduria (Fanconi-like in severe cases)",
            "Normal or mildly elevated serum Ca²⁺",
            "MLPA mandatory: large deletions in ~15%; sequencing alone insufficient",
        ],
        "treatment_alerts": [
            "MLPA MANDATORY: 15% large deletions; report must include MLPA alongside sequencing",
            "NO DISEASE-MODIFYING THERAPY: Symptomatic management; no curative treatment",
            "THIAZIDE DIURETICS: May reduce urinary Ca²⁺ excretion → reduce stone/nephrocalcinosis risk",
            "CITRATE SUPPLEMENTATION: Alkalinises urine → reduces stone formation",
            "AVOID VITAMIN D EXCESS: Absorptive hypercalciuria worsened by vitamin D supplementation",
            "MONITOR eGFR ANNUALLY: CKD progression is gradual but cumulative; avoid nephrotoxins",
            "AVOID HIGH-DOSE CALCIUM SUPPLEMENT: Dietary Ca moderate; avoid supplements worsening calciuria",
            "DENT 2 EXCLUSION (OCRL): If intellectual disability or cataracts → check OCRL gene",
        ],
        "key_ddx": [
            "Dent Disease Type 2 (OCRL): Identical renal phenotype + intellectual disability + cataracts (Lowe overlap)",
            "Lowe Syndrome (OCRL full): LMW proteinuria + cataracts + intellectual disability — OCRL LOF",
            "Primary hyperoxaluria: Nephrocalcinosis + stones but stone type = oxalate; AGXT/GRHPR/HOGA1",
            "Idiopathic hypercalciuria: HYPERcalciuria without LMW proteinuria; no CLCN5 mutation",
            "Fanconi syndrome (SLC3A1 cystinuria): Cystinuria panel; aminoaciduria pattern different",
            "Proximal RTA (SLC4A4, CA2): LMW proteinuria absent; acidosis more prominent",
        ],
        "clinical_pearls": [
            "LMW proteinuria (β2-MG, RBP, α1-MG) is the MANDATORY diagnostic feature — standard urine dipstick misses it",
            "Order β2-microglobulin or α1-microglobulin: first-line LMW proteinuria screen",
            "X-linked: all sons of carrier mothers at 50% risk; daughters of affected males = obligate carriers",
            "MLPA essential: request CLCN5 sequencing + MLPA as single panel order",
            "Thiazide + citrate combination best symptomatic approach for hypercalciuria/nephrolithiasis",
            "CKD trajectory: plan renal replacement discussion at CKD3 (typically 4th-5th decade)",
        ],
        "seed": 1356,
        "n_patients": 40,
        "etiologies": [
            ("Missense (functionally null) hemizygous", 0.40),
            ("Nonsense/frameshift hemizygous", 0.25),
            ("Large deletion Xp11.23 (MLPA)", 0.15),
            ("Splice-site hemizygous", 0.12),
            ("In-frame deletion/insertion", 0.08),
        ],
        "age_onset_months_range": (12, 300),
        "sex_ratio_M": 0.85,
    },
    # ── SLC34A1 — HHRH ──
    {
        "gene": "SLC34A1",
        "protein": "NaPi-IIa (Type IIa sodium-phosphate cotransporter)",
        "alias": (
            "SLC34A1; OMIM gene 182309; Hereditary Hypophosphataemic Rickets with Hypercalciuria (HHRH) #241530 (AR); "
            "5q35.3; 639 aa; ~75 kDa; SLC34 family Na-Pi cotransporter; "
            "expressed on apical membrane of proximal tubular cells; "
            "NaPi-IIa reabsorbs ~70% of filtered phosphate; LOF → phosphate wasting → low Pi → PTH suppressed → "
            "low FGF23 (CRITICAL DDx from XLH where FGF23 HIGH) → calcitriol rises → hypercalciuria; "
            "Calcitriol + phosphate supplements (NOT phosphate alone); "
            "FGF23 LOW is the KEY biomarker distinguishing HHRH from XLH"
        ),
        "aa": "639 aa",
        "kDa": "~75 kDa",
        "locus": "5q35.3",
        "omim_gene": 182309,
        "omim_disease": 241530,
        "inheritance": (
            "AR biallelic LOF — homozygous or compound heterozygous; "
            "heterozygous carriers: hypercalciuria + renal stones (variable expression); "
            "de novo: rare; consanguinity increases risk; "
            "HHRH is rare (~100 reported pedigrees)"
        ),
        "gene_class": (
            "SLC34A1 encodes NaPi-IIa, the primary apical Na⁺-coupled phosphate cotransporter on proximal "
            "tubular cells. NaPi-IIa transports 3Na⁺ per HPO₄²⁻ (electrogenic, Na-driven). "
            "FUNCTION: Reabsorbs ~70% of filtered phosphate; PTH and FGF23 downregulate NaPi-IIa surface "
            "expression → phosphaturia. "
            "HHRH PATHOPHYSIOLOGY: NaPi-IIa LOF → isolated renal phosphate wasting → hypophosphataemia → "
            "SECONDARY: Low Pi → suppresses PTH → PTH-independent calcitriol synthesis increases (1α-OHase "
            "no longer inhibited by PTH) → elevated calcitriol → enhanced intestinal Ca²⁺ and Pi absorption → "
            "HYPERCALCIURIA (secondary) → nephrolithiasis/nephrocalcinosis. FGF23 is LOW or LOW-NORMAL "
            "(low Pi suppresses FGF23) — CRITICAL DDx FROM XLH (where FGF23 is HIGH). "
            "TREATMENT: Calcitriol raises calciuria (but necessary for rickets); phosphate supplements essential; "
            "isolated phosphate (WITHOUT calcitriol) → worsens secondary hyperparathyroidism → more calciuria."
        ),
        "hallmarks": [
            "Rickets/osteomalacia — hypophosphataemic rachitic phenotype",
            "HYPERcalciuria — calcitriol-mediated (secondary to low Pi)",
            "Nephrolithiasis and/or nephrocalcinosis",
            "FGF23 LOW or LOW-NORMAL — CRITICAL DDx biomarker from XLH (FGF23 HIGH)",
            "Calcitriol (1,25-OH₂D) ELEVATED — upregulated due to low Pi suppressing PTH",
            "PTH normal or LOW (suppressed by hypercalciuria/high calcitriol)",
            "Bone pain, growth retardation, genu varum/valgum",
            "Heterozygous carriers: Hypercalciuria + renal stones (incomplete expression)",
        ],
        "treatment_alerts": [
            "PHOSPHATE SUPPLEMENTATION MANDATORY: Neutral phosphate or phosphate buffer (30-70 mg/kg/day elemental Pi)",
            "CALCITRIOL ADD-ON: Required to prevent secondary hyperparathyroidism from Pi loading",
            "FGF23 MEASUREMENT MANDATORY: Low FGF23 confirms HHRH vs XLH (HIGH FGF23)",
            "AVOID ISOLATED PHOSPHATE: Phosphate alone (without calcitriol) → secondary hyperPTH → worsens calciuria",
            "THIAZIDE DIURETICS: Reduce urinary Ca²⁺ if nephrocalcinosis/nephrolithiasis prominent",
            "BUROSUMAB (anti-FGF23): NOT indicated in HHRH (FGF23 already LOW — burosumab worsens calciuria)",
            "MONITOR CALCIURIA: UCa:UCr ratio 3-monthly on treatment — calcitriol worsens hypercalciuria",
            "ANNUAL RENAL USS: Nephrocalcinosis/stone surveillance",
            "XLH DRUG AVOIDANCE: Burosumab is XLH-specific (PHEX gene, HIGH FGF23); contraindicated in HHRH",
        ],
        "key_ddx": [
            "X-linked Hypophosphataemia (XLH, PHEX): FGF23 HIGH (opposite); burosumab works; AD/XLD",
            "Autosomal dominant hypophosphataemic rickets (FGF23 gene): FGF23 HIGH — gain of function",
            "Tumour-induced osteomalacia: FGF23 HIGH; acquired (tumour secreting FGF23)",
            "Hereditary hypophosphataemia, renal phosphate wasting (SLC34A3/NaPi-IIc): Same calciuriapattern; genotype differentiates",
            "Vitamin D deficiency rickets: 25-OHD LOW; FGF23 normal; responds to D supplementation",
            "Fanconi syndrome: Multiple proximal tubule defects; phosphaturia + glucosuria + aminoaciduria + RTA",
        ],
        "clinical_pearls": [
            "FGF23 is the MOST IMPORTANT biomarker: HHRH = LOW FGF23; XLH = HIGH FGF23",
            "If FGF23 is LOW and Pi is low → HHRH; if FGF23 HIGH → XLH/FGF23-GOF — opposite treatments",
            "Burosumab (anti-FGF23) treats XLH but CONTRAINDICATED in HHRH — FGF23 already suppressed",
            "Calcitriol is key: elevated calcitriol explains the hypercalciuria; monitor UCa:Cr on therapy",
            "Thiazide adjunct helps if hypercalciuria causes nephrocalcinosis: HCTZ 12.5-25 mg/day",
            "Heterozygous SLC34A1 carriers: May have hypercalciuric nephrolithiasis — genetic testing family",
        ],
        "seed": 1357,
        "n_patients": 40,
        "etiologies": [
            ("Homozygous missense (Bedouin/Mediterranean founder)", 0.35),
            ("Compound het missense", 0.30),
            ("Nonsense/frameshift biallelic", 0.18),
            ("Splice-site biallelic", 0.12),
            ("Large deletion (MLPA)", 0.05),
        ],
        "age_onset_months_range": (6, 120),
        "sex_ratio_M": 0.50,
    },
]


def _build_cohort(gene_def: dict) -> list:
    """Generate synthetic patient cohort for a gene."""
    rng = random.Random(gene_def["seed"])
    patients = []
    etiologies = gene_def["etiologies"]
    gene = gene_def["gene"]
    ages_months = gene_def.get("age_onset_months_range", (0, 240))
    sex_ratio_m = gene_def.get("sex_ratio_M", 0.50)

    for i in range(gene_def["n_patients"]):
        # Pick etiology
        r = rng.random()
        cumulative = 0
        etiology = etiologies[-1][0]
        for name, prob in etiologies:
            cumulative += prob
            if r <= cumulative:
                etiology = name
                break

        age_onset = rng.randint(ages_months[0], max(ages_months[0] + 1, ages_months[1]))
        sex = "M" if rng.random() < sex_ratio_m else "F"

        # Clinical features
        has_polyhydramnios = gene in ("SLC12A1", "KCNJ1", "BSND") and rng.random() < 0.90
        has_snhl = gene == "BSND" and rng.random() < 0.98
        has_nephrocalcinosis = gene in ("SLC12A1", "KCNJ1", "CLCNKB", "BSND", "CLCN5", "SLC34A1") and rng.random() < 0.70
        has_hypomagnesemia = gene in ("SLC12A3", "CLCNKB") and rng.random() < 0.85
        has_hypertension = gene == "SCNN1B" and rng.random() < 0.88
        has_lmw_proteinuria = gene == "CLCN5" and rng.random() < 0.98
        has_rickets = gene == "SLC34A1" and rng.random() < 0.80
        has_hyperkalemia_neonatal = gene == "KCNJ1" and rng.random() < 0.75
        has_tetany = gene == "SLC12A3" and rng.random() < 0.55
        has_nephrolithiasis = gene in ("CLCN5", "SLC34A1") and rng.random() < 0.65

        k_mmol = rng.uniform(2.0, 3.2) if gene != "KCNJ1" or age_onset > 3 else rng.uniform(5.5, 7.5)
        bicarb = rng.uniform(28, 38)
        mg_mmol = rng.uniform(0.35, 0.65) if has_hypomagnesemia else rng.uniform(0.75, 1.10)
        pi_mmol = rng.uniform(0.55, 0.85) if gene == "SLC34A1" else rng.uniform(1.0, 1.5)
        qtc_ms = rng.randint(445, 490) if has_hypomagnesemia else rng.randint(390, 440)

        indomethacin_used = gene in ("SLC12A1", "KCNJ1", "CLCNKB", "BSND") and rng.random() < 0.75
        amiloride_used = gene == "SCNN1B" and rng.random() < 0.85
        spiro_used = gene in ("SLC12A3", "CLCNKB") and rng.random() < 0.50
        thiazide_used = gene in ("CLCN5", "SLC34A1") and rng.random() < 0.45
        mg_supp = gene == "SLC12A3" and rng.random() < 0.95
        phosphate_supp = gene == "SLC34A1" and rng.random() < 0.90
        calcitriol_used = gene == "SLC34A1" and rng.random() < 0.85

        fgf23_status = "LOW" if gene == "SLC34A1" else ("HIGH" if rng.random() < 0.05 else "NORMAL")
        renin_status = "HIGH" if gene in ("SLC12A1", "KCNJ1", "CLCNKB", "BSND", "SLC12A3") else ("LOW" if gene == "SCNN1B" else "NORMAL")
        aldosterone_status = "HIGH" if gene in ("SLC12A1", "KCNJ1", "CLCNKB", "BSND", "SLC12A3") else ("LOW" if gene == "SCNN1B" else "NORMAL")
        calciuria = (
            "HYPERcalciuria" if gene in ("SLC12A1", "KCNJ1", "BSND", "CLCN5", "SLC34A1")
            else "HYPOcalciuria" if gene == "SLC12A3"
            else ("HYPERcalciuria" if rng.random() < 0.55 else "NORMALcalciuria") if gene == "CLCNKB"
            else "NORMAL"
        )

        patients.append({
            "patient_id": f"{gene}-P{i+1:03d}",
            "gene": gene,
            "etiology": etiology,
            "age_onset_months": age_onset,
            "sex": sex,
            "k_serum_mmol": round(k_mmol, 2),
            "bicarb_mmol": round(bicarb, 1),
            "mg_serum_mmol": round(mg_mmol, 2),
            "pi_serum_mmol": round(pi_mmol, 2),
            "qtc_ms": qtc_ms,
            "calciuria": calciuria,
            "renin": renin_status,
            "aldosterone": aldosterone_status,
            "fgf23_status": fgf23_status,
            "has_polyhydramnios": has_polyhydramnios,
            "has_snhl": has_snhl,
            "has_nephrocalcinosis": has_nephrocalcinosis,
            "has_hypomagnesemia": has_hypomagnesemia,
            "has_hypertension": has_hypertension,
            "has_lmw_proteinuria": has_lmw_proteinuria,
            "has_rickets": has_rickets,
            "has_hyperkalemia_neonatal": has_hyperkalemia_neonatal,
            "has_tetany": has_tetany,
            "has_nephrolithiasis": has_nephrolithiasis,
            "indomethacin_used": indomethacin_used,
            "amiloride_used": amiloride_used,
            "spiro_used": spiro_used,
            "thiazide_used": thiazide_used,
            "mg_supplementation": mg_supp,
            "phosphate_supplementation": phosphate_supp,
            "calcitriol_used": calcitriol_used,
        })
    return patients


def get_overview():
    """Atlas overview: gene list, aggregate stats, key DDx anchors."""
    genes_summary = []
    total_patients = 0
    total_antenatal = 0
    total_snhl = 0
    total_nephrocalcinosis = 0
    total_hypertension = 0
    total_lmw_proteinuria = 0
    total_rickets = 0
    total_hypomagnesemia = 0
    total_nephro_or_stone = 0

    for gd in RENAL_TUBULAR_GENES:
        cohort = _build_cohort(gd)
        n = len(cohort)
        total_patients += n
        antenatal = sum(1 for p in cohort if p["has_polyhydramnios"])
        snhl = sum(1 for p in cohort if p["has_snhl"])
        nc = sum(1 for p in cohort if p["has_nephrocalcinosis"])
        htn = sum(1 for p in cohort if p["has_hypertension"])
        lmw = sum(1 for p in cohort if p["has_lmw_proteinuria"])
        rick = sum(1 for p in cohort if p["has_rickets"])
        hypoMg = sum(1 for p in cohort if p["has_hypomagnesemia"])
        nephstone = sum(1 for p in cohort if p["has_nephrocalcinosis"] or p["has_nephrolithiasis"])

        total_antenatal += antenatal
        total_snhl += snhl
        total_nephrocalcinosis += nc
        total_hypertension += htn
        total_lmw_proteinuria += lmw
        total_rickets += rick
        total_hypomagnesemia += hypoMg
        total_nephro_or_stone += nephstone

        avg_k = round(sum(p["k_serum_mmol"] for p in cohort) / n, 2)
        avg_qtc = round(sum(p["qtc_ms"] for p in cohort) / n, 0)
        calciuria_hyper = sum(1 for p in cohort if p["calciuria"] == "HYPERcalciuria")
        calciuria_hypo = sum(1 for p in cohort if p["calciuria"] == "HYPOcalciuria")

        genes_summary.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "n_patients": n,
            "antenatal_pct": round(100 * antenatal / n, 1),
            "snhl_pct": round(100 * snhl / n, 1),
            "nephrocalcinosis_pct": round(100 * nc / n, 1),
            "hypertension_pct": round(100 * htn / n, 1),
            "lmw_proteinuria_pct": round(100 * lmw / n, 1),
            "rickets_pct": round(100 * rick / n, 1),
            "hypomagnesemia_pct": round(100 * hypoMg / n, 1),
            "nephro_or_stone_pct": round(100 * nephstone / n, 1),
            "avg_k_mmol": avg_k,
            "avg_qtc_ms": int(avg_qtc),
            "calciuria_hyper_pct": round(100 * calciuria_hyper / n, 1),
            "calciuria_hypo_pct": round(100 * calciuria_hypo / n, 1),
            "hallmarks": gd["hallmarks"][:4],
            "top_treatment_alert": gd["treatment_alerts"][0],
        })

    return {
        "atlas": "Renal-Tubular-Disorders-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Renal Tubular Transport Disorders",
        "genes": [g["gene"] for g in RENAL_TUBULAR_GENES],
        "total_patients": total_patients,
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_stats": {
            "antenatal_polyhydramnios_pct": round(100 * total_antenatal / total_patients, 1),
            "snhl_pct": round(100 * total_snhl / total_patients, 1),
            "nephrocalcinosis_or_stone_pct": round(100 * total_nephro_or_stone / total_patients, 1),
            "hypertension_pct": round(100 * total_hypertension / total_patients, 1),
            "lmw_proteinuria_pct": round(100 * total_lmw_proteinuria / total_patients, 1),
            "rickets_pct": round(100 * total_rickets / total_patients, 1),
            "hypomagnesemia_pct": round(100 * total_hypomagnesemia / total_patients, 1),
        },
        "genes_summary": genes_summary,
        "key_ddx_anchor": [
            "Calciuria DDx: HYPERcalciuria (Bartter1/2/4, Dent, HHRH) vs HYPOcalciuria (Gitelman, CLCNKB-DCT) — SINGLE MOST IMPORTANT DDx",
            "SNHL PATHOGNOMONIC: Bartter 4 (BSND) only — exclude GJB2/GJB6 first",
            "Neonatal HYPERkalemia: Bartter Type 2 (KCNJ1) ONLY — resolves spontaneously as maxi-K matures",
            "Liddle: LOW renin + LOW aldosterone + HYPOkalemia + HTN — amiloride not spironolactone",
            "Dent Disease: LMW proteinuria MANDATORY feature — standard dipstick misses it; XLR males",
            "FGF23 biomarker: HHRH = FGF23 LOW; XLH = FGF23 HIGH — opposite treatments (burosumab CI in HHRH)",
            "MLPA MANDATORY: CLCNKB (Bartter3) + CLCN5 (Dent) — whole-gene deletions missed by sequencing alone",
        ],
    }


def get_breakdown():
    """Per-gene detailed breakdown with cohort data."""
    result = []
    for gd in RENAL_TUBULAR_GENES:
        cohort = _build_cohort(gd)
        n = len(cohort)

        # Etiology counts
        etiol_counts = {}
        for p in cohort:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        # Sex breakdown
        males = sum(1 for p in cohort if p["sex"] == "M")

        # Feature rates
        feature_rates = {
            "antenatal_polyhydramnios_pct": round(100 * sum(1 for p in cohort if p["has_polyhydramnios"]) / n, 1),
            "snhl_pct": round(100 * sum(1 for p in cohort if p["has_snhl"]) / n, 1),
            "nephrocalcinosis_pct": round(100 * sum(1 for p in cohort if p["has_nephrocalcinosis"]) / n, 1),
            "hypertension_pct": round(100 * sum(1 for p in cohort if p["has_hypertension"]) / n, 1),
            "lmw_proteinuria_pct": round(100 * sum(1 for p in cohort if p["has_lmw_proteinuria"]) / n, 1),
            "rickets_pct": round(100 * sum(1 for p in cohort if p["has_rickets"]) / n, 1),
            "hypomagnesemia_pct": round(100 * sum(1 for p in cohort if p["has_hypomagnesemia"]) / n, 1),
            "tetany_pct": round(100 * sum(1 for p in cohort if p["has_tetany"]) / n, 1),
            "nephrolithiasis_pct": round(100 * sum(1 for p in cohort if p["has_nephrolithiasis"]) / n, 1),
            "hyperkalemia_neonatal_pct": round(100 * sum(1 for p in cohort if p["has_hyperkalemia_neonatal"]) / n, 1),
        }

        # Treatment rates
        tx_rates = {
            "indomethacin_pct": round(100 * sum(1 for p in cohort if p["indomethacin_used"]) / n, 1),
            "amiloride_pct": round(100 * sum(1 for p in cohort if p["amiloride_used"]) / n, 1),
            "spiro_pct": round(100 * sum(1 for p in cohort if p["spiro_used"]) / n, 1),
            "thiazide_pct": round(100 * sum(1 for p in cohort if p["thiazide_used"]) / n, 1),
            "mg_supplementation_pct": round(100 * sum(1 for p in cohort if p["mg_supplementation"]) / n, 1),
            "phosphate_supplementation_pct": round(100 * sum(1 for p in cohort if p["phosphate_supplementation"]) / n, 1),
            "calcitriol_pct": round(100 * sum(1 for p in cohort if p["calcitriol_used"]) / n, 1),
        }

        # Calciuria distribution
        calciuria_dist = {}
        for p in cohort:
            calciuria_dist[p["calciuria"]] = calciuria_dist.get(p["calciuria"], 0) + 1

        # Etiology distribution as list
        etiol_list = sorted(
            [{"etiology": k, "n": v, "pct": round(100 * v / n, 1)} for k, v in etiol_counts.items()],
            key=lambda x: -x["n"]
        )

        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gene_class": gd["gene_class"],
            "hallmarks": gd["hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "key_ddx": gd["key_ddx"],
            "clinical_pearls": gd["clinical_pearls"],
            "n_patients": n,
            "males_pct": round(100 * males / n, 1),
            "feature_rates": feature_rates,
            "treatment_rates": tx_rates,
            "etiology_distribution": etiol_list,
            "calciuria_distribution": {
                k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in calciuria_dist.items()
            },
            "avg_k_serum": round(sum(p["k_serum_mmol"] for p in cohort) / n, 2),
            "avg_bicarb": round(sum(p["bicarb_mmol"] for p in cohort) / n, 1),
            "avg_mg_serum": round(sum(p["mg_serum_mmol"] for p in cohort) / n, 2),
            "avg_qtc_ms": int(round(sum(p["qtc_ms"] for p in cohort) / n)),
        })
    return result


def get_definitions():
    """Clinical definitions and pharmacological distinctions for the atlas."""
    return {
        "atlas": "Renal-Tubular-Disorders-Atlas",
        "definitions": [
            {
                "term": "Renal Tubular Disorders (RTD)",
                "definition": (
                    "A group of inherited disorders affecting specific ion transport proteins "
                    "in the renal tubule, causing electrolyte imbalances (K⁺, Na⁺, Mg²⁺, Ca²⁺, PO₄) "
                    "without glomerular pathology. RTDs are classified by the tubular segment affected: "
                    "TAL (Bartter 1/2/3/4), DCT (Gitelman, Liddle/PHA1), Proximal tubule (Dent, HHRH)."
                ),
            },
            {
                "term": "Calciuria DDx (CRITICAL)",
                "definition": (
                    "HYPERcalciuria = Bartter types 1, 2, 4 (TAL Ca²⁺ reabsorption abolished by lumen-positive "
                    "potential loss) + Dent disease + HHRH. "
                    "HYPOcalciuria = Gitelman (NCC LOF → DCT Na depletion → upregulates NCX1 → more Ca²⁺ reabsorption). "
                    "CLCNKB (Bartter 3) can be either HYPERcalciuria (TAL phenotype) or normocalciuria/HYPOcalciuria "
                    "(DCT phenotype) — VARIABLE. "
                    "Calciuria DDx is the SINGLE MOST IMPORTANT bedside tool: spot urine Ca:Cr >0.7 mg/mg = "
                    "HYPERcalciuria (Bartter/Dent/HHRH), <0.1 mg/mg = HYPOcalciuria (Gitelman)."
                ),
            },
            {
                "term": "NKCC2 (SLC12A1) — Bartter Type 1",
                "definition": (
                    "Na-K-2Cl cotransporter of the thick ascending limb apical membrane. "
                    "Transports 1Na⁺:1K⁺:2Cl⁻ (electroneutral). Furosemide/bumetanide target = loop diuretic. "
                    "LOF = Bartter 1: antenatal polyhydramnios, severe salt-wasting, HYPERcalciuria, nephrocalcinosis. "
                    "Requires KCNJ1 (ROMK) for K⁺ recycling."
                ),
            },
            {
                "term": "ROMK Paradox — Bartter Type 2 (KCNJ1)",
                "definition": (
                    "ROMK (Kir1.1) has TWO roles: (1) K⁺ recycling in TAL (K⁺ exit = rate-limiting for NKCC2); "
                    "(2) K⁺ secretion in collecting duct. "
                    "LOF → NKCC2 stalls (Bartter phenotype) PLUS K⁺ cannot exit CCD → NEONATAL HYPERKALEMIA. "
                    "Paradox: RAAS activated (high aldosterone) but K⁺ rises — because ROMK is the K⁺ exit channel. "
                    "Hyperkalemia resolves spontaneously as BIG-K (maxi-K) channels mature (weeks-months). "
                    "MANAGEMENT: Do NOT aggressively treat neonatal hyperK — it resolves; monitor trajectory."
                ),
            },
            {
                "term": "CLCNKB / Bartter Type 3 — Most Common",
                "definition": (
                    "ClC-Kb = basolateral Cl⁻ channel in TAL and DCT. "
                    "Most common Bartter type. Later onset, milder, phenotypically variable. "
                    "KEY: MLPA MANDATORY — whole-gene deletion most common variant (especially African ancestry). "
                    "TAL-predominant phenotype: HYPERcalciuria + nephrocalcinosis. "
                    "DCT-predominant phenotype: normocalciuria or HYPOcalciuria (Gitelman-like). "
                    "Genotype essential: cannot distinguish Bartter 3 from Gitelman clinically without SLC12A3/CLCNKB sequencing."
                ),
            },
            {
                "term": "Barttin (BSND) — Bartter Type 4a: Deafness is PATHOGNOMONIC",
                "definition": (
                    "Barttin = obligate β-subunit for BOTH ClC-Ka AND ClC-Kb (dual role). "
                    "Kidney: TAL and DCT Cl⁻ exit abolished → most severe Bartter. "
                    "Cochlea: Stria vascularis K⁺ secretion into endolymph requires ClC-Ka/Kb+Barttin → "
                    "LOF → SENSORINEURAL DEAFNESS (bilateral, severe, from birth). "
                    "MANAGEMENT: GJB2/GJB6 exclusion first. Cochlear implant often appropriate. "
                    "Renal failure in 2nd-3rd decade: early transplant planning essential."
                ),
            },
            {
                "term": "Gitelman Syndrome (SLC12A3) — HYPOcalciuria + HYPOmagnesaemia",
                "definition": (
                    "NCC (Na-Cl cotransporter, DCT) LOF = genetic equivalent of chronic thiazide therapy. "
                    "MUST KNOW: HYPOcalciuria (Ca²⁺ reabsorption increases via NCX1 upregulation) — "
                    "opposite of all Bartter types. NO nephrocalcinosis. "
                    "HYPOmagnesaemia (DCT is primary Mg²⁺ reabsorption site via TRPM6 — DCT failure → Mg wasting). "
                    "TETANY from HYPOmag. QTc prolongation (HYPOmag + HYPOkalemia). "
                    "TREATMENT: Mg first (cannot correct K⁺ without Mg). Amiloride often outperforms spironolactone. "
                    "AVOID THIAZIDES ABSOLUTELY (block NCC — same as disease mechanism)."
                ),
            },
            {
                "term": "Liddle Syndrome (SCNN1B GOF) — Amiloride not Spironolactone",
                "definition": (
                    "ENaC-β PY motif truncation/mutation → Nedd4-2 cannot ubiquitinate ENaC → "
                    "ENaC constitutively at surface → volume expansion → SUPPRESSED RAAS. "
                    "DIAGNOSIS: HYPOkalemia + hypertension + LOW renin + LOW aldosterone. "
                    "TREATMENT: AMILORIDE (blocks ENaC directly) — SPIRONOLACTONE FAILS "
                    "(aldosterone is already suppressed; spironolactone blocks aldosterone receptor that is already idle). "
                    "PITFALL: Spironolactone is reflex first-choice for HYPOkalemia + HTN but WRONG in Liddle."
                ),
            },
            {
                "term": "Dent Disease Type 1 (CLCN5) — LMW Proteinuria MANDATORY",
                "definition": (
                    "ClC-5 (endosomal 2Cl⁻/H⁺ exchanger) LOF → endosomal acidification fails → "
                    "megalin/cubilin receptor recycling impaired → LMW proteinuria (β2-MG, RBP, α1-MG). "
                    "LMW proteinuria is MANDATORY — standard urine dipstick misses it (tests albumin only). "
                    "HYPERcalciuria → nephrocalcinosis + nephrolithiasis. Progressive CKD in males (XLR). "
                    "MLPA MANDATORY (15% large deletions). "
                    "Dent 2 = OCRL: same renal phenotype + intellectual disability + cataracts."
                ),
            },
            {
                "term": "HHRH (SLC34A1) — FGF23 LOW (DDx from XLH where FGF23 HIGH)",
                "definition": (
                    "NaPi-IIa (proximal tubule phosphate cotransporter) LOF → phosphate wasting → "
                    "low Pi → PTH suppressed → 1α-OHase upregulated → calcitriol elevated → "
                    "HYPERcalciuria (secondary) → nephrocalcinosis + nephrolithiasis. "
                    "FGF23 is LOW or LOW-NORMAL (low Pi suppresses FGF23). "
                    "XLH (PHEX): FGF23 HIGH — burosumab (anti-FGF23) works for XLH. "
                    "HHRH: Burosumab ABSOLUTELY CONTRAINDICATED — FGF23 already suppressed; "
                    "burosumab would lower FGF23 further → worsens hypercalciuria. "
                    "TREATMENT: Phosphate supplementation + calcitriol (NOT phosphate alone)."
                ),
            },
            {
                "term": "MLPA — Mandatory for CLCNKB and CLCN5",
                "definition": (
                    "Multiplex Ligation-dependent Probe Amplification (MLPA) detects whole-gene copy number changes. "
                    "CLCNKB (Bartter 3): Homozygous whole-gene deletion is MOST COMMON pathogenic variant "
                    "(especially African ancestry >50%); standard sequencing completely misses this. "
                    "CLCN5 (Dent 1): ~15% of pathogenic alleles are large deletions. "
                    "CLINICAL RULE: Never report CLCNKB or CLCN5 negative without MLPA. "
                    "Combined CLCNKA+CLCNKB deletion → Bartter 4b (deafness phenotype identical to BSND)."
                ),
            },
            {
                "term": "PHA1B (SCNN1B LOF) — Multi-organ vs PHA1A (Adrenal only)",
                "definition": (
                    "PHA1B: Biallelic LOF of SCNN1A/B/G (ENaC subunits) or NEDD4L → "
                    "ENaC cannot function in kidney + LUNG + colon → multi-organ salt wasting. "
                    "Lung involvement: respiratory distress (pulmonary oedema from impaired lung Na clearing). "
                    "PHA1A: MC2R (ACTH receptor) LOF → isolated adrenal hypoaldosteronism — kidney only. "
                    "DDx: PHA1B = respiratory involvement; PHA1A = renal only. "
                    "HIGH renin + HIGH aldosterone (aldosterone cannot act without ENaC in PHA1B)."
                ),
            },
            {
                "term": "Calcitriol (1,25-OH₂D) Paradox in HHRH",
                "definition": (
                    "In HHRH, low phosphate suppresses PTH → removes PTH-mediated inhibition of renal 1α-OHase → "
                    "calcitriol rises → enhances intestinal Ca²⁺ and Pi absorption → HYPERcalciuria. "
                    "Adding phosphate alone WORSENS secondary hyperPTH (Pi loads → PTH rises → phosphaturic cascade). "
                    "CORRECT: Phosphate + calcitriol co-administration prevents secondary hyperPTH. "
                    "Monitor UCa:Cr quarterly on treatment — calcitriol can worsen hypercalciuria."
                ),
            },
            {
                "term": "Cascade Genetic Testing — Renal Tubular Disorders Panel",
                "definition": (
                    "For all index cases: offer SLC12A1, KCNJ1, CLCNKB (+ MLPA), BSND, SLC12A3, SCNN1B, CLCN5 (+ MLPA), SLC34A1 panel. "
                    "First-degree relatives at 25% risk (AR disorders) or 50% risk (Liddle — AD). "
                    "Carrier testing important for CLCN5 (XLR carriers have mild phenotype: calciuria + stones). "
                    "Prenatal diagnosis: available for families with known pathogenic variant; CVS or amniocentesis."
                ),
            },
        ],
        "pharmacological_distinctions": [
            "BARTTER 1/2/3/4: Loop-diuretic equivalent — AVOID furosemide/bumetanide (same transporter as Bartter 1/2)",
            "BARTTER 1/2 (antenatal/neonatal): Indomethacin reduces PGE2 → partial improvement; KCl essential",
            "GITELMAN: Mg FIRST before K⁺ (K⁺ will not correct until Mg normalised); AVOID thiazides absolutely",
            "GITELMAN: Amiloride often outperforms spironolactone for K⁺ retention in DCT",
            "LIDDLE: AMILORIDE cures (ENaC direct block); SPIRONOLACTONE FAILS (aldosterone already suppressed)",
            "DENT: No disease-modifying Rx; thiazide + citrate for hypercalciuria/stone prevention",
            "HHRH: FGF23 LOW → burosumab CONTRAINDICATED (XLH-specific); phosphate + calcitriol combination mandatory",
            "BSND (Bartter 4): Plan cochlear implant early + renal replacement therapy planning in adolescence",
            "CLCNKB (Bartter 3): MLPA mandatory before calling panel negative; Gitelman phenocopy possible",
        ],
    }
