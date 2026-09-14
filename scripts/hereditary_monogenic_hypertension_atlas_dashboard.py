#!/usr/bin/env python3
"""Hereditary-Monogenic-Hypertension-Atlas — Complete 8-Gene Monogenic Hypertension Atlas
(SCNN1B · SCNN1G · HSD11B2 · NR3C2 · WNK4 · WNK1 · KLHL3 · CUL3).

SCNN1B  (ENaC-β subunit; 669 aa; 79 kDa; 16p12.2; AD;
         Liddle syndrome — PY-motif truncation → ENaC gain of function → Na⁺ avidity;
         hypokalemic hypertension with LOW renin, LOW aldosterone;
         AMILORIDE/TRIAMTERENE (direct ENaC blockers) — SPIRONOLACTONE IS INEFFECTIVE
         (MR-independent mechanism); seed SEED_BASE+0).
SCNN1G  (ENaC-γ subunit; 649 aa; 76 kDa; 16p12.2; AD;
         Liddle syndrome (same PY-motif mechanism, same phenotype as SCNN1B);
         test BOTH subunits when Liddle suspected;
         seed SEED_BASE+1).
HSD11B2 (11β-hydroxysteroid dehydrogenase type 2; 405 aa; 44 kDa; 16q22.1; AR;
         Apparent Mineralocorticoid Excess (AME) — cortisol acts as MR agonist
         (normally blocked by 11β-HSD2) → severe hypertension, hypokalemia,
         LOW renin/aldosterone; DEXAMETHASONE suppresses cortisol + spironolactone;
         liquorice inhibits 11β-HSD2 (acquired AME); seed SEED_BASE+2).
NR3C2   (Mineralocorticoid receptor; 984 aa; 107 kDa; 4q31.23; AD GOF;
         Geller syndrome — DRAMATIC pregnancy exacerbation
         (progesterone activates MR-GOF mutant → worse hypertension in pregnancy);
         avoid progesterone-containing OCP;
         MR antagonists avoided in pregnancy; prompt delivery resolves acute crisis;
         seed SEED_BASE+3).
WNK4    (WNK kinase 4; 1243 aa; 135 kDa; 17q21.2; AD;
         Gordon syndrome / PHAII-B — LOF loses NCC suppression → NCC hyperactivated
         → NaCl reabsorption → hyperkalemia + hypertension;
         THIAZIDES DIAGNOSTIC RESPONSE (dramatic within days — thiazide blocks NCC directly);
         seed SEED_BASE+4).
WNK1    (WNK kinase 1; 2382 aa; 256 kDa; 12p13.33; AD;
         Gordon syndrome / PHAII-A — INTRONIC LARGE DELETION (non-coding);
         STANDARD EXOME SEQUENCING MISSES — requires dedicated WNK1 MLPA/CNV analysis;
         thiazides also curative; seed SEED_BASE+5).
KLHL3   (Kelch-like protein 3; 587 aa; 67 kDa; 5q31.2; AR biallelic / AD dominant-negative;
         Gordon syndrome / PHAII-D — CRL3-KLHL3 E3 ubiquitin ligase adaptor;
         LOF → WNK1/4 not ubiquitinated → accumulate → NCC hyperactivation;
         AR = more severe; seed SEED_BASE+6).
CUL3    (Cullin-3; 768 aa; 89 kDa; 2q36.2; AD;
         Gordon syndrome / PHAII-E — de novo EXON 9 SKIP variant (most common);
         neonatal/early-childhood severe presentation;
         most severe Gordon syndrome subtype; high rate of de novo variants;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2534-2541).
"""

import random

SEED_BASE = 2534

HYPERTENSION_GENES = [
    # -- SCNN1B -- Liddle syndrome / ENaC-β -------------------------------------------------------
    {
        "gene": "SCNN1B",
        "alt_name": (
            "SCNN1B (SCNN1B-669aa-16p12.2 / AD -- "
            "LIDDLE-SYNDROME-ENaC-BETA-PY-MOTIF-TRUNCATION-GOF -- "
            "HYPOKALEMIC-HYPERTENSION-LOW-RENIN-LOW-ALDOSTERONE-PATHOGNOMONIC -- "
            "AMILORIDE-TRIAMTERENE-DIRECT-ENaC-BLOCKERS-CURATIVE -- "
            "SPIRONOLACTONE-INEFFECTIVE-MR-INDEPENDENT-MECHANISM)"
        ),
        "protein": (
            "SCNN1B -- 16p12.2 AD -- SCNN1B-669aa -- "
            "ENaC-Beta-Subunit-79kDa-Epithelial-Sodium-Channel-Beta -- "
            "PPXY-PY-Motif-C-Terminal-Intracellular-Domain-Nedd4-2-Ubiquitin-Ligase-Binding-Site -- "
            "PY-Motif-Truncation-or-Missense-Abolishes-Nedd4-2-Binding-ENaC-Not-Internalised -- "
            "ENaC-Accumulates-Apical-Membrane-Constitutive-Na+-Reabsorption-CCD-Collecting-Duct -- "
            "Liddle-1994-Original-Family-SCNN1B-R566X-Truncation-Frameshift-Missense -- "
            "SCNN1G-Same-Locus-Same-Mechanism-Test-Both-Subunits -- "
            "OMIM-Gene-600760-Disease-Liddle-Syndrome-177200"
        ),
        "locus": "16p12.2",
        "protein_size": "669 aa / 79 kDa",
        "inheritance": (
            "AD (autosomal dominant, gain-of-function); "
            "Liddle syndrome — PY motif truncation or missense in C-terminal intracellular domain; "
            "PY motif (PPxY) is the binding site for Nedd4-2 ubiquitin ligase; "
            "LOF of Nedd4-2 binding → ENaC not ubiquitinated → not internalised → accumulates at apical membrane → "
            "constitutive Na+ reabsorption in cortical collecting duct → volume expansion → "
            "suppressed renin → suppressed aldosterone → hypokalemic hypertension; "
            "LOW RENIN + LOW ALDOSTERONE + HYPOKALEMIA + HYPERTENSION = PATHOGNOMONIC for Liddle/AME; "
            "TREATMENT: amiloride or triamterene (direct ENaC channel blockers — MR-independent); "
            "SPIRONOLACTONE IS INEFFECTIVE (ENaC activation is not aldosterone-driven — blocking MR does nothing); "
            "also test SCNN1G (ENaC-γ, same locus, same phenotype — test both subunits)"
        ),
        "disease_category": (
            "Liddle Syndrome — ENaC-β gain-of-function; "
            "hypokalemic hypertension + LOW RENIN + LOW ALDOSTERONE PATHOGNOMONIC; "
            "amiloride/triamterene curative — spironolactone ineffective; "
            "AD; onset young adulthood; SCNN1G identical phenotype"
        ),
        "disease_pathway": (
            "SCNN1B encodes the β subunit of ENaC (Epithelial Sodium Channel), a heterotrimeric channel "
            "(α/β/γ subunits) expressed at the apical membrane of the cortical collecting duct (CCD), "
            "distal nephron, lung, and colon. ENaC mediates rate-limiting Na+ reabsorption driven by the "
            "electrochemical gradient maintained by basolateral Na+/K+-ATPase. "
            "Normally ENaC expression at the apical membrane is tightly regulated by aldosterone "
            "(via SGK1-mediated Nedd4-2 phosphorylation) and by ubiquitin-mediated internalisation: "
            "Nedd4-2 (a HECT-domain E3 ubiquitin ligase) binds the PY motif (PPxY sequence) in the "
            "C-terminal intracellular tail of the β and γ subunits, ubiquitinates ENaC, and triggers "
            "endocytosis — limiting Na+ reabsorption. "
            "Liddle syndrome (SCNN1B PY-motif truncation/missense): Nedd4-2 cannot bind the mutated PY motif → "
            "ENaC not ubiquitinated → not internalised → massive accumulation of ENaC at apical membrane → "
            "constitutive, aldosterone-independent Na+ reabsorption → volume expansion → suppressed renin "
            "(→ low angiotensin II) → suppressed aldosterone (→ aldosterone synthase downregulated). "
            "Result: hypertension + hypokalemia + LOW RENIN + LOW ALDOSTERONE — the hallmark biochemical pattern. "
            "Because this is MR-independent (aldosterone is low — it is not driving ENaC), "
            "spironolactone (MR antagonist) is completely ineffective. "
            "Amiloride and triamterene block ENaC directly in the channel pore — curative regardless of aldosterone."
        ),
        "pathognomonic": (
            "HYPERTENSION + HYPOKALEMIA + LOW RENIN + LOW ALDOSTERONE = LIDDLE OR AME PATHOGNOMONIC; "
            "distinguishes from primary hyperaldosteronism (HIGH aldosterone) and renovascular HTN (HIGH renin); "
            "AMILORIDE RESPONSE confirms Liddle (dramatic BP and K normalisation); "
            "SPIRONOLACTONE FAILURE confirms MR-independent mechanism; "
            "family history: AD — test first-degree relatives (50% risk)"
        ),
        "hormone_profile": (
            "Plasma renin activity (PRA): suppressed (<0.5 ng/mL/h); "
            "plasma aldosterone: suppressed (<5 ng/dL); "
            "aldosterone-renin ratio (ARR): low (not elevated — differs from primary aldosteronism); "
            "serum potassium: low (2.5-3.5 mmol/L); "
            "urine aldosterone: low; cortisol-to-cortisone ratio: normal (differs from AME)"
        ),
        "severity_sds": "Moderate-severe hypertension; hypokalemia can cause arrhythmia; onset young adult/teen",
        "treatment": (
            "AMILORIDE: 5-20 mg/day first-line (direct ENaC blocker — curative regardless of aldosterone); "
            "TRIAMTERENE: alternative ENaC blocker; "
            "SPIRONOLACTONE: DO NOT USE — completely ineffective (MR-independent mechanism); "
            "low-sodium diet potentiates response; "
            "potassium supplementation initially to correct hypokalemia; "
            "cascade testing all first-degree relatives (AD — 50% risk)"
        ),
        "key_features": [
            "Low renin + low aldosterone + hypokalemia + HTN = PATHOGNOMONIC for Liddle/AME",
            "AMILORIDE or TRIAMTERENE curative (direct ENaC blockers)",
            "SPIRONOLACTONE IS INEFFECTIVE — MR-independent mechanism (critical prescribing pitfall)",
            "PY-motif truncation/missense in SCNN1B C-terminal tail — abolishes Nedd4-2 binding",
            "ENaC accumulates at apical membrane — constitutive aldosterone-independent Na+ reabsorption",
            "Test SCNN1G as well (same locus, same phenotype — both subunits must be sequenced)",
            "AD — first-degree relatives at 50% risk; family cascade testing mandatory",
        ],
        "key_ddx": [
            "Primary hyperaldosteronism (HIGH aldosterone — opposite; ARR elevated)",
            "AME/HSD11B2 (also low renin + low aldosterone; high urinary cortisol/cortisone ratio distinguishes)",
            "NR3C2 GOF Geller syndrome (also low renin/aldosterone; pregnancy exacerbation key DDx)",
            "Renovascular hypertension (HIGH renin — opposite pattern)",
        ],
        "onset_age": "Young adult (teens to 30s); rarely presents in childhood",
        "hypertension_type": "ENaC gain-of-function (Liddle syndrome)",
        "nbs_indicated": False,
        "severity": "moderate-severe",
    },

    # -- SCNN1G -- Liddle syndrome / ENaC-γ -------------------------------------------------------
    {
        "gene": "SCNN1G",
        "alt_name": (
            "SCNN1G (SCNN1G-649aa-16p12.2 / AD -- "
            "LIDDLE-SYNDROME-ENaC-GAMMA-PY-MOTIF-TRUNCATION-SAME-MECHANISM-SCNN1B -- "
            "HYPOKALEMIC-HYPERTENSION-LOW-RENIN-LOW-ALDOSTERONE-PATHOGNOMONIC -- "
            "TEST-BOTH-SCNN1B-AND-SCNN1G-WHEN-LIDDLE-SUSPECTED -- "
            "AMILORIDE-TRIAMTERENE-CURATIVE-SPIRONOLACTONE-INEFFECTIVE)"
        ),
        "protein": (
            "SCNN1G -- 16p12.2 AD -- SCNN1G-649aa -- "
            "ENaC-Gamma-Subunit-76kDa-Epithelial-Sodium-Channel-Gamma -- "
            "PPXY-PY-Motif-C-Terminal-Tail-Nedd4-2-Binding-Same-Mechanism-SCNN1B -- "
            "PY-Motif-Truncation-ENaC-Not-Internalised-Same-Constitutive-Na+-Reabsorption -- "
            "SCNN1B-SCNN1G-Same-Chromosomal-Region-16p12.2-Adjacent-Genes -- "
            "Liddle-Original-Family-Also-Had-SCNN1B-Mutation-SCNN1G-Same-Phenotype -- "
            "OMIM-Gene-600761-Disease-Liddle-Syndrome-177200"
        ),
        "locus": "16p12.2",
        "protein_size": "649 aa / 76 kDa",
        "inheritance": (
            "AD (autosomal dominant, gain-of-function); "
            "Liddle syndrome — identical mechanism to SCNN1B: PY-motif (PPxY) truncation or missense "
            "in C-terminal tail → Nedd4-2 cannot bind → ENaC-γ not ubiquitinated → not internalised → "
            "ENaC accumulates at apical membrane → constitutive Na+ reabsorption; "
            "IDENTICAL PHENOTYPE to SCNN1B Liddle: hypokalemic hypertension, low renin, low aldosterone; "
            "SCNN1B and SCNN1G are adjacent genes at 16p12.2 — when Liddle is suspected, "
            "BOTH subunits must be sequenced; negative SCNN1B does NOT exclude Liddle; "
            "treatment identical: amiloride or triamterene; spironolactone ineffective"
        ),
        "disease_category": (
            "Liddle Syndrome — ENaC-γ gain-of-function; "
            "IDENTICAL PHENOTYPE to SCNN1B: hypokalemic HTN + LOW RENIN + LOW ALDOSTERONE; "
            "test BOTH SCNN1B and SCNN1G (adjacent genes, 16p12.2); "
            "amiloride/triamterene curative — spironolactone ineffective"
        ),
        "disease_pathway": (
            "SCNN1G encodes the γ subunit of ENaC. The α/β/γ heterotrimer forms the functional ENaC channel "
            "at the apical membrane of the cortical collecting duct. The γ subunit, like the β subunit, contains "
            "a C-terminal PPxY (PY) motif that serves as the docking site for Nedd4-2 ubiquitin ligase. "
            "Identical pathomechanism to SCNN1B: PY-motif mutations abolish Nedd4-2 binding → "
            "ENaC not ubiquitinated → not internalised from the apical membrane → "
            "constitutive Na+ and water reabsorption in the cortical collecting duct → "
            "volume expansion → renin suppression → aldosterone suppression → "
            "hypokalemia (K+ excreted via ROMK channel down electrochemical gradient created by excess Na+ reabsorption). "
            "CLINICAL NOTE: SCNN1B and SCNN1G sit in the same genomic region (chromosome 16p12.2); "
            "a panel that only sequences one subunit misses the other; "
            "some Liddle families carry SCNN1G rather than SCNN1B mutations — "
            "a negative SCNN1B result in a classic Liddle phenotype mandates SCNN1G testing before "
            "the genetic diagnosis is considered negative. "
            "Treatment logic is identical: amiloride directly blocks the ENaC pore (Na+ entry site) "
            "regardless of which subunit carries the PY-motif defect."
        ),
        "pathognomonic": (
            "IDENTICAL TO SCNN1B LIDDLE: "
            "HYPERTENSION + HYPOKALEMIA + LOW RENIN + LOW ALDOSTERONE PATHOGNOMONIC; "
            "NEGATIVE SCNN1B DOES NOT EXCLUDE LIDDLE — test SCNN1G as well; "
            "AMILORIDE RESPONSE confirms ENaC excess; "
            "SCNN1B and SCNN1G at 16p12.2 — adjacent genes — test both as a pair"
        ),
        "hormone_profile": (
            "Plasma renin activity: suppressed (<0.5 ng/mL/h); "
            "plasma aldosterone: suppressed (<5 ng/dL); "
            "serum potassium: low (2.5-3.5 mmol/L); "
            "urine aldosterone: low; cortisol/cortisone ratio: normal (differs from AME)"
        ),
        "severity_sds": "Moderate-severe; phenotypically indistinguishable from SCNN1B Liddle",
        "treatment": (
            "AMILORIDE: 5-20 mg/day first-line; "
            "TRIAMTERENE: alternative; "
            "SPIRONOLACTONE: DO NOT USE (MR-independent, ineffective); "
            "low-sodium diet; potassium supplementation initially; "
            "if SCNN1B negative in classic Liddle phenotype: sequence SCNN1G before excluding genetic diagnosis; "
            "cascade testing (AD — 50% risk)"
        ),
        "key_features": [
            "Same PY-motif mechanism and phenotype as SCNN1B — clinically indistinguishable",
            "SCNN1B and SCNN1G are adjacent at 16p12.2 — test BOTH when Liddle suspected",
            "Negative SCNN1B does NOT exclude Liddle syndrome — sequence SCNN1G next",
            "Low renin + low aldosterone + hypokalemia + HTN = Liddle or AME biochemical pattern",
            "AMILORIDE/TRIAMTERENE curative; SPIRONOLACTONE INEFFECTIVE",
            "ENaC-γ PY-motif truncation → Nedd4-2 binding abolished → constitutive Na+ reabsorption",
            "AD — 50% risk first-degree relatives; family cascade testing mandatory",
        ],
        "key_ddx": [
            "SCNN1B Liddle (identical — only distinguished by genetic testing)",
            "HSD11B2 AME (also low renin + low aldosterone; high urinary cortisol/cortisone ratio distinguishes)",
            "Primary hyperaldosteronism (HIGH aldosterone — opposite biochemical pattern)",
            "NR3C2 GOF Geller (also low renin; pregnancy exacerbation distinguishes)",
        ],
        "onset_age": "Young adult (teens to 30s)",
        "hypertension_type": "ENaC gain-of-function (Liddle syndrome — ENaC-γ)",
        "nbs_indicated": False,
        "severity": "moderate-severe",
    },

    # -- HSD11B2 -- Apparent Mineralocorticoid Excess (AME) ---------------------------------------
    {
        "gene": "HSD11B2",
        "alt_name": (
            "HSD11B2 (HSD11B2-405aa-16q22.1 / AR -- "
            "APPARENT-MINERALOCORTICOID-EXCESS-AME -- "
            "CORTISOL-ACTS-AS-MR-AGONIST-11BETAHSD2-ABSENT -- "
            "LOW-RENIN-LOW-ALDOSTERONE-HIGH-URINARY-CORTISOL-CORTISONE-RATIO-PATHOGNOMONIC -- "
            "DEXAMETHASONE-SUPPRESSES-CORTISOL-SUBSTRATE-SPIRONOLACTONE-BLOCKS-MR -- "
            "LIQUORICE-INHIBITS-11BETAHSD2-ACQUIRED-AME)"
        ),
        "protein": (
            "HSD11B2 -- 16q22.1 AR -- HSD11B2-405aa -- "
            "11beta-Hydroxysteroid-Dehydrogenase-Type-2-44kDa-NAD+-Dependent-Short-Chain-Dehydrogenase -- "
            "Expressed-Kidney-Distal-Nephron-Placenta-Colon-Converts-Cortisol-To-Cortisone -- "
            "Cortisol-And-Aldosterone-Have-Equal-MR-Affinity-11BETAHSD2-Shields-MR-From-Cortisol -- "
            "AME-LOF-Cortisol-Reaches-MR-Unimpeded-Acts-As-Aldosterone-Mimetic -- "
            "LIQUORICE-CARBENOXOLONE-Inhibit-11BETAHSD2-Acquired-AME-Phenocopy -- "
            "OMIM-Gene-614232-Disease-AME-218030"
        ),
        "locus": "16q22.1",
        "protein_size": "405 aa / 44 kDa",
        "inheritance": (
            "AR (autosomal recessive, biallelic LOF); "
            "Apparent Mineralocorticoid Excess (AME): cortisol (high plasma concentration) normally blocked "
            "from mineralocorticoid receptor (MR) by 11β-HSD2 in kidney; "
            "11β-HSD2 converts cortisol → cortisone (inactive at MR) in distal nephron; "
            "HSD11B2 LOF: cortisol reaches MR unimpeded → cortisol acts as mineralocorticoid → "
            "Na+ retention, K+ wasting, volume expansion → hypertension with LOW RENIN and LOW ALDOSTERONE; "
            "DIAGNOSTIC: urinary free cortisol/cortisone ratio >100 (normal <10) PATHOGNOMONIC; "
            "acquired AME: liquorice (glycyrrhizic acid) and carbenoxolone inhibit 11β-HSD2 — same phenotype; "
            "treatment: dexamethasone (suppresses cortisol production) + spironolactone (blocks MR)"
        ),
        "disease_category": (
            "Apparent Mineralocorticoid Excess (AME) — 11β-HSD2 deficiency; "
            "cortisol-mediated mineralocorticoid excess; LOW RENIN + LOW ALDOSTERONE + "
            "HIGH URINARY CORTISOL/CORTISONE RATIO PATHOGNOMONIC; AR; "
            "acquired AME: liquorice, carbenoxolone"
        ),
        "disease_pathway": (
            "11β-Hydroxysteroid Dehydrogenase Type 2 (11β-HSD2) is an NAD+-dependent enzyme expressed at high levels "
            "in the distal nephron (cortical collecting duct and distal tubule), placenta, and colon — "
            "precisely the tissues where the mineralocorticoid receptor (MR) is expressed. "
            "In normal physiology: cortisol and aldosterone have equal intrinsic affinity for MR; "
            "cortisol circulates at 1000x higher plasma concentration than aldosterone; "
            "without protection, cortisol would overwhelmingly dominate MR activation; "
            "11β-HSD2 provides this protection by oxidising cortisol (C-11 hydroxyl) to cortisone "
            "(C-11 ketone) — cortisone has minimal MR affinity — before cortisol reaches MR in the kidney. "
            "HSD11B2 LOF: cortisol escapes oxidation → reaches MR in collecting duct at full concentration → "
            "activates MR chronically (cortisol at ~200-700 nmol/L vs aldosterone at ~0.1-0.5 nmol/L) → "
            "Na+ retention → volume expansion → suppressed renin-angiotensin-aldosterone axis → "
            "LOW RENIN + LOW ALDOSTERONE + HYPOKALEMIA + SEVERE HYPERTENSION. "
            "DIAGNOSTIC HALLMARK: URINARY CORTISOL/CORTISONE RATIO >100 (normal <10) — "
            "urine free cortisol measured by LC-MS/MS; the ratio reflects kidney 11β-HSD2 activity. "
            "LIQUORICE: glycyrrhizic acid (active metabolite glycyrrhetinic acid) is a competitive inhibitor "
            "of 11β-HSD2 — liquorice/licorice consumption causes acquired AME; "
            "KEY CLINICAL HISTORY: ask about liquorice consumption, chewing tobacco (US), herbal supplements. "
            "TREATMENT: dexamethasone (suppresses endogenous cortisol → removes the MR-activating substrate) "
            "+ spironolactone/eplerenone (direct MR blockade) + low-sodium diet."
        ),
        "pathognomonic": (
            "HYPERTENSION + HYPOKALEMIA + LOW RENIN + LOW ALDOSTERONE + "
            "URINARY CORTISOL/CORTISONE RATIO >100 (NORMAL <10) = AME PATHOGNOMONIC; "
            "liquorice history: acquired AME phenocopy — ask about all liquorice/herbal sources; "
            "diagnosis confirmed by LC-MS/MS urinary steroid profiling; "
            "distinguished from Liddle by high cortisol/cortisone ratio (Liddle has normal ratio)"
        ),
        "hormone_profile": (
            "Plasma renin: suppressed; plasma aldosterone: suppressed; "
            "serum cortisol: normal or elevated (not suppressed); "
            "urinary free cortisol: elevated; urinary cortisone: very low; "
            "URINARY CORTISOL/CORTISONE RATIO: >100 (normal <10) PATHOGNOMONIC; "
            "serum potassium: low; urine potassium: inappropriately elevated"
        ),
        "severity_sds": "Severe — often presenting in childhood; profound hypokalemia; stroke risk in youth",
        "treatment": (
            "DEXAMETHASONE: 0.25-0.75 mg/day (suppresses cortisol substrate — removes AME trigger); "
            "SPIRONOLACTONE/EPLERENONE: MR blockade (adjunct); "
            "low-sodium diet; potassium supplementation; "
            "AVOID LIQUORICE in all forms (phenocopy of AME); "
            "monitor 24h urinary cortisol/cortisone ratio for treatment response; "
            "genetic counselling: 25% recurrence risk (AR)"
        ),
        "key_features": [
            "Urinary cortisol/cortisone ratio >100 (normal <10) = AME PATHOGNOMONIC",
            "Low renin + low aldosterone + hypokalemia + severe HTN — cortisol-mediated MR activation",
            "11β-HSD2 shields kidney MR from cortisol — LOF lifts this protection",
            "Dexamethasone suppresses cortisol substrate; spironolactone blocks MR",
            "Liquorice (glycyrrhizic acid) inhibits 11β-HSD2 — acquired AME — ask history",
            "Often presents in childhood (AR — biallelic, full LOF); stroke risk in teenagers",
            "LC-MS/MS urinary steroid profiling: gold standard for diagnosis and monitoring",
        ],
        "key_ddx": [
            "Liddle syndrome SCNN1B/SCNN1G (also low renin/aldosterone; cortisol/cortisone ratio NORMAL in Liddle)",
            "Liquorice-induced acquired AME (identical biochemistry; dietary history resolves)",
            "NR3C2 GOF Geller (also low renin/aldosterone; normal cortisol/cortisone ratio; pregnancy worsening)",
            "Primary hyperaldosteronism (HIGH aldosterone — opposite; ARR elevated)",
        ],
        "onset_age": "Childhood (AR biallelic LOF — often severe early presentation); acquired AME: any age",
        "hypertension_type": "Apparent Mineralocorticoid Excess (11β-HSD2 deficiency)",
        "nbs_indicated": False,
        "severity": "severe",
    },

    # -- NR3C2 -- Geller syndrome / MR GOF (pregnancy exacerbation) ------------------------------
    {
        "gene": "NR3C2",
        "alt_name": (
            "NR3C2 (NR3C2-984aa-4q31.23 / AD-GOF -- "
            "GELLER-SYNDROME-MINERALOCORTICOID-RECEPTOR-GOF -- "
            "PREGNANCY-EXACERBATION-PATHOGNOMONIC-PROGESTERONE-ACTIVATES-MR-GOF-MUTANT -- "
            "AVOID-ALL-PROGESTINS-INCLUDING-COMBINED-OCP -- "
            "SPIRONOLACTONE-AVOID-IN-PREGNANCY-PROMPT-DELIVERY-RESOLVES-ACUTE-CRISIS)"
        ),
        "protein": (
            "NR3C2 -- 4q31.23 AD GOF -- NR3C2-984aa -- "
            "Mineralocorticoid-Receptor-MR-107kDa-Nuclear-Receptor-Superfamily-Ligand-Binding-Domain -- "
            "LBD-S810L-Most-Common-Geller-GOF-Mutation-Alters-Ligand-Specificity -- "
            "Wildtype-MR-Progesterone-Weak-Partial-Agonist-But-Also-Weak-Antagonist -- "
            "S810L-Converts-Progesterone-To-Full-Strong-MR-Agonist-AND-Spironolactone-To-Agonist -- "
            "Pregnancy-Progesterone-1000x-Rise-Activates-Mutant-MR-Massively -- "
            "OMIM-Gene-600983-Disease-Geller-Syndrome-605115"
        ),
        "locus": "4q31.23",
        "protein_size": "984 aa / 107 kDa",
        "inheritance": (
            "AD GOF (autosomal dominant, gain-of-function); "
            "Geller syndrome — S810L (Ser810Leu) in the ligand-binding domain most common GOF mutation; "
            "wildtype MR: progesterone is a weak partial agonist with weak antagonist properties; "
            "S810L MR: progesterone becomes a POTENT FULL AGONIST — progesterone activates MR fully; "
            "progesterone concentrations rise 1000-fold in pregnancy (first trimester); "
            "DRAMATIC PREGNANCY EXACERBATION: hypertension worsens catastrophically in first trimester — "
            "PATHOGNOMONIC for NR3C2 GOF Geller syndrome; "
            "avoid all progestin-containing contraceptives (triggers same MR activation); "
            "spironolactone: wildtype MR = spironolactone is MR antagonist; "
            "S810L MR: spironolactone acts as MR AGONIST — spironolactone WORSENS hypertension; "
            "prompt delivery resolves pregnancy crisis; eplerenone may be safer (less S810L agonism)"
        ),
        "disease_category": (
            "Geller Syndrome — NR3C2 GOF; "
            "SEVERE PREGNANCY EXACERBATION PATHOGNOMONIC (progesterone activates mutant MR); "
            "LOW RENIN + LOW ALDOSTERONE + HYPOKALEMIA + HTN; "
            "AVOID PROGESTINS + SPIRONOLACTONE; AD GOF"
        ),
        "disease_pathway": (
            "NR3C2 encodes the Mineralocorticoid Receptor (MR), a nuclear receptor in the glucocorticoid/MR "
            "subfamily. MR is expressed in the distal nephron, cardiovascular tissues, brain, and colon. "
            "In normal physiology: aldosterone is the principal MR ligand; "
            "cortisol is excluded from kidney MR by 11β-HSD2; "
            "progesterone has weak partial agonist and weak competitive antagonist properties at wildtype MR. "
            "Geller syndrome (S810L GOF): the substitution in helix 5 of the ligand-binding domain (LBD) "
            "reorganises the LBD activation-function-2 (AF-2) surface → "
            "progesterone now acts as a FULL MR agonist (not partial); "
            "spironolactone (normally an MR antagonist with a C-17 lactone ring) also gains agonist activity "
            "at S810L MR. "
            "PREGNANCY: circulating progesterone rises from ~1 ng/mL to ~100-200 ng/mL in the first trimester → "
            "MR-GOF mutant activated massively → severe, acute hypertension, hypokalemia, volume expansion → "
            "suppressed renin and aldosterone; "
            "this DRAMATIC PREGNANCY EXACERBATION with low renin + low aldosterone (not expected in pre-eclampsia, "
            "which has high renin) = PATHOGNOMONIC for NR3C2 GOF. "
            "CONTRACEPTIVE TRAP: combined oral contraceptive pills (COCPs) containing a progestin component "
            "(even progestins with weak MR agonism) can activate S810L MR → "
            "paradoxical HTN worsening on OCP; avoid all progestin-containing preparations. "
            "TREATMENT IN PREGNANCY: prompt delivery (removes progesterone source); "
            "antihypertensives that avoid MR agonism; spironolactone absolutely contraindicated; "
            "eplerenone (non-lactone MR antagonist) less likely to agonise S810L MR."
        ),
        "pathognomonic": (
            "DRAMATIC HYPERTENSION EXACERBATION IN FIRST TRIMESTER OF PREGNANCY = "
            "NR3C2 GOF GELLER SYNDROME PATHOGNOMONIC; "
            "LOW RENIN + LOW ALDOSTERONE in pregnancy (pre-eclampsia has HIGH renin — opposite); "
            "PROGESTIN-CONTAINING OCP WORSENS HYPERTENSION (activates mutant MR); "
            "SPIRONOLACTONE WORSENS HYPERTENSION IN S810L (spironolactone = MR AGONIST at S810L mutant); "
            "PROMPT DELIVERY resolves pregnancy crisis"
        ),
        "hormone_profile": (
            "Plasma renin: suppressed; plasma aldosterone: suppressed; "
            "serum potassium: low; "
            "cortisol/cortisone ratio: normal (not AME); "
            "pregnancy: above pattern dramatically worsened by rising progesterone"
        ),
        "severity_sds": "Variable at baseline; LIFE-THREATENING in pregnancy; avoid progestins at all times",
        "treatment": (
            "AVOID all progestin-containing contraceptives (COCP, progestin-only pill, Depo-Provera, hormonal IUDs); "
            "AVOID spironolactone (worsens S810L MR activation); "
            "PREGNANCY: prompt delivery is the definitive treatment; "
            "eplerenone (for inter-pregnancy control if needed — lower S810L agonism); "
            "amiloride/thiazides as antihypertensives; low-sodium diet; "
            "CASCADE TESTING all female first-degree relatives (50% risk — AD; pregnancy risk in each)"
        ),
        "key_features": [
            "Dramatic HTN worsening in first trimester = PATHOGNOMONIC for NR3C2 GOF",
            "Progesterone activates S810L MR as full agonist — 1000x progesterone rise in pregnancy triggers crisis",
            "AVOID all progestins (OCP, Depo-Provera, IUD, emergency contraception)",
            "SPIRONOLACTONE CONTRAINDICATED — acts as MR agonist at S810L mutant, worsens HTN",
            "Pre-eclampsia DDx: NR3C2 GOF has LOW renin (pre-eclampsia has HIGH renin)",
            "Prompt delivery resolves the pregnancy hypertensive crisis",
            "Female first-degree relatives at 50% risk — cascade test before reproductive age",
        ],
        "key_ddx": [
            "Pre-eclampsia (HIGH renin — opposite; de novo; no family history; normal off-pregnancy BP)",
            "SCNN1B/SCNN1G Liddle (also low renin/aldosterone; no pregnancy exacerbation; normal cortisol ratio)",
            "HSD11B2 AME (also low renin/aldosterone; high urinary cortisol/cortisone ratio distinguishes)",
            "Primary hyperaldosteronism in pregnancy (HIGH aldosterone — opposite)",
        ],
        "onset_age": "Young adults; CRITICAL RECOGNITION in first pregnancy",
        "hypertension_type": "MR gain-of-function (Geller syndrome)",
        "nbs_indicated": False,
        "severity": "severe (life-threatening in pregnancy)",
    },

    # -- WNK4 -- Gordon syndrome / PHAII-B --------------------------------------------------------
    {
        "gene": "WNK4",
        "alt_name": (
            "WNK4 (WNK4-1243aa-17q21.2 / AD -- "
            "GORDON-SYNDROME-PHAII-B-FAMILIAL-HYPERKALEMIC-HYPERTENSION -- "
            "LOF-LOSES-NCC-SUPPRESSION-NCC-HYPERACTIVATED-NACL-REABSORPTION -- "
            "HYPERKALEMIA+HYPERTENSION+NORMAL-GFR-PATHOGNOMONIC-TRIAD -- "
            "THIAZIDES-DRAMATIC-DIAGNOSTIC-RESPONSE-WITHIN-DAYS-CONFIRMS-DIAGNOSIS)"
        ),
        "protein": (
            "WNK4 -- 17q21.2 AD -- WNK4-1243aa -- "
            "WNK-Kinase-4-WNK-Without-Lysine-K-135kDa-Serine-Threonine-Kinase -- "
            "Suppresses-NCC-SLC12A3-Under-Low-K+-Conditions-Phosphorylates-SPAK-OSR1-NCC -- "
            "WNK4-LOF-NCC-Released-From-Suppression-NCC-Hyperactivated-DCT-NaCl-Reabsorption -- "
            "WNK-SPAK-OSR1-NCC-Pathway-Aldosterone-Independent-Salt-Reabsorption-Distal-Convoluted-Tubule -- "
            "KLHL3-CUL3-E3-Ubiquitin-Ligase-Ubiquitinates-WNK4-For-Degradation -- "
            "OMIM-Gene-601844-Disease-PHAIIB-614491"
        ),
        "locus": "17q21.2",
        "protein_size": "1243 aa / 135 kDa",
        "inheritance": (
            "AD (autosomal dominant, LOF); "
            "Gordon syndrome (Familial Hyperkalemic Hypertension / PHAII, type B) — "
            "WNK4 normally SUPPRESSES NCC (Na-Cl cotransporter, SLC12A3) under low-K+ conditions; "
            "WNK4 LOF: NCC released from suppression → NCC hyperactivated in distal convoluted tubule (DCT) → "
            "excess NaCl reabsorption → volume expansion → hypertension + hyperkalemia; "
            "HYPERKALEMIA: NCC hyperactivation diverts Na+ away from ENaC in collecting duct → "
            "less K+ secretion via ROMK → hyperkalemia (not hypokalemia — opposite of Liddle/AME); "
            "THIAZIDES: directly block NCC → dramatic BP and K normalisation within days = DIAGNOSTIC RESPONSE; "
            "normal GFR (not renal failure despite hyperkalemia) = key distinguishing feature from chronic kidney disease"
        ),
        "disease_category": (
            "Gordon Syndrome / PHAII-B — WNK4 LOF; "
            "NCC hyperactivation → hyperkalemia + hypertension + NORMAL GFR PATHOGNOMONIC TRIAD; "
            "THIAZIDE DIAGNOSTIC RESPONSE (dramatic within days); AD"
        ),
        "disease_pathway": (
            "WNK4 (WNK kinase 4 — 'Without-No-Lysine' kinase 4) is a serine/threonine kinase expressed in the "
            "distal convoluted tubule (DCT) that acts as a central regulator of NCC (Na-Cl cotransporter, SLC12A3) activity. "
            "In normal low-potassium states: WNK4 activates the SPAK/OSR1 kinase cascade → phosphorylation of NCC → "
            "NCC activation → NaCl reabsorption in DCT to prevent Na+ wasting. "
            "In normal high-potassium states: WNK4 (via an aldosterone-independent mechanism) suppresses NCC "
            "via a second pathway → less NaCl reabsorption in DCT → more Na+ delivered to collecting duct → "
            "ENaC-mediated Na+ reabsorption + ROMK-mediated K+ secretion → K+ excretion. "
            "Gordon syndrome WNK4 LOF: the NCC suppression function is lost → NCC constitutively active → "
            "excess NaCl reabsorption in DCT → volume expansion → hypertension; "
            "additionally, hyperactive NCC in DCT reduces Na+ delivery to collecting duct → "
            "less ROMK K+ secretion → HYPERKALEMIA (K+ retained — opposite of Liddle/AME). "
            "HYPERKALEMIA + HYPERTENSION + NORMAL GFR = PATHOGNOMONIC GORDON TRIAD. "
            "THIAZIDE MECHANISM: thiazides directly block NCC (same transporter that is hyperactive in Gordon) → "
            "NaCl is no longer excessively reabsorbed → volume normalises → BP normalises; "
            "more Na+ delivered to collecting duct → ENaC activation → ROMK K+ secretion → K+ normalises; "
            "DRAMATIC RESPONSE (BP + K normalisation within days) = CONFIRMS GORDON DIAGNOSIS. "
            "WNK4 degradation pathway: KLHL3-CUL3 E3 ubiquitin ligase normally ubiquitinates WNK4 → "
            "proteasomal degradation; KLHL3 or CUL3 mutations in other PHAII subtypes have the same net effect "
            "(WNK4 accumulates → NCC hyperactivation)."
        ),
        "pathognomonic": (
            "HYPERKALEMIA + HYPERTENSION + NORMAL GFR = GORDON SYNDROME PATHOGNOMONIC TRIAD; "
            "THIAZIDE RESPONSE: dramatic BP normalisation + K normalisation within DAYS = "
            "PATHOGNOMONIC DIAGNOSTIC RESPONSE (confirms NCC-mediated mechanism); "
            "distinguishes from CKD (reduced GFR, no thiazide K response), "
            "adrenal insufficiency (low cortisol), and pseudohypoaldosteronism (normal BP)"
        ),
        "hormone_profile": (
            "Plasma renin: suppressed or normal; plasma aldosterone: suppressed or normal/mildly elevated; "
            "serum potassium: HIGH (3.5-6.5 mmol/L — hyperkalemia); "
            "serum bicarbonate: low (metabolic acidosis — Type 4 RTA); "
            "normal GFR; transtubular potassium gradient (TTKG): inappropriately low"
        ),
        "severity_sds": "Moderate-severe; hyperkalemia can cause arrhythmia; excellent thiazide response",
        "treatment": (
            "THIAZIDES: hydrochlorothiazide or chlorthalidone — DRAMATIC RESPONSE (first-line, confirms diagnosis); "
            "low-potassium, low-sodium diet; "
            "correct metabolic acidosis (bicarbonate or citrate); "
            "genetic testing panel: WNK4 + WNK1 (MLPA/CNV) + KLHL3 + CUL3 (all PHAII genes); "
            "cascade testing (AD — 50% risk)"
        ),
        "key_features": [
            "Hyperkalemia + hypertension + NORMAL GFR = Gordon triad PATHOGNOMONIC",
            "Thiazide response: dramatic BP + K normalisation within days = confirms NCC mechanism",
            "NCC hyperactivation in DCT: excess NaCl reabsorption + reduced K+ secretion",
            "Hyperkalemia (opposite of Liddle/AME) — Na+ diverted from ENaC-ROMK to NCC",
            "Metabolic acidosis (Type 4 RTA) common (K+ competition with H+ secretion impaired)",
            "WNK4 LOF releases NCC from suppression — KLHL3/CUL3 same net effect (different mechanism)",
            "AD — 50% risk first-degree relatives; Gordon panel should include all 4 genes",
        ],
        "key_ddx": [
            "Chronic kidney disease (CKD — reduced GFR; no thiazide dramatic K normalisation)",
            "Adrenal insufficiency (low cortisol; low sodium; low BP or orthostasis)",
            "Pseudohypoaldosteronism type 1 (NORMAL BP; salt wasting — opposite to Gordon)",
            "WNK1/KLHL3/CUL3 Gordon subtypes (identical phenotype — only distinguished by genetic testing)",
        ],
        "onset_age": "Variable — often childhood or young adulthood; hyperkalemia may be incidental finding",
        "hypertension_type": "Gordon syndrome / PHAII-B (NCC hyperactivation)",
        "nbs_indicated": False,
        "severity": "moderate",
    },

    # -- WNK1 -- Gordon syndrome / PHAII-A (intronic deletion, exome misses) ---------------------
    {
        "gene": "WNK1",
        "alt_name": (
            "WNK1 (WNK1-2382aa-12p13.33 / AD -- "
            "GORDON-SYNDROME-PHAII-A -- "
            "INTRONIC-LARGE-DELETION-NON-CODING-STANDARD-EXOME-MISSES -- "
            "MLPA-CNV-ANALYSIS-MANDATORY-IF-EXOME-NEGATIVE-IN-SUSPECTED-GORDON -- "
            "THIAZIDES-CURATIVE-SAME-NCC-MECHANISM-AS-WNK4)"
        ),
        "protein": (
            "WNK1 -- 12p13.33 AD -- WNK1-2382aa -- "
            "WNK-Kinase-1-Without-Lysine-K-256kDa-Largest-WNK-Family-Member -- "
            "Intron-1-Large-Deletion-41kb-Deletion-WNK1-Expression-Increased-Kidney -- "
            "WNK1-Normally-Activates-SPAK-OSR1-NCC-Pathway -- "
            "Intronic-Deletion-Removes-Kidney-Specific-Negative-Regulatory-Element -- "
            "WNK1-Overexpressed-Kidney-NCC-Hyperactivated-Same-Net-Effect-As-WNK4-LOF -- "
            "NON-CODING-VARIANT-EXOME-SEQUENCING-BLIND-MLPA-CNV-Required -- "
            "OMIM-Gene-605232-Disease-PHAIIA-614492"
        ),
        "locus": "12p13.33",
        "protein_size": "2382 aa / 256 kDa",
        "inheritance": (
            "AD (autosomal dominant, GOF by increased kidney expression); "
            "Gordon syndrome PHAII-A — INTRONIC LARGE DELETION (not a coding sequence variant); "
            "41 kb deletion in intron 1 removes a kidney-specific negative regulatory element → "
            "WNK1 overexpressed in kidney → WNK1 activates SPAK/OSR1/NCC excessively → NCC hyperactivated → "
            "same PHAII phenotype as WNK4 LOF; "
            "CRITICAL DIAGNOSTIC PITFALL: standard exome sequencing covers only CODING sequences — "
            "this intronic deletion is COMPLETELY MISSED by exome; "
            "MLPA (Multiplex Ligation-dependent Probe Amplification) or CNV array analysis is MANDATORY "
            "if exome is negative in a classic Gordon/PHAII phenotype; "
            "thiazides curative (same NCC mechanism); AD"
        ),
        "disease_category": (
            "Gordon Syndrome / PHAII-A — WNK1 intronic large deletion; "
            "IDENTICAL PHENOTYPE TO WNK4 GORDON: hyperkalemia + hypertension + normal GFR; "
            "EXOME SEQUENCING MISSES (non-coding deletion) — MLPA/CNV MANDATORY; "
            "thiazide diagnostic response; AD"
        ),
        "disease_pathway": (
            "WNK1 (WNK kinase 1) is the largest member of the WNK kinase family (2382 aa, ~256 kDa). "
            "In the kidney, WNK1 activates the SPAK/OSR1 kinase cascade → NCC phosphorylation → NCC activation. "
            "Normal regulation: WNK1 expression in the kidney is kept at low levels by a kidney-specific "
            "intronic regulatory element (silencer/repressor) in intron 1. "
            "PHAII-A deletion: a ~41 kb deletion removes this kidney-specific negative regulatory element → "
            "WNK1 is constitutively overexpressed in renal distal convoluted tubule cells → "
            "SPAK/OSR1/NCC pathway hyperactivated → NCC at apical DCT constitutively active → "
            "identical downstream pathophysiology to WNK4 LOF: excess NaCl reabsorption → volume expansion → "
            "HTN; reduced Na+ delivery to collecting duct → reduced K+ secretion via ROMK → HYPERKALEMIA. "
            "EXOME SEQUENCING LIMITATION: standard clinical exome covers coding exons ± limited flanking "
            "intronic sequence; the ~41 kb intronic deletion at WNK1 intron 1 is entirely within non-coding "
            "intronic sequence — undetectable by standard exome. "
            "Genome sequencing or MLPA (specifically designed probes spanning WNK1 intron 1) or "
            "comparative genomic hybridisation (CGH) array are required. "
            "CLINICAL ALGORITHM: Gordon phenotype (hyperkalemia + HTN + normal GFR) → "
            "exome panel (WNK4, KLHL3, CUL3 — coding) → if negative → WNK1 MLPA/CNV → "
            "if negative → whole genome sequencing."
        ),
        "pathognomonic": (
            "GORDON PHENOTYPE (HYPERKALEMIA + HTN + NORMAL GFR) + NEGATIVE EXOME = "
            "MUST ORDER WNK1 MLPA/CNV — EXOME MISSES INTRONIC LARGE DELETION; "
            "THIAZIDE RESPONSE confirms NCC mechanism (same as WNK4 Gordon); "
            "WNK1 largest WNK kinase — intronic deletion removes kidney silencer element → "
            "WNK1 kidney overexpression → NCC hyperactivation"
        ),
        "hormone_profile": (
            "Identical to WNK4 Gordon: plasma renin suppressed or normal; "
            "plasma aldosterone suppressed or normal; "
            "serum potassium: HIGH; serum bicarbonate: low (Type 4 RTA); "
            "normal GFR"
        ),
        "severity_sds": "Moderate — identical to WNK4 Gordon phenotype; thiazide curative",
        "treatment": (
            "THIAZIDES: hydrochlorothiazide or chlorthalidone — dramatic response (first-line); "
            "if exome negative in Gordon phenotype: MANDATORY WNK1 MLPA/CNV before concluding genetic diagnosis negative; "
            "consider whole genome sequencing if MLPA negative; "
            "low-potassium, low-sodium diet; "
            "bicarbonate/citrate for metabolic acidosis; "
            "cascade testing (AD — 50% risk)"
        ),
        "key_features": [
            "WNK1 intronic large deletion (~41 kb in intron 1) — non-coding variant",
            "STANDARD EXOME SEQUENCING MISSES THIS ENTIRELY — critical diagnostic pitfall",
            "MLPA or CNV analysis MANDATORY if exome negative in suspected Gordon syndrome",
            "Identical Gordon phenotype to WNK4: hyperkalemia + HTN + normal GFR + thiazide response",
            "Intronic deletion removes kidney-specific silencer → WNK1 overexpressed in kidney",
            "WNK1 overexpression → SPAK/OSR1/NCC hyperactivation → same NCC-mediated phenotype",
            "Algorithm: coding exome (WNK4/KLHL3/CUL3) → if negative → WNK1 MLPA → if negative → WGS",
        ],
        "key_ddx": [
            "WNK4 Gordon (identical phenotype — only distinguished by genetic testing; exome detects WNK4)",
            "KLHL3/CUL3 Gordon (coding variants — detected by exome; KLHL3 AR more severe)",
            "CKD hyperkalemia (reduced GFR; no thiazide dramatic response)",
            "Adrenal insufficiency (low cortisol; low Na+; low BP)",
        ],
        "onset_age": "Variable — often childhood or young adulthood",
        "hypertension_type": "Gordon syndrome / PHAII-A (WNK1 overexpression via intronic deletion)",
        "nbs_indicated": False,
        "severity": "moderate",
    },

    # -- KLHL3 -- Gordon syndrome / PHAII-D -------------------------------------------------------
    {
        "gene": "KLHL3",
        "alt_name": (
            "KLHL3 (KLHL3-587aa-5q31.2 / AR-biallelic-or-AD-dominant-negative -- "
            "GORDON-SYNDROME-PHAII-D-CRL3-KLHL3-E3-UBIQUITIN-LIGASE-ADAPTOR -- "
            "WNK1-WNK4-NOT-UBIQUITINATED-ACCUMULATE-NCC-HYPERACTIVATION -- "
            "AR-BIALLELIC-MORE-SEVERE-THAN-AD-DOMINANT-NEGATIVE -- "
            "THIAZIDES-CURATIVE)"
        ),
        "protein": (
            "KLHL3 -- 5q31.2 AR/AD -- KLHL3-587aa -- "
            "Kelch-Like-Protein-3-67kDa-BTB-Back-Domain-Kelch-Repeat-Domain -- "
            "Adaptor-For-CUL3-Cullin-3-CRL3-E3-Ubiquitin-Ligase-Complex -- "
            "KLHL3-Kelch-Domain-Binds-WNK4-PPII-Helix-Target-For-Ubiquitination -- "
            "CRL3-KLHL3-Complex-Ubiquitinates-WNK4-WNK1-Targets-Proteasomal-Degradation -- "
            "KLHL3-LOF-WNK1-WNK4-Escape-Ubiquitination-Accumulate-NCC-Hyperactivation -- "
            "AR-Biallelic-No-Functional-KLHL3-Most-Severe-AD-Dominant-Negative-One-Allele -- "
            "OMIM-Gene-605775-Disease-PHAILD-614495"
        ),
        "locus": "5q31.2",
        "protein_size": "587 aa / 67 kDa",
        "inheritance": (
            "AR (biallelic LOF — most severe) or AD (dominant-negative) — two distinct inheritance modes; "
            "KLHL3 is the substrate adaptor for the CUL3-KLHL3 E3 ubiquitin ligase complex; "
            "CRL3^KLHL3 complex ubiquitinates WNK4 and WNK1 → proteasomal degradation → limits NCC activation; "
            "KLHL3 LOF: WNK1 and WNK4 not ubiquitinated → accumulate → SPAK/OSR1/NCC hyperactivated → Gordon phenotype; "
            "AR biallelic: NO functional KLHL3 — more severe phenotype (earlier onset, more severe hyperkalemia/HTN); "
            "AD dominant-negative: heterozygous missense mutations in Kelch domain "
            "(WNK4-binding domain) — one mutant allele impairs function of remaining wildtype "
            "— partial LOF — milder; "
            "thiazides curative (same NCC mechanism); "
            "hotspot: R528H missense in Kelch domain commonly identified"
        ),
        "disease_category": (
            "Gordon Syndrome / PHAII-D — KLHL3 LOF; "
            "CRL3^KLHL3 E3 ligase: WNK4/WNK1 escape ubiquitination → NCC hyperactivation; "
            "AR biallelic = more severe; AD dominant-negative = milder; "
            "hyperkalemia + HTN + normal GFR + thiazide response"
        ),
        "disease_pathway": (
            "KLHL3 (Kelch-Like Protein 3) is the substrate-recognition adaptor of the CRL3^KLHL3 "
            "(CUL3-KLHL3 RING E3 ubiquitin ligase) complex. The complex operates in the distal convoluted tubule "
            "to maintain WNK kinase levels within the normal range: "
            "CUL3 (scaffold) + KLHL3 (adaptor) + RBX1 (RING finger, E2 recruiter) → ubiquitin transfer "
            "onto WNK4 (and WNK1) → proteasomal degradation. "
            "KLHL3 LOF (biallelic or dominant-negative): "
            "WNK4 and WNK1 escape ubiquitination → protein levels rise → "
            "WNK→SPAK/OSR1→NCC phosphorylation cascade amplified → NCC constitutively hyperactive → "
            "Gordon syndrome. "
            "TWO INHERITANCE MODES: "
            "(1) AR biallelic LOF: no functional KLHL3 from either allele — most severe WNK4/WNK1 accumulation — "
            "most severe Gordon phenotype (hyperkalemia requiring hospitalisation, severe HTN, young onset); "
            "(2) AD dominant-negative: heterozygous missense in KLHL3 Kelch domain (WNK4-binding surface) — "
            "the mutant KLHL3 still forms a CRL3 complex but cannot bind WNK4 — "
            "dominant-negative effect impairs wildtype KLHL3 function — partial WNK4 accumulation — milder. "
            "R528H hotspot: C-terminal Kelch repeat, directly contacts WNK4 PPII helix binding site; "
            "R528H mutant KLHL3 cannot ubiquitinate WNK4 — dominant-negative in the Kelch domain."
        ),
        "pathognomonic": (
            "GORDON PHENOTYPE + AR BIALLELIC INHERITANCE = KLHL3 OR CUL3 (most severe subtypes); "
            "AR KLHL3 more severe than AD KLHL3 dominant-negative; "
            "THIAZIDE RESPONSE confirms NCC mechanism; "
            "R528H hotspot in Kelch domain frequently identified; "
            "CRL3^KLHL3 complex: WNK4 ubiquitination — LOF → WNK4/WNK1 accumulate"
        ),
        "hormone_profile": (
            "Plasma renin: suppressed or normal; plasma aldosterone: suppressed or normal; "
            "serum potassium: HIGH (AR more severe elevation); "
            "serum bicarbonate: low (Type 4 RTA); normal GFR"
        ),
        "severity_sds": "AR biallelic = severe (most severe Gordon after CUL3); AD dominant-negative = moderate",
        "treatment": (
            "THIAZIDES: first-line, curative — dramatic response; "
            "AR biallelic: more aggressive thiazide dosing needed; "
            "low-potassium, low-sodium diet; "
            "Gordon panel: WNK4 + WNK1 (MLPA) + KLHL3 + CUL3; "
            "cascade testing: AR = 25% recurrence risk; AD = 50% risk; "
            "consider WGS if standard panel negative"
        ),
        "key_features": [
            "CRL3^KLHL3 E3 ubiquitin ligase: ubiquitinates WNK4/WNK1 for degradation",
            "KLHL3 LOF → WNK4/WNK1 accumulate → SPAK/OSR1/NCC hyperactivation → Gordon phenotype",
            "AR biallelic = most severe (after CUL3); AD dominant-negative = milder",
            "R528H hotspot in Kelch domain (WNK4-binding surface) — dominant-negative",
            "Thiazide curative — same NCC mechanism as all Gordon subtypes",
            "AR inheritance: both parents carriers — 25% recurrence risk; consanguinity common",
            "Kelch domain missense mutations: dominant-negative, impair CUL3 recruitment to WNK4",
        ],
        "key_ddx": [
            "CUL3 Gordon (de novo exon 9 skip; most severe; neonatal onset — distinct from KLHL3 AR)",
            "WNK4 Gordon (AD coding missense/nonsense; less severe than AR KLHL3)",
            "WNK1 Gordon (intronic large deletion; exome misses — requires MLPA)",
            "CKD hyperkalemia (reduced GFR; no thiazide dramatic response)",
        ],
        "onset_age": "AR biallelic: childhood (often severe early onset); AD: variable, young adult",
        "hypertension_type": "Gordon syndrome / PHAII-D (KLHL3 → WNK ubiquitination loss)",
        "nbs_indicated": False,
        "severity": "severe (AR biallelic) / moderate (AD dominant-negative)",
    },

    # -- CUL3 -- Gordon syndrome / PHAII-E (de novo, most severe, neonatal) ----------------------
    {
        "gene": "CUL3",
        "alt_name": (
            "CUL3 (CUL3-768aa-2q36.2 / AD-DE-NOVO -- "
            "GORDON-SYNDROME-PHAII-E-MOST-SEVERE-SUBTYPE -- "
            "DE-NOVO-EXON-9-SKIP-VARIANT-MOST-COMMON -- "
            "NEONATAL-EARLY-CHILDHOOD-SEVERE-PRESENTATION -- "
            "HIGH-RATE-DE-NOVO-MUTATIONS-OFTEN-NO-FAMILY-HISTORY -- "
            "THIAZIDES-CURATIVE)"
        ),
        "protein": (
            "CUL3 -- 2q36.2 AD de novo -- CUL3-768aa -- "
            "Cullin-3-89kDa-Scaffold-Protein-CRL3-E3-Ubiquitin-Ligase-Complex -- "
            "Cullin-Domain-Binds-RBX1-RING-Finger-Recruits-E2-Ubiquitin-Conjugating-Enzyme -- "
            "N-Terminal-Domains-Bind-BTB-Domain-Adaptors-Including-KLHL3 -- "
            "Exon-9-Skip-Produces-Dominant-Negative-CUL3-Delta-403-459 -- "
            "Delta-Exon9-CUL3-Cannot-Bind-KLHL3-Properly-WNK4-Not-Ubiquitinated -- "
            "Highest-Rate-De-Novo-Among-Gordon-Genes -- "
            "OMIM-Gene-603136-Disease-PHAIIE-614496"
        ),
        "locus": "2q36.2",
        "protein_size": "768 aa / 89 kDa",
        "inheritance": (
            "AD (autosomal dominant, de novo in most cases — dominant-negative effect); "
            "Gordon syndrome PHAII-E — MOST SEVERE Gordon subtype; "
            "MOST COMMON MUTATION: heterozygous de novo variant causing exon 9 skipping — "
            "produces a dominant-negative CUL3 protein (CUL3-Δ403-459) missing 57 aa from cullin domain; "
            "CUL3-Δexon9 cannot bind KLHL3 adaptor properly → WNK4 not ubiquitinated → accumulates → "
            "most severe NCC hyperactivation; "
            "HIGH RATE OF DE NOVO MUTATIONS: most patients have NO FAMILY HISTORY → "
            "do NOT exclude this diagnosis because family history is negative; "
            "NEONATAL/EARLY CHILDHOOD PRESENTATION: most severe presentation of all Gordon subtypes; "
            "thiazides curative (same NCC mechanism); "
            "exon 9 skip detectable by sequencing + RNA (RT-PCR confirms splice defect)"
        ),
        "disease_category": (
            "Gordon Syndrome / PHAII-E — CUL3 dominant-negative exon 9 skip; "
            "MOST SEVERE Gordon subtype; NEONATAL/EARLY CHILDHOOD presentation; "
            "HIGH DE NOVO RATE — often no family history; "
            "thiazide curative; AD de novo"
        ),
        "disease_pathway": (
            "CUL3 (Cullin-3) is the scaffold protein of the CRL3 family of E3 ubiquitin ligases. "
            "The CRL3 complex: CUL3 (scaffold, N-terminal domain binds BTB adaptors, C-terminal domain binds RBX1) "
            "+ BTB-domain adaptor (substrate recognition, e.g. KLHL3 for WNK substrates) "
            "+ RBX1 (RING finger, recruits E2 ubiquitin-conjugating enzyme) + E2 → ubiquitin chain on substrate. "
            "Normal: CUL3 + KLHL3 + WNK4 → CUL3 scaffolds the ubiquitin transfer to WNK4 → WNK4 degraded. "
            "CUL3 exon 9 skip (de novo dominant-negative): "
            "the 57 aa encoded by exon 9 is part of the cullin domain required for KLHL3 docking; "
            "CUL3-Δexon9 protein: KLHL3 binds CUL3 much less efficiently → "
            "even though KLHL3 tries to recruit WNK4 for ubiquitination, the E3 complex cannot assemble → "
            "WNK4 (and WNK1) accumulate to very high levels; "
            "the dominant-negative effect is amplified because CUL3-Δexon9 may sequester the "
            "available KLHL3 molecules without delivering ubiquitin to WNK4. "
            "MOST SEVERE SUBTYPE: WNK4 accumulates to higher levels than in WNK4 LOF or KLHL3 AR → "
            "most severe NCC hyperactivation → most severe hyperkalemia (can be life-threatening in neonates) + "
            "most severe hypertension. "
            "DE NOVO RATE: CUL3 exon 9 skip is among the highest de novo rates of any Gordon gene — "
            "a child presenting with NEONATAL/INFANTILE hyperkalemia + hypertension + normal GFR, "
            "with no family history of Gordon syndrome, should have CUL3 exon 9 skip considered FIRST "
            "among the de novo Gordon genes."
        ),
        "pathognomonic": (
            "NEONATAL/EARLY CHILDHOOD SEVERE HYPERKALEMIA + HYPERTENSION + NORMAL GFR = "
            "CUL3 GORDON PHAII-E PATHOGNOMONIC (most severe subtype); "
            "NO FAMILY HISTORY does NOT exclude CUL3 (high de novo rate); "
            "DE NOVO EXON 9 SKIP = dominant-negative CUL3-Δ403-459; "
            "THIAZIDE RESPONSE confirms NCC mechanism (dramatic + rapid); "
            "most severe thiazide requirement among Gordon subtypes"
        ),
        "hormone_profile": (
            "Plasma renin: suppressed; plasma aldosterone: suppressed; "
            "serum potassium: SEVERELY HIGH (often >6 mmol/L in neonates — arrhythmia risk); "
            "serum bicarbonate: severely low; metabolic acidosis; normal GFR"
        ),
        "severity_sds": "Most severe Gordon subtype; neonatal emergency; life-threatening hyperkalemia",
        "treatment": (
            "THIAZIDES: curative — same NCC mechanism (dramatic response but higher doses often needed); "
            "NEONATAL EMERGENCY: calcium gluconate (cardiac membrane stabilisation); "
            "insulin + glucose (K+ shift intracellular); emergency management of life-threatening hyperkalemia; "
            "long-term: hydrochlorothiazide + low-potassium diet; "
            "de novo variant: parental testing to confirm de novo status (inform future sibling recurrence risk ~low); "
            "repeat exon 9 skip confirmation: sequencing + RT-PCR on mRNA"
        ),
        "key_features": [
            "Most severe Gordon syndrome subtype — neonatal/early childhood presentation",
            "HIGH DE NOVO RATE — negative family history does NOT exclude CUL3 Gordon",
            "De novo exon 9 skip → dominant-negative CUL3-Δ403-459 → KLHL3 docking impaired",
            "WNK4 accumulates to highest levels → most severe NCC hyperactivation",
            "Neonatal life-threatening hyperkalemia (>6 mmol/L) + severe HTN + normal GFR",
            "Thiazide curative but highest doses needed among Gordon subtypes",
            "CUL3 pathogenic exon 9 skip: confirm by sequencing + RT-PCR on mRNA (splice defect)",
        ],
        "key_ddx": [
            "KLHL3 AR biallelic Gordon (severe; AR inheritance; consanguinity; older onset than CUL3 de novo)",
            "WNK4 Gordon (AD missense/nonsense; family history common; less severe than CUL3)",
            "Neonatal adrenal insufficiency (low cortisol; low Na+; low BP — opposite HTN)",
            "Pseudohypoaldosteronism type 1 (salt wasting, low BP — opposite)",
        ],
        "onset_age": "Neonatal to early childhood (most severe subtype; de novo); thiazide response dramatic",
        "hypertension_type": "Gordon syndrome / PHAII-E (CUL3 dominant-negative exon 9 skip)",
        "nbs_indicated": False,
        "severity": "severe",
    },
]


# ---------------------------------------------------------------------------
# Cohort simulation rates
# ---------------------------------------------------------------------------
_RATES = {
    "SCNN1B": {"surgery": 0.02, "amiloride": 0.88, "thiazide": 0.05, "spironolactone": 0.08,
               "dexamethasone": 0.00, "surveillance": 0.80, "genetic_diagnosis": 0.78},
    "SCNN1G": {"surgery": 0.02, "amiloride": 0.86, "thiazide": 0.05, "spironolactone": 0.10,
               "dexamethasone": 0.00, "surveillance": 0.78, "genetic_diagnosis": 0.72},
    "HSD11B2": {"surgery": 0.01, "amiloride": 0.20, "thiazide": 0.30, "spironolactone": 0.70,
                "dexamethasone": 0.88, "surveillance": 0.92, "genetic_diagnosis": 0.84},
    "NR3C2":  {"surgery": 0.03, "amiloride": 0.40, "thiazide": 0.50, "spironolactone": 0.08,
               "dexamethasone": 0.00, "surveillance": 0.85, "genetic_diagnosis": 0.76},
    "WNK4":   {"surgery": 0.01, "amiloride": 0.10, "thiazide": 0.92, "spironolactone": 0.04,
               "dexamethasone": 0.00, "surveillance": 0.87, "genetic_diagnosis": 0.80},
    "WNK1":   {"surgery": 0.01, "amiloride": 0.08, "thiazide": 0.91, "spironolactone": 0.03,
               "dexamethasone": 0.00, "surveillance": 0.85, "genetic_diagnosis": 0.62},
    "KLHL3":  {"surgery": 0.02, "amiloride": 0.10, "thiazide": 0.93, "spironolactone": 0.05,
               "dexamethasone": 0.00, "surveillance": 0.88, "genetic_diagnosis": 0.74},
    "CUL3":   {"surgery": 0.04, "amiloride": 0.12, "thiazide": 0.94, "spironolactone": 0.03,
               "dexamethasone": 0.00, "surveillance": 0.90, "genetic_diagnosis": 0.70},
}

_AGE_RANGES = {
    "SCNN1B": (12, 45),
    "SCNN1G": (14, 48),
    "HSD11B2": (3, 30),
    "NR3C2":  (16, 40),
    "WNK4":   (5, 35),
    "WNK1":   (6, 38),
    "KLHL3":  (2, 30),
    "CUL3":   (0, 5),
}

_BP_RANGES = {
    "SCNN1B": (155, 200),
    "SCNN1G": (152, 198),
    "HSD11B2": (165, 210),
    "NR3C2":  (158, 205),
    "WNK4":   (150, 190),
    "WNK1":   (148, 188),
    "KLHL3":  (155, 195),
    "CUL3":   (160, 210),
}

_K_RANGES = {
    "SCNN1B": (2.6, 3.4),
    "SCNN1G": (2.7, 3.5),
    "HSD11B2": (2.5, 3.2),
    "NR3C2":  (2.7, 3.4),
    "WNK4":   (5.0, 6.8),   # hyperkalemia — Gordon
    "WNK1":   (4.8, 6.5),
    "KLHL3":  (4.9, 6.8),
    "CUL3":   (5.2, 7.2),   # most severe
}

_LOW_RENIN_RATES = {
    "SCNN1B": 0.95, "SCNN1G": 0.94, "HSD11B2": 0.97, "NR3C2": 0.93,
    "WNK4":   0.72, "WNK1":   0.70, "KLHL3":   0.75, "CUL3":   0.78,
}

_LOW_ALDO_RATES = {
    "SCNN1B": 0.94, "SCNN1G": 0.93, "HSD11B2": 0.96, "NR3C2": 0.92,
    "WNK4":   0.68, "WNK1":   0.66, "KLHL3":   0.70, "CUL3":   0.72,
}

_TYPE_VARIANTS = {
    "SCNN1B": ["SCNN1B-PY-Motif-Truncation", "SCNN1B-PY-Motif-Missense", "SCNN1B-Frameshift-C-Term", "SCNN1B-Splice-PY-Domain"],
    "SCNN1G": ["SCNN1G-PY-Motif-Truncation", "SCNN1G-PY-Motif-Missense", "SCNN1G-Frameshift-C-Term", "SCNN1G-Nonsense-PY-Region"],
    "HSD11B2": ["HSD11B2-Biallelic-Missense", "HSD11B2-Compound-Het", "HSD11B2-Homozygous-Nonsense", "HSD11B2-Splice-Site"],
    "NR3C2":  ["NR3C2-S810L-GOF-Geller", "NR3C2-LBD-Missense-GOF", "NR3C2-AF2-Helix-GOF", "NR3C2-Other-LBD-GOF"],
    "WNK4":   ["WNK4-Kinase-Domain-Missense", "WNK4-PPII-Helix-Missense", "WNK4-Nonsense-LOF", "WNK4-Frameshift-LOF"],
    "WNK1":   ["WNK1-Intron1-41kb-Deletion", "WNK1-Intron1-Deletion-Smaller", "WNK1-Intron1-CNV-Complex", "WNK1-Regulatory-Deletion"],
    "KLHL3":  ["KLHL3-R528H-Kelch-DN", "KLHL3-Kelch-Missense-DN", "KLHL3-Biallelic-Null-AR", "KLHL3-BTB-Domain-Missense"],
    "CUL3":   ["CUL3-Exon9-Skip-De-Novo", "CUL3-Exon9-Splice-De-Novo", "CUL3-Cullin-Domain-DN", "CUL3-De-Novo-Other-Exon9"],
}

_DE_NOVO_RATES = {
    "SCNN1B": 0.10, "SCNN1G": 0.10, "HSD11B2": 0.00, "NR3C2": 0.25,
    "WNK4": 0.05, "WNK1": 0.05, "KLHL3": 0.08, "CUL3": 0.68,
}

_GENE_IDX = {e["gene"]: i for i, e in enumerate(HYPERTENSION_GENES)}


def _make_cohort(entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = entry["gene"]
    r = _RATES[gene]
    age_min, age_max = _AGE_RANGES[gene]
    sbp_min, sbp_max = _BP_RANGES[gene]
    k_min, k_max = _K_RANGES[gene]
    variants = _TYPE_VARIANTS[gene]

    cohort = []
    sexes = ["M", "F"]
    for j in range(n):
        sex = rng.choice(sexes)
        age_onset = round(rng.uniform(age_min, max(age_min, age_max - 5)), 2)
        age_dx = round(age_onset + rng.uniform(0.5, 8), 2)
        type_variant = rng.choice(variants)
        sbp = round(rng.uniform(sbp_min, sbp_max), 1)
        dbp = round(sbp * rng.uniform(0.56, 0.64), 1)
        serum_k = round(rng.uniform(k_min, k_max), 2)
        plasma_renin_low = rng.random() < _LOW_RENIN_RATES[gene]
        plasma_aldo_low = rng.random() < _LOW_ALDO_RATES[gene]
        on_amiloride = rng.random() < r["amiloride"]
        on_thiazide = rng.random() < r["thiazide"]
        on_spironolactone = rng.random() < r["spironolactone"]
        on_dexamethasone = rng.random() < r["dexamethasone"]
        had_genetic_diagnosis = rng.random() < r["genetic_diagnosis"]
        on_surveillance = rng.random() < r["surveillance"]
        de_novo = rng.random() < _DE_NOVO_RATES[gene]

        patient = {
            "patient_id": f"HTN-{seed:04d}-{j:03d}",
            "gene": gene,
            "sex": sex,
            "age_at_onset": age_onset,
            "age_at_diagnosis": age_dx,
            "type_variant": type_variant,
            "systolic_bp_mmHg": sbp,
            "diastolic_bp_mmHg": dbp,
            "serum_k_mmol_L": serum_k,
            "plasma_renin_low": plasma_renin_low,
            "plasma_aldosterone_low": plasma_aldo_low,
            "on_amiloride_triamterene": on_amiloride,
            "on_thiazide": on_thiazide,
            "on_spironolactone": on_spironolactone,
            "on_dexamethasone": on_dexamethasone,
            "had_genetic_diagnosis": had_genetic_diagnosis,
            "on_surveillance": on_surveillance,
            "de_novo_variant": de_novo,
        }
        cohort.append(patient)
    return cohort


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(HYPERTENSION_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    amiloride_count    = sum(1 for p in all_patients if p["on_amiloride_triamterene"])
    thiazide_count     = sum(1 for p in all_patients if p["on_thiazide"])
    spiro_count        = sum(1 for p in all_patients if p["on_spironolactone"])
    dexa_count         = sum(1 for p in all_patients if p["on_dexamethasone"])
    low_renin_count    = sum(1 for p in all_patients if p["plasma_renin_low"])
    low_aldo_count     = sum(1 for p in all_patients if p["plasma_aldosterone_low"])
    genetic_dx_count   = sum(1 for p in all_patients if p["had_genetic_diagnosis"])
    surv_count         = sum(1 for p in all_patients if p["on_surveillance"])
    de_novo_count      = sum(1 for p in all_patients if p["de_novo_variant"])

    gene_summary = {}
    for idx, entry in enumerate(HYPERTENSION_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "hormone_profile": entry["hormone_profile"],
            "hypertension_type": entry["hypertension_type"],
            "n_patients": len(cohort),
            "avg_sbp": round(sum(p["systolic_bp_mmHg"] for p in cohort) / len(cohort), 1),
            "avg_serum_k": round(sum(p["serum_k_mmol_L"] for p in cohort) / len(cohort), 2),
            "low_renin_pct": round(100 * sum(1 for p in cohort if p["plasma_renin_low"]) / len(cohort), 1),
            "low_aldo_pct": round(100 * sum(1 for p in cohort if p["plasma_aldosterone_low"]) / len(cohort), 1),
            "amiloride_pct": round(100 * sum(1 for p in cohort if p["on_amiloride_triamterene"]) / len(cohort), 1),
            "thiazide_pct": round(100 * sum(1 for p in cohort if p["on_thiazide"]) / len(cohort), 1),
            "genetic_dx_pct": round(100 * sum(1 for p in cohort if p["had_genetic_diagnosis"]) / len(cohort), 1),
            "surveillance_pct": round(100 * sum(1 for p in cohort if p["on_surveillance"]) / len(cohort), 1),
            "avg_age_onset": round(sum(p["age_at_onset"] for p in cohort) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Monogenic-Hypertension-Atlas",
        "subtitle": (
            "Complete 8-Gene Reference -- "
            "SCNN1B/SCNN1G (Liddle) / HSD11B2 (AME) / NR3C2 (Geller MR-GOF) / "
            "WNK4-WNK1-KLHL3-CUL3 (Gordon PHAII)"
        ),
        "genes_covered": [e["gene"] for e in HYPERTENSION_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "amiloride_triamterene_pct": round(100 * amiloride_count / total, 1),
            "thiazide_pct": round(100 * thiazide_count / total, 1),
            "spironolactone_pct": round(100 * spiro_count / total, 1),
            "dexamethasone_pct": round(100 * dexa_count / total, 1),
            "low_renin_pct": round(100 * low_renin_count / total, 1),
            "low_aldo_pct": round(100 * low_aldo_count / total, 1),
            "genetic_diagnosis_pct": round(100 * genetic_dx_count / total, 1),
            "surveillance_pct": round(100 * surv_count / total, 1),
            "de_novo_pct": round(100 * de_novo_count / total, 1),
        },
        "gene_summary": gene_summary,
        "clinical_pearls": [
            "Liddle: AMILORIDE (not spironolactone) — ENaC-not-MR mechanism; spiro is ineffective",
            "AME: LOW renin + LOW aldosterone + HIGH cortisol/cortisone ratio (urine) + liquorice history",
            "NR3C2 GOF: pregnancy is LIFE-THREATENING; avoid ALL progestins; spironolactone worsens (MR agonist at S810L)",
            "Gordon: THIAZIDE RESPONSE is PATHOGNOMONIC — dramatic BP + K normalisation within days confirms NCC mechanism",
            "WNK1 exome-miss: ALWAYS order MLPA/CNV if exome negative in suspected Gordon",
            "CUL3 de novo: highest rate of de novo mutations; often NO family history; most severe neonatal phenotype",
        ],
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(HYPERTENSION_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "hormone_profile": entry["hormone_profile"],
            "onset_age": entry["onset_age"],
            "severity_sds": entry["severity_sds"],
            "hypertension_type": entry["hypertension_type"],
            "nbs_indicated": entry["nbs_indicated"],
            "severity": entry["severity"],
            "n_patients": len(cohort),
            "avg_sbp": round(sum(p["systolic_bp_mmHg"] for p in cohort) / len(cohort), 1),
            "avg_serum_k": round(sum(p["serum_k_mmol_L"] for p in cohort) / len(cohort), 2),
            "low_renin_pct": round(100 * sum(1 for p in cohort if p["plasma_renin_low"]) / len(cohort), 1),
            "low_aldo_pct": round(100 * sum(1 for p in cohort if p["plasma_aldosterone_low"]) / len(cohort), 1),
            "amiloride_pct": round(100 * sum(1 for p in cohort if p["on_amiloride_triamterene"]) / len(cohort), 1),
            "thiazide_pct": round(100 * sum(1 for p in cohort if p["on_thiazide"]) / len(cohort), 1),
            "spironolactone_pct": round(100 * sum(1 for p in cohort if p["on_spironolactone"]) / len(cohort), 1),
            "dexamethasone_pct": round(100 * sum(1 for p in cohort if p["on_dexamethasone"]) / len(cohort), 1),
            "genetic_dx_pct": round(100 * sum(1 for p in cohort if p["had_genetic_diagnosis"]) / len(cohort), 1),
            "surveillance_pct": round(100 * sum(1 for p in cohort if p["on_surveillance"]) / len(cohort), 1),
            "de_novo_pct": round(100 * sum(1 for p in cohort if p["de_novo_variant"]) / len(cohort), 1),
            "avg_age_onset": round(sum(p["age_at_onset"] for p in cohort) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["alt_name"].split(" (")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "hormone_profile": entry["hormone_profile"],
                "onset_age": entry["onset_age"],
                "severity_sds": entry["severity_sds"],
                "hypertension_type": entry["hypertension_type"],
                "nbs_indicated": entry["nbs_indicated"],
            }
            for entry in HYPERTENSION_GENES
        },
        "hypertension_glossary": {
            "Liddle Syndrome — ENaC GOF (SCNN1B/SCNN1G)": (
                "Liddle syndrome is caused by autosomal dominant gain-of-function mutations in the β (SCNN1B) "
                "or γ (SCNN1G) subunit of the Epithelial Sodium Channel (ENaC), both located at chromosome 16p12.2. "
                "The PY motif (PPxY sequence) in the C-terminal intracellular tail of β and γ subunits is the "
                "docking site for Nedd4-2 ubiquitin ligase. PY-motif truncation or missense → Nedd4-2 cannot bind → "
                "ENaC not ubiquitinated → not internalised from the apical membrane of the cortical collecting duct → "
                "constitutive, aldosterone-independent Na+ reabsorption → volume expansion → "
                "LOW RENIN + LOW ALDOSTERONE + HYPOKALEMIA + HYPERTENSION. "
                "TREATMENT CRITICAL DISTINCTION: amiloride and triamterene are direct ENaC channel blockers — "
                "they block the channel pore regardless of aldosterone signalling — curative for Liddle. "
                "SPIRONOLACTONE IS COMPLETELY INEFFECTIVE: spironolactone blocks the mineralocorticoid receptor (MR); "
                "in Liddle syndrome, MR is NOT the driving force for ENaC activation (ENaC is constitutively active "
                "because it cannot be internalised, not because aldosterone/MR is overactive); "
                "blocking MR with spironolactone has no impact on ENaC channel activity in Liddle — "
                "a well-documented prescribing error in initial management of new Liddle cases. "
                "DIAGNOSTIC TEST: amiloride trial — dramatic BP and K normalisation confirms Liddle; "
                "spironolactone failure + subsequent amiloride response = diagnostic sequence. "
                "GENETIC TESTING ALERT: SCNN1B and SCNN1G are adjacent genes at 16p12.2; "
                "a panel that sequences only SCNN1B misses SCNN1G Liddle; always test both subunits together."
            ),
            "Apparent Mineralocorticoid Excess — HSD11B2": (
                "Apparent Mineralocorticoid Excess (AME) is caused by biallelic loss-of-function mutations in HSD11B2, "
                "encoding 11β-hydroxysteroid dehydrogenase type 2 (11β-HSD2). "
                "11β-HSD2 is expressed at high levels in the kidney distal nephron, placenta, and colon — "
                "precisely the target tissues of the mineralocorticoid receptor (MR). "
                "Its physiological role: convert cortisol (C-11 hydroxyl) → cortisone (C-11 ketone). "
                "Cortisol and aldosterone have EQUAL affinity for MR, but cortisol circulates at 1000x higher "
                "concentration; without 11β-HSD2 'shielding,' cortisol would dominate MR activation. "
                "HSD11B2 LOF: cortisol reaches kidney MR unimpeded → cortisol acts as a mineralocorticoid → "
                "Na+ retention, K+ wasting, volume expansion → hypertension + hypokalemia + "
                "LOW RENIN + LOW ALDOSTERONE. "
                "DIAGNOSTIC HALLMARK: urinary free cortisol/cortisone ratio by LC-MS/MS >100 (normal <10) — "
                "this ratio directly reflects 11β-HSD2 activity in the kidney. "
                "LIQUORICE: glycyrrhizic acid (in liquorice confectionery, chewing tobacco, herbal supplements) "
                "is a competitive inhibitor of 11β-HSD2 → acquired AME; "
                "must ask about liquorice in ALL patients with low renin + low aldosterone hypertension. "
                "TREATMENT: dexamethasone (suppresses endogenous cortisol substrate) + "
                "spironolactone/eplerenone (MR blockade) + low-sodium diet. "
                "DISTINCTION FROM LIDDLE: AME has HIGH urinary cortisol/cortisone ratio (Liddle has normal ratio); "
                "AME is AR (both parents carriers) vs Liddle AD."
            ),
            "Geller Syndrome — NR3C2 GOF and Pregnancy Crisis": (
                "Geller syndrome is caused by autosomal dominant gain-of-function mutations in NR3C2 (mineralocorticoid "
                "receptor, MR), most commonly S810L (Ser810Leu) in the ligand-binding domain (LBD). "
                "In wildtype MR: progesterone is a weak partial agonist with weak antagonist properties. "
                "In S810L MR: progesterone acts as a POTENT FULL MR AGONIST. "
                "PREGNANCY MECHANISM: circulating progesterone rises up to 1000-fold during the first trimester — "
                "in patients with S810L MR, this surge fully activates MR → severe acute hypertension, "
                "hypokalemia, volume expansion → suppressed renin and aldosterone. "
                "PATHOGNOMONIC PRESENTATION: DRAMATIC HTN EXACERBATION IN FIRST TRIMESTER OF PREGNANCY "
                "with LOW RENIN + LOW ALDOSTERONE (distinguishes from pre-eclampsia, which has HIGH renin). "
                "SPIRONOLACTONE DANGER: spironolactone (normally an MR antagonist with C-17 lactone ring) "
                "gains MR AGONIST activity at S810L mutant → spironolactone WORSENS hypertension in Geller patients. "
                "CONTRACEPTIVE AVOIDANCE: ALL progestin-containing contraceptives (combined OCP, progestin-only pill, "
                "Depo-Provera, hormonal IUDs, emergency contraception containing progestins) can activate S810L MR. "
                "MANAGEMENT IN PREGNANCY: prompt delivery is definitive treatment; "
                "antihypertensives that avoid MR pathway; eplerenone may be safer than spironolactone "
                "(non-lactone structure — less S810L agonism); "
                "cascade testing all female first-degree relatives before reproductive age."
            ),
            "Gordon Syndrome / PHAII — WNK-NCC Pathway": (
                "Gordon syndrome (Familial Hyperkalemic Hypertension, PHAII) is caused by mutations in the "
                "WNK-SPAK/OSR1-NCC pathway, producing HYPERKALEMIA + HYPERTENSION + NORMAL GFR — "
                "a triad that is pathognomonic and distinguishes Gordon from CKD (where GFR is reduced). "
                "The four genetic subtypes share the same final common pathway — NCC hyperactivation: "
                "WNK4 (PHAII-B, 17q21.2, AD, LOF): WNK4 normally suppresses NCC; LOF releases NCC; "
                "WNK1 (PHAII-A, 12p13.33, AD, INTRONIC LARGE DELETION): non-coding deletion removes kidney "
                "silencer → WNK1 overexpressed → NCC hyperactivated; STANDARD EXOME MISSES — requires MLPA/CNV; "
                "KLHL3 (PHAII-D, 5q31.2, AR or AD): CRL3^KLHL3 E3 ligase ubiquitinates WNK4/WNK1; "
                "LOF → WNK accumulation; AR biallelic = more severe; "
                "CUL3 (PHAII-E, 2q36.2, AD de novo): exon 9 skip → dominant-negative CUL3 → "
                "KLHL3 docking impaired → WNK4 not ubiquitinated; most severe; highest de novo rate. "
                "THIAZIDE DIAGNOSTIC RESPONSE: thiazides directly block NCC (the hyperactive transporter) → "
                "DRAMATIC normalisation of both BP and K within DAYS — this is PATHOGNOMONIC for Gordon syndrome "
                "and confirms NCC-mediated pathophysiology. "
                "CLINICAL ALGORITHM: Gordon phenotype → coding panel (WNK4, KLHL3, CUL3) → "
                "if negative → WNK1 MLPA/CNV (exome cannot detect intronic deletion) → "
                "if negative → whole genome sequencing. "
                "WNK1 is the most commonly MISSED Gordon gene because exome is the routine first-line test."
            ),
            "Renin-Aldosterone Profile in Monogenic Hypertension — Diagnostic Framework": (
                "The plasma renin activity (PRA) and plasma aldosterone concentration (PAC), combined with "
                "the aldosterone-renin ratio (ARR), define four biochemical subtypes of monogenic hypertension. "
                "LOW RENIN + LOW ALDOSTERONE: Liddle syndrome (SCNN1B/SCNN1G), AME (HSD11B2), Geller syndrome (NR3C2 GOF). "
                "All three have volume expansion → suppressed renin → suppressed aldosterone. "
                "Distinguishing the three: "
                "(1) Liddle: normal urinary cortisol/cortisone ratio; amiloride responsive; no pregnancy exacerbation; "
                "(2) AME: HIGH urinary cortisol/cortisone ratio (>100); dexamethasone suppresses; liquorice history; "
                "(3) Geller: normal cortisol ratio; PREGNANCY EXACERBATION; progestins worsen; spironolactone worsens. "
                "HIGH RENIN + HIGH ALDOSTERONE: primary aldosteronism phenotypes (separate genetics — not in this atlas). "
                "NORMAL/VARIABLE RENIN + HYPERKALEMIA: Gordon syndrome (WNK4/WNK1/KLHL3/CUL3) — "
                "NCC hyperactivation does not primarily suppress renin as severely as ENaC excess; "
                "hyperkalemia is the DOMINANT feature (not seen in Liddle/AME/Geller). "
                "PRACTICAL ALGORITHM: "
                "Step 1: measure renin + aldosterone + serum K. "
                "Step 2: low renin + low aldo + hypoK → urinary cortisol/cortisone ratio; "
                "if elevated → AME; if normal → amiloride trial → if BP/K respond → Liddle; "
                "if pregnancy exacerbation → Geller. "
                "Step 3: low renin + aldo + hyperK → thiazide trial → if dramatic response → Gordon; "
                "then Gordon genetic panel (WNK4/KLHL3/CUL3 exome → WNK1 MLPA)."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first key) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
