#!/usr/bin/env python3
"""Hereditary-Mineralocorticoid-Excess-Atlas — Complete 8-Gene AME/Liddle/Gordon/Geller Atlas
HSD11B2 (11β-hydroxysteroid dehydrogenase type 2; 405 aa; 11q22.1; AR;
         Apparent Mineralocorticoid Excess (AME); cortisol acts as mineralocorticoid
         because renal HSD11B2 normally inactivates cortisol → cortisone in kidney;
         LOF → cortisol accumulates in kidney → MR activation → severe early-onset HTN;
         PATHOGNOMONIC: urinary THF+5αTHF/THE ratio >10 (cortisol:cortisone metabolites);
         carbenoxolone inhibits HSD11B2 → phenocopies AME (liquorice same mechanism);
         seed SEED_BASE+0) ·
SCNN1B  (ENaC β subunit; 640 aa; 16p12.2; AD GOF;
         Liddle syndrome; PY motif mutations in C-terminal domain →
         failed NEDD4-2-mediated ubiquitination → ENaC constitutively open at apical membrane;
         renal Na+ wasting prevented → constitutive Na retention → volume expansion → HTN + hypokalaemia;
         low aldosterone + low renin (suppressed by volume); amiloride/triamterene CURATIVE;
         seed SEED_BASE+1) ·
SCNN1G  (ENaC γ subunit; 649 aa; 16p12.2; AD GOF;
         Liddle syndrome — same phenotype as SCNN1B;
         PY motif or C-terminal truncation → NEDD4-2 binding lost → ENaC open;
         SCNN1B and SCNN1G both tested in Liddle gene panel;
         seed SEED_BASE+2) ·
WNK4   (WNK kinase 4; 1243 aa; 17q21.31; AD;
         Gordon syndrome PHA2B (PHAII/familial hyperkalaemic HTN);
         missense mutations in acidic motif → WNK4 loses KLHL3/CUL3-mediated degradation →
         WNK4 accumulates → NCC phosphorylation → NCC overcorrection → Na/Cl reabsorption elevated;
         PATHOGNOMONIC: hyperkalemia + hyperchloraemic metabolic acidosis + HTN despite NORMAL/elevated aldosterone;
         thiazide CURATIVE (blocks NCC);
         seed SEED_BASE+3) ·
WNK1   (WNK kinase 1; 2382 aa; 12p13.33; AD;
         Gordon syndrome PHA2A; LARGE intronic deletions (intron 1 up to 41 kb) — not point mutations;
         deletion → kidney-specific WNK1 isoform (KS-WNK1) reduced → L-WNK1 dominant →
         NCC activation through OXSR1/SPAK pathway → same Gordon phenotype;
         DIAGNOSTIC PITFALL: standard exon sequencing MISSES; needs CNV/long-range PCR;
         seed SEED_BASE+4) ·
KLHL3  (Kelch-like 3; 587 aa; 5q31.2; AD/AR;
         Gordon syndrome PHA2C; substrate adaptor of CUL3-RING E3 ligase →
         normally ubiquitinates WNK1/4 for proteasomal degradation;
         AD LOF missense: kelch domain (WNK-binding) or BTB domain (CUL3-binding) → partial Gordon;
         AR biallelic: complete loss → severe Gordon;
         seed SEED_BASE+5) ·
CUL3   (Cullin 3; 768 aa; 2q36.2; AD;
         Gordon syndrome PHA2E; exon 9 skipping (in-frame Δexon9 / c.1221+3A>G or similar) →
         dominant-negative CUL3 isoform → entire CUL3 E3 complex dysfunctional →
         WNK1/4 NOT degraded → most SEVERE Gordon phenotype;
         de novo dominant mutations common; short stature; some patients have features of CUL3-related
         neurodevelopmental syndrome (autism, epilepsy) — SAME gene, different exon deletions;
         seed SEED_BASE+6) ·
NR3C2  (mineralocorticoid receptor; 984 aa; 4q31.23; AD GOF;
         Geller syndrome; activating mutation S810L (or related) in ligand-binding domain →
         progesterone (and other steroids without 21-OH group) becomes full agonist;
         PATHOGNOMONIC: HTN MARKEDLY WORSENED IN PREGNANCY (progesterone surge);
         early-onset HTN in non-pregnant patients (spironolactone also acts as agonist → WORSENS HTN);
         spironolactone ABSOLUTE CI in Geller;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2934–2941)
"""
import random

SEED_BASE = 2934

ATLAS_GENES = [
    {
        "gene": "HSD11B2",
        "protein": (
            "HSD11B2 -- 11q22.1 AR -- 405aa -- 11beta-Hydroxysteroid-Dehydrogenase-Type-2-"
            "44kDa-Renal-Cortisol-Inactivator-AME-Apparent-Mineralocorticoid-Excess-"
            "PATHOGNOMONIC-THF-5alphaTHF-THE-Ratio-gt10-"
            "OMIM-Gene-614232-Disease-OMIM-218030"
        ),
        "locus": "11q22.1",
        "protein_size": (
            "405 aa / 44 kDa (HSD11B2 — 11β-hydroxysteroid dehydrogenase type 2; "
            "NAD+-dependent oxidoreductase; ER-bound; "
            "FUNCTION: converts active cortisol (F, compound F) → inactive cortisone (E) in kidney; "
            "CRITICAL CONCEPT: "
            "  Cortisol binds mineralocorticoid receptor (MR) with EQUAL affinity to aldosterone; "
            "  Serum cortisol 1000× higher than aldosterone — without HSD11B2, kidney MR would be saturated by cortisol; "
            "  HSD11B2 in DCT/collecting duct creates 'cortisol-free zone' for MR → aldosterone-selective signalling; "
            "AME LOF MECHANISM: "
            "  HSD11B2 loss → cortisol accumulates in kidney → MR constitutively activated by cortisol; "
            "  Aldosterone undetectable (suppressed by volume), renin undetectable; "
            "  Clinically: severe early-onset hypertension + profound hypokalaemia + low aldosterone + low renin; "
            "LIQUORICE/CARBENOXOLONE PHENOCOPY: "
            "  Glycyrrhizinic acid (liquorice root) → inhibits HSD11B2 → acquired AME; "
            "  Important differential — inquire about liquorice/herbal supplement use; "
            "encoded 11q22.1"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) biallelic — HSD11B2 AME: "
            "  Biallelic LOF mutations → complete or near-complete HSD11B2 deficiency; "
            "  Compound heterozygous or homozygous; "
            "CLINICAL FEATURES: "
            "    Severe early-onset HTN (neonatal to childhood); low birth weight; failure to thrive; "
            "    Profound hypokalaemia (K+ <2.5 mmol/L); "
            "    Metabolic alkalosis; "
            "    Low/suppressed aldosterone; low/suppressed renin; "
            "    No features of CAH (no virilisation, no ambiguous genitalia); "
            "DIAGNOSIS: "
            "    Urinary steroid metabolite ratio: (THF + 5αTHF) / THE >10 — PATHOGNOMONIC; "
            "      THF = tetrahydrocortisol, 5αTHF = 5α-tetrahydrocortisol (cortisol metabolites); "
            "      THE = tetrahydrocortisone (cortisone metabolite); "
            "    Normal ratio = 0.5–1.5; AME >10 (often >20); "
            "    24h urine HPLC steroid profile (GC-MS) — specialist test; "
            "TREATMENT: "
            "    Dexamethasone (suppresses ACTH → reduces cortisol substrate); "
            "    Amiloride or triamterene (blocks ENaC — same target as Liddle, different mechanism); "
            "    Renal transplant: curative (new kidney has functional HSD11B2)"
        ),
        "disease_category": (
            "APPARENT MINERALOCORTICOID EXCESS (AME) — CORTISOL-MEDIATED MR ACTIVATION: "
            "  KEY LABORATORY FINGERPRINT: "
            "    Hypertension + hypokalaemia + metabolic alkalosis + LOW/UNDETECTABLE aldosterone + LOW/UNDETECTABLE renin; "
            "    (ALL other mineralocorticoid causes: aldosterone is ELEVATED); "
            "    Urinary THF+5αTHF/THE ratio >10 — confirms HSD11B2 deficiency; "
            "  SEVERITY SPECTRUM: "
            "    Complete LOF: severe neonatal/infantile HTN; cerebral complications; "
            "    Partial: milder, later-onset, partial biochemical abnormalities; "
            "    Heterozygous carriers: mild ratio elevation; no clinical disease; "
            "  COMPLICATIONS: "
            "    Stroke, cardiac hypertrophy, renal damage from severe early-onset HTN; "
            "    Nephrocalcinosis (hypokalaemic nephropathy); "
            "  DIFFERENTIAL: "
            "    Exogenous liquorice/carbenoxolone: identical biochemistry → history critical; "
            "    Cushing syndrome: cortisol ELEVATED (not just shifted to kidney); cortisone/cortisol ratio NORMAL; "
            "    11β-hydroxylase deficiency (CYP11B1): virilisation present; steroid profile different"
        ),
        "disease_pathway": (
            "HSD11B2 LOF → CORTISOL SATURATES KIDNEY MR → AME: "
            "  Normal: cortisol (F) → HSD11B2 → cortisone (E) in kidney; "
            "    MR in DCT/collecting duct: sees low cortisol + physiological aldosterone → appropriate Na+ reabsorption; "
            "  AME: HSD11B2 absent → F accumulates → F binds MR (equal affinity to aldosterone); "
            "    MR activation: increases ENaC (apical) + Na/K-ATPase (basolateral) → Na+ reabsorption ↑; "
            "    Volume expansion → renin suppressed → angiotensin I/II suppressed → aldosterone suppressed; "
            "    K+ wasting via ROMK (renal outer medullary K+ channel): hypokalaemia; "
            "  PATHOGNOMONIC RATIO: "
            "    Normal: cortisol effectively converted to cortisone → THF ratio ~0.7; "
            "    AME: conversion fails → ratio >10 (cortisol metabolites >> cortisone metabolites); "
            "  TREATMENT RATIONALE: "
            "    Dexamethasone: no HSD11B2 target → replaces cortisol with weaker ACTH suppressor → ↓ cortisol substrate; "
            "    Amiloride: ENaC blocker → directly counters MR-driven Na+ retention"
        ),
    },
    {
        "gene": "SCNN1B",
        "protein": (
            "SCNN1B -- 16p12.2 AD-GOF -- 640aa -- ENaC-Beta-Subunit-"
            "72kDa-Liddle-Syndrome-PY-Motif-NEDD4-2-Binding-Lost-"
            "ENaC-Constitutively-Open-Amiloride-Triamterene-CURATIVE-"
            "OMIM-Gene-600760-Disease-OMIM-177200"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "640 aa / 72 kDa (SCNN1B — sodium channel non-voltage-gated 1 beta; "
            "ENaC β subunit; "
            "STRUCTURE: short N-terminal intracellular domain — 2 TM helices — large extracellular loop — short C-terminal; "
            "KEY DOMAIN: C-terminal PY motif (PPPXY at aa 615-620) — binds NEDD4-2 WW domain for ubiquitination; "
            "FUNCTION: "
            "  ENaC = αβγ heterotrimer at apical membrane of DCT/cortical collecting duct; "
            "  Rate-limiting step for Na+ reabsorption in aldosterone-sensitive distal nephron; "
            "  NEDD4-2 (E3 ubiquitin ligase) binds PY motif → ubiquitinates ENaC → internalisation from apical membrane; "
            "  Aldosterone → phosphorylates/inhibits NEDD4-2 → more ENaC at membrane → more Na+ transport; "
            "GOF MECHANISM: "
            "  PY motif mutation (truncation or missense) → NEDD4-2 cannot bind → ENaC not ubiquitinated → "
            "    ENaC constitutively at apical membrane → constitutive Na+ reabsorption; "
            "  Same effect as maximal aldosterone stimulation but INDEPENDENT of aldosterone; "
            "encoded 16p12.2 (adjacent to SCNN1G)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GAIN-OF-FUNCTION — SCNN1B Liddle Syndrome: "
            "  Heterozygous PY-motif mutation (truncation or missense R564X, P616L, P616R etc.); "
            "  De novo mutations possible but familial cases more common; "
            "  Penetrance: high (>90%); expressivity variable; "
            "CLINICAL FEATURES: "
            "    Early-onset HTN (childhood/adolescence); "
            "    Hypokalaemia (K+ 2.5-3.2 mmol/L); "
            "    Metabolic alkalosis; "
            "    LOW plasma aldosterone; LOW plasma renin (suppressed by volume); "
            "    CLINICALLY IDENTICAL to HSD11B2 AME — only biochemical ratio and genetics distinguish; "
            "DIAGNOSIS: "
            "    HTN + hypokalaemia + low aldosterone + low renin → DISTINGUISH from primary aldosteronism; "
            "    ENaC gene panel (SCNN1B + SCNN1G) + HSD11B2; "
            "    THF/THE ratio: NORMAL in Liddle (HSD11B2 intact) — distinguishes from AME; "
            "TREATMENT: "
            "    Amiloride (preferred) or triamterene: directly block ENaC channel → CURATIVE; "
            "    Spironolactone/eplerenone: INEFFECTIVE (MR-independent mechanism — ENaC open regardless of MR); "
            "    Low sodium diet + amiloride: normalises BP + K+"
        ),
        "disease_category": (
            "LIDDLE SYNDROME — ENaC BETA SUBUNIT GOF: "
            "  MOLECULAR MECHANISM: PY motif loss → ENaC permanently open; "
            "  TREATMENT KEY: "
            "    Amiloride/triamterene = ENaC blockers → directly curative; "
            "    Spironolactone = MR blocker → INEFFECTIVE (ENaC activated by structural change, not by MR); "
            "  LABORATORY PROFILE: "
            "    HTN + hypokalaemia + metabolic alkalosis + LOW aldosterone + LOW renin; "
            "    Normal THF/THE ratio (HSD11B2 intact) — critical AME vs Liddle discriminator; "
            "  FAMILY SCREENING: "
            "    First-degree relatives with early-onset HTN or hypokalaemia → ENaC panel; "
            "  RENAL TRANSPLANT: curative (donor kidney provides normal ENaC regulation); "
            "  SCNN1B vs SCNN1G: identical phenotype — both tested together; "
            "    SCNN1B ~60% of Liddle cases, SCNN1G ~40%"
        ),
        "disease_pathway": (
            "SCNN1B GOF → CONSTITUTIVE ENaC OPEN → Na RETENTION: "
            "  Normal ENaC regulation: "
            "    Aldosterone → SGK1 → phosphorylates NEDD4-2 → NEDD4-2 sequesters in cytoplasm → less ubiquitination → "
            "      ENaC stays at apical membrane longer; "
            "    Without aldosterone: NEDD4-2 active → binds PY motif → ubiquitinates ENaC → internalises (reduces surface ENaC); "
            "  SCNN1B PY-motif GOF: "
            "    NEDD4-2 cannot bind → ubiquitination BLOCKED → ENaC PERMANENTLY at apical membrane; "
            "    Constitutive Na+ absorption in DCT/collecting duct; "
            "    Volume expansion → AT-II suppressed → aldosterone FALLS (feedback); "
            "    K+ wasting via ROMK (lumen-negative potential drives K+ secretion); "
            "    Aldosterone low but ENaC open → NO RESPONSE to spironolactone; "
            "    Amiloride: binds ENaC pore directly → blocks Na+ entry regardless of PY motif status"
        ),
    },
    {
        "gene": "SCNN1G",
        "protein": (
            "SCNN1G -- 16p12.2 AD-GOF -- 649aa -- ENaC-Gamma-Subunit-"
            "74kDa-Liddle-Syndrome-Same-Phenotype-SCNN1B-PY-Motif-Truncation-"
            "Adjacent-SCNN1B-16p12-2-Tested-Together-Amiloride-CURATIVE-"
            "OMIM-Gene-600761-Disease-OMIM-177200"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "649 aa / 74 kDa (SCNN1G — sodium channel non-voltage-gated 1 gamma; "
            "ENaC γ subunit; "
            "STRUCTURE: identical topology to SCNN1B — 2 TM helices + large extracellular loop + PY motif C-terminal; "
            "PY MOTIF: PPPXY at aa 635-639 — NEDD4-2 binding site; same mechanism as SCNN1B; "
            "GENE LOCATION: 16p12.2 — immediately adjacent to SCNN1B (same chromosomal region); "
            "FUNCTION: "
            "  γ subunit contributes to ENaC channel assembly and gating; "
            "  α-γ interaction required for functional channel; "
            "  β and γ subunits both regulate trafficking via NEDD4-2; "
            "GOF: PY motif truncation (most common) → same loss of NEDD4-2 binding as SCNN1B; "
            "  Some SCNN1G mutations: large deletions including C-terminal → even more severe; "
            "encoded 16p12.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GAIN-OF-FUNCTION — SCNN1G Liddle Syndrome: "
            "  Heterozygous: C-terminal truncation (STOP mutation) or PY-motif missense; "
            "  Most are familial; de novo less common than SCNN1B; "
            "PHENOTYPE: IDENTICAL to SCNN1B Liddle: "
            "    HTN + hypokalaemia + low aldosterone + low renin; "
            "    Clinical severity variable within families; "
            "    Some compound heterozygous SCNN1B+SCNN1G reported (more severe); "
            "DIAGNOSIS: "
            "    Both SCNN1B + SCNN1G sequenced simultaneously on ENaC panel; "
            "    SCNN1G: C-terminal deletion (stop codon) most common → often detected by sequencing; "
            "    Rare: promoter variants or splicing affecting SCNN1G; "
            "TREATMENT: identical to SCNN1B — amiloride or triamterene; "
            "FAMILY HISTORY: carefully screen for HTN + hypokalaemia in parents/siblings; "
            "    Cascade testing recommended for all first-degree relatives"
        ),
        "disease_category": (
            "LIDDLE SYNDROME — ENaC GAMMA SUBUNIT GOF: "
            "  Identical disease mechanism to SCNN1B — differentiated only by gene sequencing; "
            "  KEY CLINICAL POINTS: "
            "    SCNN1G accounts for ~40% of molecularly confirmed Liddle cases; "
            "    Both SCNN1B + SCNN1G always tested together — single panel; "
            "  PROGNOSIS: excellent with amiloride; "
            "    Untreated: left ventricular hypertrophy, CKD, stroke from severe HTN; "
            "    Treated: BP normalises, K+ corrects, aldosterone recovers; "
            "  PREGNANCY: amiloride should be stopped in first trimester (teratogenic potential); "
            "    Triamterene: limited safety data; manage with Na restriction + monitoring; "
            "    RISK: Geller NR3C2 worsens in pregnancy; Liddle does not worsen per se (aldosterone-independent)"
        ),
        "disease_pathway": (
            "SCNN1G GOF → CONSTITUTIVE ENaC OPEN — identical to SCNN1B: "
            "  ENaC αβγ heterotrimer: all three subunits required for surface expression; "
            "  γ-subunit PY motif: NEDD4-2 ubiquitinates both β and γ C-termini; "
            "    GOF in γ alone: partial NEDD4-2 engagement (still binds β) → less severe than biallelic loss; "
            "    Compound SCNN1B+SCNN1G: NEDD4-2 cannot bind either β or γ → maximum ENaC surface expression; "
            "  Volume retention sequence: same as SCNN1B → volume expansion → renin suppression → aldosterone suppression; "
            "  Therapeutic target: amiloride pore-block → Na+ entry stopped regardless of NEDD4-2 status"
        ),
    },
    {
        "gene": "WNK4",
        "protein": (
            "WNK4 -- 17q21.31 AD -- 1243aa -- WNK-Lysine-Deficient-Protein-Kinase-4-"
            "135kDa-Gordon-Syndrome-PHA2B-NCC-Overcorrection-"
            "THIAZIDE-CURATIVE-PATHOGNOMONIC-Hyperkalemia-Hyperchloraemic-Acidosis-HTN-Normal-High-Aldosterone-"
            "OMIM-Gene-601844-Disease-OMIM-614491"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1243 aa / 135 kDa (WNK4 — WNK lysine-deficient protein kinase 4; "
            "Serine/threonine kinase; "
            "DOMAINS: N-terminal kinase domain (catalytic) + proline-rich regions + KLHL3/CUL3-binding acidic motif; "
            "NORMAL FUNCTION: "
            "  WNK4 phosphorylates and activates OXSR1 and SPAK kinases; "
            "  OXSR1/SPAK → phosphorylate (activate) NCC (SLC12A3 = Na/Cl cotransporter, DCT); "
            "  WNK4 ALSO phosphorylates and inhibits NCC directly in some contexts (dual role); "
            "  KEY: WNK4 activity is balanced — net effect controlled by KLHL3/CUL3 degradation; "
            "GORDON SYNDROME MECHANISM (GOF): "
            "  Acidic motif missense (Q562E, D564A, E562K etc.) → cannot bind KLHL3 → not ubiquitinated → "
            "    WNK4 accumulates → excessive OXSR1/SPAK activation → NCC hyperphosphorylation → "
            "    NCC constitutively active in DCT → Na+/Cl- retention; "
            "  Na+ retention → volume expansion → aldosterone suppressed (initially); "
            "  BUT: NCC is DOWNSTREAM of aldosterone regulation → thiazide BLOCKS NCC → CURES; "
            "encoded 17q21.31"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — WNK4 Gordon syndrome PHA2B: "
            "  Heterozygous missense in KLHL3-binding acidic motif; "
            "  Phenotype: moderate-severe Gordon; "
            "CLINICAL FEATURES — PATHOGNOMONIC BIOCHEMICAL TRIAD: "
            "    Hyperkalemia (K+ 5.5-7.5 mmol/L) — CARDINAL; "
            "    Hyperchloraemic metabolic acidosis (Cl- elevated, HCO3- low); "
            "    Hypertension; "
            "    Aldosterone: NORMAL or ELEVATED (distinguishes Gordon from Liddle/AME where aldosterone LOW); "
            "    Renin: LOW (suppressed by volume); "
            "    Paradox: hyperkalemia despite aldosterone normal/elevated — NCC blocks K+ secretory stimulus; "
            "DIAGNOSIS: "
            "    Hyperkalemia + HTN + normal/raised aldosterone + low renin = Gordon phenotype; "
            "    WNK4 + WNK1 + KLHL3 + CUL3 panel; "
            "    Exclude secondary causes: ACEi/ARBs/NSAIDs/aldosterone resistance; "
            "TREATMENT: "
            "    Thiazide diuretic (hydrochlorothiazide/chlorthalidone) — directly blocks NCC — CURATIVE; "
            "    K+ normalises + BP normalises + acidosis corrects on thiazide; "
            "    Dose: low-dose thiazide often sufficient"
        ),
        "disease_category": (
            "GORDON SYNDROME (PHAII/FHH PHA2B) — WNK4 NCC HYPERACTIVATION: "
            "  INVERSE of Gitelman syndrome (SLC12A3 LOF): "
            "    Gitelman: NCC absent → salt-wasting + hypokalaemia; "
            "    Gordon: NCC constitutive → salt-retention + hyperkalemia; "
            "  ALDOSTERONE PARADOX: "
            "    NCC absorbs Na+/Cl- in DCT, BEFORE aldosterone-sensitive collecting duct; "
            "    NCC overactivity → reduced Na+ delivery to collecting duct → less aldosterone effect → less K+ secretion; "
            "    RESULT: aldosterone normal/elevated BUT K+ still rises (delivery-limited K+ secretion); "
            "  THIAZIDE MECHANISM: "
            "    Blocks NCC → Na+ delivery to collecting duct increases → K+ secretion restored + volume reduced; "
            "  DIFFERENTIAL DIAGNOSIS: "
            "    Aldosterone resistance (PHA1, PHA2): also hyperkalemia + HTN BUT aldosterone very high; "
            "    Aldosterone-deficiency (Addison): hyperkalemia + low Na, LOW aldosterone; "
            "    Gordon: HTN + hyperkalemia + NORMAL aldosterone (or slightly elevated)"
        ),
        "disease_pathway": (
            "WNK4 GOF → NCC OVERACTIVATION → GORDON SYNDROME: "
            "  Pathway: "
            "    WNK4 (mutant) → not degraded by KLHL3/CUL3 → accumulates → activates OXSR1 → activates SPAK; "
            "    SPAK phosphorylates NCC (T55, T58 residues) → NCC fully active in DCT apical membrane; "
            "    NCC: absorbs Na+ + Cl- from tubular lumen → volume expansion; "
            "    Reduced luminal Na+ delivery to connecting tubule/collecting duct → "
            "      less lumen-negative potential → less K+ secretion via ROMK → hyperkalemia; "
            "    Cl- reabsorption excess → hyperchloraemia; "
            "    H+ secretion in alpha-intercalated cells: somewhat impaired → metabolic acidosis; "
            "  THIAZIDE CURE: "
            "    Hydrochlorothiazide/chlorthalidone → binds NCC Cl- binding site → blocks Na+/Cl- cotransport; "
            "    Na+ delivered downstream → K+ secretion restores → K+ normalises; "
            "    Volume corrects → BP normalises; "
            "    Metabolic acidosis corrects"
        ),
    },
    {
        "gene": "WNK1",
        "protein": (
            "WNK1 -- 12p13.33 AD -- 2382aa -- WNK-Lysine-Deficient-Protein-Kinase-1-"
            "251kDa-Gordon-Syndrome-PHA2A-Large-Intronic-Deletion-Intron1-"
            "DIAGNOSTIC-PITFALL-Standard-Exon-Sequencing-MISSES-CNV-Required-"
            "OMIM-Gene-605232-Disease-OMIM-145260"
        ),
        "locus": "12p13.33",
        "protein_size": (
            "2382 aa / 251 kDa (WNK1 — WNK lysine-deficient protein kinase 1; "
            "Largest WNK family member; ubiquitously expressed + kidney-specific short isoform (KS-WNK1); "
            "DOMAINS: kinase domain (N-terminal, catalytic) + KLHL3-binding acidic motif + coiled-coil domains; "
            "TWO ISOFORMS: "
            "  L-WNK1 (long, full-length): ubiquitous; activates OXSR1/SPAK → NCC + NKCC2 activation; "
            "  KS-WNK1 (kidney-specific, short): lacks kinase domain; kidney DCT-specific; "
            "    Antagonises L-WNK1 (kinase-dead dominant inhibitor); "
            "    Controls NCC activity in kidney; "
            "GORDON MECHANISM (PHA2A): "
            "  LARGE INTRONIC DELETIONS in intron 1 (up to 41 kb) → "
            "    Disrupts KS-WNK1 promoter → KS-WNK1 REDUCED; "
            "    L-WNK1 becomes dominant → unopposed NCC activation → Gordon phenotype; "
            "  NOT a missense/coding mutation — MUST use CNV analysis / long-range PCR; "
            "encoded 12p13.33"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — WNK1 Gordon syndrome PHA2A: "
            "  LARGE INTRONIC DELETIONS (intron 1) — typically 41 kb deletion removing KS-WNK1 regulatory elements; "
            "  DIAGNOSTIC PITFALL: "
            "    Standard exon sequencing (Sanger / gene panel coding regions) — MISSES intronic deletion; "
            "    Must request: CNV analysis, MLPA, array-CGH, or long-range PCR spanning WNK1 intron 1; "
            "  PHA2A families: original Gordon/Paver/Sheard 1969 families — mild-to-moderate phenotype; "
            "CLINICAL: identical Gordon triad (hyperkalemia + acidosis + HTN); "
            "    Often milder than WNK4 or CUL3 forms; "
            "    Some asymptomatic relatives with only biochemical abnormalities; "
            "TREATMENT: thiazide diuretic — identical to WNK4 Gordon; "
            "MOLECULAR TESTING NOTE: "
            "    WNK1 intronic deletion should always be specifically requested in Gordon evaluation; "
            "    'Gordon panel negative' on exon sequencing still warrants WNK1 CNV testing"
        ),
        "disease_category": (
            "GORDON SYNDROME PHA2A — WNK1 INTRONIC DELETION — NCC OVERACTIVATION: "
            "  MECHANISM SUMMARY: "
            "    KS-WNK1 (kidney-specific short isoform, kinase-dead) normally dominates in DCT; "
            "    KS-WNK1 competes with L-WNK1 for OXSR1/SPAK binding → net NCC inhibition; "
            "    Intronic deletion → KS-WNK1 reduced → L-WNK1 unopposed → NCC hyperphosphorylated; "
            "  PHENOTYPIC SEVERITY: milder than CUL3 Gordon; similar to WNK4; "
            "  DIAGNOSTIC ALGORITHM for Gordon: "
            "    1. WNK4 + KLHL3 (most common by exon sequencing); "
            "    2. CUL3 exon 9 deletion analysis; "
            "    3. WNK1 intron 1 CNV (if 1+2 negative); "
            "  KEY LESSON: "
            "    Large intronic deletions (not point mutations) cause a significant proportion of Gordon PHA2A; "
            "    This is the prototype disease for intronic-deletion-only inheritance"
        ),
        "disease_pathway": (
            "WNK1 INTRONIC DELETION → KS-WNK1 LOSS → L-WNK1 DOMINANT → GORDON: "
            "  Normal DCT WNK balance: "
            "    KS-WNK1 (no kinase) competes with L-WNK1 for SPAK/OXSR1 coiled-coil binding; "
            "    Aldosterone: stimulates L-WNK1 → NCC phosphorylation → Na+ reabsorption (physiological); "
            "    Without aldosterone: KS-WNK1 dominant → SPAK less active → NCC dephosphorylated → Na+ not reabsorbed; "
            "  PHA2A: KS-WNK1 promoter in intron 1 deleted → KS-WNK1 mRNA ↓ → less antagonism of L-WNK1; "
            "    L-WNK1-OXSR1/SPAK-NCC pathway constitutively active; "
            "    Downstream: same Gordon biochemistry as WNK4: "
            "      Na+/Cl- retention → volume ↑ → renin ↓ → aldosterone variable; "
            "      Reduced Na+ delivery to collecting duct → K+ retention → hyperkalemia; "
            "    Thiazide: blocks NCC → restores K+ secretion → cures biochemistry"
        ),
    },
    {
        "gene": "KLHL3",
        "protein": (
            "KLHL3 -- 5q31.2 AD-AR -- 587aa -- Kelch-Like-3-"
            "66kDa-CUL3-RING-E3-Substrate-Adaptor-WNK1-WNK4-Ubiquitination-"
            "Gordon-Syndrome-PHA2C-AD-Missense-AR-Biallelic-More-Severe-"
            "OMIM-Gene-605775-Disease-OMIM-614495"
        ),
        "locus": "5q31.2",
        "protein_size": (
            "587 aa / 66 kDa (KLHL3 — Kelch-like family member 3; "
            "STRUCTURE: BTB domain (N-terminal, CUL3-binding) + BACK domain + 6 Kelch repeats (C-terminal β-propeller, substrate-binding); "
            "FUNCTION: "
            "  Substrate adaptor of CUL3-RING E3 ubiquitin ligase complex; "
            "  BTB domain: docks on CUL3; "
            "  Kelch β-propeller: binds WNK1/WNK4 acidic motif (QDEPEGP) → presents substrate to E3 complex; "
            "  CUL3-RING ligase → K48 ubiquitin chain on WNK → proteasomal degradation; "
            "  Without KLHL3: WNK1/4 accumulate → NCC overactive → Gordon; "
            "MUTATION SPECTRUM: "
            "  AD (missense): kelch domain mutations → cannot bind WNK acidic motif; "
            "    BTB domain mutations → cannot bind CUL3; "
            "  AR (biallelic): complete loss of KLHL3 → severe Gordon; "
            "  AD mutations: partial loss → milder phenotype (one functional allele retains partial degradation); "
            "encoded 5q31.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (missense, haploinsufficiency-like) or AUTOSOMAL RECESSIVE (biallelic) — KLHL3: "
            "  AD: single KLHL3 missense (Kelch or BTB domain) → partial WNK degradation impaired → mild-moderate Gordon; "
            "  AR: biallelic loss → complete KLHL3 absence → severe Gordon (severe hyperkalemia); "
            "  Most common Gordon gene (after WNK4); accounts for ~20-30% of Gordon families; "
            "CLINICAL: "
            "    AD: usually milder than CUL3 Gordon; "
            "    AR: severe; may have low aldosterone (volume-mediated suppression); "
            "    All: hyperkalemia + HTN + metabolic acidosis; "
            "TREATMENT: thiazide diuretic — same as all Gordon forms; "
            "    AR biallelic: may need higher thiazide dose; "
            "PHOSPHORYLATION MARKER: "
            "    Urinary phospho-NCC (pNCC): research assay; elevated in Gordon, normal in Liddle/AME; "
            "FAMILY SCREENING: "
            "    AD KLHL3: screen all first-degree relatives; "
            "    AR KLHL3: parents are obligate carriers; siblings 25% risk"
        ),
        "disease_category": (
            "GORDON SYNDROME PHA2C — KLHL3 E3 LIGASE ADAPTOR LOSS: "
            "  PATHWAY SUMMARY: "
            "    Normal: KLHL3-BTB → CUL3 + KLHL3-Kelch → WNK1/4 acidic motif → ubiquitination → WNK degraded; "
            "    KLHL3 LOF: WNK1/4 not degraded → accumulate → NCC hyperphosphorylation → Gordon; "
            "  AD vs AR SEVERITY: "
            "    AD missense: one functional allele → partial degradation → milder biochemistry; "
            "    AR biallelic: no degradation → severe; "
            "  RELATED GENE: CUL3 (Gordon PHA2E) — same E3 complex, different component; "
            "    KLHL3 mutation: substrate adaptor lost (only WNK-related substrates affected); "
            "    CUL3 mutation: entire CUL3 complex disrupted (multiple substrates affected → more severe + extra-renal features); "
            "  PHOSPHO-NCC as biomarker: elevated in Gordon regardless of which gene; "
            "    Research use: distinguishes NCC-mediated from non-NCC-mediated causes"
        ),
        "disease_pathway": (
            "KLHL3 LOF → WNK ACCUMULATION → NCC OVERACTIVATION → GORDON: "
            "  E3 ubiquitin ligase complex assembly: "
            "    CUL3 scaffold + RBX1 (RING) + KLHL3 (adaptor); "
            "    KLHL3 kelch-repeat β-propeller: binds WNK1 or WNK4 QDEPEGP motif; "
            "    CUL3-RBX1: transfers K48 ubiquitin to WNK1/4 lysines → 26S proteasome → WNK degraded; "
            "  KLHL3 LOF: "
            "    Kelch missense: WNK binding lost → WNK not recruited → not ubiquitinated → accumulates; "
            "    BTB missense: CUL3 binding lost → no E3 complex forms → same outcome; "
            "    AR biallelic: both alleles fail → maximal WNK accumulation; "
            "  WNK1/4 accumulation → SPAK/OXSR1 overactivation → NCC phospho-T53/T58 maintained; "
            "  NCC constitutively active → DCT Na+/Cl- absorption → Gordon biochemistry"
        ),
    },
    {
        "gene": "CUL3",
        "protein": (
            "CUL3 -- 2q36.2 AD -- 768aa -- Cullin-3-"
            "89kDa-RING-E3-Ubiquitin-Ligase-Scaffold-Gordon-Syndrome-PHA2E-"
            "EXON9-SKIP-Dominant-Negative-Most-Severe-Short-Stature-"
            "Neurodevelopmental-Comorbidity-Autism-Epilepsy-Different-Deletions-"
            "OMIM-Gene-603136-Disease-OMIM-614496"
        ),
        "locus": "2q36.2",
        "protein_size": (
            "768 aa / 89 kDa (CUL3 — cullin 3; "
            "Scaffold protein of CUL3-RING E3 ubiquitin ligase family; "
            "DOMAINS: N-terminal domain (NTD, aa 1-388) — BTB adaptor binding; C-terminal domain (CTD) — RBX1 binding + ubiquitin transfer; "
            "FUNCTION: "
            "  Forms CUL3-RBX1 (RING) + BTB-adaptor (KLHL3 for WNK) complex; "
            "  CUL3 is the scaffold connecting substrate adaptor to ubiquitin transfer machinery; "
            "  Substrates: WNK1/4 (kidney), KEAP1/NRF2 (redox), other BTB adaptors for diverse substrates; "
            "GORDON MECHANISM: "
            "  In-frame exon 9 deletion (Δexon9) — most common Gordon CUL3 mutation; "
            "  Δexon9 CUL3: "
            "    Retains KLHL3 (BTB) binding domain but CANNOT load RBX1 properly; "
            "    Dominant-negative: mutant CUL3 sequesters KLHL3 but cannot ubiquitinate WNK → "
            "      ALL CUL3 complexes disrupted (WT CUL3 also titrated out by KLHL3 competition); "
            "    SEVERE: disrupts ALL CUL3 substrates (multiple BTB adaptors) → extra-renal features; "
            "encoded 2q36.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (dominant negative) — CUL3 Gordon syndrome PHA2E: "
            "  Exon 9 in-frame deletion (Δexon9): dominant negative → most severe Gordon; "
            "  Frequently DE NOVO; "
            "CLINICAL FEATURES: "
            "    Most severe Gordon phenotype (hyperkalemia often >7 mmol/L at presentation); "
            "    Short stature (growth impairment — CUL3 affects multiple substrates); "
            "    Intellectual disability / autism spectrum: in some patients (depends on deletion extent); "
            "    Thiazide-responsive HTN + hyperkalemia; "
            "    Cardiac arrhythmia risk from severe hyperkalemia; "
            "NEURODEVELOPMENTAL COMORBIDITY: "
            "    Different CUL3 mutations (other exon deletions, not Δexon9) → autism/epilepsy WITHOUT Gordon; "
            "    Δexon9 specifically causes Gordon PHA2E; "
            "    CRITICAL: same gene, different mutations → very different diseases; "
            "TREATMENT: "
            "    Thiazide — CURATIVE for Gordon biochemistry; "
            "    Dose: may need higher dose (dominant negative — more severe WNK accumulation); "
            "    Monitor for neurodevelopmental co-morbidities; "
            "EMERGENCY: severe hyperkalemia (K+ >7) → cardiac monitoring + IV dextrose/insulin + kayexalate"
        ),
        "disease_category": (
            "GORDON SYNDROME PHA2E — CUL3 DOMINANT NEGATIVE: "
            "  SEVERITY: most severe Gordon phenotype (Δexon9 dominant negative); "
            "  MECHANISM: Δexon9 CUL3 sequesters KLHL3 → ALL CUL3-RING E3 complexes disrupted; "
            "    Not just WNK1/4 affected → KEAP1/NRF2 + other BTB substrates → extra-renal features; "
            "  DE NOVO: frequently sporadic → no family history; "
            "    Genetic testing essential for any child with severe HTN + hyperkalemia; "
            "  HYPERKALEMIA EMERGENCY: "
            "    K+ >7 mmol/L → treat as cardiac emergency; "
            "    IV calcium gluconate (stabilises cardiac membrane) + insulin-dextrose + bicarbonate; "
            "    Thiazide: start ASAP to prevent recurrence; "
            "  EXTRA-RENAL FEATURES: "
            "    Short stature: growth monitoring + GH evaluation; "
            "    Neurodevelopmental: autism/epilepsy in some — multidisciplinary care"
        ),
        "disease_pathway": (
            "CUL3 Δexon9 → DOMINANT NEGATIVE → ALL CUL3-RING E3 SUBSTRATES DISRUPTED → SEVERE GORDON: "
            "  Δexon9 CUL3 protein: "
            "    NTD (BTB-binding) intact → STILL BINDS KLHL3 (and other BTB adaptors); "
            "    CTD truncated → CANNOT bind RBX1 properly → NO ubiquitin transfer; "
            "  Dominant negative mechanism: "
            "    Mutant Δexon9 CUL3 + WT CUL3 both compete for KLHL3 (BTB adaptor pool limited); "
            "    Mutant complex: binds KLHL3 + WNK but CANNOT ubiquitinate → sequesters KLHL3; "
            "    WT CUL3: KLHL3-starved (competing with mutant) → fewer functional complexes; "
            "    Net: dramatic reduction in WNK1/4 ubiquitination → maximal WNK accumulation; "
            "  Multiple BTB adaptors affected: "
            "    SPOP, KEAP1, BTBD9 etc. also use CUL3 — their substrates also accumulate; "
            "    Explains extra-renal phenotype (growth, neurodevelopment); "
            "  Thiazide still curative at renal level (NCC blockade) despite upstream accumulation"
        ),
    },
    {
        "gene": "NR3C2",
        "protein": (
            "NR3C2 -- 4q31.23 AD-GOF -- 984aa -- Mineralocorticoid-Receptor-"
            "107kDa-Geller-Syndrome-S810L-Progesterone-Full-Agonist-"
            "PATHOGNOMONIC-HTN-Markedly-Worsened-Pregnancy-"
            "SPIRONOLACTONE-ABSOLUTE-CI-Agonist-Worsens-HTN-"
            "OMIM-Gene-600983-Disease-OMIM-605115"
        ),
        "locus": "4q31.23",
        "protein_size": (
            "984 aa / 107 kDa (NR3C2 — nuclear receptor subfamily 3 group C member 2; "
            "Mineralocorticoid receptor (MR); "
            "DOMAINS: N-terminal A/B domain — DNA-binding domain (DBD, C domain) — hinge region — "
            "  ligand-binding domain (LBD, E domain, aa 672-984); "
            "NORMAL LIGAND SPECIFICITY: "
            "  Aldosterone: binds LBD with high affinity (Kd ~0.5 nM); "
            "  Cortisol: equal affinity to aldosterone (Kd ~0.5 nM) BUT normally inactivated by HSD11B2 in kidney; "
            "  Progesterone: binds LBD with ANTAGONIST activity normally → occupies LBD without activating; "
            "    Pregnancy: progesterone RISES to nmol/L → competes with aldosterone → MR partially BLOCKED → "
            "      physiological aldosterone escape during pregnancy; "
            "GELLER SYNDROME S810L MECHANISM: "
            "  Ser810Leu mutation in LBD helix 5 → structural rearrangement of LBD; "
            "  Progesterone (and other steroids without 21-OH group): now become FULL AGONISTS; "
            "    Pregnancy surge: progesterone 100-1000× normal → activates Geller MR → severe HTN; "
            "  Spironolactone: normally an MR antagonist BUT binds Geller S810L MR as agonist → worsens HTN; "
            "  17α-OH-progesterone, cortisone, others: also agonist at S810L MR; "
            "encoded 4q31.23"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GAIN-OF-FUNCTION — NR3C2 Geller Syndrome: "
            "  Heterozygous Ser810Leu (S810L) in MR LBD; "
            "  Other GOF mutations reported (S810F etc.); "
            "  Penetrance: essentially complete for HTN; "
            "CLINICAL FEATURES: "
            "    Early-onset HTN (childhood/adolescence, before pregnancy); "
            "    Hypokalaemia (variable, moderate); "
            "    Suppressed aldosterone; suppressed renin; "
            "    PATHOGNOMONIC: HTN MARKEDLY WORSENED IN PREGNANCY; "
            "      Progesterone surge activates Geller MR → BP crisis in pregnancy; "
            "      Normal pregnancy physiologically vasodilates (↓ BP) → Geller is OPPOSITE; "
            "SPIRONOLACTONE — ABSOLUTE CI: "
            "    Spironolactone (MR antagonist in normal MR) → full MR agonist in Geller → "
            "      WORSENS hypertension — potentially severe; "
            "    Eplerenone (more selective MR antagonist) — also reported to have agonist activity at S810L; "
            "    NEVER start spironolactone in Geller; "
            "TREATMENT: "
            "    Amiloride/triamterene: ENaC blockade (downstream of MR) — safe; "
            "    Calcium channel blockers, beta-blockers, ACEi/ARBs: general HTN management; "
            "    AVOID spironolactone and eplerenone"
        ),
        "disease_category": (
            "GELLER SYNDROME — MR S810L GOF: PROGESTERONE AS FULL MR AGONIST: "
            "  UNIQUE FEATURES: "
            "    Only mineralocorticoid HTN syndrome that WORSENS dramatically in pregnancy; "
            "    Spironolactone — ABSOLUTE CI (unique in all HTN syndromes); "
            "  DIAGNOSTIC ALGORITHM: "
            "    HTN + hypokalaemia + low aldosterone + low renin: "
            "      → screen for history of worsening in pregnancy (females); "
            "      → screen for any prescribed spironolactone causing HTN crisis; "
            "      → NR3C2 S810L sequencing; "
            "  PREGNANCY MANAGEMENT: "
            "    BP crisis risk: close monitoring from conception; "
            "    Methyldopa (safe in pregnancy, no MR agonism); "
            "    Labetalol: safe option; "
            "    Avoid: nifedipine if pre-eclampsia overlap risk; "
            "  MALE PATIENTS: "
            "    Less dramatic presentation (no pregnancy); "
            "    Early-onset HTN + hypokalaemia + low aldosterone → Geller in differential; "
            "  FAMILY SCREENING: AD → 50% offspring risk"
        ),
        "disease_pathway": (
            "NR3C2 S810L → PROGESTERONE FULL MR AGONIST → PREGNANCY HTN CRISIS: "
            "  Normal MR activation: "
            "    Aldosterone binds LBD → conformational change → H12 helix repositions → "
            "      coactivator recruitment (SRC-1, TIF2) → transcription of ENaC, Na/K-ATPase, SGK1; "
            "  Progesterone normal MR: "
            "    Binds LBD → H12 DOES NOT reposition correctly → coactivator binding impaired → "
            "      partial or no transcription → ANTAGONIST; "
            "    Pregnancy surge blocks aldosterone without triggering ENaC/SGK1 transcription; "
            "  S810L Geller MR: "
            "    Leu810 (in place of Ser810) → hydrophobic pocket created → "
            "      progesterone binding → H12 now repositions correctly → coactivator recruited → "
            "      full transcriptional activation; "
            "    Pregnancy: progesterone 100-fold rise → massive MR activation → maximal ENaC + SGK1 → "
            "      Na+ retention → BP crisis; "
            "  Spironolactone paradox: "
            "    Normally: spironolactone → binds LBD C-ring → H12 blocked → antagonist; "
            "    S810L: spironolactone → C-ring fits differently into Leu810 pocket → "
            "      H12 repositions → coactivator bound → agonist; "
            "    CLINICAL: starting spironolactone → activates Geller MR → severe HTN spike"
        ),
    },
]


# ── Patient simulation ────────────────────────────────────────────────────────
def _make_patients(seed: int, gene: str, n: int = 40) -> list:
    rng = random.Random(seed)

    genders = ["M", "F"]
    gene_params = {
        "HSD11B2": {"onset": (0, 36),   "sbp": (170, 220), "k_range": (1.5, 2.8), "aldost": "low",    "renin": "low",    "gordon": False},
        "SCNN1B":  {"onset": (12, 144),  "sbp": (150, 200), "k_range": (2.5, 3.2), "aldost": "low",    "renin": "low",    "gordon": False},
        "SCNN1G":  {"onset": (12, 144),  "sbp": (150, 200), "k_range": (2.5, 3.2), "aldost": "low",    "renin": "low",    "gordon": False},
        "WNK4":    {"onset": (12, 120),  "sbp": (155, 210), "k_range": (5.5, 7.5), "aldost": "normal", "renin": "low",    "gordon": True},
        "WNK1":    {"onset": (24, 180),  "sbp": (145, 195), "k_range": (5.0, 7.0), "aldost": "normal", "renin": "low",    "gordon": True},
        "KLHL3":   {"onset": (12, 144),  "sbp": (150, 205), "k_range": (5.2, 7.2), "aldost": "normal", "renin": "low",    "gordon": True},
        "CUL3":    {"onset": (1, 60),    "sbp": (165, 225), "k_range": (6.0, 8.0), "aldost": "normal", "renin": "low",    "gordon": True},
        "NR3C2":   {"onset": (36, 156),  "sbp": (148, 200), "k_range": (2.8, 3.5), "aldost": "low",    "renin": "low",    "gordon": False},
    }
    p = gene_params.get(gene, gene_params["HSD11B2"])

    def treatment_choice():
        r = rng.random()
        if p["gordon"]:
            if r < 0.65: return "Hydrochlorothiazide"
            if r < 0.82: return "Chlorthalidone"
            if r < 0.91: return "Amiloride+thiazide"
            return "Thiazide+potassium supplement"
        elif gene == "NR3C2":
            if r < 0.55: return "Amiloride"
            if r < 0.75: return "Calcium channel blocker"
            if r < 0.88: return "Beta-blocker+amiloride"
            return "ACEi+amiloride"
        elif gene in ("SCNN1B", "SCNN1G"):
            if r < 0.70: return "Amiloride"
            if r < 0.88: return "Triamterene"
            return "Amiloride+sodium restriction"
        else:  # HSD11B2
            if r < 0.45: return "Dexamethasone"
            if r < 0.70: return "Amiloride+dexamethasone"
            if r < 0.85: return "Renal transplant"
            return "Amiloride"

    def aldo_level():
        if p["aldost"] == "low":
            return round(rng.uniform(1.0, 8.0), 1)   # pmol/L × 10 = low
        else:
            return round(rng.uniform(12.0, 35.0), 1)  # normal-elevated

    def thf_the_ratio():
        if gene == "HSD11B2":
            return round(rng.uniform(10.5, 35.0), 1)  # PATHOGNOMONIC >10
        else:
            return round(rng.uniform(0.5, 1.5), 2)    # normal

    patients = []
    for i in range(n):
        onset_mo  = rng.randint(*p["onset"])
        age_dx    = onset_mo + rng.randint(1, 24)
        sbp       = rng.randint(*p["sbp"])
        dbp       = sbp - rng.randint(35, 55)
        k_plus    = round(rng.uniform(*p["k_range"]), 2)
        hco3      = round(rng.uniform(27.0, 34.0) if not p["gordon"] else rng.uniform(18.0, 24.0), 1)
        chloride  = round(rng.uniform(95.0, 108.0) if not p["gordon"] else rng.uniform(108.0, 120.0), 1)
        gender    = rng.choice(genders)
        aldost    = aldo_level()
        thf_ratio = thf_the_ratio()
        response  = treatment_choice()
        bp_crisis_preg = (gene == "NR3C2") and (gender == "F") and rng.random() < 0.70
        severe_hk = p["gordon"] and k_plus > 6.5

        patients.append({
            "id":               f"{gene}-{i+1:02d}",
            "gene":             gene,
            "gender":           gender,
            "onset_months":     onset_mo,
            "age_at_dx_months": age_dx,
            "sbp_mmhg":         sbp,
            "dbp_mmhg":         dbp,
            "serum_k_mmol":     k_plus,
            "serum_hco3_mmol":  hco3,
            "serum_cl_mmol":    chloride,
            "aldosterone_low":  p["aldost"] == "low",
            "aldosterone_pmol": aldost,
            "renin_suppressed": True,
            "thf_the_ratio":    thf_ratio,
            "bp_crisis_pregnancy": bp_crisis_preg,
            "severe_hyperkalemia": severe_hk,
            "treatment":        response,
            "gordon_phenotype": p["gordon"],
        })
    return patients


# ── API surface ───────────────────────────────────────────────────────────────
def generate_overview() -> dict:
    """Atlas overview — aggregate stats across all 8 mineralocorticoid excess genes."""
    all_patients = []
    for idx, g in enumerate(ATLAS_GENES):
        all_patients.extend(_make_patients(SEED_BASE + idx, g["gene"]))

    n              = len(all_patients)
    n_gordon       = sum(1 for p in all_patients if p["gordon_phenotype"])
    n_low_aldo     = sum(1 for p in all_patients if p["aldosterone_low"])
    n_severe_hk    = sum(1 for p in all_patients if p["severe_hyperkalemia"])
    n_bp_preg      = sum(1 for p in all_patients if p["bp_crisis_pregnancy"])
    mean_sbp       = round(sum(p["sbp_mmhg"]    for p in all_patients) / n, 1)
    mean_k         = round(sum(p["serum_k_mmol"] for p in all_patients) / n, 2)
    n_hsd11b2_path = sum(1 for p in all_patients if p["thf_the_ratio"] > 10)

    gene_summary = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        gene_summary.append({
            "gene":          g["gene"],
            "locus":         g["locus"],
            "n_patients":    len(pts),
            "mean_sbp":      round(sum(p["sbp_mmhg"]    for p in pts) / len(pts), 1),
            "mean_k":        round(sum(p["serum_k_mmol"] for p in pts) / len(pts), 2),
            "low_aldo_pct":  round(100 * sum(1 for p in pts if p["aldosterone_low"])   / len(pts), 1),
            "gordon_pct":    round(100 * sum(1 for p in pts if p["gordon_phenotype"])  / len(pts), 1),
            "syndrome":      (
                "AME"    if g["gene"] == "HSD11B2" else
                "Liddle" if g["gene"] in ("SCNN1B", "SCNN1G") else
                "Gordon" if g["gene"] in ("WNK4", "WNK1", "KLHL3", "CUL3") else
                "Geller"
            ),
            "inheritance": (
                "AR"    if g["gene"] == "HSD11B2" else
                "AD GOF" if g["gene"] in ("SCNN1B", "SCNN1G", "NR3C2") else
                "AD/AR" if g["gene"] == "KLHL3" else
                "AD"
            ),
            "thiazide_curative": g["gene"] in ("WNK4", "WNK1", "KLHL3", "CUL3"),
            "amiloride_curative": g["gene"] in ("SCNN1B", "SCNN1G"),
        })

    return {
        "atlas":            "Hereditary-Mineralocorticoid-Excess-Atlas",
        "genes":            [g["gene"] for g in ATLAS_GENES],
        "n_genes":          len(ATLAS_GENES),
        "n_patients":       n,
        "seeds":            f"{SEED_BASE}–{SEED_BASE + len(ATLAS_GENES) - 1}",
        "syndromes":        ["AME (HSD11B2)", "Liddle (SCNN1B/SCNN1G)", "Gordon (WNK4/WNK1/KLHL3/CUL3)", "Geller (NR3C2)"],
        "aggregate_metrics": {
            "mean_sbp_mmhg":         mean_sbp,
            "mean_serum_k_mmol":     mean_k,
            "gordon_phenotype_pct":  round(100 * n_gordon  / n, 1),
            "low_aldosterone_pct":   round(100 * n_low_aldo / n, 1),
            "severe_hyperkalemia_pct": round(100 * n_severe_hk / n, 1),
            "bp_crisis_pregnancy_pct": round(100 * n_bp_preg  / n, 1),
            "hsd11b2_pathognomonic_ratio_pct": round(100 * n_hsd11b2_path / n, 1),
        },
        "gene_summary":     gene_summary,
        "key_clinical_rules": [
            "HTN + hypokalaemia + LOW aldosterone + LOW renin: AME / Liddle / Geller — check THF/THE ratio",
            "HTN + HYPERKALEMIA + hyperchloraemic acidosis + NORMAL aldosterone: Gordon syndrome — thiazide curative",
            "AME vs Liddle: THF/THE ratio >10 = AME; normal ratio = Liddle (or Geller)",
            "Gordon: aldosterone normal or elevated — distinguishes from all other hereditary HTN syndromes",
            "Liddle: amiloride/triamterene CURATIVE; spironolactone INEFFECTIVE (ENaC-independent of MR)",
            "Geller NR3C2: spironolactone ABSOLUTE CI — acts as agonist; BP crisis in pregnancy PATHOGNOMONIC",
            "WNK1 intronic deletion: standard exon sequencing MISSES — CNV/MLPA required",
            "CUL3 Δexon9: dominant negative, most severe Gordon; de novo mutations common",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 mineralocorticoid excess genes."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        genes_data.append({
            "gene":            g["gene"],
            "locus":           g["locus"],
            "protein":         g["protein"],
            "protein_size":    g["protein_size"],
            "inheritance":     g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "n_patients":      len(pts),
            "mean_sbp":        round(sum(p["sbp_mmhg"]     for p in pts) / len(pts), 1),
            "mean_k":          round(sum(p["serum_k_mmol"]  for p in pts) / len(pts), 2),
            "mean_hco3":       round(sum(p["serum_hco3_mmol"] for p in pts) / len(pts), 1),
            "low_aldo_pct":    round(100 * sum(1 for p in pts if p["aldosterone_low"])   / len(pts), 1),
            "gordon_pct":      round(100 * sum(1 for p in pts if p["gordon_phenotype"])  / len(pts), 1),
            "severe_hk_pct":   round(100 * sum(1 for p in pts if p["severe_hyperkalemia"]) / len(pts), 1),
            "bp_preg_pct":     round(100 * sum(1 for p in pts if p["bp_crisis_pregnancy"]) / len(pts), 1),
            "treatment_distribution": treatments,
            "patients":        pts[:10],
        })
    return {"count": len(genes_data), "genes": genes_data}


def generate_definitions() -> dict:
    """Clinical glossary for hereditary mineralocorticoid excess syndromes."""
    return {
        "count": 8,
        "terms": [
            {
                "term": "AME (Apparent Mineralocorticoid Excess) — THF/THE Ratio Pathognomonic",
                "definition": (
                    "HSD11B2 LOF → cortisol acts as mineralocorticoid in kidney. "
                    "PATHOGNOMONIC DIAGNOSTIC: urinary (THF + 5αTHF) / THE ratio >10. "
                    "THF = tetrahydrocortisol (cortisol metabolite); "
                    "5αTHF = 5α-tetrahydrocortisol; THE = tetrahydrocortisone (cortisone metabolite). "
                    "Normal ratio: 0.5–1.5. AME: >10 (often 15-30 in complete deficiency). "
                    "TEST: 24h urinary steroid profile by GC-MS (specialist biochemistry lab). "
                    "KEY: ratio NORMAL in Liddle, Geller, Gordon — distinguishes AME from all others. "
                    "LIQUORICE PHENOCOPY: glycyrrhizinic acid inhibits HSD11B2 → same ratio elevation; "
                    "  always exclude dietary or herbal supplement liquorice before genetic testing. "
                    "TREATMENT: dexamethasone (↓ cortisol substrate) + amiloride (↓ ENaC activity); "
                    "  renal transplant curative — functional HSD11B2 in donor kidney."
                ),
            },
            {
                "term": "Liddle Syndrome — Amiloride Curative / Spironolactone Ineffective",
                "definition": (
                    "SCNN1B or SCNN1G PY-motif GOF → ENaC constitutively open → MR-independent Na+ retention. "
                    "TREATMENT RULE: "
                    "  Amiloride (or triamterene): blocks ENaC pore directly → curative. "
                    "  Spironolactone/eplerenone: MR blocker → INEFFECTIVE (ENaC open by structural change, not MR activation). "
                    "MECHANISM: PY motif (PPPXY) at ENaC C-terminus → NEDD4-2 E3 ligase binding site; "
                    "  NEDD4-2 ubiquitinates ENaC → internalisation from apical membrane; "
                    "  PY mutation: NEDD4-2 cannot bind → ENaC permanently at apical membrane → permanent Na+ absorption. "
                    "LABORATORY: HTN + hypokalaemia + metabolic alkalosis + LOW aldosterone + LOW renin + NORMAL THF/THE. "
                    "FAMILY SCREENING: AD — all first-degree relatives. "
                    "SCNN1B + SCNN1G: both on same panel (adjacent 16p12.2), both PY motif tested."
                ),
            },
            {
                "term": "Gordon Syndrome (PHAII) — Thiazide Curative — Hyperkalemia + HTN Paradox",
                "definition": (
                    "WNK4/WNK1/KLHL3/CUL3: NCC (Na/Cl cotransporter, DCT) constitutively active → "
                    "Na+/Cl- reabsorption elevated + K+ retention. "
                    "PARADOX: hyperkalemia + HTN despite NORMAL or ELEVATED aldosterone. "
                    "MECHANISM: NCC overactivity → reduced Na+ delivery to collecting duct → "
                    "  reduced lumen-negative potential → less ROMK K+ secretion → hyperkalemia. "
                    "ALDOSTERONE: normal/elevated (feedback stimulated by mild volume expansion or hyperkalemia). "
                    "  This is the KEY distinguisher from Liddle/AME/Geller (all have LOW aldosterone). "
                    "THIAZIDE CURE: hydrochlorothiazide/chlorthalidone blocks NCC → Na+ delivery restored → "
                    "  K+ secretion resumes + BP normalises + acidosis corrects. "
                    "GENE ORDER: WNK4 (17q21, missense) → KLHL3 (5q31, AD missense or AR biallelic) → "
                    "  CUL3 exon 9 deletion → WNK1 intron 1 CNV (requires CNV analysis). "
                    "BIOCHEMICAL FINGERPRINT: HTN + hyperkalemia + hyperchloraemic metabolic acidosis + normal/elevated aldosterone."
                ),
            },
            {
                "term": "Geller Syndrome NR3C2 S810L — Spironolactone Absolute CI — Pregnancy Crisis",
                "definition": (
                    "NR3C2 S810L (Ser810Leu) GOF → progesterone becomes full MR agonist. "
                    "PATHOGNOMONIC: HTN markedly worsened in pregnancy (progesterone surge → MR activation). "
                    "NORMAL: progesterone is MR ANTAGONIST (occupies LBD without full coactivator recruitment); "
                    "  Pregnancy elevation physiologically vasodilates → BP decreases in normal women. "
                    "GELLER: progesterone ACTIVATES S810L MR → massive ENaC/SGK1 transcription → HTN crisis. "
                    "SPIRONOLACTONE — ABSOLUTE CI: "
                    "  Spironolactone normally = MR ANTAGONIST; "
                    "  At S810L MR: spironolactone binds Leu810 pocket → H12 helix repositions → coactivator recruited → MR AGONIST; "
                    "  Starting spironolactone → HTN crisis in Geller. "
                    "TREATMENT: amiloride (ENaC blockade, downstream of MR) + CCB + beta-blocker. "
                    "PREGNANCY: methyldopa (safe, no MR agonism); close BP monitoring from conception."
                ),
            },
            {
                "term": "WNK1 Intronic Deletion — Standard Exon Sequencing Misses — CNV Required",
                "definition": (
                    "WNK1 Gordon PHA2A: caused by LARGE INTRONIC DELETIONS in intron 1 (up to 41 kb). "
                    "DIAGNOSTIC PITFALL: standard gene panel (exon sequencing) misses intronic deletions. "
                    "MUST REQUEST: CNV analysis / MLPA / array-CGH / long-range PCR spanning WNK1 intron 1. "
                    "MECHANISM: intron 1 deletion → KS-WNK1 promoter elements removed → "
                    "  kidney-specific short isoform (KS-WNK1, kinase-dead) reduced → "
                    "  L-WNK1 (full-length, kinase-active) unopposed → NCC hyperphosphorylation. "
                    "CLINICAL: milder phenotype than CUL3 Gordon; often adolescent/adult presentation. "
                    "DIAGNOSTIC ALGORITHM: "
                    "  1. WNK4 exon sequencing → 2. KLHL3 sequencing → 3. CUL3 exon 9 deletion → "
                    "  4. WNK1 intron 1 CNV ('Gordon panel negative' on coding sequencing does not exclude PHA2A). "
                    "TREATMENT: thiazide — identical to all Gordon forms."
                ),
            },
            {
                "term": "CUL3 Δexon9 Dominant Negative — Most Severe Gordon — Extra-Renal Comorbidities",
                "definition": (
                    "CUL3 Δexon9: in-frame exon 9 deletion → dominant-negative protein → "
                    "  sequesters KLHL3 but cannot transfer ubiquitin → ALL CUL3-RING E3 complexes disrupted. "
                    "SEVERITY: most severe Gordon phenotype — K+ often >7 mmol/L at presentation. "
                    "DE NOVO: frequently sporadic (de novo dominant) — no family history. "
                    "EXTRA-RENAL: short stature, possible intellectual disability/autism (other CUL3 substrates affected). "
                    "DIFFERENT CUL3 MUTATIONS → DIFFERENT DISEASES: "
                    "  Δexon9 → Gordon PHA2E; "
                    "  Other exon deletions → neurodevelopmental syndrome (autism/epilepsy) WITHOUT Gordon. "
                    "EMERGENCY MANAGEMENT: K+ >7 mmol/L → IV calcium gluconate + insulin-dextrose; "
                    "  ECG monitoring until K+ <6.5. "
                    "THIAZIDE: start ASAP — curative for Gordon biochemistry even in most severe forms; "
                    "  Higher dose may be required (dominant-negative mechanism more complete WNK accumulation)."
                ),
            },
            {
                "term": "KLHL3 AD vs AR — E3 Ligase Adaptor — Gordon Severity Spectrum",
                "definition": (
                    "KLHL3 encodes the substrate adaptor of CUL3-RING E3 ligase for WNK1/4 ubiquitination. "
                    "AD (missense): Kelch domain (WNK-binding) or BTB domain (CUL3-binding) mutation → "
                    "  partial loss → one functional allele retains some degradation → MILDER Gordon. "
                    "AR (biallelic): complete KLHL3 absence → maximal WNK accumulation → SEVERE Gordon. "
                    "MOST COMMON GORDON GENE: KLHL3 accounts for ~20-30% of genetically confirmed Gordon. "
                    "PHOSPHO-NCC BIOMARKER (research): urinary phosphorylated-NCC elevated in Gordon regardless of which gene; "
                    "  useful for confirming NCC pathway activation; not yet clinically validated. "
                    "TREATMENT: thiazide curative for all KLHL3 forms. "
                    "RELATED: CUL3 mutation disrupts entire CUL3 complex (broader substrate effects); "
                    "  KLHL3 mutation affects only WNK1/4 (and other KLHL3-dependent substrates) → "
                    "  fewer extra-renal features than CUL3 Δexon9."
                ),
            },
            {
                "term": "Differential Diagnosis of Hereditary HTN + Low Aldosterone + Low Renin",
                "definition": (
                    "KEY DIFFERENTIAL: all present with HTN + hypokalaemia + metabolic alkalosis + "
                    "  LOW aldosterone + LOW renin (contrast: primary hyperaldosteronism has HIGH aldosterone). "
                    "STEP 1 — THF/THE RATIO: "
                    "  >10: AME (HSD11B2) → confirm with HSD11B2 sequencing; "
                    "  Normal (0.5-1.5): Liddle or Geller. "
                    "STEP 2 — ENaC PANEL (SCNN1B + SCNN1G): Liddle. "
                    "STEP 3 — NR3C2 (S810L): Geller (especially if female with pregnancy history of worsening HTN, "
                    "  or if spironolactone trial → worsened HTN). "
                    "CONTRAST WITH GORDON: "
                    "  Gordon = HYPERKALEMIA (not hypokalaemia) + NORMAL/elevated aldosterone → "
                    "  Gordon panel (WNK4, KLHL3, CUL3 exon 9, WNK1 CNV). "
                    "EXCLUDE: "
                    "  Exogenous liquorice/carbenoxolone (phenocopies AME); "
                    "  Cushing (ACTH-dependent cortisol excess); "
                    "  Congenital adrenal hyperplasia (11β-OHase deficiency = CYP11B1)."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:500])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
