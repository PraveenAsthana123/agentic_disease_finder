#!/usr/bin/env python3
"""Hereditary-Homocystinuria-Sulfur-Amino-Acid-Metabolism-Atlas -- Complete 8-Gene Reference
Homocystinuria, Remethylation Defects, and Sulfur Amino Acid Metabolism Disorders
(CBS/Classical-Homocystinuria-Ectopia-Lentis-PATHOGNOMONIC-Marfanoid-VTE-B6-Responsive-50pct-Betaine,
 MTHFR/Severe-AR-Remethylation-Methionine-LOW-DDx-CBS-NOT-Common-C677T-Polymorphism,
 MTR/cblG-Megaloblastic-Anemia-Homocystinuria-WITHOUT-MMA-Hydroxocobalamin,
 MTRR/cblE-Indistinguishable-cblG-MTRR-Reactivates-MTR-Hydroxocobalamin,
 MAT1A/Isolated-Hypermethioninemia-Methionine-HIGH-Homocysteine-NORMAL-DDx-CBS,
 GNMT/Isolated-Hypermethioninemia-Liver-Disease-AFP-ELEVATED-PATHOGNOMONIC-Very-Rare,
 AHCY/SAH-Hydrolase-Hypermethioninemia-SAH-Accumulates-Myopathy-Developmental-Delay,
 CTH/Cystathioninuria-Mostly-Benign-B6-Responsive-Plasma-Cystathionine-Elevated):

CBS (Cystathionine beta-synthase; 551aa; 21q22.3; AR-LOF;
     CBS encodes cystathionine beta-synthase, the PLP (pyridoxal-5'-phosphate / B6)-dependent
       enzyme that catalyses the first committed step of the transsulfuration pathway:
       homocysteine + serine → cystathionine (irreversible); cystathionine then → cysteine via CTH;
     CBS LOF → homocysteine CANNOT enter transsulfuration → ACCUMULATES in plasma and urine;
     CLASSICAL HOMOCYSTINURIA (CBS DEFICIENCY / HOMOCYSTINURIA TYPE I):
     BIOCHEMICAL PROFILE:
       (1) TOTAL HOMOCYSTEINE (tHcy) MARKEDLY ELEVATED: plasma tHcy >100 µmol/L (often 200-500+);
           free homocysteine + disulfide forms; urine homocystine (disulfide) on amino acid screen;
       (2) METHIONINE ELEVATED: CBS deficiency → upstream methionine accumulates (methionine → SAM →
           SAH → homocysteine → [CBS blocked]); methionine >200 µmol/L (normal <40 µmol/L);
           METHIONINE HIGH distinguishes CBS deficiency from remethylation defects (where methionine LOW);
       (3) CYSTEINE LOW: downstream cysteine deficient (cannot make cystathionine → cysteine);
           cysteine becomes essential amino acid in CBS deficiency;
     PATHOGNOMONIC CLINICAL FEATURES:
       (1) ECTOPIA LENTIS (LENS DISLOCATION): 80-90% by age 20; homocysteine disrupts
           disulfide bonds in fibrillin → zonular fiber weakening → lens subluxation;
           DOWNWARD dislocation (DDx Marfan: UPWARD); severe myopia precedes dislocation;
           annual ophthalmological slit-lamp exam MANDATORY;
       (2) MARFANOID HABITUS: tall stature, arachnodactyly, high-arched palate, pectus;
           malar flush (homocysteine-related vasodilation); OSTEOPOROSIS (vertebral fractures);
       (3) VENOUS AND ARTERIAL THROMBOEMBOLISM (VTE): MOST COMMON CAUSE OF DEATH;
           homocysteine is prothrombotic (endothelial injury, platelet activation, factor V);
           DVT, PE, stroke, MI at unusually young age; thrombotic event may be presenting feature;
       (4) INTELLECTUAL DISABILITY: ~50% of untreated patients have ID; variable (0-severe);
           psychiatric symptoms (depression, schizophrenia-like); behavioral difficulties;
     NBS POSITIVE: elevated methionine on tandem MS newborn screen; high sensitivity if methionine
       cut-off low enough; early treatment (before symptoms) prevents most complications;
     B6-RESPONSIVENESS — CRITICAL DIAGNOSTIC AND THERAPEUTIC PEARL:
       ~50% of CBS mutations have RESIDUAL CBS ACTIVITY that is enhanced by pharmacological doses
       of pyridoxine (B6 / PLP cofactor); B6-RESPONSIVE patients: lower clinical severity;
       B6 TRIAL PROTOCOL: pyridoxine 200-1000 mg/day for 4-6 weeks + check tHcy reduction;
       B6-RESPONSIVE (>50% reduction in tHcy) → pyridoxine + low-methionine diet;
       B6-NON-RESPONSIVE → low-methionine diet + BETAINE + folate + B12;
     BETAINE THERAPY: trimethylglycine (betaine) remethylates homocysteine → methionine
       via BHMT (betaine-homocysteine methyltransferase) — bypasses the CBS block;
       BETAINE 3-6 g/twice daily; effective in both B6-responsive and non-responsive;
       caution: methionine can rise further → monitor; may cause cerebral oedema at high doses;
     seed SEED_BASE+0).

MTHFR (Methylenetetrahydrofolate reductase; 656aa; 1p36.22; AR-LOF — SEVERE disease;
     CRITICAL DISTINCTION: MTHFR C677T and A1298C are COMMON POLYMORPHISMS (NOT disease) —
       carrier rate C677T: 10-15% European; homozygous C677T: 5-15% population; mildly elevated
       tHcy; NO clinical syndrome of homocystinuria; C677T/A1298C homozygosity is NOT homocystinuria;
       SEVERE AR MTHFR DEFICIENCY = RARE, DISTINCT DISEASE from C677T/A1298C polymorphisms;
     MTHFR encodes MTHFR, the FAD-dependent enzyme that reduces 5,10-methyleneTHF → 5-methylTHF;
       5-methylTHF is the methyl donor for MTR/methionine synthase: 5-methylTHF + homocysteine →
       THF + methionine (methylation of homocysteine via B12/cobalamin); MTHFR LOF → no 5-methylTHF
       → methionine synthase cannot function → homocysteine ACCUMULATES; methionine CANNOT be
       synthesised from homocysteine → METHIONINE LOW or NORMAL-LOW (key DDx from CBS);
     SEVERE MTHFR DEFICIENCY BIOCHEMICAL PROFILE:
       (1) TOTAL HOMOCYSTEINE ELEVATED: similar to CBS; tHcy >50-200 µmol/L;
       (2) METHIONINE LOW OR LOW-NORMAL: PATHOGNOMONIC DDx from CBS (CBS → methionine HIGH);
           serine also may be low; folate metabolites disrupted;
       (3) NO MMA: distinguishes from cobalamin defects (MTR/MTRR/cblC/cblD/cblG/cblE) which
           may have MMA; MTHFR deficiency has isolated homocystinuria WITHOUT MMA;
     CLINICAL PHENOTYPE (severe AR deficiency):
       neonatal/infantile: microcephaly, apneas, seizures, hypotonia, developmental delay;
       later: neurological regression, dementia, schizophrenia-like psychosis, peripheral
       neuropathy; cerebral ventricular dilation; ± ectopia lentis (less common than CBS);
       THROMBOTIC EVENTS: cerebrovascular events common;
     TREATMENT: methylfolate (5-methylTHF / folinic acid) + betaine + B12 (hydroxocobalamin);
       methylfolate supplementation provides substrate for downstream methylation cycles;
       betaine: remethylates homocysteine via BHMT bypass (as in CBS);
       low-methionine diet usually NOT needed (methionine already low);
       B6: NOT effective (MTHFR does not use PLP); B12: important cofactor for MTR;
     seed SEED_BASE+1).

MTR  (Methionine synthase / 5-methyltetrahydrofolate-homocysteine methyltransferase; 1265aa; 1q43; AR;
     MTR encodes methionine synthase (MS), the cobalamin (B12)-dependent enzyme that catalyses
       the remethylation of homocysteine using 5-methylTHF: homocysteine + 5-methylTHF →
       methionine + THF; the methyl group donor is methylcobalamin (MeCbl, bound in MS active site);
       MTRR (cblE) reactivates oxidised methylcobalamin within MTR — MTRR deficiency = same
       clinical phenotype as MTR deficiency;
     MTR LOF → methionine synthase inactive → homocysteine cannot be remethylated → ACCUMULATES;
     cblG DEFICIENCY (COBALAMIN G / MTR DEFICIENCY — ISOLATED REMETHYLATION DEFECT):
     BIOCHEMICAL PROFILE:
       (1) TOTAL HOMOCYSTEINE ELEVATED: typically >50-200 µmol/L;
       (2) METHIONINE LOW: remethylation blocked → less methionine made → low plasma methionine;
           KEY DDx from CBS: CBS has methionine HIGH; cblG/cblE have methionine LOW;
       (3) NO MMA: ISOLATED remethylation defect — adenosylcobalamin (AdoCbl) pathway intact;
           distinguishes cblG/cblE from cblC/cblD (combined MMA + homocystinuria);
           MMA absent = CRITICAL for narrowing to cblG/cblE vs combined forms;
       (4) MEGALOBLASTIC ANEMIA: MTR inactivity → 5-methylTHF cannot donate methyl group →
           THF sequestered as 5-methylTHF (methylfolate trap) → functional folate deficiency →
           megaloblastic anemia (macrocytosis, hypersegmented neutrophils);
     CLINICAL PHENOTYPE:
       neonatal/infantile onset: feeding difficulties, failure to thrive, hypotonia, seizures;
       megaloblastic anemia + homocystinuria WITHOUT methylmalonic aciduria;
       neurological: cerebral atrophy, regression, schizophrenia-like in later onset;
       ocular: nystagmus (but ectopia lentis LESS prominent than CBS);
     DIAGNOSIS: plasma amino acids (tHcy elevated, methionine low) + urine organic acids
       (MMA absent) + acylcarnitines + cobalamin studies; molecular MTR gene testing;
       differentiate cblG from cblE: complementation analysis OR MTR gene sequencing;
     TREATMENT: hydroxocobalamin (OHCbl) injections (NOT cyanocobalamin — must convert to MeCbl);
       betaine: remethylates via BHMT bypass; folate/methylfolate; methionine supplementation
       (low methionine → supplement); B6: not effective alone; pyridoxine not therapeutic here;
     seed SEED_BASE+2).

MTRR (Methionine synthase reductase; 698aa; 5p15.31; AR;
     MTRR encodes methionine synthase reductase (MSR), the FAD/FMN-containing diflavoprotein
       reductase in the P450-reductase superfamily that REACTIVATES oxidised cobalamin within
       the MTR (methionine synthase) active site; during catalytic cycling, methylcobalamin on
       MTR is oxidised to cob(II)alamin (inactive); MTRR uses NADPH → FMN → FAD → reductively
       methylates the inactive cob(II)alamin back to methylcobalamin; cob(II)alamin + CH3 → MeCbl;
     WITHOUT MTRR: MTR inactivates over time → same effect as direct MTR (cblG) deficiency;
     cblE DEFICIENCY (COBALAMIN E / MTRR DEFICIENCY):
     CLINICAL PHENOTYPE: INDISTINGUISHABLE FROM cblG (MTR deficiency):
       (1) elevated total homocysteine (tHcy);
       (2) methionine LOW (remethylation blocked);
       (3) NO MMA (adenosylcobalamin pathway unaffected);
       (4) megaloblastic anemia;
       neonatal/infantile: same as cblG; neurological regression; developmental delay;
     KEY DDx cblG vs cblE:
       Biochemically IDENTICAL — cannot distinguish by metabolites alone;
       COMPLEMENTATION ANALYSIS (fibroblast cell fusion) definitively distinguishes cblG/cblE;
       MOLECULAR: MTR sequencing (cblG) vs MTRR sequencing (cblE);
       CLINICALLY: treat as IDENTICAL with hydroxocobalamin; the distinction matters for
       prenatal diagnosis, carrier testing, and genetic counselling;
     FOUNDER MUTATION: MTRR p.I22M (c.66A>G) — common MTRR variant; population frequency
       ~20% European; p.I22M ALONE is NOT pathogenic (mild at most); homozygous severe LOF
       mutations cause disease; p.I22M compound het with severe allele may contribute but
       contested; DO NOT diagnose cblE on p.I22M alone without elevated tHcy;
     TREATMENT: identical to cblG — hydroxocobalamin + betaine + folate + methionine supp;
     seed SEED_BASE+3).

MAT1A (Methionine adenosyltransferase 1A / liver-specific MAT; 395aa; 10q22.3; AR and AD;
     MAT1A encodes the catalytic subunit of methionine adenosyltransferase I/III (MAT I/III),
       the liver-specific MAT isoform that synthesises SAM (S-adenosylmethionine) from
       methionine + ATP → SAM + PPi + Pi; MAT I/III is expressed predominantly in ADULT LIVER
       (MAT II is the ubiquitous form, not affected by MAT1A deficiency);
     MAT1A LOF → liver-specific MAT I/III absent → SAM synthesis reduced in liver →
       methionine CANNOT be metabolised via SAM pathway → METHIONINE ACCUMULATES;
     ISOLATED HYPERMETHIONINEMIA (MAT1A DEFICIENCY):
     BIOCHEMICAL PROFILE:
       (1) METHIONINE MARKEDLY ELEVATED: plasma methionine >300-1000 µmol/L (normal <40);
           most prominent on NBS tandem MS; one of the highest methioninemia levels;
       (2) HOMOCYSTEINE NORMAL OR LOW: SAM is upstream of homocysteine; MAT1A deficiency
           → SAM low → SAH low → homocysteine NORMAL or LOW; KEY DDx from CBS (hcy HIGH);
       (3) SAM LOW: S-adenosylmethionine reduced in liver → may affect methylation reactions;
     CLINICAL SPECTRUM:
       MAJORITY BENIGN: many MAT1A deficiency cases are completely asymptomatic; discovered
       incidentally on NBS or family screening; normal development, no treatment needed;
       LIVER DISEASE SUBSET: some patients develop demyelination (white matter abnormalities on MRI),
       neurological symptoms (variable penetrance); elevated liver transaminases in some;
       INHERITANCE: AR (biallelic LOF) → benign to moderate; AD heterozygous LOF (some)
       → isolated hypermethioninemia without disease; intermediate alleles complicate genotype-phenotype;
     KEY CLINICAL PEARL: HIGH methionine + NORMAL homocysteine = MAT1A, GNMT, or AHCY;
       NOT CBS (CBS → both methionine AND homocysteine high);
     TREATMENT: low-methionine diet in symptomatic/neurological cases; asymptomatic = no treatment;
       SAM supplementation may help in severe cases; methionine restricted protein;
     seed SEED_BASE+4).

GNMT (Glycine N-methyltransferase; 294aa; 6p12.1; AR;
     GNMT encodes glycine N-methyltransferase, the SAM-dependent methyltransferase that catalyses
       the transfer of the methyl group from SAM to glycine → sarcosine (N-methylglycine) + SAH;
       GNMT is the MAJOR SAM-consuming enzyme in liver — transfers ~20-40% of hepatic SAM;
       also functions as a folate-binding protein; GNMT regulates SAM homeostasis;
     GNMT LOF → SAM CANNOT be consumed via glycine methylation → SAM ACCUMULATES in liver →
       feedback inhibits MAT → methionine also ACCUMULATES;
     GNMT DEFICIENCY (VERY RARE — <30 reported cases):
     BIOCHEMICAL PROFILE:
       (1) METHIONINE ELEVATED: plasma methionine often >200-600 µmol/L;
       (2) SAM MARKEDLY ELEVATED: SAM typically 10-100x normal; SAM elevation is relatively
           SPECIFIC for GNMT deficiency (rare other causes of high SAM);
       (3) HOMOCYSTEINE NORMAL: SAM → SAH → homocysteine pathway downstream of GNMT block;
           homocysteine NORMAL or mildly elevated; KEY DDx from CBS (hcy HIGH) and MTHFR (hcy HIGH);
       (4) ALPHA-FETOPROTEIN (AFP) CHRONICALLY ELEVATED — PATHOGNOMONIC:
           AFP persistently high (>100-1000 ng/mL) from childhood; not related to HCC;
           mechanism unclear (possibly SAM effect on liver regeneration signalling);
           CHRONIC AFP ELEVATION IN HEPATIC DISEASE CONTEXT = strongly suggests GNMT;
     LIVER DISEASE: hepatomegaly, chronic liver disease, elevated transaminases; SAM may be
       hepatotoxic at high concentrations; monitoring for hepatic fibrosis/cirrhosis; liver biopsy
       shows steatosis, periportal fibrosis; potential for progression;
     CLINICAL: mild-moderate hepatopathy predominates; most survive to adulthood; variable;
     TREATMENT: low-methionine diet (controversial); no specific therapy established; liver
       transplant considered for severe progressive hepatic fibrosis; supportive care;
     seed SEED_BASE+5).

AHCY (Adenosylhomocysteinase / S-adenosylhomocysteine hydrolase; 432aa; 20q11.22; AR;
     AHCY encodes adenosylhomocysteinase (AHCY / SAHH), the enzyme that hydrolyses
       S-adenosylhomocysteine (SAH) to adenosine + homocysteine (reversible, equilibrium favours
       synthesis in vivo, but maintained in hydrolysis direction by adenosine deaminase and
       homocysteine export); SAH is the UNIVERSAL INHIBITOR of SAM-dependent methyltransferases —
       high SAH inhibits virtually all cellular methylation reactions;
     AHCY LOF → SAH CANNOT be hydrolysed → SAH ACCUMULATES → product inhibition of ALL
       SAM-dependent methyltransferases → global methylation failure;
     AHCY DEFICIENCY (SAH HYDROLASE DEFICIENCY):
     BIOCHEMICAL PROFILE:
       (1) METHIONINE ELEVATED: SAH → homocysteine pathway blocked → upstream SAH accumulates
           → feedback increases methionine and SAM;
       (2) SAH MARKEDLY ELEVATED: plasma SAH elevated; SAH:SAM ratio >1 (normally <0.1);
       (3) HOMOCYSTEINE ELEVATED: downstream homocysteine cannot be cleared via transsulfuration
           (CBS requires equilibrium flow) → homocysteine accumulates; BUT tHcy often less extreme
           than CBS;
       (4) SARCOSINE NOT ELEVATED (DDx from GNMT);
     CLINICAL PHENOTYPE — DISTINCTIVE MULTI-SYSTEM DISORDER:
       (1) MYOPATHY: elevated CK; proximal muscle weakness; early-onset myopathy DISTINCTIVE —
           most homocystinurias do NOT have significant myopathy; skeletal muscle SAM-dependent
           creatine synthesis inhibited (SAH inhibits AGAT/GAMT methyltransferases);
       (2) DEVELOPMENTAL DELAY: cognitive impairment; hypotonia; delay in motor and language;
       (3) HEPATOPATHY: liver disease; elevated transaminases; hepatomegaly; steatosis;
           liver fibrosis possible; histological similarity to GNMT;
       (4) CARDIOMYOPATHY: reported in some; SAH inhibits cardiac creatine synthesis;
     CREATINE DEFICIENCY: SAH inhibits GAMT (guanidinoacetate methyltransferase) →
       creatine synthesis impaired → reduced phosphocreatine → myopathy + cardiomyopathy;
     TREATMENT: low-methionine diet; adenosine deaminase (ADA) support; supplemental creatine
       (addresses GAMT inhibition product deficiency); betaine (limited use — methionine already high);
     seed SEED_BASE+6).

CTH  (Cystathionine gamma-lyase / cystathionase; 405aa; 1p31.3; AR;
     CTH encodes cystathionine gamma-lyase (cystathionase), the PLP (B6)-dependent enzyme that
       catalyses the second step of the transsulfuration pathway: cystathionine → cysteine + alpha-
       ketobutyrate + NH3; CTH completes transsulfuration (CBS step 1: homocysteine → cystathionine;
       CTH step 2: cystathionine → cysteine);
     CTH LOF → cystathionine CANNOT be converted to cysteine → CYSTATHIONINE ACCUMULATES;
     CYSTATHIONINURIA (CTH DEFICIENCY):
     BIOCHEMICAL PROFILE:
       (1) CYSTATHIONINE MARKEDLY ELEVATED: plasma cystathionine high (often >100-1000 µmol/L);
           urine cystathionine elevated; cystathionine on amino acid screen = key finding;
       (2) HOMOCYSTEINE NORMAL: CBS step is intact; no block at homocysteine → homocysteine
           NORMAL — KEY DDx from CBS deficiency (hcy HIGH) and remethylation defects (hcy HIGH);
       (3) METHIONINE NORMAL: upstream pathway normal;
       (4) CYSTEINE MAY BE MILDLY LOW: cysteine synthesis blocked at CTH step; cysteine becomes
           essential in CTH deficiency; usually compensated by dietary cysteine;
     CYSTATHIONINURIA IS MOSTLY BENIGN:
       VAST MAJORITY of CTH deficiency patients are clinically normal; most cases discovered
       incidentally; no consistent neurodevelopmental abnormalities in properly controlled studies;
       older reports of intellectual disability, seizures, urolithiasis are UNCONTROLLED and likely
       coincidental; modern consensus = CTH deficiency is benign;
     B6 RESPONSIVENESS: cystathionase is B6/PLP-dependent; some CTH mutations respond to
       pharmacological B6 (pyridoxine 200-500 mg/day) → normalises plasma cystathionine;
       B6 trial recommended but outcomes minimally clinically relevant (mostly benign);
     SECONDARY CYSTATHIONINURIA: important — cystathioninuria can be SECONDARY to:
       (1) Vitamin B6 deficiency (cystathionase requires PLP);
       (2) Hepatic disease (liver enzyme damage → reduced cystathionase);
       (3) Neuroblastoma (ectopic cystathioninase reduction);
       distinguish secondary from primary (genetic) with molecular CTH testing + B6 trial;
     TREATMENT: NO TREATMENT NEEDED in most cases; B6 can normalise metabolites biochemically;
       dietary intervention NOT required; reassure family about benign prognosis;
     seed SEED_BASE+7).

SEED_BASE = 4190
"""

import json
import random
from typing import Any


ATLAS_ID   = "hereditary-homocystinuria-sulfur-amino-acid-metabolism-atlas"
ATLAS_NAME = "Hereditary-Homocystinuria-Sulfur-Amino-Acid-Metabolism-Atlas"
SEED_BASE  = 4190
N_PER_GENE = 40

GENES = ["CBS", "MTHFR", "MTR", "MTRR", "MAT1A", "GNMT", "AHCY", "CTH"]

GENE_META = {
    "CBS":   {"aa": 551,  "locus": "21q22.3",  "kda": 63,  "inh": "AR",    "disease": "Classical Homocystinuria (CBS deficiency)",        "omim_gene": 613381, "omim_disease": 236200},
    "MTHFR": {"aa": 656,  "locus": "1p36.22",  "kda": 74,  "inh": "AR",    "disease": "Severe MTHFR Deficiency (Remethylation Homocystinuria)", "omim_gene": 607093, "omim_disease": 236250},
    "MTR":   {"aa": 1265, "locus": "1q43",     "kda": 140, "inh": "AR",    "disease": "cblG (Methionine Synthase Deficiency)",             "omim_gene": 156570, "omim_disease": 250940},
    "MTRR":  {"aa": 698,  "locus": "5p15.31",  "kda": 78,  "inh": "AR",    "disease": "cblE (Methionine Synthase Reductase Deficiency)",   "omim_gene": 602568, "omim_disease": 236270},
    "MAT1A": {"aa": 395,  "locus": "10q22.3",  "kda": 43,  "inh": "AR/AD", "disease": "Isolated Hypermethioninemia (MAT I/III Deficiency)","omim_gene": 610550, "omim_disease": 250850},
    "GNMT":  {"aa": 294,  "locus": "6p12.1",   "kda": 33,  "inh": "AR",    "disease": "GNMT Deficiency (Hypermethioninemia + Hepatopathy)", "omim_gene": 606628, "omim_disease": 606664},
    "AHCY":  {"aa": 432,  "locus": "20q11.22", "kda": 48,  "inh": "AR",    "disease": "AHCY Deficiency (SAH Hydrolase Deficiency)",        "omim_gene": 180960, "omim_disease": 613752},
    "CTH":   {"aa": 405,  "locus": "1p31.3",   "kda": 44,  "inh": "AR",    "disease": "Cystathioninuria (CTH Deficiency, Mostly Benign)",  "omim_gene": 607657, "omim_disease": 219500},
}

GENE_PATHOGNOMONIC = {
    "CBS":   "Ectopia lentis (DOWNWARD dislocation) + Marfanoid habitus + VTE in young + methionine HIGH + homocysteine very HIGH",
    "MTHFR": "Homocysteine HIGH + methionine LOW (opposite CBS) + NO MMA + common C677T NOT disease",
    "MTR":   "Megaloblastic anemia + homocysteine HIGH + methionine LOW + NO MMA (distinguishes from cblC/cblD) + cblG label",
    "MTRR":  "Biochemically IDENTICAL to cblG/MTR; distinguish only by complementation or gene sequencing; cblE label",
    "MAT1A": "Methionine very HIGH + homocysteine NORMAL/LOW (opposite CBS); mostly benign; NBS tandem-MS elevated met",
    "GNMT":  "Methionine HIGH + SAM HIGH + AFP chronically ELEVATED PATHOGNOMONIC + liver disease; homocysteine NORMAL",
    "AHCY":  "Methionine HIGH + SAH HIGH + myopathy (elevated CK) DISTINCTIVE + developmental delay + hepatopathy",
    "CTH":   "Cystathionine elevated + homocysteine NORMAL + MOSTLY BENIGN + B6 responsive + secondary cystathioninuria DDx",
}

GENE_KEY_DRUG = {
    "CBS":   "Betaine 3-6 g BID + low-methionine diet ± pyridoxine 200-1000 mg/day (B6-trial MANDATORY — 50% responsive)",
    "MTHFR": "Methylfolate (5-MTHF) + betaine + hydroxocobalamin; low-methionine diet NOT needed (methionine already low)",
    "MTR":   "Hydroxocobalamin IM injections + betaine + folate + methionine supplementation; cyanocobalamin LESS effective",
    "MTRR":  "Hydroxocobalamin IM + betaine + folate + methionine supp; identical treatment to cblG; distinguish via molecular",
    "MAT1A": "Low-methionine diet if symptomatic; asymptomatic = no treatment; monitor MRI for white matter changes",
    "GNMT":  "Low-methionine diet; liver monitoring; no proven specific therapy; liver transplant for severe fibrosis",
    "AHCY":  "Low-methionine diet + supplemental creatine (addresses GAMT inhibition) + adenosine deaminase support",
    "CTH":   "No treatment needed in most (benign); pyridoxine 200-500 mg/day normalises cystathionine biochemically",
}

METHIONINE_DIR = {
    "CBS":   "↑↑↑ HIGH",
    "MTHFR": "↓ LOW",
    "MTR":   "↓ LOW",
    "MTRR":  "↓ LOW",
    "MAT1A": "↑↑↑ HIGH",
    "GNMT":  "↑↑↑ HIGH",
    "AHCY":  "↑↑ HIGH",
    "CTH":   "→ NORMAL",
}

HOMOCYSTEINE_DIR = {
    "CBS":   "↑↑↑ VERY HIGH",
    "MTHFR": "↑↑ HIGH",
    "MTR":   "↑↑ HIGH",
    "MTRR":  "↑↑ HIGH",
    "MAT1A": "→ NORMAL",
    "GNMT":  "→ NORMAL",
    "AHCY":  "↑ ELEVATED",
    "CTH":   "→ NORMAL",
}

MMA_STATUS = {
    "CBS":   "→ Absent",
    "MTHFR": "→ Absent",
    "MTR":   "→ Absent (DDx cblC/cblD which have MMA)",
    "MTRR":  "→ Absent (DDx cblC/cblD which have MMA)",
    "MAT1A": "→ Absent",
    "GNMT":  "→ Absent",
    "AHCY":  "→ Absent",
    "CTH":   "→ Absent",
}

NBS_DETECTS = {
    "CBS":   True,
    "MTHFR": False,
    "MTR":   False,
    "MTRR":  False,
    "MAT1A": True,
    "GNMT":  True,
    "AHCY":  True,
    "CTH":   True,
}

B6_RESPONSIVE = {
    "CBS":   "50% responsive (MANDATORY B6 trial)",
    "MTHFR": "Not responsive (B6/PLP not cofactor for MTHFR)",
    "MTR":   "Not responsive",
    "MTRR":  "Not responsive",
    "MAT1A": "Not applicable",
    "GNMT":  "Not applicable",
    "AHCY":  "Not applicable",
    "CTH":   "Often responsive (PLP-dependent enzyme)",
}

FOUNDER_MUTATIONS = {
    "CBS":   "p.I278T (B6-responsive, Irish/European), p.G307S (B6-non-responsive), IVS11-2A>C (Italian/Eastern European)",
    "MTHFR": "C677T and A1298C are POLYMORPHISMS NOT disease; severe AR LOF mutations individually rare",
    "MTR":   "Individual rare missense; no common single founder",
    "MTRR":  "p.I22M c.66A>G common POLYMORPHISM (~20% European) — NOT pathogenic alone; distinguish from LOF mutations",
    "MAT1A": "R264H common in heterozygotes; p.R264C, p.G378E in AR cases",
    "GNMT":  "Very rare; <30 cases worldwide; no established common founder",
    "AHCY":  "p.Y143C most common (European); p.R49C; p.D86G (Turkish); autosomal recessive biallelic",
    "CTH":   "p.I278T; p.R336C; p.G352E; many variants identified; population frequency ~1:20,000",
}


def _make_cohort(gene: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    meta = GENE_META[gene]
    n = N_PER_GENE

    male_n = rng.randint(n // 3, 2 * n // 3)
    female_n = n - male_n

    # Onset in months
    if gene == "CBS":
        median_onset = rng.uniform(0, 6)   # NBS catches most; symptoms if missed: childhood
    elif gene in ("MTHFR", "MTR", "MTRR"):
        median_onset = rng.uniform(1, 12)  # infantile
    elif gene in ("MAT1A", "GNMT", "AHCY"):
        median_onset = rng.uniform(0, 3)   # NBS/infantile
    else:  # CTH
        median_onset = rng.uniform(0, 2)

    # B6 response
    b6_pct = 0.5 if gene == "CBS" else (0.6 if gene == "CTH" else 0.0)

    # VTE rate
    vte_pct = rng.uniform(0.3, 0.6) if gene == "CBS" else rng.uniform(0.0, 0.05)

    # Ectopia lentis
    el_pct = rng.uniform(0.7, 0.9) if gene == "CBS" else rng.uniform(0.0, 0.08)

    # Megaloblastic anemia
    mega_pct = rng.uniform(0.6, 0.85) if gene in ("MTR", "MTRR") else rng.uniform(0.0, 0.1)

    # MMA positive (should be 0 for all these genes)
    mma_pct = 0.0

    nbs_det = 0.85 if NBS_DETECTS[gene] else 0.03

    return {
        "gene":                   gene,
        "n":                      n,
        "seed":                   seed,
        "male_n":                 male_n,
        "female_n":               female_n,
        "disease":                meta["disease"],
        "locus":                  meta["locus"],
        "aa":                     meta["aa"],
        "kda":                    meta["kda"],
        "inh":                    meta["inh"],
        "methionine_direction":   METHIONINE_DIR[gene],
        "homocysteine_direction": HOMOCYSTEINE_DIR[gene],
        "mma_status":             MMA_STATUS[gene],
        "median_onset_months":    round(median_onset, 1),
        "b6_responsive":          B6_RESPONSIVE[gene],
        "b6_responsive_pct":      round(b6_pct, 2),
        "vte_pct":                round(vte_pct, 2),
        "ectopia_lentis_pct":     round(el_pct, 2),
        "megaloblastic_anemia_pct": round(mega_pct, 2),
        "mma_pct":                mma_pct,
        "nbs_detected_pct":       round(nbs_det, 2),
        "pathognomonic":          GENE_PATHOGNOMONIC[gene],
        "key_drug":               GENE_KEY_DRUG[gene],
        "founder_mutation":       FOUNDER_MUTATIONS[gene],
    }


def generate_overview() -> dict[str, Any]:
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total = sum(c["n"] for c in cohorts)

    b6_responsive_genes = [g for g in GENES if "responsive" in B6_RESPONSIVE[g].lower() and "not" not in B6_RESPONSIVE[g].lower() and "mandatory" in B6_RESPONSIVE[g].lower()]
    mma_absent_genes   = [g for g in GENES]  # all

    return {
        "atlas_id":       ATLAS_ID,
        "atlas_name":     ATLAS_NAME,
        "total_patients": total,
        "genes_count":    len(GENES),
        "seeds":          f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "gene_list":      GENES,
        "gene_cohorts":   cohorts,
        "atlas_summary": {
            "pathway_covered": "Transsulfuration (CBS, CTH) + Remethylation (MTR, MTRR, MTHFR) + SAM catabolism (MAT1A, GNMT, AHCY)",
            "b6_responsive_genes": b6_responsive_genes,
            "mma_absent_all":      True,
            "methionine_HIGH_genes":  [g for g in GENES if "HIGH" in METHIONINE_DIR[g]],
            "methionine_LOW_genes":   [g for g in GENES if "LOW" in METHIONINE_DIR[g]],
            "methionine_NORMAL_genes":[g for g in GENES if "NORMAL" in METHIONINE_DIR[g]],
            "hcy_elevated_genes":     [g for g in GENES if "HIGH" in HOMOCYSTEINE_DIR[g] or "ELEVATED" in HOMOCYSTEINE_DIR[g]],
            "critical_ddx_rule":      "Methionine direction is the FIRST DDx split: HIGH (CBS/MAT1A/GNMT/AHCY) vs LOW (MTHFR/MTR/MTRR) vs NORMAL (CTH); THEN check homocysteine and MMA",
        },
    }


def generate_breakdown() -> dict[str, Any]:
    pathway_groups = {
        "transsulfuration_CBS_CTH": {
            "genes":         ["CBS", "CTH"],
            "mechanism":     "CBS: homocysteine + serine → cystathionine (PLP-dependent); CTH: cystathionine → cysteine (PLP-dependent); complete transsulfuration pathway",
            "shared_feature":"Both PLP/B6-dependent; CBS → homocysteine HIGH + methionine HIGH; CTH → cystathionine HIGH + homocysteine NORMAL; CBS severe, CTH mostly benign",
            "key_ddx":       "CBS: ectopia lentis + VTE + marfanoid + high met + high hcy; CTH: cystathionine on amino acids + NORMAL hcy = benign; B6 trial for both",
        },
        "remethylation_folate_B12_MTR_MTRR_MTHFR": {
            "genes":         ["MTHFR", "MTR", "MTRR"],
            "mechanism":     "MTHFR: 5,10-methyleneTHF → 5-methylTHF (provides methyl donor for MTR); MTR (cblG): methionine synthase remethylates hcy → methionine using 5-methylTHF + MeCbl; MTRR (cblE): reactivates oxidised cobalamin in MTR active site",
            "shared_feature":"All three → homocysteine HIGH + methionine LOW (remethylation blocked); NO MMA (distinguishes from cblC/cblD/MMACHC); megaloblastic anemia (MTR/MTRR from methylfolate trap)",
            "key_ddx":       "All three: hcy HIGH + met LOW + MMA absent; distinguish MTHFR vs MTR/MTRR: MTHFR has no megaloblastic anemia (folate cycle differently affected); MTR vs MTRR: complementation analysis or molecular; MTRR p.I22M = common polymorphism NOT disease",
        },
        "sam_catabolism_MAT1A_GNMT_AHCY": {
            "genes":         ["MAT1A", "GNMT", "AHCY"],
            "mechanism":     "MAT1A: methionine → SAM (liver-specific isoform); GNMT: SAM → SAH (major SAM consumer in liver); AHCY: SAH → adenosine + homocysteine; all three affect SAM/SAH homeostasis",
            "shared_feature":"All three: methionine HIGH + homocysteine NORMAL/mildly elevated (AHCY: mildly elevated); NBS detects elevated methionine; KEY DDx from CBS: hcy NORMAL in MAT1A/GNMT vs hcy VERY HIGH in CBS",
            "key_ddx":       "MAT1A: met HIGH + hcy NORMAL + mostly benign; GNMT: met HIGH + AFP ELEVATED PATHOGNOMONIC + liver disease; AHCY: met HIGH + SAH HIGH + myopathy (CK elevated) DISTINCTIVE + hepatopathy; measure SAM/SAH levels to confirm GNMT/AHCY",
        },
    }

    hcy_methionine_diagnostic_cascade = [
        "STEP 1 — NBS Tandem MS: elevated METHIONINE (m/z 149) flagged → CBS, MAT1A, GNMT, AHCY; LOW methionine not detected by standard NBS (MTHFR/MTR/MTRR miss on standard NBS)",
        "STEP 2 — PLASMA AMINO ACIDS (quantitative): methionine + homocysteine + cystathionine + cysteine simultaneously; methionine direction (HIGH vs LOW) is primary DDx split",
        "STEP 3 — TOTAL HOMOCYSTEINE (tHcy) by HPLC/immunoassay: if elevated (>15 µmol/L) → pursue cause; tHcy >100 = CBS or severe remethylation defect",
        "STEP 4 — URINE ORGANIC ACIDS: check for methylmalonic acid (MMA); MMA present → cblC/cblD/MMACHC (refer to cobalamin processing atlas); MMA absent = consistent with CBS/MTHFR/MTR/MTRR/CTH",
        "STEP 5 — METHIONINE DIRECTION SPLIT: methionine HIGH + hcy HIGH → CBS, AHCY; methionine LOW + hcy HIGH → MTHFR/MTR/MTRR; methionine HIGH + hcy NORMAL → MAT1A, GNMT; cystathionine HIGH + hcy NORMAL → CTH",
        "STEP 6 — B6 TRIAL (if CBS suspected): pyridoxine 200-1000 mg/day × 4-6 weeks; if tHcy falls >50% → B6-responsive CBS; if no response → B6-non-responsive CBS; start betaine regardless",
        "STEP 7 — OPHTHALMIC EXAM (slit lamp): ectopia lentis → CBS confirmed clinically; DOWNWARD dislocation in CBS vs UPWARD in Marfan syndrome; annual exam mandatory in CBS",
        "STEP 8 — MOLECULAR GENE PANEL: CBS / MTHFR (severe AR alleles, NOT C677T) / MTR / MTRR / MAT1A / GNMT / AHCY / CTH comprehensive panel; complementation analysis for cblE vs cblG if needed",
        "STEP 9 — SAM/SAH QUANTIFICATION (GNMT/AHCY): liquid chromatography-tandem MS; SAM high + SAH very high + ratio SAH:SAM >1 → AHCY; SAM very high + SAH mildly high + AFP elevated → GNMT",
        "STEP 10 — ENZYME ASSAY (confirmatory if needed): CBS enzyme in fibroblasts/liver; cystathionase (CTH) enzyme in liver biopsy; MSR (MTRR) activity in fibroblasts via complementation",
    ]

    treatment_pathways = [
        {"gene": "CBS",   "disease": "Classical Homocystinuria", "first_line": "B6 trial (200-1000 mg/day) MANDATORY; betaine 3-6 g twice daily; low-methionine diet; folate + B12", "precision_therapy": "B6-responsive (50%): pyridoxine long-term; betaine maintains hcy <50 µmol/L", "avoid": "Fasting; N2O (inactivates cobalamin → exacerbates hcy); dehydration (thrombosis risk); high-methionine foods", "monitoring": "tHcy every 3 months (target <50 µmol/L); annual slit-lamp exam; DXA for osteoporosis; thrombosis surveillance"},
        {"gene": "MTHFR", "disease": "Severe MTHFR Deficiency", "first_line": "5-methylTHF (methylfolate/folinic acid); betaine; hydroxocobalamin; methionine supplementation (methionine low)", "precision_therapy": "Methylfolate directly supplements deficient 5-MTHF; bypasses MTHFR block", "avoid": "B6 (not helpful — MTHFR not PLP-dependent); methotrexate; N2O", "monitoring": "tHcy every 3 months; folate/B12 levels; neurological assessment; brain MRI annually if symptomatic"},
        {"gene": "MTR",   "disease": "cblG (Methionine Synthase Deficiency)", "first_line": "Hydroxocobalamin IM 1 mg daily (loading) then weekly; betaine; methylfolate; methionine supplementation", "precision_therapy": "OHCbl preferred over CNCbl — direct MeCbl substrate; IM more effective than oral", "avoid": "Cyanocobalamin (less effective); N2O (ABSOLUTELY contraindicated — irreversible MTR oxidation); fasting", "monitoring": "tHcy monthly until stable; CBC for megaloblastic anemia; methionine levels; MMA (to exclude combined defect)"},
        {"gene": "MTRR",  "disease": "cblE (Methionine Synthase Reductase Deficiency)", "first_line": "Hydroxocobalamin IM + betaine + methylfolate + methionine supplementation (identical to cblG)", "precision_therapy": "OHCbl provides cofactor; MTRR p.I22M ALONE does NOT require treatment — NOT pathogenic", "avoid": "N2O ABSOLUTELY contraindicated; cyanocobalamin less effective; fasting", "monitoring": "Identical to cblG monitoring; complement analysis if classification uncertain"},
        {"gene": "MAT1A", "disease": "Isolated Hypermethioninemia (MAT I/III)", "first_line": "Low-methionine diet if neurological symptoms or very high methionine; asymptomatic = OBSERVE ONLY", "precision_therapy": "SAM supplementation for severe SAM-deficient cases (controversial); protein restriction", "avoid": "Unnecessary dietary restriction in asymptomatic patients — most are benign; reassure families", "monitoring": "Brain MRI if symptomatic for white matter changes; liver function annually; methionine levels"},
        {"gene": "GNMT",  "disease": "GNMT Deficiency", "first_line": "Low-methionine diet; liver monitoring (LFTs, ultrasound, AFP); no proven specific treatment", "precision_therapy": "Liver transplant considered for severe progressive fibrosis; SAM-restricted diet", "avoid": "Hepatotoxic drugs; alcohol; additional liver stressors; high-methionine intake", "monitoring": "AFP every 6-12 months (chronically elevated — monitor for change); liver biopsy if fibrosis suspected; ultrasound annual"},
        {"gene": "AHCY",  "disease": "AHCY Deficiency (SAH Hydrolase)", "first_line": "Low-methionine diet; supplemental creatine (addresses GAMT inhibition myopathy component); adenosine deaminase support", "precision_therapy": "Creatine monohydrate 3-5 g/day addresses phosphocreatine depletion from SAH-mediated GAMT inhibition", "avoid": "High-methionine foods; drugs metabolised via SAM methylation (may compete); high-dose betaine (methionine already high)", "monitoring": "CK for myopathy; cardiac echo (cardiomyopathy); liver function; SAH:SAM ratio; developmental assessments"},
        {"gene": "CTH",   "disease": "Cystathioninuria (Mostly Benign)", "first_line": "NO TREATMENT NEEDED in most cases; pyridoxine 200-500 mg/day normalises cystathionine biochemically if desired", "precision_therapy": "B6 trial biochemically effective; clinical benefit minimal given benign nature", "avoid": "Unnecessary dietary restriction; invasive investigations without indication; parental anxiety amplification", "monitoring": "Reassure family; exclude secondary causes (B6 deficiency, liver disease, neuroblastoma); repeat amino acids to confirm primary CTH"},
    ]

    return {
        "atlas_id":       ATLAS_ID,
        "pathway_groups": pathway_groups,
        "hcy_methionine_diagnostic_cascade": hcy_methionine_diagnostic_cascade,
        "treatment_pathways":                treatment_pathways,
        "mma_absent_rule":                   "All 8 genes in this atlas have NO MMA; presence of MMA redirects to cobalamin processing atlas (MMACHC/cblC, MMADHC/cblD, MMAA/cblA, MMAB/cblB) or MUT/MMA",
        "n2o_warning":                       "N2O (nitrous oxide) IRREVERSIBLY OXIDISES cobalamin → acutely decompensates MTR/MTRR (cblG/cblE) and exacerbates CBS; ABSOLUTELY CONTRAINDICATED in all homocystinurias",
    }


def generate_definitions() -> dict[str, Any]:
    key_distinguishers = {
        "methionine_direction_primary_split": "CBS/MAT1A/GNMT/AHCY: methionine HIGH; MTHFR/MTR/MTRR: methionine LOW; CTH: methionine NORMAL; this is the FIRST diagnostic branch point",
        "homocysteine_ddx_second_split":      "CBS: hcy VERY HIGH (>100); MTHFR/MTR/MTRR: hcy HIGH (>50); AHCY: hcy mildly elevated; MAT1A/GNMT/CTH: hcy NORMAL",
        "mma_absent_universal":               "All 8 genes have MMA ABSENT; MMA present → different atlas (cobalamin processing, MUT)",
        "cbs_b6_mandatory_trial":             "CBS: B6 trial MANDATORY before any long-term therapy; 50% respond; B6-responsive has better prognosis; betaine given regardless of B6 response",
        "mthfr_c677t_not_disease":            "MTHFR C677T and A1298C are COMMON POLYMORPHISMS (~10-20% European homozygous rate); they do NOT cause clinical homocystinuria; severe AR MTHFR deficiency (rare, both alleles severely affected) is the actual disease",
        "mtrr_i22m_not_pathogenic":           "MTRR p.I22M (c.66A>G) is a common polymorphism (~20% European allele frequency); alone it is NOT pathogenic; severe MTRR disease requires biallelic LOF mutations",
        "cblg_vs_cble_identical":             "cblG (MTR) and cblE (MTRR) are biochemically INDISTINGUISHABLE; complementation analysis or molecular genetics required to differentiate; treatment identical",
        "ectopia_lentis_direction":           "CBS: ectopia lentis DOWNWARD; Marfan syndrome: ectopia lentis UPWARD; this single directional clue separates the two Marfanoid conditions at slit-lamp",
        "gnmt_afp_pathognomonic":             "Chronically elevated AFP (not due to HCC) in a child or young adult with hypermethioninemia = PATHOGNOMONIC for GNMT deficiency; AFP elevation is the defining biochemical marker",
        "ahcy_myopathy_distinctive":          "Elevated CK + myopathy + hypermethioninemia = AHCY until proven otherwise; no other hypermethioninemia disorder causes significant myopathy; creatine synthesis blocked by SAH",
        "cth_mostly_benign":                  "Cystathioninuria (CTH deficiency): MOSTLY BENIGN; vast majority asymptomatic; do NOT diagnose intellectual disability or seizures as causal without rigorous exclusion of other causes; early reports biased",
        "n2o_contraindication":               "N2O ABSOLUTELY CONTRAINDICATED in CBS, MTHFR, MTR (cblG), MTRR (cblE) — irreversibly oxidises cobalamin; in CBS: depletes methylcobalamin needed for remethylation backup; acute decompensation risk",
    }

    treatment_matrix = {
        "betaine_role":          "Trimethylglycine (betaine) remethylates homocysteine → methionine via BHMT (betaine-homocysteine methyltransferase), independent of folate/B12/CBS; effective in CBS regardless of B6 response; effective in remethylation defects as second remethylation pathway; caution: raises methionine further in CBS — monitor",
        "hydroxocobalamin_vs_cyanocobalamin": "OHCbl preferred for cblG/cblE — directly provides active cobalamin; cyanocobalamin must be converted by multiple steps; OHCbl IM achieves higher tissue levels; in cblC/cblD also OHCbl preferred (see cobalamin processing atlas)",
        "low_methionine_diet_indications":   "CBS: mandatory (reduce substrate); MTHFR/MTR/MTRR: NOT indicated (methionine already low — restriction worsens); MAT1A: only if symptomatic; GNMT: consider; AHCY: moderate restriction; CTH: not needed",
        "pyridoxine_b6_indications":         "CBS: trial MANDATORY (50% responsive); CTH: responsive (benign disease); MTHFR/MTR/MTRR: NOT responsive (different enzyme class); MAT1A/GNMT/AHCY: not applicable",
        "folate_b12_all_remethylation":      "MTHFR/MTR/MTRR: folate/methylfolate + B12 (hydroxocobalamin) ESSENTIAL; CBS: supplemental folate/B12 reduces homocysteine moderately (cofactor support); others: less critical",
        "creatine_supplementation_ahcy":     "AHCY: creatine supplementation addresses SAH-mediated GAMT (guanidinoacetate methyltransferase) inhibition → restores phosphocreatine in muscle and heart; 3-5 g/day creatine monohydrate",
    }

    definitions = [
        {"term": "Homocystinuria", "definition": "Disorder of sulfur amino acid metabolism causing elevated total homocysteine (tHcy) in plasma and urine homocystine; primary causes: CBS deficiency (transsulfuration block), MTHFR/MTR/MTRR (remethylation defects); secondary causes: nutritional B6/B12/folate deficiency; tHcy >100 µmol/L strongly suggests genetic cause; tHcy 15-100 can be genetic or nutritional"},
        {"term": "Transsulfuration Pathway", "definition": "One-way metabolic route from homocysteine to cysteine via cystathionine: CBS catalyses homocysteine + serine → cystathionine; CTH catalyses cystathionine → cysteine + alpha-ketobutyrate; irreversible; makes cysteine essential in CBS/CTH deficiency; PLP (B6) required for both steps"},
        {"term": "Remethylation Cycle", "definition": "Regenerates methionine from homocysteine via two pathways: (1) 5-methylTHF + homocysteine → methionine + THF [MTR/methionine synthase, uses methylcobalamin]; (2) betaine + homocysteine → methionine + dimethylglycine [BHMT, liver/kidney only]; MTHFR makes 5-methylTHF; MTR uses it; MTRR reactivates MTR cobalamin"},
        {"term": "S-Adenosylmethionine (SAM)", "definition": "Universal methyl donor synthesised from methionine + ATP by MAT (MAT1A in liver, MAT2 ubiquitous); donates methyl group to hundreds of substrates (DNA, RNA, proteins, phospholipids, creatine, neurotransmitters); SAM → SAH after methyl transfer; SAH inhibits all SAM-dependent methyltransferases (product inhibition); AHCY hydrolyses SAH → adenosine + homocysteine to relieve inhibition"},
        {"term": "Ectopia Lentis", "definition": "Lens dislocation/subluxation; in CBS deficiency: DOWNWARD dislocation (fibrillin disruption by homocysteine); in Marfan syndrome: UPWARD dislocation (FBN1 mutation primary); both cause severe myopia, glaucoma risk; differentiation: direction + biochemistry (homocysteine elevated in CBS, normal in Marfan)"},
        {"term": "Methylfolate Trap", "definition": "In MTR/MTRR deficiency: 5-methylTHF accumulates because MTR cannot accept its methyl group (cobalamin inactive); other THF forms depleted; functional folate deficiency despite normal/high plasma folate; causes megaloblastic anemia (impaired DNA synthesis) + elevated homocysteine; treat with hydroxocobalamin (reactivates MTR) + betaine"},
        {"term": "B6/Pyridoxine Responsiveness in CBS", "definition": "~50% of CBS mutations retain residual CBS enzyme activity enhanced by pharmacological pyridoxine (PLP cofactor); response = >50% reduction in tHcy after 4-6 weeks pyridoxine 200-1000 mg/day; B6-responsive patients: milder phenotype, better prognosis, lower ectopia lentis/VTE rates; MANDATORY to trial ALL CBS patients before classifying as non-responsive"},
        {"term": "Betaine (Trimethylglycine)", "definition": "Acts as methyl donor via BHMT (betaine-homocysteine methyltransferase) to remethylate homocysteine → methionine; functions as betaine backup for impaired remethylation or as bypass for CBS block; dose 3-6 g twice daily; raises methionine (monitor in CBS); may cause cerebral oedema at very high doses (rare); effective regardless of B6 response in CBS"},
        {"term": "Alpha-Fetoprotein (AFP) in GNMT Deficiency", "definition": "AFP is chronically and persistently elevated in GNMT deficiency from childhood (not due to HCC); AFP levels may exceed 1000 ng/mL; mechanism unclear — possible SAM effects on liver regeneration or hepatic injury markers; distinguish from AFP elevation in HCC (typically adults, rising trend, liver mass); serial AFP monitoring in GNMT deficiency"},
        {"term": "S-Adenosylhomocysteine (SAH)", "definition": "Product of SAM after methyl transfer; potent competitive inhibitor of all SAM-dependent methyltransferases (Ki ~1-10 µM); AHCY hydrolyses SAH → adenosine + homocysteine; in AHCY deficiency: SAH accumulates → globally inhibits methylation → DNA/protein/phospholipid hypomethylation; SAH:SAM ratio >1 indicates severely impaired methylation capacity"},
        {"term": "Cystathionine", "definition": "Intermediate in transsulfuration: formed from homocysteine + serine (CBS step) → cleaved to cysteine (CTH step); elevated in CTH deficiency (cystathioninuria); also mildly elevated in CBS deficiency (product of CBS that accumulates slightly); normal plasma cystathionine <5 µmol/L; cystathioninuria >50 µmol/L suggests CTH deficiency or B6 deficiency"},
        {"term": "Nitrous Oxide (N2O) Contraindication", "definition": "N2O irreversibly oxidises cob(I)alamin to cob(III)alamin within MTR active site, permanently inactivating methionine synthase; in patients with MTR/MTRR deficiency (cblG/cblE): single N2O exposure can cause catastrophic acute neurological decompensation; in CBS: depletes methylcobalamin remethylation backup; ABSOLUTELY CONTRAINDICATED as anaesthetic in all homocystinurias"},
        {"term": "MTHFR C677T Polymorphism vs Disease", "definition": "C677T (p.Ala222Val) creates thermolabile MTHFR with reduced activity at 37°C; homozygous C677T: mildly elevated tHcy (12-18 µmol/L); population prevalence: 10-15% homozygous European; this is NOT clinical homocystinuria; severe AR MTHFR deficiency requires biallelic strongly damaging alleles; tHcy >50 µmol/L in homozygous C677T is unusual and should prompt workup for other causes"},
        {"term": "Cascade Biochemical Testing Algorithm", "definition": "tHcy → methionine direction split → MMA status → gene panel → enzyme assay → complementation; methionine HIGH+hcy HIGH = CBS/AHCY territory; methionine LOW+hcy HIGH = MTHFR/MTR/MTRR territory; methionine HIGH+hcy NORMAL = MAT1A/GNMT territory; cystathionine HIGH+hcy NORMAL = CTH; MMA always checked to exclude cobalamin processing disorders"},
    ]

    return {
        "atlas_id":           ATLAS_ID,
        "key_distinguishers": key_distinguishers,
        "treatment_matrix":   treatment_matrix,
        "definitions":        definitions,
        "seeds":              f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
    }
