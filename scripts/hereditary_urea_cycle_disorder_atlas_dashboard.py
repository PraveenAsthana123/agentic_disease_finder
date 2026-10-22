#!/usr/bin/env python3
"""Hereditary-Urea-Cycle-Disorder-Atlas -- Complete 8-Gene Reference
Urea Cycle & Related Nitrogen Disposal Disorders: CPS1-OTC-ASS1-ASL-ARG1-NAGS-SLC25A15-SLC25A13
(CPS1/CPS1-Deficiency-Neonatal-Hyperammonemia-OroticAcid-ABSENT-PATHOGNOMONIC-CarglumateTrial-NAGS-DDx,
 OTC/OTC-Deficiency-MOST-COMMON-UCD-X-Linked-OroticAcid-MARKEDLY-ELEVATED-PATHOGNOMONIC-AllopurinolTest-Carriers,
 ASS1/Citrullinemia-Type1-CTLN1-Citrulline-GT1000-NBS-DETECTABLE-PATHOGNOMONIC-ArgSupplementation,
 ASL/Argininosuccinic-Aciduria-ArgininosuccinateInPlasmaUrine-PATHOGNOMONIC-HairFragility-Hypertension,
 ARG1/Arginemia-SpasticDiplegia-PATHOGNOMONIC-Arginine-MARKEDLY-HIGH-Hyperammonemia-MILD,
 NAGS/NAGS-Deficiency-Rarest-UCD-CarglumateCURATIVE-EMPIRICAL-ANY-CPS1-Like-Orotic-ABSENT,
 SLC25A15/HHH-Syndrome-Ornithine-Hyperammonemia-Homocitrullinuria-TRIAD-PATHOGNOMONIC-Canadian-Founder,
 SLC25A13/Citrin-Deficiency-NICCD-CTLN2-EastAsian-Founder-BiofoodPreference-PATHOGNOMONIC-LivTx-URGENT):

CPS1  (Carbamoyl phosphate synthetase 1, mitochondrial; 1500aa; 2q35; AR-LOF;
     CPS1 encodes the first and rate-limiting enzyme of the urea cycle in the mitochondrial matrix;
     CPS1 catalyzes: NH3 + HCO3- + 2 ATP -> carbamoyl phosphate + 2 ADP + Pi;
     CPS1 activity REQUIRES N-acetylglutamate (NAG) as obligatory allosteric activator;
     CPS1 LOF -> complete block in urea cycle at first step -> severe hyperammonemia;
     CPS1 DEFICIENCY -- OMIM 237300; also called hyperammonemia type I;
     NEONATAL PRESENTATION: rapidly progressive within 24-72h after birth; NH3 >1000 umol/L (normal <50);
     CLINICAL: lethargy -> coma -> cerebral edema -> death if untreated; seizures, respiratory failure;
     CRITICAL BIOMARKER: OROTIC ACID ABSENT or very LOW in urine -- PATHOGNOMONIC distinction from OTC deficiency;
       orotic acid absent because the metabolic block is BEFORE the cytoplasmic carbamoyl phosphate pool;
     PLASMA AMINO ACIDS: glutamine VERY HIGH (nitrogen carrier overload), alanine HIGH, citrulline VERY LOW/absent, arginine LOW;
     DIFFERENTIAL DIAGNOSIS from OTC: both have absent orotic acid -> MUST test for NAGS deficiency via carglumic acid trial;
       if NH3 normalizes with carglumic acid -> NAGS (not CPS1); molecular testing distinguishes;
     TREATMENT: emergency: hemodialysis (fastest NH3 removal > peritoneal > exchange transfusion);
       ammonia scavengers: sodium benzoate 250-500 mg/kg/day IV + sodium phenylacetate 250-500 mg/kg/day IV (Ammonul);
       arginine supplementation: 200-800 mg/kg/day (can't synthesize arginine without urea cycle);
       citrulline supplementation (alternative to arginine, may be better tolerated);
       low protein diet, nitrogen restriction, supplemental calories (prevent catabolism);
     LIVER TRANSPLANT: curative for hyperammonemia (donor hepatocytes perform normal ureagenesis);
       neonatal transplant possible in severe cases; post-transplant protein restriction not required;
     GENETICS: largest UCD gene (1500aa); many private mutations; enzyme activity in liver biopsy;
     NBS: NOT detected by standard NBS (not an amino acid or acylcarnitine marker);
     seed SEED_BASE+0).

OTC   (Ornithine transcarbamylase; 354aa; Xp11.4; XLR;
     OTC encodes ornithine transcarbamylase, the second enzyme of the urea cycle (mitochondrial matrix);
     OTC catalyzes: carbamoyl phosphate + ornithine -> citrulline + Pi;
     OTC LOF -> carbamoyl phosphate accumulates -> spills to cytoplasm -> enters pyrimidine synthesis -> OROTIC ACID ELEVATED;
     OTC DEFICIENCY -- OMIM 311250; most common urea cycle disorder (~1:14,000-40,000 males);
     X-LINKED: males with null mutations -> fatal neonatal hyperammonemia;
       female carriers (heterozygotes): VARIABLE -- lyonization determines severity (can be severely affected);
       ~15-20% of all OTC carriers have clinically significant disease; late-onset presentations common;
     OROTIC ACID MARKEDLY ELEVATED in urine -> PATHOGNOMONIC DDx from CPS1/NAGS;
       orotic acid elevation: carbamoyl phosphate overflow into cytoplasm -> orotate -> orotic acid;
     ALLOPURINOL LOADING TEST: used to detect asymptomatic female carriers;
       allopurinol blocks orotidine -> orotic acid and orotidine BOTH accumulate in carriers;
     PLASMA AMINO ACIDS: glutamine HIGH, alanine HIGH, citrulline VERY LOW/absent (same as CPS1);
     CRITICAL DISTINCTION from CPS1: OROTIC ACID -> HIGH in OTC vs LOW/ABSENT in CPS1;
     CLINICAL: neonatal males same as CPS1; female carriers: episodic hyperammonemia (protein-triggered, illness-triggered);
       protein aversion: behavioral avoidance of protein-rich foods (self-protective phenomenon);
     TREATMENT: identical to CPS1 -- benzoate + phenylacetate + arginine + low protein diet;
       liver transplant curative for hyperammonemia;
     NBS: NOT detected by standard NBS;
     seed SEED_BASE+1).

ASS1  (Argininosuccinate synthetase 1; 412aa; 9q34.11; AR-LOF;
     ASS1 encodes argininosuccinate synthetase, third enzyme of the urea cycle (cytoplasmic);
     ASS1 catalyzes: citrulline + aspartate + ATP -> argininosuccinate + AMP + PPi;
     ASS1 LOF -> citrulline ACCUMULATES in blood -> CITRULLINEMIA TYPE 1 (CTLN1);
     CITRULLINEMIA TYPE 1 -- OMIM 215700;
     PLASMA AMINO ACIDS: CITRULLINE MARKEDLY ELEVATED (>1000 umol/L in neonatal form, normal <50 umol/L) -> PATHOGNOMONIC;
     NBS-DETECTABLE: citrulline elevated on tandem mass spectrometry (standard NBS panel in most countries);
     CLINICAL FORMS: classic neonatal (NH3 >1000 umol/L) or mild/late-onset (partial enzyme);
     DIFFERENTIAL from ASL deficiency: ASL ALSO has elevated citrulline BUT ALSO argininosuccinic acid present;
       pure citrulline elevation without argininosuccinic acid -> ASS1 until proven otherwise;
     ARGININE SUPPLEMENTATION: CRITICAL -- ASS1 block prevents arginine synthesis -> arginine ESSENTIAL;
       arginine supplementation 200-800 mg/kg/day;
     LIVER TRANSPLANT: curative; hepatic ASS1 activity restored;
     seed SEED_BASE+2).

ASL   (Argininosuccinate lyase; 464aa; 7q11.21; AR-LOF;
     ASL encodes argininosuccinate lyase, fourth enzyme of the urea cycle (cytoplasmic + nuclear);
     ASL catalyzes: argininosuccinate -> arginine + fumarate;
     ASL LOF -> argininosuccinate ACCUMULATES in blood, urine, CSF;
     ARGININOSUCCINIC ACIDURIA (ASA) -- OMIM 207900;
     ARGININOSUCCINIC ACID in plasma AND urine -> PATHOGNOMONIC (detected by HPLC/tandem MS);
     NBS-DETECTABLE: argininosuccinate detected by tandem mass spec (plus elevated citrulline);
     HAIR FRAGILITY -- TRICHORRHEXIS NODOSA -> PATHOGNOMONIC of long-term ASL;
       hair shafts fracture easily (trichorrhexis nodosa on hair microscopy) in treated survivors;
       caused by arginine deficiency in hair follicles (arginine needed for creatine in hair matrix);
       NOT a vitamin deficiency -- pathognomonic for ASL specifically;
     HYPERTENSION: common in treated ASL patients (even with controlled NH3) -- PATHOGNOMONIC of ASL;
       ASL also functions in arginine-citrulline cycle in endothelium -> NO synthesis requires ASL;
       NO deficiency -> systemic hypertension;
     LIVER TRANSPLANT: corrects hepatic hyperammonemia but does NOT correct hair fragility or hypertension;
       systemic (extrahepatic) ASL deficiency persists post-liver transplant -- UNIQUE among UCDs;
     seed SEED_BASE+3).

ARG1  (Arginase 1, liver-type; 330aa; 6q23.2; AR-LOF;
     ARG1 encodes arginase 1, the fifth and final cytoplasmic enzyme of the urea cycle;
     ARG1 catalyzes: arginine -> ornithine + urea;
     ARG1 LOF -> arginine ACCUMULATES; urea cycle still partially functional (ornithine pathway);
     ARGINEMIA -- OMIM 207800;
     UNIQUE PHENOTYPE -- COMPLETELY DIFFERENT from other UCDs:
       HYPERAMMONEMIA IS MILD OR ABSENT -- unlike other UCDs, arginemia rarely causes neonatal crisis;
       PRIMARY PRESENTATION: PROGRESSIVE SPASTIC DIPLEGIA/PARAPARESIS -> PATHOGNOMONIC;
         spasticity, scissor gait, developmental regression, cerebral palsy-like picture;
         caused by arginine/guanidino compound neurotoxicity;
     ARGININE MARKEDLY ELEVATED in plasma (>200-500 umol/L, normal <120) -> PATHOGNOMONIC;
     TREATMENT: DIETARY ARGININE RESTRICTION (different from other UCDs which need arginine supplementation);
       low protein diet + essential amino acid supplementation;
       ornithine or citrulline supplementation (bypass to reduce arginine accumulation);
       ammonia scavengers generally NOT needed (hyperammonemia mild);
     LIVER TRANSPLANT: corrects arginine elevation but neurological damage may not fully reverse;
     seed SEED_BASE+4).

NAGS  (N-acetylglutamate synthase; 534aa; 17q21.31; AR-LOF;
     NAGS encodes N-acetylglutamate synthase, which produces NAG (N-acetylglutamate) from glutamate + acetyl-CoA;
     NAG is the OBLIGATORY ALLOSTERIC ACTIVATOR of CPS1 -- without NAG, CPS1 is inactive;
     NAGS LOF -> NAG deficiency -> CPS1 inactive -> identical phenotype to CPS1 deficiency;
     NAGS DEFICIENCY -- OMIM 237310; RAREST urea cycle disorder (~50 cases worldwide);
     PHENOTYPE: IDENTICAL to CPS1 -- neonatal hyperammonemia + orotic acid ABSENT;
       biochemically INDISTINGUISHABLE from CPS1 without specific testing or molecular;
     PATHOGNOMONIC TREATMENT RESPONSE: CARGLUMIC ACID (N-carbamylglutamate, NCG) is COMPLETELY CURATIVE;
       NCG is a stable analogue of NAG that activates CPS1 even without NAGS;
       dose: 100-250 mg/kg/day orally; hyperammonemia resolves COMPLETELY and PERMANENTLY on NCG;
       NCG trial is BOTH DIAGNOSTIC AND THERAPEUTIC simultaneously;
     CRITICAL RULE: IN ANY PATIENT WITH NEONATAL HYPERAMMONEMIA + OROTIC ACID ABSENT:
       ALWAYS GIVE CARGLUMIC ACID EMPIRICALLY BEFORE MOLECULAR RESULTS ARE AVAILABLE;
       if NH3 normalizes completely on NCG -> NAGS deficiency confirmed;
       if NO response to NCG -> CPS1 deficiency (then liver transplant planning);
       NCG is SAFE and has no significant adverse effects;
     SPECIAL SITUATIONS: acquired NAGS deficiency in propionic/isovaleric acidemia ->
       NCG may also benefit severe hyperammonemia in organic acidemias;
     NBS: NOT detected;
     seed SEED_BASE+5).

SLC25A15 (Solute carrier family 25 member 15 / ORNT1 / Ornithine/citrulline carrier; 301aa; 13q14.11; AR-LOF;
     SLC25A15 encodes ORNT1, the mitochondrial ornithine/citrulline antiporter;
     ORNT1 imports ornithine from cytoplasm into mitochondria (where OTC uses it) and exports citrulline;
     SLC25A15 LOF -> ornithine CANNOT enter mitochondria -> accumulates in cytoplasm;
       -> intramitochondrial ornithine depletion -> OTC has no substrate -> carbamoyl phosphate buildup;
       -> carbamoyl phosphate spills to cytoplasm -> orotate -> OROTIC ACIDURIA;
       -> excess mitochondrial carbamoyl phosphate carbamylates lysine -> HOMOCITRULLINE excreted;
     HHH SYNDROME -- OMIM 238970 (Hyperornithinemia-Hyperammonemia-Homocitrullinuria);
     PATHOGNOMONIC TRIAD:
       (1) HYPERORNITHINEMIA (ornithine >200-400 umol/L, normal <100);
       (2) EPISODIC HYPERAMMONEMIA (mild/moderate, trigger-induced);
       (3) HOMOCITRULLINURIA (homocitrulline in urine, PATHOGNOMONIC -- not seen in other UCDs);
     CLINICAL: NEUROLOGICAL DEGENERATION DOMINATES (unlike classic UCDs where hyperammonemia dominates);
       cerebellar ataxia (progressive), pyramidal signs, cognitive decline, spastic paraparesis;
       hyperammonemia is episodic and often MILDER than CPS1/OTC;
     CANADIAN FOUNDER: French Canadian population -- p.F188del (c.562delTTT) high frequency;
     TREATMENT: LOW PROTEIN diet + ornithine restriction;
       citrulline supplementation (provides intramitochondrial substrate bypassing ORNT1);
       ammonia scavengers for acute crises;
     seed SEED_BASE+6).

SLC25A13 (Solute carrier family 25 member 13 / Citrin; 675aa; 7q21.3; AR-LOF;
     SLC25A13 encodes citrin, the mitochondrial aspartate/glutamate carrier isoform 2 (AGC2);
     Citrin exchanges mitochondrial aspartate for cytoplasmic glutamate + H+;
     Critical roles: (1) malate-aspartate shuttle (NADH recycling); (2) aspartate supply for ASS1 reaction;
     SLC25A13 LOF -> cytoplasmic aspartate deficiency -> ASS1 substrate deficient -> citrulline accumulates;
       also -> NADH/NAD+ imbalance in cytoplasm -> impaired gluconeogenesis, lipid metabolism;
     CITRIN DEFICIENCY -- TWO DISTINCT AGE-DEPENDENT PHENOTYPES:
     (1) NICCD (Neonatal Intrahepatic Cholestasis by Citrin Deficiency) -- OMIM 605814:
         neonatal/infantile cholestasis, mild transaminase elevation, hypoalbuminemia;
         NBS shows CITRULLINE ELEVATED + GALACTOSE ELEVATED + METHIONINE ELEVATED (NBS pseudo-galactosemia);
         usually SELF-RESOLVING by age 1-2 years with lactose-free MCT formula;
     (2) CTLN2 (Citrullinemia Type II, Adult-Onset Citrullinemia) -- OMIM 603859:
         adult-onset (20-70yr) RECURRENT HYPERAMMONEMIC ENCEPHALOPATHY;
         BIZARRE FOOD PREFERENCES -> PATHOGNOMONIC: strong craving for protein/fat/nuts/tofu/alcohol;
           AVERSION to carbohydrates and sweets (self-selection of metabolically correct diet);
         NOCTURNAL HYPERAMMONEMIA: episodes often worse at night (fasting state);
         LETHAL: untreated CTLN2 can cause fatal hyperammonemic coma;
     EAST ASIAN FOUNDER: Japanese 1:17,000 births, Chinese, Korean;
       IVS16ins3kb deletion most common; 851del4, IVS6-14G>T also common;
       RARE in Europeans and Africans;
     CRITICAL TREATMENT RULE FOR CTLN2: LIVER TRANSPLANT URGENTLY CURATIVE;
       ABSOLUTELY CONTRAINDICATED: IV glycerol, high-glucose infusions, hyperosmolar solutions ->
         can trigger fatal hyperammonemic crisis (worsens NADH imbalance);
       LOW CARBOHYDRATE DIET (medium-chain triglyceride oil) supports patients awaiting transplant;
       NICCD: lactose-free MCT formula -> most resolve without transplant;
     seed SEED_BASE+7).

SEED_BASE = 4214
"""

import json
import random
from typing import Any


ATLAS_ID   = "hereditary-urea-cycle-disorder-atlas"
ATLAS_NAME = "Hereditary-Urea-Cycle-Disorder-Atlas"
SEED_BASE  = 4214
N_PER_GENE = 40

GENES = ["CPS1", "OTC", "ASS1", "ASL", "ARG1", "NAGS", "SLC25A15", "SLC25A13"]

GENE_META = {
    "CPS1":     {"aa": 1500, "locus": "2q35",    "kda": 165, "inh": "AR",  "disease": "CPS1 Deficiency / Hyperammonemia Type I",                          "omim_gene": 608307, "omim_disease": 237300},
    "OTC":      {"aa": 354,  "locus": "Xp11.4",  "kda": 39,  "inh": "XLR", "disease": "OTC Deficiency / Hyperammonemia Type II (X-linked)",               "omim_gene": 300461, "omim_disease": 311250},
    "ASS1":     {"aa": 412,  "locus": "9q34.11", "kda": 46,  "inh": "AR",  "disease": "Citrullinemia Type I (CTLN1)",                                     "omim_gene": 603470, "omim_disease": 215700},
    "ASL":      {"aa": 464,  "locus": "7q11.21", "kda": 52,  "inh": "AR",  "disease": "Argininosuccinic Aciduria (ASA)",                                  "omim_gene": 608310, "omim_disease": 207900},
    "ARG1":     {"aa": 330,  "locus": "6q23.2",  "kda": 35,  "inh": "AR",  "disease": "Arginemia (ARG1 deficiency)",                                      "omim_gene": 608313, "omim_disease": 207800},
    "NAGS":     {"aa": 534,  "locus": "17q21.31","kda": 58,  "inh": "AR",  "disease": "NAGS Deficiency / Hyperammonemia (carglumic-acid-responsive)",      "omim_gene": 608300, "omim_disease": 237310},
    "SLC25A15": {"aa": 301,  "locus": "13q14.11","kda": 34,  "inh": "AR",  "disease": "HHH Syndrome (Hyperornithinemia-Hyperammonemia-Homocitrullinuria)", "omim_gene": 603861, "omim_disease": 238970},
    "SLC25A13": {"aa": 675,  "locus": "7q21.3",  "kda": 74,  "inh": "AR",  "disease": "Citrin Deficiency (NICCD / Citrullinemia Type II / CTLN2)",        "omim_gene": 603859, "omim_disease": 605814},
}

GENE_PATHOGNOMONIC = {
    "CPS1":     "OROTIC ACID ABSENT in urine PATHOGNOMONIC (block before cytoplasmic CP pool); glutamine VERY HIGH; NH3 >1000 umol/L neonatal; ALWAYS trial carglumic acid (NCG) empirically -- if NH3 normalizes = NAGS not CPS1; liver transplant curative",
    "OTC":      "OROTIC ACID MARKEDLY ELEVATED PATHOGNOMONIC (distinguishes OTC from CPS1/NAGS); allopurinol loading test detects asymptomatic female carriers; X-linked -- male hemizygotes fatal neonatal; females variable lyonization-dependent severity",
    "ASS1":     "CITRULLINE >1000 umol/L PATHOGNOMONIC neonatal form; NBS-detectable (tandem MS); NO argininosuccinic acid present (DDx from ASL); arginine supplementation MANDATORY (cannot synthesize arginine); liver transplant curative",
    "ASL":      "ARGININOSUCCINIC ACID in plasma/urine PATHOGNOMONIC (two peaks on HPLC); HAIR FRAGILITY trichorrhexis nodosa PATHOGNOMONIC in survivors; HYPERTENSION (NO synthesis defect) persists post-liver-transplant; liver transplant corrects NH3 NOT hair/HTN",
    "ARG1":     "PROGRESSIVE SPASTIC DIPLEGIA not hyperammonemia PATHOGNOMONIC; ARGININE >200 umol/L markedly elevated; DIETARY ARGININE RESTRICTION (OPPOSITE of other UCDs); NO ammonia scavengers typically needed; cerebral palsy misdiagnosis common 2-4yr onset",
    "NAGS":     "CARGLUMIC ACID (NCG) COMPLETELY CURATIVE PATHOGNOMONIC response; identical biochemistry to CPS1 (orotic acid absent); EMPIRICALLY GIVE NCG 100 mg/kg in any CPS1-like presentation BEFORE molecular results; NH3 normalization confirms NAGS; rarest UCD ~50 worldwide",
    "SLC25A15": "HOMOCITRULLINE in urine PATHOGNOMONIC (not present in any other UCD); TRIAD hyperornithinemia + hyperammonemia + homocitrullinuria; NEUROLOGICAL DEGENERATION dominates (cerebellar ataxia, pyramidal); French Canadian p.F188del founder; citrulline supplement bypasses ORNT1",
    "SLC25A13": "BIZARRE FOOD PREFERENCES (protein/fat craving, carbohydrate aversion) PATHOGNOMONIC for CTLN2 in adult East Asian; NICCD: NBS citrulline+galactose+methionine elevated (pseudo-galactosemia); ABSOLUTELY CI: IV glycerol/high-glucose -> fatal crisis; liver transplant URGENTLY CURATIVE for CTLN2",
}

GENE_GROUPS = {
    "CPS1":     "proximal-ucd",
    "OTC":      "proximal-ucd",
    "ASS1":     "distal-ucd",
    "ASL":      "distal-ucd",
    "ARG1":     "distal-ucd",
    "NAGS":     "activator-ucd",
    "SLC25A15": "transport-ucd",
    "SLC25A13": "transport-ucd",
}

PROXIMAL_GROUP  = ["CPS1", "OTC"]
DISTAL_GROUP    = ["ASS1", "ASL", "ARG1"]
ACTIVATOR_GROUP = ["NAGS"]
TRANSPORT_GROUP = ["SLC25A15", "SLC25A13"]


def _make_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    meta = GENE_META[gene]

    is_xlr = meta["inh"] == "XLR"
    is_east_asian_founder = gene == "SLC25A13"
    is_canadian_founder   = gene == "SLC25A15"
    nbs_detectable        = gene in ("ASS1", "ASL", "ARG1")
    orotic_absent         = gene in ("CPS1", "NAGS")
    orotic_elevated       = gene == "OTC"
    neonatal_crisis       = gene not in ("ARG1", "SLC25A15", "SLC25A13")
    has_spastic           = gene == "ARG1"
    has_hair_fragility    = gene == "ASL"
    has_htn               = gene == "ASL"
    has_homocitrullinuria = gene == "SLC25A15"
    has_ctln2             = gene == "SLC25A13"
    carglumate_curative   = gene == "NAGS"
    liver_tx_fully_curative = gene in ("CPS1", "OTC", "ASS1", "NAGS", "SLC25A13")

    ages = [rng.uniform(0, 5) if neonatal_crisis else rng.uniform(0.1, 60) for _ in range(N_PER_GENE)]
    nh3_vals = []
    for _ in range(N_PER_GENE):
        if neonatal_crisis:
            nh3_vals.append(rng.uniform(300, 2000))
        elif gene == "ARG1":
            nh3_vals.append(rng.uniform(50, 200))
        elif gene == "SLC25A15":
            nh3_vals.append(rng.uniform(80, 400))
        elif gene == "SLC25A13":
            nh3_vals.append(rng.uniform(60, 350))
        else:
            nh3_vals.append(rng.uniform(100, 600))

    genders = ["M", "F"] if not is_xlr else (["M"] * int(N_PER_GENE * 0.7) + ["F"] * int(N_PER_GENE * 0.3))
    rng.shuffle(genders)
    while len(genders) < N_PER_GENE:
        genders.append(rng.choice(["M", "F"]))

    outcomes = []
    for nh3 in nh3_vals:
        if nh3 > 800:
            outcomes.append(rng.choice(["severe", "severe", "moderate"]))
        elif nh3 > 300:
            outcomes.append(rng.choice(["moderate", "moderate", "mild", "severe"]))
        else:
            outcomes.append(rng.choice(["mild", "mild", "moderate"]))

    biomarker_vals = {}
    for i in range(N_PER_GENE):
        if gene == "CPS1":
            biomarker_vals[i] = {"orotic_acid": rng.uniform(0, 5), "glutamine": rng.uniform(800, 2000),
                                  "citrulline": rng.uniform(1, 15), "arginine": rng.uniform(5, 30)}
        elif gene == "OTC":
            biomarker_vals[i] = {"orotic_acid": rng.uniform(50, 500), "glutamine": rng.uniform(600, 1500),
                                  "citrulline": rng.uniform(1, 10), "arginine": rng.uniform(5, 20)}
        elif gene == "ASS1":
            biomarker_vals[i] = {"citrulline": rng.uniform(500, 3000), "orotic_acid": rng.uniform(5, 50),
                                  "glutamine": rng.uniform(300, 800)}
        elif gene == "ASL":
            biomarker_vals[i] = {"argininosuccinate": rng.uniform(100, 600), "citrulline": rng.uniform(100, 600),
                                  "arginine": rng.uniform(20, 100)}
        elif gene == "ARG1":
            biomarker_vals[i] = {"arginine": rng.uniform(200, 800), "guanidino_acetic_acid": rng.uniform(50, 300),
                                  "orotic_acid": rng.uniform(5, 30)}
        elif gene == "NAGS":
            biomarker_vals[i] = {"orotic_acid": rng.uniform(0, 5), "glutamine": rng.uniform(700, 1800),
                                  "citrulline": rng.uniform(1, 15), "nag_activity": rng.uniform(0, 20)}
        elif gene == "SLC25A15":
            biomarker_vals[i] = {"ornithine": rng.uniform(200, 600), "homocitrulline_urine": rng.uniform(30, 200),
                                  "orotic_acid": rng.uniform(10, 80), "citrulline": rng.uniform(20, 80)}
        elif gene == "SLC25A13":
            biomarker_vals[i] = {"citrulline": rng.uniform(50, 400), "galactose_nbs": rng.uniform(8, 40),
                                  "methionine": rng.uniform(50, 200), "threonine": rng.uniform(200, 600)}

    treatments = {
        "CPS1":     ["benzoate+phenylacetate IV", "arginine IV", "hemodialysis", "low_protein_diet", "liver_transplant"],
        "OTC":      ["benzoate+phenylbutyrate", "arginine", "low_protein_diet", "hemodialysis", "liver_transplant"],
        "ASS1":     ["arginine_supplementation", "benzoate+phenylacetate", "citrulline", "low_protein_diet", "liver_transplant"],
        "ASL":      ["arginine_supplementation", "benzoate+phenylacetate", "low_protein_diet", "liver_transplant"],
        "ARG1":     ["arginine_restriction_diet", "low_protein_diet", "citrulline_supplementation", "liver_transplant"],
        "NAGS":     ["carglumic_acid_NCG", "benzoate+phenylacetate", "arginine", "low_protein_diet"],
        "SLC25A15": ["low_protein_diet", "ornithine_restriction", "citrulline_supplementation", "ammonia_scavengers"],
        "SLC25A13": ["MCT_formula_NICCD", "lactose_free_diet_NICCD", "low_carbohydrate_diet_CTLN2", "liver_transplant_CTLN2"],
    }

    patients = []
    for i in range(N_PER_GENE):
        p = {
            "id":        f"{gene}-{seed}-{i:03d}",
            "seed":      seed,
            "gene":      gene,
            "gender":    genders[i] if i < len(genders) else rng.choice(["M", "F"]),
            "age_dx":    round(ages[i], 2),
            "nh3_peak":  round(nh3_vals[i], 1),
            "outcome":   outcomes[i],
            "biomarkers":biomarker_vals[i],
            "treatment": rng.choice(treatments[gene]),
            "orotic_acid_elevated":    orotic_elevated,
            "orotic_acid_absent":      orotic_absent,
            "nbs_detected":            nbs_detectable and rng.random() < 0.85,
            "neonatal_crisis":         neonatal_crisis and ages[i] < 0.1,
            "has_spastic_diplegia":    has_spastic and outcomes[i] != "mild",
            "has_hair_fragility":      has_hair_fragility and ages[i] > 2.0,
            "has_hypertension":        has_htn and ages[i] > 3.0,
            "has_homocitrullinuria":   has_homocitrullinuria,
            "has_ctln2_food_pref":     has_ctln2 and ages[i] > 18,
            "carglumate_response":     carglumate_curative,
            "liver_tx_fully_curative": liver_tx_fully_curative,
            "ucd_group":               GENE_GROUPS[gene],
        }
        patients.append(p)

    return {
        "gene":                    gene,
        "seed":                    seed,
        "n":                       N_PER_GENE,
        "disease":                 meta["disease"],
        "pathognomonic":           GENE_PATHOGNOMONIC[gene],
        "ucd_group":               GENE_GROUPS[gene],
        "omim_gene":               meta["omim_gene"],
        "omim_disease":            meta["omim_disease"],
        "aa":                      meta["aa"],
        "kda":                     meta["kda"],
        "locus":                   meta["locus"],
        "inh":                     meta["inh"],
        "orotic_acid_absent":      orotic_absent,
        "orotic_acid_elevated":    orotic_elevated,
        "nbs_detectable":          nbs_detectable,
        "nbs_detected_pct":        0.85 if nbs_detectable else 0.0,
        "neonatal_crisis":         neonatal_crisis,
        "carglumate_curative":     carglumate_curative,
        "liver_tx_fully_curative": liver_tx_fully_curative,
        "has_hair_fragility":      has_hair_fragility,
        "has_hypertension":        has_htn,
        "has_spastic_diplegia":    has_spastic,
        "has_homocitrullinuria":   has_homocitrullinuria,
        "has_ctln2":               has_ctln2,
        "is_xlr":                  is_xlr,
        "is_east_asian_founder":   is_east_asian_founder,
        "is_canadian_founder":     is_canadian_founder,
        "mean_nh3":                round(sum(nh3_vals) / len(nh3_vals), 1),
        "pct_severe":              round(sum(1 for o in outcomes if o == "severe") / len(outcomes), 3),
        "patients":                patients,
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]

    gene_table = []
    for c in cohorts:
        gene_table.append({
            "gene":             c["gene"],
            "disease":          c["disease"],
            "locus":            c["locus"],
            "aa":               c["aa"],
            "kda":              c["kda"],
            "inheritance":      c["inh"],
            "ucd_group":        c["ucd_group"],
            "orotic_absent":    c["orotic_acid_absent"],
            "orotic_elevated":  c["orotic_acid_elevated"],
            "nbs_detected":     c["nbs_detectable"],
            "neonatal_crisis":  c["neonatal_crisis"],
            "carglumate_curative": c["carglumate_curative"],
            "mean_nh3":         c["mean_nh3"],
            "pct_severe":       c["pct_severe"],
            "pathognomonic":    c["pathognomonic"],
            "omim_disease":     c["omim_disease"],
        })

    return {
        "atlas_id":          ATLAS_ID,
        "total_patients":    N_PER_GENE * len(GENES),
        "genes":             GENES,
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "nbs_detectable_genes": ["ASS1", "ASL", "ARG1"],
        "orotic_absent_genes":  ["CPS1", "NAGS"],
        "orotic_elevated_genes": ["OTC"],
        "genes_requiring_arginine_supp": ["CPS1", "OTC", "ASS1", "ASL"],
        "genes_restricting_arginine":    ["ARG1"],
        "xlinked_genes":     ["OTC"],
        "proximal_ucd":      PROXIMAL_GROUP,
        "distal_ucd":        DISTAL_GROUP,
        "activator_ucd":     ACTIVATOR_GROUP,
        "transport_ucd":     TRANSPORT_GROUP,
        "primary_cascade_split": "STEP 1: Orotic acid ABSENT (CPS1/NAGS) vs ELEVATED (OTC) vs normal (ASS1+). STEP 2: If absent -> trial carglumic acid (NCG) -> if NH3 normalizes = NAGS. STEP 3: Citrulline >1000 without argininosuccinate -> ASS1. STEP 4: Citrulline + argininosuccinate both elevated -> ASL. STEP 5: Arginine HIGH + spastic diplegia -> ARG1. STEP 6: Ornithine HIGH + homocitrulline in urine -> SLC25A15 (HHH). STEP 7: East Asian NBS citrulline+galactose -> SLC25A13 (citrin NICCD).",
        "gene_table":        gene_table,
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]

    return {
        "atlas_id":    ATLAS_ID,
        "seeds":       f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "ucd_groups": {
            "proximal_ucd": {
                "genes": PROXIMAL_GROUP,
                "description": "CPS1 + OTC: first two mitochondrial steps; both cause severe neonatal hyperammonemia; orotic acid distinguishes (absent CPS1, elevated OTC)",
                "orotic_rule": "CPS1 orotic absent; OTC orotic MARKEDLY elevated",
            },
            "distal_ucd": {
                "genes": DISTAL_GROUP,
                "description": "ASS1 + ASL + ARG1: cytoplasmic steps; citrulline (ASS1/ASL) or arginine (ARG1) accumulate; all three NBS-detectable",
                "nbs_detectable": True,
            },
            "activator_ucd": {
                "genes": ACTIVATOR_GROUP,
                "description": "NAGS: NAG synthesis defect; CPS1 activator absent; phenocopy of CPS1; carglumic acid (NCG) COMPLETELY CURATIVE",
                "carglumate_critical": True,
            },
            "transport_ucd": {
                "genes": TRANSPORT_GROUP,
                "description": "SLC25A15 (ORNT1/HHH) + SLC25A13 (citrin): mitochondrial transporter defects; distinctive ethnic founders; behavioral/neurological phenotypes",
                "ethnic_founders": ["French-Canadian (SLC25A15 p.F188del)", "East-Asian (SLC25A13 851del4/IVS16ins3kb)"],
            },
        },
        "orotic_acid_algorithm": {
            "absent_low": {
                "genes": ["CPS1", "NAGS"],
                "action": "IMMEDIATELY trial carglumic acid (NCG) 100 mg/kg -- if NH3 normalizes = NAGS (CURATIVE); if no response = CPS1 -> liver transplant workup",
                "urgency": "LIFE-SAVING -- do NOT delay NCG trial pending molecular results",
            },
            "markedly_elevated": {
                "genes": ["OTC"],
                "action": "OTC deficiency; check X-linked inheritance; allopurinol test for female carriers; benzoate+phenylbutyrate+arginine",
            },
            "mild_normal": {
                "genes": ["ASS1", "ASL", "ARG1", "SLC25A15", "SLC25A13"],
                "action": "Check plasma amino acids: citrulline (ASS1/ASL), arginine (ARG1), ornithine+homocitrulline (SLC25A15), ethnicity+NBS (SLC25A13)",
            },
        },
        "distinguishing_features": [
            {"gene": "CPS1",     "feature": "Orotic acid ABSENT + severe NH3 + neonatal crisis", "action": "NCG trial first; if no response -> liver transplant"},
            {"gene": "OTC",      "feature": "Orotic acid MARKEDLY ELEVATED + XLR", "action": "Benzoate/phenylbutyrate + arginine; female carrier testing allopurinol"},
            {"gene": "ASS1",     "feature": "Citrulline >1000 umol/L ONLY (no argininosuccinate)", "action": "Arginine supplement; NBS detected; liver transplant curative"},
            {"gene": "ASL",      "feature": "Citrulline + argininosuccinate BOTH elevated; hair fragility; HTN", "action": "Liver transplant corrects NH3 NOT hair/HTN; arginine supplement"},
            {"gene": "ARG1",     "feature": "Spastic diplegia + arginine HIGH + mild/absent NH3", "action": "RESTRICT arginine (opposite UCDs); no scavengers typically"},
            {"gene": "NAGS",     "feature": "CPS1 phenocopy + orotic acid absent; NCG CURATIVE", "action": "NCG empirically; if response = NAGS confirmed; lifelong NCG"},
            {"gene": "SLC25A15", "feature": "Homocitrulline in urine PATHOGNOMONIC; ornithine HIGH; cerebellar ataxia", "action": "Ornithine restriction; citrulline supplement"},
            {"gene": "SLC25A13", "feature": "East Asian + CTLN2 food cravings PATHOGNOMONIC; NICCD pseudo-galactosemia NBS", "action": "CTLN2: urgent liver transplant; NEVER IV glycerol/glucose; NICCD: MCT diet"},
        ],
        "treatment_principles": {
            "emergency": "Hemodialysis (fastest NH3 removal) + sodium benzoate/phenylacetate IV simultaneously",
            "maintenance": "Low protein diet + essential AA supplement + ammonia scavengers + arginine/citrulline",
            "gene_specific": {
                "NAGS":     "Carglumic acid (NCG) lifelong; COMPLETELY curative; no scavengers needed on NCG",
                "ARG1":     "Arginine RESTRICTION (not supplementation); citrulline to drive cycle without arginine accumulation",
                "ASL":      "Arginine supplement; post-transplant HTN and hair fragility persist (extrahepatic defect)",
                "SLC25A13": "CTLN2: liver transplant URGENTLY; NEVER IV glycerol/hyperosmolar glucose -> fatal crisis",
                "SLC25A15": "Citrulline supplement provides mitochondrial substrate bypass; ornithine restriction",
            },
            "transplant_curative": ["CPS1", "OTC", "ASS1", "NAGS", "SLC25A13"],
            "transplant_partial":  ["ASL", "ARG1"],
        },
        "cohort_summary": [
            {"gene": c["gene"], "mean_nh3": c["mean_nh3"], "pct_severe": c["pct_severe"],
             "neonatal_crisis": c["neonatal_crisis"], "carglumate_curative": c["carglumate_curative"]}
            for c in cohorts
        ],
    }


def generate_definitions():
    key_distinguishers = [
        {
            "name":        "Orotic acid: the critical first branch in UCD diagnosis",
            "description": "Urine orotic acid separates UCDs into three groups: (1) ABSENT/LOW -> CPS1 or NAGS (block is proximal to cytoplasmic carbamoyl phosphate; no CP reaches pyrimidine synthesis); (2) MARKEDLY ELEVATED -> OTC (carbamoyl phosphate overflows to cytoplasm -> pyrimidine synthesis -> orotate); (3) NORMAL/MILD -> ASS1, ASL, ARG1, SLC25A15, SLC25A13. ACTION: absent orotic acid + severe hyperammonemia = ALWAYS trial carglumic acid (NCG) -- if NH3 normalizes = NAGS (completely curative); if no response = CPS1 -> liver transplant",
        },
        {
            "name":        "Carglumic acid (NCG) empirical trial: MANDATORY in CPS1-like presentation",
            "description": "In any neonate/patient with severe hyperammonemia AND orotic acid absent/low: ALWAYS give carglumic acid (N-carbamylglutamate, NCG) empirically while awaiting molecular diagnosis. NCG dose: 100-250 mg/kg/day. If NH3 normalizes completely -> NAGS deficiency confirmed (curative). If no response -> CPS1 deficiency (proceed to liver transplant workup). NCG is SAFE with minimal adverse effects. Rarity of NAGS (~50 worldwide cases) makes empirical trial critical -- it is easily missed; a brief NCG trial can identify and cure a completely treatable condition. DO NOT delay NCG pending molecular results.",
        },
        {
            "name":        "ASL unique systemic features: hair fragility and hypertension persist post-transplant",
            "description": "Argininosuccinic aciduria (ASL deficiency) has two pathognomonic features beyond hyperammonemia: (1) TRICHORRHEXIS NODOSA (hair fragility): hair shafts break easily on microscopy -- pathognomonic for ASL among all UCDs; caused by arginine deficiency in hair follicle (arginine needed for creatine in hair matrix); present in longer-surviving treated patients; (2) HYPERTENSION: common even with excellent metabolic control; ASL also functions in the arginine/citrulline cycle for nitric oxide (NO) synthesis in endothelium; ASL-null -> NO deficiency -> systemic hypertension -> cardiovascular risk. CRITICAL: liver transplant corrects hepatic ASL (hyperammonemia resolves) but DOES NOT correct extrahepatic ASL deficiency -> hair fragility and hypertension PERSIST post-transplant -- inform families before listing for transplant",
        },
        {
            "name":        "Arginemia: progressive spasticity NOT hyperammonemia -- cerebral palsy misdiagnosis",
            "description": "ARG1 deficiency (arginemia) presents with PROGRESSIVE SPASTIC DIPLEGIA/PARAPARESIS -- NOT acute hyperammonemic crisis. Onset: age 2-4 years; scissor gait, loss of developmental milestones. Hyperammonemia is MILD or ABSENT (partial urea cycle function). Key diagnostic: PLASMA ARGININE MARKEDLY ELEVATED (>200-500 umol/L). Treatment: ARGININE RESTRICTION diet (opposite of every other UCD which requires arginine supplementation). No ammonia scavengers typically needed. Liver transplant corrects biochemistry but neurological damage may NOT fully reverse. Often misdiagnosed as cerebral palsy -- plasma amino acids (arginine) required in any unexplained progressive spastic diplegia",
        },
        {
            "name":        "HHH syndrome: homocitrulline in urine is pathognomonic",
            "description": "SLC25A15 (HHH syndrome) produces three simultaneous biochemical abnormalities: HYPERORNITHINEMIA + HYPERAMMONEMIA + HOMOCITRULLINURIA. The homocitrullinuria is pathognomonic -- it occurs because ornithine cannot enter mitochondria (ORNT1 defect) -> mitochondrial lysine accumulates -> lysine is carbamylated by excess carbamoyl phosphate -> homocitrulline excreted in urine. This is NOT seen in any other UCD. Clinical phenotype is dominated by NEUROLOGICAL DEGENERATION (cerebellar ataxia, pyramidal signs, cognitive decline) rather than acute crisis -- hyperammonemia is often episodic and mild/moderate. French Canadian founder mutation p.F188del enriched in Quebec/Maritime populations",
        },
        {
            "name":        "Citrin deficiency CTLN2: food cravings as a diagnostic clue in East Asian adults",
            "description": "Adult CTLN2 (SLC25A13 deficiency) has a pathognomonic behavioral feature: BIZARRE FOOD PREFERENCES -- intense craving for high-protein foods (meat, eggs, tofu, nuts) and FAT (peanut butter, fatty fish), with STRONG AVERSION to carbohydrates and sweets. This self-selecting behavior is driven by metabolic necessity (carbohydrates worsen aspartate-malate shuttle failure; fat/protein are preferred fuels). In any East Asian adult presenting with episodic encephalopathy and these food preferences -> citrin deficiency until proven otherwise. CRITICAL: ABSOLUTELY CONTRAINDICATED -- IV glycerol, high-glucose infusions, hyperosmolar solutions -> can trigger fatal hyperammonemic crisis. Liver transplant is URGENTLY CURATIVE for CTLN2",
        },
        {
            "name":        "OTC female carriers: variable severity and allopurinol test",
            "description": "OTC deficiency is X-linked; female heterozygotes are carriers but NOT simply 'unaffected'. Due to random X-inactivation (lyonization), ~15-20% of carrier females have clinically significant protein intolerance, episodic hyperammonemia (often triggered by illness, surgery, high protein), or even severe late-onset presentation. ALLOPURINOL LOADING TEST: used to detect asymptomatic carrier females; allopurinol inhibits orotidine-5'-phosphate decarboxylase; after allopurinol challenge, orotidine AND orotic acid both accumulate in carriers -- detectable in urine for 12-24h; sensitivity ~80-90%. Molecular genetic testing is gold standard for carrier detection",
        },
        {
            "name":        "NICCD vs ASS1: distinguishing citrullinemia causes in neonatal NBS",
            "description": "Both ASS1 deficiency (CTLN1) and citrin deficiency (NICCD, SLC25A13) can show elevated citrulline on NBS. KEY DISTINCTIONS: (1) ETHNICITY: NICCD almost exclusively East Asian; ASS1 pan-ethnic; (2) ADDITIONAL NBS markers: NICCD also shows GALACTOSE + METHIONINE + THREONINE elevated (pseudo-galactosemia pattern from cholestasis-induced enzyme effects); ASS1 has pure citrulline elevation; (3) CITRULLINE LEVEL: CTLN1 (ASS1) typically >500-1000 umol/L; NICCD usually 100-300 umol/L; (4) CLINICAL: NICCD presents with cholestatic jaundice + liver dysfunction; CTLN1 presents with hyperammonemic crisis; (5) COURSE: NICCD resolves with MCT formula by age 1-2; CTLN1 requires aggressive UCD management. In East Asian neonate: citrulline elevated + galactose elevated -> citrin deficiency until proven otherwise",
        },
    ]

    treatment_matrix = [
        {
            "intervention":    "Sodium benzoate + sodium phenylacetate (Ammonul IV) / sodium phenylbutyrate (oral)",
            "indications":     "Acute hyperammonemic crisis in CPS1, OTC, ASS1, ASL, NAGS, SLC25A15; maintenance in most UCDs",
            "mechanism":       "Alternative nitrogen disposal: benzoate conjugates glycine -> hippurate (excreted); phenylacetate/phenylbutyrate conjugates glutamine -> phenylacetylglutamine (excreted); each removes 1 nitrogen per molecule",
            "dose":            "Acute IV: sodium benzoate 250 mg/kg + sodium phenylacetate 250 mg/kg over 90 min bolus, then same dose over 24h; oral maintenance: sodium phenylbutyrate 450-600 mg/kg/day",
            "contraindicated": "NOT for ARG1 (hyperammonemia mild, arginine restriction preferred); reduce dose in renal failure",
        },
        {
            "intervention":    "Carglumic acid (N-carbamylglutamate, NCG / Carbaglu)",
            "indications":     "NAGS deficiency (COMPLETELY CURATIVE); empirical trial in all CPS1-like presentations (orotic absent); acquired NAGS deficiency in organic acidemias (adjunct)",
            "mechanism":       "NCG is a stable structural analogue of N-acetylglutamate (NAG); binds to and activates CPS1 allosterically even without NAGS enzyme function; dose-dependently restores ureagenesis",
            "dose":            "Acute: 100-250 mg/kg/day orally or via NG; maintenance: 10-100 mg/kg/day; titrate to normal NH3",
            "contraindicated": "No absolute contraindications; give EMPIRICALLY in any CPS1-like presentation; safe",
        },
        {
            "intervention":    "Arginine supplementation",
            "indications":     "CPS1 (200-800 mg/kg/day), OTC (200-800 mg/kg/day), ASS1 (200-800 mg/kg/day), ASL (200-800 mg/kg/day); essential since these cannot synthesize arginine endogenously",
            "mechanism":       "Arginine becomes ESSENTIAL amino acid when urea cycle cannot synthesize it; deficiency causes hypoproteinemia, poor growth, impaired immune function",
            "dose":            "200-800 mg/kg/day oral; neonatal acute: 200-600 mg/kg/day IV",
            "contraindicated": "ARG1 -- ABSOLUTELY CONTRAINDICATED; will worsen arginine accumulation and neurological toxicity",
        },
        {
            "intervention":    "Liver transplantation",
            "indications":     "CPS1 (curative), OTC (curative), ASS1 (curative), NAGS (curative), SLC25A13/CTLN2 (urgent-curative); ASL (corrects NH3 but NOT hair/HTN); ARG1 (corrects biochemistry, partial neuro benefit)",
            "mechanism":       "Donor hepatocytes express normal urea cycle enzymes -> complete nitrogen disposal restored; some extrahepatic enzyme defects persist (ASL hair/HTN; ARG1 neuro)",
            "dose":            "Living-related or deceased donor; consider early transplant before severe neurological injury; OTC: can transplant neonates at specialized centers",
            "contraindicated": "Relative: severe irreversible neurological damage; SLC25A15 (rarely transplanted -- mild hyperammonemia responds to dietary management)",
        },
        {
            "intervention":    "Hemodialysis (acute severe hyperammonemia)",
            "indications":     "NH3 >500 umol/L AND worsening despite scavengers; neonatal NH3 >1000 umol/L; loss of consciousness/coma",
            "mechanism":       "Direct extracorporeal NH3 removal; 5-10x more efficient than peritoneal dialysis; continuous veno-venous hemofiltration (CVVH) preferred in neonates",
            "dose":            "CVVH or intermittent HD; target NH3 <200 umol/L within 8-12h; do NOT stop scavengers during dialysis",
            "contraindicated": "No absolute contraindications; use in any patient with NH3 >500 and clinical deterioration",
        },
    ]

    return {
        "atlas_id":          ATLAS_ID,
        "key_distinguishers":key_distinguishers,
        "treatment_matrix":  treatment_matrix,
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "primary_cascade_split": "STEP 1: Orotic acid ABSENT -> CPS1 or NAGS (trial NCG immediately). STEP 2: Orotic acid MARKEDLY ELEVATED -> OTC (X-linked). STEP 3: Citrulline >1000 umol/L without argininosuccinate -> ASS1 (NBS-detected). STEP 4: Citrulline + argininosuccinate BOTH elevated -> ASL (check hair, BP). STEP 5: Arginine HIGH + spastic diplegia (not encephalopathy) -> ARG1 (RESTRICT arginine). STEP 6: Ornithine HIGH + homocitrulline in urine -> SLC25A15 (HHH syndrome). STEP 7: East Asian neonate + citrulline+galactose+Met NBS OR East Asian adult + food cravings + encephalopathy -> SLC25A13 (citrin; CTLN2 urgent liver Tx)",
        "ucd_groups": {
            "proximal":  {"genes": PROXIMAL_GROUP,  "orotic": "ABSENT (CPS1) or ELEVATED (OTC)"},
            "distal":    {"genes": DISTAL_GROUP,     "orotic": "normal/mild; citrulline or arginine elevated"},
            "activator": {"genes": ACTIVATOR_GROUP,  "orotic": "ABSENT (phenocopy CPS1); NCG CURATIVE"},
            "transport": {"genes": TRANSPORT_GROUP,  "orotic": "variable; distinctive ethnicity/biomarkers"},
        },
    }
