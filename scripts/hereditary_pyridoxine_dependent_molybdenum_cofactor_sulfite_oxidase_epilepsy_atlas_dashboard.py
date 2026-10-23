#!/usr/bin/env python3
"""Hereditary-Pyridoxine-Dependent-Molybdenum-Cofactor-Sulfite-Oxidase-Epilepsy-Atlas -- Complete 8-Gene Reference
B6-Responsive Epilepsies + Molybdenum Cofactor Deficiencies + Isolated Sulfite Oxidase Deficiency:
ALDH7A1-PNPO-PLPBP-ALDH4A1-MOCS1-MOCS2-GPHN-SUOX
(ALDH7A1/Antiquitin-539aa-5q31.2-AR-PDE1-MostCommon-B6-Responsive-Epilepsy-AASA+PipecolicAcid-PATHOGNOMONIC-Pyridoxine100mgIV-TRIAL-MANDATORY-LysineRestriction,
 PNPO/PNPO-Deficiency-261aa-17q21.32-AR-NeonatalEpilepticEncephalopathy-PLP-ONLY-NOT-Pyridoxine-KEY-DISTINCTION-CSFPrimido,
 PLPBP/PLPHP-Deficiency-321aa-8p11.23-AR-PLP-Homeostasis-BothPyridoxineAndPLP-Work-DistinctiveBiochemistry,
 ALDH4A1/P5CDH-Hyperprolinemia-II-563aa-1p36.13-AR-ProlineCatabolism-P5C-Accumulates-B6-Catabolism-Seizures,
 MOCS1/MoCoD-TypeA-481aa-6p21.2-AR-cPMP-Synthesis-SulfiteURINE-POSITIVE-UricAcidLOW-PATHOGNOMONIC-cPMP-Rescue,
 MOCS2/MoCoD-TypeB-362aa-5q11.2-AR-MPT-Synthase-SulfitePositive-UricAcidLow-SamePhenotype-MOCS1,
 GPHN/MoCoD-TypeC-736aa-14q23.3-AR-Gephyrin-MoCoD-Scaffold-ALSO-InhibitorySynapse-GABA-Glycine-SulfitePositive,
 SUOX/IsolatedSulfiteOxidase-545aa-12q13.11-AR-SulfitePositive-UricAcidNORMAL-KeyDDx-MoCoD-NoERT):

ALDH7A1 (Alpha-aminoadipic semialdehyde dehydrogenase / antiquitin; 539aa; 5q31.2; AR-LOF;
     ALDH7A1 encodes antiquitin (alpha-AASA dehydrogenase), the enzyme oxidizing alpha-aminoadipic semialdehyde (AASA)
     to alpha-aminoadipic acid in the PIPECOLIC ACID pathway and in saccharopine pathway of lysine catabolism;
     ALDH7A1 DEFICIENCY -- PYRIDOXINE-DEPENDENT EPILEPSY type 1 (PDE1) -- OMIM 266100;
     MOST COMMON B6-RESPONSIVE EPILEPSY: accounts for the majority of pyridoxine-dependent epilepsy cases worldwide;
     MECHANISM: ALDH7A1 LOF -> AASA and pipecolic acid accumulate -> AASA spontaneously cyclizes to piperideine-6-carboxylate (P6C);
     P6C reacts with pyridoxal-5'-phosphate (PLP) via Knoevenagel condensation -> inactivates PLP;
     PLP inactivation reduces GABA synthesis (glutamate decarboxylase requires PLP) -> seizures;
     TRIPLE PATHOGNOMONIC MARKERS:
       (1) alpha-AASA: elevated in urine, plasma, CSF PATHOGNOMONIC -- even on pyridoxine treatment;
       (2) pipecolic acid: elevated in plasma and CSF PATHOGNOMONIC -- even on treatment;
       (3) P6C: can be detected in urine acutely;
     THESE MARKERS REMAIN ELEVATED ON PYRIDOXINE TREATMENT -- confirms diagnosis even after seizures controlled;
     CLINICAL PRESENTATION:
       CLASSICAL FORM: neonatal onset seizures, prenatal seizures (in utero convulsions noted by mother);
       seizures of any type (myoclonic, tonic-clonic, focal, spasms); may mimic HIE;
       seizure-free on pyridoxine but biochemical markers persist;
       LATE-ONSET FORM: onset at 1-2 years or even later; partial response to AEDs may delay diagnosis;
     PYRIDOXINE TRIAL: 100 mg pyridoxine IV -- seizures typically stop within minutes (diagnostic AND therapeutic);
       should be done in ICU setting (risk of apnea);
     TREATMENT:
       Pyridoxine: 15-30 mg/kg/day lifelong; seizures fully controlled in most;
       Lysine restriction: reduces substrate flux through ALDH7A1 pathway; improves neurodevelopmental outcomes;
       Arginine supplementation: competes with lysine at BBB transport; augments lysine restriction benefit;
       Folinic acid (leucovorin): beneficial in some cases (PLP depletion affects folate metabolism);
     NEURODEVELOPMENTAL OUTCOME: seizures excellent controlled; BUT developmental delay + autism common even with treatment;
     lysine restriction improves cognitive outcomes significantly;
     seed SEED_BASE+0).

PNPO (Pyridox(am)ine 5'-phosphate oxidase; 261aa; 17q21.32; AR-LOF;
     PNPO encodes the enzyme catalyzing the final step in PLP biosynthesis from dietary pyridoxamine/pyridoxine:
     pyridoxamine-5'-phosphate (PMP) + pyridoxine-5'-phosphate (PNP) -> pyridoxal-5'-phosphate (PLP);
     PLP is the biologically active form of vitamin B6; pyridoxine must be CONVERTED to PLP by PNPO;
     PNPO DEFICIENCY -- OMIM 610090;
     CRITICAL DISTINCTION FROM ALDH7A1: RESPONDS TO PLP ONLY, NOT TO PYRIDOXINE;
       Pyridoxine trial: seizures do NOT stop (PNPO cannot convert pyridoxine to PLP);
       PLP trial: seizures stop;
       this distinction is PATHOGNOMONIC and clinically critical -- pyridoxine is the WRONG vitamin;
     NEONATAL EPILEPTIC ENCEPHALOPATHY: severe neonatal presentation; burst suppression EEG; may mimic ALDH7A1;
       many cases previously misdiagnosed as HIE or genetic epilepsy unresponsive to treatment;
     CSF DIAGNOSTIC PATTERN:
       CSF PLP: LOW;
       CSF pyridoxamine: ELEVATED (substrate accumulates);
       CSF pyridoxine-5'-phosphate: ELEVATED;
       CSF amino acids: may show low GABA; low glutamate/aspartate (PLP-dependent transamination affected);
       THREONINE elevated in CSF (PLP-dependent threonine aldolase affected);
       GLYCINE elevated in CSF (differential from NKH: glycine/serine ratio, PNPO does not have true glycine cleavage defect);
     CSF PRIMIDONE metabolites may show distinctive pattern if primidone given;
     TREATMENT:
       PLP (pyridoxal-5'-phosphate): 30-50 mg/kg/day; hepatotoxicity risk -- monitor LFTs closely;
       pyridoxine is INSUFFICIENT and delays diagnosis;
       dietary riboflavin (B2) supplementation may be adjunct (flavin cofactor for PNPO);
     PROGNOSIS: severe; many have persistent developmental delay even with PLP treatment;
       earlier diagnosis and treatment improves outcomes;
     seed SEED_BASE+1).

PLPBP (PLP homeostasis protein / PLPHP; 321aa; 8p11.23; AR-LOF;
     PLPBP (formerly PROSC) encodes PLP homeostasis protein (PLPHP), a highly conserved protein that binds PLP
     and is thought to act as a PLP buffer/carrier preventing accumulation of free PLP (which is toxic) while
     ensuring PLP availability for apoenzymes;
     PLPBP/PLPHP DEFICIENCY -- OMIM 617290;
     RESPONDS TO BOTH PYRIDOXINE AND PLP: unlike PNPO, both forms of B6 are effective;
       this distinguishes PLPBP from PNPO;
       mechanism: PLPHP normally sequesters/transfers PLP; LOF -> free PLP accumulates and may inhibit enzymes paradoxically;
       supplemental B6 in any form overcomes the deficiency;
     CLINICAL PRESENTATION:
       Neonatal to early infantile onset seizures;
       pyridoxine-responsive (partial or complete);
       EEG: burst suppression or hypsarrhythmia;
       may be milder than PNPO;
     BIOCHEMICAL PATTERN:
       No specific pathognomonic metabolite (unlike ALDH7A1 with AASA);
       CSF and plasma B6 vitamers abnormal: low PLP relative to other vitamers;
       CSF amino acids: may show low glycine/proline ratios;
       AASA NORMAL (distinguishes from ALDH7A1);
       sulfite NEGATIVE (distinguishes from MoCoD/SUOX);
     DIAGNOSIS: sequencing PLPBP; biochemical B6 vitamer profiling (specialized labs);
     TREATMENT: pyridoxine (15-30 mg/kg/day) or PLP (30-50 mg/kg/day);
       empirical B6 trial warranted in any neonatal epileptic encephalopathy;
     seed SEED_BASE+2).

ALDH4A1 (Delta-1-pyrroline-5-carboxylate dehydrogenase / P5CDH; 563aa; 1p36.13; AR-LOF;
     ALDH4A1 encodes delta-1-pyrroline-5-carboxylate (P5C) dehydrogenase, which converts P5C to glutamate
     in the proline catabolism pathway: proline -> P5C (via PRODH) -> glutamate (via P5CDH/ALDH4A1);
     ALDH4A1 DEFICIENCY -- HYPERPROLINEMIA TYPE II (HPII) -- OMIM 239510;
     MECHANISM: P5C accumulates -> P5C inactivates PLP (same Knoevenagel mechanism as P6C in ALDH7A1);
       PLP inactivation -> GABA synthesis reduced -> seizures;
       thus hyperprolinemia II is essentially a SECONDARY PLP-depletion epilepsy;
     PATHOGNOMONIC MARKERS:
       Plasma proline: markedly ELEVATED (hyperprolinemia; typically >1000 µmol/L; normal <275);
       Urinary P5C: elevated;
       these persist even when seizures controlled;
     CLINICAL FEATURES:
       Seizures (not always present; HPII has incomplete penetrance for neurological symptoms);
       intellectual disability (variable);
       MRI typically normal;
       may be incidentally found on amino acid analysis;
     KEY DISTINCTION: not usually neonatal onset; less severe than ALDH7A1/PNPO;
       some patients only have biochemical phenotype without seizures;
     DIAGNOSTIC CLUE: hyperprolinemia type II shows ELEVATED urinary P5C;
       hyperprolinemia type I (PRODH deficiency, associated with schizophrenia, 22q11 deletion) does NOT have P5C elevated;
       the P5C elevation distinguishes HPII from HPI;
     TREATMENT: pyridoxine supplementation (overcomes PLP inactivation by P5C);
       proline-restricted diet may reduce substrate flux;
     PROLINE CATABOLIC PATHWAY LINK: B6/PLP depletion by P5C (same Knoevenagel mechanism as AASA in PDE1);
     seed SEED_BASE+3).

MOCS1 (Molybdenum cofactor synthesis protein 1; 481aa; 6p21.2; AR-LOF;
     MOCS1 encodes two protein products (MOCS1A and MOCS1B via alternative splicing): both involved in
     synthesis of cyclic pyranopterin monophosphate (cPMP), the first intermediate in molybdenum cofactor (MoCo) biosynthesis;
     MOCS1 DEFICIENCY -- MOLYBDENUM COFACTOR DEFICIENCY TYPE A (MoCoD-A) -- OMIM 252150;
     MOLYBDENUM COFACTOR (MoCo) is required by 4 human enzymes:
       (1) SULFITE OXIDASE (SUOX): sulfite -> sulfate (detoxification);
       (2) XANTHINE DEHYDROGENASE (XDH): hypoxanthine/xanthine -> uric acid;
       (3) ALDEHYDE OXIDASE (AOX1): various aldehydes;
       (4) MITOCHONDRIAL AMIDOXIME REDUCING COMPONENT (MARC1/MARC2);
     MoCoD results in COMBINED DEFICIENCY of all 4 enzymes;
     PATHOGNOMONIC BIOCHEMICAL SIGNATURE:
       SULFITE POSITIVE in urine (sulfite stick test / sulfite oxidase activity undetectable);
       URIC ACID LOW in plasma (xanthine dehydrogenase inactive -> xanthine accumulates; uric acid NOT made);
       XANTHINE ELEVATED in urine/plasma;
       HYPOXANTHINE ELEVATED;
       the combination of SULFITE POSITIVE + URIC ACID LOW is PATHOGNOMONIC for MoCoD (vs isolated SUOX where uric acid NORMAL);
     CLINICAL PRESENTATION (SEVERE NEONATAL):
       Neonatal seizures, often in first hours of life; refractory;
       encephalopathy; lens dislocation (ectopia lentis) PATHOGNOMONIC for MoCoD;
       MRI: diffuse cortical edema -> progressive atrophy;
       most die in infancy or early childhood without treatment;
     SPECIFIC BIOMARKER: cPMP NOT produced (MOCS1 block at first step);
     TREATMENT:
       cPMP REPLACEMENT THERAPY: intravenous cPMP (fosdenopterin) is FDA-approved for MoCoD type A (2021);
       works for MOCS1 but NOT for MOCS2/GPHN (different pathway steps);
       MUST START WITHIN DAYS OF BIRTH for neurological benefit;
       urinary sulfite monitoring during treatment;
     seed SEED_BASE+4).

MOCS2 (Molybdenum cofactor synthesis protein 2; 362aa; 5q11.2; AR-LOF;
     MOCS2 gene encodes two overlapping proteins (MOCS2A / MOCS2B = MPT synthase small + large subunits)
     via alternative start codons; MPT synthase converts cPMP -> molybdopterin (MPT), the second step in MoCo biosynthesis;
     MOCS2 DEFICIENCY -- MOLYBDENUM COFACTOR DEFICIENCY TYPE B (MoCoD-B) -- OMIM 252160;
     SAME OVERALL PHENOTYPE AS MOCS1 (MoCoD type A):
       sulfite POSITIVE in urine;
       uric acid LOW;
       xanthine/hypoxanthine ELEVATED;
       neonatal seizures; lens dislocation; brain atrophy;
     KEY DIFFERENCE FROM MOCS1:
       cPMP accumulates in blood (MPT synthase blocked -> substrate builds up);
       cPMP replacement therapy DOES NOT WORK for type B (cPMP cannot be processed without MPT synthase);
       no FDA-approved specific therapy as of 2024; gene therapy investigational;
     DIAGNOSIS: requires MoCo subtype differentiation:
       plasma cPMP: ELEVATED in MOCS2 (unlike MOCS1 where cPMP is absent); this distinguishes type A vs B;
       MOCS2 sequencing confirms;
     MOCS2 GENE STRUCTURE: two overlapping ORFs (MOCS2A 88aa at N-terminus; MOCS2B 301aa full gene);
       both needed for functional MPT synthase complex; pathogenic variants in either reading frame cause disease;
     seed SEED_BASE+5).

GPHN (Gephyrin; 736aa; 14q23.3; AR-LOF;
     GPHN encodes gephyrin, a dual-function protein:
     (1) MOLYBDENUM COFACTOR SYNTHESIS: gephyrin is the final enzyme in MoCo biosynthesis:
       MPT + molybdate -> molybdenum cofactor (MoCo via molybdate insertion);
       GPHN LOF -> MoCoD type C (no MoCo assembled);
     (2) INHIBITORY SYNAPSE SCAFFOLD: gephyrin is the MASTER SCAFFOLD of inhibitory (GABA-A and glycine) synapses;
       gephyrin anchors GABA-A receptors and glycine receptors at the postsynaptic density;
       gephyrin forms trimers -> hexagonal lattice on postsynaptic membrane;
       gephyrin connects receptors to cytoskeleton (tubulin, actin);
     GPHN DEFICIENCY -- MoCoD TYPE C -- OMIM 615501 (RARE):
       Same MoCoD phenotype as MOCS1/MOCS2: sulfite positive, uric acid low, neonatal seizures, lens dislocation;
       HOWEVER: in some GPHN splice variants, ONLY the MoCo domain is affected (not the scaffolding domain);
       GPHN missense variants affecting ONLY synaptic scaffolding (not MoCo): milder epilepsy, autism, intellectual disability;
     KEY DISTINCTION: GPHN has a DUAL DISEASE mechanism:
       LOF full: MoCoD type C (severe neonatal; sulfite positive, uric acid low);
       PARTIAL LOF / scaffolding domain-only: epilepsy + autism WITHOUT MoCoD biochemistry;
     GEPHYRIN AND INHIBITORY SYNAPSE: reduced gephyrin -> reduced GABA-A/GlyR clustering -> reduced inhibition -> seizures;
       this is a second, INDEPENDENT disease mechanism;
     NOTE: fosdenopterin (cPMP) does NOT work for GPHN (gephyrin inserts Mo at a different step);
     seed SEED_BASE+6).

SUOX (Sulfite oxidase; 545aa; 12q13.11; AR-LOF;
     SUOX encodes sulfite oxidase, a mitochondrial enzyme converting sulfite (SO3 2-) to sulfate (SO4 2-);
     sulfite arises from catabolism of sulfur-containing amino acids (methionine, cysteine, taurine);
     SUOX requires BOTH molybdenum cofactor (MoCo) AND heme as cofactors;
     ISOLATED SULFITE OXIDASE DEFICIENCY (ISOD) -- OMIM 272300;
     KEY DIAGNOSTIC DISTINCTION FROM MoCoD:
       SULFITE POSITIVE in urine (same as MoCoD);
       URIC ACID NORMAL (xanthine dehydrogenase intact, because MoCo is PRESENT in ISOD);
       this URIC ACID NORMAL pattern distinguishes ISOD from all three MoCoD types;
     METABOLIC MARKER:
       S-sulfocysteine (SSC) in urine PATHOGNOMONIC: sulfite reacts non-enzymatically with cystine -> S-sulfocysteine;
       SSC is also formed in MoCoD but is pathognomonic for sulfite overload in general;
       SSC is the most sensitive marker for sulfite overload disorders;
     CLINICAL PRESENTATION (VERY SIMILAR TO MoCoD):
       Neonatal seizures (refractory); may mimic HIE or pyridoxine-dependent epilepsy;
       encephalopathy; brain atrophy; lens dislocation (ectopia lentis) -- less common than MoCoD;
       movement disorder; spasticity;
       progressive neurodegeneration;
     DIAGNOSIS:
       Urine sulfite stick test positive;
       plasma uric acid NORMAL (critical DDx from MoCoD);
       urine S-sulfocysteine ELEVATED;
       SUOX enzyme activity in skin fibroblasts (near absent);
       SUOX sequencing;
     TREATMENT: no approved therapy; supportive;
       low-sulfur amino acid diet (reduces substrate);
       taurine supplementation investigated;
       gene therapy preclinical;
     NOTE: unlike MoCoD type A, fosdenopterin DOES NOT HELP (MoCo is normal; isolated SUOX LOF);
     seed SEED_BASE+7)."""

import random, hashlib, json
from typing import List, Dict, Any

ATLAS_ID  = "hereditary-pyridoxine-dependent-molybdenum-cofactor-sulfite-oxidase-epilepsy-atlas"
SEED_BASE = 4238
GENES     = ["ALDH7A1", "PNPO", "PLPBP", "ALDH4A1", "MOCS1", "MOCS2", "GPHN", "SUOX"]

B6_RESPONSIVE_GROUP  = ["ALDH7A1", "PNPO", "PLPBP", "ALDH4A1"]
MOCOД_GROUP          = ["MOCS1", "MOCS2", "GPHN"]
SULFITE_OXIDASE_GROUP = ["SUOX"]

GENE_META = {
    "ALDH7A1": {"aa": 539,  "locus": "5q31.2",  "inh": "AR",   "size_kda": 58,  "mechanism": "antiquitin-PDE1-AASA-P6C-PLP-inactivation"},
    "PNPO":    {"aa": 261,  "locus": "17q21.32", "inh": "AR",   "size_kda": 30,  "mechanism": "PNP-PMP-to-PLP-conversion-PLP-ONLY-works"},
    "PLPBP":   {"aa": 321,  "locus": "8p11.23",  "inh": "AR",   "size_kda": 35,  "mechanism": "PLP-homeostasis-buffer-both-B6-forms-work"},
    "ALDH4A1": {"aa": 563,  "locus": "1p36.13",  "inh": "AR",   "size_kda": 62,  "mechanism": "P5CDH-hyperprolinemia-II-P5C-inactivates-PLP"},
    "MOCS1":   {"aa": 481,  "locus": "6p21.2",   "inh": "AR",   "size_kda": 54,  "mechanism": "cPMP-synthesis-MoCoD-typeA-fosdenopterin-WORKS"},
    "MOCS2":   {"aa": 362,  "locus": "5q11.2",   "inh": "AR",   "size_kda": 41,  "mechanism": "MPT-synthase-MoCoD-typeB-cPMP-elevated-fosdenopterin-FAILS"},
    "GPHN":    {"aa": 736,  "locus": "14q23.3",  "inh": "AR",   "size_kda": 83,  "mechanism": "MoCoD-typeC-ALSO-inhibitory-synapse-scaffold-DUAL"},
    "SUOX":    {"aa": 545,  "locus": "12q13.11", "inh": "AR",   "size_kda": 60,  "mechanism": "isolated-sulfite-oxidase-uric-acid-NORMAL-DDx-MoCoD"},
}

PATHOGNOMONIC = {
    "ALDH7A1": "AASA + pipecolic acid elevated (urine/plasma/CSF) — PATHOGNOMONIC even on pyridoxine treatment; pyridoxine 100 mg IV trial stops seizures",
    "PNPO":    "PLP ONLY stops seizures (NOT pyridoxine) — PATHOGNOMONIC distinction; CSF pyridoxamine/PNP elevated",
    "PLPBP":   "Both pyridoxine AND PLP work; AASA normal; sulfite negative; B6 vitamer profiling diagnostic",
    "ALDH4A1": "Plasma proline markedly elevated (hyperprolinemia II); urinary P5C elevated; incomplete neurological penetrance",
    "MOCS1":   "Sulfite POSITIVE + uric acid LOW + xanthine/hypoxanthine elevated — MoCoD; cPMP ABSENT; fosdenopterin type A ONLY",
    "MOCS2":   "Sulfite POSITIVE + uric acid LOW — same as MOCS1; cPMP ELEVATED (MPT synthase blocked); fosdenopterin DOES NOT WORK",
    "GPHN":    "Sulfite POSITIVE + uric acid LOW (MoCoD type C) OR epilepsy+autism without MoCoD biochemistry (scaffolding domain-only variants)",
    "SUOX":    "Sulfite POSITIVE + uric acid NORMAL — CRITICAL DDx from MoCoD; S-sulfocysteine PATHOGNOMONIC; no fosdenopterin indication",
}

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _make_cohort(gene: str, seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    n = 40
    gene_meta = GENE_META[gene]

    # Variant type distribution (realistic for each gene)
    if gene == "ALDH7A1":
        pathogenic_frac = rng.uniform(0.88, 0.96)
    elif gene in ("MOCS1", "MOCS2", "GPHN", "SUOX"):
        pathogenic_frac = rng.uniform(0.82, 0.94)
    else:
        pathogenic_frac = rng.uniform(0.80, 0.95)

    pathogenic = int(n * pathogenic_frac)
    vus = n - pathogenic

    # Response / outcome
    if gene == "ALDH7A1":
        pyridoxine_response = rng.uniform(0.88, 0.98)
        neurodevelopmental_normal = rng.uniform(0.18, 0.35)
    elif gene == "PNPO":
        pyridoxine_response = 0.0
        plp_response = rng.uniform(0.65, 0.85)
        neurodevelopmental_normal = rng.uniform(0.10, 0.25)
    elif gene == "PLPBP":
        pyridoxine_response = rng.uniform(0.70, 0.90)
        neurodevelopmental_normal = rng.uniform(0.25, 0.50)
    elif gene == "ALDH4A1":
        pyridoxine_response = rng.uniform(0.50, 0.75)
        neurodevelopmental_normal = rng.uniform(0.35, 0.65)
    else:  # MoCoD / SUOX
        pyridoxine_response = 0.0
        neurodevelopmental_normal = rng.uniform(0.03, 0.12)

    return {
        "gene":              gene,
        "n":                 n,
        "pathogenic_n":      pathogenic,
        "vus_n":             vus,
        "locus":             gene_meta["locus"],
        "inheritance":       gene_meta["inh"],
        "aa":                gene_meta["aa"],
        "size_kda":          gene_meta["size_kda"],
        "mechanism_tag":     gene_meta["mechanism"],
        "pathognomonic":     PATHOGNOMONIC[gene],
    }


def generate_overview() -> Dict[str, Any]:
    cohorts = []
    for i, gene in enumerate(GENES):
        cohorts.append(_make_cohort(gene, SEED_BASE + i))

    summary_table = []
    for c in cohorts:
        summary_table.append({
            "gene":          c["gene"],
            "locus":         c["locus"],
            "aa":            c["aa"],
            "kda":           c["size_kda"],
            "inheritance":   c["inheritance"],
            "mechanism":     c["mechanism_tag"],
            "pathognomonic": c["pathognomonic"],
            "n":             c["n"],
        })

    return {
        "atlas_id":     ATLAS_ID,
        "genes":        GENES,
        "total_n":      len(GENES) * 40,
        "seeds":        f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "groups": {
            "b6_responsive":   B6_RESPONSIVE_GROUP,
            "mocoд_types":     MOCOД_GROUP,
            "sulfite_oxidase": SULFITE_OXIDASE_GROUP,
        },
        "summary_table": summary_table,
        "primary_cascade_split": (
            "STEP 1: Urine sulfite stick test POSITIVE? "
            "  YES → plasma uric acid: LOW → MoCoD (MOCS1/MOCS2/GPHN); NORMAL → isolated SUOX. "
            "  For MoCoD subtypes: plasma cPMP ABSENT → MOCS1 (type A, fosdenopterin WORKS); "
            "  cPMP ELEVATED → MOCS2 (type B) or GPHN (type C, needs sequencing). "
            "STEP 2: Sulfite NEGATIVE → B6/PLP trial: "
            "  Pyridoxine 100mg IV → seizures stop → ALDH7A1 (check AASA + pipecolic acid); "
            "  PLP only → seizures stop but NOT pyridoxine → PNPO; "
            "  Both work → PLPBP; "
            "  Plasma proline markedly elevated → ALDH4A1 (hyperprolinemia II, P5C elevated). "
            "KEY RULE: sulfite stick test first; then uric acid to split MoCoD from SUOX; "
            "then cPMP to subtype MoCoD."
        ),
        "universal_drug_contraindications": [
            "ALDH7A1: No specific drug CIs; avoid prolonged fasting (increases lysine catabolism flux)",
            "PNPO: Pyridoxine alone INSUFFICIENT and delays correct treatment (must use PLP); monitor LFTs on PLP",
            "PLPBP: No absolute CIs; standard B6 monitoring",
            "ALDH4A1: No major drug CIs; low-proline diet adjunct",
            "MOCS1/MOCS2/GPHN/SUOX: No specific drug CIs; supportive care; fosdenopterin ONLY for MOCS1 (type A)",
        ],
        "cohort_summaries": cohorts,
    }


def generate_breakdown() -> Dict[str, Any]:
    gene_details = {}
    for i, gene in enumerate(GENES):
        rng = _rng(SEED_BASE + i)
        meta = GENE_META[gene]

        # Build variant distribution
        total = 40
        if gene == "ALDH7A1":
            missense = int(total * rng.uniform(0.45, 0.55))
            nonsense = int(total * rng.uniform(0.12, 0.18))
            frameshift = int(total * rng.uniform(0.12, 0.18))
            splice = total - missense - nonsense - frameshift
            neonatal_pct = rng.uniform(0.55, 0.70)
            late_onset_pct = 1.0 - neonatal_pct
        elif gene == "PNPO":
            missense = int(total * rng.uniform(0.55, 0.70))
            nonsense = int(total * rng.uniform(0.10, 0.15))
            frameshift = int(total * rng.uniform(0.08, 0.14))
            splice = total - missense - nonsense - frameshift
            neonatal_pct = rng.uniform(0.80, 0.95)
            late_onset_pct = 1.0 - neonatal_pct
        elif gene in ("MOCS1", "MOCS2", "GPHN", "SUOX"):
            missense = int(total * rng.uniform(0.35, 0.50))
            nonsense = int(total * rng.uniform(0.15, 0.25))
            frameshift = int(total * rng.uniform(0.15, 0.25))
            splice = total - missense - nonsense - frameshift
            neonatal_pct = rng.uniform(0.85, 0.98)
            late_onset_pct = 1.0 - neonatal_pct
        else:  # PLPBP, ALDH4A1
            missense = int(total * rng.uniform(0.50, 0.65))
            nonsense = int(total * rng.uniform(0.10, 0.15))
            frameshift = int(total * rng.uniform(0.10, 0.15))
            splice = total - missense - nonsense - frameshift
            neonatal_pct = rng.uniform(0.40, 0.65)
            late_onset_pct = 1.0 - neonatal_pct

        # Treatment response
        if gene == "ALDH7A1":
            seizure_free = rng.uniform(0.85, 0.97)
            neurodevelopment_normal = rng.uniform(0.15, 0.35)
            notes = [
                "AASA + pipecolic acid persist in urine/plasma even on treatment — use for monitoring and diagnosis",
                "Lysine restriction (combined with pyridoxine) significantly improves neurodevelopmental outcomes",
                "Late-onset PDE1 (first seizure at months to years) has better neurodevelopmental outcome than neonatal-onset",
                "Prenatal convulsions may be first sign (reported by mother as in utero jerking)",
                "Folinic acid/leucovorin may add additional benefit in selected cases",
            ]
        elif gene == "PNPO":
            seizure_free = rng.uniform(0.60, 0.80)
            neurodevelopment_normal = rng.uniform(0.08, 0.22)
            notes = [
                "PLP ONLY — pyridoxine WILL NOT WORK — this distinction is life-saving and PATHOGNOMONIC",
                "Monitor LFTs closely — PLP at doses needed for seizure control has hepatotoxicity risk",
                "CSF threonine elevated (PLP-dependent threonine aldolase impaired) — distinguishing feature",
                "Riboflavin (B2) as adjunct may improve residual PNPO activity in hypomorphic alleles",
                "Empirical PLP trial should be given in any refractory neonatal encephalopathy after pyridoxine failure",
            ]
        elif gene == "PLPBP":
            seizure_free = rng.uniform(0.70, 0.90)
            neurodevelopment_normal = rng.uniform(0.30, 0.55)
            notes = [
                "Both pyridoxine and PLP are effective — less strict than PNPO",
                "No specific pathognomonic metabolite — diagnosis requires gene sequencing + B6 vitamer profiling",
                "Generally milder than PNPO; better neurodevelopmental prognosis",
                "AASA negative distinguishes from ALDH7A1",
                "Previously called PROSC deficiency — now unified nomenclature PLPBP",
            ]
        elif gene == "ALDH4A1":
            seizure_free = rng.uniform(0.55, 0.80)
            neurodevelopment_normal = rng.uniform(0.35, 0.65)
            notes = [
                "Hyperprolinemia type II — markedly elevated plasma proline (typically >1000 µmol/L)",
                "Urinary P5C elevated — this distinguishes HPII (ALDH4A1) from HPI (PRODH, proline elevated but NO P5C)",
                "Neurological penetrance incomplete — some HPII patients are asymptomatic",
                "Pyridoxine supplementation overcomes PLP inactivation by P5C",
                "PRODH-associated hyperprolinemia type I: associated with schizophrenia/22q11 deletion — different gene, different phenotype",
            ]
        elif gene == "MOCS1":
            seizure_free = rng.uniform(0.30, 0.55)  # fosdenopterin partial
            neurodevelopment_normal = rng.uniform(0.02, 0.10)
            notes = [
                "cPMP ABSENT in plasma — this identifies MoCoD type A (MOCS1); distinguishes from type B (cPMP elevated)",
                "Fosdenopterin (cyclic pyranopterin monophosphate; FDA-approved 2021) is ONLY effective for type A (MOCS1)",
                "Must start within days of birth for neurological benefit — every day matters",
                "Lens dislocation (ectopia lentis) PATHOGNOMONIC for MoCoD; less common in isolated SUOX",
                "Urinary sulfite monitored during fosdenopterin treatment as efficacy biomarker",
            ]
        elif gene == "MOCS2":
            seizure_free = rng.uniform(0.08, 0.20)
            neurodevelopment_normal = rng.uniform(0.02, 0.08)
            notes = [
                "cPMP ELEVATED in plasma — MOCS1 cannot be treated; MOCS2 blocks MPT synthase downstream of cPMP",
                "Fosdenopterin (cPMP) DOES NOT WORK for type B — cPMP cannot be processed without MPT synthase",
                "Same biochemical phenotype as MOCS1: sulfite positive, uric acid low — requires sequencing for subtype",
                "MOCS2 gene contains two overlapping ORFs (MOCS2A + MOCS2B) — variants in either cause disease",
                "Gene therapy (AAV-based) is in preclinical/early clinical development for type B",
            ]
        elif gene == "GPHN":
            seizure_free = rng.uniform(0.10, 0.25)
            neurodevelopment_normal = rng.uniform(0.02, 0.08)
            notes = [
                "DUAL DISEASE GENE: (1) MoCoD type C (LOF full — sulfite positive, uric acid low, neonatal severe); (2) epilepsy+autism without MoCoD biochemistry (scaffolding domain variants)",
                "MoCoD type C: fosdenopterin DOES NOT WORK (gephyrin inserts Mo at a step downstream of cPMP + MPT)",
                "Scaffolding domain-only variants: sulfite NEGATIVE, uric acid NORMAL — milder phenotype, epilepsy+ASD",
                "Gephyrin is the master scaffold of inhibitory GABA-A and glycine receptor clusters at postsynaptic densities",
                "GPHN copy number variants (deletions/duplications) also cause autism spectrum disorder without MoCoD",
            ]
        else:  # SUOX
            seizure_free = rng.uniform(0.08, 0.22)
            neurodevelopment_normal = rng.uniform(0.02, 0.08)
            notes = [
                "Uric acid NORMAL — this is the critical DDx from all three MoCoD types (where uric acid is LOW)",
                "S-sulfocysteine in urine PATHOGNOMONIC for sulfite overload; elevated in both ISOD and MoCoD",
                "Sulfite stick test positive — first-line screening in any neonatal epileptic encephalopathy",
                "SUOX requires BOTH MoCo and heme as cofactors — ISOD: MoCo is normal; MoCo disorders: SUOX secondary deficiency",
                "No FDA-approved therapy; low-sulfur amino acid diet reduces substrate load",
            ]

        gene_details[gene] = {
            "gene":              gene,
            "locus":             meta["locus"],
            "aa":                meta["aa"],
            "kda":               meta["size_kda"],
            "inheritance":       meta["inh"],
            "n":                 total,
            "variant_distribution": {
                "missense":   missense,
                "nonsense":   nonsense,
                "frameshift": frameshift,
                "splice":     splice,
            },
            "onset_pct": {
                "neonatal":   round(neonatal_pct * 100, 1),
                "late_onset": round(late_onset_pct * 100, 1),
            },
            "outcomes": {
                "seizure_free_pct":           round(seizure_free * 100, 1),
                "neurodevelopment_normal_pct": round(neurodevelopment_normal * 100, 1),
            },
            "pathognomonic": PATHOGNOMONIC[gene],
            "key_notes":     notes,
            "seed":          SEED_BASE + list(GENES).index(gene),
        }

    return {
        "atlas_id":    ATLAS_ID,
        "seeds":       f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "gene_details": gene_details,
        "diagnostic_groups": {
            "b6_responsive":  {
                "genes":   B6_RESPONSIVE_GROUP,
                "markers": "Urine AASA (ALDH7A1); CSF pyridoxamine/PNP elevated (PNPO); B6 vitamer profiling (PLPBP); plasma proline/P5C elevated (ALDH4A1)",
                "trial":   "Pyridoxine 100mg IV OR PLP 30mg/kg/day — response distinguishes subtypes",
            },
            "mocoд_types": {
                "genes":   MOCOД_GROUP,
                "markers": "Sulfite POSITIVE + uric acid LOW + xanthine elevated; cPMP absent (MOCS1) or elevated (MOCS2/GPHN)",
                "trial":   "Fosdenopterin (cPMP) ONLY for MOCS1 type A; does NOT work for MOCS2/GPHN",
            },
            "isolated_sulfite_oxidase": {
                "genes":   SULFITE_OXIDASE_GROUP,
                "markers": "Sulfite POSITIVE + uric acid NORMAL — critical DDx from MoCoD; S-sulfocysteine elevated",
                "trial":   "Low-sulfur diet; no approved pharmacotherapy",
            },
        },
    }


def generate_definitions() -> Dict[str, Any]:
    key_distinguishers = [
        {
            "name":               "Urine sulfite stick test — first-line triage in neonatal epileptic encephalopathy",
            "definition":         (
                "The urine sulfite dipstick test (e.g., Merckoquant sulfite test strips) is the critical FIRST STEP in evaluating "
                "any neonatal epileptic encephalopathy where MoCoD or SUOX is in the differential. "
                "POSITIVE sulfite → pursue MoCoD/SUOX pathway. "
                "NEGATIVE sulfite → pursue B6-responsive epilepsy pathway. "
                "LIMITATIONS: sulfite is UNSTABLE in urine; the sample must be tested FRESH (within minutes); "
                "false negatives occur if urine is left standing; always test fresh urine at bedside; "
                "SENSITIVITY is not 100% — a negative test does not exclude MoCoD/SUOX if clinical suspicion remains; "
                "confirmatory testing (plasma uric acid, xanthine/hypoxanthine, S-sulfocysteine, cPMP) must follow a positive. "
                "COLLECTION: catheterize or collect suprapubic if no spontaneous void; test IMMEDIATELY."
            ),
            "clinical_relevance": "First test in any neonatal refractory seizures with encephalopathy — positive sulfite directs MoCoD vs SUOX workup immediately",
        },
        {
            "name":               "Uric acid — critical discriminator: MoCoD (LOW) vs isolated SUOX (NORMAL)",
            "definition":         (
                "Plasma uric acid is the critical second-level discriminator after a positive sulfite test. "
                "MoCoD (all types): uric acid VERY LOW or absent; xanthine dehydrogenase (XDH) is inactive (MoCo deficiency); "
                "XDH cannot convert xanthine → uric acid; thus hypoxanthine and xanthine accumulate; "
                "uric acid: typically <1-2 mg/dL (normal: 3-7 mg/dL in children); "
                "xanthine elevated in urine (xanthinuria pattern). "
                "ISOLATED SUOX: uric acid NORMAL because MoCo is present and XDH is functional; "
                "only sulfite oxidase is deficient; xanthine dehydrogenase intact. "
                "This distinction is CRITICAL: "
                "(1) Therapeutic: fosdenopterin only for MoCoD (not SUOX); "
                "(2) Diagnostic: confirms isolated SUOX vs combined molybdoenzyme failure. "
                "S-sulfocysteine elevated in BOTH MoCoD AND SUOX (sulfite overload common denominator)."
            ),
            "clinical_relevance": "Plasma uric acid is the single most important biochemical test to distinguish MoCoD (low) from isolated SUOX (normal) after positive sulfite",
        },
        {
            "name":               "cPMP (cyclic pyranopterin monophosphate) — MoCoD type A vs B subtyping",
            "definition":         (
                "Plasma cPMP measurement is essential for subtyping MoCoD and guiding treatment: "
                "MoCoD TYPE A (MOCS1 deficiency): cPMP ABSENT in plasma/urine; MOCS1 cannot make cPMP; "
                "the pathway is blocked at the very first step; NO cPMP to measure. "
                "MoCoD TYPE B (MOCS2 deficiency): cPMP ELEVATED; cPMP is made (MOCS1 intact) but cannot be converted to MPT (MPT synthase = MOCS2 is deficient); "
                "substrate accumulates = cPMP elevated. "
                "MoCoD TYPE C (GPHN deficiency): cPMP and MPT are made; gephyrin (molybdate insertase) is deficient; "
                "cPMP may be normal or variably elevated. "
                "THERAPEUTIC IMPLICATION: "
                "Fosdenopterin (recombinant cPMP; FDA-approved 2021) ONLY works for TYPE A: "
                "providing cPMP bypasses the MOCS1 block; subsequent enzymes (MOCS2, GPHN) are intact; "
                "fosdenopterin does NOT help type B (no MPT synthase to process cPMP) or type C (no gephyrin to insert Mo). "
                "cPMP testing available at specialized metabolic labs (Cologne, Boston, etc.)."
            ),
            "clinical_relevance": "cPMP absent = type A (MOCS1); fosdenopterin FDA-approved, START IMMEDIATELY. cPMP elevated = type B/C; fosdenopterin USELESS; supportive only",
        },
        {
            "name":               "Pyridoxine vs PLP trial — PNPO vs ALDH7A1 vs PLPBP distinction",
            "definition":         (
                "The pyridoxine/PLP trial is the KEY CLINICAL DISCRIMINATOR for B6-responsive epilepsies: "
                "ALDH7A1 (PDE1): pyridoxine 100 mg IV → seizures typically stop within minutes; "
                "this is DIAGNOSTIC; pyridoxine must be converted to PLP by PNPO (intact in ALDH7A1); "
                "AASA and pipecolic acid remain elevated regardless — confirm diagnosis biochemically. "
                "PNPO DEFICIENCY: pyridoxine DOES NOT WORK (PNPO cannot convert pyridoxine to PLP); "
                "seizures continue or worsen; delaying correct diagnosis (PLP) causes irreversible brain damage; "
                "PLP 30-50 mg/kg/day → seizures stop; CSF pyridoxamine/PNP elevated; threonine elevated in CSF. "
                "PLPBP DEFICIENCY: BOTH pyridoxine AND PLP work — no strict requirement for PLP; "
                "pyridoxine responsive; less severe than PNPO; AASA negative. "
                "ALDH4A1 (hyperprolinemia II): pyridoxine may partially help (overcomes PLP inactivation by P5C); "
                "diagnosis is by plasma proline + P5C, not pyridoxine trial. "
                "PROTOCOL: give pyridoxine 100 mg IV first (ALDH7A1 most common); if no response within 30 min, "
                "give PLP 30 mg/kg/day (PNPO); monitor for apnea during IV pyridoxine (ICU setting required)."
            ),
            "clinical_relevance": "Pyridoxine fails + PLP works = PNPO (life-saving distinction); AASA + pipecolic acid confirm ALDH7A1 even on treatment; both work = PLPBP",
        },
        {
            "name":               "AASA + pipecolic acid — permanent biomarkers for PDE1 (ALDH7A1)",
            "definition":         (
                "Alpha-aminoadipic semialdehyde (AASA) and pipecolic acid are the definitive biomarkers for "
                "pyridoxine-dependent epilepsy type 1 (ALDH7A1 deficiency / antiquitin deficiency). "
                "AASA: the direct substrate of antiquitin (ALDH7A1); accumulates when ALDH7A1 is deficient; "
                "AASA spontaneously cyclizes to piperideine-6-carboxylate (P6C); "
                "P6C reacts with PLP via Knoevenagel condensation → covalently inactivates PLP → GABA deficiency → seizures. "
                "PIPECOLIC ACID: an alternative metabolite in the lysine catabolism pathway; elevated in PDE1; "
                "elevated in plasma AND CSF AND urine. "
                "CRITICAL PROPERTY: BOTH MARKERS REMAIN ELEVATED DURING PYRIDOXINE TREATMENT; "
                "this allows: (1) diagnosis to be made in a child already on pyridoxine; "
                "(2) monitoring of lysine restriction efficacy (falling AASA/pipecolic acid = good control); "
                "(3) newborn screening potential (AASA detectable in dried blood spot). "
                "ALDH7A1 founder variants: c.1279G>C (European), E414K (Arab populations); "
                "NMD (nonsense-mediated decay)-sensitive truncating variants common. "
                "AASA measured by LC-MS/MS; specialized labs only; plasma preferable for monitoring."
            ),
            "clinical_relevance": "AASA + pipecolic acid confirm PDE1 even on pyridoxine; monitor lysine restriction by falling AASA; NBS-amenable biomarkers",
        },
        {
            "name":               "Fosdenopterin (NULIBRY) — cPMP replacement for MoCoD type A (MOCS1)",
            "definition":         (
                "Fosdenopterin (brand: NULIBRY; recombinant cPMP/cyclic pyranopterin monophosphate) is the first "
                "FDA-approved therapy for molybdenum cofactor deficiency type A (MOCS1). Approved 2021. "
                "MECHANISM: exogenous cPMP bypasses the MOCS1 block; enters cells; processed by intact MOCS2 + GPHN enzymes "
                "to regenerate MoCo; restores all four molybdoenzyme activities including sulfite oxidase. "
                "EFFICACY: reduces urinary sulfite; reduces plasma xanthine; reduces S-sulfocysteine; "
                "clinical trial showed survival benefit and improved neurological outcomes vs historical controls; "
                "earlier treatment → better neurological outcome; treatment within first days of life is critical. "
                "ADMINISTRATION: daily IV infusion; requires vascular access; home infusion possible with training. "
                "MONITORING: urinary sulfite (efficacy); xanthine (efficacy); LFTs; CBC. "
                "INDICATION LIMITATION: ONLY for MOCS1 (type A); USELESS for MOCS2 (type B), GPHN (type C), SUOX; "
                "requires prior cPMP measurement to confirm absent cPMP (type A). "
                "IMPORTANT: GPHN variants causing isolated inhibitory synapse dysfunction (no MoCoD) do NOT respond "
                "and should not receive fosdenopterin."
            ),
            "clinical_relevance": "FDA-approved 2021 for MOCS1-deficiency MoCoD; cPMP absent confirms eligibility; start within days of diagnosis for best neurological outcomes",
        },
        {
            "name":               "S-sulfocysteine (SSC) — pan-sulfite-overload marker for both MoCoD and isolated SUOX",
            "definition":         (
                "S-sulfocysteine (SSC) is formed by non-enzymatic reaction of sulfite with cystine; "
                "its presence in urine and/or plasma indicates sulfite overload. "
                "ELEVATED IN BOTH MoCoD AND ISOLATED SUOX: since both disorders result in sulfite accumulation, "
                "SSC is elevated in all MoCoD types (A, B, C) and in isolated SUOX deficiency. "
                "SSC is the MOST SENSITIVE AND STABLE MARKER for sulfite overload: "
                "unlike the sulfite stick test (which requires fresh urine and has false negatives), "
                "SSC can be measured by amino acid analysis (plasma or urine) on routine clinical samples; "
                "stable for storage; high sensitivity. "
                "REFERENCE RANGES: SSC normally undetectable (<1 µmol/mmol creatinine); "
                "affected patients: typically >50 µmol/mmol creatinine. "
                "MONITORING USE: SSC falls with fosdenopterin treatment in type A; "
                "can be used to guide dose and assess biochemical response. "
                "NOTE: SSC is also a renal toxin (sulfite overload causes renal injury); "
                "elevated SSC tracks neurological and renal damage risk."
            ),
            "clinical_relevance": "SSC is more reliable than sulfite stick for diagnosis and monitoring; elevated in all MoCoD subtypes and isolated SUOX; use for NBS and treatment monitoring",
        },
        {
            "name":               "Gephyrin — dual role: MoCoD scaffold (type C) AND inhibitory synapse anchor",
            "definition":         (
                "GPHN encodes gephyrin, which has two biologically distinct and clinically important functions: "
                "(1) MOLYBDENUM COFACTOR SYNTHESIS (G-domain, C-terminus): "
                "gephyrin acts as molybdate insertase, catalyzing the final step of MoCo assembly: "
                "molybdopterin (MPT) + molybdate → MoCo; "
                "biallelic LOF → MoCoD type C (severe neonatal: sulfite positive, uric acid low, seizures, encephalopathy). "
                "(2) INHIBITORY SYNAPSE SCAFFOLD (E-domain, N-terminus): "
                "gephyrin forms trimers → hexagonal lattice at inhibitory postsynaptic densities; "
                "anchors GABA-A receptors (most γ2 and α subunits) and glycine receptors (GlyR); "
                "connects receptors to microtubules and actin cytoskeleton; "
                "recruits collybistin (ARHGEF9) for membrane targeting. "
                "(2a) SCAFFOLDING-ONLY VARIANTS: missense variants affecting only the E-domain (scaffolding): "
                "epilepsy + autism + intellectual disability WITHOUT MoCoD biochemistry; "
                "sulfite NEGATIVE; uric acid NORMAL; MoCo synthesis intact; "
                "distinct and milder disorder — avoid confusing with MoCoD type C. "
                "CLINICAL RULE: always check sulfite + uric acid in any GPHN-variant patient to confirm which mechanism."
            ),
            "clinical_relevance": "GPHN mutations may cause severe MoCoD (sulfite positive, uric acid low) OR milder epilepsy+ASD (sulfite negative, uric acid normal) — always check both",
        },
        {
            "name":               "Primary cascade split algorithm — neonatal epileptic encephalopathy with unknown etiology",
            "definition":         (
                "SYSTEMATIC APPROACH for neonatal epileptic encephalopathy (NEE) with B6/MoCoD/sulfite disorders in DDx: "
                "STEP 1: FRESH urine sulfite dipstick immediately. "
                "  POSITIVE → STEP 2a. "
                "  NEGATIVE (or cannot test fresh) → STEP 2b. "
                "STEP 2a (sulfite positive): plasma uric acid. "
                "  LOW uric acid → MoCoD; ORDER plasma cPMP immediately. "
                "    cPMP ABSENT → MOCS1 type A; START fosdenopterin IMMEDIATELY (FDA-approved); send MOCS1 sequencing. "
                "    cPMP ELEVATED → MOCS2 (type B) or GPHN (type C); supportive care; gene sequencing subtype; gene therapy trials. "
                "  NORMAL uric acid → isolated SUOX (ISOD); confirm SSC elevated; SUOX enzyme + sequencing; low-sulfur diet. "
                "STEP 2b (sulfite negative): B6 trial. "
                "  Pyridoxine 100 mg IV (ICU; monitor apnea) → seizures stop → ALDH7A1 (PDE1); confirm AASA + pipecolic acid; start pyridoxine 30 mg/kg/day + lysine restriction. "
                "  Pyridoxine NO RESPONSE → PLP 30 mg/kg/day → seizures stop → PNPO; confirm CSF pyridoxamine/PNP elevated + threonine; monitor LFTs. "
                "  BOTH work → PLPBP; B6 vitamer profiling; PLPBP sequencing. "
                "  Neither, but plasma proline >1000 µmol/L → ALDH4A1 (HPII); urinary P5C; ALDH4A1 sequencing. "
                "KEY RULES: (1) sulfite fresh test first; (2) uric acid splits MoCoD from SUOX; (3) cPMP absent = type A = ONLY fosdenopterin indication; "
                "(4) pyridoxine failure + PLP success = PNPO (do NOT delay PLP after pyridoxine failure)."
            ),
            "clinical_relevance": "Follow this cascade for any NEE: sulfite test → uric acid → cPMP OR pyridoxine/PLP trial; each step gates treatment and is time-critical in neonates",
        },
    ]

    treatment_matrix = [
        {
            "intervention":    "Pyridoxine (vitamin B6) 15-30 mg/kg/day",
            "indications":     "ALDH7A1 (PDE1) — first-line, lifelong; PLPBP — effective; ALDH4A1 — adjunct; empirical trial in NEE",
            "mechanism":       "Exogenous pyridoxine → phosphorylated to PNP by pyridoxal kinase → converted to PLP by PNPO; overcomes PLP inactivation by P6C (ALDH7A1) or P5C (ALDH4A1)",
            "dose":            "100 mg IV (emergency trial); 15-30 mg/kg/day oral (maintenance, ALDH7A1); titrate to seizure control",
            "contraindicated": "PNPO deficiency: pyridoxine WILL NOT WORK (cannot be converted to PLP); delay in PLP treatment causes irreversible brain damage",
        },
        {
            "intervention":    "Pyridoxal-5'-phosphate (PLP) 30-50 mg/kg/day",
            "indications":     "PNPO deficiency — MANDATORY (pyridoxine useless); PLPBP — alternative; ALDH7A1 if pyridoxine partially insufficient",
            "mechanism":       "PLP is the active cofactor form; directly available to all PLP-dependent enzymes including GAD (GABA synthesis), does not require PNPO conversion",
            "dose":            "30-50 mg/kg/day in divided doses; monitor LFTs closely (hepatotoxicity risk at high doses)",
            "contraindicated": "High-dose PLP can cause hepatotoxicity — monitor LFTs monthly initially; reduce dose if transaminases rise",
        },
        {
            "intervention":    "Lysine restriction + arginine supplementation (PDE1/ALDH7A1 adjunct)",
            "indications":     "ALDH7A1 deficiency — improves neurodevelopmental outcomes significantly; reduces AASA/pipecolic acid flux",
            "mechanism":       "Reduces lysine flux through ALDH7A1 pathway → less P6C formed → less PLP inactivation; arginine competes with lysine at BBB (LAT1 transporter), reducing cerebral lysine delivery",
            "dose":            "Lysine restricted to 80-100 mg/kg/day in infants; arginine 150-200 mg/kg/day supplemented; dietitian required",
            "contraindicated": "Lysine restriction should NOT be excessive — lysine is essential; maintain adequate protein nutrition",
        },
        {
            "intervention":    "Fosdenopterin (NULIBRY, recombinant cPMP) — daily IV infusion",
            "indications":     "MoCoD type A ONLY (MOCS1 deficiency; cPMP absent confirmed); FDA-approved 2021; start within days of diagnosis",
            "mechanism":       "Provides exogenous cPMP bypassing MOCS1 block; intact MOCS2 and GPHN enzymes process cPMP to MoCo; restores all four molybdoenzymes including SUOX",
            "dose":            "0.9 mg/kg/day IV daily; home infusion possible; monitor via urinary sulfite normalization",
            "contraindicated": "MoCoD type B (MOCS2) or type C (GPHN): cPMP CANNOT be processed; fosdenopterin USELESS; isolated SUOX: MoCo is normal; fosdenopterin not indicated",
        },
        {
            "intervention":    "Low-sulfur amino acid diet (MoCoD / isolated SUOX)",
            "indications":     "All MoCoD types and isolated SUOX — reduces sulfite substrate load; adjunct to fosdenopterin in type A",
            "mechanism":       "Dietary methionine and cysteine are primary sulfur sources; reducing intake decreases sulfite generation; reduces S-sulfocysteine accumulation",
            "dose":            "Methionine restriction to ~RDA; specialized metabolic formula (low-sulfur amino acid formula available); dietitian essential",
            "contraindicated": "Excessive restriction causes cystine/methionine deficiency; monitor plasma amino acids; maintain essential amino acid intake",
        },
    ]

    return {
        "atlas_id":              ATLAS_ID,
        "key_distinguishers":    key_distinguishers,
        "treatment_matrix":      treatment_matrix,
        "seeds":                 f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "primary_cascade_split": (
            "STEP 1: Fresh urine sulfite dipstick. POSITIVE → plasma uric acid. "
            "  Uric acid LOW → MoCoD; plasma cPMP: ABSENT → MOCS1 type A (fosdenopterin WORKS); ELEVATED → MOCS2/GPHN (fosdenopterin FAILS). "
            "  Uric acid NORMAL → isolated SUOX; S-sulfocysteine elevated; no fosdenopterin. "
            "STEP 2: Sulfite NEGATIVE → B6 trial: "
            "  Pyridoxine 100mg IV → stops → ALDH7A1 (check AASA + pipecolic acid); "
            "  PLP works but NOT pyridoxine → PNPO; "
            "  Both work → PLPBP; "
            "  Proline >1000 µmol/L + P5C elevated → ALDH4A1 (hyperprolinemia II)."
        ),
        "diagnostic_groups": {
            "b6_responsive": {
                "genes": B6_RESPONSIVE_GROUP,
                "discriminators": {
                    "ALDH7A1": "Pyridoxine works; AASA + pipecolic acid elevated on treatment",
                    "PNPO":    "PLP ONLY; CSF pyridoxamine/PNP elevated; threonine elevated",
                    "PLPBP":   "Both B6 forms work; AASA negative; vitamer profiling",
                    "ALDH4A1": "Proline >1000 µmol/L; P5C elevated; incomplete penetrance",
                },
            },
            "mocoд_types": {
                "genes": MOCOД_GROUP,
                "discriminators": {
                    "MOCS1": "cPMP absent; fosdenopterin indication; FDA-approved 2021",
                    "MOCS2": "cPMP elevated; MPT synthase block; no fosdenopterin; gene therapy",
                    "GPHN":  "MoCoD type C OR scaffolding-only epilepsy+ASD (dual mechanism); check sulfite",
                },
            },
            "isolated_suox": {
                "genes": SULFITE_OXIDASE_GROUP,
                "discriminators": {
                    "SUOX": "Sulfite positive + uric acid NORMAL; no fosdenopterin; low-sulfur diet",
                },
            },
        },
    }
