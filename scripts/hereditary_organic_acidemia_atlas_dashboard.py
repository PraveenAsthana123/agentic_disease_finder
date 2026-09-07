#!/usr/bin/env python3
"""Hereditary-Organic-Acidemia-Atlas — Complete 8-Gene Hereditary Organic Acidemia Atlas
PCCA    (propionyl-CoA carboxylase alpha; 728 aa; 13q32.3; AR;
         Propionic Acidemia type A — C3-propionylcarnitine NBS;
         cardiomyopathy DCM 20-30%; prolonged QTc; hyperammonaemia via NAGS inhibition;
         VPA ABSOLUTELY CI; fasting CI; liver transplant reduces crises NOT cardiomyopathy;
         seed SEED_BASE+0) .
PCCB    (propionyl-CoA carboxylase beta; 539 aa; 3q22.3; AR;
         Propionic Acidemia type B — identical clinical phenotype to PCCA;
         Korean founder p.Gln272Ter ~40% Korean PA alleles; Italian founder;
         seed SEED_BASE+1) .
MMUT    (methylmalonyl-CoA mutase; 750 aa; 6p12.3; AR;
         Methylmalonic Acidemia mut type — AdoCbl-dependent;
         mut0 (no activity) vs mut- (partial); OHCbl trial mandatory 3-5 days;
         renal tubulointerstitial nephritis → CKD major morbidity;
         kidney transplant helps renal NOT metabolic; liver-kidney combined most definitive;
         seed SEED_BASE+2) .
IVD     (isovaleryl-CoA dehydrogenase; 415 aa; 15q15.1; AR;
         Isovaleric Acidemia — C5-isovalerylcarnitine NBS PATHOGNOMONIC;
         sweaty feet/cheese odour PATHOGNOMONIC; German founder p.Ala282Val mild form;
         glycine conjugation therapy 250 mg/kg/day; L-carnitine; leucine restriction;
         seed SEED_BASE+3) .
GCDH    (glutaryl-CoA dehydrogenase; 438 aa; 19p13.2; AR;
         Glutaric Aciduria Type 1 — macrocephaly HALLMARK;
         frontotemporal atrophy + subdural hygromas PATHOGNOMONIC MRI;
         striatal necrosis from febrile crisis 6 months–6 years window;
         emergency protocol MANDATORY during febrile illness;
         Amish/Old Order Mennonite founder p.Ala421Val;
         seed SEED_BASE+4) .
MCCC1   (3-methylcrotonyl-CoA carboxylase alpha; 709 aa; 3q27.1; AR;
         3-MCC Deficiency type 1 — C5OH most common NBS OA flag in many programmes;
         USUALLY BENIGN; NOT biotin-responsive (single carboxylase, not MCD);
         leucine restriction only if symptomatic;
         seed SEED_BASE+5) .
ACAT1   (acetoacetyl-CoA thiolase; 427 aa; 11q22.3; AR;
         Beta-Ketothiolase Deficiency T2 — isoleucine catabolism + ketone body utilisation;
         episodes of severe ketoacidosis DISPROPORTIONATE to fasting/illness;
         2-methylacetoacetate + 2-methyl-3-hydroxybutyrate + tiglylglycine PATHOGNOMONIC urine;
         between episodes usually asymptomatic; excellent prognosis if crises managed;
         VPA CI;
         seed SEED_BASE+6) .
HLCS    (holocarboxylase synthetase; 726 aa; 21q22.13; AR;
         Holocarboxylase Synthetase Deficiency — neonatal multiple carboxylase deficiency;
         activates ALL four biotin-dependent carboxylases (PC, PCC, 3-MCC, ACC);
         BIOTIN-RESPONSIVE 10-40 mg/day — complete/near-complete correction;
         skin rash + alopecia + lactic acidosis + ketoacidosis + hyperammonaemia COMBINATION;
         biotin life-long mandatory; contrast BTD (recycling defect) vs HLCS (attachment defect);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1838–1845)
"""

import random

SEED_BASE = 1870

OA_GENES = [
    # -- PCCA -- Propionic Acidemia type A -----------------------------------------
    {
        "gene": "PCCA",
        "protein": (
            "PCCA -- 13q32.3 AR -- Propionyl-CoA-Carboxylase-Alpha-728aa -- "
            "Propionic-Acidemia-TypeA-PCC-Alpha-Biotin-Carboxylase-Domain -- "
            "Heterotetramer-Alpha2Beta2-Biotin-Dependent-ATP-CO2 -- "
            "C3-Propionylcarnitine-Elevated-NBS -- "
            "Cardiomyopathy-DCM-20-30pct-Prolonged-QTc -- "
            "VPA-ABSOLUTELY-CI-Fasting-CI-Liver-Transplant-NOT-Prevent-Cardiomyopathy"
        ),
        "alias": (
            "PCCA (propionyl-CoA carboxylase alpha subunit); OMIM gene 232000; "
            "Propionic Acidemia OMIM 606054. "
            "13q32.3; 728 aa; ~80 kDa; mitochondrial matrix; biotin-dependent; autosomal recessive. "
            "FUNCTION: PCCA encodes the alpha subunit of propionyl-CoA carboxylase (PCC), "
            "a biotin-dependent mitochondrial enzyme. PCC is a heterotetramer (alpha2beta2): "
            "PCCA provides the biotin-carboxylase domain (binds biotin, carboxylates it using ATP + CO2); "
            "PCCB provides the carboxyl-transferase domain (transfers carboxyl group to propionyl-CoA). "
            "REACTION: Propionyl-CoA + HCO3- + ATP → D-methylmalonyl-CoA + ADP + Pi. "
            "D-methylmalonyl-CoA → L-methylmalonyl-CoA (epimerase) → succinyl-CoA (MMUT, AdoCbl-dependent). "
            "PCC is the entry point for propionate metabolism. "
            "SUBSTRATES generating propionyl-CoA: "
            "catabolism of odd-chain fatty acids (C15, C17 — terminal 3C); "
            "catabolism of branched-chain amino acids: isoleucine, valine, threonine, methionine; "
            "gut bacterial propionate production; "
            "cholesterol side-chain degradation. "
            "In PCCA/PCCB deficiency: propionyl-CoA accumulates → "
            "propionylcarnitine (C3) accumulates in blood/urine; "
            "propionyl-CoA inhibits N-acetylglutamate synthase (NAGS) → "
            "NAGS produces N-acetylglutamate (NAG) → NAG activates CPS1 (first urea cycle enzyme); "
            "without NAG: urea cycle fails → SECONDARY HYPERAMMONAEMIA; "
            "propionyl-CoA also inhibits succinyl-CoA synthetase and other TCA cycle enzymes → "
            "secondary metabolic disruption. "
            "Methylcitrate (propionyl-CoA + oxaloacetate via citrate synthase mimic) accumulates → "
            "direct neurotoxin and inhibits pyruvate carboxylase → lactic acidosis. "
            "CLINICAL PRESENTATION: "
            "NEONATAL FORM (most common severe): "
            "normal at birth → days 2-5: poor feeding, vomiting, lethargy → "
            "metabolic acidosis (anion gap) + ketoacidosis + hyperammonaemia → "
            "encephalopathy → coma/death without treatment; "
            "neutropenia + thrombocytopenia (bone marrow suppression by propionyl-CoA); "
            "SUBACUTE/LATE-ONSET: "
            "older child/adult; episodic decompensation triggered by illness/fasting/high protein; "
            "developmental delay common from prior episodes; "
            "CARDIAC COMPLICATIONS (major cause of late morbidity/mortality): "
            "dilated cardiomyopathy (DCM) in 20-30% of patients; "
            "prolonged QTc → ventricular arrhythmia risk; "
            "cardiac complications may occur INDEPENDENT of metabolic control; "
            "cause not fully understood — propionyl-CoA may directly impair cardiac energy metabolism; "
            "echocardiogram + ECG MANDATORY at diagnosis and annually. "
            "LIVER TRANSPLANTATION: "
            "liver provides ~90% of PCC enzyme activity; "
            "liver transplant dramatically reduces acute metabolic crises; "
            "HOWEVER: liver transplant does NOT prevent cardiomyopathy progression — "
            "cardiac PCC activity not restored by liver alone; "
            "patients still need cardiac monitoring post-transplant. "
            "DIAGNOSIS: "
            "NBS: C3 (propionylcarnitine) elevated; C3/C2 (acetylcarnitine) ratio elevated; "
            "urine organic acids: 3-hydroxypropionate, methylcitrate, propionylglycine, tiglylglycine; "
            "plasma amino acids: glycine elevated (propionyl-CoA inhibits glycine cleavage); "
            "PCCA + PCCB gene panel (both must be sequenced — phenotype identical). "
            "TREATMENT: "
            "DIETARY (cornerstone): "
            "protein restriction: limit isoleucine/valine/threonine/methionine intake; "
            "natural protein reduced + amino acid formula without offending amino acids; "
            "odd-chain fat restriction (minimal in typical diet); "
            "PHARMACOLOGICAL: "
            "metronidazole (reduces gut bacterial propionate) + rifaximin: "
            "intermittent courses to reduce propionate load from gut flora; "
            "L-carnitine supplementation (propionylcarnitine formation depletes carnitine); "
            "biotin: does not help (PCCA/PCCB are structural mutations — biotin unresponsive "
            "unless rare biotin-responsive mutation); "
            "ACUTE CRISIS: "
            "STOP all protein for 24-48h (zero protein period — critical); "
            "IV glucose 10% at anti-catabolic rate 8-12 mg/kg/min; "
            "IV L-carnitine; "
            "ammonia scavengers (sodium benzoate/phenylbutyrate) if NH3 >200 µmol/L; "
            "haemodialysis for severe hyperammonaemia (>500 µmol/L) or metabolic acidosis; "
            "VPA (valproate) ABSOLUTELY CONTRAINDICATED — "
            "valproate inhibits propionyl-CoA metabolism AND worsens hyperammonaemia → "
            "can precipitate fatal metabolic crisis; "
            "FASTING CONTRAINDICATED — mobilises odd-chain FA and amino acids. "
            "KEY CLINICAL FACTS: "
            "DCM occurs in 20-30% — INDEPENDENT of metabolic control (monitor even in well-controlled patients); "
            "prolonged QTc → sudden cardiac death risk; ECG at every clinic visit; "
            "liver transplant reduces crises but does NOT cure cardiac risk; "
            "neutropenia during crisis → infection risk (avoid as trigger); "
            "secondary carnitine depletion → supplement life-long; "
            "methylcitrate urine is a sensitive biomarker of metabolic control."
        ),
        "age_of_onset": "Neonatal (days 2-5); late-onset episodic in milder genotypes",
        "inheritance": "AR",
        "locus": "13q32.3",
        "protein_size": "728 aa",
        "key_biomarker": "C3 propionylcarnitine elevated NBS; methylcitrate + 3-hydroxypropionate urine",
        "pathognomonic": "Metabolic acidosis + hyperammonaemia + C3 elevated + methylcitrate in urine",
        "treatment": "Protein restriction; zero-protein 24-48h crisis; IV glucose 8-12 mg/kg/min; VPA CI; metronidazole",
        "critical_flags": [
            "VPA-ABSOLUTELY-CI — inhibits propionyl-CoA metabolism + worsens hyperammonaemia",
            "DCM-CARDIOMYOPATHY-20-30pct — occurs independent of metabolic control; monitor ECG + echo",
            "PROLONGED-QTc-ARRHYTHMIA — sudden cardiac death risk; ECG every visit",
            "LIVER-TRANSPLANT-NOT-PREVENT-CARDIOMYOPATHY — reduces crises but cardiac risk persists",
            "ZERO-PROTEIN-24-48h-IN-CRISIS — stop all natural protein; IV glucose 8-12 mg/kg/min",
            "FASTING-CI — mobilises propionyl-CoA substrate from odd-chain FA + amino acids",
            "METHYLCITRATE-URINE-BIOMARKER — propionyl-CoA + OAA → methylcitrate; tracks metabolic control",
            "SECONDARY-HYPERAMMONAEMIA-VIA-NAGS — propionyl-CoA inhibits NAGS → CPS1 → urea cycle failure",
        ],
    },
    # -- PCCB -- Propionic Acidemia type B -----------------------------------------
    {
        "gene": "PCCB",
        "protein": (
            "PCCB -- 3q22.3 AR -- Propionyl-CoA-Carboxylase-Beta-539aa -- "
            "Propionic-Acidemia-TypeB-PCC-Carboxyl-Transferase-Domain -- "
            "Korean-Founder-p.Gln272Ter-c.814CT-40pct-Korean-PA-Alleles -- "
            "Italian-Founder -- "
            "Identical-Clinical-Phenotype-PCCA -- "
            "C3-Propionylcarnitine-NBS-Biotin-Dependent-Heterotetramer"
        ),
        "alias": (
            "PCCB (propionyl-CoA carboxylase beta subunit); OMIM gene 232050; "
            "Propionic Acidemia OMIM 606054. "
            "3q22.3; 539 aa; ~58 kDa; mitochondrial matrix; biotin-dependent; autosomal recessive. "
            "FUNCTION: PCCB encodes the beta subunit of propionyl-CoA carboxylase (PCC). "
            "PCC is a heterotetramer (alpha2beta2) — PCCA provides biotin-carboxylase domain; "
            "PCCB provides the carboxyl-transferase domain (transfers activated CO2 from "
            "N-carboxybiotin to propionyl-CoA to form D-methylmalonyl-CoA). "
            "REACTION (identical to PCCA): Propionyl-CoA + HCO3- + ATP → D-methylmalonyl-CoA. "
            "PCCB mutations disrupt the carboxyl-transfer active site. "
            "CLINICAL PHENOTYPE: "
            "IDENTICAL TO PCCA DEFICIENCY — cannot be distinguished clinically or biochemically; "
            "gene sequencing or enzyme subunit analysis required for gene assignment. "
            "Same neonatal presentation: metabolic acidosis, hyperammonaemia, ketosis, encephalopathy; "
            "same cardiac risk: DCM 20-30%, prolonged QTc; "
            "same late complications: developmental delay, renal disease, pancytopenia. "
            "Same secondary hyperammonaemia mechanism via NAGS inhibition. "
            "FOUNDER VARIANTS: "
            "KOREAN FOUNDER: p.Gln272Ter (c.814C>T): "
            "most common PCCB variant in Korean patients (~40% of Korean PA alleles); "
            "results in premature stop codon → truncated non-functional beta subunit; "
            "Korean PA patients disproportionately carry PCCB mutations vs PCCA. "
            "ITALIAN FOUNDER: specific PCCB mutation enriched in Italian PA population. "
            "DIAGNOSIS: "
            "NBS: C3 propionylcarnitine elevated (same as PCCA — cannot distinguish on NBS); "
            "urine organic acids: methylcitrate, 3-hydroxypropionate, propionylglycine; "
            "plasma glycine elevated; "
            "enzyme assay (PCC total activity) — cannot distinguish subunit without molecular testing; "
            "PCCA + PCCB gene panel required; "
            "enzyme subunit complementation assay (research): distinguishes PCCA vs PCCB. "
            "TREATMENT: "
            "IDENTICAL TO PCCA: "
            "Protein restriction (isoleucine/valine/threonine/methionine); "
            "amino acid formula; L-carnitine; metronidazole (gut propionate reduction); "
            "ACUTE CRISIS (identical to PCCA): "
            "Zero protein 24-48h; IV glucose 8-12 mg/kg/min anti-catabolic rate; "
            "ammonia scavengers; haemodialysis if severe; "
            "VPA ABSOLUTELY CONTRAINDICATED (same as PCCA); "
            "fasting CI (same mechanism as PCCA); "
            "LIVER TRANSPLANT: "
            "same considerations as PCCA — reduces metabolic crises; "
            "does NOT prevent cardiomyopathy progression. "
            "KEY CLINICAL FACTS: "
            "PCCB and PCCA are phenotypically indistinguishable — "
            "both require gene sequencing for definitive assignment; "
            "Korean population: PCCB mutations predominate; "
            "biotin supplementation: NOT effective for PCCB structural mutations "
            "(unlike HLCS where biotin IS curative); "
            "same VPA contraindication, same cardiac monitoring, same crisis protocol as PCCA."
        ),
        "age_of_onset": "Neonatal (identical presentation to PCCA); late-onset in milder genotypes",
        "inheritance": "AR",
        "locus": "3q22.3",
        "protein_size": "539 aa",
        "key_biomarker": "C3 propionylcarnitine elevated NBS (identical to PCCA); methylcitrate urine",
        "pathognomonic": "Propionic acidemia biochemistry (identical to PCCA); Korean founder p.Gln272Ter",
        "treatment": "Protein restriction; zero-protein crisis; IV glucose; VPA CI; L-carnitine; metronidazole",
        "critical_flags": [
            "VPA-ABSOLUTELY-CI — same as PCCA; worsens hyperammonaemia + propionyl-CoA metabolism",
            "CLINICALLY-IDENTICAL-TO-PCCA — cannot distinguish without gene sequencing",
            "KOREAN-FOUNDER-p.Gln272Ter — c.814C>T; ~40% Korean PA alleles",
            "ITALIAN-FOUNDER-PCCB — Italian PA disproportionately PCCB mutations",
            "DCM-CARDIOMYOPATHY-20-30pct — same cardiac risk as PCCA; monitor ECG + echo",
            "ZERO-PROTEIN-24-48h-CRISIS — anti-catabolic IV glucose 8-12 mg/kg/min",
            "BIOTIN-NOT-EFFECTIVE — structural PCCB mutations; contrast HLCS where biotin curative",
            "LIVER-TRANSPLANT-NOT-PREVENT-CARDIOMYOPATHY — reduces crises only",
        ],
    },
    # -- MMUT -- Methylmalonic Acidemia mut type ------------------------------------
    {
        "gene": "MMUT",
        "protein": (
            "MMUT -- 6p12.3 AR -- Methylmalonyl-CoA-Mutase-750aa -- "
            "Methylmalonic-Acidemia-mut-Type-AdoCbl-Dependent -- "
            "mut0-No-Residual-Activity-vs-mut-Partial-OHCbl-Trial-Mandatory-3-5days -- "
            "Renal-Tubulointerstitial-Nephritis-CKD-Major-Morbidity -- "
            "Methylmalonate-Directly-Nephrotoxic -- "
            "Liver-Kidney-Combined-Transplant-Most-Definitive"
        ),
        "alias": (
            "MMUT (methylmalonyl-CoA mutase); OMIM gene 609058; "
            "Methylmalonic Acidemia (MMAuria), mut type, OMIM 251000. "
            "6p12.3; 750 aa; ~83 kDa; mitochondrial matrix; adenosylcobalamin (AdoCbl)-dependent; "
            "autosomal recessive. "
            "FUNCTION: MMUT catalyses the isomerisation of L-methylmalonyl-CoA → succinyl-CoA, "
            "using adenosylcobalamin (AdoCbl, coenzyme form of vitamin B12) as cofactor. "
            "This reaction is the final step converting propionyl-CoA metabolites into "
            "TCA cycle intermediate succinyl-CoA. "
            "MMUT is one of only two AdoCbl-dependent enzymes in mammals "
            "(the other: leucine catabolism enzyme). "
            "SUBSTRATES: methylmalonyl-CoA from propionyl-CoA (same sources as PA: "
            "odd-chain FA, isoleucine, valine, threonine, methionine, gut propionate, cholesterol). "
            "In MMUT deficiency: L-methylmalonyl-CoA accumulates → "
            "methylmalonate + methylmalonylcarnitine (C4DC on NBS) elevated; "
            "C3 (propionylcarnitine) also elevated (upstream block); "
            "methylmalonate is directly nephrotoxic to renal tubular cells. "
            "CLASSIFICATION: "
            "mut0: no residual MMUT enzyme activity → does NOT respond to hydroxocobalamin (OHCbl); "
            "most severe; highest methylmalonate levels; worst renal and neurological prognosis; "
            "mut-: partial residual MMUT enzyme activity → MAY respond partially to OHCbl; "
            "less severe; some improvement with B12 treatment. "
            "COBALAMIN TRIAL (MANDATORY IN ALL NEW PATIENTS): "
            "OHCbl 1 mg IM daily for 3-5 days → measure plasma methylmalonate before and after; "
            "mut0: no reduction in methylmalonate → confirmed non-responsive; "
            "mut-: significant reduction → continue OHCbl + dietary management; "
            "MUST do the trial: cannot assume mut0 without testing; "
            "distinguish from cblA/cblB (cobalamin-metabolic defects — same NBS pattern but responsive). "
            "HYPERAMMONAEMIA: same NAGS-inhibition mechanism as PA (propionyl-CoA + methylmalonyl-CoA "
            "inhibit NAGS → secondary urea cycle failure). "
            "RENAL DISEASE (key distinguishing feature vs PA): "
            "Renal tubulointerstitial nephritis → progressive CKD; "
            "methylmalonate directly damages renal tubular cells (proximal tubular dysfunction, "
            "Fanconi syndrome features); "
            "renal disease is a major cause of morbidity and mortality in adult MMA-mut patients; "
            "GFR monitoring essential; ACE inhibitors for proteinuria; "
            "avoid nephrotoxins: ibuprofen/NSAIDs, aminoglycosides, IV contrast (use iso-osmolar). "
            "KIDNEY TRANSPLANT: "
            "improves renal function; BUT: metabolic episodes continue (kidney does not express MMUT); "
            "LIVER-KIDNEY COMBINED TRANSPLANT: "
            "most definitive (liver restores MMUT enzyme + kidney restored); "
            "methylmalonate levels fall dramatically; metabolic crises reduced; "
            "renal function stabilised; considered when renal failure + recurrent crises. "
            "NEUROLOGICAL COMPLICATIONS: "
            "Metabolic stroke (striatal injury — basal ganglia) during acute decompensation; "
            "optic nerve atrophy; peripheral neuropathy; "
            "intellectual disability proportional to metabolic control. "
            "DIAGNOSIS: "
            "NBS: C3 (propionylcarnitine) + C4DC (methylmalonylcarnitine) elevated; "
            "urine organic acids: methylmalonate massively elevated; methylcitrate; 3-hydroxypropionate; "
            "plasma methylmalonate; homocysteine NORMAL (unlike cblC/D which have elevated homocysteine); "
            "OHCbl responsiveness trial; MMUT gene sequencing. "
            "TREATMENT: "
            "DIETARY: protein restriction (same amino acids as PA); amino acid formula; "
            "PHARMACOLOGICAL: "
            "OHCbl (if mut-): 1 mg IM daily or alternate days; "
            "L-carnitine supplementation; "
            "metronidazole (gut propionate/methylmalonate reduction); "
            "ACUTE CRISIS (same as PA): "
            "zero protein 24-48h; IV glucose 8-12 mg/kg/min; ammonia scavengers; dialysis; "
            "RENAL PROTECTION: "
            "avoid NSAIDs/nephrotoxins; ACE inhibitor for proteinuria; "
            "renal monitoring 6-monthly; "
            "TRANSPLANT: kidney or liver-kidney combined if indicated. "
            "KEY CLINICAL FACTS: "
            "Methylmalonate nephrotoxicity is the defining long-term morbidity in mut MMA; "
            "NSAIDs CONTRAINDICATED — directly nephrotoxic in already vulnerable kidneys; "
            "mut0 vs mut- distinction drives cobalamin therapy decision; "
            "homocysteine NORMAL differentiates mut MMA from cobalamin metabolism defects (cblC elevated HCY); "
            "metabolic stroke during decompensation → basal ganglia lesions on MRI."
        ),
        "age_of_onset": "Neonatal (mut0) or infancy/childhood (mut-); late-onset renal disease",
        "inheritance": "AR",
        "locus": "6p12.3",
        "protein_size": "750 aa",
        "key_biomarker": "C3 + C4DC (methylmalonylcarnitine) elevated NBS; massive methylmalonate in urine",
        "pathognomonic": "Massive methylmalonic aciduria + C3/C4DC + normal homocysteine + renal tubulointerstitial nephritis",
        "treatment": "OHCbl trial mandatory; protein restriction; zero-protein crisis; avoid NSAIDs; renal monitoring; liver-kidney transplant",
        "critical_flags": [
            "OHCbl-TRIAL-MANDATORY-3-5days — 1 mg IM daily; distinguishes mut0 (non-responsive) vs mut- (responsive)",
            "NSAIDs-CI-NEPHROTOXIC — methylmalonate already damages renal tubules; any nephrotoxin CI",
            "RENAL-TUBULOINTERSTITIAL-NEPHRITIS-CKD — major long-term morbidity; GFR monitor 6-monthly",
            "LIVER-KIDNEY-COMBINED-MOST-DEFINITIVE — restores MMUT + renal function simultaneously",
            "KIDNEY-TRANSPLANT-NOT-METABOLIC — kidney does not express MMUT; crises continue",
            "HOMOCYSTEINE-NORMAL — DDx cblC/D (elevated HCY); critical distinction on initial workup",
            "METABOLIC-STROKE-BASAL-GANGLIA — during decompensation; MRI shows striatal lesions",
            "MUT0-vs-MUT- — mut0 does NOT respond to OHCbl; trial still mandatory to confirm",
        ],
    },
    # -- IVD -- Isovaleric Acidemia ------------------------------------------------
    {
        "gene": "IVD",
        "protein": (
            "IVD -- 15q15.1 AR -- Isovaleryl-CoA-Dehydrogenase-415aa -- "
            "Isovaleric-Acidemia-FAD-Dependent-Leucine-Catabolism-3rd-Step -- "
            "C5-Isovalerylcarnitine-NBS-PATHOGNOMONIC -- "
            "Sweaty-Feet-Cheese-Odour-Isovaleric-Acid-PATHOGNOMONIC -- "
            "German-Founder-p.Ala282Val-c.845CT-Mild-NBS-Form -- "
            "Glycine-250mgkgday-Conjugation-Therapy-L-Carnitine-Leucine-Restriction"
        ),
        "alias": (
            "IVD (isovaleryl-CoA dehydrogenase); OMIM gene 607036; "
            "Isovaleric Acidemia (IVA) OMIM 243500. "
            "15q15.1; 415 aa; ~46 kDa; mitochondrial matrix; FAD-dependent; autosomal recessive. "
            "FUNCTION: IVD catalyses the third step of leucine catabolism: "
            "isovaleryl-CoA → 3-methylcrotonyl-CoA + FADH2. "
            "Leucine catabolism pathway: "
            "Leucine → alpha-ketoisocaproate (BCKDH) → isovaleryl-CoA (IVD step) → "
            "3-methylcrotonyl-CoA (MCCC1 step) → 3-methylglutaconyl-CoA → "
            "HMG-CoA → acetoacetate + acetyl-CoA (HMGCL). "
            "IVD is homologous to SCAD, MCAD, LCAD, VLCAD — all FAD-dependent acyl-CoA dehydrogenases; "
            "electrons from FADH2 pass to ETF/ETFDH → CoQ10 → CIII → ATP. "
            "In IVD deficiency: isovaleryl-CoA accumulates → "
            "isovaleric acid (IVA) in blood/urine; "
            "isovalerylcarnitine (C5): elevated — NBS PATHOGNOMONIC marker; "
            "isovalerylglycine (IVG): elevated in urine. "
            "PATHOGNOMONIC ODOUR: "
            "Isovaleric acid (3-methylbutanoic acid) has a characteristic sweaty feet/cheese smell; "
            "smell is strongest during acute crisis and detectable in urine, sweat, breath; "
            "clinically: nursery staff, parents, or clinicians who recognise the odour → "
            "immediate diagnostic clue especially in pre-NBS era; "
            "still clinically useful as a crisis trigger recognition signal. "
            "CLINICAL PRESENTATION: "
            "PRE-NBS: acute neonatal crisis (severe form): "
            "day 3-7 of life; poor feeding, vomiting, lethargy → metabolic acidosis, "
            "pancytopenia (bone marrow suppression — isovaleryl-CoA toxic to marrow), "
            "encephalopathy → coma; "
            "sweaty feet odour prominent; "
            "NBS-DETECTED (mild/asymptomatic form): "
            "German founder p.Ala282Val — this specific allele is associated with a milder phenotype "
            "often detected via NBS with C5 elevation and NO clinical symptoms; "
            "these patients may remain asymptomatic throughout life with minimal treatment; "
            "important to distinguish from severe-allele IVA to avoid over-treatment. "
            "GLYCINE CONJUGATION THERAPY (key unique treatment): "
            "glycine 250 mg/kg/day oral (in 3-4 divided doses): "
            "glycine conjugates with isovaleryl-CoA → isovalerylglycine (IVG); "
            "IVG is water-soluble and renally excreted → reduces isovaleryl-CoA burden; "
            "this is a DETOXIFICATION pathway, not enzyme replacement; "
            "glycine therapy is specific to IVA (and to a lesser extent PA); "
            "reduces frequency and severity of crises. "
            "L-CARNITINE: "
            "isovalerylcarnitine formation depletes free carnitine; "
            "L-carnitine 100 mg/kg/day → isovalerylcarnitine excreted renally → "
            "both reduces isovaleryl-CoA AND replenishes free carnitine. "
            "DIAGNOSIS: "
            "NBS: C5 isovalerylcarnitine elevated (PATHOGNOMONIC — DDx SBCAD deficiency for C5); "
            "urine organic acids: isovalerylglycine (most abundant, specific); "
            "3-hydroxyisovalerate (minor); isovaleric acid; "
            "plasma acylcarnitines confirm C5; IVD gene sequencing. "
            "TREATMENT: "
            "LEUCINE-RESTRICTED DIET: "
            "reduce leucine (primary substrate) to tolerated amounts; "
            "natural protein reduced; leucine-free/low-leucine amino acid formula; "
            "GLYCINE 250 mg/kg/day: conjugation detoxification — life-long; "
            "L-CARNITINE 100 mg/kg/day: detoxification + replenish free carnitine; "
            "ACUTE CRISIS: "
            "zero protein 24-48h; IV glucose anti-catabolic; "
            "IV L-carnitine; oral glycine if tolerating; "
            "VPA CONTRAINDICATED (inhibits FAO; competes with isovaleryl-CoA detoxification); "
            "SICK-DAY PROTOCOL: reduce leucine intake, increase glucose polymer, ensure carnitine/glycine. "
            "KEY CLINICAL FACTS: "
            "C5 on NBS is PATHOGNOMONIC for IVA (most programmes); "
            "sweaty feet/cheese odour = instant diagnostic clue; "
            "German founder p.Ala282Val = mild/asymptomatic NBS phenotype; "
            "glycine + carnitine are the TWO unique treatments for IVA (conjugation therapy); "
            "prognosis: excellent with treatment (NBS + glycine + carnitine); "
            "pancytopenia during crisis → treat aggressively (bone marrow toxic effect)."
        ),
        "age_of_onset": "Neonatal (severe form) or asymptomatic NBS (mild, p.Ala282Val)",
        "inheritance": "AR",
        "locus": "15q15.1",
        "protein_size": "415 aa",
        "key_biomarker": "C5 isovalerylcarnitine elevated NBS; isovalerylglycine urine",
        "pathognomonic": "C5 elevated NBS + sweaty feet/cheese odour + isovalerylglycine urine",
        "treatment": "Glycine 250 mg/kg/day; L-carnitine; leucine restriction; VPA CI",
        "critical_flags": [
            "C5-ISOVALERYLCARNITINE-NBS-PATHOGNOMONIC — most specific NBS marker for IVA",
            "SWEATY-FEET-CHEESE-ODOUR-PATHOGNOMONIC — isovaleric acid; recognise in crisis",
            "GLYCINE-250mgkgday-CONJUGATION — specific detoxification therapy; reduces isovaleryl-CoA",
            "L-CARNITINE-DETOXIFICATION — isovalerylcarnitine excreted renally; replenishes free carnitine",
            "GERMAN-FOUNDER-p.Ala282Val — mild/asymptomatic NBS phenotype; avoid over-treatment",
            "VPA-CI — inhibits FAO + competes with detoxification pathways",
            "LEUCINE-RESTRICTION — primary substrate; reduce dietary leucine",
            "PANCYTOPENIA-CRISIS — bone marrow suppression by isovaleryl-CoA; treat aggressively",
        ],
    },
    # -- GCDH -- Glutaric Aciduria Type 1 ------------------------------------------
    {
        "gene": "GCDH",
        "protein": (
            "GCDH -- 19p13.2 AR -- Glutaryl-CoA-Dehydrogenase-438aa -- "
            "Glutaric-Aciduria-Type-1-FAD-Dependent-Lysine-Tryptophan-Catabolism -- "
            "Macrocephaly-Birth-HALLMARK-75-80pct -- "
            "Frontotemporal-Atrophy-Subdural-Hygromas-PATHOGNOMONIC-MRI-Mistaken-NAI -- "
            "Striatal-Necrosis-Febrile-Crisis-6months-6years-Window -- "
            "Amish-Old-Order-Mennonite-Founder-p.Ala421Val"
        ),
        "alias": (
            "GCDH (glutaryl-CoA dehydrogenase); OMIM gene 608801; "
            "Glutaric Aciduria Type 1 (GA1) OMIM 231670. "
            "19p13.2; 438 aa; ~48 kDa; mitochondrial matrix; FAD-dependent; autosomal recessive. "
            "FUNCTION: GCDH catalyses the oxidative decarboxylation of glutaryl-CoA → "
            "crotonyl-CoA + CO2 + FADH2. "
            "This is a step in the catabolism of: "
            "L-lysine (via saccharopine → 2-oxoadipate → glutaryl-CoA pathway); "
            "L-hydroxylysine (collagen turnover); "
            "L-tryptophan (via kynurenine/3-hydroxyanthranilic acid → 2-aminomuconic semialdehyde → "
            "2-oxoadipate → glutaryl-CoA pathway). "
            "In GCDH deficiency: glutaryl-CoA accumulates → "
            "glutaric acid (GA) elevated in urine; "
            "3-hydroxyglutaric acid (3-OH-GA) elevated in urine AND CSF; "
            "glutarylcarnitine (C5DC) elevated on NBS. "
            "PATHOGENESIS OF BRAIN INJURY — CRITICAL AND UNIQUE: "
            "Glutaric acid and 3-OH-GA are neurotoxic — "
            "3-OH-GA is a glutamate receptor (NMDA) agonist; "
            "immature striatum (caudate/putamen) is selectively vulnerable during a specific age window; "
            "WINDOW OF VULNERABILITY: 6 months to 6 years of age; "
            "trigger: any febrile illness, vaccination, gastroenteritis within this age window → "
            "acute encephalopathic crisis (12-72h duration): "
            "sudden severe dystonia, altered consciousness; "
            "STRIATAL NECROSIS → bilateral caudate + putamen injury on MRI; "
            "resulting dystonia is PERMANENT (irreversible); "
            "AFTER age 6 years: striatum is 'myelinated enough' — febrile illness no longer triggers crisis; "
            "BEFORE the first crisis: NEUROLOGICALLY NORMAL DEVELOPMENT. "
            "MACROCEPHALY: "
            "75-80% of GA1 patients have macrocephaly (HC >+2 SD) at birth or within first months; "
            "caused by accumulation of glutaric acid in the brain (cellular expansion, arachnoid cysts); "
            "FRONTOTEMPORAL ATROPHY: "
            "frontal and temporal lobes do not fill the enlarged skull → "
            "prominent extra-axial CSF spaces (frontotemporal) ± subdural hygromas; "
            "frontotemporal atrophy + subdural hygromas = PATHOGNOMONIC MRI pattern for GA1; "
            "DANGER: this MRI pattern (subdural hygromas) has led to false accusations of "
            "non-accidental injury (NAI / shaken baby syndrome) — GA1 must be EXCLUDED first. "
            "LOW-EXCRETOR PHENOTYPE: "
            "~30-40% of GA1 patients are 'low excretors' (low urine glutaric acid); "
            "IMPORTANT: low excretor does NOT mean mild disease — "
            "striatal injury risk is IDENTICAL; "
            "urine GA cannot be used alone to assess risk; "
            "C5DC on NBS is usually still elevated in low excretors. "
            "EMERGENCY PROTOCOL (CRITICAL — must be in place BEFORE crisis): "
            "During febrile illness in age 6 months to 6 years: "
            "IV glucose (dextrose 10%) at anti-catabolic rate; "
            "IV L-carnitine; "
            "antipyretics (paracetamol/ibuprofen — aggressively control fever); "
            "emergency admission threshold: fever >38.5°C or any vomiting within window; "
            "protocol must be in place with local emergency department; "
            "parent-held emergency letter essential. "
            "AMISH/OLD ORDER MENNONITE FOUNDER: "
            "p.Ala421Val (c.1261C>T): common in Amish/Old Order Mennonite communities; "
            "high prevalence of GA1 in these communities. "
            "RIBOFLAVIN TRIAL: riboflavin (FAD-dependent enzyme) may have some benefit; "
            "some in-vitro improvement in residual GCDH activity with riboflavin; "
            "not routinely used but considered in riboflavin-responsive genotypes. "
            "DIAGNOSIS: "
            "NBS: C5DC (glutarylcarnitine) elevated; "
            "urine organic acids: glutaric acid + 3-hydroxyglutaric acid (confirmatory); "
            "plasma acylcarnitines: C5DC; "
            "brain MRI: frontotemporal atrophy, subdural hygromas, macrocephaly; "
            "GCDH gene sequencing. "
            "TREATMENT: "
            "Lysine-restricted diet (reduce substrate load from lysine/tryptophan); "
            "L-carnitine supplementation (glutarylcarnitine excreted, depletes free carnitine); "
            "riboflavin 100-200 mg/day (trial); "
            "EMERGENCY PROTOCOL — mandatory, parent-held, hospital-held; "
            "between crises: well child; normal development if no prior striatal injury. "
            "KEY CLINICAL FACTS: "
            "Macrocephaly at birth or in first months = HALLMARK (75-80%); "
            "frontotemporal atrophy + subdural hygromas = PATHOGNOMONIC MRI; "
            "subdural hygromas → mistaken for NAI: ALWAYS exclude GA1 first with MRI + urine OA; "
            "striatal necrosis = permanent dystonia — prevent by emergency protocol during fever; "
            "low excretor ≠ mild disease (same striatal risk); "
            "window of vulnerability closes after age 6 years."
        ),
        "age_of_onset": "Neonatal (macrocephaly at birth); acute crisis 6 months–6 years (striatal injury)",
        "inheritance": "AR",
        "locus": "19p13.2",
        "protein_size": "438 aa",
        "key_biomarker": "C5DC glutarylcarnitine NBS; glutaric acid + 3-hydroxyglutaric acid urine; macrocephaly",
        "pathognomonic": "Macrocephaly + frontotemporal atrophy + subdural hygromas MRI + C5DC elevated",
        "treatment": "Lysine restriction; L-carnitine; emergency IV glucose protocol during febrile illness 6mo–6yr",
        "critical_flags": [
            "MACROCEPHALY-HALLMARK-75-80pct — at birth or first months; first diagnostic clue",
            "FRONTOTEMPORAL-ATROPHY-SUBDURAL-HYGROMAS-PATHOGNOMONIC-MRI — NOT NAI; exclude GA1 first",
            "STRIATAL-NECROSIS-FEBRILE-CRISIS — 12-72h window; 6 months to 6 years; permanent dystonia",
            "EMERGENCY-PROTOCOL-MANDATORY — IV glucose + carnitine during fever; parent-held letter",
            "LOW-EXCRETOR-NOT-MILD — low urine GA does NOT mean low striatal risk",
            "AMISH-MENNONITE-FOUNDER-p.Ala421Val — high community prevalence",
            "NAI-MISDIAGNOSIS-RISK — subdural hygromas → false accusation of shaken baby; ALWAYS exclude GA1",
            "WINDOW-OF-VULNERABILITY-CLOSES-AGE-6 — after 6 years: no longer triggers striatal crisis",
        ],
    },
    # -- MCCC1 -- 3-MCC Deficiency type 1 ------------------------------------------
    {
        "gene": "MCCC1",
        "protein": (
            "MCCC1 -- 3q27.1 AR -- 3-Methylcrotonyl-CoA-Carboxylase-Alpha-709aa -- "
            "3-MCC-Deficiency-Type1-Biotin-Carboxylase-Domain-Leucine-Catabolism -- "
            "C5OH-3-Hydroxyisovalerylcarnitine-NBS-Most-Common-OA-Flag-Many-Programmes -- "
            "USUALLY-BENIGN-Most-Asymptomatic -- "
            "NOT-Biotin-Responsive-Single-Carboxylase-NOT-MCD -- "
            "Leucine-Restriction-Only-If-Symptomatic"
        ),
        "alias": (
            "MCCC1 (3-methylcrotonyl-CoA carboxylase subunit alpha); OMIM gene 609010; "
            "3-Methylcrotonylglycinuria (3-MCC Deficiency) OMIM 210200. "
            "3q27.1; 709 aa; ~79 kDa; mitochondrial matrix; biotin-dependent; autosomal recessive. "
            "Also caused by MCCC2 (beta subunit, OMIM 609014). "
            "FUNCTION: MCCC1 encodes the alpha (biotin-carboxylase) subunit of "
            "3-methylcrotonyl-CoA carboxylase (3-MCC). "
            "3-MCC is a biotin-dependent mitochondrial enzyme (heterotetramer, alpha2beta2 like PCC). "
            "REACTION: 3-methylcrotonyl-CoA + HCO3- + ATP → 3-methylglutaconyl-CoA. "
            "This is step 4 of leucine catabolism: "
            "Leucine → alpha-KIC (BCKDH) → isovaleryl-CoA (IVD) → 3-methylcrotonyl-CoA (3-MCC) → "
            "3-methylglutaconyl-CoA → HMG-CoA → acetoacetate (HMGCL). "
            "In 3-MCC deficiency: 3-methylcrotonyl-CoA accumulates → "
            "3-methylcrotonylglycine in urine (conjugation with glycine); "
            "3-hydroxyisovaleric acid in urine; "
            "C5OH (3-hydroxyisovalerylcarnitine) elevated on NBS. "
            "NBS EPIDEMIOLOGY — MOST COMMON NBS OA FLAG: "
            "3-MCC deficiency is the MOST COMMON organic acidemia detected on NBS in many programmes "
            "(USA, Australia, Germany) due to high carrier frequency; "
            "C5OH is the NBS marker — BUT C5OH can also be elevated in: "
            "3-MCC deficiency (MCCC1/MCCC2); "
            "3-methylglutaconyl-CoA hydratase deficiency (AUH); "
            "biotinidase deficiency (BTD); "
            "holocarboxylase synthetase deficiency (HLCS); "
            "hydroxyisobutyryl-CoA hydrolase deficiency (HIBCH); "
            "IMPORTANT: C5OH elevation on NBS has broad differential — clinical context crucial. "
            "USUALLY BENIGN PHENOTYPE: "
            "Most individuals identified via NBS for elevated C5OH are ASYMPTOMATIC; "
            "frequently identified because a MOTHER has elevated C5OH on her own NBS screening "
            "(when baby is screened and mother shares same sample or incidental maternal detection); "
            "most 3-MCC patients live completely normal lives without intervention; "
            "very FEW symptomatic cases documented — typically non-specific: "
            "hypoglycaemia, ketoacidosis during illness; "
            "metabolic crises are rare and usually mild; "
            "no cardiac, renal, or major neurological complications in typical cases. "
            "NOT BIOTIN-RESPONSIVE (CRITICAL DISTINCTION): "
            "3-MCC deficiency is a SINGLE carboxylase deficiency: "
            "only 3-MCC enzyme is defective → "
            "biotin does NOT help (biotin treats multiple carboxylase deficiency — HLCS/BTD); "
            "HLCS deficiency: all 4 carboxylases deficient → biotin-responsive (HLCS attaches biotin); "
            "BTD deficiency: biotin recycling defect → biotin-responsive; "
            "3-MCC deficiency: structural enzyme mutation → biotin does NOT help; "
            "this is the most common error: giving biotin to 3-MCC patients (futile, not harmful but misleading). "
            "DIAGNOSIS: "
            "NBS: C5OH elevated (3-hydroxyisovalerylcarnitine); "
            "urine organic acids: 3-methylcrotonylglycine + 3-hydroxyisovalerate; "
            "plasma acylcarnitines: C5OH; "
            "MCCC1 + MCCC2 gene panel (or biotinidase + HLCS to exclude biotin-responsive DDx); "
            "maternal C5OH: if mother also elevated → suggests maternal 3-MCC (common scenario). "
            "TREATMENT: "
            "Most patients: NO treatment required; "
            "if symptomatic: "
            "leucine-restricted diet (reduce substrate); "
            "L-carnitine supplementation; "
            "avoid prolonged fasting; "
            "sick-day protocol (glucose polymer during illness); "
            "BIOTIN: NOT indicated for MCCC1/MCCC2 deficiency. "
            "KEY CLINICAL FACTS: "
            "3-MCC = most common OA on NBS in many countries but mostly benign; "
            "do NOT give biotin (wrong mechanism); "
            "maternal 3-MCC often detected when infant has elevated NBS; "
            "C5OH differential: always check BTD and HLCS (both biotin-responsive and more serious); "
            "management: watchful waiting with sick-day protocol in most; "
            "prognosis: excellent; normal neurodevelopment expected."
        ),
        "age_of_onset": "Usually asymptomatic (NBS incidental); rare episodic metabolic crisis in minority",
        "inheritance": "AR",
        "locus": "3q27.1",
        "protein_size": "709 aa",
        "key_biomarker": "C5OH 3-hydroxyisovalerylcarnitine elevated NBS; 3-methylcrotonylglycine urine",
        "pathognomonic": "C5OH on NBS + 3-methylcrotonylglycine urine + usually asymptomatic",
        "treatment": "Usually no treatment; leucine restriction + L-carnitine only if symptomatic; biotin NOT indicated",
        "critical_flags": [
            "USUALLY-BENIGN-MOST-ASYMPTOMATIC — do not over-treat; most NBS-detected patients healthy",
            "NOT-BIOTIN-RESPONSIVE — single carboxylase deficiency; biotin does NOT help MCCC1",
            "C5OH-MOST-COMMON-NBS-OA-FLAG — check BTD and HLCS first (both more serious, biotin-responsive)",
            "MATERNAL-3MCC-COMMON — mother may have same C5OH elevation; maternal NBS scenario",
            "NO-CARDIAC-RENAL-NEUROLOGICAL-RISK — benign course; watchful waiting appropriate",
            "LEUCINE-RESTRICTION-ONLY-IF-SYMPTOMATIC — not needed in asymptomatic patients",
            "DDx-C5OH-BROAD — AUH, BTD, HLCS, HIBCH also cause C5OH elevation",
            "BIOTIN-ERROR-COMMON — most common management mistake; biotin futile for MCCC1",
        ],
    },
    # -- ACAT1 -- Beta-Ketothiolase Deficiency (T2) ---------------------------------
    {
        "gene": "ACAT1",
        "protein": (
            "ACAT1 -- 11q22.3 AR -- Mitochondrial-Acetoacetyl-CoA-Thiolase-427aa -- "
            "Beta-Ketothiolase-Deficiency-T2-Isoleucine-Catabolism-Ketone-Utilisation -- "
            "Episodes-Severe-Ketoacidosis-DISPROPORTIONATE-pH<7.1-Bicarb<5-PATHOGNOMONIC -- "
            "2-Methylacetoacetate-2-Methyl-3-Hydroxybutyrate-Tiglylglycine-Urine-PATHOGNOMONIC -- "
            "C5:1-Tiglylcarnitine-C5OH-NBS -- "
            "Between-Episodes-Asymptomatic-Excellent-Prognosis -- "
            "VPA-CI"
        ),
        "alias": (
            "ACAT1 (acetyl-CoA acetyltransferase 1; mitochondrial acetoacetyl-CoA thiolase; "
            "T2 or beta-ketothiolase); OMIM gene 607809; "
            "Beta-Ketothiolase Deficiency (BKT; T2 deficiency) OMIM 203750. "
            "11q22.3; 427 aa; ~45 kDa; mitochondrial matrix; autosomal recessive. "
            "FUNCTION: ACAT1 (T2) catalyses two distinct reactions: "
            "1. ISOLEUCINE CATABOLISM (specific substrate): "
            "thiolytic cleavage of 2-methylacetoacetyl-CoA → "
            "propionyl-CoA + acetyl-CoA (final step of isoleucine catabolism); "
            "this reaction is UNIQUE to T2 (ACAT1) — not done by other thiolases; "
            "2. KETONE BODY UTILISATION (extrahepatic ketolysis): "
            "reversible condensation: 2 acetyl-CoA ↔ acetoacetyl-CoA (T2 catalyses both); "
            "in ketone-utilising tissues (brain, muscle, heart): "
            "acetoacetate → acetoacetyl-CoA (requires succinyl-CoA via OXCT1, then T2 thiolysis); "
            "T2 is the primary thiolase for extrahepatic ketolysis; "
            "DISTINCT FROM ACAT2 (cytoplasmic T1/T2 — cholesterol synthesis) and HMGCS2 (ketogenesis). "
            "In ACAT1 deficiency: "
            "2-methylacetoacetyl-CoA accumulates (from isoleucine catabolism) → "
            "2-methylacetoacetate (2-MAA) in urine; "
            "2-methyl-3-hydroxybutyrate (2-M-3-OHB) in urine; "
            "tiglylglycine (TG) in urine (glycine conjugate of tiglyl-CoA); "
            "acetoacetyl-CoA cannot be efficiently cleaved → "
            "impaired ketone body utilisation → ketones accumulate disproportionately. "
            "CLINICAL PRESENTATION — KEY DIAGNOSTIC FEATURE: "
            "EPISODIC SEVERE KETOACIDOSIS DISPROPORTIONATE TO CLINICAL STATE: "
            "during fasting, illness, or high-protein intake → "
            "ketoacidosis with pH <7.1 and bicarbonate <5 mmol/L; "
            "severity of ketoacidosis is OUT OF PROPORTION to degree of fasting/illness; "
            "CLINICALLY: the metabolic crisis is more severe than expected for the trigger; "
            "the combination of: severe ketoacidosis + relatively minor precipitant = "
            "diagnostic clue for BKT deficiency; "
            "consciousness: variable (encephalopathy in severe episodes); "
            "vomiting, tachypnoea (Kussmaul breathing); "
            "BETWEEN EPISODES: "
            "patients are COMPLETELY NORMAL — no residual neurological impairment between crises; "
            "EXCELLENT prognosis if crises are recognised and managed; "
            "NO structural organ damage between episodes (unlike MMA with chronic renal injury). "
            "URINE ORGANIC ACID PROFILE — PATHOGNOMONIC: "
            "2-methylacetoacetate (2-MAA): present only in BKT deficiency (most specific); "
            "2-methyl-3-hydroxybutyrate (2-M-3-OHB): elevated (2nd most specific); "
            "tiglylglycine (TG): elevated (glycine conjugate); "
            "acetoacetate and 3-hydroxybutyrate: massively elevated (ketones); "
            "urine organic acids during crisis: all four present = PATHOGNOMONIC pattern. "
            "NBS: "
            "C5:1 (tiglylcarnitine) elevated: MOST SPECIFIC NBS marker for BKT; "
            "C5OH (3-hydroxyisovalerylcarnitine or 2-methyl-3-hydroxybutyrylcarnitine): may be elevated; "
            "but C5:1 is more specific than C5OH for BKT. "
            "ISOLEUCINE: restrict during episodes (reduces 2-MAA production); "
            "not strictly restricted between episodes in many patients. "
            "DIAGNOSIS: "
            "Urine organic acids during crisis: 2-MAA + 2-M-3-OHB + TG (PATHOGNOMONIC pattern); "
            "plasma acylcarnitines: C5:1 elevated; "
            "ACAT1 gene sequencing (must exclude HADHA-T2-like variants); "
            "enzyme assay: thiolase activity in fibroblasts; "
            "clinically: severe ketoacidosis disproportionate to trigger in episodic pattern. "
            "TREATMENT: "
            "ACUTE CRISIS: "
            "IV sodium bicarbonate (correct severe acidosis pH <7.1); "
            "IV glucose 10% at anti-catabolic rate (stop catabolism); "
            "zero protein 24-48h; "
            "L-carnitine IV; "
            "CHRONIC: "
            "Avoid prolonged fasting; "
            "mild isoleucine restriction (prudent but less strict than MSUD); "
            "L-carnitine supplementation; "
            "VPA CONTRAINDICATED (inhibits FAO + worsens ketoacidosis); "
            "SICK-DAY PROTOCOL: "
            "early glucose polymer at first sign of illness; "
            "admit early (lower threshold than typical paediatric presentation); "
            "parent-held emergency letter essential (ketoacidosis out of proportion to illness). "
            "KEY CLINICAL FACTS: "
            "Severe ketoacidosis DISPROPORTIONATE to illness = hallmark clinical clue; "
            "2-MAA in urine = MOST SPECIFIC organic acid marker; "
            "between episodes: completely normal (excellent prognosis if crises managed); "
            "NBS: C5:1 tiglylcarnitine most specific; "
            "VPA CI; "
            "prognosis: excellent — normal neurodevelopment expected with crisis management."
        ),
        "age_of_onset": "Infancy to childhood (episodic); triggered by illness/fasting/high protein",
        "inheritance": "AR",
        "locus": "11q22.3",
        "protein_size": "427 aa",
        "key_biomarker": "C5:1 tiglylcarnitine NBS; 2-methylacetoacetate + 2-methyl-3-hydroxybutyrate + tiglylglycine urine",
        "pathognomonic": "Severe ketoacidosis DISPROPORTIONATE to illness + 2-MAA + 2-M-3-OHB + tiglylglycine urine",
        "treatment": "IV bicarbonate + glucose in crisis; fasting CI; VPA CI; isoleucine restriction during episodes",
        "critical_flags": [
            "KETOACIDOSIS-DISPROPORTIONATE-TO-ILLNESS-PATHOGNOMONIC — pH <7.1 bicarb <5 from minor trigger",
            "2-METHYLACETOACETATE-URINE-MOST-SPECIFIC — only organic acid specific to BKT deficiency",
            "C5:1-TIGLYLCARNITINE-NBS-MOST-SPECIFIC — best NBS marker for ACAT1",
            "BETWEEN-EPISODES-COMPLETELY-NORMAL — excellent prognosis; no chronic organ damage",
            "IV-BICARBONATE-SODIUM-CRISIS — severe acidosis pH <7.1 requires bicarb correction",
            "VPA-CI — worsens ketoacidosis + inhibits FAO",
            "EXCELLENT-PROGNOSIS — normal neurodevelopment if crises managed; reassure families",
            "EARLY-ADMISSION-THRESHOLD — lower threshold; ketoacidosis progresses faster than expected",
        ],
    },
    # -- HLCS -- Holocarboxylase Synthetase Deficiency ------------------------------
    {
        "gene": "HLCS",
        "protein": (
            "HLCS -- 21q22.13 AR -- Holocarboxylase-Synthetase-726aa -- "
            "Neonatal-Multiple-Carboxylase-Deficiency-MCD-Neonatal -- "
            "Activates-ALL-Four-Biotin-Dependent-Carboxylases-PC-PCC-3MCC-ACC -- "
            "BIOTIN-RESPONSIVE-10-40mgday-Complete-Near-Complete-Correction -- "
            "Skin-Rash-Alopecia-Lactic-Acidosis-Ketoacidosis-Hyperammonaemia-COMBINATION -- "
            "Contrast-BTD-Biotinidase-Recycling-vs-HLCS-Attachment -- "
            "Biotin-Life-Long-Mandatory"
        ),
        "alias": (
            "HLCS (holocarboxylase synthetase); OMIM gene 609018; "
            "Holocarboxylase Synthetase Deficiency (HLCS deficiency; MCD-neonatal) OMIM 253270. "
            "21q22.13; 726 aa; ~80 kDa; mitochondrial and cytoplasmic; autosomal recessive. "
            "FUNCTION: HLCS (holocarboxylase synthetase) is the enzyme that covalently "
            "ATTACHES biotin to apo-carboxylases to form active holoenzymes. "
            "Reaction: Apo-carboxylase + Biotin + ATP → Holo-carboxylase (active) + AMP + PPi. "
            "HLCS biotinylates ALL FOUR biotin-dependent carboxylases in human cells: "
            "1. Pyruvate carboxylase (PC) — gluconeogenesis + anaplerosis (mitochondrial); "
            "2. Propionyl-CoA carboxylase (PCC) — propionate metabolism (mitochondrial); "
            "3. 3-Methylcrotonyl-CoA carboxylase (3-MCC) — leucine catabolism (mitochondrial); "
            "4. Acetyl-CoA carboxylase (ACC1 + ACC2) — fatty acid synthesis regulation (cytoplasmic/mitochondrial). "
            "Without HLCS: ALL four carboxylases remain as inactive apo-forms → "
            "multiple carboxylase deficiency (MCD): "
            "PC inactive → lactic acidosis (pyruvate cannot be carboxylated to OAA); "
            "PCC inactive → propionate accumulation → propionylcarnitine (C3) + methylcitrate; "
            "3-MCC inactive → 3-methylcrotonylglycine + C5OH; "
            "ACC inactive → reduced malonyl-CoA (fatty acid synthesis impaired). "
            "BIOTIN PHARMACOLOGY: "
            "HLCS deficiency: high-dose biotin (10-40 mg/day) provides excess biotin "
            "substrate to overcome the reduced affinity of mutant HLCS for biotin → "
            "COMPLETE or near-complete biochemical correction in most mutations; "
            "biotin must be started IMMEDIATELY upon diagnosis — "
            "before irreversible neurological damage; "
            "BIOTIN IS LIFE-LONG MANDATORY — cannot stop (enzyme activity does not recover); "
            "if biotin is stopped: metabolic decompensation may occur within days-weeks. "
            "CLINICAL PRESENTATION — COMBINATION HALLMARK: "
            "NEONATAL/EARLY INFANTILE ONSET (typically day 1-7): "
            "Skin rash (perioral, perinasal, perigenital — eczematous/erythematous); "
            "alopecia (partial or total hair loss) — may not be obvious in neonates; "
            "lactic acidosis (PC inactive → pyruvate accumulates); "
            "ketoacidosis (3-MCC + PCC both inactive → substrates accumulate); "
            "hyperammonaemia (PCC inactive → propionyl-CoA → inhibits NAGS → secondary urea cycle); "
            "The COMBINATION of: "
            "skin rash + alopecia + lactic acidosis + ketoacidosis + hyperammonaemia "
            "= PATHOGNOMONIC for biotin-dependent MCD (HLCS or BTD); "
            "DIFFERENTIATE HLCS vs BTD: "
            "HLCS: neonatal onset (enzyme that attaches biotin); "
            "BTD: late infantile onset (enzyme that recycles biotin from biocytin); "
            "both are biotin-responsive; "
            "BTD activity assay: deficient in BTD, normal in HLCS; "
            "HLCS gene sequencing confirms. "
            "URINE ORGANIC ACIDS: "
            "Multiple metabolites elevated simultaneously: "
            "3-methylcrotonylglycine (3-MCC inactive); "
            "methylcitric acid (PCC inactive); "
            "3-hydroxypropionic acid (PCC inactive); "
            "lactic acid (PC inactive); "
            "3-hydroxyisovaleric acid (3-MCC inactive); "
            "MULTIPLE organic acids from MULTIPLE carboxylase pathways = MCD pattern. "
            "NBS: C5OH + C3 + elevated lactate on screening; "
            "if NBS shows C5OH + C3 simultaneously → MCD (HLCS or BTD); "
            "single C5OH without C3 → 3-MCC deficiency (not MCD). "
            "DIAGNOSIS: "
            "Urine organic acids: multiple (above pattern); "
            "plasma acylcarnitines: C5OH + C3; "
            "blood lactate: elevated; ammonia: elevated; "
            "biotinidase activity (blood): NORMAL (BTD excluded); "
            "HLCS enzyme assay (fibroblasts); "
            "HLCS gene sequencing. "
            "TREATMENT: "
            "BIOTIN 10-40 mg/day oral — START IMMEDIATELY; "
            "response within 24-72h: dramatic improvement in biochemical markers; "
            "clinical improvement: rash clears, alopecia may reverse, encephalopathy improves; "
            "BIOTIN LIFE-LONG — must be taken daily, cannot stop; "
            "SUPPORTIVE: IV glucose (acute crisis); ammonia scavengers if needed; "
            "monitor biotin responsiveness with urine organic acids; "
            "CONTRAST: BTD — also biotin-responsive but lower dose (1-10 mg); "
            "no dietary restriction needed if biotin adequate. "
            "KEY CLINICAL FACTS: "
            "HLCS = enzyme that ATTACHES biotin to apo-carboxylases (not recycling); "
            "BTD = enzyme that RECYCLES biotin from biocytin (cleavage product of biotinylated histones); "
            "both cause biotin-responsive MCD; both respond to biotin; different mechanisms; "
            "skin rash + alopecia = biotin-deficiency phenotype (same appearance as nutritional biotin deficiency); "
            "C5OH + C3 on NBS simultaneously = HLCS until proven otherwise; "
            "biotin is cheap, safe, and curative — start empirically if HLCS/BTD suspected."
        ),
        "age_of_onset": "Neonatal day 1-7 (most); rarely late infantile",
        "inheritance": "AR",
        "locus": "21q22.13",
        "protein_size": "726 aa",
        "key_biomarker": "C5OH + C3 simultaneously on NBS + elevated lactate; multiple organic acids urine",
        "pathognomonic": "Skin rash + alopecia + lactic acidosis + ketoacidosis + hyperammonaemia COMBINATION; biotin-responsive",
        "treatment": "Biotin 10-40 mg/day immediately; life-long mandatory; cannot stop",
        "critical_flags": [
            "BIOTIN-RESPONSIVE-COMPLETE-CORRECTION — 10-40 mg/day; start immediately; do not delay",
            "BIOTIN-LIFE-LONG-MANDATORY — cannot stop; metabolic crisis if withdrawn",
            "SKIN-RASH-ALOPECIA-COMBINATION-HALLMARK — with lactic acidosis + ketoacidosis + hyperammonaemia",
            "HLCS-ATTACHMENT-vs-BTD-RECYCLING — both biotin-responsive; different mechanisms; BTD assay differentiates",
            "C5OH-PLUS-C3-NBS-SIMULTANEOUSLY — multiple carboxylase = HLCS/BTD; single C5OH = 3-MCC",
            "ALL-FOUR-CARBOXYLASES-INACTIVE — PC + PCC + 3MCC + ACC; multiple organic acids simultaneously",
            "NEONATAL-ONSET-DAY-1-7 — earlier than BTD (late infantile); neonatal rash = biotin deficiency",
            "MULTIPLE-ORGANIC-ACIDS-URINE — 3-methylcrotonylglycine + methylcitrate + 3-OHP + lactate",
        ],
    },
]


def _make_patients(gene_data, seed):
    rng = random.Random(seed)
    gene = gene_data["gene"]
    inheritance = gene_data["inheritance"]
    severity_choices = ["mild", "moderate", "severe"]
    # Severity weights by clinical phenotype
    sev_weights = {
        "PCCA":  [20, 40, 40],   # neonatal severe; late-onset moderate
        "PCCB":  [20, 40, 40],   # identical to PCCA
        "MMUT":  [15, 35, 50],   # mut0 predominates; renal morbidity
        "IVD":   [45, 35, 20],   # NBS-detected mild form common (p.Ala282Val)
        "GCDH":  [25, 40, 35],   # macrocephaly + striatal risk; varies
        "MCCC1": [65, 25, 10],   # USUALLY BENIGN; most mild/asymptomatic
        "ACAT1": [40, 40, 20],   # episodic; excellent inter-episode; crises can be severe
        "HLCS":  [20, 35, 45],   # neonatal severe before biotin; biotin-treated improves
    }
    # Crisis history rates (True probability) by gene
    crisis_probs = {
        "PCCA":  0.70,
        "PCCB":  0.70,
        "MMUT":  0.75,
        "IVD":   0.45,
        "GCDH":  0.60,
        "MCCC1": 0.15,
        "ACAT1": 0.65,
        "HLCS":  0.55,
    }
    weights = sev_weights.get(gene, [25, 45, 30])
    crisis_p = crisis_probs.get(gene, 0.50)
    patients = []
    for i in range(40):
        age = rng.randint(0, 60)
        sex = rng.choice(["M", "F"])
        severity = rng.choices(severity_choices, weights=weights)[0]
        on_diet = rng.random() < 0.70
        family_cascade = rng.random() < 0.45
        crisis_history = rng.random() < crisis_p
        biotin_responsive = (gene == "HLCS")
        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis": age,
            "sex": sex,
            "severity": severity,
            "inheritance": inheritance,
            "on_diet": on_diet,
            "key_biomarker_abnormal": True,
            "family_cascade": family_cascade,
            "crisis_history": crisis_history,
            "biotin_responsive": biotin_responsive,
        })
    return patients


def _build_cohort():
    all_patients = []
    for idx, gene_data in enumerate(OA_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_patients(gene_data, seed))
    return all_patients


COHORT = _build_cohort()


# ── API response functions ─────────────────────────────────────────────────────

def overview():
    total = len(COHORT)
    gene_counts = {}
    severity_counts = {"mild": 0, "moderate": 0, "severe": 0}
    for p in COHORT:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
        severity_counts[p["severity"]] = severity_counts.get(p["severity"], 0) + 1

    on_diet      = sum(1 for p in COHORT if p.get("on_diet"))
    cascade      = sum(1 for p in COHORT if p.get("family_cascade"))
    crisis       = sum(1 for p in COHORT if p.get("crisis_history"))
    biotin_resp  = sum(1 for p in COHORT if p.get("biotin_responsive"))

    genes_covered = len(OA_GENES)
    ar_genes = sum(1 for g in OA_GENES if g["inheritance"] == "AR")

    gene_summary = []
    disease_shorts = {
        "PCCA":  "Propionic Acidemia type A",
        "PCCB":  "Propionic Acidemia type B",
        "MMUT":  "Methylmalonic Acidemia (mut type)",
        "IVD":   "Isovaleric Acidemia",
        "GCDH":  "Glutaric Aciduria Type 1",
        "MCCC1": "3-MCC Deficiency (usually benign)",
        "ACAT1": "Beta-Ketothiolase Deficiency",
        "HLCS":  "Holocarboxylase Synthetase Deficiency",
    }
    nbs_markers = {
        "PCCA":  "C3 propionylcarnitine",
        "PCCB":  "C3 propionylcarnitine",
        "MMUT":  "C3 + C4DC methylmalonylcarnitine",
        "IVD":   "C5 isovalerylcarnitine",
        "GCDH":  "C5DC glutarylcarnitine",
        "MCCC1": "C5OH 3-hydroxyisovalerylcarnitine",
        "ACAT1": "C5:1 tiglylcarnitine",
        "HLCS":  "C5OH + C3 simultaneously",
    }
    key_findings = {
        "PCCA":  "DCM cardiomyopathy 20-30%; VPA absolutely CI; liver Tx not prevent cardiac",
        "PCCB":  "Korean founder p.Gln272Ter; clinically identical to PCCA; VPA CI",
        "MMUT":  "Renal tubulointerstitial nephritis CKD; OHCbl trial mandatory; NSAIDs CI",
        "IVD":   "Sweaty feet/cheese odour PATHOGNOMONIC; glycine conjugation therapy",
        "GCDH":  "Macrocephaly 75-80%; frontotemporal atrophy MRI; striatal necrosis febrile crisis",
        "MCCC1": "USUALLY BENIGN; most asymptomatic; NOT biotin-responsive",
        "ACAT1": "Severe ketoacidosis disproportionate to illness; between episodes completely normal",
        "HLCS":  "Biotin-responsive 10-40 mg/day complete correction; all 4 carboxylases inactive",
    }
    management_pearls = {
        "PCCA":  "Zero-protein 24-48h crisis; IV glucose 8-12 mg/kg/min; monitor ECG/echo annually",
        "PCCB":  "Identical crisis protocol to PCCA; metronidazole reduces gut propionate",
        "MMUT":  "OHCbl 1 mg IM 3-5 days trial; avoid NSAIDs; liver-kidney transplant most definitive",
        "IVD":   "Glycine 250 mg/kg/day + L-carnitine 100 mg/kg/day (conjugation therapy)",
        "GCDH":  "Emergency IV glucose protocol during febrile illness 6mo-6yr; parent-held letter",
        "MCCC1": "Watchful waiting; leucine restriction only if symptomatic; do NOT give biotin",
        "ACAT1": "IV bicarbonate + glucose in crisis; early admission threshold; VPA CI",
        "HLCS":  "Start biotin 10-40 mg/day immediately; life-long mandatory; cannot stop",
    }
    for g in OA_GENES:
        gene_summary.append({
            "gene": g["gene"],
            "disease_short": disease_shorts[g["gene"]],
            "inheritance": g["inheritance"],
            "chromosome": g["locus"],
            "protein_size_aa": int(g["protein_size"].replace(" aa", "")),
            "nbs_marker": nbs_markers[g["gene"]],
            "key_finding": key_findings[g["gene"]],
            "management_pearl": management_pearls[g["gene"]],
        })

    return {
        "atlas": (
            "Hereditary-Organic-Acidemia-Atlas — Complete 8-Gene Hereditary Organic Acidemia Atlas"
        ),
        "subtitle": (
            "PCCA (PropiAcidemia-A-C3-NBS-DCM-20-30pct-VPA-CI-LivTx-NOT-Cardiac) · "
            "PCCB (PropiAcidemia-B-Korean-Founder-p.Q272X-Italian-Founder-Identical-PCCA) · "
            "MMUT (MMA-mut-AdoCbl-OHCbl-Trial-Renal-CKD-NSAIDs-CI-LiverKidney-Tx) · "
            "IVD (IVA-C5-NBS-SweakyFeet-PATHOGNOMONIC-Glycine-LCarnitine-German-Founder) · "
            "GCDH (GA1-Macrocephaly-FrontotemporalMRI-PATHOGNOMONIC-Striatal-Necrosis-Amish) · "
            "MCCC1 (3MCC-C5OH-MostCommon-NBS-OA-USUALLY-BENIGN-NOT-Biotin-Responsive) · "
            "ACAT1 (BKT-T2-SevereKetoacidosis-DISPROPORTIONATE-2MAA-NormalBetweenEpisodes) · "
            "HLCS (MCD-Neonatal-BIOTIN-RESPONSIVE-10-40mg-AllFour-Carboxylases-LifeLong) — "
            f"320 Patients (8×40, Seeds {SEED_BASE}–{SEED_BASE+7})"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "aggregate_stats": {
            "genes_covered": genes_covered,
            "ar_genes": ar_genes,
            "x_linked_genes": 0,
            "ad_genes": 0,
            "patients_per_gene": total // genes_covered,
            "on_diet_pct": round(on_diet / total * 100, 1),
            "family_cascade_pct": round(cascade / total * 100, 1),
            "crisis_history_pct": round(crisis / total * 100, 1),
            "biotin_responsive_pct": round(biotin_resp / total * 100, 1),
            "severity_mild_pct": round(severity_counts["mild"] / total * 100, 1),
            "severity_moderate_pct": round(severity_counts["moderate"] / total * 100, 1),
            "severity_severe_pct": round(severity_counts["severe"] / total * 100, 1),
        },
        "gene_summary": gene_summary,
        "top_alerts": [
            flag
            for g in OA_GENES
            for flag in g["critical_flags"][:2]
        ],
        "critical_treatment_alerts": [
            flag
            for g in OA_GENES
            for flag in g["critical_flags"]
        ],
    }


def breakdown():
    per_gene = {}
    for g in OA_GENES:
        gene = g["gene"]
        pts = [p for p in COHORT if p["gene"] == gene]
        mild     = sum(1 for p in pts if p["severity"] == "mild")
        moderate = sum(1 for p in pts if p["severity"] == "moderate")
        severe   = sum(1 for p in pts if p["severity"] == "severe")
        on_diet  = sum(1 for p in pts if p.get("on_diet"))
        crisis   = sum(1 for p in pts if p.get("crisis_history"))
        cascade  = sum(1 for p in pts if p.get("family_cascade"))
        per_gene[gene] = {
            "gene": gene,
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "n_patients": len(pts),
            "severity": {"mild": mild, "moderate": moderate, "severe": severe},
            "on_diet": on_diet,
            "on_diet_pct": round(on_diet / len(pts) * 100, 1) if pts else 0,
            "crisis_history": crisis,
            "crisis_history_pct": round(crisis / len(pts) * 100, 1) if pts else 0,
            "family_cascade": cascade,
            "family_cascade_pct": round(cascade / len(pts) * 100, 1) if pts else 0,
            "biotin_responsive": all(p.get("biotin_responsive") for p in pts),
            "protein_description": g["protein"],
            "age_of_onset": g["age_of_onset"],
        }
    return {
        "atlas": "Hereditary-Organic-Acidemia-Atlas — Per-Gene Breakdown",
        "genes": per_gene,
        "aggregate": {
            "total_patients": len(COHORT),
            "total_genes": len(OA_GENES),
            "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
            "all_inheritance": list({g["inheritance"] for g in OA_GENES}),
        },
    }


def definitions():
    defs = []
    omim_genes = {
        "PCCA": "232000", "PCCB": "232050", "MMUT": "609058",
        "IVD": "607036", "GCDH": "608801", "MCCC1": "609010",
        "ACAT1": "607809", "HLCS": "609018",
    }
    omim_diseases = {
        "PCCA": "606054 (Propionic Acidemia)",
        "PCCB": "606054 (Propionic Acidemia)",
        "MMUT": "251000 (Methylmalonic Acidemia, mut type)",
        "IVD":  "243500 (Isovaleric Acidemia)",
        "GCDH": "231670 (Glutaric Aciduria Type 1)",
        "MCCC1": "210200 (3-Methylcrotonylglycinuria)",
        "ACAT1": "203750 (Beta-Ketothiolase Deficiency)",
        "HLCS": "253270 (Holocarboxylase Synthetase Deficiency)",
    }
    full_names = {
        "PCCA": "Propionyl-CoA Carboxylase Alpha Subunit (Biotin-Carboxylase Domain)",
        "PCCB": "Propionyl-CoA Carboxylase Beta Subunit (Carboxyl-Transferase Domain)",
        "MMUT": "Methylmalonyl-CoA Mutase (Adenosylcobalamin-Dependent Isomerase)",
        "IVD":  "Isovaleryl-CoA Dehydrogenase (FAD-Dependent; Leucine Catabolism Step 3)",
        "GCDH": "Glutaryl-CoA Dehydrogenase (FAD-Dependent; Lysine/Tryptophan Catabolism)",
        "MCCC1": "3-Methylcrotonyl-CoA Carboxylase Alpha Subunit (Biotin-Carboxylase Domain; Leucine Catabolism Step 4)",
        "ACAT1": "Mitochondrial Acetoacetyl-CoA Thiolase (T2; Isoleucine Catabolism + Ketone Utilisation)",
        "HLCS": "Holocarboxylase Synthetase (Biotin-Ligase; Activates All Four Biotin-Dependent Carboxylases)",
    }
    protein_functions = {
        "PCCA": (
            "Alpha subunit of propionyl-CoA carboxylase (PCC, heterotetramer alpha2beta2). "
            "Provides the biotin-carboxylase domain: carboxylates biotin using ATP + CO2. "
            "PCC converts propionyl-CoA → D-methylmalonyl-CoA (first step of propionate disposal). "
            "Substrate sources: odd-chain FA, isoleucine, valine, threonine, methionine, gut propionate."
        ),
        "PCCB": (
            "Beta subunit of propionyl-CoA carboxylase (PCC, heterotetramer alpha2beta2). "
            "Provides the carboxyl-transferase domain: transfers activated CO2 from N-carboxybiotin "
            "to propionyl-CoA forming D-methylmalonyl-CoA. Structural mutation — biotin does not help."
        ),
        "MMUT": (
            "Methylmalonyl-CoA mutase — adenosylcobalamin (AdoCbl)-dependent isomerase. "
            "Converts L-methylmalonyl-CoA → succinyl-CoA (TCA cycle entry). "
            "One of only two AdoCbl-dependent enzymes in human cells. "
            "mut0 (no activity) vs mut- (partial); OHCbl trial 3-5 days mandatory to classify."
        ),
        "IVD": (
            "Isovaleryl-CoA dehydrogenase — FAD-dependent; mitochondrial matrix. "
            "Catalyses step 3 of leucine catabolism: isovaleryl-CoA → 3-methylcrotonyl-CoA + FADH2. "
            "Homologous to SCAD/MCAD/VLCAD family. Electrons transferred via ETF → respiratory chain."
        ),
        "GCDH": (
            "Glutaryl-CoA dehydrogenase — FAD-dependent; mitochondrial matrix. "
            "Catalyses oxidative decarboxylation of glutaryl-CoA → crotonyl-CoA + CO2. "
            "Operates in lysine, hydroxylysine, and tryptophan catabolism pathways. "
            "Deficiency leads to neurotoxic accumulation of glutaric and 3-hydroxyglutaric acids."
        ),
        "MCCC1": (
            "Alpha subunit (biotin-carboxylase domain) of 3-methylcrotonyl-CoA carboxylase (3-MCC). "
            "Step 4 of leucine catabolism: 3-methylcrotonyl-CoA → 3-methylglutaconyl-CoA. "
            "Biotin-dependent but structural mutation — biotin unresponsive (single carboxylase defect)."
        ),
        "ACAT1": (
            "Mitochondrial acetoacetyl-CoA thiolase (T2/beta-ketothiolase). "
            "Dual function: (1) thiolysis of 2-methylacetoacetyl-CoA → propionyl-CoA + acetyl-CoA "
            "(isoleucine catabolism final step, unique to T2); "
            "(2) reversible condensation of 2 acetyl-CoA ↔ acetoacetyl-CoA (ketone body utilisation)."
        ),
        "HLCS": (
            "Holocarboxylase synthetase — biotin protein ligase. "
            "Covalently attaches biotin to apo-carboxylases using ATP: "
            "apo-carboxylase + biotin + ATP → holo-carboxylase + AMP + PPi. "
            "Activates ALL four biotin-dependent carboxylases: PC, PCC, 3-MCC, ACC1/ACC2. "
            "Distinct from BTD (biotinidase) which recycles biotin from biocytin."
        ),
    }
    clinical_features = {
        "PCCA": [
            "Neonatal metabolic crisis: metabolic acidosis + ketoacidosis + hyperammonaemia (days 2-5)",
            "Dilated cardiomyopathy (DCM) in 20-30% — independent of metabolic control",
            "Prolonged QTc on ECG → ventricular arrhythmia risk → sudden cardiac death",
            "Pancytopenia/neutropenia during crisis (bone marrow suppression by propionyl-CoA)",
            "Secondary hyperammonaemia via propionyl-CoA inhibition of NAGS → urea cycle failure",
            "Methylcitrate elevated in urine (propionyl-CoA + OAA) — biomarker of metabolic control",
            "Liver transplant reduces metabolic crises but does NOT prevent cardiomyopathy",
            "VPA ABSOLUTELY CONTRAINDICATED — inhibits propionyl-CoA metabolism + worsens NH3",
        ],
        "PCCB": [
            "Clinically and biochemically identical to PCCA deficiency",
            "Korean founder p.Gln272Ter (c.814C>T): ~40% of Korean PA alleles",
            "Italian founder variant enriched in Italian PA population",
            "DCM 20-30% and prolonged QTc — same cardiac risk as PCCA",
            "Secondary hyperammonaemia via NAGS inhibition — same mechanism as PCCA",
            "VPA ABSOLUTELY CONTRAINDICATED",
            "Liver transplant reduces crises but does NOT prevent cardiomyopathy",
            "Gene sequencing required to assign PCCA vs PCCB (NBS and biochemistry identical)",
        ],
        "MMUT": [
            "Neonatal metabolic crisis (mut0): severe acidosis + hyperammonaemia + ketoacidosis",
            "mut0 (no MMUT activity): does NOT respond to hydroxocobalamin — trial still mandatory",
            "mut- (partial activity): MAY respond to OHCbl — 1 mg IM daily 3-5 days mandatory trial",
            "Renal tubulointerstitial nephritis → progressive CKD — major long-term morbidity",
            "Methylmalonate is directly nephrotoxic to renal proximal tubular cells",
            "NSAIDs CONTRAINDICATED — further nephrotoxic in already vulnerable kidneys",
            "Metabolic stroke (basal ganglia/striatal lesions) during decompensation",
            "Homocysteine NORMAL — differentiates from cblC/D (elevated HCY in cblC)",
            "Kidney transplant: improves renal but metabolic crises continue (no MMUT in kidney)",
            "Liver-kidney combined transplant: most definitive (restores MMUT + renal function)",
        ],
        "IVD": [
            "Sweaty feet/cheese odour from isovaleric acid — PATHOGNOMONIC clinical smell",
            "C5 isovalerylcarnitine on NBS — PATHOGNOMONIC (most specific NBS marker for IVA)",
            "Isovalerylglycine in urine — most abundant and specific organic acid marker",
            "Neonatal form (severe alleles): crisis day 3-7, pancytopenia, encephalopathy",
            "German founder p.Ala282Val: mild NBS-detected asymptomatic phenotype — avoid over-treatment",
            "Glycine 250 mg/kg/day: conjugation therapy; reduces isovaleryl-CoA burden",
            "L-carnitine 100 mg/kg/day: isovalerylcarnitine excreted renally; replenishes free carnitine",
            "VPA CONTRAINDICATED; leucine-restricted diet (primary substrate reduction)",
        ],
        "GCDH": [
            "Macrocephaly at birth or first months in 75-80% of patients — HALLMARK",
            "Frontotemporal atrophy + subdural hygromas: PATHOGNOMONIC MRI pattern",
            "Subdural hygromas mistaken for non-accidental injury (NAI/shaken baby) — ALWAYS exclude GA1",
            "Window of vulnerability: 6 months to 6 years — febrile illness triggers striatal necrosis",
            "Striatal necrosis (caudate/putamen bilateral) → permanent dystonia",
            "Emergency protocol MANDATORY: IV glucose + carnitine during febrile illness in window",
            "Low-excretor phenotype (low urine GA): does NOT mean mild disease — same striatal risk",
            "Amish/Old Order Mennonite founder p.Ala421Val — high community prevalence",
            "3-hydroxyglutaric acid in CSF — most sensitive CNS biomarker",
            "After age 6 years: febrile illness no longer triggers striatal crisis (window closed)",
        ],
        "MCCC1": [
            "USUALLY BENIGN — most patients identified on NBS remain asymptomatic throughout life",
            "C5OH (3-hydroxyisovalerylcarnitine) most common organic acidemia NBS flag in many programmes",
            "NOT biotin-responsive — single carboxylase deficiency; biotin does NOT help MCCC1",
            "Maternal 3-MCC common: elevated C5OH may be from mother, not infant",
            "3-methylcrotonylglycine + 3-hydroxyisovalerate in urine",
            "C5OH broad differential: also BTD, HLCS, AUH, HIBCH — always check BTD/HLCS first (more serious)",
            "Very few symptomatic patients documented; leucine restriction only if symptomatic",
            "No cardiac, renal, or major neurological risk in typical cases",
        ],
        "ACAT1": [
            "Episodes of SEVERE ketoacidosis DISPROPORTIONATE to fasting/illness — hallmark",
            "pH <7.1, bicarbonate <5 mmol/L from relatively minor trigger",
            "2-methylacetoacetate (2-MAA) in urine — MOST SPECIFIC organic acid; unique to BKT",
            "2-methyl-3-hydroxybutyrate + tiglylglycine also elevated — PATHOGNOMONIC pattern",
            "C5:1 tiglylcarnitine on NBS — most specific NBS marker for ACAT1",
            "Between episodes: completely NORMAL — no residual neurological impairment",
            "EXCELLENT prognosis if crises managed — reassure families",
            "VPA CONTRAINDICATED; early admission threshold needed (acidosis progresses fast)",
        ],
        "HLCS": [
            "Neonatal/early infantile onset (day 1-7): skin rash + alopecia + metabolic acidosis",
            "COMBINATION hallmark: skin rash + alopecia + lactic acidosis + ketoacidosis + hyperammonaemia",
            "ALL FOUR biotin-dependent carboxylases inactive: PC + PCC + 3-MCC + ACC",
            "Multiple organic acids in urine simultaneously (3-methylcrotonylglycine + methylcitrate + 3-OHP + lactate)",
            "C5OH + C3 simultaneously on NBS = multiple carboxylase deficiency (HLCS or BTD)",
            "BIOTIN-RESPONSIVE: 10-40 mg/day oral → complete or near-complete biochemical correction",
            "BIOTIN LIFE-LONG MANDATORY — cannot stop; metabolic crisis if withdrawn",
            "HLCS (attaches biotin) vs BTD (recycles biotin from biocytin) — different mechanisms, both biotin-responsive",
        ],
    }
    pathognomonic_map = {
        "PCCA":  ["Metabolic acidosis + hyperammonaemia + C3 elevated + methylcitrate urine",
                  "DCM cardiomyopathy independent of metabolic control",
                  "Secondary hyperammonaemia via NAGS inhibition by propionyl-CoA"],
        "PCCB":  ["Biochemically identical to PCCA (C3 + methylcitrate + hyperammonaemia)",
                  "Korean founder p.Gln272Ter (~40% Korean PA alleles)",
                  "Gene panel required: PCCA vs PCCB cannot be distinguished clinically"],
        "MMUT":  ["Massive methylmalonic aciduria + normal homocysteine",
                  "C3 + C4DC on NBS simultaneously",
                  "Renal tubulointerstitial nephritis in adulthood",
                  "Metabolic stroke with bilateral basal ganglia lesions on MRI"],
        "IVD":   ["Sweaty feet/cheese odour from isovaleric acid",
                  "C5 isovalerylcarnitine on NBS",
                  "Isovalerylglycine (most abundant specific organic acid in urine)"],
        "GCDH":  ["Macrocephaly at birth (75-80%)",
                  "Frontotemporal atrophy + subdural hygromas on brain MRI",
                  "Bilateral caudate/putamen (striatal) necrosis post-crisis on MRI",
                  "3-hydroxyglutaric acid in urine and CSF"],
        "MCCC1": ["C5OH elevated on NBS (most common NBS OA flag)",
                  "3-methylcrotonylglycine in urine",
                  "Usually asymptomatic — NBS incidental finding"],
        "ACAT1": ["Severe ketoacidosis disproportionate to clinical trigger (pH <7.1, bicarb <5)",
                  "2-methylacetoacetate (2-MAA) in urine — unique to BKT",
                  "C5:1 tiglylcarnitine on NBS",
                  "Complete normality between episodes"],
        "HLCS":  ["Skin rash + alopecia + lactic acidosis + ketoacidosis + hyperammonaemia COMBINATION",
                  "C5OH + C3 simultaneously on NBS",
                  "Multiple organic acids from all carboxylase pathways in urine",
                  "Complete biochemical correction with biotin 10-40 mg/day"],
    }
    management_map = {
        "PCCA":  ["VPA ABSOLUTELY CONTRAINDICATED — fatal metabolic crisis risk",
                  "Zero protein 24-48h during acute crisis",
                  "IV glucose 8-12 mg/kg/min anti-catabolic rate in crisis",
                  "ECG + echocardiogram at diagnosis and annually (DCM risk)",
                  "Metronidazole courses to reduce gut bacterial propionate",
                  "L-carnitine supplementation life-long",
                  "Liver transplant consideration for recurrent crises"],
        "PCCB":  ["VPA ABSOLUTELY CONTRAINDICATED",
                  "Identical crisis protocol to PCCA (zero protein; IV glucose 8-12 mg/kg/min)",
                  "Protein restriction: reduce Ile/Val/Thr/Met",
                  "Metronidazole for gut propionate reduction",
                  "Cardiac monitoring identical to PCCA (DCM risk)"],
        "MMUT":  ["OHCbl 1 mg IM daily 3-5 days — mandatory trial in ALL new patients",
                  "NSAIDs ABSOLUTELY CONTRAINDICATED — nephrotoxic",
                  "GFR monitoring every 6 months (renal protection priority)",
                  "ACE inhibitor for proteinuria when CKD develops",
                  "Liver-kidney combined transplant for advanced CKD + recurrent crises",
                  "Zero protein + IV glucose crisis protocol (same as PA)"],
        "IVD":   ["Glycine 250 mg/kg/day oral (3-4 divided doses) — conjugation therapy life-long",
                  "L-carnitine 100 mg/kg/day (detoxification + replenish free carnitine)",
                  "Leucine-restricted diet; amino acid formula",
                  "VPA CONTRAINDICATED",
                  "Zero protein + IV glucose + carnitine in acute crisis"],
        "GCDH":  ["Emergency IV glucose + L-carnitine protocol during ANY febrile illness (6mo-6yr)",
                  "Parent-held emergency letter to emergency department — MANDATORY",
                  "Lysine-restricted diet (reduce substrate from lysine/tryptophan)",
                  "L-carnitine supplementation (glutarylcarnitine depletes free carnitine)",
                  "Riboflavin 100-200 mg/day (trial; some in-vitro benefit)",
                  "After age 6 years: reduced crisis risk but continue diet"],
        "MCCC1": ["BIOTIN NOT INDICATED — structural MCCC1 mutation; biotin futile",
                  "Watchful waiting in asymptomatic NBS-detected patients",
                  "Leucine restriction only if symptomatic",
                  "Sick-day protocol: glucose polymer during illness",
                  "Reassurance: most patients live completely normal lives"],
        "ACAT1": ["IV sodium bicarbonate in acute crisis (pH <7.1) — correct severe acidosis",
                  "IV glucose 10% anti-catabolic rate; zero protein 24-48h",
                  "VPA CONTRAINDICATED",
                  "Lower admission threshold — ketoacidosis progresses faster than expected",
                  "Parent-held emergency letter essential",
                  "Mild isoleucine restriction between episodes (prudent not strict)"],
        "HLCS":  ["Biotin 10-40 mg/day oral — START IMMEDIATELY upon diagnosis",
                  "Biotin LIFE-LONG MANDATORY — cannot stop or omit doses",
                  "IV glucose + ammonia scavengers in acute crisis (before biotin takes effect)",
                  "Monitor urine organic acids to confirm biochemical response to biotin",
                  "Distinguish from BTD: biotinidase activity assay (normal in HLCS; low in BTD)",
                  "No dietary protein restriction needed if on adequate biotin"],
    }
    contraindicated_map = {
        "PCCA":  ["Valproate (VPA) — ABSOLUTELY CONTRAINDICATED", "Fasting (mobilises propionyl-CoA substrates)"],
        "PCCB":  ["Valproate (VPA) — ABSOLUTELY CONTRAINDICATED", "Fasting"],
        "MMUT":  ["NSAIDs (ibuprofen, diclofenac, naproxen) — nephrotoxic", "Aminoglycoside antibiotics — nephrotoxic", "IV radiocontrast — use iso-osmolar only"],
        "IVD":   ["Valproate (VPA)", "Excess leucine / leucine supplements", "Prolonged fasting"],
        "GCDH":  ["Prolonged fasting (increases lysine catabolism)", "Unrestricted lysine intake during window of vulnerability"],
        "MCCC1": ["Biotin supplementation (futile; not harmful but misleading)", "Over-restriction in asymptomatic patients"],
        "ACAT1": ["Valproate (VPA) — worsens ketoacidosis", "Prolonged fasting"],
        "HLCS":  ["Stopping biotin (causes metabolic decompensation within days-weeks)", "Dietary biotin restriction"],
    }
    founder_map = {
        "PCCA":  None,
        "PCCB":  ["Korean: p.Gln272Ter (c.814C>T) — ~40% Korean PA alleles", "Italian: specific PCCB variant enriched in Italian PA"],
        "MMUT":  None,
        "IVD":   ["German: p.Ala282Val (c.845C>T) — mild NBS-detected asymptomatic phenotype"],
        "GCDH":  ["Amish/Old Order Mennonite: p.Ala421Val (c.1261C>T) — high community prevalence"],
        "MCCC1": None,
        "ACAT1": None,
        "HLCS":  None,
    }

    for g in OA_GENES:
        gene = g["gene"]
        defs.append({
            "gene": gene,
            "full_name": full_names[gene],
            "protein_function": protein_functions[gene],
            "omim_gene": omim_genes[gene],
            "omim_disease": omim_diseases[gene],
            "chromosome": g["locus"],
            "protein_size_aa": int(g["protein_size"].replace(" aa", "")),
            "inheritance": g["inheritance"],
            "key_clinical_features": clinical_features[gene],
            "nbs_marker": g["key_biomarker"],
            "pathognomonic_findings": pathognomonic_map[gene],
            "management_mandatories": management_map[gene],
            "contraindicated": contraindicated_map[gene],
            "founder_variants": founder_map[gene],
        })

    return {
        "atlas": "Hereditary-Organic-Acidemia-Atlas — Clinical Definitions",
        "definitions": defs,
        "total_genes": len(OA_GENES),
        "total_definition_entries": len(defs),
    }
