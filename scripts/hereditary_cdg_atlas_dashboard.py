#!/usr/bin/env python3
"""Hereditary-CDG-Atlas — Complete 8-Gene Congenital Disorders of Glycosylation Atlas
PMM2    (CDG-Ia; 246 aa; 16p13.2; AR;
         Most common CDG (70%+ of all CDGs);
         Phosphomannomutase 2 — mannose-6-phosphate ↔ mannose-1-phosphate;
         Cerebellar hypoplasia PATHOGNOMONIC;
         Inverted nipples + subcutaneous fat pads;
         Olivopontocerebellar atrophy MRI;
         Transferrin IEF Type I pattern;
         p.Pro153Leu (c.458C>T) most common European allele (~70%);
         seed SEED_BASE+0) ·
MPI     (CDG-Ib; 423 aa; 15q24.1; AR;
         Mannose phosphate isomerase — fructose-6-P ↔ mannose-6-P;
         NO neurology (KEY DDx from PMM2);
         Protein-losing enteropathy + hepatopathy + coagulopathy + hypoglycaemia;
         TREATABLE with oral D-mannose (1 g/kg/day in 4 doses) — CURATIVE;
         seed SEED_BASE+1) ·
ALG6    (CDG-Ic; 507 aa; 1p31.3; AR;
         Dolichyl-PP-Man9GlcNAc2 alpha-1,3-glucosyltransferase;
         2nd most common N-glycosylation CDG;
         Milder than PMM2; cerebellar ataxia + intellectual disability;
         Transferrin IEF Type I pattern;
         seed SEED_BASE+2) ·
PGM1    (CDG-PGM1; 562 aa; 1p31.3; AR;
         Phosphoglucomutase 1 — glucose-1-P ↔ glucose-6-P;
         Bifid uvula PATHOGNOMONIC;
         Dilated cardiomyopathy + hepatopathy + exercise intolerance;
         TREATABLE with galactose (0.5 g/kg/day) — significant improvement;
         Mixed transferrin IEF Type I/II pattern;
         seed SEED_BASE+3) ·
SLC35A2 (CDG-IIm; 396 aa; Xp11.23; X-linked de novo dominant (females);
         UDP-galactose Golgi transporter;
         First X-linked CDG; heterozygous de novo in females;
         Males hemizygous = typically non-viable;
         Epilepsy + intellectual disability + dysmorphism;
         Transferrin IEF Type II pattern;
         seed SEED_BASE+4) ·
SLC35C1 (CDG-IIc / LAD-II; 364 aa; 11p11.2; AR;
         GDP-fucose Golgi transporter;
         Leukocyte Adhesion Deficiency type II;
         Bombay blood group (H-antigen absent) PATHOGNOMONIC;
         Sialyl-LewisX absent — leukocytes cannot roll → severe recurrent infections;
         TREATABLE with oral L-fucose (500 mg/kg/day);
         seed SEED_BASE+5) ·
DOLK    (CDG-Im; 538 aa; 9q34.11; AR;
         Dolichol kinase — phosphorylation of dolichol to dolichyl-P;
         Dilated cardiomyopathy DOMINANT feature;
         Ichthyosis + hepatopathy + neurological;
         Transferrin IEF Type I pattern;
         seed SEED_BASE+6) ·
COG7    (CDG-IIe; 841 aa; 16p12.2; AR;
         Conserved Oligomeric Golgi complex subunit 7;
         Severe neonatal; Golgi trafficking defect;
         Wrinkled skin + liver failure + progressive;
         High mortality neonatal;
         Transferrin IEF Type II pattern;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1862–1869)
"""

import random

SEED_BASE = 1862

CDG_GENES = [
    # -- PMM2 -- CDG-Ia -- Most Common CDG ------------------------------------
    {
        "gene": "PMM2",
        "protein": (
            "PMM2 -- 16p13.2 AR -- Phosphomannomutase-2-246aa -- "
            "CDG-Ia-Most-Common-CDG-70pct-of-all-CDGs -- "
            "Transferrin-IEF-Type-I-DIAGNOSTIC -- "
            "Cerebellar-Hypoplasia-PATHOGNOMONIC-MRI -- "
            "Inverted-Nipples-Fat-Pads-Classic-Dysmorphism -- "
            "p.Pro153Leu-70pct-European-Alleles -- "
            "No-Proven-Treatment-Inositol-Trials"
        ),
        "alias": (
            "PMM2 (phosphomannomutase 2); OMIM gene 601785. "
            "CDG-Ia (PMM2-CDG) OMIM 212065. "
            "16p13.2; 246 aa; ~28 kDa; cytoplasm; dimer; "
            "autosomal recessive (biallelic). "
            "FUNCTION: PMM2 catalyses the interconversion of mannose-6-phosphate "
            "and mannose-1-phosphate. Mannose-1-P is the precursor for GDP-mannose, "
            "which is the mannose donor for synthesis of the Glc3Man9GlcNAc2 "
            "lipid-linked oligosaccharide (LLO) that is transferred en bloc to "
            "nascent N-glycoproteins in the ER. "
            "PMM2 loss → GDP-mannose deficiency → hypoglycosylated N-glycoproteins. "
            "TRANSFERRIN IEF — DIAGNOSTIC: "
            "N-glycosylated serum transferrin (normally 4 sialic acids per molecule); "
            "PMM2-CDG: deficient N-glycosylation → Type I pattern: "
            "increased asialo- and monosialo-transferrin bands; "
            "Type I pattern = defect in LLO assembly or transfer (CDG-I group). "
            "CLINICAL FEATURES: "
            "Infantile multisystem: hypotonia + cerebellar ataxia + developmental delay; "
            "Cerebellar hypoplasia on MRI (olivopontocerebellar atrophy) — PATHOGNOMONIC; "
            "Inverted nipples (nipple retraction) + subcutaneous fat pads → "
            "classic dysmorphism; "
            "Coagulopathy: protein C + protein S REDUCED → thrombosis risk; "
            "factor XI also reduced; "
            "Hypogonadism (males): gonadotropin deficiency; "
            "Retinitis pigmentosa (later); "
            "Liver disease (transaminases elevated, hepatomegaly); "
            "Pericardial effusion + cardiomyopathy (minority). "
            "NATURAL HISTORY: "
            "Severe infantile period: hypotonia, feeding difficulties, liver failure → "
            "~20% die in first year; "
            "Stabilise in childhood: ataxia prominent, intellectual disability (mild-severe); "
            "Adult: ataxic gait, peripheral neuropathy, retinopathy, short stature. "
            "MOST COMMON VARIANTS: "
            "p.Pro153Leu (c.458C>T): ~70% of European PMM2-CDG alleles; "
            "p.Phe119Leu: ~10%; "
            "homozygous p.Pro153Leu = not reported (presumably lethal); "
            "only compound heterozygotes observed. "
            "TREATMENT: "
            "No proven specific therapy; "
            "Inositol supplementation (trials: POLARIS study) — some improvement in cerebellar function; "
            "Supportive: physiotherapy, feeding support, anti-seizure medications; "
            "Monitor protein C/S before surgery/procedures — thrombosis or bleeding risk; "
            "Contraceptive pill CAUTION — further reduces protein C/S → thrombosis."
        ),
        "age_of_onset": "Neonatal/infantile",
        "inheritance": "Autosomal recessive (biallelic PMM2 mutations)",
        "locus": "16p13.2",
        "protein_size": "246 aa",
        "key_biomarker": "Transferrin IEF Type I pattern; low protein C + protein S; elevated transaminases",
        "pathognomonic": "Cerebellar hypoplasia MRI + inverted nipples + fat pads + Type I transferrin IEF = PMM2-CDG",
        "treatment": "No proven specific treatment; inositol trials; monitor coagulopathy; supportive",
        "critical_flags": [
            "PROTEIN-C-AND-S-REDUCED — thrombosis risk; check before surgery; OCP contraindicated",
            "CEREBELLAR-HYPOPLASIA-PATHOGNOMONIC — MRI finding; olivopontocerebellar atrophy",
            "INVERTED-NIPPLES-FAT-PADS-CLASSIC — dysmorphic features present in infancy",
            "p.Pro153Leu-70pct-EUROPEAN — most common PMM2 allele; never homozygous (lethal)",
            "TRANSFERRIN-IEF-TYPE-I-PATTERN — diagnostic; deficient N-glycosylation",
            "INOSITOL-TRIALS-EXPERIMENTAL — POLARIS study; not yet standard of care",
            "CDG-SCREEN-MANDATORY-UNEXPLAINED-ATAXIA — PMM2-CDG most likely IEM cause",
            "NO-OCP-FURTHER-REDUCES-PROTEIN-C-S — thrombosis; alternative contraception mandatory",
        ],
    },
    # -- MPI -- CDG-Ib -- TREATABLE, NO neurology ----------------------------
    {
        "gene": "MPI",
        "protein": (
            "MPI -- 15q24.1 AR -- Mannose-Phosphate-Isomerase-423aa -- "
            "CDG-Ib-NO-NEUROLOGY-KEY-DDx-PMM2 -- "
            "Protein-Losing-Enteropathy-Hepatopathy-Coagulopathy-Hypoglycaemia -- "
            "Transferrin-IEF-Type-I -- "
            "TREATABLE-Oral-D-Mannose-1gkgday-4-doses-CURATIVE -- "
            "R219Q-Founder-Mutation"
        ),
        "alias": (
            "MPI (mannose phosphate isomerase); OMIM gene 154550. "
            "CDG-Ib (MPI-CDG) OMIM 602579. "
            "15q24.1; 423 aa; ~47 kDa; cytoplasm; homodimer; "
            "autosomal recessive (biallelic). "
            "FUNCTION: MPI catalyses the interconversion of mannose-6-phosphate "
            "and fructose-6-phosphate (entry point for mannose into the N-glycosylation pathway). "
            "MPI loss → mannose-6-phosphate deficiency → same downstream GDP-mannose deficiency "
            "as PMM2-CDG → hypoglycosylation of N-glycoproteins. "
            "CRITICAL DISTINGUISHING FEATURE — NO NEUROLOGY: "
            "Unlike PMM2-CDG (dominant presentation = neurological disorder), "
            "MPI-CDG has NO significant central nervous system involvement. "
            "MPI is expressed in liver+intestine; mannose can be salvaged from dietary intake "
            "and endogenous glucose via hexose interconversion in other tissues; "
            "CNS neurons not dependent on MPI for mannose supply. "
            "CLINICAL FEATURES: "
            "Protein-losing enteropathy (diarrhoea, oedema, hypoalbuminaemia); "
            "Hepatopathy (hepatomegaly, elevated transaminases, cholestasis); "
            "Coagulopathy (protein C/S, factor XI reduced — same as PMM2 but worse hepatic); "
            "Recurrent hypoglycaemia; "
            "Hyperinsulinism in some patients; "
            "Intestinal polyposis in some; "
            "Thrombotic episodes (portal vein, hepatic vein) — life-threatening. "
            "BIOMARKER: "
            "Transferrin IEF: Type I pattern (same as PMM2-CDG); "
            "distinguishing feature: clinical (no neurological involvement) not biochemical. "
            "TREATMENT — ORAL D-MANNOSE: "
            "D-mannose supplementation 1 g/kg/day (in 4-6 divided doses): "
            "bypasses MPI block → provides mannose-6-P via hexokinase; "
            "NORMALISES transferrin IEF, protein C/S, liver enzymes; "
            "PREVENTS hypoglycaemic episodes and enteropathy; "
            "must be LIFELONG — stopping leads to relapse; "
            "mannose is safe at therapeutic doses; excessive dose → hypermannosaemia (avoid). "
            "FOUNDER: R219Q (Arg219Gln) mutation — common in European cohorts. "
            "PROGNOSIS with treatment: near-normal if started early; "
            "untreated: progressive liver failure, thrombosis, death in childhood."
        ),
        "age_of_onset": "Infantile (first year)",
        "inheritance": "Autosomal recessive (biallelic MPI mutations)",
        "locus": "15q24.1",
        "protein_size": "423 aa",
        "key_biomarker": "Transferrin IEF Type I (same as PMM2); NO neurological features; low protein C/S; hypoalbuminaemia",
        "pathognomonic": "CDG Type I + NO neurology + protein-losing enteropathy + treatable with D-mannose = MPI-CDG",
        "treatment": "Oral D-mannose 1 g/kg/day (4-6 divided doses) LIFELONG — curative",
        "critical_flags": [
            "NO-NEUROLOGY-KEY-DDx-FROM-PMM2 — absence of cerebellar/neurological disease is DIAGNOSTIC",
            "D-MANNOSE-CURATIVE-1gkgday-4doses — start immediately on diagnosis; lifelong",
            "PROTEIN-C-S-REDUCED-THROMBOSIS — portal/hepatic vein thrombosis life-threatening",
            "PROTEIN-LOSING-ENTEROPATHY — hypoalbuminaemia + diarrhoea + oedema",
            "LIFELONG-MANNOSE-MANDATORY — stopping causes relapse; do not discontinue",
            "TRANSFERRIN-IEF-TYPE-I-SAME-AS-PMM2 — clinical not biochemical distinction from PMM2",
            "HYPERINSULINISM-SOME-PATIENTS — hypoglycaemia may resemble HH on initial workup",
            "THROMBOTIC-EPISODES-WITHOUT-TREATMENT — anticoagulation secondary to mannose",
        ],
    },
    # -- ALG6 -- CDG-Ic -- 2nd most common N-glycosylation CDG ---------------
    {
        "gene": "ALG6",
        "protein": (
            "ALG6 -- 1p31.3 AR -- Dolichyl-PP-Man9GlcNAc2-alpha-1,3-glucosyltransferase-507aa -- "
            "CDG-Ic-2nd-Most-Common-N-Glycosylation-CDG -- "
            "Milder-Than-PMM2-Cerebellar-Ataxia-Intellectual-Disability -- "
            "Transferrin-IEF-Type-I -- "
            "Seizures-Less-Severe -- "
            "No-Inverted-Nipples-Fat-Pads-Unlike-PMM2"
        ),
        "alias": (
            "ALG6 (alpha-1,3-glucosyltransferase); OMIM gene 604566. "
            "CDG-Ic (ALG6-CDG) OMIM 603147. "
            "1p31.3; 507 aa; ~57 kDa; ER membrane; "
            "autosomal recessive. "
            "FUNCTION: ALG6 adds the first glucose residue to Man9GlcNAc2-PP-dolichol "
            "(Glc1Man9GlcNAc2-PP-Dol), a step in the sequential assembly of the "
            "lipid-linked oligosaccharide (LLO) in the ER. "
            "The LLO must reach Glc3Man9GlcNAc2 for efficient transfer to protein by "
            "oligosaccharyltransferase (OST); ALG6 mutations → truncated LLO → "
            "hypoglycosylation of N-glycoproteins (Type I CDG). "
            "CLINICAL FEATURES (milder than PMM2-CDG): "
            "Cerebellar ataxia (moderate, non-progressive or slowly progressive); "
            "Intellectual disability (mild-moderate); "
            "Hypotonia; "
            "Seizures (less frequent and severe than PMM2-CDG); "
            "NO significant coagulopathy; "
            "Liver disease: mild or absent; "
            "NO inverted nipples/fat pads (distinguishes from PMM2-CDG); "
            "Strabismus common; "
            "Some patients survive to adulthood with moderate disability. "
            "BIOMARKER: "
            "Transferrin IEF: Type I pattern (same as PMM2-CDG); "
            "distinguish from PMM2 by: fibroblast LLO analysis (accumulation of Man9GlcNAc2-PP-Dol); "
            "PMM2 enzymatic assay (normal in ALG6-CDG); "
            "WES/gene panel. "
            "COMMON VARIANTS: "
            "Ala333Val (c.998C>T): most frequent ALG6-CDG allele; "
            "several other missense mutations across the gene. "
            "TREATMENT: "
            "No specific treatment; "
            "Supportive: physiotherapy, seizure management, educational support."
        ),
        "age_of_onset": "Infantile",
        "inheritance": "Autosomal recessive (biallelic ALG6 mutations)",
        "locus": "1p31.3",
        "protein_size": "507 aa",
        "key_biomarker": "Transferrin IEF Type I; LLO analysis: Man9GlcNAc2 accumulation in fibroblasts",
        "pathognomonic": "Type I CDG + milder phenotype than PMM2 + no inverted nipples + LLO analysis = ALG6-CDG",
        "treatment": "No specific treatment; supportive (physiotherapy, AEDs if needed)",
        "critical_flags": [
            "NO-INVERTED-NIPPLES-FAT-PADS-DISTINGUISHES-FROM-PMM2 — milder phenotype marker",
            "TRANSFERRIN-IEF-TYPE-I-SAME-AS-PMM2 — cannot distinguish PMM2 vs ALG6 by IEF alone",
            "LLO-ANALYSIS-FIBROBLASTS-KEY — Man9GlcNAc2 accumulation distinguishes ALG6 from PMM2",
            "PMM2-ENZYME-ASSAY-NORMAL — rule out PMM2 first as it is 10x more common",
            "NO-SPECIFIC-TREATMENT-YET — unlike MPI/PGM1/SLC35C1 (no treatable CDG)",
            "MILDER-NATURAL-HISTORY-THAN-PMM2 — survival to adulthood common",
            "SEIZURES-LESS-SEVERE-THAN-PMM2 — important prognostic distinction",
            "WES-PANEL-REQUIRED — cannot distinguish ALG6 vs other Type I CDGs on IEF alone",
        ],
    },
    # -- PGM1 -- CDG-PGM1 -- TREATABLE, bifid uvula pathognomonic ------------
    {
        "gene": "PGM1",
        "protein": (
            "PGM1 -- 1p31.3 AR -- Phosphoglucomutase-1-562aa -- "
            "CDG-PGM1-Bifid-Uvula-PATHOGNOMONIC -- "
            "Dilated-Cardiomyopathy-Hepatopathy-Exercise-Intolerance-Elevated-CK -- "
            "TREATABLE-Galactose-0.5gkgday-Significant-Improvement -- "
            "Mixed-Transferrin-IEF-Type-I-AND-II-PATHOGNOMONIC-MIXED-Pattern"
        ),
        "alias": (
            "PGM1 (phosphoglucomutase 1); OMIM gene 171900. "
            "PGM1-CDG (CDG type PGM1) OMIM 614921. "
            "1p31.3 (NOT same as ALG6, different region); 562 aa; ~63 kDa; cytoplasm; "
            "autosomal recessive. "
            "FUNCTION: PGM1 catalyses the interconversion of glucose-1-phosphate "
            "and glucose-6-phosphate, a critical step in glycogen metabolism AND "
            "in UDP-glucose (and UDP-galactose) synthesis for glycosylation. "
            "PGM1 loss → reduced UDP-glucose/galactose → "
            "deficiency in BOTH N-glycosylation AND O-glycosylation + glycogen metabolism. "
            "UNIQUE FEATURE — MIXED TYPE I/II TRANSFERRIN PATTERN: "
            "PGM1-CDG shows a MIXED Type I + Type II transferrin IEF pattern, "
            "distinguishing it from pure Type I CDGs (PMM2, MPI, ALG6) and pure Type II; "
            "this mixed pattern is a DIAGNOSTIC CLUE for PGM1-CDG. "
            "CLINICAL FEATURES: "
            "Bifid uvula (split/forked uvula) — PATHOGNOMONIC; present from birth; "
            "Dilated cardiomyopathy (DCM) — onset childhood; potentially fatal; "
            "Hepatopathy (elevated transaminases, hepatomegaly); "
            "Hypoglycaemia (fasting); "
            "Exercise intolerance with rhabdomyolysis (elevated CK); "
            "Intellectual disability (mild-moderate); "
            "Short stature; "
            "Myopathy (proximal weakness); "
            "Cleft palate in some (bifid uvula spectrum). "
            "TREATMENT — GALACTOSE: "
            "Oral galactose supplementation (0.5 g/kg/day in divided doses): "
            "galactose → galactose-1-P → UDP-galactose (bypasses PGM1 block partially); "
            "improves: liver enzymes, hypoglycaemia, exercise tolerance, CK; "
            "cardiomyopathy improvement documented; "
            "Must be lifelong; galactose also improves transferrin glycosylation. "
            "EXERCISE PROTOCOL: "
            "Carbohydrate loading before exercise (sucrose); "
            "avoid prolonged fasting; "
            "similar approach to GSD type V (McArdle). "
            "COMMON VARIANTS: "
            "Various missense mutations; many affect the active site or substrate binding."
        ),
        "age_of_onset": "Neonatal/infantile (bifid uvula from birth; cardiac within first years)",
        "inheritance": "Autosomal recessive (biallelic PGM1 mutations)",
        "locus": "1p31.3",
        "protein_size": "562 aa",
        "key_biomarker": "Transferrin IEF mixed Type I/II pattern; elevated CK; bifid uvula on examination",
        "pathognomonic": "Bifid uvula + dilated cardiomyopathy + mixed transferrin IEF pattern + hepatopathy = PGM1-CDG",
        "treatment": "Oral galactose 0.5 g/kg/day (lifelong); carbohydrate load pre-exercise; cardiac surveillance",
        "critical_flags": [
            "BIFID-UVULA-PATHOGNOMONIC — examine uvula in ALL CDG patients; present from birth",
            "DILATED-CARDIOMYOPATHY-POTENTIALLY-FATAL — cardiac monitoring mandatory q6 months",
            "MIXED-TYPE-I-AND-II-TRANSFERRIN-PATTERN — cannot be PMM2/MPI/ALG6 (pure Type I)",
            "GALACTOSE-0.5gkgday-TREATABLE — start immediately; improves cardiomyopathy, liver, hypoglycaemia",
            "EXERCISE-RHABDOMYOLYSIS — CK very elevated post-exercise; carbohydrate load before exercise",
            "FASTING-HYPOGLYCAEMIA-DANGEROUS — no prolonged fasting; regular feeds mandatory",
            "GALACTOSE-LIFELONG-MANDATORY — stopping causes relapse of cardiomyopathy/hepatopathy",
            "NOT-SAME-LOCUS-AS-ALG6 — both on 1p31.3 region but different genes; do NOT confuse",
        ],
    },
    # -- SLC35A2 -- CDG-IIm -- X-linked, de novo in females ------------------
    {
        "gene": "SLC35A2",
        "protein": (
            "SLC35A2 -- Xp11.23 X-linked-De-Novo-Dominant-Females -- "
            "UDP-Galactose-Golgi-Transporter-396aa -- "
            "CDG-IIm-First-X-Linked-CDG -- "
            "De-Novo-Heterozygous-Females-Males-Non-Viable -- "
            "Epilepsy-Intellectual-Disability-Dysmorphism -- "
            "Transferrin-IEF-Type-II -- "
            "Galactose-Supplementation-Emerging"
        ),
        "alias": (
            "SLC35A2 (solute carrier family 35 member A2); OMIM gene 314375. "
            "SLC35A2-CDG (CDG-IIm) OMIM 300896. "
            "Xp11.23; 396 aa; ~42 kDa; Golgi membrane; UDP-galactose transporter; "
            "X-linked (de novo heterozygous mutations in females). "
            "FUNCTION: SLC35A2 (UDP-galactose transporter, UGT) transports "
            "UDP-galactose from cytoplasm into the Golgi lumen. "
            "UDP-galactose is the galactose donor for Golgi galactosyltransferases "
            "that modify both N-glycans and O-glycans (galactosylation step). "
            "SLC35A2 loss → Golgi UDP-galactose deficiency → "
            "hypogalactosylation of N-glycans AND O-glycans "
            "→ Type II CDG pattern (processing defect, not LLO assembly defect). "
            "X-LINKED GENETICS: "
            "Hemizygous males (normal allele absent): typically non-viable "
            "(severe deficiency in galactosylation lethal in utero); "
            "reported mainly in females with DE NOVO heterozygous mutations; "
            "germline or somatic mosaic mutations in affected females; "
            "some males with hypomorphic mutations survive — very rare. "
            "CLINICAL FEATURES (in heterozygous females): "
            "Early-onset epilepsy (infantile spasms, multiple seizure types); "
            "Intellectual disability (moderate-severe); "
            "Dysmorphic features (widely spaced teeth, coarse facies); "
            "Hypotonia; "
            "Behavioural problems; "
            "Some patients: progressive course. "
            "BIOMARKER: "
            "Transferrin IEF: Type II pattern (galactosylation defect); "
            "apolipoprotein CIII isoelectric focusing (O-glycan defect); "
            "Golgi glycan analysis in fibroblasts. "
            "TREATMENT: "
            "Galactose supplementation (emerging evidence): "
            "increases UDP-galactose availability in Golgi; "
            "case reports show improvement in seizures/EEG; "
            "not yet standard of care; "
            "Anti-seizure medications for epilepsy (ketogenic diet tried in some)."
        ),
        "age_of_onset": "Infantile (epilepsy onset first year)",
        "inheritance": "X-linked de novo dominant (heterozygous females); males typically non-viable",
        "locus": "Xp11.23",
        "protein_size": "396 aa",
        "key_biomarker": "Transferrin IEF Type II; apolipoprotein CIII IEF (O-glycan defect); fibroblast Golgi glycan analysis",
        "pathognomonic": "X-linked CDG + de novo female + early epilepsy + Type II transferrin IEF = SLC35A2-CDG",
        "treatment": "Galactose supplementation (emerging); anti-epileptic drugs; ketogenic diet trialled",
        "critical_flags": [
            "DE-NOVO-MUTATION-IN-FEMALES — not inherited from unaffected parent; confirm by parental testing",
            "MALES-HEMIZYGOUS-TYPICALLY-NON-VIABLE — severe; explain to families",
            "TRANSFERRIN-IEF-TYPE-II — Golgi processing defect; not LLO assembly defect",
            "GALACTOSE-SUPPLEMENTATION-EMERGING — case reports positive; not yet standard",
            "EARLY-EPILEPSY-PROMINENT — infantile spasms; ketogenic diet may help",
            "APOLIPOPROTEIN-CIII-IEF-O-GLYCAN-DEFECT — document O-glycosylation defect separately",
            "SOMATIC-MOSAIC-SOME-PATIENTS — mosaic mutations → milder phenotype",
            "WES-WITH-X-CHROMOSOME-COVERAGE — Xp11.23 may need specific confirmation on standard panels",
        ],
    },
    # -- SLC35C1 -- CDG-IIc / LAD-II -- Bombay blood group pathognomonic -----
    {
        "gene": "SLC35C1",
        "protein": (
            "SLC35C1 -- 11p11.2 AR -- GDP-Fucose-Golgi-Transporter-364aa -- "
            "CDG-IIc-Leukocyte-Adhesion-Deficiency-Type-II-LAD-II -- "
            "Bombay-Blood-Group-H-antigen-Absent-PATHOGNOMONIC -- "
            "Sialyl-LewisX-Absent-Leukocytes-Cannot-Roll-Severe-Infections -- "
            "TREATABLE-Oral-L-Fucose-500mgkgday -- "
            "R147C-Most-Common-Mutation"
        ),
        "alias": (
            "SLC35C1 (solute carrier family 35 member C1); OMIM gene 605881. "
            "CDG-IIc / Leukocyte Adhesion Deficiency type II (LAD-II) OMIM 266265. "
            "11p11.2; 364 aa; ~39 kDa; Golgi membrane; GDP-fucose transporter; "
            "autosomal recessive (biallelic). "
            "FUNCTION: SLC35C1 (GDP-fucose transporter, FUCT1) transports "
            "GDP-fucose from cytoplasm into Golgi lumen. "
            "GDP-fucose is the fucose donor for fucosyltransferases that add "
            "fucose to N-glycans (core fucosylation) and to sialyl-Lewis X "
            "(a tetrasaccharide on leukocyte surface important for rolling on endothelium). "
            "SLC35C1 loss → Golgi GDP-fucose deficiency → absent fucosylation → "
            "absent sialyl-LewisX (CD15s) on neutrophils/monocytes → "
            "leukocytes cannot bind E/P-selectin → cannot roll and adhere → "
            "recurrent BACTERIAL INFECTIONS (Leukocyte Adhesion Deficiency type II). "
            "BLOOD GROUP PATHOGNOMONIC: "
            "H-antigen (blood group H, precursor to A/B/O antigens) requires fucosylation; "
            "SLC35C1 deficiency → absent H-antigen → Bombay blood group (Oh phenotype); "
            "Bombay individuals are incompatible with ALL ABO blood types; "
            "anti-H antibody can cause haemolytic transfusion reaction if given ABO-compatible blood. "
            "CLINICAL FEATURES: "
            "Recurrent severe bacterial infections (Staphylococcus, gram-negative bacilli); "
            "Leukocytosis (markedly elevated WBC — leukocytes cannot marginate/emigrate); "
            "Intellectual disability (moderate); "
            "Short stature; "
            "Bombay blood group (H-antigen absent). "
            "TREATMENT — ORAL L-FUCOSE: "
            "L-fucose supplementation 500 mg/kg/day: "
            "bypasses transporter defect (fucose salvage pathway bypasses Golgi transport); "
            "IMPROVES infections: restores some sialyl-LewisX; reduces hospitalisation; "
            "PARTIALLY improves intellectual function (some patients); "
            "Start early for maximal benefit; lifelong. "
            "BLOOD BANK WARNING: "
            "Bombay blood group patients CANNOT receive standard ABO-typed blood; "
            "require autologous blood bank or Bombay-compatible donors; "
            "MUST be flagged in blood bank records BEFORE any surgery."
        ),
        "age_of_onset": "Neonatal/infantile (recurrent infections from birth; Bombay blood group from birth)",
        "inheritance": "Autosomal recessive (biallelic SLC35C1 mutations)",
        "locus": "11p11.2",
        "protein_size": "364 aa",
        "key_biomarker": "Absent sialyl-LewisX (CD15s) on neutrophils by flow cytometry; Bombay blood group; leukocytosis",
        "pathognomonic": "Bombay blood group + recurrent bacterial infections + absent sialyl-LewisX + leukocytosis = CDG-IIc/LAD-II",
        "treatment": "Oral L-fucose 500 mg/kg/day (lifelong); Bombay blood bank registration MANDATORY; prophylactic antibiotics",
        "critical_flags": [
            "BOMBAY-BLOOD-GROUP-PATHOGNOMONIC — H-antigen absent; incompatible with ALL ABO blood types",
            "BLOOD-BANK-MANDATORY-FLAG — Bombay patients CANNOT receive standard ABO blood; must pre-register",
            "L-FUCOSE-500mgkgday-TREATABLE — start immediately; reduces infections; partially improves ID",
            "LEUKOCYTOSIS-WITHOUT-INFECTION — chronically elevated WBC (leukocytes cannot emigrate)",
            "FLOW-CYTOMETRY-CD15s-SIALYL-LEWISX-ABSENT — diagnostic test for LAD-II",
            "ANTI-H-HAEMOLYTIC-RISK — anti-H antibody → severe haemolysis from ABO-compatible transfusion",
            "RECURRENT-BACTERIAL-INFECTIONS-FROM-BIRTH — similar to LAD-I but different genetics",
            "LAD-II-NOT-LAD-I — LAD-I=CD18 deficiency (ITGB2); LAD-II=fucose transporter; different genes",
        ],
    },
    # -- DOLK -- CDG-Im -- Dilated cardiomyopathy + ichthyosis ---------------
    {
        "gene": "DOLK",
        "protein": (
            "DOLK -- 9q34.11 AR -- Dolichol-Kinase-538aa -- "
            "CDG-Im-Dilated-Cardiomyopathy-DOMINANT-Feature -- "
            "Ichthyosis-Hepatopathy-Neurological -- "
            "Transferrin-IEF-Type-I -- "
            "Dolichyl-P-Synthesis-ER-Membrane -- "
            "G301R-Most-Common-Mutation-Irish-Traveller-Founder"
        ),
        "alias": (
            "DOLK (dolichol kinase); OMIM gene 610746. "
            "CDG-Im (DOLK-CDG) OMIM 610768. "
            "9q34.11; 538 aa; ~60 kDa; ER membrane; "
            "autosomal recessive. "
            "FUNCTION: DOLK phosphorylates dolichol to dolichyl-phosphate (dolichyl-P). "
            "Dolichyl-P is the essential lipid carrier for LLO assembly: "
            "Man-P-Dol (mannose donor for LLO luminal steps); "
            "Glc-P-Dol (glucose donor for LLO glucosylation); "
            "the entire LLO (Glc3Man9GlcNAc2) is assembled on dolichyl-PP. "
            "DOLK loss → dolichyl-P deficiency → severely truncated LLO → "
            "profound N-glycoprotein hypoglycosylation (Type I CDG). "
            "DISTINCT CLINICAL FEATURE — CARDIAC DOMINANT: "
            "Dilated cardiomyopathy (DCM) is often the PRESENTING and DOMINANT feature, "
            "distinguishing DOLK-CDG from other Type I CDGs where neurology dominates. "
            "CLINICAL FEATURES: "
            "Dilated cardiomyopathy (onset infantile-childhood; can be fatal); "
            "Ichthyosis (congenital; dry, scaly skin from birth — key clinical marker); "
            "Hepatopathy (elevated liver enzymes); "
            "Neurological (intellectual disability, seizures — less prominent than cardiac); "
            "Hypotonia; "
            "Short stature. "
            "BIOMARKER: "
            "Transferrin IEF: Type I pattern; "
            "dolichol measurement (fibroblasts, urine): elevated free dolichol; "
            "LLO analysis: severely truncated (Man5GlcNAc2-PP-Dol or less). "
            "FOUNDER MUTATION: "
            "G301R (c.901G>A) — Irish Traveller founder mutation; "
            "homozygous G301R relatively frequent in Irish Traveller community. "
            "TREATMENT: "
            "No specific treatment; "
            "Cardiac: ACE inhibitors, beta-blockers, heart transplant considered; "
            "Skin: emollients for ichthyosis; "
            "Dolichyl-P supplementation experimental."
        ),
        "age_of_onset": "Neonatal/infantile",
        "inheritance": "Autosomal recessive (biallelic DOLK mutations)",
        "locus": "9q34.11",
        "protein_size": "538 aa",
        "key_biomarker": "Transferrin IEF Type I; dilated cardiomyopathy; ichthyosis; elevated free dolichol in urine",
        "pathognomonic": "CDG Type I + DCM + ichthyosis = DOLK-CDG; cardiac dominant distinguishes from PMM2",
        "treatment": "No specific treatment; cardiac management (ACEi, beta-blockers, transplant); emollients for ichthyosis",
        "critical_flags": [
            "DILATED-CARDIOMYOPATHY-DOMINANT-FEATURE — cardiac presentation may precede neurological",
            "ICHTHYOSIS-KEY-CLINICAL-MARKER — dry scaly skin from birth; guides CDG subtype diagnosis",
            "CARDIAC-TRANSPLANT-CONSIDERED-SEVERE-DCM — end-stage cardiomyopathy may require transplant",
            "G301R-IRISH-TRAVELLER-FOUNDER — community prevalence; cascade testing important",
            "DOLICHOL-ELEVATED-URINE-FIBROBLASTS — biochemical marker for DOLK and dolichol pathway defects",
            "TRANSFERRIN-IEF-TYPE-I-SAME-AS-PMM2 — cannot distinguish by IEF alone; gene panel needed",
            "NO-SPECIFIC-TREATMENT-YET — unlike MPI/PGM1/SLC35C1; supportive cardiac + dermatology",
            "DOLICHYL-P-SUPPLEMENTATION-EXPERIMENTAL — not standard; research context only",
        ],
    },
    # -- COG7 -- CDG-IIe -- Severe neonatal, Golgi COG complex ---------------
    {
        "gene": "COG7",
        "protein": (
            "COG7 -- 16p12.2 AR -- Conserved-Oligomeric-Golgi-Complex-Subunit-7-841aa -- "
            "CDG-IIe-Severe-Neonatal-Golgi-Trafficking-Defect -- "
            "Wrinkled-Skin-Wrinkled-Ear-Helices-PATHOGNOMONIC -- "
            "Liver-Failure-Progressive-High-Neonatal-Mortality -- "
            "Transferrin-IEF-Type-II -- "
            "West-African-Founder"
        ),
        "alias": (
            "COG7 (component of oligomeric Golgi complex 7); OMIM gene 606978. "
            "CDG-IIe (COG7-CDG) OMIM 608779. "
            "16p12.2; 841 aa; ~93 kDa; Golgi membrane (peripheral); "
            "component of Conserved Oligomeric Golgi (COG) complex lobe B; "
            "autosomal recessive. "
            "FUNCTION: The COG complex is an 8-subunit (COG1-8) tethering complex "
            "on cytoplasmic face of Golgi cisternae. COG maintains retrograde "
            "trafficking of Golgi glycosyltransferases (returning enzymes to "
            "their correct cisternae after forward transport). "
            "Without COG7: Golgi glycosyltransferases mis-localised/degraded → "
            "globally deficient Golgi glycosylation → both N-glycans AND O-glycans "
            "poorly processed → Type II CDG (processing defect). "
            "CLINICAL FEATURES (severe): "
            "Severe neonatal onset (usually first days/week of life); "
            "Wrinkled (aged) skin and wrinkled ear helices — PATHOGNOMONIC facies; "
            "Progressive liver failure (hepatomegaly + cholestasis + transaminases); "
            "Hypertonia + seizures; "
            "Microcephaly; "
            "Frequent early death (most patients die within weeks to months of birth); "
            "Survivors: severe intellectual disability, cardiomegaly. "
            "BIOMARKER: "
            "Transferrin IEF: Type II pattern (multiple Golgi glycosyltransferase deficiencies); "
            "apolipoprotein CIII IEF: also abnormal (O-glycan defect); "
            "Golgi morphology: dilated, fragmented cisternae on EM. "
            "FOUNDER MUTATION: "
            "p.Pro332Ser (c.994C>T): West African founder mutation; "
            "multiple patients from West African ancestry described with this variant. "
            "TREATMENT: "
            "No specific treatment; "
            "Supportive: management of liver failure, seizures; "
            "Liver transplant considered in severe hepatic failure (limited data). "
            "PROGNOSIS: "
            "Most patients die in neonatal period or first months; "
            "rare survivors have severe disability."
        ),
        "age_of_onset": "Neonatal (first days/week)",
        "inheritance": "Autosomal recessive (biallelic COG7 mutations)",
        "locus": "16p12.2",
        "protein_size": "841 aa",
        "key_biomarker": "Transferrin IEF Type II; apolipoprotein CIII IEF abnormal; wrinkled aged skin",
        "pathognomonic": "Wrinkled aged skin + wrinkled ear helices + severe neonatal liver failure + Type II CDG = COG7-CDG",
        "treatment": "No specific treatment; supportive (liver failure, seizures); liver transplant limited data; high mortality",
        "critical_flags": [
            "WRINKLED-AGED-SKIN-PATHOGNOMONIC — wrinkled ear helices + wrinkled skin from birth; unique facies",
            "HIGH-NEONATAL-MORTALITY — most die in first weeks/months; urgent family counselling",
            "TYPE-II-TRANSFERRIN-IEF-GOLGI-PROCESSING-DEFECT — multiple glycosyltransferases affected",
            "WEST-AFRICAN-FOUNDER-p.Pro332Ser — community surveillance in West African ancestry patients",
            "NO-SPECIFIC-TREATMENT — unlike MPI/PGM1/SLC35C1; supportive only",
            "LIVER-FAILURE-DOMINANT — cholestasis + transaminases + hepatomegaly; hepatology involvement",
            "COG-COMPLEX-8-SUBUNITS-COG1-8 — other COG genes (COG1-6,8) also cause CDG; similar phenotypes",
            "GOLGI-MORPHOLOGY-EM-FRAGMENTED — Golgi cisternae dilated/fragmented; research diagnostic",
        ],
    },
]


def _make_patients():
    all_pts = []
    for i, gene_data in enumerate(CDG_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        gene = gene_data["gene"]
        for j in range(40):
            # Age profile per gene
            if gene in ("COG7", "DOLK"):
                age = rng.choices(
                    [rng.randint(0, 0), rng.randint(1, 3)],
                    weights=[0.70, 0.30]
                )[0]
            elif gene == "SLC35A2":
                age = rng.choices(
                    [rng.randint(0, 2), rng.randint(3, 12), rng.randint(13, 25)],
                    weights=[0.45, 0.40, 0.15]
                )[0]
            elif gene == "SLC35C1":
                age = rng.choices(
                    [rng.randint(0, 1), rng.randint(2, 8), rng.randint(9, 20)],
                    weights=[0.50, 0.35, 0.15]
                )[0]
            elif gene == "PMM2":
                age = rng.choices(
                    [rng.randint(0, 2), rng.randint(3, 12), rng.randint(13, 35)],
                    weights=[0.40, 0.40, 0.20]
                )[0]
            elif gene == "MPI":
                age = rng.choices(
                    [rng.randint(0, 1), rng.randint(2, 8)],
                    weights=[0.60, 0.40]
                )[0]
            elif gene in ("ALG6", "PGM1"):
                age = rng.choices(
                    [rng.randint(0, 3), rng.randint(4, 15), rng.randint(16, 35)],
                    weights=[0.35, 0.45, 0.20]
                )[0]
            else:
                age = rng.randint(0, 20)

            # Sex
            if gene == "SLC35A2":
                sex = rng.choices(["M", "F"], weights=[0.08, 0.92])[0]  # mainly females
            else:
                sex = rng.choice(["M", "F"])

            # Severity
            if gene in ("COG7",):
                severity = rng.choices(["severe"], weights=[1.0])[0]
            elif gene == "DOLK":
                severity = rng.choices(["severe", "moderate"], weights=[0.65, 0.35])[0]
            elif gene == "SLC35A2":
                severity = rng.choices(["severe", "moderate"], weights=[0.55, 0.45])[0]
            elif gene == "PMM2":
                severity = rng.choices(["severe", "moderate", "mild"], weights=[0.30, 0.50, 0.20])[0]
            elif gene == "MPI":
                severity = rng.choices(["moderate", "severe", "mild"], weights=[0.55, 0.25, 0.20])[0]
            elif gene in ("ALG6", "PGM1", "SLC35C1"):
                severity = rng.choices(["moderate", "mild", "severe"], weights=[0.50, 0.30, 0.20])[0]
            else:
                severity = rng.choices(["severe", "moderate"], weights=[0.60, 0.40])[0]

            # Transferrin IEF pattern
            if gene in ("PMM2", "MPI", "ALG6", "DOLK"):
                transferrin_type = "Type_I"
            elif gene == "PGM1":
                transferrin_type = rng.choices(["Mixed_I_II", "Type_I", "Type_II"], weights=[0.60, 0.25, 0.15])[0]
            else:
                transferrin_type = "Type_II"

            # Neurology
            neurology = gene != "MPI"  # MPI has NO neurology
            if gene == "MPI":
                neurology = rng.random() < 0.05  # very rare in MPI

            # Cerebellar hypoplasia (PMM2 dominant)
            cerebellar_hypoplasia = gene == "PMM2" and rng.random() < 0.85

            # Hepatopathy
            if gene in ("MPI", "DOLK", "COG7", "PGM1"):
                hepatopathy = rng.random() < 0.90
            elif gene == "PMM2":
                hepatopathy = rng.random() < 0.60
            else:
                hepatopathy = rng.random() < 0.30

            # Coagulopathy (protein C/S reduced)
            coagulopathy = gene in ("PMM2", "MPI", "COG7") and rng.random() < 0.80

            # Cardiac (DCM)
            if gene in ("PGM1", "DOLK"):
                cardiac = rng.random() < 0.75
            elif gene == "PMM2":
                cardiac = rng.random() < 0.15
            else:
                cardiac = rng.random() < 0.05

            # Seizures / epilepsy
            if gene == "SLC35A2":
                seizures = rng.random() < 0.90
            elif gene in ("COG7", "PMM2"):
                seizures = rng.random() < 0.70
            elif gene == "MPI":
                seizures = rng.random() < 0.05  # not typical
            else:
                seizures = rng.random() < 0.45

            # Treatable (on specific treatment)
            if gene == "MPI":
                on_treatment = rng.random() < 0.85  # oral D-mannose
            elif gene == "PGM1":
                on_treatment = rng.random() < 0.80  # oral galactose
            elif gene == "SLC35C1":
                on_treatment = rng.random() < 0.75  # oral L-fucose
            else:
                on_treatment = False

            # Pathognomonic features
            bifid_uvula = gene == "PGM1" and rng.random() < 0.85
            bombay_blood_group = gene == "SLC35C1"  # always in SLC35C1
            fat_pads_inverted_nipples = gene == "PMM2" and rng.random() < 0.75
            wrinkled_skin = gene == "COG7" and rng.random() < 0.95
            ichthyosis = gene == "DOLK" and rng.random() < 0.80
            recurrent_infections = gene == "SLC35C1" and rng.random() < 0.95

            all_pts.append({
                "gene": gene,
                "seed": seed,
                "patient_index": j + 1,
                "age": age,
                "sex": sex,
                "severity": severity,
                "transferrin_type": transferrin_type,
                "neurology": neurology,
                "cerebellar_hypoplasia": cerebellar_hypoplasia,
                "hepatopathy": hepatopathy,
                "coagulopathy": coagulopathy,
                "cardiac": cardiac,
                "seizures": seizures,
                "on_treatment": on_treatment,
                "bifid_uvula": bifid_uvula,
                "bombay_blood_group": bombay_blood_group,
                "fat_pads_inverted_nipples": fat_pads_inverted_nipples,
                "wrinkled_skin": wrinkled_skin,
                "ichthyosis": ichthyosis,
                "recurrent_infections": recurrent_infections,
            })
    return all_pts


_PATIENTS = _make_patients()


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------

def overview():
    pts = _PATIENTS
    n = len(pts)
    severe_n = sum(1 for p in pts if p["severity"] == "severe")
    neurology_n = sum(1 for p in pts if p["neurology"])
    hepatopathy_n = sum(1 for p in pts if p["hepatopathy"])
    coagulopathy_n = sum(1 for p in pts if p["coagulopathy"])
    cardiac_n = sum(1 for p in pts if p["cardiac"])
    seizures_n = sum(1 for p in pts if p["seizures"])
    on_treatment_n = sum(1 for p in pts if p["on_treatment"])
    type_i_n = sum(1 for p in pts if "Type_I" in p["transferrin_type"])
    type_ii_n = sum(1 for p in pts if p["transferrin_type"] == "Type_II")
    mixed_n = sum(1 for p in pts if p["transferrin_type"] == "Mixed_I_II")
    bombay_n = sum(1 for p in pts if p["bombay_blood_group"])
    bifid_uvula_n = sum(1 for p in pts if p["bifid_uvula"])
    cerebellar_n = sum(1 for p in pts if p["cerebellar_hypoplasia"])

    gene_summary = []
    for g in CDG_GENES:
        gname = g["gene"]
        gpts = [p for p in pts if p["gene"] == gname]
        ng = len(gpts)
        gene_summary.append({
            "gene": gname,
            "disease_short": {
                "PMM2": "CDG-Ia (most common, 70%+)",
                "MPI": "CDG-Ib (treatable, no neurology)",
                "ALG6": "CDG-Ic (2nd most common N-glycosylation CDG)",
                "PGM1": "CDG-PGM1 (treatable, bifid uvula)",
                "SLC35A2": "CDG-IIm (X-linked de novo, epilepsy)",
                "SLC35C1": "CDG-IIc/LAD-II (treatable, Bombay blood group)",
                "DOLK": "CDG-Im (cardiomyopathy + ichthyosis)",
                "COG7": "CDG-IIe (severe neonatal, high mortality)",
            }.get(gname, "CDG"),
            "inheritance": g["inheritance"],
            "chromosome": g["locus"],
            "protein_size_aa": int(g["protein_size"].replace(" aa", "")),
            "key_finding": {
                "PMM2": "Cerebellar hypoplasia; inverted nipples; protein C/S reduced; no specific Rx",
                "MPI": "NO neurology; treatable with oral D-mannose 1 g/kg/day; protein-losing enteropathy",
                "ALG6": "Milder than PMM2; no inverted nipples; LLO Man9 accumulation",
                "PGM1": "Bifid uvula PATHOGNOMONIC; DCM; treatable with galactose 0.5 g/kg/day",
                "SLC35A2": "X-linked de novo females; early epilepsy; galactose supplementation emerging",
                "SLC35C1": "Bombay blood group; recurrent infections; treatable with L-fucose",
                "DOLK": "DCM dominant; ichthyosis; G301R Irish Traveller founder",
                "COG7": "Severe neonatal; wrinkled aged skin; high mortality; West African founder",
            }.get(gname, ""),
            "management_pearl": {
                "PMM2": "Monitor protein C/S before surgery; avoid OCP; inositol trials experimental",
                "MPI": "Start D-mannose immediately; lifelong; no protein C deficiency if treated",
                "ALG6": "No treatment; PMM2 enzyme assay normal; LLO analysis distinguishes",
                "PGM1": "Galactose 0.5 g/kg/day; carbohydrate load before exercise; cardiac echo q6m",
                "SLC35A2": "Parental testing confirms de novo; galactose supplementation emerging",
                "SLC35C1": "Register Bombay blood group urgently; L-fucose 500 mg/kg/day; AB prophylaxis",
                "DOLK": "Cardiac surveillance mandatory; emollients for ichthyosis; no specific Rx",
                "COG7": "Comfort care discussion; liver transplant limited data; high early mortality",
            }.get(gname, ""),
            "n_patients": ng,
            "severe_n": sum(1 for p in gpts if p["severity"] == "severe"),
            "on_treatment_n": sum(1 for p in gpts if p["on_treatment"]),
        })

    return {
        "atlas": (
            "Hereditary-CDG-Atlas — Complete 8-Gene Congenital Disorders of Glycosylation Atlas"
        ),
        "subtitle": (
            "PMM2-246aa-16p13.2-AR-CDG-Ia-Most-Common-70pct-Cerebellar-Hypoplasia-Inverted-Nipples-No-Rx | "
            "MPI-423aa-15q24.1-AR-CDG-Ib-NO-Neurology-Protein-Losing-Enteropathy-D-Mannose-CURATIVE | "
            "ALG6-507aa-1p31.3-AR-CDG-Ic-2nd-Most-Common-Milder-PMM2-LLO-Analysis | "
            "PGM1-562aa-1p31.3-AR-CDG-PGM1-Bifid-Uvula-PATHOGNOMONIC-DCM-Galactose-TREATABLE | "
            "SLC35A2-396aa-Xp11.23-XL-De-Novo-Female-CDG-IIm-Epilepsy-Galactose-Emerging | "
            "SLC35C1-364aa-11p11.2-AR-CDG-IIc-LAD-II-Bombay-Blood-Group-L-Fucose-TREATABLE | "
            "DOLK-538aa-9q34.11-AR-CDG-Im-DCM-Dominant-Ichthyosis-G301R-Irish-Traveller | "
            "COG7-841aa-16p12.2-AR-CDG-IIe-Severe-Neonatal-Wrinkled-Skin-High-Mortality — "
            "320 Patients (8×40, Seeds 1862–1869)"
        ),
        "total_patients": n,
        "seed_range": "1862–1869",
        "aggregate_stats": {
            "genes_covered": len(CDG_GENES),
            "ar_genes": 6,
            "x_linked_genes": 1,
            "ad_de_novo_genes": 1,
            "patients_per_gene": 40,
            "neurology_pct": round(100 * neurology_n / n, 1),
            "hepatopathy_pct": round(100 * hepatopathy_n / n, 1),
            "coagulopathy_pct": round(100 * coagulopathy_n / n, 1),
            "cardiac_pct": round(100 * cardiac_n / n, 1),
            "seizures_pct": round(100 * seizures_n / n, 1),
            "severity_severe_pct": round(100 * severe_n / n, 1),
            "on_specific_treatment_pct": round(100 * on_treatment_n / n, 1),
            "transferrin_type_i_pct": round(100 * type_i_n / n, 1),
            "transferrin_type_ii_pct": round(100 * type_ii_n / n, 1),
            "transferrin_mixed_pct": round(100 * mixed_n / n, 1),
        },
        "gene_summary": gene_summary,
        "top_alerts": [
            "PROTEIN-C-AND-S-REDUCED-PMM2 — thrombosis risk; check before surgery; OCP contraindicated",
            "CEREBELLAR-HYPOPLASIA-PMM2-PATHOGNOMONIC — MRI finding; olivopontocerebellar atrophy",
            "D-MANNOSE-CURATIVE-MPI — start immediately on diagnosis; 1 g/kg/day 4 doses; lifelong",
            "MPI-NO-NEUROLOGY-KEY-DDx — absence of cerebellar/neurological = MPI-CDG diagnosis clue",
            "BIFID-UVULA-PGM1-PATHOGNOMONIC — examine uvula in ALL suspected CDG patients",
            "GALACTOSE-PGM1-TREATABLE — 0.5 g/kg/day; improves DCM, liver, hypoglycaemia",
            "BOMBAY-BLOOD-GROUP-SLC35C1-PATHOGNOMONIC — must register in blood bank BEFORE any surgery",
            "L-FUCOSE-SLC35C1-TREATABLE — 500 mg/kg/day; reduces infections; partial ID improvement",
            "DOLK-DCM-DOMINANT — cardiac presentation may precede neurological in DOLK-CDG",
            "COG7-HIGH-NEONATAL-MORTALITY — wrinkled aged skin + liver failure; comfort care discussion",
            "TRANSFERRIN-IEF-MANDATORY-SCREEN — cannot diagnose CDG without IEF first",
            "TYPE-II-PATTERN-MEANS-GOLGI-PROCESSING-DEFECT — SLC35A2/SLC35C1/COG7",
            "MIXED-TYPE-I-II-PATTERN-UNIQUE-TO-PGM1 — pathognomonic pattern combination",
            "SLC35A2-DE-NOVO-CONFIRM-PARENTAL-TESTING — confirm neither parent carries variant",
        ],
        "critical_treatment_alerts": [
            "D-MANNOSE-CURATIVE-MPI — 1 g/kg/day in 4 doses LIFELONG; stopping = relapse",
            "GALACTOSE-PGM1-LIFELONG — 0.5 g/kg/day; do NOT stop even if asymptomatic",
            "L-FUCOSE-SLC35C1-LIFELONG — 500 mg/kg/day; reduces hospitalisations",
            "BOMBAY-BLOOD-BANK-REGISTRATION-MANDATORY — before ANY surgery or transfusion",
            "PROTEIN-C-S-PMM2-PRE-OP — check before all surgery; not just haematology patients",
            "OCP-CONTRAINDICATED-PMM2 — further reduces protein C/S → thrombosis",
            "DCM-CARDIAC-MONITORING-PGM1-DOLK — echo q6 months; cardiology co-management",
            "CARBOHYDRATE-LOAD-BEFORE-EXERCISE-PGM1 — prevents rhabdomyolysis/hypoglycaemia",
            "FASTING-DANGEROUS-PMM2-MPI-PGM1 — avoid; hypoglycaemia risk in multiple CDGs",
            "GALACTOSE-EMERGING-SLC35A2 — not yet standard; refer to CDG specialist centre",
        ],
    }


def breakdown():
    pts = _PATIENTS
    by_gene = {}
    for g in CDG_GENES:
        gname = g["gene"]
        gpts = [p for p in pts if p["gene"] == gname]
        n = len(gpts)
        by_gene[gname] = {
            "gene": gname,
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "n_patients": n,
            "severe_n": sum(1 for p in gpts if p["severity"] == "severe"),
            "severe_pct": round(100 * sum(1 for p in gpts if p["severity"] == "severe") / n, 1),
            "neurology_n": sum(1 for p in gpts if p["neurology"]),
            "hepatopathy_n": sum(1 for p in gpts if p["hepatopathy"]),
            "coagulopathy_n": sum(1 for p in gpts if p["coagulopathy"]),
            "cardiac_n": sum(1 for p in gpts if p["cardiac"]),
            "seizures_n": sum(1 for p in gpts if p["seizures"]),
            "on_treatment_n": sum(1 for p in gpts if p["on_treatment"]),
            "bifid_uvula_n": sum(1 for p in gpts if p["bifid_uvula"]),
            "bombay_blood_group_n": sum(1 for p in gpts if p["bombay_blood_group"]),
            "cerebellar_hypoplasia_n": sum(1 for p in gpts if p["cerebellar_hypoplasia"]),
            "fat_pads_inverted_nipples_n": sum(1 for p in gpts if p["fat_pads_inverted_nipples"]),
            "wrinkled_skin_n": sum(1 for p in gpts if p["wrinkled_skin"]),
            "ichthyosis_n": sum(1 for p in gpts if p["ichthyosis"]),
            "recurrent_infections_n": sum(1 for p in gpts if p["recurrent_infections"]),
            "transferrin_type_i_n": sum(1 for p in gpts if "Type_I" in p["transferrin_type"]),
            "transferrin_type_ii_n": sum(1 for p in gpts if p["transferrin_type"] == "Type_II"),
            "transferrin_mixed_n": sum(1 for p in gpts if p["transferrin_type"] == "Mixed_I_II"),
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "seed": SEED_BASE + CDG_GENES.index(g),
        }
    return {
        "atlas": "Hereditary-CDG-Atlas",
        "breakdown_by_gene": by_gene,
        "total_genes": len(CDG_GENES),
        "seed_range": "1862–1869",
    }


def definitions():
    defs = {}

    defs["CDG — What are Congenital Disorders of Glycosylation?"] = (
        "Congenital Disorders of Glycosylation (CDG) are a large group of inherited metabolic diseases "
        "affecting the synthesis, processing, or attachment of glycans (sugar chains) to proteins and lipids. "
        "GLYCOSYLATION OVERVIEW: "
        "~50-70% of all human proteins are glycosylated. "
        "Glycans perform critical functions: protein folding (quality control in ER); "
        "cell-cell recognition and adhesion; ligand binding; protease protection; sorting signals. "
        "TWO MAIN TYPES OF GLYCOSYLATION: "
        "1. N-GLYCOSYLATION: "
        "Glycans attached to asparagine (N) in the Asn-X-Ser/Thr sequon; "
        "involves ER (LLO assembly + en bloc transfer) and Golgi (processing/trimming); "
        "ALL serum glycoproteins (transferrin, clotting factors, APO-CIII) are N-glycosylated; "
        "CDG affecting LLO assembly or transfer = TYPE I CDG (transferrin IEF Type I pattern); "
        "CDG affecting Golgi processing = TYPE II CDG (transferrin IEF Type II pattern). "
        "2. O-GLYCOSYLATION: "
        "Glycans attached to serine/threonine; "
        "no LLO intermediate; directly added in Golgi; "
        "O-glycosylation defects detected by ApoC-III isoelectric focusing. "
        "CDG DIAGNOSIS WORKFLOW: "
        "Step 1: Serum transferrin isoelectric focusing (IEF) — screening test; "
        "Step 2: If Type I → PMM2 enzyme assay (most common); "
        "Step 3: Gene panel / WES for definitive identification; "
        "Step 4: Functional studies (LLO analysis, Golgi glycan analysis) if needed."
    )

    defs["Transferrin IEF — CDG Screening Test"] = (
        "Serum transferrin isoelectric focusing (IEF) is the PRIMARY SCREENING TEST for CDG. "
        "NORMAL: Transferrin bears 4 N-glycan chains, each with 2 terminal sialic acids "
        "(= 8 sialic acids total per transferrin molecule). "
        "On IEF gel: main band = tetrasialotransferrin; "
        "minor bands = tri-, di-, mono-, asialo-transferrin. "
        "TYPE I CDG PATTERN: "
        "Deficient N-glycosylation → whole glycan chains missing → "
        "INCREASED: disialo- and asialotransferrin bands; "
        "Caused by: defects in LLO assembly (PMM2, MPI, ALG6, DOLK) or OST transfer; "
        "Mnemonic: TYPE I = Two or fewer sialic acids prominent → 'I' for insufficient chains. "
        "TYPE II CDG PATTERN: "
        "Normal chain number but abnormal processing → "
        "INCREASED: trisialo- and other intermediate bands; "
        "Caused by: Golgi processing defects (SLC35A2, SLC35C1, COG7); "
        "Mnemonic: TYPE II = Trimming/processing defect. "
        "MIXED PATTERN (PGM1-CDG): "
        "Both Type I and Type II bands elevated; "
        "unique to PGM1-CDG; diagnostic clue. "
        "IMPORTANT CAVEATS: "
        "False negatives: infants <6 months (transferrin still fetal isoform); "
        "False positives: galactosaemia, alcohol, rare transferrin variants; "
        "Normal IEF does NOT exclude CDG (O-glycosylation defects, GPI disorders, etc.); "
        "ALWAYS confirm positive IEF with confirmatory testing (enzyme assay, gene panel)."
    )

    defs["Treatable CDGs — MPI, PGM1, SLC35C1"] = (
        "THREE of the eight CDG genes in this atlas have specific treatments. "
        "MPI-CDG (CDG-Ib) — ORAL D-MANNOSE: "
        "Mechanism: D-mannose → hexokinase → mannose-6-P, bypassing MPI; "
        "provides mannose for N-glycosylation without needing MPI; "
        "Dose: 1 g/kg/day in 4-6 divided doses; "
        "Effect: NORMALISES transferrin IEF; corrects protein C/S; resolves enteropathy; "
        "prevents hypoglycaemia; "
        "Must be LIFELONG; discontinuing leads to rapid relapse; "
        "Safe: mannose metabolised by hexokinase; avoid excessive dose (hypermannosaemia); "
        "PROGNOSIS: near-normal if started early. "
        "PGM1-CDG — ORAL GALACTOSE: "
        "Mechanism: galactose → galactose-1-P → UDP-galactose (via GALT) → "
        "provides UDP-galactose for Golgi galactosylation bypassing PGM1; "
        "Dose: 0.5 g/kg/day in divided doses (with meals); "
        "Effect: improves transferrin IEF, hepatopathy, cardiomyopathy, exercise tolerance; "
        "Carbohydrate loading (sucrose) before exercise also important; "
        "Must be LIFELONG; monitor galactose plasma levels. "
        "SLC35C1-CDG (LAD-II/CDG-IIc) — ORAL L-FUCOSE: "
        "Mechanism: free fucose → salvage pathway (FPGT: fucose kinase) → GDP-fucose; "
        "bypasses GDP-fucose transporter (SLC35C1) defect; "
        "Dose: 500 mg/kg/day in divided doses; "
        "Effect: partially restores sialyl-LewisX on leukocytes → "
        "reduces severity and frequency of infections; "
        "Some intellectual improvement reported (early treatment); "
        "LIFELONG; START EARLY for maximum neurological benefit."
    )

    defs["CDG and the Golgi — Type II CDGs (SLC35A2, SLC35C1, COG7)"] = (
        "Type II CDGs arise from defects in GOLGI GLYCAN PROCESSING. "
        "The Golgi apparatus processes N-glycans received from the ER "
        "(trimming mannoses, adding GlcNAc, galactose, sialic acid, fucose). "
        "SLC35A2 (UDP-galactose transporter): "
        "Imports UDP-galactose into Golgi lumen; "
        "UDP-galactose is the substrate for beta-4-galactosyltransferases (B4GALT); "
        "B4GALT adds galactose to N-glycan antennae (before sialylation); "
        "SLC35A2 loss → hypogalactosylation → Type II IEF pattern. "
        "SLC35C1 (GDP-fucose transporter): "
        "Imports GDP-fucose into Golgi; "
        "required for fucosyltransferases: FUT1/2 (H antigen), FUT3 (Lewis), "
        "FUT7 (sialyl-LewisX for leukocyte rolling); "
        "SLC35C1 loss → absent fucosylation → no H antigen (Bombay) + no sialyl-LewisX (LAD-II). "
        "COG7 (Conserved Oligomeric Golgi complex): "
        "The COG complex (8 subunits) is a Golgi tethering complex that recycles "
        "Golgi glycosyltransferases back to their correct cisternae; "
        "COG7 loss → multiple Golgi enzymes mis-localised → global hypoglycosylation "
        "of both N- and O-glycans → severe Type II CDG pattern; "
        "COG complex defects (COG1-8) are collectively called 'CDG-II Golgi tethering disorders'. "
        "DISTINGUISHING TYPE II CDGs: "
        "Transferrin IEF Type II pattern is common to all; "
        "ApoC-III IEF: distinguishes N-glycan from O-glycan involvement; "
        "CD15s (sialyl-LewisX) flow cytometry: absent in SLC35C1-CDG (LAD-II) specifically; "
        "Bombay blood group (blood bank): SLC35C1-CDG specifically; "
        "Wrinkled neonatal skin: COG7-CDG specifically; "
        "Epilepsy + X-linked de novo: SLC35A2-CDG."
    )

    defs["CDG — Newborn Screening and Cascade Testing"] = (
        "NEWBORN SCREENING (NBS) FOR CDG: "
        "CDG is NOT universally included in routine NBS programmes. "
        "Some centres include transferrin IEF in expanded metabolic NBS; "
        "MPI-CDG and PMM2-CDG can be detected by NBS transferrin IEF in pilot programmes; "
        "Pilot NBS programmes ongoing in Netherlands, some US states; "
        "Current detection: most CDG cases are diagnosed symptomatically (missed by routine NBS). "
        "CLINICAL TRIGGERS FOR CDG TESTING: "
        "Unexplained cerebellar hypoplasia or ataxia + intellectual disability → PMM2-CDG first; "
        "Protein-losing enteropathy + hepatopathy + NO neurology → MPI-CDG; "
        "Bifid uvula + DCM + hepatopathy → PGM1-CDG; "
        "Bombay blood group + recurrent infections + leukocytosis → SLC35C1/LAD-II; "
        "Wrinkled neonatal skin + liver failure → COG7-CDG; "
        "De novo epilepsy in female infant → SLC35A2-CDG. "
        "CASCADE TESTING (FAMILY SCREENING): "
        "After index case confirmed: "
        "AR CDGs (PMM2, MPI, ALG6, PGM1, SLC35C1, DOLK, COG7): "
        "parents = obligate carriers; siblings 25% recurrence risk; "
        "carrier testing: maternal + paternal sequencing; "
        "prenatal diagnosis: CVS or amniocentesis for gene sequencing; "
        "X-linked (SLC35A2): de novo in most cases; "
        "parental germline mosaic risk ~1-2% recurrence; "
        "confirm neither parent carries the variant. "
        "GENETIC COUNSELLING: "
        "All AR CDGs: emphasise 25% sibling risk; "
        "PGT-M available for known family variants; "
        "For MPI-CDG and PGM1-CDG: emphasise TREATABILITY — early diagnosis changes prognosis."
    )

    return {
        "atlas": "Hereditary-CDG-Atlas — Clinical Definitions",
        "definitions": defs,
        "total_genes": len(CDG_GENES),
        "total_definition_entries": len(defs),
        "seed_range": "1862–1869",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2, default=str))
