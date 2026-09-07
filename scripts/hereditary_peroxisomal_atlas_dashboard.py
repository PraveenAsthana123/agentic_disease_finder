#!/usr/bin/env python3
"""Hereditary-Peroxisomal-Atlas — Complete 8-Gene Hereditary Peroxisomal Disorders Atlas
PEX1    (Zellweger Spectrum Disorder type 1; 1283 aa; 7q21.2; AR;
         Most common peroxisome biogenesis disorder (60-70% of all PBD);
         VLCFA + pipecolic acid + plasmalogens reduced;
         DHA (docosahexaenoic acid) supplementation MANDATORY;
         seed SEED_BASE+0) .
PEX6    (Zellweger Spectrum Disorder type 4; 980 aa; 6p21.1; AR;
         Second most common PBD; attenuated ZSD common;
         VLCFA screen + pipecolic acid;
         seed SEED_BASE+1) .
ABCD1   (X-linked Adrenoleukodystrophy; 745 aa; Xq28; X-linked;
         C26:0/C22:0 ratio DIAGNOSTIC; Lorenzo's Oil historical;
         HSCT curative if Loes score <9 early; AMN adult phenotype;
         seed SEED_BASE+2) .
PHYH    (Refsum Disease; 338 aa; 10p13; AR;
         Phytanic acid accumulation; dietary fat restriction CURATIVE;
         NO green vegetables/dairy/ruminant fat;
         Plasmapheresis in crisis;
         seed SEED_BASE+3) .
PEX7    (Rhizomelic Chondrodysplasia Punctata type 1; 323 aa; 6q23.3; AR;
         Stippled epiphyses PATHOGNOMONIC on X-ray;
         Rhizomelic limb shortening; plasmalogens markedly reduced;
         seed SEED_BASE+4) .
HSD17B4 (D-Bifunctional Protein Deficiency; 736 aa; 5q23.1; AR;
         Most severe peroxisomal beta-oxidation disorder;
         Zellweger-like but peroxisome structure INTACT;
         VLCFA + elevated bile acid intermediates;
         seed SEED_BASE+5) .
ACOX1   (Acyl-CoA Oxidase 1 Deficiency; 700 aa; 17q25.1; AR;
         VLCFA oxidation first committed step;
         Pseudo-neonatal ALD phenotype; gain-of-function variants cause
         ACOX1 inflammatory disease (AID);
         seed SEED_BASE+6) .
AGPS    (Rhizomelic Chondrodysplasia Punctata type 3; 728 aa; 2q31.2; AR;
         DHAP-acyltransferase deficiency; ether lipid synthesis;
         Plasmalogen deficiency WITHOUT rhizomelia as severe as RCDP1;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1854–1861)
"""

import random

SEED_BASE = 1854

PEROX_GENES = [
    # -- PEX1 -- Peroxisome Biogenesis Disorder / Zellweger Spectrum -----------
    {
        "gene": "PEX1",
        "protein": (
            "PEX1 -- 7q21.2 AR -- Peroxin-1-1283aa -- "
            "Zellweger-Spectrum-Disorder-ZSD1-Most-Common-PBD-60-70pct -- "
            "VLCFA-Elevated-C26:0-Pipecolic-Acid-Plasmalogens-Reduced -- "
            "DHA-Docosahexaenoic-Acid-Supplementation-MANDATORY -- "
            "Retinitis-Pigmentosa-Sensorineural-Hearing-Loss-Adrenal -- "
            "Zellweger-Severe-Neonatal-to-IRD-Attenuated-Adult-Spectrum"
        ),
        "alias": (
            "PEX1 (peroxisomal biogenesis factor 1); OMIM gene 602136. "
            "Zellweger Spectrum Disorder type 1 OMIM 214100. "
            "7q21.2; 1283 aa; ~143 kDa; cytoplasm; AAA-ATPase; "
            "forms heterodimer with PEX6 (also AAA-ATPase); "
            "complex exports PEX5 (import receptor) from peroxisome membrane for recycling. "
            "autosomal recessive (biallelic). "
            "FUNCTION: PEX1/PEX6 heterodimer (with PEX26 as membrane anchor) "
            "uses ATP hydrolysis to retrotranslocate PEX5 from the peroxisomal membrane "
            "back to the cytoplasm for another round of peroxisomal matrix protein import. "
            "Without PEX1: PEX5 is trapped in the membrane → ubiquitinated → degraded; "
            "peroxisomal matrix protein import fails → GHOST PEROXISOMES "
            "(empty membrane remnants without matrix enzymes). "
            "CONSEQUENCES OF FAILED PEROXISOMAL IMPORT: "
            "1. VLCFA (very long chain fatty acids, C22-C26) accumulate — cannot be beta-oxidised; "
            "2. Plasmalogens (ether phospholipids, essential for myelin and cell membranes) reduced; "
            "3. Pipecolic acid accumulates (peroxisomal catabolism blocked); "
            "4. Bile acid intermediates accumulate (di/trihydroxycholestanoic acid); "
            "5. DHA (C22:6, docosahexaenoic acid) — cannot be synthesised via peroxisomal pathway. "
            "CLINICAL SPECTRUM (genotype-phenotype): "
            "1. Zellweger Syndrome (ZS, severe): neonatal onset; "
            "hypotonia + seizures + cortical neuronal migration defects + "
            "hepatic failure + chondrodysplasia punctata + dysmorphic face; "
            "death in first year; NO ghost peroxisomes visible; "
            "2. Neonatal Adrenoleukodystrophy (NALD): moderate; "
            "neonatal onset, survive beyond year 1; progressive leukodystrophy; "
            "3. Infantile Refsum Disease (IRD): mild-moderate; "
            "accumulation of phytanic acid additionally; "
            "retinitis pigmentosa + SNHL + intellectual disability; survive to adulthood; "
            "4. Attenuated ZSD: adult-onset; RP + SNHL + cerebellar ataxia ± peripheral neuropathy; "
            "sometimes diagnosed in adulthood (misdiagnosed as Usher syndrome, CMT, or olivopontocerebellar atrophy). "
            "MOST COMMON VARIANTS: "
            "p.Gly843Asp (c.2528G>A): most frequent PEX1 variant (~40% of alleles in European cohorts); "
            "leaky splice/missense — attenuated phenotype; "
            "p.Ile700Tyr: severe Zellweger; "
            "compound heterozygotes p.Gly843Asp/null → intermediate NALD/IRD phenotype. "
            "DIAGNOSIS: "
            "VLCFA plasma: C26:0, C24:0/C22:0 ratio, C26:0/C22:0 ratio ELEVATED; "
            "Pipecolic acid: ELEVATED plasma/urine; "
            "Plasmalogens (RBC): REDUCED; "
            "Bile acid intermediates (urine): DHCA, THCA; "
            "DHA (plasma): LOW; "
            "Peroxisome morphology: fibroblast immunofluorescence — ghost peroxisomes; "
            "Enzymatic: DHAP-acyltransferase, phytanic acid oxidase — globally reduced. "
            "DHA SUPPLEMENTATION — MANDATORY: "
            "DHA (docosahexaenoic acid, omega-3, 22:6 n-3) supplementation: "
            "70-100 mg/kg/day in infants; reduces VLCFA accumulation partially; "
            "improves myelination and retinal function in IRD/attenuated ZSD; "
            "does NOT reverse structural brain damage; "
            "must be started early; monitor DHA plasma levels. "
            "ADRENAL INSUFFICIENCY: "
            "VLCFA toxic to adrenal cortex → adrenal insufficiency common; "
            "screen with ACTH stimulation test; "
            "hydrocortisone replacement mandatory if confirmed. "
            "PROGNOSIS: "
            "Severe ZS: death within 6-12 months; "
            "IRD/attenuated: survival to adulthood; RP progressive → blind; SNHL progressive."
        ),
        "age_of_onset": "Neonatal (severe ZS/NALD) to adult (attenuated IRD)",
        "inheritance": "Autosomal recessive (biallelic PEX1 mutations)",
        "locus": "7q21.2",
        "protein_size": "1283 aa",
        "key_biomarker": "VLCFA C26:0 elevated; C26:0/C22:0 ratio elevated; plasmalogens (RBC) reduced; pipecolic acid elevated; DHA low",
        "pathognomonic": "Ghost peroxisomes on fibroblast IF + elevated VLCFA + reduced plasmalogens = PEX1/PBD",
        "treatment": "DHA supplementation (MANDATORY); adrenal replacement if insufficient; fat-soluble vitamins; seizure control; hearing aids/cochlear implants",
        "critical_flags": [
            "DHA-SUPPLEMENTATION-MANDATORY — 70-100 mg/kg/day; reduces VLCFA, improves retinal + myelin",
            "ADRENAL-INSUFFICIENCY-SCREEN-ACTH — hydrocortisone if confirmed; adrenal crisis risk",
            "p.Gly843Asp-MOST-COMMON-ATTENUATED — 40% European alleles; leaky variant; IRD/adult phenotype",
            "VLCFA-C26:0-PRIMARY-SCREEN — NOT C24:0 alone; C26:0/C22:0 ratio key",
            "GHOST-PEROXISOMES-FIBROBLAST-IF-DIAGNOSTIC — PMP70 staining shows empty vesicles",
            "RETINITIS-PIGMENTOSA-PROGRESSIVE — annual ophthalmology mandatory in survivors",
            "AVOID-FAT-SOLUBLE-VITAMIN-DEFICIENCY — fat malabsorption → vitamins A/D/E/K supplementation",
            "HSCT-NOT-ESTABLISHED-FOR-ZSD — unlike ALD/MLD, HSCT unproven in PBD",
        ],
    },
    # -- PEX6 -- Zellweger Spectrum Disorder type 4 ----------------------------
    {
        "gene": "PEX6",
        "protein": (
            "PEX6 -- 6p21.1 AR -- Peroxin-6-980aa -- "
            "Zellweger-Spectrum-Disorder-ZSD4-Second-Most-Common-PBD -- "
            "VLCFA-Pipecolic-Acid-Plasmalogens-Same-Biochemical-As-PEX1 -- "
            "Attenuated-ZSD-Adult-Onset-RP-SNHL-Ataxia-Common -- "
            "PEX1-PEX6-Heterodimer-AAA-ATPase-PEX5-Recycling -- "
            "p.Arg860Trp-Most-Common-Attenuated-European-Founder"
        ),
        "alias": (
            "PEX6 (peroxisomal biogenesis factor 6); OMIM gene 601498. "
            "Zellweger Spectrum Disorder type 4 OMIM 614862. "
            "6p21.1; 980 aa; ~104 kDa; cytoplasm / peroxisomal membrane; "
            "AAA-ATPase (ATPases Associated with diverse Activities); "
            "forms obligate heterodimer with PEX1; anchored to peroxisomal membrane by PEX26. "
            "autosomal recessive. "
            "FUNCTION: PEX6 (with PEX1) uses ATP hydrolysis to retrotranslocate "
            "PEX5 (the peroxisomal targeting signal type 1 receptor) from the "
            "peroxisomal membrane back to the cytoplasm. "
            "PEX5 shuttles PTS1-bearing proteins (most peroxisomal matrix enzymes) "
            "from cytoplasm into peroxisome matrix. "
            "PEX6 loss → PEX5 trapped and degraded → failed peroxisomal import. "
            "BIOCHEMICAL PROFILE (IDENTICAL TO PEX1): "
            "VLCFA elevated (C26:0, C26:0/C22:0 ratio); "
            "plasmalogens (RBC) reduced; "
            "pipecolic acid elevated; "
            "bile acid intermediates (DHCA/THCA) in urine; "
            "DHA reduced. "
            "DISTINGUISHING PEX6 FROM PEX1: "
            "Biochemistry identical; only gene sequencing differentiates; "
            "Fibroblast complementation (import of reporter PTS1 protein) distinguishes complementation group; "
            "Next-generation sequencing panel covers both. "
            "CLINICAL SPECTRUM: "
            "Same ZS-NALD-IRD-attenuated ZSD spectrum as PEX1; "
            "ATTENUATED PHENOTYPE MORE COMMON in PEX6 than PEX1 (higher proportion of leaky variants); "
            "Adult presentations: "
            "progressive hearing loss (SNHL) — often first symptom; "
            "retinitis pigmentosa → tunnel vision → blindness; "
            "cerebellar ataxia (progressive); "
            "peripheral sensorimotor neuropathy; "
            "adrenal insufficiency; "
            "often misdiagnosed as Usher syndrome, ARCA, or CMT. "
            "MOST COMMON VARIANT: "
            "p.Arg860Trp (c.2578C>T): most common PEX6 variant; "
            "attenuated, leaky — some residual PEX6/PEX1 ATPase function; "
            "compound het p.Arg860Trp/null → IRD or attenuated ZSD. "
            "DIAGNOSIS in adults (attenuated phenotype): "
            "Suspect in: RP + SNHL + ataxia combination (Usher DDx); "
            "VLCFA screen — may be borderline elevated in attenuated; "
            "repeat testing if clinical suspicion high; "
            "PEX6 gene panel or peroxisomal gene panel. "
            "TREATMENT: "
            "Same as PEX1 ZSD: DHA supplementation; adrenal screen + replacement; "
            "vitamins A/D/E/K; cochlear implants for SNHL; "
            "retinal gene therapy trials underway (not yet standard)."
        ),
        "age_of_onset": "Neonatal (severe) to adult (attenuated)",
        "inheritance": "Autosomal recessive (biallelic PEX6 mutations)",
        "locus": "6p21.1",
        "protein_size": "980 aa",
        "key_biomarker": "VLCFA C26:0/C22:0 elevated; plasmalogens reduced; pipecolic acid elevated (same as PEX1; only gene sequencing differentiates)",
        "pathognomonic": "Ghost peroxisomes + VLCFA + pipecolic acid; gene panel distinguishes from PEX1",
        "treatment": "DHA supplementation; adrenal replacement; vitamins A/D/E/K; cochlear implants; HSCT not established",
        "critical_flags": [
            "BIOCHEMISTRY-IDENTICAL-TO-PEX1 — only genetic sequencing differentiates PEX1 vs PEX6",
            "p.Arg860Trp-MOST-COMMON-ATTENUATED-EUROPEAN — high proportion attenuated ZSD in PEX6",
            "ADULT-ONSET-MISDIAGNOSIS-HIGH — RP+SNHL+ataxia mimics Usher syndrome, ARCA, CMT",
            "VLCFA-MAY-BE-BORDERLINE-IN-ATTENUATED — borderline result does NOT exclude diagnosis",
            "ADRENAL-INSUFFICIENCY-ALL-ZSD — screen all patients regardless of age at presentation",
            "DHA-SUPPLEMENTATION-AS-PEX1 — same protocol; mandatory in all ZSD",
            "RETINITIS-PIGMENTOSA-ANNUAL-OPHTHALMOLOGY — document progression; avoid bright light",
            "GENE-PANEL-PEROXISOMAL-NOT-SINGLE-GENE — always test PEX1+PEX6+other PEX genes simultaneously",
        ],
    },
    # -- ABCD1 -- X-linked Adrenoleukodystrophy --------------------------------
    {
        "gene": "ABCD1",
        "protein": (
            "ABCD1 -- Xq28 XLR -- ALDP-745aa -- "
            "X-linked-Adrenoleukodystrophy-ALD-Adrenomyeloneuropathy-AMN -- "
            "C26:0/C22:0-Ratio-DIAGNOSTIC-Screen-ALL-Males -- "
            "HSCT-Curative-IF-Loes-Score-<9-Age-<8-Asymptomatic -- "
            "Lorenzo's-Oil-Does-NOT-Halt-Neurological-Progression -- "
            "Adrenal-Insufficiency-80pct-Hydrocortisone-Mandatory"
        ),
        "alias": (
            "ABCD1 (ATP-binding cassette subfamily D member 1); OMIM gene 300371. "
            "X-linked Adrenoleukodystrophy OMIM 300100. "
            "Xq28; 745 aa; ~84 kDa; peroxisomal membrane; half-transporter (pairs with ABCD2/3); "
            "X-linked (Xq28). "
            "FUNCTION: ABCD1 (also called ALD protein, ALDP) is a peroxisomal membrane "
            "ABC half-transporter that imports activated very long chain fatty acids "
            "(VLCFA-CoA esters) from cytoplasm into peroxisomes for beta-oxidation. "
            "Without ABCD1: VLCFA (especially C26:0, C24:0, C22:0) cannot enter peroxisomes "
            "→ accumulate in plasma, tissues (especially brain white matter, adrenal cortex, "
            "testes, spinal cord). "
            "VLCFAs are toxic: intercalate into myelin, disrupt membrane lipid composition, "
            "trigger neuroinflammation (especially in CALD form). "
            "X-LINKED GENETICS: "
            "Males: hemizygous → all males with pathogenic variant are affected (penetrance ~100% for adrenal); "
            "Females: heterozygous carriers → may develop AMN-like mild neuropathy in 20% after age 40; "
            "CARRIER FEMALES: 20% develop myelopathy; adrenal insufficiency very rare in females. "
            "CLINICAL PHENOTYPES IN MALES (ALL share VLCFA accumulation): "
            "1. CHILDHOOD ALD (CALD) — cerebral — 35-40% of affected boys: "
            "onset age 3-10 years; inflammatory demyelination of posterior cerebral white matter; "
            "initial symptoms: attention problems, behaviour change; "
            "rapidly progressive to blindness, deafness, quadriplegia, dementia; "
            "death within 2-5 years if untreated; "
            "MRI: posterior white matter T2/FLAIR enhancement, contrast enhancement (active inflammation); "
            "LOES SCORE quantifies extent (0=normal, 34=maximal); "
            "2. ADRENOMYELONEUROPATHY (AMN) — 40-45% — adult onset (20-30s): "
            "spastic paraparesis (corticospinal) + peripheral neuropathy + bladder dysfunction; "
            "slowly progressive over decades; cerebral involvement in 20% of AMN adults; "
            "3. ADDISON ONLY — 10-15%: adrenal insufficiency without neurological disease (yet); "
            "may convert to AMN or CALD; "
            "4. ASYMPTOMATIC — detected by NBS or family screening; may develop any phenotype. "
            "PHENOTYPE PREDICTION: "
            "Genotype does NOT predict cerebral vs AMN vs adrenal-only phenotype; "
            "same mutation in family → different phenotypes (brothers may differ); "
            "modifier genes and environmental triggers proposed but unproven. "
            "DIAGNOSIS: "
            "C26:0/C22:0 plasma ratio: ELEVATED (primary screen); "
            "C26:0 absolute: elevated; "
            "NEWBORN SCREENING: C26:0-lysophosphatidylcholine (C26:0-LPC) on DBS — now in most US states; "
            "Females: VLCFA may be normal in ~15% of female carriers → genetic testing MANDATORY; "
            "ABCD1 sequencing: confirms diagnosis. "
            "HSCT — CURATIVE BUT TIMING-CRITICAL: "
            "Hematopoietic stem cell transplantation (HSCT) is the only disease-modifying treatment for CALD; "
            "SUCCESS REQUIRES: "
            "Loes MRI score < 9 (limited disease) AND "
            "Absence of neurological symptoms (pre-symptomatic or very early); "
            "MECHANISM: donor microglia from graft replace dysfunctional resident microglia; "
            "HSCT halts neuroinflammation; does NOT reverse existing damage; "
            "Outcome if Loes >9 or symptomatic: HSCT does NOT prevent progression; "
            "LIFELONG MRI SURVEILLANCE: 6-monthly MRI in asymptomatic boys ages 3-12. "
            "LORENZO'S OIL (oleic acid + erucic acid blend): "
            "normalises VLCFA plasma levels; "
            "DOES NOT halt cerebral demyelination or neurological decline; "
            "may reduce conversion from asymptomatic to CALD in some studies; "
            "NOT a substitute for HSCT; "
            "used in pre-symptomatic asymptomatic males to potentially reduce CALD risk. "
            "ADRENAL INSUFFICIENCY: "
            "~80% of males eventually develop primary adrenal insufficiency; "
            "hydrocortisone replacement MANDATORY; "
            "adrenal crisis can occur before neurological disease is evident; "
            "CHECK ADRENALS IN ALL ABCD1 MALES REGARDLESS OF NEUROLOGICAL STATUS. "
            "GENE THERAPY: "
            "Lenti-D (skysona, elivaldogene autotemcel): EMA/FDA approved; "
            "autologous HSC transduced with lentiviral ABCD1; "
            "alternative to matched HSCT; similar efficacy in early CALD."
        ),
        "age_of_onset": "Adrenal insufficiency any age; CALD age 3-10; AMN age 20-30s",
        "inheritance": "X-linked recessive (Xq28); carrier females rarely symptomatic",
        "locus": "Xq28",
        "protein_size": "745 aa",
        "key_biomarker": "C26:0/C22:0 plasma ratio ELEVATED; C26:0 absolute elevated; C26:0-LPC on NBS DBS",
        "pathognomonic": "Posterior white matter T2 + contrast enhancement + C26:0/C22:0 elevated = CALD",
        "treatment": "HSCT (curative if Loes <9, pre-symptomatic); hydrocortisone (adrenal); 6-monthly MRI surveillance age 3-12; Lenti-D gene therapy",
        "critical_flags": [
            "HSCT-CURATIVE-ONLY-IF-LOES-<9-AND-ASYMPTOMATIC — Loes ≥9 or symptomatic → HSCT futile",
            "ADRENAL-INSUFFICIENCY-80pct-ALL-MALES — hydrocortisone mandatory; check before neurology",
            "LORENZOS-OIL-DOES-NOT-HALT-NEUROLOGICAL-PROGRESSION — lowers VLCFA but not neuroprotective",
            "VLCFA-NORMAL-IN-15pct-FEMALE-CARRIERS — genetic testing mandatory for carrier females",
            "6-MONTHLY-MRI-AGES-3-12-ASYMPTOMATIC-BOYS — early detection allows HSCT window",
            "GENOTYPE-DOES-NOT-PREDICT-PHENOTYPE — same variant → CALD vs AMN vs adrenal-only in family",
            "LENTI-D-GENE-THERAPY-FDA-APPROVED — alternative to allogeneic HSCT when no matched donor",
            "ADRENAL-CRISIS-BEFORE-NEUROLOGY — treat adrenal first; adrenal crisis = presenting event in many",
        ],
    },
    # -- PHYH -- Refsum Disease ------------------------------------------------
    {
        "gene": "PHYH",
        "protein": (
            "PHYH -- 10p13 AR -- Phytanoyl-CoA-Hydroxylase-338aa -- "
            "Refsum-Disease-Classic-Adult-Onset -- "
            "Dietary-Phytanic-Acid-Restriction-CURATIVE-Green-Veg-Dairy-Ruminant-Fat-ELIMINATED -- "
            "Retinitis-Pigmentosa-Cerebellar-Ataxia-Peripheral-Neuropathy-TRIAD -- "
            "Plasmapheresis-LDL-Apheresis-CRISIS-Acute-Neuropathy -- "
            "Cardiac-Arrhythmia-Sudden-Death-Risk-Monitor"
        ),
        "alias": (
            "PHYH (phytanoyl-CoA 2-hydroxylase); OMIM gene 602026. "
            "Refsum Disease OMIM 266500. "
            "10p13; 338 aa; ~40 kDa; peroxisomal matrix; "
            "2-hydroxylase using 2-oxoglutarate and O2 as co-substrates; "
            "autosomal recessive. "
            "NOTE: ~10% of classic Refsum disease is caused by PEX7 mutations "
            "(impaired phytanoyl-CoA hydroxylase import into peroxisomes). "
            "FUNCTION: PHYH catalyses the alpha-oxidation of phytanoyl-CoA: "
            "Phytanoyl-CoA → 2-hydroxyphytanoyl-CoA → "
            "pristanal → pristanoyl-CoA (then undergoes peroxisomal beta-oxidation). "
            "PHYTANIC ACID (3,7,11,15-tetramethylhexadecanoic acid) is a branched-chain "
            "fatty acid derived ENTIRELY from dietary sources: "
            "chlorophyll (green vegetables, algae, seaweeds); "
            "ruminant fat (dairy: butter, cheese, milk; beef fat, mutton fat); "
            "fatty fish (tuna, cod, herring). "
            "HUMANS CANNOT SYNTHESISE phytanic acid — it is 100% dietary. "
            "Without PHYH: phytanic acid cannot undergo alpha-oxidation → accumulates; "
            "phytanic acid is neurotoxic: intercalates into myelin membranes. "
            "CLINICAL PHENOTYPE (adult onset, insidious): "
            "CLASSIC TRIAD: "
            "1. Retinitis Pigmentosa: progressive; night blindness first; bone spicule pigmentation; "
            "2. Cerebellar Ataxia: gait, limb ataxia; "
            "3. Peripheral Polyneuropathy: sensorimotor; elevated CSF protein (albuminocytological dissociation); "
            "ADDITIONAL FEATURES: "
            "Ichthyosis (fish-scale skin): non-inflammatory; "
            "Anosmia (loss of smell) — very characteristic; "
            "Sensorineural hearing loss; "
            "Cardiac: cardiomyopathy + cardiac arrhythmia → SUDDEN CARDIAC DEATH risk; "
            "Skeletal: shortening of 4th metatarsal; epiphyseal dysplasia. "
            "DIETARY TREATMENT — CORNERSTONE (CURATIVE): "
            "Phytanic acid restriction lowers plasma phytanic acid → prevents/arrests neurological progression; "
            "ELIMINATE: "
            "Green vegetables (chlorophyll-containing): spinach, broccoli, kale, green beans, peas; "
            "Dairy products: milk, butter, cream, cheese (all contain phytanic acid from ruminant gut); "
            "Ruminant meat fat: beef, mutton, lamb fat trimmings; "
            "Fatty fish: herring, tuna, cod liver oil; "
            "ALLOWED: "
            "Lean white meat (chicken, turkey); "
            "Non-ruminant fat (pork, vegetable oils — olive oil is low phytanic); "
            "Fruit, grains, legumes; "
            "Target: plasma phytanic acid <200 µmol/L (normal <10 µmol/L). "
            "PLASMAPHERESIS / LDL-APHERESIS — CRISIS: "
            "Rapid reduction of plasma phytanic acid in acute deterioration; "
            "used when dietary restriction alone inadequate acutely; "
            "effective because phytanic acid is protein-bound (>99% albumin-bound); "
            "monthly or fortnightly cycles until plasma normalised. "
            "FASTING PARADOX — CRITICAL SAFETY RULE: "
            "FASTING IS DANGEROUS in Refsum disease: "
            "starvation mobilises stored phytanic acid from adipose tissue → "
            "acute elevation of plasma phytanic acid → acute neuropathy/arrhythmia; "
            "NEVER fast patients; perioperative glucose infusion mandatory; "
            "RAPID WEIGHT LOSS also dangerous — mobilises adipose phytanic acid stores. "
            "CARDIAC MONITORING: "
            "Annual ECG + echocardiogram; "
            "24h Holter for arrhythmia; "
            "ICD consider if arrhythmia documented; "
            "sudden cardiac death is a recognised cause of death in Refsum disease."
        ),
        "age_of_onset": "Late childhood to adulthood (typically 2nd-4th decade)",
        "inheritance": "Autosomal recessive (biallelic PHYH; ~10% due to PEX7)",
        "locus": "10p13",
        "protein_size": "338 aa",
        "key_biomarker": "Plasma phytanic acid MARKEDLY ELEVATED (>200 µmol/L; normal <10); pristanic acid elevated; VLCFA normal",
        "pathognomonic": "Elevated phytanic acid + RP + ataxia + neuropathy + anosmia = Refsum disease",
        "treatment": "Phytanic acid dietary restriction (MANDATORY, curative); plasmapheresis/LDL-apheresis (crisis); NEVER fast; cardiac monitoring (ICD if arrhythmia)",
        "critical_flags": [
            "DIETARY-RESTRICTION-CURATIVE — eliminate chlorophyll sources, dairy, ruminant fat, fatty fish",
            "FASTING-ABSOLUTELY-DANGEROUS — mobilises adipose phytanic → acute neuropathy/arrhythmia",
            "PLASMAPHERESIS-IN-CRISIS — rapid phytanic reduction; phytanic acid >99% albumin bound",
            "VLCFA-NORMAL — differentiates Refsum from ZSD/ALD (VLCFA normal, phytanic elevated)",
            "ANOSMIA-CHARACTERISTIC — loss of smell; rare finding in differential; ask specifically",
            "CARDIAC-SUDDEN-DEATH-RISK — annual ECG + echo + Holter; ICD threshold low",
            "RAPID-WEIGHT-LOSS-DANGEROUS — mobilises phytanic stores; avoid crash diets",
            "PEX7-CAUSES-10pct-REFSUM — if PHYH sequencing negative, test PEX7 (RCDP1 gene)",
        ],
    },
    # -- PEX7 -- Rhizomelic Chondrodysplasia Punctata type 1 -------------------
    {
        "gene": "PEX7",
        "protein": (
            "PEX7 -- 6q23.3 AR -- Peroxin-7-323aa -- "
            "Rhizomelic-Chondrodysplasia-Punctata-type-1-RCDP1 -- "
            "Stippled-Epiphyses-X-ray-PATHOGNOMONIC -- "
            "Rhizomelic-Limb-Shortening-Proximal-Humerus-Femur -- "
            "Plasmalogens-Markedly-Reduced-Phytanic-Acid-Elevated -- "
            "PTS2-Receptor-Imports-3-Peroxisomal-Enzymes-Only-Not-Global-Import"
        ),
        "alias": (
            "PEX7 (peroxisomal biogenesis factor 7, peroxisomal targeting signal 2 receptor); "
            "OMIM gene 601757. "
            "Rhizomelic Chondrodysplasia Punctata type 1 OMIM 215100. "
            "6q23.3; 323 aa; ~36 kDa; cytoplasm; WD-repeat protein; "
            "autosomal recessive. "
            "FUNCTION: PEX7 is the receptor for peroxisomal targeting signal type 2 (PTS2). "
            "UNLIKE PEX1/PEX6 (global import failure), PEX7 imports ONLY 3 specific enzymes: "
            "1. AGPS (alkylglycerone phosphate synthase) — ether lipid (plasmalogen) synthesis; "
            "2. PHYH (phytanoyl-CoA hydroxylase) — phytanic acid alpha-oxidation; "
            "3. HACL1 (2-hydroxy phytanoyl-CoA lyase) — alpha-oxidation; "
            "Peroxisomal matrix generally INTACT in RCDP (not ghost peroxisomes); "
            "VLCFA NORMAL (because ACOX1/MFP2 etc import via PTS1, not PTS2). "
            "BIOCHEMICAL CONSEQUENCES: "
            "PLASMALOGENS MARKEDLY REDUCED — AGPS cannot enter peroxisome; "
            "PHYTANIC ACID ELEVATED — PHYH cannot enter peroxisome; "
            "VLCFA NORMAL (differentiates from ZSD). "
            "RCDP1 CLINICAL FEATURES (severe and classic): "
            "PATHOGNOMONIC: Stippled epiphyses (chondrodysplasia punctata) on X-ray — "
            "punctate calcifications in cartilage (epiphyses, vertebrae, trachea); "
            "appears on prenatal ultrasound or postnatal X-ray; "
            "RHIZOMELIC LIMB SHORTENING: proximal limbs (humerus, femur) shorter; "
            "characteristic skeletal dysplasia; "
            "CATARACTS: bilateral congenital cataracts (plasmalogens essential for lens); "
            "INTELLECTUAL DISABILITY: severe; "
            "SEIZURES: refractory; "
            "SHORT STATURE: profound growth deficiency; "
            "CONTRACTURES: joint contractures from early age; "
            "SKIN: mild ichthyosis; "
            "BRAIN: abnormal myelination; "
            "SURVIVAL: most severe die in early childhood; some survive to teens. "
            "MILD/ATTENUATED RCDP1: "
            "Incomplete penetrance for rhizomelia; "
            "May have only cataracts + mild intellectual disability; "
            "Includes milder alleles like p.Leu292ter (leaky); "
            "Some cases diagnosed in adulthood (Refsum-like: RP + neuropathy + elevated phytanic acid). "
            "DIAGNOSIS: "
            "X-ray: stippled epiphyses (key first investigation); "
            "Plasmalogens (RBC): MARKEDLY REDUCED (diagnostic); "
            "Phytanic acid (plasma): elevated (PHYH import blocked); "
            "VLCFA: NORMAL (key differentiator from ZSD); "
            "PEX7 sequencing: confirms. "
            "TREATMENT: "
            "No specific therapy; symptomatic only; "
            "Dietary phytanic acid restriction (reduces phytanic load); "
            "Cataract surgery: early if cataracts impair vision; "
            "Seizure management; "
            "Plasmalogen replacement: oral plasmalogen precursors (PEMBA study, in clinical trials); "
            "DHA supplementation: uncertain benefit in RCDP1."
        ),
        "age_of_onset": "Prenatal (stippled epiphyses on ultrasound) to neonatal (classic); mild forms: later",
        "inheritance": "Autosomal recessive (biallelic PEX7 mutations)",
        "locus": "6q23.3",
        "protein_size": "323 aa",
        "key_biomarker": "Plasmalogens (RBC) MARKEDLY REDUCED; phytanic acid elevated; VLCFA NORMAL (critical differentiator from ZSD)",
        "pathognomonic": "Stippled epiphyses on X-ray + VLCFA normal + plasmalogens markedly reduced = RCDP1/PEX7",
        "treatment": "Supportive (no curative therapy); dietary phytanic restriction; early cataract surgery; seizure control; plasmalogen precursor trials (PEMBA)",
        "critical_flags": [
            "STIPPLED-EPIPHYSES-X-RAY-PATHOGNOMONIC — punctate calcifications in cartilage; first key investigation",
            "VLCFA-NORMAL — differentiates RCDP from ZSD/ALD (PEX7 imports PTS2 only, not global import)",
            "PLASMALOGENS-RBC-MARKEDLY-REDUCED — primary biochemical marker; more sensitive than phytanic acid",
            "PHYTANIC-ACID-ELEVATED-PHYH-IMPORT-BLOCKED — same mechanism as Refsum; dietary restriction",
            "PTS2-RECEPTOR-SELECTIVE — only 3 enzymes affected (AGPS, PHYH, HACL1); NOT global peroxisomal failure",
            "CONGENITAL-CATARACTS-BILATERAL — early surgery if vision impaired; plasmalogens critical for lens",
            "RCDP1-MOST-COMMON-RCDP — accounts for ~90% of RCDP; PEX7 most common gene",
            "PLASMALOGEN-REPLACEMENT-TRIALS — PEMBA oral precursors in clinical trials; no approved therapy yet",
        ],
    },
    # -- HSD17B4 -- D-Bifunctional Protein Deficiency --------------------------
    {
        "gene": "HSD17B4",
        "protein": (
            "HSD17B4 -- 5q23.1 AR -- D-Bifunctional-Protein-DBP-736aa -- "
            "DBP-Deficiency-Most-Severe-Peroxisomal-Beta-Oxidation-Disorder -- "
            "Zellweger-Like-Phenotype-But-Peroxisome-Structure-INTACT -- "
            "VLCFA-Elevated-Bile-Acid-Intermediates-Pristanic-Acid-Elevated -- "
            "Neonatal-Severe-Hypotonia-Seizures-Facial-Dysmorphism-Early-Death -- "
            "No-Treatment-Available-Supportive-Only"
        ),
        "alias": (
            "HSD17B4 (hydroxysteroid 17-beta dehydrogenase 4); OMIM gene 601860. "
            "D-bifunctional protein deficiency OMIM 261515. "
            "5q23.1; 736 aa; ~79 kDa; peroxisomal matrix; "
            "multifunctional enzyme (DBP, D-BP, MFP2): "
            "carries two catalytic activities: "
            "(1) 2-enoyl-CoA hydratase (D-specific); "
            "(2) L-3-hydroxyacyl-CoA dehydrogenase (D-specific) + "
            "sterol carrier protein-2 (SCP-2) homology domain; "
            "autosomal recessive; imported via PTS1. "
            "FUNCTION: DBP (D-bifunctional protein) catalyses steps 2 and 3 of "
            "peroxisomal beta-oxidation (the D-specific pathway): "
            "After ACOX1 initiates oxidation (step 1), DBP carries out: "
            "Step 2: Enoyl-CoA hydration (D-specific); "
            "Step 3: L-hydroxyacyl-CoA dehydrogenation. "
            "Substrates: VLCFA, bile acid intermediates, pristanic acid "
            "(all require DBP for beta-oxidation after the first step). "
            "Compare to MFP1 (L-bifunctional protein / EHHADH): "
            "MFP1 catalyses the same steps but with L-specificity; "
            "DBP and MFP1 are NOT redundant: VLCFA and bile acid intermediates "
            "go through DBP pathway predominantly. "
            "WITHOUT DBP: "
            "VLCFA accumulate (C26:0, C26:0/C22:0 elevated); "
            "Pristanic acid accumulates (branched-chain FA); "
            "Bile acid intermediates accumulate (DHCA, THCA); "
            "DHA synthesis impaired. "
            "BIOCHEMICAL DIFFERENTIATOR from ZSD: "
            "DBP deficiency: VLCFA + pristanic acid + bile acid intermediates elevated; "
            "PEROXISOMES STRUCTURALLY INTACT (fibroblast IF: normal peroxisome number and size); "
            "ZSD (PEX1/6): ghost peroxisomes; global import failure; "
            "This distinguishes DBP deficiency from a biogenesis disorder. "
            "CLINICAL PHENOTYPE — MOST SEVERE PEROXISOMAL BETA-OXIDATION DISORDER: "
            "Neonatal onset; "
            "Profound hypotonia at birth; "
            "Neonatal seizures (refractory); "
            "Dysmorphic facial features (resembling Zellweger syndrome): "
            "high forehead, large anterior fontanelle, epicanthal folds, broad nasal bridge; "
            "Cortical neuronal migration defects on MRI (polymicrogyria, pachygyria); "
            "Hepatomegaly + cholestasis; "
            "Adrenal insufficiency (less common than ZSD); "
            "Most die in first year; rare survival beyond 2 years. "
            "GENOTYPE-PHENOTYPE: "
            "Type I DBP deficiency: both hydratase and dehydrogenase activities absent (most severe); "
            "Type II: hydratase only; Type III: dehydrogenase only; "
            "Type IV: mild with cerebellar ataxia + Leber congenital amaurosis phenotype (attenuated). "
            "DIAGNOSIS: "
            "VLCFA: C26:0 elevated; "
            "Pristanic acid: elevated (>5 µmol/L); "
            "Bile acid intermediates: DHCA, THCA in urine; "
            "Fibroblast peroxisome morphology: NORMAL (key differentiator from ZSD); "
            "DBP enzyme activity in fibroblasts: markedly reduced; "
            "HSD17B4 sequencing. "
            "TREATMENT: Supportive only; no disease-modifying therapy available."
        ),
        "age_of_onset": "Neonatal (classic severe); rare attenuated in later childhood/adulthood",
        "inheritance": "Autosomal recessive (biallelic HSD17B4 mutations)",
        "locus": "5q23.1",
        "protein_size": "736 aa",
        "key_biomarker": "VLCFA elevated + pristanic acid elevated + bile acid intermediates; peroxisome structure INTACT on fibroblast IF",
        "pathognomonic": "Zellweger-like neonatal phenotype + VLCFA + pristanic acid elevated + INTACT peroxisomes (not ghost) = DBP deficiency",
        "treatment": "Supportive only (no specific therapy); seizure management; adrenal screening; DHA supplementation may be tried",
        "critical_flags": [
            "PEROXISOMES-STRUCTURALLY-INTACT — key distinguisher from ZSD (NOT ghost peroxisomes on IF)",
            "PRISTANIC-ACID-ELEVATED — differentiates DBP from simple VLCFA-only elevation (ALD/ZSD)",
            "MOST-SEVERE-PEROXISOMAL-BETA-OXIDATION-DISORDER — neonatal lethal in most",
            "BILE-ACID-INTERMEDIATES-DHCA-THCA-IN-URINE — specific for peroxisomal beta-oxidation defects",
            "ZELLWEGER-LIKE-FACE-BUT-DIFFERENT-MECHANISM — brain and face similar to ZSD but biochemistry differs",
            "NO-DISEASE-MODIFYING-THERAPY — supportive only; genetic counselling essential",
            "FIBROBLAST-DBP-ENZYME-ASSAY — confirm before gene sequencing if rapid confirmation needed",
            "ATTENUATED-TYPE-IV-MISDIAGNOSED — cerebellar ataxia + LCA phenotype may be missed; VLCFA screen",
        ],
    },
    # -- ACOX1 -- Acyl-CoA Oxidase 1 Deficiency --------------------------------
    {
        "gene": "ACOX1",
        "protein": (
            "ACOX1 -- 17q25.1 AR -- Acyl-CoA-Oxidase-1-700aa -- "
            "VLCFA-Beta-Oxidation-First-Step-Peroxisomal -- "
            "Pseudo-Neonatal-ALD-Phenotype-VLCFA-Elevated -- "
            "ACOX1-Gain-of-Function-Variants-Cause-Inflammatory-ACOX1-Disease-AID -- "
            "Loss-of-Function-AR-Neurological-Gain-of-Function-AD-Inflammatory -- "
            "No-HSCT-Unlike-ALD-Supportive-Only-Loss-of-Function"
        ),
        "alias": (
            "ACOX1 (acyl-CoA oxidase 1, palmitoyl); OMIM gene 609751. "
            "Peroxisomal acyl-CoA oxidase deficiency (pseudo-neonatal ALD) OMIM 264470. "
            "ACOX1 Inflammatory Disease (AID) — gain-of-function; OMIM 617579. "
            "17q25.1; 700 aa; ~74 kDa; peroxisomal matrix; "
            "FAD-containing flavoenzyme; "
            "functions as homodimer; "
            "AR (loss of function); AD or de novo (gain of function). "
            "FUNCTION: ACOX1 catalyses the FIRST and RATE-LIMITING step of "
            "peroxisomal beta-oxidation of straight-chain VLCFA: "
            "VLCFA-CoA → 2-trans-enoyl-CoA + H2O2. "
            "H2O2 is immediately detoxified by catalase within peroxisomes. "
            "Without ACOX1: VLCFA cannot be beta-oxidised → accumulate. "
            "Downstream enzymes (DBP/HSD17B4, MFP1/EHHADH, SCPx) cannot proceed. "
            "LOSS-OF-FUNCTION (RECESSIVE — PSEUDO-NEONATAL ALD): "
            "Clinical phenotype resembles neonatal ALD (now ZSD/NALD): "
            "Neonatal hypotonia + seizures + progressive leukodystrophy; "
            "VLCFA elevated (isolated — pristanic acid NORMAL; differs from DBP deficiency); "
            "Plasmalogens NORMAL (ether lipid synthesis unaffected); "
            "Peroxisome morphology INTACT (not ghost peroxisomes); "
            "Brain MRI: posterior white matter leukodystrophy (similar to ALD); "
            "Prognosis: severe neurological deterioration; death in childhood typically; "
            "No specific treatment. "
            "GAIN-OF-FUNCTION (DOMINANT/DE NOVO — ACOX1 INFLAMMATORY DISEASE, AID): "
            "COMPLETELY DIFFERENT DISEASE from loss-of-function; "
            "p.Asn237Ser (most common): creates a novel substrate specificity; "
            "mutant ACOX1 generates toxic H2O2 and reactive oxygen species (ROS) directly; "
            "mechanism: gain of oxidase activity on normally non-oxidised substrates; "
            "CLINICAL (inflammatory): "
            "Childhood onset inflammatory encephalopathy; "
            "White matter changes (resembling ALD but inflammatory); "
            "Elevated inflammatory markers; "
            "Elevated VLCFA (from downstream inhibition by toxic products); "
            "TREATMENT DIFFERS: anti-inflammatory agents (steroids, IVIG); "
            "NOT treated like recessive ACOX1 deficiency; "
            "IMPORTANT DDx: "
            "Loss-of-function (recessive): neonatal onset, no inflammation markers, supportive only; "
            "Gain-of-function (dominant/de novo): inflammatory, childhood onset, steroid-responsive. "
            "DIAGNOSIS: "
            "VLCFA elevated (C26:0/C22:0 elevated); "
            "Pristanic acid: NORMAL (unlike DBP deficiency); "
            "Plasmalogens: NORMAL; "
            "ACOX1 enzyme activity in fibroblasts: reduced (LOF) or altered specificity (GOF); "
            "ACOX1 sequencing: essential to distinguish LOF vs GOF variants. "
            "BIOCHEMICAL DDx: "
            "vs ALD (ABCD1): ABCD1 X-linked; adrenal involvement; ALD no adrenal in ACOX1; "
            "vs ZSD (PEX1/6): ACOX1 has intact peroxisomes; normal plasmalogens/pipecolic; "
            "vs DBP (HSD17B4): pristanic acid normal in ACOX1."
        ),
        "age_of_onset": "Neonatal (LOF recessive); childhood (GOF inflammatory)",
        "inheritance": "AR (loss-of-function); AD or de novo (gain-of-function = ACOX1 Inflammatory Disease)",
        "locus": "17q25.1",
        "protein_size": "700 aa",
        "key_biomarker": "VLCFA elevated (C26:0); pristanic acid NORMAL; plasmalogens NORMAL; intact peroxisomes on IF",
        "pathognomonic": "VLCFA + normal pristanic acid + normal plasmalogens + intact peroxisomes = ACOX1 deficiency (LOF) vs ALD/ZSD/DBP",
        "treatment": "LOF: supportive only (no specific therapy); GOF (AID): anti-inflammatory steroids/IVIG (different treatment from LOF)",
        "critical_flags": [
            "GOF-VS-LOF-CRITICAL-DISTINCTION — gain-of-function (inflammatory/steroid-responsive) vs loss-of-function (supportive only); same gene, opposite mechanism",
            "PRISTANIC-ACID-NORMAL — differentiates ACOX1 from DBP/HSD17B4 (where pristanic acid elevated)",
            "PLASMALOGENS-NORMAL — differentiates from ZSD/PEX7 (plasmalogens reduced in those)",
            "PSEUDO-NEONATAL-ALD-NAME-MISLEADING — resembles ALD clinically but X-chromosome not involved",
            "H2O2-TOXICITY-GOF-MECHANISM — mutant ACOX1 generates excess H2O2/ROS; inflammatory encephalopathy",
            "PEROXISOMES-INTACT-ON-IF — not ghost peroxisomes; pure beta-oxidation step 1 defect",
            "ACOX1-SEQUENCING-ESSENTIAL — enzyme assay alone cannot distinguish GOF vs LOF variant",
            "ANTI-INFLAMMATORY-FOR-AID-ONLY — steroids/IVIG for GOF; LOF does not respond to immunosuppression",
        ],
    },
    # -- AGPS -- Rhizomelic Chondrodysplasia Punctata type 3 -------------------
    {
        "gene": "AGPS",
        "protein": (
            "AGPS -- 2q31.2 AR -- Alkylglycerone-Phosphate-Synthase-728aa -- "
            "Rhizomelic-Chondrodysplasia-Punctata-type-3-RCDP3 -- "
            "Ether-Lipid-Synthesis-Second-Step-Plasmalogen-Pathway -- "
            "Stippled-Epiphyses-Cataracts-RCDP-Phenotype-Without-Phytanic-Elevation -- "
            "VLCFA-Normal-Plasmalogens-Reduced-Phytanic-Acid-NORMAL-Unlike-RCDP1 -- "
            "PEX7-Imports-AGPS-RCDP1-AGPS-Itself-Defective-RCDP3"
        ),
        "alias": (
            "AGPS (alkylglycerone phosphate synthase); OMIM gene 603051. "
            "Rhizomelic Chondrodysplasia Punctata type 3 OMIM 600121. "
            "2q31.2; 728 aa; ~82 kDa; peroxisomal matrix; "
            "PTS2-containing enzyme (imported by PEX7); "
            "FAD-containing flavoenzyme; "
            "autosomal recessive. "
            "FUNCTION: AGPS catalyses step 2 of ether lipid (plasmalogen) synthesis: "
            "Acyl-DHAP + long-chain alcohol → alkyl-DHAP + long-chain acid. "
            "The complete plasmalogen synthesis pathway: "
            "Step 1: GNPAT (DHAP-acyltransferase) — acyl-DHAP from DHAP + acyl-CoA; "
            "Step 2: AGPS — alkyl-DHAP from acyl-DHAP + fatty alcohol (this step); "
            "Step 3-onward: ER-based enzymes complete plasmalogen synthesis. "
            "Plasmalogens are ether phospholipids (~18% of all mammalian phospholipids): "
            "critical components of: myelin (brain white matter); "
            "cardiac cell membranes; lung surfactant; "
            "antioxidant protection (act as ROS scavengers). "
            "WITHOUT AGPS: ether lipid synthesis blocked at step 2; "
            "plasmalogens markedly reduced. "
            "RCDP3 vs RCDP1 (PEX7): "
            "RCDP1 (PEX7 defect): AGPS CANNOT ENTER peroxisome → "
            "BOTH phytanic acid elevated AND plasmalogens reduced; "
            "RCDP3 (AGPS itself defective): already in peroxisome but non-functional → "
            "Plasmalogens reduced BUT phytanic acid NORMAL "
            "(PHYH can still enter peroxisome and function normally); "
            "VLCFA: NORMAL in both (PTS1-imported enzymes unaffected). "
            "RCDP3 CLINICAL FEATURES: "
            "IDENTICAL to RCDP1 (PEX7) phenotypically: "
            "Stippled epiphyses (chondrodysplasia punctata) on X-ray; "
            "Rhizomelic limb shortening (proximal); "
            "Congenital cataracts (bilateral); "
            "Intellectual disability (severe); "
            "Seizures; "
            "Growth retardation; "
            "MILDER THAN RCDP1 in some cases (genotype dependent). "
            "RCDP3 is RARER than RCDP1 (accounts for ~5% of RCDP). "
            "DIAGNOSIS: "
            "X-ray: stippled epiphyses (same as RCDP1); "
            "Plasmalogens (RBC): MARKEDLY REDUCED (diagnostic); "
            "VLCFA: NORMAL; "
            "Phytanic acid: NORMAL in RCDP3 (elevated in RCDP1); "
            "This key difference (phytanic acid NORMAL vs elevated) biochemically "
            "distinguishes RCDP3 (AGPS) from RCDP1 (PEX7); "
            "AGPS sequencing confirms. "
            "RCDP DIFFERENTIAL DIAGNOSIS SUMMARY: "
            "RCDP1 (PEX7): plasmalogens low, phytanic acid HIGH, VLCFA normal; "
            "RCDP2 (GNPAT): plasmalogens low, phytanic acid normal, VLCFA normal; "
            "RCDP3 (AGPS): plasmalogens low, phytanic acid NORMAL, VLCFA normal; "
            "Gene panel required to differentiate. "
            "TREATMENT: "
            "Supportive (as RCDP1); "
            "Cataract surgery; "
            "Seizure control; "
            "Plasmalogen oral precursor trials (PEMBA — same trial applies to RCDP1/2/3)."
        ),
        "age_of_onset": "Prenatal / neonatal (stippled epiphyses on imaging)",
        "inheritance": "Autosomal recessive (biallelic AGPS mutations)",
        "locus": "2q31.2",
        "protein_size": "728 aa",
        "key_biomarker": "Plasmalogens (RBC) MARKEDLY REDUCED; phytanic acid NORMAL (unlike RCDP1/PEX7); VLCFA NORMAL",
        "pathognomonic": "Stippled epiphyses + plasmalogens reduced + phytanic acid NORMAL = RCDP3 (AGPS) vs RCDP1 (PEX7 — phytanic elevated)",
        "treatment": "Supportive; cataract surgery; seizure control; plasmalogen precursor trials (PEMBA); no approved specific therapy",
        "critical_flags": [
            "PHYTANIC-ACID-NORMAL-IN-RCDP3 — key differentiator from RCDP1 (PEX7, phytanic elevated); same phenotype, different biochemistry",
            "PLASMALOGENS-MARKEDLY-REDUCED — primary diagnostic biomarker; order RBC plasmalogens",
            "VLCFA-NORMAL-ALL-RCDP — stippled epiphyses + normal VLCFA = RCDP, not ZSD/ALD",
            "RCDP-GENE-PANEL-REQUIRED — RCDP1/2/3 identical clinically; only biochemistry + gene distinguishes",
            "STIPPLED-EPIPHYSES-PATHOGNOMONIC — punctate calcifications on plain X-ray; seen in ALL RCDP types",
            "RARER-THAN-RCDP1 — RCDP3 accounts for ~5% of RCDP; PEX7 most common",
            "ETHER-LIPID-STEP-2 — AGPS = step 2; GNPAT = step 1 (RCDP2); clinically identical",
            "PEMBA-TRIAL-ALL-RCDP — oral plasmalogen precursor trials apply to RCDP1/2/3 equally",
        ],
    },
]


# ---------------------------------------------------------------------------
# Patient cohort generation
# ---------------------------------------------------------------------------

def _make_patients():
    all_pts = []
    for i, gene_data in enumerate(PEROX_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        gene = gene_data["gene"]
        for j in range(40):
            age = rng.randint(0, 45)
            # Age profile per gene
            if gene in ("PEX1", "PEX6"):
                age = rng.choices(
                    [rng.randint(0, 2), rng.randint(3, 18), rng.randint(19, 50)],
                    weights=[0.45, 0.30, 0.25]
                )[0]
            elif gene == "ABCD1":
                age = rng.choices(
                    [rng.randint(3, 12), rng.randint(20, 40), rng.randint(40, 60)],
                    weights=[0.38, 0.42, 0.20]
                )[0]
            elif gene == "PHYH":
                age = rng.randint(15, 55)
            elif gene == "PEX7":
                age = rng.choices(
                    [rng.randint(0, 2), rng.randint(3, 12)],
                    weights=[0.70, 0.30]
                )[0]
            elif gene in ("HSD17B4", "ACOX1"):
                age = rng.choices(
                    [rng.randint(0, 1), rng.randint(2, 10)],
                    weights=[0.75, 0.25]
                )[0]
            elif gene == "AGPS":
                age = rng.choices(
                    [rng.randint(0, 2), rng.randint(3, 10)],
                    weights=[0.70, 0.30]
                )[0]

            sex = rng.choice(["M", "F"]) if gene != "ABCD1" else rng.choices(["M", "F"], weights=[0.85, 0.15])[0]

            # Severity
            if gene in ("HSD17B4", "PEX7", "AGPS"):
                severity = rng.choices(["severe", "moderate"], weights=[0.80, 0.20])[0]
            elif gene == "ABCD1":
                severity = rng.choices(["severe", "moderate", "mild"], weights=[0.38, 0.42, 0.20])[0]
            elif gene == "PHYH":
                severity = rng.choices(["moderate", "mild", "severe"], weights=[0.55, 0.30, 0.15])[0]
            elif gene in ("PEX1", "PEX6"):
                severity = rng.choices(["severe", "moderate", "mild"], weights=[0.40, 0.35, 0.25])[0]
            else:
                severity = rng.choices(["severe", "moderate"], weights=[0.80, 0.20])[0]

            # VLCFA elevated?
            vlcfa_elevated = gene not in ("PHYH", "PEX7", "AGPS") or rng.random() < 0.15

            # Plasmalogens reduced?
            plasmalogens_reduced = gene in ("PEX1", "PEX6", "PEX7", "AGPS") or rng.random() < 0.10

            # Phytanic acid elevated?
            phytanic_elevated = gene in ("PHYH", "PEX7") or (gene in ("PEX1", "PEX6") and rng.random() < 0.60)

            # Retinal involvement
            retinal = gene in ("PEX1", "PEX6", "PHYH", "ABCD1") and rng.random() < 0.75

            # Adrenal involvement
            adrenal = gene == "ABCD1" and rng.random() < 0.80

            # HSCT eligible (ABCD1 only, Loes <9)
            hsct_eligible = gene == "ABCD1" and severity != "severe" and rng.random() < 0.55

            # Cardiac involvement
            cardiac = gene == "PHYH" and rng.random() < 0.35

            # Cataracts
            cataracts = gene in ("PEX7", "AGPS") and rng.random() < 0.90

            all_pts.append({
                "gene": gene,
                "seed": seed,
                "patient_index": j + 1,
                "age": age,
                "sex": sex,
                "severity": severity,
                "vlcfa_elevated": vlcfa_elevated,
                "plasmalogens_reduced": plasmalogens_reduced,
                "phytanic_elevated": phytanic_elevated,
                "retinal": retinal,
                "adrenal": adrenal,
                "hsct_eligible": hsct_eligible,
                "cardiac": cardiac,
                "cataracts": cataracts,
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
    vlcfa_n = sum(1 for p in pts if p["vlcfa_elevated"])
    plasm_n = sum(1 for p in pts if p["plasmalogens_reduced"])
    retinal_n = sum(1 for p in pts if p["retinal"])
    adrenal_n = sum(1 for p in pts if p["adrenal"])
    hsct_n = sum(1 for p in pts if p["hsct_eligible"])
    phytanic_n = sum(1 for p in pts if p["phytanic_elevated"])
    cataracts_n = sum(1 for p in pts if p["cataracts"])
    cardiac_n = sum(1 for p in pts if p["cardiac"])

    return {
        "atlas": "Hereditary-Peroxisomal-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Peroxisomal Disorders Atlas — "
            "PEX1-1283aa-7q21.2-AR-ZSD1-Most-Common-PBD-60-70pct-VLCFA-Elevated-DHA-MANDATORY | "
            "PEX6-980aa-6p21.1-AR-ZSD4-Second-Most-Common-PBD-Attenuated-ZSD-Common | "
            "ABCD1-745aa-Xq28-XLR-ALD-C26:0/C22:0-DIAGNOSTIC-HSCT-Curative-Loes<9-Lorenzo-NOT-Halt-Progression | "
            "PHYH-338aa-10p13-AR-Refsum-Dietary-Phytanic-Restriction-CURATIVE-NEVER-Fast-Plasmapheresis-Crisis | "
            "PEX7-323aa-6q23.3-AR-RCDP1-Stippled-Epiphyses-PATHOGNOMONIC-VLCFA-Normal-Plasmalogens-Reduced | "
            "HSD17B4-736aa-5q23.1-AR-DBP-Most-Severe-Peroxisomal-Beta-Oxidation-Zellweger-Like-Peroxisomes-INTACT | "
            "ACOX1-700aa-17q25.1-AR-AD-LOF-Neonatal-ALD-GOF-Inflammatory-AID-Steroids | "
            "AGPS-728aa-2q31.2-AR-RCDP3-Ether-Lipid-Step2-Stippled-Epiphyses-Phytanic-NORMAL-Unlike-RCDP1 | "
            "320-Patient-Aggregate-8x40-seeds-1854-1861"
        ),
        "total_patients": n,
        "total_genes": len(PEROX_GENES),
        "severe_n": severe_n,
        "severe_pct": round(100 * severe_n / n, 1),
        "vlcfa_elevated_n": vlcfa_n,
        "vlcfa_elevated_pct": round(100 * vlcfa_n / n, 1),
        "plasmalogens_reduced_n": plasm_n,
        "plasmalogens_reduced_pct": round(100 * plasm_n / n, 1),
        "retinal_n": retinal_n,
        "retinal_pct": round(100 * retinal_n / n, 1),
        "adrenal_n": adrenal_n,
        "adrenal_pct": round(100 * adrenal_n / n, 1),
        "hsct_eligible_n": hsct_n,
        "hsct_eligible_pct": round(100 * hsct_n / n, 1),
        "phytanic_elevated_n": phytanic_n,
        "phytanic_elevated_pct": round(100 * phytanic_n / n, 1),
        "cataracts_n": cataracts_n,
        "cataracts_pct": round(100 * cataracts_n / n, 1),
        "cardiac_n": cardiac_n,
        "cardiac_pct": round(100 * cardiac_n / n, 1),
        "seed_range": "1854-1861",
        "genes": [g["gene"] for g in PEROX_GENES],
        "gene_loci": {g["gene"]: g["locus"] for g in PEROX_GENES},
        "gene_sizes": {g["gene"]: g["protein_size"] for g in PEROX_GENES},
        "gene_inheritance": {g["gene"]: g["inheritance"] for g in PEROX_GENES},
    }


def breakdown():
    pts = _PATIENTS
    by_gene = {}
    for g in PEROX_GENES:
        gname = g["gene"]
        gpts = [p for p in pts if p["gene"] == gname]
        n = len(gpts)
        severe_n = sum(1 for p in gpts if p["severity"] == "severe")
        by_gene[gname] = {
            "gene": gname,
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "n_patients": n,
            "severe_n": severe_n,
            "severe_pct": round(100 * severe_n / n, 1),
            "vlcfa_elevated_n": sum(1 for p in gpts if p["vlcfa_elevated"]),
            "plasmalogens_reduced_n": sum(1 for p in gpts if p["plasmalogens_reduced"]),
            "phytanic_elevated_n": sum(1 for p in gpts if p["phytanic_elevated"]),
            "retinal_n": sum(1 for p in gpts if p["retinal"]),
            "adrenal_n": sum(1 for p in gpts if p["adrenal"]),
            "hsct_eligible_n": sum(1 for p in gpts if p["hsct_eligible"]),
            "cataracts_n": sum(1 for p in gpts if p["cataracts"]),
            "cardiac_n": sum(1 for p in gpts if p["cardiac"]),
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "seed": SEED_BASE + PEROX_GENES.index(g),
        }
    return {
        "atlas": "Hereditary-Peroxisomal-Atlas",
        "breakdown_by_gene": by_gene,
        "total_genes": len(PEROX_GENES),
        "seed_range": "1854-1861",
    }


def definitions():
    defs = {}

    defs["Peroxisome Biogenesis — Function and Import Machinery Overview"] = (
        "Peroxisomes are single-membrane organelles present in virtually all eukaryotic cells. "
        "FUNCTIONS: "
        "(1) Peroxisomal beta-oxidation of very long chain fatty acids (VLCFA, C22-C26+); "
        "(2) Alpha-oxidation of branched-chain fatty acids (phytanic acid via PHYH); "
        "(3) Ether lipid (plasmalogen) synthesis — first 2 steps in peroxisome; "
        "(4) Bile acid synthesis (oxidation of bile acid precursors); "
        "(5) Glyoxylate metabolism; "
        "(6) DHA (C22:6, docosahexaenoic acid) synthesis. "
        "IMPORT MACHINERY: "
        "Two targeting signals direct proteins to peroxisomes: "
        "PTS1 (peroxisomal targeting signal 1): C-terminal tripeptide (SKL or variant); "
        "recognized by PEX5 receptor; imports >90% of peroxisomal matrix proteins; "
        "PTS2: N-terminal nonapeptide; recognised by PEX7 receptor; "
        "imports only 3 proteins: AGPS, PHYH, HACL1. "
        "Import cycle: PEX5 (or PEX7) binds cargo in cytoplasm → "
        "docks at peroxisomal membrane (PEX14, PEX13) → "
        "cargo translocated across membrane → "
        "PEX5 retrotranslocated back by PEX1/PEX6/PEX26 AAA-ATPase complex. "
        "BIOGENESIS DISORDERS (PBD): "
        "PEX1, PEX6 (most common): global import failure → ghost peroxisomes; "
        "PEX7: selective PTS2 import failure → plasmalogens + phytanic acid affected only. "
        "SINGLE ENZYME DEFECTS (not biogenesis): "
        "ABCD1 (VLCFA import transporter — membrane); "
        "PHYH (alpha-oxidation enzyme — matrix); "
        "HSD17B4 (beta-oxidation step 2+3 — matrix); "
        "ACOX1 (beta-oxidation step 1 — matrix); "
        "AGPS (ether lipid step 2 — matrix); "
        "peroxisomes structurally INTACT in these disorders."
    )

    defs["VLCFA — Biochemical Interpretation and Normal Ranges"] = (
        "Very long chain fatty acids (VLCFA): saturated fatty acids with chain length ≥22 carbons. "
        "Primary analytes: "
        "C24:0 (lignoceric acid) — normally present in small amounts; "
        "C26:0 (cerotic acid) — primary indicator; normally very low; "
        "C24:0/C22:0 ratio — elevated in peroxisomal disease; "
        "C26:0/C22:0 ratio — most sensitive and specific; "
        "NORMAL VALUES (approximate): "
        "C26:0 plasma: <0.9 µg/mL; "
        "C26:0/C22:0 ratio: <0.023; "
        "C24:0/C22:0 ratio: <1.39. "
        "VLCFA ELEVATED in: "
        "ZSD (PEX1, PEX6, other PEX): global import failure; "
        "X-ALD (ABCD1): specific VLCFA transport failure; "
        "DBP deficiency (HSD17B4): beta-oxidation step 2+3; "
        "ACOX1 deficiency: beta-oxidation step 1; "
        "VLCFA NORMAL in: "
        "RCDP1 (PEX7): only PTS2 import affected (ether lipids and phytanic acid); "
        "RCDP3 (AGPS): ether lipid synthesis step 2; "
        "Refsum disease (PHYH): phytanic acid only; "
        "IMPORTANT: Female carriers of ALD may have NORMAL VLCFA in 15% → "
        "genetic testing is MANDATORY for female carriers. "
        "SPECIMEN: Plasma (fasting preferred); blood spot (NBS for ALD). "
        "CONFOUNDERS: Recent large fat meal can mildly elevate C26:0. "
        "Repeat if borderline; fibroblast testing if plasma equivocal."
    )

    defs["Plasmalogens — Synthesis, Function, and Diagnostic Interpretation"] = (
        "Plasmalogens (plasmenyl phospholipids) are ether-linked phospholipids: "
        "vinyl-ether bond at sn-1 position (instead of ester bond in diacyl phospholipids). "
        "~18% of all mammalian phospholipids are plasmalogens. "
        "FUNCTIONS: "
        "(1) Major myelin component (~70% of white matter phospholipids are plasmalogens); "
        "(2) Cardiac sarcolemma (heart function); "
        "(3) Antioxidant: vinyl-ether bond scavenges ROS (sacrificial antioxidant); "
        "(4) Membrane fluidity and curvature; "
        "(5) Cell signalling (platelet-activating factor precursors). "
        "SYNTHESIS PATHWAY (first 2 steps in peroxisome): "
        "Step 1: GNPAT: DHAP → acyl-DHAP (acyltransferase, PTS2-imported via PEX7); "
        "Step 2: AGPS: acyl-DHAP + fatty alcohol → alkyl-DHAP (PTS2-imported via PEX7); "
        "Step 3+: ER-based enzymes complete synthesis. "
        "MEASUREMENT: "
        "Red blood cell (RBC) plasmalogens: best clinical sample; "
        "Expressed as % of total RBC phospholipids; "
        "NORMAL: >70% of expected (laboratory-specific reference range); "
        "SEVERELY REDUCED (<20% of expected): RCDP1 (PEX7), RCDP2 (GNPAT), RCDP3 (AGPS), ZSD; "
        "MODERATELY REDUCED: some attenuated ZSD. "
        "CLINICAL CORRELATE: "
        "Degree of plasmalogen reduction correlates with severity; "
        "Absent cataracts, stippled epiphyses, and rhizomelia if plasmalogens >30% residual. "
        "NOTES ON DIAGNOSIS: "
        "RCDP1/2/3: all have markedly reduced plasmalogens; "
        "ZSD: also reduced (global peroxisomal failure); "
        "ALD (ABCD1): plasmalogens NORMAL (VLCFA transport only); "
        "Refsum (PHYH): plasmalogens NORMAL."
    )

    defs["Zellweger Spectrum Disorder (ZSD) — Diagnostic and Management Framework"] = (
        "Zellweger Spectrum Disorders (ZSD) represent a continuum caused by biallelic "
        "pathogenic variants in any of 13 PEX genes (most commonly PEX1 and PEX6). "
        "UNIFIED BIOCHEMICAL PROFILE (all ZSD, regardless of PEX gene): "
        "VLCFA elevated (C26:0, C26:0/C22:0 ratio); "
        "Plasmalogens (RBC) reduced; "
        "Pipecolic acid elevated (plasma and urine); "
        "Bile acid intermediates in urine (DHCA, THCA); "
        "DHA (docosahexaenoic acid) reduced. "
        "CLINICAL SPECTRUM (ZS → NALD → IRD → attenuated): "
        "Zellweger Syndrome (ZS): "
        "most severe; neonatal hypotonia; seizures; cortical migration defects; "
        "hepatic failure; characteristic face; chondrodysplasia punctata; "
        "death in first year; "
        "Neonatal ALD (NALD): "
        "intermediate; survive year 1; progressive leukodystrophy; "
        "Infantile Refsum Disease (IRD): "
        "milder; RP + SNHL + intellectual disability; "
        "may also accumulate phytanic acid; survive to adulthood; "
        "Attenuated ZSD: "
        "adult onset; RP + SNHL ± ataxia ± neuropathy; "
        "easily misdiagnosed as Usher, ARCA, CMT; "
        "VLCFA may be only mildly elevated → screen with full ZSD panel. "
        "MANAGEMENT CHECKLIST (all ZSD patients): "
        "(1) DHA supplementation: 70-100 mg/kg/day (neonates); adjust with age; monitor plasma DHA; "
        "(2) Adrenal: ACTH stimulation test; hydrocortisone if adrenal insufficient; stress dosing protocol; "
        "(3) Vitamins A/D/E/K: fat malabsorption risk; supplement all; "
        "(4) Ophthalmology: annual fundoscopy; dark adaptation test; ERG; "
        "(5) Audiology: ABR (auditory brainstem response); SNHL management; cochlear implant if profound; "
        "(6) Hepatology: liver function, coagulation, bile acid levels; ursodeoxycholic acid if cholestatic; "
        "(7) Neurology: seizure management; developmental support; physiotherapy; "
        "(8) Genetics: PEX1/PEX6 panel; family cascade testing. "
        "HSCT NOT ESTABLISHED FOR ZSD: "
        "Unlike ALD, HSCT has not been shown to halt ZSD progression; "
        "experimental gene therapy approaches in research phase."
    )

    defs["X-ALD HSCT Decision Framework — Loes Score, Timing, and Surveillance"] = (
        "Hematopoietic stem cell transplantation (HSCT) is the established curative therapy "
        "for childhood cerebral ALD (CALD) but requires precise patient selection. "
        "LOES MRI SCORING SYSTEM: "
        "Quantifies cerebral demyelination on MRI (scale 0-34): "
        "0 = normal; 34 = maximal disease burden; "
        "Domains scored: T2 signal (posterior, anterior, internal capsule, cerebellum, brainstem); "
        "contrast enhancement (active inflammation bonus points); "
        "global atrophy. "
        "HSCT ELIGIBILITY CRITERIA (ALL must be met): "
        "(1) Loes score <9 (limited disease); "
        "(2) Absent or very early neurological dysfunction; "
        "(3) No severe neurological deterioration in past 6 weeks; "
        "(4) Available donor or autologous gene therapy; "
        "OUTCOME BY TIMING: "
        "Asymptomatic + Loes <4: best outcomes; full neurological preservation; "
        "Early symptomatic + Loes 4-9: good outcomes; some deficits possible; "
        "Loes >9 or significant neurological symptoms: HSCT does NOT prevent deterioration; "
        "risk-benefit unfavourable; focus shifts to supportive care. "
        "SURVEILLANCE PROTOCOL (asymptomatic males age 3-12): "
        "Brain MRI + Loes score: every 6 months; "
        "Neuropsychological testing: annually; "
        "VLCFA: annually (monitor trend); "
        "Adrenal function (ACTH stimulation): annually; "
        "GOAL: detect conversion to CALD before Loes 9 → HSCT window. "
        "LENTI-D (GENE THERAPY): "
        "elivaldogene autotemcel (Skysona, bluebird bio); "
        "FDA approved 2022, EMA approved; "
        "autologous HSC lentivirally transduced with functional ABCD1; "
        "efficacy comparable to allogeneic HSCT; "
        "avoids GVHD; preferred when matched allogeneic donor unavailable. "
        "CARRIER FEMALES: "
        "No HSCT indicated; "
        "VLCFA normal in 15% of carriers (genetic testing mandatory for all at-risk females); "
        "~20% of carrier females develop myelopathy (AMN-like) after age 40; "
        "adrenal insufficiency rare (<1%) in females."
    )

    defs["Refsum Disease — Phytanic Acid Metabolism and Dietary Protocol"] = (
        "Refsum disease is caused by deficiency of PHYH (phytanoyl-CoA 2-hydroxylase), "
        "blocking alpha-oxidation of phytanic acid. "
        "PHYTANIC ACID SOURCES (100% dietary — humans cannot synthesise): "
        "Chlorophyll: the phytol side-chain of chlorophyll is released during gut digestion "
        "and converted to phytanic acid by intestinal bacteria; "
        "found in ALL green vegetables (spinach, broccoli, peas, green beans, kale, etc.); "
        "Ruminant animals: cows, sheep, goats have gut bacteria that convert dietary phytol → phytanic; "
        "found in dairy products (milk, butter, cream, yoghurt, ALL cheeses), beef fat, mutton fat; "
        "Fatty fish: tuna, cod, herring, mackerel, sardines; "
        "ELIMINATION DIET: "
        "Remove ALL chlorophyll-containing foods; "
        "Remove ALL dairy; "
        "Remove ruminant fat (lean beef/lamb MEAT is low fat, so small amounts allowed if trimmed); "
        "Remove fatty ocean fish; "
        "ALLOWED: lean white meat (chicken, pork, turkey); fruit; grains; legumes; "
        "vegetable oils (olive, sunflower — non-ruminant, non-chlorophyll); eggs (minimal phytanic); "
        "TARGET: plasma phytanic acid <200 µmol/L (ideally <100 µmol/L; normal <10 µmol/L). "
        "FASTING PARADOX: "
        "Fasting mobilises adipose phytanic acid stores → acute elevation; "
        "can trigger acute neuropathy, arrhythmia; "
        "RULE: always maintain caloric intake; never NPO without IV glucose; "
        "perioperative management: dextrose infusion throughout. "
        "PLASMAPHERESIS/LDL-APHERESIS: "
        "Rapid removal of phytanic acid (99% albumin-bound): "
        "indications: acute deterioration; plasma phytanic >1000 µmol/L; crisis; "
        "LDL apheresis also removes albumin-bound phytanic acid; "
        "monthly/fortnightly cycles until dietary control achieved. "
        "CARDIAC MONITORING: "
        "Cardiomyopathy: dilated or hypertrophic; "
        "Arrhythmia: heart block, QT prolongation; "
        "SUDDEN CARDIAC DEATH: recognised complication; "
        "Annual ECG + echocardiogram + Holter monitoring mandatory; "
        "ICD implantation if documented arrhythmia."
    )

    defs["Peroxisomal Disorders — Differential Diagnosis by Biomarker Pattern"] = (
        "STEP 1 — VLCFA screen (plasma): "
        "ELEVATED → proceed to step 2 (ZSD/ALD/DBP/ACOX1 spectrum); "
        "NORMAL → check plasmalogens and phytanic acid (RCDP/Refsum spectrum). "
        "STEP 2 — VLCFA ELEVATED subgroup: "
        "Check: peroxisome morphology (fibroblast IF), pipecolic acid, plasmalogens, pristanic acid, sex: "
        "Ghost peroxisomes + pipecolic high + plasmalogens low → ZSD (PEX1/PEX6/other PEX); "
        "INTACT peroxisomes + X-linked + normal pipecolic → ALD (ABCD1); "
        "INTACT peroxisomes + AR + pristanic elevated + bile acid intermediates → DBP/HSD17B4; "
        "INTACT peroxisomes + AR + pristanic NORMAL + plasmalogens normal → ACOX1; "
        "STEP 3 — VLCFA NORMAL subgroup (check plasmalogens and phytanic): "
        "Stippled epiphyses + plasmalogens VERY LOW + phytanic HIGH → RCDP1 (PEX7); "
        "Stippled epiphyses + plasmalogens VERY LOW + phytanic NORMAL → RCDP2 (GNPAT) or RCDP3 (AGPS); "
        "No stippled epiphyses + phytanic HIGH + plasmalogens NORMAL + RP/ataxia/neuropathy → Refsum (PHYH); "
        "SUMMARY DDx TABLE: "
        "Gene  | VLCFA | Plasmalogens | Phytanic | Pristanic | Peroxisome-IF "
        "PEX1  | HIGH  | LOW          | var      | elevated  | Ghost "
        "PEX6  | HIGH  | LOW          | var      | elevated  | Ghost "
        "ABCD1 | HIGH  | NORMAL       | NORMAL   | NORMAL    | INTACT "
        "PHYH  | NORMAL| NORMAL       | HIGH     | NORMAL    | INTACT "
        "PEX7  | NORMAL| LOW          | HIGH     | NORMAL    | INTACT "
        "HSD17B4|HIGH  | NORMAL       | NORMAL   | HIGH      | INTACT "
        "ACOX1 | HIGH  | NORMAL       | NORMAL   | NORMAL    | INTACT "
        "AGPS  | NORMAL| LOW          | NORMAL   | NORMAL    | INTACT"
    )

    defs["Peroxisomal Beta-Oxidation — Step-by-Step Pathway and Defects"] = (
        "Peroxisomal beta-oxidation shortens VLCFA to medium-chain acyl-CoA products "
        "that can then be transferred to mitochondria for complete oxidation. "
        "STRAIGHT-CHAIN VLCFA PATHWAY (analogous to mitochondrial beta-oxidation but different enzymes): "
        "Step 1: Oxidation (FAD-dependent): ACOX1 (ACOX1 deficiency if absent); "
        "VLCFA-CoA → 2-trans-enoyl-CoA + H2O2 (H2O2 detoxified by catalase); "
        "Step 2: Hydration: D-enoyl-CoA hydratase domain of DBP (HSD17B4 deficiency if absent); "
        "2-trans-enoyl-CoA → (3R)-3-hydroxyacyl-CoA; "
        "Step 3: Dehydrogenation: L-hydroxyacyl-CoA dehydrogenase domain of DBP; "
        "(3R)-3-hydroxyacyl-CoA → 3-ketoacyl-CoA + NADH; "
        "Step 4: Thiolysis: SCPx (SCPX-thiolase, sterol carrier protein X); "
        "3-ketoacyl-CoA → acyl-CoA (2 carbons shorter) + acetyl-CoA. "
        "BRANCHED-CHAIN FA (pristanic acid, bile acid precursors): "
        "Requires PRISTANOYL-CoA OXIDASE (ACOX2/3) for step 1 (NOT ACOX1); "
        "Then D-specific hydratase and dehydrogenase of DBP (same HSD17B4); "
        "→ PRISTANIC ACID elevated in DBP deficiency (not in ACOX1 deficiency); "
        "This explains why pristanic acid differentiates ACOX1 from HSD17B4. "
        "CLINICAL IMPLICATIONS: "
        "ACOX1 deficiency: only straight-chain VLCFA affected; pristanic normal; "
        "DBP (HSD17B4) deficiency: BOTH straight-chain AND branched-chain affected; "
        "VLCFA + pristanic + bile acids ALL elevated."
    )

    defs["Peroxisomal Disorders — Newborn Screening and Cascade Testing"] = (
        "NEWBORN SCREENING (NBS): "
        "X-ALD (ABCD1): "
        "C26:0-lysophosphatidylcholine (C26:0-LPC) on dried blood spot; "
        "included in NBS panels in >30 US states and several other countries; "
        "detects affected males and carrier females (some); "
        "positive NBS → confirmatory plasma VLCFA + ABCD1 sequencing; "
        "NBS for ALD is the main driver of pre-symptomatic HSCT opportunity. "
        "ZSD (PEX1/PEX6): "
        "No specific NBS marker established; "
        "C26:0-LPC may be elevated in severe ZSD but not always detected; "
        "Not currently included in routine NBS panels globally; "
        "Severe ZSD diagnosed by clinical features + VLCFA. "
        "REFSUM / RCDP: "
        "Not on routine NBS; clinical recognition required. "
        "CASCADE TESTING (FAMILY SCREENING): "
        "After index case confirmed: "
        "ALD (X-linked): all at-risk maternal relatives (carrier females) → ABCD1 sequencing; "
        "all at-risk boys under 12 → VLCFA + ABCD1 + 6-monthly MRI if confirmed; "
        "ZSD (AR): parents are obligate carriers; siblings 25% risk of ZSD; "
        "RCDP (AR): siblings 25% risk; prenatal diagnosis available; "
        "Prenatal diagnosis: chorionic villus sampling (CVS) for VLCFA, plasmalogens, "
        "gene sequencing on fetal DNA. "
        "GENETIC COUNSELLING: "
        "ALD: explain X-linked carrier status to all at-risk females; "
        "emphasise that VLCFA can be normal in 15% of carriers; "
        "all AR disorders: 25% recurrence risk per pregnancy; "
        "PGT-M (preimplantation genetic testing) available for known family variants."
    )

    return {
        "atlas": "Hereditary-Peroxisomal-Atlas — Clinical Definitions",
        "definitions": defs,
        "total_genes": len(PEROX_GENES),
        "total_definition_entries": len(defs),
        "seed_range": "1854-1861",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2, default=str))
