#!/usr/bin/env python3
"""Hereditary-Pancreatitis-Atlas — Complete 8-Gene Hereditary Pancreatitis and
Exocrine Pancreatic Insufficiency Atlas
PRSS1   (Serine Protease 1 / Cationic Trypsinogen; 247 aa; 7q35; AD;
          Hereditary Pancreatitis (HP) / Hereditary Pancreatitis with Acinar Pain (HPAP);
          R122H and N29I GAIN-OF-FUNCTION hot spots — trypsin autoactivation / fails autolysis;
          40% cumulative pancreatic cancer risk by age 70 — annual MRI surveillance mandatory;
          Endoscopic stenting of PD obstruction before surgical consideration;
          Penetrance ~80% — same family can have widely varying severity) ·
SPINK1  (Serine Protease Inhibitor Kazal Type 1 / PSTI; 79 aa; 5q32; AR/modifier;
          Chronic Pancreatitis risk modifier — N34S most common risk allele (~1% population);
          N34S NOT causal alone — modifier allele amplifies risk 3x; requires 2nd hit;
          Trypsin inhibitor: blocks premature intrapancreatic trypsin activation;
          Tropical pancreatitis: SPINK1 N34S + S1P + environmental triggers;
          Compound modifiers: N34S + PRSS1 GOF or CTRC LOF or alcohol greatly increases CP risk) ·
CTRC    (Chymotrypsin C; 271 aa; 1p36.21; AR;
          Chronic Pancreatitis — loss of protective trypsin-cleaving enzyme;
          CTRC normally cleaves trypsin at Arg122 — inactivating it (protective autolysis);
          LOF → trypsin accumulates uncleaved → progressive acinar cell injury;
          CTRC A73T and G61del most common pathogenic alleles;
          AR inheritance but individual heterozygous variants also increase CP risk) ·
CPA1    (Carboxypeptidase A1; 419 aa; 7q32.2; AD;
          Hereditary Pancreatitis — distinct ER stress / misfolding mechanism, NOT trypsin pathway;
          CPA1 misfolding variants trigger unfolded protein response (UPR) in acinar cells;
          DDIT3/CHOP upregulation is a biomarker of ER stress in CPA1-HP (NOT elevated trypsin);
          Penetrance variable; childhood-onset typical; heterozygous dominant mechanism;
          R256C and E178K most common pathogenic CPA1 variants) ·
CFTR    (Cystic Fibrosis Transmembrane conductance Regulator; 1480 aa; 7q31.2; AR;
          CFTR-related Pancreatitis (CFTR-P) — compound heterozygote: one severe + one mild allele;
          DeltaF508 + R117H (5T allele) classic CFTR-P genotype;
          Bicarbonate secretion impaired → thick inspissated secretions → ductal obstruction;
          Sweat test normal or borderline (NOT classic CF — residual CFTR function >10%);
          CFTR-sufficient vs CFTR-insufficient: residual function determines exocrine phenotype) ·
CLDN2   (Claudin-2; 211 aa; Xq22.3; X-linked;
          Alcoholic Pancreatitis risk — claudin-2 is a tight junction paracellular pore protein;
          CLDN2 risk alleles increase ductal permeability → ethanol + metabolites enter ductal cells;
          MALE ONLY (hemizygous): Xq22.3 risk alleles → 5x increased alcoholic pancreatitis risk;
          FEMALE HETEROZYGOTES: no significant clinical risk (X-inactivation);
          rs12688220 is the key GWAS risk SNP; not a Mendelian disease — population risk modifier) ·
CEL     (Carboxyl Ester Lipase / Bile Salt-Stimulated Lipase; 722 aa; 9q34.13; AD;
          MODY8 / Maturity-Onset Diabetes of the Young type 8 + Exocrine Pancreatic Insufficiency;
          CEL-HYB hybrid allele (CEL-CELP recombinant) — normal sequencing MISSES; VNTR analysis required;
          Dual pancreatic failure hallmark: diabetes PLUS exocrine insufficiency (EPI);
          Fat malabsorption + steatorrhoea + diabetes presenting in 3rd-5th decade;
          Pancreatic atrophy + lipomatosis on imaging; pancreatic enzyme replacement mandatory) ·
CASR    (Calcium-Sensing Receptor; 1078 aa; 3q21.1; AD;
          Hypercalcemia-Induced Pancreatitis — CASR LOF → hypercalcemia → calcium activates trypsinogen;
          Treat underlying hypercalcemia first — pancreatitis resolves with normocalcaemia;
          Calcimimetic Cinacalcet lowers calcium by sensitising CaSR → prevents recurrent pancreatitis;
          MEN1 / primary hyperparathyroidism in family → CASR LOF DDx;
          Hypercalcemia >3.0 mmol/L strongly linked to pancreatitis risk; MEN2A: RET not CASR)
320-patient aggregate cohort (8 × 40, seeds 1406-1413)
"""

import random

SEED_BASE = 1406

HP_GENES = [
    # ── PRSS1 — Hereditary Pancreatitis ──
    {
        "gene": "PRSS1",
        "protein": "Cationic Trypsinogen (Serine Protease 1)",
        "alias": (
            "PRSS1; OMIM gene 276000; Hereditary Pancreatitis #167800; "
            "7q35; 247 aa; ~26 kDa digestive serine protease; "
            "Cationic trypsinogen is the major pancreatic proenzyme — activated to trypsin by enterokinase in duodenum; "
            "R122H (Arg122His) GOF: eliminates the Arg122 trypsin autolysis site → active trypsin NOT degraded → acinar digest; "
            "N29I (Asn29Ile) GOF: increases autoactivation of trypsinogen → premature intrapancreatic trypsin; "
            "Both hot spots MANDATORY to test first — if negative, full gene sequencing warranted; "
            "PANCREATIC CANCER: 40% cumulative risk by age 70 — annual MRI/MRCP from age 40 (or 20 yr after diagnosis); "
            "Penetrance ~80%; pain attacks from childhood/teens; progressive chronic pancreatitis → EPI + diabetes"
        ),
        "aa": "247 aa",
        "kDa": "~26 kDa",
        "locus": "7q35",
        "omim_gene": 276000,
        "omim_disease": 167800,
        "inheritance": "AD — gain-of-function R122H or N29I; penetrance ~80%; de novo rare",
        "gene_class": (
            "PRSS1 encodes cationic trypsinogen, the most abundant pancreatic proteolytic proenzyme. "
            "Under normal conditions, trypsin undergoes rapid autolysis at Arg122 — a safety mechanism "
            "preventing acinar cell digestion. The R122H mutation eliminates this autolysis site, "
            "allowing trypsin to remain active indefinitely once triggered. N29I increases spontaneous "
            "autoactivation of trypsinogen to trypsin. Both variants lead to premature, unregulated "
            "intrapancreatic trypsin activity — the common pathway for acute-on-chronic pancreatitis "
            "attacks, progressive fibrosis, exocrine insufficiency, and markedly elevated pancreatic "
            "cancer risk (40% cumulative by age 70 — surveillance essential)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("R122H (c.365G>A) — eliminates Arg122 autolysis site — most common HP allele", 0.60),
            ("N29I (c.86A>T) — increases trypsinogen autoactivation", 0.30),
            ("Other PRSS1 GOF (A16V, D22G, K23R, R122C)", 0.10),
        ],
        "age_onset_years_range": (5, 30),
        "sex_ratio_M": 0.50,
        "rates": {
            "recurrent_acute_pancreatitis":         0.95,
            "chronic_pancreatitis":                 0.80,
            "exocrine_pancreatic_insufficiency":    0.60,
            "diabetes_mellitus_type3c":             0.45,
            "pancreatic_ductal_stones":             0.70,
            "main_pd_dilatation":                   0.65,
            "pancreatic_cancer_by_70":              0.40,
            "pain_requiring_opioids":               0.55,
            "steatorrhoea":                         0.50,
            "endoscopic_or_surgical_intervention":  0.55,
        },
        "hallmarks": [
            "R122H AND N29I HOT SPOTS: test first — if negative, proceed to full PRSS1 gene sequencing",
            "GAIN-OF-FUNCTION MECHANISM: R122H eliminates Arg122 autolysis site → trypsin not degraded in acinus",
            "CHILDHOOD ONSET: recurrent acute pancreatitis from age 5-15 — genetic cause if unexplained recurrent AP",
            "PANCREATIC CANCER RISK 40%: annual pancreatic MRI from age 40 (or 20 years after first attack, whichever earlier)",
            "PD STONES: calcium carbonate stones in dilated main pancreatic duct — ESWL before ERCP stenting",
            "ENDOSCOPIC STENTING: PD obstruction/stricture — decompression before surgical planning (Frey/Puestow/Beger)",
            "EPI + DIABETES: progressive gland destruction → both exocrine and endocrine failure; PERT + insulin needed",
            "SMOKING DOUBLES RISK: absolute contraindication — accelerates fibrosis and malignant transformation",
        ],
        "treatment_alerts": [
            "SMOKING CESSATION ABSOLUTE: tobacco doubles pancreatitis attack frequency and cancer risk — major modifiable factor",
            "ALCOHOL ABSTINENCE: alcohol triggers attacks even without ALDH variants — strict abstinence mandatory",
            "PANCREATIC ENZYME REPLACEMENT THERAPY (PERT): when EPI confirmed — >90% of patients need dose titration",
            "ANNUAL MRI PANCREAS: from age 40 or 20 years after diagnosis — surveillance for PDAC (40% lifetime risk)",
            "ENDOSCOPIC STENTING: biliary/PD strictures — ERCP + stent; ESWL for stones; surgery if fails endoscopy",
        ],
        "organ_system": "pancreas (exocrine + endocrine) — recurrent pancreatitis → fibrosis → EPI + diabetes + cancer",
        "primary_treatment": "Smoking/alcohol cessation; PERT; pain management; annual MRI surveillance; endoscopic/surgical decompression",
    },
    # ── SPINK1 — Chronic Pancreatitis modifier ──
    {
        "gene": "SPINK1",
        "protein": "Serine Protease Inhibitor Kazal Type 1 (Pancreatic Secretory Trypsin Inhibitor / PSTI)",
        "alias": (
            "SPINK1; OMIM gene 167790; Chronic Pancreatitis modifier; "
            "5q32; 79 aa; ~8 kDa Kazal-type serine protease inhibitor; "
            "PSTI/SPINK1 is the first line of defence against premature trypsin activation in the pancreas; "
            "N34S (Asn34Ser) most common risk allele — ~1% general population; NOT causal alone; "
            "MODIFIER NOT MENDELIAN: amplifies risk 3x in homozygotes; requires 2nd hit (PRSS1 GOF / CTRC LOF / alcohol); "
            "Tropical pancreatitis: N34S + cassava/malnutrition + environment → fibrocalculous pancreatic diabetes; "
            "SPINK1 blocks active trypsin at 1:1 molar ratio — insufficient buffering if trypsin overproduced"
        ),
        "aa": "79 aa",
        "kDa": "~8 kDa",
        "locus": "5q32",
        "omim_gene": 167790,
        "omim_disease": 167790,
        "inheritance": "AR/modifier — N34S homozygous or compound heterozygous; modifier in trans with PRSS1/CTRC/CFTR",
        "gene_class": (
            "SPINK1 (PSTI) is a 79-amino-acid Kazal-type serine protease inhibitor secreted by pancreatic "
            "acinar cells alongside zymogens. It acts as a first-line safety buffer, inhibiting up to ~20% "
            "of prematurely activated trypsin within acinar cells before it can cause autodigestion. "
            "The N34S variant (frequency ~1%) reduces SPINK1 secretion and/or function, lowering this "
            "protective buffer. Critically, N34S is a MODIFIER, not a causal Mendelian variant — "
            "it amplifies risk 3-fold in homozygotes but pancreatitis does not consistently follow "
            "without additional genetic (PRSS1, CTRC, CFTR) or environmental (alcohol, tobacco, tropical diet) triggers. "
            "N34S compound heterozygosity with splice variants (IVS3+2T>C) creates stronger LOF."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("N34S homozygous — moderate trypsin inhibitor reduction; requires 2nd hit for pancreatitis", 0.45),
            ("N34S + IVS3+2T>C compound heterozygous — stronger LOF, more likely symptomatic", 0.30),
            ("N34S + co-occurring CTRC LOF or PRSS1 GOF — digenic high-risk combination", 0.25),
        ],
        "age_onset_years_range": (10, 45),
        "sex_ratio_M": 0.55,
        "rates": {
            "chronic_pancreatitis":                    0.80,
            "recurrent_acute_pancreatitis":            0.65,
            "tropical_pancreatitis_fibrocalculous":    0.30,
            "exocrine_pancreatic_insufficiency":       0.50,
            "diabetes_mellitus_type3c":                0.40,
            "pancreatic_calcifications":               0.55,
            "alcohol_as_2nd_hit":                      0.35,
            "pain_chronic_requiring_opioids":          0.45,
            "steatorrhoea":                            0.40,
            "co_occurring_genetic_variant":            0.55,
        },
        "hallmarks": [
            "N34S NOT CAUSAL ALONE: risk modifier — 3x CP risk in homozygotes; requires 2nd hit (PRSS1/CTRC/alcohol)",
            "TROPICAL PANCREATITIS: SPINK1 N34S most prevalent cause of fibrocalculous pancreatic diabetes in tropics",
            "MODIFIER + ENVIRONMENT: alcohol, tobacco, and malnutrition convert N34S from risk to reality",
            "DIGENIC RISK: N34S + CTRC LOF or PRSS1 GOF creates high-penetrance hereditary CP phenotype",
            "CHRONIC > ACUTE PREDOMINANCE: recurrent AP less prominent than in PRSS1; chronic fibrotic progression typical",
            "PANCREATIC CALCIFICATIONS: calcium carbonate stones in ducts — fibrocalculous pattern on CT",
            "EPI + DIABETES: fibrotic gland destruction → both failures; same management as PRSS1-HP EPI",
            "FAMILY HISTORY: N34S common — test all first-degree relatives; compound heterozygosity matters most",
        ],
        "treatment_alerts": [
            "N34S TESTING: NOT curative target but identifies patients needing aggressive modifiable risk factor control",
            "ALCOHOL + TOBACCO: strict abstinence doubles benefit in N34S carriers — primary prevention",
            "PERT: pancreatic enzyme replacement when EPI confirmed — start at 40,000-80,000 lipase units per main meal",
            "PAIN MANAGEMENT: WHO ladder; gabapentin; coeliac plexus block; surgical decompression if PD dilated >6 mm",
            "GENETIC COUNSELLING: explain modifier (not dominant) inheritance; cascade N34S testing in family",
        ],
        "organ_system": "pancreas — chronic pancreatitis (fibrosis + calcifications) → EPI + type 3c diabetes",
        "primary_treatment": "Modifiable risk factor control (alcohol/tobacco); PERT; pain management; genetic counselling",
    },
    # ── CTRC — Chronic Pancreatitis ──
    {
        "gene": "CTRC",
        "protein": "Chymotrypsin C",
        "alias": (
            "CTRC; OMIM gene 601405; Chronic Pancreatitis (CTRC-CP) #167800; "
            "1p36.21; 271 aa; ~30 kDa chymotrypsin isoform; "
            "Chymotrypsin C (CTRC) is the key enzyme that cleaves trypsin at Arg122, inactivating it; "
            "CTRC is the physiological 'off switch' for intraductal trypsin — LOF → trypsin accumulates; "
            "A73T and G61del: most common pathogenic CTRC alleles causing CP; "
            "AR inheritance but heterozygous carriers also show increased CP risk; "
            "CTRC also cleaves SPINK1 — CTRC LOF removes both trypsin degradation AND SPINK1 amplification"
        ),
        "aa": "271 aa",
        "kDa": "~30 kDa",
        "locus": "1p36.21",
        "omim_gene": 601405,
        "omim_disease": 167800,
        "inheritance": "AR — biallelic LOF; heterozygous A73T or G61del also increase CP risk",
        "gene_class": (
            "CTRC (Chymotrypsin C) is a digestive protease that plays a unique protective role in the pancreas: "
            "it cleaves trypsin at the Arg122-Val123 bond, inactivating trypsin and preventing auto-amplification "
            "of the trypsin activation cascade. CTRC also degrades SPINK1, but its net effect is strongly "
            "protective against uncontrolled trypsin activity. LOF variants (A73T, G61del) reduce CTRC "
            "secretion or catalytic activity, allowing trypsin to accumulate when activated — leading to "
            "progressive acinar cell injury, duct obstruction, and chronic pancreatitis. "
            "Unlike PRSS1 (gain-of-function), CTRC is loss-of-function — the shared downstream consequence "
            "(excess active trypsin) is identical, but the mechanism and inheritance differ."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("A73T (c.217G>A) homozygous — reduced secretion of CTRC into pancreatic juice", 0.45),
            ("G61del compound heterozygous with A73T — combined LOF", 0.35),
            ("Other CTRC truncating/splice variant — null allele", 0.20),
        ],
        "age_onset_years_range": (15, 50),
        "sex_ratio_M": 0.55,
        "rates": {
            "chronic_pancreatitis":                 0.88,
            "recurrent_acute_pancreatitis":         0.60,
            "exocrine_pancreatic_insufficiency":    0.55,
            "diabetes_mellitus_type3c":             0.40,
            "pancreatic_ductal_dilation":           0.60,
            "pancreatic_calcifications":            0.50,
            "pain_requiring_intervention":          0.50,
            "steatorrhoea":                         0.45,
            "alcohol_exacerbation":                 0.40,
            "endoscopic_or_surgical_treatment":     0.45,
        },
        "hallmarks": [
            "CTRC CLEAVES TRYPSIN AT Arg122: physiological off-switch for intrapancreatic trypsin activity",
            "LOF FAILS TO CLEAVE TRYPSIN: trypsin accumulates → same end-pathway as PRSS1 R122H (blocks same site)",
            "A73T AND G61del: most common pathogenic alleles; test as first-tier in unexplained CP",
            "HETEROZYGOUS RISK: single A73T or G61del allele increases CP risk — not purely AR",
            "DIGENIC WITH SPINK1: CTRC LOF + SPINK1 N34S = high-risk combination; severe CP phenotype",
            "PROGRESSIVE FIBROSIS: similar trajectory to PRSS1-HP — EPI + type 3c diabetes over decades",
            "DUCTAL CALCIFICATIONS: calcium carbonate stones in dilated PD — imaging hallmark",
            "CANCER RISK: lower than PRSS1 but elevated — pancreatic surveillance MRI recommended",
        ],
        "treatment_alerts": [
            "SAME MANAGEMENT AS PRSS1-HP: smoking/alcohol cessation; PERT; pain management; endoscopic/surgical",
            "PERT TITRATION: 80,000 lipase units/main meal starting dose; adjust by faecal elastase + clinical response",
            "NO CTRC REPLACEMENT: enzyme replacement not available; treatment is supportive/downstream",
            "GENETIC PANEL: CTRC must be on every unexplained CP gene panel; A73T commonly missed if panel incomplete",
            "SURVEILLANCE MRI: annual pancreatic MRI from age 50 in CTRC-CP — cancer risk elevated above population",
        ],
        "organ_system": "pancreas (exocrine gland) — chronic pancreatitis → fibrosis → EPI + type 3c diabetes",
        "primary_treatment": "Smoking/alcohol cessation; PERT; pain management; endoscopic/surgical decompression; MRI surveillance",
    },
    # ── CPA1 — Hereditary Pancreatitis (ER stress) ──
    {
        "gene": "CPA1",
        "protein": "Carboxypeptidase A1",
        "alias": (
            "CPA1; OMIM gene 114850; Hereditary Pancreatitis (ER stress) #167800; "
            "7q32.2; 419 aa; ~47 kDa zinc-containing exopeptidase; "
            "CPA1 is a pancreatic acinar cell proenzyme — digests C-terminal aromatic/aliphatic amino acids; "
            "CPA1 misfolding variants trigger ER stress (unfolded protein response / UPR) in acinar cells; "
            "Mechanism DISTINCT from PRSS1/SPINK1/CTRC: NOT elevated trypsin — ER stress pathway; "
            "DDIT3/CHOP upregulation: biomarker of CPA1-HP ER stress (not raised on standard labs); "
            "AD heterozygous: R256C and E178K most common; moderate penetrance; "
            "Childhood-onset; can be misdiagnosed as idiopathic chronic pancreatitis"
        ),
        "aa": "419 aa",
        "kDa": "~47 kDa",
        "locus": "7q32.2",
        "omim_gene": 114850,
        "omim_disease": 167800,
        "inheritance": "AD — heterozygous misfolding variant; moderate penetrance; de novo described",
        "gene_class": (
            "CPA1 encodes Carboxypeptidase A1, a zinc exopeptidase that cleaves C-terminal aromatic "
            "and aliphatic amino acids from dietary proteins. CPA1 is synthesised as a proenzyme in "
            "pancreatic acinar cells and secreted into the pancreatic duct. Pathogenic CPA1 variants "
            "(R256C, E178K and others) cause protein misfolding and ER retention — triggering the "
            "Unfolded Protein Response (UPR) with DDIT3/CHOP upregulation and acinar cell apoptosis. "
            "This ER stress mechanism is fundamentally distinct from the trypsin-centric pathway "
            "of PRSS1/SPINK1/CTRC variants. Importantly, serum trypsin levels may be normal in "
            "CPA1-HP, and standard trypsin-focused panels may miss this diagnosis if CPA1 is excluded."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("R256C (c.766C>T) — most common CPA1 HP allele; severe misfolding", 0.50),
            ("E178K (c.532G>A) — significant misfolding + ER retention", 0.30),
            ("Other CPA1 missense (Y250C, G203R) — moderate misfolding", 0.20),
        ],
        "age_onset_years_range": (3, 25),
        "sex_ratio_M": 0.50,
        "rates": {
            "recurrent_acute_pancreatitis":           0.85,
            "chronic_pancreatitis":                   0.65,
            "er_stress_ddit3_chop_marker":            0.90,
            "normal_serum_trypsin":                   0.70,
            "exocrine_pancreatic_insufficiency":      0.45,
            "diabetes_mellitus_type3c":               0.30,
            "childhood_onset_under_15":               0.70,
            "pain_attacks":                           0.80,
            "misdiagnosed_idiopathic_cp":             0.40,
            "family_history_pancreatitis":            0.60,
        },
        "hallmarks": [
            "ER STRESS MECHANISM: NOT trypsin pathway — CPA1 misfolding → UPR → DDIT3/CHOP → acinar apoptosis",
            "DDIT3/CHOP BIOMARKER: upregulation of ER stress markers (NOT elevated serum trypsin)",
            "NORMAL SERUM TRYPSIN: distinguishes CPA1-HP from PRSS1-HP; trypsin-centric tests may be normal",
            "CHILDHOOD ONSET: recurrent AP from age 3-15 in most; misdiagnosed as idiopathic if CPA1 not tested",
            "AD INHERITANCE: moderate penetrance; same family can have silent carriers and severely affected members",
            "CPA1 MUST BE ON PANEL: not included in all HP gene panels — request specifically in unexplained childhood CP",
            "PROGRESSIVE FIBROSIS: leads to EPI and type 3c diabetes despite different molecular mechanism",
            "STANDARD LABS MISS: PRSS1 hot spots negative + elevated lipase/amylase only during attacks — gene panel essential",
        ],
        "treatment_alerts": [
            "GENE PANEL MUST INCLUDE CPA1: standard PRSS1/SPINK1/CTRC panel insufficient for childhood recurrent AP",
            "NO ER STRESS TARGETED THERAPY: supportive management; trials of chemical chaperones (TUDCA) ongoing",
            "SMOKING/ALCOHOL ABSTINENCE: same modifiable risk factors apply despite different mechanism",
            "PERT WHEN EPI CONFIRMED: faecal elastase <200 mcg/g → start PERT; titrate to symptom/weight response",
            "GENETIC COUNSELLING: AD — 50% risk per child; variable penetrance complicates cascade counselling",
        ],
        "organ_system": "pancreas (acinar cells — ER stress pathway) → recurrent pancreatitis → fibrosis → EPI + diabetes",
        "primary_treatment": "Gene panel including CPA1; smoking/alcohol cessation; PERT; pain management; genetic counselling",
    },
    # ── CFTR — CFTR-related Pancreatitis ──
    {
        "gene": "CFTR",
        "protein": "Cystic Fibrosis Transmembrane Conductance Regulator",
        "alias": (
            "CFTR; OMIM gene 602421; CFTR-related Pancreatitis (CFTR-P) / Pancreatitis #167800; "
            "7q31.2; 1480 aa; ~170 kDa ABC transporter; "
            "CFTR is an ATP-gated chloride/bicarbonate channel in pancreatic ductal cells; "
            "CFTR-P: compound heterozygote — one severe (class I-III) allele + one mild (class IV-VI/5T); "
            "DeltaF508 + R117H-5T: classic CFTR-P genotype (NOT classic CF — residual function >10%); "
            "Bicarbonate secretion impaired → viscous inspissated secretions → ductal obstruction → pancreatitis; "
            "Sweat chloride NORMAL or BORDERLINE (30-60 mmol/L) in CFTR-P (unlike classic CF >60 mmol/L); "
            "CFTR-sufficient (EPI-S) vs CFTR-insufficient (EPI-I): residual CFTR function determines exocrine phenotype"
        ),
        "aa": "1480 aa",
        "kDa": "~170 kDa",
        "locus": "7q31.2",
        "omim_gene": 602421,
        "omim_disease": 167800,
        "inheritance": "AR — compound heterozygote: severe allele + mild allele (5T/R117H); biallelic severe = classic CF",
        "gene_class": (
            "CFTR (Cystic Fibrosis Transmembrane Conductance Regulator) is an ABC transporter chloride/bicarbonate "
            "channel expressed on the apical membrane of pancreatic ductal epithelial cells. Its role in the "
            "pancreas is to secrete bicarbonate-rich fluid that flushes zymogens into the duodenum. "
            "CFTR-related pancreatitis (CFTR-P) occurs in compound heterozygotes who carry one severe allele "
            "(DeltaF508, G542X, W1282X — class I-III, <10% CFTR function) and one mild allele "
            "(R117H with 5T, G85E, D1152H — class IV-V, >10% CFTR function). "
            "This intermediate CFTR function causes ductal secretory failure — viscous secretions, "
            "protein plugging, and ductal hypertension — without the pulmonary disease of classic CF. "
            "Sweat chloride is normal or intermediate, distinguishing CFTR-P from classic CF."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("DeltaF508 + R117H-5T — classic CFTR-P compound heterozygote", 0.50),
            ("DeltaF508 + D1152H or G85E — severe + mild allele combination", 0.30),
            ("Other severe + 5T allele combinations — borderline CFTR function", 0.20),
        ],
        "age_onset_years_range": (15, 50),
        "sex_ratio_M": 0.55,
        "rates": {
            "recurrent_acute_pancreatitis":           0.80,
            "chronic_pancreatitis":                   0.60,
            "normal_or_borderline_sweat_chloride":    0.85,
            "pancreatic_sufficient_at_onset":         0.70,
            "exocrine_pancreatic_insufficiency":      0.45,
            "diabetes_mellitus_type3c":               0.30,
            "no_pulmonary_disease":                   0.80,
            "sinusitis_or_rhinitis":                  0.40,
            "obstructive_azoospermia_males":          0.30,
            "ductal_protein_plugging":                0.65,
        },
        "hallmarks": [
            "COMPOUND HETEROZYGOTE: severe allele + mild allele (class IV-V) — residual CFTR function 10-50%",
            "SWEAT TEST NORMAL OR BORDERLINE: NOT classic CF — CFTR-P has 30-60 mmol/L (classic CF >60 mmol/L)",
            "BICARBONATE SECRETION IMPAIRED: thick inspissated ductal secretions → protein plugging → obstruction",
            "PANCREATIC-SUFFICIENT AT ONSET: most CFTR-P patients retain exocrine function initially — EPI develops later",
            "5T ALLELE: poly-T tract in intron 8 — reduces CFTR exon 9 splicing → low-level CFTR; combine with R117H",
            "NO PULMONARY DISEASE: distinguishes CFTR-P from classic CF — residual CFTR function protects lungs",
            "AZOOSPERMIA IN MALES: congenital bilateral absence of vas deferens (CBAVD) — fertility counselling",
            "SWEAT TEST + FULL CFTR SEQUENCING: 23/39 common variants miss some CFTR-P alleles — comprehensive panel",
        ],
        "treatment_alerts": [
            "CFTR MODULATOR THERAPY: ivacaftor/lumacaftor/elexacaftor-tezacaftor — eligibility depends on CFTR genotype",
            "PERT: when EPI confirmed (faecal elastase <200 mcg/g); titrate dose; fat-soluble vitamin supplementation",
            "AVOID TRIGGERS: alcohol, smoking, high-fat meals — reduce attack frequency",
            "SINUSITIS MANAGEMENT: nasal irrigation + steroid sprays; ENT referral for polyps",
            "FERTILITY COUNSELLING: CBAVD in males; IVF/ICSI with epididymal sperm extraction if paternity desired",
        ],
        "organ_system": "pancreas (ductal — bicarbonate secretion) + sinuses + vas deferens; NOT lungs in CFTR-P",
        "primary_treatment": "CFTR modulators (genotype-dependent); PERT if EPI; avoid alcohol/tobacco; fertility counselling",
    },
    # ── CLDN2 — Alcoholic Pancreatitis risk ──
    {
        "gene": "CLDN2",
        "protein": "Claudin-2",
        "alias": (
            "CLDN2; OMIM gene 300520; Alcoholic Pancreatitis risk modifier (X-linked); "
            "Xq22.3; 211 aa; ~22 kDa tight junction protein; "
            "Claudin-2 forms paracellular pores (cation channels) in tight junctions of pancreatic ductal epithelium; "
            "CLDN2 risk alleles increase claudin-2 expression → increased paracellular permeability → ethanol + metabolite entry; "
            "MALE ONLY (hemizygous): Xq22.3 — hemizygous males have 5x increased alcoholic pancreatitis risk; "
            "FEMALE HETEROZYGOTES: no significant clinical risk — X-inactivation mosaicism protects; "
            "rs12688220: key GWAS risk SNP at CLDN2-MORC4 locus (X chromosome pancreatitis association); "
            "NOT a Mendelian disease — population-level risk modifier integrated with alcohol exposure"
        ),
        "aa": "211 aa",
        "kDa": "~22 kDa",
        "locus": "Xq22.3",
        "omim_gene": 300520,
        "omim_disease": 300520,
        "inheritance": "X-linked — hemizygous males at risk; heterozygous females protected by X-inactivation",
        "gene_class": (
            "CLDN2 encodes Claudin-2, a tetraspanning tight junction protein that forms paracellular "
            "cation-selective pores (allowing Na+, K+, and water flux) in epithelial monolayers. "
            "In the pancreatic ductal epithelium, elevated Claudin-2 expression increases paracellular "
            "permeability, allowing ethanol and its toxic metabolites (acetaldehyde, fatty acid ethyl esters) "
            "to penetrate the ductal epithelial barrier. This disrupts ductal homeostasis and exacerbates "
            "alcohol-induced acinar injury. CLDN2 is at Xq22.3 — the GWAS-identified pancreatitis susceptibility "
            "locus on the X chromosome. Males are hemizygous (one allele), so risk alleles have full penetrance "
            "for this alcohol-dependent susceptibility. Female heterozygotes express wild-type CLDN2 from "
            "the inactive X chromosome through mosaicism, diluting the tight junction permeability effect."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("rs12688220 risk allele (hemizygous male) — increased CLDN2 expression + alcohol exposure", 0.70),
            ("CLDN2 overexpression variant + heavy alcohol use — synergistic ductal permeability", 0.20),
            ("CLDN2 risk allele + SPINK1 N34S — polygenic high-risk alcoholic pancreatitis", 0.10),
        ],
        "age_onset_years_range": (25, 55),
        "sex_ratio_M": 0.90,
        "rates": {
            "male_sex":                              0.90,
            "heavy_alcohol_use_trigger":             0.95,
            "alcoholic_acute_pancreatitis":          0.85,
            "chronic_alcoholic_pancreatitis":        0.65,
            "exocrine_pancreatic_insufficiency":     0.45,
            "diabetes_mellitus_type3c":              0.35,
            "pancreatic_calcifications":             0.50,
            "pain_chronic":                          0.55,
            "family_history_alcoholic_cp":           0.40,
            "alcohol_cessation_reduces_attacks":     0.70,
        },
        "hallmarks": [
            "MALE ONLY RISK: Xq22.3 hemizygous males — CLDN2 risk allele → 5x alcoholic pancreatitis risk",
            "FEMALE HETEROZYGOTES: NO significant clinical risk — X-inactivation mosaicism protects",
            "PARACELLULAR PERMEABILITY: claudin-2 pores allow ethanol/acetaldehyde into ductal epithelium",
            "ALCOHOL IS REQUIRED: CLDN2 risk allele alone insufficient — alcohol is the obligate environmental trigger",
            "GWAS LOCUS: rs12688220 at CLDN2-MORC4 locus on X chromosome — highest X-linked pancreatitis association",
            "MODIFIER NOT MENDELIAN: CLDN2 is a risk variant, not a diagnostic single-gene cause",
            "ALCOHOL CESSATION REDUCES ATTACKS 70%: primary intervention; CLDN2 risk allele without alcohol = low risk",
            "POLYGENIC RISK: CLDN2 + SPINK1 N34S + heavy alcohol = very high CP risk (multiplicative)",
        ],
        "treatment_alerts": [
            "ALCOHOL CESSATION MANDATORY: primary intervention — CLDN2 risk allele without alcohol confers minimal risk",
            "ALCOHOL DEPENDENCE TREATMENT: addiction medicine referral; naltrexone; acamprosate; counselling",
            "PERT: when EPI confirmed — alcoholic CP patients often present late with advanced exocrine failure",
            "THIAMINE SUPPLEMENTATION: Wernicke's prevention in heavy drinkers — B1 before glucose in acute setting",
            "GENETIC COUNSELLING: X-linked; brothers of affected males at risk; sons of carrier females at 50% risk",
        ],
        "organ_system": "pancreas (ductal epithelial barrier) — alcohol-dependent paracellular injury → alcoholic CP",
        "primary_treatment": "Alcohol cessation (primary); addiction management; PERT; pain management; nutritional support",
    },
    # ── CEL — MODY8 / Exocrine Pancreatic Insufficiency ──
    {
        "gene": "CEL",
        "protein": "Carboxyl Ester Lipase (Bile Salt-Stimulated Lipase / BSSL)",
        "alias": (
            "CEL; OMIM gene 114840; MODY8 / Maturity-Onset Diabetes of the Young type 8 #609812; "
            "9q34.13; 722 aa; ~78 kDa pancreatic lipase; "
            "CEL encodes carboxyl ester lipase — the main pancreatic enzyme for dietary fat/cholesterol ester digestion; "
            "CEL-HYB (CEL-CELP hybrid allele): recombination between CEL and its pseudogene CELP creates pathogenic hybrid; "
            "NORMAL SEQUENCING MISSES CEL-HYB: VNTR analysis required — standard exome/NGS fails to detect; "
            "MODY8 HALLMARK: Diabetes PLUS Exocrine Pancreatic Insufficiency — dual pancreatic failure; "
            "Pancreatic atrophy + lipomatosis on imaging: distinguishes from type 2 diabetes; "
            "Steatorrhoea + fat malabsorption from EPI precedes or accompanies diabetes"
        ),
        "aa": "722 aa",
        "kDa": "~78 kDa",
        "locus": "9q34.13",
        "omim_gene": 114840,
        "omim_disease": 609812,
        "inheritance": "AD — CEL-HYB hybrid allele (CEL-CELP recombinant); full-gene deletion also described",
        "gene_class": (
            "CEL (Carboxyl Ester Lipase / Bile Salt-Stimulated Lipase) is secreted by pancreatic acinar cells "
            "and hydrolyses cholesterol esters, triglycerides, lysophospholipids, and fat-soluble vitamins "
            "in the duodenum in a bile salt-dependent manner. The CEL gene contains a VNTR (variable number "
            "tandem repeat) region in exon 11 that is highly similar to the adjacent pseudogene CELP. "
            "Pathogenic CEL-HYB alleles arise from unequal crossover between CEL and CELP, introducing "
            "CELP sequences into the CEL VNTR. The chimeric protein misfolds, accumulates in acinar cells, "
            "and triggers both direct acinar cell loss (exocrine insufficiency) and insulin-secreting cell "
            "damage (MODY8 diabetes). Standard Sanger sequencing and exome WES cannot detect this VNTR "
            "rearrangement — VNTR-specific PCR or long-read sequencing is required."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("CEL-HYB hybrid allele (CEL-CELP VNTR recombinant) — standard sequencing MISSES; VNTR assay required", 0.80),
            ("CEL full-gene deletion — MLPA detectable", 0.10),
            ("CEL truncating variant disrupting VNTR region", 0.10),
        ],
        "age_onset_years_range": (30, 55),
        "sex_ratio_M": 0.50,
        "rates": {
            "exocrine_pancreatic_insufficiency":            0.95,
            "diabetes_mellitus_mody8":                      0.85,
            "steatorrhoea_fat_malabsorption":               0.90,
            "pancreatic_atrophy_imaging":                   0.80,
            "pancreatic_lipomatosis":                       0.70,
            "weight_loss":                                  0.65,
            "fat_soluble_vitamin_deficiency":               0.55,
            "misdiagnosed_type2_diabetes":                  0.50,
            "vntr_analysis_required_for_dx":                0.80,
            "family_history_mody_or_epi":                   0.60,
        },
        "hallmarks": [
            "CEL-HYB MISSED BY STANDARD SEQUENCING: VNTR analysis required — exome/NGS does not detect hybrid allele",
            "DUAL PANCREATIC FAILURE HALLMARK: Diabetes (MODY8) PLUS Exocrine Insufficiency — unique CEL phenotype",
            "PANCREATIC ATROPHY + LIPOMATOSIS: imaging key — distinguish from T2DM (normal pancreas) and CP (calcifications)",
            "STEATORRHOEA BEFORE DIABETES: exocrine failure often precedes islet destruction by years",
            "MISDIAGNOSED AS TYPE 2 DM: 50% misdiagnosed — MODY8 in slim patient with EPI features; check CEL",
            "FAT-SOLUBLE VITAMIN DEFICIENCY: A, D, E, K all at risk from EPI malabsorption — monitor levels annually",
            "PANCREATIC ENZYME REPLACEMENT: MODY8 EPI requires PERT — linaclotide + pancreatin; high dose needed",
            "FAMILY HISTORY: AD inheritance; EPI + MODY in same family strongly suggests CEL-HYB",
        ],
        "treatment_alerts": [
            "VNTR ANALYSIS: request specifically — NGS exome misses CEL-HYB; send to specialist lab",
            "PERT MANDATORY: all MODY8 patients with EPI require pancreatic enzyme replacement; high-dose (80,000 IU/meal)",
            "DIABETES MANAGEMENT: insulin typically required for MODY8 (beta cell destruction not insulin resistance)",
            "FAT-SOLUBLE VITAMINS: A, D, E, K supplementation; monitor serum levels every 6 months",
            "ANNUAL IMAGING: CT/MRI abdomen — pancreatic atrophy progression; exclude malignant transformation",
        ],
        "organ_system": "pancreas (acinar cells + islets of Langerhans) — EPI + MODY8 diabetes; dual failure",
        "primary_treatment": "PERT (high dose); insulin for MODY8 diabetes; fat-soluble vitamin supplementation; VNTR diagnosis",
    },
    # ── CASR — Hypercalcemia-Induced Pancreatitis ──
    {
        "gene": "CASR",
        "protein": "Calcium-Sensing Receptor",
        "alias": (
            "CASR; OMIM gene 601199; Hypercalcemia-Induced Pancreatitis / Familial Hypocalciuric Hypercalcemia #145980; "
            "3q21.1; 1078 aa; ~120 kDa G-protein coupled receptor; "
            "CaSR senses extracellular calcium and suppresses PTH release — key calcium homeostasis regulator; "
            "CASR LOF → hypercalcemia → elevated ionised Ca2+ activates trypsinogen in pancreatic juice; "
            "TREAT UNDERLYING HYPERCALCEMIA FIRST: pancreatitis resolves when normocalcaemia achieved; "
            "CALCIMIMETIC CINACALCET: lowers calcium by sensitising CaSR → prevents recurrent pancreatitis episodes; "
            "DDx: primary hyperparathyroidism (PHPT), MEN1 (hyperparathyroid), vitamin D toxicity; "
            "FHH (Familial Hypocalciuric Hypercalcemia) vs PHPT: urine Ca/Cr ratio <0.01 → FHH (no surgery needed)"
        ),
        "aa": "1078 aa",
        "kDa": "~120 kDa",
        "locus": "3q21.1",
        "omim_gene": 601199,
        "omim_disease": 145980,
        "inheritance": "AD — heterozygous LOF (FHH/hypercalcemia); biallelic severe LOF → neonatal severe hyperparathyroidism",
        "gene_class": (
            "CASR (Calcium-Sensing Receptor) is a G-protein coupled receptor (GPCR) expressed on "
            "parathyroid chief cells and renal tubular cells that detects extracellular calcium concentration. "
            "Normal CaSR → high calcium → suppresses PTH → reduces calcium reabsorption. "
            "CASR LOF → CaSR is 'deaf' to high calcium → PTH continues secreting → hypercalcemia. "
            "Hypercalcemia-induced pancreatitis arises because elevated ionised calcium directly activates "
            "trypsinogen in pancreatic secretions, initiating the trypsin cascade and acinar autodigestion. "
            "This is not a primary pancreatic defect — it is secondary to hypercalcemia. "
            "Cinacalcet (calcimimetic) allosterically activates mutant CaSR or sensitises residual CaSR activity, "
            "lowering calcium and preventing recurrent pancreatitis episodes."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Heterozygous CASR LOF — Familial Hypocalciuric Hypercalcemia (FHH1) → recurrent hypercalcaemic pancreatitis", 0.55),
            ("CASR LOF in context of MEN1 — hyperparathyroidism driving hypercalcemia", 0.25),
            ("CASR LOF — neonatal severe hyperparathyroidism (biallelic) — severe hypercalcemia at birth", 0.20),
        ],
        "age_onset_years_range": (20, 60),
        "sex_ratio_M": 0.50,
        "rates": {
            "hypercalcemia_ionised_ca_3mmol":          0.90,
            "recurrent_acute_pancreatitis":            0.80,
            "pancreatitis_resolves_with_normocalcemia":0.85,
            "low_urine_calcium_creatinine":            0.75,
            "elevated_pth":                            0.65,
            "kidney_stones":                           0.35,
            "neuropsychiatric_symptoms":               0.40,
            "family_history_hypercalcemia":            0.65,
            "cinacalcet_response":                     0.70,
            "chronic_pancreatitis_if_untreated":       0.40,
        },
        "hallmarks": [
            "TREAT HYPERCALCEMIA FIRST: pancreatitis is secondary — treat the cause; pancreatitis resolves with normocalcaemia",
            "CALCIUM ACTIVATES TRYPSINOGEN: ionised Ca2+ >3 mmol/L directly triggers intrapancreatic trypsin cascade",
            "FHH vs PHPT: urine Ca/Cr <0.01 → FHH (CASR LOF) — parathyroidectomy NOT curative in FHH; cinacalcet preferred",
            "CINACALCET PREVENTS RECURRENCE: allosteric CaSR activator → lowers PTH → lowers Ca2+ → no trypsin activation",
            "CASR PANEL: CASR full sequencing in any unexplained recurrent pancreatitis + hypercalcaemia",
            "MEN1 SCREEN: if parathyroid adenoma + recurrent pancreatitis → CASR + MEN1 gene panel (menin)",
            "NEUROPSYCHIATRIC SYMPTOMS: depression, cognitive decline, fatigue — symptoms of hypercalcaemia",
            "BIALLELIC CASR LOF: neonatal severe hyperparathyroidism — surgical emergency; extreme hypercalcaemia",
        ],
        "treatment_alerts": [
            "CINACALCET: calcimimetic — lowers PTH + calcium; first-line in FHH (avoids unnecessary parathyroidectomy)",
            "PARATHYROIDECTOMY: only for PHPT (not FHH) — subtotal parathyroidectomy; monitor post-op calcium closely",
            "IV HYDRATION: acute hypercalcaemia attack — normal saline 2-4 L/day + loop diuretic (furosemide) if needed",
            "BISPHOSPHONATES: pamidronate/zoledronic acid for acute hypercalcaemia if CaSR treatment not yet started",
            "AVOID THIAZIDES: thiazide diuretics increase calcium reabsorption — contraindicated in all hypercalcaemia",
        ],
        "organ_system": "parathyroid → hypercalcemia → pancreas (trypsinogen activation); kidneys; neuropsychiatric",
        "primary_treatment": "Cinacalcet (FHH/CASR LOF); treat hypercalcemia first; parathyroidectomy only for PHPT; hydration for acute attacks",
    },
]


def _make_patients(gene_data: dict) -> list:
    rng = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    rates = gene_data["rates"]
    ptx = []

    for pid in range(1, gene_data["n_patients"] + 1):
        age_lo, age_hi = gene_data["age_onset_years_range"]
        sex = "M" if rng.random() < gene_data["sex_ratio_M"] else "F"
        onset_age = rng.randint(age_lo, age_hi)

        etio_choices = [e[0] for e in gene_data["etiologies"]]
        etio_probs = [e[1] for e in gene_data["etiologies"]]
        etiology = rng.choices(etio_choices, weights=etio_probs, k=1)[0]

        clinical = {k: (rng.random() < v) for k, v in rates.items()}

        ptx.append({
            "id": f"{gene}-{pid:03d}",
            "gene": gene,
            "sex": sex,
            "onset_age_years": onset_age,
            "etiology": etiology,
            **clinical,
        })
    return ptx


def _aggregate(patients: list, rate_keys: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    return {k: round(sum(1 for p in patients if p.get(k, False)) / n * 100, 1)
            for k in rate_keys}


ALL_PATIENTS = []
for _gd in HP_GENES:
    ALL_PATIENTS.extend(_make_patients(_gd))


def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    rate_keys = list(HP_GENES[0]["rates"].keys())
    agg = _aggregate(ALL_PATIENTS, rate_keys)
    return {
        "atlas": "Hereditary Pancreatitis Atlas",
        "subtitle": "Complete 8-Gene Hereditary Pancreatitis and Exocrine Pancreatic Insufficiency Reference",
        "total_patients": n,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["gene"] for g in HP_GENES],
        "diseases": {
            "PRSS1":  "Hereditary Pancreatitis (HP) — Cationic Trypsinogen GOF; R122H/N29I; 40% pancreatic cancer by 70; annual MRI",
            "SPINK1": "Chronic Pancreatitis modifier — PSTI/SPINK1; N34S risk allele 3x; NOT causal alone; requires 2nd hit",
            "CTRC":   "Chronic Pancreatitis (CTRC-CP) — Chymotrypsin C LOF; fails to cleave trypsin Arg122; A73T/G61del",
            "CPA1":   "Hereditary Pancreatitis (ER stress) — Carboxypeptidase A1 misfolding; DDIT3/CHOP; NOT trypsin pathway",
            "CFTR":   "CFTR-related Pancreatitis — compound heterozygote; DeltaF508+R117H-5T; borderline sweat test; bicarbonate",
            "CLDN2":  "Alcoholic Pancreatitis risk — Claudin-2 Xq22.3; hemizygous males 5x risk; alcohol required 2nd hit",
            "CEL":    "MODY8 + Exocrine Pancreatic Insufficiency — CEL-HYB hybrid allele; standard sequencing MISSES; VNTR required",
            "CASR":   "Hypercalcemia-Induced Pancreatitis — CaSR LOF; treat hypercalcemia first; cinacalcet prevents recurrence",
        },
        "aggregate_stats": agg,
        "top_alerts": [
            "PRSS1 R122H/N29I: test HOT SPOTS first — if negative, full gene sequencing; 40% pancreatic cancer risk by age 70",
            "PRSS1: annual pancreatic MRI from age 40 (or 20 yr after first attack) — mandatory PDAC surveillance",
            "SPINK1 N34S: NOT causal alone — modifier 3x; requires 2nd hit (PRSS1 GOF / CTRC LOF / alcohol / tropical diet)",
            "CPA1: ER stress mechanism — NOT trypsin pathway; DDIT3/CHOP marker; trypsin levels may be NORMAL",
            "CEL-HYB: standard sequencing MISSES — VNTR analysis required; MODY8 = diabetes + EPI dual failure hallmark",
            "CFTR-P: sweat test NORMAL or BORDERLINE (30-60 mmol/L) — NOT classic CF; compound heterozygote",
            "CLDN2: MALE ONLY (hemizygous Xq22.3) — 5x alcoholic pancreatitis risk; females: NO risk (X-inactivation)",
            "CASR: treat hypercalcemia FIRST — pancreatitis resolves with normocalcaemia; cinacalcet prevents recurrence",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for gd in HP_GENES:
        pts = [p for p in ALL_PATIENTS if p["gene"] == gd["gene"]]
        rate_keys = list(gd["rates"].keys())
        agg = _aggregate(pts, rate_keys)
        result[gd["gene"]] = {
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "organ_system": gd["organ_system"],
            "primary_treatment": gd["primary_treatment"],
            "stats": agg,
            "hallmarks": gd["hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "etiology_distribution": [
                {"etiology": e[0], "fraction": e[1]} for e in gd["etiologies"]
            ],
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas": "Hereditary Pancreatitis Atlas",
        "classification": {
            "Trypsin_Pathway_GOF": {
                "HP_PRSS1":    "Hereditary Pancreatitis — cationic trypsinogen R122H/N29I GOF; trypsin autolysis site eliminated; 40% PDAC risk",
            },
            "Trypsin_Regulation_LOF": {
                "CP_SPINK1":   "Chronic Pancreatitis modifier — PSTI/SPINK1 N34S risk allele; trypsin inhibitor reduced; NOT causal alone",
                "CP_CTRC":     "Chronic Pancreatitis — chymotrypsin C LOF; fails to cleave trypsin Arg122; A73T/G61del pathogenic alleles",
            },
            "ER_Stress_Pathway": {
                "HP_CPA1":     "Hereditary Pancreatitis (ER stress) — carboxypeptidase A1 misfolding; UPR/DDIT3/CHOP; NOT trypsin pathway",
            },
            "Ductal_Secretion": {
                "CFTR_P":      "CFTR-related Pancreatitis — compound heterozygote; bicarbonate secretion impaired; viscous secretions; borderline sweat test",
            },
            "Barrier_and_Risk_Modifiers": {
                "AP_CLDN2":    "Alcoholic Pancreatitis risk — claudin-2 Xq22.3; hemizygous males only; paracellular permeability; alcohol required",
            },
            "Exocrine_Pancreatic_Insufficiency": {
                "MODY8_CEL":   "MODY8 + EPI — CEL-HYB hybrid allele; VNTR analysis required; dual failure: diabetes + exocrine insufficiency",
            },
            "Hypercalcemia_Induced": {
                "HP_CASR":     "Hypercalcemia-induced Pancreatitis — CaSR LOF; ionised Ca2+ activates trypsinogen; treat hypercalcemia first; cinacalcet",
            },
        },
        "key_diagnostic_rules": {
            "PRSS1_HOTSPOT_FIRST": (
                "PRSS1 R122H and N29I are the two gain-of-function hot spots accounting for >80% of hereditary "
                "pancreatitis cases. These should be tested first (targeted assay or gene panel). "
                "If both hot spots are negative in a family with strong hereditary pancreatitis features "
                "(multiple affected relatives, childhood onset, unexplained recurrent AP), "
                "proceed to full PRSS1 gene sequencing to detect rarer pathogenic variants (A16V, D22G, K23R, R122C). "
                "Annual pancreatic MRI surveillance from age 40 (or 20 years after first attack) is mandatory "
                "in all PRSS1 pathogenic variant carriers — cumulative PDAC risk reaches 40% by age 70."
            ),
            "SPINK1_N34S_MODIFIER": (
                "SPINK1 N34S (frequency ~1% in general population) is NOT a Mendelian cause of pancreatitis. "
                "It is a risk modifier that reduces PSTI trypsin-inhibitory capacity, lowering the threshold "
                "for pancreatitis when a second hit is present (PRSS1 GOF, CTRC LOF, alcohol, or tropical diet). "
                "Homozygous N34S amplifies risk 3-fold. Clinical significance requires context: "
                "isolated N34S in a healthy person without family history is unlikely to cause pancreatitis alone. "
                "Report N34S as a 'risk modifier' — NOT as a causative pathogenic variant."
            ),
            "CPA1_ER_STRESS_NOT_TRYPSIN": (
                "CPA1-related hereditary pancreatitis has a fundamentally distinct mechanism from PRSS1/SPINK1/CTRC: "
                "it is an ER stress / misfolded protein disease, not a trypsin dysregulation disease. "
                "Serum trypsin levels may be normal (or only mildly elevated during attacks), misleading trypsin-centric tests. "
                "DDIT3 (CHOP) upregulation is a biomarker of ER stress in acinar cells. "
                "CPA1 MUST be included on hereditary pancreatitis gene panels — standard PRSS1/SPINK1/CTRC triple-gene "
                "panels miss all CPA1-HP cases. In childhood unexplained recurrent pancreatitis with negative trypsin "
                "pathway genes, test CPA1."
            ),
            "CEL_HYB_VNTR_REQUIRED": (
                "The pathogenic CEL-HYB allele arises from recombination between CEL and its pseudogene CELP "
                "in the VNTR region of exon 11. Standard Sanger sequencing and exome next-generation sequencing "
                "CANNOT detect this rearrangement — the VNTR region is excluded or collapsed in most NGS pipelines. "
                "VNTR-specific PCR or Southern blot analysis is required to diagnose CEL-HYB. "
                "The hallmark of MODY8 is the unique dual pancreatic failure: exocrine pancreatic insufficiency "
                "(steatorrhoea, fat malabsorption) PLUS MODY diabetes — both from the same gene defect."
            ),
            "CFTR_SWEAT_TEST_NOT_CLASSIC_CF": (
                "CFTR-related pancreatitis (CFTR-P) is distinct from classic cystic fibrosis. "
                "CFTR-P patients carry one severe allele + one mild allele, giving residual CFTR function >10%. "
                "This residual function protects the lungs from classic CF but is insufficient for normal "
                "pancreatic bicarbonate secretion. Sweat chloride in CFTR-P is typically normal (< 30 mmol/L) "
                "or intermediate (30-60 mmol/L) — NOT elevated as in classic CF (>60 mmol/L). "
                "A normal sweat test does NOT exclude CFTR-P. Comprehensive CFTR sequencing + 5T allele analysis required."
            ),
            "CLDN2_MALE_ONLY": (
                "CLDN2 is located at Xq22.3. Risk alleles (rs12688220) for alcoholic pancreatitis are expressed "
                "in hemizygous males (single X chromosome copy). Males with CLDN2 risk alleles have 5x increased "
                "risk of alcoholic pancreatitis compared to males without, ONLY in the presence of heavy alcohol use. "
                "Heterozygous females have mosaic X-inactivation — approximately 50% of ductal cells express "
                "the wild-type X chromosome, diluting the claudin-2 permeability effect. "
                "Female heterozygotes show no significant clinical risk elevation. "
                "Report: CLDN2 risk allele is relevant ONLY in male patients with alcoholic pancreatitis."
            ),
            "CASR_TREAT_HYPERCALCEMIA_FIRST": (
                "In CASR LOF (Familial Hypocalciuric Hypercalcemia or related conditions), pancreatitis is secondary "
                "to hypercalcemia — elevated ionised calcium directly activates trypsinogen in pancreatic secretions. "
                "The correct treatment sequence is: (1) achieve normocalcaemia first — pancreatitis attacks cease. "
                "(2) For FHH (CASR LOF, urine Ca/Cr < 0.01): cinacalcet is first-line — parathyroidectomy is NOT "
                "curative in FHH and risks post-op hypoparathyroidism. (3) For primary PHPT (adenoma): "
                "parathyroidectomy is curative. Never proceed to parathyroid surgery in FHH without testing urine Ca/Cr ratio "
                "— this distinguishes FHH (conservative) from PHPT (surgical)."
            ),
            "CASCADE_TESTING": (
                "PRSS1 (AD, ~80% penetrance): 50% risk per child — cascade test all first-degree relatives; "
                "children should be tested by age 12 and informed of cancer surveillance requirements. "
                "SPINK1 N34S: inform relatives it is a common risk modifier; testing optional unless symptomatic. "
                "CPA1 (AD): 50% risk; cascade test in families with childhood recurrent AP. "
                "CFTR: AR compound heterozygote; parents are obligate carriers; children: 25% risk of same genotype. "
                "CLDN2 (X-linked): maternal carrier daughters at 50% risk of carrier status; sons at 50% risk of hemizygosity. "
                "CEL-HYB (AD): 50% risk per child; specifically request VNTR analysis in at-risk relatives. "
                "CASR (AD): 50% risk; test all first-degree relatives with unexplained hypercalcaemia."
            ),
        },
        "treatment_hierarchy": {
            "PRSS1_HP": [
                "1. Smoking cessation absolute (doubles attack frequency and cancer risk) + strict alcohol abstinence",
                "2. Pain management: WHO analgesic ladder; gabapentin; coeliac plexus block if chronic pain",
                "3. PERT: when EPI confirmed (faecal elastase <200 mcg/g); titrate to fat excretion + weight",
                "4. ERCP + stenting / ESWL: PD stones + ductal obstruction before surgical referral",
                "5. Annual pancreatic MRI from age 40 (or 20 yr post-diagnosis) — PDAC surveillance mandatory",
            ],
            "CEL_MODY8": [
                "1. VNTR analysis (not standard sequencing) to confirm CEL-HYB diagnosis before treatment planning",
                "2. PERT high-dose (80,000+ IU lipase per main meal): steatorrhoea control; fat-soluble vitamin supplementation",
                "3. Insulin therapy: MODY8 diabetes typically requires insulin (beta cell destruction); not sulphonylurea",
                "4. Annual CT/MRI abdomen: pancreatic atrophy progression; malignant transformation surveillance",
                "5. Fat-soluble vitamins A, D, E, K: supplement + monitor serum levels every 6 months",
            ],
            "CFTR_P": [
                "1. Comprehensive CFTR genotyping + 5T allele analysis to confirm CFTR-P diagnosis",
                "2. CFTR modulator therapy (ivacaftor/elexacaftor-tezacaftor-ivacaftor): genotype-eligibility check",
                "3. Avoid alcohol, tobacco, high-fat meals — modifiable triggers",
                "4. PERT when EPI develops; sinusitis management; fertility counselling (CBAVD males)",
                "5. Annual review: CFTR-P → watch for evolving pulmonary features if CFTR function declines",
            ],
            "CASR_HP": [
                "1. Urine calcium/creatinine ratio: if <0.01 → FHH (CASR LOF) → cinacalcet, NOT parathyroidectomy",
                "2. Cinacalcet: allosteric CaSR activator — lowers PTH + Ca2+; prevents recurrent pancreatitis",
                "3. Acute hypercalcaemia: IV normal saline 2-4 L/day + loop diuretic; bisphosphonate if needed",
                "4. MEN1 screen: if parathyroid adenoma → CASR + MEN1 gene panel",
                "5. AVOID THIAZIDES: increase calcium reabsorption — contraindicated in all hypercalcaemia syndromes",
            ],
        },
        "genes_summary": {g["gene"]: g["alias"] for g in HP_GENES},
    }


# ── Convenience wrappers (called by api_backend.py) ──
def overview():
    return get_overview()


def breakdown():
    return get_breakdown()


def definitions():
    return get_definitions()
